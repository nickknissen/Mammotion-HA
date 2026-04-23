"""Mammotion map image entities.

Two entities per mower, designed to be stacked via a Lovelace
picture-elements card:

* :class:`MammotionMapImage` — the static base: mowing areas, obstacles,
  planned mow path, dock and RTK markers.  Regenerates only when the
  map geometry itself changes, so it stays stable while the mower is
  actively mowing.
* :class:`MammotionMowerPositionImage` — a transparent, same-sized PNG
  containing only the live robot marker, mowed swathe, and dynamics
  polyline.  Refreshes on the reporting coordinator (throttled).

The split keeps the base image rock-solid during mow sessions — earlier
behaviour bundled both into a single entity whose composite redraw hit
the event loop on every reporting tick.
"""

from __future__ import annotations

import datetime
import logging

from homeassistant.components.image import ImageEntity
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util

from . import MammotionConfigEntry
from .coordinator import (
    MammotionMapUpdateCoordinator,
    MammotionReportUpdateCoordinator,
)
from .entity import MammotionBaseEntity
from .map_renderer import EMPTY_OVERLAY_PNG, PLACEHOLDER_PNG, render_overlay_png

_LOGGER = logging.getLogger(__name__)

PARALLEL_UPDATES = 0

# Drop reporting-coord redraws that arrive faster than this.  The mower
# streams position at a few Hz during mowing; a 1 Hz ceiling is more than
# fast enough for a dashboard image and caps recorder/history churn.
_MIN_OVERLAY_INTERVAL = datetime.timedelta(seconds=1)

# Upper bound on the accumulated-position ring buffer.  At 1 Hz this caps
# the "mowed-so-far" overlay at ~2.8 hours of continuous travel, which is
# longer than any single Luba session.  Older points fall off the start.
_POSITION_HISTORY_MAX = 10_000

# Minimum distance (m) between successive recorded positions.  Below this
# we treat new fixes as jitter and skip.  Keeps the history buffer sparse
# without losing meaningful swathe coverage.
_POSITION_HISTORY_MIN_STEP_M = 0.15


async def async_setup_entry(
    hass: HomeAssistant,
    entry: MammotionConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the static map and the mower-position overlay per mower."""
    entities: list[ImageEntity] = []
    for mower in entry.runtime_data.mowers:
        entities.append(MammotionMapImage(mower.map_coordinator, hass))
        entities.append(
            MammotionMowerPositionImage(
                mower.map_coordinator, mower.reporting_coordinator, hass
            )
        )
    async_add_entities(entities)


class MammotionMapImage(MammotionBaseEntity, ImageEntity):
    """Static base map — geometry, dock, RTK.  No live mower layers."""

    _attr_translation_key = "map"
    _attr_content_type = "image/png"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(
        self,
        map_coordinator: MammotionMapUpdateCoordinator,
        hass: HomeAssistant,
    ) -> None:
        """Wire the map coordinator — no reporting feed needed."""
        MammotionBaseEntity.__init__(self, map_coordinator, "map")
        ImageEntity.__init__(self, hass)
        self._map_coordinator = map_coordinator

    async def async_added_to_hass(self) -> None:
        """Seed ``image_last_updated`` from the coordinator's last render."""
        await super().async_added_to_hass()
        if self._map_coordinator.map_last_render is not None:
            self._attr_image_last_updated = self._map_coordinator.map_last_render

    @callback
    def _handle_coordinator_update(self) -> None:
        """Bump timestamp whenever the coordinator produced a new base PNG."""
        if self._map_coordinator.map_last_render is not None:
            self._attr_image_last_updated = self._map_coordinator.map_last_render
        super()._handle_coordinator_update()

    async def async_image(self) -> bytes | None:
        """Return the coordinator-cached base PNG, or placeholder pre-render."""
        return self._map_coordinator.map_base_png_bytes or PLACEHOLDER_PNG


class MammotionMowerPositionImage(MammotionBaseEntity, ImageEntity):
    """Transparent overlay with the live robot marker + mowed track.

    Sized and projected to match :class:`MammotionMapImage`, so stacking the
    two in a picture-elements card produces the combined view without any
    per-tick composite work on the event loop.
    """

    _attr_translation_key = "mower_position"
    _attr_content_type = "image/png"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(
        self,
        map_coordinator: MammotionMapUpdateCoordinator,
        reporting_coordinator: MammotionReportUpdateCoordinator,
        hass: HomeAssistant,
    ) -> None:
        """Wire both coordinators — map for bbox, reporting for position."""
        MammotionBaseEntity.__init__(self, map_coordinator, "mower_position")
        ImageEntity.__init__(self, hass)
        # Narrowly-typed handles so Pyright sees coordinator-specific attrs
        # without fighting the CoordinatorEntity[] generic on the parent.
        self._map_coordinator = map_coordinator
        self._reporting_coordinator = reporting_coordinator
        self._cached_output: bytes | None = None
        self._last_overlay_at: datetime.datetime | None = None
        # Rolling buffer of (x_east, y_north) points visited by the mower,
        # sampled from ``real_pos`` on every reporting tick.  Drawn as a
        # wide green swathe to approximate the "mowed so far" area the
        # Mammotion app shows.  Reset on work-zone changes.
        self._position_history: list[tuple[float, float]] = []
        self._last_zone_hash: int | None = None

    async def async_added_to_hass(self) -> None:
        """Subscribe to the reporting coordinator and seed the first frame."""
        await super().async_added_to_hass()
        self.async_on_remove(
            self._reporting_coordinator.async_add_listener(
                self._handle_reporting_update
            )
        )
        self._rebuild()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Map coordinator ticked — bbox may have shifted, force a redraw."""
        self._rebuild(force=True)
        super()._handle_coordinator_update()

    @callback
    def _handle_reporting_update(self) -> None:
        """Redraw overlay when the reporting coordinator ticks (throttled)."""
        now = dt_util.utcnow()
        if (
            self._last_overlay_at is not None
            and now - self._last_overlay_at < _MIN_OVERLAY_INTERVAL
        ):
            return
        self._rebuild()
        self.async_write_ha_state()

    # -- internal helpers ----------------------------------------------------

    @callback
    def _rebuild(self, *, force: bool = False) -> None:
        """Re-render the transparent overlay PNG."""
        bbox = self._map_coordinator.map_base_bbox
        if bbox is None:
            # No base to align against — serve an empty transparent PNG so
            # the entity stays usable when stacked over the placeholder.
            output = EMPTY_OVERLAY_PNG
        else:
            x, y, heading = self._current_position()
            self._update_position_history(x, y)
            dynamics = self._current_dynamics_line()
            try:
                output = render_overlay_png(
                    bbox,
                    x,
                    y,
                    heading,
                    dynamics_line=dynamics,
                    mowed_track=self._position_history,
                )
            except Exception as err:  # noqa: BLE001 - never brick the entity
                _LOGGER.warning(
                    "Mower-position overlay failed for %s: %s",
                    self.coordinator.device_name,
                    err,
                )
                return

        self._last_overlay_at = dt_util.utcnow()
        if force or output != self._cached_output:
            self._cached_output = output
            self._attr_image_last_updated = dt_util.utcnow()

    @callback
    def _update_position_history(
        self, x: float | None, y: float | None
    ) -> None:
        """Append *(x, y)* to the rolling mowed-track buffer.

        No-ops when the position isn't available or the step is under
        :data:`_POSITION_HISTORY_MIN_STEP_M` (keeps the buffer sparse).
        Clears the buffer when the active work zone changes so the
        overlay resets on a new session.
        """
        if x is None or y is None:
            return

        # Detect session/zone change and reset the track.  ``work_zone``
        # on the Location struct follows the active zone hash; a change
        # means we're mowing a different area now.
        mower = self._mower()
        if mower is not None:
            try:
                current_zone = int(mower.location.work_zone or 0)
            except (AttributeError, TypeError, ValueError):
                current_zone = 0
            if self._last_zone_hash is None:
                self._last_zone_hash = current_zone
            elif current_zone != self._last_zone_hash:
                self._position_history = []
                self._last_zone_hash = current_zone

        if self._position_history:
            lx, ly = self._position_history[-1]
            if (
                (lx - x) * (lx - x) + (ly - y) * (ly - y)
                < _POSITION_HISTORY_MIN_STEP_M * _POSITION_HISTORY_MIN_STEP_M
            ):
                return

        self._position_history.append((x, y))
        if len(self._position_history) > _POSITION_HISTORY_MAX:
            # Drop oldest points in bulk to avoid slicing every tick.
            del self._position_history[: len(self._position_history) - _POSITION_HISTORY_MAX]

    def _mower(self):
        """Return the MowerDevice wrapper for this entity, or None."""
        try:
            return self.coordinator.manager.get_device_by_name(
                self.coordinator.device_name
            )
        except Exception:  # noqa: BLE001 - device may be removed mid-render
            return None

    def _current_position(
        self,
    ) -> tuple[float | None, float | None, float | None]:
        """Extract (x_enu_m, y_enu_m, heading_deg) for the robot.

        Preference order:

        1. ``report_data.locations[0].real_pos_x / real_pos_y`` (scaled int,
           ÷1e4).  Axis swap per ``pymammotion/data/model/device.py:175``:
           ``real_pos_y`` → east (x), ``real_pos_x`` → north (y).  Heading
           comes from ``real_toward / 1e4``.  This field is populated once
           the mower has a GNSS fix and matches the polygon ENU frame —
           empirically reliable across all device states (idle, mowing,
           returning).
        2. ``report_data.work.path_pos_x / path_pos_y`` (scaled int, ÷1e4).
           PyMammotion's docstring claims ENU-from-RTK but on at least some
           Luba devices this lands in a different frame (possibly path- or
           breakpoint-relative).  Used only as a fallback when ``real_pos``
           is unavailable.

        Returns ``(None, None, None)`` when no usable position is available.
        """
        mower = self._mower()
        if mower is None:
            return None, None, None

        # Default heading: orientation on the Location struct (degrees,
        # typically 0 until the mower has had its first fix).
        heading: float | None = (
            float(mower.location.orientation)
            if mower.location.orientation
            else None
        )

        # 1. GNSS fix from report_data.locations — reliable ENU frame.
        try:
            locs = mower.report_data.locations
            if locs:
                loc = locs[0]
                if loc.real_pos_x != 0 or loc.real_pos_y != 0:
                    # Axis mapping verified empirically against
                    # ``location.device`` lat/lng deltas relative to RTK:
                    #   real_pos_x → east (x)
                    #   real_pos_y → north (y)
                    # Note this is the opposite of what device.py:175
                    # suggests — PyMammotion's enu_to_lla has east/north
                    # swapped internally vs its parameter names.
                    x_east = loc.real_pos_x / 10000.0
                    y_north = loc.real_pos_y / 10000.0
                    if loc.real_toward:
                        heading = loc.real_toward / 10000.0
                    return x_east, y_north, heading
        except (AttributeError, IndexError):
            pass

        # 2. Fallback: path_pos (unreliable on some devices — see docstring).
        try:
            work = mower.report_data.work
            if work.path_pos_x != 0 or work.path_pos_y != 0:
                return (
                    work.path_pos_x / 10000.0,
                    work.path_pos_y / 10000.0,
                    heading,
                )
        except AttributeError:
            pass

        return None, None, heading

    def _current_dynamics_line(self):
        """Snapshot ``HashList.dynamics_line`` to detach from live mutation."""
        mower = self._mower()
        if mower is None:
            return None
        try:
            # shallow copy is enough — CommDataCouple is immutable-by-convention
            return list(mower.map.dynamics_line)
        except AttributeError:
            return None

    # -- ImageEntity API ----------------------------------------------------

    async def async_image(self) -> bytes | None:
        """Return the cached overlay bytes — empty transparent pre-render."""
        return self._cached_output if self._cached_output is not None else EMPTY_OVERLAY_PNG
