"""Mammotion map image entity.

Serves the rendered HashList as a slow-changing PNG, composited with the
live robot marker and the mow-progress polyline on each report update.

Structure mirrors Home Assistant's Roborock ``image.py``:

* dual inheritance ``MammotionBaseEntity + ImageEntity`` so the existing
  coordinator wiring is reused
* static base render lives on the coordinator (off-loop, PIL in an executor)
* the entity only does the cheap composite + cache + change detection
* ``_attr_image_last_updated`` is bumped only when output bytes change, so
  the Lovelace picture card doesn't thrash on pixel-identical redraws
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
from .map_renderer import PLACEHOLDER_PNG, compose_with_position

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
    """Set up one map image per mower."""
    async_add_entities(
        MammotionMapImage(
            mower.map_coordinator, mower.reporting_coordinator, hass
        )
        for mower in entry.runtime_data.mowers
    )


class MammotionMapImage(MammotionBaseEntity, ImageEntity):
    """Rasterized mowing-area map with live robot position and mow progress."""

    _attr_translation_key = "map"
    _attr_content_type = "image/png"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(
        self,
        map_coordinator: MammotionMapUpdateCoordinator,
        reporting_coordinator: MammotionReportUpdateCoordinator,
        hass: HomeAssistant,
    ) -> None:
        """Wire both coordinators in — base via MammotionBaseEntity, reporting bolted on."""
        MammotionBaseEntity.__init__(self, map_coordinator, "map")
        ImageEntity.__init__(self, hass)
        # Keep a narrowly-typed handle so Pyright sees the render-cache attrs
        # without us fighting the CoordinatorEntity[] generic on the parent.
        self._map_coordinator = map_coordinator
        self._reporting_coordinator = reporting_coordinator
        self._cached_base: bytes | None = None
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
        if self._map_coordinator.map_last_render is not None:
            self._attr_image_last_updated = self._map_coordinator.map_last_render
        self._pull_base()
        self._rebuild()

    @callback
    def _handle_coordinator_update(self) -> None:
        """Map coordinator ticked — maybe re-composite with a new base."""
        changed = self._pull_base()
        if changed:
            # Position-only overlay would never refresh the base; force one.
            self._rebuild(force=True)
        else:
            self._rebuild()
        super()._handle_coordinator_update()

    @callback
    def _handle_reporting_update(self) -> None:
        """Reporting coordinator ticked — redraw overlay (throttled)."""
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
    def _pull_base(self) -> bool:
        """Mirror the coordinator's latest base PNG onto the entity.

        Returns True when the cached base changed — caller can skip extra work
        on no-ops.
        """
        base = self._map_coordinator.map_base_png_bytes
        if base is not None and base is not self._cached_base:
            self._cached_base = base
            return True
        return False

    @callback
    def _rebuild(self, *, force: bool = False) -> None:
        """Re-composite the cached output PNG from base + live overlays."""
        base = self._cached_base
        if base is None:
            output = PLACEHOLDER_PNG
        else:
            bbox = self._map_coordinator.map_base_bbox
            x, y, heading = self._current_position()
            self._update_position_history(x, y)
            dynamics = self._current_dynamics_line()
            if bbox is None:
                output = base
            else:
                try:
                    output = compose_with_position(
                        base,
                        bbox,
                        x,
                        y,
                        heading,
                        dynamics,
                        mowed_track=self._position_history,
                    )
                except Exception as err:  # noqa: BLE001 - never brick the entity
                    _LOGGER.warning(
                        "Map overlay composite failed for %s: %s",
                        self.coordinator.device_name,
                        err,
                    )
                    output = base

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
        """Return cached bytes — never raises, serves placeholder pre-sync."""
        return self._cached_output if self._cached_output is not None else PLACEHOLDER_PNG
