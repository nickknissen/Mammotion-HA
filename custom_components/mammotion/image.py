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
            dynamics = self._current_dynamics_line()
            if bbox is None:
                output = base
            else:
                try:
                    output = compose_with_position(
                        base, bbox, x, y, heading, dynamics
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

        1. ``report_data.work.path_pos_x / path_pos_y`` — scaled int, 1e4 → m,
           already in the local ENU frame so no projection needed.  Present
           during mowing.
        2. ``location.device`` (lat/lng, degrees) projected through the
           RTK-origin ENU converter — available whenever we have a GNSS fix.

        ``heading_deg`` comes from ``location.orientation`` (signed degrees).
        Returns ``(None, None, None)`` when no usable position is available.
        """
        mower = self._mower()
        if mower is None:
            return None, None, None

        heading = (
            float(mower.location.orientation)
            if mower.location.orientation is not None
            else None
        )

        try:
            work = mower.mower_state.report_data.work
            if (work.path_pos_x or work.path_pos_y) and (
                work.path_pos_x != 0 or work.path_pos_y != 0
            ):
                return (
                    work.path_pos_x / 10000.0,
                    work.path_pos_y / 10000.0,
                    heading,
                )
        except AttributeError:
            pass

        # Fall back to lat/lng → ENU via the RTK converter.
        try:
            import math

            from pymammotion.utility.map import CoordinateConverter

            rtk = mower.location.RTK
            dev = mower.location.device
            if (
                rtk.latitude
                and rtk.longitude
                and dev.latitude
                and dev.longitude
            ):
                conv = CoordinateConverter(
                    latitude_rad=math.radians(rtk.latitude),
                    longitude_rad=math.radians(rtk.longitude),
                )
                east, north = conv.lla_to_enu(
                    longitude_deg=dev.longitude, latitude_deg=dev.latitude
                )
                return float(east), float(north), heading
        except Exception as err:  # noqa: BLE001 - best-effort
            _LOGGER.debug(
                "Position projection failed for %s: %s",
                self.coordinator.device_name,
                err,
            )

        return None, None, heading

    def _current_dynamics_line(self):
        """Snapshot ``HashList.dynamics_line`` to detach from live mutation."""
        mower = self._mower()
        if mower is None:
            return None
        try:
            # shallow copy is enough — CommDataCouple is immutable-by-convention
            return list(mower.mower_state.map.dynamics_line)
        except AttributeError:
            return None

    # -- ImageEntity API ----------------------------------------------------

    async def async_image(self) -> bytes | None:
        """Return cached bytes — never raises, serves placeholder pre-sync."""
        return self._cached_output if self._cached_output is not None else PLACEHOLDER_PNG
