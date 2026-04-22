# Plan — Mammotion Map Image Entity (Roborock pattern)

Target repo: `../Mammotion-HA/` · Library: `../PyMammotion/` (no changes expected).

## Guiding principles (from Roborock reference + counselors review)

- `ImageEntity` via dual inheritance with the existing `MammotionBaseEntity(CoordinatorEntity)` — identical shape to Roborock's `RoborockCoordinatedEntityV1 + ImageEntity`.
- Render **off-loop in the coordinator** (`hass.async_add_executor_job`). The entity itself only composites cheap overlays and returns cached bytes.
- Cache rendered bytes on the **coordinator**, not the entity (survives entity reload; matches Roborock's `coordinator.last_home_update`).
- Bump `_attr_image_last_updated` only when the rendered bytes actually change (Roborock does the same with `self.cached_map` diff).
- Set `_attr_content_type = "image/png"` explicitly — `ImageEntity` defaults to JPEG.
- `PARALLEL_UPDATES = 0`.
- Multi-device: unique_id via existing `coordinator.unique_name` pattern.
- Y-axis flip (ENU +Y up → image +Y down). Supersample 2× + LANCZOS for AA.
- Placeholder PNG for "map not yet fetched"; never raise in `async_image()` during startup.

## File changes

```
custom_components/mammotion/
├── __init__.py                 [modify]  add Platform.IMAGE to PLATFORMS
├── coordinator.py              [modify]  cache rendered base PNG on MammotionMapUpdateCoordinator
├── image.py                    [NEW]     MammotionMapImage platform
├── map_renderer.py             [NEW]     Pillow rendering (pure functions, no HA imports)
├── placeholder_map.png         [NEW]     "No map available" fallback
└── tests/
    ├── fixtures/
    │   └── hash_list_sample.json   [NEW]  dumped real HashList
    └── test_map_renderer.py        [NEW]  regression tests
```

No changes to PyMammotion.

## Data flow

```
MapFetchSaga (PyMammotion)
   │  fills MowingDevice.map: HashList
   ▼
MammotionMapUpdateCoordinator._async_update_data          (every 30 min)
   │  if HashList content-hash changed AND no missing_hashlist():
   │     base_png = await hass.async_add_executor_job(render_base_png, snapshot)
   │     self.map_base_png_bytes = base_png
   │     self.map_base_bbox      = bbox
   │     self.map_last_render    = utcnow()
   │  async_update_listeners()                           (drives _handle_coordinator_update)
   ▼
MammotionMapImage._handle_coordinator_update             (map tick → full re-composite)
MammotionMapImage._handle_reporting_update               (report tick → position-only re-composite)
   │  current_pos = reporting_coord.data.location (x, y, heading)
   │  out = compose_with_position(base, current_pos, bbox)
   │  if out != self._cached_output_png:
   │     self._cached_output_png  = out
   │     self._attr_image_last_updated = utcnow()
   │     self.async_write_ha_state()
   ▼
async_image() → return self._cached_output_png           (cheap)
```

## Key code shapes (to be filled in, not final)

### `map_renderer.py` — pure, HA-free

```python
from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
from PIL import Image, ImageDraw

CANVAS_W, CANVAS_H = 1024, 768
SS = 2  # supersample factor

# mirror keys from generate_geojson.py style dicts
AREA_FILL    = (0, 100, 0, 77)        # darkgreen @ 30%
AREA_STROKE  = (0, 128, 0, 204)       # green @ 80%
OBSTACLE_FILL   = (255, 140, 0, 102)  # darkorange @ 40%
OBSTACLE_STROKE = (255, 77, 0, 230)
PATH_WHITE   = (255, 255, 255, 255)
PATH_CENTER  = (105, 105, 105, 255)   # #696969
DOCK_COLOR   = (211, 211, 211, 255)   # lightgray
RTK_COLOR    = (128, 0, 128, 255)     # purple

@dataclass
class BBox:
    xmin: float; ymin: float; xmax: float; ymax: float

def render_base_png(hash_list, rtk_xy=None, dock_xy=None, dock_rot=0) -> tuple[bytes, BBox]:
    bbox = _compute_bbox(hash_list, rtk_xy, dock_xy)
    img = Image.new("RGBA", (CANVAS_W * SS, CANVAS_H * SS), (250, 250, 250, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    # dump → area → obstacle → mow_path → dock → RTK
    ...
    img = img.resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
    buf = BytesIO(); img.save(buf, "PNG", optimize=True)
    return buf.getvalue(), bbox

def compose_with_position(base_png: bytes, x, y, heading_deg, bbox: BBox) -> bytes:
    img = Image.open(BytesIO(base_png)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    px, py = _world_to_pixel(x, y, bbox, img.size)
    _draw_robot(d, px, py, heading_deg)
    out = Image.alpha_composite(img, overlay)
    buf = BytesIO(); out.save(buf, "PNG", optimize=True)
    return buf.getvalue()
```

### `coordinator.py` — extend existing class

```python
# MammotionMapUpdateCoordinator (existing class ~line 1354)
class MammotionMapUpdateCoordinator(MammotionBaseUpdateCoordinator[MowerInfo]):
    map_base_png_bytes: bytes | None = None
    map_base_bbox: BBox | None = None
    map_last_render: datetime | None = None
    _last_map_content_hash: int | None = None

    async def _async_update_data(self):
        data = await super()._async_update_data()
        # existing logic that runs MapFetchSaga …
        device = self.manager.get_device_by_name(self.device_name).mower_state
        hl = device.map
        if not hl.hashlist or hl.missing_hashlist():
            return data  # skip render until complete
        content_hash = self._content_hash(hl)
        if content_hash != self._last_map_content_hash:
            # snapshot to detach from live mutation
            snapshot = copy.deepcopy(hl)
            rtk_xy  = _extract_rtk(device)
            dock_xy = _extract_dock(device)
            png, bbox = await self.hass.async_add_executor_job(
                render_base_png, snapshot, rtk_xy, dock_xy,
            )
            self.map_base_png_bytes = png
            self.map_base_bbox = bbox
            self.map_last_render = dt_util.utcnow()
            self._last_map_content_hash = content_hash
        return data
```

### `image.py` — new platform

```python
from __future__ import annotations
import logging
from homeassistant.components.image import ImageEntity
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util

from . import MammotionConfigEntry
from .entity import MammotionBaseEntity
from .map_renderer import compose_with_position, PLACEHOLDER_PNG

_LOGGER = logging.getLogger(__name__)
PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    entry: MammotionConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    async_add_entities(
        MammotionMapImage(m.map_coordinator, m.reporting_coordinator, hass)
        for m in entry.runtime_data.mowers
    )


class MammotionMapImage(MammotionBaseEntity, ImageEntity):
    _attr_translation_key = "map"
    _attr_content_type = "image/png"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, map_coord, reporting_coord, hass: HomeAssistant) -> None:
        MammotionBaseEntity.__init__(self, map_coord, "map")
        ImageEntity.__init__(self, hass)
        self._reporting_coord = reporting_coord
        self._cached_base: bytes | None = None
        self._cached_output: bytes | None = None

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self.async_on_remove(
            self._reporting_coord.async_add_listener(self._handle_reporting_update)
        )
        if self.coordinator.map_last_render:
            self._attr_image_last_updated = self.coordinator.map_last_render
        self._rebuild()

    @callback
    def _handle_coordinator_update(self) -> None:
        base = self.coordinator.map_base_png_bytes
        if base is not None and base != self._cached_base:
            self._cached_base = base
        self._rebuild()
        super()._handle_coordinator_update()

    @callback
    def _handle_reporting_update(self) -> None:
        self._rebuild()
        self.async_write_ha_state()

    @callback
    def _rebuild(self) -> None:
        base = self._cached_base
        if base is None:
            out = PLACEHOLDER_PNG
        else:
            pos = self._current_position()
            bbox = self.coordinator.map_base_bbox
            out = (compose_with_position(base, *pos, bbox)
                   if pos and bbox else base)
        if out != self._cached_output:
            self._cached_output = out
            self._attr_image_last_updated = dt_util.utcnow()

    async def async_image(self) -> bytes | None:
        if self._cached_output is None:
            return PLACEHOLDER_PNG
        return self._cached_output

    def _current_position(self):
        # extract (x, y, heading_deg) from reporting_coord.data — tbd
        ...
```

### `__init__.py` — one-line change

```python
PLATFORMS: list[Platform] = [
    ...,
    Platform.CAMERA,
    Platform.IMAGE,   # ← add
]
```

## Open decisions (to resolve while coding)

1. **Where does live position live?** `reporting_coord.data` is a `MowingDevice`; find which field carries `(x, y, heading)` — likely `device.location` or inside `toapp_report_data.dev.pos`. Verify during step 3.
2. **RTK origin / dock position** — grabbed from `MowingDevice` or `RTKBaseStationDevice`? Needed for renderer markers. Check `generate_geojson.py` — it already resolves these from the same model.
3. **Rate-limit position redraws.** `reporting_coord` may tick faster than 1 Hz during mowing. Add a simple `self._last_position_render` throttle in `_handle_reporting_update` (drop if <1 s since last).
4. **`_attr_image_last_updated` churn.** Roborock only updates on byte-diff; keep that. If position jitter would flap bytes, round x/y/heading to a few decimals before compositing so the diff is stable.
5. **Heading field units.** Device may emit rad or deg — check `MctrlSys` report parser in PyMammotion and normalise once.

## In scope for v1 (expanded)

### Live `dynamics_line` progress overlay
`HashList.dynamics_line: list[CommDataCouple]` is the "mowed-so-far" polyline for the active session. Key facts (from `pymammotion/data/model/hash_list.py:307` and the counselors review):
- Not hash-keyed. Replaced wholesale each fetch.
- Fetched via `CommonDataSaga` with `action=8, type=18` — already handled by PyMammotion; we just read the field.
- Meaningful only mid-mow; empty otherwise.

Rendering: **in the overlay, not the base** — it mutates at reporting cadence. `compose_with_position()` takes an optional `dynamics_line` list and draws it as a bright-green polyline (`#00E676`) on top of the baked white/grey planned path, so the user sees planned vs completed at a glance. Snapshot the list at overlay-time so a concurrent refresh doesn't tear the render.

### `entity_picture` on `LawnMowerEntity`
Show the map thumbnail inline in the default more-info card. Mechanism:
1. After platform setup, look up the map image entity's `entity_id` via `entity_registry.async_get_entity_id(Platform.IMAGE, DOMAIN, f"{coordinator.unique_name}_map")`.
2. Set `self._attr_entity_picture = f"/api/image_proxy/{image_entity_id}?token={token}"` where `token` is a short-lived access token — but the simpler path HA uses is just `/api/image_proxy/{entity_id}` which is already auth-scoped.
3. Cache-bust by updating the URL's `?t=` query param when `image.image_last_updated` changes (listen via `async_track_state_change_event`), otherwise the image_proxy can serve stale bytes to the dashboard.

## Out of scope for v1

- SVG output / GeoJSON view (Opus recommended; defer pending whether `ha-mammotion-assets` is a real planned frontend).
- Configurable canvas size / styles via entry options.

## Testing checklist

- Renderer returns bytes starting `\x89PNG`.
- Empty `HashList` → placeholder returned.
- Position outside bbox → marker clamped to edge, not off-canvas.
- Multi-mower config entry yields one image entity per mower with distinct unique_ids.
- `async_image()` never raises when called before first saga completes.
- No Pillow work on the event loop (`asyncio.run(..., debug=True)` reports no blocking >100 ms on the event loop during updates).

## Implementation order (matches task list)

1. Renderer + placeholder (no HA)
2. Compose-with-position overlay
3. Coordinator cache extension
4. Image platform
5. Platform registration
6. Placeholder asset + fallback
7. Tests + fixture capture
8. `dynamics_line` overlay (extension of step 2)
9. `entity_picture` on LawnMowerEntity

Steps 1, 2, 8 are pure Pillow and can all be finished + fixtured-tested before touching any HA code.

Steps 1-2 can be done and unit-tested standalone against a dumped `HashList` fixture before touching HA wiring — lowest risk path.
