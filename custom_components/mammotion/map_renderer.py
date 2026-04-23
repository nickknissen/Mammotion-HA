"""Rasterize a Mammotion ``HashList`` to a PNG image.

Pure functions — no Home Assistant imports — so this module is unit-testable
against a dumped ``HashList`` fixture without touching the HA event loop.

All coordinates on the wire are ``CommDataCouple(x, y)`` metres in the local
RTK frame (ENU-ish; +Y points north).  This renderer fixes a canvas bbox over
all map geometry, scales-to-fit, and flips Y so +Y renders upward in the
output image.

Intended use:

* ``render_base_png(hash_list, rtk_xy, dock_xy, dock_rotation)`` — called
  off-loop from the map coordinator whenever the hash-keyed content changes.
  Produces the static background.
* ``compose_with_position(base, bbox, x, y, heading, dynamics_line)`` — called
  on the event loop from the image entity on every reporting-coordinator tick.
  Overlays the live robot marker plus the mow-progress polyline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from pymammotion.data.model.hash_list import (
        CommDataCouple,
        FrameList,
        HashList,
    )

CANVAS_W = 1024
CANVAS_H = 768
SUPERSAMPLE = 2  # render at 2x then LANCZOS-downsample for AA on thick strokes
MARGIN_FRAC = 0.05  # pad bbox by 5% of its own span

BG_COLOR = (245, 245, 245, 255)

# Mirrors pymammotion/data/model/generate_geojson.py so the raster matches the
# GeoJSON card visually.  Alpha comes from the original fillOpacity / opacity.
AREA_FILL = (0, 100, 0, 77)          # darkgreen @ 30%
AREA_STROKE = (0, 128, 0, 204)       # green    @ 80%
OBSTACLE_FILL = (255, 140, 0, 102)   # darkorange @ 40%
OBSTACLE_STROKE = (255, 77, 0, 230)  # #FF4D00   @ 90%
DUMP_FILL = (255, 215, 0, 90)        # gold-ish fill
DUMP_STROKE = (184, 134, 11, 200)    # darkgoldenrod
PATH_OUTER = (255, 255, 255, 255)    # planned mow stripes
PATH_CENTER = (105, 105, 105, 255)   # #696969
LINE_COLOR = (255, 165, 0, 220)      # breakpoint/resume line (orange)
DOCK_FILL = (211, 211, 211, 255)     # lightgray
DOCK_STROKE = (80, 80, 80, 255)
RTK_FILL = (128, 0, 128, 255)        # purple
RTK_STROKE = (50, 0, 50, 255)
ROBOT_FILL = (0, 120, 255, 255)      # bright blue
ROBOT_STROKE = (0, 0, 0, 255)
DYNAMICS_LINE_COLOR = (0, 230, 118, 255)  # #00E676 — "mowed so far"
MOWED_SWATHE_COLOR = (76, 175, 80, 160)   # #4CAF50 alpha — filled-in mowed area

# Stroke widths in metres (converted to pixels per render).  Using world units
# means a 10 m yard and a 100 m yard render with visually-consistent stroke
# thickness relative to the map scale.
STROKE_AREA_M = 0.20
STROKE_OBSTACLE_M = 0.15
STROKE_PATH_OUTER_M = 0.25
STROKE_PATH_CENTER_M = 0.08
STROKE_DOCK_M = 0.10
STROKE_RTK_M = 0.10
STROKE_DYNAMICS_M = 0.30
# Approximate mower cutting swathe in metres — wide enough that the
# accumulated position track visually fills the mowed area like the app.
STROKE_MOWED_SWATHE_M = 0.40

# Marker radii in metres.
RADIUS_DOCK_M = 0.30
RADIUS_RTK_M = 0.25
RADIUS_ROBOT_M = 0.40

# Absolute pixel floors so strokes are never subpixel on tiny yards.
MIN_STROKE_PX = 2
MIN_MARKER_PX = 6

# Used by HA tests and unit tests to detect valid PNG output.
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in world metres."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    def expanded(self, frac: float) -> "BBox":
        """Grow by *frac* of the current span in each direction."""
        dx = max(self.width * frac, 0.5)
        dy = max(self.height * frac, 0.5)
        return BBox(self.xmin - dx, self.ymin - dy, self.xmax + dx, self.ymax + dy)

    def union_point(self, x: float, y: float) -> "BBox":
        return BBox(
            min(self.xmin, x),
            min(self.ymin, y),
            max(self.xmax, x),
            max(self.ymax, y),
        )

    def contains(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


# Loaded lazily at module import time for the "no map yet" fallback.
_PLACEHOLDER_PATH = Path(__file__).parent / "placeholder_map.png"


def _load_placeholder() -> bytes:
    if _PLACEHOLDER_PATH.exists():
        return _PLACEHOLDER_PATH.read_bytes()
    # Fallback if the asset is missing — generate in-memory so imports never fail.
    img = Image.new("RGBA", (CANVAS_W, CANVAS_H), BG_COLOR)
    draw = ImageDraw.Draw(img)
    text = "No map available yet"
    # default PIL font is fine; we don't ship a TTF.
    tw, th = draw.textbbox((0, 0), text)[2:]
    draw.text(
        ((CANVAS_W - tw) // 2, (CANVAS_H - th) // 2),
        text,
        fill=(120, 120, 120, 255),
    )
    buf = BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


PLACEHOLDER_PNG: bytes = _load_placeholder()


# ---------------------------------------------------------------------------
# Bbox + projection
# ---------------------------------------------------------------------------


def _points_from_framelist(frame_list: "FrameList") -> list[tuple[float, float]]:
    """Concatenate data_couple lists across all frames of a FrameList.

    Mirrors ``GeojsonGenerator._collect_frame_coordinates`` (PyMammotion).
    Skip SvgMessage frames — they carry a different transform.
    """
    # Local import to avoid pulling PyMammotion at module import time for tests
    # that stub the library.
    from pymammotion.data.model.hash_list import NavGetCommData

    pts: list[tuple[float, float]] = []
    for frame in frame_list.data:
        if isinstance(frame, NavGetCommData):
            pts.extend((p.x, p.y) for p in frame.data_couple)
    return pts


def _iter_hashlist_points(hash_list: "HashList") -> list[tuple[float, float]]:
    """Every (x, y) point known across the static HashList content."""
    acc: list[tuple[float, float]] = []
    for frame_list in hash_list.area.values():
        acc.extend(_points_from_framelist(frame_list))
    for frame_list in hash_list.obstacle.values():
        acc.extend(_points_from_framelist(frame_list))
    for frame_list in hash_list.dump.values():
        acc.extend(_points_from_framelist(frame_list))
    for frame_list in hash_list.path.values():
        acc.extend(_points_from_framelist(frame_list))
    # current_mow_path has a nested structure: dict[transaction_id, dict[frame_idx, MowPath]]
    for frames_by_tx in hash_list.current_mow_path.values():
        for mow_path in frames_by_tx.values():
            for packet in mow_path.path_packets:
                acc.extend((p.x, p.y) for p in packet.data_couple)
    for frame_list in hash_list.line.values():
        acc.extend(_points_from_framelist(frame_list))
    return acc


def _compute_bbox(
    hash_list: "HashList",
    rtk_xy: tuple[float, float] | None,
    dock_xy: tuple[float, float] | None,
) -> BBox | None:
    """Return padded bbox over all static geometry, or None if nothing to render."""
    pts = _iter_hashlist_points(hash_list)
    if rtk_xy is not None:
        pts.append(rtk_xy)
    if dock_xy is not None:
        pts.append(dock_xy)
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    bbox = BBox(min(xs), min(ys), max(xs), max(ys))
    # Guarantee a non-zero area for single-point edge cases.
    if bbox.width == 0 or bbox.height == 0:
        bbox = BBox(bbox.xmin - 1.0, bbox.ymin - 1.0, bbox.xmax + 1.0, bbox.ymax + 1.0)
    return bbox.expanded(MARGIN_FRAC)


def _projector(bbox: BBox, canvas_size: tuple[int, int]):
    """Return a closure mapping world (x, y) metres → image pixel (px, py).

    Y is flipped: world +Y (north) → image top (0) at ``ymax``.
    """
    cw, ch = canvas_size
    sx = cw / bbox.width
    sy = ch / bbox.height
    scale = min(sx, sy)  # uniform → never distort aspect
    # Center the map in the canvas when the aspect ratios don't match.
    used_w = bbox.width * scale
    used_h = bbox.height * scale
    offset_x = (cw - used_w) / 2.0
    offset_y = (ch - used_h) / 2.0

    def project(x: float, y: float) -> tuple[float, float]:
        px = offset_x + (x - bbox.xmin) * scale
        py = offset_y + (bbox.ymax - y) * scale  # flip Y
        return px, py

    return project, scale


def _m_to_px(scale: float, metres: float, min_px: int = 1) -> int:
    """Convert a world-metres distance to pixels, floored to *min_px*."""
    return max(int(round(metres * scale)), min_px)


# ---------------------------------------------------------------------------
# Base render
# ---------------------------------------------------------------------------


def render_base_png(
    hash_list: "HashList",
    rtk_xy: tuple[float, float] | None = None,
    dock_xy: tuple[float, float] | None = None,
    dock_rotation: float = 0.0,
) -> tuple[bytes, BBox] | tuple[None, None]:
    """Render the static layers of *hash_list* to PNG bytes.

    Returns ``(png_bytes, bbox)`` so the entity can use *bbox* later to
    project the live robot position into the same image space.

    Returns ``(None, None)`` when there's nothing to draw — the entity then
    falls back to :data:`PLACEHOLDER_PNG`.
    """
    bbox = _compute_bbox(hash_list, rtk_xy, dock_xy)
    if bbox is None:
        return None, None

    ss_size = (CANVAS_W * SUPERSAMPLE, CANVAS_H * SUPERSAMPLE)
    img = Image.new("RGBA", ss_size, BG_COLOR)
    draw = ImageDraw.Draw(img, "RGBA")

    project, scale = _projector(bbox, ss_size)

    def to_px(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        return [project(x, y) for x, y in pts]

    # Draw order: dump (lowest) → area → obstacle → path → mow path → dock → RTK.
    # Each layer is keyed-by-hash in HashList; we iterate in hash order so the
    # render is deterministic (important for byte-diff change detection).

    for hash_key in sorted(hash_list.dump.keys()):
        pts = to_px(_points_from_framelist(hash_list.dump[hash_key]))
        if len(pts) >= 3:
            draw.polygon(
                pts,
                fill=DUMP_FILL,
                outline=DUMP_STROKE,
                width=_m_to_px(scale, STROKE_AREA_M, MIN_STROKE_PX),
            )

    for hash_key in sorted(hash_list.area.keys()):
        pts = to_px(_points_from_framelist(hash_list.area[hash_key]))
        if len(pts) >= 3:
            draw.polygon(
                pts,
                fill=AREA_FILL,
                outline=AREA_STROKE,
                width=_m_to_px(scale, STROKE_AREA_M, MIN_STROKE_PX),
            )

    for hash_key in sorted(hash_list.obstacle.keys()):
        pts = to_px(_points_from_framelist(hash_list.obstacle[hash_key]))
        if len(pts) >= 3:
            draw.polygon(
                pts,
                fill=OBSTACLE_FILL,
                outline=OBSTACLE_STROKE,
                width=_m_to_px(scale, STROKE_OBSTACLE_M, MIN_STROKE_PX),
            )

    # `path` (type 2) is the recorded travel path for Luba 1 path-mode.
    for hash_key in sorted(hash_list.path.keys()):
        pts = to_px(_points_from_framelist(hash_list.path[hash_key]))
        if len(pts) >= 2:
            draw.line(
                pts,
                fill=PATH_CENTER,
                width=_m_to_px(scale, STROKE_PATH_CENTER_M, MIN_STROKE_PX),
                joint="curve",
            )

    # Planned mow-path stripes from current_mow_path.  White outer + dark centre
    # dashed centreline, to match the GeoJSON card.
    mow_segments = _mow_path_segments(hash_list)
    path_outer_w = _m_to_px(scale, STROKE_PATH_OUTER_M, MIN_STROKE_PX)
    path_center_w = max(_m_to_px(scale, STROKE_PATH_CENTER_M, 1), 1)
    for seg in mow_segments:
        if len(seg) < 2:
            continue
        px_seg = [project(x, y) for x, y in seg]
        draw.line(px_seg, fill=PATH_OUTER, width=path_outer_w, joint="curve")
        draw.line(px_seg, fill=PATH_CENTER, width=path_center_w, joint="curve")

    # Dock + RTK origin markers last so they sit on top of geometry.
    if dock_xy is not None:
        _draw_marker(
            draw,
            project(*dock_xy),
            radius_px=_m_to_px(scale, RADIUS_DOCK_M, MIN_MARKER_PX),
            fill=DOCK_FILL,
            outline=DOCK_STROKE,
            rotation_deg=dock_rotation,
            chevron=True,
        )

    if rtk_xy is not None:
        _draw_marker(
            draw,
            project(*rtk_xy),
            radius_px=_m_to_px(scale, RADIUS_RTK_M, MIN_MARKER_PX),
            fill=RTK_FILL,
            outline=RTK_STROKE,
        )

    # Supersample downscale.
    if SUPERSAMPLE != 1:
        img = img.resize((CANVAS_W, CANVAS_H), Image.Resampling.LANCZOS)

    buf = BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue(), bbox


def _mow_path_segments(hash_list: "HashList") -> list[list[tuple[float, float]]]:
    """Return one point-list per planned mow-path segment (one per path_hash)."""
    # Group packets by path_hash, order within hash by path_cur.
    packets_by_hash: dict[int, dict[int, list[tuple[float, float]]]] = {}
    for frames_by_tx in hash_list.current_mow_path.values():
        for mow_path in frames_by_tx.values():
            for packet in mow_path.path_packets:
                by_cur = packets_by_hash.setdefault(packet.path_hash, {})
                by_cur[packet.path_cur] = [(p.x, p.y) for p in packet.data_couple]

    segments: list[list[tuple[float, float]]] = []
    for path_hash in sorted(packets_by_hash.keys()):
        by_cur = packets_by_hash[path_hash]
        ordered: list[tuple[float, float]] = []
        for cur in sorted(by_cur.keys()):
            ordered.extend(by_cur[cur])
        segments.append(ordered)
    return segments


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    center_px: tuple[float, float],
    radius_px: int,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
    rotation_deg: float = 0.0,
    chevron: bool = False,
) -> None:
    """Filled circle optionally augmented with an orientation chevron."""
    cx, cy = center_px
    draw.ellipse(
        (cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px),
        fill=fill,
        outline=outline,
        width=max(radius_px // 6, 1),
    )
    if chevron:
        # Chevron points in +X world direction by default, rotated by rotation_deg.
        # Image Y is flipped so the rotation sign is flipped too.
        theta = -math.radians(rotation_deg)
        tip = (
            cx + radius_px * 1.6 * math.cos(theta),
            cy + radius_px * 1.6 * math.sin(theta),
        )
        back_r = radius_px * 0.8
        left = (
            cx + back_r * math.cos(theta + math.pi * 0.75),
            cy + back_r * math.sin(theta + math.pi * 0.75),
        )
        right = (
            cx + back_r * math.cos(theta - math.pi * 0.75),
            cy + back_r * math.sin(theta - math.pi * 0.75),
        )
        draw.polygon([tip, left, right], fill=outline, outline=outline)


# ---------------------------------------------------------------------------
# Live overlay
# ---------------------------------------------------------------------------


def compose_with_position(
    base_png: bytes,
    bbox: BBox,
    x: float | None,
    y: float | None,
    heading_deg: float | None,
    dynamics_line: list["CommDataCouple"] | None = None,
    mowed_track: list[tuple[float, float]] | None = None,
) -> bytes:
    """Composite a robot marker (and optional mowed polyline) onto *base_png*.

    * ``(x, y, heading_deg)`` — live device position in the same local ENU
      frame as the base map.  Any may be ``None`` to suppress the marker
      (e.g. pre-fix state).  Position is clamped to the bbox edge if outside.
    * ``dynamics_line`` — the live ``HashList.dynamics_line`` list.  Drawn as
      a narrow polyline in bright green on top of the swathe.
    * ``mowed_track`` — accumulated ``(x, y)`` positions from the entity's
      rolling buffer.  Drawn first, at mower-cutting-width so the visited
      area fills in like the Mammotion app.

    Runs on the event loop — keep it cheap: one decode + one alpha-composite +
    one re-encode.  For typical yard complexity (hundreds of dynamics points)
    this is well under 50 ms on Pi 4 class hardware.
    """
    img = Image.open(BytesIO(base_png)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    project, scale = _projector(bbox, img.size)

    # Mowed swathe (bottom-most overlay) — wide stroke at cutting width so
    # the visited area visibly fills in.
    if mowed_track and len(mowed_track) >= 2:
        swathe_pts = [project(mx, my) for mx, my in mowed_track]
        draw.line(
            swathe_pts,
            fill=MOWED_SWATHE_COLOR,
            width=_m_to_px(scale, STROKE_MOWED_SWATHE_M, MIN_STROKE_PX + 2),
            joint="curve",
        )

    # Dynamics line on top of the swathe — narrower, brighter, traces the
    # live path from the device itself.
    if dynamics_line and len(dynamics_line) >= 2:
        dyn_pts = [(p.x, p.y) for p in dynamics_line]
        px_pts = [project(px_, py_) for px_, py_ in dyn_pts]
        draw.line(
            px_pts,
            fill=DYNAMICS_LINE_COLOR,
            width=_m_to_px(scale, STROKE_DYNAMICS_M, MIN_STROKE_PX),
            joint="curve",
        )

    # Robot marker on top.
    if x is not None and y is not None:
        clamped_x, clamped_y, clamped = _clamp_to_bbox(x, y, bbox)
        px, py = project(clamped_x, clamped_y)
        _draw_marker(
            draw,
            (px, py),
            radius_px=_m_to_px(scale, RADIUS_ROBOT_M, MIN_MARKER_PX),
            fill=ROBOT_FILL,
            outline=ROBOT_STROKE,
            rotation_deg=heading_deg if heading_deg is not None else 0.0,
            chevron=heading_deg is not None,
        )
        if clamped:
            # Draw a hollow ring around the clamped marker to signal "off-map".
            r = _m_to_px(scale, RADIUS_ROBOT_M, MIN_MARKER_PX) + 4
            draw.ellipse(
                (px - r, py - r, px + r, py + r),
                outline=(255, 0, 0, 255),
                width=2,
            )

    out = Image.alpha_composite(img, overlay)
    buf = BytesIO()
    out.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _clamp_to_bbox(x: float, y: float, bbox: BBox) -> tuple[float, float, bool]:
    """Return (clamped_x, clamped_y, was_clamped) inside bbox."""
    cx = min(max(x, bbox.xmin), bbox.xmax)
    cy = min(max(y, bbox.ymin), bbox.ymax)
    return cx, cy, (cx != x or cy != y)


__all__ = [
    "BBox",
    "CANVAS_H",
    "CANVAS_W",
    "PLACEHOLDER_PNG",
    "PNG_MAGIC",
    "compose_with_position",
    "render_base_png",
]
