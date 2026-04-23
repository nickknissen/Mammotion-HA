"""Tests for the Mammotion map renderer.

Pure-Pillow tests — no Home Assistant dependency.  We load
``custom_components/mammotion/map_renderer.py`` by path so importing the
parent package (which pulls in ``homeassistant``) is avoided.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from pymammotion.data.model.hash_list import (
    CommDataCouple,
    FrameList,
    HashList,
    MowPath,
    MowPathPacket,
    NavGetCommData,
)

_RENDERER_PATH = (
    Path(__file__).parent.parent
    / "custom_components"
    / "mammotion"
    / "map_renderer.py"
)


def _load_renderer():
    spec = importlib.util.spec_from_file_location("_mmap_renderer", _RENDERER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_mmap_renderer"] = module
    spec.loader.exec_module(module)
    return module


renderer = _load_renderer()


def _couples(points: list[tuple[float, float]]) -> list[CommDataCouple]:
    return [CommDataCouple(x=x, y=y) for x, y in points]


def _sample_hashlist() -> HashList:
    hl = HashList()
    # One rectangular mowing area.
    hl.area[111] = FrameList(
        total_frame=1,
        sub_cmd=0,
        data=[
            NavGetCommData(
                total_frame=1,
                current_frame=1,
                data_couple=_couples([(-5, -3), (5, -3), (5, 3), (-5, 3)]),
            )
        ],
    )
    # One keep-out obstacle.
    hl.obstacle[222] = FrameList(
        total_frame=1,
        sub_cmd=0,
        data=[
            NavGetCommData(
                total_frame=1,
                current_frame=1,
                data_couple=_couples(
                    [(-1, -0.5), (1, -0.5), (1, 0.5), (-1, 0.5)]
                ),
            )
        ],
    )
    # One serpentine planned mow path.
    stripes: list[tuple[float, float]] = []
    for i, y in enumerate([-2, -1, 0, 1, 2]):
        xs = [-4, 4] if i % 2 == 0 else [4, -4]
        stripes.extend((x, y) for x in xs)
    hl.current_mow_path[1] = {
        1: MowPath(
            total_frame=1,
            current_frame=1,
            path_packets=[
                MowPathPacket(
                    path_hash=1, path_cur=0, data_couple=_couples(stripes)
                )
            ],
        )
    }
    return hl


def test_render_base_png_returns_valid_png() -> None:
    png, bbox = renderer.render_base_png(_sample_hashlist(), rtk_xy=(0, 0))
    assert png is not None
    assert bbox is not None
    assert png.startswith(renderer.PNG_MAGIC)
    # Non-trivial output — placeholder is ~18 KB, base ~20 KB; both well below 1 MB.
    assert 1_000 < len(png) < 1_000_000
    assert bbox.xmin < -4 and bbox.xmax > 4
    assert bbox.ymin < -2 and bbox.ymax > 2


def test_render_base_png_empty_hashlist_returns_none() -> None:
    png, bbox = renderer.render_base_png(HashList())
    assert png is None and bbox is None


def test_placeholder_is_valid_png() -> None:
    assert renderer.PLACEHOLDER_PNG.startswith(renderer.PNG_MAGIC)
    assert len(renderer.PLACEHOLDER_PNG) > 100


def test_render_overlay_png_with_position_is_valid() -> None:
    hl = _sample_hashlist()
    _, bbox = renderer.render_base_png(hl, rtk_xy=(0, 0))
    out = renderer.render_overlay_png(
        bbox, x=2.5, y=1.5, heading_deg=45.0, dynamics_line=None
    )
    assert out.startswith(renderer.PNG_MAGIC)
    assert out != renderer.EMPTY_OVERLAY_PNG


def test_render_overlay_png_without_any_layers_is_transparent() -> None:
    """No position, no dynamics_line, no track → output is a fully-transparent PNG."""
    hl = _sample_hashlist()
    _, bbox = renderer.render_base_png(hl, rtk_xy=(0, 0))
    out = renderer.render_overlay_png(
        bbox, x=None, y=None, heading_deg=None, dynamics_line=None
    )
    assert out.startswith(renderer.PNG_MAGIC)
    assert out == renderer.EMPTY_OVERLAY_PNG


def test_render_overlay_png_dynamics_line_changes_output() -> None:
    hl = _sample_hashlist()
    _, bbox = renderer.render_base_png(hl, rtk_xy=(0, 0))
    out_no_dyn = renderer.render_overlay_png(
        bbox, x=0.0, y=0.0, heading_deg=0.0, dynamics_line=None
    )
    dyn = _couples([(-4, -2), (4, -2), (4, -1), (-4, -1)])
    out_with_dyn = renderer.render_overlay_png(
        bbox, x=0.0, y=0.0, heading_deg=0.0, dynamics_line=dyn
    )
    assert out_no_dyn != out_with_dyn


def test_render_overlay_png_mowed_track_changes_output() -> None:
    """Accumulated ``mowed_track`` should render a visible swathe."""
    hl = _sample_hashlist()
    _, bbox = renderer.render_base_png(hl, rtk_xy=(0, 0))
    out_bare = renderer.render_overlay_png(
        bbox, x=0.0, y=0.0, heading_deg=0.0
    )
    track = [(-4, -2), (-2, -2), (0, -2), (2, -2), (4, -2)]
    out_track = renderer.render_overlay_png(
        bbox, x=0.0, y=0.0, heading_deg=0.0, mowed_track=track
    )
    assert out_track.startswith(renderer.PNG_MAGIC)
    assert out_bare != out_track


def test_position_outside_bbox_is_clamped() -> None:
    """Marker must stay on-canvas; dry-run the clamp helper + sanity-check overlay."""
    hl = _sample_hashlist()
    _, bbox = renderer.render_base_png(hl, rtk_xy=(0, 0))
    cx, cy, clamped = renderer._clamp_to_bbox(999.0, 999.0, bbox)
    assert clamped is True
    assert cx <= bbox.xmax and cy <= bbox.ymax
    # Rendering must not raise for out-of-bbox positions.
    out = renderer.render_overlay_png(
        bbox, x=999.0, y=999.0, heading_deg=0.0, dynamics_line=None
    )
    assert out.startswith(renderer.PNG_MAGIC)


def test_bbox_expands_and_contains() -> None:
    bbox = renderer.BBox(0.0, 0.0, 10.0, 10.0)
    wide = bbox.expanded(0.1)
    assert wide.xmin < 0 and wide.xmax > 10
    assert bbox.contains(5, 5)
    assert not bbox.contains(-1, 5)
    assert bbox.union_point(-1, 20) == renderer.BBox(-1.0, 0.0, 10.0, 20.0)


def test_projector_flips_y() -> None:
    bbox = renderer.BBox(0.0, 0.0, 10.0, 10.0)
    project, _scale = renderer._projector(bbox, (100, 100))
    _, py_top = project(0.0, 10.0)  # world +Y (north) → image top
    _, py_bot = project(0.0, 0.0)  # world y=0    → image bottom
    assert py_top < py_bot
