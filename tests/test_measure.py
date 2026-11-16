"""One runnable check for the geometry. `python -m pytest tests/` or just run this file.

A synthetic ellipsoid with known semi-axes, on deliberately anisotropic spacing, because
anisotropy is exactly where a diameter calculation silently goes wrong.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caliper.measure import axial_axes, classify, measure_lesion, sheet_row, volume_mm3

SPACING = (2.5, 0.7, 0.7)  # RIDER-like: thick slices, fine in-plane
A, B, C = 20.0, 12.0, 8.0  # semi-axes in mm along x, y, z


def ellipsoid():
    half = np.array([C, B, A]) / np.array(SPACING) + 2
    z, y, x = np.mgrid[
        -int(half[0]) : int(half[0]) + 1,
        -int(half[1]) : int(half[1]) + 1,
        -int(half[2]) : int(half[2]) + 1,
    ]
    zz, yy, xx = z * SPACING[0], y * SPACING[1], x * SPACING[2]
    return ((xx / A) ** 2 + (yy / B) ** 2 + (zz / C) ** 2) <= 1.0


def demo():
    mask = ellipsoid()

    long_axis, short_axis, slice_index = axial_axes(mask, SPACING)
    # Voxel centres, not voxel edges, so a diameter reads ~1 voxel short. 6% covers it.
    assert abs(long_axis - 2 * A) / (2 * A) < 0.06, long_axis
    assert abs(short_axis - 2 * B) / (2 * B) < 0.06, short_axis
    # Longest axial cut is the equator, i.e. the middle slice.
    assert slice_index == mask.shape[0] // 2, slice_index

    expected = 4 / 3 * np.pi * A * B * C
    assert abs(volume_mm3(mask, SPACING) - expected) / expected < 0.05

    # Short axis must come from the long axis's own slice. A 30mm disc on another slice is
    # wider than this lesion's 24mm short axis but shorter than its 40mm long axis, so a
    # naive "max width anywhere" would report 30 and be wrong.
    disc = np.zeros_like(mask)
    yy, xx = np.mgrid[: mask.shape[1], : mask.shape[2]]
    centre = (mask.shape[1] // 2, mask.shape[2] // 2)
    disc[0] = ((yy - centre[0]) * SPACING[1]) ** 2 + ((xx - centre[1]) * SPACING[2]) ** 2 <= 15.0**2
    long_combined, short_combined, _ = axial_axes(mask | disc, SPACING)
    assert abs(long_combined - long_axis) < 1e-6, long_combined
    assert abs(short_combined - short_axis) < 1e-6, short_combined

    # Density split: everything solid, then everything ground-glass.
    solid = np.full(mask.shape, 50.0)
    assert measure_lesion(mask, solid, SPACING)["type"] == "solid"
    assert measure_lesion(mask, np.full(mask.shape, -700.0), SPACING)["type"] == "ground-glass"
    assert classify(0.5) == "part-solid"

    # Empty mask returns zeros rather than raising, because a rejected candidate still
    # has to render a row.
    empty = measure_lesion(np.zeros((4, 4, 4), bool), None, SPACING)
    assert empty["long_axis_mm"] == 0.0 and empty["slice_index"] == -1

    row = sheet_row(
        measure_lesion(mask, solid, SPACING),
        patient_id="RIDER-1129164940",
        scan_date="2006-09-20",
        timepoint="test",
        series_number=8,
        slice_thickness=SPACING[0],
        nodule_id="N1",
        lobe="lung_upper_lobe_right",
    )
    from caliper.measure import SHEET_COLUMNS

    assert list(row) == SHEET_COLUMNS, "export must match the lab's sheet exactly"
    assert row["Measured by"] == "", "the tool pre-fills; a human signs"

    print(f"long {long_axis:.1f}mm (want {2*A}) short {short_axis:.1f}mm (want {2*B}) ok")


if __name__ == "__main__":
    demo()


def test_measure():
    demo()
