"""Lesion geometry, in millimetres, straight into the lab's measurement sheet.

Why this file is small and separate: every number the researchers publish comes out of
here, so it has to be readable and testable without a CT scan or a GPU anywhere near it.
`tests/test_measure.py` checks it against a synthetic ellipsoid of known size.

Conventions: arrays are (z, y, x). `spacing` is (z, y, x) in mm, taken from the DICOM
series, never assumed isotropic — RIDER slice thickness is several times the in-plane
pixel size, so an unscaled measurement would be wrong by a factor of three.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

# Threshold separating a solid component from ground-glass density.
# ponytail: one global HU cut. Daniel overrides `Type` in the UI and we log the override
# rate; if that rate is high, this becomes a trained classifier rather than a constant.
SOLID_HU = -160.0

# Fraction of a lesion's volume above SOLID_HU that makes it solid / part-solid / ground-glass.
SOLID_FRACTION = 0.9
GROUND_GLASS_FRACTION = 0.1

# The lab's sheet, in the lab's order. Daniel's words, not ours — anything else makes him re-key.
SHEET_COLUMNS = [
    "Patient ID",
    "Scan Date",
    "Timepoint",
    "Series #",
    "Slice Thickness",
    "Nodule ID",
    "Lobe",
    "Slice #",
    "Type",
    "Long axis",
    "Short axis",
    "Mean diameter (calculated)",
    "Solid component",
    "Volume",
    "Match to prior",
    "Minutes on scan",
    "Measured by",
    "Notes",
]


def _hull_points(mask_slice: np.ndarray, spacing_yx: tuple[float, float]) -> np.ndarray:
    """Boundary points of one axial slice, in mm, as (x, y).

    The convex hull is only a speed trick: the two most distant points of a set are always
    hull vertices, so this throws away the interior without changing the answer.
    """
    ys, xs = np.nonzero(mask_slice)
    if len(xs) == 0:
        return np.empty((0, 2))
    pts = np.column_stack([xs * spacing_yx[1], ys * spacing_yx[0]])
    if len(pts) < 3:
        return pts
    try:
        return pts[ConvexHull(pts).vertices]
    except Exception:
        # Degenerate (collinear) slice — the hull is undefined but the points are still fine.
        return pts


def axial_axes(mask: np.ndarray, spacing: tuple[float, float, float]) -> tuple[float, float, int]:
    """RECIST-style axes: longest in-plane diameter, and the width perpendicular to it.

    Returns (long_axis_mm, short_axis_mm, slice_index). Both axes come from the *same* slice —
    the one where the long axis is largest — because that is how a radiologist reports them.
    Measuring the short axis on a different slice would quietly inflate it.
    """
    best = (0.0, 0.0, -1)
    for z in range(mask.shape[0]):
        pts = _hull_points(mask[z], (spacing[1], spacing[2]))
        if len(pts) < 2:
            continue
        delta = pts[:, None, :] - pts[None, :, :]
        dist = np.hypot(delta[..., 0], delta[..., 1])
        i, j = np.unravel_index(int(dist.argmax()), dist.shape)
        long_axis = float(dist[i, j])
        if long_axis <= best[0]:
            continue
        direction = (pts[j] - pts[i]) / long_axis
        perpendicular = np.array([-direction[1], direction[0]])
        projection = pts @ perpendicular
        best = (long_axis, float(projection.max() - projection.min()), z)
    return best


def volume_mm3(mask: np.ndarray, spacing: tuple[float, float, float]) -> float:
    return float(mask.sum()) * float(np.prod(spacing))


def classify(solid_fraction: float) -> str:
    if solid_fraction >= SOLID_FRACTION:
        return "solid"
    if solid_fraction <= GROUND_GLASS_FRACTION:
        return "ground-glass"
    return "part-solid"


def measure_lesion(
    mask: np.ndarray,
    ct_hu: np.ndarray | None,
    spacing: tuple[float, float, float],
) -> dict:
    """Every geometric number for one lesion. Pure function of the mask and the image."""
    long_axis, short_axis, slice_index = axial_axes(mask, spacing)
    total = volume_mm3(mask, spacing)

    solid_long = solid_volume = 0.0
    solid_fraction = 0.0
    if ct_hu is not None and total > 0:
        solid_mask = mask & (ct_hu >= SOLID_HU)
        solid_volume = volume_mm3(solid_mask, spacing)
        solid_fraction = solid_volume / total
        solid_long = axial_axes(solid_mask, spacing)[0]

    return {
        "slice_index": slice_index,
        "long_axis_mm": round(long_axis, 1),
        "short_axis_mm": round(short_axis, 1),
        "mean_diameter_mm": round((long_axis + short_axis) / 2, 1),
        "volume_mm3": round(total, 1),
        # Daniel records the solid component as a diameter, so report that and keep the
        # volume fraction beside it for the Type call.
        "solid_component_mm": round(solid_long, 1),
        "solid_fraction": round(solid_fraction, 3),
        "type": classify(solid_fraction) if ct_hu is not None else "",
    }


def sheet_row(
    measurement: dict,
    *,
    patient_id: str,
    scan_date: str,
    timepoint: str,
    series_number: int | str,
    slice_thickness: float,
    nodule_id: str,
    lobe: str = "",
    match_to_prior: str = "",
    minutes_on_scan: float | str = "",
    measured_by: str = "",
    notes: str = "",
) -> dict:
    """Pack a measurement into the lab's 18 columns.

    Deliberately dumb: no column is invented, renamed or reordered. `Measured by` stays
    empty until a human accepts the row — the tool pre-fills, it does not sign.
    """
    return {
        "Patient ID": patient_id,
        "Scan Date": scan_date,
        "Timepoint": timepoint,
        "Series #": series_number,
        "Slice Thickness": slice_thickness,
        "Nodule ID": nodule_id,
        "Lobe": lobe,
        "Slice #": measurement["slice_index"],
        "Type": measurement["type"],
        "Long axis": measurement["long_axis_mm"],
        "Short axis": measurement["short_axis_mm"],
        "Mean diameter (calculated)": measurement["mean_diameter_mm"],
        "Solid component": measurement["solid_component_mm"],
        "Volume": measurement["volume_mm3"],
        "Match to prior": match_to_prior,
        "Minutes on scan": minutes_on_scan,
        "Measured by": measured_by,
        "Notes": notes,
    }
