"""Faint ground-glass pass, and the vessel filter that makes it affordable.

What it's for, in Daniel's words from the walkthrough: the tool found 5 of 7 candidates on
a practice scan and missed a faint ground-glass nodule, and "if it misses those hazy ones,
I'm still stuck doing the full two-pass hunt manually." The miss costs him the whole second
pass, 14 of his 29 minutes. Noise costs him about 10 seconds a mark. So recall on faint
nodules is worth far more than precision, as long as the noise stays dismissable.

Both noise marks on that scan were vessel bifurcations. That is the lever: map the vessel
tree first and drop candidates that are mostly vessel. With vessels gone, the size floor
can come down to Daniel's own threshold — he records nodules of 4mm and up — instead of the
12mm the density pass needs to stay quiet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from caliper.measure import axial_axes
from caliper.pipeline import CROP_MARGIN, LOBES, _bbox, _linearity, array

# Ground-glass only. Solid tissue above -350 is already covered by the trained model and
# the density pass; parenchyma sits below -750.
FAINT_LO, FAINT_HI = -750.0, -350.0
#: Daniel: "I record any nodule 4 mm or larger. Smaller than that and it's too hard to
#: reliably follow or measure." His threshold, in his units.
FAINT_MIN_DIAMETER_MM = 4.0
FAINT_MAX_DIAMETER_MM = 30.0
FAINT_MAX_LINEARITY = 2.5
#: Reject a candidate when more than this share of it sits on the vessel tree. Not zero:
#: nodules often touch a vessel, and a small contact patch must not cost a real lesion.
VESSEL_MAX_FRACTION = 0.3

#: TotalSegmentator `lung_vessels` (Apache-2.0). Airway walls are in the same density band
#: and tube-shaped, so they go in the reject mask too.
VESSEL_STRUCTURES = ("lung_arteries", "lung_veins", "lung_airways_wall")


def segment_vessels(nifti: Path, work: Path, device: str = "cpu") -> np.ndarray:
    """Vessel tree plus airway walls, on the full CT grid, run on the lung crop for speed."""
    from totalsegmentator.python_api import totalsegmentator

    full = sitk.ReadImage(str(nifti))
    shape = sitk.GetArrayFromImage(full).shape
    lungs = np.zeros(shape, bool)
    for name in LOBES:
        path = work / "lobes" / f"{name}.nii.gz"
        if path.exists():
            lungs |= array(path).astype(bool)
    box = _bbox(lungs, CROP_MARGIN)

    crop_path = work / "ct_lung.nii.gz"
    if not crop_path.exists():
        crop = sitk.GetImageFromArray(sitk.GetArrayFromImage(full)[box])
        crop.SetSpacing(full.GetSpacing())
        sitk.WriteImage(crop, str(crop_path))

    out = work / "lung_vessels"
    if not (out / f"{VESSEL_STRUCTURES[0]}.nii.gz").exists():
        totalsegmentator(input=crop_path, output=out, task="lung_vessels", device=device)

    vessels = np.zeros(shape, bool)
    for name in VESSEL_STRUCTURES:
        path = out / f"{name}.nii.gz"
        if path.exists():
            part = array(path).astype(bool)
            if part.shape == shape:
                vessels |= part
            else:
                vessels[box] |= part
    return vessels


def widen(vessels: np.ndarray) -> np.ndarray:
    """One in-plane voxel of margin, so a candidate wrapped around a vessel counts as on it.

    In-plane only, because slices are about twice as thick as pixels are wide; growing in z
    would swallow a slice either side of every vessel.
    """
    return ndimage.binary_dilation(vessels, np.ones((1, 3, 3), bool))


def trim_vessels(labels: np.ndarray, vessels_wide: np.ndarray, spacing,
                 min_keep_fraction: float = VESSEL_MAX_FRACTION,
                 min_diameter_mm: float = FAINT_MIN_DIAMETER_MM) -> np.ndarray:
    """Cut the vessel tree out of each candidate, and keep what's left if it's still a lesion.

    The first version rejected any candidate that was mostly vessel, and on RIDER-1801720707
    that threw away the expert's tumour: the density candidate was the lesion plus an
    attached vessel, 53% vessel overall, while the lesion alone is 18%. Trimming instead of
    rejecting keeps the lesion, and also removes the attached vessel from its measurement —
    which is exactly why that lesion had been measured 44% too long.

    A bifurcation is almost all vessel, so little is left after the trim and it drops out.
    The floor after trimming is Daniel's 4mm, not the density pass's 12mm: that 12mm exists
    to keep the raw pass quiet, and it has already been applied before this step.
    """
    kept = np.zeros_like(labels)
    next_id = 0
    for label, box in enumerate(ndimage.find_objects(labels), 1):
        if box is None:
            continue
        mask = labels[box] == label
        rest = mask & ~vessels_wide[box]
        if rest.sum() < min_keep_fraction * mask.sum():
            continue  # mostly vessel: a bifurcation, not a lesion
        pieces, n = ndimage.label(rest)
        if n > 1:  # the trim can split it; the lesion is the biggest piece
            rest = pieces == (np.bincount(pieces.ravel())[1:].argmax() + 1)
        if axial_axes(rest, spacing)[0] < min_diameter_mm:
            continue
        next_id += 1
        kept[box][rest] = next_id
    return kept


def faint_candidates(
    ct: np.ndarray,
    lung: np.ndarray,
    vessels_wide: np.ndarray,
    spacing,
    min_diameter_mm: float = FAINT_MIN_DIAMETER_MM,
    max_linearity: float = FAINT_MAX_LINEARITY,
) -> tuple[np.ndarray, int]:
    """Small, hazy, roughly round blobs inside the lung that are not part of a vessel."""
    band = lung & (ct >= FAINT_LO) & (ct <= FAINT_HI)
    # In-plane opening only. A 4mm nodule spans about three 1.25mm slices, and a 3D opening
    # needs three slices just to survive the erosion — it would delete exactly the smallest
    # nodules this pass exists for. In-plane opening still strips the one-pixel
    # partial-volume rim that paints this band around every vessel.
    band = ndimage.binary_opening(band, np.ones((1, 3, 3), bool))

    labels, n = ndimage.label(band)
    if n == 0:
        return labels, 0

    voxel_mm3 = float(np.prod(spacing))
    cap = (4 / 3) * np.pi * (FAINT_MAX_DIAMETER_MM / 2) ** 3 / voxel_mm3
    counts = np.bincount(labels.ravel())

    kept = np.zeros_like(labels)
    next_id = 0
    for label, box in enumerate(ndimage.find_objects(labels), 1):
        if box is None or counts[label] < 4 or counts[label] > cap:
            continue
        mask = labels[box] == label
        if (mask & vessels_wide[box]).sum() / mask.sum() > VESSEL_MAX_FRACTION:
            continue
        if _linearity(mask, spacing) > max_linearity:
            continue
        long_axis = axial_axes(mask, spacing)[0]
        if not min_diameter_mm <= long_axis <= FAINT_MAX_DIAMETER_MM:
            continue
        next_id += 1
        kept[box][mask] = next_id
    return kept, next_id
