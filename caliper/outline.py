"""Grow an under-drawn outline to the whole tumour.

The problem, measured: on RIDER-1129164940 the tool's long axis is 20% short of the
expert's (54.8 vs 68.8mm) and its volume 30% short (43.8 vs 62.2 cm3).

My first diagnosis was that the outline stopped at the edge of the aerated lung, so the
first version of this let it grow past that edge into collapsed lung. Measuring where the
expert's outlines actually sit disproved it: on all four RIDER patients 91-100% of the
expert's tumour lies inside the tool's own lung mask, and 100% within 5mm of it. The
nodule model simply under-draws large tumours from the inside. So this grows the outline
into connected tumour-density tissue, under limits that each stop a different leak:

- **Within the lung mask plus 5mm.** 5mm covers every expert outline measured, and keeps
  growth from running into the chest wall, which has the same density as tumour.
- **Not into vessels.** Hilar vessels are soft tissue too; growing along them would
  follow the vessel tree to the heart.
- **At most 5mm from the original outline.** Enough to restore the rim the model leaves
  off; growth further out mostly added collapsed lung and overshot.

ponytail: density-and-geometry growth, no model. Where tumour touches collapsed lung of the
same density, nothing here can tell them apart; the expert's own manual and semi-automatic
contours disagree most in exactly those cases.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

#: Tumour and soft tissue. Below this is aerated lung; above, bone.
#: Tuned on 14 RIDER patients, checked on 13 held out: median long-axis error vs the
#: expert went from 15.7% to 7.3%, median Dice 0.76 to 0.77. -150 tied with -50 on tuning.
GROW_LO, GROW_HI = -50.0, 250.0
GROW_MAX_MM = 5.0
#: How far outside the lung mask growth may reach. Measured: covers 100% of four expert
#: outlines; the lung mask alone covers 91-100%.
LUNG_MARGIN_MM = 5.0


def envelope(lungs: np.ndarray, spacing) -> np.ndarray:
    """The lung mask widened by LUNG_MARGIN_MM."""
    return ndimage.distance_transform_edt(~lungs, sampling=spacing) <= LUNG_MARGIN_MM


def refine(mask: np.ndarray, ct: np.ndarray, envelope_mask: np.ndarray,
           vessels: np.ndarray | None, spacing) -> np.ndarray:
    """Grow one candidate into connected tumour-density tissue, within the limits above.

    All arrays share one grid; work happens in a box around the candidate, padded by the
    growth limit, so a large scan costs no more than a small one.
    """
    if not mask.any():
        return mask
    pad = [int(np.ceil(GROW_MAX_MM / s)) + 1 for s in spacing]
    idx = np.nonzero(mask)
    box = tuple(slice(max(0, int(i.min()) - p), min(n, int(i.max()) + p + 1))
                for i, p, n in zip(idx, pad, mask.shape))

    seed = mask[box]
    allowed = envelope_mask[box] & (ct[box] >= GROW_LO) & (ct[box] <= GROW_HI)
    if vessels is not None:
        allowed &= ~vessels[box]
    allowed |= seed
    allowed &= ndimage.distance_transform_edt(~seed, sampling=spacing) <= GROW_MAX_MM

    labels, _ = ndimage.label(allowed)
    keep = np.unique(labels[seed])
    keep = keep[keep > 0]

    grown = mask.copy()
    grown[box] |= np.isin(labels, keep)
    return grown


def refine_all(labels: np.ndarray, ct: np.ndarray, lungs: np.ndarray,
               vessels: np.ndarray | None, spacing) -> np.ndarray:
    """refine() every candidate; a candidate only grows into voxels no other one owns."""
    env = envelope(lungs, spacing)
    out = labels.copy()
    for lab, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        grown = refine(labels == lab, ct, env, vessels, spacing)
        out[grown & (out == 0)] = lab
    return out
