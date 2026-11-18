"""Check the outline fix on a synthetic chest: an under-drawn tumour grows to its full size,
but not into the chest wall beyond the lung margin, and not along a vessel.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caliper.outline import LUNG_MARGIN_MM, envelope, refine

SPACING = (1.25, 0.7, 0.7)
SHAPE = (40, 200, 200)
AIR, TISSUE = -850.0, 40.0


def demo():
    z, y, x = np.indices(SHAPE)
    mm = lambda cz, cy, cx: np.sqrt(((z - cz) * SPACING[0]) ** 2 + ((y - cy) * SPACING[1]) ** 2
                                    + ((x - cx) * SPACING[2]) ** 2)

    ct = np.full(SHAPE, TISSUE)                    # chest wall: soft tissue everywhere outside
    lungs = mm(20, 100, 100) < 50                  # a 50mm-radius lung
    ct[lungs] = AIR

    tumour = mm(20, 100, 140) < 12                 # 24mm tumour against the lung's edge
    ct[tumour] = TISSUE
    lungs |= tumour                                # the lobe mask includes the tumour, as measured
    seed = mm(20, 100, 140) < 8                    # the model left a 4mm rim off

    grown = refine(seed, ct, envelope(lungs, SPACING), None, SPACING)

    covered = (grown & tumour).sum() / tumour.sum()
    assert covered > 0.95, f"tumour only {covered:.0%} covered"
    outside = grown & ~envelope(lungs, SPACING)
    assert not outside.any(), "grew past the lung margin into the chest wall"
    beyond = grown & ~lungs
    reach = np.sqrt(((np.argwhere(beyond) - [20, 100, 140]) * SPACING) ** 2).sum(1).max() if beyond.any() else 0
    assert reach <= 12 + LUNG_MARGIN_MM + 2, "chest-wall leak larger than the margin"

    # A vessel leaving the tumour is soft tissue too; with the vessel mask it isn't followed.
    vessels = np.zeros(SHAPE, bool)
    tube = (np.abs(y - 100) <= 1) & (np.abs(z - 20) <= 1) & (x > 60) & (x < 123)
    ct2 = ct.copy()
    ct2[tube] = TISSUE
    vessels[tube] = True
    grown2 = refine(seed, ct2, envelope(lungs, SPACING), vessels, SPACING)
    assert not (grown2 & tube & ~tumour).any(), "grew along a vessel"

    print(f"outline fix: grows an under-drawn tumour to {covered:.0%}, stops at the lung margin and vessels")


if __name__ == "__main__":
    demo()


def test_outline():
    demo()
