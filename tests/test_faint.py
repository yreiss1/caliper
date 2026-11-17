"""Check the faint pass and vessel filter on a synthetic chest.

The cases are the walkthrough's: a 5mm hazy nodule must be found, a candidate that is
mostly vessel must not be, and a real nodule that merely touches a vessel must survive.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caliper.faint import faint_candidates, trim_vessels, widen

SPACING = (1.25, 0.7, 0.7)
SHAPE = (60, 120, 120)
PARENCHYMA = -820.0
GROUND_GLASS = -550.0


def sphere_at(centre, diameter_mm):
    z, y, x = np.indices(SHAPE).astype(float)
    z = (z - centre[0]) * SPACING[0]
    y = (y - centre[1]) * SPACING[1]
    x = (x - centre[2]) * SPACING[2]
    return (z**2 + y**2 + x**2) <= (diameter_mm / 2) ** 2


def chest():
    return np.full(SHAPE, PARENCHYMA), np.ones(SHAPE, bool), np.zeros(SHAPE, bool)


def demo():
    # 1. A faint 5mm nodule, Daniel's walkthrough case, is found.
    ct, lung, vessels = chest()
    ct[sphere_at((30, 60, 60), 5.0)] = GROUND_GLASS
    _, n = faint_candidates(ct, lung, widen(vessels), SPACING)
    assert n == 1, f"5mm faint nodule missed ({n})"

    # 2. A 3mm one is not: below the 4mm he records, so reporting it is noise by his rules.
    ct, lung, vessels = chest()
    ct[sphere_at((30, 60, 60), 3.0)] = GROUND_GLASS
    _, n = faint_candidates(ct, lung, widen(vessels), SPACING)
    assert n == 0, f"sub-4mm blob reported ({n})"

    # 3. The same faint blob sitting on the vessel tree — a bifurcation — is dropped.
    ct, lung, vessels = chest()
    blob = sphere_at((30, 60, 60), 6.0)
    ct[blob] = GROUND_GLASS
    vessels[blob] = True
    _, n = faint_candidates(ct, lung, widen(vessels), SPACING)
    assert n == 0, f"vessel bifurcation kept ({n})"

    # 4. A real nodule that only touches a vessel survives. Dropping these would trade one
    #    kind of miss for another.
    ct, lung, vessels = chest()
    ct[sphere_at((30, 60, 60), 6.0)] = GROUND_GLASS
    vessels[25:35, 60, 66:70] = True  # thin vessel grazing the nodule's edge
    _, n = faint_candidates(ct, lung, widen(vessels), SPACING)
    assert n == 1, f"vessel-adjacent nodule dropped ({n})"

    # 5. Trimming another detector's output. A blob that is all vessel goes; a lesion with a
    #    vessel attached keeps the lesion and loses the vessel — the RIDER-1801720707 case.
    labels = np.zeros(SHAPE, int)
    all_vessel = sphere_at((30, 90, 90), 8.0)
    labels[all_vessel] = 1
    lesion = sphere_at((30, 30, 30), 8.0)
    attached = np.zeros(SHAPE, bool)
    attached[28:33, 30, 34:60] = True  # vessel running off the lesion
    labels[lesion | attached] = 2
    vessels = np.zeros(SHAPE, bool)
    vessels[all_vessel] = True
    vessels[attached & ~lesion] = True
    kept = trim_vessels(labels, widen(vessels), SPACING)
    assert kept.max() == 1, f"expected only the lesion to survive, got {kept.max()}"
    assert (kept > 0)[lesion].mean() > 0.8, "lesion was trimmed away"
    assert not (kept > 0)[all_vessel].any(), "pure-vessel blob survived"

    print("faint pass: finds 5mm hazy nodules, drops sub-4mm and vessels, keeps vessel-adjacent")


if __name__ == "__main__":
    demo()


def test_faint():
    demo()
