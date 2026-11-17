"""Check the density-based candidate detector on a synthetic chest.

Four things have to hold, and each is a way the detector could be useless: a faint round
lesion is found, a solid one is found, a vessel-like tube is not, and a flat patch is kept
even though it looks elongated by the naive measure.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caliper.pipeline import DENSITY_HI, DENSITY_LO, density_candidates, merge_labels

SPACING = (1.25, 0.7, 0.7)
SHAPE = (80, 160, 160)
PARENCHYMA = -820.0  # normal lung, below the window
GROUND_GLASS = -500.0  # faint subsolid lesion
SOLID = 50.0  # soft-tissue density, like the nodule the trained model missed
#: Pinned here so this check tests the logic, not whatever the tuned default is today.
MIN_MM = 8.0


def chest():
    return np.full(SHAPE, PARENCHYMA), np.ones(SHAPE, bool)


def sphere_at(centre, diameter_mm):
    z, y, x = np.indices(SHAPE).astype(float)
    z = (z - centre[0]) * SPACING[0]
    y = (y - centre[1]) * SPACING[1]
    x = (x - centre[2]) * SPACING[2]
    return (z**2 + y**2 + x**2) <= (diameter_mm / 2) ** 2


def demo():
    assert DENSITY_LO < GROUND_GLASS < SOLID < DENSITY_HI

    # 1. A faint 10mm round lesion is found, exactly once.
    ct, lung = chest()
    lesion = sphere_at((40, 80, 80), 10.0)
    ct[lesion] = GROUND_GLASS
    labels, n = density_candidates(ct, lung, SPACING, MIN_MM)
    assert n == 1, f"expected one candidate, got {n}"
    assert ((labels > 0) & lesion).sum() / lesion.sum() > 0.5, "candidate barely covers it"

    # 2. A solid lesion of the same size is found too. The window has to span ground-glass
    #    through solid: on RIDER-1801720707 the lesion the trained model missed has a core
    #    of +56 HU, so a ground-glass-only band could never have caught it.
    ct, lung = chest()
    ct[sphere_at((40, 80, 80), 10.0)] = SOLID
    _, n = density_candidates(ct, lung, SPACING, MIN_MM)
    assert n == 1, f"solid lesion missed ({n} candidates)"

    # 3. A long thin tube is rejected as vessel-shaped. Every vessel in the lung sits in
    #    this window, so without this filter the worklist is unusable.
    ct, lung = chest()
    ct[20:70, 78:82, 78:82] = SOLID
    _, n = density_candidates(ct, lung, SPACING, MIN_MM)
    assert n == 0, f"vessel-shaped structure was kept ({n} candidates)"

    # 4. A flat patch — 12mm across, three slices thick, the real shape on
    #    RIDER-1801720707 — must survive. A longest-over-shortest elongation filter would
    #    silently discard it, which is the regression worth guarding.
    ct, lung = chest()
    ct[39:42, 72:89, 72:89] = GROUND_GLASS
    _, n = density_candidates(ct, lung, SPACING, MIN_MM)
    assert n == 1, f"flat patch was rejected ({n} candidates)"

    # 5. Parenchyma alone yields nothing, and nothing outside the lung mask counts.
    ct, lung = chest()
    _, n = density_candidates(ct, lung, SPACING, MIN_MM)
    assert n == 0, "found something in empty lung"
    ct[sphere_at((40, 80, 80), 10.0)] = SOLID
    _, n = density_candidates(ct, np.zeros(SHAPE, bool), SPACING, MIN_MM)
    assert n == 0, "candidate found outside the lungs"

    # 6. Merging: a density candidate on top of a model candidate is dropped, not doubled.
    model = sphere_at((40, 80, 80), 10.0)
    density = np.zeros(SHAPE, int)
    density[sphere_at((40, 80, 80), 8.0)] = 1  # same lesion
    density[sphere_at((20, 40, 40), 8.0)] = 2  # elsewhere
    _, sources = merge_labels(model, [(density, "density")])
    assert sorted(sources.values()) == ["density", "model"], sources

    print("density detector: finds faint and solid blobs, rejects tubes, keeps flat patches")


if __name__ == "__main__":
    demo()


def test_density():
    demo()
