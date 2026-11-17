"""Fill `Match to prior`: carry each lesion from the earlier scan into the later one.

Daniel's third hard case is re-finding a tumour on a follow-up scan after breathing has
shifted everything, and `Match to prior` is a column on his sheet. Both are the same
problem: register the prior scan onto the current one, move each prior candidate's centre
across, and pair it with the current candidate sitting there.

RIDER's test/retest pairs make this testable for free: same patient, same day, different
breath hold, and the expert outlined the tumour on both. If the match is right, the expert's
tumour in the test scan must pair with the expert's tumour in the retest scan.

    .venv/bin/python -m caliper.match data/rider/RIDER-1129164940/test data/rider/RIDER-1129164940/retest
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from caliper.measure import SHEET_COLUMNS
from caliper.pipeline import LOBES, array, load_series

#: A prior lesion pairs with a current candidate only if its moved centre lands this close.
#: Generous on purpose: rigid registration leaves a few millimetres of breathing residual,
#: and an unmatched real lesion costs Daniel a hunt, which is the thing we are removing.
MATCH_MM = 15.0


def lung_mask(series_dir: Path, image: sitk.Image) -> sitk.Image:
    """The lobes from the detection pass, as a mask on this scan's grid."""
    lungs = None
    for name in LOBES:
        path = series_dir / "caliper" / "lobes" / f"{name}.nii.gz"
        if path.exists():
            part = array(path).astype(bool)
            lungs = part if lungs is None else lungs | part
    mask = sitk.GetImageFromArray(lungs.astype(np.uint8))
    mask.CopyInformation(image)
    return sitk.BinaryDilate(mask, [3, 3, 3])


def register(prior: sitk.Image, current: sitk.Image, prior_lungs: sitk.Image | None = None) -> sitk.Transform:
    """Rigid transform taking points in the prior scan to the same anatomy in the current one.

    ponytail: rigid only. Same-day scans differ by breath hold, which is mostly a shift; a
    year-apart follow-up with weight change or atelectasis will need deformable (BSpline)
    on top. Measure the residual on real follow-ups before adding it.
    """
    # Only the lungs are allowed to vote. Registering the whole scan let the body wall and
    # the scanner table win: on RIDER-2799584460 it moved the tumour from 18mm off its
    # retest position to 37mm off — worse than not registering at all.
    # Clamp to air..soft tissue so bone doesn't dominate the metric either.
    fixed = sitk.Cast(sitk.Clamp(prior, lowerBound=-1000, upperBound=400), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.Clamp(current, lowerBound=-1000, upperBound=400), sitk.sitkFloat32)

    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS
    )
    reg = sitk.ImageRegistrationMethod()
    if prior_lungs is not None:
        reg.SetMetricFixedMask(prior_lungs)
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.05, seed=7)  # fixed seed: same scans, same answer
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-3, numberOfIterations=200, relaxationFactor=0.5
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial, inPlace=False)
    return reg.Execute(fixed, moving)


def to_physical(image: sitk.Image, centroid_zyx) -> np.ndarray:
    z, y, x = centroid_zyx
    return np.array(image.TransformContinuousIndexToPhysicalPoint((x, y, z)))


def match(prior_dir: Path, current_dir: Path) -> dict:
    prior_img, current_img = load_series(prior_dir), load_series(current_dir)
    prior = json.loads((prior_dir / "caliper" / "worklist.json").read_text())
    current = json.loads((current_dir / "caliper" / "worklist.json").read_text())

    transform = register(prior_img, current_img, lung_mask(prior_dir, prior_img))
    current_pts = {c["nodule_id"]: to_physical(current_img, c["centroid_zyx"]) for c in current}

    # Greedy, closest pairs first, each current candidate used once. Lists are tens of
    # items, so an optimal assignment would buy nothing worth the extra code.
    pairs = []
    for p in prior:
        moved = np.array(transform.TransformPoint(tuple(to_physical(prior_img, p["centroid_zyx"]))))
        for cid, pt in current_pts.items():
            pairs.append((float(np.linalg.norm(moved - pt)), p["nodule_id"], cid))
    pairs.sort()

    matched, used_prior, used_current = {}, set(), set()
    for dist, pid, cid in pairs:
        if dist > MATCH_MM:
            break
        if pid in used_prior or cid in used_current:
            continue
        matched[cid] = {"prior": pid, "distance_mm": round(dist, 1)}
        used_prior.add(pid)
        used_current.add(cid)

    label = prior_dir.name
    for c in current:
        m = matched.get(c["nodule_id"])
        c["match_to_prior"] = f"{label}/{m['prior']}" if m else ""
        c["match_distance_mm"] = m["distance_mm"] if m else None
    (current_dir / "caliper" / "worklist.json").write_text(json.dumps(current, indent=1))

    sheet = current_dir / "caliper" / "sheet.csv"
    if sheet.exists():
        rows = list(csv.DictReader(sheet.open()))
        by_id = {c["nodule_id"]: c for c in current}
        for row in rows:
            row["Match to prior"] = by_id.get(row["Nodule ID"], {}).get("match_to_prior", "")
        with sheet.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    return {
        "prior_candidates": len(prior),
        "current_candidates": len(current),
        "matched": len(matched),
        "pairs": matched,
    }


def lesion_ids(series_dir: Path) -> str | None:
    """The candidate that overlaps the expert's contour, from the evaluation already run."""
    path = series_dir / "caliper" / "evaluation.json"
    return json.loads(path.read_text()).get("nodule_id") if path.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Match candidates from a prior scan into a current one.")
    ap.add_argument("prior", type=Path)
    ap.add_argument("current", type=Path)
    args = ap.parse_args()

    result = match(args.prior, args.current)
    print(
        f"{result['matched']} of {result['prior_candidates']} prior candidates matched "
        f"into {result['current_candidates']} current"
    )

    # The check that matters: does the expert's tumour in the prior scan pair with the
    # expert's tumour in the current one?
    prior_lesion, current_lesion = lesion_ids(args.prior), lesion_ids(args.current)
    if prior_lesion and current_lesion:
        got = result["pairs"].get(current_lesion, {})
        ok = got.get("prior") == prior_lesion
        print(
            f"expert lesion: {args.prior.name}/{prior_lesion} -> {args.current.name}/{current_lesion}  "
            f"{'MATCHED' if ok else 'NOT MATCHED'}"
            + (f" ({got['distance_mm']} mm)" if ok else f" (paired with {got.get('prior') or 'nothing'})")
        )


if __name__ == "__main__":
    main()
