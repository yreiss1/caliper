"""The expert's contour, and how far the tool is from it.

The ground truth for a RIDER series is the `GTVp_*_man` ROI inside that series' RTSTRUCT —
delineated by a radiation oncologist blinded to earlier contours. `GTVp_*_auto` is the
dataset authors' own semi-automatic contour, kept here as a second comparison point.

Note what this file does NOT do: it does not score false positives. The expert contoured
the gross tumour volume only, so a region with no contour is not evidence of no lesion.
Detection has to be scored on a dataset that annotates nodules exhaustively.
"""

from __future__ import annotations

import numpy as np

MANUAL_SUFFIX = "_man"
#: Tumour tissue sits near water density. If a loaded mask averages colder than this, it
#: is sitting in air, which means the slice axis came back in the wrong order.
MIN_PLAUSIBLE_HU = -400.0


def load_roi(rtstruct_dir, dicom_dir, ct_hu: np.ndarray, name_contains: str = MANUAL_SUFFIX):
    """One ROI from an RTSTRUCT, as a (z, y, x) bool array on the CT's own grid.

    rt-utils returns (y, x, slice) and does not promise the slice axis runs the same way
    SimpleITK stacked the series. Rather than trust either, check the HU under the mask: a
    correctly aligned tumour mask covers soft tissue, a flipped one covers air. Cheap, and
    it fails loudly instead of producing a plausible-looking wrong Dice.
    """
    from rt_utils import RTStructBuilder

    rt = RTStructBuilder.create_from(
        dicom_series_path=str(dicom_dir), rt_struct_path=str(next(rtstruct_dir.glob("*.dcm")))
    )
    names = [n for n in rt.get_roi_names() if name_contains in n]
    if not names:
        raise SystemExit(f"no ROI matching {name_contains!r} in {rt.get_roi_names()}")

    mask = np.transpose(rt.get_roi_mask_by_name(names[0]), (2, 0, 1)).astype(bool)
    if mask.shape != ct_hu.shape:
        raise SystemExit(f"ROI shape {mask.shape} != CT shape {ct_hu.shape}")

    if mask.any() and ct_hu[mask].mean() < MIN_PLAUSIBLE_HU:
        flipped = mask[::-1]
        if flipped.any() and ct_hu[flipped].mean() >= MIN_PLAUSIBLE_HU:
            mask = flipped
        else:
            raise SystemExit(
                f"{names[0]} lands in air either way "
                f"({ct_hu[mask].mean():.0f} HU) — alignment is wrong, not just reversed"
            )
    return names[0], mask


def main() -> None:
    """Score one processed series against the expert contour.

        .venv/bin/python -m caliper.truth data/rider/RIDER-1129164940/test

    Needs the detection pass to have run first (it reads `worklist.json` and `labels.npy`).
    """
    import argparse
    import json
    from pathlib import Path

    import SimpleITK as sitk

    from caliper.measure import axial_axes
    from caliper.pipeline import load_series

    ap = argparse.ArgumentParser(description="Compare candidates against the expert contour.")
    ap.add_argument("series_dir", type=Path)
    args = ap.parse_args()

    series_dir = args.series_dir
    work = series_dir / "caliper"
    image = load_series(series_dir)
    ct = sitk.GetArrayFromImage(image)
    spacing = tuple(reversed(image.GetSpacing()))

    rt_dir = series_dir.parent / f"rtstruct_{series_dir.name}"
    name, truth = load_roi(rt_dir, series_dir, ct, MANUAL_SUFFIX)

    found = json.loads((work / "worklist.json").read_text())
    labels = np.load(work / "labels.npy")
    result = compare(found, labels, truth, spacing)
    result["roi"] = name
    result["candidates"] = len(found)

    # The dataset ships a second contour of the same tumour, by the same authors. The gap
    # between their two contours is a quality ceiling I did not have to measure myself.
    try:
        auto_name, auto = load_roi(rt_dir, series_dir, ct, "_auto")
        auto_long = axial_axes(auto, spacing)[0]
        result["reference_spread"] = {
            "between": [name, auto_name],
            "dice": round(dice(truth, auto), 3),
            "long_axis_diff_mm": round(auto_long - result["truth_long_axis_mm"], 1),
            "long_axis_diff_pct": round(
                100 * (auto_long - result["truth_long_axis_mm"]) / result["truth_long_axis_mm"], 1
            ),
        }
    except SystemExit:
        pass

    (work / "evaluation.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def dice(a: np.ndarray, b: np.ndarray) -> float:
    total = a.sum() + b.sum()
    return float(2 * (a & b).sum() / total) if total else 0.0


def compare(candidates: list[dict], labels: np.ndarray, truth: np.ndarray, spacing) -> dict:
    """Match the tool's candidates against the expert contour.

    `labels` is the labelled candidate volume from the detection pass, so a candidate's own
    voxels can be recovered by id. The matched candidate is the one overlapping the truth
    most — not the nearest centroid, which picks the wrong lesion when two sit close.
    """
    from caliper.measure import axial_axes

    best, best_overlap = None, 0
    for c in candidates:
        mask = labels == c["label"]
        overlap = int((mask & truth).sum())
        if overlap > best_overlap:
            best, best_overlap = c, overlap

    truth_long, truth_short, _ = axial_axes(truth, spacing)
    if best is None:
        return {
            "detected": False,
            "dice": 0.0,
            "truth_long_axis_mm": round(truth_long, 1),
            "truth_short_axis_mm": round(truth_short, 1),
        }

    return {
        "detected": True,
        "nodule_id": best["nodule_id"],
        "dice": round(dice(labels == best["label"], truth), 3),
        "truth_long_axis_mm": round(truth_long, 1),
        "truth_short_axis_mm": round(truth_short, 1),
        "tool_long_axis_mm": best["long_axis_mm"],
        "tool_short_axis_mm": best["short_axis_mm"],
        "long_axis_error_mm": round(best["long_axis_mm"] - truth_long, 1),
        # The number that matters for their research: a percentage error on the diameter
        # they publish, not a Dice score they have never used.
        "long_axis_error_pct": round(100 * (best["long_axis_mm"] - truth_long) / truth_long, 1)
        if truth_long
        else None,
    }


if __name__ == "__main__":
    main()
