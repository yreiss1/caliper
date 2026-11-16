"""Assemble the review packet to go through with the lab.

Not a report. The point is to put real candidate lists in front of Daniel and have him
sort them, because the question that decides the next build step — are these marks junk or
real nodules nobody outlined? — cannot be answered from RIDER, which contours the primary
tumour only.

    .venv/bin/python -m caliper.packet data/rider --out packet
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from caliper.pipeline import load_series
from caliper.truth import load_roi


def lesion_rank(series_dir: Path) -> dict:
    """Where the expert's lesion sits in the candidate list, and who found it.

    Rank matters because capping the list is only honest if the real lesion is near the
    top. On the first scan I checked it ranked 13th, which is what killed the idea of
    simply showing Daniel the top ten.
    """
    work = series_dir / "caliper"
    found = json.loads((work / "worklist.json").read_text())
    labels = np.load(work / "labels.npy")

    image = load_series(series_dir)
    ct = sitk.GetArrayFromImage(image)
    _, truth = load_roi(series_dir.parent / f"rtstruct_{series_dir.name}", series_dir, ct, "_man")

    for position, candidate in enumerate(found, 1):
        if ((labels == candidate["label"]) & truth).any():
            return {
                "rank": position,
                "of": len(found),
                "source": candidate["source"],
                "nodule_id": candidate["nodule_id"],
            }
    return {"rank": None, "of": len(found), "source": None, "nodule_id": None}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the review packet for the lab.")
    ap.add_argument("root", type=Path, help="data/rider")
    ap.add_argument("--out", type=Path, default=Path("packet"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for series_dir in sorted(args.root.glob("*/test")):
        work = series_dir / "caliper"
        if not (work / "evaluation.json").exists():
            continue
        patient = series_dir.parent.name
        evaluation = json.loads((work / "evaluation.json").read_text())
        found = json.loads((work / "worklist.json").read_text())
        rank = lesion_rank(series_dir)

        shutil.copy(work / "sheet.csv", args.out / f"{patient}_candidates.csv")
        rows.append(
            {
                "Patient": patient,
                "Candidates": len(found),
                "By model": sum(1 for c in found if c["source"] == "model"),
                "By density": sum(1 for c in found if c["source"] == "density"),
                "Lesion found": "yes" if evaluation["detected"] else "no",
                "Found by": rank["source"] or "",
                "Rank in list": rank["rank"] or "",
                "Dice": evaluation["dice"],
                "Expert long axis mm": evaluation["truth_long_axis_mm"],
                "Tool long axis mm": evaluation.get("tool_long_axis_mm", ""),
                "Diameter error %": evaluation.get("long_axis_error_pct", ""),
                "Expert man vs auto Dice": evaluation.get("reference_spread", {}).get("dice", ""),
            }
        )

    if not rows:
        raise SystemExit(f"no processed series under {args.root}")

    with (args.out / "accuracy.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = "\n".join(
        f"| {r['Patient']} | {r['Candidates']} | {r['By model']} / {r['By density']} | "
        f"{r['Lesion found']} ({r['Found by']}, #{r['Rank in list']}) | {r['Dice']} | "
        f"{r['Diameter error %']}% | {r['Expert man vs auto Dice']} |"
        for r in rows
    )

    (args.out / "README.md").write_text(
        f"""# Candidate review — what I need from you

Four scans from the public RIDER practice set, each pre-processed by the tool. One CSV per
scan, in your sheet's columns and order. Every row is a candidate the tool put up; none of
them has been reviewed by anyone.

**Research use only. Not for clinical decisions.**

## The three questions

**1. Which of these marks would you want to see?** This is the one I cannot answer. RIDER
only has a contour for the primary tumour, so when the tool marks something else I have no
way to tell a false alarm from a real nodule that nobody outlined. If you go down one list
and split it into "worth a look" and "noise", that single pass tells me whether the problem
is the detector or the way the list is ordered. Those need opposite fixes.

**2. Is a 10-20% under-read on diameter disqualifying?** On three of the four scans the
tool's long axis comes in 10-20% below the expert's. On the fourth it is 44% over. For a
measurement you publish, is that correctable at review, or does it make the number
untrustworthy?

**3. Where should the sensitivity dial sit?** You said 5-10 false marks per scan would be a
huge win against 10-20 minutes of searching. Measured, the trade-off is harsher than that:

| Setting | Candidates per scan | Lesions found |
|---|---|---|
| Trained model only | 2-12 | 3 of 4 |
| Model + density, 12mm floor | 7-41 | 4 of 4 |
| Model + density, 8mm floor | 20-92 | 4 of 4 |

The fourth lesion — the one only the wider setting finds — is a flat plaque, 12.3mm across
and three slices thick. The nodule model does not fire on it because it is not
nodule-shaped. Catching it costs roughly four times the marks to dismiss. Your call, not
mine, and the dial will be in the interface.

The lists are ordered by how lesion-like a candidate looks, not by size: sorting by size
buried the real lesion 34th of 41 on one of these scans, because vessel segments are bigger
than nodules. With the current order the real lesion is 1st, 1st, 4th and 12th. If the order
feels wrong to you as you read down a list, say so — that is cheaper to fix than the
detector.

## What the tool found

| Patient | Candidates | model / density | Expert's lesion | Dice | Diameter error | Your two contours agree |
|---|---|---|---|---|---|---|
{table}

The last column is the dataset's own manual and semi-automatic contours of the same tumour,
scored against each other by the same code. It is there as a yardstick: where your own two
methods disagree most, so does the tool.

## What is not here yet

No interface — these are CSVs, not a review screen. Nothing measured on your own scans, and
nothing can be until the on-premise question is settled. No time-saved number yet, because
that needs you reviewing a real list, which is what this packet is for.
""",
        encoding="utf-8",
    )

    print(f"packet written to {args.out}/")
    for r in rows:
        print(
            f"  {r['Patient']}  {r['Candidates']:>3} candidates  "
            f"lesion #{r['Rank in list']} by {r['Found by']}  dice {r['Dice']}"
        )


if __name__ == "__main__":
    main()
