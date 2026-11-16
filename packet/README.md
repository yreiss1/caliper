# Candidate review — what I need from you

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
| RIDER-1129164940 | 9 | 2 / 7 | yes (model, #1) | 0.814 | -20.4% | 0.914 |
| RIDER-1225316081 | 7 | 2 / 5 | yes (model, #1) | 0.615 | -20.6% | 0.939 |
| RIDER-1801720707 | 16 | 4 / 12 | yes (density, #12) | 0.428 | 43.6% | 0.634 |
| RIDER-2799584460 | 41 | 12 / 29 | yes (model, #4) | 0.795 | -10.1% | 0.995 |

The last column is the dataset's own manual and semi-automatic contours of the same tumour,
scored against each other by the same code. It is there as a yardstick: where your own two
methods disagree most, so does the tool.

## What is not here yet

No interface — these are CSVs, not a review screen. Nothing measured on your own scans, and
nothing can be until the on-premise question is settled. No time-saved number yet, because
that needs you reviewing a real list, which is what this packet is for.
