# Caliper — Results

Lung tumour measurement tool for the Thoracic Oncology Research Lab (simulated), Base44 FDE assignment, Part 1.
**Research use only. Not for clinical decisions.**

## The problem, as the lab described it

A researcher reads each CT slice by slice, finds every tumour, outlines and measures it, and types it into an 18-column sheet. My first framing was that outlining was the bottleneck. In their first reply both contacts corrected me: **finding** the tumours is as slow as outlining them. That reshaped the tool from "click a tumour, get a measurement" into a review-first worklist: the tool pre-processes a scan and hands over candidates already outlined and measured, to keep or dismiss.

**Baseline.** Daniel narrated one scan live: 29 minutes, 22 of them finding (8 on a first pass, 14 on a slow second pass for small faint nodules, 3 of those on a single 5mm ground-glass nodule). His own minutes log of 20 baseline scans averages **40.4 minutes** (median 37.5, range 29–61), which we agreed as the baseline.

## What works

| | Result | Data |
|---|---|---|
| Finds the main tumour | 27 / 30 scans | RIDER, all 30 test scans with an expert outline |
| Long-axis diameter vs the expert | median error 8.0%; 16 of 27 within 10%, 20 within 20%. The expert's own two outlines differ by a median 6.7% | RIDER, 30 test scans |
| Outline overlap with the expert (Dice) | median 0.79 | RIDER, 30 test scans |
| Repeatability, same tumour on two same-day scans | tool moves ≤3% on 3 of 4 tumours; the expert's own re-reads move 2–12% | RIDER test/retest |
| Matches a tumour to the earlier scan (`Match to prior`) | 4 / 4, within 0.9–9.5mm | RIDER test → retest |
| Fills the lab's sheet | all 18 columns, their names and order; `Measured by` is left for the human | — |
| Dismissing a false mark | about 10 seconds, one keystroke | Daniel, in the app |

## What doesn't

**Large tumours the model only half-draws.** 4 of 27 read 43–51% short (Dice 0.26–0.53). The outline fix grows each outline up to 5mm into tumour-density tissue; that halved median error (15.6% to 8.0%, settings chosen on 14 patients and checked on the 13 held out: 15.7% to 7.3%), but cannot fill a tumour the model mostly missed. These need edge editing in the app.

**Not found: 3 of 30.** RIDER-1286684383, -1760553574, -2541949645.

**The tumour isn't always near the top of the list.** On 4 scans it ranks 14th–26th of 15–35 candidates, so the reader still goes through the whole worklist.

**Faint nodules — the second pass is not replaced.** Tested on 6 LIDC-IDRI scans with 7 faint nodules marked by at least two of four radiologists:

| Approach | Faint nodules found | False marks per scan |
|---|---|---|
| Trained nodule model (TotalSegmentator `lung_nodules`) | 0 / 7, at any probability threshold | ~1–2 |
| Current tool (model + density pass) | 1 / 7 | 9.5 |
| Dedicated ground-glass pass | **6 / 7** | **215** |
| Ground-glass pass ranked by local contrast | top-10 keeps 1 / 5 | — |
| Ground-glass pass ranked by model probability | top-5 keeps 2 / 5 | — |

The faint nodules are findable, but nothing tested can separate them from normal haze and small vessels well enough for Daniel to trust the list and skip his search. The ground-glass pass is in the code behind `--faint`, off by default.

**Noise on the main list.** 4–37 candidates per scan against Daniel's stated tolerance of 5–10. Ordering by lesion-likeness puts the true tumour 1st, 1st, 4th and 12th.

## How time saved is measured

The baseline is the lab's own `Minutes on scan` log (40 min/scan). The tool keeps that column and records time per decision in the app, so the comparison lives in the lab's data rather than mine.

**Projected saving: about 10 minutes per scan (~25%)** — the first pass and most of the measuring and typing, from Daniel's own breakdown. This is a projection. Daniel is an AI persona reviewing through chat, so no timed review session in the app exists yet; that is the first thing to measure with a real reader.

Replacing the second pass as well would bring a scan toward ~10–15 minutes, but only with a validated faint-nodule detector.

## Data and validation

- **RIDER Lung CT** (CC BY 4.0) with the RIDER-LungCT-Seg expert contours: one radiation oncologist, blinded, on same-day test and retest scans. CT series are selected by the series each expert contour references, because study dates and slice counts are unreliable in this collection.
- **LIDC-IDRI**, via the pylidc annotation database: four radiologists, every nodule ≥3mm, texture rated 1–5. Used for faint-nodule recall and false marks, which RIDER cannot measure because it outlines only the primary tumour.

## Limits worth stating

Small samples: 4 RIDER patients and 7 faint LIDC nodules. Diameters are from the tool's 3D outline, not a reader's ruler. Everything runs on public data; the lab's own scans can't leave the hospital, so running on them needs an on-prem deployment discussed with their IT.

## Next phase

1. A learned faint-nodule scorer on a GPU (MONAI's LUNA16 detector is wired up and failed only on CPU speed), validated on 50+ faint nodules.
2. Edge editing for large chest-wall tumours, the slowest scans in Daniel's log.
3. A timed review session with a real reader, to replace the projection with a measurement.

## Mistakes made and fixed along the way

Read the mean HU of a lesion (−163) as ground-glass when the median (−19) showed it was solid. Filtered vessels by whole-candidate rejection and deleted a real tumour; trimming instead fixed it and a 44% over-measurement. Registered whole scans, which made one alignment worse (18mm → 37mm off); registering on the lungs alone fixed it (0.9mm). Scored a probability map that turned out to be mirrored left–right against the CT.

**App:** https://caliper-review-e9ae211a.base44.app · **Code:** https://github.com/yreiss1/caliper
