# Caliper

Review-first lung tumour measurement for the Thoracic Oncology Research Lab.

**Research use only. Not for clinical decisions.**

## What it does

A researcher's current scan costs 25–40 minutes, 10–20 of which is spent *finding* the
tumours before any outlining starts. Caliper inverts that: it pre-processes a scan, then
hands over a ranked worklist of candidate lesions, each already outlined and measured, to
accept, reject or correct.

```
DICOM series → lung + lobe segmentation
  → nodule candidates (tunable sensitivity)   ──┐
  → prior timepoint registered, its lesions   ──┤→ worklist → review → 18-column sheet + DICOM SEG
    propagated forward as candidates          ──┘
```

Tuned for recall, deliberately. The lab's stated tolerance is 5–10 false marks per scan in
exchange for not missing a small, faint nodule, so dismissing a candidate is one keystroke
and the sensitivity threshold is a control in the UI rather than a constant in the code.

## Status

| Piece | State |
|---|---|
| Lesion geometry + the lab's sheet format | done, `caliper/measure.py`, self-checked |
| TCIA fetch | done, `caliper/tcia.py` |
| Lung / lobe / nodule segmentation | next |
| Prior-timepoint registration | next |
| Review UI | next |
| Validation vs expert contours | next |

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python caliper/tcia.py --patients 1     # fetch a test/retest pair
.venv/bin/python tests/test_measure.py            # geometry self-check
```

## Data

- **RIDER Lung CT** — Zhao, B., Schwartz, L. H., & Kris, M. G. (2015). *Data From RIDER_Lung CT*.
  The Cancer Imaging Archive. <https://doi.org/10.7937/K9/TCIA.2015.U1X8A5NR> (CC BY 4.0)
- **RIDER-LungCT-Seg** — expert gross tumour volume delineations on the same patients.
  <https://doi.org/10.7937/TCIA.2019.x1kjhvbp>

Fetched over the public NBIA REST API rather than the desktop Data Retriever, so the
download is scriptable and the exact series used are recorded in `data/rider/*/series.json`.

`data/` is gitignored. Nothing in this repository contains patient data from the lab, and
the hosted pilot runs on public TCIA data only.

## Models

Off-the-shelf, nothing trained here.

- **TotalSegmentator** (Apache-2.0) — `total` for lung lobes, `lung_nodules` for candidate
  detection with `--save_probabilities` giving the sensitivity knob.
- **MONAI VISTA3D** — point-prompted segmentation, for repairing a candidate the detector
  got wrong or missed entirely.
- **SimpleITK** — registration between timepoints.

## Measurements

`caliper/measure.py` is the only place numbers are produced, and it is a pure function of
the mask and the image so it can be tested without a scan. Long axis is the largest
in-plane diameter across slices; short axis is the perpendicular width **in that same
slice**, since taking it from elsewhere would inflate it. Spacing is always read from the
DICOM — RIDER slice thickness is several times the in-plane pixel size, and ignoring that
would be wrong by a factor of three.

Every row records the click, the model version and the mask that produced it, so any
published number can be regenerated. `Measured by` stays empty until a human accepts the
row: the tool pre-fills, it does not sign.
