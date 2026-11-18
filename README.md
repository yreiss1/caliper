# Caliper

Review-first lung tumour measurement for the Thoracic Oncology Research Lab.
**Research use only. Not for clinical decisions.**

**Live app:** https://caliper-review-e9ae211a.base44.app (built on Base44) · **Results:** [RESULTS.md](RESULTS.md)

A researcher's scan takes about 40 minutes, more than half of it spent finding tumours before any measuring starts. Caliper pre-processes the scan and hands over a ranked worklist of candidates, already outlined and measured in the lab's own 18 sheet columns, to keep or dismiss with one keystroke.

```
CT series → lungs + lobes → trained nodule model ─┐
          → vessel tree   → density pass, vessels ─┤→ outlines grown 5mm → ranked worklist → review app → lab's sheet
                            trimmed out            ┘
prior scan → lung-masked registration → Match to prior
```

## Run it

Python 3.13, CPU is enough (a scan's first pass takes 5–10 minutes; later runs reuse cached segmentations and take seconds).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

**1. Get data** — RIDER Lung CT from TCIA's public API, no login. Picks the CT series each expert contour references.

```bash
.venv/bin/python -m caliper.tcia --survey                       # tumour sizes from contours only, ~10MB
.venv/bin/python -m caliper.tcia --patient RIDER-1225316081     # one patient, test + retest
```

**2. Process a scan** — writes `worklist.json`, `sheet.csv` and `labels.npy` under the series folder.

```bash
.venv/bin/python -m caliper.pipeline data/rider/RIDER-1225316081/test
.venv/bin/python -m caliper.pipeline data/rider/RIDER-1225316081/test --faint   # experimental ground-glass pass
```

To run every RIDER patient that has an expert contour (about 20 minutes a scan on CPU the first time):

```bash
for p in $(.venv/bin/python -c "import json; print(' '.join(sorted({s['PatientID'] for s in json.load(open('data/rider/series_index.json')) if s.get('SeriesDescription')=='TEST'})))"); do
  .venv/bin/python -m caliper.tcia --patient $p --out data/rider
  .venv/bin/python -m caliper.pipeline data/rider/$p/test
  .venv/bin/python -m caliper.truth data/rider/$p/test
done
```

`data/rider/series_index.json` is written by the first `caliper.tcia` run in step 1.

**3. Score against the expert contour, and match to the prior scan**

```bash
.venv/bin/python -m caliper.truth data/rider/RIDER-1225316081/test
.venv/bin/python -m caliper.match data/rider/RIDER-1225316081/test data/rider/RIDER-1225316081/retest
```

**4. Faint-nodule evaluation on LIDC-IDRI**

```bash
.venv/bin/python -m caliper.lidc select --n 6
.venv/bin/python -m caliper.lidc fetch
for d in data/lidc/LIDC-IDRI-*/scan; do .venv/bin/python -m caliper.pipeline $d --faint; done
.venv/bin/python -m caliper.lidc evaluate
```

**5. Review UI** — a static version for local use; the hosted app is the Base44 one above.

```bash
.venv/bin/python -m caliper.review data/rider --out review
python3 -m http.server 8765 --directory review
```

**Checks**

```bash
.venv/bin/python tests/test_measure.py && .venv/bin/python tests/test_density.py && .venv/bin/python tests/test_faint.py && .venv/bin/python tests/test_outline.py
```

## Where things live

| File | Does |
|---|---|
| `caliper/measure.py` | Every number the lab publishes: long and short axis on the same slice, volume, solid component, the 18-column row |
| `caliper/pipeline.py` | Lung and lobe segmentation, nodule model, density pass, merging and ranking |
| `caliper/outline.py` | Grows each candidate's outline up to 5mm into tumour-density tissue; the model under-draws large tumours |
| `caliper/faint.py` | Vessel tree, vessel trimming, experimental ground-glass pass |
| `caliper/match.py` | `Match to prior` via lung-masked rigid registration |
| `caliper/truth.py` | Scoring against the RIDER expert contours |
| `caliper/lidc.py` | Faint-nodule recall and false marks on LIDC-IDRI |
| `caliper/tcia.py` | Downloads from TCIA's public API |
| `caliper/review.py` | Static review UI |

## Data

- **RIDER Lung CT** — Zhao, B., Schwartz, L. H., & Kris, M. G. (2015). *Data From RIDER_Lung CT*. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.U1X8A5NR (CC BY 4.0)
- **RIDER-LungCT-Seg** — expert gross tumour volume contours on the same patients. https://doi.org/10.7937/TCIA.2019.x1kjhvbp
- **LIDC-IDRI** — Armato et al. (2015). The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX (CC BY 3.0)

`data/` is gitignored. Nothing here contains the lab's data; everything runs on public scans.

## Models

Off the shelf, nothing trained here: **TotalSegmentator** (Apache-2.0) for lobes, nodules and vessels; **pylidc** for LIDC annotations. MONAI's `lung_nodule_ct_detection` was tried for faint nodules and needs a GPU (see RESULTS.md).
