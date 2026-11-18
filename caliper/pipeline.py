"""Scan in, candidate worklist out. This is the pre-pass, not the review UI.

Runs as a background job: lung/lobe segmentation and nodule detection take minutes, and
nobody watches a spinner for minutes. The reviewer opens an already-processed worklist.

    .venv/bin/python caliper/pipeline.py data/rider/RIDER-1129164940/test

Writes `worklist.json` (candidates, with the mask path for each) and `sheet.csv` (the lab's
18 columns) next to the input.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from caliper.measure import SHEET_COLUMNS, axial_axes, measure_lesion, sheet_row

LOBES = [
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
]

# Candidates below this are dropped. ponytail: a precision knob, not a recall one — the
# real recall dial needs nnU-Net's probability map, and that is only worth the resampling
# work once we have measured recall at the default threshold on RIDER. Measure first.
MIN_LONG_AXIS_MM = 3.0


def load_series(dicom_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not files:
        raise SystemExit(f"no DICOM series in {dicom_dir}")
    reader.SetFileNames(files)
    reader.MetaDataDictionaryArrayUpdateOn()
    return reader.Execute()


def to_nifti(image: sitk.Image, path: Path) -> Path:
    if not path.exists():
        sitk.WriteImage(image, str(path))
    return path


#: Margin around the lung bounding box, in voxels. Enough that a lesion sitting against
#: the pleura keeps its surroundings, which is where the boundary judgement happens.
CROP_MARGIN = 8


def _bbox(mask: np.ndarray, margin: int) -> tuple[slice, slice, slice]:
    idx = np.nonzero(mask)
    return tuple(
        slice(max(0, int(i.min()) - margin), min(dim, int(i.max()) + 1 + margin))
        for i, dim in zip(idx, mask.shape)
    )


def segment(nifti: Path, out_dir: Path, device: str = "cpu") -> np.ndarray:
    """Lobes first, then nodule detection inside the lungs only.

    Order matters for speed. The lobe pass uses the 6mm model and is quick; its output
    gives a lung bounding box, and running the full-resolution nodule model on that crop
    instead of the whole chest cuts the work by roughly the volume ratio. The first run of
    this on CPU, uncropped, took over ten minutes for one scan — tolerable for a nightly
    pre-pass, far too slow to iterate against.

    Returns the nodule mask back on the full CT grid.
    """
    from totalsegmentator.python_api import totalsegmentator

    lobe_dir = out_dir / "lobes"
    if not (lobe_dir / f"{LOBES[0]}.nii.gz").exists():
        totalsegmentator(
            input=nifti, output=lobe_dir, task="total", roi_subset=LOBES, fast=True, device=device
        )

    full = sitk.ReadImage(str(nifti))
    shape = sitk.GetArrayFromImage(full).shape
    lungs = np.zeros(shape, bool)
    for name in LOBES:
        path = lobe_dir / f"{name}.nii.gz"
        if path.exists():
            lungs |= array(path).astype(bool)
    if not lungs.any():
        raise SystemExit("no lung found — wrong body part, or the lobe pass failed")

    box = _bbox(lungs, CROP_MARGIN)
    nodule_dir = out_dir / "lung_nodules"
    cropped_out = nodule_dir / "lung_nodules.nii.gz"

    if not cropped_out.exists():
        crop = sitk.GetArrayFromImage(full)[box]
        image = sitk.GetImageFromArray(crop)
        image.SetSpacing(full.GetSpacing())
        crop_path = out_dir / "ct_lung.nii.gz"
        sitk.WriteImage(image, str(crop_path))
        totalsegmentator(input=crop_path, output=nodule_dir, task="lung_nodules", device=device)

    nodules_cropped = array(cropped_out).astype(bool)
    nodules = np.zeros(shape, bool)
    if nodules_cropped.shape == shape:
        nodules = nodules_cropped  # an earlier uncropped run; reuse it as-is
    else:
        nodules[box] = nodules_cropped
    return nodules, lungs


def array(path: Path) -> np.ndarray:
    """NIfTI as a (z, y, x) array matching SimpleITK's axis order."""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def load_lobes(lobe_dir: Path, shape) -> np.ndarray:
    """All five lobes as one label volume, read once.

    Read once because the first version re-read five NIfTI files per candidate. At 92
    candidates that is 460 file reads to fill one spreadsheet column.
    """
    lobes = np.zeros(shape, np.uint8)
    for i, name in enumerate(LOBES, 1):
        path = lobe_dir / f"{name}.nii.gz"
        if path.exists():
            lobes[array(path).astype(bool)] = i
    return lobes


def lobe_of(sub_mask: np.ndarray, lobes_sub: np.ndarray) -> str:
    """Which lobe holds most of the lesion. Empty string rather than a guess if none do."""
    values = lobes_sub[sub_mask]
    if values.size == 0:
        return ""
    counts = np.bincount(values, minlength=len(LOBES) + 1)
    counts[0] = 0  # outside every lobe is not an answer
    return LOBES[int(counts.argmax()) - 1] if counts.any() else ""


# Everything in the lung denser than parenchyma and not bone-bright: ground-glass through
# solid. The band started at -750..-350, i.e. ground-glass only, on the assumption that the
# trained model covered the solid end. It doesn't: the lesion it missed on
# RIDER-1801720707 has a core of +56 HU. Width here is deliberate, and the shape filter is
# what keeps the result usable.
DENSITY_LO, DENSITY_HI = -750.0, 300.0
# 12mm, chosen by sweeping all four annotated patients and taking the cheapest setting that
# still catches every lesion. Note what it is tuned against: the trained model already finds
# 3 of the 4, so this generator exists for the single flat plaque it missed, and that lesion
# is 12.3mm. Tuning a threshold to catch one lesion is overfitting by definition — it is a
# dial in the UI, not a constant, and it needs re-fitting on LIDC-IDRI where there are
# hundreds of lesions instead of one.
DENSITY_MIN_DIAMETER_MM = 12.0
DENSITY_MAX_DIAMETER_MM = 40.0
#: Rejects tube-shaped structures. See `_linearity` for why this is the ratio of the first
#: two principal axes and not the first and last.
DENSITY_MAX_LINEARITY = 2.5


def _linearity(mask: np.ndarray, spacing) -> float:
    """How tube-like a component is: the ratio of its two largest principal extents.

    The obvious filter — longest extent over shortest — would throw away the lesions this
    detector exists to find. A ground-glass patch is often flat: the real one on
    RIDER-1801720707 is 12.3mm across and three 1.25mm slices thick, so by longest-over-
    shortest it looks as elongated as a vessel.

    Shape tells them apart once you look at all three axes. A tube has one long axis and
    two short ones, so first-over-second is large. A flat patch has two long axes and one
    short, and a sphere has three equal, so first-over-second is near 1 for both. That
    makes this ratio the thing that separates vessel from lesion, and it keeps flat
    lesions that the naive ratio would discard.
    """
    coords = np.argwhere(mask).astype(float) * np.asarray(spacing)
    if len(coords) < 3:
        return 1.0
    coords -= coords.mean(axis=0)
    extents = np.linalg.svd(coords, compute_uv=False) / np.sqrt(len(coords))
    return float(extents[0] / max(extents[1], 1e-6))


def density_candidates(
    ct: np.ndarray,
    lung: np.ndarray,
    spacing,
    min_diameter_mm: float = DENSITY_MIN_DIAMETER_MM,
    max_linearity: float = DENSITY_MAX_LINEARITY,
) -> tuple[np.ndarray, int]:
    """Shape-and-density candidates, to catch what the trained model doesn't.

    Evidence for building this: on RIDER-1801720707 the expert's 12.3mm lesion was missed
    entirely by `lung_nodules`, which returned four other marks instead. The lesion is
    solid (median -19 HU, eroded core +56) but flat — 12.3 x 9.1mm across and three 1.25mm
    slices thick — so it is simply not nodule-shaped, and a model trained on round LIDC
    nodules has no reason to fire on it.

    My first version of this looked only at the ground-glass band, because the mean HU of
    that contour is -163 and I read it as faint. The mean was dragged down by a rim of air
    voxels inside the contour; the median tells the real story. Hence the wide window.

    Deliberately crude and deliberately noisy. Daniel said 5-10 false marks per scan beats
    missing one faint nodule, so this spends that budget: threshold everything denser than
    parenchyma inside the lungs, open away the partial-volume shells that coat every vessel,
    then keep only components that are lesion-sized and not tube-shaped. The vessel tree
    survives as one enormous connected component and falls out on size, which is the main
    reason such a wide window is workable at all — and also its main weakness, since a
    lesion touching a large vessel gets absorbed into that component and lost.

    ponytail: fixed HU window plus a shape filter. If the false-positive count turns out to
    be unacceptable, the upgrade is nnU-Net's probability map thresholded low, not more
    hand-tuned rules.
    """
    band = lung & (ct >= DENSITY_LO) & (ct <= DENSITY_HI)
    # One pass of opening removes the one-voxel skin that partial volume paints around
    # every vessel and fissure, and barely touches a solid-ish blob.
    band = ndimage.binary_opening(band, np.ones((3, 3, 3), bool))

    labels, n = ndimage.label(band)
    if n == 0:
        return labels, 0

    voxel_mm3 = float(np.prod(spacing))
    sphere = lambda d: (4 / 3) * np.pi * (d / 2) ** 3 / voxel_mm3
    counts = np.bincount(labels.ravel())

    kept = np.zeros_like(labels)
    next_id = 0
    # Each component's own bounding box. Without this, every `labels == label` allocates a
    # mask the size of the whole scan and every geometry call walks all 200-odd slices of
    # it — for hundreds of components. Same answers, a fraction of the work, since both
    # geometry measures are translation-invariant.
    boxes = ndimage.find_objects(labels)
    for label in range(1, n + 1):
        # Cheapest filters first. The volume cap is what removes the vessel tree, which
        # survives the opening as one enormous component.
        if counts[label] < 8 or counts[label] > sphere(DENSITY_MAX_DIAMETER_MM):
            continue
        box = boxes[label - 1]
        mask = labels[box] == label
        if _linearity(mask, spacing) > max_linearity:
            continue
        # Size is judged on diameter, never on volume. A volume floor assumes the lesion is
        # a sphere, and the flat lesion on RIDER-1801720707 fails a 10mm volume floor while
        # measuring 17.7mm across — the measured sweep lost it for exactly that reason.
        # Diameter is also what Daniel records, so this threshold is in his units.
        long_axis = axial_axes(mask, spacing)[0]
        if long_axis < min_diameter_mm:
            continue
        # Cap the diameter as well as the volume. The volume cap alone lets a vessel tree
        # through: one candidate came out 108.5mm long at 21.5cm3, which is under a 40mm
        # sphere's volume and so passed, while being obviously not a lesion. Large masses
        # are the trained model's job; this generator exists for odd-shaped small ones.
        if long_axis > DENSITY_MAX_DIAMETER_MM:
            continue
        next_id += 1
        kept[box][mask] = next_id  # basic slicing gives a view, so this writes through
    return kept, next_id


def merge_labels(nodules: np.ndarray, extras: list[tuple[np.ndarray, str]]) -> tuple[np.ndarray, dict]:
    """One labelled volume from every detector, recording which found what.

    The trained model goes first and keeps its boundary. Each further detector only adds
    candidates that overlap nothing already there: two marks on one lesion costs the reviewer
    a decision they shouldn't have to make.
    """
    labels, n = ndimage.label(nodules.astype(bool))
    sources = {i: "model" for i in range(1, n + 1)}
    for extra, source in extras:
        if extra is None or not extra.max():
            continue
        for extra_id, box in enumerate(ndimage.find_objects(extra), 1):
            if box is None:
                continue
            mask = extra[box] == extra_id
            if (labels[box][mask] > 0).any():
                continue
            n += 1
            labels[box][mask] = n
            sources[n] = source
    return labels, sources


def candidates(
    labels: np.ndarray, sources: dict, ct: np.ndarray, spacing, lobe_dir: Path
) -> tuple[list[dict], np.ndarray]:
    lobes = load_lobes(lobe_dir, labels.shape)
    boxes = ndimage.find_objects(labels)
    found = []
    for label in sorted(sources):
        box = boxes[label - 1]
        mask = labels[box] == label
        m = measure_lesion(mask, ct[box], spacing)
        if m["long_axis_mm"] < MIN_LONG_AXIS_MM:
            continue
        m["label"] = int(label)  # lets the evaluation recover this candidate's voxels
        m["source"] = sources[label]
        m["lobe"] = lobe_of(mask, lobes[box])
        m["voxels"] = int(mask.sum())
        m["linearity"] = round(_linearity(mask, spacing), 2)
        # Slice index and centroid must be reported in the full scan's coordinates, not the
        # crop's, or Daniel scrolls to the wrong slice.
        offset = [s.start for s in box]
        m["slice_index"] += offset[0]
        centre = ndimage.center_of_mass(mask)
        m["centroid_zyx"] = [round(float(c) + o, 1) for c, o in zip(centre, offset)]
        found.append(m)
    # Order the worklist by how lesion-like a candidate is, not by how big it is. Sorting by
    # size put the true lesion 34th of 41 on one scan, because vessel segments are larger
    # than nodules. The trained model's candidates come first — it found the lesion on three
    # of four scans — and within each group the roundest come first, since the leftovers in
    # this density band are mostly vessel.
    # Faint candidates rank above density ones: missing a hazy nodule costs Daniel his whole
    # second pass, so those are the marks he most needs to see early.
    priority = {"model": 0, "faint": 1, "density": 2}
    found.sort(key=lambda m: (priority[m["source"]], m["linearity"], -m["long_axis_mm"]))
    for i, m in enumerate(found, 1):
        m["nodule_id"] = f"N{i}"
    return found, labels


def run(series_dir: Path, device: str = "cpu", faint: bool = False) -> dict:
    image = load_series(series_dir)
    work = series_dir / "caliper"
    work.mkdir(exist_ok=True)

    nifti = to_nifti(image, work / "ct.nii.gz")
    nodules, lungs = segment(nifti, work, device=device)

    ct = sitk.GetArrayFromImage(image)
    # SimpleITK reports spacing as (x, y, z); every array here is (z, y, x).
    spacing = tuple(reversed(image.GetSpacing()))

    # Imported here because faint.py builds on helpers in this module.
    from caliper.faint import faint_candidates, segment_vessels, trim_vessels, widen
    from caliper.outline import refine_all

    raw_vessels = segment_vessels(nifti, work, device=device)
    vessels = widen(raw_vessels)
    dense, _ = density_candidates(ct, lungs, spacing)
    dense = trim_vessels(dense, vessels, spacing)  # both noise marks Daniel saw were bifurcations
    extras = [(dense, "density")]
    # Off by default. A fixed ground-glass HU window found 89-526 candidates per scan, mostly
    # gravity-dependent haze in the posterior lower lobes and small vessels seen end-on.
    # Kept behind --faint until it can be measured against real faint-nodule ground truth.
    if faint:
        faint_labels, _ = faint_candidates(ct, lungs, vessels, spacing)
        extras.append((faint_labels, "faint"))
    labels, sources = merge_labels(nodules, extras)
    # The nodule model under-draws large tumours from the inside; grow each outline a few mm.
    labels = refine_all(labels, ct, lungs, raw_vessels, spacing)
    found, labels = candidates(labels, sources, ct, spacing, work / "lobes")
    # Keep the labelled volume so evaluation and review can recover a candidate's voxels
    # without re-running a ten-minute segmentation.
    np.save(work / "labels.npy", labels.astype(np.int16))

    meta = json.loads((series_dir / "series.json").read_text()) if (series_dir / "series.json").exists() else {}
    rows = [
        sheet_row(
            m,
            patient_id=meta.get("PatientID", series_dir.parent.name),
            scan_date=str(meta.get("StudyDate", ""))[:10],
            timepoint=series_dir.name,
            series_number=meta.get("SeriesNumber", ""),
            slice_thickness=round(spacing[0], 2),
            nodule_id=m["nodule_id"],
            lobe=m["lobe"],
            notes=f"candidate, unreviewed, found by {m['source']}",
        )
        for m in found
    ]

    (work / "worklist.json").write_text(json.dumps(found, indent=1))
    with (work / "sheet.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    by_source = {s: sum(1 for m in found if m["source"] == s) for s in ("model", "faint", "density")}
    print(
        f"{len(found)} candidates ({by_source['model']} model, {by_source['faint']} faint, "
        f"{by_source['density']} density) -> {work/'sheet.csv'}"
    )
    for m in found:
        print(
            f"  {m['nodule_id']}  {m['long_axis_mm']:>5.1f} x {m['short_axis_mm']:>5.1f} mm  "
            f"{m['volume_mm3']:>8.0f} mm3  {m['type']:<12} {m['source']:<8} {m['lobe']}"
        )
    return {"candidates": found, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-process one CT series into a candidate worklist.")
    ap.add_argument("series_dir", type=Path)
    ap.add_argument("--device", default="cpu", help="cpu | gpu | mps")
    ap.add_argument("--faint", action="store_true", help="experimental ground-glass pass")
    args = ap.parse_args()
    run(args.series_dir, device=args.device, faint=args.faint)


if __name__ == "__main__":
    main()
