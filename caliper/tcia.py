"""Fetch RIDER Lung CT from TCIA.

The lab's instructions point at the NBIA Data Retriever, which is a desktop Java app. The
public REST API serves the same series over plain HTTP with no login, which is scriptable
and reproducible. Verified against the live API: `getSeries` for "RIDER Lung CT" returns
2072 series, licensed CC BY 4.0.

Attribution is required and lives in README.md.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import requests

BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "RIDER Lung CT"
TIMEOUT = 600


def list_series(collection: str = COLLECTION) -> list[dict]:
    r = requests.get(f"{BASE}/getSeries", params={"Collection": collection}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def patients(series: list[dict]) -> dict[str, list[dict]]:
    """Series grouped by patient, each patient's series ordered by study then series number."""
    out: dict[str, list[dict]] = {}
    for s in series:
        out.setdefault(s["PatientID"], []).append(s)
    for v in out.values():
        v.sort(key=lambda s: (s.get("StudyDate", ""), s.get("SeriesNumber", 0)))
    return out


#: The expert delineations ship as RTSTRUCT series described exactly "TEST" and "RETEST"
#: (30 and 29 of them across the collection, one radiation oncologist, blinded).
EXPERT_RTSTRUCT = ("TEST", "RETEST")


def pick_timepoints(patient_series: list[dict], out: Path) -> list[tuple[str, dict, Path]]:
    """Let the expert's contours choose the CT series.

    A RIDER patient has dozens of CT series: several reconstructions per study, two-image
    scouts, and StudyDate values that do not line up with the test/retest design (one
    StudyInstanceUID here appears under two different dates). Guessing from slice count or
    date gets it wrong — my first attempt picked a two-image scout.

    The RTSTRUCT files remove the guesswork: each one names the CT series the oncologist
    actually contoured, in ReferencedFrameOfReferenceSequence. That is the only series
    worth measuring, because it is the only one with ground truth attached.

    Returns [(label, ct_series, rtstruct_dir)].
    """
    import pydicom

    out.mkdir(parents=True, exist_ok=True)
    by_uid = {s["SeriesInstanceUID"]: s for s in patient_series}
    picked = []
    for s in patient_series:
        if s.get("Modality") != "RTSTRUCT" or s.get("SeriesDescription") not in EXPERT_RTSTRUCT:
            continue
        label = s["SeriesDescription"].lower()
        rt_dir = download_series(s["SeriesInstanceUID"], out / f"rtstruct_{label}")
        ds = pydicom.dcmread(next(rt_dir.glob("*.dcm")))
        ref = (
            ds.ReferencedFrameOfReferenceSequence[0]
            .RTReferencedStudySequence[0]
            .RTReferencedSeriesSequence[0]
            .SeriesInstanceUID
        )
        ct = by_uid.get(ref)
        if ct is None:
            print(f"  {label}: referenced CT series not in this collection, skipping")
            continue
        picked.append((label, ct, rt_dir))
    return sorted(picked, key=lambda t: t[0] != "test")  # test first, then retest


def lesion_long_axis_mm(rtstruct_file: Path, name_contains: str = "_man") -> float:
    """Longest in-plane diameter of a contoured lesion, straight from the RTSTRUCT.

    `ContourData` is patient-space millimetres, so this needs no CT at all — which is the
    point: 30 RTSTRUCT files are ~10MB and tell me how big every annotated tumour is,
    where downloading the scans to find out would be ~4GB.
    """
    import numpy as np
    import pydicom

    ds = pydicom.dcmread(rtstruct_file)
    names = {r.ROINumber: r.ROIName for r in ds.StructureSetROISequence}
    longest = 0.0
    for roi in ds.ROIContourSequence:
        if name_contains not in names.get(roi.ReferencedROINumber, ""):
            continue
        for contour in getattr(roi, "ContourSequence", []):
            pts = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)[:, :2]
            if len(pts) < 2:
                continue
            spread = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
            longest = max(longest, float(spread.max()))
    return longest


def survey(series: list[dict], out: Path) -> list[tuple[float, str]]:
    """Rank every annotated patient by tumour size, cheaply.

    Needed because the first patient I pulled had a 69mm primary — fine for plumbing,
    useless for testing the small-lesion behaviour Daniel actually struggles with.
    """
    sizes = []
    for patient_id, patient_series in sorted(patients(series).items()):
        rt = [s for s in patient_series if s.get("SeriesDescription") == "TEST"]
        if not rt:
            continue
        rt_dir = download_series(rt[0]["SeriesInstanceUID"], out / patient_id / "rtstruct_test")
        try:
            longest = lesion_long_axis_mm(next(rt_dir.glob("*.dcm")))
        except Exception as exc:  # malformed RTSTRUCT shouldn't stop the survey
            print(f"{patient_id}: {exc}")
            continue
        sizes.append((longest, patient_id))
    sizes.sort()
    for longest, patient_id in sizes:
        print(f"{longest:6.1f} mm  {patient_id}")
    return sizes


def download_series(series_uid: str, dest: Path) -> Path:
    """Download one series and unzip it into `dest`. Skips if already present."""
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.glob("*.dcm")):
        return dest
    r = requests.get(f"{BASE}/getImage", params={"SeriesInstanceUID": series_uid}, timeout=TIMEOUT)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="Download RIDER Lung CT test/retest pairs from TCIA.")
    ap.add_argument("--out", type=Path, default=Path("data/rider"))
    ap.add_argument("--patients", type=int, default=1, help="how many patients to fetch")
    ap.add_argument(
        "--survey",
        action="store_true",
        help="rank annotated patients by tumour size from RTSTRUCTs only, download no scans",
    )
    ap.add_argument("--patient", action="append", help="fetch these patient IDs instead")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    index = args.out / "series_index.json"
    if index.exists():
        series = json.loads(index.read_text())
    else:
        series = list_series()
        index.write_text(json.dumps(series, indent=1))

    if args.survey:
        survey(series, args.out)
        return

    by_patient = patients(series)
    annotated = sorted(
        pid
        for pid, ss in by_patient.items()
        if any(s.get("SeriesDescription") in EXPERT_RTSTRUCT for s in ss)
    )
    wanted = args.patient or annotated[: args.patients]
    print(f"{len(annotated)} patients with expert contours; fetching {len(wanted)}")

    for patient_id in wanted:
        for label, ct, _ in pick_timepoints(by_patient[patient_id], args.out / patient_id):
            dest = args.out / patient_id / label
            print(
                f"{patient_id} {label}: series {ct.get('SeriesNumber')}, "
                f"{ct['ImageCount']} images, {ct['FileSize']/1e6:.0f}MB"
            )
            download_series(ct["SeriesInstanceUID"], dest)
            (dest / "series.json").write_text(json.dumps(ct, indent=1))
    print("done")


if __name__ == "__main__":
    main()
