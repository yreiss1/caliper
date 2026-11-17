"""Measure faint-nodule recall on LIDC-IDRI, which RIDER cannot do.

RIDER contours one primary tumour per patient, so it says nothing about the small hazy
nodules Daniel worries about, and nothing about false positives. LIDC-IDRI has four
radiologists marking every nodule of 3mm and up, and rating each one's texture from 1
(non-solid, i.e. ground-glass) to 5 (solid). pylidc ships that whole annotation database,
so picking scans with faint nodules needs no download at all.

    .venv/bin/python -m caliper.lidc select --n 6        # list scans, write data/lidc/selection.json
    .venv/bin/python -m caliper.lidc fetch               # download those CT series
    .venv/bin/python -m caliper.lidc evaluate            # recall + noise per scan
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# pylidc predates NumPy 1.24 and still uses the removed `np.int` alias.
# ponytail: a two-line shim instead of a fork; drop it if pylidc ever releases a fix.
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

ROOT = Path("data/lidc")
#: Texture 1 is non-solid, 2 non-solid/mixed. That is the hazy end Daniel misses on a first pass.
FAINT_TEXTURE = 2
#: Daniel records nodules of 4mm and up.
MIN_DIAMETER_MM = 4.0
#: At least this many of the four radiologists marked it. One reader alone is too weak a truth.
MIN_READERS = 2
#: A candidate counts as finding a nodule if its centre lands this close to the nodule's.
HIT_MM = 6.0
#: A candidate this far from every radiologist's mark, of any texture, counts as noise.
NOISE_MM = 10.0


def _nodules(scan) -> list[dict]:
    """Consensus nodules in one scan, with centre in patient millimetres."""
    out = []
    for group in scan.cluster_annotations():
        if len(group) < MIN_READERS:
            continue
        textures = [a.texture for a in group]
        diameter = float(np.median([a.diameter for a in group]))
        points = []
        for a in group:
            for c in a.contours:
                rc = c.to_matrix(include_k=False)  # (row, col) pixel coords on that slice
                points.extend((r, col, c.image_z_position) for r, col in rc)
        row, col, z = np.mean(points, axis=0)
        out.append({
            "texture": float(np.median(textures)),
            "faint": float(np.median(textures)) <= FAINT_TEXTURE,
            "diameter_mm": round(diameter, 1),
            "readers": len(group),
            "row": float(row), "col": float(col), "z_mm": float(z),
        })
    return out


def select(n: int) -> list[dict]:
    import pylidc as pl

    picked = []
    # Thin slices only, to match the 1.25mm series Daniel reads.
    for scan in pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 1.5).order_by(pl.Scan.patient_id):
        nods = _nodules(scan)
        faint = [x for x in nods if x["faint"] and x["diameter_mm"] >= MIN_DIAMETER_MM]
        if not faint:
            continue
        picked.append({
            "patient_id": scan.patient_id,
            "series_uid": scan.series_instance_uid,
            "slice_thickness": scan.slice_thickness,
            "pixel_spacing": scan.pixel_spacing,
            "faint_nodules": len(faint),
            "nodules": nods,
        })
        print(f"{scan.patient_id}: {len(faint)} faint of {len(nods)} consensus nodules")
        if len(picked) >= n:
            break
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "selection.json").write_text(json.dumps(picked, indent=1))
    return picked


def fetch() -> None:
    from caliper.tcia import download_series

    for s in json.loads((ROOT / "selection.json").read_text()):
        dest = ROOT / s["patient_id"] / "scan"
        download_series(s["series_uid"], dest)
        (dest / "series.json").write_text(json.dumps({
            "PatientID": s["patient_id"], "SeriesInstanceUID": s["series_uid"],
        }))
        print(f"{s['patient_id']}: {len(list(dest.glob('*.dcm')))} slices")


def _to_mm(image, row: float, col: float, z_mm: float) -> np.ndarray:
    """LIDC (row, col, slice z) to patient millimetres, via this series' own origin."""
    ox, oy, _ = image.GetOrigin()
    sx, sy, _ = image.GetSpacing()
    return np.array([ox + col * sx, oy + row * sy, z_mm])


def evaluate() -> None:
    from caliper.pipeline import load_series

    rows = []
    for s in json.loads((ROOT / "selection.json").read_text()):
        series_dir = ROOT / s["patient_id"] / "scan"
        work = series_dir / "caliper"
        if not (work / "worklist.json").exists():
            print(f"{s['patient_id']}: not processed yet")
            continue
        image = load_series(series_dir)
        found = json.loads((work / "worklist.json").read_text())
        cand = {
            c["nodule_id"]: np.array(
                image.TransformContinuousIndexToPhysicalPoint(tuple(reversed(c["centroid_zyx"])))
            )
            for c in found
        }
        truth = [(x, _to_mm(image, x["row"], x["col"], x["z_mm"])) for x in s["nodules"]]

        def hit(nodule_mm, sources=None):
            for c in found:
                if sources and c["source"] not in sources:
                    continue
                if np.linalg.norm(cand[c["nodule_id"]] - nodule_mm) <= HIT_MM:
                    return c["source"]
            return None

        faint = [(x, p) for x, p in truth if x["faint"] and x["diameter_mm"] >= MIN_DIAMETER_MM]
        noise = sum(
            1 for c in found
            if min((np.linalg.norm(cand[c["nodule_id"]] - p) for _, p in truth), default=1e9) > NOISE_MM
        )
        # Same run, scored twice: every candidate carries its detector, so dropping the
        # faint pass's marks gives the "faint pass off" numbers without a second run.
        base = {"model", "density"}
        row = {
            "patient": s["patient_id"],
            "faint_nodules": len(faint),
            "faint_found_without_faint_pass": sum(1 for _, p in faint if hit(p, base)),
            "noise_without_faint_pass": sum(
                1 for c in found if c["source"] in base and
                min((np.linalg.norm(cand[c["nodule_id"]] - q) for _, q in truth), default=1e9) > NOISE_MM
            ),
            "faint_found": sum(1 for _, p in faint if hit(p)),
            "faint_found_by": [hit(p) for _, p in faint],
            "all_nodules": len(truth),
            "all_found": sum(1 for _, p in truth if hit(p)),
            "candidates": len(found),
            "noise": noise,
            "by_source": {src: sum(1 for c in found if c["source"] == src)
                          for src in ("model", "density", "faint")},
        }
        rows.append(row)
        print(
            f"{row['patient']}: faint off {row['faint_found_without_faint_pass']}/{row['faint_nodules']} "
            f"noise {row['noise_without_faint_pass']} | faint on "
            f"{row['faint_found']}/{row['faint_nodules']} "
            f"{row['faint_found_by']}  all {row['all_found']}/{row['all_nodules']}  "
            f"candidates {row['candidates']} {row['by_source']}  noise {row['noise']}"
        )
    (ROOT / "evaluation.json").write_text(json.dumps(rows, indent=1))
    if rows:
        total = sum(r["faint_nodules"] for r in rows)
        print(
            f"\nfaint pass OFF: faint recall {sum(r['faint_found_without_faint_pass'] for r in rows)}/{total}, "
            f"noise {np.mean([r['noise_without_faint_pass'] for r in rows]):.1f}/scan"
        )
        print(
            f"faint pass ON:  faint recall {sum(r['faint_found'] for r in rows)}/{total}, "
            f"noise {np.mean([r['noise'] for r in rows]):.1f}/scan"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="LIDC-IDRI faint-nodule evaluation.")
    ap.add_argument("step", choices=["select", "fetch", "evaluate"])
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    {"select": lambda: select(args.n), "fetch": fetch, "evaluate": evaluate}[args.step]()


if __name__ == "__main__":
    main()
