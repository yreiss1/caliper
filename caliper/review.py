"""Build the review UI as a static site: one HTML page, one JSON file, pre-rendered images.

Why static: the lab can't install anything, and Naomi wants a link she can open. A folder
of files needs no server, no database and no DICOM viewer — it can be hosted anywhere or
opened from disk. Each candidate gets a filmstrip of slices around it with the outline
drawn on, rendered here once, so the browser never touches DICOM.

ponytail: pre-rendered PNGs, no window/level control or free scrolling. Fine for "is this a
lesion, yes or no"; add Cornerstone3D when Daniel needs to scroll the whole scan.

    .venv/bin/python -m caliper.review data/rider --out review
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage

from caliper.pipeline import load_series

# Standard lung window: centre -600, width 1500.
WINDOW_LO, WINDOW_HI = -1350.0, 150.0
SLICES_EACH_SIDE = 2
TILE_PX = 220
OUTLINE_RGB = (255, 70, 70)


def window(hu: np.ndarray) -> np.ndarray:
    return (np.clip((hu - WINDOW_LO) / (WINDOW_HI - WINDOW_LO), 0, 1) * 255).astype(np.uint8)


def filmstrip(ct, labels, candidate, spacing) -> Image.Image:
    """Slices around the candidate's measured slice, cropped to it, outline in red."""
    mask_full = labels == candidate["label"]
    cz, cy, cx = (int(round(c)) for c in candidate["centroid_zyx"])
    z0 = candidate["slice_index"] if candidate["slice_index"] >= 0 else cz

    # Crop wide enough to see what the lesion touches — the pleura is where Daniel's
    # judgement calls happen, so the context matters as much as the lesion.
    half_mm = max(30.0, candidate["long_axis_mm"] * 1.2)
    half = int(half_mm / spacing[1])
    ys = slice(max(0, cy - half), min(ct.shape[1], cy + half))
    xs = slice(max(0, cx - half), min(ct.shape[2], cx + half))

    tiles = []
    for z in range(z0 - SLICES_EACH_SIDE, z0 + SLICES_EACH_SIDE + 1):
        if not 0 <= z < ct.shape[0]:
            tiles.append(Image.new("RGB", (TILE_PX, TILE_PX)))
            continue
        grey = window(ct[z, ys, xs])
        rgb = np.stack([grey] * 3, axis=-1)
        m = mask_full[z, ys, xs]
        edge = m & ~ndimage.binary_erosion(m)
        rgb[edge] = OUTLINE_RGB
        tile = Image.fromarray(rgb).resize((TILE_PX, TILE_PX), Image.NEAREST)
        tiles.append(tile)

    strip = Image.new("RGB", (TILE_PX * len(tiles), TILE_PX))
    for i, tile in enumerate(tiles):
        strip.paste(tile, (i * TILE_PX, 0))
    return strip


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the static review UI.")
    ap.add_argument("root", type=Path, help="data/rider")
    ap.add_argument("--out", type=Path, default=Path("review"))
    args = ap.parse_args()

    images = args.out / "img"
    images.mkdir(parents=True, exist_ok=True)
    scans = []

    for series_dir in sorted(args.root.glob("*/test")):
        work = series_dir / "caliper"
        if not (work / "worklist.json").exists():
            continue
        patient = series_dir.parent.name
        found = json.loads((work / "worklist.json").read_text())
        labels = np.load(work / "labels.npy")
        image = load_series(series_dir)
        ct = sitk.GetArrayFromImage(image)
        spacing = tuple(reversed(image.GetSpacing()))
        meta = json.loads((series_dir / "series.json").read_text())

        rows = []
        for c in found:
            name = f"{patient}_{c['nodule_id']}.png"
            filmstrip(ct, labels, c, spacing).save(images / name, optimize=True)
            rows.append({**{k: c[k] for k in (
                "nodule_id", "long_axis_mm", "short_axis_mm", "mean_diameter_mm",
                "volume_mm3", "solid_component_mm", "type", "lobe", "source", "slice_index",
            )}, "image": f"img/{name}"})

        scans.append({
            "patient": patient,
            "scan_date": str(meta.get("StudyDate", ""))[:10],
            "timepoint": series_dir.name,
            "series": meta.get("SeriesNumber", ""),
            "slice_thickness": round(spacing[0], 2),
            "candidates": rows,
        })
        print(f"  {patient}: {len(rows)} candidates rendered")

    (args.out / "data.json").write_text(json.dumps({"scans": scans}))
    shutil.copy(Path(__file__).parent / "static" / "index.html", args.out / "index.html")
    print(f"review UI written to {args.out}/ — open index.html or serve the folder")


if __name__ == "__main__":
    main()
