"""Bereitet dataset2 (YOLO-Format) auf.

1. Fester Train/Val-Split auf Bildebene -> datasets/dataset2/{train,val}.txt + data.yaml
   (für YOLO-Training mit Ultralytics)
2. Schneidet jede gelabelte Münze quadratisch aus und speichert sie nach
   data/crops/dataset2/{train,val}/<klasse>/ (gleiche Ordnerstruktur wie coins-dataset/classified),
   damit ein Klassifikations-CNN mit beiden Datensätzen trainiert werden kann.

Der Split der Crops folgt dem Split der Bilder, damit keine Münze aus einem Val-Bild im Training landet.

Aufruf (aus dem Projekt-Root):  python -m scripts.prepare_dataset2 [--val-frac 0.2] [--margin 0.1]
"""
import argparse
import random
import shutil
from collections import Counter

import cv2
import yaml

from coincounter.constants import CLASS_NAMES, DATA_OUT, DATASETS

DS = DATASETS / "dataset2"


def read_labels(label_path):
    boxes = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls, x, y, w, h = line.split()
        boxes.append((int(cls), float(x), float(y), float(w), float(h)))
    return boxes


def square_crop(img, x, y, w, h, margin):
    """Quadratischer Ausschnitt um die Box (normierte YOLO-Koordinaten), Rand per Reflect aufgefüllt."""
    H, W = img.shape[:2]
    side = max(w * W, h * H) * (1 + margin)
    cx, cy = x * W, y * H
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    pad = max(0, -x0, -y0, x1 - W, y1 - H)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    return img[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--margin", type=float, default=0.1, help="Rand um die Box, relativ zur Boxgröße")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images = sorted(p for p in (DS / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    missing = [p.name for p in images if not (DS / "labels" / f"{p.stem}.txt").exists()]
    if missing:
        raise SystemExit(f"Labels fehlen für: {missing}")

    rng = random.Random(args.seed)
    shuffled = images[:]
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * args.val_frac)
    splits = {"val": sorted(shuffled[:n_val]), "train": sorted(shuffled[n_val:])}

    # 1) Split-Listen + data.yaml für YOLO
    for name, imgs in splits.items():
        (DS / f"{name}.txt").write_text("\n".join(f"./images/{p.name}" for p in imgs) + "\n")
    # Kein "path"-Key: Ultralytics nimmt dann den Ordner der YAML -> portabel für alle im Team
    data_yaml = {
        "train": "train.txt",
        "val": "val.txt",
        "names": dict(enumerate(CLASS_NAMES)),
    }
    (DS / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True))

    # 2) Crops
    out_root = DATA_OUT / "crops" / "dataset2"
    if out_root.exists():
        shutil.rmtree(out_root)
    for split, imgs in splits.items():
        counts = Counter()
        for img_path in imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                raise SystemExit(f"Kann Bild nicht lesen: {img_path}")
            for i, (cls, x, y, w, h) in enumerate(read_labels(DS / "labels" / f"{img_path.stem}.txt")):
                name = CLASS_NAMES[cls]
                out_dir = out_root / split / name
                out_dir.mkdir(parents=True, exist_ok=True)
                crop = square_crop(img, x, y, w, h, args.margin)
                cv2.imwrite(str(out_dir / f"{img_path.stem}_{i}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                counts[name] += 1
        print(f"{split:5s}: {len(imgs):3d} Bilder, {sum(counts.values()):3d} Münzen  "
              + "  ".join(f"{c}={counts[c]}" for c in CLASS_NAMES))

    print(f"\ndata.yaml -> {DS / 'data.yaml'}\nCrops     -> {out_root}")


if __name__ == "__main__":
    main()
