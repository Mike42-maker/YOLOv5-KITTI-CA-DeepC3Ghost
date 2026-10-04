#!/usr/bin/env python3
"""Validate a converted KITTI YOLO dataset and optionally save sample images."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

NAMES = ["Car", "Van", "Truck", "Tram"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("datasets/kitti-yolo"))
    parser.add_argument("--visualize", type=int, default=0, metavar="N", help="Save N deterministic samples per split.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    errors: list[str] = []
    class_counts = Counter()
    for split in ("train", "val"):
        image_dir, label_dir = args.data / "images" / split, args.data / "labels" / split
        images = {path.stem: path for path in image_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES}
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        missing_labels, missing_images = sorted(set(images) - set(labels)), sorted(set(labels) - set(images))
        empty_labels = 0
        for image_id, label_path in labels.items():
            lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty_labels += 1
            for number, line in enumerate(lines, start=1):
                fields = line.split()
                try:
                    class_id = int(fields[0])
                    values = [float(value) for value in fields[1:]]
                except (IndexError, ValueError):
                    errors.append(f"{label_path}:{number}: invalid label format")
                    continue
                if len(fields) != 5 or class_id not in range(4) or any(value < 0 or value > 1 for value in values):
                    errors.append(f"{label_path}:{number}: class or coordinate out of bounds")
                else:
                    class_counts[class_id] += 1
        print(f"{split}: images={len(images)}, labels={len(labels)}, empty_labels={empty_labels}, missing_labels={len(missing_labels)}, missing_images={len(missing_images)}")
        if missing_labels:
            errors.append(f"{split}: labels missing for {len(missing_labels)} images")
        if missing_images:
            errors.append(f"{split}: images missing for {len(missing_images)} labels")
        if args.visualize:
            output_dir = args.data / "checks" / split
            output_dir.mkdir(parents=True, exist_ok=True)
            for image_id in random.Random(args.seed).sample(sorted(images), min(args.visualize, len(images))):
                with Image.open(images[image_id]).convert("RGB") as image:
                    draw = ImageDraw.Draw(image)
                    for line in labels.get(image_id, Path()).read_text(encoding="utf-8").splitlines() if image_id in labels else []:
                        cls, xc, yc, width, height = map(float, line.split())
                        x1, y1 = (xc - width / 2) * image.width, (yc - height / 2) * image.height
                        x2, y2 = (xc + width / 2) * image.width, (yc + height / 2) * image.height
                        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
                        draw.text((x1, max(0, y1 - 12)), NAMES[int(cls)], fill="red")
                    image.save(output_dir / images[image_id].name)
    print("class_instances:", {NAMES[key]: class_counts[key] for key in range(4)})
    if errors:
        print("errors:")
        print("\n".join(errors[:50]))
        raise SystemExit(1)
    print("Dataset checks passed.")


if __name__ == "__main__":
    main()

