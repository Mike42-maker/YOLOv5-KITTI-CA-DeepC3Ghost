#!/usr/bin/env python3
"""Convert completed KITTI training ZIP archives into a four-class YOLO dataset.

This script is preparation only until it is explicitly run after both ZIP files
have finished downloading. It never writes to, extracts beside, or deletes the
source archives.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image

CLASSES = {"Car": 0, "Van": 1, "Truck": 2, "Tram": 3}
SEED = 42
VAL_FRACTION = 0.20


def parse_label(line: str, image_width: int, image_height: int) -> str | None:
    """Return one normalized YOLO label, or None for excluded/invalid KITTI objects."""
    fields = line.split()
    if len(fields) < 8 or fields[0] not in CLASSES:
        return None
    try:
        xmin, ymin, xmax, ymax = map(float, fields[4:8])
    except ValueError:
        return None
    xmin, xmax = max(0.0, xmin), min(float(image_width), xmax)
    ymin, ymax = max(0.0, ymin), min(float(image_height), ymax)
    if xmax <= xmin or ymax <= ymin:
        return None
    xc = (xmin + xmax) / (2 * image_width)
    yc = (ymin + ymax) / (2 * image_height)
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height
    if not all(0.0 <= value <= 1.0 for value in (xc, yc, width, height)):
        return None
    return f"{CLASSES[fields[0]]} {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}"


def archive_members(archive: zipfile.ZipFile, suffix: str) -> dict[str, str]:
    """Map filename stems to archive members under the requested KITTI directory."""
    return {
        Path(member).stem: member
        for member in archive.namelist()
        if f"training/{suffix}/" in member and not member.endswith("/")
    }


def directory_members(directory: Path, suffix: str) -> dict[str, Path]:
    """Map filename stems to files in an already extracted KITTI directory."""
    return {
        path.stem: path
        for path in directory.glob(f"*{suffix}")
        if path.is_file()
    }


def image_size(image_source: Path | bytes) -> tuple[int, int]:
    """Read and fully decode an image before returning its dimensions."""
    source = BytesIO(image_source) if isinstance(image_source, bytes) else image_source
    with Image.open(source) as image:
        image.load()
        return image.size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-zip", type=Path, default=Path("data_object_image_2.zip"))
    parser.add_argument("--labels-zip", type=Path, default=Path("data_object_label_2.zip"))
    parser.add_argument("--images-dir", type=Path, help="Use extracted training image_2 files.")
    parser.add_argument("--labels-dir", type=Path, help="Use extracted training label_2 files.")
    parser.add_argument("--output", type=Path, default=Path("datasets/kitti-yolo"))
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory.")
    args = parser.parse_args()

    directory_mode = args.images_dir is not None or args.labels_dir is not None
    if directory_mode and (args.images_dir is None or args.labels_dir is None):
        sys.exit("--images-dir and --labels-dir must be provided together; no conversion was performed.")
    if directory_mode and (not args.images_dir.is_dir() or not args.labels_dir.is_dir()):
        sys.exit("Both extracted KITTI directories are required; no conversion was performed.")
    if not directory_mode and (not args.images_zip.is_file() or not args.labels_zip.is_file()):
        sys.exit("Both completed KITTI ZIP files are required; no conversion was performed.")
    if args.output.exists() and any(args.output.iterdir()) and not args.force:
        sys.exit(f"Output already exists: {args.output}. Use --force only after reviewing it.")

    if args.output.exists() and args.force:
        shutil.rmtree(args.output)
    counts = Counter()
    if directory_mode:
        image_members = directory_members(args.images_dir, ".png")
        label_members = directory_members(args.labels_dir, ".txt")
        image_ids = sorted(set(image_members) & set(label_members))
        readable_ids = []
        for image_id in image_ids:
            try:
                image_size(image_members[image_id])
            except Exception:
                continue
            readable_ids.append(image_id)
        image_ids = readable_ids

        random.Random(SEED).shuffle(image_ids)
        val_count = round(len(image_ids) * VAL_FRACTION)
        split_ids = {"val": set(image_ids[:val_count]), "train": set(image_ids[val_count:])}
        for split, ids in split_ids.items():
            (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
            (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)
            for image_id in sorted(ids):
                image_path = image_members[image_id]
                (args.output / "images" / split / f"{image_id}.png").write_bytes(image_path.read_bytes())
                width, height = image_size(image_path)
                label_lines = []
                for line in label_members[image_id].read_text(encoding="utf-8").splitlines():
                    yolo_label = parse_label(line, width, height)
                    if yolo_label:
                        counts[int(yolo_label[0])] += 1
                        label_lines.append(yolo_label)
                (args.output / "labels" / split / f"{image_id}.txt").write_text(
                    "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8"
                )
    else:
        # testzip reads archive integrity but makes no changes to the source ZIPs.
        with zipfile.ZipFile(args.images_zip) as images_archive, zipfile.ZipFile(args.labels_zip) as labels_archive:
            if images_archive.testzip() or labels_archive.testzip():
                sys.exit("A ZIP archive is incomplete or corrupt; no conversion was performed.")
            image_members = archive_members(images_archive, "image_2")
            label_members = archive_members(labels_archive, "label_2")
            image_ids = sorted(set(image_members) & set(label_members))
            if len(image_ids) != 7481:
                sys.exit(f"Expected 7481 matched KITTI training pairs, found {len(image_ids)}; no conversion was performed.")

            random.Random(SEED).shuffle(image_ids)
            val_count = round(len(image_ids) * VAL_FRACTION)
            split_ids = {"val": set(image_ids[:val_count]), "train": set(image_ids[val_count:])}
            for split, ids in split_ids.items():
                (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
                (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)
                for image_id in sorted(ids):
                    image_bytes = images_archive.read(image_members[image_id])
                    (args.output / "images" / split / f"{image_id}.png").write_bytes(image_bytes)
                    width, height = image_size(image_bytes)
                    label_lines = []
                    for line in labels_archive.read(label_members[image_id]).decode("utf-8").splitlines():
                        yolo_label = parse_label(line, width, height)
                        if yolo_label:
                            counts[int(yolo_label[0])] += 1
                            label_lines.append(yolo_label)
                    (args.output / "labels" / split / f"{image_id}.txt").write_text(
                        "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8"
                    )

    (args.output / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnc: 4\nnames: [Car, Van, Truck, Tram]\n",
        encoding="utf-8",
    )
    print(f"Converted {len(split_ids['train'])} train and {len(split_ids['val'])} val images.")
    print(f"Class instances: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()

