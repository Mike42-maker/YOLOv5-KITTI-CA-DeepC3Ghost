# YOLOv5-KITTI

A reproducible research repository for four-class vehicle detection on a KITTI-derived dataset. The repository is based on **Ultralytics YOLOv5** and contains the project-specific Coordinate Attention, C3Ghost, Deep-C3Ghost, KITTI conversion, training, validation, and complexity-analysis code.

The original YOLOv5 implementation remains the base detection framework. The project additions are:

- **Coordinate Attention (CA):** an attention module inserted in the PANet feature-fusion path.
- **C3Ghost:** a Ghost-based C3 replacement used in controlled ablations.
- **Deep-C3Ghost:** selective C3Ghost replacement in the deep P4/P5 backbone stages.
- **KITTI adaptation and experiments:** four-class label conversion, data checks, model YAML files, training/validation commands, and reported measurements.

The upstream YOLOv5 LICENSE is retained unchanged. This repository acknowledges the Ultralytics YOLOv5 authors and distributes the source under the upstream AGPL-3.0 terms.

## Method

The final model combines Coordinate Attention with Deep-C3Ghost. Coordinate Attention is applied after the P3/8 and P4/16 PANet fusion outputs. C3Ghost replaces the deep backbone C3 blocks at the P4/16 and P5/32 stages while the detection head keeps the standard YOLOv5s C3 blocks. The final detection layers receive the P3, P4, and P5 features.

Model definitions are in yolov5/models/:

- yolov5s.yaml: original YOLOv5s baseline.
- yolov5s_ca.yaml: YOLOv5s with Coordinate Attention.
- yolov5s_c3ghost.yaml: YOLOv5s with C3Ghost ablation.
- yolov5s_ca_c3ghost.yaml: CA with broad C3Ghost replacement.
- yolov5s_ca_backbone_c3ghost.yaml: CA with backbone C3Ghost replacement.
- yolov5s_ca_deep_c3ghost.yaml: final CA + Deep-C3Ghost model.

## Dataset

The repository does not contain KITTI data. Download KITTI from its official distribution channel and prepare it locally. The conversion script maps:

| ID | Class |
|---:|---|
| 0 | Car |
| 1 | Van |
| 2 | Truck |
| 3 | Tram |

Expected prepared layout:

    datasets/kitti-yolo/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

Convert downloaded archives or extracted KITTI folders:

    python scripts/prepare_kitti_yolo.py --images-dir path/to/training/image_2 --labels-dir path/to/training/label_2 --output datasets/kitti-yolo
    python scripts/check_kitti_yolo.py --data datasets/kitti-yolo

The dataset and all generated caches are excluded by .gitignore.

## Installation

The dependency versions are pinned to the environment used for the recorded experiments. PyTorch was tested as the CUDA 12.6 build.

    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip
    python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
    python -m pip install -r requirements.txt

No pretrained weight file is committed. Download or provide the required checkpoint locally.

## Training

Run commands from the repository root.

Baseline:

    python yolov5/train.py --weights yolov5s.pt --cfg yolov5/models/yolov5s.yaml --data data/kitti.yaml --img 640 --batch-size 8 --epochs 100 --project runs --name baseline-100

Final model:

    python yolov5/train.py --weights "" --cfg yolov5/models/yolov5s_ca_deep_c3ghost.yaml --data data/kitti.yaml --img 640 --batch-size 8 --epochs 100 --project runs --name ca-deep-c3ghost-100

Validation:

    python yolov5/val.py --weights path/to/best.pt --data data/kitti.yaml --img 640 --batch-size 8 --device 0

Architecture and complexity smoke check:

    python scripts/check_ca_model.py

The forward benchmark script accepts a local checkpoint and writes an output JSON. It does not modify the checkpoint.

## Experimental results

The main table below is rounded to exactly match the submitted final paper: 640x640 input, batch size 8, 100 epochs, seed 42, and the fixed KITTI-derived split. The final model is selected for a favorable accuracy-complexity trade-off relative to the YOLOv5s baseline; it is not claimed to have the highest accuracy across every ablation, since CA alone reports 0.734 mAP50-95.

| Model | P | R | mAP50 | mAP50-95 | Params | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv5s Baseline | 0.925 | 0.885 | 0.956 | 0.694 | 7.03M | 15.97 |
| CA | 0.935 | 0.901 | 0.965 | 0.734 | 7.04M | 15.97 |
| C3Ghost | 0.937 | 0.873 | 0.954 | 0.702 | 4.90M | 10.60 |
| CA + Full-C3Ghost | 0.930 | 0.875 | 0.953 | 0.695 | 4.91M | 10.60 |
| CA + Backbone-C3Ghost | 0.921 | 0.891 | 0.959 | 0.713 | 5.88M | 12.57 |
| **CA + Deep-C3Ghost** | 0.950 | 0.899 | 0.965 | 0.724 | 5.96M | 14.00 |

The detailed forward benchmark below follows the same protocol reported in the paper; its extra decimal places are measurement detail and are kept separate from the rounded main-results table.

Forward benchmark:

| Model | Params | GFLOPs | Latency | FPS |
|---|---:|---:|---:|---:|
| YOLOv5s Baseline | 7,030,417 | 15.9691 | 5.9062 ms | 169.31 |
| CA + Deep-C3Ghost | 5,962,273 | 14.0021 | 6.2604 ms | 159.73 |

Forward timing uses an NVIDIA RTX 4060 Laptop GPU, FP32, batch size 1, 50 warmup forwards, and 500 timed forwards. Data loading, image transfer, NMS, and drawing are excluded.

## Repository layout

    data/       Public KITTI YAML template
    results/    Sanitized metrics and protocol summaries
    scripts/    KITTI preparation, checks, model smoke test, benchmark helper
    yolov5/     Upstream YOLOv5 source plus project model additions

The excluded items include KITTI data, runs/, virtual environments, IDE settings, caches, logs, temporary files, and model checkpoints.

## Attribution

This work modifies and extends Ultralytics YOLOv5. Please see LICENSE for the retained upstream AGPL-3.0 license and original attribution.
