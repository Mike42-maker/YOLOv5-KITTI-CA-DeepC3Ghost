"""Build and smoke-test the YOLOv5s + Coordinate Attention model without training."""

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = PROJECT_ROOT / "yolov5"
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

from models.yolo import Model as BaselineModel
from models.yolo_ca import ModelCA

try:
    import thop
except ImportError as exc:
    raise SystemExit("thop is required for the FLOPs check") from exc


def count_flops(model, image):
    model.eval()
    with torch.no_grad():
        macs, _ = thop.profile(model, inputs=(image,), verbose=False)
    return int(macs * 2), int(macs)


def summarize(model, image):
    model.eval()
    with torch.no_grad():
        predictions, raw = model(image)
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "prediction_shape": list(predictions.shape),
        "raw_output_shapes": [list(t.shape) for t in raw],
    }


def main():
    cfg = YOLO_ROOT / "models" / "yolov5s_ca.yaml"
    baseline_cfg = YOLO_ROOT / "models" / "yolov5s.yaml"
    image = torch.zeros(1, 3, 640, 640)

    ca_model = ModelCA(cfg, nc=4)
    ca_summary = summarize(ca_model, image)
    ca_flops, ca_macs = count_flops(ca_model, image)

    baseline_model = BaselineModel(baseline_cfg, nc=4)
    baseline_summary = summarize(baseline_model, image)
    baseline_flops, baseline_macs = count_flops(baseline_model, image)

    result = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "input_shape": list(image.shape),
        "ca": {**ca_summary, "flops": ca_flops, "macs": ca_macs},
        "baseline": {**baseline_summary, "flops": baseline_flops, "macs": baseline_macs},
        "ca_locations": {
            "P3": "layer 18, after layer 17 PANet P3/8 C3 output",
            "P4": "layer 22, after layer 21 PANet P4/16 C3 output",
            "detect_inputs": [18, 22, 25],
        },
    }
    output = PROJECT_ROOT / "results" / "ca_model_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

