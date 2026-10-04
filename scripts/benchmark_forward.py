"""Standalone, read-only checkpoint benchmark; run_all_benchmarks.ps1 orchestrates environments."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hash_stream(stream)


def hash_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-pattern", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--family", choices=["v5", "v7", "v8"], required=True)
    args = parser.parse_args()
    project = Path(args.project).resolve()
    candidates = sorted(project.glob(args.run_pattern))
    candidates = [p for p in candidates if p.is_file() and p.name == "best.pt"]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one matching best.pt, found {candidates}")
    weights = candidates[0]
    before_hash = sha256(weights)
    os.environ["YOLOv5_AUTOINSTALL"] = "false"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_CONFIG_DIR"] = str(Path(args.output).resolve().parent / "ultralytics_config")
    sys.path.insert(0, str(project / "yolov5" if args.family == "v5" else project))
    import torch
    from thop import profile
    if args.family in ("v5", "v8"):
        import ultralytics  # Fail before legacy imports can try auto-installing it.
        assert hasattr(ultralytics, "__version__")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in the existing project environment")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(42)
    checkpoint = torch.load(str(weights), map_location="cpu", weights_only=False)
    source = "ema" if checkpoint.get("ema") is not None else "model"
    model = checkpoint[source].float().eval().to("cuda:0")
    params = sum(p.numel() for p in model.parameters())
    bn_layers = sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    if any(getattr(m, "include_nms", False) or getattr(m, "end2end", False) for m in model.modules()):
        raise RuntimeError("Checkpoint has NMS/end-to-end enabled; refusing to alter model")
    x = torch.rand(1, 3, 640, 640, device="cuda:0", dtype=torch.float32)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    latencies = []
    print(f"Benchmarking {args.name}: {weights}", flush=True)
    with torch.inference_mode():
        for _ in range(50):
            model(x)
        torch.cuda.synchronize()
        for iteration in range(500):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            output = model(x)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            latencies.append(elapsed_ms)
            del output
            if (iteration + 1) % 100 == 0:
                print(f"  {iteration + 1}/500 forwards", flush=True)
        # THOP hooks affect only a disposable copy, after all timed forwards.
        profiled_model = copy.deepcopy(model)
        macs, _ = profile(profiled_model, inputs=(x,), verbose=False)
        del profiled_model
    after_hash = sha256(weights)
    assert before_hash == after_hash, "Checkpoint changed during benchmark"
    latency = statistics.mean(latencies)
    result = {
        "model": args.name, "best_pt": str(weights), "params": params,
        "gflops_640": float(macs * 2 / 1e9), "latency_ms": latency,
        "fps": 1000 / latency, "latencies_ms": latencies,
        "python": sys.executable, "torch": torch.__version__, "cuda": torch.version.cuda,
        "thop": __import__("thop").__version__,
        "ultralytics": getattr(sys.modules.get("ultralytics"), "__version__", None),
        "gpu": torch.cuda.get_device_name(0), "imgsz": 640, "batch": 1,
        "precision": "FP32", "tf32": False, "amp": False,
        "warmup": 50, "forwards": 500, "checkpoint_source": source,
        "extra_fusion": False, "batchnorm_layers": bn_layers,
        "cudnn_benchmark": False, "torch_threads": torch.get_num_threads(),
        "checkpoint_sha256_before": before_hash, "checkpoint_sha256_after": after_hash,
        "flops_convention": "THOP module operations, 1 MAC = 2 FLOPs; functional operations may be omitted",
        "timing": "perf_counter_ns with torch.cuda.synchronize before and after each forward",
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"RESULT: Params={params:,}, GFLOPs={result['gflops_640']:.4f}, Latency={latency:.4f} ms, FPS={result['fps']:.4f}", flush=True)


if __name__ == "__main__":
    main()
