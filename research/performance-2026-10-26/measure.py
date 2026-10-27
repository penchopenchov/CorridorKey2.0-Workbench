"""Research-only synthetic benchmark; does not modify the inference pipeline.

Run from repository root with runtime/python.exe research/performance-2026-10-26/measure.py.
These timings are not matte-quality tests or representative end-to-end exports.
"""
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import numpy as np
import torch
from corridorkey.runtime import RuntimePaths, load_model, select_vram_profile
from corridorkey.video_io import preprocess_video_frame_views

OUT = Path(__file__).with_name("measurements.json")
results = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(), "bf16": torch.cuda.is_bf16_supported(),
           "stage2": [], "cpu_preprocess": []}

def save():
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results), flush=True)

torch.manual_seed(0)
start = time.perf_counter()
model = load_model(RuntimePaths.from_package(ROOT), device="cuda")
torch.cuda.synchronize()
results["load_seconds"] = time.perf_counter() - start
results["parameter_counts"] = {name: sum(p.numel() for p in module.parameters())
                                for name, module in model.named_children()}
results["parameter_dtypes"] = {name: sorted({str(p.dtype) for p in module.parameters()})
                                for name, module in model.named_children()}
model.foundations.cpu()
model.stage1.cpu()
model.stage2.cuda().eval()
torch.cuda.empty_cache()
save()
with torch.inference_mode():
    for native_h, native_w in [(1080, 1920), (2160, 3840)]:
        h, w = ((native_h + 127) // 128 * 128, (native_w + 127) // 128 * 128)
        meta = {"shape_hw": (h,w), "model_native_hw": (native_h,native_w)}
        frame = np.random.default_rng(0).integers(0, 256, (native_h,native_w,3), dtype=np.uint8)
        times = []
        for _ in range(4):
            t = time.perf_counter()
            tensors = preprocess_video_frame_views(frame, meta, need_model=True)
            times.append((time.perf_counter()-t)*1000)
            del tensors
        results["cpu_preprocess"].append({"native_hw": [native_h,native_w],
                                          "ms": times, "warm_median_ms": statistics.median(times[1:])})
        del frame
        for batch in ([1,2] if native_h == 2160 else [1]):
            rgb = torch.randn((1,batch,3,h,w),device="cuda",dtype=torch.bfloat16)
            baton = torch.randn((1,batch,512,h//16,w//16),device="cuda",dtype=torch.float32)
            torch.cuda.reset_peak_memory_stats()
            samples = []
            for _ in range(4):
                torch.cuda.synchronize()
                t = time.perf_counter()
                a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record()
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    y = model.stage2(rgb,baton)
                b.record()
                b.synchronize()
                samples.append({"wall_ms": (time.perf_counter()-t)*1000,
                                "gpu_ms": a.elapsed_time(b)})
                del y
            results["stage2"].append({"native_hw": [native_h,native_w],"padded_hw": [h,w],
                                      "batch": batch,"samples": samples,
                                      "warm_ms_per_frame": statistics.median(s["wall_ms"] for s in samples[1:])/batch,
                                      "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30})
            del rgb,baton
            save()
    results["auto_profiles_4k_long"] = {str(gb): select_vram_profile((2176,3840),240,gb)
                                        for gb in (20,21,22,23,24)}
save()
