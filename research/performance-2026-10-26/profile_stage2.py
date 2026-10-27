"""Research experiments only: synchronized stage-2 profiling and dtype probes."""
import json
from pathlib import Path
import statistics
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from corridorkey.runtime import RuntimePaths, load_model

OUT = Path(__file__).parent
torch.manual_seed(0)
model = load_model(RuntimePaths.from_package(ROOT), device="cuda")
model.foundations.cpu()
model.stage1.cpu()
model.stage2.cuda().eval()
torch.cuda.empty_cache()
rgb = torch.randn((1,1,3,2176,3840),device="cuda",dtype=torch.bfloat16)
baton = torch.randn((1,1,512,136,240),device="cuda",dtype=torch.float32)
results = {}
with torch.inference_mode():
    for label,dtype,baton_dtype in [("baseline",torch.bfloat16,torch.float32),
                                    ("bf16_baton",torch.bfloat16,torch.bfloat16),
                                    ("fp16_autocast",torch.float16,torch.float32)]:
        bt = baton.to(baton_dtype)
        timings=[]
        for i in range(4):
            torch.cuda.synchronize()
            t=time.perf_counter()
            with torch.autocast("cuda",dtype=dtype):
                y=model.stage2(rgb,bt)
            torch.cuda.synchronize()
            timings.append((time.perf_counter()-t)*1000)
            if label == "baseline" and i==3:
                reference=y.clone()
        results[label]={"wall_ms":timings,"warm_median_ms":statistics.median(timings[1:]),
                        "output_dtype":str(y.dtype),"finite":bool(y.isfinite().all()),
                        "max_abs_logit_delta":float((y.float()-reference.float()).abs().max()),
                        "mean_abs_logit_delta":float((y.float()-reference.float()).abs().mean())}
        del y,bt
        print(label,results[label],flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                          torch.profiler.ProfilerActivity.CUDA]) as prof:
        with torch.autocast("cuda",dtype=torch.bfloat16):
            y=model.stage2(rgb,baton)
        torch.cuda.synchronize()
    table=prof.key_averages().table(sort_by="self_cuda_time_total",row_limit=35)
    (OUT/"stage2-profile.txt").write_text("\n".join(line.rstrip() for line in table.splitlines())+"\n",encoding="utf-8")
    results["profile_ops"]=[{"key":e.key,"count":e.count,
                              "self_device_time_us":e.self_device_time_total,
                              "device_time_us":e.device_time_total,
                              "self_cpu_time_us":e.self_cpu_time_total}
                             for e in prof.key_averages()]
(OUT/"stage2-experiments.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
print((OUT/"stage2-profile.txt").read_text(),flush=True)
