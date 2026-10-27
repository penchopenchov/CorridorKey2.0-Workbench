"""Research-only full stage-2 comparison of SDPA implementations."""
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from corridorkey.runtime import RuntimePaths,load_model

torch.manual_seed(0)
model=load_model(RuntimePaths.from_package(ROOT),device="cuda")
model.foundations.cpu()
model.stage1.cpu()
model.stage2.cuda().eval()
torch.cuda.empty_cache()
results={}
with torch.inference_mode():
    rgb=torch.randn((1,1,3,2176,3840),device="cuda",dtype=torch.bfloat16)
    baton=torch.randn((1,1,512,136,240),device="cuda",dtype=torch.float32)
    reference=None
    for name,backend in [("efficient",SDPBackend.EFFICIENT_ATTENTION),
                          ("cudnn",SDPBackend.CUDNN_ATTENTION)]:
        record={}
        try:
            times=[]
            with sdpa_kernel(backend):
                for _ in range(4):
                    torch.cuda.synchronize()
                    t=time.perf_counter()
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        y=model.stage2(rgb,baton)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter()-t)*1000)
            if reference is None:
                reference=y.clone()
            record.update(wall_ms=times,warm_median_ms=statistics.median(times[1:]),
                          finite=bool(y.isfinite().all()),
                          max_abs_delta=float((y-reference).abs().max()),
                          mean_abs_delta=float((y-reference).abs().mean()))
        except Exception as e:
            record["error"]=f"{type(e).__name__}: {e}"
        results[name]=record
        print(name,record,flush=True)
Path(__file__).with_name("backend-stage2.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
