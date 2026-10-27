"""Isolated SDPA backend feasibility/timing probe, not a model speed claim."""
import json
from pathlib import Path
import statistics
import time
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

torch.manual_seed(0)
results={"flash_built":torch.backends.cuda.is_flash_attention_available(),"cases":[]}
with torch.inference_mode():
    q=torch.randn((135,16,256,80),device="cuda",dtype=torch.bfloat16)
    k,v=torch.randn_like(q),torch.randn_like(q)
    for mask_label,mask in [("none",None),("key_mask",torch.ones((135,1,1,256),device="cuda",dtype=torch.bool))]:
        if mask is not None:
            mask[:,:,:,-16:]=False
        reference=None
        for backend in [SDPBackend.EFFICIENT_ATTENTION,SDPBackend.CUDNN_ATTENTION,SDPBackend.FLASH_ATTENTION]:
            record={"mask":mask_label,"backend":str(backend)}
            try:
                timings=[]
                with sdpa_kernel(backend):
                    for _ in range(4):
                        torch.cuda.synchronize()
                        t=time.perf_counter()
                        y=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask)
                        torch.cuda.synchronize()
                        timings.append((time.perf_counter()-t)*1000)
                if reference is None:
                    reference=y.clone()
                record.update(wall_ms=timings,warm_median_ms=statistics.median(timings[1:]),
                              max_abs_delta=float((y-reference).abs().max()))
            except Exception as e:
                record["error"]=f"{type(e).__name__}: {e}"
            results["cases"].append(record)
            print(record,flush=True)
Path(__file__).with_name("attention-probe.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
