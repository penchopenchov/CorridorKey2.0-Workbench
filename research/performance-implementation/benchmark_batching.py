"""4K Stage 2 batch-size sweep: actual checkpoint, repeated synthetic frame.

This isolates compute and excludes decoding, Stage 1 and encoding. Other model
weights are on CPU, so reported VRAM is not whole-pipeline memory. Identical
input content allows numerical comparison of batching against batch 1.
"""
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from corridorkey.runtime import RuntimePaths,load_model

torch.manual_seed(24)
model=load_model(RuntimePaths.from_package(ROOT),device='cuda')
model.foundations.cpu()
model.stage1.cpu()
model.stage2.cuda().eval()
torch.cuda.empty_cache()
rgb_base=torch.randn(1,1,3,2176,3840,device='cuda',dtype=torch.bfloat16)
baton_base=torch.randn(1,1,512,136,240,device='cuda',dtype=torch.float32)
result={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'native_hw':[2160,3840],'padded_hw':[2176,3840],
        'scope':'Stage 2 only; other models on CPU; synthetic RGB/baton repeated',
        'runs':[]}
output=Path(__file__).with_name('batching-results.json')
reference=None
with torch.inference_mode():
    for batch in (1,2,4,6):
        rgb=rgb_base.expand(1,batch,-1,-1,-1).contiguous()
        baton=baton_base.expand(1,batch,-1,-1,-1).contiguous()
        torch.cuda.reset_peak_memory_stats()
        times=[]
        for _ in range(4):
            torch.cuda.synchronize()
            start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=model.stage2(rgb,baton)
            torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
        peak=torch.cuda.max_memory_allocated()/2**30
        if reference is None:
            reference=prediction[0,0].cpu().contiguous()
        changed=0
        max_abs=0.
        finite=True
        for frame in prediction[0]:
            actual=frame.cpu().contiguous()
            changed+=int(torch.count_nonzero(actual.view(torch.uint16)!=reference.view(torch.uint16)))
            max_abs=max(max_abs,float((actual.float()-reference.float()).abs().max()))
            finite=finite and bool(actual.isfinite().all())
        record={'batch':batch,'wall_ms':times,
                'warm_median_ms_per_frame':statistics.median(times[1:])/batch,
                'peak_allocated_gib':peak,'changed_values':changed,
                'max_abs':max_abs,'finite':finite}
        result['runs'].append(record)
        output.write_text(json.dumps(result,indent=2))
        print(record,flush=True)
        del rgb,baton,prediction,actual
        torch.cuda.empty_cache()
