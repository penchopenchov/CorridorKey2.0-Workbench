"""CPU thread tuning and optional CUDA preprocessing: timing plus exactness."""
import json
import os
from pathlib import Path
import statistics
import sys
import time
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from corridorkey.streaming import preprocess_linear_frame,preprocess_linear_frame_cuda

meta={'shape_hw':(2176,3840),'model_native_hw':(2160,3840)}
# Random signed/HDR plate exposes more rounding boundaries than a test pattern.
frame=np.random.default_rng(81).uniform(-.1,4,(2160,3840,3)).astype(np.float32)
reference=preprocess_linear_frame(frame,meta,need_model=True)
results={'original_threads':torch.get_num_threads(),'runs':[]}
for threads in [1,2,4,8,16]:
    torch.set_num_threads(threads)
    times=[]
    for _ in range(4):
        start=time.perf_counter()
        output=preprocess_linear_frame(frame,meta,need_model=True)
        times.append((time.perf_counter()-start)*1000)
    results['runs'].append({'cpu_threads':threads,'warm_median_ms':statistics.median(times[1:]),
                            'bit_exact':all(torch.equal(a,b) for a,b in zip(output,reference))})
torch.set_num_threads(results['original_threads'])
times=[]
for _ in range(4):
    torch.cuda.synchronize()
    start=time.perf_counter()
    output=preprocess_linear_frame_cuda(frame,meta,need_model=True)
    torch.cuda.synchronize()
    times.append((time.perf_counter()-start)*1000)
results['cuda']={'warm_median_ms':statistics.median(times[1:]),
                 'bit_exact':all(torch.equal(a.cpu(),b) for a,b in zip(output,reference)),
                 'max_abs':max(float((a.cpu().float()-b.float()).abs().max()) for a,b in zip(output,reference))}
Path(__file__).with_name('preprocess-results.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
