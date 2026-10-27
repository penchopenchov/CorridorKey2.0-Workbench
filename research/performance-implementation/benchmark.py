"""Clean paired 4K exports against the actual original runtime, no capture I/O.

Loads original runtime/video_io from Git into an isolated package namespace.
Weights are loaded once and reused equally; cold-equivalent totals add that
same measured load cost. Baseline includes original EXR preparation.
"""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).parent))
import torch
from corridorkey import runtime as fast
import corridorkey_launcher as launcher

FOLDER=Path(__file__).parent
from reference_runtime import load_reference_runtime
baseline=load_reference_runtime()
source=ROOT/'research/performance-2026-10-26/synthetic4k.mp4'
t=time.perf_counter()
model=fast.load_model(fast.RuntimePaths.from_package(ROOT),device='cuda')
torch.cuda.synchronize()
load=time.perf_counter()-t
results={'frames':12,'native_hw':[2160,3840],'fps':24,'hann':[8,4],
         'outputs':['alpha','fg'],'load_seconds':load,'runs':[]}
for iteration in range(2):
    for label,rt in [('baseline',baseline),('optimized',fast)]:
        start=time.perf_counter()
        if label=='baseline':
            prepared=launcher.decode_video(source,FOLDER/'prepared_bench','srgb','auto',12,ROOT/'ffmpeg.exe')
            path=prepared.path
        else:
            path=source
        preparation=time.perf_counter()-start
        cfg=rt.InferenceConfig(hann_chunk=8,hann_stride=4,frame_dir_linear=True,
                               temp_dir=FOLDER/'cache',outputs={'alpha','fg'})
        if label=='optimized':
            cfg.video_decoder='ffmpeg'
        torch.cuda.reset_peak_memory_stats()
        infer_start=time.perf_counter()
        rt.run_inference(model,str(path),str(FOLDER/f'out_bench_{label}_{iteration}'),cfg,
                         rt.WriterConfig(ffmpeg_bin=str(ROOT/'ffmpeg.exe')),
                         initial_hint_path=None,hint_video_path=None)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        record={'label':label,'iteration':iteration,'preparation_seconds':preparation,
                'inference_and_export_seconds':time.perf_counter()-infer_start,
                'total_seconds':elapsed,'cold_equivalent_seconds':elapsed+load,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
        results['runs'].append(record)
        (FOLDER/'benchmark-results.json').write_text(json.dumps(results,indent=2))
        print(record,flush=True)
