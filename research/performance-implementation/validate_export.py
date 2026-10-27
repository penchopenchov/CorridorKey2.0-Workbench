"""Capture/compare uncompressed model predictions; timing includes capture I/O.

python validate_export.py capture|compare|speed --input PATH [--stream]
"""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from corridorkey import runtime as rt

ap=argparse.ArgumentParser()
ap.add_argument('mode',choices=['capture','compare','speed'])
ap.add_argument('--input',default=str(ROOT/'research/performance-2026-10-26/prepared_exrs'))
ap.add_argument('--stream',action='store_true')
ap.add_argument('--label',default='baseline')
args=ap.parse_args()
folder=Path(__file__).parent
gold=folder/'golden'
gold.mkdir(exist_ok=True)
original=rt._render_predictions_torch
metrics={'frames':0,'changed_values':0,'max_abs':0.,'sum_abs':0.,'values':0}
def render(meta,outputs,checker,logits):
    data=logits.detach().to('cpu').contiguous().view(torch.uint16).numpy()
    for a in data:
        n=metrics['frames']
        path=gold/f'{n:04d}.npz'
        if args.mode=='capture':
            np.savez_compressed(path,bits=a)
        elif args.mode=='compare':
            b=np.load(path)['bits']
            delta=(torch.from_numpy(a.copy()).view(torch.bfloat16).float()-torch.from_numpy(b.copy()).view(torch.bfloat16).float()).abs()
            metrics['changed_values']+=int(np.count_nonzero(a!=b))
            metrics['max_abs']=max(metrics['max_abs'],float(delta.max()))
            metrics['sum_abs']+=float(delta.sum())
            metrics['values']+=delta.numel()
        metrics['frames']+=1
    return original(meta,outputs,checker,logits)
if args.mode!='speed':
    rt._render_predictions_torch=render
t=time.perf_counter()
model=rt.load_model(rt.RuntimePaths.from_package(ROOT),device='cuda')
torch.cuda.synchronize()
load=time.perf_counter()-t
cfg=rt.InferenceConfig(hann_chunk=8,hann_stride=4,frame_dir_linear=True,
                       temp_dir=folder/'cache',outputs={'alpha','fg'})
if args.stream:
    cfg.video_decoder='ffmpeg'
    cfg.input_color='srgb'
t=time.perf_counter()
rt.run_inference(model,args.input,str(folder/f'out_{args.label}'),cfg,
                 rt.WriterConfig(ffmpeg_bin=str(ROOT/'ffmpeg.exe')),
                 initial_hint_path=None,hint_video_path=None)
result={'label':args.label,'mode':args.mode,'load_seconds':load,'export_seconds':time.perf_counter()-t,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,**metrics}
(folder/f'{args.label}-{args.mode}.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
