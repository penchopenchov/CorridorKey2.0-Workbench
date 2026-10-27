"""Actual-model equivalence across hints, HDR, irregular tails and memory modes."""
import json
import os
from pathlib import Path
import sys
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
from corridorkey import runtime as rt

FOLDER=Path(__file__).parent
SOURCE=FOLDER/'matrix_source'
HINTS=FOLDER/'matrix_hints'
SOURCE.mkdir(exist_ok=True)
HINTS.mkdir(exist_ok=True)
rng=np.random.default_rng(12)
for j in range(13):
    cv2.imwrite(str(SOURCE/f'{j:03d}.exr'),rng.uniform(-.1,4,(73,101,3)).astype(np.float32))
    cv2.imwrite(str(HINTS/f'{j:03d}.png'),rng.integers(0,256,(73,101),dtype=np.uint8))
model=rt.load_model(rt.RuntimePaths.from_package(ROOT),device='cuda')
original=rt._render_predictions_exr_torch
collected=[]
def capture(meta,outputs,checker,logits,**kwargs):
    collected.extend(x.detach().cpu().clone() for x in logits)
    return original(meta,outputs,checker,logits,**kwargs)
rt._render_predictions_exr_torch=capture
results=[]
writer=rt.WriterConfig(ffmpeg_bin=str(ROOT/'ffmpeg.exe'))
def run(label,resident,low=False,first=None,full=None,carry=True):
    collected.clear()
    cfg=rt.InferenceConfig(hann_chunk=8,hann_stride=4,frame_dir_linear=True,
        temp_dir=FOLDER/'cache',outputs={'alpha','fg','checker','cutout'},
        output_format='exr',exr_dtype='float',resident_buffers=resident,low_vram=low,
        carry_hint=carry)
    rt.run_inference(model,str(SOURCE),str(FOLDER/f'out_{label}'),cfg,writer,
        initial_hint_path=first,hint_video_path=full)
    assert cfg.low_vram==low and cfg.hann_chunk==8
    return torch.stack(collected)
# Staged job followed by normal jobs proves placement/chunk settings reset.
low=run('low_memory',True,low=True)
results.append({'case':'low_memory','frames':len(low),'finite':bool(low.isfinite().all())})
del low
for label,first,full,carry in [('none',None,None,True),
                             ('first',str(HINTS/'000.png'),None,True),
                             ('full',None,str(HINTS),True),
                             ('full_authoritative',None,str(HINTS),False)]:
    ref=run(label+'_reference',False,first=first,full=full,carry=carry)
    fast=run(label+'_resident',True,first=first,full=full,carry=carry)
    delta=(ref.float()-fast.float()).abs()
    results.append({'case':label,'frames':len(fast),'bit_exact':torch.equal(ref,fast),
                    'max_abs':float(delta.max()),'finite':bool(fast.isfinite().all())})
    print(results[-1],flush=True)
(FOLDER/'matrix-results.json').write_text(json.dumps(results,indent=2))
assert all(x.get('bit_exact',True) and x['finite'] and x['frames']==13 for x in results)
