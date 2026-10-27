"""Paired 4K preprocessing/rendering checks, with identical input tensors.

Run with embedded Python. Full export timing is separate from this benchmark.
"""
import importlib
import json
from pathlib import Path
import sys
import time
FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
sys.path[:0] = [str(ROOT), str(FOLDER)]
import numpy as np
import torch
from corridorkey import runtime as current
from corridorkey import streaming
from micro_reference import load_reference, REVISION

reference = load_reference()
old_stream = importlib.import_module('corridorkey_micro_reference.streaming')
torch.set_num_threads(8)
torch.manual_seed(16)
meta = {'despill_strength': .5}
# Derive cropping metadata using the actual probe, rather than a made-up layout.
meta.update(current.probe_video_meta(str(ROOT/'research/performance-2026-10-26/synthetic4k.mp4'),
    num_frames=-1,shape_quantum=128,native_resolution=True,start_mode='first',seed=0,
    model_scale=1,linear_brightness=1,linear_contrast=1,linear_contrast_pivot=.18))
frame = np.random.default_rng(16).uniform(-.2, 4, (2160,3840,3)).astype(np.float32)
logits = torch.randn(1,4,2176,3840,device='cuda',dtype=torch.bfloat16)
checker = torch.from_numpy(current.make_checker(2160,3840))
result = {'reference_revision': REVISION, 'preprocess': [], 'render': []}

def resolve(rendered):
    return {k: (v.resolve() if hasattr(v,'resolve') else v) for k,v in rendered.items()}

preprocess_rows = []
for module in (old_stream, streaming):
    views = module.preprocess_linear_frame(frame,meta,need_model=True)
    if module is old_stream:
        golden = views
    else:
        assert all(torch.equal(a,b) for a,b in zip(views,golden))
    preprocess_rows.append({'implementation':module.__name__,'milliseconds':[]})
    del views
for i in range(8):
    for index in ((0,1) if i % 2 == 0 else (1,0)):
        module = (old_stream,streaming)[index]
        t=time.perf_counter()
        views=module.preprocess_linear_frame(frame,meta,need_model=True)
        preprocess_rows[index]['milliseconds'].append((time.perf_counter()-t)*1000)
        del views
result['preprocess']=preprocess_rows
del frame, golden

for outputs in ({'alpha','fg'}, {'alpha','fg','checker','cutout'}):
    golden = resolve(reference._render_predictions_torch(meta,outputs,checker,logits))
    render_rows=[]
    render_kwargs=[]
    for module in (reference,current):
        kwargs = {}
        if module is current and 'checker_linear' in __import__('inspect').signature(module._render_predictions_torch).parameters:
            kwargs['checker_linear'] = module.cu.srgb_to_linear(checker.cuda()).unsqueeze(0) if 'checker' in outputs else None
        actual=resolve(module._render_predictions_torch(meta,outputs,checker,logits,**kwargs))
        assert all(np.array_equal(actual[k],golden[k]) for k in golden)
        del actual
        render_kwargs.append(kwargs)
        render_rows.append({'implementation':module.__name__,'outputs':sorted(outputs),'milliseconds':[]})
    for i in range(12):
        for index in ((0,1) if i % 2 == 0 else (1,0)):
            module=(reference,current)[index]
            torch.cuda.synchronize()
            t=time.perf_counter()
            resolved=resolve(module._render_predictions_torch(meta,outputs,checker,logits,**render_kwargs[index]))
            torch.cuda.synchronize()
            render_rows[index]['milliseconds'].append((time.perf_counter()-t)*1000)
            del resolved
    result['render'].extend(render_rows)
    del golden
result['exact_preprocess_and_pixels'] = True
(FOLDER/'micro-benchmark.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
