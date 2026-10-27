"""Actual 4K exports: separate numerical checks from clean throughput runs."""
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from corridorkey import runtime as rt

FOLDER=Path(__file__).parent
SOURCE=ROOT/'research/performance-2026-10-26/synthetic4k.mp4'
OUT=FOLDER/'speed-choice-results.json'
model=rt.load_model(rt.RuntimePaths.from_package(ROOT),device='cuda')
base=rt.InferenceConfig(hann_chunk=8,hann_stride=4,video_decoder='ffmpeg',input_color='srgb',
    temp_dir=FOLDER/'cache',outputs={'alpha','fg'},stage2_attention='default')
choices={
    'reference':base,
    'cudnn':replace(base,stage2_attention='cudnn'),
    'gpu_preprocess':replace(base,preprocess_device='cuda'),
    'feature_reuse':replace(base,cache_foundations=True),
    'native_convolutions':replace(base,stage2_native_convolutions=True),
    'all_speed_options':replace(base,stage2_attention='cudnn',preprocess_device='cuda',
                               cache_foundations=True,stage2_native_convolutions=True),
}
result={'frames':12,'resolution':[3840,2160],'outputs':['alpha','fg'],
        'quality_scope':'Synthetic test pattern; numerical differences, not perceptual accuracy against ground truth',
        'quality':{},'clean_exports':[]}
original=rt._stage2_forward

def run(label,cfg,suffix):
    start=time.perf_counter()
    rt.run_inference(model,str(SOURCE),str(FOLDER/f'out_choice_{label}_{suffix}'),cfg,
        rt.WriterConfig(ffmpeg_bin=str(ROOT/'ffmpeg.exe')),
        initial_hint_path=None,hint_video_path=None)
    return time.perf_counter()-start

def quality_record():
    return dict(frames=0,raw_changed_values=0,raw_values=0,raw_max_abs=0.,alpha_abs_sum=0.,
        alpha_values=0,alpha_max_abs=0.,alpha_gt_1_over_255=0,alpha_gt_0_01=0,
        fg_srgb_abs_sum=0.,fg_srgb_values=0,fg_srgb_max_abs=0.,fg_srgb_gt_1_over_255=0,
        finite=True,stage2_ms=[])

for label,cfg in choices.items():
    metrics=quality_record()
    def checked(model,rgb,baton,active_cfg):
        torch.cuda.synchronize()
        start=time.perf_counter()
        prediction=original(model,rgb,baton,active_cfg)
        torch.cuda.synchronize()
        metrics['stage2_ms'].append((time.perf_counter()-start)*1000)
        logits=prediction[0] if isinstance(prediction,tuple) else prediction
        for actual in logits[0]:
            bits=np.load(FOLDER/'golden'/f"{metrics['frames']:04d}.npz")['bits']
            reference=torch.from_numpy(bits).view(torch.bfloat16).to(actual.device).float()
            actual=actual.float()
            raw_delta=(actual-reference).abs()
            metrics['raw_changed_values']+=int(torch.count_nonzero(actual!=reference))
            metrics['raw_values']+=actual.numel()
            metrics['raw_max_abs']=max(metrics['raw_max_abs'],float(raw_delta.max()))
            metrics['finite']=metrics['finite'] and bool(actual.isfinite().all())
            a=actual[0,8:-8].clamp(0,1)
            b=reference[0,8:-8].clamp(0,1)
            delta=(a-b).abs()
            metrics['alpha_abs_sum']+=float(delta.double().sum())
            metrics['alpha_values']+=delta.numel()
            metrics['alpha_max_abs']=max(metrics['alpha_max_abs'],float(delta.max()))
            metrics['alpha_gt_1_over_255']+=int(torch.count_nonzero(delta>1/255))
            metrics['alpha_gt_0_01']+=int(torch.count_nonzero(delta>.01))
            def srgb(x):
                linear=rt.cu.asinh_to_linear(x[1:4,8:-8].unsqueeze(0)).clamp(min=0)
                return rt.cu.linear_to_srgb_nonnegative(linear).clamp(0,1)
            delta=(srgb(actual)-srgb(reference)).abs()
            metrics['fg_srgb_abs_sum']+=float(delta.double().sum())
            metrics['fg_srgb_values']+=delta.numel()
            metrics['fg_srgb_max_abs']=max(metrics['fg_srgb_max_abs'],float(delta.max()))
            metrics['fg_srgb_gt_1_over_255']+=int(torch.count_nonzero(delta>1/255))
            metrics['frames']+=1
        return prediction
    rt._stage2_forward=checked
    try:
        metrics['instrumented_export_seconds']=run(label,cfg,'quality')
    finally:
        rt._stage2_forward=original
    metrics['alpha_mae']=metrics['alpha_abs_sum']/metrics['alpha_values']
    metrics['fg_srgb_mae']=metrics['fg_srgb_abs_sum']/metrics['fg_srgb_values']
    metrics['alpha_pct_gt_1_over_255']=100*metrics['alpha_gt_1_over_255']/metrics['alpha_values']
    metrics['alpha_pct_gt_0_01']=100*metrics['alpha_gt_0_01']/metrics['alpha_values']
    metrics['fg_srgb_pct_gt_1_over_255']=100*metrics['fg_srgb_gt_1_over_255']/metrics['fg_srgb_values']
    metrics['stage2_warm_median_ms']=statistics.median(metrics['stage2_ms'][1:])
    assert metrics['frames']==12 and metrics['finite'],metrics
    if label=='reference':
        assert metrics['raw_changed_values']==0,metrics
    result['quality'][label]=metrics
    OUT.write_text(json.dumps(result,indent=2))
    print('Quality:',label,metrics,flush=True)

# Engines/kernels are warm. These timings exclude all numerical comparison work.
# Reference bookends help expose timing drift; no component percentages are added.
for i,label in enumerate(['reference',*list(choices)[1:],'reference']):
    seconds=run(label,choices[label],f'speed_{i}')
    result['clean_exports'].append({'choice':label,'seconds':seconds,
                                   'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
    OUT.write_text(json.dumps(result,indent=2))
    print('Clean export:',result['clean_exports'][-1],flush=True)
