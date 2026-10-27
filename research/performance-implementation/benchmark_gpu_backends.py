"""Stage 2 backend experiments: same real checkpoint, same changing inputs.

No production backend is changed by this script. CUDA Graph is a spike and
must pass separate lifecycle and complete-export tests before being retained.
"""
from contextlib import nullcontext
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from corridorkey.runtime import RuntimePaths,load_model

OUT=Path(__file__).with_name('gpu-backends.json')
parser=argparse.ArgumentParser()
parser.add_argument('--backend')
args=parser.parse_args()
if args.backend is None:
    rows=[]
    summary={'isolated_processes':True,'runs':rows}
    for name in ('eager_default','efficient','cudnn_attention','cudnn_tune','native_convolutions','cuda_graph'):
        with OUT.with_name(f'gpu-backend-{name}.log').open('w') as log:
            child=subprocess.run([sys.executable,__file__,'--backend',name],stdout=log,stderr=subprocess.STDOUT)
        path=OUT.with_name(f'gpu-backend-{name}.json')
        if child.returncode or not path.exists():
            rows.append({'backend':name,'error':f'Child process exited {child.returncode}; see log'})
        else:
            data=json.loads(path.read_text())
            rows.extend(data['runs'])
            summary.update({key:data[key] for key in ('torch','gpu','shape_hw','scope','available','flash_attention')})
        OUT.write_text(json.dumps(summary,indent=2))
        print(rows[-1],flush=True)
    sys.exit(0)
OUT=OUT.with_name(f'gpu-backend-{args.backend}.json')
torch.manual_seed(27)
model=load_model(RuntimePaths.from_package(ROOT),device='cuda')
model.foundations.cpu()
model.stage1.cpu()
stage=model.stage2.cuda().eval()
torch.cuda.empty_cache()
rgb=torch.randn(1,1,3,2176,3840,device='cuda',dtype=torch.bfloat16)
baton=torch.randn(1,1,512,136,240,device='cuda',dtype=torch.float32)
result={'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'shape_hw':[2176,3840],'scope':'Stage 2 only; other model weights on CPU; synthetic inputs',
        'available':{name:importlib.util.find_spec(name) is not None for name in ('triton','onnx','onnxruntime','tensorrt','torch_tensorrt')},
        'flash_attention':torch.backends.cuda.is_flash_attention_available(),'runs':[]}

def save():
    OUT.write_text(json.dumps(result,indent=2))

def forward():
    with torch.autocast('cuda',dtype=torch.bfloat16):
        return stage(rgb,baton)

def compare(a,b):
    delta=(a.float()-b.float()).abs()
    return {'changed_values':int(torch.count_nonzero(a.view(torch.uint16)!=b.view(torch.uint16))),
            'values':a.numel(),'max_abs':float(delta.max()),'finite':bool(a.isfinite().all())}

with torch.inference_mode():
    gold=forward().cpu().contiguous()
    for name in (args.backend,):
        entry={'backend':name}
        static_rgb=static_baton=output=graph=None
        try:
            context=(sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION) if name=='efficient' else
                     sdpa_kernel(SDPBackend.CUDNN_ATTENTION) if name=='cudnn_attention' else
                     torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,benchmark=True,benchmark_limit=8,
                         deterministic=torch.backends.cudnn.deterministic,allow_tf32=torch.backends.cudnn.allow_tf32)
                         if name=='cudnn_tune' else
                     torch.backends.cudnn.flags(enabled=False,benchmark=False,
                         deterministic=torch.backends.cudnn.deterministic,allow_tf32=torch.backends.cudnn.allow_tf32)
                         if name=='native_convolutions' else nullcontext())
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start=time.perf_counter()
            with context:
                if name=='cuda_graph':
                    static_rgb=rgb.clone()
                    static_baton=baton.clone()
                    side=torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(3):
                            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                                output=stage(static_rgb,static_baton)
                    torch.cuda.current_stream().wait_stream(side)
                    torch.cuda.synchronize()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        try:
                            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                                output=stage(static_rgb,static_baton)
                        except Exception:
                            entry['capture_inner_error']=traceback.format_exc()
                            raise
                    def run():
                        static_rgb.copy_(rgb)
                        static_baton.copy_(baton)
                        graph.replay()
                        # Real integration needs independent output lifetime.
                        return output.clone()
                    actual=run()
                else:
                    run=forward
                    actual=run()
                torch.cuda.synchronize()
                entry['first_use_seconds']=time.perf_counter()-start
                entry['initial_comparison']=compare(actual.cpu().contiguous(),gold)
                del actual
                samples=[]
                for _ in range(7):
                    torch.cuda.synchronize()
                    start=time.perf_counter()
                    actual=run()
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter()-start)*1000)
                    del actual
                entry['wall_ms']=samples
                entry['median_ms']=statistics.median(samples)
                entry['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
                # Verify fresh values are consumed on replay, rather than replaying a stale frame.
                rgb.mul_(.5).add_(.03125)
                baton.mul_(.5).sub_(.125)
                actual=run().cpu().contiguous()
            # The changing-input reference always uses the default eager backend.
            changed_gold=forward().cpu().contiguous()
            entry['changed_input_comparison']=compare(actual,changed_gold)
            gold=changed_gold
            del actual
        except Exception as error:
            entry['error']=f'{type(error).__name__}: {error}'
        result['runs'].append(entry)
        save()
        print(json.dumps(entry),flush=True)
        del graph,output,static_rgb,static_baton
        gc.collect()
        torch.cuda.empty_cache()
save()
