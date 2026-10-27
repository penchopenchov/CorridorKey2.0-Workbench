"""Alternating 4K timings against pre-change attention, including graph I/O.

Graph replay is an isolated spike; no graph backend is installed in runtime.
"""
import ast
from functools import lru_cache
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from corridorkey import blocks
from corridorkey.runtime import RuntimePaths,load_model

source=subprocess.check_output(['git','show','9984b58:corridorkey/blocks.py'],cwd=ROOT).decode()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_sdpa_with_region_groups')
namespace={'torch':torch,'F':F}
exec(compile(ast.Module(body=[node],type_ignores=[]),'<9984b58 grouped attention>','exec'),namespace)
before=namespace['_sdpa_with_region_groups']
@lru_cache(maxsize=256)
def cached_index(values,device):
    return torch.tensor(values,device=device,dtype=torch.long)
# The cache was rejected for production: its measured gain is below noise.
# Keep the spike here so graph-capture and speed results remain reproducible.
cached_source=ast.get_source_segment(source,node).replace(
    'torch.tensor(base, device=q.device, dtype=torch.long)','cached_index(base,q.device)').replace(
    'torch.tensor(token_tuple, device=q.device, dtype=torch.long)','cached_index(token_tuple,q.device)')
cached_namespace={'torch':torch,'F':F,'cached_index':cached_index}
exec(compile(cached_source,'<experimental cached indices>','exec'),cached_namespace)
cached=cached_namespace['_sdpa_with_region_groups']

torch.manual_seed(27)
model=load_model(RuntimePaths.from_package(ROOT),device='cuda')
model.foundations.cpu()
model.stage1.cpu()
model.stage2.cuda().eval()
torch.cuda.empty_cache()
rgb=torch.randn(1,1,3,2176,3840,device='cuda',dtype=torch.bfloat16)
baton=torch.randn(1,1,512,136,240,device='cuda')
outpath=Path(__file__).with_name('gpu-replay-results.json')
result={'baseline_revision':'9984b58','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'scope':'Stage 2; synthetic changing inputs; other models on CPU',
        'shape_hw':[2176,3840],'runs':{k:[] for k in ('before','cached_indices','cuda_graph')},
        'comparisons':[]}

def eager(function):
    blocks._sdpa_with_region_groups=function
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        return model.stage2(rgb,baton)

with torch.inference_mode():
    eager(before)
    eager(cached)
    torch.cuda.synchronize()
    start=time.perf_counter()
    static_rgb=rgb.clone()
    static_baton=baton.clone()
    side=torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                static_output=model.stage2(static_rgb,static_baton)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            static_output=model.stage2(static_rgb,static_baton)
    torch.cuda.synchronize()
    result['graph_setup_seconds']=time.perf_counter()-start
    result['peak_with_graph_gib']=torch.cuda.max_memory_allocated()/2**30

    def replay():
        static_rgb.copy_(rgb)
        static_baton.copy_(baton)
        graph.replay()
        return static_output.clone()

    functions={'before':lambda:eager(before),'cached_indices':lambda:eager(cached),'cuda_graph':replay}
    names=list(functions)
    for iteration in range(9):
        rgb.mul_(.875).add_(.015625)
        baton.mul_(.875).sub_(.015625)
        predictions={}
        for name in names[iteration%3:]+names[:iteration%3]:
            torch.cuda.synchronize()
            start=time.perf_counter()
            actual=functions[name]()
            torch.cuda.synchronize()
            result['runs'][name].append((time.perf_counter()-start)*1000)
            predictions[name]=actual.cpu().contiguous()
            del actual
        for name in ('cached_indices','cuda_graph'):
            actual=predictions[name]
            expected=predictions['before']
            record={'iteration':iteration,'backend':name,
                    'changed_values':int(torch.count_nonzero(actual.view(torch.uint16)!=expected.view(torch.uint16))),
                    'max_abs':float((actual.float()-expected.float()).abs().max())}
            result['comparisons'].append(record)
            assert record['changed_values']==0,record
        del predictions
        outpath.write_text(json.dumps(result,indent=2))
        print({name:result['runs'][name][-1] for name in names},flush=True)
    result['median_ms']={k:statistics.median(v) for k,v in result['runs'].items()}
    result['index_cache_entries']=cached_index.cache_info().currsize
    outpath.write_text(json.dumps(result,indent=2))
    print(result['median_ms'],flush=True)
blocks._sdpa_with_region_groups=before
