"""Short synthetic 4K baseline using the original launcher preparation/runtime.

Timings synchronize at selected boundaries to attribute work; this adds overhead.
No production source edits. Uses 12 frames, Hann 8/4, alpha+fg, default encoder.
"""
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import torch
import corridorkey_launcher as launcher
from corridorkey import runtime as rt
OUT=Path(__file__).parent
results={"frames":12,"native_hw":[2160,3840],"fps":24,"hann_chunk":8,"hann_stride":4,
         "outputs":["alpha","fg"],"boundaries":{}}
def save():
    (OUT/"export-baseline.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
def wrap(obj,name):
    original=getattr(obj,name)
    def timed(*args,**kwargs):
        torch.cuda.synchronize()
        t=time.perf_counter()
        value=original(*args,**kwargs)
        torch.cuda.synchronize()
        results["boundaries"].setdefault(name,[]).append(time.perf_counter()-t)
        save()
        return value
    setattr(obj,name,timed)
total=time.perf_counter()
t=time.perf_counter()
prepared=launcher.decode_video(OUT/"synthetic4k.mp4",OUT/"prepared_exrs","srgb","auto",12,ROOT/"ffmpeg.exe",threads=0)
results["preparation_seconds"]=time.perf_counter()-t
results["prepared_bytes"]=sum(p.stat().st_size for p in prepared.path.iterdir())
save()
t=time.perf_counter()
model=rt.load_model(rt.RuntimePaths.from_package(ROOT),device="cuda")
torch.cuda.synchronize()
results["load_seconds"]=time.perf_counter()-t
for obj,name in [(model.stage1,"forward"),(model.stage2,"forward"),
                 (rt,"precompute_rvm_to_memmap"),(rt,"hann_stage1_stream"),
                 (rt,"_render_predictions_torch")]:
    # Distinguish forwards in the results.
    if name=="forward":
        key="stage1_forward" if obj is model.stage1 else "stage2_forward"
        original=obj.forward
        def timed_forward(*args,_original=original,_key=key,**kwargs):
            torch.cuda.synchronize()
            start=time.perf_counter()
            value=_original(*args,**kwargs)
            torch.cuda.synchronize()
            results["boundaries"].setdefault(_key,[]).append(time.perf_counter()-start)
            save()
            return value
        obj.forward=timed_forward
    else:
        wrap(obj,name)
torch.cuda.reset_peak_memory_stats()
t=time.perf_counter()
rt.run_inference(model,str(prepared.path),str(OUT/"out_baseline"),
                 rt.InferenceConfig(hann_chunk=8,hann_stride=4,frame_dir_linear=True,
                                    temp_dir=OUT/"feature_cache",outputs={"alpha","fg"}),
                 rt.WriterConfig(ffmpeg_bin=str(ROOT/"ffmpeg.exe")),
                 initial_hint_path=None,hint_video_path=None)
results["inference_and_export_seconds"]=time.perf_counter()-t
results["total_seconds_excluding_python_import"]=time.perf_counter()-total
results["peak_allocated_gib"]=torch.cuda.max_memory_allocated()/2**30
save()
print(json.dumps(results,indent=2),flush=True)
