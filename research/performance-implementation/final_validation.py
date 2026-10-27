"""Final exact-output check followed by two clean exports with the same model."""
import importlib
import json
from pathlib import Path
import sys
import time
FOLDER=Path(__file__).parent
ROOT=FOLDER.parents[1]
sys.path.insert(0,str(FOLDER))
sys.argv=['validate_export.py','compare','--stream','--input',
          str(ROOT/'research/performance-2026-10-26/synthetic4k.mp4'),'--label','final']
v=importlib.import_module('validate_export')
assert v.result['changed_values']==0 and v.result['frames']==12
v.rt._render_predictions_torch=v.original
runs=[]
for i in range(2):
    start=time.perf_counter()
    v.rt.run_inference(v.model,v.args.input,str(FOLDER/f'out_final_speed_{i}'),v.cfg,
                       v.rt.WriterConfig(ffmpeg_bin=str(ROOT/'ffmpeg.exe')),
                       initial_hint_path=None,hint_video_path=None)
    elapsed=time.perf_counter()-start
    runs.append({'iteration':i,'export_seconds':elapsed,
                 'cold_equivalent_seconds':elapsed+v.load,
                 'peak_allocated_gib':v.torch.cuda.max_memory_allocated()/2**30})
    (FOLDER/'final-benchmark.json').write_text(json.dumps({'load_seconds':v.load,'runs':runs},indent=2))
    print(runs[-1],flush=True)
