"""Paired complete exports versus the previous shipped default pipeline.

Reuses one loaded model, alternates order, keeps encoder settings identical,
and compares the actual encoded files. Input loops retain source timestamps
at 24 fps and exercise more Hann windows than the original 12-frame fixture.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
FOLDER=Path(__file__).resolve().parent
ROOT=FOLDER.parents[1]
sys.path[:0]=[str(ROOT),str(FOLDER)]
import torch
from corridorkey import runtime as current
from micro_reference import load_reference, REVISION

reference=load_reference()
source=FOLDER/'micro48.mp4'
subprocess.run([str(ROOT/'ffmpeg.exe'),'-v','error','-y','-stream_loop','3',
    '-i',str(ROOT/'research/performance-2026-10-26/synthetic4k.mp4'),
    '-frames:v','48','-c','copy',str(source)],check=True)
t=time.perf_counter()
model=current.load_model(current.RuntimePaths.from_package(ROOT),device='cuda')
torch.cuda.synchronize()
result={'reference_revision':REVISION,'frames':48,'resolution':[3840,2160],
        'outputs':['alpha','fg','checker','cutout'],'load_seconds':time.perf_counter()-t,'runs':[]}

def hashes(folder):
    result={}
    for path in folder.iterdir():
        if path.suffix in ('.mp4','.mov'):
            with path.open('rb') as stream:
                result[path.suffix+'_'+path.stem.split('micro48_')[-1]]=hashlib.file_digest(stream,'sha256').hexdigest()
    return result

expected=None
for i,label in enumerate(('before','after','after','before')):
    module=reference if label=='before' else current
    cfg=module.InferenceConfig(hann_chunk=8,hann_stride=4,video_decoder='ffmpeg',input_color='srgb',
        temp_dir=FOLDER/'cache',outputs=set(result['outputs']))
    folder=FOLDER/f'out_micro_export_{i}_{label}'
    torch.cuda.synchronize()
    start=time.perf_counter()
    module.run_inference(model,str(source),str(folder),cfg,
        module.WriterConfig(ffmpeg_bin=str(ROOT/'ffmpeg.exe')),
        initial_hint_path=None,hint_video_path=None)
    torch.cuda.synchronize()
    row={'implementation':label,'export_seconds':time.perf_counter()-start,
         'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'sha256':hashes(folder)}
    assert len(row['sha256'])==4, 'Missing encoded output'
    if expected is None:
        expected=row['sha256']
    assert row['sha256']==expected, 'Encoded outputs changed'
    result['runs'].append(row)
    (FOLDER/'micro-export-benchmark.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(row),flush=True)
result['all_encoded_files_identical']=True
(FOLDER/'micro-export-benchmark.json').write_text(json.dumps(result,indent=2))
