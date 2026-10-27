"""Actual guided jobs: all speed options, hints, HDR EXRs and model reuse."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import corridorkey_launcher as launcher
from corridorkey import runtime

FOLDER=Path(__file__).parent
DATA=FOLDER/'speed_launcher_source'
DATA.mkdir(exist_ok=True)
video=DATA/'clip.mp4'
subprocess.run([str(ROOT/'ffmpeg.exe'),'-v','error','-y','-f','lavfi','-i',
    'testsrc2=size=128x96:rate=24','-frames:v','4','-c:v','libx264',str(video)],check=True)
sequence=DATA/'hdr'
hints=DATA/'hints'
video_hints=DATA/'video_hints'
for folder in (sequence,hints,video_hints):
    folder.mkdir(exist_ok=True)
rng=np.random.default_rng(61)
for i in range(13):
    cv2.imwrite(str(sequence/f'{i:03d}.exr'),rng.uniform(-.1,4,(73,101,3)).astype(np.float32))
    cv2.imwrite(str(hints/f'{i:03d}.png'),np.full((73,101),170,np.uint8))
for i in range(4):
    cv2.imwrite(str(video_hints/f'{i:03d}.png'),np.full((96,128),170,np.uint8))

original=runtime.load_model
loads=[]
def counted(*args,**kwargs):
    loads.append(1)
    return original(*args,**kwargs)
runtime.load_model=counted
model=None
identity=None
records=[]

def run(label,source,hint=None,**options):
    global model,identity
    settings=launcher.Settings(hann_chunk=8,hann_stride=4,cache_root=DATA/'cache',
        output_root=FOLDER/'out_speed_launcher'/label,**options)
    model=launcher.run_job(launcher.Job(source,source,hint),settings,ROOT/'ffmpeg.exe',model=model)
    if identity is None:
        identity=id(model)
    assert id(model)==identity
    folder=settings.output_root/f'out_{source.stem}'
    performance=json.loads(next(folder.glob('*_performance.json')).read_text())
    assert performance['frames']==(13 if source.is_dir() else 4)
    assert performance['stage2_attention']==settings.stage2_attention
    assert performance['stage2_native_convolutions']==settings.stage2_native_convolutions
    files={}
    for name in settings.outputs:
        if settings.output_format=='exr':
            directory=next(p for p in folder.iterdir() if p.is_dir() and p.name.endswith('_exr') and f'_{name}' in p.name)
            frames=sorted(directory.glob('*.exr'))
            assert len(frames)==13,(directory,len(frames))
            data=cv2.imread(str(frames[0]),cv2.IMREAD_UNCHANGED)
            assert data.shape[:2]==(73,101) and np.isfinite(data).all()
            channels=1 if data.ndim==2 else data.shape[-1]
            assert channels==({'alpha':1,'cutout':4}.get(name,3)),(name,data.shape)
        else:
            path=next(p for p in folder.iterdir() if p.suffix in ('.mp4','.mov') and f'_{name}' in p.name)
            cap=cv2.VideoCapture(str(path))
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==4
            assert abs(cap.get(cv2.CAP_PROP_FPS)-24)<.001
            cap.release()
            with path.open('rb') as stream:
                files[name]=hashlib.file_digest(stream,'sha256').hexdigest()
    records.append({'case':label,'completed':True,'frames':performance['frames'],
                    'settings':options,'video_sha256':files})
    return files

baseline=run('reference_before',video,stage2_attention='default')
run('default_cudnn',video)
run('all_speed_full_hint',video,video_hints,hint_mode='full',authoritative=True,
    preprocess_device='cuda',cache_foundations=True,stage2_native_convolutions=True,
    cutout_linear=True,despill_strength=.7)
run('low_video_all_speed',video,video_hints,hint_mode='full',authoritative=True,
    low_vram=True,preprocess_device='cuda',cache_foundations=True,stage2_native_convolutions=True)
run('low_hdr_first_float',sequence,hints/'000.png',hint_mode='first',input_color='linear',
    low_vram=True,output_format='exr',exr_dtype='float',sequence_fps=23.976,
    stage2_native_convolutions=True,preprocess_device='cuda')
run('hdr_full_half',sequence,hints,hint_mode='full',authoritative=True,input_color='linear',
    output_format='exr',exr_dtype='half',sequence_fps=23.976,cache_foundations=True)
after=run('reference_after',video,stage2_attention='default')
assert baseline==after,'Reference output changed after speed options / low-VRAM jobs'
assert len(loads)==1 and model.stage1.low_vram_enc_frame_chunk==0
result={'model_loads':len(loads),'same_model':id(model)==identity,
        'reference_encoded_files_identical_before_after':True,'cases':records}
(FOLDER/'speed-launcher-validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
