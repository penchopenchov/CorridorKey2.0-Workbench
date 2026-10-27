"""Exercise actual launcher jobs, shared model and low-to-normal transition."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import corridorkey_launcher as launcher
from corridorkey import runtime
FOLDER=Path(__file__).parent
SOURCES=FOLDER/'batch_sources'
SOURCES.mkdir(exist_ok=True)
for name in ('one','two'):
    subprocess.run([str(ROOT/'ffmpeg.exe'),'-v','error','-y','-f','lavfi','-i',
                    'testsrc2=size=128x96:rate=24','-frames:v','4',str(SOURCES/f'{name}.mp4')],check=True)
original=runtime.load_model
loads=[]
def counted(*args,**kwargs):
    value=original(*args,**kwargs)
    loads.append(1)
    return value
runtime.load_model=counted
settings=launcher.Settings(outputs=('alpha','fg'),hann_chunk=4,hann_stride=2,
                           cache_root=SOURCES/'cache',output_root=FOLDER/'out_launcher',low_vram=True)
first=SOURCES/'one.mp4'
model=launcher.run_job(launcher.Job(first,first),settings,ROOT/'ffmpeg.exe')
identity=id(model)
settings.low_vram=False
second=SOURCES/'two.mp4'
model=launcher.run_job(launcher.Job(second,second),settings,ROOT/'ffmpeg.exe',model=model)
result={'model_load_count':len(loads),'same_model':id(model)==identity,
        'low_to_normal_completed':True}
assert result['model_load_count']==1 and result['same_model']
(FOLDER/'launcher-batch-results.json').write_text(json.dumps(result,indent=2))
