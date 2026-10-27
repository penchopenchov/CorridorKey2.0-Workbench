"""Run GPU verification sequentially, without contaminating export timings."""
from pathlib import Path
import json
import subprocess
import sys

FOLDER=Path(__file__).resolve().parent
ROOT=FOLDER.parents[1]
commands=[
    ['benchmark_micro.py'],
    ['validate_micro_render.py'],
    ['validate_export.py','compare','--stream','--input',
     str(ROOT/'research/performance-2026-10-26/synthetic4k.mp4'),'--label','micro'],
    ['validate_matrix.py'],
]
for args in commands:
    with (FOLDER/(Path(args[0]).stem+'-micro.log')).open('w') as log:
        subprocess.run([sys.executable,str(FOLDER/args[0]),*args[1:]],cwd=ROOT,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    if args[0]=='validate_export.py':
        comparison=json.loads((FOLDER/'micro-compare.json').read_text())
        assert comparison['frames']==12 and comparison['changed_values']==0, comparison
    print('Passed:',args[0],flush=True)
with (FOLDER/'tests-micro-final.log').open('w') as log:
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,
                   stdout=log,stderr=subprocess.STDOUT,check=True)
print('Passed: unittest discovery',flush=True)
