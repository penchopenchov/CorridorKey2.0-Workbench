"""Load original application code from Git for honest baseline reproduction."""
import importlib
from pathlib import Path
import subprocess
import sys
import types

def load_reference_runtime():
    root=Path(__file__).resolve().parents[2]
    code=Path(__file__).parent/'baseline_code'
    code.mkdir(exist_ok=True)
    for name in ('runtime','video_io'):
        data=subprocess.check_output(['git','show',f'6f987fa:corridorkey/{name}.py'],cwd=root)
        (code/f'{name}.py').write_bytes(data)
    package=types.ModuleType('corridorkey_baseline')
    package.__path__=[str(code),str(root/'corridorkey')]
    sys.modules.setdefault(package.__name__,package)
    return importlib.import_module('corridorkey_baseline.runtime')
