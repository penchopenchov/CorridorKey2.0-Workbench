"""Freeze the previously shipped pipeline for paired optimization checks."""
import importlib
from pathlib import Path
import subprocess
import sys
import types

REVISION = 'f952e61cf4b4ef2b8676b3727aa9f3a72707c42c'

def load_reference():
    root = Path(__file__).resolve().parents[2]
    code = Path(__file__).parent / 'baseline_code' / 'micro'
    code.mkdir(parents=True, exist_ok=True)
    for name in ('runtime', 'streaming', 'resident', 'color_utils', 'video_io'):
        data = subprocess.check_output(['git', 'show', f'{REVISION}:corridorkey/{name}.py'], cwd=root)
        (code / f'{name}.py').write_bytes(data)
    package = types.ModuleType('corridorkey_micro_reference')
    package.__path__ = [str(code), str(root / 'corridorkey')]
    sys.modules.setdefault(package.__name__, package)
    return importlib.import_module(package.__name__ + '.runtime')
