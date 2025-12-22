import importlib.util
import sys
from pathlib import Path


def load_module_from_file(path: str):
    path = Path(path).resolve()
    assert path.is_file() and path.suffix == '.py', f'Invalid python file: {path}'
    module_name = f'smileyllama.user_plugin_{hash(path)}'

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module
