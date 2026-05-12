import importlib.util
import os, sys
from pathlib import Path
import shutil
import hashlib


PLUGIN_DIR = Path(
    os.environ.get('SL_PLUGIN_DIR', '~/.smileyllama_plugins')
).expanduser().resolve()
PLUGIN_DIR.mkdir(exist_ok=True)
sys.path.append(str(PLUGIN_DIR))


def md5_file(path: str | Path, chunk_size: int = 8192) -> str:
    """Compute md5 for a file based on its content"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module_from_file(path: str):
    """Load a Python module from a file path.
    
    Dynamically loads a Python module from a file, allowing users to
    extend SmileyLlama functionality with custom plugins. The plugin file
    is copied to the plugin directory (``~/.smileyllama_plugins`` by default,
    or the directory specified by ``SL_PLUGIN_DIR`` environment variable)
    and registered in :data:`sys.modules` with a unique name based on the MD5 hash
    of the file content.
    
    Parameters
    ----------
    path : str
        Path to the Python file to load as a module. Must be a .py file.
    
    Returns
    -------
    module
        The loaded Python module object, registered with a name like
        ``plugin_{md5_hash}``.
    
    Raises
    ------
    AssertionError
        If the path does not exist, is not a file, or does not have
        a ``.py`` extension.
    ImportError
        If the module cannot be loaded or executed.
    
    Examples
    --------
    Load a custom plugin that defines new Score classes:
    
    >>> plugin = load_module_from_file("my_plugin.py")
    >>> # Plugin classes are now registered and available
    """
    module_name = f'plugin_{md5_file(path)}'
    shutil.copyfile(path, PLUGIN_DIR / f'{module_name}.py')
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module
