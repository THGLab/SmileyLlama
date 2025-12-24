import importlib.util
import sys
from pathlib import Path


def load_module_from_file(path: str):
    """Load a Python module from a file path.
    
    Dynamically loads a Python module from a file, allowing users to
    extend SmileyLlama functionality with custom plugins. The module
    is registered in sys.modules with a unique name based on the file path.
    
    Parameters
    ----------
    path : str
        Path to the Python file to load as a module. Must be a .py file.
    
    Returns
    -------
    module
        The loaded Python module object.
    
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
    """
    path = Path(path).resolve()
    assert path.is_file() and path.suffix == '.py', f'Invalid python file: {path}'
    module_name = f'smileyllama.user_plugin_{hash(path)}'

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module
