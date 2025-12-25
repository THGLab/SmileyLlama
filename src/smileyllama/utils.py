import os
import subprocess
import contextlib
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from pathlib import Path
from typing import List, Union, Optional, Tuple

from ruamel.yaml import YAML

if TYPE_CHECKING:
    from .score.base import Score


def safe_read_yaml(path: os.PathLike):
    """Safely read and parse a YAML file.
    
    Reads a YAML file and returns its contents as a Python dictionary.
    Uses safe YAML loading to prevent arbitrary code execution.
    
    Parameters
    ----------
    path : os.PathLike
        Path to the YAML file to read.
    
    Returns
    -------
    dict
        Parsed YAML content as a dictionary.
    
    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    yaml.YAMLError
        If the file contains invalid YAML syntax.
    """
    ry = YAML(typ="safe")
    ry.allow_duplicate_keys = False
    with open(path) as f:
        data = ry.load(f)
    return data


def modify_yaml(old_path: os.PathLike, new_path: os.PathLike, args: Dict[str, Any]):
    """Modify a YAML file by updating nested values.
    
    Reads a YAML file, modifies specified nested values using dot notation
    (e.g., "base_model" or "datasets.0.path"), and writes the modified
    configuration to a new file. Preserves YAML formatting and quotes.
    
    Parameters
    ----------
    old_path : os.PathLike
        Path to the source YAML file.
    new_path : os.PathLike
        Path to write the modified YAML file.
    args : dict
        Dictionary mapping dot-notation paths to new values.
        For example: ``{"base_model": "/path/to/model", "datasets.0.path": "/data"}``
        Numeric keys in paths are treated as list indices.
    
    Raises
    ------
    FileNotFoundError
        If the source file does not exist.
    KeyError
        If a path in args does not exist in the YAML structure.
    yaml.YAMLError
        If the file contains invalid YAML syntax.
    """
    ry = YAML()
    ry.preserve_quotes = True
    with open(old_path) as f:
        cfg = ry.load(f)
    for path, value in args.items():
        keys = path.split('.')
        cur = cfg
        for key in keys[:-1]:
            k = int(key) if key.isdigit() else key
            cur = cur[k]
        last = int(keys[-1]) if keys[-1].isdigit() else keys[-1]
        cur[last] = value
    with open(new_path, 'w') as f:
        ry.dump(cfg, f)


def run_score_test(
    score: Score, 
    test_smiles: List[str], 
    nprocs: Optional[int] = None, 
    wdir: Optional[os.PathLike] = None, 
    dependency_scores: Dict[str, Score] = dict()
):
    """Run a score computation test on a list of SMILES strings.
    
    Configures a :class:`~smileyllama.score.base.Score` instance with
    working directory, number of processes, and dependency scores, then
    computes scores for all test SMILES strings.
    
    Parameters
    ----------
    score : Score
        Score instance to use for computation.
    test_smiles : list of str
        List of SMILES strings to score.
    nprocs : int, optional
        Configure the number of processors to use in the score.
    wdir : os.PathLike, optional
        Working directory for file-based score operations. If None,
        no working directory is set. Default is None.
    dependency_scores : dict of str to Score, optional
        Dictionary of dependency scores to add to the score instance.
        Default is empty dict.
    
    Returns
    -------
    numpy.ndarray
        Array of computed scores, one for each input SMILES string.
    """
    if wdir is not None:
        score.set_working_dir(wdir)
    if nprocs is not None:
        score.set_nprocs(nprocs)
    for name, other_score in dependency_scores.items():
        score.add_dependency_score(name, other_score)
    return score.compute_batch(test_smiles)


class CommandExecuteError(Exception):
    """Exception raised when a command execution fails.
    
    Raised by :func:`safe_run_command` when a subprocess command returns
    a non-zero exit status.
    """
    def __init__(self, msg: str):
        """Initialize the exception.
        
        Parameters
        ----------
        msg : str
            Error message describing the command failure.
        """
        self.msg = msg
    
    def __str__(self):
        return self.msg
    
    def __repr__(self):
        return self.msg


def safe_run_command(cmd, check=True, text=True, capture_output=True, **kwargs):
    """Safely run a shell command with error handling.
    
    Executes a command using subprocess and provides detailed error
    messages if execution fails. Wraps subprocess.CalledProcessError
    in a more informative :exc:`CommandExecuteError`.
    
    Parameters
    ----------
    cmd : str or list
        Command to execute. Can be a string (for shell=True) or list
        of command and arguments.
    check : bool, optional
        If True, raise exception on non-zero exit. Default is True.
    text : bool, optional
        If True, decode stdout/stderr as text. Default is True.
    capture_output : bool, optional
        If True, capture stdout and stderr. Default is True.
    **kwargs
        Additional arguments passed to :func:`subprocess.run`, such as
        shell, cwd, env, etc.
    
    Returns
    -------
    subprocess.CompletedProcess
        Completed process object with returncode, stdout, and stderr.
    
    Raises
    ------
    CommandExecuteError
        If ``check=True`` and command returns non-zero exit status.
    """
    if check:
        try:
            res = subprocess.run(cmd, check=check, text=text, capture_output=capture_output, **kwargs)
        except subprocess.CalledProcessError as e:
            msg = (
                f'Command {cmd} returned non-zero exit status {e.returncode}:\n\n'
                f'  --stdout:\n{e.stdout}\n\n'
                f'  --stderr:\n{e.stderr}\n'
            )
            raise CommandExecuteError(msg)
    else:
        return subprocess.run(cmd, check=check, text=text, capture_output=capture_output, **kwargs)


@contextlib.contextmanager
def set_directory(dirname: os.PathLike, mkdir: bool = False):
    """Context manager to temporarily change working directory.
    
    Changes the current working directory for the duration of the context,
    then restores the original directory when exiting.
    
    Parameters
    ----------
    dirname : os.PathLike
        Directory path to change to. Can be relative or absolute.
    mkdir : bool, optional
        If True, create the directory (and parents) if it doesn't exist.
        Default is False.
    
    Yields
    ------
    Path
        Absolute path of the changed working directory.
    
    Examples
    --------
    >>> with set_directory("some_path", mkdir=True):
    ...     # Current directory is now "some_path"
    ...     do_something()
    >>> # Original directory is restored
    """
    pwd = os.getcwd()
    path = Path(dirname).resolve()
    if mkdir:
        path.mkdir(exist_ok=True, parents=True)
    os.chdir(path)
    yield path
    os.chdir(pwd)