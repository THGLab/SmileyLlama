import os
from typing import Dict, Any, List, Optional
from ruamel.yaml import YAML

from .score import Score


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