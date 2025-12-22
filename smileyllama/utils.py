import os
from typing import Dict, Any, List, Dict, Any, Optional
from ruamel.yaml import YAML

from .score import Score


def safe_read_yaml(path: os.PathLike):
    ry = YAML(typ="safe")
    ry.allow_duplicate_keys = False
    with open(path) as f:
        data = ry.load(f)
    return data


def modify_yaml(old_path: os.PathLike, new_path: os.PathLike, args: Dict[str, Any]):
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
        last = int(keys[:-1]) if keys[-1].isdigit() else keys[-1]
        cur[last] = value
    with open(new_path, 'w') as f:
        ry.dump(cfg, f)


def run_score_test(
    score: Score, 
    test_smiles: List[str], 
    nprocs: int = -1, 
    wdir: Optional[os.PathLike] = None, 
    dependency_scores: Dict[str, Score] = dict()
):
    if wdir is not None:
        score.set_working_dir(wdir)
    score.set_nprocs(nprocs)
    for name, other_score in dependency_scores.items():
        score.add_dependency_score(name, other_score)
    return score.compute_batch(test_smiles)