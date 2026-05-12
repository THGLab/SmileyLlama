import os, shutil
from pathlib import Path

from smileyllama.score.clogs import CLogS
from smileyllama.utils import run_score_test

cfg = [
    ("clogs", CLogS, '/global/scratch/users/gbalteri/CACHE/calc_cLogS/clogs_alteri'),
]

test_smiles = [
    'c1ccccn1', 'x', 'CN1C=C2C=C(C(=CC2=N1)Cl)NC3=NC(=O)N(C(=O)N3CC4=CC(=C(C=C4F)F)F)CC5=NN(C=N5)C',
    'C' * 150
]


def test_clogs():
    for tag, cls, exec_path in cfg:
        print(f"===== Testing {tag} =====")
        workdir = Path(__file__).parent / f'_test_{tag}'
        if workdir.is_dir():
            shutil.rmtree(workdir)
        score_obj = cls(exec_path=exec_path)
        print(tag, run_score_test(score_obj, test_smiles, wdir=workdir))
