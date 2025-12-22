import os, shutil
from pathlib import Path
from smileyllama.score.vina import *
from smileyllama.utils import run_score_test

cfg = [
    ("vina", Vina, 'vina'),
    ("vina_gpu", VinaGPU, 'vina_gpu'),
    ("unidock", UniDock, "/global/home/groups/fc_armada2/conda_envs/unidock/bin/unidock")
]

test_smiles = [
    'c1ccccn1', 'x', 'CN1C=C2C=C(C(=CC2=N1)Cl)NC3=NC(=O)N(C(=O)N3CC4=CC(=C(C=C4F)F)F)CC5=NN(C=N5)C',
    'C' * 150
]

def test_vina():
    protein = Path(__file__).parent / 'data/protein.pdb'
    for tag, cls, exec_path in cfg:
        print(f"===== Testing {tag} =====")
        workdir = Path(__file__).parent / f'_test_{tag}'
        if workdir.is_dir():
            shutil.rmtree(workdir)
        score_obj = cls(protein=protein, box_center=(21.74425, -5.3926, 27.91045), box_size=(25.0, 25.0, 25.0), exec_path=exec_path)
        print(tag, run_score_test(score_obj, test_smiles, wdir=workdir))