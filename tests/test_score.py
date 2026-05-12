import pytest
import os
from rdkit import Chem
from smileyllama.score import REGISTRY
from smileyllama.utils import run_score_test


data_path = os.path.join(os.path.dirname(__file__), 'data')

score_configs = {
    "Similarity": {"molecule": "CN1C=C2C=C(C(=CC2=N1)Cl)NC3=NC(=O)N(C(=O)N3CC4=CC(=C(C=C4F)F)F)CC5=NN(C=N5)C"},
    "SubstructureMatch": {"substruct": "[#6]"}
}

test_smiles = [
    "CN1C=C2C=C(C(=CC2=N1)Cl)NC3=NC(=O)N(C(=O)N3CC4=CC(=C(C=C4F)F)F)CC5=NN(C=N5)COCOC",
    "CCOCc1cccccx1",
    'c1ccccn1',
    'c.ccc',
    'cc[*]'
]

def test_score():
    for name, cls in REGISTRY['rdkit_scores'].items():
        score = cls(**score_configs.get(name, {}))
        print(name, run_score_test(score, test_smiles))
