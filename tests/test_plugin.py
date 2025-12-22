import pytest
import os
from smileyllama.plugin import load_module_from_file
from smileyllama.score import REGISTRY


def test_plugin():
    load_module_from_file(os.path.join(os.path.dirname(__file__), 'data/plugin.py'))
    score = REGISTRY['score']['RandomScore']()
    print(score.compute_batch(['C', 'CO']))
    norm = REGISTRY['normalizer']['Sigmoid']()
    print(norm(-1))
