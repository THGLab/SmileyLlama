import numpy as np
from typing import List, Union

from .base import *
from .rdkit_props import *
from .iminer_props import *
from .vina import *
from .normalizer import *

from .registry import REGISTRY


def aggregate(
    numeric_scores: Union[np.ndarray, List[np.ndarray]],
    binary_scores: Union[np.ndarray, List[np.ndarray]],
    weights: np.ndarray
):

    numeric_scores = np.array(numeric_scores)
    numeric_scores[np.isnan(numeric_scores)] = 0.0
    weights = np.array(weights).reshape(-1, 1)
    weights /= np.sum(weights)
    numeric_scores = np.sum(numeric_scores * weights, axis=0)

    binary_scores = np.array(binary_scores)
    binary_scores[np.isnan(binary_scores)] = 0.0
    binary_scores = np.prod(binary_scores, axis=0).astype(np.bool).astype(numeric_scores.dtype)

    total = numeric_scores * binary_scores

    return total
