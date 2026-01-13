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
    """Aggregate numeric and binary scores with weights.
    
    Combines multiple numeric scores using weighted averaging and multiple
    binary scores using logical AND (product). The final score is the product
    of the weighted numeric score and the binary score.
    
    Parameters
    ----------
    numeric_scores : numpy.ndarray or list of numpy.ndarray
        Numeric scores to aggregate. Can be a single array or list of arrays.
        NaN values are replaced with 0.0 before aggregation.
    binary_scores : numpy.ndarray or list of numpy.ndarray
        Binary scores (0 or 1) to aggregate. Can be a single array or list of arrays.
        NaN values are replaced with 0.0 before aggregation. All binary scores
        are combined using logical AND (product).
    weights : numpy.ndarray
        Weights for numeric scores. Will be normalized to sum to 1.
        Shape should match the number of numeric score arrays.
    
    Returns
    -------
    numpy.ndarray
        Aggregated scores. Computed as (weighted_sum(numeric_scores)) * (product(binary_scores)).
    """
    numeric_scores = np.array(numeric_scores)
    nanmask = np.isnan(numeric_scores).sum(axis=0) > 0
    numeric_scores[np.isnan(numeric_scores)] = 0.0
    weights = np.array(weights).reshape(-1, 1)
    weights /= np.sum(np.abs(weights))
    numeric_scores = np.sum(numeric_scores * weights, axis=0)
    numeric_scores[nanmask] = 0.0

    binary_scores = np.array(binary_scores)
    binary_scores[np.isnan(binary_scores)] = 0.0
    binary_scores = np.prod(binary_scores, axis=0).astype(np.bool).astype(numeric_scores.dtype)

    total = numeric_scores * binary_scores

    return total
