__all__ = [
    'Normalizer', 'Identity', 'MinMaxNormalizer', 'StepNormalizer'
]

from typing import Optional, Dict, Union, Literal, Tuple, List
from abc import ABC, abstractmethod
import inspect
import operator
import numpy as np
import math
from .registry import register_class


class Normalizer(ABC):
    def __init__(self, *args, **kwargs):
        ...
    
    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        ...

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return self.transform(data)
    
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        if not inspect.isabstract(cls):
            register_class("normalizer", cls)


class Identity(Normalizer):

    def __init__(self):
        super().__init__()
    
    def transform(self, data):
        return data


class Negate(Normalizer):

    def __init__(self):
        super().__init__()
    
    def transform(self, data):
        return -data



class MinMaxNormalizer(Normalizer):
    def __init__(self, vmin: Optional[float] = None, vmax: Optional[float] = None, negate: bool = False):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        
        if self.vmin is not None and self.vmax is not None:
            assert self.vmin <= self.vmax, f'Lower bound ({self.vmin}) is larger than upper bound ({self.vmax}) '
        
        self.negate = negate
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        data_transformed = np.clip(data, self.vmin, self.vmax)
        data_range = np.nanmax(data_transformed) - np.nanmin(data_transformed)
        data_transformed = (data_transformed - np.nanmin(data_transformed)) / data_range
        return data_transformed if not self.negate else 1 - data_transformed


NumericType = Union[int, float]

class StepNormalizer(Normalizer):

    OPS = {
        '=': operator.eq,
        '>': operator.gt, '<': operator.lt,
        '>=': operator.ge, '<=': operator.le,
    }

    def __init__(
        self, 
        sign: Literal['>', '<', '=', '>=', '<=', ''],
        val: Union[NumericType, Tuple[NumericType, NumericType]]
    ):
        super().__init__()
        self.sign = sign
        self.val = val
        if not sign:
            try:
                lo, hi = val[0], val[1]
            except Exception as e:
                raise RuntimeError(f"Invalid range: {values}")
            assert lo < hi, f'upper bound is smaller than lower bound'
    
    def __str__(self):
        if self.sign:
            return f'{self.sign} {self.val}'
        else:
            return f'between {self.val[0]} and {self.val[1]}'
    
    def __repr__(self):
        return f'<{self.__class__.__name__} ({str(self)})>'
    
    def transform(
        self, 
        data: Union[np.ndarray, NumericType, List[NumericType]],
    ) -> np.ndarray:
        if isinstance(data, float) or isinstance(data, int):
            if math.isnan(data):
                return data
            return self.val[0] <= data <= self.val[1] if not self.sign else self.OPS[self.sign](data, self.val)
        else:
            data = np.array(data) if not isinstance(data, np.ndarray) else data
            nan_mask = np.isnan(data)
            if self.sign:
                condition = self.OPS[self.sign](data, self.val)
            else:
                condition = np.logical_and(data <= self.val[1], data >= self.val[0])
            data_transformed = np.where(condition, 1.0, 0.0)
            data_transformed = np.where(nan_mask, np.nan, data_transformed)
            return data_transformed
