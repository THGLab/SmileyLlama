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
    """Abstract base class for data normalization.
    
    Normalizers transform numerical data according to specific rules.
    Subclasses must implement the :meth:`transform` method.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the normalizer.
        
        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        ...
    
    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform the input data.
        
        This method must be implemented by subclasses to define the
        specific normalization logic.

        .. note::
            The input data may contain NaN values. 

            When implementing or calling reduction functions (such as calculating maxima, minima, means, etc.), you should use their `np.nan*` 
            counterparts (e.g., `np.nanmax` instead of `np.max`) to ensure NaNs do not influence the results. 
            
            Additionally, after normalization, any NaN values present in the input should remain as NaN in the output. This will help maintain robustness 
            and consistency when dealing with missing or undefined data points.
        
        Parameters
        ----------
        data : numpy.ndarray
            Input data to normalize.
        
        Returns
        -------
        numpy.ndarray
            Normalized data.
        """
        ...

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Call the normalizer as a function.
        
        Parameters
        ----------
        data : numpy.ndarray
            Input data to normalize.
        
        Returns
        -------
        numpy.ndarray
            Normalized data.
        """
        return self.transform(data)
    
    def __init_subclass__(cls, *args, **kwargs):
        """
        Called when a subclass of Normalizer is defined.

        This hook is used to automatically register all non-abstract
        subclasses of Normalizer in the normalizer registry. If the subclass is
        still abstract (i.e., has unimplemented abstract methods), it will not 
        be registered.
        """
        super().__init_subclass__(*args, **kwargs)
        if not inspect.isabstract(cls):
            register_class("normalizer", cls)


class Identity(Normalizer):
    """Identity normalizer that returns data unchanged.
    
    This normalizer performs no transformation and simply returns
    the input data as-is.
    """
    def __init__(self):
        """Initialize the identity normalizer."""
        super().__init__()
    
    def transform(self, data):
        """Return data unchanged.
        
        Parameters
        ----------
        data : numpy.ndarray
            Input data.
        
        Returns
        -------
        numpy.ndarray
            Input data unchanged.
        """
        return data


class Negate(Normalizer):
    """Normalizer that negates the input data.
    
    This normalizer multiplies all values by -1.
    """
    def __init__(self):
        """Initialize the negate normalizer."""
        super().__init__()
    
    def transform(self, data):
        """Negate the input data.
        
        Parameters
        ----------
        data : numpy.ndarray
            Input data.
        
        Returns
        -------
        numpy.ndarray
            Negated data (multiplied by -1).
        """
        return -data



class MinMaxNormalizer(Normalizer):
    """Min-max normalizer that scales data to ``[0, 1]`` range.
    
    Clips data to optional min/max bounds (if provided), then normalizes to ``[0, 1]`` range.
    The normalization uses the provided bounds if specified, otherwise uses the actual
    min/max of the clipped data. The transform is given by:

    .. code-block:: python

        X_clip = clip(X, vmin, vmax)  # No clipping if vmin/vmax is None
        vmin_norm = vmin if vmin is not None else nanmin(X_clip)
        vmax_norm = vmax if vmax is not None else nanmax(X_clip)
        X_norm = (X_clip - vmin_norm) / (vmax_norm - vmin_norm)

    """
    def __init__(self, vmin: Optional[float] = None, vmax: Optional[float] = None, negate: bool = False):
        """Initialize the min-max normalizer.
        
        Parameters
        ----------
        vmin : float, optional
            Minimum value to clip data to. If None, no lower clipping is performed.
            Default is None.
        vmax : float, optional
            Maximum value to clip data to. If None, no upper clipping is performed.
            Default is None.
        negate : bool, optional
            If True, return ``1 - normalized_value``. Default is False.
        
        Raises
        ------
        AssertionError
            If ``vmin`` > ``vmax`` when both are provided.
        """
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        
        if self.vmin is not None and self.vmax is not None:
            assert self.vmin <= self.vmax, f'Lower bound ({self.vmin}) is larger than upper bound ({self.vmax}) '
        
        self.negate = negate
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to ``[0, 1]`` range.
        
        First clips data to ``[vmin, vmax]`` if provided (no clipping if None).
        Then determines normalization bounds: uses provided ``vmin``/``vmax`` if specified,
        otherwise uses the actual min/max (computed with ``np.nanmin``/``np.nanmax``) of
        the clipped data. Finally normalizes using these bounds.
        
        Parameters
        ----------
        data : numpy.ndarray
            Input data to normalize.
        
        Returns
        -------
        numpy.ndarray
            Normalized data in ``[0, 1]`` range (or ``[1, 0]`` if ``negate=True``).
        """
        data_clipped = np.clip(data, self.vmin, self.vmax)
        vmin = np.nanmin(data_clipped) if self.vmin is None else self.vmin
        vmax = np.nanmax(data_clipped) if self.vmax is None else self.vmax
        data_transformed = (data_clipped - vmin) / (vmax - vmin)
        return data_transformed if not self.negate else 1 - data_transformed


NumericType = Union[int, float]

class StepNormalizer(Normalizer):
    """Step function normalizer that converts values to binary (0 or 1).
    
    Applies a threshold or range check to convert continuous values to
    binary outputs. Supports comparison operators (>, <, =, >=, <=) or
    range checks (between two values).
    """
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
        """Initialize the step normalizer.
        
        Parameters
        ----------
        sign : {'>', '<', '=', '>=', '<=', ''}
            Comparison operator. If empty string, performs range check.
        val : int or float or (int, int) or (float, float)
            Threshold value for comparison operators, or `(min, max)` tuple
            for range check.
        
        Raises
        ------
        RuntimeError
            If sign is empty and val is not a tuple of length 2.
        AssertionError
            If sign is empty and ``val[0] >= val[1]``.
        """
        super().__init__()
        self.sign = sign
        self.val = val
        if not sign:
            try:
                lo, hi = val[0], val[1]
            except Exception as e:
                raise RuntimeError(f"Invalid range: {val}")
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
        """Transform data to binary values based on threshold or range.
        
        For scalar input, returns 1 if condition is met, 0 otherwise.
        For array input, returns array of 1.0 (condition met) or 0.0 (not met).
        NaN values are preserved as NaN.
        
        Parameters
        ----------
        data : numpy.ndarray, int, float, or list of int/float
            Input data to evaluate.
        
        Returns
        -------
        numpy.ndarray
            Binary array (1.0 or 0.0) or scalar (1 or 0) for scalar input.
            NaN values are preserved.
        """
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
