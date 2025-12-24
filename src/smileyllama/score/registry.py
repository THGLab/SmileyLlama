'''Scores/Normalizer Registry'''
from typing import Optional

REGISTRY = {
    "score": {},
    "normalizer": {},
    "rdkit_scores": {}, # only used in unittests
}
"""
Global registry for scores and normalizers. Classes can be registered and later retrieved from this registry by their key.

:meta hide-value:

.. code-block:: python

    REGISTRY = {
        "score": {},        # For Score classes, keyed by name.
        "normalizer": {},   # For Normalizer classes, keyed by name.
        "rdkit_scores": {}, # For unittests and RDKit-based scores, keyed by name.
    }
"""


def register_class(type: str, cls, name: Optional[str] = None):
    """Register a class in the registry.
    
    Registers a class (typically a Score or Normalizer subclass) in the
    appropriate registry dictionary. The class can be accessed later by
    its name.
    
    Parameters
    ----------
    type : str
        Type of registry to register in. Must be one of: "score", "normalizer",
        or "rdkit_scores".
    cls : type
        Class to register.
    name : str, optional
        Name to register the class under. If None, uses the class's ``__name__``.
        Default is None.
    
    Raises
    ------
    KeyError
        If `type` is not a valid registry name, or if `name` already exists
        in the registry.
    """
    if type not in REGISTRY:
        raise KeyError(f"Invalid resitry name: {type}")
    name = cls.__name__ if name is None else name
    if name in REGISTRY[type]:
        raise KeyError(f"{name} already exists. Pick a different class name.")
    REGISTRY[type][name] = cls


def register(type: str, name: Optional[str] = None):
    """Decorator to register a class in the registry.
    
    A convenience decorator that automatically registers a class when it
    is defined. This is typically used as a class decorator.
    
    Parameters
    ----------
    type : str
        Type of registry to register in. Must be one of: "score", "normalizer",
        or "rdkit_scores".
    name : str, optional
        Name to register the class under. If None, uses the class's ``__name__``.
        Default is None.
    
    Returns
    -------
    callable
        Decorator function that registers the class and returns it unchanged.
    
    Examples
    --------
    >>> @register("score")
    ... class MyScore(Score):
    ...     pass
    """
    def wrapper(cls):
        register_class(type, cls, name)
        return cls

    return wrapper 
