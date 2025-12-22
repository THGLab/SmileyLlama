'''Scores/Normalizer Registry'''
from typing import Optional

REGISTRY = {
    "score": {},
    "normalizer": {},
    "rdkit_scores": {}, # only used in unittests
}

def register_class(type: str, cls, name: Optional[str] = None):
    if type not in REGISTRY:
        raise KeyError(f"Invalid resitry name: {type}")
    name = cls.__name__ if name is None else name
    if name in REGISTRY[type]:
        raise KeyError(f"{name} already exists. Pick a different class name.")
    REGISTRY[type][name] = cls


def register(type: str, name: Optional[str] = None):

    def wrapper(cls):
        register_class(type, cls)
        return cls

    return wrapper 
