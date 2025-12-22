
from __future__ import annotations

__all__ = ['safe_mol_from_smiles', 'accept_smiles', 'Score']

import os
import math
import inspect
import multiprocessing as mp 
from abc import ABC, abstractmethod
from typing import Any, Union, Literal, Callable, List
from functools import partial, wraps
from pathlib import Path

import numpy as np
from tqdm import tqdm
from rdkit import Chem

from .registry import register_class


def safe_mol_from_smiles(smi: str):
    if (smi == '') or ('*' in smi) or ('.' in smi):
        return None
    if not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        Chem.Kekulize(mol)
        Chem.RemoveHs(Chem.AddHs(mol))
    except Exception as e:
        return None
    return mol


def accept_smiles(func: Callable):
    @wraps(func)
    def wrapper(self, mol: Union[str, Chem.Mol], *args, **kwargs):
        if isinstance(mol, str):
            mol_obj = safe_mol_from_smiles(mol)
            if mol_obj is None:
                return math.nan
        elif isinstance(mol, Chem.Mol):
            mol_obj = mol
        else:
            raise TypeError(
                f"{func.__name__} expects rdkit.Chem.Mol or SMILES string, "
                f"got {type(mol)}"
            )
        return func(self, mol_obj, *args, **kwargs)
    return wrapper


class Score(ABC):
    
    def __init__(self, *args, **kwargs):
        self._nprocs = max(1, mp.cpu_count() - 2)
        self._dependency_scores = {}
        self._wdir = None
    
    @abstractmethod
    def compute(self, mol: Union[str, Chem.Mol]) -> Any:
        ...
    
    def set_nprocs(self, nprocs: int = -1):
        self._nprocs = max(1, mp.cpu_count()-2 if nprocs <= 0 else nprocs)
    
    @property
    def nprocs(self) -> int:
        return self._nprocs
    
    def add_dependency_score(self, name: str, score: Score):
        self._dependency_scores[name] = score
    
    @property
    def dependency_scores(self) -> Dict[str, Score]:
        return self._dependency_scores

    def compute_batch(
        self, 
        mols: List[Union[str, Chem.Mol]]
    ) -> np.ndarray:
        desc = f'Compute {self.__class__.__name__}'
        if self.nprocs == 1:
            scores = np.array([self.compute(m) for m in tqdm(mols, desc=desc)])
        else:
            scores = []
            with mp.Pool(self.nprocs) as p:
                for score in tqdm(p.imap(self.compute, mols), total=len(mols), desc=desc):
                    scores.append(score)
            scores = np.array(scores)
        
        return scores
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        if not inspect.isabstract(cls):
            register_class("score", cls)
    
    @property
    def wdir(self) -> Path:
        assert self._wdir is not None, 'No working directory set. Run set_working_dir first.'
        return self._wdir

    def set_working_dir(self, wdir: os.PathLike, mkdir: bool = True):
        self._wdir = Path(wdir).resolve()
        if mkdir:
            self._wdir.mkdir(exist_ok=True, parents=True)

