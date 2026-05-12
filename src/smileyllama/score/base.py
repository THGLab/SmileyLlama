
from __future__ import annotations

__all__ = ['safe_mol_from_smiles', 'accept_smiles', 'Score']

import os
import math
import inspect
import multiprocessing as mp 
from abc import ABC, abstractmethod
from typing import Union, Callable, List, Dict
from functools import wraps
from pathlib import Path

import numpy as np
from tqdm import tqdm
from rdkit import Chem

from .registry import register_class


def safe_mol_from_smiles(smi: str):
    """Safely convert a SMILES string to an RDKit molecule object.

    Attempts to parse the SMILES string and perform basic molecule
    validation and sanitization. Returns None if the SMILES string
    is invalid, contains wildcards, or contains disconnected fragments.

    Parameters
    ----------
    smi : str
        SMILES string to convert to a molecule object.

    Returns
    -------
    rdkit.Chem.Mol or None
        RDKit molecule object if conversion succeeds, None otherwise.
    """
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
    """Decorator to accept both SMILES strings and RDKit molecule objects.

    Wraps a method of :class:`Score` to automatically convert SMILES strings to RDKit
    molecule objects. If a SMILES string cannot be coverted to RDKit molecule with :func:`safe_mol_from_smiles`, 
    returns ``math.nan``. If the input is neither a string nor a ``rdkit.Chem.Mol``, raises
    a TypeError.

    Parameters
    ----------
    func : Callable
        The method function to wrap. Should accept a molecule object as its
        second argument.

    Returns
    -------
    Callable
        Wrapped function that accepts both SMILES strings and molecule
        objects.

    Raises
    ------
    TypeError
        If the input molecule is neither a string nor a rdkit.Chem.Mol.
    """
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
    """Abstract base class for computing scores for molecules.
    
    This class provides a framework for scoring molecules with support
    for batch processing, parallelization, dependency management, and
    working directory configuration. Subclasses must implement the
    :meth:`compute` method to define the specific scoring logic.
   
    In the workflow, Score objects are instantiated and setup from configuration. Their
    :meth:`compute_batch` methods will be called to efficiently evaluate a batch of SMILES strings 
    and returns a numpy array of scores.
    
    By default, :meth:`compute_batch` will call :meth:`compute` with :mod:`multiprocessing` to compute
    scores in parallel, but users may want to define their way of doing parallelization. For example, when the
    scores are computed by external softwares and the parallelization is handled there.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the Score instance.
        
        Sets default number of processes to ``max(1, multiprocessing.cpu_count() - 2)``, initializes
        empty dependency scores dictionary, and sets working directory
        to None.
        
        Parameters
        ----------
        args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        self._nprocs = max(1, mp.cpu_count() - 2)
        self._dependency_scores = {}
        self._wdir = None
    
    @abstractmethod
    def compute(self, mol: Union[str, Chem.Mol]) -> Union[int, float]:
        """Compute the score for a single molecule.
        
        This method must be implemented by subclasses to define the
        specific scoring computation logic.
        
        Parameters
        ----------
        mol : str or rdkit.Chem.Mol
            Molecule to score, either as a SMILES string or RDKit
            molecule object. If the logic takes rdkit.Chem.Mol as input,
            please use :func:`accept_smiles` to decorate this method.
        
        Returns
        -------
        int or float
            The computed score for the molecule. The type depends on
            the specific implementation, but usually is int or float
        """
        ...
    
    def set_nprocs(self, nprocs: int = -1):
        """Set the number of processes for parallel batch computation.
        
        Parameters
        ----------
        nprocs : int, optional
            Number of processes to use. If 0 or negative, uses
            :func:`multiprocessing.cpu_count()` ``- 2``. Default is -1. The actual number of
            processes will be at least 1.
        """
        self._nprocs = max(1, mp.cpu_count()-2 if nprocs <= 0 else nprocs)
    
    @property
    def nprocs(self) -> int:
        """Get the number of processes for parallel batch computation.
        
        Returns
        -------
        int
            Number of processes currently configured for parallel
            computation.
        """
        return self._nprocs
    
    def add_dependency_score(self, name: str, score: Score):
        """Add a dependency score that this score may use.
        
        Dependency scores are other Score instances that this score
        may need to access during computation.
        
        Parameters
        ----------
        name : str
            Name to assign to the dependency score for later access.
        score : Score
            Score instance to add as a dependency.
        """
        self._dependency_scores[name] = score
    
    @property
    def dependency_scores(self) -> Dict[str, Score]:
        """Get all dependency scores. Dependency scores are other Score 
        instances that this score may need to access during computation.
        
        Returns
        -------
        Dict[str, Score]
            Dictionary mapping dependency names to their Score instances.
        """
        return self._dependency_scores

    def compute_batch(
        self, 
        mols: List[Union[str, Chem.Mol]]
    ) -> np.ndarray:
        """Compute scores for a batch of molecules.
        
        Processes multiple molecules in parallel if :attr:`nprocs` > 1,
        otherwise processes sequentially. Displays a progress bar
        during computation.
        
        Parameters
        ----------
        mols : list of str or rdkit.Chem.Mol
            List of molecules to score, each as a SMILES string or
            RDKit molecule object.
        
        Returns
        -------
        numpy.ndarray
            Array of computed scores, one for each input molecule.
        """
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
        """Get the name of this score class.
        
        Returns
        -------
        str
            The class name of this Score instance.
        """
        return self.__class__.__name__
    
    def __init_subclass__(cls, *args, **kwargs):
        """
        Automatically registers non-abstract Score subclasses into the scores registry upon class definition.
        """
        super().__init_subclass__(*args, **kwargs)
        if not inspect.isabstract(cls):
            register_class("score", cls)
    
    @property
    def wdir(self) -> Path:
        """Get the working directory for file-based operations.
        
        Returns
        -------
        Path
            The current working directory path.
        
        Raises
        ------
        AssertionError
            If the working directory has not been set. If that happens, call
            :meth:`set_working_dir` first.
        """
        assert self._wdir is not None, 'No working directory set. Run `self.set_working_dir()` first.'
        return self._wdir

    def set_working_dir(self, wdir: os.PathLike, mkdir: bool = True):
        """Set the working directory for file-based operations.
        
        Parameters
        ----------
        wdir : os.PathLike
            Path to the working directory. Can be a string or Path object.
        mkdir : bool, optional
            If True, create the directory if it doesn't exist (including
            parent directories). Default is True.
        """
        self._wdir = Path(wdir).resolve()
        if mkdir:
            self._wdir.mkdir(exist_ok=True, parents=True)

