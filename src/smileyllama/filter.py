# __all__ = [
#     'PropertyControl', 'PropertyControlResult', 'apply_controls',
#     'HBDControl', 'HBAControl', 'MolWtControl', 'LogPControl', 'RotBondsControl',
#     'TPSAControl', 'SP3FractionControl', 'QEDControl', 'SAControl', 'MacrocycleControl',
#     'CovalentControl', 'FormulaControl', 'SubstructureControl', 'NoBadSubstructureControl'
# ]

from __future__ import annotations

from typing import Union, List
from abc import ABC, abstractmethod
import math
from dataclasses import dataclass, field
import multiprocessing as mp
from functools import partial

from tqdm import tqdm
from rdkit import Chem

from .score import Score, StepNormalizer, safe_mol_from_smiles


class Filter(ABC):
    """Abstract base class for molecular filters.
    
    Filters check whether molecules satisfy certain criteria and can
    generate prompt strings describing those criteria.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the filter.
        
        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """
        ...

    @abstractmethod
    def create_prompt(self) -> str:
        """Create a prompt string describing the filter criteria.
        
        Returns
        -------
        str
            Human-readable description of what the filter checks.
        """
        ...
    
    @abstractmethod
    def apply(self, mol: Chem.Mol) -> bool:
        """Apply the filter to a molecule.
        
        Parameters
        ----------
        mol : rdkit.Chem.Mol
            Molecule to check.
        
        Returns
        -------
        bool
            True if molecule passes the filter, False otherwise.
        """
        ...


class BinaryScoreFilter(Filter):
    """Filter based on binary (0/1) score values.
    
    Uses a :class:`~smileyllama.score.base.Score` that returns binary
    values (0 or 1) to filter molecules. Molecules with score=1 pass,
    score=0 or NaN fail.

    Attributes
    ----------
    score : Score
        The binary score function used for filtering. Should return 0 (fail) or 1 (pass) for a molecule.
    name_in_prompt : str
        The human-readable name or prompt used to describe what this filter checks.
    """
    def __init__(self, score: Score, name_in_prompt: str = ''):
        """Initialize binary score filter.
        
        Parameters
        ----------
        score : Score
            Score instance that returns binary values (0 or 1).
        name_in_prompt : str, optional
            Name to use in prompt generation. If empty, uses score's
            name_in_prompt attribute if available. Default is ''.
        
        Raises
        ------
        AssertionError
            If name_in_prompt is empty and score has no name_in_prompt attribute.
        """
        self.score = score
        self.name_in_prompt = getattr(score, 'name_in_prompt', '') if not name_in_prompt else name_in_prompt
        assert self.name_in_prompt, f'Must provide an non-empty name_in_prompt argument'
    
    @classmethod
    def init_from_class(cls, score_cls: type, name_in_prompt: str = '', **params):
        """Create filter from a score class.
        
        Parameters
        ----------
        score_cls : type
            Score class to instantiate.
        name_in_prompt : str, optional
            Name for prompt generation. Default is an empty string.
        **params
            Parameters to pass to score_cls constructor.
        
        Returns
        -------
        BinaryScoreFilter
            Initialized filter instance.
        """
        return cls(score_cls(**params), name_in_prompt)
    
    def create_prompt(self) -> str:
        """Create prompt string from filter name.
        
        Returns
        -------
        str
            The :attr:`name_in_prompt` string.
        """
        return self.name_in_prompt
    
    def apply(self, mol: Union[str, Chem.Mol]):
        """Apply binary score filter to molecule.
        
        Parameters
        ----------
        mol : str or rdkit.Chem.Mol
            Molecule to check, as SMILES string or RDKit molecule.
        
        Returns
        -------
        bool
            True if score is 1, False if 0 or NaN.
        """
        x = self.score.compute(mol)
        if math.isnan(x):
            return False
        return bool(x)


class NumericScoreFilter(Filter):
    """Filter based on numeric score values with threshold/range checks.
    
    Uses a :class:`~smileyllama.score.base.Score` combined with a
    :class:`~smileyllama.score.normalizer.StepNormalizer` to filter
    molecules based on numeric criteria (e.g., "> 300 molecular weight").
    """

    def __init__(self, score: Score, normalizer: StepNormalizer, name_in_prompt: str = ''):
        """Initialize numeric score filter.
        
        Parameters
        ----------
        score : Score
            Score instance that returns numeric values.
        normalizer : StepNormalizer
            Step normalizer that defines the threshold/range check.
        name_in_prompt : str, optional
            Name to use in prompt generation. If empty, uses score's
            name_in_prompt attribute if available. Default is ''.
        
        Raises
        ------
        AssertionError
            If name_in_prompt is empty and score has no name_in_prompt attribute.
        """
        self.score = score
        self.normalizer = normalizer

        self.name_in_prompt = getattr(score, 'name_in_prompt', '') if not name_in_prompt else name_in_prompt
        assert self.name_in_prompt, f'Must provide an non-empty name_in_prompt argument'
    
    @classmethod
    def init_from_class(cls, score_cls, sign, val, name_in_prompt='', **params):
        """Create filter from a score class and normalizer parameters.
        
        Parameters
        ----------
        score_cls : type
            Score class to instantiate.
        sign : str
            Comparison operator for StepNormalizer ('>', '<', '=', etc.).
        val : int, float, or tuple
            Threshold or range value for StepNormalizer.
        name_in_prompt : str, optional
            Name for prompt generation. Default is ''.
        **params
            Parameters to pass to score_cls constructor.
        
        Returns
        -------
        NumericScoreFilter
            Initialized filter instance.
        """
        return cls(
            score_cls(**params),
            StepNormalizer(sign, val),
            name_in_prompt
        )

    def create_prompt(self) -> str:
        """Create prompt string from normalizer and name.
        
        Returns
        -------
        str
            String like "> 300 molecular weight" or "between 0.5 and 0.8 LogP".
        """
        return f'{str(self.normalizer)} {self.name_in_prompt}'

    def apply(self, mol: Union[str, Chem.Mol]) -> bool:
        """Apply numeric score filter to molecule.
        
        Parameters
        ----------
        mol : str or rdkit.Chem.Mol
            Molecule to check, as SMILES string or RDKit molecule.
        
        Returns
        -------
        bool
            True if score passes the normalizer check, False otherwise.
        """
        x = self.score.compute(mol)
        if math.isnan(x):
            return False
        return self.normalizer(x)
    
    
@dataclass
class FilterResultSingle:
    """Result of applying filters to a single SMILES string.
    
    Note: Field descriptions are provided via type annotations and defaults.
    See individual field definitions below for details.
    """
    smiles: str  #: Original SMILES string.
    success: bool  #: Whether all filters passed.
    valid: bool  #: Whether SMILES is parseable by RDKit.
    reason: str = ""  #: Reason for failure if success=False. Empty if successful.
    canonical_smiles: str = field(init=False)  #: Canonical SMILES representation (auto-computed if valid).
    unique: bool = True  #: Whether this SMILES is unique (set externally).

    def __post_init__(self):
        """Post-initialization: compute canonical SMILES and validate."""
        if not self.valid:
            self.success = False
            self.canonical_smiles = ''
        else:
            self.canonical_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.smiles))
        
        if not self.success:
            assert self.reason, 'Must provide a reason if not success'


def apply_filters_for_single_smiles(smi: str, filters: List[Filter]):
    """Apply all filters to a single SMILES string.
    
    Checks if a SMILES string is valid and passes all provided filters.
    Returns a result object with validation and filter pass status.
    
    Parameters
    ----------
    smi : str
        SMILES string to check.
    filters : list of Filter
        List of :class:`Filter` instances to apply.
    
    Returns
    -------
    FilterResultSingle
        Result object containing validation status, filter pass status,
        and failure reasons if applicable.
    """
    mol = safe_mol_from_smiles(smi)
    if mol is None:
        return FilterResultSingle(smi, False, False, 'Invalid SMILES')

    success = True
    reasons = []
    for ft in filters:
        if not ft.apply(mol):
            success = False
            reasons.append(ft.create_prompt())
    reasons = ','.join(reasons) + ' not satisfied' if reasons else ''
    return FilterResultSingle(smi, success, True, reasons)


def apply_filters(smiles: List[str], filters: List[Filter], nprocs: int = 1):
    """Apply filters to a list of SMILES strings.
    
    Processes SMILES strings in parallel or sequentially, applying all
    filters to each and returning results.
    
    Parameters
    ----------
    smiles : list of str
        List of SMILES strings to filter.
    filters : list of Filter
        List of :class:`Filter` instances to apply to each SMILES.
    nprocs : int, optional
        Number of processes for parallel processing. If 1, processes
        sequentially. Default is 1.
    
    Returns
    -------
    list of FilterResultSingle
        List of filter results, one for each input SMILES string.
    """
    res = []
    function = partial(apply_filters_for_single_smiles, filters=filters)
    if nprocs == 1:
        for smi in smiles:
            r = function(smi)
            res.append(r)
    else:
        with mp.Pool(nprocs) as p:
            for r in tqdm(p.imap(function, smiles), total=len(smiles), desc='Apply filters'):
                res.append(r)
    return res


# class HBDControl(NumericPropertyControl):
#     NAME = 'H-bond donors'
#     PREDEF_RANGES = [3, 4, 5, 7]

#     @classmethod
#     def compute_prop(self, mol: Chem.Mol):
#         return Descriptors.NumHDonors(mol)


# class HBAControl(NumericPropertyControl):
#     NAME = 'H-bond acceptors'
#     PREDEF_RANGES = [3, 4, 5, 10, 15]

#     @classmethod
#     def compute_prop(self, mol: Chem.Mol):
#         return Descriptors.NumHAcceptors(mol)


# class MolWtControl(NumericPropertyControl):
#     NAME = 'Molecular weight'
#     PREDEF_RANGES = [300, 400, 500, 600, 1000]

#     @classmethod
#     def compute_prop(self, mol: Chem.Mol):
#         return Descriptors.MolWt(mol)


# class LogPControl(NumericPropertyControl):
#     NAME = 'LogP'
#     PREDEF_RANGES = [3, 4, 5, 6]

#     @classmethod
#     def compute_prop(self, mol: Chem.Mol):
#         return Descriptors.MolLogP(mol)


# class RotBondsControl(NumericPropertyControl):
#     NAME = 'Rotatable bonds'
#     PREDEF_RANGES = [5, 7, 10]

#     @classmethod
#     def compute_prop(self, mol: Chem.Mol):
#         return Descriptors.NumRotatableBonds(mol)


# class TPSAControl(NumericPropertyControl):
#     NAME = 'TPSA'
#     PREDEF_RANGES = [90, 140, 200]

#     @classmethod
#     def compute_prop(self, mol: Chem.Mol):
#         return Descriptors.TPSA(mol)


# class SP3FractionControl(NumericPropertyControl):
#     NAME = 'Fraction sp3'
#     PREDEF_RANGES = [0.4, 0.5, 0.6, (0.4, 0.6)]
#     PREDEF_RANGES_REVERSE = True

#     @classmethod
#     def compute_prop(self, mol: Chem.Mol):
#         return Descriptors.FractionCSP3(mol)


# class QEDControl(NumericPropertyControl):
#     NAME = 'QED score'

#     @classmethod
#     def compute_prop(self, mol: Chem.Mol):
#         return QED.qed(mol)


# class SAControl(NumericPropertyControl):
#     NAME = 'SA score'

#     @classmethod
#     def compute_prop(self, mol: Chem.Mol):
#         return sascorer.calculateScore(mol)


# class MacrocycleControl(PropertyControl):
#     def __init__(self, size: int = 12):
#         self.size = size
    
#     def create_prompt_for_inference(self):
#         return 'a macrocycle'
    
#     def create_prompt_for_training(self, mol: Chem.Mol):
#         return 'a macrocycle' if self.apply(mol) else 'no macrocycles'
    
#     def apply(self, mol: Chem.Mol):
#         return any([len(ring) >= self.size for ring in mol.GetRingInfo().AtomRings()])


# class CovalentControl(PropertyControl):
#     WARHEADS_SMARTS = {
#         "sulfonyl fluorides": "[#16](=[#8])(=[#8])-[#9]",
#         "chloroacetamides": "[#8]=[#6](-[#6]-[#17])-[#7]",
#         "cyanoacrylamides": "[#7]-[#6](=[#8])-[#6](-[#6]#[#7])=[#6]",
#         "epoxides": "[#6]1-[#6]-[#8]-1",
#         "aziridines": "[#6]1-[#6]-[#7]-1",
#         "disulfides": "[#16]-[#16]",
#         "aldehydes": "[#6H1](=[#8])",
#         "vinyl sulfones": "[#6]=[#6]-[#16](=[#8])(=[#8])-[#7]",
#         "boronic acids/esters": "[#6]-[#5](-[#8])-[#8]",
#         "acrylamides": "[#6]=[#6]-[#6](=[#8])-[#7]",
#         "cyanamides": "[#6]-[#7](-[#6]#[#7])-[#6]",
#         "chloroFluoroAcetamides": "[#7]-[#6](=[#8])-[#6](-[#9])-[#17]",
#         "butynamides": "[#6]#[#6]-[#6](=[#8])-[#7]-[#6]",
#         "chloropropionamides": "[#7]-[#6](=[#8])-[#6](-[#6])-[#17]",
#         "fluorosulfates": "[#8]=[#16](=[#8])(-[#9])-[#8]",
#         "beta lactams": "[#7]1-[#6]-[#6]-[#6]-1=[#8]"
#     }
#     WARHEADS = {key: Chem.MolFromSmarts(val) for key, val in WARHEADS_SMARTS.items()}

#     def __init__(self, type: Union[str, None]):
#         if type is not None:
#             assert type in self.WARHEADS, f'Unrecognized warhead type, select one from {",".join(self.WARHEADS.keys())}'
#         self.type = type
    
#     def create_prompt_for_inference(self):
#         return 'has covalent warheads' + '' if self.type is None else f' ({self.type})'
    
#     @classmethod
#     def create_prompt_for_training(cls, mol: Chem.Mol):
#         warheads = cls.get_warhead_types(mol)
#         if len(warheads) > 0:
#             return f'has covalent warheads ({" ".join(warheads)})'
#         else:
#             return 'lacks covalent warheads'
    
#     def apply(self, mol: Chem.Mol):
#         if not self.type:
#             return len(self.get_warhead_types(mol)) > 0
#         else:
#             return mol.HasSubstructMatch(self.WARHEADS[self.type])
    
#     @classmethod
#     def get_warhead_types(cls, mol: Chem.Mol):
#         types = []
#         for type, warhead in cls.WARHEADS.items():
#             if mol.HasSubstructMatch(warhead):
#                 types.append(type)
#         return types


# class FormulaControl(PropertyControl):
#     def __init__(self, formula: str):
#         self.formula = formula
    
#     def apply(self, mol: Chem.Mol):
#         return Chem.rdMolDescriptors.CalcMolFormula(mol) == self.formula
    
#     def create_prompt_for_inference(self):
#         return f'A formula of {self.formula}'
    
#     @classmethod
#     def create_prompt_for_training(cls, mol: Chem.Mol):
#         return f'A formula of {Chem.rdMolDescriptors.CalcMolFormula(mol)}'


# def replace_special_markers(mol, explicit=False):
#     rw_mol = Chem.RWMol(mol) # Iterate over atoms in the molecule
#     for atom in rw_mol.GetAtoms(): # Check if the atom is a special marker (atomic number 0 and a specific isotope)
#         if atom.GetAtomicNum() == 0:
#             if explicit:
#                 new_atom = Chem.Atom(0)# Replace with special marker
#             else:
#                 new_atom = Chem.Atom(1) #Replace with generic Hydrogen atom
#             rw_mol.ReplaceAtom(atom.GetIdx(), new_atom)
#     if explicit:
#         return Chem.MolToSmiles(rw_mol, doRandom=True)
#     else:
#         return Chem.MolToSmiles(Chem.RemoveHs(rw_mol))


# def get_brics(mol):
#     substructs = list(BRICS.BRICSDecompose(mol, returnMols=True, singlePass=True))
#     return [replace_special_markers(s, explicit=False) for s in substructs[:-1] if 50 < Descriptors.MolWt(s) < 250]


# class SubstructureControl(PropertyControl):
#     def __init__(self, substruct: str):
#         self.sub_str = substruct
#         sub = Chem.MolFromSmiles(self.sub_str)
#         if sub is None:
#             sub = Chem.MolFromSmarts(self.sub_str)
#         assert sub is not None, f'Invalid substructure {self.sub_str}'
#         self.sub = sub
    
#     def apply(self, mol: Chem.Mol):
#         return mol.HasSubstructMatch(self.sub)
    
#     def create_prompt_for_inference(self):
#         return f'a substructure of {self.sub_str}'
    
#     @classmethod
#     def create_prompt_for_training(cls, mol: Chem.Mol):
#         brics = get_brics(mol)
#         if len(brics) > 0:
#             return f'a substructure of {random.choice(brics)}'
#         else:
#             return ''


# # PAINS
# params = FilterCatalog.FilterCatalogParams()
# params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
# PAINS_catalog = FilterCatalog.FilterCatalog(params)


# class NoBadSubstructureControl(PropertyControl):

#     UNDESIRABLE_SMARTS = [
#         "[C^2]1=[C^2]-[C^2]=[C^2]~[C;!d4]~[C;!^2;d2]1", 
#         "[C^2]1~[C^2]~[C^2]~[C^2]~[C;!^2;d2]~[N]1",
#         "[#6^2]1~[#6^2]~[#6^3;!d4]~[#6^2]2~[#6^2]~[#6^2]~[#6^2]~[#6^2](~[*])~[#6^2]~2~[#6^2]~1",
#         "[#6]1(=[*])[#6]=[#6][#6]=[#6]1", 
#         "[#6]1=[#6][R{2-}]=[R{2-}]1", 
#         "[#6^2]1~[#6^2]~[#6^2]~[#6^2]~[#6^1]~[#6^1]~1",
#         "[#7,#8,#16]-[#9,#17,#35,#53]", 
#         "[r3,r4]@[r5,r6]", 
#         "[*]=[#6,#7,#8]=[*]",
#         "[#7,#16]=[#16]", 
#         "[#8]-[#8]",
#     ]
#     PYRROLE_FORM_SMARTS = [
#         "[N^2]1~[C,N;^2]~[C,N;^2]~[C,N;^2]~[C;^3]1", 
#         "[C,N;^2]1~[N;^2]~[C,N;^2]~[C,N;^2]~[C;^3]1"
#     ]
#     CORRECT_PYRROLE_SMARTS = [
#         "[N^2]1~[C,N;^2](=[*])~[C,N;^2]~[C,N;^2]~[C;^3]1", 
#         "[N^2]1~[C,N;^2]~[C,N;^2](=[*])~[C,N;^2]~[C;^3]1",
#         "[N^2]1~[C,N;^2]~[C,N;^2]~[C,N;^2](=[*])~[C;^3]1", 
#         "[C,N;^2](=[*])1~[N;^2]~[C,N;^2]~[C,N;^2]~[C;^3]1",
#         "[C,N;^2]1~[N;^2]~[C,N;^2](=[*])~[C,N;^2]~[C;^3]1", 
#         "[C,N;^2]1~[N;^2]~[C,N;^2]~[C,N;^2](=[*])~[C;^3]1"
#     ]

#     UNDESIRABLE = [Chem.MolFromSmarts(s) for s in UNDESIRABLE_SMARTS]
#     PYRROLE_FORM = [Chem.MolFromSmarts(s) for s in PYRROLE_FORM_SMARTS]
#     CORRECT_PYRROLE = [Chem.MolFromSmarts(s) for s in CORRECT_PYRROLE_SMARTS]

#     @classmethod
#     def has_pains_alert(cls, mol: Chem.Mol):
#         return PAINS_catalog.HasMatch(mol)
    
#     @classmethod
#     def has_wrong_pyrrole(cls, mol: Chem.Mol):
#         if any([mol.HasSubstructMatch(s) for s in cls.PYRROLE_FORM]) and not any([mol.HasSubstructMatch(s) for s in cls.CORRECT_PYRROLE]):
#             return True
    
#     @classmethod
#     def has_bad_substructure(cls, mol: Chem.Mol):
#         return any([mol.HasSubstructMatch(s) for s in cls.UNDESIRABLE])
    
#     @classmethod
#     def apply(cls, mol: Chem.Mol):
#         if cls.has_pains_alert(mol):
#             return False
#         if cls.has_wrong_pyrrole(mol):
#             return False
#         if cls.has_bad_substructure(mol):
#             return False
#         return True
    
#     @classmethod
#     def create_prompt_for_inference(cls):
#         return 'lacks bad SMARTS'
    
#     @classmethod
#     def create_prompt_for_training(cls, mol: Chem.Mol):
#         if cls.apply(mol):
#             return 'lacks bad SMARTS'
#         else:
#             return 'has bad SMARTS'

