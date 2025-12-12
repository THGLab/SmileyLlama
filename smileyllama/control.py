from typing import Literal, Any, Union, Tuple, Optional, List
from abc import ABC, abstractmethod
import os
import sys
import operator
import random
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, RDConfig, FilterCatalog, BRICS
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

# PAINS
params = FilterCatalog.FilterCatalogParams()
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_catalog = FilterCatalog.FilterCatalog(params)


class PropertyControl(ABC):
    def __init__(self, *args, **kwargs):
        ...

    @abstractmethod
    def create_prompt_for_inference(self) -> str:
        ...
    
    @abstractmethod
    def apply(self, mol: Chem.Mol) -> bool:
        ...


class RangePropertyControl(PropertyControl):

    NAME = None
    OPS = {
        '=': operator.eq,
        '>': operator.gt, '<': operator.lt,
        '>=': operator.ge, '<=': operator.le,
    }
    PROP_FUNC = None
    PREDEF_RANGES = list()
    PREDEF_RANGES_REVERSE: bool = False

    def __init__(self, sign: Literal['>', '<', '=', '>=', '<=', ''], val: Union[Any, Tuple[Any, Any]]):
        super().__init__()
        if sign != '':
            self.func = lambda x : self.OPS[sign](x, val)
        else:
            assert isinstance(val, tuple) and len(val) == 2, 'upper and lower bounds should be set if sign is empty'
            assert val[1] > val[0], 'upper bound must be larger than lower bound'
            self.func = lambda x: x <= val[1] and x >= val[0]

        self.val = val
        self.sign = sign
        if self.NAME is None:
            raise NotImplementedError(f"Subclass of {self.__class__.__name__} must define a class variable NAME.")
        if self.PROP_FUNC is None:
            raise NotImplementedError(f"Subclass of {self.__class__.__name__} must define a function to calculate property.")

    def create_prompt_for_inference(self):
        return f"{self.sign} {self.val} {self.NAME}" if self.sign else f'between {self.val[0]} and {self.val[1]} {self.NAME}'

    def apply(self, mol: Chem.Mol):
        return self.func(self.__class__.PROP_FUNC(mol))
    
    @classmethod
    def create_prompt_for_training(cls, mol: Chem.Mol, ranges: Optional[List] = None, reverse: Optional[bool] = False):
        if ranges is None:
            ranges = cls.PREDEF_RANGES
        if reverse is None:
            reverse = cls.PREDEF_RANGES_REVERSE

        assert len(ranges) > 0, 'Please specify ranges'
        
        prop = cls.PROP_FUNC(mol)
        intervals = []
        points = []
        for r in ranges:
            if isinstance(r, tuple):
                assert len(r) == 2 and r[1] > r[0], f'{r} if not a valid interval specification'
                intervals.append(r)
            else:
                points.append(r)
        
        points.sort(reverse=reverse)
        prompts = []
        if not reverse:
            for x in points:
                if prop <= x:
                    prompts.append(f'<= {x} {cls.NAME}')
            if len(prompts) == 0:
                prompts.append(f'> {points[-1]} {cls.NAME}')
        else:
            for x in points:
                if prop > x:
                    prompts.append(f'> {x} {cls.NAME}')
            if len(prompts) == 0:
                prompts.append(f'<= {points[-1]} {cls.NAME}')
        
        for interval in intervals:
            if prop <= interval[1] and prop >= interval[0]:
                prompts.append(f'between {interval[0]} and {interval[1]} {cls.NAME}')
        
        return random.choice(prompts)


@dataclass
class PropertyControlResult:
    smiles: str
    success: bool = True
    reason: str = ""


def apply_controls(smiles: List[str], controls: List[PropertyControl]):
    res = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            res.append(PropertyControlResult(smi, False, 'Invalid SMILES'))
            continue

        success = True
        reasons = []
        for c in controls:
            if not c.apply(mol):
                success = False
                reasons.append(c.__class__.__name__)
        reasons = ','.join(reasons) + ' not satisfied' if reasons else ''
        r = PropertyControlResult(smi, success, reasons)
        res.append(r)
    return res



class HBDControl(RangePropertyControl):
    NAME = 'H-bond donors'
    PROP_FUNC = Descriptors.NumHDonors
    PREDEF_RANGES = [3, 4, 5, 7]


class HBAControl(RangePropertyControl):
    NAME = 'H-bond acceptors'
    PROP_FUNC = Descriptors.NumHAcceptors
    PREDEF_RANGES = [3, 4, 5, 10, 15]


class MolWtControl(RangePropertyControl):
    NAME = 'Molecular weight'
    PROP_FUNC = Descriptors.MolWt
    PREDEF_RANGES = [300, 400, 500, 600]


class LogPControl(RangePropertyControl):
    NAME = 'LogP'
    PROP_FUNC = Descriptors.MolLogP
    PREDEF_RANGES = [3, 4, 5, 6]
        

class RotBondsControl(RangePropertyControl):
    NAME = 'Rotatable bonds'
    PROP_FUNC = Descriptors.NumRotatableBonds
    PREDEF_RANGES = [7, 10]


class TPSAControl(RangePropertyControl):
    NAME = 'TPSA'
    PROP_FUNC = Descriptors.TPSA
    PREDEF_RANGES = [90, 140, 200]


class SP3FractionControl(RangePropertyControl):
    NAME = 'Fraction sp3'
    PROP_FUNC = Descriptors.FractionCSP3
    PREDEF_RANGES = [0.4, 0.5, 0.6]
    PREDEF_RANGES_REVERSE = True


class QEDControl(RangePropertyControl):
    NAME = 'QED score'
    PROP_FUNC = QED.qed


class SAControl(RangePropertyControl):
    NAME = 'SA score'
    PROP_FUNC = sascorer.calculateScore
    

class MacrocycleControl(PropertyControl):
    def __init__(self, size: int = 8):
        self.size = size
    
    def create_prompt_for_inference(self):
        return 'a macrocycle'
    
    def create_prompt_for_training(self, mol: Chem.Mol):
        return 'a macrocycle' if self.apply(mol) else 'no macrocycles'
    
    def apply(self, mol: Chem.Mol):
        return any([len(ring) >= self.size for ring in mol.GetRingInfo().AtomRings()])


class CovalentControl(PropertyControl):
    WARHEADS_SMARTS = {
        "sulfonyl fluorides": "[#16](=[#8])(=[#8])-[#9]",
        "chloroacetamides": "[#8]=[#6](-[#6]-[#17])-[#7]",
        "cyanoacrylamides": "[#7]-[#6](=[#8])-[#6](-[#6]#[#7])=[#6]",
        "epoxides": "[#6]1-[#6]-[#8]-1",
        "aziridines": "[#6]1-[#6]-[#7]-1",
        "disulfides": "[#16]-[#16]",
        "aldehydes": "[#6H1](=[#8])",
        "vinyl sulfones": "[#6]=[#6]-[#16](=[#8])(=[#8])-[#7]",
        "boronic acids/esters": "[#6]-[#5](-[#8])-[#8]",
        "acrylamides": "[#6]=[#6]-[#6](=[#8])-[#7]",
        "cyanamides": "[#6]-[#7](-[#6]#[#7])-[#6]",
        "chloroFluoroAcetamides": "[#7]-[#6](=[#8])-[#6](-[#9])-[#17]",
        "butynamides": "[#6]#[#6]-[#6](=[#8])-[#7]-[#6]",
        "chloropropionamides": "[#7]-[#6](=[#8])-[#6](-[#6])-[#17]",
        "fluorosulfates": "[#8]=[#16](=[#8])(-[#9])-[#8]",
        "beta lactams": "[#7]1-[#6]-[#6]-[#6]-1=[#8]"
    }
    WARHEADS = {key: Chem.MolFromSmarts(val) for key, val in WARHEADS_SMARTS.items()}

    def __init__(self, type: Union[str, None]):
        if type is not None:
            assert type in self.WARHEADS, f'Unrecognized warhead type, select one from {",".join(self.WARHEADS.keys())}'
        self.type = type
    
    def create_prompt_for_inference(self):
        return 'has covalent warheads' + '' if self.type is None else f' ({self.type})'
    
    @classmethod
    def create_prompt_for_training(cls, mol: Chem.Mol):
        warheads = cls.get_warhead_types(mol)
        if len(warheads) > 0:
            return f'has covalent warheads ({" ".join(warheads)})'
        else:
            return 'lacks covalent warheads'
    
    def apply(self, mol: Chem.Mol):
        if not self.type:
            return len(self.get_warhead_types(mol)) > 0
        else:
            return mol.HasSubstructMatch(self.WARHEADS[self.type])
    
    @classmethod
    def get_warhead_types(cls, mol: Chem.Mol):
        types = []
        for type, warhead in cls.WARHEADS.items():
            if mol.HasSubstructMatch(warhead):
                types.append(type)
        return types


class FormulaControl(PropertyControl):
    def __init__(self, formula: str):
        self.formula = formula
    
    def apply(self, mol: Chem.Mol):
        return Chem.rdMolDescriptors.CalcMolFormula(mol) == self.formula
    
    def create_prompt_for_inference(self):
        return f'A formula of {self.formula}'
    
    @classmethod
    def create_prompt_for_training(cls, mol: Chem.Mol):
        return f'A formula of {Chem.rdMolDescriptors.CalcMolFormula(mol)}'


def replace_special_markers(mol, explicit=False):
    rw_mol = Chem.RWMol(mol) # Iterate over atoms in the molecule
    for atom in rw_mol.GetAtoms(): # Check if the atom is a special marker (atomic number 0 and a specific isotope)
        if atom.GetAtomicNum() == 0:
            if explicit:
                new_atom = Chem.Atom(0)# Replace with special marker
            else:
                new_atom = Chem.Atom(1) #Replace with generic Hydrogen atom
            rw_mol.ReplaceAtom(atom.GetIdx(), new_atom)
    if explicit:
        return Chem.MolToSmiles(rw_mol, doRandom=True)
    else:
        return Chem.MolToSmiles(Chem.RemoveHs(rw_mol))


def get_brics(mol):
    substructs = list(BRICS.BRICSDecompose(mol, returnMols=True, singlePass=True))
    return [replace_special_markers(s, explicit=False) for s in substructs[:-1] if 50 < Descriptors.MolWt(s) < 250]


class SubstructureControl(PropertyControl):
    def __init__(self, substruct: str):
        self.sub_str = substruct
        sub = Chem.MolFromSmiles(self.sub_str)
        if sub is None:
            sub = Chem.MolFromSmarts(self.sub_str)
        assert sub is not None, f'Invalid substructure {self.sub_str}'
        self.sub = sub
    
    def apply(self, mol: Chem.Mol):
        return mol.HasSubstructMatch(self.sub)
    
    def create_prompt_for_inference(self):
        return f'a substructure of {self.sub_str}'
    
    @classmethod
    def create_prompt_for_training(cls, mol: Chem.Mol):
        brics = get_brics(mol)
        if len(brics) > 0:
            return f'a substructure of {random.choice(brics)}'
        else:
            return ''
