__all__ = [
    'NumHBD', 'NumHBA', 'MolWt', 'LogP', 'NumRotBonds', 'TPSA',
    'FractionCSP3', 'QED', 'SAScore', 'NumHeavyAtoms', 'NumAliphaticRings',
    'NumAromaticRings', 'NumQEDStructureAlerts', 'MaxRingSize', 'MinRingSize',
    'HeteroAtomsFraction', 'Similarity', 'SubstructureMatch'
]

from typing import Union, Optional
import sys, os
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, RDConfig, Lipinski
from rdkit.Chem import QED as rdQED
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from .base import Score, accept_smiles
from .registry import register


@register("rdkit_scores")
class NumHBD(Score):
    '''Number of hydrogen bond donor'''
    name_in_prompt = 'H-Bond donors'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        return Descriptors.NumHDonors(mol)


@register("rdkit_scores")
class NumHBA(Score):
    '''Number of hydrogen bond acceptor'''
    name_in_prompt = 'H-bond acceptors'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        return Descriptors.NumHAcceptors(mol)


@register("rdkit_scores")
class MolWt(Score):
    '''Molecular weight'''
    name_in_prompt = 'Molecular weight'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> float:
        return Descriptors.MolWt(mol)


@register("rdkit_scores")
class LogP(Score):
    '''Wildman-Crippen LogP value'''
    name_in_prompt = 'LogP'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> float:
        return Descriptors.MolLogP(mol)


@register("rdkit_scores")
class NumRotBonds(Score):
    '''Number of rotatable bonds'''
    name_in_prompt = 'Rotatable bonds'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        return Descriptors.NumRotatableBonds(mol)
    

@register("rdkit_scores")
class TPSA(Score):
    '''Topological polar surface area'''
    name_in_prompt = 'TPSA'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> float:
        return Descriptors.TPSA(mol)


@register("rdkit_scores")
class FractionCSP3(Score):
    '''Sp3 carbon fraction'''
    name_in_prompt = 'Fraction sp3'
    
    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> float:
        return Descriptors.FractionCSP3(mol)


@register("rdkit_scores")
class QED(Score):
    '''Quantitative estimation of drug-likeliness'''
    name_in_prompt = 'QED score'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> float:
        return rdQED.qed(mol)
    

@register("rdkit_scores")
class SAScore(Score):
    '''Synthetic accessibility score'''
    name_in_prompt = 'SA score'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> float:
        return sascorer.calculateScore(mol)
    

@register("rdkit_scores")
class NumHeavyAtoms(Score):
    '''Number of heavy atoms'''
    name_in_prompt = 'heavy atoms'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        return Lipinski.HeavyAtomCount(mol)


@register("rdkit_scores")
class NumAliphaticRings(Score):
    '''Number of aliphatic rings'''
    name_in_prompt = 'aliphatic rings'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        return Lipinski.NumAliphaticRings(mol)
    

@register("rdkit_scores")
class NumAromaticRings(Score):
    '''Number of aromatic rings'''
    name_in_prompt = 'aromatic rings'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        return Lipinski.NumAromaticRings(mol)


@register("rdkit_scores")
class NumQEDStructureAlerts(Score):
    '''Number of QED structure alerts'''
    name_in_prompt = 'QED structure alerts'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        return sum([1 for alert in rdQED.StructuralAlerts if mol.HasSubstructMatch(alert)])
    

@register("rdkit_scores")
class MaxRingSize(Score):
    '''Maximum ring size'''

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        rs = [len(r) for r in mol.GetRingInfo().AtomRings()]
        return 0 if len(rs) == 0 else max(rs)


@register("rdkit_scores")
class MinRingSize(Score):
    '''Min ring size'''

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        rs = [len(r) for r in mol.GetRingInfo().AtomRings()]
        return 0 if len(rs) == 0 else min(rs)


@register("rdkit_scores")
class NumHeteroAtoms(Score):
    '''Number of hetero atoms'''

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        return Lipinski.NumHeteroatoms(mol)


@register("rdkit_scores")
class HeteroAtomsFraction(Score):
    '''Fraction of hetero atoms'''

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        return NumHeteroAtoms.compute(mol) / NumHeavyAtoms.compute(mol)


@register("rdkit_scores")
class Similarity(Score):
    """Tanimoto similarity between a given molecule and a reference molecule."""

    def __init__(
        self,
        molecule: Union[str, Chem.Mol],
        *,
        radius: int = 2,       # ECFP4 -> radius=2
        n_bits: int = 2048,
        use_chirality: bool = False,
        nprocs: int = 0
    ):
        super().__init__(nprocs)
        # Parse reference molecule
        if isinstance(molecule, str):
            self.ref = Chem.MolFromSmiles(molecule)
            assert self.ref is not None, f"Invalid SMILES: {molecule}"
        elif isinstance(molecule, Chem.Mol):
            self.ref = molecule
        else:
            raise TypeError(f"`molecule` must be SMILES (str) or RDKit Mol, got {type(molecule)}")

        self.radius = radius
        self.n_bits = n_bits
        self.use_chirality = use_chirality

        # Precompute reference fingerprint
        self.ref_fp = self._fp(self.ref)

    def _fp(self, mol: Chem.Mol):
        """Compute Morgan fingerprint for a molecule."""
        return AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=self.radius,
            nBits=self.n_bits,
            useChirality=self.use_chirality,
        )
    
    @accept_smiles
    def compute(self, mol: Chem.Mol) -> float:
        fp = self._fp(mol)
        return float(DataStructs.TanimotoSimilarity(fp, self.ref_fp))


@register("rdkit_scores")
class SubstructureMatch(Score):
    def __init__(self, substruct: str):
        self.sub_str = substruct
        sub = Chem.MolFromSmiles(self.sub_str)
        if sub is None:
            sub = Chem.MolFromSmarts(self.sub_str)
        assert sub is not None, f'Invalid substructure {self.sub_str}'
        self.sub = sub
    
    @accept_smiles
    def compute(self, mol: Chem.Mol):
        return mol.HasSubstructMatch(self.sub)