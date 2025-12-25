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
    """Number of hydrogen bond donors in a molecule.
    """
    name_in_prompt = 'H-Bond donors'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        """Compute number of hydrogen bond donors.
        
        Parameters
        ----------
        mol : str or rdkit.Chem.Mol
            Molecule to score, either as a SMILES string or RDKit
            molecule object.
        
        Returns
        -------
        int
            Number of hydrogen bond donors.
        """
        return Descriptors.NumHDonors(mol)


@register("rdkit_scores")
class NumHBA(Score):
    """Number of hydrogen bond acceptors in a molecule.
    """
    name_in_prompt = 'H-bond acceptors'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        """Compute number of hydrogen bond acceptors.
        
        Parameters
        ----------
        mol : str or rdkit.Chem.Mol
            Molecule to score, either as a SMILES string or RDKit
            molecule object.
        
        Returns
        -------
        int
            Number of hydrogen bond acceptors.
        """
        return Descriptors.NumHAcceptors(mol)


@register("rdkit_scores")
class MolWt(Score):
    """Molecular weight of a molecule in atomic mass units (amu).
    """
    name_in_prompt = 'Molecular weight'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> float:
        """Compute molecular weight.
        
        Parameters
        ----------
        mol : str or rdkit.Chem.Mol
            Molecule to score, either as a SMILES string or RDKit
            molecule object.
        
        Returns
        -------
        float
            Molecular weight in amu.
        """
        return Descriptors.MolWt(mol)


@register("rdkit_scores")
class LogP(Score):
    """Wildman-Crippen LogP value.
    
    Computes the octanol-water partition coefficient (log P) using the
    Wildman-Crippen method.
    """
    name_in_prompt = 'LogP'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> float:
        """Compute LogP value.
        
        Parameters
        ----------
        mol : str or rdkit.Chem.Mol
            Molecule to score, either as a SMILES string or RDKit
            molecule object.
        
        Returns
        -------
        float
            LogP value.
        """
        return Descriptors.MolLogP(mol)


@register("rdkit_scores")
class NumRotBonds(Score):
    """Number of rotatable bonds in a molecule.
    """
    name_in_prompt = 'Rotatable bonds'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        """Compute number of rotatable bonds.
        
        Parameters
        ----------
        mol : str or rdkit.Chem.Mol
            Molecule to score, either as a SMILES string or RDKit
            molecule object.
        
        Returns
        -------
        int
            Number of rotatable bonds.
        """
        return Descriptors.NumRotatableBonds(mol)
    

@register("rdkit_scores")
class TPSA(Score):
    """Topological polar surface area (TPSA).
    
    Computes the sum of surfaces of polar atoms in a molecule, which is
    related to drug absorption and bioavailability.
    """
    name_in_prompt = 'TPSA'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> float:
        """Compute TPSA.
        
        Parameters
        ----------
        mol : str or rdkit.Chem.Mol
            Molecule to score, either as a SMILES string or RDKit
            molecule object.
        
        Returns
        -------
        float
            Topological polar surface area.
        """
        return Descriptors.TPSA(mol)


@register("rdkit_scores")
class FractionCSP3(Score):
    """Fraction of sp3 hybridized carbons.
    
    Computes the ratio of sp3 carbons to total carbons, which is related
    to molecular complexity and saturation.
    """
    name_in_prompt = 'Fraction sp3'
    
    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> float:
        """Compute fraction of sp3 carbons.
        
        Parameters
        ----------
        mol : str or rdkit.Chem.Mol
            Molecule to score, either as a SMILES string or RDKit
            molecule object.
        
        Returns
        -------
        float
            Fraction of sp3 carbons (0.0 to 1.0).
        """
        return Descriptors.FractionCSP3(mol)


@register("rdkit_scores")
class QED(Score):
    """Quantitative estimation of drug-likeliness (QED).
    
    Computes a score between 0 and 1 indicating how "drug-like" a molecule
    is based on multiple molecular properties.
    """
    name_in_prompt = 'QED score'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> float:
        """Compute QED score.
        
        Parameters
        ----------
        mol : str or rdkit.Chem.Mol
            Molecule to score, either as a SMILES string or RDKit
            molecule object.
        
        Returns
        -------
        float
            QED score (0.0 to 1.0).
        """
        return rdQED.qed(mol)
    

@register("rdkit_scores")
class SAScore(Score):
    """Synthetic accessibility score (SA score).
    
    Computes a score indicating how easy it is to synthesize a molecule,
    with lower scores indicating easier synthesis.
    """
    name_in_prompt = 'SA score'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> float:
        """Compute SA score.
        
        Parameters
        ----------
        mol : str or rdkit.Chem.Mol
            Molecule to score, either as a SMILES string or RDKit
            molecule object.
        
        Returns
        -------
        float
            SA score (typically 1-10, lower is easier to synthesize).
        """
        return sascorer.calculateScore(mol)
    

@register("rdkit_scores")
class NumHeavyAtoms(Score):
    """Number of heavy atoms (non-hydrogen atoms) in a molecule."""
    name_in_prompt = 'heavy atoms'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        """Compute number of heavy atoms.
        
        Parameters
        ----------
        mol : str or rdkit.Chem.Mol
            Molecule to score, either as a SMILES string or RDKit
            molecule object.
        
        Returns
        -------
        int
            Number of heavy atoms.
        """
        return Lipinski.HeavyAtomCount(mol)


@register("rdkit_scores")
class NumAliphaticRings(Score):
    """Number of aliphatic rings in a molecule."""
    name_in_prompt = 'aliphatic rings'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        """Compute number of aliphatic rings.
        
        Parameters
        ----------
        mol : str or rdkit.Chem.Mol
            Molecule to score, either as a SMILES string or RDKit
            molecule object.
        
        Returns
        -------
        int
            Number of aliphatic rings.
        """
        return Lipinski.NumAliphaticRings(mol)
    

@register("rdkit_scores")
class NumAromaticRings(Score):
    """Number of aromatic rings in a molecule."""
    name_in_prompt = 'aromatic rings'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        """Compute number of aromatic rings.
        
        Parameters
        ----------
        mol : rdkit.Chem.Mol or str
            Molecule to analyze, either as RDKit molecule or SMILES string.
        
        Returns
        -------
        int
            Number of aromatic rings.
        """
        return Lipinski.NumAromaticRings(mol)


@register("rdkit_scores")
class NumQEDStructureAlerts(Score):
    """Number of QED structure alerts.
    
    Counts the number of problematic substructures that may indicate
    poor drug-likeness or potential toxicity.
    """
    name_in_prompt = 'QED structure alerts'

    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        """Compute number of QED structure alerts.
        
        Parameters
        ----------
        mol : str or rdkit.Chem.Mol
            Molecule to score, either as a SMILES string or RDKit
            molecule object.
        
        Returns
        -------
        int
            Number of QED structure alerts found.
        """
        return sum([1 for alert in rdQED.StructuralAlerts if mol.HasSubstructMatch(alert)])
    

@register("rdkit_scores")
class MaxRingSize(Score):
    """
    Finds the largest ring (by number of atoms) in the molecule.
    """
    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        """Compute maximum ring size.
        
        Parameters
        ----------
        mol : str or rdkit.Chem.Mol
            Molecule to score, either as a SMILES string or RDKit
            molecule object.
        
        Returns
        -------
        int
            Maximum ring size, or 0 if molecule has no rings.
        """
        rs = [len(r) for r in mol.GetRingInfo().AtomRings()]
        return 0 if len(rs) == 0 else max(rs)


@register("rdkit_scores")
class MinRingSize(Score):
    """Finds the smallest ring (by number of atoms) in the molecule.
    """
    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        """Compute minimum ring size.
        
        Parameters
        ----------
        mol : str or rdkit.Chem.Mol
            Molecule to score, either as a SMILES string or RDKit
            molecule object.
        
        Returns
        -------
        int
            Minimum ring size, or 0 if molecule has no rings.
        """
        rs = [len(r) for r in mol.GetRingInfo().AtomRings()]
        return 0 if len(rs) == 0 else min(rs)


@register("rdkit_scores")
class NumHeteroAtoms(Score):
    """Number of heteroatoms (non-carbon, non-hydrogen atoms) in a molecule."""
    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        """Compute number of heteroatoms.
        
        Parameters
        ----------
        mol : str or rdkit.Chem.Mol
            Molecule to score, either as a SMILES string or RDKit
            molecule object.
        
        Returns
        -------
        int
            Number of heteroatoms.
        """
        return Lipinski.NumHeteroatoms(mol)


@register("rdkit_scores")
class HeteroAtomsFraction(Score):
    """Fraction of heteroatoms relative to heavy atoms.
    
    Computes the ratio of heteroatoms to total heavy atoms.
    """
    @classmethod
    @accept_smiles
    def compute(cls, mol: Chem.Mol) -> int:
        """Compute fraction of heteroatoms.
        
        Parameters
        ----------
        mol : str or rdkit.Chem.Mol
            Molecule to score, either as a SMILES string or RDKit
            molecule object.
        
        Returns
        -------
        float
            Fraction of heteroatoms (0.0 to 1.0).
        """
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
        use_chirality: bool = False
    ):
        """Initialize Similarity scorer.
        
        Parameters
        ----------
        molecule : str or rdkit.Chem.Mol
            Reference molecule to compare against, as SMILES string or
            RDKit molecule object.
        radius : int, optional
            Radius for Morgan fingerprint (ECFP4 uses radius=2).
            Default is 2.
        n_bits : int, optional
            Number of bits in the fingerprint. Default is 2048.
        use_chirality : bool, optional
            Whether to include chirality in fingerprint. Default is False.
        
        Raises
        ------
        AssertionError
            If molecule cannot be parsed from SMILES string.
        TypeError
            If molecule is not a string or RDKit Mol object.
        """
        super().__init__()
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
        """Compute Tanimoto similarity to reference molecule.
        
        Parameters
        ----------
        mol : rdkit.Chem.Mol or str
            Molecule to compare, either as RDKit molecule or SMILES string.
        
        Returns
        -------
        float
            Tanimoto similarity score (0.0 to 1.0).
        """
        fp = self._fp(mol)
        return float(DataStructs.TanimotoSimilarity(fp, self.ref_fp))


@register("rdkit_scores")
class SubstructureMatch(Score):
    """Substructure matching score.
    
    Checks whether a molecule contains a specific substructure pattern.
    Returns 1 if the substructure is found, 0 otherwise.
    """
    def __init__(self, substruct: str):
        """Initialize SubstructureMatch scorer.
        
        Parameters
        ----------
        substruct : str
            Substructure pattern as SMILES or SMARTS string.
        
        Raises
        ------
        AssertionError
            If substructure cannot be parsed as SMILES or SMARTS.
        """
        super().__init__()
        self.sub_str = substruct
        sub = Chem.MolFromSmiles(self.sub_str)
        if sub is None:
            sub = Chem.MolFromSmarts(self.sub_str)
        assert sub is not None, f'Invalid substructure {self.sub_str}'
        self.sub = sub
    
    @accept_smiles
    def compute(self, mol: Chem.Mol):
        """Check if molecule contains the substructure.
        
        Parameters
        ----------
        mol : rdkit.Chem.Mol or str
            Molecule to check, either as RDKit molecule or SMILES string.
        
        Returns
        -------
        bool
            True if substructure is found, False otherwise.
        """
        return mol.HasSubstructMatch(self.sub)