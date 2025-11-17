from rdkit import Chem
from rdkit.Chem import Descriptors, BRICS, FilterCatalog
import random
import pandas as pd
import numpy as np
import pandas as pd

OUT_OF_RANGE = -10

UNDESIRABLE_PATTERNS = ["[C^2]1=[C^2]-[C^2]=[C^2]~[C;!d4]~[C;!^2;d2]1", "[C^2]1~[C^2]~[C^2]~[C^2]~[C;!^2;d2]~[N]1",
    "[#6^2]1~[#6^2]~[#6^3;!d4]~[#6^2]2~[#6^2]~[#6^2]~[#6^2]~[#6^2](~[*])~[#6^2]~2~[#6^2]~1",
    "[#6]1(=[*])[#6]=[#6][#6]=[#6]1", "[#6]1=[#6][R{2-}]=[R{2-}]1", "[#6^2]1~[#6^2]~[#6^2]~[#6^2]~[#6^1]~[#6^1]~1",
    "[#7,#8,#16]-[#9,#17,#35,#53]", "[r3,r4]@[r5,r6]", "[*]=[#6,#7,#8]=[*]",  # bad patterns by Eric
    "[#7,#16]=[#16]", "[#8]-[#8]",
]
PYRROLE_FORM = ["[N^2]1~[C,N;^2]~[C,N;^2]~[C,N;^2]~[C;^3]1", "[C,N;^2]1~[N;^2]~[C,N;^2]~[C,N;^2]~[C;^3]1"]
CORRECT_PYRROLE = ["[N^2]1~[C,N;^2](=[*])~[C,N;^2]~[C,N;^2]~[C;^3]1", "[N^2]1~[C,N;^2]~[C,N;^2](=[*])~[C,N;^2]~[C;^3]1",
            "[N^2]1~[C,N;^2]~[C,N;^2]~[C,N;^2](=[*])~[C;^3]1", "[C,N;^2](=[*])1~[N;^2]~[C,N;^2]~[C,N;^2]~[C;^3]1",
            "[C,N;^2]1~[N;^2]~[C,N;^2](=[*])~[C,N;^2]~[C;^3]1", "[C,N;^2]1~[N;^2]~[C,N;^2]~[C,N;^2](=[*])~[C;^3]1"]

COVALENT_WARHEADS = {
    "sulfonyl fluorides": "[#16](=[#8])(=[#8])-[#9]",
    "chloroacetamides": "[#8]=[#6](-[#6]-[#17])-[#7]",
    "cyanoacrylamides": "[#7]-[#6](=[#8])-[#6](-[#6]#[#7])=[#6]",
    "epoxides": "[#6]1-[#6]-[#8]-1",
    "aziridines": "[#6]1-[#6]-[#7]-1",
    "disulfides": "[#16]-[#16]",
    "aldehydes": "[#6](=[#8])-[#1]",
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

params = FilterCatalog.FilterCatalogParams()
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_catalog = FilterCatalog.FilterCatalog(params)

def has_pains_alert(mol):
    """Check if a molecule has PAINS alert. If has, return True"""
    return PAINS_catalog.HasMatch(mol)

def has_bad_ring(mol):
    """Check if a molecule has more than 3 rings fused by one atom. If has, return True"""
    ringAtoms = []
    for ring in Chem.GetSSSR(mol):
        ringAtoms += list(ring)
    if len(ringAtoms) == 0:
        return False
    _, counts = np.unique(ringAtoms, return_counts=True)
    return counts.max() >= 3

def check_valid_pattern(input):
    """
    Check if a molecule has undesirable patterns or violates PAINS. 
    Returns True if none detected. Mols with violations will have drug_likeliness score zerod.
    """
    if type(input) is str:
        mol = Chem.MolFromSmiles(input)
    else:
        mol = input
    #for atom in mol.GetAtoms():
    #    if atom.GetSymbol() in  ["B", "Si", "Te", "Se", "P"]: 
    #        return False
    for smarts in UNDESIRABLE_PATTERNS:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            return False
    if any([mol.HasSubstructMatch(Chem.MolFromSmarts(p)) for p in PYRROLE_FORM]):
        if not any([mol.HasSubstructMatch(Chem.MolFromSmarts(p)) for p in CORRECT_PYRROLE]):
            return False
    return not has_pains_alert(mol) and not has_bad_ring(mol)

def get_covalent_warheads(input):
    """
    return True if it contains any covalent warheads
    """
    warhead_list = []
    if type(input) is str:
        mol = Chem.MolFromSmiles(input)
    else:
        mol = input
    mol = Chem.AddHs(mol)
    for warhead_name, smarts in COVALENT_WARHEADS.items():
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            warhead_list.append(warhead_name)
    return warhead_list

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

def generate_inp_prompt(mol, **kwargs):
    '''
    kwargs are as follows
    hbd_range: if True, of the ranges that apply to the molecule (<=3, <=4, <=5, <=7, >7) choose a random one and put it in the prompt
    hba_range: if True, of the ranges that apply to the molecule (<=3, <=4, <=5, <=10, <=15, >15) choose a random one and put it in the prompt
    mw_range: if True, of the ranges that apply to the molecule (<=300, <=400, <=500, <=600, >600) choose a random one and put it in the prompt
    logp_range: if True, of the ranges that apply to the molecule (<=3, <=4, <=5, <=6, >6) choose a random one and put it in the prompt
    rotb_range: if True, of the ranges that apply to the molecule (<=7, <=10, >10) choose a random one and put it in the prompt
    fracsp3_range: if True, of the ranges that apply to the molecule (<=0.4, >0.4, >0.5, >0.6) choose a random one and put it in the prompt
    tpsa_range: if True, of the ranges that apply to the molecule (<=90, <=140, <=200, >200) choose a random one and put it in the prompt
    substruct: do a BRICS decomposition and pick a random substructure
    macrocycles: specify whether the molecule has macrocycles or not
    formula: the molecular formula in string format
    undesirable_smarts: specify whether there are any undesirable SMARTS substructures
    cov_warhead: specify whether there are any covalent warhead SMARTS substructures
    '''
    inp_base = 'Output a SMILES string for a drug-like molecule with the following properties: '
    if all(value is False for value in kwargs.values()):
        return "Output a SMILES string for a drug-like molecule:"
    specifications = []
    if mol is None: raise ValueError("Invalid SMILES string")
    if kwargs.get('hbd_range', True):
        num_hbd = Descriptors.NumHDonors(mol)
        hbd_ranges = []
        if num_hbd > 7: hbd_ranges.append('> 7')
        if num_hbd <= 7: hbd_ranges.append('<= 7')
        if num_hbd <= 5: hbd_ranges.append('<= 5')
        if num_hbd <= 4: hbd_ranges.append('<= 4')
        if num_hbd <= 3: hbd_ranges.append('<= 3')
        specifications.append(random.choice(hbd_ranges) + ' H-bond donors')
        
    if kwargs.get('hba_range', True):
        num_hba = Descriptors.NumHAcceptors(mol)
        hba_ranges = []
        if num_hba > 15: hba_ranges.append('> 15')
        if num_hba <= 15: hba_ranges.append('<= 15')
        if num_hba <= 10: hba_ranges.append('<= 10')
        if num_hba <= 5: hba_ranges.append('<= 5')
        if num_hba <= 4: hba_ranges.append('<= 4')
        if num_hba <= 3: hba_ranges.append('<= 3')
        specifications.append(random.choice(hba_ranges) + ' H-bond acceptors')
    
    if kwargs.get('mw_range', True):
        mol_weight = Descriptors.MolWt(mol)
        mw_ranges = []
        if mol_weight > 600: mw_ranges.append('> 600')
        if mol_weight <= 600: mw_ranges.append('<= 600')
        if mol_weight <= 500: mw_ranges.append('<= 500')
        if mol_weight <= 400: mw_ranges.append('<= 400')
        if mol_weight <= 300: mw_ranges.append('<= 300')
        specifications.append(random.choice(mw_ranges) + ' Molecular weight')
    
    if kwargs.get('logp_range', True):
        logp = Descriptors.MolLogP(mol)
        logp_ranges = []
        if logp > 6: logp_ranges.append('> 6')
        if logp <= 6: logp_ranges.append('<= 6')
        if logp <= 5: logp_ranges.append('<= 5')
        if logp <= 4: logp_ranges.append('<= 4')
        if logp <= 3: logp_ranges.append('<= 3')
        specifications.append(random.choice(logp_ranges) + ' LogP')

    if kwargs.get('rotb_range', True):
        num_rotb = Descriptors.NumRotatableBonds(mol)
        rotb_ranges = []
        if num_rotb > 10: rotb_ranges.append('> 10')
        if num_rotb <= 10: rotb_ranges.append('<= 10')
        if num_rotb <= 7: rotb_ranges.append('<= 7')
        specifications.append(random.choice(rotb_ranges) + ' Rotatable bonds')
    
    if kwargs.get('fracsp3_range', True):
        fracsp3 = Descriptors.FractionCSP3(mol)
        fracsp3_ranges = []
        if fracsp3 > 0.6: fracsp3_ranges.append('> 0.6')
        if fracsp3 > 0.5: fracsp3_ranges.append('> 0.5')
        if fracsp3 > 0.4: fracsp3_ranges.append('> 0.4')
        if fracsp3 <= 0.6 and fracsp3 >= 0.4: fracsp3_ranges.append('between 0.4 and 0.6')
        if fracsp3 <= 0.4: fracsp3_ranges.append('<= 0.4')
        specifications.append(random.choice(fracsp3_ranges) + ' Fraction sp3')
    
    if kwargs.get('tpsa_range', True):
        tpsa = Descriptors.TPSA(mol)
        tpsa_ranges = []
        if tpsa > 200: tpsa_ranges.append('> 200')
        if tpsa <= 200: tpsa_ranges.append('<= 200')
        if tpsa <= 140: tpsa_ranges.append('<= 140')
        if tpsa <= 90: tpsa_ranges.append('<= 90')
        specifications.append(random.choice(tpsa_ranges) + ' TPSA')

    if kwargs.get('substruct', True):
        brics_list = get_brics(mol)
        if brics_list != []:
            subs = random.choice(brics_list)
            specifications.append('a substructure of ' + subs)
        else:
            specifications.append('no BRICS substructure')
    
    if kwargs.get('macrocycle', True):
        ring_info = mol.GetRingInfo()
        has_macrocycle = False
        for ring in ring_info.AtomRings():
            if len(ring) >= 8:
                has_macrocycle = True
        if has_macrocycle: specifications.append('a macrocycle')
        else: specifications.append('no macrocycles')

    if kwargs.get('formula', True):
        specifications.append('A formula of ' + Chem.rdMolDescriptors.CalcMolFormula(mol))

    if kwargs.get('undesirable_smarts', True):
        if check_valid_pattern(mol): specifications.append('lacks bad SMARTS')
        else: 
            specifications.append('has bad SMARTS')

    if kwargs.get('cov_warhead', True):
        warheads_list = get_covalent_warheads(mol)
        if warheads_list != []:
            specifications.append('has covalent warheads (' + ' '.join(warheads_list) + ')')
        else: specifications.append('lacks covalent warheads')
    
    random.shuffle(specifications)
    try:
        inp_base += specifications[0]
        for s in specifications[1:]:
            inp_base += ', ' + s
        inp_base += ':'
        return inp_base
    except:
        return "Output a SMILES string for a drug-like molecule:"
