from rdkit import Chem
from rdkit.Chem import FilterCatalog, Descriptors, Crippen, rdMolDescriptors, BRICS
import json
from tqdm import tqdm
import random
from collections import namedtuple
import numpy as np
from pathlib import Path
import pickle
import sys

from sltools import property_tools
from sltools.property_tools import OUT_OF_RANGE, UNDESIRABLE_PATTERNS, CORRECT_PYRROLE, COVALENT_WARHEADS, params, PAINS_catalog
from sltools.inference_tools import InferenceObject

from sltools.property_tools import has_pains_alert, has_bad_ring, check_valid_pattern

def check_for_covalent_warheads(input):
    """
    return True if it contains any covalent warheads
    """
    if type(input) is str:
        mol = Chem.MolFromSmiles(input)
    else:
        mol = input
    for smarts in COVALENT_WARHEADS:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            return True
    return False

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


def check_for_macrocycle(mol):
    """
    return True if it contains any macrocycles
    """
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) >= 8:
            return True
    return False

def is_substructure(smiles_a, smiles_b):
    # Convert SMILES strings to RDKit molecules
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)

    if mol_a is None or mol_b is None:
        return False  # Invalid SMILES strings

    # Check if A contains a ghost atom
    if '*' in smiles_a:
        print("Ghost atom detected!")
        canonical_smiles_a = Chem.MolToSmiles(mol_a, canonical=True)
        frags = BRICS.BRICSDecompose(mol_b, returnMols=True, singlePass=True)
        for mol in frags:
            rw_mol = Chem.RWMol(mol)
            # Iterate over atoms in the molecule
            for atom in rw_mol.GetAtoms():
                # Check if the atom is a special marker (atomic number 0 and a specific isotope)
                if atom.GetAtomicNum() == 0:
                    # Replace the special marker with a hydrogen atom
                    new_atom = Chem.Atom(0)
                    rw_mol.ReplaceAtom(atom.GetIdx(), new_atom)
            canonical_smiles_frag = Chem.MolToSmiles(rw_mol, canonical=True)
            if canonical_smiles_frag == canonical_smiles_a:
                return True
        return False
    else:
        # Standard substructure check
        return mol_b.HasSubstructMatch(mol_a)

def check_molecular_properties(smiles, property_constraints, substructure=False):
    # Create RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False  # Invalid SMILES
    # Define property calculation functions
    property_functions = {
        'hbd': Descriptors.NumHDonors,
        'hba': Descriptors.NumHAcceptors,
        'mw': Descriptors.ExactMolWt,
        'logp': Crippen.MolLogP,
        'rotb': Descriptors.NumRotatableBonds,
        'fracsp3': Descriptors.FractionCSP3,
        'tpsa': Descriptors.TPSA,
        'macrocycle': check_for_macrocycle,
        'no_undesirable_smarts': check_valid_pattern,
        'cov_warhead': get_covalent_warheads,
        'formula': rdMolDescriptors.CalcMolFormula,
    }
    # Check each constraint
    for prop, constraint in property_constraints.items():
        if prop not in property_functions:
            raise ValueError(f"Unsupported property: {prop}")
        value = property_functions[prop](mol)
        if isinstance(constraint, tuple):
            operator, limit = constraint
            if operator == '<=':
                if not value <= limit:
                    return False
            elif operator == '<':
                if not value < limit:
                    return False
            elif operator == '>=':
                if not value >= limit:
                    return False
            elif operator == '>':
                if not value > limit:
                    return False
            elif operator == '==':
                if not value == limit:
                    return False
            elif operator == 'between':
                if not limit[0] <= value <= limit[1]:
                    return False
            elif operator == 'is a superset of':
                assert isinstance(limit, set) and isinstance(value, list), "need to pass in a set for limit and a list for value"
                set_value = set(value)
                if not set_value >= limit:
                    return False
            else:
                raise ValueError(f"Unsupported operator: {operator}")
        else:
            raise ValueError(f"Invalid constraint format for {prop}")
    if substructure != False:
        return is_substructure(substructure, smiles)
    return True

with open('essential_frags.csv', 'r') as frag_csv:
    lines = frag_csv.readlines()
substructs = [l.split(',')[0] for l in lines[1:]]

properties = ['exactly ' + str(k) + ' H-bond donors' for k in range(0, 6)]
properties += ['exactly ' + str(k) + ' H-bond acceptors' for k in range(0, 11)]
properties += ['<= ' + str(k) + ' H-bond donors' for k in [3,4,5,7]]
properties += ['<= ' + str(k) + ' H-bond acceptors' for k in [3,4,5,10,15]]
properties += ['<= ' + str(k) + ' Molecular weight' for k in [300,400,500,600]]
properties += ['<= ' + str(k) + ' LogP' for k in [3,4,5,6]]
properties += ['<= ' + str(k) + ' Rotatable bonds' for k in [7,10]]
properties += ['> 10 Rotatable bonds']
properties += ['> ' + str(k) + ' Fraction sp3' for k in [0.4, 0.5, 0.6]]
properties += ['< 0.4 Fraction sp3']
properties += ['<= ' + str(k) + ' TPSA' for k in [90, 140, 200]]
properties += ['a macrocycle', 'no macrocycles']
properties += ['lacks bad SMARTS', 'has bad SMARTS']
properties += ['lacks covalent warheads'] + ['has covalent warheads ('+warhead_name+')' for warhead_name in COVALENT_WARHEADS.keys()]
properties += ['a substructure of ' + s for s in substructs]
properties += ['<= 5 H-bond donors, <= 10 H-bond acceptors, <= 500 Molecular weight, <= 5 LogP']
properties += ['<= 3 H-bond donors, <= 3 H-bond acceptors, <= 300 Molecular weight, <= 3 LogP']

constraint_list = [{'hbd': ('==', k)} for k in range(0, 6)]
constraint_list += [{'hba': ('==', k)} for k in range(0, 11)]
constraint_list += [{'hbd': ('<=', k)} for k in [3, 4, 5, 7]]
constraint_list += [{'hba': ('<=', k)} for k in [3, 4, 5, 10, 15]]
constraint_list += [{'mw': ('<=', k)} for k in [300, 400, 500, 600]]
constraint_list += [{'logp': ('<=', k)} for k in [3, 4, 5, 6]]
constraint_list += [{'rotb': ('<=', k)} for k in [7, 10]]
constraint_list += [{'rotb': ('>', 10)}]
constraint_list += [{'fracsp3': ('>', k)} for k in [0.4, 0.5, 0.6]]
constraint_list += [{'fracsp3': ('<', 0.4)}]
constraint_list += [{'tpsa': ('<=', k)} for k in [90, 140, 200]]
constraint_list += [{'macrocycle': ('==', True)}, {'macrocycle': ('==', False)}]
constraint_list += [{'no_undesirable_smarts': ('==', True)}, {'no_undesirable_smarts': ('==', False)}]
constraint_list += [{'cov_warhead': ('==', [])}] + [{'cov_warhead': ('is a superset of', {warhead_name})} for warhead_name in COVALENT_WARHEADS.keys()]
constraint_list += [{'substruct': s} for s in substructs]
constraint_list += [{'mw': ('<=', 500), 'hbd': ('<=', 5), 'hba': ('<=', 10), 'logp': ('<=', 5)}]
constraint_list += [{'mw': ('<=', 300), 'hbd': ('<=', 3), 'hba': ('<=', 3), 'logp': ('<=', 3)}]

run_params = list(zip(properties, constraint_list))


temperature = float(sys.argv[1])
model_path = sys.argv[2]
output_path = sys.argv[3]

system_text = "You love and excel at generating SMILES strings of drug-like molecules"
tokenizer_path = model_path
num_return_sequences = 128
max_new_tokens = 128
io = InferenceObject(model_path, tokenizer_path, num_return_sequences, temperature, max_new_tokens)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

outputs = []
for rp in tqdm(run_params):
    user_text = 'Output a SMILES string for a drug like molecule with the following properties: ' + rp[0] + ':'
    prompts = [f"### Instruction:\n{system_text}\n\n### Input:\n{user_text}\n\n### Response:\n"]*8
    raw_results = []
    strings = io.generate_strings(prompts, generation_params={"temperature":temperature}, disable_tqdm=True)
    for s in strings[0:1000]:
        raw_results += s[1]
    prompt_results = []
    for raw_result in raw_results:
        if 'substruct' in rp[1]:
            valid = check_molecular_properties(raw_result, {}, substructure=rp[1]['substruct'])
        else:
            valid = check_molecular_properties(raw_result, rp[1], substructure=False)
        prompt_results.append((raw_result, valid))
    outputs.append((prompts[0], prompt_results))


with open(output_path+'/temperature_'+str(temperature)+'.pkl', 'wb') as fil:
    pickle.dump(outputs, fil)

