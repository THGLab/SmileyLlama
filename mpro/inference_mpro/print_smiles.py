import sys
from tqdm import tqdm
from tdc import Oracle
import subprocess
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import pickle

with open("s2p.pkl", 'rb') as f:
    s2p = pickle.load(f)

with open("ro5.pkl", 'rb') as f:
    ro5 = pickle.load(f)


def check_lipinski(smiles, debug=False):
    """
    Check if a SMILES string satisfies Lipinski's Rule of Five.
    Returns True if all rules are satisfied, False otherwise.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    if debug:
        print('H-bond donors:    ',hbd)
        print('H-bond acceptors: ',hba)
        print('Molecular weight: ',mw)
        print('LogP:             ',logp)
    
    return (mw <= 500 and 
            logp <= 5 and 
            hbd <= 5 and 
            hba <= 10)

def print_lipinski_followers(smiles_scores, print_scores=True, count=True, sort=True, reverse=True, remove_duplicates=True):
    lipinski_followers = []
    for s in smiles_scores:
        if check_lipinski(s[0]):
            lipinski_followers.append(s)
    
    if remove_duplicates:
        lipinski_followers = list(set(lipinski_followers))
    
    if sort:
        lipinski_followers.sort(key=lambda x: x[1], reverse=reverse)
    
    for s in lipinski_followers:
        if print_scores:
            print(s[0], s[1])
        else:
            print(s[0])
    
    if count:
        print(len(lipinski_followers), "molecules obey Lipinski's rule of 5")
        
print("SARS2PRO with scores")        
print_lipinski_followers(s2p)

print("SARS2PRO + Rule of 5 with scores")
print_lipinski_followers(ro5)


print("SARS2PRO without scores")
print_lipinski_followers(s2p, print_scores=False)

print("SARS2PRO + Rule of 5 without scores")
print_lipinski_followers(ro5, print_scores=False)



