from rdkit import Chem
from smileyllama.control import *

controls = dict(
    hbd = HBDControl('>', 5),
    hba = HBAControl('<', 5),
    mwt = MolWtControl('>', 300),
    logp = LogPControl('>', 3),
    rtb = RotBondsControl('>', 5),
    tpsa = TPSAControl('>', 10),
    sp3 = SP3FractionControl('', (0.4, 0.6)),
    mc = MacrocycleControl(),
    cov = CovalentControl(type='acrylamides'),
    formula = FormulaControl('C2H6'),
    subs = SubstructureControl('COC'),
)


def test_controls():
    mol = Chem.MolFromSmiles('C=CC(=O)NCOCCCCc1ccccc1')
    print("\n===== Test Generate Inference Prompts =====")
    for k, c in controls.items():
        print(k, ':', c.create_prompt_for_inference())

    print("\n===== Test Generate Training Prompts =====")
    for k, c in controls.items():
        print(k, ':', c.create_prompt_for_training(mol))

    print("\n===== Test Applying Controls to Molecule =====")
    for k, c in controls.items():
        print(f"{k} {c.apply(mol)}")
