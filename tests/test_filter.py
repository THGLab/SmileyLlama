import pytest
from rdkit import Chem
from smileyllama.score import *
from smileyllama.filter import NumericScoreFilter

# controls = dict(
#     hbd = HBDControl('>', 5),
#     hba = HBAControl('<', 5),
#     mwt = MolWtControl('>', 300),
#     logp = LogPControl('>', 3),
#     rtb = RotBondsControl('>', 5),
#     tpsa = TPSAControl('>', 10),
#     sp3 = SP3FractionControl('', (0.4, 0.6)),
#     mc = MacrocycleControl(12),
#     cov = CovalentControl(type='acrylamides'),
#     formula = FormulaControl('C2H6'),
#     subs = SubstructureControl('COC'),
#     bad = NoBadSubstructureControl(),
# )

filters = {
    "hbd": NumericScoreFilter(NumHBD(), StepNormalizer('>', 5)),
    "hba": NumericScoreFilter(NumHBA(), StepNormalizer('<', 5)),
    "molwt": NumericScoreFilter(MolWt(), StepNormalizer('>', 3)),
    "logp": NumericScoreFilter.init_from_class(LogP, '<', 3)
}


@pytest.mark.parametrize('smi', [
    'C=CC(=O)NCOCCCCc1ccccc1',
    'OO'
])
def test_filters(smi):
    print("\n===== Test Generate Inference Prompts =====")
    for k, ft in filters.items():
        print(k, ':', ft.create_prompt())

    print("\n===== Test Applying Controls to Molecule =====")
    for k, ft in filters.items():
        print(f"{k} {ft.apply(smi)}")