import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data'))
import iminer_dl

from smileyllama.score import iMinerDrugLikeliness


def test_iminer_dl():
    smiles = [
        "CCCCOC",
        "c1ccccn1"
    ]
    ref_obj = iminer_dl.DrugLikeliness()
    sl_obj = iMinerDrugLikeliness()
    for smi in smiles:
        ref = ref_obj.calc_score(smi)
        sl = sl_obj.compute(smi)
        assert abs(ref - sl) < 1e-5, f'SmileyLlama implementation not correct for {smi}: {sl:.5f} vs {ref:.5f} (sl vs ref)'