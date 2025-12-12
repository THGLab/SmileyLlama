from smileyllama.control import MolWtControl
from smileyllama.inference import SmileyLlamaInference


def test_inference():
    model_path = ''
    pipline = SmileyLlamaInference(model_path)
    response = pipline.generate(1000, [MolWtControl('>', 500)])
    print(response)