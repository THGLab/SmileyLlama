import os
import numpy as np
from smileyllama.score import *
from smileyllama.filter import *
from smileyllama.inference import Pipeline


def test_inference():
    # model_path = '/global/scratch/users/ericwangyz/smiley-enamine/model-1B-chat-from-alcapa/model/merged'
    model_path = '/global/scratch/users/ericwangyz/smiley-enamine/model-1B/model/merged'
    pipline = Pipeline(model_path, prompt_format='instruct', nprocs=4)

    print("===== Without Control =====")
    response = pipline.generate(200, [])
    print(response)

    print("===== With Control =====")
    control = NumericScoreFilter.init_from_class(MolWt, '<', 500)
    response = pipline.generate(200, ['High SARS2PRO', control])
    print(response)