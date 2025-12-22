import random
import numpy as np
from smileyllama.score import Score, accept_smiles, Normalizer


class RandomScore(Score):

    @accept_smiles
    def compute(self, mol):
        return random.random()


class Sigmoid(Normalizer):

    def transform(self, data):
        return 1/(1+np.exp(-data))