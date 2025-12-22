import os
import multiprocessing as mp
from typing import List, Literal, Tuple, Dict
from tqdm import tqdm
import random
from functools import partial
import json

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
# from .control import *


# def make_sft_data_one_smi(
#     smi: str, 
#     controls: List[PropertyControl], 
#     format: Literal['chat', 'instruct'],
#     system_prompt: str,
#     user_prompt_base: str,
#     shuffle: bool,
#     random_controls: bool,
#     random_smiles: bool,
#     kekule_smiles: bool, 
#     explict_h_smiles: bool,
#     isomeric_smiles: bool
# ):
#     m = Chem.MolFromSmiles(smi)
#     if m is None:
#         return {}
    
#     newsmi = Chem.MolToSmiles(
#         m, 
#         isomericSmiles=isomeric_smiles,
#         doRandom=random_smiles,
#         allHsExplicit=explict_h_smiles,
#         kekuleSmiles=kekule_smiles
#     )

#     property_prompts = []
#     for c in controls:
#         do = True if not random_controls else 0.5 > random.random()
#         if do:
#             prompt = c.create_prompt_for_training(m)
#             if prompt:
#                 property_prompts.append(prompt)

#     if len(property_prompts) > 0:
#         if shuffle:
#             random.shuffle(property_prompts)
#         user_prompt = f'{user_prompt_base} with the following properties: {", ".join(property_prompts)}:'
#     else:
#         user_prompt = f'{user_prompt_base}:'

#     if format == 'instruct':
#         return {
#             "instruction": system_prompt, "input": user_prompt, "response": newsmi
#         }
#     elif format == 'chat':
#         return {
#             "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}, {"role": "assistant", "content": newsmi}]
#         }
#     else:
#         raise NotImplementedError(f"Not supported format: {format}")


# DEFAULT_CONTROLS = [
#     HBDControl,
#     HBAControl,
#     MolWtControl,
#     LogPControl,
#     RotBondsControl,
#     TPSAControl,
#     SP3FractionControl,
#     MacrocycleControl(),
#     CovalentControl,
#     FormulaControl,
#     SubstructureControl,
#     NoBadSubstructureControl
# ]

# def make_sft_data(
#     smiles: List[str],
#     out_jsonl: os.PathLike,
#     controls: List[PropertyControl] = DEFAULT_CONTROLS,
#     format: Literal['chat', 'instruct'] = 'chat',
#     nprocs: int = 1,
#     system_prompt: str = 'You love and excel at generating SMILES strings of drug-like molecules',
#     user_prompt_base: str = 'Output a SMILES string for a drug like molecule',
#     shuffle: bool = True,
#     random_controls: bool = True,
#     random_smiles: bool = True,
#     kekule_smiles: bool = False, 
#     explict_h_smiles: bool = False,
#     isomeric_smiles: bool = True
# ):
#     func = partial(
#         make_sft_data_one_smi, 
#         controls=controls, format=format, system_prompt=system_prompt,
#         user_prompt_base=user_prompt_base, shuffle=shuffle, random_controls=random_controls,
#         random_smiles=random_smiles, kekule_smiles=kekule_smiles, explict_h_smiles=explict_h_smiles, isomeric_smiles=isomeric_smiles
#     )
    
#     if nprocs > 1:
#         datas = []
#         with mp.Pool(nprocs) as p:
#             for data in tqdm(p.imap_unordered(func, smiles), total=len(smiles), desc='Processing SMILES'):
#                 datas.append(data)
#     else:
#         datas = [func(smi) for smi in tqdm(smiles, desc='Processing SMILES')]
    
#     with open(out_jsonl, 'w') as f:
#         for data in tqdm(datas, desc='Write to jsonl'):
#             if data:
#                 f.write(json.dumps(data) + '\n')
    
#     return datas


def make_dpo_data_from_pairs(
    pairs: List[Tuple[str, str]],
    out_jsonl: os.PathLike,
    format: Literal['chat', 'instruct'] = 'instruct',
    system_prompt: str = 'You love and excel at generating SMILES strings of drug-like molecules',
    user_prompt_base: str = 'Output a SMILES string for a drug like molecule',
    properties: List[str] = list()
) -> List[Dict[str, str]]:
    if len(properties) > 0:
        user_prompt = f'{user_prompt_base} with the following properties: {", ".join(properties)}:'
    else:
        user_prompt = f'{user_prompt_base}:'
    
    datas = []
    random.shuffle(pairs)
    for pair in pairs:
        datas.append({"system": system_prompt, "prompt": user_prompt, "chosen": pair[0], "rejected": pair[1]})
    
    with open(out_jsonl, 'w') as f:
        for data in tqdm(datas, desc='Make DPO data'):
            if data:
                f.write(json.dumps(data) + '\n')
    return datas


def make_dpo_data_from_scores(
    responses: List[str],
    scores: np.ndarray,
    out_jsonl: os.PathLike,
    num_pairs_per_response: int = 2,
    margin: float = 0.1,
    **kwargs
):

    pairs = []
    scores = np.array(scores).flatten() if not isinstance(scores, np.ndarray) else scores.flatten()
    assert scores.shape[0] == len(responses), "Length of scores and SMILES does not match"
    for i in range(len(responses)):
        choices = np.argwhere(np.abs(scores - scores[i]) > margin).flatten()
        npair = min(num_pairs_per_response, len(choices))
        indices = np.random.choice(choices, npair)
        for index in indices:
            if scores[index] > scores[i]:
                chosen = responses[index]
                rejected = responses[i]
            else:
                chosen = responses[i]
                rejected = responses[index]
            pairs.append((chosen, rejected))
    
    return make_dpo_data_from_pairs(pairs, out_jsonl, **kwargs)

    

