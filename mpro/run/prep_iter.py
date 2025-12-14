#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import random
import json
import sys, os
import csv
import transformers
import torch
from tqdm import tqdm
import sys
from rdkit import Chem

def make_dpo_dataset(scoresfile, num_pairings, prompt):
    # Read the file
    smiles_scores=[]
    with open(scoresfile, 'r') as fil:
        for row in fil:
            try:
                r = row.rstrip().split(',')
                smiles_scores.append((r[0], float(r[1])))
            except:
                print('score not found in row: ' + str(row))

    # Sort the list according to the scores
    data = sorted(smiles_scores, key=lambda x: x[1])

    # Pair elements randomly
    pairings = []
    for _ in range(num_pairings):
        for k in range(0, len(data)):
            for l in range(0, 1000):
                sample = random.choice(data)
                if sample != data[k]:
                    break
            if sample[1] > data[k][1]:
                pairings.append((sample[0], data[k][0], sample[1], data[k][1]))
            else:
                pairings.append((data[k][0], sample[0], data[k][1], sample[1]))

    # Build the jsonl dataset for DPO
    template = {
        'source':'',
        'instruction': prompt,
        'chosen_response':'',
        'rejected_response': '',
        'chosen_avg_rating': '',
        'rejected_avg_rating': '',
        'chosen_model': ''
        }

    jsondata = [{**template, 'chosen_response':nl[0], 'rejected_response':nl[1],
                 'chosen_avg_rating':nl[2], 'rejected_avg_rating':nl[3]} for nl in pairings]
    return jsondata


def write_jsondata(jsondata, outputfile):
    with open(outputfile, 'w+') as out:
        for item in jsondata:
            out.write(json.dumps(item) + '\n')

def load_model(model, nsamples):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Pytorch Using {device} device')
    pipeline=transformers.pipeline('text-generation',
                                   model=model,
                                   max_new_tokens=128,
                                   temperature=1.0,
                                   num_return_sequences=nsamples,
                                   model_kwargs={'torch_dtype': torch.bfloat16}, device_map='auto')
    return pipeline

def inference(prompt, pipeline, outputfile, samplefile, ntries, max_structs):
    smileys = set([])
    sample = []
    for k in tqdm(range(0, ntries)):
        p = pipeline(prompt)
        smileys.update([s['generated_text'].splitlines()[-1] for s in p])
        sample += [s['generated_text'].splitlines()[-1] for s in p]
        if len(smileys) >= max_structs:
            break
    smileys = list(smileys)[0:max_structs]
    with open(outputfile, 'w+') as out:
        for item in smileys:
            out.write(item + '\n')
    with open(samplefile, 'w+') as smp:
        for item in sample:
            smp.write(item + '\n')

def create_prompt(task_name):
    system_txt="You love and excel at generating SMILES strings of drug-like molecules"
    user_txt="Output a SMILES string for a drug like molecule with the following properties: " + task_name
    prompt = f"### Instruction:\n{system_txt}\n\n### Input:\n{user_txt}\n\n### Response:\n"
    return prompt

def score_sars2pro(smilesfile, scoresfile, property_string):
    scoring_python = '/path/to/iminer/python'
    inp_smiles = ['--input_smiles', str(smilesfile)]
    out_scores = ['--output_csv', str(scoresfile)]
    scoring_program = '/path/to/mpro/tools/integrated_score.py'
    subprocess.run([scoring_python, scoring_program] + inp_smiles + out_scores)
    return


def inf_and_score(pipeline, job_directory, task_name, iteration, property_string, ntries):
    prompt = create_prompt(task_name)
    max_structs = 2000
    smilesfile = os.path.join(job_directory, 'smiles_'+property_string+'_iter'+str(iteration)+'.txt')
    samplefile = os.path.join(job_directory, 'sample_'+property_string+'_iter'+str(iteration)+'.txt')
    scoresfile = os.path.join(job_directory, 'scores_'+property_string+'_iter'+str(iteration)+'.csv')
    inference(prompt, pipeline, smilesfile, samplefile, ntries, max_structs)
    if 'QED' in task_name:
        score_tdc(smilesfile, scoresfile, property_string)
    elif 'SARS2PRO' in task_name:
        score_sars2pro(smilesfile, scoresfile, property_string)
    return make_dpo_dataset(scoresfile, num_pairings, prompt)

if __name__ == '__main__':
    job_directory = '/path/to/mpro/run'
    iteration = sys.argv[1]
    nsamples = 128
    num_pairings = 8
    jsd = []
    properties = ['SARS2PRO']
    pipeline = load_model(os.path.join(job_directory, 'outputs/merged'), nsamples)
    for p in properties:
        task_name = 'High '+p
        jsd += inf_and_score(pipeline, job_directory, task_name, iteration, p, 100)
    dpodataset = os.path.join(job_directory,'dpodataset/dpodataset.jsonl')
    write_jsondata(jsd, dpodataset)
