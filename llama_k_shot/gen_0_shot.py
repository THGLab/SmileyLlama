import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from tqdm import tqdm
import sys
sys.path.append('/path/to/scripts/')
from inference_tools import InferenceObject


random.seed(42)

def make_n_shot_prompt(n_shots):
    system_txt = 'You love and excel at generating SMILES strings of drug-like molecules'
    user_txt = 'Please generate a drug-like smiles string and no other output:'
    batches = []
    p=''
    samples = random.sample(lines, n_shots)
    samples = [s.rstrip() for s in samples]
    p += '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n' + system_txt
    for k in range(n_shots):
        p += '<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n' + user_txt
        p += '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' + samples[k]
    p += '<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n' + user_txt
    p += '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    return p



chembl = "/path/to/chembl_random_smiles.txt"
with open(chembl) as sf:
    lines = sf.readlines()


temperature = 1.0
model_path = "/path/to/Llama-3.1-8B-Instruct"
tokenizer_path = model_path
num_return_sequences = 128
max_new_tokens = 128#256
io = InferenceObject(model_path, tokenizer_path, num_return_sequences, temperature, max_new_tokens)

num_shots=0
label="128mnt"
print(num_shots)

k_shot = [make_n_shot_prompt(num_shots) for k in range(200)]

temps = [1.0]
for t in temps:
    raw_results = io.generate_strings(k_shot, generation_params={"temperature":t}, disable_tqdm=False)
    print(raw_results[0])
    strings = []
    for s in raw_results:
        strings += s[1]
    
    with open('t_'+str(t)+'_'+str(num_shots)+'_shot_label_'+label+'.txt', 'w+') as out:
        for s in strings:
            out.write(s+'\n')


