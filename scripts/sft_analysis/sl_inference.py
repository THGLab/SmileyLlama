'''
This runs Llama or SmileyLlama in parallel using accelerate. 
Make sure to run with the appropriate env (axo works great)
Run this (and files that use InferenceObject) using accelerate launch program.py
'''
import torch
from accelerate import PartialState
from accelerate.utils import gather_object
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, BitsAndBytesConfig
import bitsandbytes, flash_attn
import time
from tqdm import tqdm

class InferenceObject():
    def __init__(self, model_path, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        self.distributed_state = PartialState()
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.distributed_state.device)

    def create_prompt(
        self, 
        user_txt="Output a SMILES string for a drug like molecule:", 
        system_txt="You love and excel at generating SMILES strings of drug-like molecules"):
        messages = [{"role": "system", "content": system_txt}, {"role": "user", "content": user_txt}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def generate_smiles_strings(self, prompts, num_return_sequences, max_new_tokens, generation_params={}):
        #tokenized_prompts = [self.tokenizer(prompts[k], padding=True, return_tensors="pt") for k in range(len(prompts))]
        tokenized_prompts = [self.tokenizer(p, padding=True, return_tensors="pt") for p in prompts]
        completions = []
        assert num_return_sequences > 1
        with self.distributed_state.split_between_processes(tokenized_prompts, apply_padding=True) as batched_prompts:
            for batch in tqdm(batched_prompts, desc=f"Generating completions on device {self.distributed_state.device}"):
                batch = batch.to(self.distributed_state.device)
                outputs = self.model.generate(
                    **batch, 
                    max_new_tokens=max_new_tokens,   
                    do_sample=True, 
                    num_return_sequences=num_return_sequences, 
                    eos_token_id=self.tokenizer.eos_token_id, 
                    **generation_params)
                cut_outputs = outputs[:,len(batch["input_ids"][0]):]
                generated_text = self.tokenizer.batch_decode(cut_outputs, skip_special_tokens=True)
                completions += generated_text
        completions_gathered = gather_object(completions)
        if self.distributed_state.is_main_process:
            return completions_gathered
        else:
            return None

if __name__ == "__main__":
    sl_path = 'THGLab/Llama-3.1-8B-SmileyLlama-1.1'
    sl = InferenceObject(sl_path, sl_path)
    prompt = sl.create_prompt()
    prompts = [prompt, prompt, prompt]
    smiles = sl.generate_smiles_strings(prompts, 5,256, generation_params={"top_k":3})
    if smiles is not None:
        for s in smiles:
            print(s.rstrip())
        print(len(smiles))