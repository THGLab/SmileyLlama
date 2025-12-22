__all__ = ['Response', 'Pipeline']
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Union, Literal
import warnings
import math
import multiprocessing as mp
import logging
import gc

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
# transformers tokenizers will raise warnings when multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .filter import Filter, FilterResultSingle, apply_filters



class Response:
    def __init__(self):
        self._identifiers = set()
        self._all_results = []
        self._num_generated = 0
        self._num_valid = 0
        self._num_unique = 0
        self._num_pass_filters = 0
        self._num_unique_valid = 0
        self._num_unique_pass_filters = 0
    
    def add_result(self, result: List[FilterResultSingle]):
        identifier = result.canonical_smiles if result.valid else result.smiles
        result.unique = identifier not in self._identifiers
        self._identifiers.add(identifier)
        self._all_results.append(result)
        self._num_generated += 1
        self._num_valid += result.valid
        self._num_unique += result.unique
        self._num_pass_filters += result.success
        self._num_unique_valid += result.unique and result.valid
        self._num_unique_pass_filters += result.unique and result.success
    
    @property
    def num_generated(self) -> int:
        return self._num_generated
    
    @property
    def num_valid(self) -> int:
        return self._num_valid
    
    @property
    def num_pass_filters(self) -> int:
        return self._num_pass_filters
    
    @property
    def num_unique(self) -> int:
        return self._num_unique
    
    @property
    def num_unique_valid(self) -> int:
        return self._num_unique_valid
    
    @property
    def num_unique_pass_filters(self) -> int:
        return self._num_unique_pass_filters
    
    @property
    def df(self) -> pd.DataFrame:
        df = pd.DataFrame([asdict(r) for r in self._all_results])
        random_smiles = []
        for r in self._all_results:
            if not r.valid:
                random_smiles.append(r.smiles)
            else:
                random_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(r.smiles), doRandom=True))
        df['random_smiles'] = random_smiles
        return df
    
    def get_all_smiles(self) -> List[str]:
        return [r.smiles for r in self._all_results]

    def __str__(self) -> str:
        return (
            'Response'
            f'\n - generated: {self.num_generated}'
            f'\n - valid: {self.num_valid} ({self.num_valid/self.num_generated*100:.2f}%)'
            f'\n - unique: {self.num_unique} ({self.num_unique/self.num_generated*100:.2f}%)'
            f'\n - pass_filters: {self.num_pass_filters} ({self.num_pass_filters/self.num_generated*100:.2f}%)'
            f'\n - unique & valid: {self.num_unique_valid} ({self.num_unique_valid/self.num_generated*100:.2f}%)'
            f'\n - unique & pass_filters: {self.num_unique_pass_filters} ({self.num_unique_pass_filters/self.num_generated*100:.2f}%)'
        )


class Pipeline:

    def __init__(
        self, 
        model_path, 
        lora_model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None, 
        device: Optional[str] = None, 
        system_prompt: str = 'You love and excel at generating SMILES strings of drug-like molecules',
        user_prompt_base: str = 'Output a SMILES string for a drug like molecule',
        prompt_format: Literal['chat', 'instruct'] = 'instruct',
        nprocs: int = -1,
        logger: Optional[logging.Logger] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # tokenizer
        tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
        )
        if lora_model_path:
            model = PeftModel.from_pretrained(model, lora_model_path)
        self.model = model
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)
        self.model.eval()

        # prompt
        self.system_prompt = system_prompt
        self.user_prompt_base = user_prompt_base
        self.prompt_format = prompt_format

        # leave 2 processors for other processes
        self.nprocs = mp.cpu_count() - 2 if nprocs <= 0 else nprocs
        self.nprocs = max(1, self.nprocs)

        self.logger = logger
    
    def create_prompt(self, filters: List[Union[Filter, str]] = list()):
        if len(filters) > 0:
            properties = []
            for ft in filters:
                if isinstance(ft, Filter) or hasattr(ft, 'create_prompt'):
                    properties.append(ft.create_prompt())
                elif isinstance(ft, str):
                    properties.append(ft)
                else:
                    raise RuntimeError(f"Not valid property: {ft}")
            user_prompt = self.user_prompt_base + ' with the following properties: ' + ', '.join(properties) + ':'
        else:
            user_prompt = self.user_prompt_base + ":"
        
        if self.prompt_format == 'instruct':
            prompt = f"### Instruction:\n{self.system_prompt}\n\n### Input:\n{user_prompt}\n\n### Response:\n"
        else:
            messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_prompt}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        return prompt
    
    def log(self, msg):
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)
    
    def warn(self, msg):
        if self.logger is not None:
            self.logger.warning(msg)
        else:
            warnings.warn(msg)

    def generate(
        self,
        num: int,
        properties: List[Union[Filter, str]] = list(),
        *,
        raw_prompt: Optional[str] = None,
        num_per_call: int = 128,
        enforce_valid: bool = False,
        enforce_filters: bool = False,
        enforce_unique: bool = False,
        num_max_generation: Union[int, Literal['auto']] = 'auto',
        temperature_increment: float = 0.0,
        max_temperature: float = 1.1,
        **generation_params
    ):

        if not enforce_valid:
            if enforce_filters:
                enforce_filters = False
                self.warn("enforce_filters is forcibly set to False because enforce_valid is False")

        # max generation
        if num_max_generation == 'auto':
            num_max_generation = 5 * num
        elif num_max_generation <= 0:
            num_max_generation = math.inf
        elif not isinstance(num_max_generation, int):
            raise RuntimeError(f"num_max_generation is invalid: {num_max_generation}")
        
        # transformers parameters
        do_sample = generation_params.pop('do_sample', True)
        max_new_tokens = generation_params.pop('max_new_tokens', 128)
        num_per_call = min(num_per_call, num)
        if 'num_return_sequences' in generation_params:
            self.warn("num_per_call will be overridden by num_return_sequences")
        num_return_sequences = generation_params.pop('num_return_sequences', num_per_call)
        eos_token_id = generation_params.pop(
            'eos_token_id',
            [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        )
        # temperature scheduling
        start_temperature = generation_params.pop('temperature', 1.0)
        temperature = start_temperature

        # prompt
        prompt = self.create_prompt(properties) if raw_prompt is None else raw_prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        # filters
        filters = [ft for ft in properties if isinstance(ft, Filter)]

        # response object
        response = Response()
        num_generated = 0
        num_success = 0
        with torch.no_grad():
            while num_success < num and num_generated < num_max_generation:
                num_gen = min(num - num_success, num_max_generation-num_generated)
                num_calls = int(math.ceil(num_gen/ num_per_call))

                smiles_temp = []
                for _ in tqdm(range(num_calls), desc='Generating SMILES'):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        num_return_sequences=num_return_sequences,
                        eos_token_id=eos_token_id,
                        temperature=temperature,
                        **generation_params,
                    )
                    for output in outputs:
                        smi_raw = self.tokenizer.decode(output[input_len:], skip_special_tokens=True)
                        smi_raw = smi_raw.strip()
                        smiles_temp.append(smi_raw)    
                
                # apply filters
                filter_results = apply_filters(smiles_temp, filters, self.nprocs)
                for result in filter_results:
                    response.add_result(result)

                    num_generated = response.num_generated
                    num_success = response.num_unique if enforce_unique else response.num_generated
                    if enforce_valid:
                        num_success = response.num_unique_valid if enforce_unique else response.num_valid
                    if enforce_filters:
                        num_success = response.num_unique_pass_filters if enforce_unique else response.num_pass_filters

                    if (num_success >= num) or (num_generated >= num_max_generation):
                        break
                
                # increase temperature for next round
                temperature = min(max_temperature, temperature+temperature_increment)
        
        return response
    
    def free(self):
        self.model.to("cpu")
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
