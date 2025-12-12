from typing import Optional, List, Union
import warnings
import math
from tqdm import tqdm
import torch
from rdkit import Chem
from dataclasses import dataclass, field
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from .control import PropertyControl, PropertyControlResult, apply_controls


@dataclass
class SmileyLlamaResponse:
    num_success: int
    num_generated: int
    control_enforced: bool
    success_rate: Union[float, None] = field(init=False)
    controls: List[PropertyControl]
    smiles: List[str]
    control_results: List[PropertyControlResult] = field(repr=False)

    def __post_init__(self):
        if not self.control_enforced:
            self.success_rate = None
        else:
            self.success_rate = self.num_success / self.num_generated


class SmileyLlamaInference:
    def __init__(self, model_path, tokenizer_path: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # tokenizer
        tokenizer_path = model_path if tokenizer_path is None else tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
        )
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)
        self.model.eval()
    
    @classmethod
    def create_prompt(self, controls: List[PropertyControl] = list()):
        system_txt = "You love and excel at generating SMILES strings of drug-like molecules"
        user_txt = "Output a SMILES string for a drug like molecule with the following properties:"
        if len(control_txt):
            control_txt = ' ' + ', '.join([c.create_prompt_for_inference() for c in controls]) + ':'
        else:
            control_txt = ""
        prompt = f"### Instruction:\n{system_txt}\n\n### Input:\n{user_txt}{control_txt}\n\n### Response:\n"
        return prompt
    
    def log(self, msg):
        print(msg)
    
    def warn(self, msg):
        warnings.warn(msg)

    def generate(
        self,
        num: int,
        controls: List[PropertyControl] = list(),
        raw_prompt: Optional[str] = None,
        num_per_call: int = 128,
        enforce: bool = True,
        **generation_params
    ):
        
        if raw_prompt is None:
            prompt = self.create_prompt(controls)
        else:
            prompt = raw_prompt
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        do_sample = generation_params.pop('do_sample', True)
        max_new_tokens = generation_params.pop('max_new_tokens', 256)
        if 'num_return_sequences' in generation_params:
            self.warn("num_return_sequences will be overwritten by num_per_call")
        num_return_sequences = generation_params.pop('num_return_sequences', num_per_call)
        eos_token_id = generation_params.pop(
            'eos_token_id',
            [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        )

        smiles_list = []
        control_results = []
        num_success = 0
        num_generated = 0
        with torch.no_grad():
            while num_success < num:
                num_calls = int(math.ceil((num - num_success)/ num_per_call))
                smiles_temp = []
                for _ in tqdm(num_calls):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        num_return_sequences=num_return_sequences,
                        eos_token_id=eos_token_id,
                        **generation_params,
                    )
                    for output in outputs:
                        generated_text = self.tokenizer.decode(output[input_len:], skip_special_tokens=True)
                        smiles_temp.append(generated_text.strip())
                
                if not enforce:
                    smiles_list += smiles_temp[:(num-num_success)]
                    num_success += len(smiles_list)
                    num_generated + len(smiles_list)
                else:
                    crs = apply_controls(smiles_temp, controls)
                    for cr in crs:
                        if cr.success:
                            smiles_list.append(cr.smiles)
                            num_success += 1
                        num_generated += 1
                        control_results.append(cr)
                        if num_success == num:
                            break
                    self.log(f"Generated {len(control_results)} SMILES, {len(smiles_list)} satisfy controls")
        
        return SmileyLlamaResponse(num_success, num_generated, enforce, controls, smiles_list, control_results)
