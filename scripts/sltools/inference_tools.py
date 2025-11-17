import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, BitsAndBytesConfig
import bitsandbytes, flash_attn
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class InferenceObject():
    def __init__(self, model_path, tokenizer_path, num_return_sequences, temperature, max_new_tokens):
        self.num_gpus = torch.cuda.device_count()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'PyTorch Using {device} device with {self.num_gpus} GPUs')
        self.pipelines = []
        for i in range(self.num_gpus):
            with torch.cuda.device(i):
                pipeline = transformers.pipeline(
                'text-generation',
                model=model_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                model_kwargs={'torch_dtype': torch.bfloat16},
                device_map=f'cuda:{i}'
                )
                self.pipelines.append(pipeline)

    def single_inference(self, pipeline, prompts, generation_params, disable_tqdm=False):
        outputs = []
        for prompt in tqdm(prompts, disable=disable_tqdm):
            generation = pipeline(prompt, pad_token_id=pipeline.tokenizer.eos_token_id, **generation_params)
            output = [item["generated_text"][len(prompt):] for item in generation]
            outputs.append((prompt, output))
        return outputs

    def generate_strings(self, prompts, generation_params={}, disable_tqdm=False):
        prompt_groups = [prompts[k::self.num_gpus] for k in range(self.num_gpus)]
        all_results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit inference tasks to each GPU
            futures = []
            for i, pipeline in enumerate(self.pipelines):
                # Submit inference task
                future = executor.submit(
                    self.single_inference,
                    pipeline, 
                    prompt_groups[i],
                    generation_params,
                    disable_tqdm,
                )
                futures.append(future)
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    print(f"GPU inference task failed: {e}")
        return all_results


