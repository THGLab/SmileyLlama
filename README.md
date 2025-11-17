
# SmileyLlama
This repository contains code and data used by the [SmileyLlama](https://arxiv.org/abs/2409.02231) project to train SmileyLlama and its variants, and to produce results used in the paper. The SmileyLlama model is not hosted here; rather, it's hosted on [huggingface](https://huggingface.co/THGLab/Llama-3.1-8B-SmileyLlama-1.1), along with variants trained for [adhering to properties specified in a prompt](https://huggingface.co/THGLab/Llama-3.1-8B-SmileyLlama-1.1-Prompt-Following) and for [generating binders to SARS-CoV-2 Main Protease (MPro)](https://huggingface.co/THGLab/Llama-3.1-8B-SmileyLlama-1.1-Mpro). 

For those who want a gentle, yet hands-on introduction to SmileyLlama, download the Demo.ipynb jupyter notebook, which provides a demonstration of SmileyLlama's abilities and a brief tutorial on writing prompts for it and related models.


# System Requirements

Supervised fine-tuning and DPO of SmileyLlama is very memory-intensive due to the number of parameters. To replicate the work in this study, a 4xGPU node with 48GB VRAM per GPU is recommended. Lower VRAM will result in an out of memory error. For smaller setups, the `gradient_accumulation_steps` setting in the relevant axolotl configuration files (`sft/8b-lora32/cf_lora.yml`, `prompt_following/dpo-instr/cf_dpo_lora.yml`, `prompt_following/dpo-instr/cf_dpo_lora.yml`) should be adjusted such that the overall batch size remains unchanged. In axolotl, the total batch size is the product of the micro batch size, gradient accumulation steps, and number of GPUs. This was tested using Nvidia A40 GPUs.

Inference on SmileyLlama should not be done on a GPU with less than 16 GB VRAM. Inference can be done using the CPU, but this will be slow.

 Tested on `python 3.10.12` (Python 3.10 can be found at [python's website](https://www.python.org/downloads/release/python-31011/) and version management can be done with [pyenv](https://github.com/pyenv/pyenv)), [`gcc 11.4.0`](https://ftp.gnu.org/gnu/gcc/gcc-11.4.0/) and [`cuda 11.8.0`](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux). Runs on Linux, was tested on Rocky Linux 8.10 (Green Obsidian).


# Installation Guide
A few environments are required to be able to replicate the work in SmileyLlama, including finetuning the models.


### axo (use for fine-tuning)
Make sure to have `python 3.10.12`, `gcc 11.4.0` and `cuda 11.8.0` or compatible versions
```
cd envs
python -m venv axo
source axo/bin/activate
pip install packaging wheel
pip install torch==2.3.1
pip install -r axo-requirements.txt
```

### ana-env (main environment for analysis)
Make sure to have `python 3.10.12`, `gcc 11.4.0` and `cuda 11.8.0` or compatible versions
```
cd envs
python -m venv ana-env
source ana-env/bin/activate
pip install packaging wheel
pip install torch==2.3.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r ana-env-requirements.txt
cd ../scripts
pip install -e .
```

Also, remember to create kernels for use in jupyter notebooks.
#### mol-benchmark (for guacamol analysis)
Follow steps on https://github.com/BenevolentAI/guacamol

Installing these will take somewhere on the order of 10 minutes on a "normal" desktop computer if all goes well. It can take much longer if flash attention is compiled (on the order of hours) instead of loaded from a prebuilt binary.
# Demo

You can use the `ana-env` to run this demo, or anything with `torch`, `transformers`, and `rdkit`. The demo folder contains a jupyter notebook will take you through how to download and use SmileyLlama to generate molecules with some features. SmileyLlama's weights are about 16GB, so the time it takes to download them will vary based on your internet speed. Outside of this, a "normal" desktop will probably take on the order of 5 minutes to run the demo. The outputs of the notebook are already shown, although some part requires randomness.
# Instructions for Use

To download and use SmileyLlama or its derivative models, you can visit this [link](https://huggingface.co/collections/THGLab/smileyllama-6880b14e4e4d708001564062) . All scripts and jupyter notebooks in this codebase reference either these models or the Llama models by their huggingface identifiers (e.g. "THGLab/Llama-3.1-8B-SmileyLlama-1.1").

We've included code and data required to regenerate the figures in this paper. However, some of the scripts in `mpro` require the iMiner library to run, which is not yet released. Still, the derivative model of SmileyLlama produced by optimizing with iMiner is available at  `https://huggingface.co/THGLab/Llama-3.1-8B-SmileyLlama-1.1-Mpro`. 

The code and data required to generate SMILES strings and run the guacamol benchmark on Llama-3.1-Instruct is in `llama_k_shot`, code and data used to compare SmileyLlama and Llama is found in mmlu (need to install [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to run the benchmark). The lm-evaluation-harness can also be used to calculate perplexity on wikitest using
```
lm_eval --model=hf --model_args="pretrained=/path/to/model" --tasks=wikitext
```

### sft
To get code and data used in SFT of SmileyLlama and early analysis of SmileyLlama, decompress the sft_archive with 
```
7z x sft_archive/sft_part.7z.001
```

`sft/8b-lora32` contains the config file used for axolotl to fine-tune. To restart fine-tuning, modify the paths in this file to specify where the data resides, and where the outputs and prepared data should be saved. Then, preprocess, begin fine-tuning, and merge the LoRA into the weights with
```
CUDA_VISIBLE_DEVICES="" python3 -m axolotl.cli.preprocess cf_lora.yml
srun accelerate launch -m axolotl.cli.train cf_lora.yml
python3 -m axolotl.cli.merge_lora $(pwd)/cf_lora.yml --lora_model_dir="$(pwd)/outputs"
```

### prompt_following
To get code and data for analyzing the ability of SmileyLlama to follow instructions in the prompt, and how DPO can be used to improve this, decompress `prompt-following.7z` with `7z x prompt_following.7z` 

Similarly to the previous section, to restart DPO, simply modify `prompt_following/dpo-instr/cf_dpo_lora.yml` to have the relevant paths in your system and run
```
srun accelerate launch --use-deepspeed -m axolotl.cli.train cf_dpo_lora.yml --dataset_processes=1
python3 -m axolotl.cli.merge_lora $(pwd)/cf_dpo_lora.yml --lora_model_dir="$(pwd)/outputs"
```
The data required for this can be found in `prompt_following/dpo-instr/dpodataset/dpodataset.jsonl`.

### mpro
This directory (when decompressed with `7z x mpro.7z`) contains all relevant parts of the project used to optimize SmileyLlama for inhibition of SARS-CoV-2 Main Protease (MPro). Analysis for a few sample ligands generated by SmileyLlama after this optimization (the model which generated them can be found on huggingface as `THGLab/Llama-3.1-8B-SmileyLlama-1.1-Mpro`)  can be found in `mpro/ligand_analysis` . files for DPO and outputs of the model throughout the training process can be found in `mpro/run/`. Outputs from iMiner used for comparison in our paper can be found in `mpro/iminer_ref_details`. The Jupyter notebooks used to generate figures relating to optimization for MPro inhibition can be found at `mpro/MproFigures.ipynb` and `mpro/cleaner_inference.ipynb`. 

### To reproduce key results
To reproduce the Llama 0-shot and 20-shot values in Table 1, use the llama_k_shot/guacamol_analysis.py notebook. These analyze pre-generated molecules which were produced with the gen_0_shot and gen_20_shot scripts.
To reproduce the SmileyLlama values in Table 1 and Figure S1, use the sft/guacamol_analysis.ipynb jupyter notebook.
To reproduce the visualizations of properties in Figure 2, use the sft/distribution_vis.ipynb notebook
To reproduce the SFT and DPO results in Table 2 and Figure 3b, use the prompt_following/prompt_following_analysis.ipynb notebook.
To reproduce Figure 3a, use the prompt_following/figure3a.ipynb notebook.
To reproduce Figure 4, use the mpro/MproFigures.ipynb and mpro/cleaner_inference.ipynb notebooks.
To visualize the interactions between selected generations and Mpro as in Figure 5, use the results from the [Protein-Ligand Interaction Profiler (PLIP)](https://github.com/pharmai/plip) in mpro/ligand_analysis/plip 