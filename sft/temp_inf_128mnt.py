'''
This runs the guacamol suite of tests on some SmileyLlama-like LLM.
In particular, it runs inference, creating files of 200,000 SMILES strings in some directory
Then, it analyzes each file using the guacamol analysis in guacamol_minimal
'''
import sys
from sltools import sl_inference
import os
from accelerate import Accelerator
from tqdm import tqdm
import argparse

def inference_and_analysis(llm_inference_obj, smilesfile, generation_params):
    prompts = [llm_inference_obj.create_prompt()]*1600
    prompts = ["### Instruction:\nYou love and excel at generating SMILES strings of drug-like molecules\n\n### Input:\nOutput a SMILES string for a drug like molecule:\n\n### Response: \n"]*1600
    completions = []
    generation = llm_inference_obj.generate_smiles_strings(prompts, 128, 128, generation_params)
    if generation is not None:
        completions = generation
        print(f"Generated {len(completions)} completions")
        #Truncate completions to 200,000
        assert len(completions) >= 200000
        completions = completions[:200000]
        #Write to smilesfile
        with open(smilesfile, 'w+') as f:
            for c in completions:
                f.write(c.strip() + '\n')
        #Perform guacomol analysis
    return None

if __name__=="__main__":
    accelerator = Accelerator()
    parser = argparse.ArgumentParser(description="Run molecule generation experiments")
    parser.add_argument("--project_dir", type=str, required=True, help="Directory of the project")
    parser.add_argument("--smiles_dir", type=str, required=False, help="Directory of the smiles files")
    args = parser.parse_args()
    #Make a directory for the smiles files if it doesn't exist
    if args.smiles_dir is None:
        smilesfile_dir = args.project_dir + '/samples'
    else:
        smilesfile_dir = args.smiles_dir
    if not os.path.exists(smilesfile_dir) and accelerator.is_main_process:
        os.makedirs(smilesfile_dir)
    print(smilesfile_dir)
    #Create the inference object and load the model
    sl_path = args.project_dir + '/outputs/merged/'
    sl = sl_inference.InferenceObject(sl_path, 'THGLab/Llama-3.1-8B-SmileyLlama-1.1')
    temperatures = [0.1*n for n in range(5, 15)]
    for t in temperatures:
        smilesfile = smilesfile_dir + '/temp' + str(round(t,2)) + '.txt'
        generation_params = {"temperature":t}
        inference_and_analysis(sl, smilesfile, generation_params)
        accelerator.wait_for_everyone()

