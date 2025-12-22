import os, sys, shutil
from pathlib import Path
from typing import List, Dict, Any, Literal, Union, Optional
import multiprocessing as mp
import subprocess
import logging

from pydantic import BaseModel, field_validator
import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from .score import REGISTRY, aggregate
from .inference import Pipeline
from .train import make_dpo_data_from_scores
from .utils import modify_yaml
from .plugin import load_module_from_file


class NormalizerConfig(BaseModel):
    name: str = 'Identity'
    parameters: Dict[str, Any] = dict()


class ScoreConfig(BaseModel):
    name: str
    parameters: Dict[str, Any] = dict()
    start_iter: int = 0
    end_iter: int = -1
    normalizer: NormalizerConfig
    weight: float = 1.0
    as_filter: bool = False
    dependencies: List[str] = list()


class RLConfig(BaseModel):
    algorithm: Literal['dpo'] = 'dpo'
    config_file: str
    dpo_num_pairs_per_smiles: int = 8
    dpo_score_margin: float = 0.1
    dpo_use_random_smiles: bool = False
    axo_extra_configs: Dict[str, Any] = dict()
    axo_launcher_configs: Dict[str, Any] = dict()
    

class WorkflowConfig(BaseModel):
    directory: str
    niters: int
    model_path: str
    lora_model_path: str = ''
    nprocs: int = -1
    remove_iter_model: bool = False
    log_level: Literal['info', 'debug'] = 'info'
    plugin: str = ''

    # Sample
    num_samples_per_iter: int = 1000
    system_prompt: str = 'You love and excel at generating SMILES strings of drug-like molecules'
    user_prompt_base: str = 'Output a SMILES string for a drug like molecule'
    user_prompt_properties: List[str] = list()
    prompt_format: Literal['instruct', 'chat'] = 'instruct'

    # Scoring
    scores: Dict[str, ScoreConfig] = dict()

    # RL
    rl_config: RLConfig

    @field_validator("scores")
    @classmethod
    def validate_score_keys(cls, scores):
        bad = ['total', 'smiles']
        for key in scores:
            assert key not in bad, f'score key {key} should not in {bad}'
            assert not key.endswith("_norm"), f"score key {key} should not end with '_norm'"
        return scores


def _tag_done(dir):
    fp = open(os.path.join(dir, 'done.tag'), 'w')
    fp.close()


class Workflow:
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.directory = Path(config.directory)
        self.directory.mkdir(exist_ok=True)

        self.niters = config.niters

        steps = [self.run_sample, self.run_scores, self.run_rl, self.update_iter] * self.niters
        count = 0
        for it in range(self.niters):
            if not self._is_sample_done(it):
                break
            count += 1
            if not self._is_score_done(it):
                break
            count += 1
            if not self._is_rl_done(it):
                break
            count += 1
            count += 1
        
        self.iter = it
        self.steps = steps[count:]
        self.nprocs = mp.cpu_count() - 2 if config.nprocs <= 0 else config.nprocs

        self.prompt_format = self.config.prompt_format
        self.system_prompt = self.config.system_prompt
        self.user_prompt_base = self.config.user_prompt_base
        self.user_prompt_properties = self.config.user_prompt_properties

        with open(self.directory / 'config.json', 'w') as f:
            f.write(self.config.model_dump_json(indent=2))
        
        # Logging
        self.init_logger()

        # Plugin
        if self.config.plugin:
            load_module_from_file(self.config.plugin)
    
    def init_logger(self):
        log_file = str(self.directory / 'workflow.log')
        log_level = logging.DEBUG if self.config.log_level == 'debug' else logging.INFO 
        logger = logging.getLogger('workflow')
        logger.setLevel(log_level) 

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)  
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)    
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.propagate = False
        self.logger = logger
        self.log_file = log_file
            
    def run(self):
        for step in self.steps:
            step()

    def log(self, msg: str):
        print(msg)
    
    def update_iter(self):
        self.iter += 1

    def _is_sample_done(self, it: Optional[int] = None):
        done_tag = self.get_sample_dir(it) / 'done.tag'
        return done_tag.is_file()
    
    def _is_score_done(self, it: Optional[int] = None):
        done_tag = self.get_score_dir(it) / 'done.tag'
        return done_tag.is_file()
    
    def _is_rl_done(self, it: Optional[int] = None):
        done_tag = self.get_rl_dir(it) / 'done.tag'
        return done_tag.is_file()
    
    def get_iter_dir(self, it: Optional[int] = None):
        it = self.iter if it is None else it
        return self.directory / f'iter.{it}'

    def get_sample_dir(self, it: Optional[int] = None):
        return self.get_iter_dir(it) / '01.sample'

    def get_score_dir(self, it: Optional[int] = None):
        return self.get_iter_dir(it) / '02.score'
    
    def get_rl_dir(self, it: Optional[int] = None):
        return self.get_iter_dir(it) / '03.rl'

    def init_scores(self):
        scores = {}
        for tag, score_config in self.config.scores.items():
            start = score_config.start_iter
            end = score_config.end_iter if score_config.end_iter > 0 else self.niters
            if self.iter >= start and self.iter < end:
                norm_config = score_config.normalizer
                norm = REGISTRY['normalizer'][norm_config.name](**norm_config.parameters)
                score = REGISTRY['score'][score_config.name](**score_config.parameters)
                score.set_nprocs(self.nprocs)
                score.set_working_dir(self.get_score_dir() / tag)
                scores[tag] = (score, norm)
        
        # add dependency scores
        for tag, score_config in self.config.scores.items():
            if tag not in scores:
                continue
            for dep in score_config.dependencies:
                if dep in scores:
                    scores[tag][0].add_dependency_score(dep, scores[dep][0])        
                
        return scores
    
    def run_scores(self):
        self.logger.info(f"##### Iter {self.iter} : Score #####")
        stage_dir = self.get_score_dir()
        stage_dir.mkdir(exist_ok=True, parents=True)

        df = pd.read_csv(str(self.get_sample_dir() / 'sample_unique.csv'))
        smiles = df['smiles'].tolist()

        scores = self.init_scores()
        data = {'index': np.arange(len(smiles)), 'smiles': smiles, 'random_smiles': df['random_smiles'].tolist(), 'total': np.zeros(len(smiles))}
        for tag in scores:
            score, norm = scores[tag]
            raw_scores = score.compute_batch(smiles)
            data[tag] = raw_scores
            data[tag+'_norm'] = norm(raw_scores)
        
        # aggregator
        numeric_scores = []
        binary_scores = []
        weights = []
        for tag in scores:
            scfg = self.config.scores[tag]
            num_nan = np.isnan(data[tag]).sum()
            num_tot = len(data[tag])
            if scfg.as_filter:
                score = data[tag+'_norm']
                binary_scores.append(score)
                score_wo_nan = score[~np.isnan(score)].astype(np.bool)
                num_true = np.sum(score_wo_nan == True)
                num_false = np.sum(score_wo_nan == False)
                msg = (
                    f'Score {tag} - true: {num_true} ({num_true/num_tot*100:.2f}%), '
                    f'false: {num_false} ({num_false/num_tot*100:.2f}%), failed: {num_nan} ({num_nan/num_tot*100:.2f}%)'
                )
            else:
                msg = (
                    f'Score {tag} - mean: {np.nanmean(data[tag])}, median: {np.nanmedian(data[tag])}, '
                    f'max: {np.nanmax(data[tag])}, min: {np.nanmin(data[tag])}, failed: {num_nan}'
                )
                numeric_scores.append(data[tag+'_norm'])
                weights.append(scfg.weight)
            self.logger.info(msg)
        
        data['total'] += aggregate(numeric_scores, binary_scores, weights)
        
        df = pd.DataFrame(data)
        df.sort_values('total', inplace=True, ascending=False)
        df.to_csv(str(stage_dir / 'score.csv'), index=None)

        _tag_done(stage_dir)
    
    def run_sample(self):
        self.logger.info(f"##### Iter {self.iter} : Sample #####")
        stage_dir = self.get_sample_dir()
        stage_dir.mkdir(exist_ok=True, parents=True)

        if self.iter == 0:
            model_path = self.config.model_path
            lora_model_path = self.config.lora_model_path
        else:
            model_path = self.get_rl_dir(self.iter-1) / 'lora/merged'
            lora_model_path = ''
        pipeline = Pipeline(
            model_path, 
            lora_model_path=lora_model_path, 
            nprocs=self.nprocs, 
            prompt_format=self.prompt_format,
            system_prompt=self.system_prompt,
            user_prompt_base=self.user_prompt_base,
        )
        response = pipeline.generate(
            self.config.num_samples_per_iter,
            properties=self.user_prompt_properties,
            enforce_valid=False, enforce_unique=True
        )
        self.logger.info(str(response))

        # print smiles
        response.df.to_csv(str(self.get_sample_dir() / 'sample_all.csv'), index=None)
        unqiue_df = response.df.query('unique')
        with open(self.get_sample_dir() / 'smiles.txt', 'w') as f:
            f.write('\n'.join(unqiue_df['smiles'].tolist()))
        unqiue_df.to_csv(str(self.get_sample_dir() / 'sample_unique.csv'), index=None)
        
        # clean up the memory
        self.logger.info("Cleaning up GPU memory...")
        pipeline.free()
        self._print_gpu_mem_info()
        _tag_done(stage_dir)
    
    def make_rl_dataset(self):
        df = pd.read_csv(str(self.get_score_dir() / 'score.csv'))
        dataset = self.get_rl_dir() / 'dataset.jsonl'
        col = 'random_smiles' if self.config.rl_config.dpo_use_random_smiles else 'smiles'
        smiles = df[col].tolist() 
        scores = df['total'].values
        dpo_data = make_dpo_data_from_scores(
            smiles, scores, dataset, 
            self.config.rl_config.dpo_num_pairs_per_smiles,
            self.config.rl_config.dpo_score_margin,
            format=self.prompt_format,
            system_prompt=self.system_prompt,
            user_prompt_base=self.user_prompt_base,
            properties=self.user_prompt_properties
        )
        assert len(dpo_data) > 0, "No DPO dataset are generated, because all the scores are the same"
        return dataset

    def run_rl(self):
        '''Axolotl to run RL'''
        self.logger.info(f"##### Iter {self.iter} : RL #####")
        stage_dir = self.get_rl_dir()
        stage_dir.mkdir(exist_ok=True, parents=True)

        rl_cfg = self.config.rl_config

        config_file = self.get_rl_dir() / 'rl.yml'
        dataset = self.make_rl_dataset()

        if self.iter == 0:
            model_path = self.config.model_path
            lora_model_path = self.config.lora_model_path
        else:
            model_path = self.get_rl_dir(self.iter-1) / 'lora/merged'
            lora_model_path = ''
        spec = {
            "base_model": str(model_path),
            "lora_model_dir": str(lora_model_path),
            "output_dir": str(stage_dir / 'lora'),
            "datasets": [{
                "path": str(dataset),
                "type": {
                    "field_system": "system", "field_prompt": "prompt", "field_chosen": "chosen", "field_rejected": "rejected",
                    "prompt_format":  r"### Instruction:\n{system}\n\n### Input:\n{prompt}\n\n### Response:\n",
                    "chosen_format": "", "rejected_format": ""
                },
                "split": "train"
            }]
        }
        modify_yaml(rl_cfg.config_file, config_file, spec)

        argv = []
        for k, v in rl_cfg.axo_extra_configs.items():
            argv.append(f'--{k} {v}')
        if rl_cfg.axo_launcher_configs:
            argv.append('--')
            for k, v in rl_cfg.axo_launcher_configs.items():
                argv.append(f'--{k} {v}')
        cmd = f'axolotl train {config_file} {" ".join(argv)}'

        logfile = stage_dir / 'rl.log'
        self.logger.info(f"Running Axolotl for RL. Log file: {logfile}")
        with open(logfile, 'w') as f:
            # run RL
            subprocess.run(cmd, shell=True, stdout=f, stderr=f, check=True)
            # Merge lora model
            subprocess.run(f'axolotl merge-lora {config_file} --lora-model-dir={stage_dir/"lora"}', shell=True, stdout=f, stderr=f, check=True)
        self.logger.info("RL finished!")
        
        if self.config.remove_iter_model and self.iter > 0:
            self.logger.info("Removing merged model of previous iteration.")
            shutil.rmtree(model_path)

        _tag_done(stage_dir)

    def _print_gpu_mem_info(self):
        if not torch.cuda.is_available():
            return
        device_count = torch.cuda.device_count()
        self.logger.debug(f"Found {device_count} GPUs:")
        for i in range(device_count):
            free, total = torch.cuda.mem_get_info(i)
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            self.logger.debug(f"--- GPU {i} ---")
            self.logger.debug(f"  Free: {free / 1024**2:.2f} MB / {total / 1024**2:.2f} MB")
            self.logger.debug(f"  PyTorch allocated: {allocated / 1024**2:.2f} MB")
            self.logger.debug(f"  PyTorch reserved: {reserved / 1024**2:.2f} MB")


