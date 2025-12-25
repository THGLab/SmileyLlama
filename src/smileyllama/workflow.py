import os, sys, shutil
from pathlib import Path
from typing import List, Dict, Any, Literal, Union, Optional
import multiprocessing as mp
import subprocess
import logging

from pydantic import BaseModel, field_validator, Field
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
    """Configuration for a normalizer."""
    name: str = Field(default='Identity', description="Name of the normalizer class (must be registered in the normalizer registry).")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters to pass to the normalizer constructor.")


class ScoreConfig(BaseModel):
    """Configuration for a score in the workflow."""
    name: str = Field(description="Name of the score class (must be registered in the score registry).")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters to pass to the score constructor.")
    start_iter: int = Field(default=0, description="First iteration where this score is active (0-indexed).")
    end_iter: int = Field(default=-1, description="Last iteration where this score is active. If -1, active until end.")
    normalizer: NormalizerConfig = Field(description="Normalizer configuration to apply to raw scores.")
    weight: float = Field(default=1.0, description="Weight for score aggregation.")
    as_filter: bool = Field(default=False, description="If True, treat as binary filter (0/1) rather than numeric score. The binary filters will pass its score as binary_scores in aggregate(), which will zero-out all the other scores if it is False.")
    dependencies: List[str] = Field(default_factory=list, description="List of score names that this score depends on.")


class RLConfig(BaseModel):
    """Configuration for reinforcement learning (RL) training."""
    algorithm: Literal['dpo'] = Field(default='dpo', description="RL algorithm to use. Currently only 'dpo' (Direct Preference Optimization) is supported.")
    config_file: str = Field(description="Path to base Axolotl configuration file.")
    dpo_num_pairs_per_smiles: int = Field(default=8, description="Number of (chosen, rejected) pairs to generate per SMILES string.")
    dpo_score_margin: float = Field(default=0.1, description="Minimum score difference required to create a pair.")
    dpo_use_random_smiles: bool = Field(default=False, description="If True, use random SMILES representations instead of the model direct outputs.")
    axo_extra_configs: Dict[str, Any] = Field(default_factory=dict, description="Additional Axolotl configuration parameters.")
    axo_launcher_configs: Dict[str, Any] = Field(default_factory=dict, description="Axolotl launcher configuration parameters.")
    

class WorkflowConfig(BaseModel):
    """Configuration for the SmileyLlama workflow.
    
    Defines all parameters for running the iterative RL workflow including
    sampling, scoring, and training steps.
    """
    directory: str = Field(description="Base directory for workflow output files.")
    niters: int = Field(description="Number of iterations to run.")
    model_path: str = Field(description="Base language model (path or HuggingFace format).")
    lora_model_path: str = Field(default='', description="Path to LoRA adapter weights for initial iteration.")
    nprocs: int = Field(default=-1, description="Number of processes for parallel computation. If -1, ``uses multiprocessing.cpu_count() - 2``.")
    remove_iter_model: bool = Field(default=False, description="If True, remove merged models from previous iterations to save space.")
    log_level: Literal['info', 'debug'] = Field(default='info', description="Logging level.")
    plugin: str = Field(default='', description="Path to Python plugin file to load.")

    # Sample
    num_samples_per_iter: int = Field(default=1000, description="Number of SMILES strings to generate per iteration.")
    system_prompt: str = Field(default='You love and excel at generating SMILES strings of drug-like molecules', description="System prompt for generation.")
    user_prompt_base: str = Field(default='Output a SMILES string for a drug like molecule', description="Base user prompt.")
    user_prompt_properties: List[str] = Field(default_factory=list, description="List of property strings to include in prompts.")
    prompt_format: Literal['instruct', 'chat'] = Field(default='instruct', description="Prompt format for generation.")

    # Scoring
    scores: Dict[str, ScoreConfig] = Field(default_factory=dict, description="Dictionary mapping score names to their configurations.")

    # RL
    rl_config: RLConfig = Field(description="Reinforcement learning configuration.")

    @field_validator("scores")
    @classmethod
    def validate_score_keys(cls, scores):
        bad = ['total', 'smiles']
        for key in scores:
            assert key not in bad, f'score key {key} should not in {bad}'
            assert not key.endswith("_norm"), f"score key {key} should not end with '_norm'"
        return scores


def _tag_done(dir):
    """Create a ``done.tag`` file to mark a workflow stage as complete.
    
    Parameters
    ----------
    dir : str or os.PathLike
        Directory where the done.tag file should be created.
    """
    fp = open(os.path.join(dir, 'done.tag'), 'w')
    fp.close()


class Workflow:
    """Main workflow class for iterative RL-based molecular generation.
    
    Executes an iterative pipeline of: (1) sample molecules, (2) score them,
    (3) train RL model. Handles resumption from checkpoints and manages all intermediate files and directories.
    """
    def __init__(self, config: WorkflowConfig):
        """Initialize the workflow.
        
        Sets up directories, logging, and determines which steps to run
        based on existing checkpoint files.
        
        Parameters
        ----------
        config : WorkflowConfig
            Workflow configuration object.
        """
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
        """Initialize logging for the workflow.
        
        Sets up both console and file logging with appropriate formatters.
        Log file is written to ``workflow.log`` in the workflow directory.
        """
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
        """Run the workflow.
        
        Executes all remaining workflow steps in sequence. Steps are
        determined at initialization based on checkpoint status.
        """
        for step in self.steps:
            step()

    def log(self, msg: str):
        """Log a message.
        
        Parameters
        ----------
        msg : str
            Message to log.
        """
        print(msg)
    
    def update_iter(self):
        """Increment the current iteration counter."""
        self.iter += 1

    def _is_sample_done(self, it: Optional[int] = None):
        """Check if sampling stage is complete for an iteration.
        
        Parameters
        ----------
        it : int, optional
            Iteration number. If None, uses current iteration.
        
        Returns
        -------
        bool
            True if done.tag exists in sample directory.
        """
        done_tag = self.get_sample_dir(it) / 'done.tag'
        return done_tag.is_file()
    
    def _is_score_done(self, it: Optional[int] = None):
        """Check if scoring stage is complete for an iteration.
        
        Parameters
        ----------
        it : int, optional
            Iteration number. If None, uses current iteration.
        
        Returns
        -------
        bool
            True if done.tag exists in score directory.
        """
        done_tag = self.get_score_dir(it) / 'done.tag'
        return done_tag.is_file()
    
    def _is_rl_done(self, it: Optional[int] = None):
        """Check if RL training stage is complete for an iteration.
        
        Parameters
        ----------
        it : int, optional
            Iteration number. If None, uses current iteration.
        
        Returns
        -------
        bool
            True if done.tag exists in RL directory.
        """
        done_tag = self.get_rl_dir(it) / 'done.tag'
        return done_tag.is_file()
    
    def get_iter_dir(self, it: Optional[int] = None):
        """Get directory path for an iteration.
        
        Parameters
        ----------
        it : int, optional
            Iteration number. If None, uses current iteration.
        
        Returns
        -------
        Path
            Path to iteration directory (e.g., "iter.0").
        """
        it = self.iter if it is None else it
        return self.directory / f'iter.{it}'

    def get_sample_dir(self, it: Optional[int] = None):
        """Get directory path for sampling stage.
        
        Parameters
        ----------
        it : int, optional
            Iteration number. If None, uses current iteration.
        
        Returns
        -------
        Path
            Path to sample directory (e.g., "iter.0/01.sample").
        """
        return self.get_iter_dir(it) / '01.sample'

    def get_score_dir(self, it: Optional[int] = None):
        """Get directory path for scoring stage.
        
        Parameters
        ----------
        it : int, optional
            Iteration number. If None, uses current iteration.
        
        Returns
        -------
        Path
            Path to score directory (e.g., "iter.0/02.score").
        """
        return self.get_iter_dir(it) / '02.score'
    
    def get_rl_dir(self, it: Optional[int] = None):
        """Get directory path for RL training stage.
        
        Parameters
        ----------
        it : int, optional
            Iteration number. If None, uses current iteration.
        
        Returns
        -------
        Path
            Path to RL directory (e.g., "iter.0/03.rl").
        """
        return self.get_iter_dir(it) / '03.rl'

    def init_scores(self):
        """Initialize score instances for the current iteration.
        
        Creates score and normalizer instances based on configuration,
        sets up working directories and dependencies, and filters scores
        based on start_iter and end_iter settings.
        
        Returns
        -------
        dict of str to tuple of (Score, Normalizer)
            Dictionary mapping score names to (score, normalizer) tuples.
        """
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
        """Run the scoring stage for the current iteration.
        
        Loads unique SMILES from sampling stage, computes all active scores,
        applies normalizers, aggregates scores, and saves results to CSV.
        Logs statistics for each score.
        """
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
        """Run the sampling stage for the current iteration.
        
        Generates SMILES strings using the language model pipeline,
        saves all results and unique results to CSV files, and writes
        unique SMILES to a text file. Uses model from previous iteration
        if available, otherwise uses base model.
        """
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
        """Create DPO training dataset from scored SMILES.
        
        Reads scores from scoring stage, pairs SMILES based on score
        differences, and generates DPO training data using
        :func:`~smileyllama.train.make_dpo_data_from_scores`.
        
        Returns
        -------
        Path
            Path to the generated dataset.jsonl file.
        
        Raises
        ------
        AssertionError
            If no DPO pairs can be generated (all scores are identical).
        """
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
        """Run reinforcement learning training stage.
        
        Creates DPO dataset, configures Axolotl training, runs RL training,
        and merges LoRA weights.
        """
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
        """Print GPU memory information for debugging.
        
        Logs free, allocated, and reserved memory for all available GPUs.
        Only logs if CUDA is available and log level is DEBUG.
        """
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


