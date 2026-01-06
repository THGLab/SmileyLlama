
import math
import os, sys, shutil
from pathlib import Path
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from smileyllama.score import Score
from smileyllama.utils import safe_run_command


PYSCRIPT = '''
import sys
import multiprocessing as mp
import json
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from easybfe.gbsa import run_gbsa_for_ligand_conformers


def run_task(config):
    try:
        run_gbsa_for_ligand_conformers(**config)
    except Exception as e:
        print(f"Error processing config {config}: {e}")


if __name__ == '__main__':
    config_file = sys.argv[1]
    nprocs = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, mp.cpu_count() - 2)
    with open(config_file, 'r') as f:
        configs = json.load(f)
    
    if nprocs == 1:
        for config in tqdm(configs, desc="EasyBFE GBSA"):
            run_task(config)
    else:
        with mp.Pool(nprocs) as pool:
            results = list(tqdm(pool.imap_unordered(run_task, configs), total=len(configs), desc="EasyBFE GBSA"))
'''

class GBSAEasyBFE(Score):


    def __init__(self, protein: os.PathLike, env: os.PathLike = ''):
        """
        Score class to compute GBSA free energies using EasyBFE.
        """
        super().__init__()
        self.protein = protein
        self.env = Path(env).expanduser().resolve() if env else ''
        assert self.env.is_file(), f"EasyBFE environment file not found: {self.env}"

        self.default_configs = {
            "protein_ff": 'ff14SB', "ligand_ff": 'gaff2',
            'remove_tmp': True, 'igb': 2, 'saltcon': 0.15, 'epsin': 4.0, 'epsout': 80.0,
            "charge_method": 'gas', 'run_em': True, 'em_constraint': True,
        }
            # protein_pdb: os.PathLike,
            # ligand_sdf: os.PathLike,
            # ligand_confs: os.PathLike | List[os.PathLike] | np.ndarray,
            # wdir: os.PathLike = '.',
    
    def _find_output_sdf_dir(self):
        dirname = None
        for score in self.dependency_scores.values():
            tmp = score.wdir / 'output_sdf'
            if tmp.is_dir():
                dirname = tmp
                break
        assert dirname is not None, "No dependency score with output_sdf directory found."
        return dirname

    def compute_batch(self, ligands: list[os.PathLike]):
        """
        Compute GBSA free energies for a batch of ligands using EasyBFE.
        """
        configs = []
        # prepare protein
        protein_path = self.wdir / 'protein.pdb'
        shutil.copyfile(self.protein, protein_path)
        # prepare ligands
        output_sdf_dir = self._find_output_sdf_dir()
        for ligand_sdf in output_sdf_dir.glob("*.sdf"):
            ligand_id = ligand_sdf.stem[:-4] # x_out.sdf -> x
            ligand_dir = self.wdir / str(ligand_id)
            ligand_dir.mkdir(exist_ok=True)
            mol = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)[0]
            ligand_sdf_first = ligand_dir / f'{ligand_id}.sdf'
            with Chem.SDWriter(str(ligand_sdf_first)) as writer:
                writer.write(mol)
            config = {
                "protein_pdb": str(protein_path),
                "ligand_sdf": str(ligand_sdf_first),
                "ligand_confs": str(ligand_sdf),
                "wdir": str(ligand_dir)
            }
            config.update(self.default_configs)
            configs.append(config)
        
        config_file = self.wdir / 'gbsa_configs.json'
        with open(config_file, 'w') as f:
            json.dump(configs, f, indent=4)

        # run command
        gbsa_runner = self.wdir /'run_gbsa.py'
        with open(gbsa_runner, 'w') as f:
            f.write(PYSCRIPT)
        command = f'source {self.env} && echo $(which python) && ' if self.env else ''
        command += f'python {gbsa_runner} {config_file} {self.nprocs}'
        safe_run_command(command, shell=True, capture_output=False)

        # collect results
        scores = []
        for i in range(len(ligands)):
            gbsa_out = self.wdir / f'{i}/gbsa.csv'
            if not gbsa_out.is_file():
                scores.append(math.nan)
            else:
                try:
                    df = pd.read_csv(str(gbsa_out))
                    scores.append(min(df.iloc[:, -1].tolist()))
                except Exception as e:
                    print(f"Error reading GBSA output for ligand {i}: {e}")
                    scores.append(math.nan)
        
        return np.array(scores)
    
    def compute(self, mol):
        return self.compute_batch([mol])


if __name__ == '__main__':
    from smileyllama.score import REGISTRY
    print(list(REGISTRY['score'].keys()))