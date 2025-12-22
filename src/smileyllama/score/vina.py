__all__ = ['Vina', 'UniDock', 'VinaGPU']

import os, shutil
from typing import Tuple, Union, Optional, List, Literal, Dict, Any
from pathlib import Path
import math
import warnings
import multiprocessing as mp
import subprocess

import numpy as np
from tqdm import tqdm
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate

from .base import Score, safe_mol_from_smiles
from ..cmd import safe_run_command


class Vina(Score):

    DEFAULT_DOCKING_SETTINGS = {
        "num_modes": 5,
        "min_rmsd": 0.5,
        "energy_range": 5,
        "exhaustiveness": 8
    }

    def __init__(
        self,
        protein: os.PathLike,
        box_center: Tuple[float, float, float],
        box_size: Tuple[float, float, float],
        exec_path: str = 'vina',
        *,
        wdir: Optional[os.PathLike] = None,
        protein_prep_exec: str = 'prepare_receptor',
        post_process: bool = True,
        optimize_ligand_during_prep: bool = True,
        thresh_for_fail: float = -20.0,
        extra_docking_settings: Dict[str, Any] = dict(),
        
    ):
        super().__init__()
        assert os.path.isfile(protein), f'{protein} not exist'
        self.protein_input = Path(protein)
        self.exec = exec_path
        self.protein_prep_exec = protein_prep_exec
        self.config = {
            'center_x': box_center[0],
            'center_y': box_center[1],
            'center_z': box_center[2],
            'size_x': box_size[0], 
            'size_y': box_size[1], 
            'size_z': box_size[2],
        }
        self.config.update(self.DEFAULT_DOCKING_SETTINGS)
        self.config.update(extra_docking_settings)
        self.post_process = post_process
        self.optimize_ligand_during_prep = optimize_ligand_during_prep
        self.thresh_for_fail = thresh_for_fail
        if wdir is not None:
            self.set_working_dir(wdir)

    def prepare_protein(self):
        if self.protein_input.suffix == '.pdbqt':
            self.protein_pdbqt = self.protein_input.resolve()
        elif self.protein_input.suffix == '.pdb':
            self.protein_pdbqt = self.wdir / f'{self.protein_input.stem}.pdbqt'
            self.convert_protein_pdb_to_pdbqt(
                self.protein_input, 
                self.protein_pdbqt,
                self.protein_prep_exec
            )
        else:
            raise RuntimeError(f"Invalid protein file format {suffix}")
    
    def prepare_config(self):
        self.config_path = self.wdir / 'config.txt'
        self.write_config(self.config, self.config_path)

    @classmethod
    def convert_protein_pdb_to_pdbqt(
        cls,
        protein_pdb: os.PathLike, 
        output_path: os.PathLike, 
        binary: str = 'prepare_receptor'
    ):
        if os.path.basename(binary) == 'prepare_receptor':
            safe_run_command([binary, '-r', protein_pdb, '-o', output_path, '-A', 'checkhydrogens'])
        elif os.path.basename(binary) == 'obabel':
            safe_run_command([obabel, '-ipdb', protein_pdb, '-opdbqt', '-O', output_path, '-xr'])
        else:
            raise NotImplementedError(f'Unsupported tool: {tool}')
    
    @classmethod
    def convert_rdkit_to_pdbqt(cls, mol: Chem.Mol, out: os.PathLike = '', tool: str = 'meeko') -> str:
        if tool == 'meeko':
            pdbqt = PDBQTWriterLegacy.write_string(MoleculePreparation(rigid_macrocycles=True)(mol)[0])[0]
            if sum([line.startswith("ATOM") for line in pdbqt.split('\n')]) > 140:
                return False
            if sum([line.startswith("BRANCH") for line in pdbqt.split('\n')]) > 40:
                return False
            if pdbqt.strip() == '':
                return False
            if out:
                with open(out, 'w') as f:
                    f.write(pdbqt)
        else:
            raise NotImplementedError(f"Not supported rdkit->pdbqt tool: {tool}")
        return True
    
    @classmethod
    def convert_pdbqt_to_sdf(cls, pdbqt_path: os.PathLike, sdf_path: os.PathLike):
        pdbqt_mol = PDBQTMolecule.from_file(str(pdbqt_path), skip_typing=True)
        sdstring, _ = RDKitMolCreate.write_sd_string(pdbqt_mol)
        with open(sdf_path, 'w') as f:
            f.write(sdstring)
        return True

    @classmethod
    def write_config(cls, config: dict, path: os.PathLike):
        with open(path, 'w') as f:
            for k, v in config.items():
                f.write(f'{k} = {v}\n')

    def prepare_ligand_single(self, args: Tuple[int, str]):
        index, smi = args
        mol = safe_mol_from_smiles(smi)
        if mol is None:
            return False
        if any([at.GetSymbol() not in ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'] for at in mol.GetAtoms()]):
            return False
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            if self.optimize_ligand_during_prep:
                AllChem.MMFFOptimizeMolecule(mol)
            prep = self.convert_rdkit_to_pdbqt(mol, self.ligand_input_dir/f'{index}.pdbqt')
        except Exception as e:
            return False
        return prep
    
    def prepare_ligands(self, smiles: List[str]):
        # ligand directory
        self.ligand_input_dir = self.wdir / 'input'
        self.ligand_input_dir.mkdir(exist_ok=True)
        self.ligand_output_pdbqt_dir = self.wdir / 'output_pdbqt'
        self.ligand_output_pdbqt_dir.mkdir(exist_ok=True)

        self.ligand_output_sdf_dir = self.wdir / 'output_sdf'
        self.ligand_output_sdf_dir.mkdir(exist_ok=True)

        desc = f'{self.__class__.__name__} preparing Ligands'
        count = len(smiles)
        if self.nprocs == 1:
            for i, smi in tqdm(enumerate(smiles), total=count, desc=desc):
                self.prepare_ligand_single((i, smi))
        else:
            args = [(i, smi) for i, smi in enumerate(smiles)]
            with mp.Pool(self.nprocs) as p:
                results = [r for r in tqdm(p.imap_unordered(self.prepare_ligand_single, args), total=count, desc=desc)]
    
    def get_dock_command(self):
        command = (
            f'{self.exec} --config {self.config_path} '
            f'--batch {self.ligand_input_dir}/*.pdbqt --receptor {self.protein_pdbqt} '
            f'--dir {self.ligand_output_pdbqt_dir}'
        )
        return command

    def run_dock(self):
        with open(self.wdir / f'{self.__class__.__name__}.log', 'w') as f:
            safe_run_command(self.get_dock_command(), shell=True, capture_output=False, stdout=f, stderr=f)

    def parse_results(self, smiles: List[str]):
        scores = []
        
        for i in range(len(smiles)):
            out_pdbqt = self.ligand_output_pdbqt_dir / f'{i}_out.pdbqt'
            if not out_pdbqt.is_file():
                scores.append(math.nan)
                continue
            if self.post_process:
                out_sdf = self.ligand_output_sdf_dir / f'{i}_out.sdf'
                self.convert_pdbqt_to_sdf(out_pdbqt, out_sdf)

            results = []
            with open(out_pdbqt) as f:
                for line in f:
                    if line.startswith("REMARK VINA RESULT"):
                        results.append(float(line.split()[-3]))
            
            results.sort()
            score = results[0] if len(results) > 0 else math.nan
            score = math.nan if score < self.thresh_for_fail else score
            scores.append(score)
        
        return np.array(scores)
    
    def compute_batch(self, smiles: List[str]):
        self.prepare_config()
        self.prepare_protein()
        self.prepare_ligands(smiles)
        self.run_dock()
        return self.parse_results(smiles)
    
    def compute(self, smiles):
        return self.compute_batch([smiles])


class VinaGPU(Vina):

    DEFAULT_DOCKING_SETTINGS = {
        "num_modes": 5,
        "energy_range": 5,
        "thread": 8000
    }

    def __init__(
        self,
        protein: os.PathLike,
        box_center: Tuple[float, float, float],
        box_size: Tuple[float, float, float],
        exec_path: str = 'vina_gpu',
        **kwargs
    ):
        extra_docking_settings = kwargs.pop("extra_docking_settings", {})  
        opencl_binary_path = extra_docking_settings.pop(
            'opencl_binary_path',
            Path(shutil.which(exec_path)).resolve().parent
        )
        _bin1_path = os.path.join(opencl_binary_path, 'Kernel1_Opt.bin')
        _bin2_path = os.path.join(opencl_binary_path, 'Kernel2_Opt.bin')
        if not ( os.path.isfile(_bin1_path) and os.path.isfile(_bin2_path) ):
            raise RuntimeError(f"Invalid OpenCL path: {opencl_binary_path}, both Kernel1_Opt.bin and Kernel2_Opt.bin has to exist")
        self.opencl_binary_path = Path(opencl_binary_path).resolve()

        super().__init__(protein, box_center, box_size, exec_path=exec_path, **kwargs)
    
    def get_dock_command(self):
        ligand_files = self.wdir / "ligands.txt"
        with open(ligand_files, 'w') as f:
            f.write(" ".join([str(x) for x in self.ligand_input_dir.glob("*.pdbqt")]))
        command = (
            f'{self.exec} --config {self.config_path} '
            f'--ligand_directory {self.ligand_input_dir} --receptor {self.protein_pdbqt} '
            f'--output_directory {self.ligand_output_pdbqt_dir} '
            f'--opencl_binary_path {self.opencl_binary_path}'
        )
        return command


class UniDock(Vina):
    def __init__(
        self,
        protein: os.PathLike,
        box_center: Tuple[float, float, float],
        box_size: Tuple[float, float, float],
        exec_path: str = 'unidock',
        **kwargs
    ):

        extra_docking_settings = kwargs.pop("extra_docking_settings", {})
        search_mode = extra_docking_settings.get("search_mode", 'fast')
        # handle the search mode
        if search_mode == 'fast':
            extra_docking_settings.update({'exhaustiveness': 128, 'max_step': 20})
        elif search_mode == 'balance':
            extra_docking_settings.update({'exhaustiveness': 384, 'max_step': 40})
        elif search_mode == 'detail': 
            extra_docking_settings.update({'exhaustiveness': 512, 'max_step': 40})

        super().__init__(
            protein, box_center, box_size, 
            exec_path=exec_path, extra_docking_settings=extra_docking_settings,
            **kwargs
        )
    
    def get_dock_command(self):
        ligand_files = self.wdir / "ligands.txt"
        with open(ligand_files, 'w') as f:
            f.write(" ".join([str(x) for x in self.ligand_input_dir.glob("*.pdbqt")]))
        command = (
            f'{self.exec} --config {self.config_path} '
            f'--ligand_index {ligand_files} --receptor {self.protein_pdbqt} '
            f'--dir {self.ligand_output_pdbqt_dir}'
        )
        return command
