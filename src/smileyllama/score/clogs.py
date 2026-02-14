__all__ = ['CLogS']

import os, math, subprocess
from typing import Optional, List
from pathlib import Path

import numpy as np

from .base import Score
from ..utils import safe_run_command


class CLogS(Score):
    """Score for aqueous solubility (cLogS) using the clogs_alteri executable.

    Computes cLogS values for a list of SMILES by writing them to a .smi file,
    invoking the clogs_alteri tool, and parsing the tab-separated results
    from stdout.
    """

    def __init__(
        self,
        exec_path: str = '/global/scratch/users/gbalteri/CACHE/calc_cLogS/clogs_alteri',
        *,
        wdir: Optional[os.PathLike] = None,
    ):
        """Initialize CLogS scorer.

        Parameters
        ----------
        exec_path : str, optional
            Path to the clogs_alteri executable.
        wdir : os.PathLike, optional
            Working directory for intermediate files. If None, must be set later.
        """
        super().__init__()
        assert os.path.isfile(exec_path), f'{exec_path} not found'
        self.exec = exec_path
        if wdir is not None:
            self.set_working_dir(wdir)

    def write_smiles_file(self, smiles: List[str], path: os.PathLike):
        """Write SMILES strings to a .smi file (one per line).

        Parameters
        ----------
        smiles : list of str
            SMILES strings.
        path : os.PathLike
            Output file path.
        """
        with open(path, 'w') as f:
            for smi in smiles:
                f.write(f'{smi}\n')

    def parse_results(self, stdout: str, n: int) -> np.ndarray:
        """Parse cLogS results from stdout of clogs_alteri.

        The stdout has a header line ``SMILES\\tcLogS`` followed by
        tab-separated SMILES and score lines. Lines with ``ERROR``
        are treated as failures (NaN).

        Parameters
        ----------
        stdout : str
            Captured stdout from the clogs_alteri executable.
        n : int
            Expected number of results.

        Returns
        -------
        numpy.ndarray
            Array of cLogS scores. NaN for any lines that failed to parse.
        """
        scores = [math.nan] * n
        lines = stdout.strip().split('\n')
        # first line is the header "SMILES\tcLogS"
        for i, line in enumerate(lines[1:]):
            if i >= n:
                break
            parts = line.strip().split('\t')
            if len(parts) >= 2 and 'ERROR' not in parts[1]:
                try:
                    scores[i] = float(parts[1])
                except ValueError:
                    pass
        return np.array(scores)

    def compute_batch(self, smiles: List[str]) -> np.ndarray:
        """Compute cLogS scores for a batch of molecules.

        Parameters
        ----------
        smiles : list of str
            List of SMILES strings.

        Returns
        -------
        numpy.ndarray
            Array of cLogS scores.
        """
        smi_path = self.wdir / 'input.smi'
        self.write_smiles_file(smiles, smi_path)
        result = subprocess.run(
            [self.exec, str(smi_path)],
            capture_output=True, text=True
        )
        return self.parse_results(result.stdout, len(smiles))

    def compute(self, smiles: str) -> float:
        """Compute cLogS score for a single molecule.

        Parameters
        ----------
        smiles : str
            SMILES string.

        Returns
        -------
        float
            cLogS value.
        """
        return self.compute_batch([smiles])[0].item()
