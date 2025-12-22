"""
Author: Eric Wang
Date: 12/16/2025

This package contains useful functions to manipulate command line executions.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import warnings
from typing import List, Union, Optional, Tuple
import contextlib


class CommandExecuteError(Exception):
    """
    Exception for command line exec error
    """
    def __init__(self, msg: str):
        self.msg = msg
    
    def __str__(self):
        return self.msg
    
    def __repr__(self):
        return self.msg


def safe_run_command(cmd, check=True, text=True, capture_output=True, **kwargs):
    if check:
        try:
            res = subprocess.run(cmd, check=check, text=text, capture_output=capture_output, **kwargs)
        except subprocess.CalledProcessError as e:
            msg = (
                f'Command {cmd} returned non-zero exit status {e.returncode}:\n\n'
                f'  --stdout:\n{e.stdout}\n\n'
                f'  --stderr:\n{e.stderr}\n'
            )
            raise CommandExecuteError(msg)
    else:
        return subprocess.run(cmd, check=check, text=text, capture_output=capture_output, **kwargs)


@contextlib.contextmanager
def set_directory(dirname: os.PathLike, mkdir: bool = False):
    """
    Set current workding directory within context
    
    Parameters
    ----------
    dirname : os.PathLike
        The directory path to change to
    mkdir: bool
        Whether make directory if `dirname` does not exist
    
    Yields
    ------
    path: Path
        The absolute path of the changed working directory
    
    Examples
    --------
    >>> with set_directory("some_path"):
    ...    do_something()
    """
    pwd = os.getcwd()
    path = Path(dirname).resolve()
    if mkdir:
        path.mkdir(exist_ok=True, parents=True)
    os.chdir(path)
    yield path
    os.chdir(pwd)