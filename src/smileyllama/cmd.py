"""
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
    """Exception raised when a command execution fails.
    
    Raised by :func:`safe_run_command` when a subprocess command returns
    a non-zero exit status.
    """
    def __init__(self, msg: str):
        """Initialize the exception.
        
        Parameters
        ----------
        msg : str
            Error message describing the command failure.
        """
        self.msg = msg
    
    def __str__(self):
        return self.msg
    
    def __repr__(self):
        return self.msg


def safe_run_command(cmd, check=True, text=True, capture_output=True, **kwargs):
    """Safely run a shell command with error handling.
    
    Executes a command using subprocess and provides detailed error
    messages if execution fails. Wraps subprocess.CalledProcessError
    in a more informative :exc:`CommandExecuteError`.
    
    Parameters
    ----------
    cmd : str or list
        Command to execute. Can be a string (for shell=True) or list
        of command and arguments.
    check : bool, optional
        If True, raise exception on non-zero exit. Default is True.
    text : bool, optional
        If True, decode stdout/stderr as text. Default is True.
    capture_output : bool, optional
        If True, capture stdout and stderr. Default is True.
    **kwargs
        Additional arguments passed to :func:`subprocess.run`, such as
        shell, cwd, env, etc.
    
    Returns
    -------
    subprocess.CompletedProcess
        Completed process object with returncode, stdout, and stderr.
    
    Raises
    ------
    CommandExecuteError
        If ``check=True`` and command returns non-zero exit status.
    """
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
    """Context manager to temporarily change working directory.
    
    Changes the current working directory for the duration of the context,
    then restores the original directory when exiting.
    
    Parameters
    ----------
    dirname : os.PathLike
        Directory path to change to. Can be relative or absolute.
    mkdir : bool, optional
        If True, create the directory (and parents) if it doesn't exist.
        Default is False.
    
    Yields
    ------
    Path
        Absolute path of the changed working directory.
    
    Examples
    --------
    >>> with set_directory("some_path", mkdir=True):
    ...     # Current directory is now "some_path"
    ...     do_something()
    >>> # Original directory is restored
    """
    pwd = os.getcwd()
    path = Path(dirname).resolve()
    if mkdir:
        path.mkdir(exist_ok=True, parents=True)
    os.chdir(path)
    yield path
    os.chdir(pwd)