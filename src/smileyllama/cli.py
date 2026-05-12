from __future__ import annotations

import argparse
import sys
from .utils import safe_read_yaml
from .workflow import Workflow, WorkflowConfig


def main():
    """Main entry point for the SmileyLlama workflow CLI.
    
    Parses command-line arguments, loads workflow configuration from YAML,
    and runs the workflow. This is the entry point when running the
    ``sl`` command script.
    
    Command-line Arguments
    ----------------------
    config : str
        Path to workflow YAML configuration file.
    
    See Also
    --------
    :class:`~smileyllama.workflow.Workflow` : Main workflow class that executes the pipeline.
    :class:`~smileyllama.workflow.WorkflowConfig` : Configuration model for workflow parameters.
    """
    parser = argparse.ArgumentParser(prog="smileyllama-workflow")
    parser.add_argument("config", type=str, help="Path to workflow YAML")
    args = parser.parse_args()

    cfg_dict = safe_read_yaml(args.config)
    config = WorkflowConfig(**cfg_dict)

    wf = Workflow(config)
    wf.run()


if __name__ == "__main__":
    main()
