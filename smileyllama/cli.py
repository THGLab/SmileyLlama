from __future__ import annotations

import argparse
import sys
from .utils import safe_read_yaml
from .workflow import Workflow, WorkflowConfig


def main():
    parser = argparse.ArgumentParser(prog="smileyllama-workflow")
    parser.add_argument("config", type=str, help="Path to workflow YAML")
    args = parser.parse_args()

    cfg_dict = safe_read_yaml(args.config)
    config = WorkflowConfig(**cfg_dict)

    wf = Workflow(config)
    wf.run()


if __name__ == "__main__":
    main()
