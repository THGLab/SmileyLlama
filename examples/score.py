from smileyllama.utils import safe_read_yaml
from smileyllama.workflow import Workflow, WorkflowConfig

cfg_file = './sl_config_basic.yml'
in_csv = './score_standalone.csv'
out_csv = './score_standalone_output.csv'
working_dir = '.'

cfg = WorkflowConfig.model_validate(safe_read_yaml(cfg_file))
wf = Workflow(cfg)
res = wf.score_df(in_csv, smiles_col='smiles', working_dir=working_dir)
res.sort_values('total', inplace=True, ascending=False)
res.to_csv(out_csv, index=None)