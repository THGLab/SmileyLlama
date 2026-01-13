import json
from smileyllama.workflow import Workflow, WorkflowConfig

cfg_file = './sl_config_basic.yml'
in_csv = './score_standalone.csv'
out_csv = './score_standalone_output.csv'

with open(cfg_file) as f:
    data = json.load(f)
cfg = WorkflowConfig.model_validate(data)
wf = Workflow(cfg)
res = wf.score_df(in_csv)
res.sort_values('total', inplace=True, ascending=False)
res.to_csv(out_csv, index=None)