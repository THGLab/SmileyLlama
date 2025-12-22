import os, shutil
from smileyllama.workflow import Workflow, WorkflowConfig


config_dict = {
    "directory": os.path.join(os.path.dirname(__file__), '_test_workflow'),
    "niters": 2,
    "model_path": "/global/scratch/users/ericwangyz/smiley-enamine/model-1B/model/merged",
    # "model_path": "THGLab/Llama-3.1-8B-SmileyLlama-1.1",
    "num_samples_per_iter": 128,
    "log_level": 'debug',
    'user_prompt_properties': ['High SARS2PRO', '<= 5 H-bond donors', '<= 10 H-bond acceptors', '<= 500 molecular weight', '<= 5 logP'],
    "scores": {
        "dock": {
            "name": "UniDock", "start_iter": 0, "end_iter": -1,
            "parameters": {
                "protein": os.path.join(os.path.dirname(__file__), 'data/protein.pdb'),
                "box_center": [21.74425, -5.3926, 27.91045],
                "box_size": [25.0, 25.0, 25.0],
                "exec_path": '/global/home/groups/fc_armada2/conda_envs/unidock/bin/unidock'
                # "exec_path": 'vina_gpu'
            },
            "normalizer": {
                "name": "MinMaxNormalizer",
                "parameters": {
                    "vmin": -12.0,
                    "vmax": 0.0,
                    "negate": True,
                },
            },
        },
        "qed": {
            "name": "QED", "start_iter": 0, "end_iter": -1, "weight": 0.0,
            "parameters": {},
            "normalizer": {"name": "Identity"},
        },
        "molwt": {
            "name": "MolWt", "start_iter": 0, "end_iter": -1, "weight": 1.0, "as_filter": True,
            "normalizer": {"name": "StepNormalizer", "parameters": {"sign": '<=', 'val': 500.0}}
        },
        "hbd": {
            "name": "NumHBD", "start_iter": 0, "end_iter": -1, "weight": 1.0, "as_filter": True,
            "normalizer": {"name": "StepNormalizer", "parameters": {"sign": '<=', 'val': 5}}
        },
        "hba": {
            "name": "NumHBA", "start_iter": 0, "end_iter": -1, "weight": 1.0, "as_filter": True,
            "normalizer": {"name": "StepNormalizer", "parameters": {"sign": '<=', 'val': 10}}
        },
        "logp": {
            "name": "LogP", "start_iter": 0, "end_iter": -1, "weight": 1.0, "as_filter": True,
            "normalizer": {"name": "StepNormalizer", "parameters": {"sign": '<=', 'val': 5}}
        },
    },
    "rl_config": {
        "config_file": os.path.join(os.path.dirname(__file__), 'data/dpo.yml'),
        "dpo_num_pairs_per_smiles": 2,
        "dpo_score_margin": 0.1,
        "use_random_smiles": True
    }
}


def test_workflow():
    config = WorkflowConfig(**config_dict)
    if os.path.isdir(config.directory):
        shutil.rmtree(config.directory)
    wf = Workflow(config)
    wf.run()
