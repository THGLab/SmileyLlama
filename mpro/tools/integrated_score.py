import sys, os, yaml, argparse
import numpy as np
import pandas as pd
from iMiner.rl_generate.core.reward import RewardAssigner
from pathlib import Path
script_dir = Path(__file__).parent


def get_score(smiles, output_dir):
    config = script_dir / "integrated_config.yaml"
    with open(config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    output_directory = output_dir
    if output_directory[-1] == "/":
        output_directory = output_directory[:-1] # remove trailing slash
    if os.path.exists(output_directory):
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        output_directory += "_" + timestamp

    os.makedirs(output_directory)
    os.makedirs(output_directory + "/docking")
    #os.makedirs(output_directory + "/models")
    print("All results logged in", output_directory)
    config["run"]["output_dir"] = output_directory

    rewards = RewardAssigner(reward_combination_method="sum", tokens=None,
        logger=None, output_path=output_directory + "/docking", start_iteration=0)
    for item in config["rewards"]:
        if type(item) is str:
            rewards.add_reward(item)
        elif type(item) is dict:
            if "weight" in item and "params" in item:
                rewards.add_reward(item["type"], weight=item["weight"], extra_params=item["params"])
            elif "weight" in item:
                rewards.add_reward(item["type"], extra_params=item["weight"])
            elif "params" in item:
                rewards.add_reward(item["type"], extra_params=item["params"])
            else:
                rewards.add_reward(item["type"])

    retry_count = 0
    maximum_retry = 3
    print("Querying reward...")
    while True:
        try:
            collected_results = rewards.calc_reward_parallel(smiles)
        except Exception as e:
            collected_results = None
            print(e)
        if collected_results is None:
            if retry_count <= maximum_retry:
                print("Failed collecting valid data on trial %d. Retrying..." % retry_count)
                retry_count += 1
                continue
            else:
                raise RuntimeError("Cannot collect valid data within %d retries" % maximum_retry)
        else:
            all_rewards, all_validities, mean_valid_rewards, df = collected_results
            break
    print("Reward collection finished")
    return all_rewards

def main():
    parser = argparse.ArgumentParser(description="Process SMILES strings and save their scores to a CSV file.")
    parser.add_argument('--input_smiles', type=str, required=True, help="Path to the list of SMILES strings file")
    parser.add_argument('--output_directory', type=str, default="./", help="Path to save docking information")
    parser.add_argument('--output_csv', type=str, required=True, help="Path to the output CSV file")
    
    args = parser.parse_args()
    
    with open(args.input_smiles, "r") as f:
        smiles = f.read().splitlines()  # read lines and remove newline characters
    
    output_directory = args.output_directory
    scores = get_score(smiles, output_directory)
    
    df = pd.DataFrame({'smiles': smiles, 'score': scores})
    df.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    main()
