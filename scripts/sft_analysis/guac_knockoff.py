import multiprocessing
from rdkit import Chem, RDLogger
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os

def canonicalize_smiles(sm_str):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(sm_str))
    except:
        return None

def canonicalize_smiles_list(sm_list):
    #Parallelized version of canonicalize_smiles
    cpu_count = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=cpu_count) as pool:
        results = list(tqdm(pool.imap(canonicalize_smiles, sm_list), total=len(sm_list)))
        #remove the element None if it exists
        if None in results:
            results.remove(None)
        return results

def get_guac_score(smiles_to_analyze, training_set_smiles):
    #Get the Guac score for a list of smiles
    print("Calculating validity ratio...")
    valid_smiles = [s for s in smiles_to_analyze if Chem.MolFromSmiles(s) is not None]
    validity_ratio = len(valid_smiles) / len(smiles_to_analyze)
    print("Calculating uniqueness ratio...")
    canonical_valid_smiles = canonicalize_smiles_list(valid_smiles)
    #Unique ratio is the ratio of unique valid smiles to the total number of valid smiles
    unique_canonical_valid_smiles_set = set(canonical_valid_smiles)
    unique_ratio = len(unique_canonical_valid_smiles_set) / len(canonical_valid_smiles)
    print("Calculating novelty ratio...")
    #Novelty ratio is the ratio of unique valid smiles not in the training set to the total number of unique valid smiles
    novel_ratio = len(unique_canonical_valid_smiles_set - training_set_smiles) / len(unique_canonical_valid_smiles_set)
    overall_score = validity_ratio * unique_ratio * novel_ratio
    return {"Validity": validity_ratio, "Uniqueness": unique_ratio, "Novelty": novel_ratio, "Score": overall_score}

def plot_results(all_results, pngfile, loggerfile=None):
    #Plot the results of the guacamol analysis
    #all_results is a list of dictionaries
    #each dictionary is the result of a guacamol analysis
    #pngfile is the name of the file to save the plot to
    if loggerfile is not None:
        with open(loggerfile, "a") as f:
            f.write("Plotting results for " + pngfile + "\n")
            for result in all_results:
                f.write(str(result) + "\n")
            f.write("\n")
    import matplotlib.pyplot as plt
    # Extract filenames and metrics
    filenames = [entry['Filename'] for entry in all_results]
    validity = [entry['Validity'] for entry in all_results]
    uniqueness = [entry['Uniqueness'] for entry in all_results]
    novelty = [entry['Novelty'] for entry in all_results]
    score = [entry['Score'] for entry in all_results]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each metric with a different color
    plt.plot(filenames, validity, label='Validity', color='blue', marker='o')
    plt.plot(filenames, uniqueness, label='Uniqueness', color='green', marker='o')
    plt.plot(filenames, novelty, label='Novelty', color='red', marker='o')
    plt.plot(filenames, score, label='Score', color='purple', marker='o')

    # Add labels and title
    plt.xlabel('Files')
    plt.ylabel('Metrics')
    plt.title('Metrics by Filename')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add a legend to identify the metrics
    plt.legend()

    # Display the plot
    plt.savefig(pngfile)
    plt.show()


class Knockoff_Analysis:
    #Takes in the path to a txt file with the training set smiles
    def __init__(self, training_set_path):
        cpu_count = multiprocessing.cpu_count()
        print("Using", cpu_count, "cores")
        with open(training_set_path, "r") as f:
            training_set_smiles = f.read().splitlines()
        print("Canonicalizing training set smiles...")
        self.training_set_smiles = set(canonicalize_smiles_list(training_set_smiles))
    
    def score_smiles_from_file(self, smiles_file_path):
        with open(smiles_file_path, "r") as f:
            smiles_to_analyze = f.read().splitlines()
        scores = get_guac_score(smiles_to_analyze, self.training_set_smiles)
        scores["Filename"] = smiles_file_path
        return scores


def main():
    #Suppress RDKit warnings
    RDLogger.DisableLog('rdApp.*')
    #Take in a directory of smiles files and a chembl path
    #Run the guacamol analysis on each smiles file
    parser = argparse.ArgumentParser(description="Run molecule generation benchmarks")
    parser.add_argument("--chembl_path", type=str, default='/path/to/chembl_dataset.txt', help="Path to ChEMBL dataset")
    parser.add_argument("--smiles_dir", type=str, default='/path/to/smiles_files', help="Directory of smiles files to analyze")
    args = parser.parse_args()
    guac = Knockoff_Analysis(args.chembl_path)
    print("Running guacamol analysis on smiles files in", args.smiles_dir)
    all_results = []
    all_smilesfiles = sorted(os.listdir(args.smiles_dir))
    for smilesfile in all_smilesfiles:
        smiles_path = os.path.join(args.smiles_dir, smilesfile)
        results = guac.score_smiles_from_file(smiles_path)
        print(smilesfile, results)
        all_results.append(results)
    plot_results(all_results, "guacamol_results.png", loggerfile="guacamol_results.log")


if __name__ == "__main__":
    main()



