import argparse, os
from guacamol.benchmark_suites import distribution_learning_suite_v1
from guacamol.distribution_matching_generator import DistributionMatchingGenerator


class MyModelWrapper(DistributionMatchingGenerator):
    def __init__(self, data):
        self.data = data
        self.cursor = 0

    def generate(self, number_samples):
        end = self.cursor + number_samples
        to_return = self.data[self.cursor:end]
        self.cursor = end
        return to_return


class GuacamolAnalysis():
    # A class that loads the chembl path at initialization
    # Has a method that lets it do a quick guacamol analysis of a path to a smilesfile
    def __init__(self, chembl_path, num_samples=10000):
        self.chembl_path = chembl_path
        self.num_samples = num_samples
        self.benchmark_list = distribution_learning_suite_v1(chembl_path, number_samples=num_samples)
    
    def run_benchmarks(self, smiles_path):
        with open(smiles_path, "r") as f:
            smiles_all = f.read().split("\n")
        to_test = MyModelWrapper(smiles_all)
        results_dict = {}
        for i, benchmark in enumerate(self.benchmark_list):
            if i == 3 or i == 4:
                continue
            result = benchmark.assess_model(to_test)
            results_dict[result.benchmark_name] = result.score
        overall_score = results_dict['Validity']*results_dict['Uniqueness']*results_dict['Novelty']
        results_dict['Score'] = overall_score
        results_dict['Filename'] = smiles_path.split("/")[-1]
        return results_dict

def plot_results(all_results, all_smilesfiles, filename):
    #Plot the results of the guacamol analysis
    #all_results is a list of dictionaries
    #each dictionary is the result of a guacamol analysis
    #filename is the name of the file to save the plot to
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
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()



def main():
    #Take in a directory of smiles files and a chembl path
    #Run the guacamol analysis on each smiles file
    parser = argparse.ArgumentParser(description="Run molecule generation benchmarks")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--chembl_path", type=str, default='/path/to/chembl_dataset.txt', help="Path to ChEMBL dataset")
    parser.add_argument("--smiles_dir", type=str, default='/path/to/smiles_files', help="Directory of smiles files to analyze")   
    args = parser.parse_args()
    guac = GuacamolAnalysis(args.chembl_path, args.num_samples)
    print("Running guacamol analysis on smiles files in", args.smiles_dir)
    all_results = []
    all_smilesfiles = sorted(os.listdir(args.smiles_dir))
    for smilesfile in all_smilesfiles:
        smiles_path = os.path.join(args.smiles_dir, smilesfile)
        results = guac.run_benchmarks(smiles_path)
        print(smilesfile, results)
        all_results.append(results)
    plot_results(all_results, all_smilesfiles, "guacamol_results.png")
    
if __name__ == "__main__":
    main()
