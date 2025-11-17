import argparse
from guacamol.benchmark_suites import distribution_learning_suite_v1
from guacamol.distribution_matching_generator import DistributionMatchingGenerator

class MyModelWrapper(DistributionMatchingGenerator):
    def __init__(self, my_model, data):
        self.model = my_model
        self.data = data

    def generate(self, number_samples):
        return self.data[:number_samples]

def run_benchmarks(chembl_path, llama_smiles_path, num_samples=10000):
    benchmark_list = distribution_learning_suite_v1(chembl_path, number_samples=num_samples)

    with open(llama_smiles_path, "r") as f:
        llama31_smiles_all = f.read().split("\n")

    to_test = MyModelWrapper([], llama31_smiles_all)
    results = []
    results_dict = {}
    for i, benchmark in enumerate(benchmark_list):
        if i == 4:
            continue
        result = benchmark.assess_model(to_test)
        results.append(result)
        print(result.benchmark_name, result.score)
        results_dict[result.benchmark_name] = result.score

    return results_dict

def main():
    parser = argparse.ArgumentParser(description="Run molecule generation benchmarks")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--chembl_path", type=str, default='/path/to/chembl_dataset.txt', help="Path to ChEMBL dataset")
    parser.add_argument("--llama_smiles_path", type=str, default='/path/to/smiles_file.txt', help="Path to LLAMA SMILES file")
    
    args = parser.parse_args()

    results = run_benchmarks(args.chembl_path, args.llama_smiles_path, args.num_samples)

    # You can add more processing or output of results here if needed
    print(results)

if __name__ == "__main__":
    main()
