import argparse
from guacamol.benchmark_suites import distribution_learning_suite_v1
from guacamol.distribution_matching_generator import DistributionMatchingGenerator


class MyModelWrapper(DistributionMatchingGenerator):
    def __init__(self, my_model, data):
        self.model = my_model
        self.data = data
        self.counter = 0

    def generate(self, number_samples):
        to_return = self.data[self.counter : self.counter + number_samples]
        self.counter += number_samples
        #print("counter is ", self.counter)
        return to_return

def run_benchmarks(chembl_path, llama_smiles_path, num_samples=10000):
    benchmark_list = distribution_learning_suite_v1(chembl_path, number_samples=num_samples)

    with open(llama_smiles_path, "r") as f:
        llama31_smiles_all = f.read().split("\n")

    
    
    to_test = MyModelWrapper([], llama31_smiles_all)
    results = []
    for i, benchmark in enumerate(benchmark_list):
        result = benchmark.assess_model(to_test)
        results.append(result)
        print(result.benchmark_name, result.score)
    return results
chembl_path = '/path/to/sft/chembl_random_smiles.txt'
num_samples=10000


results_0shot_128mnt = run_benchmarks(chembl_path, '0_shot_128mnt.txt', num_samples)
results_20shot_128mnt = run_benchmarks(chembl_path, '20_shot_128mnt.txt', num_samples)