GRANDPARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source $GRANDPARENT_DIR/envs/ana-env/bin/activate

for i in {1..16}
do
	source $GRANDPARENT_DIR/envs/ana-env/bin/activate
        #Prepare next iteration
        python prep_iter.py $i
	
	deactivate
	source $GRANDPARENT_DIR/envs/axo/bin/activate
        #Run DPO with LoRA
        accelerate launch --use-deepspeed -m axolotl.cli.train cf_dpo_lora.yml --dataset_processes=1

        #Merge LoRA with original model
        python3 -m axolotl.cli.merge_lora $(pwd)/cf_dpo_lora.yml --lora_model_dir="$(pwd)/outputs"
	deactivate
        mv dpodataset/dpodataset.jsonl old_dpodatasets/dpodataset$i.jsonl
        cp -r $(pwd)/outputs/merged $(pwd)/model_iter$i
done