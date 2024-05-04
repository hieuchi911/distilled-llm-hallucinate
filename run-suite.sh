#!/bin/bash

#SBATCH --job-name=multinode
#SBATCH --account=yzhao010_1246
#SBATCH --partition=gpu
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=4
#SBATCH --cpus-per-task=4

source ~/.bashrc

module purge
module load gcc/11.3.0 jq/1.7.1

conda activate hallu

pip -V

nvidia-smi

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

json_file="models.json"

for index in $(jq 'keys | .[]' $json_file); do
    task_name=$(jq -r ".[$index].task_name" $json_file)
    echo "Task: $task_name"
    for dataset in $(jq -r ".[$index].datasets | keys | .[]" $json_file); do
        dname=$(jq -r ".[$index].datasets[$dataset].name" $json_file)
        hf_path=$(jq -r ".[$index].datasets[$dataset].hf_path" $json_file)
        subset=$(jq -r ".[$index].datasets[$dataset].subset" $json_file)
        split=$(jq -r ".[$index].datasets[$dataset].split" $json_file)
        # echo "---> Datasets: name: $dname, HF Path: $hf_path, Subset: $subset, Split: $split"
        
        for model in $(jq -r ".[$index].models | keys | .[]" $json_file); do
            language_model=$(jq -r ".[$index].models[$model].language_model" $json_file)
            tokenizer=$(jq -r ".[$index].models[$model].tokenizer" $json_file)
            srun torchrun --nnodes 8 --nproc_per_node 1 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29603 main_ddp.py -t $task_name -dn $dname -dp $hf_path -ds $subset -dspl $split -dm ./data_map.py -bs 4 -o ./eval_results_db -db -tok $tokenizer -lm $language_model
        done
    done

    echo "-------------------------"
    echo ""
done