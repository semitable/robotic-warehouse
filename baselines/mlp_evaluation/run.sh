#!/bin/bash
#SBATCH --job-name=a2c_ind
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=5GB
#SBATCH --nodes=4
#SBATCH --tasks-per-node 1
#SBATCH --gres=gpu:4

if ! [ "$#" -ge 4 ]
then
    echo 'Usage: run <train_script> <job_name> <num_nodes> <conda_env> (<cpus_per_task=12>) (<mem_per_cpu=5>) (<num_gpus=4>)'
    return
fi

train_script="$1"
job_name="$2"
num_nodes="$3"
conda_env="$4"

if [ "$#" -ge 5 ]
then
    cpus_per_task="$5"
else
    cpus_per_task=5
fi

if [ "$#" -ge 6 ]
then
    mem_per_cpu="$6"
else
    mem_per_cpu=5
fi

if [ "$#" -eq 7 ]
then
    num_gpus="$7"
else
    num_gpus=4
fi

if [ "$#" -ge 8 ]
then
    echo 'Usage: run <train_script> <job_name> <num_nodes> <conda_env> (<cpus_per_task=12>) (<mem_per_cpu=5>) (<num_gpus=4>)'
    return
fi

username=$(whoami)

worker_num=3 # Must be one less that the total number of nodes

mkdir -p /disk/scratch/$username
source ~/.bashrc
source activate $conda_env

# module load Langs/Python/3.6.4 # This will vary depending on your environment
# source venv/bin/activate

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=$( nodes )
node1=${nodes_array[0]}
ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)
export ip_head # Exporting for latter access by trainer.py

head_gpus=2
#$(num_gpus)/2

srun --nodes=1 --ntasks=1 --gres=gpu:2 -w $node1 ray start --num-cpus=$cpus_per_task --block --head --temp-dir /disk/scratch/$username --redis-port=6379 --redis-password=$redis_password & # Starting the head

sleep 5

worker_gpus=2
#$(num_gpus)/2

for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 --gres=gpu:2 -w $node2 ray start --num-cpus=12 --temp-dir /disk/scratch/$username --block --address=$ip_head --redis-password=$redis_password & # Starting the workers
  sleep 5
done

python -u $train_script --ip-head $ip_head --redis-pwd $redis_password
