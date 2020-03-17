#!/bin/bash
#SBATCH --job-name=all
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12GB
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1

if ! [ "$#" -eq 1 ]
then
    echo 'Usage: run <conda_env>'
    return
fi

conda_env="$1"

username=$(whoami)

worker_num="$(($SLURM_JOB_NUM_NODES-1))" # Must be one less that the total number of nodes

mkdir -p /disk/scratch/$username
source ~/.bashrc
source activate $conda_env

# module load Langs/Python/3.6.4 # This will vary depending on your environment
# source venv/bin/activate

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )
node1=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)
export ip_head # Exporting for latter access by trainer.py

srun --nodes=1 --ntasks=1 -w $node1 ray start --num-cpus=$SLURM_CPUS_ON_NODE --block --head --temp-dir /disk/scratch/$username --redis-port=6379 --redis-password=$redis_password & # Starting the head

sleep 10

for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --num-cpus=$SLURM_CPUS_ON_NODE --temp-dir /disk/scratch/$username --block --address=$ip_head --redis-password=$redis_password & # Starting the workers
  sleep 5
done

python -u ../a2c_ind.py --ip-head $ip_head --redis-pwd $redis_password &
python -u ../dqn.py --ip-head $ip_head --redis-pwd $redis_password &
python -u ../ppo.py --ip-head $ip_head --redis-pwd $redis_password
