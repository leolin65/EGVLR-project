#!/bin/bash
# NOTE: this SBATCH template targets a specific university HPC cluster (module/conda paths below).
# Adapt --account, module load, and the conda activate/source lines to your own cluster.
#SBATCH --account=YOUR_PROJECT_ACCOUNT                 # (-A) iService Project ID
#SBATCH --job-name=Qwen_t                   # (-J) Job name
#SBATCH --partition=normal                  # (-p) Slurm partition
#SBATCH --nodes=1                           # (-N) Maximum number of nodes to be allocated
#SBATCH --gpus-per-node=4                   # GPUs per node
#SBATCH --cpus-per-task=4                   # (-c) Number of cores per MPI task
#SBATCH --ntasks-per-node=1                 # Maximum number of tasks on each node
#SBATCH --time=2-00:00:00                   # (-t) Wall time limit (days-hrs:min:sec)
#SBATCH --output=out/job-%j.out             # (-o) Path to the standard output file
#SBATCH --error=out/job-%j.err              # (-e) Path to the standard error file
#SBATCH --mail-type=BEGIN,FAIL,END          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=YOUR_EMAIL@example.com # Where to send mail

module purge
module load cuda/12.6

source ~/.bashrc
source /work/HPC_software/LMOD/miniconda3/miniconda3_app/24.11.1/etc/profile.d/conda.sh
conda activate train
echo "Python path: $(which python)"

export WANDB_MODE=offline

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export PATH=$CONDA_PREFIX/bin:$PATH

ln -sf $CC $CONDA_PREFIX/bin/gcc
ln -sf $CXX $CONDA_PREFIX/bin/g++

echo "=== Using GCC version ==="
which gcc
gcc --version
echo "========================="

cd /path/to/Qwen-VL-Series-Finetune  # your Qwen-VL-Series-Finetune checkout, with this repo's training/ merged in
CMD="bash training/finetune_stage3_gs_dpo.sh"

srun $CMD

echo "Finish"