#!/bin/bash
#SBATCH --job-name=tng  # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=g.kerex@gmail.com     # Where to send mail
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH -p gpu --gpus=1 -c 16

###SBATCH --partition=cmbas
###SBATCH -C skylake,opa
###SBATCH --nodes=1
###SBATCH --ntasks-per-node=2
###SBATCH -p cmbas -C rome,ib
###SBATCH -p gen
###SBATCH -p gpu --gpus=<gpu_type>:<N> -c <M>

pwd; hostname; date

module add cuda
module add cudnn
module add python3


cd $(pwd)

~/pyenv/venv/bin/python3 ./data_extraction_TNG_SIMBA.py >stdout 2>stderr
#~/pyenv/venv/bin/python3 ./data_extraction_ASTRID.py >stdouta 2>stderra

date
