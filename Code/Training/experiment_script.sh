#!/bin/bash

# Setup env
#source $HOME/miniconda3/etc/profile.d/conda.sh
source activate blood-pressure-project
echo "hello from $(python --version) in $(which python)"

# Run some arbitrary python
python -c 'import torch; print(f"i can haz gpu? {torch.cuda.is_available()}")'

echo "Dias Model"
echo "Learning rate experiment"

echo "Batch size = 64"
for lr in 0.1 0.01 0.001 0.005
  do
    python Experiments.py run-exp -n exp_lr_64 --epochs 20 --bs-train 64 --batches 1000 --seed 42 --early-stopping 20 --model-name dias_model --lr $lr;
  done
printf "\n"

echo "Batch size = 32"
for lr in 0.1 0.01 0.001 0.005
  do
    python Experiments.py run-exp -n exp_lr_32 --epochs 20 --bs-train 32 --batches 1000 --seed 42 --early-stopping 20 --model-name dias_model --lr $lr;
  done
printf "\n"
#
echo "Sys Model"
echo "Learning rate experiment"

echo "Batch size = 64"
for lr in 0.1 0.01 0.001 0.005
  do
    python Experiments.py run-exp -n exp_lr_64 --epochs 20 --bs-train 64 --batches 1000 --seed 42 --early-stopping 20 --model-name sys_model --lr $lr;
  done
printf "\n"
#
echo "Batch size = 32"
for lr in 0.1 0.01 0.001 0.005
  do
    python Experiments.py run-exp -n exp_lr_32 --epochs 20 --bs-train 32 --batches 1000 --seed 42 --early-stopping 20 --model-name sys_model --lr $lr;
  done
printf "\n"