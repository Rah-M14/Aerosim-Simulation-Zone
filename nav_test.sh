#!/bin/bash

read -p "Algo to be used : " algo
read -p "Wandb Project name : " wandb_p
read -p "GPUs to be used : " gpu

while true; do
    echo "[$(date)] Starting training process..."
    ./python.sh "/isaac-sim/standalone_examples/api/omni.isaac.kit/MAIN_FILES/Docker_Trainer.py" --algo $algo --botname jackal --wandb_project $wandb_p --state_normalize --resume_checkpoint --headless -g $gpu
    
    echo "[$(date)] Process ended with exit code $?. Restarting in 3 seconds..."
    sleep 3
done