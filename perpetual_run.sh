#!/bin/bash

read -p "Algo to be used : " algo
read -p "GPUs to be used : " gpu

echo "Algo being used : $algo"
echo "GPUs being used : $gpu"

while true; do
    echo "[$(date)] Starting training process..."
    ./python.sh "/isaac-sim/standalone_examples/api/omni.isaac.kit/Docker_Trainer.py" --algo $algo --botname jackal --state_normalize --resume_checkpoint --headless -g $gpu
    
    echo "[$(date)] Process ended with exit code $?. Restarting in 3 seconds..."
    sleep 3
done