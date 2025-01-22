#!/bin/bash

while true; do
    echo "[$(date)] Starting training process..."
    ./python.sh "/isaac-sim/standalone_examples/api/omni.isaac.kit/TEST_FILES/data_gen.py"
    
    echo "[$(date)] Process ended with exit code $?. Restarting in 3 seconds..."
    sleep 3
done