#!/bin/bash

read -p "Server's GPU : " sgpu

if [ $sgpu = '4090' ]; then
	kubectl cp ./Final_Files/Docker_Trainer.py omniverse-test-49-99786f589-nw848:/isaac-sim/standalone_examples/api/omni.isaac.kit/Docker_Trainer.py 
    kubectl cp ./Final_Files/Docker_Trainer_New.py omniverse-test-49-99786f589-nw848:/isaac-sim/standalone_examples/api/omni.isaac.kit/Docker_Trainer_New.py 
    kubectl cp ./perpetual_run.sh omniverse-test-49-99786f589-nw848:/isaac-sim/perpetual_run.sh 
    kubectl cp ./new_perpetual_run.sh omniverse-test-49-99786f589-nw848:/isaac-sim/new_perpetual_run.sh 
    kubectl cp ./Final_Files/Adv_SocRewards_Neg.py omniverse-test-49-99786f589-nw848:/isaac-sim/standalone_examples/api/omni.isaac.kit/Adv_SocRewards_Neg.py 
    kubectl cp ./Final_Files/Adv_SocRewards_Coll.py omniverse-test-49-99786f589-nw848:/isaac-sim/standalone_examples/api/omni.isaac.kit/Adv_SocRewards_Coll.py

elif [ $sgpu = '3090' ]; then
	kubectl cp ./Final_Files/Docker_Trainer.py omniverse-test-1-67666bcf6b-jddzg:/isaac-sim/standalone_examples/api/omni.isaac.kit/Docker_Trainer.py 
    kubectl cp ./Final_Files/Docker_Trainer_New.py omniverse-test-1-67666bcf6b-jddzg:/isaac-sim/standalone_examples/api/omni.isaac.kit/Docker_Trainer_New.py 
    kubectl cp ./perpetual_run.sh omniverse-test-1-67666bcf6b-jddzg:/isaac-sim/perpetual_run.sh 
    kubectl cp ./new_perpetual_run.sh omniverse-test-1-67666bcf6b-jddzg:/isaac-sim/new_perpetual_run.sh 
    kubectl cp ./Final_Files/Adv_SocRewards_Neg.py omniverse-test-1-67666bcf6b-jddzg:/isaac-sim/standalone_examples/api/omni.isaac.kit/Adv_SocRewards_Neg.py 
    kubectl cp ./Final_Files/Adv_SocRewards_Coll.py omniverse-test-1-67666bcf6b-jddzg:/isaac-sim/standalone_examples/api/omni.isaac.kit/Adv_SocRewards_Coll.py
else
	echo "Pod with the specified gpu doesnt exist. Try again"
fi


