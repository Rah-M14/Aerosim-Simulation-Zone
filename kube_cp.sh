#!/bin/bash

read -p "Server's GPU : " sgpu

kubectl cp ./Final_Files/Docker_Trainer.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/Docker_Trainer.py 
kubectl cp ./Final_Files/Docker_Trainer_New.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/Docker_Trainer_New.py 
kubectl cp ./Final_Files/Adv_SocRewards_Neg.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/Adv_SocRewards_Neg.py 
kubectl cp ./Final_Files/Adv_SocRewards_Coll.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/Adv_SocRewards_Coll.py
kubectl cp ./perpetual_run.sh $sgpu:/isaac-sim/perpetual_run.sh 
kubectl cp ./new_perpetual_run.sh $sgpu:/isaac-sim/new_perpetual_run.sh 

kubectl cp ./Final_Files/GL_Trainer.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/. 
kubectl cp ./Final_Files/WR_World.png $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/.
kubectl cp ./Final_Files/Adv_SocRewards_RRT.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/.
kubectl cp ./Final_Files/RRTStar.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/.
kubectl cp ./Final_Files/Path_Manager.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/.
kubectl cp ./gl_perpetual.sh $sgpu:/isaac-sim/.

echo "Done Copying"