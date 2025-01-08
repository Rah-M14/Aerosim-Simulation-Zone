#!/bin/bash

read -p "Server's GPU : " sgpu

# kubectl cp ./Final_Files/Docker_Trainer.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/Docker_Trainer.py 
# kubectl cp ./Final_Files/Docker_Trainer_New.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/Docker_Trainer_New.py 
# kubectl cp ./Final_Files/Adv_SocRewards_Neg.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/Adv_SocRewards_Neg.py 
# kubectl cp ./Final_Files/Adv_SocRewards_Coll.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/Adv_SocRewards_Coll.py
# kubectl cp ./perpetual_run.sh $sgpu:/isaac-sim/perpetual_run.sh 
# kubectl cp ./new_perpetual_run.sh $sgpu:/isaac-sim/new_perpetual_run.sh 

# kubectl cp ./Final_Files/GL_Trainer.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/. 
# kubectl cp ./Final_Files/WR_World.png $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/.
# kubectl cp ./Final_Files/Adv_SocRewards_RRT.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/.
# kubectl cp ./Final_Files/RRTStar.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/.
# kubectl cp ./Final_Files/Path_Manager.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/.
# kubectl cp ./gl_perpetual.sh $sgpu:/isaac-sim/.

# kubectl cp ./MAIN_FILES/configs.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/MAIN_FILES/.
# kubectl cp ./MAIN_FILES/Docker_Trainer.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/MAIN_FILES/.
# kubectl cp ./MAIN_FILES/LiDAR_Feed.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/MAIN_FILES/.
# kubectl cp ./MAIN_FILES/Mod_Pegasus_App.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/MAIN_FILES/.
# kubectl cp ./MAIN_FILES/New_RL_Bot_Control.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/MAIN_FILES/.
# kubectl cp ./MAIN_FILES/New_RL_Bot.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/MAIN_FILES/.
# kubectl cp ./MAIN_FILES/Reward_Manager.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/MAIN_FILES/.
# kubectl cp ./MAIN_FILES/RL_Feature_Extractor_n_Model.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/MAIN_FILES/.
# kubectl cp ./MAIN_FILES/Testing_Env.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/MAIN_FILES/.
# kubectl cp ./MAIN_FILES/Traj_Gen $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/MAIN_FILES/.

kubectl cp ./NAV_TEST/configs.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/NAV_TEST/.
kubectl cp ./NAV_TEST/Docker_Trainer.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/NAV_TEST/.
kubectl cp ./NAV_TEST/LiDAR_Feed.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/NAV_TEST/.
kubectl cp ./NAV_TEST/Mod_Pegasus_App.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/NAV_TEST/.
kubectl cp ./NAV_TEST/New_RL_Bot_Control.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/NAV_TEST/.
kubectl cp ./NAV_TEST/New_RL_Bot.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/NAV_TEST/.
kubectl cp ./NAV_TEST/Reward_Manager.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/NAV_TEST/.
kubectl cp ./NAV_TEST/RL_Feature_Extractor_n_Model.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/NAV_TEST/.
kubectl cp ./NAV_TEST/RL_Simple_Models.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/NAV_TEST/.
kubectl cp ./NAV_TEST/Test_Reward_Manager.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/NAV_TEST/.
kubectl cp ./NAV_TEST/Testing_Env.py $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/NAV_TEST/.
kubectl cp ./NAV_TEST/Traj_Gen $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/NAV_TEST/.

kubectl cp ./base_test.sh $sgpu:/isaac-sim/.
# kubectl cp ./nav_test.sh $sgpu:/isaac-sim/.
# kubectl cp ./Final_WR_World/New_Core.usd $sgpu:/isaac-sim/standalone_examples/api/omni.isaac.kit/Final_WR_World/.

echo "Done Copying"