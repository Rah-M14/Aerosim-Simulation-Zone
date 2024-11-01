FROM nvcr.io/nvidia/isaac-sim:4.1.0 as isaac-sim

WORKDIR /isaac-sim

RUN ln -s exts/omni.isaac.examples/omni/isaac/examples extension_examples

COPY . /isaac-sim/

ENV WANDB_API_KEY=$(WANDB_API)

# WORKDIR /app

# # Install Python 3.10 and set it as the default
RUN apt-get update && \
    apt-get -y install git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

    # apt-get install -y python3.10 python3.10-dev python3.10-distutils git && \
#     update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
#     apt-get clean && \
#     curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# WORKDIR /isaac-sim

RUN git clone https://github.com/PegasusSimulator/PegasusSimulator.git

RUN cd ./PegasusSimulator/extensions && \
    /isaac-sim/python.sh -m pip install --editable pegasus.simulator

RUN cd /isaac-sim

RUN ./python.sh -m pip install --no-cache-dir -r requirements.txt && \
    ./python.sh -m pip install --no-cache-dir torch torchvision torchaudio wandb stable-baselines3[extra] gym ultralytics scikit-learn

RUN ./python.sh -m wandb login $WANDB_API_KEY

RUN mv /isaac-sim/Final_Files/* /isaac-sim/standalone_examples/api/omni.isaac.kit && \
    mv /isaac-sim/configs/* /isaac-sim/exts/omni.isaac.sensor/data/lidar_configs/SLAMTEC && \
    mv /isaac-sim/SIM_Files /isaac-sim/standalone_examples/api/omni.isaac.kit 

EXPOSE 80

CMD ["./python.sh", "/isaac-sim/standalone_examples/api/omni.isaac.kit/L_Theta_RL.py", "--algo", "ppo", "--botname", "jackal", "--headless", "--state_normalize" ]