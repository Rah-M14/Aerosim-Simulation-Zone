FROM nvcr.io/nvidia/isaac-sim:4.2.0 AS isaac-sim

WORKDIR /isaac-sim

RUN ln -s exts/omni.isaac.examples/omni/isaac/examples extension_examples

COPY . /isaac-sim/.

ARG WAPI_KEY
ENV WANDB_API_KEY=$WAPI_KEY

WORKDIR /app

# # Install Python 3.10 and set it as the default
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils git htop nvtop screen tmux vim && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

WORKDIR /isaac-sim/standalone_examples/api/omni.isaac.kit/

RUN git clone https://github.com/PegasusSimulator/PegasusSimulator.git && \
    cd ./PegasusSimulator/extensions && \
    /isaac-sim/python.sh -m pip install --editable pegasus.simulator && \
    cd /isaac-sim

WORKDIR /isaac-sim

RUN ./python.sh -m pip install --no-cache-dir -r requirements.txt && \
    ./python.sh -m pip install --no-cache-dir torch torchvision torchaudio wandb stable-baselines3[extra] gym ultralytics scikit-learn shapely shimmy

RUN ./python.sh -m wandb login $WANDB_API_KEY

RUN mv /isaac-sim/Final_Files/* /isaac-sim/standalone_examples/api/omni.isaac.kit/. && \
    mv /isaac-sim/configs/* /isaac-sim/exts/omni.isaac.sensor/data/lidar_configs/SLAMTEC/. && \
    mv /isaac-sim/Final_WR_World /isaac-sim/standalone_examples/api/omni.isaac.kit/. 

CMD ["./python.sh", "/isaac-sim/standalone_examples/api/omni.isaac.kit/Docker_Trainer.py", "--algo", "ppo", "--botname", "jackal", "--headless", "--state_normalize" ]

# sudo docker build --build-arg WAPI_KEY=$WANDB_API -t isaac_build .