FROM nvcr.io/nvidia/isaac-sim:4.1.0 as isaac-sim

WORKDIR /isaac-sim

RUN ln -s exts/omni.isaac.examples/omni/isaac/examples extension_examples

COPY . /app

WORKDIR /app

# Install Python 3.10 and set it as the default
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install torch torchvision torchaudio
RUN pip3 install wandb

RUN mv /app/Final_Files/* /isaac-sim/standalone_examples/api/omni.isaac.kit && \
    mv /app/configs/* /isaac-sim/exts/omni.isaac.sensor/data/lidar_configs/SLAMTEC && \
    mv /app/SIM_Files /isaac-sim/standalone_examples/api/omni.isaac.kit 

EXPOSE 80

CMD ["python3.10", "/isaac-sim/standalone_examples/api/omni.isaac.kit/L_Theta_RL.py"]