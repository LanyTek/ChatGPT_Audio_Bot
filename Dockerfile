FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

WORKDIR /app

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install python3 python3-pip ffmpeg libsndfile1 curl -y

COPY requirements.txt requirements.txt

RUN python3 -m pip install pip --upgrade
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install nemo_toolkit['all']

COPY . .