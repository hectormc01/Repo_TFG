FROM nvidia/cuda:11.3.0-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo python3-pip && \
  rm -rf /var/lib/apt/lists/*

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
# RUN pip3 install torch torchvision tensorboard cython
RUN pip3 install cython
RUN pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN pip3 install 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
RUN git clone --branch v0.3 https://github.com/facebookresearch/detectron2 detectron2_repo
ENV FORCE_CUDA="1"
# This will build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing;Ampere"
RUN pip3 install -e detectron2_repo

RUN pip3 install scipy pandas scikit-image sklearn

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
ENV PYTHONPATH="/workspace/detectron"

# Install LVIS
RUN pip install lvis