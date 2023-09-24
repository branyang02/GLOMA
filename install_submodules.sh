#!/bin/bash

# Update and initialize submodules
git submodule update --init --recursive

# Environment variable checks
if [ "$AM_I_DOCKER" == "True" ]; then
    echo "Cannot build in Docker container"
    exit 1
fi

if [ "$BUILD_WITH_CUDA" == "False" ]; then
    echo "Cannot build without CUDA"
    exit 1
fi

# Grounded Segment Anything
cd submodules/Grounded-Segment-Anything

# Using python -m pip ensures you're using the pip associated with the current python
echo "Installing SAM"
python -m pip install -e segment_anything
if [ $? -ne 0 ]; then
    echo "Error installing segment_anything"
    exit 1
fi

echo "Installing GroundingDINO"
python -m pip install -e GroundingDINO
if [ $? -ne 0 ]; then
    echo "Error installing GroundingDINO"
    exit 1
fi

echo "Installing diffusers[torch]"
pip install --upgrade diffusers[torch]
if [ $? -ne 0 ]; then
    echo "Error updating diffusers[torch]"
    exit 1
fi

echo "Installing required packages"
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel

cd ..

echo "Installing GLIGEN"
# GLIGEN
cd GLIGEN
# do stuff
# Remember to check the exit status if necessary
cd ../..


# # install checkpoints
# mkdir checkpoints && cd checkpoints
# wget wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# pip install wldhx.yadisk-direct
# curl -L $(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) -o big-lama.zip
# unzip big-lama.zip

