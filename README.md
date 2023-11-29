# GLOMA - Grounded Location for Object Manipulation

GLOMA is an AI-powered image editing tool that leverages state-of-the-art Large Language Models (LLM) and Diffusion models for seamless image manipulations based on user-provided text prompts. As an advanced image editing tool, GLOMA combines natural language processing and image processing to execute complex image manipulations described in textual instructions.

The key strength of GLOMA lies in its ability to interpret and act on text prompts that describe actions to be performed on an image. Users can provide instructions such as "move the cup behind the drawer," and GLOMA can intelligently execute such commands, resulting in a manipulated image that reflects the requested changes. This unique approach enables a wide range of applications, from creative image editing to more advanced use cases like scene generation and object relocation.

GLOMA's intuitive interface allows users to communicate their image editing needs in natural language, removing the barriers associated with traditional image editing tools. It's a powerful solution for artists, content creators, and anyone interested in transforming their images based on textual descriptions.

<img width="1279" alt="image" src="https://github.com/branyang02/GLOMA/assets/107154811/3564dc6d-6733-445d-9143-ef914ce04642">

## Table of Contents
- [Features](#features)
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)


## Features

- **Natural Language Processing:** GLOMA can understand and interpret textual prompts provided by users for image manipulation.
- **Advanced Image Editing:** Utilizes cutting-edge Diffusion models for high-quality image manipulations.
- **User-friendly Interface:** Provides an intuitive interface that allows users to describe their image editing needs in natural language.
- **Versatile Applications:** Suitable for a wide range of image editing tasks, from simple object movements to more complex scene generation.

Explore the following sections to learn more about how to use GLOMA and its various features.


## Overview

This repository contains the code and resources for the GLOMA project. The project involves multiple submodules and relies on specific Python packages. To ensure reproducibility and consistency, we use a Conda environment that specifies all the required dependencies.

## Installation
First request access to the LLama 2 models from: https://huggingface.co/meta-llama/Llama-2-13b-chat-hf if installing from scratch

```bash
### There is a working implementation in /project/CollabRoboGroup/GLOMA/GLOMA!

### Setup from scratch

git clone https://github.com/branyang02/GLOMA
cd GLOMA

# Configure Environment Variables
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-[version]

# Env Setup
conda env create -f environment.yml && conda activate GLOMA

# Module Installation
pip install -e .

# Download Checkpoint
mkdir checkpoints && cd checkpoints
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1Nv51TpPnHPwR7c5YdD-wCvXekEF2kQp5
unzip big-lama.zip

# llama-finetuning
# still 'in checkpoints'
gdown https://drive.google.com/uc?id=1xE7QNcnp4_4msnOsXNnCJp_WiWQAi7d6
unzip checkpoint-17120.zip
# in checkpoint-17120/adapter_config.json, change 'base_model_name_or_path' to /project/CollabRoboGroup/llama-2-13b-chat-hf
# or wherever you have the llama model

cd ..
conda env update --file environment_llama.yml --prune

## Usage

Navigate to GLOMA/gloma and run the following example

```bash
export TRANSFORMERS_CACHE=/project/CollabRoboGroup/llama-2-13b-chat-hf # or wherever you store the llama models
conda activate GLOMA # deactivate base if not already done so
# If in project:
conda activate ./GLOMA_ENV

python run_gloma.py --action_prompt="put the red cube to the right of the blue cube" --image_path="assets/1.jpg" --llm llama --cuda
```
## Arguments:
- `--action_prompt`: Textual action for image manipulation. Default: "stack the blue cube on top of the red cube".
- `--box_threshold`: Confidence level for bounding box detection. Default: 0.3.
- `--text_threshold`: Confidence level for text detection. Default: 0.25.
- `--nms_threshold`: Overlap threshold for merging bounding boxes. Default: 0.2.
- `--llm`:  Language model choice: "chatgpt" or "llama". Default: "chatgpt".
- `--image_path`: Path to the input image. Required.
- `--debug_mode`:  Toggle debug mode. This will save the intermediate masks and images that are processed in folder "debug_images". Default: False.
- `--image_size`:  Resolution of input image (in pixels). Images are processed as squared images. Default: 512.
- `--dilution_factor`: Dilution Factor. Default: 15.
- `--starting_noise`: Starting noise type. Choices: random, None. Default: None.
- `--guidance_scale`:  Adherence strength to textual guidance. Default: 7.5.

## Project Structure
```bash
GLOMA/
│
├── checkpoints/
│   ├── big-lama/
│   ├── checkpoint_inpainting_text_image.pth
│   ├── groundingdino_swint_ogc.pth
│   └── sam_vit_h_4b8939.pth
│
├── gloma/
│   ├── assets/
│   ├── debug_images
│   ├── generation_samples/
│   ├── lama/
│   ├── LLM/
│   ├── gligen_inference.py
│   ├── gloma.py
│   ├── lama_inpaint.py
│   ├── object_removal.py
│   ├── run_gloma.py
│   ├── SAM_detection.py
│   └── utils/
│
├── submodules/
│   ├── GLIGEN/
│   └── Grounded-Segment-Anything/
│
├── environment.yml
├── install_submodules.sh
├── README.md
├── requirements.txt
└── setup.py
```


## Contributing

If you'd like to contribute to this project, please create a new branch, make your changes, and submit a pull request. Ensure that your code follows the project's coding style and that all tests pass.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
