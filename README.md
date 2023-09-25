# GLOMA - Grounded Location for Object Manipulation

GLOMA is an AI-powered image editing tool that leverages state-of-the-art Large Language Models (LLM) and Diffusion models for seamless image manipulations based on user-provided text prompts. As an advanced image editing tool, GLOMA combines natural language processing and image processing to execute complex image manipulations described in textual instructions.

The key strength of GLOMA lies in its ability to interpret and act on text prompts that describe actions to be performed on an image. Users can provide instructions such as "move the cup behind the drawer," and GLOMA can intelligently execute such commands, resulting in a manipulated image that reflects the requested changes. This unique approach enables a wide range of applications, from creative image editing to more advanced use cases like scene generation and object relocation.

GLOMA's intuitive interface allows users to communicate their image editing needs in natural language, removing the barriers associated with traditional image editing tools. It's a powerful solution for artists, content creators, and anyone interested in transforming their images based on textual descriptions.

<img width="1279" alt="image" src="https://github.com/branyang02/GLOMA/assets/107154811/3564dc6d-6733-445d-9143-ef914ce04642">


## Features

- **Natural Language Processing:** GLOMA can understand and interpret textual prompts provided by users for image manipulation.
- **Advanced Image Editing:** Utilizes cutting-edge Diffusion models for high-quality image manipulations.
- **User-friendly Interface:** Provides an intuitive interface that allows users to describe their image editing needs in natural language.
- **Versatile Applications:** Suitable for a wide range of image editing tasks, from simple object movements to more complex scene generation.

Explore the following sections to learn more about how to use GLOMA and its various features.


## Overview

This repository contains the code and resources for the GLOMA project. The project involves multiple submodules and relies on specific Python packages. To ensure reproducibility and consistency, we use a Conda environment that specifies all the required dependencies.

## Installation

1. **Clone the repository:** 
```bash
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
pip install wldhx.yadisk-direct
curl -L $(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) -o big-lama.zip
unzip big-lama.zip
wget https://huggingface.co/gligen/gligen-inpainting-text-image-box/resolve/main/diffusion_pytorch_model.bin
mv diffusion_pytorch_model.bin checkpoint_inpainting_text_image.pth
```

## Usage

Example usage:

```bash
python run_gloma.py --image_path assets/2.jpg --action_prompt "put the green cube on top of the yellow cube and in front of the blue cube"
```
### The following command-line arguments are available:
* --action_prompt: Action prompt for GroundedSAM. This is a text * description of the action you want the GroundedSAM to perform. Default: "stack the blue cube on top of the red cube"
* --box_threshold: Box Threshold. This is the confidence threshold for bounding box detection. Default: 0.3
* --text_threshold: Text Threshold. This is the confidence threshold for text detection. Default: 0.25
* --nms_threshold: NMS Threshold. This is the non-maximum suppression threshold for bounding box overlap. Default: 0.2
* --llm: Choose either ChatGPT or Llama for the language model. Options: chatgpt, llama. Default: chatgpt
* --image_path: Input Image. The path to the image file on which the GroundedSAM will perform the action. This argument is required.


## Project Structure
```bash
GLOMA/
│
├── checkpoints/
│    ├── checkpoint_inpainting_text_image.pth
│    ├── groundingdino_swint_ogc.pth
│    ├── GroundingDINO_SwinT_OGC.py
│    └── sam_vit_h_4b8939.pth
│
├── submodules/
│    ├── gligen/
│    ├── grounded_sam/
│    └── lama_cleaner/
│
├── gloma/
     ├── const/
     ├── examples/
     ├── generation_samples/
     ├── LLM/
     ├── detection.py
     ├── gloma_class.py
     ├── object_removal.py
     └── run_gloma.py
```


## Contributing

If you'd like to contribute to this project, please create a new branch, make your changes, and submit a pull request. Ensure that your code follows the project's coding style and that all tests pass.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
