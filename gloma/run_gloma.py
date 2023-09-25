from gloma import GLOMA
import argparse
import cv2
import os
from utils import helper

def main():
    parser = argparse.ArgumentParser(description='GLOMA - Grounded Location for Object Manipulation Agent')
    parser.add_argument('--action_prompt', help='action prompt for GroundedSAM', default="stack the blue cube on top of the red cube")
    parser.add_argument('--box_threshold', help='Box Threshold', default=0.3, type=float)
    parser.add_argument('--text_threshold', help='Text Threshold', default=0.25, type=float)
    parser.add_argument('--nms_threshold', help='NMS Threshold', default=0.2, type=float)
    parser.add_argument('--llm', help='Choose either ChatGPT or Llama', choices=['chatgpt', 'llama'], default='chatgpt')
    parser.add_argument('--image_path', help='Input Image', required=True)
    parser.add_argument('--debug_mode', help='Debug Mode', default=False, type=bool)
    parser.add_argument('--image_size', help='Image Size', default=512, type=int)
    parser.add_argument('--dilution_factor', help='Dilution Factor', default=30, type=int)
    args = parser.parse_args()

    action_prompt = args.action_prompt
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    nms_threshold = args.nms_threshold
    llm_choice = args.llm
    image_path = args.image_path

    # Check if image file exists
    if not os.path.exists(image_path):
        raise RuntimeError(f"Error: File '{image_path}' does not exist.")
    # Get image and resize
    rgb_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    rgb_image = cv2.resize(rgb_image, (args.image_size, args.image_size))

    gloma = GLOMA(
        action_prompt=action_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        nms_threshold=nms_threshold,
        llm_choice=llm_choice,
        rgb_image=rgb_image,
        debug_mode=args.debug_mode,
        dilution_factor=args.dilution_factor
    )
    result_image = gloma.run_gloma()

main()