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
    # Get image
    rgb_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # resize image to 512 x 512
    rgb_image = cv2.resize(rgb_image, (512, 512))

    gloma = GLOMA(action_prompt, box_threshold, text_threshold, nms_threshold, llm_choice, rgb_image)
    result_image = gloma.run_gloma()

main()