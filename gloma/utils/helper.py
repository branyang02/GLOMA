import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image


def save_image(image, file_name):
    """
    Saves the image in the "experiments" folder.
    """
    # Ensure the "experiments" directory exists
    output_dir = "experiments"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the image
    output_path = os.path.join(output_dir, file_name)
    cv2.imwrite(output_path, image)

def draw_predicted_bbox(image_cpy: np.ndarray, bbox: List[float], name="bbox.jpg"):
    """
    Draw the predicted bounding box on the image and save it.
    """
    image = image_cpy.copy()
    
    # Extract coordinates and convert them to integers
    x1, y1, x2, y2 = map(int, bbox)
    
    color = (0, 255, 0)  # Green color
    thickness = 2
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if not os.path.exists("experiments"):
        os.makedirs("experiments")
    cv2.imwrite("experiments/{}".format(name), image)

def draw_bounding_boxes(image_cpy, detections, name="bbox.jpg"):
    """
    Show the bounding boxes on the image.
    """
    image = image_cpy.copy()
    for bbox in detections.xyxy:
        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 255, 0) 
        thickness = 2    
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if not os.path.exists("experiments"):
        os.makedirs("experiments")
    cv2.imwrite("experiments/{}".format(name), image)

def apply_mask(image, mask):
    """Applies the mask to the image to highlight the segmentation."""
    # Create an RGB version of the mask
    mask_rgb = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    # Return the image where the mask is applied and pure white where it isn't
    return np.where(mask_rgb == 1, image, [255, 255, 255])

def load_img_to_array(img_p):
    img = Image.open(img_p)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)

def save_array_to_img(img_arr, img_p):
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)


def draw_masks(image_cpy, masks):
    """
    Visualizes all segmentations together on the same image and saves it in the "experiments" folder.

    Args:
    - image: The original image.
    - masks: List of binary masks, one for each detection.
    """
    image = image_cpy.copy()
    combined_mask = np.sum(masks, axis=0)
    combined_mask[combined_mask > 1] = 1  # Ensure binary mask (values are either 0 or 1)

    masked_image = apply_mask(image, combined_mask)

    # Ensure the "experiments" directory exists
    output_dir = "experiments"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the combined segmentation image
    output_path = os.path.join(output_dir, "objs_of_reference_masks.jpg")
    cv2.imwrite(output_path, masked_image)
    

def parse_input(obj_separation):
    """
    Parse the object of motion and objects of reference from a given JSON-like string format.
    
    Args:
    - obj_separation (str): A formatted string containing object details in JSON format.
    
    Returns:
    - tuple: A tuple containing the object of motion as a string, and objects of reference as a list of strings.
    """
    
    json_str = obj_separation.strip()
    data = json.loads(json_str)
    obj_of_motion = data["object_of_motion"]
    obj_of_reference = data["objects_of_reference"]

    return obj_of_motion, obj_of_reference

def parse_bbox(bbox_str: str) -> List[float]:
    """
    Parse the bounding box coordinates from a given JSON-like string format.
    
    Args:
    - bbox_str (str): A formatted string containing bounding box coordinates in JSON format.
    
    Returns:
    - list: A list containing the bounding box coordinates.
    """
    
    json_str = bbox_str.strip()
    data = json.loads(json_str)
    bbox = data["predicted_bbox"]
    return bbox

def save_image_with_incremental_number(filename_prefix, image):
    # Check if the filename already exists
    counter = 1
    original_filename = filename_prefix + ".jpg"
    filename = original_filename
    while os.path.isfile(filename):
        # Increment the counter and modify the filename
        counter += 1
        filename = f"{filename_prefix}{counter}.jpg"

    # Resize the image to 512 x 512
    resized_image = cv2.resize(image, (512, 512))

    # Save the resized image with the modified filename
    cv2.imwrite(filename, resized_image)
    print(f"Resized image saved as {filename}")

def convert_to_relative_bbox(locations, image_width, image_height):
    print("Image Width: ", image_width)
    print("Image Height: ", image_height)
    x_min, y_min, x_max, y_max = locations

    # Calculate relative coordinates
    relative_x_min = x_min / image_width
    relative_y_min = y_min / image_height
    relative_x_max = x_max / image_width
    relative_y_max = y_max / image_height

    # Return the relative bounding box
    return [relative_x_min, relative_y_min, relative_x_max, relative_y_max]