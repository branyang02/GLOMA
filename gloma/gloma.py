import sys
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import supervision as sv
from SAM_detection import GroundedSAM
from LLM.llm_factory import LLMFactory
from LLM.llm_input_prompt import BOUNDING_BOX_PROMPT, OBJECT_PROMPT
from utils import helper

# # Get the absolute path of the script's directory
# script_dir = Path(__file__).resolve().parent

# # Go up one level to the 'GLOMA' directory
# base_dir = script_dir.parent

# # Construct the path to 'submodules/gligen'
# gligen_path = base_dir / 'submodules' / 'gligen'

# # Add the path to the system path
# sys.path.insert(0, str(gligen_path))


from object_removal import ObjectRemoval
# from run_gligen import run_model




class GLOMA:

    def __init__(
            self,
            action_prompt,
            box_threshold,
            text_threshold,
            nms_threshold,
            llm_choice,
            rgb_image,
            debug_mode=False
    ):
        self.action_prompt = action_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        self.llm_choice = llm_choice
        self.rgb_image = rgb_image
        self.debug_mode = debug_mode
    

    def get_object_names(self) -> Tuple[str, List[str]]:
        """
        Returns:
        - obj_of_motion: object of motion
        - obj_of_reference: [object of references]
        """
        llm_object = LLMFactory.create_chat_object(self.llm_choice)
        obj_separation = llm_object.query_message(OBJECT_PROMPT.format(action_prompt=self.action_prompt))
        obj_of_motion, obj_of_reference = helper.parse_input(obj_separation)
        return obj_of_motion, obj_of_reference

    def grounded_sam_detections(self, class_prompt: List[str]) -> sv.Detections:
        """
        Get bbox, masks from SAM

        Returns:
        - detections: Detection object
        """
        grounded_sam = GroundedSAM(self.rgb_image, class_prompt, self.box_threshold, self.text_threshold, self.nms_threshold)
        detections, class_prompt = grounded_sam.get_detections()
        return detections, class_prompt
    
    def remove_object(
        self,
        rgb_image: np.ndarray,
        detections: sv.Detections,
        class_prompt: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:

        """
        Inpaint image by removing object of motion and extracts bounding boxes and related images for the specified object of 
        motion and objects of reference from the given RGB image based on the detections and class prompts.

        The process involves:
        1. Visualizing objects of reference.
        2. Extracting a cropped image of the object of motion.
        3. Removing the object of motion from the original image.
        4. Extracting bounding boxes for both the object of motion and the objects of reference.

        Parameters:
        - rgb_image (np.ndarray): The input RGB image.
        - detections (sv.Detections): The detection data for objects in the image.
        - class_prompt (List[str]): Class prompt information guiding the detections.

        Returns:
        - tuple: 
            - inpainted_image (np.ndarray): Image with the object of motion removed.
            - obj_of_motion_image (np.ndarray): Cropped image of the object of motion.
            - obj_of_motion_bbox (dict): Bounding box for the object of motion.
                - ex. {
                        object_name: [x_min, y_min, x_max, y_max]
                    }
            - objs_of_reference_bbox (dict): Bounding boxes for the objects of reference.
                - ex. {
                        "object_name": array([x_min, y_min, x_max, y_max]),
                        "object_name_2": array([x_min, y_min, x_max, y_max]),
                    }

        Raises:
        - AssertionError: If any of the outputs don't match the expected types.

        Note:
        Debugging images, such as "obj_of_motion_image.jpg" and "inpainted_image.jpg", may be saved during the process for visualization.
        """

        # 1. create ObjectRemoval object
        remover = ObjectRemoval(rgb_image, detections, class_prompt)
        if self.debug_mode:
            # DEBUG: visualize objs_of_reference
            helper.draw_masks(rgb_image, list(remover.masks["objs_of_reference"].values()))

        # 2. extract object of motion image (cropped)
        obj_of_motion_mask = remover.get_obj_of_motion_mask()
        obj_of_motion_image = remover.get_image_of_mask(rgb_image, obj_of_motion_mask)
        if self.debug_mode:
            # DEBUG: visualize obj_of_action
            helper.save_image(obj_of_motion_image, "obj_of_motion_image.jpg")

        # 3. remove object of motion from image
        inpainted_image = remover.inpaint_image(
            image=rgb_image,
            mask=obj_of_motion_mask,
        )

        if self.debug_mode:
            # DEBUG: visualize inpainted_image
            helper.save_image(inpainted_image, "inpainted_image.jpg")

        # 4. extract bounding boxes
        obj_of_motion_bbox = remover.get_obj_of_motion_bbox()
        objs_of_reference_bbox = remover.get_objs_of_reference_bboxes()

        return inpainted_image, obj_of_motion_image, obj_of_motion_bbox, objs_of_reference_bbox
    

    def predict_new_bbox(self, obj_of_motion_bbox: np.ndarray, objs_of_reference_bbox: Dict) -> List[float]:
        # convert bbox values from float to int
        obj_of_motion_bbox = {key: [int(val) for val in value] for key, value in obj_of_motion_bbox.items()}
        objs_of_reference_bbox = {key: [int(val) for val in value] for key, value in objs_of_reference_bbox.items()}

        llm_object = LLMFactory.create_chat_object(self.llm_choice)
        predicted_bbox = llm_object.query_message(BOUNDING_BOX_PROMPT.format(
            action_prompt=self.action_prompt,
            obj_of_motion_box=obj_of_motion_bbox,
            objs_of_reference_boxes=objs_of_reference_bbox
        ))
        return helper.parse_bbox(predicted_bbox)
        

    def run_gloma(self):
        print("🚀🚀🚀 GLOMA IS RUNNING! 🚀🚀🚀")
        
        #1. Get Objects
        obj_of_motion, objs_of_reference = self.get_object_names()
        print("object of motion: ", obj_of_motion)
        print("objects of reference: ", objs_of_reference)

        # 2. SAM
        detections, class_prompt = self.grounded_sam_detections(class_prompt=[obj_of_motion] + objs_of_reference)
        if self.debug_mode:
            # DEBUG: visualize detected bouding boxes 
            helper.draw_bounding_boxes(self.rgb_image, detections, name="detected_bounding_boxes.jpg")

        # 3. Object Removal
        inpainted_image, obj_of_motion_image, obj_of_motion_bbox, objs_of_reference_bbox = self.remove_object(self.rgb_image, detections, class_prompt)
        print("object of motion BBOX: ", obj_of_motion_bbox)
        print("object of reference BBOX: ", objs_of_reference_bbox)

        # 4. Predict new bbox
        predicted_bbox = self.predict_new_bbox(obj_of_motion_bbox, objs_of_reference_bbox)
        print("predicted bbox: ", predicted_bbox)
        if self.debug_mode:
            # DEBUG: visualize predicted bbox
            helper.draw_predicted_bbox(inpainted_image, predicted_bbox, "predicted_bbox.jpg")

        return None

        # 5. Generate new image (GLIGEN)















        # save_image_with_incremental_number("original", self.rgb_image)
        cv2.imwrite("removed.jpg", removed_image)
        cv2.imwrite("object.jpg", obj_image)
        new_bounding_box = self.get_new_bounding_box(obj_of_motion, obj_of_reference, obj_of_motion_box, obj_of_reference_box)

        print(new_bounding_box)

        run_model(
            input_image = "removed.jpg",
            prompt = self.action_prompt,
            images = ['object.jpg'],
            locations = [new_bounding_box],
        )
        return None