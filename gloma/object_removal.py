from typing import Dict, List, Tuple

import cv2
import numpy as np
from lama_inpaint import inpaint_img_with_lama
from utils import helper


class ObjectRemoval:

    def __init__(self, image, detections, class_prompt):
        self.image = image
        self.detections = detections
        self.class_ids = detections.class_id  # array([2, 0, 1])
        self.class_prompt = class_prompt  # ['green cube', 'yellow cube', 'blue cube']
        self.masks = self._create_masks()
        self.bboxes = self._create_bboxes()

    def _create_bboxes(self):
        """
        Create bounding boxes for all objects in the image.
        
        Format:
        {
            "obj_of_motion": {
                object_name: [x1, y1, x2, y2]
            },
            "objs_of_reference": 
            {
                object_name: [x1, y1, x2, y2],
                object_name: [x1, y1, x2, y2],
                ...
            }
        }
        """
        bboxes = {
            "obj_of_motion": {},
            "objs_of_reference": {}
        }
        for i, bbox in enumerate(self.detections.xyxy):
            if self.detections.class_id[i] == 0:
                self.obj_of_motion = self.class_prompt[self.detections.class_id[i]]
                bboxes["obj_of_motion"][self.class_prompt[self.detections.class_id[i]]] = bbox
            else:
                bboxes["objs_of_reference"][self.class_prompt[self.detections.class_id[i]]] = bbox
        return bboxes

    def _create_masks(self):
        """
        Create masks for all objects in the image.
        
        Format:
        {
            "obj_of_motion": {
                object_name: mask
            },
            "objs_of_reference": 
            {
                object_name: mask,
                object_name: mask,
                ...
            }
        }
        """
        masks = {
            "obj_of_motion": {},
            "objs_of_reference": {}
        }
        for i, mask in enumerate(self.detections.mask):
            if self.detections.class_id[i] == 0:
                masks["obj_of_motion"][self.class_prompt[self.detections.class_id[i]]] = self._dilate_mask(mask)
            else:
                masks["objs_of_reference"][self.class_prompt[self.detections.class_id[i]]] = self._dilate_mask(mask)
        return masks
    
    def _dilate_mask(self, mask, dilate_factor=30):
        mask = mask.astype(np.uint8)
        mask = cv2.dilate(
            mask,
            np.ones((dilate_factor, dilate_factor), np.uint8),
            iterations=1
        )
        return mask

    def get_obj_of_motion_mask(self) -> np.ndarray:
        """
        Returns
        - numpy.ndarray: Binary mask of the object of motion.
        """
        return self.masks["obj_of_motion"][self.obj_of_motion]
    
    def get_obj_of_motion_bbox(self) -> Dict:
        """
        Returns
        -   {
                object_name: [x1, y1, x2, y2]
            }
        """
        return self.bboxes["obj_of_motion"]
    
    def get_objs_of_reference_masks(self) -> Dict:
        """
        Returns
        -   {
                object_name: mask,
                object_name: mask,
                ...
            }
        """
        return self.masks["objs_of_reference"]
    
    def get_objs_of_reference_bboxes(self) -> Dict:
        """
        Returns
        -   {
                object_name: [x1, y1, x2, y2],
                object_name: [x1, y1, x2, y2],
                ...
            }
        """
        return self.bboxes["objs_of_reference"]
    
    def get_image_of_mask(self, image_cpy, mask) -> np.ndarray:
        """
        Given an image and a binary mask, extracts the object from the image within the mask.

        Args:
        - image (numpy.ndarray): Original image.
        - mask (numpy.ndarray): Binary mask with the same dimensions as the image.

        Returns:
        - numpy.ndarray: Cropped image of the object within the mask.
        """

        image = image_cpy.copy()
        assert np.isin(mask, [0, 1]).all(), "Mask must be binary"
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in mask")
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped_image = image[y:y+h, x:x+w]

        return cropped_image
    
    def inpaint_image(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:

        """
        Inpaints an image using the LAMA method.
        
        Given an input image and a corresponding mask, this function uses the LAMA 
        method to inpaint the masked region of the image.

        Parameters:
        - image (np.ndarray): The input image to be inpainted. The image should be 
                            in the form of a numpy ndarray.
        - mask (np.ndarray): A mask with the same dimensions as the input image. 
                            The mask indicates the region to be inpainted where 
                            non-zero values correspond to the region to be inpainted.

        Returns:
        - np.ndarray: The inpainted image.

        Note:
        - Ensure the provided configurations and checkpoint paths are accessible.
        - This function assumes that the device to use is a CUDA-enabled GPU.

        Examples:
        inpainted_image = obj.inpaint_image(image, mask)

        """

        return inpaint_img_with_lama(
            img=image,
            mask=mask,
            config_p="./lama/configs/prediction/default.yaml",
            ckpt_p="../checkpoints/big-lama",
            device="cuda",
        )