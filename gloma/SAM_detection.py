import os

import cv2
import numpy as np
import supervision as sv
import torch
import torchvision
from groundingdino.util.inference import Model
from segment_anything import SamPredictor, sam_model_registry

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
# GROUNDING_DINO_CONFIG_PATH = "../submodules/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# GROUNDING_DINO_CHECKPOINT_PATH = "../checkpoints/groundingdino_swint_ogc.pth"
GROUNDING_DINO_CONFIG_PATH = "/scratch/rhm4nj/GLOMA/GLOMA/submodules/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "/scratch/rhm4nj/GLOMA/GLOMA/checkpoints/groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
# SAM_CHECKPOINT_PATH = "../checkpoints/sam_vit_h_4b8939.pth"
SAM_CHECKPOINT_PATH = "/scratch/rhm4nj/GLOMA/GLOMA/checkpoints/sam_vit_h_4b8939.pth"

paths_to_check = [GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH, SAM_CHECKPOINT_PATH]
for path in paths_to_check:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File '{path}' does not exist.")

# Building GroundingDINO inference model
grounding_dino_model = Model(
    model_config_path=GROUNDING_DINO_CONFIG_PATH,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH
)
grounding_dino_model.dtype = torch.float16

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
SAM_PREDICTOR = SamPredictor(sam)


class GroundedSAM:
    def __init__(
            self,
            source_image,
            class_prompt,
            box_threshold=0.25,
            text_threshold=0.25,
            nms_threshold=0.8
    ):
        self.image = source_image
        self.class_prompt = class_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold

        # Detect objects
        print("SAM is searching for ", self.class_prompt)
        self.detections = grounding_dino_model.predict_with_classes(
            image=self.image,
            classes=self.class_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )
        self._NMS_post_process()


    def _NMS_post_process(self):
        print(f"Before NMS: {len(self.detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(self.detections.xyxy), 
            torch.from_numpy(self.detections.confidence), 
            self.nms_threshold
        ).numpy().tolist()
        self.detections.xyxy = self.detections.xyxy[nms_idx]
        self.detections.confidence = self.detections.confidence[nms_idx]
        self.detections.class_id = self.detections.class_id[nms_idx]
        print(f"After NMS: {len(self.detections.xyxy)} boxes")


    def get_detections(self):
        # convert detections to masks
        self.detections.mask = []
        SAM_PREDICTOR.set_image(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        for box in self.detections.xyxy:
            masks, scores, logits = SAM_PREDICTOR.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            self.detections.mask.append(masks[index])
        self.detections.mask = np.array(self.detections.mask)

        return self.detections, self.class_prompt