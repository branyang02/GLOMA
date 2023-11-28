OBJECT_PROMPT = """
In this task, you are required to analyze sentences describing actions involving objects. Your goal is to identify two key elements in each sentence: 
1. The 'object of motion' - This is the primary item in the sentence that is being moved, manipulated, or acted upon.
2. The 'objects of reference' - These are objects or locations that provide context to the action, indicating where, towards, or between which the action is taking place.

The understanding of these elements is crucial for interpreting actions in a given context. Your response should be formatted as a JSON object, clearly distinguishing the 'object of motion' from the 'objects of reference'.

Please follow this JSON format for your response:

{{
    "object_of_motion": "<identified object of motion>",
    "objects_of_reference": ["<list of identified objects of reference>"]
}}

Do not give me anything else other than the JSON return.

Consider these examples to understand how to apply this analysis:

Q: "Place the book on the table and chair."
A: {{
    "object_of_motion": "book",
    "objects_of_reference": ["table", "chair"]
}}

Q: "Push the ball towards the goalpost."
A: {{
    "object_of_motion": "ball",
    "objects_of_reference": ["goalpost"]
}}

Q: "Hang the jacket between the door and window."
A: {{
    "object_of_motion": "jacket",
    "objects_of_reference": ["door", "window"]
}}

Q: "{action_prompt}"
A:
"""


BOUNDING_BOX_PROMPT = """
You are an intelligent bounding box generator based on an action prompt. I will provide you with bounding boxes of different objects in an image of size 512 x 512. The camera is facing the scene; therefore, when I say "put A in front of B", it means "put A closer to the camera than B". These objects can either be an 'Object of motion' or 'Objects of reference'. I will also provide an action prompt that describes the transformation of the image. An action prompt could be "stack the blue cube on top of the red cube". In this case, you should deduce what the final image should look like based on the action prompt that I provide, and give me the corresponding bounding box of the object of motion. The format of the bounding box is described below:

If I mention to move one block on top of another (or stack or any similar behavior), make sure the bottom edges are aligned when calculating the new bounding box. Note that bottom edges are not necessarily the bounding box edges, but the edges related to the shape of the objects. If I mention moving the block to the left or to the right, make sure to leave a tiny gap between the two blocks. Note that the new bounding box should have the same size as the bounding box of the object of motion.

Please return a JSON object with the following format:

{{
    "predicted_bbox": []
}}

Examples:

Q: Predict the following transformation:
Action Prompt: "Move the green sphere to the right of the yellow square."
Object of motion: [100, 100, 150, 150]
Objects of reference: [200, 100, 250, 150]
A:
{{
    "predicted_bbox": [260, 100, 310, 150]
}}

Q: Predict the following transformation:
Action Prompt: "Place the small red triangle on top of the large blue rectangle."
Object of motion: [50, 50, 100, 100]
Objects of reference: [150, 200, 300, 250]
A:
{{
    "predicted_bbox": [175, 150, 225, 200]
}}

Q: Predict the following transformation:
Action Prompt: "Stack the orange circle on top of the purple hexagon."
Object of motion: [120, 120, 170, 170]
Objects of reference: [300, 250, 350, 300]
A:
{{
    "predicted_bbox": [325, 200, 375, 250]
}}

Q: Predict the following transformation:
Action Prompt: {action_prompt}
Object of motion: {obj_of_motion_box}
Objects of reference: {objs_of_reference_boxes}
A:
"""