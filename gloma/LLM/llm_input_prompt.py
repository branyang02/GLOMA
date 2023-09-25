OBJECT_PROMPT = """
Identify the 'object of motion' and the 'objects of reference' based on the action described. The 'object of motion' is the singular item being moved or acted upon, while the 'objects of reference' can be one or more objects or locations in relation to which the motion is being described.

Examples:

"place the book on the table and chair" should yield:
{{"object_of_motion": "book", "objects_of_reference": ["table", "chair"]}}

"push the ball towards the goalpost" would be:
{{"object_of_motion": "ball", "objects_of_reference": ["goalpost"]}}

"hang the jacket between the door and window" translates to:
{{"object_of_motion": "jacket", "objects_of_reference": ["door", "window"]}}

Please return a JSON object with the following format:

{{
    "object_of_motion": "jacket",
    "objects_of_reference": ["door", "window"]
}}

Q: What are the object of motion and objects of reference in the sentence "{action_prompt}"?
A:
"""


BOUNDING_BOX_PROMPT = """
You are an intelligent bounding box generator based on an action prompt. I will provide \
you with bounding boxes of different objects in an image of size 512 x 512. The camera is facing the scene; \
therefore, when I say "put A in front of B", it means "put A closer to the camera than B". These objects 
can either be an 'Object of motion' or 'Objects of reference'. I will also \
provide an action prompt that describes the transformation of the image. An action prompt could be \
"stack the blue cube on top of the red cube". In this case, you should deduce what the final \
image should look like based on the action prompt that I provide, and give me the corresponding \
bounding box of the object of motion. The format of the bounding box is described below:

If I mention to move one block on top of another (or stack or any similar behavior), make sure the bottom edges \
are aligned when calculating the new bounding box. Note that bottom edges are not necessarilly the bounding box edges, but the edges related to the shape of the objects. If I mention move the block to the left or to the right \
make sure leave a tiny gap between the two blocks. Note that the new bounding box should have the same size as the bounding box \
of the object of motion.

Please return a JSON object with the following format:

{{
    "predicted_bbox": []
}}


Q: Predict the following transformation:
Action Prompt: {action_prompt}
Object of motion: {obj_of_motion_box}
Objects of reference: {objs_of_reference_boxes}
A:

Output: 
"""


# BOUNDING_BOX_PROMPT = """
# You are an intelligent bounding box generator based on an action prompt. I will provide \
# you with two bounding boxes of two different objects in an image of size 2048 x 1526. The two \
# different objects are either an 'Object of motion' or an 'Object of reference'. I will also \
# provide an action prompt that describes the transformation of the image. An action prompt could be \
# "stack the blue cube on top of the red cube". In this case, you should learn what the final \
# image should look like based on the action prompt that I provide, and give me the corresponding \
# bounding box of the object of motion. The format of the bounding box is described below:


# Object of motion: (object name: [top-left x coordinate, top-left y coordinate, box width, box height])
# Object of reference: (object name: [top-left x coordinate, top-left y coordinate, box width, box height])

# If I mention to move one block on top of another (or stack or any similar behavior), make sure the bottom edges \
# are aligned when calculating the new bounding box. If I mention move the block to the left or to the right \
# make sure leave a tiny gap between the two blocks. 

# DO NOT write me anything else other than just the bounding box.

# Below is a complete example of the input:
# Action prompt: move the red cube behind the blue cube
# Object of motion: (red cube: [839.4738, 417.7555, 962.9438, 568.1781])
# Object of reference: (blue cube: [1266.4738, 259.968, 1385.3822, 400.89212])
# Output: [1258.0583, 175.13162, 1366.0491, 302.87888]

# Action prompt: move the blue cube behind the red cube
# Object of motion: (blue cube: [1058.583, 317.43393, 1158.7446, 458.84732])
# Object of reference: (red cube: [787.00653, 427.04688, 919.6506, 575.8333])
# Output: [817.57306, 337.97363, 939.5981, 478.3122 ]

# Action Prompt: stack the blue cube on top of the red cube.
# Object of motion: (blue cube: [1291.6324, 598.7196, 1458.1327, 772.59326])
# Object of reference: (red cube: [839.1231, 411.08096, 958.4046, 561.6876])
# Output: [787.2792, 249.24765, 940.1393, 423.95773]

# Now, complete the following. Make sure your the size of the newly generated bounding box is as small as appropiate. Remember that the image has size 2048 x 1536.

# Action Prompt: {action_prompt}
# Object of motion: ({obj_of_motion}: {obj_of_motion_box})
# Object of reference: ({obj_of_reference}: {obj_of_reference_box})
# Output: 
# """