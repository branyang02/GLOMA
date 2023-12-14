import os
import pickle


import cv2

from gloma import GLOMA

DATASET_DIR = '/scratch/jqm9ba/rl_bench_dataset/generated_dataset'
OUTPUT_DIR = '/scratch/jqm9ba/eval_gloma_dataset'

GLOMA_OBJ = GLOMA(
    # action_prompt=prompt,
    box_threshold=0.3,
    text_threshold=0.25,
    nms_threshold=0.2,
    llm_choice='llama',
    # rgb_image=cv2.imread(os.path.join(view_dir, first_image)),
    debug_mode=True,
    dilution_factor=15,
    starting_noise=None,
    guidance_scale=7.5
)

def main():
    # Loop through each task
    for task in os.listdir(DATASET_DIR):
        task_dir = os.path.join(DATASET_DIR, task)
        
        # Loop through each variation
        for variation in os.listdir(task_dir):
            output_dir = os.path.join(OUTPUT_DIR, task, variation)
            os.makedirs(output_dir, exist_ok=True)
            
            variation_dir = os.path.join(task_dir, variation)
            
            # Check if the pickle file exists
            pickle_file = os.path.join(variation_dir, 'variation_descriptions.pkl')
            if os.path.isfile(pickle_file):
                # Open and print the content of the pickle file
                with open(pickle_file, 'rb') as f:
                    prompt_list = pickle.load(f)
                    # randomly choose a prompt
                    prompt = prompt_list[0]
                    
            episodes_dir = os.path.join(variation_dir, 'episodes')
            if os.path.isdir(episodes_dir):
                episode_dir = os.path.join(episodes_dir, 'episode0')
                if os.path.isdir(episode_dir):
                    # for camera_view in ['front_rgb', 'wrist_rgb']:
                    for camera_view in ['front_rgb']:
                        view_dir = os.path.join(episode_dir, camera_view)
                        # Inside the camera_view loop
                        if os.path.isdir(view_dir):
                            if os.path.isfile(os.path.join(output_dir, 'predicted.png')):
                                continue
                            # place a dummy file to indicate that this view has been processed
                            with open(os.path.join(output_dir, 'predicted.png'), 'w') as f:
                                f.write('dummy')
                            # Get a list of all images, convert to integers and sort them
                            images = sorted(os.listdir(view_dir), key=lambda img: int(img.split('.')[0]))
                            
                            # Get the first and last image
                            first_image = images[0]
                            last_image = images[-1]
                            
                            
                            print(f"Task: {task}, Variation: {variation}")
                            print(f"First image in {camera_view}: {first_image}")
                            print(f"Last image in {camera_view}: {last_image}")
                            print(f"Your prompt is: {prompt}")
                            
                            try:
                                result_images = GLOMA_OBJ.run_gloma(
                                    rgb_image=cv2.imread(os.path.join(view_dir, first_image)),
                                    action_prompt=prompt
                                )
                            except Exception as e:
                                print(f"Error: {e}")
                                with open(os.path.join(output_dir, 'error.txt'), 'w') as f:
                                    f.write(str(e))
                                continue
                            
                            cv2.imwrite(os.path.join(output_dir, 'input.png'), cv2.imread(os.path.join(view_dir, first_image)))
                            cv2.imwrite(os.path.join(output_dir, 'predicted.png'), result_images[0])
                            cv2.imwrite(os.path.join(output_dir, 'expected.png'), cv2.imread(os.path.join(view_dir, last_image)))

                            with open(os.path.join(output_dir, 'prompt.txt'), 'w') as f:
                                f.write(prompt)
                            # exit()
if __name__ == '__main__':
    main()