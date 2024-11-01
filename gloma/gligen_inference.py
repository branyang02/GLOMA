import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submodules/GLIGEN"))

import argparse
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import os 
from transformers import CLIPProcessor, CLIPModel
from copy import deepcopy
import torch 
from ldm.util import instantiate_from_config
from trainer import read_official_ckpt, batch_to_device
from inpaint_mask_func import draw_masks_from_boxes
import numpy as np
import clip 
from scipy.io import loadmat
from functools import partial
import torchvision.transforms.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import cv2

device = "cuda"


def set_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale


def alpha_generator(length, type=None):
    """
    length is total timestpes needed for sampling. 
    type should be a list containing three values which sum should be 1
    
    It means the percentage of three stages: 
    alpha=1 stage 
    linear deacy stage 
    alpha=0 stage. 
    
    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.    
    """
    if type == None:
        type = [1,0,0]

    assert len(type)==3 
    assert type[0] + type[1] + type[2] == 1
    
    stage0_length = int(type[0]*length)
    stage1_length = int(type[1]*length)
    stage2_length = length - stage0_length - stage1_length
    
    if stage1_length != 0: 
        decay_alphas = np.arange(start=0, stop=1, step=1/stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []
        
    
    alphas = [1]*stage0_length + decay_alphas + [0]*stage2_length
    
    assert len(alphas) == length
    
    return alphas



# def load_ckpt(ckpt_path):
    
#     saved_ckpt = torch.load(ckpt_path)
#     config = saved_ckpt["config_dict"]["_content"]

#     model = instantiate_from_config(config['model']).to(device).eval()
#     autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
#     text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
#     diffusion = instantiate_from_config(config['diffusion']).to(device)

#     # donot need to load official_ckpt for self.model here, since we will load from our ckpt
#     model.load_state_dict( saved_ckpt['model'] )
#     autoencoder.load_state_dict( saved_ckpt["autoencoder"]  )

#     text_encoder.load_state_dict( saved_ckpt["text_encoder"]  )
#     diffusion.load_state_dict( saved_ckpt["diffusion"]  )

#     return model, autoencoder, text_encoder, diffusion, config

def load_ckpt(ckpt_path):
    saved_ckpt = torch.load(ckpt_path)
    config = saved_ckpt["config_dict"]["_content"]

    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # donot need to load official_ckpt for self.model here, since we will load from our ckpt
    model.load_state_dict(saved_ckpt['model'])
    autoencoder.load_state_dict(saved_ckpt["autoencoder"])

    # Filter out the unexpected keys for the text_encoder
    expected_keys = set(text_encoder.state_dict().keys())
    filtered_state_dict = {k: v for k, v in saved_ckpt["text_encoder"].items() if k in expected_keys}
    
    text_encoder.load_state_dict(filtered_state_dict)
    diffusion.load_state_dict(saved_ckpt["diffusion"])

    return model, autoencoder, text_encoder, diffusion, config




def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    this function will return the CLIP feature (without normalziation)
    """
    return x@torch.transpose(projection_matrix, 0, 1)


def get_clip_feature(model, processor, input, is_image=False):
    which_layer_text = 'before'
    which_layer_image = 'after_reproject'

    if is_image:
        # Image is already loaded.
        if input is None:
            return None

        # Check if input is a numpy array and convert to PIL Image if so.
        if isinstance(input, np.ndarray):
            # If it appears that the channels are BGR (based on visual inspection), swap channels to RGB.
            input = input[..., ::-1]  # Swap channels
            # Convert numpy image to PIL. Assuming the ndarray is in 'uint8' format and has values in [0, 255].
            image = Image.fromarray(input)
        else:
            # If it's already a PIL Image, just proceed.
            image = input

        image = image.convert("RGB")
        inputs = processor(images=[image], return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].cuda()  # we use our own preprocessing without center_crop 
        inputs['input_ids'] = torch.tensor([[0, 1, 2, 3]]).cuda()  # placeholder
        outputs = model(**inputs)
        feature = outputs.image_embeds 
        if which_layer_image == 'after_reproject':
            # print("Your current path is: ", os.getcwd())
            feature = project(feature, torch.load('../submodules/GLIGEN/projection_matrix').cuda().T).squeeze(0)
            feature = (feature / feature.norm()) * 28.7 
            feature = feature.unsqueeze(0)


    else:
        if input == None:
            return None
        inputs = processor(text=input,  return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['pixel_values'] = torch.ones(1,3,224,224).cuda() # placeholder 
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = model(**inputs)
        if which_layer_text == 'before':
            feature = outputs.text_model_output.pooler_output
    return feature


def complete_mask(has_mask, max_objs):
    mask = torch.ones(1,max_objs)
    if has_mask == None:
        return mask 

    if type(has_mask) == int or type(has_mask) == float:
        return mask * has_mask
    else:
        for idx, value in enumerate(has_mask):
            mask[0,idx] = value
        return mask



@torch.no_grad()
def prepare_batch(meta, batch=1, max_objs=30):
    phrases, images = meta.get("phrases"), meta.get("images")
    images = [None]*len(phrases) if images==None else images 
    phrases = [None]*len(images) if phrases==None else phrases 

    version = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(version).cuda()
    processor = CLIPProcessor.from_pretrained(version)

    boxes = torch.zeros(max_objs, 4)
    masks = torch.zeros(max_objs)
    text_masks = torch.zeros(max_objs)
    image_masks = torch.zeros(max_objs)
    text_embeddings = torch.zeros(max_objs, 768)
    image_embeddings = torch.zeros(max_objs, 768)
    
    text_features = []
    image_features = []
    for phrase, image in zip(phrases,images):
        text_features.append(  get_clip_feature(model, processor, phrase, is_image=False) )
        image_features.append( get_clip_feature(model, processor, image,  is_image=True) )

    for idx, (box, text_feature, image_feature) in enumerate(zip( meta['locations'], text_features, image_features)):
        boxes[idx] = torch.tensor(box)
        masks[idx] = 1
        if text_feature is not None:
            text_embeddings[idx] = text_feature
            text_masks[idx] = 1 
        if image_feature is not None:
            image_embeddings[idx] = image_feature
            image_masks[idx] = 1 

    out = {
        "boxes" : boxes.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
        "text_masks" : text_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("text_mask"), max_objs ),
        "image_masks" : image_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("image_mask"), max_objs ),
        "text_embeddings"  : text_embeddings.unsqueeze(0).repeat(batch,1,1),
        "image_embeddings" : image_embeddings.unsqueeze(0).repeat(batch,1,1)
    }

    return batch_to_device(out, device) 


def crop_and_resize(image):
    crop_size = min(image.size)
    image = TF.center_crop(image, crop_size)
    image = image.resize( (512, 512) )
    return image



@torch.no_grad()
def prepare_batch_kp(meta, batch=1, max_persons_per_image=8):
    
    points = torch.zeros(max_persons_per_image*17,2)
    idx = 0 
    for this_person_kp in meta["locations"]:
        for kp in this_person_kp:
            points[idx,0] = kp[0]
            points[idx,1] = kp[1]
            idx += 1
    
    # derive masks from points
    masks = (points.mean(dim=1)!=0) * 1 
    masks = masks.float()

    out = {
        "points" : points.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
    }

    return batch_to_device(out, device) 


@torch.no_grad()
def prepare_batch_hed(meta, batch=1):
    
    pil_to_tensor = transforms.PILToTensor()

    hed_edge = Image.open(meta['hed_image']).convert("RGB")
    hed_edge = crop_and_resize(hed_edge)
    hed_edge = ( pil_to_tensor(hed_edge).float()/255 - 0.5 ) / 0.5

    out = {
        "hed_edge" : hed_edge.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 


@torch.no_grad()
def prepare_batch_canny(meta, batch=1):
    """ 
    The canny edge is very sensitive since I set a fixed canny hyperparamters; 
    Try to use the same setting to get edge 

    img = cv.imread(args.image_path, cv.IMREAD_GRAYSCALE)
    edges = cv.Canny(img,100,200)
    edges = PIL.Image.fromarray(edges)

    """
    
    pil_to_tensor = transforms.PILToTensor()

    canny_edge = Image.open(meta['canny_image']).convert("RGB")
    canny_edge = crop_and_resize(canny_edge)

    canny_edge = ( pil_to_tensor(canny_edge).float()/255 - 0.5 ) / 0.5

    out = {
        "canny_edge" : canny_edge.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 


@torch.no_grad()
def prepare_batch_depth(meta, batch=1):
    
    pil_to_tensor = transforms.PILToTensor()

    depth = Image.open(meta['depth']).convert("RGB")
    depth = crop_and_resize(depth)
    depth = ( pil_to_tensor(depth).float()/255 - 0.5 ) / 0.5

    out = {
        "depth" : depth.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 



@torch.no_grad()
def prepare_batch_normal(meta, batch=1):
    """
    We only train normal model on the DIODE dataset which only has a few scene.

    """
    
    pil_to_tensor = transforms.PILToTensor()

    normal = Image.open(meta['normal']).convert("RGB")
    normal = crop_and_resize(normal)
    normal = ( pil_to_tensor(normal).float()/255 - 0.5 ) / 0.5

    out = {
        "normal" : normal.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 





def colorEncode(labelmap, colors):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)

    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    return labelmap_rgb

@torch.no_grad()
def prepare_batch_sem(meta, batch=1):

    pil_to_tensor = transforms.PILToTensor()

    sem = Image.open( meta['sem']  ).convert("L") # semantic class index 0,1,2,3,4 in uint8 representation 
    sem = TF.center_crop(sem, min(sem.size))
    sem = sem.resize( (512, 512), Image.NEAREST ) # acorrding to official, it is nearest by default, but I don't know why it can prodice new values if not specify explicitly
    try:
        sem_color = colorEncode(np.array(sem), loadmat('color150.mat')['colors'])
        Image.fromarray(sem_color).save("sem_vis.png")
    except:
        pass 
    sem = pil_to_tensor(sem)[0,:,:]
    input_label = torch.zeros(152, 512, 512)
    sem = input_label.scatter_(0, sem.long().unsqueeze(0), 1.0)

    out = {
        "sem" : sem.unsqueeze(0).repeat(batch,1,1,1),
        "mask" : torch.ones(batch,1),
    }
    return batch_to_device(out, device) 



@torch.no_grad()
def run(
    meta,
    folder,
    batch_size,
    no_plms,
    guidance_scale,
    negative_prompt,
    starting_noise,
    image_size,
    debug_mode
):
    # - - - - - prepare models - - - - - # 
    model, autoencoder, text_encoder, diffusion, config = load_ckpt(meta["ckpt"])

    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input
    
    grounding_downsampler_input = None
    if "grounding_downsampler_input" in config:
        grounding_downsampler_input = instantiate_from_config(config['grounding_downsampler_input'])



    # - - - - - update config from args - - - - - # 
    # config.update( vars(args) )
    # config = OmegaConf.create(config)


    # - - - - - prepare batch - - - - - #
    if "keypoint" in meta["ckpt"]:
        batch = prepare_batch_kp(meta, batch_size)
    elif "hed" in meta["ckpt"]:
        batch = prepare_batch_hed(meta, batch_size)
    elif "canny" in meta["ckpt"]:
        batch = prepare_batch_canny(meta, batch_size)
    elif "depth" in meta["ckpt"]:
        batch = prepare_batch_depth(meta, batch_size)
    elif "normal" in meta["ckpt"]:
        batch = prepare_batch_normal(meta, batch_size)
    elif "sem" in meta["ckpt"]:
        batch = prepare_batch_sem(meta, batch_size)
    else:
        batch = prepare_batch(meta, batch_size)
    context = text_encoder.encode(  [meta["prompt"]]*batch_size  )
    uc = text_encoder.encode( batch_size*[""] )
    if negative_prompt is not None:
        uc = text_encoder.encode( batch_size*[negative_prompt] )


    # - - - - - sampler - - - - - # 
    alpha_generator_func = partial(alpha_generator, type=meta.get("alpha_type"))
    if no_plms:
        sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 250 
    else:
        sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
        steps = 50 


    # - - - - - inpainting related - - - - - #
    inpainting_mask = z0 = None  # used for replacing known region in diffusion process
    inpainting_extra_input = None # used as model input 

    if "input_image" in meta:
        # inpaint mode 
        # assert config.inpaint_mode, 'input_image is given, the ckpt must be the inpaint model, are you using the correct ckpt?'
        
        inpainting_mask = draw_masks_from_boxes(batch['boxes'], model.image_size).cuda()
        
        # Check the type of meta["input_image"]
        input_data = meta["input_image"]
        if isinstance(input_data, np.ndarray):
            # If it appears that the channels are BGR (based on visual inspection), swap channels to RGB.
            input_data = input_data[..., ::-1]  # Swap channels
            # Convert the numpy array to a PIL Image.
            input_image_pil = Image.fromarray(input_data)
        elif isinstance(input_data, Image.Image):  # Already a PIL Image
            input_image_pil = input_data
        else:
            raise ValueError("Unsupported type for input_image")
        
        # Convert and normalize the image
        input_image = F.pil_to_tensor(input_image_pil.convert("RGB"))
        input_image = (input_image.float().unsqueeze(0).cuda() / 255 - 0.5) / 0.5
        
        z0 = autoencoder.encode(input_image)
        masked_z = z0 * inpainting_mask
        inpainting_extra_input = torch.cat([masked_z, inpainting_mask], dim=1)

    

    # - - - - - input for gligen - - - - - #
    grounding_input = grounding_tokenizer_input.prepare(batch)
    grounding_extra_input = None
    if grounding_downsampler_input != None:
        grounding_extra_input = grounding_downsampler_input.prepare(batch)

    input = dict(
                x = starting_noise, 
                timesteps = None, 
                context = context, 
                grounding_input = grounding_input,
                inpainting_extra_input = inpainting_extra_input,
                grounding_extra_input = grounding_extra_input,

            )


    # - - - - - start sampling - - - - - #
    shape = (batch_size, model.in_channels, model.image_size, model.image_size)

    samples_fake = sampler.sample(S=steps, shape=shape, input=input,  uc=uc, guidance_scale=guidance_scale, mask=inpainting_mask, x0=z0)
    samples_fake = autoencoder.decode(samples_fake)

    # - - - - - save images - - - - - #
    output_folder = os.path.join(folder, meta["save_folder_name"])

    if debug_mode:
        os.makedirs(output_folder, exist_ok=True)

    start = len(os.listdir(output_folder)) if debug_mode else 0
    image_ids = list(range(start, start + batch_size)) if debug_mode else list(range(batch_size))
    print(image_ids)

    opencv_images = []

    for image_id, sample in zip(image_ids, samples_fake):
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample_np = sample.cpu().numpy().transpose(1, 2, 0) * 255
        sample_pil = Image.fromarray(sample_np.astype(np.uint8))
        
        if debug_mode:
            img_name = str(int(image_id)) + '.png'
            sample_pil.save(os.path.join(output_folder, img_name))
        
        # Convert the PIL image to OpenCV format regardless of the debug_mode
        sample_cv = cv2.cvtColor(np.array(sample_pil), cv2.COLOR_RGB2BGR)
        opencv_images.append(sample_cv)

    # If not in debug_mode, opencv_images will contain all your images in OpenCV format without saving them.
    return opencv_images



def generate_new_img(
        input_image, 
        prompt, 
        images, 
        locations, 
        folder="generation_samples", 
        batch_size=1, 
        no_plms=False, 
        guidance_scale=7.5, 
        negative_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality',
        image_size=512,
        starting_noise_flag=None,
        debug_mode=False,
        ):

    config = dict(
        ckpt = "../checkpoints/checkpoint_inpainting_text_image.pth",
        input_image = input_image,
        prompt = prompt,
        images = images,
        locations = locations, 
        save_folder_name="inpainting_box_image"
    )

    if starting_noise_flag == "random":
        starting_noise = torch.randn(batch_size, 4, 64, 64).to(device)
    else:
        starting_noise = None

    return run(
        config,
        folder,
        batch_size,
        no_plms,
        guidance_scale,
        negative_prompt,
        starting_noise,
        image_size,
        debug_mode
    )
