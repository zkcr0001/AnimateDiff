from diffusers import AutoencoderKL
from omegaconf import OmegaConf
from PIL import Image
import requests

from diffusers.models import UNet2DConditionModel
from animatediff.models.animate_anyone_network import AnimateAnyoneModel
from animatediff.models.animate_anyone_network import UNet3DConditionModel
from animatediff.models.animate_anyone_network import PoseGuider3D
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch
from einops import rearrange
    
pretrained_model_path = "models/StableDiffusion/stable-diffusion-v1-5"
unet_additional_kwargs =  OmegaConf.create({'use_motion_module': True, 'motion_module_resolutions': [1, 2, 4, 8], 'unet_use_cross_frame_attention': False, 'unet_use_temporal_attention': False, 'motion_module_type': 'Vanilla', 'motion_module_kwargs': {'num_attention_heads': 8, 'num_transformer_block': 1, 'attention_block_types': ['Temporal_Self', 'Temporal_Self'], 'temporal_position_encoding': True, 'temporal_position_encoding_max_len': 24, 'temporal_attention_dim_div': 1, 'zero_initialize': True}})
vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").cuda(2)
clip = SentenceTransformer('clip-ViT-L-14').cuda(2)

# initialize animate anyone model
unet = UNet3DConditionModel.from_pretrained_2d(
    pretrained_model_path, subfolder="unet", shrink_half = True,
    unet_additional_kwargs=OmegaConf.to_container(unet_additional_kwargs)
)
pretrained_reference_model_path = 'models/StableDiffusion/stable-diffusion-v1-5'
reference_unet = UNet2DConditionModel.from_pretrained(pretrained_reference_model_path, subfolder="unet")
pose_guider = PoseGuider3D()

animate_anyone_model = AnimateAnyoneModel(ReferenceNet = reference_unet, Pose_Guider3D = pose_guider, Unet_3D = unet)
animate_anyone_model.cuda(2)

# initialize inputs

pose_sequence = torch.randn(1, 16, 3, 256 ,256).cuda(2)
noise_sequence = torch.randn(1, 4, 16, 256 // 8, 256 // 8).cuda(2)
timesteps = torch.tensor(42).cuda(2)

# print(next(vae.parameters()).dtype)
# print(next(clip.parameters()).dtype)
# print(next(reference_unet.parameters()).dtype)
# print(next(pose_guider.parameters()).dtype)
# print(next(unet.parameters()).dtype)
# print(reference_image_torch.dtype)

# reference_image_torch # [B, C, H, W]
# reference_image_pil # PIL image
reference_image_torch = torch.randn(1, 3, 256, 256).cuda(2)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
reference_image_pil = Image.open(requests.get(url, stream=True).raw).resize((256, 256))

reference_latents = vae.encode(reference_image_torch).latent_dist.sample() * 0.18215 # [B, 4, H // 8, W // 8]
reference_prompt_encode = clip.encode(reference_image_pil) # numpy (768,)
reference_prompt_encode = torch.from_numpy(reference_prompt_encode).repeat(1, 77, 1).cuda(2) # torch.size([1, 77, 768])

output = animate_anyone_model(noise_sequence, reference_latents, reference_prompt_encode, pose_sequence, timesteps)
