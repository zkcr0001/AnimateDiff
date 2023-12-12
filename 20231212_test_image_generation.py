import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import requests

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, use_safetensors=True
)

pipe.to("cuda")


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
init_image = Image.open(requests.get(url, stream=True).raw).resize((256, 256))

prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
