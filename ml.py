from pathlib import Path
from tkinter import Image

import torch
from diffusers import StableDiffusionPipeline


#hugging face token - https://huggingface.co/CompVis/stable-diffusion-v1-4
token_path = Path("token.txt")
token= token_path.read_text().strip()

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16",  use_auth_token=token,torch_dtype=torch.float16 )

pipe.to("cuda")

prompt = "a photograph of messi with naruto"


def obtain_image(prompt: str,*,seed: int ,num_inference_steps: int = 50,guidance_scale: float = 7.5):
    generator = None if seed is None else torch.Generator("cuda").manual_seed(1024)
    print(f"Using Device: {pipe.device}")
    image:Image = pipe(
        prompt, 
        guidance_scale=guidance_scale, 
        generator=generator
        ).images[0]
        
    return image

#image = obtain_image(prompt, num_inference_steps=5, seed=1024)



