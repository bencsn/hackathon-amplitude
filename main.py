from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
app = FastAPI()


# Setup the StableDifussion model. This will take a while to load so we do it once here and reuse the model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16)


@app.get("/")
def home():
    return {"message": "The service is running"}


class Data(BaseModel):
    prompt: str


@app.post("/analyse")
def analyse_data(data: Data):
    if not data.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    image = pipe(data.prompt, height=768, width=768).images[0]
    # Make a directory called "images" and save the image there
    image.save("images/image.png")
    return {"prompt": data.prompt, "success": True}


@app.get("/image")
def serve_image():
    # Check if the image exists
    if os.path.exists("images/image.png"):
        with open("images/image.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return encoded_string
    else:
        raise HTTPException(status_code=404, detail="Image not found")
