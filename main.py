from typing import Union
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import os
import base64
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup the StableDifussion model. This will take a while to load so we do it once here and reuse the model
model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(
    model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to("cuda")


@app.get("/")
def home():
    return {"message": "The service is running"}

class Data(BaseModel):
  input: list

@app.post("/analyse")
def analyse_data(data: Data):
    if not data.input:
        raise HTTPException(status_code=400, detail="Input is required")

    # loop through the input and generate the images
    images = []
    for prompt in data.input:
        image = pipe(prompt['phrase'], height=768, width=768).images[0]
        images.append({
            "id": prompt["id"],
            "image": image
        })
    # image = pipe(data.prompt, height=768, width=768).images[0]
    # delete folder images and all its contents
    os.makedirs("images", exist_ok=True)


    for count, image in enumerate(images):
        # save the image to the images folder
        image["image"].save(f"images/{image['id']}.png")
    return {"success": True}

# an endpoint that accepts url param as image id and returns the image
@app.get("/image/{image_id}")
def serve_image(image_id: str):
    if os.path.exists(f"images/{image_id}.png"):
        image_bytes = None
        with open(f"images/{image_id}.png", "rb") as f:
            image_bytes = f.read()
        return Response(content=image_bytes, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Image not found")
