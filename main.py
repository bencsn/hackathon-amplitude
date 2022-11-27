from typing import Union
from fastapi import FastAPI, HTTPException, Response, JSONResponse
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
    prompt: str


@app.post("/analyse")
def analyse_data(data: Data):
    if not data.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    image = pipe(data.prompt, height=768, width=768).images[0]
    # delete folder images and all its contents
    if os.path.exists("images"):
        for file in os.listdir("images"):
            os.remove(os.path.join("images", file))
        os.rmdir("images")

    os.makedirs("images", exist_ok=True)
    image.save("images/image.png")
    return {"prompt": data.prompt, "success": True}


@app.get("/image")
def serve_image():
    if os.path.exists("images/image.png"):
        image_bytes = None
        with open("images/image.png", "rb") as f:
            image_bytes = f.read()
        return Response(content=image_bytes, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Image not found")
