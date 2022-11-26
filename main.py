from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def home():
    return {"Hello": "World"}

class Data(BaseModel):
    prompt: str

@app.post("/analyse")
def analyse_data(data: Data):
    return {"prompt": data.prompt}
