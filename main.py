from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

class Data(BaseModel):
    prompt: str

@app.post("/analyse")
def read_item(data: Data):
    return {"prompt": data.prompt}
