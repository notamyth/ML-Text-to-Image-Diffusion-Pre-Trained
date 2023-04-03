from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ml import obtain_image

app = FastAPI()

@app.get("/")
def read_root():
    return{"hello":"world"}

#class Item(BaseModel):

@app.get("/generate")
def generate_image(prompt: str):
    image = obtain_image(prompt, num_inference_steps=5, seed=1024)
    image.save("image.png")
    return FileResponse("image.png")
    
