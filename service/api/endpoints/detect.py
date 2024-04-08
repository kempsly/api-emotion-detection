from fastapi import APIRouter, UploadFile, HTTPException
from PIL import Image
from io import BytesIO
import numpy as np

# from emotions_detection.service.core.logic.onnx_inference import emotions_detector

from service.core.logic.onnx_inference import emotions_detector
from service.core.schemas.output import APIOutput





emo_router = APIRouter()


@emo_router.post("/detect", response_model=APIOutput)
async def detect(im: UploadFile):
    
    # Getting the filename extension, to ensure that it is an image
    if im.filename.split(".")[-1] in ("jpg", "jpeg", "png"):
        pass
    else:
        raise HTTPException(status_code = 415, detail="Not in image, load a normal picture")
    
    # Read the image from the client
    image = Image.open(BytesIO(im.file.read()))
    # Convert the image to a numpy array
    image = np.array(image)
    
    
    return emotions_detector(image)