from fastapi import FastAPI, File
from segmentation import get_yolov5, get_image_from_bytes
from starlette.responses import Response
import io
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2


model = get_yolov5()

app = FastAPI(
    title="Custom YOLOV5 Machine Learning API",
)

@app.post("/object-to-img")
async def detect_food_return_base64_img(file: bytes = File(...)):
    input_image = get_image_from_bytes(file)	
    #cv2.imwrite('C:/Users/N/Desktop/Test_gray.jpg', image_gray)
    #cv2.imshow("result",input_image)
    results = model(input_image)
    results.render()  # updates results.imgs with boxes and labels
    for img in results.imgs:
        bytes_io = io.BytesIO()
        img_base64 = Image.fromarray(img)
        img_base64.save(bytes_io, format="jpeg")
    return Response(content=bytes_io.getvalue(), media_type="image/jpeg")