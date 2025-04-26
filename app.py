from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
from io import BytesIO

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None  # Global model

@app.on_event("startup")
async def load_model():
    global model
    model = YOLO('yolov8n.pt')

@app.get("/")
async def root():
    return {"message": "Server is running 🚀"}

@app.get("/ping")
async def ping():
    return {"status": "up"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()  # Read file into memory
    npimg = np.frombuffer(contents, np.uint8)  # Convert bytes to numpy array
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)  # Decode numpy array into OpenCV image

    results = model(img, save=False, save_txt=False, save_conf=False, verbose=False)

    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0].item())
        confidence = round(float(box.conf[0].item()), 2)
        label = model.names[cls]
        detections.append({"object": label, "confidence": confidence})

    return JSONResponse(content={"detections": detections})
