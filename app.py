from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import shutil
import os

# Initialize FastAPI app
app = FastAPI()

# Allow all CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    temp_file = file.filename
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(temp_file)

    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0].item())
        confidence = round(float(box.conf[0].item()), 2)
        label = model.names[cls]
        detections.append({"object": label, "confidence": confidence})

    os.remove(temp_file)

    return JSONResponse(content={"detections": detections})
