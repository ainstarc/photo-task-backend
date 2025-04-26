from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
import logging
from collections import defaultdict

# Create FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
MODEL_NAME = 'yolov8s.pt'

# Load YOLOv8 model at startup
@app.on_event("startup")
async def load_model():
    global model
    logger.info(f"Loading model {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    logger.info("Model loaded successfully.")


# Basic root endpoint
@app.get("/")
async def root():
    return {"message": "Server is running 🚀"}

# Ping endpoint
@app.get("/ping")
async def ping():
    return {"status": "up"}

# Object detection endpoint
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}, size: {file.size} bytes")

    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(img, save=False, save_txt=False, save_conf=False, verbose=False)

    label_info = defaultdict(list)

    for box in results[0].boxes:
        cls = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        label = model.names[cls]
        label_info[label].append(confidence)

    detections = []
    for label, confs in label_info.items():
        detections.append({
            "object": label,
            "count": len(confs),
            "average_confidence": round(sum(confs) / len(confs), 2)
        })

    logger.info(f"Detections: {detections}")

    return JSONResponse(content={"detections": detections})
