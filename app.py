from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
import logging
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO)  # Log messages at INFO level and above
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS Middleware
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
    try:
        logger.info("Loading YOLO model...")
        model = YOLO('yolov8n.pt')
        logger.info("YOLO model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")

@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {"message": "Server is running 🚀"}

@app.get("/ping")
async def ping():
    logger.info("Ping endpoint accessed.")
    return {"status": "up"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file for detection: {file.filename}")

        contents = await file.read()  # Read file into memory
        npimg = np.frombuffer(contents, np.uint8)  # Convert bytes to numpy array
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)  # Decode numpy array into OpenCV image

        logger.info(f"Image loaded and processed for detection.")

        results = model(img, save=False, save_txt=False, save_conf=False, verbose=False)
        
        logger.info(f"Detection results obtained for {file.filename}.")

        detections = []
        for box in results[0].boxes:
            cls = int(box.cls[0].item())
            confidence = round(float(box.conf[0].item()), 2)
            label = model.names[cls]
            detections.append({"object": label, "confidence": confidence})

        logger.info(f"Detections: {detections}")

        return JSONResponse(content={"detections": detections})

    except Exception as e:
        logger.error(f"Error during detection for {file.filename}: {e}")
        return JSONResponse(content={"error": "Error during detection"}, status_code=500)
