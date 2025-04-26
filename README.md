# Photo Task Backend (FastAPI + YOLOv8)

## How to Deploy to Render

1. Create a GitHub repository.
2. Add `app.py`, `requirements.txt`, `.gitignore`, and this README.
3. Go to [Render.com](https://render.com).
4. Click **New** → **Web Service**.
5. Connect your GitHub repository.
6. Settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port 10000`
7. Deploy!

## API Usage

- `POST /detect`
- FormData with `file` (image upload).
- Response:
```json
{
  "detections": [
    {"object": "plant", "confidence": 0.92},
    {"object": "bottle", "confidence": 0.81}
  ]
}
