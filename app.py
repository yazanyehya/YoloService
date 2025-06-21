import os
import uuid
import shutil
import json
import time
from threading import Thread
from collections import Counter
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image

import boto3
import torch
from ultralytics import YOLO

from storage.sqlite_storage import SQLiteStorage
from storage.dynamodb_storage import DynamoDBStorage

# --- App Init ---
app = FastAPI()

UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# --- YOLO Model ---
model = YOLO("yolov8n.pt")
torch.cuda.is_available = lambda: False

# --- AWS Setup ---
AWS_REGION = os.getenv("AWS_REGION", "us-west-1")
S3_BUCKET = os.getenv("AWS_S3_BUCKET", "yazan-dev-images-polybot")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")
s3_client = boto3.client("s3", region_name=AWS_REGION)
sqs_client = boto3.client("sqs", region_name=AWS_REGION)

# --- Storage Injection ---
storage_type = os.getenv("STORAGE_TYPE", "sqlite")
if storage_type == "dynamodb":
    storage = DynamoDBStorage()
else:
    storage = SQLiteStorage()

# --- S3 Helper ---
def upload_to_s3(local_path: str, s3_key: str):
    try:
        s3_client.upload_file(local_path, S3_BUCKET, s3_key)
    except Exception as e:
        print(f"❌ Failed to upload {s3_key}: {e}")

# --- Request Model ---
class PredictRequest(BaseModel):
    image_key: Optional[str] = None

# --- Routes ---
@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(None)):
    ext = ".jpg"
    uid = str(uuid.uuid4())
    original_path = os.path.join(UPLOAD_DIR, uid + ext)
    predicted_path = os.path.join(PREDICTED_DIR, uid + ext)

    body = None
    try:
        body = await request.json()
    except:
        pass

    image_key = body.get("image_key") if body else None

    if image_key:
        try:
            s3_client.download_file(S3_BUCKET, image_key, original_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download from S3: {e}")
    elif file:
        ext = os.path.splitext(file.filename)[1]
        original_path = os.path.join(UPLOAD_DIR, uid + ext)
        with open(original_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    else:
        raise HTTPException(status_code=400, detail="Provide either file or image_key.")

    results = model(original_path, device="cpu")
    annotated = Image.fromarray(results[0].plot())
    annotated.save(predicted_path)

    upload_to_s3(predicted_path, f"predicted/{uid}{ext}")
    storage.save_prediction(uid, original_path, predicted_path)

    detected_labels = []
    for box in results[0].boxes:
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        storage.save_detection(uid, label, score, str(bbox))
        detected_labels.append(label)

    label_counts = dict(Counter(detected_labels))

    return {
        "prediction_uid": uid,
        "label_counts": label_counts
    }

    }

@app.get("/prediction/{uid}")
def get_prediction(uid: str):
    data = storage.get_prediction(uid)
    if not data:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return data

@app.get("/predictions/label/{label}")
def get_by_label(label: str):
    return storage.get_predictions_by_label(label)

@app.get("/predictions/score/{min_score}")
def get_by_score(min_score: float):
    return storage.get_predictions_by_score(min_score)

@app.get("/image/{type}/{filename}")
def get_image(type: str, filename: str):
    if type not in ["original", "predicted"]:
        raise HTTPException(status_code=400, detail="Invalid image type")
    path = os.path.join("uploads", type, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)

@app.get("/prediction/{uid}/image")
def get_prediction_image(uid: str, request: Request):
    accept = request.headers.get("accept", "")
    data = storage.get_prediction(uid)
    if not data:
        raise HTTPException(status_code=404, detail="Prediction not found")

    image_path = data["predicted_image"]
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found")

    if "image/png" in accept:
        return FileResponse(image_path, media_type="image/png")
    elif "image/jpeg" in accept or "image/jpg" in accept:
        return FileResponse(image_path, media_type="image/jpeg")
    else:
        raise HTTPException(status_code=406, detail="Client does not accept image format")

@app.get("/health")
def health():
    return {"status": "ok"}

# --- Background SQS Consumer ---
def process_sqs_message(body):
    image_key = body["image_key"]
    chat_id = body["chat_id"]
    prediction_id = body["prediction_id"]

    filename = image_key.split("/")[-1]
    local_path = os.path.join(UPLOAD_DIR, filename)
    predicted_path = os.path.join(PREDICTED_DIR, filename)

    try:
        s3_client.download_file(S3_BUCKET, image_key, local_path)
        print(f"✅ Downloaded {image_key} from S3")
    except Exception as e:
        print(f"❌ Failed to download from S3: {e}")
        return

    results = model(local_path, device="cpu")
    annotated = Image.fromarray(results[0].plot())
    annotated.save(predicted_path)
    s3_client.upload_file(predicted_path, S3_BUCKET, f"predicted/{filename}")
    storage.save_prediction(prediction_id, local_path, predicted_path)

    for box in results[0].boxes:
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        storage.save_detection(prediction_id, label, score, str(bbox))

    print(f"✅ Processed prediction {prediction_id}")


def poll_sqs():
    print("🔁 YOLO background SQS consumer started...")
    while True:
        try:
            response = sqs_client.receive_message(
                QueueUrl=SQS_QUEUE_URL,
                MaxNumberOfMessages=5,
                WaitTimeSeconds=20
            )
            messages = response.get("Messages", [])
            if not messages:
                time.sleep(1)
                continue

            for msg in messages:
                try:
                    body = json.loads(msg["Body"])
                    process_sqs_message(body)
                    sqs_client.delete_message(
                        QueueUrl=SQS_QUEUE_URL,
                        ReceiptHandle=msg["ReceiptHandle"]
                    )
                    print(f"🗑️ Deleted message {msg['MessageId']}")
                except Exception as e:
                    print(f"❌ Failed to process message: {e}")
        except Exception as e:
            print(f"❌ SQS polling error: {e}")
            time.sleep(5)

@app.on_event("startup")
def start_background_consumer():
    thread = Thread(target=poll_sqs, daemon=True)
    thread.start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)