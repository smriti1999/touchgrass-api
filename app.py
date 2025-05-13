from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import img_to_array
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import os

# === Focal Loss Function ===
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        loss = -alpha_t * tf.pow(1 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(loss)
    return focal_loss_fixed

# === Initialize FastAPI ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load Model ===
MODEL_NAME = "touch_grass_model_mobilenetv2_aug.h5"
model_path = os.path.join(os.path.dirname(__file__), MODEL_NAME)
model = load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss()})
IMG_SIZE = (224, 224)

# === Prediction Endpoint ===
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize(IMG_SIZE)

    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0]
    label = "Touching Grass" if np.argmax(pred) == 1 else "Not Touching Grass"
    confidence = float(pred[np.argmax(pred)])

    return JSONResponse({
        "label": label,
        "confidence": round(confidence, 4),
        "model": MODEL_NAME
    })

# === Health Check Endpoint ===
@app.get("/ping", include_in_schema=False)
@app.head("/ping", include_in_schema=False)
def ping():
    return {"status": "ok"}