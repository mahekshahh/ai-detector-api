from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from PIL import Image
import numpy as np
import io
import h5py
import json
import tensorflow as tf
import cv2
import base64
import os
import tempfile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Load Image Model ─────────────────────────────────────────
def load_image_model():
    model_path = "best_model_BACKUP.h5"
    with h5py.File(model_path, 'r+') as f:
        model_config = json.loads(f.attrs['model_config'])
        def remove_quantization(obj):
            if isinstance(obj, dict):
                obj.pop('quantization_config', None)
                for v in obj.values():
                    remove_quantization(v)
            elif isinstance(obj, list):
                for item in obj:
                    remove_quantization(item)
        remove_quantization(model_config)
        f.attrs['model_config'] = json.dumps(model_config)
    return load_model(model_path, compile=False)

print("Loading image model...")
model = load_image_model()
print("Image model loaded!")

# ── Load Video Model ─────────────────────────────────────────
def load_video_model():
    model_path = "v2_best_model.h5"
    with h5py.File(model_path, 'r+') as f:
        model_config = json.loads(f.attrs['model_config'])
        def remove_quantization(obj):
            if isinstance(obj, dict):
                obj.pop('quantization_config', None)
                for v in obj.values():
                    remove_quantization(v)
            elif isinstance(obj, list):
                for item in obj:
                    remove_quantization(item)
        remove_quantization(model_config)
        f.attrs['model_config'] = json.dumps(model_config)
    return tf.keras.models.load_model(model_path, compile=False)

print("Loading video model...")
video_model = load_video_model()
print("Video model loaded!")

# ── Grad-CAM ─────────────────────────────────────────────────
def get_gradcam_heatmap(model, img_array):
    base_model = model.layers[0]
    last_conv_layer = base_model.get_layer('out_relu')
    feature_extractor = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )
    img_tensor = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs = feature_extractor(img_tensor)
        tape.watch(conv_outputs)
        x = model.layers[1](conv_outputs)
        x = model.layers[2](x)
        x = model.layers[3](x)
        predictions = model.layers[4](x)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(img_array, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    original = np.uint8(255 * img_array) if img_array.max() <= 1.0 else np.uint8(img_array)
    superimposed = cv2.addWeighted(original, 1 - alpha, heatmap_colored, alpha, 0)
    return original, superimposed

def img_to_base64(img_array):
    img = Image.fromarray(img_array.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ── Face Crop ─────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def crop_face_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        h, w = frame.shape[:2]
        size = min(h, w)
        y1 = (h - size) // 2
        x1 = (w - size) // 2
        return cv2.resize(frame[y1:y1+size, x1:x1+size], (224, 224))
    x, y, w, h = faces[0]
    margin = int(0.3 * max(w, h))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame.shape[1], x + w + margin)
    y2 = min(frame.shape[0], y + h + margin)
    return cv2.resize(frame[y1:y2, x1:x2], (224, 224))

# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    display_img = Image.open(io.BytesIO(contents)).convert('RGB')
    img_64 = display_img.resize((64, 64))
    img_array = np.array(img_64) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    prediction = float(model.predict(img_input, verbose=0)[0][0])
    is_real = prediction > 0.5

    heatmap = get_gradcam_heatmap(model, img_input)
    original, superimposed = overlay_gradcam(img_array, heatmap)

    heatmap_resized = cv2.resize(heatmap, (64, 64))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    return JSONResponse({
        "label": "REAL" if is_real else "AI GENERATED",
        "confidence": round((prediction if is_real else 1 - prediction) * 100, 1),
        "raw": prediction,
        "heatmap": img_to_base64(heatmap_rgb),
        "overlay": img_to_base64(superimposed),
        "original": img_to_base64(np.array(display_img))
    })

@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, 15, dtype=int)

    preds = []
    extracted = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        face = crop_face_from_frame(frame)
        img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        img = efficientnet_preprocess(img.astype(np.float32))
        img = np.expand_dims(img, 0)
        preds.append(float(video_model.predict(img, verbose=0)[0][0]))
        extracted += 1

    cap.release()
    os.remove(tmp_path)

    if not preds:
        return JSONResponse({"error": "No frames extracted"}, status_code=400)

    avg = float(np.mean(preds))
    is_real = avg >= 0.5
    confidence = round((avg if is_real else 1 - avg) * 100, 1)

    return JSONResponse({
        "label": "REAL" if is_real else "AI GENERATED",
        "confidence": confidence,
        "raw": avg,
        "frames": extracted
    })