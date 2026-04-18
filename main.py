from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import h5py
import json
import tensorflow as tf
import cv2
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Load Model ──────────────────────────────
def load_my_model():
    model_path = r"C:\Users\mahek\Desktop\best_model_BACKUP.h5"

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

print("Loading model...")
model = load_my_model()
print("Model loaded!")

# ── Grad-CAM ────────────────────────────────
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


# ── Routes ──────────────────────────────────
@app.get("/")
def home():
    return {"status": "running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # Full res for display
    display_img = Image.open(io.BytesIO(contents)).convert('RGB')

    # 64x64 for model
    img_64 = display_img.resize((64, 64))
    img_array = np.array(img_64) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = float(model.predict(img_input, verbose=0)[0][0])
    is_real = prediction > 0.5

    # Grad-CAM
    heatmap = get_gradcam_heatmap(model, img_input)
    original, superimposed = overlay_gradcam(img_array, heatmap)

    # Heatmap to RGB for display
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