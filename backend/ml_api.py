from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import base64
import time
import os
import sys
import tensorflow as tf

app = FastAPI(title="EyeHealth ML API")

# allow local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to models (relative to repository root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.basename(os.path.dirname(__file__)) == 'backend' else os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ANEMIA_MODEL_PATH = os.path.join(MODELS_DIR, 'model_anemia.h5')
JAUNE_MODEL_PATH = os.path.join(MODELS_DIR, 'jaunenet_full_model.h5')

anemia_model = None
jaundice_model = None


@app.on_event("startup")
def load_models():
    global anemia_model, jaundice_model
    print("Loading models...")

    # Anemia model
    try:
        if os.path.exists(ANEMIA_MODEL_PATH):
            anemia_model = tf.keras.models.load_model(ANEMIA_MODEL_PATH, compile=False)
            print("Anemia model loaded")
        else:
            print(f"Anemia model not found at {ANEMIA_MODEL_PATH}")
    except Exception as e:
        print("Failed to load anemia model:", e)
        anemia_model = None

    # Jaundice model (may require custom ConvNeXt layers)
    try:
        # ensure custom layers path is available
        convnext_models = os.path.join(BASE_DIR, 'jaundice_model', 'models')
        if convnext_models not in sys.path:
            sys.path.insert(0, convnext_models)
        # import custom layers if available
        try:
            from ConvNeXt import LayerScale, StochasticDepth
            custom_objects = {'LayerScale': LayerScale, 'StochasticDepth': StochasticDepth}
        except Exception:
            custom_objects = None

        if os.path.exists(JAUNE_MODEL_PATH):
            if custom_objects:
                jaundice_model = tf.keras.models.load_model(JAUNE_MODEL_PATH, custom_objects=custom_objects, compile=False)
            else:
                jaundice_model = tf.keras.models.load_model(JAUNE_MODEL_PATH, compile=False)
            print("Jaundice model loaded")

            # cache candidate gradcam target layer name
            try:
                for layer in reversed(jaundice_model.layers):
                    try:
                        shape = layer.output_shape
                    except Exception:
                        shape = None
                    if shape and isinstance(shape, tuple) and len(shape) == 4:
                        jaundice_model._gradcam_target_layer = layer.name
                        break
            except Exception:
                jaundice_model._gradcam_target_layer = None
        else:
            print(f"Jaundice model not found at {JAUNE_MODEL_PATH}")
    except Exception as e:
        print("Failed to load jaundice model:", e)
        jaundice_model = None


def preprocess_anemia(pil_image: Image.Image):
    img = pil_image.convert('RGB')
    arr = np.array(img).astype('float32') / 255.0
    arr = tf.image.resize(arr, (64, 64)).numpy()
    arr = np.expand_dims(arr, 0)
    return arr


def preprocess_jaundice(pil_image: Image.Image):
    img = pil_image.convert('RGB')
    arr = np.array(img).astype('float32') / 255.0
    h, w = arr.shape[0], arr.shape[1]
    # make square crop based on min dimension
    side = min(h, w)
    # center crop
    start_h = (h - side) // 2
    start_w = (w - side) // 2
    arr_cropped = arr[start_h:start_h+side, start_w:start_w+side]
    # zoom_rate logic used in training
    zoom_rate = 1.05
    target_size = int(128 * zoom_rate)
    arr_resized = tf.image.resize(arr_cropped, (target_size, target_size)).numpy()
    final = tf.image.resize(arr_resized, (128, 128)).numpy()
    final = np.expand_dims(final, 0)
    return final


def make_data_url_from_png_bytes(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def compute_gradcam_image(model, preprocessed_tensor, class_index: int):
    # Find target conv layer
    target_layer_name = getattr(model, '_gradcam_target_layer', None)
    if not target_layer_name:
        # fallback: find first 4D output layer from the end
        for layer in reversed(model.layers):
            try:
                shape = layer.output_shape
            except Exception:
                shape = None
            if shape and isinstance(shape, tuple) and len(shape) == 4:
                target_layer_name = layer.name
                break
    if not target_layer_name:
        return None

    # Build a model that maps the input to the activations and predictions
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(target_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        inputs = tf.cast(preprocessed_tensor, tf.float32)
        (conv_outputs, predictions) = grad_model(inputs)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    # compute guided weights
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[..., i] *= pooled_grads[i]

    heatmap = tf.reduce_mean(conv_outputs, axis=-1).numpy()
    # normalize
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    # convert to RGBA heatmap overlay using PIL and a colormap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('jet')
    heatmap_rgba = cmap(heatmap)
    heatmap_img = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    # resize heatmap to model input size (128x128)
    heat_pil = Image.fromarray(heatmap_img).resize((128, 128), resample=Image.BILINEAR)
    # return bytes of blended overlay later by caller
    buf = io.BytesIO()
    heat_pil.save(buf, format='PNG')
    return buf.getvalue()


@app.post('/analyze')
async def analyze(image: UploadFile = File(...), analysis_type: str = Form(...)):
    # Use the shared inference helper so other endpoints can reuse logic
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
    try:
        result = _run_inference(pil_image, analysis_type, include_heatmap=True)
    except ValueError as ve:
        return JSONResponse(status_code=400, content={'error': str(ve)})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})

    return JSONResponse(status_code=200, content=result)


def _run_inference(pil_image: Image.Image, analysis_type: str, include_heatmap: bool = False):
    start = time.time()
    result = {
        'detected': False,
        'confidence': 0.0,
        'predicted': None,
        'probabilities': None,
        'heatmap_url': None,
        'model_version': '1.0.0',
        'processing_time': None,
    }

    if analysis_type == 'anemia':
        if anemia_model is None:
            raise RuntimeError('Anemia model not loaded')
        x = preprocess_anemia(pil_image)
        preds = anemia_model.predict(x, verbose=0)
        prob = float(preds[0][0])
        # lower prob -> Anemic
        if prob < 0.5:
            predicted = 'Anemic'
            confidence = (1 - prob) * 100
            detected = True
        else:
            predicted = 'Non-Anemic'
            confidence = prob * 100
            detected = False

        result.update({
            'detected': detected,
            'confidence': round(confidence, 3),
            'predicted': predicted,
            'probabilities': {'anemia_prob': prob},
        })

    elif analysis_type == 'jaundice':
        if jaundice_model is None:
            raise RuntimeError('Jaundice model not loaded')
        x = preprocess_jaundice(pil_image)
        preds = jaundice_model.predict(x, verbose=0)
        probs = preds[0].tolist()
        labels = ['Healthy', 'Obvious', 'Occult']
        max_idx = int(np.argmax(probs))
        predicted = labels[max_idx]
        threshold = 0.32
        detected = (probs[1] >= threshold) or (probs[2] >= threshold)
        confidence = float(max(probs)) * 100

        data_url = None
        if include_heatmap:
            try:
                png_bytes = compute_gradcam_image(jaundice_model, x, max_idx)
                if png_bytes:
                    data_url = make_data_url_from_png_bytes(png_bytes)
            except Exception as e:
                print('Grad-CAM generation failed:', e)

        result.update({
            'detected': bool(detected),
            'confidence': round(confidence, 3),
            'predicted': predicted,
            'probabilities': {'healthy': probs[0], 'obvious': probs[1], 'occult': probs[2]},
            'heatmap_url': data_url,
        })

    else:
        raise ValueError('Unknown analysis_type')

    duration = time.time() - start
    result['processing_time'] = round(duration, 3)
    return result


@app.post('/predict')
async def predict(image: UploadFile = File(...), analysis_type: str = Form(...)):
    """Returns JSON prediction (no heatmap). Matches older frontend endpoint naming."""
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
    try:
        result = _run_inference(pil_image, analysis_type, include_heatmap=False)
    except ValueError as ve:
        return JSONResponse(status_code=400, content={'error': str(ve)})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})
    return JSONResponse(status_code=200, content=result)


@app.post('/gradcam')
async def gradcam(image: UploadFile = File(...), analysis_type: str = Form(...)):
    """Returns a PNG image (bytes) for the Grad-CAM overlay for jaundice analyses."""
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
    if analysis_type != 'jaundice':
        return JSONResponse(status_code=400, content={'error': 'Grad-CAM only supported for jaundice'})

    if jaundice_model is None:
        return JSONResponse(status_code=500, content={'error': 'Jaundice model not loaded'})

    x = preprocess_jaundice(pil_image)
    try:
        # pick top class for gradcam
        preds = jaundice_model.predict(x, verbose=0)
        max_idx = int(np.argmax(preds[0].tolist()))
        png_bytes = compute_gradcam_image(jaundice_model, x, max_idx)
        if png_bytes is None:
            return JSONResponse(status_code=500, content={'error': 'Grad-CAM generation failed'})
        return JSONResponse(status_code=200, content={'png_bytes_base64': base64.b64encode(png_bytes).decode('utf-8')})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)