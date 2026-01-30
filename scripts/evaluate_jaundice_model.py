import os
import sys
import time
from typing import List

import numpy as np
from PIL import Image
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "jaunenet_full_model.h5")

TEST_ROOT = os.path.join(BASE_DIR, "jaundice_model", "test")
CLASSES: List[str] = ["Healthy", "Obvious", "Occult"]


def preprocess_jaundice(pil_image: Image.Image) -> np.ndarray:
    """Mirror backend.preprocess_jaundice for consistent evaluation."""
    img = pil_image.convert("RGB")
    arr = np.array(img).astype("float32") / 255.0
    h, w = arr.shape[0], arr.shape[1]
    side = min(h, w)
    start_h = (h - side) // 2
    start_w = (w - side) // 2
    arr_cropped = arr[start_h : start_h + side, start_w : start_w + side]
    zoom_rate = 1.05
    target_size = int(128 * zoom_rate)
    arr_resized = tf.image.resize(arr_cropped, (target_size, target_size)).numpy()
    final = tf.image.resize(arr_resized, (128, 128)).numpy()
    final = np.expand_dims(final, 0)
    return final


def load_model() -> tf.keras.Model:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    # ensure ConvNeXt layers are importable
    convnext_models = os.path.join(BASE_DIR, "jaundice_model", "models")
    if convnext_models not in sys.path:
        sys.path.insert(0, convnext_models)

    try:
        from ConvNeXt import LayerScale, StochasticDepth

        custom_objects = {"LayerScale": LayerScale, "StochasticDepth": StochasticDepth}
    except Exception:
        custom_objects = None

    if custom_objects:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
    else:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model


def collect_samples() -> List[tuple]:
    samples = []
    for idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(TEST_ROOT, cls)
        if not os.path.isdir(cls_dir):
            print(f"[WARN] No directory for class '{cls}' at {cls_dir}")
            continue
        for root, _, files in os.walk(cls_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    samples.append((os.path.join(root, f), idx))
    return samples


def main():
    print("Base dir:", BASE_DIR)
    print("Model path:", MODEL_PATH)
    print("Test root:", TEST_ROOT)

    samples = collect_samples()
    if not samples:
        print("No test images found. Expected structure:")
        print("  jaundice_model/test/Healthy/*.jpg|png")
        print("  jaundice_model/test/Obvious/*.jpg|png")
        print("  jaundice_model/test/Occult/*.jpg|png")
        return

    print(f"Found {len(samples)} test images.")

    model = load_model()
    print("Model loaded.")

    y_true = []
    y_pred = []

    start = time.time()
    for path, label_idx in samples:
        try:
            img = Image.open(path)
        except Exception as e:
            print(f"[SKIP] Failed to open {path}: {e}")
            continue

        x = preprocess_jaundice(img)
        preds = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        y_true.append(label_idx)
        y_pred.append(pred_idx)

    duration = time.time() - start
    if not y_true:
        print("No successful predictions.")
        return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = (y_true == y_pred).mean()
    print(f"\nOverall accuracy: {accuracy * 100:.2f}% on {len(y_true)} samples")
    print(f"Average time per image: {duration / len(y_true):.3f}s")

    # confusion matrix
    num_classes = len(CLASSES)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    print("\nConfusion matrix (rows=true, cols=pred):")
    header = "      " + "  ".join(f"{c[:7]:>7}" for c in CLASSES)
    print(header)
    for i, cls in enumerate(CLASSES):
        row = " ".join(f"{cm[i, j]:7d}" for j in range(num_classes))
        print(f"{cls[:7]:>7} {row}")


if __name__ == "__main__":
    main()
