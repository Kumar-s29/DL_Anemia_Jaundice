import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

# Setup paths
sys.path.append(os.getcwd())
models_path = os.path.join(os.getcwd(), 'jaundice_model', 'models')
if models_path not in sys.path:
    sys.path.insert(0, models_path)

print(f"Checking paths...")
print(f"Models path: {models_path}")
print(f"Exists: {os.path.exists(models_path)}")

try:
    from ConvNeXt import LayerScale, StochasticDepth
    print("Successfully imported ConvNeXt custom layers.")
except ImportError as e:
    print(f"Error importing ConvNeXt: {e}")
    sys.exit(1)

def load_jaundice_model():
    custom_objects = {
        'LayerScale': LayerScale,
        'StochasticDepth': StochasticDepth,
    }
    model_path = 'models/jaunenet_full_model.h5'
    print(f"Loading model from: {model_path}")
    
    try:
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects,
            compile=False
        )
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def preprocess_image(image):
    # Logic copied from app.py for testing
    img_tensor = tf.constant(np.array(image), dtype=tf.float32) / 255.0
    zoom_rate = 1.05
    IMAGE_WIDTH = 128
    IMAGE_HEIGHT = 128
    
    height = min(img_tensor.shape[0], img_tensor.shape[1])
    img_tensor = tf.image.resize_with_crop_or_pad(img_tensor, target_height=height, target_width=height)
    
    if height < IMAGE_WIDTH:
        img_tensor = tf.image.resize_with_crop_or_pad(
            img_tensor, 
            target_height=int(IMAGE_WIDTH * zoom_rate),
            target_width=int(IMAGE_WIDTH * zoom_rate)
        )
    else:
        img_tensor = tf.image.resize(
            img_tensor, 
            size=(int(IMAGE_WIDTH * zoom_rate), int(IMAGE_WIDTH * zoom_rate))
        )
    
    resized_img = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
    preprocessed_img = tf.expand_dims(resized_img, axis=0)
    return preprocessed_img

def main():
    model = load_jaundice_model()
    if not model:
        return

    # Create dummy white image (sclera-ish)
    dummy_img = Image.new('RGB', (300, 300), color='white')
    processed = preprocess_image(dummy_img)
    
    print("Running prediction on dummy white image...")
    pred = model.predict(processed, verbose=0)
    probs = pred[0]
    print(f"Raw Probabilities: {probs}")
    print(f"Healthy: {probs[0]:.4f}")
    print(f"Obvious: {probs[1]:.4f}")
    print(f"Occult: {probs[2]:.4f}")
    
    # Check threshold logic
    threshold = 0.32
    healthy_prob = float(probs[0])
    obvious_prob = float(probs[1])
    occult_prob = float(probs[2])
    
    if occult_prob >= threshold or obvious_prob >= threshold:
        if occult_prob >= obvious_prob:
            print("Prediction: Occult Jaundice")
        else:
            print("Prediction: Obvious Jaundice")
    else:
        print("Prediction: Healthy")

if __name__ == "__main__":
    main()
