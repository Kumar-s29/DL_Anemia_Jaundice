import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

# Mocking the BASE_DIR logic from ml_api.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
JAUNE_MODEL_PATH = os.path.join(MODELS_DIR, 'jaunenet_full_model.h5')

print(f"Base Dir: {BASE_DIR}")
print(f"Model Path: {JAUNE_MODEL_PATH}")
print(f"Exists: {os.path.exists(JAUNE_MODEL_PATH)}")

def load_model():
    # Jaundice model (may require custom ConvNeXt layers)
    try:
        # ensure custom layers path is available
        convnext_models = os.path.join(BASE_DIR, 'jaundice_model', 'models')
        if convnext_models not in sys.path:
            sys.path.insert(0, convnext_models)
        
        print(f"Jaundice models dir: {convnext_models}")
        
        # import custom layers if available
        try:
            from ConvNeXt import LayerScale, StochasticDepth
            custom_objects = {'LayerScale': LayerScale, 'StochasticDepth': StochasticDepth}
            print("imported custom layers")
        except Exception as e:
            print(f"failed to import custom layers: {e}")
            custom_objects = None

        if os.path.exists(JAUNE_MODEL_PATH):
            if custom_objects:
                jaundice_model = tf.keras.models.load_model(JAUNE_MODEL_PATH, custom_objects=custom_objects, compile=False)
            else:
                jaundice_model = tf.keras.models.load_model(JAUNE_MODEL_PATH, compile=False)
            print("Jaundice model loaded successfully")
            return jaundice_model
        else:
            print(f"Jaundice model not found at {JAUNE_MODEL_PATH}")
            return None
    except Exception as e:
        print("Failed to load jaundice model:", e)
        import traceback
        traceback.print_exc()
        return None

def preprocess_jaundice(pil_image):
    try:
        img = pil_image.convert('RGB')
        arr = np.array(img).astype('float32') / 255.0
        # Simple resize to 128x128
        final = tf.image.resize(arr, (128, 128)).numpy()
        final = np.expand_dims(final, 0)
        print(f"Preprocessed shape: {final.shape}")
        return final
    except Exception as e:
        print(f"Preprocessing error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    model = load_model()
    if not model:
        print("Model failed to load. Exiting.")
        return

    # Create dummy image
    img = Image.new('RGB', (300, 200), color='white')
    input_tensor = preprocess_jaundice(img)
    
    if input_tensor is None:
        print("Preprocessing failed")
        return

    try:
        print("Predicting...")
        preds = model.predict(input_tensor, verbose=0)
        print("Prediction success:", preds)
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
