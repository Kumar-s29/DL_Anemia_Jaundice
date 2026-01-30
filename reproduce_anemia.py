
import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Path to the image
IMAGE_PATH = r"k:\DL_Anemia_Jaundice\anemia_model\dataset\New_Augmented_Anemia_Dataset\Conjuctiva\Validation\Non-Anemic\Non-Anemic-001_aug2.png"
MODEL_PATH = r"models\model_anemia.h5"

def preprocess_anemia_api(pil_image: Image.Image):
    img = pil_image.convert('RGB')
    arr = np.array(img).astype('float32') / 255.0
    arr = tf.image.resize(arr, (64, 64)).numpy()
    arr = np.expand_dims(arr, 0)
    return arr

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return

    if not os.path.exists(IMAGE_PATH):
        print(f"Image not found at {IMAGE_PATH}")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded.")

    print(f"Loading image from {IMAGE_PATH}")
    img = Image.open(IMAGE_PATH)
    
    # Preprocess using API logic
    processed = preprocess_anemia_api(img)
    
    print("Predicting...")
    preds = model.predict(processed, verbose=0)
    prob = float(preds[0][0])
    
    print(f"Probability output: {prob}")
    
    # API Logic
    if prob < 0.5:
        predicted = 'Anemic'
        confidence = (1 - prob) * 100
    else:
        predicted = 'Non-Anemic'
        confidence = prob * 100
        
    print(f"Prediction: {predicted} (Confidence: {confidence:.2f}%)")
    print(f"Expected: Anemic")

if __name__ == "__main__":
    main()
