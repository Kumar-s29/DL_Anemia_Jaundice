import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import sys
import os

# --- Function to load and cache models ---
@st.cache_resource
def load_anemia_model():
    """Loads the pre-trained Anemia detection model."""
    try:
        model_path = 'models/model_anemia.h5'
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("✅ Anemia model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading Anemia model: {e}")
        st.error(f"Model path: {os.path.abspath('models/model_anemia.h5')}")
        st.error(f"File exists: {os.path.exists('models/model_anemia.h5')}")
        return None

@st.cache_resource
def load_jaundice_model():
    """Loads the pre-trained Jaundice detection model with custom objects."""
    try:
        # Add the models directory to path (outside cache to ensure it's always available)
        models_path = os.path.join(os.getcwd(), 'jaundice_model', 'models')
        if models_path not in sys.path:
            sys.path.insert(0, models_path)
        
        # Import custom layers
        from ConvNeXt import LayerScale, StochasticDepth
        
        # Create custom objects dictionary
        custom_objects = {
            'LayerScale': LayerScale,
            'StochasticDepth': StochasticDepth,
        }
        
        # Load model with custom objects
        model_path = 'models/jaunenet_full_model.h5'
        
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects,
            compile=False
        )
        # cache last conv layer name to speed up Grad-CAM later
        try:
            # find a 4D output tensor layer (conv-like)
            for layer in reversed(model.layers):
                try:
                    shape = layer.output_shape
                except Exception:
                    shape = None
                if shape and isinstance(shape, tuple) and len(shape) == 4:
                    model._gradcam_target_layer = layer.name
                    break
        except Exception:
            model._gradcam_target_layer = None
        
        return model
        
    except Exception as e:
        st.error(f"Error loading Jaundice model: {str(e)}")
        st.error("Please ensure the model file and ConvNeXt layers are available")
        return None

# --- Initialize paths for custom layers ---
def initialize_paths():
    """Initialize system paths for custom layers"""
    models_path = os.path.join(os.getcwd(), 'jaundice_model', 'models')
    if models_path not in sys.path:
        sys.path.insert(0, models_path)

# Initialize paths
initialize_paths()

# --- Main Streamlit App ---
st.set_page_config(
    page_title="Eye Disease Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("👁️ Deep Learning-Based Eye Image Analysis")

# Model status check
with st.expander("🔧 Model Status", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Anemia Model:**")
        anemia_model = load_anemia_model()
        if anemia_model is not None:
            st.success("✅ Loaded successfully")
        else:
            st.error("❌ Failed to load")
    
    with col2:
        st.write("**Jaundice Model:**")
        jaundice_model = load_jaundice_model()
        if jaundice_model is not None:
            st.success("✅ Loaded successfully")
        else:
            st.error("❌ Failed to load")

st.markdown("---")

# Option selection (explicit)
option = st.selectbox(
    "Select the condition to analyze:",
    ("Anemia (Conjunctiva)", "Jaundice (Sclera)"),
    index=1,
    help="Choose which model to run on the uploaded image"
)

# File uploader
uploaded_file = st.file_uploader(
    "Upload an eye image for analysis", 
    type=['png', 'jpg', 'jpeg']
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader(f"Analysis: {option}")
        # Add threshold control and show raw probabilities for jaundice when selected
        if option == "Jaundice (Sclera)":
            st.write("#### ⚙️ Jaundice Detection Settings")
            jaundice_threshold = st.slider(
                "Jaundice probability threshold (use if you want stricter detection)",
                min_value=0.0,
                max_value=1.0,
                value=0.32,
                step=0.01,
            )
            show_raw = st.checkbox("Show raw softmax probabilities", value=True)
        
        if option == "Anemia (Conjunctiva)":
            anemia_model = load_anemia_model()
            if anemia_model:
                # Preprocessing for Anemia model (64x64)
                img_array = np.array(image) / 255.0
                h, w = 64, 64  # Anemia model input size
                resized_img = tf.image.resize(img_array, (h, w))
                preprocessed_img = tf.expand_dims(resized_img, axis=0)

                # Predict
                with st.spinner('Analyzing image for anemia...'):
                    prediction = anemia_model.predict(preprocessed_img, verbose=0)
                    probability = prediction[0][0]  # sigmoid output between 0 and 1
                
                # For binary classification with sigmoid:
                # 0 = Anemic, 1 = Non-Anemic (based on alphabetical order from ImageDataGenerator)
                # So lower probability means Anemic, higher means Non-Anemic
                
                # Display result with progress
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if probability < 0.5:  # Low probability = Anemic
                        st.error("🩸 **Anemia Detected**")
                        confidence = (1 - probability) * 100
                        st.write(f"**Confidence:** {confidence:.1f}%")
                        if confidence > 90:
                            st.write("⚠️ **High confidence detection - Please consult a healthcare professional**")
                    else:  # High probability = Non-Anemic
                        st.success("✅ **No Anemia Detected**")
                        confidence = probability * 100
                        st.write(f"**Confidence:** {confidence:.1f}%")
                        
                with col2:
                    # Show confidence meter
                    st.metric(
                        label="Confidence", 
                        value=f"{confidence:.1f}%",
                        delta=None
                    )

        elif option == "Jaundice (Sclera)":
            jaundice_model = load_jaundice_model()
            if jaundice_model:
                # Preprocessing for Jaundice model - EXACT match to training preprocessing
                # Convert PIL image to tensor (this matches tf.io.decode_image behavior)
                # Preprocessing for Jaundice model - Simplified to match Anemia model (squash/resize)
                # This ensures the entire image is seen by the model, even if the eye is not centered
                img_tensor = tf.constant(np.array(image), dtype=tf.float32) / 255.0
                IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128
                
                # Simple resize to target dimensions
                resized_img = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
                preprocessed_img = tf.expand_dims(resized_img, axis=0)

                # Debug: Show what the model sees
                with st.expander("Show preprocessed image (what the model sees)"):
                    st.image(resized_img.numpy(), clamp=True, caption="Resized Input (128x128)")

                # Predict
                with st.spinner('Analyzing image for jaundice...'):
                    prediction = jaundice_model.predict(preprocessed_img, verbose=0)
                
                # Jaundice model has 3 classes with softmax output (alphabetical order)
                # Class 0: Healthy, Class 1: Obvious, Class 2: Occult
                class_probabilities = prediction[0]
                healthy_prob = float(class_probabilities[0])
                obvious_prob = float(class_probabilities[1])
                occult_prob = float(class_probabilities[2])

                # Optionally show raw probabilities
                if 'show_raw' in locals() and show_raw:
                    st.write("Raw softmax probabilities (Healthy, Obvious, Occult):")
                    st.write([round(healthy_prob, 4), round(obvious_prob, 4), round(occult_prob, 4)])

                # Use slider threshold for deciding detection (user-controlled)
                # Note: slider value is in [0,1]
                threshold = jaundice_threshold

                # (Grad-CAM integration removed — interactive thresholding remains)


                # Decide based on threshold: if either obvious or occult exceed threshold, mark as jaundice
                if occult_prob >= threshold or obvious_prob >= threshold:
                    # pick the higher of the two
                    if occult_prob >= obvious_prob:
                        predicted_class = 2
                        confidence = occult_prob * 100
                        result_text = "⚠️ **Occult (Mild) Jaundice Detected**"
                        advice = "💡 *Mild jaundice detected - monitor symptoms and consider medical consultation*"
                        color = "warning"
                    else:
                        predicted_class = 1
                        confidence = obvious_prob * 100
                        result_text = "🟡 **Obvious Jaundice Detected**"
                        advice = "⚠️ **Seek immediate medical attention**"
                        color = "error"
                else:
                    predicted_class = 0
                    confidence = healthy_prob * 100
                    if healthy_prob >= 0.75:
                        result_text = "✅ **Normal (Healthy)**"
                        advice = "No signs of jaundice detected. Maintain good health!"
                        color = "success"
                    else:
                        result_text = "🔍 **Inconclusive Result**"
                        advice = "Unclear result. Consider retaking photo with better lighting or consult healthcare provider if concerned."
                        color = "info"
                
                # Check for very low confidence across all classes
                max_confidence = max(healthy_prob, obvious_prob, occult_prob)
                if max_confidence < 40:
                    st.warning("⚠️ **Low Confidence Prediction**")
                    st.write("The model is not confident in this prediction. Consider:")
                    st.write("- Better lighting conditions")
                    st.write("- Clearer image of the sclera (white part of eye)")
                    st.write("- Professional medical evaluation")
                
                # Display result with detailed breakdown
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if color == "success":
                        st.success(result_text)
                    elif color == "warning":
                        st.warning(result_text)
                    elif color == "error":
                        st.error(result_text)
                    else:
                        st.info(result_text)
                        
                    st.write(f"**Detection Confidence:** {confidence:.1f}%")
                    st.write(advice)
                        
                with col2:
                    st.metric(
                        label="Detection Confidence", 
                        value=f"{confidence:.1f}%",
                        delta=None
                    )
                    
                    # Show probability breakdown
                    with st.expander("📊 Detailed Probabilities"):
                        st.write(f"**Healthy (Normal):** {healthy_prob:.1f}%")
                        st.write(f"**Obvious Jaundice:** {obvious_prob:.1f}%")
                        st.write(f"**Occult (Mild) Jaundice:** {occult_prob:.1f}%")
                        st.write("---")
                        st.write("**Detection Thresholds:**")
                        st.write("• Occult: >32% • Possible: >26% • Obvious: >23% • Healthy: >58%")
                        st.write("**Baseline (normal white):** H=~57%, O=~21%, C=~21%")

# Information section
if uploaded_file is None:
    st.markdown("---")
    st.header("📋 How to Use")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🩸 Anemia Detection")
        st.markdown("""
        - **Target Area**: Conjunctiva (inner eyelid)
        - **Image Requirements**: 
          - Clear, well-lit photo
          - Focus on the inner eyelid area
          - Minimal shadows or glare
        - **Detection**: Identifies pale conjunctiva indicating potential anemia
        """)
        
    with col2:
        st.subheader("🟡 Jaundice Detection")
        st.markdown("""
        - **Target Area**: Sclera (white part of the eye)
        - **Image Requirements**:
          - Clear view of the eye's white area
          - Natural lighting preferred
          - Avoid heavy makeup or filters
        - **Detection**: Identifies yellowing of sclera indicating potential jaundice
        """)
    
    st.markdown("---")
    st.info("💡 **Tip**: For best results, take photos in natural daylight without flash.")
    
    st.markdown("### ⚠️ Medical Disclaimer")
    st.markdown("""
    This application is for educational and research purposes only. It is **NOT** intended for medical diagnosis. 
    Always consult qualified healthcare professionals for proper medical evaluation and treatment.
    """)
