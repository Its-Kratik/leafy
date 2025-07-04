
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🌱 Plant Disease Classification",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #F0F8FF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .confidence-high { color: #228B22; font-weight: bold; }
    .confidence-medium { color: #FFA500; font-weight: bold; }
    .confidence-low { color: #FF4500; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load models with simple error handling"""
    try:
        # Try to load any available model
        model_files = [
            'models_production/voting_ensemble.pkl',
            'models_production/random_forest.pkl',
            'models_production/svm.pkl'
        ]
        
        model = None
        for model_file in model_files:
            if os.path.exists(model_file):
                model = joblib.load(model_file)
                break
        
        if model is None:
            return None, None, None
        
        scaler = joblib.load('models_production/scaler.pkl')
        class_names = joblib.load('models_production/class_names.pkl')
        
        return model, scaler, class_names
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def simple_preprocess(image):
    """Simple image preprocessing without complex dependencies"""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Simple resize (you may need to adjust this based on your model)
        from PIL import Image as PILImage
        if img_array.shape[0] != 256 or img_array.shape[1] != 256:
            image_pil = PILImage.fromarray(img_array)
            image_pil = image_pil.resize((256, 256))
            img_array = np.array(image_pil)
        
        # Simple normalization
        img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None

def extract_simple_features(image):
    """Extract simplified features without complex dependencies"""
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Simple statistical features
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.var(gray),
            np.min(gray),
            np.max(gray)
        ])
        
        # Color channel statistics (if RGB)
        if len(image.shape) == 3:
            for channel in range(3):
                channel_data = image[:, :, channel]
                features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.var(channel_data)
                ])
        
        # Pad features to expected length (adjust based on your model)
        while len(features) < 100:  # Adjust this number
            features.append(0.0)
        
        return np.array(features[:100])  # Adjust this number
        
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🌱 Plant Disease Classification</h1>', unsafe_allow_html=True)
    st.markdown("Upload an image of a plant leaf to detect diseases")
    
    # Load models
    model, scaler, class_names = load_models()
    
    if model is None:
        st.error("⚠️ Unable to load models. Please check model files.")
        st.info("Expected files in models_production/:")
        st.code("""
        models_production/
        ├── random_forest.pkl (or other model)
        ├── scaler.pkl
        └── class_names.pkl
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Info")
        st.write(f"Classes: {len(class_names)}")
        st.write("Model: Loaded Successfully ✅")
        
        st.header("🔬 Disease Classes")
        for class_name in class_names:
            st.write(f"• {class_name.replace('_', ' ').title()}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a plant leaf image...", 
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        st.header("🎯 Results")
        
        if uploaded_file is not None:
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing..."):
                    # Process image
                    processed_image = simple_preprocess(image)
                    
                    if processed_image is not None:
                        # Extract features
                        features = extract_simple_features(processed_image)
                        
                        if features is not None:
                            try:
                                # Scale features
                                features_scaled = scaler.transform(features.reshape(1, -1))
                                
                                # Predict
                                prediction = model.predict(features_scaled)[0]
                                probabilities = model.predict_proba(features_scaled)[0]
                                
                                # Display results
                                predicted_class = class_names[prediction]
                                confidence = probabilities[prediction]
                                
                                # Confidence styling
                                if confidence > 0.8:
                                    conf_class = "confidence-high"
                                elif confidence > 0.6:
                                    conf_class = "confidence-medium"
                                else:
                                    conf_class = "confidence-low"
                                
                                st.markdown(f"""
                                <div class="prediction-box">
                                    <h3>Prediction: {predicted_class.replace('_', ' ').title()}</h3>
                                    <p class="{conf_class}">Confidence: {confidence:.1%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Show all probabilities
                                st.subheader("📊 All Probabilities")
                                prob_df = pd.DataFrame({
                                    'Disease': [name.replace('_', ' ').title() for name in class_names],
                                    'Probability': [f"{p:.1%}" for p in probabilities]
                                })
                                st.dataframe(prob_df, use_container_width=True)
                                
                                # Simple bar chart
                                st.bar_chart(pd.Series(probabilities, index=class_names))
                                
                            except Exception as e:
                                st.error(f"Prediction error: {e}")
                                st.info("This might be due to feature dimension mismatch. Please retrain the model with the simplified features.")
                        else:
                            st.error("❌ Feature extraction failed")
                    else:
                        st.error("❌ Image preprocessing failed")

if __name__ == "__main__":
    main()
