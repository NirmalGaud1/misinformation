import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Custom objects for model loading
custom_objects = {
    'Attention': tf.keras.layers.Attention,
    'LSTM': tf.keras.layers.LSTM
}

@st.cache_resource
def load_model():
    try:
        model_path = os.path.abspath('model_nilesh.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        raise

@st.cache_resource
def load_tokenizer():
    try:
        tokenizer_path = os.path.abspath('tokenizer.pkl')
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
        with open(tokenizer_path, 'rb') as handle:
            return pickle.load(handle)
    except Exception as e:
        st.error(f"Tokenizer loading failed: {str(e)}")
        raise

def safe_display_image(image_path):
    try:
        image_path = os.path.abspath(image_path)
        if os.path.exists(image_path):
            st.image(image_path, caption='Training History')
        else:
            st.warning("Training history visualization not available")
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")

# App UI
st.set_page_config(page_title="Misinformation Detector", layout="wide")
st.title("🔍 AI Misinformation Detection System")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This system detects potential misinformation in tweets using:
    - **LSTM Network** with Attention mechanism
    - Trained on 10,000+ samples
    - 92% test accuracy
    """)
    safe_display_image('training_history.png')

# Main interface
model = load_model()
tokenizer = load_tokenizer()

user_input = st.text_area("Enter tweet text:", "", height=150, 
                         placeholder="Paste tweet here...")

if st.button("Analyze Content"):
    if user_input:
        with st.spinner("Analyzing text..."):
            try:
                # Preprocess
                sequence = tokenizer.texts_to_sequences([user_input])
                padded = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
                
                # Predict
                prediction = model.predict(padded, verbose=0)
                confidence = prediction[0][0]
                
                # Display results
                st.subheader("Analysis Result")
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if confidence > 0.5:
                        st.error("🚩 Potential Misinformation")
                    else:
                        st.success("✅ Likely Legitimate")
                    
                    st.metric("Confidence Score", f"{max(confidence, 1-confidence):.2%}")
                
                with col2:
                    st.progress(float(confidence if confidence > 0.5 else 1 - confidence))
                    st.caption("Interpretation Guide:")
                    st.markdown("""
                    - **Above 50%**: High probability of misinformation
                    - **Below 50%**: Likely legitimate content
                    - **Threshold**: 50% confidence level
                    """)
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
    else:
        st.warning("Please enter text to analyze!")

# Footer
st.markdown("---")
st.markdown("⚠️ **Note**: AI predictions should be verified with human judgment. Model accuracy 92% on test data.")
