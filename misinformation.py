import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Custom objects for model loading
custom_objects = {
    'Attention': tf.keras.layers.Attention,
    'LSTM': tf.keras.layers.LSTM
}

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('model_nilesh.h5', custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        raise

@st.cache_resource
def load_tokenizer():
    try:
        with open('tokenizer.pkl', 'rb') as handle:
            return pickle.load(handle)
    except Exception as e:
        st.error(f"Tokenizer loading failed: {str(e)}")
        raise

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
    st.image('training_history.png', caption='Training History')

# Main interface
model = load_model()
tokenizer = load_tokenizer()

user_input = st.text_area("Enter tweet text:", "", height=150, 
                         placeholder="Paste tweet here...")

if st.button("Analyze Content"):
    if user_input:
        with st.spinner("Analyzing text..."):
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
    else:
        st.warning("Please enter text to analyze!")

# Footer
st.markdown("---")
st.markdown("⚠️ **Note**: AI predictions should be verified with human judgment. Model accuracy 92% on test data.")
