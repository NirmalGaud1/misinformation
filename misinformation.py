import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define custom LSTM class to match training implementation
class CompatibleLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)  # Remove problematic argument
        super().__init__(*args, **kwargs)

# Custom objects for model loading
custom_objects = {
    'Attention': tf.keras.layers.Attention,
    'CompatibleLSTM': CompatibleLSTM
}

@st.cache_resource
def load_model():
    try:
        model_path = os.path.abspath('model_nilesh.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        return tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects
        )
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

@st.cache_resource
def load_tokenizer():
    try:
        tokenizer_path = os.path.abspath('tokenizer.pkl')
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
            
        with open(tokenizer_path, 'rb') as handle:
            return pickle.load(handle)
    except Exception as e:
        st.error(f"❌ Tokenizer loading failed: {str(e)}")
        st.stop()

def preprocess_text(text, tokenizer, max_length=100):
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')

# App Interface
st.set_page_config(page_title="Misinformation Detector", layout="wide")
st.title("🔍 AI Misinformation Detection System")

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This system analyzes text to detect potential misinformation using:
    - **LSTM** with **Attention Mechanism**
    - Trained on 10,000+ samples
    - 92% test accuracy
    """)
    
    try:
        st.image('training_history.png', 
                caption='Model Training History',
                use_column_width=True)
    except FileNotFoundError:
        st.warning("Training history visualization not available")

# Load components
model = load_model()
tokenizer = load_tokenizer()

# Main interface
user_input = st.text_area(
    "Enter text to analyze:",
    "",
    height=150,
    placeholder="Paste news text or tweet here..."
)

if st.button("Analyze", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze")
        st.stop()
        
    with st.spinner("Analyzing text..."):
        try:
            # Preprocess and predict
            processed_text = preprocess_text(user_input, tokenizer)
            prediction = model.predict(processed_text, verbose=0)
            confidence = prediction[0][0]
            
            # Display results
            st.subheader("Analysis Result")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if confidence > 0.5:
                    st.error("🚩 Potential Misinformation")
                    score = confidence
                else:
                    st.success("✅ Likely Legitimate")
                    score = 1 - confidence
                    
                st.metric("Confidence", f"{score:.2%}")
                
            with col2:
                st.progress(float(score))
                st.markdown("""
                **Interpretation Guide:**
                - **Above 50%**: Higher chance of misinformation
                - **Below 50%**: More likely legitimate
                """)
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
⚠️ **Important Notes:**
- This is an AI-assisted tool, not a definitive verdict
- Always verify with fact-checking sources
- Model accuracy: 92% on test data
""")
