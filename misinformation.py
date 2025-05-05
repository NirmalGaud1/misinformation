import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
@st.cache_resource
def load_components():
    model = tf.keras.models.load_model('model_nilesh.h5', custom_objects={'Attention': tf.keras.layers.Attention})
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_components()

# Preprocessing function
def preprocess_text(text):
    max_length = 100
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    return padded

# Streamlit interface
st.title("🔍 Misinformation Detector")
st.write("Analyze tweets for potential misinformation using AI")

# Input section
user_input = st.text_area("Enter tweet text:", "", height=150)

if st.button("Analyze"):
    if user_input:
        # Process and predict
        processed_text = preprocess_text(user_input)
        prediction = model.predict(processed_text)
        confidence = prediction[0][0]
        
        # Display results
        st.subheader("Analysis Result")
        if confidence > 0.5:
            st.error(f"🚩 Potential misinformation detected (confidence: {confidence:.2%})")
        else:
            st.success(f"✅ Likely legitimate content (confidence: {1-confidence:.2%})")
        
        # Confidence visualization
        st.markdown("### Confidence Level")
        st.progress(float(confidence if confidence > 0.5 else 1 - confidence))
        
        # Explanation
        st.markdown("""
        **Interpretation Guide:**
        - Scores above 50% indicate higher likelihood of misinformation
        - Scores below 50% suggest legitimate content
        - Confidence percentage shows model's certainty
        """)
    else:
        st.warning("Please enter some text to analyze!")

# About section
st.markdown("---")
st.markdown("""
**About this tool:**
- Achieves 92% test accuracy
- Uses LSTM with Attention mechanism
- Processes text through neural network embeddings
