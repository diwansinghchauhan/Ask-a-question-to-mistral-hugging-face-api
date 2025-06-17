import os
import streamlit as st
from transformers import pipeline

# Use Hugging Face API token from secrets
HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Load HF inference pipeline
generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", token=HUGGINGFACEHUB_API_TOKEN)

# Streamlit app
st.title("Ask a Question to Mistral (Hugging Face API)")

input_text = st.text_input("Ask a question:")

if input_text:
    with st.spinner("Thinking..."):
        output = generator(input_text, max_new_tokens=100, do_sample=True)
        st.success("Answer:")
        st.write(output[0]['generated_text'])
