import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env
load_dotenv()

# LangSmith optional tracking (only needed if using LangSmith)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "")

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the question asked."),
    ("user", "Question: {question}")
])

# Streamlit UI
st.title("LangChain-based-chatbot-using-Ollama-s-Mistral-model")
input_text = st.text_input("Ask a question:")

# Ollama model setup
llm = Ollama(model="mistral")
output_parser = StrOutputParser()

# Build the chain
chain = prompt | llm | output_parser

# Process input
if input_text:
    with st.spinner("Thinking..."):
        response = chain.invoke({"question": input_text})
    st.success("Answer:")
    st.write(response)
