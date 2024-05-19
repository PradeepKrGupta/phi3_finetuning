import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import config
from inference import predict

# Load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Streamlit interface
st.title("Script Generation App")
st.write("Generate engaging scripts based on a provided topic.")

# User input
topic = st.text_input("Enter a topic:")

if topic:
    st.write("Generating script...")
    script = predict(topic, model, tokenizer)
    st.write("Generated Script:")
    st.write(script)
