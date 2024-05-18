import streamlit as st
from transformers import pipeline

# Load the fine-tuned model
model_name = 'models/phi3-finetuned'
generator = pipeline('text-generation', model=model_name, tokenizer=model_name)

st.title('Script Generator')
topic = st.text_input('Enter a topic:')
if st.button('Generate Script'):
    script = generator(f"### Human: {topic} ### Assistant:", max_length=500)[0]['generated_text']
    st.write(script)
