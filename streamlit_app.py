import streamlit as st
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
@st.cache_resource  # Caches the model for better performance
def load_model():
    # Assuming the model and tokenizer have been downloaded from GitHub
    model_path = "temp_repo/saved_model/bert_model"
    tokenizer_path = "temp_repo/saved_model/bert_tokenizer"

    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model = BertModel.from_pretrained(model_path)

    return model, tokenizer

# Function to get embeddings from text
def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Load the model and tokenizer
model, tokenizer = load_model()

# Streamlit UI
st.title("BERT Model - Text Embedding App")

# User input
user_input = st.text_area("Enter text to analyze:")

if st.button("Generate Embedding"):
    if user_input:
        embedding = get_embedding(user_input, model, tokenizer)
        st.write("Generated Embedding:")
        st.write(embedding.detach().numpy())
    else:
        st.write("Please enter text to generate an embedding.")

