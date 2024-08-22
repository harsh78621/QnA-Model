import streamlit as st
import torch
from transformers import BertForQuestionAnswering, BertTokenizer
import requests

# URL to download the model from Google Drive
MODEL_URL = 'https://drive.google.com/uc?id=1c5N3uwnLeaP1TKNgpRmYTacKm5Ew454O'

# Function to download the model
import requests

# Function to download the file
def download_file(url, local_filename):
    response = requests.get(url)
    with open(local_filename, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {local_filename}")

# Download the file
model_url = 'https://drive.google.com/uc?id=1c5N3uwnLeaP1TKNgpRmYTacKm5Ew454O'
model_file = 'model.pth'
download_file(model_url, model_file)

# Check if the file is downloaded and has a valid size
import os
print(f"File size: {os.path.getsize(model_file)} bytes")

# Try to load the model
import torch
from transformers import BertForQuestionAnswering

try:
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")


# Streamlit app
st.title('Model Deployment')

# Download model file
if st.button('Download Model'):
    model_file = 'model.pth'
    download_file(MODEL_URL, model_file)
    st.write("Model downloaded successfully!")

    # Load the model
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    st.write("Model loaded successfully!")

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Example context and question
    context = "The capital of India is Delhi."
    question = "What is the capital of India?"

    def predict_answer(model, tokenizer, context, question, device):
        inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
        
        start_index = torch.argmax(start_logits)
        end_index = torch.argmax(end_logits)
        
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        answer = tokenizer.convert_tokens_to_string(tokens[start_index:end_index+1])
        
        return answer

    # Use the model for prediction
    device = torch.device('cpu')
    answer = predict_answer(model, tokenizer, context, question, device)
    st.write(f'Question: {question}')
    st.write(f'Answer: {answer}')
