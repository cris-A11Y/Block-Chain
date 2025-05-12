import torch
from transformers import RobertaTokenizerFast
from main import CodeBERTClassifier  # Assuming the model is defined in main.py (you can copy your core code here)
import os
from data_processing import preprocess_single_file  # You need to provide a function to process a single .sol file

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "checkpoint/codebert_classifier.pt"
test_file = "dataset/access_control/arbitrary_location_write_simple.sol"

# 1. Load tokenizer and model
tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")

# Load label mapping and model parameters
checkpoint = torch.load(model_path, map_location=device)
label2id = checkpoint['label2id']
id2label = checkpoint['id2label']

model = CodeBERTClassifier(num_labels=len(label2id))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# 2. Preprocess the test input (Assuming you implemented `preprocess_single_file` in data_processing.py)
#    This function should return input_ids and attention_mask
input_ids, attention_mask = preprocess_single_file(test_file, tokenizer)

# Convert to tensor and add batch dimension
input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)

# 3. Inference and prediction
with torch.no_grad():
    logits = model(input_ids, attention_mask)
    prediction = torch.argmax(logits, dim=1).item()

# 4. Output the predicted label
print(f"Prediction result: {id2label[prediction]}")
