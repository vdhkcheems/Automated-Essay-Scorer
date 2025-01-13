from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import torch
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
import torch.nn.functional as F
from utils import *
import joblib

# Initialize FastAPI app
app = FastAPI()

# Serve static files (CSS, images, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve templates
templates = Jinja2Templates(directory="templates")

feature_scaler = joblib.load("feature_scaler.pkl")
score_scaler = joblib.load("score_scaler.pkl")

# Load the fine-tuned RoBERTa model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_roberta_with_features")

num_features = 6
model = RobertaWithFeatures.from_pretrained('roberta-base', num_features=num_features)

class Essay(BaseModel):
    text: str

@app.get("/")
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/score/")
async def score_essay(essay: Essay):
    # Validate the input
    if not essay.text.strip():
        return {"error": "Essay text cannot be empty"}

    # Preprocess the essay text
    clean_text = dataPreprocessing(essay.text)

    # Extract features from the essay
    extracted_features = extract_features(clean_text)

    # Convert extracted features to a NumPy array and reshape for scaling
    features_array = np.array(extracted_features).reshape(1, -1)

    # Scale the features using the loaded scaler
    scaled_features = feature_scaler.transform(features_array)

    # Convert the scaled features to a PyTorch tensor
    features_tensor = torch.tensor(scaled_features, dtype=torch.float32)

    # Tokenize the essay text
    inputs = tokenizer(clean_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

    # Get the score from the model
    model.eval()
    with torch.no_grad():
        logits = model(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'], 
            features=features_tensor
        )

    # Convert logits to a NumPy array and reshape for scaling
    score = float(logits.item())
    original_score = ((score+1)/2)*10

    return {"score": round(float(original_score), 2), "message": "Essay scored successfully!"}

