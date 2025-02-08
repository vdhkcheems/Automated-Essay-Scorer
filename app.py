from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import torch
from pydantic import BaseModel
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import torch.nn.functional as F
from utils import *
from huggingface_hub import snapshot_download

# Initialize FastAPI app
app = FastAPI()

# Serve static files (CSS, images, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve templates
templates = Jinja2Templates(directory="templates")

# Automatically download the model from Hugging Face
model_dir = snapshot_download(repo_id="vdhkcheems/deberta_v3_large_finetuned")

# Load the tokenizer and model
tokenizer = DebertaV2Tokenizer.from_pretrained(model_dir)
model = DebertaV2ForSequenceClassification.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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

    # Tokenize the essay text
    inputs = tokenizer(clean_text, return_tensors="pt", padding=True, truncation=True)

    # Move the inputs to the GPU
    inputs = {key: value.to(device) for key, value in inputs.items()}


    # Run the model to get predictions
    with torch.no_grad():  # Inference mode (no gradient calculation)
        outputs = model(**inputs)
        logits = outputs.logits

    # Apply argmax to get the predicted class (ordinal score)
    predicted_class = torch.argmax(logits, dim=-1).item()

    return {"score": predicted_class, "message": "Essay scored successfully!"}

