from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import re

# -----------------------------
# Load model and tokenizer
# -----------------------------
BERT_MODEL_PATH = "models/bert_emotion_model_final"

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
model.eval()

# -----------------------------
# Emotion labels
# -----------------------------
LABELS = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]

# -----------------------------
# Clean text
# -----------------------------
def clean_text(text):
    if ',' in text:
        parts = text.rsplit(',', 1)
        if parts[1].strip() in LABELS:
            text = parts[0]
            
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return " ".join(text.split())

# -----------------------------
# Main prediction function
# -----------------------------
def predict_bert(text):
    cleaned_text = clean_text(text)

    if not cleaned_text.strip():
        return {
            "emotion": "Confused", 
            "confidence": 0.20, 
            "scores": {l: 0.20 for l in LABELS}, 
            "cleaned_text": ""
        }

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=80
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # 🚨 PURE AI OUTPUTS: Remove manual scales to see what the model actually learned
    probs = F.softmax(outputs.logits, dim=1)[0]

    scores = {}
    for i, emotion in enumerate(LABELS):
        scores[emotion] = probs[i].item()

    # Final prediction determination
    predicted_emotion = max(scores, key=scores.get)
    confidence = scores[predicted_emotion]

    return {
        "emotion": predicted_emotion,
        "confidence": round(confidence, 4),
        "scores": {emotion: round(score, 4) for emotion, score in scores.items()},
        "cleaned_text": cleaned_text
    }

if __name__ == "__main__":
    print(predict_bert("i love maths"))