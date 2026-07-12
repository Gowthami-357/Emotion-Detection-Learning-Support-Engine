import os
import pandas as pd
from datetime import datetime

FILE = "logs/emotion_response_examples.csv"


def save_interaction(
    text,
    emotion,
    confidence,
    response="",
    field=""
):

    os.makedirs("logs", exist_ok=True)

    row = {
        "text": text,
        "emotion": emotion,
        "confidence": confidence,
        "response": response,
        "field": field,
        "timestamp": datetime.now().isoformat()
    }

    if os.path.exists(FILE):
        df = pd.read_csv(FILE)
        df = pd.concat(
            [df, pd.DataFrame([row])],
            ignore_index=True
        )
    else:
        df = pd.DataFrame([row])

    df.to_csv(FILE, index=False)