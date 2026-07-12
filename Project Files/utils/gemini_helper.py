import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)


def get_ai_response(field, emotion, confidence, problem):
    try:
        prompt = f"""
You are an empathetic AI learning assistant.

Student field: {field}
Detected emotion: {emotion}
Confidence: {confidence:.2%}

Problem:
{problem}

Please:
1. Acknowledge the student's feelings.
2. Give field-specific advice.
3. Suggest practical next steps.
4. End with encouragement.

Keep the response concise and supportive.
"""

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        return response.text.strip()

    except Exception:
        fallback = {
            "Bored": "😴 You seem tired or disengaged. Try breaking the task into smaller, more interesting pieces.",
            "Confused": "🤔 Confusion is part of learning. Focus on one concept at a time and ask questions.",
            "Frustrated": "😤 Take a short break and return with a fresh perspective. Progress often comes gradually.",
            "Curious": "🧐 Your curiosity is a strength. Explore examples and experiment with new ideas.",
            "Confident": "💪 Great confidence! Keep challenging yourself with more advanced problems."
        }

        return fallback.get(
            emotion,
            "Keep learning step by step. You're making progress."
        )