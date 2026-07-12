import re

CONFIDENT_KEYWORDS = [
    "confident", "sure", "understand", "know", "ready", "prepared"
]

CONFUSED_KEYWORDS = [
    "confused", "unclear", "don't understand", "why", "stuck", "help"
]


def clean_text(text):
    text = text.lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9!?., ]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def keyword_boost(scores, text):
    text = text.lower()

    if any(word in text for word in CONFIDENT_KEYWORDS):
        scores["Confident"] *= 2.5

    if any(word in text for word in CONFUSED_KEYWORDS):
        scores["Confused"] *= 2.0

    total = sum(scores.values())

    for key in scores:
        scores[key] /= total

    return scores