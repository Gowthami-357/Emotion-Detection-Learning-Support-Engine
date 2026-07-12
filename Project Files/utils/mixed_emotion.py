def detect_mixed_emotions(scores, threshold=0.15):

    emotions = []

    for emotion, score in scores.items():

        if score >= threshold:
            emotions.append(emotion)

    return emotions