import pandas as pd
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load data
df = pd.read_csv("data/combined_emotions.csv")

# Parameters
MAX_WORDS = 30000
MAX_LEN = 80

# Create tokenizer
tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(df["text"])

# Convert to sequences
sequences = tokenizer.texts_to_sequences(df["text"])

# Pad sequences
padded = pad_sequences(
    sequences,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

print("Vocabulary size:", len(tokenizer.word_index))
print("Padded shape:", padded.shape)

# Save tokenizer
with open("models/bltsm/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("Tokenizer saved to models/tokenizer.pkl")