import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# Load data
df = pd.read_csv("data/combined_emotions.csv")

# Load tokenizer
with open("models/bltsm/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(df["text"])

X = pad_sequences(
    sequences,
    maxlen=80,
    padding="post",
    truncating="post"
)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(df["emotion"])
y = to_categorical(y)

# Save label mapping
with open("models/bltsm/label_mapping.txt", "w") as f:
    for label in encoder.classes_:
        f.write(label + "\n")


# Build model
model = Sequential([
    Embedding(
        input_dim=30000,
        output_dim=128,
        input_shape=(80,)
    ),

    Bidirectional(
        LSTM(128)
    ),

    Dropout(0.3),

    Dense(64, activation="relu"),

    Dense(5, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Explicitly build the model
model.build(input_shape=(None, 80))

# Show summary
model.summary()
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# Train model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=4,
    callbacks=[early_stop]
)

# Save model
model.save("models/bltsm/bilstm_student_adaptive.keras")

print("\nModel saved successfully!")