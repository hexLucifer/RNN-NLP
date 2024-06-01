import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import unicodedata
import nltk
from nltk.corpus import stopwords
import spacy
import configparser
import os

# Download NLTK stopwords if not already downloaded
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load SpaCy model for NER
nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])

# Function to load JSON data from a file
def load_json_from_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


# Function to remove control characters, newlines, and unwanted Unicode symbols from text
def remove_unwanted_characters(text):
    cleaned_text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    cleaned_text = cleaned_text.replace("\n", " ")
    return cleaned_text


# Function to extract comments and NER annotations
def extract_comments(data):
    comments = []
    ner_annotations = []

    for post_title, post_comments in data["posts"].items():
        for comment_data in post_comments:
            comment_text = remove_unwanted_characters(comment_data["comment"])
            comments.append("<START> " + comment_text + " <END>")

            # Handle missing 'ner' key
            if "ner" in comment_data:
                ner_annotations.append(comment_data["ner"])
            else:
                ner_annotations.append(
                    []
                )  # If 'ner' key is missing, append an empty list

    return comments, ner_annotations


# Tokenization and padding function
def tokenize_and_pad_sequences(comments, max_length):
    tokenizer = Tokenizer(
        num_words=10000, filters=""
    )  # Limit number of words in tokenizer and use custom filters
    tokenizer.fit_on_texts(comments)

    sequences = tokenizer.texts_to_sequences(comments)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post")

    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, vocab_size, padded_sequences


# Function to save tokenizer to a file
def save_tokenizer(tokenizer, tokenizer_path):
    tokenizer_json = tokenizer.to_json()
    with open(tokenizer_path, "w") as file:
        file.write(tokenizer_json)


# Function to define and compile model
def define_model(vocab_size, max_length, embedding_dim, lstm_units, learning_rate):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
            tf.keras.layers.LSTM(
                lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2
            ),
            tf.keras.layers.LSTM(
                lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2
            ),
            tf.keras.layers.Dense(vocab_size, activation="softmax"),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Function to train the model
def train(config):
    train_ratio = config.getfloat("training", "train_ratio")
    val_ratio = config.getfloat("training", "val_ratio")
    test_ratio = config.getfloat("training", "test_ratio")
    file_path = config.get("data", "file_path")
    tokenizer_path = config.get("data", "tokenizer_path")
    model_path = config.get("data", "model_path")
    max_length = config.getint("data", "max_length")
    embedding_dim = config.getint("model", "embedding_dim")
    lstm_units = config.getint("model", "lstm_units")
    learning_rate = config.getfloat("model", "learning_rate")
    epochs = config.getint("training_options", "epochs")
    batch_size = config.getint("training_options", "batch_size")
    patience = config.getint("training_options", "patience")

    # Check if GPU is available
    gpu_devices = tf.config.list_physical_devices("GPU")
    if not gpu_devices:
        print("No GPU found. Using CPU.")
        device = "/CPU:0"
    else:
        print("GPU found. Using GPU.")
        device = "/GPU:0"

    # Load JSON data
    try:
        data = load_json_from_file(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: JSON decoding failed for file '{file_path}'.")
        return

    # Extract comments and NER annotations
    comments, ner_annotations = extract_comments(data)

    # Tokenize and pad sequences
    try:
        tokenizer, vocab_size, padded_sequences = tokenize_and_pad_sequences(
            comments, max_length
        )
    except ValueError as ve:
        print(f"Error: {ve}")
        return

    # Save tokenizer
    save_tokenizer(tokenizer, tokenizer_path)

    # Split data into training, validation, and testing sets
    total_size = len(padded_sequences)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    X_train = padded_sequences[:train_size]
    X_val = padded_sequences[train_size : train_size + val_size]
    X_test = padded_sequences[train_size + val_size :]

    y_train = padded_sequences[:train_size]
    y_val = padded_sequences[train_size : train_size + val_size]
    y_test = padded_sequences[train_size + val_size :]

    # Create TensorFlow datasets
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(train_size)
        .batch(batch_size)
    )
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(
        batch_size
    )

    # Define model
    with tf.device(device):
        model = define_model(
            vocab_size, max_length, embedding_dim, lstm_units, learning_rate
        )

        # Define early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )

        # Train model
        try:
            model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset,
                callbacks=[early_stopping],
            )
            model.save(model_path)  # Save the trained model

            # Evaluate model on test set
            test_loss, test_accuracy = model.evaluate(test_dataset)
            print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")
            model.save(model_path)  # Save the trained model when interrupted
            sys.exit(0)  # Exit the script

    # Save chosen options to config file
    config.set("training", "train_ratio", str(train_ratio))
    config.set("training", "val_ratio", str(val_ratio))
    config.set("training", "test_ratio", str(test_ratio))
    with open("config.ini", "w") as configfile:
        config.write(configfile)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")

    train(config)
