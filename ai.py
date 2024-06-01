import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import unicodedata
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to load JSON data from a file
def load_json_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to remove control characters, newlines, and unwanted Unicode symbols from text
def remove_unwanted_characters(text):
    cleaned_text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    cleaned_text = cleaned_text.replace('\n', ' ')
    return cleaned_text

# Function to extract comments
def extract_comments(data):
    comments = []
    for post_title, post_comments in data['posts'].items():
        for comment_data in post_comments:
            comment_text = remove_unwanted_characters(comment_data['comment'])
            comments.append("<START> " + comment_text + " <END>")
    return comments

# Generator to yield tokenized sequences
def generate_tokenized_sequences(comments, tokenizer, max_length):
    for comment in comments:
        token_list = tokenizer.texts_to_sequences([comment])[0]
        if token_list:
            yield pad_sequences([token_list], maxlen=max_length, padding='pre')[0]

# Function to tokenize and pad sequences
def tokenize_and_pad_sequences(comments):
    tokenizer = Tokenizer(num_words=10000, filters='')  # Limit number of words in tokenizer and use custom filters
    tokenizer.fit_on_texts(comments)

    max_length = 100  # Set a fixed maximum sequence length
    sequences = generate_tokenized_sequences(comments, tokenizer, max_length)
    sequences = list(sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post")

    vocab_size = len(tokenizer.word_index) + 1
    labels = np.zeros((len(padded_sequences), max_length), dtype=np.int32)
    for i, sequence in enumerate(padded_sequences):
        labels[i] = np.array(sequence)

    return tokenizer, max_length, padded_sequences, labels

# Function to generate text
def generate_text(seed_text, next_words, max_sequence_len, tokenizer, model):
    generated_text = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]

        # Get the index of the predicted word with the highest probability
        predicted_index = np.argmax(predicted_probs[-1])

        output_word = tokenizer.index_word.get(predicted_index, '')
        generated_text += " " + output_word
        seed_text += " " + output_word
        seed_text = " ".join(seed_text.split()[-(max_sequence_len-1):])

    return generated_text

# Function to save tokenizer to a file
def save_tokenizer(tokenizer, tokenizer_path):
    tokenizer_json = tokenizer.to_json()
    with open(tokenizer_path, 'w') as file:
        file.write(tokenizer_json)

# Define and compile model
def define_model(vocab_size, max_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128),
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.Dense(vocab_size, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)  # Decreased learning rate to 0.0005
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Main function
def main():
    file_path = 'subreddit_top_comments.json'
    tokenizer_path = 'tokenizer.json'

    # Check if GPU is available
    gpu_devices = tf.config.list_physical_devices('GPU')
    if not gpu_devices:
        print("No GPU found. Using CPU.")
        device = '/CPU:0'
    else:
        print("GPU found. Using GPU.")
        device = '/GPU:0'

    # Load JSON data
    try:
        data = load_json_from_file(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: JSON decoding failed for file '{file_path}'.")
        return

    # Extract comments
    comments = extract_comments(data)

    # Tokenize and pad sequences
    try:
        tokenizer, max_length, padded_sequences, labels = tokenize_and_pad_sequences(comments)
    except ValueError as ve:
        print(f"Error: {ve}")
        return

    # Save tokenizer
    save_tokenizer(tokenizer, tokenizer_path)

    # Define model
    with tf.device(device):
        model = define_model(len(tokenizer.word_index) + 1, max_length)

        while True:
            try:
                choice = int(input("Enter 1 to train , or 2 to generate text: "))
                if choice == 1:
                    # Define early stopping callback
                    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

                    # Train model
                    model.fit(padded_sequences, labels, epochs=25, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
                    break
                elif choice == 2:
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        if choice == 1 or choice == 2:
            # Generate text example
            seed_text = "I"
            generated_text = generate_text(seed_text, next_words=10, max_sequence_len=max_length, tokenizer=tokenizer, model=model)
            print("Generated Text:", generated_text)

if __name__ == "__main__":
    main()
