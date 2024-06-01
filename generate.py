import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

def generate():
    tokenizer_path = 'tokenizer.json'
    model_path = 'trained_model.h5'
    seed_text = "I"

    # Load tokenizer
    with open(tokenizer_path, 'r') as file:
        tokenizer_json = file.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

    max_sequence_len = 100  # Set the same maximum sequence length used in training

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Generate text
    generated_text = generate_text(seed_text, next_words=10, max_sequence_len=max_sequence_len, tokenizer=tokenizer, model=model)
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    generate()
