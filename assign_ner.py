import json
import unicodedata
import nltk
from nltk.corpus import stopwords
import spacy
from tqdm import tqdm
from colorama import init, Fore, Back, Style
import os
import shutil

# Initialize colorama
init(autoreset=True)

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

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
            comments.append(comment_text)
    return comments

# Function to assign NER tags to comments
def assign_ner(comments):
    ner_results = []
    # Define a custom bar format for tqdm
    custom_bar_format = "{l_bar}" + "{bar}" + "{r_bar}"
    for comment in tqdm(comments, desc="Assigning NER tags", bar_format=custom_bar_format):
        try:
            doc = nlp(comment)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            ner_results.append(entities)
        except Exception as e:
            print(f"Error processing comment: {e}")
            ner_results.append([])
    return ner_results

def main():
    file_path = 'subreddit_top_comments.json'
    output_path = 'subreddit_top_comments_with_ner.json'
    temp_output_path = 'subreddit_top_comments_with_ner_temp.json'

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

    # Assign NER tags to comments
    ner_tags = assign_ner(comments)

    # Add NER tags to the original data
    comment_index = 0
    for post_title in data['posts']:
        for comment_data in data['posts'][post_title]:
            comment_data['ner'] = ner_tags[comment_index]
            comment_index += 1

    # Save the updated data with NER tags to a temporary file
    with open(temp_output_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"\nTemporary file '{temp_output_path}' created with NER tags")

    # If processing completed successfully, overwrite the original file
    shutil.move(temp_output_path, output_path)
    print(f"NER tags assigned and saved to '{output_path}'")

if __name__ == "__main__":
    main()
