import os
import pymongo
from pymongo import MongoClient
from transformers import MarianMTModel, MarianTokenizer

# Connect to MongoDB (assuming local instance)
client = MongoClient('localhost', 27017)
db = client['health_literacy_db']
collection = db['data_sources_processing']

# Directory to scan
data_dir = '/home/satoru/repos/maia/lab_souces/bridging-the-gap-in-health-literacy/data_collection_and_processing'

# Load translation model
model_name = "Helsinki-NLP/opus-mt-en-es"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Function to translate text from English to Spanish
def translate_text(text):
    max_chunk_length = 512
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(tokenizer(" ".join(current_chunk), return_tensors="pt").input_ids[0]) > max_chunk_length:
            current_chunk.pop()
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    translated_chunks = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        translated_chunks.append(tokenizer.decode(translated[0], skip_special_tokens=True))
    
    return " ".join(translated_chunks)

# Function to determine source based on path
def get_source_from_path(file_path):
    if 'ClinicalTrials.gov' in file_path:
        return 'ClinicalTrials.gov'
    elif 'Cochrane' in file_path:
        return 'Cochrane'
    elif 'Pfizer' in file_path:
        return 'Pfizer'
    elif 'Trial Summaries' in file_path:
        return 'Trial Summaries'
    elif 'pfizer_and_clinicaltrials_processing' in file_path:
        return 'Pfizer and ClinicalTrials Processing'
    elif 'web_scrapping' in file_path:
        return 'Web Scrapping'
    else:
        return 'Other'

# Function to recursively find txt files
def find_txt_files(directory):
    txt_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files

# Find all txt files
txt_files = find_txt_files(data_dir)
print(f"Found {len(txt_files)} txt files")

# Insert into MongoDB
for file_path in txt_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Determine source
        source = get_source_from_path(file_path)
        
        # Translate content to Spanish
        translated_content = translate_text(content)
        
        # Create document
        document = {
            'file_path': file_path,
            'content': content,
            'translated_content': translated_content,
            'filename': os.path.basename(file_path),
            'source': source
        }
        
        # Insert
        collection.insert_one(document)
        print(f"Inserted {file_path} from source: {source}")
    except Exception as e:
        print(f"Error inserting {file_path}: {e}")

print("Insertion complete")
