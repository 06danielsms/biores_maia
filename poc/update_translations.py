import pymongo
from pymongo import MongoClient
from transformers import MarianMTModel, MarianTokenizer

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['health_literacy_db']
collection = db['data_sources_processing']

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

# Find documents without translated_content
documents = collection.find({"translated_content": {"$exists": False}})

for doc in documents:
    try:
        content = doc['content']
        translated_content = translate_text(content)
        
        # Update document
        collection.update_one(
            {"_id": doc["_id"]},
            {"$set": {"translated_content": translated_content}}
        )
        print(f"Updated document {doc['filename']} with translation")
    except Exception as e:
        print(f"Error updating {doc['filename']}: {e}")

print("Translation update complete")
