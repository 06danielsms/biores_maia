import os
import pymongo
from pymongo import MongoClient

# Connect to MongoDB (assuming local instance)
client = MongoClient('localhost', 27017)
db = client['health_literacy_db']
collection = db['data_sources']

# Directory to scan
data_dir = '/home/satoru/repos/maia/lab_souces/bridging-the-gap-in-health-literacy/data_analysis/data'

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
        
        # Create document
        document = {
            'file_path': file_path,
            'content': content,
            'filename': os.path.basename(file_path)
        }
        
        # Insert
        collection.insert_one(document)
        print(f"Inserted {file_path}")
    except Exception as e:
        print(f"Error inserting {file_path}: {e}")

print("Insertion complete")
