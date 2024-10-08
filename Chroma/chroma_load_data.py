# fetch_postgres_and_store_to_chroma.py
import chromadb
import time
import numpy as np
import os
from config import load_config
from connect import postgres_conn

# Specify the same folder as a directory for the Chroma database
local_db_path = os.path.join(os.path.dirname(__file__), "chroma_db")  # Uses the same directory as the script

# Create the directory if it doesn't exist
os.makedirs(local_db_path, exist_ok=True)

# Load the Chroma client with local storage
client = chromadb.PersistentClient(
    path=local_db_path
)

# Create or access the collection in Chroma
collection_name = "bookcorpus"
if collection_name not in client.list_collections():
    collection = client.create_collection(collection_name)
else:
    collection = client.get_collection(collection_name)

# Load the database configuration
config = load_config()

# Use the postgres_conn class to connect to the PostgreSQL database
connection = postgres_conn()
connection.post_connect(config)

# Fetch all rows from the table (no LIMIT applied here)
connection.execute_query("SELECT id, sentence FROM bookcorpus")
records = connection.cursor_fetch()  # Fetch results from the cursor

# Lists to store execution times
time_to_store_text = []

unique_id = 0

# Insert only the fetched text data into Chroma (no embeddings yet)
for record in records:
    sentence_id, text_part = record
    sentences = text_part.split('. ')  # Assuming simple sentence splitting by period
    
    for sentence in sentences:
        # Measure time for storing text only
        start_time = time.time()
        
        # Add the sentence to the Chroma collection (without embeddings for now)
        collection.add(
            ids=[str(unique_id)],  # Unique ID for the sentence
            documents=[sentence],
            metadatas=[{"sentence_id": sentence_id}]
        )
        
        end_time = time.time()
        time_to_store_text.append(end_time - start_time)

        unique_id += 1

print(f"Collection size after loading text: {collection.count()}")

# Close the database connection
connection.close_connection()

# Calculate statistics for storing text
min_time_store_text = np.min(time_to_store_text)
max_time_store_text = np.max(time_to_store_text)
avg_time_store_text = np.mean(time_to_store_text)
std_time_store_text = np.std(time_to_store_text)

# Print statistics
print("\nStatistics for Storing Text:")
print(f"Min: {min_time_store_text:.6f} s")
print(f"Max: {max_time_store_text:.6f} s")
print(f"Avg: {avg_time_store_text:.6f} s")
print(f"Std: {std_time_store_text:.6f} s")

print("Text data loaded into Chroma.")
