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

batch_size = 100
sentence_batch = []
id_batch = []
metadata_batch = []

unique_id = 0

# Process the records in batches of 100
for record in records:
    sentence_id, text_part = record
    
   
    # Add the sentence, ID, and metadata to the respective batches
    sentence_batch.append(text_part)
    id_batch.append(str(unique_id))
    metadata_batch.append({"sentence_id": sentence_id})

    unique_id += 1

    # If the batch reaches the defined size, insert it into Chroma
    if len(sentence_batch) == batch_size:
        # Measure time for storing text
        start_time = time.time()
        
        # Add the batch of sentences to the Chroma collection
        collection.add(
            ids=id_batch,
            documents=sentence_batch,
            metadatas=metadata_batch
        )
            
        end_time = time.time()
        time_to_store_text.append(end_time - start_time)

        # Clear the batches after insertion
        sentence_batch = []
        id_batch = []
        metadata_batch = []

        print("batch " + str(unique_id) + " processed")

# Insert any remaining sentences (less than batch size)
if sentence_batch:
    start_time = time.time()
    
    collection.add(
        ids=id_batch,
        documents=sentence_batch,
        metadatas=metadata_batch
    )
    
    end_time = time.time()
    time_to_store_text.append(end_time - start_time)

print(f"Collection size after bulk loading: {collection.count()}")

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
