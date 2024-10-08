# attach_embeddings_to_chroma.py
from sentence_transformers import SentenceTransformer
import chromadb
import time
import numpy as np
import os

# Specify the same folder as a directory for the Chroma database
local_db_path = os.path.join(os.path.dirname(__file__), "chroma_db")  # Uses the same directory as the script

# Load the Chroma client with local storage
client = chromadb.PersistentClient(path=local_db_path)

# Load pre-trained sentence embedding model (SentenceTransformer)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load the existing Chroma collection
collection = client.get_collection("bookcorpus")

time_to_store_embeddings = []

# Retrieve the documents from Chroma to compute embeddings
# Remove 'ids' from the include list
all_docs = collection.get(include=["documents", "metadatas"])

for i, doc in enumerate(all_docs["documents"]):
    sentence = doc
    sentence_id = all_docs["metadatas"][i]["sentence_id"]
    # Use the index to create a doc_id
    doc_id = str(i)  

    # Measure time for embedding generation
    start_time = time.time()
    embedding = model.encode(sentence).tolist()  # Generate embedding using SentenceTransformer
    end_time = time.time()
    time_to_store_embeddings.append(end_time - start_time)

    # Update the document with the embedding
    collection.update(
        ids=str([sentence_id]),  # Use the sentence_id from metadatas as the document ID
        embeddings=[embedding]  # Attach embedding to the existing document
    )

print(f"Embeddings attached to {len(all_docs['documents'])} documents in Chroma.")

# Calculate statistics for storing embeddings
min_time_store_embeddings = np.min(time_to_store_embeddings)
max_time_store_embeddings = np.max(time_to_store_embeddings)
avg_time_store_embeddings = np.mean(time_to_store_embeddings)
std_time_store_embeddings = np.std(time_to_store_embeddings)

# Print statistics
print("\nStatistics for Storing Embeddings:")
print(f"Min: {min_time_store_embeddings:.6f} s")
print(f"Max: {max_time_store_embeddings:.6f} s")
print(f"Avg: {avg_time_store_embeddings:.6f} s")
print(f"Std: {std_time_store_embeddings:.6f} s")
