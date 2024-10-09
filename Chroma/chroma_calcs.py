
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine, euclidean
import time
import os

# Load pre-trained sentence embedding model (SentenceTransformer)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Specify the same folder as a directory for the Chroma database
local_db_path = os.path.join(os.path.dirname(__file__), "chroma_db")  # Uses the same directory as the script

# Load the Chroma client with local storage
client = chromadb.PersistentClient(
    path=local_db_path)

# Access the collection in Chroma
collection = client.get_collection("bookcorpus")



# Lists to store execution times for similarity calculations
time_to_calculate_cosine = []  # For storing cosine similarity calculation times
time_to_calculate_euclidean = []  # For storing Euclidean distance calculation times

# Performing similarity search for 10 sentences
sample_sentences = [
    "It was raining heavily outside.",
    "The quick brown fox jumps over the lazy dog.",
    "She smiled warmly and handed him the book.",
    "He walked slowly towards the horizon.",
    "The sun set behind the mountains.",
    "A sudden gust of wind blew through the trees.",
    "The children played happily in the park.",
    "He opened the door to find an empty room.",
    "The cat curled up on the windowsill.",
    "They shared a laugh over coffee."
]

# Perform similarity search and measure time
for sample_sentence in sample_sentences:
    # Generate embedding for the sample sentence using SentenceTransformer
    sample_embedding = model.encode(sample_sentence).tolist()

    # Query the collection for top-2 most similar sentences
    results = collection.query(
        query_embeddings=[sample_embedding],
        n_results=2,
        include=["embeddings", "documents", "metadatas"]  # Ensure embeddings and documents are included in the results
    )

    # Retrieve embeddings and manually compute distances
    if "embeddings" in results and "documents" in results:
        result_docs = results["documents"][0]
        result_embeddings = results["embeddings"][0]
    else:
        print(f"No embeddings found for: {sample_sentence}")
        continue

    # Display results
    print(f"\nSentence: {sample_sentence}")

    # Measure time for cosine similarity calculation
    start_time = time.time()
    print("Top-2 Cosine Similarity Results:")
    for i, result in enumerate(result_docs):
        cosine_sim = 1 - cosine(sample_embedding, result_embeddings[i])
        print(f"Document: {result}, Cosine Similarity: {cosine_sim:.4f}")
    end_time = time.time()
    time_to_calculate_cosine.append(end_time - start_time)

    # Measure time for Euclidean distance calculation
    start_time = time.time()
    print("Top-2 Euclidean Distance Results:")
    for i, result in enumerate(result_docs):
        euclidean_dist = euclidean(sample_embedding, result_embeddings[i])
        print(f"Document: {result}, Euclidean Distance: {euclidean_dist:.4f}")
    end_time = time.time()
    time_to_calculate_euclidean.append(end_time - start_time)

# Calculate statistics for cosine similarity calculation
min_time_cosine = np.min(time_to_calculate_cosine)
max_time_cosine = np.max(time_to_calculate_cosine)
avg_time_cosine = np.mean(time_to_calculate_cosine)
std_time_cosine = np.std(time_to_calculate_cosine)

# Calculate statistics for Euclidean distance calculation
min_time_euclidean = np.min(time_to_calculate_euclidean)
max_time_euclidean = np.max(time_to_calculate_euclidean)
avg_time_euclidean = np.mean(time_to_calculate_euclidean)
std_time_euclidean = np.std(time_to_calculate_euclidean)

# Print statistics
print("\nStatistics for Cosine Similarity Calculation:")
print(f"Min: {min_time_cosine:.6f} s")
print(f"Max: {max_time_cosine:.6f} s")
print(f"Avg: {avg_time_cosine:.6f} s")
print(f"Std: {std_time_cosine:.6f} s")

print("\nStatistics for Euclidean Distance Calculation:")
print(f"Min: {min_time_euclidean:.6f} s")
print(f"Max: {max_time_euclidean:.6f} s")
print(f"Avg: {avg_time_euclidean:.6f} s")
print(f"Std: {std_time_euclidean:.6f} s")
