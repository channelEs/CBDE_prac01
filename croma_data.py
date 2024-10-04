import psycopg2
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from scipy.spatial.distance import cosine, euclidean
import time

# Load pre-trained sentence embedding model (SentenceTransformer)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load the Chroma client
client = chromadb.Client()

# Create a collection in Chroma
collection = client.create_collection("bookcorpus")

# Connect to PostgreSQL
conn = psycopg2.connect(database="suppliers", user="***", password="***", host="localhost", port="***")
cursor = conn.cursor()

# Fetch all rows from the table (no LIMIT applied here)
cursor.execute("SELECT id, part_number, text_part FROM bookcorpus")
records = cursor.fetchall()

# Lists to store execution times
time_to_store_text = []
time_to_store_embeddings = []

unique_id = 0

# Insert the fetched data into Chroma with embeddings
for record in records:
    sentence_id, part_number, text_part = record
    sentences = text_part.split('. ')  # Assuming simple sentence splitting by period
    
    for sentence in sentences:
        # Measure time for embedding generation
        start_time = time.time()
        embedding = model.encode(sentence).tolist()  # Generate embedding using SentenceTransformer
        end_time = time.time()
        time_to_store_embeddings.append(end_time - start_time)

        # Measure time for storing text and embedding
        start_time = time.time()
        
        # Add the sentence and its embedding to the Chroma collection
        collection.add(
            ids=[str(unique_id)],  # Unique ID for the sentence
            documents=[sentence],
            metadatas=[{"sentence_id": sentence_id, "part_number": part_number}],
            embeddings=[embedding]
        )
        
        end_time = time.time()
        time_to_store_text.append(end_time - start_time)

        unique_id += 1

print(f"Collection size after insertion: {collection.count()}")

# Close the database connection
cursor.close()
conn.close()

print("Data loaded into Chroma.")

# Example: Performing similarity search for 10 sentences
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

    print("Top-2 Cosine Similarity Results:")
    for i, result in enumerate(result_docs):
        cosine_sim = 1 - cosine(sample_embedding, result_embeddings[i])
        print(f"Document: {result}, Cosine Similarity: {cosine_sim:.4f}")

    print("Top-2 Euclidean Distance Results:")
    for i, result in enumerate(result_docs):
        euclidean_dist = euclidean(sample_embedding, result_embeddings[i])
        print(f"Document: {result}, Euclidean Distance: {euclidean_dist:.4f}")

# Calculate statistics for storing text
min_time_store_text = np.min(time_to_store_text)
max_time_store_text = np.max(time_to_store_text)
avg_time_store_text = np.mean(time_to_store_text)
std_time_store_text = np.std(time_to_store_text)

# Calculate statistics for storing embeddings
min_time_store_embeddings = np.min(time_to_store_embeddings)
max_time_store_embeddings = np.max(time_to_store_embeddings)
avg_time_store_embeddings = np.mean(time_to_store_embeddings)
std_time_store_embeddings = np.std(time_to_store_embeddings)

# Print statistics
print("\nStatistics for Storing Text:")
print(f"Min: {min_time_store_text:.6f} s")
print(f"Max: {max_time_store_text:.6f} s")
print(f"Avg: {avg_time_store_text:.6f} s")
print(f"Std: {std_time_store_text:.6f} s")

print("\nStatistics for Storing Embeddings:")
print(f"Min: {min_time_store_embeddings:.6f} s")
print(f"Max: {max_time_store_embeddings:.6f} s")
print(f"Avg: {avg_time_store_embeddings:.6f} s")
print(f"Std: {std_time_store_embeddings:.6f} s")
