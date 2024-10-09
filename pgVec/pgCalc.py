from sentence_transformers import SentenceTransformer
from connect import postgres_conn as postG
from config import load_config
import numpy as np
import time

#main
if __name__ == '__main__':
    config = load_config()
    connection = postG()
    connection.post_connect(config)

    # connection.execute_query(
    #     "SELECT id, sentence, embedding FROM pgVec_sentences"
    # )
    # records = connection.cursor_fetch() 
    # print(f'PGvec bookcorpus data readed: {len(records)}')

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

    # Lists to store execution times
    time_to_store_embedding = []
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    for sample_sentence in sample_sentences:
        
        random_embedding = model.encode(sample_sentence).tolist()
        
        # print(type(random_embedding))

        start_time = time.time()

        #  cosine similarity
        # connection.execute_query(
        #     """
        #         SELECT id, sentence, embedding, 1 - (embedding <=> %s::vector) AS similarity
        #         FROM pgVec_sentences
        #         ORDER BY embedding <=> %s::vector ASC
        #         LIMIT 2;
        #     """,
        #     (random_embedding, random_embedding,)
        # )

        # Euclidean distance
        connection.execute_query(
            """
                SELECT id, sentence, embedding, 1 - (l2_distance(embedding, %s::vector)) AS similarity
                FROM pgVec_sentences
                ORDER BY l2_distance(embedding, %s::vector) ASC 
                LIMIT 2;
            """,
            (random_embedding, random_embedding,)
        )

        similarity = connection.cursor_fetch() 
        end_time = time.time()
        time_to_store_embedding.append(end_time - start_time)

        print(f"FOR SENTENCE: {sample_sentence}")
        for row in similarity:
            print(f"ID: {row[0]}, Sentence: {row[1]}, Similarity: {row[3]}")
        print()
        
    # Calculate statistics for storing sentences
    min_time_store_embeddings = np.min(time_to_store_embedding)
    max_time_store_embeddings = np.max(time_to_store_embedding)
    avg_time_store_embeddings = np.mean(time_to_store_embedding)
    std_time_store_embeddings = np.std(time_to_store_embedding)

    print("\nStatistics for Storing Embeddings:")
    print(f"Min: {min_time_store_embeddings:.6f} s")
    print(f"Max: {max_time_store_embeddings:.6f} s")
    print(f"Avg: {avg_time_store_embeddings:.6f} s")
    print(f"Std: {std_time_store_embeddings:.6f} s")

    connection.close_connection()