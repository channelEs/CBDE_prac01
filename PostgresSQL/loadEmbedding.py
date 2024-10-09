from connect import postgres_conn as postG
from config import load_config
from sentence_transformers import SentenceTransformer
import numpy as np
import time

# main
if __name__ == '__main__':
    config = load_config()
    connection = postG()
    connection.post_connect(config)

    connection.execute_query(
        f""" 
        CREATE TABLE IF NOT EXISTS text_embeddings (
            id SERIAL PRIMARY KEY,
            id_chunk INTEGER NOT NULL,
            id_sentence INTEGER NOT NULL,
            embedding FLOAT8[] NOT NULL
        );
        """
    )

    connection.execute_query(
        "SELECT id, chunk_text FROM bookcorpus"
    )
    records = connection.cursor_fetch() 

    print(f'bookcorpus data readed: {len(records)}')

    # Lists to store execution times
    time_to_store_embeddings = []

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    for record in records:
        chunk_id, chunk = record
        
        for sentence_id, sentence in enumerate(chunk):
            # measure time for embedding generation
            start_time = time.time()
            embedding = model.encode(sentence).tolist()  # Generate embedding using SentenceTransformer
            end_time = time.time()
            time_to_store_embeddings.append(end_time - start_time)
            
            # query = "INSERT INTO text_embeddings (id_sentence, embedding) VALUES (%s, %s)"
            # params = (sentence_id, embedding)
            # connection.execute_query(query, params)
            connection.execute_query(
                    "INSERT INTO text_embeddings (id_chunk, id_sentence, embedding) VALUES (%s, %s, %s)",
                    (chunk_id, sentence_id, embedding,)
                )
            

    print("Embeddings loaded into Postgres")

    # Calculate statistics for storing embeddings
    min_time_store_embeddings = np.min(time_to_store_embeddings)
    max_time_store_embeddings = np.max(time_to_store_embeddings)
    avg_time_store_embeddings = np.mean(time_to_store_embeddings)
    std_time_store_embeddings = np.std(time_to_store_embeddings)


    print("\nStatistics for Storing Embeddings:")
    print(f"Min: {min_time_store_embeddings:.6f} s")
    print(f"Max: {max_time_store_embeddings:.6f} s")
    print(f"Avg: {avg_time_store_embeddings:.6f} s")
    print(f"Std: {std_time_store_embeddings:.6f} s")

    connection.close_connection()