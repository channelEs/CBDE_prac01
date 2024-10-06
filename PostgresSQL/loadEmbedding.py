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
            id_sentence INTEGER NOT NULL,
            embedding FLOAT8[] NOT NULL
        );
        """
    )

    connection.execute_query(
        "SELECT id, sentence FROM bookcorpus"
    )
    records = connection.cursor_fetch() 

    print(f'bookcorpus data readed: {len(records)}')

    # Lists to store execution times
    time_to_store_text = []
    time_to_store_embeddings = []

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    for record in records:
        sentence_id, sentence = record
        
        # measure time for embedding generation
        start_time = time.time()
        embedding = model.encode(sentence).tolist()  # Generate embedding using SentenceTransformer
        end_time = time.time()
        time_to_store_embeddings.append(end_time - start_time)

        # measure time for storing text and embedding
        start_time = time.time()
        # query = "INSERT INTO text_embeddings (id_sentence, embedding) VALUES (%s, %s)"
        # params = (sentence_id, embedding)
        # connection.execute_query(query, params)
        connection.execute_query(
                "INSERT INTO text_embeddings (id_sentence, embedding) VALUES (%s, %s)",
                (sentence_id, embedding,)
            )
        
        end_time = time.time()
        time_to_store_text.append(end_time - start_time)

    print("Data loaded into Postgres")

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