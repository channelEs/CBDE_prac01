from connect import postgres_conn as postG
from config import load_config
import numpy as np
import time

#main
if __name__ == '__main__':
    config = load_config()
    connection = postG()
    connection.post_connect(config)


    create_table_query = """
    CREATE TABLE IF NOT EXISTS pgVec_sentences (
        id SERIAL PRIMARY KEY,
        sentence TEXT,
        embedding VECTOR(384)
    )
    """
    connection.execute_query(create_table_query)


    connection.execute_query(
        "SELECT id, chunk_text FROM bookcorpus"
    )
    records = connection.cursor_fetch() 
    print(f'bookcorpus data readed: {len(records)}')

    # Lists to store execution times
    time_to_store_text = []

    # Step 4: Insert sentences into the pgvector table
    for chunk_id, chunk in records:
        
        for sentence_id, sentence in enumerate(chunk):
            start_time = time.time()
            connection.execute_query(
                    "INSERT INTO pgVec_sentences (sentence) VALUES (%s)",
                    (sentence,)
                )
            end_time = time.time()
            time_to_store_text.append(end_time - start_time)
            
    print("Data loaded into Postgres with PGVec")

    # Calculate statistics for storing sentences
    min_time_store_embeddings = np.min(time_to_store_text)
    max_time_store_embeddings = np.max(time_to_store_text)
    avg_time_store_embeddings = np.mean(time_to_store_text)
    std_time_store_embeddings = np.std(time_to_store_text)


    print("\nStatistics for Storing Embeddings:")
    print(f"Min: {min_time_store_embeddings:.6f} s")
    print(f"Max: {max_time_store_embeddings:.6f} s")
    print(f"Avg: {avg_time_store_embeddings:.6f} s")
    print(f"Std: {std_time_store_embeddings:.6f} s")

    connection.close_connection()