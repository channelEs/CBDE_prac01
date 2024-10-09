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

    # create_table_query = """
    # CREATE TABLE IF NOT EXISTS pgVec_embeddings (
    #     id SERIAL PRIMARY KEY,
    #     id_sentence INTEGER NOT NULL,
    #     embedding VECTOR(384)
    # )
    # """
    # connection.execute_query(create_table_query)

    connection.execute_query(
        "SELECT id, sentence FROM pgVec_sentences"
    )
    records = connection.cursor_fetch() 
    print(f'PGvec bookcorpus data readed: {len(records)}')

    # Lists to store execution times
    time_to_store_embedding = []
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Step 4: Insert sentences into the pgvector table
    for sentence_id, sentence in enumerate(records):
        start_time = time.time()
        embedding = model.encode(sentence).tolist()
        end_time = time.time()
        time_to_store_embedding.append(end_time - start_time)
        
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'

        # print(len(embedding))
        # print(len(embedding[0]))

        # print(sentence_id)
        connection.execute_query(
                "UPDATE pgVec_sentences SET embedding = %s WHERE id = %s;",
                (embedding[0], sentence_id,)
            )
                    
    print("Embeddings updated into Postgres with PGVec")

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