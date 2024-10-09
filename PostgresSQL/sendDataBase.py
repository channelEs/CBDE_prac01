from datasets import load_dataset
from config import load_config
from connect import postgres_conn as postG
import numpy as np
import time

# main
if __name__ == '__main__':
    config = load_config()
    connection = postG()
    connection.post_connect(config)
    connection.execute_query(
        f""" 
        CREATE TABLE IF NOT EXISTS bookCorpus (
            id SERIAL PRIMARY KEY,
            chunk_text TEXT []
        );
        """
    )
    
    total_size = 10000
    chunk_size = 100
    # dataset = load_dataset("bookcorpus", split="train")
    # dataset = load_dataset("bookcorpus", split='train[:10%]', streaming=True)
    dataset = load_dataset("bookcorpus", split=f'train[:{total_size}]')
    chunks = [dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]

    print(len(chunks))

    time_to_store_text = []
    for chunk in chunks:
        start_time = time.time()
        chunk_text = [sentence for sentence in chunk['text']]
        try:
            connection.execute_query(
                "INSERT INTO bookcorpus (chunk_text) VALUES (%s)",
                (chunk_text,)
            )
        except (Exception) as error:
            print(f"Error while executing query: {error}")
        
        end_time = time.time()
        time_to_store_text.append(end_time - start_time)

    connection.commit_trans()

    # Close the cursor and connection
    connection.close_connection()

    print(f"Successfully inserted {len(dataset)} sentences into the bookcorpus table.")

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