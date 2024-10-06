from datasets import load_dataset
from config import load_config
from connect import postgres_conn as postG

# main
if __name__ == '__main__':
    config = load_config()
    connection = postG()
    connection.post_connect(config)
    connection.execute_query(
        f""" 
        CREATE TABLE IF NOT EXISTS bookCorpus (
            id SERIAL PRIMARY KEY,
            sentence TEXT
        );
        """
    )
    
    chunk_size = 1000
    # dataset = load_dataset("bookcorpus", split="train")
    # dataset = load_dataset("bookcorpus", split='train[:10%]', streaming=True)
    dataset = load_dataset("bookcorpus", split=f'train[:{chunk_size}]')
    # dataset = load_dataset("bookcorpus", streaming=True)

    # chunk = dataset.select(range(100))

    # iterate over each sentence of the database and insert into the database
    for i, sentence in enumerate(dataset):
        # print(sentence["text"])
        try:
            connection.execute_query(
                "INSERT INTO bookcorpus (sentence) VALUES (%s)",
                (sentence["text"],)
            )
        except (Exception) as error:
            print(f"Error while executing query: {error}")

    # Commit the transaction
    connection.commit_trans()

    # Close the cursor and connection
    connection.close_connection()

    print(f"Successfully inserted {len(dataset)} sentences into the bookcorpus table.")
