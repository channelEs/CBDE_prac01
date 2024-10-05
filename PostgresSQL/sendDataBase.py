from datasets import load_dataset
from config import load_config
from sentence_transformers import SentenceTransformer
from connect import postgres_conn as postG
import re

# Function to split text into sentences using regular expressions
def split_into_sentences(text):
    # Regular expression to split sentences based on punctuation
    sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s')
    sentences = sentence_endings.split(text)
    return sentences

# main
if __name__ == '__main__':
    config = load_config()
    connection = postG
    connection.connect(config)
    connection.execute_query(
        f""" 
        CREATE TABLE IF NOT EXISTS bookCorpus (
            id SERIAL PRIMARY KEY,
            part_number INTEGER,
            text_part TEXT
        );
        """
    )
    
    # Load BookCorpus dataset from Hugging Face
    dataset = load_dataset("bookcorpus", split="train")

    # Initialize a list to hold the sentences
    all_sentences = []

    # Go through each record in the dataset, split the text into sentences, and add to the list
    for record in dataset:
        sentences = split_into_sentences(record['text'])
        all_sentences.extend(sentences)
        
        # Stop when we have 10,000 sentences
        if len(all_sentences) >= 10000:
            break

    # Slice the list to ensure we only have 1,000 sentences
    all_sentences = all_sentences[:10000]

    # Split the sentences into chunks of 100 sentences each
    chunk_size = 100
    chunks = [all_sentences[i:i + chunk_size] for i in range(0, len(all_sentences), chunk_size)]

    # Iterate over each chunk and insert into the database
    for part_number, chunk in enumerate(chunks, start=1):
        # Combine sentences in each chunk into a single text block
        text_part = ' '.join(chunk)
        
        connection.execute_query(
            "INSERT INTO bookcorpus (part_number, text_part) VALUES (%s, %s)",
            (part_number, text_part)
        )

    # Commit the transaction
    connection.commit_trans()

    # Close the cursor and connection
    connection.close_connection()

    print(f"Successfully inserted {len(chunks)} chunks into the bookcorpus table.")
