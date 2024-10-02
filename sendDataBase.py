from config import load_config
from sentence_transformers import SentenceTransformer
import psycopg2

# Connection to PostGres DataBase
def connect(config):
    """ Connect to the PostgreSQL database server """
    try:
        # connecting to the PostgreSQL server
        with psycopg2.connect(**config) as conn:
            print('Connected to the PostgreSQL server.')
            return conn
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)

# split text
def splitText(text, chunk_size=500):
    """
    Splits the text into chunks of approximately `chunk_size` tokens.
    """
    sentences = text.split('. ')
    chunks = []
    current_chunk = []

    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence.split())

        if current_length + sentence_length > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# main
if __name__ == '__main__':
    config = load_config()
    connection = connect(config)
    
    # table creation
    try:
        cursor = connection.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS bookCorpus (
            id SERIAL PRIMARY KEY,
            part_number INTEGER,
            text_part TEXT
        );
        """
        cursor.execute(create_table_query)
        connection.commit()
        print("Table created successfully.")
    except Exception as e:
        print(f"Error creating table: {e}")

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    text = """
    Your large text goes here. It can be multiple paragraphs or a long piece of content that needs to be split into chunks and processed.
    """
    # Split text into chunks
    chunks = splitText(text, chunk_size=100)

    embeddings = model.encode(chunks)

