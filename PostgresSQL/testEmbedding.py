from config import load_config
from connect import postgres_conn as postG
from sentence_transformers import SentenceTransformer
import numpy as np
import time

# main
if __name__ == '__main__':
    config = load_config()
    connection = postG()
    connection.post_connect(config)

    # connection.execute_query(
    #     "SELECT id_chunk, id_sentence, embedding FROM text_embeddings"
    # )
    # embedding_store = connection.cursor_fetch() 
    # print(f'data readed: {len(embedding_store)}')

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

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    similar_embbedings = []
    time_to_store_top2 = []

    # compute the top-2 most similar sentences
    for sample_sentence in sample_sentences:
        sample_embedding = model.encode(sample_sentence).tolist()
        similarity_scores = []
        new_query = (

        '''
            WITH embedding_similarity AS (
            SELECT
                id,
                id_chunk,
                id_sentence,
                embedding,
                    -- Calculate the dot product between your input vector and the database embeddings
                    (SELECT SUM(a * b) FROM unnest(embedding) WITH ORDINALITY AS emb1(a, i)
                    JOIN unnest(ARRAY[%s]) WITH ORDINALITY AS emb2(b, j)
                    ON emb1.i = emb2.j) AS dot_product,
                    -- Calculate the norm (magnitude) of the database embedding
                    SQRT(SUM(POW(e, 2))) AS norm_db_embedding,
                    -- Calculate the norm (magnitude) of the input embedding
                    SQRT(SUM(POW(x, 2))) AS norm_input_embedding
                FROM text_embeddings
                CROSS JOIN LATERAL unnest(embedding) AS e
                CROSS JOIN LATERAL unnest(ARRAY[%s]) AS x
                GROUP BY id, id_chunk, id_sentence, embedding
            )
            SELECT
                id,
                id_chunk,
                id_sentence,
                dot_product / (norm_db_embedding * norm_input_embedding) AS cosine_similarity
            FROM embedding_similarity
            ORDER BY cosine_similarity DESC
            LIMIT 2;
        '''
        )


        start_time = time.time()
        for chunk_id, sentence_id, stored_embedding in embedding_store:
            stored_embedding_np = np.array(stored_embedding)
            similarity = np.dot(sample_embedding, stored_embedding_np) / (np.linalg.norm(sample_embedding) * np.linalg.norm(stored_embedding_np))
            similarity_scores.append((chunk_id, sentence_id, similarity))

        similarity_scores = sorted(similarity_scores, key=lambda x: x[1])
        similar_embbedings.append((similarity_scores[:2]))
        end_time = time.time()

        time_to_store_top2.append(end_time - start_time)        

    for count, similar_embbeding in enumerate(similar_embbedings):
        similar_sentence = []
        for chunk_id, id_sentence, similarity_value in similar_embbeding:
            connection.execute_query(
            f'SELECT sentence FROM text_embeddings WHERE id_chunk = {chunk_id} AND id_sentence = {id_sentence}'
            )
            sentence = connection.cursor_fetch()
            similar_sentence.append(sentence)
        print(f'FOR SENTENCE: {sample_sentences[count]}:')
        print(f'similarity_01: {similar_sentence[0]}')
        print(f'similarity_02: {similar_sentence[1]}')
        print()

    connection.close_connection()

    # calculate times
    min_time_store_text = np.min(time_to_store_top2)
    max_time_store_text = np.max(time_to_store_top2)
    avg_time_store_text = np.mean(time_to_store_top2)
    std_time_store_text = np.std(time_to_store_top2)

    # print times
    print("\nStatistics for Storing Text:")
    print(f"Min: {min_time_store_text:.6f} s")
    print(f"Max: {max_time_store_text:.6f} s")
    print(f"Avg: {avg_time_store_text:.6f} s")
    print(f"Std: {std_time_store_text:.6f} s")