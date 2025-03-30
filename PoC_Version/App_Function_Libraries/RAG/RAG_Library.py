import numpy as np
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import math
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import openai
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
import psycopg2
from psycopg2.extras import execute_values
import sqlite3
#
########################################################################################################################################################################################################################################
#
# RAG Chunking
# To fully integrate this chunking system, you'd need to:
#
# Create the UnvectorizedMediaChunks table in your SQLite database.
# Modify your document ingestion process to use chunk_and_store_unvectorized.
# Implement a background process that periodically calls vectorize_all_documents to process unvectorized chunks.

# This chunking is pretty weak and needs improvement
# See notes for improvements #FIXME
import json
from typing import List, Dict, Any
from datetime import datetime


def chunk_and_store_unvectorized(
        db_connection,
        media_id: int,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 100,
        chunk_type: str = 'fixed-length'
) -> List[int]:
    chunks = create_chunks(text, chunk_size, overlap)
    return store_unvectorized_chunks(db_connection, media_id, chunks, chunk_type)


def create_chunks(text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_text = ' '.join(words[i:i + chunk_size])
        start_char = text.index(words[i])
        end_char = start_char + len(chunk_text)
        chunks.append({
            'text': chunk_text,
            'start_char': start_char,
            'end_char': end_char,
            'index': len(chunks)
        })
    return chunks


def store_unvectorized_chunks(
        db_connection,
        media_id: int,
        chunks: List[Dict[str, Any]],
        chunk_type: str
) -> List[int]:
    cursor = db_connection.cursor()
    chunk_ids = []
    for chunk in chunks:
        cursor.execute("""
            INSERT INTO UnvectorizedMediaChunks 
            (media_id, chunk_text, chunk_index, start_char, end_char, chunk_type, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            media_id,
            chunk['text'],
            chunk['index'],
            chunk['start_char'],
            chunk['end_char'],
            chunk_type,
            json.dumps({'length': len(chunk['text'])})  # Example metadata
        ))
        chunk_ids.append(cursor.lastrowid)
    db_connection.commit()
    return chunk_ids


def get_unvectorized_chunks(
        db_connection,
        media_id: int,
        limit: int = 100,
        offset: int = 0
) -> List[Dict[str, Any]]:
    cursor = db_connection.cursor()
    cursor.execute("""
        SELECT id, chunk_text, chunk_index, start_char, end_char, chunk_type, metadata
        FROM UnvectorizedMediaChunks
        WHERE media_id = ? AND is_processed = FALSE
        ORDER BY chunk_index
        LIMIT ? OFFSET ?
    """, (media_id, limit, offset))
    return [
        {
            'id': row[0],
            'text': row[1],
            'index': row[2],
            'start_char': row[3],
            'end_char': row[4],
            'type': row[5],
            'metadata': json.loads(row[6])
        }
        for row in cursor.fetchall()
    ]


def mark_chunks_as_processed(db_connection, chunk_ids: List[int]):
    cursor = db_connection.cursor()
    cursor.executemany("""
        UPDATE UnvectorizedMediaChunks
        SET is_processed = TRUE, last_modified = ?
        WHERE id = ?
    """, [(datetime.now(), chunk_id) for chunk_id in chunk_ids])
    db_connection.commit()


# Usage example
def process_media_chunks(db_connection, media_id: int, text: str):
    chunk_ids = chunk_and_store_unvectorized(db_connection, media_id, text)
    print(f"Stored {len(chunk_ids)} unvectorized chunks for media_id {media_id}")

    # Later, when you want to process these chunks:
    unprocessed_chunks = get_unvectorized_chunks(db_connection, media_id)
    # Process chunks (e.g., vectorize them)
    # ...
    # After processing, mark them as processed
    mark_chunks_as_processed(db_connection, [chunk['id'] for chunk in unprocessed_chunks])
###########################################################################################################################################################################################################
#
# RAG System

# To use this updated RAG system in your existing application:
#
# Install required packages:
# pip install sentence-transformers psycopg2-binary scikit-learn transformers torch
# Set up PostgreSQL with pgvector:
#
# Install PostgreSQL and the pgvector extension.
# Create a new database for vector storage.
#
# Update your main application to use the RAG system:
#
# Import the RAGSystem class from this new file.
# Initialize the RAG system with your SQLite and PostgreSQL configurations.
# Use the vectorize_all_documents method to initially vectorize your existing documents.
#
#
# Modify your existing PDF_Ingestion_Lib.py and Book_Ingestion_Lib.py:
#
# After successfully ingesting a document into SQLite, call the vectorization method from the RAG system.

# Example modification for ingest_text_file in Book_Ingestion_Lib.py:
# from RAG_Library import RAGSystem
#
# # Initialize RAG system (do this once in your main application)
# rag_system = RAGSystem(sqlite_path, pg_config)
#
# def ingest_text_file(file_path, title=None, author=None, keywords=None):
#     try:
#         # ... (existing code)
#
#         # Add the text file to the database
#         doc_id = add_media_with_keywords(
#             url=file_path,
#             title=title,
#             media_type='document',
#             content=content,
#             keywords=keywords,
#             prompt='No prompt for text files',
#             summary='No summary for text files',
#             transcription_model='None',
#             author=author,
#             ingestion_date=datetime.now().strftime('%Y-%m-%d')
#         )
#
#         # Vectorize the newly added document
#         rag_system.vectorize_document(doc_id, content)
#
#         return f"Text file '{title}' by {author} ingested and vectorized successfully."
#     except Exception as e:
#         logging.error(f"Error ingesting text file: {str(e)}")
#         return f"Error ingesting text file: {str(e)}"

# Constants
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
VECTOR_DIM = 384  # Dimension of the chosen embedding model


class RAGSystem:
    def __init__(self, sqlite_path: str, pg_config: Dict[str, str], cache_size: int = 100):
        self.sqlite_path = sqlite_path
        self.pg_config = pg_config
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.cache_size = cache_size

        self._init_postgres()

    def _init_postgres(self):
        with psycopg2.connect(**self.pg_config) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                CREATE TABLE IF NOT EXISTS document_vectors (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER UNIQUE,
                    vector vector(384)
                )
                """)
            conn.commit()

    @lru_cache(maxsize=100)
    def _get_embedding(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0]

    def vectorize_document(self, doc_id: int, content: str):
        chunks = create_chunks(content, chunk_size=1000, overlap=100)
        for chunk in chunks:
            vector = self._get_embedding(chunk['text'])

            with psycopg2.connect(**self.pg_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                    INSERT INTO document_vectors (document_id, chunk_index, vector, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (document_id, chunk_index) DO UPDATE SET vector = EXCLUDED.vector
                    """, (doc_id, chunk['index'], vector.tolist(), json.dumps(chunk)))
                conn.commit()

    def vectorize_all_documents(self):
        with sqlite3.connect(self.sqlite_path) as sqlite_conn:
            unprocessed_chunks = get_unvectorized_chunks(sqlite_conn, limit=1000)
            for chunk in unprocessed_chunks:
                self.vectorize_document(chunk['id'], chunk['text'])
            mark_chunks_as_processed(sqlite_conn, [chunk['id'] for chunk in unprocessed_chunks])

    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[int, int, float]]:
        query_vector = self._get_embedding(query)

        with psycopg2.connect(**self.pg_config) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                SELECT document_id, chunk_index, 1 - (vector <-> %s) AS similarity
                FROM document_vectors
                ORDER BY vector <-> %s ASC
                LIMIT %s
                """, (query_vector.tolist(), query_vector.tolist(), top_k))
                results = cur.fetchall()

        return results

    def get_document_content(self, doc_id: int) -> str:
        with sqlite3.connect(self.sqlite_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT content FROM media WHERE id = ?", (doc_id,))
            result = cur.fetchone()
            return result[0] if result else ""

    def bm25_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        with sqlite3.connect(self.sqlite_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, content FROM media")
            documents = cur.fetchall()

        vectorizer = TfidfVectorizer(use_idf=True)
        tfidf_matrix = vectorizer.fit_transform([doc[1] for doc in documents])

        query_vector = vectorizer.transform([query])
        doc_lengths = tfidf_matrix.sum(axis=1).A1
        avg_doc_length = np.mean(doc_lengths)

        k1, b = 1.5, 0.75
        scores = []
        for i, doc_vector in enumerate(tfidf_matrix):
            score = np.sum(
                ((k1 + 1) * query_vector.multiply(doc_vector)).A1 /
                (k1 * (1 - b + b * doc_lengths[i] / avg_doc_length) + query_vector.multiply(doc_vector).A1)
            )
            scores.append((documents[i][0], score))

        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    def combine_search_results(self, bm25_results: List[Tuple[int, float]], vector_results: List[Tuple[int, float]],
                               alpha: float = 0.5) -> List[Tuple[int, float]]:
        combined_scores = {}
        for idx, score in bm25_results + vector_results:
            if idx in combined_scores:
                combined_scores[idx] += score * (alpha if idx in dict(bm25_results) else (1 - alpha))
            else:
                combined_scores[idx] = score * (alpha if idx in dict(bm25_results) else (1 - alpha))
        return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    def expand_query(self, query: str) -> str:
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")

        input_text = f"expand query: {query}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
        expanded_query = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return f"{query} {expanded_query}"

    def cross_encoder_rerank(self, query: str, initial_results: List[Tuple[int, float]], top_k: int = 5) -> List[
        Tuple[int, float]]:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        candidate_docs = [self.get_document_content(doc_id) for doc_id, _ in initial_results[:top_k * 2]]
        pairs = [[query, doc] for doc in candidate_docs]
        scores = model.predict(pairs)

        reranked = sorted(zip(initial_results[:top_k * 2], scores), key=lambda x: x[1], reverse=True)
        return [(idx, score) for (idx, _), score in reranked[:top_k]]

    def rag_query(self, query: str, search_type: str = 'combined', top_k: int = 5, use_hyde: bool = False,
                  rerank: bool = False, expand: bool = False) -> List[Dict[str, any]]:
        try:
            if expand:
                query = self.expand_query(query)

            if use_hyde:
                # Implement HyDE if needed
                pass
            elif search_type == 'vector':
                results = self.semantic_search(query, top_k)
            elif search_type == 'bm25':
                results = self.bm25_search(query, top_k)
            elif search_type == 'combined':
                bm25_results = self.bm25_search(query, top_k)
                vector_results = self.semantic_search(query, top_k)
                results = self.combine_search_results(bm25_results, vector_results)
            else:
                raise ValueError("Invalid search type. Choose 'vector', 'bm25', or 'combined'.")

            if rerank:
                results = self.cross_encoder_rerank(query, results, top_k)

            enriched_results = []
            for doc_id, score in results:
                content = self.get_document_content(doc_id)
                enriched_results.append({
                    "document_id": doc_id,
                    "score": score,
                    "content": content[:500]  # Truncate content for brevity
                })

            return enriched_results
        except Exception as e:
            logger.error(f"An error occurred during RAG query: {str(e)}")
            return []


# Example usage
if __name__ == "__main__":
    sqlite_path = "path/to/your/sqlite/database.db"
    pg_config = {
        "dbname": "your_db_name",
        "user": "your_username",
        "password": "your_password",
        "host": "localhost"
    }

    rag_system = RAGSystem(sqlite_path, pg_config)

    # Vectorize all documents (run this once or periodically)
    rag_system.vectorize_all_documents()

    # Example query
    query = "programming concepts for beginners"
    results = rag_system.rag_query(query, search_type='combined', expand=True, rerank=True)

    print(f"Search results for query: '{query}'\n")
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Document ID: {result['document_id']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Content snippet: {result['content']}")
        print("---")