import numpy as np
import os
import re
import json
from sentence_transformers import SentenceTransformer
from lib.search_utils import CACHE_DIR, load_movies

class SemanticSearch:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text.strip():
            raise ValueError("The text cannot be empty.")
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents
        docs_representations = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            docs_representations.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(docs_representations, show_progress_bar = True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        docs_representations = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        else: 
            return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        q_embedding = self.generate_embedding(query)
        cosine_similarities = []
        for i, embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(q_embedding, embedding)
            cosine_similarities.append((similarity_score, self.documents[i]))
        results = sorted(cosine_similarities, key=lambda x: x[0], reverse=True)
        listofdicts = []
        for _, result in enumerate(results[:limit]):
            listofdicts.append({
                "score": result[0],
                "title": result[1]["title"],
                "description": result[1]["description"],
            })
        return listofdicts

def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    movies = load_movies()
    semantic_search.load_or_create_embeddings(movies)
    print(f"Number of docs: {len(semantic_search.documents)}")
    print(f"Embeddings shape: {semantic_search.embeddings.shape[0]} vectors in {semantic_search.embeddings.shape[1]} dimensions")
    return semantic_search

def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search(query, limit):
    semantic_search = verify_embeddings()
    results = semantic_search.search(query, limit)
    for i, movie in enumerate(results, start=1):
        print(f"{i}. {movie['title']} (score: {movie['score']})\n{movie['description']}")

def chunk(text, limit, overlap):
    words = text.split(" ")
    chunks = [
        " ".join(words[(i if i == 0 else i - overlap): i + limit]) 
        for i in range(0, len(words), limit)
    ]
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")

def semantic_chunk(text, max_chunk_size, overlap):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    step = max(1, max_chunk_size - overlap)
    chunks = [
    " ".join(sentences[i : i + max_chunk_size])
    for i in range(0, len(sentences), step)
    ]
    if len(chunks) > 1 and len(sentences) <= max_chunk_size:
        chunks = chunks[:1]

    return chunks

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        self.chunk_metadata = None
        self.chunk_metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        docs_representations = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
        all_chunks = []
        chunks_metadata = []
        for i, doc in enumerate(documents):
            if doc["description"] == "":
                continue
            chunks = semantic_chunk(doc["description"], 4, 1)
            all_chunks = all_chunks + chunks
            for j, chunk in enumerate(chunks):
                chunks_metadata.append({
                    "movie_idx": i,
                    "chunk_idx": j,
                    "total_chunks": len(chunks),
                })
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar = True)
        self.chunk_metadata = chunks_metadata
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w", encoding="utf-8") as f:
            json.dump({"chunks": chunks_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents):
        self.documents = documents
        docs_representations = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
        
        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r", encoding = "utf-8") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings
        else: 
            return self.build_chunk_embeddings(documents)

def embed_chunks():
    movies = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    chunked_semantic_search.load_or_create_chunk_embeddings(movies)
    return chunked_semantic_search.chunk_embeddings