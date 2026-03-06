import json
import re
import numpy as np


from utils.utils import PROJECT_ROOT, get_data_file, clean_text
from sentence_transformers import SentenceTransformer


def verify_model():
    model = SemanticSearch()

    print(f"Model loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")


def verify_embeddings():
    model = SemanticSearch()
    with open(get_data_file("movies.json"), "r") as f:
        data = json.load(f)
    documents = []
    for movie in data["movies"]:
        documents.append(movie)
    embeddings = model.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def embed_query_text(query):
    model = SemanticSearch()
    embedding = model.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def embed_text(text):
    model = SemanticSearch()
    embedding = model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def overlap_chunk(words, overlap, chunk_size):
    chunks = []
    i = 0
    while chunk_size < len(words):
        chunks.append(" ".join(words[i:chunk_size]))
        i = chunk_size - overlap
        chunk_size = i + chunk_size

    if i < len(words):
        chunks.append(" ".join(words[i:]))

    return chunks


def semantic_chunk(text, max_chunk_size, overlap):
    chunks = clean_text(text)
    final_chunks = []
    chunk = []
    i = 0
    while i < len(chunks):
        chunk.append(chunks[i])
        if len(chunk) == max_chunk_size:
            final_chunks.append(" ".join(chunk))
            chunk = []
            i += 1
            i -= overlap
        else:
            i += 1

    if len(chunk) > overlap:
        final_chunks.append(" ".join(chunk))

    return final_chunks


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = dict()

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("text required to generate embedding")
        input = []
        input.append(text)
        embedding = self.model.encode(input)
        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        string_docs = []
        for doc_id in self.document_map:
            string_docs.append(
                f"{self.document_map[doc_id]['title']}: {self.document_map[doc_id]['description']}"
            )

        self.embeddings = self.model.encode(string_docs, show_progress_bar=False)
        with open(PROJECT_ROOT / "cache" / "movie_embeddings.npy", "wb") as f:
            np.save(f, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        embed_path = PROJECT_ROOT / "cache" / "movie_embeddings.npy"
        if embed_path.exists():
            with open(embed_path, "rb") as f:
                self.embeddings = np.load(f)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        query_embedding = self.generate_embedding(query)
        sim_score_docs = []
        for i in range(len(self.embeddings)):
            cos_similarilty = cosine_similarity(self.embeddings[i], query_embedding)
            sim_score_docs.append((cos_similarilty, self.documents[i]))
        sorted_docs = sorted(sim_score_docs, key=lambda t: t[0], reverse=True)
        top_results = []
        for i in range(limit):
            top_results.append(
                {
                    "score": sorted_docs[i][0],
                    "title": sorted_docs[i][1]["title"],
                    "description": sorted_docs[i][1]["description"],
                }
            )
        return top_results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
        all_chunks = []
        chunk_meta = []
        for i in range(len(self.documents)):
            if self.document_map[self.documents[i]["id"]]["description"] is None:
                continue
            chunks = semantic_chunk(
                self.document_map[self.documents[i]["id"]]["description"], 4, 1
            )
            for j in range(len(chunks)):
                all_chunks.append(chunks[j])
                chunk_meta.append(
                    {"movie_idx": i, "chunk_idx": j, "total_chunks": len(chunks)}
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=False)
        self.chunk_metadata = chunk_meta
        with open(PROJECT_ROOT / "cache" / "chunk_embeddings.npy", "wb") as f:
            np.save(f, self.chunk_embeddings)
        with open(PROJECT_ROOT / "cache" / "chunk_metadata.json", "w") as f:
            json.dump(chunk_meta, f)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        embed_path = PROJECT_ROOT / "cache" / "chunk_embeddings.npy"
        meta_path = PROJECT_ROOT / "cache" / "chunk_metadata.json"
        if embed_path.exists() and meta_path.exists():
            with open(embed_path, "rb") as f:
                self.chunk_embeddings = np.load(f)
            with open(meta_path, "r") as f:
                self.chunk_metadata = json.load(f)
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit=10):
        query_embedding = self.generate_embedding(query)
        chunk_scores = []
        for i in range(len(self.chunk_embeddings)):
            cos_similarity = cosine_similarity(
                query_embedding, self.chunk_embeddings[i]
            )
            chunk_scores.append(
                {
                    "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                    "movie_idx": self.chunk_metadata[i]["movie_idx"],
                    "score": cos_similarity,
                }
            )
        movie_idx_scores = dict()
        for score in chunk_scores:
            if (
                score["movie_idx"] not in movie_idx_scores
                or score["score"] > movie_idx_scores[score["movie_idx"]]
            ):
                movie_idx_scores[score["movie_idx"]] = score["score"]
        sorted_scores = sorted(
            movie_idx_scores.items(), key=lambda x: x[1], reverse=True
        )
        limit_scores = sorted_scores[:limit]
        final_results = []
        for score in limit_scores:
            final_results.append(
                {
                    "id": self.documents[score[0]]["id"],
                    "title": self.documents[score[0]]["title"],
                    "document": self.documents[score[0]]["description"][:100],
                    "score": round(score[1], 2),
                    "metadata": self.documents[score[0]] or {},
                }
            )
        return final_results
