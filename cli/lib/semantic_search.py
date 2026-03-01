import os
import json
import numpy as np

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from utils.utils import PROJECT_ROOT, get_data_file
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

        self.embeddings = self.model.encode(string_docs, show_progress_bar=True)
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
