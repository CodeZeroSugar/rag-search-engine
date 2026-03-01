import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from sentence_transformers import SentenceTransformer


def verify_model():
    model = SemanticSearch()

    print(f"Model loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")


def embed_text(text):
    model = SemanticSearch()
    embedding = model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("text required to generate embedding")
        input = []
        input.append(text)
        embedding = self.model.encode(input)
        return embedding[0]
