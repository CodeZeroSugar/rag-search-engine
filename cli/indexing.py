import json
import os
from pickle import dump, load
from pathlib import Path
from utils import get_data_file, tokenize


class InvertedIndex:
    def __init__(self):
        self.index = dict()
        self.docmap = dict()

    def __add_document(self, doc_id, text):
        tokens = tokenize(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term):
        term_lower = term.lower()
        if term_lower not in self.index:
            print(f"{term} not found in index")
            return []
        doc_ids = list(self.index[term_lower])
        return sorted(doc_ids)

    def build(self):
        with open(get_data_file("movies.json"), "r") as f:
            data = json.load(f)
        for movie in data["movies"]:
            input_text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"], input_text)
            self.docmap[movie["id"]] = movie

    def save(self):
        base_dir = Path(__file__).resolve().parent
        cache_path = base_dir.parent / "cache"
        cache_path.mkdir(parents=True, exist_ok=True)
        with open(cache_path / "index.pkl", "wb") as index:
            dump(self.index, index)
        with open(cache_path / "docmap.pkl", "wb") as docmap:
            dump(self.docmap, docmap)

    def load(self):
        base_dir = Path(__file__).resolve().parent
        cache_path = base_dir.parent / "cache"
        index_path = cache_path / "index.pkl"
        map_path = cache_path / "docmap.pkl"
        if not index_path.exists() or not map_path.exists():
            raise FileNotFoundError("Cache file not found")
        with open(index_path, "rb") as f:
            index = load(f)
        with open(map_path, "rb") as f:
            docmap = load(f)
        return index, docmap
