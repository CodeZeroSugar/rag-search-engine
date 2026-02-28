from collections import Counter
import json
from pickle import dump, load
from pathlib import Path
from utils import get_data_file, tokenize


class InvertedIndex:
    def __init__(self):
        self.index = dict()
        self.docmap = dict()
        self.term_frequencies = dict()

    def __add_document(self, doc_id, text):
        tokens = tokenize(text)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

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
        with open(cache_path / "term_frequencies.pkl", "wb") as term_freq:
            dump(self.term_frequencies, term_freq)

    def load(self):
        base_dir = Path(__file__).resolve().parent
        cache_path = base_dir.parent / "cache"
        index_path = cache_path / "index.pkl"
        map_path = cache_path / "docmap.pkl"
        freq_path = cache_path / "term_frequencies.pkl"
        if not index_path.exists() or not map_path.exists() or not freq_path.exists():
            raise FileNotFoundError("Cache file not found")
        with open(index_path, "rb") as f:
            self.index = load(f)
        with open(map_path, "rb") as f:
            self.docmap = load(f)
        with open(freq_path, "rb") as f:
            self.term_frequencies = load(f)

    def get_tf(self, doc_id, term):
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise Exception("get_tf only accepts 1 term")
        token = tokens[0]
        if self.term_frequencies[doc_id].get(token) is None:
            return 0
        return self.term_frequencies[doc_id][token]
