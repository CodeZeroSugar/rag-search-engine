from collections import Counter
import json
import math
from pickle import dump, load
from utils.utils import get_data_file, tokenize, PROJECT_ROOT

BM25_K1 = 1.5
BM25_B = 0.75


def bm25_idf_command(term):
    indexer = InvertedIndex()
    try:
        indexer.load()
    except FileNotFoundError:
        print("Index not found. Please build first.")
        return
    return indexer.get_bm25_idf(term)


def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    indexer = InvertedIndex()
    try:
        indexer.load()
    except FileNotFoundError:
        print("Index not found. Please build first.")
        return
    return indexer.get_bm25_tf(doc_id, term, k1, b)


class InvertedIndex:
    def __init__(self):
        self.index = dict()
        self.docmap = dict()
        self.term_frequencies = dict()
        self.doc_lengths = dict()
        self.cache_path = PROJECT_ROOT / "cache"
        self.index_path = self.cache_path / "index.pkl"
        self.map_path = self.cache_path / "docmap.pkl"
        self.freq_path = self.cache_path / "term_frequencies.pkl"
        self.doc_lengths_path = self.cache_path / "doc_lengths.pkl"

    def __add_document(self, doc_id, text):
        tokens = tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        total_lengths = 0
        for value in self.doc_lengths.values():
            total_lengths += value
        return total_lengths / len(self.doc_lengths)

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
        self.cache_path.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as index:
            dump(self.index, index)
        with open(self.map_path, "wb") as docmap:
            dump(self.docmap, docmap)
        with open(self.freq_path, "wb") as term_freq:
            dump(self.term_frequencies, term_freq)
        with open(self.doc_lengths_path, "wb") as doc_lengths:
            dump(self.doc_lengths, doc_lengths)

    def load(self):
        if (
            not self.index_path.exists()
            or not self.map_path.exists()
            or not self.freq_path.exists()
        ):
            raise FileNotFoundError("Cache file not found")
        with open(self.index_path, "rb") as f:
            self.index = load(f)
        with open(self.map_path, "rb") as f:
            self.docmap = load(f)
        with open(self.freq_path, "rb") as f:
            self.term_frequencies = load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = load(f)

    def get_tf(self, doc_id, term):
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise Exception("get_tf only accepts 1 term")
        token = tokens[0]
        if self.term_frequencies[doc_id].get(token) is None:
            return 0
        return self.term_frequencies[doc_id][token]

    def get_idf(self, term):
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise Exception("get_tf only accepts 1 term")
        token = tokens[0]
        total_docs = len(self.docmap)
        total_match = len(self.get_documents(token))
        idf = math.log((total_docs + 1) / (total_match + 1))
        return idf

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise Exception("get_bm25_idf only accepts 1 term")
        token = tokens[0]
        n = len(self.docmap)
        df = len(self.get_documents(token))
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id, term):
        bm25tf = self.get_bm25_tf(doc_id, term)
        bm25idf = self.get_bm25_idf(term)
        return bm25tf * bm25idf

    def bm25_search(self, query, limit=5):
        tokens = tokenize(query)
        scores_dict = dict()
        for id in self.docmap:
            running_score = 0
            for token in tokens:
                running_score += self.bm25(id, token)
            scores_dict[id] = running_score
        sort_scores = sorted(
            scores_dict.items(), key=lambda item: item[1], reverse=True
        )
        return sort_scores[:limit]
