import os

from cli.utils.utils import normalize
from index_search import InvertedIndex
from semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_search_results = self.semantic_search.search(query, limit * 500)
        bm25_scores = []
        for item in bm25_results:
            bm25_scores.append(item[1])
        normalized_bm25_scores = normalize(bm25_scores)

        semantic_scores = []
        for item in semantic_search_results:
            semantic_scores.append(item["score"])
        normalized_semantic_scores = normalize(semantic_scores)

        doc_dict = dict()
        for i in range(len(bm25_results)):
            doc_dict[bm25_results[i][0]] = self.documents[bm25_results[i][0]]

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
