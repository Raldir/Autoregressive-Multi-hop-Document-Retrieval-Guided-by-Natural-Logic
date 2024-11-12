import logging
import json
from pyserini.search.lucene import LuceneSearcher
import os
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import dataset_readers
import numpy as np
import re
import regex

logger = logging.getLogger(__name__)


class SimpleTokenizer:
    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"

    def __init__(self):
        """
        Initializes the tokenizer with regex pattern for tokenizing words.
        """
        self._regexp = regex.compile(
            "(%s)|(%s)" % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text):
        # Extract only the tokens (ignoring whitespace and spans)
        return [m.group() for m in self._regexp.finditer(text)]


class BM25(object):

    def __init__(self, config, current_iteration, granularity):
        self.config = config
        self.current_iteration = current_iteration
        self.reader = dataset_readers.get_class(config.dataset)(**{'granularity':"paragraph", "config": config}) # Not really elegant since we have two searchers here overall. TODO: Fix.
        self.searcher = LuceneSearcher(config.pyserini_index_path_passage)
        self.searcher.set_bm25(self.config.bm25_document_k1, self.config.bm25_document_b)

        if not os.path.exists(self.config.genre_title_path):
            db_titles = {"titles": [self.searcher.doc(i).docid() for i in range(self.searcher.num_docs)]}
            with open(self.config.genre_title_path, "w", encoding="utf-8") as f_out:
                json.dump(db_titles, f_out)
            self.db_titles = db_titles["titles"]
        else:
            with open(self.config.genre_title_path, "r") as f_in:
                self.db_titles = json.load(f_in)["titles"]


    def retrieve_documents(self, qid, claim):
        # print(claim)
        hits = self.searcher.search(claim, k=self.config.bm25_num_retrieved_docs)
        hits_formatted = []

        for i in range(0, len(hits)):
            doc_dict = {"qid": qid, "doc":hits[i].docid, "score": hits[i].score, "rank":i+1, "retriever": "bm25"}
            # print(i)
            hits_formatted.append(doc_dict)
        return hits_formatted
    
    def save_documents_to_file(self, retrieved_documents, current_iteration):
        with open(self.config.bm25_pyserini_dev_docs.replace("{ITER}", str(current_iteration)), "w") as f_out:
            for sample in retrieved_documents:
                for i,hit in enumerate(sample):
                    curr_str = "{}\t{}\t{}\t{}\t{:.5f}\t{}".format(hit["qid"], "XX", hit["doc"],  hit["rank"], hit["score"], hit["retriever"])
                    f_out.write("{}\n".format(curr_str))
    