
import os
import argparse
import json
import copy
from tqdm import tqdm

from .dataset_reader import RetrievalDataset
from pyserini.search import SimpleSearcher
from pyserini.search.lucene import LuceneSearcher

import unicodedata
import re
import numpy as np

from src.utils import drqa_tokenizers

class FeverDatasetReader(RetrievalDataset):

    def __init__(self, **kwargs):
        self.granularity = kwargs.get('granularity')
        config = kwargs.get('config')
        if config:
            self.searcher = LuceneSearcher(config.pyserini_index_path_pipeline)


    def process_sentence(self, sentence):
        sentence = re.sub(" \-LSB\-.*?\-RSB\-", "", sentence)
        sentence = re.sub("\-LRB\- \-RRB\- ", "", sentence)
        sentence = re.sub(" -LRB-", " ( ", sentence)
        sentence = re.sub("-RRB-", " )", sentence)
        sentence = re.sub("--", "-", sentence)
        sentence = re.sub("``", '"', sentence)
        sentence = re.sub("''", '"', sentence)
        return sentence

    def process_sentence_reverse(self, sentence):
        # sentence = re.sub("-", "--", sentence)
        sentence = re.sub(" \( ", " -LRB-", sentence)
        sentence = re.sub(" \)", "-RRB-", sentence)
        sentence = re.sub('"', "''", sentence)
        sentence = re.sub('"', "``", sentence)
        return sentence

    def process_title_reverse_genre(self, title):
        title = re.sub(" ", "_", title)
        title = re.sub("\(", "-LRB-", title)
        title = re.sub("\)", "-RRB-", title)
        title = re.sub(":", "-COLON-", title)
        return title
        
    def process_title(self, title):
        title = re.sub("_", " ", title)
        title = re.sub(" -LRB-", " ( ", title)
        title = re.sub("-RRB-", " )", title)
        title = re.sub("-COLON-", ":", title)
        return title

    def process_title_reverse(self, title):
        title = re.sub(" ", "_", title)
        title = re.sub(" \( ", "-LRB-", title)
        title = re.sub(" \)", "-RRB-", title)
        title = re.sub(":", "-COLON-", title)
        return title

    def read_proofver_proof_train(self, input_file):
        natural_operations_sequences = []
        proofs = []
        with open(input_file, 'r', encoding = 'utf-8') as f:
            lines = f.readlines()
            for line in lines:
                content = line
                proof = content.strip()
                natural_operations_sequences.append(proof)
        for sequence in natural_operations_sequences:
            matches = re.findall(r"\{ (.+?) \} \[ (.+?) \] (.+?)", sequence)
            curr_proof = []
            for match in matches:
                assert len(match) == 3, f"Does not match, got {match}"
                assert match[2] in ['<', '>', '!', '=', '|', '#']
                curr_proof.append(match)
            proofs.append(curr_proof)
        return proofs

    def read_annotations(self, input_path):
        with open(input_path, 'r', encoding='utf-8') as f_in:
            # open qrels file if provided

            for line in f_in:
                line_json = json.loads(line.strip())
                qid = line_json['id']
                query = line_json['claim']
                if 'label' in line_json:  # no "label" field in test datasets
                    label = line_json['label']
                    if label == 'NOT ENOUGH INFO':
                        evidences = [[]]
                    else:
                        #     continue
                        evidences = []
                        for annotator in line_json['evidence']:
                            ev = []
                            for evidence in annotator:
                                if self.granularity == 'sentence':
                                    ev.append('{}_{}'.format(evidence[2], evidence[3]))
                                    # ev.append(evidence[2])
                                else:  # args.granularity == 'paragraph'
                                    ev.append(evidence[2])
                            evidences.append(ev)
                else: # Test mode, no labels
                    label = None
                    evidences = None
                yield (qid, query, label, evidences)


    def read_corpus(self, input_path):
        files = os.listdir(input_path)
        for file in files:
            with open(os.path.join(input_path, file), 'r', encoding='utf-8') as f:
                for line in f:
                    line_json = json.loads(line.strip())
                    if self.granularity == 'sentence':
                        docs = []
                        docs_titles = []
                        for li in line_json['lines'].split('\n'):
                            if li == '':  # don't split by tabs if "lines" is empty
                                continue
                            else:
                                if li.split('\t')[0].isnumeric():
                                    docs.append(li.split('\t')[1])
                                    docs_titles.append(line_json['id'] + '_' + li.split('\t')[0])#line_json['id'] + '_' + str(count))
                            # count+=
                    elif self.granularity == 'pipeline':
                        docs = [line_json['lines']]
                        docs_titles = [line_json['id']]
                    else:
                        docs = [line_json['text']]
                        docs_titles = [line_json['id']]
                        docs_length = len(docs[0].split())
                        if docs_length > 1500: # these are mostly verbose lists and no "real" Wiki introduction sections, following: https://github.com/dominiksinsaarland/document-level-FEVER
                            # print("Too long, skipping...")
                            continue

                    for i,doc in enumerate(docs):
                        yield {'id': docs_titles[i], 'contents': doc}


    def read_predictions_as_dicts(self, input_path, granularity, max_evidence = 100):
        elements = []
        curr_id_elements = []
        curr_query = -1
        with open(input_path, 'r', encoding='utf-8') as f_in:
            for i, line in enumerate(f_in):
                content = line.strip().split('\t')
                if granularity == "passage":
                    query_id, _, doc_id, rank, score, retriever = content
                    curr_dict = {"qid": query_id, "doc": doc_id, "rank": rank, "score": float(score), "retriever": retriever}
                elif granularity == "sentence":
                    query_id, _, sent_id, sentence, rank, score, retriever = line.strip().split('\t')
                    curr_dict = {"qid": query_id, "sent_id": sent_id, "sentence": sentence, "rank": rank, "score": float(score), "retriever": retriever}

                if query_id != curr_query:
                    if i > 0:
                        elements.append(copy.copy(curr_id_elements))
                    curr_query = query_id
                    curr_id_elements.clear()
                if int(rank) <= max_evidence:
                    curr_id_elements.append(copy.copy(curr_dict))
            elements.append(curr_id_elements) # Last sample
        return elements

    def read_sufficiency_proofs(self, input_path, max_proofs=20):
        elements = []
        curr_id_elements = []
        curr_query = -1
        with open(input_path, 'r', encoding='utf-8') as f_in:
            for i, line in enumerate(f_in):
                if len(line.strip().split('\t')) != 4:
                    continue
                query_id, rank, score, proof = line.strip().split('\t')
                curr_dict = {"qid": query_id, "rank":rank, "score": float(score), "proof": proof}
                if query_id != curr_query:
                    if i > 0:
                        elements.append(copy.copy(curr_id_elements))
                    curr_query = query_id
                    curr_id_elements.clear()
                if int(rank) <= max_proofs:
                    curr_id_elements.append(copy.copy(curr_dict))
            
            elements.append(curr_id_elements)
        return elements

    def read_predictions(self, input_path, max_evidence = 100): #Read Anserini predictions
        curr_query = -1
        predicted_docs = []
        ranks = []
        scores = []
        retrievers = []
        with open(input_path, 'r', encoding='utf-8') as f_in:
            for i, line in enumerate(f_in):
                if self.granularity in ["passage", "paragraph"]:
                    if len(line.strip().split('\t')) == 5:
                        query_id, _, doc_id, rank, score = line.strip().split('\t')
                        retriever = "not set"
                    else:
                        query_id, _, doc_id, rank, score, retriever = line.strip().split('\t')
                elif self.granularity == "sentence":
                    query_id, _, doc_id, _, rank, score, retriever = line.strip().split('\t')
                    
                query_id = int(query_id)

                if query_id != curr_query:
                    if i > 0:
                        yield copy.deepcopy((curr_query, predicted_docs, ranks, scores, retrievers))
                    curr_query = query_id
                    predicted_docs.clear()
                    ranks.clear()
                    scores.clear()
                    retrievers.clear()

                if int(rank) <= max_evidence:
                    # predicted_docs.append(unicodedata.normalize('NFD', doc_id))
                    predicted_docs.append(doc_id)
                    ranks.append(rank)
                    scores.append(score)
                    retrievers.append(retriever)

            yield copy.deepcopy((query_id, predicted_docs, ranks, scores, retrievers))


    def extract_sentence(self, sent_id):
        doc_element = self.searcher.doc(sent_id)
        if doc_element:
            element = json.loads(doc_element.raw())
            return self.process_title(" ".join(str(element["id"]).split("_")[:-1])) + " ; " + self.process_sentence(str(element["contents"]))
        else:
            return ""

    def extract_sentences_from_document(self, doc_title):
        sentences = []
        corpus = []
        corpus_ids = []
        doc_element = self.searcher.doc(doc_title)
        if doc_element == None:
            return {"corpus": corpus, "corpus_ids": corpus_ids, "sentences": sentences}
        doc_content = json.loads(doc_element.raw())['contents']
        for j, li in enumerate(doc_content.split('\n')):
            if li.strip().strip("\t") == '':
                return {"corpus": corpus, "corpus_ids": corpus_ids, "sentences": sentences}
            if li.split('\t')[0].isnumeric():
                li_text = li.split('\t')[1]
                if li_text.strip().strip("\t") == '':
                    return {"corpus": corpus, "corpus_ids": corpus_ids, "sentences": sentences}
            corpus.append(self.process_title(doc_title) + ' . ' + self.process_sentence(li_text))
            sentences.append(li_text)
            corpus_ids.append(doc_title + '_' + li.split('\t')[0])
    
        return {"corpus": corpus, "corpus_ids": corpus_ids, "sentences": sentences}
