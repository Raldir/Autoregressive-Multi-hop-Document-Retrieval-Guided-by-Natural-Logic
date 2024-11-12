from typing import Iterator
from rank_bm25 import BM25Okapi
import numpy
import json
import unicodedata
import re


class RetrievalDataset(object):
    def read_annotations(self, input_path) -> Iterator[list]:
        """Returns list of all ids in that element"""
        pass

    def read_corpus(self, input_path) ->Iterator[list]:
        """Return the specific id of that element"""
        pass

    def read_predictions(self, input_path) ->Iterator[list]:
        """Return the specific id of that element"""
        pass

    def retrieval_pipeline(self, prediction_file, document_hits, evidence_hits) -> Iterator[list]:
        """ FILL """
        pass

    def process_claim(self, claim):
        return claim

    def process_sentence(self, sentence):
        return sentence

    def process_sentence_reverse(self, sentence):
        return sentence

    def process_title(self, title):
        return title

    def process_title_reverse(self, title):
        return title

    def get_sentence_content_from_id(self, sentence, sentence_searcher):
         doc_element = sentence_searcher.doc(sentence) #sentence index
         if doc_element == None:
             return ""
         doc_content = json.loads(doc_element.raw())['contents']
         return doc_content


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

    def read_proofver_proof(self, input_file):
        natural_operations_sequences = {}
        proofs = {}
        with open(input_file, 'r', encoding = 'utf-8') as f:
            lines = f.readlines()
            for line in lines:
                content = line.split('\t')
                if content[0] in natural_operations_sequences:
                    continue
                proof = content[1].strip()
                natural_operations_sequences[int(content[0])] = proof
        for id, sequence in natural_operations_sequences.items():
            matches = re.findall(r"\{ (.+?) \} \[ (.+?) \] (.+?)", sequence)
            proofs[id] = []
            for match in matches:
                assert len(match) == 3, f"Does not match, got {match}"
                assert match[2] in ['<', '>', '!', '=', '|', '#']
                proofs[id].append(match)
        return proofs


    def run_bm25(self, corpus, query):
        # tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(corpus, k1=0.9, b=0.4)
        sentence_scores = bm25.get_scores(query)
        # print(sentence_scores)
        index = numpy.argsort(sentence_scores).tolist()
        index.reverse()
        # print(index)
        return index,sentence_scores


    def get_documents_from_anserini_predictions(self, prediction_file, num_docs = 5, normalize = True):
        documents = {}
        with open(prediction_file, 'r', encoding = 'utf-8') as in_file:
            lines = in_file.readlines()
            docs = []
            current_id = 1

            for i, line in enumerate(lines):
                line = line.strip()
                if int(line.split(' ')[0]) != current_id:#(i % document_hits == 0) and (i != 0):
                    #documents.append(docs)
                    documents[current_id] = docs[:num_docs] #consider only ten documents for sentence retrieval
                    docs = []
                    doc = ' '.join(line.strip().split(' ')[2:-3])
                    if normalize:
                        docs.append(unicodedata.normalize('NFC', doc))
                    else:
                        docs.append(doc)
                    current_id=int(line.split(' ')[0])
                else:
                    doc = ' '.join(line.strip().split(' ')[2:-3])
                    if normalize:
                        docs.append(unicodedata.normalize('NFC',(doc)))
                    else:
                        docs.append(doc)
                    current_id = int(line.split(' ')[0])
            if docs != []:
                documents[current_id] = docs

        for key, value in documents.items():
            documents[key] = list(dict.fromkeys(value))
            
        return documents


    def get_documents_from_anserini_predictions_with_scores(self, prediction_file, num_docs = 5, normalize = True):
        documents = {}
        with open(prediction_file, 'r', encoding = 'utf-8') as in_file:
            lines = in_file.readlines()
            docs = []
            current_id = 1

            for i, line in enumerate(lines):
                line = line.strip()
                if int(line.split(' ')[0]) != current_id:#(i % document_hits == 0) and (i != 0):
                    #documents.append(docs)
                    documents[current_id] = docs[:num_docs] #consider only ten documents for sentence retrieval
                    docs = []
                    doc = ' '.join(line.strip().split(' ')[2:-3])
                    if doc in set([x[0] for x in docs]):
                        continue
                    score = float(line.strip().split(' ')[-2])
                    if normalize:
                        docs.append((unicodedata.normalize('NFC', doc), score))
                    else:
                        docs.append((doc, score))
                    current_id=int(line.split(' ')[0])
                else:
                    doc = ' '.join(line.strip().split(' ')[2:-3])
                    if doc in set([x[0] for x in docs]):
                        continue
                    score = float(line.strip().split(' ')[-2])
                    if normalize:
                        docs.append((unicodedata.normalize('NFC',doc), score))
                    else:
                        docs.append((doc, score))
                    current_id = int(line.split(' ')[0])
            if docs != []:
                documents[current_id] = docs[:num_docs]

        return documents

    def get_sentence_predictions(self, prediction_file, num_docs = 5):
        documents = {}
        with open(prediction_file, 'r') as in_file:
            lines = in_file.readlines()
            docs = []
            current_id = 1

            for i, line in enumerate(lines):
                line = line.strip()
                if int(line.split(' ')[0]) != current_id:#(i % document_hits == 0) and (i != 0):
                    #documents.append(docs)
                    documents[current_id] = list(dict.fromkeys(docs[:num_docs])) #consider only ten documents for sentence retrieval
                    docs = []
                    doc = ' '.join(line.strip().split(' ')[2:-3])
                    docs.append(unicodedata.normalize('NFC',doc))
                    current_id=int(line.split(' ')[0])
                else:
                    doc = ' '.join(line.strip().split(' ')[2:-3])
                    docs.append(unicodedata.normalize('NFC',doc))
                    current_id = int(line.split(' ')[0])
            if docs != []:
                documents[current_id] = list(dict.fromkeys(docs))

        return documents

    def get_sentence_predictions_with_scores(self, prediction_file, num_docs = 5):
        documents = {}
        with open(prediction_file, 'r') as in_file:
            lines = in_file.readlines()
            docs = []
            current_id = 1

            for i, line in enumerate(lines):
                line = line.strip()
                if int(line.split(' ')[0]) != current_id:#(i % document_hits == 0) and (i != 0):
                    #documents.append(docs)
                    documents[current_id] = docs[:num_docs] #consider only ten documents for sentence retrieval
                    docs = []
                    doc = ' '.join(line.strip().split(' ')[2:-3])
                    if doc in [x[0] for x in docs]:
                        continue
                    score = float(line.strip().split(' ')[-2])
                    docs.append((unicodedata.normalize('NFC',doc), score))
                    current_id=int(line.split(' ')[0])
                else:
                    doc = ' '.join(line.strip().split(' ')[2:-3])
                    if doc in [x[0] for x in docs]:
                        continue
                    score = float(line.strip().split(' ')[-2])
                    docs.append((unicodedata.normalize('NFC',doc), score))
                    current_id = int(line.split(' ')[0])
            if docs != []:
                documents[current_id] = docs[:num_docs]

        return documents


    def read_generated_queries(self, input_file):
        queries = {}
        with open(input_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                content = line.split('\t')
                id = int(content[0].strip())
                if id in queries:
                    continue
                query = content[1].split(']')
                if len(query) > 1 and len(query[1].strip()) > 1:
                    # queries[id] = self.process_sentence(query[1].strip())
                    # queries[id] = query[1].strip()
                    title = query[0].split('[')
                    if len(title) > 1:
                         title = title[1]
                    else:
                        title = ""
                    if '[' in query[1]:
                        queries[id] = (title, query[1].split('[')[0].strip())
                    else:
                        queries[id] = (title, query[1].strip())

                    # print(id, queries[id])
                    # break # process more than just the first in the future
        return queries
