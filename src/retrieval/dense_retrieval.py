import pickle
import numpy
import json
import logging

# from sentence_transformers import CrossEncoder
from tqdm import tqdm

from transformers import T5ForConditionalGeneration

from pyserini.search import LuceneSearcher
from pygaggle.rerank.transformer import MonoT5
from pygaggle.rerank.base import Query, Text

import dataset_readers

logger = logging.getLogger(__name__)


class DenseRetrieval():

    def __init__(self, config, current_iteration, granularity):
        self.reader = dataset_readers.get_class(config.dataset)(**{'granularity': granularity, "config": config})
        self.config = config
        self.current_iteration = current_iteration
        if current_iteration == 0:
            model = T5ForConditionalGeneration.from_pretrained(config.reranker_model_init, max_length=config.reranker_max_input_length).to(config.device)
        else:
            model = T5ForConditionalGeneration.from_pretrained(config.reranker_model.replace("{ITER}", str(current_iteration)),  max_length=config.reranker_max_input_length).to(config.device)
        self.model = MonoT5(model=model)
        self.max_docs_to_consider = config.reranker_num_docs_to_consider

    def save_documents_to_file(self, retrieved_documents):
        with open(self.config.reranked_dev_docs.replace("{ITER}", str(self.current_iteration)), "w", encoding="utf-8") as f_out:
            for sample in retrieved_documents:
                for i,hit in enumerate(sample):
                    curr_str = "{}\t{}\t{}\t{}\t{:.5f}\t{}".format(hit["qid"], "XX", hit["doc"],  hit["rank"], hit["score"], hit["retriever"])
                    f_out.write("{}\n".format(curr_str))

    def save_sentences_to_file(self, retrieved_sentences):
        with open(self.config.reranked_dev_sentences.replace("{ITER}", str(self.current_iteration)), "w", encoding="utf-8") as f_out:
            for sample in retrieved_sentences:
                for i,hit in enumerate(sample):
                    curr_str = "{}\t{}\t{}\t{}\t{}\t{:.5f}\t{}".format(hit["qid"], "XX", hit["sent_id"], hit["sentence"], hit["rank"], hit["score"], hit["retriever"])
                    f_out.write("{}\n".format(curr_str))

    def save_tables_to_file(self, retrieved_tables):
        with open(self.config.reranked_dev_tables.replace("{ITER}", str(self.current_iteration)), "w", encoding="utf-8") as f_out:
            for sample in retrieved_tables:
                for i,hit in enumerate(sample):
                    curr_str = "{}\t{}\t{}\t{}\t{}\t{:.5f}\t{}".format(hit["qid"], "XX", hit["table_id"], hit["table"], hit["rank"], hit["score"], hit["retriever"])
                    f_out.write("{}\n".format(curr_str))


    def retrieve_sentences(self, qid, claim, retrieved_docs, sentences_previous_iteration=None):
        if self.current_iteration == 0:
            predicted_sentences, reranked_documents = self.retrieve_sentences_init(qid, claim, retrieved_docs)
        else:
            predicted_sentences, reranked_documents = self.retrieve_sentences_hop(qid, claim, retrieved_docs, sentences_previous_iteration)
        
        return [predicted_sentences, reranked_documents]

    
    def retrieve_sentences_init(self, qid, claim, retrieved_docs):
        corpus = []
        sentences = []
        corpus_ids = []

        for k,doc in enumerate(retrieved_docs[:self.max_docs_to_consider]):
            doc_title = doc["doc"]
            content = self.reader.extract_sentences_from_document(doc_title)
            corpus += content["corpus"]
            corpus_ids += content["corpus_ids"]
            sentences += content["sentences"]

        prediction_pairs = [[corpus_ids[o], ele] for o,ele in enumerate(corpus)]

        if len(prediction_pairs) == 0:
            print(claim, retrieved_docs)
            return [[], []]

        texts = [ Text(p[1], {'docid': p[0]}, 0) for p in prediction_pairs]
        reranked = self.model.rerank(Query(claim.lower()), texts)
        reranked_dict = {}
        for k in range(len(reranked)):
            reranked_dict[reranked[k].metadata["docid"]] = reranked[k].score

        reranked_sorted = sorted(reranked_dict.items(), key=lambda x: x[1], reverse=True)
        predicted_evidence = [{"qid": qid, "sent_id": ele[0], "sentence": sentences[corpus_ids.index(ele[0])], "score": ele[1], "rank": i + 1, "retriever": "reranker"} for i, ele in enumerate(reranked_sorted)]

        reranked_documents = []
        for sent in predicted_evidence:
            doc = "_".join(sent["sent_id"].split("_")[:-1]) if self.config.dataset == "fever" else sent["sent_id"].split("_")[0]
            if doc not in [x["doc"] for x in reranked_documents]:
                new_dict = {"qid": qid, "doc": doc, "score": sent["score"], "rank": len(reranked_documents) + 1, "retriever": "reranker"}
                reranked_documents.append(new_dict)
            
        print(claim)
        print(reranked_documents)


        return [predicted_evidence, reranked_documents]

    
    """
    Always keep the top k documents as a k-hop situation. 
    """
    def format_query(self, claim, sentences_previous_iteration):
        if self.config.reranker_prompt_format == "concat":
            sentences_docs = [self.reader.process_title("_".join(x["sent_id"].split("_")[:-1])) if self.config.dataset == "fever" else self.reader.process_title(x["sent_id"].split("_")[0]) for x in sentences_previous_iteration]
            context = ["{} . {}".format(sentences_docs[i], self.reader.process_sentence(sentences_previous_iteration[i]["sentence"])) if len(sentences_previous_iteration) > 0 else '' for i in range(len(sentences_previous_iteration[:self.current_iteration]))]
            query = '{} </s> {}'.format(claim, " <s> ".join(context))
        return query


    """
    Currently always keeps the top k sentences/documents for hop k and adds new ones on top of them.
    """
    def retrieve_sentences_hop(self, qid, claim, retrieved_docs, sentences_previous_iteration):
        # NOTE: Sentences from documents that are not part of the retrieved docs from GENRE for this current iteration are not considered for reranking and are subsequently dropped from the reranked sentences. This is a TODO to fix potentially.
        query = self.format_query(claim, sentences_previous_iteration)

        corpus = []
        sentences = []
        corpus_ids = []

        for k,doc in enumerate(retrieved_docs[:self.max_docs_to_consider]):
            doc_title = doc["doc"]
            content = self.reader.extract_sentences_from_document(doc_title)
            corpus += content["corpus"]
            corpus_ids += content["corpus_ids"]
            sentences += content["sentences"]

        prediction_pairs = [[corpus_ids[o], ele] for o,ele in enumerate(corpus)]
        # print(prediction_pairs)

        if len(prediction_pairs) == 0:
            print(qid, query, retrieved_docs)
            return [[], []]

        texts = [ Text(p[1], {'docid': p[0]}, 0) for p in prediction_pairs]
        reranked = self.model.rerank(Query(query.lower()), texts)
        reranked_dict = {}
        for k in range(len(reranked)):
            reranked_dict[reranked[k].metadata["docid"]] = reranked[k].score

        reranked_sorted = sorted(reranked_dict.items(), key=lambda x: x[1], reverse=True)

        predicted_evidence = []
        for i, ele in enumerate(reranked_sorted):
            if ele[0] in [x["sent_id"] for x in sentences_previous_iteration[:self.current_iteration]]:
                continue
            else:
                curr_dict = {"qid": qid, "sent_id": ele[0], "sentence": sentences[corpus_ids.index(ele[0])], "score": ele[1] * self.config.multi_hop_weighting, "rank": len(predicted_evidence) + 1, "retriever": "reranker-hop-{}".format(self.current_iteration)}
                predicted_evidence.append(curr_dict)

        
        predicted_evidence = sentences_previous_iteration[:self.current_iteration] + predicted_evidence
        if self.config.sort_after_reranking:
            predicted_evidence.sort(key = lambda x : x["score"], reverse=True)

        for i, element in enumerate(predicted_evidence): # update ranks
            element["rank"] = i + 1


        reranked_documents = []
        for sent in predicted_evidence:
            doc = "_".join(sent["sent_id"].split("_")[:-1]) if self.config.dataset == "fever" else sent["sent_id"].split("_")[0]
            if doc not in [x["doc"] for x in reranked_documents]:
                new_dict = {"qid": qid, "doc": doc, "score": sent["score"], "rank": len(reranked_documents) + 1, "retriever": sent["retriever"]}
                reranked_documents.append(new_dict)
            

        return [predicted_evidence, reranked_documents]