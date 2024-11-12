
import os
import argparse
import json
import re
from tqdm import tqdm
import copy 
import unicodedata

from .dataset_reader import RetrievalDataset
from pyserini.search import SimpleSearcher
from utils import drqa_tokenizers

from feverous.utils.annotation_processor import AnnotationProcessor
from feverous.database.feverous_db import FeverousDB
from feverous.utils.wiki_page import WikiPage, WikiTable

class FeverousDatasetReader(RetrievalDataset):

    def __init__(self, **kwargs):
        self.granularity = kwargs.get('granularity')
        self.config = kwargs.get('config')
        if self.config:
            self.searcher = FeverousDB(self.config.pyserini_index_path_pipeline)

    def process_claim(self, claim):
        return unicodedata.normalize("NFD", claim).strip()

    def process_title(self, title):
        return unicodedata.normalize("NFD", title).strip()

    def process_sentence(self, sentence):
        # Regular expression to match hyperlinks in the format [[link|text]] or [[link]]
        sentence = unicodedata.normalize("NFD", sentence).strip()
        pattern = r'\[\[([^|]+?)(?:\|([^]]+))?\]\]'
        
        def replace_match(match):
            # If there's a pipe (|), use the text after it; otherwise, use the whole link
            return match.group(2) if match.group(2) else match.group(1)
        
        # Replace all matches in the text
        sentence = re.sub(pattern, replace_match, sentence)
        sentence = sentence.replace("\n", " ")

        return sentence

    def process_table(self, table):
        # Regular expression to match hyperlinks in the format [[link|text]] or [[link]]
        table = unicodedata.normalize("NFD", table).strip()
        pattern = r'\[\[([^|]+?)(?:\|([^]]+))?\]\]'
        
        def replace_match(match):
            # If there's a pipe (|), use the text after it; otherwise, use the whole link
            return match.group(2) if match.group(2) else match.group(1)
        
        # Replace all matches in the text
        table = re.sub(pattern, replace_match, table)
        return table


    def process_title_reverse_genre(self, title):
        return unicodedata.normalize("NFD", title).strip()

    def read_annotations(self, input_path):
        annotations = AnnotationProcessor(input_path)
        for i, annotation in enumerate(annotations):
            qid = annotation.get_id()
            query = annotation.get_claim()
            label = annotation.get_verdict()
            evidence_type = annotation.get_evidence_type()
            if self.granularity == 'paragraph':
                evidences = [list(set(title_set)) for title_set in annotation.get_titles()]
            elif self.granularity == "all_sentence":
                evidences = annotation.get_evidence()
                new_evidences = []
                for evidence in evidences:
                    if all(["_sentence_" in x for x in evidence]):
                        new_ev = evidence
                    else:
                        new_ev = []
                    if new_ev:
                        new_evidences.append(copy.copy(new_ev))
                evidences = new_evidences
            elif self.granularity == "sentence":
                evidences = annotation.get_evidence()
                new_evidences = []
                for evidence in evidences:
                    new_ev = [x for x in evidence if "_sentence_" in x]
                    if new_ev:
                        # for x in new_ev:
                        #     new_evidences.append([x])
                        new_evidences.append(copy.copy(new_ev))
                evidences = new_evidences
            elif self.granularity == 'table':
                evidences = annotation.get_evidence()
                evidences_new = []
                for evidence in evidences:
                    evidence_new = []
                    for ev in evidence:
                        if "_cell_" not in ev:
                            continue
                        page = ev.split("_")[0]
                        table_pos = ev.split("_")[-3] if "_cell_" in ev else ev_split("_")[-1] # if _table_caption_0 then it is the last index
                        table_id = page + "_table_" + table_pos
                        if table_id not in evidence_new:
                            evidence_new.append(table_id)
                    if evidence_new:
                        evidences_new.append(copy.copy(evidence_new))
                evidences = evidences_new
            evidences = [[unicodedata.normalize('NFD', x) for x in ev] for ev in evidences]
            yield (qid, query, label, evidences)


    def read_corpus(self, input_path):
        db =  FeverousDB(input_path)
        doc_ids = db.get_doc_ids()
        for doc_id in tqdm(doc_ids):
            page_json = db.get_doc_json(doc_id)
            wiki_page = WikiPage(doc_id, page_json)
            if self.granularity == 'paragraph':  # args.granularity == 'paragraph'
                all_sentences = wiki_page.get_sentences()
                intro_sents_index = wiki_page.page_order.index('section_0') if 'section_0' in wiki_page.page_order else len(wiki_page.page_order) -1
                sentences_in_intro = [sent for sent in all_sentences if wiki_page.page_order.index(sent.get_id()) < intro_sents_index]
                # sentences_in_intro = [sent for sent in all_sentences]
                docs = [' '.join([self.process_sentence(str(s)) for s in sentences_in_intro])]
                # print(docs)
                # docs = [' '.join([str(s) for s in wiki_page.get_sentences()])]
                docs_titles = [doc_id]
            for i,doc in enumerate(docs):
                yield {'id': docs_titles[i], 'contents': doc}


    def read_predictions_as_dicts(self, input_path, granularity, max_evidence = 100):
        elements = []
        curr_id_elements = []
        curr_query = -1
        with open(input_path, 'r', encoding='utf-8') as f_in:
            for i, line in enumerate(f_in):
                content = line.strip().split('\t')
                if granularity == "paragraph":
                    query_id, _, doc_id, rank, score, retriever = content
                    curr_dict = {"qid": query_id, "doc": doc_id, "rank": rank, "score": float(score), "retriever": retriever}
                elif granularity == "sentence":
                    try:
                        query_id, _, sent_id, sentence, rank, score, retriever = line.strip().split('\t')
                    except:
                        print("skipping loading in line {}".format(line))
                        continue
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

    def read_predictions(self, input_path, max_evidence = 100): #Read Anserini predictions
        curr_query = -1
        predicted_items = []
        ranks = []
        scores = []
        retrievers = []
        with open(input_path, 'r', encoding='utf-8') as f_in:
            for i, line in enumerate(f_in):
                if self.granularity == "paragraph":
                    if len(line.strip().split('\t')) == 5:
                        query_id, _, item_id, rank, score = line.strip().split('\t')
                        retriever = "not set"
                    else:
                        query_id, _, item_id, rank, score, retriever = line.strip().split('\t')
                elif self.granularity in ["sentence", "table"]:
                    try:
                        query_id, _, item_id, _, rank, score, retriever = line.strip().split('\t')
                    except:
                        print("skipping loading in line {}".format(line))
                        continue

                query_id = int(query_id)

                if query_id != curr_query:
                    if i > 0:
                        yield copy.copy((curr_query, predicted_items, ranks, scores, retrievers))
                    curr_query = query_id
                    predicted_items = []
                    ranks = []
                    scores = []
                    retrievers = []

                if int(rank) <= max_evidence:
                    predicted_items.append(item_id)
                    ranks.append(rank)
                    scores.append(score)
                    retrievers.append(retriever)
            yield copy.copy((query_id, predicted_items, ranks, scores, retrievers))

    def extract_sentences_from_document(self, doc_title):
        sentences = []
        corpus = []
        corpus_ids = []
        page_json = self.searcher.get_doc_json(doc_title)
        wiki_page = WikiPage(doc_title, page_json)
        all_sentences = wiki_page.get_sentences()
        for sentence in all_sentences:
            sentence_id = sentence.get_id()
            context = ' '.join([str(x) for x in wiki_page.get_context(sentence_id)])
            sentence_text = self.process_sentence(str(sentence))
            corpus.append(self.process_title(context) + " . " + sentence_text)
            corpus_ids.append(self.process_title(doc_title) + "_" + sentence_id)
            sentences.append(sentence_text)
        return {"corpus": corpus, "corpus_ids": corpus_ids, "sentences": sentences}     

    def extract_hyperlinks_from_sentence(self, sentence_id):
        def hyperlink_pattern(sentence):
            pattern = r'\[\[(.*?)\|' # Technically no pipe links also exist but ignore for now

            hyperlinks = re.findall(pattern, sentence)
            
            return hyperlinks

        doc_title = sentence_id.split("_")[0]
        sentence_id = "_".join(sentence_id.split("_")[1:])
        tables = []
        corpus = []
        corpus_ids = []
        page_json = self.searcher.get_doc_json(doc_title)
        wiki_page = WikiPage(doc_title, page_json)
        sentence = page_json[sentence_id]
        hyperlinks = hyperlink_pattern(sentence)
        hyperlinks = [x.replace("_", " ") for x in hyperlinks]
        return hyperlinks

