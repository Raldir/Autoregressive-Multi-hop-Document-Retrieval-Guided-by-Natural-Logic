import pickle
import re
import dataset_readers

from src.retrieval.genre.hf_model import GENRE
from src.retrieval.genre.trie import Trie
from src.retrieval.genre.entity_linking_hop import get_end_to_end_prefix_allowed_tokens_fn_hf as get_prefix_allowed_tokens_fn

class GenreRetrieval():

    def __init__(self, config, current_iteration, db_titles, granularity): # mode between init and completion
        self.config = config
        self.current_iteration = current_iteration
        self.reader = dataset_readers.get_class(config.dataset)(**{'granularity': granularity, "config": config})
        if current_iteration == 0:
            self.model = GENRE.from_pretrained(config.genre_init_model_path).to(config.device).eval()
        else:
            self.model = GENRE.from_pretrained(config.genre_model_path.replace("{ITER}", str(self.current_iteration))).to(config.device).eval()
        
        trie_path = config.genre_trie_path if current_iteration==0 else config.genre_trie_completion_path
        with open(trie_path, "rb") as f:
            self.trie = Trie.load_from_dict(pickle.load(f))
        
        self.db_titles = set(db_titles)

    def save_documents_to_file(self, retrieved_documents):
        with open(self.config.genre_dev_docs.replace("{ITER}", str(self.current_iteration)), "w", encoding="utf-8") as f_out:
            for sample in retrieved_documents:
                for i,hit in enumerate(sample):
                    curr_str = "{}\t{}\t{}\t{}\t{:.5f}\t{}".format(hit["qid"], "XX", hit["doc"],  hit["rank"], hit["score"], hit["retriever"])
                    f_out.write("{}\n".format(curr_str))

    def retrieve_documents(self, qid, claim, sentences_previous_iteration=None):
        if self.current_iteration == 0:
            documents = self.retrieve_documents_init(qid, claim)
        else:
            documents = self.retrieve_documents_completion(qid, claim, sentences_previous_iteration)
        return documents

    def retrieve_documents_init(self, qid, claim):
        claim = [claim]
        print(claim)
        output = self.model.sample(
            claim,
            prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent.tolist()),
            num_return_sequences=self.config.genre_init_num_beams,
            num_beams=self.config.genre_init_num_beams,
        )
        # breakpoint()
        # ADDED the process title reverse. Make sure it does not affect performance negatively.
        documents = [[(self.reader.process_title_reverse(e['text']), e["score"].item()) for e in el] for el in output]
        print("GENRE DOCS", documents)
        documents = [item for sublist in output for item in documents][0]

        documents_formatted = []
        for (doc, score) in documents:
            doc_formatted = self.reader.process_title_reverse_genre(doc) 
            # print(doc, doc_formatted)
            if doc_formatted not in self.db_titles or doc_formatted in [x["doc"] for x in documents_formatted]:
                documents.remove((doc, score))
            else:
                curr_doc = {"qid": qid, "doc": doc_formatted, "score": score, "rank":len(documents_formatted) + 1, "retriever": "genre"}
                documents_formatted.append(curr_doc)
    
        return documents_formatted[:self.config.genre_init_num_retrieved_docs]

    def format_query_completion(self, claim, relevant_sentences, sentences_documents, format_mode):
        if format_mode == 'papelo':
            claim_formatted = '{} </s> {}'.format(claim, ' '.join(['{} [ {} ] {}'.format("< E" + str(i) + ' >', self.reader.process_title(sentences_documents[i]["doc"]), self.reader.process_sentence(x)) for i, x in enumerate(relevant_sentences)]))
        return claim_formatted

    def retrieve_documents_completion(self, qid, claim, sentences_previous_iteration):
        # TODO: Check wether the already used sentence needs to be dealt with in special way (i.e. starting hop 2)
        sentences_previous_iteration = sentences_previous_iteration[:self.config.genre_hop_num_sentences_context]
        relevant_sentences = [x["sentence"] for x in sentences_previous_iteration]
        sentences_documents = []
        for i, element in enumerate(sentences_previous_iteration):
            doc = "_".join(element["sent_id"].split("_")[:-1]) if self.config.dataset == "fever" else element["sent_id"].split("_")[0]
            curr_dict = {"qid": qid, "doc": doc, "score": sentences_previous_iteration[i]["score"], "rank":len(sentences_documents) + 1, "retriever": "reranker"}
            sentences_documents.append(curr_dict)

        prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
        self.model,
        [claim],
        mention_trie=self.trie
        )

        claim = self.format_query_completion(claim, relevant_sentences, sentences_documents, format_mode="papelo")
        claim = self.reader.process_sentence(claim)
        claim = [claim]

        output = self.model.sample(
                claim,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_return_sequences=self.config.genre_hop_num_beams,
                num_beams=self.config.genre_hop_num_beams,
        )
        print(output)

        # THis was process_sentence_reverse before. Check if it causes difference in performance.
        output = [[(self.reader.process_title_reverse(e['text']).strip(), e["score"].item()) for e in el] for el in output]
        output = [item for sublist in output for item in sublist]

        documents_formatted = []
        linked_sentences_docs = []
        for doc_generated in output:
            doc = doc_generated[0]
            matches = re.findall(r"\[ (.+?) \]", doc)
            matches_links = re.findall(r"E(\d+)", doc)
            for match in matches:
                doc = self.reader.process_title_reverse_genre(match).strip()
                if doc not in self.db_titles or doc in [x["doc"] for x in documents_formatted]:
                    continue
                else:
                    curr_doc = {"qid": qid, "doc": doc, "score":doc_generated[1], "rank":len(documents_formatted) + 1, "retriever": "genre-hop-{}".format(self.current_iteration)}
                    documents_formatted.append(curr_doc)
                    linked_sentences_docs.append([int(x) for x in matches_links])
        
        documents_formatted = documents_formatted[:self.config.genre_hop_num_retrieved_docs]
        linked_sentences_docs = linked_sentences_docs[:self.config.genre_hop_num_retrieved_docs]

        print(documents_formatted)
        print("-----")
                        
        # TODO Check whether always first having the initial documents (i.e. sentneces_documents) and then the new ones work better.
        merged_documents = []
        for i, linked_sentences_ids in enumerate(linked_sentences_docs):
            if self.config.genre_documents_merge_mode == "linked":
                mapped_documents = [sentences_documents[linked_sentences_id] for linked_sentences_id in linked_sentences_ids if linked_sentences_id < len(sentences_documents)]
                mapped_documents.sort(key = lambda x : x["score"], reverse=True) # Sort generated ids by probability
                merged_documents += mapped_documents
            merged_documents.append(documents_formatted[i])
        
        if self.config.genre_documents_merge_mode == "default":
            merged_documents = sentences_documents + merged_documents
        
        merged_documents_no_duplicates = []
        for entry in merged_documents:
            if entry["doc"] not in [x["doc"] for x in merged_documents_no_duplicates]:
                merged_documents_no_duplicates.append(entry)
                
        return merged_documents_no_duplicates, documents_formatted




