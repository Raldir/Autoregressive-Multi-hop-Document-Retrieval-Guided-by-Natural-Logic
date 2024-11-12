import argparse
import os
import json
from datetime import datetime
import logging
import subprocess
import torch
import gc
import copy

from tqdm import tqdm

import dataset_readers

from src.utils.convert_dataset_to_pyserini import convert_collection
from src.utils.Config import Config
from src.utils.util import ParseKwargs, set_seeds, init_logging, setup_spacy

from src.retrieval import BM25, GenreRetrieval, DenseRetrieval
from src.utils.evaluate import evaluate_retrieval, evaluate_sufficiency
from src.sufficiency_proof.sufficiency_proofver import SufficiencyProoFVer

logger = logging.getLogger(__name__)

def prepare_index(config):
    # Prepare data in pyserini format
    if not os.path.exists(config.pyserini_document_path):
        convert_collection(dataset=config.dataset, raw_folder=config.raw_document_path, output_folder=config.pyserini_document_path, max_docs_per_file=config.max_docs_per_file, granularity="paragraph")

    # Prepare data for sentence retrieval
    if not os.path.exists(config.pyserini_pipeline_path) and config.dataset != "feverous":
        convert_collection(dataset=config.dataset, raw_folder=config.raw_document_path, output_folder=config.pyserini_pipeline_path, max_docs_per_file=config.max_docs_per_file, granularity="pipeline")

    # Call pyserini to index documents
    if not os.path.exists(config.pyserini_index_path_passage):
        subprocess.run(["python3", "-m", "pyserini.index.lucene", "-collection", "JsonCollection", "-generator", "DefaultLuceneDocumentGenerator", "-threads", "9", "-input", config.pyserini_document_path, "-index", config.pyserini_index_path_passage, "-storePositions", "-storeDocvectors", "-storeRaw"]) 
        # subprocess.run(["python3", "-m", "pyserini.index.lucene", "-collection", "JsonCollection", "-generator", "DefaultLuceneDocumentGenerator", "-threads", "9", "-input", config.pyserini_document_path, "-index", config.pyserini_index_path_passage, "-pretokenized", "-storePositions", "-storeDocvectors", "-storeRaw"]) 

     # Call pyserini to index pipeline
    if not os.path.exists(config.pyserini_index_path_pipeline) and config.dataset != "feverous":
        subprocess.run(["python3", "-m", "pyserini.index.lucene", "-collection", "JsonCollection", "-generator", "DefaultLuceneDocumentGenerator", "-threads", "9", "-input", config.pyserini_pipeline_path, "-index", config.pyserini_index_path_pipeline, "-pretokenized", "-storePositions", "-storeDocvectors", "-storeRaw"])

def evaluate_iteration_initial(config, current_iteration, bm25_retriever, genre_retriever, dense_retriever, all_docs_bm25, all_docs_genre, all_docs_combined, all_docs_reranked, all_sentences_reranked):
    logger.info("Computing BM25 performance.")
    bm25_retriever.save_documents_to_file(all_docs_bm25, current_iteration)
    evaluate_retrieval(dataset=config.dataset, truth_file=config.annotation_path, run_file=config.bm25_pyserini_dev_docs.replace("{ITER}", current_iteration), save_path=config.dev_score_file_bm25.replace("{ITER}", current_iteration), evaluate_type="all", granularity="paragraph")

    logger.info("Computing Genre performance.")
    genre_retriever.save_documents_to_file(all_docs_genre)
    evaluate_retrieval(dataset=config.dataset, truth_file=config.annotation_path, run_file=config.genre_dev_docs.replace("{ITER}", current_iteration), save_path=config.dev_score_file_genre.replace("{ITER}", current_iteration), evaluate_type="all", granularity="paragraph")

    evaluate_iteration(config, current_iteration, dense_retriever, all_docs_combined, all_docs_reranked, all_sentences_reranked)

def evaluate_iteration(config, current_iteration, dense_retriever, all_docs_combined, all_docs_reranked, all_sentences_reranked):
    logger.info("Computing combined performance.")
    # saving combined file
    with open(config.combined_dev_docs.replace("{ITER}", current_iteration), "w", encoding="utf-8") as f_out:
        for sample in all_docs_combined:
            for i,hit in enumerate(sample):
                curr_str = "{}\t{}\t{}\t{}\t{:.5f}\t{}".format(hit["qid"], "XX", hit["doc"],  hit["rank"], hit["score"], hit["retriever"])
                f_out.write("{}\n".format(curr_str))

    evaluate_retrieval(dataset=config.dataset, truth_file=config.annotation_path, run_file=config.combined_dev_docs.replace("{ITER}", current_iteration), save_path=config.dev_score_file_combined.replace("{ITER}", current_iteration), evaluate_type="all", granularity="paragraph")

    # # saving combined file
    if not config.genre_only:
        dense_retriever.save_documents_to_file(all_docs_reranked)
        dense_retriever.save_sentences_to_file(all_sentences_reranked)
        evaluate_retrieval(dataset=config.dataset, truth_file=config.annotation_path, run_file=config.reranked_dev_docs.replace("{ITER}", current_iteration), save_path=config.dev_score_file_reranked.replace("{ITER}", current_iteration), evaluate_type="all", granularity="paragraph")
    
    if config.dataset == "feverous":
        reader = dataset_readers.get_class("feverous")(**{'granularity': "paragraph", 'config':config})
        iter_anno = reader.read_annotations(config.annotation_path)
        annos = [x for x in iter_anno]
        with open(config.combined_dev_docs.replace("{ITER}", current_iteration).replace(".txt", "drqa_format.jsonl"), "w") as f_out:
            for i, entry in enumerate(all_docs_combined):
                qid, query, label, evidences = annos[i]
                docs = [x["doc"] for x in entry]
                docs_scores = [x["score"] for x in entry]
                pred_pages = [[doc, docs_scores[j]] for j, doc in enumerate(docs)]
                new_dict = {"claim": query, "id": qid, "predicted_pages": pred_pages[:10]}
                f_out.write("{}\n".format(json.dumps(new_dict)))


def evaluate_sufficiency_proofs(config, current_iteration, sufficiency_proofver, all_sufficiency_proofs):
    logger.info("Evaluating sufficiency proofs.")
    sufficiency_proofver.save_proofs_to_file(all_sufficiency_proofs, current_iteration)
    evaluate_sufficiency(dataset=config.dataset, truth_file=config.annotation_path, run_file=config.generated_proofs_dev.replace("{ITER}", current_iteration), run_file_evidence = config.reranked_dev_sentences.replace("{ITER}", current_iteration), save_path=config.dev_score_sufficiency.replace("{ITER}", current_iteration), sufficiency_proofver=sufficiency_proofver)


def combine_documents(document_collection1, document_collection2):
    combined_documents = []
    collection1_titles = set([x["doc"] for x in document_collection1])
    complement_collection2 = []
    for element in document_collection2:
        if element["doc"] not in collection1_titles:
            updated_dict = {**element, "rank": len(complement_collection2) + len(document_collection1) +1}
            complement_collection2.append(updated_dict)

    return document_collection1 + complement_collection2

def continue_from_iteration(config, reader, limit):
    if config.continue_from_iteration > 0:
        docs_previous_iteration = reader.read_predictions_as_dicts(config.combined_dev_docs.replace("{ITER}", str(config.continue_from_iteration -1)), granularity="paragraph")
        # docs_previous_iteration = reader.read_predictions_as_dicts(config.reranked_dev_docs.replace("{ITER}", str(config.continue_from_iteration -1)), granularity="passage")
        sentences_previous_iteration = reader.read_predictions_as_dicts(config.reranked_dev_sentences.replace("{ITER}", str(config.continue_from_iteration -1)), granularity="sentence")
    else:
        docs_previous_iteration = []
        sentences_previous_iteration = []
    return docs_previous_iteration, sentences_previous_iteration

def read_document_predictions(config, reader, limit):
    docs_combined = reader.read_predictions_as_dicts(config.combined_dev_docs.replace("{ITER}", str(config.continue_from_iteration)), granularity="paragraph")
    docs_genre = [[x for x in y if "genre-hop" in x["retriever"] or "reranker" == x["retriever"]] for y in docs_combined]
    return docs_combined, docs_genre


def main(config):
    
    logger.info("Loading index...")
    prepare_index(config)

    reader = dataset_readers.get_class(config.dataset)(**{'granularity':"paragraph", "config": config})
    iterator = reader.read_annotations(config.annotation_path)
    bm25_retriever = BM25(config, current_iteration = 0, granularity="paragraph")
    db_titles = bm25_retriever.db_titles

    limit = None if not config.is_debug else 100
    annotations = [anno for anno in iterator][:limit] # Don't stream for now so to measure time

    skip_retrieval = set([])
    if config.enable_sufficiency_check:
        sufficiency_proofver = SufficiencyProoFVer(config)

    docs_previous_iteration, sentences_previous_iteration = continue_from_iteration(config, reader, limit)

    if config.use_precomputed_genre:
        docs_previous_iteration, hop_documents_only_previous_iteration = read_document_predictions(config, reader, limit)

    for hop in range(config.continue_from_iteration, config.max_iterations):
        logger.info("Loading BM25, GENRE, and T5 Reranker...")
        genre_retriever = GenreRetrieval(config, hop, db_titles=db_titles, granularity="paragraph")
        dense_retriever = DenseRetrieval(config, hop, granularity="sentence")

        all_docs_bm25, all_docs_genre, all_docs_combined, all_docs_reranked, all_sentences_reranked, all_sufficiency_proofs = ([] for x in range(6))

        for j, annotation in enumerate(tqdm(annotations)):
            qid, query, label, evidences = annotation
            query = reader.process_claim(query)

            if hop == 0:
                # BM25 initial document retrieval
                retrieved_docs_bm25 = bm25_retriever.retrieve_documents(qid, query)
                all_docs_bm25.append(retrieved_docs_bm25)
                # Genre initial document retrieval
                retrieved_docs_genre = genre_retriever.retrieve_documents(qid, query)
                all_docs_genre.append(retrieved_docs_genre)
                # Merge documents retrieved from bm25 and genre
                retrieved_docs = combine_documents(retrieved_docs_genre, retrieved_docs_bm25)
                all_docs_combined.append(retrieved_docs)
                # Rerank sentences in retrieved documents
                reranked_sentences, reranked_docs = dense_retriever.retrieve_sentences(qid, query, retrieved_docs)
                reranked_docs = combine_documents(reranked_docs, retrieved_docs)
                all_docs_reranked.append(reranked_docs)
                all_sentences_reranked.append(reranked_sentences)
            else:
                docs_previous_iteration_sample = docs_previous_iteration[j]
                sentences_previous_iteration_sample = sentences_previous_iteration[j]

                if qid in skip_retrieval:
                    # If sufficienct, just use documents from last iteration
                    all_docs_combined.append(docs_previous_iteration_sample)
                    all_docs_reranked.append(docs_previous_iteration_sample)
                    all_sentences_reranked.append(sentences_previous_iteration_sample)
                
                # Sufficiency check, whether to retrieve more evidence
                elif config.enable_sufficiency_check:
                    is_sufficient, sufficiency_proof = sufficiency_proofver.evidence_sufficiency_check(qid, query, sentences_previous_iteration[j])
                    all_sufficiency_proofs.append(sufficiency_proof)
                    if is_sufficient:
                        skip_retrieval.add(qid)
                    
                if not is_sufficient:
                    # If insufficienct, do multi hop retrieval
                    if not config.use_precomputed_genre:
                        retrieved_docs_genre, hop_documents_only = genre_retriever.retrieve_documents(qid, query, sentences_previous_iteration_sample)
                        retrieved_docs = combine_documents(retrieved_docs_genre, docs_previous_iteration_sample)
                        all_docs_combined.append(retrieved_docs)
                    else:
                        retrieved_docs = docs_previous_iteration[j]
                        hop_documents_only = hop_documents_only_previous_iteration[j]
                        all_docs_combined.append(retrieved_docs)
                    # Rerank these documents again using a cross-encoder reranker.
                    if not config.genre_only:
                        reranked_sentences, reranked_docs = dense_retriever.retrieve_sentences(qid, query, hop_documents_only, sentences_previous_iteration_sample)
                        reranked_docs = combine_documents(reranked_docs, retrieved_docs)
                        all_docs_reranked.append(reranked_docs)
                        all_sentences_reranked.append(reranked_sentences)
        if hop == 0:
            evaluate_iteration_initial(config, str(hop), bm25_retriever, genre_retriever, dense_retriever, all_docs_bm25, all_docs_genre, all_docs_combined, all_docs_reranked, all_sentences_reranked)
        else:
            evaluate_iteration(config, str(hop), dense_retriever, all_docs_combined, all_docs_reranked, all_sentences_reranked)

        docs_previous_iteration = copy.deepcopy(all_docs_reranked)
        sentences_previous_iteration = copy.deepcopy(all_sentences_reranked)

        del genre_retriever
        del dense_retriever

        torch.cuda.empty_cache()
        gc.collect()
        






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_files", required=True)
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    args = parser.parse_args()

    config = Config(args.config_files, args.kwargs)
    setup_spacy()
    init_logging(logging, config)
    print(f"Start experiment {config.exp_name}")
    # Setup config
    assert config.compute_strategy in ["none", "ddp", "deepspeed_stage_3_offload", "deepspeed_stage_3"]

    print(config.to_json())

    set_seeds(config.seed)
    main(config)
