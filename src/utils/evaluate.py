import argparse
import json
import dataset_readers
import numpy as np
import copy
import unicodedata
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
"""
Evaluates the (1) percentage of fully supported claims and (2) oracle accuracy
for retrieval given an anserini run file. Intends to replicate the metrics in
Table 2 in this paper: https://www.aclweb.org/anthology/N18-1074.pdf.
"""


def load_annotations(dataset, truth_file, reader, evaluate_type):
    evidences = {}
    num_queries = 0
    num_supported_queries = 0
    num_nei_queries = 0
    num_queries_more_one = 0
    ids_more_one_queries = []
    num_queries_more_two = 0
    ids_more_two_queries = []
    queries = {}

    iterator = reader.read_annotations(truth_file)

    for i, output_dict in enumerate(iterator):
        query_id, query, label, evidence = output_dict
        if label in ['REFUTES', 'CONTRADICT', 'NOT_SUPPORTED'] and evaluate_type=='supported':
            evidence = set([])
        elif label in ['SUPPORTS', 'SUPPORT', 'SUPPORTED'] and evaluate_type=='refuted':
            evidence = set([])
        elif label == 'NOT ENOUGH INFO' and dataset == "fever":
            evidence = set([])
            num_nei_queries +=1
        elif not evidence: # For tabular data or text data, not all samples necessarily have it.
            evidence = set([])
        else:
            num_supported_queries += 1
            queries[query_id] = query
            if dataset == 'hover':
                threshold =  all([len(list(set(ev))) == 2 for ev in evidence])
                threshold_two = all([len(list(set(ev))) == 3 for ev in evidence])
            else:
                threshold =  all([len(list(set(ev))) > 1 for ev in evidence])
                threshold_two = all([len(list(set(ev))) > 1 for ev in evidence])

            if threshold:
                num_queries_more_one +=1
                ids_more_one_queries.append(query_id)
            if threshold_two:
                num_queries_more_two +=1
                ids_more_two_queries.append(query_id)

        evidences[query_id] = list(evidence)
        num_queries += 1
    return evidences, queries, num_queries, num_supported_queries, num_nei_queries, num_queries_more_one, num_queries_more_two, ids_more_one_queries, ids_more_two_queries

def evaluate_r_precision(query_id, evidences, predicted_docs, ids_more_one_queries, ids_more_two_queries):
    r_precision = []
    r_precision_more_one = []
    r_precision_more_one_not_strict = []

    if not evidences[query_id]:
        return [], [] ,[]
    max_r_precision = 0
    max_r_precision_not_strict = 0
    for evid_set in evidences[query_id]:
        r_precision_value_not_strict = len([x for x in predicted_docs[:len(evid_set)] if x in evid_set]) / len(evid_set)
        r_precision_value = 1 if len([x for x in predicted_docs[:len(evid_set)] if x in evid_set])  == len(evid_set) else 0
        if r_precision_value > max_r_precision:
            max_r_precision = r_precision_value
        if r_precision_value_not_strict > max_r_precision_not_strict:
            max_r_precision_not_strict = r_precision_value_not_strict

    r_precision.append(max_r_precision)
    if query_id in ids_more_one_queries:
            r_precision_more_one.append(max_r_precision)
            r_precision_more_one_not_strict.append(max_r_precision_not_strict)
    
    return r_precision, r_precision_more_one, r_precision_more_one_not_strict


def evaluate_em(query_id, evidences, predicted_docs, ids_more_one_queries, ids_more_two_queries):
    exact_match = []
    exact_match_more_one = []
    exact_match_more_two = []

    if not evidences[query_id]:
        return [], [], []
    max_exact_match = 0
    for evid_set in evidences[query_id]:
        temp = predicted_docs[:len(evid_set)]
        if all([x in temp for x in evid_set]):
        # if any([x in evid_set for x in temp]):
            max_exact_match = 1
    exact_match.append(max_exact_match)
    if query_id in ids_more_one_queries:
            exact_match_more_one.append(max_exact_match)
    if query_id in ids_more_two_queries:
            exact_match_more_two.append(max_exact_match)
    
    return exact_match, exact_match_more_one, exact_match_more_two

# evaluates whether a query's predicted docs covers one complete set of evidences
def evaluate(query_id, evidences, predicted_docs, top_ks, ids_more_one_queries, ids_more_two_queries):
    num_predictions = {k: 0 for k in top_ks}
    num_predictions_more_one = {k: 0 for k in top_ks}
    num_predictions_more_two = {k: 0 for k in top_ks}

    fully_supported = {k: 0 for k in top_ks}
    correct_count = {k: 0 for k in top_ks}
    oracle_accuracy = {k: 0 for k in top_ks}

    fully_supported_more_one = {k: 0 for k in top_ks}
    correct_count_more_one = {k: 0 for k in top_ks}

    fully_supported_more_two = {k: 0 for k in top_ks}
    correct_count_more_two = {k: 0 for k in top_ks}


    all_evi = [item for sublist in evidences[query_id] for item in sublist]
    for k in top_ks:
        # print(evidences[query_id])
        if not evidences[query_id]:  # query is labelled "NOT ENOUGH INFO"
            oracle_accuracy[k] += 1
        else:  # query is labelled "SUPPORTS" or "REFUTES"
            for ev in predicted_docs[:k]:
                if ev in all_evi:
                    correct_count[k] +=1
                    if  query_id in ids_more_one_queries:
                        correct_count_more_one[k] +=1
                    if  query_id in ids_more_two_queries:
                        correct_count_more_two[k] +=1
            num_predictions[k] += len(predicted_docs[:k])
            if  query_id in ids_more_one_queries:
                num_predictions_more_one[k] += len(predicted_docs[:k])
            if  query_id in ids_more_two_queries:
                num_predictions_more_two[k] += len(predicted_docs[:k])
            # print(predicted_docs[:k], evidences[query_id])
            for evid_set in evidences[query_id]:
                if all([evid in predicted_docs[:k] for evid in evid_set]):
                    fully_supported[k] += 1
                    oracle_accuracy[k] += 1
                    if  query_id in ids_more_one_queries:
                        fully_supported_more_one[k] +=1
                    if  query_id in ids_more_two_queries:
                        fully_supported_more_two[k] +=1
                    break
    
    return num_predictions, num_predictions_more_one, num_predictions_more_two, fully_supported, correct_count, oracle_accuracy, fully_supported_more_one, correct_count_more_one, fully_supported_more_two, correct_count_more_two


def evaluate_retrieval(dataset, truth_file, run_file, save_path, evaluate_type, granularity):
    # return # FOR TEST SET EVALUATION SIMPLY SKIP
    top_ks = [1, 2, 3, 5, 10, 25, 50, 100]
    num_predictions, num_predictions_more_one, num_predictions_more_two, fully_supported, correct_count, oracle_accuracy, fully_supported_more_one, correct_count_more_one, fully_supported_more_two, correct_count_more_two = ({k: 0 for k in top_ks} for _ in range(10))
    r_precision, r_precision_more_one, r_precision_more_one_not_strict = ([] for _ in range(3))
    exact_match, exact_match_more_one, exact_match_more_two = ([] for _ in range(3))
    total_num_predictions = set()

    reader = dataset_readers.get_class(dataset)(**{'granularity':granularity})

    evidences, queries, num_queries, num_supported_queries, num_nei_queries, num_queries_more_one, num_queries_more_two, ids_more_one_queries, ids_more_two_queries = load_annotations(dataset, truth_file, reader, evaluate_type)

    # read in run file and calculate metrics: % of fully supported and oracle accuracy
    iterator = reader.read_predictions(run_file, max_evidence=20)
    predicted_evidences = {x[0]: x[1] for x in iterator}

    average_len = []
    for query_id in queries:
        predicted_evidence = predicted_evidences[query_id] if query_id in predicted_evidences else [] # if no evidence exists for given id, e.g. in the case of tables
        # if not predicted_evidence:
        #     continue
    # for i, output_dict in enumerate(iterator):
    #     query_id, predicted_evidence, rank, _, _ = output_dict
        predicted_evidence = [unicodedata.normalize('NFD', doc_id) for doc_id in predicted_evidence]
        if query_id in queries:
            total_num_predictions.add(query_id)
        average_len.append(len(predicted_evidence))
        num_predictions_ele, num_predictions_more_one_ele, num_predictions_more_two_ele, fully_supported_ele, correct_count_ele, oracle_accuracy_ele, fully_supported_more_one_ele, correct_count_more_one_ele, fully_supported_more_two_ele, correct_count_more_two_ele = evaluate(query_id, evidences, list(dict.fromkeys(predicted_evidence)), top_ks, ids_more_one_queries, ids_more_two_queries)

        for j, curr_dict in enumerate([num_predictions, num_predictions_more_one, num_predictions_more_two, fully_supported, correct_count, oracle_accuracy, fully_supported_more_one, correct_count_more_one, fully_supported_more_two, correct_count_more_two]):
            curr_ele_dict = [num_predictions_ele, num_predictions_more_one_ele, num_predictions_more_two_ele, fully_supported_ele, correct_count_ele, oracle_accuracy_ele, fully_supported_more_one_ele, correct_count_more_one_ele, fully_supported_more_two_ele, correct_count_more_two_ele][j]
            for key, value in curr_dict.items():
                curr_dict[key] += curr_ele_dict[key]

        r_precision_ele, r_precision_more_one_ele, r_precision_more_one_not_strict_ele = evaluate_r_precision(query_id, evidences, list(dict.fromkeys(predicted_evidence)),  ids_more_one_queries, ids_more_two_queries)
        r_precision += r_precision_ele
        r_precision_more_one += r_precision_more_one_ele
        r_precision_more_one_not_strict += r_precision_more_one_not_strict_ele

        exact_match_ele, exact_match_more_one_ele, exact_match_more_two_ele = evaluate_em(query_id, evidences, list(dict.fromkeys(predicted_evidence)), ids_more_one_queries, ids_more_two_queries)
        exact_match += exact_match_ele
        exact_match_more_one += exact_match_more_one_ele
        exact_match_more_two += exact_match_more_two_ele
    
    scores = calculate_and_print_scores(
        average_len, r_precision, r_precision_more_one, r_precision_more_one_not_strict,
        exact_match, exact_match_more_one, exact_match_more_two, total_num_predictions,
        top_ks, fully_supported, oracle_accuracy, correct_count, num_predictions,
        fully_supported_more_one, correct_count_more_one, num_predictions_more_one,
        fully_supported_more_two, correct_count_more_two, num_predictions_more_two,
        num_queries, num_queries_more_one, num_queries_more_two, ids_more_one_queries, 
        ids_more_two_queries
    )

    save_scores_to_json(scores, save_path)


def save_scores_to_json(scores, filename):
    with open(filename, 'w') as f:
        json.dump(scores, f, indent=4)

def calculate_and_print_scores(
    average_len, r_precision, r_precision_more_one, r_precision_more_one_not_strict,
    exact_match, exact_match_more_one, exact_match_more_two, total_num_predictions,
    top_ks, fully_supported, oracle_accuracy, correct_count, num_predictions,
    fully_supported_more_one, correct_count_more_one, num_predictions_more_one,
    fully_supported_more_two, correct_count_more_two, num_predictions_more_two,
    num_queries, num_queries_more_one, num_queries_more_two, ids_more_one_queries, 
    ids_more_two_queries
):
    scores = {}

    print('Evaluating on {} samples'.format(len(total_num_predictions)))
    num_supported_queries = len(total_num_predictions)
    num_queries_more_one = len([x for x in ids_more_one_queries if x in total_num_predictions])

    # Calculate and print evidence length
    evidence_length = sum(average_len) / len(average_len)
    print('Evidence length: {:.4f}'.format(evidence_length))
    scores['evidence_length'] = evidence_length

    # Calculate and print R-precision
    r_precision_score = sum(r_precision) / len(r_precision)
    print('R-precision: {:.4f}'.format(r_precision_score))
    scores['r_precision'] = r_precision_score

    # Calculate and print R-precision more one
    if len(r_precision_more_one) > 0:
        r_precision_more_one_score = sum(r_precision_more_one) / len(r_precision_more_one)
    else:
        r_precision_more_one_score = 0
    print('R-precision more one: {:.4f}'.format(r_precision_more_one_score))
    scores['r_precision_more_one'] = r_precision_more_one_score

    # Calculate and print R-precision more one not strict
    if len(r_precision_more_one) > 0:
        r_precision_more_one_not_strict_score = sum(r_precision_more_one_not_strict) / len(r_precision_more_one)
    else:
        r_precision_more_one_not_strict_score = 0
    print('R-precision more one not strict: {:.4f}'.format(r_precision_more_one_not_strict_score))
    scores['r_precision_more_one_not_strict'] = r_precision_more_one_not_strict_score

    print('---------------------')

    # Calculate and print exact match
    exact_match_score = sum(exact_match) / len(exact_match)
    print('Exact match: {:.4f}'.format(exact_match_score))
    scores['exact_match'] = exact_match_score

    # Calculate and print exact match more one
    if len(exact_match_more_one) > 0:
        exact_match_more_one_score = sum(exact_match_more_one) / len(exact_match_more_one)
    else:
        exact_match_more_one_score = 0
    print('Exact match more one: {:.4f}'.format(exact_match_more_one_score))
    scores['exact_match_more_one'] = exact_match_more_one_score

    # Calculate and print exact match more two
    if len(exact_match_more_two) > 0:
        exact_match_more_two_score = sum(exact_match_more_two) / len(exact_match_more_two)
    else:
        exact_match_more_two_score = 0
    print('Exact match more two: {:.4f}'.format(exact_match_more_two_score))
    scores['exact_match_more_two'] = exact_match_more_two_score

    print('---------------------')

    print('k\tFully Supported\tOracle Accuracy\tPrecision\tF1')
    for k in top_ks:
        recall_score = fully_supported[k] / num_supported_queries
        precision_score = correct_count[k] / num_predictions[k]
        if (recall_score + precision_score) > 0:
            f1 = 2 * recall_score * precision_score / (recall_score + precision_score)
        else:
            f1 = 0
        print('{:.0f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(k, recall_score, oracle_accuracy[k] / num_queries, precision_score, f1))
        scores[f'k_{k}_fully_supported'] = recall_score
        scores[f'k_{k}_oracle_accuracy'] = oracle_accuracy[k] / num_queries
        scores[f'k_{k}_precision'] = precision_score
        scores[f'k_{k}_f1'] = f1

    print('k\tOne hop')
    for k in top_ks:
        if num_queries_more_one > 0:
            recall_score = fully_supported_more_one[k] / num_queries_more_one
            precision_score = correct_count_more_one[k] / num_predictions_more_one[k]
            if (recall_score + precision_score) > 0:
                f1 = 2 * recall_score * precision_score / (recall_score + precision_score)
            else:
                f1 = 0
        else:
            recall_score = 0
            precision_score = 0
            f1 = 0
        print('{:.0f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(k, recall_score, precision_score, f1))
        scores[f'k_{k}_one_hop_recall'] = recall_score
        scores[f'k_{k}_one_hop_precision'] = precision_score
        scores[f'k_{k}_one_hop_f1'] = f1

    print('k\tTwo hop')
    for k in top_ks:
        if num_queries_more_two > 0:
            recall_score = fully_supported_more_two[k] / num_queries_more_two
            if num_predictions_more_two[k] > 0:
                precision_score = correct_count_more_two[k] / num_predictions_more_two[k]
            else:
                precision_score = 0
            if (recall_score + precision_score) > 0:
                f1 = 2 * recall_score * precision_score / (recall_score + precision_score)
            else:
                f1 = 0
        else:
            recall_score = 0
            precision_score = 0
            f1 = 0
        print('{:.0f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(k, recall_score, precision_score, f1))
        scores[f'k_{k}_two_hop_recall'] = recall_score
        scores[f'k_{k}_two_hop_precision'] = precision_score
        scores[f'k_{k}_two_hop_f1'] = f1

    return scores
