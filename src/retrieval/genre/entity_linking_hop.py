# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import torch

from src.retrieval.genre.trie import DummyTrieEntity, DummyTrieMention, Trie
from pyserini.search import SimpleSearcher
import json
import traceback
import string
import nltk
from nltk.corpus import stopwords
import spacy
import sys

def get_end_to_end_prefix_allowed_tokens_fn_hf(
    model,
    sentences: List[str],
    start_mention_token="[",
    end_mention_token="]",
    start_entity_token="{",
    end_entity_token="}",
    mention_trie: Trie = None,
    candidates_trie: Trie = None,
    sentence_trie: Trie = None,
    mention_to_candidates_dict: Dict[str, List[str]] = None,
    document_pipeline = None,
    generate_output_labels = None,
):
    return _get_end_to_end_prefix_allowed_tokens_fn(
        lambda x: model.tokenizer.encode(x),
        lambda x: model.tokenizer.decode(torch.tensor(x)),
        model.tokenizer.bos_token_id,
        model.tokenizer.pad_token_id,
        model.tokenizer.eos_token_id,
        len(model.tokenizer) - 1,
        sentences,
        start_mention_token,
        end_mention_token,
        start_entity_token,
        end_entity_token,
        mention_trie,
        candidates_trie,
        sentence_trie,
        mention_to_candidates_dict,
        document_pipeline,
        generate_output_labels
    )


def get_end_to_end_prefix_allowed_tokens_fn_fairseq(
    model,
    sentences: List[str],
    start_mention_token="{",
    end_mention_token="}",
    start_entity_token="[",
    end_entity_token="]",
    mention_trie: Trie = None,
    candidates_trie: Trie = None,
    sentence_trie: Trie = None,
    mention_to_candidates_dict: Dict[str, List[str]] = None,
    document_pipeline = None,
    generate_output_labels = None
):
    return _get_end_to_end_prefix_allowed_tokens_fn(
        lambda x: model.encode(x).tolist(),
        lambda x: model.decode(torch.tensor(x)),
        model.model.decoder.dictionary.bos(),
        model.model.decoder.dictionary.pad(),
        model.model.decoder.dictionary.eos(),
        len(model.model.decoder.dictionary),
        sentences,
        start_mention_token,
        end_mention_token,
        start_entity_token,
        end_entity_token,
        mention_trie,
        candidates_trie,
        sentence_trie,
        mention_to_candidates_dict,
        document_pipeline,
        generate_output_labels
    )


def _get_end_to_end_prefix_allowed_tokens_fn(
    encode_fn,
    decode_fn,
    bos_token_id,
    pad_token_id,
    eos_token_id,
    vocabulary_length,
    sentences: List[str],
    start_mention_token="{",
    end_mention_token="}",
    start_entity_token="[",
    end_entity_token="]",
    mention_trie: Trie = None,
    candidates_trie: Trie = None,
    sentence_trie: Trie = None,
    mention_to_candidates_dict: Dict[str, List[str]] = None,
    document_pipeline = None,
    generate_output_labels = None
):

    assert not (
        candidates_trie is not None and mention_to_candidates_dict is not None
    ), "`candidates_trie` and `mention_to_candidates_dict` cannot be both != `None`"

    codes = {
        n: encode_fn(" {}".format(c))[1]
        for n, c in zip(
            (
                "start_mention_token",
                "end_mention_token",
                "start_entity_token",
                "end_entity_token",
            ),
            (
                start_mention_token,
                end_mention_token,
                start_entity_token,
                end_entity_token,
            ),
        )
    }
    codes["EOS"] = eos_token_id

    if mention_trie is None:
        mention_trie = DummyTrieMention(
            [
                i
                for i in range(vocabulary_length)
                if i
                not in (
                    bos_token_id,
                    pad_token_id,
                )
            ]
        )
    #
    # content = [
    #     encode_fn("{}".format(e))[1:]
    #     for e in [' ] E0', ' ] E1', ' ] E2', ' ] E3', ' ] E4']
    # ]
    # content += [[2, 0, 646]]
    #
    # content = [item for sublist in content for item in sublist]
    # print(content)
    #
    # print(decode_fn([2, 0, 646]))
    # print(decode_fn([381, 288, 134, 27779]))
    # #
    # # sys.exit()
    #
    # sentence_trie=DummyTrieMention(content)
    # sentence_trie=Trie([
    #     encode_fn("{}".format(e)[1:])
    #     for e in [" E0", " E1", " E2", " E3", " E4", ' ', ' [', ' [ ', '] ', ' ] ', '[ ', '] ', '] ', '] E', '] E0', '] E1', '] E2', '] E3', '] E4', ']', '[']
    # ] + [[2]] + [[0]]  + [[646]] + [[1]])

    sentence_trie = DummyTrieMention(
        [
            i
            for i in range(vocabulary_length)
            if i
            not in (
                pad_token_id,
            )
        ]
    )
    # sentence_trie = DummyTrieMention(
    #     [
    #         i
    #         for i in range(vocabulary_length)
    #         if i
    #         not in (
    #             pad_token_id,
    #         )
    #     ]
    # )

    if candidates_trie is None and mention_to_candidates_dict is None:
        candidates_trie = DummyTrieEntity(
            [
                i
                for i in range(vocabulary_length)
                if i
                not in (
                    bos_token_id,
                    pad_token_id,
                )
            ],
            codes,
        )
    if document_pipeline is not None:
        searcher = SimpleSearcher(document_pipeline)

    general_vocab = set()

    for x in string.punctuation:
        general_vocab = general_vocab | set(encode_fn(x))

    nlp = spacy.load('en_core_web_sm')
    all_stopwords = nlp.Defaults.stop_words
    for x in all_stopwords:
        general_vocab = general_vocab | set(encode_fn(x))

    # print(general_vocab)


    # sent_origs = [[codes["EOS"]] + encode_fn(sent)[1:] for sent in sentences]
    # print(sent_origs)

    def prefix_allowed_tokens_fn(batch_id, sent):
        sent = sent.tolist()
        status = get_status(sent)

        # trie_out = get_trie_mention(sent)
        # trie_out = get_trie_outside(sent)

        if status == "l":
            trie_out = get_trie_outside_labels(sent)
        elif status == "s":
            trie_out = get_trie_outside_labels_supports(sent)
        elif status == "r":
            trie_out = get_trie_outside_labels_refutes(sent)
        elif status == "o":
            trie_out = get_trie_outside(sent)
        elif status == "m":
            trie_out = get_trie_mention(sent)
        elif status == 'c':
            trie_out = get_trie_outside_constraint(sent)
            # if trie_out == codes["EOS"]:
            #     trie_out = get_trie_outside(sent)
        # else:
        #     raise RuntimeError

        return trie_out

    def get_status(sent):
        c = [
            codes[e]
            for e in (
                "start_mention_token",
                "end_mention_token",
            )
        ]

        status = sum(e in c for e in sent)
        if status == 0 and generate_output_labels is not None and generate_output_labels is not False:
            # print('yao')
            if generate_output_labels == 'refutes':
                return "r"
            elif generate_output_labels == 'supports':
                return 's'
            else:
                return 'l'
        elif status == 0:
            return "o"
            # return "m"
        elif status == 1:
            # print('in  generation')
            return "m"
        elif status > 1:
            if document_pipeline != None:
                # print('mee')
                return 'c'
            else:
                return "o"


    def get_trie_outside(sent):
        # print(sentence_trie.get([]))
        # pointer_end = get_pointer_end(sent, sent_orig)
        return sentence_trie.get(sent)

    def get_trie_outside_labels(sent):
        trie=Trie([
            [2]  + encode_fn("{}".format(e))[1:]
            # for e in ["REFUTES [", "SUPPORTS [", "None"]
            for e in ["REFUTES ["]
        ])
        # print(sentence_trie.get([]))
        # pointer_end = get_pointer_end(sent, sent_orig)
        return trie.get(sent)

    def get_trie_outside_labels_supports(sent):
        trie=Trie([
            [2]  + encode_fn("{}".format(e))[1:]
            for e in ["SUPPORTS ["]
        ])
        # print(sentence_trie.get([]))
        # pointer_end = get_pointer_end(sent, sent_orig)
        return trie.get(sent)

    def get_trie_outside_labels_refutes(sent):
        trie=Trie([
            [2]  + encode_fn("{}".format(e))[1:]
            for e in ["REFUTES ["]
        ])
        # print(sentence_trie.get([]))
        # pointer_end = get_pointer_end(sent, sent_orig)
        return trie.get(sent)

    def get_trie_outside_constraint(sent):
        pointer_start, pointer_end = get_pointer_mention(sent)

        mention = decode_fn(sent[pointer_start + 1 : pointer_end]).strip()

        mention_formatted = mention.strip().replace("( ", "-LRB-").replace(" )", "-RRB-").replace(":", "-COLON-").replace(' ', '_')
        doc_element = searcher.doc(mention_formatted)
        if doc_element == None:
            return sentence_trie.get(sent)
        doc_content = json.loads(doc_element.raw())['contents']
        corpus = []
        for li in doc_content.split('\n'):
            if li == '':
                continue
            if li.split('\t')[0].isnumeric():
                corpus.append(li.split('\t')[1])
        try:
            id_list = set()#general_vocab#set()
            for entity in corpus:
                id_list = id_list | set(encode_fn(entity))

            trie = DummyTrieMention(
                [
                    i
                    for i in range(vocabulary_length)
                    if i in id_list
                ]
            )

        except Exception as e:
            print('yao')
            print(traceback.format_exc())
            return sentence_trie.get(sent)

        return trie.get(sent[pointer_end+1:])

    def get_trie_mention(sent):
        # print(sent)
        pointer_start, pointer_end = get_pointer_mention(sent)
        # print(pointer_start)
        # mention = decode_fn(sent[pointer_start]).strip()
        # print(mention)
        if mention_trie is not None:
            mention_trie_tmp = mention_trie
        # print(sent)
        # print(pointer_start, pointer_end)
        # print(mention_trie_tmp.get([]))
        # print(mention_trie_tmp.get(sent[pointer_start:]))
        # print('---------------------')
        return mention_trie_tmp.get(sent[pointer_start:])


    def get_pointer_mention(sent):
        pointer_start = -1
        pointer_end = -1
        for i, e in enumerate(sent):
            if e == codes["start_mention_token"]:
                pointer_start = i
            elif e == codes["end_mention_token"]:
                pointer_end = i

        return pointer_start, pointer_end

    return prefix_allowed_tokens_fn
