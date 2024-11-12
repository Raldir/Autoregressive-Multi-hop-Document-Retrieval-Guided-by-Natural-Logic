# from genre.fairseq_model import GENRE
from src.retrieval.genre.hf_model import GENRE
from src.retrieval.genre.trie import Trie
import pickle
import argparse
import dataset_readers
from tqdm import tqdm
from pyserini.index import IndexReader

def generate_trie(args):
    model =  GENRE.from_pretrained("models/hf_wikipage_retrieval").eval()

    reader = dataset_readers.get_class(args.dataset)(**{'granularity':'paragraph'})

    lucene_reader = IndexReader(args.database)

    entities = []

    print("NUM ARTICLES: {}".format(lucene_reader.reader.maxDoc()))
    for i in tqdm(range(lucene_reader.reader.maxDoc())):
        doc_name = lucene_reader.reader.document(i).get('id')
        if args.dataset == 'fever':
            doc_name = reader.process_title(doc_name.replace('_', ' '))
        elif args.dataset == "feverous":
            doc_name = reader.process_title(doc_name)

        if args.mode == 'multi-hop':
            entities.append(' [ {} ] '.format(doc_name))
        elif args.mode == 'start':
            entities.append(doc_name)

    if args.mode == 'start':
        trie = Trie( [2] + model.encode(entity)[1:].tolist() for entity in entities).trie_dict
    elif args.mode == 'multi-hop':
        trie = Trie( model.encode(entity)[1:].tolist() for entity in entities).trie_dict
    with open(args.output_file, 'wb') as w_f:
        pickle.dump(trie, w_f)
    print("finish running!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Constructs the wikipedia trie for GENRE retrieval.')
    parser.add_argument('--output_file', required=True, help='Wikipedia Trie output path.')
    parser.add_argument('--mode', required=True, help='Whether trie is for initial retrieval or for multi-hop retrieval')
    parser.add_argument('--database', required=True, help='Pyserini index path for knowledge base')
    parser.add_argument('--dataset',
                        required=True,
                        choices=['fever', 'feverous', 'scifact', 'tabfact', 'hover'],
                        help='The dataset.')
    parser.add_argument('--granularity',
                        required=True,
                        choices=['paragraph', 'sentence'],
                        help='The granularity of the source documents to index. Either "paragraph" or "sentence".')
    args = parser.parse_args()

    generate_trie(args)
