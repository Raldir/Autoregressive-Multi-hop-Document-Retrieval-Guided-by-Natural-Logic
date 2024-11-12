#
# Pyserini: Python interface to the Anserini IR toolkit built on Lucene
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
The name of this file is a bit misleading since the original FEVER dataset is
also in JSONL format. This script converts them into a JSONL format compatible
with anserini.
"""

import json
import os
import argparse
import logging

import dataset_readers

logger = logging.getLogger(__name__)



def convert_collection(dataset, raw_folder, output_folder, max_docs_per_file, granularity):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    logger.info('Converting retrieval collection...')


    reader = dataset_readers.get_class(dataset)(**{'granularity':granularity})

    iterator = reader.read_corpus(raw_folder)
    doc_index = 0
    file_index=0
    for i, output_dict in enumerate(iterator):
        if doc_index % max_docs_per_file == 0:
            if doc_index > 0:
                output_jsonl_file.close()
            output_path = os.path.join(output_folder, f'docs{file_index:02d}.json')
            output_jsonl_file = open(output_path, 'w', encoding='utf-8', newline='\n')
            file_index += 1
        output_jsonl_file.write(json.dumps(output_dict) + '\n')
        doc_index += 1

        if doc_index % 100000 == 0:
            logger.info('Converted {} docs in {} files'.format(doc_index, file_index))


    output_jsonl_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts FEVER jsonl wikipedia dump to anserini jsonl files.')
    parser.add_argument('--collection_folder', required=True, help='FEVER wiki-pages directory.')
    parser.add_argument('--output_folder', required=True, help='Output directory.')
    parser.add_argument('--max_docs_per_file',
                        default=1000000,
                        type=int,
                        help='Maximum number of documents in each jsonl file.')
    parser.add_argument('--granularity',
                        required=True,
                        choices=['paragraph', 'sentence', 'pipeline'],
                        help='The granularity of the source documents to index. Either "paragraph", "sentence", or "pipeline. The latter indexes document title to its individual sentences.')
    parser.add_argument('--dataset',
                        required=True,
                        choices=['fever', 'feverous', 'scifact', 'tabfact', 'hover'],
                        help='The dataset.')
    args = parser.parse_args()


    convert_collection(dataset=args.dataset, raw_folder=args.collection_folder, output_folder=args.output_folder, max_docs_per_file=args.max_docs_per_file, granularity=args.granularity)

    print('Done!')
