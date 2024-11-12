# Installation

Setup a new conda environment, e.g. python3.9 (tested only Python version 3.9):

conda create -n admiral python=3.9
conda activate admiral
Then install AdMIRaL and all relevant dependencies:

python3 -m pip install -e .

The pipeline to incorporate retrieved data uses Pyserini. The dependencies are already installed, however you also need to download Java and place it into the root path of the repository: https://jdk.java.net/19/

Set Java Path: `export JAVA_HOME=$PWD/jdk-19.0.2/`


gdown --folder https://drive.google.com/drive/folders/1sObMn8YZ8GKxWUiRWMJqoMBnHRzASbKl?usp=sharing (checkpoint-436, bart_completion_none_space_only_docs_joint)
gdown --folder https://drive.google.com/drive/folders/1XxTTsYKJ89gLs-EUoSDdZH2nqd2BgGuy?usp=sharing (checkpoint-2800, models_hover_all_max_neg_800-multi-hop)
gdown --folder https://drive.google.com/drive/folders/1xjdrxDF9ZZm3INhZ4O8S80yJc8mh70ZD?usp=sharing (checkpoint-2260, sufficiency_classifier)
gdown --folder https://drive.google.com/drive/folders/1mUZ58Vh8k50a3iDYCgR8w4SSKw8fn2sk?usp=sharingring (models_hover_all_max_neg_800)

## If installing fairseq

Cone Repository fork by De cao 

git clon ehttps://github.com/nicola-decao/fairseq/tree/fixing_prefix_allowed_tokens_fn
python3 -m pip install --editable ./ 
python3 -m pip install omegaconf==2.0.0
python3 -m pip install hydra-core==1.0.0
python3 -m pip install numpy==1.23.0
python3 -m pip install pydantic==1.10.8
export PYTHONPATH=$PYTHONPATH:/home/rmya2/scratch/AdMIRaL_new/fairseq

## Build Wikipedia Trie (FEVEROUS)

python3 -m src.retrieval.genre.generate_trie --output_file datasets/feverous/genre_feverous_trie.pkl --mode start --database indexes/lucene-index-feverous-passage/ --dataset feverous --granularity passage

# Running AdMiRaL


```
./bin/run_admiral_stammbach.sh k1_09_b_01 fever top3_beam_20 10_docs_inpars none_2hop 42
```

## Running AdMIRaL on FEVEROUS
``
./bin/run_admiral.sh k1_09_b_01 feverous top3_beam_20 10_docs none_2hop 42
```

Then go to FEVEROUS repository, update input and output file, and run:
python -m feverous.baseline.retriever.table_tfidf_drqa --db ../data/feverous_wikiv1.db --split dev --max_page 5 --max_tabs 3 --use_precomputed false --data_path ../data/

Then check you have the right sentence and table file in `python3 -m src.utils.export_to_tabver` and run:

```
python3 -m src.utils.export_to_tabver
```

Then copy the resulting file over to TabVer. Adjust name of all intermediate files from previous run and then fill in retrieved evidence in existing intermedite files by running:

```
python3 -m src.utils.transfer_evidence_between_files.py
```


# Notes
This repository is a complete reimplementation of the original codebase and therefore deviates slightly from the paper:
- Instead of using the [Stammbach retriever](https://github.com/dominiksinsaarland/document-level-FEVER) for FEVER, we use a T5-reranker. The retrieval scores on FEVER are thus slightly lower. However, using a T5-reranker makes the codebase more flexibility, e.g. to incorporate longer documents (as needed for datasets like FEVEROUS). Please reach out to me if you want the retrieval results for FEVER shown in the paper.
- We only keep track of the top i=1 D^i_t document set at a given iteration. Since most evaluation metrics consider recall@k with k >> 1, we fill up with documents d_t selected in the current iteration. 


## Training
python3 -m src.training.train_bart --input_path data/fever/training_data/genre_sufficiency_check/ --output_model_path models/sufficiency_proofver --save_every_n_steps 100 --epochs 10 --add_space


## Train Reranker:

Run entity_mention_matcher
run bm25_train

python3.7 experiments/list5/merge_runs.py  --input_run_file /local/scratch/rmya2/Retrieval-Benchmark/runs/initial_retriever/hover/hover-docs-script-train.bm25tuned_k04_b04.tsv  --input_run_file /local/scratch/rmya2/Retrieval-Benchmark/runs/initial_retriever/hover/entity_matcher_train.tsv  --output_run_file runs/list5/run.hover-paragraph.train.tsv  --strategy zip

python experiments/list5/expand_docs_to_sentences.py  --input_run_file runs/list5/run.hover-paragraph.train.tsv  --collection_folder ../Retrieval-Benchmark/datasets/HoVer/enwiki-20171001-pages-meta-current-withlinks-abstracts/  --output_run_file runs/list5/run.hover-sentence-top-200.train.tsv --k 200

python experiments/list5/generate_sentence_selection_data.py --dataset_file /local/scratch/rmya2/Retrieval-Benchmark/datasets/HoVer/hover_train_release_v1.1.json --run_file runs/list5/run.hover-sentence-top-200.train.tsv --collection_folder /local/scratch/rmya2/Retrieval-Benchmark/datasets/HoVer/enwiki-20171001-pages-meta-current-withlinks-abstracts/ --output_id_file data/list5/query-doc-pairs-id-train-model-rerank-top-200-hover.txt  --output_text_file data/list5/query-doc-pairs-text-train-rerank-top-200-hover.txt --min_rank 50 --max_rank 200

python pygaggle/run/finetune_monot5.py --base_model castorini/monot5-large-msmarco --input_path data/list5/query-doc-pairs-text-train-rerank-top-200-hover.txt --output_model_path models_hover_all




# Pipeline
1. `./scrips/run_bm25_document_retrieval.sh` -> Bm25 Ranking
2. `generate_genre_trie.py`
3. `run_genre.py -> GENRE document predictions
4. `merge_predictions.py` -> merges GENRE and Bm25
5. `run_bm25_pipeline_seq2seq_retrieval.sh` -> initial sentence prediction
For each hop:
1. `run_genre_completion.py` -> Generates missing documents
2. `run_genre_completion.py` -> Generates missing sentences
3. `run_bm25_pipeline_seq2seq_retrieval_completion.sh` -> Updates evidence candidates.

# FEVER
## Initial Retriever
`runs/initial_retriever/fever-pipeline-seq2seq_finetuned_long.bm25tuned.txt`

## Document Generation

File with best results: `runs/document_generation/fever-bart-completion_docs_only_no_labels_finetuned.txt`


# HoVer

Generate Trie:
```
PYTHONPATH=src python3.7 src/utils/generate_genre_trie.py --output_file datasets/HoVer/hover_trie.pkl --database indexes/lucene-index-hover-docs/ --dataset hover --granularity paragraph
```

Run GENRE for documents:
```
PYTHONPATH=src python3.7 src/initial_retriever/run_genre.py --output_file runs/initial_retriever/hover-docs-script-genre.txt --database indexes/lucene-index-hover-docs/ --dataset hover --granularity paragraph --trie_path datasets/HoVer/hover_trie.pkl --mode start --claim_file datasets/HoVer/hover_dev_release_v1.1.json
```

Merge predictions:
``
PYTHONPATH=src python3.7 src/initial_retriever/merge_predictions.py --document_prediction_file_1 runs/initial_retriever/hover-
docs-script-genre.txt --document_prediction_file_2 runs/hover-docs-script.bm25tuned.txt  --query_file datasets/HoVer/hover_dev_release_v1.1.json --top_file_1 2 --top_file_2 100 --output_file runs/initial_retriever/hover/hover-docs-merged.txt --dataset hover
``

Run:
```
run_bm25_pipeline_seq2seq_retrieval.sh
```

## Document Generation


```
PYTHONPATH=src python3.7 src/query_generation_model/run_genre_completion_joint.py --claim_file datasets/FEVER/shared_task_dev.jsonl --run_path runs/initial_retriever/fever-pipeline-seq2seq_finetuned_long.bm25tuned.txt --database indexes/lucene-index-fever-sentences-script/ --database_pipeline indexes/lucene-index-fever-pipeline/ --model models/bart_completion_none_space_only_docs_no_labels_joint/ --output_file runs/document_generation_joint/fever-bart-completion_finetuned_only_docs_no_labels_joint_top_4_beam_20.txt --constrained_beam --constrain documents --num_beams 20 --num_initial_sentences 4 --num_return_sequences 20 --trie_path datasets/FEVER/completion_trie.pkl --dataset fever
```

Evidence length 6.789752826978885
R-precision : 0.5849094366056239
R-precision more one : 0.32577903682719545
R-precision more one not strict : 0.6420668645172894
k       Fully Supported Oracle Accuracy
1       0.8597  0.9062
3       0.9347  0.9563
5       0.9485  0.9654
10      0.9533  0.9687
25      0.9536  0.9688
50      0.9536  0.9688
100     0.9536  0.9688
k       Fully Supported
1       0.0000
3       0.5524
5       0.6808
10      0.7214
25      0.7214
50      0.7214
100     0.7214

## Sentence Generator
Create training dataset:
```
PYTHONPATH=src python3.7 src/query_generation_model/convert_to_genre_completion_latent_ordering.py --output_path datasets/FEVER/genre_completion_latent_ordering_no_labels_output_space --database_sentence indexes/lucene-index-fever-sentences-script/ --database_pipeline indexes/lucene-index-fever-pipeline/ --prediction_file runs/initial_retriever/fever-train.genre.txt --split_name train --claim_file datasets/FEVER/train.jsonl --dataset fever --format papelo
```
Train model:
```
PYTHONPATH=src python3.7 src/query_generation_model/train_completion_model.py --input_path datasets/FEVER/genre_completion_latent_ordering_no_labels_output_space/ --output_model_path models/bart_completion_none_space_no_labels_latent_ordering --save_every_n_steps 3000 --epochs 5
```
