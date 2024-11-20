# Description

This repository maintains the code for **AdMIRaL**, with the associated EMNLP 2022 paper: [Natural Logic-guided Autoregressive Multi-hop Document Retrieval
for Fact Verification](https://arxiv.org/abs/2406.13124).

> A key component of fact verification is the evidence retrieval, often from multiple documents. Recent approaches use dense representations and condition the retrieval of each document on the previously retrieved ones. The latter step is performed over all the documents in the collection, requiring storing their dense representations in an index, thus incurring a high memory footprint. An alternative paradigm is retrieve-and-rerank, where  documents are retrieved using methods such as BM25, their sentences are reranked, and further documents are retrieved  conditioned on these sentences, reducing the memory requirements. However, such approaches can be brittle as they rely on heuristics and assume hyperlinks between documents. We propose a novel retrieve-and-rerank method for multi-hop retrieval, that consists of a retriever that 
jointly scores documents in the knowledge source and sentences from previously retrieved documents using an autoregressive formulation and is guided by a proof system based on natural logic that dynamically terminates the retrieval process if the evidence is deemed sufficient. This method is competitive with current state-of-the-art methods on FEVER, HoVer and FEVEROUS-S, while using $5$ to $10$ times less memory than competing systems. Evaluation on an adversarial dataset indicates improved stability of our approach compared to commonly deployed threshold-based methods. Finally, the proof system helps humans predict model decisions correctly more often than using the evidence alone.

# Installation

Setup a new conda environment, e.g. python3.9 (tested only Python version 3.9):

```bash
conda create -n admiral python=3.9
conda activate admiral
```

Then install AdMIRaL and all relevant dependencies:

```
python3 -m pip install -e .
python3 -m pip install git+https://github.com/castorini/pygaggle.git
```


The pipeline to incorporate retrieved data uses Pyserini. The dependencies are already installed, however you also need to download Java and place it into the root path of the repository: https://jdk.java.net/19/

Set Java Path: `export JAVA_HOME=$PWD/jdk-19.0.2/`

## Downloading Data + Models

To download relevant FEVER data run the script:

```bash
./bin/download_fever.sh
/bin/download_feverous.sh
```


Then download relevant retrieval models to a folder `./models`. For convinience you can use the gdown command, installed as part of this repository:

```
gdown --folder https://drive.google.com/drive/folders/1sObMn8YZ8GKxWUiRWMJqoMBnHRzASbKl?usp=sharing (Autoregressive retrieval)
gdown --folder https://drive.google.com/drive/folders/1mUZ58Vh8k50a3iDYCgR8w4SSKw8fn2sk?usp=sharing (Reranker Iteration 0)
gdown --folder https://drive.google.com/drive/folders/1XxTTsYKJ89gLs-EUoSDdZH2nqd2BgGuy?usp=sharing (Reranker Iteration 1)
gdown --folder https://drive.google.com/drive/folders/1_nPGY2zMaRW_aYsJxygr_gEf-KhNzGOj?usp=sharing (Sufficiency model)
```

## Build Wikipedia Trie (FEVEROUS)

For FEVER, it is sufficient to use the Trie available from the GENRE repository: http://dl.fbaipublicfiles.com/GENRE/kilt_titles_trie_dict.pkl

For other datasets (such as FEVEROUS), you might want to build your own trie. Run this command in the case of FEVEROUS:

```
python3 -m src.retrieval.genre.generate_trie --output_file datasets/feverous/genre_feverous_trie.pkl --mode start --database indexes/lucene-index-feverous-passage/ --dataset feverous --granularity passage
```

# Running AdMiRaL

Run the following command with the default settings:

```
./bin/run_admiral_stammbach.sh k1_09_b_01 fever top3_beam_20 10_docs_inpars proofver_2hop 42
```

If you want to speed up the retrieval process and maximize recall, at the cost of substentially worse precision and without sufficiency explainability, you can disable the sufficiency module:

```
./bin/run_admiral_stammbach.sh k1_09_b_01 fever top3_beam_20 10_docs none_2hop 42
```

# Notes

This repository is a reimplementation of the original codebase and deviates slightly from the paper, largely for simplification:
- Instead of using the [Stammbach retriever](https://github.com/dominiksinsaarland/document-level-FEVER) for FEVER, we use a T5-reranker. The retrieval scores on FEVER are thus slightly lower. However, using a T5-reranker makes the codebase more flexibility, e.g. to incorporate longer documents (as needed for datasets like FEVEROUS). Please reach out to me if you want the retrieval results for FEVER shown in the paper.
- We only keep track of the top i=1 D^i_t document set at a given iteration. Since most evaluation metrics consider recall@k with k >> 1, we fill up with documents d_t selected in the current iteration. 

### TODO
- Detailed instructions for retrieval over text and tables for FEVEROUS
- Instructions for training models
