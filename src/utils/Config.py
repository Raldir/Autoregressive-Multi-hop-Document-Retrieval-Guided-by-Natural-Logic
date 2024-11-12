import ast
import json
import os


class Config(object):
    def __init__(self, filenames=None, kwargs=None):
        # Experiment configs
        self.exp_dir = None
        self.exp_name = None
        self.seed = 42
        self.is_debug = False

        # Compute backend configs
        self.compute_precision = "bf16"
        self.compute_strategy = "none"
        self.device = "cuda"
        
        self.dataset = "fever"
        self.genre_only = False # Whether to not use the reranker and instead only use genre output (only works for single iteration, debugging purposes)
        self.use_precomputed_genre = False # When genre is already computed for iteration (debugging purposes)

        # BM25 settings
        self.bm25_num_retrieved_docs = 100
        self.bm25_document_k1 = 0.6
        self.bm25_document_b = 0.5
        self.max_docs_per_file = 1000000

        # GENRE settings
        self.genre_init_num_retrieved_docs = 2
        self.genre_init_num_beams = 5
        self.genre_hop_num_beams = 20
        self.genre_hop_num_sentences_context = 5
        self.genre_hop_num_retrieved_docs = 3
        self.genre_documents_merge_mode = "default" # default, linked
        self.genre_trie_path = os.path.join("data", "{DATASET}", "trie", "kilt_titles_trie_dict.pkl")
        self.genre_trie_completion_path = os.path.join("data", "{DATASET}", "trie", "completion_trie.pkl")
        self.genre_title_path = os.path.join("data", "{DATASET}", "document_titles.json")

        # Sentence reranker settings
        self.reranker_num_docs_to_consider = 5
        self.reranker_prompt_format = "concat"
        self.multi_hop_weighting = 1
        self.reranker_max_input_length = 120
        self.sort_after_reranking = False

        # Iterative retrieval sttings
        self.continue_from_iteration = 0
        self.max_iterations = 1

        # Sufficiency proof
        self.enable_sufficiency_check = False
        self.sufficiency_proofver_model = os.path.join("models", "sufficiency_proofver_new_26102024_proofver_based_REALLY_GOOD/checkpoint-2000")
        self.sufficiency_sentences_to_consider = 5
        self.sufficiency_model_beam_size = 25 #5
        self.sufficiency_model_num_proofs = 10
        self.sufficiency_confidence_cutoff = -0.9

        # Index paths
        self.pyserini_index_path_passage = os.path.join("indexes", "lucene-index-{DATASET}-passage")
        self.pyserini_index_path_pipeline = os.path.join("indexes", "lucene-index-{DATASET}-pipeline")
        self.annotation_path = os.path.join("data", "{DATASET}", "shared_task_dev.jsonl")
        self.raw_document_path = os.path.join("data", "{DATASET}", "wiki-pages")
        self.pyserini_document_path = self.raw_document_path.replace("wiki-pages", "wiki-pages-pyserini")
        self.pyserini_pipeline_path = self.raw_document_path.replace("wiki-pages", "wiki-pages-pipeline-pyserini")

        # Model paths
        self.genre_init_model_path = os.path.join("models", "hf_wikipage_retrieval")
        self.genre_model_path = os.path.join("models", "fever", "iter_{ITER}", "bart_multihop")
        self.reranker_model_init = os.path.join("models", "fever", "iter_0", "monot5_reranker") #"castorini/monot5-large-msmarco" # see https://huggingface.co/castorini for other models. Also see https://arxiv.org/pdf/2301.01820 for the 3B model.
        self.reranker_model = os.path.join("models", "fever", "iter_{ITER}", "monot5_reranker") # see https://huggingface.co/castorini for other models



        if filenames:
            for filename in filenames.split("+"):
                if not os.path.exists(filename):
                    filename = os.path.join(os.getenv("CONFIG_PATH", default="configs"), filename)

                self.update_kwargs(json.load(open(filename)), eval=False)
        if kwargs:
            self.update_kwargs(kwargs)

        self.set_exp_dir()

        assert self.sufficiency_model_beam_size >= self.sufficiency_model_num_proofs, "Number of returned proofs cannot be larger than beam size,  but found {} and {}".format(self.sufficiency_model_beam_size, self.sufficiency_model_num_proofs)

        assert not (self.genre_only and self.use_precomputed_genre), "Cannot combine flags: for using precomputed genre and then only use genre"


    def update_kwargs(self, kwargs, eval=True):
        for k, v in kwargs.items():
            if eval:
                try:
                    if "+" in v:  # Spaces are replaced via symbol
                        v = v.replace("+", " ")
                    else:
                        v = ast.literal_eval(v)
                except ValueError:
                    v = v
            else:
                v = v
            if not hasattr(self, k):
                raise ValueError(f"{k} is not in the config")
            setattr(self, k, v)

    def set_exp_dir(self):
        """
        Updates the config default values based on parameters passed in from config file
        """

        if self.exp_name is not None:
            self.exp_dir = os.path.join(os.getenv("OUTPUT_PATH", default="exp_out"), self.exp_name)
        else:
            self.exp_dir = os.getenv("OUTPUT_PATH", default="exp_out")
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        for i in range(self.max_iterations):
            if not os.path.exists(os.path.join(self.exp_dir, "iter_{}".format(i))):
                os.makedirs(os.path.join(self.exp_dir, "iter_{}".format(i)))

        if self.exp_dir is not None:
            self.save_config(os.path.join(self.exp_dir, os.path.join("config.json")))
            self.finish_flag_file = os.path.join(self.exp_dir, "exp_completed.txt")


            # Update model and index paths
            self.genre_trie_path = self.genre_trie_path.replace("{DATASET}", self.dataset)
            self.genre_trie_completion_path = self.genre_trie_completion_path.replace("{DATASET}", self.dataset)
            self.genre_title_path = self.genre_title_path.replace("{DATASET}", self.dataset)
            self.pyserini_index_path_passage = self.pyserini_index_path_passage.replace("{DATASET}", self.dataset)
            self.pyserini_index_path_pipeline = self.pyserini_index_path_pipeline.replace("{DATASET}", self.dataset)
            self.annotation_path = self.annotation_path.replace("{DATASET}", self.dataset)
            self.raw_document_path = self.raw_document_path.replace("{DATASET}", self.dataset)
            self.pyserini_document_path = self.pyserini_document_path.replace("{DATASET}", self.dataset)
            self.pyserini_pipeline_path = self.pyserini_pipeline_path.replace("{DATASET}", self.dataset)


            # Score paths
            self.dev_score_file_bm25 = os.path.join(self.exp_dir, "iter_{ITER}", "dev_scores_bm25.json")
            self.dev_score_file_genre = os.path.join(self.exp_dir, "iter_{ITER}", "dev_scores_genre.json")
            self.dev_score_file_combined = os.path.join(self.exp_dir, "iter_{ITER}", "dev_scores_combined.json")
            self.dev_score_file_reranked = os.path.join(self.exp_dir, "iter_{ITER}", "dev_scores_reranked.json")
            self.dev_score_file_sentences = os.path.join(self.exp_dir, "iter_{ITER}", "dev_scores_sentences.json")
            self.dev_score_file_tables = os.path.join(self.exp_dir, "iter_{ITER}", "dev_scores_tables.json")
            self.dev_score_sufficiency = os.path.join(self.exp_dir, "iter_{ITER}", "dev_scores_sufficiency.json")

            # Retrieval paths
            self.bm25_pyserini_dev_docs = os.path.join(self.exp_dir, "iter_{ITER}", "bm25_pyserini_dev_docs.txt")
            self.genre_dev_docs = os.path.join(self.exp_dir, "iter_{ITER}", "genre_dev_docs.txt")
            self.combined_dev_docs = os.path.join(self.exp_dir, "iter_{ITER}", "combined_dev_docs.txt")
            self.reranked_dev_docs = os.path.join(self.exp_dir, "iter_{ITER}", "reranked_dev_docs.txt")
            self.reranked_dev_sentences = os.path.join(self.exp_dir, "iter_{ITER}", "reranked_dev_sentences.txt")
            self.reranked_dev_tables = os.path.join(self.exp_dir, "iter_{ITER}", "reranked_dev_tables.txt")
            self.generated_proofs_dev = os.path.join(self.exp_dir, "iter_{ITER}", "generated_proofs_dev.txt")

    def to_json(self):
        """
        Converts parameter values in config to json
        :return: json
        """
        return json.dumps(self.__dict__, indent=4, sort_keys=False)

    def save_config(self, filename):
        """
        Saves the config
        """
        with open(filename, "w+") as fout:
            fout.write(self.to_json())
            fout.write("\n")
