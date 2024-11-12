# from genre import GENRE
from src.retrieval.genre.hf_model import GENRE

from src.retrieval.genre.trie import Trie
import dataset_readers
import re

#Inference procedure for GENRE'sEntity linking
from src.retrieval.genre.entity_linking_proofver_fixed import get_end_to_end_prefix_allowed_tokens_fn_hf as get_prefix_allowed_tokens_fn


# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
from spacy.lang.en import English

NATLOG_TRANSITION_MATRIX = {
    "SUPPORTS": {
        "=": "SUPPORTS",
        "<": "SUPPORTS",
        "!": "REFUTES",
        "|": "REFUTES",
        ">": "NOT ENOUGH INFO",
        "#": "NOT ENOUGH INFO",
    },
    "REFUTES": {
        ">": "REFUTES",
        "=": "REFUTES",
        "|": "NOT ENOUGH INFO",
        "<": "NOT ENOUGH INFO",
        "!": "SUPPORTS",
        "#": "NOT ENOUGH INFO",
    },
    "NOT ENOUGH INFO": {
        "=": "NOT ENOUGH INFO",
        "<": "NOT ENOUGH INFO",
        "!": "NOT ENOUGH INFO",
        "|": "NOT ENOUGH INFO",
        ">": "NOT ENOUGH INFO",
        "#": "NOT ENOUGH INFO",
    },
}

class SufficiencyProoFVer():

    def __init__(self, config):
        self.config = config
        nlp = English()
        self.tokenizer = nlp.tokenizer
        self.model = GENRE.from_pretrained(config.sufficiency_proofver_model).eval().to(config.device)
        self.reader = dataset_readers.get_class(config.dataset)(**{'granularity': "sentence"})

    def evidence_sufficiency_check(self, claim_id, claim, evidence_sentences):
        sufficiency_proofs = self.generate_sufficiency_proof(claim, evidence_sentences)
        # Only consider proofs above certain confidence cutoff
        print(sufficiency_proofs)
        processed_proofs = [SufficiencyProoFVer.read_natural_operations(proof[0]) for proof in sufficiency_proofs if proof[1] > self.config.sufficiency_confidence_cutoff]
        # print("PROCESSED", processed_proofs)

        # Check whether a single proof exists that has sufficient evidence.
        sufficiency = [SufficiencyProoFVer.check_status(x) for x in processed_proofs]
        # print(sufficiency)

        # In case no valid proof vas generated fallback
        if sufficiency_proofs:
            sufficiency_proofs_dict = {"qid": claim_id, "proofs": [x[0] for x in sufficiency_proofs], "scores": [x[1] for x in sufficiency_proofs]}
        else:
            sufficiency_proofs_dict = [{"qid": claim_id, "proofs": "N/A", "scores": -999999999999}]
  
        return 1 if len(sufficiency) > 0 and all(sufficiency)==1 else 0, sufficiency_proofs_dict

    def save_proofs_to_file(self, generated_proofs, current_iteration):
        with open(self.config.generated_proofs_dev.replace("{ITER}", str(current_iteration)), "w") as f_out:
            for sample in generated_proofs:
                for i,proof in enumerate(sample["proofs"]):
                    f_out.write("{}\t{}\t{}\t{}\n".format(sample["qid"], i+1, sample["scores"][i], proof))

    @staticmethod
    def read_natural_operations(proof):
        matches = re.findall(r" \] (.+?) \{", proof.strip() + " {")
        natlog_operations = []
        for match in matches:
            if match.strip() == "":
                continue
            natlog_operations.append(match.strip().split()[0][0])

        return natlog_operations

    @staticmethod
    def check_status(proof_ops):
        proof_ops = [x for x in proof_ops if x in ["<", ">", "|", "=", "#", "!"]]
        final_state = SufficiencyProoFVer.natlog_automaton(proof_ops)
        if final_state in ["SUPPORTS", "REFUTES"]: #, "REFUTES"]:
            return 1
        else: 
            return 0
    
    @staticmethod
    def natlog_automaton(sequence):
        """
        (
            "eq",
            "gorward entailment",
            "negation",
            "alternation",
            "reverse entailment",
            "independent"
        ),
        (
            " =",
            " <",
            " !",
            " |",
            " >",
            " #"
        ),
        """

        current_status = "SUPPORTS"

        for relation in sequence:
            current_status = NATLOG_TRANSITION_MATRIX[current_status][relation]

        return current_status

    def generate_sufficiency_proof(self, claim, evidence_sentences):
        sentences = [self.reader.process_sentence(x["sentence"]) for x in evidence_sentences][:self.config.sufficiency_sentences_to_consider]

        filtered_evidence = set()
        claim_text = claim
        claim = claim.split()

        #Kept some additional symbols at the beginning of the claim - Every encoder side input starts with "^ {"
        # del claim[0:2]
        k = 7
        answers = set([" " + " ".join(claim[start: start + i]) for start in range(len(claim)) for i in range(k)
                if len(claim[start: start + k]) <= k])

        answers = [item.strip() if item[0:2] == " ^"  else item for item in answers  ]
        answers.remove(" ")

        #remove spans which only have stop words
        toRemove = set()
        for item in answers:
            toks = [tok.text for tok in self.tokenizer(item.strip()) if tok.is_punct == False and tok.is_stop == False]
            if len(toks) < 1:
                toRemove.add(item)
        answers = set(answers) - toRemove

        span_to_evidence = {}

        for i, ev in enumerate(sentences):
            # title = reader.process_title(predicted_sentences_title[id][i])
            title = self.reader.process_title('_'.join(evidence_sentences[i]["sent_id"].split('_')[:-1]))
            sentences[i] = title + ' ; ' + ev

        #create evidence spans
        evidence = [sent.strip().split() for sent in sentences]
        evidSet = set()
        for i, sent in enumerate(evidence):
            _temp = set([ " ".join(sent[start: start + i]) for start in range(len(sent)) for i in range(k)
            if len(sent[start: start + k]) <= k])
            evidSet = evidSet.union(_temp)

            for x in list(_temp):
                if len(x) < 1:
                    continue
                if x in span_to_evidence:
                    span_to_evidence[x].add(evidence_sentences[i]["sent_id"])
                else:
                    span_to_evidence[x] = set([evidence_sentences[i]["sent_id"]])

            toRemove = set()
            for item in evidSet:
                toks = [tok.text for tok in self.tokenizer(item.strip()) if tok.is_punct == False and tok.is_stop == False]
                if len(toks) < 1:
                    toRemove.add(item)

            evidSet = evidSet - toRemove

        model_input = ["^ { " + claim_text + ' </s> ' + ' </s> '.join(sentences)]

        
        # prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn_original(
        prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
            self.model,
            # input, #[claim_text]
            ["^ { " + claim_text],
            mention_trie=Trie([
                self.model.encode(e)[1:].tolist()
                for e in answers
            ]),
            candidates_trie=Trie([
                self.model.encode(" }} [ {} ]".format(e))[1:].tolist()
                for e in evidSet
            ])
        )

        # print(model_input)
        output = self.model.sample(
            model_input,
            num_beams=self.config.sufficiency_model_beam_size,
            num_return_sequences = self.config.sufficiency_model_num_proofs,
            # prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, # TODO: Bug caused by package versions. FIX. Commented out for now.
            max_new_tokens=256,
        )

        proofs = [[(e['text'].strip(), e["score"].item()) for e in el] for el in output]
        proofs = [item for sublist in proofs for item in sublist]

        return proofs
