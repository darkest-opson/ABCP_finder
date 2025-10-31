#!/usr/bin/env python3
"""
ABCP Finder - Command Line Tool
--------------------------------
Predict anticancer peptide activity using either ProtBERT or ESM-2 embeddings
and one or more pre-trained models (.pkl files).

Usage Examples:
---------------
1. Single model prediction:
    python abcp_finder.py --embedding esm2 --model ./ESM_models/model.pkl --sequence GKLFGKILVGKL

2. Multiple model prediction (directory of .pkl models):
    python abcp_finder.py --embedding protbert --model ./ProtBERT_models/ --sequence GKLFGKILVGKL

3. FASTA input:
    python abcp_finder.py --embedding esm2 --model ./ESM_models/ --fasta peptides.fasta
"""

import os
import re
import pickle
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, EsmTokenizer, EsmModel
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin
import argparse
import warnings

warnings.filterwarnings("ignore", "Some weights of EsmModel were not initialized from the model checkpoint")
warnings.filterwarnings("ignore", category=UserWarning)


# -------------------- MODEL DEFINITIONS --------------------
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes if num_classes > 2 else 1)
        )

    def forward(self, x):
        return self.layers(x)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, num_classes=2, hidden_dim=256, dropout=0.3, device=None):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.device = device

    def predict(self, X):
        self.model_.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model_(X_tensor)
            if self.num_classes == 2:
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                return (probs >= 0.5).astype(int)
            else:
                return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_proba(self, X):
        self.model_.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model_(X_tensor)
            if self.num_classes == 2:
                probs = torch.sigmoid(logits).cpu().numpy()
                return np.hstack([1 - probs, probs])
            else:
                return torch.softmax(logits, dim=1).cpu().numpy()


class ManualHardVotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        self.estimators = estimators

    def predict(self, X):
        predictions = np.asarray([est.predict(X) for est in self.estimators_]).T
        majority_vote = mode(predictions, axis=1, keepdims=False)[0]
        return self.le_.inverse_transform(majority_vote)

    def predict_proba(self, X):
        all_probas = [est.predict_proba(X) for est in self.estimators_ if hasattr(est, 'predict_proba')]
        if not all_probas:
            raise AttributeError("None of the base estimators support predict_proba.")
        return np.mean(all_probas, axis=0)


# -------------------- EMBEDDINGS --------------------
def get_protbert_embedding(sequence, model, tokenizer, device):
    sequence_spaced = " ".join(list(sequence))
    sequence_cleaned = re.sub(r"[UZOB]", "X", sequence_spaced)
    encoded = tokenizer(sequence_cleaned, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        output = model(**encoded)
    return output.last_hidden_state.mean(dim=1).cpu().numpy()


def get_esm_embedding(sequence, model, tokenizer, device):
    encoded = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1022).to(device)
    with torch.no_grad():
        output = model(**encoded)
    embedding = output.last_hidden_state[0, 1:-1].mean(dim=0).squeeze().cpu().numpy()
    return embedding.reshape(1, -1)


# -------------------- HELPERS --------------------
def load_model(path):
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Loaded model: {path}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None


def read_fasta(fasta_path):
    sequences = []
    with open(fasta_path, "r") as f:
        seq = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                    seq = ""
            else:
                seq += line
        if seq:
            sequences.append(seq)
    return sequences


def predict(sequence, model, embed_func, embed_model, tokenizer, device):
    if len(sequence) < 5 or len(sequence) > 51:
        print(f"⚠️ Skipping {sequence} (length not between 5–51)")
        return None

    emb = embed_func(sequence, embed_model, tokenizer, device)
    pred = model.predict(emb)[0]
    probs = model.predict_proba(emb)[0] if hasattr(model, "predict_proba") else None
    return pred, probs


# -------------------- MAIN --------------------
def main():
    parser = argparse.ArgumentParser(description="ABCP Finder - Anticancer Peptide Predictor")
    parser.add_argument("--embedding", required=True, choices=["protbert", "esm2"], help="Embedding model to use.")
    parser.add_argument("--model", required=True, help="Path to model file (.pkl) or directory of models.")
    parser.add_argument("--sequence", help="Single peptide sequence to predict.")
    parser.add_argument("--fasta", help="FASTA file containing multiple peptide sequences.")
    parser.add_argument("--output", default="ABCP_results.txt", help="Output results file.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🧠 Using device: {device}")

    # Load embedding model
    if args.embedding == "protbert":
        print("\n🔹 Loading ProtBERT...")
        model_id = "Rostlab/prot_bert_bfd"
        tokenizer = BertTokenizer.from_pretrained(model_id, do_lower_case=False)
        embed_model = BertModel.from_pretrained(model_id).to(device)
        embed_func = get_protbert_embedding
    else:
        print("\n🔹 Loading ESM-2...")
        model_id = "facebook/esm2_t33_650M_UR50D"
        tokenizer = EsmTokenizer.from_pretrained(model_id)
        embed_model = EsmModel.from_pretrained(model_id).to(device)
        embed_func = get_esm_embedding

    embed_model.eval()

    # Determine model files
    model_paths = []
    if os.path.isdir(args.model):
        model_paths = [os.path.join(args.model, f) for f in os.listdir(args.model) if f.endswith(".pkl")]
    else:
        model_paths = [args.model]

    # Determine sequences
    sequences = []
    if args.sequence:
        sequences = [args.sequence]
    elif args.fasta:
        sequences = read_fasta(args.fasta)
    else:
        print("❌ Error: You must provide either --sequence or --fasta input.")
        return

    # Output file setup
    with open(args.output, "w") as out:
        out.write("Peptide\tModel\tPredicted_Class\tProb_Class0\tProb_Class1\n")

        for seq in sequences:
            seq = seq.strip().upper()
            print("Prediction is started for : ",seq)
            for model_path in model_paths:
                model = load_model(model_path)
                if not model:
                    continue

                result = predict(seq, model, embed_func, embed_model, tokenizer, device)
                if result is None:
                    continue

                pred, probs = result
                if probs is not None:
                    out.write(f"{seq}\t{os.path.basename(model_path)}\t{pred}\t{probs[0]:.4f}\t{probs[1]:.4f}\n")
                else:
                    out.write(f"{seq}\t{os.path.basename(model_path)}\t{pred}\tNA\tNA\n")

    print(f"\n✅ All predictions saved in: {args.output}")


if __name__ == "__main__":
    main()
