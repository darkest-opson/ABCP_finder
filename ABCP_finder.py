#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
from tqdm import tqdm
import re

############################################################
# MODEL CLASS
############################################################

class TorchMLPClassifier:

    def __init__(self):
        self.model = None
        self.device = None

    def predict_proba(self, X):

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model.eval()

        with torch.no_grad():

            outputs = self.model(X_tensor)

            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()


############################################################
# FASTA READER
############################################################

def read_fasta(file):

    ids = []
    seqs = []
    seq = ""

    with open(file) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):

                if seq:
                    seqs.append(seq)
                    seq = ""

                ids.append(line[1:])

            else:
                seq += line

        if seq:
            seqs.append(seq)

    if len(ids) != len(seqs):

        raise ValueError(
            f"FASTA parsing error: {len(ids)} IDs but {len(seqs)} sequences"
        )

    return ids, seqs


############################################################
# DEVICE SELECTION
############################################################

def get_device(user_device):

    if user_device == "cpu":
        return "cpu"

    if user_device == "gpu":

        if torch.cuda.is_available():
            return "cuda"
        else:
            print("GPU not available. Falling back to CPU.")
            return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


############################################################
# PROTBERT EMBEDDINGS
############################################################

def protbert_embeddings(seqs, device, batch_size=5):

    print("Loading ProtBERT model...")

    tokenizer = BertTokenizer.from_pretrained(
        "Rostlab/prot_bert",
        do_lower_case=False
    )

    model = BertModel.from_pretrained(
        "Rostlab/prot_bert"
    ).to(device)

    model.eval()

    embeddings = []

    for i in tqdm(range(0, len(seqs), batch_size)):

        batch = seqs[i:i+batch_size]

        batch = [" ".join(list(s)) for s in batch]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():

            outputs = model(**encoded)

            emb = outputs.last_hidden_state.mean(dim=1)

        embeddings.extend(emb.cpu().numpy())

    return np.array(embeddings)


############################################################
# ESM2 EMBEDDINGS
############################################################

def esm_embeddings(seqs, device, batch_size=5):

    print("Loading ESM2 model...")

    model_name = "facebook/esm2_t33_650M_UR50D"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(
        model_name,
        add_pooling_layer=False
    ).to(device)

    model.eval()

    embeddings = []

    seqs_processed = [
        " ".join(list(re.sub(r"[UZOB]", "X", s))) for s in seqs
    ]

    for i in tqdm(range(0, len(seqs_processed), batch_size)):

        batch = seqs_processed[i:i+batch_size]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)

        with torch.no_grad():

            outputs = model(**encoded)

        last_hidden_state = outputs.last_hidden_state

        batch_embeddings = last_hidden_state[:,1:-1].mean(dim=1).cpu().numpy()

        embeddings.extend(batch_embeddings)

    return np.array(embeddings)


############################################################
# LOAD SAVED MODEL
############################################################

def load_model(model_path, device):

    checkpoint = torch.load(model_path, map_location=device)

    model = TorchMLPClassifier()

    model.device = device

    input_dim = checkpoint["input_dim"]
    hidden_dim = checkpoint["hidden_dim"]
    num_layers = checkpoint["num_layers"]

    layers = []

    in_dim = input_dim

    for _ in range(num_layers):

        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))

        in_dim = hidden_dim

    layers.append(nn.Linear(in_dim, 2))

    model.model = nn.Sequential(*layers).to(device)

    model.model.load_state_dict(checkpoint["state_dict"])

    threshold = checkpoint["threshold"]

    return model, threshold


############################################################
# MAIN
############################################################

def main():

    parser = argparse.ArgumentParser(
        description="ABCP Finder: Anti-Breast Cancer Peptide Predictor"
    )

    parser.add_argument("-i","--input",required=True,help="Input FASTA file")

    parser.add_argument(
        "-m","--model",
        choices=["protbert","esm","both"],
        default="both"
    )

    parser.add_argument(
        "-o","--output",
        default="ABCP_predictions.csv"
    )

    parser.add_argument(
        "-d","--device",
        default="auto",
        choices=["auto","cpu","gpu"],
        help="Compute device"
    )

    args = parser.parse_args()

    device = get_device(args.device)

    print("Using device:",device)

    ids, seqs = read_fasta(args.input)

    print("Total sequences:",len(seqs))

    results = pd.DataFrame({
        "Sequence_ID":ids,
        "Sequence":seqs
    })

    ########################################################
    # PROTBERT MODEL
    ########################################################

    if args.model in ["protbert","both"]:

        print("\nGenerating ProtBERT embeddings...")

        X = protbert_embeddings(seqs,device)

        model,threshold = load_model(
            "./models/protbert/ABCP_mlp_model.pt",
            device
        )

        probs = model.predict_proba(X)

        non_abcp = probs[:,0]
        abcp = probs[:,1]

        preds = (abcp >= threshold).astype(int)

        results["ProtBERT_nonABCP_probability"] = non_abcp
        results["ProtBERT_ABCP_probability"] = abcp
        results["ProtBERT_prediction"] = preds

    ########################################################
    # ESM MODEL
    ########################################################

    if args.model in ["esm","both"]:

        print("\nGenerating ESM2 embeddings...")

        X = esm_embeddings(seqs,device)

        model,threshold = load_model(
            "./models/esm/ABCP_mlp_model.pt",
            device
        )

        probs = model.predict_proba(X)

        non_abcp = probs[:,0]
        abcp = probs[:,1]

        preds = (abcp >= threshold).astype(int)

        results["ESM_nonABCP_probability"] = non_abcp
        results["ESM_ABCP_probability"] = abcp
        results["ESM_prediction"] = preds

    ########################################################

    results.to_csv(args.output,index=False)

    print("\nPrediction completed.")
    print("Results saved to:",args.output)


############################################################

if __name__ == "__main__":
    main()