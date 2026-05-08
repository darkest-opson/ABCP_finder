# ABCP_finder

## Anti-Breast Cancer Peptide Prediction Using Transformer-Based Protein Language Models


- ProtBERT
- ESM2

The generated embeddings are classified using trained Multi-Layer Perceptron (MLP) models to identify potential anti-breast cancer peptides.

---

# Features

- Supports **ProtBERT**, **ESM2**, or both models simultaneously
- Accepts peptide sequences in **FASTA format**
- Automatic CPU/GPU device selection
- Batch prediction support
- Generates probability scores for:
  - ABCP class
  - Non-ABCP class
- Saves prediction results in CSV format
- Compatible with large-scale peptide screening

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/darkest-opson/ABCP_finder.git
cd ABCP_finder
```

---

## 2. Create Conda Environment

```bash
conda create -n abcp_finder python=3.10
conda activate abcp_finder
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install PyTorch

### CPU Version

```bash
pip install torch torchvision torchaudio
```

### GPU Version (CUDA)

Visit:

https://pytorch.org/get-started/locally/

---

# Directory Structure

```bash
ABCP_finder/
│
├── ABCP_finder.py
├── requirements.txt
├── models/
│   ├── protbert/
│   │   └── ABCP_mlp_model.pt
│   │
│   └── esm/
│       └── ABCP_mlp_model.pt
│
├── example/
│   └── sample.fasta
│
└── output/
```

---

# Input Format

Input sequences must be provided in FASTA format.

Example:

```fasta
>peptide_1
GIMSLFKGVLKTAGKHVAG
>peptide_2
KLLKLLKKLLKLLKKK
```

---

# Usage

## Basic Command

```bash
python ABCP_finder.py -i input.fasta
```

---

# Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|----------|
| `-i, --input` | Input FASTA file | Required |
| `-m, --model` | Prediction model (`protbert`, `esm`, `both`) | `both` |
| `-o, --output` | Output CSV file | `ABCP_predictions.csv` |
| `-d, --device` | Compute device (`auto`, `cpu`, `gpu`) | `auto` |

---

# Example Commands

## Predict Using Both Models

```bash
python ABCP_finder.py \
-i peptides.fasta \
-m both \
-o predictions.csv
```

---

## Predict Using ProtBERT Only

```bash
python ABCP_finder.py \
-i peptides.fasta \
-m protbert
```

---

## Predict Using ESM2 Only

```bash
python ABCP_finder.py \
-i peptides.fasta \
-m esm
```

---

## Force GPU Usage

```bash
python ABCP_finder.py \
-i peptides.fasta \
-d gpu
```

---

# Output Format

The prediction output is saved as a CSV file.

Example:

| Sequence_ID | Sequence | ProtBERT_ABCP_probability | ProtBERT_prediction |
|-------------|----------|---------------------------|---------------------|
| peptide_1 | GIMSLFKGVLKTAGKHVAG | 0.9213 | 1 |
| peptide_2 | KLLKLLKKLLKLLKKK | 0.1342 | 0 |

---

# Prediction Labels

| Label | Meaning |
|------|----------|
| `1` | Anti-Breast Cancer Peptide |
| `0` | Non-ABCP |

---

# Models Used

## ProtBERT

- Model: `Rostlab/prot_bert`
- Source: HuggingFace Transformers

## ESM2

- Model: `facebook/esm2_t33_650M_UR50D`
- Source: Meta AI

---

# Methodology

1. Read peptide sequences from FASTA
2. Generate transformer embeddings
3. Load trained MLP classifier
4. Predict probabilities
5. Apply optimized threshold
6. Export predictions

---

# Requirements

Main dependencies include:

```txt
torch
transformers
numpy
pandas
tqdm
```

---

# Citation

If you use ABCP_finder in your research, please cite:

```text
ABCP_finder: Transformer-Based Prediction of Anti-Breast Cancer Peptides
```

---

# License

This project is released under the MIT License.

---

# Contact

For issues, suggestions, or collaborations:

- GitHub Issues
- Email: your_email@example.com

---

# Acknowledgements

- HuggingFace Transformers
- Meta AI
- Rostlab ProtTrans Team
- PyTorch
