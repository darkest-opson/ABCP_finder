# 🧬 ABCP_finder

**A Command-Line Tool for Predicting Anti-Breast Cancer Peptides (ABCPs)**

ABCP_finder is a command-line utility that allows users to predict whether a given peptide sequence is anti-cancer using either **ESM-2** or **ProtBERT** embeddings and corresponding trained models.  
It supports both **single-sequence** and **batch predictions** using one or multiple trained `.pkl` models.

---

## 🚀 Features

- 🧠 Choose between **ESM-2** or **ProtBERT** embeddings  
- 🧩 Support for **single model** or **multiple models (directory input)**  
- 📄 Accepts **direct sequence input** or **FASTA file input**  
- 🔍 Automatically skips **invalid peptide lengths** (not between 5–51 residues)  
- ⏱️ Displays **real-time progress** and model information during predictions  
- 💾 Saves results in a structured **.tsv output file**

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/darkest-opson/ABCP_finder.git
cd abcp_finder
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. (Optional) For GPU acceleration
Make sure you have a **PyTorch** version installed with **CUDA** support.

---

## Command-Line Arguments
|  **Argument** | **Description**                                                       | **Example**               |
| ------------: | --------------------------------------------------------------------- | ------------------------- |
| `--embedding` | Embedding model to use (`esm2` or `protbert`)                         | `--embedding esm2`        |
|     `--model` | Path to a single `.pkl` model or directory containing multiple models | `--model ./ESM_models/`   |
|  `--sequence` | Input peptide sequence for prediction                                 | `--sequence GKLFGKILVGKL` |
|     `--fasta` | Input FASTA file containing multiple peptide sequences                | `--fasta peptides.fasta`  |
|    `--output` | Path to save results (default: `output_predictions.tsv`)              | `--output results.tsv`    |

### Usage Examples
1️⃣ Single Model Prediction
```
python ABCP_finder.py --embedding esm2 --model ./ESM_models/model.pkl --sequence GKLFGKILVGKL
```
2️⃣ Multiple Model Prediction (directory of .pkl models)
```
python ABCP_finder.py --embedding protbert --model ./ProtBERT_models/ --sequence GKLFGKILVGKL
```
3️⃣ FASTA Input
```
python ABCP_finder.py --embedding esm2 --model ./ESM_models/ --fasta peptides.fasta
```
### Output Format
Results are saved in a tab-separated (.tsv) file with the following structure:
| **Peptide**  | **Model**  | **Predicted_Class** | **Prob_Class0** | **Prob_Class1** |
| ------------ | ---------- | ------------------- | --------------- | --------------- |
| GKLFGKILVGKL | model1.pkl | 1                   | 0.1234          | 0.8766          |
| GLYFGKILVGL  | model2.pkl | 0                   | 0.8234          | 0.1766          |

#### Predicted_Class

0 → Non-ABCP

1 → ABCP

#### Prob_Class0 / Prob_Class1

Confidence scores for each predicted class.

### Model Compatibility

Embedding	Supported Model Directory	Example Model File

ESM-2	./ESM_models/	esm_classifier.pkl

ProtBERT	./ProtBERT_models/	protbert_classifier.pkl

### Notes
Peptide sequences must be 5–51 amino acids in length.

Models must be trained with compatible embeddings (e.g., ESM-2 models require ESM-2 embeddings).

When a directory is passed, all .pkl models within it are evaluated.

Output results are aggregated in one file containing per-sequence, per-model predictions.

