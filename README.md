# 🧬 ABCP-Finder

**A Command-Line Tool for Predicting Anti-Breast Cancer Peptides (ABCPs)**

ABCP-Finder is a command-line utility that allows users to predict whether a given peptide sequence is anti-cancer using either **ESM-2** or **ProtBERT** embeddings and corresponding trained models.  
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

## 🧩 Command-Line Arguments
```
| Argument | Description | Example |
|-----------|--------------|----------|
| `--embedding` | Embedding model to use (`esm2` or `protbert`) | `--embedding esm2` |
| `--model` | Path to a single `.pkl` model or directory containing multiple models | `--model ./ESM_models/` |
| `--sequence` | Input peptide sequence for prediction | `--sequence GKLFGKILVGKL` |
| `--fasta` | Input FASTA file containing multiple peptide sequences | `--fasta peptides.fasta` |
| `--output` | Path to save results (default: `output_predictions.tsv`) | `--output results.tsv` |

```
