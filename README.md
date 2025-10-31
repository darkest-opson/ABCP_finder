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
git clone https://github.com/yourusername/abcp-finder.git
cd abcp_finder
```
