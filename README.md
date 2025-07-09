# MMFE

This repository provides tools to run the **MMFE** (Multi‐modal Molecular Feature Extraction for Drug
Selectivity Prediction) analysis at two different concentrations:

- **MMFE_3uM.ipynb**: run MMFE at **3 µM** concentration  
- **MMFE_300nM.ipynb**: run MMFE at **300 nM** concentration  

Each notebook is self-contained and shows how to load input data, configure parameters, and visualize results for the specified concentration.

---

## Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/jjjabcd/MMFE.git
cd MMFE
```

### 2. Environment Setting
```bash
conda env create -f environment.yml
conda activate mmfe
```

### 3. Jupyter notebook
```bash
jupyter notebook --ip 0.0.0.0
```

