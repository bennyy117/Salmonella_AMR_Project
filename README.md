# Salmonella Antimicrobial Resistance Prediction

## Overview

This project develops a machine learning pipeline to predict antimicrobial resistance (AMR) phenotypes in *Salmonella enterica* isolates from genomic data. Using publicly available data from the NCBI Pathogen Detection database, the model predicts resistance to five clinically important antibiotics based on the presence of AMR genes.

The goal is to demonstrate the application of AI in bioinformatics and computational biology, bridging genotype-to-phenotype prediction while addressing real-world challenges such as imperfect genetic determinants and co-occurrence patterns on mobile genetic elements.

This project is particularly relevant to AI-driven drug discovery and biotechnology, leveraging PyTorch with GPU acceleration, aligning with modern accelerated computing workflows.

## Dataset

- **Source**: NCBI Pathogen Detection Isolates Browser (*Salmonella enterica*)
- **Size**: >210,000 isolates
- **Key Column**: `AMR genotypes` (annotated by AMRFinderPlus)
- **Unique AMR Genes Detected**: 429

The `AMR genotypes` field lists acquired resistance genes in the format `gene_name=COMPLETE` (e.g., `blaTEM-1=COMPLETE`, `tet(A)=COMPLETE`).

## Biological Background

Antimicrobial resistance in *Salmonella* is a major public health concern (WHO priority pathogen). Common mechanisms include:

- **Ampicillin**: β-lactamases (e.g., blaTEM-1, blaCMY-2, blaCARB-2)
- **Tetracycline**: Efflux pumps or ribosomal protection (e.g., tet(A), tet(B))
- **Streptomycin**: Aminoglycoside-modifying enzymes (e.g., aph(3'')-Ib, aph(6)-Id, aadA genes)
- **Sulfonamides**: Resistant dihydropteroate synthase (e.g., sul1, sul2)
- **Chloramphenicol**: Acetyltransferases or efflux (e.g., floR, catA1)

Even with known resistance genes, phenotype prediction is not always 100% accurate due to gene expression regulation, silent mutations, or accessory factors (e.g., efflux pumps like mdsA/mdsB).

## Methodology

### 1. Feature Engineering
- Parsed `AMR genotypes` into a binary matrix (429 features): 1 if gene present and `COMPLETE`.
- Defined **primary resistance determinants** for each antibiotic (based on CARD database and literature).
- **Intentionally excluded primary genes from features** to prevent data leakage and force the model to learn indirect patterns (e.g., co-occurrence on plasmids, shared mobile elements).

### 2. Label Creation
- Binary label per antibiotic: `resistant = 1` if at least one primary gene is present.

### 3. Sampling
- Stratified subsampling to 20,000 isolates for faster training while preserving class distribution.

### 4. Models
- **Baseline**: Random Forest (MultiOutputClassifier, 200 trees)
- **Deep Learning**: Feed-forward neural network in PyTorch
  - Architecture: Input → 512 → 256 → 5 (ReLU + Dropout 0.3 + Sigmoid)
  - Loss: Binary Cross-Entropy
  - Optimizer: Adam (lr=0.001)
  - Trained on GPU (CUDA)

### 5. Evaluation
- 5-fold stratified cross-validation
- Metric: Accuracy per antibiotic (multi-label setting)

## Results

### Random Forest (5-fold CV)

| Antibiotic         | Accuracy (mean ± std) |
|--------------------|-----------------------|
| Ampicillin         | 0.9139 ± 0.0033      |
| Tetracycline       | 0.8698 ± 0.0037      |
| Streptomycin       | 0.8916 ± 0.0023      |
| Sulfonamides       | 0.9104 ± 0.0018      |
| Chloramphenicol    | 0.9709 ± 0.0027      |

### PyTorch Neural Network (5-fold CV)

| Antibiotic         | Accuracy (mean ± std) |
|--------------------|-----------------------|
| Ampicillin         | 0.9131 ± 0.0044      |
| Tetracycline       | 0.8697 ± 0.0037      |
| Streptomycin       | 0.8917 ± 0.0024      |
| Sulfonamides       | 0.9093 ± 0.0034      |
| Chloramphenicol    | 0.9704 ± 0.0028      |

- Highest accuracy: Chloramphenicol (~97%) - strong association with floR.
- Lowest accuracy: Tetracycline (~87%) - more complex and diverse resistance mechanisms.

A final model was trained on the full subsample (30 epochs) and saved.

## Key Design Choices

- **No data leakage**: Primary genes removed from input features.
- **Realistic performance**: Accuracy < 100% reflects biological complexity (imperfect genotype-phenotype mapping).
- **GPU acceleration**: Full training using PyTorch + CUDA.
- **Robust evaluation**: 5-fold cross-validation with low variance.

## Future Work

- Integrate protein language models (e.g., ESM-2) to embed resistance protein sequences.
- Extend to generative modeling of mutant protein structures for resistance mechanism studies.
- Predict quantitative MIC values or expand to other bacterial species.
- Add interpretability (e.g., SHAP values) to identify key indirect predictors.

## Requirements

- Python 3.x
- pandas, numpy, scikit-learn
- torch (PyTorch)
- Google Colab recommended (free GPU access)

## Usage

All code is implemented in Google Colab notebooks:
- Data processing and feature engineering
- Model training and cross-validation
- Final model export (`final_salmonella_amr_pytorch_model.pth`)

## Conclusion

This project successfully demonstrates an end-to-end AI pipeline for predicting antimicrobial resistance in *Salmonella* from genomic data, achieving realistic and robust performance (87–97% accuracy) using deep learning on GPU. It highlights careful experimental design to avoid common pitfalls (data leakage) and reflects real biological challenges in genotype-phenotype prediction.