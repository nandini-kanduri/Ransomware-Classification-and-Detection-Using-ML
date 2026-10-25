# Ransomware Classification and Detection Using Machine Learning Algorithms

An educational machine-learning pipeline for **binary ransomware-versus-benign classification** using labeled, tabular features. This is a **new reconstruction** inspired by the original college project presentation; it is not the recovered original source code.

## Algorithms
- Decision Tree
- Random Forest
- Naïve Bayes (Gaussian)
- Logistic Regression
- Neural Network (MLP)

## Requirements
Python 3.10+ recommended. Install with:

```bash
pip install -r requirements.txt
```

## Dataset
Supply your own **safe, labeled CSV of extracted features**, not live ransomware executables. The CSV must have a binary target column (default `label`) and feature columns. Example schema: `entropy,api_call_count,network_connections,label`. These are illustrative columns only, not a bundled dataset. Review and remove dataset-specific identifiers, target leakage, and sensitive information before training.

## Run
```bash
python train.py --data dataset.csv --target label
```

The script performs an 80/20 stratified split, trains five models, prints classification reports and confusion matrices, and writes model files and a comparison CSV/PNG locally. Generated artifacts are excluded from Git by default. A holdout score is not proof of production detection effectiveness. Validate with separate samples and account for data leakage and class imbalance.

## Original project presentation
The original presentation is available separately in this GitHub repository (`RAMSOMEWARE PPT1.pptx`). It reports Random Forest as the strongest model in its experiments. **This reconstructed pipeline has not been run on the original dataset; no accuracy claims are made here.**

## Safety
This project uses static datasets of features only. It does not execute ransomware or scan unknown binaries.
