IMOHAG Project — Full Implementation

**This repository contains a complete implementation of the IMOHAG hybrid AI.
The framework integrates:

-Hybrid Recommendation Modeling

-Ranking Evaluation Metrics (Precision@K, Recall@K, MAP, nDCG)

-Sentiment Accuracy & Resolution-Time Analytics

-User Experience Modeling (SUS, NPS)

-Economic Insights (Revenue, Booking Patterns)

**Data Availability Statement

All datasets, code, evaluation scripts, and reproducible notebooks used in the IMOHAG study are publicly available in this repository.
No personal or sensitive data is included; all user identifiers are fully anonymized.

IMOHAG_Recommendation-System-main/
│
├── Datasets + Results/               # Datasets, final metrics, plots, reports
├── notebooks/                        # Reproducible Jupyter notebooks
├── scripts/                          # Executable scripts for each IMOHAG module
├── src/                              # Source code for the IMOHAG framework
│   ├── recommender/
│   ├── evaluation/
│   ├── sentiment/
│   ├── user_experience/
│   ├── economics/
│
├── IMOHAG-Practical implications     # Additional descriptive files
├── run_all.py                        # Unified pipeline runner (generated)
├── requirements.txt
├── LICENSE
└── README.md
Installation
Clone the repository
git clone https://github.com/<your-username>/IMOHAG_Recommendation-System-main.git
cd IMOHAG_Recommendation-System-main

Install dependencies (pip)
pip install -r requirements.txt

OR create a conda environment (recommended)
conda create -n imohag python=3.10
conda activate imohag
pip install -r requirements.txt
