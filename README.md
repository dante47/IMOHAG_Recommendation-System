# IMOHAG Project — Simulation-Based Decision Support Framework

This repository contains the full implementation of IMOHAG, a simulation-driven decision-support framework designed for tourist-experience optimization in data-sparse environments. The implementation supports the offline evaluation of recommendation and decision-support behaviors using synthetic datasets that reflect the characteristics of remote tourism destinations.

The framework provides:

Recommendation & Ranking Evaluation

Precision@K, Recall@K, MAP, nDCG

Baseline Model Comparison

Random, Relevance-Only, Revenue-Driven, Static Weighted-Sum, IMOHAG

User-Experience Proxies

Satisfaction indicators, SUS/NPS-style analytical proxies (synthetic context)

Economic Outcome Indicators

Revenue estimates and booking-source analysis

Sentiment & Feedback Modeling (Offline)

Sentiment-accuracy and resolution-time analytics derived from simulated interaction data

The repository is intended to support reproducible experimentation, comparison against baseline models, and transparent evaluation under synthetic but domain-aligned scenarios,

## Data Availability Statement

All datasets, code, and evaluation scripts used in the IMOHAG study are publicly available in this repository.

No personal or sensitive data is included; all user identifiers are fully anonymized.



# Overall Installation


**Clone the repository

$git clone https://github.com/<your-username>/IMOHAG_Recommendation-System-main.git

cd IMOHAG_Recommendation-System-main

**Install dependencies (pip)

$pip install -r requirements.txt

$python run_all.py




**OR create a conda environment (recommended)

conda create -n imohag python=3.10

conda activate imohag

pip install -r requirements.txt
