# ImageCLEFmed 2026 Concept Detection

## Project Title
Hybrid Vision–Language Approach for Concept-Based Medical Caption Generation – Concept Detection

## Student
Bashar Alsaleh

## Supervisors
Dr. Riad Sonbol  
Eng. Zeinab Baghdadi

## Faculty
Faculty of Artificial Intelligence Engineering  
Syrian Private University

## Project Overview
This repository contains the final thesis and implementation code for the Senior 1 Graduation Project.

The project focuses on the Concept Detection Task of ImageCLEFmed Caption 2026. The goal is to predict UMLS-aligned medical concepts from radiology images as a semantic foundation for future concept-based medical caption generation.

## Official ImageCLEFmed Caption 2026 Result
The submitted run achieved:

- Official test score: **0.5725**
- Secondary score: **0.9419**
- Corrected final ranking: **4th place**
- Task: **Concept Detection Task**

## Proposed Method
The final proposed model is a frequency-aware multi-head Swin Transformer for long-tailed multi-label medical concept detection.

The model uses:

- Swin Transformer backbone
- Frequency-aware multi-head classification
- Asymmetric Loss
- Weighted sampling
- Exponential Moving Average
- Three-fold multilabel stratified validation
- Per-head threshold calibration

## Repository Files

- [Final Thesis PDF](./Thesis_Bashar_f.pdf)
- [Final Notebook](./S1_Final_GitHub.ipynb)
- [Backup Python Script](./S1_Final_GitHub_backup.py)
- [Requirements](./requirements.txt)

## Dataset
The ImageCLEFmed Caption 2026 dataset is not included in this repository due to size and usage restrictions.

Expected local dataset structure:

```text
data/
└── dev_concept/
    ├── images/
    ├── concepts.csv
    └── attribution.csv
