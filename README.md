# Radiology Text Processing Module (MIMIC-IV)

## Overview

This project implements a complete pipeline for processing radiology notes from the MIMIC-IV dataset and transforming them into structured numerical features for downstream modeling.

The objective is to extract clinically meaningful information from unstructured radiology reports and convert them into compact, model-ready representations.

---

## Project Structure

project/
│
├── scripts/
│   ├── 01_text_eda.py
│   ├── 02_text_cleaning.py
│   ├── 03_text_vectorization.py
│
├── features/
│   └── text_features.csv (not included due to size)
│
├── outputs/
│   ├── text_vectorization_summary.csv
│   └── text_tfidf_top_terms.csv
│
├── README.md
├── requirements.txt
└── .gitignore

---

## Data Description

This module processes radiology notes from the MIMIC-IV dataset.

Each record corresponds to one ICU stay (stay_id) and contains:

- Radiology report text
- Timestamp information
- Patient identifiers

---

## Pipeline

### Step 1: Exploratory Data Analysis

Run:

python scripts/01_text_eda.py

Purpose:

- Inspect data structure
- Analyze missing values
- Check text length distribution
- Verify uniqueness of stay_id

---

### Step 2: Text Cleaning and Section Extraction

Run:

python scripts/02_text_cleaning.py

Processing includes:

- Removing formatting artifacts and placeholders
- Splitting multiple reports within a single record
- Extracting clinically relevant sections:
  - IMPRESSION (priority)
  - FINDINGS (fallback)
  - FULL TEXT (fallback)
- Generating cleaned text field (text_final)

---

### Step 3: Text Vectorization

Run:

python scripts/03_text_vectorization.py

Methods:

- TF-IDF vectorization
  - max_features = 2000
  - n-gram range = (1, 2)
- Dimensionality reduction:
  - Truncated SVD
  - Final dimension = 100

---

## Output

### Main Output

features/text_features.csv

Structure:

- One row per stay_id
- 100-dimensional feature vector
- Ready for merging with other modalities (static / time-series)

---

### Additional Outputs

outputs/text_vectorization_summary.csv

- Summary of parameters and feature statistics

outputs/text_tfidf_top_terms.csv

- Top TF-IDF terms for interpretability

---

## Reproducibility

Large files are not included due to GitHub size limitations.

To reproduce all outputs, run:

python scripts/01_text_eda.py  
python scripts/02_text_cleaning.py  
python scripts/03_text_vectorization.py  

---

## Processed Data (Not Included)

The following intermediate file is not included:

processed/text_cleaned.csv

To generate it:

python scripts/02_text_cleaning.py

Reason for exclusion:

- File size exceeds GitHub limits
- Fully reproducible from provided scripts

---

## Text Features (Not Included)

The final feature file is not included:

features/text_features.csv

To generate it:

python scripts/03_text_vectorization.py

Reason for exclusion:

- File size exceeds GitHub limits
- Deterministically reproducible

---

## Key Design Decisions

- No aggregation required  
  Each stay_id corresponds to a single record

- Missing text preserved  
  Empty strings are used instead of deleting rows

- Section-based extraction  
  Prioritizes clinically meaningful content (IMPRESSION > FINDINGS > FULL TEXT)

- TF-IDF + SVD  
  Provides an efficient and interpretable feature representation

---

## Dependencies

Install required packages:

pip install -r requirements.txt

Main libraries:

- pandas
- numpy
- scikit-learn

---

## Notes

- Raw MIMIC-IV data is not included due to access restrictions
- All outputs can be regenerated using the provided scripts
- Features are aligned by stay_id and can be directly merged with other data sources

---

## Author

Developed as part of a clinical data science project using MIMIC-IV.
