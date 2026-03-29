# Processed Data (Intermediate Outputs)

## Overview

This folder contains intermediate data generated during the text processing pipeline.

Due to GitHub file size limitations, large files are not included in this repository.

---

## Files

### text_cleaned.csv (NOT included)

This file contains the cleaned and structured radiology notes after preprocessing.

It includes:
- Original radiology note text
- Cleaned text
- Extracted sections (IMPRESSION / FINDINGS / fallback)
- Block-level statistics
- Final processed text (`text_final`) used for feature extraction

---

## How to Reproduce

To generate `text_cleaned.csv`, run:

```bash
python scripts/02_text_cleaning.py
