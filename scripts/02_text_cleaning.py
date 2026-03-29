"""
02_text_cleaning.py

Purpose:
    Clean and structure the MIMIC-IV radiology text table for downstream NLP modeling.

Main steps:
    1. Read raw text CSV
    2. Keep key columns
    3. Fill missing text with empty string
    4. Clean formatting artifacts
    5. Split one note into multiple report blocks using "-----"
    6. Extract clinically informative section with priority:
           IMPRESSION > FINDINGS > FULL TEXT
    7. Concatenate extracted block-level text into one final text per stay_id
    8. Save cleaned output

Outputs:
    - processed/text_cleaned.csv

Recommended output columns:
    - stay_id
    - subject_id
    - radiology_note_time_min
    - radiology_note_time_max
    - radiology_note_text
    - text_clean_basic
    - n_blocks
    - n_impression_blocks
    - n_findings_blocks
    - text_final
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np


# =========================
# Configuration
# =========================
BASE_DIR = Path(__file__).resolve().parent
RAW_TEXT_PATH = BASE_DIR / "Assignment2_mimic dataset" / "MIMIC-IV-text(Group Assignment).csv"
OUTPUT_DIR = BASE_DIR / "processed"
OUTPUT_PATH = OUTPUT_DIR / "text_cleaned.csv"


# =========================
# Utility functions
# =========================
def print_title(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def basic_clean_text(text: str) -> str:
    """
    Basic text cleaning that preserves medical meaning.

    Steps:
        - convert to string
        - remove anonymization placeholders like ___
        - normalize newlines/tabs
        - remove repeated separators
        - collapse repeated spaces
        - strip leading/trailing spaces
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove anonymization placeholders
    text = re.sub(r"___+", " ", text)

    # Normalize line breaks and tabs
    text = text.replace("\r", "\n").replace("\t", " ")

    # Remove repeated separator lines
    text = re.sub(r"-{3,}", " [SEP] ", text)

    # Collapse multiple spaces/newlines
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)

    # Remove spaces around newlines
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


def split_into_blocks(text: str) -> list[str]:
    """
    Split a note into report blocks.

    The raw notes often contain multiple reports separated by lines like:
        -----

    After basic cleaning, those separators are normalized to [SEP].
    """
    if not text:
        return []

    blocks = [block.strip() for block in text.split("[SEP]") if block.strip()]
    return blocks


def normalize_for_output(text: str) -> str:
    """
    Final normalization before saving extracted text.
    Keeps sentence content while making it one clean string.
    """
    if not text:
        return ""

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_section(block: str, section_name: str) -> str:
    """
    Extract a target section from one report block.

    Example sections:
        IMPRESSION
        FINDINGS

    We capture text after the target heading until the next likely heading or the end.
    """
    if not block:
        return ""

    # Robust pattern:
    # section heading may appear like:
    #   IMPRESSION:
    #   IMPRESSION
    #   FINDINGS:
    #
    # Stop when reaching another heading in ALL CAPS followed by colon,
    # or end of string.
    pattern = rf"(?is)\b{section_name}\b\s*:?\s*(.*?)(?=\n[A-Z][A-Z /()\-]{2,}:|\Z)"
    match = re.search(pattern, block)

    if match:
        extracted = match.group(1).strip()
        return normalize_for_output(extracted)

    return ""


def clean_block_fallback(block: str) -> str:
    """
    Fallback when IMPRESSION/FINDINGS are unavailable.
    Use the whole block after light normalization.
    """
    return normalize_for_output(block)


def extract_best_text_from_block(block: str) -> tuple[str, str]:
    """
    Return:
        extracted_text, source_label

    Priority:
        1. IMPRESSION
        2. FINDINGS
        3. FULL_TEXT
    """
    if not block:
        return "", "empty"

    impression = extract_section(block, "IMPRESSION")
    if impression:
        return impression, "impression"

    findings = extract_section(block, "FINDINGS")
    if findings:
        return findings, "findings"

    full_text = clean_block_fallback(block)
    if full_text:
        return full_text, "full_text"

    return "", "empty"


def process_note(text: str) -> dict:
    """
    Process one radiology_note_text cell.

    Returns:
        {
            "text_clean_basic": ...,
            "n_blocks": ...,
            "n_impression_blocks": ...,
            "n_findings_blocks": ...,
            "text_final": ...,
            "source_summary": ...
        }
    """
    text_clean_basic = basic_clean_text(text)

    if not text_clean_basic:
        return {
            "text_clean_basic": "",
            "n_blocks": 0,
            "n_impression_blocks": 0,
            "n_findings_blocks": 0,
            "text_final": "",
            "source_summary": "empty"
        }

    blocks = split_into_blocks(text_clean_basic)

    extracted_texts = []
    source_labels = []

    for block in blocks:
        extracted_text, source_label = extract_best_text_from_block(block)

        if extracted_text:
            extracted_texts.append(extracted_text)
            source_labels.append(source_label)

    n_impression_blocks = sum(label == "impression" for label in source_labels)
    n_findings_blocks = sum(label == "findings" for label in source_labels)

    # Concatenate block-level extracted texts into one final text
    text_final = " ".join(extracted_texts)
    text_final = normalize_for_output(text_final)

    source_summary = "+".join(source_labels) if source_labels else "empty"

    return {
        "text_clean_basic": text_clean_basic,
        "n_blocks": len(blocks),
        "n_impression_blocks": n_impression_blocks,
        "n_findings_blocks": n_findings_blocks,
        "text_final": text_final,
        "source_summary": source_summary
    }


# =========================
# Main
# =========================
def main() -> None:
    print_title("1. READ RAW DATA")

    if not RAW_TEXT_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {RAW_TEXT_PATH}\n"
            f"Please check the folder name and file location."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_TEXT_PATH)

    print(f"Loaded file: {RAW_TEXT_PATH}")
    print(f"Original shape: {df.shape}")

    required_cols = [
        "subject_id",
        "stay_id",
        "radiology_note_time_min",
        "radiology_note_time_max",
        "radiology_note_text"
    ]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    # Keep key columns only
    df = df[required_cols].copy()

    print_title("2. FILL MISSING TEXT")
    missing_before = df["radiology_note_text"].isna().sum()
    print(f"Missing radiology_note_text before fillna: {missing_before}")

    df["radiology_note_text"] = df["radiology_note_text"].fillna("")

    missing_after = df["radiology_note_text"].isna().sum()
    print(f"Missing radiology_note_text after fillna: {missing_after}")

    print_title("3. PROCESS TEXT")

    processed_rows = df["radiology_note_text"].apply(process_note)
    processed_df = pd.DataFrame(processed_rows.tolist())

    result_df = pd.concat([df, processed_df], axis=1)

    print("Processed shape:", result_df.shape)

    print_title("4. SANITY CHECK")

    print("Empty final text ratio:",
          round((result_df["text_final"].str.strip() == "").mean(), 4))

    print("\nBlock count summary:")
    print(result_df["n_blocks"].describe())

    print("\nFinal text length summary (characters):")
    result_df["text_final_len_char"] = result_df["text_final"].apply(len)
    print(result_df["text_final_len_char"].describe())

    print("\nFinal text length summary (words):")
    result_df["text_final_len_word"] = result_df["text_final"].apply(lambda x: len(str(x).split()))
    print(result_df["text_final_len_word"].describe())

    print("\nSource summary examples:")
    print(result_df["source_summary"].value_counts().head(10))

    print_title("5. SAMPLE OUTPUT")

    sample_cols = [
        "stay_id",
        "radiology_note_text",
        "n_blocks",
        "n_impression_blocks",
        "n_findings_blocks",
        "source_summary",
        "text_final"
    ]

    sample_df = result_df[sample_cols].sample(n=min(3, len(result_df)), random_state=42)

    pd.set_option("display.max_colwidth", 500)

    for _, row in sample_df.iterrows():
        print("\n" + "-" * 100)
        print(f"stay_id: {row['stay_id']}")
        print(f"n_blocks: {row['n_blocks']}")
        print(f"n_impression_blocks: {row['n_impression_blocks']}")
        print(f"n_findings_blocks: {row['n_findings_blocks']}")
        print(f"source_summary: {row['source_summary']}")
        print("\ntext_final preview:")
        print(str(row["text_final"])[:2000])
        print("-" * 100)

    print_title("6. SAVE OUTPUT")

    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved cleaned text file to: {OUTPUT_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()