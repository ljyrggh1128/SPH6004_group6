"""
01_text_eda.py

Purpose:
    Perform exploratory data analysis (EDA) and sanity checks for the
    MIMIC-IV radiology text table.

Main checks:
    1. Basic dataset structure
    2. Data types and missing values
    3. stay_id uniqueness and note count per stay
    4. Duplicate rows
    5. Text length distribution
    6. Time field parsing and consistency
    7. Section keyword frequency (IMPRESSION / FINDINGS / INDICATION / COMPARISON)
    8. Random text samples
    9. Example stay_id with multiple notes
    10. Save summary outputs

Outputs:
    - outputs/eda_text_summary.csv
    - outputs/stay_note_count.csv
    - outputs/missing_summary.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np


# =========================
# Configuration
# =========================
BASE_DIR = Path(__file__).resolve().parent
RAW_TEXT_PATH = BASE_DIR / "Assignment2_mimic dataset" / "MIMIC-IV-text(Group Assignment).csv"
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42
N_RANDOM_SAMPLES = 3


# =========================
# Utility functions
# =========================
def print_title(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def safe_preview(text: str, max_chars: int = 2000) -> str:
    if pd.isna(text):
        return "[MISSING]"
    text = str(text)
    return text[:max_chars]


def main() -> None:
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================
    # 1. Read data
    # =========================
    print_title("1. READ DATA")

    if not RAW_TEXT_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {RAW_TEXT_PATH}\n"
            f"Please place the CSV file at this location or update RAW_TEXT_PATH."
        )

    df = pd.read_csv(RAW_TEXT_PATH)

    # Display settings
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 300)

    print(f"Loaded file: {RAW_TEXT_PATH}")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 3 rows:")
    print(df.head(3))

    # Check required columns
    required_cols = [
        "stay_id",
        "radiology_note_text",
        "radiology_note_time_min",
        "radiology_note_time_max",
    ]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(
            f"Missing required columns: {missing_required}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # =========================
    # 2. Data types and missing values
    # =========================
    print_title("2. DATA TYPES AND MISSING VALUES")

    print("DataFrame info:")
    print(df.info())

    missing_count = df.isna().sum()
    missing_ratio = (df.isna().mean() * 100).round(2)

    missing_summary = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_ratio_percent": missing_ratio,
        }
    ).sort_values(by="missing_ratio_percent", ascending=False)

    print("\nMissing summary:")
    print(missing_summary)

    missing_summary.to_csv(OUTPUT_DIR / "missing_summary.csv", index=True)

    # =========================
    # 3. stay_id sanity check
    # =========================
    print_title("3. STAY_ID SANITY CHECK")

    total_rows = len(df)
    unique_stays = df["stay_id"].nunique(dropna=True)

    print(f"Total rows: {total_rows}")
    print(f"Unique stay_id: {unique_stays}")
    print(f"Are stay_id unique? {total_rows == unique_stays}")

    stay_counts = df.groupby("stay_id").size().reset_index(name="note_count")

    print("\nDistribution of number of notes per stay_id:")
    print(stay_counts["note_count"].describe())

    print("\nTop 10 stay_id with most notes:")
    print(stay_counts.sort_values("note_count", ascending=False).head(10))

    stay_counts.to_csv(OUTPUT_DIR / "stay_note_count.csv", index=False)

    # =========================
    # 4. Duplicate check
    # =========================
    print_title("4. DUPLICATE CHECK")

    duplicate_rows = df.duplicated().sum()
    duplicate_stay_text = df.duplicated(subset=["stay_id", "radiology_note_text"]).sum()

    print(f"Number of fully duplicated rows: {duplicate_rows}")
    print(f"Number of duplicated (stay_id, radiology_note_text): {duplicate_stay_text}")

    # =========================
    # 5. Text length analysis
    # =========================
    print_title("5. TEXT LENGTH ANALYSIS")

    df["radiology_note_text"] = df["radiology_note_text"].fillna("")

    df["text_len_char"] = df["radiology_note_text"].apply(len)
    df["text_len_word"] = df["radiology_note_text"].apply(lambda x: len(str(x).split()))

    print("Character length summary:")
    print(df["text_len_char"].describe())

    print("\nWord length summary:")
    print(df["text_len_word"].describe())

    print("\nTop 5 longest notes by character length:")
    longest_notes = (
        df[["stay_id", "text_len_char", "text_len_word", "radiology_note_text"]]
        .sort_values("text_len_char", ascending=False)
        .head(5)
    )
    print(longest_notes)

    # =========================
    # 6. Time field check
    # =========================
    print_title("6. TIME FIELD CHECK")

    # The year values in MIMIC are shifted and may not be parseable by default.
    # We still attempt parsing for a rough sanity check.
    for col in ["radiology_note_time_min", "radiology_note_time_max"]:
        df[f"{col}_parsed"] = pd.to_datetime(df[col], errors="coerce")

    print("Parsed time missing values:")
    print(df[["radiology_note_time_min_parsed", "radiology_note_time_max_parsed"]].isna().sum())

    time_issue_count = (
        (df["radiology_note_time_min_parsed"].notna())
        & (df["radiology_note_time_max_parsed"].notna())
        & (df["radiology_note_time_min_parsed"] > df["radiology_note_time_max_parsed"])
    ).sum()

    print(f"Rows with radiology_note_time_min > radiology_note_time_max: {time_issue_count}")

    # Also keep a simple raw-string equality check for missing or odd values
    print("\nRaw time sample:")
    print(df[["radiology_note_time_min", "radiology_note_time_max"]].head(5))

    # =========================
    # 7. Section keyword check
    # =========================
    print_title("7. SECTION KEYWORD CHECK")

    text_upper = df["radiology_note_text"].str.upper()

    section_stats = {
        "contains_IMPRESSION_percent": round(text_upper.str.contains("IMPRESSION", na=False).mean() * 100, 2),
        "contains_FINDINGS_percent": round(text_upper.str.contains("FINDINGS", na=False).mean() * 100, 2),
        "contains_INDICATION_percent": round(text_upper.str.contains("INDICATION", na=False).mean() * 100, 2),
        "contains_COMPARISON_percent": round(text_upper.str.contains("COMPARISON", na=False).mean() * 100, 2),
    }

    for key, value in section_stats.items():
        print(f"{key}: {value}%")

    # Estimate multiple report blocks joined in one cell
    df["separator_count"] = df["radiology_note_text"].str.count("-----")

    print("\nSeparator count summary:")
    print(df["separator_count"].describe())

    # =========================
    # 8. Random sample notes
    # =========================
    print_title("8. RANDOM SAMPLE NOTES")

    if len(df) > 0:
        sample_size = min(N_RANDOM_SAMPLES, len(df))
        sample_df = df[
            ["stay_id", "radiology_note_text", "text_len_char", "text_len_word"]
        ].sample(n=sample_size, random_state=RANDOM_STATE)

        for _, row in sample_df.iterrows():
            print("\n" + "-" * 100)
            print(f"stay_id: {row['stay_id']}")
            print(f"text_len_char: {row['text_len_char']}")
            print(f"text_len_word: {row['text_len_word']}")
            print("Text preview:")
            print(safe_preview(row["radiology_note_text"], max_chars=2000))
            print("-" * 100)
    else:
        print("Dataset is empty. No random samples to show.")

    # =========================
    # 9. Example stay_id with multiple notes
    # =========================
    print_title("9. EXAMPLE STAY_ID WITH MULTIPLE NOTES")

    multi_note_stays = stay_counts.loc[stay_counts["note_count"] > 1, "stay_id"].tolist()

    if len(multi_note_stays) > 0:
        example_stay = multi_note_stays[0]
        print(f"Example stay_id with multiple notes: {example_stay}")

        example_rows = df.loc[
            df["stay_id"] == example_stay,
            ["stay_id", "radiology_note_time_min", "radiology_note_time_max", "radiology_note_text"],
        ].head(10)

        print(example_rows)
    else:
        print("No stay_id with multiple notes found.")

    # =========================
    # 10. EDA summary output
    # =========================
    print_title("10. EDA SUMMARY")

    empty_text_ratio = (df["radiology_note_text"].str.strip() == "").mean()

    eda_summary = {
        "total_rows": total_rows,
        "total_columns": df.shape[1],
        "unique_stay_id": unique_stays,
        "stay_id_unique_ratio": round(unique_stays / total_rows, 4) if total_rows > 0 else np.nan,
        "empty_text_ratio": round(empty_text_ratio, 4) if total_rows > 0 else np.nan,
        "duplicate_rows": int(duplicate_rows),
        "duplicate_stay_text_rows": int(duplicate_stay_text),
        "avg_text_len_char": round(df["text_len_char"].mean(), 2) if total_rows > 0 else np.nan,
        "median_text_len_char": round(df["text_len_char"].median(), 2) if total_rows > 0 else np.nan,
        "max_text_len_char": int(df["text_len_char"].max()) if total_rows > 0 else np.nan,
        "avg_text_len_word": round(df["text_len_word"].mean(), 2) if total_rows > 0 else np.nan,
        "median_text_len_word": round(df["text_len_word"].median(), 2) if total_rows > 0 else np.nan,
        "max_text_len_word": int(df["text_len_word"].max()) if total_rows > 0 else np.nan,
        "avg_notes_per_stay": round(stay_counts["note_count"].mean(), 2) if len(stay_counts) > 0 else np.nan,
        "max_notes_per_stay": int(stay_counts["note_count"].max()) if len(stay_counts) > 0 else np.nan,
        "contains_IMPRESSION_percent": section_stats["contains_IMPRESSION_percent"],
        "contains_FINDINGS_percent": section_stats["contains_FINDINGS_percent"],
        "contains_INDICATION_percent": section_stats["contains_INDICATION_percent"],
        "contains_COMPARISON_percent": section_stats["contains_COMPARISON_percent"],
    }

    eda_summary_df = pd.DataFrame.from_dict(eda_summary, orient="index", columns=["value"])
    print(eda_summary_df)

    eda_summary_df.to_csv(OUTPUT_DIR / "eda_text_summary.csv", index=True)

    print("\nSaved files:")
    print(f"- {OUTPUT_DIR / 'eda_text_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'stay_note_count.csv'}")
    print(f"- {OUTPUT_DIR / 'missing_summary.csv'}")

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()