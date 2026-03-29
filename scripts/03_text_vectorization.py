"""
03_text_vectorization.py

Purpose:
    Convert cleaned radiology text into numerical features for downstream modeling.

Pipeline:
    1. Read processed/text_cleaned.csv
    2. Use text_final as the input text field
    3. Apply TF-IDF vectorization
    4. Apply TruncatedSVD for dimensionality reduction
    5. Save text_features.csv

Outputs:
    - features/text_features.csv
    - outputs/text_vectorization_summary.csv
    - outputs/text_tfidf_top_terms.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# =========================
# Configuration
# =========================
BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "processed" / "text_cleaned.csv"
FEATURE_DIR = BASE_DIR / "features"
OUTPUT_DIR = BASE_DIR / "outputs"

FEATURE_OUTPUT_PATH = FEATURE_DIR / "text_features.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "text_vectorization_summary.csv"
VOCAB_OUTPUT_PATH = OUTPUT_DIR / "text_tfidf_top_terms.csv"

TEXT_COL = "text_final"
ID_COL = "stay_id"

# TF-IDF parameters
MAX_FEATURES = 2000
MIN_DF = 5
MAX_DF = 0.9
NGRAM_RANGE = (1, 2)

# SVD parameters
SVD_DIM = 100
RANDOM_STATE = 42


# =========================
# Utility functions
# =========================
def print_title(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def get_top_terms_from_tfidf(vectorizer: TfidfVectorizer, tfidf_matrix, top_n: int = 100) -> pd.DataFrame:
    """
    Get top terms by average TF-IDF score across all documents.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()

    top_idx = np.argsort(mean_scores)[::-1][:top_n]

    top_terms_df = pd.DataFrame({
        "term": feature_names[top_idx],
        "mean_tfidf": mean_scores[top_idx]
    })

    return top_terms_df


def main() -> None:
    print_title("1. READ CLEANED TEXT DATA")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            f"Please run 02_text_cleaning.py first."
        )

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    print(f"Loaded file: {INPUT_PATH}")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    if ID_COL not in df.columns:
        raise ValueError(f"Missing required ID column: {ID_COL}")

    if TEXT_COL not in df.columns:
        raise ValueError(f"Missing required text column: {TEXT_COL}")

    # Keep only needed columns
    df = df[[ID_COL, TEXT_COL]].copy()

    # Fill missing text just in case
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    print("\nBasic text stats:")
    df["text_len_char"] = df[TEXT_COL].apply(len)
    df["text_len_word"] = df[TEXT_COL].apply(lambda x: len(x.split()))
    print(df[["text_len_char", "text_len_word"]].describe())

    # =========================
    # 2. Sanity check
    # =========================
    print_title("2. SANITY CHECK")

    n_rows = len(df)
    n_unique_stays = df[ID_COL].nunique()
    empty_ratio = (df[TEXT_COL].str.strip() == "").mean()

    print(f"Total rows: {n_rows}")
    print(f"Unique stay_id: {n_unique_stays}")
    print(f"stay_id unique? {n_rows == n_unique_stays}")
    print(f"Empty text ratio: {empty_ratio:.4f}")

    if n_rows != n_unique_stays:
        raise ValueError(
            "stay_id is not unique in text_cleaned.csv. "
            "Please check the cleaning script before vectorization."
        )

    # =========================
    # 3. TF-IDF
    # =========================
    print_title("3. TF-IDF VECTORIZATION")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=NGRAM_RANGE
    )

    tfidf_matrix = vectorizer.fit_transform(df[TEXT_COL])

    print("TF-IDF matrix shape:", tfidf_matrix.shape)

    vocab_size = len(vectorizer.get_feature_names_out())
    print("Vocabulary size after filtering:", vocab_size)

    # =========================
    # 4. Truncated SVD
    # =========================
    print_title("4. TRUNCATED SVD")

    # SVD dim cannot exceed vocabulary size - 1
    actual_svd_dim = min(SVD_DIM, max(1, vocab_size - 1))

    if actual_svd_dim < SVD_DIM:
        print(f"Requested SVD_DIM={SVD_DIM}, adjusted to {actual_svd_dim} due to vocabulary size.")

    svd = TruncatedSVD(
        n_components=actual_svd_dim,
        random_state=RANDOM_STATE
    )

    text_embeddings = svd.fit_transform(tfidf_matrix)

    print("SVD output shape:", text_embeddings.shape)

    explained_variance_sum = svd.explained_variance_ratio_.sum()
    print(f"Explained variance ratio sum: {explained_variance_sum:.4f}")

    # =========================
    # 5. Build feature table
    # =========================
    print_title("5. BUILD FEATURE TABLE")

    feature_col_names = [f"text_feat_{i+1}" for i in range(text_embeddings.shape[1])]

    features_df = pd.DataFrame(text_embeddings, columns=feature_col_names)
    features_df.insert(0, ID_COL, df[ID_COL].values)

    print("Feature table shape:", features_df.shape)
    print("\nFirst 5 rows:")
    print(features_df.head())

    # =========================
    # 6. Feature quality check
    # =========================
    print_title("6. FEATURE QUALITY CHECK")

    duplicate_id_count = features_df[ID_COL].duplicated().sum()
    missing_value_count = features_df.isna().sum().sum()

    numeric_feature_cols = [col for col in features_df.columns if col != ID_COL]
    zero_variance_cols = [
        col for col in numeric_feature_cols
        if np.isclose(features_df[col].var(), 0)
    ]

    print(f"Duplicated stay_id count: {duplicate_id_count}")
    print(f"Total missing values in feature table: {missing_value_count}")
    print(f"Zero-variance feature count: {len(zero_variance_cols)}")

    if len(zero_variance_cols) > 0:
        print("Example zero-variance columns:", zero_variance_cols[:10])

    # =========================
    # 7. Save outputs
    # =========================
    print_title("7. SAVE OUTPUTS")

    features_df.to_csv(FEATURE_OUTPUT_PATH, index=False)
    print(f"Saved text features to: {FEATURE_OUTPUT_PATH}")

    summary_dict = {
        "n_rows": n_rows,
        "n_unique_stay_id": n_unique_stays,
        "empty_text_ratio": round(float(empty_ratio), 6),
        "tfidf_vocab_size": int(vocab_size),
        "tfidf_max_features": int(MAX_FEATURES),
        "tfidf_min_df": int(MIN_DF),
        "tfidf_max_df": float(MAX_DF),
        "tfidf_ngram_min": int(NGRAM_RANGE[0]),
        "tfidf_ngram_max": int(NGRAM_RANGE[1]),
        "svd_requested_dim": int(SVD_DIM),
        "svd_actual_dim": int(actual_svd_dim),
        "svd_explained_variance_ratio_sum": round(float(explained_variance_sum), 6),
        "duplicate_stay_id_count": int(duplicate_id_count),
        "missing_value_count": int(missing_value_count),
        "zero_variance_feature_count": int(len(zero_variance_cols))
    }

    summary_df = pd.DataFrame.from_dict(summary_dict, orient="index", columns=["value"])
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=True)
    print(f"Saved vectorization summary to: {SUMMARY_OUTPUT_PATH}")

    top_terms_df = get_top_terms_from_tfidf(vectorizer, tfidf_matrix, top_n=100)
    top_terms_df.to_csv(VOCAB_OUTPUT_PATH, index=False)
    print(f"Saved top TF-IDF terms to: {VOCAB_OUTPUT_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()