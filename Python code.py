# Reproducible keyword-title overlap analysis.
#
# This script calculates token-level overlap between article titles and
# provided keywords. It is intended as supplementary code that can be run on
# any independent metadata dataset with the required columns.
#
# Required input columns:
#     - title
#     - keywords
#
# Optional input column:
#     - id
#
# The script does not include or require the original research dataset. It does
# not contain dataset identifiers, personal names, internal paths, or
# local/private infrastructure details.
#
# Example:
#     python python-for-keywords.py \
#         --input input/metadata.csv \
#         --output output
#
# The default CSV separator is a semicolon because keyword fields may contain
# commas internally in some exported metadata files. Use --sep "," if your
# input file is comma-separated.

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    from nltk.stem import WordNetLemmatizer
except ImportError:
    WordNetLemmatizer = None


# This stop-word list is used only for the sensitivity checks. The main
# analysis retains stop words in order to measure formal token-level overlap.
ENGLISH_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am",
    "an", "and", "any", "are", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "did", "do", "does", "doing", "don", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "you", "your",
    "yours", "yourself", "yourselves"
}


def parse_args():
    # Read command-line arguments.
    parser = argparse.ArgumentParser(
        description="Calculate keyword-title token overlap indicators."
    )
    parser.add_argument(
        "--input",
        default="input/metadata.csv",
        help="Path to an input CSV file with title and keywords columns."
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Directory where output CSV files will be saved."
    )
    parser.add_argument(
        "--sep",
        default=";",
        help="CSV separator used in the input file. Default: ';'."
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Input and output encoding. Default: utf-8-sig."
    )
    return parser.parse_args()


def tokenize(text):
    # Convert text into lowercase tokens.
    #
    # Processing steps:
    # - lowercase conversion;
    # - removal of characters other than letters, digits, spaces and hyphens;
    # - whitespace normalization;
    # - splitting into tokens;
    # - removal of leading/trailing hyphens;
    # - exclusion of one-character tokens.
    if pd.isna(text):
        return []

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [token.strip("-") for token in tokens]
    tokens = [token for token in tokens if token and len(token) > 1]

    return tokens


def split_keywords(text):
    # Split the keyword field into keyword items.
    #
    # Keywords are split by comma or semicolon. Multi-word keyword expressions
    # are preserved at this stage and tokenized later.
    if pd.isna(text):
        return []

    parts = re.split(r"[;,]", str(text))
    parts = [re.sub(r"\s+", " ", part).strip(" .;,:") for part in parts]

    return [part for part in parts if part]


def get_wordnet_lemmatizer():
    # Return a WordNet lemmatizer for the lemmatized sensitivity mode.
    #
    # This mode requires the optional NLTK package and WordNet data:
    #     pip install nltk
    #     python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
    if WordNetLemmatizer is None:
        raise ImportError(
            "The lemmatized sensitivity mode requires NLTK. "
            "Install it with: pip install nltk"
        )

    lemmatizer = WordNetLemmatizer()

    try:
        lemmatizer.lemmatize("students")
    except LookupError as exc:
        raise LookupError(
            "The lemmatized sensitivity mode requires NLTK WordNet data. "
            "Run: python -c \"import nltk; "
            "nltk.download('wordnet'); nltk.download('omw-1.4')\""
        ) from exc

    return lemmatizer


def lemmatize_token(token, lemmatizer):
    # Apply conservative WordNet lemmatization without POS tagging.
    #
    # Several WordNet POS categories are tried so common variants such as
    # students/student and learning/learn can be normalized in the sensitivity
    # analysis without changing the main analysis.
    lemma = lemmatizer.lemmatize(token, pos="n")
    lemma = lemmatizer.lemmatize(lemma, pos="v")
    lemma = lemmatizer.lemmatize(lemma, pos="a")
    lemma = lemmatizer.lemmatize(lemma, pos="r")
    return lemma


def preprocess_tokens_for_sensitivity(
    text,
    remove_stopwords=False,
    lemmatize=False,
    lemmatizer=None
):
    # Tokenize text for the sensitivity analysis.
    #
    # Base tokenization is identical to the main analysis. Stop-word removal
    # and lemmatization are applied only when requested by a sensitivity mode.
    tokens = tokenize(text)

    if remove_stopwords:
        tokens = [token for token in tokens if token not in ENGLISH_STOP_WORDS]

    if lemmatize:
        if lemmatizer is None:
            lemmatizer = get_wordnet_lemmatizer()
        tokens = [lemmatize_token(token, lemmatizer) for token in tokens]

    return tokens


def calculate_article_metrics(df):
    # Calculate article-level token metrics in the main preprocessing mode.
    #
    # Main mode:
    # - stop words retained;
    # - no lemmatization.
    rows = []
    all_keyword_tokens = []
    all_overlap_tokens = []
    all_expansion_tokens = []

    for _, row in df.iterrows():
        title = row["title"]
        keywords = row["keywords"]

        title_tokens = tokenize(title)
        keyword_items = split_keywords(keywords)

        keyword_tokens = []
        for keyword_item in keyword_items:
            keyword_tokens.extend(tokenize(keyword_item))

        title_set = set(title_tokens)
        keyword_set = set(keyword_tokens)

        overlap_tokens = [token for token in keyword_tokens if token in title_set]
        expansion_tokens = [
            token for token in keyword_tokens if token not in title_set
        ]

        total_keyword_tokens = len(keyword_tokens)
        overlap_count = len(overlap_tokens)
        expansion_count = len(expansion_tokens)

        # TKOI: Title Keyword Overlap Index.
        tkoi = overlap_count / total_keyword_tokens if total_keyword_tokens else None

        # TKEI: Title Keyword Expansion Index.
        tkei = expansion_count / total_keyword_tokens if total_keyword_tokens else None

        # Lexical coverage ratio:
        # unique keyword tokens divided by unique title tokens.
        coverage_ratio = (
            len(keyword_set) / len(title_set)
            if len(title_set) > 0 else None
        )

        all_keyword_tokens.extend(keyword_tokens)
        all_overlap_tokens.extend(overlap_tokens)
        all_expansion_tokens.extend(expansion_tokens)

        rows.append({
            "id": row["id"],
            "title": title,
            "keywords": keywords,
            "title_tokens": "; ".join(title_tokens),
            "keyword_tokens": "; ".join(keyword_tokens),
            "title_token_count": len(title_tokens),
            "title_unique_token_count": len(title_set),
            "keyword_item_count": len(keyword_items),
            "keyword_token_count": total_keyword_tokens,
            "keyword_unique_token_count": len(keyword_set),
            "overlap_token_count": overlap_count,
            "expansion_token_count": expansion_count,
            "TKOI": tkoi,
            "TKEI": tkei,
            "TKEI_plus_TKOI": (
                tkei + tkoi
                if tkei is not None and tkoi is not None
                else None
            ),
            "coverage_ratio": coverage_ratio,
            "overlap_tokens": "; ".join(overlap_tokens),
            "expansion_tokens": "; ".join(expansion_tokens)
        })

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df.dropna(subset=["TKEI", "TKOI"]).reset_index(drop=True)

    token_lists = {
        "all_keyword_tokens": all_keyword_tokens,
        "all_overlap_tokens": all_overlap_tokens,
        "all_expansion_tokens": all_expansion_tokens
    }

    return metrics_df, token_lists


def add_quartile_groups(metrics_df):
    # Add dataset-relative quartile groups for TKEI and TKOI.
    #
    # Quartile groups describe the internal distribution of values in the input
    # dataset. They should not be interpreted as universal thresholds for
    # direct comparison across unrelated datasets.
    q1_tkei = metrics_df["TKEI"].quantile(0.25)
    q2_tkei = metrics_df["TKEI"].quantile(0.50)
    q3_tkei = metrics_df["TKEI"].quantile(0.75)

    q1_tkoi = metrics_df["TKOI"].quantile(0.25)
    q2_tkoi = metrics_df["TKOI"].quantile(0.50)
    q3_tkoi = metrics_df["TKOI"].quantile(0.75)

    def classify_by_tkei(tkei):
        if pd.isna(tkei):
            return "undefined"
        if tkei == 0:
            return "complete title replication"
        if tkei <= q1_tkei:
            return "low expansion"
        if tkei <= q2_tkei:
            return "moderate-low expansion"
        if tkei <= q3_tkei:
            return "moderate-high expansion"
        return "high expansion"

    def classify_by_tkoi(tkoi):
        if pd.isna(tkoi):
            return "undefined"
        if tkoi == 0:
            return "complete title independence"
        if tkoi <= q1_tkoi:
            return "low overlap"
        if tkoi <= q2_tkoi:
            return "moderate-low overlap"
        if tkoi <= q3_tkoi:
            return "moderate-high overlap"
        return "high overlap"

    metrics_df = metrics_df.copy()
    metrics_df["TKEI_quartile_group"] = metrics_df["TKEI"].apply(
        classify_by_tkei
    )
    metrics_df["TKOI_quartile_group"] = metrics_df["TKOI"].apply(
        classify_by_tkoi
    )

    quartiles = {
        "q1_TKEI": q1_tkei,
        "q2_TKEI": q2_tkei,
        "q3_TKEI": q3_tkei,
        "q1_TKOI": q1_tkoi,
        "q2_TKOI": q2_tkoi,
        "q3_TKOI": q3_tkoi
    }

    return metrics_df, quartiles


def build_summary(metrics_df, quartiles):
    # Build a one-row summary table for the main preprocessing mode.
    total_keyword_tokens = metrics_df["keyword_token_count"].sum()

    summary = {
        "total_articles": len(metrics_df),
        "mean_keywords_per_article": metrics_df["keyword_item_count"].mean(),
        "median_keywords_per_article": metrics_df["keyword_item_count"].median(),
        "min_keywords_per_article": metrics_df["keyword_item_count"].min(),
        "max_keywords_per_article": metrics_df["keyword_item_count"].max(),
        "mean_keyword_tokens": metrics_df["keyword_token_count"].mean(),
        "median_keyword_tokens": metrics_df["keyword_token_count"].median(),
        "min_keyword_tokens": metrics_df["keyword_token_count"].min(),
        "max_keyword_tokens": metrics_df["keyword_token_count"].max(),
        "mean_title_tokens": metrics_df["title_token_count"].mean(),
        "median_title_tokens": metrics_df["title_token_count"].median(),
        "min_title_tokens": metrics_df["title_token_count"].min(),
        "max_title_tokens": metrics_df["title_token_count"].max(),
        "mean_TKEI": metrics_df["TKEI"].mean(),
        "median_TKEI": metrics_df["TKEI"].median(),
        "std_TKEI": metrics_df["TKEI"].std(),
        "min_TKEI": metrics_df["TKEI"].min(),
        "q1_TKEI": quartiles["q1_TKEI"],
        "q2_TKEI": quartiles["q2_TKEI"],
        "q3_TKEI": quartiles["q3_TKEI"],
        "max_TKEI": metrics_df["TKEI"].max(),
        "mean_TKOI": metrics_df["TKOI"].mean(),
        "median_TKOI": metrics_df["TKOI"].median(),
        "std_TKOI": metrics_df["TKOI"].std(),
        "min_TKOI": metrics_df["TKOI"].min(),
        "q1_TKOI": quartiles["q1_TKOI"],
        "q2_TKOI": quartiles["q2_TKOI"],
        "q3_TKOI": quartiles["q3_TKOI"],
        "max_TKOI": metrics_df["TKOI"].max(),
        "global_overlap_ratio": (
            metrics_df["overlap_token_count"].sum() / total_keyword_tokens
            if total_keyword_tokens else None
        ),
        "global_expansion_ratio": (
            metrics_df["expansion_token_count"].sum() / total_keyword_tokens
            if total_keyword_tokens else None
        ),
        "complete_title_replication_articles": (metrics_df["TKEI"] == 0).sum(),
        "complete_title_replication_percent": (
            (metrics_df["TKEI"] == 0).mean() * 100
        ),
        "complete_title_independence_articles": (metrics_df["TKOI"] == 0).sum(),
        "complete_title_independence_percent": (
            (metrics_df["TKOI"] == 0).mean() * 100
        ),
        "mean_coverage_ratio": metrics_df["coverage_ratio"].mean(),
        "median_coverage_ratio": metrics_df["coverage_ratio"].median()
    }

    return pd.DataFrame([summary])


def build_strategy_distribution(metrics_df):
    # Build ordered distributions for TKEI and TKOI quartile groups.
    tkei_distribution = (
        metrics_df["TKEI_quartile_group"]
        .value_counts()
        .rename_axis("group")
        .reset_index(name="count")
    )
    tkei_distribution["percent"] = tkei_distribution["count"] / len(metrics_df) * 100
    tkei_distribution["metric"] = "TKEI"

    tkei_group_order = [
        "low expansion",
        "moderate-low expansion",
        "moderate-high expansion",
        "high expansion",
        "complete title replication"
    ]

    tkei_distribution["group"] = pd.Categorical(
        tkei_distribution["group"],
        categories=tkei_group_order,
        ordered=True
    )
    tkei_distribution = tkei_distribution.sort_values("group").reset_index(
        drop=True
    )
    tkei_distribution["group"] = tkei_distribution["group"].astype(str)

    tkoi_distribution = (
        metrics_df["TKOI_quartile_group"]
        .value_counts()
        .rename_axis("group")
        .reset_index(name="count")
    )
    tkoi_distribution["percent"] = tkoi_distribution["count"] / len(metrics_df) * 100
    tkoi_distribution["metric"] = "TKOI"

    tkoi_group_order = [
        "low overlap",
        "moderate-low overlap",
        "moderate-high overlap",
        "high overlap",
        "complete title independence"
    ]

    tkoi_distribution["group"] = pd.Categorical(
        tkoi_distribution["group"],
        categories=tkoi_group_order,
        ordered=True
    )
    tkoi_distribution = tkoi_distribution.sort_values("group").reset_index(
        drop=True
    )
    tkoi_distribution["group"] = tkoi_distribution["group"].astype(str)

    return pd.concat([tkei_distribution, tkoi_distribution], ignore_index=True)


def build_frequency_tables(token_lists):
    # Build top-token frequency tables for all, overlapping and new tokens.
    top_keyword_tokens = pd.DataFrame(
        Counter(token_lists["all_keyword_tokens"]).most_common(50),
        columns=["token", "frequency"]
    )
    top_overlap_tokens = pd.DataFrame(
        Counter(token_lists["all_overlap_tokens"]).most_common(50),
        columns=["token", "frequency"]
    )
    top_expansion_tokens = pd.DataFrame(
        Counter(token_lists["all_expansion_tokens"]).most_common(50),
        columns=["token", "frequency"]
    )

    return top_keyword_tokens, top_overlap_tokens, top_expansion_tokens


def build_sensitivity_table(df):
    # Compare the main result across three preprocessing modes.
    #
    # Modes:
    # 1. stop words retained; no lemmatization;
    # 2. stop words removed; no lemmatization;
    # 3. stop words removed; WordNet lemmatization applied.
    sensitivity_modes = [
        {
            "mode": "original",
            "description": "stop words retained; no lemmatization",
            "remove_stopwords": False,
            "lemmatize": False
        },
        {
            "mode": "no_stopwords",
            "description": "stop words removed; no lemmatization",
            "remove_stopwords": True,
            "lemmatize": False
        },
        {
            "mode": "no_stopwords_lemmatized",
            "description": "stop words removed; WordNet lemmatization applied",
            "remove_stopwords": True,
            "lemmatize": True
        }
    ]

    lemmatizer = None
    if any(mode["lemmatize"] for mode in sensitivity_modes):
        lemmatizer = get_wordnet_lemmatizer()

    sensitivity_rows = []

    for mode in sensitivity_modes:
        mode_rows = []

        for _, row in df.iterrows():
            title_tokens = preprocess_tokens_for_sensitivity(
                row["title"],
                remove_stopwords=mode["remove_stopwords"],
                lemmatize=mode["lemmatize"],
                lemmatizer=lemmatizer
            )

            keyword_tokens = []
            for keyword_item in split_keywords(row["keywords"]):
                keyword_tokens.extend(
                    preprocess_tokens_for_sensitivity(
                        keyword_item,
                        remove_stopwords=mode["remove_stopwords"],
                        lemmatize=mode["lemmatize"],
                        lemmatizer=lemmatizer
                    )
                )

            title_set = set(title_tokens)
            overlap_tokens = [
                token for token in keyword_tokens if token in title_set
            ]
            expansion_tokens = [
                token for token in keyword_tokens if token not in title_set
            ]

            total_keyword_tokens = len(keyword_tokens)
            overlap_count = len(overlap_tokens)
            expansion_count = len(expansion_tokens)

            tkoi = (
                overlap_count / total_keyword_tokens
                if total_keyword_tokens else None
            )
            tkei = (
                expansion_count / total_keyword_tokens
                if total_keyword_tokens else None
            )

            mode_rows.append({
                "keyword_token_count": total_keyword_tokens,
                "overlap_token_count": overlap_count,
                "expansion_token_count": expansion_count,
                "TKOI": tkoi,
                "TKEI": tkei
            })

        mode_df = pd.DataFrame(mode_rows)
        mode_df = mode_df.dropna(subset=["TKEI", "TKOI"]).reset_index(drop=True)
        total_keyword_tokens = mode_df["keyword_token_count"].sum()

        sensitivity_rows.append({
            "mode": mode["mode"],
            "description": mode["description"],
            "total_articles": len(mode_df),
            "keyword_token_count": total_keyword_tokens,
            "overlap_token_count": mode_df["overlap_token_count"].sum(),
            "expansion_token_count": mode_df["expansion_token_count"].sum(),
            "global_overlap_ratio": (
                mode_df["overlap_token_count"].sum() / total_keyword_tokens
                if total_keyword_tokens else None
            ),
            "global_expansion_ratio": (
                mode_df["expansion_token_count"].sum() / total_keyword_tokens
                if total_keyword_tokens else None
            ),
            "mean_TKOI": mode_df["TKOI"].mean(),
            "mean_TKEI": mode_df["TKEI"].mean(),
            "median_TKOI": mode_df["TKOI"].median(),
            "median_TKEI": mode_df["TKEI"].median(),
            "complete_title_replication_articles": (
                mode_df["TKEI"] == 0
            ).sum(),
            "complete_title_replication_percent": (
                (mode_df["TKEI"] == 0).mean() * 100
            ),
            "complete_title_independence_articles": (
                mode_df["TKOI"] == 0
            ).sum(),
            "complete_title_independence_percent": (
                (mode_df["TKOI"] == 0).mean() * 100
            )
        })

    return pd.DataFrame(sensitivity_rows)


def load_input_data(input_path, sep, encoding):
    # Load metadata and check required columns.
    df = pd.read_csv(
        input_path,
        sep=sep,
        encoding=encoding,
        quotechar='"',
        engine="python"
    )

    df.columns = df.columns.str.strip().str.lower()

    required_columns = {"title", "keywords"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"The input file must contain the following columns: {missing_columns}"
        )

    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))

    return df


def save_outputs(
    output_dir,
    encoding,
    metrics_df,
    summary_df,
    strategy_distribution,
    top_keyword_tokens,
    top_overlap_tokens,
    top_expansion_tokens,
    sensitivity_df
):
    # Save all output tables as CSV files.
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "article_level_metrics": output_dir / "metadata_with_metrics.csv",
        "summary_metrics": output_dir / "summary_metrics.csv",
        "strategy_distribution": output_dir / "strategy_distribution.csv",
        "top_keyword_tokens": output_dir / "top_keyword_tokens.csv",
        "top_overlap_tokens": output_dir / "top_overlap_tokens.csv",
        "top_expansion_tokens": output_dir / "top_expansion_tokens.csv",
        "sensitivity_analysis": output_dir / "sensitivity_preprocessing_modes.csv"
    }

    metrics_df.to_csv(
        output_paths["article_level_metrics"],
        index=False,
        encoding=encoding
    )
    summary_df.to_csv(
        output_paths["summary_metrics"],
        index=False,
        encoding=encoding
    )
    strategy_distribution.to_csv(
        output_paths["strategy_distribution"],
        index=False,
        encoding=encoding
    )
    top_keyword_tokens.to_csv(
        output_paths["top_keyword_tokens"],
        index=False,
        encoding=encoding
    )
    top_overlap_tokens.to_csv(
        output_paths["top_overlap_tokens"],
        index=False,
        encoding=encoding
    )
    top_expansion_tokens.to_csv(
        output_paths["top_expansion_tokens"],
        index=False,
        encoding=encoding
    )
    sensitivity_df.to_csv(
        output_paths["sensitivity_analysis"],
        index=False,
        encoding=encoding
    )

    return output_paths


def main():
    # Run the full reproducible analysis.
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)

    df = load_input_data(input_path, sep=args.sep, encoding=args.encoding)

    metrics_df, token_lists = calculate_article_metrics(df)
    metrics_df, quartiles = add_quartile_groups(metrics_df)
    summary_df = build_summary(metrics_df, quartiles)
    strategy_distribution = build_strategy_distribution(metrics_df)
    (
        top_keyword_tokens,
        top_overlap_tokens,
        top_expansion_tokens
    ) = build_frequency_tables(token_lists)
    sensitivity_df = build_sensitivity_table(df)

    output_paths = save_outputs(
        output_dir=output_dir,
        encoding=args.encoding,
        metrics_df=metrics_df,
        summary_df=summary_df,
        strategy_distribution=strategy_distribution,
        top_keyword_tokens=top_keyword_tokens,
        top_overlap_tokens=top_overlap_tokens,
        top_expansion_tokens=top_expansion_tokens,
        sensitivity_df=sensitivity_df
    )

    print("Analysis completed successfully.")
    for label, path in output_paths.items():
        print(f"{label}: {path}")

    print("\nSummary metrics:")
    print(summary_df)

    print("\nStrategy distribution:")
    print(strategy_distribution)

    print("\nSensitivity analysis:")
    print(sensitivity_df)


if __name__ == "__main__":
    main()
