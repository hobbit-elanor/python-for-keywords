import pandas as pd
import re
from collections import Counter
from pathlib import Path

# ============================================================
# Supplementary reproducibility materials for the manuscript:
#
# This script contains the computational procedure used for preprocessing metadata, tokenization, calculation of quantitative indicators, quartile classification, and generation of statistical tables.
#
# The original manuscript corpus is not distributed because it contains confidential editorial materials.
#
# All file paths, dataset identifiers, and references to internal editorial infrastructure have been anonymized.
#
# Required input:
#   A CSV file containing at least two columns:
#       - title
#       - keywords
#
# Optional column:
#       - id
#
# The script is documented with explanatory comments to support computational reproducibility on independent metadata corpora.
# ============================================================


# ===== PATHS =====
# Replace this path with the location of your anonymised input CSV file.
input_csv = Path("input/anonymised_metadata.csv")

# Output files will be saved in the selected output directory.
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

output_metrics_csv = output_dir / "metadata_with_metrics.csv"
output_summary_csv = output_dir / "summary_metrics.csv"
output_strategy_csv = output_dir / "strategy_distribution.csv"
output_top_keywords_csv = output_dir / "top_keyword_tokens.csv"
output_top_overlap_csv = output_dir / "top_overlap_tokens.csv"
output_top_expansion_csv = output_dir / "top_expansion_tokens.csv"


# ===== LOAD DATA =====
# The CSV file is expected to use semicolon separators and UTF-8 encoding.
df = pd.read_csv(
    input_csv,
    sep=";",
    encoding="utf-8-sig",
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

# Add a numeric ID if the dataset does not already contain one.
if "id" not in df.columns:
    df.insert(0, "id", range(1, len(df) + 1))


# ===== TEXT PREPROCESSING FUNCTIONS =====
def tokenize(text):
    """
    Convert text into lowercase lexical tokens.

    Processing steps:
    - lowercase conversion;
    - removal of characters other than letters, digits, spaces and hyphens;
    - whitespace normalisation;
    - splitting into tokens;
    - removal of leading/trailing hyphens;
    - exclusion of one-character tokens.

    Stop words are not removed because the study measures formal lexical
    overlap between metadata fields.
    """
    if pd.isna(text):
        return []

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [t.strip("-") for t in tokens]
    tokens = [t for t in tokens if t and len(t) > 1]

    return tokens


def split_keywords(text):
    """
    Split the author-provided keyword field into individual keyword items.

    Keywords are split by comma or semicolon. The function preserves
    multi-word keyword expressions before tokenisation.
    """
    if pd.isna(text):
        return []

    parts = re.split(r"[;,]", str(text))
    parts = [re.sub(r"\s+", " ", p).strip(" .;,:") for p in parts]

    return [p for p in parts if p]


# ===== CALCULATE ARTICLE-LEVEL METRICS =====
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

    overlap_tokens = [t for t in keyword_tokens if t in title_set]
    expansion_tokens = [t for t in keyword_tokens if t not in title_set]

    total_keyword_tokens = len(keyword_tokens)
    overlap_count = len(overlap_tokens)
    expansion_count = len(expansion_tokens)

    # TKOI: Title Keyword Overlap Index
    tkoi = overlap_count / total_keyword_tokens if total_keyword_tokens else None

    # TKEI: Title Keyword Expansion Index
    tkei = expansion_count / total_keyword_tokens if total_keyword_tokens else None

    # Lexical coverage ratio:
    # number of unique keyword tokens divided by number of unique title tokens.
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

# Exclude rows without calculable TKEI/TKOI values.
metrics_df = metrics_df.dropna(subset=["TKEI", "TKOI"]).reset_index(drop=True)


# ===== QUARTILE-BASED CLASSIFICATION =====
q1_tkei = metrics_df["TKEI"].quantile(0.25)
q2_tkei = metrics_df["TKEI"].quantile(0.50)
q3_tkei = metrics_df["TKEI"].quantile(0.75)

q1_tkoi = metrics_df["TKOI"].quantile(0.25)
q2_tkoi = metrics_df["TKOI"].quantile(0.50)
q3_tkoi = metrics_df["TKOI"].quantile(0.75)


def classify_by_tkei(tkei):
    """
    Classify articles by the degree of keyword expansion relative to the title.
    Complete title replication is separated before quartile-based grouping.
    """
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
    """
    Classify articles by the degree of keyword overlap with the title.
    Complete title independence is separated before quartile-based grouping.
    """
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


metrics_df["TKEI_quartile_group"] = metrics_df["TKEI"].apply(classify_by_tkei)
metrics_df["TKOI_quartile_group"] = metrics_df["TKOI"].apply(classify_by_tkoi)


# ===== SUMMARY METRICS =====
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
    "q1_TKEI": q1_tkei,
    "q2_TKEI": q2_tkei,
    "q3_TKEI": q3_tkei,
    "max_TKEI": metrics_df["TKEI"].max(),

    "mean_TKOI": metrics_df["TKOI"].mean(),
    "median_TKOI": metrics_df["TKOI"].median(),
    "std_TKOI": metrics_df["TKOI"].std(),
    "min_TKOI": metrics_df["TKOI"].min(),
    "q1_TKOI": q1_tkoi,
    "q2_TKOI": q2_tkoi,
    "q3_TKOI": q3_tkoi,
    "max_TKOI": metrics_df["TKOI"].max(),

    # Corpus-level ratios are calculated from total token counts,
    # not as means of article-level TKEI/TKOI values.
    "global_overlap_ratio": (
        metrics_df["overlap_token_count"].sum()
        / metrics_df["keyword_token_count"].sum()
    ),

    "global_expansion_ratio": (
        metrics_df["expansion_token_count"].sum()
        / metrics_df["keyword_token_count"].sum()
    ),

    "complete_title_replication_articles": (metrics_df["TKEI"] == 0).sum(),
    "complete_title_replication_percent": (metrics_df["TKEI"] == 0).mean() * 100,

    "complete_title_independence_articles": (metrics_df["TKOI"] == 0).sum(),
    "complete_title_independence_percent": (metrics_df["TKOI"] == 0).mean() * 100,

    "mean_coverage_ratio": metrics_df["coverage_ratio"].mean(),
    "median_coverage_ratio": metrics_df["coverage_ratio"].median()
}

summary_df = pd.DataFrame([summary])


# ===== GROUP DISTRIBUTIONS =====
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
tkei_distribution = tkei_distribution.sort_values("group").reset_index(drop=True)
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
tkoi_distribution = tkoi_distribution.sort_values("group").reset_index(drop=True)
tkoi_distribution["group"] = tkoi_distribution["group"].astype(str)

strategy_distribution = pd.concat(
    [tkei_distribution, tkoi_distribution],
    ignore_index=True
)


# ===== FREQUENCY TABLES =====
# Frequencies are token frequencies across the entire corpus.
# Repeated occurrences of the same token are counted separately.
top_keyword_tokens = pd.DataFrame(
    Counter(all_keyword_tokens).most_common(50),
    columns=["token", "frequency"]
)

top_overlap_tokens = pd.DataFrame(
    Counter(all_overlap_tokens).most_common(50),
    columns=["token", "frequency"]
)

top_expansion_tokens = pd.DataFrame(
    Counter(all_expansion_tokens).most_common(50),
    columns=["token", "frequency"]
)


# ===== SAVE OUTPUTS =====
metrics_df.to_csv(output_metrics_csv, index=False, encoding="utf-8-sig")
summary_df.to_csv(output_summary_csv, index=False, encoding="utf-8-sig")
strategy_distribution.to_csv(output_strategy_csv, index=False, encoding="utf-8-sig")
top_keyword_tokens.to_csv(output_top_keywords_csv, index=False, encoding="utf-8-sig")
top_overlap_tokens.to_csv(output_top_overlap_csv, index=False, encoding="utf-8-sig")
top_expansion_tokens.to_csv(output_top_expansion_csv, index=False, encoding="utf-8-sig")


# ===== CONSOLE OUTPUT =====
print("Analysis completed successfully.")
print(f"Article-level metrics: {output_metrics_csv}")
print(f"Summary metrics: {output_summary_csv}")
print(f"Strategy distribution: {output_strategy_csv}")
print(f"Top keyword tokens: {output_top_keywords_csv}")
print(f"Top overlapping tokens: {output_top_overlap_csv}")
print(f"Top expansion tokens: {output_top_expansion_csv}")

print("\nSummary metrics:")
print(summary_df)

print("\nStrategy distribution:")
print(strategy_distribution)

print("\nFirst rows of article-level metrics:")
print(metrics_df.head())
