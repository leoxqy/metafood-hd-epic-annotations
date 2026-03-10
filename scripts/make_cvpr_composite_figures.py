from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="paper")


FIGURE_CAPTIONS = {
    "fig01_global_overview": (
        "(a) Profile counts by question type label. "
        "(b) Prevalence (%) of language-only/BBOX/numeric/temporal/spatial signals. "
        "(c) Share of rows with known vs unknown duration metadata. "
        "(d) Question count per task family. "
        "(e) Family-wise signal-rate heatmap (% within each family). "
        "(f) Distribution of answer index labels (`correct_idx`)."
    ),
    "fig02_numeric_bbox_locality": (
        "(a) Total numeric-signal detections by location (question/choices/metadata). "
        "(b) Total BBOX-signal detections by location. "
        "(c) Family-wise numeric location counts (question vs choices). "
        "(d) Family-wise BBOX location counts (question vs choices). "
        "(e) Top files ranked by BBOX-signal count. "
        "(f) File-level numeric counts: x=question, y=choices, marker size=question volume."
    ),
    "fig03_duration_analytics": (
        "(a) Histogram of clip duration with logarithmic x-axis. "
        "(b) Duration boxplots by family (log y-axis). "
        "(c) Duration violin plots for top profile groups (log y-axis). "
        "(d) File-level mean vs median duration scatter with y=x reference line. "
        "(e) Family-wise known-duration coverage rate (%). "
        "(f) Summed invalid time-range count by family."
    ),
    "fig04_input_answer_structure": (
        "(a) Distribution of `inputs_count`. "
        "(b) Distribution of `choices_count`. "
        "(c) Distribution of `clip_ranges_count`. "
        "(d) Mean question length (words) by family. "
        "(e) Question character length vs choice count scatter (sampled rows). "
        "(f) Family-wise span of parsed numeric `question_id` indices."
    ),
    "fig05_file_level_comparison": (
        "(a) Question count per source file. "
        "(b) File-level temporal vs spatial signal counts. "
        "(c) File-level numeric vs BBOX signal counts. "
        "(d) Known-duration rate (%) per file. "
        "(e) Language-only vs spatial counts for top-12 largest files. "
        "(f) Token-count proxy derived from semicolon-separated `top_profiles` per file."
    ),
    "fig06_cross_factor_matrix": (
        "(a) Raw count matrix: family × profile. "
        "(b) Row-normalized matrix (%): within-family profile composition. "
        "(c) Time-token rates by family and location (question vs choices). "
        "(d) Clip-duration availability rate (%) by family. "
        "(e) Highest-count language-only file-family groups. "
        "(f) Pearson correlation heatmap over structural and signal features."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CVPR-style composite figures from VQA analysis outputs.")
    parser.add_argument("--input-dir", type=Path, default=Path("analysis-output"), help="Directory containing analysis CSV/JSON outputs.")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis-output/figures"), help="Directory to write figure files.")
    parser.add_argument("--dpi", type=int, default=360, help="Figure DPI for export.")
    return parser.parse_args()


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def embed_caption(fig: plt.Figure, stem: str) -> None:
    caption = FIGURE_CAPTIONS.get(stem)
    if not caption:
        return
    wrapped = textwrap.fill(f"Interpretation guide: {caption}", width=190)
    fig.text(0.01, 0.01, wrapped, ha="left", va="bottom", fontsize=9)


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prep_data(input_dir: Path):
    pq = pd.read_csv(input_dir / "vqa_profiles_per_question.csv")
    by_file = pd.read_csv(input_dir / "vqa_summary_by_file.csv")
    by_family = pd.read_csv(input_dir / "vqa_summary_by_family.csv")
    summary = json.loads((input_dir / "vqa_summary.json").read_text(encoding="utf-8"))

    pq["duration_sec"] = pd.to_numeric(pq["clip_duration_sum_sec"], errors="coerce")
    pq["has_clip_duration"] = pq["has_clip_duration"].astype(bool)
    pq["question_words"] = pq["question"].fillna("").astype(str).str.split().str.len()
    pq["question_chars"] = pq["question"].fillna("").astype(str).str.len()

    qid_num = pq["question_id"].astype(str).str.extract(r"(\d+)$", expand=False)
    pq["question_index"] = pd.to_numeric(qid_num, errors="coerce")

    for c in [
        "choices_count",
        "correct_idx",
        "inputs_count",
        "clip_ranges_count",
        "invalid_time_ranges",
    ]:
        pq[c] = pd.to_numeric(pq[c], errors="coerce")

    for frame in (by_file, by_family):
        for c in frame.columns:
            if c.endswith("_count") or c.endswith("_sec"):
                frame[c] = pd.to_numeric(frame[c], errors="coerce")

    return pq, by_file, by_family, summary


def fig_01_dataset_overview(pq: pd.DataFrame, by_family: pd.DataFrame, summary: dict, out_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("Figure 1. Global dataset overview and signal composition", fontsize=15, fontweight="bold")

    # 1) Profile distribution
    profile_counts = pq["profile"].value_counts().sort_values(ascending=False)
    sns.barplot(x=profile_counts.values, y=profile_counts.index, ax=axes[0, 0], palette="viridis")
    axes[0, 0].set_title("Profile distribution")
    axes[0, 0].set_xlabel("Questions")
    axes[0, 0].set_ylabel("Profile")

    # 2) Boolean signal prevalence
    sig_cols = ["language_only", "bbox_signal", "numeric_signal", "temporal_signal", "spatial_signal"]
    sig_rate = pq[sig_cols].mean().mul(100).sort_values(ascending=False)
    sns.barplot(x=sig_rate.index, y=sig_rate.values, ax=axes[0, 1], palette="mako")
    axes[0, 1].set_title("Signal prevalence (%)")
    axes[0, 1].set_ylabel("Percent")
    axes[0, 1].tick_params(axis="x", rotation=20)

    # 3) Duration availability
    known = int(summary["overall"]["duration_known_count"])
    unknown = int(summary["overall"]["duration_unknown_count"])
    axes[0, 2].pie([known, unknown], labels=["Known", "Unknown"], autopct="%1.1f%%", colors=["#2a9d8f", "#e76f51"], startangle=90)
    axes[0, 2].set_title("Duration availability")

    # 4) Family question counts
    fam_counts = by_family.sort_values("question_count", ascending=False)
    sns.barplot(data=fam_counts, x="family", y="question_count", ax=axes[1, 0], palette="crest")
    axes[1, 0].set_title("Questions per family")
    axes[1, 0].set_xlabel("Family")
    axes[1, 0].set_ylabel("Questions")
    axes[1, 0].tick_params(axis="x", rotation=20)

    # 5) Family-wise signal heatmap (rates)
    rate_cols = ["language_only_count", "bbox_signal_count", "numeric_signal_count", "temporal_signal_count", "spatial_signal_count"]
    heat = by_family[["family", "question_count"] + rate_cols].copy()
    for c in rate_cols:
        heat[c] = 100 * heat[c] / heat["question_count"].replace(0, np.nan)
    heat = heat.set_index("family")[rate_cols]
    sns.heatmap(heat, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[1, 1], cbar_kws={"label": "%"})
    axes[1, 1].set_title("Family signal rates")

    # 6) Correct index distribution
    correct_hist = pq["correct_idx"].value_counts().sort_index()
    sns.barplot(x=correct_hist.index.astype(str), y=correct_hist.values, ax=axes[1, 2], palette="flare")
    axes[1, 2].set_title("Correct option index distribution")
    axes[1, 2].set_xlabel("correct_idx")
    axes[1, 2].set_ylabel("Questions")

    embed_caption(fig, "fig01_global_overview")
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    save_figure(fig, out_dir, "fig01_global_overview", dpi)


def fig_02_numeric_bbox_locality(pq: pd.DataFrame, by_family: pd.DataFrame, by_file: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("Figure 2. Numeric and BBOX localization diagnostics", fontsize=15, fontweight="bold")

    # 1) Overall numeric location
    num_loc = pd.Series(
        {
            "Question": pq["numeric_in_question"].sum(),
            "Choices": pq["numeric_in_choices"].sum(),
            "Metadata": pq["numeric_in_metadata"].sum(),
        }
    )
    sns.barplot(x=num_loc.index, y=num_loc.values, ax=axes[0, 0], palette="Blues_d")
    axes[0, 0].set_title("Numeric signal location (overall)")
    axes[0, 0].set_ylabel("Count")

    # 2) Overall BBOX location
    bbox_loc = pd.Series(
        {
            "Question": pq["bbox_in_question"].sum(),
            "Choices": pq["bbox_in_choices"].sum(),
            "Metadata": pq["bbox_in_metadata"].sum(),
        }
    )
    sns.barplot(x=bbox_loc.index, y=bbox_loc.values, ax=axes[0, 1], palette="Reds_d")
    axes[0, 1].set_title("BBOX signal location (overall)")
    axes[0, 1].set_ylabel("Count")

    # 3) Family numeric where
    fam_num = by_family[["family", "numeric_in_question_count", "numeric_in_choices_count"]].copy()
    fam_num = fam_num.melt(id_vars="family", var_name="where", value_name="count")
    sns.barplot(data=fam_num, x="family", y="count", hue="where", ax=axes[0, 2], palette="Set2")
    axes[0, 2].set_title("Numeric by family and location")
    axes[0, 2].tick_params(axis="x", rotation=20)
    axes[0, 2].legend(fontsize=8)

    # 4) Family BBOX where
    fam_bbox = by_family[["family", "bbox_in_question_count", "bbox_in_choices_count"]].copy()
    fam_bbox = fam_bbox.melt(id_vars="family", var_name="where", value_name="count")
    sns.barplot(data=fam_bbox, x="family", y="count", hue="where", ax=axes[1, 0], palette="Set1")
    axes[1, 0].set_title("BBOX by family and location")
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].legend(fontsize=8)

    # 5) File-level top bbox files
    top_bbox = by_file.sort_values("bbox_signal_count", ascending=False).head(12)
    sns.barplot(data=top_bbox, y="source_file", x="bbox_signal_count", ax=axes[1, 1], palette="rocket")
    axes[1, 1].set_title("Top files by BBOX count")
    axes[1, 1].set_xlabel("BBOX signal count")
    axes[1, 1].set_ylabel("File")

    # 6) Numeric question vs options scatter per file
    sns.scatterplot(
        data=by_file,
        x="numeric_in_question_count",
        y="numeric_in_choices_count",
        size="question_count",
        sizes=(40, 400),
        hue="question_count",
        palette="viridis",
        legend=False,
        ax=axes[1, 2],
    )
    axes[1, 2].set_title("File-level numeric signal (Q vs choices)")
    axes[1, 2].set_xlabel("Numeric in question")
    axes[1, 2].set_ylabel("Numeric in choices")

    embed_caption(fig, "fig02_numeric_bbox_locality")
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    save_figure(fig, out_dir, "fig02_numeric_bbox_locality", dpi)


def fig_03_duration_analytics(pq: pd.DataFrame, by_family: pd.DataFrame, by_file: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("Figure 3. Temporal and duration analytics", fontsize=15, fontweight="bold")

    dur = pq["duration_sec"].dropna()

    # 1) Duration histogram (log x)
    bins = np.geomspace(max(dur.min(), 1e-2), max(dur.max(), 1), 50) if len(dur) else np.linspace(0, 1, 10)
    axes[0, 0].hist(dur, bins=bins, color="#1f77b4", alpha=0.8)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_title("Duration distribution (log scale)")
    axes[0, 0].set_xlabel("Seconds (log)")
    axes[0, 0].set_ylabel("Count")

    # 2) Duration by family boxplot
    fam_d = pq.dropna(subset=["duration_sec"]).copy()
    order = fam_d.groupby("family")["duration_sec"].median().sort_values().index
    sns.boxplot(data=fam_d, x="family", y="duration_sec", order=order, ax=axes[0, 1], showfliers=False)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Duration by family (log y)")
    axes[0, 1].tick_params(axis="x", rotation=20)

    # 3) Duration by major profile
    top_profiles = pq["profile"].value_counts().head(6).index
    prof_d = pq[pq["profile"].isin(top_profiles) & pq["duration_sec"].notna()].copy()
    sns.violinplot(data=prof_d, x="profile", y="duration_sec", ax=axes[0, 2], inner="quartile", cut=0)
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("Duration by top profiles")
    axes[0, 2].tick_params(axis="x", rotation=25)

    # 4) File-level mean vs median duration
    dur_file = by_file.dropna(subset=["duration_mean_sec", "duration_median_sec"]).copy()
    sns.scatterplot(
        data=dur_file,
        x="duration_median_sec",
        y="duration_mean_sec",
        size="question_count",
        sizes=(50, 400),
        hue="question_count",
        palette="magma",
        legend=False,
        ax=axes[1, 0],
    )
    axes[1, 0].plot([0, max(dur_file["duration_median_sec"].max(), 1)], [0, max(dur_file["duration_median_sec"].max(), 1)], "--", color="gray", linewidth=1)
    axes[1, 0].set_title("File duration mean vs median")
    axes[1, 0].set_xlabel("Median duration (sec)")
    axes[1, 0].set_ylabel("Mean duration (sec)")

    # 5) Known-duration rate by family
    fam_known = by_family.copy()
    fam_known["known_rate"] = 100 * fam_known["duration_known_count"] / fam_known["question_count"]
    sns.barplot(data=fam_known.sort_values("known_rate", ascending=False), x="family", y="known_rate", ax=axes[1, 1], palette="cubehelix")
    axes[1, 1].set_title("Known duration rate by family")
    axes[1, 1].set_ylabel("% with known duration")
    axes[1, 1].tick_params(axis="x", rotation=20)

    # 6) Invalid range diagnostics by family
    inv = pq.groupby("family", as_index=False)["invalid_time_ranges"].sum().sort_values("invalid_time_ranges", ascending=False)
    sns.barplot(data=inv, x="family", y="invalid_time_ranges", ax=axes[1, 2], palette="pastel")
    axes[1, 2].set_title("Invalid time ranges by family")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].tick_params(axis="x", rotation=20)

    embed_caption(fig, "fig03_duration_analytics")
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    save_figure(fig, out_dir, "fig03_duration_analytics", dpi)


def fig_04_input_answer_structure(pq: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("Figure 4. Input, answer, and textual structure", fontsize=15, fontweight="bold")

    # 1) Inputs count distribution
    sns.histplot(pq["inputs_count"].dropna(), bins=20, ax=axes[0, 0], color="#4c78a8")
    axes[0, 0].set_title("inputs_count distribution")

    # 2) choices_count distribution
    sns.histplot(pq["choices_count"].dropna(), bins=20, ax=axes[0, 1], color="#f58518")
    axes[0, 1].set_title("choices_count distribution")

    # 3) clip_ranges_count distribution
    sns.histplot(pq["clip_ranges_count"].dropna(), bins=20, ax=axes[0, 2], color="#54a24b")
    axes[0, 2].set_title("clip_ranges_count distribution")

    # 4) Question word length by family
    fam_word = pq.groupby("family", as_index=False)["question_words"].mean().sort_values("question_words", ascending=False)
    sns.barplot(data=fam_word, x="family", y="question_words", ax=axes[1, 0], palette="viridis")
    axes[1, 0].set_title("Average question length (words)")
    axes[1, 0].tick_params(axis="x", rotation=20)

    # 5) Question char length vs options count
    sample = pq.sample(n=min(len(pq), 7000), random_state=42)
    sns.scatterplot(data=sample, x="question_chars", y="choices_count", hue="family", alpha=0.35, s=14, linewidth=0, ax=axes[1, 1], legend=False)
    axes[1, 1].set_title("Question chars vs choices_count")
    axes[1, 1].set_xlabel("Question chars")

    # 6) question_index coverage per family
    qidx = pq.dropna(subset=["question_index"]).groupby("family", as_index=False).agg(q_min=("question_index", "min"), q_max=("question_index", "max"), q_count=("question_index", "count"))
    qidx["span"] = qidx["q_max"] - qidx["q_min"]
    sns.barplot(data=qidx.sort_values("span", ascending=False), x="family", y="span", ax=axes[1, 2], palette="mako")
    axes[1, 2].set_title("question_id index span by family")
    axes[1, 2].tick_params(axis="x", rotation=20)

    embed_caption(fig, "fig04_input_answer_structure")
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    save_figure(fig, out_dir, "fig04_input_answer_structure", dpi)


def fig_05_file_level_comparison(by_file: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle("Figure 5. File-level comparative diagnostics", fontsize=15, fontweight="bold")

    file_order = by_file.sort_values("question_count", ascending=False)["source_file"]

    # 1) question counts
    sns.barplot(data=by_file.sort_values("question_count", ascending=False), y="source_file", x="question_count", ax=axes[0, 0], palette="crest")
    axes[0, 0].set_title("Question count by file")

    # 2) temporal vs spatial signal counts
    sns.scatterplot(data=by_file, x="temporal_signal_count", y="spatial_signal_count", size="question_count", sizes=(40, 400), hue="question_count", legend=False, palette="viridis", ax=axes[0, 1])
    axes[0, 1].set_title("Temporal vs spatial signals (file-level)")

    # 3) numeric vs bbox signal counts
    sns.scatterplot(data=by_file, x="numeric_signal_count", y="bbox_signal_count", size="question_count", sizes=(40, 400), hue="question_count", legend=False, palette="rocket", ax=axes[0, 2])
    axes[0, 2].set_title("Numeric vs BBOX signals (file-level)")

    # 4) duration known rate by file
    d = by_file.copy()
    d["duration_known_rate"] = 100 * d["duration_known_count"] / d["question_count"].replace(0, np.nan)
    sns.barplot(data=d.sort_values("duration_known_rate", ascending=False), y="source_file", x="duration_known_rate", ax=axes[1, 0], palette="flare")
    axes[1, 0].set_title("Duration known rate by file (%)")

    # 5) language-only and spatial counts
    counts = by_file[["source_file", "language_only_count", "spatial_signal_count"]].melt(id_vars="source_file", var_name="metric", value_name="count")
    topf = by_file.sort_values("question_count", ascending=False).head(12)["source_file"]
    counts = counts[counts["source_file"].isin(topf)]
    sns.barplot(data=counts, x="count", y="source_file", hue="metric", ax=axes[1, 1], palette="Set2")
    axes[1, 1].set_title("Language-only vs spatial (top 12 files)")

    # 6) top profile textual complexity proxy
    tp = by_file.copy()
    tp["top_profile_token_count"] = tp["top_profiles"].fillna("").astype(str).str.split(";").str.len()
    sns.barplot(data=tp.sort_values("top_profile_token_count", ascending=False), y="source_file", x="top_profile_token_count", ax=axes[1, 2], palette="coolwarm")
    axes[1, 2].set_title("Top-profile mixture complexity by file")

    embed_caption(fig, "fig05_file_level_comparison")
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    save_figure(fig, out_dir, "fig05_file_level_comparison", dpi)


def fig_06_profile_family_matrix(pq: pd.DataFrame, by_family: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("Figure 6. Cross-factor matrix and rare-case diagnostics", fontsize=15, fontweight="bold")

    # 1) profile x family counts
    mat = pd.crosstab(pq["family"], pq["profile"])
    sns.heatmap(mat, cmap="YlOrRd", ax=axes[0, 0])
    axes[0, 0].set_title("Profile x family counts")

    # 2) normalized profile x family
    matn = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0) * 100
    sns.heatmap(matn, cmap="PuBuGn", fmt=".1f", ax=axes[0, 1])
    axes[0, 1].set_title("Profile x family rates (%)")

    # 3) time token question vs choices by family
    t = pq.groupby("family", as_index=False).agg(
        time_q=("time_token_in_question", "mean"),
        time_c=("time_token_in_choices", "mean"),
    )
    tm = t.melt(id_vars="family", var_name="where", value_name="rate")
    tm["rate"] *= 100
    sns.barplot(data=tm, x="family", y="rate", hue="where", ax=axes[0, 2], palette="Set1")
    axes[0, 2].set_title("Time token rate by family")
    axes[0, 2].tick_params(axis="x", rotation=20)

    # 4) has_clip_duration by family
    clip_rate = pq.groupby("family", as_index=False)["has_clip_duration"].mean()
    clip_rate["has_clip_duration"] *= 100
    sns.barplot(data=clip_rate.sort_values("has_clip_duration", ascending=False), x="family", y="has_clip_duration", ax=axes[1, 0], palette="Set3")
    axes[1, 0].set_title("Clip-duration availability by family")
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].set_ylabel("%")

    # 5) language-only file/family rarity
    lo = pq[pq["language_only"]].groupby(["family", "source_file"], as_index=False).size().sort_values("size", ascending=False)
    if len(lo) == 0:
        axes[1, 1].text(0.5, 0.5, "No language-only rows", ha="center", va="center")
        axes[1, 1].set_axis_off()
    else:
        lo_top = lo.head(12)
        sns.barplot(data=lo_top, y="source_file", x="size", hue="family", dodge=False, ax=axes[1, 1], palette="tab10")
        axes[1, 1].set_title("Language-only rare cases")
        axes[1, 1].legend(fontsize=8)

    # 6) correlation matrix of numeric fields
    corr_cols = [
        "choices_count",
        "correct_idx",
        "inputs_count",
        "clip_ranges_count",
        "invalid_time_ranges",
        "duration_sec",
        "question_words",
        "question_chars",
        "question_index",
        "numeric_signal",
        "bbox_signal",
        "temporal_signal",
        "spatial_signal",
    ]
    corr_df = pq[corr_cols].copy()
    for c in ["numeric_signal", "bbox_signal", "temporal_signal", "spatial_signal"]:
        corr_df[c] = corr_df[c].astype(int)
    corr = corr_df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=axes[1, 2])
    axes[1, 2].set_title("Feature correlation matrix")

    embed_caption(fig, "fig06_cross_factor_matrix")
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    save_figure(fig, out_dir, "fig06_cross_factor_matrix", dpi)


def write_manifest(output_dir: Path) -> None:
    lines = [
        "# Composite Figure Manifest",
        "",
        "Generated figures:",
        "- fig01_global_overview.(png/pdf)",
        "- fig02_numeric_bbox_locality.(png/pdf)",
        "- fig03_duration_analytics.(png/pdf)",
        "- fig04_input_answer_structure.(png/pdf)",
        "- fig05_file_level_comparison.(png/pdf)",
        "- fig06_cross_factor_matrix.(png/pdf)",
        "",
        "All figures are exported at high DPI with publication-oriented layout and typography.",
    ]
    (output_dir / "FIGURE_MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")

    pq, by_file, by_family, summary = prep_data(input_dir)

    fig_01_dataset_overview(pq, by_family, summary, output_dir, args.dpi)
    fig_02_numeric_bbox_locality(pq, by_family, by_file, output_dir, args.dpi)
    fig_03_duration_analytics(pq, by_family, by_file, output_dir, args.dpi)
    fig_04_input_answer_structure(pq, output_dir, args.dpi)
    fig_05_file_level_comparison(by_file, output_dir, args.dpi)
    fig_06_profile_family_matrix(pq, by_family, output_dir, args.dpi)
    write_manifest(output_dir)

    print(f"Generated composite figures in: {output_dir}")


if __name__ == "__main__":
    main()
