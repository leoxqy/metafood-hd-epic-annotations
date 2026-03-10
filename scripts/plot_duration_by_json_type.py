from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a large visualization of question duration by VQA JSON type (source file)."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("analysis-output/vqa_profiles_per_question.csv"),
        help="Per-question analysis CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis-output/figures"),
        help="Directory to write outputs.",
    )
    parser.add_argument("--dpi", type=int, default=360, help="Export DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    required = {"source_file", "clip_duration_sum_sec", "has_clip_duration"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["duration_sec"] = pd.to_numeric(df["clip_duration_sum_sec"], errors="coerce")
    df["has_clip_duration"] = df["has_clip_duration"].astype(bool)

    all_types = sorted(df["source_file"].dropna().astype(str).unique())

    stats = (
        df.groupby("source_file", as_index=False)
        .agg(
            total_questions=("source_file", "size"),
            known_duration_count=("has_clip_duration", "sum"),
        )
        .assign(known_rate=lambda x: 100.0 * x["known_duration_count"] / x["total_questions"])
    )

    known = df[df["has_clip_duration"] & df["duration_sec"].notna() & (df["duration_sec"] > 0)].copy()

    med = (
        known.groupby("source_file", as_index=False)["duration_sec"]
        .median()
        .rename(columns={"duration_sec": "median_duration_sec"})
    )

    stats = stats.merge(med, on="source_file", how="left")

    order = (
        stats.sort_values(["median_duration_sec", "known_rate"], ascending=[False, False])["source_file"]
        .fillna("")
        .tolist()
    )

    if not order:
        order = all_types

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(28, max(16, 0.55 * len(all_types))),
        gridspec_kw={"width_ratios": [2.6, 1.2]},
    )

    ax0, ax1 = axes

    if len(known) > 0:
        sns.boxplot(
            data=known,
            y="source_file",
            x="duration_sec",
            order=order,
            ax=ax0,
            color="#4C78A8",
            showfliers=False,
            linewidth=1,
        )
    ax0.set_xscale("log")
    ax0.set_title("Question Duration Distribution by JSON Type (log scale)", fontsize=18, fontweight="bold")
    ax0.set_xlabel("Duration per question (seconds, log)")
    ax0.set_ylabel("JSON type (source file)")

    sns.barplot(
        data=stats,
        y="source_file",
        x="known_rate",
        order=order,
        ax=ax1,
        color="#F58518",
    )
    ax1.set_xlim(0, 100)
    ax1.set_title("Duration availability", fontsize=18, fontweight="bold")
    ax1.set_xlabel("Questions with known duration (%)")
    ax1.set_ylabel("")

    label_map = stats.set_index("source_file")["total_questions"].to_dict()
    for idx, source_file in enumerate(order):
        count = int(label_map.get(source_file, 0))
        ax1.text(
            101,
            idx,
            f"n={count}",
            va="center",
            fontsize=9,
        )

    fig.suptitle(
        "Large Visualization: Duration Signals for All VQA JSON Types",
        fontsize=22,
        fontweight="bold",
        y=0.995,
    )

    caption = (
        "Left: per-type distribution of question duration (clip_duration_sum_sec) across all JSON task files; "
        "box = IQR, center line = median, whiskers = spread (outliers hidden), x-axis in log seconds. "
        "Right: percentage of questions with available duration metadata for each JSON type; "
        "labels show total question count per type."
    )
    fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=11)

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    png_path = args.output_dir / "big_duration_by_json_type.png"
    pdf_path = args.output_dir / "big_duration_by_json_type.pdf"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
