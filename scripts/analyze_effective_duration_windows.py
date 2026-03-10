from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")

TIME_TAG_PATTERN = re.compile(r"<\s*TIME\s+(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\b[^>]*>", re.IGNORECASE)
TIME_FALLBACK_PATTERN = re.compile(r"\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate effective VLM viewing duration per question using inputs and textual time spans."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("vqa-benchmark"), help="Directory containing VQA JSON files.")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis-output"), help="Directory for CSV/JSON summaries.")
    parser.add_argument("--fig-dir", type=Path, default=Path("analysis-output/figures"), help="Directory for figure export.")
    parser.add_argument("--dpi", type=int, default=360, help="Figure export DPI.")
    return parser.parse_args()


def parse_hhmmss(value: str) -> Optional[float]:
    parts = value.split(":")
    if len(parts) != 3:
        return None
    try:
        hh = int(parts[0])
        mm = int(parts[1])
        ss = float(parts[2])
    except ValueError:
        return None
    if mm < 0 or mm >= 60 or ss < 0:
        return None
    return hh * 3600 + mm * 60 + ss


def extract_times(text: str) -> List[float]:
    times: List[float] = []
    if not text:
        return times

    matched = TIME_TAG_PATTERN.findall(text)
    raw_values = matched if matched else TIME_FALLBACK_PATTERN.findall(text)
    for raw in raw_values:
        sec = parse_hhmmss(raw)
        if sec is not None:
            times.append(sec)
    return times


def spans_from_inputs(inputs: Any) -> List[float]:
    spans: List[float] = []
    if not isinstance(inputs, dict):
        return spans

    for value in inputs.values():
        if not isinstance(value, dict):
            continue
        start = parse_hhmmss(str(value.get("start_time", "")).strip())
        end = parse_hhmmss(str(value.get("end_time", "")).strip())
        if start is None or end is None:
            continue
        delta = end - start
        if delta >= 0:
            spans.append(delta)
    return spans


def span_from_text(text: str) -> Optional[float]:
    times = extract_times(text)
    if len(times) < 2:
        return None
    return max(times) - min(times)


def spans_from_choices(choices: Any) -> List[float]:
    spans: List[float] = []
    if not isinstance(choices, list):
        return spans

    for item in choices:
        text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        span = span_from_text(text)
        if span is not None and span >= 0:
            spans.append(span)
    return spans


def summarize(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "mean_sec": None,
            "median_sec": None,
            "p10_sec": None,
            "p25_sec": None,
            "p75_sec": None,
            "p90_sec": None,
            "min_sec": None,
            "max_sec": None,
        }

    arr = np.array(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean_sec": float(np.mean(arr)),
        "median_sec": float(np.median(arr)),
        "p10_sec": float(np.percentile(arr, 10)),
        "p25_sec": float(np.percentile(arr, 25)),
        "p75_sec": float(np.percentile(arr, 75)),
        "p90_sec": float(np.percentile(arr, 90)),
        "min_sec": float(np.min(arr)),
        "max_sec": float(np.max(arr)),
    }


def build_figure(df: pd.DataFrame, stats: pd.DataFrame, fig_dir: Path, dpi: int) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    order = stats.sort_values(["median_sec", "known_rate_pct"], ascending=[False, False])["source_file"].tolist()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(30, max(16, 0.6 * len(order))),
        gridspec_kw={"width_ratios": [2.8, 1.1]},
    )

    ax0, ax1 = axes

    known = df[df["effective_duration_sec"].notna() & (df["effective_duration_sec"] > 0)].copy()

    if len(known) > 0:
        sns.boxplot(
            data=known,
            y="source_file",
            x="effective_duration_sec",
            order=order,
            ax=ax0,
            color="#4C78A8",
            showfliers=False,
            linewidth=1,
        )
    ax0.set_xscale("log")
    ax0.set_title("Effective viewing duration by JSON type (log scale)", fontsize=18, fontweight="bold")
    ax0.set_xlabel("Effective duration per question (seconds, log)")
    ax0.set_ylabel("JSON type (source file)")

    # annotate no-duration rows
    no_duration_files = stats[stats["known_count"] == 0]["source_file"].tolist()
    for idx, source_file in enumerate(order):
        if source_file in no_duration_files:
            ax0.text(0.98, idx, "No duration", ha="right", va="center", transform=ax0.get_yaxis_transform(), color="#b22222", fontsize=9)

    source_order = ["inputs_start_end", "question_time_span", "choices_time_span", "none"]
    source_palette = {
        "inputs_start_end": "#1f77b4",
        "question_time_span": "#2ca02c",
        "choices_time_span": "#ff7f0e",
        "none": "#9e9e9e",
    }

    mix = (
        df.groupby(["source_file", "duration_source"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    mix["source_file"] = pd.Categorical(mix["source_file"], categories=order, ordered=True)
    pivot = mix.pivot(index="source_file", columns="duration_source", values="count").fillna(0)
    for c in source_order:
        if c not in pivot.columns:
            pivot[c] = 0
    pivot = pivot[source_order]
    row_sum = pivot.sum(axis=1).replace(0, np.nan)
    prop = pivot.div(row_sum, axis=0).fillna(0) * 100

    left = np.zeros(len(prop))
    y = np.arange(len(prop))
    for c in source_order:
        vals = prop[c].values
        ax1.barh(y, vals, left=left, color=source_palette[c], label=c)
        left += vals

    ax1.set_yticks(y)
    ax1.set_yticklabels(prop.index.tolist())
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Source composition per JSON type (%)")
    ax1.set_title("Where effective duration comes from", fontsize=18, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)

    fig.suptitle("VLM Effective Duration Coverage Across All VQA JSON Types", fontsize=23, fontweight="bold", y=0.995)

    caption = (
        "Effective duration per question is computed by priority: (1) sum of inputs start/end spans; "
        "(2) if missing, span inferred from TIME tags in question text (max-min); "
        "(3) if missing, median span across TIME-tagged choices; otherwise missing. "
        "Left shows per-type duration distribution (log scale). Right shows source mix of computed durations per type."
    )
    fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=11)

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    png_path = fig_dir / "big_effective_duration_by_json_type.png"
    pdf_path = fig_dir / "big_effective_duration_by_json_type.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    json_files = sorted(args.input_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in: {args.input_dir}")

    for file_path in json_files:
        with file_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, dict):
            continue

        for question_id, payload in data.items():
            if not isinstance(payload, dict):
                continue

            inputs = payload.get("inputs")
            question = payload.get("question", "")
            choices = payload.get("choices", [])

            input_spans = spans_from_inputs(inputs)
            input_duration = float(sum(input_spans)) if input_spans else None

            question_span = span_from_text(question if isinstance(question, str) else json.dumps(question, ensure_ascii=False))

            choice_spans = spans_from_choices(choices)
            choices_span_median = float(median(choice_spans)) if choice_spans else None

            if input_duration is not None and input_duration >= 0:
                effective_duration = input_duration
                source = "inputs_start_end"
            elif question_span is not None and question_span >= 0:
                effective_duration = question_span
                source = "question_time_span"
            elif choices_span_median is not None and choices_span_median >= 0:
                effective_duration = choices_span_median
                source = "choices_time_span"
            else:
                effective_duration = None
                source = "none"

            rows.append(
                {
                    "source_file": file_path.name,
                    "question_id": str(question_id),
                    "inputs_duration_sum_sec": input_duration,
                    "question_time_span_sec": question_span,
                    "choices_time_span_median_sec": choices_span_median,
                    "effective_duration_sec": effective_duration,
                    "duration_source": source,
                }
            )

    if not rows:
        raise RuntimeError("No rows parsed from JSON files")

    per_question_path = args.output_dir / "vlm_effective_duration_per_question.csv"
    with per_question_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["source_file"], []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for source_file in sorted(grouped.keys()):
        subset = grouped[source_file]
        vals = [
            r["effective_duration_sec"]
            for r in subset
            if isinstance(r["effective_duration_sec"], (int, float)) and r["effective_duration_sec"] > 0
        ]
        s = summarize(vals)

        total = len(subset)
        known_count = sum(1 for r in subset if isinstance(r["effective_duration_sec"], (int, float)))
        zero_duration_count = sum(1 for r in subset if isinstance(r["effective_duration_sec"], (int, float)) and r["effective_duration_sec"] == 0)
        source_counts = {
            "inputs_start_end": sum(1 for r in subset if r["duration_source"] == "inputs_start_end"),
            "question_time_span": sum(1 for r in subset if r["duration_source"] == "question_time_span"),
            "choices_time_span": sum(1 for r in subset if r["duration_source"] == "choices_time_span"),
            "none": sum(1 for r in subset if r["duration_source"] == "none"),
        }

        summary_rows.append(
            {
                "source_file": source_file,
                "question_count": total,
                "known_count": known_count,
                "known_rate_pct": round(100.0 * known_count / max(total, 1), 4),
                "mean_sec": s["mean_sec"],
                "median_sec": s["median_sec"],
                "p10_sec": s["p10_sec"],
                "p25_sec": s["p25_sec"],
                "p75_sec": s["p75_sec"],
                "p90_sec": s["p90_sec"],
                "min_sec": s["min_sec"],
                "max_sec": s["max_sec"],
                "source_inputs_start_end_count": source_counts["inputs_start_end"],
                "source_question_time_span_count": source_counts["question_time_span"],
                "source_choices_time_span_count": source_counts["choices_time_span"],
                "source_none_count": source_counts["none"],
                "zero_duration_count": zero_duration_count,
            }
        )

    summary_path = args.output_dir / "vlm_effective_duration_by_file.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    df = pd.DataFrame(rows)
    stats = pd.DataFrame(summary_rows)
    build_figure(df, stats, args.fig_dir, args.dpi)

    print(f"Saved: {per_question_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {args.fig_dir / 'big_effective_duration_by_json_type.png'}")
    print(f"Saved: {args.fig_dir / 'big_effective_duration_by_json_type.pdf'}")


if __name__ == "__main__":
    main()
