from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

KNOWN_FAMILIES = [
    "3d_perception",
    "fine_grained",
    "gaze",
    "ingredient",
    "nutrition",
    "object_motion",
    "recipe",
]

SPATIAL_KEYWORDS = {
    "left",
    "right",
    "above",
    "below",
    "behind",
    "front",
    "beside",
    "between",
    "inside",
    "outside",
    "around",
    "near",
    "over",
    "under",
    "where",
    "location",
    "position",
    "coordinate",
    "counter",
    "cupboard",
    "drawer",
    "shelf",
    "windowsill",
}

NUMERIC_KEYWORDS = {
    "how many",
    "how much",
    "count",
    "number",
    "quantity",
    "weight",
    "mass",
    "calorie",
    "calories",
    "kcal",
    "gram",
    "grams",
    "ml",
    "l",
    "percent",
    "percentage",
    "total",
    "sum",
    "difference",
    "change",
}

BBOX_PATTERNS = [
    re.compile(r"\bbbox\b", re.IGNORECASE),
    re.compile(r"\bxyxy\b", re.IGNORECASE),
    re.compile(r"\bxywh\b", re.IGNORECASE),
    re.compile(r"<\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*>", re.IGNORECASE),
]

TIME_TOKEN_PATTERN = re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:\.\d+)?\b")
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?![A-Za-z])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze HD-EPIC VQA benchmark JSON files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("vqa-benchmark"),
        help="Directory containing VQA benchmark JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis-output"),
        help="Directory where analysis artifacts are written.",
    )
    return parser.parse_args()


def infer_family(filename_stem: str) -> str:
    for family in KNOWN_FAMILIES:
        if filename_stem.startswith(family):
            return family
    return filename_stem.split("_")[0]


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def parse_hhmmss(value: str) -> Optional[float]:
    if not value or not isinstance(value, str):
        return None
    raw = value.strip()
    parts = raw.split(":")
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


def has_bbox_signal(text: str) -> bool:
    for pattern in BBOX_PATTERNS:
        if pattern.search(text):
            return True
    return False


def has_numeric_signal(text: str) -> bool:
    lowered = text.lower()
    if NUMBER_PATTERN.search(text):
        return True
    return any(keyword in lowered for keyword in NUMERIC_KEYWORDS)


def has_temporal_signal(text: str) -> bool:
    lowered = text.lower()
    return bool(TIME_TOKEN_PATTERN.search(text)) or "time" in lowered or "before" in lowered or "after" in lowered or "during" in lowered


def has_spatial_signal(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in SPATIAL_KEYWORDS)


def flatten_choices(choices: Any) -> List[str]:
    if isinstance(choices, list):
        return [safe_text(item).strip() for item in choices]
    return []


def collect_clip_durations(inputs: Dict[str, Any]) -> Tuple[List[float], int]:
    durations: List[float] = []
    invalid_ranges = 0
    for _, input_obj in inputs.items():
        if not isinstance(input_obj, dict):
            continue
        start = parse_hhmmss(safe_text(input_obj.get("start_time")))
        end = parse_hhmmss(safe_text(input_obj.get("end_time")))
        if start is None or end is None:
            continue
        delta = end - start
        if delta >= 0:
            durations.append(delta)
        else:
            invalid_ranges += 1
    return durations, invalid_ranges


def compute_profile(flags: Dict[str, bool]) -> str:
    active = [k for k, v in flags.items() if v]
    if not active:
        return "language_only"
    if len(active) == 1:
        return active[0]
    return "mixed:" + "+".join(sorted(active))


def pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def summarize_numeric(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    schema_warnings: List[str] = []

    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise RuntimeError(f"No JSON files found in {input_dir}")

    for file_path in files:
        family = infer_family(file_path.stem)
        with file_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, dict):
            schema_warnings.append(f"{file_path.name}: top-level is not a JSON object")
            continue

        for question_id, payload in data.items():
            if not isinstance(payload, dict):
                schema_warnings.append(f"{file_path.name}:{question_id}: payload is not an object")
                continue

            inputs = payload.get("inputs") or {}
            if not isinstance(inputs, dict):
                schema_warnings.append(f"{file_path.name}:{question_id}: inputs is not an object")
                inputs = {}

            question_text = safe_text(payload.get("question")).strip()
            choices = flatten_choices(payload.get("choices"))
            choices_text = " ".join(choices)

            clip_durations, invalid_ranges = collect_clip_durations(inputs)
            clip_duration_sum = sum(clip_durations)
            has_clip_duration = len(clip_durations) > 0

            metadata_text = safe_text(payload.get("metadata"))
            others_text = safe_text(payload.get("others"))
            stat_text = safe_text(payload.get("stat"))
            misc_text = " ".join([metadata_text, others_text, stat_text]).strip()

            q_bbox = has_bbox_signal(question_text)
            c_bbox = has_bbox_signal(choices_text)
            m_bbox = has_bbox_signal(misc_text)

            q_numeric = has_numeric_signal(question_text)
            c_numeric = has_numeric_signal(choices_text)
            m_numeric = has_numeric_signal(misc_text)

            temporal_from_inputs = has_clip_duration
            temporal_from_text = has_temporal_signal(question_text) or has_temporal_signal(choices_text)
            spatial_from_text = has_spatial_signal(question_text) or has_spatial_signal(choices_text)

            signal_flags = {
                "bbox": q_bbox or c_bbox or m_bbox,
                "numeric": q_numeric or c_numeric or m_numeric,
                "temporal": temporal_from_inputs or temporal_from_text,
                "spatial": spatial_from_text,
            }

            profile = compute_profile(signal_flags)

            rows.append(
                {
                    "source_file": file_path.name,
                    "family": family,
                    "question_id": question_id,
                    "question": question_text,
                    "choices_count": len(choices),
                    "correct_idx": payload.get("correct_idx"),
                    "inputs_count": len(inputs),
                    "has_clip_duration": has_clip_duration,
                    "clip_duration_sum_sec": round(clip_duration_sum, 6) if has_clip_duration else None,
                    "clip_ranges_count": len(clip_durations),
                    "invalid_time_ranges": invalid_ranges,
                    "profile": profile,
                    "language_only": profile == "language_only",
                    "bbox_signal": signal_flags["bbox"],
                    "numeric_signal": signal_flags["numeric"],
                    "temporal_signal": signal_flags["temporal"],
                    "spatial_signal": signal_flags["spatial"],
                    "bbox_in_question": q_bbox,
                    "bbox_in_choices": c_bbox,
                    "bbox_in_metadata": m_bbox,
                    "numeric_in_question": q_numeric,
                    "numeric_in_choices": c_numeric,
                    "numeric_in_metadata": m_numeric,
                    "time_token_in_question": has_temporal_signal(question_text),
                    "time_token_in_choices": has_temporal_signal(choices_text),
                }
            )

    total_questions = len(rows)
    if total_questions == 0:
        raise RuntimeError("No question rows parsed from VQA files.")

    per_question_csv = output_dir / "vqa_profiles_per_question.csv"
    with per_question_csv.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_file = defaultdict(list)
    by_family = defaultdict(list)
    for row in rows:
        by_file[row["source_file"]].append(row)
        by_family[row["family"]].append(row)

    def build_group_summary(group_name: str, grouped: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        summary_rows: List[Dict[str, Any]] = []
        for key, subset in sorted(grouped.items(), key=lambda kv: kv[0]):
            durations = [r["clip_duration_sum_sec"] for r in subset if isinstance(r["clip_duration_sum_sec"], (int, float))]
            profile_counts = Counter(r["profile"] for r in subset)
            numeric_stats = summarize_numeric(durations)

            summary_rows.append(
                {
                    group_name: key,
                    "question_count": len(subset),
                    "language_only_count": sum(1 for r in subset if r["language_only"]),
                    "bbox_signal_count": sum(1 for r in subset if r["bbox_signal"]),
                    "numeric_signal_count": sum(1 for r in subset if r["numeric_signal"]),
                    "temporal_signal_count": sum(1 for r in subset if r["temporal_signal"]),
                    "spatial_signal_count": sum(1 for r in subset if r["spatial_signal"]),
                    "duration_known_count": numeric_stats["count"],
                    "duration_mean_sec": numeric_stats["mean"],
                    "duration_median_sec": numeric_stats["median"],
                    "duration_min_sec": numeric_stats["min"],
                    "duration_max_sec": numeric_stats["max"],
                    "top_profiles": "; ".join([f"{k}:{v}" for k, v in profile_counts.most_common(5)]),
                    "numeric_in_question_count": sum(1 for r in subset if r["numeric_in_question"]),
                    "numeric_in_choices_count": sum(1 for r in subset if r["numeric_in_choices"]),
                    "bbox_in_question_count": sum(1 for r in subset if r["bbox_in_question"]),
                    "bbox_in_choices_count": sum(1 for r in subset if r["bbox_in_choices"]),
                }
            )
        return summary_rows

    file_summary_rows = build_group_summary("source_file", by_file)
    family_summary_rows = build_group_summary("family", by_family)

    for filename, group_rows in [
        ("vqa_summary_by_file.csv", file_summary_rows),
        ("vqa_summary_by_family.csv", family_summary_rows),
    ]:
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(group_rows[0].keys()))
            writer.writeheader()
            writer.writerows(group_rows)

    addon_by_family = []
    for family, subset in sorted(by_family.items(), key=lambda kv: kv[0]):
        addon_by_family.append(
            {
                "family": family,
                "questions": len(subset),
                "numeric_any": sum(1 for r in subset if r["numeric_signal"]),
                "numeric_in_question": sum(1 for r in subset if r["numeric_in_question"]),
                "numeric_in_choices": sum(1 for r in subset if r["numeric_in_choices"]),
                "numeric_in_metadata": sum(1 for r in subset if r["numeric_in_metadata"]),
                "bbox_any": sum(1 for r in subset if r["bbox_signal"]),
                "bbox_in_question": sum(1 for r in subset if r["bbox_in_question"]),
                "bbox_in_choices": sum(1 for r in subset if r["bbox_in_choices"]),
                "bbox_in_metadata": sum(1 for r in subset if r["bbox_in_metadata"]),
            }
        )

    overall = {
        "files_analyzed": len(files),
        "questions_analyzed": total_questions,
        "language_only_count": sum(1 for r in rows if r["language_only"]),
        "language_only_pct": pct(sum(1 for r in rows if r["language_only"]), total_questions),
        "duration_known_count": sum(1 for r in rows if isinstance(r["clip_duration_sum_sec"], (int, float))),
        "duration_unknown_count": sum(1 for r in rows if not isinstance(r["clip_duration_sum_sec"], (int, float))),
        "numeric_any_count": sum(1 for r in rows if r["numeric_signal"]),
        "bbox_any_count": sum(1 for r in rows if r["bbox_signal"]),
        "numeric_in_question_count": sum(1 for r in rows if r["numeric_in_question"]),
        "numeric_in_choices_count": sum(1 for r in rows if r["numeric_in_choices"]),
        "bbox_in_question_count": sum(1 for r in rows if r["bbox_in_question"]),
        "bbox_in_choices_count": sum(1 for r in rows if r["bbox_in_choices"]),
        "profile_distribution": dict(Counter(r["profile"] for r in rows)),
        "schema_warnings": schema_warnings,
    }

    summary_json = {
        "overall": overall,
        "by_family": addon_by_family,
        "file_summary_rows": file_summary_rows,
        "family_summary_rows": family_summary_rows,
    }

    with (output_dir / "vqa_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary_json, fh, indent=2)

    report_lines: List[str] = []
    report_lines.append("# HD-EPIC VQA Benchmark Analysis Report")
    report_lines.append("")
    report_lines.append("## 1) Dataset Coverage")
    report_lines.append(f"- Files analyzed: {overall['files_analyzed']}")
    report_lines.append(f"- Questions analyzed: {overall['questions_analyzed']}")
    report_lines.append(f"- Language-only items: {overall['language_only_count']} ({overall['language_only_pct']:.2f}%)")
    report_lines.append(f"- Known clip durations: {overall['duration_known_count']}")
    report_lines.append(f"- Unknown clip durations: {overall['duration_unknown_count']}")
    report_lines.append("")

    report_lines.append("## 2) Signal Profiles (All Questions)")
    for profile, count in Counter(r["profile"] for r in rows).most_common():
        report_lines.append(f"- {profile}: {count} ({pct(count, total_questions):.2f}%)")
    report_lines.append("")

    report_lines.append("## 3) Addon: Numeric and BBOX Signals (Where they appear)")
    report_lines.append("### Overall")
    report_lines.append(f"- Numeric signal in any field: {overall['numeric_any_count']}")
    report_lines.append(f"- Numeric in question body: {overall['numeric_in_question_count']}")
    report_lines.append(f"- Numeric in options: {overall['numeric_in_choices_count']}")
    report_lines.append(f"- BBOX/coordinate signal in any field: {overall['bbox_any_count']}")
    report_lines.append(f"- BBOX in question body: {overall['bbox_in_question_count']}")
    report_lines.append(f"- BBOX in options: {overall['bbox_in_choices_count']}")
    report_lines.append("")

    report_lines.append("### By family")
    report_lines.append("| family | q_count | numeric_any | num_q | num_opt | bbox_any | bbox_q | bbox_opt |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in addon_by_family:
        report_lines.append(
            f"| {row['family']} | {row['questions']} | {row['numeric_any']} | {row['numeric_in_question']} | {row['numeric_in_choices']} | {row['bbox_any']} | {row['bbox_in_question']} | {row['bbox_in_choices']} |"
        )
    report_lines.append("")

    report_lines.append("## 4) Average Duration by Family (Known durations only)")
    report_lines.append("| family | q_count | duration_known | mean_sec | median_sec |")
    report_lines.append("|---|---:|---:|---:|---:|")
    for row in family_summary_rows:
        mean_val = "n/a" if row["duration_mean_sec"] is None else f"{row['duration_mean_sec']:.3f}"
        median_val = "n/a" if row["duration_median_sec"] is None else f"{row['duration_median_sec']:.3f}"
        report_lines.append(
            f"| {row['family']} | {row['question_count']} | {row['duration_known_count']} | {mean_val} | {median_val} |"
        )
    report_lines.append("")

    report_lines.append("## 5) Outputs")
    report_lines.append("- vqa_profiles_per_question.csv: row-level flattened table")
    report_lines.append("- vqa_summary_by_file.csv: aggregate stats by benchmark JSON file")
    report_lines.append("- vqa_summary_by_family.csv: aggregate stats by family")
    report_lines.append("- vqa_summary.json: machine-readable combined summary")
    report_lines.append("- vqa_full_report.md: this report")
    report_lines.append("")

    if schema_warnings:
        report_lines.append("## 6) Data / Schema Warnings")
        for warning in schema_warnings[:200]:
            report_lines.append(f"- {warning}")
        if len(schema_warnings) > 200:
            report_lines.append(f"- ... {len(schema_warnings) - 200} more warnings omitted")

    with (output_dir / "vqa_full_report.md").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines) + "\n")

    print(f"Analysis complete. Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()
