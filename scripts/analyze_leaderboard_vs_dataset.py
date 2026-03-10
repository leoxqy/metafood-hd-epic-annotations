from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


@dataclass
class TeamRow:
    team_name: str
    overall: float
    recipe: float
    ingredient: float
    nutrition: float
    action: float
    three_d: float
    motion: float
    gaze: float


LEADERBOARD_ROWS: List[TeamRow] = [
    TeamRow("DeepFrames", 44.21, 64.62, 54.67, 38.00, 48.30, 42.59, 30.17, 31.15),
    TeamRow("HelloWorld", 41.55, 64.75, 43.33, 37.00, 42.03, 40.88, 29.90, 32.95),
    TeamRow("Gemini Pro [HD-EPIC Baseline]", 37.57, 60.50, 46.17, 34.67, 39.63, 32.51, 20.83, 28.65),
    TeamRow("LLaVA-Video [HD-EPIC Baseline]", 32.43, 36.25, 33.50, 38.67, 43.02, 27.31, 18.93, 29.30),
    TeamRow("LongVA [HD-EPIC Baseline]", 29.28, 29.62, 30.83, 33.67, 30.68, 32.91, 22.73, 24.50),
    TeamRow("VideoLLaMA 2 [HD-EPIC Baseline]", 27.39, 30.75, 25.67, 32.67, 27.24, 25.74, 28.50, 21.20),
]

FAMILY_COLUMNS: List[Tuple[str, str, str]] = [
    ("Recipe", "recipe", "recipe"),
    ("Ingredient", "ingredient", "ingredient"),
    ("Nutrition", "nutrition", "nutrition"),
    ("Action", "action", "fine_grained"),
    ("3D", "three_d", "3d_perception"),
    ("Motion", "motion", "object_motion"),
    ("Gaze", "gaze", "gaze"),
]


def safe_float(value: str) -> Optional[float]:
    text = (value or "").strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_family_summary(csv_path: Path) -> Dict[str, Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row["family"]: row for row in reader}


def radar_plot(output_png: Path, output_pdf: Path) -> None:
    labels = [item[0] for item in FAMILY_COLUMNS]
    angles = [2 * math.pi * index / len(labels) for index in range(len(labels))]
    angles += angles[:1]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 70)
    ax.set_yticks([10, 20, 30, 40, 50, 60, 70])
    ax.set_yticklabels(["10", "20", "30", "40", "50", "60", "70"], fontsize=9)
    ax.grid(alpha=0.35)

    for row in LEADERBOARD_ROWS:
        values = [getattr(row, attr_name) for _, attr_name, _ in FAMILY_COLUMNS]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row.team_name)
        ax.fill(angles, values, alpha=0.08)

    ax.set_title("HD-EPIC Official Benchmark: Team Performance by VQA Family", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=8)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def variance(values: Sequence[float]) -> float:
    center = mean(values)
    return sum((value - center) ** 2 for value in values) / len(values)


def pearson(x_values: Sequence[float], y_values: Sequence[float]) -> Optional[float]:
    if len(x_values) < 3 or len(x_values) != len(y_values):
        return None
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator = math.sqrt(sum((x - x_mean) ** 2 for x in x_values) * sum((y - y_mean) ** 2 for y in y_values))
    if denominator == 0:
        return None
    return numerator / denominator


def rank(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    output = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        next_index = index + 1
        while next_index < len(indexed) and indexed[next_index][1] == indexed[index][1]:
            next_index += 1
        avg_rank = (index + 1 + next_index) / 2
        for position in range(index, next_index):
            output[indexed[position][0]] = avg_rank
        index = next_index
    return output


def spearman(x_values: Sequence[float], y_values: Sequence[float]) -> Optional[float]:
    if len(x_values) < 3 or len(x_values) != len(y_values):
        return None
    return pearson(rank(x_values), rank(y_values))


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    family_summary_path = root / "analysis-output" / "vqa_summary_by_family.csv"
    figure_dir = root / "analysis-output" / "figures"
    radar_png = figure_dir / "fig07_official_leaderboard_radar.png"
    radar_pdf = figure_dir / "fig07_official_leaderboard_radar.pdf"
    family_accuracy_csv = root / "analysis-output" / "leaderboard_family_accuracy.csv"
    correlation_csv = root / "analysis-output" / "leaderboard_accuracy_correlations.csv"
    report_md = root / "analysis-output" / "leaderboard_difficulty_report.md"

    family_summary = load_family_summary(family_summary_path)
    radar_plot(radar_png, radar_pdf)

    family_accuracy_rows: List[Dict[str, object]] = []
    family_accuracy_map: Dict[str, float] = {}

    for label, attr_name, family_key in FAMILY_COLUMNS:
        values = [getattr(row, attr_name) for row in LEADERBOARD_ROWS]
        avg_acc = mean(values)
        family_accuracy_map[family_key] = avg_acc
        family_accuracy_rows.append(
            {
                "column": label,
                "family": family_key,
                "mean_accuracy": round(avg_acc, 4),
                "std_proxy": round(math.sqrt(variance(values)), 4),
                "min_accuracy": round(min(values), 4),
                "max_accuracy": round(max(values), 4),
            }
        )

    write_csv(
        family_accuracy_csv,
        ["column", "family", "mean_accuracy", "std_proxy", "min_accuracy", "max_accuracy"],
        family_accuracy_rows,
    )

    feature_extractors = {
        "question_count": lambda row: safe_float(row.get("question_count", "")),
        "bbox_density": lambda row: ratio(row, "bbox_signal_count", "question_count"),
        "numeric_density": lambda row: ratio(row, "numeric_signal_count", "question_count"),
        "temporal_density": lambda row: ratio(row, "temporal_signal_count", "question_count"),
        "spatial_density": lambda row: ratio(row, "spatial_signal_count", "question_count"),
        "duration_known_rate": lambda row: ratio(row, "duration_known_count", "question_count"),
        "duration_mean_sec": lambda row: safe_float(row.get("duration_mean_sec", "")),
        "duration_median_sec": lambda row: safe_float(row.get("duration_median_sec", "")),
    }

    corr_rows: List[Dict[str, object]] = []
    for feature_name, extractor in feature_extractors.items():
        x_values: List[float] = []
        y_values: List[float] = []
        used_families: List[str] = []
        for _, _, family_key in FAMILY_COLUMNS:
            if family_key not in family_summary or family_key not in family_accuracy_map:
                continue
            x_val = extractor(family_summary[family_key])
            if x_val is None:
                continue
            x_values.append(x_val)
            y_values.append(family_accuracy_map[family_key])
            used_families.append(family_key)

        p_val = pearson(x_values, y_values)
        s_val = spearman(x_values, y_values)
        corr_rows.append(
            {
                "feature": feature_name,
                "n_families": len(x_values),
                "pearson_r": round(p_val, 6) if p_val is not None else "",
                "spearman_rho": round(s_val, 6) if s_val is not None else "",
                "mean_feature_value": round(mean(x_values), 6) if x_values else "",
                "families_used": ";".join(used_families),
            }
        )

    corr_rows.sort(key=lambda row: abs(float(row["pearson_r"])) if row["pearson_r"] != "" else -1, reverse=True)
    write_csv(
        correlation_csv,
        ["feature", "n_families", "pearson_r", "spearman_rho", "mean_feature_value", "families_used"],
        corr_rows,
    )

    strongest = [row for row in corr_rows if row["pearson_r"] != ""][:3]
    hardest = sorted(family_accuracy_rows, key=lambda row: float(row["mean_accuracy"]))

    report_lines = [
        "# Leaderboard Difficulty Diagnostics",
        "",
        "This report compares official family-level benchmark accuracy (6 participants) against family-level dataset diagnostics from `vqa_summary_by_family.csv`.",
        "",
        "## Key caveat",
        "- Correlations are computed over only 7 VQA families, so they are directional diagnostics rather than statistically robust claims.",
        "",
        "## Hardest families by mean accuracy",
    ]

    for row in hardest:
        report_lines.append(f"- {row['family']}: {float(row['mean_accuracy']):.2f}%")

    report_lines.extend(["", "## Strongest feature correlations with accuracy (by |Pearson r|)"])
    for row in strongest:
        report_lines.append(
            f"- {row['feature']}: Pearson r={row['pearson_r']}, Spearman rho={row['spearman_rho']} (n={row['n_families']})"
        )

    report_lines.extend(
        [
            "",
            "## Generated artifacts",
            f"- Radar chart: `{radar_png.relative_to(root)}` and `{radar_pdf.relative_to(root)}`",
            f"- Family accuracy table: `{family_accuracy_csv.relative_to(root)}`",
            f"- Correlation table: `{correlation_csv.relative_to(root)}`",
        ]
    )

    report_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[ok] Wrote {radar_png}")
    print(f"[ok] Wrote {radar_pdf}")
    print(f"[ok] Wrote {family_accuracy_csv}")
    print(f"[ok] Wrote {correlation_csv}")
    print(f"[ok] Wrote {report_md}")


def ratio(row: Dict[str, str], numerator_key: str, denominator_key: str) -> Optional[float]:
    numerator = safe_float(row.get(numerator_key, ""))
    denominator = safe_float(row.get(denominator_key, ""))
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


if __name__ == "__main__":
    main()
