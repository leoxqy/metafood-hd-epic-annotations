from __future__ import annotations

import csv
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


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


def draw_team_radar(
    ax: plt.Axes,
    team_rows: Sequence[TeamRow],
    title: str,
    show_legend: bool,
    legend_anchor: Tuple[float, float] = (1.2, 1.1),
) -> None:
    labels = [item[0] for item in FAMILY_COLUMNS]
    angles = [2 * math.pi * index / len(labels) for index in range(len(labels))]
    angles += angles[:1]
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 70)
    ax.set_yticks([10, 20, 30, 40, 50, 60, 70])
    ax.set_yticklabels(["10", "20", "30", "40", "50", "60", "70"], fontsize=8)
    ax.grid(alpha=0.35)

    for row in team_rows:
        values = [getattr(row, attr_name) for _, attr_name, _ in FAMILY_COLUMNS]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.8, label=row.team_name)
        ax.fill(angles, values, alpha=0.08)

    ax.set_title(title, fontsize=10, pad=14)
    if show_legend:
        ax.legend(loc="upper right", bbox_to_anchor=legend_anchor, fontsize=7, frameon=True)


def radar_plot(output_png: Path, output_pdf: Path, team_rows: Sequence[TeamRow], title: str) -> None:
    fig = plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)
    draw_team_radar(ax, team_rows, title=title, show_legend=True, legend_anchor=(1.35, 1.1))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def leaderboard_composite_figure(
    output_png: Path,
    output_pdf: Path,
    family_accuracy_rows: Sequence[Dict[str, object]],
    corr_rows: Sequence[Dict[str, object]],
    family_summary: Dict[str, Dict[str, str]],
) -> None:
    fig = plt.figure(figsize=(18, 18))
    fig.suptitle(
        "Figure 8. Leaderboard-to-dataset difficulty diagnostics (official benchmark snapshot)",
        fontsize=16,
        fontweight="bold",
    )
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.28)

    family_display = [item[0] for item in FAMILY_COLUMNS]
    family_keys = [item[2] for item in FAMILY_COLUMNS]
    family_to_acc = {str(row["family"]): float(row["mean_accuracy"]) for row in family_accuracy_rows}

    # (a) Team overall ranking
    ax1 = fig.add_subplot(gs[0, 0])
    ordered_teams = sorted(LEADERBOARD_ROWS, key=lambda row: row.overall, reverse=True)
    team_names = [row.team_name for row in ordered_teams]
    team_scores = [row.overall for row in ordered_teams]
    bars = ax1.barh(team_names, team_scores, color="#5B8FF9")
    ax1.invert_yaxis()
    ax1.set_title("(a) Overall score by team")
    ax1.set_xlabel("Overall accuracy (%)")
    for bar, score in zip(bars, team_scores):
        ax1.text(score + 0.35, bar.get_y() + bar.get_height() / 2, f"{score:.2f}", va="center", fontsize=8)
    ax1.set_box_aspect(1)

    # (b) Family mean accuracy (descending)
    ax2 = fig.add_subplot(gs[0, 1])
    fam_sorted = sorted(family_accuracy_rows, key=lambda row: float(row["mean_accuracy"]), reverse=True)
    fam_labels = [str(row["family"]) for row in fam_sorted]
    fam_scores = [float(row["mean_accuracy"]) for row in fam_sorted]
    ax2.bar(fam_labels, fam_scores, color="#61DDAA")
    ax2.set_title("(b) Mean accuracy by family")
    ax2.set_ylabel("Mean accuracy (%)")
    ax2.tick_params(axis="x", rotation=25)
    ax2.set_box_aspect(1)

    # (c) Top correlations with accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    corr_valid = [row for row in corr_rows if row["pearson_r"] != ""]
    corr_top = sorted(corr_valid, key=lambda row: abs(float(row["pearson_r"])), reverse=True)[:6]
    corr_features = [str(row["feature"]) for row in corr_top]
    corr_values = [float(row["pearson_r"]) for row in corr_top]
    colors = ["#E8684A" if value < 0 else "#6DC8EC" for value in corr_values]
    ax3.barh(corr_features, corr_values, color=colors)
    ax3.axvline(0, color="black", linewidth=1)
    ax3.invert_yaxis()
    ax3.set_title("(c) Pearson correlation with accuracy")
    ax3.set_xlabel("Pearson r (left=negative, right=positive)")
    ax3.set_box_aspect(1)

    # (d) Accuracy vs top-5 feature diagnostics in one scatter (z-scored per feature)
    ax4 = fig.add_subplot(gs[1, 0])
    feature_extractors = {
        "bbox_density": lambda row: ratio(row, "bbox_signal_count", "question_count"),
        "spatial_density": lambda row: ratio(row, "spatial_signal_count", "question_count"),
        "numeric_density": lambda row: ratio(row, "numeric_signal_count", "question_count"),
        "temporal_density": lambda row: ratio(row, "temporal_signal_count", "question_count"),
        "duration_mean_sec": lambda row: safe_float(row.get("duration_mean_sec", "")),
    }
    scatter_specs: List[Tuple[str, str, str]] = [
        ("bbox_density", "BBOX density", "#5B8FF9"),
        ("spatial_density", "Spatial density", "#61DDAA"),
        ("numeric_density", "Numeric density", "#F6BD16"),
        ("temporal_density", "Temporal density", "#7262FD"),
        ("duration_mean_sec", "Duration mean (sec)", "#E8684A"),
    ]

    for feature_key, display_name, color in scatter_specs:
        extractor = feature_extractors[feature_key]
        x_raw: List[float] = []
        y_vals: List[float] = []
        for _, _, family_key in FAMILY_COLUMNS:
            summary_row = family_summary.get(family_key)
            if not summary_row or family_key not in family_to_acc:
                continue
            x_val = extractor(summary_row)
            if x_val is None:
                continue
            x_raw.append(x_val)
            y_vals.append(family_to_acc[family_key])

        if not x_raw:
            continue
        x_center = mean(x_raw)
        x_std = math.sqrt(variance(x_raw)) if len(x_raw) > 1 else 0
        if x_std == 0:
            x_z = [0.0 for _ in x_raw]
        else:
            x_z = [(value - x_center) / x_std for value in x_raw]

        ax4.scatter(x_z, y_vals, s=65, color=color, alpha=0.82, label=display_name)

    ax4.set_title("(d) Accuracy vs 5 feature diagnostics")
    ax4.set_xlabel("Feature z-score (normalized per feature)")
    ax4.set_ylabel("Mean accuracy (%)")
    ax4.legend(loc="upper right", fontsize=8, frameon=True)
    ax4.text(
        0.02,
        0.98,
        "z = (x - mean) / std\ncomputed within each feature",
        transform=ax4.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "#D9D9D9", "pad": 2.5},
    )
    ax4.grid(alpha=0.25)
    ax4.set_box_aspect(1)

    # (e) Accuracy vs BBOX density (raw-space scatter)
    ax5 = fig.add_subplot(gs[1, 1])
    bbox_x: List[float] = []
    bbox_y: List[float] = []
    bbox_labels: List[str] = []
    for _, _, family_key in FAMILY_COLUMNS:
        summary_row = family_summary.get(family_key)
        if not summary_row or family_key not in family_to_acc:
            continue
        density = ratio(summary_row, "bbox_signal_count", "question_count")
        if density is None:
            continue
        bbox_x.append(density)
        bbox_y.append(family_to_acc[family_key])
        bbox_labels.append(family_key)
    ax5.scatter(bbox_x, bbox_y, s=70, color="#5B8FF9", alpha=0.85)
    for x_val, y_val, label in zip(bbox_x, bbox_y, bbox_labels):
        ax5.annotate(label, (x_val, y_val), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax5.set_title("(e) Accuracy vs BBOX density")
    ax5.set_xlabel("bbox_density")
    ax5.set_ylabel("Mean accuracy (%)")
    ax5.grid(alpha=0.25)
    ax5.set_box_aspect(1)

    # (f) Accuracy vs spatial density (raw-space scatter)
    ax6 = fig.add_subplot(gs[1, 2])
    spatial_x: List[float] = []
    spatial_y: List[float] = []
    spatial_labels: List[str] = []
    for _, _, family_key in FAMILY_COLUMNS:
        summary_row = family_summary.get(family_key)
        if not summary_row or family_key not in family_to_acc:
            continue
        density = ratio(summary_row, "spatial_signal_count", "question_count")
        if density is None:
            continue
        spatial_x.append(density)
        spatial_y.append(family_to_acc[family_key])
        spatial_labels.append(family_key)
    ax6.scatter(spatial_x, spatial_y, s=70, color="#61DDAA", alpha=0.85)
    for x_val, y_val, label in zip(spatial_x, spatial_y, spatial_labels):
        ax6.annotate(label, (x_val, y_val), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax6.set_title("(f) Accuracy vs spatial density")
    ax6.set_xlabel("spatial_density")
    ax6.set_ylabel("Mean accuracy (%)")
    ax6.grid(alpha=0.25)
    ax6.set_box_aspect(1)

    # (g) Family radar of mean accuracy
    ax7 = fig.add_subplot(gs[2, 0], polar=True)
    radar_values = [family_to_acc.get(key, float("nan")) for key in family_keys]
    angles = np.linspace(0, 2 * math.pi, len(family_display), endpoint=False).tolist()
    angles_closed = angles + angles[:1]
    values_closed = radar_values + radar_values[:1]
    ax7.set_theta_offset(math.pi / 2)
    ax7.set_theta_direction(-1)
    ax7.plot(angles_closed, values_closed, linewidth=2, color="#F6BD16")
    ax7.fill(angles_closed, values_closed, alpha=0.2, color="#F6BD16")
    ax7.set_xticks(angles)
    ax7.set_xticklabels(family_display, fontsize=8)
    ax7.set_ylim(0, 70)
    ax7.set_title("(g) Family radar (mean accuracy)", fontsize=10)

    # (h) Top-group radar (DeepFrames + HelloWorld + Gemini Pro + LLaVA-Video)
    ax8 = fig.add_subplot(gs[2, 1], polar=True)
    top_group = [
        row
        for row in LEADERBOARD_ROWS
        if row.team_name
        in {
            "DeepFrames",
            "HelloWorld",
            "Gemini Pro [HD-EPIC Baseline]",
            "LLaVA-Video [HD-EPIC Baseline]",
        }
    ]
    draw_team_radar(
        ax8,
        top_group,
        title="(h) Radar: top group + strong baseline",
        show_legend=True,
        legend_anchor=(1.42, 1.05),
    )

    # (i) Baselines-only radar (excluding DeepFrames and HelloWorld)
    ax9 = fig.add_subplot(gs[2, 2], polar=True)
    baseline_group = [
        row
        for row in LEADERBOARD_ROWS
        if row.team_name not in {"DeepFrames", "HelloWorld"}
    ]
    draw_team_radar(
        ax9,
        baseline_group,
        title="(i) Radar: baselines only",
        show_legend=True,
        legend_anchor=(1.42, 1.05),
    )

    interpretation_text = (
        "Interpretation guide: (a-b) Team/family performance snapshots. "
        "(c) Pearson coefficients where sign gives direction (left<0 negative, right>0 positive) and |r| gives association strength "
        "(~<0.3 weak, 0.3-0.5 moderate, >0.5 strong). "
        "(d) Overlayed scatter of five diagnostics against accuracy using per-feature z-score normalization, where "
        "z=(x-mean(feature))/std(feature) across families, so 0 is family-average and ±1 is one standard deviation. "
        "(e-f) Raw-space scatter for two strongest negative factors. Definitions: bbox_density=bbox_signal_count/question_count; "
        "spatial_density=spatial_signal_count/question_count; density means the fraction of family questions containing the signal. "
        "(g-i) Family, top-group, and baselines-only radars. Caveat: n=7 families, interpret directionally."
    )
    fig.text(
        0.5,
        0.016,
        textwrap.fill(interpretation_text, width=210),
        ha="center",
        va="bottom",
        fontsize=11,
        wrap=True,
    )
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=320, bbox_inches="tight")
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
    radar_top_png = figure_dir / "fig07a_official_leaderboard_radar_top_group.png"
    radar_top_pdf = figure_dir / "fig07a_official_leaderboard_radar_top_group.pdf"
    radar_baseline_png = figure_dir / "fig07b_official_leaderboard_radar_baselines_only.png"
    radar_baseline_pdf = figure_dir / "fig07b_official_leaderboard_radar_baselines_only.pdf"
    composite_png = figure_dir / "fig08_leaderboard_difficulty_composite.png"
    composite_pdf = figure_dir / "fig08_leaderboard_difficulty_composite.pdf"
    family_accuracy_csv = root / "analysis-output" / "leaderboard_family_accuracy.csv"
    correlation_csv = root / "analysis-output" / "leaderboard_accuracy_correlations.csv"
    report_md = root / "analysis-output" / "leaderboard_difficulty_report.md"

    family_summary = load_family_summary(family_summary_path)
    top_group = [
        row
        for row in LEADERBOARD_ROWS
        if row.team_name
        in {
            "DeepFrames",
            "HelloWorld",
            "Gemini Pro [HD-EPIC Baseline]",
            "LLaVA-Video [HD-EPIC Baseline]",
        }
    ]
    baseline_group = [row for row in LEADERBOARD_ROWS if row.team_name not in {"DeepFrames", "HelloWorld"}]
    radar_plot(
        radar_top_png,
        radar_top_pdf,
        top_group,
        "HD-EPIC Official Benchmark: Radar (DeepFrames, HelloWorld, Gemini Pro, LLaVA-Video)",
    )
    radar_plot(
        radar_baseline_png,
        radar_baseline_pdf,
        baseline_group,
        "HD-EPIC Official Benchmark: Radar (Baselines excluding DeepFrames/HelloWorld)",
    )

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

    leaderboard_composite_figure(
        composite_png,
        composite_pdf,
        family_accuracy_rows,
        corr_rows,
        family_summary,
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
            "## Feature definitions used in Figure 8",
            "- `bbox_density = bbox_signal_count / question_count`",
            "- `spatial_density = spatial_signal_count / question_count`",
            "- Panel (d) x-axis uses per-feature z-score: `z = (x - mean(feature)) / std(feature)` computed across families.",
            "",
            "## Composite figure preview",
            f"![Figure 8 — Leaderboard-to-dataset difficulty diagnostics](figures/{composite_png.name})",
            "",
            "## Generated artifacts",
            f"- Radar chart (top group): `{radar_top_png.relative_to(root)}` and `{radar_top_pdf.relative_to(root)}`",
            f"- Radar chart (baselines only): `{radar_baseline_png.relative_to(root)}` and `{radar_baseline_pdf.relative_to(root)}`",
            f"- Family accuracy table: `{family_accuracy_csv.relative_to(root)}`",
            f"- Correlation table: `{correlation_csv.relative_to(root)}`",
            f"- Composite figure: `{composite_png.relative_to(root)}` and `{composite_pdf.relative_to(root)}`",
        ]
    )

    report_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[ok] Wrote {radar_top_png}")
    print(f"[ok] Wrote {radar_top_pdf}")
    print(f"[ok] Wrote {radar_baseline_png}")
    print(f"[ok] Wrote {radar_baseline_pdf}")
    print(f"[ok] Wrote {family_accuracy_csv}")
    print(f"[ok] Wrote {correlation_csv}")
    print(f"[ok] Wrote {report_md}")
    print(f"[ok] Wrote {composite_png}")
    print(f"[ok] Wrote {composite_pdf}")


def ratio(row: Dict[str, str], numerator_key: str, denominator_key: str) -> Optional[float]:
    numerator = safe_float(row.get(numerator_key, ""))
    denominator = safe_float(row.get(denominator_key, ""))
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


if __name__ == "__main__":
    main()
