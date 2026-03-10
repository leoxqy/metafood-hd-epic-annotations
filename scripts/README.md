# VQA Benchmark Analysis Scripts

## What this does

`analyze_vqa_benchmark.py` profiles all JSON files in `vqa-benchmark/` and generates:

- per-question profiling table
- per-file and per-family aggregate summaries
- duration statistics (using explicit `start_time`/`end_time` clip ranges)
- numeric and BBOX/coordinate signal counts, including **where** they appear (question body vs options vs metadata)
- markdown report for quick benchmark comparison

## Run

From repository root:

```powershell
python scripts/analyze_vqa_benchmark.py
```

For leaderboard-focused diagnostics (participant radar chart + family-level difficulty correlations):

```powershell
python scripts/analyze_leaderboard_vs_dataset.py
```

Optional arguments:

```powershell
python scripts/analyze_vqa_benchmark.py --input-dir vqa-benchmark --output-dir analysis-output
```

## Output files

Generated in `analysis-output/`:

- `vqa_profiles_per_question.csv`
- `vqa_summary_by_file.csv`
- `vqa_summary_by_family.csv`
- `vqa_summary.json`
- `vqa_full_report.md`

## Composite figures (CVPR-style)

Generate high-resolution composite figures with subplots:

```powershell
python scripts/make_cvpr_composite_figures.py --input-dir analysis-output --output-dir analysis-output/figures
```

Artifacts are saved as both PNG and PDF in `analysis-output/figures/`.

## Notes on language-only profile

A question is labeled `language_only` only when **none** of these signals are detected:

- temporal (explicit clip times or temporal markers)
- spatial/location cues
- numeric/quantitative cues
- BBOX/coordinate cues
