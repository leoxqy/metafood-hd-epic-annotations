# Leaderboard Difficulty Diagnostics

This report compares official family-level benchmark accuracy (6 participants) against family-level dataset diagnostics from `vqa_summary_by_family.csv`.

## Key caveat
- Correlations are computed over only 7 VQA families, so they are directional diagnostics rather than statistically robust claims.

## Hardest families by mean accuracy
- object_motion: 25.18%
- gaze: 27.96%
- 3d_perception: 33.66%
- nutrition: 35.78%
- fine_grained: 38.48%
- ingredient: 39.03%
- recipe: 47.75%

## Strongest feature correlations with accuracy (by |Pearson r|)
- bbox_density: Pearson r=-0.605248, Spearman rho=-0.668153 (n=7)
- spatial_density: Pearson r=-0.554777, Spearman rho=-0.714286 (n=7)
- duration_mean_sec: Pearson r=0.533733, Spearman rho=0.8 (n=4)

## Generated artifacts
- Radar chart: `analysis-output\figures\fig07_official_leaderboard_radar.png` and `analysis-output\figures\fig07_official_leaderboard_radar.pdf`
- Family accuracy table: `analysis-output\leaderboard_family_accuracy.csv`
- Correlation table: `analysis-output\leaderboard_accuracy_correlations.csv`
