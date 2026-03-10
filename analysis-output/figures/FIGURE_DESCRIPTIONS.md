# Figure Descriptions (Publication-Style, Interpretation-Only)

This document provides neutral, paper-ready descriptions for each generated PDF figure and all subplots. It explains what each panel visualizes and how to interpret axes, scales, and encodings, without drawing empirical conclusions.

## Figure 1 — Global dataset overview and signal composition
File: `fig01_global_overview.pdf`

**(a) Profile distribution**
- Type: horizontal bar chart.
- Encoding: y-axis lists profile labels; x-axis is number of questions.
- Interpretation: compare bar lengths to read relative frequency of profile categories.

**(b) Signal prevalence (%)**
- Type: bar chart.
- Encoding: x-axis is boolean signal type; y-axis is percentage of questions where the signal is true.
- Interpretation: each bar is an overall prevalence rate for one signal; values are directly comparable across signals.

**(c) Duration availability**
- Type: pie chart.
- Encoding: slices represent counts of questions with known vs unknown duration metadata.
- Interpretation: slice area (and percentage labels) indicates share of total questions in each availability state.

**(d) Questions per family**
- Type: bar chart.
- Encoding: x-axis is family; y-axis is question count.
- Interpretation: bar height gives dataset volume per family.

**(e) Family signal rates**
- Type: heatmap with annotations.
- Encoding: rows are families; columns are signal categories; cell values are within-family percentages.
- Interpretation: read each cell as the rate of that signal within that family; use color intensity and numeric text together.

**(f) Correct option index distribution**
- Type: bar chart.
- Encoding: x-axis is `correct_idx`; y-axis is number of questions.
- Interpretation: bar heights show how often each answer index appears as the labeled correct option.

---

## Figure 2 — Numeric and BBOX localization diagnostics
File: `fig02_numeric_bbox_locality.pdf`

**(a) Numeric signal location (overall)**
- Type: bar chart.
- Encoding: categories are Question / Choices / Metadata; y-axis is count.
- Interpretation: each bar counts rows where numeric cues are detected in that location type.

**(b) BBOX signal location (overall)**
- Type: bar chart.
- Encoding: categories are Question / Choices / Metadata; y-axis is count.
- Interpretation: same reading rule as (a), but for BBOX-related cues.

**(c) Numeric by family and location**
- Type: grouped bar chart.
- Encoding: x-axis is family; hue separates location (`numeric_in_question_count`, `numeric_in_choices_count`); y-axis is count.
- Interpretation: within each family, compare grouped bars to see where numeric cues are represented.

**(d) BBOX by family and location**
- Type: grouped bar chart.
- Encoding: x-axis is family; hue separates location (`bbox_in_question_count`, `bbox_in_choices_count`); y-axis is count.
- Interpretation: same grouped comparison as (c), for BBOX location counts.

**(e) Top files by BBOX count**
- Type: horizontal bar chart (top-k ranking).
- Encoding: y-axis is source file; x-axis is BBOX signal count.
- Interpretation: panel ranks files by absolute BBOX-related question count.

**(f) File-level numeric signal (Q vs choices)**
- Type: bubble scatter plot.
- Encoding: x-axis is numeric-in-question count; y-axis is numeric-in-choices count; marker size/color scale by question_count.
- Interpretation: each point is one file; position encodes the two location counts and marker size indicates file volume.

---

## Figure 3 — Temporal and duration analytics
File: `fig03_duration_analytics.pdf`

**(a) Duration distribution (log scale)**
- Type: histogram.
- Encoding: x-axis is duration seconds on logarithmic scale; y-axis is frequency.
- Interpretation: bin heights show how many samples fall in each duration range; log x-axis expands short-duration resolution.

**(b) Duration by family (log y)**
- Type: box plot.
- Encoding: x-axis is family; y-axis is duration on logarithmic scale; box/whiskers summarize distribution.
- Interpretation: median, interquartile spread, and whiskers can be compared across families on a multiplicative scale.

**(c) Duration by top profiles**
- Type: violin plot with quartiles.
- Encoding: x-axis is major profile category; y-axis is duration (log scale).
- Interpretation: violin width reflects density; quartile markers indicate central tendency and spread for each profile.

**(d) File duration mean vs median**
- Type: bubble scatter plot with reference diagonal.
- Encoding: x-axis is per-file median duration; y-axis is per-file mean duration; marker size by question_count.
- Interpretation: each point is a file; distance from the dashed y=x line indicates mean-median asymmetry.

**(e) Known duration rate by family**
- Type: bar chart.
- Encoding: x-axis is family; y-axis is percentage of rows with known duration.
- Interpretation: bar height is coverage rate of valid duration metadata within each family.

**(f) Invalid time ranges by family**
- Type: bar chart.
- Encoding: x-axis is family; y-axis is summed invalid range count.
- Interpretation: bars quantify accumulated time-range parsing/validity issues per family.

---

## Figure 4 — Input, answer, and textual structure
File: `fig04_input_answer_structure.pdf`

**(a) inputs_count distribution**
- Type: histogram.
- Encoding: x-axis is number of inputs per question; y-axis is frequency.
- Interpretation: shows the empirical distribution of input cardinality.

**(b) choices_count distribution**
- Type: histogram.
- Encoding: x-axis is number of answer choices; y-axis is frequency.
- Interpretation: shows option-set size distribution across questions.

**(c) clip_ranges_count distribution**
- Type: histogram.
- Encoding: x-axis is number of clip ranges; y-axis is frequency.
- Interpretation: indicates how many temporal segments are referenced per question.

**(d) Average question length (words)**
- Type: bar chart.
- Encoding: x-axis is family; y-axis is mean word count in question text.
- Interpretation: compares average textual length across families.

**(e) Question chars vs choices_count**
- Type: scatter plot (sampled points).
- Encoding: x-axis is character length of question text; y-axis is choices_count; point color maps family.
- Interpretation: each point is a question instance; panel displays joint structure of text length and option count.

**(f) question_id index span by family**
- Type: bar chart.
- Encoding: x-axis is family; y-axis is `max(question_index)-min(question_index)`.
- Interpretation: reflects numeric span of parsed question ID suffixes per family.

---

## Figure 5 — File-level comparative diagnostics
File: `fig05_file_level_comparison.pdf`

**(a) Question count by file**
- Type: horizontal bar chart.
- Encoding: y-axis is source file; x-axis is question count.
- Interpretation: ranks files by dataset size.

**(b) Temporal vs spatial signals (file-level)**
- Type: bubble scatter plot.
- Encoding: x-axis temporal_signal_count; y-axis spatial_signal_count; marker size/color by question_count.
- Interpretation: each point is one file positioned by two signal totals, with volume shown by marker size.

**(c) Numeric vs BBOX signals (file-level)**
- Type: bubble scatter plot.
- Encoding: x-axis numeric_signal_count; y-axis bbox_signal_count; marker size/color by question_count.
- Interpretation: compares co-occurrence intensity of numeric and BBOX signals across files.

**(d) Duration known rate by file (%)**
- Type: horizontal bar chart.
- Encoding: y-axis source file; x-axis known-duration percentage.
- Interpretation: each bar is metadata coverage rate at file granularity.

**(e) Language-only vs spatial (top 12 files)**
- Type: grouped horizontal bar chart.
- Encoding: y-axis source file; x-axis count; hue is metric (`language_only_count` vs `spatial_signal_count`).
- Interpretation: side-by-side metric comparison for the largest files by question volume.

**(f) Top-profile mixture complexity by file**
- Type: horizontal bar chart.
- Encoding: y-axis source file; x-axis token count derived from semicolon-separated `top_profiles`.
- Interpretation: larger values indicate more profile tokens listed in the file-level summary field.

---

## Figure 6 — Cross-factor matrix and rare-case diagnostics
File: `fig06_cross_factor_matrix.pdf`

**(a) Profile × family counts**
- Type: heatmap.
- Encoding: rows are families; columns are profile labels; cell values are raw counts.
- Interpretation: read each cell as absolute number of questions in that family-profile combination.

**(b) Profile × family rates (%)**
- Type: normalized heatmap.
- Encoding: same matrix axes as (a), row-normalized to percentages.
- Interpretation: each row sums to ~100%; cells show within-family profile composition.

**(c) Time token rate by family**
- Type: grouped bar chart.
- Encoding: x-axis family; hue distinguishes question vs choices location; y-axis percentage rate.
- Interpretation: compares where time-token patterns are detected within each family.

**(d) Clip-duration availability by family**
- Type: bar chart.
- Encoding: x-axis family; y-axis percentage with `has_clip_duration=true`.
- Interpretation: panel reports family-level availability rate for clip-duration metadata.

**(e) Language-only rare cases**
- Type: horizontal bar chart (top rare-case groups).
- Encoding: y-axis source file; x-axis count of language-only rows; hue indicates family.
- Interpretation: displays highest-count language-only file-family groups among rare cases.

**(f) Feature correlation matrix**
- Type: correlation heatmap.
- Encoding: rows/columns are numeric/boolean features; cell color encodes Pearson correlation coefficient.
- Interpretation: values near +1/-1 indicate strong positive/negative linear association; values near 0 indicate weak linear association.

---

## Suggested publication phrasing template
Use this style in captions or main text:
- "Panel (x) visualizes <metric> as a function of <grouping>, where <color/size/hue> encodes <variable>."
- "Axes are interpreted as follows: x=<...>, y=<...>; therefore higher/lower values indicate <measurement direction only>."
- "For heatmaps, each cell reports <raw count or normalized rate>; row-normalization implies within-row comparability."
