# HD-EPIC VQA Benchmark Analysis Report

## 1) Dataset Coverage
- Files analyzed: 30
- Questions analyzed: 26550
- Language-only items: 9 (0.03%)
- Known clip durations: 23100
- Unknown clip durations: 3450

## 2) Signal Profiles (All Questions)
- mixed:numeric+spatial+temporal: 12858 (48.43%)
- mixed:numeric+temporal: 11345 (42.73%)
- mixed:bbox+numeric+spatial+temporal: 1228 (4.63%)
- mixed:bbox+numeric+temporal: 672 (2.53%)
- numeric: 420 (1.58%)
- mixed:numeric+spatial: 18 (0.07%)
- language_only: 9 (0.03%)

## 3) Addon: Numeric and BBOX Signals (Where they appear)
### Overall
- Numeric signal in any field: 26541
- Numeric in question body: 26010
- Numeric in options: 26469
- BBOX/coordinate signal in any field: 1900
- BBOX in question body: 1900
- BBOX in options: 0

### By family
| family | q_count | numeric_any | num_q | num_opt | bbox_any | bbox_q | bbox_opt |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3d_perception | 1500 | 1500 | 1500 | 1494 | 1000 | 1000 | 0 |
| fine_grained | 21000 | 21000 | 20773 | 20990 | 0 | 0 | 0 |
| gaze | 2000 | 2000 | 2000 | 1957 | 0 | 0 | 0 |
| ingredient | 400 | 394 | 301 | 393 | 0 | 0 | 0 |
| nutrition | 200 | 200 | 91 | 193 | 0 | 0 | 0 |
| object_motion | 900 | 900 | 900 | 896 | 900 | 900 | 0 |
| recipe | 550 | 547 | 445 | 546 | 0 | 0 | 0 |

## 4) Average Duration by Family (Known durations only)
| family | q_count | duration_known | mean_sec | median_sec |
|---|---:|---:|---:|---:|
| 3d_perception | 1500 | 0 | n/a | n/a |
| fine_grained | 21000 | 21000 | 161.758 | 12.895 |
| gaze | 2000 | 2000 | 6.261 | 10.000 |
| ingredient | 400 | 50 | 35.844 | 20.032 |
| nutrition | 200 | 50 | 7.648 | 6.511 |
| object_motion | 900 | 0 | n/a | n/a |
| recipe | 550 | 0 | n/a | n/a |

## 5) Outputs
- vqa_profiles_per_question.csv: row-level flattened table
- vqa_summary_by_file.csv: aggregate stats by benchmark JSON file
- vqa_summary_by_family.csv: aggregate stats by family
- vqa_summary.json: machine-readable combined summary
- vqa_full_report.md: this report

