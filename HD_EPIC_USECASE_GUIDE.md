# HD-EPIC: Annotation & VQA Usecase Guide for Downstream Agents

> **Dataset**: HD-EPIC — A Highly-Detailed Egocentric Video Dataset (CVPR 2025)  
> **Paper**: [arXiv:2502.04144](https://arxiv.org/abs/2502.04144)  
> **Source**: [hd-epic.github.io](https://hd-epic.github.io/)

This document provides everything a downstream agent needs to correctly load, parse, and use the HD-EPIC annotations for training/testing, including exact JSON/CSV/PKL schemas, concrete examples, and critical pitfalls.

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Naming Conventions & ID Formats](#2-naming-conventions--id-formats)
3. [Narrations & Action Segments](#3-narrations--action-segments)
4. [VQA Benchmark — Master Schema](#4-vqa-benchmark--master-schema)
5. [VQA Category Details & Examples](#5-vqa-category-details--examples)
6. [Scene & Object Movements](#6-scene--object-movements)
7. [Eye Gaze Priming](#7-eye-gaze-priming)
8. [High-Level Activities & Recipes](#8-high-level-activities--recipes)
9. [Audio Annotations](#9-audio-annotations)
10. [Training / Testing Tips & Pitfalls](#10-training--testing-tips--pitfalls)

---

## 1. Repository Structure

```
hd-epic-annotations/
├── narrations-and-action-segments/
│   ├── HD_EPIC_Narrations.pkl          # Main narrations (pickle)
│   ├── HD_EPIC_Narrations_erratum.csv  # Corrections
│   ├── HD_EPIC_noun_classes.csv        # Noun taxonomy
│   └── HD_EPIC_verb_classes.csv        # Verb taxonomy
├── vqa-benchmark/                       # 31 VQA JSON files
│   ├── fine_grained_action_recognition.json
│   ├── fine_grained_action_localization.json
│   ├── fine_grained_how_recognition.json
│   ├── fine_grained_why_recognition.json
│   ├── gaze_gaze_estimation.json
│   ├── gaze_interaction_anticipation.json
│   ├── 3d_perception_fixture_interaction_counting.json
│   ├── 3d_perception_fixture_location.json
│   ├── 3d_perception_object_contents_retrieval.json
│   ├── 3d_perception_object_location.json
│   ├── ingredient_*.json               # 6 files
│   ├── nutrition_*.json                # 3 files
│   ├── object_motion_*.json            # 3 files
│   ├── recipe_*.json                   # 8 files
│   └── VQA_summary.txt                # (empty)
├── scene-and-object-movements/
│   ├── assoc_info.json                 # Object movement associations
│   └── mask_info.json                  # Per-mask spatial data
├── eye-gaze-priming/
│   └── priming_info.json              # Gaze priming events
├── high-level/
│   ├── activities/PXX_recipe_timestamps.csv  # Per-participant
│   └── complete_recipes.json          # Full recipe + nutrition
├── audio-annotations/
│   ├── HD_EPIC_Sounds.csv
│   └── HD_EPIC_Sounds.pkl
└── youtube-links/
```

---

## 2. Naming Conventions & ID Formats

| Entity | Format | Example |
|--------|--------|---------|
| Participant | `PXX` | `P01`, `P09` |
| Video | `PXX-YYYYMMDD-HHMMSS` | `P01-20240202-110250` |
| Unique narration | `PXX-YYYYMMDD-HHMMSS-N` | `P01-20240202-110250-1` |
| Recipe | `PXX-RYY` | `P01-R06` |
| Ingredient | `PXX_RYY_IZZ` | `P09_R04_I03` |
| VQA question | `{category}_{index}` | `fine_grained_action_recognition_42` |

> [!IMPORTANT]
> Video IDs are **always** `PXX-YYYYMMDD-HHMMSS` (hyphen-separated). Do NOT confuse with participant-only IDs (`P01`).

---

## 3. Narrations & Action Segments

### 3.1 Loading `HD_EPIC_Narrations.pkl`

```python
import pandas as pd

df = pd.read_pickle("narrations-and-action-segments/HD_EPIC_Narrations.pkl")
print(df.columns.tolist())
# ['unique_narration_id', 'participant_id', 'video_id', 'narration',
#  'start_timestamp', 'end_timestamp', 'nouns', 'verbs', 'pairs',
#  'main_actions', 'verb_classes', 'noun_classes', 'pair_classes',
#  'main_action_classes', 'hands', 'narration_timestamp']
```

### 3.2 Column Schema

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| `unique_narration_id` | str | `P01-20240202-110250-1` | Globally unique |
| `participant_id` | str | `P01` | |
| `video_id` | str | `P01-20240202-110250` | |
| `narration` | str | `"Open the upper cupboard by holding the handle..."` | Free-form English |
| `start_timestamp` | float64 | `7.44` | **Seconds** (not HH:MM:SS) |
| `end_timestamp` | float64 | `8.75` | **Seconds** |
| `nouns` | list[str] | `['upper cupboard', 'handle of cupboard']` | |
| `verbs` | list[str] | `['open', 'hold']` | |
| `pairs` | list[tuple] | `[('open', 'upper cupboard'), ('hold', 'handle of cupboard')]` | verb-noun pairs |
| `main_actions` | list[tuple] | `[('open', 'upper cupboard')]` | Primary actions only |
| `verb_classes` | list[int] | `[3, 34]` | Maps to `HD_EPIC_verb_classes.csv` |
| `noun_classes` | list[int] | `[3, 3]` | Maps to `HD_EPIC_noun_classes.csv` |
| `pair_classes` | list[tuple] | `[(3, 3), (34, 3)]` | (verb_class, noun_class) |
| `main_action_classes` | list[tuple] | `[(3, 3)]` | |
| `hands` | list[str] | `['left hand']` | Values: `left hand`, `right hand`, `both hands` |
| `narration_timestamp` | float64 | `8.0` | When participant narrated |

> [!WARNING]
> Timestamps in the narrations PKL are in **seconds** (float). VQA question timestamps are in **`HH:MM:SS.mmm`** string format. You must convert between the two when cross-referencing.

### 3.3 Noun & Verb Class CSVs

```python
nouns = pd.read_csv("narrations-and-action-segments/HD_EPIC_noun_classes.csv")
# Columns: ID (int), Key (str), Instances (list-as-string), Category (str)
# Example: ID=0, Key="tap", Instances="['tap', 'tap:water', ...]", Category="appliances"

verbs = pd.read_csv("narrations-and-action-segments/HD_EPIC_verb_classes.csv")
# Same structure. Example: ID=0, Key="take", Category="retrieve"
```

> [!TIP]
> The `Instances` column is stored as a string representation of a Python list. Use `ast.literal_eval()` to parse it.

### 3.4 Erratum

Check `HD_EPIC_Narrations_erratum.csv` for corrected entries before using narrations for training.

---

## 4. VQA Benchmark — Master Schema

**All 31 VQA JSON files share a common top-level structure:**

```jsonc
{
    "<category>_<index>": {
        "inputs": { /* visual inputs — varies by question type */ },
        "question": "string",           // natural language question
        "choices": ["A", "B", "C", "D", "E"],  // ALWAYS exactly 5 choices
        "correct_idx": 2,               // 0-indexed into choices[]
        // optional extra fields depending on category
        "others": { ... },              // fine_grained_action_recognition
        "stat": { ... },                // fine_grained_action_localization
        "metadata": [ ... ]             // nutrition_image_nutrition_estimation
    }
}
```

> [!CAUTION]
> **`correct_idx` is 0-indexed.** `choices[correct_idx]` is the ground-truth answer. Always verify indexing.

### 4.1 Input Pattern Taxonomy

There are **6 distinct input patterns** across the 31 files:

| Pattern | Input Keys | Fields per Input | Used By |
|---------|-----------|-----------------|---------|
| **A: Single video clip** | `"video 1"` | `id`, `start_time`, `end_time` | action_recognition, action_localization, how/why_recognition, gaze_* |
| **B: Full video (no time range)** | `"video 1"` | `id` only | fixture_interaction_counting, fixture_location, object_contents, object_location |
| **C: Multi-video (no time range)** | `"video 1"` … `"video N"` | `id` only | ingredient_*, recipe_*, nutrition_video_* |
| **D: Image-based** | `"image 1"` … `"image N"` | `id`, `time` | nutrition_image_nutrition_estimation |
| **E: Video + Image (mixed)** | `"video 1"` + `"image 1"` | video: `id`; image: `id`, `time` | object_motion_* |
| **F: Multi-video with time ranges** | `"video 1"` … `"video N"` | `id`, `start_time`, `end_time` | *(not observed — all multi-video is full)* |

### 4.2 Special Tokens in Questions and Choices

Questions and answer choices contain embedded **special tokens**:

| Token | Format | Meaning |
|-------|--------|---------|
| `<BBOX x1 y1 x2 y2>` | `<BBOX 419.298 759.812 946.760 1122.678>` | Bounding box `[xmin, ymin, xmax, ymax]` in pixel coords |
| `<TIME HH:MM:SS.mmm video N>` | `<TIME 00:03:1.8 video 1>` | Timestamp reference tied to a specific video input |
| `<verb noun>` | `<hit spatula>` | Action class label (verb-noun pair) |

> [!IMPORTANT]
> Your model/tokenizer **must** handle these special tokens. They appear in both questions AND choices. If you strip them, you lose critical spatial/temporal grounding.

### 4.3 Timestamp Formats

| Context | Format | Example |
|---------|--------|---------|
| VQA questions/choices | `HH:MM:SS.mmm` | `00:03:1.8`, `00:13:35.100` |
| VQA video clip inputs | `HH:MM:SS.mmm` | `00:01:15.539` |
| VQA image inputs | `HH:MM:SS.mmm` | `00:13:43.393` |
| Narrations PKL | **seconds (float)** | `7.44` |
| Audio annotations CSV | `HH:mm:ss.SSS` | `00:00:00.476` |
| Audio annotations PKL | sample index (int at 48kHz) | `22848` |

---

## 5. VQA Category Details & Examples

### 5.1 Fine-Grained Action Recognition

**File**: `fine_grained_action_recognition.json` (~10,000 questions, 14MB)  
**Task**: Given a short video clip, select the sentence that best describes the action(s).  
**Input pattern**: Single video clip (`id` + `start_time` + `end_time`)

```json
{
    "fine_grained_action_recognition_0": {
        "inputs": {
            "video 1": {
                "id": "P03-20240216-205923",
                "start_time": "00:01:15.539",
                "end_time": "00:01:16.310"
            }
        },
        "question": "Which of these sentences best describe the ongoing action(s) in the video 1?",
        "choices": [
            "Rub my hands against each other to remove the oil...",
            "Rub my hands against each other to apply the handwash...",
            "Rub my hands against each other to shake off the semolina flour...",
            "Rub hands above the sink to get rid of garlic...",
            "Rub my hands above the kitchen top to shake the semolina flour..."
        ],
        "correct_idx": 1,
        "others": {
            "actions_num": "1",
            "num_diffs_verbs": "0",
            "num_diffs_nouns": "0"
        }
    }
}
```

**`others` metadata**:
- `actions_num`: Number of distinct actions in the clip
- `num_diffs_verbs`: How many distractor choices have different verbs
- `num_diffs_nouns`: How many distractor choices have different nouns

> [!TIP]
> When `num_diffs_verbs=0` and `num_diffs_nouns=0`, all choices share the same verb and noun — the model must distinguish based on **fine-grained context** (hand used, location, purpose).

---

### 5.2 Fine-Grained Action Localization

**File**: `fine_grained_action_localization.json` (~10,000 questions, 9.5MB)  
**Task**: Given a video clip and an action label `<verb noun>`, select the correct temporal window.  
**Input pattern**: Single video clip (longer duration window)

```json
{
    "fine_grained_action_localization_0": {
        "inputs": {
            "video 1": {
                "id": "P01-20240204-121042",
                "start_time": "00:04:23.750",
                "end_time": "00:10:51.250"
            }
        },
        "question": "When did the action <hit spatula> happen in the video 1?",
        "choices": [
            "video 1 from <TIME 00:08:00.870 video 1> to <TIME 00:08:01.920 video 1>",
            "video 1 from <TIME 00:07:16.470 video 1> to <TIME 00:07:17.149 video 1>",
            "video 1 from <TIME 00:09:53.840 video 1> to <TIME 00:09:54.639 video 1>",
            "video 1 from <TIME 00:07:19.709 video 1> to <TIME 00:07:20.209 video 1>",
            "video 1 from <TIME 00:10:36.600 video 1> to <TIME 00:10:37.509 video 1>"
        ],
        "correct_idx": 2,
        "stat": { "hard": 4, "middle": 0, "easy": 0 }
    }
}
```

**`stat` metadata**: Difficulty distribution of distractor choices.

---

### 5.3 Fine-Grained How/Why Recognition

**Files**: `fine_grained_how_recognition.json`, `fine_grained_why_recognition.json`  
**Task**: Given a video clip, answer **how** or **why** an action was performed.  
**Input pattern**: Single video clip

---

### 5.4 Gaze Estimation & Interaction Anticipation

**Files**: `gaze_gaze_estimation.json`, `gaze_interaction_anticipation.json`  
**Task**: Determine what the person is looking at / will interact with next.  
**Input pattern**: Single video clip

```json
{
    "gaze_gaze_estimation_0": {
        "inputs": {
            "video 1": {
                "id": "P05-20240424-090812",
                "start_time": "00:00:05.366",
                "end_time": "00:00:05.966"
            }
        },
        "question": "What is the person looking at in this video segment?",
        "choices": [
            "At the freezer.",
            "At the cupboard to the right of and below the freezer.",
            "At the counter to the right of the fridge.",
            "At the microwave.",
            "At the fridge."
        ],
        "correct_idx": 2
    }
}
```

---

### 5.5 3D Perception (Fixture Counting, Location, Object Contents/Location)

**Files**: `3d_perception_fixture_interaction_counting.json`, `3d_perception_fixture_location.json`, `3d_perception_object_contents_retrieval.json`, `3d_perception_object_location.json`  
**Task**: Spatial reasoning about fixtures/objects in the kitchen.  
**Input pattern**: Full video (no time range) + bounding box / time tokens in question

```json
{
    "3d_perception_fixture_interaction_counting_0": {
        "inputs": {
            "video 1": { "id": "P01-20240202-110250" }
        },
        "question": "How many times did I close the item indicated by bounding box <BBOX 419.29 759.81 946.76 1122.67> in <TIME 00:03:1.8 video 1>?",
        "choices": ["0", "3", "4", "2", "1"],
        "correct_idx": 0
    }
}
```

> [!WARNING]
> BBOX coordinates are **floating point pixel values**, not normalized. Verify against the video resolution (HD-EPIC uses 1408×1408 Aria camera frames).

---

### 5.6 Object Motion (Movement Counting, Itinerary, Stationary Localization)

**Files**: `object_motion_object_movement_counting.json`, `object_motion_object_movement_itinerary.json`, `object_motion_stationary_object_localization.json`  
**Task**: Track object movements across an entire video.  
**Input pattern**: Full video + reference image (mixed `video 1` + `image 1`)

```json
{
    "object_motion_object_movement_counting_0": {
        "inputs": {
            "video 1": { "id": "P03-20240217-131219" },
            "image 1": {
                "time": "00:01:21.733",
                "id": "P03-20240217-131219"
            }
        },
        "question": "How many times did the object <BBOX 787.03 344.17 1054.19 565.60> seen at <TIME 00:01:21.733 video 1> change locations in the video?",
        "choices": ["6", "7", "5", "1", "3"],
        "correct_idx": 4
    }
}
```

> [!NOTE]
> The `image 1` input provides a **reference frame** (same video, specific timestamp) where the object's bounding box is visible. The question asks about movements across the **entire** video.

---

### 5.7 Ingredient Questions (6 subtypes)

**Files**: `ingredient_exact_ingredient_recognition.json`, `ingredient_ingredient_adding_localization.json`, `ingredient_ingredient_recognition.json`, `ingredient_ingredient_retrieval.json`, `ingredient_ingredient_weight.json`, `ingredient_ingredients_order.json`  
**Task**: Questions about ingredients used in recipes — quantities, order, timing.  
**Input pattern**: Multi-video (full, no time ranges) — all videos from one participant

```json
{
    "ingredient_exact_ingredient_recognition_0": {
        "inputs": {
            "video 1": { "id": "P09-20240624-160737" },
            "video 2": { "id": "P09-20240621-093545" },
            "video 3": { "id": "P09-20240621-153208" },
            "video 4": { "id": "P09-20240624-165332" },
            "video 5": { "id": "P09-20240622-194642" },
            "video 6": { "id": "P09-20240622-150155" },
            "video 7": { "id": "P09-20240622-154652" },
            "video 8": { "id": "P09-20240623-120359" }
        },
        "question": "What was the exact quantity of garlic and ginger paste used in Mangsho Bhuna",
        "choices": ["3 tbsp", "2 tbsp", "1 tbsp", "4 tbsp", "0 tbsp"],
        "correct_idx": 1
    }
}
```

> [!IMPORTANT]
> Ingredient questions often reference **ALL videos from a participant** (up to 19 videos). This requires cross-video reasoning or access to the `complete_recipes.json` metadata.

---

### 5.8 Nutrition (Image & Video Estimation, Change)

**Files**: `nutrition_image_nutrition_estimation.json`, `nutrition_video_nutrition_estimation.json`, `nutrition_nutrition_change.json`  
**Task**: Compare nutritional content (calories, fat, carbs, protein) of ingredients.

**Image-based** (has rich `metadata` array):
```json
{
    "nutrition_image_nutrition_estimation_0": {
        "inputs": {
            "image 1": { "id": "P09-20240623-120359", "time": "00:13:43.393" },
            "image 2": { "id": "P08-20240618-171546", "time": "00:11:29.074" },
            "image 3": { "id": "P06-20240510-115307", "time": "00:14:24.915" },
            "image 4": { "id": "P01-20240203-132119", "time": "00:01:14.560" },
            "image 5": { "id": "P05-20240427-145526", "time": "00:04:57.470" }
        },
        "question": "Which of the ingredients in these images showcase higher carbs?",
        "choices": ["yoghurt", "mixed herbs", "tomatoes", "chicken stock cube", "peanut butter"],
        "correct_idx": 2,
        "metadata": [
            {
                "ingredient": "yoghurt",
                "ingredient_id": "P09_R04_I03",
                "video_id": "P09-20240623-120359",
                "add_start": 823.393,
                "add_end": 824.056,
                "amount": 4.0,
                "calories": 62.0,
                "fat": 4.0,
                "carbs": 2.0,
                "protein": 2.0,
                "carbs_amount_ratio": 0.5
            }
            // ... one entry per choice
        ]
    }
}
```

> [!TIP]
> The `metadata` array in nutrition questions is **gold-standard** — it contains the actual nutritional values and can be used for both training supervision and evaluation. The `*_amount_ratio` field key changes depending on the nutritional property being asked about (`carbs_amount_ratio`, `protein_amount_ratio`, `fat_amount_ratio`, `calories_amount_ratio`).

---

### 5.9 Recipe Questions (8 subtypes)

**Files**: `recipe_following_activity_recognition.json`, `recipe_multi_recipe_recognition.json`, `recipe_multi_step_localization.json`, `recipe_prep_localization.json`, `recipe_recipe_recognition.json`, `recipe_rough_step_localization.json`, `recipe_step_localization.json`, `recipe_step_recognition.json`  
**Task**: Recipe-related reasoning — recognition, step localization, temporal ordering.  
**Input pattern**: Multi-video (1–6 videos, full, no time ranges)

**Step localization choices use `<TIME>` tokens with multiple time segments:**

```json
{
    "recipe_step_localization_0": {
        "inputs": {
            "video 1": { "id": "P07-20240529-191007" },
            "video 2": { "id": "P07-20240529-194518" }
        },
        "question": "When did the participant perform step Grate the cheese from recipe Mushroom Risotto?",
        "choices": [
            "<TIME 00:07:25.128 video 2> to <TIME 00:07:31.667 video 2> (video 2), ...",
            "<TIME 00:00:42.081 video 1> to <TIME 00:00:46.211 video 1> (video 1), ...",
            "...",
            "...",
            "<TIME 00:11:34.701 video 2> to <TIME 00:13:22.210 video 2> (video 2)"
        ],
        "correct_idx": 4
    }
}
```

> [!NOTE]
> Recipe step localization choices contain **comma-separated lists of time segments**, potentially spanning multiple videos. Each segment is `<TIME start video N> to <TIME end video N> (video N)`.

---

## 6. Scene & Object Movements

### 6.1 Association Info (`assoc_info.json`)

Groups individual object movements into "associations" (same object tracked over time).

```json
{
    "P01-20240202-110250": {
        "association_0": {
            "name": "plate",
            "tracks": [
                {
                    "track_id": "track_001",
                    "time_segment": [12.5, 15.3],
                    "masks": ["mask_id_001", "mask_id_002"]
                }
            ]
        }
    }
}
```

### 6.2 Mask Info (`mask_info.json`)

Per-mask spatial data, queryable by mask IDs from `assoc_info.json`.

```json
{
    "P01-20240202-110250": {
        "mask_id_001": {
            "frame_number": 375,
            "3d_location": [1.2, 0.5, 2.3],
            "bbox": [693.1, 847.2, 775.0, 979.8],
            "fixture": "P01_cupboard.009"
        }
    }
}
```

| Field | Type | Notes |
|-------|------|-------|
| `frame_number` | int | MP4 frame index, **0-indexed** |
| `3d_location` | [x, y, z] | World coordinates |
| `bbox` | [xmin, ymin, xmax, ymax] | Pixel coordinates |
| `fixture` | str | Scene fixture name, or `"Null"` |

> [!WARNING]
> Masks and bounding boxes were completed by **different annotation teams** and may be inconsistent in some cases. Validate before relying on tight spatial alignment.

---

## 7. Eye Gaze Priming

**File**: `eye-gaze-priming/priming_info.json`

Tracks whether gaze anticipates object interactions (pick-up/put-down).

```json
{
    "P01-20240202-110250": {
        "0": {
            "start": {
                "frame": 150,
                "3d_location": [1.0, 0.3, 2.1],
                "prime_stats": {
                    "prime_window_start": 100,
                    "frame_primed": 130,
                    "gaze_point": [1.1, 0.3, 2.0],
                    "dist_to_cam": 1.5,
                    "prime_gap": 0.67
                }
            },
            "end": { /* same structure */ }
        }
    }
}
```

**Key `frame_primed` values:**
- `>= 0`: Exact frame of priming
- `-1`: Valid location, but **no priming occurred**
- `-2`: **Excluded** (off-screen / ongoing manipulation)

---

## 8. High-Level Activities & Recipes

### 8.1 Activity Timestamps (`high-level/activities/PXX_recipe_timestamps.csv`)

| Column | Description |
|--------|-------------|
| `video_id` | e.g., `P01-20240202-110250` |
| `recipe_id` | Recipe ID for this participant (empty = background) |
| `high_level_activity_label` | Free-form description |

### 8.2 Complete Recipes (`high-level/complete_recipes.json`)

Comprehensive JSON with recipe metadata, steps, ingredients, and nutrition:

```json
{
    "P01-R06": {
        "participant": "P01",
        "name": "Carrot Soup",
        "type": "modified",
        "source": "https://...",
        "steps": {
            "step_1": "Peel and chop the carrots...",
            "step_2": "Heat oil in a pan..."
        },
        "captures": {
            "capture_1": {
                "videos": ["P01-20240203-132119", "P01-20240203-135502"],
                "ingredients": {
                    "P01_R06_I01": {
                        "name": "carrots",
                        "amount": 500,
                        "amount_unit": "g",
                        "calories": 205,
                        "carbs": 48,
                        "fat": 1.2,
                        "protein": 4.7,
                        "weigh": [{"video_id": "...", "start": 10.5, "end": 15.2}],
                        "add": [{"video_id": "...", "start": 30.1, "end": 32.5}]
                    }
                }
            }
        }
    }
}
```

> [!TIP]
> The `weigh` and `add` segments provide temporal links between ingredient usage and specific video timestamps. Use these for training ingredient detection / addition recognition models.

---

## 9. Audio Annotations

**File**: `audio-annotations/HD_EPIC_Sounds.csv`

| Column | Type | Example |
|--------|------|---------|
| `participant_id` | str | `P01` |
| `video_id` | str | `P01-20240202-110250` |
| `start_timestamp` | str | `00:00:00.476` (HH:mm:ss.SSS) |
| `stop_timestamp` | str | `00:00:02.520` |
| `start_sample` | int | `22848` (at 48kHz) |
| `stop_sample` | int | `120960` |
| `class` | str | `rustle` |
| `class_id` | int | `4` |

> [!TIP]
> To convert samples to seconds: `seconds = sample_index / 48000`

---

## 10. Training / Testing Tips & Pitfalls

### Critical Formatting Rules

1. **VQA is always 5-way multiple choice.** `correct_idx` is **0-indexed**. The answer is `choices[correct_idx]`.

2. **Timestamp format varies by file type:**
   - VQA JSON: `"HH:MM:SS.mmm"` (string)
   - Narrations PKL: `float` seconds
   - Audio CSV: `"HH:mm:ss.SSS"` (string)
   - Conversion: parse the string to get total seconds.

3. **Bounding boxes are in absolute pixels**, NOT normalized `[0,1]`. The Aria glasses produce 1408×1408 frames. BBOX format is always `[xmin, ymin, xmax, ymax]`.

4. **Multi-video inputs**: Some questions reference up to **19 videos** simultaneously. Your data loader must handle variable-length video lists.

### Data Loading Patterns

```python
import json

# Loading any VQA file
with open("vqa-benchmark/fine_grained_action_recognition.json") as f:
    data = json.load(f)

for qid, item in data.items():
    question = item["question"]
    choices = item["choices"]         # Always len=5
    answer_idx = item["correct_idx"]  # 0-indexed
    answer_text = choices[answer_idx]

    # Parse inputs (varies by category)
    for input_key, input_val in item["inputs"].items():
        video_id = input_val["id"]
        if "start_time" in input_val:
            # Pattern A: video clip
            start = input_val["start_time"]
            end = input_val["end_time"]
        elif "time" in input_val:
            # Pattern D: single image frame
            frame_time = input_val["time"]
        else:
            # Pattern B/C: full video
            pass
```

### Evaluation Protocol

For benchmarking against the HD-EPIC VQA:
- Report **accuracy** per category (31 files → up to 31 sub-metrics)
- Group into **8 high-level categories**: fine_grained, gaze, 3d_perception, object_motion, ingredient, nutrition, recipe
- The paper likely reports both per-category and aggregate accuracy

### Common Pitfalls

| Pitfall | Impact | Fix |
|---------|--------|-----|
| Treating `correct_idx` as 1-indexed | All predictions off by 1 | Use `choices[correct_idx]` directly |
| Normalizing BBOX to [0,1] | Wrong spatial grounding | Use raw pixel coords (1408×1408 frame) |
| Ignoring `<TIME>` tokens in choices | Can't localize temporal segments | Parse `<TIME HH:MM:SS.mmm video N>` tokens |
| Loading only 1 video for multi-video Qs | Missing context for ingredient/recipe Qs | Load ALL listed video IDs |
| Mixing up narration seconds vs VQA timestamp strings | Temporal misalignment | Convert consistently |
| Ignoring `erratum.csv` | Using corrected narrations | Apply corrections before training |
| Treating `metadata` in nutrition questions as always present | KeyError for other categories | Check for key existence |
| Assuming `time_segment` in assoc_info uses frames | Wrong temporal alignment | Values are in **seconds** |

### Train/Test Split

> [!CAUTION]
> The repository does **not** include an explicit train/test split. Check the [project webpage](https://hd-epic.github.io/) or paper for the official split. Do NOT create your own split without verifying against the published protocol, or you risk train-test leakage across participants/recipes.

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│  HD-EPIC VQA: Loading Checklist                             │
├─────────────────────────────────────────────────────────────┤
│  ✓ JSON files: json.load() → dict of question_id → item    │
│  ✓ Always 5 choices, correct_idx is 0-indexed               │
│  ✓ Parse <BBOX x1 y1 x2 y2> tokens (absolute pixels)       │
│  ✓ Parse <TIME HH:MM:SS.mmm video N> tokens                │
│  ✓ Handle variable # of video/image inputs per question     │
│  ✓ Narrations PKL: timestamps in seconds (float)            │
│  ✓ Apply erratum corrections                                │
│  ✓ Noun/Verb class CSVs: parse Instances with literal_eval  │
│  ✓ BBOX coords are for 1408×1408 Aria frames                │
│  ✓ Mask/bbox annotations may be inconsistent (diff teams)   │
└─────────────────────────────────────────────────────────────┘
```
