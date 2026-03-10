## MetaFood Updates (March 2026)

This repository includes additional tooling and analysis outputs for practical HD-EPIC annotation usage and VQA benchmark diagnostics.

### Official benchmark column mapping (VQA categories)

The official HD-EPIC benchmark columns correspond to VQA JSON families by filename prefix:

- `Recipe (%)` → `recipe_*.json`
- `Ingredient (%)` → `ingredient_*.json`
- `Nutrition (%)` → `nutrition_*.json`
- `Action (%)` → `fine_grained_*.json`
- `3D (%)` → `3d_perception_*.json`
- `Motion (%)` → `object_motion_*.json`
- `Gaze (%)` → `gaze_*.json`

### Official benchmark snapshot (HD-EPIC)

| # | Participant | Date | ID | Team Name | Overall (%) | Recipe (%) | Ingredient (%) | Nutrition (%) | Action (%) | 3D (%) | Motion (%) | Gaze (%) |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | EPIC-KITCHENS | 2026-02-05 07:01 | 525125 | DeepFrames | 44.21 | 64.62 | 54.67 | 38.00 | 48.30 | 42.59 | 30.17 | 31.15 |
| 2 | EPIC-KITCHENS | 2026-02-05 07:01 | 525126 | HelloWorld | 41.55 | 64.75 | 43.33 | 37.00 | 42.03 | 40.88 | 29.90 | 32.95 |
| 3 | EPIC-KITCHENS | 2026-02-05 06:59 | 525118 | Gemini Pro [HD-EPIC Baseline] | 37.57 | 60.50 | 46.17 | 34.67 | 39.63 | 32.51 | 20.83 | 28.65 |
| 4 | EPIC-KITCHENS | 2026-02-05 06:59 | 525119 | LLaVA-Video [HD-EPIC Baseline] | 32.43 | 36.25 | 33.50 | 38.67 | 43.02 | 27.31 | 18.93 | 29.30 |
| 5 | EPIC-KITCHENS | 2026-02-05 07:00 | 525121 | LongVA [HD-EPIC Baseline] | 29.28 | 29.62 | 30.83 | 33.67 | 30.68 | 32.91 | 22.73 | 24.50 |
| 6 | EPIC-KITCHENS | 2026-02-05 07:00 | 525124 | VideoLLaMA 2 [HD-EPIC Baseline] | 27.39 | 30.75 | 25.67 | 32.67 | 27.24 | 25.74 | 28.50 | 21.20 |

### Difficulty diagnostics (leaderboard vs dataset factors)

We added a focused analysis that compares official family-level leaderboard accuracy (6 participants) against family-level dataset diagnostics in `analysis-output/vqa_summary_by_family.csv`.

- Radar chart (participants × VQA families):
  - `analysis-output/figures/fig07_official_leaderboard_radar.png`
  - `analysis-output/figures/fig07_official_leaderboard_radar.pdf`
- Family mean accuracy table: `analysis-output/leaderboard_family_accuracy.csv`
- Correlation table: `analysis-output/leaderboard_accuracy_correlations.csv`
- Summary report: `analysis-output/leaderboard_difficulty_report.md`

Current directional findings (small-sample, family-level):
- Hardest families by mean accuracy: `object_motion` (25.18%), `gaze` (27.96%), `3d_perception` (33.66%).
- Strongest negative correlations with accuracy:
  - `bbox_density` (Pearson r = -0.605)
  - `spatial_density` (Pearson r = -0.555)
- Interpretation: families with denser spatial/BBOX grounding tend to be harder for current systems in this benchmark snapshot.

### What we added
- One-click launchers for the annotation interface on both Windows and macOS.
- VQA analysis scripts and generated summary outputs in `analysis-output/`.
- Publication-style composite figures in `analysis-output/figures/` with supporting notes.

### One-click annotation interface usage (Windows / macOS)
The annotation UI is the HTML file `HD_EPIC_VQA_Interface.html`. We provide launcher scripts that open the page and start a local HTTP server on port 8000.

#### Windows
- Double-click `START_HD_EPIC_INTERFACE.bat`
- Or run: `./START_HD_EPIC_INTERFACE.bat`

#### macOS
- Double-click `START_HD_EPIC_INTERFACE.command`
- If macOS blocks execution on first run, open Terminal in this folder and run:
  - `chmod +x START_HD_EPIC_INTERFACE.command`
  - `./START_HD_EPIC_INTERFACE.command`

Both launchers open:
- `http://127.0.0.1:8000/HD_EPIC_VQA_Interface.html`

### Figure motivation and descriptions
The figures in `analysis-output/figures/` are intended to make dataset structure, annotation signal composition, and potential benchmark edge cases quickly auditable before model training/evaluation.

- **Figure 1 (`fig01_global_overview`)**
  - **Motivation:** establish a high-level dataset baseline and check class/signal balance.
  - **What it shows:** profile mix, signal prevalence, duration availability, family volumes, family-level signal rates, and correct option index distribution.

- **Figure 2 (`fig02_numeric_bbox_locality`)**
  - **Motivation:** locate where numeric and BBOX cues appear, and detect leakage-like concentration patterns.
  - **What it shows:** overall cue location (question/choices/metadata), family-level breakdowns, top BBOX-heavy files, and file-level numeric-in-question vs numeric-in-choices relationship.

- **Figure 3 (`fig03_duration_analytics`)**
  - **Motivation:** characterize temporal difficulty and duration metadata quality.
  - **What it shows:** global log-scale duration distribution, family/profile duration spread, mean-vs-median file diagnostics, known-duration rates, and invalid time-range counts.

- **Figure 4 (`fig04_input_answer_structure`)**
  - **Motivation:** verify consistency of question schema and textual/option structure.
  - **What it shows:** distributions of input/choice/clip-range counts, family-wise question length, question-length vs choice-count relation, and question-id index span by family.

- **Figure 5 (`fig05_file_level_comparison`)**
  - **Motivation:** compare JSON files as benchmark sources and identify skewed contributors.
  - **What it shows:** per-file question volume, temporal-vs-spatial and numeric-vs-BBOX signal scatter, per-file duration coverage, language-only vs spatial comparisons, and profile-mixture complexity.

- **Figure 6 (`fig06_cross_factor_matrix`)**
  - **Motivation:** inspect cross-factor interactions and rare-case structures that may impact generalization.
  - **What it shows:** profile×family count/rate matrices, time-token location rates, clip-duration availability by family, language-only rare cases, and feature correlation heatmap.

- **Large-duration diagnostics (`big_duration_by_json_type`, `big_effective_duration_by_json_type`)**
  - **Motivation:** provide a detailed per-JSON-type temporal reference beyond compact composites.
  - **What they show:** per-type duration distributions/coverage and effective-viewing-duration composition across all VQA JSON types.

For neutral publication-style panel-level explanations, see:
- `analysis-output/figures/FIGURE_DESCRIPTIONS.md`
- `analysis-output/figures/FIGURE_MANIFEST.md`

## ![logo](logo-white.png) HD-EPIC: A Highly-Detailed Egocentric Video Dataset (CVPR 2025)


<!-- start badges -->
[![arXiv-2502.04144](https://img.shields.io/badge/arXiv-2502.04144-green.svg)](https://arxiv.org/abs/2502.04144)
<!-- end badges -->

## Project Webpage
Dataset - download and further information is available from [Project Webpage](https://hd-epic.github.io/)

Paper is available at [ArXiv](https://hd-epic.github.io/)

## Citing
When using the dataset, kindly reference:
```
@InProceedings{perrett2025hdepic,
  author    = {Perrett, Toby and Darkhalil, Ahmad and Sinha, Saptarshi and Emara, Omar and Pollard, Sam and Parida, Kranti and Liu, Kaiting and Gatti, Prajwal and Bansal, Siddhant and Flanagan, Kevin and Chalk, Jacob and Zhu, Zhifan and Guerrier, Rhodri and Abdelazim, Fahd and Zhu, Bin and Moltisanti, Davide and Wray, Michael and Doughty, Hazel and Damen, Dima},
  title     = {HD-EPIC: A Highly-Detailed Egocentric Video Dataset},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
  month     = {June}
}
```

## ![NEW](https://img.shields.io/badge/NEW-red?style=for-the-badge) HD-EPIC Intermediate Data

We have released new intermediate data for HD-EPIC, aligned to the MP4 videos, providing per-video
Aria glasses device calibration and frame-wise camera pose and gaze information.

**Includes:**
- Per-video static device calibration (cameras and sensors), including sensor to device transforms
- Per-frame device to world transforms
- Per-frame gaze centre (image space) and 3D gaze direction (world space)

**[Download intermediate data](https://uob-my.sharepoint.com/:f:/g/personal/jc17360_bristol_ac_uk/IgCCGb5qDbiOR7cmj1R9OyUWAXQFYL7FP_d0eMzB4ENPVQk?e=3hngWD)**

Full details are provided in the README in the link above.

## Narrations and Action Segments

This folder contains narration annotations structured as follows:

- `HD_EPIC_Narrations.pkl`: labels narration/action segments and associated annotations.
- `HD_EPIC_verb_classes.csv`: labels verb clusters.
- `HD_EPIC_noun_classes.csv`: labels noun clusters.

Details about each file are provided below.

### `HD_EPIC_Narrations.pkl`

This pickle file contains the action descriptions for HD-EPIC and contains 16 columns:

| Column Name           | Type    | Example                                                                             | Description                                                                           |
| --------------------- | ------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `unique_narration_id` | string  | `P01-20240202-110250-1`                                                             | Unique ID for the narration/action as a string with participant ID, video ID, and action index. |
| `participant_id`      | string  | `P01`                                                                               | ID of the participant (unique per participant).                                       |
| `video_id`            | string  | `P01-20240202-110250`                                                               | ID of the video where the action originated from (unique per video).                  |
| `narration`           | string  | `Open the upper cupboard by holding the handle of the cupboard with the left hand.` | Narration or description of the performed action.                                     |
| `start_timestamp`     | float64 | `7.44`                                                                              | Narration/action segment start time in seconds.                                                 |
| `end_timestamp`       | float64 | `8.75`                                                                              | Narration/action segment end time in seconds.                                                   |
| `nouns`               | list  | `['upper cupboard', 'handle of cupboard']`                                          | List of nounds extracted from the narration description.                                 |
| `verbs`               | list  | `['open', 'hold']`                                                                  | List of verbs extracted from the narration description.                                  |
| `pairs`               | list  | `[('open', 'upper cupboard'), ('hold', 'handle of cupboard')]`                      | List of (verb, noun) pairs extracted from the narration description.                     |
| `main_actions`        | list  | `[('open', 'upper cupboard')]`                                                      | List of main actions classes performed.                                               |
| `verb_classes`        | list  | `[3, 34]`                                                                           | Numeric labels for extracted verbs.                                                   |
| `noun_classes`        | list  | `[3, 3]`                                                                            | Numeric labels for extracted nouns.                                                   |
| `pair_classes`        | list  | `[(3, 3), (34, 3)]`                                                                 | Numeric labels for extracted verb-noun pairs.                                         |
| `main_action_classes` | list  | `[(3, 3)]`                                                                          | Numeric labels for main action categories.                                            |
| `hands`               | list  | `['left hand']`                                                                     | List of hands (`left hand`, `right hand`, `both hands`) mentioned in the narration. |
| `narration_timestamp`  | float64 | `8.0`                                                                               | Timestamp when the narration was recorded by the participant, in seconds.                                   |

### `HD_EPIC_noun_classes.csv`

This file contains information of nouns extracted from narration descriptions in HD-EPIC and contains 4 columns:

| Column Name | Type   | Example                                             | Description                                     |
| ----------- | ------ | --------------------------------------------------- | ----------------------------------------------- |
| `ID`        | int    | `0`                                                 | Numerical label assigned to the noun.           |
| `Key`       | string | `tap`                                               | Base form label for the noun.                   |
| `Instances` | list   | `['tap', 'tap:water', 'water:tap', ...` | List of parsed variations mapped to this label. |
| `Category`  | string | `appliances`                                        | High-level taxonomic category of the noun.      |


### `HD_EPIC_verb_classes.csv`

This file contains information of verbs extracted from narration descriptions in HD-EPIC and contains 4 columns:

| Column Name | Type   | Example                                             | Description                                     |
| ----------- | ------ | --------------------------------------------------- | ----------------------------------------------- |
| `ID`        | int    | `0`                                                 | Numerical label assigned to the verb.           |
| `Key`       | string | `take`                                              | Base form label for the verb.                   |
| `Instances` | list   | `['collect-from', 'collect-into', 'draw', ...` | List of parsed variations mapped to this label. |
| `Category`  | string | `retrieve`                                          | High-level taxonomic category of the verb.      |


## Digital Twin: Scene & Object Movements
We annotate object movements by labeling temporal segments from pick-up to placement and 2D bounding boxes at movement onset and end. Tracks include even slight shifts/pushes, ensuring full coverage of movements. Every object movement is annotated and assgin to a scene fixture, providing a rich dataset for analysis. Movements of the same object are then grouped into "associations" by human annotators. This association data is stored across two JSON files. The first (`scene-and-object-movements/assoc_info.json`) is a JSON object where the keys are video names and the values are groupings of each object's movements throughout the video (referred to as "associations"). The structure for this file is as follows:
```jsonc
{
  "video_id": {
    "association_id": {
      "name": "string",
      "tracks": [
        {
          "track_id": "string",
          "time_segment": [start_time, end_time],
          "masks": ["string", ...]
        },
        ...
      ]
    },
    ...
  }
}
```
The string IDs in "masks" can then be used to query the second JSON file (`scene-and-object-movements/mask_info.json`) for information on MP4 frame number, 3D location, bounding box and scene fixture of each object mask. The structure of this JSON object is as follows:
```jsonc
{
  "video_id": {
    "mask_id": {
      "frame_number": integer,
      "3d_location": [x, y, z],
      "bbox": [xmin, ymin, xmax, ymax],
      "fixture": "string"
    },
    ...
  }
}
```
Each `mask_id` can be matched to a mask file name (e.g. `frame_id.png`) in the [dropbox](https://www.dropbox.com/scl/fo/f7hwei2m8y3ihlhp669h4/ALM8_1LDETY40O-06-ptr3A?rlkey=yrmqm3zk284htr5yjxb4z5nwp&e=1&st=815ovw6m&dl=0). It should be noted that the masks and bounding boxes were completed by different teams and therefore may be inconsistent in places.

**Field Descriptions**
- **`video_id`**: The name of the video, i.e. `P01-20240202-110250`
- **`association_id`**: A unique identifier for the object movement tracks
- **`name`**: The name of the association, i.e. `plate`
- **`tracks`**: A list of object movements that make up the association
- **`track_id`**: A unique identifier for the single movement of the object in the association
- **`time_segment`**: A start and end time for the single movement of the object in the association
- **`masks`**: A list of unique identifiers for each object mask connected to this particular movement of the object
- **`mask_id`**: A unique identifier for the object mask. This can be matched to a mask ID in the `masks` field of `assoc_info.json`, if this frame is connected to an association
- **`frame_number`**: The MP4 frame number for the particular frame, starting from 0 index.
- **`bbox`**: A four-element list specifying the 2D bounding box `[xmin, ymin, xmax, ymax]`, i.e. `[693.1, 847.2, 775.00, 979.8]`.
- **`fixture`**: A string indicating the fixture the object is assigned to, i.e. `P01_cupboard.009` and `Null` if no assigned fixture.

## Eye Gaze Priming
We annotate priming moments when gaze anticipates object interactions—either by fixating on the pick-up location before the object is moved, or the placement location before it is put down. For pick-up priming, we project 3D gaze onto object locations within a 10-second window before the labelled interaction. For put-down priming, we use a similar window, starting either up to 10 seconds before placement or from the moment the object is lifted for shorter interactions. Near misses, where gaze is close but doesn’t directly intersect the object, are also captured using a proximity-based threshold. We exclude off-screen interactions and discard cases where gaze is already near the object long before motion starts, to avoid capturing ongoing manipulation.

Priming data is stored in a single JSON file (`eye_gaze_priming/priming_info.json`), where the top-level keys correspond to `video_ids`. Each value is a dictionary keyed by an object identifier (e.g. `"0"`, `"1"`, etc.), which contains information about the object’s pick-up (start) and put-down (end) events, along with associated priming metadata. The structure is as follows:
```jsonc
{
  "video_id": {
    "object_id": {
      "start": {
        "frame": integer,
        "3d_location": [x, y, z],
        "prime_stats": {
          "prime_window_start": integer,
          "frame_primed": integer,
          "gaze_point": [x, y, z],
          "dist_to_cam": float,
          "prime_gap": float
        }
      },
      "end": {
        "frame": integer,
        "3d_location": [x, y, z],
        "prime_stats": {
          "prime_window_start": integer,
          "frame_primed": integer,
          "gaze_point": [x, y, z],
          "dist_to_cam": float,
          "prime_gap": float
        }
      }
    },
    ...
  }
}
```

**Field Descriptions**

- **`video_id`**: The name of the video (e.g. `P01-20240202-110250`).  
- **`object_id`**: A string identifier for the object in the scene (e.g. `"0"`).  
- **`start` / `end`**: Contain data for the pick-up and put-down events of the object, respectively.  
  - **`frame`**: The frame number when the object is picked up or put down.  
  - **`3d_location`**: The 3D world coordinates \([x, y, z]\) of the object at pick-up or put-down.  
  - **`prime_stats`**: Metadata related to the priming event:  
    - **`prime_window_start`**: The frame at which the priming window begins.  
    - **`frame_primed`**: The frame when gaze priming was detected:  
      - `>= 0`: The exact frame of priming.  
      - `-1`: The location was valid, but no priming occurred.  
      - `-2`: The sample was excluded (e.g. off-screen movement or ongoing object manipulation).  
    - **`gaze_point`**: The 3D location where gaze intersects the object’s bounding box, or the closest point to its centre if no direct intersection occurred.  
    - **`dist_to_cam`**: The Euclidean distance from the object to the camera wearer at the time of priming.  
    - **`prime_gap`**: Time in seconds between the priming frame and the interaction frame.  

## High Level

This contains the high level activities as well as recipe and nutrition information

### activities / PXX_recipe_timestamps.csv

**Field Descriptions**:
- **`video_id`**: A unique identifier for the video ID, i.e. `P01-20240202-110250`.
- **`recipe_id`**: If the activity is part of the recipe, the recipe ID (for this participant) is noted. Leave empty for background activities.
- **`high_level_activity_label`**: General description of high level activity.

### complete_recipes.json

**Field Descriptions**:
- A unique identifier for each recipe formed of PXX-RYY, where XX is the participant id and YY is the recipe ID, unique for that participant
- **`participant`**: Participant ID.
- **`name`**: Name of that recipe.
- **`type`**: Indicates whether the recipe is available as is online, or has been modified/adapted from an online or written source
- **`source`**: A link to the online recipe before adaptation. Note that these links might no longer be available if the recipe is taken down from source.
- **`steps`**: The ordered free form steps (as done by the participant, so could be modified from the source). Each step has a unique step ID
- **`captures`**: If the recipe is done multiple times, then each is considered a separate capture. This is the case for a few recipes like coffee and cereal breakfast.
   - **`videos`**: These are the one or more videos that contain the steps of this recipe
   - **`ingredients`**: The list of ingredients and their nutrition. Note that the nutrition might differ across captures.
      - A unique ingredient ID
      - **`name`**: name of the ingredient in free form
      - **`amount`**: If known, the amount of the ingredient added to the recipe.
      - **`amount_unit`**: whether the measurement is in units, grams, ml, ...
      - **`calories`**: the amount of calories of this ingredient in the amount specified.
      - **`carbs`**: carbs
      - **`fat`**: fat
      - **`protein`**: protein
      - **`weigh`**: the segments in the videos of when this ingredient is weighed - whether on the digital scale or through another measurement (e.g. spoon)
      - **`add`**: the segments in the videos when this ingredient is added to the recipe.

## Audio annotations

This folder contains audio annotations HD_EPIC_Sounds (in csv and pkl) structured as follows: 

### `HD-EPIC-Sounds.csv`

This CSV file contains the sound annotations for HD-EPIC and contains 9 columns:

| Column Name           | Type                       | Example                     | Description                                                                   |
| --------------------- | -------------------------- | --------------------------- | ----------------------------------------------------------------------------------- |
| `participant_id`      | string                     | `P01`                       | ID of the participant (unique per participant).                                     |
| `video_id`            | string                     | `P01-20240202-110250`       | ID of the video where the segment originated from (unique per video).               |
| `start_timestamp`     | string                     | `00:00:00.476`              | Start time in `HH:mm:ss.SSS` of the audio annotation.                               |
| `stop_timestamp`      | string                     | `00:00:02.520`              | End time in `HH:mm:ss.SSS` of the audio annotation.                                 |
| `start_sample`        | int                        | `22848`                     | Index of the start audio sample (48KHz) in the untrimmed audio of `video_id`.  |
| `stop_sample`         | int                        | `120960`                    | Index of the stop audio sample (48KHz) in the untrimmed audio of `video_id`.        |
| `class`               | string                     | `rustle`                    | Assigned class name.                                                                |
| `class_id`            | int                        | `4`                         | Numeric ID of the class.                                                      |

## VQA-benchmark

These JSON files contain all the questions for our benchmark, with each file containing the questions for one question prototype

**Field Descriptions**:
- **`inputs`**: The visual input for the question and any bounding boxes. This could be one or more videos, one or more clips and optionally one bounding box.
- **`question`**: The question in the VQA
- **`choices`**: The 5-option choices
- **`correct_idx`**: The index (start from 0) of the correct answer.

## Youtube Links

This contains the links to all videos of the dataset. Notice that YouTube introduces artifacts to the videos, so these should only be used for viewing the videos. Please download the videos themselves from our [webpage](https://hd-epic.github.io/index#download) in the full quality to do any processing or replicate the VQA results

### `HD_EPIC_VQA_Interface.html`

An interface to visualise all our VQA questions

**Contact:** [uob-epic-kitchens@bristol.ac.uk](mailto:uob-epic-kitchens@bristol.ac.uk)
