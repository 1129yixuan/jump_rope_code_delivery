# TCN Model Training and Prediction Guide

This guide explains how to train the TCN model and use a trained checkpoint on a new test set.

The current workflow is:

```text
manual per-jump intervals -> frame-level heatmap -> per-frame TCN probabilities -> peak counting
```

Only `peak_count` is used. Integral-based counting is no longer part of this workflow.

## 1. Scripts

The examples below assume that you first enter the project root and define `PROJECT_ROOT`:

```bash
cd /path/to/jump_rope_code_delivery
export PROJECT_ROOT="$PWD"
```

Training script:

```text
$PROJECT_ROOT/jump_rope/train_labelclip_tcn.py
```

Prediction script:

```text
$PROJECT_ROOT/jump_rope/predict_labelclip.py
```

Install the required packages with:

```bash
python -m pip install -r "$PROJECT_ROOT/requirements.txt"
```

## 2. Training Data

### 2.1 Pose JSON

Training input is a directory of pose-estimation JSON files, not raw videos:

```text
$PROJECT_ROOT/jump_rope/clips01_pose_json
$PROJECT_ROOT/jump_rope/clips02_pose_json
```

Each JSON file must contain `frames`, per-frame `persons`, each person's `bbox`, and `keypoints`.

### 2.2 Manual Per-Jump Labels

Recommended annotation directory:

```text
$PROJECT_ROOT/jump_rope/annotations
```

Annotation files normally use this naming convention:

```text
video_name.jump.json
```

Typical structure:

```json
{
  "zones": {
    "1": [
      {"start": 120, "end": 128, "label": 1}
    ],
    "2": []
  }
}
```

Training reads intervals by `video + zone`. The center of every accepted interval becomes a peak in the target heatmap.

### 2.3 Matching File Names

Pose JSON:

```text
214_30s_41_2.json
```

Matching annotation:

```text
214_30s_41_2.jump.json
```

The loader can also search a venue-named subdirectory, but placing annotations together in `annotations/` is the most predictable layout.

## 3. Train a Model

### 3.1 Recommended Venue 1 Training Command

```bash
cd "$PROJECT_ROOT/jump_rope"

python \
  "$PROJECT_ROOT/jump_rope/train_labelclip_tcn.py" \
  --pose-json-root "$PROJECT_ROOT/jump_rope/clips01_pose_json" \
  --labelclip-root "$PROJECT_ROOT/jump_rope/annotations" \
  --out "$PROJECT_ROOT/jump_rope/outputs/tcn_clips01_train" \
  --save-model "$PROJECT_ROOT/jump_rope/outputs/tcn_clips01_train/model.pt" \
  --length 1024 \
  --folds 5 \
  --epochs 80 \
  --patience 15 \
  --batch-size 16
```

Force CPU on macOS with `--device cpu`, or force Apple Silicon MPS with `--device mps`.

### 3.2 Important Training Arguments

- `--pose-json-root`: one or more pose JSON directories
- `--labelclip-root`: manual per-jump annotation directory
- `--out`: training output directory
- `--save-model`: final prediction checkpoint path
- `--length 1024`: resampled sequence length
- `--folds 5`: grouped cross-validation fold count
- `--epochs 80`: maximum epochs per fold
- `--patience 15`: early-stopping patience
- `--batch-size 16`: training batch size
- `--peak-threshold-min`, `--peak-threshold-max`, `--peak-threshold-steps`: peak-threshold search range
- `--peak-min-distances`: candidate minimum peak distances
- `--min-labelclip-count 1.0`: exclude samples with fewer than one labeled interval

### 3.3 Training Output

The training output directory normally contains:

```text
metrics.json
predictions.csv
model.pt
```

- `metrics.json`: overall metrics, fold metrics, and selected peak parameters
- `predictions.csv`: peak-count predictions for training and validation samples
- `model.pt`: checkpoint used for future prediction

Common `predictions.csv` fields include `sample_id`, `json`, `video`, `position`, `actual`, `peak_count`, `error`, `abs_error`, `fold`, `threshold`, and `min_distance`.

## 4. Predict a New Test Set

### 4.1 Predict Venue 2

```bash
cd "$PROJECT_ROOT/jump_rope"

python \
  "$PROJECT_ROOT/jump_rope/predict_labelclip.py" \
  --model "$PROJECT_ROOT/jump_rope/outputs/tcn_clips01_train/model.pt" \
  --inputs "$PROJECT_ROOT/jump_rope/clips02_pose_json" \
  --out-dir "$PROJECT_ROOT/jump_rope/outputs/tcn_predict_clips02" \
  --device cpu
```

### 4.2 Predict with Ground-Truth Intervals

Add `--gt-labelclip-dir` to produce interval-matching output and accuracy summaries:

```bash
python \
  "$PROJECT_ROOT/jump_rope/predict_labelclip.py" \
  --model "$PROJECT_ROOT/jump_rope/outputs/tcn_clips01_train/model.pt" \
  --inputs "$PROJECT_ROOT/jump_rope/clips02_pose_json" \
  --out-dir "$PROJECT_ROOT/jump_rope/outputs/tcn_predict_clips02_with_gt" \
  --gt-labelclip-dir "$PROJECT_ROOT/jump_rope/annotations" \
  --device cpu
```

This adds comparison files for inspecting predicted peaks against manually labeled intervals.

### 4.3 Predict One JSON File

```bash
python \
  "$PROJECT_ROOT/jump_rope/predict_labelclip.py" \
  --model "$PROJECT_ROOT/jump_rope/outputs/tcn_clips01_train/model.pt" \
  --inputs "$PROJECT_ROOT/jump_rope/clips02_pose_json/214_30s_41_2.json" \
  --out-dir "$PROJECT_ROOT/jump_rope/outputs/tcn_predict_one_json" \
  --device cpu
```

## 5. Prediction Output

Common output paths:

```text
predictions.csv
predicted_labelclips/
```

Important `predictions.csv` fields:

- `json`: input pose JSON
- `video`: video name
- `position`: position or zone
- `track_id`: predicted person track ID
- `peak_count`: final TCN count
- `threshold`: peak threshold used for prediction
- `min_distance`: minimum allowed distance between peaks
- `frame_count`: original frame count

`predicted_labelclips/` contains exported `.jump.json` files with a frame interval for each predicted peak. To generate only CSV output, use `--no-export-labelclips`.

## 6. Override Peak Parameters

Prediction uses the best peak parameters stored in the checkpoint by default. Override them with:

```text
--peak-threshold 0.45
--peak-min-distance 5
```

Complete example:

```bash
python \
  "$PROJECT_ROOT/jump_rope/predict_labelclip.py" \
  --model "$PROJECT_ROOT/jump_rope/outputs/tcn_clips01_train/model.pt" \
  --inputs "$PROJECT_ROOT/jump_rope/clips02_pose_json" \
  --out-dir "$PROJECT_ROOT/jump_rope/outputs/tcn_predict_clips02_threshold045" \
  --peak-threshold 0.45 \
  --peak-min-distance 5 \
  --device cpu
```

## 7. Troubleshooting

### Missing `torch`, `numpy`, or `openpyxl`

```bash
python -m pip install -r "$PROJECT_ROOT/requirements.txt"
```

### `feature_dim` Mismatch During Prediction

The checkpoint feature dimension does not match the input JSON. Common causes include different pose schemas, 17-keypoint versus extended-keypoint data, or a checkpoint saved by another script version.

### Some Samples Are Excluded from Training

The trainer excludes samples with insufficient pose coverage or fewer intervals than `--min-labelclip-count`. Review the exclusion summary in `metrics.json`.

### Predicting Without Manual Labels

Manual labels are not required for inference. Omit `--gt-labelclip-dir`; the script will still generate `predictions.csv` and, unless disabled, `predicted_labelclips/`.
