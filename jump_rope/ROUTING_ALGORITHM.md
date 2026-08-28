# Routing Algorithm Guide

This guide explains how to count rope jumps for each position from existing pose-estimation JSON files with the rule-based routing algorithm.

## 1. Entry Point

The examples below assume that you first enter the project root and define `PROJECT_ROOT`:

```bash
cd /path/to/jump_rope_code_delivery
export PROJECT_ROOT="$PWD"
```

Routing entry point:

```text
$PROJECT_ROOT/skip-NeuroNetwork/CountOri_routing/scripts/count_video_people.py
```

Use a Python environment with `numpy` installed. The top-level dependencies can be installed with:

```bash
python -m pip install -r "$PROJECT_ROOT/requirements.txt"
```

## 2. Required Input

### 2.1 Pose JSON

The input must be pose-estimation JSON, not a raw video. Each file must contain:

- A `frames` list
- A `persons` list in each frame
- A `bbox` for each person
- A `keypoints` array for each person
- At least the COCO shoulder, hip, ankle, and wrist keypoints

Recommended external data directories:

```text
$PROJECT_ROOT/jump_rope/clips01_pose_json
$PROJECT_ROOT/jump_rope/clips02_pose_json
```

These large data directories are not included in this delivery package.

### 2.2 Zone Configuration

The routing algorithm uses rectangular image regions to associate a person with a competition position:

```text
$PROJECT_ROOT/skip-NeuroNetwork/CountOri_routing/config/zones-1.json
$PROJECT_ROOT/skip-NeuroNetwork/CountOri_routing/config/zones-2.json
```

Use `zones-1.json` for Venue 1 and `zones-2.json` for Venue 2. Each zone contains `id`, `x1`, `y1`, `x2`, and `y2`. The algorithm assigns a person to a zone from the ankle location.

## 3. Run One JSON File

Example for one Venue 2 file:

```bash
mkdir -p "$PROJECT_ROOT/jump_rope/outputs"

python \
  "$PROJECT_ROOT/skip-NeuroNetwork/CountOri_routing/scripts/count_video_people.py" \
  "$PROJECT_ROOT/jump_rope/clips02_pose_json/214_30s_41_2.json" \
  --zones "$PROJECT_ROOT/skip-NeuroNetwork/CountOri_routing/config/zones-2.json" \
  --count-mode shoulder_ankle \
  --counts-output "$PROJECT_ROOT/jump_rope/outputs/214_30s_41_2.counts.json"
```

Important arguments:

- Positional JSON path: pose-estimation JSON to count
- `--zones`: zone configuration for the matching venue
- `--count-mode shoulder_ankle`: recommended routing mode
- `--count-mode shoulder`: original shoulder-only count without routing
- `--counts-output`: output path for the summary JSON
- `--zone 1`: optional filter that prints only one position

## 4. Batch Run Venue 1

```bash
OUT="$PROJECT_ROOT/jump_rope/outputs/routing_results_venue_1"
mkdir -p "$OUT"

for json in "$PROJECT_ROOT"/jump_rope/clips01_pose_json/*.json; do
  base="$(basename "$json" .json)"
  python \
    "$PROJECT_ROOT/skip-NeuroNetwork/CountOri_routing/scripts/count_video_people.py" \
    "$json" \
    --zones "$PROJECT_ROOT/skip-NeuroNetwork/CountOri_routing/config/zones-1.json" \
    --count-mode shoulder_ankle \
    --counts-output "$OUT/$base.counts.json"
done
```

## 5. Batch Run Venue 2

```bash
OUT="$PROJECT_ROOT/jump_rope/outputs/routing_results_venue_2"
mkdir -p "$OUT"

for json in "$PROJECT_ROOT"/jump_rope/clips02_pose_json/*.json; do
  base="$(basename "$json" .json)"
  python \
    "$PROJECT_ROOT/skip-NeuroNetwork/CountOri_routing/scripts/count_video_people.py" \
    "$json" \
    --zones "$PROJECT_ROOT/skip-NeuroNetwork/CountOri_routing/config/zones-2.json" \
    --count-mode shoulder_ankle \
    --counts-output "$OUT/$base.counts.json"
done
```

## 6. Output Format

`--counts-output` creates a `.counts.json` file with these top-level fields:

- `inference_json`: input pose JSON
- `count_mode`: selected counting mode
- `routing_version`: routing implementation version
- `results`: one result object per zone

Frequently used fields in each zone result:

- `id`: zone identifier, such as `zone_1`
- `jump_count`: final routed count
- `frames_with_person`: number of frames with a person assigned to the zone
- `raw_primary_count`: unmodified primary-path count
- `routing_level`: selected routing level
- `selected_algorithm_path`: candidate path selected for the final result
- `algorithm_counts`: all candidate counts
- `diagnostics`: shoulder, ankle, wrist, and related diagnostic features
- `routing_reasons`: reasons for the selected path

`algorithm_counts` contains these candidate paths:

```text
low
middle
high
zone_aware
shoulder
shoulder_ankle_no_zoneaware
ankle_only
```

For a spreadsheet export, the usual mapping is:

```text
position = id
count = jump_count
```

## 7. Troubleshooting

### Missing `numpy`

Install the project dependencies or activate the intended environment:

```bash
python -m pip install -r "$PROJECT_ROOT/requirements.txt"
```

### Do Not Mix Venue Configurations

Using `zones-2.json` with `clips01_pose_json`, or the reverse, assigns people to incorrect image regions and produces invalid counts.

### A Zone Is Missing or Has an Unusually Low Count

Check the following:

- The input JSON is complete
- `frames` is not empty
- The person's ankle location falls inside the expected zone
- The correct `zones-1.json` or `zones-2.json` is used
- Video resolution matches `frame_width` and `frame_height` in the zone configuration
