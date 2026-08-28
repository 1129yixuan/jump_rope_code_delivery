# Jump Rope Counting Algorithm Delivery

This directory is the GitHub-ready delivery package exported from the local development workspace. It contains source code, configuration, documentation, a de-identified evaluation dataset, and a prebuilt analysis dashboard.

The package does not include:

- Original videos
- Large pose-estimation JSON files
- Training output directories
- Source evaluation workbooks containing participant information
- Local virtual environments
- Publishing tokens or host-specific credentials

## Project Layout

```text
analysis_dashboard/                         # Dashboard source and prebuilt standalone page
skip-NeuroNetwork/CountOri_routing/         # Rule-based routing counter and zone configuration
jump_rope/                                  # TCN training, prediction, and operating guides
requirements.txt                            # Top-level Python dependencies
```

## Rule-Based Routing Counter

Detailed guide:

```text
jump_rope/ROUTING_ALGORITHM.md
```

Entry point:

```text
skip-NeuroNetwork/CountOri_routing/scripts/count_video_people.py
```

The counter reads pose-estimation JSON, assigns people to configured zones, runs several candidate counters, and selects a final jump count through routing rules.

## TCN Model

Detailed guide:

```text
jump_rope/TCN_MODEL.md
```

Main scripts:

```text
jump_rope/train_labelclip_tcn.py
jump_rope/predict_labelclip.py
jump_rope/train_tcn.py
```

The supervised TCN workflow converts manually labeled jump intervals into a frame-level heatmap, predicts a jump probability for each frame, and counts valid probability peaks.

## Analysis Dashboard

Open the standalone page directly:

```text
analysis_dashboard/public_dashboard.html
```

The page is self-contained and does not require a web server.

The dashboard covers Venue 1 and Venue 2 using every record with both a ground-truth score and a TCN output. The included source dataset is:

```text
analysis_dashboard/evaluation_records.csv
```

It contains only analytical fields and excludes participant names and schools. See `analysis_dashboard/DATASET.md` for field definitions and evaluation-split details.

To rebuild the dashboard from the included CSV:

```bash
python analysis_dashboard/build_data.py
python analysis_dashboard/build_public.py
```

## Dependencies

Install the top-level dependencies with:

```bash
python -m pip install -r requirements.txt
```

The routing subproject also includes its own `pyproject.toml` and `uv.lock`.

## Large Artifacts

Pose JSON data, trained `.pt` checkpoints, and evaluation workbooks should be delivered separately through Git LFS, release attachments, object storage, or another large-file channel instead of a regular Git repository.
