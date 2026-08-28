# Evaluation Dataset

`evaluation_records.csv` contains the de-identified records used by the analysis dashboard. Every row has a ground-truth score and outputs from the shoulder, routing, and TCN methods.

## Coverage

- Venue 1: 250 positions from 52 videos
- Venue 2: 245 positions from 49 videos
- Combined: 495 evaluated positions

Venue 1 TCN counts are out-of-fold predictions from grouped five-fold cross-validation. Venue 2 TCN counts are independent predictions produced by the Venue 1-trained model. The dashboard reports combined results and provides venue-level breakdowns so the two evaluation splits can also be inspected separately.

## Fields

| Field | Description |
| --- | --- |
| `record_id` | Stable de-identified record key |
| `venue` | Venue 1 or Venue 2 |
| `video_id` | De-identified source video identifier |
| `file_name` | Competition video file name without participant information |
| `position` | Position number within the video, from 1 to 5 |
| `actual_count` | Manually verified ground-truth count |
| `shoulder_count` | Original shoulder-algorithm output |
| `routing_count` | New routing-algorithm output |
| `tcn_count` | TCN peak-count output |
| `evaluation_split` | Out-of-fold cross-validation or independent evaluation |
| `source_row` | Row number in the analytical source table |

Participant names, schools, local file paths, and other personal or host-specific fields are intentionally excluded from the public delivery.
