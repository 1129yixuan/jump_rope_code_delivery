import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "evaluation_records.csv"
OUT = ROOT / "data.js"
THRESHOLD = 10


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def as_number(value):
    if is_number(value):
        return value
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def level_for(actual):
    if actual <= 100:
        return "Low 0-100"
    if actual <= 129:
        return "Intermediate 101-129"
    return "High 130+"


def summarize_accuracy(rows, key):
    valid = [row for row in rows if is_number(row.get(f"{key}AbsError"))]
    if not valid:
        return {
            "n": 0,
            "ok10": 0,
            "acc10": 0,
            "ok5": 0,
            "acc5": 0,
            "mae": 0,
            "medianAbs": 0,
            "bias": 0,
            "maxAbs": 0,
            "over10": 0,
            "under10": 0,
            "severe30": 0,
        }
    errors = [row[f"{key}Error"] for row in valid]
    abs_errors = [abs(error) for error in errors]
    ok10 = sum(value <= 10 for value in abs_errors)
    ok5 = sum(value <= 5 for value in abs_errors)
    return {
        "n": len(valid),
        "ok10": ok10,
        "acc10": ok10 / len(valid),
        "ok5": ok5,
        "acc5": ok5 / len(valid),
        "mae": sum(abs_errors) / len(abs_errors),
        "medianAbs": median(abs_errors),
        "bias": sum(errors) / len(errors),
        "maxAbs": max(abs_errors),
        "over10": sum(error > 10 for error in errors),
        "under10": sum(error < -10 for error in errors),
        "severe30": sum(value > 30 for value in abs_errors),
    }


def summarize_diff(rows):
    valid = [row for row in rows if is_number(row.get("diff"))]
    if not valid:
        return {
            "n": 0,
            "same": 0,
            "newHigher": 0,
            "newLower": 0,
            "avgDiff": 0,
            "medianDiff": 0,
            "avgAbsDiff": 0,
            "maxAbsDiff": 0,
            "gt10": 0,
            "gt20": 0,
            "gt30": 0,
        }
    diffs = [row["diff"] for row in valid]
    abs_diffs = [abs(diff) for diff in diffs]
    return {
        "n": len(valid),
        "same": sum(diff == 0 for diff in diffs),
        "newHigher": sum(diff > 0 for diff in diffs),
        "newLower": sum(diff < 0 for diff in diffs),
        "avgDiff": sum(diffs) / len(diffs),
        "medianDiff": median(diffs),
        "avgAbsDiff": sum(abs_diffs) / len(abs_diffs),
        "maxAbsDiff": max(abs_diffs),
        "gt10": sum(diff > 10 for diff in abs_diffs),
        "gt20": sum(diff > 20 for diff in abs_diffs),
        "gt30": sum(diff > 30 for diff in abs_diffs),
    }


def group_summary(rows, field):
    result = []
    values = sorted({row.get(field) for row in rows}, key=lambda item: str(item))
    for value in values:
        group = [row for row in rows if row.get(field) == value]
        result.append(
            {
                "group": value,
                "n": len(group),
                "original": summarize_accuracy(group, "original"),
                "new": summarize_accuracy(group, "new"),
                "tcnPeak": summarize_accuracy(group, "tcnPeak"),
                "diff": summarize_diff(group),
            }
        )
    return result


def video_sort_key(item):
    area, video, lane, file_name = item[0]
    digits = "".join(ch for ch in str(file_name) if ch.isdigit())
    return (str(area), int(digits or 0), str(video), str(lane))


def video_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["area"], row["video"], row["lane"], row["file"])].append(row)
    result = []
    for (area, video, lane, file_name), group in sorted(groups.items(), key=video_sort_key):
        result.append(
            {
                "area": area,
                "video": video,
                "lane": lane,
                "file": file_name,
                "n": len(group),
                "actualN": sum(is_number(row.get("actual")) for row in group),
                "original": summarize_accuracy(group, "original"),
                "new": summarize_accuracy(group, "new"),
                "tcnPeak": summarize_accuracy(group, "tcnPeak"),
                "diff": summarize_diff(group),
            }
        )
    return result


def histogram(values, bins):
    counts = []
    for label, lower, upper in bins:
        if upper is None:
            count = sum(value >= lower for value in values)
        else:
            count = sum(lower <= value < upper for value in values)
        counts.append({"label": label, "count": count})
    return counts


def build_record(row_index, row):
    actual = as_number(row.get("actual_count"))
    original = as_number(row.get("shoulder_count"))
    routed = as_number(row.get("routing_count"))
    tcn = as_number(row.get("tcn_count"))
    file_name = str(row.get("file_name") or "")
    video = str(row.get("video_id") or file_name.replace(".mkv", ""))
    zone = as_number(row.get("position"))
    area = str(row.get("venue") or "")
    record = {
        "id": row.get("record_id") or f"record-{row_index}",
        "area": area,
        "row": as_number(row.get("source_row")) or row_index,
        "file": file_name,
        "video": video,
        "lane": area.replace("Venue ", ""),
        "zone": zone,
        "evaluationSplit": row.get("evaluation_split") or "",
        "original": original,
        "new": routed,
        "tcnPeak": tcn,
        "actual": actual,
    }
    if is_number(original) and is_number(routed):
        record["diff"] = routed - original
        record["absDiff"] = abs(record["diff"])
    if is_number(routed) and is_number(tcn):
        record["tcnPeakDiff"] = tcn - routed
        record["tcnPeakAbsDiff"] = abs(record["tcnPeakDiff"])
    if is_number(actual):
        record["level"] = level_for(actual)
        for key in ("original", "new", "tcnPeak"):
            if is_number(record.get(key)):
                error = record[key] - actual
                record[f"{key}Error"] = error
                record[f"{key}AbsError"] = abs(error)
                record[f"{key}Acc10"] = abs(error) <= 10
                record[f"{key}Acc5"] = abs(error) <= 5
    return record


def load_records():
    with DATASET.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "record_id",
        "venue",
        "video_id",
        "file_name",
        "position",
        "actual_count",
        "shoulder_count",
        "routing_count",
        "tcn_count",
        "evaluation_split",
        "source_row",
    }
    missing = sorted(required - set(rows[0] if rows else ()))
    if missing:
        raise ValueError(f"Evaluation dataset is missing columns: {', '.join(missing)}")
    return [build_record(index, row) for index, row in enumerate(rows, start=2)]


def main():
    records = [
        row
        for row in load_records()
        if is_number(row.get("actual")) and is_number(row.get("tcnPeak"))
    ]
    actual_rows = [row for row in records if is_number(row.get("actual"))]
    output_rows = [row for row in records if any(is_number(row.get(key)) for key in ("original", "new", "tcnPeak"))]
    tcn_rows = [row for row in actual_rows if is_number(row.get("tcnPeak"))]
    status_counts = Counter((row.get("newAcc10"), row.get("tcnPeakAcc10")) for row in tcn_rows)
    output_abs_diffs = [abs(row["diff"]) for row in output_rows if is_number(row.get("diff"))]

    data = {
        "meta": {
            "sourceDataset": "analysis_dashboard/evaluation_records.csv",
            "generatedAt": "2026-08-28",
            "threshold": THRESHOLD,
            "scope": "Venue 1 and Venue 2 records with both ground truth and TCN output",
            "evaluationNote": "Venue 1 uses out-of-fold cross-validation predictions; Venue 2 uses independent predictions.",
            "actualRows": len(actual_rows),
            "outputRows": len(output_rows),
            "tcnPeakRows": len(tcn_rows),
        },
        "actualRows": actual_rows,
        "outputRows": output_rows,
        "summaries": {
            "actual": {
                "original": summarize_accuracy(actual_rows, "original"),
                "new": summarize_accuracy(actual_rows, "new"),
                "tcnPeak": summarize_accuracy(actual_rows, "tcnPeak"),
                "tcnPeakStatus": {
                    "bothCorrect": status_counts[(True, True)],
                    "newWrongTcnPeakCorrect": status_counts[(False, True)],
                    "newCorrectTcnPeakWrong": status_counts[(True, False)],
                    "bothWrong": status_counts[(False, False)],
                },
                "tcnPeakPaired": {
                    "original": summarize_accuracy(tcn_rows, "original"),
                    "new": summarize_accuracy(tcn_rows, "new"),
                    "tcnPeak": summarize_accuracy(tcn_rows, "tcnPeak"),
                },
                "byArea": group_summary(actual_rows, "area"),
                "byLevel": group_summary(actual_rows, "level"),
                "byZone": group_summary(actual_rows, "zone"),
                "byVideo": video_summary(actual_rows),
            },
            "output": {
                "diff": summarize_diff(output_rows),
                "byArea": group_summary(output_rows, "area"),
                "byZone": group_summary(output_rows, "zone"),
                "byVideo": video_summary(output_rows),
                "histogram": histogram(
                    output_abs_diffs,
                    [
                        ("0", 0, 1),
                        ("1-5", 1, 6),
                        ("6-10", 6, 11),
                        ("11-20", 11, 21),
                        ("21-30", 21, 31),
                        (">30", 31, None),
                    ],
                ),
            },
        },
    }
    OUT.write_text("window.DASHBOARD_DATA = " + json.dumps(data, ensure_ascii=True, indent=2) + ";\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"actualRows={len(actual_rows)} outputRows={len(output_rows)}")


if __name__ == "__main__":
    main()
