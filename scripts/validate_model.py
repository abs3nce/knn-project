"""Validation script for single-person bounding box localization.

This module evaluates model outputs stored in JSONL files where each row contains:
- `expected_bounding_box`: ground truth box in XYXY format
- `annotation_result`: model-predicted box in XYXY format
- `success`: whether the model produced a valid prediction

For each configured input file, metrics are computed for IoU thresholds 0.1..0.9 and
saved into a threshold-oriented JSON output in the results directory.

NOTE: This file was refactored using code generation tool.
"""

import json
from pathlib import Path
from typing import TypedDict

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
INPUT_PATHS = [
    Path("../answers/output_3b_base_all.jsonl"),
    Path("../answers/output_7b_base_all.jsonl"),
    Path("../answers/output_32b_base_all.jsonl"),
    Path("../answers/output_3b_base_test42.jsonl"),
    Path("../answers/output_7b_base_test42.jsonl"),
    Path("../answers/output_32b_base_test42.jsonl"),
    Path("../answers/output_3b_lora_test42.jsonl"),
    Path("../answers/output_7b_lora_test42.jsonl"),
    # Path("../answers/output_32b_lora_test42.jsonl"),
]
IOU_THRESHOLDS = [round(i / 10, 1) for i in range(1, 10)]


class ModelOutputRow(TypedDict):
    """Input JSONL row schema used by the validator."""

    expected_bounding_box: list[int]
    annotation_result: list[int] | None
    success: int


class ValidationResults(TypedDict):
    """Metrics container for one evaluated input file."""

    input_file: str
    thresholds: list[float]
    sample_count: int
    average_iou: float
    success: int
    failure: int
    true_positives: dict[str, int]
    false_positives: dict[str, int]
    false_negatives: dict[str, int]
    true_negatives: dict[str, int]
    precision: dict[str, float]
    recall: dict[str, float]
    f1_score: dict[str, float]


class SampleEvaluation(TypedDict):
    """Per-sample reduced evaluation state used for threshold sweeps."""

    iou: float
    has_prediction: bool
    success: int


def log_step(message: str) -> None:
    """Print a prominent progress message for a major phase."""

    print(f"\n[STEP] {message}")


def log_success(message: str) -> None:
    """Print a success message for completed work."""

    print(f"[OK]   {message}")


def log_info(message: str) -> None:
    """Print an informational message."""

    print(f"[INFO] {message}")


def log_warning(message: str) -> None:
    """Print a warning message for non-fatal issues."""

    print(f"[WARN] {message}")


def calculate_iou(box_a: list[int], box_b: list[int]) -> float:
    """Calculate IoU for two XYXY bounding boxes."""

    inter_xmin = max(box_a[0], box_b[0])
    inter_ymin = max(box_a[1], box_b[1])
    inter_xmax = min(box_a[2], box_b[2])
    inter_ymax = min(box_a[3], box_b[3])

    # Calculate the area of intersection
    inter_area = max(0, inter_xmax - inter_xmin + 1) * max(
        0, inter_ymax - inter_ymin + 1
    )

    # Calculate the areas of both bounding boxes
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # Calculate IoU
    denominator = float(box_a_area + box_b_area - inter_area)
    iou = inter_area / denominator if denominator > 0 else 0.0

    return iou


def parse_model_bbox(
    annotation_result: list[int] | None, success: int
) -> list[int] | None:
    """Parse model bbox from a row or return None for missing/invalid prediction."""

    if not success or annotation_result is None:
        return None

    if len(annotation_result) != 4:
        return None

    return annotation_result


def calculate_precision_recall_f1(
    tp: int, fp: int, fn: int
) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 from confusion counts."""

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def calculate_confusion_matrix(
    samples: list[SampleEvaluation],
    iou_threshold: float,
) -> tuple[int, int, int, int]:
    """Calculate confusion-matrix counts for a given IoU threshold.

    This task assumes one expected face per sample.
    - TP: prediction exists and IoU >= threshold
    - FP: prediction exists and IoU < threshold
    - FN: expected face missed (no prediction) or badly localized prediction
    - TN: always 0 in this setup (no true-negative class in positive-only prompts)
    """

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for sample in samples:
        if sample["has_prediction"] and sample["iou"] >= iou_threshold:
            true_positives += 1
        elif sample["has_prediction"]:
            false_positives += 1
            false_negatives += 1
        else:
            false_negatives += 1

    return true_positives, false_positives, false_negatives, true_negatives


def load_samples(input_path: str) -> list[SampleEvaluation]:
    """Load and reduce one JSONL input file into sample IoUs and prediction flags."""

    samples: list[SampleEvaluation] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            row: ModelOutputRow = json.loads(line)

            expected_box = row["expected_bounding_box"]
            predicted_box = parse_model_bbox(
                row.get("annotation_result"), row.get("success", 0)
            )

            if predicted_box is None:
                samples.append(
                    {
                        "iou": 0.0,
                        "has_prediction": False,
                        "success": row.get("success", 0),
                    }
                )
                continue

            iou = calculate_iou(predicted_box, expected_box)
            samples.append(
                {"iou": iou, "has_prediction": True, "success": row.get("success", 0)}
            )

    return samples


def build_output_path(input_path: Path) -> Path:
    """Build output file path in results directory based on the input filename."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{input_path.stem}_statistics_thresholds.json"


def evaluate_file(input_path: Path, iou_thresholds: list[float]) -> ValidationResults:
    """Evaluate one input file across all IoU thresholds and return metrics."""

    samples = load_samples(str(input_path))
    success_count = sum(1 for sample in samples if sample["success"] == 1)
    failure_count = sum(1 for sample in samples if sample["success"] == 0)
    sample_count = len(samples)
    average_iou = (
        sum(sample["iou"] for sample in samples) / sample_count
        if sample_count > 0
        else 0.0
    )

    log_info(f"Evaluating {sample_count} samples")
    log_info(f"Success 1: {success_count}")
    log_info(f"Success 0: {failure_count}")
    log_info(f"Average IoU: {average_iou:.4f}")
    log_info(f"IoU thresholds: {iou_thresholds}")

    true_positives: dict[str, int] = {}
    false_positives: dict[str, int] = {}
    false_negatives: dict[str, int] = {}
    true_negatives: dict[str, int] = {}
    precision_scores: dict[str, float] = {}
    recall_scores: dict[str, float] = {}
    f1_scores: dict[str, float] = {}

    for threshold in iou_thresholds:
        key = f"{threshold:.1f}"
        tp, fp, fn, tn = calculate_confusion_matrix(samples, threshold)
        precision, recall, f1 = calculate_precision_recall_f1(tp, fp, fn)

        true_positives[key] = tp
        false_positives[key] = fp
        false_negatives[key] = fn
        true_negatives[key] = tn
        precision_scores[key] = precision
        recall_scores[key] = recall
        f1_scores[key] = f1

        log_info(
            f"IoU >= {threshold:.1f}: TP={tp}, FP={fp}, FN={fn}, "
            f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )

    return {
        "input_file": str(input_path),
        "thresholds": iou_thresholds,
        "sample_count": sample_count,
        "average_iou": average_iou,
        "success": success_count,
        "failure": failure_count,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "precision": precision_scores,
        "recall": recall_scores,
        "f1_score": f1_scores,
    }


def main() -> None:
    """Run validation for all configured input files."""

    log_step("Starting validation run")
    log_info(f"Configured input files: {len(INPUT_PATHS)}")

    for input_path in INPUT_PATHS:
        resolved_input = (SCRIPT_DIR / input_path).resolve()
        if not resolved_input.exists():
            log_warning(f"Input file not found, skipping: {resolved_input}")
            continue

        log_step(f"Evaluating file: {resolved_input.name}")
        results = evaluate_file(resolved_input, IOU_THRESHOLDS)

        output_path = build_output_path(resolved_input)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        log_success(f"Saved validation results: {output_path}")

    log_step("Validation run finished")


if __name__ == "__main__":
    main()
