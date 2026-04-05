import json
from typing import TypedDict


class BoundingBoxLabelItem(TypedDict):
    bounding_box: list[int]
    label: str


class ValidationResults(TypedDict):
    iou_threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    name_predictions_correct: int
    f1_score: float
    iou_histogram: list[int]
    ious: list[float]


def load_ground_truth_boxes(image_path: str) -> list[BoundingBoxLabelItem]:
    page_folder = image_path.split("/")[:-1]
    page_id = page_folder[-1]
    page_folder = "/".join(page_folder)
    ground_truth: list[BoundingBoxLabelItem] = []
    with open("../" + page_folder + "/" + page_id + "_faces.jsonl", "r") as f2:
        for line2 in f2:
            page_data = json.loads(line2)
            page_width = page_data["page_width"]
            page_height = page_data["page_height"]
            page_left = page_data["page_left"]
            page_top = page_data["page_top"]
            width = page_data["width"]
            height = page_data["height"]
            xmin = int((page_left / page_width) * 1000)
            xmax = int(((page_left + width) / page_width) * 1000)
            ymin = int((page_top / page_height) * 1000)
            ymax = int(((page_top + height) / page_height) * 1000)
            bounding_box = [xmin, ymin, xmax, ymax]
            ground_truth.append(
                {"bounding_box": bounding_box, "label": page_data["person_name"]}
            )
    return ground_truth


def print_bounding_boxes(
    predictions: list[BoundingBoxLabelItem], ground_truth: list[BoundingBoxLabelItem]
):
    print("Predictions:")
    for item in predictions:
        print(item["bounding_box"])
        print(item["label"])

    print("Ground truth:")
    for item in ground_truth:
        print(item["bounding_box"])
        print(item["label"])


def calculate_iou(boxA: list[int], boxB: list[int]) -> float:
    inter_xmin = max(boxA[0], boxB[0])
    inter_ymin = max(boxA[1], boxB[1])
    inter_xmax = min(boxA[2], boxB[2])
    inter_ymax = min(boxA[3], boxB[3])

    # Calculate the area of intersection
    inter_area = max(0, inter_xmax - inter_xmin + 1) * max(
        0, inter_ymax - inter_ymin + 1
    )

    # Calculate the areas of both bounding boxes
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Calculate IoU
    iou = inter_area / float(boxA_area + boxB_area - inter_area)

    return iou


def calculate_confusion_matrix(
    predictions: list[BoundingBoxLabelItem],
    ground_truth: list[BoundingBoxLabelItem],
    iou_threshold: float,
) -> tuple[int, int, int, int, list[float]]:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    name_predictions_correct = 0
    ious: list[float] = []

    for pred in predictions:
        pred_box = pred["bounding_box"]
        pred_label = pred["label"]
        matched_gt = None
        for gt in ground_truth:
            gt_box = gt["bounding_box"]
            gt_label = gt["label"]
            iou = calculate_iou(pred_box, gt_box)
            ious.append(iou)
            if iou >= iou_threshold:
                matched_gt = gt
                if pred_label == gt_label:
                    name_predictions_correct += 1
                break
        if matched_gt:
            true_positives += 1
            ground_truth.remove(
                matched_gt
            )  # Remove matched GT to prevent double counting
        else:
            false_positives += 1

    false_negatives = len(ground_truth)  # Remaining GT boxes are false negatives

    return (
        true_positives,
        false_positives,
        false_negatives,
        name_predictions_correct,
        ious,
    )


def calculate_f1_score(tp: int, fp: int, fn: int) -> float:
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


# encapsulate in main
def main() -> None:
    iou_threshold = 0.75

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    # TN is not needed

    name_predictions_correct = 0

    ious: list[float] = []

    print("Evaluating model on dataset with IoU threshold:", iou_threshold)

    with open("../data/qwen2_5_vl_3b.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            result = data["annotation_result"]
            predictions: list[BoundingBoxLabelItem] = result["detections"]
            ground_truth = load_ground_truth_boxes(data["image_path"])

            # print_bounding_boxes(predictions, ground_truth)

            tp, fp, fn, name_correct, iou_hist = calculate_confusion_matrix(
                predictions, ground_truth, iou_threshold
            )
            true_positives += tp
            false_positives += fp
            false_negatives += fn
            name_predictions_correct += name_correct
            ious.extend(iou_hist)

        print(f"TP: {true_positives}")
        print(f"FP: {false_positives}")
        print(f"FN: {false_negatives}")
        print(f"Name predictions correct: {name_predictions_correct}")

        f1 = calculate_f1_score(true_positives, false_positives, false_negatives)
        print(f"F1 Score: {f1}")

        histogram = [0] * 10
        for iou in ious:
            if iou >= 0 and iou < 0.1:
                histogram[0] += 1
            elif iou >= 0.1 and iou < 0.2:
                histogram[1] += 1
            elif iou >= 0.2 and iou < 0.3:
                histogram[2] += 1
            elif iou >= 0.3 and iou < 0.4:
                histogram[3] += 1
            elif iou >= 0.4 and iou < 0.5:
                histogram[4] += 1
            elif iou >= 0.5 and iou < 0.6:
                histogram[5] += 1
            elif iou >= 0.6 and iou < 0.7:
                histogram[6] += 1
            elif iou >= 0.7 and iou < 0.8:
                histogram[7] += 1
            elif iou >= 0.8 and iou < 0.9:
                histogram[8] += 1
            elif iou >= 0.9 and iou <= 1.0:
                histogram[9] += 1

        print("IoU Histogram:")
        for i in range(10):
            print(f"{i*0.1:.1f}-{(i+1)*0.1:.1f}: {histogram[i]}")

        # save everything into a json file
        results: ValidationResults = {
            "iou_threshold": iou_threshold,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "name_predictions_correct": name_predictions_correct,
            "f1_score": f1,
            "iou_histogram": histogram,
            "ious": ious,
        }

        with open(
            f"../results/qwen2_5_vl_3b_statistics_{iou_threshold}.json", "w"
        ) as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
