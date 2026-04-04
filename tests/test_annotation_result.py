from n2f.annotation_result import AnnotationResult


def test_annotation_result_from_json() -> None:
    json_string = '{"detections": [{"bounding_box": [214, 540, 480, 920], "label": "Chuck Norris"}]}'
    annotation_result = AnnotationResult.from_json(json_string)
    assert len(annotation_result.annotated_bounding_boxes) == 1
    annotated_bounding_box = annotation_result.annotated_bounding_boxes[0]
    assert annotated_bounding_box.bounding_box.x_min == 214
    assert annotated_bounding_box.bounding_box.y_min == 540
    assert annotated_bounding_box.bounding_box.x_max == 480
    assert annotated_bounding_box.bounding_box.y_max == 920
    assert annotated_bounding_box.label == "Chuck Norris"
