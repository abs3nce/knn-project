import n2f.utils as utils


def test_strip_markdown_json() -> None:
    input_string = """```json{"key": "value","number": 123,"array": [1, 2, 3]}```"""
    expected_output = """{"key": "value","number": 123,"array": [1, 2, 3]}"""
    assert utils.strip_markdown_json(input_string) == expected_output
