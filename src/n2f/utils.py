import re


def strip_markdown_json(text: str) -> str:
    code_block_pattern = r"```json\s*(.*?)\s*```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text
