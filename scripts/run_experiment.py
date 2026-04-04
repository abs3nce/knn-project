from datetime import datetime
from pathlib import Path
import argparse
import re

from n2f.annotation_result import AnnotationResult
from n2f.model import Model
from n2f.prompt import AnnotatePrompt
from n2f.statistics import Statistics
from n2f.model_factory import ModelFactory


def get_model_identifier(args: argparse.Namespace) -> str:
    if args.command == "local":
        return f"local:{args.model_name}:{args.model_path}"
    return f"remote:{args.provider}:{args.model_name}"


def strip_markdown_json(text: str) -> str:
    code_block_pattern = r"```json\s*(.*?)\s*```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def create_model(args: argparse.Namespace) -> tuple[Model, str]:
    model_identifier = get_model_identifier(args)
    model_factory = ModelFactory()
    if args.command == "remote":
        model = model_factory.create_model(model_identifier, api_key=args.api_key)
    else:
        model = model_factory.create_model(model_identifier)
    return model, model_identifier


def main() -> None:
    args = parse_arguments()
    model, model_name = create_model(args)

    image_path = Path(
        "./data/pages/with_ner/0a0b17fb-4179-11ec-9fc7-00155d012102/0a0b17fb-4179-11ec-9fc7-00155d012102.jpg"
    )

    prompt = AnnotatePrompt()
    start_time = datetime.now()
    response = model.predict(
        prompt.render(),
        [image_path],
        max_tokens=args.max_tokens,
    )
    end_time = datetime.now()

    try:
        annotation_result = AnnotationResult.from_json(
            strip_markdown_json(response.text)
        )
        print(annotation_result)

        statistics = Statistics(
            image_path=image_path,
            image_id="0",
            model=model_name,
            annotation_result=annotation_result,
            raw_response=response.text,
            success=response.success,
            error_message=response.error_message,
            start_timestamp=start_time,
            end_timestamp=end_time,
            total_time=end_time - start_time,
            tokens_used=response.tokens_used,
        )
        print(statistics.to_dict())
    except (KeyError, ValueError) as error:
        print(f"Failed to parse annotation result: {error}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--max-tokens",
        type=int,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    local_parser = subparsers.add_parser(
        "local",
        parents=[shared],
    )
    local_parser.add_argument(
        "--model-name",
        type=str,
        default="qwen2_5_vl",
    )
    local_parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
    )

    remote_parser = subparsers.add_parser(
        "remote",
        parents=[shared],
    )
    remote_parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-5-nano",
    )
    remote_parser.add_argument(
        "--provider",
        type=str,
        default="openai",
    )
    remote_parser.add_argument(
        "--api-key",
        type=str,
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
