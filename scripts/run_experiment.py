import argparse
from pathlib import Path
from loguru import logger

from n2f.qwen2_5_vl_model import Qwen2_5_VLModel
from n2f.openai_model import OpenAIModel
from n2f.prompt import AnnotatePrompt
from n2f.annotation_result import AnnotationResult


def main() -> None:
    args = parse_arguments()

    # TODO: model factory
    if args.command == "local":
        model = Qwen2_5_VLModel(model_path=args.model_path)
    elif args.command == "remote":
    model = OpenAIModel(
        api_key=args.api_key,
        model_name=args.model_name,
    )

    prompt = AnnotatePrompt()
    response = model.predict(
        prompt.render(),
        [Path("./data/test_image.jpg")],
        max_tokens=args.max_tokens,
    )

    try:
        annotation_result = AnnotationResult.from_json(response)
        print(annotation_result)
    except KeyError as e:
        print(f"Failed to parse annotation result: {e}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    local_parser = subparsers.add_parser(
        "local",
        parents=[shared],
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
    remote_parser.add_argument("--model-name", type=str, default="gpt-5-nano")
    remote_parser.add_argument(
        "--api-key",
        type=str,
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
