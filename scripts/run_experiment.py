from datetime import datetime
from pathlib import Path
import argparse

from tqdm import tqdm
from loguru import logger

from n2f.annotation_result import AnnotationResult
from n2f.model import Model
from n2f.prompt import AnnotatePrompt
from n2f.statistics import Statistics
from n2f.model_factory import ModelFactory
from n2f.utils import strip_markdown_json


def main() -> None:
    args = parse_arguments()
    initialize_logger(args.jsonl_output_path)
    model_identifier = get_model_identifier(args)
    model = get_model(args, model_identifier)
    dataset_path = args.dataset_path
    prompt = AnnotatePrompt(args.prompt_path)
    for image_path in tqdm(get_dataset_image_paths(dataset_path)):
        statistic = run_model_prediction(
            prompt=prompt,
            model_identifier=model_identifier,
            model=model,
            image_path=image_path,
            max_tokens=args.max_tokens,
        )
        logger.info(statistic.to_json())


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--dataset-path",  # TODO: default dataset path
        type=Path,
        required=True,
    )
    shared.add_argument(
        "--prompt-path",
        type=Path,
        default=Path("prompts/annotate_prompt.j2"),
    )
    shared.add_argument(
        "--jsonl-output-path",
        type=Path,
        default=Path("output.jsonl"),
    )
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


def get_model_identifier(args: argparse.Namespace) -> str:
    if args.command == "local":
        return f"local:{args.model_name}:{args.model_path}"
    return f"remote:{args.provider}:{args.model_name}"  # TODO: elif and error handling


def get_model(args: argparse.Namespace, model_identifier: str) -> Model:
    model_factory = ModelFactory()
    if args.command == "remote":  # TODO: elif and error handling
        model = model_factory.create_model(model_identifier, api_key=args.api_key)
    else:
        model = model_factory.create_model(model_identifier)
    return model


def get_dataset_image_paths(dataset_path: Path) -> list[Path]:
    image_paths: list[Path] = []
    for image_directory in tqdm(dataset_path.iterdir()):
        for image_path in image_directory.iterdir():
            if image_path.is_file() and image_path.suffix.lower() == ".jpg":
                image_paths.append(image_path)
    return image_paths


def initialize_logger(log_path: Path) -> None:
    logger.remove()
    logger.add(log_path, format="{message}")


def run_model_prediction(
    prompt: AnnotatePrompt,
    model_identifier: str,
    model: Model,
    image_path: Path,
    max_tokens: int | None,
) -> Statistics:
    start_timestamp = datetime.now()
    prediction_response = model.predict(
        prompt.render(),
        [image_path],
        max_tokens=max_tokens,
    )
    end_timestamp = datetime.now()

    annotation_result = AnnotationResult.empty()
    success = prediction_response.success
    error_message = prediction_response.error_message

    try:
        annotation_result = AnnotationResult.from_json(
            strip_markdown_json(prediction_response.text)
        )
    except (KeyError, ValueError) as exception:
        success = False
        error_message = str(exception)

    return Statistics(
        image_path=image_path,
        image_id=image_path.stem,
        model=model_identifier,
        prompt_path=prompt.path,
        annotation_result=annotation_result,
        raw_response=prediction_response.text,
        success=success,
        error_message=error_message,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        total_time=end_timestamp - start_timestamp,
        tokens_used=prediction_response.tokens_used,
    )


if __name__ == "__main__":
    main()
