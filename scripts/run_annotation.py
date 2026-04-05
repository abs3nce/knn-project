"""Script for annotating a dataset using a model."""

from datetime import datetime
from pathlib import Path
import argparse

from tqdm import tqdm
from loguru import logger

from n2f.core.annotation_result import AnnotationResult
from n2f.core.prompt import AnnotatePrompt
from n2f.models.model import Model
from n2f.models.model_factory import ModelFactory
from n2f.models.model_identifier import ModelIdentifier
from n2f.utils.statistics import Statistics
from n2f.utils.utils import strip_markdown_json, format_error_message


def main() -> None:
    """Runs the annotation process."""
    arguments = parse_arguments()
    api_key = vars(arguments).get("api_key")

    initialize_logger(arguments.jsonl_output_path)

    model_identifier = get_model_identifier(arguments)
    model_factory = ModelFactory()
    model = model_factory.create_model(
        model_identifier,
        api_key=api_key,
    )

    prompt = AnnotatePrompt(arguments.prompt_path)
    for image_path in tqdm(get_dataset_image_paths(arguments.dataset_path)):
        statistic = run_model_prediction(
            model=model,
            model_identifier=model_identifier,
            prompt=prompt,
            image_path=image_path,
            max_tokens=arguments.max_tokens,
        )
        logger.info(statistic.to_json())


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run annotation on a dataset using a model and prompt.",
    )

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/pages/with_ner/"),
        help="Dataset directory with folders containing .jpg files.",
    )
    shared.add_argument(
        "--prompt-path",
        type=Path,
        default=Path("prompts/annotate_prompt.j2"),
        help="Path to the Jinja2 prompt template.",
    )
    shared.add_argument(
        "--jsonl-output-path",
        type=Path,
        default=Path("output.jsonl"),
        help="Destination JSONL file.",
    )
    shared.add_argument(
        "--max-tokens",
        type=int,
        help="Token limit for the model response.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    local_parser = subparsers.add_parser(
        "local",
        parents=[shared],
        help="Run a local model from disk.",
    )
    local_parser.add_argument(
        "--model-name",
        type=str,
        default="qwen2_5_vl",
        help="Local model registry key.",
    )
    local_parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the local model directory.",
    )

    remote_parser = subparsers.add_parser(
        "remote",
        parents=[shared],
        help="Run a remote API model.",
        description="Run a remote model via API.",
    )
    remote_parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-5-nano",
        help="Remote model name accepted by the provider.",
    )
    remote_parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="Remote model provider (default: openai).",
    )
    remote_parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="API key.",
    )

    return parser.parse_args()


def get_model_identifier(arguments: argparse.Namespace) -> ModelIdentifier:
    """Returns a ModelIdentifier based on the provided command-line arguments."""
    match arguments.command:
        case "local":
            return ModelIdentifier(
                environment_category="local",
                registry_key=arguments.model_name,
                target_value=str(arguments.model_path),
            )
        case "remote":
            return ModelIdentifier(
                environment_category="remote",
                registry_key=arguments.provider,
                target_value=arguments.model_name,
            )
        case unrecognized_command:
            raise ValueError(
                f"Unknown command category provided: '{unrecognized_command}'. "
                f"Expected 'local' or 'remote'."
            )


def get_dataset_image_paths(dataset_path: Path) -> list[Path]:
    """Returns a list of image file paths found in the dataset directory."""
    image_paths: list[Path] = []
    for image_directory in tqdm(dataset_path.iterdir()):
        for image_path in image_directory.iterdir():
            if image_path.is_file() and image_path.suffix.lower() == ".jpg":
                image_paths.append(image_path)
    return image_paths


def initialize_logger(log_path: Path) -> None:
    """Initializes the logger to write to the specified file."""
    logger.remove()
    logger.add(log_path, format="{message}")


def run_model_prediction(
    model: Model,
    model_identifier: ModelIdentifier,
    prompt: AnnotatePrompt,
    image_path: Path,
    max_tokens: int | None,
) -> Statistics:
    """Runs the model prediction and returns statistics about the annotation process."""
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

    if success:
        try:
            cleaned_response_text = strip_markdown_json(prediction_response.text)
            annotation_result = AnnotationResult.from_json(cleaned_response_text)
        except Exception as exception:
            success = False
            error_message = format_error_message(exception)

    return Statistics(
        image_path=image_path,
        image_id=image_path.stem,
        model=str(model_identifier),
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
