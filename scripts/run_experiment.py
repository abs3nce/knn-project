import argparse
from pathlib import Path

from n2f.qwen2_5_vl_model import Qwen2_5_VLModel
from n2f.openai_model import OpenAIModel


def main() -> None:
    args = parse_arguments()

    if args.command == "local":
        model = Qwen2_5_VLModel(model_path=args.model_path)
    elif args.command == "remote":
        model = OpenAIModel(
            api_key=args.api_key,
            model_name=args.model_name,
        )

    response = model.predict(
        "Describe the following images in one sentence.",
        [
            Path(
                "./data/people_gator__data_export/people_gator__data/nkp/bb6b2ed0-46cc-11de-b577-000d606f5dc6.images/561b8104-1b63-40a1-84a5-be41ec0f287d.jpg"
            ),
            Path(
                "./data/people_gator__data_export/people_gator__data/nkp/4314d930-9c41-11dd-8141-000d606f5dc6.images/8db44c60-6a38-4230-afac-a8de0bff2cb9.jpg",
            ),
        ],
        max_tokens=args.max_tokens,
    )
    print(response)


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
