from pathlib import Path
import argparse
import random
import shutil


def main() -> None:
    args = parse_args()

    items = collect_items(args.source)
    rng = random.Random(args.seed)
    rng.shuffle(items)

    train_count, val_count, _ = split_counts(len(items))
    train_items = items[:train_count]
    val_items = items[train_count : train_count + val_count]
    test_items = items[train_count + val_count :]

    train_dir, val_dir, test_dir = prepare_output_dirs(
        args.output_root,
        args.clear_output,
    )

    transfer_items(train_items, train_dir, args.move)
    transfer_items(val_items, val_dir, args.move)
    transfer_items(test_items, test_dir, args.move)

    print(f"Source: {args.source}")
    print(f"Output root: {args.output_root}")
    print(f"Total items: {len(items)}")
    print(f"Training: {len(train_items)}")
    print(f"Validation: {len(val_items)}")
    print(f"Testing: {len(test_items)}")
    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split data/pages/with_ner into training/validation/testing folders."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("./data/pages/with_ner"),
        help="Source directory containing one sample per subdirectory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./data/pages"),
        help="Root where training/validation/testing folders will be created.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move directories instead of copying them.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete existing training/validation/testing folders before splitting.",
    )
    return parser.parse_args()


def collect_items(source: Path) -> list[Path]:
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")

    items = [path for path in source.iterdir() if path.is_dir()]
    if not items:
        raise RuntimeError(f"No subdirectories found in source: {source}")
    return items


def split_counts(total: int) -> tuple[int, int, int]:
    train = int(total * 0.8)
    val = int(total * 0.1)
    test = total - train - val
    return train, val, test


def prepare_output_dirs(
    output_root: Path, clear_output: bool
) -> tuple[Path, Path, Path]:
    train_dir = output_root / "training"
    val_dir = output_root / "validation"
    test_dir = output_root / "testing"

    if clear_output:
        for directory in (train_dir, val_dir, test_dir):
            if directory.exists():
                shutil.rmtree(directory)

    for directory in (train_dir, val_dir, test_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return train_dir, val_dir, test_dir


def transfer_items(items: list[Path], destination: Path, move: bool) -> None:
    for item in items:
        target = destination / item.name
        if target.exists():
            raise FileExistsError(
                f"Target already exists: {target}. Use --clear-output or remove it first."
            )
        if move:
            shutil.move(str(item), str(target))
        else:
            shutil.copytree(item, target)


if __name__ == "__main__":
    main()