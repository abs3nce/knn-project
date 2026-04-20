"""
Dataset Parser for KNN Project.

This module provides a robust pipeline to process raw JSONL exports from the
PeopleGator tool. It handles:
1. Merging and deduplicating face detection records.
2. Partitioning records into page-specific directories.
3. Synchronizing image assets (full pages, raw crops, and aligned crops).
4. Merging and redistributing Named Entity Recognition (NER) data.
5. Validating folder integrity and sorting into 'with_ner', 'without_ner', or 'defective'.

Usage:
    Adjust the paths in `DatasetConfig` and run the script.
"""

import json
import shutil
from pathlib import Path
from typing import List, Set, Dict, TextIO
from contextlib import ExitStack


# --- CONFIGURATION ---
class DatasetConfig:
    """Centralized path configuration for the KNN Project dataset pipeline."""
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    DATA_ROOT = SCRIPT_DIR.parent / "data"
    EXPORT_ROOT = DATA_ROOT / "people_gator__data_export"
    SOURCE_ASSETS = EXPORT_ROOT / "people_gator__data"

    COMBINED_FACES = DATA_ROOT / "combined.jsonl"
    COMBINED_NER = DATA_ROOT / "combined.ner.jsonl"
    PAGES_DIR = DATA_ROOT / "pages"

    # List of specific face export files to merge
    INPUT_JSONLS = [
        EXPORT_ROOT / "people_gator__corresponding_faces__2026-02-11.dev.jsonl",
        EXPORT_ROOT / "people_gator__corresponding_faces__2026-02-11.test.jsonl",
        EXPORT_ROOT
        / "people_gator__corresponding_faces__2026-02-11.wo_eliska.test.jsonl",
        EXPORT_ROOT
        / "people_gator__corresponding_faces__2026-02-11.wo_filip.test.jsonl",
        EXPORT_ROOT
        / "people_gator__corresponding_faces__2026-02-11.wo_jakub.test.jsonl",
        EXPORT_ROOT
        / "people_gator__corresponding_faces__2026-02-11.wo_martin.test.jsonl",
    ]


# --- UTILITIES ---
def log_step(message: str):
    """Prints a formatted step header to the console."""
    print(f"\n🚀 {message}")


def log_success(message: str):
    """Prints a formatted success message to the console."""
    print(f"✅ {message}")


def log_warning(message: str):
    """Prints a formatted warning message to the console."""
    print(f"⚠️  {message}")


# --- CORE FUNCTIONS ---


def merge_and_filter_jsonl(
    input_files: List[Path], output_file: Path, filter_key: str
) -> None:
    """
    Combines multiple JSONL files into one, deduplicating records based on a specific key.

    Args:
        input_files: List of Paths to source JSONL files.
        output_file: Path where the merged file will be saved.
        filter_key: The dictionary key used to identify duplicates (e.g., 'crop_name').
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    seen_keys: Set[str] = set()
    total_scanned = 0

    log_step(f"Merging and filtering {len(input_files)} files...")

    with open(output_file, "w", encoding="utf-8") as outfile:
        for file_path in input_files:
            if not file_path.exists():
                log_warning(f"Missing file: {file_path.name}")
                continue

            with open(file_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    if not (stripped := line.strip()):
                        continue
                    total_scanned += 1
                    try:
                        record = json.loads(stripped)
                        val = record.get(filter_key)
                        if val and str(val) not in seen_keys:
                            outfile.write(json.dumps(record) + "\n")
                            seen_keys.add(str(val))
                    except json.JSONDecodeError:
                        continue

    log_success(
        f"Saved {len(seen_keys)} unique records. (Removed {total_scanned - len(seen_keys)} duplicates)"
    )


def split_records_by_page(input_file: Path, output_base_dir: Path) -> None:
    """
    Groups records by their 'page' attribute and creates individual folder/file structures.
    Groups records in memory first to prevent hitting OS open-file limits.

    Args:
        input_file: Path to the combined face detection JSONL.
        output_base_dir: Root directory where page-specific folders will be created.
    """
    log_step("Splitting records into page folders...")
    output_base_dir.mkdir(parents=True, exist_ok=True)

    records_processed = 0
    grouped_records = {}

    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            if not (stripped := line.strip()):
                continue
            
            record = json.loads(stripped)
            if not (page_val := record.get("page")):
                continue

            page_id = str(page_val).replace(".jpg", "")
            if page_id not in grouped_records:
                grouped_records[page_id] = []
                
            grouped_records[page_id].append(record)
            records_processed += 1

    for page_id, records in grouped_records.items():
        page_dir = output_base_dir / page_id
        page_dir.mkdir(exist_ok=True)

        f_path = page_dir / f"{page_id}_faces.jsonl"
        with open(f_path, "w", encoding="utf-8") as outfile:
            for rec in records:
                outfile.write(json.dumps(rec) + "\n")

    log_success(
        f"Processed {records_processed} records across {len(grouped_records)} folders."
    )


def synchronize_page_assets(pages_root: Path, source_root: Path) -> None:
    """
    Copies physical image assets (full pages, aligned crops, raw crops) from
    the original source into the newly created page folders.

    Args:
        pages_root: The target directory containing page folders.
        source_root: The original data export directory to copy from.
    """
    log_step("Synchronizing image assets...")
    stats = {"full_pages": 0, "aligned": 0, "raw": 0, "missing": 0}

    for page_folder in filter(Path.is_dir, pages_root.iterdir()):
        try:
            # Look for the faces metadata file in the folder
            metadata_file = next(page_folder.glob("*_faces.jsonl"))
        except StopIteration:
            continue

        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                if not (stripped := line.strip()):
                    continue
                rec = json.loads(stripped)

                # Define asset mapping (Source Path -> Destination Directory)
                mapping = {
                    "full_pages": (
                        source_root
                        / rec["library"]
                        / f"{rec['document']}.images"
                        / rec["page"],
                        page_folder,
                    ),
                    "aligned": (
                        source_root
                        / rec["library"]
                        / f"{rec['document']}.peoplegator_aligned_crops"
                        / rec["crop_name"],
                        page_folder / "aligned_crops",
                    ),
                    "raw": (
                        source_root
                        / rec["library"]
                        / f"{rec['document']}.peoplegator_crops"
                        / rec["crop_name"],
                        page_folder / "crops",
                    ),
                }

                for key, (src, dest_dir) in mapping.items():
                    if src.exists():
                        dest_dir.mkdir(exist_ok=True)
                        dest_file = dest_dir / src.name
                        if not dest_file.exists():
                            shutil.copy2(src, dest_file)
                            stats[key] += 1
                    else:
                        stats["missing"] += 1

    log_success(
        f"Assets synced: {stats['full_pages']} pages, {stats['aligned']} aligned, {stats['raw']} raw."
    )


def combine_ner_files(source_root: Path, output_file: Path) -> None:
    """
    Finds all .ner.jsonl files recursively in the source data and merges them into one.

    Args:
        source_root: Directory to search for NER files.
        output_file: Path for the combined NER output.
    """
    log_step(f"Merging NER files from {source_root.name}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_file, "w", encoding="utf-8") as outfile:
        for ner_file in source_root.rglob("*.ner.jsonl"):
            # Avoid self-reference if output is in the same directory
            if ner_file.absolute() == output_file.absolute():
                continue
            with open(ner_file, "r", encoding="utf-8") as infile:
                for line in infile:
                    if line.strip():
                        outfile.write(line)
                        count += 1
    log_success(f"Merged {count} NER records into {output_file.name}")


def split_ner_by_page(combined_ner_file: Path, pages_root: Path) -> None:
    """
    Distributes records from the combined NER file into corresponding page folders.
    Groups records in memory first to prevent hitting OS open-file limits.

    Args:
        combined_ner_file: Path to the merged NER JSONL.
        pages_root: Target directory with page folders.
    """
    log_step("Distributing NER records to page folders...")
    count = 0
    grouped_ner = {}

    with open(combined_ner_file, "r", encoding="utf-8") as infile:
        for line in infile:
            if not (stripped := line.strip()):
                continue
            
            rec = json.loads(stripped)
            page_id = str(rec.get("page", "")).replace(".jpg", "")

            page_dir = pages_root / page_id
            if not page_dir.exists():
                continue

            if page_id not in grouped_ner:
                grouped_ner[page_id] = []

            grouped_ner[page_id].append(stripped)
            count += 1

    for page_id, lines in grouped_ner.items():
        page_dir = pages_root / page_id
        out_path = page_dir / f"{page_id}.ner.jsonl"
        
        with open(out_path, "w", encoding="utf-8") as outfile:
            for ner_line in lines:
                outfile.write(ner_line + "\n")

    log_success(f"Assigned {count} NER records across {len(grouped_ner)} page directories.")


def validate_and_sort_pages(pages_root: Path) -> None:
    """
    Validates folder contents (checking for images and metadata) and moves them
    into status subfolders: 'with_ner', 'without_ner', or 'defective'.
    """
    log_step("Final validation and categorization...")
    cats = ["with_ner", "without_ner", "defective"]
    for c in cats:
        (pages_root / c).mkdir(exist_ok=True)

    stats = {c: 0 for c in cats}
    # Process only folders at the root, ignoring the category folders themselves
    folders = [f for f in pages_root.iterdir() if f.is_dir() and f.name not in cats]

    for page_dir in folders:
        uuid = page_dir.name
        has_ner = any(page_dir.glob("*.ner.jsonl"))

        # Check integrity: Needs metadata, main page image, and crop directories
        meta_files = list(page_dir.glob("*_faces.jsonl"))
        is_defective = not (
            meta_files
            and (page_dir / f"{uuid}.jpg").exists()
            and (page_dir / "crops").exists()
        )

        dest = (
            "defective" if is_defective else ("with_ner" if has_ner else "without_ner")
        )

        try:
            shutil.move(str(page_dir), str(pages_root / dest / uuid))
            stats[dest] += 1
        except Exception as e:
            log_warning(f"Failed to move {uuid}: {e}")

    # Summary Display
    print("-" * 35)
    print(f"| {'Category':<15} | {'Count':<10} |")
    print("-" * 35)
    for k, v in stats.items():
        print(f"| {k:<15} | {v:<10} |")
    print("-" * 35)


# --- MAIN EXECUTION ---


def main():
    """Orchestrates the full dataset parsing pipeline."""
    cfg = DatasetConfig()

    # Phase 1: Face Detection Processing
    merge_and_filter_jsonl(cfg.INPUT_JSONLS, cfg.COMBINED_FACES, "crop_name")
    split_records_by_page(cfg.COMBINED_FACES, cfg.PAGES_DIR)

    # Phase 2: Assets & Named Entity Recognition
    synchronize_page_assets(cfg.PAGES_DIR, cfg.SOURCE_ASSETS)
    combine_ner_files(cfg.SOURCE_ASSETS, cfg.COMBINED_NER)
    split_ner_by_page(cfg.COMBINED_NER, cfg.PAGES_DIR)

    # Phase 3: Cleanup and Categorization
    validate_and_sort_pages(cfg.PAGES_DIR)

    log_success("Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()
