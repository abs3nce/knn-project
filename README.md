# n2f

`n2f` is a project for extracting names and faces from historical newspaper pages using vision-language models.

## What this repository contains

- Dataset parsing and splitting utilities
- Dataset building scripts for finetuning
- Model annotation and validation scripts
- Local and remote model wrappers
- Example data, prompts, and evaluation outputs

## Project layout

- `src/n2f/`: core package with models, prompt handling, bounding boxes, and annotation results
- `scripts/`: command-line utilities for parsing data, splitting datasets, finetuning, annotation, and validation
- `data/`: raw exports and processed page datasets
- `prompts/`: prompt templates used during annotation and finetuning
- `logs/`: annotation logs, outputs, and validation results

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -e .
```

The provided `build.sh` script automates the same setup and can also download model files when configured.

## Common scripts

Parse the raw PeopleGator export into page folders:

```bash
python scripts/parse_dataset.py
```

Split page folders into training, validation, and testing sets:

```bash
python scripts/split_dataset.py
```

Build a finetuning dataset from annotated pages:

```bash
python scripts/build_finetune_dataset.py --dataset-path <path> --prompt-path <path> --output-path <path>
```

Run annotation with a local or remote model:

```bash
python scripts/run_annotation.py local --model-path <model_dir>
python scripts/run_annotation.py remote --api-key <api_key>
```

Validate saved model outputs:

```bash
python scripts/validate_model.py
```


