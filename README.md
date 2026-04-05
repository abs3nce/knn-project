# n2f

Utilizing VLM for extracting names and photos of individuals from a single page of historical newspaper.

### 📂 Dataset Parsing & Structure

The `parse_dataset.py` script is designed to process the **PeopleGator** dataset. To initialize the pipeline, ensure the raw dataset is unzipped into the `./data` directory.

#### 🚀 Quick Start
1. Place the zipped dataset in `./data`.
2. Unzip it.
3. Make sure the `./data` contains folder `people_gator__data_export` and inside it is folder `people_gator__data` and `jsonl` files containing the labeled bounding boxes of faces from the dataset.
4. Run the pipeline: `python parse_dataset.py`.
5. The script will generate a structured `./data/pages` directory.

#### 🗂 Directory Hierarchy
After the script completes, pages are validated and moved into one of three status categories:

* **`with_ner/`**: Pages containing both face detections and verified OCR/NER data.
* **`without_ner/`**: Pages with face detections but no associated NER records.
* **`defective/`**: Pages missing critical assets (e.g., metadata, images, or crops).

#### 📄 Inside a Page Folder
Each individual page folder (e.g., `./data/pages/with_ner/{uuid}/`) contains:

```text
├── {uuid}.jpg            # The full-resolution source image of the newspaper page.
├── {uuid}.ner.jsonl      # Text OCR and Named Entity Recognition data.
├── {uuid}_faces.jsonl    # Bounding boxes and names for detected faces.
├── crops/                # Raw cropped images of individual faces.
└── aligned_crops/        # Pre-processed and aligned face crops.
```

### 🤖 Dataset Annotation

The `run_annotation.py` script runs visual-language model (VLM) inference over page images and stores one JSON record per image in a JSONL output file.

It supports two modes:

* **`local`**: Loads a model from disk.
* **`remote`**: Calls a hosted API model.

#### 🚀 Quick Start
1. Ensure your dataset is prepared (typically `./data/pages/with_ner/`).
2. Ensure your prompt template exists (default: `./prompts/annotate_prompt.j2`).
3. Run annotation in either `local` or `remote` mode.
4. Review the resulting JSONL logs (default: `./output.jsonl`).

#### ▶️ Command Examples

Run with a **local** model:

```bash
python scripts/run_annotation.py local \
	--model-path models/model3b \
	--dataset-path data/pages/with_ner \
	--prompt-path prompts/annotate_prompt.j2 \
	--jsonl-output-path output.local.jsonl
```

Run with a **remote** model:

```bash
python scripts/run_annotation.py remote \
	--provider openai \
	--model-name gpt-5-nano \
	--api-key "$OPENAI_API_KEY" \
	--dataset-path data/pages/with_ner \
	--prompt-path prompts/annotate_prompt.j2 \
	--jsonl-output-path output.remote.jsonl
```

#### ⚙️ CLI Arguments

Run ```python scripts/run_annotation.py --help``` for information about command-line arguments.

#### 📄 Output JSONL Schema
Each output line is a JSON object with statistics about the annotation.

```json
{
	"image_path": "data/pages/with_ner/<uuid>/<uuid>.jpg",
	"image_id": "<uuid>",
	"model": "remote:openai:gpt-5-nano",
	"prompt_path": "prompts/annotate_prompt.j2",
	"raw_response": "...original model text...",
	"annotation_result": {
		"detections": [
			{
				"bounding_box": [x_min, y_min, x_max, y_max],
				"label": "person_name"
			}
		]
	},
	"success": 1,
	"error_message": "",
	"start_timestamp": "2026-04-05T12:34:56.000000",
	"end_timestamp": "2026-04-05T12:34:57.000000",
	"total_time_seconds": 1.0,
	"tokens_used": 123
}
```
