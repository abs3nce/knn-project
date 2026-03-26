# n2f

Utilizing VLM for extracting names and photos of individuals from a single page of historical newspaper.

### 📂 Dataset Parsing & Structure

The `parse_dataset.py` script is designed to process the **PeopleGator** dataset. To initialize the pipeline, ensure the raw dataset is unzipped into the `./data` directory.

#### 🚀 Quick Start
1. Place the unzipped dataset in `./data/people_gator__data_export`.
2. Run the pipeline: `python parse_dataset.py`.
3. The script will generate a structured `./data/pages` directory.

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