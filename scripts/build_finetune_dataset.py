import json
import argparse
from pathlib import Path
from tqdm import tqdm

from n2f.core.prompt import AnnotatePrompt
from run_annotation import get_dataset_annotated_images, get_annotations_from_json

def create_pretraining_dataset(
    dataset_path: Path, 
    prompt_path: Path, 
    output_json_path: Path
) -> None:
    """
    Creates a pretraining/finetuning JSON dataset from the annotated images.
    """
    prompt = AnnotatePrompt(prompt_path)
    dataset = []

    # iterate through the dataset via the existing helper function
    for image_path, json_path in tqdm(
        get_dataset_annotated_images(dataset_path),
        desc="Building grounding dataset",
    ):
        # retrieve all ground truth annotations for current image
        for annotation in get_annotations_from_json(json_path):
            
            # construct the user prompt via template
            user_prompt_text = prompt.render({"label": annotation.label})
            
            # format the human request with the <image> token as expected by qwen/llava formats
            human_value = f"<image>\n{user_prompt_text}"
            
            # extract the bounding box coordinates
            bbox_coords = annotation.bounding_box.to_list()
            
            # construct json model output
            gpt_value = f'{{\n    "bbox_2d": {bbox_coords}\n}}'
            
            # sample format
            sample = {
                "image": str(image_path.resolve()),
                "conversations": [
                    {
                        "from": "human",
                        "value": human_value
                    },
                    {
                        "from": "gpt",
                        "value": f"```json\n{gpt_value}```\n"
                    }
                ]
            }
            
            dataset.append(sample)

    # save ggregated dataset
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Successfully saved {len(dataset)} training samples to {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset for Qwen2.5-VL PEFT Finetuning")
    parser.add_argument(
        "--dataset-path", 
        type=Path, 
        # default=Path("data/pages/with_ner/")
        required=True
    )
    parser.add_argument(
        "--prompt-path", 
        type=Path, 
        # default=Path("prompts/annotate_prompt.j2")
        required=True
    )
    parser.add_argument(
        "--output-path", 
        type=Path, 
        # default=Path("finetuning_dataset.json")
        required=True
    )
    
    args = parser.parse_args()

    # checks
    if not args.dataset_path.exists():
        parser.error(f"Dataset path does not exist: {args.dataset_path}")
    if not args.prompt_path.exists():
        parser.error(f"Prompt path does not exist: {args.prompt_path}")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_pretraining_dataset(args.dataset_path, args.prompt_path, args.output_path)