import json
import torch
import argparse
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info

class QwenVLDataset(Dataset):
    """Custom Dataset to process LLaVA-style JSON into Qwen2.5-VL inputs."""
    def __init__(self, data_path: Path, processor: AutoProcessor):
        super().__init__()
        with data_path.open("r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]
        conversations = item["conversations"]
        
        user_text = conversations[0]["value"].replace("<image>\n", "").strip()
        assistant_text = conversations[1]["value"]
        
        # qwen2.5-vl conversational structure
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_text}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_text}
                ]
            }
        ]
        
        # apply prompt template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        # process conversation
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )
        
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        
        # only calculate token length of the user prompt
        # model computes loss for expected bounding box json output
        prompt_messages = [messages[0]]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )
        prompt_length = prompt_inputs["input_ids"].shape[1]
        
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        
        # collect outputs
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"]
            
        return result


def custom_collate_fn(features, processor):
    """Custom collator to handle sequence padding and image tensor stacking."""
    input_ids = [f["input_ids"] for f in features]
    attention_masks = [f["attention_mask"] for f in features]
    labels = [f["labels"] for f in features]
    
    pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id
    
    # pad text sequences
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    attention_masks = torch.nn.utils.rnn.pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }
    
    pixel_values = [f["pixel_values"] for f in features if "pixel_values" in f]
    image_grid_thw = [f["image_grid_thw"] for f in features if "image_grid_thw" in f]
    
    if pixel_values:
        batch["pixel_values"] = torch.cat(pixel_values, dim=0)
    if image_grid_thw:
        batch["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)
        
    return batch


def main(args):
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.model_path, 
        min_pixels=512 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # enable checkpointing
    model.enable_input_require_grads()
    
    print("Applying LoRA...")
    # targeting only linear layers ("*_proj")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    print("Preparing dataset...")
    train_dataset = QwenVLDataset(args.dataset_path, processor)
    val_dataset = QwenVLDataset(args.val_dataset_path, processor) 
    
    collator = lambda features: custom_collate_fn(features, processor)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        eval_strategy="epoch",
        per_device_eval_batch_size=1,
        optim="adamw_torch",
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True, 
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving LoRA adapter to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/qwen_2_5_vl_3b_instruct")
    parser.add_argument("--dataset-path", type=Path, default=Path("training_dataset.json"))
    parser.add_argument("--val-dataset-path", type=Path, default=Path("validation_dataset.json"))
    parser.add_argument("--output-dir", type=str, default="./models/qwen2_5_vl_lora_adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    
    args = parser.parse_args()
    main(args)