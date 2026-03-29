from pathlib import Path

from qwen_vl_utils import process_vision_info
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BatchEncoding,
)

from n2f.local_model import LocalModel
from n2f.message import Message


class Qwen2_5_VLModel(LocalModel):
    def __init__(self, model_path: Path) -> "Qwen2_5_VLModel":
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        patch_size = 28
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=512 * patch_size * patch_size,
            max_pixels=1280 * patch_size * patch_size,
        )

    def predict(
        self,
        prompt: str,
        image_paths: list[Path],
        max_tokens: int = 2048,
    ) -> str:
        inputs = self._prepare_inputs(prompt, image_paths)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            repetition_penalty=1.1,
        )
        trimmed_generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed_generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def train(self) -> None:
        raise NotImplementedError("Training not implemented for Qwen2_5_VLModel")

    def save(self, save_path: Path) -> None:
        raise NotImplementedError("Saving not implemented for Qwen2_5_VLModel")

    def _prepare_inputs(self, prompt: str, image_paths: list[Path]) -> BatchEncoding:
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path.resolve())}
                    for image_path in image_paths
                ]
                + [{"type": "text", "text": prompt}],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)
        return self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
