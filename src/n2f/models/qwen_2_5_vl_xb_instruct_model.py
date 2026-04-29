"""A module for defining the Qwen2.5-VL model."""

from pathlib import Path
from peft import PeftModel

from qwen_vl_utils import process_vision_info
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BatchEncoding,
)

from n2f.core.message import Message
from n2f.core.response import Response
from n2f.models.local_model import LocalModel
from n2f.utils.utils import format_error_message


class Qwen_2_5_vl_xb_instruct_model(LocalModel):
    """A class for interacting with the Qwen2.5-VL model."""

    def __init__(
        self, 
        model_path: Path,
        lora_path: Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )

        if lora_path is not None:
            self.model = PeftModel.from_pretrained(
                self.model, 
                lora_path, 
                # torch_dtype="auto", device_map="auto"
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
        max_tokens: int | None = None,
    ) -> Response:
        try:
            inputs = self._prepare_inputs(prompt, image_paths)
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens if max_tokens is not None else 2048,
                repetition_penalty=1.1,
            )
            trimmed_generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            text = self.processor.batch_decode(
                trimmed_generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            tokens_used = sum(len(output_ids) for output_ids in trimmed_generated_ids)
            return Response(
                text=text,
                tokens_used=tokens_used,
                success=True,
                error_message=None,
            )
        except Exception as exception:
            return Response(
                text="",
                tokens_used=0,
                success=False,
                error_message=format_error_message(exception),
            )

    def train(self) -> None:
        raise NotImplementedError("Training not implemented for Qwen2_5_VLModel.")

    def save(self, save_path: Path) -> None:
        raise NotImplementedError("Saving not implemented for Qwen2_5_VLModel.")

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
