"""A module for defining the Youtu-VL model."""

from pathlib import Path

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BatchEncoding,
)

from n2f.core.message import Message
from n2f.core.response import Response
from n2f.models.local_model import LocalModel
from n2f.utils.utils import format_error_message


class YoutuVLModel(LocalModel):
    """A class for interacting with the Youtu-VL-4B-Instruct model."""

    def __init__(self, model_path: Path) -> None:
        super().__init__(model_path)
        
        # Initialize the model with specific Youtu-VL arguments
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            torch_dtype="auto",
            device_map="cuda",
            trust_remote_code=True,
        ).eval()
        
        # Initialize the processor
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            use_fast=True, 
            trust_remote_code=True
        )

    def predict(
        self,
        prompt: str,
        image_paths: list[Path],
        max_tokens: int | None = None,
    ) -> Response:
        try:
            inputs = self._prepare_inputs(prompt, image_paths)
            
            # Youtu-VL requires passing the image path directly to generate()
            # If multiple images are provided, we pass them as a list of strings; 
            # otherwise, we pass the single string as shown in the original snippet.
            img_paths_str = [str(p.resolve()) for p in image_paths]
            img_input_arg = img_paths_str[0] if len(img_paths_str) == 1 else img_paths_str

            generated_ids = self.model.generate(
                **inputs,
                temperature=0.1,
                top_p=0.001,
                repetition_penalty=1.05,
                do_sample=True,
                max_new_tokens=max_tokens if max_tokens is not None else 2048,
                img_input=img_input_arg,
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
        raise NotImplementedError("Training not implemented for YoutuVLModel.")

    def save(self, save_path: Path) -> None:
        raise NotImplementedError("Saving not implemented for YoutuVLModel.")

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

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        return inputs