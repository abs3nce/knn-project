import base64
from openai import OpenAI
from pathlib import Path

from n2f.remote_model import RemoteModel
from n2f.message import Message


class OpenAIModel(RemoteModel):
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key)

    def predict(
        self,
        prompt: str,
        image_paths: list[Path],
        max_tokens: int = 2048,
    ) -> str:
        content = [{"type": "text", "text": prompt}]

        for path in image_paths:
            base64_image = self._encode_image(path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )

        messages: list[Message] = [
            {
                "role": "user",
                "content": content,
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def _encode_image(self, image_path: Path) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
