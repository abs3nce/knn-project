"""A module for defining the OpenAI model."""

from pathlib import Path
import base64

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionContentPartParam

from n2f.core.response import Response
from n2f.models.remote_model import RemoteModel
from n2f.utils.utils import format_error_message


class OpenAIModel(RemoteModel):
    """A class for interacting with OpenAI's API."""

    def __init__(self, api_key: str, model_name: str) -> None:
        super().__init__(api_key, model_name)
        self.client = OpenAI(api_key=self.api_key)

    def predict(
        self,
        prompt: str,
        image_paths: list[Path],
        max_tokens: int | None = None,
    ) -> Response:
        content: list[ChatCompletionContentPartParam] = [
            {"type": "text", "text": prompt}
        ]

        for image_path in image_paths:
            base64_image = self._encode_image(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )

        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "user",
                "content": content,
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=max_tokens,
            )

            text = response.choices[0].message.content or "" if response.choices else ""
            tokens_used = response.usage.completion_tokens if response.usage else 0

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

    def _encode_image(self, image_path: Path) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
