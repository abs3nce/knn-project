from pathlib import Path
from typing import Any

from n2f.local_model import LocalModel
from n2f.model import Model
from n2f.openai_model import OpenAIModel
from n2f.qwen2_5_vl_model import Qwen2_5_VLModel
from n2f.remote_model import RemoteModel


class ModelFactory:
    def __init__(self) -> None:
        self.remote_model_registry: dict[str, type[RemoteModel]] = {
            "openai": OpenAIModel,
        }
        self.local_model_registry: dict[str, type[LocalModel]] = {
            "qwen2_5_vl": Qwen2_5_VLModel,
        }

    def create_model(self, model_identifier: str, **keyword_arguments: Any) -> Model:
        identifier_parts = model_identifier.split(":", maxsplit=2)

        if len(identifier_parts) != 3:
            raise ValueError(
                f"Invalid model identifier '{model_identifier}'. "
                "Expected format: local:model_name:model_path or "
                "remote:provider:model_name"
            )

        environment_category = identifier_parts[0]
        registry_key = identifier_parts[1]
        target_value = identifier_parts[2]

        if environment_category == "remote":
            return self._create_remote_model(
                provider_name=registry_key,
                model_name=target_value,
                keyword_arguments=keyword_arguments,
            )

        if environment_category == "local":
            return self._create_local_model(
                model_name=registry_key,
                model_path_string=target_value,
            )

        raise ValueError(
            f"Unknown model category '{environment_category}'. Expected 'local' or 'remote'."
        )

    def _create_remote_model(
        self,
        provider_name: str,
        model_name: str,
        keyword_arguments: dict[str, Any],
    ) -> Model:
        model_class = self.remote_model_registry.get(provider_name)
        if model_class is None:
            available_providers = ", ".join(self.remote_model_registry.keys())
            raise ValueError(
                f"Unknown remote provider '{provider_name}'. "
                f"Available providers are: {available_providers}"
            )

        api_key = keyword_arguments.get("api_key")
        if not api_key or not isinstance(api_key, str):
            raise ValueError("Missing or invalid required keyword argument: api_key")

        return model_class(api_key=api_key, model_name=model_name)

    def _create_local_model(self, model_name: str, model_path_string: str) -> Model:
        model_class = self.local_model_registry.get(model_name)
        if model_class is None:
            available_models = ", ".join(self.local_model_registry.keys())
            raise ValueError(
                f"Unknown local model '{model_name}'. "
                f"Available local models are: {available_models}"
            )

        model_path = Path(model_path_string)
        return model_class(model_path=model_path)
