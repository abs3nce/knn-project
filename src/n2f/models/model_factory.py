"""A module for defining a factory that creates model instances based on model identifiers."""

from pathlib import Path
from typing import Any

from n2f.models.local_model import LocalModel
from n2f.models.model import Model
from n2f.models.model_identifier import ModelIdentifier
from n2f.models.openai_model import OpenAIModel
from n2f.models.qwen_2_5_vl_xb_instruct_model import Qwen_2_5_vl_xb_instruct_model
from n2f.models.remote_model import RemoteModel


class ModelFactory:
    """A factory class for creating model instances based on model identifiers."""

    def __init__(self) -> None:
        self.remote_model_registry: dict[str, type[RemoteModel]] = {
            "openai": OpenAIModel,
        }
        self.local_model_registry: dict[str, type[LocalModel]] = {
            "qwen_2_5_vl_3b_instruct": Qwen_2_5_vl_xb_instruct_model,
            "qwen_2_5_vl_7b_instruct": Qwen_2_5_vl_xb_instruct_model,
            "qwen_2_5_vl_32b_instruct": Qwen_2_5_vl_xb_instruct_model
        }

    def create_model(
        self,
        model_identifier: ModelIdentifier,
        **keyword_arguments: Any,
    ) -> Model:
        """Creates a model instance based on the model identifier."""
        match model_identifier.environment_category:
            case "remote":
                return self._create_remote_model(
                    provider_name=model_identifier.registry_key,
                    model_name=model_identifier.target_value,
                    keyword_arguments=keyword_arguments,
                )
            case "local":
                return self._create_local_model(
                    model_name=model_identifier.registry_key,
                    model_path_string=model_identifier.target_value,
                )
            case _:
                raise ValueError(
                    f"Unknown model category '{model_identifier.environment_category}'. "
                    f"Expected 'local' or 'remote'."
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
            raise ValueError("Missing required keyword argument: api_key")

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
