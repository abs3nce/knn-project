"""A module for defining remote models."""

from n2f.models.model import Model


class RemoteModel(Model):
    """A base class for remote models."""

    def __init__(self, api_key: str, model_name: str) -> None:
        self.api_key = api_key
        self.model_name = model_name
