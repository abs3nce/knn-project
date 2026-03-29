from n2f.model import Model


class RemoteModel(Model):
    def __init__(self, api_key: str, model_name: str) -> None:
        self.api_key = api_key
        self.model_name = model_name
