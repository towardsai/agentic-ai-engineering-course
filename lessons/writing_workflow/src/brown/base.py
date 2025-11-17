from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Loader(ABC):
    def __init__(self, uri: Any) -> None:
        self.uri = uri

    @abstractmethod
    def load(self, working_uri: Any | None = None, *args, **kwargs) -> Any:
        pass


class Renderer(ABC):
    @abstractmethod
    def render(self, content: Any, *, output_uri: Path) -> None:
        pass
