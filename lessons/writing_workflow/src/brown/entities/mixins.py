from abc import abstractmethod

from brown.utils.s import camel_to_snake


class ContextMixin:
    @property
    def xml_tag(self) -> str:
        return camel_to_snake(self.__class__.__name__)

    @abstractmethod
    def to_context(self) -> str:
        """Context representation of the object."""

        pass


class MarkdownMixin:
    @abstractmethod
    def to_markdown(self) -> str:
        """Markdown representation of the object."""

        pass
