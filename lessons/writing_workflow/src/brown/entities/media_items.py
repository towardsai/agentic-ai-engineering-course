from abc import ABC

from pydantic import BaseModel

from brown.entities.mixins import ContextMixin


class MediaItem(BaseModel, ContextMixin, ABC):
    location: str
    content: str
    caption: str

    def to_context(self) -> str:
        return f"""
<{self.xml_tag}>
    <location>{self.location}</location>
    <content>{self.content}</content>
    <caption>{self.caption}</caption>
</{self.xml_tag}>
"""


class MediaItems(BaseModel, ContextMixin):
    media_items: list[MediaItem] | None

    @classmethod
    def build(cls, media_items: list[MediaItem] | None = None) -> "MediaItems":
        return cls(media_items=media_items)

    @property
    def are_available_only_in_source(self) -> bool:
        return self.media_items is None

    def to_context(self) -> str:
        if self.are_available_only_in_source:
            media_items_context = "media items are available only in the source file"
        else:
            media_items_context = "\n".join([media_item.to_context() for media_item in self.media_items])

        return f"""
<{self.xml_tag}>
{media_items_context}
</{self.xml_tag}>
"""


class MermaidDiagram(MediaItem):
    pass
