from pydantic import BaseModel

from .mixins import ContextMixin


class SectionGuideline(BaseModel, ContextMixin):
    title: str
    content: str

    def to_context(self) -> str:
        return f"""
<{self.xml_tag}>
    <title>{self.title}</title>
    <content>{self.content}</content>
</{self.xml_tag}>
        """

    def __str__(self) -> str:
        return f"SectionGuideline(title={self.title}, len_content={len(self.content)})"


class ArticleGuideline(BaseModel, ContextMixin):
    content: str

    def to_context(self) -> str:
        return f"""
<{self.xml_tag}>
    <content>{self.content}</content>
</{self.xml_tag}>
"""

    def __str__(self) -> str:
        return f"ArticleGuideline(len_content={len(self.content)})"
