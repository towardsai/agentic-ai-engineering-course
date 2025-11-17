from pydantic import BaseModel

from brown.entities.mixins import ContextMixin, MarkdownMixin


class Article(BaseModel, ContextMixin, MarkdownMixin):
    content: str

    def to_context(self) -> str:
        return f"""
<{self.xml_tag}>
    {self.content}
</{self.xml_tag}>
"""

    def to_markdown(self) -> str:
        return self.content

    def __str__(self) -> str:
        return f"Article(len_content={len(self.content)})"


class SelectedText(BaseModel, ContextMixin):
    article: Article
    content: str
    first_line_number: int
    last_line_number: int

    def to_context(self, include_article: bool = False) -> str:
        return f"""
<{self.xml_tag}>
    {f"<article>{self.article.to_context()}</article>" if include_article else ""}
    <content>{self.content}</content>
    <first_line_number>{self.first_line_number}</first_line_number>
    <last_line_number>{self.last_line_number}</last_line_number>
</{self.xml_tag}>
"""


class ArticleExample(BaseModel, ContextMixin):
    content: str

    def to_context(self) -> str:
        return f"""
<{self.xml_tag}>
    {self.content}
</{self.xml_tag}>
"""

    def __str__(self) -> str:
        return f"ArticleExample(len_content={len(self.content)})"


class ArticleExamples(BaseModel, ContextMixin):
    examples: list[ArticleExample]

    def to_context(self) -> str:
        return f"""
<{self.xml_tag}>
        {"\n".join([example.to_context() for example in self.examples])}
</{self.xml_tag}>
"""
