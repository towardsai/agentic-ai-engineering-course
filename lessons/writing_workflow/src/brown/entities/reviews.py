from pydantic import BaseModel, Field

from brown.entities.articles import Article, SelectedText
from brown.entities.mixins import ContextMixin


class Review(BaseModel, ContextMixin):
    profile: str = Field(description="The profile type listing the constraints based on which we will write the comment.")
    location: str = Field(
        description="The location from within the article where the comment is made. For example, the title of a section."
    )
    comment: str = Field(description="The comment made by the reviewer stating the issue relative to the profile.")

    def to_context(self) -> str:
        return f"""
<{self.xml_tag}>
    <profile>{self.profile}</profile>
    <location>{self.location}</location>
    <comment>{self.comment}</comment>
</{self.xml_tag}>
"""


class ArticleReviews(BaseModel, ContextMixin):
    article: Article
    reviews: list[Review]

    def to_context(self, include_article: bool = False) -> str:
        reviews_str = "\n".join([review.to_context() for review in self.reviews])
        return f"""
<{self.xml_tag}>
    {f"<article>{self.article}</article>" if include_article else ""}
    <reviews>
    {reviews_str}
    </reviews>
</{self.xml_tag}>
"""

    def __str__(self) -> str:
        return f"Reviews(len_reviews={len(self.reviews)})"


class SelectedTextReviews(BaseModel, ContextMixin):
    article: Article
    selected_text: SelectedText
    reviews: list[Review]

    def to_context(self, include_article: bool = False) -> str:
        reviews_str = "\n".join([review.to_context() for review in self.reviews])
        return f"""
<{self.xml_tag}>
    {f"<article>{self.article.to_context()}</article>" if include_article else ""}
    <selected_text>{self.selected_text.to_context()}</selected_text>
    <reviews>
    {reviews_str}
    </reviews>
</{self.xml_tag}>
"""


class HumanFeedback(BaseModel, ContextMixin):
    content: str

    def to_context(self) -> str:
        return f"""
<{self.xml_tag}>
    {self.content}
</{self.xml_tag}>
"""
