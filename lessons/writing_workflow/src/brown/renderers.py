from pathlib import Path

from brown.base import Renderer
from brown.entities.articles import Article
from brown.entities.reviews import ArticleReviews


class MarkdownArticleRenderer(Renderer):
    def render(self, content: Article, *, output_uri: Path) -> None:
        if output_uri.exists():
            output_uri.unlink()
        output_uri.write_text(content.to_markdown())


class MarkdownArticleReviewsRenderer(Renderer):
    def render(self, content: ArticleReviews, *, output_uri: Path) -> None:
        if output_uri.exists():
            output_uri.unlink()
        output_uri.write_text(content.to_context(include_article=False))
