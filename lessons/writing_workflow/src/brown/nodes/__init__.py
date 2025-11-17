from typing import Type

from brown.nodes.base import ToolNode

from .article_reviewer import ArticleReviewer
from .article_writer import ArticleWriter
from .media_generator import MediaGeneratorOrchestrator, Toolkit
from .tool_nodes import MermaidDiagramGenerator

TOOL_NODES: dict[str, Type[ToolNode]] = {
    "mermaid_diagram_generator": MermaidDiagramGenerator,
}


__all__ = ["ArticleWriter", "MediaGeneratorOrchestrator", "Toolkit", "MermaidDiagramGenerator", "ArticleReviewer", "TOOL_NODES"]
