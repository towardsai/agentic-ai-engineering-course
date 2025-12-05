from typing import Literal, TypedDict

from langchain_core.runnables import Runnable

from brown.base import Loader, Renderer
from brown.config_app import AppConfig
from brown.entities.exceptions import InvalidConfigurationException
from brown.loaders import (
    MarkdownArticleExampleLoader,
    MarkdownArticleGuidelineLoader,
    MarkdownArticleLoader,
    MarkdownArticleProfilesLoader,
    MarkdownResearchLoader,
)
from brown.memory import build_in_memory_checkpointer, build_sqlite_checkpointer
from brown.models import get_model
from brown.nodes import TOOL_NODES, Toolkit
from brown.renderers import MarkdownArticleRenderer

###################
##### Loaders #####
###################


class Loaders(TypedDict):
    article: Loader
    article_guideline: Loader
    research: Loader
    profiles: Loader
    examples: Loader


def build_article_loader(app_config: AppConfig) -> MarkdownArticleLoader:
    if app_config.context.article_loader == "markdown":
        return MarkdownArticleLoader(uri=app_config.context.article_uri)
    else:
        raise InvalidConfigurationException(f"Invalid article loader: {app_config.context.article_loader}")


def build_article_guideline_loader(app_config: AppConfig) -> MarkdownArticleGuidelineLoader:
    if app_config.context.article_guideline_loader == "markdown":
        return MarkdownArticleGuidelineLoader(uri=app_config.context.article_guideline_uri)
    else:
        raise InvalidConfigurationException(f"Invalid article guideline loader: {app_config.context.article_guideline_loader}")


def build_research_loader(app_config: AppConfig) -> MarkdownResearchLoader:
    if app_config.context.research_loader == "markdown":
        return MarkdownResearchLoader(uri=app_config.context.research_uri)
    else:
        raise InvalidConfigurationException(f"Invalid research loader: {app_config.context.research_loader}")


def build_profiles_loader(app_config: AppConfig) -> MarkdownArticleProfilesLoader:
    if app_config.context.profiles_loader == "markdown":
        return MarkdownArticleProfilesLoader(
            uri={
                "article": app_config.context.profiles_uri / "article_profile.md",
                "character": app_config.context.profiles_uri / "character_profiles" / app_config.context.character_profile,
                "mechanics": app_config.context.profiles_uri / "mechanics_profile.md",
                "structure": app_config.context.profiles_uri / "structure_profile.md",
                "terminology": app_config.context.profiles_uri / "terminology_profile.md",
                "tonality": app_config.context.profiles_uri / "tonality_profile.md",
            }
        )
    else:
        raise InvalidConfigurationException(f"Invalid article profiles loader: {app_config.context.article_profiles_loader}")


def build_example_loader(app_config: AppConfig) -> MarkdownArticleExampleLoader:
    if app_config.context.examples_loader == "markdown":
        return MarkdownArticleExampleLoader(uri=app_config.context.examples_uri)
    else:
        raise InvalidConfigurationException(f"Invalid article example loader: {app_config.context.article_example_loader}")


def build_loaders(app_config: AppConfig) -> Loaders:
    return Loaders(
        article=build_article_loader(app_config),
        article_guideline=build_article_guideline_loader(app_config),
        research=build_research_loader(app_config),
        profiles=build_profiles_loader(app_config),
        examples=build_example_loader(app_config),
    )


#####################
##### Renderers #####
#####################


def build_article_renderer(app_config: AppConfig) -> Renderer:
    if app_config.context.article_renderer == "markdown":
        return MarkdownArticleRenderer()
    else:
        raise InvalidConfigurationException(f"Invalid article renderer: {app_config.context.article_renderer}")


##################
##### Memory #####
##################


def build_short_term_memory(app_config: AppConfig):
    """Build a checkpointer based on app config.

    Returns:
        For in_memory: Returns an async context manager that yields InMemorySaver
        For sqlite: Returns an async context manager that yields AsyncSqliteSaver

    Usage:
        # For both in_memory and sqlite (use async with)
        async with build_short_term_memory(app_config) as checkpointer:
            workflow = build_generate_article_workflow(checkpointer=checkpointer)
            await workflow.ainvoke(...)
    """

    if app_config.memory.checkpointer == "in_memory":
        return build_in_memory_checkpointer()
    elif app_config.memory.checkpointer == "sqlite":
        return build_sqlite_checkpointer()
    else:
        raise InvalidConfigurationException(f"Invalid memory checkpointer: {app_config.memory.checkpointer}")


##################
##### Models #####
##################


def build_model(
    app_config: AppConfig,
    *,
    node: Literal["write_article", "review_article", "generate_media_items", "edit_article", "review_selected_text", "edit_selected_text"],
) -> tuple[Runnable, Toolkit]:
    model = app_config.nodes[node]

    node_model_client = get_model(model.model_id, model.config)
    compiled_tools = []
    for tool_id, tool in model.tools.items():
        tool_model_client = get_model(tool.model_id, tool.config)
        tool_node = TOOL_NODES[tool_id](model=tool_model_client)
        compiled_tools.append(tool_node.as_tool())

    return node_model_client, Toolkit(tools=compiled_tools)
