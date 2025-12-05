import asyncio
from pathlib import Path
from typing import TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy

from brown.base import Loader
from brown.builders import build_article_renderer, build_loaders, build_model
from brown.config_app import get_app_config
from brown.entities.articles import Article, ArticleExamples
from brown.entities.guidelines import ArticleGuideline
from brown.entities.media_items import MediaItem, MediaItems
from brown.entities.profiles import ArticleProfiles
from brown.entities.research import Research
from brown.entities.reviews import ArticleReviews
from brown.nodes.article_reviewer import ArticleReviewer
from brown.nodes.article_writer import ArticleWriter
from brown.nodes.media_generator import MediaGeneratorOrchestrator
from brown.workflows.types import WorkflowProgress

app_config = get_app_config()

retry_policy = RetryPolicy(max_attempts=3, retry_on=Exception)


def build_generate_article_workflow(checkpointer: BaseCheckpointSaver):
    """Create a generate article workflow with optional checkpointer.

    Args:
        checkpointer: Optional checkpointer to use. If None, workflow runs without persistence.

    Returns:
        Configured workflow entrypoint
    """

    return entrypoint(checkpointer=checkpointer)(_generate_article_workflow)


class GenerateArticleInput(TypedDict):
    dir_path: Path


async def _generate_article_workflow(inputs: GenerateArticleInput, config: RunnableConfig) -> str:
    dir_path = inputs["dir_path"]
    dir_path.mkdir(parents=True, exist_ok=True)

    writer = get_stream_writer()

    writer(WorkflowProgress(progress=0, message="Loading context").model_dump(mode="json"))
    context = {}
    loaders = build_loaders(app_config)
    for context_name in ["article_guideline", "research", "profiles", "examples"]:
        loader = cast(Loader, loaders[context_name])
        context[context_name] = loader.load(working_uri=dir_path)
    writer(WorkflowProgress(progress=2, message="Loaded context").model_dump(mode="json"))

    writer(WorkflowProgress(progress=3, message="Genererating media items").model_dump(mode="json"))
    media_items = await generate_media_items(context["article_guideline"], context["research"])
    writer(WorkflowProgress(progress=10, message="Generated media items").model_dump(mode="json"))

    writer(WorkflowProgress(progress=15, message="Writing article").model_dump(mode="json"))
    article = await write_article(context["article_guideline"], context["research"], context["profiles"], media_items, context["examples"])
    writer(WorkflowProgress(progress=20, message="Written raw article").model_dump(mode="json"))

    article_path = dir_path / app_config.context.build_article_uri(0)
    article_renderer = build_article_renderer(app_config)
    article_renderer.render(article, output_uri=article_path)
    writer(WorkflowProgress(progress=25, message=f"Rendered raw article to `{article_path}`").model_dump(mode="json"))

    # Distribute progress evenly from 25 to 100 across review/edit/render steps
    steps_per_iteration = 3  # review, edit, render
    total_steps = max(1, app_config.num_reviews * steps_per_iteration)
    step_size = 75 / total_steps  # remaining percentage after 25
    for i in range(1, app_config.num_reviews + 1):
        base_step_index = (i - 1) * steps_per_iteration

        # Review step
        p_review = int(25 + step_size * (base_step_index + 1))
        p_review = min(p_review, 99)
        writer(
            WorkflowProgress(progress=p_review, message=f"Rewiewing article [Iteration {i} / {app_config.num_reviews}]").model_dump(
                mode="json"
            )
        )
        reviews = await generate_reviews(article, context["article_guideline"], context["profiles"])
        writer(WorkflowProgress(progress=p_review, message="Generated reviews").model_dump(mode="json"))

        # Edit step
        p_edit = int(25 + step_size * (base_step_index + 2))
        p_edit = min(p_edit, 99)
        writer(WorkflowProgress(progress=p_edit, message="Editing article").model_dump(mode="json"))
        article = await edit_based_on_reviews(
            context["article_guideline"],
            context["research"],
            context["profiles"],
            media_items,
            context["examples"],
            reviews,
        )
        writer(WorkflowProgress(progress=p_edit, message="Edited article").model_dump(mode="json"))

        # Render step
        p_render = int(25 + step_size * (base_step_index + 3))
        p_render = min(p_render, 99)
        article_path = dir_path / app_config.context.build_article_uri(i)
        article_renderer.render(article, output_uri=article_path)
        writer(WorkflowProgress(progress=p_render, message=f"Rendered article to `{article_path}`").model_dump(mode="json"))

    article_path = dir_path / app_config.context.article_uri
    article_renderer.render(article, output_uri=article_path)
    writer(WorkflowProgress(progress=100, message=f"Final article rendered to `{article_path}`").model_dump(mode="json"))

    return f"Final article rendered to`{article_path}`."


@task(retry_policy=retry_policy)
async def generate_media_items(article_guideline: ArticleGuideline, research: Research) -> MediaItems:
    writer = get_stream_writer()

    model, toolkit = build_model(app_config, node="generate_media_items")
    media_generator_orchestrator = MediaGeneratorOrchestrator(
        article_guideline=article_guideline,
        research=research,
        model=model,
        toolkit=toolkit,
    )
    media_items_to_generate_jobs = await media_generator_orchestrator.ainvoke()

    writer(f"Found {len(media_items_to_generate_jobs)} media items to generate using the following tool configurations:")
    for i, job in enumerate(media_items_to_generate_jobs):
        writer(f"  • Tool {i + 1}: {job['name']} - {job.get('args', {}).get('description_of_the_diagram', 'No description')}")

    coroutines = []
    for media_item_to_generate_job in media_items_to_generate_jobs:
        tool_name = media_item_to_generate_job["name"]
        tool = media_generator_orchestrator.toolkit.get_tool_by_name(tool_name)
        if tool is None:
            writer(f"⚠️ Warning: Unknown tool '{tool_name}', skipping...")
            continue
        coroutine = tool.ainvoke(media_item_to_generate_job["args"])
        coroutines.append(coroutine)

    writer(f"Executing {len(coroutines)} media item generation jobs in parallel.")
    media_items: list[MediaItem] = await asyncio.gather(*coroutines)
    writer(f"Generated {len(media_items)} media items.")

    return MediaItems.build(media_items)


@task(retry_policy=retry_policy)
async def write_article(
    article_guideline: ArticleGuideline,
    research: Research,
    article_profiles: ArticleProfiles,
    media_items: MediaItems,
    article_examples: ArticleExamples,
) -> Article:
    model, _ = build_model(app_config, node="write_article")
    article_writer = ArticleWriter(
        article_guideline=article_guideline,
        research=research,
        article_profiles=article_profiles,
        media_items=media_items,
        article_examples=article_examples,
        model=model,
    )
    article = await article_writer.ainvoke()

    return cast(Article, article)


@task(retry_policy=retry_policy)
async def generate_reviews(article: Article, article_guideline: ArticleGuideline, article_profiles: ArticleProfiles) -> ArticleReviews:
    model, _ = build_model(app_config, node="review_article")
    article_reviewer = ArticleReviewer(
        to_review=article,
        article_guideline=article_guideline,
        article_profiles=article_profiles,
        model=model,
    )
    reviews = await article_reviewer.ainvoke()

    return cast(ArticleReviews, reviews)


@task(retry_policy=retry_policy)
async def edit_based_on_reviews(
    article_guideline: ArticleGuideline,
    research: Research,
    article_profiles: ArticleProfiles,
    media_items: MediaItems,
    article_examples: ArticleExamples,
    reviews: ArticleReviews,
) -> Article:
    model, _ = build_model(app_config, node="edit_article")
    article_writer = ArticleWriter(
        article_guideline=article_guideline,
        research=research,
        article_profiles=article_profiles,
        media_items=media_items,
        article_examples=article_examples,
        model=model,
        reviews=reviews,
    )
    article = await article_writer.ainvoke()

    return cast(Article, article)
