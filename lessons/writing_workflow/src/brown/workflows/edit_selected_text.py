from pathlib import Path
from typing import TypedDict, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.func import entrypoint, task
from langgraph.types import RetryPolicy

from brown.base import Loader
from brown.builders import build_loaders, build_model
from brown.config_app import get_app_config
from brown.entities.articles import ArticleExamples, SelectedText
from brown.entities.guidelines import ArticleGuideline
from brown.entities.media_items import MediaItems
from brown.entities.profiles import ArticleProfiles
from brown.entities.research import Research
from brown.entities.reviews import HumanFeedback, SelectedTextReviews
from brown.nodes.article_reviewer import ArticleReviewer
from brown.nodes.article_writer import ArticleWriter
from brown.workflows.types import WorkflowProgress

app_config = get_app_config()

retry_policy = RetryPolicy(max_attempts=3, retry_on=Exception)


def build_edit_selected_text_workflow(checkpointer: BaseCheckpointSaver):
    """Create an edit selected text workflow with checkpointer.

    Args:
        checkpointer: Checkpointer to use for workflow persistence.

    Returns:
        Configured workflow entrypoint
    """

    return entrypoint(checkpointer=checkpointer)(_edit_selected_text_workflow)


class EditSelectedTextInput(TypedDict):
    dir_path: Path
    human_feedback: str
    selected_text: str
    number_line_before_selected_text: int
    number_line_after_selected_text: int


async def _edit_selected_text_workflow(inputs: EditSelectedTextInput, config: RunnableConfig) -> str:
    writer = get_stream_writer()

    # Progress: Loading context
    writer(WorkflowProgress(progress=0, message="Loading context").model_dump(mode="json"))
    context = {}
    loaders = build_loaders(app_config)
    for context_name, loader in loaders.items():
        loader = cast(Loader, loader)
        context[context_name] = loader.load(working_uri=inputs["dir_path"])

    selected_text = SelectedText(
        article=context["article"],
        content=inputs["selected_text"],
        first_line_number=inputs["number_line_before_selected_text"],
        last_line_number=inputs["number_line_after_selected_text"],
    )
    human_feedback = HumanFeedback(content=inputs["human_feedback"])
    writer(WorkflowProgress(progress=5, message="Loaded context").model_dump(mode="json"))

    # Progress: Reviewing
    writer(WorkflowProgress(progress=20, message="Reviewing selected text").model_dump(mode="json"))
    reviews = await generate_reviews(selected_text, human_feedback, context["article_guideline"], context["profiles"])
    writer(WorkflowProgress(progress=40, message="Generated reviews").model_dump(mode="json"))

    # Progress: Editing
    writer(WorkflowProgress(progress=60, message="Editing selected text").model_dump(mode="json"))
    selected_text = await edit_based_on_reviews(
        context["article_guideline"], context["research"], context["profiles"], context["examples"], reviews
    )
    writer(WorkflowProgress(progress=80, message="Edited selected text").model_dump(mode="json"))

    # Progress: Complete
    writer(WorkflowProgress(progress=100, message="Selected text editing completed").model_dump(mode="json"))

    return f"""
Here is the edited selected text:
{selected_text.to_context()}

Here is what you have to do with edited selected text:
- print the edited selected text to the console for the user to see
- give a quick summary of the changes you made
- always apply the changes to the source file without waiting for the user to explicitly ask you to do so
- in case you haven't changed anything, just say that you haven't changed anything
"""


@task(retry_policy=retry_policy)
async def generate_reviews(
    selected_text: SelectedText,
    human_feedback: HumanFeedback,
    article_guideline: ArticleGuideline,
    article_profiles: ArticleProfiles,
) -> SelectedTextReviews:
    model, _ = build_model(app_config, node="review_selected_text")
    selected_text_reviewer = ArticleReviewer(
        to_review=selected_text,
        human_feedback=human_feedback,
        article_guideline=article_guideline,
        article_profiles=article_profiles,
        model=model,
    )
    reviews = await selected_text_reviewer.ainvoke()

    return cast(SelectedTextReviews, reviews)


@task(retry_policy=retry_policy)
async def edit_based_on_reviews(
    article_guideline: ArticleGuideline,
    research: Research,
    article_profiles: ArticleProfiles,
    article_examples: ArticleExamples,
    reviews: SelectedTextReviews,
) -> SelectedText:
    model, _ = build_model(app_config, node="edit_selected_text")
    article_writer = ArticleWriter(
        article_guideline=article_guideline,
        research=research,
        article_profiles=article_profiles,
        media_items=MediaItems.build(),
        article_examples=article_examples,
        reviews=reviews,
        model=model,
    )
    edited_selected_text = cast(SelectedText, await article_writer.ainvoke())

    return edited_selected_text
