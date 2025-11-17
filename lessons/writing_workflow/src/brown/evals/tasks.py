"""Task functions for running article generation evaluations.

This module provides functions for executing article generation workflows
on evaluation samples, managing caching, and creating reusable evaluation tasks.
"""

import shutil
import uuid
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict

from langchain_core.runnables.config import RunnableConfig
from loguru import logger

from brown.evals.dataset import EvalSampleDict
from brown.observability import tracing
from brown.utils import a
from brown.workflows import generate_article_workflow


@a.as_sync
async def evaluation_task(
    sample: EvalSampleDict,
    cache_dir: Path,
    read_from_cache: bool = False,
    clean_cache: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    """Generate an article using the Brown agent for a single evaluation sample.

    This function calls the Brown agent to generate an article based on the provided
    evaluation sample data, using the same workflow as the main article generation script.

    Args:
        sample: Dictionary containing the evaluation sample data with keys:
            - name: Sample name
            - directory: Sample directory path
            - article_guideline: Article guideline content
            - research: Research content
            - ground_truth_article: Ground truth article content
            - is_few_shot_example: Whether this is a few-shot example
        style_guideline_dir: Directory containing style guidelines.
        examples_dir: Directory containing example articles.
        evaluation_rules_path: Path to evaluation rules file.
        writer_profile_path: Path to writer profile file.
        cache_dir: Cache directory for generated articles.
        read_from_cache: Whether to read from cache instead of generating.
        clean_cache: Whether to clean up cache after processing.
        debug: Whether to enable debug mode.
        online_human_review: Whether to enable online human review.
        skip_to_stage_3: Whether to skip to stage 3 of the workflow.

    Returns:
        Dictionary containing evaluation results with keys:
            - input: Original input data (article_guideline, research)
            - context: Context used for generation (style guidelines, examples, etc.)
            - output: Generated article content
            - expected_output: Ground truth article content
            - reference: Ground truth article content (alias for expected_output)
            - name: Name of the evaluation sample

    """
    logger.info(f"Processing evaluation sample: {sample['name']}")

    cache_dir = cache_dir / sample["directory"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        article_guideline_path = cache_dir / "article_guideline.md"
        research_path = cache_dir / "research.md"

        article_guideline_path.write_text(sample["article_guideline"], encoding="utf-8")
        research_path.write_text(sample["research"], encoding="utf-8")

        thread_id = str(uuid.uuid4())
        inputs = {
            "dir_path": cache_dir,
        }
        tracer = tracing.build_handler(thread_id, tags=["generate-evaluation"])
        config = RunnableConfig(
            configurable={
                "thread_id": thread_id,
            },
            callbacks=[tracer],
        )

        try:
            generated_article = await __run(config, inputs, read_from_cache)

            logger.success(f"Successfully generated article for sample: `{sample['name']}`")
        except Exception:
            generated_article = "ERROR: Failed to generate article"

            logger.exception(f"Failed to generate article for sample `{sample['name']}` with error:")

        return {
            "input": sample["article_guideline"],
            "context": {
                "research": sample["research"],
                "debug": debug,
            },
            "output": generated_article,
            "expected_output": sample["ground_truth_article"],
            "reference": sample["ground_truth_article"],
            "name": sample["name"],
        }
    finally:
        try:
            if clean_cache and cache_dir.exists():
                shutil.rmtree(cache_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up cache directory `{cache_dir}`: {e}")


async def __run(config: RunnableConfig, inputs: Dict[str, Any], read_from_cache: bool = False) -> str:
    """Run the article generation workflow and extract the final article.

    This function executes the Brown agent workflow to generate an article
    based on the provided inputs and configuration, then extracts the
    generated article content from the output directory.

    Args:
        graph: The compiled state graph for article generation.
        config: Runnable configuration containing thread_id, debug settings, etc.
        inputs: Input parameters for the workflow including file paths and settings.
        read_from_cache: Whether to read from cache instead of running the workflow.

    Returns:
        The generated article content as a string.

    Raises:
        AssertionError: If the article file is not found in cache when read_from_cache is True.
        FileNotFoundError: If the generated article file is not found.

    """
    article_path = inputs["dir_path"] / "article.md"
    if read_from_cache:
        assert article_path.exists(), f"Article file not found in cache at `{article_path}`"
        logger.success(f"Successfully read article from cache at `{article_path}`")
    else:
        await generate_article_workflow.ainvoke(inputs, config)

    article = article_path.read_text(encoding="utf-8")

    return article


def create_evaluation_task(
    cache_dir: Path,
    read_from_cache: bool = False,
    clean_cache: bool = False,
    debug: bool = False,
) -> Callable:
    """Create a reusable evaluation task with fixed runtime parameters.

    Returns a callable that accepts only the evaluation sample and delegates to
    `evaluation_task` with the provided fixed arguments.

    Args:
        cache_dir: Directory used to cache per-sample artifacts.
        read_from_cache: If True, read outputs from cache instead of generating.
        clean_cache: If True, delete the sample cache directory after completion.
        debug: If True, enables additional debugging information in the workflow.

    Returns:
        Callable[[EvalSampleDict], Dict[str, Any]]: A function that takes the
        sample dict and returns the evaluation result structure.
    """
    return partial(
        evaluation_task,
        cache_dir=cache_dir,
        read_from_cache=read_from_cache,
        clean_cache=clean_cache,
        debug=debug,
    )
