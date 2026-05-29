"""Implementation of the FollowsGT evaluation metric.

This module implements a metric that evaluates how well generated articles
follow the structure and content of ground truth articles across four dimensions:
content, flow, structure, and mechanics.
"""

from typing import Any, cast

from opik.evaluation.metrics import score_result

from brown.evals.metrics.base import BrownBaseMetric
from brown.models import ModelConfig, SupportedModels

from . import prompts
from .types import FollowsGTArticleScores


class FollowsGTMetricLLMJudge(BrownBaseMetric):
    """A metric that evaluates the quality of article content across multiple sections and dimensions.

    This metric uses a language model to assess how well the generated article content
    matches the expected output across three key dimensions: content, flow, and structure.
    It evaluates multiple sections and computes average scores per dimension,
    providing detailed breakdowns for comprehensive analysis.

    The scoring system evaluates each section across three dimensions (content, flow,
    structure), computes average scores per dimension across all sections,
    returns separate ScoreResult objects for each dimension, and provides detailed
    reasoning for each dimension and section.

    Args:
        model: The language model to use for evaluation. Defaults to GOOGLE_GEMINI_25_FLASH.
        name: The name of the metric. Defaults to "article".
        model_config: Configuration for the model including temperature, thinking budget,
            and retry settings. If None, uses default configuration with temperature=0.0,
            thinking_budget=4096, include_thoughts=False, and max_retries=3.
        track: Whether to track the metric in observability tools. Defaults to True.
        project_name: Optional project name to track the metric in for cases when there are
            no parent span/trace to inherit project name from.

    Attributes:
        few_shot_examples: Default few-shot examples used for prompt engineering.
        structured_output_type: The ArticleScores type used for structured output parsing.

    Example:
        >>> from brown.evals.metrics.follows_gt.metric import FollowsGTMetricLLMJudge
        >>> follows_gt_metric = FollowsGTMetricLLMJudge()
        >>> results = await follows_gt_metric.ascore(
            ...     output="Generated article content...",
            ...     expected_output="Expected article content..."
            ... )
        >>> # results is a list of ScoreResult objects, one per dimension
        >>> for result in results:
        ...     print(f"{result.name}: {result.value}")  # e.g., "follows_gt_flow: 0.85"
        ...     print(f"Reason: {result.reason}")

    """

    def __init__(
        self,
        model: SupportedModels = SupportedModels.GOOGLE_GEMINI_25_FLASH,
        name: str = "follows_gt",
        model_config: ModelConfig | None = None,
        track: bool = True,
        project_name: str | None = None,
    ) -> None:
        """Initialize the FollowsGTMetric instance.

        Args:
            model: The language model to use for evaluation. Defaults to GOOGLE_GEMINI_25_FLASH.
            name: The name of the metric. Defaults to "follows_gt".
            model_config: Configuration for the model including temperature, thinking budget,
                and retry settings. If None, uses default configuration.
            track: Whether to track the metric in observability tools. Defaults to True.
            project_name: Optional project name to track the metric in for cases when there are
                no parent span/trace to inherit project name from.

        """
        model_config = model_config or ModelConfig(temperature=0.0, thinking_budget=1024 * 4, include_thoughts=False, max_retries=3)
        super().__init__(
            model=model,
            name=name,
            structured_output_type=FollowsGTArticleScores,
            few_shot_examples=prompts.DEFAULT_FEW_SHOT_EXAMPLES,
            model_config=model_config,
            track=track,
            project_name=project_name,
        )

    async def ascore(
        self,
        output: str,
        expected_output: str,
        **ignored_kwargs: Any,
    ) -> score_result.ScoreResult | list[score_result.ScoreResult]:
        """Asynchronously calculate the article evaluation score with dimension-wise analysis.

        This method uses an LLM to evaluate the article content across multiple sections
        and three dimensions (content, flow, structure). It computes
        average scores per dimension and returns separate ScoreResult objects for each
        dimension with detailed reasoning.

        The evaluation process involves:
        1. Initializing a fresh model client to avoid coroutine reuse issues
        2. Constructing an evaluation prompt with few-shot examples
        3. Getting structured output from the LLM with ArticleScores format
        4. Converting the response to ScoreResult objects for each dimension

        Args:
            output: The generated article content to be evaluated.
            expected_output: The expected article content to compare against.
            **ignored_kwargs: Additional keyword arguments that are ignored to maintain
                compatibility with the base metric interface.

        Returns:
            list[score_result.ScoreResult]: A list of ScoreResult objects, one for each
                dimension (content, flow, structure), containing the
                aggregated score (between 0.0 and 1.0) and detailed reasons broken
                down by sections.

        Raises:
            ValueError: If the model fails to return a structured response or if the
                response cannot be parsed into the expected ArticleScores format.

        Note:
            A new model client is initialized for each call to avoid coroutine reuse
            issues when running in multiple threads due to sharing the same model
            instance across threads.

        """
        # Initialize the model client at the function level to avoid coroutine reuse issues when running in
        # multiple threads due to sharing the same model instance across threads.
        model_client = self.init_model()

        llm_query = prompts.get_eval_prompt(
            output=output,
            expected_output=expected_output,
            few_shot_examples=self.few_shot_examples,
        )
        article_response = cast(
            FollowsGTArticleScores,
            await model_client.ainvoke(
                [
                    {
                        "role": "user",
                        "content": llm_query,
                    }
                ]
            ),
        )

        if not article_response:
            raise ValueError("Model failed to return a structured response.")

        return article_response.to_score_result(self.name)
