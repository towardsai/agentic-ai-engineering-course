"""Implementation of the UserIntent evaluation metric.

This module implements a metric that evaluates how well generated articles
follow provided guidelines and are anchored in research across two dimensions:
guideline adherence and research anchoring.
"""

from typing import Any, cast

from opik.evaluation.metrics import score_result

from brown.evals.metrics.base import BrownBaseMetric
from brown.models import ModelConfig, SupportedModels

from . import prompts
from .types import UserIntentArticleScores


class UserIntentMetricLLMJudge(BrownBaseMetric):
    """A metric that evaluates how well generated articles follow article guidelines and are anchored in research.

    This metric uses a language model to assess how well the generated article content
    adheres to the provided article guideline (input) and is properly supported by the
    provided research material (context). It evaluates multiple sections and computes
    binary scores for two key dimensions: guideline_adherence and research_anchoring.

    The scoring system evaluates each section across two dimensions (guideline_adherence,
    research_anchoring), computes average scores per dimension across all sections,
    returns separate ScoreResult objects for each dimension, and provides detailed
    reasoning for each dimension and section.

    Args:
        model: The language model to use for evaluation. Defaults to GOOGLE_GEMINI_25_FLASH.
        name: The name of the metric. Defaults to "user_intent".
        model_config: Configuration for the model including temperature, thinking budget,
            and retry settings. If None, uses default configuration with temperature=0.0,
            thinking_budget=4096, include_thoughts=False, and max_retries=3.
        track: Whether to track the metric in observability tools. Defaults to True.
        project_name: Optional project name to track the metric in for cases when there are
            no parent span/trace to inherit project name from.

    Attributes:
        few_shot_examples: Default few-shot examples used for prompt engineering.
        structured_output_type: The UserIntentArticleScores type used for structured output parsing.

    Example:
        >>> from brown.evals.metrics.user_intent.metric import UserIntentMetricLLMJudge
        >>> user_intent_metric = UserIntentMetricLLMJudge()
        >>> results = await user_intent_metric.ascore(
        ...     input="Article guideline content...",
        ...     context={"research": "Research content..."},
        ...     output="Generated article content..."
        ... )
        >>> # results is a list of ScoreResult objects, one per dimension
        >>> for result in results:
        ...     print(f"{result.name}: {result.value}")  # e.g., "user_intent_guideline_adherence: 0.85"
        ...     print(f"Reason: {result.reason}")

    """

    def __init__(
        self,
        model: SupportedModels = SupportedModels.GOOGLE_GEMINI_25_FLASH,
        name: str = "user_intent",
        model_config: ModelConfig | None = None,
        track: bool = True,
        project_name: str | None = None,
    ) -> None:
        """Initialize the UserIntentMetric instance.

        Args:
            model: The language model to use for evaluation. Defaults to GOOGLE_GEMINI_25_FLASH.
            name: The name of the metric. Defaults to "user_intent".
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
            structured_output_type=UserIntentArticleScores,
            few_shot_examples=prompts.DEFAULT_FEW_SHOT_EXAMPLES,
            model_config=model_config,
            track=track,
            project_name=project_name,
        )

    async def ascore(
        self,
        input: str,
        context: dict[str, Any],
        output: str,
        **ignored_kwargs: Any,
    ) -> score_result.ScoreResult | list[score_result.ScoreResult]:
        """Asynchronously calculate the user intent evaluation score with dimension-wise analysis.

        This method uses an LLM to evaluate the article content across multiple sections
        and two dimensions (guideline_adherence, research_anchoring). It computes
        average scores per dimension and returns separate ScoreResult objects for each
        dimension with detailed reasoning.

        The evaluation process involves:
        1. Extracting research content from the context dictionary
        2. Initializing a fresh model client to avoid coroutine reuse issues
        3. Constructing an evaluation prompt with few-shot examples
        4. Getting structured output from the LLM with UserIntentArticleScores format
        5. Converting the response to ScoreResult objects for each dimension

        Args:
            input: The article guideline content that defines the expected structure,
                style, and requirements for the article.
            context: Dictionary containing context information. Must have a "research" key
                with research content that should support the article.
            output: The generated article content to be evaluated against the guideline
                and research.
            **ignored_kwargs: Additional keyword arguments that are ignored to maintain
                compatibility with the base metric interface.

        Returns:
            list[score_result.ScoreResult]: A list of ScoreResult objects, one for each
                dimension (guideline_adherence, research_anchoring), containing the
                aggregated score (between 0.0 and 1.0) and detailed reasons broken
                down by sections.

        Raises:
            ValueError: If the model fails to return a structured response, if the
                response cannot be parsed into the expected UserIntentArticleScores format,
                or if context doesn't contain the required "research" key.

        Note:
            A new model client is initialized for each call to avoid coroutine reuse
            issues when running in multiple threads due to sharing the same model
            instance across threads.

        """
        # Extract research from context
        if "research" not in context:
            raise ValueError("Context must contain a 'research' key with research content")

        research_content = context["research"]

        # Initialize the model client at the function level to avoid coroutine reuse issues when running in
        # multiple threads due to sharing the same model instance across threads.
        model_client = self.init_model()

        llm_query = prompts.get_eval_prompt(
            input=input,
            context=research_content,
            output=output,
            few_shot_examples=self.few_shot_examples,
        )

        user_intent_response = cast(
            UserIntentArticleScores,
            await model_client.ainvoke(
                [
                    {
                        "role": "user",
                        "content": llm_query,
                    }
                ]
            ),
        )

        if not user_intent_response:
            raise ValueError("Model failed to return a structured response.")

        return user_intent_response.to_score_result(self.name)
