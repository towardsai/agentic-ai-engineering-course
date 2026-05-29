"""Base classes and utilities for evaluation metrics.

This module provides abstract base classes and common functionality for building
evaluation metrics that use language models to assess content quality across
multiple dimensions with structured output.
"""

import abc
from typing import Annotated, Any, Generic, TypeVar

import pydantic
from annotated_types import Ge, Le
from langchain_core.runnables import Runnable
from opik.evaluation.metrics import base_metric, score_result
from pydantic import BaseModel

from brown.models import ModelConfig, SupportedModels, get_model
from brown.utils import a

FewShotExamplesT = TypeVar("FewShotExamplesT", bound=BaseModel)
StructuredOutputTypeT = TypeVar("StructuredOutputTypeT", bound=BaseModel)

CriteriaScoresT = TypeVar("CriteriaScoresT", bound="CriteriaScores")
ExampleT = TypeVar("ExampleT", bound="BaseExample")


class CriterionAggregatedScore(pydantic.BaseModel):
    """Base model for aggregated scores used to validate the scores.

    This model represents a single aggregated score for a specific dimension
    across all sections of an article evaluation.

    Attributes:
        name: The name of the metric dimension.
        score: The aggregated score for this dimension, between 0 and 1.
        reason: Detailed explanation for the aggregated score.

    """

    name: str
    score: Annotated[float, Ge(0), Le(1)]
    reason: str

    def to_score_result(self) -> score_result.ScoreResult:
        """Convert the aggregated score to a ScoreResult object.

        Returns:
            A ScoreResult object containing the aggregated score information.

        """
        return score_result.ScoreResult(name=self.name, value=self.score, reason=self.reason)


class CriterionScore(pydantic.BaseModel):
    """Base model for a single score representing a specific evaluation dimension.

    This model holds a score and its corresponding reason for one evaluation dimension.

    Attributes:
        score: The score for this dimension (binary 0-1 or float 0.0-1.0).
        reason: Detailed explanation for the given score.

    """

    score: Annotated[int, Ge(0), Le(1)] = pydantic.Field(description="Binary score of the section.")
    reason: str = pydantic.Field(description="The reason for the given score.")


class CriteriaScores(pydantic.BaseModel):
    """Abstract base class for scores across multiple evaluation dimensions.

    This class provides the generic structure and `to_context()` functionality
    that can be used by any scores class containing multiple score dimensions.

    Note:
        This is a generic base class. Concrete implementations should define
        specific score fields as needed (e.g., content, flow, structure, mechanics).

    """

    def to_context(self) -> str:
        """Convert the scores to a formatted XML string for use as context in prompts.

        This method automatically generates XML structure based on the model fields,
        making it adaptable to any subclass with different score dimensions.

        Returns:
            An XML string representation of all score dimensions.

        """
        scores_fields = self.__class__.model_fields
        scores_xml = ""
        for field_name in scores_fields.keys():
            field_score = getattr(self, field_name)
            scores_xml += f"""    <{field_name}>
        <score>{field_score.score}</score>
        <reason>{field_score.reason}</reason>
    </{field_name}>
"""
        return scores_xml


def aggregate_section_scores_to_results(
    section_scores: list,
    prefix: str,
) -> list[score_result.ScoreResult]:
    """Convert section-level evaluation results to aggregated ScoreResult objects.

    This function aggregates scores across all sections for each dimension and creates
    separate ScoreResult objects for each dimension. It works with any section type
    that has 'title' and 'scores' attributes where scores is a CriteriaScores subclass.

    Args:
        section_scores: List of section objects, each with 'title' (str) and 'scores' (CriteriaScores) attributes.
        prefix: The prefix to use for the metric names (e.g., "follows_gt", "user_intent").

    Returns:
        A list of ScoreResult objects, one for each dimension, containing the
        aggregated score and detailed reasons.

    Example:
        >>> sections = [
        ...     FollowsGTSectionScores(title="Intro", scores=FollowsGTCriterionScores(...)),
        ...     FollowsGTSectionScores(title="Body", scores=FollowsGTCriterionScores(...)),
        ... ]
        >>> results = aggregate_section_scores_to_results(sections, "follows_gt")
        >>> # Returns [ScoreResult(name="follows_gt_content", ...), ScoreResult(name="follows_gt_flow", ...), ...]

    """
    if not section_scores:
        return []

    # Automatically infer dimensions from the first section's scores class
    scores_class = type(section_scores[0].scores)  # type: ignore[attr-defined]
    scores_fields = scores_class.model_fields
    aggregated_scores: dict[str, dict[str, list[int] | str]] = {
        field_name: {
            "scores": [],
            "reason": "",
        }
        for field_name in scores_fields.keys()
    }

    for section in section_scores:
        for dimension in aggregated_scores.keys():
            dimension_score = getattr(section.scores, dimension)  # type: ignore[attr-defined]
            aggregated_scores[dimension]["scores"].append(dimension_score.score)  # type: ignore[union-attr]
            aggregated_scores[dimension]["reason"] += f"{section.title}:\n"  # type: ignore[attr-defined]
            aggregated_scores[dimension]["reason"] += f"**{dimension_score.score}:** {dimension_score.reason}\n\n"

    results: list[score_result.ScoreResult] = []
    for dimension, scores_data in aggregated_scores.items():
        scores_list = scores_data["scores"]
        aggregated_score = CriterionAggregatedScore(
            name=f"{prefix}_{dimension}",
            score=sum(scores_list) / len(scores_list),  # type: ignore[arg-type]
            reason=str(scores_data["reason"]),
        )
        results.append(aggregated_score.to_score_result())

    return results


class BaseExample(pydantic.BaseModel, Generic[ExampleT]):
    """Base class for examples used in evaluation metrics.

    This class provides common functionality for different types of examples
    (e.g., user intent examples, article examples) while maintaining type safety.
    """

    @abc.abstractmethod
    def to_context(self) -> str:
        """Convert the example to a formatted string for use as context in prompts.

        This method should be implemented by subclasses to define the specific
        format for their example type.

        Returns:
            A string representation of the example.

        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_markdown(cls, *args: Any, **kwargs: Any) -> "BaseExample":
        """Create an example instance from markdown files.

        This method should be implemented by subclasses to define how to load
        their specific example format from files.

        Args:
            *args: Variable positional arguments specific to each implementation.
            **kwargs: Variable keyword arguments specific to each implementation.

        Returns:
            An instance of the BaseExample subclass populated with content from files.

        """
        pass


class BaseFewShotExamples(pydantic.BaseModel, Generic[ExampleT]):
    """Base class for few-shot examples collections.

    This class provides common functionality for managing collections of examples
    used in prompt engineering for different evaluation metrics.

    Attributes:
        examples: List of examples of type ExampleT.

    """

    examples: list[ExampleT]

    def to_context(self) -> str:
        """Convert the list of few-shot examples to a formatted string for prompt context.

        Returns:
            A string containing all examples formatted for insertion into an LLM prompt.

        """
        examples = "\n\n".join(
            [f"<example_{i + 1}>\n\t{example.to_context()}\n</example_{i + 1}>\n" for i, example in enumerate(self.examples)]
        )

        return examples


class BrownBaseMetric(base_metric.BaseMetric, Generic[FewShotExamplesT, StructuredOutputTypeT], abc.ABC):
    """Abstract base class for Brown evaluation metrics that use LLMs for structured scoring.

    This abstract base class provides a foundation for implementing evaluation metrics that
    use language models to assess content quality across multiple dimensions. It handles
    common functionality like model initialization, structured output parsing, and
    synchronous/asynchronous scoring interfaces.

    The class is designed to work with structured outputs using Pydantic models and
    supports various language models through the SupportedModels enum. It integrates
    with the Opik evaluation framework for tracking and observability.

    Type Parameters:
        FewShotExamplesT: Type of the few-shot examples model, must inherit from BaseModel.
        StructuredOutputTypeT: Type of the structured output model, must inherit from BaseModel.

    Args:
        model: The language model to use for evaluation from SupportedModels enum.
        name: The name of the metric for identification and tracking purposes.
        structured_output_type: The Pydantic model class that defines the expected
            structure of the LLM response.
        few_shot_examples: An instance of the few-shot examples model containing
            example inputs and outputs for prompt engineering.
        model_config: Configuration for the model including temperature, thinking budget,
            retry settings, and optional mocked responses for testing.
        track: Whether to track the metric execution in observability tools. Defaults to True.
        project_name: Optional project name for tracking when there's no parent span/trace
            to inherit the project name from.

    Attributes:
        model: The configured language model type.
        structured_output_type: The Pydantic model class for structured output.
        model_config: The model configuration settings.
        few_shot_examples: The few-shot examples for prompt engineering.

    Raises:
        AssertionError: If using FAKE_MODEL without providing a mocked_response in model_config.

    Note:
        This is an abstract base class and cannot be instantiated directly. Subclasses
        must implement the `ascore` method to define specific evaluation logic.

    Example:
        >>> class MyMetric(BrownBaseMetric[MyExamples, MyResponse]):
        ...     async def ascore(self, input: str, context: dict, output: str, **kwargs):
        ...         # Implementation specific to MyMetric
        ...         pass
        >>>
        >>> metric = MyMetric(
        ...     model=SupportedModels.GOOGLE_GEMINI_25_FLASH,
        ...     name="my_metric",
        ...     structured_output_type=MyResponse,
        ...     few_shot_examples=MyExamples(),
        ...     model_config=ModelConfig(temperature=0.0)
        ... )

    """

    def __init__(
        self,
        model: SupportedModels,
        name: str,
        structured_output_type: type[StructuredOutputTypeT],
        few_shot_examples: FewShotExamplesT,
        model_config: ModelConfig,
        track: bool = True,
        project_name: str | None = None,
    ) -> None:
        """Initialize the BrownBaseMetric instance.

        Args:
            model: The language model to use for evaluation from SupportedModels enum.
            name: The name of the metric for identification and tracking purposes.
            structured_output_type: The Pydantic model class that defines the expected
                structure of the LLM response.
            few_shot_examples: An instance of the few-shot examples model containing
                example inputs and outputs for prompt engineering.
            model_config: Configuration for the model including temperature, thinking budget,
                retry settings, and optional mocked responses for testing.
            track: Whether to track the metric execution in observability tools.
            project_name: Optional project name for tracking when there's no parent span/trace
                to inherit the project name from.

        Raises:
            AssertionError: If using FAKE_MODEL without providing a mocked_response in model_config.

        """
        super().__init__(
            name=name,
            track=track,
            project_name=project_name,
        )

        self.model = model
        self.structured_output_type = structured_output_type
        self.model_config = model_config
        self.few_shot_examples = few_shot_examples

        if self.model == SupportedModels.FAKE_MODEL:
            assert self.model_config and self.model_config.mocked_response is not None, "Mocked response is required for fake model"

    def init_model(self) -> Runnable:
        """Initialize the language model with structured output capabilities.

        This method configures the language model using the instance's model configuration
        (temperature, thinking budget, max retries, etc.) and wraps it to enforce
        structured output according to the specified Pydantic model type.

        The method creates a fresh model instance each time it's called to avoid
        coroutine reuse issues when running in multiple threads or concurrent contexts.

        Returns:
            Runnable: A configured LangChain Runnable instance ready for invocation,
                with structured output enforcement based on self.structured_output_type.

        Note:
            This method should be called at the function level rather than stored as
            an instance variable to prevent threading issues with shared model instances.

        """
        model_instance = get_model(self.model, self.model_config)
        model_instance = model_instance.with_structured_output(self.structured_output_type)

        return model_instance

    def score(self, *args: Any, **kwargs: Any) -> score_result.ScoreResult | list[score_result.ScoreResult]:
        """Calculate evaluation scores synchronously by wrapping the async implementation.

        This method provides a synchronous interface to the evaluation functionality
        by running the async `ascore` method in a new event loop. It serves as a
        compatibility layer for scenarios where async/await syntax cannot be used.

        The actual evaluation logic is implemented in the abstract `ascore` method
        which must be overridden by subclasses to define specific scoring behavior.

        Args:
            *args: Variable positional arguments including:
                - input: The input content to be used for evaluation (e.g., guidelines,
                    prompts, or reference material).
                - context: Dictionary containing additional context information needed
                    for evaluation. The specific keys depend on the metric implementation.
                - output: The generated content to be evaluated and scored.
            **kwargs: Additional keyword arguments that are passed through
                to the async method but may be ignored by specific implementations.

        Returns:
            score_result.ScoreResult | list[score_result.ScoreResult]: Either a single
                ScoreResult object or a list of ScoreResult objects, depending on the
                specific metric implementation. Each ScoreResult contains a score value
                (typically between 0.0 and 1.0) and reasoning text.

        Note:
            This method creates a new event loop to run the async implementation.
            For better performance in async contexts, use the `ascore` method directly.

        """
        return a.asyncio_run(self.ascore(*args, **kwargs))

    @abc.abstractmethod
    async def ascore(self, *args: Any, **kwargs: Any) -> score_result.ScoreResult | list[score_result.ScoreResult]:
        """Abstract async method for implementing metric-specific evaluation logic.

        This abstract method must be implemented by all subclasses to define the
        specific evaluation behavior for each metric type. It performs the actual
        assessment of content quality using language models and structured output.

        Subclasses should implement this method to:
        1. Initialize a fresh model client using `self.init_model()`
        2. Construct evaluation prompts with few-shot examples
        3. Invoke the LLM with structured output requirements
        4. Process and return results as ScoreResult objects

        Args:
            *args: Variable positional arguments that depend on the specific metric
                implementation. Common patterns include input, context, and output.
            **kwargs: Variable keyword arguments for additional parameters and
                compatibility with the base metric interface.

        Returns:
            Either a single ScoreResult object or a list of ScoreResult objects containing
            evaluation scores and reasoning. The specific return type depends on whether
            the metric evaluates single or multiple dimensions.

        Raises:
            NotImplementedError: Always raised since this is an abstract method that
                must be implemented by subclasses.

        Note:
            Implementations should handle model initialization at the method level
            to avoid coroutine reuse issues in multi-threaded environments.

        """
        pass
