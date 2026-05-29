"""Type definitions for the user_intent evaluation metric.

This module contains Pydantic models and types used for evaluating articles
based on how well they follow provided guidelines and are anchored in research
across two dimensions (guideline_adherence and research_anchoring).
"""

from pathlib import Path

import pydantic
from opik.evaluation.metrics import score_result

from brown.evals.metrics.base import (
    BaseExample,
    BaseFewShotExamples,
    CriteriaScores,
    CriterionScore,
    aggregate_section_scores_to_results,
)


class UserIntentCriteriaScores(CriteriaScores):
    """Represents scores for both evaluation dimensions of a section.

    This model contains binary scores for a single section across both
    evaluation dimensions used in user intent assessment.

    Attributes:
        guideline_adherence: Score for how well the section follows the article guideline.
        research_anchoring: Score for how well the section is anchored in the provided research.

    """

    guideline_adherence: CriterionScore
    research_anchoring: CriterionScore


class UserIntentSectionScores(pydantic.BaseModel):
    """Section evaluation with scores across all UserIntent dimensions.

    This class holds the evaluation results for one section of an article,
    containing scores for guideline_adherence and research_anchoring.

    Attributes:
        title: The title of the section being evaluated.
        scores: The scores for this section across all UserIntent dimensions.

    """

    title: str = pydantic.Field(description="The title of the section being evaluated.")
    scores: UserIntentCriteriaScores = pydantic.Field(description="The scores of the section being evaluated.")

    def to_context(self) -> str:
        """Convert the section scores to a formatted XML string for use as context in prompts.

        Returns:
            An XML string representation of the section scores.

        """
        scores_xml = self.scores.to_context()

        return f"""
<section_scores>
    <section_title>{self.title}</section_title>
{scores_xml}</section_scores>
"""


class UserIntentArticleScores(pydantic.BaseModel):
    """Article-level scores for the UserIntent evaluation metric.

    This class represents the complete evaluation results for an article,
    containing scores for all sections across the two UserIntent dimensions
    (guideline_adherence, research_anchoring).

    Attributes:
        sections: List of evaluated sections, each containing scores for all dimensions.

    """

    sections: list[UserIntentSectionScores]

    def to_context(self) -> str:
        """Convert all section scores to a formatted string for use as context in prompts.

        Returns:
            A string containing all section scores formatted for prompt context.

        """
        context_body = "".join([f"\t{section.to_context()}\n" for section in self.sections])

        return f"<article_scores>\n{context_body}\n</article_scores>"

    def to_score_result(self, prefix: str) -> list[score_result.ScoreResult]:
        """Convert the evaluation results to ScoreResult objects with dimension-wise scoring.

        This method aggregates scores across all sections for each dimension and creates
        separate ScoreResult objects for each dimension.

        Args:
            prefix: The prefix to use for the metric names.

        Returns:
            A list of ScoreResult objects, one for each dimension, containing the
            aggregated score and detailed reasons.

        """
        return aggregate_section_scores_to_results(self.sections, prefix)


class UserIntentMetricFewShotExample(BaseExample):
    """Represents a single example for the user intent evaluation.

    Attributes:
        input: The article guideline (input).
        context: The research content (context).
        output: The generated article content.
        scores: The UserIntentArticleScores associated with this example.

    """

    input: str
    context: str
    output: str
    scores: UserIntentArticleScores

    @classmethod
    def from_markdown(
        cls,
        input_file: Path,
        context_file: Path,
        output_file: Path,
        scores: UserIntentArticleScores,
    ) -> "UserIntentMetricFewShotExample":
        """Create a UserIntentExample instance from markdown files.

        Args:
            input_file: Path to the article guideline content.
            context_file: Path to the research content.
            output_file: Path to the generated article content.
            scores: The UserIntentArticleScores associated with this example.

        Returns:
            An instance of UserIntentExample populated with content from files and scores.

        """
        input_content = input_file.read_text()
        context_content = context_file.read_text()
        output_content = output_file.read_text()

        return cls(
            input=input_content,
            context=context_content,
            output=output_content,
            scores=scores,
        )

    def to_context(self) -> str:
        """Convert the example to a formatted string for use as context in prompts.

        Returns:
            A string representation of the example, including input, context, output, and scores.

        """
        return f"""
<input>
{self.input}
</input>
<context>
{self.context}
</context>
<output>
{self.output}
</output>
{self.scores.to_context()}
"""


class UserIntentMetricFewShotExamples(BaseFewShotExamples[UserIntentMetricFewShotExample]):
    """Collection of few-shot examples for the UserIntent evaluation metric.

    This class contains examples used for prompt engineering to guide the
    language model in evaluating articles for guideline adherence and research anchoring.
    """

    pass
