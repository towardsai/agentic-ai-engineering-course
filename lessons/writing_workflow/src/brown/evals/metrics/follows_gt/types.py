"""Type definitions for the follows_gt evaluation metric.

This module contains Pydantic models and types used for evaluating articles
against ground truth content across multiple dimensions (content, flow,
structure).
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


class FollowsGTCriterionScores(CriteriaScores):
    """Represents scores for all three evaluation dimensions of a section.

    This model contains scores for a single section across all three
    evaluation dimensions used in article assessment.

    Attributes:
        content: Score for accuracy, depth, and relevance of information.
        flow: Score for narrative flow and logical progression.
        structure: Score for clarity, tone, and readability.

    """

    content: CriterionScore
    flow: CriterionScore
    structure: CriterionScore


class FollowsGTSectionScores(pydantic.BaseModel):
    """Section evaluation with scores across all FollowsGT dimensions.

    This class holds the evaluation results for one section of an article,
    containing scores for content, flow, and structure.

    Attributes:
        title: The title of the section being evaluated.
        scores: The scores for this section across all FollowsGT dimensions.

    """

    title: str = pydantic.Field(description="The title of the section being evaluated.")
    scores: FollowsGTCriterionScores = pydantic.Field(description="The scores of the section being evaluated.")

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


class FollowsGTArticleScores(pydantic.BaseModel):
    """Article-level scores for the FollowsGT evaluation metric.

    This class represents the complete evaluation results for an article,
    containing scores for all sections across the three FollowsGT dimensions
    (content, flow, structure).

    Attributes:
        sections: List of evaluated sections, each containing scores for all dimensions.

    """

    sections: list[FollowsGTSectionScores]

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


class FollowsGTMetricFewShotExample(BaseExample):
    """Represents a single example for the follows_gt evaluation.

    Attributes:
        output: The generated article content.
        expected_output: The expected article content.
        scores: The FollowsGTArticleScores associated with this example.

    """

    output: str
    expected_output: str
    scores: FollowsGTArticleScores

    @classmethod
    def from_markdown(
        cls, output_file: Path, expected_output_file: Path, scores: FollowsGTArticleScores
    ) -> "FollowsGTMetricFewShotExample":
        """Create a FollowsGTMetricExample instance from markdown files.

        Args:
            output_file: Path to the generated article content.
            expected_output_file: Path to the expected article content.
            scores: The FollowsGTArticleScores associated with this example.

        Returns:
            An instance of FollowsGTMetricExample populated with content from files and scores.

        """
        output = output_file.read_text()
        expected_output = expected_output_file.read_text()

        return cls(output=output, expected_output=expected_output, scores=scores)

    def to_context(self) -> str:
        """Convert the example to a formatted string for use as context in prompts.

        Returns:
            A string representation of the example, including output, expected output, and scores.

        """
        return f"""
<output>
{self.output}
</output>
<expected_output>
{self.expected_output}
</expected_output>
{self.scores.to_context()}
"""


class FollowsGTMetricFewShotExamples(BaseFewShotExamples[FollowsGTMetricFewShotExample]):
    """Collection of few-shot examples for the FollowsGT evaluation metric.

    This class contains examples used for prompt engineering to guide the
    language model in evaluating articles against ground truth content.
    """

    pass
