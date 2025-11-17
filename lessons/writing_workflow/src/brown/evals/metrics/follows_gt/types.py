"""Type definitions for the follows_gt evaluation metric.

This module contains Pydantic models and types used for evaluating articles
against ground truth content across multiple dimensions (content, flow,
structure, and mechanics).
"""

from pathlib import Path

from brown.evals.metrics.base import (
    ArticleScores,
    BaseExample,
    BaseFewShotExamples,
    CriteriaScores,
    CriterionScore,
)


class FollowsGTCriterionScores(CriteriaScores):
    """Represents scores for all four evaluation dimensions of a section.

    This model contains scores for a single section across all four
    evaluation dimensions used in article assessment.

    Attributes:
        content: Score for accuracy, depth, and relevance of information.
        flow: Score for narrative flow and logical progression.
        structure: Score for clarity, tone, and readability.
        mechanics: Score for grammar, spelling, and technical correctness.

    """

    content: CriterionScore
    flow: CriterionScore
    structure: CriterionScore
    mechanics: CriterionScore


class FollowsGTArticleScores(ArticleScores[FollowsGTCriterionScores]):
    """Article-level scores for the FollowsGT evaluation metric.

    This class represents the complete evaluation results for an article,
    containing scores for all sections across the four FollowsGT dimensions.
    """

    pass


class FollowsGTMetricExample(BaseExample):
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
    def from_markdown(cls, output_file: Path, expected_output_file: Path, scores: FollowsGTArticleScores) -> "FollowsGTMetricExample":
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


class FollowsGTMetricFewShotExamples(BaseFewShotExamples[FollowsGTMetricExample]):
    """Collection of few-shot examples for the FollowsGT evaluation metric.

    This class contains examples used for prompt engineering to guide the
    language model in evaluating articles against ground truth content.
    """

    pass
