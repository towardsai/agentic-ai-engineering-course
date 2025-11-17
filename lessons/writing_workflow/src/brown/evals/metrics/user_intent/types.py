"""Type definitions for the user_intent evaluation metric.

This module contains Pydantic models and types used for evaluating articles
based on how well they follow provided guidelines and are anchored in research
across two dimensions (guideline_adherence and research_anchoring).
"""

from pathlib import Path

from brown.evals.metrics.base import (
    ArticleScores,
    BaseExample,
    BaseFewShotExamples,
    CriteriaScores,
    CriterionScore,
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


class UserIntentArticleScores(ArticleScores[UserIntentCriteriaScores]):
    """Article-level scores for the UserIntent evaluation metric.

    This class represents the complete evaluation results for an article,
    containing scores for all sections across the two UserIntent dimensions.
    """

    pass


class UserIntentExample(BaseExample):
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
    ) -> "UserIntentExample":
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


class UserIntentMetricFewShotExamples(BaseFewShotExamples[UserIntentExample]):
    """Collection of few-shot examples for the UserIntent evaluation metric.

    This class contains examples used for prompt engineering to guide the
    language model in evaluating articles for guideline adherence and research anchoring.
    """

    pass
