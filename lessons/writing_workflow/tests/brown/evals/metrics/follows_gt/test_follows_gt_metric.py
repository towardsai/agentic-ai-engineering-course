import pytest

from brown.evals.metrics.base import CriterionScore, SectionCriteriaScores
from brown.evals.metrics.follows_gt.metric import FollowsGTMetric
from brown.evals.metrics.follows_gt.types import (
    FollowsGTArticleScores,
    FollowsGTCriterionScores,
)
from brown.models import ModelConfig, SupportedModels


@pytest.fixture
def mock_article_scores_perfect() -> FollowsGTArticleScores:
    """
    Fixture for a perfect FollowsGTArticleScores response.
    """
    return FollowsGTArticleScores(
        sections=[
            SectionCriteriaScores(
                title="Introduction",
                scores=FollowsGTCriterionScores(
                    content=CriterionScore(score=1, reason="Perfect content."),
                    flow=CriterionScore(score=1, reason="Perfect flow."),
                    structure=CriterionScore(score=1, reason="Perfect structure."),
                    mechanics=CriterionScore(score=1, reason="Perfect mechanics."),
                ),
            ),
            SectionCriteriaScores(
                title="Body",
                scores=FollowsGTCriterionScores(
                    content=CriterionScore(score=1, reason="Perfect content."),
                    flow=CriterionScore(score=1, reason="Perfect flow."),
                    structure=CriterionScore(score=1, reason="Perfect structure."),
                    mechanics=CriterionScore(score=1, reason="Perfect mechanics."),
                ),
            ),
        ]
    )


@pytest.fixture
def mock_article_scores_mixed() -> FollowsGTArticleScores:
    """
    Fixture for a mixed FollowsGTArticleScores response.
    """
    return FollowsGTArticleScores(
        sections=[
            SectionCriteriaScores(
                title="Introduction",
                scores=FollowsGTCriterionScores(
                    content=CriterionScore(score=0, reason="Bad content."),
                    flow=CriterionScore(score=1, reason="Good flow."),
                    structure=CriterionScore(score=0, reason="Bad structure."),
                    mechanics=CriterionScore(score=1, reason="Good mechanics."),
                ),
            ),
            SectionCriteriaScores(
                title="Body",
                scores=FollowsGTCriterionScores(
                    content=CriterionScore(score=1, reason="Good content."),
                    flow=CriterionScore(score=0, reason="Bad flow."),
                    structure=CriterionScore(score=1, reason="Good structure."),
                    mechanics=CriterionScore(score=0, reason="Bad mechanics."),
                ),
            ),
        ]
    )


@pytest.fixture
def mock_article_scores_empty_sections() -> FollowsGTArticleScores:
    """
    Fixture for a FollowsGTArticleScores response with empty sections.
    """
    return FollowsGTArticleScores(sections=[])


@pytest.fixture
def mock_article_metric(mock_article_scores_perfect: FollowsGTArticleScores) -> FollowsGTMetric:
    """
    Fixture for a FollowsGTMetric instance with a mocked perfect response.
    """
    model_config = ModelConfig(mocked_response=mock_article_scores_perfect)
    return FollowsGTMetric(model=SupportedModels.FAKE_MODEL, model_config=model_config)


@pytest.fixture
def mock_article_metric_mixed(mock_article_scores_mixed: FollowsGTArticleScores) -> FollowsGTMetric:
    """
    Fixture for a FollowsGTMetric instance with a mocked mixed response.
    """
    model_config = ModelConfig(mocked_response=mock_article_scores_mixed)
    return FollowsGTMetric(model=SupportedModels.FAKE_MODEL, model_config=model_config)


@pytest.fixture
def mock_article_metric_empty_sections(mock_article_scores_empty_sections: FollowsGTArticleScores) -> FollowsGTMetric:
    """
    Fixture for a FollowsGTMetric instance with a mocked empty sections response.
    """
    model_config = ModelConfig(mocked_response=mock_article_scores_empty_sections)
    return FollowsGTMetric(model=SupportedModels.FAKE_MODEL, model_config=model_config)


def test_article_metric_perfect_score(mock_article_metric: FollowsGTMetric) -> None:
    """
    Test that FollowsGTMetric returns perfect scores when mocked with a perfect response.
    """
    output = "This is a well-written article."
    expected_output = "This is the ideal article."

    results = mock_article_metric.score(output=output, expected_output=expected_output)

    assert isinstance(results, list) and len(results) == 4
    for result in results:
        assert result.value == 1.0
        assert result.name.startswith("follows_gt_")


def test_article_metric_mixed_scores(mock_article_metric_mixed: FollowsGTMetric) -> None:
    """
    Test that FollowsGTMetric returns mixed scores when mocked with a mixed response.
    """
    output = "This article has some issues."
    expected_output = "This is the ideal article."

    results = mock_article_metric_mixed.score(output=output, expected_output=expected_output)

    assert isinstance(results, list) and len(results) == 4

    # Expected averages for mixed scores (0, 1) per section
    # (0+1)/2 = 0.5
    for result in results:
        assert result.value == 0.5
        assert result.name.startswith("follows_gt_")


def test_article_metric_empty_sections(mock_article_metric_empty_sections: FollowsGTMetric) -> None:
    """
    Test that FollowsGTMetric handles empty sections gracefully (should return empty list).
    """
    output = ""
    expected_output = ""

    results = mock_article_metric_empty_sections.score(output=output, expected_output=expected_output)
    assert isinstance(results, list) and len(results) == 0  # Empty sections return empty list


def test_article_metric_init_requires_mocked_response_for_fake_model() -> None:
    """
    Test that FollowsGTMetric raises an AssertionError if FAKE_MODEL is used without a mocked_response.
    """
    with pytest.raises(AssertionError, match="Mocked response is required for fake model"):
        model_config = ModelConfig(mocked_response=None)
        FollowsGTMetric(model=SupportedModels.FAKE_MODEL, model_config=model_config)


def test_article_metric_init_default_model() -> None:
    """
    Test that FollowsGTMetric initializes with the default model if not specified.
    """
    metric = FollowsGTMetric()
    assert metric.model == SupportedModels.GOOGLE_GEMINI_25_FLASH


def test_article_metric_init_custom_name() -> None:
    """
    Test that FollowsGTMetric initializes with a custom name.
    """
    custom_name = "my_custom_follows_gt_metric"
    metric = FollowsGTMetric(name=custom_name)
    assert metric.name == custom_name
