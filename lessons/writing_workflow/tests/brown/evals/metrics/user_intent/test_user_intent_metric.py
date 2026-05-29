import pytest

from brown.evals.metrics.base import CriterionScore
from brown.evals.metrics.user_intent.metric import UserIntentMetricLLMJudge
from brown.evals.metrics.user_intent.types import (
    UserIntentArticleScores,
    UserIntentCriteriaScores,
    UserIntentMetricFewShotExamples,
    UserIntentSectionScores,
)
from brown.models import ModelConfig, SupportedModels


@pytest.fixture
def mock_user_intent_scores_perfect() -> UserIntentArticleScores:
    """
    Fixture for a perfect UserIntentArticleScores response.
    """
    return UserIntentArticleScores(
        sections=[
            UserIntentSectionScores(
                title="Introduction",
                scores=UserIntentCriteriaScores(
                    guideline_adherence=CriterionScore(score=1, reason="Perfect adherence to guidelines."),
                    research_anchoring=CriterionScore(score=1, reason="Perfect research anchoring."),
                ),
            ),
            UserIntentSectionScores(
                title="Body",
                scores=UserIntentCriteriaScores(
                    guideline_adherence=CriterionScore(score=1, reason="Perfect adherence to guidelines."),
                    research_anchoring=CriterionScore(score=1, reason="Perfect research anchoring."),
                ),
            ),
        ]
    )


@pytest.fixture
def mock_user_intent_scores_mixed() -> UserIntentArticleScores:
    """
    Fixture for a mixed UserIntentArticleScores response.
    """
    return UserIntentArticleScores(
        sections=[
            UserIntentSectionScores(
                title="Introduction",
                scores=UserIntentCriteriaScores(
                    guideline_adherence=CriterionScore(score=0, reason="Poor adherence to guidelines."),
                    research_anchoring=CriterionScore(score=1, reason="Good research anchoring."),
                ),
            ),
            UserIntentSectionScores(
                title="Body",
                scores=UserIntentCriteriaScores(
                    guideline_adherence=CriterionScore(score=1, reason="Good adherence to guidelines."),
                    research_anchoring=CriterionScore(score=0, reason="Poor research anchoring."),
                ),
            ),
        ]
    )


@pytest.fixture
def mock_user_intent_scores_poor() -> UserIntentArticleScores:
    """
    Fixture for a poor UserIntentArticleScores response.
    """
    return UserIntentArticleScores(
        sections=[
            UserIntentSectionScores(
                title="Introduction",
                scores=UserIntentCriteriaScores(
                    guideline_adherence=CriterionScore(score=0, reason="Poor adherence to guidelines."),
                    research_anchoring=CriterionScore(score=0, reason="Poor research anchoring."),
                ),
            ),
        ]
    )


@pytest.fixture
def mock_user_intent_scores_empty_sections() -> UserIntentArticleScores:
    """
    Fixture for a UserIntentArticleScores response with empty sections.
    """
    return UserIntentArticleScores(sections=[])


@pytest.fixture
def mock_user_intent_metric(mock_user_intent_scores_perfect: UserIntentArticleScores) -> UserIntentMetricLLMJudge:
    """
    Fixture for a UserIntentMetric instance with a mocked perfect response.
    """
    model_config = ModelConfig(mocked_response=mock_user_intent_scores_perfect)
    return UserIntentMetricLLMJudge(model=SupportedModels.FAKE_MODEL, model_config=model_config)


@pytest.fixture
def mock_user_intent_metric_mixed(mock_user_intent_scores_mixed: UserIntentArticleScores) -> UserIntentMetricLLMJudge:
    """
    Fixture for a UserIntentMetric instance with a mocked mixed response.
    """
    model_config = ModelConfig(mocked_response=mock_user_intent_scores_mixed)
    return UserIntentMetricLLMJudge(model=SupportedModels.FAKE_MODEL, model_config=model_config)


@pytest.fixture
def mock_user_intent_metric_poor(mock_user_intent_scores_poor: UserIntentArticleScores) -> UserIntentMetricLLMJudge:
    """
    Fixture for a UserIntentMetric instance with a mocked poor response.
    """
    model_config = ModelConfig(mocked_response=mock_user_intent_scores_poor)
    return UserIntentMetricLLMJudge(model=SupportedModels.FAKE_MODEL, model_config=model_config)


@pytest.fixture
def mock_user_intent_metric_empty_sections(
    mock_user_intent_scores_empty_sections: UserIntentArticleScores,
) -> UserIntentMetricLLMJudge:
    """
    Fixture for a UserIntentMetric instance with a mocked empty sections response.
    """
    model_config = ModelConfig(mocked_response=mock_user_intent_scores_empty_sections)
    return UserIntentMetricLLMJudge(model=SupportedModels.FAKE_MODEL, model_config=model_config)


def test_user_intent_metric_perfect_score(mock_user_intent_metric: UserIntentMetricLLMJudge) -> None:
    """
    Test that UserIntentMetric returns perfect scores when mocked with a perfect response.
    """
    input_guideline = "This is the article guideline."
    context = {"research": "This is the research content."}
    output = "This is a well-written article that follows guidelines and uses research."

    results = mock_user_intent_metric.score(input=input_guideline, context=context, output=output)

    assert isinstance(results, list) and len(results) == 2
    for result in results:
        assert result.value == 1.0
        assert result.name.startswith("user_intent_")

    # Check that we have both expected dimensions
    result_names = {result.name for result in results}
    assert "user_intent_guideline_adherence" in result_names
    assert "user_intent_research_anchoring" in result_names


def test_user_intent_metric_mixed_scores(mock_user_intent_metric_mixed: UserIntentMetricLLMJudge) -> None:
    """
    Test that UserIntentMetric returns mixed scores when mocked with a mixed response.
    """
    input_guideline = "This is the article guideline."
    context = {"research": "This is the research content."}
    output = "This article has some issues with guidelines or research anchoring."

    results = mock_user_intent_metric_mixed.score(input=input_guideline, context=context, output=output)

    assert isinstance(results, list) and len(results) == 2

    # Expected averages for mixed scores (0, 1) per section
    # (0+1)/2 = 0.5
    for result in results:
        assert result.value == 0.5
        assert result.name.startswith("user_intent_")


def test_user_intent_metric_poor_scores(mock_user_intent_metric_poor: UserIntentMetricLLMJudge) -> None:
    """
    Test that UserIntentMetric returns poor scores when mocked with a poor response.
    """
    input_guideline = "This is the article guideline."
    context = {"research": "This is the research content."}
    output = "This article doesn't follow guidelines and lacks research anchoring."

    results = mock_user_intent_metric_poor.score(input=input_guideline, context=context, output=output)

    assert isinstance(results, list) and len(results) == 2

    # Expected averages for poor scores (all 0s)
    for result in results:
        assert result.value == 0.0
        assert result.name.startswith("user_intent_")


def test_user_intent_metric_empty_sections(mock_user_intent_metric_empty_sections: UserIntentMetricLLMJudge) -> None:
    """
    Test that UserIntentMetric handles empty sections gracefully by returning an empty list.
    """
    input_guideline = ""
    context = {"research": ""}
    output = ""

    results = mock_user_intent_metric_empty_sections.score(input=input_guideline, context=context, output=output)

    # Should return an empty list when there are no sections
    assert isinstance(results, list)
    assert len(results) == 0


def test_user_intent_metric_missing_research_in_context() -> None:
    """
    Test that UserIntentMetric raises ValueError when context doesn't contain 'research' key.
    """
    model_config = ModelConfig(mocked_response=UserIntentArticleScores(sections=[]))
    metric = UserIntentMetricLLMJudge(
        model=SupportedModels.FAKE_MODEL,
        model_config=model_config,
    )

    input_guideline = "This is the article guideline."
    context = {"other_key": "value"}  # Missing 'research' key
    output = "This is the output."

    with pytest.raises(ValueError, match="Context must contain a 'research' key"):
        metric.score(input=input_guideline, context=context, output=output)


def test_user_intent_metric_init_requires_mocked_response_for_fake_model() -> None:
    """
    Test that UserIntentMetric raises an AssertionError if FAKE_MODEL is used without a mocked_response.
    """
    with pytest.raises(AssertionError, match="Mocked response is required for fake model"):
        model_config = ModelConfig(mocked_response=None)
        UserIntentMetricLLMJudge(model=SupportedModels.FAKE_MODEL, model_config=model_config)


def test_user_intent_metric_init_default_model() -> None:
    """
    Test that UserIntentMetric initializes with the default model if not specified.
    """
    metric = UserIntentMetricLLMJudge()
    assert metric.model == SupportedModels.GOOGLE_GEMINI_25_FLASH


def test_user_intent_metric_init_custom_name() -> None:
    """
    Test that UserIntentMetric initializes with a custom name.
    """
    custom_name = "my_custom_user_intent_metric"
    metric = UserIntentMetricLLMJudge(name=custom_name)
    assert metric.name == custom_name


def test_user_intent_metric_init_custom_few_shot_examples() -> None:
    """
    Test that UserIntentMetric initializes with default few-shot examples.
    """
    metric = UserIntentMetricLLMJudge()
    # Verify that the metric has few_shot_examples attribute and it's of the correct type
    assert hasattr(metric, "few_shot_examples")
    assert isinstance(metric.few_shot_examples, UserIntentMetricFewShotExamples)


def test_user_intent_metric_score_result_structure(mock_user_intent_metric: UserIntentMetricLLMJudge) -> None:
    """
    Test that UserIntentMetric returns results with the correct structure and naming.
    """
    input_guideline = "Article guideline content."
    context = {"research": "Research content."}
    output = "Generated article content."

    results = mock_user_intent_metric.score(input=input_guideline, context=context, output=output)

    assert isinstance(results, list) and len(results) == 2

    # Verify we have exactly the expected metric names
    result_dict = {result.name: result for result in results}
    assert "user_intent_guideline_adherence" in result_dict
    assert "user_intent_research_anchoring" in result_dict

    # Verify each result has the expected attributes
    for result in results:
        assert hasattr(result, "name")
        assert hasattr(result, "value")
        assert hasattr(result, "reason")
        assert isinstance(result.value, float)
        assert 0.0 <= result.value <= 1.0


@pytest.mark.asyncio
async def test_user_intent_metric_async_score(mock_user_intent_metric: UserIntentMetricLLMJudge) -> None:
    """
    Test that UserIntentMetric async score method works correctly.
    """
    input_guideline = "Article guideline content."
    context = {"research": "Research content."}
    output = "Generated article content."

    results = await mock_user_intent_metric.ascore(input=input_guideline, context=context, output=output)

    assert isinstance(results, list) and len(results) == 2
    for result in results:
        assert result.value == 1.0  # Perfect scores from the mock
        assert result.name.startswith("user_intent_")


def test_user_intent_scores_to_context() -> None:
    """
    Test that UserIntentSectionScores.to_context() generates the expected XML format.
    """
    section_scores = UserIntentSectionScores(
        title="Test Section",
        scores=UserIntentCriteriaScores(
            guideline_adherence=CriterionScore(score=1, reason="Good adherence"),
            research_anchoring=CriterionScore(score=0, reason="Poor anchoring"),
        ),
    )

    context_xml = section_scores.to_context()

    assert "<section_scores>" in context_xml
    assert "<section_title>Test Section</section_title>" in context_xml
    assert "<guideline_adherence>" in context_xml
    assert "<score>1</score>" in context_xml
    assert "<reason>Good adherence</reason>" in context_xml
    assert "<research_anchoring>" in context_xml
    assert "<score>0</score>" in context_xml
    assert "<reason>Poor anchoring</reason>" in context_xml


def test_user_intent_article_scores_to_context() -> None:
    """
    Test that UserIntentArticleScores.to_context() generates the expected XML format.
    """
    article_scores = UserIntentArticleScores(
        sections=[
            UserIntentSectionScores(
                title="Section 1",
                scores=UserIntentCriteriaScores(
                    guideline_adherence=CriterionScore(score=1, reason="Good"),
                    research_anchoring=CriterionScore(score=0, reason="Poor"),
                ),
            ),
        ]
    )

    context_xml = article_scores.to_context()

    assert "<article_scores>" in context_xml
    assert "Section 1" in context_xml
    assert "guideline_adherence" in context_xml
    assert "research_anchoring" in context_xml
