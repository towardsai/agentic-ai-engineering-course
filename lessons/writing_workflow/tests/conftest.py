"""Root-level pytest configuration and shared fixtures."""

import os
from pathlib import Path
from typing import Any

import pytest

from brown.entities.articles import Article, ArticleExample, ArticleExamples, SelectedText
from brown.entities.guidelines import ArticleGuideline
from brown.entities.media_items import MediaItems
from brown.entities.profiles import (
    ArticleProfile,
    ArticleProfiles,
    CharacterProfile,
    MechanicsProfile,
    StructureProfile,
    TerminologyProfile,
    TonalityProfile,
)
from brown.entities.research import Research
from brown.entities.reviews import ArticleReviews, HumanFeedback, Review
from brown.loaders import (
    MarkdownArticleGuidelineLoader,
    MarkdownArticleProfilesLoader,
    MarkdownResearchLoader,
)
from brown.models import ModelConfig


@pytest.fixture(autouse=True, scope="session")
def setup_config_file() -> None:
    """Set CONFIG_FILE environment variable to use mocked config for all tests."""

    config_path = Path("tests/fixtures/configs/mocked.yaml")
    os.environ["CONFIG_FILE"] = str(config_path)


@pytest.fixture
def mock_research_content() -> str:
    """Basic research content for testing."""
    return """# Research: AI Agents

## Introduction
AI agents are autonomous systems that can perceive, reason, and act in their environment.

## Key Concepts
- **Perception**: Understanding the environment
- **Reasoning**: Making decisions based on information
- **Action**: Executing tasks to achieve goals

## Applications
AI agents are used in various domains including:
- Customer service chatbots
- Autonomous vehicles
- Smart home systems
- Financial trading systems

## Future Directions
The field continues to evolve with advances in:
- Large language models
- Multi-agent systems
- Reinforcement learning
"""


@pytest.fixture
def mock_article_guideline_content() -> str:
    """Basic article guideline content for testing."""
    return """# Article Guideline: AI Agents

## Objective
Write a comprehensive article about AI agents that educates readers on the fundamentals and applications.

## Target Audience
- Software engineers
- AI practitioners
- Technical decision makers

## Key Points to Cover
1. Definition and core concepts
2. Real-world applications
3. Technical implementation details
4. Future trends and challenges

## Writing Style
- Technical but accessible
- Use concrete examples
- Include code snippets where relevant
- Maintain professional tone
"""


@pytest.fixture
def mock_character_profile_content() -> str:
    """Character profile content for testing."""
    return """# Character Profile: Paul Iusztin

## Background
Senior AI Engineer and founder of Decoding AI Magazine.

## Writing Style
- Direct and technical
- No-nonsense approach
- Focus on practical implementation
- Minimal fluff, maximum value

## Expertise Areas
- AI Engineering
- Production AI Systems
- MLOps/LLMOps
- RAG and LLMs
"""


@pytest.fixture
def mock_article_content() -> str:
    """Sample article content for testing."""
    return """# Understanding AI Agents: A Technical Deep Dive

## Introduction

AI agents represent a paradigm shift in how we think about artificial intelligence systems. Unlike traditional AI that operates in 
isolation, agents are designed to interact with their environment, make decisions, and take actions to achieve specific goals.

## What Are AI Agents?

An AI agent is an autonomous system that can:
- **Perceive** its environment through sensors or data inputs
- **Reason** about the information it receives
- **Act** to influence its environment and achieve objectives

## Technical Architecture

The typical architecture of an AI agent includes:

1. **Perception Layer**: Processes raw input data
2. **Reasoning Engine**: Makes decisions based on available information
3. **Action Interface**: Executes decisions in the environment
4. **Memory System**: Stores and retrieves relevant information

## Conclusion

AI agents represent the future of intelligent systems, offering unprecedented capabilities for automation and decision-making.
"""


@pytest.fixture
def mock_research(tmp_path: Path, mock_research_content: str) -> Research:
    """Create a mock Research object from file content."""
    research_file = tmp_path / "research.md"
    research_file.write_text(mock_research_content)
    loader = MarkdownResearchLoader(uri="research.md")
    return loader.load(working_uri=tmp_path)


@pytest.fixture
def mock_article_guideline(tmp_path: Path, mock_article_guideline_content: str) -> ArticleGuideline:
    """Create a mock ArticleGuideline object from file content."""
    guideline_file = tmp_path / "article_guideline.md"
    guideline_file.write_text(mock_article_guideline_content)
    loader = MarkdownArticleGuidelineLoader(uri="article_guideline.md")
    return loader.load(working_uri=tmp_path)


@pytest.fixture
def mock_article(mock_article_content: str) -> Article:
    """Create a mock Article object."""
    return Article(content=mock_article_content)


@pytest.fixture
def mock_article_example(mock_article_content: str) -> ArticleExample:
    """Create a mock ArticleExample object."""
    return ArticleExample(content=mock_article_content)


@pytest.fixture
def mock_article_examples(mock_article_content: str) -> ArticleExamples:
    """Create mock ArticleExamples with multiple examples."""
    examples = [
        ArticleExample(content=mock_article_content),
        ArticleExample(content="# Another Example\n\nThis is another example article."),
    ]
    return ArticleExamples(examples=examples)


@pytest.fixture
def mock_character_profile(mock_character_profile_content: str) -> CharacterProfile:
    """Create a mock CharacterProfile object."""
    return CharacterProfile(name="character", content=mock_character_profile_content)


@pytest.fixture
def mock_article_profiles(tmp_path: Path, mock_character_profile_content: str) -> ArticleProfiles:
    """Create a mock ArticleProfiles object with all profile types."""
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()

    # Create profile files
    profile_files = {
        "article": profiles_dir / "article_profile.md",
        "character": profiles_dir / "character_profiles" / "paul_iusztin.md",
        "mechanics": profiles_dir / "mechanics_profile.md",
        "structure": profiles_dir / "structure_profile.md",
        "terminology": profiles_dir / "terminology_profile.md",
        "tonality": profiles_dir / "tonality_profile.md",
    }

    # Create character profiles directory
    (profiles_dir / "character_profiles").mkdir()

    # Write profile content
    profile_contents = {
        "article": "# Article Profile\n\nWrite technical articles with clear structure.",
        "character": mock_character_profile_content,
        "mechanics": "# Mechanics Profile\n\nUse proper grammar and formatting.",
        "structure": "# Structure Profile\n\nFollow standard article structure.",
        "terminology": "# Terminology Profile\n\nUse precise technical terms.",
        "tonality": "# Tonality Profile\n\nMaintain professional and engaging tone.",
    }

    for profile_type, file_path in profile_files.items():
        file_path.write_text(profile_contents[profile_type])

    # Load profiles using the loader
    loader = MarkdownArticleProfilesLoader(uri=profile_files)
    return loader.load(working_uri=tmp_path)


@pytest.fixture
def mock_model_config() -> ModelConfig:
    """Create a mock ModelConfig for testing."""
    return ModelConfig(
        temperature=0.7,
        include_thoughts=False,
        thinking_budget=1000,
        mocked_response=None,
    )


@pytest.fixture
def mock_model_config_with_response() -> ModelConfig:
    """Create a mock ModelConfig with mocked response for testing."""
    return ModelConfig(
        temperature=0.0,
        include_thoughts=False,
        thinking_budget=None,
        mocked_response={"content": "This is a mocked response"},
    )


@pytest.fixture
def mock_structured_response() -> dict[str, Any]:
    """Mock structured response for testing."""
    return {
        "sections": [
            {
                "title": "Introduction",
                "content": "This is the introduction section.",
            },
            {
                "title": "Body",
                "content": "This is the body section with detailed content.",
            },
        ]
    }


@pytest.fixture
def mock_workflow_progress() -> dict[str, Any]:
    """Mock workflow progress data for testing."""
    return {
        "progress": 50,
        "message": "Processing article content",
    }


# Pydantic Model Fixtures (simple constructors, no file I/O)
@pytest.fixture
def mock_character_profile_simple() -> CharacterProfile:
    """Mock CharacterProfile for testing (simple constructor)."""
    return CharacterProfile(name="test_character", content="Test character profile content")


@pytest.fixture
def mock_article_profile() -> ArticleProfile:
    """Mock ArticleProfile for testing."""
    return ArticleProfile(name="test_article", content="Test article profile content")


@pytest.fixture
def mock_structure_profile() -> StructureProfile:
    """Mock StructureProfile for testing."""
    return StructureProfile(name="test_structure", content="Test structure profile content")


@pytest.fixture
def mock_mechanics_profile() -> MechanicsProfile:
    """Mock MechanicsProfile for testing."""
    return MechanicsProfile(name="test_mechanics", content="Test mechanics profile content")


@pytest.fixture
def mock_terminology_profile() -> TerminologyProfile:
    """Mock TerminologyProfile for testing."""
    return TerminologyProfile(name="test_terminology", content="Test terminology profile content")


@pytest.fixture
def mock_tonality_profile() -> TonalityProfile:
    """Mock TonalityProfile for testing."""
    return TonalityProfile(name="test_tonality", content="Test tonality profile content")


@pytest.fixture
def mock_article_profiles_simple(
    mock_character_profile_simple: CharacterProfile,
    mock_article_profile: ArticleProfile,
    mock_structure_profile: StructureProfile,
    mock_mechanics_profile: MechanicsProfile,
    mock_terminology_profile: TerminologyProfile,
    mock_tonality_profile: TonalityProfile,
) -> ArticleProfiles:
    """Mock ArticleProfiles for testing (simple constructor)."""
    return ArticleProfiles(
        character=mock_character_profile_simple,
        article=mock_article_profile,
        structure=mock_structure_profile,
        mechanics=mock_mechanics_profile,
        terminology=mock_terminology_profile,
        tonality=mock_tonality_profile,
    )


@pytest.fixture
def mock_article_guideline_simple() -> ArticleGuideline:
    """Mock ArticleGuideline for testing (simple constructor)."""
    return ArticleGuideline(content="Test article guideline content")


@pytest.fixture
def mock_research_simple() -> Research:
    """Mock Research for testing (simple constructor)."""
    return Research(content="Test research content")


@pytest.fixture
def mock_article_examples_simple() -> ArticleExamples:
    """Mock ArticleExamples for testing (simple constructor)."""
    return ArticleExamples(examples=[])


@pytest.fixture
def mock_media_items() -> MediaItems:
    """Mock MediaItems for testing."""
    return MediaItems(media_items=[])


@pytest.fixture
def mock_review() -> Review:
    """Mock Review for testing."""
    return Review(profile="test_profile", location="test_location", comment="Test review comment")


@pytest.fixture
def mock_article_reviews(mock_article: Article, mock_review: Review) -> ArticleReviews:
    """Mock ArticleReviews for testing."""
    return ArticleReviews(article=mock_article, reviews=[mock_review])


@pytest.fixture
def mock_article_simple() -> Article:
    """Mock Article for testing (simple constructor)."""
    return Article(content="# Test Article\n\nTest content")


@pytest.fixture
def mock_selected_text() -> SelectedText:
    """Mock SelectedText for testing."""
    return SelectedText(content="Test selected text content", first_line_number=1, last_line_number=5)


@pytest.fixture
def mock_human_feedback() -> HumanFeedback:
    """Mock HumanFeedback for testing."""
    return HumanFeedback(content="Test human feedback")
