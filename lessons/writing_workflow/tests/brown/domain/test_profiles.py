"""Tests for brown.entities.profiles module."""

from brown.entities.profiles import (
    ArticleProfile,
    ArticleProfiles,
    CharacterProfile,
    MechanicsProfile,
    Profile,
    StructureProfile,
    TerminologyProfile,
    TonalityProfile,
)


class TestProfile:
    """Test the base Profile class."""

    def test_profile_creation(self) -> None:
        """Test creating a profile."""
        profile = Profile(name="test", content="Test content")
        assert profile.name == "test"
        assert profile.content == "Test content"

    def test_profile_xml_tag(self) -> None:
        """Test XML tag generation from class name."""
        profile = Profile(name="test", content="Content")
        assert profile.xml_tag == "profile"

    def test_profile_to_context(self) -> None:
        """Test context generation."""
        profile = Profile(name="test", content="Test content")
        context = profile.to_context()

        assert "<profile>" in context
        assert "</profile>" in context
        assert "Test content" in context


class TestCharacterProfile:
    """Test the CharacterProfile class."""

    def test_character_profile_creation(self) -> None:
        """Test creating a character profile."""
        profile = CharacterProfile(name="character", content="Character content")
        assert profile.name == "character"
        assert profile.content == "Character content"

    def test_character_profile_to_context(self) -> None:
        """Test character profile context generation."""
        profile = CharacterProfile(name="character", content="Paul Iusztin's style")
        context = profile.to_context()

        assert "<character_profile>" in context
        assert "</character_profile>" in context
        assert "Paul Iusztin's style" in context


class TestArticleProfile:
    """Test the ArticleProfile class."""

    def test_article_profile_creation(self) -> None:
        """Test creating an article profile."""
        profile = ArticleProfile(name="article", content="Article guidelines")
        assert profile.name == "article"
        assert profile.content == "Article guidelines"

    def test_article_profile_to_context(self) -> None:
        """Test article profile context generation."""
        profile = ArticleProfile(name="article", content="Write technical articles")
        context = profile.to_context()

        assert "<article_profile>" in context
        assert "</article_profile>" in context
        assert "Write technical articles" in context


class TestStructureProfile:
    """Test the StructureProfile class."""

    def test_structure_profile_creation(self) -> None:
        """Test creating a structure profile."""
        profile = StructureProfile(name="structure", content="Structure rules")
        assert profile.name == "structure"
        assert profile.content == "Structure rules"

    def test_structure_profile_to_context(self) -> None:
        """Test structure profile context generation."""
        profile = StructureProfile(name="structure", content="Follow standard structure")
        context = profile.to_context()

        assert "<structure_profile>" in context
        assert "</structure_profile>" in context
        assert "Follow standard structure" in context


class TestMechanicsProfile:
    """Test the MechanicsProfile class."""

    def test_mechanics_profile_creation(self) -> None:
        """Test creating a mechanics profile."""
        profile = MechanicsProfile(name="mechanics", content="Grammar rules")
        assert profile.name == "mechanics"
        assert profile.content == "Grammar rules"

    def test_mechanics_profile_to_context(self) -> None:
        """Test mechanics profile context generation."""
        profile = MechanicsProfile(name="mechanics", content="Use proper grammar")
        context = profile.to_context()

        assert "<mechanics_profile>" in context
        assert "</mechanics_profile>" in context
        assert "Use proper grammar" in context


class TestTerminologyProfile:
    """Test the TerminologyProfile class."""

    def test_terminology_profile_creation(self) -> None:
        """Test creating a terminology profile."""
        profile = TerminologyProfile(name="terminology", content="Terminology rules")
        assert profile.name == "terminology"
        assert profile.content == "Terminology rules"

    def test_terminology_profile_to_context(self) -> None:
        """Test terminology profile context generation."""
        profile = TerminologyProfile(name="terminology", content="Use precise terms")
        context = profile.to_context()

        assert "<terminology_profile>" in context
        assert "</terminology_profile>" in context
        assert "Use precise terms" in context


class TestTonalityProfile:
    """Test the TonalityProfile class."""

    def test_tonality_profile_creation(self) -> None:
        """Test creating a tonality profile."""
        profile = TonalityProfile(name="tonality", content="Tone guidelines")
        assert profile.name == "tonality"
        assert profile.content == "Tone guidelines"

    def test_tonality_profile_to_context(self) -> None:
        """Test tonality profile context generation."""
        profile = TonalityProfile(name="tonality", content="Maintain professional tone")
        context = profile.to_context()

        assert "<tonality_profile>" in context
        assert "</tonality_profile>" in context
        assert "Maintain professional tone" in context


class TestArticleProfiles:
    """Test the ArticleProfiles class."""

    def test_article_profiles_creation(self) -> None:
        """Test creating complete article profiles."""
        character = CharacterProfile(name="character", content="Character content")
        article = ArticleProfile(name="article", content="Article content")
        structure = StructureProfile(name="structure", content="Structure content")
        mechanics = MechanicsProfile(name="mechanics", content="Mechanics content")
        terminology = TerminologyProfile(name="terminology", content="Terminology content")
        tonality = TonalityProfile(name="tonality", content="Tonality content")

        profiles = ArticleProfiles(
            character=character,
            article=article,
            structure=structure,
            mechanics=mechanics,
            terminology=terminology,
            tonality=tonality,
        )

        assert profiles.character == character
        assert profiles.article == article
        assert profiles.structure == structure
        assert profiles.mechanics == mechanics
        assert profiles.terminology == terminology
        assert profiles.tonality == tonality

    def test_article_profiles_to_context(self) -> None:
        """Test article profiles context generation."""
        character = CharacterProfile(name="character", content="Character content")
        article = ArticleProfile(name="article", content="Article content")
        structure = StructureProfile(name="structure", content="Structure content")
        mechanics = MechanicsProfile(name="mechanics", content="Mechanics content")
        terminology = TerminologyProfile(name="terminology", content="Terminology content")
        tonality = TonalityProfile(name="tonality", content="Tonality content")

        profiles = ArticleProfiles(
            character=character,
            article=article,
            structure=structure,
            mechanics=mechanics,
            terminology=terminology,
            tonality=tonality,
        )

        # ArticleProfiles doesn't have a to_context method, so we test individual profiles
        character_context = profiles.character.to_context()
        article_context = profiles.article.to_context()
        structure_context = profiles.structure.to_context()

        # Should contain profile contexts
        assert "<character_profile>" in character_context
        assert "<article_profile>" in article_context
        assert "<structure_profile>" in structure_context

        # Should contain the content
        assert "Character content" in character_context
        assert "Article content" in article_context
        assert "Structure content" in structure_context
