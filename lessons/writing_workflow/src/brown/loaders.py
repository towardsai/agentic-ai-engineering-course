from pathlib import Path
from typing import TypedDict

from brown.base import Loader
from brown.entities import profiles
from brown.entities.articles import Article, ArticleExample, ArticleExamples
from brown.entities.exceptions import InputNotFoundException
from brown.entities.guidelines import ArticleGuideline
from brown.entities.profiles import ArticleProfiles
from brown.entities.research import Research
from brown.utils.s import clean_markdown_links


class MarkdownArticleLoader(Loader):
    def load(self, *, working_uri: Path) -> Article:
        article_path = working_uri / self.uri
        if not article_path.exists():
            raise InputNotFoundException(working_uri)

        raw_markdown_content = article_path.read_text()

        return Article(content=raw_markdown_content)


class MarkdownArticleGuidelineLoader(Loader):
    def load(self, *, working_uri: Path) -> ArticleGuideline:
        """
        Parses the markdown file and returns an ArticleGuideline object.

        Returns:
            ArticleGuideline: Parsed guideline object containing global context and outline sections.
        """

        article_guideline_path = working_uri / self.uri
        if not article_guideline_path.exists():
            raise InputNotFoundException(article_guideline_path)

        raw_markdown_content = article_guideline_path.read_text()

        return ArticleGuideline(content=raw_markdown_content)


class MarkdownResearchLoader(Loader):
    def load(self, *, working_uri: Path) -> Research:
        research_path = working_uri / self.uri
        if not research_path.exists():
            raise InputNotFoundException(research_path)

        raw_markdown_content = research_path.read_text()

        cleaned_content = clean_markdown_links(raw_markdown_content)

        return Research(content=cleaned_content)


class ArticleProfileLoaderInput(TypedDict):
    article: Path

    character: Path
    mechanics: Path
    structure: Path
    terminology: Path
    tonality: Path


class MarkdownArticleProfilesLoader(Loader):
    profiles_mapping = {
        "character": profiles.CharacterProfile,
        "article": profiles.ArticleProfile,
        "structure": profiles.StructureProfile,
        "mechanics": profiles.MechanicsProfile,
        "terminology": profiles.TerminologyProfile,
        "tonality": profiles.TonalityProfile,
    }

    def __init__(self, uri: ArticleProfileLoaderInput) -> None:
        super().__init__(uri)

    @classmethod
    def get_supported_profiles(cls) -> list[str]:
        return list(cls.profiles_mapping.keys())

    def load(self, working_uri: Path | None = None, *args, **kwargs) -> ArticleProfiles:
        profiles = {}

        for profile_name, profile_path in self.uri.items():
            if not profile_path.exists():
                raise InputNotFoundException(profile_path)

            raw_profile_content = profile_path.read_text()
            cleaned_profile_content = clean_markdown_links(raw_profile_content)

            ProfileType = self.profiles_mapping[profile_name]

            profiles[profile_name] = ProfileType(name=profile_name, content=cleaned_profile_content)

        return ArticleProfiles(**profiles)


class MarkdownArticleExampleLoader(Loader):
    def load(self, working_uri: Path | None = None, *args, **kwargs) -> ArticleExamples:
        if not self.uri.exists():
            raise InputNotFoundException(self.uri)

        examples = []
        for file_path in self.uri.glob("*.md"):
            content = file_path.read_text()
            example = ArticleExample(content=content)
            examples.append(example)

        return ArticleExamples(examples=examples)
