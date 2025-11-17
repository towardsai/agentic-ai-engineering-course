"""Dataset utilities for evaluation samples and data loading.

This module provides data structures and utilities for loading and managing
evaluation datasets containing article guidelines, research materials, and
ground truth articles for evaluation purposes.
"""

import json
from pathlib import Path
from typing import Self, TypedDict

from pydantic.main import BaseModel

from .constants import (
    DEFAULT_ARTICLE_GUIDELINE_PATH,
    DEFAULT_GROUND_TRUTH_ARTICLE_PATH,
    DEFAULT_RESEARCH_PATH,
)


class EvalSampleDict(TypedDict):
    """A TypedDict representing a single evaluation sample for article generation.

    This TypedDict defines the required fields for an evaluation sample used in the
    article generation evaluation process. It contains the core data needed for
    evaluation, including input materials and ground truth articles.

    Attributes:
        name (str): The unique identifier for this evaluation sample.
        directory (Path): Path to the directory containing this sample's files.
        article_guideline (str): The article writing guidelines in markdown format.
        research (str): The research/source material in markdown format.
        ground_truth_article (str): The reference/ground truth article in markdown format.

    """

    name: str
    directory: Path
    article_guideline: str
    research: str
    ground_truth_article: str
    is_few_shot_example: bool


class EvalSample(BaseModel):
    """Represents a single evaluation sample containing article data.

    This model holds all the necessary data for evaluating article generation,
    including the input data (guidelines and research) and both ground truth
    and generated articles for comparison.

    Attributes:
        name: The name/identifier of the evaluation sample.
        directory: The directory path where the sample files are located.
        article_guideline: The content of the article guideline markdown file.
        research: The content of the research markdown file.
        ground_truth_article: The content of the ground truth article markdown file.
        is_few_shot_example: Whether this sample is used as a few-shot example.

    """

    name: str
    directory: Path
    article_guideline: str
    research: str
    ground_truth_article: str
    is_few_shot_example: bool = False


class EvalDataset(BaseModel):
    """Represents a collection of evaluation samples for article generation assessment.

    This model contains multiple evaluation samples along with metadata about
    the dataset, providing a structured way to organize and manage evaluation data.

    Attributes:
        name: The name of the evaluation dataset.
        description: A description of what the dataset contains and its purpose.
        samples: A list of evaluation samples in this dataset.

    """

    name: str
    description: str
    samples: list[EvalSample]

    @classmethod
    def load_dataset(cls, directory: Path, name: str, description: str) -> Self:
        """Load evaluation dataset from directory containing metadata.json and markdown files.

        Args:
            directory: Path to the evaluation data directory containing metadata.json
            name: The name to assign to the loaded dataset
            description: A description of the dataset's purpose and contents

        Returns:
            EvalDataset instance with loaded markdown content

        Raises:
            FileNotFoundError: If metadata.json or any referenced markdown files are missing
            ValueError: If metadata.json contains invalid structure

        """
        metadata_file = directory / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with metadata_file.open() as f:
            metadata = json.load(f)

        samples = []
        for sample_metadata in metadata:
            sample_dir = directory / sample_metadata["directory"]

            article_guideline = cls._load_markdown_file(
                sample_dir / sample_metadata.get("article_guideline_path", DEFAULT_ARTICLE_GUIDELINE_PATH)
            )
            research = cls._load_markdown_file(sample_dir / sample_metadata.get("research_path", DEFAULT_RESEARCH_PATH))
            ground_truth_article = cls._load_markdown_file(
                sample_dir / sample_metadata.get("ground_truth_article_path", DEFAULT_GROUND_TRUTH_ARTICLE_PATH)
            )

            sample = EvalSample(
                name=sample_metadata["name"],
                directory=sample_metadata["directory"],
                is_few_shot_example=sample_metadata.get("is_few_shot_example", False),
                article_guideline=article_guideline,
                research=research,
                ground_truth_article=ground_truth_article,
            )
            samples.append(sample)

        return cls(name=name, description=description, samples=samples)

    @staticmethod
    def _load_markdown_file(file_path: Path) -> str:
        """Load content from a markdown file.

        Args:
            file_path: Path to the markdown file.

        Returns:
            Content of the markdown file as string.

        Raises:
            FileNotFoundError: If the file doesn't exist.

        """
        if not file_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        with file_path.open(encoding="utf-8") as f:
            return f.read()
