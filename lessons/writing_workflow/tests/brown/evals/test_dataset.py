import json
from pathlib import Path

import pytest

from brown.evals.dataset import EvalDataset


@pytest.fixture
def mock_markdown_files(tmp_path: Path) -> dict[str, Path]:
    """
    Fixture to create mock markdown files for testing.

    Args:
        tmp_path: pytest's built-in fixture for a temporary directory.

    Returns:
        A dictionary mapping file names to their created Path objects.
    """
    files_content = {
        "article_guideline.md": """# Guideline
This is a guideline.""",
        "research.md": """# Research
This is research content.""",
        "article_ground_truth.md": """# Ground Truth
This is the ground truth article.""",
    }
    created_files = {}
    for name, content in files_content.items():
        file_path = tmp_path / name
        file_path.write_text(content)
        created_files[name] = file_path
    return created_files


@pytest.fixture
def mock_eval_dataset_directory(tmp_path: Path, mock_markdown_files: dict[str, Path]) -> Path:
    """
    Fixture to set up a mock evaluation dataset directory structure.

    Args:
        tmp_path: pytest's built-in fixture for a temporary directory.
        mock_markdown_files: Fixture providing mock markdown file paths.

    Returns:
        Path to the created mock dataset directory.
    """
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    data_dir = dataset_dir / "data"
    data_dir.mkdir()
    sample_dir = data_dir / "sample_article"
    sample_dir.mkdir()

    # Create metadata.json
    metadata = [
        {
            "name": "Sample Article 1",
            "directory": "data/sample_article",
            "article_guideline_path": "article_guideline.md",
            "research_path": "research.md",
            "ground_truth_article_path": "article_ground_truth.md",
            "is_few_shot_example": False,
        }
    ]
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata))

    # Copy mock markdown files into the sample directory
    for file_path in mock_markdown_files.values():
        (sample_dir / file_path.name).write_text(file_path.read_text())

    return dataset_dir


@pytest.fixture
def mock_eval_dataset_directory_few_shot(tmp_path: Path, mock_markdown_files: dict[str, Path]) -> Path:
    """
    Fixture to set up a mock evaluation dataset directory structure with a few-shot example.

    Args:
        tmp_path: pytest's built-in fixture for a temporary directory.
        mock_markdown_files: Fixture providing mock markdown file paths.

    Returns:
        Path to the created mock dataset directory.
    """
    dataset_dir = tmp_path / "dataset_few_shot"
    dataset_dir.mkdir()
    data_dir = dataset_dir / "data"
    data_dir.mkdir()
    sample_dir = data_dir / "sample_article_few_shot"
    sample_dir.mkdir()

    metadata = [
        {
            "name": "Sample Article Few Shot",
            "directory": "data/sample_article_few_shot",
            "article_guideline_path": "article_guideline.md",
            "research_path": "research.md",
            "ground_truth_article_path": "article_ground_truth.md",
            "is_few_shot_example": True,
        }
    ]
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata))

    for file_name, file_path in mock_markdown_files.items():
        (sample_dir / file_path.name).write_text(file_path.read_text())

    return dataset_dir


@pytest.fixture
def mock_eval_dataset_directory_missing_file(tmp_path: Path) -> Path:
    """
    Fixture to set up a mock evaluation dataset directory structure with a missing markdown file.

    Args:
        tmp_path: pytest's built-in fixture for a temporary directory.

    Returns:
        Path to the created mock dataset directory.
    """
    dataset_dir = tmp_path / "dataset_missing_file"
    dataset_dir.mkdir()
    data_dir = dataset_dir / "data"
    data_dir.mkdir()
    sample_dir = data_dir / "sample_missing"
    sample_dir.mkdir()

    # Create metadata.json referencing a missing guideline file
    metadata = [
        {
            "name": "Sample Missing Guideline",
            "directory": "data/sample_missing",
            "article_guideline_path": "missing_guideline.md",  # This file will not be created
            "research_path": "research.md",
            "ground_truth_article_path": "article_ground_truth.md",
        }
    ]
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata))

    # Create other files to ensure only guideline is missing
    (sample_dir / "research.md").write_text("# Research")
    (sample_dir / "article_ground_truth.md").write_text("# Ground Truth")

    return dataset_dir


@pytest.fixture
def mock_eval_dataset_directory_empty_metadata(tmp_path: Path) -> Path:
    """
    Fixture for a dataset directory with an empty metadata.json.
    """
    dataset_dir = tmp_path / "dataset_empty_metadata"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.json").write_text("[]")
    return dataset_dir


@pytest.fixture
def mock_eval_dataset_directory_invalid_metadata(tmp_path: Path) -> Path:
    """
    Fixture for a dataset directory with invalid metadata.json (missing 'name').
    """
    dataset_dir = tmp_path / "dataset_invalid_metadata"
    dataset_dir.mkdir()
    data_dir = dataset_dir / "data"
    data_dir.mkdir()
    sample_dir = data_dir / "sample_invalid"
    sample_dir.mkdir()

    metadata = [
        {
            "directory": "data/sample_invalid",
            "article_guideline_path": "article_guideline.md",
            "research_path": "research.md",
            "ground_truth_article_path": "article_ground_truth.md",
        }
    ]
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata))

    (sample_dir / "article_guideline.md").write_text("# Guideline")
    (sample_dir / "research.md").write_text("# Research")
    (sample_dir / "article_ground_truth.md").write_text("# Ground Truth")

    return dataset_dir


def test_load_dataset_success(mock_eval_dataset_directory: Path) -> None:
    """
    Test that EvalDataset.load_dataset successfully loads a well-formed dataset.
    """
    dataset = EvalDataset.load_dataset(directory=mock_eval_dataset_directory, name="Test Dataset", description="A test dataset")

    assert dataset.name == "Test Dataset"
    assert dataset.description == "A test dataset"
    assert len(dataset.samples) == 1

    sample = dataset.samples[0]
    assert sample.name == "Sample Article 1"
    assert sample.directory == Path("data/sample_article")
    assert "# Guideline" in sample.article_guideline
    assert "# Research" in sample.research
    assert "# Ground Truth" in sample.ground_truth_article
    assert sample.is_few_shot_example is False


def test_load_dataset_few_shot_example(mock_eval_dataset_directory_few_shot: Path) -> None:
    """
    Test that EvalDataset.load_dataset correctly loads a few-shot example sample.
    """
    dataset = EvalDataset.load_dataset(directory=mock_eval_dataset_directory_few_shot, name="Test Few Shot", description="")

    assert len(dataset.samples) == 1
    sample = dataset.samples[0]
    assert sample.name == "Sample Article Few Shot"
    assert sample.is_few_shot_example is True


def test_load_dataset_file_not_found(mock_eval_dataset_directory_missing_file: Path) -> None:
    """
    Test that EvalDataset.load_dataset raises FileNotFoundError for missing markdown files.
    """
    with pytest.raises(FileNotFoundError, match="missing_guideline.md"):
        EvalDataset.load_dataset(directory=mock_eval_dataset_directory_missing_file, name="Test Missing File", description="")


def test_load_dataset_metadata_not_found(tmp_path: Path) -> None:
    """
    Test that EvalDataset.load_dataset raises FileNotFoundError if metadata.json is missing.
    """
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        EvalDataset.load_dataset(directory=tmp_path, name="Test No Metadata", description="")


def test_load_dataset_empty_metadata(mock_eval_dataset_directory_empty_metadata: Path) -> None:
    """
    Test that EvalDataset.load_dataset loads an empty list of samples when metadata.json is empty.
    """
    dataset = EvalDataset.load_dataset(directory=mock_eval_dataset_directory_empty_metadata, name="Empty Dataset", description="")
    assert len(dataset.samples) == 0
    assert dataset.name == "Empty Dataset"


def test_load_dataset_invalid_metadata_structure(mock_eval_dataset_directory_invalid_metadata: Path) -> None:
    """
    Test that EvalDataset.load_dataset raises a Pydantic ValidationError for invalid metadata structure.
    """
    with pytest.raises(Exception):  # Pydantic will raise a ValidationError internally
        EvalDataset.load_dataset(directory=mock_eval_dataset_directory_invalid_metadata, name="Invalid Metadata", description="")
