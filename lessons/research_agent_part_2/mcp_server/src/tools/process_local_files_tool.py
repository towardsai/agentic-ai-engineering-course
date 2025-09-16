"""Local files processing tool implementation."""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

from ..app.notebook_handler import NotebookToMarkdownConverter
from ..config.constants import (
    GUIDELINES_FILENAMES_FILE,
    LOCAL_FILES_FROM_RESEARCH_FOLDER,
    NOVA_FOLDER,
)
from ..utils.file_utils import validate_guidelines_filenames_file, validate_research_folder

logger = logging.getLogger(__name__)


def build_result_message(
    research_directory: str,
    processed: int,
    local_files: List[str],
    dest_folder: Path,
    warnings: List[str],
    errors: List[str],
) -> str:
    """
    Build a comprehensive result message for the local files processing operation.

    Args:
        research_directory: Path to the research directory
        processed: Number of files successfully processed
        local_files: List of all local files that were attempted
        dest_folder: Destination folder where files were copied
        warnings: List of warning messages
        errors: List of error messages

    Returns:
        Formatted result message string
    """
    # Build result message
    result_parts = [
        f"Processed local files in research folder '{research_directory}'.",
        f"Successfully copied {processed}/{len(local_files)} files to {dest_folder.name}/",
    ]

    if warnings:
        result_parts.append(f"Warnings: {len(warnings)} files not found")

    if errors:
        result_parts.append(f"Errors: {len(errors)} files failed to copy")

    result_message = " | ".join(result_parts)

    # Add details if there were issues
    if warnings or errors:
        details = []
        if warnings:
            details.extend([f"Missing: {w}" for w in warnings[:3]])  # Show first 3
            if len(warnings) > 3:
                details.append(f"... and {len(warnings) - 3} more missing files")

        if errors:
            details.extend([f"Error: {e}" for e in errors[:3]])  # Show first 3
            if len(errors) > 3:
                details.append(f"... and {len(errors) - 3} more errors")

        result_message += f". Details: {'; '.join(details)}"

    return result_message


def process_local_files_tool(research_directory: str) -> Dict[str, Any]:
    """
    Process local files referenced in the article guidelines.

    Reads the guidelines JSON file and copies each referenced local file
    to the local files subfolder. Path separators in filenames are
    replaced with underscores to avoid creating nested folders.

    Args:
        research_directory: Path to the research directory containing the guidelines JSON file

    Returns:
        Dict with status, processing results, and file paths
    """
    logger.debug(f"Processing local files from research folder: {research_directory}")

    # Convert to Path object
    research_path = Path(research_directory)
    nova_path = research_path / NOVA_FOLDER

    # Validate folders and files
    validate_research_folder(research_path)

    # Look for GUIDELINES_FILENAMES_FILE
    metadata_path = nova_path / GUIDELINES_FILENAMES_FILE

    # Validate the guidelines filenames file
    validate_guidelines_filenames_file(metadata_path)

    # Load JSON metadata
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    local_files = data.get("local_files", [])

    if not local_files:
        return {
            "status": "success",
            "message": f"No local files to process in research folder '{research_directory}'.",
            "files_processed": 0,
            "files_total": 0,
            "warnings": [],
            "errors": [],
        }

    # Create destination folder if it doesn't exist
    dest_folder = nova_path / LOCAL_FILES_FROM_RESEARCH_FOLDER
    dest_folder.mkdir(parents=True, exist_ok=True)

    processed = 0
    warnings = []
    errors = []
    processed_files = []

    # Initialize notebook converter for .ipynb files
    notebook_converter = NotebookToMarkdownConverter(include_outputs=True, include_metadata=False)

    for rel_path in local_files:
        # Local files are relative to the research folder
        src_path = research_path / rel_path

        if not src_path.exists():
            warnings.append(f"Referenced local file not found: {rel_path}")
            continue

        # Sanitize destination filename (replace path separators with underscores)
        dest_name = rel_path.replace("/", "_").replace("\\", "_")

        try:
            # Handle .ipynb files specially by converting to markdown
            if src_path.suffix.lower() == ".ipynb":
                # Convert .ipynb to .md extension for destination
                dest_name = dest_name.rsplit(".ipynb", 1)[0] + ".md"
                dest_path = dest_folder / dest_name

                # Convert notebook to markdown string
                markdown_content = notebook_converter.convert_notebook_to_string(src_path)

                # Write markdown content to destination
                dest_path.write_text(markdown_content, encoding="utf-8")
            else:
                # For other file types, copy as before
                dest_path = dest_folder / dest_name
                shutil.copy2(src_path, dest_path)

            processed += 1
            processed_files.append(dest_name)
        except Exception as e:
            errors.append(f"Failed to process {rel_path}: {str(e)}")

    # Build result message using the dedicated function
    result_message = build_result_message(research_directory, processed, local_files, dest_folder, warnings, errors)

    return {
        "status": "success" if processed > 0 else "warning",
        "files_processed": processed,
        "files_total": len(local_files),
        "processed_files": processed_files,
        "warnings": warnings,
        "errors": errors,
        "output_directory": str(dest_folder.resolve()),
        "message": result_message,
    }
