"""Notebook processing utilities."""

import json
import logging
from pathlib import Path
from typing import List

import nbformat

logger = logging.getLogger(__name__)


class NotebookToMarkdownConverter:
    """Convert Jupyter notebooks to markdown format with output preservation."""

    def __init__(self, include_outputs: bool = True, include_metadata: bool = False) -> None:
        """
        Initialize the converter.

        Args:
            include_outputs: Whether to include cell outputs in the markdown.
            include_metadata: Whether to include cell metadata as comments.
        """
        self.include_outputs = include_outputs
        self.include_metadata = include_metadata

    def convert_notebook_to_string(self, notebook_path: Path) -> str:
        """
        Convert a Jupyter notebook to markdown format without writing to disk.

        Args:
            notebook_path: Path to the input notebook file.

        Returns:
            The markdown content as a string.

        Raises:
            FileNotFoundError: If the notebook file doesn't exist.
            ValueError: If the notebook format is invalid.
        """
        if not notebook_path.exists():
            msg = f"Notebook file not found: {notebook_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                notebook = nbformat.read(f, as_version=4)
        except Exception as e:
            msg = f"Failed to read notebook: {e}"
            logger.error(msg, exc_info=True)
            raise ValueError(msg) from e

        return self._convert_notebook_to_markdown(notebook)

    def _convert_notebook_to_markdown(self, notebook: nbformat.NotebookNode) -> str:
        """
        Convert notebook content to markdown format.

        Args:
            notebook: The notebook object from nbformat.

        Returns:
            Markdown content as a string.
        """
        markdown_lines: List[str] = []

        # Add notebook title if available
        if "metadata" in notebook and "title" in notebook.metadata:
            markdown_lines.append(f"# {notebook.metadata.title}\n")

        for cell_idx, cell in enumerate(notebook.cells):
            if self.include_metadata and cell.metadata:
                markdown_lines.append(f"<!-- Cell {cell_idx} metadata: {json.dumps(cell.metadata)} -->\n")

            if cell.cell_type == "markdown":
                markdown_lines.append(cell.source)
                markdown_lines.append("")  # Add spacing

            elif cell.cell_type == "code":
                # Add code block
                markdown_lines.append("```python")
                markdown_lines.append(cell.source)
                markdown_lines.append("```\n")

                # Add outputs if requested and available
                if self.include_outputs and hasattr(cell, "outputs") and cell.outputs:
                    for output in cell.outputs:
                        markdown_lines.extend(self._process_output(output))

                markdown_lines.append("")  # Add spacing

        return "\n".join(markdown_lines)

    def _clip_text(self, text: str, max_length: int = 3000) -> str:
        """
        Clip text to a maximum length, adding ellipsis if truncated.

        Args:
            text: The text to clip.
            max_length: Maximum allowed length.

        Returns:
            The clipped text.
        """
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."

    def _process_output(self, output: nbformat.NotebookNode) -> List[str]:
        """
        Process a single cell output and convert to markdown format.

        Args:
            output: Cell output dictionary.

        Returns:
            List of markdown lines for the output.
        """
        lines: List[str] = []

        if output.output_type == "stream":
            lines.append("**Output:**")
            lines.append("```")
            lines.append(self._clip_text(output.text.rstrip()))
            lines.append("```\n")

        elif output.output_type in ["execute_result", "display_data"]:
            data = output.data

            # Handle text/plain output
            if "text/plain" in data:
                lines.append("**Output:**")
                lines.append("```")
                lines.append(self._clip_text(data["text/plain"].rstrip()))
                lines.append("```\n")

            # Handle HTML output
            if "text/html" in data:
                lines.append("**HTML Output:**")
                lines.append(self._clip_text(data["text/html"]))
                lines.append("")

            # Handle image output
            if "image/png" in data:
                lines.append(f"![Output Image](data:image/png;base64,{data['image/png']})\n")

            # Handle other image formats
            for image_format in ["image/jpeg", "image/svg+xml"]:
                if image_format in data:
                    lines.append(f"![Output Image](data:{image_format};base64,{data[image_format]})\n")

        elif output.output_type == "error":
            lines.append("**Error:**")
            lines.append("```")
            error_text = f"{output.ename}: {output.evalue}"
            if hasattr(output, "traceback"):
                error_text += "\n" + "\n".join(output.traceback)
            lines.append(self._clip_text(error_text))
            lines.append("```\n")

        return lines
