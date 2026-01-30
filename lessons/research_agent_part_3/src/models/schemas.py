"""Data models and schemas for the Research MCP Server."""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ResearchSource(BaseModel):
    """Represents a research source (URL, file, etc.)."""

    id: str
    title: str
    url: Optional[str] = None
    file_path: Optional[str] = None
    source_type: str  # 'url', 'file', 'github', 'youtube', etc.
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    processed: bool = False
    created_at: datetime = Field(default_factory=datetime.now)


class ResearchDirectory(BaseModel):
    """Represents a research directory structure."""

    path: str
    name: str
    subdirectories: List[str] = Field(default_factory=list)
    files: Dict[str, str] = Field(default_factory=dict)  # filename -> content summary
    guidelines_file: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


class ProcessingStatus(str, Enum):
    """Status of research processing operations."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingResult(BaseModel):
    """Result of a research processing operation."""

    operation: str
    status: ProcessingStatus
    directory: str
    files_processed: int = 0
    urls_processed: int = 0
    errors: List[str] = Field(default_factory=list)
    results: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class QueryResult(BaseModel):
    """Result from a web search query."""

    query: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    total_sources: int = 0
    selected_sources: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)


class FileProcessingResult(BaseModel):
    """Result of processing a file."""

    file_path: str
    file_type: str
    content_length: int
    extracted_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processed_at: datetime = Field(default_factory=datetime.now)


# Utility classes for research data management
class ResearchDataManager:
    """Simple in-memory data manager for research operations."""

    def __init__(self):
        self.processing_results: Dict[str, ProcessingResult] = {}
        self.research_sources: Dict[str, ResearchSource] = {}

    def store_processing_result(self, directory: str, result: ProcessingResult) -> None:
        """Store a processing result."""
        self.processing_results[directory] = result

    def get_processing_result(self, directory: str) -> Optional[ProcessingResult]:
        """Retrieve a processing result."""
        return self.processing_results.get(directory)

    def add_research_source(self, source: ResearchSource) -> None:
        """Add a research source."""
        self.research_sources[source.id] = source

    def get_research_source(self, source_id: str) -> Optional[ResearchSource]:
        """Get a research source by ID."""
        return self.research_sources.get(source_id)

    def list_research_sources(self) -> List[ResearchSource]:
        """List all research sources."""
        return list(self.research_sources.values())


# Global instance
research_data_manager = ResearchDataManager()
