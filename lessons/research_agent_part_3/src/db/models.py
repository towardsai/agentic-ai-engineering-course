"""SQLAlchemy models for Nova MCP Server."""

import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Text, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class ArticleStatus(str, enum.Enum):
    """Status of an article research workflow."""

    CREATED = "created"
    EXTRACTING_URLS = "extracting_urls"
    PROCESSING_SOURCES = "processing_sources"
    RESEARCHING = "researching"
    SELECTING_SOURCES = "selecting_sources"
    SCRAPING_RESEARCH = "scraping_research"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


class Article(Base):
    """
    Main research article entity.

    Stores the article guideline and all intermediate outputs from the research workflow.
    Each article belongs to a user (identified by user_id from Descope).
    """

    __tablename__ = "articles"

    # Primary key - article ID
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique article identifier",
    )

    # User association (from Descope authentication)
    user_id: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
        comment="User ID from Descope authentication",
    )

    # Article guideline content (the uploaded article_guideline.md)
    guideline_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Full content of the article guideline markdown file",
    )

    # Workflow status
    status: Mapped[ArticleStatus] = mapped_column(
        Enum(
            ArticleStatus,
            name="article_status",
            create_constraint=True,
            values_callable=lambda x: [e.value for e in x],
        ),
        default=ArticleStatus.CREATED,
        nullable=False,
        comment="Current status of the research workflow",
    )

    # =========================================================================
    # Intermediate workflow outputs (replacing file-based storage)
    # =========================================================================

    # Replaces: .nova/guidelines_filenames.json
    # Contains: {github_urls, youtube_videos_urls, other_urls, local_file_paths}
    extracted_urls: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
        comment="Extracted URLs from guidelines (JSON): github_urls, youtube_videos_urls, other_urls, local_file_paths",
    )

    # Replaces: .nova/perplexity_results.md
    perplexity_results: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Raw Perplexity research results (markdown format)",
    )

    # Replaces: .nova/perplexity_results_selected.md
    perplexity_results_selected: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Filtered Perplexity results containing only selected high-quality sources (markdown format)",
    )

    # Replaces: .nova/perplexity_sources_selected.md
    perplexity_sources_selected: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Comma-separated IDs of selected Perplexity sources",
    )

    # Replaces: .nova/urls_to_scrape_from_research.md
    urls_to_scrape_from_research: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="URLs selected for full content scraping (one per line)",
    )

    # Replaces: research.md
    research: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Final comprehensive research markdown content",
    )

    # =========================================================================
    # Timestamps
    # =========================================================================

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Timestamp when the article was created",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Timestamp when the article was last updated",
    )

    def __repr__(self) -> str:
        """String representation of the Article."""
        return f"<Article(id={self.id}, user_id={self.user_id}, status={self.status.value})>"


class LocalFile(Base):
    """
    Local file uploads for article guidelines.

    Stores local files that users upload through the UI.
    Each file belongs to an article and a user.
    """

    __tablename__ = "local_files"

    # Primary key - local file ID
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique local file identifier",
    )

    # User association (from Descope authentication)
    user_id: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
        comment="User ID from Descope authentication",
    )

    # Article association (foreign key)
    article_guideline_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to articles table",
    )

    # File metadata
    filename: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Name of the uploaded file",
    )

    # File content
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Content of the uploaded file",
    )

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Timestamp when the file was uploaded",
    )

    def __repr__(self) -> str:
        """String representation of the LocalFile."""
        return f"<LocalFile(id={self.id}, filename={self.filename}, article_id={self.article_guideline_id})>"


class ScrapedUrl(Base):
    """
    Scraped and cleaned URLs from article guidelines.

    Stores web content scraped from URLs referenced in article guidelines.
    Each scraped URL belongs to an article and a user.
    """

    __tablename__ = "scraped_urls"

    # Primary key - scraped URL ID
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique scraped URL identifier",
    )

    # User association (from Descope authentication)
    user_id: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
        comment="User ID from Descope authentication",
    )

    # Article association (foreign key)
    article_guideline_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to articles table",
    )

    # URL
    url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="The URL that was scraped",
    )

    # Scraped content
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Cleaned markdown content from the URL",
    )

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Timestamp when the URL was scraped",
    )

    def __repr__(self) -> str:
        """String representation of the ScrapedUrl."""
        return f"<ScrapedUrl(id={self.id}, url={self.url[:50]}..., article_id={self.article_guideline_id})>"


class GitHubIngest(Base):
    """
    GitHub repository ingests from article guidelines.

    Stores GitIngest analysis results for GitHub URLs referenced in article guidelines.
    Each GitHub ingest belongs to an article and a user.
    """

    __tablename__ = "github_ingests"

    # Primary key - GitHub ingest ID
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique GitHub ingest identifier",
    )

    # User association (from Descope authentication)
    user_id: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
        comment="User ID from Descope authentication",
    )

    # Article association (foreign key)
    article_guideline_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to articles table",
    )

    # GitHub URL
    github_url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="The GitHub URL that was processed",
    )

    # GitIngest result
    gitingest_result: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Markdown output from GitIngest processing",
    )

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Timestamp when the GitHub URL was processed",
    )

    def __repr__(self) -> str:
        """String representation of the GitHubIngest."""
        return f"<GitHubIngest(id={self.id}, url={self.github_url[:50]}..., article_id={self.article_guideline_id})>"


class YouTubeTranscript(Base):
    """
    YouTube video transcripts from article guidelines.

    Stores transcription results for YouTube videos referenced in article guidelines.
    Each YouTube transcript belongs to an article and a user.
    """

    __tablename__ = "youtube_transcripts"

    # Primary key - YouTube transcript ID
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique YouTube transcript identifier",
    )

    # User association (from Descope authentication)
    user_id: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
        comment="User ID from Descope authentication",
    )

    # Article association (foreign key)
    article_guideline_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to articles table",
    )

    # YouTube URL
    youtube_url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="The YouTube URL that was transcribed",
    )

    # Transcription
    transcription: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Markdown transcription content from the video",
    )

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Timestamp when the video was transcribed",
    )

    def __repr__(self) -> str:
        """String representation of the YouTubeTranscript."""
        return f"<YouTubeTranscript(id={self.id}, url={self.youtube_url[:50]}..., article_id={self.article_guideline_id})>"


class ScrapedResearchUrl(Base):
    """
    Scraped research URLs from step 5.2 of the research workflow.

    Stores web content scraped from URLs selected for full scraping in step 5.1.
    Each scraped research URL belongs to an article and a user.
    """

    __tablename__ = "scraped_research_urls"

    # Primary key - scraped research URL ID
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique scraped research URL identifier",
    )

    # User association (from Descope authentication)
    user_id: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
        comment="User ID from Descope authentication",
    )

    # Article association (foreign key)
    article_guideline_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to articles table",
    )

    # URL
    url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="The research URL that was scraped",
    )

    # Scraped content
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Cleaned markdown content from the research URL",
    )

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Timestamp when the URL was scraped",
    )

    def __repr__(self) -> str:
        """String representation of the ScrapedResearchUrl."""
        return f"<ScrapedResearchUrl(id={self.id}, url={self.url[:50]}..., article_id={self.article_guideline_id})>"


class ToolCallUsage(Base):
    """
    Track MCP tool calls per user for rate limiting.

    Records each tool call with user ID, tool name, and month for
    enforcing monthly usage limits.
    """

    __tablename__ = "tool_call_usage"

    # Primary key - tool call usage ID
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique tool call usage identifier",
    )

    # User association (from Descope authentication)
    user_id: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="User ID from Descope authentication",
    )

    # Tool name
    tool_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Name of the MCP tool that was called",
    )

    # Year-month for efficient monthly counting (e.g., "2026-01")
    year_month: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Year and month of the tool call (YYYY-MM format)",
    )

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Timestamp when the tool was called",
    )

    # Composite index for efficient user+month queries
    __table_args__ = (
        Index("ix_tool_call_usage_user_month", "user_id", "year_month"),
        Index("ix_tool_call_usage_user_id", "user_id"),
    )

    def __repr__(self) -> str:
        """String representation of the ToolCallUsage."""
        return f"<ToolCallUsage(id={self.id}, user_id={self.user_id}, tool={self.tool_name}, month={self.year_month})>"
