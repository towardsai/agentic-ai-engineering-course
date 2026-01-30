"""Database session management for async SQLAlchemy."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from ..config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class DatabaseResources:
    """Database resources bound to a specific event loop."""

    engine: AsyncEngine
    session_factory: async_sessionmaker
    connector: Any  # Connector | None (imported conditionally for Cloud SQL)


# Map event loop ID -> DatabaseResources
# This ensures each event loop gets its own Connector instance, avoiding
# the "Running event loop does not match 'connector._loop'" error on Cloud Run
_loop_resources: dict[int, DatabaseResources] = {}


async def _init_database() -> tuple[AsyncEngine, async_sessionmaker]:
    """
    Initialize database connection lazily within the running event loop.

    This function creates per-event-loop database resources to ensure the Cloud SQL
    connector is always used within the same event loop where it was created.
    This is necessary because the Cloud SQL Python Connector binds to the event loop
    at creation time, and FastMCP may handle different request types (UI routes vs
    MCP protocol handlers) in different event loop contexts.

    Returns:
        Tuple of (engine, session_factory)
    """
    loop = asyncio.get_running_loop()
    loop_id = id(loop)

    # Return cached resources if they exist for this event loop
    if loop_id in _loop_resources:
        resources = _loop_resources[loop_id]
        return resources.engine, resources.session_factory

    # Create new resources for this event loop
    connector = None
    if settings.is_cloud_sql:
        # Cloud SQL mode: use the Cloud SQL Python Connector
        # This provides automatic IAM authentication and secure connections
        from google.cloud.sql.connector import create_async_connector  # noqa: PLC0415

        # IMPORTANT: Use create_async_connector() instead of Connector()!
        # Connector() creates its own event loop in a background thread, which causes
        # "Running event loop does not match 'connector._loop'" errors when connect_async()
        # is called from the actual request's event loop.
        # create_async_connector() properly binds to the current running event loop.
        # refresh_strategy="lazy" is recommended for Cloud Run where CPU may be throttled.
        connector = await create_async_connector(refresh_strategy="lazy")

        async def get_cloud_sql_conn():  # noqa: ANN202
            """Create an asyncpg connection to Cloud SQL."""
            return await connector.connect_async(
                settings.cloud_sql_instance,
                "asyncpg",
                user=settings.db_user,
                password=settings.db_pass.get_secret_value() if settings.db_pass else "",
                db=settings.db_name,
            )

        logger.info("Creating Cloud SQL engine for instance: %s (event loop: %s)", settings.cloud_sql_instance, loop_id)
        engine = create_async_engine(
            "postgresql+asyncpg://",
            async_creator=get_cloud_sql_conn,
            echo=False,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    else:
        # Local development mode: use DATABASE_URL directly
        logger.info("Creating local PostgreSQL engine (event loop: %s)", loop_id)
        engine = create_async_engine(
            settings.database_url,
            echo=False,  # Set to True for SQL query logging
            pool_pre_ping=True,  # Verify connections before use
            pool_size=5,
            max_overflow=10,
        )

    # Create session factory bound to the engine
    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    # Cache resources for this event loop
    _loop_resources[loop_id] = DatabaseResources(
        engine=engine,
        session_factory=session_factory,
        connector=connector,
    )

    return engine, session_factory


async def get_async_engine() -> AsyncEngine:
    """
    Get the async SQLAlchemy engine (initializes on first call per event loop).

    Returns:
        AsyncEngine instance
    """
    engine, _ = await _init_database()
    return engine


async def get_async_session_factory() -> async_sessionmaker:
    """
    Get the async session factory (initializes on first call per event loop).

    Returns:
        async_sessionmaker instance
    """
    _, session_factory = await _init_database()
    return session_factory


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI/tool functions to get a database session.

    Usage:
        async with get_async_session() as session:
            # use session here
            pass

    Or as a context manager in tools:
        session = await anext(get_async_session())
        try:
            # use session
        finally:
            await session.close()
    """
    _, session_factory = await _init_database()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def cleanup_database_resources() -> None:
    """
    Clean up database resources for the current event loop.

    Call this when shutting down to properly close connections and connectors.
    """
    loop = asyncio.get_running_loop()
    loop_id = id(loop)

    if loop_id in _loop_resources:
        resources = _loop_resources.pop(loop_id)
        logger.info("Cleaning up database resources for event loop: %s", loop_id)
        await resources.engine.dispose()
        if resources.connector:
            # Use close_async() for connectors created with create_async_connector()
            await resources.connector.close_async()
