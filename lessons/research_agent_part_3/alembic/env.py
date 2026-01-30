"""Alembic environment configuration for async SQLAlchemy."""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import your models' Base and all model classes
from src.config.settings import settings
from src.db.base import Base
from src.db.models import Article  # noqa: F401 - needed for Alembic to detect models

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the SQLAlchemy URL from settings
config.set_main_option("sqlalchemy.url", settings.database_url)

# MetaData object for 'autogenerate' support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """Run migrations with the given connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Run migrations in 'online' mode with async engine.

    In this scenario we need to create an Engine and associate a connection
    with the context.

    Supports both local PostgreSQL (via DATABASE_URL) and Cloud SQL (via Connector).
    """
    connector = None

    if settings.is_cloud_sql:
        # Cloud SQL mode: use the Cloud SQL Python Connector
        # This is needed on Cloud Run where there's no direct database URL
        from google.cloud.sql.connector import create_async_connector  # noqa: PLC0415
        from sqlalchemy.ext.asyncio import create_async_engine  # noqa: PLC0415

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

        connectable = create_async_engine(
            "postgresql+asyncpg://",
            async_creator=get_cloud_sql_conn,
            poolclass=pool.NullPool,
        )
    else:
        # Local development: use DATABASE_URL directly
        configuration = config.get_section(config.config_ini_section, {})
        configuration["sqlalchemy.url"] = settings.database_url

        connectable = async_engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()

    # Clean up the Cloud SQL connector if used
    if connector is not None:
        await connector.close_async()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
