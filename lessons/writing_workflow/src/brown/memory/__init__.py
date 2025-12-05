from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


@asynccontextmanager
async def build_in_memory_checkpointer() -> AsyncIterator[InMemorySaver]:
    """Build an in-memory checkpointer.

    Returns an async context manager that yields an InMemorySaver.

    Yields:
        InMemorySaver instance
    """

    yield InMemorySaver()


@asynccontextmanager
async def build_sqlite_checkpointer(
    uri: Path = Path("outputs") / "short_term_memory" / "checkpoints.sqlite",
) -> AsyncIterator[AsyncSqliteSaver]:
    """Build an async SQLite checkpointer.

    Returns an async context manager that yields an AsyncSqliteSaver.
    This must be used with `async with` in an async context.

    Args:
        uri: Path to the SQLite database file. Defaults to outputs/short_term_memory/checkpoints.sqlite.
            The parent directory will be created if it doesn't exist.

    Yields:
        AsyncSqliteSaver instance configured with the given database path

    Example:
        from pathlib import Path

        async with build_sqlite_checkpointer(Path("checkpoints.db")) as checkpointer:
            workflow = build_generate_article_workflow(checkpointer=checkpointer)
            await workflow.ainvoke(...)
    """

    uri.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(str(uri)) as checkpointer:
        yield checkpointer
