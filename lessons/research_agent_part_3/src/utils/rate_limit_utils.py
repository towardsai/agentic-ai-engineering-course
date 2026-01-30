"""Rate limiting utilities for MCP tools."""

import logging
from datetime import UTC, datetime
from functools import wraps
from typing import Callable, TypeVar

from fastmcp.server.dependencies import get_access_token
from sqlalchemy import func, select

from ..config.settings import settings
from ..db.models import ToolCallUsage
from ..db.session import get_async_session_factory

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


class RateLimitExceededError(Exception):
    """Raised when a user exceeds their monthly tool call limit."""

    def __init__(self, user_id: str, current_usage: int, limit: int):
        self.user_id = user_id
        self.current_usage = current_usage
        self.limit = limit
        # Calculate next month reset date
        now = datetime.now(UTC)
        if now.month == 12:
            self.resets_at = datetime(now.year + 1, 1, 1, tzinfo=UTC)
        else:
            self.resets_at = datetime(now.year, now.month + 1, 1, tzinfo=UTC)

        super().__init__(
            f"Monthly tool usage limit exceeded. You have used {current_usage}/{limit} "
            f"tool calls this month. Your quota resets on {self.resets_at.strftime('%B %d, %Y')}."
        )


async def check_and_record_tool_call(user_id: str, tool_name: str) -> int:
    """
    Check if user has remaining quota and record the tool call.

    Args:
        user_id: The authenticated user's ID
        tool_name: Name of the tool being called

    Returns:
        Current usage count after recording this call

    Raises:
        RateLimitExceededError: If user has exceeded their monthly limit
    """
    # Skip rate limiting if limit is 0 (unlimited)
    if settings.monthly_tool_call_limit == 0:
        return 0

    year_month = datetime.now(UTC).strftime("%Y-%m")

    session_factory = await get_async_session_factory()
    async with session_factory() as session:
        # Count existing calls this month
        stmt = select(func.count()).select_from(ToolCallUsage).where(
            ToolCallUsage.user_id == user_id,
            ToolCallUsage.year_month == year_month,
        )
        result = await session.execute(stmt)
        current_count = result.scalar() or 0

        # Check if limit exceeded
        if current_count >= settings.monthly_tool_call_limit:
            logger.warning(
                f"Rate limit exceeded for user {user_id}: {current_count}/{settings.monthly_tool_call_limit}"
            )
            raise RateLimitExceededError(user_id, current_count, settings.monthly_tool_call_limit)

        # Record this tool call
        usage = ToolCallUsage(
            user_id=user_id,
            tool_name=tool_name,
            year_month=year_month,
        )
        session.add(usage)
        await session.commit()

        new_count = current_count + 1
        logger.debug(
            f"Recorded tool call for user {user_id}: {tool_name} ({new_count}/{settings.monthly_tool_call_limit})"
        )
        return new_count


def rate_limited(func: F) -> F:
    """
    Decorator to enforce rate limiting on MCP tools.

    Extracts user_id from the access token, checks the monthly usage limit,
    and records the tool call before executing the wrapped function.

    Usage:
        @mcp.tool()
        @opik.track(type="tool")
        @rate_limited
        async def my_tool(arg: str) -> Dict[str, Any]:
            ...

    Raises:
        RateLimitExceededError: If user has exceeded their monthly limit
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Get user ID from access token
        token = get_access_token()
        user_id = token.claims["sub"]

        # Check rate limit and record the call
        await check_and_record_tool_call(user_id, func.__name__)

        # Execute the actual tool
        return await func(*args, **kwargs)

    return wrapper  # type: ignore
