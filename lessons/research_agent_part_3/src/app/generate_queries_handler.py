"""LLM operations and query generation utilities."""

import logging
from typing import List, Tuple

from ..config.prompts import PROMPT_GENERATE_QUERIES_AND_REASONS
from ..config.settings import settings
from ..models.query_models import GeneratedQueries
from ..utils.llm_utils import get_chat_model

logger = logging.getLogger(__name__)


async def generate_queries_with_reasons(
    article_guidelines: str,
    past_research: str,
    scraped_ctx: str,
    n_queries: int = 5,
) -> List[Tuple[str, str]]:
    """Return a list of tuples (query, reason)."""

    prompt = PROMPT_GENERATE_QUERIES_AND_REASONS.format(
        n_queries=n_queries,
        article_guidelines=article_guidelines or "<none>",
        past_research=past_research or "<none>",
        scraped_ctx=scraped_ctx or "<none>",
    )

    chat_llm = get_chat_model(settings.query_generation_model, GeneratedQueries)
    logger.debug("Generating candidate queries")

    try:
        response = await chat_llm.ainvoke(prompt)

        if not isinstance(response, GeneratedQueries):
            msg = f"LLM returned unexpected type: {type(response)}"
            logger.error(msg)
            raise RuntimeError(msg)

        queries_and_reasons = [(item.question, item.reason) for item in response.queries]

        if len(queries_and_reasons) < n_queries:
            msg = f"LLM returned only {len(queries_and_reasons)} queries, expected {n_queries}."
            logger.error(msg)
            raise RuntimeError(msg)

        return queries_and_reasons[:n_queries]

    except Exception as exc:
        logger.error(f"⚠️ LLM call failed ({exc}).", exc_info=True)
        raise
