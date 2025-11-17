from opik.integrations.langchain import OpikTracer

from brown.config import settings
from brown.config_app import get_app_config

app_config = get_app_config()


def build_handler(thread_id: str, tags: list[str] | None = None):
    metadata = {"config_file": settings.CONFIG_FILE.as_posix(), **app_config.model_dump(mode="json")}
    opik_tracer = OpikTracer(
        tags=tags,
        thread_id=thread_id,
        metadata=metadata,
    )
    return opik_tracer
