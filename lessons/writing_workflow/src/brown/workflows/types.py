from typing import Annotated

from annotated_types import Ge, Le
from pydantic import BaseModel


class WorkflowProgress(BaseModel):
    progress: Annotated[int, Ge(0), Le(100)]
    message: str
