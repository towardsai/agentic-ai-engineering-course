from abc import ABC, abstractmethod
from typing import Any, Iterable, Literal, TypedDict, cast

from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, StructuredTool

from brown.models import FakeModel
from brown.utils import s


class ToolCall(TypedDict):
    name: str
    args: dict[str, Any]
    id: str
    type: Literal["tool_call"]


class Toolkit(ABC):
    """Base class for toolkits following LangChain's toolkit pattern."""

    def __init__(self, tools: list[BaseTool]) -> None:
        self._tools: list[BaseTool] = tools
        self._tools_mapping: dict[str, BaseTool] = {tool.name: tool for tool in self._tools}

    def get_tools(self) -> list[BaseTool]:
        """Get all registered media item generation tools.

        Returns:
            List of all available media item generation tools
        """
        return self._tools.copy()

    def get_tools_mapping(self) -> dict[str, BaseTool]:
        """Get a mapping of tool names to tool instances.

        Returns:
            Dictionary mapping tool names to tool instances
        """
        return self._tools_mapping

    def get_tool_by_name(self, name: str) -> BaseTool | None:
        """Get a specific tool by name.

        Args:
            name: The name of the tool to retrieve

        Returns:
            The tool if found, None otherwise
        """
        return self._tools_mapping.get(name)

    def list_tool_names(self) -> list[str]:
        """Get names of all registered tools.

        Returns:
            List of tool names
        """
        return [tool.name for tool in self._tools]


class Node(ABC):
    def __init__(self, model: Runnable, toolkit: Toolkit) -> None:
        self.toolkit = toolkit
        self.model = self._extend_model(model)

    def _extend_model(self, model: Runnable) -> Runnable:
        if isinstance(model, FakeModel):
            model = cast(FakeModel, model)
            has_mocked_response_set = bool(model.responses)
            if has_mocked_response_set is False:
                model = self._set_default_model_mocked_responses(model)

        return model

    def _set_default_model_mocked_responses(self, model: FakeModel) -> FakeModel:
        model.responses = ["This is a mocked response from the fake model - initialized in the Node base class."]

        return model

    def build_user_input_content(self, inputs: Iterable[str], image_urls: list[str] | None = None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if image_urls:
            for image_url in image_urls:
                messages.append(
                    {
                        "type": "text",
                        "text": "Use the following images as <research> context:",
                    }
                )
                messages.append({"type": "image_url", "image_url": {"url": image_url}})

        # Add the messages last to prioritize focusing on it rather than the images.
        for input_ in inputs:
            messages.append(
                {
                    "type": "text",
                    "text": input_,
                }
            )

        return messages

    @abstractmethod
    async def ainvoke(self) -> Any:
        pass


class ToolNode(Node):
    def __init__(self, model: Runnable) -> None:
        super().__init__(model, toolkit=Toolkit(tools=[]))

    def as_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            coroutine=self.ainvoke,
            name=f"{s.camel_to_snake(self.__class__.__name__)}_tool",
        )

    @abstractmethod
    async def ainvoke(self) -> Any:
        pass
