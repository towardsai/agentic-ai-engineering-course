from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from loguru import logger

from brown.config_app import get_app_config
from brown.entities.guidelines import ArticleGuideline
from brown.entities.research import Research
from brown.models import FakeModel

from .base import Node, ToolCall, Toolkit

app_config = get_app_config()


class MediaGeneratorOrchestrator(Node):
    system_prompt_template = """You are an Media Generation Orchestrator responsible for analyzing article 
guidelines and research to identify what media items need to be generated for the article.

Your task is to:
1. Carefully analyze the article guideline and research content provided
2. Identify ALL explicit requests for media items (diagrams, charts, visual illustrations, etc.)
3. For each identified media requirement, call the appropriate tool to generate the media item
4. Provide clear, detailed descriptions for each media item based on the guideline requirements and research context

## Analysis Guidelines

**Look for these explicit indicators in the article guideline:**
- Direct mentions: "Render a Mermaid diagram", "Draw a diagram", "Create a visual", "Illustrate with", etc.
- Visual requirements: "diagram visually explaining", "chart showing", "figure depicting", "visual representation"
- Process flows: descriptions of workflows, architectures, data flows, or system interactions
- Structural elements: hierarchies, relationships, comparisons, or step-by-step processes

**Key places to look:**
- Section requirements and descriptions  
- Specific instructions mentioning visual elements
- Complex concepts that would benefit from visual explanation
- Architecture or system descriptions
- Process flows or workflows

## Tool Usage Instructions

You will call multiple tools to generate the media items. You will use the tool that is most appropriate for the media item you are 
generating.

For each identified media requirement:

**When to use MermaidDiagramGenerator:**
- Explicit requests for Mermaid diagrams
- System architectures and workflows
- Process flows and data pipelines  
- Organizational structures or hierarchies
- Flowcharts for decision-making processes
- Sequence diagrams for interactions
- Entity-relationship diagrams
- Class diagrams for software structures
- State diagrams for system states
- Mind maps for concept relationships

**Description Requirements:**
When calling tools, provide detailed descriptions that include:
- The specific purpose and context from the article guideline
- Key components that should be included based on the research
- The type of diagram most appropriate (flowchart, sequence, architecture, etc.)
- Specific elements, relationships, or flows to highlight
- Any terminology or technical details from the research

## Example Analysis Process

1. **Scan the guideline** for phrases like:
   - "Render a Mermaid diagram of..."
   - "Draw a diagram showing..."
   - "Illustrate the architecture..."
   - "Visual representation of..."

2. **For each found requirement:**
   - Extract the specific context and purpose
   - Identify what should be visualized
   - Determine the most appropriate diagram type
   - Craft a detailed description incorporating research insights

3. **Call the appropriate tool** with the comprehensive description

## Input Context

Here is the article guideline:
{article_guideline}

Here is the research:
{research}

## Your Response

Analyze the provided article guideline and research, then call the appropriate tools for each 
identified media item requirement. Each tool call should include a detailed description that ensures 
the generated media item will be relevant, accurate, and valuable for the article's educational goals.

If no explicit media requirements are found in the guideline, respond with: 
"No explicit media item requirements found in the article guideline."
"""

    def __init__(
        self,
        article_guideline: ArticleGuideline,
        research: Research,
        model: Runnable,
        toolkit: Toolkit,
    ) -> None:
        self.article_guideline = article_guideline
        self.research = research

        super().__init__(model, toolkit)

    def _extend_model(self, model: Runnable) -> Runnable:
        model = cast(BaseChatModel, super()._extend_model(model))
        model = model.bind_tools(self.toolkit.get_tools(), tool_choice="any")

        return model

    async def ainvoke(self) -> list[ToolCall]:
        system_prompt = self.system_prompt_template.format(
            article_guideline=self.article_guideline.to_context(),
            research=self.research.to_context(),
        )
        user_input_content = self.build_user_input_content(inputs=[system_prompt], image_urls=self.research.image_urls)
        inputs = [
            {
                "role": "user",
                "content": user_input_content,
            }
        ]
        response = await self.model.ainvoke(inputs)

        if isinstance(response, AIMessage) and response.tool_calls:
            jobs = cast(list[ToolCall], response.tool_calls)
        else:
            logger.warning(f"No tool calls found in the response. Instead found response of type `{type(response)}`.")
            jobs = []

        return jobs

    def _set_default_model_mocked_responses(self, model: FakeModel) -> FakeModel:
        model.responses = []

        return model
