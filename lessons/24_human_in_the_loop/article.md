# Lesson 24: Human-in-the-Loop for Brown Writing Workflow

In Lessons 20 and 21, we designed the architecture of Brown, our AI writing assistant. Then, in Lesson 22, we implemented the foundation using the orchestrator-worker pattern. In Lesson 23, we added the evaluator-optimizer pattern to enable Brown to self-correct and adhere to style guidelines.

However, even with self-reflection, AI models are not perfect. They can hallucinate, misunderstand nuance, or simply miss the creative spark that you, the human expert, provide. Writing is subjective. An article might be grammatically perfect but fail to capture the specific tone or angle you envisioned.

To bridge this gap, we need to move from a "fire and forget" automation to a collaborative workflow. We need to add the human to the loop.

In this lesson, we will expand Brown to support an interactive editing process. We will decouple our workflows using the Model Context Protocol (MCP) to allow you to generate an article, review it, and then iteratively refine it. You can edit either the whole document or specific sections using your feedback.

Here is what we will cover:

* **The AI Generation / Human Validation Loop:** Why balancing automation with human expertise is critical for practical AI applications.
* **Human Feedback Integration:** How to inject human instructions directly into the article reviewer node.
* **Editing Workflows:** Implementing two new workflows to edit the entire article or just selected text based on feedback.
* **MCP Server Integration:** Serving Brown as an MCP server to expose these workflows as tools.
* **Cursor Integration:** Connecting Brown to Cursor to create a coding-like experience for writing articles.

## Understanding the AI Generation Human Validation Loop

Before we write code, we need to understand the design philosophy behind human-in-the-loop (HITL) systems.

AI systems display what is known as "jagged intelligence." They can perform complex reasoning tasks instantly but might fail at simple logic or make errors no human would make. Because of this fallibility, we cannot fully delegate the "thinking" and "planning" to the AI. Instead, we must design a cooperative loop where the AI handles the heavy lifting of generation, and the human handles the high-value task of verification.

Andrej Karpathy describes this as the **AI generation / human validation loop** [[1]](https://www.youtube.com/watch?v=LCEmiRjPEtQ). The AI generates a draft or a solution, and the human validates it. This loop repeats until the output meets the human's standards. To make this practical, we need to optimize two things: speeding up verification and keeping the AI on a leash.

### Speeding Up Verification

Reading raw text output and typing out corrections is slow. To make the loop spin faster, we need application-specific interfaces (GUIs) that use our visual processing. In coding, this is solved by "diffs"—showing additions in green and deletions in red. This allows us to scan changes instantly rather than reading the entire file again.

For writing, we want a similar experience. Instead of the AI rewriting the whole article and forcing us to search for changes, we want to see exactly what changed and accept or reject it with a click.

### Keeping the AI on a Leash

There is a tendency to want "autonomous agents" that do everything. But as Karpathy notes, receiving a 10,000-line code change from an agent is not helpful because you still have to verify it. If the AI hallucinates or drifts off course in a large batch of work, you waste time debugging massive outputs.

To keep the AI on a leash, we should work in small, incremental chunks. We need to constrain the AI to perform a "single concrete thing" at a time. This increases the probability of success and makes verification easy.

This concept relates directly to the **autonomy slider** we discussed in Lesson 2. As we move from workflows to agents, we increase autonomy. But for high-quality writing, we want to keep the slider closer to the "copilot" side—building an "Iron Man suit" that augments your capabilities rather than an "Iron Man robot" that tries to replace you [[1]](https://www.youtube.com/watch?v=LCEmiRjPEtQ).

## Adding Human-In-The-Loop In Our Writing Workflow

We have the generation step from Lessons 22 and 23, but currently, it is a one-way street. You give a topic, and Brown gives you an article. If you want to change the second paragraph, you have to rewrite it yourself or hack the prompt and rerun the whole pipeline.

To fix this, we need to treat writing an article like writing code. In modern AI-powered IDEs like Cursor, you do not just ask the AI to "write the app." You ask it to generate a file, you review it, you highlight a specific function, and ask it to "refactor this to be more efficient."

We will apply this same logic to Brown. We have implemented two new editing workflows that apply the evaluator-optimizer pattern but prioritize human feedback:

1. **Edit Article Workflow:** Reviews and edits the entire article based on your feedback.
2. **Edit Selected Text Workflow:** Reviews and edits only a specific portion of the text you select.

To make this flexible, we decouple these workflows using the Model Context Protocol (MCP). Instead of one monolithic script, we expose three independent tools: `generate_article`, `edit_article`, and `edit_selected_text`.

Image 1: The Brown writing workflow with human-in-the-loop, showing the iterative process of human input, AI tool usage, and human review for content generation and refinement. (Source https://raw.githubusercontent.com/towardsai/agentic-ai-engineering-course/main/assets/images/l24_writing_workflow.png)

This architecture allows you to generate a draft, review it, and then choose your next move. If the whole tone is off, you use `edit_article`. If just the introduction is weak, you highlight it and use `edit_selected_text`. This keeps the AI on a leash and avoids the risk of it rewriting sections you already like.

## Introducing Human Feedback Into the Article Reviewer

To enable this workflow, we first need to update our core components to understand human feedback. We start with the `ArticleReviewer` node.

1. First, we define a simple entity to hold the feedback. We use a Pydantic model that implements our `ContextMixin` to easily convert the feedback into an XML format the LLM can understand.
    ```python
    class HumanFeedback(BaseModel, ContextMixin):
        content: str

        def to_context(self) -> str:
            return f"""
    <{self.xml_tag}>
        {self.content}
    </{self.xml_tag}>
    """
    ```

2. Next, we update the `ArticleReviewer` node to accept this feedback. We add an optional `human_feedback` parameter to the initialization.
    ```python
    def __init__(
        self,
        to_review: Article | SelectedText,
        article_guideline: ArticleGuideline,
        model: Runnable,
        article_profiles: ArticleProfiles,
        human_feedback: HumanFeedback | None = None,
    ) -> None:
        self.to_review = to_review
        self.article_guideline = article_guideline
        self.article_profiles = article_profiles
        self.human_feedback = human_feedback

        super().__init__(model, toolkit=Toolkit(tools=[]))
    ```

3. We modify the system prompt to explicitly instruct the LLM on how to use this feedback. We tell it to use the feedback to guide the review process and to always generate action points if feedback is present.
    ```python
    system_prompt_template = """
    You are Brown, an expert article writer...

    ## Human Feedback

    Along with the expected requirements, a human already reviewed the article and provided the following feedback:

    {human_feedback}

    If empty, completely ignore it, otherwise the feedback will ALWAYS be used in two ways:
    1. First you will use the <human_feedback> to guide your reviewing process...
    2. Secondly you will extract one or more action points based on the <human_feedback>...
    3. As long the <human_feedback> is not empty, you will always return at least 1 action point...
    """
    ```

4. Finally, when we invoke the reviewer, we inject the feedback into the prompt.
    ```python
    async def ainvoke(self) -> ArticleReviews | SelectedTextReviews:
        system_prompt = self.system_prompt_template.format(
            human_feedback=self.human_feedback.to_context() if self.human_feedback else "",
            # ... other context
        )
    ```

Let us see this in action. We will load the sample article from our test data and provide feedback to "Make the introduction more engaging."

```python
from brown.entities.reviews import HumanFeedback
from brown.nodes.article_reviewer import ArticleReviewer

# ... Loading article, guideline, and profiles (omitted for brevity) ...

human_feedback = HumanFeedback(
    content="""Make the introduction more engaging and catchy. 
Also, expand on the definition of both workflows and agents from the first section"""
)

article_reviewer = ArticleReviewer(
    to_review=article,
    article_guideline=article_guideline,
    model=model,
    article_profiles=profiles,
    human_feedback=human_feedback,
)

reviews = await article_reviewer.ainvoke()
```

When we inspect the output, we see that the reviewer generates specific reviews based on our feedback, tagged with `profile="human_feedback"`.

```text
1. Human Feedback Review
{
  "profile": "human_feedback",
  "location": "Article level",
  "comment": "Make the introduction more engaging and catchy."
}
```

The reviewer also continues to generate reviews based on the other profiles (structure, tone, mechanics), ensuring that while we address the human's specific request, we do not violate our general writing standards.

## Implementing the Article Editing Workflow

Now that our reviewer understands human feedback, we can build the `edit_article` workflow. This workflow takes an existing article and applies a single review-edit loop driven by your instructions.

We reuse the clean architecture patterns from previous lessons, using the app layer to orchestrate our nodes.

1. We use LangGraph's Functional API to define the workflow entry point. We define a typed dictionary `EditArticleInput` to specify the input structure, which includes the directory path and the feedback string.
    ```python
    class EditArticleInput(TypedDict):
        dir_path: Path
        human_feedback: str

    def build_edit_article_workflow(checkpointer: BaseCheckpointSaver):
        return entrypoint(checkpointer=checkpointer)(_edit_article_workflow)
    ```

2. The main workflow function orchestrates the process. It loads the context, generates reviews using our updated `ArticleReviewer`, and then edits the article using the `ArticleWriter`.
    ```python
    async def _edit_article_workflow(inputs: EditArticleInput, config: RunnableConfig) -> str:
        writer = get_stream_writer()
        
        # 1. Load Context
        context = {}
        loaders = build_loaders(app_config)
        for context_name, loader in loaders.items():
            context[context_name] = loader.load(working_uri=inputs["dir_path"])
        human_feedback = HumanFeedback(content=inputs["human_feedback"])

        # 2. Generate Reviews
        reviews = await generate_reviews(
            context["article"], 
            human_feedback, 
            context["article_guideline"], 
            context["profiles"]
        )

        # 3. Edit Article
        article = await edit_based_on_reviews(
            context["article_guideline"], 
            context["research"], 
            context["profiles"], 
            context["examples"], 
            reviews
        )

        return f"""
    Here is the edited article:
    {article.to_context()}
    ... instructions for the client ...
    """
    ```

3. The `generate_reviews` task is a wrapper around the `ArticleReviewer` node. It builds the model and invokes the reviewer.
    ```python
    @task(retry_policy=retry_policy)
    async def generate_reviews(
        article: Article,
        human_feedback: HumanFeedback,
        article_guideline: ArticleGuideline,
        article_profiles: ArticleProfiles,
    ) -> ArticleReviews:
        model, _ = build_model(app_config, node="review_article")
        article_reviewer = ArticleReviewer(
            to_review=article,
            # ... inputs
        )
        return await article_reviewer.ainvoke()
    ```

4. Similarly, the `edit_based_on_reviews` task wraps the `ArticleWriter` node. Notice that we reuse the same `ArticleWriter` class we defined in Lesson 22. By passing it `reviews`, it automatically switches to "editing mode," focusing on addressing the feedback rather than writing from scratch.
    ```python
    @task(retry_policy=retry_policy)
    async def edit_based_on_reviews(
        # ... inputs
        reviews: ArticleReviews,
    ) -> Article:
        model, _ = build_model(app_config, node="edit_article")
        article_writer = ArticleWriter(
            # ... inputs
            reviews=reviews,
            model=model,
        )
        return await article_writer.ainvoke()
    ```

This workflow enables a powerful feedback loop. You do not need to guess the perfect prompt upfront. You can generate a draft with minimal iterations (saving cost and time), and then use this workflow to refine it dynamically based on what you actually see in the output.

## Implementing the Selected Text Editing Workflow

While editing the whole article is useful for global changes (for example, "Make the tone more professional"), it can be overkill for small fixes. If you only want to rewrite the introduction, running the `edit_article` workflow might accidentally change a perfect conclusion.

To solve this, we implemented the `edit_selected_text` workflow. It follows the exact same pattern as `edit_article` but operates on a `SelectedText` entity.

1. We define the input structure to include the selected text and its location (line numbers). This context helps the LLM understand where the text fits in the overall article.
    ```python
    class EditSelectedTextInput(TypedDict):
        dir_path: Path
        human_feedback: str
        selected_text: str
        number_line_before_selected_text: int
        number_line_after_selected_text: int
    ```

2. The workflow logic is nearly identical, but it constructs a `SelectedText` object instead of using the full `Article`.
    ```python
    async def _edit_selected_text_workflow(inputs: EditSelectedTextInput, config: RunnableConfig) -> str:
        # ... loading context ...

        selected_text = SelectedText(
            article=context["article"],
            content=inputs["selected_text"],
            first_line_number=inputs["number_line_before_selected_text"],
            last_line_number=inputs["number_line_after_selected_text"],
        )
        
        # ... generate reviews and edit ...
    ```

3. The `generate_reviews` task for selected text uses the `review_selected_text` node configuration. The `ArticleReviewer` is smart enough to handle `SelectedText` objects, focusing its critique only on that section while using the rest of the article as background context.
    ```python
    @task(retry_policy=retry_policy)
    async def generate_reviews(
        selected_text: SelectedText,
        # ...
    ) -> SelectedTextReviews:
        model, _ = build_model(app_config, node="review_selected_text")
        # ... invoke reviewer
        return cast(SelectedTextReviews, reviews)
    ```

4. The `edit_based_on_reviews` task takes `SelectedTextReviews` and returns a `SelectedText` object.
    ```python
    @task(retry_policy=retry_policy)
    async def edit_based_on_reviews(
        # ...
        reviews: SelectedTextReviews,
    ) -> SelectedText:
        model, _ = build_model(app_config, node="edit_selected_text")
        # ... invoke writer
        return cast(SelectedText, await article_writer.ainvoke())
    ```

This granularity is what allows us to keep the AI "on a leash." By restricting the edit scope to a specific paragraph, we ensure the rest of the article remains untouched, making the verification step much faster.

## Serving Brown as an MCP Server

Now that we have our workflows, we need to expose them to the outside world. We use the Model Context Protocol (MCP) to serve Brown as a set of tools that any MCP-compliant client (like Cursor or Claude Desktop) can use.

We implement the server using `FastMCP` in the `brown.mcp` module. This layer acts as a clean interface, initializing the app components and injecting infrastructure dependencies.

[Insert screenshot of how Brown looks with Cursor when working and hooked as an MCP server here]

1. We initialize the server and define our three tools. Notice how we use detailed docstrings and type hints. These are what the MCP client uses to understand how to call our tools. Inside each tool, we follow a consistent pattern. We build the appropriate workflow with an in-memory checkpointer. Then, we create a unique thread ID to isolate the execution. Finally, we run the workflow and stream progress updates back to the client context.

    1. First, we initialize the FastMCP server and define the `generate_article` tool.
       ```python
       mcp = FastMCP("Brown Writing Assistant")

       @mcp.tool()
       async def generate_article(dir_path: str, ctx: Context = None) -> str:
           """Generates a new article from scratch based on the guideline."""
           async with build_in_memory_checkpointer() as checkpointer:
               workflow = build_generate_article_workflow(checkpointer=checkpointer)
               thread_id = str(uuid.uuid4())
               config = {"configurable": {"thread_id": thread_id}}

               async for message in workflow.astream(
                   {"dir_path": Path(dir_path)}, config=config, stream_mode="custom"
               ):
                   parse_message(message, ctx)
               
               return "Article generated successfully."
       ```

    2. Next, we define the `edit_article` tool, which applies human feedback to the entire article.
       ```python
       @mcp.tool()
       async def edit_article(article_path: str, human_feedback: str, ctx: Context = None) -> str:
           """Edits an existing article based on human feedback."""
           async with build_in_memory_checkpointer() as checkpointer:
               workflow = build_edit_article_workflow(checkpointer=checkpointer)
               thread_id = str(uuid.uuid4())
               config = {"configurable": {"thread_id": thread_id}}

               async for message in workflow.astream(
                   {
                       "dir_path": Path(article_path).parent, 
                       "human_feedback": human_feedback
                   },
                   config=config,
                   stream_mode="custom",
               ):
                   parse_message(message, ctx)

               return "Article edited successfully."
       ```

    3. Finally, we define the `edit_selected_text` tool for more granular control.
       ```python
       @mcp.tool()
       async def edit_selected_text(
           article_path: str, 
           human_feedback: str, 
           selected_text: str, 
           first_line_number: int,
           last_line_number: int,
           ctx: Context = None
       ) -> str:
           """Edits a specific section of the article based on human feedback."""
           async with build_in_memory_checkpointer() as checkpointer:
               workflow = build_edit_selected_text_workflow(checkpointer=checkpointer)
               thread_id = str(uuid.uuid4())
               config = {"configurable": {"thread_id": thread_id}}

               async for message in workflow.astream(
                   {
                       "dir_path": Path(article_path).parent,
                       "human_feedback": human_feedback,
                       "selected_text": selected_text,
                       "number_line_before_selected_text": first_line_number,
                       "number_line_after_selected_text": last_line_number,
                   },
                   config=config,
                   stream_mode="custom",
               ):
                   parse_message(message, ctx)

               return "Selected text edited successfully."
       ```

2. To improve the user experience, we parse the workflow's streaming messages and report progress back to the client. This gives you real-time feedback in Cursor (for example, "Reviewing article...", "Editing selected text...").
    ```python
    def parse_message(message: str | dict, ctx: Context) -> None:
        if isinstance(message, dict) and "progress" in message:
            ctx.report_progress(message["progress"], total=100)
            ctx.info(f"[{message['progress']}%] {message['message']}")
    ```

3. We also define Prompts. These are pre-configured templates that appear in the MCP client, making it easy to trigger tools without typing out complex arguments.
    ```python
    @mcp.prompt()
    def edit_selected_text_prompt(human_feedback: str = "") -> str:
        return f"""
        I want to edit the selected text in the article.
        Feedback: {human_feedback}
        Please use the 'edit_selected_text' tool.
        """
    ```

4. Finally, we expose Resources to allow clients to read the app configuration or character profiles directly.
    ```python
    @mcp.resource("resource://config/app")
    def get_app_config() -> str:
        return app_config.model_dump_json(indent=2)
    ```

This setup completely separates our serving logic from our business logic. We could easily swap MCP for a FastAPI REST API without changing a single line of our workflow code.

## Hooking to the MCP Server

There are two main ways to connect to Brown's MCP server: using a CLI script for testing or integrating directly with Cursor for the full production experience.

### Option 1: Brown CLI Script

Brown includes a CLI utility located at `lessons/writing_workflow/scripts/brown_mcp_cli.py`. It uses an in-memory MCP client to call the tools, which is great for quick testing.

You can run commands like:

```bash
# Edit a selected section
python scripts/brown_mcp_cli.py edit-selected-text \
    --dir-path /path/to/article \
    --human-feedback "Make this shorter. Remove all em-dashes." \
    --first-line 10 \
    --last-line 20
```

### Option 2: Cursor Integration

The real power comes from integrating with Cursor. We configure the server in the `.cursor/mcp.json` file. This tells Cursor how to launch our server using `uv`.

```json
{
  "mcpServers": {
    "brown": {
      "command": "uv",
      "args": [
        "run",
        "brown/mcp/server.py"
      ],
      "cwd": "/absolute/path/to/course-ai-agents/lessons/writing_workflow",
      "env": {
        "GOOGLE_API_KEY": "..."
      }
    }
  }
}
```

Once configured, Brown appears as a set of tools in Cursor's "Composer" or chat interface.

### The Brown + Human-in-the-Loop Writing Experience

With this integration, the writing process feels like pair programming:

1. **Generate:** You ask Brown to "Generate an article about X." Brown runs the tool and produces a markdown file.
2. **Review:** You read the article in the editor.
3. **Feedback:** You highlight a paragraph that feels clunky.
4. **Edit:** You type in the chat: "Rewrite this to be more punchy." Cursor calls the `edit_selected_text` tool.
5. **Diff:** Brown returns the edited text. Cursor shows you a "diff" view (red/green). You accept or reject the change.

This is the AI generation/human validation loop in action. You guide the direction; the AI handles the execution; you verify the result.

Note on HTTP transports: In this lesson, we used the stdio transport for local development. We will show you how to use remote HTTP-based transports in Part 3 when we cover deployment.

For more details on running Brown as a standalone project, see the documentation at `lessons/writing_workflow/README.md`.

## Running Brown Through the MCP Server (Video)

To demonstrate exactly how this feels in practice, we have recorded a walkthrough of using Brown within Cursor.

In this video, we will show you:
* How to configure and connect the MCP Server in Cursor.
* Generating an initial article draft.
* The human validation loop: reviewing the text and providing feedback.
* Using `edit_selected_text` to surgically refine specific sections.
* The "diff" experience in Cursor where you accept/reject AI changes.

[Link to video demonstration will be provided]

We actually used this exact workflow to generate parts of this lesson—a real-world proof of work for the system we are building.

## Conclusion

Understanding how to properly add humans in the loop is critical for building practical AI applications. AI systems are imperfect—they hallucinate and make reasoning mistakes. The balance we have built allows AI to handle the heavy lifting of drafting and editing while keeping you, the human expert, in control of the final quality and direction.

In this lesson, we learned:

1. **The AI Generation / Human Validation Loop:** We explored why we need to speed up verification (using GUIs/diffs) and keep the AI on a leash (using incremental edits) to make this loop efficient.
2. **Human Feedback Integration:** We updated our `ArticleReviewer` node to prioritize human feedback, creating actionable reviews that guide the writer.
3. **Editing Workflows:** We implemented `edit_article` and `edit_selected_text` workflows, allowing for both global and local refinement.
4. **MCP Server Integration:** We served Brown using FastMCP, exposing our workflows as tools and prompts while maintaining a clean separation of concerns.
5. **Cursor Integration:** We connected everything to Cursor, enabling a coding-like experience where we can interactively write and refine articles.

Here are some ideas on how you can further extend this code:
* **Hook Brown to Claude Desktop:** Instead of using Cursor, integrate Brown with Claude Desktop for a different AI assistant experience.
* **Use Resource Templates:** Parameterize the writing profiles and easily add support for all available profiles.
* **Serve Brown through FastAPI:** Replace the MCP server with a FastAPI REST API for web-based integrations.
* **Add Guideline Review Tool:** Create a workflow to review and edit your article guidelines themselves.

This wraps up our deep dive into the Brown writing workflow. We started with the basic architecture in Lesson 22, added self-correction in Lesson 23, and now added the important human element in Lesson 24.

In Part 2D, we will take the final step in our capstone journey: orchestrating both **Nova** (our deep researcher) and **Brown** (our writer) as a multi-agent system within a single MCP client, automating the entire research-to-article pipeline. Then, in Part 3, we will shift gears to production engineering, covering AI Evals, monitoring, and deployment.

## References

1. Andrej Karpathy. (2025, January 16). Software Is Changing (Again). YouTube. https://www.youtube.com/watch?v=LCEmiRjPEtQ

2. Anthropic. (2024, December 19). Building effective agents. Anthropic. https://www.anthropic.com/engineering/building-effective-agents

3. The FastMCP Server. (n.d.). FastMCP. https://gofastmcp.com/servers/server

4. LangGraph Functional API. (n.d.). LangChain. https://docs.langchain.com/oss/python/langgraph/functional-api

5. Use the functional API. (n.d.). LangChain. https://docs.langchain.com/oss/python/langgraph/use-functional-api