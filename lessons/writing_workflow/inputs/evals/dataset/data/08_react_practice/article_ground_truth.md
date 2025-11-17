# Lesson 8: ReAct Practice

In this lesson, we will get to practical implementation and build a minimal ReAct agent from scratch using Python and the Gemini API. You will implement the complete Thought → Action → Observation loop to get a concrete mental model of how these systems work. By the end, you will have a working agent you can extend, debug, and customize.

<aside>
💡

You can find the code of this lesson in the notebook of Lesson 8, in the GitHub repository of the course.

</aside>

## Setup and Environment

First, we prepare the Python environment by securely loading the API key from an environment file and importing the libraries used throughout the course. This includes `google-genai` for interacting with the Gemini API, `pydantic` and `enum` for structured data models, and a utility function for pretty-printing outputs.

```python
from enum import Enum
from typing import Callable

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from lessons.utils import env, pretty_print

# Load environment variables after imports
env.load(required_env_vars=["GOOGLE_API_KEY"])
```

With the key loaded, we can initialize a `genai.Client` and use the `gemini-2.5-flash` model to power our ReAct agent.

```python
client = genai.Client()

MODEL_ID = "gemini-2.5-flash"
```

The next step is to give our agent a tool it can use to interact with the world.

## The Tool Layer

To demonstrate and understand how external capabilities integrate with the ReAct framework, we will create a *mock* `search` tool. A function that simulates looking up information. 

If it receives a query it does not recognize, it will return a "not found" message.

Pay close attention to the docstring and the function signature in the implementation. As learned in lesson 6 on Tools, the `google-genai` library can directly take a list of functions as input and create tools. The library will use this docstring to understand what the tool does and how to use it, automatically extracting its purpose and parameters. It's the primary description for the tool.

```python
def search(query: str) -> str:
    """Search for information about a specific topic or query.

    Args:
        query (str): The search query or topic to look up.
    """
    query_lower = query.lower()

    # Predefined responses for demonstration
    if all(word in query_lower for word in ["capital", "france"]):
        return "Paris is the capital of France and is known for the Eiffel Tower."
    elif "react" in query_lower:
        return "The ReAct (Reasoning and Acting) framework enables LLMs to solve complex tasks by interleaving thought generation, action execution, and observation processing."

    # Generic response for unhandled queries
    return f"Information about '{query}' was not found."
```

<aside>
💡

When designing tools, remember to ensure their descriptions are clear, concise, and unambiguous, specifying input/output formats and usage constraints. This clarity helps the LLM choose the right tool at the right time and use it properly. [[1]](https://docs.getdynamiq.ai/low-code-builder/llm-agents/guide-to-implementing-llm-agents-react-and-simple-agents), [[2]](https://arize.com/blog-course/react-agent-llm/), [[3]](https://www.anthropic.com/research/building-effective-agents).

</aside>

To keep track of our agents’ tools, we use a tool registry, which is a simple Python dictionary that maps a tool's name to its function. We’ll use it later when passing the tools definitions in the “Thought” phase of ReAct.

```python
TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    search.__name__: search,
}
```

In a real-world application, you could easily replace this mock function `search` with a call to an actual search API, a database query, or an external data source. The agent's logic interacts with the tool through a consistent input/output structure, meaning only the tool's underlying implementation needs to change, not the agent's core reasoning or prompt logic [[4]](https://blog.motleycrew.ai/blog/reliable-ai-at-your-fingertips-how-we-built-universal-react-agents-that-just-work), [[5]](https://www.promptingguide.ai/techniques/react), [[6]](https://technofile.substack.com/p/how-to-build-a-react-ai-agent-with), [[7]](https://www.youtube.com/watch?v=Lvrv9I276ps), [[8]](https://maven.com/rakeshgohel/ai-agent-engineering-react-rag-multi-agent).

Now that we have defined a tool, we need to enable the agent to "think" about when and how to use it.

## The Thought Phase

Now, we will focus on the “Thought” phase of the ReAct loop. We will construct a prompt template that includes tool descriptions and a placeholder for the conversation history. You will learn how this prompt guides the LLM to produce a concise, purposeful thought, directing the agent’s following action.

In the ReAct framework, the “Thought” phase is where the agent processes what has happened so far and chooses the best immediate course of action [[9]](https://shafiqulai.github.io/blogs/blog_3.html). The first Thought only includes the users’ query and a list of available tools, but by the end of the loop, the thoughts include all the previous decisions and tool results. 

This internal monologue, or verbal reasoning trace, is crucial because it requires the agent to explicitly state its reasoning before taking any action. Seeing the “reasoning,” i.e., the *why* behind the decisions*,* promotes transparency, which is suitable for us to develop the agent. It also makes the agent easier to *debug.* Also, prompting the model to *think* increases the agent’s performance and capacity in multi-step problem-solving [[5]](https://www.promptingguide.ai/techniques/react), [[11]](https://arize.com/docs/phoenix/cookbook/prompt-engineering/react-prompting), [[10]](https://www.wordware.ai/blog/why-the-react-agent-matters-how-ai-can-now-reason-and-act).

To help the LLM reason about and select tools, we format their descriptions into an XML-like structure. This helps the model clearly understand the available tools and their functionalities [[12]](https://ai.gopubby.com/react-ai-agent-from-scratch-using-deepseek-handling-memory-tools-without-frameworks-cabda9094273).

We define the `build_tools_xml_description` function. This function takes our `TOOL_REGISTRY` and generates an XML string describing each available tool, using its name and docstring. This XML is then embedded into our main prompt template, giving the LLM an up-to-date overview of its capabilities.

Note that here we include the tools *directly in the prompt* instead of passing them into the Gemini config. The reason is that in the **Thought** phase the model is not expected to call tools yet; only to reason about which one might be useful. Later, when we move to the **Action** phase, we will register the tools in the Gemini config so the model can actually invoke them.

```python
def build_tools_xml_description(tool_registry: dict[str, Callable[..., str]]) -> str:
    """Build a minimal XML description of tools using only their docstrings."""
    lines = []
    for tool_name, fn in tool_registry.items():
        doc = (fn.__doc__ or "").strip()
        lines.append(f'\t<tool name="{tool_name}">')
        if doc:
            lines.append("\t\t<description>")
            for line in doc.split("\n"):
                lines.append(f"\t\t\t{line}")
            lines.append("\t\t</description>")
        lines.append("\t</tool>")
    return "\n".join(lines)

# Build a string of XML describing the tools
tools_xml = build_tools_xml_description(TOOL_REGISTRY)

PROMPT_TEMPLATE_THOUGHT = """
You are deciding the next best step for reaching the user goal. You have some tools available to you.

Available tools:
<tools>
{tools_xml}
</tools>

Conversation so far:
<conversation>
{conversation}
</conversation>

State your next **thought** about what to do next as one short paragraph focused on the next action you intend to take and why.
Avoid repeating the same strategies that didn't work previously. Prefer different approaches.

Remember:
- Return ONLY plain natural language text.
- Do NOT emit JSON, XML, function calls, or code.
""".strip()
```

The `PROMPT_TEMPLATE_THOUGHT` guides the LLM's reasoning. It states the agent's role, includes the descriptions of `<tools>` inside the XML block, and contains a `conversation` placeholder for the dialogue history. This allows the agent to consider all previous interactions when formulating its next step. The final instruction directs the LLM to produce a short paragraph about its intended action and reasoning.

We also remind the model that we only want plain text that corresponds to its thoughts, not a tool call.

Let’s see the formatted prompt in its entirety.

```python
print(PROMPT_TEMPLATE_THOUGHT.format(tools_xml=tools_xml, conversation=""))
```

It outputs:

```
You are deciding the next best step for reaching the user goal. You have some tools available to you.

Available tools:
<tools>
	<tool name="search">
		<description>
			Search for information about a specific topic or query.
			
			Args:
			    query (str): The search query or topic to look up.
		</description>
	</tool>
</tools>

Conversation so far:
<conversation>

</conversation>

State your next thought about what to do next as one short paragraph focused on the next action you intend to take and why.
Avoid repeating the same strategies that didn't work previously. Prefer different approaches.
```

The output shows the agent's objective, the `search` tool's details, and a placeholder for the dynamic `<conversation>`. This structured prompt design is highly effective for facilitating effective reasoning, as it provides the LLM with all the necessary context to generate coherent thoughts [[5]](https://www.promptingguide.ai/techniques/react).

Finally, we wrap this logic in a function, `generate_thought`. This function takes the current conversation history, formats the prompt, sends it to the Gemini model, and returns the model's response as the agent's thought.

```python
def generate_thought(conversation: str, tool_registry: dict[str, Callable[..., str]]) -> str:
    """Generate a thought as plain text (no structured output)."""
    tools_xml: str = build_tools_xml_description(tool_registry)
    prompt: str = PROMPT_TEMPLATE_THOUGHT.format(tools_xml=tools_xml, conversation=conversation)

    response: types.GenerateContentResponse = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
    )
    return response.text.strip()
```

With a thought generated, the agent must translate this into a concrete action.

## The Action Phase

The “Action” phase is where the agent decides its next move: call a tool or provide a final answer. We use Gemini's native function calling for this.

Our action-generation prompt is simpler than the thought prompt. As thought in Lesson 6, we pass tool functions directly to the Gemini API, which automatically extracts the name, description, and parameters from the function's signature and docstring [[15]](https://developers.googleblog.com/en/building-agents-google-gemini-open-source-frameworks/). This keeps our prompt clean and separates strategic guidance from technical details [[12]](https://ai.gopubby.com/react-ai-agent-from-scratch-using-deepseek-handling-memory-tools-without-frameworks-cabda9094273), [[16]](https://ai.google.dev/gemini-api/docs/langgraph-example).

Here are the prompt templates for the action phase:

```python
PROMPT_TEMPLATE_ACTION = """
You are selecting the best next action to reach the user goal.

Conversation so far:
<conversation>
{conversation}
</conversation>

Respond either with a tool call (with arguments) or a final answer, but only if you can confidently conclude.
""".strip()

# Dedicated prompt used when we must force a final answer
PROMPT_TEMPLATE_ACTION_FORCED = """
You must now provide a final answer to the user.

Conversation so far:
<conversation>
{conversation}
</conversation>

Provide a concise final answer that best addresses the user's goal.
""".strip()

```

We use a separate prompt to produce a final answer. Just as in lesson 6, in some cases, we cap the number of turns to avoid infinite loops and to control things like costs and latency. When that cap is reached, we might want to force the next step to be a final answer. In that case, the model does not need to choose between “call a tool” or “answer now”; it should simply compose the answer.

In real-world applications, you might also choose to return an `*Error`* message to users. Explaining that the agent failed to achieve the goal within the app's limits.

In our example, we keep a dedicated prompt template for this forced-final step for clarity and reliability. You can implement this differently, but a separate template keeps the intent explicit.

Next, we define Pydantic models to structure the two possible outcomes: a `ToolCallRequest` or a `FinalAnswer`. This ensures the output is predictable and easy to parse.

```python
class ToolCallRequest(BaseModel):
    """A request to call a tool with its name and arguments."""
    tool_name: str = Field(description="The name of the tool to call.")
    arguments: dict = Field(description="The arguments to pass to the tool.")

class FinalAnswer(BaseModel):
    """A final answer to present to the user when no further action is needed."""
    text: str = Field(description="The final answer text to present to the user.")
```

The core of this phase is the `generate_action` function. Our function parses the Gemini output into a structured `ToolCallRequest` or `FinalAnswer`. We also include a `force_final` flag to handle cases where the agent must conclude, preventing infinite loops and ensuring a graceful exit [[1]](https://docs.getdynamiq.ai/low-code-builder/llm-agents/guide-to-implementing-llm-agents-react-and-simple-agents).

Here is the implementation of the `generate_action` function:

```python
def generate_action(
    conversation: str, tool_registry: dict[str, Callable[..., str]] | None = None, force_final: bool = False
) -> ToolCallRequest | FinalAnswer:
    """Generate an action by passing tools to the LLM and parsing function calls or final text.

    When force_final is True or no tools are provided, the model is instructed to produce a final answer
    and tool calls are disabled.
    """
    # Use a dedicated prompt when forcing a final answer or no tools are provided
    if force_final or not tool_registry:
        prompt: str = PROMPT_TEMPLATE_ACTION_FORCED.format(conversation=conversation)
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        return FinalAnswer(text=response.text.strip())

    # Default action prompt
    prompt = PROMPT_TEMPLATE_ACTION.format(conversation=conversation)

    # Provide the available tools to the model; disable auto-calling so we can parse and run ourselves
    tools: list[Callable[..., str]] = list(tool_registry.values())
    config = types.GenerateContentConfig(
        tools=tools, automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
    )
    response: types.GenerateContentResponse = client.models.generate_content(
        model=MODEL_ID, contents=prompt, config=config
    )
    
		# From the reponse, we parse each "part" and check if it's a function call
    candidate = response.candidates[0]
    for part in candidate.content.parts:
        if getattr(part, "function_call", None):
            name = part.function_call.name
            args = dict(part.function_call.args or {})
            return ToolCallRequest(tool_name=name, arguments=args)

    # Otherwise, it's a final answer
    final_answer = "".join(part.text for part in candidate.content.parts)
    return FinalAnswer(text=final_answer.strip())
```

We now turn to the Observation step, where the tool’s raw result is captured and will be fed back into the loop to inform the next Thought.

## The Observation Phase

In Lesson 7, we defined Observation as the result of executing an action. In code, that simply means: take the `ToolCallRequest` created by the `generate_action` function, run the tool, and return the output:

```python
def observe(action_request: ToolCallRequest, tool_registry: dict[str, Callable[..., str]]) -> str:
    """
    Execute the selected tool and return the observation text
    (either a result or an error message)
    """
    name = action_request.tool_name
    args = action_request.arguments

    if name not in tool_registry:
        return f"Unknown tool '{name}'. Available: {', '.join(tool_registry)}"

    try:
        return tool_registry[name](**args)
    except Exception as e:
        return f"Error executing tool '{name}': {e}"
```

Let’s see what an observation looks like given the execution of our mock tool:

```python
action_request = generate_action(conversation="What is the capital of France?", tool_registry=TOOL_REGISTRY)
print(observe(req, TOOL_REGISTRY))
```

It outputs:

```markdown
Paris is the capital of France and is known for the Eiffel Tower.
```

As expected, the agent chose the `search` tool and we can see the predefined result for this question.

Now that we have defined all the main components of the ReAct loop, let’s implement the **control loop**, the system that connects them. 

## The Control Loop: Messages, Scratchpad, Orchestration

The control loop is the heart of our agent, managing the iterative flow of the ReAct cycle until the user's question is answered or a limit is reached. To recap how ReAct works, here’s a diagram showing its process, which we’ll implement in this section into a function that controls the main ReAct loop.

![Main ReAct Loop](Lesson%208%20ReAct%20Practice%2024ff9b6f4270805a8696ca3acc7cbde6/image.png)

Main ReAct Loop

An important component of ReAct is the `scratchpad`, which maintains the conversation history and the agent's internal state. This running log of thoughts, actions, and observations provides the agent with a “memory” of its past steps, which is useful for informing its subsequent step [[17]](https://www.dailydoseofds.com/ai-agents-crash-course-part-10-with-implementation/), [[18]](https://geekyants.com/blog/implementing-ai-agents-from-scratch-using-langchain-and-openai). The `scratchpad` mirrors the sequence of steps shown in the following example from the ReAct paper [[24]](https://arxiv.org/abs/2210.03629).

![A ReAct agent in action.](https://huyenchip.com/assets/pics/agents/5-ReAct.png)

A ReAct agent in action. (Media from [[24]](https://arxiv.org/abs/2210.03629))

So, to manage this history, we first define a structured data system. We use an `Enum` for `MessageRole` to categorize interactions and a `Pydantic` model for `Message` to ensure every entry is structured.

```python
class MessageRole(str, Enum):
    """Enumeration for the different roles a message can have."""
    USER = "user"
    THOUGHT = "thought"
    TOOL_REQUEST = "tool request"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final answer"
    
class Message(BaseModel):
    """A message with a role and content, used for all message types."""
    role: MessageRole = Field(
		    description="The role of the message in the ReAct loop."
		)
    content: str = Field(description="The textual content of the message.")

    def __str__(self) -> str:
        """Provides a user-friendly string representation of the message."""
        return f"{self.role.value.capitalize()}: {self.content}"
```

With these two building blocks in place, we can now design the `Scratchpad` class. It serves as the running log of the agent, storing all messages (user queries, thoughts, tool calls, and results) and providing methods to append or review them.

```python
class Scratchpad:
    """Container for ReAct messages with optional pretty-print on append."""

    def __init__(self, max_turns: int) -> None:
        self.messages: list[Message] = []
        self.max_turns: int = max_turns
        self.current_turn: int = 1

    def set_turn(self, turn: int) -> None:
        self.current_turn = turn

    def append(
		    self,
		    message: Message,
		    verbose: bool = False,
		    is_forced_final_answer: bool = False
		) -> None:
        self.messages.append(message)
        if verbose:
            role_to_color = {
                MessageRole.USER: pretty_print.Color.RESET,
                MessageRole.THOUGHT: pretty_print.Color.ORANGE,
                MessageRole.TOOL_REQUEST: pretty_print.Color.GREEN,
                MessageRole.OBSERVATION: pretty_print.Color.YELLOW,
                MessageRole.FINAL_ANSWER: pretty_print.Color.CYAN,
            }
            header_color = role_to_color.get(
		            message.role,
		            pretty_print.Color.YELLOW
		        )
            pretty_print_message(
                message=message,
                turn=self.current_turn,
                max_turns=self.max_turns,
                header_color=header_color,
                is_forced_final_answer=is_forced_final_answer,
            )

    def to_string(self) -> str:
        return "\n".join(str(m) for m in self.messages)
```

Now, we build the main `react_agent_loop` function. This function manages the iterative cycle, ensuring the agent progresses toward its goal while maintaining a clear state [[20]](https://dylancastillo.co/posts/react-agent-langgraph.html). The `react_agent_loop` function begins by initializing the `Scratchpad` and adding the user's initial query.

```python
def react_agent_loop(
    initial_question: str, tool_registry: dict[str, Callable[..., str]], max_turns: int = 5, verbose: bool = False
) -> str | None:
    """
    Implements the main ReAct (Thought -> Action -> Observation) control loop.
    Uses a unified message class for the scratchpad.
    """
    scratchpad = Scratchpad(max_turns=max_turns)

    # Add the user's question to the scratchpad
    user_message = Message(role=MessageRole.USER, content=initial_question)
    scratchpad.append(user_message, verbose=verbose)

    for turn in range(1, max_turns + 1):
        scratchpad.set_turn(turn)
```

Within the main loop, the agent generates a `Thought` based on the current `scratchpad` content. This thought is then appended to the `scratchpad`, making the agent's reasoning explicit [[17]](https://www.dailydoseofds.com/ai-agents-crash-course-part-10-with-implementation/).

```python
# Generate a thought based on the current scratchpad
thought_content = generate_thought(
    scratchpad.to_string(),
    tool_registry,
)
thought_message = Message(role=MessageRole.THOUGHT, content=thought_content)
scratchpad.append(thought_message, verbose=verbose)
```

Following the thought, the agent generates an `Action` by calling `generate_action`, which decides whether to use a tool or provide a final answer.

```python
# Generate an action based on the current scratchpad
action_result = generate_action(
    scratchpad.to_string(),
    tool_registry=tool_registry,
)
```

Depending on the next action, we have two cases:

1. If the `action_result` is a `FinalAnswer`, the loop terminates, and the answer is returned.
    
    ```python
    # If the model produced a final answer, return it
    if isinstance(action_result, FinalAnswer):
        final_answer = action_result.text
        final_message = Message(
    		    role=MessageRole.FINAL_ANSWER,
    		    content=final_answer
    		)
        scratchpad.append(final_message, verbose=verbose)
        return final_answer
    ```
    
2. If the `action_result` is a `ToolCallRequest`, the agent executes the tool. The tool's output becomes the `observation`, which is added back to the `scratchpad`. This feedback is critical because it directly influences the agent's subsequent steps, enabling it to dynamically adapt its strategy [[17]](https://www.dailydoseofds.com/ai-agents-crash-course-part-10-with-implementation/), [[20]](https://dylancastillo.co/posts/react-agent-langgraph.html), [[23]](https://www.nutrient.io/blog/rewoo-vs-react-choosing-right-agent-architecture/).
    
    ```python
    # Otherwise, it is a tool request
    if isinstance(action_result, ToolCallRequest):
        # Log the tool request
        params_str = ", ".join(f"{k}={repr(v)}" for k, v in action_result.arguments.items())
        scratchpad.append(
            Message(role=MessageRole.TOOL_REQUEST, content=f"{action_result.tool_name}({params_str})"),
            verbose=verbose,
        )
    
        # Execute and capture the observation
        observation_content = observe(action_result, tool_registry)
    
        # Log the observation
        scratchpad.append(
            Message(role=MessageRole.OBSERVATION, content=observation_content),
            verbose=verbose,
        )
    ```
    

Finally, the loop includes a termination condition. If the `max_turns` limit is reached, the agent is forced to generate a final answer. This mechanism is important for preventing infinite loops and ensuring a graceful exit [[1]](https://docs.getdynamiq.ai/low-code-builder/llm-agents/guide-to-implementing-llm-agents-react-and-simple-agents), [[21]](https://dev.to/jamesli/react-vs-plan-and-execute-a-practical-comparison-of-llm-agent-patterns-4gh9). For debugging, verbose logging is an effective strategy to trace state transitions and identify issues early [[20]](https://dylancastillo.co/posts/react-agent-langgraph.html), [[22]](https://www.neradot.com/post/building-a-python-react-agent-class-a-step-by-step-guide).

```python
# Check if the maximum number of turns has been reached. If so, force the action selector to produce a final answer
if turn == max_turns:
    forced_action = generate_action(
        scratchpad.to_string(),
        force_final=True,
    )
    if isinstance(forced_action, FinalAnswer):
        final_answer = forced_action.text
    else:
        final_answer = "Unable to produce a final answer within the allotted turns."
    final_message = Message(
		    role=MessageRole.FINAL_ANSWER,
		    content=final_answer
		)
    scratchpad.append(
		    final_message,
			   verbose=verbose,
			   is_forced_final_answer=True
		)
    return final_answer
```

With the full loop implemented, it is time to test our agent and see it in action.

## **T**ests and Traces: Success and Graceful Fallback

We now validate the complete ReAct agent with two distinct tests. The first uses a factual question to demonstrate a successful run, while the second uses an unsupported query to show how the agent handles failure and forced termination. We will analyze the printed traces for both runs to ensure the loop, tool integration, and fallback mechanisms behave as designed.

First, let's ask a question to our mock `search` tool that knows how to answer: `"What is the capital of France?"` We will limit the agent to two turns and run the loop with `verbose=True` to see the full trace of its thought process.

```python
# A straightforward question requiring a search.
question = "What is the capital of France?"
final_answer = react_agent_loop(
		question, TOOL_REGISTRY, max_turns=2, verbose=True
)
```

The output trace clearly shows the ReAct cycle in action:

```
-------------------- User (Turn 1/2): --------------------
What is the capital of France?
------------------------------------------------------------
-------------------- Thought (Turn 1/2): --------------------
I need to find the capital of France to answer the user's question. The `search` tool can be used to retrieve this factual information.
------------------------------------------------------------
-------------------- Tool request (Turn 1/2): --------------------
search(query='capital of France')
------------------------------------------------------------
-------------------- Observation (Turn 1/2): --------------------
Paris is the capital of France and is known for the Eiffel Tower.
------------------------------------------------------------
-------------------- Thought (Turn 2/2): --------------------
I have successfully found the capital of France using the search tool. The next step is to communicate this answer to the user.
------------------------------------------------------------
-------------------- Final answer (Turn 2/2): --------------------
Paris is the capital of France.
------------------------------------------------------------
```

This trace validates that the entire end-to-end loop works as intended. The agent correctly identifies the need for a tool, generates a valid tool call, executes it, processes the observation, and then uses that new information to formulate a final answer.

Now, let's test the agent's ability to handle a query our mock tool does not know: `"What is the capital of Italy?"` This will test the agent's reasoning when a tool fails and demonstrate the forced termination logic.

```python
# A question about a concept the mock search tool doesn't know.
question = "What is the capital of Italy?"
final_answer = react_agent_loop(
		question, TOOL_REGISTRY, max_turns=2, verbose=True
)
```

The trace for this query demonstrates the agent's ability to adapt and handle failure gracefully:

```
-------------------- User (Turn 1/2): --------------------
What is the capital of Italy?
------------------------------------------------------------
-------------------- Thought (Turn 1/2): --------------------
I need to find the capital of Italy to answer the user's question. The `search` tool can provide this information efficiently. I will use the `search` tool to find the capital of Italy.
------------------------------------------------------------
-------------------- Tool request (Turn 1/2): --------------------
search(query='capital of Italy')
------------------------------------------------------------
-------------------- Observation (Turn 1/2): --------------------
Information about 'capital of Italy' was not found.
------------------------------------------------------------
-------------------- Thought (Turn 2/2): --------------------
The previous search query "capital of Italy" did not return information, which is unexpected for a common fact. I will try a broader search query like "Italy" to see if more general information or a list of facts about Italy can provide the capital, as the tool might respond better to less specific phrasing or have had a temporary issue with the exact previous query.
------------------------------------------------------------
-------------------- Tool request (Turn 2/2): --------------------
search(query='Italy')
------------------------------------------------------------
-------------------- Observation (Turn 2/2): --------------------
Information about 'Italy' was not found.
------------------------------------------------------------
-------------------- Final answer (Forced): --------------------
I'm sorry, but I couldn't find information about the capital of Italy.
------------------------------------------------------------
```

This test confirms that our agent does not get stuck in a loop when its tools fail, preventing common failure modes like improper loop termination or error propagation [[1]](https://docs.getdynamiq.ai/low-code-builder/llm-agents/guide-to-implementing-llm-agents-react-and-simple-agents), [[25]](https://huyenchip.com/2025/01/07/agents.html). It recognizes the failure, attempts an alternative strategy, and provides a sensible final response after hitting the `max_turns` limit. This resilience is a key feature of a well-designed agent.

## Alternative: ReAct with Model-Native Reasoning

In Lesson 7, we introduced reasoning models and how they are changing how we develop agentic applications with LLMs. We will use their “thinking” features to showcase their differences and build an alternative ReAct loop that doesn’t include an explicit “Thought Phase”.

Modern Gemini models reason in every response, including during tool calls. You do not need to prompt the model to *think*. If you want to see the short explanation of what the model reasoned about, set `include_thoughts=True`. That flag will enable the return of “*thought summaries”* [[30]](https://ai.google.dev/gemini-api/docs/thinking). As the name suggests, these are summaries generated for transparency, not the model’s full raw internal reasoning trace. They can still be very helpful for debugging and understanding why the model made a particular decision.

A second feature called “*thought signatures*” is available in the Gemini API [[13]](https://ai.google.dev/gemini-api/docs/function-calling?example=meeting#thinking). Thought signatures are opaque metadata attached to the model’s response. They allow the model to carry forward its internal reasoning across turns without exposing the *raw* chain-of-thought to us. To preserve these signatures, pass the exact `Content` returned by the model back into the next request. 

In short, “thought summaries” are optional text you may show in your application; “thought signatures” are how the model actually maintains its private reasoning state across the loop.

Now let’s get back to our new ReAct implementation.

In the following code, we enable thought summaries, set a thinking budget, and define two helpers. The first helper extracts any thought summary text the API returns. The second finds the first function call in a response when the model decides to use a tool.

```python
THINKING_CONFIG = types.ThinkingConfig(
    include_thoughts=True,  # human-readable summaries for transparency/debugging
    thinking_budget=1024,  # tune for latency vs. depth; -1 lets the model decide
)

def extract_thought_summary(response: types.GenerateContentResponse) -> str | None:
    """Collect human-readable thought summaries if present."""
    parts = getattr(response.candidates[0].content, "parts", []) or []
    chunks = [p.text for p in parts if getattr(p, "thought", False) and getattr(p, "text", None)]
    return "\n".join(chunks).strip() if chunks else None

def extract_first_function_call(response: types.GenerateContentResponse):
    """Return (name, args) for the first function call, or None if the model produced a final answer."""
    if getattr(response, "function_calls", None):
        fc = response.function_calls[0]
        return fc.name, dict(fc.args or {})
    parts = getattr(response.candidates[0].content, "parts", []) or []
    for p in parts:
        if getattr(p, "function_call", None):
            return p.function_call.name, dict(p.function_call.args or {})
    return None
```

Here, we build the request configuration. We provide the Python functions as tools and enable built-in thinking. Automatic function calling is disabled, so we can log each step and run tools ourselves with the `observe` function. 

```python
def build_config_with_tools(tools: list[Callable[..., str]]) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        tools=tools,
        thinking_config=THINKING_CONFIG,
        # We disable the automatic execution of tools, we will use the observe function to run them instead.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
```

Now this is the alternative ReAct loop. The conversation is maintained as a list of `types.Content`. 

After each model turn, we append `response.candidates[0].content` back into `contents` to preserve thought signatures. 

When the model calls a tool, we execute it, log the observation, and then append a `function_response` part so the model can use that observation on the next turn. 

For the visible trace, we keep using the `Scratchpad` and our `pretty_print_message` helper function.

```python
def react_agent_loop_thinking(
    initial_question: str,
    tool_registry: dict[str, Callable[..., str]],
    max_turns: int = 5,
    verbose: bool = True,
) -> str:
    """
    ReAct loop that relies on model-native reasoning:
      - optional thought summaries for visibility,
      - thought signatures preserved by appending model Content back into `contents`,
      - pretty-printed trace using Lesson 8's Scratchpad utilities.
    """

    scratchpad = Scratchpad(max_turns=max_turns)
    scratchpad.append(Message(role=MessageRole.USER, content=initial_question), verbose=verbose)

    # Structured "contents" conversation for thought signatures
    contents: list[types.Content] = [types.Content(role="user", parts=[types.Part(text=initial_question)])]
    tools = list(tool_registry.values())
    config = build_config_with_tools(tools)

    for turn in range(1, max_turns + 1):
        scratchpad.set_turn(turn)

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config=config,
        )

        # 1) Thought summary (if any) — log as your THOUGHT message
        thoughts = extract_thought_summary(response)
        if thoughts:
            scratchpad.append(Message(role=MessageRole.THOUGHT, content=thoughts), verbose=verbose)

        # 2) Function/Tool call?
        fc = extract_first_function_call(response)
        if fc:
            name, args = fc

            # We keep the model's full response content to preserve the thought signatures
            contents.append(response.candidates[0].content)

            # Log the tool request
            params_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
            scratchpad.append(
                Message(role=MessageRole.TOOL_REQUEST, content=f"{name}({params_str})"),
                verbose=verbose,
            )

            # Execute the tool
            action_request = ToolCallRequest(tool_name=name, arguments=args)
            observation = observe(action_request, tool_registry)

            # Log observation
            scratchpad.append(Message(role=MessageRole.OBSERVATION, content=observation), verbose=verbose)

            # Send the function response back (standard function-calling protocol)
            fn_resp = types.Part.from_function_response(
                name=name,
                response={"result": observation},
            )
            contents.append(types.Content(role="user", parts=[fn_resp]))
            continue  # next turn

        # 3) No function call => final text
        final_text = (response.text or "").strip()
        scratchpad.append(Message(role=MessageRole.FINAL_ANSWER, content=final_text), verbose=verbose)
        return final_text

    # 4) Forced finish if we hit max turns: disable tool-calling for the last shot
    forced_config = types.GenerateContentConfig(
        thinking_config=THINKING_CONFIG,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.NONE)
        ),
    )
    forced_response = client.models.generate_content(model=MODEL_ID, contents=contents, config=forced_config)
    final_text = (forced_response.text or "Unable to produce a final answer within the allotted turns.").strip()
    scratchpad.append(
        Message(role=MessageRole.FINAL_ANSWER, content=final_text),
        verbose=verbose,
        is_forced_final_answer=True,
    )
    return final_text
```

Now let’s test this new loop using the same questions we used earlier. 

```python
question = "What is the capital of France?"
final_answer = react_agent_loop_thinking(question, TOOL_REGISTRY, max_turns=3, verbose=True)
```

```
------------------------------ User (Turn 1/3): --------------------------------
  What is the capital of France?
---------------------------------------------------------------------------------
------------------------------ Thought (Turn 1/3): ------------------------------
  **Processing a Simple Factual Inquiry**

Okay, I've been given a straightforward factual question: "What is the capital of France?" This is the kind of query I encounter constantly.  The beauty of this is its simplicity – it's ripe for a quick and direct answer. My understanding is that the `default_api.search` tool is the perfect fit here. This is exactly the kind of task it was designed for. All I need to do is formulate the `query` parameter as accurately as possible, and in this case, "capital of France" is the most direct and efficient phrasing to get the information I need. I'm going to set it up to use that right away.
------------------------------ Tool request (Turn 1/3): -------------------------
  search(query='capital of France')
---------------------------------------------------------------------------------
------------------------------ Observation (Turn 1/3): --------------------------
  Paris is the capital of France and is known for the Eiffel Tower.
---------------------------------------------------------------------------------
------------------------------ Final answer (Turn 2/3): -------------------------
  The capital of France is Paris.
```

We get the same answer, but now the Thought comes from the summarized version of the model’s internal thinking. Notice how verbose these thought summaries are by default. We have some control over the maximum length using the `thinking_budget=1024` parameter.

Now let’s ask our agent the second question.

```python
question = "What is the capital of Italy?"
final_answer = react_agent_loop_thinking(question, TOOL_REGISTRY, max_turns=3, verbose=True)
```

```
------------------------------- User (Turn 1/3): --------------------------------
  What is the capital of Italy?
---------------------------------------------------------------------------------
------------------------------- Thought (Turn 1/3): -----------------------------
  **The Italian Capital Inquiry**

Okay, so the user has a straightforward factual question: they want to know the capital of Italy.  That's a quick one! My first thought is to leverage my access to the `search` tool. It's designed for exactly this kind of scenario. To get the answer, I need to formulate the right query.  "Capital of Italy" is concise, unambiguous, and should be easily understood by the search engine.  Therefore, I'll initiate the `search` tool with the query:  `"capital of Italy"`.  Should be a simple retrieval.
---------------------------------------------------------------------------------
------------------------------- Tool request (Turn 1/3): ------------------------
  search(query='capital of Italy')
---------------------------------------------------------------------------------
------------------------------- Observation (Turn 1/3): -------------------------
  Information about 'capital of Italy' was not found.
---------------------------------------------------------------------------------
------------------------------- Thought (Turn 2/3): -----------------------------
  **Navigating a Search Failure**

Okay, so that "capital of Italy" search just bombed. Huh. That's a bit surprising. I'm going to chalk it up to a potentially overly specific query. It's time to adjust my approach. I'm thinking a broader search, something like just "Italy," would be a better starting point. That should hopefully yield general information, and, fingers crossed, the city I'm after. Let's see if this revised tactic proves more fruitful.
---------------------------------------------------------------------------------
------------------------------- Tool request (Turn 2/3): ------------------------
  search(query='Italy')
---------------------------------------------------------------------------------
------------------------------- Observation (Turn 2/3): -------------------------
  Information about 'Italy' was not found.
---------------------------------------------------------------------------------
------------------------------- Thought (Turn 3/3): -----------------------------
  **Addressing the Knowledge Gap**

Okay, so I just tried to retrieve the capital of Italy and then, just Italy itself, thinking maybe I was being too specific, but nothing came back. It's becoming pretty clear this tool I'm using, at least in its current state, doesn't have the capability to answer that specific question or perhaps any related to geographical locations. Based on my expertise, I'm going to have to let the user know directly that I can't find the answer they're looking for right now. No point in spinning my wheels on this. Honesty is the best policy, especially when dealing with someone who likely knows the answer already.
---------------------------------------------------------------------------------
------------------------------- Final answer (Turn 3/3): ------------------------
  I am sorry, but I cannot find the capital of Italy. The search function is not returning any information.
---------------------------------------------------------------------------------
```

We also get the same answer as our classic ReAct agent.

## **Conclusion**

By building this agent piece by piece, we moved beyond abstract diagrams. We implemented the core engine that powers autonomous agents: a control loop that iteratively reasons, acts, and observes. You now have a concrete mental model of how an agent uses its tools to act and how it incorporates feedback to refine its approach. Each new observation informs the agent’s next step. You can prompt the model to produce an explicit Thought, or rely on a reasoning model that performs this reflection by default.

In the upcoming lessons, we will explore more advanced topics. For instance, Lesson 9: RAG Focus will show you how to extend your agent with external knowledge using retrieval-augmented generation. After that, we will cover how to give your agent a *memory* to recall past interactions.

## **References**

- [[1]](https://docs.getdynamiq.ai/low-code-builder/llm-agents/guide-to-implementing-llm-agents-react-and-simple-agents) Guide to Implementing LLM Agents: ReAct and Simple Agents - DynamiQ
- [[2]](https://arize.com/blog-course/react-agent-llm/) How to Build a ReAct Agent from Scratch with LLMs - Arize AI
- [[3]](https://www.anthropic.com/research/building-effective-agents) Building effective agents - Anthropic
- [[4]](https://blog.motleycrew.ai/blog/reliable-ai-at-your-fingertips-how-we-built-universal-react-agents-that-just-work) Reliable AI at Your Fingertips: How We Built Universal ReAct Agents That Just Work - Motley Crew AI
- [[5]](https://www.promptingguide.ai/techniques/react) ReAct - Prompting Guide
- [[6]](https://technofile.substack.com/p/how-to-build-a-react-ai-agent-with) How to build a ReAct AI agent with Python from scratch - Technofile
- [[7]](https://www.youtube.com/watch?v=Lvrv9I276ps) LangChain Crash Course for Beginners - Custom Tools for Agents - Patrick Loeber
- [[8]](https://maven.com/rakeshgohel/ai-agent-engineering-react-rag-multi-agent) AI Agent Engineering: ReAct, RAG, & Multi-Agent - Rakesh Gohel
- [[9]](https://shafiqulai.github.io/blogs/blog_3.html) ReAct: The Future of AI-Powered Problem Solving - Shafiqul Islam
- [[10]](https://www.wordware.ai/blog/why-the-react-agent-matters-how-ai-can-now-reason-and-act) Why the ReAct Agent Matters: How AI Can Now Reason and Act - Wordware
- [[11]](https://arize.com/docs/phoenix/cookbook/prompt-engineering/react-prompting) ReAct Prompting - Arize AI
- [[12]](https://ai.gopubby.com/react-ai-agent-from-scratch-using-deepseek-handling-memory-tools-without-frameworks-cabda9094273) ReAct AI Agent from Scratch using DeepSeek: Handling Memory & Tools without Frameworks - GoPubby
- [[13]](https://ai.google.dev/gemini-api/docs/function-calling?example=meeting#thinking) Function calling with the Gemini API - Google AI for Developers
- [[14]](https://www.leewayhertz.com/react-agents-vs-function-calling-agents/) ReAct Agents vs. Function Calling Agents: A Comparative Analysis - LeewayHertz
- [[15]](https://developers.googleblog.com/en/building-agents-google-gemini-open-source-frameworks/) Building agents with Google Gemini and open source frameworks - Google for Developers Blog
- [[16]](https://ai.google.dev/gemini-api/docs/langgraph-example) ReAct agent from scratch with Gemini 2.5 and LangGraph - Google AI for Developers
- [[17]](https://www.dailydoseofds.com/ai-agents-crash-course-part-10-with-implementation/) AI Agents Crash Course (Part 10): The ReAct Agentic Pattern - Daily Dose of DS
- [[18]](https://geekyants.com/blog/implementing-ai-agents-from-scratch-using-langchain-and-openai) Implementing AI Agents from Scratch Using LangChain and OpenAI - GeekyAnts
- [[19]](https://airbyte.com/data-engineering-resources/using-langchain-react-agents) Using LangChain ReAct Agents for Data Engineering Tasks - Airbyte
- [[20]](https://dylancastillo.co/posts/react-agent-langgraph.html) Building a ReAct Agent in LangGraph - Dylan Castillo
- [[21]](https://dev.to/jamesli/react-vs-plan-and-execute-a-practical-comparison-of-llm-agent-patterns-4gh9) ReAct vs. Plan-and-Execute: A Practical Comparison of LLM Agent Patterns - James Li
- [[22]](https://www.neradot.com/post/building-a-python-react-agent-class-a-step-by-step-guide) Building a Python ReAct Agent Class: A Step-by-Step Guide - Nerando
- [[23]](https://www.nutrient.io/blog/rewoo-vs-react-choosing-right-agent-architecture/) ReWOO vs. ReAct: Choosing the Right Agent Architecture for Your Needs - Nutrient
- [[24]](https://arxiv.org/abs/2210.03629) ReAct: Synergizing Reasoning and Acting in Language Models - Yao et al., 2022
- [[25]](https://huyenchip.com/2025/01/07/agents.html) LLM Agents - Lilian Weng
- [[26]](https://arxiv.org/pdf/2210.03629) ReAct: Synergizing Reasoning and Acting in Language Models
- [[27]](https://www.ibm.com/think/topics/react-agent) ReAct Agent - IBM
- [[28]](https://www.ibm.com/think/topics/ai-agent-planning) AI Agent Planning - IBM
- [[29]](https://www.anthropic.com/engineering/building-effective-agents) Building effective agents - Anthropic
- [[30]](https://ai.google.dev/gemini-api/docs/thinking) Gemini thinking - Google