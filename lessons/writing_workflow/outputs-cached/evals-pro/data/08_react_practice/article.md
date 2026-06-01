# Writing ReAct From Scratch
### Building the engine of autonomy with Python and Gemini

In the previous lesson, we explored the theory behind agentic reasoning patterns like ReAct. Now, it is time to put that theory into practice. While high-level frameworks like LangChain or CrewAI offer powerful abstractions for building agents, they can also obscure the fundamental mechanics of how an agent actually "thinks" and "acts."

To truly master AI engineering, you need to look under the hood. You need to understand the raw control loop that transforms a static LLM into an autonomous agent.

This is **Lesson 8 of Part 1** in our AI Agents course. We are building directly on the theoretical foundations laid in Lesson 7, where we dissected the ReAct (Reason+Act) and Plan-and-Execute patterns. If you recall, we defined ReAct as a synergy between reasoning traces and actions. Today, we stop talking about it and start building it.

In this lesson, we will build a minimal, but complete, ReAct agent from scratch using only Python and the Gemini API. By the end, you will have implemented the entire Thought → Action → Observation loop yourself. You will see exactly how an LLM is prompted to reason, how its intent to use a tool is captured via function calling, and how the results of that tool are fed back into its context to inform the next step.

This is a 100% hands-on guide. We will write the code to setup the environment, define tools, generate thoughts, handle actions, and orchestrate the control loop.

## Setup and Environment

First, we need to set up our Python environment to ensure the code runs smoothly. This involves loading necessary credentials, importing libraries, and initializing the Gemini client.

1. We begin by loading our API key from an environment file. This is a standard practice to keep secrets out of your source code. We use a utility function `load` to ensure the `GOOGLE_API_KEY` is present in the environment before proceeding. This prevents runtime errors later when we attempt to connect to the API.
    ```python
    from lessons.utils import env

    env.load(required_env_vars=["GOOGLE_API_KEY"])
    ```

2. Next, we import the required libraries. We rely on `google.genai` for interacting with the model. We also import `pydantic` and `enum`. These are necessary for defining the structured data our agent will use. `pydantic` allows us to create strict data models for our agent's messages and tool calls, ensuring that the data flowing through our control loop is valid and typed. `enum` helps us define fixed roles for our messages (like "User" or "Thought"), preventing typo-related bugs in our state management.
    ```python
    from enum import Enum
    from pydantic import BaseModel, Field
    from typing import List

    from google import genai
    from google.genai import types

    from lessons.utils import pretty_print
    ```

3. With the API key loaded, we initialize the Gemini client. We will also select the `gemini-2.5-flash` model. We choose the "Flash" variant specifically for this lesson because agentic loops involve multiple round-trips to the LLM (one for thinking, one for acting, potentially many turns). A faster, lower-latency model like Flash ensures the agent feels responsive and keeps costs low during development and testing loops.
    ```python
    client = genai.Client()

    MODEL_ID = "gemini-2.5-flash"
    ```
    It outputs:
    ```text
    Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.
    ```

With our client and model ready, the next step is to give our agent a capability—a tool it can use to interact with the world.

## Tool Layer: Mock Search Implementation

To demonstrate the "Action" part of the ReAct cycle, our agent needs a tool. For this lesson, we will create a simple mock `search` tool.

### Why Use a Mock Tool?

You might wonder why we are not connecting to a real Google Search API immediately. We deliberately use a mock tool for three key engineering reasons:

1.  **Isolation of Concerns:** We want to focus entirely on the *agent's reasoning loop*—how it decides to call a tool and how it handles the result. Debugging network issues or API quotas distracts from learning the core ReAct mechanics.
2.  **Deterministic Testing:** A mock tool returns the exact same output every time. This allows us to verify the agent's behavior consistently. If the agent fails, we know it is a logic issue, not the search engine changing its ranking algorithms.
3.  **Zero Dependencies:** It eliminates the need for you to sign up for external services or manage additional API keys just to follow this lesson.

### Implementing the Mock Search

1. Our mock tool acts as the external knowledge source. It recognizes specific queries about "France" or "ReAct" and returns hardcoded answers. If it receives a query it does not recognize, it returns a generic "not found" message.
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
            return "The ReAct framework enables LLMs to solve complex tasks by interleaving thought generation, action execution, and observation processing."

        # Generic response for unhandled queries
        return f"Information about '{query}' was not found."
    ```

2. We maintain a mapping from the tool name to the tool function, known as the `TOOL_REGISTRY`. This allows the model to plan using symbolic tool names, while our code safely resolves those names to actual Python functions for execution.
    ```python
    TOOL_REGISTRY = {
        search.__name__: search,
    }
    ```

### Real-World Context

While this function is simple, it shares the exact same interface as a production-grade tool: it takes a string input and returns a string output. In a real-world application, you would simply replace the body of this function with a call to `google_search_api.run(query)` or a vector database retrieval. The agent's control loop, which we will build next, would remain 100% identical. This modularity is a key advantage of the ReAct pattern—the reasoning engine is decoupled from the tool implementation.

## Thought Phase: Prompt Construction and Generation

The first step in the ReAct loop is "Thought." Here, the agent analyzes the user's request and its available tools, then formulates a plan. This plan is a short, internal monologue that guides its next action. We generate this thought by prompting the LLM with the conversation history and a description of the available tools [[1]](https://arxiv.org/pdf/2210.03629).

1. To make the tools understandable to the LLM during the thought phase, we format their descriptions into an XML-like structure. This helps the model clearly distinguish the tools and their functionalities based on their docstrings.
    ```python
    def build_tools_xml_description(tools: dict[str, callable]) -> str:
        """Build a minimal XML description of tools using only their docstrings."""
        lines = []
        for tool_name, fn in tools.items():
            doc = (fn.__doc__ or "").strip()
            lines.append(f"\t<tool name=\"{tool_name}\">")
            if doc:
                lines.append(f"\t\t<description>")
                for line in doc.split("\n"):
                    lines.append(f"\t\t\t{line}")
                lines.append(f"\t\t</description>")
            lines.append("\t</tool>")
        return "\n".join(lines)

    tools_xml = build_tools_xml_description(TOOL_REGISTRY)
    ```

2. Next, we create the prompt template for the thought-generation step. This template instructs the LLM to decide the next best step. We explicitly tell it to output a short paragraph focused on the next action it intends to take.
    ```python
    PROMPT_TEMPLATE_THOUGHT = f"""
    You are deciding the next best step for reaching the user goal. You have some tools available to you.

    Available tools:
    <tools>
    {tools_xml}
    </tools>

    Conversation so far:
    <conversation>
    {{conversation}}
    </conversation>

    State your next thought about what to do next as one short paragraph focused on the next action you intend to take and why.
    Avoid repeating the same strategies that didn't work previously. Prefer different approaches.
    """.strip()
    ```

3. Let's inspect what this prompt actually looks like when rendered. This is essential for debugging—if the LLM doesn't know what tools it has, it can't reason about them.
    ```python
    print(PROMPT_TEMPLATE_THOUGHT)
    ```
    It outputs:
    ```text
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
    {conversation}
    </conversation>

    State your next thought about what to do next as one short paragraph focused on the next action you intend to take and why.
    Avoid repeating the same strategies that didn't work previously. Prefer different approaches.
    ```
    As you can see, the `tools_xml` helper successfully injected the `search` tool's docstring into the prompt. The `{conversation}` placeholder is ready to be filled with the chat history.

4. We wrap this logic in a `generate_thought` function. It takes the current conversation history, formats the prompt, sends it to the Gemini model, and returns the model's textual response.
    ```python
    def generate_thought(conversation: str, tool_registry: dict[str, callable]) -> str:
        """Generate a thought as plain text (no structured output)."""
        tools_xml = build_tools_xml_description(tool_registry)
        prompt = PROMPT_TEMPLATE_THOUGHT.format(conversation=conversation, tools_xml=tools_xml)

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        return response.text.strip()
    ```

With a thought generated (e.g., "I need to use the search tool to find the capital of France"), the agent must translate this intention into a concrete action.

## Action Phase: Function Calling and Parsing

The "Action" phase is where the agent decides its next move. This could be calling a tool or, if the task is complete, providing the final answer to the user. We will leverage Gemini's native function calling capability to handle this decision-making process [[2]](https://ai.google.dev/gemini-api/docs/function-calling).

### System Prompt Strategy

Notice that our action prompt is significantly simpler than our thought prompt. We do not need to manually describe the tools or their arguments in the text. Why? Because we are shifting from "prompt engineering" to "context engineering" using the native capabilities of the model. The system prompt focuses purely on the *strategy*—deciding between acting and finishing—rather than the *mechanics* of how to call a tool.

### Automatic Tool Integration

Gemini's SDK provides a feature where it can automatically inspect Python functions and convert them into the JSON schema required by the model.
- **Docstrings become descriptions:** The model reads the function's docstring to understand *what* the tool does.
- **Type hints become parameters:** The `query: str` type hint tells the model that this tool requires a string argument named `query`.

This means we don't have to manually write JSON schemas. We just pass our Python functions to the client, and it handles the translation.

However, there is a catch. By default, the SDK might try to execute the function for us. For a ReAct agent, we explicitly **disable** this. We want the model to *propose* the tool call, but we want to execute it ourselves. This gives us control to log the "Observation" step, handle errors gracefully, and maintain the scratchpad state.

1. We define Pydantic models to structure the agent's output. `ToolCallRequest` captures the tool name and arguments, while `FinalAnswer` captures the concluding text.
    ```python
    PROMPT_TEMPLATE_ACTION = """
    You are selecting the best next action to reach the user goal.

    Conversation so far:
    <conversation>
    {conversation}
    </conversation>

    Respond either with a tool call (with arguments) or a final answer if you can confidently conclude.
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

    class ToolCallRequest(BaseModel):
        """A request to call a tool with its name and arguments."""
        tool_name: str = Field(description="The name of the tool to call.")
        arguments: dict = Field(description="The arguments to pass to the tool.")

    class FinalAnswer(BaseModel):
        """A final answer to present to the user when no further action is needed."""
        text: str = Field(description="The final answer text to present to the user.")
    ```

2. The core of this phase is the `generate_action` function. It configures the Gemini client with the list of available tools. We pass the Python functions directly to the `tools` parameter. Crucially, we set `automatic_function_calling={"disable": True}`. This ensures the model returns a structured `function_call` object that we can inspect, rather than running the code invisibly.
    ```python
    def generate_action(conversation: str, tool_registry: dict[str, callable] | None = None, force_final: bool = False) -> (ToolCallRequest | FinalAnswer):
        """Generate an action by passing tools to the LLM and parsing function calls or final text."""
        
        # Use a dedicated prompt when forcing a final answer or no tools are provided
        if force_final or not tool_registry:
            prompt = PROMPT_TEMPLATE_ACTION_FORCED.format(conversation=conversation)
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt
            )
            return FinalAnswer(text=response.text.strip())

        # Default action prompt
        prompt = PROMPT_TEMPLATE_ACTION.format(conversation=conversation)

        # Provide the available tools to the model; disable auto-calling so we can parse and run ourselves
        tools = list(tool_registry.values())
        config = types.GenerateContentConfig(
            tools=tools,
            automatic_function_calling={"disable": True}
        )
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=config
        )

        # Extract the function call from the response (if present)
        candidate = response.candidates[0]
        parts = candidate.content.parts
        if parts and getattr(parts[0], "function_call", None):
            name = parts[0].function_call.name
            args = dict(parts[0].function_call.args) if parts[0].function_call.args is not None else {}
            return ToolCallRequest(tool_name=name, arguments=args)
        
        # Otherwise, it's a final answer
        final_answer = "".join(part.text for part in candidate.content.parts)
        return FinalAnswer(text=final_answer.strip())
    ```

Now we have all the individual components: a tool, a way to generate thoughts, and a way to decide on actions. It is time to assemble them into a cohesive control loop.

## Control Loop: Messages, Scratchpad, Orchestration

The control loop is the heart of our agent. It manages the flow of the ReAct cycle: Thought → Action → Observation. It maintains the state of the conversation in a "scratchpad" and iterates through the cycle until the user's question is answered or a set limit is reached.

```mermaid
graph TD
    A["User Query"] --> B["Generate Thought"]
    B --> C{"Is a Tool Call Needed?"}
    C -->| "Yes" | D["Execute Tool"]
    D --> E["Observe Result"]
    E --> F{"Max Turns Reached?"}
    F -->| "No" | B
    F -->| "Yes" | G["Provide Final Answer (Max Turns)"]
    C -->| "No (Final Answer)" | H["Provide Final Answer"]
    G --> I["End"]
    H --> I
```
Image 1: The ReAct control loop, showing the iterative process of Thought, Action, and Observation, including initial user query, tool interaction, and termination conditions.

### Message Structure Foundation

To build a robust loop, we first need a robust way to store history. We define a `Message` class and a `MessageRole` enum. This allows us to categorize every interaction.
- **User:** The initial input.
- **Thought:** The agent's internal reasoning.
- **Tool Request:** The intent to act.
- **Observation:** The result from the world.
- **Final Answer:** The conclusion.

This structured history, or "scratchpad," is what allows the agent to remember what it just did. Without it, the agent would be amnesiac, repeating the same search over and over.

1. We define the data structures.
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
        role: MessageRole = Field(description="The role of the message in the ReAct loop.")
        content: str = Field(description="The textual content of the message.")

        def __str__(self) -> str:
            """Provides a user-friendly string representation of the message."""
            return f"{self.role.value.capitalize()}: {self.content}"
    ```

2. We use a `Scratchpad` class to wrap the list of messages. It handles appending new messages and provides a `to_string` method to serialize the history for the LLM prompt. We also include a helper to pretty-print messages, which is vital for tracing the agent's logic during development.
    ```python
    def pretty_print_message(message: Message, turn: int, max_turns: int, header_color: str = pretty_print.Color.YELLOW, is_forced_final_answer: bool = False) -> None:
        if not is_forced_final_answer:
            title = f"{message.role.value.capitalize()} (Turn {turn}/{max_turns}):"
        else:
            title = f"{message.role.value.capitalize()} (Forced):"

        pretty_print.wrapped(
            text=message.content,
            title=title,
            header_color=header_color,
        )

    class Scratchpad:
        """Container for ReAct messages with optional pretty-print on append."""

        def __init__(self, max_turns: int) -> None:
            self.messages: List[Message] = []
            self.max_turns: int = max_turns
            self.current_turn: int = 1

        def set_turn(self, turn: int) -> None:
            self.current_turn = turn

        def append(self, message: Message, verbose: bool = False, is_forced_final_answer: bool = False) -> None:
            self.messages.append(message)
            if verbose:
                # ... (Color mapping logic) ...
                header_color = role_to_color.get(message.role, pretty_print.Color.YELLOW)
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

### Integrated Observation Processing

The core logic of ReAct functions when the agent *observes* the result of its action. In our loop, this is handled explicitly. When `generate_action` returns a `ToolCallRequest`, we do not just print it; we execute it.
- We look up the function in `TOOL_REGISTRY`.
- We call it with the arguments provided by the LLM.
- We catch any exceptions (e.g., network errors) and return them as text. This prevents the agent from crashing and allows it to "reason" about the error (e.g., "The tool failed, I should try a different query").
- The result is wrapped in a `Message` with the role `OBSERVATION` and added to the scratchpad.

This observation becomes part of the context for the *next* turn's thought process, closing the feedback loop.

3. Finally, we implement the `react_agent_loop`. This function orchestrates the entire process. We will break it down into logical steps to make it easier to follow. First, we initialize the scratchpad and add the user's initial question.
    ```python
    def react_agent_loop(initial_question: str, tool_registry: dict[str, callable], max_turns: int = 5, verbose: bool = False) -> str:
        """Implements the main ReAct control loop."""
        scratchpad = Scratchpad(max_turns=max_turns)

        # Add the user's question to the scratchpad
        user_message = Message(role=MessageRole.USER, content=initial_question)
        scratchpad.append(user_message, verbose=verbose)

        for turn in range(1, max_turns + 1):
            scratchpad.set_turn(turn)
    ```

4. Inside the loop, we first generate a thought and then determine the next action based on the current conversation history.
    ```python
            # Generate a thought based on the current scratchpad
            thought_content = generate_thought(
                scratchpad.to_string(),
                tool_registry,
            )
            thought_message = Message(role=MessageRole.THOUGHT, content=thought_content)
            scratchpad.append(thought_message, verbose=verbose)

            # Generate an action based on the current scratchpad
            action_result = generate_action(
                scratchpad.to_string(),
                tool_registry=tool_registry,
            )
    ```

5. We then handle the result of the action generation. If it is a final answer, we return it immediately. If it is a tool request, we execute the tool and append the observation.
    ```python
            # If the model produced a final answer, return it
            if isinstance(action_result, FinalAnswer):
                final_answer = action_result.text
                final_message = Message(role=MessageRole.FINAL_ANSWER, content=final_answer)
                scratchpad.append(final_message, verbose=verbose)
                return final_answer

            # Otherwise, it is a tool request
            if isinstance(action_result, ToolCallRequest):
                action_name = action_result.tool_name
                action_params = action_result.arguments

                # Add the action to the scratchpad
                params_str = ", ".join([f"{k}='{v}'" for k, v in action_params.items()])
                action_content = f"{action_name}({params_str})"
                action_message = Message(role=MessageRole.TOOL_REQUEST, content=action_content)
                scratchpad.append(action_message, verbose=verbose)

                # Run the action and get the observation
                observation_content = ""
                tool_function = tool_registry[action_name]
                try:
                    observation_content = tool_function(**action_params)
                except Exception as e:
                    observation_content = f"Error executing tool '{action_name}': {e}"

                # Add the observation to the scratchpad
                observation_message = Message(role=MessageRole.OBSERVATION, content=observation_content)
                scratchpad.append(observation_message, verbose=verbose)
    ```

6. Finally, we check if the maximum number of turns has been reached. If so, we force the agent to generate a final answer to ensure the loop terminates gracefully.
    ```python
            # Check if max turns reached. If so, force final answer.
            if turn == max_turns:
                forced_action = generate_action(
                    scratchpad.to_string(),
                    force_final=True,
                )
                if isinstance(forced_action, FinalAnswer):
                    final_answer = forced_action.text
                else:
                    final_answer = "Unable to produce a final answer within the allotted turns."
                final_message = Message(role=MessageRole.FINAL_ANSWER, content=final_answer)
                scratchpad.append(final_message, verbose=verbose, is_forced_final_answer=True)
                return final_answer
    ```

### Code Outputs Analysis

When you run this loop (as we will see in the next section), the state transitions are clearly visible.
- **Turn 1:** The scratchpad contains only the User message. The agent generates a Thought ("I need to search..."). It generates a Tool Request (`search(query='...')`). The code executes this, and the scratchpad now ends with an Observation ("Paris is...").
- **Turn 2:** The scratchpad now contains the full history including the observation. The agent sees this new information. It generates a new Thought ("I have the answer..."). It generates a Final Answer. The loop detects this `FinalAnswer` type and exits.

This step-by-step accumulation of context is what makes the agent "stateful" and adaptive.

With the full loop implemented, it is time to test our agent and see it in action.

## Tests and Traces: Success and Graceful Fallback

We will validate our ReAct agent with two test cases. Running the loop with `verbose=True` allows us to see the full trace of its thought process, confirming that the thought, action, observation, and control loop components are working together as expected.

1. First, let's ask a question that our mock `search` tool knows how to answer: "What is the capital of France?" We limit the agent to two turns.
    ```python
    question = "What is the capital of France?"
    final_answer = react_agent_loop(question, TOOL_REGISTRY, max_turns=2, verbose=True)
    ```
    The output trace clearly shows the ReAct cycle in action. In Turn 1, the agent generates a **Thought** identifying the need to find the capital. It issues a **Tool request** for `search(query='capital of France')`. The **Observation** returns "Paris is the capital of France...". In Turn 2, the agent generates a new Thought confirming it has the answer and provides the **Final answer**.

2. Now, let's test the agent's ability to handle failure. We ask a question our mock tool does not know: "What is the capital of Italy?"
    ```python
    question = "What is the capital of Italy?"
    final_answer = react_agent_loop(question, TOOL_REGISTRY, max_turns=2, verbose=True)
    ```
    In Turn 1, the agent searches for "capital of Italy", and the **Observation** is "Information about 'capital of Italy' was not found." In Turn 2, the agent adapts its strategy in the **Thought** phase, deciding to try a broader search for just "Italy". The tool again returns "not found." Since we hit the `max_turns` limit, the loop triggers the forced final answer logic. The agent gracefully concludes: "I'm sorry, but I couldn't find information about the capital of Italy."

These tests confirm that our simple, from-scratch ReAct agent is fully functional. It can reason about a user's query, use a tool to find information, process the result, and provide a sensible final response even when it fails to find the answer.

## Conclusion

We have successfully built the core engine of an autonomous agent. We did not use any "magic" frameworks—just Python, a dictionary of functions, and the Gemini API. We defined a tool, created a way for the model to "think" about using it, and built a control loop to manage the conversation.

This hands-on exercise gives you a solid mental model of how agents actually work. In the next lesson, we will expand on this foundation by adding **Memory**, allowing our agent to retain information over longer interactions and become even more capable.

## References

1. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. arXiv.  https://arxiv.org/pdf/2210.03629 

2. ReAct Agent. (n.d.). IBM.  https://www.ibm.com/think/topics/react-agent 

3. AI Agent Planning. (n.d.). IBM.  https://www.ibm.com/think/topics/ai-agent-planning 

4. Schluntz, E., & Zhang, B. (n.d.). Building effective agents. Anthropic.  https://www.anthropic.com/engineering/building-effective-agents 

5. Schmid, P. (n.d.). ReAct agent from scratch with Gemini 2.5 and LangGraph. Google AI for Developers.  https://ai.google.dev/gemini-api/docs/langgraph-example 

6. Function calling with the Gemini API. (n.d.). Google AI for Developers.  https://ai.google.dev/gemini-api/docs/function-calling 

7. Shankar, A. (2024, May 18). Building ReAct Agents from Scratch using Gemini. Medium.  https://medium.com/google-cloud/building-react-agents-from-scratch-a-hands-on-guide-using-gemini-ffe4621d90ae 

8. AI Agent Orchestration. (n.d.). IBM.  https://www.ibm.com/think/topics/ai-agent-orchestration 

9. From LLM Reasoning to Autonomous AI Agents. (n.d.). ArXiv.  https://arxiv.org/pdf/2504.19678