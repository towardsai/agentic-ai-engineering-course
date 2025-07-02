## Global Context

- **What I’m planning to share:** This lesson transitions from theory to practice. We will build a simple ReAct agent from the ground up using Python and a standard LLM API. The focus will be on the practical implementation of the `Thought -> Action -> Observation` loop. We will write the code for the main agentic control loop, define and use simple (or mocked) tools, parse the LLM's output to determine the next step, and manage the agent's state (its "scratchpad") across turns.
- **Why I think it’s valuable:** Building an agent, even a very simple one, solidifies the conceptual understanding gained in the previous lesson. This hands-on experience is crucial for AI Engineers to grasp the core mechanics of agentic systems. It demystifies the process and provides a foundational codebase and mental model that can be extended to build more sophisticated and robust agents, making it easier to debug and customize them later.
- **Who the intended audience is:** People learning for the first time how reasoning and planning can be implemented with LLMs.
- **Expected length of the article in words** (where 200-250 words ~= 1 minute of reading time): 3000 words (15 minutes of reading time).

## Outline

### Section 1: Building a Simple ReAct Agent from Scratch (Practical Notebook)

-   **Objective:** Guide the student through the implementation of a basic ReAct agent using Python and an LLM API Google AI. The agent's task will be simple, designed to require a tool lookup, such as answering a question using a mocked search tool.
-   Use code from the provided notebook below. Provide one code cell at a time (don't split it) and explain it. Also explain the code outputs if provided.
-   **Focus Areas for Learning:**
    -   **Prompt Engineering for ReAct:** Demonstrate how to meticulously craft the system prompt and iterative prompts to constrain the LLM's output to the desired `Thought/Action` format.
    -   **Parsing LLM Output:** Cover simple but effective string parsing techniques to extract the action and its arguments from the model's free-form text response.
    -   **State Management:** Explicitly show how to manage the agent's state by accumulating the `Thought-Action-Observation` trail in a "scratchpad" or history list, which is then fed back into the prompt for the next turn to provide context.
-   Provide clear, commented Python code snippets for each component to ensure the focus remains on the ReAct mechanics rather than on complex tool integrations or advanced error handling.
-   This section contains the whole code of the lesson. It should be comprehensive and detailed.

## References

- https://www.ibm.com/think/topics/ai-agent-orchestration
- https://ar5iv.labs.arxiv.org/html/2504.19678
- https://www.ibm.com/think/topics/react-agent
- https://www.ibm.com/think/topics/ai-agent-planning
- https://medium.com/google-cloud/building-react-agents-from-scratch-a-hands-on-guide-using-gemini-ffe4621d90ae

## Notebook

<notebook>
```python
import google.generativeai as genai
import os
import json
from pydantic import BaseModel
```


## Connect to Gemini

```python
# Ensure the GOOGLE_API_KEY environment variable is set.
# You can get an API key from Google AI Studio: https://aistudio.google.com/
from google.colab import userdata

api_key = userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=userdata.get('GOOGLE_API_KEY'))

# Initialize the LLM model (e.g., Gemini)
llm_model = genai.GenerativeModel('gemini-2.5-flash')
```


## Search Tool Definition

This is a simple and mocked tool as per the lesson's focus.

```python
def search(query: str) -> str:
    """
    Search for information about a specific topic or query.

    Args:
        query (str): The search query or topic to look up

    Returns:
        str: Search results containing information about the queried topic

    Note:
        This is a simple mocked search tool for demonstration purposes.
        In a real scenario, this would call a search API like Google Search,
        Bing Search, or a specialized knowledge base API.
    """
    query_lower = query.lower()

    # Predefined responses for demonstration
    if all(word in query_lower for word in ["capital", "france"]):
        return "Paris is the capital of France and is known for the Eiffel Tower."
    elif "react" in query_lower:
        return "The ReAct (Reasoning and Acting) framework enables LLMs to solve complex tasks by interleaving thought generation, action execution, and observation processing."

    # Generic response for unhandled queries
    return f"Mock search result: Information about '{query}' was not found in the predefined mock responses. A real search tool would provide more."
```


## Thought Phase

```python
SYSTEM_PROMPT_THOUGHT = """You are a reasoning agent analyzing the user's question and your current knowledge to determine what information is needed next.

Based on the conversation history, think about:
1. What information do you currently have?
2. What information is still needed to answer the user's question?
3. What should be the next step?

Provide your detailed reasoning and analysis of the current situation."""


class ThoughtResponse(BaseModel):
    thought: str


def generate_thought(scratchpad_content: str) -> str:
    """Generate a thought using structured output"""
    prompt = f"{SYSTEM_PROMPT_THOUGHT}\n\nConversation so far:\n{scratchpad_content}\n\nWhat is your thought about the next step?"

    response = llm_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            response_schema=ThoughtResponse
        )
    )

    thought_data = json.loads(response.text)
    return thought_data["thought"]
```


## Action Phase

```python
SYSTEM_PROMPT_ACTION = """You are a reasoning agent that takes actions based on your thoughts.

Available tools:
- search: Use this to find information about a specific topic

Choose the appropriate action based on your previous thought. If you need more information, use the search tool. If you have enough information to provide a complete answer, respond directly with your final answer (do not use any tool)."""


def generate_action(scratchpad_content: str) -> tuple[str, str]:
    """Generate an action using function calling or direct text response"""
    prompt = f"{SYSTEM_PROMPT_ACTION}\n\nConversation so far:\n{scratchpad_content}\n\nBased on your thought, what action should you take?"

    # Configure the model with function declarations
    model_with_tools = genai.GenerativeModel(
        'gemini-2.5-flash',
        tools=[search]
    )

    response = model_with_tools.generate_content(prompt)

    # Check if response contains a function call or text
    first_part = response.candidates[0].content.parts[0]

    if hasattr(first_part, 'function_call') and first_part.function_call:
        # It's a function call
        function_call = first_part.function_call
        action_name = function_call.name
        if action_name == "search":
            action_arg = function_call.args["query"]
        else:
            action_arg = ""
        return action_name, action_arg
    else:
        # It's a text response (final answer)
        return "finish", response.text
```


## ReAct Control Loop

```python
def react_agent_loop(initial_question: str, max_turns: int = 3, verbose: bool = False) -> str:
    """
    Implements the main ReAct (Thought -> Action -> Observation) control loop.
    Uses structured outputs for thoughts and function calling for actions.
    """
    # The scratchpad stores the history of thoughts, actions, and observations.
    scratchpad = []

    for turn in range(max_turns):
        if verbose:
            print(f"\n--- Turn {turn + 1}/{max_turns} ---")

        # If first turn, then write the user question
        if not scratchpad:
            user_question_log = f"User Question: {initial_question}"
            scratchpad.append(user_question_log)
            if verbose:
                print(user_question_log)

        # Generate thought using structured output
        scratchpad_content = "\n".join(scratchpad)
        thought = generate_thought(scratchpad_content)
        current_thought_log = f"Thought: {thought}"
        scratchpad.append(current_thought_log)
        if verbose:
            print(current_thought_log)

        # Generate action using function calling
        action_name, action_arg = generate_action(scratchpad_content)
        current_action_log = f"Action: {action_name}['{action_arg}']"
        scratchpad.append(current_action_log)
        if verbose:
            print(current_action_log)

        # Handle the action
        if action_name == "search":
            tool_output = search(action_arg)
            current_observation_log = f"Observation: {tool_output}"
            scratchpad.append(current_observation_log)
            if verbose:
                print(current_observation_log)
        elif action_name == "finish":
            final_answer = action_arg
            final_answer_log = f"Final Answer: {final_answer}"
            scratchpad.append(final_answer_log)
            if verbose:
                print(final_answer_log)
            return final_answer # Terminate the loop and return the answer
        else:
            # Unknown action
            current_observation_log = f"Observation: Error - Unknown action '{action_name}'. Available tools are [search]."
            scratchpad.append(current_observation_log)
            if verbose:
                print(current_observation_log)

        # Check if max turns reached without completing
        if turn == max_turns - 1:
            current_observation_log = f"Observation: Error - Max turns reached without providing final answer."
            scratchpad.append(current_observation_log)
            if verbose:
                print(current_observation_log)
            return "Agent did not provide a final answer within the turn limit."
```


Let's test the `react_agent_loop` function.

```python
# A straightforward question requiring a search.
question = "What is the capital of France?"
final_answer = react_agent_loop(question, max_turns=5, verbose=True)
```

**Output:**
```

--- Turn 1/5 ---
User Question: What is the capital of France?
Thought: The user is asking a direct factual question about the capital of France. This information is readily available as general knowledge. The next step should be to directly provide the answer to the user.
Action: search['capital of France']
Observation: Paris is the capital of France and is known for the Eiffel Tower.

--- Turn 2/5 ---
Thought: I have the user's question: "What is the capital of France?" I performed a search action and received the observation: "Paris is the capital of France and is known for the Eiffel Tower." This observation directly answers the user's question. Therefore, all necessary information has been gathered. The next step is to provide the answer to the user.
Action: finish['Paris is the capital of France.']
Final Answer: Paris is the capital of France.
```


```python
# A question about a concept the mock search tool might know.
question = "Can you explain the ReAct framework in AI?"
final_answer = react_agent_loop(question, max_turns=5, verbose=True)
```

**Output:**
```

--- Turn 1/5 ---
User Question: Can you explain the ReAct framework in AI?
Thought: The user is asking for an explanation of the ReAct framework. This is a direct knowledge-based question. I possess the necessary information to provide a comprehensive explanation of ReAct. Therefore, the next step is to generate the explanation directly.
Action: search['ReAct framework in AI']
Observation: The ReAct (Reasoning and Acting) framework enables LLMs to solve complex tasks by interleaving thought generation, action execution, and observation processing.

--- Turn 2/5 ---
Thought: I have the user's question about the ReAct framework and a concise definition obtained from the previous search (Observation). The definition states that ReAct enables LLMs to solve complex tasks by interleaving thought generation, action execution, and observation processing. While this provides the core concept, a comprehensive explanation requires elaborating on each of these components, how they interact, the benefits of the framework, and potentially an illustrative example. My current information is sufficient to build this detailed explanation without needing further searches. Therefore, the next logical step is to synthesize this information and provide the direct answer to the user's question.
Action: finish['The ReAct (Reasoning and Acting) framework is an AI paradigm that allows large language models (LLMs) to tackle complex tasks by combining reasoning and acting. It achieves this by interleaving three key processes:

1.  **Thought Generation:** The LLM generates internal thoughts, which are essentially its reasoning process, planning the next steps or considering different approaches.
2.  **Action Execution:** Based on its thoughts, the LLM takes an action. This could involve using a tool, querying a knowledge base, or interacting with an environment.
3.  **Observation Processing:** After executing an action, the LLM processes the observation, which is the result or feedback from the executed action. This observation then informs the subsequent thought generation, creating a continuous loop of reasoning, acting, and learning from the environment.

This iterative process of thinking, acting, and observing allows LLMs to break down complex problems, learn from their interactions, and ultimately arrive at more robust and accurate solutions.']
Final Answer: The ReAct (Reasoning and Acting) framework is an AI paradigm that allows large language models (LLMs) to tackle complex tasks by combining reasoning and acting. It achieves this by interleaving three key processes:

1.  **Thought Generation:** The LLM generates internal thoughts, which are essentially its reasoning process, planning the next steps or considering different approaches.
2.  **Action Execution:** Based on its thoughts, the LLM takes an action. This could involve using a tool, querying a knowledge base, or interacting with an environment.
3.  **Observation Processing:** After executing an action, the LLM processes the observation, which is the result or feedback from the executed action. This observation then informs the subsequent thought generation, creating a continuous loop of reasoning, acting, and learning from the environment.

This iterative process of thinking, acting, and observing allows LLMs to break down complex problems, learn from their interactions, and ultimately arrive at more robust and accurate solutions.
```
</notebook>