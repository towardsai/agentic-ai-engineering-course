## Global Context

- **What I'm planning to share:** This lesson transitions from theory to practice. We will build a simple ReAct agent from the ground up using Python and the Gemini LLM API. The focus will be on the practical implementation of the `Thought -> Action -> Observation` loop. We will write the code for the main agentic control loop, define and use simple (or mocked) tools, parse the LLM's output to determine the next step, and manage the agent's state (its "scratchpad") across turns. Most AI agents in production use some form of ReAct pattern, making this understanding critical for building intuition to extend, debug, and maintain sophisticated agentic systems.
- **Why I think it's valuable:** Building an agent, even a very simple one, solidifies the conceptual understanding gained in the previous lessons. This hands-on experience is crucial for AI Engineers to grasp the core mechanics of agentic systems. Since most AI agents rely on ReAct patterns, understanding how it works under the hood provides the foundation to build more sophisticated and robust agents, making it easier to debug and customize them later. It clarifies the process and provides a foundational codebase and mental model that can be extended.
- **Who the intended audience is:** Aspiring AI engineers learning for the first time how reasoning and planning can be implemented with LLMs using the ReAct pattern.
- **Expected length of the article in words** (where 200-250 words ~= 1 minute of reading time): 3000 words (15 minutes of reading time).

## Outline 

1. Building the Tool Layer: Mock Search Implementation
2. Implementing the Thought Phase with Structured Outputs
3. Implementing the Action Phase with Function Calling
4. Creating the ReAct Control Loop with Observation Processing and Testing

## Section 1: Building the Tool Layer: Mock Search Implementation

- **Objective:** Create a simple but effective mock search tool that demonstrates how external tools integrate with the ReAct framework.
- **Tool Design Philosophy:** Explain why we use a mock tool rather than real API calls:
  - Simplifies the learning focus to ReAct mechanics
  - Eliminates external dependencies and API key requirements
  - Provides predictable responses for testing
- **Implementation Details:**
  - Walk through the search function implementation from the notebook
  - Explain the function signature and comprehensive docstring documentation
  - Show how the mock responses are structured for different query types
  - Demonstrate the fallback behavior for unhandled queries
- **Automatic Tool Discovery:** Explain how Gemini automatically extracts tool information:
  - By passing a function with its docstring and signature to the tools config, Gemini automatically picks up the tool's description and signature
  - The function's docstring serves as the tool description that the LLM sees
  - Parameter types and names are automatically inferred from the function signature
  - This eliminates the need for manual tool schema definition
- **Real-World Context:** Discuss how this mock search could be replaced with actual search APIs (Google Search, Bing, specialized knowledge bases) in production.
- Give step-by-step examples from the "Search Tool Definition" section of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
- **Section length:** 500 words

## Section 2: Implementing the Thought Phase with Structured Outputs

- **Objective:** Implement the "Thought" component of ReAct using Gemini's structured output capabilities to ensure consistent and parseable reasoning.
- **System Prompt Design:** Analyze the thought system prompt:
  - How it guides the LLM to analyze the current situation
  - The two key questions it asks the agent to consider
  - How it maintains focus on the reasoning process
  - Emphasis on prioritizing external information retrieval over internal knowledge
- **Structured Output Implementation:**
  - Explain the `ThoughtResponse` Pydantic model and its purpose
  - Show how to configure Gemini for JSON output with schema validation
  - Demonstrate the `generate_thought` function step by step
  - Highlight the use of `response.parsed` to directly get the Pydantic object instead of manual JSON parsing
- **State Management:** Explain how the scratchpad content is passed to maintain context across turns.
- **Why Structured Outputs:** Discuss the benefits of using structured outputs over free-form text parsing for the thought phase.
- Give step-by-step examples from the "Thought Phase" section of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
- **Section length:** 600 words

## Section 3: Implementing the Action Phase with Function Calling

- **Objective:** Build the "Action" component that determines what the agent should do next, using Gemini's function calling capabilities.
- **System Prompt Strategy:** Analyze the action system prompt:
  - How the prompt focuses on high-level decision making rather than tool details
  - The emphasis on external information retrieval
  - Why tool descriptions and signatures are not needed in the system prompt
- **Automatic Tool Integration:** Explain how Gemini handles tool information automatically:
  - When functions are passed to the tools config, their docstrings become the tool descriptions
  - Parameter information is extracted from the function signature automatically
  - The system prompt can focus on strategic guidance rather than technical tool details
  - This separation allows for cleaner prompts and easier tool management
- **Function Calling Implementation:**
  - Show how to configure Gemini with tool definitions using the search function
  - Demonstrate the parsing logic for function calls vs. text responses
  - Explain the dual return format: action name and parameters
  - Show handling of different action types (tool calls vs. finish action)
- **Response Parsing:** Walk through the logic for:
  - Detecting function calls in the response using hasattr checks
  - Extracting function names and arguments from function_call objects
  - Handling direct text responses for final answers using the ACTION_FINISH pattern
- **Error Handling:** Discuss how to handle unknown actions and malformed responses.
- Give step-by-step examples from the "Action Phase" section of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
- **Section length:** 700 words

## Section 4: Creating the ReAct Control Loop with Observation Processing and Testing

- **Objective:** Implement the main ReAct control loop that orchestrates the complete thought-action-observation cycle, including integrated observation processing, and demonstrate its functionality with practical examples.
- **Message Structure Foundation:** Start by explaining the Message and MessageRole system from the notebook:
  - How different types of interactions are categorized (user, thought, tool_request, observation, final_answer)
  - The role of the scratchpad in maintaining conversation history
  - How the structured message format enables clear tracking of the ReAct cycle
  - Helper functions for formatting scratchpad content and pretty printing
- **Control Loop Architecture:**
  - Explain the main loop structure with turn-based iteration
  - Detail the scratchpad mechanism for maintaining conversation history using the Message class
  - Show how the loop terminates with final answers or timeout
  - Provide a mermaid diagram illustrating the ReAct control loop architecture, showing the complete cycle: User Query → Initialize Scratchpad → Thought Generation → Action Decision → Tool Execution (if needed) → Observation → Update Scratchpad → Loop Back or Terminate.
- **Integrated Observation Processing:** Explain how observations are seamlessly integrated within the main loop:
  - How tool functions are executed with the extracted parameters using the TOOL_REGISTRY
  - The error handling mechanism for tool execution failures with informative error messages
  - How unknown tool names are handled gracefully with available tool feedback
  - How observations are added to the scratchpad as structured messages
  - The importance of preserving tool results for subsequent reasoning steps
- **Complete Implementation:** Present the full `react_agent_loop` function following the notebook structure:
  - Turn-by-turn processing logic with clear iteration bounds
  - Scratchpad content management using the Message and MessageRole system
  - Action execution and integrated observation handling with proper error handling
  - Termination conditions and forced final answer generation when max turns reached
  - The role of helper functions like `format_scratchpad_for_llm` and `pretty_print_message`
- **Testing and Demonstration:** Walk through all three test cases from the notebook:
  - Show the first test example with a simple factual question about France's capital
  - Present the second test with a more complex conceptual question about ReAct framework
  - Show the third test demonstrating handling of unknown information (Italy's capital)
  - Analyze the output traces showing the complete ReAct cycle with color-coded output
  - Explain how the agent's reasoning and actions differ between examples
- **Code Outputs Analysis:** Comment on the actual outputs from the notebook, showing:
  - How the agent reasons through different types of questions
  - The search tool integration and response handling
  - The observation processing and state updates within the loop
  - The final answer synthesis process and forced termination behavior
  - How error recovery works when tools return mock "not found" responses
- **Extension Possibilities:** Briefly discuss how this basic implementation could be extended with more sophisticated tools, better error handling, and more complex reasoning patterns.
- Give step-by-step examples from the "ReAct Control Loop" section and all testing examples from the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
- **Section length:** 1000 words

## Article code

Links to code that will be used to support the article. Always prioritize this code over every other source code: 

- [Notebook code for the lesson](https://github.com/towardsai/course-ai-agents/blob/main/lessons/08_react_practice/notebook.ipynb)

## Golden Sources

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/pdf/2210.03629)
- [ReAct Agent - IBM](https://www.ibm.com/think/topics/react-agent)
- [AI Agent Planning - IBM](https://www.ibm.com/think/topics/ai-agent-planning)
- [Building effective agents - Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- [ReAct agent from scratch with Gemini 2.5 and LangGraph](https://ai.google.dev/gemini-api/docs/langgraph-example)

## Other Sources

- [From LLM Reasoning to Autonomous AI Agents - ArXiv](https://arxiv.org/pdf/2504.19678)
- [Building ReAct Agents from Scratch using Gemini - Medium](https://medium.com/google-cloud/building-react-agents-from-scratch-a-hands-on-guide-using-gemini-ffe4621d90ae)
- [AI Agent Orchestration - IBM](https://www.ibm.com/think/topics/ai-agent-orchestration)
- [Gemini Function Calling Documentation](https://ai.google.dev/gemini-api/docs/function-calling)