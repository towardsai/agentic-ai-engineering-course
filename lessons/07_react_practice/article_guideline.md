## Global Context

- **What I'm planning to share:** This lesson transitions from theory to practice. We will build a simple ReAct agent from the ground up using Python and a standard LLM API. The focus will be on the practical implementation of the `Thought -> Action -> Observation` loop. We will write the code for the main agentic control loop, define and use simple (or mocked) tools, parse the LLM's output to determine the next step, and manage the agent's state (its "scratchpad") across turns.
- **Why I think it's valuable:** Building an agent, even a very simple one, solidifies the conceptual understanding gained in the previous lessons. This hands-on experience is crucial for AI Engineers to grasp the core mechanics of agentic systems. It clarifies the process and provides a foundational codebase and mental model that can be extended to build more sophisticated and robust agents, making it easier to debug and customize them later.
- **Who the intended audience is:** Aspiring AI engineers learning for the first time how reasoning and planning can be implemented with LLMs using the ReAct pattern.
- **Expected length of the article in words** (where 200-250 words ~= 1 minute of reading time): 3000 words (15 minutes of reading time).

## Outline 

1. Setting Up the Environment and API Connection
2. Building the Tool Layer: Mock Search Implementation
3. Implementing the Thought Phase with Structured Outputs
4. Implementing the Action Phase with Function Calling
5. Creating the ReAct Control Loop and Testing

## Section 1: Setting Up the Environment and API Connection

- **Objective:** Prepare the development environment and establish connection to Google's Gemini API for our ReAct agent implementation.
- Set up necessary imports and explain their purpose:
  - `google.generativeai` for LLM API access
  - `json` for parsing structured responses
  - `pydantic` for data validation in structured outputs
- **API Configuration:** Guide students through getting and configuring the Google API key:
  - Explain how to obtain an API key from Google AI Studio (https://aistudio.google.com/)
  - Show how to set it in Google Colab using `from google.colab import userdata`
  - Initialize the Gemini model with proper configuration
- Provide the import code and API setup code from the notebook, explaining each step clearly.
- **Focus Areas:**
  - Proper API key management and security considerations
  - Understanding the role of each import in the ReAct implementation
  - Setting up the foundation for structured interactions with the LLM

## Section 2: Building the Tool Layer: Mock Search Implementation

- **Objective:** Create a simple but effective mock search tool that demonstrates how external tools integrate with the ReAct framework.
- **Tool Design Philosophy:** Explain why we use a mock tool rather than real API calls:
  - Simplifies the learning focus to ReAct mechanics
  - Eliminates external dependencies and API key requirements
  - Provides predictable responses for testing
- **Implementation Details:**
  - Walk through the search function implementation from the notebook
  - Explain the function signature and documentation
  - Show how the mock responses are structured for different query types
  - Demonstrate the fallback behavior for unhandled queries
- **Real-World Context:** Discuss how this mock search could be replaced with actual search APIs (Google Search, Bing, specialized knowledge bases) in production.
- **Code Explanation:** Provide the complete search function code with detailed comments explaining the pattern matching and response logic.

## Section 3: Implementing the Thought Phase with Structured Outputs

- **Objective:** Implement the "Thought" component of ReAct using Gemini's structured output capabilities to ensure consistent and parseable reasoning.
- **System Prompt Design:** Analyze the thought system prompt:
  - How it guides the LLM to analyze the current situation
  - The three key questions it asks the agent to consider
  - How it maintains focus on the reasoning process
- **Structured Output Implementation:**
  - Explain the `ThoughtResponse` Pydantic model and its purpose
  - Show how to configure Gemini for JSON output with schema validation
  - Demonstrate the `generate_thought` function step by step
- **State Management:** Explain how the scratchpad content is passed to maintain context across turns.
- **Code Implementation:** Present the complete thought phase code from the notebook, including:
  - The system prompt definition
  - The Pydantic model for structured output
  - The thought generation function with proper error handling
- **Why Structured Outputs:** Discuss the benefits of using structured outputs over free-form text parsing for the thought phase.

## Section 4: Implementing the Action Phase with Function Calling

- **Objective:** Build the "Action" component that determines what the agent should do next, using Gemini's function calling capabilities.
- **System Prompt Strategy:** Analyze the action system prompt:
  - How it presents available tools to the LLM
  - The decision logic for choosing between tool use and final response
  - The importance of clear tool descriptions
- **Function Calling Implementation:**
  - Explain how to configure Gemini with tool definitions
  - Show the parsing logic for function calls vs. text responses
  - Demonstrate handling of different action types (search, finish, unknown)
- **Response Parsing:** Walk through the logic for:
  - Detecting function calls in the response
  - Extracting function names and arguments
  - Handling direct text responses for final answers
- **Code Walkthrough:** Present the complete action phase implementation:
  - The action system prompt
  - The `generate_action` function with dual response handling
  - The tuple return format for action name and arguments
- **Error Handling:** Discuss how to handle unknown actions and malformed responses.

## Section 5: Creating the ReAct Control Loop and Testing

- **Objective:** Implement the main ReAct control loop that orchestrates the thought-action-observation cycle and demonstrate its functionality with practical examples.
- **Control Loop Architecture:**
  - Explain the main loop structure with turn-based iteration
  - Detail the scratchpad mechanism for maintaining conversation history
  - Show how the loop terminates with final answers or timeout
  - Provide a mermaid diagram illustrating the ReAct control loop architecture, showing the complete cycle: User Query → Initialize Scratchpad → Thought Generation → Action Decision → Tool Execution (if needed) → Observation → Update Scratchpad → Loop Back or Terminate.
- **State Management Deep Dive:**
  - How the scratchpad accumulates the conversation history
  - The format for logging thoughts, actions, and observations
  - How context is preserved across multiple turns
- **Complete Implementation:** Present the full `react_agent_loop` function:
  - Turn-by-turn processing logic
  - Scratchpad content management
  - Action execution and observation handling
  - Termination conditions and error handling
- **Testing and Demonstration:**
  - Show the first test example with a simple factual question
  - Analyze the output trace showing the complete ReAct cycle
  - Present the second test with a more complex conceptual question
  - Explain how the agent's reasoning and actions differ between examples
- **Code Outputs Analysis:** Comment on the actual outputs from the notebook, showing:
  - How the agent reasons through different types of questions
  - The search tool integration and response handling
  - The final answer synthesis process
- **Extension Possibilities:** Briefly discuss how this basic implementation could be extended with more sophisticated tools, better error handling, and more complex reasoning patterns.

## Article code

Links to code that will be used to support the article. Always prioritize this code over every other source code: 

- [Notebook code for the lesson](https://github.com/towardsai/course-ai-agents/blob/main/lessons/07_react_practice/notebook.ipynb)

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