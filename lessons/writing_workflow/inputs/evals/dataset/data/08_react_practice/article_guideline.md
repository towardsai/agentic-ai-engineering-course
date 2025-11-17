## Global Context of the Lesson

### What We Are Planning to Share

This is a 100% practice lesson. It shows how to build a minimal ReAct agent end-to-end using Python and the Gemini API, mirroring the code in the associated notebook. You will implement the full Thought → Action → Observation loop: define a mock tool, generate thoughts, select actions with function calling, execute tools, process observations, and run a turn-based control loop.

### Why We Think It's Valuable

Hands-on construction of the ReAct loop gives you a concrete mental model for how reasoning agents actually work. With a working control loop, you can extend, debug, and customize agents with confidence.

### Expected Length of the Lesson

3000 words (without titles and references), where we assume 200–250 words ≈ 1 minute of reading time.

### Theory / Practice Ratio

0% theory – 100% practical, step-by-step following the notebook.

## Achoring the Lesson in the Course

### Details About the Course

This piece is part of a broader course on AI agents and LLM workflows. The course consists of 3 parts, each with multiple lessons. 

Thus, it's essential to always anchor this piece into the broader course, understanding where the reader is in its journey. You will be careful to consider the following:
- The points of view.
- To not reintroduce concepts already thought in the previous lesson.
- To be careful when talking about concepts introduced only in future lessons.
- To always reference previous and future lessons when discussing topics outside the piece's scope.

### Lesson Scope

This is lesson 8 (from part 1) of the course on AI Agents.

### Point of View

The course is created by a team writing for a single reader, also known as the student. Thus, for voice consistency across the course, we will always use 'we,' 'our,' and 'us' to refer to the team who creates the course, and 'you' or 'your' to address the reader. Avoid singular first person and don't use 'we' to refer to the student.

Examples of correct point of view:
- Instead of "Before we can choose between workflows and agents, we need a clear understanding of what they are." word it as "To choose between workflows and agents, you need a clear understanding of what they are."

### Who Is the Intended Audience

Aspiring AI engineers learning to implement a ReAct agent from scratch.

### Concepts Introduced in Previous Lessons

In previous lessons of the course, we introduced the following concepts:

Part 1:
- Lesson 1 - AI Engineering & Agent Landscape: Role, stack, and why agents matter now
- Lesson 2 - Workflows vs. Agents: Predefined logic vs. LLM-driven autonomy
- Lesson 3 - Context Engineering: Managing information flow to LLMs
- Lesson 4 - Structured Outputs: Reliable data extraction from LLM responses
- Lesson 5 - Basic Workflow Ingredients: Chaining, parallelization, routing, orchestrator-worker
- Lesson 6 - Agent Tools & Function Calling: Giving your LLM the ability to take action
- Lesson 7 - LLM Planning & Reasoning (ReAct and Plan-and-Executre)

As this is only the 8th lesson of the course, we haven't introduced too many concepts. At this point, the reader only knows what an LLM is and a few high-level ideas about the LLM workflows and AI agents landscape.

### Concepts That Will Be Introduced in Future Lessons

In future lessons of the course, we will introduce the following concepts:

Part 1:
- Lesson 9 - Agent Memory & Knowledge: Procedural, episodic, semantic
- Lesson 10 - RAG Deep Dive: Knowledge-augmented retrieval and generation
- Lesson 11 - Multimodal Processing: Documents, images, and complex data

Part 2:

- MCP
- Developing the research agent and the writing agent

Part 3:

- Making the research and writing agents ready for production
- Monitoring
- Evaluations

If you must mention these, keep it high-level and note we will cover them in their respective lessons.

### Anchoring the Reader in the Educational Journey

Within the course we are teaching the reader multiple topics and concepts. Thus, understanding where the reader is in it's educational journey it's critical for this piece. You have to use only previous introduced concepts, while being reluctant about using concepts that haven't been introduced yet.

When discussing the **concepts introduced in previous lessons** listed in the `Concepts Introduced in Previous Lessons` section, avoid reintroducing them to the reader. Especially don't reintroduce the acronyms. Use them as if the reader already knows what they are. 

Avoid using all the **concepts that haven't been introduced in previous lessons** listed in the `Concepts That Will Be Introduced in Future Lessons` subsection. Whenever another concept requires references to these banned concepts, instead of directly using them, use intuitive analogies and or explanations that are more general and easier to understand, as you would explain them to a 7-year-old. For example:
- If the "tools" concept wasn't introduced yet and you have to talk about agents, refer to them as "actions".
- If the "routing" concept wasn't introduced yet and you have to talk about it, refer to it as "guiding the workflow between multiple decisions".
You can use the concepts that haven't been introduced in previous lessons listed in the `Concepts That Will Be Introduced in Future Lessons` subsection, only if we explicitly specify them. Still, even in that case, as the reader doesn't know how that concept works, you are just allowed to use the term, while keeping the explanation extremely high-level and intuitive, as if you were explaining it to a 7-year-old.
Whenever you use a concept from the `Concepts That Will Be Introduced in Future Lessons` subsection, explicitly specify in what lesson it will be explained in more detail, leveraging the particulars from the subsection. If not explicitly specified in the subsection, simply state that we will cover it in future lessons without providing a concrete lesson number. 

In all use cases avoid using acronyms that aren't explicitly stated in the guidelines. Rather use other more accessible synonyms or descriptions that are easier to understand by non-experts.

## Narrative Flow of the Lesson

Follow the next narrative flow when writing the end-to-end lesson:

- What problem are we learning to solve? Why is it essential to solve it?
    - Start with a personal story where we encountered the problem
- Why other solutions are not working and what's wrong with them.
- At a theoretical level, explain our solution or transformation. Highlight:
    - The theoretical foundations.
    - Why is it better than other solutions?
    - What tools or algorithms can we use?
- Provide some hands-on examples.
- Go deeper into the advanced theory.
- Provide a more complex example supporting the advanced theory.
- Connect our solution to the bigger field of AI Engineering. Add course next steps.

## Lesson Outline 

1. Setup and Environment: Kernel, env, client, model selection
2. Tool Layer: Mock `search` tool and registry
3. Thought Phase: Prompt construction and thought generation
4. Action Phase: Function calling and action parsing
5. Control Loop: Messages, scratchpad, orchestration
6. Tests and Traces: Two runs demonstrating success and graceful fallback

## Section 1 - Setup and Environment

- Objective: Ensure your environment runs the notebook seamlessly and that outputs match expected traces.
- Step-by-step from the notebook:
  - Code cell [2]: Load environment variables via `lessons.utils.env.load(...)`.
  - Code cell [3]: Imports (`google-genai`, `pydantic`, `enum`, `typing`, `lessons.utils.pretty_print`).
  - Code cell [4]: Initialize `client = genai.Client()`.
    - Expected stderr: “Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.” (your message may vary).
  - Code cell [5]: Define `MODEL_ID = "gemini-2.5-flash"`.
- Transition to tools: With the client and model in place, we can define an external capability the agent can use.
- Section length: 250–350 words

## Section 2 - Tool Layer: Mock Search Implementation

- **Objective:** Create a simple but effective mock search tool that demonstrates how external tools integrate with the ReAct framework.
- **Tool Design Philosophy:** Explain why we use a mock tool rather than real API calls:
  - Simplifies the learning focus to ReAct mechanics
  - Eliminates external dependencies and API key requirements
  - Provides predictable responses for testing
- **Implementation Details:**
  - Walk through the search function implementation from the notebook
  - Explain the function signature and docstring documentation
  - Show how the mock responses are structured for different query types
  - Demonstrate the fallback behavior for unhandled queries
- **Real-World Context:** Discuss how this mock search could be replaced with actual search APIs (Google Search, Bing, specialized knowledge bases) in production.
- **Section length:** 500 words

## Section 3 - Thought Phase: Prompt Construction and Generation

- Objective: Produce a short, purposeful thought guiding the next step for the ReAct agent.
- Step-by-step from the notebook:
  - Code cell [8]: Build tools XML with `build_tools_xml_description(TOOL_REGISTRY)` and define `PROMPT_TEMPLATE_THOUGHT` using `{conversation}`.
  - Code cell [9]: `print(PROMPT_TEMPLATE_THOUGHT)` to inspect the full prompt.
    - Explain the output: XML block with one `<tool name="search">` containing the docstring, plus the `<conversation>` placeholder.
  - Code cell [10]: Implement `generate_thought(conversation, tool_registry)` that formats the prompt and returns `response.text.strip()`.
- What to verify: The printed prompt shows the tool description and the conversation placeholder exactly as expected.
- Transition to acting: With a coherent thought, we must either call a tool or conclude with a final answer.
- Section length: 400–500 words

## Section 4 - Action Phase: Function Calling and Parsing

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

## Section 5 - Control Loop: Messages, Scratchpad, Orchestration

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
  - The role of helper utilities like `pretty_print_message` for readable traces
- **Code Outputs Analysis:** Comment on the actual outputs from the notebook, showing:
  - How the agent reasons through different types of questions
  - The search tool integration and response handling
  - The observation processing and state updates within the loop
  - The final answer synthesis process and forced termination behavior
  - How error recovery works when tools return mock "not found" responses
- **Extension Possibilities:** Briefly discuss how this basic implementation could be extended with more sophisticated tools, better error handling, and more complex reasoning patterns.
- Give step-by-step examples from the "ReAct Control Loop" section of the Notebook. Testing examples are covered in Section 6. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
 - **Section length:** 1000 words

## Section 6 - Tests and Traces: Success and Graceful Fallback

- **Objective:** Validate the full ReAct cycle with two examples and analyze the printed traces to ensure the loop, tool integration, and forced termination behave as designed.
- **Step-by-step from the notebook:**
  - Code cell [17]: Simple factual question — "What is the capital of France?" with `max_turns=2, verbose=True`.
    - Expected trace highlights (color-coded titles in the notebook output):
      - Thought (Turn 1/2): Intends to use the `search` tool for a factual lookup.
      - Tool request (Turn 1/2): `search(query='capital of France')`
      - Observation (Turn 1/2): "Paris is the capital of France and is known for the Eiffel Tower."
      - Thought (Turn 2/2): Summarizes that the answer was found and will be communicated.
      - Final answer (Turn 2/2): "Paris is the capital of France."
    - What to verify:
      - The action phase correctly produces a `ToolCallRequest` with the proper function name and arguments.
      - The control loop executes the tool, captures the observation, and concludes within the turn budget.
  - Code cell [19]: Unknown/unsupported query for the mock tool — "What is the capital of Italy?" with `max_turns=2, verbose=True`.
    - Expected trace highlights:
      - Thought (Turn 1/2) → Tool request (Turn 1/2): `search(query='capital of Italy')`
      - Observation (Turn 1/2): "Information about 'capital of Italy' was not found."
      - Thought (Turn 2/2): Adopts a broader strategy.
      - Tool request (Turn 2/2): `search(query='Italy')`
      - Observation (Turn 2/2): "Information about 'Italy' was not found."
      - Final answer (Forced): "I'm sorry, but I couldn't find information about the capital of Italy."
    - What to verify:
      - Strategy changes across turns are reflected in the Thought messages.
      - Forced final answer path is triggered at the max turn boundary and returns a concise response.
- **Transition:** These tests confirm the end-to-end loop and provide a baseline for extending the agent with richer tools and behaviors in later lessons.
- **Section length:** 450–600 words

## Article Code

Links to code that will be used to support the article. Always prioritize this code over every other source code: 

- [Notebook code for the lesson](https://github.com/towardsai/course-ai-agents/blob/dev/lessons/08_react_practice/notebook.ipynb)

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