# Research

## Research Results

<details>
<summary>How does the Gemini API's function calling feature compare to traditional ReAct prompting for tool selection and execution?</summary>

### Source [1]: https://codelabs.developers.google.com/codelabs/gemini-function-calling

Query: How does the Gemini API's function calling feature compare to traditional ReAct prompting for tool selection and execution?

Answer: The **Gemini API's function calling** feature works by allowing developers to define one or more function declarations within a tool, making the Gemini model aware of which functions it can call and how to call them. When the user sends a prompt, Gemini analyzes both the user input and available function declarations, then returns a structured Function Call response that specifies the function name and required parameters. The developer is responsible for actually executing the function (e.g. using Python’s `requests` library to call an external REST API) and then passing the API response back to Gemini for further processing or user response. Notably, Gemini does not directly execute external API calls, giving developers flexibility in the choice of APIs, including Google Cloud services or any REST endpoint. This structured approach enables clean separation between model reasoning and tool execution, contrasting with traditional prompt engineering approaches where tool selection and invocation often depend on informal, text-based model outputs[1].

-----

-----

-----

### Source [2]: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling

Query: How does the Gemini API's function calling feature compare to traditional ReAct prompting for tool selection and execution?

Answer: Gemini function calling on Vertex AI requires the developer to submit both the prompt and explicit function declarations (in OpenAPI-compatible schema format) to the model. The model then evaluates whether to generate a standard response or propose one or more function calls, returning structured JSON information with function names and parameters. Gemini supports **parallel function calling** (multiple functions in one turn), enhancing its ability to execute complex multi-step tasks efficiently. This contrasts with traditional ReAct-style prompting, where tool selection and execution logic are typically embedded in the model’s generated text and require parsing or additional reasoning by the developer. Gemini’s approach offers higher reliability and clarity for tool selection and execution, as the model’s outputs are structured and predictable[2].

-----

-----

-----

### Source [3]: https://firebase.google.com/docs/ai-logic/function-calling

Query: How does the Gemini API's function calling feature compare to traditional ReAct prompting for tool selection and execution?

Answer: With Gemini API function calling in Firebase AI Logic, developers can expose external functions (tools) to the model and use a multi-turn chat interface for interaction. Developers provide a schema for each function, specifying expected input and output parameters. The Gemini model will return a function call suggestion with structured parameters when it determines tool use is required. This offers a more robust, programmatic alternative to ReAct prompting, where the model might output a textual action description that must be parsed and mapped to an API call. Gemini’s function calling separates reasoning and execution, reducing ambiguity and error risk in tool selection and invocation[3].

-----

-----

-----

### Source [4]: https://www.youtube.com/watch?v=mVXrdvXplj0

Query: How does the Gemini API's function calling feature compare to traditional ReAct prompting for tool selection and execution?

Answer: The Gemini API’s function calling feature allows the model to produce structured data representing a function call whenever it determines that external tool use is needed. Developers must define function declarations that specify the function’s name, description, and expected parameters. Upon receiving a prompt, Gemini analyzes the input and function declarations, returning structured JSON for the function call. Crucially, Gemini does not execute the function itself; developers must implement the function execution and handle the response. This design is more systematic and less error-prone than ReAct prompting, where the model’s tool selection and execution instructions are usually embedded in free-form text and require interpretation[4].

-----

-----

-----

### Source [5]: https://ai.google.dev/gemini-api/docs/function-calling

Query: How does the Gemini API's function calling feature compare to traditional ReAct prompting for tool selection and execution?

Answer: The Gemini API’s function calling lets developers connect models to external tools and APIs by providing explicit function declarations. When prompted, Gemini analyzes both the input and available functions, deciding whether to respond directly or to call a function. If a function call is needed, Gemini returns a structured response with the function name and its parameters. Developers then execute the function and return results to the model. This process is available in multiple programming languages (Python, JavaScript, etc.), and the Gemini API supports direct configuration of function schemas, making tool use more reliable, scalable, and maintainable than traditional ReAct-style prompting, which relies on interpreting model-generated text for tool actions[5].

-----

-----

</details>

<details>
<summary>What are best practices for designing a ReAct agent's control loop, specifically regarding state management and termination conditions?</summary>

### Source [6]: https://docs.getdynamiq.ai/low-code-builder/llm-agents/guide-to-implementing-llm-agents-react-and-simple-agents

Query: What are best practices for designing a ReAct agent's control loop, specifically regarding state management and termination conditions?

Answer: The execution flow of a **ReAct Agent** is built around a cycle of thinking, acting, and evaluating results. The agent:
- Receives an input query.
- Determines necessary actions.
- Utilizes appropriate tools (such as web search or data extraction tools).
- Assesses the gathered data.
- Repeats the loop if additional information or reasoning is needed.
- Concludes with a formatted response.

**Best practices** highlighted for control loop design include:
- Setting a reasonable `max_loops` value based on the complexity of the task. This prevents infinite loops and ensures timely termination.
- Defining clear agent roles and specific behavioral guidelines, which help manage agent state and ensure consistent decision-making.
- Configuring robust error handling for loop termination, so that the agent can gracefully exit in case of tool failures or unexpected results.
- Using combinations of complementary tools, which enhances the agent’s reasoning and acting capabilities within each loop iteration.

Testing is crucial: the agent should be evaluated with a variety of queries to ensure it handles state transitions correctly and terminates as expected when `max_loops` is reached or when the goal is achieved.

-----

-----

-----

### Source [7]: https://dylancastillo.co/posts/react-agent-langgraph.html

Query: What are best practices for designing a ReAct agent's control loop, specifically regarding state management and termination conditions?

Answer: The control logic of a ReAct agent revolves around a `should_continue` function, which inspects the last message from the LLM:
- If the last message is a tool request, the loop routes to a tool execution step.
- If not, the loop ends and the conversation terminates.

State is managed through a `MessagesState` object, which tracks all messages (thoughts, actions, and observations) in the conversation. Each function (`call_llm` for reasoning, `call_tool` for acting) takes the current state as input and returns updated state, ensuring that all steps have access to the latest context.

A typical loop structure is:
1. Call the LLM for reasoning.
2. If an action is determined, execute the tool and record the result.
3. Pass the result back to the LLM for further reasoning.
4. Continue this loop until `should_continue` determines that the goal is met or no further action is required.

This structure ensures:
- State is consistently updated and passed between steps.
- Termination conditions are explicit, relying either on the LLM’s signal to stop or on loop limits.

-----

-----

-----

### Source [8]: https://dev.to/jamesli/react-vs-plan-and-execute-a-practical-comparison-of-llm-agent-patterns-4gh9

Query: What are best practices for designing a ReAct agent's control loop, specifically regarding state management and termination conditions?

Answer: This article outlines the **ReAct reasoning-action loop mechanism** as a core design for LLM agents. The focus is on:
- Implementing the ReAct pattern with careful attention to the agent’s reasoning and acting steps.
- Comparing ReAct with Plan-and-Execute patterns, emphasizing that ReAct relies on iterative loops of decision-making and action.

While specific control loop or state management implementation details are not provided, the article underlines:
- The importance of matching the loop design to the complexity and requirements of the scenario.
- The need for systematic selection of loop termination criteria to balance performance (response time, cost) and accuracy.

-----

-----

-----

### Source [9]: https://www.dailydoseofds.com/ai-agents-crash-course-part-10-with-implementation/

Query: What are best practices for designing a ReAct agent's control loop, specifically regarding state management and termination conditions?

Answer: The ReAct paradigm is described as using a **chain-of-thought reasoning** and **tool-using action** cycle:
- The agent maintains a structured trace of its reasoning (thoughts) and actions.
- Each loop iteration involves the agent reasoning about what action to take (or whether to terminate), then executing that action using a tool.

Best practices illustrated include:
- Making the agent’s internal state and reasoning explicit, which aids in debugging and transparency.
- Using structured data (like JSON) for tool inputs and outputs, keeping state management clear and deterministic.
- Automating the loop as much as possible, so the agent can independently reason, act, and decide when to stop.

Although some implementation details are reserved for subscribers, the emphasis is on clarity of state transitions and explicit handling of loop termination.

-----

-----

-----

### Source [10]: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/

Query: What are best practices for designing a ReAct agent's control loop, specifically regarding state management and termination conditions?

Answer: Modern ReAct agents combine:
- **Tool calling**: The LLM selects and uses tools as needed, based on user input and ongoing reasoning.
- **Memory**: The agent retains and references information from previous steps, maintaining a history that informs subsequent reasoning and actions.
- **Planning**: The LLM creates and follows multi-step plans, deciding dynamically when to act, when to reason, and when to stop.

State is managed as a list of messages (thoughts, actions, observations), which is passed and updated at each step. Termination conditions are typically defined by:
- The LLM producing a message that signals completion (e.g., a final answer).
- Hitting a predefined loop limit (to prevent infinite cycles).

Tool integration is facilitated through explicit interface definitions, ensuring the LLM can issue structured requests and process structured responses, which supports robust state management and controlled loop progression.

-----

-----

</details>

<details>
<summary>What are common failure modes and debugging strategies for a ReAct agent built from scratch?</summary>

### Source [11]: https://www.amplework.com/blog/debugging-agentic-ai-tools-techniques/

Query: What are common failure modes and debugging strategies for a ReAct agent built from scratch?

Answer: **Common Failure Modes:**
- **Opaque Decision-Making:** Agentic AI systems, like ReAct agents, can behave unexpectedly due to their autonomous and adaptive nature, making their decision processes difficult to interpret.
- **Unexpected Behavior Sequences:** Since agents operate based on evolving goals and inputs, they may enter unanticipated states or action sequences that are hard to trace without sufficient observability.

**Debugging Strategies:**
- **Behavior Tracing and Action Logging:** Capture every agent action, input, and context. This comprehensive logging enables reconstruction of the agent’s full reasoning process and helps pinpoint why specific decisions were made.
- **Reconstructing Behavior Paths:** By analyzing action logs, developers can trace sequences leading to a problematic outcome, clarifying the root cause of failures.
- **Time-Travel Debugging:** Regularly record system state snapshots, making it possible to compare agent behavior before and after changes, and to identify when issues first appeared. This longitudinal analysis is critical in systems where agent behavior evolves over time.
- **Iterative Comparison:** Use snapshots to observe the impact of code or configuration modifications, isolating the introduction points of bugs or regressions.

These strategies rely on maximizing transparency and traceability in agentic AI systems to facilitate debugging and continuous improvement.

-----

-----

-----

### Source [15]: https://neon.com/blog/the-3-levels-of-debugging-with-ai

Query: What are common failure modes and debugging strategies for a ReAct agent built from scratch?

Answer: **Common Failure Modes:**
- **Semantic Misunderstanding:** AI agents may misinterpret instructions or context, leading to errors in execution or reasoning, especially when the agent’s logic is complex or data-driven.
- **Improper Data Handling:** Errors can arise when the agent processes undefined or uninitialized variables, causing failures in downstream reasoning or actions.

**Debugging Strategies:**
- **Error Message Analysis:** Directly inputting error messages or failure outputs into a large language model (LLM) can yield explanations and likely root causes. For example, an LLM can infer if an error is due to mapping over an undefined variable.
- **Component and Line Tracing:** Examine the specific agent component and execution line where the failure occurs, focusing on the state and initialization of variables at those points.
- **Iterative Prompting:** Use LLMs interactively to explore the agent’s reasoning, propose likely causes, and suggest concrete code or logic changes to resolve issues.

This approach leverages LLMs as interactive debugging agents, capable of contextualizing errors and guiding developers through fixing them based on the agent's observed execution path.

-----

-----

</details>

<details>
<summary>How can a mock tool in a ReAct agent be designed to be easily replaceable with real-world APIs like Google Search?</summary>

### Source [16]: https://blog.motleycrew.ai/blog/reliable-ai-at-your-fingertips-how-we-built-universal-react-agents-that-just-work

Query: How can a mock tool in a ReAct agent be designed to be easily replaceable with real-world APIs like Google Search?

Answer: To make a mock tool in a ReAct agent easily replaceable with real-world APIs like Google Search, the agent should be designed for **universality and modularity**. The process described involves initially using a standard ReAct prompt that lists available tools and describes their usage, but issues with reliability and reasoning were encountered. The solution was to separate reasoning and acting into distinct steps, ensuring the agent reasons before selecting a tool. 

This approach means that each tool is described by its capabilities and interface in the prompt, allowing the large language model (LLM) to understand what each tool can do without implementation details. When swapping out a mock tool for a real API, as long as the interface (input/output structure and tool description) remains consistent, the agent does not require any changes in reasoning or prompt logic. This modular and interface-driven design is critical for making tool replacement seamless and reliable[1].

-----

-----

-----

### Source [17]: https://www.promptingguide.ai/techniques/react

Query: How can a mock tool in a ReAct agent be designed to be easily replaceable with real-world APIs like Google Search?

Answer: Using the LangChain framework, a mock tool within a ReAct agent can be replaced with a real-world API, such as Google Search, by **adhering to a standardized tool interface**. In LangChain, tools are defined as callable Python functions or classes with a specific input/output schema. The agent is initialized with a list of tools (either mock or real), and the LLM uses these through a clear, documented API.

This separation enables an easy swap: to replace a mock search tool with Google Search, you simply substitute the mock implementation with a function/class that calls the Google Search API, while keeping the tool name, input, and output formats the same. The agent’s reasoning and control flow remain unchanged because they interact with the tool interface, not the implementation. This modular tooling pattern is central to LangChain’s ReAct agent design and is essential for maintainability and scalability[2].

-----

-----

-----

### Source [18]: https://technofile.substack.com/p/how-to-build-a-react-ai-agent-with

Query: How can a mock tool in a ReAct agent be designed to be easily replaceable with real-world APIs like Google Search?

Answer: A mock tool in a ReAct agent is invoked by pattern-matching a specific action string (e.g., `Action: search: query`). The Python implementation exemplifies this by extracting the tool name and input from the agent’s output using regular expressions. Known actions are registered in advance, and when an action is detected, the corresponding function is called.

To make tools easily replaceable, the code structure uses a **mapping of action names to functions**. Swapping a mock function for a real-world API (such as Google Search) is a matter of updating the mapping for that action name to reference the new function. As long as the function signature and expected output remain consistent, the agent logic and prompt do not require changes. This decoupled, dictionary-driven approach ensures that tools can be substituted or upgraded with minimal disruption to the agent’s workflow[3].

-----

-----

-----

### Source [19]: https://www.youtube.com/watch?v=Lvrv9I276ps

Query: How can a mock tool in a ReAct agent be designed to be easily replaceable with real-world APIs like Google Search?

Answer: Within the LangChain ecosystem, the video demonstrates creating a custom tool for a ReAct agent by **implementing each tool as a callable unit with a predictable interface**. The agent is constructed by supplying a list of such tools, each registered by name. When the agent "acts," it references the tool by name and passes input as defined by the interface.

To replace a mock tool with a real-world API, you only need to switch out the callable associated with the tool name while maintaining the same input/output contract. The rest of the agent’s logic, prompt, and execution process remain untouched because of this consistent interface-driven design. This method allows for rapid prototyping (with mock tools) and straightforward productionization (by swapping in real APIs) without altering the agent's reasoning or orchestration code[4].

-----

-----

-----

### Source [20]: https://maven.com/rakeshgohel/ai-agent-engineering-react-rag-multi-agent

Query: How can a mock tool in a ReAct agent be designed to be easily replaceable with real-world APIs like Google Search?

Answer: Building agentic systems with ReAct benefits from **software engineering best practices** such as modularity and interface abstraction. Production-ready starter kits and frameworks emphasize structuring tools as modular, interchangeable components. Each tool is encapsulated, and the agent interacts with it via a defined interface.

To replace a mock tool with a real API like Google Search, you implement the API call in a module that conforms to the same interface as the mock. Unit testing and debugging are facilitated because the interface remains unchanged, and version control systems track changes to individual tools, not the agent as a whole. This approach supports maintainability, scalability, and the ability to develop and test with mocks before deploying real-world integrations[5].

-----

-----

</details>

<details>
<summary>What is the minimal but effective prompt structure for the "Thought" generation phase in a ReAct agent?</summary>

### Source [21]: https://shafiqulai.github.io/blogs/blog_3.html

Query: What is the minimal but effective prompt structure for the "Thought" generation phase in a ReAct agent?

Answer: The **Thought** generation phase in a ReAct agent is where the agent processes the user's input, breaks down complex queries into smaller steps, and determines the best immediate course of action. The minimal but effective prompt structure for this phase should:
- Direct the agent to **read and understand** the user's query.
- Encourage **breaking down** complex tasks into logical, manageable steps.
- Guide the agent to **decide** which information or tool is necessary for the next step.

An example of minimal prompt structure is:
- “Thought: What do I need to know or do next to answer the user’s question?”

This structure ensures the agent does not jump to conclusions, but instead reasons explicitly about each requirement, retrieving and verifying information step by step, which enhances accuracy and completeness.

-----

-----

-----

### Source [22]: https://www.wordware.ai/blog/why-the-react-agent-matters-how-ai-can-now-reason-and-act

Query: What is the minimal but effective prompt structure for the "Thought" generation phase in a ReAct agent?

Answer: In the ReAct framework, the **Thought** phase is initiated as soon as the agent receives an input question. The language model is prompted to:
- **Interpret the prompt** to understand the core requirements.
- **Break down the question** into actionable parts.
- **Formulate an initial action plan** and identify what additional data or steps might be needed.

A minimal but effective prompt structure for this phase would be:
- “Thought: Analyze the question and identify key requirements and next steps.”

This approach prompts the agent to generate clear **reasoning traces**, which are essential for transparency, iterative improvement, and effective problem decomposition before any actions are taken.

-----

-----

-----

### Source [23]: https://www.promptingguide.ai/techniques/react

Query: What is the minimal but effective prompt structure for the "Thought" generation phase in a ReAct agent?

Answer: The ReAct prompting paradigm requires the agent to generate a **verbal reasoning trace** before taking action. The prompt structure for the Thought phase usually follows a simple yet effective template:
- “Thought: [Reasoning about what needs to be done next]”
- “Action: [Action or tool use based on the above thought]”
- “Observation: [Result from the environment/tool]”

For the Thought phase specifically, the prompt should simply instruct the model to “think aloud”:
- “Thought: What is the next logical step or piece of information needed?”

This minimal structure is effective because it makes the agent explicitly state its reasoning, supporting dynamic problem-solving and reducing the risk of skipping important steps.

-----

-----

-----

### Source [24]: https://xaibo.ai/how-to/orchestrator/customize-react-prompts/

Query: What is the minimal but effective prompt structure for the "Thought" generation phase in a ReAct agent?

Answer: According to the Xaibo ReActOrchestrator documentation, the **Thought** prompt is defined in the configuration as:
- `thought_prompt: "Generate thoughts about next steps"`

In practice, the system prompt instructs:
- “Always start with THOUGHT to analyze the user's request”
- “THOUGHT: Reason about what information you need”

The minimal but effective prompt for the Thought phase is thus:
- “THOUGHT: Reason about what information you need.”

This approach ensures the agent begins every cycle by considering what is required for progress, establishing a clear, concise reasoning step at the start of each loop.

-----

-----

-----

### Source [25]: https://arize.com/docs/phoenix/cookbook/prompt-engineering/react-prompting

Query: What is the minimal but effective prompt structure for the "Thought" generation phase in a ReAct agent?

Answer: In the Phoenix ReAct prompt engineering guide, the **Thought** phase is described as the model reasoning step-by-step before deciding on an action. The minimal prompt structure is:
- Instruct the model to “reason through the problem first.”
- Example: “Thought: What needs to be considered or found out before acting?”

By prompting the model to articulate its thought process before proceeding to action, this structure promotes transparency, multi-step problem-solving, and better traceability of the agent’s decisions.

-----

-----

</details>

<details>
<summary>What are the best practices for structuring the "scratchpad" or message history in a ReAct agent to effectively manage the thought-action-observation loop?</summary>

### Source [26]: https://geekyants.com/blog/implementing-ai-agents-from-scratch-using-langchain-and-openai

Query: What are the best practices for structuring the "scratchpad" or message history in a ReAct agent to effectively manage the thought-action-observation loop?

Answer: The agent scratchpad in a ReAct agent is structured within the agent prompt template, which defines the LLM's behavior and interaction with tools. Best practices for structuring the scratchpad include:
- **Clear System Instructions**: Begin with a section that explicitly defines the agent's role, for example, specifying that the agent answers questions using available tools, performs reasoning, fetches information, and carries out calculations.
- **Tool Descriptions**: List each tool with its name and a clear description, so the model understands its capabilities. Each tool section typically includes a usage guideline and a format example, such as specifying how to structure an action call (e.g., `Action: Calculator` and `Action Input: 2 + 2`).
- **Format Guide**: Specify how the agent should structure its output, often in a structured format like JSON or markdown. This ensures consistent communication between the thought process, tool invocation, and observation integration.
- **Guardrails**: Include explicit behavioral constraints, such as only calling tools when necessary, not fabricating information, and never executing code directly.
- **Iterative State Tracking**: The scratchpad logs the sequence of thoughts, actions, and observations. Each cycle is captured, enabling the agent to reference past steps and build coherent multi-step reasoning.
- **Customization for Robustness**: The prompt, including the scratchpad format, is highly customizable based on the agent's tasks. Using libraries like Pydantic for structured outputs is recommended for precision, especially in JSON-based tool calls.

This approach ensures the agent maintains transparency in its reasoning and actions, effectively manages intermediate state, and can be easily extended or debugged.

-----

-----

-----

### Source [27]: https://airbyte.com/data-engineering-resources/using-langchain-react-agents

Query: What are the best practices for structuring the "scratchpad" or message history in a ReAct agent to effectively manage the thought-action-observation loop?

Answer: LangChain implements the ReAct pattern by maintaining a running log called the agent scratchpad, which records all thoughts, actions, and observations throughout the reasoning process. Key best practices include:
- **Systematic Logging**: The scratchpad consistently records each cycle—starting with the agent’s thought, the action it takes (such as calling a tool), the action input, and the resulting observation. This iterative log provides a transparent and traceable history of the agent’s decision-making.
- **Support for Iterative Reasoning**: The scratchpad structure allows the agent to recognize when additional information is needed, enabling it to return to the thought-action-observation loop as many times as necessary before producing a final answer.
- **Separation of Reasoning and Action**: By alternating between “thought” (reasoning step) and “action” (tool invocation), the scratchpad enforces a clear boundary between internal deliberation and external interaction, improving both interpretability and modularity.
- **Finalization Step**: After sufficient iterations, the agent can provide a comprehensive final answer, referencing the full context preserved in the scratchpad.

This method supports complex, multi-step queries and makes it easy to debug or audit the agent's process by reviewing the detailed message history.

-----

-----

-----

### Source [28]: https://arize.com/blog-course/react-agent-llm/

Query: What are the best practices for structuring the "scratchpad" or message history in a ReAct agent to effectively manage the thought-action-observation loop?

Answer: In practical ReAct agent implementations, the scratchpad is built directly into the prompt template and explicitly guides the agent through the thought-action-observation sequence. The recommended structure includes:
- **Prompt Template Variables**: The prompt template defines a sequence where each step is explicitly marked:
    - `Thought:` (agent's reasoning)
    - `Action:` (name of the tool or action)
    - `Action Input:` (parameters or input for the tool)
    - `Observation:` (the result or output from the tool)
- **Iterative Expansion**: The pattern `(Thought / Action / Action Input / Observation)` can repeat as many times as needed, enabling the agent to chain multiple tool invocations and observations.
- **Final Answer Declaration**: After iterating, the template includes
    - `Thought: I now know the final answer`
    - `Final Answer:` (the agent's answer to the original input)
- **Prompt Continuity**: The variable `{agent_scratchpad}` is appended at the end of the prompt, ensuring that the agent receives its entire message history with every new step, maintaining context and continuity.
- **Customization**: The prompt template can be tailored for specific tasks, and the structure of the scratchpad should match the requirements of the agent and domain.

This format ensures that the agent’s reasoning and tool interactions are both explicit and auditable, supporting robust multi-step reasoning.

-----

-----

-----

### Source [29]: https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/

Query: What are the best practices for structuring the "scratchpad" or message history in a ReAct agent to effectively manage the thought-action-observation loop?

Answer: In LangGraph, the recommended scratchpad structure for a ReAct agent is to maintain the message history as a list of message objects within the agent's state. Key implementation points include:
- **Graph State as Message Sequence**: The agent’s state includes a `messages` list, where each entry is a message object representing either a user input, an agent thought, an action, or an observation.
- **Reducer Functionality**: The message list is updated using a reducer (such as `add_messages`), ensuring that every new message (thought, action, observation, or response) is appended to the state in order.
- **Extensibility**: The state can be extended with additional keys as needed for different use cases, but the core principle is to preserve the complete chronological message history.
- **Tool Integration**: Tools are bound to the model and invoked as needed, with their calls and results logged as message objects, preserving transparency and reproducibility in the agent’s operations.

This design aligns with best practices for modularity, extensibility, and traceability, making it easy to follow and audit the agent’s reasoning and actions at every step.

-----

-----

</details>

<details>
<summary>How does Gemini's function calling mechanism simplify the "Action" phase of a ReAct agent compared to traditional text-parsing approaches?</summary>

### Source [30]: https://ai.google.dev/gemini-api/docs/function-calling

Query: How does Gemini's function calling mechanism simplify the "Action" phase of a ReAct agent compared to traditional text-parsing approaches?

Answer: **Gemini's function calling mechanism** allows developers to define functions (using OpenAPI-compatible schemas) and provide these definitions to the Gemini model along with the user prompt. When the model determines that a function call is needed to fulfill the user's request, it outputs a structured `functionCall` object specifying which function to call and with what parameters, rather than generating a text instruction to be parsed. This replaces the need for the agent or developer to parse text output to extract actions and parameters, as is required in traditional ReAct or text-based approaches. The structured response makes the "Action" phase more reliable and programmatically accessible, eliminating ambiguity and reducing the risk of parsing errors. Additionally, Gemini supports parallel function calls, enabling efficient handling of multi-part requests. However, only a subset of the OpenAPI schema is supported, and some limitations exist regarding parameter types and SDK availability[1].

-----

-----

-----

### Source [31]: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling

Query: How does Gemini's function calling mechanism simplify the "Action" phase of a ReAct agent compared to traditional text-parsing approaches?

Answer: **Function calling in Gemini** (also called "tool use") lets the model decide—based on the prompt—if an external function is needed and, if so, it outputs structured data specifying the function to call and its parameters, such as `get_current_weather(location='Boston')`. This structure means the "Action" phase for a ReAct agent no longer requires parsing free-form text to determine which tool to invoke or how to invoke it. Instead, the agent directly receives clear, machine-readable instructions. This greatly simplifies orchestration, reduces the chance of misinterpretation, and makes chaining tool calls or parallel actions straightforward. The process bridges the LLM and external systems, extending the agent’s capabilities to fetch data or perform operations seamlessly. OpenAPI schema declarations allow for up to 512 functions, and best practices are provided for defining these actions[2].

-----

-----

-----

### Source [32]: https://www.philschmid.de/langgraph-gemini-2-5-react-agent

Query: How does Gemini's function calling mechanism simplify the "Action" phase of a ReAct agent compared to traditional text-parsing approaches?

Answer: Traditional ReAct agents operate by prompting the LLM to output action instructions as text, typically following a "Thought, Action, Observation" pattern. For the "Action" phase, the LLM describes (in text) what action to take—such as calling a tool or querying an API. The developer or agent framework must then parse this output text to extract the function name and parameters, which introduces ambiguity and potential errors if the text format changes or is misinterpreted. This text-parsing approach requires robust, often brittle parsing logic and careful prompt engineering to keep outputs machine-readable[3].

-----

-----

-----

### Source [33]: https://www.leewayhertz.com/react-agents-vs-function-calling-agents/

Query: How does Gemini's function calling mechanism simplify the "Action" phase of a ReAct agent compared to traditional text-parsing approaches?

Answer: In a classic ReAct agent, the LLM’s output for the "Action" phase is a natural language description of what tool to use and how, which must be interpreted and parsed by the system to determine the intended action. This iterative approach is flexible and allows dynamic planning, but places the burden of reliable parsing on the agent developer. Parsing natural language is error-prone, especially as tasks grow more complex or LLM output varies. In contrast, function calling agents (like those using Gemini's mechanism) receive structured outputs directly from the model, specifying tool names and parameters as data, not text. This removes ambiguity and parsing complexity, making the "Action" phase simpler, more robust, and easier to scale[4].

-----

-----

-----

### Source [34]: https://www.youtube.com/watch?v=SC-Y_o_fkbY

Query: How does Gemini's function calling mechanism simplify the "Action" phase of a ReAct agent compared to traditional text-parsing approaches?

Answer: The video tutorial demonstrates that with Gemini’s function calling, developers provide a set of existing functions to the model. When the model determines a function should be called, it outputs a structured response (not text), directly indicating which function to execute and with which parameters. This contrasts with earlier approaches where the model would generate textual commands that required parsing to extract actions and parameters. Gemini’s structured function call output makes it straightforward for agents to trigger actions, reducing opportunities for error and simplifying integration with external systems[5].

-----

-----

</details>

<details>
<summary>What are the most common errors or failure modes when implementing a ReAct agent's control loop from scratch, and what are the best debugging strategies?</summary>

### Source [35]: https://www.neradot.com/post/building-a-python-react-agent-class-a-step-by-step-guide

Query: What are the most common errors or failure modes when implementing a ReAct agent's control loop from scratch, and what are the best debugging strategies?

Answer: Common errors when implementing a ReAct agent’s control loop from scratch include:

- **Improper initialization**: Failing to set up agent state correctly (e.g., not initializing `intermediate_steps`, `start_time`, or setting `is_started` properly) can cause later steps to fail or behave unpredictably.
- **State mutation issues**: Not appending or managing steps in the reasoning/action loop properly, such as missing calls to `add_intermediate_step`, can result in incomplete reasoning traces or lost data.
- **Verbosity and debugging**: Without proper use of the `verbose` flag for logging intermediate steps, tracking agent progress and identifying issues is difficult.

Best debugging strategies:
- **Verbose logging**: Enable the `verbose` flag to print each intermediate step and all agent actions. Review printed output during execution to spot logical errors or missed steps.
- **Test with simple queries**: Start with basic inputs (e.g., “What is 2 + 2?”) to verify that the control loop runs and state updates as expected before handling complex tasks.
- **Incremental development**: Build and test each method (e.g., `start`, `add_intermediate_step`) individually to isolate problems early.

These strategies help ensure that the agent’s control loop operates as designed and errors in state management or execution are caught quickly[1].

-----

-----

-----

### Source [36]: https://docs.getdynamiq.ai/low-code-builder/llm-agents/guide-to-implementing-llm-agents-react-and-simple-agents

Query: What are the most common errors or failure modes when implementing a ReAct agent's control loop from scratch, and what are the best debugging strategies?

Answer: Common errors and failure modes:

- **Tool misconfiguration**: Incorrect setup of external tools (e.g., ScaleSerp, ZenRows) may result in failed or inaccurate data retrieval.
- **Improper loop termination**: Not setting a reasonable `max_loops` or lacking error handling can cause infinite loops or premature agent termination.
- **Role ambiguity**: Unclear agent roles and behavioral guidelines may lead to inconsistent decision-making in the control loop.
- **Error handling gaps**: Failure to configure robust error handling for tool failures, API timeouts, or data parsing errors can break the loop or produce misleading outputs.

Best debugging strategies:
- **Test with a series of queries** to evaluate agent responses, loop behavior, and data handling.
- **Set practical `max_loops` limits** to prevent runaway execution and aid in diagnosing termination logic.
- **Define clear roles and guidelines** for the agent to improve decision consistency and make debugging more straightforward.
- **Configure comprehensive error handling** for both loop termination and tool usage, capturing exceptions and logging errors for review[2].

-----

-----

-----

### Source [37]: https://dylancastillo.co/posts/react-agent-langgraph.html

Query: What are the most common errors or failure modes when implementing a ReAct agent's control loop from scratch, and what are the best debugging strategies?

Answer: Common failure modes:

- **Incorrect routing logic**: Errors in the `should_continue` control function can cause the agent to skip necessary tool invocations or terminate the loop prematurely.
- **State update inconsistencies**: Mistakes in updating `MessagesState` after each step may lead to lost messages or improper sequencing, breaking the reasoning/action trace.
- **Edge misconfiguration**: Incorrectly connecting nodes and edges in the agent’s graph (e.g., missing conditional transitions between LLM and tool nodes) can interrupt the control loop.

Best debugging strategies:
- **Inspect control logic** (e.g., `should_continue`): Carefully verify conditions for tool calls and loop termination.
- **Trace state transitions**: Print or log the updated `MessagesState` after each step to confirm that messages and actions are stored and sequenced correctly.
- **Validate graph connections**: Review agent graph structure to ensure all nodes and edges are correctly defined and transitions match expected control flow[3].

-----

-----

-----

### Source [38]: https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/

Query: What are the most common errors or failure modes when implementing a ReAct agent's control loop from scratch, and what are the best debugging strategies?

Answer: Common errors and failure modes:

- **State mismanagement**: Failing to properly define and update the agent’s state (e.g., the list of messages) can result in incomplete or incorrect reasoning/action cycles.
- **Tool interface errors**: Incorrectly implementing or binding tools (such as not matching expected input/output formats) may prevent successful tool calls.
- **Model-tool integration issues**: Problems in binding tools to the model, such as mismatched tool names or signatures, can break the agent’s action step.

Best debugging strategies:
- **Test state updates**: Use reducers (like `add_messages`) and confirm that the agent’s state reflects all intended messages and actions after each step.
- **Validate tool calls**: Implement mock tools for initial debugging and verify that the agent calls tools correctly and receives expected outputs.
- **Check model-tool binding**: Ensure tools are properly registered with the model and accessible during action selection[4].

-----

-----

-----

### Source [39]: https://docs.nvidia.com/aiqtoolkit/latest/workflows/about/react-agent.html

Query: What are the most common errors or failure modes when implementing a ReAct agent's control loop from scratch, and what are the best debugging strategies?

Answer: Common errors and failure modes:

- **Skipping reasoning steps**: Not alternating properly between reasoning and acting can cause the agent to jump to actions without sufficient thought, reducing answer quality.
- **Action/observation mismatch**: Failing to correctly process tool outputs or misinterpreting feedback can break the iterative loop and lead to irrelevant answers.
- **Loop repetition errors**: Not handling repeated reasoning-action cycles may result in incomplete answers or missed opportunities to refine responses.

Best debugging strategies:
- **Walk through each iteration**: Manually trace the agent’s reasoning and action steps for sample queries to ensure the loop alternates correctly and feedback is incorporated.
- **Monitor tool responses**: Verify that tool outputs are correctly received and summarized by the agent.
- **Test with varied input problems**: Use diverse queries to check the agent’s ability to reason, act, and refine answers across multiple cycles[5].
-----

-----

</details>

<details>
<summary>What are the essential components of a prompt template for the "Thought" generation step in a ReAct agent, and how should it be structured to include tool descriptions and conversation history?</summary>

### Source [40]: https://www.mercity.ai/blog-post/react-prompting-and-react-based-agentic-systems

Query: What are the essential components of a prompt template for the "Thought" generation step in a ReAct agent, and how should it be structured to include tool descriptions and conversation history?

Answer: The essential components of a prompt template for the "Thought" generation step in a ReAct agent include a clear structure and explicit instructions to guide the language model through a **Thought/Action/Observation** cycle. The template must instruct the model to reason step-by-step about the current situation (Thought), specify an action (Action), and observe the result (Observation). The template should also include examples that show multiple cycles of this process, which illustrate how the agent should alternate between generating thoughts, performing actions, and observing outcomes.

A typical prompt template should include:
- **User Query**: The initial question or task.
- **Example Trajectories**: Concrete, task-specific examples demonstrating several iterations of the Thought/Action/Observation cycle, culminating in a Final Answer. This shows the agent precisely how to reason and act.
- **Explicit Format Instructions**: Clear formatting instructions such as:
  - Thought: [agent's reasoning]
  - Action: [the specific action to take]
  - Observation: [result of the action]
  - Repeat cycle as needed.
  - Final Answer: [final response]
- This structure supports the agent in maintaining a logical flow and encourages step-by-step reasoning.

Including **tool descriptions** and the **conversation history** is implied through the context provided in the examples—where each action can represent a tool call, and previous thoughts, actions, and observations serve as the conversation history. The agent is expected to reference these prior steps as it continues reasoning, ensuring coherence and relevance in multi-step tasks.

The prompt ends with instructions to always output in this specific format, reinforcing the cyclical structure critical for ReAct agents[1].

-----

-----

-----

### Source [41]: https://arize.com/blog-course/react-agent-llm/

Query: What are the essential components of a prompt template for the "Thought" generation step in a ReAct agent, and how should it be structured to include tool descriptions and conversation history?

Answer: The prompt template for a ReAct agent must provide:
- **Descriptions and purposes of each tool** available to the agent, along with the specific input format required to trigger each tool.
- An explicit **Thought/Action/Action Input/Observation** structure that can repeat for multiple steps as needed.
- The inclusion of **{agent_scratchpad}** at the end of the prompt. This variable represents the running conversation history, allowing the model to reference all prior thoughts, actions, and observations, ensuring continuity and informed subsequent reasoning.
- **Task-specific instructions** and, if desired, stylistic or output constraints (e.g., respond in a specific tone or with certain keywords).
- The user's question ({input}) is included at the end, paired with the agent_scratchpad, so the agent always has access to both the original query and the evolving reasoning context.

This structure ensures the agent can:
- Leverage tool descriptions to select and use tools appropriately.
- Use conversation history (via agent_scratchpad) to inform ongoing reasoning and avoid repetition or contradictions.
- Produce outputs in a consistent, expected format for both reasoning and action steps[2].

-----

-----

-----

### Source [42]: https://www.width.ai/post/react-prompting

Query: What are the essential components of a prompt template for the "Thought" generation step in a ReAct agent, and how should it be structured to include tool descriptions and conversation history?

Answer: The key components for a ReAct prompt template, especially for the "Thought" step, are:
- Inclusion of **task-specific in-context examples** that demonstrate the alternating cycle of thought, action, and observation. These few-shot examples are crucial for teaching the model the expected reasoning process and output structure.
- Each example clearly labels:
  - Thought: The agent's reasoning at that point.
  - Action: The specific operation or tool usage.
  - Observation: The result or data returned from the action.
- The action step in examples should use specific, context-appropriate values, illustrating how the agent should translate generic instructions into concrete actions based on current input.
- The number and content of examples should be determined experimentally, but only a handful (3-6) are typically needed to define expectations.

While the content does not explicitly focus on tool descriptions, it implies that the examples themselves should demonstrate how to use available tools, and the history of thoughts/actions/observations forms the conversation context that the agent references in subsequent steps[3].

-----

-----

-----

### Source [43]: https://www.promptingguide.ai/techniques/react

Query: What are the essential components of a prompt template for the "Thought" generation step in a ReAct agent, and how should it be structured to include tool descriptions and conversation history?

Answer: A ReAct prompt template is organized to generate **verbal reasoning traces** ("Thought") and **actions** for a given task, leveraging both reasoning and acting steps. The essential components include:
- A **user question** as the initial input.
- A sequence of alternating "Thought" (reasoning) and "Action" steps, with "Observation" reflecting the real-world or external feedback received from each action.
- Incorporation of **in-context examples** (few-shot learning) to demonstrate the expected process, though not shown in detail in the example provided.
- The template effectively combines both reasoning (planning/deciding what to do next) and acting (interacting with tools or knowledge sources).

The **conversation history** is maintained by including all prior thoughts, actions, and observations as context for the next step, allowing the agent to build upon what has already been deduced or discovered.

While explicit instructions about tool descriptions are not given, the structure supports interaction with external tools or APIs, and the examples can be tailored to include tool usage as part of the action steps[4].

-----

</details>

<details>
<summary>What are effective strategies for designing a mock tool for a ReAct agent that is both simple for a tutorial and easily replaceable with a real-world API?</summary>

### Source [44]: https://docs.getdynamiq.ai/low-code-builder/llm-agents/guide-to-implementing-llm-agents-react-and-simple-agents

Query: What are effective strategies for designing a mock tool for a ReAct agent that is both simple for a tutorial and easily replaceable with a real-world API?

Answer: For a ReAct agent, the tool integration process is modular and designed to be easily swappable, which is ideal for mock tool development in tutorials. The configuration steps include:

- **Name**: Assign a unique identifier to distinguish the agent.
- **Tools**: Integrate external services (mock or real APIs), such as search engines or data scrapers. These are modular, so you can start with a simple mock implementation and later replace it with a real-world API.
- **Prompt Template**: Customize the agent’s prompt to describe tool functionality, input formats, and expected outputs for clarity.
- **Max Loop & Behavior**: Set the maximum number of reasoning-action cycles and define the agent's response when the loop limit is reached, which is useful for controlling tutorial complexity.
- **Inference Mode**: Use a standardized response format (XML is recommended) to facilitate consistent tool output parsing, making the switch from a mock to real tool seamless.
- **Streaming**: Optionally enable incremental response generation to demonstrate real-world interaction patterns.
- **Tool Parameters**: Allow specification of tool parameters in responses, ensuring the interface remains consistent across mock and production tools.

By following these steps and leveraging the modularity in tool integration, you ensure that your mock tool is simple for demonstration but can be easily swapped for a real API by updating the tool configuration, not the agent logic.

-----

-----

-----

### Source [45]: https://dev.to/jamesli/react-vs-plan-and-execute-a-practical-comparison-of-llm-agent-patterns-4gh9

Query: What are effective strategies for designing a mock tool for a ReAct agent that is both simple for a tutorial and easily replaceable with a real-world API?

Answer: The ReAct agent pattern employs a reasoning-action loop, which iteratively selects and calls tools based on the agent’s reasoning process. When designing a mock tool for tutorials:

- **Code Modularity**: Tools are implemented as independent, callable functions or classes. This makes it straightforward to substitute a mock tool with a real-world API by keeping the interface (method names, parameters, and outputs) unchanged.
- **Prompt Engineering**: The ReAct pattern relies heavily on well-designed prompt templates. Mock tools should be described clearly in the prompt, with explicit input-output expectations, so that switching to a real API only requires updating the backend logic, not the user-facing interface or prompt.
- **Testing and Analysis**: The modular structure allows for easy benchmarking of performance and accuracy between mock and real tools, ensuring the tutorial remains relevant as the system evolves.

This approach ensures that ReAct agents can be taught with simple, mock tools that mirror the interface of real systems, making later replacement with production APIs trivial.

-----

-----

-----

### Source [46]: https://arize.com/blog-course/react-agent-llm/

Query: What are effective strategies for designing a mock tool for a ReAct agent that is both simple for a tutorial and easily replaceable with a real-world API?

Answer: When implementing a custom ReAct agent—especially for tutorials—most of the complexity lies in the prompt template and tool interface design:

- **Tool List Definition**: Define tools as a list of callable entities (e.g., search or lookup functions). These can be implemented as simple mock functions for a tutorial, then swapped for real APIs later without changing the agent logic.
- **Prompt Customization**: The prompt should include a description and purpose of each tool, the input format, and how the tool should be invoked. This clarity ensures that the agent (and users) interact with the mock tool in the same way as they would with a real API.
- **Agent Memory**: Retain explicit asks and responses in memory (e.g., using an agent scratchpad), which supports step-by-step reasoning and is agnostic to whether the tool is mock or real.
- **Iterative Design**: Customize prompt templates and tool outputs by running examples and refining as needed, making it easy to identify and fix mismatches before integrating a real-world API.

These strategies ensure the mock tool is both simple for learning and structured for easy replacement.

-----

-----

-----

### Source [47]: https://docs.nvidia.com/aiqtoolkit/latest/workflows/about/react-agent.html

Query: What are effective strategies for designing a mock tool for a ReAct agent that is both simple for a tutorial and easily replaceable with a real-world API?

Answer: The NVIDIA Agent Intelligence Toolkit provides a flexible ReAct agent system emphasizing ease of tool integration and replacement:

- **Pre-built Tools & Plugins**: The system supports the integration of both built-in and custom tools via a plugin system. For tutorials, you can implement a mock tool as a plugin, and later swap it for a real-world API by simply updating the plugin.
- **Customizable Prompts**: The agent’s prompt can be modified to specify tool names, descriptions, and usage, ensuring clarity for both mock and real tools.
- **Configurable Workflows**: Agent workflows are fully configurable (e.g., via YAML), allowing developers to switch between mock and real tools without altering the agent’s core logic.
- **Developer Experience**: The toolkit is designed for rapid prototyping and deployment, facilitating fast iteration on mock tools for tutorials and straightforward migration to production APIs.

This modular, plugin-based approach ensures mock tools are easy to implement, demonstrate, and later replace.

-----

-----

-----

### Source [48]: https://www.anthropic.com/research/building-effective-agents

Query: What are effective strategies for designing a mock tool for a ReAct agent that is both simple for a tutorial and easily replaceable with a real-world API?

Answer: Anthropic emphasizes the importance of designing agent-computer interfaces (ACI) with the same care as human-computer interfaces:

- **Clarity in Tool Definition**: Provide clear tool descriptions, example usage, input format requirements, and boundaries between tools. This makes mock tools easy to use and later replace because their interface is unambiguous.
- **Parameter Naming & Documentation**: Name parameters and describe them clearly, as if writing documentation for a junior developer. This reduces errors and simplifies the transition from mock to real-world APIs.
- **Test and Iterate**: Run extensive tests with sample inputs to identify and correct model errors, iterating on the tool design until the agent consistently uses tools correctly. This process ensures that both mock and real tools are robustly integrated.
- **Poka-yoke (Error-proofing)**: Structure tool arguments to minimize mistakes. For example, require absolute file paths if relative paths cause errors. Such constraints apply equally to mock and real tools, making replacement seamless.

These practices ensure that a mock tool is not only simple for tutorials but also structured to be easily swapped for a production API with minimal risk of error.

-----

</details>

<details>
<summary>What are the fundamental principles of the ReAct (Reason+Act) framework and how does it create synergy between reasoning and acting in large language models?</summary>

### Source [49]: https://dev.to/rijultp/react-reason-act-a-smarter-way-for-language-models-to-think-and-do-344o

Query: What are the fundamental principles of the ReAct (Reason+Act) framework and how does it create synergy between reasoning and acting in large language models?

Answer: The ReAct (Reason + Act) framework is designed to enhance large language models (LLMs) by structuring their approach to complex tasks around two alternating steps: **Reason** and **Act**. In the Reason step, the model articulates its current understanding, relevant facts, and what it intends to do next. In the Act step, the model performs an action based on its reasoning, such as calling an API, executing a command, or retrieving information. The result of each action is looped back as new input, prompting the model to reason again, and the process repeats until the task is complete.

This structured alternation provides several benefits:
- **Transparency:** Every step is explicit, allowing users to follow the model's thought process, which facilitates debugging and performance improvement.
- **Efficiency:** By separating reasoning from action, the model is less likely to take unnecessary steps, staying focused on the task.
- **Control:** Developers can monitor and intervene at each step, refining reasoning or actions as needed.
- **Synergy:** The loop leverages LLMs’ strengths in both reasoning and action selection, enabling them to solve more complex, multi-step tasks.

-----

-----

-----

### Source [50]: https://learnprompting.org/docs/agents/react

Query: What are the fundamental principles of the ReAct (Reason+Act) framework and how does it create synergy between reasoning and acting in large language models?

Answer: The ReAct paradigm enables LLMs to solve complex tasks by combining **natural language reasoning** with **actions** such as retrieving external information. ReAct extends Modular Reasoning, Knowledge, and Language (MRKL) systems by allowing models to reason about the actions they can perform.

A typical ReAct loop involves:
- The model generating a reasoning step (a "thought").
- The model performing an action (e.g., querying a search engine).
- The environment providing an observation in response to the action.
- The model reasoning again based on new information, continuing the loop until the task is solved.

This iterative thought-action cycle is likened to the reinforcement learning paradigm of state, action, and reward, formalizing the process of reasoning and acting. By integrating external information and reasoning steps, ReAct improves LLMs' ability to handle tasks requiring multiple, coordinated actions based on dynamic information.

-----

-----

-----

### Source [51]: https://tsmatz.wordpress.com/2023/03/07/react-with-openai-gpt-and-langchain/

Query: What are the fundamental principles of the ReAct (Reason+Act) framework and how does it create synergy between reasoning and acting in large language models?

Answer: ReAct (Reasoning + Acting) is a foundational framework for building agentic applications with LLMs, underpinning tools like Microsoft Copilot, ChatGPT plugins, and AutoGPT. The ReAct method decomposes complex tasks into simpler subtasks through iterative reasoning and acting.

The LLM alternates between:
- Reasoning to decide what external tool or action is needed.
- Pausing to receive the outcome (observation) from the action or tool.
- Updating its reasoning based on the accumulated observations and deciding the next action.

For instance, the LLM can issue "Search" or "Lookup" commands to external APIs, simulating human behaviors like web searches or document lookups. This loop continues until an answer is found or the model determines the task is impossible.

This architecture is sometimes called "Augmented Language Models" (ALMs), highlighting the synergy between intrinsic reasoning and extrinsic tool use, which enhances the LLM’s ability to solve real-world, multi-step problems.

-----

-----

-----

### Source [52]: https://react-lm.github.io

Query: What are the fundamental principles of the ReAct (Reason+Act) framework and how does it create synergy between reasoning and acting in large language models?

Answer: The ReAct approach combines the traditionally separate abilities of LLMs to reason (such as through chain-of-thought prompting) and to act (such as generating action plans for decision making). ReAct prompts consist of **interleaved reasoning traces, actions, and observations** from the environment.

**Key principles:**
- Reasoning traces help the model induce, track, and update action plans and manage exceptions.
- Actions enable the model to interface with external sources (like knowledge bases or APIs) to gather information.
- The synergy between reasoning and acting allows the model to dynamically adapt its plan based on new evidence, reducing hallucinations and error propagation.

ReAct has been applied successfully to tasks such as question answering and fact verification, where it demonstrated fewer errors and greater human interpretability compared to methods using only reasoning or acting. In interactive decision-making tasks, ReAct outperformed both imitation and reinforcement learning methods, achieving higher success rates with minimal in-context examples. The approach produces human-like, transparent task-solving trajectories, increasing interpretability and trustworthiness.

-----

</details>

<details>
<summary>How does the iterative, single-step reasoning of the ReAct agent architecture compare to the multi-step approach of Plan-and-Execute agents, particularly in terms of adaptability and error handling?</summary>

### Source [53]: https://www.dailydoseofds.com/ai-agents-crash-course-part-10-with-implementation/

Query: How does the iterative, single-step reasoning of the ReAct agent architecture compare to the multi-step approach of Plan-and-Execute agents, particularly in terms of adaptability and error handling?

Answer: The **ReAct agent architecture** operates through an iterative loop: Thought → Action → Observation, repeating until a solution is found. In each cycle, the agent analyzes the current context, reasons internally (the "Thought" step), performs an action (such as calling a tool), and then observes the result of that action before starting the next reasoning step. This single-step, iterative approach allows the agent to dynamically adjust its reasoning and actions in response to real-time feedback from the environment. Adaptability is thus high, as each new observation can immediately influence the next reasoning step. In terms of **error handling**, if an action produces unexpected results, the agent can incorporate this feedback and choose a new course of action in the next iteration. This contrasts with architectures that plan multiple steps ahead without intermediate feedback, which may be less responsive to errors that occur mid-plan.

-----

-----

-----

### Source [54]: https://arxiv.org/html/2404.11584v1

Query: How does the iterative, single-step reasoning of the ReAct agent architecture compare to the multi-step approach of Plan-and-Execute agents, particularly in terms of adaptability and error handling?

Answer: Single-agent models like **ReAct** and related architectures emphasize the importance of dedicated reasoning stages before each action. The architecture allows for **self-evaluation and self-correction** at each reasoning step. The paper observes that if agents lack the ability to self-correct or plan effectively at each step, they can become stuck in execution loops or fail to meet user expectations. For straightforward tasks requiring tool calls, this single-step, iterative approach is effective. However, when comparing to **multi-step Plan-and-Execute agents**, which create and follow more extensive plans before acting, the document highlights that adaptability and real-time adjustment may be more limited in such multi-step approaches, particularly if feedback is not incorporated between planning and execution phases. Therefore, ReAct's stepwise reasoning is especially advantageous for tasks where adaptability and error recovery are important.

-----

-----

-----

### Source [55]: https://aws.plainenglish.io/what-is-react-reasoning-pattern-how-it-makes-amazon-bedrock-agents-a-powerful-service-to-build-ai-c29e2d883d05

Query: How does the iterative, single-step reasoning of the ReAct agent architecture compare to the multi-step approach of Plan-and-Execute agents, particularly in terms of adaptability and error handling?

Answer: The **ReAct reasoning pattern** enables agents to break down problems into a sequence of reasoning and actions, interleaving them to allow step-by-step adaptation. In this model, after each action is taken, the agent evaluates the result and updates its reasoning trace accordingly. This **stepwise approach** is distinct from generating a full solution or plan at once, as is common with some Plan-and-Execute agents. The advantage for **adaptability** is clear: the agent can respond to unforeseen changes or errors as soon as they are detected, rather than waiting until the end of a multi-step plan. In terms of **error handling**, mistakes can be caught and addressed in near real-time, reducing the risk of propagating errors across multiple steps. By contrast, multi-step Plan-and-Execute agents may be less able to adjust mid-execution, especially if feedback mechanisms are not integrated into the execution phase.

-----

-----

-----

### Source [56]: https://www.promptingguide.ai/techniques/react

Query: How does the iterative, single-step reasoning of the ReAct agent architecture compare to the multi-step approach of Plan-and-Execute agents, particularly in terms of adaptability and error handling?

Answer: **ReAct** combines verbal reasoning and actions, prompting LLMs to generate a stepwise reasoning trace and take actions after each step. This allows for dynamic reasoning, where the agent creates, maintains, and adjusts its plan with each new observation from the environment. The architecture is explicitly designed to incorporate **external feedback at every step**, enabling the agent to update its knowledge and plans continually. This stands in contrast to pure chain-of-thought prompting (which may hallucinate or propagate errors due to static knowledge) and to multi-step plan-execute models, which may be less responsive to real-time changes. In **error handling**, the iterative structure of ReAct allows for continuous correction, as actions and observations are interleaved, making it well-suited for environments where adaptability is required.

-----

-----

-----

### Source [57]: https://www.dailydoseofds.com/p/intro-to-react-reasoning-and-action-agents/

Query: How does the iterative, single-step reasoning of the ReAct agent architecture compare to the multi-step approach of Plan-and-Execute agents, particularly in terms of adaptability and error handling?

Answer: **ReAct agents** integrate reasoning and actions, allowing LLMs not only to plan but also to interact with the external world in response to changing circumstances. In a practical workflow, the agent understands the query, selects appropriate tools, gathers information, and adapts its answers based on feedback—repeating this process as needed. This architecture is ideal for real-time adaptation and continuous decision-making, as each step in the process can be revised based on new information or errors encountered. This is a core distinction from **multi-step Plan-and-Execute agents**, which may formulate an entire plan before acting and therefore may be slower to respond to unexpected changes or errors detected during execution. The ReAct architecture inherently supports fine-grained adaptability and iterative error correction thanks to its single-step cycle structure.

-----

</details>

<details>
<summary>What is the specific role of the 'Observation' step in a ReAct agent's control loop, and how does processing external feedback from tools influence the agent's subsequent 'Thought' and 'Action' steps?</summary>

### Source [58]: https://www.dailydoseofds.com/ai-agents-crash-course-part-10-with-implementation/

Query: What is the specific role of the 'Observation' step in a ReAct agent's control loop, and how does processing external feedback from tools influence the agent's subsequent 'Thought' and 'Action' steps?

Answer: In the ReAct agentic pattern, the agent operates in a loop of **Thought → Action → Observation**. The 'Observation' step occurs after the agent takes an action, such as querying a tool or searching the internet. This step involves processing the output from the tool, which might include structured data, text snippets, or URLs provided in response to the agent's action.

The specific role of the 'Observation' step is to **integrate the external feedback or results from the tool back into the agent's reasoning loop**. For example, after issuing a search query, the agent observes the results and uses them as new context for its next reasoning cycle. This observed information directly influences the subsequent 'Thought' step, as the agent must reason about the new information, decide if its goal has been met, or if further actions are necessary. The agent's next 'Action' will be chosen based on both its internal plan and the latest observed feedback, enabling dynamic adaptation and iterative improvement towards the goal.

Thus, **processing external feedback in 'Observation' is crucial**—it closes the loop, enabling the agent to refine its approach, validate outcomes, and avoid acting blindly or repeatedly without learning from previous steps. This makes the agent interactive and adaptive rather than static or rigid.

-----

-----

-----

### Source [59]: https://dylancastillo.co/posts/react-agent-langgraph.html

Query: What is the specific role of the 'Observation' step in a ReAct agent's control loop, and how does processing external feedback from tools influence the agent's subsequent 'Thought' and 'Action' steps?

Answer: In a ReAct agent, the **think-act-observe loop** is implemented as a cycle in which the output of the 'Action' (typically tool execution) is passed into the 'Observation' step. The observation is then routed back as input to the LLM for the next round of reasoning.

The 'Observation' step serves as the **mechanism to capture and process the outcome of the agent's most recent action**. This could be the response from an API, the result of a database lookup, or any relevant output from an external tool. Once observed, this feedback is appended to the agent's message state or conversation history.

This updated state is then used as the primary context for the next 'Thought' step, where the LLM re-evaluates its plan based on the new information. Thus, **the agent's ability to process and incorporate external feedback through 'Observation' directly influences the subsequent reasoning ('Thought') and the selection of future actions**, ensuring the agent's behavior is responsive and context-aware.

-----

-----

-----

### Source [60]: https://www.nutrient.io/blog/rewoo-vs-react-choosing-right-agent-architecture/

Query: What is the specific role of the 'Observation' step in a ReAct agent's control loop, and how does processing external feedback from tools influence the agent's subsequent 'Thought' and 'Action' steps?

Answer: In the ReAct agent architecture, the loop of **generating a thought, taking an action, and then observing the result** is fundamental. The 'Observation' step is where the agent **receives and interprets the result of its previous action**, such as the output from a search or an API call.

Processing this external feedback is what enables the agent to **adapt its strategy dynamically**. After each observation, the agent re-engages its reasoning process ('Thought'), now informed by the latest data, and decides on the next best action. This cycle continues until the agent achieves its objective or exhausts its options.

The feedback loop created by the 'Observation' step is particularly powerful for tasks that are open-ended or where the pathway to the solution is not predefined. **The agent's effectiveness relies on how well it integrates observed results into its ongoing reasoning**, often outperforming agents that lack this iterative, feedback-driven structure.

-----

-----

-----

### Source [61]: https://langchain-ai.github.io/langgraphjs/concepts/agentic_concepts/

Query: What is the specific role of the 'Observation' step in a ReAct agent's control loop, and how does processing external feedback from tools influence the agent's subsequent 'Thought' and 'Action' steps?

Answer: In the ReAct architecture, the 'Observation' step follows tool execution and involves the **agent processing the output from an external tool**. This could be data retrieved from an API, a result from a search, or any other structured response.

The agent uses 'Observation' to **update its internal memory and context** with the latest feedback, which becomes essential input for subsequent reasoning ('Thought') steps. This allows the agent to **make informed, multi-step decisions**, leveraging both its prior knowledge and the most current external information.

Integrating tool results via 'Observation' ensures that the agent doesn't act in isolation but rather **continuously refines its plan and actions based on real-world feedback**. This dynamic loop is key to enabling flexible, robust, and interactive agent behaviors across complex tasks.

-----

-----

-----

### Source [62]: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/

Query: What is the specific role of the 'Observation' step in a ReAct agent's control loop, and how does processing external feedback from tools influence the agent's subsequent 'Thought' and 'Action' steps?

Answer: The 'Observation' step in ReAct agent architectures is the **process by which the agent receives and processes outputs from external tools**. For instance, after calling an API or function, the agent observes the result and incorporates it into its current context or memory.

This step is critical because it **enables the agent to ground its reasoning and future actions in the most recent, externally validated information**. With each observation, the agent can update its plan, correct mistakes, or decide if additional steps are necessary.

By processing feedback from tools in the 'Observation' step, the agent's next 'Thought' and 'Action' are **informed by a richer, continuously updated context**, supporting more sophisticated and adaptive problem-solving. This iterative, observation-informed process distinguishes ReAct agents from simpler, single-pass or non-interactive agent designs.

-----

-----

</details>

<details>
<summary>What are the best practices for designing the 'Action Phase' of a ReAct agent when using Gemini's native function calling, specifically regarding prompt strategy and the separation of strategic guidance from technical tool details?</summary>

### Source [63]: https://ai.gopubby.com/react-ai-agent-from-scratch-using-deepseek-handling-memory-tools-without-frameworks-cabda9094273

Query: What are the best practices for designing the 'Action Phase' of a ReAct agent when using Gemini's native function calling, specifically regarding prompt strategy and the separation of strategic guidance from technical tool details?

Answer: This guide details the architecture and execution flow of a ReAct agent, emphasizing the **Thought-Action-Observation loop** as the agent’s core. In the Action Phase, the agent’s logic involves:
- **Processing user input** and determining the required action using the `think` function, which performs reasoning before any tool invocation.
- **Executing the action** (i.e., function call or tool use) only after a reasoning step has clarified the intent and requirements.
- **Separating strategy and technical details**: Strategic guidance, such as when and why to use a tool, is embedded in the system prompt and demonstrated through few-shot examples. The technical details about tool usage (API parameters, expected input/output format) are encapsulated within the action execution logic, not exposed in strategic reasoning or prompts.
- **System prompt design**: The system prompt serves to guide the agent’s high-level decision-making and includes instructions/examples for when to trigger an action. It avoids technical API specifics, which are handled in the code, keeping prompts focused on strategy and intent.

Best practices from this approach:
- **Keep the system prompt focused on reasoning and intent**, using few-shot demonstrations for tool selection.
- **Encapsulate tool-specific technical instructions within the agent’s code**, not within the prompt.
- **Maintain a clear separation between strategic guidance (when and why to act) and the technical mechanics (how to act).**
- The agent executes the action and then observes the result, feeding it back into the reasoning loop for the next step.

This structure ensures modularity and clarity, making the agent’s reasoning transparent while isolating technical details for easier maintenance and adaptation to new tools.

-----

-----

-----

### Source [64]: https://ai.google.dev/gemini-api/docs/langgraph-example

Query: What are the best practices for designing the 'Action Phase' of a ReAct agent when using Gemini's native function calling, specifically regarding prompt strategy and the separation of strategic guidance from technical tool details?

Answer: This practical example demonstrates building a ReAct agent with Gemini 2.5 and LangGraph, highlighting **state management and action execution**:
- The agent’s state includes the conversation history and a step counter, ensuring context is maintained across iterative reasoning and action steps.
- The action phase is triggered after a reasoning step, where the agent decides, based on accumulated context, whether and which function to call (e.g., a weather API).
- **Prompt strategy**: The agent’s prompt provides high-level instructions and examples of tool use, but does not expose technical details of the function signature. Instead, the code manages function invocation and response handling.
- **Separation of concerns**: Strategic guidance on “when” and “why” to use a tool is modeled in the prompt and agent logic, whereas the “how” (parameters, function schema) is abstracted away in code, reducing cognitive load on the LLM.

The example uses helper utilities like `add_messages` to keep action results and observations organized, maintaining a strict loop: Reason → Act (function call) → Observe → Reason.

-----

-----

-----

### Source [65]: https://developers.googleblog.com/en/building-agents-google-gemini-open-source-frameworks/

Query: What are the best practices for designing the 'Action Phase' of a ReAct agent when using Gemini's native function calling, specifically regarding prompt strategy and the separation of strategic guidance from technical tool details?

Answer: This blog post outlines **best practices in agent design with Gemini and open-source frameworks**, emphasizing:
- **Advanced function calling**: Gemini’s models support direct, structured function calls, allowing the agent to trigger specific actions based on its reasoning.
- **Framework-assisted separation**: When using frameworks like LangGraph, the agent’s workflow is explicitly modeled as a graph, where each node represents either a reasoning step or an action (function call/tool use).
- **Prompt role**: The prompt guides the LLM’s strategic reasoning and intent—when to use a tool, not how to use it. Tool schemas and calling details are configured in the framework and abstracted from the prompt.
- **Iterative reflection**: The agent reflects on observations (function call results) before determining the next action, ensuring that technical tool details remain outside the agent’s high-level reasoning.

The recommended approach:
- Use prompts for **strategic guidance and intent articulation**.
- **Encapsulate technical API and tool details in code or framework configuration**, not in prompts.
- Enable stepwise, transparent agent reasoning by keeping action execution modular and separate from strategy.

-----

-----

</details>

<details>
<summary>What are the most common failure modes and debugging strategies for a ReAct agent's control loop, specifically addressing issues like hallucination, error propagation, and improper loop termination?</summary>

### Source [66]: https://arxiv.org/pdf/2503.13657

Query: What are the most common failure modes and debugging strategies for a ReAct agent's control loop, specifically addressing issues like hallucination, error propagation, and improper loop termination?

Answer: This paper introduces the MAST taxonomy, which organizes multi-agent system (MAS) failure modes—including those relevant to ReAct agents—into three main categories: Specification Issues, Execution Failures, and Post-Execution Failures. 

- **Specification Issues (FC1):** These failures are rooted in system design choices and ambiguous prompt specifications. Poorly defined control logic or insufficiently clear instructions can lead to improper loop termination, hallucination, or error propagation. For example, an agent may terminate too early if the stop condition is vague, or may propagate errors if the design does not properly handle intermediate mistakes.
- The taxonomy emphasizes that many observed failures (like hallucination and error propagation) are not solely due to language model limitations, but also stem from structural and design flaws in the agent's control loop.
- Debugging strategies require more than superficial fixes; they often necessitate structural redesigns to clarify specifications, handle error states, and enforce correct loop termination conditions.

The paper provides detailed definitions and examples for each failure mode, demonstrating that effective debugging involves both analyzing agent outputs and systematically evaluating system design choices impacting the control loop[1].

-----

-----

-----

### Source [67]: https://inspect.aisi.org.uk/react-agent.html

Query: What are the most common failure modes and debugging strategies for a ReAct agent's control loop, specifically addressing issues like hallucination, error propagation, and improper loop termination?

Answer: This resource focuses on the **control loop termination problem** in ReAct agents. 

- Some agents may unintentionally stop calling tools, for instance, by stating an intention to call a tool but failing to execute the actual call. This leads to **improper loop termination**.
- The recommended debugging and mitigation strategy is to implement an explicit `submit()` tool. By requiring the agent to call `submit()` to signal task completion, accidental or premature termination is avoided, and the agent is encouraged to keep iterating until it is truly finished.
- This approach enables support for **multiple attempts** at solving a task and provides clearer boundaries for loop termination.
- If disabling the `submit()` tool, developers can control termination with a custom handler, but this increases the risk that the agent will halt at the wrong time.

Explicit signaling with a `submit()` tool is a practical solution for debugging and reducing control loop errors, though it may not be appropriate in every application domain[2].

-----

-----

-----

### Source [68]: https://huyenchip.com/2025/01/07/agents.html

Query: What are the most common failure modes and debugging strategies for a ReAct agent's control loop, specifically addressing issues like hallucination, error propagation, and improper loop termination?

Answer: This source outlines several **common failure modes** for agents—including ReAct agents—and offers practical **evaluation and debugging strategies**:

- **Hallucination:** The agent may be convinced that it has accomplished a task when it has not, such as miscounting assignments or missing requirements. This is often due to errors in reflection and planning.
- **Error Propagation:** Errors in one tool call or step can lead to compounding mistakes in subsequent steps, especially if the agent does not verify outputs before proceeding.
- **Improper Loop Termination:** The agent might stop execution before the task is complete due to misinterpretation of goal state or an error in control logic.

**Debugging strategies include:**
- Creating datasets of tasks and tool inventories, generating multiple plans, and measuring the validity of outputs.
- Tracking metrics such as: proportion of valid plans, frequency of invalid tool calls, and parameter correctness.
- Analyzing output patterns to identify recurrent failure types or particularly troublesome tools.
- Improving performance with better prompting, more examples, or finetuning; if a specific tool is problematic, consider replacing it.

These strategies help isolate failure points (hallucination, error propagation, loop termination) and inform targeted interventions[3].

-----

-----

-----

### Source [69]: https://www.nutrient.io/blog/rewoo-vs-react-choosing-right-agent-architecture/

Query: What are the most common failure modes and debugging strategies for a ReAct agent's control loop, specifically addressing issues like hallucination, error propagation, and improper loop termination?

Answer: This article compares ReAct and ReWOO agent architectures, focusing on the **trade-offs of the ReAct loop**:

- The ReAct agent's "think-act-observe" loop is well suited to interactive, dynamic tasks requiring real-time adjustment.
- **Drawbacks** include the risk of high computational cost and latency, as each iteration requires a new prompt with the full history. If the agent does not converge quickly, the loop may become slow and expensive.
- The variance in reliability and efficiency is largely due to the agent’s tool access and the model’s ability to reason and act effectively.
- The article implies that failures such as hallucination or improper termination are often related to the agent's difficulty in correctly using tools or interpreting feedback, especially in ambiguous tasks.

While not focused on specific debugging strategies, it highlights the importance of matching the agent architecture to the task and being mindful of the loop's potential inefficiencies and error sources[4].

-----

-----

-----

### Source [70]: https://www.lesswrong.com/posts/sekmz9EiBD6ByZpyp/detecting-ai-agent-failure-modes-in-simulations

Query: What are the most common failure modes and debugging strategies for a ReAct agent's control loop, specifically addressing issues like hallucination, error propagation, and improper loop termination?

Answer: This case study describes simulation-based detection and resolution of AI agent failure modes, with findings relevant to ReAct agents:

- **Unintended Tool Usage:** The agent sometimes used tools (commands) in ways that were not intended, such as acting outside the permitted spatial boundaries or using commands that crashed the system. This is a form of **error propagation** and improper action selection.
- **Prompt Refinement:** Early versions of the system prompt led to mistakes; refining the prompt helped reduce errors. This demonstrates the importance of prompt clarity for controlling hallucination and minimizing error propagation.
- **Debugging Strategy:** The system maintained a sliding window of recent actions and continually prompted the agent to "Continue," preventing the context window from overflowing and reducing the risk of hallucination or infinite loops.
- **Outcome:** By iteratively refining the prompt and monitoring the agent’s actions, the team was able to identify and address failure modes, improving overall reliability.

This approach underscores the importance of **prompt engineering, monitoring, and iterative refinement** in debugging and improving agent control loops[5].
-----

-----

</details>


## Sources Scraped From Research Results

<details>
<summary>First proposed by [ReAct](https://arxiv.org/abs/2210.03629) (Yao et al., 2022), interleaving reasoning and action has become a common pattern for agents. Yao et al. used the term “reasoning” to encompass both planning and reflection. At each step, the agent is asked to explain its thinking (planning), take actions, then analyze observations (reflection), until the task is considered finished by the agent. The agent is typically prompted, using examples, to generate outputs in the following format:</summary>

First proposed by [ReAct](https://arxiv.org/abs/2210.03629) (Yao et al., 2022), interleaving reasoning and action has become a common pattern for agents. Yao et al. used the term “reasoning” to encompass both planning and reflection. At each step, the agent is asked to explain its thinking (planning), take actions, then analyze observations (reflection), until the task is considered finished by the agent. The agent is typically prompted, using examples, to generate outputs in the following format:

```
Thought 1: …
Act 1: …
Observation 1: …

… [continue until reflection determines that the task is finished] …

Thought N: …
Act N: Finish [Response to query]

```

Figure 6-12 shows an example of an agent following the ReAct framework responding to a question from HotpotQA ( [Yang et al., 2018](https://arxiv.org/abs/1809.09600)), a benchmark for multi-hop question answering.https://huyenchip.com/assets/pics/agents/5-ReAct.png
Figure 6-12: A ReAct agent in action.

</details>

<details>
<summary>This content is not relevant to the article guidelines.</summary>

This content is not relevant to the article guidelines.

</details>

<details>
<summary>Over the past year, we've worked with dozens of teams building large language model (LLM) agents across industries. Consistently, the most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns.</summary>

Over the past year, we've worked with dozens of teams building large language model (LLM) agents across industries. Consistently, the most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns.

In this post, we share what we’ve learned from working with our customers and building agents ourselves, and give practical advice for developers on building effective agents.

### Workflow: Prompt chaining

Prompt chaining decomposes a task into a sequence of steps, where each LLM call processes the output of the previous one. You can add programmatic checks (see "gate” in the diagram below) on any intermediate steps to ensure that the process is still on track.

**When to use this workflow:** This workflow is ideal for situations where the task can be easily and cleanly decomposed into fixed subtasks. The main goal is to trade off latency for higher accuracy, by making each LLM call an easier task.

**Examples where prompt chaining is useful:**

- Generating Marketing copy, then translating it into a different language.
- Writing an outline of a document, checking that the outline meets certain criteria, then writing the document based on the outline.

### Workflow: Routing

Routing classifies an input and directs it to a specialized followup task. This workflow allows for separation of concerns, and building more specialized prompts. Without this workflow, optimizing for one kind of input can hurt performance on other inputs.

**When to use this workflow:** Routing works well for complex tasks where there are distinct categories that are better handled separately, and where classification can be handled accurately, either by an LLM or a more traditional classification model/algorithm.

**Examples where routing is useful:**

- Directing different types of customer service queries (general questions, refund requests, technical support) into different downstream processes, prompts, and tools.
- Routing easy/common questions to smaller models like Claude 3.5 Haiku and hard/unusual questions to more capable models like Claude 3.5 Sonnet to optimize cost and speed.

### Workflow: Parallelization

LLMs can sometimes work simultaneously on a task and have their outputs aggregated programmatically. This workflow, parallelization, manifests in two key variations:

- **Sectioning**: Breaking a task into independent subtasks run in parallel.
- **Voting:** Running the same task multiple times to get diverse outputs.

**When to use this workflow:** Parallelization is effective when the divided subtasks can be parallelized for speed, or when multiple perspectives or attempts are needed for higher confidence results. For complex tasks with multiple considerations, LLMs generally perform better when each consideration is handled by a separate LLM call, allowing focused attention on each specific aspect.

**Examples where parallelization is useful:**

- **Sectioning**:
  - Implementing guardrails where one model instance processes user queries while another screens them for inappropriate content or requests. This tends to perform better than having the same LLM call handle both guardrails and the core response.
  - Automating evals for evaluating LLM performance, where each LLM call evaluates a different aspect of the model’s performance on a given prompt.
- **Voting**:
  - Reviewing a piece of code for vulnerabilities, where several different prompts review and flag the code if they find a problem.
  - Evaluating whether a given piece of content is inappropriate, with multiple prompts evaluating different aspects or requiring different vote thresholds to balance false positives and negatives.

### Workflow: Orchestrator-workers

In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results.

**When to use this workflow:** This workflow is well-suited for complex tasks where you can’t predict the subtasks needed (in coding, for example, the number of files that need to be changed and the nature of the change in each file likely depend on the task). Whereas it’s topographically similar, the key difference from parallelization is its flexibility—subtasks aren't pre-defined, but determined by the orchestrator based on the specific input.

**Example where orchestrator-workers is useful:**

- Coding products that make complex changes to multiple files each time.
- Search tasks that involve gathering and analyzing information from multiple sources for possible relevant information.

### Workflow: Evaluator-optimizer

In the evaluator-optimizer workflow, one LLM call generates a response while another provides evaluation and feedback in a loop.

**When to use this workflow:** This workflow is particularly effective when we have clear evaluation criteria, and when iterative refinement provides measurable value. The two signs of good fit are, first, that LLM responses can be demonstrably improved when a human articulates their feedback; and second, that the LLM can provide such feedback. This is analogous to the iterative writing process a human writer might go through when producing a polished document.

**Examples where evaluator-optimizer is useful:**

- Literary translation where there are nuances that the translator LLM might not capture initially, but where an evaluator LLM can provide useful critiques.
- Complex search tasks that require multiple rounds of searching and analysis to gather comprehensive information, where the evaluator decides whether further searches are warranted.

### Agents

Agents are emerging in production as LLMs mature in key capabilities—understanding complex inputs, engaging in reasoning and planning, using tools reliably, and recovering from errors. Agents begin their work with either a command from, or interactive discussion with, the human user. Once the task is clear, agents plan and operate independently, potentially returning to the human for further information or judgement. During execution, it's crucial for the agents to gain “ground truth” from the environment at each step (such as tool call results or code execution) to assess its progress. Agents can then pause for human feedback at checkpoints or when encountering blockers. The task often terminates upon completion, but it’s also common to include stopping conditions (such as a maximum number of iterations) to maintain control.

Agents can handle sophisticated tasks, but their implementation is often straightforward. They are typically just LLMs using tools based on environmental feedback in a loop. It is therefore crucial to design toolsets and their documentation clearly and thoughtfully.

**When to use agents:** Agents can be used for open-ended problems where it’s difficult or impossible to predict the required number of steps, and where you can’t hardcode a fixed path. The LLM will potentially operate for many turns, and you must have some level of trust in its decision-making. Agents' autonomy makes them ideal for scaling tasks in trusted environments.

The autonomous nature of agents means higher costs, and the potential for compounding errors. We recommend extensive testing in sandboxed environments, along with the appropriate guardrails.

</details>

<details>
<summary>(empty)</summary>

(empty)

</details>

<details>
<summary>AI agents within LangChain take a language model and tie it together with a set of tools to address larger, more complex tasks. Unlike a static chain of instructions, an agent dynamically decides at each step which action (tool) to take based on the conversation and intermediate results.</summary>

AI agents within LangChain take a language model and tie it together with a set of tools to address larger, more complex tasks. Unlike a static chain of instructions, an agent dynamically decides at each step which action (tool) to take based on the conversation and intermediate results.

In practice, this means the agent prompts the LLM (the "reasoning engine") to formulate its next action or question, potentially invoke a tool (like a web search or calculator), and then use whatever results show up as new information in its reasoning. The agent continues to loop until it develops a final answer.

## **What Is a LangChain Agent?**

A LangChain agent is an LLM-based system that can decide actions dynamically, such as calling tools or answering directly. Unlike a chain (fixed steps), an agent reasons step-by-step, adapting based on context.

## **Why use an agent instead of a simple chain?**

If your task is always the same process that follows a fixed sequence, a chain would suffice (a hard-coded sequence). If you would like to have flexibility which would allow operate with different tools or follow a more branching logic, you would need an agent. For instance, agents are useful for a process where you want the agent to be able to search the web, interact with a knowledge base, do something computational, and then summarize these steps in a single seamless process.

This guide will cover the basics and follow the building of a LangChain agent in detail step-by-step. This guide will cover the primary components (tools, LLMs, prompts), how the agent loop works, and best practices to create more robust agents. The examples will use the most current LangChain API (2025 version), and it is expected that the reader is familiar with Python and large language models (LLMs).

## **How It Works:**

- Follows a **loop**: Think → Act → Observe → Repeat.
- The LLM decides whether to act or respond.
- Tool results are fed back as context for the next step.

## **Analogy:**

Like a detective solving a case:

- Uses a **notebook** (scratchpad) for thoughts.
- Chooses from a **toolbox** (APIs/functions).
- Stops when confident in the answer.

## **Key Components of Agents:**

- **Tools**:
  - External functions or APIs with names and descriptions.
  - Used by the agent to perform tasks (e.g., search, math).
- **LLM**:
  - The decision-making model (e.g., GPT-4o-mini, Gemini 2.0 ).
  - Chooses the next action or gives the final answer based on tthe ool outputs.
- **Prompt/Scratchpad**:
  - Guides LLM for proper tool usage, its guardrails and clear tool differentiation.
  - Stores previous actions and results for context.

## **Tools: Building Blocks for Actions**

A tool is simply a Python function wrapped with metadata. For example, to make a calculator tool that evaluates arithmetic expressions, you might write:

```python
from langchain.tools import Tool

def calculate_expression(expr: str) -> str:
    try:
        result = eval(expr)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def return_dummy_weather(city: str) -> str:
    return f"The weather in {city} is cloudy"

calc_tool = Tool(
    name="Calculator",
    description="Performs simple arithmetic. Input should be a valid Python expression, e.g. '2+2'.",
    func=calculate_expression
)

# Dummy weather tool
weather_tool = Tool(
    name="WeatherSearch",
    description="Tells current weather of a city. Input should be a valid city in string, e.g 'paris'.",
    func=calculate_expression
)
```

This **calc\_tool** tells the agent that whenever it needs to do math, it can call the tool **"Calculator"** with an input string. The agent's prompt will include this tool's name and description (and optionally how to format the input). The description should be clear and specific – vagueness can confuse the agent, causing it to choose the wrong tool or misuse it.

LangChain comes with many built-in tools and tool wrappers. For example:

- **Web Search Tool**: Interfaces like TavilySearchResults or GoogleSerperAPIWrapper let an agent search the web. (You'll need API keys for external search services.)
- **Retriever Tool**: Wraps a vector database or document store. As described in one example, you might first load documents and create a retriever, then use create\_retriever\_tool to expose it to the agent. This tool fetches relevant text snippets from your data given the query.
- **Custom API Tools**: You can define your tool that calls any API. For instance, a weather\_tool calling a weather API, or a jira\_tool that creates JIRA tickets. The agent only needs the Python function; LangChain handles calling it when the agent decides.

When giving tools to an agent, we put them in a list:

```python
tools = [calc_tool, weather_tool, search_tool, retriever_tool, ...]
```

The agent will see this list (typically as part of the prompt, or via the Tool objects) and may choose among them.

\*\* Each tool should ideally perform a clear, atomic function. Complex or multi-step logic can confuse the agent. If needed, break tasks into simpler tools or chains and let the agent sequence them. \*\*

## **Language Model: The Reasoning Engine**

The LLM (often a chat model) processes prompts and generates next steps. In LangChain 2025, a common import is:

```python
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0)
```

You may also use ChatAnthropic, ChatGooglePalm, etc. Setting temperature=0 (or low) can make the agent's decisions more deterministic, which is often desirable for tool use. You'll pass this LLM to the agent.

## **Prompt (Agent Scratchpad)**

The agent prompt template defines how the LLM is instructed to behave. A common pattern (ReAct-style) is to include:

- **System/Instruction** : Explains to the assistant that it is an agent with certain tools. Example: "You are an agent designed to answer questions. You have access to the following tools…"
- **Tool Descriptions** : Lists each tool's name and description, so the model knows what actions it can take.
- **Format Guide** : Tells the model how to output its reasoning. For example, it might use a structured JSON or markdown format like you can also use libraries like **Pydantic** to get more precise and formatted JSON objects for tool calls.

Example Prompt based on our Calculator tool.

<Persona>

You are a helpful, precise AI assistant capable of solving user queries using available tools.

You can perform reasoning, fetch information, and carry out calculations when needed.

</Persona>

<Guardrails>

- Only call a tool if it's required to answer the question.
- Do not guess values or fabricate information.
- Never perform code execution or arithmetic by yourself; use the Calculator tool for all such tasks.

</Guardrails>

<AvailableTools>

<Tool>
    <Name>Calculator</Name>
    <Description>
      Performs simple arithmetic. Input must be a valid Python expression, such as '3 * (4 + 5)'.
      Use this tool only for basic math operations (e.g., +, -, *, /, parentheses).
    </Description>
    <Format>
      To use this tool, return:
      Action: Calculator
      Action Input: 2 + 2
    </Format>
</Tool>

<Tool>
    <Name>Weather</Name>
    <Description>
      Tells current weather of a city. Input should be a valid city in string, e.g 'paris'.
    </Description>
    <Format>
      To use this tool, return:
      Action: Weather
      Action Input: Paris
    </Format>
</Tool>

</AvailableTools>

## **How the Agent Loop Works**

Under the hood, an agent uses a loop to repeatedly query the LLM, parse its output, execute tools, and update context. Conceptually:

1. **Initial Input** : The user's question is given to the agent (and any system instructions).
2. **LLM Response** : The agent prompts the LLM, which returns either a final answer or an action.
3. **Tool Invocation (if any)** : If the output is an action, the agent executes the corresponding tool function with the provided input. (E.g. \*\*\* _calc\_tool(query="2_ 2") )
4. **Observe** : The result from the tool (text, JSON, etc.) is captured. The agent adds this result to the scratchpad.
5. **Loop or End** : The agent checks if the LLM signaled a final answer or if any stopping criteria (max steps/time) are met. If not finished, it goes back to step 2: it calls the LLM again, now including the new observations in the prompt. This continues, building up a chain of reasoning.
6. **Return Answer** : Once the agent decides it's done, it returns the final answer to the user.

This process is illustrated by the pseudocode in the LangChain source (simplified):

```python
from langchain.schema import HumanMessage, AIMessage, SystemMessage

def process_with_tool_loop(user_input: str):
    MAX_ITERATIONS = 10
    current_iteration = 0
    messages = [
        SystemMessage(content="You are a helpful assistant with access to a calculator tool."),
        HumanMessage(content=user_input)
    ]

    while current_iteration < MAX_ITERATIONS:
        print(f"Iteration {current_iteration + 1}")
        response = llm.invoke(messages)
        # Check if LLM wants to call a function
        if not response.additional_kwargs.get("function_call"):
            print(f"Final answer: {response.content}")
            break
        
        function_call = response.additional_kwargs["function_call"]
        function_name = function_call["name"]
        function_args = function_call["arguments"]

        # Execute the tool
        if function_name == "Calculator":
            import json
            args = json.loads(function_args)
            tool_result = calculate_expression(args.get("expr", ""))
        
        if function_name == "WeatherSearch":
            import json
            args = json.loads(function_args)
            tool_result = weather_tool(args.get("city", ""))
        
        # Add function call and result to conversation.
        messages.append(response)
        messages.append(AIMessage(content=f"Function result: {tool_result}"))
        current_iteration += 1
    
    return response.content
```

## **Managing History for the conversation**

When building AI chat systems, preserving conversation history is essential for providing contextual, coherent responses. The ConversationHistoryService handles this by transforming stored messages into LangChain-compatible message objects that the model can understand. This formatting is especially important when using OpenAI models, as LangChain standardizes the message structure (e.g., HumanMessage, AIMessage, ToolMessage) required for proper tool invocation and response handling.

**Different models may expect varying formats for tool calls and conversation history. For other LLMs, such as Gemini, the history format may differ especially when supporting Agentic behavior, so the message transformation logic must be adapted to match each model’s specific input requirements.**

This system:

- Handles multiple sender types (USER, AI, TOOL)
- Ensures messages are properly ordered and valid according to OpenAI LLM ( gpt-4o-mini ) requirements.
- Constructs an array of Langchain messages starting with the system prompt

We must store the complete conversation history along with the tool call and tool response in the Database, then at every LLM call, we should fetch the history from the DB and formulate that according to the langchain / LLM requirement.

For example :

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def convert_to_langchain_message(message, next_message=None):
    sender_type = message.get("sender_type")
    if sender_type == "TOOL":
        return ToolMessage(
            tool_call_id=message.get("tool_call_id"),
            name=message.get("content"),
            content=message.get("content")
        )
    elif sender_type == "USER":
        return HumanMessage(content=message.get("content"))
    else:  # Assume AI
        if next_message is None:
            return None
        
        if message.get("additional_metadata", {}).get("tool_calls") and next_message.get("sender_type") != "TOOL":
            return None
        
        return AIMessage(
            content=message.get("content"),
            additional_kwargs=message.get("additional_metadata", {})
        )
```

- It loops through stored conversation messages and based on the sender\_type, it converts each into the appropriate LangChain message:
  - TOOL ➜ ToolMessage
  - USER ➜ HumanMessage
  Otherwise (typically AI) ➜ AIMessage

## **Best Practices & Advanced Tips**

Building robust agents often requires attention to detail and thoughtful configuration. Here are some tips and advanced considerations:

- Clear Tool Description \*\*\*\*: The agent relies heavily on the text descriptions of tools. Make these descriptions concise and unambiguous. For example, specify input/output formats or any usage constraints. Poor descriptions can cause the agent to pick the wrong tool or misuse a tool.
- Zero-Shot vs Few-shot Prompts \*\*\*\*You can provide examples in the system prompt to guide the agent's reasoning (few-shot prompting), especially if the default behavior is incorrect. For example, give one or two example interactions showing how to use each tool.
- Control Temperature \*\*\*\*: Use a low temperature (e.g. 0.1 or 0.2) for agents to make consistent decisions. High randomness can lead to inconsistent tool use.

Set Iteration Limits **:** To avoid infinite loops, configure the agent executor's limits. LangChain's AgentExecutor has parameters max\_iterations (default 10) and max\_execution\_time to halt runaway loops.

## **Conclusion**

LangChain makes it surprisingly straightforward to build intelligent agents by combining LLM reasoning with tool usage. By defining clear tools and prompt instructions, you can create a system that handles multi-step questions and leverages external data or computation.

Remember that agents are powerful but also require careful crafting of prompts, descriptions, and limits to behave reliably. Whether you're building a QA chatbot that searches the web, an analytics assistant that processes databases, or any autonomous tool-based LLM system, understanding the agent loop and its components is key.

With the foundations in this guide, you can start designing your LangChain agents and explore more advanced topics like multi-agent coordination or integration with LangGraph for complex pipelines. Happy agent-building

</details>

<details>
<summary>In the previous lesson, we explored the theory behind agentic reasoning patterns like ReAct. Now, it's time to put that theory into practice. While high-level frameworks like LangChain or CrewAI offer powerful abstractions for building agents, they can also obscure the fundamental mechanics of how an agent actually "thinks" and "acts." Understanding these core mechanics is crucial for debugging, customizing, and truly mastering agent behavior.</summary>

In the previous lesson, we explored the theory behind agentic reasoning patterns like ReAct. Now, it's time to put that theory into practice. While high-level frameworks like LangChain or CrewAI offer powerful abstractions for building agents, they can also obscure the fundamental mechanics of how an agent actually "thinks" and "acts." Understanding these core mechanics is crucial for debugging, customizing, and truly mastering agent behavior.

This lesson is 100% hands-on. We are going to build a minimal, but complete, ReAct agent from scratch using only Python and the Gemini API. By the end, you will have implemented the entire Thought → Action → Observation loop yourself. You'll see exactly how an LLM can be prompted to reason, how its intent to use a tool is captured via function calling, and how the results of that tool are fed back into its context to inform the next step.

Our goal is to give you a concrete mental model of the ReAct pattern. With a working agent that you've built piece by piece, you'll have the confidence and clarity to tackle more complex agentic systems. We'll follow the code in the associated notebook, building the agent step-by-step.

Let's get started.

## 1. Setup and Environment

First, we need to set up our Python environment to ensure the code runs smoothly. This involves loading necessary credentials, importing libraries, and initializing the Gemini client.

We begin by loading our API key from an environment file. This is a standard practice to keep secrets out of your source code.

```python
from lessons.utils.env import load_env
import os

# Your API key should be stored in a .env file
# with the key GOOGLE_API_KEY
load_env()
```

Next, we import the required libraries: the Gemini client for interacting with the model, Pydantic and Enum for creating structured data classes, and a utility for pretty-printing our agent's outputs.

```python
import google.generativeai as genai
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional, Type
from lessons.utils.pretty_print import pretty_print_message
```

With the API key loaded, we initialize the `genai.GenerativeModel` client. We'll also select the model we'll use for our agent's "brain." For this lesson, we're using `gemini-1.5-flash-latest`, a fast and capable model well-suited for this task.

```python
# Initialize the Gemini client
try:
    client = genai.GenerativeModel("gemini-1.5-flash-latest")
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    print("Please ensure your GOOGLE_API_KEY is set correctly in your .env file.")
```

With our client and model ready, the next step is to give our agent a capability—a tool it can use to interact with the world.

## 2. Tool Layer: A Mock Search Tool

To demonstrate the "Action" part of the ReAct cycle, our agent needs a tool. For this lesson, we will create a simple mock `search` tool. Using a mock tool instead of a real search API offers several advantages for learning:

-   **Focus:** It keeps our focus on the agent's reasoning loop, not on the complexities of external API integrations.
-   **Simplicity:** It removes the need for additional API keys and dependencies.
-   **Predictability:** It provides consistent, predictable responses, which is essential for testing and understanding the agent's behavior.

Our mock tool will be a simple Python function that simulates looking up information. It's designed to recognize a few specific queries and return a predefined answer. If it receives a query it doesn't recognize, it will return a "not found" message.

Here is the implementation. Notice the docstring—it's not just a comment. As we'll see later, Gemini's function calling feature uses this docstring to understand what the tool does and how to use it.

```python
def search(query: str) -> str:
    """
    Searches for information about a given query.
    Only knows about the capital of France and the 2024 Summer Olympics.

    Args:
        query: The search query.
    """
    query = query.lower()
    if "capital of france" in query:
        return "Paris is the capital of France and is known for the Eiffel Tower."
    elif "2024 summer olympics" in query:
        return "The 2024 Summer Olympics were held in Paris, France."
    else:
        return f"Information about '{query}' was not found."

# A tool registry to hold our agent's tools
TOOL_REGISTRY = {"search": search}
```

In a real-world application, you could easily replace this mock function with a call to an actual search API like Google Search, a database query, or any other external data source. The agent's logic would remain the same; only the tool's implementation would change.

Now that our agent has a tool, we need to enable it to "think" about when and how to use it.

## 3. Thought Phase: Generating a Plan

The first step in the ReAct loop is "Thought." Here, the agent analyzes the user's request and its available tools, then formulates a plan. This plan is a short, internal monologue that guides its next action. We generate this thought by prompting the LLM with the conversation history and a description of the available tools.

To make the tools understandable to the LLM, we first format their descriptions into an XML-like structure. This structured format helps the model clearly distinguish the tools and their functionalities.

```python
def build_tools_xml_description(tool_registry: dict) -> str:
    """Builds an XML description of the tools for the LLM."""
    xml_description = "<tools>\n"
    for tool_name, tool_function in tool_registry.items():
        xml_description += f"<tool name='{tool_name}'>\n"
        xml_description += f"<description>{tool_function.__doc__}</description>\n"
        xml_description += "</tool>\n"
    xml_description += "</tools>"
    return xml_description

tools_xml = build_tools_xml_description(TOOL_REGISTRY)
```

Next, we create the prompt template for the thought-generation step. This template instructs the LLM to act as a helpful assistant, review the conversation, and decide on the next step. It explicitly tells the model to either use a tool or, if it has enough information, to prepare a final answer.

```python
PROMPT_TEMPLATE_THOUGHT = f"""
You are a helpful assistant. Your goal is to answer the user's question.
You have access to the following tools:

{tools_xml}

The conversation history is as follows:
{{conversation}}

Based on the conversation, what is the next step?
If you can answer the question, state that you have the final answer.
Otherwise, state which tool you will use and what you will search for.
Your response should be a single, short paragraph.
"""
```

Finally, we wrap this logic in a function, `generate_thought`. This function takes the current conversation history, formats the prompt, sends it to the Gemini model, and returns the model's response as the agent's thought.

```python
def generate_thought(conversation: str, tool_registry: dict) -> str:
    """Generates a thought for the agent."""
    prompt = PROMPT_TEMPLATE_THOUGHT.format(conversation=conversation)
    response = client.generate_content(prompt)
    return response.text.strip()
```

With a thought generated (e.g., "I need to use the search tool to find the capital of France"), the agent must translate this intention into a concrete action.

## 4. Action Phase: Deciding What to Do

The "Action" phase is where the agent decides its next move. This could be calling a tool or, if the task is complete, providing the final answer to the user. We'll leverage Gemini's native function calling capability to handle this decision-making process.

Function calling allows the model to indicate when it wants to execute one of the tools we've provided. Instead of just returning text, the model can return a structured `FunctionCall` object specifying the name of the tool to use and the arguments to pass to it.

Our action-generation prompt is simpler than the thought prompt. It doesn't need detailed tool descriptions because we will pass the tool functions directly to the Gemini API. The API automatically extracts the necessary information (name, description, parameters) from the Python function's signature and docstring. This keeps our prompt clean and focused on the high-level task.

```python
ACTION_SYSTEM_PROMPT = """
You are a helpful assistant. Your goal is to answer the user's question.
Given the conversation history, you must decide what to do next.
You have two options:
1.  Call a tool to get more information.
2.  Provide the final answer to the user if you have enough information.
"""
```

We define two special constants to represent the agent's decision: `ACTION_FINISH` for when it's ready to give a final answer, and `ACTION_UNKNOWN` for when it fails to make a clear decision.

```python
ACTION_FINISH = "finish"
ACTION_UNKNOWN = "unknown"
```

The core of this phase is the `generate_action` function. It takes the conversation history and the tool registry. It configures the Gemini client with the system prompt and the list of available tools. When we call `generate_content`, the model will either:

1.  Return a text response, which we interpret as the agent's readiness to provide a final answer.
2.  Return a `FunctionCall` object, indicating a tool should be used.

Our function parses the model's output and returns the action to be taken (e.g., `"search"`) and the corresponding arguments (e.g., `{"query": "capital of France"}`).

```python
def generate_action(conversation: str, tool_registry: dict) -> tuple[str, dict]:
    """Generates an action for the agent."""
    prompt = f"{ACTION_SYSTEM_PROMPT}\n\n{conversation}"
    
    response = client.generate_content(
        prompt,
        tools=list(tool_registry.values())
    )
    
    # Check if the model wants to call a tool
    if response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        action = function_call.name
        args = {key: value for key, value in function_call.args.items()}
        return action, args
    
    # If no tool call, assume it's a final answer
    return ACTION_FINISH, {"answer": response.text}
```

Now we have all the individual components: a tool, a way to generate thoughts, and a way to decide on actions. It's time to assemble them into a cohesive control loop that orchestrates the entire ReAct process.

## 5. Control Loop: Orchestrating the ReAct Cycle

The control loop is the heart of our agent. It manages the flow of the ReAct cycle: Thought → Action → Observation. It maintains the state of the conversation in a "scratchpad" and iterates through the cycle until the user's question is answered or a set limit is reached.

First, let's define a simple data structure to keep our conversation history organized. We'll use an `Enum` for message roles and a `Pydantic` model for the messages themselves. This ensures every entry in our scratchpad is structured and easy to read.

```python
class MessageRole(Enum):
    USER = "user"
    THOUGHT = "thought"
    TOOL_REQUEST = "tool_request"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"

class Message(BaseModel):
    role: MessageRole
    content: str
```

The scratchpad will simply be a list of these `Message` objects. We'll also create a helper function to format the scratchpad content into a single string for the LLM prompts.

```python
def format_scratchpad(scratchpad: List[Message]) -> str:
    """Formats the scratchpad into a string for the LLM."""
    formatted_str = ""
    for msg in scratchpad:
        formatted_str += f"{msg.role.value.capitalize()}: {msg.content}\n"
    return formatted_str.strip()
```

Now, we build the main `react_agent_loop`. This function orchestrates the entire process:

1.  **Initialization**: It starts by adding the user's initial query to the scratchpad.
2.  **Iteration**: It enters a loop that runs for a maximum number of turns.
3.  **Thought**: Inside the loop, it calls `generate_thought` to determine the agent's plan.
4.  **Action**: It then calls `generate_action` to decide on the next concrete step.
5.  **Execution/Observation**:
    *   If the action is to call a tool, the loop finds the tool in the `TOOL_REGISTRY` and executes it with the provided arguments. The tool's output is recorded as an "Observation."
    *   If the action is `ACTION_FINISH`, the loop breaks, and the final answer is recorded.
    *   It includes error handling for cases where the agent tries to call a tool that doesn't exist.
6.  **Termination**: The loop terminates when a final answer is generated or the maximum number of turns is exceeded. If the loop times out, it forces the agent to generate a final answer based on the information it has gathered so far.

Here is the complete implementation of the control loop:

```python
def react_agent_loop(
    query: str, 
    tool_registry: dict, 
    max_turns: int = 5, 
    verbose: bool = False
) -> str:
    scratchpad = [Message(role=MessageRole.USER, content=query)]
    
    for i in range(max_turns):
        if verbose:
            print(f"--- Turn {i+1}/{max_turns} ---")

        # 1. Thought
        conversation = format_scratchpad(scratchpad)
        thought = generate_thought(conversation, tool_registry)
        thought_message = Message(role=MessageRole.THOUGHT, content=thought)
        scratchpad.append(thought_message)
        if verbose:
            pretty_print_message(thought_message)

        # 2. Action
        action, args = generate_action(conversation, tool_registry)
        
        if action == ACTION_FINISH:
            final_answer = Message(role=MessageRole.FINAL_ANSWER, content=args.get("answer", "No answer found."))
            scratchpad.append(final_answer)
            if verbose:
                pretty_print_message(final_answer)
            return final_answer.content

        tool_request_message = Message(role=MessageRole.TOOL_REQUEST, content=f"Tool: {action}, Args: {args}")
        scratchpad.append(tool_request_message)
        if verbose:
            pretty_print_message(tool_request_message)

        # 3. Observation
        if action in tool_registry:
            tool_function = tool_registry[action]
            try:
                observation_content = tool_function(**args)
            except Exception as e:
                observation_content = f"Error executing tool {action}: {e}"
        else:
            observation_content = f"Tool '{action}' not found. Available tools: {list(tool_registry.keys())}"
        
        observation_message = Message(role=MessageRole.OBSERVATION, content=observation_content)
        scratchpad.append(observation_message)
        if verbose:
            pretty_print_message(observation_message)

    # Force a final answer if max turns are reached
    final_answer_prompt = f"{format_scratchpad(scratchpad)}\n\nBased on the conversation, what is the final answer?"
    final_answer_text = client.generate_content(final_answer_prompt).text
    final_answer = Message(role=MessageRole.FINAL_ANSWER, content=final_answer_text)
    if verbose:
        print("--- Max turns reached, forcing final answer ---")
        pretty_print_message(final_answer)

    return final_answer.content
```

With the full loop implemented, it's time to test our agent and see it in action.

## 6. Tests and Traces: Validating the Agent

The final step is to validate our ReAct agent with a couple of test cases. We'll run the loop with `verbose=True` to see the full trace of its thought process. This will allow us to confirm that each part of the system—thought, action, observation, and control loop—is working as expected.

### Test Case 1: A Factual Question (Success Path)

First, let's ask a question that our mock `search` tool knows how to answer: "What is the capital of France?" We'll limit the agent to two turns.

```python
query_france = "What is the capital of France?"
result_france = react_agent_loop(query_france, TOOL_REGISTRY, max_turns=2, verbose=True)
```

The output trace clearly shows the ReAct cycle in action:

-   **Turn 1**:
    -   **Thought**: The agent correctly identifies that it needs to use the `search` tool to find the capital of France.
    -   **Tool Request**: It generates a valid call to `search(query='capital of France')`.
    -   **Observation**: The control loop executes the tool, which returns the factual answer: "Paris is the capital of France..."
-   **Turn 2**:
    -   **Thought**: With the observation in its context, the agent recognizes it now has the information needed to answer the user's question.
    -   **Final Answer**: It decides to conclude the loop and provides the correct answer: "Paris is the capital of France."

This successful run validates that the entire end-to-end loop works. The agent can reason about a user's query, use a tool to find information, process the result, and formulate a final answer.

### Test Case 2: An Unknown Question (Graceful Fallback)

Now, let's test the agent's ability to handle a query our mock tool doesn't know about: "What is the capital of Italy?" This will test the agent's reasoning when a tool fails to provide useful information and will also demonstrate the forced termination logic.

```python
query_italy = "What is the capital of Italy?"
result_italy = react_agent_loop(query_italy, TOOL_REGISTRY, max_turns=2, verbose=True)
```

The trace for this query demonstrates the agent's ability to adapt and handle failure gracefully:

-   **Turn 1**:
    -   **Thought & Tool Request**: The agent correctly attempts to search for "capital of Italy."
    -   **Observation**: The mock tool returns the "not found" message.
-   **Turn 2**:
    -   **Thought**: The agent observes the failure and adjusts its strategy. It decides to try a broader search for just "Italy," hoping to find relevant information.
    -   **Tool Request**: It calls `search(query='Italy')`.
    -   **Observation**: This search also fails, as our mock tool doesn't know about Italy.
-   **Forced Final Answer**: Having reached the `max_turns` limit of 2 without finding a definitive answer, the control loop forces a conclusion. The agent synthesizes the information it has (which is that it failed to find the answer) and provides a polite, honest response like, "I'm sorry, but I couldn't find information about the capital of Italy."

This test confirms that our agent doesn't get stuck in a loop when its tools fail. It can recognize failure, attempt alternative strategies, and provide a sensible final response when it exhausts its options. This resilience is a key feature of a well-designed agent.

These tests show that our simple, from-scratch ReAct agent is fully functional. We have successfully built the core engine of an autonomous agent, providing a solid foundation for more advanced capabilities, which we will explore in future lessons.

</details>

<details>
<summary><none></summary>

<none>

</details>


## Code Sources

<details>
<summary>Repository analysis for https://github.com/towardsai/course-ai-agents/blob/dev/lessons/08_react_practice/notebook.ipynb</summary>

# Repository analysis for https://github.com/towardsai/course-ai-agents/blob/dev/lessons/08_react_practice/notebook.ipynb

## Summary
Repository: towardsai/course-ai-agents
Branch: dev
File: notebook.ipynb
Lines: 579

Estimated tokens: 4.8k

## File tree
```Directory structure:
└── notebook.ipynb

```

## Extracted content
================================================
FILE: lessons/08_react_practice/notebook.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Lesson 8: ReAct Practice

This notebook explores practical ReAct (Reasoning and Acting) with Google's Gemini. We will use the `google-genai` library to interact with Gemini models. It includes a mock search tool, a thought generation phase using structured outputs, and an action phase with function calling, all orchestrated by a ReAct control loop.
"""

"""
**Learning Objectives:**

1. Understand how ReAct breaks problems into Thought → Action → Observation.
2. Practice orchestrating the full ReAct loop end-to-end.
"""

"""
## 1. Setup

First, we define some standard Magic Python commands to autoreload Python packages whenever they change:
"""

%load_ext autoreload
%autoreload 2

"""
### Set Up Python Environment

To set up your Python virtual environment using `uv` and load it into the Notebook, follow the step-by-step instructions from the `Course Admin` lesson from the beginning of the course.

**TL/DR:** Be sure the correct kernel pointing to your `uv` virtual environment is selected.
"""

"""
### Configure Gemini API

To configure the Gemini API, follow the step-by-step instructions from the `Course Admin` lesson.

But here is a quick check on what you need to run this Notebook:

1.  Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  From the root of your project, run: `cp .env.example .env` 
3.  Within the `.env` file, fill in the `GOOGLE_API_KEY` variable:

Now, the code below will load the key from the `.env` file:
"""

from lessons.utils import env

env.load(required_env_vars=["GOOGLE_API_KEY"])
# Output:
#   Trying to load environment variables from `/Users/fabio/Desktop/course-ai-agents/.env`

#   Environment variables loaded successfully.


"""
### Import Key Packages
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import List

from google import genai
from google.genai import types

from lessons.utils import pretty_print

"""
### Initialize the Gemini Client
"""

client = genai.Client()
# Output:
#   Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.


"""
### Define Constants

We will use the `gemini-2.5-flash` model, which is fast and cost-effective:
"""

MODEL_ID = "gemini-2.5-flash"

"""
## 2. Tools Definition

Let's implement our mock search tool that will serve as the external knowledge source for our agent. This simplified version focuses on the ReAct mechanics rather than real API integration:
"""

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

"""
We maintain a mapping from tool name to tool function (the tool registry). This lets the model plan with symbolic tool names, while our code safely resolves those names to actual Python functions to execute.
"""

TOOL_REGISTRY = {
    search.__name__: search,
}

"""
## 3. ReAct Thought Phase

Now let's implement the thought generation phase. This component analyzes the current situation and determines what the agent should do next, potentially suggesting using tools.
"""

"""
First, we prepare the prompt for the thinking part. We implement a function that converts the `TOOL_REGISTRY` to a string XML representation of it, which we insert into the prompt. This way, the LLM knows which tools available and can reason around them.
"""

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

"""
Here we `print` the full prompt with the tool definitions inside.
"""

print(PROMPT_TEMPLATE_THOUGHT)
# Output:
#   You are deciding the next best step for reaching the user goal. You have some tools available to you.

#   

#   Available tools:

#   <tools>

#   	<tool name="search">

#   		<description>

#   			Search for information about a specific topic or query.

#   			

#   			Args:

#   			    query (str): The search query or topic to look up.

#   		</description>

#   	</tool>

#   </tools>

#   

#   Conversation so far:

#   <conversation>

#   {conversation}

#   </conversation>

#   

#   State your next thought about what to do next as one short paragraph focused on the next action you intend to take and why.

#   Avoid repeating the same strategies that didn't work previously. Prefer different approaches.


"""
We can now implement the `generate_thought` function, which reasons on the best next action to take according to the conversation history.
"""

def generate_thought(conversation: str, tool_registry: dict[str, callable]) -> str:
    """Generate a thought as plain text (no structured output)."""
    tools_xml = build_tools_xml_description(tool_registry)
    prompt = PROMPT_TEMPLATE_THOUGHT.format(conversation=conversation, tools_xml=tools_xml)

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt
    )
    return response.text.strip()

"""
## 4. ReAct Action Phase
"""

"""
Next, let's implement the action phase using function calling. This component determines whether to use a tool or provide a final answer.
"""

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


def generate_action(conversation: str, tool_registry: dict[str, callable] | None = None, force_final: bool = False) -> (ToolCallRequest | FinalAnswer):
    """Generate an action by passing tools to the LLM and parsing function calls or final text.

    When force_final is True or no tools are provided, the model is instructed to produce a final answer and tool calls are disabled.
    """
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

"""
Why we provide an option to force the final answer? In a ReAct loop we sometimes need to terminate cleanly after a budget of turns (e.g., to avoid infinite loops or excessive tool calls). The force flag lets us ask the model to conclude with a final answer even if, under normal conditions, it might keep calling tools. This ensures graceful shutdown and a usable output at the end of the loop.
"""

"""
Note: In the Action phase we do not inline tool descriptions into the prompt (unlike the Thought phase). Instead, we pass the available Python tool functions through the `tools` parameter to `generate_content`. The client automatically parses these tools and incorporates their definitions/arguments into the model's prompt context, enabling function calling without duplicating tool specs in our prompt text.
"""

"""
## 5. ReAct Control Loop
"""

"""
Now we build the main ReAct control loop that orchestrates the Thought → Action → Observation cycle end-to-end. We treat the conversation between the user and the agent as a sequence of messages. Each message is a step in the dialogue, and each step corresponds to one ReAct unit: it can be a user message, an internal thought, a tool request, the tool's observation, or the final answer.

We'll start by defining the data structures for these messages.
"""

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

"""
We also add a small printer that uses our `pretty_print` module to render each message nicely in the notebook. This makes it easy to follow how the agent alternates between Thought, Action (tool call), and Observation across turns.
"""

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

"""
We now use a `Scratchpad` class that wraps a list of `Message` objects and provides `append(..., verbose=False)` to both store and (optionally) pretty-print messages with role-based colors. The scratchpad is serialized each turn so the model can plan the next step.
"""

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
            role_to_color = {
                MessageRole.USER: pretty_print.Color.RESET,
                MessageRole.THOUGHT: pretty_print.Color.ORANGE,
                MessageRole.TOOL_REQUEST: pretty_print.Color.GREEN,
                MessageRole.OBSERVATION: pretty_print.Color.YELLOW,
                MessageRole.FINAL_ANSWER: pretty_print.Color.CYAN,
            }
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

"""
We can now implement the control loop.
- On the first turn, we add the user question.
- Then, at each turn: (1) we get a Thought from the model; (2) we get an Action. If the action is a `FinalAnswer`, we stop. If it's a `ToolCallRequest`, we execute the tool and append the resulting `Observation`, then continue. If we reach the maximum number of turns, we run the action selector one last time with a flag that forces a final answer (no tool calls).
"""

def react_agent_loop(initial_question: str, tool_registry: dict[str, callable], max_turns: int = 5, verbose: bool = False) -> str:
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
            final_message = Message(role=MessageRole.FINAL_ANSWER, content=final_answer)
            scratchpad.append(final_message, verbose=verbose, is_forced_final_answer=True)
            return final_answer

"""
Let's test our ReAct agent with a simple factual question that requires a search:
"""

# A straightforward question requiring a search.
question = "What is the capital of France?"
final_answer = react_agent_loop(question, TOOL_REGISTRY, max_turns=2, verbose=True)
# Output:
#   [0m----------------------------------------- User (Turn 1/2): -----------------------------------------[0m

#     What is the capital of France?

#   [0m----------------------------------------------------------------------------------------------------[0m

#   [38;5;208m--------------------------------------- Thought (Turn 1/2): ---------------------------------------[0m

#     I need to find the capital of France to answer the user's question. The `search` tool can be used to retrieve this factual information.

#   [38;5;208m----------------------------------------------------------------------------------------------------[0m

#   [92m------------------------------------- Tool request (Turn 1/2): -------------------------------------[0m

#     search(query='capital of France')

#   [92m----------------------------------------------------------------------------------------------------[0m

#   [93m------------------------------------- Observation (Turn 1/2): -------------------------------------[0m

#     Paris is the capital of France and is known for the Eiffel Tower.

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [38;5;208m--------------------------------------- Thought (Turn 2/2): ---------------------------------------[0m

#     I have successfully found the capital of France using the search tool. The next step is to communicate this answer to the user.

#   [38;5;208m----------------------------------------------------------------------------------------------------[0m

#   [96m------------------------------------- Final answer (Turn 2/2): -------------------------------------[0m

#     Paris is the capital of France.

#   [96m----------------------------------------------------------------------------------------------------[0m


"""
Last, let's test it with a question that our mock search tool doesn't have knowledge about:
"""

# A question about a concept the mock search tool doesn't know.
question = "What is the capital of Italy?"
final_answer = react_agent_loop(question, TOOL_REGISTRY, max_turns=2, verbose=True)
# Output:
#   [0m----------------------------------------- User (Turn 1/2): -----------------------------------------[0m

#     What is the capital of Italy?

#   [0m----------------------------------------------------------------------------------------------------[0m

#   [38;5;208m--------------------------------------- Thought (Turn 1/2): ---------------------------------------[0m

#     I need to find the capital of Italy to answer the user's question. The `search` tool can provide this information efficiently.I will use the `search` tool to find the capital of Italy.

#   [38;5;208m----------------------------------------------------------------------------------------------------[0m

#   [92m------------------------------------- Tool request (Turn 1/2): -------------------------------------[0m

#     search(query='capital of Italy')

#   [92m----------------------------------------------------------------------------------------------------[0m

#   [93m------------------------------------- Observation (Turn 1/2): -------------------------------------[0m

#     Information about 'capital of Italy' was not found.

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [38;5;208m--------------------------------------- Thought (Turn 2/2): ---------------------------------------[0m

#     The previous search query "capital of Italy" did not return information, which is unexpected for a common fact. I will try a broader search query like "Italy" to see if more general information or a list of facts about Italy can provide the capital, as the tool might respond better to less specific phrasing or have had a temporary issue with the exact previous query.I will try a broader search query like "Italy" to see if more general information or a list of facts about Italy can provide the capital, as the tool might respond better to less specific phrasing or have had a temporary issue with the exact previous query.

#   [38;5;208m----------------------------------------------------------------------------------------------------[0m

#   [92m------------------------------------- Tool request (Turn 2/2): -------------------------------------[0m

#     search(query='Italy')

#   [92m----------------------------------------------------------------------------------------------------[0m

#   [93m------------------------------------- Observation (Turn 2/2): -------------------------------------[0m

#     Information about 'Italy' was not found.

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [96m-------------------------------------- Final answer (Forced): --------------------------------------[0m

#     I'm sorry, but I couldn't find information about the capital of Italy.

#   [96m----------------------------------------------------------------------------------------------------[0m


"""
Notice how the ReAct agent tried different strategies to find an answer for the user query, demonstrating live adaptation.
"""

</details>


## YouTube Video Transcripts

_No YouTube video transcripts found._


## Additional Sources Scraped

<details>
<summary>arxiv-org</summary>

# ABSTRACT

While large language models (LLMs) have demonstrated impressive performance across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g. chain-of-thought prompting) and acting (e.g. action plan generation) have primarily been studied as separate topics. In this paper, we explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with and gather additional information from external sources such as knowledge bases or environments. We apply our approach, named ReAct, to a diverse set of language and decision making tasks and demonstrate its effectiveness over state-of-the-art baselines in addition to improved human interpretability and trustworthiness. Concretely, on question answering (HotpotQA) and fact verification (Fever), ReAct overcomes prevalent issues of hallucination and error propagation in chain-of-thought reasoning by interacting with a simple Wikipedia API, and generating human-like task-solving trajectories that are more interpretable than baselines without reasoning traces. Furthermore, on two interactive decision making benchmarks (ALFWorld and WebShop), ReAct outperforms imitation and reinforcement learning methods by an absolute success rate of $34 %$ and $10 %$ respectively, while being prompted with only one or two in-context examples.

# 1 INTRODUCTION

A unique feature of human intelligence is the ability to seamlessly combine task-oriented actions with verbal reasoning (or inner speech, Alderson-Day & Fernyhough, 2015), which has been theorized to play an important role in human cognition for enabling self-regulation or strategization (Vygotsky, 1987; Luria, 1965; Fernyhough, 2010) and maintaining a working memory (Baddeley, 1992). Consider the example of cooking up a dish in the kitchen. Between any two specific actions, we may reason in language in order to track progress (“now that everything is cut, I should heat up the pot of water”), to handle exceptions or adjust the plan according to the situation (“I don’t have salt, so let me use soy sauce and pepper instead”), and to realize when external information is needed (“how do I prepare dough? Let me search on the Internet”). We may also act (open a cookbook to read the recipe, open the fridge, check ingredients) to support the reasoning and to answer questions (“What dish can I make right now?”). This tight synergy between “acting” and “reasoning” allows humans to learn new tasks quickly and perform robust decision making or reasoning, even under previously unseen circumstances or facing information uncertainties.

Recent results have hinted at the possibility of combining verbal reasoning with interactive decision making in autonomous systems. On one hand, properly prompted large language models (LLMs) have demonstrated emergent capabilities to carry out several steps of reasoning traces to derive answers from questions in arithmetic, commonsense, and symbolic reasoning tasks (Wei et al., 2022). However, this “chain-of-thought” reasoning is a static black box, in that the model uses its own internal representations to generate thoughts and is not grounded in the external world, which limits its ability to reason reactively or update its knowledge. This can lead to issues like fact hallucination and error propagation over the reasoning process (Figure 1 (1b)). On the other hand, recent work has explored the use of pre-trained language models for planning and acting in interactive environments (Ahn et al., 2022; Nakano et al., 2021; Yao et al., 2020; Huang et al., 2022a), with a focus on predicting actions via language priors. These approaches usually convert multi-modal observations into text, use a language model to generate domain-specific actions or plans, and then use a controller to choose or execute them. However, they do not employ language models to reason abstractly about high-level goals or maintain a working memory to support acting, barring Huang et al. (2022b) who perform a limited form of verbal reasoning to reiterate spatial facts about the current state. Beyond such simple embodied tasks to interact with a few blocks, there have not been studies on how reasoning and acting can be combined in a synergistic manner for general task solving, and if such a combination can bring systematic benefits compared to reasoning or acting alone.https://arxiv.org/pdf/images/dd0e9f64b42d2cab71cdcecddd80ea2cf5aa212b5bf9a21882834d8e50a5302d.jpg

Figure 1: (1) Comparison of 4 prompting methods, (a) Standard, (b) Chain-of-thought (CoT, Reason Only), (c) Act-only, and (d) ReAct (Reason+Act), solving a HotpotQA (Yang et al., 2018) question. (2) Comparison of (a) Act-only and (b) ReAct prompting to solve an AlfWorld (Shridhar et al., 2020b) game. In both domains, we omit in-context examples in the prompt, and only show task solving trajectories generated by the model (Act, Thought) and the environment (Obs).

In this work, we present ReAct, a general paradigm to combine reasoning and acting with language models for solving diverse language reasoning and decision making tasks (Figure 1). ReAct prompts LLMs to generate both verbal reasoning traces and actions pertaining to a task in an interleaved manner, which allows the model to perform dynamic reasoning to create, maintain, and adjust high-level plans for acting (reason to act), while also interact with the external environments (e.g. Wikipedia) to incorporate additional information into reasoning (act to reason).

# 2 REAC T: SYNERGIZING REASONING + ACTING

Consider a general setup of an agent interacting with an environment for task solving. At time step $t$ , an agent receives an observation $o \_ { t } \\in \\mathcal { O }$ from the environment and takes an action $a \_ { t } \\in \\mathcal A$ following some policy $\\pi ( \\boldsymbol { a } \_ { t } \| \\boldsymbol { c } \_ { t } )$ , where $c \_ { t } = \\left( o \_ { 1 } , a \_ { 1 } , \\cdot \\cdot \\cdot , o \_ { t - 1 } , a \_ { t - 1 } , o \_ { t } \\right)$ is the context to the agent. Learning a policy is challenging when the mapping $c \_ { t } \\mapsto a \_ { t }$ is highly implicit and requires extensive computation. For example, the agent shown in Figure 1(1c) is unable to generate the correct final action (Act 4) to finish the QA task as it requires complex reasoning over the trajectory context (Question, Act 1-3, Obs 1-3). Similarly, the agent shown in Figure 1(2a) fails to comprehend from the context that sinkbasin 1 does not contain peppershaker 1, thus keep producing hallucinating actions.

The idea of ReAct is simple: we augment the agent’s action space to ${ \\hat { \\mathcal { A } } } = { \\mathcal { A } } \\cup { \\mathcal { L } }$ , where $\\mathcal { L }$ is the space of language. An action $\\hat { a } \_ { t } \\in \\mathcal { L }$ in the language space, which we will refer to as a thought or a reasoning trace, does not affect the external environment, thus leading to no observation feedback. Instead, a thought $\\hat { a } \_ { t }$ aims to compose useful information by reasoning over the current context $c \_ { t }$ , and update the context $\\boldsymbol c \_ { t + 1 } = \\left( c \_ { t } , \\hat { a } \_ { t } \\right)$ to support future reasoning or acting. As shown in Figure 1, there could be various types of useful thoughts, e.g. decomposing task goals and create action plans (2b, Act 1; 1d, Thought 1), injecting commonsense knowledge relevant to task solving (2b, Act 1), extracting important parts from observations (1d, Thought2, 4), track progress and transit action plans (2b, Act 8), handle exceptions and adjust action plans (1d, Thought 3), and so on.

However, as the language space $\\mathcal { L }$ is unlimited, learning in this augmented action space is difficult and requires strong language priors. In this paper, we mainly focus on the setup where a frozen large language model, PaLM-540B (Chowdhery et al., 2022)1, is prompted with few-shot in-context examples to generate both domain-specific actions and free-form language thoughts for task solving (Figure 1 (1d), (2b)). Each in-context example is a human trajectory of actions, thoughts, and environment observations to solve a task instance (see Appendix C). For the tasks where reasoning is of primary importance (Figure 1(1)), we alternate the generation of thoughts and actions so that the task-solving trajectory consists of multiple thought-action-observation steps. In contrast, for decision making tasks that potentially involve a large number of actions (Figure 1(2)), thoughts only need to appear sparsely in the most relevant positions of a trajectory, so we let the language model decide the asynchronous occurrence of thoughts and actions for itself.

# 3 KNOWLEDGE-INTENSIVE REASONING TASKS

We begin with knowledge-intensive reasoning tasks like multi-hop question answering and fact verification. As shown in Figure 1(1d), by interacting with a Wikipedia API, ReAct is able to retrieve information to support reasoning, while also use reasoning to target what to retrieve next, demonstrating a synergy of reasoning and acting.

</details>

<details>
<summary>building-effective-ai-agents-anthropic</summary>

Over the past year, we've worked with dozens of teams building large language model (LLM) agents across industries. Consistently, the most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns.

In this post, we share what we’ve learned from working with our customers and building agents ourselves, and give practical advice for developers on building effective agents.

</details>

<details>
<summary>building-react-agents-from-scratch-a-hands-on-guide-using-ge</summary>

Throughout human history, tool use has been a defining characteristic of our species, shaping our evolution and cognitive development. Tools have been instrumental in human evolution, enhancing both our physical and mental abilities. They’ve enabled us to surpass our natural limitations, fostering cognitive growth and a deeper understanding of cause and effect. Through tools, we’ve developed increasingly compact technologies and learned to view external objects as extensions of ourselves, significantly expanding our capabilities to interact with and shape our environment. As we venture into the era of artificial intelligence (AI), we find ourselves at a fascinating juncture where AI agents are beginning to mirror this fundamental aspect of human behavior. By incorporating tool use and complex reasoning, AI agents are pushing the boundaries of what machines can accomplish, potentially revolutionizing how AI interacts with and understands the world around it.

Before we dive into the building process, it’s important to clearly define AI agents and distinguish them from traditional AI systems that perform similar tasks. Let’s examine these concepts in detail:

# What is an AI Agent?

An AI agent is a system designed to perceive its environment, reason about it, and execute actions to achieve specified objectives. It operates by decomposing complex goals into manageable subtasks, selecting appropriate tools (such as search engines, database queries, code execution environments, API calls, and other agents) for each subtask, and iteratively executing these tools while analyzing the resulting observations. The agent adapts its strategy based on intermediate outcomes, refines inputs to optimize tool usage, and maintains a historical context to avoid repeating ineffective approaches. Throughout this process, the agent balances short-term subtasks with its overarching objective, efficiently aggregating and synthesizing results to formulate a comprehensive solution.

In this post, we’ll focus on ReAct as a methodology for creating AI agents capable of performing quite complex tasks given a set of tools. Note that there are other popular approaches to this, like _function calling_, which we’ll cover in a later post.

# The ReAct Framework: A Paradigm Shift

The ReAct framework introduces a unified architecture for AI agents that capitalizes on recent advancements in LLM capabilities. Unlike traditional designs that compartmentalize reasoning and action, ReAct integrates these functionalities into a single, cohesive system. This innovative approach allows AI agents to seamlessly combine thought processes and actions, potentially leading to more efficient and adaptable artificial intelligence systems. This paradigm shift is characterized by three key features:

1. _Unified Reasoning and Acting_: ReAct agents utilize LLMs as centralized components that concurrently reason about the environment and determine appropriate actions. This unification allows the agent to process observations, generate plans, and execute actions seamlessly, eliminating the need for separate, manually designed modules. By integrating reasoning and acting, the agent can adapt more fluidly to complex and dynamic environments.
2. _Dynamic Tool Utilization_: ReAct agents can incorporate a variety of external tools and APIs, selecting and employing them based on the current context and objectives. The LLM facilitates tool selection by analyzing the user’s task and past observations to determine the most appropriate resources to leverage. This dynamic integration enables the agent to extend its capabilities on-the-fly, accessing search engines, databases, code execution environments, and other utilities as needed to achieve its goals.
3. _Iterative Problem Solving_: The framework empowers agents to address complex tasks through an iterative cycle of thought, action, and observation. This feedback loop allows the agent to evaluate the outcomes of its actions, refine its strategies based on effectiveness, and plan subsequent steps accordingly. The iterative process is guided by user-provided prompts, which can include few-shot examples if needed to better illustrate the task. The LLM utilizes both current and historical observations to inform decision-making. Incorporating a memory component to trace the interaction history further enhances the agent’s adaptability and learning over time.

By adopting an integrated approach, ReAct agents overcome the limitations of traditional architectures, particularly in scenarios that require flexible reasoning and adaptive behavior. The fusion of reasoning and acting within an LLM-centric framework enables more sophisticated and context-aware problem-solving capabilities.

At the heart of a ReAct agent lies a LLM, such as Gemini Pro 1.5 as covered in this article. These models serve as the “ _brain_” of the agent and are capable of natural language understanding and generation, complex reasoning and decision-making, maintaining context and utilizing past observations, and generating structured outputs for tool selection and execution.

By framing the agent’s thought processes and actions as a sequence of natural language interactions, the ReAct framework allows agents to leverage the full power of LLMs. This approach facilitates navigation through complex problem-solving scenarios, enabling agents to adapt dynamically to new information and challenges while maintaining a coherent strategy toward achieving their objectives.

# I. Building a ReAct Agent with Gemini

Now that we’ve established the foundations, let’s explore the process of building a ReAct agent using Gemini as our LLM of choice.https://miro.medium.com/v2/resize:fit:1000/1*TQspcqCDuqzbv5bCiIG26w.png

## Overview of ReAct Agent Design

ReAct agents bring a new approach to AI by combining reasoning and action in a continuous cycle. As we learned previously, traditional AI systems separate decision-making from execution, whereas ReAct agents are designed to think and act in a loop. They take in a task, break it down, use external resources to gather information, and adapt based on what they learn. This makes them ideal for handling dynamic tasks that require constant adjustment and interaction with external environments. A 1000-foot overview of the agent architecture is shown above. Let’s break it down one by one:

1. **Input**: The agent starts by receiving a task in natural language. This task goes into the core language model (LLM), like Gemini Pro, which interprets what needs to be done. Thus, the LLM acts as the agent’s “brain,” setting the task in motion. The task is provided by the user. The goal here for the agent is to leverage the tools available at hand to solve the task.
2. **Reasoning**: The LLM analyzes the task and breaks it down into steps. It plans which actions to take and decides how to approach the problem based on available information and tools.
3. **Action with External Environments**: In our current setup, the agent has access to two main environments — Google Search and Wikipedia. Using specific tools connected via APIs, it can look up information on Google for the latest updates or gather facts from Wikipedia. Each action the agent takes depends on what it determines to be the best source for the task. By connecting to these external environments, the agent can quickly find relevant information or get additional context.
4. **Observation and Memory**: After executing each action, the agent observes the results and saves relevant information in its memory. This tracing allows it to keep track of past actions and build on previous observations, so it doesn’t repeat itself or lose context. Each new piece of information enriches the agent’s understanding of the task, making future actions more informed.
5. **Feedback Loop**: The agent cycles through reasoning, action, and observation steps continuously. Every time it gathers new information, it goes back to the reasoning stage, where the LLM considers the updated knowledge. This iterative loop helps the agent refine its approach and stay aligned with the task. The reasoning loop can be either constrained based on an end condition or capped by max iterations. Note that we leverage past observations here from the memory component.
6. **Response**: Finally, once it has gathered enough information and reached a solid understanding, the agent generates a response based on all the information it has collected and refined over multiple cycles. Again, this can be solely decided by the LLM or based on an end condition, or we may fail to arrive at an outcome given the constrained number of iterations.

By continuously interacting with external sources, storing observations, and revisiting its reasoning, the ReAct agent can tackle complex problems in a more flexible and adaptable way than traditional models. This design enables it to handle real-time information, respond to changing scenarios, and produce high-quality results with minimal human intervention.

Now that we have an understanding of the overall architecture we want to build, let’s start breaking it down into individual steps one by one.

## Step 1: Setting Up the Environment

For this exercise, we can choose between two types of search environments: Google Search and Wikipedia Search. Google Search is available via the SERP API, while Wikipedia has its own API. Our goal is to provide tools — Python functions that encapsulate API calling logic — which take a query (search term) and return the results from these environments.

For Google Search, we receive the top 10 results ranked and have the following information: rank, title, link, and snippet. A snippet in Google is a brief summary or description of a web page that appears in a search result. Snippets are designed to help users understand the content of a page and decide whether to click on it. While providing minimal information, snippets often pack crisp pieces of information that we need.

In the case of Wikipedia, we get the title and summary — a short paragraph about the topic we search for.

The tools for these searches are implemented in `serp.py` and `wiki.py` under the `src/tools` directory.

## Step 2: Defining the Agent Structure and Supporting Classes

To build our ReAct agent, we need to define several classes and structures that work together. Let’s explore each of these components:

**2.1 Enums and Custom Types**

First, we’ll define enums and custom types for use throughout our implementation. Enums will map the tool choices, while a custom type will capture observations during each iteration of tool usage. The complete code for the ReAct agent is located in `agent.py` under the `src/react` directory.

```
from enum import Enum, auto
from typing import Union, Callable

class Name(Enum):
    """Enumeration for tool names available to the agent."""
    WIKIPEDIA = auto()
    GOOGLE = auto()
    NONE = auto()
    def __str__(self) -> str:
        return self.name.lower()

Observation = Union[str, Exception]
```

**2.2 Message and Choice Models**

Next, we’ll define Pydantic models for messages and tool choices:

```
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str = Field(..., description="The role of the message sender.")
    content: str = Field(..., description="The content of the message.")

class Choice(BaseModel):
    name: Name = Field(..., description="The name of the tool chosen.")
    reason: str = Field(..., description="The reason for choosing this tool.")
```

**2.3 Tool Class**

The `Tool` class encapsulates the functionality of individual tools:

```
class Tool:
    def __init__(self, name: Name, func: Callable[[str], str]):
            self.name = name
            self.func = func

        def use(self, query: str) -> Observation:
            try:
                return self.func(query)
            except Exception as e:
                logger.error(f"Error executing tool {self.name}: {e}")
                return str(e)
```

**2.4 Agent Class**

Now, let’s look into the `Agent` class and its methods. This class defines the agent responsible for executing queries and handling tool interactions.

```
class Agent:
    def __init__(self, model: GenerativeModel) -> None:
        self.model = model
        self.tools: Dict[Name, Tool] = {}
        self.messages: List[Message] = []
        self.query = ""
        self.max_iterations = 5
        self.current_iteration = 0
        self.template = self.load_template()

    def load_template(self) -> str: ...

    def register(self, name: Name, func: Callable[[str], str]) -> None: ...

    def trace(self, role: str, content: str) -> None: ...

    def get_history(self) -> str: ...

    def think(self) -> None: ...

    def decide(self, response: str) -> None: ...

    def act(self, tool_name: Name, query: str) -> None: ...

    def execute(self, query: str) -> str: ...

    def ask_gemini(self, prompt: str) -> str: ...
```

These classes and structures work together to create a flexible and powerful ReAct agent. By organizing our code this way, we’ve built a modular and extensible framework for ReAct agents. This structure allows for easy addition of new tools, modification of the reasoning process, and integration with different LLMs or external services. Next, we’ll dive into the actual implementation of the most crucial component — the think-act-observe loop.

## Step 3: Implementing the Think-Act-Observe Loop

The core of the ReAct agent is its ability to think, act, and observe in an iterative loop. A high-level flow diagram below illustrates how a ReAct agent functions in this loop. The core pieces are the thinking (reasoning) phase, the acting phase (calling APIs and accessing the environment through tool use), and finally, the observation phase (collecting results). This cycle repeats, allowing the agent to improve and move towards a common goal set initially. In the following sections, we’ll examine each component of this loop in detail.https://miro.medium.com/v2/resize:fit:700/1*O8k5Oy65KYhkNFQ13PRpUA.png

**_Think_**

The `think` method forms the core of this ReAct agent's cognitive loop. It manages iteration tracking, dynamically constructs prompts using the current context, interacts with the Gemini language model, and logs the agent's thoughts. By calling the `decide` method based on the model's response, it initiates the next phase of reasoning or action.

```
def think(self) -> None:
    self.current_iteration += 1
    if self.current_iteration > self.max_iterations:
        logger.warning("Reached maximum iterations. Stopping.")
        return
    prompt = self.prompt_template.format(
        query=self.query,
        history=self.get_history(),
        tools=', '.join([str(tool.name) for tool in self.tools.values()])
    )
    response = self.ask_gemini(prompt)
    self.trace("assistant", f"Thought: {response}")
    self.decide(response)
```

**_Decide_**

The `decide` method is another pivotal component in the ReAct agent’s decision-making process, directly following the `think` component. It parses the JSON response from the language model (Gemini), determining whether to take an action using a specific tool or to provide a final answer. If an action is required, it calls the `act` method with the chosen tool and input. For final answers, it logs the result. The method includes error handling to manage parsing issues or unexpected response formats, reverting to the `think` method if problems arise. This approach ensures a robust cycle of thought and action, allowing the agent to navigate complex queries by seamlessly transitioning between reflection and tool utilization until a satisfactory answer is reached or the iteration limit is met.

```
def decide(self, response: str) -> None:
    try:
        parsed_response = json.loads(response.strip().strip('`').strip())
        if "action" in parsed_response:
            action = parsed_response["action"]
            tool_name = Name[action["name"].upper()]
            self.act(tool_name, action.get("input", self.query))
        elif "answer" in parsed_response:
            self.trace("assistant", f"Final Answer: {parsed_response['answer']}")
        else:
            raise ValueError("Invalid response format")
    except Exception as e:
        logger.error(f"Error processing response: {str(e)}")
        self.think()
```

**_Act_**

The `act` method completes the ReAct agent's cognitive cycle, following the `think` and `decide` phases. It executes the chosen tool based on Gemini decision, applying it to the current query or a refined input. Crucially, after every action, the method captures and logs the tool's output as an observation, integrating it into the agent's memory through the `trace` method. These observations, both current and accumulated from past iterations, are incorporated into the next `think` cycle, enabling the agent to reason based on a growing pool of information.

This iterative process of action, observation, and reflection allows the ReAct agent to build upon its knowledge incrementally, making increasingly informed decisions as it progresses through complex queries. By tying each action to logged observations, the agent maintains a comprehensive trace of its problem-solving journey, facilitating both effective reasoning and transparent decision-making.

```
def act(self, tool_name: Name, query: str) -> None:
    tool = self.tools.get(tool_name)
    if tool:
        result = tool.use(query)
        observation = f"Observation from {tool_name}: {result}"
        self.trace("system", observation)
        self.think()
    else:
        logger.error(f"No tool registered for choice: {tool_name}")
        self.think()
```

The complete code for the ReAct agent is located in `agent.py` under the `src/react` directory.

## Step 4: Crafting the Prompt Template

Crafting an effective ReAct prompt template is crucial after understanding the reasoning loop. This well-structured prompt guides the agent’s behavior and decision-making process, serving as the initial seed for interactions. A ReAct prompt typically includes four key components: i) the current query, ii) any previous reasoning steps and observations, iii) available tools, and iv) output format instructions. These instructions cover reasoning for tool selection, insights from past observations, and guidelines for concluding the reasoning loop.

Prompts can be either _zero-shot_, providing instructions without examples, or _few-shot_, which includes examples of reasoning and actions. For this exercise, we use a zero-shot approach. The prompt’s goal is to effectively teach the model to adopt ReAct-like behavior through carefully crafted instructions. It structures the agent’s thought process, encouraging it to break down problems, seek information, and take appropriate steps. By incorporating these elements, the prompt facilitates a structured approach to problem-solving, enabling the language model to navigate complex tasks more effectively.

The prompt template we used for building our ReAct agent is shown below. Note that except for the tool name within parentheses, everything else is agnostic to the genre and is not tied to anything specific we are building here.

```
You are a ReAct (Reasoning and Acting) agent tasked with answering the following query:

Query: {query}

Your goal is to reason about the query and decide on the best course of action to answer it accurately.

Previous reasoning steps and observations: {history}

Available tools: {tools}

Instructions:
1. Analyze the query, previous reasoning steps, and observations.
2. Decide on the next action: use a tool or provide a final answer.
3. Respond in the following JSON format:

If you need to use a tool:
{{
    "thought": "Your detailed reasoning about what to do next",
    "action": {{
        "name": "Tool name (wikipedia, google, or none)",
        "reason": "Explanation of why you chose this tool",
        "input": "Specific input for the tool, if different from the original query"
    }}
}}

If you have enough information to answer the query:
{{
    "thought": "Your final reasoning process",
    "answer": "Your comprehensive answer to the query"
}}

Remember:
- Be thorough in your reasoning.
- Use tools when you need more information.
- Always base your reasoning on the actual observations from tool use.
- If a tool returns no results or fails, acknowledge this and consider using a different tool or approach.
- Provide a final answer only when you're confident you have sufficient information.
- If you cannot find the necessary information after using available tools, admit that you don't have enough information to answer the query confidently.
```

You can find the prompt template in the shared repo [here](https://github.com/arunpshankar/react-from-scratch/blob/main/data/input/react.txt).

# II. Comparing Approaches: Traditional vs. ReAct Agentshttps://miro.medium.com/v2/resize:fit:700/1*8xI6wr1Pj32h5ylDU5Tr1w.jpeg

To appreciate the power and flexibility of the ReAct framework, comparing it to traditional methods for tool selection and query processing reveals key differences. Traditional systems often rely on predefined rules or pattern matching, such as routing biographical queries to Wikipedia and location queries to Google. ReAct, however, leverages a language model to dynamically analyze and select tools based on context, offering distinct advantages:

Traditional systems are inherently rigid, bound by preset logic, which limits flexibility. ReAct, by contrast, adapts fluidly to a wide range of queries, using its reasoning capability for multi-step decisions. This context-driven approach enables ReAct to maintain conversation history, track prior interactions, and craft more effective responses. Moreover, unlike traditional systems that require code updates to integrate new tools, ReAct seamlessly incorporates new capabilities using only tool descriptions. Its natural language understanding also enhances error handling, providing constructive suggestions and supporting complex, multi-step problem-solving.

For a simple example, consider `src/tools/manager.py`, which demonstrates a rule-driven approach to selecting tools based on query cues, without using an LLM like Gemini. Here, the manager routes queries based on prefixes: `/people` queries go to Wikipedia for biographical information, while `/location` queries go to Google for location-based searches. However, this structure forces us to pre-format queries and encode rigid rules, creating a bottleneck. To expand capabilities, we must add more rules and modify the code—a limitation that ReAct addresses effortlessly by dynamically interpreting context.

```
class Manager:
    """
    Manages tool registration, selection, and execution.
    """
    def __init__(self) -> None:
        self.tools: Dict[Name, Tool] = {}

    def register(self, name: Name, func: Callable[[str], str]) -> None:
        """
        Register a new tool.
        """
        self.tools[name] = Tool(name, func)

    def act(self, name: Name, query: str) -> Observation:
        """
        Retrieve and use a registered tool to process the given query.

        Parameters:
            name (Name): The name of the tool to use.
            query (str): The input query string.

        Returns:
            Observation: The result of the tool's execution or an error.
        """
        if name not in self.tools:
            raise ValueError(f"Tool {name} not registered")

        processed_query = query.split(' ', 1)[1] if ' ' in query else query
        return self.tools[name].use(processed_query)

    def choose(self, query: str) -> Choice:
        """
        Choose the appropriate tool based on the query prefix.
        """
        if query.startswith("/people"):
            return Choice(
                name=Name.WIKIPEDIA,
                reason="Query starts with /people, using Wikipedia for biographical information."
            )
        elif query.startswith("/location"):
            return Choice(
                name=Name.GOOGLE,
                reason="Query starts with /location, using Google for location-specific information."
            )
        else:
            raise ValueError("Unsupported query. Use /people or /location prefix.")
```

# III. ReAct in Action: Analyzing the Output Trace

To better understand how a ReAct agent operates, let’s examine some output traces from real interactions. We’ll look at three example queries with diverse complexity.

## **Example 1: Streamlined 3-Step Query**

Let’s start with the first example, which demonstrates the agent’s thought process as it answers a simple question requiring current world knowledge. The agent retrieves the latest information from its environment using provided tools. The query is: _Who is older, Cristiano Ronaldo or Lionel Messi?_

Imagine how a human would figure this out — using Google and Wikipedia. We’d either go to Wikipedia to get the birthdates of both players and compare them, or do the same with Google. However, we’d have to break the question into two parts, one for each player, and perform the same type of search for both before finally comparing to arrive at the answer. Let’s see how this process is emulated by our ReAct agent.

Our ReAct agent solves this correctly in three iterations, as shown below, giving us a peek into its thought process.

**Iteration 1: Initial Analysis and Tool Selection**

```
Thought: {
 "thought": "The query asks for a comparison of age between Cristiano Ronaldo and Lionel Messi. To determine who is older, I need to find their birth dates and compare.",
 "action": {
  "name": "wikipedia",
  "reason": "Wikipedia is a reliable source for biographical information, including birth dates of famous individuals.",
  "input": "Cristiano Ronaldo"
 }
}

Action: Using wikipedia tool
Observation: [Wikipedia summary for Cristiano Ronaldo, including his birth date: February 5, 1985]
```

In this first iteration, the agent:

1. Analyzes the query and identifies the need for birth dates.
2. Chooses the Wikipedia tool as a reliable source for this information.
3. Decides to look up Cristiano Ronaldo first.

**Iteration 2: Gathering Additional Information**

```
Thought: {
 "thought": "I have found Cristiano Ronaldo's birth date from Wikipedia: February 5, 1985. Now I need to find Lionel Messi's birth date to compare and determine who is older.",
 "action": {
  "name": "wikipedia",
  "reason": "Wikipedia is a reliable source for biographical information, and I need to find Lionel Messi's birth date to compare with Cristiano Ronaldo's.",
  "input": "Lionel Messi"
 }
}

Action: Using wikipedia tool
Observation: [Wikipedia summary for Lionel Messi, including his birth date: June 24, 1987]
```

In the second iteration, the agent:

1. Acknowledges the information gathered about Ronaldo.
2. Identifies the need for Messi’s birth date to complete the comparison.
3. Again chooses Wikipedia as the appropriate tool.

**Iteration 3: Final Analysis and Answer**

```
Thought: {
 "thought": "I now have Cristiano Ronaldo's birth date (February 5, 1985) and Lionel Messi's birth date (June 24, 1987) from Wikipedia. I can compare these dates to determine who is older.",
 "answer": "Cristiano Ronaldo is older than Lionel Messi. Ronaldo was born on February 5, 1985, while Messi was born on June 24, 1987."
}

Final Answer: Cristiano Ronaldo is older than Lionel Messi. Ronaldo was born on February 5, 1985, while Messi was born on June 24, 1987.
```

In the final iteration, the agent:

1. Synthesizes the information gathered from both Wikipedia searches.
2. Compares the birth dates to determine who is older.
3. Provides a clear, concise answer to the original query.

This trace demonstrates several key aspects of the ReAct framework: the agent uses iterative reasoning by breaking down the problem into steps, gathering necessary information before making a final decision; it utilizes tools effectively, such as the Wikipedia tool, to gather relevant data; it exhibits adaptive behavior by adjusting its actions based on the information it has and what it still needs; and its decision-making process is transparent, with each thought clearly articulated, showing how the agent arrives at its decisions and final answer. By examining this trace, we see how the ReAct agent combines reasoning and action to solve queries in a structured, transparent, and effective manner. You can find the complete trace of the agent [here](https://github.com/arunpshankar/react-from-scratch/blob/main/data/output/trace_1.txt).

## **Example 2:** Multi-Step Research Query

Now let’s test it with a more challenging query: “ _What is the average temperature in the capital city of the country where the current FIFA World Cup champion’s team captain was born?_”

This query contains several unknowns. To solve it, we need to break it down into subtasks and solve them in order. More importantly, we need to use a combination of tools at our disposal — both Wikipedia and Google — and know which tool to use when. Let’s take a look at how our agent solves this in four iterations.

**Iteration 1: Query Breakdown**

```
Thought: To answer this query, I need to break it down into smaller steps:
1. Find out who the current FIFA World Cup champion team captain is.
2. Find out where they were born.
3. Determine the capital city of that country.
4. Look up the average temperature in that capital city.

Action: Use Wikipedia to search for "2022 FIFA World Cup"
```

**Explanation**:

- **Thought**: The agent breaks down the query into steps: identifying the FIFA World Cup champion team captain, determining their birthplace, finding the capital of that country, and then looking up the average temperature in the capital city.
- **Action**: The agent begins by using Wikipedia to search for “2022 FIFA World Cup” to identify the champion team captain.
- **Observation**: The Wikipedia search reveals that Lionel Messi was the captain of the Argentinian team that won the 2022 FIFA World Cup.

**Iteration 2: Finding Messi’s Birthplace**

```
Thought: The Wikipedia summary tells us that Lionel Messi was the captain of the Argentinian team that won the 2022 FIFA World Cup. Now I need to find out where Lionel Messi was born.

Action: Use Wikipedia to search for "Lionel Messi"
```

**Explanation**:

- **Thought**: With Lionel Messi identified as the team captain, the agent now focuses on finding Messi’s birthplace to answer the next step of the query.
- **Action**: It searches Wikipedia for “Lionel Messi.”
- **Observation**: The Wikipedia page on Lionel Messi confirms that he was born in Rosario, Argentina.

**Iteration 3: Capital City and Temperature**

```
Thought: The Wikipedia summary on Lionel Messi states that he was born in Rosario, Argentina. To answer the query, I need to find the average temperature in Buenos Aires, the capital of Argentina.

Action: Use Google to search for "average temperature in Buenos Aires"
```

**Explanation**:

- **Thought**: Knowing Messi was born in Rosario, Argentina, the agent identifies Buenos Aires as the capital of Argentina, where it needs to find the average temperature.
- **Action**: The agent uses Google to search for the “average temperature in Buenos Aires.”
- **Observation**: Google results provide average temperature data for Buenos Aires, with several sources mentioning an average of around 18°C (64°F).

**Iteration 4: Final Answer**

```
Thought: I have gathered all the necessary information. Lionel Messi, captain of the 2022 FIFA World Cup winning team, was born in Argentina. The capital of Argentina is Buenos Aires. Google search results show that the average temperature in Buenos Aires varies throughout the year, but several sources mention an average temperature of around 18°C (64°F).

Answer: The average temperature in the capital city of the country where the current FIFA World Cup champion's team captain was born is approximately 18°C (64°F). This refers to Buenos Aires, Argentina, the birthplace of Lionel Messi.
```

**Explanation**:

- **Thought**: The agent has gathered all the necessary information: Lionel Messi, captain of the 2022 FIFA World Cup-winning team, was born in Argentina. The capital city is Buenos Aires, with an average temperature of approximately 18°C (64°F).
- **Action**: It constructs a response summarizing these findings.
- **Observation**: The agent provides a complete answer: “ _The average temperature in the capital city of the country where the current FIFA World Cup champion’s team captain was born is approximately 18°C (64°F), referring to Buenos Aires, Argentina, the birthplace of Lionel Messi._”

The complete agent trace can be accessed at this [location](https://github.com/arunpshankar/react-from-scratch/blob/main/data/output/trace_2.txt).

## Example 3: Comparative Analysis Query

Let’s consider a more complex query that requires advanced reasoning and critical thinking. This type of question doesn’t have a definitive answer, but rather involves exploring multiple sub-questions in a hierarchical manner. The goal is to compare and synthesize all the information gathered to arrive at a comprehensive conclusion. We’ll examine how our React-based agent performs in handling this type of sophisticated, multi-layered analysis.

Query: “ _What is the most common ingredient in the national dishes of the top 5 countries by GDP?_”

**Iteration 1: Initial Query and Breakdown**

```
{
  "thought": "To answer this query, I need to break it down into steps:\n1. Identify the top 5 countries by GDP.\n2. Determine the national dish of each country.\n3. Identify the ingredients in each national dish.\n4. Analyze the ingredients to find the most common one.\n\nI need to use a tool to get the information for steps 1 and 2.",
  "action": {
    "name": "google",
    "reason": "Google can provide a list of the top 5 countries by GDP and information about their national dishes.",
    "input": "top 5 countries by GDP and their national dishes"
  }
}
```

**Explanation**:

- **Thought**: The agent structures the query into four logical steps: identifying GDP rankings, discovering national dishes, listing ingredients, and analyzing commonalities.
- **Action**: It chooses Google to retrieve a list of countries with high GDP and corresponding national dishes.
- **Observation**: The tool returns general results on high GDP countries, but without a clear, unified list of national dishes.

**Iteration 2: Refining the Search**

```
{
  "thought": "The Google search results provided lists of countries by GDP and some information about national dishes, but not a clear and consistent pairing of the two for the top 5 countries. I need to refine my search to get more specific information.",
  "action": {
    "name": "google",
    "reason": "Google is still the best tool to find this information, but I need to be more specific in my query.",
    "input": "National dish of each of the top 5 countries by GDP"
  }
}
```

**Explanation**:

- **Thought**: Observing that the previous search didn’t yield a cohesive list, the agent decides to refine the query by explicitly requesting the “National dish of each of the top 5 countries by GDP.”
- **Action**: The agent queries Google again with this refined search.
- **Observation**: Google still provides general results about national dishes and GDP rankings but lacks a structured pairing of each country with its national dish.

**Iteration 3: Switching to Wikipedia for Specificity**

```
{
  "thought": "The Google search results still don't offer a clear and readily available list of national dishes for the top 5 economies. Wikipedia might have more detailed information about national dishes for each country. I can search for each country individually on Wikipedia and look for information about their national dish.",
  "action": {
    "name": "wikipedia",
    "reason": "Wikipedia is a more comprehensive source for information about specific topics like national dishes.",
    "input": "United States national dish"
  }
}
```

**Explanation**:

- **Thought**: The agent, realizing that Google results lack specificity, considers Wikipedia as a better source for individually searching each country’s national dish, starting with the United States.
- **Action**: It queries Wikipedia for “United States national dish.”
- **Observation**: Wikipedia does not yield a definitive “national dish” for the U.S., highlighting that this approach may not work universally across countries.

**Iteration 4: Expanding Scope to Popular Dishes**

```
{
  "thought": "The Wikipedia search for \"United States national dish\" returned no results. This suggests that a direct search for a single national dish for each country might not be the most effective approach. National dishes can be subjective and vary by region or cultural group. Instead of searching for a single dish, I should broaden my search to include popular or common dishes in each country. I can then analyze the ingredients of these dishes to identify any commonalities.",
  "action": {
    "name": "google",
    "reason": "Google is a better tool for broader searches and can provide a wider range of information about popular dishes in each country.",
    "input": "Popular dishes in the United States"
  }
}
```

**Explanation**:

- **Thought**: Noting Wikipedia’s limitations for national dishes, the agent broadens its scope to search for “popular dishes” instead. This broader approach should yield enough data on common foods, even if a single national dish is elusive.
- **Action**: It switches back to Google with the broader query “Popular dishes in the United States.”
- **Observation**: Google successfully provides a list of popular U.S. dishes, enabling the agent to compile a useful list of notable foods.

**Iteration 5: Continuing with Popular Dishes for Each Country**

```
{
  "thought": "The Google search provided a list of popular dishes in the United States. I can now repeat this process for the remaining top 4 countries by GDP (China, Japan, Germany, and India) to gather information about their popular dishes. Once I have a list of popular dishes for each country, I can then research common ingredients in those dishes to determine the most common one.",
  "action": {
    "name": "google",
    "reason": "Google is effective for finding lists of popular dishes in specific countries.",
    "input": "Popular dishes in China"
  }
}
```

**Explanation**:

- **Thought**: Satisfied with the U.S. results, the agent decides to repeat this search process for China, Japan, Germany, and India to collect popular dishes from each.
- **Action**: It queries Google with “Popular dishes in China” to gather similar data for China.
- **Observation**: Google provides relevant results for popular dishes in China, enabling the agent to proceed confidently with this approach for each remaining country.

**Iteration 6: Gathering Dishes for Japan**

```
{
  "thought": "I've successfully found lists of popular dishes for the United States and China. I need to continue this process for Japan, Germany, and India to get a comprehensive list of popular dishes for all five countries. Once I have those lists, I can research common ingredients and identify the most frequent one.",
  "action": {
    "name": "google",
    "reason": "Google is effective for finding lists of popular dishes in specific countries.",
    "input": "Popular dishes in Japan"
  }
}
```

**Explanation**:

- **Thought**: With data on the U.S. and China, the agent moves on to Japan, maintaining the plan to compile popular dishes for each of the top 5 GDP countries.
- **Action**: It uses Google to search “Popular dishes in Japan.”
- **Observation**: Google successfully provides a list of popular Japanese dishes, further supporting this method for gathering culinary data country-by-country.

**Iteration 7: Researching Germany’s Popular Dishes**

```
{
  "thought": "I now have lists of popular dishes for the United States, China, and Japan. I need to find similar lists for Germany and India to complete my data on the top 5 economies. I will use Google again as it has been effective for this purpose.",
  "action": {
    "name": "google",
    "reason": "Google is effective for finding lists of popular dishes in specific countries.",
    "input": "Popular dishes in Germany"
  }
}
```

**Explanation**:

- **Thought**: With data on three countries, the agent turns to Germany as it continues gathering popular dishes for each of the top 5 GDP countries.
- **Action**: It queries Google with “Popular dishes in Germany.”
- **Observation**: Google returns a comprehensive list of German dishes, allowing the agent to proceed confidently with its plan.

**Iteration 8: Compiling India’s Popular Dishes**

```
{
  "thought": "I have successfully gathered lists of popular dishes for the United States, China, Japan, and Germany. I need to find a similar list for India to complete my data on the top 5 economies. I will use Google again as it has been effective for this purpose.",
  "action": {
    "name": "google",
    "reason": "Google is effective for finding lists of popular dishes in specific countries.",
    "input": "Popular dishes in India"
  }
}
```

**Explanation**:

- **Thought**: With lists for four countries completed, India is the final country needed to round out the data set for the top five GDP countries.
- **Action**: The agent queries Google with “Popular dishes in India.”
- **Observation**: Google successfully provides a list of popular Indian dishes, completing the agent’s data gathering.

**Iteration 9: Planning to Analyze Ingredients**

```
{
  "thought": "I have now gathered lists of popular dishes for all five countries: the United States, China, Japan, Germany, and India. My next step is to analyze these lists and identify common ingredients across the dishes. However, this task is too complex to perform without a tool. I need a way to efficiently research the ingredients of multiple dishes across different cuisines. Since there isn't a specific tool for this purpose, I will try using Google to research the ingredients of each dish individually and note any recurring patterns.",
  "action": {
    "name": "google",
    "reason": "While there isn't a perfect tool for this task, Google can be used to research the ingredients of each dish individually. I will manually analyze the results for commonalities.",
  "input": "Ingredients in [Dish Name]"
 }
}
```

**Explanation**:

- **Thought**: With popular dishes identified for each country, the agent now plans to analyze the ingredients for these dishes to spot commonalities. Recognizing that this analysis is complex, the agent decides to look up each dish individually.
- **Action**: The agent begins a series of Google searches focused on identifying ingredients in each specific dish.
- **Observation**: As Google yields results for each dish’s ingredients, the agent proceeds to manually compare and analyze these ingredients to find any common patterns.

**Iteration 10: Realizing Complexity and Focusing on Common Themes**

```
{
  "thought": "The last Google search returned a list of recipe websites and ingredient resources, but not specific ingredient lists for popular dishes. It seems that finding a single most common ingredient across such diverse cuisines will be difficult. Many ingredients like salt, sugar, and cooking oil are likely used in almost every dish but are not very insightful. Additionally, directly comparing ingredient frequency across vastly different cooking styles and cultural preferences might not be the best way to approach the query. \n\nInstead of focusing on a single ingredient, I can aim for a more general answer about common ingredient types or flavor profiles. To do this, I will research common ingredients and cooking styles for each country's cuisine and look for overarching themes or similarities.",
  "action": {
    "name": "google",
    "reason": "Google can provide information about common ingredients and cooking styles for each country's cuisine.",
    "input": "Common ingredients in [Country] cuisine"
 }
}
```

**Explanation**:

- **Thought**: The agent realizes that pinpointing a single common ingredient is impractical due to the diversity in cooking styles and ingredient usage. Instead, it shifts focus to identifying broad ingredient types or flavor profiles that are common across these cuisines.
- **Action**: It adjusts its queries to gather general data on common ingredients and cooking styles for each country.
- **Observation**: Google provides information about the general culinary profiles and common ingredients, allowing the agent to summarize overarching themes instead of finding a single ingredient.

**Iteration 11: Gathering U.S. Ingredient Data**

```
{
  "thought": "The previous attempts to find a single most common ingredient across the national dishes of the top 5 GDP countries highlighted the difficulty of directly comparing ingredients across diverse culinary traditions.  It's clear that a more nuanced approach is needed. Instead of looking for a single ingredient, I can provide a more insightful answer by identifying common ingredient types or flavor profiles prevalent in these countries' cuisines. I will research common ingredients and cooking styles for each country and then synthesize the information to identify overarching themes.",
  "action": {
    "name": "google",
    "reason": "Google can provide information about common ingredients and cooking styles for each country's cuisine.",
    "input": "Common ingredients in United States cuisine"
 }
}
```

**Explanation**:

- **Thought**: Having determined that ingredient themes are more insightful, the agent begins by collecting information on common ingredients in American cuisine, expecting to identify regional or staple ingredients that define U.S. culinary styles.
- **Action**: The agent searches Google with “Common ingredients in United States cuisine.”
- **Observation**: Google provides detailed information on core American ingredients, including wheat, corn, and various proteins, underscoring the diversity within American food traditions.

**Iteration 12: Researching Chinese Cuisine Ingredients**

```
{
  "thought": "The Google search provided insights into common ingredients in US cuisine, highlighting the diversity and regional variations. To effectively compare culinary trends across the top 5 GDP countries, I need to gather similar information about common ingredients and cooking styles for China, Japan, Germany, and India. Once I have a better understanding of each country's culinary landscape, I can then analyze for common themes or patterns.",
  "action": {
    "name": "google",
    "reason": "Google is effective for gathering information about culinary traditions and common ingredients in different countries.",
    "input": "Common ingredients in Chinese cuisine"
 }
}
```

**Explanation**:

- **Thought**: With U.S. data gathered, the agent continues with China, aiming to identify common ingredients and cooking methods that characterize Chinese cuisine.
- **Action**: It queries Google with “Common ingredients in Chinese cuisine.”
- **Observation**: Google reveals staple Chinese ingredients such as soy sauce, ginger, and garlic, giving the agent a foundation of core components in Chinese cooking.

**Iteration 13: Researching Japanese Cuisine Ingredients**

```
{
  "thought": "I've started gathering information about common ingredients and cooking styles in the cuisines of the top 5 GDP countries. I have data for the United States and China, but I still need to research Japan, Germany, and India. Once I have a better understanding of the culinary landscape for all five countries, I can analyze the information for common themes or patterns and provide a more insightful answer than simply identifying a single ingredient.",
  "action": {
    "name": "google",
    "reason": "Google is effective for gathering information about culinary traditions and common ingredients in different countries.",
    "input": "Common ingredients in Japanese cuisine"
 }
}
```

**Explanation**:

- **Thought**: After obtaining data on U.S. and Chinese cuisines, the agent moves to Japan to further capture common ingredient patterns and culinary themes.
- **Action**: The agent uses Google to search for “Common ingredients in Japanese cuisine.”
- **Observation**: Google provides a list of Japanese staples, such as soy sauce, miso, and rice, which helps the agent outline fundamental elements of Japanese cuisine.

**Iteration 14: Researching German Cuisine Ingredients**

```
{
  "thought": "I have gathered information about common ingredients and cooking styles for three out of the five countries: the United States, China, and Japan. To make a comprehensive analysis and identify overarching themes across the top 5 GDP countries' cuisines, I need to gather similar information for Germany and India.  I will continue using Google as it has been effective for this research.",
  "action": {
    "name": "google",
    "reason": "Google is effective for gathering information about culinary traditions and common ingredients in different countries.",
    "input": "Common ingredients in German cuisine"
 }
}
```

**Explanation**:

- **Thought**: With information on three countries, the agent turns to Germany, aiming to gather data on the essential ingredients and cooking styles characteristic of German food.
- **Action**: It searches Google for “Common ingredients in German cuisine.”
- **Observation**: Google lists ingredients like potatoes, sausages, and cabbage, helping the agent understand the staples of German cuisine.

**Iteration 15: Researching Indian Cuisine Ingredients**

```
{
  "thought": "I have gathered information about common ingredients and cooking styles for four out of the five countries: the United States, China, Japan, and Germany. To make a comprehensive analysis and identify overarching themes across the top 5 GDP countries' cuisines, I need to gather similar information for India.  I will continue using Google as it has been effective for this research.",
  "action": {
    "name": "google",
    "reason": "Google is effective for gathering information about culinary traditions and common ingredients in different countries.",
    "input": "Common ingredients in Indian cuisine"
 }
}
```

**Explanation**:

- **Thought**: Having researched the first four countries, the agent now focuses on India, the final country, to complete its dataset of common ingredients across the top five GDP nations.
- **Action**: The agent queries Google with “Common ingredients in Indian cuisine.”
- **Observation**: Google provides typical Indian ingredients, including spices like turmeric and cumin as well as staples like rice and lentils, rounding out the agent’s ingredient data for all five countries.

**Iteration 16: Summary of Outcomes**

```
{
  "thought": "I have successfully gathered information about common ingredients and cooking styles for all five countries: the United States, China, Japan, Germany, and India. Now I can analyze this information to identify overarching themes or similarities in ingredient types or flavor profiles. I don't need to use a tool for this step, as I can reason based on the information I've already collected.",
  "answer": "It's challenging to pinpoint a single most common ingredient across the national dishes of the top 5 GDP countries due to their diverse culinary traditions. However, analyzing common ingredients and cooking styles reveals some overarching themes:\n\n* **Starchy Staples:** All five countries heavily feature starchy staples in their cuisines. In the US, it's wheat (bread, pasta) and corn. China relies on rice and noodles. Japan emphasizes rice and noodles (ramen, udon). Germany favors potatoes and bread. India heavily uses rice and wheat (naan, roti).\n* **Protein Variety:** While specific proteins vary, all five countries incorporate a variety of protein sources. The US includes beef, pork, and chicken. China features pork, chicken, duck, and seafood. Japan utilizes seafood, chicken, and pork. Germany commonly uses pork, beef, and poultry. India, with diverse dietary preferences, incorporates chicken, seafood, lamb, and plant-based proteins like lentils.\n* **Flavor Profiles:** While specific flavor combinations are diverse, some common themes emerge. The US leans towards savory and sweet flavors. China often balances sweet, sour, spicy, and salty. Japan emphasizes umami, often achieved through ingredients like soy sauce and dashi. Germany favors hearty and savory flavors. India is renowned for its diverse and aromatic spice blends, often combining heat, sweetness, sourness, and earthiness.\n\nIn conclusion, while a single most common ingredient is difficult to define, these culinary powerhouses share a reliance on starchy staples and diverse protein sources, while expressing unique flavor profiles through their individual culinary traditions."
}
```

**Explanation**:

- **Thought**: With comprehensive data on common ingredients for the U.S., China, Japan, Germany, and India, the agent is ready to synthesize overarching themes or commonalities in ingredient types and culinary profiles across these cuisines.
- **Action**: No additional search is required; the agent begins to analyze and formulate a response based on the data collected.
- **Observation**: The agent concludes that, while a single common ingredient is impractical to define, there are shared themes among the cuisines: reliance on starchy staples, diverse protein sources, and unique flavor profiles.

You can find the full agent trace [here](https://github.com/arunpshankar/react-from-scratch/blob/main/data/output/trace_3.txt). All code files associated with everything we covered here are available in this [GitHub repository](https://github.com/arunpshankar/react-from-scratch/tree/main).

# Future Directions

The ReAct framework opens up numerous possibilities for enhanced functionality and adaptability in agent-based systems. Future developments could focus on integrating the ability to process diverse data types such as images, audio, and video, enabling agents to interpret a broader spectrum of information for richer, context-aware decisions. Organizing agents into layered hierarchies, where primary agents delegate specialized tasks to sub-agents, would improve efficiency and task segmentation. Additionally, empowering agents to collaborate by sharing observations, tools, and resources would amplify insights and support cohesive decision-making in complex, multi-perspective environments. Higher-level agents can dynamically guide and coordinate other agents, orchestrating actions across a multi-agent setup to handle complex, multi-step tasks efficiently.

In future posts, we’ll explore a multi-agent scenario where a single steering agent interacts with multiple sub-agents, each fulfilling distinct tasks and communicating their findings back for a cohesive outcome. This will extend our current exercise, building a foundation for scalable, collaborative agent networks.

</details>

<details>
<summary>function-calling-with-the-gemini-api-google-ai-for-developer</summary>

# Function calling with the Gemini API

Function calling lets you connect models to external tools and APIs.
Instead of generating text responses, the model determines when to call specific
functions and provides the necessary parameters to execute real-world actions.
This allows the model to act as a bridge between natural language and real-world
actions and data. Function calling has 3 primary use cases:

- **Augment Knowledge:** Access information from external sources like
databases, APIs, and knowledge bases.
- **Extend Capabilities:** Use external tools to perform computations and
extend the limitations of the model, such as using a calculator or creating
charts.
- **Take Actions:** Interact with external systems using APIs, such as
scheduling appointments, creating invoices, sending emails, or controlling
smart home devices.

Get WeatherSchedule MeetingCreate Chart

## How function calling workshttps://ai.google.dev/static/gemini-api/docs/images/function-calling-overview.png

Function calling involves a structured interaction between your application, the
model, and external functions. Here's a breakdown of the process:

1. **Define Function Declaration:** Define the function declaration in your
application code. Function Declarations describe the function's name,
parameters, and purpose to the model.
2. **Call LLM with function declarations:** Send user prompt along with the
function declaration(s) to the model. It analyzes the request and determines
if a function call would be helpful. If so, it responds with a structured
JSON object.
3. **Execute Function Code (Your Responsibility):** The Model _does not_
execute the function itself. It's your application's responsibility to
process the response and check for Function Call, if

   - **Yes**: Extract the name and args of the function and execute the
     corresponding function in your application.
   - **No:** The model has provided a direct text response to the prompt
     (this flow is less emphasized in the example but is a possible outcome).
4. **Create User friendly response:** If a function was executed, capture the
result and send it back to the model in a subsequent turn of the
conversation. It will use the result to generate a final, user-friendly
response that incorporates the information from the function call.

This process can be repeated over multiple turns, allowing for complex
interactions and workflows. The model also supports calling multiple functions
in a single turn ( [parallel function\
calling](https://ai.google.dev/gemini-api/docs/function-calling#parallel_function_calling)) and in
sequence ( [compositional function\
calling](https://ai.google.dev/gemini-api/docs/function-calling#compositional_function_calling)).

### Step 1: Define a function declaration

Define a function and its declaration within your application code that allows
users to set light values and make an API request. This function could call
external services or APIs.

[Python](https://ai.google.dev/gemini-api/docs/function-calling#python)[JavaScript](https://ai.google.dev/gemini-api/docs/function-calling#javascript)More

```
# Define a function that the model can call to control smart lights
set_light_values_declaration = {
    "name": "set_light_values",
    "description": "Sets the brightness and color temperature of a light.",
    "parameters": {
        "type": "object",
        "properties": {
            "brightness": {
                "type": "integer",
                "description": "Light level from 0 to 100. Zero is off and 100 is full brightness",
            },
            "color_temp": {
                "type": "string",
                "enum": ["daylight", "cool", "warm"],
                "description": "Color temperature of the light fixture, which can be `daylight`, `cool` or `warm`.",
            },
        },
        "required": ["brightness", "color_temp"],
    },
}

# This is the actual function that would be called based on the model's suggestion
def set_light_values(brightness: int, color_temp: str) -> dict[str, int | str]:
    """Set the brightness and color temperature of a room light. (mock API).

    Args:
        brightness: Light level from 0 to 100. Zero is off and 100 is full brightness
        color_temp: Color temperature of the light fixture, which can be `daylight`, `cool` or `warm`.

    Returns:
        A dictionary containing the set brightness and color temperature.
    """
    return {"brightness": brightness, "colorTemperature": color_temp}

```

```
import { Type } from '@google/genai';

// Define a function that the model can call to control smart lights
const setLightValuesFunctionDeclaration = {
  name: 'set_light_values',
  description: 'Sets the brightness and color temperature of a light.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      brightness: {
        type: Type.NUMBER,
        description: 'Light level from 0 to 100. Zero is off and 100 is full brightness',
      },
      color_temp: {
        type: Type.STRING,
        enum: ['daylight', 'cool', 'warm'],
        description: 'Color temperature of the light fixture, which can be `daylight`, `cool` or `warm`.',
      },
    },
    required: ['brightness', 'color_temp'],
  },
};

/**

*   Set the brightness and color temperature of a room light. (mock API)
*   @param {number} brightness - Light level from 0 to 100. Zero is off and 100 is full brightness
*   @param {string} color_temp - Color temperature of the light fixture, which can be `daylight`, `cool` or `warm`.
*   @return {Object} A dictionary containing the set brightness and color temperature.
*/
function setLightValues(brightness, color_temp) {
  return {
    brightness: brightness,
    colorTemperature: color_temp
  };
}

```

### Step 2: Call the model with function declarations

Once you have defined your function declarations, you can prompt the model to
use them. It analyzes the prompt and function declarations and decides whether
to respond directly or to call a function. If a function is called, the response
object will contain a function call suggestion.

[Python](https://ai.google.dev/gemini-api/docs/function-calling#python)[JavaScript](https://ai.google.dev/gemini-api/docs/function-calling#javascript)More

```
from google.genai import types

# Configure the client and tools
client = genai.Client()
tools = types.Tool(function_declarations=[set_light_values_declaration])
config = types.GenerateContentConfig(tools=[tools])

# Define user prompt
contents = [\
    types.Content(\
        role="user", parts=[types.Part(text="Turn the lights down to a romantic level")]\
    )\
]

# Send request with function declarations
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=contents
    config=config,
)

print(response.candidates[0].content.parts[0].function_call)

```

```
import { GoogleGenAI } from '@google/genai';

// Generation config with function declaration
const config = {
  tools: [{\
    functionDeclarations: [setLightValuesFunctionDeclaration]\
  }]
};

// Configure the client
const ai = new GoogleGenAI({});

// Define user prompt
const contents = [\
  {\
    role: 'user',\
    parts: [{ text: 'Turn the lights down to a romantic level' }]\
  }\
];

// Send request with function declarations
const response = await ai.models.generateContent({
  model: 'gemini-2.5-flash',
  contents: contents,
  config: config
});

console.log(response.functionCalls[0]);

```

The model then returns a `functionCall` object in an OpenAPI compatible
schema specifying how to call one or more of the declared functions in order to
respond to the user's question.

[Python](https://ai.google.dev/gemini-api/docs/function-calling#python)[JavaScript](https://ai.google.dev/gemini-api/docs/function-calling#javascript)More

```
id=None args={'color_temp': 'warm', 'brightness': 25} name='set_light_values'

```

```
{
  name: 'set_light_values',
  args: { brightness: 25, color_temp: 'warm' }
}

```

### Step 3: Execute set\_light\_values function code

Extract the function call details from the model's response, parse the arguments
, and execute the `set_light_values` function.

[Python](https://ai.google.dev/gemini-api/docs/function-calling#python)[JavaScript](https://ai.google.dev/gemini-api/docs/function-calling#javascript)More

```
# Extract tool call details, it may not be in the first part.
tool_call = response.candidates[0].content.parts[0].function_call

if tool_call.name == "set_light_values":
    result = set_light_values(**tool_call.args)
    print(f"Function execution result: {result}")

```

```
// Extract tool call details
const tool_call = response.functionCalls[0]

let result;
if (tool_call.name === 'set_light_values') {
  result = setLightValues(tool_call.args.brightness, tool_call.args.color_temp);
  console.log(`Function execution result: ${JSON.stringify(result)}`);
}

```

### Step 4: Create user friendly response with function result and call the model again

Finally, send the result of the function execution back to the model so it can
incorporate this information into its final response to the user.

[Python](https://ai.google.dev/gemini-api/docs/function-calling#python)[JavaScript](https://ai.google.dev/gemini-api/docs/function-calling#javascript)More

```
# Create a function response part
function_response_part = types.Part.from_function_response(
    name=tool_call.name,
    response={"result": result},
)

# Append function call and result of the function execution to contents
contents.append(response.candidates[0].content) # Append the content from the model's response.
contents.append(types.Content(role="user", parts=[function_response_part])) # Append the function response

final_response = client.models.generate_content(
    model="gemini-2.5-flash",
    config=config,
    contents=contents,
)

print(final_response.text)

```

```
// Create a function response part
const function_response_part = {
  name: tool_call.name,
  response: { result }
}

// Append function call and result of the function execution to contents
contents.push(response.candidates[0].content);
contents.push({ role: 'user', parts: [{ functionResponse: function_response_part }] });

// Get the final response from the model
const final_response = await ai.models.generateContent({
  model: 'gemini-2.5-flash',
  contents: contents,
  config: config
});

console.log(final_response.text);

```

This completes the function calling flow. The model successfully used the
`set_light_values` function to perform the request action of the user.

## Function declarations

When you implement function calling in a prompt, you create a `tools` object,
which contains one or more `function declarations`. You define functions using
JSON, specifically with a [select subset](https://ai.google.dev/api/caching#Schema)
of the [OpenAPI schema](https://spec.openapis.org/oas/v3.0.3#schemaw) format. A
single function declaration can include the following parameters:

- `name` (string): A unique name for the function ( `get_weather_forecast`,
`send_email`). Use descriptive names without spaces or special characters
(use underscores or camelCase).
- `description` (string): A clear and detailed explanation of the function's
purpose and capabilities. This is crucial for the model to understand when
to use the function. Be specific and provide examples if helpful ("Finds
theaters based on location and optionally movie title which is currently
playing in theaters.").
- `parameters` (object): Defines the input parameters the function
expects.

  - `type` (string): Specifies the overall data type, such as `object`.
  - `properties` (object): Lists individual parameters, each with:

    - `type` (string): The data type of the parameter, such as `string`,
      `integer`, `boolean, array`.
    - `description` (string): A description of the parameter's purpose and
      format. Provide examples and constraints ("The city and state,
      e.g., 'San Francisco, CA' or a zip code e.g., '95616'.").
    - `enum` (array, optional): If the parameter values are from a fixed
      set, use "enum" to list the allowed values instead of just describing
      them in the description. This improves accuracy ("enum":
      \["daylight", "cool", "warm"\]).
  - `required` (array): An array of strings listing the parameter names that
    are mandatory for the function to operate.

You can also construct FunctionDeclarations from Python functions directly using
`types.FunctionDeclaration.from_callable(client=client, callable=your_function)`.

## Function calling with thinking

Enabling " [thinking](https://ai.google.dev/gemini-api/docs/thinking)" can improve function call
performance by allowing the model to reason through a request before suggesting
function calls. The Gemini API is stateless, the model's reasoning context will
be lost between turns in a multi-turn conversation. To preserve this context,
you can use thought signatures. A thought signature is an encrypted
representation of the model's internal thought process that you pass back to
the model on subsequent turns.

The [standard pattern for multi-turn tool](https://ai.google.dev/gemini-api/docs/function-calling?example=weather#step-4)
use is to append the model's complete previous response to the conversation
history. The `content` object includes the `thought_signatures` automatically.
If you follow this pattern **No code changes are required**.

### Manually managing thought signatures

If you modify the conversation history manually—instead of sending the complete previous response and want to benefit from thinking you must correctly handle the `thought_signature` included in the model's turn.

Follow these rules to ensure the model's context is preserved:

- Always send the `thought_signature` back to the model inside its original `Part`.
- Don't merge a `Part` containing a signature with one that does not. This breaks the positional context of the thought.
- Don't combine two `Parts` that both contain signatures, as the signature strings cannot be merged.

### Inspecting Thought Signatures

While not necessary for implementation, you can inspect the response to see the
`thought_signature` for debugging or educational purposes.

[Python](https://ai.google.dev/gemini-api/docs/function-calling#python)[JavaScript](https://ai.google.dev/gemini-api/docs/function-calling#javascript)More

```
import base64
# After receiving a response from a model with thinking enabled
# response = client.models.generate_content(...)

# The signature is attached to the response part containing the function call
part = response.candidates[0].content.parts[0]
if part.thought_signature:
  print(base64.b64encode(part.thought_signature).decode("utf-8"))

```

```
// After receiving a response from a model with thinking enabled
// const response = await ai.models.generateContent(...)

// The signature is attached to the response part containing the function call
const part = response.candidates[0].content.parts[0];
if (part.thoughtSignature) {
  console.log(part.thoughtSignature);
}

```

Learn more about limitations and usage of thought signatures, and about thinking
models in general, on the [Thinking](https://ai.google.dev/gemini-api/docs/thinking#signatures) page.

## Parallel function calling

In addition to single turn function calling, you can also call multiple
functions at once. Parallel function calling lets you execute multiple functions
at once and is used when the functions are not dependent on each other. This is
useful in scenarios like gathering data from multiple independent sources, such
as retrieving customer details from different databases or checking inventory
levels across various warehouses or performing multiple actions such as
converting your apartment into a disco.

[Python](https://ai.google.dev/gemini-api/docs/function-calling#python)[JavaScript](https://ai.google.dev/gemini-api/docs/function-calling#javascript)More

```
power_disco_ball = {
    "name": "power_disco_ball",
    "description": "Powers the spinning disco ball.",
    "parameters": {
        "type": "object",
        "properties": {
            "power": {
                "type": "boolean",
                "description": "Whether to turn the disco ball on or off.",
            }
        },
        "required": ["power"],
    },
}

start_music = {
    "name": "start_music",
    "description": "Play some music matching the specified parameters.",
    "parameters": {
        "type": "object",
        "properties": {
            "energetic": {
                "type": "boolean",
                "description": "Whether the music is energetic or not.",
            },
            "loud": {
                "type": "boolean",
                "description": "Whether the music is loud or not.",
            },
        },
        "required": ["energetic", "loud"],
    },
}

dim_lights = {
    "name": "dim_lights",
    "description": "Dim the lights.",
    "parameters": {
        "type": "object",
        "properties": {
            "brightness": {
                "type": "number",
                "description": "The brightness of the lights, 0.0 is off, 1.0 is full.",
            }
        },
        "required": ["brightness"],
    },
}

```

```
import { Type } from '@google/genai';

const powerDiscoBall = {
  name: 'power_disco_ball',
  description: 'Powers the spinning disco ball.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      power: {
        type: Type.BOOLEAN,
        description: 'Whether to turn the disco ball on or off.'
      }
    },
    required: ['power']
  }
};

const startMusic = {
  name: 'start_music',
  description: 'Play some music matching the specified parameters.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      energetic: {
        type: Type.BOOLEAN,
        description: 'Whether the music is energetic or not.'
      },
      loud: {
        type: Type.BOOLEAN,
        description: 'Whether the music is loud or not.'
      }
    },
    required: ['energetic', 'loud']
  }
};

const dimLights = {
  name: 'dim_lights',
  description: 'Dim the lights.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      brightness: {
        type: Type.NUMBER,
        description: 'The brightness of the lights, 0.0 is off, 1.0 is full.'
      }
    },
    required: ['brightness']
  }
};

```

Configure the function calling mode to allow using all of the specified tools.
To learn more, you can read about
[configuring function calling](https://ai.google.dev/gemini-api/docs/function-calling#function_calling_modes).

[Python](https://ai.google.dev/gemini-api/docs/function-calling#python)[JavaScript](https://ai.google.dev/gemini-api/docs/function-calling#javascript)More

```
from google import genai
from google.genai import types

# Configure the client and tools
client = genai.Client()
house_tools = [\
    types.Tool(function_declarations=[power_disco_ball, start_music, dim_lights])\
]
config = types.GenerateContentConfig(
    tools=house_tools,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(
        disable=True
    ),
    # Force the model to call 'any' function, instead of chatting.
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode='ANY')
    ),
)

chat = client.chats.create(model="gemini-2.5-flash", config=config)
response = chat.send_message("Turn this place into a party!")

# Print out each of the function calls requested from this single call
print("Example 1: Forced function calling")
for fn in response.function_calls:
    args = ", ".join(f"{key}={val}" for key, val in fn.args.items())
    print(f"{fn.name}({args})")

```

```
import { GoogleGenAI } from '@google/genai';

// Set up function declarations
const houseFns = [powerDiscoBall, startMusic, dimLights];

const config = {
    tools: [{\
        functionDeclarations: houseFns\
    }],
    // Force the model to call 'any' function, instead of chatting.
    toolConfig: {
        functionCallingConfig: {
            mode: 'any'
        }
    }
};

// Configure the client
const ai = new GoogleGenAI({});

// Create a chat session
const chat = ai.chats.create({
    model: 'gemini-2.5-flash',
    config: config
});
const response = await chat.sendMessage({message: 'Turn this place into a party!'});

// Print out each of the function calls requested from this single call
console.log("Example 1: Forced function calling");
for (const fn of response.functionCalls) {
    const args = Object.entries(fn.args)
        .map(([key, val]) => `${key}=${val}`)
        .join(', ');
    console.log(`${fn.name}(${args})`);
}

```

Each of the printed results reflects a single function call that the model has
requested. To send the results back, include the responses in the same order as
they were requested.

The Python SDK supports [automatic function calling](https://ai.google.dev/gemini-api/docs/function-calling#automatic_function_calling_python_only),
which automatically converts Python functions to declarations, handles the
function call execution and response cycle for you. Following is an example for
the disco use case.

[Python](https://ai.google.dev/gemini-api/docs/function-calling#python)More

```
from google import genai
from google.genai import types

# Actual function implementations
def power_disco_ball_impl(power: bool) -> dict:
    """Powers the spinning disco ball.

    Args:
        power: Whether to turn the disco ball on or off.

    Returns:
        A status dictionary indicating the current state.
    """
    return {"status": f"Disco ball powered {'on' if power else 'off'}"}

def start_music_impl(energetic: bool, loud: bool) -> dict:
    """Play some music matching the specified parameters.

    Args:
        energetic: Whether the music is energetic or not.
        loud: Whether the music is loud or not.

    Returns:
        A dictionary containing the music settings.
    """
    music_type = "energetic" if energetic else "chill"
    volume = "loud" if loud else "quiet"
    return {"music_type": music_type, "volume": volume}

def dim_lights_impl(brightness: float) -> dict:
    """Dim the lights.

    Args:
        brightness: The brightness of the lights, 0.0 is off, 1.0 is full.

    Returns:
        A dictionary containing the new brightness setting.
    """
    return {"brightness": brightness}

# Configure the client
client = genai.Client()
config = types.GenerateContentConfig(
    tools=[power_disco_ball_impl, start_music_impl, dim_lights_impl]
)

# Make the request
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Do everything you need to this place into party!",
    config=config,
)

print("\nExample 2: Automatic function calling")
print(response.text)
# I've turned on the disco ball, started playing loud and energetic music, and dimmed the lights to 50% brightness. Let's get this party started!

```

## Compositional function calling

Compositional or sequential function calling allows Gemini to chain multiple
function calls together to fulfill a complex request. For example, to answer
"Get the temperature in my current location", the Gemini API might first invoke
a `get_current_location()` function followed by a `get_weather()` function that
takes the location as a parameter.

The following example demonstrates how to implement compositional function
calling using the Python SDK and automatic function calling.

[Python](https://ai.google.dev/gemini-api/docs/function-calling#python)[JavaScript](https://ai.google.dev/gemini-api/docs/function-calling#javascript)More

This example uses the automatic function calling feature of the
`google-genai` Python SDK. The SDK automatically converts the Python
functions to the required schema, executes the function calls when requested
by the model, and sends the results back to the model to complete the task.

```
import os
from google import genai
from google.genai import types

# Example Functions
def get_weather_forecast(location: str) -> dict:
    """Gets the current weather temperature for a given location."""
    print(f"Tool Call: get_weather_forecast(location={location})")
    # TODO: Make API call
    print("Tool Response: {'temperature': 25, 'unit': 'celsius'}")
    return {"temperature": 25, "unit": "celsius"}  # Dummy response

def set_thermostat_temperature(temperature: int) -> dict:
    """Sets the thermostat to a desired temperature."""
    print(f"Tool Call: set_thermostat_temperature(temperature={temperature})")
    # TODO: Interact with a thermostat API
    print("Tool Response: {'status': 'success'}")
    return {"status": "success"}

# Configure the client and model
client = genai.Client()
config = types.GenerateContentConfig(
    tools=[get_weather_forecast, set_thermostat_temperature]
)

# Make the request
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="If it's warmer than 20°C in London, set the thermostat to 20°C, otherwise set it to 18°C.",
    config=config,
)

# Print the final, user-facing response
print(response.text)

```

**Expected Output**

When you run the code, you will see the SDK orchestrating the function
calls. The model first calls `get_weather_forecast`, receives the
temperature, and then calls `set_thermostat_temperature` with the correct
value based on the logic in the prompt.

```
Tool Call: get_weather_forecast(location=London)
Tool Response: {'temperature': 25, 'unit': 'celsius'}
Tool Call: set_thermostat_temperature(temperature=20)
Tool Response: {'status': 'success'}
OK. I've set the thermostat to 20°C.

```

This example shows how to use JavaScript/TypeScript SDK to do comopositional
function calling using a manual execution loop.

```
import { GoogleGenAI, Type } from "@google/genai";

// Configure the client
const ai = new GoogleGenAI({});

// Example Functions
function get_weather_forecast({ location }) {
  console.log(`Tool Call: get_weather_forecast(location=${location})`);
  // TODO: Make API call
  console.log("Tool Response: {'temperature': 25, 'unit': 'celsius'}");
  return { temperature: 25, unit: "celsius" };
}

function set_thermostat_temperature({ temperature }) {
  console.log(
    `Tool Call: set_thermostat_temperature(temperature=${temperature})`,
  );
  // TODO: Make API call
  console.log("Tool Response: {'status': 'success'}");
  return { status: "success" };
}

const toolFunctions = {
  get_weather_forecast,
  set_thermostat_temperature,
};

const tools = [\
  {\
    functionDeclarations: [\
      {\
        name: "get_weather_forecast",\
        description:\
          "Gets the current weather temperature for a given location.",\
        parameters: {\
          type: Type.OBJECT,\
          properties: {\
            location: {\
              type: Type.STRING,\
            },\
          },\
          required: ["location"],\
        },\
      },\
      {\
        name: "set_thermostat_temperature",\
        description: "Sets the thermostat to a desired temperature.",\
        parameters: {\
          type: Type.OBJECT,\
          properties: {\
            temperature: {\
              type: Type.NUMBER,\
            },\
          },\
          required: ["temperature"],\
        },\
      },\
    ],\
  },\
];

// Prompt for the model
let contents = [\
  {\
    role: "user",\
    parts: [\
      {\
        text: "If it's warmer than 20°C in London, set the thermostat to 20°C, otherwise set it to 18°C.",\
      },\
    ],\
  },\
];

// Loop until the model has no more function calls to make
while (true) {
  const result = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents,
    config: { tools },
  });

  if (result.functionCalls && result.functionCalls.length > 0) {
    const functionCall = result.functionCalls[0];

    const { name, args } = functionCall;

    if (!toolFunctions[name]) {
      throw new Error(`Unknown function call: ${name}`);
    }

    // Call the function and get the response.
    const toolResponse = toolFunctions[name](args);

    const functionResponsePart = {
      name: functionCall.name,
      response: {
        result: toolResponse,
      },
    };

    // Send the function response back to the model.
    contents.push({
      role: "model",
      parts: [\
        {\
          functionCall: functionCall,\
        },\
      ],
    });
    contents.push({
      role: "user",
      parts: [\
        {\
          functionResponse: functionResponsePart,\
        },\
      ],
    });
  } else {
    // No more function calls, break the loop.
    console.log(result.text);
    break;
  }
}

```

**Expected Output**

When you run the code, you will see the SDK orchestrating the function
calls. The model first calls `get_weather_forecast`, receives the
temperature, and then calls `set_thermostat_temperature` with the correct
value based on the logic in the prompt.

```
Tool Call: get_weather_forecast(location=London)
Tool Response: {'temperature': 25, 'unit': 'celsius'}
Tool Call: set_thermostat_temperature(temperature=20)
Tool Response: {'status': 'success'}
OK. It's 25°C in London, so I've set the thermostat to 20°C.

```

Compositional function calling is a native [Live\\
API](https://ai.google.dev/gemini-api/docs/live) feature. This means Live API
can handle the function calling similar to the Python SDK.

[Python](https://ai.google.dev/gemini-api/docs/function-calling#python)[JavaScript](https://ai.google.dev/gemini-api/docs/function-calling#javascript)More

```
# Light control schemas
turn_on_the_lights_schema = {'name': 'turn_on_the_lights'}
turn_off_the_lights_schema = {'name': 'turn_off_the_lights'}

prompt = """
  Hey, can you write run some python code to turn on the lights, wait 10s and then turn off the lights?
  """

tools = [\
    {'code_execution': {}},\
    {'function_declarations': [turn_on_the_lights_schema, turn_off_the_lights_schema]}\
]

await run(prompt, tools=tools, modality="AUDIO")

```

```
// Light control schemas
const turnOnTheLightsSchema = { name: 'turn_on_the_lights' };
const turnOffTheLightsSchema = { name: 'turn_off_the_lights' };

const prompt = `
  Hey, can you write run some python code to turn on the lights, wait 10s and then turn off the lights?
`;

const tools = [\
  { codeExecution: {} },\
  { functionDeclarations: [turnOnTheLightsSchema, turnOffTheLightsSchema] }\
];

await run(prompt, tools=tools, modality="AUDIO")

```

## Function calling modes

The Gemini API lets you control how the model uses the provided tools
(function declarations). Specifically, you can set the mode within
the. `function_calling_config`.

- `AUTO (Default)`: The model decides whether to generate a natural language
response or suggest a function call based on the prompt and context. This is the
most flexible mode and recommended for most scenarios.
- `ANY`: The model is constrained to always predict a function call and
guarantees function schema adherence. If `allowed_function_names` is not
specified, the model can choose from any of the provided function declarations.
If `allowed_function_names` is provided as a list, the model can only choose
from the functions in that list. Use this mode when you require a function
call response to every prompt (if applicable).
- `NONE`: The model is _prohibited_ from making function calls. This is
equivalent to sending a request without any function declarations. Use this to
temporarily disable function calling without removing your tool definitions.


[Python](https://ai.google.dev/gemini-api/docs/function-calling#python)[JavaScript](https://ai.google.dev/gemini-api/docs/function-calling#javascript)More

```
from google.genai import types

# Configure function calling mode
tool_config = types.ToolConfig(
    function_calling_config=types.FunctionCallingConfig(
        mode="ANY", allowed_function_names=["get_current_temperature"]
    )
)

# Create the generation config
config = types.GenerateContentConfig(
    tools=[tools],  # not defined here.
    tool_config=tool_config,
)

```

```
import { FunctionCallingConfigMode } from '@google/genai';

// Configure function calling mode
const toolConfig = {
  functionCallingConfig: {
    mode: FunctionCallingConfigMode.ANY,
    allowedFunctionNames: ['get_current_temperature']
  }
};

// Create the generation config
const config = {
  tools: tools, // not defined here.
  toolConfig: toolConfig,
};

```

## Automatic function calling (Python only)

When using the Python SDK, you can provide Python functions directly as tools.
The SDK converts these functions into declarations, manages the function call
execution, and handles the response cycle for you. Define your function with
type hints and a docstring. For optimal results, it is recommended to use
[Google-style docstrings.](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)
The SDK will then automatically:

1. Detect function call responses from the model.
2. Call the corresponding Python function in your code.
3. Send the function's response back to the model.
4. Return the model's final text response.

The SDK currently does not parse argument descriptions into the property
description slots of the generated function declaration. Instead, it sends the
entire docstring as the top-level function description.

[Python](https://ai.google.dev/gemini-api/docs/function-calling#python)More

```
from google import genai
from google.genai import types

# Define the function with type hints and docstring
def get_current_temperature(location: str) -> dict:
    """Gets the current temperature for a given location.

    Args:
        location: The city and state, e.g. San Francisco, CA

    Returns:
        A dictionary containing the temperature and unit.
    """
    # ... (implementation) ...
    return {"temperature": 25, "unit": "Celsius"}

# Configure the client
client = genai.Client()
config = types.GenerateContentConfig(
    tools=[get_current_temperature]
)  # Pass the function itself

# Make the request
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What's the temperature in Boston?",
    config=config,
)

print(response.text)  # The SDK handles the function call and returns the final text

```

You can disable automatic function calling with:

[Python](https://ai.google.dev/gemini-api/docs/function-calling#python)More

```
config = types.GenerateContentConfig(
    tools=[get_current_temperature],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
)

```

### Automatic function schema declaration

The API is able to describe any of the following types. `Pydantic` types are
allowed, as long as the fields defined on them are also composed of allowed
types. Dict types (like `dict[str: int]`) are not well supported here, don't
use them.

[Python](https://ai.google.dev/gemini-api/docs/function-calling#python)More

```
AllowedType = (
  int | float | bool | str | list['AllowedType'] | pydantic.BaseModel)

```

To see what the inferred schema looks like, you can convert it using
[`from_callable`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionDeclaration.from_callable):

[Python](https://ai.google.dev/gemini-api/docs/function-calling#python)More

```
def multiply(a: float, b: float):
    """Returns a * b."""
    return a * b

fn_decl = types.FunctionDeclaration.from_callable(callable=multiply, client=client)

# to_json_dict() provides a clean JSON representation.
print(fn_decl.to_json_dict())

```

## Multi-tool use: Combine native tools with function calling

You can enable multiple tools combining native tools with
function calling at the same time. Here's an example that enables two tools,
[Grounding with Google Search](https://ai.google.dev/gemini-api/docs/grounding) and
[code execution](https://ai.google.dev/gemini-api/docs/code-execution), in a request using the
[Live API](https://ai.google.dev/gemini-api/docs/live).

[Python](https://ai.google.dev/gemini-api/docs/function-calling#python)[JavaScript](https://ai.google.dev/gemini-api/docs/function-calling#javascript)More

```
# Multiple tasks example - combining lights, code execution, and search
prompt = """
  Hey, I need you to do three things for me.

    1.  Turn on the lights.
    2.  Then compute the largest prime palindrome under 100000.
    3.  Then use Google Search to look up information about the largest earthquake in California the week of Dec 5 2024.

  Thanks!
  """

tools = [\
    {'google_search': {}},\
    {'code_execution': {}},\
    {'function_declarations': [turn_on_the_lights_schema, turn_off_the_lights_schema]} # not defined here.\
]

# Execute the prompt with specified tools in audio modality
await run(prompt, tools=tools, modality="AUDIO")

```

```
// Multiple tasks example - combining lights, code execution, and search
const prompt = `
  Hey, I need you to do three things for me.

    1.  Turn on the lights.
    2.  Then compute the largest prime palindrome under 100000.
    3.  Then use Google Search to look up information about the largest earthquake in California the week of Dec 5 2024.

  Thanks!
`;

const tools = [\
  { googleSearch: {} },\
  { codeExecution: {} },\
  { functionDeclarations: [turnOnTheLightsSchema, turnOffTheLightsSchema] } // not defined here.\
];

// Execute the prompt with specified tools in audio modality
await run(prompt, {tools: tools, modality: "AUDIO"});

```

Python developers can try this out in the [Live API Tool Use\\
notebook](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI_tools.ipynb).

## Model context protocol (MCP)

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) is
an open standard for connecting AI applications with external tools and data.
MCP provides a common protocol for models to access context, such as functions
(tools), data sources (resources), or predefined prompts.

The Gemini SDKs have built-in support for the MCP, reducing boilerplate code and
offering
[automatic tool calling](https://ai.google.dev/gemini-api/docs/function-calling#automatic_function_calling_python_only)
for MCP tools. When the model generates an MCP tool call, the Python and
JavaScript client SDK can automatically execute the MCP tool and send the
response back to the model in a subsequent request, continuing this loop until
no more tool calls are made by the model.

Here, you can find an example of how to use a local MCP server with Gemini and
`mcp` SDK.

[Python](https://ai.google.dev/gemini-api/docs/function-calling#python)[JavaScript](https://ai.google.dev/gemini-api/docs/function-calling#javascript)More

Make sure the latest version of the
[`mcp` SDK](https://modelcontextprotocol.io/introduction) is installed on
your platform of choice.

```
pip install mcp

```

```
import os
import asyncio
from datetime import datetime
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai

client = genai.Client()

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="npx",  # Executable
    args=["-y", "@philschmid/weather-mcp"],  # MCP Server
    env=None,  # Optional environment variables
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Prompt to get the weather for the current day in London.
            prompt = f"What is the weather in London in {datetime.now().strftime('%Y-%m-%d')}?"

            # Initialize the connection between client and server
            await session.initialize()

            # Send request to the model with MCP function declarations
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0,
                    tools=[session],  # uses the session, will automatically call the tool
                    # Uncomment if you **don't** want the SDK to automatically call the tool
                    # automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(
                    #     disable=True
                    # ),
                ),
            )
            print(response.text)

# Start the asyncio event loop and run the main function
asyncio.run(run())

```

Make sure the latest version of the `mcp` SDK is installed on your platform
of choice.

```
npm install @modelcontextprotocol/sdk

```

```
import { GoogleGenAI, FunctionCallingConfigMode , mcpToTool} from '@google/genai';
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

// Create server parameters for stdio connection
const serverParams = new StdioClientTransport({
  command: "npx", // Executable
  args: ["-y", "@philschmid/weather-mcp"] // MCP Server
});

const client = new Client(
  {
    name: "example-client",
    version: "1.0.0"
  }
);

// Configure the client
const ai = new GoogleGenAI({});

// Initialize the connection between client and server
await client.connect(serverParams);

// Send request to the model with MCP tools
const response = await ai.models.generateContent({
  model: "gemini-2.5-flash",
  contents: `What is the weather in London in ${new Date().toLocaleDateString()}?`,
  config: {
    tools: [mcpToTool(client)],  // uses the session, will automatically call the tool
    // Uncomment if you **don't** want the sdk to automatically call the tool
    // automaticFunctionCalling: {
    //   disable: true,
    // },
  },
});
console.log(response.text)

// Close the connection
await client.close();

```

### Limitations with built-in MCP support

Built-in MCP support is a [experimental](https://ai.google.dev/gemini-api/docs/models#preview)
feature in our SDKs and has the following limitations:

- Only tools are supported, not resources nor prompts
- It is available for the Python and JavaScript/TypeScript SDK.
- Breaking changes might occur in future releases.

Manual integration of MCP servers is always an option if these limit what you're
building.

## Supported models

This section lists models and their function calling capabilities. Experimental
models are not included. You can find a comprehensive capabilities overview on
the [model overview](https://ai.google.dev/gemini-api/docs/models) page.

| Model | Function Calling | Parallel Function Calling | Compositional Function Calling |
| --- | --- | --- | --- |
| Gemini 2.5 Pro | ✔️ | ✔️ | ✔️ |
| Gemini 2.5 Flash | ✔️ | ✔️ | ✔️ |
| Gemini 2.5 Flash-Lite | ✔️ | ✔️ | ✔️ |
| Gemini 2.0 Flash | ✔️ | ✔️ | ✔️ |
| Gemini 2.0 Flash-Lite | X | X | X |

## Best practices

- **Function and Parameter Descriptions:** Be extremely clear and specific in
your descriptions. The model relies on these to choose the correct function
and provide appropriate arguments.
- **Naming:** Use descriptive function names (without spaces, periods, or
dashes).
- **Strong Typing:** Use specific types (integer, string, enum) for parameters
to reduce errors. If a parameter has a limited set of valid values, use an
enum.
- **Tool Selection:** While the model can use an arbitrary number of tools,
providing too many can increase the risk of selecting an incorrect or
suboptimal tool. For best results, aim to provide only the relevant tools
for the context or task, ideally keeping the active set to a maximum of
10-20. Consider dynamic tool selection based on conversation context if you
have a large total number of tools.
- **Prompt Engineering:**
  - Provide context: Tell the model its role (e.g., "You are a helpful
    weather assistant.").
  - Give instructions: Specify how and when to use functions (e.g., "Don't
    guess dates; always use a future date for forecasts.").
  - Encourage clarification: Instruct the model to ask clarifying questions
    if needed.
- **Temperature:** Use a low temperature (e.g., 0) for more deterministic and
reliable function calls.
- **Validation:** If a function call has significant consequences (e.g.,
placing an order), validate the call with the user before executing it.
- **Error Handling**: Implement robust error handling in your functions to
gracefully handle unexpected inputs or API failures. Return informative
error messages that the model can use to generate helpful responses to the
user.
- **Security:** Be mindful of security when calling external APIs. Use
appropriate authentication and authorization mechanisms. Avoid exposing
sensitive data in function calls.
- **Token Limits:** Function descriptions and parameters count towards your
input token limit. If you're hitting token limits, consider limiting the
number of functions or the length of the descriptions, break down complex
tasks into smaller, more focused function sets.

## Notes and limitations

- Only a [subset of the OpenAPI\\
schema](https://ai.google.dev/api/caching#FunctionDeclaration) is supported.
- Supported parameter types in Python are limited.
- Automatic function calling is a Python SDK feature only.

</details>

<details>
<summary>react-agent-from-scratch-with-gemini-2-5-and-langgraph-gemin</summary>

# ReAct agent from scratch with Gemini 2.5 and LangGraph

LangGraph is a framework for building stateful LLM applications, making it a good choice for constructing ReAct (Reasoning and Acting) Agents.

ReAct agents combine LLM reasoning with action execution. They iteratively think, use tools, and act on observations to achieve user goals, dynamically adapting their approach. Introduced in ["ReAct: Synergizing Reasoning and Acting in Language Models"](https://arxiv.org/abs/2210.03629) (2023), this pattern tries to mirror human-like, flexible problem-solving over rigid workflows.

While LangGraph offers a prebuilt ReAct agent ( [`create_react_agent`](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent)), it shines when you need more control and customization for your ReAct implementations.

LangGraph models agents as graphs using three key components:

- `State`: Shared data structure (typically `TypedDict` or `Pydantic BaseModel`) representing the application's current snapshot.
- `Nodes`: Encodes logic of your agents. They receive the current State as input, perform some computation or side-effect, and return an updated State, such as LLM calls or tool calls.
- `Edges`: Define the next `Node` to execute based on the current `State`, allowing for conditional logic and fixed transitions.

If you don't have an API Key yet, you can get one for free at the [Google AI Studio](https://aistudio.google.com/app/apikey).

```
pip install langgraph langchain-google-genai geopy requests

```

Set your API key in the environment variable `GEMINI_API_KEY`.

```
import os

# Read your API key from the environment variable or set it manually
api_key = os.getenv("GEMINI_API_KEY")

```

To better understand how to implement a ReAct agent using LangGraph, let's walk through a practical example. You will create a simple agent whose goal is to use a tool to find the current weather for a specified location.

For this weather agent, its `State` will need to maintain the ongoing conversation history (as a list of messages) and a counter for the number of steps taken to further illustrate state management.

LangGraph provides a convenient helper, `add_messages`, for updating message lists in the state. It functions as a [reducer](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers), meaning it takes the current list and new messages, then returns a combined list. It smartly handles updates by message ID and defaults to an "append-only" behavior for new, unique messages.

```
from typing import Annotated,Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages # helper function to add messages to the state

class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    number_of_steps: int

```

Next, you define your weather tool.

```
from langchain_core.tools import tool
from geopy.geocoders import Nominatim
from pydantic import BaseModel, Field
import requests

geolocator = Nominatim(user_agent="weather-app")

class SearchInput(BaseModel):
    location:str = Field(description="The city and state, e.g., San Francisco")
    date:str = Field(description="the forecasting date for when to get the weather format (yyyy-mm-dd)")

@tool("get_weather_forecast", args_schema=SearchInput, return_direct=True)
def get_weather_forecast(location: str, date: str):
    """Retrieves the weather using Open-Meteo API for a given location (city) and a date (yyyy-mm-dd). Returns a list dictionary with the time and temperature for each hour."""
    location = geolocator.geocode(location)
    if location:
        try:
            response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={location.latitude}&longitude={location.longitude}&hourly=temperature_2m&start_date={date}&end_date={date}")
            data = response.json()
            return {time: temp for time, temp in zip(data["hourly"]["time"], data["hourly"]["temperature_2m"])}
        except Exception as e:
            return {"error": str(e)}
    else:
        return {"error": "Location not found"}

tools = [get_weather_forecast]

```

Next, you initialize your model and bind the tools to the model.

```
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI

# Create LLM class
llm = ChatGoogleGenerativeAI(
    model= "gemini-2.5-pro",
    temperature=1.0,
    max_retries=2,
    google_api_key=api_key,
)

# Bind tools to the model
model = llm.bind_tools([get_weather_forecast])

# Test the model with tools
res=model.invoke(f"What is the weather in Berlin on {datetime.today()}?")

print(res)

```

The last step before you can run your agent is to define your nodes and edges. In this example, you have two nodes and one edge.
\- `call_tool` node that executes your tool method. LangGraph has a prebuilt node for this called [ToolNode](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/).
\- `call_model` node that uses the `model_with_tools` to call the model.
\- `should_continue` edge that decides whether to call the tool or the model.

The number of nodes and edges is not fixed. You can add as many nodes and edges as you want to your graph. For example, you could add a node for adding structured output or a self-verification/reflection node to check the model output before calling the tool or the model.

```
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

tools_by_name = {tool.name: tool for tool in tools}

# Define our tool node
def call_tool(state: AgentState):
    outputs = []
    # Iterate over the tool calls in the last message
    for tool_call in state["messages"][-1].tool_calls:
        # Get the tool by name
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=tool_result,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}

def call_model(
    state: AgentState,
    config: RunnableConfig,
):
    # Invoke the model with the system prompt and the messages
    response = model.invoke(state["messages"], config)
    # We return a list, because this will get added to the existing messages state using the add_messages reducer
    return {"messages": [response]}

# Define the conditional edge that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]
    # If the last message is not a tool call, then we finish
    if not messages[-1].tool_calls:
        return "end"
    # default to continue
    return "continue"

```

Now you have all the components to build your agent. Let's put them together.

```
from langgraph.graph import StateGraph, END

# Define a new graph with our state
workflow = StateGraph(AgentState)

# 1. Add our nodes
workflow.add_node("llm", call_model)
workflow.add_node("tools",  call_tool)
# 2. Set the entrypoint as `agent`, this is the first node called
workflow.set_entry_point("llm")
# 3. Add a conditional edge after the `llm` node is called.
workflow.add_conditional_edges(
    # Edge is used after the `llm` node is called.
    "llm",
    # The function that will determine which node is called next.
    should_continue,
    # Mapping for where to go next, keys are strings from the function return, and the values are other nodes.
    # END is a special node marking that the graph is finish.
    {
        # If `tools`, then we call the tool node.
        "continue": "tools",
        # Otherwise we finish.
        "end": END,
    },
)
# 4. Add a normal edge after `tools` is called, `llm` node is called next.
workflow.add_edge("tools", "llm")

# Now we can compile and visualize our graph
graph = workflow.compile()

```

You can visualize your graph using the `draw_mermaid_png` method.

```
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

```https://ai.google.dev/static/gemini-api/docs/images/langgraph-react-agent_16_0.png

Now let's run the agent.

```
from datetime import datetime
# Create our initial message dictionary
inputs = {"messages": [("user", f"What is the weather in Berlin on {datetime.today()}?")]}

# call our graph with streaming to see the steps
for state in graph.stream(inputs, stream_mode="values"):
    last_message = state["messages"][-1]
    last_message.pretty_print()

```

You can now continue with your conversation and for example ask for the weather in another city or let it compare it.

```
state["messages"].append(("user", "Would it be in Munich warmer?"))

for state in graph.stream(state, stream_mode="values"):
    last_message = state["messages"][-1]
    last_message.pretty_print()

```

</details>

<details>
<summary>scraping-failed</summary>

⚠️ Error scraping https://arxiv.org/pdf/2504.19678 after 3 attempts: Internal Server Error: Failed to make POST request. (Internal server error) - Scrape resulted in unsupported file: File size exceeds 10MB - No additional error details provided.

</details>

<details>
<summary>what-is-a-react-agent-ibm</summary>

A ReAct agent is an [AI agent](https://www.ibm.com/think/topics/ai-agents) that uses the “reasoning and acting” (ReAct) framework to combine [chain of thought (CoT)](https://www.ibm.com/think/topics/chain-of-thoughts) reasoning with external tool use. The ReAct framework enhances the ability of a [large language model (LLM)](https://www.ibm.com/think/topics/large-language-models) to handle complex tasks and decision-making in [agentic workflows](https://www.ibm.com/think/topics/agentic-workflows).

First introduced by Yao and others in the 2023 paper, “ReACT: Synergizing Reasoning and Acting in Language Models,” ReAct can be understood most generally as a [machine learning](https://www.ibm.com/think/topics/machine-learning) (ML) paradigm to integrate the reasoning and action-taking capabilities of LLMs.

More specifically, ReAct is a conceptual framework for building AI agents that can interact with their environment in a structured but adaptable way, by using an LLM as the agent’s “brain” to coordinate anything from simple [retrieval augmented generation (RAG)](https://research.ibm.com/blog/retrieval-augmented-generation-RAG) to intricate [multiagent](https://www.ibm.com/think/topics/multiagent-system) workflows.

Unlike traditional artificial intelligence (AI) systems, ReAct agents don’t separate decision-making from task execution. Therefore, the development of the ReAct paradigm was an important step in the evolution of [generative AI (gen AI)](https://www.ibm.com/think/topics/generative-ai) beyond mere conversational [chatbots](https://www.ibm.com/think/topics/chatbots) and toward complex problem-solving.

ReAct agents and derivative approaches continue to power AI applications that can autonomously plan, execute and adapt to unforeseen circumstances.

## How do ReAct agents work?

The ReAct framework is inspired by the way humans can intuitively use natural language—often through our own inner monologue—in the step-by-step planning and execution of complex tasks.

Rather than implementing rule-based or otherwise predefined workflows, ReAct agents rely on their LLM’s reasoning capabilities to dynamically adjust their approach based on new information or the results of previous steps.

Imagine packing for a brief trip. You might start by identifying key considerations (“ _What will the weather be like while I’m there?_”), then actively consult external sources (“ _I’ll check the local weather forecast_”).

By using that new information (“ _It’s going to be cold_”), you determine your next consideration (“ _What warm clothes do I have?_”) and action (“ _I’ll check my closet_”). Upon taking that action, you might encounter an unexpected obstacle (“ _All of my warm clothes are in storage_”) and adjust your next step accordingly (“ _What clothes can I layer together?_”).

In a similar fashion, the ReAct framework uses [prompt engineering](https://www.ibm.com/think/topics/prompt-engineering) to structure an AI agent’s activity in a formal pattern of alternating thoughts, actions and observations:

- The verbalized CoT reasoning steps ( _thoughts_) help the model decompose the larger task into more manageable subtasks.


- Predefined _actions_ enable the model to use tools, make [application programming interface (API)](https://www.ibm.com/think/topics/api) calls and gather more information from external sources (such as search engines) or knowledge bases (such as an internal docstore).


- After taking an action, the model then reevaluates its progress and uses that _observation_ to either deliver a final answer or inform the next _thought_. The observation might ideally also consider prior information, whether from earlier in the model’s standard context window or from an external memory component.


Because the performance of a ReAct agent depends heavily on the ability of its central LLM to “verbally” think its way through complex tasks, ReAct agents benefit greatly from highly capable models with advanced reasoning and instruction-following ability.

To minimize cost and [latency](https://www.ibm.com/think/topics/latency), a multiagent ReAct framework might rely primarily on a larger, more performant model to serve as the central agent whose reasoning process or actions might involve delegating subtasks to more agents built using smaller, more efficient models.

### ReAct agent loops

This framework inherently creates a feedback loop in which the model problem-solves by iteratively repeating this interleaved _thought-action-observation_ process.

Each time this loop is completed—that is, each time the agent has taken an action and made an observation based on the results of that action—the agent must then decide whether to repeat or end the loop.

When and how to end the reasoning loop is an important consideration in the design of a ReAct agent. Establishing a maximum number of loop iterations is a simple way to limit latency, costs and token usage, and avoid the possibility of an endless loop.

Conversely, the loop can be set to end when some specific condition is met, such as when the model has identified a potential final answer that exceeds a certain confidence threshold.

To implement this kind of reasoning and acting loop, ReAct agents typically use some variant of _ReAct prompting_, whether in the system prompt provided to the LLM or in the context of the user query itself.

## ReAct prompting

ReAct prompting is a specific prompting technique designed to guide an LLM to follow the ReAct paradigm of _thought_, _action_ and _observation_ loops. While the explicit use of conventional ReAct prompting methods is not strictly necessary to build a ReAct agent, most ReAct-based agents implement or at least take direct inspiration from it.

First outlined in the original ReAct paper, ReAct prompting’s primary function is to instruct an LLM to follow the ReAct loop and establish which tools can be used—that is, which actions can be taken—when handling user queries.

Whether through explicit instructions or the inclusion of [few-shot](https://www.ibm.com/think/topics/few-shot-learning) examples, ReAct prompting should:

- **Guide the model to use chain of thought reasoning:** Prompt the model to reason its way through tasks by thinking step by step, interleaving thoughts with actions.


- **Define actions:** Establish the specific actions available to the model. An action might entail the generation of a specific type of next thought or subprompt but usually involves [using external tools](https://www.ibm.com/think/topics/tool-calling) or making APIs.


- **Instruct the model to make observations:** Prompt the model to reassess its context after each action step and use that updated context to inform the next reasoning step.


- **Loop:** Instruct the model to repeat the previous steps if necessary. You could provide specific conditions for ending that loop, such as a maximum number of loops, or instruct the agent to end its reasoning process whenever it feels it has arrived at the correct final output.


- **Output final answer:** Whenever those end conditions have been met, provide the user with the final output in response to their initial query. As with many uses of LLMs, as reasoning models employing chain of thought reasoning before determining a final output, ReAct agents are often prompted to conduct their reasoning process within a [“scratchpad.”](https://arxiv.org/abs/2112.00114)


A classic demonstration of ReAct prompting is the system prompt for the prebuiltZERO\_SHOT\_REACT-DESCRIPTION
ReAct agent module in [Langchain](https://www.ibm.com/think/topics/langchain)’s LangGraph. It’s called “ [zero-shot](https://www.ibm.com/think/topics/zero-shot-learning)” because, with this predefined system prompt, the LLM being used with the module does not need any further examples to behave as a ReAct agent.

```
Answer the following questions as best you can. You have access to the following tools:

Wikipedia: A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query.
duckduckgo_search: A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events. Input should be a search query.
Calculator: Useful for when you need to answer questions about math.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Wikipedia, duckduckgo_search, Calculator]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}

```

## Benefits of ReAct agents

The introduction of the ReAct framework was an important step in the advancement of LLM-driven [agentic workflows](https://www.ibm.com/think/topics/agentic-workflows). From grounding LLMs in real time, real-world external information through (RAG) to contributing to subsequent breakthroughs—such as [Reflexion](https://arxiv.org/abs/2303.11366), which led to modern reasoning models—ReAct has helped catalyze the use of LLMs for tasks well beyond text generation.

The utility of ReAct agents is drawn largely from some of the inherent qualities of the ReAct framework:

- **Versatility:** ReAct agents can be configured to work with a wide variety of external tools and APIs. Though [fine-tuning](https://www.ibm.com/think/topics/fine-tuning) relevant ReAct prompts (using relevant tools) can improve performance, no prior configuration of the model is required to execute [tool calls](https://www.ibm.com/think/topics/tool-calling).


- **Adaptability:** This versatility, along with the dynamic and situational nature of how they determine the appropriate tool or API to call, means that ReAct agents can use their reasoning process to adapt to new challenges. Especially when operating within a lengthy context window or augmented with external memory, they can learn from past mistakes and successes to tackle unforeseen obstacles and situations. This makes ReAct agents flexible and resilient.


- **Explainability:** The verbalized reasoning process of a ReAct agent is simple to follow, which facilitates debugging and helps make them relatively user-friendly to build and optimize.


- **Accuracy:** As the original ReAct paper asserts, chain of thought (CoT) reasoning alone has many benefits for LLMs, but also runs an increased risk of hallucination. ReAct’s combination of CoT with a connection external to information sources significantly reduces [hallucinations](https://www.ibm.com/think/topics/ai-hallucinations), making ReAct agents more accurate and trustworthy.


## ReAct agents vs. function calling

Another prominent paradigm for agentic AI is function calling, originally [introduced by OpenAI in June 2023](https://openai.com/index/function-calling-and-other-api-updates/) to supplement the agentic abilities of its [GPT models](https://www.ibm.com/think/topics/gpt).

The function calling paradigm entails [fine-tuning](https://www.ibm.com/think/topics/fine-tuning) models to recognize when a particular situation should result in a tool call and output a structured [JSON](https://www.ibm.com/docs/en/baw/24.x?topic=formats-javascript-object-notation-json-format) object containing the arguments necessary to call those functions.

Many proprietary and open source LLM families, [including IBM® Granite®](https://www.ibm.com/granite/docs/models/granite/#function-calling), Meta’s [Llama](https://www.ibm.com/think/news/meta-llama-3-2-models) series, Anthropic’s [Claude](https://www.ibm.com/think/topics/claude-ai) and [Google Gemini](https://www.ibm.com/think/topics/google-gemini), now support function calling.

Whether ReAct or function calling is “better” will generally depend on the nature of your specific use case. In scenarios involving relatively straightforward (or at least predictable) tasks, function calling can execute faster, save tokens, and be simpler to implement than a ReAct agent.

In such circumstances, the number of tokens that would be spent on a ReAct agent’s iterative loop of CoT reasoning might be seen as inefficient.

The inherent tradeoff is a relative lack of ability to customize how and when the model chooses which tool to use. Likewise, when an agent handles tasks that call for complex reasoning, or scenarios that are dynamic or unpredictable, the rigidity of function calling might limit the agent’s adaptability. In such situations, it’s often beneficial to be able to view the step-by-step reasoning that led to a specific tool call.

## Getting started with ReAct agents

ReAct agents can be designed and implemented in multiple ways, whether coded from scratch in Python or developed with the help of open source frameworks such as [BeeAI](https://research.ibm.com/blog/bee-ai-app). The popularity and staying power of the ReAct paradigm have yielded extensive literature and tutorials for ReAct agents on GitHub and other developer communities.

As an alternative to developing custom ReAct agents, many agentic AI frameworks, including BeeAI, [LlamaIndex](https://www.ibm.com/think/topics/llamaindex) and LangChain’s LangGraph, offer [preconfigured ReAct agent modules](https://github.com/i-am-bee/beeai-framework/blob/main/python/examples/agents/react.py) for [specific use cases](https://beeai.dev/agents).

</details>

<details>
<summary>what-is-ai-agent-orchestration-ibm</summary>

[Artificial intelligence (AI)](https://www.ibm.com/think/artificial-intelligence) agent orchestration is the process of coordinating multiple specialized [AI agents](https://www.ibm.com/think/topics/ai-agents) within a unified system to efficiently achieve shared objectives.

Rather than relying on a single, general-purpose AI solution, AI agent orchestration employs a network of AI agents, each designed for specific tasks, working together to automate complex workflows and processes.

To fully understand AI agent orchestration, it's essential to first understand AI agents themselves. This involves [understanding the differences](https://www.ibm.com/think/topics/agentic-ai-vs-generative-ai) between two key types of AI: [generative AI](https://www.ibm.com/think/topics/generative-ai), which creates original content based on a user’s prompt, and [agentic AI](https://www.ibm.com/think/insights/agentic-ai), which autonomously makes decisions and acts to pursue complex goals with minimal supervision.

AI assistants exist on a continuum, starting with rule-based chatbots, progressing to more advanced virtual assistants and evolving into generative AI and [large language model (LLM)](https://www.ibm.com/think/topics/large-language-models)-powered assistants capable of handling single-step tasks. At the top of this progression are AI agents, which operate autonomously. These agents make decisions, design workflows and use function calling to connect with external tools—such as [application programming interfaces (APIs)](https://www.ibm.com/think/topics/api), data sources, web searches and even other AI agents—to fill gaps in their knowledge. This is agentic AI.

AI agents are specialized, meaning each one is optimized for a particular function. Some agents focus on business and customer-facing tasks like billing, troubleshooting, scheduling and decision-making, while others handle more technical functions like [natural language processing (NLP),](https://www.ibm.com/think/topics/natural-language-processing) data retrieval and process automation. Advanced LLMs such as OpenAI's ChatGPT-4o or Google's Gemini often power these agents, with generative AI capabilities enabling them to create human-like responses and handle complex tasks autonomously.

Multi-agent systems (MAS) emerge when multiple AI agents collaborate, either in a structured or decentralized manner, to solve complex tasks more efficiently than a single agent might.

In practice, AI agent orchestration functions like a digital symphony. Each agent has a unique role and the system is guided by an orchestrator—either a central AI agent or framework —that manages and coordinates their interactions. The orchestrator helps synchronize these specialized agents, ensuring that the right agent is activated at the right time for each task. This coordination is crucial for handling multifaceted workflows that involve various tasks, helping ensure that processes are run seamlessly and efficiently.

For example, as part of [customer service automation](https://www.ibm.com/think/topics/customer-service-automation), the orchestrator agent (the system responsible for managing AI agents) might determine whether to engage a billing agent versus a technical support agent, helping ensure that customers receive seamless and relevant assistance. In MAS, agents might coordinate without a single orchestrator, dynamically communicating to collaboratively solve problems (see “Types of AI orchestration” below)

The benefits of AI agent orchestration are significant in industries with complex, dynamic needs such as telecommunications, banking and healthcare. By deploying specialized agents that are trained on targeted datasets and workflows, businesses can enhance [operational efficiency](https://www.ibm.com/think/topics/operational-efficiency), improve decision-making and deliver more accurate, efficient and context-aware results for both employees and customers.

## Why AI agent orchestration is important

As AI systems grow more advanced, a single AI model or agent is often insufficient for handling complex tasks. Autonomous systems frequently struggle to collaborate because they are built across multiple clouds and applications, leading to siloed operations and inefficiencies. AI agent orchestration bridges these gaps, enabling multiple AI agents to work together efficiently and ensuring that sophisticated tasks are run seamlessly.

In large-scale applications such as healthcare, finance and customer service, multiple agents often need to work together to handle different aspects of a task. For example, in healthcare, AI agents can coordinate between diagnostic tools, patient management systems and administrative workflows to streamline operations and enhance treatment accuracy. Without orchestration, these agents might work in isolation, leading to inefficiencies, redundancies or gaps in execution.

By managing interactions between multi-agent systems, orchestration helps ensure that each agent contributes effectively toward a shared goal. It optimizes workflows, minimizes errors and enhances interoperability, allowing AI systems to dynamically allocate resources, prioritize tasks and respond to changing conditions in real time. This capability is valuable in fields requiring continuous optimization such as supply chain management and personalized digital assistants.

As AI systems continue to evolve, AI agent orchestration becomes increasingly essential for unlocking their full potential.

## Types of AI agent orchestration

There are several types of AI agent orchestration. Real-world systems often combine multiple orchestration styles for more effective results.

**Centralized orchestration**: A single AI orchestrator agent acts as the "brain" of the system, directing all the other agents, assigning tasks and making final decisions. This structured approach helps ensure consistency, control and predictable workflows.

**Decentralized orchestration**: This model shifts away from a single, controlling entity, allowing MAS to function through direct communication and collaboration. Agents make independent decisions or reach a consensus as a group. This makes the system more scalable and resilient since no single failure can bring it down.

**Hierarchical orchestration**: Here, AI agents are arranged in layers, resembling a tiered command structure. Higher-level orchestrator agents oversee and manage lower-level agents, striking a balance between strategic control and task-specific execution. This allows for more organized workflows while still enabling specialized agents to operate with some autonomy. If the hierarchy becomes too rigid, adaptability can suffer.

**Federated orchestration**: This approach focuses on collaboration between independent AI agents or separate organizations, allowing them to work together without fully sharing data or relinquishing control over their individual systems. This is especially useful in situations where privacy, security or regulatory constraints prevent unrestricted data sharing, such as in healthcare, banking or cross-company collaborations.

## Comparing AI agent orchestration with related practices

**AI orchestration** manages and automates various AI components—like machine learning models, data pipelines and APIs—to help ensure that they work together efficiently within a system. It focuses on optimizing performance, automating repetitive tasks, supporting scalability and system-wide performance.

**AI agent orchestration** is a subset of AI orchestration that focuses specifically on coordinating autonomous AI agents—software entities that can make independent decisions and take actions. It helps ensure that agents collaborate effectively, assigning tasks and structuring workflows.

**Multi-agent orchestration** goes a step further, managing multiple AI agents working together on complex problems. It deals with communication, role allocation and conflict resolution to help ensure seamless collaboration between agents.

## AI agent orchestration steps

AI agent orchestration is a structured process to help ensure seamless collaboration between AI agents. The goal is to manage specialized agents effectively so they can autonomously complete tasks, share data flow and optimize workflows.

Initial steps involving design, configuration and implementation are performed by humans, including as AI engineers, developers and business strategists. Once the orchestrator agent is set up, it autonomously manages AI applications, assigning tasks, coordinating workflows and facilitating real-time collaboration.

The process generally follows these key steps:

- Assessment and planning
- Selection of specialized AI agents
- Orchestration framework implementation
- Agent selection and assignment
- Workflow coordination and execution
- Data sharing and context management
- Continuous optimization and learning

### Assessment and planning (human-driven)

Before orchestration begins, organizations assess their existing AI ecosystem and identify processes that might benefit from multi-agent orchestration. The orchestration team defines clear objectives, determines the scope of integration and selects the appropriate AI technologies.

### Selection of specialized AI agents (human-driven)

AI engineers and developers choose task-specific AI agents, such as those specializing in data analysis, automation or decision-making. These agents use gen AI and machine learning models to enhance their functions.

### Orchestration framework implementation (human-driven)

System architects integrate selected AI agents into a unified orchestration framework, establishing workflows that facilitate smooth agent-to-agent communication. This involves:

- Defining task execution sequences

- Setting up API integrations for data access

- Implementing open source orchestration tools such as IBM watsonx Orchestrate, Microsoft Power Automate and LangChain

Once this is complete, the orchestrator agent takes over real-time execution.

### Agent selection and assignment (orchestrator-driven)

The orchestrator dynamically identifies the best-suited AI agents for each task based on real-time data, workload balancing and predefined rules.

### Workflow coordination and execution (orchestrator-driven)

The orchestrator platform manages task sequencing and execution, helping to ensure smooth collaboration between agents. This includes:

- Breaking down tasks into subtasks

- Assigning the right AI agents to handle each step

- Managing inter-agent dependencies

- Integrating with external systems through API calls to access necessary data and services

### Data sharing and context management (orchestrator-driven)

To help ensure accuracy and prevent redundant work, AI agents continuously exchange information, maintaining a shared knowledge base. The orchestrator updates agents with real-time context.

### Continuous optimization and learning (orchestrator + human input)

The orchestrator monitors agent performance, detects inefficiencies and can autonomously adjust workflows. Human oversight is often required for refining orchestration strategies, retraining AI models or modifying orchestration rules for long-term improvements.

## AI agent orchestration benefits

AI agent orchestration offers several key benefits across various industries, making it a valuable approach for businesses aiming to enhance their operations and customer interactions.

**Enhanced efficiency**: Coordinating multiple specialized agents helps businesses streamline workflows, reduce redundancies and improve overall operational performance.

**Agility and flexibility**: AI agent orchestration allows organizations to adapt their operations rapidly as market conditions change.

**Improved experiences**: Orchestrated AI agents can enhance operational efficiency and provide more accurate and personalized support, resulting in more satisfying experiences for customers and employees.

**Increased reliability and fault tolerance**: The failure of one agent can be mitigated by others, which enhances system reliability and helps ensure continuous service delivery.

**Self-improving workflows**: Unlike traditional integration patterns, agent orchestration enables the creation of workflows that can autonomously adapt to new data and evolving requirements, improving over time.

**Scalability**: AI agent orchestration allows organizations to handle increased demand without compromising performance or accuracy.

## AI agent orchestration challenges

AI agent orchestration comes with several challenges, but each has potential solutions. By addressing these challenges, AI agent orchestration can be more efficient, scalable and resilient.

**Multi-agent dependencies**: When deploying multi-agent frameworks, there is a risk of malfunction. Systems built on the same foundation models may be susceptible to shared vulnerabilities, which might lead to a widespread failure of all agents that are involved or make them more prone to external attacks. This underscores the importance of data governance in building foundation models and thorough training and testing processes.

**Coordination and communication**: If agents don’t interact properly, they can end up working against each other or duplicating efforts. To prevent this, it’s important to establish clear protocols, standardized APIs and reliable message-passing systems to keep everything running smoothly.

**Scalability**: As the number of AI agents increases, maintaining system performance and manageability becomes more complex. A poorly designed orchestration system may struggle with increased workloads, leading to delays or system failures. This can be avoided by using decentralized or hierarchical orchestration models that distribute decision-making, preventing a single point of failure or congestion.

**Decision-making complexity**: In multi-agent environments, determining how tasks should be allocated and executed can become highly complex. Without a clear structure, agents may struggle to make decisions, particularly in dynamic environments where conditions frequently change. Reinforcement learning, prioritization algorithms and predefined roles can help ensure that agents can autonomously determine their tasks while maintaining efficiency.

**Fault tolerance**: What happens if an agent or the orchestrator itself fails? Fault tolerance is crucial and needs to be reinforced by designing failover mechanisms, redundancy strategies and self-healing architectures that allow the system to recover automatically without human intervention.

**Data privacy and security**: AI agents frequently process and share sensitive information, raising concerns about data security and privacy. To mitigate these risks, organizations should implement strong encryption protocols, enforce strict access controls and use federated learning techniques that allow AI models to improve collaboratively without exposing raw data.

**Adaptability and learning**: AI agents must continuously adapt to new tasks and challenges. Systems that require constant manual updates can become inefficient and costly to maintain. To enhance adaptability, machine learning techniques, continuous monitoring and feedback loops can be integrated into the orchestration process. These methods enable AI agents to refine their behavior over time, improving individual and system-wide performance without requiring frequent human intervention.

</details>

<details>
<summary>what-is-ai-agent-planning-ibm</summary>

AI agent planning refers to the process by which an artificial intelligence (AI) agent determines a sequence of actions to achieve a specific goal. It involves decision-making, goal prioritization and action sequencing, often using various planning algorithms and frameworks.

[AI agent](https://www.ibm.com/think/topics/ai-agents) planning is a [module common to many types of agents](https://www.ibm.com/think/topics/components-of-ai-agents) that exists alongside other modules such as perception, reasoning, decision-making, action, memory, communication and learning. Planning works in conjunction with these other modules to help ensure that agents achieve outcomes desired by their designers.

Not all agents can plan. Unlike simple reactive agents that respond immediately to inputs, planning agents anticipate future states and generate a structured action plan before execution. This makes AI planning essential for [automation](https://www.ibm.com/think/topics/enterprise-automation) tasks that require multistep decision-making, optimization and adaptability.

## How AI agent planning works

Advances in [large language models](https://www.ibm.com/think/topics/large-language-models) (LLMs) such as OpenAI’s [GPT](https://www.ibm.com/think/topics/gpt) and related techniques involving [machine learning algorithms](https://www.ibm.com/think/topics/machine-learning-algorithms) resulted in the [generative AI](https://www.ibm.com/think/topics/generative-ai) (gen AI) boom of recent years, and further advancements have led to the emerging field of autonomous agents.

By integrating tools, APIs, hardware interfaces and other external resources, agentic AI systems are increasingly autonomous, capable of real-time decision-making and adept at problem-solving across various use cases.

Complex agents can’t act without making a decision, and they can’t make good decisions without first making a plan. Agentic planning consists of several key components that work together to encourage optimal decision-making.

### Goal definition

The first and most critical step in AI planning is defining a clear objective. The goal serves as the guiding principle for the agent’s decision-making process, determining the end state it seeks to achieve. Goals can either be static, remaining unchanged throughout the planning process, or dynamic, adjusting based on environmental conditions or user interactions.

For instance, a self-driving car might have a goal of reaching a specific destination efficiently while adhering to safety regulations. Without a well-defined goal, an agent would lack direction, leading to erratic or inefficient behavior.

If the goal is complex, agentic [AI models](https://www.ibm.com/think/topics/ai-model) will break it down into smaller, more manageable sub-goals in a process called task decomposition. This allows the system to focus on complex tasks in a hierarchical manner.

LLMs play a vital role in task decomposition, breaking down a high-level goal into smaller subtasks and then executing those subtasks through various steps. For instance, a user might ask a [chatbot](https://www.ibm.com/think/topics/chatbots) with a natural language prompt to plan a trip.

The agent would first decompose the task into components such as booking flights, finding hotels and planning an itinerary. Once decomposed, the agent can use [application programming interfaces (APIs)](https://www.ibm.com/think/topics/api) to fetch real-time data, check pricing and even suggest destinations.

### State representation

To plan effectively, an agent must have a structured understanding of its environment. This understanding is achieved through state representation, which models the current conditions, constraints and contextual factors that influence decision-making.

Agents have some built-in knowledge from their training data or [datasets](https://www.ibm.com/think/topics/dataset) representing previous interactions, but perception is required for agents to have a real-time understanding of their environment. Agents collect data through sensory input, allowing it to model its environment, along with user input and data describing its own internal state.

The complexity of state representation varies depending on the task. For example, in a chess game, the state includes the position of all pieces on the board, while in a robotic navigation system, the state might involve spatial coordinates, obstacles and terrain conditions.

The accuracy of state representation directly impacts an agent’s ability to make informed decisions, as it determines how well the agent can predict the outcomes of its actions.

### Action sequencing

Once the agent has established its goal and assessed its environment, it must determine a sequence of actions that will transition it from its current state to the desired goal state. This process, known as action sequencing, involves structuring a logical and efficient set of steps that the agent must follow.

The agent needs to identify potential actions, reduce that list to optimal actions, prioritize them and identifying dependencies between actions and conditional steps based on potential changes in the environment. The agent might allocate resources to each step in the sequence, or schedule actions based on environmental constraints.

For example, a robotic vacuum cleaner needs to decide the most effective path to clean a room, ensuring it covers all necessary areas without unnecessary repetition. If the sequence of actions is not well planned, the AI agent might take inefficient or redundant steps, leading to wasted resources and increased execution time.

The ReAct framework is a methodology used in AI for handling dynamic decision-making. In the ReAct framework, reasoning refers to the cognitive process where the agent determines what actions or strategies are required to achieve a specific goal.

This phase is similar to the planning phase in agentic AI, where the agent generates a sequence of steps to solve a problem or fulfill a task. Other emerging frameworks include ReWOO, RAISE and Reflexion, each of which has its own strengths and weaknesses.

### Optimization and evaluation

AI planning often involves selecting the most optimal path to achieving a goal, especially when multiple options are available. Optimization helps ensure that an agent's chosen sequence of actions is the most efficient, cost-effective or otherwise beneficial given the circumstances. This process often requires evaluating different factors such as time, resource consumption, risks and potential rewards.

For example, a warehouse robot tasked with retrieving items must determine the shortest and safest route to avoid collisions and reduce operational time. Without proper optimization, AI agents might execute plans that are functional but suboptimal, leading to inefficiencies. Several methods can be used to optimize decision-making, including:

#### Heuristic search

Heuristic search algorithms help agents find optimal solutions by estimating the best path toward a goal. These algorithms rely on heuristic functions—mathematical estimates of how close a given state is to the desired goal. Heuristic searches are particularly effective for structured environments where agents need to find optimal paths quickly.

#### Reinforcement learning

Reinforcement learning enables agents to optimize planning through trial and error, learning which sequences of actions lead to the best outcomes over time. An agent interacts with an environment, receives feedback in the form of rewards or penalties, and refines its strategies accordingly.

#### Probabilistic planning

In real-world scenarios, AI agents often operate in uncertain environments where outcomes are not deterministic. Probabilistic planning methods account for uncertainty by evaluating multiple possible outcomes and selecting actions with the highest expected utility.

### Collaboration

Single agent planning is one thing, but in a [multiagent system](https://www.ibm.com/think/topics/multiagent-system), AI agents must work autonomously while interacting with each other to achieve individual or collective goals.

The planning process for AI agents in a multiagent system is more complex than for a single agent because agents must not only plan their own actions but also consider the actions of other agents and how their decisions interact with those of others.

Depending on the [agentic architecture](https://www.ibm.com/think/topics/agentic-architecture), each agent in the system typically has its own individual goals, which might involve accomplishing specific tasks or maximizing a reward function. In many multiagent systems, agents need to work together to achieve shared goals.

These goals could be defined by an overarching system or emerge from the agents’ interactions. Agents need mechanisms to communicate and align their goals, especially in cooperative scenarios. This could be done through explicit messaging, shared task definitions or implicit coordination.

Planning in multiagent systems can be centralized, where a single entity or controller—likely an LLM agent—generates the plan for the entire system.

Each agent receives instructions or plans from this central authority. It can also be decentralized, where agents generate their own plans but work collaboratively to help ensure that they align with each other and contribute to global objectives, often requiring communication and negotiation.

This collaborative decision-making process enhances efficiency, reduces [biases](https://www.ibm.com/think/topics/ai-bias) in task execution, helps to avoid [hallucinations](https://www.ibm.com/think/topics/ai-hallucinations) through cross-validation and consensus-building and encourages the agents to work toward a common goal.

## After planning

The phases in agentic [AI workflows](https://www.ibm.com/think/topics/ai-workflow) do not always occur in a strict step-by-step linear fashion. While these phases are often distinct in conceptualization, in practice, they are frequently interleaved or iterative, depending on the nature of the task and the complexity of the environment in which the agent operates.

AI solutions can differ depending on their design, but in a typical [agentic workflow](https://www.ibm.com/think/topics/agentic-workflows), the next phase after planning is action execution, where the agent carries out the actions defined in the plan. This involves performing tasks and interacting with external systems or knowledge bases with [retrieval augmented generation (RAG)](https://www.ibm.com/think/topics/retrieval-augmented-generation), tool use and function calling ( [tool calling](https://www.ibm.com/think/topics/tool-calling)).

Building [AI agents](https://www.ibm.com/ai-agents) for these capabilities might involve [LangChain](https://www.ibm.com/think/topics/langchain). Python scripts, JSON data structures and other programmatic tools enhance the AI’s ability to make decisions.

After executing plans, some agents can use memory to learn from their experiences and iterate their behavior accordingly.

In dynamic environments, the planning process must be adaptive. Agents continuously receive feedback about the environment and other agents’ actions and must adjust their plans accordingly. This might involve revising goals, adjusting action sequences, or adapting to new agents entering or leaving the system.

When an agent detects that its current plan is no longer feasible (for example, due to a conflict with another agent or a change in the environment), it might engage in replanning to adjust its strategy. Agents can adjust their strategies using [chain of thought](https://www.ibm.com/think/topics/chain-of-thoughts) reasoning, a process where they reflect on the steps needed to reach their objective before taking action.

</details>
