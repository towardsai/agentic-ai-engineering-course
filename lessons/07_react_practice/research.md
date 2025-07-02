# Research based on provided article guidelines

## Research Results

---

<details>
<summary>What is the ReAct (Reasoning and Acting) agent framework, and how does it enable large language models to reason and interact with tools?</summary>

### Source: https://arxiv.org/abs/2210.03629
The ReAct (Reasoning and Acting) agent framework is introduced to enable large language models (LLMs) to interleave reasoning traces (thoughts) and task-specific actions in a single process. Traditionally, LLMs have been used either to generate explicit reasoning or to invoke actions (such as API calls), but not both in a tightly integrated sequence. In ReAct, the model is prompted to produce reasoning steps that justify its decisions as well as actions that interact with the external environment, such as querying a knowledge base or performing calculations.

By integrating reasoning and acting, ReAct allows LLMs to synergistically use their cognitive abilities for logical inference and their capability to interact with external tools or environments. This interleaving makes the agent more robust in solving complex, multi-step tasks and allows for dynamic adaptation based on new observations and feedback from actions. The framework is demonstrated to improve performance in various domains, such as question answering and web interactions, by providing a transparent trace of reasoning and actions taken, improving interpretability and reliability.

-----

### Source: https://react-lm.github.io
ReAct is a framework that prompts large language models to generate both verbal reasoning traces (explanatory thoughts) and concrete actions (such as tool use) in a sequence, allowing for close synergy between reasoning and acting. This approach enables LLMs to not only reason about a problem but also interact with external sources or tools to gather information, verify facts, or perform computations as needed during task execution.

This method enhances the agent’s capability to solve tasks requiring both logic and interaction, such as navigating websites, answering questions based on external documents, or performing calculations. The ReAct framework is shown to outperform models that use only reasoning or only acting, as it enables more flexible and adaptive problem-solving. By making reasoning and acting explicit in the model’s outputs, ReAct also improves the interpretability and debuggability of LLM-based agents.

-----

### Source: https://www.promptingguide.ai/techniques/react
ReAct is a prompt engineering technique and paradigm that combines reasoning and acting in LLMs, prompting them to generate a sequence of thoughts (reasoning steps) and actions (tool usage, environment interaction) for a task. The LLM is guided to alternate between these two modes in a structured, interleaved way, allowing it to reason through problems step-by-step and use external resources as needed.

This approach is particularly effective for complex tasks where a single step of reasoning or action is insufficient. By making both the reasoning process and the actions explicit, ReAct enables better transparency and traceability of the agent's decisions, and allows LLMs to solve more sophisticated tasks than using reasoning or acting alone.

-----

### Source: https://www.ibm.com/think/topics/react-agent
A ReAct agent leverages the ReAct framework to combine chain-of-thought (CoT) reasoning with external tool use, enhancing the problem-solving capabilities of large language models (LLMs). In this paradigm, the LLM’s "brain" coordinates both logical reasoning and the execution of real-world actions, such as retrieving information, invoking APIs, or interacting with databases.

Unlike traditional AI systems where reasoning and acting are separated, ReAct integrates decision-making and task execution in a single workflow. This enables ReAct agents to autonomously plan, execute, and adapt to unforeseen circumstances in complex environments. The framework supports a range of applications, from simple retrieval-augmented generation to intricate multi-agent workflows, and represents a significant evolution from conversational chatbots toward more general, adaptable AI systems capable of complex, autonomous problem-solving.

-----

</details>

---

<details>
<summary>How can you build a simple ReAct agent using Python and a large language model API, including best practices for code structure and state management?</summary>

### Source: https://www.youtube.com/watch?v=xUYFgUtucqE
This video tutorial demonstrates how to build a powerful ReAct (Reasoning + Acting) agent in Python using only plain Python and a large language model (LLM) API. Key points covered include:

- **ReAct Pattern Theory**: The ReAct pattern enables agents to both reason (analyze problems) and act (use tools) in a stepwise fashion. The agent can decide when to call external tools, execute them, observe results, and then provide a solution, making it suitable for complex workflows.
- **Tool Calling with Decorators**: The tutorial covers implementing tool-calling functionality using Python decorators, which allows the agent to flexibly invoke different tools as needed.
- **Action-Observation Loop**: A core part of the agent is an action-observation loop, where the agent:
  1. Receives a task or question.
  2. Thinks (reasons) about what action to take.
  3. Calls a tool (if required) via a decorator.
  4. Observes the output from the tool.
  5. Iterates this process until a solution is reached.
- **Best Practices**:
  - Structure your code modularly, separating tool definitions, reasoning logic, and the main loop.
  - Use decorators for tool registration to keep code clean and extensible.
  - Include debugging and optimization steps to improve agent performance and reliability.

The tutorial also recommends building on existing code for tool-calling agents and emphasizes the importance of code clarity and maintainability when constructing more advanced agentic workflows.

-----

### Source: https://www.dailydoseofds.com/ai-agents-crash-course-part-10-with-implementation/
This article provides a detailed guide for implementing a ReAct agent from scratch in Python, leveraging an LLM. The main points include:

- **Understanding the ReAct Pattern**: The ReAct agent alternates between reasoning (using an LLM to generate next steps) and acting (calling external tools or APIs).
- **Code Structure**:
  - Define a core agent class that holds the current state, including the conversation history, observations, and any intermediate results.
  - Implement a loop where the agent:
    1. Prompts the LLM for the next action or reasoning step.
    2. Parses the LLM output to determine whether to act (call a tool) or provide a final answer.
    3. Updates the state with new observations or results.
- **State Management**:
  - Use a structured state variable (such as a dictionary or a dedicated class) to keep track of all context needed for the agent to reason and act in sequence.
  - Maintain a clear boundary between LLM reasoning and tool invocation logic.

- **Best Practices**:
  - Make your tool-calling logic modular so it is easy to add or remove tools.
  - Carefully parse and validate LLM outputs to avoid errors or hallucinations.
  - Log all actions and observations for easier debugging and traceability.

The approach emphasizes explicit state management and modular code, ensuring clarity and extensibility.

-----

### Source: https://www.neradot.com/post/building-a-python-react-agent-class-a-step-by-step-guide
This step-by-step guide details constructing a Python ReAct agent class, focusing on good code structure and method organization:

- **Class Interface**: Begin by clearly defining the interface for your agent class, laying out public methods such as `reason()`, `act()`, and `observe()`.
- **Minimal Setup**: Implement initialization to set up required components (e.g., LLM API connector, list of available tools).
- **Reasoning and Messaging**: Add methods like `reason()` for generating LLM prompts and `build_messages()` for constructing message histories.
- **Action Loop**: The agent operates in a loop, alternating between reasoning (LLM call), acting (tool call), and observing (updating state with results).
- **State Handling**: Use instance variables to track the current state, including conversation history, tool results, and whether the final answer has been reached.
- **Extensibility**: The design encourages adding more tools or capabilities by simply expanding the class or adding new methods.

This guide emphasizes a clean, object-oriented approach and suggests keeping code maintainable by separating responsibilities into distinct methods.

-----

### Source: https://mlpills.substack.com/p/diy-14-step-by-step-implementation
This article explains how to build a basic ReAct agent using the LangGraph (LangChain's graph-based framework) and Python:

- **ReAct Loop**: The agent alternates between:
  - *Thinking*: Using the LLM to reason about next steps.
  - *Acting*: Calling a tool or API as needed.
  - *Observing*: Integrating the result of the tool call back into its ongoing reasoning.

- **Implementation Steps**:
  - Set up the environment with LangChain and LangGraph.
  - Explicitly define graph nodes for each stage (reasoning, acting, observing) and conditional edges for transitions.
  - Implement a stateless agent for single-turn questions (no memory), and note the possibility of extending to a stateful agent with conversation memory.

- **Best Practices**:
  - Avoid using pre-built agent utilities for educational purposes; explicitly define each step for clarity.
  - For stateful agents, manage memory to track conversation history, enabling multi-turn reasoning.

- **Example Use Case**: The agent can, for example, use a Wikipedia search tool when it determines its knowledge is insufficient for a query.

This guide highlights the value of explicit, modular design, and demonstrates both stateless and stateful approaches to agent state management.

</details>

---

<details>
<summary>What are effective techniques for designing prompts and parsing output for controlled Thought-Action-Observation loops in LLM-based agents?</summary>

### Source: https://huggingface.co/learn/agents-course/en/unit1/agent-steps-and-structure
Effective prompt design for controlled Thought-Action-Observation (TAO) loops in LLM-based agents involves carefully structuring the system prompt to embed the agent's behavior, the available tools, and the TAO cycle itself. In many agent frameworks, these rules and workflow guidelines are directly embedded into the system prompt, ensuring each cycle adheres to a defined logic. The system prompt should explicitly define:
- The agent’s expected behavior.
- The tools or actions the agent can access and invoke.
- The structure and order of the TAO cycle, so the LLM understands to reason (Thought), act (Action), and then reflect on output (Observation) in a loop until the task is complete.
This prompt engineering ensures the LLM maintains the correct workflow, remains within operational boundaries, and continually updates its reasoning based on new observations. The TAO cycle is often implemented as a loop—similar to a while loop in programming—repeating until the agent’s objective is met.

-----

### Source: https://publish.obsidian.md/drjerryasmith/Notes/Public/Thought-Action-Observation+Loop
The Thought-Action-Observation Loop is a structured reasoning paradigm widely used in agent frameworks like ReACT to enhance LLM problem-solving. The loop consists of:
- Generating a Thought: The LLM plans the next step or infers what needs to happen.
- Performing an Action: The LLM executes an operation, such as querying a database or invoking an API.
- Receiving an Observation: The LLM integrates the resulting output into its context for further reasoning.
The process iterates until the model converges on a solution or completes the task.

For prompt design and output parsing:
- Prompts should clearly delineate the three components: Thought, Action, and Observation—both in instruction and expected output format.
- Parsing can be reliably performed by enforcing structure (e.g., using explicit labels like “Thought:”, “Action:”, “Observation:”) so that downstream code or agents can easily extract and process each step.
- This explicit structuring in prompts and outputs maximizes reliability and interpretability, allowing for effective chaining of reasoning and tool use.

-----

### Source: https://www.gocodeo.com/post/building-react-agents-practical-techniques-for-tracing-acting-observing-in-llm-workflows
For ReAct agents, the loop is strictly: Thought (reasoning about the problem), Action (invoking a tool or function call with a structured argument), and Observation (processing the tool’s response). Effective techniques include:
- Using highly structured prompt templates that force the LLM to reason, act, and observe in a predictable sequence.
- Designing the output format with explicit, machine-parseable markers for each stage (e.g., “Thought: ...”, “Action: ...”, “Observation: ...”) to facilitate reliable extraction of each step during agent execution.
- Ensuring that tool calls are formatted as structured, valid function calls or API queries, which can be programmatically executed and monitored.
- Feeding the output (Observation) back into the agent’s context for the next iteration, maintaining a continuous and coherent workflow.
Such structure helps in both controlling the agent’s behavior and in automating the parsing and processing of the agent’s reasoning and actions in multi-step workflows.

-----

### Source: https://www.dailydoseofds.com/ai-agents-crash-course-part-10-with-implementation/
In practical implementations of the ReAct agentic pattern, prompt design and output parsing are critical for maintaining a robust TAO loop. Effective strategies include:
- Appending each Thought and Observation step to the ongoing dialogue context, so the LLM has access to its own history and can reason coherently over multiple steps.
- Marking final answers explicitly (e.g., with “Answer:” or “Final Answer:”) to signal the end of the loop and distinguish between intermediate and final outputs.
- Maintaining a consistent, labeled format for each stage (Thought, Action, Observation) to ensure both human and machine agents can parse and act on the outputs.
- Supporting multi-step tool chaining, where the agent can perform several tool uses in sequence, each time updating its context with new observations before reasoning again.
This approach ensures the agent remains in control, tracks its reasoning process, and can be reliably steered and monitored programmatically.

</details>

---

<details>
<summary>What are the core steps and considerations for implementing the agentic control loop (Thought -> Action -> Observation) in AI agents using modern frameworks?</summary>

### Source: https://www.sketchdev.io/blog/agentic-ai-implementation-guide
Implementing the agentic control loop in AI agents involves coordinated execution and continuous learning. In complex workflows, multiple specialized agents collaborate, each contributing distinct expertise—such as customer analysis, inventory assessment, or crafting communications—while a coordinator agent manages context-sharing and timing to ensure coherence. Agents can negotiate priorities and resolve conflicts, maintaining system adaptability as circumstances evolve.

Crucially, agentic systems incorporate ongoing learning and iteration. Individual agents refine their capabilities, coordination mechanisms become more efficient, and the overall system learns to escalate issues or try alternative strategies as needed. This feedback loop leverages both automated performance analysis and human input, driving continuous improvement.

Real-world deployment requires acknowledging system limitations. While agentic AI excels at tasks involving coordination, consistency, and rapid information processing, it struggles with deep ambiguity, creative problem-solving, and complex human dynamics. Effective implementations amplify human strengths—rather than attempt to replace them—by designing workflows that incorporate both AI and human judgment within clearly defined parameters.

-----

### Source: https://www.plivo.com/blog/ai-agentic-workflows-how-to-implement-them/
Successful implementation of agentic control loops begins with setting highly specific, actionable goals. Vague objectives like "improve efficiency" are insufficient; instead, define measurable results (e.g., "reduce response time from 10 minutes to 2 minutes"). This clarity guides agent behavior within the Thought → Action → Observation loop.

Next, construct teams of specialized agents, each assigned to well-defined tasks aligned with their capabilities—mirroring how employees specialize in particular roles. For example, in healthcare, one agent might analyze medical data while another schedules appointments. In finance, distinct agents might handle fraud detection and customer communication separately. Careful assignment of agents to workflow steps ensures optimal performance and clarity of responsibility within the control loop.

-----

### Source: https://globalcybersecuritynetwork.com/blog/how-to-build-agentic-ai/
A stepwise approach is recommended for building agentic AI systems. Begin in a controlled, limited environment with strict operational guardrails to mitigate risk. Integrate real-time monitoring and alerting mechanisms to observe agent behavior and outcomes, enabling immediate detection and correction of anomalies. Implement fallback strategies to ensure continuity and safety if the agent encounters unexpected scenarios or failures.

These foundational practices ensure the Thought → Action → Observation loop operates safely, with the system able to learn from each cycle through careful monitoring and structured feedback. Starting small and incrementally expanding scope allows for robust, scalable agentic AI deployments.

-----

### Source: https://www.dailydoseofds.com/ai-agents-crash-course-part-14-with-implementation/
To improve agentic systems, introduce validation guardrails that catch issues before they impact live operations. Incorporate a human-in-the-loop step, empowering people to guide or approve agent outputs, especially for high-stakes or ambiguous tasks. Allow agents to collaborate, sharing information and dividing labor based on their respective strengths. This approach ensures quality control and helps the system adapt effectively within the agentic control loop, especially when handling novel or complex scenarios.

</details>

---

<details>
<summary>What challenges and limitations exist when building and deploying ReAct agents or agentic LLM systems, and how can these be addressed in practical applications?</summary>

### Source: https://www.ibm.com/think/topics/react-agent
A ReAct agent combines chain of thought (CoT) reasoning with external tool use, enabling it to reason step-by-step and interact with APIs or tools as needed. While this approach allows for flexible and dynamic problem solving, it also introduces several practical challenges:
- The agent’s performance is highly dependent on the quality of prompts and the accuracy of external tool outputs, which can introduce errors or biases.
- The iterative reasoning and acting process can lead to increased latency and resource consumption, especially for complex or multi-step tasks.
- Ensuring consistent and reliable behavior in open-ended or ambiguous scenarios requires careful system design and ongoing monitoring.
To address these challenges in practical applications, developers must focus on robust prompt engineering, rigorous validation of tool integrations, and the implementation of monitoring and fallback mechanisms to handle unexpected errors or failures.

-----

### Source: https://www.promptingguide.ai/research/llm-agents
LLM agents, including those using the ReAct framework, face several notable challenges and limitations:
- Role-playing capability: Current LLM agents have limited ability to maintain consistent personas or roles over long interactions, which can affect reliability in complex workflows.
- Tool integration: Effectively integrating and managing external tools remains a technical challenge, especially as the number of tools or their complexity increases.
- Error propagation: Mistakes made in one reasoning or action step can cascade, leading to compounding errors in downstream steps.
- Generalization: LLM agents can struggle to generalize their behavior across diverse tasks and environments.
To mitigate these issues, practical strategies include: designing modular agent architectures, implementing robust error handling and recovery strategies, and ongoing evaluation and fine-tuning based on real-world usage data.

-----

### Source: https://github.com/NVIDIA/AgentIQ/blob/develop/docs/source/components/react-agent.md
ReAct agents come with several specific limitations:
- Require More LLM Calls: The step-by-step reasoning and acting process results in multiple LLM calls per task, increasing latency and API costs.
- Prompt-Sensitivity & Tuning Overhead: ReAct agents are highly sensitive to prompt design, necessitating careful and ongoing tuning. Poor prompts can lead to inefficient reasoning or incorrect tool usage.
- Risk of Hallucination: The reasoning steps can introduce hallucinations—incorrect or fabricated information—especially if the agent misinterprets tool responses. There are no built-in constraints to prevent invalid reasoning paths.
- Complexity in Long Chains: For tasks involving many steps, errors early in the process can propagate throughout, making debugging and maintaining consistency difficult.
- Lack of Parallelism: ReAct agents operate sequentially, which limits their efficiency for tasks that could benefit from parallel processing (e.g., making multiple API calls at once).
Addressing these challenges involves prompt engineering best practices, adding safeguards against hallucination, and considering alternative agent architectures or hybrid strategies for tasks requiring parallelism or high reliability.

-----

### Source: https://dev.to/jamesli/react-vs-plan-and-execute-a-practical-comparison-of-llm-agent-patterns-4gh9
When comparing the ReAct pattern to alternatives like Plan-and-Execute, several practical limitations emerge:
- Performance and Cost: The ReAct pattern, due to its iterative reasoning-action loop, can have longer response times and higher token/API usage, impacting operational costs.
- Pattern Matching: The effectiveness of ReAct versus other agent patterns depends on the nature of the task. For instance, Plan-and-Execute may be better for tasks that can be pre-planned, while ReAct excels in more dynamic, interactive scenarios.
- Hybrid Strategies: In practice, combining ReAct with other patterns (hybrid strategies) can yield better results for complex applications.
In practice, careful selection of the agent pattern based on task characteristics, scene analysis, and response/cost requirements is recommended. Hybrid approaches can also help address the limitations of a single pattern.

-----

### Source: https://www.comet.com/site/blog/using-the-react-framework-in-langchain/
A key limitation of ReAct agents is their heavy reliance on external tool integration. The agent’s effectiveness is partially determined by the capabilities and reliability of these external tools:
- External Tool Dependency: If the external tools are inaccurate or unreliable, the overall agent may introduce biases or errors from those sources.
- Bias and Inaccuracy: As with any system that ingests external data, the risk of propagating external biases or inaccuracies remains.
In practical terms, mitigating these challenges involves careful selection and validation of external tools, regular monitoring of tool outputs for accuracy, and implementing fallback or redundancy mechanisms to minimize the impact of tool failures.

</details>

---

<details>
<summary>What are the specific roles and responsibilities of the 'scratchpad' or conversation history in maintaining context and improving ReAct agent performance?</summary>

### Source: https://langchain-ai.github.io/langgraph/how-tos/create-react-agent-manage-message-history/
The conversation history in a ReAct agent plays a critical role in maintaining context for long or complex interactions. Since message history can quickly surpass the context window of large language models (LLMs), it is essential to manage this history efficiently to ensure the agent has access to relevant information without exceeding model limits. The primary responsibilities and strategies for managing conversation history include:

- **Message Trimming**: Removing the first or last N messages to keep the most relevant context within the available token window.
- **Summarization**: Condensing earlier messages into a summary, which is then used to replace the detailed history, thus preserving essential context while minimizing token usage.
- **Custom Strategies**: Implementing tailored approaches like filtering based on message type or recency.

To integrate these strategies into a ReAct agent, developers define a `pre_model_hook` function that processes the message history before each LLM invocation. This function can trim, summarize, or otherwise modify the history to ensure the agent operates within constraints while maintaining necessary context. The revised message history is then provided as input to the model, thereby directly influencing the agent’s ability to understand and respond accurately in ongoing conversations.

This management of conversation history—sometimes referred to as a 'scratchpad'—is crucial in sustaining coherent multi-turn interactions and optimizing agent performance by ensuring that each model call receives the most relevant and contextually rich information possible.

-----

### Source: https://api.python.langchain.com/en/latest/agents/langchain.agents.react.agent.create_react_agent.html
The 'agent_scratchpad' in LangChain's ReAct agent framework serves as a string that contains previous agent actions and tool outputs. Its core responsibility is to provide a running log of the agent’s reasoning process, including the tools called and the results returned during the agent's step-by-step problem-solving process.

This scratchpad is included in the prompt to the LLM, ensuring that the model is continually aware of what actions have already been taken and what information has already been retrieved. By doing so, it allows the agent to:

- Maintain a coherent chain of reasoning across multiple steps.
- Avoid repeating actions or tool calls unnecessarily.
- Reference prior tool outputs for more informed decision-making.

The explicit inclusion of the scratchpad in the agent’s prompt template is fundamental for maintaining context, enabling the agent to build upon previous steps to reach accurate and efficient solutions.

-----

### Source: https://smith.langchain.com/hub/langchain-ai/react-agent-template?organizationId=d9ea56f8-ed98-5632-b376-dff134eb0275
In the standard prompt template for a ReAct agent, both the previous conversation history and the agent_scratchpad are explicitly included. The prompt is structured as follows:

> Begin! Previous conversation history: {chat_history}. New input: {input}. {agent_scratchpad}.

This design ensures that:

- **chat_history** provides the agent with relevant previous user and agent messages, maintaining the conversational context.
- **agent_scratchpad** supplies a detailed log of internal reasoning, actions, and tool outputs up to the current point.

By combining these elements, the agent receives a comprehensive context, allowing it to make informed decisions, avoid redundant actions, and deliver responses that are coherent with the ongoing conversation. This approach significantly enhances agent performance in multi-turn, tool-augmented dialogues.

-----

### Source: https://www.aurelio.ai/learn/langchain-agent-executor
The scratchpad in a LangChain agent is implemented as a list of messages that alternate between the AI (representing tool calls) and the tool (representing tool execution outputs). Its main responsibilities include:

- Recording each tool invocation (e.g., which tool was called, with what arguments).
- Capturing the output of each tool call and appending it to the scratchpad.
- Providing a clear, step-by-step trace of the agent’s reasoning and tool usage process.

During execution, each tool call and its corresponding output are added to the scratchpad, and this record is continually updated as the agent reasons through a query. The final answer, once generated, is appended to the overall chat history, which includes both user inputs and agent responses. This mechanism ensures transparency, traceability, and robust context retention throughout the agent's multi-step reasoning process.

-----

</details>

---

<details>
<summary>How does the Google Gemini LLM API's function calling and structured output capabilities compare to other popular LLMs (e.g., OpenAI GPT-4, Anthropic Claude) when building ReAct-style agents?</summary>

### Source: https://ai.google.dev/gemini-api/docs/function-calling
Google Gemini's function calling capability lets developers define functions (tools) that the model can call instead of generating plain text responses. Developers declare these functions with structured schemas, enabling the model to determine when and how to call specific functions based on user queries. This integration allows you to connect Gemini-powered agents to external APIs and tools, supporting advanced agentic behaviors such as ReAct-style agents. The model analyzes user input, identifies the need to invoke a declared function, and returns a response specifying the function name and required parameters in a structured format. This mechanism enables reliable extraction of structured data and robust chaining of external actions, which is essential for building agents that can reason and act iteratively.

The documentation emphasizes that Gemini’s function calling is natively supported and designed to integrate seamlessly into developer workflows, providing a simple yet powerful approach to structured output and tool use. This stands in contrast to earlier prompt engineering techniques, which attempted to coerce models into producing JSON or other structured outputs without explicit function declaration support.

-----

### Source: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling
Gemini’s function calling on Google Cloud Vertex AI requires developers to declare available functions using schemas compatible with OpenAPI. When invoking the model, both the user prompt and these function declarations are sent. The model then determines if a function should be called and returns a structured response containing the function name and parameters. This schema-driven approach ensures strong typing and reliable parameter extraction, which is critical for building agents that interact with external APIs in a loop (such as in ReAct-style architectures).

The structured output from Gemini is designed to be programmatically consumed, making it easier for developers to build multi-step, tool-using agents. This approach provides a clearer and more robust interface for function invocation and output parsing compared to models that only support text-based prompting.

-----

### Source: https://codelabs.developers.google.com/codelabs/gemini-function-calling
Gemini’s function calling is specifically designed to facilitate structured data outputs and programmatic tool use. The workflow involves:

- Developers defining function declarations within tools in the API call.
- On receiving user input, Gemini returns a function call response, including the function name and parameters as structured data.
- The application then uses this output to invoke the corresponding external API or function.
- The API response is returned to Gemini for follow-up actions or to generate the final user-facing output.

This structured loop supports multi-step reasoning and acting, as needed in ReAct-style agents, by allowing the agent to call external functions, retrieve results, and possibly iterate through further function calls based on new information. The emphasis on structured, schema-based output makes Gemini particularly well-suited to scenarios where reliable tool use and chaining are required, in contrast to relying solely on prompt-based output formatting.

-----

### Source: https://ai.google.dev/gemma/docs/capabilities/function-calling
Function calling with Google's Gemma (related to Gemini) involves constructing prompts that specify both the desired output format and the available functions. By explicitly defining these functions and their expected parameters, developers can ensure that the model produces reliably structured outputs and invokes external functionality as part of the agent’s reasoning loop. This approach enables the development of agents that can interact with APIs in a controlled, robust manner, supporting iterative action and reasoning cycles.

-----

### Source: https://firebase.google.com/docs/ai-logic/function-calling
Function calling in the Gemini API, also referred to as tool use, allows models to interact with external APIs or functions as part of their response generation process. This is particularly useful in agentic settings, where the model might need to call multiple tools or APIs to fulfill a user’s request. By formally supporting function calls and structured outputs, Gemini provides a more reliable and maintainable way to build agents that require external actions, as compared to earlier prompt-only approaches.

-----

</details>

---

<details>
<summary>What are common pitfalls or failure modes (such as mis-parsing, prompt drift, or tool mishandling) encountered when parsing LLM outputs in step-by-step ReAct workflows, and how can they be mitigated?</summary>

### Source: https://github.com/langchain-ai/langchain/discussions/7403
A common failure mode in parsing LLM outputs within ReAct workflows is when the LLM produces both a final answer and a parse-able action in a single output. This hybrid output confuses the output parser, which expects only one type of response at a time. For example, the LLM might inadvertently combine the "Question/Thought" pattern, resulting in outputs that cannot be parsed properly. To mitigate this, one recommended approach is to configure the agent to handle parsing errors explicitly, such as by instructing the LLM to "Check your output and make sure it conforms! Do not output an action and a final answer at the same time." Using a properly configured agent (e.g., with AzureChatOpenAI and strict error handling instructions) can ensure that the output adheres to the expected format, allowing the workflow to proceed correctly without mis-parsing or prompt drift.

-----

### Source: https://community.n8n.io/t/handling-llm-output-and-truncation-in-workflows/106830
One pitfall encountered in step-by-step ReAct workflows is output truncation, especially when large inputs (such as batches of 250 articles) are sent to the LLM. This leads to incomplete outputs, which in turn cause parsing issues. The result is often partial, malformed, or entirely missing data that the parser cannot handle effectively. To mitigate this, it is recommended to break large tasks into smaller batches, ensuring that the LLM does not hit token or output length limits. Keeping the interaction size manageable helps maintain output integrity and improves downstream parsing reliability.

-----

### Source: https://github.com/hwchase17/langchain/issues/1358
A major challenge in ReAct workflows is the inherent randomness in LLM outputs: the model may not strictly follow the instructed format, causing parsers to fail. Several mitigation strategies are suggested:

- Use a specific tool for error handling, such as a search tool or prompting the user for more details, though these may not always yield meaningful results.
- Implement a retry mechanism that catches parse exceptions and tries again, potentially using the LLM itself to reformat or clarify its output.
- Switch to a structured output format such as JSON, along with a dedicated JSON parser, to reduce ambiguity.
- Fine-tune the LLM with examples of properly formatted inputs and outputs to reinforce the desired output structure.

These approaches help address failures due to unexpected output formats, prompt drift, or tool mishandling by making the output more predictable and easier to parse.

-----

### Source: https://minimaxir.com/2023/07/langchain-problem/
The ReAct workflow, while effective in certain contexts, can be costly, complex, and challenging to use in practice. The intricacy of chaining reasoning, action planning, and tool invocation increases the likelihood of mis-parsing or workflow failures, especially as the number of steps grows. This makes the system less robust and more prone to breakdowns if not carefully managed and monitored. Mitigating these issues often requires additional engineering effort, robust error handling, and possibly simplifying the workflow where possible to reduce failure modes.

-----

### Source: https://www.width.ai/post/react-prompting
ReAct prompting combines reasoning, action planning, and integration of external knowledge, but this complexity introduces potential pitfalls. Common issues include mis-parsing of LLM responses, prompt drift (where the model's output style or structure deviates over time), and tool mishandling (incorrect or inappropriate use of integrated tools). Addressing these problems involves designing clear, explicit prompts, using structured response formats, and incorporating fallback mechanisms to recover from unexpected outputs. Ensuring consistent, parse-able outputs is critical for reliable step-by-step reasoning and action workflows in LLM-powered agents.
-----

</details>

---

<details>
<summary>Are there recommended testing and debugging strategies specifically suited for ReAct agent implementations to help identify and fix errors in the agentic control loop?</summary>

### Source: https://www.gocodeo.com/post/building-react-agents-practical-techniques-for-tracing-acting-observing-in-llm-workflows
For ReAct agent implementations, recommended debugging and testing strategies include:

- **Action and Trace Logging**: Implement structured logs that capture each reasoning step, action taken, and observation received. This enables you to reconstruct the agent's full decision path and diagnose where failures or unexpected behaviors occur in the control loop.
- **Stepwise Execution**: Design agents to allow step-by-step execution, so you can pause after each thought, action, or observation. This approach helps in isolating faulty logic or tool-integration issues at each stage.
- **Unit Testing for Tool Invocations**: Test each tool or API the agent may call independently, verifying input validation, expected outputs, and error handling, before integrating into the agentic loop.
- **Simulated Environments**: Use controlled, synthetic test environments where you can predict agent behavior. By comparing the agent’s decisions and outputs to expected traces, you can identify reasoning or state management errors.
- **Trace Comparison**: Automatically compare execution traces from test runs to “golden” reference traces. Discrepancies highlight regressions or subtle bugs in the agent’s logic or control flow.

These strategies are tailored for ReAct-style workflows, where the agent’s reasoning, actions, and observations interleave in complex ways, making granular visibility and traceability essential for effective debugging and validation.

-----

### Source: https://apxml.com/courses/agentic-llm-memory-architectures/chapter-6-evaluation-optimization-agentic-systems/debugging-strategies-agents
Recommended strategies for testing and debugging ReAct agents include:

- **Visualization of Execution Flow**: Use diagrams or tools to visualize the full execution trace of the agent, representing goals, thoughts, actions, observations, and results as nodes. This helps quickly identify where the agent’s reasoning diverges from expectations or where control loop errors arise.
- **Step-Level Inspection**: Examine the agent’s internal state, inputs, and outputs at each step of the loop (reasoning, action selection, observation handling). This allows you to pinpoint errors in the reasoning process or action-effect mismatches.
- **Test Scenarios with Known Outcomes**: Design scenarios with predetermined correct answers or behaviors. Use these to verify that the agent produces correct intermediate steps and final results.
- **Logging and Trace Analysis**: Maintain detailed logs for every step in the agent’s control loop, including timestamps, input prompts, selected actions, observed results, and internal reasoning. Analyze these logs to debug failures or performance bottlenecks.

These methods are particularly effective for complex or multi-step ReAct agents, where bugs often stem from reasoning drift, incorrect state transitions, or unexpected tool/API responses.

-----

</details>

---

<details>
<summary>What small-scale, practical extensions or enhancements (e.g., logging, custom tool support, output validation) can be added to a simple ReAct agent to make it more robust or feature-complete for AI engineering novices?</summary>

### Source: https://docs.nvidia.com/agentiq/1.0.0/components/react-agent.html
The NVIDIA AgentIQ documentation for the ReAct Agent highlights several features and extensibility points that can be leveraged to enhance robustness and feature-completeness, especially for AI engineering novices:

- **Pre-built Tools**: The agent supports integration with a library of core tools, making it easy to extend capabilities without building new components from scratch. This modularity allows users to add domain-specific or custom tools, enhancing practical utility.
- **ReAct Reasoning**: The agent performs reasoning between tool calls and can utilize tool names and descriptions for better interpretability. This structure makes it possible to add logging of tool decisions or visualize reasoning steps for improved transparency.
- **Logging and Observability**: The documentation suggests that the ReAct agent is designed to facilitate observability around tool usage and reasoning. Implementing logging of each tool invocation, input, and resulting output can help users understand agent decisions and debug issues.
- **Custom Tool Support**: The architecture encourages users to define and register custom tools, which can be a simple and practical way for novices to tailor the agent for their specific use cases.
- **Output Validation**: Output from tools and the agent itself can be validated or filtered before being acted upon, helping to ensure correctness and robustness of the workflow.
- **Step-by-Step Reasoning**: By exposing intermediate reasoning steps and actions, the agent can be enhanced to support interactive debugging, stepwise execution, or even undo functionality.

These features provide a solid foundation for practical enhancements such as:
- Detailed logging of reasoning steps and tool outputs.
- Simple plug-and-play custom tool modules.
- Output validation hooks to catch errors or unexpected results.
- Interactive step-through or visualization of agent decisions.

These extensions can help novices build confidence and understanding in how their ReAct agent operates and why it makes certain decisions.

-----

### Source: https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/
The LangGraph guide on implementing a ReAct agent from scratch suggests several practical enhancements for robustness:

- **Custom Tool Integration**: The agent is designed to be initialized with a flexible set of tools, making it straightforward for users to add custom tools tailored to their needs.
- **Verbose Mode/Logging**: The example code demonstrates enabling verbose output, which logs each decision, tool call, and result. This is a simple but highly effective way to provide transparency for users learning how the agent operates.
- **Prompt Inspection**: Users can inspect and modify the agent's prompt, which helps understand and debug agent reasoning. Logging prompt changes or the agent's intermediate reasoning can be valuable for learning and troubleshooting.
- **Output Observation**: The framework records the output of each tool as an "observation," which can be logged or validated before further reasoning steps.
- **Stepwise Execution**: The agent repeats actions until a stopping condition is reached. Users can add hooks to log or validate output at each step, making it easier to spot where reasoning may go astray.

These features support practical enhancements such as:
- Detailed step-by-step logging.
- Plug-in output validators at each reasoning step.
- Easy addition of new or custom tools with clear documentation.

Such extensions make the agent more transparent, reliable, and accessible for AI engineering novices.

-----

### Source: https://www.comet.com/site/blog/using-the-react-framework-in-langchain/
Comet ML’s blog on using the ReAct framework in LangChain highlights workflow features supporting practical extensions:

- **Tool Descriptions and Selection**: The agent uses descriptions for each tool and selects them dynamically. Users can add or edit tool descriptions for better clarity and maintainability.
- **History Tracking**: The agent tracks the history of tool usage, tool inputs, and outputs ("observations"). This history can be logged or exported for auditability, debugging, or visualization.
- **Verbose Mode**: The framework supports running the agent in verbose mode, which logs each step in the agent’s reasoning and actions. This is valuable for novices to learn how agents interact with tools.
- **Output Recording**: Every tool output is recorded and can be validated or inspected before being used in further reasoning.
- **Prompt Customization**: Users can inspect and adjust the agent’s prompt, which is useful for tailoring agent behavior and for educational purposes.

These features easily enable practical enhancements:
- Comprehensive logging of tool calls, reasoning, and outputs.
- Output validation steps after each tool invocation.
- Custom tool support through an extensible tool registry.
- Exportable histories for further analysis.

These enhancements contribute to agent robustness and make the system more approachable for those new to AI engineering.

-----

</details>

---

<details>
<summary>How can AI engineering novices enhance a simple ReAct agent with practical features such as logging, output validation, and custom tool integration in Python?</summary>

### Source: https://github.com/mattambrogi/agent-implementation
This repository provides a basic Python implementation of the ReAct Agent Pattern for LLMs. The code is designed to be straightforward and is adapted from Simon Willison's approach. It demonstrates how to structure the agent's reasoning and acting loop using prompt engineering and parsing logic. While the implementation is intentionally simple, it forms a foundation that novices can extend with practical features:

- **Logging:** You can add Python logging statements at key steps (e.g., before/after calling the LLM, when a tool is invoked, or when output is parsed) to monitor the agent's behavior and facilitate debugging.
- **Output Validation:** After parsing the LLM's output, implement checks to validate that the output matches expected formats or contains necessary fields before processing further.
- **Custom Tool Integration:** The agent's action system is extensible. You can define new Python functions as tools, register them in the agent's tool registry, and ensure the prompt guides the LLM to invoke these tools using specific action names.
- The modular structure allows for easy insertion of additional utility functions or hooks for advanced features such as error handling or interaction tracking.

This basic setup enables AI engineering novices to iteratively enhance their agent with best practices and additional capabilities, building on a clear and well-documented codebase.

-----

### Source: https://www.neradot.com/post/building-a-python-react-agent-class-a-step-by-step-guide
This step-by-step guide outlines the process of building a Python ReAct agent class from scratch, emphasizing extensibility and practical enhancements:

- **Class Interface:** Begin by defining a flexible class interface to encapsulate agent functionality.
- **Logging:** Integrate logging early in the development process. Use Python's `logging` library to record inputs, outputs, intermediate reasoning steps, and tool invocations. This allows for easier debugging and performance monitoring.
- **Output Validation:** After the agent receives a response from the LLM, implement methods to validate the output structure (e.g., checking for required keys like "Thought" and "Action"). Raise exceptions or handle errors gracefully if the output does not meet expectations.
- **Custom Tool Integration:** The guide provides patterns for defining new tools as Python methods or external functions. Register these tools with the agent and update the prompt template to inform the LLM about the available tools and their usage.
- **Extensibility:** The class-based design encourages the addition of new features, such as memory modules or advanced output parsers, without disrupting the core agent logic.

By systematically following these practices, novices can scaffold a robust ReAct agent and expand its capabilities with real-world, production-ready features.

-----

### Source: https://www.dailydoseofds.com/ai-agents-crash-course-part-10-with-implementation/
This article describes the process of building a ReAct agent from scratch using only Python and an LLM. Key practical enhancements for novices include:

- **Prompt Design and Parsing:** The agent's reasoning and acting abilities stem from well-crafted prompt templates and robust parsing logic. Ensure your prompt instructs the LLM to reply in a structured format (e.g., with "Thought" and "Action" annotations) and parse outputs accordingly.
- **Logging:** Incorporate logging throughout the agent's reasoning-acting loop. Log incoming user queries, LLM outputs, tool invocations, and final responses to enable transparency and facilitate debugging.
- **Output Validation:** After receiving the LLM's response, parse and validate that it conforms to the expected structure before executing any actions. Handle ill-formed outputs with fallback logic or error messages.
- **Custom Tool Integration:** The agent can interact with various tools by defining Python functions (e.g., for math, database queries) and exposing them via the prompt. Extend the agent by adding new tools and updating the prompt to reflect their capabilities.
- **Modularity:** Structure your code so that logging, validation, and tool integrations are modular and easy to maintain or extend.

By focusing on clear prompt engineering, structured output handling, and modular code, novices can iteratively add practical features to a simple ReAct agent.

-----

### Source: https://api.python.langchain.com/en/latest/agents/langchain.agents.react.agent.create_react_agent.html
The `langchain.agents.react.agent.create_react_agent` function in LangChain provides a structured approach to building a ReAct agent with extensibility for practical features:

- **Logging:** While not built-in, logging can be incorporated by wrapping calls to the LLM, tool invocations, and output parsing within logging statements using Python's `logging` module.
- **Output Validation:** The function accepts an optional `output_parser` parameter, allowing users to supply a custom `AgentOutputParser` for robust parsing and validation of LLM outputs before further processing or tool execution.
- **Custom Tool Integration:** The agent is initialized with a sequence of tools (`tools` parameter) that can be any Python functions adhering to the required interface. The `tools_renderer` parameter controls how the list of tools is described to the LLM in prompts, ensuring the agent is aware of available actions.
- **Prompt Customization:** Users can define custom prompts to instruct the agent on the structure of its reasoning and acting, supporting new features or behaviors.

By leveraging these parameters and extending the agent's initialization logic, AI engineering novices can enhance their agent with logging, output validation, and a wide variety of custom tools, following best practices for modularity and robustness.

-----

### Source: https://github.com/miguelvc6/react-agent/blob/main/README.md
This Python ReAct agent implementation offers features and design considerations that support practical enhancements for novices:

- **Unified API Interface:** The agent supports seamless interaction with different LLM providers, making it flexible for experimentation and extension.
- **Tool Integration:** Out of the box, the agent supports tools for SQL queries and mathematical calculations, demonstrating how to integrate Python functions as callable tools. To add custom tools, define new functions and register them in the agent's tool registry.
- **Memory Management:** The agent maintains a simple memory for past interactions, which can be extended or customized for more sophisticated state tracking.
- **Reasoning and Acting:** The agent uses reflection for dynamic decision-making, combining language model reasoning with tool execution.
- **Extensibility:** The codebase is structured to facilitate the addition of new features. Novices can implement logging by adding logging statements to key methods, and output validation can be added by checking the structure and content of LLM outputs before proceeding with tool actions.
- **Assistants:** The agent can decompose complex queries into sub-questions, which can be recursively solved using available tools.

These features and modular design choices offer a clear path for AI engineering novices to enhance a simple ReAct agent with logging, output validation, and custom tool integration in Python.

-----

</details>

---

<details>
<summary>What are reliable strategies for managing the ReAct agent's scratchpad or conversation history in Python implementations to ensure context retention and avoid exceeding token limits?</summary>

### Source: https://github.com/mattambrogi/agent-implementation
This GitHub repository provides a basic ReAct agent implementation in Python and can be used as a starting point to understand managing the agent’s scratchpad or conversation history. While it does not explicitly detail token management strategies, the simplicity of the implementation implies that the agent maintains a running history of observations and actions that constitute the scratchpad. To ensure context retention, the common practice is to keep appending new reasoning and action steps to the scratchpad. However, to avoid exceeding token limits, developers typically implement truncation strategies on the scratchpad content, such as keeping only the most recent relevant steps or summarizing older parts of the conversation. This repo can be a reference for minimal implementations where manual management of the scratchpad is done by controlling the size or content of the stored dialogue steps. This approach is essential when using large language models with token constraints since the prompt (including the scratchpad) must fit within the model’s maximum token window.

### Source: https://www.neradot.com/post/building-a-python-react-agent-class-a-step-by-step-guide
This detailed step-by-step guide on building a Python ReAct agent class highlights key methods such as `reason()` and `build_messages()` that are crucial for managing the agent’s internal state and conversation history. The `build_messages()` method constructs the prompt messages including the scratchpad, which contains prior reasoning and actions. A reliable strategy mentioned is to maintain the conversation history as a list of messages and selectively control how much of that history is included when building the prompt for the LLM call. This selective inclusion helps in retaining necessary context while avoiding token overflow. The author suggests that effective prompt building involves carefully balancing the retention of important context with the pruning or summarization of older or less relevant parts of the scratchpad. This approach ensures that the agent consistently has the context needed for reasoning without exceeding token limits imposed by the LLM.

### Source: https://api.python.langchain.com/en/latest/agents/langchain.agents.react.agent.create_react_agent.html
The LangChain official documentation for `create_react_agent` provides a foundation for implementing ReAct agents with built-in support for managing prompts and conversation history. The agent is created with a prompt template that integrates the tools and the agent’s past reasoning/actions (the scratchpad). LangChain internally handles the conversation history by encoding previous interactions into the prompt based on the provided prompt template and stop sequences. To maintain context retention and avoid token limits, LangChain’s prompt templates and stop sequence parameters allow developers to customize how much history is injected into the prompt. This flexibility enables truncation, selective history inclusion, or summarization strategies to keep the input within token bounds. Thus, using LangChain’s abstractions, a reliable strategy is to leverage prompt templates and output parsers to dynamically manage the scratchpad content so the agent always operates with relevant context without exceeding token limits.

### Source: https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/
This LangGraph how-to guide on creating a ReAct agent from scratch in Python outlines a structured approach to managing the agent’s state, specifically the conversation history stored as a list of messages. The example shows that the agent state is defined using a TypedDict containing a sequence of `BaseMessage` objects, with reducers (like `add_messages`) to update this state incrementally. This design pattern facilitates reliable scratchpad management by structurally adding new messages while maintaining prior context. To ensure the scratchpad does not exceed token limits, developers can implement reducers that prune or summarize older messages. Additionally, the guide demonstrates defining tools and integrating them into the model, implying that the agent’s prompt construction can be customized to include only relevant parts of the message history. This method provides a reliable Pythonic pattern to keep the conversation history manageable and contextually rich, balancing retention with token budget constraints.

-----
In summary, reliable strategies for managing the ReAct agent's scratchpad or conversation history in Python implementations to ensure context retention while avoiding token overflow include:

- Maintaining the scratchpad as an append-only list of reasoning and action steps or messages.
- Implementing truncation or pruning of older or less relevant history entries based on token budget.
- Using summarization techniques to condense older conversation parts preserving key context.
- Leveraging prompt templates and output parsers (such as in LangChain) to dynamically control how much history is included during prompt building.
- Structuring conversation history with typed message sequences and reducers to systematically manage additions and removals.
- Balancing the inclusion of essential context with token limits by selectively including messages relevant to the current reasoning step.

These approaches are supported by official implementations and detailed guides from repositories and frameworks such as LangChain and LangGraph, as well as by Python example projects and tutorials.

</details>

---

<details>
<summary>What are common parsing errors or workflow failures when using Google Gemini's function calling for ReAct agents, and what code-level mitigations (e.g., retry logic, output checks) are recommended?</summary>

### Source: https://ai.google.dev/gemini-api/docs/function-calling
The official Gemini API documentation outlines the basic workflow for function calling but does not explicitly list common parsing errors or workflow failures specific to ReAct agents. The documentation demonstrates how to configure the client, declare functions, and handle function call responses. The API returns a `functionCall` object in an OpenAPI-compatible schema, which ReAct agents or orchestrators are expected to interpret and dispatch to user-defined functions.

Although it does not enumerate specific error types, the code samples imply best practices that help mitigate common issues:
- Always check whether the function call was returned before accessing its contents (e.g., `if response.candidates.content.parts.function_call` in Python).
- Validate the structure of the returned `functionCall` object before parsing or invoking downstream handlers.
- Consider defensive programming: use try/except (Python) or try/catch (JavaScript) blocks around code that parses or executes the function call.
- Implement logging to capture unexpected response structures or missing fields, which are common integration pitfalls when model responses do not match the declared schema.

Retry logic is not explicitly covered, but the need to check the structure before use and to guard against missing or malformed function calls is demonstrated in the example code.

-----

### Source: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling
The Vertex AI documentation focuses on the mechanics of submitting prompts and function declarations, emphasizing the importance of schema compatibility. It does not specifically address ReAct agent workflows or parsing errors, but it highlights a key failure mode: mismatches between function declarations and returned function call objects.

Recommended mitigations inferred from the documentation include:
- Ensure that function declarations strictly follow the OpenAPI-compatible schema to minimize parsing errors.
- When processing model responses, validate that the returned function call objects conform to the expected schema and parameter types.
- If the model response does not contain the expected function call, handle this gracefully (e.g., default responses, error messages, or retries).
- Since the API supports REST and other interfaces, robust error handling for HTTP errors, malformed payloads, or authentication issues is necessary at the code level.

While retry logic is not explicitly detailed, the focus on schema compatibility and response validation suggests that defensive coding is critical for reliable integration with ReAct agents.

-----

### Source: https://ai.google.dev/gemini-api/docs/langgraph-example
This example describes how ReAct agents are constructed using Gemini 2.5 and LangGraph, combining LLM reasoning with action execution. Although the documentation does not list parsing errors directly, it does illustrate the iterative process of reasoning, tool usage, and observation handling.

Implied workflow failure points include:
- Incorrect parsing or interpretation of the model’s tool call outputs, which can lead to failure in selecting or invoking the correct function.
- Misalignment between declared tool schemas and the structure of model outputs, resulting in runtime errors or failed dispatches.
- Failure to handle cases where the model does not suggest a function call when one is expected.

Recommended code-level mitigations include:
- Implementing robust output validation to ensure the model’s function call matches the declared schema before invoking any downstream logic.
- Introducing retry or fallback logic if the model output is ambiguous or missing a required function call.
- Logging and exception handling to capture and diagnose workflow failures during the agent’s reasoning-action loop.

-----

### Source: https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb
The Colab notebook provides a step-by-step guide for building a ReAct agent with Gemini and function calling. It shows the orchestration of reasoning and acting, but does not explicitly document common parsing errors or workflow failures.

From the code and comments, best practices for mitigation are demonstrated:
- Explicitly inspect the model’s response to determine whether a function call is present and well-formed before attempting to execute it.
- Use assertions or conditional checks to verify the presence of required output fields.
- Implement exception handling to catch and log errors that occur during parsing or function execution.
- When an error or unexpected output is encountered, the agent can re-prompt the model or return a clarifying message.

The emphasis is on careful output validation and error handling to create a resilient agent workflow.

-----

### Source: https://www.philschmid.de/langgraph-gemini-2-5-react-agent
This blog post explains how to create a ReAct agent using Gemini 2.5 and LangGraph. Although not an official source, it highlights several practical issues and code-level mitigations:
- Parsing errors often occur if the model’s output does not conform to expected function call formats, especially when the function declaration schema is complex.
- Workflow failures can result from the agent failing to recognize when to invoke a function versus when to continue reasoning.
- To mitigate these issues, the post recommends:
  - Comprehensive output checks—validate the structure and data types of returned function calls.
  - Retry logic—re-prompt the model or retry the function call if parsing fails or the output is incomplete.
  - Clear logging of failures and unexpected outputs for debugging.
  - Defensive programming practices around all model output handling, including early exit or fallback behaviors when outputs are missing or malformed.

-----

</details>

---

<details>
<summary>What are concrete, step-by-step debugging and trace inspection techniques recommended specifically for Python-based ReAct agent development?</summary>

### Source: https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/
This guide provides a step-by-step implementation of a ReAct agent using LangGraph and highlights several debugging and trace inspection techniques that are especially relevant to Python-based ReAct agent development:

- **State Inspection:** When building a ReAct agent, the guide recommends structuring the agent’s state as a Python TypedDict, such as `AgentState`, which stores a list of messages. By using typed states and reducers (like `add_messages`), you can easily inspect and trace the evolution of the agent's internal state during execution. This structure allows for straightforward print-based or IDE-based inspection of the agent's state at different points in the code.
- **Modular Tool Functions:** The example demonstrates defining external tools as Python functions decorated with `@tool`. This modular approach means you can unit test and debug each tool function independently of the agent, either with traditional Python debugging tools or by adding print/log statements for trace inspection.
- **Incremental Development:** The guide's step-by-step approach (defining state, tools, and model separately) naturally supports incremental debugging—allowing you to validate and trace each piece in isolation before integrating them into the full agent loop.
- **Message Logging:** By saving or printing the `messages` list at each reasoning/action step, you can trace the agent's thought process and actions, making it easier to debug unexpected behaviors.

These techniques are Pythonic and integrate well with standard IDEs or notebooks, making step-by-step debugging and trace inspection accessible for ReAct agent development.

-----

### Source: https://python.langchain.com/docs/how_to/debugging/
LangChain’s official debugging documentation recommends several concrete, step-by-step techniques for debugging and tracing Python-based ReAct agents:

- **LangSmith Tracing:** The most robust way to debug LLM applications is to use LangSmith—a specialized platform for logging traces. After setting up your account and environment variables, LangSmith will automatically record detailed traces of all agent steps, including tool calls, intermediate reasoning, and final outputs.
- **Trace Inspection:** Once traces are logged to LangSmith, you can visually inspect every step in the agent’s reasoning chain, including inputs, outputs, and tool invocations. This enables line-by-line analysis of where errors or unexpected behavior may occur.
- **Error Highlighting:** LangSmith’s interface highlights errors or anomalies in the agent’s trace, making it easier to identify which step failed or produced a surprising output.
- **Incremental Logging:** By integrating LangSmith early in development, you can iteratively debug and improve your agent as you build, rather than waiting until a full workflow is complete.

These recommended practices are specifically tailored to Python-based LLM agents using the ReAct framework and offer both high-level and granular visibility into agent execution for efficient debugging.

-----

### Source: https://code.visualstudio.com/docs/python/debugging
Official Visual Studio Code documentation outlines several step-by-step debugging and trace inspection techniques for Python applications that are directly applicable to ReAct agent development:

- **Breakpoints:** Set breakpoints in your agent’s code to pause execution at critical points (e.g., after a reasoning step or tool invocation), allowing you to inspect variables such as the agent’s state, messages, and tool inputs/outputs.
- **Variable Inspection:** Use the variable explorer in the VS Code debugger to examine the current values of variables, including complex data structures like the agent’s state dictionary or message history.
- **Step Execution:** Single-step through the agent’s logic (step in, step over, step out) to follow the reasoning and action sequence in real time, observing how decisions are made at each step.
- **Console Output:** Use the integrated or external terminal to display print/log statements for real-time trace inspection of the agent’s execution flow.
- **Conditional Breakpoints:** Set conditions on breakpoints (e.g., when a certain message type appears) to halt execution only when specific situations arise, making it easier to debug complex, branching reasoning paths.

These techniques are standard for Python debugging but are especially powerful for tracing the iterative, stepwise execution of a ReAct agent’s reasoning and acting loop.

-----

</details>

---

<details>
<summary>How does Gemini's function calling and structured output API differ in practical use (parsing, error handling, extensibility) from OpenAI's GPT-4 and Anthropic Claude when building stepwise ReAct agent loops?</summary>

### Source: https://ai.google.dev/gemini-api/docs/structured-output
Gemini’s structured output feature allows developers to specify a response schema, ensuring that outputs consistently adhere to a predefined format such as JSON or enums. This reduces the need for complex post-processing and parsing, as the model’s output directly matches the required schema. For example, developers can define schemas for tasks like recipes or sentiment classification, and Gemini will return outputs that fit these specifications—making it suitable for downstream automation and reliable integration into agent loops. Gemini also supports structured responses for multimodal requests, allowing for schema-conformant outputs across text, image, video, and audio inputs.

-----

### Source: https://ai.google.dev/gemini-api/docs/function-calling
With Gemini’s function calling, developers define function declarations that the model can call, specifying input parameters and expected output structure. When the Gemini API receives a prompt, it returns a response that includes the function name and the structured parameters to use. This process enables precise extraction of actionable data, facilitating stepwise or multi-turn agent loops—critical for ReAct-style workflows. The model’s adherence to the declared schema ensures minimal ambiguity, easing parsing and error handling compared to less structured approaches.

-----

### Source: https://firebase.google.com/docs/ai-logic/generate-structured-output
By default, Gemini returns unstructured text, but the API allows you to enforce a specific schema using a responseSchema in the generateContent call. This ensures that outputs, such as JSON for downstream tasks or classification enums, always conform to a prescribed structure—drastically reducing post-processing and error-prone parsing. For example, recipe responses will always have required fields, and classification tasks can be strictly limited to set enum values, eliminating free-form or ambiguous outputs. This approach is highly extensible, as schemas can be adapted for various use cases, including multimodal input scenarios.

-----

### Source: https://isaac-the-man.dev/posts/get-structured-output-with-function-calling-from-google-gemini/
Gemini’s function calling allows developers to obtain structured output (like JSON) by defining a schema within the function declaration. Unlike traditional prompt engineering—which often leads to inconsistent or markdown-wrapped outputs—Gemini’s function calling produces responses that strictly follow the defined schema, with no extra wrappers or format deviations. This results in cleaner, more reliable parsing and straightforward error handling, as the output is guaranteed to match the schema. The extensibility is also notable: by simply modifying the function declaration, developers can adjust the output format without changing core agent logic.

-----

### Source: https://codelabs.developers.google.com/codelabs/gemini-function-calling
Gemini’s function calling workflow consists of several steps: (1) the developer defines one or more function declarations (schemas) in the API call, (2) the model returns a Function Call response containing the function name and structured parameters, (3) the application uses those parameters to make an external API request (handled outside Gemini), and (4) the API response is returned to Gemini for further processing or as a user response. This process ensures that structured data is consistently passed between the model and the agent code, making stepwise agent loops more robust and less error-prone compared to systems that rely solely on free-form text outputs. Parsing is streamlined, error handling is simplified by schema validation, and the system is easily extensible by adding or modifying function declarations.
-----

</details>

---

## Sources Scraped From Research Results

---
<details>
<summary>[2210.03629] ReAct: Synergizing Reasoning and Acting in Language Models</summary>

> Abstract:While large language models (LLMs) have demonstrated impressive capabilities across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g. chain-of-thought prompting) and acting (e.g. action plan generation) have primarily been studied as separate topics. In this paper, we explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with external sources, such as knowledge bases or environments, to gather additional information. We apply our approach, named ReAct, to a diverse set of language and decision making tasks and demonstrate its effectiveness over state-of-the-art baselines, as well as improved human interpretability and trustworthiness over methods without reasoning or acting components. Concretely, on question answering (HotpotQA) and fact verification (Fever), ReAct overcomes issues of hallucination and error propagation prevalent in chain-of-thought reasoning by interacting with a simple Wikipedia API, and generates human-like task-solving trajectories that are more interpretable than baselines without reasoning traces. On two interactive decision making benchmarks (ALFWorld and WebShop), ReAct outperforms imitation and reinforcement learning methods by an absolute success rate of 34% and 10% respectively, while being prompted with only one or two in-context examples. Project site with code: this https URL @https://react-lm.github.io/

### Original URL
https://arxiv.org/abs/2210.03629
</details>

---
<details>
<summary>Implementing ReAct Agentic Pattern From Scratch</summary>

Apr 13, 2025

# Implementing ReAct Agentic Pattern From Scratch

AI Agents Crash Course—Part 10 (with implementation).

- ![Avi Chawla @https://www.dailydoseofds.com/content/images/size/w100/2024/12/avi-google.jpg](https://www.dailydoseofds.com/author/avi/ "Avi Chawla")
- ![Akshay Pachaar @https://www.dailydoseofds.com/content/images/size/w100/2024/11/38653995.png](https://www.dailydoseofds.com/author/akshay/ "Akshay Pachaar")

Avi Chawla @https://www.dailydoseofds.com/author/avi/, Akshay Pachaar @https://www.dailydoseofds.com/author/akshay/

Table of Contents

1. Introduction @https://www.dailydoseofds.com/ai-agents-crash-course-part-10-with-implementation/#introduction
2. What is ReAct pattern? @https://www.dailydoseofds.com/ai-agents-crash-course-part-10-with-implementation/#what-is-react-pattern
1. Sample 1 @https://www.dailydoseofds.com/ai-agents-crash-course-part-10-with-implementation/#sample-1
2. Sample 2 @https://www.dailydoseofds.com/ai-agents-crash-course-part-10-with-implementation/#sample-2
3. But why does this matter? @https://www.dailydoseofds.com/ai-agents-crash-course-part-10-with-implementation/#but-why-does-this-matter
4. Reasoning + Acting: How ReAct Agents Work @https://www.dailydoseofds.com/ai-agents-crash-course-part-10-with-implementation/#reasoning-acting-how-react-agents-work
3. ReAct Implementation from Scratch @https://www.dailydoseofds.com/ai-agents-crash-course-part-10-with-implementation/#react-implementation-from-scratch

👉

Hey! This is a member-only post. But it looks like you are from **United States of America 🇺🇸**. Join today by visiting this **membership page @https://www.dailydoseofds.com/membership-7XQP25LK-HRVB64C2A/** for relief pricing of **20%** off on your subscription, FOREVER.

* * *

## Introduction

Building advanced AI agents has been greatly simplified by high-level frameworks like LangChain and CrewAI.

However, these frameworks often hide the underlying logic that makes an AI agent tick.

More specifically, under the hood, many such frameworks use the ReAct (Reasoning and Acting) pattern to let large language models (LLMs) think through problems and use tools to act on the world​.

!@https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd02b2eaa-16c3-4f92-8f97-06329fbcccd4_716x550.gif

For example, an agent in CrewAI typically alternates between reasoning about a task and acting (using a tool) to gather information or execute steps, following the ReAct paradigm​.

This approach enhances an LLM agent’s ability to handle complex tasks and decisions by combining chain-of-thought reasoning with external tool use​.

While frameworks like CrewAI implement this logic for you and you should also keep using them in production systems (also because they are built by experts), often times, relying on pre-packaged libraries without understanding their inner workings can lead to confusion​.

You might use an agent framework and wonder how it decides to, say, perform a web search or a calculation as part of answering a question.

An example is shown below where the Agent is going through a series of thought activities before producing a response.

!@https://www.dailydoseofds.com/content/images/2025/04/image-28.png

In this article, we’ll demystify that process by building a ReAct agent from scratch using only Python and an LLM.

!@https://www.dailydoseofds.com/content/images/2025/04/image-29.png

By doing so, we gain full control over the agent’s behavior, making it easier to optimize and troubleshoot​.

We’ll use OpenAI, but if you prefer to do it with Ollama locally, an open-source tool for running LLMs locally, with a model like Llama3 to power the agent, you can do that as well.

Along the way, we’ll explain the ReAct pattern, design an agent loop that interleaves reasoning and tool usage, and implement multiple tools that the agent can call.

The goal is to help you understand both the theory and implementation of ReAct agents.

By the end, you’ll have a working agent and a clear picture of how frameworks like CrewAI leverage ReAct internally.

Let's begin!

* * *

## What is ReAct pattern?

ReAct (short for Reasoning and Acting) is a paradigm for AI agent design where an agent uses chain-of-thought reasoning and tool-using actions in aggregation.

Instead of generating a direct answer in one step, a ReAct agent thinks step-by-step and can perform intermediate actions (like looking something up or calculating a value) before finalizing its answer.

!@https://www.dailydoseofds.com/content/images/2025/04/image-30.png

To get a clear perspective on this, Like think about how ReAct pattern works.

### Sample 1

Formally, an LLM following ReAct generates reasoning traces (the “Thoughts”) and task-specific actions (calls to tools) in an interleaved manner​. This means the model’s output might look something like:

- Thought: I should calculate the total.
- Action: `Calculator("123 + 456")`
- Observation: `579`
- Thought: Now I have the sum; next, I need to multiply it.
- Action: `Calculator("579 * 789")`
- Observation: `456,831`.
- Thought: I have the final result.
- Final Answer: `456,831`.

The reasoning traces (the chain of thoughts) help the model plan and keep track of what to do next, while the actions let it consult external sources or perform calculations to gather information it otherwise wouldn’t have​ inherent access to.

In effect, the model isn’t limited to its internal knowledge; it can reach out to tools, databases, or the internet as needed and then reason about the results.

This significantly enhances what the agent can do.

IBM describes ReAct agents as ones that use an LLM “brain” to coordinate reasoning and action, enabling interactions with the environment in a structured but adaptable way​.

Unlike a basic chatbot that answers with whatever is in its static knowledge, a ReAct agent can think, search, calculate, and then combine the results into an answer.

### Sample 2

Consider the output of an Agent I built in a multi-agent system (we'll get to the code shortly):

!@https://www.dailydoseofds.com/content/images/2025/04/Screenshot-2025-04-15-at-7.07.56-PM.png

In the example shown above, we see a live trace of an AI News Reporter Agent executing a task using the ReAct paradigm. The agent has been asked to create a news headline related to the "Agent2Agent Protocol". However, rather than jumping to conclusions, it reasons step by step, as seen in its structured trace.

Let’s break this down:

!@https://www.dailydoseofds.com/content/images/2025/04/Screenshot-2025-04-15-at-7.07.56-PM.png

- Agent Role: `News Reporter` – This agent specializes in curating and validating news content.
- Task: The agent has been instructed to generate a news headline and ensure it's relevant to the Agent2Agent Protocol.
- Thought: The agent first reasons internally that it should validate the information by performing a quick search for any recent updates about the protocol. This is the reasoning part of the ReAct cycle.
- Action: It proceeds to use a tool labeled `Search the internet`, passing in a structured JSON input with the query: `"Agent2Agent Protocol news November 2023"`. This is the acting step where the agent leverages external tools to gather real-world data.

!@https://www.dailydoseofds.com/content/images/2025/04/Screenshot-2025-04-15-at-7.13.26-PM.png

- Tool Output: It contains the results retrieved by the search tool—potentially news snippets, article summaries, or relevant URLs.

This illustrates the power of combining structured thought and external actions: rather than relying on the model's internal knowledge alone, the agent cross-checks facts using tools.

It's an example of how the ReAct pattern encourages transparency, accuracy, and verifiability in agent behavior—an essential feature for any system tasked with real-world information synthesis.

You can imagine scaling this further with multi-agent setups: a News Collector gathers raw feeds, a Fact Verifier checks reliability, and this News Reporter constructs the headline—all coordinating asynchronously using reasoning and tool-based actions.

Here's the implementation if you want to replicate the same output above.

💡

Make sure you have created a `.env` file with the `OPENAI_API_KEY` specified in it. It will make things much easier and faster for you. Also, add these two lines of code to handle asynchronous operations within a Jupyter Notebook environment, which will allow us to make asynchronous calls smoothly to your Crew Agent.

Agents crash course part 8 - snappify.com

notebook.ipynb

Copy

```
import nest_asyncio

nest_asyncio.apply()
```

💡

If you don't want to use OpenAI and stick to an open-source LLM, you can use Ollama as well.

Follow these steps:

\- Download Ollama from https://ollama.com @https://ollama.com/?ref=dailydoseofds.com.

\- Run the Ollama app.

\- Pull an open-source LLM.

\- Define the LLM as shown below.

\- Pass the LLM as the `llm` parameter to the Agent.

AI Agent crash course Part 10 - snappify.com

notebook.ipynb

Copy

```
from crewai import LLM

llm = LLM(
    model="ollama/llama3.2:1b",
    base_url="http://localhost:11434"
)
```

Command line

Copy

```
ollama pull llama3.2:1b
```

We begin by importing the essential classes from `crewai` and a useful tool: `SerperDevTool`. This tool wraps a real-time web search capability (via serper.dev @https://serper.dev/?ref=dailydoseofds.com) and allows our agents to fetch live information from the internet.

AI Agent crash course Part 10 - snappify.com

notebook.ipynb

Copy

```
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
```

Also, get a Serper Dev API key from serper.dev @https://serper.dev/?ref=dailydoseofds.com and store it in the `.env` file created earlier:

AI Agent crash course Part 10 - snappify.com

.env

Copy

```
SERPER_API_KEY="..."
OPENAI_API_KEY="sk-...."
```

Next, initialize the Web Search Tool, which the Agents can invoke when they need web results:

AI Agent crash course Part 10 - snappify.com

notebook.ipynb

Copy

```
news_search_tool = SerperDevTool()
```

Moving on, define the First Agent—The News Collector:

AI Agent crash course Part 10 - snappify.com

notebook.ipynb

Copy

```
latest_news_agent = Agent(
    role="News Collector",
    goal="Get the latest news on a given topic and gather insights",
    backstory="""You are a reporter who is responsible for
                 getting the latest news on a given topic""",
    verbose=True,
)

news_search_task = Task(
    description="Search for the latest news on {topic}",
    expected_output="A list of the latest news on a given topic",
    tool=news_search_tool,
    agent=latest_news_agent
)
```

- This agent is designed to behave like a digital journalist. Its responsibility is to gather news stories related to a given topic using the Serper tool. The `verbose=True` flag ensures we get detailed logging—this is what creates the transparent ReAct-style trace showcased earlier.
- Also, the task instructs the `News Collector` to actively search for the latest information on the specified `{topic}`. The `tool` parameter links the task to `SerperDevTool`, enabling it to actually perform the search rather than hallucinating.

Next, define the Second Agent—The News Reporter:

AI Agent crash course Part 10 - snappify.com

notebook.ipynb

Copy

```
news_reporter_agent = Agent(
    role="News Reporter",
    goal="""Use the latest news from the News Collector
            and create a news headline.""",
    backstory="""You are a reporter who is uses the latest news on
                 a given topic and creating a headline.""",
    verbose=True,
)

news_headline_task = Task(
    description="""Create a news headline on {topic} from the News
                   Collector. Also validate the news is relevant to
                   {topic} using the web search tool.""",
    expected_output="A news headline on a given topic",
    tools=[news_search_tool],
    agent=news_reporter_agent
)
```

- This agent is the headline writer. It consumes the output from the previous task and crafts a concise headline. Like the collector, it’s also verbose—meaning we’ll see its reasoning steps, tool calls, and decisions in the logs.
- This Agent's task is particularly interesting because it challenges the reporter agent to do two things:
  - Use the prior output (collected news articles).
  - Perform its own validation using the search tool again—double-checking that the news is relevant.
- That’s a ReAct pattern:
  - First, the agent reasons: “Do I have enough information? Is this valid?”
  - Then, it acts: makes a tool call to confirm.

Moving on, we connect our agents and tasks in a sequential crew pipeline:

AI Agent crash course Part 10 - snappify.com

notebook.ipynb

Copy

```
news_reporter_crew = Crew(
    agents=[latest_news_agent, news_reporter_agent],
    tasks=[news_search_task, news_headline_task],
    process=Process.sequential,
    verbose=True
)

```

- The Collector performs the initial search.
- The Reporter builds and validates the headline.

Finally, we start the workflow by passing a topic: Agent2Agent Protocol. The agents will dynamically process this input, use external tools, think through their next move, and generate a validated news headline.

AI Agent crash course Part 10 - snappify.com

notebook.ipynb

Copy

```
topic = "Agent2Agent Protocol"

response = news_reporter_crew.kickoff(inputs={"topic": topic})
```

This produces the following verbose output, which shows how your agent "thinks" in natural language, plans its next move, and uses external tools to carry out actions.

!@https://www.dailydoseofds.com/content/images/2025/04/image-31.png

But that’s not all. Look at one more thing towards the end of this verbose output:

!@https://www.dailydoseofds.com/content/images/2025/04/Screenshot-2025-04-15-at-7.34.59-PM.png

Under the hood, this ReAct-style behavior is governed by a very specific format template—what we call the action protocol. When using tools like `SerperDevTool`, CrewAI instructs the LLM to follow a rigid response schema. This helps ensure agents interact with tools in a safe, deterministic, and interpretable manner.

First, we have the Agent tool prompt format:

```python
"""
You ONLY have access to the following tools,
and should NEVER make up tools that are not listed here:

Tool Name: Search the internet
Tool Arguments: {'search_query': {'description':
                 'Mandatory search query you want to use
                 to search the internet', 'type': 'str'}}
Tool Description: A tool that can be used to
                  search the internet with a search_query.

"""
```

Copy

This is part of the tool context injected into the LLM prompt. It tells the agent:

- What tools are available.
- What arguments are required.
- That it must not invent tools or go off-protocol.

This creates strong constraints around agent behavior—which is important when you want to avoid hallucinations or misuse of capabilities.

The prompt also includes this critical instruction:

````python
"""
IMPORTANT: Use the following format in your response:

```
Thought: you should always think about what to do
Action: the action to take, only one name of
[Search the internet], just the name, as it's written.

Action Input: the input to the action, just a
simple JSON object, enclosed in curly braces,
using " to wrap keys and values.

Observation: the result of the action
```
"""
````

Copy

This is the _reasoning + acting_ loop spelled out in literal terms:

1. Thought: The agent expresses its internal reasoning.
2. Action: The agent picks the tool to use—verbatim.
3. Action Input: Arguments for the tool, formatted as strict JSON.
4. Observation: What the tool returned (i.e., the raw output).

This schema ensures that tools can be executed reliably, logged consistently, and traced clearly. And crucially—it’s LLM-friendly. Everything is written in natural language that the model understands and responds to.

Once all information is gathered, the agent is told to conclude like this:

````python
"""
Once all necessary information is gathered,
return the following format:

```
Thought: I now know the final answer
Final Answer: the final answer to
the original input question
```
"""
````

Copy

This signals the end of the reasoning chain. At this point, the agent has completed its research and can produce a confident, top-level answer.

That’s the essence of ReAct, which is implemented seamlessly using CrewAI.

### But why does this matter?

ReAct was introduced as a major step in making LLM-based agents more reliable and powerful​.

By having the model explain its thought process and check facts via tools, we reduce problems like hallucination and error propagation.

The original ReAct research by Yao et al. showed that this approach can overcome hallucination in question answering by letting the model retrieve real information (e.g. querying Wikipedia) to verify its facts​.

!@https://www.dailydoseofds.com/content/images/2025/04/Screenshot-2025-04-15-at-6.13.51-PM.png

It also improves the transparency of the agent’s decision-making, as we can inspect the chain of thought for debugging or trustworthiness​.

Overall, the ReAct pattern turns a passive LLM into an active problem solver that can break down complex tasks and interact with external data sources, much like an autonomous assistant.

This also explains why it is widely used in almost Agentic frameworks. The actual implementation could vary but everything does connect back to something that's derived from a ReAct pattern.

To reiterate, this format:

- Forces your LLM to operate step by step,
- Clearly separates thinking from acting,
- Guarantees deterministic input-output behavior for tools,
- And produces traceable reasoning chains you can inspect or debug.

### Reasoning + Acting: How ReAct Agents Work

A ReAct agent operates in a loop of Thought → Action → Observation, repeating until it reaches a solution or a final answer.

!@https://www.dailydoseofds.com/content/images/2025/04/image-33.png

This is analogous to how humans solve problems:

- we think about what to do
- perform an action (like looking something up or doing a calculation),
- observe the result
- and then incorporate that into our next thought.

The ReAct framework uses prompt engineering to enforce this structured approach, alternating the model’s thoughts and actions/observations​.

Here’s a step-by-step breakdown of the ReAct cycle in an AI agent:

- Thought: The Agent (powered by an LLM) analyzes the user’s query and internal context, and produces a reasoning step in natural language. This is typically not shown to the end user but is part of the agent’s self-talk. For example: “The question asks for the population of a country; I should use a web search to find the latest figure.”

!@https://www.dailydoseofds.com/content/images/2025/04/image-34.png

- Action: Based on the thought, the agent decides on an external tool or operation to perform. It outputs a prescribed format indicating the action. For instance: `Action: WebSearch("population of Canada 2023")`. The agent essentially “calls” a function (tool) by name, often with some input parameters.

!@https://www.dailydoseofds.com/content/images/2025/04/image-42.png

- Observation: The agent’s environment (our code) executes the requested action and returns the result (observation) back to the agent. For example, the web search tool might return: “ `Observation: The population of Canada in 2023 is 38 million.`” This observation is fed into the agent’s context.

!@https://www.dailydoseofds.com/content/images/2025/04/image-41.png

- The agent takes the new information and goes back to step 1 (another Thought). It will reason with the fresh data. In our example, it might think: “Now I have the population figure; I can answer the question.”

!@https://www.dailydoseofds.com/content/images/2025/04/image-40.png

1. This Thought/Action/Observation cycle repeats, allowing the agent to chain multiple tool uses if needed (search, then maybe a calculation, then another search, etc.). Eventually, the agent decides it can answer the user. At that point, instead of an Action, it outputs a Final Answer (sometimes marked as `Answer:` or `Final Answer:` in the format).

As we shall see shortly in the implementation from scratch, throughout this process, the agent maintains the conversation and its own intermediate steps.

Each Thought and Observation can be appended to the dialogue context so the LLM remembers what it has done so far.

!@https://www.dailydoseofds.com/content/images/2025/04/image-43.png

This is crucial for coherence. The end result is that the agent effectively plans its approach on the fly, mixing reasoning and acting.

This dynamic approach is much more adaptable than a rigid script or a single-turn response. It allows handling unforeseen sub-tasks, similar to how humans adjust plans when new information comes up​.

It’s important to note that all these “Thought” and “Action” annotations are not magical features of the LLM—they come from how we prompt the model.

As we shall see below, we explicitly instruct the model to format its responses in this structured way. In other words, ReAct is implemented via carefully crafted prompt templates and parsing logic, not via any built-in LLM ability.

The LLM is guided to behave like an agent that reasons and acts, through the examples and instructions we give it.

Now that we understand the ReAct pattern conceptually, we can start building our own agent that follows this logic. We’ll need a language model to serve as the agent’s brain, some tools the agent can use, and a loop that ties them together.

In the next section, we’ll step away from CrewAI and build a ReAct agent from scratch—in pure Python, using only local LLMs and a simple set of tool definitions. You'll see that everything we covered here is not magic—it’s just smart prompt design combined with controlled I/O.

💡

Make sure you have created a `.env` file with the `OPENAI_API_KEY` specified in it. It will make things much easier and faster for you. Also, add these two lines of code to handle asynchronous operations within a Jupyter Notebook environment, which will allow us to make asynchronous calls smoothly to your Crew Agent.

## ReAct Implementation from Scratch

Below, we shall implement a ReAct Agent in two ways:

- Manually executing each step for better clarity.
- Without manual intervention to fully automate the Reasoning and Action process.

You can download the code below:

## Read the full article

Sign up
now to read the full article and get access to all articles for
paying subscribers only.


Join today! @https://www.dailydoseofds.com/membership/

Already have an account? Sign in @https://www.dailydoseofds.com/signin/

## Read next

![The Full MCP Blueprint: Integrating Sampling into MCP Workflows @https://www.dailydoseofds.com/content/images/size/w100/2025/06/Sanyog-MCP_Course--14--3.png](https://www.dailydoseofds.com/model-context-protocol-crash-course-part-5/)

MCP @https://www.dailydoseofds.com/tag/mcp/ Jun 22, 2025•23 min read

## The Full MCP Blueprint: Integrating Sampling into MCP Workflows @https://www.dailydoseofds.com/model-context-protocol-crash-course-part-5/

Model context protocol crash course—Part 5.

- ![Avi Chawla @https://www.dailydoseofds.com/content/images/size/w100/2024/12/avi-google.jpg](https://www.dailydoseofds.com/author/avi/ "Avi Chawla")
- ![Akshay Pachaar @https://www.dailydoseofds.com/content/images/size/w100/2024/11/38653995.png](https://www.dailydoseofds.com/author/akshay/ "Akshay Pachaar")

Avi Chawla @https://www.dailydoseofds.com/author/avi/, Akshay Pachaar @https://www.dailydoseofds.com/author/akshay/

![The Full MCP Blueprint: Building a Full-Fledged MCP Workflow using Tools, Resources, and Prompts @https://www.dailydoseofds.com/content/images/size/w100/2025/06/Sanyog-MCP_Course--14--2.png](https://www.dailydoseofds.com/model-context-protocol-crash-course-part-4/)

MCP @https://www.dailydoseofds.com/tag/mcp/ Jun 15, 2025•23 min read

## The Full MCP Blueprint: Building a Full-Fledged MCP Workflow using Tools, Resources, and Prompts @https://www.dailydoseofds.com/model-context-protocol-crash-course-part-4/

Model context protocol crash course—Part 4.

- ![Avi Chawla @https://www.dailydoseofds.com/content/images/size/w100/2024/12/avi-google.jpg](https://www.dailydoseofds.com/author/avi/ "Avi Chawla")
- ![Akshay Pachaar @https://www.dailydoseofds.com/content/images/size/w100/2024/11/38653995.png](https://www.dailydoseofds.com/author/akshay/ "Akshay Pachaar")

Avi Chawla @https://www.dailydoseofds.com/author/avi/, Akshay Pachaar @https://www.dailydoseofds.com/author/akshay/

![The Full MCP Blueprint: Building a Custom MCP Client from Scratch @https://www.dailydoseofds.com/content/images/size/w100/2025/06/Sanyog-MCP_Course--14--1.png](https://www.dailydoseofds.com/model-context-protocol-crash-course-part-3/)

Agents @https://www.dailydoseofds.com/tag/agents/ Jun 8, 2025•20 min read

## The Full MCP Blueprint: Building a Custom MCP Client from Scratch @https://www.dailydoseofds.com/model-context-protocol-crash-course-part-3/

Model context protocol crash course—Part 3.

- ![Avi Chawla @https://www.dailydoseofds.com/content/images/size/w100/2024/12/avi-google.jpg](https://www.dailydoseofds.com/author/avi/ "Avi Chawla")
- ![Akshay Pachaar @https://www.dailydoseofds.com/content/images/size/w100/2024/11/38653995.png](https://www.dailydoseofds.com/author/akshay/ "Akshay Pachaar")

Avi Chawla @https://www.dailydoseofds.com/author/avi/, Akshay Pachaar @https://www.dailydoseofds.com/author/akshay/

## Join the Daily Dose of Data Science Today!

A daily column with insights, observations, tutorials, and best practices on data science.

Get Started! @https://www.dailydoseofds.com/membership/

!Join the Daily Dose of Data Science Today! @https://images.unsplash.com/photo-1545987796-200677ee1011?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3wxMTc3M3wwfDF8c2VhcmNofDV8fGRhdGF8ZW58MHx8fHwxNjg4ODgyMjkwfDA&ixlib=rb-4.0.3&q=80&w=100

Great! You’ve successfully signed up. Please check your email.

Welcome back! You've successfully signed in.

You've successfully subscribed to Daily Dose of Data Science.

Your link has expired.

Success! Check your email for magic link to sign-in.

Success! Your billing info has been updated.

Your billing was not updated.

StripeM-Inner

### Original URL
https://www.dailydoseofds.com/ai-agents-crash-course-part-10-with-implementation/
</details>

---
<details>
<summary>Building a Python React Agent Class: A Step-by-Step Guide - Nov 05, 2024  | NeraBlog by Neradot</summary>

November 5, 2024

# Building a Python React Agent Class: A Step-by-Step Guide

Roy Pasternak \| CTO at Neradot

!Building a Python React Agent Class: A Step-by-Step Guide @https://cdn.prod.website-files.com/645a411baa30ca12b39891d5/672a29a3b6cedc447bd49ba2_Building-React-Agent.png

In this blog post, we’ll walk through the creation of a React agent class in Python. After going over our previous post ( ReAct: Merging Reasoning and Action to Elevate AI Task Solving @https://www.neradot.com/post/react) you should already be familiar with the `React` flow. Here is a flow diagram to quickly recall how React handles the reason-action loop:

Now, let’s jump to the coding part. We’ll start with a minimal setup and gradually add functionality until we have a fully functional agent. Each step is highlighted with code snippets that focus on the differences from the previous step, providing a clear understanding of the incremental changes.

> **NOTE:** If you’d like to see the full code directly, you can check out the complete implementation on **react-simple** @https://github.com/neradot/react-simple

!@https://cdn.prod.website-files.com/645a411baa30ca12b39891d5/672a407dd7dd024a21617304_672a406a80c9aabdaddd767b_Screenshot%25202024-11-05%2520at%252017.56.50.png

React flow diagram of reasoning and action

##### Step 0: Define the class interface

Here is the `React` class structure we are going to create together:

```python
class React:
    def __init__(self, tools: list[callable], max_steps: int = 10, verbose: bool = False):
        # Initializes tools, settings, and internal state variables, such as `max_steps` and `verbose`.
        ...

    def reason(self) -> tuple[str, str, str, bool]:
        # Generates reasoning steps and determines the next action.
        ...

    def act(self, action: str, action_input: dict | str) -> str:
        # Executes actions by calling tool functions based on the current reasoning.
        ...

    def start(self, question: str):
        # Begins a new session with a given question.
        ...

    def next(self):
        # Advances the agent by processing reasoning, acting, and handling completion.
        ...

    def finish(self):
        # Ends the session, storing the final answer and completion status.
        ...

    def add_step(self, step: dict):
        # Logs each step of the agent’s process, such as thoughts, actions, and observations.
        ...

    def build_messages(self) -> list[dict]:
        # Constructs prompt messages for the reasoning process, integrating available tools and instructions.
        ...

```

​

- **`__init__`** `:` Initializes the agent with tools, step limit ( `max_steps`), and verbosity option ( `verbose`).
- **`reason`** `:` Produces reasoning steps, decides the next action, and determines if the agent should continue.
- **`act`** `:` Executes a specified action using a tool with provided inputs, returning the result (observation).
- **`start`**: Begins a session by setting up the agent with a question.
- **`next`** `:` Advances the agent by running one reasoning/action cycle and checking for completion.
- **`finish`** `:` Ends the session, saving the final answer and marking it complete.
- **`add_step`**: Logs each step of the process, including thoughts, actions, and observations.
- **`build_messages`**: Creates prompt messages that integrate tools and instructions for reasoning.

‍

##### Step 1: Minimal Setup

```python
import time

# Minimal setup with initialization and start method
class React:
    def __init__(self, verbose: bool = False):
        self.start_time: float | None = None
        self.is_started: bool = False
        self.intermediate_steps: list[dict] = []
        self.verbose = verbose

    def add_intermediate_step(self, intermediate_step: dict):
        self.intermediate_steps.append(intermediate_step)
        if self.verbose:
            print(intermediate_step)

    def start(self, question: str):
        if self.verbose:
          print(f"Starting agent with:")
        self.start_time = time.time()
        self.is_started = True
        self.intermediate_steps = []
        self.add_intermediate_step({"question": question})

# Testing the initial start function
react_agent = React(verbose=True)
react_agent.start("What is 2 + 2?")
```

‍

In this initial step, we set up the basic structure for the React class. This minimal version includes an initialization method, where we define the attributes to track the agent’s state, and a simple start method to begin an interaction. Here, we also introduce the intermediate\_steps list, which will store each phase of the agent’s process, such as questions, thoughts, actions, and observations. This list acts as a log for the interaction, making it possible for the agent to reference prior steps and maintain a coherent flow of actions and reasoning as we expand its functionality.

‍

###### **Output:**

```python
Starting agent with:
{'question': 'What is 2 + 2?'}
```

At this point, we’ve created a basic class that can start an interaction by saving the question as an intermediate step.

##### Step 2: Adding Reasoning with reason() and build\_messages() Methods

```python
import json
from utils import parse_json, get_completion
from prompts import (
    REACT_TOOLS_DESCRIPTION,
    REACT_VALID_ACTIONS,
    REACT_JSON_FORMAT,
    REACT_PROCESS_FORMAT,
    REACT_INTERMEDIATE_STEPS,
    REACT_ADDITIONAL_INSTRUCTIONS
)

class React:

	...

	# Adding the reasoning logic
	def reason(self) -> tuple[str, str, str, bool]:
	    messages = self.build_messages()
	    completion_response = get_completion(messages, model="gpt-4o", temperature=0, max_tokens=256, stop=["</reasoning>"])
	    completion = completion_response.choices[0].message.content
	    parsed_completion = parse_json(completion)
	    thought = parsed_completion["thought"]
	    action = parsed_completion["action"]
	    action_input = parsed_completion["action_input"]
	    is_final_answer = action == "Final Answer"
	    return thought, action, action_input, is_final_answer

	# Building the prompt messages from templates
	def build_messages(self) -> list[dict]:
      question = self.intermediate_steps[0]["question"]
      intermediate_steps=json.dumps(self.intermediate_steps[1:])
      system_prompt_message = \
              REACT_TOOLS_DESCRIPTION.format(tools_description="") + \
              REACT_VALID_ACTIONS.format(tools_names="") + \
              REACT_JSON_FORMAT + \
              REACT_PROCESS_FORMAT + \
              REACT_ADDITIONAL_INSTRUCTIONS + \
              REACT_INTERMEDIATE_STEPS.format(question=question, intermediate_steps=intermediate_steps)
        messages = [{ "role": "system", "content": system_prompt_message }]
        return messages


```

‍

##### Step 3: Driving Agent Progress with next() method

```python
class React:

	...

  @property
  def steps_count(self):
      return int(len(self.intermediate_steps) - 1 )

  def next(self):
      if not self.is_started:
          raise ValueError("React was not started")

      if self.verbose:
          print(f"Step {self.steps_count}")
      thought, action, action_input, is_final_answer = self.reason()
      self.add_intermediate_step({"thought": thought, "action": action, "action_input": action_input})

react_agent = React(verbose=True)
react_agent.start("What is 2 + 2?")
react_agent.next()
```

‍

With the reasoning structure in place, we now add the `next` method to control the agent's progression through steps. The `next` method initiates a reasoning process by calling `reason`, captures the generated `thought`, `action`, and `action_input`, and adds them as a new entry in `intermediate_steps`. we also introduce a `steps_count` property. This property calculates the number of reasoning of `intermediate_steps` (excluding the initial question).

###### **Output:**

```python
Starting agent with:
{'question': 'What is 2 + 2?'}
Step 0
{'thought': 'The question asks for the sum of 2 + 2, which is a basic arithmetic operation that I can solve without any tools.', 'action': 'Final Answer', 'action_input': 4}

```

‍

Here, the agent now has the capability to reason, generating an action and thought about the question. But, we are still missing the tool usage!

##### Step 4: Introducing Tools

```python
class React:
	...

# Adding tools functionality
	def __init__(self, tools: list[callable], verbose: bool = False):
			...
	    self.tools_dict: dict[str, callable] = {tool.__name__: tool for tool in tools}
	    self.tools_description: str = "\n".join([f"{tool_name}: {tool.__doc__}" for tool_name, tool in self.tools_dict.items()])
	    self.tools_names: list[str] = list(self.tools_dict.keys())
      if self.verbose:
          print("Initialized agent with tools:")
          print(f"{self.tools_description}")
          print()

def build_messages(self) -> list[dict]:
        ...
        system_prompt_message = \
                REACT_TOOLS_DESCRIPTION.format(tools_description=self.tools_description) + \
                REACT_VALID_ACTIONS.format(tools_names=self.tools_names) + \
        ...
def calculator(expression: str) -> float:
  """Evaluates a mathematical expression and returns the result"""
  return eval(expression)
react_agent = React(tools=[calculator], verbose=True)
react_agent.start("What is 2 + 2?")
react_agent.next()
react_agent.next()
```

In this step, we enhance the agent's abilities by adding a tool system. Tools are specific functions that the agent can call to perform certain actions, such as calculations or data retrieval. Each tool is passed to the `React` class upon initialization, allowing the agent to select and use them as needed.

Inside the `__init__` method, we create a dictionary, `tools_dict`, that maps each tool’s name to its function. We also generate `tools_description`, a formatted string that provides a description of each tool’s purpose (using the tool’s docstring), and `tools_names`, a list of tool names.

Additionally, we update the `build_messages` method to integrate `tools_description` and `tools_names` directly into the system prompt, allowing the agent to reference the tools available during its reasoning process.

##### **Expected Output:**

```python
Initialized agent with tools:
calculator: Evaluates a mathematical expression and returns the result

Starting agent with:
{'question': 'What is 2 + 2?'}
Step 0
{'thought': 'To solve the problem of finding what 2 + 2 equals, I should perform a simple arithmetic addition calculation.', 'action': 'calculator', 'action_input': '2 + 2'}
Step 1
{'thought': 'I should calculate the sum of 2 and 2 to find the answer.', 'action': 'calculator', 'action_input': '2 + 2'}

```

‍

The agent now has a calculator tool it can reference to answer questions involving calculations.

However, while the agent can select the tool, it doesn’t yet actually call the tool function to produce results. This is why the output shows no progress based on the action taken.

‍

##### Step 4: Taking Actions and Observations

```python
class React:
		...
    @property
    def steps_count(self):
+        return int((len(self.intermediate_steps) - 1 ) / 2)

# Adding action and observation handling
def act(self, action: str, action_input: dict | str) -> str:
    tool_func = self.tools_dict[action]
    if isinstance(action_input, dict):
        tool_result = tool_func(**action_input)
    else:
        tool_result = tool_func(action_input)
    return tool_result

# Updating `next` method to include observation step
def next(self):
    ...
+    thought, action, action_input, is_final_answer = self.reason()
+    self.add_intermediate_step({"thought": thought, "action": action, "action_input": action_input})
+    observation = self.act(action, action_input)
+    self.add_intermediate_step({"observation": observation})

react_agent = React(tools=[calculator], verbose=True)
react_agent.start("What is 2 + 2?")
react_agent.next()
react_agent.next()

```

‍

Now, we add the `act` method, allowing the agent to execute actions based on parsed inputs from the `reason` method. This `act` method calls a relevant tool function and records the observation for each action taken.

##### **Expected Output:**

```python
Initialized agent with tools:
calculator: Evaluates a mathematical expression and returns the result

Starting agent with:
{'question': 'What is 2 + 2?'}
Step 0
{'thought': 'To solve the problem, I need to add the numbers 2 and 2.', 'action': 'calculator', 'action_input': '2 + 2'}
{'observation': 4}
Step 1
{'thought': 'The calculation has been performed and the result is the addition of the two numbers.', 'action': 'Final Answer', 'action_input': 4}

---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
...
KeyError: 'Final Answer'

```

‍

The agent now has an `act` method to execute tool functions, and an observation step that records the results of each action. This allows the agent to see progress after calling a tool.

However, because the `act` method expects a tool name, the agent raises an error if the action is `"Final Answer"` rather than a tool. In the next step, we’ll handle this case to allow the agent to complete its reasoning when a final answer is reached.

#### ‍

##### Step 5: Adding a Finish Step

```python
class React:
	...
	def __init__(self, tools: list[callable], verbose: bool = False):
			...
			self.not_finished: bool = True
	# Adding the finish method
  def finish(self):
      self.not_finished = False
      self.add_intermediate_step({"answer": self.last["action_input"]})

	# Updating `next` method to finalize when the final answer is reached
  def next(self):
    ...
    thought, action, action_input, is_final_answer = self.reason()
    self.add_intermediate_step({"thought": thought, "action": action, "action_input": action_input})
    if is_final_answer:
        self.finish()
    else:
        observation = self.act(action, action_input)
        self.add_intermediate_step({"observation": observation})

react_agent = React(tools=[calculator], verbose=True)
react_agent.start("What is 2 + 2?")
while react_agent.not_finished:
    react_agent.next()
```

‍

In this step, we add the `finish` method to mark the agent's session as complete. We add  additional attribute: `not_finished`, which becomes `False` when the agent has completed its task.

Additionally, we update the `next` method to handle the `"Final Answer"` action gracefully. If `reason` returns `"Final Answer"`, `next` now calls `finish` rather than attempting to execute a tool, resolving the error from the previous step where the agent tried to run `"Final Answer"` as a tool name.

###### **Output:**

```python
Initialized agent with tools:
calculator: Evaluates a mathematical expression and returns the result

Starting agent with:
{'question': 'What is 2 + 2?'}
Step 0
{'thought': 'I can calculate the sum of 2 + 2 using a basic mathematical operation.', 'action': 'calculator', 'action_input': '2 + 2'}
{'observation': 4}
Step 1
{'thought': 'The sum of 2 + 2 is calculated using a simple arithmetic operation, and the result was confirmed as 4.', 'action': 'Final Answer', 'action_input': 4}
{'answer': 4}

```

‍

With the addition of the finish method and the updated next method, the agent now recognizes when it has reached a final answer and stops further actions, fully resolving the error from the previous step.

#### ‍

##### Final Touch: Adding Max Steps, Finish Reason, Duration, and Token Usage

```python
class React:
    def __init__(self, tools: list[callable], max_steps: int = 10, verbose: bool = False):
+       self.max_steps = max_steps
+       self.finish_reason: Literal["final answer", "max_steps_reached"] | None = None
+       self.completion_records: list[openai.types.completion.Completion] = []

    @property
+   def is_max_steps_reached(self):
+       return self.steps_count >= self.max_steps

    @property
+   def duration(self):
+       if not self.start_time or not self.end_time:
+           return None
+       return round((self.end_time - self.start_time)*1000, 3)

    @property
+   def token_usage(self):
+       usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
+       for res in self.completion_records:
+           usage['prompt_tokens'] += res.usage.prompt_tokens
+           usage['completion_tokens'] += res.usage.completion_tokens
+           usage['total_tokens'] += res.usage.total_tokens
+       return usage

    def next(self):
        ...
        else:
            observation = self.act(action, action_input)
            self.add_intermediate_step({"observation": observation})
+           if self.is_max_steps_reached:
+               self.finish()

    def reason(self) -> tuple[str, str, str, bool]:
        ...
+       self.completion_records.append(completion_response)
        ...

    def finish(self):
        self.not_finished = False
        if not self.is_max_steps_reached:
            self.finish_reason = "final answer"
            self.add_intermediate_step({"answer": self.last["action_input"]})
        else:
            self.finish_reason = "max_steps_reached"
        self.end_time = time.time()

react_agent = React(tools=[calculator], verbose=True)
react_agent.start("What is 2 + 2?")
while react_agent.not_finished:
    react_agent.next()
```

‍

In this final step, we add few features to enhance the agent's performance and monitoring:

1. **Max Steps**: The `max_steps` parameter limits the number of reasoning steps the agent can take. If this limit is reached, `finish` is called automatically to prevent looping.
2. **Finish Reason**: The `finish_reason` attribute records why the agent stopped: either `"final answer"` when a solution is reached or `"max_steps_reached"` if the step limit is hit.
3. **Duration**: The `duration` property calculates the total runtime from start to finish in milliseconds, giving insight into performance.
4. **Token Usage**: The `token_usage` property sums up tokens used in all completions, tracking prompt and completion tokens for resource monitoring.

With these additions, the agent is now fully functional, self-limiting, and capable of providing detailed usage insights.

###### **Output:**

```python
Initialized agent with tools:
calculator: Evaluates a mathematical expression and returns the result

Starting agent with:
{'question': 'What is 2 + 2?'}
Step 0
{'thought': 'To find the sum of 2 + 2, I will use the calculator tool to perform the arithmetic operation.', 'action': 'calculator', 'action_input': '2 + 2'}
{'observation': 4}
Step 1
{'thought': 'I have used the calculator tool to perform the addition of 2 + 2 and obtained the result.', 'action': 'Final Answer', 'action_input': 4}
{'answer': 4}
Finish reason: final answer
Duration: 2686.926 ms
Token Usage: {'prompt_tokens': 652, 'completion_tokens': 100, 'total_tokens': 752}

```

‍

###### Next

In the next post, we’ll dive into practical methods for evaluating agent performance, focusing on metrics that are particularly relevant for the `React` agent, such as correctness, efficiency, and response time. We’ll explore techniques for assessing the accuracy of the agent’s reasoning and action choices, testing for correctness in both intermediate steps and final answers.

‍

###### Full Prompt

```python
The available tools are:
{tools_description}

Use the tool name as the "action" value in the JSON response.

Valid actions are:
- "Final Answer"
- Any of {tools_names}

Respond in the following $JSON_BLOB format:
<reasoning>
{
        "thought": // Explain your reasoning for the chosen action, consider previous and subsequent steps
        "action": // The name of the tool to use
        "action_input": // The input required for the tool
}
</reasoning>

After you select an action, you will receive an observation. Then you can select another action or provide a final answer.
The pattern looks like this:

<reasoning>
$JSON_BLOB
</reasoning>
<observation>
{"observation": // action result}
</observation>
<reasoning>
$JSON_BLOB
</reasoning>
<observation>
{"observation": // action result}
</observation>

... (repeated until you have enough observations to answer the question)

<reasoning>
{
        "thought": // Explain why you have enough information to provide a final answer,
        "action": "Final Answer",
        "action_input": // Your final answer to the question
}
</reasoning>

Instructions:
1. Do not use comments in your JSON answer;
2. ALWAYS respond with a valid json blob of a single action;
3. ALWAYS think before choosing an action;
4. Respond in a JSON blob no matter what.

Here is the user question:
"{question}"

Here are the intermediate steps so far:
{intermediate_steps}
```

‍

‍

‍

‍

### Sign up for our newsletter today and never miss a Neradot update

Thank you! Your submission has been received!

Oops! Something went wrong while submitting the form.

### Original URL
https://www.neradot.com/post/building-a-python-react-agent-class-a-step-by-step-guide
</details>

---
<details>
<summary>Function calling with the Gemini API  |  Google AI for Developers</summary>

Skip to main content @https://ai.google.dev/gemini-api/docs/function-calling#main-content

![Google AI for Developers @https://www.gstatic.com/devrel-devsite/prod/va55008f56463f12ba1a0c4ec3fdc81dac4d4d331f95ef7b209d2570e7d9e879b/googledevai/images/lockup-new.svg](https://ai.google.dev/)

`/`

- English
- Deutsch
- Español – América Latina
- Français
- Indonesia
- Italiano
- Polski
- Português – Brasil
- Shqip
- Tiếng Việt
- Türkçe
- Русский
- עברית
- العربيّة
- فارسی
- हिंदी
- বাংলা
- ภาษาไทย
- 中文 – 简体
- 中文 – 繁體
- 日本語
- 한국어

Sign in

Introducing updates to our 2.5 family of thinking models. Learn more @https://ai.google.dev/gemini-api/docs/models

- Home @https://ai.google.dev/
- Gemini API @https://ai.google.dev/gemini-api
- Models @https://ai.google.dev/gemini-api/docs



 Send feedback



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

## How function calling works

!function calling overview @https://ai.google.dev/static/gemini-api/docs/images/function-calling-overview.png

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
in a single turn ( parallel function\\
calling @https://ai.google.dev/gemini-api/docs/function-calling#parallel_function_calling) and in
sequence ( compositional function\\
calling @https://ai.google.dev/gemini-api/docs/function-calling#compositional_function_calling).

### Step 1: Define a function declaration

Define a function and its declaration within your application code that allows
users to set light values and make an API request. This function could call
external services or APIs.

### Python

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

### JavaScript

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

### Python

```
from google.genai import types

# Configure the client and tools
client = genai.Client(api_key="GOOGLE_API_KEY")
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

### JavaScript

```
import { GoogleGenAI } from '@google/genai';

// Generation config with function declaration
const config = {
  tools: [{\
    functionDeclarations: [setLightValuesFunctionDeclaration]\
  }]
};

// Configure the client
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

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

### Python

```
id=None args={'color_temp': 'warm', 'brightness': 25} name='set_light_values'

```

### JavaScript

```
{
  name: 'set_light_values',
  args: { brightness: 25, color_temp: 'warm' }
}

```

### Step 3: Execute set\_light\_values function code

Extract the function call details from the model's response, parse the arguments
, and execute the `set_light_values` function.

### Python

```
# Extract tool call details, it may not be in the first part.
tool_call = response.candidates[0].content.parts[0].function_call

if tool_call.name == "set_light_values":
    result = set_light_values(**tool_call.args)
    print(f"Function execution result: {result}")

```

### JavaScript

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

### Python

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

### JavaScript

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
JSON, specifically with a select subset @https://ai.google.dev/api/caching#Schema
of the OpenAPI schema @https://spec.openapis.org/oas/v3.0.3#schemaw format. A
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

## Parallel function calling

In addition to single turn function calling, you can also call multiple
functions at once. Parallel function calling lets you execute multiple functions
at once and is used when the functions are not dependent on each other. This is
useful in scenarios like gathering data from multiple independent sources, such
as retrieving customer details from different databases or checking inventory
levels across various warehouses or performing multiple actions such as
converting your apartment into a disco.

### Python

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

### JavaScript

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
configuring function calling @https://ai.google.dev/gemini-api/docs/function-calling#function_calling_modes.

### Python

```
from google import genai
from google.genai import types

# Configure the client and tools
client = genai.Client(api_key="GOOGLE_API_KEY")
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

### JavaScript

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
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

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

The Python SDK supports automatic function calling @https://ai.google.dev/gemini-api/docs/function-calling#automatic_function_calling_python_only,
which automatically converts Python functions to declarations, handles the
function call execution and response cycle for you. Following is an example for
the disco use case.

### Python

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
client = genai.Client(api_key="GOOGLE_API_KEY")
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

### Python

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
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)  # Replace with your actual API key setup
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

### JavaScript

This example shows how to use JavaScript/TypeScript SDK to do comopositional
function calling using a manual execution loop.

```
import { GoogleGenAI, Type } from "@google/genai";

// Configure the client
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

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
    const toolResponse = toolFunctionsname @args;

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

Compositional function calling is a native Live\\
API @https://ai.google.dev/gemini-api/docs/live feature. This means Live API
can handle the function calling similar to the Python SDK.

### Python

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

### JavaScript

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


### Python

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

### JavaScript

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
The SDK automatically converts the Python function to declarations, handles the
function call execution and the response cycle for you. The Python SDK then
automatically:

1. Detects function call responses from the model.
2. Call the corresponding Python function in your code.
3. Sends the function response back to the model.
4. Returns the model's final text response.

To use this, define your function with type hints and a docstring, and then pass
the function itself (not a JSON declaration) as a tool:

### Python

```
from google import genai
from google.genai import types

# Define the function with type hints and docstring
def get_current_temperature(location: str) -> dict:
    """Gets the current temperature for a given location.

    Args:
        location: The city and state, e.g. San Francisco

    Returns:
        A dictionary containing the temperature and unit.
    """
    # ... (implementation) ...
    return {"temperature": 25, "unit": "Celsius"}

# Configure the client
client = genai.Client(api_key="GOOGLE_API_KEY")
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

### Python

```
config = types.GenerateContentConfig(
    tools=[get_current_temperature],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
)

```

### Automatic function schema declaration

Automatic schema extraction from Python functions doesn't work in all cases. For
example, it doesn't handle cases where you describe the fields of a nested
dictionary-object. The API is able to describe any of the following types:

### Python

```
AllowedType = (int | float | bool | str | list['AllowedType'] | dict[str, AllowedType])

```

To see what the inferred schema looks like, you can convert it using
`from_callable` @https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionDeclaration.from_callable:

### Python

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
Grounding with Google Search @https://ai.google.dev/gemini-api/docs/grounding and
code execution @https://ai.google.dev/gemini-api/docs/code-execution, in a request using the
Live API @https://ai.google.dev/gemini-api/docs/live.

### Python

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

### JavaScript

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

Python developers can try this out in the Live API Tool Use notebook @https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI_tools.ipynb.

## Model context protocol (MCP)

Model Context Protocol (MCP) @https://modelcontextprotocol.io/introduction is
an open standard for connecting AI applications with external tools and data.
MCP provides a common protocol for models to access context, such as functions
(tools), data sources (resources), or predefined prompts.

The Gemini SDKs have built-in support for the MCP, reducing boilerplate code and
offering
automatic tool calling @https://ai.google.dev/gemini-api/docs/function-calling#automatic_function_calling_python_only
for MCP tools. When the model generates an MCP tool call, the Python and
JavaScript client SDK can automatically execute the MCP tool and send the
response back to the model in a subsequent request, continuing this loop until
no more tool calls are made by the model.

Here, you can find an example of how to use a local MCP server with Gemini and
`mcp` SDK.

### Python

Make sure the latest version of the
`mcp` SDK @https://modelcontextprotocol.io/introduction is installed on
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

client = genai.Client(api_key="GOOGLE_API_KEY")

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

### JavaScript

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
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

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

Built-in MCP support is a experimental @https://ai.google.dev/gemini-api/docs/models#preview
feature in our SDKs and has the following limitations:

- Only tools are supported, not resources nor prompts
- It is available for the Python and JavaScript/TypeScript SDK.
- Breaking changes might occur in future releases.

Manual integration of MCP servers is always an option if these limit what you're
building.

## Supported models

This section lists models and their function calling capabilities. Experimental
models are not included. You can find a comprehensive capabilities overview on
the model overview @https://ai.google.dev/gemini-api/docs/models page.

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

- Only a subset of the OpenAPI\\
schema @https://ai.google.dev/api/caching#FunctionDeclaration is supported.
- Supported parameter types in Python are limited.
- Automatic function calling is a Python SDK feature only.



 Send feedback



Except as otherwise noted, the content of this page is licensed under the Creative Commons Attribution 4.0 License @https://creativecommons.org/licenses/by/4.0/, and code samples are licensed under the Apache 2.0 License @https://www.apache.org/licenses/LICENSE-2.0. For details, see the Google Developers Site Policies @https://developers.google.com/site-policies. Java is a registered trademark of Oracle and/or its affiliates.

Last updated 2025-06-24 UTC.

### Original URL
https://ai.google.dev/gemini-api/docs/function-calling
</details>

---
<details>
<summary>intro_diy_react_agent.ipynb - Colab</summary>

closeclose

info

This notebook is open with private outputs. Outputs will not be saved. You can disable this in Notebook settings @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#.


close

Commands
Code
Text
Copy to Drive
linksettingsexpand\_lessexpand\_more

format\_list\_bulleted

find\_in\_page

code

vpn\_key

folder

### Table of contents

tabclose

Introduction to ReAct Agents with Gemini & Function Calling @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Introduction_to_ReAct_Agents_with_Gemini_Function_Callingmore\_vert

Overview @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Overviewmore\_vert

Function / Tool-Calling APIs and Agent Frameworks @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Function_Tool_Calling_APIs_and_Agent_Frameworksmore\_vert

Objectives @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Objectivesmore\_vert

Background @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Backgroundmore\_vert

Costs @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Costsmore\_vert

Getting Started @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Getting_Startedmore\_vert

Install Vertex AI SDK for Python @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Install_Vertex_AI_SDK_for_Pythonmore\_vert

Restart current runtime @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Restart_current_runtimemore\_vert

Authenticate your notebook environment (Colab only) @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Authenticate_your_notebook_environment_Colab_only_more\_vert

Set Google Cloud project information and initialize Vertex AI SDK @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Set_Google_Cloud_project_information_and_initialize_Vertex_AI_SDKmore\_vert

Imports Libraries @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Imports_Librariesmore\_vert

Prepare a model with system instructions @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Prepare_a_model_with_system_instructionsmore\_vert

Helper Functions @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Helper_Functionsmore\_vert

Action definitions @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Action_definitionsmore\_vert

Explicit goals checking @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Explicit_goals_checkingmore\_vert

Prompt Templates @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Prompt_Templatesmore\_vert

Example 1: Multiple single-turn generate\_content calls with full prompts @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Example_1_Multiple_single_turn_generate_content_calls_with_full_promptsmore\_vert

The Main ReAct Loop @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=The_Main_ReAct_Loopmore\_vert

The Model decides when the goal is reached @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=The_Model_decides_when_the_goal_is_reachedmore\_vert

Example 2: Incremental Messaging Using the Chat API @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Example_2_Incremental_Messaging_Using_the_Chat_APImore\_vert

The Chat session loop @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=The_Chat_session_loopmore\_vert

Display the full chat history @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Display_the_full_chat_historymore\_vert

Example 3: Leveraging Gemini Function Calling Support @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Example_3_Leveraging_Gemini_Function_Calling_Supportmore\_vert

Tool Declarations @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Tool_Declarationsmore\_vert

Model with tool declarations @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=Model_with_tool_declarationsmore\_vert

The function calling model response @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=The_function_calling_model_responsemore\_vert

The Main ReAct Loop @https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb#scrollTo=The_Main_ReAct_Loopmore\_vert

addSection


Notebook

more\_horiz

* * *

sparkGemini


# Copyright 2024 Google LLC

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

#     https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

* * *

sparkGemini


keyboard\_arrow\_down

# Introduction to ReAct Agents with Gemini & Function Calling

|     |     |     |     |
| --- | --- | --- | --- |
| ![Google Colaboratory logo @https://cloud.google.com/ml-engine/images/colab-logo-32px.png\<br>\<br> Run in Colab](https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb) | ![Google Cloud Colab Enterprise logo @https://lh3.googleusercontent.com/JmcxdQi-qOpctIvWKgPtrzZdJJK-J3sWE1RsfjZNwshCFgE_9fULcNpuXYTilIR2hjwN\<br>\<br> Run in Colab Enterprise](https://www.google.com/url?q=https%3A%2F%2Fconsole.cloud.google.com%2Fvertex-ai%2Fcolab%2Fimport%2Fhttps%3A%252F%252Fraw.githubusercontent.com%252FGoogleCloudPlatform%252Fgenerative-ai%252Fmain%252Fgemini%252Ffunction-calling%252Fintro_diy_react_agent.ipynb) | ![GitHub logo @https://cloud.google.com/ml-engine/images/github-logo-32px.png\<br>\<br> View on GitHub](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb) | ![Vertex AI logo @https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32\<br>\<br> \<br> Open in Vertex AI Workbench](https://www.google.com/url?q=https%3A%2F%2Fconsole.cloud.google.com%2Fvertex-ai%2Fworkbench%2Fdeploy-notebook%3Fdownload_url%3Dhttps%3A%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Ffunction-calling%2Fintro_diy_react_agent.ipynb) |

**Share to:**
![LinkedIn logo @https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg](https://www.google.com/url?q=https%3A%2F%2Fwww.linkedin.com%2Fsharing%2Fshare-offsite%2F%3Furl%3Dhttps%253A%2F%2Fgithub.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fblob%2Fmain%2Fgemini%2Ffunction-calling%2Fintro_diy_react_agent.ipynb) ![Bluesky logo @https://upload.wikimedia.org/wikipedia/commons/7/7a/Bluesky_Logo.svg](https://www.google.com/url?q=https%3A%2F%2Fbsky.app%2Fintent%2Fcompose%3Ftext%3Dhttps%253A%2F%2Fgithub.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fblob%2Fmain%2Fgemini%2Ffunction-calling%2Fintro_diy_react_agent.ipynb) ![X logo @https://upload.wikimedia.org/wikipedia/commons/5/5a/X_icon_2.svg](https://www.google.com/url?q=https%3A%2F%2Ftwitter.com%2Fintent%2Ftweet%3Furl%3Dhttps%253A%2F%2Fgithub.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fblob%2Fmain%2Fgemini%2Ffunction-calling%2Fintro_diy_react_agent.ipynb) ![Reddit logo @https://redditinc.com/hubfs/Reddit%20Inc/Brand/Reddit_Logo.png](https://www.google.com/url?q=https%3A%2F%2Freddit.com%2Fsubmit%3Furl%3Dhttps%253A%2F%2Fgithub.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fblob%2Fmain%2Fgemini%2Ffunction-calling%2Fintro_diy_react_agent.ipynb) ![Facebook logo @https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg](https://www.google.com/url?q=https%3A%2F%2Fwww.facebook.com%2Fsharer%2Fsharer.php%3Fu%3Dhttps%253A%2F%2Fgithub.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fblob%2Fmain%2Fgemini%2Ffunction-calling%2Fintro_diy_react_agent.ipynb)

subdirectory\_arrow\_right53 cells hidden

* * *

sparkGemini


|  |  |
| --- | --- |
| Author(s) | Gary Ng @https://github.com/gkcng |

subdirectory\_arrow\_right0 cells hidden

* * *

sparkGemini


keyboard\_arrow\_down

## Overview

This notebook illustrates that at its simplest, a ReAct agent is a piece of code that coordinates between reasoning and acting, where:

- The reasoning is carried out by the language model
- The application code performs the acting, at the instruction of the language model.

This allows problems to be solved by letting a model 'think' through the tasks step-by-step, taking actions and getting action feedback before determining the next steps.

|     |
| --- |
| !The Reasoning and Acting Cycle @https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiuuYg9Pduep9GkUfjloNVOiy3qjpPbT017GKlgGEGMaLNu_TCheEeJ7r8Qok6-0BK3KMfLvsN2vSgFQ8xOvnHM9CAb4Ix4I62bcN2oXFWfqAJzGAGbVqbeCyVktu3h9Dyf5ameRe54LEr32Emp0nG52iofpNOTXCxMY12K7fvmDZNPPmfJaT5zo1OBQA/s595/Screen%20Shot%202022-11-08%20at%208.53.49%20AM.png |
| _From the paper: ReAct: Synergizing Reasoning and Acting in Language Models @https://www.google.com/url?q=https%3A%2F%2Fresearch.google%2Fblog%2Freact-synergizing-reasoning-and-acting-in-language-models%2F_ |

This coordination between the language model and the environment is made possible by asking the language model to communicate the intended actions in a specific and structured manner. The response is 'specific' in that the list of possible actions are predefined functions and thus necessarily constrained. The response is also 'structured', so the function parameters given in the response can be used directly by the application code, minimizing the need for further parsing, interpretation, or transformations.

Both requirements can be supported by many language models, as they are equivalent to performing natural language tasks such as classification and information extraction. As illustrated in the first two examples in this notebook, the task of identifying suitable function names and extraction of function parameters can be done using prompting and response parsing alone.

For strengthened quality on the function call responses however, in terms of validity, reliability, and consistency, many models now feature built-in APIs supporting 'Function Calling' or 'Tools Calling' (these terms are often used interchangeably). Such built-in support reduces the amount of defensive safeguards a developer has to build around response handling in their applications.

subdirectory\_arrow\_right4 cells hidden

* * *

sparkGemini


keyboard\_arrow\_down

### Function / Tool-Calling APIs and Agent Frameworks

In the third example in this notebook, we leverage Function Calling in Gemini @https://www.google.com/url?q=https%3A%2F%2Fcloud.google.com%2Fvertex-ai%2Fgenerative-ai%2Fdocs%2Fmultimodal%2Ffunction-calling to build our simple agent. It lets developers create a description of a function in their code, then pass that description to a language model in a request. The response from the model includes the name of a function that matches the description and the arguments to call it with.

There are also other tools-calling and agents building frameworks to increase developers productivity. For example, the Tool-Calling Agents @https://www.google.com/url?q=https%3A%2F%2Fpython.langchain.com%2Fv0.1%2Fdocs%2Fmodules%2Fagents%2Fagent_types%2Ftool_calling%2F from LangChain, and at an even higher level of abstraction, Reasoning Engine @https://www.google.com/url?q=https%3A%2F%2Fcloud.google.com%2Fvertex-ai%2Fgenerative-ai%2Fdocs%2Freasoning-engine%2Foverview is a Google Cloud managed service that helps you to build and deploy an agent reasoning framework ( See sample notebooks @https://github.com/GoogleCloudPlatform/generative-ai/tree/main/gemini/reasoning-engine). Reasoning Engine integrates closely with the Python SDK for the Gemini model in Vertex AI, and it can manage prompts, agents, and examples in a modular way. Reasoning Engine is compatible with LangChain, LlamaIndex, or other Python frameworks.

subdirectory\_arrow\_right0 cells hidden

* * *

sparkGemini


keyboard\_arrow\_down

### Objectives

To illustrate the basic building blocks of function calling and its utility, this notebook illustrates building the same agent with Gemini in three different ways, via:

1. Prompting alone - using the single turn `generate_content` API.
2. Prompting alone - using the `ChatSession` API instead.
3. Function Calling - Modified from the `ChatSession` example.

In the first example, the list of possible functions are presented to the API every time because the API is stateless. In the second example, because the `ChatSession` is stateful on the client side, we only need to present the list of function choices at the beginning of the session. The first two examples will introduce to the audience the building blocks that are now reliably supported by Gemini and many other model APIs as 'Tool' / 'Function' calling, and the Gemini API is demonstrated in the third example.

The raw prompting examples are only used to explain the building blocks and help understand the dedicated APIs. For your productivity and reliability of responses you are encouraged to use an API that supports function calling.

In the first example, we also illustrate the concept of explicit goal checking vs model-based goal checking. Use explicit goal checking when the goal can easily be define in code, it can save some cost and improves speed. Otherwise use model-based goal checking when the goal is too complex or variable, and specifying the goal in natural language and let the model handle the interpretation is simpler and faster than writing the full checks in code.

subdirectory\_arrow\_right0 cells hidden

* * *

sparkGemini


keyboard\_arrow\_down

### Background

This example was suggested by Gemini Advanced as a simple, text-based demo that highlights the core ReAct concept: Autonomy, Cyclic, Reasoning. The agent's thoughts demonstrate a simple form of reasoning, connecting observations to actions.

|     |
| --- |
| !Gemini's suggestion @https://services.google.com/fh/files/misc/gemini_react_suggestion.jpg |
| _Scenario: A ReAct agent designed to tidy up a virtual room._ |

subdirectory\_arrow\_right0 cells hidden

* * *

sparkGemini


keyboard\_arrow\_down

### Costs

This tutorial uses billable components of Google Cloud:

- Google Foundational Models on Vertex AI ( Function Calling @https://www.google.com/url?q=https%3A%2F%2Fcloud.google.com%2Fvertex-ai%2Fgenerative-ai%2Fdocs%2Fmultimodal%2Ffunction-calling%23pricing)

Learn about Generative AI on Vertex AI Pricing @https://www.google.com/url?q=https%3A%2F%2Fcloud.google.com%2Fvertex-ai%2Fgenerative-ai%2Fpricing and use the Pricing Calculator @https://www.google.com/url?q=https%3A%2F%2Fcloud.google.com%2Fproducts%2Fcalculator%2F to generate a cost estimate based on your projected usage.

subdirectory\_arrow\_right0 cells hidden

* * *

sparkGemini


keyboard\_arrow\_down

## Getting Started

subdirectory\_arrow\_right13 cells hidden

* * *

sparkGemini


keyboard\_arrow\_down

### Install Vertex AI SDK for Python

This notebook uses the Vertex AI SDK for Python @https://www.google.com/url?q=https%3A%2F%2Fcloud.google.com%2Fvertex-ai%2Fgenerative-ai%2Fdocs%2Freference%2Fpython%2Flatest.

subdirectory\_arrow\_right1 cell hidden

* * *

sparkGemini


%pip install --upgrade --user google-cloud-aiplatform

* * *

sparkGemini


keyboard\_arrow\_down

### Restart current runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which will restart the current kernel.

subdirectory\_arrow\_right2 cells hidden

* * *

sparkGemini


# Restart kernel after installs so that your environment can access the new packages

import IPython

app = IPython.Application.instance()

app.kernel.do\_shutdown(True)

* * *

sparkGemini


**⚠️ The kernel is going to restart. Please wait until it is finished before continuing to the next step. ⚠️**

subdirectory\_arrow\_right0 cells hidden

* * *

sparkGemini


keyboard\_arrow\_down

### Authenticate your notebook environment (Colab only)

If you are running this notebook on Google Colab, run the following cell to authenticate your environment. This step is not required if you are using Vertex AI Workbench @https://www.google.com/url?q=https%3A%2F%2Fcloud.google.com%2Fvertex-ai-workbench.

subdirectory\_arrow\_right1 cell hidden

* * *

sparkGemini


import sys

# Additional authentication is required for Google Colab

if"google.colab"in sys.modules:

# Authenticate user to Google Cloud

from google.colab import auth

    auth.authenticate\_user()

* * *

sparkGemini


keyboard\_arrow\_down

### Set Google Cloud project information and initialize Vertex AI SDK

To get started using Vertex AI, you must have an existing Google Cloud project and enable the Vertex AI API @https://www.google.com/url?q=https%3A%2F%2Fconsole.cloud.google.com%2Fflows%2Fenableapi%3Fapiid%3Daiplatform.googleapis.com.

Learn more about setting up a project and a development environment @https://www.google.com/url?q=https%3A%2F%2Fcloud.google.com%2Fvertex-ai%2Fdocs%2Fstart%2Fcloud-environment.

subdirectory\_arrow\_right1 cell hidden

* * *

sparkGemini


PROJECT\_ID = "\[your-project-id\]"# @param {type:"string"}

LOCATION = "us-central1"# @param {type:"string"}

import vertexai

vertexai.init(project=PROJECT\_ID, location=LOCATION)

PROJECT\_ID:

"

"

edit

LOCATION:

"

"

edit

Show code

* * *

sparkGemini


keyboard\_arrow\_down

### Imports Libraries

subdirectory\_arrow\_right1 cell hidden

* * *

sparkGemini


from collections.abc import Callable

import json

import sys

import traceback

from google.protobuf.json\_format import MessageToJson

from vertexai import generative\_models

from vertexai.generative\_models import FunctionDeclaration, GenerativeModel, Part, Tool

* * *

sparkGemini


keyboard\_arrow\_down

### Prepare a model with system instructions

subdirectory\_arrow\_right1 cell hidden

* * *

sparkGemini


model = GenerativeModel(

"gemini-2.0-flash",

    system\_instruction=\[\
\
"You are an assistant that helps me tidy my room."\
\
"Your goal is to make sure all the books are on the shelf, all clothes are in the hamper, and the trash is empty.",\
\
"You cannot receive any input from me.",\
\
\],

    generation\_config={"temperature": 0.0},

    safety\_settings=\[\
\
        generative\_models.SafetySetting(\
\
            category=generative\_models.HarmCategory.HARM\_CATEGORY\_DANGEROUS\_CONTENT,\
\
            method=generative\_models.SafetySetting.HarmBlockMethod.PROBABILITY,\
\
            threshold=generative\_models.HarmBlockThreshold.BLOCK\_ONLY\_HIGH,\
\
),\
\
\],

)

* * *

sparkGemini


keyboard\_arrow\_down

## Helper Functions

subdirectory\_arrow\_right8 cells hidden

* * *

sparkGemini


verbose = True

* * *

sparkGemini


# Conveience function to print multiline text indented

defindent(text, amount, ch=" "):

    padding = amount \* ch

return"".join(padding + line for line in text.splitlines(True))

# Convenience function for logging statements

deflogging(msg):

global verbose

print(msg)if verbose elseNone

# Retrieve the text from a model response

defget\_text(resp):

return resp.candidates\[0\].content.parts\[0\].text

# Retrieve the function call information from a model response

defget\_function\_call(resp):

return resp.candidates\[0\].function\_calls\[0\]

defget\_action\_label(json\_payload, log, role="MODEL"):

    log(f"{role}: {json\_payload}")

    answer = json.loads(json\_payload)

    action = answer\["next\_action"\]

return action

defget\_action\_from\_function\_call(func\_payload, log, role="MODEL"):

    json\_payload = MessageToJson(func\_payload.\_pb)

    log(f"{role}: {json\_payload}")

return func\_payload.name

* * *

sparkGemini


keyboard\_arrow\_down

### Action definitions

These are the pseudo actions declared as simple Python functions. With the Function Calling pattern, the orchestration layer of an agent will be calling these Tools to carry out actions.

subdirectory\_arrow\_right1 cell hidden

* * *

sparkGemini


# Initial room state

defreset\_room\_state(room\_state):

    room\_state.clear()

    room\_state\["clothes"\] = "floor"

    room\_state\["books"\] = "scattered"

    room\_state\["wastebin"\] = "empty"

# Functions for actions (replace these with Gemini function calls)

defpick\_up\_clothes(room\_state):

    room\_state\["clothes"\] = "carrying by hand"

return room\_state, "The clothes are now being carried."

defput\_clothes\_in\_hamper(room\_state):

    room\_state\["clothes"\] = "hamper"

return room\_state, "The clothes are now in the hamper."

defpick\_up\_books(room\_state):

    room\_state\["books"\] = "in hand"

return room\_state, "The books are now in my hand."

defplace\_books\_on\_shelf(room\_state):

    room\_state\["books"\] = "shelf"

return room\_state, "The books are now on the shelf."

defempty\_wastebin(room\_state):

    room\_state\["wastebin"\] = "empty"

return room\_state, "The wastebin is emptied."

# Maps a function string to its respective function reference.

defget\_func(action\_label):

returnNoneif action\_label == ""elsegetattr(sys.modules\[\_\_name\_\_\], action\_label)

* * *

sparkGemini


keyboard\_arrow\_down

### Explicit goals checking

This is only used in the first example to illustrate the concept: The goal checking responsibility can be either in code or be delegated to the model, depending on factors such as the complexity of the goal, ease of defining in code for example.

subdirectory\_arrow\_right1 cell hidden

* * *

sparkGemini


# Function to check if the room is tidy

# Some examples below do not call this function,

# for those examples the model takes on the goal validation role.

defis\_room\_tidy(room\_state):

returnall(

\[\
\
            room\_state\["clothes"\] == "hamper",\
\
            room\_state\["books"\] == "shelf",\
\
            room\_state\["wastebin"\] == "empty",\
\
\]

)

* * *

sparkGemini


keyboard\_arrow\_down

### Prompt Templates

subdirectory\_arrow\_right1 cell hidden

* * *

sparkGemini


functions = """

<actions>

    put\_clothes\_in\_hamper - place clothes into hamper, instead of carrying them around in your hand.

    pick\_up\_clothes - pick clothes up from the floor.

    pick\_up\_books - pick books up from anywhere not on the shelf

    place\_books\_on\_shelf - self explanatory.

    empty\_wastebin - self explanatory.

    done - when everything are in the right place.

</actions>"""

defget\_next\_step\_full\_prompt(state, cycle, log):

    observation = f"The room is currently in this state: {state}."

    prompt = "\\n".join(

\[\
\
            observation,\
\
f"You can pick any of the following action labels: {functions}",\
\
"Which one should be the next step to achieve the goal? ",\
\
'Return a single JSON object containing fields "next\_action" and "rationale".',\
\
\]

)

(

        log("PROMPT:\\n{}".format(indent(prompt, 1, "\\t")))

if cycle == 1

else log(f"OBSERVATION: {observation}")

)

return prompt

* * *

sparkGemini


keyboard\_arrow\_down

## Example 1: Multiple single-turn `generate_content` calls with full prompts

subdirectory\_arrow\_right6 cells hidden

* * *

sparkGemini


An example turn.

```
You are an assistant that helps me tidy my room.
Your goal is to make sure all the books are on the shelf, all clothes are in the hamper, and the trash is empty.
You cannot receive any input from me.

The room is currently in this state: {'clothes': 'floor', 'books': 'scattered', 'wastebin': 'empty'}.

You can pick any of the following action labels:
<actions>
    put_clothes_in_hamper - place clothes into hamper, instead of carrying them around in your hand.
    pick_up_clothes - pick clothes up from the floor.
    pick_up_books - pick books up from anywhere not on the shelf
    place_books_on_shelf - self explanatory.
    empty_wastebin - self explanatory.
    done - when everything are in the right place.
</actions>
Which one should be the next step to achieve the goal?
Return a single JSON object containing fields "next_action" and "rationale".

RAW MODEL RESPONSE:

candidates {
  content {
    role: "model"
    parts {
      text: "{\"next_action\": \"pick_up_clothes\", \"rationale\": \"The clothes are on the floor and need to be picked up before they can be put in the hamper.\"}\n"
    }
  }
  finish_reason: STOP,
  ...
}
```

subdirectory\_arrow\_right0 cells hidden

* * *

sparkGemini


keyboard\_arrow\_down

### The Main ReAct Loop

Interleaving asking for next steps and executing the steps.

Notice that at cycle 4 the environment has changed to have a non-empty wastebin.
With the goal that includes trash being empty, the model is recognizing the change and behaves accordingly, without the need to restate anything.

This is also well within expectation as this loop prompts the model with all the information every time.

subdirectory\_arrow\_right2 cells hidden

* * *

sparkGemini


# Main ReAct loop

defmain\_react\_loop(loop\_continues, log):

    room\_state = {}

    reset\_room\_state(room\_state)

    trash\_added = False

    cycle = 1

while loop\_continues(cycle, room\_state):

        log(f"Cycle #{cycle}")

# Observe the environment (use Gemini to generate an action thought)

try:  # REASON #

            response = model.generate\_content(

                get\_next\_step\_full\_prompt(room\_state, cycle, log),

                generation\_config={"response\_mime\_type": "application/json"},

)# JSON Mode

            action\_label = get\_action\_label(get\_text(response).strip(), log)

except Exception:

            traceback.print\_exc()

            log(response)

break

# Execute the action and get the observation

if action\_label == "done":

break

try:  # ACTION #

# Call the function mapped from the label

            room\_state, acknowledgement = get\_func(action\_label)(room\_state)

            log(f"ACTION:   {action\_label}\\nEXECUTED: {acknowledgement}\\n")

except Exception:

            log("No action suggested.")

# Simulating a change in environment

if cycle == 4andnot trash\_added:

            room\_state\["wastebin"\] = "1 item"

            trash\_added = True

        cycle += 1

# End of while loop

# Determine the final result

    result = (

"The room is tidy!"if is\_room\_tidy(room\_state)else"The room is not tidy!"

)

return room\_state, result

* * *

sparkGemini


# We are passing in a while loop continuation test function:

# Continue while loop when number of cycles <= 10 AND the room is not yet tidy.

# We are explicitly testing if the room is tidy within code.

#

# To save space, only the first cycle prints the full prompt.

# The same prompt template is used for every model call with a modified room state.

room\_state, result = main\_react\_loop(

lambda c, r: c <= 10andnot is\_room\_tidy(r), logging

)

print(room\_state, result)

```
Cycle #1
PROMPT:
	The room is currently in this state: {'clothes': 'floor', 'books': 'scattered', 'wastebin': 'empty'}.
	You can pick any of the following action labels:
	<actions>
	    put_clothes_in_hamper - place clothes into hamper, instead of carrying them around in your hand.
	    pick_up_clothes - pick clothes up from the floor.
	    pick_up_books - pick books up from anywhere not on the shelf
	    place_books_on_shelf - self explanatory.
	    empty_wastebin - self explanatory.
	    done - when everything are in the right place.
	</actions>
	Which one should be the next step to achieve the goal?
	Return a single JSON object containing fields "next_action" and "rationale".
MODEL: {"next_action": "pick_up_clothes", "rationale": "The clothes are on the floor and need to be picked up before they can be put in the hamper."}
ACTION:   pick_up_clothes
EXECUTED: The clothes are now being carried.

Cycle #2
OBSERVATION: The room is currently in this state: {'clothes': 'carrying by hand', 'books': 'scattered', 'wastebin': 'empty'}.
MODEL: {"next_action": "put_clothes_in_hamper", "rationale": "The clothes need to be in the hamper, and they are currently being carried. So the next step is to put them in the hamper."}
ACTION:   put_clothes_in_hamper
EXECUTED: The clothes are now in the hamper.

Cycle #3
OBSERVATION: The room is currently in this state: {'clothes': 'hamper', 'books': 'scattered', 'wastebin': 'empty'}.
MODEL: {"next_action": "pick_up_books", "rationale": "The goal is to have all books on the shelf, so we need to pick them up first."}
ACTION:   pick_up_books
EXECUTED: The books are now in my hand.

Cycle #4
OBSERVATION: The room is currently in this state: {'clothes': 'hamper', 'books': 'in hand', 'wastebin': 'empty'}.
MODEL: {"next_action": "place_books_on_shelf", "rationale": "The books need to be on the shelf, and they are currently in hand."}
ACTION:   place_books_on_shelf
EXECUTED: The books are now on the shelf.

Cycle #5
OBSERVATION: The room is currently in this state: {'clothes': 'hamper', 'books': 'shelf', 'wastebin': '1 item'}.
MODEL: {"next_action": "empty_wastebin", "rationale": "The wastebin has one item in it and needs to be emptied to achieve the goal."}
ACTION:   empty_wastebin
EXECUTED: The wastebin is emptied.

{'clothes': 'hamper', 'books': 'shelf', 'wastebin': 'empty'} The room is tidy!

```

* * *

sparkGemini


keyboard\_arrow\_down

### The Model decides when the goal is reached

The model can also decide if the goal has been reached, instead of the application explicitly testing for the condition.
This is useful in scenarios where the goal state is variable and/or too complex to define in code.

To facilitate that,

Instead of:

```
while cycle <= 10 and not is_room_tidy(room_state):
```

We just have

```
while cycle <= 10:
```

Remember we have previously defined an action "done" above, even though it is not a real function,
the model and the application can utilize that to determine termination. Note this creates an extra cycle.

subdirectory\_arrow\_right1 cell hidden

* * *

sparkGemini


# We are passing in a while loop continuation test function:

# Continue while loop when number of cycles <= 10

# We are no longer testing if the room is tidy within code.

# The decision is now up to the model.

room\_state, result = main\_react\_loop(lambda c, r: c <= 10, logging)

print(room\_state, result)

```
Cycle #1
PROMPT:
	The room is currently in this state: {'clothes': 'floor', 'books': 'scattered', 'wastebin': 'empty'}.
	You can pick any of the following action labels:
	<actions>
	    put_clothes_in_hamper - place clothes into hamper, instead of carrying them around in your hand.
	    pick_up_clothes - pick clothes up from the floor.
	    pick_up_books - pick books up from anywhere not on the shelf
	    place_books_on_shelf - self explanatory.
	    empty_wastebin - self explanatory.
	    done - when everything are in the right place.
	</actions>
	Which one should be the next step to achieve the goal?
	Return a single JSON object containing fields "next_action" and "rationale".
MODEL: {"next_action": "pick_up_clothes", "rationale": "The clothes are on the floor and need to be picked up before they can be put in the hamper."}
ACTION:   pick_up_clothes
EXECUTED: The clothes are now being carried.

Cycle #2
OBSERVATION: The room is currently in this state: {'clothes': 'carrying by hand', 'books': 'scattered', 'wastebin': 'empty'}.
MODEL: {"next_action": "put_clothes_in_hamper", "rationale": "The clothes need to be in the hamper, and they are currently being carried. So the next step is to put them in the hamper."}
ACTION:   put_clothes_in_hamper
EXECUTED: The clothes are now in the hamper.

Cycle #3
OBSERVATION: The room is currently in this state: {'clothes': 'hamper', 'books': 'scattered', 'wastebin': 'empty'}.
MODEL: {"next_action": "pick_up_books", "rationale": "The goal is to have all books on the shelf, so we need to pick them up first."}
ACTION:   pick_up_books
EXECUTED: The books are now in my hand.

Cycle #4
OBSERVATION: The room is currently in this state: {'clothes': 'hamper', 'books': 'in hand', 'wastebin': 'empty'}.
MODEL: {"next_action": "place_books_on_shelf", "rationale": "The books need to be on the shelf, and they are currently in hand."}
ACTION:   place_books_on_shelf
EXECUTED: The books are now on the shelf.

Cycle #5
OBSERVATION: The room is currently in this state: {'clothes': 'hamper', 'books': 'shelf', 'wastebin': '1 item'}.
MODEL: {"next_action": "empty_wastebin", "rationale": "The wastebin has one item in it and needs to be emptied."}
ACTION:   empty_wastebin
EXECUTED: The wastebin is emptied.

Cycle #6
OBSERVATION: The room is currently in this state: {'clothes': 'hamper', 'books': 'shelf', 'wastebin': 'empty'}.
MODEL: {"next_action": "done", "rationale": "All items are already in their correct places: clothes in the hamper, books on the shelf, and the wastebin is empty."}
{'clothes': 'hamper', 'books': 'shelf', 'wastebin': 'empty'} The room is tidy!

```

* * *

sparkGemini


keyboard\_arrow\_down

## Example 2: Incremental Messaging Using the Chat API

subdirectory\_arrow\_right5 cells hidden

* * *

sparkGemini


keyboard\_arrow\_down

### The Chat session loop

The difference between using the stateless API and the stateful chat session is that the list of function choices is only given to the session object once. In subsequent chat messaging we are only sending a message with the action response and the new current
state of the environment. You can see in this loop we formulate the prompt / message differently depending on whether we are at the start of session or we have just performed an action.

subdirectory\_arrow\_right2 cells hidden

* * *

sparkGemini


# Main ReAct loop

defmain\_react\_loop\_chat(session, loop\_continues, log):

    room\_state = {}

    reset\_room\_state(room\_state)

    trash\_added = False

    prev\_action = None

    msg = ""

    cycle = 1

while loop\_continues(cycle, room\_state):

        log(f"Cycle #{cycle}")

# Observe the environment (use Gemini to generate an action thought)

try:  # REASON #

if prev\_action:

                msg = "\\n".join(

\[\
\
                        prev\_action,\
\
f"ENVIRONMENT: The room is currently in this state:{room\_state}.",\
\
"Which should be the next action?",\
\
\]

)

                log("MESSAGE:\\n{}".format(indent(msg, 1, "\\t")))

else:

                msg = get\_next\_step\_full\_prompt(room\_state, cycle, log)

# MODEL CALL

            response = session.send\_message(

                msg, generation\_config={"response\_mime\_type": "application/json"}

)

            action\_label = get\_action\_label(get\_text(response).strip(), log)

except Exception:

            traceback.print\_exc()

            log(response)

break

# Execute the action and get the observation

if action\_label == "done":

break

try:  # ACTION #

# Call the function mapped from the label

            room\_state, acknowledgement = get\_func(action\_label)(room\_state)

            prev\_action = f"ACTION:   {action\_label}\\nEXECUTED: {acknowledgement}\\n"

            log(prev\_action)

except Exception:

            log("No action suggested.")

# Simulating a change in environment

if cycle == 4andnot trash\_added:

            room\_state\["wastebin"\] = "1 item"

            trash\_added = True

        cycle += 1

# End of while loop

# Determine the final result

    result = (

"The room is tidy!"if is\_room\_tidy(room\_state)else"The room is not tidy!"

)

return room\_state, result

* * *

sparkGemini


session = model.start\_chat()

room\_state, result = main\_react\_loop\_chat(session, lambda c, r: c <= 10, logging)

print(room\_state, result)

```
Cycle #1
PROMPT:
	The room is currently in this state: {'clothes': 'floor', 'books': 'scattered', 'wastebin': 'empty'}.
	You can pick any of the following action labels:
	<actions>
	    put_clothes_in_hamper - place clothes into hamper, instead of carrying them around in your hand.
	    pick_up_clothes - pick clothes up from the floor.
	    pick_up_books - pick books up from anywhere not on the shelf
	    place_books_on_shelf - self explanatory.
	    empty_wastebin - self explanatory.
	    done - when everything are in the right place.
	</actions>
	Which one should be the next step to achieve the goal?
	Return a single JSON object containing fields "next_action" and "rationale".
MODEL: {"next_action": "pick_up_clothes", "rationale": "The clothes are on the floor and need to be picked up before they can be put in the hamper."}
ACTION:   pick_up_clothes
EXECUTED: The clothes are now being carried.

Cycle #2
MESSAGE:
	ACTION:   pick_up_clothes
	EXECUTED: The clothes are now being carried.

	ENVIRONMENT: The room is currently in this state: {'clothes': 'carrying by hand', 'books': 'scattered', 'wastebin': 'empty'}.
	Which should be the next action?
MODEL: {"next_action": "put_clothes_in_hamper", "rationale": "Now that the clothes are picked up, they should be put in the hamper."}
ACTION:   put_clothes_in_hamper
EXECUTED: The clothes are now in the hamper.

Cycle #3
MESSAGE:
	ACTION:   put_clothes_in_hamper
	EXECUTED: The clothes are now in the hamper.

	ENVIRONMENT: The room is currently in this state: {'clothes': 'hamper', 'books': 'scattered', 'wastebin': 'empty'}.
	Which should be the next action?
MODEL: {"next_action": "pick_up_books", "rationale": "The clothes are put away, so now we should pick up the scattered books."}
ACTION:   pick_up_books
EXECUTED: The books are now in my hand.

Cycle #4
MESSAGE:
	ACTION:   pick_up_books
	EXECUTED: The books are now in my hand.

	ENVIRONMENT: The room is currently in this state: {'clothes': 'hamper', 'books': 'in hand', 'wastebin': 'empty'}.
	Which should be the next action?
MODEL: {"next_action": "place_books_on_shelf", "rationale": "The books need to be placed on the shelf to achieve the goal."}
ACTION:   place_books_on_shelf
EXECUTED: The books are now on the shelf.

Cycle #5
MESSAGE:
	ACTION:   place_books_on_shelf
	EXECUTED: The books are now on the shelf.

	ENVIRONMENT: The room is currently in this state: {'clothes': 'hamper', 'books': 'shelf', 'wastebin': '1 item'}.
	Which should be the next action?
MODEL: {"next_action": "empty_wastebin", "rationale": "The wastebin has one item in it and needs to be emptied."}
ACTION:   empty_wastebin
EXECUTED: The wastebin is emptied.

Cycle #6
MESSAGE:
	ACTION:   empty_wastebin
	EXECUTED: The wastebin is emptied.

	ENVIRONMENT: The room is currently in this state: {'clothes': 'hamper', 'books': 'shelf', 'wastebin': 'empty'}.
	Which should be the next action?
MODEL: {"next_action": "done", "rationale": "All clothes are in the hamper, books are on the shelf, and the wastebin is empty. The room is tidy."}
{'clothes': 'hamper', 'books': 'shelf', 'wastebin': 'empty'} The room is tidy!

```

* * *

sparkGemini


keyboard\_arrow\_down

### Display the full chat history

subdirectory\_arrow\_right1 cell hidden

* * *

sparkGemini


print(session.history)

```
[role: "user"\
parts {\
  text: "The room is currently in this state: {\'clothes\': \'floor\', \'books\': \'scattered\', \'wastebin\': \'empty\'}.\nYou can pick any of the following action labels: \n<actions>\n    put_clothes_in_hamper - place clothes into hamper, instead of carrying them around in your hand.\n    pick_up_clothes - pick clothes up from the floor.\n    pick_up_books - pick books up from anywhere not on the shelf\n    place_books_on_shelf - self explanatory.\n    empty_wastebin - self explanatory.\n    done - when everything are in the right place.\n</actions>\nWhich one should be the next step to achieve the goal? \nReturn a single JSON object containing fields \"next_action\" and \"rationale\"."\
}\
, role: "model"\
parts {\
  text: "{\"next_action\": \"pick_up_clothes\", \"rationale\": \"The clothes are on the floor and need to be picked up before they can be put in the hamper.\"}\n\n"\
}\
, role: "user"\
parts {\
  text: "ACTION:   pick_up_clothes\nEXECUTED: The clothes are now being carried.\n\nENVIRONMENT: The room is currently in this state: {\'clothes\': \'carrying by hand\', \'books\': \'scattered\', \'wastebin\': \'empty\'}.\nWhich should be the next action?"\
}\
, role: "model"\
parts {\
  text: "{\"next_action\": \"put_clothes_in_hamper\", \"rationale\": \"Now that the clothes are picked up, they should be put in the hamper.\"}\n"\
}\
, role: "user"\
parts {\
  text: "ACTION:   put_clothes_in_hamper\nEXECUTED: The clothes are now in the hamper.\n\nENVIRONMENT: The room is currently in this state: {\'clothes\': \'hamper\', \'books\': \'scattered\', \'wastebin\': \'empty\'}.\nWhich should be the next action?"\
}\
, role: "model"\
parts {\
  text: "{\"next_action\": \"pick_up_books\", \"rationale\": \"The clothes are put away, so now we should pick up the scattered books.\"}\n"\
}\
, role: "user"\
parts {\
  text: "ACTION:   pick_up_books\nEXECUTED: The books are now in my hand.\n\nENVIRONMENT: The room is currently in this state: {\'clothes\': \'hamper\', \'books\': \'in hand\', \'wastebin\': \'empty\'}.\nWhich should be the next action?"\
}\
, role: "model"\
parts {\
  text: "{\"next_action\": \"place_books_on_shelf\", \"rationale\": \"The books need to be placed on the shelf to achieve the goal.\"}\n"\
}\
, role: "user"\
parts {\
  text: "ACTION:   place_books_on_shelf\nEXECUTED: The books are now on the shelf.\n\nENVIRONMENT: The room is currently in this state: {\'clothes\': \'hamper\', \'books\': \'shelf\', \'wastebin\': \'1 item\'}.\nWhich should be the next action?"\
}\
, role: "model"\
parts {\
  text: "{\"next_action\": \"empty_wastebin\", \"rationale\": \"The wastebin has one item in it and needs to be emptied.\"}\n"\
}\
, role: "user"\
parts {\
  text: "ACTION:   empty_wastebin\nEXECUTED: The wastebin is emptied.\n\nENVIRONMENT: The room is currently in this state: {\'clothes\': \'hamper\', \'books\': \'shelf\', \'wastebin\': \'empty\'}.\nWhich should be the next action?"\
}\
, role: "model"\
parts {\
  text: "{\"next_action\": \"done\", \"rationale\": \"All clothes are in the hamper, books are on the shelf, and the wastebin is empty. The room is tidy.\"}\n"\
}\
]

```

* * *

sparkGemini


keyboard\_arrow\_down

## Example 3: Leveraging Gemini Function Calling Support

For more details please refer to the documentation on Function Calling @https://www.google.com/url?q=https%3A%2F%2Fcloud.google.com%2Fvertex-ai%2Fgenerative-ai%2Fdocs%2Fmultimodal%2Ffunction-calling.

In the last two examples we simulated the function calling feature by explicitly prompting the model with a list of action labels and setting a JSON mode output. This example uses the Function Calling feature, the list of possible actions are supplied as 'Tool' declarations, and by default the function calling feature returns structured results.

subdirectory\_arrow\_right10 cells hidden

* * *

sparkGemini


keyboard\_arrow\_down

### Tool Declarations

See Best Practices @https://www.google.com/url?q=https%3A%2F%2Fcloud.google.com%2Fvertex-ai%2Fgenerative-ai%2Fdocs%2Fmultimodal%2Ffunction-calling%23best-practices for guidance on achieving good results with Function Calling.

subdirectory\_arrow\_right1 cell hidden

* * *

sparkGemini


# Functions for actions (replace these with Gemini function calls)

pick\_up\_clothes\_func = FunctionDeclaration(

    name="pick\_up\_clothes",

    description="The act of picking clothes up from any place",

    parameters={"type": "object"},

)

put\_clothes\_in\_hamper\_func = FunctionDeclaration(

    name="put\_clothes\_in\_hamper",

    description="Put the clothes being carried into a hamper",

    parameters={"type": "object"},

)

pick\_up\_books\_func = FunctionDeclaration(

    name="pick\_up\_books",

    description="The act of picking books up from any place",

    parameters={"type": "object"},

)

place\_books\_on\_shelf\_func = FunctionDeclaration(

    name="place\_books\_on\_shelf",

    description="Put the books being carried onto a shelf",

    parameters={"type": "object"},

)

empty\_wastebin\_func = FunctionDeclaration(

    name="empty\_wastebin",

    description="Empty out the wastebin",

    parameters={"type": "object"},

)

done\_func = FunctionDeclaration(

    name="done", description="The goal has been reached", parameters={"type": "object"}

)

room\_tools = Tool(

    function\_declarations=\[\
\
        pick\_up\_clothes\_func,\
\
        put\_clothes\_in\_hamper\_func,\
\
        pick\_up\_books\_func,\
\
        place\_books\_on\_shelf\_func,\
\
        empty\_wastebin\_func,\
\
        done\_func,\
\
\],

)

* * *

sparkGemini


keyboard\_arrow\_down

### Model with tool declarations

subdirectory\_arrow\_right1 cell hidden

* * *

sparkGemini


NOTE: Tools can be passed in during the initial creation of the model reference as below, or during `send_message()`, and `generate_content()`. The choice depends on the variability of the set of tools to be used.

```
model_fc = GenerativeModel(
    "gemini-2.0-flash",
    system_instruction=[\
       "You are an assistant that helps me tidy my room."\
       "Your goal is to make sure all the books are on the shelf, all clothes are in the hamper, and the trash is empty.",\
       "You cannot receive any input from me."\
    ],
    tools=[ room_tools ],
)
```

subdirectory\_arrow\_right0 cells hidden

* * *

sparkGemini


keyboard\_arrow\_down

### The function calling model response

With Function Calling, the choices of the tools are supplied through the API and is no longer necessary to include them in your prompt, and also unnecessary to specify the output format. For more details see the function calling API Reference @https://www.google.com/url?q=https%3A%2F%2Fcloud.google.com%2Fvertex-ai%2Fgenerative-ai%2Fdocs%2Fmodel-reference%2Ffunction-calling%23python_1.

```
response = session.send_message( msgs, tools=[ room_tools ])
```

The following raw model response is expected:

```
MESSAGE:
    ENVIRONMENT: The room is currently in this state: {'clothes': 'floor', 'books': 'scattered', 'wastebin': 'empty'}.
    Which should be the next action?

RAW RESPONSE:

candidates {
  content {
    role: "model"
    parts {
      function_call {
        name: "pick_up_clothes"
        args {
        }
      }
    }
  },
  finish_reason: STOP,
  ...
}
```

Use the following function to extract the function calling information from the response object:

```
# Helper function to extract one or more function calls from a Gemini Function Call response
def extract_function_calls(response: GenerationResponse) -> List[Dict]:
    function_calls = []
    if response.candidates[0].function_calls:
        for function_call in response.candidates[0].function_calls:
            function_call_dict = {function_call.name: {}}
            for key, value in function_call.args.items():
                function_call_dict[function_call.name][key] = value
            function_calls.append(function_call_dict)
    return function_calls
```

In recent versions of specific Gemini models (from May 2024 and on), Gemini has the ability to return two or more function calls in parallel (i.e., two or more function call responses within the first function call response object). Parallel function calling allows you to fan out and parallelize your API calls or other actions that you perform in your application code, so you don't have to work through each function call response and return one-by-one! Refer to the Gemini Function Calling documentation @https://www.google.com/url?q=https%3A%2F%2Fcloud.google.com%2Fvertex-ai%2Fgenerative-ai%2Fdocs%2Fmultimodal%2Ffunction-calling for more information on which Gemini model versions support parallel function calling, and this notebook on parallel function calling @https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/parallel_function_calling.ipynb for examples.

subdirectory\_arrow\_right0 cells hidden

* * *

sparkGemini


keyboard\_arrow\_down

### The Main ReAct Loop

In this third example we reorganized the code for easier comprehension. The 3 main components of the loop are broken out into separate functions:

- observe and reason - modified to use the Function Calling feature\*
- execute action - simplified
- main loop - calling the other two functions cyclically.

\\* Main changes:

- The list of tools declared above are sent to the model via the `tools=` argument of the `send_message()` call.
- Any function execution responses are reported back to the model as a structured input 'Part' object in the next cycle.

subdirectory\_arrow\_right4 cells hidden

* * *

sparkGemini


# Wrapping the observation and model calling code into a function for better main loop readability.

defobserve\_and\_reason(session, state: dict, prev\_action: str, log: Callable) -> str:

"""Uses the language model (Gemini) to select the next action."""

try:

        msgs = \[\]

if prev\_action:

            msgs.append(

                Part.from\_function\_response(

                    name="previous\_action", response={"content": prev\_action}

)

)

        prompt = "\\n".join(

\[\
\
f"ENVIRONMENT: The room is currently in this state:{state}.",\
\
"Which should be the next action?",\
\
\]

)

        msgs.append(prompt)

        log(

"MESSAGE:\\n{}".format(

                indent(

"\\n".join(\[prev\_action, prompt\]if prev\_action else\[prompt\]),

1,

"\\t",

)

)

)

        response = session.send\_message(

            msgs, tools=\[room\_tools\]

)# JSON mode unnecessary.

        action\_label = get\_action\_from\_function\_call(get\_function\_call(response), log)

return action\_label

except Exception:

        log(f"Error during action selection: {e}")

        traceback.print\_exc()

return"done"# Or a suitable default action

* * *

sparkGemini


# Wrapping the action execution code into a function for better main loop readability.

defexecute\_action(state: dict, action\_label: str, log: Callable) -> tuple\[dict, str\]:

"""Executes the action on the room state and returns the updated state and an acknowledgement."""

try:

# Call the function mapped from the label

        state, acknowledgement = get\_func(action\_label)(state)

except Exception:

        acknowledgement = "No action suggested or action not recognized."

return state, acknowledgement

* * *

sparkGemini


# Main ReAct loop

defmain\_react\_loop\_chat\_fc(session, loop\_continues, log):

    room\_state = {}

    reset\_room\_state(room\_state)

    trash\_added = False

    prev\_action = None

    cycle = 1

while loop\_continues(cycle, room\_state):

        log(f"Cycle #{cycle}")

# Observe the environment (use Gemini to generate an action thought)

        action\_label = observe\_and\_reason(session, room\_state, prev\_action, log)

# Execute the action and get the observation

if action\_label == "done":

break

        room\_state, acknowledgement = execute\_action(room\_state, action\_label, log)

        prev\_action = f"ACTION:   {action\_label}\\nEXECUTED: {acknowledgement}"

        log(prev\_action + "\\n")

# Simulating a change in environment

if cycle == 4andnot trash\_added:

            room\_state\["wastebin"\] = "1 item"

            trash\_added = True

        cycle += 1

# End of while loop

# Determine the final result

    result = (

"The room is tidy!"if is\_room\_tidy(room\_state)else"The room is not tidy!"

)

return room\_state, result

* * *

sparkGemini


session = model.start\_chat()

room\_state, result = main\_react\_loop\_chat\_fc(session, lambda c, r: c <= 10, logging)

print(room\_state, result)

```
Cycle #1
MESSAGE:
	ENVIRONMENT: The room is currently in this state: {'clothes': 'floor', 'books': 'scattered', 'wastebin': 'empty'}.
	Which should be the next action?
MODEL: {
  "name": "pick_up_clothes",
  "args": {}
}
ACTION:   pick_up_clothes
EXECUTED: The clothes are now being carried.

Cycle #2
MESSAGE:
	ACTION:   pick_up_clothes
	EXECUTED: The clothes are now being carried.
	ENVIRONMENT: The room is currently in this state: {'clothes': 'carrying by hand', 'books': 'scattered', 'wastebin': 'empty'}.
	Which should be the next action?
MODEL: {
  "name": "put_clothes_in_hamper",
  "args": {}
}
ACTION:   put_clothes_in_hamper
EXECUTED: The clothes are now in the hamper.

Cycle #3
MESSAGE:
	ACTION:   put_clothes_in_hamper
	EXECUTED: The clothes are now in the hamper.
	ENVIRONMENT: The room is currently in this state: {'clothes': 'hamper', 'books': 'scattered', 'wastebin': 'empty'}.
	Which should be the next action?
MODEL: {
  "name": "pick_up_books",
  "args": {}
}
ACTION:   pick_up_books
EXECUTED: The books are now in my hand.

Cycle #4
MESSAGE:
	ACTION:   pick_up_books
	EXECUTED: The books are now in my hand.
	ENVIRONMENT: The room is currently in this state: {'clothes': 'hamper', 'books': 'in hand', 'wastebin': 'empty'}.
	Which should be the next action?
MODEL: {
  "name": "place_books_on_shelf",
  "args": {}
}
ACTION:   place_books_on_shelf
EXECUTED: The books are now on the shelf.

Cycle #5
MESSAGE:
	ACTION:   place_books_on_shelf
	EXECUTED: The books are now on the shelf.
	ENVIRONMENT: The room is currently in this state: {'clothes': 'hamper', 'books': 'shelf', 'wastebin': '1 item'}.
	Which should be the next action?
MODEL: {
  "name": "empty_wastebin",
  "args": {}
}
ACTION:   empty_wastebin
EXECUTED: The wastebin is emptied.

Cycle #6
MESSAGE:
	ACTION:   empty_wastebin
	EXECUTED: The wastebin is emptied.
	ENVIRONMENT: The room is currently in this state: {'clothes': 'hamper', 'books': 'shelf', 'wastebin': 'empty'}.
	Which should be the next action?
MODEL: {
  "name": "done",
  "args": {}
}
{'clothes': 'hamper', 'books': 'shelf', 'wastebin': 'empty'} The room is tidy!

```

* * *

Colab paid products @https://colab.research.google.com/signup?utm_source=footer&utm_medium=link&utm_campaign=footer_links
 -
Cancel contracts here @https://colab.research.google.com/cancel-subscription

more\_horiz

more\_horiz

more\_horiz

iframe @https://qu701z6wyoq-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab_20250623-060057_RC00_774714275

iframe @https://al6ga6kl88a-496ff2e9c6d22116-0-colab.googleusercontent.com/outputframe.html?vrz=colab_20250623-060057_RC00_774714275

data\_objectVariablesterminalTerminal

View on GitHub

New notebook in Drive

Open notebook

Upload notebook

Rename

Save a copy in Drive

Save a copy as a GitHub Gist

Save

Revision history

Download
►

Print

Download .ipynb

Download .py

Undo

Redo

Select all cells

Cut cell or selection

Copy cell or selection

Paste

Delete selected cells

Find and replace

Find next

Find previous

Notebook settings

Clear all outputs

check

Table of contents

Notebook info

Executed code history

Comments
►

Collapse sections

Expand sections

Save collapsed section layout

Show/hide code

Show/hide output

Focus next tab

Focus previous tab

Move tab to next pane

Move tab to previous pane

Hide comments

Minimize comments

Expand comments

Code cell

Text cell

Section header cell

Scratch code cell

Code snippets

Add a form field

Run all

Run before

Run the focused cell

Run selection

Run cell and below

Interrupt execution

Restart session

Restart session and run all

Disconnect and delete runtime

Change runtime type

Manage sessions

View resources

View runtime logs

Command palette

Settings

Keyboard shortcuts

Diff notebooks(opens in a new tab)

Frequently asked questions

View release notes

Search code snippets

Report a bug

Send feedback

View terms of service

iframe @/_/bscframe

iframe @https://www.google.com/recaptcha/api2/anchor?ar=1&k=6LfQPtEUAAAAAHBpAdFng54jyuB1V5w5dofknpip&co=aHR0cHM6Ly9jb2xhYi5yZXNlYXJjaC5nb29nbGUuY29tOjQ0Mw..&hl=en&v=h7qt2xUGz2zqKEhSc8DD8baZ&size=invisible&cb=jp9bmwfcohdq

### Original URL
https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/function-calling/intro_diy_react_agent.ipynb
</details>

---

## Additional Sources Scraped

---
<details>
<summary>What is AI Agent Orchestration? | IBM</summary>

# What is AI agent orchestration?

Artificial intelligence (AI) agent orchestration is the process of coordinating multiple specialized AI agents within a unified system to efficiently achieve shared objectives.

Rather than relying on a single, general-purpose AI solution, AI agent orchestration employs a network of AI agents, each designed for specific tasks, working together to automate complex workflows and processes.

To fully understand AI agent orchestration, it's essential to first understand AI agents themselves. This involves understanding the differences between two key types of AI: generative AI, which creates original content based on a user’s prompt, and agentic AI, which autonomously makes decisions and acts to pursue complex goals with minimal supervision.

AI assistants exist on a continuum, starting with rule-based chatbots, progressing to more advanced virtual assistants and evolving into generative AI and large language model (LLM) powered assistants capable of handling single-step tasks. At the top of this progression are AI agents, which operate autonomously. These agents make decisions, design workflows and use function calling to connect with external tools—such as application programming interfaces (APIs), data sources, web searches and even other AI agents—to fill gaps in their knowledge. This is agentic AI.

AI agents are specialized, meaning each one is optimized for a particular function. Some agents focus on business and customer-facing tasks like billing, troubleshooting, scheduling and decision-making, while others handle more technical functions like natural language processing (NLP), data retrieval and process automation. Advanced LLMs such as OpenAI's ChatGPT-4o or Google's Gemini often power these agents, with generative AI capabilities enabling them to create human-like responses and handle complex tasks autonomously.

Multi-agent systems (MAS) emerge when multiple AI agents collaborate, either in a structured or decentralized manner, to solve complex tasks more efficiently than a single agent might.

In practice, AI agent orchestration functions like a digital symphony. Each agent has a unique role and the system is guided by an orchestrator—either a central AI agent or framework—that manages and coordinates their interactions. The orchestrator helps synchronize these specialized agents, ensuring that the right agent is activated at the right time for each task. This coordination is crucial for handling multifaceted workflows that involve various tasks, helping ensure that processes are run seamlessly and efficiently.

For example, as part of customer service automation, the orchestrator agent (the system responsible for managing AI agents) might determine whether to engage a billing agent versus a technical support agent, helping ensure that customers receive seamless and relevant assistance. In MAS, agents might coordinate without a single orchestrator, dynamically communicating to collaboratively solve problems.

The benefits of AI agent orchestration are significant in industries with complex, dynamic needs such as telecommunications, banking and healthcare. By deploying specialized agents that are trained on targeted datasets and workflows, businesses can enhance operational efficiency, improve decision-making and deliver more accurate, efficient and context-aware results for both employees and customers.

## Why AI agent orchestration is important

As AI systems grow more advanced, a single AI model or agent is often insufficient for handling complex tasks. Autonomous systems frequently struggle to collaborate because they are built across multiple clouds and applications, leading to siloed operations and inefficiencies. AI agent orchestration bridges these gaps, enabling multiple AI agents to work together efficiently and ensuring that sophisticated tasks are run seamlessly.

In large-scale applications such as healthcare, finance and customer service, multiple agents often need to work together to handle different aspects of a task. For example, in healthcare, AI agents can coordinate between diagnostic tools, patient management systems and administrative workflows to streamline operations and enhance treatment accuracy. Without orchestration, these agents might work in isolation, leading to inefficiencies, redundancies or gaps in execution.

By managing interactions between multi-agent systems, orchestration helps ensure that each agent contributes effectively toward a shared goal. It optimizes workflows, minimizes errors and enhances interoperability, allowing AI systems to dynamically allocate resources, prioritize tasks and respond to changing conditions in real time. This capability is valuable in fields requiring continuous optimization such as supply chain management and personalized digital assistants.

As AI systems continue to evolve, AI agent orchestration becomes increasingly essential for unlocking their full potential.

## Types of AI agent orchestration

There are several types of AI agent orchestration. Real-world systems often combine multiple orchestration styles for more effective results.

**Centralized orchestration**: A single AI orchestrator agent acts as the "brain" of the system, directing all the other agents, assigning tasks and making final decisions. This structured approach helps ensure consistency, control and predictable workflows.

**Decentralized orchestration**: This model shifts away from a single, controlling entity, allowing MAS to function through direct communication and collaboration. Agents make independent decisions or reach a consensus as a group. This makes the system more scalable and resilient since no single failure can bring it down.

**Hierarchical orchestration**: Here, AI agents are arranged in layers, resembling a tiered command structure. Higher-level orchestrator agents oversee and manage lower-level agents, striking a balance between strategic control and task-specific execution. This allows for more organized workflows while still enabling specialized agents to operate with some autonomy. If the hierarchy becomes too rigid, adaptability can suffer.

**Federated orchestration**: This approach focuses on collaboration between independent AI agents or separate organizations, allowing them to work together without fully sharing data or relinquishing control over their individual systems. This is especially useful in situations where privacy, security or regulatory constraints prevent unrestricted data sharing, such as in healthcare, banking or cross-company collaborations.

## Comparing AI agent orchestration with related practices

**AI orchestration** manages and automates various AI components—like machine learning models, data pipelines and APIs—to help ensure that they work together efficiently within a system. It focuses on optimizing performance, automating repetitive tasks, supporting scalability and system-wide performance.

**AI agent orchestration** is a subset of AI orchestration that focuses specifically on coordinating autonomous AI agents—software entities that can make independent decisions and take actions. It helps ensure that agents collaborate effectively, assigning tasks and structuring workflows.

**Multi-agent orchestration** goes a step further, managing multiple AI agents working together on complex problems. It deals with communication, role allocation and conflict resolution to help ensure seamless collaboration between agents.

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

### Original URL
https://www.ibm.com/think/topics/ai-agent-orchestration
</details>

---
<details>
<summary>[2504.19678] From LLM Reasoning to Autonomous AI Agents: A Comprehensive Review</summary>

## I Introduction

Large Language Models (LLMs) such as OpenAI’s GPT-4 [1], Qwen2.5-Omni [2], DeepSeek-R1 [3], and Meta’s LLaMA [4] have transformed AI by enabling human-like text generation and advanced natural language processing, spurring innovation in conversational agents, automated content creation, and real-time translation [5]. Recent enhancements have extended their utility to multimodal tasks, including text-to-image and text-to-video generation that broaden the scope of generative AI applications [6]. However, their dependence on static pre-training data can lead to outdated outputs and hallucinated responses [7, 8], a limitation that Retrieval-Augmented Generation (RAG) addresses by incorporating real-time data from knowledge bases, APIs, or the web [9, 10]. Building on this, the evolution of intelligent agents employing reflection, planning, and multi-agent collaboration has given rise to Agentic RAG systems, which dynamically orchestrate information retrieval and iterative refinement to manage complex workflows effectively [11, 12].

Recent advances in large language models have paved the way for highly autonomous AI systems that can independently handle complex research tasks. These systems, often referred to as agentic AI, can generate hypotheses, conduct literature reviews, design experiments, analyze data, accelerate scientific discovery, and reduce research costs [13, 14, 15, 16]. Several frameworks, such as LitSearch, ResearchArena, and Agent Laboratory, have been developed to automate various research tasks, including citation management and academic survey generation [17, 18, 19]. However, challenges persist, especially in executing domain-specific literature reviews and ensuring the reproducibility and reliability of automated processes [20, 21]. Parallel to these developments in research automation, large language model-based agents have also begun to transform the medical field [22]. These agents are increasingly used for diagnostic support, patient communication, and medical education by integrating clinical guidelines, medical knowledge bases, and healthcare systems. Despite their promise, these applications face significant hurdles, including concerns over reliability, reproducibility, ethical governance, and safety [23, 24, 25]. Addressing these issues is crucial for ensuring that LLM-based agents can be effectively and responsibly incorporated into clinical practice, underscoring the need for comprehensive evaluation frameworks that can reliably measure their performance across various healthcare tasks [26, 27, 28].

LLM-based agents are emerging as a promising frontier in AI, combining reasoning and action to interact with complex digital environments [29, 30]. Therefore, various approaches have been explored to enhance LLM-based agents, from combining reasoning and acting using techniques like React [31] and Monte Carlo Tree Search [32] to synthesizing high-quality data with methods like Learn-by-Interact [33], which sidestep assumptions such as state reversals. Other strategies involve training on human-labeled or GPT-4 distilled data with systems like AgentGen [34] and AgentTuning [35] to generate trajectory data. At the same time, reinforcement learning methods utilize offline algorithms and iterative refinement through reward models and feedback to enhance efficiency and performance in realistic environments [36, 37].

LLM-based Multi-Agents harness the collective intelligence of multiple specialized agents, enabling advanced capabilities over single-agent systems by simulating complex real-world environments through collaborative planning, discussion, and decision-making. This approach leverages the communicative strengths and domain-specific expertise of LLMs, allowing distinct agents to interact effectively, much like human teams tackling problem-solving tasks [38, 39]. Recent research highlights promising applications across various fields, including software development [40, 41], multi-robot systems [42, 43], society simulation [44], policy simulation [45], and game simulation [46].

The main contributions of this study are:

- We present a comparative table of benchmarks developed between 2019 and 2025 that rigorously evaluate large language models and autonomous AI agents across multiple domains.
- We propose a taxonomy of approximately 60 LLM and AI-agent benchmarks, including general and academic knowledge reasoning, mathematical problem solving, code generation and software engineering, factual grounding and retrieval, domain-specific evaluations, multimodal and embodied tasks, task orchestration, and interactive and agentic assessments.
- We present prominent AI-agent frameworks from 2023 to 2025 that integrate large language models with modular toolkits, enabling autonomous decision-making and multi-step reasoning.
- We provide applications of autonomous AI agents in various fields, including materials science and biomedical research, academic ideation and software engineering, synthetic data generation and chemical reasoning, mathematical problem-solving and geographic information systems, as well as multimedia, healthcare, and finance.
- We survey agent-to-agent collaboration protocols, namely the Agent Communication Protocol (ACP), the Model Context Protocol (MCP), and the Agent-to-Agent Protocol (A2A).
- We outline recommendations for future research on autonomous AI agents, specifically advanced reasoning strategies, failure modes in multi-agent large language model (LLM) systems, automated scientific discovery, dynamic tool integration via reinforcement learning, integrated search capabilities, and security vulnerabilities in agent protocols.

## II Related Works

The growing field of autonomous AI agents powered by large language models has inspired a wide range of research efforts across multiple domains. In this section, we review the most relevant studies that investigate the integration of LLM-based agents into software engineering, propose agent architectures and evaluation frameworks, explore the development of multi-agent systems, and examine domain-specific applications, including healthcare, game-theoretic scenarios, GUI interactions, personal assistance, scientific discovery, and chemistry.

### II-A LLM-based Agents in Software Engineering

Wang et al. [47] present a survey that bridges Large Language Model (LLM)-based agent technologies with software engineering (SE). It highlights how LLMs have achieved significant success in various domains and have been integrated into SE tasks, often under the agent paradigm, whether explicitly or implicitly. The study presents a structured framework for LLM-based agents in SE, comprising three primary modules: perception, memory, and action. Jin et al. [48] investigate the use of large language models (LLMs) and LLM-based agents in software engineering, distinguishing between the traditional capabilities of LLMs and the enhanced functionalities offered by autonomous agents. It highlights the significant success of LLMs in tasks such as code generation and vulnerability detection, while also addressing their limitations, specifically the issues of autonomy and self-improvement that LLM-based agents aim to overcome. The paper provides an extensive review of current practices across six key domains: requirement engineering, code generation, autonomous decision-making, software design, test generation, and software maintenance. In a complementary study, Jin et al. [48] investigate the use of large language models (LLMs) and LLM-based agents in software engineering, distinguishing between the traditional capabilities of LLMs and the enhanced functionalities offered by autonomous agents. It highlights the significant success of LLMs in tasks such as code generation and vulnerability detection, while also addressing their limitations, specifically, issues of autonomy and self-improvement that LLM-based agents aim to overcome. The paper provides an extensive review of current practices across six key domains: requirement engineering, code generation, autonomous decision-making, software design, test generation, and software maintenance.

### II-B Agent Architectures and Evaluation Frameworks

Singh et al. [49] delves into Agentic Retrieval-Augmented Generation (Agentic RAG), a sophisticated evolution of traditional Retrieval-Augmented Generation systems that enhances the capabilities of large language models (LLMs). While LLMs have transformed AI through human-like text generation and language understanding, their dependence on static training data often results in outdated or imprecise responses. The paper addresses these limitations by embedding autonomous agents within the RAG framework, enabling dynamic, real-time data retrieval and adaptive workflows. It details how agentic design patterns such as reflection, planning, tool utilization, and multi-agent collaboration equip these systems to manage complex tasks and support multi-step reasoning. The survey offers a comprehensive taxonomy of Agentic RAG architectures, highlights key applications across various sectors, including healthcare, finance, and education, and outlines practical implementation strategies.

Complementing this architectural perspective, Yehudai et al. [50] mark a significant milestone in artificial intelligence by surveying evaluation methodologies for agents powered by large language models (LLMs). It thoroughly reviews the capabilities of these agents, focusing on core functions such as planning, tool utilization, self-reflection, and memory, while assessing specialized applications ranging from web interactions to software engineering and conversational tasks. The authors uncover a clear trend toward developing more rigorous, dynamically updated evaluation frameworks by examining both targeted benchmarks for domain-specific applications and those designed for more generalist agents. Moreover, the paper critically highlights existing deficiencies in the field, notably the need for metrics that more effectively capture cost efficiency, safety, and robustness. In doing so, it maps the current landscape of agent evaluation and sets forth compelling directions for future inquiry, underscoring the importance of scalable and fine-grained evaluation techniques in the rapidly evolving AI domain.

Similarly, Chen et al. [51] focus on Role-Playing Agents (RPAs), a growing class of LLM-based agents that mimic human behavior across various tasks. Recognizing the inherent challenges in evaluating such diverse systems, the authors systematically reviewed 1,676 papers published between January 2021 and December 2024. Their extensive analysis identifies six key agent attributes, seven task attributes, and seven evaluation metrics that are prevalent in the current literature. Based on these insights, the paper proposes an evidence-based, actionable, and generalizable evaluation guideline designed to standardize the assessment of RPAs.

### II-C Multi-Agent Systems

Yan et al. [52] provides a comprehensive survey on integrating LLMs into multi-agent systems (MAS). Their work emphasizes the communication-centric aspects that enable agents to engage in both cooperative and competitive interactions, thereby tackling tasks that are unmanageable for individual agents. The paper examines system-level features, internal communication mechanisms, and challenges, including scalability, security, and multimodal integration. In a related study, Guo et al. [38] offer an extensive overview of LLM-based multi-agent systems, charting the evolution from single-agent decision-making to collaborative frameworks that enhance collective problem-solving and world simulation. In a related study, Guo et al. [38] provide an extensive overview of large language model (LLM)-based multi-agent systems, building on the success of LLMs in autonomous planning and reasoning. The authors detail how the evolution from single-agent decision-making to collaborative multi-agent frameworks has enabled significant advances in complex problem-solving and world simulation. Key aspects of these systems are examined, including the domains and environments they simulate, the profiling and communication strategies employed by individual agents, and the mechanisms that underpin the enhancement of their collective capacities.

### II-D Domain-Specific Applications

#### II-D1 Healthcare

Wang et al. [28] explores the transformative impact of LLM-based agents on healthcare, presenting a detailed review of their architectures, applications, and inherent challenges. It dissects the core components of medical agent systems, such as system profiles, clinical planning mechanisms, and medical reasoning frameworks, while also discussing methods to enhance external capacities. Major application areas include clinical decision support, medical documentation, training simulations, and overall healthcare service optimization. The survey further evaluates the performance of these agents using established frameworks and metrics, identifying persistent challenges such as hallucination management, multimodal integration, and ethical considerations.

#### II-D2 Social Agents in Game-Theoretic Scenarios

Feng et al. [53] provide a review of research on LLM-based social agents in game-theoretic scenarios. This area has gained prominence for assessing social intelligence in AI systems. The authors categorize the literature into three main components. First, the game framework is examined, highlighting various choice- and communication-focused scenarios. Second, the paper explores the attributes of social agents, examining their preferences, beliefs, and reasoning capabilities. Third, it discusses evaluation protocols incorporating game-agnostic and game-specific metrics to assess performance. By synthesizing current studies and outlining future research directions, the survey offers valuable insights to further the development and systematic evaluation of social agents within game-theoretic contexts.

#### II-D3 GUI Agents

Zhang et al. [54] review LLM-brained GUI agents, marking a paradigm shift in human-computer interaction through integrating multimodal LLMs. It traces the historical evolution of GUI automation, detailing how advancements in natural language understanding, code generation, and visual processing have enabled these agents to interpret complex graphical user interface (GUI) elements and execute multi-step tasks from conversational commands. The survey systematically examines the core components of these systems, including existing frameworks, data collection and utilization methods for training, and the development of specialized large-scale action models for GUI tasks.

#### II-D4 Personal LLM Agents

Li et al. [55] explore the evolution of intelligent personal assistants (IPAs) by focusing on Personal LLM Agents LLM-based agents that deeply integrate personal data and devices to provide enhanced personal assistance. The authors outline the limitations of traditional IPAs, including insufficient understanding of user intent, task planning, and tool utilization, which have hindered their practicality and scalability. In contrast, the emergence of foundation models like LLMs offer new possibilities by leveraging advanced semantic understanding and reasoning for autonomous problem-solving. The survey systematically reviews the architecture and design choices underlying Personal LLM Agents, informed by expert opinions, and examines key challenges related to intelligence, efficiency, and security. Furthermore, it comprehensively analyzes representative solutions addressing these challenges, laying the groundwork for Personal LLM Agents to become a major paradigm in next-generation end-user software.

#### II-D5 Scientific Discovery

Gridach et al. [21] explore the transformative role of Agentic AI in scientific discovery, underscoring its potential to automate and enhance research processes. It reviews how these systems, endowed with reasoning, planning, and autonomous decision-making capabilities, are revolutionizing traditional research activities, including literature reviews, hypothesis generation, experimental design, and data analysis. The paper highlights recent advancements across multiple scientific domains, such as chemistry, biology, and materials science, by categorizing existing Agentic AI systems and tools. It provides a detailed discussion on key evaluation metrics, implementation frameworks, and datasets used in the field, offering valuable insights into current practices. Moreover, the paper critically addresses significant challenges, including automating comprehensive literature reviews, ensuring system reliability, and addressing ethical concerns. It outlines future research directions, emphasizing the importance of human-AI collaboration and improved system calibration.

#### II-D6 Chemistry

Ramos et al. [56] examine the transformative impact of large language models (LLMs) in chemistry, focusing on their roles in molecule design, property prediction, and synthesis optimization. It highlights how LLMs not only accelerate scientific discovery through automation but also discuss the advent of LLM-based autonomous agents. These agents extend the functionality of LLMs by interfacing with their environment and performing tasks such as literature scraping, automated laboratory control, and synthesis planning. Expanding the discussion beyond chemistry, the review also considers applications across other scientific domains.

### II-E Comparison with Our Survey

Table I presents a consolidated view of how existing works cover key themes, benchmarks, AI agent frameworks, AI agent applications, AI agents protocols, and challenges & open problems against our survey. While prior studies typically focus on one or two aspects (e.g., Yehudai et al. [50] on evaluation benchmarks, Singh et al. [49] on RAG architectures, Yan et al. [52] on multi-agent communication, or Wang et al. [28] on domain-specific applications), none integrate the full spectrum of developments in a single, unified treatment. In contrast, our survey is the first to systematically combine state-of-the-art benchmarks, framework design, application domains, communication protocols, and a forward-looking discussion of challenges and open problems, thereby providing researchers with a comprehensive roadmap for advancing LLM-based autonomous AI agents.

### Original URL
https://ar5iv.labs.arxiv.org/html/2504.19678
</details>

---
<details>
<summary>What is a ReAct Agent? | IBM</summary>

# What is a ReAct agent?

A ReAct agent is an AI agent that uses the “reasoning and acting” (ReAct) framework to combine chain of thought (CoT) reasoning with external tool use. The ReAct framework enhances the ability of a large language model (LLM) to handle complex tasks and decision-making in agentic workflows.

First introduced by Yao and others in the 2023 paper, “ReACT: Synergizing Reasoning and Acting in Language Models,” ReAct can be understood most generally as a machine learning (ML) paradigm to integrate the reasoning and action-taking capabilities of LLMs.

More specifically, ReAct is a conceptual framework for building AI agents that can interact with their environment in a structured but adaptable way, by using an LLM as the agent’s “brain” to coordinate anything from simple retrieval augmented generation (RAG) to intricate multiagent workflows.

Unlike traditional artificial intelligence (AI) systems, ReAct agents don’t separate decision-making from task execution. Therefore, the development of the ReAct paradigm was an important step in the evolution of generative AI (gen AI) beyond mere conversational chatbots and toward complex problem-solving.

ReAct agents and derivative approaches continue to power AI applications that can autonomously plan, execute and adapt to unforeseen circumstances.

## How do ReAct agents work?

The ReAct framework is inspired by the way humans can intuitively use natural language—often through our own inner monologue—in the step-by-step planning and execution of complex tasks.

Rather than implementing rule-based or otherwise predefined workflows, ReAct agents rely on their LLM’s reasoning capabilities to dynamically adjust their approach based on new information or the results of previous steps.

Imagine packing for a brief trip. You might start by identifying key considerations (“ _What will the weather be like while I’m there?_”), then actively consult external sources (“ _I’ll check the local weather forecast_”).

By using that new information (“ _It’s going to be cold_”), you determine your next consideration (“ _What warm clothes do I have?_”) and action (“ _I’ll check my closet_”). Upon taking that action, you might encounter an unexpected obstacle (“ _All of my warm clothes are in storage_”) and adjust your next step accordingly (“ _What clothes can I layer together?_”).

In a similar fashion, the ReAct framework uses prompt engineering to structure an AI agent’s activity in a formal pattern of alternating thoughts, actions and observations:

- The verbalized CoT reasoning steps ( _thoughts_) help the model decompose the larger task into more manageable subtasks.
- Predefined _actions_ enable the model to use tools, make application programming interface (API) calls and gather more information from external sources (such as search engines) or knowledge bases (such as an internal docstore).
- After taking an action, the model then reevaluates its progress and uses that _observation_ to either deliver a final answer or inform the next _thought_. The observation might ideally also consider prior information, whether from earlier in the model’s standard context window or from an external memory component.

Because the performance of a ReAct agent depends heavily on the ability of its central LLM to “verbally” think its way through complex tasks, ReAct agents benefit greatly from highly capable models with advanced reasoning and instruction-following ability.

To minimize cost and latency, a multiagent ReAct framework might rely primarily on a larger, more performant model to serve as the central agent whose reasoning process or actions might involve delegating subtasks to more agents built using smaller, more efficient models.

### ReAct agent loops

This framework inherently creates a feedback loop in which the model problem-solves by iteratively repeating this interleaved _thought-action-observation_ process.

Each time this loop is completed—that is, each time the agent has taken an action and made an observation based on the results of that action—the agent must then decide whether to repeat or end the loop.

When and how to end the reasoning loop is an important consideration in the design of a ReAct agent. Establishing a maximum number of loop iterations is a simple way to limit latency, costs and token usage, and avoid the possibility of an endless loop.

Conversely, the loop can be set to end when some specific condition is met, such as when the model has identified a potential final answer that exceeds a certain confidence threshold.

To implement this kind of reasoning and acting loop, ReAct agents typically use some variant of _ReAct prompting_, whether in the system prompt provided to the LLM or in the context of the user query itself.

## ReAct prompting

ReAct prompting is a specific prompting technique designed to guide an LLM to follow the ReAct paradigm of _thought_, _action_ and _observation_ loops. While the explicit use of conventional ReAct prompting methods is not strictly necessary to build a ReAct agent, most ReAct-based agents implement or at least take direct inspiration from it.

First outlined in the original ReAct paper, ReAct prompting’s primary function is to instruct an LLM to follow the ReAct loop and establish which tools can be used—that is, which actions can be taken—when handling user queries.

Whether through explicit instructions or the inclusion of few-shot examples, ReAct prompting should:

- **Guide the model to use chain of thought reasoning:** Prompt the model to reason its way through tasks by thinking step by step, interleaving thoughts with actions.
- **Define actions:** Establish the specific actions available to the model. An action might entail the generation of a specific type of next thought or subprompt but usually involves using external tools or making APIs.
- **Instruct the model to make observations:** Prompt the model to reassess its context after each action step and use that updated context to inform the next reasoning step.
- **Loop:** Instruct the model to repeat the previous steps if necessary. You could provide specific conditions for ending that loop, such as a maximum number of loops, or instruct the agent to end its reasoning process whenever it feels it has arrived at the correct final output.
- **Output final answer:** Whenever those end conditions have been met, provide the user with the final output in response to their initial query. As with many uses of LLMs, as reasoning models employing chain of thought reasoning before determining a final output, ReAct agents are often prompted to conduct their reasoning process within a “scratchpad.”

A classic demonstration of ReAct prompting is the system prompt for the prebuiltZERO\_SHOT\_REACT-DESCRIPTION ReAct agent module in Langchain’s LangGraph. It’s called “zero-shot” because, with this predefined system prompt, the LLM being used with the module does not need any further examples to behave as a ReAct agent.

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

The introduction of the ReAct framework was an important step in the advancement of LLM-driven agentic workflows. From grounding LLMs in real time, real-world external information through (RAG) to contributing to subsequent breakthroughs—such as Reflexion, which led to modern reasoning models—ReAct has helped catalyze the use of LLMs for tasks well beyond text generation.

The utility of ReAct agents is drawn largely from some of the inherent qualities of the ReAct framework:

- **Versatility:** ReAct agents can be configured to work with a wide variety of external tools and APIs. Though fine-tuning relevant ReAct prompts (using relevant tools) can improve performance, no prior configuration of the model is required to execute tool calls.
- **Adaptability:** This versatility, along with the dynamic and situational nature of how they determine the appropriate tool or API to call, means that ReAct agents can use their reasoning process to adapt to new challenges. Especially when operating within a lengthy context window or augmented with external memory, they can learn from past mistakes and successes to tackle unforeseen obstacles and situations. This makes ReAct agents flexible and resilient.
- **Explainability:** The verbalized reasoning process of a ReAct agent is simple to follow, which facilitates debugging and helps make them relatively user-friendly to build and optimize.
- **Accuracy:** As the original ReAct paper asserts, chain of thought (CoT) reasoning alone has many benefits for LLMs, but also runs an increased risk of hallucination. ReAct’s combination of CoT with a connection external to information sources significantly reduces hallucinations, making ReAct agents more accurate and trustworthy.

## ReAct agents vs. function calling

Another prominent paradigm for agentic AI is function calling, originally introduced by OpenAI in June 2023 to supplement the agentic abilities of its GPT models.

The function calling paradigm entails fine-tuning models to recognize when a particular situation should result in a tool call and output a structured JSON object containing the arguments necessary to call those functions.

Many proprietary and open source LLM families, including IBM® Granite®, Meta’s Llama series, Anthropic’s Claude and Google Gemini, now support function calling.

Whether ReAct or function calling is “better” will generally depend on the nature of your specific use case. In scenarios involving relatively straightforward (or at least predictable) tasks, function calling can execute faster, save tokens, and be simpler to implement than a ReAct agent.

In such circumstances, the number of tokens that would be spent on a ReAct agent’s iterative loop of CoT reasoning might be seen as inefficient.

The inherent tradeoff is a relative lack of ability to customize how and when the model chooses which tool to use. Likewise, when an agent handles tasks that call for complex reasoning, or scenarios that are dynamic or unpredictable, the rigidity of function calling might limit the agent’s adaptability. In such situations, it’s often beneficial to be able to view the step-by-step reasoning that led to a specific tool call.

## Getting started with ReAct agents

ReAct agents can be designed and implemented in multiple ways, whether coded from scratch in Python or developed with the help of open source frameworks such as BeeAI. The popularity and staying power of the ReAct paradigm have yielded extensive literature and tutorials for ReAct agents on GitHub and other developer communities.

As an alternative to developing custom ReAct agents, many agentic AI frameworks, including BeeAI, LlamaIndex and LangChain’s LangGraph, offer preconfigured ReAct agent modules for specific use cases.

### Original URL
https://www.ibm.com/think/topics/react-agent
</details>

---
<details>
<summary>What is AI Agent Planning? | IBM</summary>

# What is AI agent planning?

## What is AI agent planning?

AI agent planning refers to the process by which an artificial intelligence (AI) agent determines a sequence of actions to achieve a specific goal. It involves decision-making, goal prioritization and action sequencing, often using various planning algorithms and frameworks.

AI agent planning is a module common to many types of agents that exists alongside other modules such as perception, reasoning, decision-making, action, memory, communication and learning. Planning works in conjunction with these other modules to help ensure that agents achieve outcomes desired by their designers.

Not all agents can plan. Unlike simple reactive agents that respond immediately to inputs, planning agents anticipate future states and generate a structured action plan before execution. This makes AI planning essential for automation tasks that require multistep decision-making, optimization and adaptability.

## How AI agent planning works

Advances in large language models (LLMs) such as OpenAI’s GPT and related techniques involving machine learning algorithms resulted in the generative AI (gen AI) boom of recent years, and further advancements have led to the emerging field of autonomous agents.

By integrating tools, APIs, hardware interfaces and other external resources, agentic AI systems are increasingly autonomous, capable of real-time decision-making and adept at problem-solving across various use cases.

Complex agents can’t act without making a decision, and they can’t make good decisions without first making a plan. Agentic planning consists of several key components that work together to encourage optimal decision-making.

### Goal definition

The first and most critical step in AI planning is defining a clear objective. The goal serves as the guiding principle for the agent’s decision-making process, determining the end state it seeks to achieve. Goals can either be static, remaining unchanged throughout the planning process, or dynamic, adjusting based on environmental conditions or user interactions.

For instance, a self-driving car might have a goal of reaching a specific destination efficiently while adhering to safety regulations. Without a well-defined goal, an agent would lack direction, leading to erratic or inefficient behavior.

If the goal is complex, agentic AI models will break it down into smaller, more manageable sub-goals in a process called task decomposition. This allows the system to focus on complex tasks in a hierarchical manner.

LLMs play a vital role in task decomposition, breaking down a high-level goal into smaller subtasks and then executing those subtasks through various steps. For instance, a user might ask a chatbot with a natural language prompt to plan a trip.

The agent would first decompose the task into components such as booking flights, finding hotels and planning an itinerary. Once decomposed, the agent can use application programming interfaces (APIs) to fetch real-time data, check pricing and even suggest destinations.

### State representation

To plan effectively, an agent must have a structured understanding of its environment. This understanding is achieved through state representation, which models the current conditions, constraints and contextual factors that influence decision-making.

Agents have some built-in knowledge from their training data or datasets representing previous interactions, but perception is required for agents to have a real-time understanding of their environment. Agents collect data through sensory input, allowing it to model its environment, along with user input and data describing its own internal state.

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

Single agent planning is one thing, but in a multiagent system, AI agents must work autonomously while interacting with each other to achieve individual or collective goals.

The planning process for AI agents in a multiagent system is more complex than for a single agent because agents must not only plan their own actions but also consider the actions of other agents and how their decisions interact with those of others.

Depending on the agentic architecture, each agent in the system typically has its own individual goals, which might involve accomplishing specific tasks or maximizing a reward function. In many multiagent systems, agents need to work together to achieve shared goals.

These goals could be defined by an overarching system or emerge from the agents’ interactions. Agents need mechanisms to communicate and align their goals, especially in cooperative scenarios. This could be done through explicit messaging, shared task definitions or implicit coordination.

Planning in multiagent systems can be centralized, where a single entity or controller—likely an LLM agent—generates the plan for the entire system.

Each agent receives instructions or plans from this central authority. It can also be decentralized, where agents generate their own plans but work collaboratively to help ensure that they align with each other and contribute to global objectives, often requiring communication and negotiation.

This collaborative decision-making process enhances efficiency, reduces biases in task execution, helps to avoid hallucinations through cross-validation and consensus-building and encourages the agents to work toward a common goal.

## After planning

The phases in agentic AI workflows do not always occur in a strict step-by-step linear fashion. While these phases are often distinct in conceptualization, in practice, they are frequently interleaved or iterative, depending on the nature of the task and the complexity of the environment in which the agent operates.

AI solutions can differ depending on their design, but in a typical agentic workflow, the next phase after planning is action execution, where the agent carries out the actions defined in the plan. This involves performing tasks and interacting with external systems or knowledge bases with retrieval augmented generation (RAG), tool use and function calling (tool calling).

Building AI agents for these capabilities might involve LangChain. Python scripts, JSON data structures and other programmatic tools enhance the AI’s ability to make decisions.

After executing plans, some agents can use memory to learn from their experiences and iterate their behavior accordingly.

In dynamic environments, the planning process must be adaptive. Agents continuously receive feedback about the environment and other agents’ actions and must adjust their plans accordingly. This might involve revising goals, adjusting action sequences, or adapting to new agents entering or leaving the system.

When an agent detects that its current plan is no longer feasible (for example, due to a conflict with another agent or a change in the environment), it might engage in replanning to adjust its strategy. Agents can adjust their strategies using chain of thought reasoning, a process where they reflect on the steps needed to reach their objective before taking action.

### Original URL
https://www.ibm.com/think/topics/ai-agent-planning
</details>

---
<details>
<summary>Building ReAct Agents from Scratch: A Hands-On Guide using Gemini | by Arun Shankar | Google Cloud - Community | Medium</summary>

Sitemap @https://medium.com/sitemap/sitemap.xml

Open in app @https://rsci.app.link/?%24canonical_url=https%3A%2F%2Fmedium.com%2Fp%2Fffe4621d90ae&%7Efeature=LoOpenInAppButton&%7Echannel=ShowPostUnderCollection&%7Estage=mobileNavBar&source=post_page---top_nav_layout_nav-----------------------------------------

Sign up

Sign in @https://medium.com/m/signin?operation=login&redirect=https%3A%2F%2Fmedium.com%2Fgoogle-cloud%2Fbuilding-react-agents-from-scratch-a-hands-on-guide-using-gemini-ffe4621d90ae&source=post_page---top_nav_layout_nav-----------------------global_nav------------------

Medium Logo @https://medium.com/?source=post_page---top_nav_layout_nav-----------------------------------------

Write @https://medium.com/m/signin?operation=register&redirect=https%3A%2F%2Fmedium.com%2Fnew-story&source=---top_nav_layout_nav-----------------------new_post_topnav------------------

Sign up

Sign in @https://medium.com/m/signin?operation=login&redirect=https%3A%2F%2Fmedium.com%2Fgoogle-cloud%2Fbuilding-react-agents-from-scratch-a-hands-on-guide-using-gemini-ffe4621d90ae&source=post_page---top_nav_layout_nav-----------------------global_nav------------------

!@https://miro.medium.com/v2/resize:fill:32:32/1*dmbNkD5D-u45r44go_cf0g.png

**Google Cloud - Community** @https://medium.com/google-cloud?source=post_page---publication_nav-e52cf94d98af-ffe4621d90ae---------------------------------------

·

Follow publication @https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2F_%2Fsubscribe%2Fcollection%2Fgoogle-cloud&operation=register&redirect=https%3A%2F%2Fmedium.com%2Fgoogle-cloud%2Fbuilding-react-agents-from-scratch-a-hands-on-guide-using-gemini-ffe4621d90ae&collection=Google+Cloud+-+Community&collectionId=e52cf94d98af&source=post_page---publication_nav-e52cf94d98af-ffe4621d90ae---------------------publication_nav------------------

![Google Cloud - Community @https://miro.medium.com/v2/resize:fill:38:38/1*FUjLiCANvATKeaJEeg20Rw.png](https://medium.com/google-cloud?source=post_page---post_publication_sidebar-e52cf94d98af-ffe4621d90ae---------------------------------------)

A collection of technical articles and blogs published or curated by Google Cloud Developer Advocates. The views expressed are those of the authors and don't necessarily reflect those of Google.

Follow publication @https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2F_%2Fsubscribe%2Fcollection%2Fgoogle-cloud&operation=register&redirect=https%3A%2F%2Fmedium.com%2Fgoogle-cloud%2Fbuilding-react-agents-from-scratch-a-hands-on-guide-using-gemini-ffe4621d90ae&collection=Google+Cloud+-+Community&collectionId=e52cf94d98af&source=post_page---post_publication_sidebar-e52cf94d98af-ffe4621d90ae---------------------post_publication_sidebar------------------

# Building ReAct Agents from Scratch: A Hands-On Guide using Gemini

![Arun Shankar @https://miro.medium.com/v2/resize:fill:32:32/1*vEPGpdaRi4Ms10NymQypiw.png](https://medium.com/@shankar.arunp?source=post_page---byline--ffe4621d90ae---------------------------------------)

Arun Shankar @https://medium.com/@shankar.arunp?source=post_page---byline--ffe4621d90ae---------------------------------------

Follow

32 min read

·

Oct 11, 2024

602

12

Listen @https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2Fplans%3Fdimension%3Dpost_audio_button%26postId%3Dffe4621d90ae&operation=register&redirect=https%3A%2F%2Fmedium.com%2Fgoogle-cloud%2Fbuilding-react-agents-from-scratch-a-hands-on-guide-using-gemini-ffe4621d90ae&source=---header_actions--ffe4621d90ae---------------------post_audio_button------------------

Share

`tldr` **:** _ReAct (Reason + Act) is a powerful framework for building AI agents that seamlessly integrates reasoning and decision-making with task execution. By leveraging large language models (LLMs) like Gemini, ReAct agents can dynamically analyze problems, choose appropriate tools, and iteratively work towards solutions. This article explores the foundations of ReAct, provides a step-by-step guide to building a ReAct agent from scratch, and discusses its implications for the future of generative AI. It includes a_ _GitHub repo_ @https://github.com/arunpshankar/react-from-scratch _with all the code and artifacts needed to get started quickly. Even though the post uses Gemini, the underlying LLM can easily be replaced with any foundational models._

!@https://miro.medium.com/v2/resize:fit:700/1*o307XLpR20oxEXi6iZEe7g.jpeg

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

Now that we’ve established the foundations, let’s explore the process of building a ReAct agent using Gemini as our LLM of choice.

!@https://miro.medium.com/v2/resize:fit:1000/1*TQspcqCDuqzbv5bCiIG26w.png

ReAct Agent — A 1000 foot view

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

The core of the ReAct agent is its ability to think, act, and observe in an iterative loop. A high-level flow diagram below illustrates how a ReAct agent functions in this loop. The core pieces are the thinking (reasoning) phase, the acting phase (calling APIs and accessing the environment through tool use), and finally, the observation phase (collecting results). This cycle repeats, allowing the agent to improve and move towards a common goal set initially. In the following sections, we’ll examine each component of this loop in detail.

!@https://miro.medium.com/v2/resize:fit:700/1*O8k5Oy65KYhkNFQ13PRpUA.png

Think-Act-Observe loop

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

You can find the prompt template in the shared repo here @https://github.com/arunpshankar/react-from-scratch/blob/main/data/input/react.txt.

# II. Comparing Approaches: Traditional vs. ReAct Agents

!@https://miro.medium.com/v2/resize:fit:700/1*8xI6wr1Pj32h5ylDU5Tr1w.jpeg

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

This trace demonstrates several key aspects of the ReAct framework: the agent uses iterative reasoning by breaking down the problem into steps, gathering necessary information before making a final decision; it utilizes tools effectively, such as the Wikipedia tool, to gather relevant data; it exhibits adaptive behavior by adjusting its actions based on the information it has and what it still needs; and its decision-making process is transparent, with each thought clearly articulated, showing how the agent arrives at its decisions and final answer. By examining this trace, we see how the ReAct agent combines reasoning and action to solve queries in a structured, transparent, and effective manner. You can find the complete trace of the agent here @https://github.com/arunpshankar/react-from-scratch/blob/main/data/output/trace_1.txt.

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

The complete agent trace can be accessed at this location @https://github.com/arunpshankar/react-from-scratch/blob/main/data/output/trace_2.txt.

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

You can find the full agent trace here @https://github.com/arunpshankar/react-from-scratch/blob/main/data/output/trace_3.txt. All code files associated with everything we covered here are available in this GitHub repository @https://github.com/arunpshankar/react-from-scratch/tree/main.

# Future Directions

The ReAct framework opens up numerous possibilities for enhanced functionality and adaptability in agent-based systems. Future developments could focus on integrating the ability to process diverse data types such as images, audio, and video, enabling agents to interpret a broader spectrum of information for richer, context-aware decisions. Organizing agents into layered hierarchies, where primary agents delegate specialized tasks to sub-agents, would improve efficiency and task segmentation. Additionally, empowering agents to collaborate by sharing observations, tools, and resources would amplify insights and support cohesive decision-making in complex, multi-perspective environments. Higher-level agents can dynamically guide and coordinate other agents, orchestrating actions across a multi-agent setup to handle complex, multi-step tasks efficiently.

In future posts, we’ll explore a multi-agent scenario where a single steering agent interacts with multiple sub-agents, each fulfilling distinct tasks and communicating their findings back for a cohesive outcome. This will extend our current exercise, building a foundation for scalable, collaborative agent networks.

Thanks for reading the article and for your engagement. Your follow and claps mean a lot. If you have any questions or doubts about the content or the shared notebooks, feel free to contact me at _arunpshankar@google.com_ @mailto:arunprsh@amazon.com or _shankar.arunp@gmail.com_ @mailto:shankar.arunp@gmail.com. You can also find me on @https://www.linkedin.com/in/arunprasath-shankar/

I welcome all feedback and suggestions. If you’re interested in large-scale ML, NLP or NLU, and eager to collaborate, I’d be delighted to connect. Moreover, if you’re an individual, startup, or enterprise seeking to comprehend Google Cloud, VertexAI, and the various Generative AI components and their applications in NLP/ML, I’m here to assist. Please feel free to reach out.

Google Cloud Platform @https://medium.com/tag/google-cloud-platform?source=post_page-----ffe4621d90ae---------------------------------------

Large Language Models @https://medium.com/tag/large-language-models?source=post_page-----ffe4621d90ae---------------------------------------

Artificial Intelligence @https://medium.com/tag/artificial-intelligence?source=post_page-----ffe4621d90ae---------------------------------------

Machine Learning @https://medium.com/tag/machine-learning?source=post_page-----ffe4621d90ae---------------------------------------

NLP @https://medium.com/tag/nlp?source=post_page-----ffe4621d90ae---------------------------------------

![Google Cloud - Community @https://miro.medium.com/v2/resize:fill:48:48/1*FUjLiCANvATKeaJEeg20Rw.png](https://medium.com/google-cloud?source=post_page---post_publication_info--ffe4621d90ae---------------------------------------)

![Google Cloud - Community @https://miro.medium.com/v2/resize:fill:64:64/1*FUjLiCANvATKeaJEeg20Rw.png](https://medium.com/google-cloud?source=post_page---post_publication_info--ffe4621d90ae---------------------------------------)

Follow

**Published in Google Cloud - Community** @https://medium.com/google-cloud?source=post_page---post_publication_info--ffe4621d90ae---------------------------------------

65K followers @https://medium.com/google-cloud/followers?source=post_page---post_publication_info--ffe4621d90ae---------------------------------------

· Last published just now @https://medium.com/google-cloud/6-ways-to-become-a-database-pro-with-the-gemini-cli-793c94a5859b?source=post_page---post_publication_info--ffe4621d90ae---------------------------------------

A collection of technical articles and blogs published or curated by Google Cloud Developer Advocates. The views expressed are those of the authors and don't necessarily reflect those of Google.

Follow

![Arun Shankar @https://miro.medium.com/v2/resize:fill:48:48/1*vEPGpdaRi4Ms10NymQypiw.png](https://medium.com/@shankar.arunp?source=post_page---post_author_info--ffe4621d90ae---------------------------------------)

![Arun Shankar @https://miro.medium.com/v2/resize:fill:64:64/1*vEPGpdaRi4Ms10NymQypiw.png](https://medium.com/@shankar.arunp?source=post_page---post_author_info--ffe4621d90ae---------------------------------------)

Follow

**Written by Arun Shankar** @https://medium.com/@shankar.arunp?source=post_page---post_author_info--ffe4621d90ae---------------------------------------

1.4K followers @https://medium.com/@shankar.arunp/followers?source=post_page---post_author_info--ffe4621d90ae---------------------------------------

· 230 following @https://medium.com/@shankar.arunp/following?source=post_page---post_author_info--ffe4621d90ae---------------------------------------

Global Lead Architect at Google. Ex-AWS. 13+ yrs exp in NLP/NLU, seasoned speaker & author, passionate about tackling intricate NLU challenges at scale.

Follow

## Responses (12)

!@https://miro.medium.com/v2/resize:fill:32:32/1*dmbNkD5D-u45r44go_cf0g.png

Write a response

What are your thoughts? @https://medium.com/m/signin?operation=register&redirect=https%3A%2F%2Fmedium.com%2Fgoogle-cloud%2Fbuilding-react-agents-from-scratch-a-hands-on-guide-using-gemini-ffe4621d90ae&source=---post_responses--ffe4621d90ae---------------------respond_sidebar------------------

Cancel

Respond

![Sm @https://miro.medium.com/v2/resize:fill:32:32/0*o6FULdV50DMF0xFi](https://medium.com/@sm_90584?source=post_page---post_responses--ffe4621d90ae----0-----------------------------------)

Sm @https://medium.com/@sm_90584?source=post_page---post_responses--ffe4621d90ae----0-----------------------------------

Oct 21, 2024 @https://medium.com/@sm_90584/excellent-in-writing-excellent-in-content-60cc2c701cbf?source=post_page---post_responses--ffe4621d90ae----0-----------------------------------

```

Excellent in writing, excellent in content. Amazing performance shown: I am really stunned after reading the food ingredient example. Very inspiring.
```

9

1 reply

Reply

![Krishan Walia @https://miro.medium.com/v2/resize:fill:32:32/1*wd2Rq2ky4uzHKl-BXtE6KQ.jpeg](https://medium.com/@krishanw30?source=post_page---post_responses--ffe4621d90ae----1-----------------------------------)

Krishan Walia\\
\\
he @https://medium.com/@krishanw30?source=post_page---post_responses--ffe4621d90ae----1-----------------------------------

Oct 15, 2024 @https://medium.com/@krishanw30/an-impeccable-resource-with-complete-information-from-theory-to-practical-9c81c73acd3b?source=post_page---post_responses--ffe4621d90ae----1-----------------------------------

```

An impeccable resource, with complete information from theory to practical.
Thank you for sharing this.

Do you have any references to real world application of this technology?
```

12

1 reply

Reply

![Anatole Martins @https://miro.medium.com/v2/resize:fill:32:32/1*1zCB32lXcQHi4i0VncopWg.png](https://medium.com/@anatole.martins6730?source=post_page---post_responses--ffe4621d90ae----2-----------------------------------)

Anatole Martins @https://medium.com/@anatole.martins6730?source=post_page---post_responses--ffe4621d90ae----2-----------------------------------

Oct 12, 2024 @https://medium.com/@anatole.martins6730/this-hands-on-guide-is-a-brilliant-resource-1efd28f85c8a?source=post_page---post_responses--ffe4621d90ae----2-----------------------------------

```

This hands-on guide is a brilliant resource! How do you envision ReAct agents evolving in real-world applications?
```

5

1 reply

Reply

See all responses

## More from Arun Shankar and Google Cloud - Community

!Architectural Patterns for Text-to-SQL: Leveraging LLMs for Enhanced BigQuery Interactions @https://miro.medium.com/v2/resize:fit:679/0*J4kixp-RV8WGiJBC

![Google Cloud - Community @https://miro.medium.com/v2/resize:fill:20:20/1*FUjLiCANvATKeaJEeg20Rw.png](https://medium.com/google-cloud?source=post_page---author_recirc--ffe4621d90ae----0---------------------12bdf31f_e930_437f_82ad_805d3ebb4ff1--------------)

In

Google Cloud - Community @https://medium.com/google-cloud?source=post_page---author_recirc--ffe4621d90ae----0---------------------12bdf31f_e930_437f_82ad_805d3ebb4ff1--------------

by

Arun Shankar @https://medium.com/@shankar.arunp?source=post_page---author_recirc--ffe4621d90ae----0---------------------12bdf31f_e930_437f_82ad_805d3ebb4ff1--------------

**Architectural Patterns for Text-to-SQL: Leveraging LLMs for Enhanced BigQuery Interactions** @https://medium.com/google-cloud/architectural-patterns-for-text-to-sql-leveraging-llms-for-enhanced-bigquery-interactions-59756a749e15?source=post_page---author_recirc--ffe4621d90ae----0---------------------12bdf31f_e930_437f_82ad_805d3ebb4ff1--------------

Nov 11, 2023

A clap icon956\\
\\
A response icon12 @https://medium.com/google-cloud/architectural-patterns-for-text-to-sql-leveraging-llms-for-enhanced-bigquery-interactions-59756a749e15?source=post_page---author_recirc--ffe4621d90ae----0---------------------12bdf31f_e930_437f_82ad_805d3ebb4ff1--------------

!Model Context Protocol(MCP) with Google Gemini LLM — A Deep Dive (Full Code) @https://miro.medium.com/v2/resize:fit:679/1*xXrGQq3-fXdRTv75Z0hkHg.png

![Google Cloud - Community @https://miro.medium.com/v2/resize:fill:20:20/1*FUjLiCANvATKeaJEeg20Rw.png](https://medium.com/google-cloud?source=post_page---author_recirc--ffe4621d90ae----1---------------------12bdf31f_e930_437f_82ad_805d3ebb4ff1--------------)

In

Google Cloud - Community @https://medium.com/google-cloud?source=post_page---author_recirc--ffe4621d90ae----1---------------------12bdf31f_e930_437f_82ad_805d3ebb4ff1--------------

by

Arjun Prabhulal @https://medium.com/@arjun-prabhulal?source=post_page---author_recirc--ffe4621d90ae----1---------------------12bdf31f_e930_437f_82ad_805d3ebb4ff1--------------

**Model Context Protocol(MCP) with Google Gemini LLM — A Deep Dive (Full Code)** @https://medium.com/google-cloud/model-context-protocol-mcp-with-google-gemini-llm-a-deep-dive-full-code-ea16e3fac9a3?source=post_page---author_recirc--ffe4621d90ae----1---------------------12bdf31f_e930_437f_82ad_805d3ebb4ff1--------------

Apr 4

A clap icon1.1K\\
\\
A response icon17 @https://medium.com/google-cloud/model-context-protocol-mcp-with-google-gemini-llm-a-deep-dive-full-code-ea16e3fac9a3?source=post_page---author_recirc--ffe4621d90ae----1---------------------12bdf31f_e930_437f_82ad_805d3ebb4ff1--------------

!Your First Multi-agent system: A Beginner’s Guide to Building an AI Trend finder with ADK @https://miro.medium.com/v2/resize:fit:679/1*_uqJ2-BOxiVJ2cPrGLmgig.png

![Google Cloud - Community @https://miro.medium.com/v2/resize:fill:20:20/1*FUjLiCANvATKeaJEeg20Rw.png](https://medium.com/google-cloud?source=post_page---author_recirc--ffe4621d90ae----2---------------------12bdf31f_e930_437f_82ad_805d3ebb4ff1--------------)

In

Google Cloud - Community @https://medium.com/google-cloud?source=post_page---author_recirc--ffe4621d90ae----2---------------------12bdf31f_e930_437f_82ad_805d3ebb4ff1--------------

by

Shir Meir Lador @https://medium.com/@DataLady?source=post_page---author_recirc--ffe4621d90ae----2---------------------12bdf31f_e930_437f_82ad_805d3ebb4ff1--------------

**Your First Multi-agent system: A Beginner’s Guide to Building an AI Trend finder with ADK** @https://medium.com/google-cloud/your-first-multi-agent-system-a-beginners-guide-to-building-an-ai-trend-finder-with-adk-6991cf587f22?source=post_page---author_recirc--ffe4621d90ae----2---------------------12bdf31f_e930_437f_82ad_805d3ebb4ff1--------------

Jun 10

A clap icon315\\
\\
A response icon4 @https://medium.com/google-cloud/your-first-multi-agent-system-a-beginners-guide-to-building-an-ai-trend-finder-with-adk-6991cf587f22?source=post_page---author_recirc--ffe4621d90ae----2---------------------12bdf31f_e930_437f_82ad_805d3ebb4ff1--------------

!Designing Cognitive Architectures: Agentic Workflow Patterns from Scratch @https://miro.medium.com/v2/resize:fit:679/1*oQEioKuhaMsVQwvB1pZWhQ.png

![Google Cloud - Community @https://miro.medium.com/v2/resize:fill:20:20/1*FUjLiCANvATKeaJEeg20Rw.png](https://medium.com/google-cloud?source=post_page---author_recirc--ffe4621d90ae----3---------------------12bdf31f_e930_437f_82ad_805d3ebb4ff1--------------)

In

Google Cloud - Community @https://medium.com/google-cloud?source=post_page---author_recirc--ffe4621d90ae----3---------------------12bdf31f_e930_437f_82ad_805d3ebb4ff1--------------

by

Arun Shankar @https://medium.com/@shankar.arunp?source=post_page---author_recirc--ffe4621d90ae----3---------------------12bdf31f_e930_437f_82ad_805d3ebb4ff1--------------

**Designing Cognitive Architectures: Agentic Workflow Patterns from Scratch** @https://medium.com/google-cloud/designing-cognitive-architectures-agentic-workflow-patterns-from-scratch-63baa74c54bc?source=post_page---author_recirc--ffe4621d90ae----3---------------------12bdf31f_e930_437f_82ad_805d3ebb4ff1--------------

Oct 24, 2024

A clap icon1.2K\\
\\
A response icon5 @https://medium.com/google-cloud/designing-cognitive-architectures-agentic-workflow-patterns-from-scratch-63baa74c54bc?source=post_page---author_recirc--ffe4621d90ae----3---------------------12bdf31f_e930_437f_82ad_805d3ebb4ff1--------------

See all from Arun Shankar @https://medium.com/@shankar.arunp?source=post_page---author_recirc--ffe4621d90ae---------------------------------------

See all from Google Cloud - Community @https://medium.com/google-cloud?source=post_page---author_recirc--ffe4621d90ae---------------------------------------

## Recommended from Medium

!ReAct Agent Using LangGraph and Gemini @https://miro.medium.com/v2/resize:fit:679/0*7ac8I39iDTeCiUlY

![Fundamentals of Artificial Intelligence @https://miro.medium.com/v2/resize:fill:20:20/1*kkxEzrm3Z2uBLwmLemLAWA.png](https://medium.com/fundamentals-of-artificial-intellegence?source=post_page---read_next_recirc--ffe4621d90ae----0---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------)

In

Fundamentals of Artificial Intelligence @https://medium.com/fundamentals-of-artificial-intellegence?source=post_page---read_next_recirc--ffe4621d90ae----0---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------

by

Arts2Survive @https://medium.com/@pankajchandravanshi?source=post_page---read_next_recirc--ffe4621d90ae----0---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------

**ReAct Agent Using LangGraph and Gemini** @https://medium.com/fundamentals-of-artificial-intellegence/building-a-react-agent-using-langgraph-and-gemini-a1e4b3406e13?source=post_page---read_next_recirc--ffe4621d90ae----0---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------

Jun 17

!Building a Multi-Agent AI System with LangGraph and LangSmith @https://miro.medium.com/v2/resize:fit:679/1*BX5lOdJVWoW4CfS6F6YEAA.png

![Level Up Coding @https://miro.medium.com/v2/resize:fill:20:20/1*5D9oYBd58pyjMkV_5-zXXQ.jpeg](https://medium.com/gitconnected?source=post_page---read_next_recirc--ffe4621d90ae----1---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------)

In

Level Up Coding @https://medium.com/gitconnected?source=post_page---read_next_recirc--ffe4621d90ae----1---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------

by

Fareed Khan @https://medium.com/@fareedkhandev?source=post_page---read_next_recirc--ffe4621d90ae----1---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------

**Building a Multi-Agent AI System with LangGraph and LangSmith** @https://medium.com/gitconnected/building-a-multi-agent-ai-system-with-langgraph-and-langsmith-6cb70487cd81?source=post_page---read_next_recirc--ffe4621d90ae----1---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------

Jun 1

A clap icon1.2K\\
\\
A response icon20 @https://medium.com/gitconnected/building-a-multi-agent-ai-system-with-langgraph-and-langsmith-6cb70487cd81?source=post_page---read_next_recirc--ffe4621d90ae----1---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------

!Upload a CSV, Get a Dashboard — Building the AI Behind It (Part 1) @https://miro.medium.com/v2/resize:fit:679/1*bmxwgpdNB_bkoM45mOrt3g.png

![Artificial Intelligence in Plain English @https://miro.medium.com/v2/resize:fill:20:20/1*9zAmnK08gUCmZX7q0McVKw@2x.png](https://medium.com/ai-in-plain-english?source=post_page---read_next_recirc--ffe4621d90ae----0---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------)

In

Artificial Intelligence in Plain English @https://medium.com/ai-in-plain-english?source=post_page---read_next_recirc--ffe4621d90ae----0---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------

by

Vivek Singh Pathania @https://medium.com/@viveksinghpathania?source=post_page---read_next_recirc--ffe4621d90ae----0---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------

**Upload a CSV, Get a Dashboard — Building the AI Behind It (Part 1)** @https://medium.com/ai-in-plain-english/upload-a-csv-get-a-dashboard-building-the-ai-behind-it-part-1-02a3ed7e3be7?source=post_page---read_next_recirc--ffe4621d90ae----0---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------

5d ago

A clap icon67\\
\\
A response icon6 @https://medium.com/ai-in-plain-english/upload-a-csv-get-a-dashboard-building-the-ai-behind-it-part-1-02a3ed7e3be7?source=post_page---read_next_recirc--ffe4621d90ae----0---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------

!From Zero to Multi-Agents: A Beginner’s Guide to Google Agent Development Kit (ADK) @https://miro.medium.com/v2/resize:fit:679/0*O_xAobQB1qSxR5Gl

![Dr Sokratis Kartakis @https://miro.medium.com/v2/resize:fill:20:20/1*eEmPAEaBLAdJjYHZszsm5A.png](https://medium.com/@sokratis.kartakis?source=post_page---read_next_recirc--ffe4621d90ae----1---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------)

Dr Sokratis Kartakis @https://medium.com/@sokratis.kartakis?source=post_page---read_next_recirc--ffe4621d90ae----1---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------

**From Zero to Multi-Agents: A Beginner’s Guide to Google Agent Development Kit (ADK)** @https://medium.com/@sokratis.kartakis/from-zero-to-multi-agents-a-beginners-guide-to-google-agent-development-kit-adk-b56e9b5f7861?source=post_page---read_next_recirc--ffe4621d90ae----1---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------

Apr 9

A clap icon661\\
\\
A response icon9 @https://medium.com/@sokratis.kartakis/from-zero-to-multi-agents-a-beginners-guide-to-google-agent-development-kit-adk-b56e9b5f7861?source=post_page---read_next_recirc--ffe4621d90ae----1---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------

!Getting Started with LangGraph Agents — A Basic ReAct Agent @https://miro.medium.com/v2/resize:fit:679/1*6ke334hWd6nvZaImmItckw.jpeg

![Lachlan Chavasse @https://miro.medium.com/v2/resize:fill:20:20/1*Ip-xTAPcMRGwpQWWa77jJw.png](https://medium.com/@lachlan.chavasse?source=post_page---read_next_recirc--ffe4621d90ae----2---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------)

Lachlan Chavasse @https://medium.com/@lachlan.chavasse?source=post_page---read_next_recirc--ffe4621d90ae----2---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------

**Getting Started with LangGraph Agents — A Basic ReAct Agent** @https://medium.com/@lachlan.chavasse/getting-started-with-langgraph-agents-a-basic-react-agent-c61bfb32aa72?source=post_page---read_next_recirc--ffe4621d90ae----2---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------

Jan 25

A clap icon46\\
\\
A response icon1 @https://medium.com/@lachlan.chavasse/getting-started-with-langgraph-agents-a-basic-react-agent-c61bfb32aa72?source=post_page---read_next_recirc--ffe4621d90ae----2---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------

!Logging + LLM + FastAPI @https://miro.medium.com/v2/resize:fit:679/1*wDDQf-uTF1bD2XrvjKXNgg.png

![alejandro @https://miro.medium.com/v2/resize:fill:20:20/1*nyuXdD14AqgV4Dipv0ziGA@2x.jpeg](https://medium.com/@alejandro7899871776?source=post_page---read_next_recirc--ffe4621d90ae----3---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------)

alejandro @https://medium.com/@alejandro7899871776?source=post_page---read_next_recirc--ffe4621d90ae----3---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------

**Logging + LLM + FastAPI** @https://medium.com/@alejandro7899871776/logging-llm-fastapi-69fe88e01a4d?source=post_page---read_next_recirc--ffe4621d90ae----3---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------

Jun 16

A clap icon23 @https://medium.com/@alejandro7899871776/logging-llm-fastapi-69fe88e01a4d?source=post_page---read_next_recirc--ffe4621d90ae----3---------------------0d7ad299_f033_445d_af74_640d3bb15354--------------

See more recommendations @https://medium.com/?source=post_page---read_next_recirc--ffe4621d90ae---------------------------------------

Help @https://help.medium.com/hc/en-us?source=post_page-----ffe4621d90ae---------------------------------------

Status @https://medium.statuspage.io/?source=post_page-----ffe4621d90ae---------------------------------------

About @https://medium.com/about?autoplay=1&source=post_page-----ffe4621d90ae---------------------------------------

Careers @https://medium.com/jobs-at-medium/work-at-medium-959d1a85284e?source=post_page-----ffe4621d90ae---------------------------------------

Press @mailto:pressinquiries@medium.com

Blog @https://blog.medium.com/?source=post_page-----ffe4621d90ae---------------------------------------

Privacy @https://policy.medium.com/medium-privacy-policy-f03bf92035c9?source=post_page-----ffe4621d90ae---------------------------------------

Rules @https://policy.medium.com/medium-rules-30e5502c4eb4?source=post_page-----ffe4621d90ae---------------------------------------

Terms @https://policy.medium.com/medium-terms-of-service-9db0094a1e0f?source=post_page-----ffe4621d90ae---------------------------------------

Text to speech @https://speechify.com/medium?source=post_page-----ffe4621d90ae---------------------------------------

iframe @https://www.google.com/recaptcha/enterprise/anchor?ar=1&k=6Le-uGgpAAAAAPprRaokM8AKthQ9KNGdoxaGUvVp&co=aHR0cHM6Ly9tZWRpdW0uY29tOjQ0Mw..&hl=en&v=h7qt2xUGz2zqKEhSc8DD8baZ&size=invisible&cb=by6z39okk46f

### Original URL
https://medium.com/google-cloud/building-react-agents-from-scratch-a-hands-on-guide-using-gemini-ffe4621d90ae
</details>

---
<details>
<summary>Google AI Studio</summary>



### Original URL
https://aistudio.google.com/
</details>

