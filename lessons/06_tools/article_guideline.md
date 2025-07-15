## Global Context

- **What I'm planning to share:** In this article, we will explore agent tool usage. We want to highlight that through tools, we can enable agents to take actions and interact with the external world. By implementing tool calling from scratch, we will understand how the LLM decides which tool to call, how it generates the correct parameters for the function, and how it calls the function. First, we will implement tool definition and calling from scratch, then show how to do it with a popular API such as Gemini. We'll also explore using structured outputs as tools through Pydantic models. Next, we'll examine scenarios requiring multiple tools and introduce tool chaining, before discussing the limitations that lead to more sophisticated patterns like ReAct. Finally, we'll survey the most essential tool categories such as knowledge/memory access (RAG), web search, and code execution. All our ideas will be supported by code.
- **Why I think it's valuable:** For an AI Engineer, tools are what transform an LLM from a simple text generator into an agent that can manipulate the external world. Tools are one of the key components and foundational blocks of an AI agent. Therefore, it's critical to open the black box and understand how an agent works with these tools.
- **Who the intended audience is:** Aspiring AI Engineers who are learning about LLM tools and agents for the first time.
- **Expected length of the article in words** (where 200-250 words ~= 1 minute of reading time): 2800 words (around 11.5 - 14 minutes reading time)


## Outline

1. Understanding why agents need tools
2. Implementing tool calls from scratch
3. Implementing a small tool calling framework from scratch
4. Implementing production-level tool calls with Gemini
5. Using Pydantic models as tools for on-demand structured outputs
6. The downsides of running tools in a loop
7. Going through popular tools used within the industry


## Section 1: Understanding why agents need tools

- Explain the inherent limitations of LLMs: They are primarily pattern matchers and text generators that operate on input text. They cannot, by themselves, interact with the external world, such as:
    - Access real-time information (e.g., today's weather, latest news).
    - Interact with external databases or APIs.
    - Execute code.
    - Perform precise calculations beyond their training data.
    - Reliably remember information beyond their context window for long periods.
- Introduce "Tools" as the bridge: Functions or capabilities that an agent can execute on the LLM's behalf, based on the LLM's instructions.
- Analogy: The LLM is the brain, while the tools are an LLM's "hands and senses," allowing it to perceive and act in the world beyond its textual interface. Together, they build up the agent.
- Use a representative image from the research to explain at a high level how tools work.
-  **Section length:** 400 words


## Section 2: Implementing tool calls from scratch

- In this section, we want to explain, using code, how LLMs use tools. We want to explain, with code support, how a tool is defined, how their schema looks like, how the LLM discovers what tools are available and how to call them through function schemas, and how it interprets its output. 
- Provide a mermaid diagram illustrating this request-execute-respond flow of calling a tool.
- Detail the key components and flow:
    1. **Defining tools and their schemas:** Explain how developers define tools for the LLM:
        - Define the Python function
        - Define the schema containing the function name, description (crucial for the LLM to understand when to use the tool), and parameters (type, required/optional properties). Often described using JSON. 
        - Explain that this is the industry standard when working with OpenAI, Gemini and other LLM providers.
    2. **Using the LLM to call a tool:**
        - The LLM is prompted with the user's query AND the list of available tool definitions.
        - Based on its training, the LLM *decides* if a tool call is appropriate to fulfill the query. Add a quick note specifying that the LLM is specially tuned through instruction fine-tuning to interpret tool schema inputs and output tool calls.
        - If so, it *selects* the most relevant tool.
        - It then *generates* the arguments for that tool in a structured format (typically JSON). Highlight how we need structured outputs for function calling.
        - Use the provided system prompt from the provided code to support all these ideas and show how we can guide the LLM to use tools.
    3. **Executing the tool:**
        - The LLM outputs the function's name and arguments. Everything necessary to call the tool.
        - The application code executes the actual tool/function.
        - Explain the importance of structured outputs, as we need to format the function call as a structured output to interpret it properly. 
    4. **Interpreting the tool results with an LLM:**
        - The output/result from the tool execution is sent back to the LLM as a new message in the conversation.
        - The LLM then uses this result to formulate a final response to the user or decide on the next step.
- Follow the narrative from the Notebook, where we first define the tools, then we manually define the schemas. Then explain the system prompt and provide some examples on how to use an LLM to call a specific tool. Ultimately, show how to call the tool and interpret it with an LLM.
- Give step-by-step examples from section `## 2. Implementing tool calls from scratch`, of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
-  **Section length:** 600 words (without counting the code or mermaid diagram)

## Section 3: Implementing a small tool calling framework from scratch

- As a natural progression to defining the tool schemas from scratch, we want to show how to quickly implement a small tool framework by building a `@tool` decorator that allows us to automatically compute the schema of each function that we want to use as a tool.
- Specify that this is a more production-ready approach taken in popular frameworks such as LangGraph and MCP, as we respect the Don't Repeat Yourself (DRY) software principle by having a modular central method to compute the tool schema.
- Go over the `tool` decorator implementation and show how to reimplement the logic from `Section 2` using this new approach.
- Give step-by-step examples from section `## 3. Implementing tool calls from scratch using @tool decorators`, of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
-  **Section length:** 400 words (without counting the code or mermaid diagram)


## Section 4: Implementing production-level tool calls with Gemini

- We want to show how to rewrite the from-scratch code using Gemini's native tool calling capabilities rather than prompt engineering. Use the provided code examples to show how we can rewrite the "from scratch code" from above using Gemini's interface.
- Highlight that when using an LLM provider API, such as Gemini, we still have to define our function schemas and handle the tool calling ourselves. This is where frameworks such as LangGraph can help us by abstracting these steps.
- Explain how the tool calling works. More exactly highlight that now instead of defining a huge system prompt where we guide the LLM how to use the tools, we pass the function schemas to the LLM provider which handles the tool schema injection internally. Emphasize that this is much more efficient than manually crafting tool descriptions in prompts.
- Explain that all popular APIs, such as OpenAI or Anthropic, use a similar strategy with minimal interface differences.
- Give step-by-step examples from section `4. Implementing tool calls with Gemini's Native API`, of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
-  **Section length:** 300 words (without counting the code)


## Section 5: Using Pydantic models as tools for on-demand structured outputs

- Give a concrete, yet extremely popular example on how we can use a Pydantic model as a tool to generate structured outputs dynamically.
- Explain that adding the Pydantic model as a tool is super useful in agentic scenarios, where you want to take multiple intermediate steps using unstructured outputs (which are easily interpreted by an LLM) and dynamically decide when to output the final answer in a structured form (which is easily interpreted in Python downstream).
- Also highlight that leveraging Pydantic objects is useful for multi-agent communication, ensuring out-of-the-box data quality checks and clearer schemas useful for debugging and monitoring.
- Give step-by-step examples from section `5. Using a Pydantic Model as a Tool for Structured Outputs`, of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
-  **Section length:** 200 words (without counting the code)

## Section 6: The downsides of running tools in a loop

- As a natural progression, we want not only to call a single tool, but also to allow an LLM to run them in a loop, chaining multiple tools and letting the LLM decide what tool to choose at each step.
- Create a mermaid diagram illustrating the tool calling loop:
    - user prompt
    - tool call
    - tool result
    - tool call
    - tool result
    - ...
- Explain the benefits: flexibility, adaptability, and the ability to handle complex multi-step tasks.
- Parallel Tool calls: Introduce the concept of parallel tool execution when tools are independent:
    - Explain scenarios where multiple tools can be called simultaneously (e.g., fetching weather data and stock prices independently)
    - Discuss the performance benefits: reduced latency and improved efficiency
- Discuss the limitations of sequential tool loops: 
    - Assumes the agent needs to run a tool at each step
    - Doesn't allow for reflection or reasoning between tool calls
    - Can lead to inefficient tool usage or getting stuck in loops
    - Limited ability to plan ahead or consider multiple approaches
- Connect these limitations to the need for more sophisticated patterns like **ReAct** (Reasoning and Acting), which we will explore in the next lesson.
- Give step-by-step examples from section `6. Running Tools in a Loop`, of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
-  **Section length:** 400 words (without counting the code)

## Section 7: Popular tools used within the industry

Theoretical discussion on popular tools, such as:
**Knowledge & Memory Access:**
    - **RAG Tools:** Tools that query vector databases, document stores, graph databases or other knowledge bases to retrieve relevant context for the LLM. Explain the "Retrieve" step of RAG as a tool. This pattern is known as agentic RAG.
    - **Database Query Tools:** Tools that can construct SQL/NoSQL queries and interact with classic databases. This pattern is known as text-to-SQL.
**Web Search & Browsing:**
    - Tools that interface with search engine APIs (e.g., Google Search, Bing Search, SerpAPI).
    - Tools that can fetch and parse content from web pages (simplified browsing).
    - Tools that can scrape web pages on the fly.
    - Importance for accessing current information and broad knowledge.
**Code Execution:**
    - **Python Interpreter Tool:** Allows the agent to write and execute Python code in a sandboxed environment. Invaluable for calculations, data manipulation, statistics, and using Python libraries, etc.
    - Mention security considerations (sandboxing is critical).
**Other Common Tools:**
    - Interacting with external APIs (e.g., calendar, email, project management).
    - File system operations (read/write files, list directories) - with caution.
- Connect these examples back to the multi-tool scenarios and tool chaining concepts discussed earlier.
- **Section length:** 400 words (without counting the code)

## Article code

Links to code that will be used to support the article. Always prioritize this code over every other piece of code found in the sources: 

- [Notebook 1](https://github.com/towardsai/course-ai-agents/blob/main/lessons/06_tools/notebook.ipynb)

## Golden Sources

- [Function calling with the Gemini API](https://ai.google.dev/gemini-api/docs/function-calling)
- [Function calling with OpenAI's API](https://platform.openai.com/docs/guides/function-calling)
- [Tool Calling Agent From Scratch](https://www.youtube.com/watch?v=ApoDzZP8_ck)
- [Efficient Tool Use with Chain-of-Abstraction Reasoning](https://arxiv.org/pdf/2401.17464v3)
- [Building AI Agents from scratch - Part 1: Tool use](https://www.newsletter.swirlai.com/p/building-ai-agents-from-scratch-part)

## Other Sources

- [What is Tool Calling? Connecting LLMs to Your Data](https://www.youtube.com/watch?v=h8gMhXYAv1k)
- [ReAct vs Plan-and-Execute: A Practical Comparison of LLM Agent Patterns](https://dev.to/jamesli/react-vs-plan-and-execute-a-practical-comparison-of-llm-agent-patterns-4gh9)
- [Agentic Design Patterns Part 3, Tool Use](https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-3-tool-use/)
