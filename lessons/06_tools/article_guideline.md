## Global Context

- **What I’m planning to share:** In this article we will talk about agent tool usage. We want to highlight that through tools we can enable agents to take actions and interact with the external world. By implementing tool calling from scratch we will understand how the LLM decides which tool to call, how it generates the correct parameters for the function and how it calls it. First we will implement tool definition and calling from scratch, then show how to do it with a popular API such as Gemini. Finally, we'll list the most essential tool categories such as knowledge/memory access (RAG), web search, and code execution. All of our ideas will be supported by code.
- **Why I think it’s valuable:** For an AI Engineer, tools are what transform an LLM from a text generator into an agent. Thus, tools are one of the key components of an AI agent. Tools are one of the foundations of agents. Thus, it's critical to open the black box and understand how an agent works with these tools.
- **Who the intended audience is:** Aspiring AI Engineers learning for the first time about LLM tools and agents.
- **Expected length of the article in words** (where 200-250 words ~= 1 minute of reading time): 2500 words (around 10 - 12.5 minutes reading time)


## Outline

1. Understanding why agents need tools
2. Implementing tool calls from scratch
3. Implementing tool calls with Gemini
4. Using Pydantic models as tools for on-demand structured outputs
5. The downsides of running tools in a loop
6. Going through popular tools used within the industry


## Section 1: Understanding why agents need tools

- Explain the inherent limitations of LLMs: They are primarily pattern matchers and text generators operating on input text. They cannot, by themselves:
    - Access real-time information (e.g., today's weather, latest news).
    - Interact with external databases or APIs.
    - Execute code.
    - Perform precise calculations beyond their training data.
    - Remember information beyond their context window reliably for long periods.
- Introduce "Tools" as the bridge: Functions or capabilities that an agent's orchestrator can execute on the LLM's behalf, based on the LLM's instructions.
- Analogy: Tools are an agent's "hands and senses," allowing it to perceive and act upon the world beyond its textual interface.
- Use a representative image to explain at a high level how tools work
-  **Section length:** 400 words


## Section 2: Implementing tool calls from scratch

- In this section we want to explain using code how LLMs use tools. We want to explain, with code support, how a tool is defined, how the LLM discovers what tools are available and how to call them through function schemas and how it interprets it's output. 
- Provide a mermaid diagram illustrating this request-execute-respond flow of calling a tool.
- Detail the key components and flow:
    1. **Tool Definition and Schema:** Explain how developers define tools for the LLM:
        - Define the Python function
        - Define the schema containing the function name, description (crucial for the LLM to understand when to use the tool), and parameters (name, type, description, required/optional). Often described using JSON Schema.
    2. **LLM's Decisional Role (Instruction Fine-tuning):**
        - The LLM is prompted with the user's query AND the list of available tool definitions.
        - Based on its training, the LLM *decides* if a tool call is appropriate to fulfill the query.
        - If so, it *selects* the most relevant tool.
        - It then *generates* the arguments for that tool in a structured format (typically JSON).
    3. **Application-Side Execution:**
        - The application receives the LLM's request to call a function (with specific arguments).
        - The application code executes the actual tool/function.
        - Explain the importance of structured outputs, as we have to output the function call as a structured output to interpret it. 
    4. **Returning Results to LLM:**
        - The output/result from the tool execution is sent back to the LLM as a new message in the conversation.
        - The LLM then uses this result to formulate a final response to the user or decide on the next step.
- Give step-by-step examples from section `5. Implementing tool calls from scratch` of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
- Follow the narrative from the Notebook, where we first define the tools, then we manually define the schemas, then we show to implement the `@tool` decorator often used in popular frameworks such as LangGraph. Then explain the system prompt and provde some examples on how to use it with some LLM calls with all the provided tools attached. Ultimately show to interpret the output and call the LLM.
-  **Section length:** 600 words (without counting the code or mermaid diagram)


## Section 3: Implementing tool calls with Gemini

- We want to show how to rewrite the written from scratch tool code from above using Gemini.
- Use the provided code examples to show how can we rewrite the "from scratch code"
from above using Gemini's interface.
- Give step-by-step examples from section `6. Implementing tool calls with Gemini` of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
- Explain that we still need the function schemas to pass them to Gemini through it's API. 
- Explain that all popular APIs, such as OpenAI or Anthropic, use a similar strategy with minimal interface differences.
-  **Section length:** 300 words (without counting the code)


## Section 4: Using Pydantic models as tools for on-demand structured outputs

- Another way to output structured outputs is by attaching a Pydantic schema as a tool.
- Explain that adding the pydantic model as a tool is super useful in agentic scenarios,
where you want to take multiple intermediate steps and dynamically decide when to output the final answer in a structured form.
- Also highlight that leveraging Pydantic objects is useful for multi-agent communication ensuring out-of-the-box data quality checks and clearer schemas useful for debugging and monitoring.
- Give step-by-step examples from section `7. Using a Pydantic Model as a Tool for Structured Outputs` of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
-  **Section length:** 200 words (without counting the code)

## Section 5: The downsides of running tools in a loop

- As a natural progression, we want not only to call a single tool, but allow an LLM to run them in a loop, while deciding what tool to choose at each step.
- Create a mermaid diagram illustrating the loop.
- Give step-by-step examples from section `8. Implementing tool calls with Gemini: Running tools in a loop` of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
- Explain that running tools in a loop is amazing, but it assumes the agent needs to run a tool at each step. Also, as the tools might pile up, it doesn't has the chance to think about each tools output, interpret it's result and take a further decision. Here is where ReAct kicks in, which we will explore further in the next lesson.
-  **Section length:** 300 words (without counting the code)

## Section 6: Going through popular tools used within the industry

Theoretical discussion on key categories of tools that empower agents:
- **Knowledge & Memory Access:**
    - **RAG Tools:** Tools that query vector databases, document stores, or other knowledge bases to retrieve relevant context for the LLM. Explain the "Retrieve" step of RAG as a tool.
    - **Database Query Tools:** Tools that can construct and execute SQL queries or interact with NoSQL databases.
- **Web Search & Browsing:**
    - Tools that interface with search engine APIs (e.g., Google Search, Bing Search, SerpAPI).
    - Tools that can fetch and parse content from web pages (simplified browsing).
    - Importance for accessing current information and broad knowledge.
- **Code Execution:**
    - **Python Interpreter Tool:** Allows the agent to write and execute Python code in a sandboxed environment. Invaluable for calculations, data manipulation, using Python libraries, etc.
    - Mention security considerations (sandboxing is critical).
- **Other Common Tools:**
    - Interacting with external APIs (e.g., calendar, email, project management).
    - File system operations (read/write files, list directories) - with caution.
    - Mathematical calculators.
- **Section length:** 400 words (without counting the code)

## References
