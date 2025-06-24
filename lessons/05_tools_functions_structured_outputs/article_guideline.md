## Global Context

- **What I’m planning to share:** In this article we will talk about agent tool usage and structure outputs. We want to highlight that through tools we can enable agents to take actions and interact with the external world. We will look into how **Function Calling** works, understanding how the LLM decides  which tool to call, how it generates the correct parameters for the function and how it calls it. Afterward, we'll introduce **Structured Outputs** as a way to achieve reliable data extraction. Finally, we'll list the most essential tool categories such as knowledge/memory access (RAG), web search, and code execution. All of our ideas will be supported by code.
- **Why I think it’s valuable:** For an AI Engineer, tools are what transform an LLM from a text generator into an agent. Thus, tools are one of the key components of an AI agent. Tools, together with structured outputs, are one of the foundations of agents. Thus, it's critical to open the black box and understand how an agent works with these tools, when we need them and how to guide the agent to output structured outputs that can be validated and used programatically in Python or other programming language.
- **Who the intended audience is:** People learning for the first time about LLM tool usage and structured outputs.
- **Expected length of the article in words** (where 200-250 words ~= 1 minute of reading time): 3200 words (15 minutes reading time)

## Outline

1. Understanding why agents need tools
2. Implementing tool calls from scratch
3. Implementing tool calls with Gemini
4. Understanding why agents need structured outputs
5. Implementing structured outputs from scratch using JSON
6. Implementing structured outputs from scratch using Pydantic
7. Implementing structured ouputs using Gemini and Pydantic
8. Going through popular tools used within the industry

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

- In this section we want to explain using code how LLM use tools. We want to explain, with code support, how a tool is defined, how the LLM decides what tool to use, how it's called, and how it interprets it's output. 
- Provide a mermaid diagram illustrating this request-execute-respond flow.
- Detail the key components and flow:
    1. **Tool Definition/Schema:** Explain how developers define tools for the LLM:
        - Function name.
        - Description (crucial for the LLM to understand when to use the tool).
        - Parameters (name, type, description, required/optional). Often described using JSON Schema.
    2. **LLM's Decisional Role (Instruction Fine-tuning):**
        - The LLM is prompted with the user's query AND the list of available tool definitions.
        - Based on its training, the LLM *decides* if a tool call is appropriate to fulfill the query.
        - If so, it *selects* the most relevant tool.
        - It then *generates* the arguments for that tool in a structured format (typically JSON).
    3. **Application-Side Execution:**
        - The application receives the LLM's request to call a function (with specific arguments).
        - The application code executes the actual tool/function.
    4. **Returning Results to LLM:**
        - The output/result from the tool execution is sent back to the LLM as a new message in the conversation.
        - The LLM then uses this result to formulate a final response to the user or decide on the next step.
- Use the provided code examples on defining tools from scratch to support the ideas.
-  **Section length:** 600 words (without counting the code or mermaid diagram)

## Section 3: Implementing tool calls with Gemini

- We want to show how to rewrite the written from scratch tool code from above using Gemini.
- Use the provided code examples to show how can we rewrite the "from scratch code"
from above using Gemini's interface.
- Explain that all popular APIs, such as OpenAI or Anthropic, use a similar strategy with minimal interface differences.
-  **Section length:** 300 words (without counting the code)

## Section 4: Understanding why agents need structured outputs

- First we want to highlight at a theoretical level why we need structured ouputs when integrating LLMs in our application.
- Benefits:
    - Greatly improves reliability when needing to extract specific pieces of information or structured data from an LLM's free-text response.
    - Reduces the need for fragile regex or string parsing.
- Use Cases:
    - Extracting entities from text (names, dates, locations).
    - Formatting LLM output into a predefined data structure for downstream processing.
    - Can be used as an alternative to function calling when the goal is just data extraction, not necessarily an action.
- Generate a mermaid diagram to support the idea.
-  **Section length:** 300 words (without counting the mermaid diagram)

## Section 5: Implementing structured outputs from scratch using JSON

- To support out theory section from above, we want to show to implement structured outputs from scratch using JSON schemas into our prompt templates
- Use the provided code examples, to explain how to enforce the LLM to output data structures in JSON format and how to parse them to transform them into Python dicts that can be used within the code
-  **Section length:** 500 words (without counting the code)

## Section 6: Implementing structured outputs from scratch using Pydantic

- Rewrite the section from above using Pydantic schemas to model the structured outputs
- Use the provided code examples to support your ideas
- Show both options: injecting pydantic schema into the prompt vs. adding it as a tool
- Explain that adding the pydantic model as a tool is super useful in agentic scenarios, 
where you want the final output to be structured
- Explain that Pydantic objects are the go-to method to model structured outputs as they offer field and type checking bypassing the ambiguity of Python dictionaries. 
-  **Section length:** 300 words (without counting the code)

## Section 7: Implementing structured ouputs using Gemini and Pydantic

- As a more industry-level example, explain how we can directly enforce the Gemini API to output Pydantic objects
-  **Section length:** 100 words (without counting the code)

## Section 8: Going through popular tools used within the industry

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
