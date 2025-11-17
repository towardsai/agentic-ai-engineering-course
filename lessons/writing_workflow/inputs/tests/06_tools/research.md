# Research

## Research Results

<details>
<summary>What are the recommended ways to implement and register Python functions as agent “tools” using decorators (such as LangChain’s or LangGraph’s @tool), and what open-source examples illustrate this pattern in practice?</summary>

### Source [1]: https://python.langchain.com/docs/concepts/tools/

Query: What are the recommended ways to implement and register Python functions as agent “tools” using decorators (such as LangChain’s or LangGraph’s @tool), and what open-source examples illustrate this pattern in practice?

Answer: The **recommended way to implement and register Python functions as agent "tools" in LangChain** is by using the `@tool` decorator. This decorator simplifies tool creation and is suggested for most use cases. After defining a function, you add the `@tool` decorator, which then implements the Tool Interface for that function. For example:

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
```

Once decorated, the tool can be invoked directly, for example with `multiply.invoke({"a": 2, "b": 3})`. Additionally, you can inspect the tool’s schema and properties like its name, description, and argument structure via attributes such as `multiply.name`, `multiply.description`, and `multiply.args`. While there are alternative methods—such as subclassing `BaseTool` or using `StructuredTool`—the decorator approach is generally preferred for its simplicity and directness.

-----

-----

-----

### Source [3]: https://langchain-opentutorial.gitbook.io/langchain-opentutorial/15-agent/01-tools

Query: What are the recommended ways to implement and register Python functions as agent “tools” using decorators (such as LangChain’s or LangGraph’s @tool), and what open-source examples illustrate this pattern in practice?

Answer: This guide provides a practical explanation and code examples for **creating custom tools in LangChain using the `@tool` decorator**. To define a tool, simply decorate a standard Python function with `@tool` from `langchain.tools`. For instance:

```python
from langchain.tools import tool

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b
```

You can then execute these tools using their `.invoke()` method, passing arguments as dictionaries (e.g., `add_numbers.invoke({"a": 3, "b": 4})`). The decorator also supports customization via its parameters, facilitating automated documentation and flexible interface creation. This approach is recommended for turning regular Python functions into agent-ready tools quickly and with minimal boilerplate.

-----

-----

-----

### Source [5]: https://pub.towardsai.net/crafting-langchain-tools-a-complete-guide-to-custom-tool-development-f21fd2f16622

Query: What are the recommended ways to implement and register Python functions as agent “tools” using decorators (such as LangChain’s or LangGraph’s @tool), and what open-source examples illustrate this pattern in practice?

Answer: The article recommends **starting with the `@tool` decorator for prototyping custom tools in LangChain** due to its simplicity and ease of use. As development progresses and requirements become more complex, you can transition to using `StructuredTool`, `BaseTool`, or `Runnable` for greater control and customization. This pattern is commonly found in open-source projects, where initial functionality is implemented with decorators and later refactored into more advanced classes as needed. The article provides a comprehensive guide to custom tool development and illustrates how decorators streamline the process of registering and managing Python functions as agent tools.

-----

-----

</details>

<details>
<summary>How can Pydantic models be employed to generate and validate structured outputs from large-language-model agents, and what real-world examples show Pydantic being treated as a “tool” for on-demand schemas?</summary>

### Source [6]: https://xebia.com/blog/enforce-and-validate-llm-output-with-pydantic/

Query: How can Pydantic models be employed to generate and validate structured outputs from large-language-model agents, and what real-world examples show Pydantic being treated as a “tool” for on-demand schemas?

Answer: Pydantic models can be used to **enforce and validate the structured outputs** from large-language-model (LLM) agents by defining explicit schemas for expected responses. For example, a Pydantic model can specify that the "difficulty" field in a model must be one of "easy", "medium", or "hard" using Python’s `Literal` type. When an LLM output does not match this schema (e.g., returning "Unknown" for difficulty), Pydantic raises a `ValidationError` indicating the allowed values. The validation is performed using methods like `model_validate_json(response)`, which parses and validates the output against the defined Pydantic model. This approach lets developers **gain greater control and robustness** over LLM outputs, ensuring that only structurally correct and semantically valid data is accepted, thereby helping to build more reliable AI systems.

-----

-----

-----

### Source [7]: https://www.leewayhertz.com/structured-outputs-in-llms/

Query: How can Pydantic models be employed to generate and validate structured outputs from large-language-model agents, and what real-world examples show Pydantic being treated as a “tool” for on-demand schemas?

Answer: Pydantic models are employed in **structured output pipelines** for LLMs by serving as output parsers. After prompting an LLM to return information in a specific format, the raw (often imperfectly structured) output is passed through a Pydantic OutputParser. This parser enforces a schema on the output, validating and transforming it into a **consistent and dependable structured format**. The process involves:
- Using a prompt template with format instructions for the LLM.
- Having the LLM generate a raw output.
- Parsing and validating this output with Pydantic, which checks if the data matches the expected schema.
This approach ensures that even if the LLM’s initial output is inconsistent, the final result strictly adheres to the required structure, supporting use cases like **tool calling**, where Pydantic acts as an on-demand schema validator to guarantee reliable interactions with downstream processes.

-----

-----

-----

### Source [8]: https://docs.pydantic.dev/latest/concepts/models/

Query: How can Pydantic models be employed to generate and validate structured outputs from large-language-model agents, and what real-world examples show Pydantic being treated as a “tool” for on-demand schemas?

Answer: Pydantic provides several methods for **validating data against defined schemas**:
- `model_validate()`: Validates a dictionary or object against the model, raising a `ValidationError` if constraints aren’t met.
- `model_validate_json()`: Validates data provided as a JSON string, which is often more efficient for processing LLM outputs that return JSON payloads.
- `model_validate_strings()`: Validates dictionaries with string keys/values, coercing them into the correct types as needed.
These validation methods are critical for **transforming unstructured or loosely structured LLM outputs into strongly-typed, schema-conformant Python objects**. When an LLM produces output, developers can use these methods to ensure the result matches the expected data structure, catching errors early and ensuring application robustness.

-----

-----

-----

### Source [9]: https://neo4j.com/blog/developer/agentic-graphrag-for-commercial-contracts/

Query: How can Pydantic models be employed to generate and validate structured outputs from large-language-model agents, and what real-world examples show Pydantic being treated as a “tool” for on-demand schemas?

Answer: Pydantic is used as a **tool for structured data extraction from LLMs** by defining explicit output schemas. For example, a `Location` Pydantic model specifies fields like `address`, `city`, `state`, and `country`—each with defined types and descriptions. These schemas can be provided to the LLM, which then attempts to produce outputs matching the schema. Fields may include type requirements (e.g., `Optional[str]` for nullable fields) and descriptions that serve as guidance for the LLM. This practice ensures outputs are **well-structured and aligned with downstream requirements**, allowing Pydantic to function as a “tool” for on-demand schemas, especially in complex applications like extracting structured information from commercial contracts.

-----

-----

-----

### Source [10]: https://pydantic.dev/articles/llm-intro

Query: How can Pydantic models be employed to generate and validate structured outputs from large-language-model agents, and what real-world examples show Pydantic being treated as a “tool” for on-demand schemas?

Answer: The official Pydantic site discusses how Pydantic can be employed to **validate structured outputs from language models** such as those from OpenAI. Using Pydantic, developers can define the expected schema for LLM outputs and validate actual responses, ensuring they match the specified model. This helps in writing **reliable code** by catching mismatches and errors early, streamlining the use of LLMs in production environments where consistent structure and correctness are critical for downstream processing and integration.

-----

</details>

<details>
<summary>What security risks accompany giving LLM agents a code-execution tool (e.g., Python interpreter) and what sandboxing or isolation strategies do experts recommend to mitigate them?</summary>

### Source [12]: Dida.do

Query: What security risks accompany giving LLM agents a code-execution tool (e.g., Python interpreter) and what sandboxing or isolation strategies do experts recommend to mitigate them?

Answer: Providing LLM agents with code-execution tools like a Python interpreter increases security risks. These risks include **arbitrary code execution** (e.g., using `os.system` or `subprocess`), **resource exhaustion** (e.g., CPU, memory, or disk overload), and **unauthorized file system access**. To mitigate these risks, implementing a secure Python sandbox is recommended. This sandbox should manage resources and create safe execution environments that encapsulate potentially harmful code, preventing it from affecting the broader system.

-----

-----

### Source [14]: OWASP

Query: What security risks accompany giving LLM agents a code-execution tool (e.g., Python interpreter) and what sandboxing or isolation strategies do experts recommend to mitigate them?

Answer: The OWASP Top 10 for Large Language Model Applications lists several risks relevant to using LLMs with code-execution tools. **Prompt injection** (LLM01) can lead to unauthorized access and data breaches. **Insecure output handling** (LLM02) can result in code execution that compromises systems. **Excessive agency** (LLM08) is a risk when LLMs are granted unchecked autonomy to execute actions. While OWASP does not detail specific sandboxing strategies, it emphasizes the importance of securing LLM outputs and inputs to prevent these risks.

-----

</details>

<details>
<summary>What empirical findings or research analyses describe inefficiencies or failure modes when LLM agents repeatedly call tools in a simple loop, and how do these limitations motivate more advanced reasoning-and-acting patterns like ReAct?</summary>

### Source [15]: https://arxiv.org/pdf/2503.13657

Query: What empirical findings or research analyses describe inefficiencies or failure modes when LLM agents repeatedly call tools in a simple loop, and how do these limitations motivate more advanced reasoning-and-acting patterns like ReAct?

Answer: A comprehensive empirical study by Cemri et al. (2025) analyzes failure modes in multi-agent large language model (LLM) systems, categorizing them with the MAST taxonomy. Several failure patterns are directly relevant to inefficiencies observed when LLM agents repeatedly call tools in a naive loop:

- **Step Repetition**: Agents may get stuck repeating the same or similar actions, such as repeatedly calling a tool without making progress toward the task goal. This occurs in 13.98% of observed failures, indicating a substantial inefficiency where agents do not properly track state or results, leading to unnecessary or redundant tool invocations.
  
- **Reasoning-Action Mismatch**: In 6.82% of cases, agents show a disconnect between their reasoning steps and their actions—such as requesting a tool call that does not logically follow from their internal dialogue or prior context. This reflects a lack of integrated reasoning and acting, which can result in loops or irrelevant tool use.
  
- **No or Incomplete Verification**: Some agents fail to verify whether a tool call has achieved the intended effect, leading to repeated or unnecessary actions (seen in 7.15% of failures). This signals the need for more sophisticated reasoning to assess the impact of actions before proceeding.
  
- **Premature Termination and Loss of Conversation History**: Agents sometimes terminate loops too early or lose track of conversation state (11.65% and 6.66% of failures, respectively), further compounding inefficiency and failure to complete multi-step tasks.

The paper argues that these failure modes motivate the need for advanced *reasoning-and-acting* patterns, such as the ReAct framework, which tightly integrates reasoning steps with action execution and verification. By combining stepwise reasoning with action selection and outcome assessment, frameworks like ReAct can mitigate repetitive loops, bridge reasoning-action gaps, and ensure more goal-directed tool use.

-----

-----

-----

### Source [16]: https://arxiv.org/html/2412.01130v2

Query: What empirical findings or research analyses describe inefficiencies or failure modes when LLM agents repeatedly call tools in a simple loop, and how do these limitations motivate more advanced reasoning-and-acting patterns like ReAct?

Answer: Chen et al. (2024) systematically investigate the function-calling capabilities of LLMs and identify several challenges in current tool use patterns. They highlight that while LLMs can perform zero-shot tool usage, naive approaches often result in **inefficient tool invocation**—such as repeatedly calling the same function in a loop without adapting behavior based on results. This inefficiency is attributed to LLMs lacking robust mechanisms for integrating prior tool call outcomes into subsequent reasoning and decision-making.

To address these limitations, the authors introduce enhancements such as **decision tokens** and the integration of instruction-following data, which enable LLMs to better determine when tool use is appropriate and when to update their strategy based on tool responses. They observe that:

- Standard LLM prompting can lead to repetitive calls and poor relevance detection, especially when the agent does not condition its next action on the tool’s previous output.
- Incorporating explicit reasoning (e.g., via chain-of-thought) and mixed data with instruction-following tasks significantly improves both the accuracy and efficiency of tool use.
- These improvements mirror the principles behind advanced patterns like ReAct, which emphasize alternating between reasoning steps and tool actions, explicitly conditioning actions on intermediate results to avoid redundant loops and failure to progress.

The empirical findings underline the need for reasoning-and-acting patterns that allow LLMs to reflect on tool outputs before deciding the next step.

-----

-----

</details>

<details>
<summary>Which categories of external tools are most frequently integrated with production LLM agents (e.g., retrieval-augmented search, web browsing, database querying, code execution), and what industry case studies highlight their effectiveness?</summary>

### Source [18]: https://www.mercity.ai/blog-post/guide-to-integrating-tools-and-apis-with-language-models

Query: Which categories of external tools are most frequently integrated with production LLM agents (e.g., retrieval-augmented search, web browsing, database querying, code execution), and what industry case studies highlight their effectiveness?

Answer: This source provides an in-depth overview of integrating **external tools and APIs with production LLM agents**. It highlights that LLMs like GPT-4 serve as the core user interface for these integrations. Integration approaches discussed include function calling, prompting techniques, and the use of specialized tool-use prompts. The guide notes that:

- **Larger models (e.g., GPT-4, LLaMa 70B, Falcon 180B)** are generally more effective at complex tool integration due to superior instruction-following and conversation management abilities.
- **Smaller models** (e.g., LLaMa-13B) can be fine-tuned using techniques like Prefix Tuning and IA3 for more targeted tool use.
- **Typical tool integration categories** identified include:
  - Retrieval-augmented search (accessing up-to-date or proprietary information)
  - Database querying (extracting and analyzing structured data)
  - Code execution (enabling mathematical or data processing tasks)
  - Web browsing (fetching current web data)
- The article underscores the necessity for custom frameworks when integrating LLMs with private or internal tools, as out-of-the-box solutions (like ChatGPT extensions) may not suffice for enterprise needs.

No specific industry case studies are detailed, but the guide emphasizes the importance of reliable, scalable integration pipelines for production deployments.

-----

-----

-----

### Source [19]: https://aman.ai/primers/ai/agents/

Query: Which categories of external tools are most frequently integrated with production LLM agents (e.g., retrieval-augmented search, web browsing, database querying, code execution), and what industry case studies highlight their effectiveness?

Answer: This source details **practical examples of tool calling in LLM agents** with a focus on two major categories:

- **Web search tools:** LLMs can perform real-time web searches to answer queries requiring current information. For instance, Bing Copilot uses a web search tool to fetch up-to-date product recommendations, demonstrating the value of integrating live search capabilities for questions reliant on the latest data.
- **Code execution tools:** LLMs like ChatGPT can generate and execute code to solve computational problems, such as calculating compound interest. This capability allows the agent to provide precise numeric answers and handle complex logic, moving beyond simple text reasoning.

The examples illustrate how production LLM agents dynamically select the appropriate external tool based on user intent, showcasing the flexibility and effectiveness of integrated tool use for both knowledge retrieval and computational tasks.

-----

-----

-----

### Source [20]: https://mirascope.com/blog/llm-integration

Query: Which categories of external tools are most frequently integrated with production LLM agents (e.g., retrieval-augmented search, web browsing, database querying, code execution), and what industry case studies highlight their effectiveness?

Answer: This source provides a comprehensive guide to **LLM integration with external resources and APIs**. It identifies the main categories of tool integrations as:

- **Retrieval-augmented search** (accessing external knowledge bases or documents)
- **Database querying** (interfacing with structured databases for data extraction and analysis)
- **Code execution** (running scripts or code for calculations and data processing)
- **Device or application control** (sending commands to external systems)

The article emphasizes that while API integration is fundamental, production systems must also address challenges such as non-deterministic LLM outputs, prompt engineering, error handling, and network reliability. The toolkit and platform discussed (Mirascope and Lilypad) are positioned to help manage and observe these integrations in real-world deployments.

Although no detailed industry case studies are provided in the excerpt, the focus is on the technical best practices and frameworks required for robust, production-grade LLM agent deployments.

-----

-----

-----

### Source [21]: https://arxiv.org/html/2506.18096v1

Query: Which categories of external tools are most frequently integrated with production LLM agents (e.g., retrieval-augmented search, web browsing, database querying, code execution), and what industry case studies highlight their effectiveness?

Answer: This academic paper systematically examines **Deep Research (DR) agents**, a class of LLM-powered systems, and details their integration with several categories of external tools:

- **Code interpreter:** Enables script execution (mainly Python, sometimes Java), supporting data processing, algorithm verification, and model simulations. Most DR agents include this feature for real-time computational reasoning and literature-driven analysis.
- **Data analytics modules:** Transform retrieved data into structured insights, supporting summary statistics, visualizations, and quantitative evaluations. Examples include SQL-based queries for aggregate analyses and chart/report generation within platforms like CoSearchAgent, and structured dataset extraction and analysis in agents like AutoGLM.
- **Retrieval-augmented search and web browsing:** DR agents retrieve external knowledge in real time, often via embedded web browsers, to inform downstream analysis and reporting.
- **Reason-in-Documents:** Components such as that in Search-o1 refine and summarize lengthy retrieved texts to extract key metrics for further use.

Industry case studies are not directly cited, but academic examples (e.g., CoSearchAgent, AutoGLM, Search-o1) demonstrate the effectiveness of these tool integrations in accelerating complex research workflows, supporting hypothesis testing, and automating decision-making processes.

-----

-----

</details>

<details>
<summary>What security and sandboxing strategies do experts recommend when granting an LLM agent access to a code-execution tool (e.g., a Python interpreter in a Jupyter-style sandbox)?</summary>

### Source [22]: https://arxiv.org/html/2506.08837v1

Query: What security and sandboxing strategies do experts recommend when granting an LLM agent access to a code-execution tool (e.g., a Python interpreter in a Jupyter-style sandbox)?

Answer: This source outlines principled design patterns for securing LLM agents, emphasizing isolation between untrusted data and agent control flow. Key recommendations include:

- **Action-Selector Pattern:** The agent acts only as an *action selector*, mapping user requests to a set of predefined, vetted tool calls. This prevents arbitrary code execution and feedback loops that could propagate prompt injection attacks.
- **Best Practices:** The authors stress that every LLM agent should run actions in a sandboxed environment to limit the impact of potentially malicious code. Additionally, users should be asked for confirmation before sensitive actions are taken.
- **Prompt Injection Mitigation:** By removing user prompts from outputs before replying, agents can reduce the risk of prompt injection attacks propagating through the system.

The paper distinguishes between general best practices (like sandboxing and user confirmation) and specific design patterns (such as the action-selector), recommending that both be incorporated for robust security.

-----

-----

-----

### Source [23]: https://dida.do/blog/setting-up-a-secure-python-sandbox-for-llm-agents

Query: What security and sandboxing strategies do experts recommend when granting an LLM agent access to a code-execution tool (e.g., a Python interpreter in a Jupyter-style sandbox)?

Answer: This source discusses threats from LLM-generated code and describes sandboxing as a primary mitigation strategy:

- **Risks Addressed:** Arbitrary code execution (e.g., `os.system`, `subprocess`), resource exhaustion (CPU, memory, disk), and unauthorized filesystem access.
- **Sandboxing Solution:** Implement a secure Python sandbox to manage resources and restrict code execution. The sandbox should encapsulate potentially harmful code, protecting the broader system.
- **Example Technologies:** The article suggests combining gVisor (a user-space kernel for isolating containers) with Jupyter Notebook to create a secure, interactive Python execution environment.

The primary goal is to build execution environments that strictly manage what code can do, preventing untrusted code from affecting the system outside the sandbox.

-----

-----

-----

### Source [24]: https://huggingface.co/docs/smolagents/en/tutorials/secure_code_execution

Query: What security and sandboxing strategies do experts recommend when granting an LLM agent access to a code-execution tool (e.g., a Python interpreter in a Jupyter-style sandbox)?

Answer: This documentation emphasizes that **robust security isolation** for LLM-generated code can only be achieved through remote execution environments, such as Docker containers or services like E2B:

- **Risks of Local Execution:** Running code directly in your environment is inherently risky—malicious or erroneous code can damage the filesystem, exploit resources, abuse APIs, or compromise network security.
- **Types of Threats:** These include accidental errors, supply chain attacks, prompt injection, and exploitation of publicly accessible agents.
- **Recommended Measures:**
  - Use **remote execution** systems (e.g., Docker) to isolate the code from the host.
  - Accept that no solution is 100% safe; always weigh the level of agent autonomy against the security risk.
  - Be cautious about agent exposure and always assume that any executed code may be adversarial.

The documentation advises that security comes at the cost of greater setup complexity, but is essential for safe deployment.

-----

-----

-----

### Source [25]: https://amirmalik.net/2025/03/07/code-sandboxes-for-llm-ai-agents

Query: What security and sandboxing strategies do experts recommend when granting an LLM agent access to a code-execution tool (e.g., a Python interpreter in a Jupyter-style sandbox)?

Answer: This blog post explains how code sandboxes provide isolation for running untrusted code generated by LLMs:

- **Sandboxing Methods:**
  - **Containers (e.g., Docker, LXC):** Offer strong isolation with minimal performance penalty, and are a standard approach for running untrusted code.
  - **User-mode Kernels:** Intercept and handle system calls at the kernel level, further isolating applications from the host.
  - **Virtual Machines:** Use lightweight hypervisors to provide hardware-level isolation, trading a slight performance overhead for increased security.
  - **Other Technologies:** Mentions WebAssembly (Wasm) and the JVM, which can run code in isolated virtual machines, though compatibility may be limited for some languages or modules.
- **Practical Considerations:** The sandbox can be persistent per user if code execution needs to access user files, but the core principle is to never execute LLM-generated code directly on the host.

This source stresses that while simple methods like `eval()` are unsafe, modern sandboxing using containers or VMs is essential for secure code execution.

-----

-----

</details>

<details>
<summary>How does Retrieval-Augmented Generation (RAG) function as a “knowledge access tool” in modern agent frameworks, and what production examples demonstrate agentic RAG in action?</summary>

### Source [27]: https://www.weka.io/learn/guide/ai-ml/retrieval-augmented-generation/

Query: How does Retrieval-Augmented Generation (RAG) function as a “knowledge access tool” in modern agent frameworks, and what production examples demonstrate agentic RAG in action?

Answer: Retrieval-Augmented Generation (RAG) functions as a **knowledge access tool** by integrating generative language models with external retrieval systems. The implementation involves several steps:
- **Setting up a document store or knowledge base:** RAG relies on a store of relevant knowledge, which can be structured (databases), unstructured (documents, articles), or even live sources (APIs).
- **Preprocessing and indexing:** Documents are converted into **semantic embeddings** (often using transformer models like BERT or SBERT) and stored in a vector database for efficient retrieval.
- **Building the retrieval system:** User queries are embedded and matched against document embeddings to find the most relevant content.
- **Integration with the generative model:** The retrieved information is concatenated with the user query and used as input for the generative model.
- **Generation and post-processing:** The model produces a response that is informed by both internal knowledge and retrieved data. Additional steps such as summarization or fact-checking can refine the output.

This framework allows agents to **access up-to-date, contextually relevant knowledge**, overcoming the static nature of pretrained models and enabling applications such as customer support, enterprise knowledge management, and dynamic FAQ bots[1].

-----

-----

-----

### Source [28]: https://www1.hkexnews.hk/app/sehk/2025/107494/documents/sehk25062703133.pdf

Query: How does Retrieval-Augmented Generation (RAG) function as a “knowledge access tool” in modern agent frameworks, and what production examples demonstrate agentic RAG in action?

Answer: Retrieval-Augmented Generation is described as a **knowledge access tool** that provides powerful retrieval capabilities to support **employee knowledge acquisition**. It enables **multi-modal and multi-scenario** knowledge access, which means it can handle different types of data (text, images, etc.) and be applied to various contexts within an organization. This makes RAG especially valuable for enterprise environments where employees need timely and accurate access to organizational knowledge and resources[2].

-----

-----

-----

### Source [29]: https://www.purestorage.com/knowledge/what-is-retrieval-augmented-generation.html

Query: How does Retrieval-Augmented Generation (RAG) function as a “knowledge access tool” in modern agent frameworks, and what production examples demonstrate agentic RAG in action?

Answer: RAG extends the capabilities of large language models (LLMs) by **integrating them with an external, authoritative knowledge base**. The process works as follows:
- A **pre-trained LLM** (like GPT or BERT) is used for language understanding and generation.
- The **retrieval mechanism** uses ranking functions (like Okapi BM25) to fetch relevant information from a knowledge base (which could be a database, document collection, or curated web pages).
- The **user query** is processed, and relevant content is retrieved and **fused with the original query** to provide a **context-rich input** to the LLM.
- The LLM then generates an output informed by the external knowledge.

This architecture allows generative AI agents to **answer questions, summarize information, or generate content** that is up-to-date and grounded in real, authoritative data, rather than just relying on their fixed training data[3].

-----

-----

-----

### Source [30]: https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview

Query: How does Retrieval-Augmented Generation (RAG) function as a “knowledge access tool” in modern agent frameworks, and what production examples demonstrate agentic RAG in action?

Answer: RAG is presented as an **architecture that augments LLMs** (such as ChatGPT) by adding an **information retrieval system** for providing grounding data. In modern agent frameworks, this enables agents to:
- **Restrict and ground responses** to enterprise content, such as proprietary documents and images, by leveraging vectorized documents and embedding models.
- Ensure that only **relevant, secure, and up-to-date information** is supplied to the LLM, enhancing accuracy and compliance.
- Provide **indexing strategies** for efficient and scalable content management, and **query capabilities** for relevance tuning.
- Offer **security, reliability, and integration** with embedding and chat models.

A production example is Microsoft's **Azure AI Search**, which serves as the information retrieval backbone in enterprise RAG solutions, ensuring that generative agents supply responses based on company-specific content rather than open-web or outdated sources[4].

-----

-----

-----

### Source [31]: https://www.k2view.com/what-is-retrieval-augmented-generation

Query: How does Retrieval-Augmented Generation (RAG) function as a “knowledge access tool” in modern agent frameworks, and what production examples demonstrate agentic RAG in action?

Answer: RAG is a **Generative AI architecture** that augments LLMs with **fresh, trusted data** from authoritative internal knowledge sources. The data flow is:
- The **user prompt** triggers the retrieval model to access internal sources (both structured and unstructured).
- The **retrieval model** queries the company’s systems and knowledge bases, then **enriches the user’s prompt** with this contextual information.
- The **generation model (LLM)** receives the enriched prompt and produces a response grounded in both its model weights and the newly retrieved data.

This practical approach enables agents to **overcome the limitations of static model knowledge** and deliver responses that are accurate, timely, and context-specific—key for production use cases such as customer support, compliance, and data-driven decision-making[5].

-----

-----

</details>

<details>
<summary>How is LangGraph’s @tool (or similar) decorator used to auto-generate JSON schemas and register functions as tools, and what open-source examples illustrate this pattern in practice?</summary>

### Source [32]: https://python.langchain.com/docs/concepts/tools/

Query: How is LangGraph’s @tool (or similar) decorator used to auto-generate JSON schemas and register functions as tools, and what open-source examples illustrate this pattern in practice?

Answer: LangChain provides a **@tool decorator** to simplify the creation and registration of tools. When you decorate a function with @tool, the following happens:

- **Automatic registration:** The function is converted into an object that implements the Tool Interface, making it usable in agent workflows.
- **Auto-generated JSON schema:** The decorator inspects the function signature (parameter names, types, and docstring) to generate a schema describing the tool’s expected inputs and outputs.
- **Naming and description:** By default, the tool’s name is the function name, and its description is either the docstring or the function signature.
- **Usage example:**
  ```python
  from langchain_core.tools import tool

  @tool
  def multiply(a: int, b: int) -> int:
      """Multiply two numbers."""
      return a * b
  ```
- **Direct invocation:** The tool can be directly called with arguments as a dictionary (e.g., `multiply.invoke({"a": 2, "b": 3})`).
- **Introspection:** You can inspect the tool’s schema and metadata via attributes like `.name`, `.description`, and `.args`.

This approach is recommended for most scenarios, though LangChain also supports subclassing or structured tools for advanced needs[1].

-----

-----

-----

### Source [33]: https://itnext.io/the-mcp-revolution-transforming-agents-with-mcp-2f053da01e8c

Query: How is LangGraph’s @tool (or similar) decorator used to auto-generate JSON schemas and register functions as tools, and what open-source examples illustrate this pattern in practice?

Answer: The **@mcp.tool decorator** in the MCP project provides similar functionality to LangChain’s @tool decorator:

- **Automatic tool registration:** Functions decorated with @mcp.tool are automatically registered as agent tools.
- **Schema generation:** The decorator uses Python type hints to auto-generate the JSON schema for each tool's expected inputs and outputs.
- **Decorator-based pattern:** This pattern abstracts away manual schema creation, allowing developers to focus on function logic and type annotations for schema definition.
- **Open-source example:** The MCP project itself is referenced as using this decorator-based approach for tool creation and schema handling[2].

-----

-----

-----

### Source [34]: https://python.langchain.com/docs/how_to/custom_tools/

Query: How is LangGraph’s @tool (or similar) decorator used to auto-generate JSON schemas and register functions as tools, and what open-source examples illustrate this pattern in practice?

Answer: LangChain’s custom tools documentation expands on the **@tool decorator** usage:

- **Default name and schema:** The function name becomes the tool’s name, and parameter types become the tool’s schema.
- **Docstring parsing:** If you enable `@tool(parse_docstring=True)`, the decorator parses Google-style docstrings to enrich the JSON schema with argument descriptions and function documentation.
- **Schema inspection:** The schema is accessible via `.args_schema.model_json_schema()`, producing a JSON schema like:
  ```python
  {
      'description': 'The foo.',
      'properties': {
          'bar': {'description': 'The bar.', 'title': 'Bar', 'type': 'string'},
          'baz': {'description': 'The baz.', 'title': 'Baz', 'type': 'integer'}
      },
      'required': ['bar', 'baz'],
      'title': 'fooSchema',
      'type': 'object'
  }
  ```
- **Validation:** If the docstring does not parse correctly when `parse_docstring=True`, a ValueError is raised, ensuring schema consistency.

This pattern allows for rapid tool development and robust schema generation directly from Python code[3].

-----

-----

-----

### Source [35]: https://langchain-ai.github.io/langgraph/how-tos/tool-calling/

Query: How is LangGraph’s @tool (or similar) decorator used to auto-generate JSON schemas and register functions as tools, and what open-source examples illustrate this pattern in practice?

Answer: LangGraph allows functions to be registered as tools for agent workflows, aligning with the decorator-based approach of LangChain:

- **ToolNode usage:** The `ToolNode` class can wrap plain Python functions to make them available as agent tools.
- **Binding tools to models:** The `model.bind_tools([your_function])` method registers the tool with the agent, enabling tool-calling workflows.
- **Example:**
  ```python
  def get_weather(location: str):
      """Call to get the current weather."""
      ...

  tool_node = ToolNode([get_weather])
  model_with_tools = model.bind_tools([get_weather])
  ```
- **Schema inference:** While this example does not explicitly show a decorator, the mechanism is similar—functions are introspected for their signatures and docstrings to generate the tool schema and registration metadata.

This approach is core to how LangGraph integrates tool functions into agent-based architectures[5].

-----

-----

</details>

<details>
<summary>What real-world applications integrate web-search or browsing tools into LLM agents (e.g., Bing Copilot, Perplexity AI, or Google’s Search-enabled Gemini) to fetch up-to-date information for users?</summary>

### Source [36]: https://www.business-standard.com/technology/tech-news/microsoft-brings-copilot-ai-powered-web-search-mode-on-bing-how-it-works-125022500477_1.html

Query: What real-world applications integrate web-search or browsing tools into LLM agents (e.g., Bing Copilot, Perplexity AI, or Google’s Search-enabled Gemini) to fetch up-to-date information for users?

Answer: Microsoft has integrated a **Copilot AI-powered web search mode** into Bing, allowing users to receive more personalized and context-aware results. When users select Copilot Search, Bing's AI refines and expands the original query, conducts multiple related web searches, and then summarizes and links to relevant sources within the response. The system allows users to see the reasoning behind the AI's answer by providing a "See reasoning" button, detailing the steps and queries used. Copilot Search presents results as text summaries with embedded backlinks, and the interface also displays images, videos, and related suggested queries. This integration enables users to ask follow-up questions and interactively refine their search for up-to-date information.

Google is reportedly working on a similar "AI Mode" for its Search platform, aiming to offer comparable AI-powered web search capabilities in the near future.

-----

-----

-----

### Source [37]: https://gaper.io/perplexity-ai-vs-google-gemini-vs-chatgpt/

Query: What real-world applications integrate web-search or browsing tools into LLM agents (e.g., Bing Copilot, Perplexity AI, or Google’s Search-enabled Gemini) to fetch up-to-date information for users?

Answer: **Perplexity AI** integrates real-time **web search** capabilities, retrieving the latest information from the internet almost instantly. Its primary use is information retrieval based on current web content, making it effective for fact-based and up-to-date queries. Perplexity AI excels in delivering accurate and trustworthy answers quickly, though its responses are closely tied to the information found on the web and it is primarily text-focused.

**Google Gemini** leverages advanced LLMs with multimodal capabilities, integrating data from text, images, audio, and video. While it is capable of handling complex and diverse input, Gemini also accesses the web to ground its responses in current information, especially for queries requiring the latest facts or multimedia integration. This enables Gemini to answer a broader range of questions and provide richer, more nuanced responses.

Both platforms demonstrate real-world applications where LLM agents are directly connected to web search or browsing tools to fetch and summarize up-to-date information for users.

-----

-----

-----

### Source [38]: https://support.microsoft.com/en-us/topic/copilot-in-bing-our-approach-to-responsible-ai-45b5eae8-7466-43e1-ae98-b48f8ff8fd44

Query: What real-world applications integrate web-search or browsing tools into LLM agents (e.g., Bing Copilot, Perplexity AI, or Google’s Search-enabled Gemini) to fetch up-to-date information for users?

Answer: **Copilot in Bing** integrates state-of-the-art LLMs with web search to provide users with grounded and contextually relevant responses. When a user submits a prompt, the system combines the prompt, recent conversation history, a guiding metaprompt, and top search results as input to the LLM. This ensures that generated responses are contextualized and grounded in high-quality, current web content.

Responses are delivered as traditional web links, AI-generated summaries, images, and chat responses. Summaries and chat results that use web search data are accompanied by references and a "Learn more" section, linking to the original search results for user verification. The conversational interface allows users to refine their queries interactively, ask follow-up questions, and select from pre-written chat suggestions to further explore topics. This design supports real-world use by enabling dynamic, up-to-date information retrieval and interactive exploration within the Bing search environment.

-----

-----

-----

### Source [39]: https://www.byteplus.com/en/topic/560528

Query: What real-world applications integrate web-search or browsing tools into LLM agents (e.g., Bing Copilot, Perplexity AI, or Google’s Search-enabled Gemini) to fetch up-to-date information for users?

Answer: **Perplexity AI** is highlighted as a platform focused on delivering fact-based, context-aware answers, primarily using a hybrid of retrieval and generative AI. Its core function is to perform real-time searches on the web to fetch up-to-date information, which it then summarizes and presents to the user. 

**Google Gemini** is positioned as a multimodal AI solution, capable of processing and integrating text, images, and other media formats. It leverages Google's search infrastructure and LLM capabilities to deliver responses based on the latest available data from the web, supporting a wide range of complex, real-world queries.

**ChatGPT**, according to this source, is more focused on conversational AI and content generation, with less emphasis on real-time web search integration compared to Perplexity AI and Google Gemini. This comparison reinforces the real-world application of web-connected LLM agents for instant information retrieval and user assistance in platforms like Perplexity AI and Google’s Gemini.

-----

-----

</details>

<details>
<summary>In current LLM workflows, how are Pydantic models leveraged to validate and enforce structured outputs, and what examples show them being exposed to the model as callable “tools”?</summary>

### Source [40]: https://pydantic.dev/articles/llm-validation

Query: In current LLM workflows, how are Pydantic models leveraged to validate and enforce structured outputs, and what examples show them being exposed to the model as callable “tools”?

Answer: Pydantic models are used to enforce and validate structured outputs from LLMs by defining explicit validation rules, including integrating LLM-driven validators for complex or contextual requirements. The article demonstrates using the `instructor` package, which patches OpenAI clients to leverage Pydantic's validation capabilities. A key feature is the use of `llm_validator`, which lets developers specify validation rules in natural language (e.g., "don't say objectionable things") and have the LLM generate a validator function accordingly.

For example, you can annotate a string field with a Pydantic validator that uses an LLM to check for objectionable content. If the LLM-generated output violates the rule, the validation fails with an error generated by the LLM. This mechanism allows developers to enforce nuanced content moderation or business logic through natural language, which is otherwise difficult to encode directly in standard Python validators.

The approach allows for the enforcement of both strict, schema-based constraints and more flexible, context-dependent validations, all within the Pydantic validation framework. The error messages are informative and generated dynamically based on the LLM’s analysis, providing actionable feedback when validation fails.

-----

-----

-----

### Source [41]: https://arxiv.org/html/2505.03049v2

Query: In current LLM workflows, how are Pydantic models leveraged to validate and enforce structured outputs, and what examples show them being exposed to the model as callable “tools”?

Answer: Pydantic models are utilized to exchange structured atomic information between LLMs and tools in agentic workflows. The agent, powered by an LLM, is augmented with tools for domain-specific decision-making. Here, Pydantic models serve as a schema for both the LLM outputs and the inputs to the tools, ensuring that the data exchanged is well-structured and validated. This facilitates seamless integration between LLM-generated suggestions and downstream tools, especially in scientific applications like materials science, where the agent can leverage Pydantic models to maintain data integrity across various automated steps.

-----

-----

-----

### Source [42]: https://xebia.com/blog/enforce-and-validate-llm-output-with-pydantic/

Query: In current LLM workflows, how are Pydantic models leveraged to validate and enforce structured outputs, and what examples show them being exposed to the model as callable “tools”?

Answer: Pydantic is used in LLM workflows to enforce and validate output by defining data models that specify the expected structure and permissible values. For example, a Pydantic model can define a field with a restricted set of allowed string values using `Literal`. When the LLM returns output, Pydantic validates it against the model. If the output does not conform (for example, the value is not among the allowed options), a validation error is raised.

This validation step can be performed immediately after parsing the LLM’s response, providing immediate feedback and ensuring that downstream processes receive only well-formed and compliant data. This method grants developers greater control over LLM outputs, improves robustness, and helps prevent unwanted or unexpected results from propagating through the system.

-----

-----

-----

### Source [43]: https://dylancastillo.co/posts/agentic-workflows-langgraph.html

Query: In current LLM workflows, how are Pydantic models leveraged to validate and enforce structured outputs, and what examples show them being exposed to the model as callable “tools”?

Answer: In agentic workflows, such as those built with LangGraph, Pydantic models are used to define the state and data structures that the LLM and other tools operate on throughout the workflow. Each node in the workflow (representing a function, tool, or model) expects input and produces output adhering to these Pydantic models.

For example, a node might evaluate text appropriateness by invoking an LLM with a structured output format defined by a Pydantic model. Another node might aggregate results, also structured by a Pydantic model. The LLM is called with prompts, and its structured response is parsed and validated using the predefined models. This approach ensures that each step in the workflow receives validated, well-structured data, reducing errors and simplifying the integration between LLMs and other components.

-----

-----

-----

### Source [44]: https://pydantic.dev/articles/llm-intro

Query: In current LLM workflows, how are Pydantic models leveraged to validate and enforce structured outputs, and what examples show them being exposed to the model as callable “tools”?

Answer: Pydantic models provide a mechanism for validating structured outputs from language models, such as those from OpenAI. By defining strict schemas with Pydantic, developers can ensure that the outputs from LLMs match the expected format and data types. This not only catches errors early but also increases the reliability and predictability of integrating LLMs into production systems. The article highlights the importance of using Pydantic for robust validation, ensuring that LLM outputs are both syntactically and semantically correct before being consumed by downstream processes or tools.

-----

-----

-----

### Source [135]: https://pydantic.dev/articles/llm-validation

Query: In current LLM workflows, how are Pydantic models leveraged to validate and enforce structured outputs, and what examples show them being exposed to the model as callable “tools”?

Answer: Pydantic models are used to validate and enforce structured outputs from LLMs by defining schemas that the LLM outputs must conform to, and by attaching custom validation logic—including validators that leverage LLMs themselves. The article describes an approach where Pydantic's `BeforeValidator` is used to attach an LLM-powered validator to a model field. For example, a field can be annotated with a validator that checks if the LLM's answer contains objectionable content:

```python
NoEvil = Annotated[
    str,
    BeforeValidator(
        llm_validator("don't say objectionable things", openai_client=client)
    )
]

class QuestionAnswer(BaseModel):
    question: str
    answer: NoEvil
```

If the model receives an output that violates the rule (e.g., contains objectionable content), it raises a validation error, with the error message generated by the LLM itself. This allows developers to express complex validation logic in natural language and have the LLM both generate and enforce output constraints. The approach illustrates how Pydantic models, when combined with LLM-based validators, can act as callable “tools” exposed to the model for runtime validation and control of outputs.

-----

-----

-----

### Source [136]: https://xebia.com/blog/enforce-and-validate-llm-output-with-pydantic/

Query: In current LLM workflows, how are Pydantic models leveraged to validate and enforce structured outputs, and what examples show them being exposed to the model as callable “tools”?

Answer: Pydantic is leveraged to enforce structured outputs by defining explicit types and constraints in the models that the LLM's output must match. For instance, if a model defines a field using a `Literal` type (e.g., `"easy"`, `"medium"`, `"hard"` for a `difficulty` field), Pydantic will automatically raise a validation error if the LLM outputs a value not in the allowed set:

```python
Difficulty = Literal["easy", "medium", "hard"]

class ThoughtAnswerResponse(BaseModel):
    thought: str
    answer: str
    difficulty: Difficulty
```

If the LLM outputs an invalid value for `difficulty`, such as `"Unknown"`, Pydantic validation fails and provides a detailed error. This mechanism ensures that only strictly validated, structured outputs proceed further in the workflow. The article emphasizes that combining Pydantic with prompt engineering provides robust control and validation over LLM outputs, making it easier to build reliable AI systems.

-----

-----

-----

### Source [137]: https://blog.kusho.ai/from-chaos-to-order-structured-json-with-pydantic-and-instructor-in-llms-part-ii/

Query: In current LLM workflows, how are Pydantic models leveraged to validate and enforce structured outputs, and what examples show them being exposed to the model as callable “tools”?

Answer: Pydantic, in conjunction with libraries such as Instructor, is used to validate LLM responses and enforce structured output by checking both structure and content types. The workflow typically involves:

- Defining a Pydantic model representing the desired output schema.
- Using the Instructor library to patch the LLM client so that responses are automatically validated against the Pydantic model.
- Implementing a retry mechanism (e.g., `max_retries`) so that if the LLM's output fails validation, the model is prompted to regenerate the output until the response matches the schema.

This setup ensures that the LLM not only produces structured JSON outputs but also that these outputs are reliable and fit-for-purpose. Additional “smart validation tricks,” such as custom validators within Pydantic models, further refine the checking process, allowing for sophisticated output enforcement and correction loops.

-----

-----

-----

### Source [138]: https://www.deeplearning.ai/short-courses/pydantic-for-llm-workflows/

Query: In current LLM workflows, how are Pydantic models leveraged to validate and enforce structured outputs, and what examples show them being exposed to the model as callable “tools”?

Answer: Official training materials highlight that Pydantic models are central in validating and enforcing structured outputs from LLMs at multiple workflow stages. These stages include validating user input, checking LLM responses, and defining parameters for tool-calling. Using Pydantic, developers can guarantee that LLM responses conform to expected formats, making the outputs reliably processable by downstream applications. The material also covers how Pydantic models are used to define the schema for callable tools—functions or APIs that the LLM can invoke with structured arguments validated by Pydantic. This approach ensures that all tool parameters and responses are robustly typed and validated, supporting error handling and reliability in tool-augmented LLM applications.

-----

-----

-----

### Source [139]: https://dev.to/devasservice/a-practical-guide-on-structuring-llm-outputs-with-pydantic-50b4

Query: In current LLM workflows, how are Pydantic models leveraged to validate and enforce structured outputs, and what examples show them being exposed to the model as callable “tools”?

Answer: This guide describes a practical workflow where Pydantic models are used to validate LLM outputs, ensuring strong structure enforcement. The process involves:

- Defining a Pydantic model that specifies the desired output structure.
- Validating the LLM's output against this model.
- Implementing a retry mechanism: if validation fails, the prompt is adjusted or retried until the output passes validation.

The article emphasizes that structured validation with Pydantic is crucial in production LLM systems, enabling predictable outputs, stronger data pipelines, and easier debugging. By integrating Pydantic validation layers, developers can enforce schemas and catch errors early, leading to more reliable and maintainable AI workflows.

-----

-----

</details>

<details>
<summary>What fundamental limitations prevent large language models from accessing real-time information or executing external actions on their own, and how do AI researchers justify the introduction of “tools” (function calling, APIs) to overcome these gaps?</summary>

### Source [45]: https://www.projectpro.io/article/llm-limitations/1045

Query: What fundamental limitations prevent large language models from accessing real-time information or executing external actions on their own, and how do AI researchers justify the introduction of “tools” (function calling, APIs) to overcome these gaps?

Answer: Large language models (LLMs) face several **fundamental limitations** that prevent them from independently accessing real-time information or executing external actions. Key issues include:

- **Limited Knowledge Update**: LLMs are trained on static datasets and cannot update their knowledge in real time. Once trained, their knowledge remains fixed until retraining occurs, which means they cannot access or process events or data that emerge after their last training cut-off.
- **Computational Constraints**: LLMs have strict limits on how much information (tokens) they can process at once. This restricts their ability to handle large or dynamic datasets and maintain context over extended interactions.
- **Lack of Long-Term Memory**: They do not possess persistent memory of past interactions or the ability to recall information outside their current session.
- **Struggles with Complex Reasoning**: Their responses are based on learned patterns, not on dynamic reasoning or real-time data evaluation.

To **overcome these gaps**, researchers introduce "tools"—such as function calling and APIs—that allow LLMs to access up-to-date information and perform actions by interfacing with external systems. These tools effectively augment LLMs, enabling functionalities like retrieving real-time data or executing specific commands, which pure language models cannot perform on their own[1].

-----

-----

-----

### Source [46]: https://arxiv.org/html/2504.14872v1

Query: What fundamental limitations prevent large language models from accessing real-time information or executing external actions on their own, and how do AI researchers justify the introduction of “tools” (function calling, APIs) to overcome these gaps?

Answer: LLM-driven agents **rely on external user functions (tools)** to expand their capabilities because of two major limitations:

- **Lack of automatic parallel orchestration**: LLMs cannot independently schedule or execute multiple actions in parallel, requiring manual orchestration.
- **Inefficient handling of resource constraints**: In complex systems (like AI-powered operating systems or robotic systems), LLMs alone cannot efficiently manage the scheduling and execution of numerous I/O- or compute-intensive tasks.

Researchers justify the introduction of tools (including function calling and APIs) by noting that these tools allow LLMs to organize, schedule, and execute external actions in a way that compensates for their inherent inability to interact autonomously with the outside world. Tools like LLMOrch are designed to manage these interactions efficiently, enabling LLMs to offload tasks, access current data, and perform actions that pure language modeling cannot achieve[2].

-----

-----

-----

### Source [47]: https://memgraph.com/blog/llm-limitations-query-enterprise-data

Query: What fundamental limitations prevent large language models from accessing real-time information or executing external actions on their own, and how do AI researchers justify the introduction of “tools” (function calling, APIs) to overcome these gaps?

Answer: LLMs face significant **limitations in accessing and processing real-time or external data** due to:

- **Context window limitations**: LLMs can only process information within a fixed token limit, making them unable to handle large-scale or real-time data directly.
- **Lack of real-time updates**: Once trained, LLMs cannot incorporate new information without retraining, so they cannot access or respond to the latest data or events.
- **Absence of human-like reasoning and dynamic data access**: LLMs predict text based on training data patterns and cannot reason over or query external databases or APIs for fresh insights.

AI researchers introduce tools (function calling, APIs) to bridge these gaps. These tools allow LLMs to **query up-to-date information**, interact with external databases, and perform specialized actions that standard LLMs cannot execute. This approach enables LLMs to provide relevant, current answers that would otherwise be impossible due to their static nature and context limitations[3].

-----

-----

-----

### Source [48]: https://www.edpb.europa.eu/system/files/2025-04/ai-privacy-risks-and-mitigations-in-llms.pdf

Query: What fundamental limitations prevent large language models from accessing real-time information or executing external actions on their own, and how do AI researchers justify the introduction of “tools” (function calling, APIs) to overcome these gaps?

Answer: The modular design of modern AI systems, including LLMs, enables them to **handle complex tasks and interact dynamically with their environment** through specialized modules. Examples of such modules include:

- **Perception modules**: These process and format external inputs so the LLM can understand and reason about them.
- **Reasoning modules**: These help the LLM to analyze structured data or respond to dynamic environments.

The introduction of these modular tools (including function calling and APIs) is justified as it allows LLMs to **overcome their inherent static and isolated nature**. By integrating with external modules, LLMs can process real-time inputs, execute actions, and refine their performance iteratively—capabilities that are unattainable for standalone LLMs[4].

-----

-----

</details>

<details>
<summary>How can a developer implement LLM function (tool) calling entirely from scratch in Python—defining JSON schemas, parsing the model’s function call, executing the function, and returning results—without relying on LangChain or similar frameworks?</summary>

### Source [49]: https://python.langchain.com/docs/how_to/function_calling/

Query: How can a developer implement LLM function (tool) calling entirely from scratch in Python—defining JSON schemas, parsing the model’s function call, executing the function, and returning results—without relying on LangChain or similar frameworks?

Answer: Although this source focuses on LangChain, it explains the *general architecture* of tool/function calling with LLMs, which is applicable even when not using LangChain or similar frameworks. The process works as follows:

- **Define a JSON schema** for each function/tool you want the LLM to access. The schema specifies the function’s name and the structure of its arguments as a dictionary (`{argument_name: argument_value}`).
- **Provide these schemas to the LLM** as part of your prompt or API request. Many LLM providers (such as OpenAI, Anthropic, Google, etc.) support variants of this "tool calling" feature, where you send the list of available tools and their JSON schemas with the LLM request.
- When the user issues a prompt, the LLM responds with a *function call specification*, which includes:
  - The name of the function/tool to invoke
  - The values for each argument as specified in the schema
  - (Optionally) an identifier for the call instance
- **Parse the model's function call output**: Extract the name and arguments from the model's response (often provided as a JSON object or structured dictionary).
- **Execute the corresponding function**: In your Python code, map the function name to the actual Python function and call it using the parsed arguments.
- **Return the results**: Format the result as needed (e.g., as JSON) and optionally pass it back to the LLM for further reasoning or to generate a final answer to the user.

This process allows you to build your own tool-calling pipeline from scratch, relying only on standard Python and JSON libraries for parsing and execution. The model is only responsible for suggesting which function to call and with what arguments; your code handles the actual execution and result passing.

-----

-----

-----

### Source [51]: https://github.com/rasbt/LLMs-from-scratch

Query: How can a developer implement LLM function (tool) calling entirely from scratch in Python—defining JSON schemas, parsing the model’s function call, executing the function, and returning results—without relying on LangChain or similar frameworks?

Answer: This official code repository provides comprehensive resources for building, pretraining, and fine-tuning a GPT-like LLM from scratch in Python. Although the focus is on building the LLM and not specifically on function calling, the following concepts are relevant:

- **You have full control over the input and output formats** of your LLM when building it from scratch. This means you can design the output template to match a function call specification (e.g., a JSON object describing the function and its arguments).
- **Integration with Python code:** Once your model is trained to output structured function calls, you can write Python code to:
  - Parse the output (using the `json` module)
  - Map function names to Python functions (using a dictionary or similar structure)
  - Dynamically call the function with the provided arguments (using `getattr`, `globals()`, or a custom registry)
  - Return and format the results as needed

The repository does not include a function-calling pipeline out of the box, but by following their guidance on I/O handling, you can implement your own.

-----

-----

</details>

<details>
<summary>Which studies or technical reports analyze failure modes where LLM agents repeatedly invoke tools in naive loops, and how do their findings motivate more sophisticated patterns such as ReAct or Chain-of-Thought with reflection?</summary>

### Source [59]: https://www.promptingguide.ai/techniques/react

Query: Which studies or technical reports analyze failure modes where LLM agents repeatedly invoke tools in naive loops, and how do their findings motivate more sophisticated patterns such as ReAct or Chain-of-Thought with reflection?

Answer: ReAct, proposed by Yao et al. (2022), addresses a common failure mode where LLM agents, when naively using tools, may get stuck in repetitive or ineffective action loops due to a lack of explicit reasoning. By prompting LLMs to interleave **reasoning traces** (verbal explanations of thought processes) with **actions** (tool invocations), ReAct encourages models to plan, track progress, and adapt strategies based on intermediate results.

This approach allows LLMs to:
- Induce and update action plans dynamically.
- Handle exceptions and unexpected outcomes.
- Interface more robustly with external tools or knowledge sources for improved factual accuracy.

Empirical results show ReAct outperforms baseline agents that lack explicit reasoning, especially in more complex or multi-step tasks. The findings motivate more sophisticated agent patterns—like combining ReAct with Chain-of-Thought (CoT) reasoning—because the mixture of explicit reasoning and action-taking helps avoid simplistic, repetitive tool use and improves both reliability and interpretability[1].

-----

-----

-----

### Source [60]: https://aman.ai/primers/ai/agents/

Query: Which studies or technical reports analyze failure modes where LLM agents repeatedly invoke tools in naive loops, and how do their findings motivate more sophisticated patterns such as ReAct or Chain-of-Thought with reflection?

Answer: Reflection is highlighted as a crucial strategy for addressing LLM agent shortcomings, including naive, repetitive tool usage. By integrating a **reflective mechanism**, LLMs can self-evaluate their outputs, recognize gaps or inefficiencies (such as unnecessary loops), and iteratively refine their behavior.

Key aspects of the reflection pattern:
- The model assesses its performance and output quality.
- It uses feedback—either external or self-generated—to prompt improvements.
- The cycle of "generate, evaluate, refine" transforms static, one-shot LLM behavior into an adaptive, self-improving process.

This reflective workflow has been shown to notably enhance output relevance and efficiency, making agents less likely to persist in unproductive patterns. The report argues that such reflective, meta-cognitive processes are foundational for sophisticated, robust agentic behaviors, directly motivating advanced patterns like ReAct and Chain-of-Thought with Reflection[2].

-----

-----

-----

### Source [61]: https://www.autonoly.com/blog/685e784a08412e725c1d0f4c/chain-of-thought-react-and-reflection-the-complete-guide-to-ai-agent-reasoning-patterns

Query: Which studies or technical reports analyze failure modes where LLM agents repeatedly invoke tools in naive loops, and how do their findings motivate more sophisticated patterns such as ReAct or Chain-of-Thought with reflection?

Answer: This guide details the **Reflection** reasoning pattern as an explicit solution to agent failure modes, including repetitive or naive tool invocation loops. Reflection enables AI agents to:
- Critically examine their own reasoning and performance.
- Identify incomplete, illogical, or inefficient steps.
- Adjust their approach in real-time or for future tasks.

Reflection is structured as a cycle:
1. **Generate:** Produce an initial answer or tool call.
2. **Reflect:** Critically assess the reasoning and outcome.
3. **Regenerate:** Improve the answer or strategy based on insights.

Types of reflection include:
- **Performance Reflection:** Did the approach solve the problem efficiently, or did it get stuck in loops?
- **Process Reflection:** Was the sequence of steps logical and comprehensive?
- **Outcome Reflection:** Did the actions achieve the intended goals?

By embedding this meta-cognitive loop, agents can break out of naive patterns, learn from mistakes, and iterate towards more effective strategies—providing a clear rationale for adopting advanced agentic patterns beyond naive chaining[3].

-----

-----

-----

### Source [62]: https://arxiv.org/html/2507.02097v1

Query: Which studies or technical reports analyze failure modes where LLM agents repeatedly invoke tools in naive loops, and how do their findings motivate more sophisticated patterns such as ReAct or Chain-of-Thought with reflection?

Answer: This technical report discusses the limitations of fixed, chained workflows—where agents break tasks into sequential subtasks or tool calls—highlighting that such approaches often lack true autonomy and iterative decision-making. While these systems can act as a form of memory or stepwise execution, they are prone to breakdowns when faced with complex or dynamic environments, including getting stuck in unproductive loops.

True agentic behavior is defined by an internal loop of **observe → decide → act**, iteratively refining approach and incorporating new information or feedback until goals are met. The report asserts that more sophisticated, iterative agentic patterns—such as those involving reasoning, reflection, or dynamic re-planning—are necessary to avoid the failure modes of simplistic chaining and to fully utilize LLM capabilities for robust, autonomous performance[4].

-----

-----

-----

### Source [63]: https://www.promptingguide.ai/techniques/reflexion

Query: Which studies or technical reports analyze failure modes where LLM agents repeatedly invoke tools in naive loops, and how do their findings motivate more sophisticated patterns such as ReAct or Chain-of-Thought with reflection?

Answer: Reflexion is presented as a paradigm for improving agentic behavior by combining **agent memory encoding** with adaptive LLM parameters, effectively operationalizing verbal reinforcement and self-assessment. Reflexion enables agents to remember outcomes and decisions, then use this internal record to modify future actions, thus avoiding repeated mistakes or ineffective loops.

This self-improving design directly targets the failure mode of naive, repeated tool invocation; by reflecting on past actions and their outcomes, agents become capable of breaking out of unproductive cycles and adopting more sophisticated, context-aware strategies—demonstrating the practical motivation for the Reflexion (reflection) pattern in agent design[5].
-----

-----

</details>

<details>
<summary>What are the most common categories of external tools integrated into production LLM agents (e.g., retrieval/RAG, web search, database querying, code execution), and what documented industry case studies illustrate their practical benefits?</summary>

### Source [64]: https://www.mercity.ai/blog-post/guide-to-integrating-tools-and-apis-with-language-models

Query: What are the most common categories of external tools integrated into production LLM agents (e.g., retrieval/RAG, web search, database querying, code execution), and what documented industry case studies illustrate their practical benefits?

Answer: This guide discusses the integration of external tools and APIs with language models such as GPT-4, LLaMa 70B, and Falcon 180B. The most common categories of external tools integrated into production LLM agents include:

- **Retrieval-augmented generation (RAG) systems:** These allow LLMs to access and utilize external knowledge sources, such as private databases or document stores, to ground their responses in up-to-date or proprietary information.
- **Web search:** LLMs can be connected to real-time internet search APIs to retrieve the latest information or verify facts.
- **Database querying:** Integration with databases enables LLMs to answer questions based on structured data or perform analytics.
- **Code execution:** LLMs can be linked to code interpreters or execution environments, allowing them to perform computations, data transformations, or automate programming tasks.

The article emphasizes that the LLM serves as the interface for tools, and integration is accomplished via function calling and prompting techniques. Larger models are generally preferred due to better multi-step reasoning and instruction following. Fine-tuning smaller models using techniques like Prefix Tuning and IA3 can also yield good results for specific tool use scenarios.

No specific industry case studies are provided, but the guide notes that proprietary tool integration is a common need, and describes the technical approaches and benefits of such integrations, like increased productivity and more sophisticated automation.

-----

-----

-----

### Source [65]: https://www.leewayhertz.com/advanced-rag/

Query: What are the most common categories of external tools integrated into production LLM agents (e.g., retrieval/RAG, web search, database querying, code execution), and what documented industry case studies illustrate their practical benefits?

Answer: This source outlines advanced retrieval-augmented generation (RAG) architectures and their integration with LLM agents. The most common category detailed is **multi-document retrieval**—where LLM agents are enhanced with features such as:

- **Chat history management**
- **Knowledge storage**
- **Document uploading interfaces**
- **Function-calling APIs** (where natural language is converted into actionable commands)

A documented case is OpenAI’s assistants, which integrate these features to extend LLM capabilities. In a practical setup, each document is managed by its own agent (for tasks like summarization and Q&A), while a central agent orchestrates these document-specific agents to handle complex queries involving comparisons and synthesis across multiple sources.

Benefits cited include:
- The ability to perform complex, multi-step analyses
- Improved accuracy through context summarization and iterative refinement
- Enhanced user experience in applications requiring extensive document management and knowledge retrieval

While the article doesn’t cite specific industry deployments by name, it details architectures and techniques used in production-grade retrieval systems, particularly within enterprise knowledge management and customer support platforms.

-----

-----

-----

### Source [66]: https://mirascope.com/blog/llm-integration

Query: What are the most common categories of external tools integrated into production LLM agents (e.g., retrieval/RAG, web search, database querying, code execution), and what documented industry case studies illustrate their practical benefits?

Answer: This guide provides an overview of LLM integration with external tools and APIs, highlighting several core categories:

- **APIs for data retrieval:** LLMs connect to APIs to fetch real-time information, such as weather, news, or financial data.
- **Action APIs:** LLMs can trigger workflows, perform transactions, or control devices through integration with action-oriented APIs.
- **Multi-step task execution:** LLM agents interpret complex instructions, breaking them down and using various tools to achieve sophisticated goals.

The article also discusses the necessity of robust software design to handle challenges such as non-deterministic LLM responses, network reliability, and vendor lock-in. Error handling, prompt versioning, and observability are identified as essential components for production use.

Real-world examples are referenced in general terms, such as applications for automated content generation, workflow automation, and customer-facing chatbots that leverage LLMs with external tools for enhanced capabilities. Specific industry case studies are not named, but the described patterns reflect common best practices in sectors like enterprise automation, customer service, and operations.

-----

-----

</details>

<details>
<summary>How can a developer implement LLM function calling from scratch in Python by defining JSON schemas, parsing the model's output, and executing the corresponding function, without using a library like LangChain or Gemini's native tooling?</summary>

### Source [68]: https://docs.x.ai/docs/guides/function-calling

Query: How can a developer implement LLM function calling from scratch in Python by defining JSON schemas, parsing the model's output, and executing the corresponding function, without using a library like LangChain or Gemini's native tooling?

Answer: A developer can implement LLM function calling by first defining local Python functions that perform the desired operations (e.g., fetching weather data or performing calculations). Each function should be mapped to a string name so the system can call the right function when the LLM requests it. Function parameters are specified and sent as part of the initial LLM prompt, typically using a JSON schema to specify parameter types and requirements. For example, you can define the function schema as a Python dictionary or use Pydantic for more type safety.
When the LLM returns a response, the developer checks if the output contains a tool call (an RPC-style request from the LLM). If so, the developer parses the model’s output, extracts the function name and arguments, and invokes the corresponding local Python function using those arguments. The result is then sent back to the LLM for further processing or user-facing output. The process involves:
- Defining function schemas (as JSON or with Pydantic).
- Mapping function names to actual Python functions.
- Parsing LLM output for tool/function calls.
- Executing the function with the provided arguments.
- Returning function results to the LLM for continued dialogue or user response.

[The guide provides Python pseudocode and emphasizes the importance of maintaining a clear function mapping and robust output parsing to ensure the correct function is executed.][1]

-----

-----

-----

### Source [69]: https://blog.christoolivier.com/p/llms-and-functiontool-calling

Query: How can a developer implement LLM function calling from scratch in Python by defining JSON schemas, parsing the model's output, and executing the corresponding function, without using a library like LangChain or Gemini's native tooling?

Answer: The process for implementing LLM function calling from scratch is as follows:
1. **Define a set of functions** (with schemas) and provide them to the LLM as part of your initial request.
2. **Receive the LLM's response**, which may include a function call with arguments. The arguments are typically a stringified JSON object that matches your schema—though the LLM may hallucinate or make mistakes, so validation is critical.
3. **Parse the LLM output** to extract the function name and arguments, deserialize the JSON arguments, and then call the relevant Python function with those arguments.
4. **Send the function result back** to the LLM as part of a new prompt, allowing the LLM to summarize or use the information as needed.
Key points include the need for robust code to handle possible LLM errors in argument formatting, and the awareness that this process does not require any special library—just standard Python JSON handling and function dispatch.

[This approach is described as “good old fashioned software development,” with the only LLM-specific aspect being the model’s ability to suggest function names and arguments in response to user prompts.][2]

-----

-----

-----

### Source [70]: https://www.promptingguide.ai/applications/function_calling

Query: How can a developer implement LLM function calling from scratch in Python by defining JSON schemas, parsing the model's output, and executing the corresponding function, without using a library like LangChain or Gemini's native tooling?

Answer: To implement LLM function calling:
- **Define your function schema**: Describe your function (such as `get_current_weather`) using a JSON schema, specifying the function name, description, parameter types, and required fields.
- **Send the schema with your prompt**: When calling the LLM API, include the list of available function schemas (as JSON) so the model knows which functions it may invoke and what arguments are expected.
- **Parse the model’s output**: If the model chooses to call a function, it will output a JSON object with the function name and arguments. Your application code must parse this JSON, validate the arguments (e.g., ensure required fields are present and types are correct), and then dispatch the call to your actual Python function.
- **Pass the function output to the model**: Return the function result to the LLM as a new message, so it can continue the conversation or summarize the result for the user.

[This method ensures the LLM can interact with external APIs or tools by tightly coupling function definitions with clear, machine-readable schemas and robust output parsing.][3]

-----

-----

-----

### Source [71]: https://platform.openai.com/docs/guides/function-calling

Query: How can a developer implement LLM function calling from scratch in Python by defining JSON schemas, parsing the model's output, and executing the corresponding function, without using a library like LangChain or Gemini's native tooling?

Answer: The official OpenAI guide for function calling details a step-by-step workflow:
1. **Prepare the function schema** as a JSON object, including the function name, description, and parameters (using JSON Schema to describe types, enums, and requirements).
2. **Prompt the model** with both the user’s input and the list of available function schemas.
3. **Parse the response**: The model may return a special object indicating a function call, with arguments as a JSON string. Parse these arguments using Python’s `json.loads`.
4. **Execute the function**: Call the corresponding Python function with the parsed arguments.
5. **Return the result to the LLM**: Send the function result back to the model as a new message, typically using a special message type or structure (such as `function_call_output`).
6. **Let the model generate a user-facing answer**: The model incorporates the function output into its final response.

[The process emphasizes robust parsing and error handling, as malformed JSON or incorrect arguments may be returned by the model. All exchanges are structured as a sequence of messages, with clear separation between user input, model output, function call, and function result.][5]
-----

-----

-----

### Source [97]: https://blog.christoolivier.com/p/llms-and-functiontool-calling

Query: How can a developer implement LLM function calling from scratch in Python by defining JSON schemas, parsing the model's output, and executing the corresponding function, without using a library like LangChain or Gemini's native tooling?

Answer: A developer can implement LLM function calling from scratch in Python by following these steps, even if the LLM API does not natively support function calling:

- **Define Functions and JSON Schemas:** Start by specifying the functions you want the LLM to call, and create a custom JSON schema for each function. This schema outlines the expected parameters and their types.
- **Prompt the LLM:** Send the user query along with the function definitions and schemas as part of your prompt to the LLM.
- **Interpret Model Output:** The LLM may respond with a stringified JSON object indicating which function to call and what arguments to use (note: the model may hallucinate parameters or arguments).
- **Parse Output and Execute Function:** Parse the LLM’s output (e.g., using Python’s `json.loads`). If the output matches one of your defined schemas, call the corresponding Python function with the provided arguments.
- **Return Results to LLM:** After executing the function, append the result as a new message in the conversation history and prompt the LLM again to summarize or continue the interaction.

The process is essentially: prompt > parse response > execute function > return result > repeat as needed. It's important to anticipate and handle cases where the LLM may invent functions or pass incorrect arguments by validating and sanitizing the parsed data before execution[1].

-----

-----

-----

### Source [98]: https://www.promptingguide.ai/applications/function_calling

Query: How can a developer implement LLM function calling from scratch in Python by defining JSON schemas, parsing the model's output, and executing the corresponding function, without using a library like LangChain or Gemini's native tooling?

Answer: To implement LLM function calling in Python without libraries like LangChain or Gemini tooling, follow this practical approach:

- **Define Function and JSON Schema:** Create a Python function and a corresponding JSON schema describing the function interface, including parameter types, descriptions, and requirements. For example, a weather function might have `location` and `unit` as parameters.
- **Pass Schema to LLM:** When sending a prompt to the LLM, include the function schema in the request (if the API supports it), or embed it in the prompt so the model understands the available functions and their arguments.
- **Parse Model Output:** The model may return a structured response (often as a stringified JSON object) specifying which function to call and its arguments. Use Python’s JSON parsing tools to extract the relevant information.
- **Call Function and Handle Result:** Use the parsed arguments to execute the Python function and obtain the result.
- **Feed Results Back:** Send the function’s output to the LLM as a new message, allowing it to generate a user-facing answer or continue the workflow.

This process enables the LLM to act as an orchestrator, determining function calls and arguments, while your code handles the actual execution and result communication[2].

-----

-----

-----

### Source [99]: https://friendli.ai/blog/llm-function-calling

Query: How can a developer implement LLM function calling from scratch in Python by defining JSON schemas, parsing the model's output, and executing the corresponding function, without using a library like LangChain or Gemini's native tooling?

Answer: Function calling allows LLMs to interact with external APIs and modular functions for performing complex tasks. The developer’s implementation involves:

- **Expose Functions via Interface:** Make specific Python functions accessible, each with well-defined input arguments.
- **Provide Context and Function Descriptions:** When prompting the LLM, provide descriptions of available functions and their parameter schemas, typically using JSON format.
- **LLM Suggests Function Calls:** The LLM chooses which function to call and supplies arguments based on the context of the conversation.
- **Execute and Return Data:** Your application parses the suggested function call (usually formatted in JSON), executes the function (e.g., using Python code), and returns the result to the LLM.
- **Iterative Interaction:** The LLM may request further function calls or actions, creating an agentic workflow.

This architecture transforms LLMs into intelligent agents, with your code serving as the bridge between model output and actual execution, entirely independent of third-party libraries. Validation and error handling are crucial, as LLMs may output invalid or hallucinated function calls[3].

-----

-----

-----

### Source [100]: https://community.openai.com/t/how-does-function-calling-actually-work-for-the-assistants-api/641440

Query: How can a developer implement LLM function calling from scratch in Python by defining JSON schemas, parsing the model's output, and executing the corresponding function, without using a library like LangChain or Gemini's native tooling?

Answer: The function calling workflow, as outlined in OpenAI’s documentation, utilizes schemas similar to OpenAPI for defining functions:

- **Function Definition:** Specify functions in Python and describe their schema using JSON, including names, descriptions, parameter types, and required fields.
- **Schema Example:** A weather function might have a JSON schema specifying that `location` is a required string and `unit` is an optional string with enumeration.
- **Model Interaction:** The LLM is prompted with available function schemas and, upon receiving a relevant query, responds with a mock function call (formatted as JSON), specifying the function name and arguments.
- **Validation and Execution:** Developers must validate and clean the model’s output, ensuring arguments conform to expected types and handling errors gracefully before executing the function in Python.
- **Result Communication:** After execution, the result is sent back to the LLM, which may further process or summarize the answer.

This process is robust but requires careful validation and error handling to protect against model hallucinations or incorrect parameter values[4].

-----

-----

-----

### Source [121]: https://blog.christoolivier.com/p/llms-and-functiontool-calling

Query: How can a developer implement LLM function calling from scratch in Python by defining JSON schemas, parsing the model's output, and executing the corresponding function, without using a library like LangChain or Gemini's native tooling?

Answer: A developer can implement LLM function calling from scratch in Python by following these steps:

- **Define JSON schemas for each function** you wish the LLM to call. This schema describes the function's name, parameters, and their types.
- **Send the user's query and the function schemas to the LLM**. The LLM decides, based on the user query and the provided function definitions, whether to call one of the defined functions and, if so, generates a stringified JSON object with the function name and arguments.
- **Parse the LLM’s output** in your Python code. Extract the function name and arguments from the JSON string.
- **Execute the corresponding function** in your code, passing in the parsed arguments.
- **Return the function’s result to the LLM** as part of the next message, allowing the LLM to summarize or process the result as needed.

The process is iterative: after executing and returning the function result, you may prompt the LLM again, appending the function output to the conversation, so it can generate a final or follow-up response. Note that LLMs may hallucinate parameter values or structures, so your code should be robust against invalid or unexpected input.

This approach works with any LLM, even those that do not natively support function calling in their API or SDK[1].

-----

-----

-----

### Source [122]: https://www.promptingguide.ai/applications/function_calling

Query: How can a developer implement LLM function calling from scratch in Python by defining JSON schemas, parsing the model's output, and executing the corresponding function, without using a library like LangChain or Gemini's native tooling?

Answer: To implement LLM function calling manually in Python:

- **Define function schemas in JSON**. For example, for a weather function:
  ```json
  {
    "type": "function",
    "function": {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
          }
        },
        "required": ["location"]
      }
    }
  }
  ```
- **Pass these schemas to the LLM along with the user query**. Prompt the LLM to use the JSON schema for any function call it needs to perform.
- **Parse the LLM output** by extracting and deserializing the JSON function call and its parameters.
- **Execute the relevant Python function** using the extracted arguments.
- **Send the function’s output back to the LLM** (if you want it to summarize or act further).

This approach is compatible with any LLM and provides a standard method for connecting LLM text output to Python code without relying on wrapper libraries[3].

-----

-----

-----

### Source [123]: https://www.apideck.com/blog/llm-tool-use-and-function-calling

Query: How can a developer implement LLM function calling from scratch in Python by defining JSON schemas, parsing the model's output, and executing the corresponding function, without using a library like LangChain or Gemini's native tooling?

Answer: The core steps for implementing LLM function calling from scratch are:

- **User Query**: The user asks a question or requests an action.
- **LLM Processing**: The LLM analyzes the query and, based on the prompt and any provided function schemas, decides if an external function or API needs to be called.
- **Function Call Decision**: The LLM outputs a description of the function to call, including the function name and arguments, typically in a structured (e.g., JSON) format.
- **Data Retrieval**: Your Python code parses this output, extracts the function and parameters, and executes the corresponding function or API call.
- **Data Integration**: The result of the function is returned (optionally appended to the conversation) so the LLM can use the new information to generate a user-facing response.

This process allows the LLM to offload specific tasks to code or external APIs and integrate the results into its responses, even without specialized libraries or tooling[4].

-----

-----

-----

### Source [124]: https://friendli.ai/blog/llm-function-calling

Query: How can a developer implement LLM function calling from scratch in Python by defining JSON schemas, parsing the model's output, and executing the corresponding function, without using a library like LangChain or Gemini's native tooling?

Answer: Function calling connects LLMs with external functions and APIs by:

- **Providing the LLM with a list of available functions (in JSON schema format)**, describing their names and required arguments.
- **Prompting the LLM to select and call the appropriate function** by returning a JSON object specifying the function name and arguments.
- **Parsing the LLM’s output in your code**, validating and extracting the function name and arguments.
- **Executing the corresponding code or API call** in Python, using the parsed arguments.
- **Returning the result to the LLM** as additional context, so it can produce a final answer or take further action.

Care should be taken to validate the arguments, as LLMs might generate invalid or unexpected data. This approach is foundational for building AI agents that can interact with software, databases, web APIs, and more, all without relying on frameworks like LangChain[5].

-----

-----

</details>

<details>
<summary>What are the primary security risks when giving an LLM agent access to a code execution tool, such as a Python interpreter, and what specific sandboxing or isolation strategies are recommended to mitigate these threats?</summary>

### Source [72]: https://www.legitsecurity.com/aspm-knowledge-base/llm-security-risks

Query: What are the primary security risks when giving an LLM agent access to a code execution tool, such as a Python interpreter, and what specific sandboxing or isolation strategies are recommended to mitigate these threats?

Answer: Primary security risks when giving an LLM agent access to a code execution tool include **prompt injection** and **sensitive information disclosure**. Prompt injection occurs when a user crafts inputs that manipulate the model’s instructions, potentially causing it to execute unintended actions, leak sensitive data, or produce harmful content. This risk increases when an LLM is allowed to interact with downstream systems such as code execution environments. Attackers can exploit weak input validation or context isolation to gain unauthorized access or trigger unintended functions.

**Sensitive information disclosure** can also result from LLMs regurgitating confidential or proprietary information embedded in training data or accessible through execution environments. Preventing these exposures requires both strict data governance and runtime output filtering.

To mitigate these risks, the source recommends **strong prompt isolation**, **input validation**, and **context enforcement**. These strategies help ensure that user-supplied inputs cannot subvert intended LLM behavior and that the LLM can only access authorized data and functions during code execution.

-----

-----

-----

### Source [73]: https://www.cobalt.io/blog/llm-failures-large-language-model-security-risks

Query: What are the primary security risks when giving an LLM agent access to a code execution tool, such as a Python interpreter, and what specific sandboxing or isolation strategies are recommended to mitigate these threats?

Answer: LLM agents with code execution capabilities face risks such as **privacy violation** and **operational disruption**. Privacy violations can occur if attackers use prompt injection to manipulate an LLM into accessing or leaking sensitive data, including personally identifiable information. For example, a compromised LLM could be tricked into extracting private data from a connected database and disseminating it through its outputs.

Operational disruption is another major threat: prompt injection and insufficient output sanitization can allow malicious code to be passed to other applications, potentially granting attackers broader access to systems, networks, or accounts. If input controls are lacking, attackers can submit resource-intensive prompts, leading to **denial-of-service** conditions.

To address these issues, robust **input/output sanitization**, **strict access control**, and **resource limitation** are recommended to prevent the LLM from being used as a vector for attacks on code execution tools and broader system infrastructure.

-----

-----

-----

### Source [74]: https://owasp.org/www-project-top-10-for-large-language-model-applications/

Query: What are the primary security risks when giving an LLM agent access to a code execution tool, such as a Python interpreter, and what specific sandboxing or isolation strategies are recommended to mitigate these threats?

Answer: The OWASP Top 10 for Large Language Model Applications identifies several key risks relevant to LLM agents with code execution access:

- **LLM01: Prompt Injection** – Crafted inputs can manipulate LLM behavior, leading to unauthorized access, data breaches, or compromised operations.
- **LLM02: Insecure Output Handling** – Failing to validate LLM outputs can enable downstream exploits, including remote code execution and system compromise.
- **LLM07: Insecure Plugin Design** – Plugins or tools that process untrusted LLM outputs without proper access control may be exploited for remote code execution.
- **LLM08: Excessive Agency** – Granting LLMs broad autonomy (such as unrestricted code execution) can result in unintended or harmful actions.

**Sandboxing and isolation strategies** recommended by OWASP include: 
- Rigorous **output validation and sanitization** to prevent execution of malicious code.
- **Access controls** and **least-privilege principles** for any tools or plugins invoked by the LLM.
- Limiting the LLM’s scope of agency and continuously monitoring its outputs and downstream effects to detect and block abnormal behavior or security violations.

-----

-----

-----

### Source [75]: https://www.exabeam.com/explainers/ai-cyber-security/llm-security-top-10-risks-and-7-security-best-practices/

Query: What are the primary security risks when giving an LLM agent access to a code execution tool, such as a Python interpreter, and what specific sandboxing or isolation strategies are recommended to mitigate these threats?

Answer: A primary risk when LLMs have access to code execution tools is **insecure output handling**, where the LLM generates content that can enable web attacks (e.g., XSS, CSRF, privilege escalation, or remote code execution). If an LLM’s output is processed by a software system without proper filtering, malicious payloads can compromise the application.

Recommended mitigation strategies include:

- **Robust output filtering and sanitization**: Implement mechanisms to detect and remove malicious code or sensitive information from LLM outputs before downstream processing.
- **Checks and validation**: Ensure that only safe, authorized code is executed by validating the intent and content of LLM-generated instructions.
- **Rate limiting and load balancing**: Protect code execution environments from denial-of-service attacks by restricting the rate and volume of execution requests.
- **Rigorous data validation**: Guard against training data poisoning, which can bias LLM outputs toward unsafe behaviors.

-----

-----

-----

### Source [76]: https://isc.upenn.edu/security/LLM-guide

Query: What are the primary security risks when giving an LLM agent access to a code execution tool, such as a Python interpreter, and what specific sandboxing or isolation strategies are recommended to mitigate these threats?

Answer: Key risks include **data leakage**, **flawed code generation**, and the potential for **malicious input** to cause harmful behavior. When an LLM is given access to code execution tools, it may inadvertently generate or execute code that is incorrect or insecure, leading to system compromise. Additionally, confidential data provided to the LLM could be retained or leaked if not properly isolated.

To mitigate such threats, the guidance emphasizes:

- **Human review of LLM-generated code and outputs** before execution, which helps catch errors or malicious payloads that automated systems might miss.
- **Strong isolation of execution environments**: Code generated or executed by the LLM should be sandboxed, restricting its access to only necessary resources and preventing lateral movement or data exfiltration.
- **Continuous monitoring** for abnormal or unauthorized activities within the execution environment.

These practices collectively reduce the likelihood and impact of attacks that exploit the LLM’s access to code execution tools.

-----

-----

### Source [106]: https://www.legitsecurity.com/aspm-knowledge-base/llm-security-risks

Query: What are the primary security risks when giving an LLM agent access to a code execution tool, such as a Python interpreter, and what specific sandboxing or isolation strategies are recommended to mitigate these threats?

Answer: **Primary security risks when giving an LLM agent access to a code execution tool include:**
- **Prompt Injection:** Attackers can manipulate the model via crafted inputs, causing it to override intended behaviors, execute unintended code, or reveal sensitive data. When an LLM is connected to powerful tools like a Python interpreter, prompt injection can trigger unauthorized or dangerous downstream actions.
- **Sensitive Information Disclosure:** LLMs can inadvertently expose private, proprietary, or personally identifiable information included in their responses, especially if output is not filtered or access controls are weak.

**Recommended mitigation strategies:**
- Implement **strong prompt isolation** and rigorous **input validation** to prevent prompt injection.
- Enforce **context isolation** so each user interaction is kept separate and cannot influence others.
- Apply **runtime output filtering** to block leakage of sensitive information.
- Ensure **strict data governance** both during model training and runtime.

These practices help maintain control over what the LLM can do with its code execution capabilities and what information it may expose[1].

-----

-----

-----

### Source [107]: https://www.cobalt.io/blog/llm-failures-large-language-model-security-risks

Query: What are the primary security risks when giving an LLM agent access to a code execution tool, such as a Python interpreter, and what specific sandboxing or isolation strategies are recommended to mitigate these threats?

Answer: **Key security risks related to LLM code execution:**
- **Privacy Violation:** Malicious prompt injection or unsanitized data may allow the LLM to access or leak sensitive customer, employee, or company data, especially if the LLM is permitted to execute code or interact with databases.
- **Operational Disruption:** Attackers may exploit code execution to gain control over application functionality, steal or delete data, or run malicious code that spreads across the network. Uncontrolled inputs can also lead to denial-of-service by exhausting resources.
- **Output Sanitization:** Failing to sanitize outputs before passing results to other systems can let malicious code propagate, causing broader security incidents.

**Mitigation recommendations:**
- Sanitize both **inputs and outputs** rigorously, especially when outputs are used by other applications.
- Restrict LLM permissions to the minimum required for its role to reduce the impact of a successful attack.
- Monitor and control the types of actions the LLM is authorized to perform, particularly when code execution tools are involved[2].

-----

-----

-----

### Source [108]: https://owasp.org/www-project-top-10-for-large-language-model-applications/

Query: What are the primary security risks when giving an LLM agent access to a code execution tool, such as a Python interpreter, and what specific sandboxing or isolation strategies are recommended to mitigate these threats?

Answer: **OWASP identifies these top risks for LLMs with code execution:**
- **LLM01: Prompt Injection:** Crafted prompts can lead to unauthorized access and execution of unintended code, risking data breaches and control over downstream systems.
- **LLM02: Insecure Output Handling:** If LLM outputs (e.g., code snippets) are not validated, executing them can compromise systems, leading to remote code execution or data exposure.
- **LLM07: Insecure Plugin Design:** Plugins (such as code execution tools) that process untrusted inputs without sufficient access control can enable severe exploits, including remote code execution.
- **LLM08: Excessive Agency:** Granting LLMs unchecked autonomy to act (e.g., execute arbitrary code) can result in severe unintended consequences.

**Recommended sandboxing or isolation strategies:**
- **Validate and sanitize LLM inputs and outputs** before any code is executed or used in critical systems.
- **Restrict and isolate execution environments**: Run code produced by LLMs in tightly controlled sandboxes with minimal privileges and no access to sensitive systems or data.
- **Implement strict access controls** on plugins and downstream tools.
- Apply **rate limiting and resource monitoring** to prevent denial-of-service and excessive resource consumption[3].

-----

-----

-----

### Source [109]: https://www.exabeam.com/explainers/ai-cyber-security/llm-security-top-10-risks-and-7-security-best-practices/

Query: What are the primary security risks when giving an LLM agent access to a code execution tool, such as a Python interpreter, and what specific sandboxing or isolation strategies are recommended to mitigate these threats?

Answer: **Security risks with LLM code execution tools:**
- **Insecure Output Handling:** LLM-generated content can lead to attacks like remote code execution, privilege escalation, or web vulnerabilities (XSS, CSRF), especially if outputs are processed without validation.
- **Training Data Poisoning:** Manipulated or malicious training data can bias outputs, potentially making LLMs generate insecure or malicious code.
- **Model Denial of Service:** Attackers can overwhelm LLMs with excessive requests, causing outages.

**Mitigation strategies:**
- Employ **robust output filtering and sanitization** to ensure generated code is safe before execution.
- Use **rigorous data validation** in both training and runtime phases.
- Implement **rate limiting** and **load balancing** to defend against denial-of-service attacks.
- **Isolate code execution** in sandboxes that limit access to the host system and sensitive resources[4].

-----

-----

-----

### Source [110]: https://arxiv.org/html/2412.15004v1

Query: What are the primary security risks when giving an LLM agent access to a code execution tool, such as a Python interpreter, and what specific sandboxing or isolation strategies are recommended to mitigate these threats?

Answer: While the full text is not visible, the abstract highlights several risks:
- **LLMs can introduce new vulnerabilities during code generation**, potentially inserting insecure constructs without the user's knowledge.
- **LLMs may miss clear vulnerabilities** or flag non-existent ones, meaning their outputs are not always reliable or secure.
- When LLM-generated code is executed directly, these security flaws can be exploited, making it critical to review and sandbox any code prior to execution.

**Mitigation strategies (inferred from context):**
- Implement **human-in-the-loop review** for LLM-generated code before execution.
- Use **sandboxing or isolation environments** that strictly control resource access, limit privileges, and prevent LLM-generated code from interacting with sensitive systems or data.
- Continuously monitor and audit execution environments for suspicious activity[5].

-----

-----

-----

### Source [130]: https://www.legitsecurity.com/aspm-knowledge-base/llm-security-risks

Query: What are the primary security risks when giving an LLM agent access to a code execution tool, such as a Python interpreter, and what specific sandboxing or isolation strategies are recommended to mitigate these threats?

Answer: Primary security risks when giving an LLM agent access to a code execution tool include **prompt injection** and **sensitive information disclosure**. Prompt injection attacks involve users crafting inputs that manipulate the model’s behavior, potentially tricking the LLM into revealing confidential data, executing unintended actions, or producing harmful content—especially if the LLM connects to downstream systems like a Python interpreter. Such vulnerabilities are common when implementation lacks security reviews and context isolation. To mitigate these threats, strong **prompt isolation**, **input validation**, and **context enforcement** are recommended. Sensitive information disclosure can also occur if the LLM unintentionally outputs private, proprietary, or personally identifiable information. Preventing this requires strict **data governance** during training and **runtime output filtering** to block sensitive details from leaking.

-----

-----

-----

### Source [131]: https://www.cobalt.io/blog/llm-failures-large-language-model-security-risks

Query: What are the primary security risks when giving an LLM agent access to a code execution tool, such as a Python interpreter, and what specific sandboxing or isolation strategies are recommended to mitigate these threats?

Answer: LLM security failures relevant to code execution tools include **privacy violations** and **operational disruption**. Privacy violations arise from attacks like prompt injection, where an LLM could be manipulated to access databases or perform unauthorized actions such as sending phishing emails. If the LLM is granted the ability to execute code or call functions, attackers may gain excessive control, enabling them to steal or delete data. Operational disruption can occur if attackers exploit the LLM’s code execution rights to flood the system with resource-intensive tasks, potentially causing service outages or network crashes. Risks are heightened if input/output is not sanitized before being passed to or from the LLM and connected applications.

-----

-----

-----

### Source [132]: https://owasp.org/www-project-top-10-for-large-language-model-applications/

Query: What are the primary security risks when giving an LLM agent access to a code execution tool, such as a Python interpreter, and what specific sandboxing or isolation strategies are recommended to mitigate these threats?

Answer: The OWASP Top 10 for LLM Applications highlights **Prompt Injection** (LLM01) as a critical risk: crafted inputs can cause unauthorized access, data breaches, and compromised decision-making, especially dangerous if the LLM is allowed to execute code. **Insecure Output Handling** (LLM02) is also significant: failing to validate LLM outputs can lead to downstream exploits, including arbitrary code execution that compromises systems and exposes data. **Insecure Plugin Design** (LLM07) is particularly relevant when LLMs interact with tools like Python interpreters; untrusted inputs and poor access control can result in remote code execution. **Excessive Agency** (LLM08) warns against granting LLMs unchecked autonomy, as this can lead to unintended, potentially dangerous outcomes. Recommended mitigation strategies include **output validation**, **access control**, and **sandboxing** to contain the impact of any exploit.

-----

-----

-----

### Source [133]: https://www.exabeam.com/explainers/ai-cyber-security/llm-security-top-10-risks-and-7-security-best-practices/

Query: What are the primary security risks when giving an LLM agent access to a code execution tool, such as a Python interpreter, and what specific sandboxing or isolation strategies are recommended to mitigate these threats?

Answer: A primary security concern is **insecure output handling**—LLM-generated content may contain malicious code that, if executed, can compromise receiving applications through attacks like remote code execution or privilege escalation. To mitigate, applications must implement **robust output filtering and sanitization**, removing sensitive or dangerous content before processing. Additional risks include **model denial of service** (DoS), where attackers overload the LLM with resource-heavy requests, potentially causing crashes or downtime. To address DoS, implement **rate limiting** (restrict the number of requests per user/IP) and **load balancing** (distribute load across model instances). **Training data poisoning** is also a concern and requires **rigorous data validation** and **integrity checking** to ensure model reliability.

-----

-----

-----

### Source [134]: https://isc.upenn.edu/security/LLM-guide

Query: What are the primary security risks when giving an LLM agent access to a code execution tool, such as a Python interpreter, and what specific sandboxing or isolation strategies are recommended to mitigate these threats?

Answer: Information security risks when allowing LLMs access to code execution tools include **data leakage**, **generation of flawed or malicious code**, and **potential for incorrect responses**. LLMs can inadvertently leak confidential information if user data is retained or used for further training. Generated code may be insecure or contain vulnerabilities, requiring **human review** to catch potential issues from malicious or erroneous inputs. The guidance emphasizes the need for **careful oversight**, **strict access controls**, and **sanitization** of both inputs and outputs when deploying LLMs with sensitive or critical systems.

-----

</details>

<details>
<summary>What are the most common categories of external tools integrated with production LLM agents, such as web search, database querying, and code execution, and what are some documented industry case studies that illustrate their practical benefits?</summary>

### Source [77]: https://arxiv.org/html/2507.08034v1

Query: What are the most common categories of external tools integrated with production LLM agents, such as web search, database querying, and code execution, and what are some documented industry case studies that illustrate their practical benefits?

Answer: The integration of external tools with LLMs is methodically categorized into four major types: **retrieval-augmented generation (RAG), code execution, database querying, and web search**. The architecture for such integration typically involves an ExternalServiceIntegrator, which manages a tools repository with detailed schema-like structures describing each tool’s capabilities, arguments, and usage. These schemas allow the LLM to be explicitly aware of each tool’s functions and how to invoke them. The system uses a RunMonitoring service to analyze user queries, extract intent, and determine which external tool is appropriate based on keywords or query complexity. For example, a weather-related query might trigger a weather data API, while a complex computation could prompt code execution. This structured approach enables LLMs to efficiently extend their capabilities to real-time information retrieval, complex computations, and domain-specific data queries.

-----

-----

-----

### Source [78]: https://www.mercity.ai/blog-post/guide-to-integrating-tools-and-apis-with-language-models

Query: What are the most common categories of external tools integrated with production LLM agents, such as web search, database querying, and code execution, and what are some documented industry case studies that illustrate their practical benefits?

Answer: Integrating external tools and APIs with LLMs typically involves function calling or prompting techniques, with the LLM serving as the user interface to these tools. **Common external tool integrations include web search APIs, private databases, code execution environments, and third-party business APIs**. Larger LLMs like GPT-4 are especially effective at managing multi-step tool interactions and maintaining context. Techniques such as Prefix Tuning and IA3 are used to fine-tune smaller models for better tool use. The integration pipeline generally involves designing prompts or API calls that direct the LLM to trigger the desired tool, process the result, and return a coherent response to the user. This approach is crucial for scenarios where up-to-date information, proprietary data, or computation is required.

-----

-----

-----

### Source [79]: https://arxiv.org/pdf/2507.08034

Query: What are the most common categories of external tools integrated with production LLM agents, such as web search, database querying, and code execution, and what are some documented industry case studies that illustrate their practical benefits?

Answer: The framework described in this paper, Athena, manages APIs for external tools to enable LLMs to provide **accurate, up-to-date, data-driven responses** across domains. The system supports external tools such as RAG for document retrieval, code execution tools for calculations or logic, and database/API querying for real-time or proprietary data. A notable industry case study highlighted is the use of LLMs that create and use bespoke Python tools for tasks in the Big-Bench benchmark, demonstrating performance parity with higher-cost models but at reduced inference costs. This two-phase approach (tool creation and tool use) allows LLMs to independently enhance their problem-solving capabilities, making them scalable and adaptable for production environments.

-----

-----

-----

### Source [81]: https://www.getdynamiq.ai/post/llm-agents-explained-complete-guide-in-2025

Query: What are the most common categories of external tools integrated with production LLM agents, such as web search, database querying, and code execution, and what are some documented industry case studies that illustrate their practical benefits?

Answer: Modern LLM agent frameworks are pivotal for integrating external tools. **LangChain** is widely used for chaining prompts, tool use, and memory, supporting agent-based workflows that mix web search, RAG, database queries, and code execution. **OpenAI’s Function Calling** API enables lightweight agentic behaviors, such as calling APIs or running code during conversations. **CrewAI** supports multi-agent collaboration for complex workflows, and **AutoGen** (Microsoft) orchestrates multi-agent task automation, often involving data workflows and multi-step reasoning. **Haystack Agents** focus on RAG for document-heavy domains. These frameworks illustrate the practical industry benefits of tool integration: delivering up-to-date answers, automating complex business tasks, and supporting domain-specific assistants in sectors like research, legal, and data analytics.

-----

-----

-----

### Source [111]: https://arxiv.org/html/2507.08034v1

Query: What are the most common categories of external tools integrated with production LLM agents, such as web search, database querying, and code execution, and what are some documented industry case studies that illustrate their practical benefits?

Answer: The integration of external tools with large language models (LLMs) can be categorized into four major types: **retrieval-augmented generation (RAG), code execution, API calling, and database querying**. The paper presents a framework where an ExternalServiceIntegrator manages a repository of available tools, each described with a schema specifying its functionalities, required parameters, and expected outputs. This explicit tool registration allows the LLM to recognize when a user query requires external data or computation. For example, a weather query triggers the weather data tool if keywords and intent match its schema. The framework also includes a MessageSubmission component for user interaction and a RunMonitoring service to decide when to call external tools by parsing user intent and keywords. This structured approach supports practical scenarios such as real-time information retrieval, calculations, or procedural tasks that exceed the LLM's internal capacities.

-----

-----

-----

### Source [112]: https://www.mercity.ai/blog-post/guide-to-integrating-tools-and-apis-with-language-models

Query: What are the most common categories of external tools integrated with production LLM agents, such as web search, database querying, and code execution, and what are some documented industry case studies that illustrate their practical benefits?

Answer: This guide details practical methods for integrating tools and APIs with LLMs, especially through **function calling, prompting techniques, and pipelines**. At the core is the LLM, which acts as an interface for tool and API use. Larger models like GPT-4 are preferred for tool integration due to superior instruction following and conversational memory, but smaller models (like LLaMa-13B) can be tuned for specific tool-use prompts. Techniques such as **Prefix Tuning and IA3** are mentioned for adapting smaller models to follow tool-use instructions. The document emphasizes that the integration pipeline relies on the model's ability to interpret prompt signals for when to invoke external tools, and that effective integration boosts productivity and utility across a wide range of applications, from accessing web search to automating workflows.

-----

-----

-----

### Source [113]: https://arxiv.org/pdf/2507.08034

Query: What are the most common categories of external tools integrated with production LLM agents, such as web search, database querying, and code execution, and what are some documented industry case studies that illustrate their practical benefits?

Answer: The Athena framework enables LLMs to use external tools via managed APIs, enhancing their capability to provide **accurate, up-to-date, and data-driven responses**. Tool integration is divided into four main categories: **retrieval-augmented generation (RAG), code execution, API calls, and database queries**. The framework also introduces the concept of "tool making," where LLMs create custom Python functions for specific tasks, which are then registered and used for subsequent problem-solving. This two-phase process—involving both the creation and application of tools—has been validated on benchmarks like Big-Bench, demonstrating that LLMs equipped with external tool use can match the performance of higher-cost models while reducing inference cost. The paper cites case studies where these integrations enable LLMs to handle complex, multi-step reasoning tasks, such as data analysis and fact-checking, that are otherwise beyond the LLM’s standalone capabilities.

-----

-----

-----

### Source [115]: https://www.getdynamiq.ai/post/llm-agents-explained-complete-guide-in-2025

Query: What are the most common categories of external tools integrated with production LLM agents, such as web search, database querying, and code execution, and what are some documented industry case studies that illustrate their practical benefits?

Answer: This guide outlines the most common frameworks for building LLM agents that utilize external tools, such as **LangChain, OpenAI’s Function Calling, CrewAI, AutoGen, and Haystack Agents**. Each framework addresses key aspects of tool integration:

- **LangChain**: Popular for chaining prompts, tool use, and memory, allowing LLMs to make agentic decisions and quickly assemble prototypes.
- **OpenAI’s Function Calling**: Provides lightweight agentic capabilities by letting LLMs call APIs or tools mid-conversation.
- **CrewAI and AutoGen**: Facilitate multi-agent collaboration for complex workflows, such as data workflows or report generation.
- **Haystack Agents**: Focused on retrieval-augmented generation (RAG), especially for tasks relying on structured or document-based knowledge.

These frameworks support **web search, database querying, code execution, and API interactions**. Documented use cases include knowledge management in legal and research sectors, enterprise data workflows, and automated report generation—demonstrating practical benefits such as improved efficiency, accuracy, and the ability to handle tasks that require real-time or external data.

-----

</details>

<details>
<summary>How do modern LLM APIs, such as Google's Gemini or OpenAI's, handle parallel and sequential (compositional) function calling, and what are the key differences in their implementation?</summary>

### Source [82]: https://codelabs.developers.google.com/codelabs/gemini-function-calling

Query: How do modern LLM APIs, such as Google's Gemini or OpenAI's, handle parallel and sequential (compositional) function calling, and what are the key differences in their implementation?

Answer: Modern LLM APIs such as Google Gemini handle function calling by letting developers define one or more **function declarations** within a "tool" and sending these together with the user's prompt to the Gemini API. The model analyzes the prompt and the available functions, then returns a structured **Function Call** response specifying the function to call and its parameters. The application (outside the model) then executes the function, obtains the external result, and passes that result back to Gemini if needed. This enables multi-step, compositional workflows: after receiving a function result, Gemini may request additional function calls if more information is needed to complete the original user request. This back-and-forth supports both parallel (multiple independent calls) and sequential (chained) function calling, depending on how the application manages the loop between model and external functions.

Key points:
- Function calling is orchestrated by the developer's application logic; Gemini provides structured suggestions.
- Compositional workflows (sequential function calls) are supported by repeatedly exchanging outputs and new prompts between app and model.
- The Gemini API itself does not natively orchestrate parallel or sequential multi-function workflows; the application manages this logic[1].

-----

-----

-----

### Source [83]: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling

Query: How do modern LLM APIs, such as Google's Gemini or OpenAI's, handle parallel and sequential (compositional) function calling, and what are the key differences in their implementation?

Answer: To use function calling with Gemini on Vertex AI, developers define function schemas using an OpenAPI-compatible format and submit them together with the user prompt. The Gemini model analyzes the prompt and returns a structured response if a function call is needed. The schema allows for complex parameter types, supporting sophisticated function definitions.

The process is:
- Submit prompt with function declarations (schemas) as part of the API request.
- The API returns a function call suggestion if needed.
- The developer executes the function and can return the result to the model for further reasoning or additional calls.

There is no explicit mention of native parallel or sequential orchestration at the model level; instead, the logic for chaining or concurrently calling functions is implemented by the developer in the application layer. The API is flexible and allows for multi-step workflows by iteratively sending updated prompts and function results[2].

-----

-----

-----

### Source [84]: https://ai.google.dev/gemini-api/docs/function-calling

Query: How do modern LLM APIs, such as Google's Gemini or OpenAI's, handle parallel and sequential (compositional) function calling, and what are the key differences in their implementation?

Answer: Gemini's official documentation emphasizes that the model can return a **functionCall** object (in OpenAPI-compatible schema) indicating which declared function(s) should be called. This is supported in both Python and JavaScript SDKs. The function call suggestion is based on the content of the prompt and the developer-defined function declarations.

- After the model suggests a function call, the developer's code is responsible for executing the function and optionally sending the result back to Gemini.
- The cycle can be repeated, enabling **sequential (compositional) function calling**.
- The documentation does not describe built-in support for orchestrating multiple function calls in parallel or chaining them automatically; this control is left to the application logic.

Thus, Gemini supports compositional workflows through repeated request/response cycles, with the application managing the orchestration and state management for both sequential and (if desired) parallel function calls[3].

-----

-----

-----

### Source [85]: https://firebase.google.com/docs/ai-logic/function-calling

Query: How do modern LLM APIs, such as Google's Gemini or OpenAI's, handle parallel and sequential (compositional) function calling, and what are the key differences in their implementation?

Answer: Firebase AI Logic describes that when using Gemini function calling, the recommended pattern is a **multi-turn chat interface** where the application and the model pass information back and forth. The developer provides the model with a set of tools (functions) it can call. When the model determines a function is needed, it returns structured information specifying the function and parameters. The application then executes the function, retrieves the result, and may send it back to the model for further processing or additional function calls.

This approach naturally supports **sequential composition**: the model can reason, call a function, process the result, and possibly request subsequent function calls as needed to fulfill the user's original intent. Parallelization (executing multiple independent function calls simultaneously) is not handled by Gemini itself, but could be implemented by the developer's application logic if multiple function calls are returned or inferred[4].

-----

-----

-----

### Source [86]: https://www.philschmid.de/gemini-function-calling

Query: How do modern LLM APIs, such as Google's Gemini or OpenAI's, handle parallel and sequential (compositional) function calling, and what are the key differences in their implementation?

Answer: According to this guide, the **google-genai Python SDK** can automatically generate JSON Schemas from Python function signatures and docstrings, simplifying function declaration. When using Python functions as tools, the SDK supports **automatic function calling:** after the model requests a function, the SDK can automatically execute it and send the result back to Gemini for further reasoning. This enables compositional, multi-step workflows with minimal manual boilerplate.

Additionally, Gemini offers **OpenAI compatible API endpoints**, allowing developers to use OpenAI-style function calling interfaces with Gemini models. This means the mechanisms for function calling, including sequential and compositional chaining, are broadly similar to OpenAI’s approach. However, both Gemini and OpenAI expect the application to handle orchestration—whether executing function calls in sequence or in parallel (if multiple are suggested)—rather than the model handling this natively[5].

-----

-----

-----

### Source [125]: https://codelabs.developers.google.com/codelabs/gemini-function-calling

Query: How do modern LLM APIs, such as Google's Gemini or OpenAI's, handle parallel and sequential (compositional) function calling, and what are the key differences in their implementation?

Answer: Modern LLM APIs like Google's Gemini support function calling by allowing developers to define function declarations within a tool, which are provided to the Gemini API at runtime. When a user sends a prompt, the Gemini API analyzes the input and, if appropriate, returns a Function Call response containing structured data with the function name and parameters. The actual execution of the function (e.g., making an API request to an external service) is handled by the developer's application code, not the Gemini API itself. After executing the function, the developer passes the result back to Gemini, which can then either generate a final response to the user or, if more information is needed, suggest another function call[1].

-----

-----

-----

### Source [126]: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling

Query: How do modern LLM APIs, such as Google's Gemini or OpenAI's, handle parallel and sequential (compositional) function calling, and what are the key differences in their implementation?

Answer: To use function calling in Gemini (as integrated on Vertex AI), developers must submit both the user prompt and function declarations—encoded in an OpenAPI-compatible schema—when making a request. These function declarations inform the model about available actions it can suggest. The example provided shows how a function like `get_current_weather` is declared and sent along with a prompt. The model can then analyze the prompt and decide whether a function should be called, returning structured data that specifies which function and what parameters to use. The actual function call and its execution remain the responsibility of the developer's application logic[2].

-----

-----

-----

### Source [127]: https://ai.google.dev/gemini-api/docs/function-calling

Query: How do modern LLM APIs, such as Google's Gemini or OpenAI's, handle parallel and sequential (compositional) function calling, and what are the key differences in their implementation?

Answer: The Gemini API allows developers to define function declarations and pass them to the model along with user prompts. The model analyzes the prompt and, if it determines a function should be called, returns a `functionCall` object in an OpenAPI-compatible schema. This object specifies how to perform the call, including the function name and parameter values. The developer's application is responsible for executing the function and handling its results. This process is consistent across multiple languages (e.g., Python, JavaScript), and the model can suggest one or more function calls as needed, supporting compositional workflows if multiple steps are required for a response[3].

-----

-----

-----

### Source [128]: https://firebase.google.com/docs/ai-logic/function-calling

Query: How do modern LLM APIs, such as Google's Gemini or OpenAI's, handle parallel and sequential (compositional) function calling, and what are the key differences in their implementation?

Answer: When using function calling with Gemini via Firebase AI Logic, the process involves providing the model with a set of tools (function declarations) during initialization. The model can then suggest function calls based on user prompts, returning structured information about which function to call and with what parameters. The application must execute the suggested function(s) and supply results back to the model if a multi-turn dialogue or additional function calls are required. This setup enables compositional or sequential function calling, where multiple back-and-forth exchanges may occur until the user's request is fully resolved[4].

-----

-----

-----

### Source [129]: https://www.youtube.com/watch?v=mVXrdvXplj0

Query: How do modern LLM APIs, such as Google's Gemini or OpenAI's, handle parallel and sequential (compositional) function calling, and what are the key differences in their implementation?

Answer: According to the Gemini API overview, function calling is enabled by providing Gemini with explicit function declarations that describe each function's name, purpose, and parameters—a form of "instruction manual" for the model. Upon analyzing a prompt, Gemini determines if a function call is needed and returns a structured JSON function call suggestion. Importantly, Gemini itself does not execute the function; the developer must implement the execution logic. This approach supports both single and potentially compositional (multi-step) workflows, as additional function call suggestions can be generated if more information is needed at each step[5].

-----

-----

</details>

<details>
<summary>What are the fundamental limitations that prevent large language models from accessing real-time information or executing external actions on their own, and how do AI researchers justify the introduction of 'tools' to overcome these gaps?</summary>

### Source [87]: https://www.projectpro.io/article/llm-limitations/1045

Query: What are the fundamental limitations that prevent large language models from accessing real-time information or executing external actions on their own, and how do AI researchers justify the introduction of 'tools' to overcome these gaps?

Answer: Large language models (LLMs) are fundamentally limited in accessing real-time information and executing external actions due to several reasons. They suffer from **computational constraints**, meaning they can only process a fixed number of tokens at once, which restricts their ability to handle large or continuous data streams in real time. LLMs also have **limited knowledge updating**—their knowledge is static and confined to their training data, so they cannot natively access or incorporate new information after deployment. Additionally, LLMs lack **long-term memory** and struggle with **complex reasoning**, making it difficult for them to manage ongoing tasks or adapt to dynamic external environments. These limitations collectively mean that LLMs cannot autonomously interact with external systems or update themselves with live data; instead, they require manual intervention or the integration of external tools and APIs to bridge these gaps. Researchers justify the introduction of 'tools'—such as plugins, retrieval-augmented generation, or API interfaces—as necessary extensions that allow LLMs to access up-to-date resources, execute actions beyond text generation, and interact with the world in ways that pure model inference cannot achieve[1].

-----

-----

-----

### Source [88]: https://blog.gdeltproject.org/large-language-models-llms-planetary-scale-realtime-data-current-limitations/

Query: What are the fundamental limitations that prevent large language models from accessing real-time information or executing external actions on their own, and how do AI researchers justify the introduction of 'tools' to overcome these gaps?

Answer: LLMs face significant challenges when tasked with **processing planetary-scale, real-time data** such as global news coverage. Their **attention mechanisms** are optimized for small prompts generating large outputs, not for ingesting massive, continually updating data streams. When asked to summarize or act on vast, real-time information, these models quickly reach their attention or token limits, resulting in errors or degraded performance. LLMs also frequently **hallucinate** (generate inaccurate or fabricated information) and struggle with **conflict and ambiguity resolution** in fast-changing contexts. Because most public LLMs cannot natively update their knowledge or manage unbounded real-time input, the research community introduces external 'tools'—such as cascading summarization, external knowledge stores, or API integrations—to overcome these architectural limitations. These tools allow LLMs to access current information and perform external actions, compensating for the inherent gap between static model inference and dynamic real-world requirements[2].

-----

-----

-----

### Source [89]: https://direct.mit.edu/opmi/article/doi/10.1162/opmi_a_00160/124234/The-Limitations-of-Large-Language-Models-for

Query: What are the fundamental limitations that prevent large language models from accessing real-time information or executing external actions on their own, and how do AI researchers justify the introduction of 'tools' to overcome these gaps?

Answer: LLMs are fundamentally distinguished from human conversational agents by their **mechanistic limitations**. Unlike humans who can fluidly engage in overlapping dialogue, repair communication in real-time, and adapt responses based on ongoing feedback (such as facial expressions), LLMs must plan and generate their entire utterance before 'speaking.' This means LLMs cannot **clarify requests in-progress** or **modify outputs dynamically** based on external context as humans do. The process of converting requests into text and back into speech introduces delays and restricts natural conversational flow. These limitations prevent LLMs from autonomously interacting with the real world or executing external actions without explicit integration or tool augmentation. Researchers therefore justify the incorporation of 'tools' that enable dynamic interaction, external action execution, and real-time adaptation, bridging the gap between static model outputs and the demands of complex, interactive tasks[3].

-----

-----

-----

### Source [90]: https://arxiv.org/html/2412.04503v1

Query: What are the fundamental limitations that prevent large language models from accessing real-time information or executing external actions on their own, and how do AI researchers justify the introduction of 'tools' to overcome these gaps?

Answer: LLMs are **pre-trained on large, static datasets**, typically through unsupervised or self-supervised learning. Their knowledge and capabilities are determined entirely by the data available at the time of training, which means they do not possess mechanisms for **real-time knowledge updating** or **external action execution** after deployment. The architecture of LLMs is designed for inference from static weights, not for ongoing interaction with the outside world. To enable LLMs to access current information or perform actions, researchers introduce 'tools' such as retrieval-augmented generation, plugins, or system-level APIs. These tools act as intermediaries, allowing the model to query external databases, fetch live information, or trigger actions outside the model's static confines. The justification for this approach is rooted in the recognition that static, text-only models are inherently limited in real-world applicability, and only through integration with external systems can they become truly useful in dynamic, evolving environments[4].

-----

-----

-----

### Source [91]: https://uit.stanford.edu/service/techtraining/ai-demystified/llm

Query: What are the fundamental limitations that prevent large language models from accessing real-time information or executing external actions on their own, and how do AI researchers justify the introduction of 'tools' to overcome these gaps?

Answer: LLMs have several core limitations: they can **generate biased or incorrect information**, lack the ability to **update knowledge post-training**, and are not designed to autonomously interact with external systems or environments. Their training process is based on static datasets, so any new information after this period is inaccessible. These models do not possess intrinsic capabilities for **real-time data retrieval, execution of external actions, or ongoing adaptation**. Consequently, AI researchers introduce 'tools'—such as external APIs, retrievers, or plugins—to enable LLMs to access up-to-date information or perform tasks beyond pure text generation. These tools supplement the inherent gaps in LLM functionality and are justified by the need to make LLMs more practically useful for complex, dynamic, and interactive tasks that require real-time input or external action[5].

-----

-----

-----

### Source [116]: https://www.projectpro.io/article/llm-limitations/1045

Query: What are the fundamental limitations that prevent large language models from accessing real-time information or executing external actions on their own, and how do AI researchers justify the introduction of 'tools' to overcome these gaps?

Answer: Large language models (LLMs) face several fundamental limitations. One major constraint is **computational boundaries**: LLMs are limited by a fixed number of tokens they can process at once, which restricts their ability to maintain context and efficiently handle large or continuous data streams. Another limitation is **knowledge updating**—LLMs cannot access or incorporate real-time information after their last training update, resulting in outdated responses. They also have no **long-term memory**, meaning information from previous interactions is not retained. Additionally, LLMs struggle with **complex reasoning** and may produce inaccurate or hallucinated outputs, especially when dealing with nuanced or evolving topics. These constraints mean that, on their own, LLMs cannot fetch current data or interact with external systems. To address these gaps, AI researchers have introduced 'tools' that allow LLMs to access external APIs, databases, or other resources for real-time information and to execute actions outside their own architecture. This tool integration compensates for the inherent limitations in processing, memory, and knowledge freshness[1].

-----

-----

-----

### Source [117]: https://blog.gdeltproject.org/large-language-models-llms-planetary-scale-realtime-data-current-limitations/

Query: What are the fundamental limitations that prevent large language models from accessing real-time information or executing external actions on their own, and how do AI researchers justify the introduction of 'tools' to overcome these gaps?

Answer: LLMs are currently not optimized for large-scale, real-time data ingestion or constant knowledge updating. Their architecture is designed for "small-in, large-out" workflows—meaning a small prompt can generate large outputs. However, "large-in, small-out" tasks, such as summarizing massive real-time news feeds, quickly exhaust their attention capacity and computational limits. Most public LLMs cannot consume or process the entirety of continuous, real-time information, leading to errors or degraded performance. Architectural challenges also hinder **unbounded real-time knowledge updating** and effective conflict or ambiguity resolution when new data arrives. As a result, the research community has justified the use of external tools, like cascading summarization and external knowledge stores, to overcome these deficits. These tools help LLMs access up-to-date information and manage large-scale input beyond their built-in attention and memory limits[2].

-----

-----

-----

### Source [118]: https://www.intuitivedataanalytics.com/gne-blogs/the-limitations-and-challenges-of-large-language-models-llms/

Query: What are the fundamental limitations that prevent large language models from accessing real-time information or executing external actions on their own, and how do AI researchers justify the introduction of 'tools' to overcome these gaps?

Answer: LLMs fundamentally lack **real-time awareness** and cannot update their knowledge base after training. This gap means that they cannot reliably answer queries about current events, recent developments, or dynamic data such as sports scores or geopolitical shifts. In these scenarios, LLMs may fabricate or hallucinate facts based on outdated training, which undermines their reliability for time-sensitive decision-making. Researchers and practitioners acknowledge these limits and justify tool integration by emphasizing that external tools can connect LLMs to live databases, APIs, or other sources of current information—making the models useful for real-world, real-time tasks where static, pre-trained knowledge is insufficient[3].

-----

-----

-----

### Source [119]: https://arxiv.org/html/2412.04503v1

Query: What are the fundamental limitations that prevent large language models from accessing real-time information or executing external actions on their own, and how do AI researchers justify the introduction of 'tools' to overcome these gaps?

Answer: LLMs are pre-trained on massive datasets, typically comprised of publicly available data, and may be fine-tuned for specific tasks. However, the **pre-training process is inherently static**—models do not learn from new data after training unless explicitly retrained or fine-tuned. Supervised fine-tuning for specific tasks requires human annotation, which is infeasible at scale, further limiting the ability of LLMs to stay current. The underlying architecture focuses on pattern recognition and data structure within the training set, not on direct interaction with external systems or real-time environments. Researchers have explored creating bespoke LLMs or integrating external systems to address the need for up-to-date information and action execution. Introducing tools or APIs allows LLMs to bridge the gap between static training and dynamic, real-time tasks, thereby compensating for their inability to natively update knowledge or interact with external data sources[4].

-----

-----

-----

### Source [120]: https://uit.stanford.edu/service/techtraining/ai-demystified/llm

Query: What are the fundamental limitations that prevent large language models from accessing real-time information or executing external actions on their own, and how do AI researchers justify the introduction of 'tools' to overcome these gaps?

Answer: LLMs are limited by their training data and can generate **biased or incorrect information** if presented with queries outside their knowledge cutoff. The training and deployment processes do not include mechanisms for real-time learning or external action execution. During deployment, models may be optimized for performance, but they remain fundamentally disconnected from live data sources and external environments. This limitation means LLMs cannot autonomously update themselves or interact with the world beyond generating text. To overcome these gaps, researchers have introduced external tools that enable LLMs to access live information, execute actions, or retrieve data from outside databases. These integrations are necessary to make LLMs relevant for applications requiring up-to-date information or direct external interaction, as their native architecture is not designed for such capabilities[5].
-----

-----

</details>

<details>
<summary>What are the fundamental limitations of large language models that prevent them from accessing real-time information or executing external actions on their own, and how do AI researchers justify the introduction of 'tools' to overcome these gaps?</summary>

### Source [92]: https://www.projectpro.io/article/llm-limitations/1045

Query: What are the fundamental limitations of large language models that prevent them from accessing real-time information or executing external actions on their own, and how do AI researchers justify the introduction of 'tools' to overcome these gaps?

Answer: Large language models (LLMs) have several fundamental limitations impacting their ability to access real-time information or perform external actions autonomously. One key limitation is **computational constraints**: LLMs are restricted by a fixed number of tokens they can process at once, which is necessary for maintaining efficient performance and timely responses. Exceeding this token limit results in errors, helping to maintain context within manageable computational boundaries. Additional limitations include **issues with accuracy and knowledge updating**, as LLMs typically cannot update their internal knowledge base after training and lack mechanisms to incorporate new, real-time data on their own. Furthermore, they possess a **lack of long-term memory**, struggle with complex reasoning, and are subject to the limitations imposed by their training data, which necessarily becomes outdated over time. These factors prevent LLMs from autonomously accessing or integrating fresh, real-time information or reliably executing external actions without additional systems or interventions.

To overcome these gaps, researchers introduce **'tools'**—external plugins, APIs, or retrieval systems—that allow LLMs to access up-to-date information, perform complex computations, or interact with external environments. This modular approach enables LLMs to remain efficient while extending their capabilities beyond their inherent architectural and training data limitations.

-----

-----

-----

### Source [93]: https://blog.gdeltproject.org/large-language-models-llms-planetary-scale-realtime-data-current-limitations/

Query: What are the fundamental limitations of large language models that prevent them from accessing real-time information or executing external actions on their own, and how do AI researchers justify the introduction of 'tools' to overcome these gaps?

Answer: Current state-of-the-art LLMs face substantial challenges in processing and summarizing **planetary-scale, real-time data**. They are optimized for "small-in, large-out" workflows, excelling at generating extended outputs from brief prompts, but are not suited for "massive input to small output" scenarios, such as real-time news monitoring at a global scale. LLMs are prone to **attention loss** and **hallucination**, particularly when the writing style of input data does not closely match their training prompts. Most importantly, public LLMs are not architected for true unbounded real-time knowledge updating, and architectural challenges hinder their ability to resolve conflicting or ambiguous information in real time. As a result, LLMs typically require **external knowledge stores** or **cascading summarization tools** to handle large volumes of up-to-date information, highlighting the need for external tools to bridge their inherent limitations in real-time data processing and action execution.

-----

-----

-----

### Source [94]: https://www.decodable.co/blog/llms-need-real-time-data-to-deliver-contextual-results

Query: What are the fundamental limitations of large language models that prevent them from accessing real-time information or executing external actions on their own, and how do AI researchers justify the introduction of 'tools' to overcome these gaps?

Answer: LLMs inherently lack **real-time awareness** and **business-specific context** because they are trained on static datasets and cannot dynamically ingest or process live information after deployment. This limitation means their answers may sound plausible but are often generic, outdated, or misaligned with evolving business needs. In enterprise settings, the inability to respond to rapidly changing events—such as recent transactions, updated policies, or trending topics—can erode user trust and diminish the value of AI solutions. To address these deficits, researchers and practitioners justify the integration of **external tools** that ground LLMs in real-time, proprietary, or contextual data, ensuring outputs reflect current realities and specific organizational priorities.

-----

-----

-----

### Source [95]: https://learnprompting.org/docs/basics/pitfalls

Query: What are the fundamental limitations of large language models that prevent them from accessing real-time information or executing external actions on their own, and how do AI researchers justify the introduction of 'tools' to overcome these gaps?

Answer: LLMs possess **limited knowledge** because their training data is fixed in time, and without internet or external system access, they cannot provide information on events or facts that occurred after their last training update. This fundamental limitation means LLMs are unable to answer questions about recent events or execute actions that require current data. To mitigate these shortcomings, AI researchers advocate for the **combination of LLMs with specialized tools or plugins**. These external systems can supply real-time data, perform complex reasoning, or handle tasks such as mathematical calculations, supplementing the LLM's capabilities and compensating for its static knowledge base.

-----

-----

</details>

<details>
<summary>In what ways do modern LLM APIs, such as Google's Gemini or OpenAI's, handle parallel and sequential (compositional) function calling, and what are the key differences in their implementation?</summary>

### Source [101]: https://atamel.dev/posts/2024/08-06_deepdive_function_calling_gemini/

Query: In what ways do modern LLM APIs, such as Google's Gemini or OpenAI's, handle parallel and sequential (compositional) function calling, and what are the key differences in their implementation?

Answer: Gemini's function calling can handle both **parallel and sequential (compositional) function calls**. When a prompt requests information that can be fetched independently (such as weather for multiple cities), Gemini sometimes issues **parallel function calls**—multiple requests in a single response, each targeting a different entity (e.g., Mumbai, Seoul, Jakarta). The underlying response structure shows separate function call objects for each city, within a single response payload. However, Gemini may also opt for **sequential function calls**: issuing one function call per response, requiring the client to handle each call and subsequent follow-ups. The model itself decides whether to parallelize or sequence the calls, influenced by prompt structure and internal heuristics. This flexibility allows Gemini to optimize for context, but places some responsibility on the developer to handle multi-turn or compositional workflows.

-----

-----

-----

### Source [102]: https://discuss.ai.google.dev/t/function-calling-multiple-function-results/60678

Query: In what ways do modern LLM APIs, such as Google's Gemini or OpenAI's, handle parallel and sequential (compositional) function calling, and what are the key differences in their implementation?

Answer: Gemini's API enables **multiple function calls per prompt**, and developers can receive several function call requests in a single round-trip. In practice, this allows for parallel workflows, where results from multiple functions are processed together. For sequential or compositional workflows, the developer must implement a pattern to feed multiple `functionResponse` objects back to the model, potentially in a single prompt. The discussion highlights the use of techniques akin to `Promise.all` from concurrent programming, where all function results are awaited before re-invoking the model with the combined results. Currently, this pattern relies on the developer to format and associate function responses (such as using saga IDs), and there is no built-in API mechanism for compositional chaining or automatic aggregation of multiple results beyond what the developer structures in the prompt.

-----

-----

-----

### Source [103]: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling

Query: In what ways do modern LLM APIs, such as Google's Gemini or OpenAI's, handle parallel and sequential (compositional) function calling, and what are the key differences in their implementation?

Answer: Vertex AI supports **parallel function calling** for Gemini models. For prompts that request information for multiple entities (e.g., weather in Boston and San Francisco), the model can propose several parallel function calls within the same response payload. The function calls are structured as distinct objects, each with its own parameters. The documentation provides examples showing how to declare and handle such function calls, including how to define parameters (with support for enums and JSON schema formats). Sequential or compositional function calling (where later calls depend on earlier results) is not explicitly described as a built-in feature; developers are expected to manage multi-turn workflows by feeding results back into the model as needed. The documentation focuses on best practices for parallelization and efficient response handling.

-----

-----

-----

### Source [105]: https://ai.google.dev/gemini-api/docs/function-calling

Query: In what ways do modern LLM APIs, such as Google's Gemini or OpenAI's, handle parallel and sequential (compositional) function calling, and what are the key differences in their implementation?

Answer: The official Gemini API documentation details **parallel function calling**, allowing developers to call multiple functions at once when the operations are independent. The API supports declaring and executing several functions in a single request, which is useful for collecting information or performing actions that do not depend on each other's results. The documentation provides examples in Python, showing how to structure requests and responses for multiple simultaneous function calls. However, it notes limitations: only a subset of OpenAPI schema is supported and automatic function calling is a Python SDK feature. Sequential or compositional calling (where functions depend on previous results) must be handled manually by the developer, structuring multi-turn interactions and feeding outputs back into the model. There is no built-in API support for automatic chaining or aggregation of results for compositional workflows.

-----

-----

</details>


## Sources Scraped From Research Results

<details>
<summary>(no content)</summary>

(no content)

</details>

<details>
<summary>Comprehensive Guide to Integrating Tools and APIs with Language Models</summary>

# Comprehensive Guide to Integrating Tools and APIs with Language Models

With a sudden rise of LLMs like GPT-4, we are seeing a massive rise in productivity. Language models have more utility than ever. Everyone is using them and they are everywhere. This raises the need to integrate LLMs with external tools and APIs. ChatGPT with extensions solves this to some extent, but not fully.

Issues like integrating ChatGPT with private tools are still not fully solved. This requires building your own framework for integration and then using ChatGPT. In this article we will cover how you can integrate tools and third-party APIs with GPT-4 using function calling, prompting techniques, and other methods.

## Why do we need to integrate tools and APIs with LLMs

Language Models are obviously going to be anywhere. I like to call them “a unified UI” for a thousand tools. We have to use hundreds if not thousands of applications on a monthly basis to do everyday things. Using tools like Excel, Email Systems, CRMs, Project management tools, etc., can add so much friction to doing something as simple as replying to an email. If we can have an LLM tightly integrated with these tools, we can just write single-line prompts and LLM does all the tasks. Here are some reasons why we need to integrate tools into LLMs and how it would help

### Using Private LLMs with Private Data

As mentioned before, it is almost necessary to use LLMs, and more so with private data. Industries like Healthcare, Oil and Gas, and Government agencies cannot provide their data to OpenAI at all, they need to use self-hosted solutions using models like LLaMA and Falcon. These self-hosted solutions don’t have access to plugins like ChatGPT does. They need to build their own pipelines and plugins. This is difficult and time-consuming.

However if done properly, using self-hosted LLMs with private data can solve many issues and can boost productivity. For example, in the healthcare industry, integrating LLMs with electronic health records (EHRs) can assist doctors and medical professionals in analyzing patient data and providing more accurate diagnoses. This can save time and improve patient outcomes.

### Higher Utility

Integrating tools with LLMs can also increase the utility of these tools AND the LLMs. With LLMs, users can perform tasks more efficiently and accurately, reducing the need for manual labor and multiple tools. Just a single command to an LLM and everything gets taken care of. This can lead to cost savings for businesses and increased productivity for individuals.

### Better, More Accurate Responses

A simple yet VERY effective reason to integrate tools with LLMs is to increase the quality of the response. LLMs on their own are simply _text-generation machines_, very powerful, but they lack proper context. This means you can ask questions like “How to write a good proposal?” but you cannot ask “How to write a good proposal to sell my services to Mercity?”, this is because the LLM has no context of what your services are and who Mercity is. This is where the need for custom context arises.

Answering questions based on private data is actually much simpler. We have written an excellent guide on how to [integrate custom private data with GPT-4](https://www.mercity.ai/blog-post/custom-gpt-4-chatbot). You can check it out. We use an embeddings-based retrieval system to extract the relevant chunks of text to answer questions based on private data.

https://cdn.prod.website-files.com/640f56f76d313bbe39631bfd/650f66ff38e964525a33753d_It9AmDibaYSRHYjW2gvzPFSjMdgtYflkC3jljDbGfGB-EGCrnxYsGtgRWCP6d0rulXolHBBDKJ2moUaHOfIAtn5IM3A8xItozOZBdAUuO4oU3MewKH_mW9AV4x0vDDlWfiUdRVekEDVi4DRoZpeViJI.png

### Single Interface for Multiple Tools

As mentioned before, LLMs can act as a unified UI for multiple tools. This is becoming necessary because the number of tools we use is increasing rapidly. Also, the information and knowledge are completely spread out on different platforms. LLMs can reduce the friction of these multi-platform tasks. Language models like GPT-4 are smart enough to string function calls together and use multiple tools in a chain, collecting data, and planning and executing the given task.

### ‍

For example, if you need to [extract meeting notes from sales calls](https://www.mercity.ai/blog-post/gpt-nlp-in-sales#sales-call-analytics), write a proposal and send over it to the client. An LLM can find the meeting transcripts, extract notes from them, write a proposal, and have it ready for your review. This will save around 1~2 hours of time. Once the LLM has drafted a proposal, all you need to do is make any changes necessary and once done, you are ready to send it. So all you need to do is review the generation of the model, while the model takes care of the research and compiling the proposal, which takes more time.

## How to Integrate Tools and APIs with LLMs

At Mercity, we have built our own pipelines for tool integration. At the core is an LLM, and the Tool-use Prompt. Note that LLM here can be any capable instruction following the Language Model. GPT-4 is the smartest and the best model out there, but bigger models like LLaMa 70B and Falcon 180B can also be used here. Models just have to be smart enough to follow the prompts and should be able to generate sophisticated outputs.

Let’s break this pipeline down step by step.

https://cdn.prod.website-files.com/640f56f76d313bbe39631bfd/650f6700a38686bf04abf9ae_oYM1xj0eyB4-37H6LC2k60wZN98clF_vgT6B7Rv4NTkmMqppzhWmw6fOeFw-Vp5m_daHM0Mas4j3DSf5dkVXQQTq_77WGU0GlopNEGhB5Cg-J26wDv6syQqfOdzoUUVfAr2z9a5m_iunOX2nM4oAsJk.png

### Large Language Model

A Language model like GPT-4 is at the core of it all, it acts as the user interface for the tools and the APIs we want to integrate. The model doesn’t necessarily need to be finetuned for chatting, but it would be better if it is. In our findings, we have noticed that larger models work better for these applications, simply because they are better instruction followers and are much better at maintaining multi-step conversations without losing the nature of the conversation. Smaller models like LLaMa-13B can be finetuned to a great degree to follow specific tool use prompts using [PEFT Techniques](https://www.mercity.ai/blog-post/fine-tuning-llms-using-peft-and-lora). Specifically, techniques like Prefix Tuning and IA3 are very popular for tuning LLMs with smaller datasets.

Once the LLM is selected and is validated to follow instructions properly, we can tightly integrate it with a Tool Database and Tool-Use Prompt.

#### API Tool Database

A tool database is simply a collection of all the tools you might have or want to use with the language model. This can simply be a list of tools and APIs in text or can be a much more sophisticated dynamically fetched pipeline. Most of the time, we use simple text, with the name and description of the API along with how to use it and when, and provide it to the LLM in the form of a tool library.

When providing API, we abstract it as a function call and use the arguments to construct a schema for the API call.

Here’s what a demo tool library would look like:

```
Tool Library:

- web_search(query) - Use this function to find information on the Internet. It's a general search function that can be applied to almost any topic. You pass the query string here. Make sure your queries are precise.

- embedding_database_search(query) - This function is specifically designed for retrieving information. You can use this tool over others to find information about very specific personal topics. In `query` pass the information you want to find about a topic.

- wikipedia_search(query) - Use this function when the information needed can be found on Wikipedia. It serves as a direct conduit to this comprehensive knowledge base. This provides extensive knowledge on a specific topic. Use this function accordingly.
```

#### Tool Use Prompt

Once we have the tool library ready, we can put it in the tool use prompt and explain to the LLM how to call and use tools. This is a very important part of the pipeline as this determines exactly how and when the language model will use the tool. There are multiple prompting techniques that can be used here, but at Mercity we like to use our own self-built prompts. We will take a deeper look at the prompting techniques for tool use now.

### Prompting for Tool Using and API Calling

As said above, this is perhaps the most important part. LLMs need to be prompted properly on how to use the tools you have provided and when to use them. We need to craft the perfect prompt for this. There are many ways to do this, but we like to use the most basic technique.

We simply provide the _tool library_ to the model and ask the model to output in a very specific format so that we can parse and use the tool when needed. This is what the prompt looks like when combined with the aforementioned tool library:

```
You are a masterful tool user, you must assist the user with their queries, but you must also use the provided tools when necessary.

You must reply in a concise and simple way and must always follow the provided rules.

===========================================================

Tool Library:

- web_search(query) - Use this function to find information on the Internet. It's a general search function that can be applied to almost any topic. You pass the query string here. Make sure your queries are precise.

- embedding_database_search(query) - This function is specifically designed for retrieving information. You can use this tool over others to find information about very specific personal topics. In `query` pass the information you want to find about pets in the specified categories.

- wikipedia_search(query) - Use this function when the information needed can be found on Wikipedia. It serves as a direct conduit to this comprehensive knowledge base. This provides extensive knowledge on a specific topic. Use this function accordingly.


===========================================================

To use these tools you can output func_name(query) in middle of generation or ONLY output the function call.

Example outputs:

- The current president of the United States of America is web_search("Who is the current president of United States")

- wikipedia_search("Joe Biden") = You can output like this when user wants extensive detail on a specific topic or person.


===========================================================

Note that you must always follow the provided rules and output in the given manner. Using a function is not always necessary, use only when needed.
```

This is what the outputs look like from this prompt:

https://cdn.prod.website-files.com/640f56f76d313bbe39631bfd/650f66ff4065734724fd4745_NOBxGTkIvbHJamqYaS47q6gzW_qSrAf2HYNkOUn35qTFond-XCLfAvHAbcCZdLFRodgMZPQbd37SU-mzi1amLyQlLsq4ftZ3AomS8CzvKOqO_1YnNIwaQw-hbOt2KF4tSukggX707J3p2npRfYjunpI.png

You can see that the model was able to properly identify when it needed to call the function. It did not call the function when I asked it about the topics I could write on, but did call the function when it was absolutely needed.

Even more so, it was properly able to identify when it needed to query my personal documents and was able to write an excellent query to use the embedding search tool with:

https://cdn.prod.website-files.com/640f56f76d313bbe39631bfd/650f66ff33137a6ce375b155_V2ZjK0TqDsWAjtI7bFJXWo1KaT2loYn1AijL1jz_ixQx3DkoyzT7S4N_2pAJhv_RUzsedckHcvvLjigtOQ9ecJanIAMgFS-th94o_onXMktEqT7ZvjlRNSaCGclBRhvifXO4_vWEVksoyDufkQMaR6Y.png

This method of prompting is extremely easy and works beautifully. In our experiments, we have seen some issues arise when you try to combine this method with already very long prompts. With a longer prompt, it gets extremely hard to make the model output in the proper format, and the accuracy of tool use drops. But these issues are largely only seen with GPT3.5, and not with GPT-4. GPT-4 is much better at following complicated formats and instructions.

#### React Based Prompting

https://cdn.prod.website-files.com/640f56f76d313bbe39631bfd/650f66ff788d34c59019f6d3_W6QEEql3bAz9ukUQTh7VmDibP_f54v7jVRlAS1nHS9Ov7-WQI_rd6SOa-ktEXLpvJFdebV-vqqvcgvKp5tQZcpzUG_XLWpKQ50JgChkpYlDPiXmS9qcpZfDM4hAKJpMs4S2e9IGENRlxdTwxM-ahfE4.png

There is another better prompting method called [ReAct](https://arxiv.org/abs/2210.03629). This method has been popular lately. ReAct breaks down the LLM outputs into 3 aspects, **Thought**, **Act,** and **Observation**. Here is a breakdown of these parts:

- **Thought:** This is the part where LLM THINKS what it needs to do. The model analyzes the input and generates somewhat of a plan on what do to.
- **Act:** This is the action part. Based on the **thought** the LLM now acts. This can be using a tool or calling an API or interacting with something else. Or this can be left blank if needed.
- **Observation:** In the end, based on the thoughts and the output of the Action, an observation is made. This observation can be an answer to a question or starting of yet another ReAct chain.

Here is an application of this prompt:

https://cdn.prod.website-files.com/640f56f76d313bbe39631bfd/650f67009c7332290b91eba8_yPyjYkUsWn8KYAHAKRf2aOBm7_qUm2KDBpl1UvruX2si1nD-UehscOMADJFEI1eGq9AU8PGYPO7pCJ8QjjIfYr6ctzca3bnu87dTxexFIQkMB_Q5vkE6flfDU34E96xhflswPs7xyoWaKxCpYzMAAwY.png

You can see that LLM was able to correctly identify the need for the tool and call it accordingly. ReAct prompting is better than the basic prompting we showed above because this allows the model to analyze the input before providing an output, and this boosts the accuracy. The only downside is a bit increase in token usage, but the increase in accuracy and control over outputs makes that worth it.

Here is the SYSTEM prompt we use:

```
You are a masterful tool user, you must assist the user with their queries, but you must also use the provided tools when necessary.

You must reply in a concise and simple way and must always follow the provided rules.

===========================================================

Tool Library:

- web_search(query) - Use this function to find information on the Internet. It's a general search function that can be applied to almost any topic. You pass the query string here. Make sure your queries are precise.

- embedding_database_search(query) - This function is specifically designed for retrieving information. You can use this tool over others to find information about very specific personal topics. In `query` pass the information you want to find about pets in the specified categories.

- wikipedia_search(query) - Use this function when the information needed can be found on Wikipedia. It serves as a direct conduit to this comprehensive knowledge base. This provides extensive knowledge on a specific topic. Use this function accordingly.


===========================================================

This is the format you need to output in:

Thought: THINK AND ANALYZE THE INPUT, GOAL AND SITUATION
Act: If you need to call a tool or use a function, you can do it here: func(query). If no need to use a tool, leave this empty. The output of the function will be provided here.
Observation: Based on the Thoughts and results of the Act, provide a reply. If you are using a tool, no need to output this.

Example Acts to use tools:

- The current president of the United States of America is web_search("Who is the current president of United States")

- wikipedia_search("Joe Biden") = You can output like this when the user wants extensive detail on a specific topic or person.


===========================================================

Note that you must always follow the provided rules and output in the given manner. Using a function is not always necessary, use it only when needed.
```

### Function Calling

[Function calling](https://openai.com/blog/function-calling-and-other-api-updates) is a feature released by OpenAI. This allows you to integrate your Chat GPT models like GPT-3.5 and GPT-4 directly with the functions you want to call. You can provide the schema of your functions or APIs and the model will use the provided functions when needed.

Function Calling is the go-to way and probably the first step to take if you are looking to integrate an API with your GPT-4 or GPT-3.5.

Here is an example of how you are supposed to pass the schema of your function to the model input:

https://cdn.prod.website-files.com/640f56f76d313bbe39631bfd/650f66ff65468e3379420799_SXL1TZz_vD5NS3fGxzbM0mK3Sfe0JSz8WjpnNVeJkJFAa4lfoTIivla7F20_Bn5gRtBvwFKdZ4sYBNlLIj6kZGF25fiED2D3_7lF8icBySCm6AxT_Nvdsjp8kurMzCJZVtm8sXiZSLWYfozQ5-hWJkM.png

The model will use the function as needed:

https://cdn.prod.website-files.com/640f56f76d313bbe39631bfd/650f66ff0ef65200ed45d160_bT3AbWWWY4uZRlhFk9xRpCYUkJvys8Ap9iEpoVqhzeLEPERvFwNHEtLSuyqNuBlLT8I-tp1xDrlCnW-W0CDlTaS3YpBrj4lk9_mhQasexQ3Lo1ya1Fkch7w3pepzr_FQ_IpPfTR8iJpnXPCKWhtH1Y4.png

#### How effective is Function Calling for Tool Use?

In our experience, we have found that OpenAI models usually work well with function calling. But as the number of functions grows and you try to add functions that are more custom to your needs, the quality drops quickly and drastically. Also, we have found that the token usage also increases greatly, this is because OpenAI adds the prompts to the system prompt, and the JSON schema takes up a lot of tokens when compared to ReAct or simple tool use prompting.

Many users have also reported that models sometimes hallucinate and output random function names. Here is a good forum post to read that shows how unreliable function calling is: [Function Calling Very Unreliable](https://community.openai.com/t/function-calling-very-unreliable/268439).

## Training LLMs to Use Tools

### Toolformer

[Toolformer](https://arxiv.org/abs/2302.04761) is a model by Meta trained to decide which API to call and then call it. Meta trained this model specifically for tool use and has shown great results. The model calls the functions and stops the generation, then the tool use pipeline provides an answer and the generation continues.

This approach even though simple, is quite effective. But has major issues. For example, this approach works as long as the responses from the tools are short. If the responses grow in length, the quality of outputs will stop dropping. And most of the time the responses from the tools are going to be long and complicated, this can lead to context overflow and model forgetting what was being talked about originally.

https://cdn.prod.website-files.com/640f56f76d313bbe39631bfd/650f66ffa38686bf04abf980_mt9P0TbtdfrbjbG4kK-GDHjbDTiMxx9ElrWke0dc4X-83xdMBsaTGHTMclTEp0-GHR6clMt9UTe8dm3j7M9PlqKv-JvfOeeefnbXWO37OlL3kpdOZtlZeNO5RBp7YS-Agg3uxfzVF2ewI37XlwNuTqs.png

### Gorilla LLM

[Gorilla LLM](https://gorilla.cs.berkeley.edu/) is a large language model coming out of Microsoft and UC Berkeley that can generate API calls from natural language queries. Gorilla LLM can understand and use over 1,600 APIs from various domains, such as machine learning, cloud computing, and web development.

Gorilla LLM is trained on three massive machine learning hub datasets: Torch Hub, TensorFlow Hub, and HuggingFace. It also uses a document retriever to adapt to changes in the API documentation and provide accurate and up-to-date results. Gorilla LLM outperforms other large language models, such as GPT-4, Chat-GPT, and Claude, in writing API calls.

## Use Cases of LLMs integrated with APIs and other Tools

Now that we have discussed how to connect APIs and Tools with LLMs, let’s talk about some of the use cases we have for this.

### Integrating with Email Service Providers

This is perhaps the most obvious one. Email inboxes have become very messy with hundreds of emails coming in every day. This makes it incredibly difficult to process information properly and reply to them timely. We already have spam filters, but they do not help clean up the mess we have in our inboxes.

LLMs can be paired together with these inboxes to read your emails and provide summaries, prioritize what emails to reply and even reply to your emails if allowed to. Even very simple, 3 billion parameter models can be deployed to take care of these tasks.

You can build a private assistant to take care of your emails end to end using an LLM connected with GMail API, or via IMAP and SMTP.

### Integrating with CRM Systems

Customer Relationship Management tools are extremely messy. Multiple teams use it, from sales to marketing to support and whatnot. CRMs are used for multiple things like storing customer data, call transcripts, feedback, and a ton of other data. And this data needs to be shared across teams. Hence, maintaining CRMs can be very complicated for everyone.

LLMs can be integrated with CRM APIs to simplify a ton of workflows, for example, LLMs can [generate meeting notes](https://www.mercity.ai/blog-post/gpt-nlp-in-sales#sales-call-analytics), which can save salespeople a ton of time. It can extract valuable insights from support and customer meetings for marketing teams and put everything in a proper, consumable format.

LLMs can compress information spread in CRMs and generate simple reports for pretty much anything or any specific customer you have.

### Integrating with CMS

Similar to CRMs, content management systems, and pipelines also have multiple functionalities, from creating and writing content to editing and to SEO optimization and whatnot. Language models can easily be integrated with every bit of these pipelines.

LLMs can be used to generate content, edit, and remove any unnecessary parts. LLM agents can also be deployed to plan, and generate content outlines, and then go ahead and generate the actual content and publish it.

WordPress APIs are one of the best and easiest to integrate with LLMs as you can access almost all parts of the pipeline.

## Challenges of Integrating Tools with LLMs

Even though we have outlined many approaches to integrate LLMs with tools, there are still many challenges that make this task difficult. Let’s go over them.

### Context Overflow

As seen with Toolformer and OpenAI function calling, context overflow is a big issue. This happens because we need to prompt the model with the tools we want to integrate. This means we need to add the tool names, descriptions of the tools, examples of how to use it when to use it, and more details. This can lead to major issues like a reduction in the output length because the prompt itself is so long Or a significant increase in token usage costs.

### Accuracy drops as the number of tools grows

This is pretty evident. As the number of tools integrated with an LLM increases, maintaining accuracy and efficiency can become a difficult task. If your tools are very similar to each other, or if you don’t provide good enough examples, the model can get confused and call the wrong functions at times. Or not call a function at all! This can be fixed with better prompting.

### Latency

Latency is a significant challenge when integrating tools with LLMs. The time it takes for data to be processed and for results to be produced from the tools. High latency can lead to delays in decision-making and can negatively impact the user experience. This is particularly problematic in real-time applications, where delays of even a few seconds can have significant issues.

### Trust

This is not a huge issue, but if you are using the model to generate code or to act on your behalf, you need to trust the model. If the model, for example, replies incorrectly or deletes the wrong files from your folder, it can cause major problems. This can simply be fixed by making sure there is a human in the loop and reviewing the steps taken by the LLM.

</details>

<details>
<summary>Introduction</summary>

## Introduction

Large Language Models (LLMs) excel in generating text but often struggle to produce structured output. By leveraging [Pydantic](https://docs.pydantic.dev/latest/)'s type validation and prompt engineering, we can enforce and validate the output generated by LLMs.

_All code examples in this blog post are written in Python. The LLM used is [OpenAI's gpt-3.5-turbo](https://platform.openai.com/docs/guides/gpt)._

## Query the LLM

To query the LLM, we use the following function:

```python
import openai

def query(prompt: str) -> str:
    """Query the LLM with the given prompt."""
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[\
            {\
                "role": "user",\
                "content": prompt,\
            }\
        ],
        temperature=0.0,
    )
    return completion.choices[0].message.content

```

We then call the function with a simple question:

```python
response = query("What is the largest planet in our solar system?")
print(response)
'The largest planet in our solar system is Jupiter.'

```

## Enforcing JSON output with a prompt

In our prompt, we can ask the LLM to respond in a certain format:

````python
prompt = """
I will ask you questions and you will respond. Your response should be in the following format:
```json
{
    "thought": "How you think about the question",
    "answer": "The answer to the question"
}
```
"""

````

Then, we query the model:

```python
question = "What is the largest planet in our solar system?"
response = query(prompt + question)
print(response)
'{
    "thought": "This is a factual question that can be answered with scientific knowledge.",
    "answer": "The largest planet in our solar system is Jupiter."
}'

```

This is great, because we can easily parse the structured output:

```python
import json

parsed_response = json.loads(response)
print(parsed_response["answer"])
'The largest planet in our solar system is Jupiter.'

```

## Validating the output

```python
from pydantic import BaseModel

class ThoughtAnswerResponse(BaseModel):
    thought: str
    answer: str

raw_response = query(prompt)

# Note: When you are using pydantic<2.0, use parse_raw instead of model_validate_json
validated_response = ThoughtAnswerResponse.model_validate_json(raw_response)

print(validated_response)
thought='This is a factual question that can be answered with scientific knowledge.' answer='The largest planet in our solar system is Jupiter.'

print(type(validated_response))
<class 'ThoughtAnswerResponse'>

```

## Using the Pydantic model in the prompt

At this moment, we describe our response format in two places:

- a JSON description in our prompt
- a corresponding Pydantic model

When we want to update the response format, we need to change both the prompt and the Pydantic model. This can cause inconsistencies.

We can solve this by [exporting the Pydantic model to a JSON schema](https://docs.pydantic.dev/latest/usage/json_schema/) and adding the schema to the prompt. This will make the response and the Pydantic model consistent.

````python
response_schema_dict = ThoughtAnswerResponse.model_json_schema()
response_schema_json = json.dumps(response_schema_dict, indent=2)

prompt = f"""
I will ask you questions, and you will respond.
Your response should be in the following format:
```json
{response_schema_json}
```
"""

````

The prompt will now look like this:

````
I will ask you questions, and you will respond. Your response should be in the following format:
```json
{
    "properties": {
        "thought": { "title": "Thought", "type": "string" },
        "answer": { "title": "Answer", "type": "string" }
    },
    "required": ["thought", "answer"],
    "title": "ThoughtAnswerResponse",
    "type": "object"
}

````

The response will look like this:

```json
{
"thought": "The largest planet in our solar system is Jupiter.",
"answer": "Jupiter"
}

```

Now, whenever you change the Pydantic model, the corresponding schema will be put in the prompt. Note that the schema has become more complex than it was before. One benefit is that it allows us to be more specific in what responses we require.

## Error handling

The LLM may still produce results that are not consistent with our model. We can add some code to catch this:

```python
from pydantic import ValidationError

try:
    validated_response = ThoughtAnswerResponse.model_validate_json(raw_response)
except ValidationError as e:
    print("Unable to validate LLM response.")
    # Add your own error handling here
    raise e

```

## Enforce specific values using a Literal

Sometimes, you want to enforce the use of specific values for a given field. We add the field "difficulty" to our response object. The LLM should use it to provide information about the difficulty of the question. In a regular prompt, we would do the following:

````python
prompt = """Your response should be in the following format:
```json
{
  "thought": "How you think about the question",
  "answer": "The answer to the question",
  "difficulty": "How difficult the question was. One of easy, medium or hard"
}
```
"""

````

Of course, the model could potentially still use other values. To validate it, we would need to write custom code.

With Pydantic, it is a lot easier. We create a new type called `Difficulty` using a [Literal](https://docs.python.org/3/library/typing.html#typing.Literal). A Literal allows us to specify the use of a select list of values. We add a `Difficulty` type hint to the `difficulty` field in our Pydantic model:

```python
from typing import Literal

from pydantic import BaseModel

# We create a new type
Difficulty = Literal["easy", "medium", "hard"]

class ThoughtAnswerResponse(BaseModel):
    thought: str
    answer: str
    difficulty: Difficulty

```

The LLM responds may respond with a value we do not allow:

```json
{
"thought": "The largest planet in our solar system is Jupiter.",
"answer": "Jupiter",
"difficulty": "Unknown"
}

```

When we parse this result, Pydantic will validate the values for the `difficulty` field. `Unknown` does not match one of the values specified in the Literal type we have defined. So we get the following error:

```python
validated_response = ThoughtAnswerResponse.model_validate_json(response)

ValidationError: 1 validation error for ThoughtAnswerResponse
difficulty
    Input should be 'easy', 'medium' or 'hard' [type=literal_error, input_value='Unknown', input_type=str]

```

## Conclusion

By using Pydantic and prompt engineering, you can enforce and validate the output of LLMs. This provides you with greater control of the LLM output and allow you to build more robust AI systems.

</details>

<details>
<summary><h1 id="retrieval-augmented-generation-everything-you-need-to-know-about-rag-in-ai">Retrieval Augmented Generation: Everything You Need to Know About RAG in AI</h1></summary>

<h1 id="retrieval-augmented-generation-everything-you-need-to-know-about-rag-in-ai">Retrieval Augmented Generation: Everything You Need to Know About RAG in AI</h1>
<p>October 24, 2024</p>
<p><img src="https://www.weka.io/wp-content/uploads/files/2024/10/Retrieval-Augmented-Generation_featured-image.jpg" alt=""></p>
<h2 id="what-is-rag-in-ai">What is RAG in AI?</h2>
<p>Retrieval augmented generation (RAG) is a framework for blending generative models such as ChatGPT-3 or 4 with retrieval systems. Instead of solely relying on the knowledge base embedded within a model, which can be static or limited by its training cut-off date, <a href="https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/">RAG in AI</a> dynamically retrieves up-to-date, relevant information from external data sources such as documents, vector databases, or the web. With these search results, it helps the system answer questions with more accurate and contextually relevant responses.</p>
<h2 id="what-is-rag-in-generative-ai">What is RAG in Generative AI?</h2>
<p>To define retrieval augmented generation, it’s important to consider its origins.</p>
<h2 id="rag-ai-definition-a-history-of-rag-in-generative-ai">RAG AI Definition: A History of RAG in Generative AI</h2>
<h3 id="who-invented-retrieval-augmented-generation">Who invented retrieval augmented generation?</h3>
<p>Before the RAG framework existed, generative models primarily used static knowledge embedded in their training data. This left them prone to errors when dealing with real-time, factual questions or specialized topics outside their training corpus.</p>
<p>AI researchers from Facebook (now Meta) <a href="https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/">introduced a new approach</a> called retrieval augmented generation or RAG in a 2020 paper. The concept was a major leap forward, allowing models to dynamically access external knowledge repositories, combining the strengths of both tools: the precise, domain-specific information of retrieval systems and the natural language generation of generative models.</p>
<h3 id="what-does-rag-mean-in-ai-now-and-why-is-it-important">What does RAG mean in AI now and why is it important?</h3>
<p>Over time, the RAG AI meaning has further sharpened and the approach has taken on more importance. The technique has improved as retrieval algorithms such as dense vector search have advanced and integrated more effectively with language models. Retrieval augmented generation techniques can now retrieve highly relevant context even from vast, unstructured data sources.</p>
<p>By definition, retrieval augmented generation offers numerous advantages and has become important in the context of <a href="https://www.weka.io/learn/guide/ai-ml/what-is-ai/">AI</a> in several ways:</p>
<ul>
<li><strong>Accuracy.</strong> RAG enhances accuracy with more up-to-date information by pulling from new, open sources. This is especially important for current events or fast-moving fields like science and technology.</li>
<li><strong>Scalability.</strong> Instead of training a new model whenever new data becomes available, retrieval augmented generation allows existing models to query a knowledge base. This reduces computational cost and time.</li>
<li><strong>Domain specialization.</strong> RAG allows models to be highly specialized within certain domains without retraining. For example, integrating a legal or medical database enables the system to generate accurate answers on those topics.</li>
<li><strong>Bridging knowledge gaps.</strong> Pre-trained models such as GPT-4 often have knowledge cut-offs, but RAG overcomes this issue, allowing them to fetch up-to-date information.Data efficiency. The generative model doesn’t need to memorize everything; instead, it relies on external retrieval, reducing the need to load massive amounts of data into the model during training.</li>
</ul>
<h2 id="how-does-retrieval-augmented-generation-work">How Does Retrieval Augmented Generation Work?</h2>
<p>To break down how retrieval augmented generation works, consider its two larger phases, the three key components of retrieval augmented generation architecture, and how they work together in a process of several steps.</p>
<p>In terms of how to do retrieval augmented generation, it takes place in two or three basic phases, depending on how you characterize them.</p>
<ul>
<li><strong>Retrieval.</strong> Based on a query or input, retrieval systems such as search engines scan external knowledge bases—such as databases of documents or web pages—and retrieve relevant chunks of text or data.</li>
<li><strong>Generation.</strong> A generative model, such as a large language model (LLM), then uses this retrieved information to generate a more informed and precise response.</li>
<li><strong>Fusion mechanism.</strong> This blends the retrieved information with the query to improve accuracy.</li>
</ul>
<p>This combination of AI RAG architecture makes RAG extremely effective for providing real-time, up-to-date, and domain-specific answers, especially in cases where pre-trained models alone might lack the necessary information.</p>
<p>A step-by-step process characterizes the retrieval augmented generation pipeline:</p>
<ul>
<li><strong>User query/input.</strong> The process begins when the user provides a query or input to the retrieval augmented generation model. The query could be a question, prompt, or request for information that the model needs to respond to, such as, “What is a RAG definition in AI?”</li>
<li><strong>Encoding the query.</strong> The input query is transformed into a numerical representation (an embedding) using a language model such as BERT or GPT. This embedding captures the semantic meaning of the query, so the retrieval system can more easily find relevant information. It does this by encoding the query into a vector using a pre-trained neural network model that understands language semantics.</li>
<li><strong>Retrieval system (search phase).</strong> The system now needs to retrieve relevant information from an external knowledge base such as a set of documents, a database, or even the web). The retrieval system can use either traditional keyword-based methods (sparse retrieval) or modern methods (dense retrieval). The retrieval system scans through the source to find and curate the most relevant information.</li>
<li><strong>Ranking and filtering.</strong> The system ranks information based on relevance. Typically, only the top N documents (where N is a small number, like 5-10) are considered for further processing, ensuring the top sources have the most useful content for the query.</li>
<li><strong>Contextual embedding generation</strong>. Each retrieved document or text chunk is then also converted into a numerical embedding. This ensures the generative model can effectively incorporate the retrieved information when generating a response.</li>
<li><strong>Embedding context.</strong> The retrieved documents or their relevant sections are also turned into vectors, allowing the RAG AI model to understand and process their content.</li>
<li><strong>Fusion of retrieved information.</strong> The generative model now has the original query, the retrieved documents, and their embeddings. The next step is to fuse the information. This can be achieved with early fusion, in which retrieved documents are fused with the input query, and both are fed into the AI RAG model at the same time to generate a response, or late fusion, in which retrieved documents are used after the generative model starts producing text to refine or update the response.</li>
<li><strong>Response generation.</strong> The model uses the retrieved knowledge and the user’s input to generate a natural language response. RAG models for AI use both the initial input (query) and the additional information from the retrieved documents to generate coherent, informative responses.</li>
<li><strong>Post-processing (optional).</strong> In some retrieval augmented generation implementations, the generated response may go through post-processing. This could involve fact-checking, summarization where the information is condensed for brevity and clarity, and/or formatting the response in a user-friendly structure.</li>
<li><strong>Response delivery.</strong> Finally, the generated response is sent back to the user.</li>
</ul>
<p>A step-by-step example of retrieval augmented generation workflow (RAG workflow) works like this:</p>
<ul>
<li><strong>Input.</strong> The user asks, “How does climate change affect coral reefs?”</li>
<li><strong>Query encoding.</strong> A pre-trained language model encodes the input query into a vector.</li>
<li><strong>Retrieval.</strong> The system searches a database of scientific articles for relevant documents on coral bleaching, ocean acidification, and ecosystem changes based on the encoding.</li>
<li><strong>Ranking.</strong> The 5 most relevant articles are ranked and passed onto the next step.</li>
<li><strong>Contextual embedding.</strong> The top articles are converted into vector embeddings so that the generative model can “understand” the retrieved information.</li>
<li><strong>Fusion.</strong> The model combines the query and the top-ranked articles using context from both the query and retrieved information.</li>
<li><strong>Generation.</strong> The model generates a response, such as: “The main negative impacts from climate change that affect coral reefs are ocean warming and acidification. Both cause coral bleaching and disrupt marine ecosystems.”</li>
<li><strong>Post-processing.</strong> The response is refined for clarity or checked for factual correctness (optional).</li>
<li><strong>Response delivery.</strong> The response is sent to the user.</li>
</ul>
<h2 id="types-of-retrieval-augmented-generation">Types of Retrieval Augmented Generation</h2>
<p>In the evolving retrieval augmented generation landscape, there are various specialized <a href="https://blog.jayanthk.in/types-of-rag-an-overview-0e2b3ed71b82">types of RAG</a> that optimize the process in different ways to address different use cases. Here are a few notable types of retrieval augmented generation frameworks, and a brief discussion of how they differ:</p>
<ul>
<li><strong>Active RAG:</strong> Iterative query refinement for improved relevance</li>
<li><strong>Corrective RAG:</strong> Corrects or cross-checks output for factual accuracy</li>
<li><strong>Multimodal RAG:</strong> Incorporates multiple data types like text, images, and video for richer responses</li>
<li><strong>Advanced RAG:</strong> Uses cutting-edge retrieval methods (dense retrieval, transformers) for high performance</li>
<li><strong>Knowledge-intensive RAG:</strong> Specializes in very technical or domain-specific information</li>
<li><strong>Memory RAG:</strong> Retains and recalls previous interactions, improving the quality, continuity, and personalization of future responses</li>
<li><strong>Meta-learning RAG:</strong> Adapts quickly with few-shot learning or zero-shot capabilities</li>
</ul>
<h3 id="active-retrieval-augmented-generation">Active Retrieval Augmented Generation</h3>
<p>What is AI RAG with active retrieval? Active retrieval augmented generation (Active RAG) emphasizes dynamic interaction between the model and the retrieval system during the generation process, iteratively improving the relevance of retrieved information by refining queries in real-time.</p>
<p>The model actively engages in multiple rounds of query generation and retrieval to get better, more accurate, and contextually relevant information. For example, used with a customer support chatbot, the system could refine its search based on an initial query and user feedback, retrieving more specific troubleshooting steps with each interaction.</p>
<h3 id="corrective-retrieval-augmented-generation">Corrective Retrieval Augmented Generation</h3>
<p>What is retrieval augmented generation that is considered corrective? Corrective retrieval augmented generation (Corrective RAG) minimizes errors or hallucinations during the retrieval or generation phase to correct the model when it generates information that is inaccurate or not grounded in reality. This approach either retrieves additional sources to verify or cross-check information, or corrects the output during post-processing by comparing it to reliable external knowledge sources.</p>
<p>For example, as it generates legal advice, the system relies upon the correctness of a particular ruling. To validate it, the model retrieves multiple legal documents and cases to ensure its foundational information is grounded in fact and legally accurate.</p>
<h3 id="knowledge-intensive-retrieval-augmented-generation-ki-rag">Knowledge-Intensive Retrieval-Augmented Generation (KI-RAG)</h3>
<p>What is retrieval-augmented generation that is considered knowledge-intensive? Knowledge-intensive generative augmented retrieval focuses on domains that require deep, specialized knowledge, such as scientific research, law, or healthcare. This type of RAG is designed to retrieve highly technical or domain-specific information that is not generally available in the model’s pre-trained knowledge base.</p>
<p>For example, KI-RAG can assist scientific researchers by retrieving the most relevant studies, datasets, and citations from specialized academic databases like PubMed or arXiv to generate literature reviews or summaries.</p>
<h3 id="multimodal-retrieval-augmented-generation">Multimodal Retrieval Augmented Generation</h3>
<p>What is RAG AI that is considered multimodal? Multimodal retrieval augmented generation for images and other kinds of data (Multimodal RAG) enables information retrieval and generation across multiple data modalities such as text, images, audio, or video rather than being limited to text-based information. For example, an AI-powered museum guide could retrieve relevant information from textual databases about an artifact, and also pull up related images or videos to provide a more comprehensive experience for users asking about art history.</p>
<h3 id="advanced-retrieval-augmented-generation">Advanced Retrieval Augmented Generation</h3>
<p>What is RAG retrieval augmented generation that is considered advanced? Advanced retrieval augmented generation (Advanced RAG) refers to cutting-edge variations of the RAG framework that leverage more sophisticated mechanisms for retrieval and generation such as dense retrieval and other deep learning-based retrieval techniques, neural search algorithms, and cross-encoders. They may also incorporate more powerful models to improve performance in specific domains.</p>
<p>Advanced retrieval augmented generation is often used in medicine, to retrieve the latest research papers or clinical trials related to a patient’s symptoms and help generate a tailored diagnosis or treatment plan based on the most current medical knowledge.</p>
<h3 id="memory-augmented-retrieval-augmented-generation-memory-rag">Memory-Augmented Retrieval-Augmented Generation (Memory RAG)</h3>
<p>What is a memory-augmented retrieval augmented generation definition? Memory RAG introduces a persistent memory component that stores and retrieves previously generated responses or relevant facts during interactions. This type of RAG is useful in cases where the system needs to build on past conversations or retrieved information to generate more coherent and consistent outputs over time.</p>
<p>For example, a virtual assistant for technical support empowered in this way can remember previous troubleshooting steps and avoid repeating information, providing a more efficient and user-friendly experience over multiple sessions.</p>
<h3 id="meta-learning-or-few-shot-retrieval-augmented-generation">Meta-Learning or Few-Shot Retrieval-Augmented Generation</h3>
<p>Meta-learning, few-shot, or zero-shot learning methods allow RAG systems to improve their retrieval and generation capabilities with minimal data, so the system retrieves information and generates accurate responses with few or no examples.</p>
<p>For example, meta-learning retrieval augmented generation can allow an educational assistant to generate curriculum-specific answers based on a few examples, or an AI tutor to adapt to different subjects with little prior training.</p>
<h2 id="alternatives-to-retrieval-augmented-generation">Alternatives to Retrieval Augmented Generation</h2>
<p>How does retrieval augmented generation compare to other strategies for improving AI and LLM outputs?</p>
<h3 id="retrieval-augmented-generation-vs-fine-tuning">Retrieval Augmented Generation vs Fine Tuning</h3>
<p>Retrieval augmented generation connects an LLM to a curated external knowledge base, search engine, or database to improve outputs by integrating reliable information. A fine tuned model’s parameters are trained on a specialized dataset to improve performance on specific tasks.</p>
<p>The core AI RAG meaning is that the model supplements its generative capabilities with real-time retrieval of external knowledge. In contrast, fine-tuning allows the model to adapt its internal parameters to better handle specific tasks by learning from additional training data.</p>
<p>For these reasons, retrieval augmented generation is better-suited for real-time queries, evolving knowledge, while fine-tuning works best with domain-specific, static knowledge. In action, a news chatbot using RAG could pull up-to-date information on global events by retrieving relevant articles in real-time, while a legal advice chatbot fine-tuned on legal cases could generate expert responses on a narrow set of legal queries while struggling to adapt to new laws or regulations without re-training.</p>
<h3 id="rag-retrieval-augmented-generation-vs-semantic-search">RAG Retrieval-Augmented Generation vs Semantic Search</h3>
<p>Retrieval augmented generation and <a href="https://www.youtube.com/watch?v=buFay8nCdnc">semantic search</a> are both used in AI for information retrieval, but while the primary goal of RAG is to use both the user query and the retrieved data to generate responses, the primary focus in semantic search is to retrieve relevant information, not to generate new text.</p>
<p>Semantic search is typically used in search engines, recommendation systems, and document retrieval to surface the most contextually appropriate documents or answers. For example, a search engine retrieves the most relevant articles about renewable energy from its indexed database but doesn’t create a summary or new text.</p>
<h3 id="rag-gen-ai-vs-prompt-engineering-with-uncorrected-llms">RAG Gen AI vs prompt engineering with uncorrected LLMs</h3>
<p>There are significant differences between retrieval augmented generation AI vs uncorrected large language models (LLMs), particularly in information retrieval and access to external knowledge beyond the training data.</p>
<p>LLM retrieval augmented generation is more accurate and less vulnerable to the AI “hallucinations” that chatbots often present. Retrieval augmented generation LLMs can also include specific information the user includes, like the most recent data available on the subject or an internal dataset for real-time applications and fact-based, dynamic knowledge tasks.</p>
<h3 id="ai-rag-vs-pretraining">AI RAG vs Pretraining</h3>
<p>Retrieval augmented generation and pretraining are two distinct processes in the development and use of AI models, particularly in the context of large language models (LLMs). <a href="https://blogs.nvidia.com/blog/what-is-a-pretrained-ai-model/">Pretraining</a>, in contrast to RAG as already described, equips a model with broad linguistic and factual knowledge, enabling it to handle general tasks without relying on external data sources, but at the risk of outdated or incomplete information.</p>
<h2 id="retrieval-augmented-generation-examples">Retrieval Augmented Generation Examples</h2>
<p>Any complete picture of retrieval augmented generation explained should include examples of current products on the market. Here are some commonly-used retrieval augmented generation applications:</p>
<p>Google products related to retrieval-augmented generation include <a href="https://cloud.google.com/enterprise-search">Vertex AI Search</a> and <a href="https://cloud.google.com/bigquery">BigQuery</a>. Users build and deploy AI applications and ML models with Vertex AI.</p>
<p>With the fully managed BigQuery data warehouse, users can engage in large-scale analysis to support business intelligence, ML applications, and geospatial analysis.</p>
<p>Retrieval augmented generation AWS capabilities include <a href="https://aws.amazon.com/bedrock/">Amazon Bedrock</a> knowledge bases, <a href="https://aws.amazon.com/q/business/">Amazon Q for Business</a>, <a href="https://aws.amazon.com/kendra/">Amazon Kendra</a>, and <a href="https://lancedb.com/">LanceDB</a>.</p>
<p>Amazon Bedrock knowledge bases integrate with <a href="https://www.weka.io/learn/glossary/ai-ml/generative-ai-understanding-the-next-wave-of-artificial-intelligence/">generative AI</a> applications to search data and answer natural language questions. The Amazon Q for Business tool allows users to quickly create, tune, and deploy RAG solutions.</p>
<p>The Amazon Kendra intelligent search engine can search data lakes and connect to third-party data sources. And the LanceDB open-source vector database can connect directly to S3 to simplify embedding retrieval, filtering, and management.</p>
<p>Generative AI RAG options with Oracle include the platform’s Generative AI Agents and Oracle Cloud Infrastructure which combine LLMs and RAG with the user’s enterprise data.For retrieval augmented generation, Azure AI Search provides features that index data across sources and formats. The process is optimized for relevance and speed to ensure that generative models can retrieve the best possible data for response generation.</p>
<h2 id="retrieval-augmented-use-cases">Retrieval Augmented Use Cases</h2>
<p>There are three main zones or types of retrieval augmented generation use cases:</p>
<ul>
<li><strong>Customer support applications</strong> use RAG to pull the most recent and relevant documentation or troubleshooting guides</li>
<li><strong>Scientific research applications</strong> leverage updated papers and datasets for technical queries</li>
<li><strong>Conversational AI platforms</strong> retrieve knowledge in real-time to provide accurate answers via chatbots or virtual assistants</li>
</ul>
<p>Here are some examples of how RAG AI is used in various industries:</p>
<p><strong>Customer support systems and technical support automation.</strong> A RAG-powered chatbot can retrieve knowledge base articles, product documentation, or FAQs from a company’s database to answer customer inquiries in real-time and assist users in resolving technical issues with software or hardware. Instead of relying solely on pre-trained knowledge, the chatbot can pull specific troubleshooting steps or product information to provide accurate and contextual responses.</p>
<p>For example, a customer asks, “How do I reset my router?” or “How do I fix a blue screen error on Windows 11?” The chatbot retrieves and responds with a tailored version of the latest router reset instructions from the company’s technical support documents or relevant troubleshooting steps from the recent Windows support articles and generates a step-by-step guide to help the user fix their specific issue.</p>
<p><strong>Legal research and advice.</strong> A legal assistant AI powered by retrieval augmented generation can pull relevant case laws, statutes, or legal documents from legal databases like <a href="https://legal.thomsonreuters.com/en/westlaw">Westlaw</a> or <a href="https://www.lexisnexis.com/en-us">LexisNexis</a> to respond to a query. The AI then uses the retrieved data to generate a legal memo or offer advice on the legal issue at hand.</p>
<p>For example, a lawyer queries, “What precedents exist for wrongful termination cases in California from the last two years?” The RAG system retrieves relevant case law and summarizes the findings in a concise memo.</p>
<p><strong>Medical research or diagnosis.</strong> Virtual healthcare assistants and medical professionals can access recent research papers, current clinical guidelines, or patient records with retrieval augmented generation to assist with diagnosis or recommending treatments.</p>
<p>For example, a doctor asks, “What do applicable clinical guidelines for the management of type 2 diabetes recommend for patients with neuropathic pain and multiple comorbidities?” The system retrieves the relevant guidelines and research on diabetes management and generates details focused on these specific patients for the doctor to review.</p>
<p><strong>Scientific research assistance.</strong> A RAG system for scientists can pull the latest scientific articles, papers, or experiment data from academic databases such as <a href="https://pubmed.ncbi.nlm.nih.gov/">PubMed</a>, <a href="https://arxiv.org/">ArXiv</a>, or <a href="https://www.springer.com/us">Springer</a>. It then uses this information to generate insights, summaries, or assist in writing research proposals or literature reviews.</p>
<p>For example, a researcher queries, “What recent advancements in quantum computing are most likely to lead to practical applications?” The AI retrieves the latest publications and papers on quantum computing and summarizes key breakthroughs for the researcher.</p>
<p><strong>Financial advisory systems.</strong> Retrieval augmented generation pulls real-time data from stock markets, financial reports, and economic indicators. In this way, RAG systems <a href="https://www.weka.io/learn/machine-learning-and-gpu/gpu-ai-in-financial-services/">help financial advisors</a> offer better advice and investment recommendations and empower retail investors to make more informed decisions.</p>
<p>For example, an investor asks, “Are current market trends in renewable energy stocks favorable for solo investors?” The RAG system retrieves real-time stock performance data and recent news articles from within the renewable energy sector and analyzes current market trends in this niche area to provide an informed answer.</p>
<p><strong>Academic writing and content generation.</strong> RAG can assist students, researchers, or writers by retrieving articles, research papers, or historical documents and using them to generate summaries or reports, or assist in academic drafting.</p>
<p>For example, a student looking for an unusual paper topic might ask, “What is the most controversial theme in Shakespeare’s Hamlet?” The system retrieves scholarly articles and expert opinions on Hamlet, and compares its most debated or polarizing themes and their meaning.</p>
<p><strong>E-commerce product recommendations.</strong> Retrieval augmented generation can provide personalized recommendations based on real-time customer queries and external reviews or product specifications.</p>
<p>For example, a shopper asks, “Which digital camera under $1,000 is best for wildlife photography?” The system makes a recommendation along with a brief description of each choice based on product reviews, expert opinions, and e-commerce listings.</p>
<p><strong>Real-time news generation.</strong> RAG can be used in journalism or content creation platforms to generate real-time news articles by retrieving the latest information from reliable sources like news agencies, social media, or government databases.</p>
<p>For example, a news agency needs an article on a breaking news event. The RAG system retrieves real-time information from multiple sources such as social media updates and press releases and generates a draft article, summarizing key facts about the breaking news event.</p>
<p><strong>Language translation and multilingual summarization.</strong> Retrieval augmented generation can be used for real-time, domain-specific language translation and summarization by retrieving relevant terminology and context-specific phrases from a multilingual database.</p>
<p>For example, a business asks, “Can you translate this legal document into French?” The system retrieves relevant legal terminology from a bilingual legal database and generates a precise translation that maintains the original document’s context.</p>
<p><strong>Business intelligence and reporting.</strong> RAG systems can pull data from business intelligence tools, reports, and databases to generate insights, analyses, or reports based on the latest business performance metrics.</p>
<p>For example, a user might query, “What products did well sell the most in Q3?” The AI retrieves sales data and generates a report highlighting best-selling products, sales trends, and insights.</p>
<p><strong>Virtual personal assistants.</strong> When powered by retrieval augmented generation these platforms can retrieve calendar events, emails, documents, or other relevant data to assist users with tasks such as scheduling, organizing, or answering complex queries.</p>
<p>For example, a user asks, “Can you get me ready for my meetings today?” The system retrieves information from the user’s calendar and email to generate a detailed agenda for the day, including meeting times, participants, and topics.</p>
<p><strong>Content moderation and policy compliance.</strong> RAG can retrieve and cross-reference community guidelines, legal regulations, and past precedents to help determine whether user-generated content complies with the platform policies.</p>
<p>For example, a content reviewer might ask, “Does this post violate our policy on hate speech?” The system retrieves the relevant policy sections and past similar cases, providing a well-informed recommendation that human decision-makers can then review.</p>
<p><strong>Tourism and travel assistance.</strong> RAG systems can retrieve travel guides, hotel information, flight details, and local events to help users plan trips and get recommendations for accommodations, transportation, and activities.</p>
<p>For example, a traveler asks, “What are the best activities in Paris for a weekend visit?” The system retrieves data from travel blogs, tourist websites, and event listings to generate a curated itinerary for the user.</p>
<p><strong>Retrieval augmented generation for code.</strong> RAG can be used to develop software, generate code, documentation, and fix errors. For example, a developer asks, “Write code that asks a user for their first name when they initiate a new chat,” and the system generates the appropriate code given other parameters.</p>
<h2 id="how-to-implement-retrieval-augmented-generation">How to Implement Retrieval Augmented Generation</h2>
<p>To implement <a href="https://arxiv.org/pdf/2005.11401">RAG for knowledge-intensive NLP tasks</a>, follow this step-by-step guide to retrieval augmented generation implementation:</p>
<p><strong>Step 1: Set up a document store or knowledge base</strong></p>
<ul>
<li>Choose a document store with relevant knowledge or data. It can be structured data (databases), unstructured data (text documents, articles), or external APIs (news sources, medical records).</li>
<li>You can use retrieval augmented generation tools like Elasticsearch, FAISS (Facebook AI Similarity Search), or Pinecone to build a document store with vector embeddings for efficient retrieval.</li>
</ul>
<p><strong>Step 2: Preprocess and index the documents</strong></p>
<ul>
<li>Preprocess the documents by creating representative semantic embeddings. These are intended for use in a high-dimensional vector space, where semantically similar documents are grouped closer together.</li>
<li>Convert each document into an embedding with a transformer-based model such as BERT or SBERT. Store them in the chosen vector database for efficient retrieval later.</li>
</ul>
<p><strong>Step 3: Build the retrieval system</strong></p>
<ul>
<li>Implement a system that encodes the user query as a vector using the same model that was used to encode the documents.</li>
<li>The system should also perform a similarity search between the query and document embeddings to find the top-k most relevant documents and return them (or passages of them) as input to the generative model.</li>
</ul>
<p><strong>Step 4: Integrate with the generative model</strong></p>
<ul>
<li>Concatenate the original query with the retrieved context, documents, and other knowledge into the generative model.</li>
</ul>
<p><strong>Step 5: Generate the response</strong></p>
<ul>
<li>The response is informed by both its internal knowledge and the retrieved external knowledge.</li>
</ul>
<p><strong>Step 6: Post-processing and output</strong></p>
<ul>
<li>Summarization, fact-checking, or ranking can refine the generated response before it is output to the user.</li>
</ul>
<h3 id="how-to-use-retrieval-augmented-generation-technical-tools-and-libraries-for-implementing-rag">How to Use Retrieval Augmented Generation: Technical Tools and Libraries for Implementing RAG</h3>
<p>Several libraries and platforms provide retrieval augmented generation tools for implementing RAG systems:</p>
<ul>
<li><strong><a href="https://huggingface.co/">Hugging Face’s</a></strong> transformers and datasets libraries provide pre-trained transformer models such as GPT and BART as well as datasets for fine-tuning retrieval systems.</li>
<li><strong><a href="https://ai.meta.com/tools/faiss/">Facebook AI Similarity Search (FAISS)</a></strong> is an open-source library for efficient similarity search and clustering of dense vectors, making it ideal for document retrieval tasks.</li>
<li><strong><a href="https://haystack.deepset.ai/">Haystack (by deepset.ai)</a></strong> is a Python framework that helps build end-to-end NLP pipelines, including RAG pipelines for information retrieval and response generation.</li>
<li><a href="https://www.elastic.co/elasticsearch"><strong>ElasticSearch</strong></a> is a powerful search engine that can index and retrieve documents in response to queries.</li>
<li><strong><a href="https://openai.com/api/">OpenAI API</a></strong> easily integrates powerful generative models like GPT, which can be used in conjunction with custom retrieval systems.</li>
</ul>
<h2 id="retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks">Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks</h2>
<p>Using retrieval augmented generation for knowledge intensive NLP tasks such as legal, scientific, and technical domains is a strategic move. For these tasks, users often ask complex, detailed questions that require precise answers that demand accurate, specific information. RAG systems introduce real-time or highly specialized knowledge from external sources that the generative model alone might not have.</p>
<p>AI retrieval augmented generation reduces hallucination by grounding the generation process in real, retrieved data, making the responses more factual and accurate. And for tasks which depend on rapidly evolving knowledge such as technology or science, RAG systems can continuously retrieve and integrate new documents or research papers, keeping the system up-to-date without requiring frequent model retraining.</p>
<p>Knowledge-intensive applications rely heavily on interpretability, particularly in fields like law or medicine, where users need to understand the foundation for any advice given. RAG systems provide supporting documents or references, allowing users to verify the provenance of the information used in the response.</p>
<h2 id="benefits-of-retrieval-augmented-generation">Benefits of Retrieval Augmented Generation</h2>
<p>Retrieval augmented generation offers several basic benefits:</p>
<ul>
<li><strong>Accuracy.</strong> RAG ensures up-to-date, factually grounded responses by retrieving real-world data.</li>
<li><strong>Broader coverage.</strong> It expands a model’s knowledge base without extensive retraining.</li>
<li><strong>Adaptability.</strong> RAG adapts to fast-changing domains by retrieving fresh data.</li>
<li><strong>Efficiency.</strong> It reduces the need for model fine-tuning and lowers computational demands.</li>
<li><strong>Transparency.</strong> Users can verify the output by reviewing the retrieved sources, enhancing trust.</li>
</ul>
<h2 id="retrieval-augmented-generation-best-practices">Retrieval Augmented Generation Best Practices</h2>
<p>There are several key retrieval augmented generation benchmarks and best practices essential to evaluating retrieval augmented generation systems.Here are a few tools for benchmarking large language models in retrieval-augmented generation:</p>
<ul>
<li><strong>Natural questions (NQ) dataset.</strong> This contains real-world questions with long and short answer types, typically requiring retrieval from Wikipedia. NQ measures a model’s ability to retrieve relevant documents and generate precise, fact-based answers, making it ideal for evaluating RAG’s performance in question-answering tasks.</li>
<li><strong>MS MARCO (Microsoft Machine Reading Comprehension).</strong> MS MARCO is a large-scale dataset for document retrieval and passage ranking that contains real queries from Bing search logs with corresponding passages and answers. It is used to test RAG’s ability to retrieve the most relevant passages and generate high-quality, coherent answers.</li>
<li><strong>TriviaQA.</strong> This question-answering dataset pairs questions with web documents that contain the correct answers. It evaluates how well RAG can retrieve relevant, factual information and incorporate it into accurate responses, especially for trivia-based or general knowledge queries.</li>
<li><strong>FEVER (Fact Extraction and Verification).</strong> This dataset is designed for fact verification and retrieval. FEVER provides claims and asks models to retrieve evidence and verify the correctness of the claims and is ideal for evaluating how well RAG retrieves relevant evidence and generates factual, grounded responses.</li>
<li><strong>TREC CAR (Complex Answer Retrieval).</strong> This benchmarks complex information retrieval with the task of retrieving and generating comprehensive answers to long, multi-faceted questions using multiple retrieved Wikipedia articles.</li>
<li><strong>Open-domain QA datasets.</strong> Datasets such as SQuAD Open and Web Questions focus on open-domain questions—those posed without a predefined context. This requires RAG systems to handle knowledge-intensive tasks with minimal supervision.</li>
<li><strong>Eli5.</strong> This dataset of open-domain questions typically asked in online forums often features complex, multi-sentence answers and detailed explanations. Eli5 evaluates how well RAG systems can generate long-form, informative responses based on retrieved content, especially for educational or explanation-heavy use cases.</li>
</ul>
<p>Searching for best practices in retrieval-augmented generation systems invariably leads to these key tactics:</p>
<ul>
<li><strong>Use pretrained embeddings for retrieval.</strong> High-quality embeddings ensure that semantically relevant documents are retrieved, even if the language of the query and document differ slightly.</li>
<li><strong>Optimize the retrieval.</strong> Store and search document embeddings efficiently with vector databases (like FAISS, Pinecone, or Elasticsearch) to improve response time and accuracy.</li>
<li><strong>Choose the Right retrieval augmented generative AI model.</strong> Use dual-encoder or bi-encoder models such as dense passage retrieval (DPR), which scale well to large datasets and provide better retrieval accuracy compared to simpler methods like BM25. They also create separate embeddings for queries and documents, allowing fast similarity searches.</li>
<li><strong>Incorporate re-ranking.</strong> After initial retrieval, re-rank the documents using a more sophisticated model to ensure the most relevant documents are prioritized.</li>
<li><strong>Tune the retrieval-generation balance.</strong> Too much reliance on retrieval may result in responses that are highly factual but lack creativity, while too little may cause inaccuracies.</li>
<li><strong>Regularly update the knowledge base.</strong> Ensure that the document store is frequently updated to reflect the latest information, especially in dynamic fields like healthcare, finance, or technology, to minimize the risk of generating outdated or incorrect responses.</li>
<li><strong>Implement feedback loops for continuous improvement.</strong> Collect user feedback on the quality of retrieved documents and generated responses. Use it to retrain or fine-tune both retrieval and generative components over time. Such a feedback loop allows the system to continuously adapt to user preferences, improve performance, and optimize retrieval and generation.</li>
<li><strong>Test for latency and efficiency.</strong> Ensure that the retrieval component is optimized for low-latency searches and that the generative model can process results efficiently. Consider using techniques like approximate nearest neighbor (ANN) searches to speed up retrieval. Balancing accuracy with speed is essential for smooth user experiences.</li>
<li><strong>Ensure data privacy and security.</strong> Any RAG system that deals with sensitive data such as medical records or financial information must ensure that its knowledge store is encrypted, control access to it, and implement privacy-preserving methods to prevent data breaches and protect user privacy.</li>
<li><strong>Evaluate responsiveness to ambiguous queries.</strong> Ensure the system can manage ambiguous or incomplete retrieval augmented generation prompts and queries by retrieving multiple potential contexts or prompting users for clarification.</li>
</ul>

</details>

<details>
<summary>As large language models become more integrated into computational systems, their role in enhancing application efficiency and accuracy grows. However, this expanded capability brings new risks when executing autonomously generated code.</summary>

As large language models become more integrated into computational systems, their role in enhancing application efficiency and accuracy grows. However, this expanded capability brings new risks when executing autonomously generated code.

This blog post explores how to establish a secure Python sandbox for LLM agents. We will cover the threats involved with LLM-generated code and introduce a sandbox solution using gVisor and Jupyter Notebook.

## LLM Agents

Incorporating LLMs into software applications can be achieved in several ways, which lie on the agency spectrum. At one end of this spectrum is the simple use of LLMs, where the software makes API calls and parses responses. While straightforward, this approach is vulnerable to errors and hallucinations. Towards the high agency end are more sophisticated agentic systems where LLMs have the autonomy to use tools to achieve tasks. These systems stand out for their ability to navigate scenarios lacking a predetermined workflow, which is often the case in real-world applications.

While some agentic systems define custom workflows by deciding to use one of the predefined functions called **tools** with specific parameters, at the high end of the agency spectrum LLM agents can write and execute their own code. This capability is particularly useful in dynamic environments or for complex tasks like creating custom data visualizations, where precise and customized solutions are necessary. By generating and executing code, these agents can adapt to the specific requirements of each problem, achieving a level of customization that standard functions cannot provide.

## Sandboxing Code

With increased agency in LLM systems comes increased risk. Executing potentially unsafe code generated by these agents can expose systems to security issues, including arbitrary code execution ( `os.system`, `subprocess`, etc.), resource exhaustion (Denial-of-service attack via CPU, memory or disc overload), file system access (unauthorized reads/writes to files) and many others. Implementing a secure method to execute this code is crucial.

Mitigating these risks can be achieved through the implementation of a secure Python sandbox. The essential goal of such sandbox is to manage resources and create safe execution environments that encapsulate potentially harmful code, preventing it from affecting the broader system.

## The Demo Solution

One potential solution to securely execute Python code remotely consists of a FastAPI server that runs a jupyter notebook kernel inside a gVisor container. Here is how different components of the solution work together:

- **Jupyter Notebook** allows to run interactive code notebooks. Jupyter kernels support different environments, including Python, R, Julia, JavaScript, and others. Jupyter kernels are isolated and have limited permissions but do not offer other security features. In our solution Jupyter Notebook plays the role of a code execution environment that works out of the box.
- **FastAPI** is a modern web framework for building APIs with Python. FastAPI serves as the interface between the LLM agent and the Jupyter kernel, allowing the agent to send code for execution over the network and receive results. FastAPI helps us to decouple the agent and the execution environment, which is important for resource management and sandbox scaling.
- **gVisor** is a user-space kernel that provides a secure environment for running untrusted code. It acts as a barrier between the code and the host operating system, preventing unauthorized access to system resources. gVisor intercepts system calls made by the code and enforces security policies, ensuring that only safe operations are allowed. This is a crucial layer of protection for the host system from potential threats posed by executing arbitrary code.

The following code runs FastAPI sandbox server:

```hljs kotlin
# ./main.py
import asyncio
from asyncio import TimeoutError, wait_for
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from jupyter_client.manager import AsyncKernelManager
from pydantic import BaseModel

app = FastAPI()

allowed_packages = ["numpy", "pandas", "matplotlib", "scikit-learn"]
installed_packages: List[str] = []

class CodeRequest(BaseModel):
    code: str

class InstallRequest(BaseModel):
    package: str

class ExecutionResult(BaseModel):
    output: str

@asynccontextmanager
async def kernel_client():
    km = AsyncKernelManager(kernel_name="python3")
    await km.start_kernel()
    kc = km.client()
    kc.start_channels()
    await kc.wait_for_ready()
    try:
        yield kc
    finally:
        kc.stop_channels()
        await km.shutdown_kernel()

async def execute_code(code: str) -> str:
    async with kernel_client() as kc:
        msg_id = kc.execute(code)
        try:
            while True:
                reply = await kc.get_iopub_msg()
                if reply["parent_header"]["msg_id"] != msg_id:
                    continue
                msg_type = reply["msg_type"]
                if msg_type == "stream":
                    return reply["content"]["text"]
                elif msg_type == "error":
                    return f"Error executing code: {reply['content']['evalue']}"
                elif msg_type == "status" and reply["content"]["execution_state"] == "idle":
                    break
        except asyncio.CancelledError:
            raise
    return ""

async def install_package(package: str) -> None:
    if package not in installed_packages and package in allowed_packages:
        async with kernel_client() as kc:
            try:
                kc.execute(f"!pip install {package}")
                while True:
                    reply = await kc.get_iopub_msg()
                    if (
                        reply["msg_type"] == "status"
                        and reply["content"]["execution_state"] == "idle"
                    ):
                        break
                installed_packages.append(package)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error installing package: {str(e)}")

@app.post("/install")
async def install(request: InstallRequest):
    try:
        await wait_for(install_package(request.package), timeout=120)
    except TimeoutError:
        raise HTTPException(status_code=400, detail="Package installation timed out")
    return {"message": f"Package '{request.package}' installed successfully."}

@app.post("/execute", response_model=ExecutionResult)
async def execute(request: CodeRequest) -> ExecutionResult:
    try:
        output = await wait_for(execute_code(request.code), timeout=120)
    except TimeoutError:
        raise HTTPException(status_code=400, detail="Code execution timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ExecutionResult(output=output)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
```

This minimalistic sandbox implementation exposes two endpoints: `/execute` for executing code and `/install` for installing whitelisted packages. Code execution is performed in a separate Jupyter kernel, which is managed by the `AsyncKernelManager`, and the console output text is returned to the client. The server is designed to handle timeouts and exceptions gracefully.

The following Dockerfile builds the container image for the sandbox server:

```hljs bash
# Dockerfile
FROM jupyter/base-notebook

WORKDIR /app
COPY main.py /app/main.py
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Switch to jovyan non-root user defined in the base image
USER jovyan

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Although this dockerfile is very simple, it enables deployment of the sandbox solution in a containerized environment. The container runs as a non-root user, which is a good security practice.

At dida we use **Google Kubernetes Engine** to manage our **Kubernetes** clusters, which natively supports gVisor as a container runtime. To enable deployment of gVisor protected workloads, we first need to create a node pool that enables GKE sandbox. Note that in order to turn this security feature on the cluster should have a second standard node pool because GKE-managed system workloads must run separately from untrusted sandboxed workloads.

Once the node pool is created, we can deploy the sandbox container image to the cluster with the following Kubernetes manifest:

```hljs yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-sandbox
  namespace: demos
  labels:
    app: agent-sandbox
spec:
  replicas: 1
  selector:
    matchLabels:
      app: agent-sandbox
  template:
    metadata:
      labels:
        app: agent-sandbox
    spec:
      runtimeClassName: gvisor
      containers:
        - name: agent-sandbox
          image: "${IMAGE_REGISTRY}/${IMAGE_REPOSITORY}:${IMAGE_TAG}"
          ports:
            - name: http
              containerPort: 8000
              protocol: TCP
          resources:
            requests:
              memory: "250Mi"
              cpu: "250m"
            limits:
              memory: "500Mi"
              cpu: "500m"
```

Note that the `runtimeClassName` field is set to `gvisor`, which instructs Kubernetes to use gVisor as the container runtime for this deployment. To control sandbox resource allocation, we set resource requests and limits for CPU and memory. This ensures that the sandbox container has sufficient resources to operate while preventing it from consuming excessive resources that could affect other workloads in the cluster.

### **Capabilities of the Demo Solution**

The demo solution is easy to deploy and manage, making it suitable for various use cases. The interface is accessible via a REST API, which is framework-agnostic and can be integrated with any LLM agent. The solution is designed to be extensible, allowing for the addition of new features and enhancements as needed. For example, one can add support for additional programming languages or integrate with other tools and services. In addition, the solution can be easily scaled to handle increased workloads by deploying multiple instances of the sandbox container in a Kubernetes cluster. Containerization minimizes performance overhead compared to traditional virtual machines, making it suitable for high-performance applications.

While being a proof of concept for code sandbox, the demo showcases the following security features:

- A standalone containerized sandbox provides isolation and minimizes dependencies between agents.
- Python imports are limited, reducing risks associated with dependency threats.
- The following security features are provided by using gVisor as the container runtime:
  - Isolation of the execution environment from the host system.
  - Sandboxing gVisor itself from the host kernel.
  - Running the container with least amount of privileges.
  - Continuous development and maintenance of gVisor by security experts, ensuring up-to-date security features.
- Kubernetes enables efficient CPU, memory, and storage resource management.

### **Limitations of the Demo Solution**

The following limitation of the demo should be addressed before it can be used in production:

- At the moment every request to the sandbox creates a new Jupyter kernel, which is not efficient. This can be improved by reusing existing kernels or implementing a more sophisticated kernel management strategy.
- In addition to managing the lifecycle of Jupyter kernels, the solution should also handle session and state management. This includes authentication, authorization, and maintaining user sessions to ensure secure access to the sandbox environment.
- It might be beneficial to LLM agents to generate responses that include non-textual elements, in particular images. The current solution does not support these types of responses, even though image output is supported by Jupyter.
- Filter sandbox ingress and egress traffic to prevent data exfiltration and unauthorized access to external resources.

## Conclusion

The demo solution includes features like easy deployment, framework-agnostic integration, and scalability through Kubernetes. It effectively isolates execution environments using gVisor, ensuring robust security with minimal performance overhead. However, some limitations need addressing for production use, such as optimizing Jupyter kernel management, enabling authentication and authorization, and enforcing strong network security controls.

By leveraging code sandboxes teams can build advanced LLM solutions with high agency, allowing these applications to autonomously execute tasks while minimizing security risks. As the technology behind LLMs continues to advance, keeping pace with robust and flexible security measures will be essential for utilizing their full potential in innovative and impactful ways.

</details>

<details>
<summary>Overview</summary>

## Overview

The **tool** abstraction in LangChain associates a Python **function** with a **schema** that defines the function's **name**, **description** and **expected arguments**.

**Tools** can be passed to [chat models](https://python.langchain.com/docs/concepts/chat_models/) that support [tool calling](https://python.langchain.com/docs/concepts/tool_calling/) allowing the model to request the execution of a specific function with specific inputs.

## Key concepts

- Tools are a way to encapsulate a function and its schema in a way that can be passed to a chat model.
- Create tools using the [@tool](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.convert.tool.html) decorator, which simplifies the process of tool creation, supporting the following:
  - Automatically infer the tool's **name**, **description** and **expected arguments**, while also supporting customization.
  - Defining tools that return **artifacts** (e.g. images, dataframes, etc.)
  - Hiding input arguments from the schema (and hence from the model) using **injected tool arguments**.

## Tool interface

The tool interface is defined in the [BaseTool](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.base.BaseTool.html#langchain_core.tools.base.BaseTool) class which is a subclass of the [Runnable Interface](https://python.langchain.com/docs/concepts/runnables/).

The key attributes that correspond to the tool's **schema**:

- **name**: The name of the tool.
- **description**: A description of what the tool does.
- **args**: Property that returns the JSON schema for the tool's arguments.

The key methods to execute the function associated with the **tool**:

- **invoke**: Invokes the tool with the given arguments.
- **ainvoke**: Invokes the tool with the given arguments, asynchronously. Used for [async programming with Langchain](https://python.langchain.com/docs/concepts/async/).

## Create tools using the `@tool` decorator

The recommended way to create tools is using the [@tool](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.convert.tool.html) decorator. This decorator is designed to simplify the process of tool creation and should be used in most cases. After defining a function, you can decorate it with [@tool](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.convert.tool.html) to create a tool that implements the [Tool Interface](https://python.langchain.com/docs/concepts/tools/#tool-interface).

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
   """Multiply two numbers."""
   return a * b
```

**API Reference:** [tool](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.convert.tool.html)

For more details on how to create tools, see the [how to create custom tools](https://python.langchain.com/docs/how_to/custom_tools/) guide.

LangChain has a few other ways to create tools; e.g., by sub-classing the [BaseTool](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.base.BaseTool.html#langchain_core.tools.base.BaseTool) class or by using `StructuredTool`. These methods are shown in the [how to create custom tools guide](https://python.langchain.com/docs/how_to/custom_tools/), but
we generally recommend using the `@tool` decorator for most cases.

## Use the tool directly

Once you have defined a tool, you can use it directly by calling the function. For example, to use the `multiply` tool defined above:

```python
multiply.invoke({"a": 2, "b": 3})
```

### Inspect

You can also inspect the tool's schema and other properties:

```python
print(multiply.name) # multiply
print(multiply.description) # Multiply two numbers.
print(multiply.args)
# {
# 'type': 'object',
# 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
# 'required': ['a', 'b']
# }
```

If you're using pre-built LangChain or LangGraph components like [create\_react\_agent](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent), you might not need to interact with tools directly. However, understanding how to use them can be valuable for debugging and testing. Additionally, when building custom LangGraph workflows, you may find it necessary to work with tools directly.

## Configuring the schema

The `@tool` decorator offers additional options to configure the schema of the tool (e.g., modify name, description
or parse the function's doc-string to infer the schema).

Please see the [API reference for @tool](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.convert.tool.html) for more details and review the [how to create custom tools](https://python.langchain.com/docs/how_to/custom_tools/) guide for examples.

## Tool artifacts

**Tools** are utilities that can be called by a model, and whose outputs are designed to be fed back to a model. Sometimes, however, there are artifacts of a tool's execution that we want to make accessible to downstream components in our chain or agent, but that we don't want to expose to the model itself. For example if a tool returns a custom object, a dataframe or an image, we may want to pass some metadata about this output to the model without passing the actual output to the model. At the same time, we may want to be able to access this full output elsewhere, for example in downstream tools.

```python
@tool(response_format="content_and_artifact")
def some_tool(...) -> Tuple[str, Any]:
    """Tool that does something."""
    ...
    return 'Message for chat model', some_artifact
```

See [how to return artifacts from tools](https://python.langchain.com/docs/how_to/tool_artifacts/) for more details.

## Special type annotations

There are a number of special type annotations that can be used in the tool's function signature to configure the run time behavior of the tool.

The following type annotations will end up **removing** the argument from the tool's schema. This can be useful for arguments that should not be exposed to the model and that the model should not be able to control.

- **InjectedToolArg**: Value should be injected manually at runtime using `.invoke` or `.ainvoke`.
- **RunnableConfig**: Pass in the RunnableConfig object to the tool.
- **InjectedState**: Pass in the overall state of the LangGraph graph to the tool.
- **InjectedStore**: Pass in the LangGraph store object to the tool.

You can also use the `Annotated` type with a string literal to provide a **description** for the corresponding argument that **WILL** be exposed in the tool's schema.

- **Annotated\[..., "string literal"\]** \-\- Adds a description to the argument that will be exposed in the tool's schema.

### InjectedToolArg

There are cases where certain arguments need to be passed to a tool at runtime but should not be generated by the model itself. For this, we use the `InjectedToolArg` annotation, which allows certain parameters to be hidden from the tool's schema.

For example, if a tool requires a `user_id` to be injected dynamically at runtime, it can be structured in this way:

```python
from langchain_core.tools import tool, InjectedToolArg

@tool
def user_specific_tool(input_data: str, user_id: InjectedToolArg) -> str:
    """Tool that processes input data."""
    return f"User {user_id} processed {input_data}"
```

**API Reference:** [tool](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.convert.tool.html) \| [InjectedToolArg](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.base.InjectedToolArg.html)

Annotating the `user_id` argument with `InjectedToolArg` tells LangChain that this argument should not be exposed as part of the
tool's schema.

See [how to pass run time values to tools](https://python.langchain.com/docs/how_to/tool_runtime/) for more details on how to use `InjectedToolArg`.

### RunnableConfig

You can use the `RunnableConfig` object to pass custom run time values to tools.

If you need to access the [RunnableConfig](https://python.langchain.com/docs/concepts/runnables/#runnableconfig) object from within a tool. This can be done by using the `RunnableConfig` annotation in the tool's function signature.

```python
from langchain_core.runnables import RunnableConfig

@tool
async def some_func(..., config: RunnableConfig) -> ...:
    """Tool that does something."""
    # do something with config
    ...

await some_func.ainvoke(..., config={"configurable": {"value": "some_value"}})
```

**API Reference:** [RunnableConfig](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.config.RunnableConfig.html)

The `config` will not be part of the tool's schema and will be injected at runtime with appropriate values.

You may need to access the `config` object to manually propagate it to subclass. This happens if you're working with python 3.9 / 3.10 in an [async](https://python.langchain.com/docs/concepts/async/) environment and need to manually propagate the `config` object to sub-calls.

Please read [Propagation RunnableConfig](https://python.langchain.com/docs/concepts/runnables/#propagation-of-runnableconfig) for more details to learn how to propagate the `RunnableConfig` down the call chain manually (or upgrade to Python 3.11 where this is no longer an issue).

### InjectedState

Please see the [InjectedState](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.tool_node.InjectedState) documentation for more details.

### InjectedStore

Please see the [InjectedStore](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.tool_node.InjectedStore) documentation for more details.

## Best practices

When designing tools to be used by models, keep the following in mind:

- Tools that are well-named, correctly-documented and properly type-hinted are easier for models to use.
- Design simple and narrowly scoped tools, as they are easier for models to use correctly.
- Use chat models that support [tool-calling](https://python.langchain.com/docs/concepts/tool_calling/) APIs to take advantage of tools.

## Toolkits

LangChain has a concept of **toolkits**. This a very thin abstraction that groups tools together that
are designed to be used together for specific tasks.

### Interface

All Toolkits expose a `get_tools` method which returns a list of tools. You can therefore do:

```python
# Initialize a toolkit
toolkit = ExampleToolkit(...)

# Get list of tools
tools = toolkit.get_tools()
```

## Related resources

See the following resources for more information:

- [API Reference for @tool](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.convert.tool.html)
- [How to create custom tools](https://python.langchain.com/docs/how_to/custom_tools/)
- [How to pass run time values to tools](https://python.langchain.com/docs/how_to/tool_runtime/)
- [All LangChain tool how-to guides](https://docs.langchain.com/docs/how_to/#tools)
- [Additional how-to guides that show usage with LangGraph](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/)
- Tool integrations, see the [tool integration docs](https://docs.langchain.com/docs/integrations/tools/).

</details>

<details>
<summary>Introduction</summary>

Introduction
Large Language Models (LLMs) excel in generating text but often struggle to produce structured output. By leveraging Pydantic‘s type validation and prompt engineering, we can enforce and validate the output generated by LLMs.
All code examples in this blog post are written in Python. The LLM used is OpenAI’s gpt-3.5-turbo.

Query the LLM
To query the LLM, we use the following function:
```python
import openai

def query(prompt: str) -> str:
    """Query the LLM with the given prompt."""
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.0,
    )
    return completion.choices[0].message.content
```
We then call the function with a simple question:
```python
response = query("What is the largest planet in our solar system?")
print(response)
# 'The largest planet in our solar system is Jupiter.'
```

Enforcing JSON output with a prompt
In our prompt, we can ask the LLM to respond in a certain format:
```python
prompt = """
I will ask you questions and you will respond. Your response should be in the following format:
```json
{
    "thought": "How you think about the question",
    "answer": "The answer to the question"
}
```
"""
```
Then, we query the model:
```python
question = "What is the largest planet in our solar system?"
response = query(prompt + question)
print(response)
# '{
#     "thought": "This is a factual question that can be answered with scientific knowledge.",
#     "answer": "The largest planet in our solar system is Jupiter."
# }'
```

This is great, because we can easily parse the structured output:
```python
import json

parsed_response = json.loads(response)
print(parsed_response["answer"])
# 'The largest planet in our solar system is Jupiter.'
```

Validating the output
```python
from pydantic import BaseModel

class ThoughtAnswerResponse(BaseModel):
    thought: str
    answer: str

raw_response = query(prompt)

# Note: When you are using pydantic<2.0, use parse_raw instead of model_validate_json
validated_response = ThoughtAnswerResponse.model_validate_json(raw_response)

print(validated_response)
# thought='This is a factual question that can be answered with scientific knowledge.' answer='The largest planet in our solar system is Jupiter.'

print(type(validated_response))
# <class 'ThoughtAnswerResponse'>
```

Using the Pydantic model in the prompt
At this moment, we describe our response format in two places:
- a JSON description in our prompt
- a corresponding Pydantic model

When we want to update the response format, we need to change both the prompt and the Pydantic model. This can cause inconsistencies.

We can solve this by exporting the Pydantic model to a JSON schema and adding the schema to the prompt. This will make the response and the Pydantic model consistent.
```python
response_schema_dict = ThoughtAnswerResponse.model_json_schema()
response_schema_json = json.dumps(response_schema_dict, indent=2)

prompt = f"""
I will ask you questions, and you will respond.
Your response should be in the following format:
```json
{response_schema_json}
```
"""
```
The prompt will now look like this:
I will ask you questions, and you will respond. Your response should be in the following format:
```json
{
    "properties": {
        "thought": { "title": "Thought", "type": "string" },
        "answer": { "title": "Answer", "type": "string" }
    },
    "required": ["thought", "answer"],
    "title": "ThoughtAnswerResponse",
    "type": "object"
}
```
The response will look like this:
```json
{
  "thought": "The largest planet in our solar system is Jupiter.",
  "answer": "Jupiter"
}
```

Now, whenever you change the Pydantic model, the corresponding schema will be put in the prompt. Note that the schema has become more complex than it was before. One benefit is that it allows us to be more specific in what responses we require.

Error handling
The LLM may still produce results that are not consistent with our model. We can add some code to catch this:
```python
from pydantic import ValidationError

try:
    validated_response = ThoughtAnswerResponse.model_validate_json(raw_response)
except ValidationError as e:
    print("Unable to validate LLM response.")
    # Add your own error handling here
    raise e
```

Enforce specific values using a Literal
Sometimes, you want to enforce the use of specific values for a given field. We add the field “difficulty” to our response object. The LLM should use it to provide information about the difficulty of the question. In a regular prompt, we would do the following:
```python
prompt = """Your response should be in the following format:
```json
{
  "thought": "How you think about the question",
  "answer": "The answer to the question",
  "difficulty": "How difficult the question was. One of easy, medium or hard"
}
```
"""
```
Of course, the model could potentially still use other values. To validate it, we would need to write custom code.

With Pydantic, it is a lot easier. We create a new type called Difficulty using a Literal. A Literal allows us to specify the use of a select list of values. We add a Difficulty type hint to the difficulty field in our Pydantic model:
```python
from typing import Literal
from pydantic import BaseModel

# We create a new type
Difficulty = Literal["easy", "medium", "hard"]

class ThoughtAnswerResponse(BaseModel):
    thought: str
    answer: str
    difficulty: Difficulty
```
The LLM responds may respond with a value we do not allow:
```json
{
  "thought": "The largest planet in our solar system is Jupiter.",
  "answer": "Jupiter",
  "difficulty": "Unknown"
}
```
When we parse this result, Pydantic will validate the values for the difficulty field. Unknown does not match one of the values specified in the Literal type we have defined. So we get the following error:
```python
validated_response = ThoughtAnswerResponse.model_validate_json(response)

# ValidationError: 1 validation error for ThoughtAnswerResponse
# difficulty
#     Input should be 'easy', 'medium' or 'hard' [type=literal_error, input_value='Unknown', input_type=str]
```

Conclusion
By using Pydantic and prompt engineering, you can enforce and validate the output of LLMs. This provides you with greater control of the LLM output and allow you to build more robust AI systems.

</details>


## Code Sources

<details>
<summary>Repository analysis for https://github.com/towardsai/course-ai-agents/blob/dev/lessons/06_tools/notebook.ipynb</summary>

# Repository analysis for https://github.com/towardsai/course-ai-agents/blob/dev/lessons/06_tools/notebook.ipynb

## Summary
Repository: towardsai/course-ai-agents
Branch: dev
File: notebook.ipynb
Lines: 1,288

Estimated tokens: 9.7k

## File tree
```Directory structure:
└── notebook.ipynb

```

## Extracted content
================================================
FILE: lessons/06_tools/notebook.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Lesson 6: Tools

This notebook explores **Tools (Function Calling)**, one of the most critical building blocks of any AI Agent. 

We will use the `google-genai` library to interact with Google's Gemini models.

**Learning Objectives:**

1.  **Understand and implement tool use (function calling)** from scratch to allow an LLM to interact with external systems.
2.  **Build a custom tool calling framework** using decorators similar to production frameworks like LangGraph.
3.  **Use Gemini's native tool calling API** for production-ready implementations.
4.  **Implement structured data extraction** using Pydantic models as tools for reliable JSON output.
5.  **Run tools in loops** to handle multi-step tasks and understand the limitations that lead to ReAct patterns.
"""

"""
## 1. Setup

First, let's install the necessary Python libraries using pip.
"""

"""
!pip install -q google-genai pydantic python-dotenv
"""

"""
### Configure Gemini API Key

To use the Gemini API, you need an API key. 

1.  Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  Create a file named `.env` in the root of this project.
3.  Add the following line to the `.env` file, replacing `your_api_key_here` with your actual key:
    ```
    GOOGLE_API_KEY="your_api_key_here"
    ```
The code below will load this key from the `.env` file.
"""

%load_ext autoreload
%autoreload 2

from lessons.utils import env

env.load(required_env_vars=["GOOGLE_API_KEY"])
# Output:
#   Trying to load environment variables from `/Users/pauliusztin/Documents/01_projects/TAI/course-ai-agents/.env`

#   Environment variables loaded successfully.


"""
### Import Key Packages
"""

import json
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from lessons.utils import pretty_print

"""
### Initialize the Gemini Client
"""

client = genai.Client()

"""
### Define Constants

We will use the `gemini-2.5-flash` model, which is fast, cost-effective, and supports advanced features like tool use. We also define a sample financial document that will be used throughout our examples.
"""

MODEL_ID = "gemini-2.5-flash"

DOCUMENT = """
# Q3 2023 Financial Performance Analysis

The Q3 earnings report shows a 20% increase in revenue and a 15% growth in user engagement, 
beating market expectations. These impressive results reflect our successful product strategy 
and strong market positioning.

Our core business segments demonstrated remarkable resilience, with digital services leading 
the growth at 25% year-over-year. The expansion into new markets has proven particularly 
successful, contributing to 30% of the total revenue increase.

Customer acquisition costs decreased by 10% while retention rates improved to 92%, 
marking our best performance to date. These metrics, combined with our healthy cash flow 
position, provide a strong foundation for continued growth into Q4 and beyond.
"""

"""
## 2. Implementing tool calls from scratch

LLMs are trained on text and can't perform actions in the real world on their own. Tools (or Function Calling) are the mechanism we use to bridge this gap. We provide the LLM with a list of available tools, and it can decide which one to use and with what arguments to fulfill a user's request.

The process of calling a tool looks as follows:

1. **You:** Send the LLM a prompt and a list of available tools.
2. **LLM:** Responds with a function_call request, specifying the tool and arguments.
3. **You:** Execute the requested function in your code.
4. **You:** Send the function's output back to the LLM.
5. **LLM:** Uses the tool's output to generate a final, user-facing response.

"""

"""
### Define Mock Tools

Let's create three simple, mocked functions. One simulates searching Google Drive, another simulates sending a Discord message, and the last one simulates summarizing a document. 

The function signature (input parameters and output type) and docstrings are crucial, as the LLM uses them to understand what each tool does.
"""

def search_google_drive(query: str) -> dict:
    """
    Searches for a file on Google Drive and returns its content or a summary.

    Args:
        query (str): The search query to find the file, e.g., 'Q3 earnings report'.

    Returns:
        dict: A dictionary representing the search results, including file names and summaries.
    """

    # In a real scenario, this would interact with the Google Drive API.
    # Here, we mock the response for demonstration.
    return {
        "files": [
            {
                "name": "Q3_Earnings_Report_2024.pdf",
                "id": "file12345",
                "content": DOCUMENT,
            }
        ]
    }


def send_discord_message(channel_id: str, message: str) -> dict:
    """
    Sends a message to a specific Discord channel.

    Args:
        channel_id (str): The ID of the channel to send the message to, e.g., '#finance'.
        message (str): The content of the message to send.

    Returns:
        dict: A dictionary confirming the action, e.g., {"status": "success"}.
    """

    # Mocking a successful API call to Discord.
    return {
        "status": "success",
        "status_code": 200,
        "channel": channel_id,
        "message_preview": f"{message[:50]}...",
    }


def summarize_financial_report(text: str) -> str:
    """
    Summarizes a financial report.

    Args:
        text (str): The text to summarize.

    Returns:
        str: The summary of the text.
    """

    return "The Q3 2023 earnings report shows strong performance across all metrics \
with 20% revenue growth, 15% user engagement increase, 25% digital services growth, and \
improved retention rates of 92%."

"""
Now we need to define the metadata for each function, which will be used as input to the LLM to understand what tool to use and how to call it:
"""

search_google_drive_schema = {
    "name": "search_google_drive",
    "description": "Searches for a file on Google Drive and returns its content or a summary.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find the file, e.g., 'Q3 earnings report'.",
            }
        },
        "required": ["query"],
    },
}

send_discord_message_schema = {
    "name": "send_discord_message",
    "description": "Sends a message to a specific Discord channel.",
    "parameters": {
        "type": "object",
        "properties": {
            "channel_id": {
                "type": "string",
                "description": "The ID of the channel to send the message to, e.g., '#finance'.",
            },
            "message": {
                "type": "string",
                "description": "The content of the message to send.",
            },
        },
        "required": ["channel_id", "message"],
    },
}

summarize_financial_report_schema = {
    "name": "summarize_financial_report",
    "description": "Summarizes a financial report.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to summarize.",
            },
        },
        "required": ["text"],
    },
}


"""
Ultimately, we will aggregate all the tools in a single dictionary:
"""

TOOLS = {
    "search_google_drive": {
        "handler": search_google_drive,
        "declaration": search_google_drive_schema,
    },
    "send_discord_message": {
        "handler": send_discord_message,
        "declaration": send_discord_message_schema,
    },
    "summarize_financial_report": {
        "handler": summarize_financial_report,
        "declaration": summarize_financial_report_schema,
    },
}
TOOLS_BY_NAME = {tool_name: tool["handler"] for tool_name, tool in TOOLS.items()}
TOOLS_SCHEMA = [tool["declaration"] for tool in TOOLS.values()]

"""
Let's take a look at them:
"""

for tool_name, tool in TOOLS_BY_NAME.items():
    print(f"Tool name: {tool_name}")
    print(f"Tool handler: {tool}")
    print("-" * 75)
# Output:
#   Tool name: search_google_drive

#   Tool handler: <function search_google_drive at 0x104c7df80>

#   ---------------------------------------------------------------------------

#   Tool name: send_discord_message

#   Tool handler: <function send_discord_message at 0x104c7de40>

#   ---------------------------------------------------------------------------

#   Tool name: summarize_financial_report

#   Tool handler: <function summarize_financial_report at 0x1274f5c60>

#   ---------------------------------------------------------------------------


pretty_print.wrapped(json.dumps(TOOLS_SCHEMA[0], indent=2), title="`search_google_drive` Tool Schema")
# Output:
#   [93m-------------------------------- `search_google_drive` Tool Schema --------------------------------[0m

#     {

#     "name": "search_google_drive",

#     "description": "Searches for a file on Google Drive and returns its content or a summary.",

#     "parameters": {

#       "type": "object",

#       "properties": {

#         "query": {

#           "type": "string",

#           "description": "The search query to find the file, e.g., 'Q3 earnings report'."

#         }

#       },

#       "required": [

#         "query"

#       ]

#     }

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m


pretty_print.wrapped(json.dumps(TOOLS_SCHEMA[1], indent=2), title="`send_discord_message` Tool Schema")
# Output:
#   [93m-------------------------------- `send_discord_message` Tool Schema --------------------------------[0m

#     {

#     "name": "send_discord_message",

#     "description": "Sends a message to a specific Discord channel.",

#     "parameters": {

#       "type": "object",

#       "properties": {

#         "channel_id": {

#           "type": "string",

#           "description": "The ID of the channel to send the message to, e.g., '#finance'."

#         },

#         "message": {

#           "type": "string",

#           "description": "The content of the message to send."

#         }

#       },

#       "required": [

#         "channel_id",

#         "message"

#       ]

#     }

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
Now, let's see how to call these tools using an LLM. First, we need to define the system prompt:
"""

TOOL_CALLING_SYSTEM_PROMPT = """
You are a helpful AI assistant with access to tools that enable you to take actions and retrieve information to better 
assist users.

## Tool Usage Guidelines

**When to use tools:**
- When you need information that is not in your training data
- When you need to perform actions in external systems and environments
- When you need real-time, dynamic, or user-specific data
- When computational operations are required

**Tool selection:**
- Choose the most appropriate tool based on the user's specific request
- If multiple tools could work, select the one that most directly addresses the need
- Consider the order of operations for multi-step tasks

**Parameter requirements:**
- Provide all required parameters with accurate values
- Use the parameter descriptions to understand expected formats and constraints
- Ensure data types match the tool's requirements (strings, numbers, booleans, arrays)

## Tool Call Format

When you need to use a tool, output ONLY the tool call in this exact format:

```tool_call
{{"name": "tool_name", "args": {{"param1": "value1", "param2": "value2"}}}}
```

**Critical formatting rules:**
- Use double quotes for all JSON strings
- Ensure the JSON is valid and properly escaped
- Include ALL required parameters
- Use correct data types as specified in the tool definition
- Do not include any additional text or explanation in the tool call

## Response Behavior

- If no tools are needed, respond directly to the user with helpful information
- If tools are needed, make the tool call first, then provide context about what you're doing
- After receiving tool results, provide a clear, user-friendly explanation of the outcome
- If a tool call fails, explain the issue and suggest alternatives when possible

## Available Tools

<tool_definitions>
{tools}
</tool_definitions>

Remember: Your goal is to be maximally helpful to the user. Use tools when they add value, but don't use them unnecessarily. Always prioritize accuracy and user experience.
"""


"""
Let's try the prompt with a few examples.
"""

USER_PROMPT = """
Can you help me find the latest quarterly report and share key insights with the team?
"""

messages = [TOOL_CALLING_SYSTEM_PROMPT.format(tools=str(TOOLS_SCHEMA)), USER_PROMPT]

response = client.models.generate_content(
    model=MODEL_ID,
    contents=messages,
)

pretty_print.wrapped(response.text, title="LLM Tool Call Response")
# Output:
#   [93m-------------------------------------- LLM Tool Call Response --------------------------------------[0m

#     ```tool_call

#   {"name": "search_google_drive", "args": {"query": "latest quarterly report"}}

#   ```

#   [93m----------------------------------------------------------------------------------------------------[0m


USER_PROMPT = """
Please find the Q3 earnings report on Google Drive and send a summary of it to 
the #finance channel on Discord.
"""

messages = [TOOL_CALLING_SYSTEM_PROMPT.format(tools=str(TOOLS_SCHEMA)), USER_PROMPT]

response = client.models.generate_content(
    model=MODEL_ID,
    contents=messages,
)
pretty_print.wrapped(response.text, title="LLM Tool Call Response")
# Output:
#   [93m-------------------------------------- LLM Tool Call Response --------------------------------------[0m

#     ```tool_call

#   {"name": "search_google_drive", "args": {"query": "Q3 earnings report"}}

#   ```

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
The next step is to parse the LLM response and call the tool using Python.

First, we parse the LLM output to extract the JSON from the response:
"""

def extract_tool_call(response_text: str) -> str:
    """
    Extracts the tool call from the response text.
    """
    return response_text.split("```tool_call")[1].split("```")[0].strip()


tool_call_str = extract_tool_call(response.text)
tool_call_str
# Output:
#   '{"name": "search_google_drive", "args": {"query": "Q3 earnings report"}}'

"""
Next, we parse the stringified JSON to a Python dict:
"""

tool_call = json.loads(tool_call_str)
tool_call
# Output:
#   {'name': 'search_google_drive', 'args': {'query': 'Q3 earnings report'}}

"""
Now, we retrieve the tool handler, which is a Python function:
"""

tool_handler = TOOLS_BY_NAME[tool_call["name"]]
tool_handler
# Output:
#   <function __main__.search_google_drive(query: str) -> dict>

"""
Ultimately, we call the Python function using the arguments generated by the LLM:
"""

tool_result = tool_handler(**tool_call["args"])
pretty_print.wrapped(tool_result, indent=2, title="LLM Tool Call Response")
# Output:
#   [93m-------------------------------------- LLM Tool Call Response --------------------------------------[0m

#     {

#     "files": [

#       {

#         "name": "Q3_Earnings_Report_2024.pdf",

#         "id": "file12345",

#         "content": "\n# Q3 2023 Financial Performance Analysis\n\nThe Q3 earnings report shows a 20% increase in revenue and a 15% growth in user engagement, \nbeating market expectations. These impressive results reflect our successful product strategy \nand strong market positioning.\n\nOur core business segments demonstrated remarkable resilience, with digital services leading \nthe growth at 25% year-over-year. The expansion into new markets has proven particularly \nsuccessful, contributing to 30% of the total revenue increase.\n\nCustomer acquisition costs decreased by 10% while retention rates improved to 92%, \nmarking our best performance to date. These metrics, combined with our healthy cash flow \nposition, provide a strong foundation for continued growth into Q4 and beyond.\n"

#       }

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
We can summarize the tool execution in the following function:
"""

def call_tool(response_text: str, tools_by_name: dict) -> Any:
    """
    Call a tool based on the response from the LLM.
    """

    tool_call_str = extract_tool_call(response_text)
    tool_call = json.loads(tool_call_str)
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool = tools_by_name[tool_name]

    return tool(**tool_args)

pretty_print.wrapped(
    json.dumps(call_tool(response.text, tools_by_name=TOOLS_BY_NAME), indent=2), title="LLM Tool Call Response"
)
# Output:
#   [93m-------------------------------------- LLM Tool Call Response --------------------------------------[0m

#     {

#     "files": [

#       {

#         "name": "Q3_Earnings_Report_2024.pdf",

#         "id": "file12345",

#         "content": "\n# Q3 2023 Financial Performance Analysis\n\nThe Q3 earnings report shows a 20% increase in revenue and a 15% growth in user engagement, \nbeating market expectations. These impressive results reflect our successful product strategy \nand strong market positioning.\n\nOur core business segments demonstrated remarkable resilience, with digital services leading \nthe growth at 25% year-over-year. The expansion into new markets has proven particularly \nsuccessful, contributing to 30% of the total revenue increase.\n\nCustomer acquisition costs decreased by 10% while retention rates improved to 92%, \nmarking our best performance to date. These metrics, combined with our healthy cash flow \nposition, provide a strong foundation for continued growth into Q4 and beyond.\n"

#       }

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
Usually we want the LLM to interpret the tool output:
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=f"Interpret the tool result: {json.dumps(tool_result, indent=2)}",
)
pretty_print.wrapped(response.text, title="LLM Tool Call Response")
# Output:
#   [93m-------------------------------------- LLM Tool Call Response --------------------------------------[0m

#     The tool result provides the content of a file named `Q3_Earnings_Report_2024.pdf`.

#   

#   This document is a **Q3 2023 Financial Performance Analysis** and details exceptionally strong results, significantly beating market expectations.

#   

#   **Key highlights from the report include:**

#   

#   *   **Revenue Growth:** A 20% increase in revenue.

#   *   **User Engagement:** 15% growth in user engagement.

#   *   **Core Business Performance:** Digital services led growth at 25% year-over-year.

#   *   **Market Expansion Success:** New markets contributed 30% of the total revenue increase.

#   *   **Efficiency & Retention:**

#       *   Customer acquisition costs decreased by 10%.

#       *   Retention rates improved to 92%, marking the best performance to date.

#   *   **Financial Health:** The company maintains a healthy cash flow position.

#   

#   The report attributes these impressive results to a successful product strategy and strong market positioning, indicating a robust foundation for continued growth into Q4 and beyond.

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
That's the basic concept of tool calling! We've successfully implemented function calling from scratch.
"""

"""
## 3. Implementing tool calls from scratch using @tool decorators
"""

"""
For a better analogy with what we see in frameworks such as LangGraph or MCP, let's define a `@tool` decorator that automatically computes the schemas defined above based on the function signature and docstring:
"""

from inspect import Parameter, signature
from typing import Any, Callable, Dict, Optional


class ToolFunction:
    def __init__(self, func: Callable, schema: Dict[str, Any]) -> None:
        self.func = func
        self.schema = schema
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


def tool(description: Optional[str] = None) -> Callable[[Callable], ToolFunction]:
    """
    A decorator that creates a tool schema from a function.

    Args:
        description: Optional override for the function's docstring

    Returns:
        A decorator function that wraps the original function and adds a schema
    """

    def decorator(func: Callable) -> ToolFunction:
        # Get function signature
        sig = signature(func)

        # Create parameters schema
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            # Skip self for methods
            if param_name == "self":
                continue

            param_schema = {
                "type": "string",  # Default to string, can be enhanced with type hints
                "description": f"The {param_name} parameter",  # Default description
            }

            # Add to required if parameter has no default value
            if param.default == Parameter.empty:
                required.append(param_name)

            properties[param_name] = param_schema

        # Create the tool schema
        schema = {
            "name": func.__name__,
            "description": description or func.__doc__ or f"Executes the {func.__name__} function.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

        return ToolFunction(func, schema)

    return decorator


@tool()
def search_google_drive_example(query: str) -> dict:
    """Search for files in Google Drive."""
    return {"files": ["Q3 earnings report"]}


@tool()
def send_discord_message_example(channel_id: str, message: str) -> dict:
    """Send a message to a Discord channel."""
    return {"message": "Message sent successfully"}


@tool()
def summarize_financial_report_example(text: str) -> str:
    """Summarize the contents of a financial report."""
    return "Financial report summarized successfully"


tools = [
    search_google_drive_example,
    send_discord_message_example,
    summarize_financial_report_example,
]
tools_by_name = {tool.schema["name"]: tool.func for tool in tools}
tools_schema = [tool.schema for tool in tools]

"""
After the function has been decorated, it has been wrapped into a `ToolFunction` object:
"""

type(search_google_drive_example)
# Output:
#   __main__.ToolFunction

"""
Which has the following fields:
"""

pretty_print.wrapped(json.dumps(search_google_drive_example.schema, indent=2), title="Search Google Drive Example")
# Output:
#   [93m----------------------------------- Search Google Drive Example -----------------------------------[0m

#     {

#     "name": "search_google_drive_example",

#     "description": "Search for files in Google Drive.",

#     "parameters": {

#       "type": "object",

#       "properties": {

#         "query": {

#           "type": "string",

#           "description": "The query parameter"

#         }

#       },

#       "required": [

#         "query"

#       ]

#     }

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
...and the actual function handler:
"""

search_google_drive_example.func
# Output:
#   <function __main__.search_google_drive_example(query: str) -> dict>

"""
Let's see how this new method works with LLMs:
"""

USER_PROMPT = """
Please find the Q3 earnings report on Google Drive and send a summary of it to 
the #finance channel on Discord.
"""

messages = [TOOL_CALLING_SYSTEM_PROMPT.format(tools=str(tools_schema)), USER_PROMPT]

response = client.models.generate_content(
    model=MODEL_ID,
    contents=messages,
)
pretty_print.wrapped(response.text, title="LLM Tool Call Response")
# Output:
#   [93m-------------------------------------- LLM Tool Call Response --------------------------------------[0m

#     ```tool_call

#   {"name": "search_google_drive_example", "args": {"query": "Q3 earnings report"}}

#   ```

#   [93m----------------------------------------------------------------------------------------------------[0m


pretty_print.wrapped(
    json.dumps(call_tool(response.text, tools_by_name=tools_by_name), indent=2), title="LLM Tool Call Response"
)
# Output:
#   [93m-------------------------------------- LLM Tool Call Response --------------------------------------[0m

#     {

#     "files": [

#       "Q3 earnings report"

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
Voilà! We have our little tool calling framework.
"""

"""
## 4. Implementing tool calls with Gemini's Native API

In production, most of the time, we don't implement tool calling from scratch, but instead leverage the native interface of a specific API such as Gemini or OpenAI. So, let's see how we can use Gemini's built-in tool calling capabilities instead of our custom implementation.
"""

tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(**search_google_drive_schema),
            types.FunctionDeclaration(**send_discord_message_schema),
        ]
    )
]
config = types.GenerateContentConfig(
    tools=tools,
    # Force the model to call 'any' function, instead of chatting.
    tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY")),
)


pretty_print.wrapped(USER_PROMPT, title="User Prompt")
response = client.models.generate_content(
    model=MODEL_ID,
    contents=USER_PROMPT,
    config=config,
)
# Output:
#   [93m------------------------------------------- User Prompt -------------------------------------------[0m

#     

#   Please find the Q3 earnings report on Google Drive and send a summary of it to 

#   the #finance channel on Discord.

#   

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
As you can see, here we don't explictly define a system prompt that guides the LLM how to use the tools. Instead we pass the tools schema to the LLM provider which will handle them internally. This is more efficient, as they take care of optimizing tool/function calling for every specific model.
"""

response_message_part = response.candidates[0].content.parts[0]
function_call = response_message_part.function_call
function_call
# Output:
#   FunctionCall(id=None, args={'query': 'Q3 earnings report'}, name='search_google_drive')

tool_handler = TOOLS_BY_NAME[function_call.name]
tool_handler
# Output:
#   <function __main__.search_google_drive(query: str) -> dict>

tool_handler(**function_call.args)
# Output:
#   {'files': [{'name': 'Q3_Earnings_Report_2024.pdf',

#      'id': 'file12345',

#      'content': '\n# Q3 2023 Financial Performance Analysis\n\nThe Q3 earnings report shows a 20% increase in revenue and a 15% growth in user engagement, \nbeating market expectations. These impressive results reflect our successful product strategy \nand strong market positioning.\n\nOur core business segments demonstrated remarkable resilience, with digital services leading \nthe growth at 25% year-over-year. The expansion into new markets has proven particularly \nsuccessful, contributing to 30% of the total revenue increase.\n\nCustomer acquisition costs decreased by 10% while retention rates improved to 92%, \nmarking our best performance to date. These metrics, combined with our healthy cash flow \nposition, provide a strong foundation for continued growth into Q4 and beyond.\n'}]}

"""
Now let's create a simplified function that works with Gemini's native function call objects:
"""

def call_tool(function_call) -> Any:
    tool_name = function_call.name
    tool_args = function_call.args

    tool_handler = TOOLS_BY_NAME[tool_name]

    return tool_handler(**tool_args)

tool_result = call_tool(response_message_part.function_call)
pretty_print.wrapped(tool_result, indent=2, title="Tool Result")
# Output:
#   [93m------------------------------------------- Tool Result -------------------------------------------[0m

#     {

#     "files": [

#       {

#         "name": "Q3_Earnings_Report_2024.pdf",

#         "id": "file12345",

#         "content": "\n# Q3 2023 Financial Performance Analysis\n\nThe Q3 earnings report shows a 20% increase in revenue and a 15% growth in user engagement, \nbeating market expectations. These impressive results reflect our successful product strategy \nand strong market positioning.\n\nOur core business segments demonstrated remarkable resilience, with digital services leading \nthe growth at 25% year-over-year. The expansion into new markets has proven particularly \nsuccessful, contributing to 30% of the total revenue increase.\n\nCustomer acquisition costs decreased by 10% while retention rates improved to 92%, \nmarking our best performance to date. These metrics, combined with our healthy cash flow \nposition, provide a strong foundation for continued growth into Q4 and beyond.\n"

#       }

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
## 5. Using a Pydantic Model as a Tool for Structured Outputs

A more elegant and powerful pattern is to treat our Pydantic model *as a tool*. We can ask the model to "call" this Pydantic tool, and the arguments it generates will be our structured data.

This combines the power of function calling with the robustness of Pydantic for structured data extraction. It's the recommended approach for complex data extraction tasks.

Let's define the same Pydantic model as in the structured outputs lesson:
"""

class DocumentMetadata(BaseModel):
    """A class to hold structured metadata for a document."""

    summary: str = Field(description="A concise, 1-2 sentence summary of the document.")
    tags: list[str] = Field(description="A list of 3-5 high-level tags relevant to the document.")
    keywords: list[str] = Field(description="A list of specific keywords or concepts mentioned.")
    quarter: str = Field(description="The quarter of the financial year described in the document (e.g., Q3 2023).")
    growth_rate: str = Field(description="The growth rate of the company described in the document (e.g., 10%).")

"""
Now, let's see how to use it as a tool:
"""

# The Pydantic class 'DocumentMetadata' is now our 'tool'
extraction_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="extract_metadata",
            description="Extracts structured metadata from a financial document.",
            parameters=DocumentMetadata.model_json_schema(),
        )
    ]
)
config = types.GenerateContentConfig(
    tools=[extraction_tool],
    tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY")),
)

prompt = f"""
Please analyze the following document and extract its metadata.

Document:
--- 
{DOCUMENT}
--- 
"""

response = client.models.generate_content(model=MODEL_ID, contents=prompt, config=config)
response_message_part = response.candidates[0].content.parts[0]

if hasattr(response_message_part, "function_call"):
    function_call = response_message_part.function_call
    pretty_print.function_call(function_call, title="Function Call")

    try:
        document_metadata = DocumentMetadata(**function_call.args)
        pretty_print.wrapped(document_metadata.model_dump_json(indent=2), title="Pydantic Validated Object")
    except Exception as e:
        pretty_print.wrapped(f"Validation failed: {e}", title="Validation Error")
else:
    pretty_print.wrapped("The model did not call the extraction tool.", title="No Function Call")
# Output:
#   [93m------------------------------------------ Function Call ------------------------------------------[0m

#     [38;5;208mFunction Name:[0m `extract_metadata

#     [38;5;208mFunction Arguments:[0m `{

#     "growth_rate": "20%",

#     "summary": "The Q3 2023 earnings report shows a 20% increase in revenue and 15% growth in user engagement, driven by successful product strategy and market expansion. This performance provides a strong foundation for continued growth.",

#     "quarter": "Q3 2023",

#     "keywords": [

#       "Revenue",

#       "User Engagement",

#       "Market Expansion",

#       "Customer Acquisition",

#       "Retention Rates",

#       "Digital Services",

#       "Cash Flow"

#     ],

#     "tags": [

#       "Financials",

#       "Earnings",

#       "Growth",

#       "Business Strategy",

#       "Market Analysis"

#     ]

#   }`

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m------------------------------------ Pydantic Validated Object ------------------------------------[0m

#     {

#     "summary": "The Q3 2023 earnings report shows a 20% increase in revenue and 15% growth in user engagement, driven by successful product strategy and market expansion. This performance provides a strong foundation for continued growth.",

#     "tags": [

#       "Financials",

#       "Earnings",

#       "Growth",

#       "Business Strategy",

#       "Market Analysis"

#     ],

#     "keywords": [

#       "Revenue",

#       "User Engagement",

#       "Market Expansion",

#       "Customer Acquisition",

#       "Retention Rates",

#       "Digital Services",

#       "Cash Flow"

#     ],

#     "quarter": "Q3 2023",

#     "growth_rate": "20%"

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
## 6. Running Tools in a Loop

Now, let's implement a more sophisticated approach where we put tool calling in a loop with a conversation history. This allows the agent to perform multi-step tasks by calling multiple tools in sequence. Let's create a scenario where we ask the agent to find a report on Google Drive and then communicate its findings on Discord.
"""

tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(**search_google_drive_schema),
            types.FunctionDeclaration(**send_discord_message_schema),
            types.FunctionDeclaration(**summarize_financial_report_schema),
        ]
    )
]
config = types.GenerateContentConfig(
    tools=tools,
    tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY")),
)


USER_PROMPT = """
Please find the Q3 earnings report on Google Drive and send a summary of it to 
the #finance channel on Discord.
"""

messages = [USER_PROMPT]

pretty_print.wrapped(USER_PROMPT, title="User Prompt")
response = client.models.generate_content(
    model=MODEL_ID,
    contents=messages,
    config=config,
)
response_message_part = response.candidates[0].content.parts[0]
pretty_print.function_call(response_message_part.function_call, title="Function Call")

messages.append(response.candidates[0].content)

# Loop until the model stops requesting function calls or we reach the max number of iterations
max_iterations = 3
while hasattr(response_message_part, "function_call") and max_iterations > 0:
    tool_result = call_tool(response_message_part.function_call)
    pretty_print.wrapped(tool_result, title="Tool Result", indent=2)

    # Add the tool result to the messages creating the following structure:
    # - user prompt
    # - tool call
    # - tool result
    # - tool call
    # - tool result
    # ...
    function_response_part = types.Part.from_function_response(
        name=response_message_part.function_call.name,
        response={"result": tool_result},
    )
    messages.append(function_response_part)

    # Ask the LLM to continue with the next step (which may involve calling another tool)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=messages,
        config=config,
    )

    response_message_part = response.candidates[0].content.parts[0]
    pretty_print.function_call(response_message_part.function_call, only_name=True, title="Function Call")

    messages.append(response.candidates[0].content)

    max_iterations -= 1

pretty_print.wrapped(response.candidates[0].content, title="Final Agent Response")

# Output:
#   [93m------------------------------------------- User Prompt -------------------------------------------[0m

#     

#   Please find the Q3 earnings report on Google Drive and send a summary of it to 

#   the #finance channel on Discord.

#   

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m------------------------------------------ Function Call ------------------------------------------[0m

#     [38;5;208mFunction Name:[0m `search_google_drive

#     [38;5;208mFunction Arguments:[0m `{

#     "query": "Q3 earnings report"

#   }`

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m------------------------------------------- Tool Result -------------------------------------------[0m

#     {

#     "files": [

#       {

#         "name": "Q3_Earnings_Report_2024.pdf",

#         "id": "file12345",

#         "content": "\n# Q3 2023 Financial Performance Analysis\n\nThe Q3 earnings report shows a 20% increase in revenue and a 15% growth in user engagement, \nbeating market expectations. These impressive results reflect our successful product strategy \nand strong market positioning.\n\nOur core business segments demonstrated remarkable resilience, with digital services leading \nthe growth at 25% year-over-year. The expansion into new markets has proven particularly \nsuccessful, contributing to 30% of the total revenue increase.\n\nCustomer acquisition costs decreased by 10% while retention rates improved to 92%, \nmarking our best performance to date. These metrics, combined with our healthy cash flow \nposition, provide a strong foundation for continued growth into Q4 and beyond.\n"

#       }

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m------------------------------------------ Function Call ------------------------------------------[0m

#     [38;5;208mFunction Name:[0m `summarize_financial_report

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m------------------------------------------- Tool Result -------------------------------------------[0m

#     The Q3 2023 earnings report shows strong performance across all metrics with 20% revenue growth, 15% user engagement increase, 25% digital services growth, and improved retention rates of 92%.

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m------------------------------------------ Function Call ------------------------------------------[0m

#     [38;5;208mFunction Name:[0m `send_discord_message

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m------------------------------------------- Tool Result -------------------------------------------[0m

#     {

#     "status": "success",

#     "status_code": 200,

#     "channel": "#finance",

#     "message_preview": "The Q3 2023 earnings report shows strong performan..."

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m------------------------------------------ Function Call ------------------------------------------[0m

#     [38;5;208mFunction Name:[0m `send_discord_message

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m--------------------------------------- Final Agent Response ---------------------------------------[0m

#     ('parts', [Part(video_metadata=None, thought=None, inline_data=None, file_data=None, thought_signature=b'\n\xec\x02\x01T\xa8\\\xee?[\xd4\x1f\xc1\x14\x08\xc9\x87\xd6ij-{\xea\xd3\xa9E\xa3\x9eiG\x16\xb41\xad\x90\x92\x01\x17C=\xbc^\x90\x84T\xb3Z\x86\x1d%T\xb4\x10\xe1\x02\xf9\xa3\xcfJ\xc4+\xa1\x0b\xe4\r\xee\xc3e\xc5j\x82W\x8bP\xe55B\xbf\xe5@%\x1c_\xda1hE\x00\xeec\xb2\xc2\x9fGI\xaf\xbe\x06\xf8M\x1fm\xe1\xfd7!]\xe12\x93\x94\xdd\x19B\xba\\\xd1\x0caI\xfbR5\xd4\xa9\xa9\x06x\x86\xd0\x06\x94gq\xf9\xda\x80D\xba\x95\xd0[u\xa9V\x8fb\xf7%\xb0\xc3J\x8d\x1e\x9e\xca\xa6fP\x12\xd2\xe5G\xc7\x08\xd5R\xcdn\xf2YeFQ\x80\xcec\xd7h\x1e\xcb\x1c\xbbW\xfe\xd7\xe8\xe2\xcc\xdc\x06\x8e^\xa5m\xd5\x10Y[\x8b\xa2\x89+\x12\xb54k\x073\xfc\x0f\x9c!\x8f\x83t\xfe\xcb\xb01v\x8f\xa0\xb23c\xa7\x0b\xb7y\xd1?\xb4\xc5\xa0\xef\x01\xdc\xa0\xb7\xd1\r\x87\x9445\xeb\x08\x86\xd66m\xe4\xab)6vN\x99!\x87\x01Q-\x9cL*\x0b\x97\x1a\x0f\xb0v\x16\xb3\xfc2\xe1\x88c\xadj<\xbb^\x1b\'\xbb}\xa8l\x0c%\x83??,|\xc2mB\xb7\x95\xe2GF\xee\xf6\xf2\x95\x03\xbb\xf9\xba\xfe\x0c1J\xf2\x93\x83O\x95."Pl\x87\xa6[\x8c,b\x17,c\xa3\xd0\x19\x893P\xd9\xe8C\x93.o&8\x0f\x0c\x0c\x90e\xdb\xae\x97\xed\x12\x00\xd5\xbcV\xf0\xcf\xea', code_execution_result=None, executable_code=None, function_call=FunctionCall(id=None, args={'channel_id': '#finance', 'message': 'The Q3 2023 earnings report shows strong performance across all metrics with 20% revenue growth, 15% user engagement increase, 25% digital services growth, and improved retention rates of 92%.'}, name='send_discord_message'), function_response=None, text=None)])

#   [93m----------------------------------------------------------------------------------------------------[0m

#     ('role', 'model')

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
Running tools in a loop is powerful for multi-step tasks, but this approach has limitations. It assumes the agent should call a tool at each iteration and doesn't provide explicit opportunities for the model to reason about tool outputs before deciding on the next action. The agent immediately moves to the next function call without pausing to think about what it learned or whether it should change strategy.

This limitation leads us to more sophisticated patterns like **ReAct** (Reasoning and Acting), which explicitly interleaves reasoning steps with tool calls, allowing the agent to think through problems more deliberately. We will explore ReAct patterns in the next lesson.
"""

</details>


## YouTube Video Transcripts

<details>
<summary>Hello everybody. Welcome to The Neural Maze. (The video shows a GitHub page for a repository named 'agentic_patterns'. A large diagram titled "Agentic Patterns" is visible, showing four patterns: Reflection Pattern, Tool Use Pattern, Planning Pattern, and MultiAgent Pattern.) So in today's video we are going to keep working on the project of implementing the four agentic patterns from scratch that we started a week ago when we implemented the reflection pattern. So today we are going to move into the second pattern that is the tool pattern. And before we begin, I'm pretty sure that you're already familiar with this pattern in a practical sense.</summary>

Hello everybody. Welcome to The Neural Maze. (The video shows a GitHub page for a repository named 'agentic_patterns'. A large diagram titled "Agentic Patterns" is visible, showing four patterns: Reflection Pattern, Tool Use Pattern, Planning Pattern, and MultiAgent Pattern.) So in today's video we are going to keep working on the project of implementing the four agentic patterns from scratch that we started a week ago when we implemented the reflection pattern. So today we are going to move into the second pattern that is the tool pattern. And before we begin, I'm pretty sure that you're already familiar with this pattern in a practical sense.

[00:30:00]
What I mean by this is that you have probably used in the past tools in LangChain, in LlamaIndex or in Crew AI. (The speaker navigates through browser tabs showing documentation pages for "Tools" in LangChain, LlamaIndex, and crewAI.) And the thing is that in today's video, I'm not going to teach you how to use these tools in specific frameworks. I'm just going to teach you how these tools work under the hood. And I think that's really insightful because if we really understand how things work under the hood, I think it's much easier for us to learn how to apply them in the proper way.

*The speaker plans to explain the underlying mechanics of the "Tool Use" agentic pattern, rather than just how to use it within existing frameworks like LangChain or crewAI.*

[01:00:00]
So, as we did in the previous video, we are going to start with a Jupyter Notebook that covers all the theory step by step. (The speaker switches to a Jupyter Notebook titled 'TOOL_PATTERN.ipynb'. It shows a detailed diagram of the "Tool Use Pattern" with numbered steps.) And then I will move into VS Code where I will show you all the abstractions and all the classes that I have implemented to make this tool more robust, to try to mimic the structure that all of these frameworks offer at this moment. You know, having like a tool class and a tool agent class, very similar to what we did with the reflection pattern, but with the tool pattern. Okay, so let's begin with the theory of the tool pattern.

[01:30:00]
You have this diagram right here that tries to offer a simplified description of what the pattern does or tries to implement under the hood. (The speaker scrolls down in the notebook, focusing on the "Tool Use Pattern" diagram which shows a user prompt leading to the use of "Tool A," "Tool B," or "Tool C," which in turn connect to external services like Wikipedia, Google, and YouTube.) But basically, let's start by defining what is a tool. And a tool, let's put it in simple terms, it's just a way for the LLM to access the outside world. And what do I mean by this? Uh, remember that LLMs store all the information in their weights.

[02:00:00]
So, when you ask an LLM about specific information, that information is going to be retrieved by the weights. But sometimes the information stored in these weights is not enough, and we need a way for the LLM to access the outside world. And that's exactly what a tool does. A tool is just uh like a Python function that the LLM can access and run and fetch some relevant results using an API or uh parsing a web content or um consulting uh Wolfram Alpha to to calculate some difficult integrals.

[02:30:00]
But you get the point. It's a way for the LLM to get outside the information stored in its weights. Okay. So let's start by defining a simple Python function. You have it in here. (The screen shows a Jupyter cell with Python code defining a function called 'get_current_weather'.) So, uh this Python function, which uh I'm a bit ashamed of it because it's uh too simple. Uh, basically gets the current weather. And as you can see, uh if location is uh Madrid, it's going to return a temperature of 25 uh, it varies on the unit that you want to to put, but given that it's Madrid, it will be unit Celsius. So it's going to return a temperature of 25 degrees Celsius, and otherwise it's going to return 58.

[03:30:00]
So as you can see, don't pay too much attention to this function because it's uh trivial, but uh it will help us to illustrate how a tool works. So, if we run this, as I was saying, is if we run this function with location Madrid and unit Celsius, it's going to return this um dictionary, well, this string containing a dictionary with temperature 25 and unit Celsius. So, nothing to add about this thing. This is trivial. Okay, so let's proceed.

[04:00:00]
Now the question is, how can we make this function available to an LLM? Because as you already know, LLMs are just NLP systems, a Natural Language Processing system, so they expect text as input. But we need a way to for the LLM to really understand that this is a Python function and I can call this Python function to retrieve some relevant results. And how can we do that? Okay. So what I propose here is to use this system prompt. So, as you can see in this system prompt, we are telling the LLM to behave as a function calling AI model.

[04:30:00]
We are going to provide the function signatures within these XML tags, these uh tools tags. And you may call one or more functions to assist with the user query, don't make assumptions about what values, blah, blah, blah. Okay, but the important thing is that we are going to pass all the relevant information within this XML tag and the LLM is going to return the function call inside this XML tag. Okay, this tool_underscore tag, uh underscore call, sorry.

[05:00:00]
(The speaker scrolls down to a section labeled "A System Prompt that works," which contains a long text prompt with placeholders and XML-like tags such as `<tools>` and `<tool_call>`.)
You can see here an example of how we expect the LLM to return the tool call. It's going to be something like this. We are going to uh, the LLM is going to provide a name, the name of the function, and also the arguments that we need to use to retrieve the relevant information with this Python function. And then a list of the available tools. In this case, I'm just using this one like get current weather because uh I need it to hard code everything for this tiny example. But as you will see in the VS Code, we are going to make it automatic.

[05:30:00]
So, given a Python function, we are going to retrieve all of this information, all of these uh function signature, it's going to be retrieved automatically in the VS Code uh implementation. But yeah, if you checked the way the information that we are providing for each tool, you can see that we are providing the name of the tool, a description. This is something that we can get from the doc string, by the way. You we will see that later. But yeah, like get the current weather in a given location, blah, blah, blah.

[06:00:00]
And then the parameters, where we are putting all the different parameters and this is really important, the type of these parameters. In this case, both the location and the unit are going to be strings, but suppose that we are passing, I don't know, uh the month and we want it to behave like an integer, then we should put that type inside the the function signature. Okay, so now that we know how this system prompt works, let's put it into practice.

[06:30:00]
Just a quick reminder, today we are going to use a different LLM than the previous video. In the previous video we were using Llama 3 70 billion, but today we are going to use a slightly different LLM because it's the Llama 3 70 billion tool use. So, it's a version of Llama 3 that's been uh fine tuned for tool use and that's exactly what we want to do today. So it made sense to to use this LLM.

[07:00:00]
Okay. Uh we defined uh a constant, the system prompt where we copy and paste the system prompt that I shared with you uh right in the in the cell below. And and now let's run this cell. We are going to ask the LLM, "What's the current temperature in Madrid in Celsius?" We're going to add the system prompt and we are also going to add the user uh message to the history.

[07:30:00]
(The speaker executes a code cell in the notebook. The output shows a `tool_call` with the function name `get_current_weather` and arguments `{'location': 'Madrid', 'unit': 'celsius'}`.)
And yeah, and let's run this. Okay, so as you can see, we are having a structure similar to the one we asked for the LLM to return in the system prompt. The LLM is returning the name of the tool and it's also returning the arguments. Since we ask, "What's the current temperature in Madrid in Celsius?", the argument is going to be Madrid as the location and Celsius as the unit. Okay, but now this is not usable by the LLM. I mean, we have a string and inside that string, we have this dictionary inside these two XML tags.

[08:30:00]
So, we need a way to get rid of the XML tags and also transform this dictionary, this string dictionary, into a proper dictionary using the JSON package, the JSON library. Okay, and that's exactly what this function does. This function will get rid of the tool call, or to be more specific, it will gather, it will get the code inside the tool call XML tags, and then it will transform that string dictionary into a proper dictionary.

[09:00:00]
So let me show you how it works. Uh as you can see, when we call this parse tool called string this method to the output. The output, remember that it's uh this one here. It's going to return a proper Python dictionary. And now if we run the get current weather, the function that we defined at the beginning of the notebook, if we run this function with the parameters that we have just parsed, it will return the result.

[09:30:00]
So temperature 25 and unit is going to be Celsius. Okay? Without any information about the XML tags. That's something that we want to get rid of. Nice. Okay, so now we have the result, as you can see, it's this Python dictionary right here, but we are not over because we don't want the LLM to respond with this structure. I mean, if I ask for the current temperature in Madrid, I expect the LLM to respond me something like, "The current temperature in Madrid is uh is 25 degrees Celsius," for example.

[10:00:00]
But not something like this, not this dictionary. So, the last thing that we need to do is to add this observation, the dictionary in here to the chat history. Okay? And we are going to add this into the prompt, this observation uh text, the observation tag to the prompt. And finally, just call the the agent. And as you can see, the result is exactly what we expected.

[10:30:00]
So, the current temperature in Madrid is 25 degrees Celsius. Okay. So now, this is everything for this dynamic or step by step way of doing things, but as you might imagine, this is not scalable. I mean, we can't generate this function signature for everything that we are going to build. I mean, we could, but it's not going to be efficient. We need a way for the agent to, given a Python function, being able to extract the function signature.

[11:00:00]
And by signature, I mean this type of structure right here. And also to decide between different tools. So instead of doing all of this process, we need the agent to extract all of this logic away from the user and to do it under the hood. And that's exactly what we are going to do right now, the the logic that I'm going to show you in VS Code, how to implement all of this the proper way.

*The speaker demonstrates a manual, step-by-step process for LLM tool use in a notebook and highlights the need for a more scalable, automated implementation that abstracts this logic away from the user.*

[11:30:00]
So, let's get into VS Code. (The screen switches to Visual Studio Code. The file explorer on the left shows a directory structure including 'src/agentic_patterns' with subdirectories 'reflection_pattern' and 'tool_pattern'. Inside 'tool_pattern', there are files like 'tool.py' and 'tool_agent.py'.) Okay, so here we are in VS Code. Let me show you the new modules that I have added to the repository. So if you go to the source agentic patterns folder, you will find a new folder, the tool pattern folder. And inside you have three modules, the tool agent, the tool, and the utils. Uh let's begin with the tool because I think it's the most important topic of today's video and the tool agent at the end of the day is just a way to interact with the tool. Okay, so this module starts by implementing a method that allows you to get the signature out of a Python function.

[12:30:00]
So, this is basically the the method I'm referring to. It receives as parameter a function and it will get the schema and out of the schema also the function signature. And the function signature, it's basically the structure that we defined on the system prompt previously. All right. Next, we have this class right here, tool class that has three attributes, a name, the function and the function signature.

[13:00:00]
The function signature, as you might imagine, it's going to be generated by this function right here and the function is basically the function that we want to call when the LLM uh decides that it wants to use a specific tool, this function is the Python function that's going to be used under the hood. And then we have this tool decorator that can be used to decorate your Python function and to automatically transform the Python function into a tool object.

[13:30:00]
If you inspect a little bit the implementation of this decorator, you can see that first, uh it generates the function signature out of the get function signature method that we explained uh before. And then it returns a tool object by uh defining the name using the function signature, passing the function that you are decorating as the function attribute that the tool expects. And finally, getting the function signature uh from the variable that we defined previously because remember that we were getting the function signature using this method.

[14:00:00]
And yeah, and having these three attributes, we are able to to generate a tool. Okay. Now let's move into the tool agent, which as you can imagine, is an agent that has the capability of using tools. You pass a list of tools and it will uh select the proper tool, the the right tool for the specific question that we are asking and then it will run the tool to fetch the relevant details that it needs from the outside world and then returning all this information in a natural language to you.

[14:30:00]
Okay. So, things that you are already familiar with. So this tool system prompt is basically the one that we explained earlier in on the video. And then the tool agent consists of the following attributes. So we need to generate uh the the groq client. Then the model that remember that by default we are going to use the Llama 3 70 billion tool use. And then this is the the important part. This is the the tricky part of this agent.

[15:00:00]
But we need to define the list of tools that we are going to to use for this agent. And then this list of tools are going to be used in the run method. So the run method consists of the following steps. First of all, we expect this user message and we transform this user message into a user prompt using the OpenAI API definition. Then we are going to generate both the tool chat history and the agent chat history. And now we are going to generate the first completion. We are going to make the first call to the groq model.

[15:30:00]
And what this is going to do, these two blocks of code is to generate basically the logic that we explained in the notebook. Let me be specific. So it's going to first of all return the tool call. Okay? This first call, this tool call string is basically this output. And then the parse tool called string, it's a method that mimics the same logic that we implemented in this function.

[16:00:00]
Okay? So, at the end, this uh tool call it's going to be something like this. Okay, so now that we have the tool call information, we can get the tool name from this object, from the tool call. We can also get the tool by using this tool dict because now that we have the tool name, we have also defined a dictionary that contains a relationship between uh the tool name and the tool. Okay? Then we are going to validate the arguments.

[16:30:00]
So to make sure that if the function expects a string, the LLM is not sending an integer. We want to make sure that the types that the LLM has generated in the tool call and the types expected by the Python function match. Okay. And then we are just going to run the tool with this tool.run and we are passing the arguments that we have just uh defined on the tool call. Remember that if we go to to the tool call, remember that we have this arguments key that contains the arguments and its values to to retrieve the the proper information.

[17:30:00]
And then, as you can see, this is a very simple function. So we just need this top end argument and it's of type integer. So everything seems to be working fine. And now let's move into the tool agent. Okay, so the tool agent, to instantiate this tool, we just need a list of tools, in this case, we are only using one tool, the HN tool. And now let's run the agent. And in this case, I wanted to check that everything works properly by doing the following strategy.

[21:30:00]
(The speaker scrolls to the section in the notebook titled "The ToolAgent" and executes a cell to create a ToolAgent instance.)
So, first of all, I'm going to ask the agent about something that it's not related to Hacker News. For example, "Tell me your name." And if everything works properly, we should see, yeah, something not related with the agent, with the tool, sorry. And as you can see, giving the output, the agent has not used any kind of tool. And that's the proper way to work because if the user message is not related to any tool, we don't want the agent to spend time on interacting with tools.

[22:00:00]
But what happens if we ask the same agent about the top five Hacker News stories right now? (The speaker runs the next cell, prompting the agent.) So in this case, we should expect the agent to use the tool. And as you can see, uh I have added some login to make it easier to see, but check this. So the agent is using the tool, the fetch top Hacker News stories. It's using the tool with this call dict. So this is the name and the arguments, the top N with a value of five.

[22:30:00]
And finally, it's generating a result. But remember that we don't want this kind of result. I mean, if I'm asking about the five top stories in Hacker News right now, I'm expecting something easier to understand. (The speaker prints the final output, which is a nicely formatted list of the top 5 stories from Hacker News.) And that's what we achieve when we print the output. And here we have the five top stories in Hacker News. The first one is the the article about "Too much efficiency makes everything worse" that we saw in the Hacker News page, and if we click the URL attached, you can see that everything seems to be working fine.

[23:00:00]
I mean, it's not like the agent redirected us to some broken URLs. I mean, the URLs are real and it's working as expected. So yeah, this is everything I wanted to teach you about tools. My hope is that now when you start using or keep using uh tools from LangChain, LlamaIndex, or Crew AI, you have a deeper understanding of how these objects uh work under the hood.

[23:30:00]
And and this is everything for today. I'm working on the next videos of this series, the video about the planning pattern and the video about the multi-agent pattern. I think you are also going to to enjoy uh those ones. And but yeah, this is everything for today. I hope you have enjoyed the video. Subscribe to the channel if you haven't and if you like the content. Click the like button if you you have enjoyed this video. And I'll see you in the next video. (The video ends with an outro showing "THE NEURAL MAZE" logo.)

*The speaker details the VS Code implementation, explaining how to automatically generate tool signatures from Python functions, create tool objects, and build a ToolAgent that can intelligently select and execute the correct tool, then use the result to formulate a natural language response.*

</details>

<details>
<summary>(The video starts with a man in a black t-shirt and glasses standing in front of a black background. The text "IBM Technology" is in the top-left corner.)</summary>

(The video starts with a man in a black t-shirt and glasses standing in front of a black background. The text "IBM Technology" is in the top-left corner.)

So what is tool calling? Tool calling is a powerful technique where you make the LLM context-aware of real-time data, such as databases or APIs. Typically, you use tool calling via a chat interface. So you would have your client application in one hand, and then the LLM on the other side.

[00:30]
(The speaker draws two vertical boxes on the screen. He labels the left box "APP" and the right box "LLM". Above the space between them, he writes "chat". He then draws a green arrow pointing from the "APP" box to the "LLM" box.)

So you would have your messages here, together with your list of tools. The LLM will look at both your message and the list of tools, and it's going to recommend a tool you should call.

*Tool calling is a technique that enables a Large Language Model (LLM) to interact with real-time data sources like APIs and databases through a chat interface, where the application sends messages and a list of available tools to the LLM, which then recommends a specific tool to call.*

[01:00]
(The speaker draws a green arrow pointing back from the "LLM" to the "APP", labeling it "tool to call". He then draws a third arrow from "APP" to "LLM" labeled "tool response" and a final arrow from "LLM" to "APP" for the final answer. On the left, under the "APP" box, he draws another box labeled "tool definition" with a list inside: "-name", "-description", "-input". Below that, he draws a box labeled "tools" with arrows pointing to circles labeled "API", "DB", and "Code".)

From your client application, you should call this tool and then supply the answer back to the LLM. So this tool response will be interpreted by the LLM, and this will either tell you the next tool to call or it will give you the final answer. In your application, you're responsible for creating the tool definition. So this tool definition includes a couple of things, such as the name of every tool. It also includes a description for the tool. So this is where you can give additional information about how to use the tool or when to use it. It also includes the input parameters needed to make a tool call. And the tools can be anything. So the tools could be APIs or databases. But it could also be code that you interpret via a code interpreter. So let's look at an example.

[01:30]
(The speaker begins writing on the arrows in the diagram.)

Assume you want to find the weather in Miami. You might ask the LLM about the temperature in Miami.

[02:00]
(He writes "temp in Miami" on the first arrow from APP to LLM. He then writes "Weather" next to it.)

You also provide a list of tools. And one of these tools is the weather API. The LLM will look at both your question, which is what is the temperature in Miami. It will also look at the weather API, and then based on the tool definition for the weather API, it's going to tell you how to call the weather tool. So in here, it's going to create a tool that you can use right here on this side where you call the API to collect the weather information. You would then supply the weather information back to the LLM. So let's say it would be 71 degrees.

[02:30]
(He writes "71°" on the "tool response" arrow.)

The LLM will look at the tool response and then give the final answer, which might be something in the trend of the weather in Miami is pretty nice, it's 71 degrees. This has some downsides. So when you do traditional tool calling where you have an LLM and a client application, you could see the LLM hallucinate.

*In a traditional tool calling workflow, an application sends a user's query (e.g., "weather in Miami") and a list of defined tools (e.g., a weather API) to an LLM, which then suggests the correct tool and parameters to call; after the application executes the call and returns the result (e.g., 71°), the LLM provides a final, natural language answer.*

(Under the "LLM" column, the speaker writes "-hallucinate" and "-incorrect".)

Sometimes the LLM can also make up incorrect tool calls. That's why I also want to look at embedded tool calling.

[03:00]
(He writes "embedded" at the top of the screen.)

We just looked at traditional tool calling. But traditional tool calling has its flaws. As I mentioned, the LLM could hallucinate or create incorrect tool calls. That's why you also want to take embedded tool calling into account. With embedded tool calling, you use a library or framework to interact with the LLM and your tool definitions. The library would be somewhere between your application and the large language model.

[03:30]
(He draws a new, larger box in the middle of the screen, between the "APP" and "LLM" boxes, and labels it "library". Inside the box, he writes "tool def" and "tool exec".)

In the library, you would do the tool definition, but you will also execute the tool calls. So let's draw a line between these sections here. So the library will contain your tool definition. It will also contain the tool execution. So when you send a message from your application to the large language model, it will go through the library. So your message could still be, what is the temperature in Miami?

[04:00]
(He draws a new set of arrows. The first goes from "APP" into the "library" box, labeled "temp in Miami?". The next arrow goes from the "library" to the "LLM", labeled "message + tool".)

The library will then append the tool definition and send your message together with the tools to the LLM. So this will be your message, plus your list of tools. Instead of sending the tool to call to the application or the user, it will be sent to the library, which will then do the tool execution. And this way, the library will provide you with the final answer, which could be it's 71 degrees in Miami.

[04:30]
(He draws an arrow from "LLM" to the "library", then a final arrow from the "library" to the "APP", labeled "71°".)

When you use embedded tool calling, the LLM will no longer hallucinate as the library to help you with the tool calling, or the embedded tool calling, is going to take care of the tool execution and will retry the tool calls in case it's needed. So in this video, we looked at both traditional tool calling and also embedded tool calling. Where especially embedded tool calling will help you to prevent hallucination or help you with the execution of tools, which could be APIs, databases, or code.

(The video ends with a solid blue screen with the IBM logo in the bottom-left corner.)

*Embedded tool calling introduces a library layer that manages tool definitions and execution, preventing LLM hallucinations and incorrect calls by handling the interaction, retrying failed executions, and directly returning the final data to the application.*

</details>


## Additional Sources Scraped

<details>
<summary>agentic-design-patterns-part-3-tool-use</summary>

Tool Use, in which an LLM is given functions it can request to call for gathering information, taking action, or manipulating data, is a key design pattern of [AI agentic workflows](https://www.deeplearning.ai/the-batch/how-agents-can-improve-llm-performance/?utm_campaign=The%20Batch&utm_source=hs_email&utm_medium=email&_hsenc=p2ANqtz--9ARMthd09q0ABUi-abo6BH62BLbcwPo13LrXs9hUezs-L050Ay7b_rHdWuRIqBVOD6k_S). You may be familiar with LLM-based systems that can perform a web search or execute code. Indeed, some large, consumer-facing LLMs already incorporate these features. But Tool Use goes well beyond these examples.

If you prompt an online LLM-based chat system, “What is the best coffee maker according to reviewers?”, it might decide to carry out a web search and download one or more web pages to gain context. Early on, LLM developers realized that relying only on a pre-trained transformer to generate output tokens is limiting, and that giving an LLM a tool for web search lets it do much more. With such a tool, an LLM is either fine-tuned or prompted (perhaps with few-shot prompting) to generate a special string like _{tool: web-search, query: "coffee maker reviews"}_ to request calling a search engine. (The exact format of the string depends on the implementation.) A post-processing step then looks for strings like these, calls the web search function with the relevant parameters when it finds one, and passes the result back to the LLM as additional input context for further processing.

Similarly, if you ask, “If I invest $100 at compound 7% interest for 12 years, what do I have at the end?”, rather than trying to generate the answer directly using a transformer network — which is unlikely to result in the right answer — the LLM might use a code execution tool to run a Python command to compute 1 _00 \* (1+0.07)\*\*12_ to get the right answer. The LLM might generate a string like this: _{tool: python-interpreter, code: "100 \* (1+0.07)\*\*12"}_.

But Tool Use in agentic workflows now goes much further. Developers are using functions to search different sources (web, Wikipedia, arXiv, etc.), to interface with productivity tools (send email, read/write calendar entries, etc.), generate or interpret images, and much more. We can prompt an LLM using context that gives detailed descriptions of many functions. These descriptions might include a text description of what the function does plus details of what arguments the function expects. And we’d expect the LLM to automatically choose the right function to call to do a job. Further, systems are being built in which the LLM has access to hundreds of tools. In such settings, there might be too many functions at your disposal to put all of them into the LLM context, so you might use heuristics to pick the most relevant subset to include in the LLM context at the current step of processing. This technique, which is described in the Gorilla paper cited below, is reminiscent of how, if there is too much text to include as context, retrieval augmented generation (RAG) systems offer heuristics for picking a subset of the text to include.

Early in the history of LLMs, before widespread availability of large multimodal models (LMMs)  like LLaVa, GPT-4V, and Gemini, LLMs could not process images directly, so a lot of work on Tool Use was carried out by the computer vision community. At that time, the only way for an LLM-based system to manipulate an image was by calling a function to, say, carry out object recognition or some other function on it. Since then, practices for Tool Use have exploded. GPT-4’s function calling capability, released in the middle of last year, was a significant step toward a general-purpose implementation. Since then, more and more LLMs are being developed to be similarly facile with Tool Use.

</details>

<details>
<summary>arxiv-org</summary>

In this work, we propose a new method for LLMs to better leverage tools in multi-step reasoning. Our method, Chain-of-Abstraction (CoA), trains LLMs to first decode reasoning chains with abstract placeholders, and then call domain tools to reify each reasoning chain by filling in specific knowledge. This planning with abstract chains enables LLMs to learn more general reasoning strategies, which are robust to shifts of domain knowledge (e.g., math results) relevant to different reasoning questions. It also allows LLMs to perform decoding and calling of external tools in parallel, which avoids the inference delay caused by waiting for tool responses. In mathematical reasoning and Wiki QA domains, we show that our method consistently outperforms previous chain-of-thought and tool-augmented baselines on both in-distribution and out-of-distribution test sets, with an average $\\sim 6 %$ absolute QA accuracy improvement. LLM agents trained with our method also show more efficient tool use, with inference speed being on average ${ \\sim } 1 . 4 \\times$ faster than baseline tool-augmented LLMs.

# 1 Introduction

Recent large language models (LLMs; Touvron et al., 2023b; Anil et al., 2023; OpenAI, 2023), have made progress at interpreting and executing instructions (Wei et al., 2021; Chung et al., 2022), but still make errors when recalling and composing world knowledge for their responses, e.g., making unfactual statements (Maynez et al., 2020; Ji et al., 2023), incorrect calculations (Patel et al., 2021), etc. Using auxiliary tools (e.g., a search engine to provide credible facts, a calculator for accurate math operations, etc.) at inference time can mitigate some of these errors, motivating tool-augmented language models that integrate external API calls into their output generations (Parisi et al., 2022; Schick et al., 2023; Hao et al., 2023b).https://arxiv.org/pdf/images/531cf1db549587897999e68f15d990a961defd9f109574b6d15228058913e550.jpg

However, we show that current tool-augmented LLMs, e.g., Toolformer (Schick et al., 2023), struggle to reliably and efficiently leverage tools in multi-step reasoning. In particular, tool calls in multi-step reasoning tasks are often interleaved (i.e., the response of an API call is often part of the query of a subsequent call; as shown in Figure 1). Without explicitly modeling these interconnections in reasoning chains, LLMs do not learn effective planning for tool use, which leads to less accurate reasoning with tools.1 Meanwhile, interleaving text generation with API calls also introduces inefficient inference “waiting times,” where the model must wait for the response from the API call before resuming the decoding process. This inefficiency becomes more significant in multi-step reasoning scenarios, when multiple rounds of API calls are typically required for each reasoning process.

In this work, we propose Chain-of-Abstraction (CoA) reasoning, a robust and efficient method for LLMs to perform multi-step reasoning with tools. As shown in Figure 1, LLMs are fine-tuned with a goal of making reasoning chains with abstract placeholders. The placeholders do not affect LLMs’ reasoning flow, and are subsequently infilled with specific knowledge retrieved from specialized tools, to ground the final answer generations. Planning abstract chain of reasoning encourages LLMs to inter-connect multiple tool calls and adopt more feasible reasoning strategies, which are robust to the variation of domain knowledge involved in each reasoning process, e.g., specific calculation results. Unlike previous methods where LLM decoding and API calls are executed in an interleaved manner, our method leverages tools to infill knowledge once after the whole chain of reasoning is generated. This enables more efficient decoding across multiple examples (e.g., as in a stream) because CoA traces for subsequent examples can be decoded while tool calls are made for the preceding ones, amortizing overall inference time. We develop a simple pipeline to build fine-tuning data for models to learn CoA, where we first prompt LLMs to re-write existing responses to instructions as abstract chains, and then use domain tools to check the validity of re-writing, as shown in Figure 2.

After training LLMs to learn CoA reasoning, we evaluate the finetuned models on two representative multi-step reasoning domains, including mathematical reasoning (Cobbe et al., 2021; Miao et al., 2020; Patel et al., 2021; Koncel-Kedziorski et al., 2016), and Wikipedia (Wiki) QA (Yang et al., 2018; Berant et al., 2013; Kwiatkowski et al., 2019; Joshi et al., 2017) that involves reasoning on factual descriptive knowledge. We show that our method boosts LLMs’ performances, with average ${ \\sim } 7 . 5 %$ and $4 . 5 %$ absolute accuracy improvements on math and Wiki QA, respectively. These improvements are consistent across both in-distribution and (zeroshot) out-of-distribution test sets, and are especially pronounced on questions that require complex chain-of-thought reasoning.2 Meanwhile, our method also uses tools more efficiently than previous augmentation methods, with average ${ \\sim } 1 . 4 7 \\times$ and $1 . 3 3 \\times$ faster inference speeds on math and Wiki QA tasks, respectively. Finally, extensive human evaluation demonstrates that our method guides LLMs to learn more accurate reasoning, which leads to $\\sim 8 %$ fewer reasoning errors.

# 2 Related Work

Tool-Augmented LLMs There is growing interest in augmenting LLMs using external tools. Considerable work has tried to adapt LLMs as tool-using reasoners through in-context learning, demonstrating promising performance improvements in various applications, e.g., math problem solving (Gao et al., 2023; Chen et al., 2022), biomedical question answering (Jin et al., 2023) and self-critiquing (Gou et al., 2023). Nevertheless, guiding LLMs to effectively use tools using in-context demonstrations is challenging, which requires elaborate task-specific prompt engineering and is restricted by the model’s instruction following ability (Jacovi et al., 2023). Noticing the limitations of in-context learning, several works teach LLMs to learn the usage of tools by fine-tuning (Parisi et al., 2022; Schick et al., 2023; Hao et al., 2023b), which more robustly improves LLMs’ performance. However, all above approaches adopt sequential interactions with tools throughout reasoning, slowing the inference speed as a function of the latency of the tool (or API) and the number of API calls that are made.

Some other prior works focus on using LLMs for multi-step reasoning with other modules. In particular, ReAct (Yao et al., 2023b) and FireAct (Chen et al., 2023) integrate LLMs with tools into a closed loop of thought, action and observation steps. This verbose reasoning loop slows down the LLM decoding, and still incorporates tools via sequential interactions, resulting in inefficient inference. Another line of work, Program of Thoughts (Chen et al., 2022), DECLARATIVE (He-Yueya et al., 2023) and PAL (Gao et al., 2023) prompt LLMs to generate program-based reasoning and interact with code executors, which however heavily rely on closed source coding models, i.e., Codex (Chen et al., 2021), and are restricted to procedural arithmetic reasoning. Building on these works, CoA proposes a framework to convert natural language reasoning traces into abstract representations, and uses the abstract reasoning traces as fine-tuning data to improve tool-augmented LLMs. CoA also accelerates tool-augmented reasoning, by holistically planning the CoA traces and calling tools only once at inference time.

Tool Usage Planning Several previous works research tool usage planning in LLMs. Specifically, HuggingGPT (Shen et al., 2023), Chameleon (Lu et al., 2023), OpenAGI (Ge et al., 2023) and MetaTool (Huang et al., 2023) focus on planning the high-level sequence of using multiple tools to address multi-domain mixed tasks. Similarly, LATM (Cai et al., 2023), ML-BENCH (Liu et al., 2023) and Gorilla (Patil et al., 2023) aim at planning program-level integration of multiple APIs for designing scripts of procedural tasks, e.g., a script for training a model described by a GitHub repository. ToolChain\* (Zhuang et al., 2023) combines the planning of tool usage with tree-search-based reasoning (Yao et al., 2023a; Hao et al., 2023a), which is especially useful for procedural tasks (Xu et al., 2023; Cobbe et al., 2021). Different from above work, we focus on the planning of general chain-of-thought (Wei et al., 2022) reasoning with awareness of domain specialized tools.

# 3 Method

Chain-of-Abstraction (CoA) Reasoning Our method decouples the general reasoning of LLMs from domain-specific knowledge obtained from external tools. Figure 1 shows an overview of our method. In particular, we first fine-tune LLMs to generate reasoning chains with abstract placeholders, e.g., y1, $y 2$ and $y 3$ ,3 as shown in Figure 1. In the second stage, we reify each reasoning chain by replacing placeholders with domain-specific knowledge obtained from external tools, e.g., calculation results from a calculator, relevant articles retrieved from web search engine, etc. Finally, the question is answered based on the reified reasoning chain.

Note that since the LLMs are trained to generate abstract chain of reasoning instead of regular chain-of-thought (CoT) reasoning with explicit values, this enables LLMs to focus on learning general and holistic reasoning strategies without needing to generate instance-specific knowledge for the model’s parameters. Moreover, decoupling general reasoning and domain-specific knowledge enables LLM decoding to proceed and switch between different samples in parallel with API calling (via a pipeline), i.e., LLM can start generating the next abstract chain while the tool fills the current chain, which speeds up the overall inference process.https://arxiv.org/pdf/images/e4041df46a6b9775dfa5b0ae38406af5eeaa36a4a6901c6d564126311c579e3b.jpg

Fine-tuning Data Construction To construct chain-of-abstraction (CoA) data for fine-tuning LLMs, we collect question answering (QA) samples from existing open-source QA datasets (Cobbe et al., 2021; Miao et al., 2020; Yang et al., 2018), and prompt LLaMa-70B (Touvron et al., 2023a) to re-write the answer of each sampled question, as shown in Figure 2. Specifically, we prompt LLaMa-70B to label the spans in gold answers that correspond to knowledge operations (e.g., math derivations, statements based on Wikipedia references) and then to re-write the sentences with labeled spans as fillable CoA traces, where the operation results are replaced with abstract placeholders. For example, the two derivations in the example in Figure 2 are re-written as $\\mathbf { \\dot { \\ell } } \[ 2 0 + 3 5 = y 1 \] "$ and “ $\[ 9 0 - y 1 = y 2 \] "$ , respectively.

Note that an intermediate knowledge operation result may appear multiple times in an answer, e.g., in Figure 2, the first equation’s result 55 is used in the second equation. We prompt LLaMa-70B to replace all occurrences of the same intermediate result with the same placeholder, thereby explicitly connecting the multiple reasoning steps. To ensure that the re-written data is accurate, we use domainspecialized tools to verify the correctness of each CoA reasoning trace.4 Specifically, we use the tools to execute the labeled operations in each CoA, and only keep questions whose CoA can be infilled with valid results by the tools.

# 4 Experimental Settings

We conduct our experiments on two representative domains: mathematical reasoning and Wikipedia (Wiki) QA, which involves commonsense and logical reasoning on factual descriptive knowledge.

# 4.1 Mathematical Reasoning

Given a math question, the QA system needs to generate a natural language solution to the problem with step-by-step arithmetic derivations (as demonstrated in the left column of Figure 1). We assume that the derivations involved in the solution are the specialized knowledge operations required in this domain, which are labeled in square brackets with derivation results being replaced by abstract placeholders, e.g., “ $\[ 2 0 + 3 5 = y 1 \] ^ { \\prime }$ .

Datasets We construct most of our fine-tuning CoA data by re-writing the GSM8K (Cobbe et al., 2021) training set, which contains 7473 linguistically diverse grade school math problems. As GSM8K dataset focuses on multi-step reasoning, it lacks coverage of single-step arithmetic problems, so we also re-write an additional set of 691 singlestep math problems from the ASDiv (Miao et al., 2020) dataset. Across these re-written datasets, we find that $\\sim 7 6 . 6 %$ of the CoA reasoning traces generated by LLaMa-70B are verified by our equation solver (described below). Table 1 shows the reasoning step distribution (i.e., number of derivations) of our constructed fine-tuning data.

For an in-distribution evaluation, we test models on GSM8K and ASDiv, containing 1319 and 2305 testing problems. To further test the models’ generalization ability, we also conduct zero-shot evaluation on other representative math datasets, including SVAMP (Patel et al., 2021) and MAWPS (Koncel-Kedziorski et al., 2016), which contain 1000 and 2065 testing samples, respectively.5

|     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Source | Reasoning Step |
| 1 | 2 | 3 | 4 | 5 | >5 | All |
| GSM8K | 8 | 1540 | 1648 | 1164 | 666 | 553 | 5579 |
| ASDiv | 677 | 0 | 0 | 0 | 0 | 0 | 677 |

Domain Tool We use an equation solver to perform the arithmetic derivations required in the math domain. Our equation solver first extracts the derivations labeled in the CoA reasoning, e.g., “ $\[ 2 0 + 3 5 = y 1 \] "$ and $^ { \\bullet } \[ 9 0 - y 1 = y 2 \] ^ { \\prime }$ , and combines all derivations into a system of equations. Then the system of equations is solved by the SymPy toolkit,6 to get the true value of each variable (i.e., the value of the abstract placeholder). Finally, our equation solver returns the reified chain of reasoning by replacing all the variables with their solved true values (including the final answer).

# 4.2 Wikipedia QA

Given a question based on Wikipedia knowledge, the model needs to first identify Wikipedia articles as references related to the question, and then reason on key knowledge in the reference articles to answer the question (as shown in the right column of Figure 1). We assume that the specialized knowledge operation in this domain is the retrieval of relevant Wikipedia articles and important named-entities, which are re-written as Wikipedia searching (WikiSearch) and named-entity recognition (NER)7 queries. Table 2 shows an example of a re-written CoA trace for Wiki QA.

|     |     |
| --- | --- |
| Question | The director of the romantic comedy “Big Stone Gap”is based in whatNew York city? |
| Answer | Greenwich Village |
| Wikipedia References | Big Stone Gap (film) > Big Stone Gap is a 2014 American romantic comedy film directed by Adriana Trigiani. Adriana Trigiani >Adriana Trigiani is an Italian American film director based in Greenwich Village. |
| CoA Trace | Find the \[director of romantic comedy“Big Stone Gap”-Wiki-> y1\]. The name of this film's director is \[y1 -NER(person)-> y2\]. Then determine \[y2 in what New York city -Wiki-> y3\]. |

Datasets We use the HotpotQA (Yang et al., 2018) dataset to construct our fine-tuning CoA data in the Wiki QA domain. HotpotQA contains 113K multi-hop QA examples, each labeled with two Wikipedia articles that provide supporting knowledge. Among the 90447 training QA pairs, we identify 72991 as Bridge QA pairs, where an intermediate entity must be identified to link the answer to the question, as shown in Table 2. The remaining 17456 are Comparison QA pairs, where the attributes of two entities are compared, e.g., “Are Randal Kleiser and Kyle Schickner of the same nationality?”. We prompt LLaMa-70B to re-write these training QAs into CoAs with WikiSearch and NER queries, and verify each CoA with our domain tools (described below), by checking whether all the articles returned by the WikiSearch queries match one of the titles in the gold articles. Finally, 8956 Bridge QAs and 5405 Comparison QAs are used as fine-tuning data, whose re-written CoAs pass the verification.8 For Wiki QA, we note that besides training a LLM to produce CoA data using WikiSearch, we also fine-tune a second LLM to learn to generate the final gold answer based on a correctly reified CoA reasoning trace.

We evaluate models on the HotpotQA development set, which contains 5918 Bridge QA pairs and 1487 Comparison QA pairs. Similar to the mathematical reasoning domain, we also conduct zeroshot evaluation on other open-domain QA datasets: WebQuestions (WQ; Berant et al., 2013), NaturalQuestions (NQ; Kwiatkowski et al., 2019) and TriviaQA (Joshi et al., 2017), which contain 2032, 3610 and 17944 test questions, respectively.

Domain Tools The specialized tools required for Wiki QA include a Wikipedia search engine to retrieve reference articles, and a NER toolkit to extract entities that bridge multi-step searching queries. We follow Toolformer (Schick et al., 2023) and implement a Wikipedia search engine as a BM25 retriever (Robertson et al., 1995; BaezaYates et al., 1999) that indexes the Wikipedia dump from the KILT benchmark (Petroni et al., 2021). We use the BM25 retriever to search the top-10 articles relevant to the input query, and then re-rank the articles based on their Sentence-BERT (Reimers and Gurevych, 2019) embedding cosine similarity with the question. After re-ranking, the top-1 article is selected to be the final search result.

|     |     |
| --- | --- |
| General Class | SpaCy NER Types included in each General Class |
| person | PERSON |
| group | NORP,ORG,LANGUAGE |
| location | GPE,FAC,LOC |
| culture | EVENT,WORK\_OF\_ART,LAW,PRODUCT |
| date | DATE,TIME |
| numeral | CARDINAL,PERCENT,MONEY,QUANTITY, ORDINAL |

We use $\\operatorname { S p a C y } ^ { 9 }$ (en\_core\_web\_sm) as the NER toolkit to extract named entities. To simplify NER, we aggregate the numerous SpaCy NER types into 6 general classes, as shown in Table 3. If multiple named entities are recognized, we input each recognized entity to the subsequent WikiSearch query, and select the entity whose subsequent search result has the highest Sentence-BERT embedding cosine similarity with the question.

# 4.3 Baselines

We apply our CoA reasoning method to both 7B and 70B LLaMa models, and test various model versions including the first version of LLaMa (Touvron et al., 2023a) and the more advanced LLaMa-2 and LLaMa-2-Chat (Touvron et al., 2023b). We compare our method to several baselines, including: a) few-shot prompting using 8 randomly sampled QA exemplars from the original (i.e., not rewritten) chain-of-thought data (CoT-FSP), b) finetuning with original chain-of-thought data (CoT$\\mathbf { F T } ) ^ { 1 0 }$ , and c) Toolformer (Schick et al., 2023) which fine-tunes LLMs on CCNet (Wenzek et al., 2020) texts augmented with API calls. For evaluation on Wiki QA, we also compared our method with FireAct (Chen et al., 2023), which fine-tunes LLMs on HotpotQA ReAct (Yao et al., 2023b) trajectories distilled from GPT-4 (OpenAI, 2023).

# 5 Results and Analysis

# 5.1 Mathematical Reasoning

Table 4 shows the evaluation results for the LLaMa2 and LLaMa-2-Chat models.11 On the GSM8K and ASDiv datasets, our CoA method outperforms the few-shot baseline CoT-FSP and the regular finetuning baseline CoT-FT, demonstrating that CoA fine-tuning with tool augmentation is more effective in adapting LLMs to multi-step reasoning tasks. Similarly, when evaluated on out-of-distribution datasets, SVAMP and MAWPS, CoA also consistently outperforms the baselines. Interestingly, for these out-of-distribution datasets, CoT-FT lags further behind CoA, particularly for 7B models, showing that CoA reasoning yields more distributionally robust reasoning performance.

|     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | Method | Use Tool | GSM8K | ASDiv | SVAMP | MAWPS |
| AddSub | SingleEQ | SingleOp | MultiArith | All |
| LLaMa-2 -7B | CoT-FSP CoT-FT | X | 16.38 | 47.85 57.18 | 38.40 | 52.41 | 63.39 | 82.03 | 43.33 | 60.53 |
| Toolformer |  | 35.33 |  | 48.20 | 66.08 | 74.41 | 85.23 | 65.00 | 73.03 |
| CoA | 17.59 37.83\* | 48.55 57.61 | 37.10 | 47.34 | 58.46 | 79.54 | 50.67 | 59.81 |
| CoT-FSP CoT-FT LLaMa-2 |  |  | 54.14 | 51.70\* | 72.15\* | 82.48\* | 86.48\* | 73.17\* | 78.89\* |
|  | 24.03 | 51.30 | 71.90 | 72.44 | 85.41 | 74.00 | 76.32 |
| CoA (no Tool) | X 35.41 | 59.00 | 46.90 | 58.23 | 72.24 | 85.41 | 73.00 | 73.37 |
| Toolformer |  | 35.03 | 58.79 | 51.50 | 68.10 | 74.21 | 86.48 | 77.67 | 77.38 |
| Toolformer - Math | 23.65 | 50.85 | 48.80 | 61.01 69.09 | 81.85 | 68.50 | 70.85 |
| LLaMa-2 -Chat-70B | CoA |  | 36.01 38.29\* | 59.18 59.57 | 47.60 54.20\* | 58.99 72.41 | 72.44 81.89\* | 85.94 88.26\* | 75.50 83.00\* | 74.43 82.13\* |
|  |  |  |  |  |  |  |  |
| CoT-FSP CoT-FT | X | 56.18 | 65.94 | 70.60 | 86.08 | 89.17 | 92.88 | 84.50 | 88.23 |
| 60.50 | 70.24 | 70.40 | 81.52 | 87.60 | 92.35 | 89.17 | 88.18 |
| Toolformer | Toolformer- Math | 52.54 | 69.07 | 73.60 | 86.84 | 89.76 | 91.46 | 81.50 | 87.26 |
| 61.03 62.32\* | 70.59 71.89\* | 73.20 73.40 | 85.57 86.33 | 91.34 94.49\* | 91.99 93.06 | 92.00 92.33 | 90.60 91.91\* |

Our CoA method also surpasses the toolaugmented baseline Toolformer, which implies that planning the abstract variables in CoA can improve the accuracy of reasoning with tools. However, as Toolformer is not originally trained with in-domain fine-tuning data,12 we also fine-tune a new version of Toolformer with the chain-of-thought data from GSM8K and ASDiv, denoted as Toolformer - Math in Table 4. We also observe that CoA performs better than Toolformer - Math, confirming that the introduction of abstract variables enables more robust tool use compared to direct integration of API calls within chain-of-thought reasoning.

Ablation Study We verify that the robust generalization performance of our CoA method does not merely benefit from using additional tools, by finetuning another LLM to solve the equation (from the same model backbone), rather than calling the equation solver, denoted as CoA (no Tool) in Table 4.

We find that CoA (no Tool) performs consistently worse than CoA across all datasets, confirming that using specialized tools enables LLM agents to conduct more precise operations, rather than directly solving the same operations. However, CoA (no Tool) still outperforms all baseline methods on zero-shot generalization to SVAMP and MAWPS datasets, implying that learning abstract reasoning chains also contributes to better robustness of CoA, perhaps due to better planning of multiple reasoning steps indexed by abstract variables.

Reasoning Steps Our findings suggest that the benefits of chain-of-abstraction reasoning are most pronounced when problems require long reasoning chains to be solved. Figure 3 shows the stratified performance of three models on GSM8K QA, relative to the number of reasoning steps in the predicted and gold reasoning chains. Compared to the few-shot CoT-FSP, CoA produces reasoning chains that more often match the length of the gold reasoning chains, as reflected by the heat-map statistics (left column) being more aggregated around the diagonal (comparable to CoT-FT). At the same time, we observe that models achieve better QA accuracy when the number of reasoning steps in their generated answers are aligned with the gold references (i.e., the diagonal of heat-maps in right column). Above results show that fine-tuned models are better at learning to produce reasoning chains that match the true reasoning chain for the problem.https://arxiv.org/pdf/images/fc98df9552418713a5ab53c790d0abe22ed530e6c793a328d60c4a7c20485a1c.jpg

|     |     |     |
| --- | --- | --- |
| Method | Error Rate |
| Arithmetic | Reasoning |
| CoT-FSP | 17.3 | 70.3 |
| CoT-FT | 25.2 | 67.8 |
| CoA | 0.0 | 60.4 |

Interestingly, we find that CoA, compared to CoT-FT, achieves higher performance especially on questions that require more reasoning steps. In the right column of Figure 3, CoA’s improvement over CoT-FT is more pronounced on questions with more than 3 steps in the gold reasoning chain (highlighted with red squares). This indicates that the model trained with CoA has more robust long chain-of-thought reasoning capability, which is learned from planning with abstractions.

Human Evaluation To more comprehensively verify that CoA improves both knowledge operation (i.e., arithmetic by using tools) and reasoning accuracy, we conduct a human evaluation on different model answers to 200 randomly sampled GSM8K test questions. Specifically, given a GSM8K question and a model’s answer to the question, we ask human workers to judge whether the answer contains any arithmetic errors (e.g., wrong calculations, invalid equations) or reasoning errors unrelated to math derivations (e.g., misunderstanding of the question, improper strategy for solving the question), and report how often the model makes these two kinds of errors. In Table 5, we find that CoA effectively reduces arithmetic errors to zero, due to the use of equation solver to perform accurate calculations. More importantly, our method also makes fewer reasoning errors compared to the baselines, verifying that CoA finetuning guides the model to learn more accurate reasoning through the holistic planning of abstract reasoning chains. By contrast, ordinary fine-tuning (i.e., CoT-FT) produces a more limited reasoning improvement compared to the few-shot CoT-FSP, while also failing to suppress arithmetic errors.https://arxiv.org/pdf/images/3a7c00ccb41343669f57e19f4da7bea32368a7bf3b94cb800e84b8af14e996f1.jpg

|     |     |
| --- | --- |
| Method | Accuracy |
| CoT-FSP | 27.90 |
| CoT-FT | 39.12 |
| Toolformer | 24.56 |
| Toolformer-Math | 35.25 |
| CoA | 40.79 |

|     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | Method | Use Tool | HotpotQA | WQ | NQ | TriviaQA |
| Bridge | Comparison | Both | Time |
| LLaMa-2 -Chat-7B | CoT-FSP CoT-FT | × | 11.69 | 45.46 | 18.47 | 2.074 | 34.65 | 30.91 | 53.48 |
|  |  | 14.24 | 56.69 | 22.77 | 1.937 | 33.51 | 25.40 | 51.05 |
| Toolformer |  | 12.99 | 44.59 | 20.00 | 2.350 | 36.22 | 30.22 | 54.15 |
| Toolformer - Wiki |  | 15.68 | 56.42 | 23.86 | 2.301 | 36.61 | 32.96 | 55.08 |
| FireAct |  | 19.18 | 54.14 | 26.20 | 2.706 | 36.02 | 35.87 | 52.96 |
| CoA CoT-FSP |  | 21.00\* | 56.96 | 28.22\* | 1.896 | 35.97 | 38.67\* | 57.90\* |
| 21.39 X | 56.62 | 28.47 | 6.668 | 34.89 | 37.42 | 63.61 |
| LLaMa-2 -Chat-70B | CoT-FT |  | 23.84 | 63.95 | 31.90 | 6.401 | 34.15 | 39.75 | 62.28 |
| Toolformer |  | 22.24 | 56.09 | 29.04 | 6.888 | 37.16 | 40.42 | 64.31 |
| Toolformer- Wiki CoA |  | 26.38 | 63.82 | 33.90 | 6.855 | 37.70 | 41.25 | 66.64 |
|  | 27.61\* |  | 64.09 | 34.94\* | 6.369 | 36.37 | 43.57\* | 69.08\* |

Inference Efficiency Importantly, we find that the performance benefits of CoA reasoning do not come with increased computational costs. In Figure 4, we show the average time (seconds) that CoA and baseline agents (seeded with LLaMa2-Chat-7B) needs to answer a question w.r.t. required gold reasoning steps. Compared to the CoT baselines, CoA requires less time than the fewshot baseline CoT-FSP, whose generation needs to be conditioned on additional examples. However, CoA is slightly less inference-efficient compared to CoT-FT, likely due to the decoding of additional tokens (e.g., “\[” and “\]”) for the abstract statements.

Compared to Toolformer, CoA has a lower and flatter inference time curve, indicating better scaling as the number of reasoning steps increases. This difference arises because CoA decouples the generation of (abstract) reasoning chains from the retrieval of knowledge (i.e., tool use), allowing full reasoning chains to be decoded before any tool is called. This procedure amortizes inference costs in two ways. First, tool calls are made after the CoA trace has been decoded, enabling parallel tool calls for the same trace (e.g., using an equation solver once rather than multiple calls to a calculator), and avoiding the time delay caused by waiting for external API responses. Consequently, the model fine-tuned with CoA is more efficient at multi-step reasoning, especially when the number of reasoning steps (i.e., tool calls) increases. Second, across multiple examples, the model can generate the CoA trace of the next example while tool calls are made for the preceding one, parallelizing CoA decoding and tools calls across examples.

Self-Consistency Decoding Besides of greedy decoding, we also test more advanced inference strategy, i.e., self-consistency (Wang et al., 2022) decoding, on our CoA reasoning method. We test all methods on the GSM8K dataset seeded with LLaMa-2-Chat-7B. Each method samples 16 reasoning chains and uses majority voting to aggregate the 16 answers derived by the reasoning chains, to get the final answer. For the hyperparameters of sampling, we set the temperature, top- $\\mathbf { \\nabla } \\cdot \\mathbf { k }$ and top-p as 1.0, 40 and 0.5, respectively. Table 6 shows our evaluation results. We find that our CoA method consistently outperforms all baseline methods when shifting from greedy decoding to selfconsistency decoding. This shows that our method also has better potential to be generalized to different LLM decoding schemes.

# 5.2 Wiki QA

Table 7 shows our Wiki QA results using LLaMa2-Chat models.13 Similar to mathematical reasoning, we fine-tune a new version of Toolformer with in-domain chain-of-thought data from HotpotQA, denoted as Toolformer - Wiki. On HotpotQA, CoA achieves higher exact match rates with the gold reference compared to the few-shot or finetuning baselines. In particular, CoA outperforms all baselines on the more challenging bridge-type QAs, where two steps of reasoning over Wikipedia knowledge are consecutively entangled, i.e., cannot be performed independently in parallel as in comparison-type QAs. Compared to FireAct finetuning, CoA also achieves better performance on both bridge and comparison QAs, without requiring data distilled from closed source GPT-4.

As with mathematical reasoning, CoA agents also perform more efficient inference than Toolformer and FireAct agents when answering HotpotQA questions. We also find that CoA is more efficient (Time column) than both CoT-FSP and CoTFT, as CoA does not require few-shot examples as additional inputs and does not need to generate long Wiki articles, which are instead provided by the search engine. Finally, CoA improves over the baseline methods in zero-shot generalization experiments on other Wiki QA datasets, outperforming all baselines on NaturalQuestions and TriviaQA, and matching the best baselines on WebQuestions.

# 6 Conclusion

In this work, we propose to decouple the general reasoning of LLM agents from specialized knowledge obtained via external tools. Our method, chain-of-abstraction (CoA), encourages LLMs to learn the planning of abstract multi-step reasoning, which are more robust to out-of-distribution knowledge shifts. CoA also achieves a more efficient pipeline for tool usage that significantly improves the speed of tool-augmented multi-step reasoning. The simple, yet effective, implementations of our method on two diverse tasks (i.e., math reasoning and open-domain QA) demonstrate its potential for being adapted to new reasoning scenarios.

</details>

<details>
<summary>building-ai-agents-from-scratch-part-1-tool-use</summary>

### What is an AI Agent?

In it’s simplest high level definition, an AI agent is an application that uses LLM at the core as it’s reasoning engine to decide on the steps it needs to take to solve for users intent. It is usually depicted similar to the picture bellow and is composed of multiple building blocks:

[https://substackcdn.com/image/fetch/$s_!fVcp!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3eb64772-fbb5-4f2d-8120-d473c74fe124_2926x2198.png](https://substackcdn.com/image/fetch/$s_!fVcp!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3eb64772-fbb5-4f2d-8120-d473c74fe124_2926x2198.png) AI Agent

- Planning - the capability to plan a sequence of actions that the application needs to perform in order to solve for the provided intent.

- Memory - short-term and long-term memory containing any information that the agent might need to reason about the actions it needs to take. This information is usually passed to LLM via a system prompt as part of the core.
- Tools - any functions that the application can call to enhance it’s reasoning capabilities. One should not be fooled by the simplicity of this definition as a tool can be literally anything:
  - Simple functions defined in code.
  - VectorDBs and other data stores containing context.
  - Regular Machine Learning model APIs.
  - Other Agents!
  - …

In the following set of articles, I will implement most of the moving parts of an agent from scratch without using any orchestration frameworks. This episode is about Tool use.

If you are using any orchestration frameworks for agentic applications, you might be abstracted away from what using a tool really means. This article will help you understand what providing a tool and using it via an agent involves. I believe that understanding applications from the base building blocks is really important for few reasons:

- Frameworks hide the implementation details of the system prompts used, different approaches might be needed in different use cases.
- You might want to tune the low level details to achieve most optimal performance of the agent.
- Having clarity of how the systems actually work helps build up your systems thinking enabling you to craft advanced applications more efficiently.

### Tool use on a high level.

The basic thing one needs to understand when building agentic applications is that LLMs do not run code, they are only used to produce intent via prompting. Why can ChatGPT browse the internet and return more accurate and recent results? Because ChatGPT IS an agent and there are many non LLM building blocks hidden from us behind the API.

Prompt engineering becomes critical when building agentic applications. More specifically, how you craft the system prompt. Simplified prompt structure looks like the following.

[https://substackcdn.com/image/fetch/$s_!rZHR!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F663cac67-4b46-428f-8876-d648f621f0e5_1878x766.png](https://substackcdn.com/image/fetch/$s_!rZHR!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F663cac67-4b46-428f-8876-d648f621f0e5_1878x766.png) Prompt Structure

The agent will only perform well if you are able to efficiently provide the system prompt with available tool definitions and expected outputs which are in a form of planned actions or raw answers.

### Implementing the Agent.

In this part, we will create an AI Agent, that is capable of checking currency conversion rates online and performing the conversion if needed to answer the user query.

#### Preparing python functions to be used as tools.

The easiest and most convenient way to provide tools to an agent is through functions, in our project we will be using Python for this.

We do not need to provide the function code itself to the system prompt but we need to extract useful information about it so that LLM can decide if and how the function should be invoked.

We’ll define a dataclass that contains desired information including the function runnable.

```
@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., str]
    parameters: Dict[str, Dict[str, str]]

    def __call__(self, *args, **kwargs) -> str:
        return self.func(*args, **kwargs)
```

The information we are extracting includes:

- Function name.
- function description (we will extract this from a docstring).
- Function callable so that we can invoke it as part of the agent.
- Parameters that should be used with the function so that the LLM can decide on how to call the function.

Now we will need to extract the above information from the functions we define. One requirement for the functions we will enforce is to have properly formatted docstrings. We will require the following format:

```
"""Description of what the tool does.

Parameters:
    - param1: Description of first parameter
    - param2: Description of second parameter
"""
```

The following function extracts information about parameters - parameter names and descriptions.

```
def parse_docstring_params(docstring: str) -> Dict[str, str]:
    """Extract parameter descriptions from docstring."""
    if not docstring:
        return {}

    params = {}
    lines = docstring.split('\n')
    in_params = False
    current_param = None

    for line in lines:
        line = line.strip()
        if line.startswith('Parameters:'):
            in_params = True
        elif in_params:
            if line.startswith('-') or line.startswith('*'):
                current_param = line.lstrip('- *').split(':')[0].strip()
                params[current_param] = line.lstrip('- *').split(':')[1].strip()
            elif current_param and line:
                params[current_param] += ' ' + line.strip()
            elif not line:
                in_params = False

    return params
```

We will be extracting function parameter types from typehints provided via function definition. The bellow function will help format them.

```
def get_type_description(type_hint: Any) -> str:
    """Get a human-readable description of a type hint."""
    if isinstance(type_hint, _GenericAlias):
        if type_hint._name == 'Literal':
            return f"one of {type_hint.__args__}"
    return type_hint.__name__
```

A very convenient way to turn a function into a tool is to use a decorator. The below code defines a tool decorator that wraps a function if used. It uses either function name for the tool name or a variable provided via decorator.

```
def tool(name: str = None):
    def decorator(func: Callable[..., str]) -> Tool:
        tool_name = name or func.__name__
        description = inspect.getdoc(func) or "No description available"

        type_hints = get_type_hints(func)
        param_docs = parse_docstring_params(description)
        sig = inspect.signature(func)

        params = {}
        for param_name, param in sig.parameters.items():
            params[param_name] = {
                "type": get_type_description(type_hints.get(param_name, Any)),
                "description": param_docs.get(param_name, "No description available")
            }

        return Tool(
            name=tool_name,
            description=description.split('\n\n')[0],
            func=func,
            parameters=params
        )
    return decorator
```

#### Currency exchange tool.

The below creates a tool from a function that takes in the amount of currency to exchange from, the currency code to be converted from and the currency code to convert to. The function searches for the relevant currency exchange rate and performs the calculation of resulting currency amount.

```
@tool()
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Converts currency using latest exchange rates.

    Parameters:
        - amount: Amount to convert
        - from_currency: Source currency code (e.g., USD)
        - to_currency: Target currency code (e.g., EUR)
    """
    try:
        url = f"https://open.er-api.com/v6/latest/{from_currency.upper()}"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read())

        if "rates" not in data:
            return "Error: Could not fetch exchange rates"

        rate = data["rates"].get(to_currency.upper())
        if not rate:
            return f"Error: No rate found for {to_currency}"

        converted = amount * rate
        return f"{amount} {from_currency.upper()} = {converted:.2f} {to_currency.upper()}"

    except Exception as e:
        return f"Error converting currency: {str(e)}"
```

Let’s just run

```
convert_currency
```

It should return something like

```
Tool(name='convert_currency', description='Converts currency using latest exchange rates.', func=<function convert_currency at 0x106d8fa60>, parameters={'amount': {'type': 'float', 'description': 'Amount to convert'}, 'from_currency': {'type': 'str', 'description': 'Source currency code (e.g., USD)'}, 'to_currency': {'type': 'str', 'description': 'Target currency code (e.g., EUR)'}})
```

This is great! We have successfully extracted information we will be providing to the LLM as a tool definition.

#### Crafting the system prompt.

We will be using gpt-4o-mini as our reasoning engine. It is known that GPT model family performs better when the input prompt is formatted as json. So we will do exactly that. Actually, the system prompt is the most important part of our agent, here is the final one we will be using:

```
{
    "role": "AI Assistant",
    "capabilities": [\
        "Using provided tools to help users when necessary",\
        "Responding directly without tools for questions that don't require tool usage",\
        "Planning efficient tool usage sequences"\
    ],
    "instructions": [\
        "Use tools only when they are necessary for the task",\
        "If a query can be answered directly, respond with a simple message instead of using tools",\
        "When tools are needed, plan their usage efficiently to minimize tool calls"\
    ],
    "tools": [\
        {\
            "name": tool.name,\
            "description": tool.description,\
            "parameters": {\
                name: {\
                    "type": info["type"],\
                    "description": info["description"]\
                }\
                for name, info in tool.parameters.items()\
            }\
        }\
        for tool in self.tools.values()\
    ],
    "response_format": {
        "type": "json",
        "schema": {
            "requires_tools": {
                "type": "boolean",
                "description": "whether tools are needed for this query"
            },
            "direct_response": {
                "type": "string",
                "description": "response when no tools are needed",
                "optional": True
            },
            "thought": {
                "type": "string",
                "description": "reasoning about how to solve the task (when tools are needed)",
                "optional": True
            },
            "plan": {
                "type": "array",
                "items": {"type": "string"},
                "description": "steps to solve the task (when tools are needed)",
                "optional": True
            },
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {
                            "type": "string",
                            "description": "name of the tool"
                        },
                        "args": {
                            "type": "object",
                            "description": "parameters for the tool"
                        }
                    }
                },
                "description": "tools to call in sequence (when tools are needed)",
                "optional": True
            }
        },
        "examples": [\
            {\
                "query": "Convert 100 USD to EUR",\
                "response": {\
                    "requires_tools": True,\
                    "thought": "I need to use the currency conversion tool to convert USD to EUR",\
                    "plan": [\
                        "Use convert_currency tool to convert 100 USD to EUR",\
                        "Return the conversion result"\
                    ],\
                    "tool_calls": [\
                        {\
                            "tool": "convert_currency",\
                            "args": {\
                                "amount": 100,\
                                "from_currency": "USD",\
                                "to_currency": "EUR"\
                            }\
                        }\
                    ]\
                }\
            },\
            {\
                "query": "What's 500 Japanese Yen in British Pounds?",\
                "response": {\
                    "requires_tools": True,\
                    "thought": "I need to convert JPY to GBP using the currency converter",\
                    "plan": [\
                        "Use convert_currency tool to convert 500 JPY to GBP",\
                        "Return the conversion result"\
                    ],\
                    "tool_calls": [\
                        {\
                            "tool": "convert_currency",\
                            "args": {\
                                "amount": 500,\
                                "from_currency": "JPY",\
                                "to_currency": "GBP"\
                            }\
                        }\
                    ]\
                }\
            },\
            {\
                "query": "What currency does Japan use?",\
                "response": {\
                    "requires_tools": False,\
                    "direct_response": "Japan uses the Japanese Yen (JPY) as its official currency. This is common knowledge that doesn't require using the currency conversion tool."\
                }\
            }\
        ]
    }
}
```

A lot to unpack, let’s analyse it step by step:

```
"role": "AI Assistant",
"capabilities": [\
    "Using provided tools to help users when necessary",\
    "Responding directly without tools for questions that don't require tool usage",\
    "Planning efficient tool usage sequences"\
],
"instructions": [\
    "Use tools only when they are necessary for the task",\
    "If a query can be answered directly, respond with a simple message instead of using tools",\
    "When tools are needed, plan their usage efficiently to minimize tool calls"\
]
```

This is where we define the qualities of the Agent, in general we are enforcing the behaviour that tools should be used only when necessary.

```
"tools": [\
    {\
        "name": tool.name,\
        "description": tool.description,\
        "parameters": {\
            name: {\
                "type": info["type"],\
                "description": info["description"]\
            }\
            for name, info in tool.parameters.items()\
        }\
    }\
    for tool in self.tools.values()\
]
```

This is where we unpack the tools into a list. The tool list will be part of Agent class, that is why we loop through self.tools. Remember, each tool is defined by the Dataclass we created in the first part.

```
"response_format": {
    "type": "json",
    "schema": {
        "requires_tools": {
            "type": "boolean",
            "description": "whether tools are needed for this query"
        },
        "direct_response": {
            "type": "string",
            "description": "response when no tools are needed",
            "optional": True
        },
        "thought": {
            "type": "string",
            "description": "reasoning about how to solve the task (when tools are needed)",
            "optional": True
        },
        "plan": {
            "type": "array",
            "items": {"type": "string"},
            "description": "steps to solve the task (when tools are needed)",
            "optional": True
        },
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "description": "name of the tool"
                    },
                    "args": {
                        "type": "object",
                        "description": "parameters for the tool"
                    }
                }
            },
            "description": "tools to call in sequence (when tools are needed)",
            "optional": True
        }
    }
}
```

Above enforces the LLM output schema. We provide strict instructions here:

- requires\_tools: return if tool usage is required.
- direct\_response: if above is false return a direct response.
- thought: description on how the task should be solved.
- plan: steps to solve the task.
- tool\_calls: tool calls in sequence including functions and parameters to be used. Our example only includes one tool, but it does not necessarily have to.

```
"examples": [\
    {\
        "query": "Convert 100 USD to EUR",\
        "response": {\
            "requires_tools": True,\
            "thought": "I need to use the currency conversion tool to convert USD to EUR",\
            "plan": [\
                "Use convert_currency tool to convert 100 USD to EUR",\
                "Return the conversion result"\
            ],\
            "tool_calls": [\
                {\
                    "tool": "convert_currency",\
                    "args": {\
                        "amount": 100,\
                        "from_currency": "USD",\
                        "to_currency": "EUR"\
                    }\
                }\
            ]\
        }\
    },\
    {\
        "query": "What's 500 Japanese Yen in British Pounds?",\
        "response": {\
            "requires_tools": True,\
            "thought": "I need to convert JPY to GBP using the currency converter",\
            "plan": [\
                "Use convert_currency tool to convert 500 JPY to GBP",\
                "Return the conversion result"\
            ],\
            "tool_calls": [\
                {\
                    "tool": "convert_currency",\
                    "args": {\
                        "amount": 500,\
                        "from_currency": "JPY",\
                        "to_currency": "GBP"\
                    }\
                }\
            ]\
        }\
    },\
    {\
        "query": "What currency does Japan use?",\
        "response": {\
            "requires_tools": False,\
            "direct_response": "Japan uses the Japanese Yen (JPY) as its official currency. This is common knowledge that doesn't require using the currency conversion tool."\
        }\
    }\
]
```

Finally, we provide some examples of correct reasoning above.

#### Implementing the Agent Class

The agent class is quite lengthy due to the long system prompt:

```
class Agent:
    def __init__(self):
        """Initialize Agent with empty tool registry."""
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.tools: Dict[str, Tool] = {}

    def add_tool(self, tool: Tool) -> None:
        """Register a new tool with the agent."""
        self.tools[tool.name] = tool

    def get_available_tools(self) -> List[str]:
        """Get list of available tool descriptions."""
        return [f"{tool.name}: {tool.description}" for tool in self.tools.values()]

    def use_tool(self, tool_name: str, **kwargs: Any) -> str:
        """Execute a specific tool with given arguments."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}")

        tool = self.tools[tool_name]
        return tool.func(**kwargs)

    def create_system_prompt(self) -> str:
        """Create the system prompt for the LLM with available tools."""
        tools_json = {
            "role": "AI Assistant",
            "capabilities": [\
                "Using provided tools to help users when necessary",\
                "Responding directly without tools for questions that don't require tool usage",\
                "Planning efficient tool usage sequences"\
            ],
            "instructions": [\
                "Use tools only when they are necessary for the task",\
                "If a query can be answered directly, respond with a simple message instead of using tools",\
                "When tools are needed, plan their usage efficiently to minimize tool calls"\
            ],
            "tools": [\
                {\
                    "name": tool.name,\
                    "description": tool.description,\
                    "parameters": {\
                        name: {\
                            "type": info["type"],\
                            "description": info["description"]\
                        }\
                        for name, info in tool.parameters.items()\
                    }\
                }\
                for tool in self.tools.values()\
            ],
            "response_format": {
                "type": "json",
                "schema": {
                    "requires_tools": {
                        "type": "boolean",
                        "description": "whether tools are needed for this query"
                    },
                    "direct_response": {
                        "type": "string",
                        "description": "response when no tools are needed",
                        "optional": True
                    },
                    "thought": {
                        "type": "string",
                        "description": "reasoning about how to solve the task (when tools are needed)",
                        "optional": True
                    },
                    "plan": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "steps to solve the task (when tools are needed)",
                        "optional": True
                    },
                    "tool_calls": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool": {
                                    "type": "string",
                                    "description": "name of the tool"
                                },
                                "args": {
                                    "type": "object",
                                    "description": "parameters for the tool"
                                }
                            }
                        },
                        "description": "tools to call in sequence (when tools are needed)",
                        "optional": True
                    }
                },
                "examples": [\
                    {\
                        "query": "Convert 100 USD to EUR",\
                        "response": {\
                            "requires_tools": True,\
                            "thought": "I need to use the currency conversion tool to convert USD to EUR",\
                            "plan": [\
                                "Use convert_currency tool to convert 100 USD to EUR",\
                                "Return the conversion result"\
                            ],\
                            "tool_calls": [\
                                {\
                                    "tool": "convert_currency",\
                                    "args": {\
                                        "amount": 100,\
                                        "from_currency": "USD",\
                                        "to_currency": "EUR"\
                                    }\
                                }\
                            ]\
                        }\
                    },\
                    {\
                        "query": "What's 500 Japanese Yen in British Pounds?",\
                        "response": {\
                            "requires_tools": True,\
                            "thought": "I need to convert JPY to GBP using the currency converter",\
                            "plan": [\
                                "Use convert_currency tool to convert 500 JPY to GBP",\
                                "Return the conversion result"\
                            ],\
                            "tool_calls": [\
                                {\
                                    "tool": "convert_currency",\
                                    "args": {\
                                        "amount": 500,\
                                        "from_currency": "JPY",\
                                        "to_currency": "GBP"\
                                    }\
                                }\
                            ]\
                        }\
                    },\
                    {\
                        "query": "What currency does Japan use?",\
                        "response": {\
                            "requires_tools": False,\
                            "direct_response": "Japan uses the Japanese Yen (JPY) as its official currency. This is common knowledge that doesn't require using the currency conversion tool."\
                        }\
                    }\
                ]
            }
        }

        return f"""You are an AI assistant that helps users by providing direct answers or using tools when necessary.
Configuration, instructions, and available tools are provided in JSON format below:

{json.dumps(tools_json, indent=2)}

Always respond with a JSON object following the response_format schema above.
Remember to use tools only when they are actually needed for the task."""

    def plan(self, user_query: str) -> Dict:
        """Use LLM to create a plan for tool usage."""
        messages = [\
            {"role": "system", "content": self.create_system_prompt()},\
            {"role": "user", "content": user_query}\
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )

        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            raise ValueError("Failed to parse LLM response as JSON")

    def execute(self, user_query: str) -> str:
        """Execute the full pipeline: plan and execute tools."""
        try:
            plan = self.plan(user_query)

            if not plan.get("requires_tools", True):
                return plan["direct_response"]

            # Execute each tool in sequence
            results = []
            for tool_call in plan["tool_calls"]:
                tool_name = tool_call["tool"]
                tool_args = tool_call["args"]
                result = self.use_tool(tool_name, **tool_args)
                results.append(result)

            # Combine results
            return f"""Thought: {plan['thought']}
Plan: {'. '.join(plan['plan'])}
Results: {'. '.join(results)}"""

        except Exception as e:
            return f"Error executing plan: {str(e)}"
```

Let’s look into it step by step (skipping the create\_system\_prompt method as we already analysed it in the previous part).

```
def add_tool(self, tool: Tool) -> None:
    """Register a new tool with the agent."""
    self.tools[tool.name] = tool

def get_available_tools(self) -> List[str]:
    """Get list of available tool descriptions."""
    return [f"{tool.name}: {tool.description}" for tool in self.tools.values()]

def use_tool(self, tool_name: str, **kwargs: Any) -> str:
    """Execute a specific tool with given arguments."""
    if tool_name not in self.tools:
        raise ValueError(f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}")

    tool = self.tools[tool_name]
    return tool.func(**kwargs)
```

Above contain methods to manage tools:

- Attaching tools to the agent.
- List attached tools.
- Invoke execution of a tool.

```
def plan(self, user_query: str) -> Dict:
    """Use LLM to create a plan for tool usage."""
    messages = [\
        {"role": "system", "content": self.create_system_prompt()},\
        {"role": "user", "content": user_query}\
    ]

    response = self.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        raise ValueError("Failed to parse LLM response as JSON")
```

The above simply executes the system prompt, we defined the expected output as part of the system prompt. It exactly provides the actions that the LLM planned or a direct answer if the tool calling is not needed.

```
def execute(self, user_query: str) -> str:
    """Execute the full pipeline: plan and execute tools."""
    try:
        plan = self.plan(user_query)

        if not plan.get("requires_tools", True):
            return plan["direct_response"]

        # Execute each tool in sequence
        results = []
        for tool_call in plan["tool_calls"]:
            tool_name = tool_call["tool"]
            tool_args = tool_call["args"]
            result = self.use_tool(tool_name, **tool_args)
            results.append(result)

        # Combine results
        return f"""Thought: {plan['thought']}
Plan: {'. '.join(plan['plan'])}
Results: {'. '.join(results)}"""

    except Exception as e:
        return f"Error executing plan: {str(e)}"
```

The above executes the plan method and acts on it. You might remember that the plan can include multiple sequential tool executions, that is why we are looping through planned tool calls.

#### Running the Agent.

That’s it, we have all of the necessary code to create and use the Agent. in the following code we initialise the agent, attach a convert\_currency tool to it and loop through two user queries. First one should require the tool use while the second not.

```
agent = Agent()
agent.add_tool(convert_currency)

query_list = ["I am traveling to Japan from Serbia, I have 1500 of local currency, how much of Japanese currency will I be able to get?",\
                "How are you doing?"]

for query in query_list:
    print(f"\nQuery: {query}")
    result = agent.execute(query)
    print(result)
```

The output should be similar to:

```
Query: I am traveling to Japan from Serbia, I have 1500 of local currency, how much of Japanese currency will I be able to get?
Thought: I need to convert 1500 Serbian Dinars (RSD) to Japanese Yen (JPY) using the currency conversion tool.
Plan: Use convert_currency tool to convert 1500 RSD to JPY. Return the conversion result
Results: 1500 RSD = 2087.49 JPY

Query: How are you doing?
I'm just a computer program, so I don't have feelings, but I'm here and ready to help you!
```

As expected! First query uses the tool, while the second does not.

#### That’s it for today, we’ve learned:

- How to wrap python functions to be provided as tools to the Agent.
- How to craft a system prompt that uses the tool definitions in planning the execution.
- How to implement the agent that executes on the plan.

</details>

<details>
<summary>efficient-tool-use-with-chain-of-abstraction-reasoning</summary>

# Efficient Tool Use with Chain-of-Abstraction Reasoning

Silin Gao1,2∗, Jane Dwivedi-Yu2, Ping Yu2, Xiaoqing Ellen Tan2,

Ramakanth Pasunuru2, Olga Golovneva2, Koustuv Sinha2

Asli Celikyilmaz2, Antoine Bosselut1†, Tianlu Wang2†

1EPFL, 2FAIR @ Meta

1{silin.gao,antoine.bosselut}@epfl.ch

2{silingao,janeyu,pingyu,ellenxtan}@meta.com

2{rpasunuru,olggol,koustuvs,aslic,tianluwang}@meta.com

###### Abstract

To achieve faithful reasoning that aligns with human expectations, large language models (LLMs) need to ground their reasoning to real-world knowledge (e.g., web facts, math and physical rules).
Tools help LLMs access this external knowledge, but there remains challenges for fine-tuning LLM agents (e.g., Toolformer) to invoke tools in multi-step reasoning problems, where inter-connected tool calls require holistic and efficient tool usage planning.

In this work, we propose a new method for LLMs to better leverage tools in multi-step reasoning.
Our method, Chain-of-Abstraction (CoA), trains LLMs to first decode reasoning chains with abstract placeholders, and then call domain tools to reify each reasoning chain by filling in specific knowledge.
This planning with abstract chains enables LLMs to learn more general reasoning strategies, which are robust to shifts of domain knowledge (e.g., math results) relevant to different reasoning questions.
It also allows LLMs to perform decoding and calling of external tools in parallel, which avoids the inference delay caused by waiting for tool responses.
In mathematical reasoning and Wiki QA domains, we show that our method consistently outperforms previous chain-of-thought and tool-augmented baselines on both in-distribution and out-of-distribution test sets, with an average ∼6% absolute QA accuracy improvement.
LLM agents trained with our method also show more efficient tool use, with inference speed being on average ∼1.4× faster than baseline tool-augmented LLMs.

https://arxiv.org/html/x1.png
Figure 1: Overview of chain-of-abstraction reasoning with tools. Given a domain question (green scroll), a LLM is fine-tuned to first generate an abstract multi-step reasoning chain (blue bubble), and then call external tools to reify the chain with domain-specific knowledge (orange label). The final answer (yellow bubble) is obtained based on the reified chain of reasoning.

## 1 Introduction

Recent large language models (LLMs; Touvron et al., [2023b](https://arxiv.org/html/2401.17464v3#bib.bib39 ""); Anil et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib1 ""); OpenAI, [2023](https://arxiv.org/html/2401.17464v3#bib.bib29 "")), have made progress at interpreting and executing instructions (Wei et al., [2021](https://arxiv.org/html/2401.17464v3#bib.bib41 ""); Chung et al., [2022](https://arxiv.org/html/2401.17464v3#bib.bib8 "")), but still make errors when recalling and composing world knowledge for their responses, e.g., making unfactual statements (Maynez et al., [2020](https://arxiv.org/html/2401.17464v3#bib.bib27 ""); Ji et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib19 "")), incorrect calculations (Patel et al., [2021](https://arxiv.org/html/2401.17464v3#bib.bib31 "")), etc. Using auxiliary tools (e.g., a search engine to provide credible facts, a calculator for accurate math operations, etc.) at inference time can mitigate some of these errors, motivating tool-augmented language models that integrate external API calls into their output generations (Parisi et al., [2022](https://arxiv.org/html/2401.17464v3#bib.bib30 ""); Schick et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib36 ""); Hao et al., [2023b](https://arxiv.org/html/2401.17464v3#bib.bib14 "")).

However, we show that current tool-augmented LLMs, e.g., Toolformer (Schick et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib36 "")), struggle to reliably and efficiently leverage tools in multi-step reasoning.
In particular, tool calls in multi-step reasoning tasks are often interleaved (i.e., the response of an API call is often part of the query of a subsequent call; as shown in Figure 1).
Without explicitly modeling these interconnections in reasoning chains, LLMs do not learn effective planning for tool use, which leads to less accurate reasoning with tools.
Meanwhile, interleaving text generation with API calls also introduces inefficient inference “waiting times,” where the model must wait for the response from the API call before resuming the decoding process. This inefficiency becomes more significant in multi-step reasoning scenarios, when multiple rounds of API calls are typically required for each reasoning process.

In this work, we propose Chain-of-Abstraction (CoA) reasoning, a robust and efficient method for LLMs to perform multi-step reasoning with tools.
As shown in Figure 1, LLMs are fine-tuned with a goal of making reasoning chains with abstract placeholders.
The placeholders do not affect LLMs’ reasoning flow, and are subsequently infilled with specific knowledge retrieved from specialized tools, to ground the final answer generations.
Planning abstract chain of reasoning encourages LLMs to inter-connect multiple tool calls and adopt more feasible reasoning strategies, which are robust to the variation of domain knowledge involved in each reasoning process, e.g., specific calculation results.
Unlike previous methods where LLM decoding and API calls are executed in an interleaved manner, our method leverages tools to infill knowledge once after the whole chain of reasoning is generated.
This enables more efficient decoding across multiple examples (e.g., as in a stream) because CoA traces for subsequent examples can be decoded while tool calls are made for the preceding ones, amortizing overall inference time.
We develop a simple pipeline to build fine-tuning data for models to learn CoA, where we first prompt LLMs to re-write existing responses to instructions as abstract chains, and then use domain tools to check the validity of re-writing, as shown in Figure 2.

After training LLMs to learn CoA reasoning, we evaluate the finetuned models on two representative multi-step reasoning domains, including mathematical reasoning (Cobbe et al., [2021](https://arxiv.org/html/2401.17464v3#bib.bib9 ""); Miao et al., [2020](https://arxiv.org/html/2401.17464v3#bib.bib28 ""); Patel et al., [2021](https://arxiv.org/html/2401.17464v3#bib.bib31 ""); Koncel-Kedziorski et al., [2016](https://arxiv.org/html/2401.17464v3#bib.bib22 "")), and Wikipedia (Wiki) QA (Yang et al., [2018](https://arxiv.org/html/2401.17464v3#bib.bib45 ""); Berant et al., [2013](https://arxiv.org/html/2401.17464v3#bib.bib3 ""); Kwiatkowski et al., [2019](https://arxiv.org/html/2401.17464v3#bib.bib23 ""); Joshi et al., [2017](https://arxiv.org/html/2401.17464v3#bib.bib21 "")) that involves reasoning on factual descriptive knowledge.
We show that our method boosts LLMs’ performances, with average ∼7.5% and 4.5% absolute accuracy improvements on math and Wiki QA, respectively.
These improvements are consistent across both in-distribution
and (zero-shot) out-of-distribution test sets, and are especially pronounced on questions that require complex chain-of-thought reasoning.
Meanwhile, our method also uses tools more efficiently than previous augmentation methods, with average ∼1.47× and 1.33× faster inference speeds on math and Wiki QA tasks, respectively.
Finally, extensive human evaluation demonstrates that our method guides LLMs to learn more accurate reasoning, which leads to ∼8% fewer reasoning errors.

## 2 Related Work

#### Tool-Augmented LLMs

There is growing interest in augmenting LLMs using external tools.
Considerable work has tried to adapt LLMs as tool-using reasoners through in-context learning, demonstrating promising performance improvements in various applications, e.g., math problem solving (Gao et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib10 ""); Chen et al., [2022](https://arxiv.org/html/2401.17464v3#bib.bib7 "")), biomedical question answering (Jin et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib20 "")) and self-critiquing (Gou et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib12 "")).
Nevertheless, guiding LLMs to effectively use tools using in-context demonstrations is challenging, which requires elaborate task-specific prompt engineering and is restricted by the model’s instruction following ability (Jacovi et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib18 "")). Noticing the limitations of in-context learning, several works teach LLMs to learn the usage of tools by fine-tuning (Parisi et al., [2022](https://arxiv.org/html/2401.17464v3#bib.bib30 ""); Schick et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib36 ""); Hao et al., [2023b](https://arxiv.org/html/2401.17464v3#bib.bib14 "")), which more robustly improves LLMs’ performance.
However, all above approaches adopt sequential interactions with tools throughout reasoning, slowing the inference speed as a function of the latency of the tool (or API) and the number of API calls that are made.

Some other prior works focus on using LLMs for multi-step reasoning with other modules.
In particular, ReAct (Yao et al., [2023b](https://arxiv.org/html/2401.17464v3#bib.bib47 "")) and FireAct (Chen et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib5 "")) integrate LLMs with tools into a closed loop of thought, action and observation steps.
This verbose reasoning loop slows down the LLM decoding, and still incorporates tools via sequential interactions, resulting in inefficient inference.
Another line of work, Program of Thoughts (Chen et al., [2022](https://arxiv.org/html/2401.17464v3#bib.bib7 "")), DECLARATIVE (He-Yueya et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib15 "")) and PAL (Gao et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib10 "")) prompt LLMs to generate program-based reasoning and interact with code executors, which however heavily rely on closed source coding models, i.e., Codex (Chen et al., [2021](https://arxiv.org/html/2401.17464v3#bib.bib6 "")), and are restricted to procedural arithmetic reasoning.
Building on these works, CoA proposes a framework to convert natural language reasoning traces into abstract representations, and uses the abstract reasoning traces as fine-tuning data to improve tool-augmented LLMs.
CoA also accelerates tool-augmented reasoning, by holistically planning the CoA traces and calling tools only once at inference time.

#### Tool Usage Planning

Several previous works research tool usage planning in LLMs.
Specifically, HuggingGPT (Shen et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib37 "")), Chameleon (Lu et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib26 "")), OpenAGI (Ge et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib11 "")) and MetaTool (Huang et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib17 "")) focus on planning the high-level sequence of using multiple tools to address multi-domain mixed tasks.
Similarly, LATM (Cai et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib4 "")), ML-BENCH (Liu et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib24 "")) and Gorilla (Patil et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib32 "")) aim at planning program-level integration of multiple APIs for designing scripts of procedural tasks, e.g., a script for training a model described by a GitHub repository.
ToolChain\* (Zhuang et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib48 "")) combines the planning of tool usage with tree-search-based reasoning (Yao et al., [2023a](https://arxiv.org/html/2401.17464v3#bib.bib46 ""); Hao et al., [2023a](https://arxiv.org/html/2401.17464v3#bib.bib13 "")), which is especially useful for procedural tasks (Xu et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib44 ""); Cobbe et al., [2021](https://arxiv.org/html/2401.17464v3#bib.bib9 "")).
Different from above work, we focus on the planning of general chain-of-thought (Wei et al., [2022](https://arxiv.org/html/2401.17464v3#bib.bib42 "")) reasoning with awareness of domain specialized tools.

## 3 Method

https://arxiv.org/html/x2.png
Figure 2: Illustration of gold data re-writing for fine-tuning data construction. Given a pair of domain question (green scroll) and gold answer (yellow scroll), an LLM is prompted to re-write the gold answer as a reasoning chain with abstract variables (purple bubble). Then, domain specialized tools validate the correctness of the re-writing by checking whether the abstract chain can be reified to get the final answer (orange label).

#### Chain-of-Abstraction (CoA) Reasoning

Our method decouples the general reasoning of LLMs from domain-specific knowledge obtained from external tools.
Figure 1 shows an overview of our method.
In particular, we first fine-tune LLMs to generate reasoning chains with abstract placeholders, e.g., y₁, y₂ and y₃, as shown in Figure 1.
In the second stage, we reify each reasoning chain by replacing placeholders with domain-specific knowledge obtained from external tools, e.g., calculation results from a calculator, relevant articles retrieved from web search engine, etc.
Finally, the question is answered based on the reified reasoning chain.

Note that since the LLMs are trained to generate abstract chain of reasoning instead of regular chain-of-thought (CoT) reasoning with explicit values, this enables LLMs to focus on learning general and holistic reasoning strategies without needing to generate instance-specific knowledge for the model’s parameters.
Moreover, decoupling general reasoning and domain-specific knowledge enables LLM decoding to proceed and switch between different samples in parallel with API calling (via a pipeline), i.e., LLM can start generating the next abstract chain while the tool fills the current chain, which speeds up the overall inference process.

#### Fine-tuning Data Construction

To construct chain-of-abstraction (CoA) data for fine-tuning LLMs, we collect question answering (QA) samples from existing open-source QA datasets (Cobbe et al., [2021](https://arxiv.org/html/2401.17464v3#bib.bib9 ""); Miao et al., [2020](https://arxiv.org/html/2401.17464v3#bib.bib28 ""); Yang et al., [2018](https://arxiv.org/html/2401.17464v3#bib.bib45 "")), and prompt LLaMa-70B (Touvron et al., [2023a](https://arxiv.org/html/2401.17464v3#bib.bib38 "")) to re-write the answer of each sampled question, as shown in Figure 2.
Specifically, we prompt LLaMa-70B to label the spans in gold answers that correspond to knowledge operations (e.g., math derivations, statements based on Wikipedia references) and then to re-write the sentences with labeled spans as fillable CoA traces, where the operation results are replaced with abstract placeholders.
For example, the two derivations in the example in Figure 2 are re-written as “[20+35=y₁]” and “[90−y₁=y₂]”, respectively.

Note that an intermediate knowledge operation result may appear multiple times in an answer, e.g., in Figure 2, the first equation’s result is used in the second equation.
We prompt LLaMa-70B to replace all occurrences of the same intermediate result with the same placeholder, thereby explicitly connecting the multiple reasoning steps.
To ensure that the re-written data is accurate, we use domain-specialized tools to verify the correctness of each CoA reasoning trace.
Specifically, we use the tools to execute the labeled operations in each CoA, and only keep questions whose CoA can be infilled with valid results by the tools.

## 4 Experimental Settings

We conduct our experiments on two representative domains: mathematical reasoning and Wikipedia (Wiki) QA, which involves commonsense and logical reasoning on factual descriptive knowledge.

### 4.1 Mathematical Reasoning

Given a math question, the QA system needs to generate a natural language solution to the problem with step-by-step arithmetic derivations (as demonstrated in the left column of Figure 1).
We assume that the derivations involved in the solution are the specialized knowledge operations required in this domain, which are labeled in square brackets with derivation results being replaced by abstract placeholders, e.g., “[20+35=y₁]".

#### Datasets

We construct most of our fine-tuning CoA data by re-writing the GSM8K (Cobbe et al., [2021](https://arxiv.org/html/2401.17464v3#bib.bib9 "")) training set, which contains 7473 linguistically diverse grade school math problems.
As GSM8K dataset focuses on multi-step reasoning, it lacks coverage of single-step arithmetic problems, so we also re-write an additional set of 691 single-step math problems from the ASDiv (Miao et al., [2020](https://arxiv.org/html/2401.17464v3#bib.bib28 "")) dataset.
Across these re-written datasets, we find that ∼76.6% of the CoA reasoning traces generated by LLaMa-70B are verified by our equation solver (described below).
Table 1 shows the reasoning step distribution (i.e., number of derivations) of our constructed fine-tuning data.

| Source | Reasoning Step |
| --- | --- |
| 1 | 2 | 3 | 4 | 5 | >5 | All |
| GSM8K | 8 | 1540 | 1648 | 1164 | 666 | 553 | 5579 |
| ASDiv | 677 | 0 | 0 | 0 | 0 | 0 | 677 |

Table 1: Reasoning step distribution of correctly re-written reasoning chains in math domain.

For an in-distribution evaluation, we test models on GSM8K and ASDiv, containing 1319 and 2305 testing problems.
To further test the models’ generalization ability, we also conduct zero-shot evaluation on other representative math datasets, including SVAMP (Patel et al., [2021](https://arxiv.org/html/2401.17464v3#bib.bib31 "")) and MAWPS (Koncel-Kedziorski et al., [2016](https://arxiv.org/html/2401.17464v3#bib.bib22 "")), which contain 1000 and 2065 testing samples, respectively.

#### Domain Tool

We use an equation solver to perform the arithmetic derivations required in the math domain.
Our equation solver first extracts the derivations labeled in the CoA reasoning, e.g., “[20+35=y₁]” and “[90−y₁=y₂]”, and combines all derivations into a system of equations.
Then the system of equations is solved by the SymPy toolkit, to get the true value of each variable (i.e., the value of the abstract placeholder).
Finally, our equation solver returns the reified chain of reasoning by replacing all the variables with their solved true values (including the final answer).

|     |     |
| --- | --- |
| Question | The director of the romantic comedy “Big Stone Gap” is based in |
| what New York city? |
| Answer | Greenwich Village |
| Wikipedia | Big Stone Gap (film) > Big Stone Gap is a 2014 American romantic |
| References | comedy film directed by Adriana Trigiani. |
| Adriana Trigiani > Adriana Trigiani is an Italian American film |
| director based in Greenwich Village. |
| CoA Trace | Find the [director of romantic comedy “Big Stone Gap” -Wiki-> y1]. |
| The name of this film’s director is [y1 -NER(person)-> y2]. |
| Then determine [y2 in what New York city -Wiki-> y3]. |

Table 2: Example of CoA fine-tuning data construction in Wiki QA domain.

### 4.2 Wikipedia QA

Given a question based on Wikipedia knowledge, the model needs to first identify Wikipedia articles as references related to the question, and then reason on key knowledge in the reference articles to answer the question (as shown in the right column of Figure 1).
We assume that the specialized knowledge operation in this domain is the retrieval of relevant Wikipedia articles and important named-entities, which are re-written as Wikipedia searching (WikiSearch) and named-entity recognition (NER) queries.
Table 2 shows an example of a re-written CoA trace for Wiki QA.

#### Datasets

We use the HotpotQA (Yang et al., [2018](https://arxiv.org/html/2401.17464v3#bib.bib45 "")) dataset to construct our fine-tuning CoA data in the Wiki QA domain.
HotpotQA contains 113K multi-hop QA examples, each labeled with two Wikipedia articles that provide supporting knowledge.
Among the 90447 training QA pairs, we identify 72991 as Bridge QA pairs, where an intermediate entity must be identified to link the answer to the question, as shown in Table 2.
The remaining 17456 are Comparison QA pairs, where the attributes of two entities are compared, e.g., “Are Randal Kleiser and Kyle Schickner of the same nationality?”.
We prompt LLaMa-70B to re-write these training QAs into CoAs with WikiSearch and NER queries, and verify each CoA with our domain tools (described below), by checking whether all the articles returned by the WikiSearch queries match one of the titles in the gold articles.
Finally, 8956 Bridge QAs and 5405 Comparison QAs are used as fine-tuning data, whose re-written CoAs pass the verification.
For Wiki QA, we note that besides training a LLM to produce CoA data using WikiSearch, we also fine-tune a second LLM to learn to generate the final gold answer based on a correctly reified CoA reasoning trace.

We evaluate models on the HotpotQA development set, which contains 5918 Bridge QA pairs and 1487 Comparison QA pairs. Similar to the mathematical reasoning domain, we also conduct zero-shot evaluation on other open-domain QA datasets: WebQuestions (WQ; Berant et al., [2013](https://arxiv.org/html/2401.17464v3#bib.bib3 "")), NaturalQuestions (NQ; Kwiatkowski et al., [2019](https://arxiv.org/html/2401.17464v3#bib.bib23 "")) and TriviaQA (Joshi et al., [2017](https://arxiv.org/html/2401.17464v3#bib.bib21 "")), which contain 2032, 3610 and 17944 test questions, respectively.

#### Domain Tools

The specialized tools required for Wiki QA include a Wikipedia search engine to retrieve reference articles, and a NER toolkit to extract entities that bridge multi-step searching queries.
We follow Toolformer (Schick et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib36 "")) and implement a Wikipedia search engine as a BM25 retriever (Robertson et al., [1995](https://arxiv.org/html/2401.17464v3#bib.bib35 ""); Baeza-Yates et al., [1999](https://arxiv.org/html/2401.17464v3#bib.bib2 "")) that indexes the Wikipedia dump from the KILT benchmark (Petroni et al., [2021](https://arxiv.org/html/2401.17464v3#bib.bib33 "")).
We use the BM25 retriever to search the top-10 articles relevant to the input query, and then re-rank the articles based on their Sentence-BERT (Reimers and Gurevych, [2019](https://arxiv.org/html/2401.17464v3#bib.bib34 "")) embedding cosine similarity with the question.
After re-ranking, the top-1 article is selected to be the final search result.

We use SpaCy as the NER toolkit to extract named entities.
To simplify NER, we aggregate the numerous SpaCy NER types into 6 general classes, as shown in Table 3.
If multiple named entities are recognized, we input each recognized entity to the subsequent WikiSearch query, and select the entity whose subsequent search result has the highest Sentence-BERT embedding cosine similarity with the question.

| General | SpaCy NER Types included in each General Class |
| Class |
| person | PERSON |
| group | NORP, ORG, LANGUAGE |
| location | GPE, FAC, LOC |
| culture | EVENT, WORK_OF_ART, LAW, PRODUCT |
| date | DATE, TIME |
| numeral | CARDINAL, PERCENT, MONEY, QUANTITY, ORDINAL |

Table 3: Aggregation of SpaCy NER types.

### 4.3 Baselines

We apply our CoA reasoning method to both 7B and 70B LLaMa models, and test various model versions including the first version of LLaMa (Touvron et al., [2023a](https://arxiv.org/html/2401.17464v3#bib.bib38 "")) and the more advanced LLaMa-2 and LLaMa-2-Chat (Touvron et al., [2023b](https://arxiv.org/html/2401.17464v3#bib.bib39 "")).
We compare our method to several baselines, including: a) few-shot prompting using 8 randomly sampled QA exemplars from the original (i.e., not re-written) chain-of-thought data (CoT-FSP), b) fine-tuning with original chain-of-thought data (CoT-FT), and c) ToolformerSchick et al. ( [2023](https://arxiv.org/html/2401.17464v3#bib.bib36 "")) which fine-tunes LLMs on CCNet (Wenzek et al., [2020](https://arxiv.org/html/2401.17464v3#bib.bib43 "")) texts augmented with API calls.
For evaluation on Wiki QA, we also compared our method with FireAct(Chen et al., [2023](https://arxiv.org/html/2401.17464v3#bib.bib5 "")), which fine-tunes LLMs on HotpotQA ReAct (Yao et al., [2023b](https://arxiv.org/html/2401.17464v3#bib.bib47 "")) trajectories distilled from GPT-4 (OpenAI, [2023](https://arxiv.org/html/2401.17464v3#bib.bib29 "")).

## 5 Results and Analysis

### 5.1 Mathematical Reasoning

Table 4 shows the evaluation results for the LLaMa-2 and LLaMa-2-Chat models.
On the GSM8K and ASDiv datasets, our CoA method outperforms the few-shot baseline CoT-FSP and the regular fine-tuning baseline CoT-FT, demonstrating that CoA fine-tuning with tool augmentation is more effective in adapting LLMs to multi-step reasoning tasks.
Similarly, when evaluated on out-of-distribution datasets, SVAMP and MAWPS, CoA also consistently outperforms the baselines.
Interestingly, for these out-of-distribution datasets, CoT-FT lags further behind CoA, particularly for 7B models, showing that CoA reasoning yields more distributionally robust reasoning performance.

Our CoA method also surpasses the tool-augmented baseline Toolformer, which implies that planning the abstract variables in CoA can improve the accuracy of reasoning with tools.
However, as Toolformer is not originally trained with in-domain fine-tuning data, we also fine-tune a new version of Toolformer with the chain-of-thought data from GSM8K and ASDiv, denoted as Toolformer - Math in Table 4.
We also observe that CoA performs better than Toolformer - Math, confirming that the introduction of abstract variables enables more robust tool use compared to direct integration of API calls within chain-of-thought reasoning.

#### Ablation Study

We verify that the robust generalization performance of our CoA method does not merely benefit from using additional tools, by fine-tuning another LLM to solve the equation (from the same model backbone), rather than calling the equation solver, denoted as CoA (no Tool) in Table 4.
We find that CoA (no Tool) performs consistently worse than CoA across all datasets, confirming that using specialized tools enables LLM agents to conduct more precise operations, rather than directly solving the same operations.
However, CoA (no Tool) still outperforms all baseline methods on zero-shot generalization to SVAMP and MAWPS datasets, implying that learning abstract reasoning chains also contributes to better robustness of CoA, perhaps due to better planning of multiple reasoning steps indexed by abstract variables.

#### Reasoning Steps

Our findings suggest that the benefits of chain-of-abstraction reasoning are most pronounced when problems require long reasoning chains to be solved. Figure 3 shows the stratified performance of three models on GSM8K QA, relative to the number of reasoning steps in the predicted and gold reasoning chains.
Compared to the few-shot CoT-FSP, CoA produces reasoning chains that more often match the length of the gold reasoning chains, as reflected by the heat-map statistics (left column) being more aggregated around the diagonal (comparable to CoT-FT).
At the same time, we observe that models achieve better QA accuracy when the number of reasoning steps in their generated answers are aligned with the gold references (i.e., the diagonal of heat-maps in right column).
Above results show that fine-tuned models are better at learning to produce reasoning chains that match the true reasoning chain for the problem.

https://arxiv.org/html/x3.png
Figure 3: GSM8K evaluation results on LLaMa-2-Chat-7B w.r.t. the number of reasoning steps in the predicted and gold reasoning chain. (Left) The number of test examples that belong to each stratum. (Right) The corresponding model accuracy (%) for those examples. Non-diagonal cells with fewer than 15 examples are ignored.

| Method | Error Rate |
| --- | --- |
| Arithmetic | Reasoning |
| CoT-FSP | 17.3 | 70.3 |
| CoT-FT | 25.2 | 67.8 |
| CoA | 0.0 | 60.4 |

Table 5: Human evaluation results of arithmetic and reasoning error rates on 200 GSM8K test samples. Models developed based on LLaMa-2-Chat-7B are presented.

https://arxiv.org/html/x4.png
Figure 4: Wall-clock inference time on GSM8K (seeded with LLaMa-2-Chat-7B). Average time of answering a question is measured (in seconds) w.r.t. the number of gold reasoning steps required for the question.

| Method | Accuracy |
| --- | --- |
| CoT-FSP | 27.90 |
| CoT-FT | 39.12 |
| Toolformer | 24.56 |
| Toolformer - Math | 35.25 |
| CoA | 40.79 |

Table 6:
Evaluation results on GSM8K with self-consistency decoding (seeded with LLaMa-2-Chat-7B). Each model uses majority voting to aggregate the answers of 16 sampled reasoning chains

| Model | Method | Use | GSM8K | ASDiv | SVAMP | MAWPS |
| Tool | AddSub | SingleEQ | SingleOp | MultiArith | All |
| LLaMa-2 | CoT-FSP | ✗ | 16.38 | 47.85 | 38.40 | 52.41 | 63.39 | 82.03 | 43.33 | 60.53 |
| -7B | CoT-FT | 35.33 | 57.18 | 48.20 | 66.08 | 74.41 | 85.23 | 65.00 | 73.03 |
| Toolformer | ✓ | 17.59 | 48.55 | 37.10 | 47.34 | 58.46 | 79.54 | 50.67 | 59.81 |
| CoA | 37.83∗ | 57.61 | 51.70∗ | 72.15∗ | 82.48∗ | 86.48∗ | 73.17∗ | 78.89∗ |
| LLaMa-2 | CoT-FSP | ✗ | 24.03 | 54.14 | 51.30 | 71.90 | 72.44 | 85.41 | 74.00 | 76.32 |
| -Chat-7B | CoT-FT | 35.41 | 59.00 | 46.90 | 58.23 | 72.24 | 85.41 | 73.00 | 73.37 |
| CoA (no Tool) | 35.03 | 58.79 | 51.50 | 68.10 | 74.21 | 86.48 | 77.67 | 77.38 |
| Toolformer | ✓ | 23.65 | 50.85 | 48.80 | 61.01 | 69.09 | 81.85 | 68.50 | 70.85 |
| Toolformer - Math | 36.01 | 59.18 | 47.60 | 58.99 | 72.44 | 85.94 | 75.50 | 74.43 |
| CoA | 38.29∗ | 59.57 | 54.20∗ | 72.41 | 81.89∗ | 88.26∗ | 83.00∗ | 82.13∗ |
| LLaMa-2 | CoT-FSP | ✗ | 56.18 | 65.94 | 70.60 | 86.08 | 89.17 | 92.88 | 84.50 | 88.23 |
| -Chat-70B | CoT-FT | 60.50 | 70.24 | 70.40 | 81.52 | 87.60 | 92.35 | 89.17 | 88.18 |
| Toolformer | ✓ | 52.54 | 69.07 | 73.60 | 86.84 | 89.76 | 91.46 | 81.50 | 87.26 |
| Toolformer - Math | 61.03 | 70.59 | 73.20 | 85.57 | 91.34 | 91.99 | 92.00 | 90.60 |
| CoA | 62.32∗ | 71.89∗ | 73.40 | 86.33 | 94.49∗ | 93.06 | 92.33 | 91.91∗ |

Table 4: Evaluation results on LLaMa-2 and LLaMa-2-Chat for mathematical reasoning. “All” denotes the averaged results on four MAWPS portions. Exact match rate to the final gold answer (i.e., accuracy) is reported.
For each base model, the best and second-best results are bolded and underlined, respectively. The best results labeled with ∗ are significantly better than their corresponding second-best results, with the significant test p-value <0.05.

Interestingly, we find that CoA, compared to CoT-FT, achieves higher performance especially on questions that require more reasoning steps.
In the right column of Figure 3, CoA’s improvement over CoT-FT is more pronounced on questions with more than 3 steps in the gold reasoning chain (highlighted with red squares).
This indicates that the model trained with CoA has more robust long chain-of-thought reasoning capability, which is learned from planning with abstractions.

#### Human Evaluation

To more comprehensively verify that CoA improves both knowledge operation (i.e., arithmetic by using tools) and reasoning accuracy, we conduct a human evaluation on different model answers to 200 randomly sampled GSM8K test questions.
Specifically, given a GSM8K question and a model’s answer to the question, we ask human workers to judge whether the answer contains any arithmetic errors (e.g., wrong calculations, invalid equations) or reasoning errors unrelated to math derivations (e.g., misunderstanding of the question, improper strategy for solving the question), and report how often the model makes these two kinds of errors.
In Table 5, we find that CoA effectively reduces arithmetic errors to zero, due to the use of equation solver to perform accurate calculations.
More importantly, our method also makes fewer reasoning errors compared to the baselines, verifying that CoA fine-tuning guides the model to learn more accurate reasoning through the holistic planning of abstract reasoning chains.
By contrast, ordinary fine-tuning (i.e., CoT-FT) produces a more limited reasoning improvement compared to the few-shot CoT-FSP, while also failing to suppress arithmetic errors.

#### Inference Efficiency

Importantly, we find that the performance benefits of CoA reasoning do not come with increased computational costs.
In Figure 4, we show the average time (seconds) that CoA and baseline agents (seeded with LLaMa-2-Chat-7B) needs to answer a question w.r.t. required gold reasoning steps.
Compared to the CoT baselines, CoA requires less time than the few-shot baseline CoT-FSP, whose generation needs to be conditioned on additional examples.
However, CoA is slightly less inference-efficient compared to CoT-FT, likely due to the decoding of additional tokens (e.g., “[" and "]”) for the abstract statements.

Compared to Toolformer, CoA has a lower and flatter inference time curve, indicating better scaling as the number of reasoning steps increases.
This difference arises because CoA decouples the generation of (abstract) reasoning chains from the retrieval of knowledge (i.e., tool use), allowing full reasoning chains to be decoded before any tool is called.
This procedure amortizes inference costs in two ways.
First, tool calls are made after the CoA trace has been decoded, enabling parallel tool calls for the same trace (e.g., using an equation solver once rather than multiple calls to a calculator), and avoiding the time delay caused by waiting for external API responses. Consequently, the model fine-tuned with CoA is more efficient at multi-step reasoning, especially when the number of reasoning steps (i.e., tool calls) increases.
Second, across multiple examples, the model can generate the CoA trace of the next example while tool calls are made for the preceding one, parallelizing CoA decoding and tools calls across examples.

#### Self-Consistency Decoding

Besides of greedy decoding, we also test more advanced inference strategy, i.e., self-consistency (Wang et al., [2022](https://arxiv.org/html/2401.17464v3#bib.bib40 "")) decoding, on our CoA reasoning method.
We test all methods on the GSM8K dataset seeded with LLaMa-2-Chat-7B.
Each method samples 16 reasoning chains and uses majority voting to aggregate the 16 answers derived by the reasoning chains, to get the final answer.
For the hyperparameters of sampling, we set the temperature, top-k and top-p as 1.0, 40 and 0.5, respectively.
Table 6 shows our evaluation results.
We find that our CoA method consistently outperforms all baseline methods when shifting from greedy decoding to self-consistency decoding.
This shows that our method also has better potential to be generalized to different LLM decoding schemes.

### 5.2 Wiki QA

Table 7 shows our Wiki QA results using LLaMa-2-Chat models.
Similar to mathematical reasoning, we fine-tune a new version of Toolformer with in-domain chain-of-thought data from HotpotQA, denoted as Toolformer - Wiki.
On HotpotQA, CoA achieves higher exact match rates with the gold reference compared to the few-shot or fine-tuning baselines.
In particular, CoA outperforms all baselines on the more challenging bridge-type QAs, where two steps of reasoning over Wikipedia knowledge are consecutively entangled, i.e., cannot be performed independently in parallel as in comparison-type QAs.
Compared to FireAct fine-tuning, CoA also achieves better performance on both bridge and comparison QAs, without requiring data distilled from closed source GPT-4.

As with mathematical reasoning, CoA agents also perform more efficient inference than Toolformer and FireAct agents when answering HotpotQA questions.
We also find that CoA is more efficient (Time column) than both CoT-FSP and CoT-FT, as CoA does not require few-shot examples as additional inputs and does not need to generate long Wiki articles, which are instead provided by the search engine.
Finally, CoA improves over the baseline methods in zero-shot generalization experiments on other Wiki QA datasets, outperforming all baselines on NaturalQuestions and TriviaQA, and matching the best baselines on WebQuestions.

| Model | Method | Use | HotpotQA | WQ | NQ | TriviaQA |
| Tool | Bridge | Comparison | Both | Time |
| LLaMa-2 | CoT-FSP | ✗ | 11.69 | 45.46 | 18.47 | 2.074 | 34.65 | 30.91 | 53.48 |
| -Chat-7B | CoT-FT | 14.24 | 56.69 | 22.77 | 1.937 | 33.51 | 25.40 | 51.05 |
| Toolformer | ✓ | 12.99 | 44.59 | 20.00 | 2.350 | 36.22 | 30.22 | 54.15 |
| Toolformer - Wiki | 15.68 | 56.42 | 23.86 | 2.301 | 36.61 | 32.96 | 55.08 |
| FireAct | 19.18 | 54.14 | 26.20 | 2.706 | 36.02 | 35.87 | 52.96 |
| CoA | 21.00∗ | 56.96 | 28.22∗ | 1.896 | 35.97 | 38.67∗ | 57.90∗ |
| LLaMa-2 | CoT-FSP | ✗ | 21.39 | 56.62 | 28.47 | 6.668 | 34.89 | 37.42 | 63.61 |
| -Chat-70B | CoT-FT | 23.84 | 63.95 | 31.90 | 6.401 | 34.15 | 39.75 | 62.28 |
| Toolformer | ✓ | 22.24 | 56.09 | 29.04 | 6.888 | 37.16 | 40.42 | 64.31 |
| Toolformer - Wiki | 26.38 | 63.82 | 33.90 | 6.855 | 37.70 | 41.25 | 66.64 |
| CoA | 27.61∗ | 64.09 | 34.94∗ | 6.369 | 36.37 | 43.57∗ | 69.08∗ |

Table 7: Wiki QA evaluation results on LLaMa-2-Chat-based models. “Both” denotes the overall evaluation results on both bridge and comparison portions of HotpotQA. “Time” denotes the average seconds that each agent needs to answer a question in HotpotQA. Exact match rate to the final gold answer (i.e., accuracy) is reported.
For each base model, the best and second-best results are bolded and underlined, respectively. The best results labeled with ∗ are significantly better than their corresponding second-best results, with the significant test p-value <0.05.

## 6 Conclusion

In this work, we propose to decouple the general reasoning of LLM agents from specialized knowledge obtained via external tools.
Our method, chain-of-abstraction (CoA), encourages LLMs to learn the planning of abstract multi-step reasoning, which are more robust to out-of-distribution knowledge shifts.
CoA also achieves a more efficient pipeline for tool usage that significantly improves the speed of tool-augmented multi-step reasoning.
The simple, yet effective, implementations of our method on two diverse tasks (i.e., math reasoning and open-domain QA) demonstrate its potential for being adapted to new reasoning scenarios.

## Limitations

We acknowledge a few limitations in our work.
First, datasets used for testing our method cannot have exhaustive coverage of all real-world reasoning scenarios.
We instead consider two representative reasoning domains, i.e., mathematical reasoning and general open-domain (Wikipedia) QA, and use English as a primary language in our testing.
Furthermore, our method is tested on the setting of fine-tuning the full LLMs, which requires considerable computational resources, while more efficient model training schemes, e.g., LoRA (Hu et al., [2021](https://arxiv.org/html/2401.17464v3#bib.bib16 "")), can be applied in future work.

## Acknowledgements

We thank Beatriz Borges, Gail Weiss, Syrielle Montariol, Li Mi and Zeming Chen for reading and providing comments on drafts of this paper.
Antoine Bosselut gratefully acknowledges the support of the Swiss National Science Foundation (No. 215390), Innosuisse (PFFS-21-29), the EPFL Science Seed Fund, the EPFL Center for Imaging, Sony Group Corporation, and the Allen Institute for AI.

## References

- Anil et al. (2023)  
Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. 2023.  
Palm 2 technical report.  
_arXiv preprint arXiv:2305.10403_.

- Baeza-Yates et al. (1999)  
Ricardo Baeza-Yates, Berthier Ribeiro-Neto, et al. 1999.  
_Modern information retrieval_, volume 463.  
ACM press New York.

- Berant et al. (2013)  
Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. 2013.  
Semantic parsing on freebase from question-answer pairs.  
In _Proceedings of the 2013 conference on empirical methods in natural language processing_, pages 1533–1544.

- Cai et al. (2023)  
Tianle Cai, Xuezhi Wang, Tengyu Ma, Xinyun Chen, and Denny Zhou. 2023.  
Large language models as tool makers.  
_arXiv preprint arXiv:2305.17126_.

- Chen et al. (2023)  
Baian Chen, Chang Shu, Ehsan Shareghi, Nigel Collier, Karthik Narasimhan, and Shunyu Yao. 2023.  
Fireact: Toward language agent fine-tuning.  
_arXiv preprint arXiv:2310.05915_.

- Chen et al. (2021)  
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021.  
Evaluating large language models trained on code.  
_arXiv preprint arXiv:2107.03374_.

- Chen et al. (2022)  
Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. 2022.  
Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks.  
_arXiv preprint arXiv:2211.12588_.

- Chung et al. (2022)  
Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2022.  
Scaling instruction-finetuned language models.  
_arXiv preprint arXiv:2210.11416_.

- Cobbe et al. (2021)  
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021.  
Training verifiers to solve math word problems.  
_arXiv preprint arXiv:2110.14168_.

- Gao et al. (2023)  
Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023.  
Pal: Program-aided language models.  
In _International Conference on Machine Learning_, pages 10764–10799. PMLR.

- Ge et al. (2023)  
Yingqiang Ge, Wenyue Hua, Jianchao Ji, Juntao Tan, Shuyuan Xu, and Yongfeng Zhang. 2023.  
Openagi: When llm meets domain experts.  
_arXiv preprint arXiv:2304.04370_.

- Gou et al. (2023)  
Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. 2023.  
Critic: Large language models can self-correct with tool-interactive critiquing.  
_arXiv preprint arXiv:2305.11738_.

- Hao et al. (2023a)  
Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, and Zhiting Hu. 2023a.  
Reasoning with language model is planning with world model.  
_arXiv preprint arXiv:2305.14992_.

- Hao et al. (2023b)  
Shibo Hao, Tianyang Liu, Zhen Wang, and Zhiting Hu. 2023b.  
Toolkengpt: Augmenting frozen language models with massive tools via tool embeddings.  
_arXiv preprint arXiv:2305.11554_.

- He-Yueya et al. (2023)  
Joy He-Yueya, Gabriel Poesia, Rose E Wang, and Noah D Goodman. 2023.  
Solving math word problems by combining language models with symbolic solvers.  
_arXiv preprint arXiv:2304.09102_.

- Hu et al. (2021)  
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021.  
Lora: Low-rank adaptation of large language models.  
_arXiv preprint arXiv:2106.09685_.

- Huang et al. (2023)  
Yue Huang, Jiawen Shi, Yuan Li, Chenrui Fan, Siyuan Wu, Qihui Zhang, Yixin Liu, Pan Zhou, Yao Wan, Neil Zhenqiang Gong, et al. 2023.  
Metatool benchmark for large language models: Deciding whether to use tools and which to use.  
_arXiv preprint arXiv:2310.03128_.

- Jacovi et al. (2023)  
Alon Jacovi, Avi Caciularu, Jonathan Herzig, Roee Aharoni, Bernd Bohnet, and Mor Geva. 2023.  
A comprehensive evaluation of tool-assisted generation strategies.  
In _Findings of the Association for Computational Linguistics: EMNLP 2023_, pages 13856–13878.

- Ji et al. (2023)  
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023.  
Survey of hallucination in natural language generation.  
_ACM Computing Surveys_, 55:1–38.

- Jin et al. (2023)  
Qiao Jin, Yifan Yang, Qingyu Chen, and Zhiyong Lu. 2023.  
[Genegpt: Augmenting large language models with domain tools for improved access to biomedical information](https://arxiv.org/abs/2304.09667 "").  
_Preprint_, arXiv:2304.09667.

- Joshi et al. (2017)  
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. 2017.  
Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension.  
In _Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_, pages 1601–1611.

- Koncel-Kedziorski et al. (2016)  
Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. 2016.  
Mawps: A math word problem repository.  
In _Proceedings of the 2016 conference of the north american chapter of the association for computational linguistics: human language technologies_, pages 1152–1157.

- Kwiatkowski et al. (2019)  
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. 2019.  
Natural questions: a benchmark for question answering research.  
_Transactions of the Association for Computational Linguistics_, 7:452–466.

- Liu et al. (2023)  
Yuliang Liu, Xiangru Tang, Zefan Cai, Junjie Lu, Yichi Zhang, Yanjun Shao, Zexuan Deng, Helan Hu, Zengxian Yang, Kaikai An, et al. 2023.  
Ml-bench: Large language models leverage open-source libraries for machine learning tasks.  
_arXiv preprint arXiv:2311.09835_.

- Loshchilov and Hutter (2018)  
Ilya Loshchilov and Frank Hutter. 2018.  
Decoupled weight decay regularization.  
In _International Conference on Learning Representations_.

- Lu et al. (2023)  
Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, and Jianfeng Gao. 2023.  
Chameleon: Plug-and-play compositional reasoning with large language models.  
_arXiv preprint arXiv:2304.09842_.

- Maynez et al. (2020)  
Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. 2020.  
On faithfulness and factuality in abstractive summarization.  
In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_, pages 1906–1919.

- Miao et al. (2020)  
Shen-Yun Miao, Chao-Chun Liang, and Keh-Yih Su. 2020.  
A diverse corpus for evaluating and developing english math word problem solvers.  
In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_, pages 975–984.

- OpenAI (2023)  
OpenAI. 2023.  
[Gpt-4 technical report](https://arxiv.org/abs/2303.08774 "").  
_Preprint_, arXiv:2303.08774.

- Parisi et al. (2022)  
Aaron Parisi, Yao Zhao, and Noah Fiedel. 2022.  
Talm: Tool augmented language models.  
_arXiv preprint arXiv:2205.12255_.

- Patel et al. (2021)  
Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021.  
Are nlp models really able to solve simple math word problems?  
In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_, pages 2080–2094.

- Patil et al. (2023)  
Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez. 2023.  
Gorilla: Large language model connected with massive apis.  
_arXiv preprint arXiv:2305.15334_.

- Petroni et al. (2021)  
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, et al. 2021.  
Kilt: a benchmark for knowledge intensive language tasks.  
In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies_, pages 2523–2544.

- Reimers and Gurevych (2019)  
Nils Reimers and Iryna Gurevych. 2019.  
Sentence-bert: Sentence embeddings using siamese bert-networks.  
In _Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)_, pages 3982–3992.

- Robertson et al. (1995)  
Stephen E Robertson, Steve Walker, Susan Jones, Micheline M Hancock-Beaulieu, Mike Gatford, et al. 1995.  
Okapi at trec-3.  
_Nist Special Publication Sp_, 109:109.

- Schick et al. (2023)  
Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023.  
Toolformer: Language models can teach themselves to use tools.  
_arXiv preprint arXiv:2302.04761_.

- Shen et al. (2023)  
Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. 2023.  
Hugginggpt: Solving ai tasks with chatgpt and its friends in huggingface.  
_arXiv preprint arXiv:2303.17580_.

- Touvron et al. (2023a)  
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023a.  
Llama: Open and efficient foundation language models.  
_arXiv preprint arXiv:2302.13971_.

- Touvron et al. (2023b)  
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b.  
Llama 2: Open foundation and fine-tuned chat models.  
_arXiv preprint arXiv:2307.09288_.

- Wang et al. (2022)  
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022.  
Self-consistency improves chain of thought reasoning in language models.  
In _The Eleventh International Conference on Learning Representations_.

- Wei et al. (2021)  
Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. 2021.  
Finetuned language models are zero-shot learners.  
_arXiv preprint arXiv:2109.01652_.

- Wei et al. (2022)  
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022.  
Chain-of-thought prompting elicits reasoning in large language models.  
_Advances in Neural Information Processing Systems_, 35:24824–24837.

- Wenzek et al. (2020)  
Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Joulin, and Édouard Grave. 2020.  
Ccnet: Extracting high quality monolingual datasets from web crawl data.  
In _Proceedings of the Twelfth Language Resources and Evaluation Conference_, pages 4003–4012.

- Xu et al. (2023)  
Qiantong Xu, Fenglu Hong, Bo Li, Changran Hu, Zhengyu Chen, and Jian Zhang. 2023.  
On the tool manipulation capability of open-source large language models.  
_arXiv preprint arXiv:2305.16504_.

- Yang et al. (2018)  
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018.  
Hotpotqa: A dataset for diverse, explainable multi-hop question answering.  
In _Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing_. Association for Computational Linguistics.

- Yao et al. (2023a)  
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. 2023a.  
Tree of thoughts: Deliberate problem solving with large language models.  
_arXiv preprint arXiv:2305.10601_.

- Yao et al. (2023b)  
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023b.  
[React: Synergizing reasoning and acting in language models](https://arxiv.org/abs/2210.03629 "").  
_Preprint_, arXiv:2210.03629.

- Zhuang et al. (2023)  
Yuchen Zhuang, Xiang Chen, Tong Yu, Saayan Mitra, Victor Bursztyn, Ryan A Rossi, Somdeb Sarkhel, and Chao Zhang. 2023.  
Toolchain\*: Efficient action space navigation in large language models with a\* search.  
_arXiv preprint arXiv:2310.13227_.

</details>

<details>
<summary>function-calling-openai-api</summary>

# Function calling

Enable models to fetch data and take actions.

**Function calling** provides a powerful and flexible way for OpenAI models to interface with your code or external services. This guide will explain how to connect the models to your own custom code to fetch data or take action.

Get weather

Function calling example with get\_weather function

python

```python
from openai import OpenAI

client = OpenAI()

tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country e.g. Bogotá, Colombia"
            }
        },
        "required": [
            "location"
        ],
        "additionalProperties": False
    }
}]

response = client.responses.create(
    model="gpt-4.1",
    input=[{"role": "user", "content": "What is the weather like in Paris today?"}],
    tools=tools
)

print(response.output)
```

```javascript
import { OpenAI } from "openai";

const openai = new OpenAI();

const tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country e.g. Bogotá, Colombia"
            }
        },
        "required": [
            "location"
        ],
        "additionalProperties": false
    }
}];

const response = await openai.responses.create({
    model: "gpt-4.1",
    input: [{ role: "user", content: "What is the weather like in Paris today?" }],
    tools,
});

console.log(response.output);
```

```bash
curl https://api.openai.com/v1/responses \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $OPENAI_API_KEY" \
-d '{
    "model": "gpt-4.1",
    "input": "What is the weather like in Paris today?",
    "tools": [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get current temperature for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country e.g. Bogotá, Colombia"
                    }
                },
                "required": [
                    "location"
                ],
                "additionalProperties": false
            }
        }
    ]
}'
```

Output

```json
[{
    "type": "function_call",
    "id": "fc_12345xyz",
    "call_id": "call_12345xyz",
    "name": "get_weather",
    "arguments": "{\"location\":\"Paris, France\"}"
}]
```

Send email

Function calling example with send\_email function

python

```python
from openai import OpenAI

client = OpenAI()

tools = [{
    "type": "function",
    "name": "send_email",
    "description": "Send an email to a given recipient with a subject and message.",
    "parameters": {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "The recipient email address."
            },
            "subject": {
                "type": "string",
                "description": "Email subject line."
            },
            "body": {
                "type": "string",
                "description": "Body of the email message."
            }
        },
        "required": [
            "to",
            "subject",
            "body"
        ],
        "additionalProperties": False
    }
}]

response = client.responses.create(
    model="gpt-4.1",
    input=[{"role": "user", "content": "Can you send an email to ilan@example.com and katia@example.com saying hi?"}],
    tools=tools
)

print(response.output)
```

```javascript
import { OpenAI } from "openai";

const openai = new OpenAI();

const tools = [{
    "type": "function",
    "name": "send_email",
    "description": "Send an email to a given recipient with a subject and message.",
    "parameters": {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "The recipient email address."
            },
            "subject": {
                "type": "string",
                "description": "Email subject line."
            },
            "body": {
                "type": "string",
                "description": "Body of the email message."
            }
        },
        "required": [
            "to",
            "subject",
            "body"
        ],
        "additionalProperties": false
    }
}];

const response = await openai.responses.create({
    model: "gpt-4.1",
    input: [{ role: "user", content: "Can you send an email to ilan@example.com and katia@example.com saying hi?" }],
    tools,
});

console.log(response.output);
```

```bash
curl https://api.openai.com/v1/responses \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $OPENAI_API_KEY" \
-d '{
    "model": "gpt-4.1",
    "input": "Can you send an email to ilan@example.com and katia@example.com saying hi?",
    "tools": [
        {
            "type": "function",
            "name": "send_email",
            "description": "Send an email to a given recipient with a subject and message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "The recipient email address."
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line."
                    },
                    "body": {
                        "type": "string",
                        "description": "Body of the email message."
                    }
                },
                "required": [
                    "to",
                    "subject",
                    "body"
                ],
                "additionalProperties": false
            }
        }
    ]
}'
```

Output

```json
[
    {
        "type": "function_call",
        "id": "fc_12345xyz",
        "call_id": "call_9876abc",
        "name": "send_email",
        "arguments": "{\"to\":\"ilan@example.com\",\"subject\":\"Hello!\",\"body\":\"Just wanted to say hi\"}"
    },
    {
        "type": "function_call",
        "id": "fc_12345xyz",
        "call_id": "call_9876abc",
        "name": "send_email",
        "arguments": "{\"to\":\"katia@example.com\",\"subject\":\"Hello!\",\"body\":\"Just wanted to say hi\"}"
    }
]
```

Search knowledge base

Function calling example with search\_knowledge\_base function

python

```python
from openai import OpenAI

client = OpenAI()

tools = [{
    "type": "function",
    "name": "search_knowledge_base",
    "description": "Query a knowledge base to retrieve relevant info on a topic.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user question or search query."
            },
            "options": {
                "type": "object",
                "properties": {
                    "num_results": {
                        "type": "number",
                        "description": "Number of top results to return."
                    },
                    "domain_filter": {
                        "type": [
                            "string",
                            "null"
                        ],
                        "description": "Optional domain to narrow the search (e.g. 'finance', 'medical'). Pass null if not needed."
                    },
                    "sort_by": {
                        "type": [
                            "string",
                            "null"
                        ],
                        "enum": [
                            "relevance",
                            "date",
                            "popularity",
                            "alphabetical"
                        ],
                        "description": "How to sort results. Pass null if not needed."
                    }
                },
                "required": [
                    "num_results",
                    "domain_filter",
                    "sort_by"
                ],
                "additionalProperties": False
            }
        },
        "required": [
            "query",
            "options"
        ],
        "additionalProperties": False
    }
}]

response = client.responses.create(
    model="gpt-4.1",
    input=[{"role": "user", "content": "Can you find information about ChatGPT in the AI knowledge base?"}],
    tools=tools
)

print(response.output)
```

```javascript
import { OpenAI } from "openai";

const openai = new OpenAI();

const tools = [{
    "type": "function",
    "name": "search_knowledge_base",
    "description": "Query a knowledge base to retrieve relevant info on a topic.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user question or search query."
            },
            "options": {
                "type": "object",
                "properties": {
                    "num_results": {
                        "type": "number",
                        "description": "Number of top results to return."
                    },
                    "domain_filter": {
                        "type": [
                            "string",
                            "null"
                        ],
                        "description": "Optional domain to narrow the search (e.g. 'finance', 'medical'). Pass null if not needed."
                    },
                    "sort_by": {
                        "type": [
                            "string",
                            "null"
                        ],
                        "enum": [
                            "relevance",
                            "date",
                            "popularity",
                            "alphabetical"
                        ],
                        "description": "How to sort results. Pass null if not needed."
                    }
                },
                "required": [
                    "num_results",
                    "domain_filter",
                    "sort_by"
                ],
                "additionalProperties": false
            }
        },
        "required": [
            "query",
            "options"
        ],
        "additionalProperties": false
    }
}];

const response = await openai.responses.create({
    model: "gpt-4.1",
    input: [{ role: "user", content: "Can you find information about ChatGPT in the AI knowledge base?" }],
    tools,
});

console.log(response.output);
```

```bash
curl https://api.openai.com/v1/responses \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $OPENAI_API_KEY" \
-d '{
    "model": "gpt-4.1",
    "input": "Can you find information about ChatGPT in the AI knowledge base?",
    "tools": [
        {
            "type": "function",
            "name": "search_knowledge_base",
            "description": "Query a knowledge base to retrieve relevant info on a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user question or search query."
                    },
                    "options": {
                        "type": "object",
                        "properties": {
                            "num_results": {
                                "type": "number",
                                "description": "Number of top results to return."
                            },
                            "domain_filter": {
                                "type": [
                                    "string",
                                    "null"
                                ],
                                "description": "Optional domain to narrow the search (e.g. 'finance', 'medical'). Pass null if not needed."
                            },
                            "sort_by": {
                                "type": [
                                    "string",
                                    "null"
                                ],
                                "enum": [
                                    "relevance",
                                    "date",
                                    "popularity",
                                    "alphabetical"
                                ],
                                "description": "How to sort results. Pass null if not needed."
                            }
                        },
                        "required": [
                            "num_results",
                            "domain_filter",
                            "sort_by"
                        ],
                        "additionalProperties": false
                    }
                },
                "required": [
                    "query",
                    "options"
                ],
                "additionalProperties": false
            }
        }
    ]
}'
```

Output

```json
[{
    "type": "function_call",
    "id": "fc_12345xyz",
    "call_id": "call_4567xyz",
    "name": "search_knowledge_base",
    "arguments": "{\"query\":\"What is ChatGPT?\",\"options\":{\"num_results\":3,\"domain_filter\":null,\"sort_by\":\"relevance\"}}"
}]
```

## Overview

You can give the model access to your own custom code through **function calling**. Based on the system prompt and messages, the model may decide to call these functions — **instead of (or in addition to) generating text or audio**.

You'll then execute the function code, send back the results, and the model will incorporate them into its final response.

Function calling has two primary use cases:

|  |  |
| --- | --- |
| **Fetching Data** | Retrieve up-to-date information to incorporate into the model's response (RAG). Useful for searching knowledge bases and retrieving specific data from APIs (e.g. current weather data). |
| **Taking Action** | Perform actions like submitting a form, calling APIs, modifying application state (UI/frontend or backend), or taking agentic workflow actions (like [handing off](https://cookbook.openai.com/examples/orchestrating_agents) the conversation). |

### Sample function

Let's look at the steps to allow a model to use a real `get_weather` function defined below:

Sample get\_weather function implemented in your codebase

python

```python
import requests

def get_weather(latitude, longitude):
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()
    return data['current']['temperature_2m']
```

```javascript
async function getWeather(latitude, longitude) {
    const response = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m`);
    const data = await response.json();
    return data.current.temperature_2m;
}
```

Unlike the diagram earlier, this function expects precise `latitude` and `longitude` instead of a general `location` parameter. (However, our models can automatically determine the coordinates for many locations!)

### Function calling steps

**Call model with [functions defined](https://platform.openai.com/docs/guides/function-calling?api-mode=responses#defining-functions)** – along with your system and user messages.

Step 1: Call model with get\_weather tool defined

python

```python
from openai import OpenAI
import json

client = OpenAI()

tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for provided coordinates in celsius.",
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {"type": "number"},
            "longitude": {"type": "number"}
        },
        "required": ["latitude", "longitude"],
        "additionalProperties": False
    },
    "strict": True
}]

input_messages = [{"role": "user", "content": "What's the weather like in Paris today?"}]

response = client.responses.create(
    model="gpt-4.1",
    input=input_messages,
    tools=tools,
)
```

```javascript
import { OpenAI } from "openai";

const openai = new OpenAI();

const tools = [{
    type: "function",
    name: "get_weather",
    description: "Get current temperature for provided coordinates in celsius.",
    parameters: {
        type: "object",
        properties: {
            latitude: { type: "number" },
            longitude: { type: "number" }
        },
        required: ["latitude", "longitude"],
        additionalProperties: false
    },
    strict: true
}];

const input = [
    {
        role: "user",
        content: "What's the weather like in Paris today?"
    }
];

const response = await openai.responses.create({
    model: "gpt-4.1",
    input,
    tools,
});
```

**Model decides to call function(s)** – model returns the **name** and **input arguments**.

response.output

```json
[{
    "type": "function_call",
    "id": "fc_12345xyz",
    "call_id": "call_12345xyz",
    "name": "get_weather",
    "arguments": "{\"latitude\":48.8566,\"longitude\":2.3522}"
}]
```

**Execute function code** – parse the model's response and [handle function calls](https://platform.openai.com/docs/guides/function-calling?api-mode=responses#handling-function-calls).

Step 3: Execute get\_weather function

python

```python
tool_call = response.output[0]
args = json.loads(tool_call.arguments)

result = get_weather(args["latitude"], args["longitude"])
```

```javascript
const toolCall = response.output[0];
const args = JSON.parse(toolCall.arguments);

const result = await getWeather(args.latitude, args.longitude);
```

**Supply model with results** – so it can incorporate them into its final response.

Step 4: Supply result and call model again

python

```python
input_messages.append(tool_call)  # append model's function call message
input_messages.append({                               # append result message
    "type": "function_call_output",
    "call_id": tool_call.call_id,
    "output": str(result)
})

response_2 = client.responses.create(
    model="gpt-4.1",
    input=input_messages,
    tools=tools,
)
print(response_2.output_text)
```

```javascript
input.push(toolCall); // append model's function call message
input.push({                               // append result message
    type: "function_call_output",
    call_id: toolCall.call_id,
    output: result.toString()
});

const response2 = await openai.responses.create({
    model: "gpt-4.1",
    input,
    tools,
    store: true,
});

console.log(response2.output_text)
```

**Model responds** – incorporating the result in its output.

response\_2.output\_text

```json
"The current temperature in Paris is 14°C (57.2°F)."
```

## Defining functions

Functions can be set in the `tools` parameter of each API request.

A function is defined by its schema, which informs the model what it does and what input arguments it expects. It comprises the following fields:

| Field | Description |
| --- | --- |
| `type` | This should always be `function` |
| `name` | The function's name (e.g. `get_weather`) |
| `description` | Details on when and how to use the function |
| `parameters` | [JSON schema](https://json-schema.org/) defining the function's input arguments |
| `strict` | Whether to enforce strict mode for the function call |

Take a look at this example or generate your own below (or in our [Playground](https://platform.openai.com/playground)).

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Retrieves current weather for the given location.",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City and country e.g. Bogotá, Colombia"
      },
      "units": {
        "type": "string",
        "enum": [
          "celsius",
          "fahrenheit"
        ],
        "description": "Units the temperature will be returned in."
      }
    },
    "required": [
      "location",
      "units"
    ],
    "additionalProperties": false
  },
  "strict": true
}
```

Because the `parameters` are defined by a [JSON schema](https://json-schema.org/), you can leverage many of its rich features like property types, enums, descriptions, nested objects, and, recursive objects.

### Best practices for defining functions

1. **Write clear and detailed function names, parameter descriptions, and instructions.**
   - **Explicitly describe the purpose of the function and each parameter** (and its format), and what the output represents.
   - **Use the system prompt to describe when (and when not) to use each function.** Generally, tell the model _exactly_ what to do.
   - **Include examples and edge cases**, especially to rectify any recurring failures. ( **Note:** Adding examples may hurt performance for [reasoning models](https://platform.openai.com/docs/guides/reasoning).)
2. **Apply software engineering best practices.**
   - **Make the functions obvious and intuitive**. ( [principle of least surprise](https://en.wikipedia.org/wiki/Principle_of_least_astonishment))
   - **Use enums** and object structure to make invalid states unrepresentable. (e.g. `toggle_light(on: bool, off: bool)` allows for invalid calls)
   - **Pass the intern test.** Can an intern/human correctly use the function given nothing but what you gave the model? (If not, what questions do they ask you? Add the answers to the prompt.)
3. **Offload the burden from the model and use code where possible.**
   - **Don't make the model fill arguments you already know.** For example, if you already have an `order_id` based on a previous menu, don't have an `order_id` param – instead, have no params `submit_refund()` and pass the `order_id` with code.
   - **Combine functions that are always called in sequence.** For example, if you always call `mark_location()` after `query_location()`, just move the marking logic into the query function call.
4. **Keep the number of functions small for higher accuracy.**
   - **Evaluate your performance** with different numbers of functions.
   - **Aim for fewer than 20 functions** at any one time, though this is just a soft suggestion.
5. **Leverage OpenAI resources.**
   - **Generate and iterate on function schemas** in the [Playground](https://platform.openai.com/playground).
   - **Consider [fine-tuning](https://platform.openai.com/docs/guides/fine-tuning) to increase function calling accuracy** for large numbers of functions or difficult tasks. ( [cookbook](https://cookbook.openai.com/examples/fine_tuning_for_function_calling))

### Token Usage

Under the hood, functions are injected into the system message in a syntax the model has been trained on. This means functions count against the model's context limit and are billed as input tokens. If you run into token limits, we suggest limiting the number of functions or the length of the descriptions you provide for function parameters.

It is also possible to use [fine-tuning](https://platform.openai.com/docs/guides/fine-tuning#fine-tuning-examples) to reduce the number of tokens used if you have many functions defined in your tools specification.

## Handling function calls

When the model calls a function, you must execute it and return the result. Since model responses can include zero, one, or multiple calls, it is best practice to assume there are several.

The response `output` array contains an entry with the `type` having a value of `function_call`. Each entry with a `call_id` (used later to submit the function result), `name`, and JSON-encoded `arguments`.

Sample response with multiple function calls

```json
[
    {
        "id": "fc_12345xyz",
        "call_id": "call_12345xyz",
        "type": "function_call",
        "name": "get_weather",
        "arguments": "{\"location\":\"Paris, France\"}"
    },
    {
        "id": "fc_67890abc",
        "call_id": "call_67890abc",
        "type": "function_call",
        "name": "get_weather",
        "arguments": "{\"location\":\"Bogotá, Colombia\"}"
    },
    {
        "id": "fc_99999def",
        "call_id": "call_99999def",
        "type": "function_call",
        "name": "send_email",
        "arguments": "{\"to\":\"bob@email.com\",\"body\":\"Hi bob\"}"
    }
]
```

Execute function calls and append results

python

```python
for tool_call in response.output:
    if tool_call.type != "function_call":
        continue

    name = tool_call.name
    args = json.loads(tool_call.arguments)

    result = call_function(name, args)
    input_messages.append({
        "type": "function_call_output",
        "call_id": tool_call.call_id,
        "output": str(result)
    })
```

```javascript
for (const toolCall of response.output) {
    if (toolCall.type !== "function_call") {
        continue;
    }

    const name = toolCall.name;
    const args = JSON.parse(toolCall.arguments);

    const result = callFunction(name, args);
    input.push({
        type: "function_call_output",
        call_id: toolCall.call_id,
        output: result.toString()
    });
}
```

In the example above, we have a hypothetical `call_function` to route each call. Here’s a possible implementation:

Execute function calls and append results

python

```python
def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)
    if name == "send_email":
        return send_email(**args)
```

```javascript
const callFunction = async (name, args) => {
    if (name === "get_weather") {
        return getWeather(args.latitude, args.longitude);
    }
    if (name === "send_email") {
        return sendEmail(args.to, args.body);
    }
};
```

### Formatting results

A result must be a string, but the format is up to you (JSON, error codes, plain text, etc.). The model will interpret that string as needed.

If your function has no return value (e.g. `send_email`), simply return a string to indicate success or failure. (e.g. `"success"`)

### Incorporating results into response

After appending the results to your `input`, you can send them back to the model to get a final response.

Send results back to model

python

```python
response = client.responses.create(
    model="gpt-4.1",
    input=input_messages,
    tools=tools,
)
```

```javascript
const response = await openai.responses.create({
    model: "gpt-4.1",
    input,
    tools,
});
```

Final response

```json
"It's about 15°C in Paris, 18°C in Bogotá, and I've sent that email to Bob."
```

## Additional configurations

### Tool choice

By default the model will determine when and how many tools to use. You can force specific behavior with the `tool_choice` parameter.

1. **Auto:** ( _Default_) Call zero, one, or multiple functions. `tool_choice: "auto"`
2. **Required:** Call one or more functions.
`tool_choice: "required"`

3. **Forced Function:** Call exactly one specific function.
`tool_choice: {"type": "function", "name": "get_weather"}`

You can also set `tool_choice` to `"none"` to imitate the behavior of passing no functions.

### Parallel function calling

The model may choose to call multiple functions in a single turn. You can prevent this by setting `parallel_tool_calls` to `false`, which ensures exactly zero or one tool is called.

**Note:** Currently, if you are using a fine tuned model and the model calls multiple functions in one turn then [strict mode](https://platform.openai.com/docs/guides/function-calling?api-mode=responses#strict-mode) will be disabled for those calls.

**Note for `gpt-4.1-nano-2025-04-14`:** This snapshot of `gpt-4.1-nano` can sometimes include multiple tools calls for the same tool if parallel tool calls are enabled. It is recommended to disable this feature when using this nano snapshot.

### Strict mode

Setting `strict` to `true` will ensure function calls reliably adhere to the function schema, instead of being best effort. We recommend always enabling strict mode.

Under the hood, strict mode works by leveraging our [structured outputs](https://platform.openai.com/docs/guides/structured-outputs) feature and therefore introduces a couple requirements:

1. `additionalProperties` must be set to `false` for each object in the `parameters`.
2. All fields in `properties` must be marked as `required`.

You can denote optional fields by adding `null` as a `type` option (see example below).

Strict mode enabled

```json
{
    "type": "function",
    "name": "get_weather",
    "description": "Retrieves current weather for the given location.",
    "strict": true,
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country e.g. Bogotá, Colombia"
            },
            "units": {
                "type": ["string", "null"],
                "enum": ["celsius", "fahrenheit"],
                "description": "Units the temperature will be returned in."
            }
        },
        "required": ["location", "units"],
        "additionalProperties": false
    }
}
```

Strict mode disabled

```json
{
    "type": "function",
    "name": "get_weather",
    "description": "Retrieves current weather for the given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country e.g. Bogotá, Colombia"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Units the temperature will be returned in."
            }
        },
        "required": ["location"],
    }
}
```

All schemas generated in the [playground](https://platform.openai.com/playground) have strict mode enabled.

While we recommend you enable strict mode, it has a few limitations:

1. Some features of JSON schema are not supported. (See [supported schemas](https://platform.openai.com/docs/guides/structured-outputs?context=with_parse#supported-schemas).)

Specifically for fine tuned models:

1. Schemas undergo additional processing on the first request (and are then cached). If your schemas vary from request to request, this may result in higher latencies.
2. Schemas are cached for performance, and are not eligible for [zero data retention](https://platform.openai.com/docs/models#how-we-use-your-data).

## Streaming

Streaming can be used to surface progress by showing which function is called as the model fills its arguments, and even displaying the arguments in real time.

Streaming function calls is very similar to streaming regular responses: you set `stream` to `true` and get different `event` objects.

Streaming function calls

python

```python
from openai import OpenAI

client = OpenAI()

tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country e.g. Bogotá, Colombia"
            }
        },
        "required": [
            "location"
        ],
        "additionalProperties": False
    }
}]

stream = client.responses.create(
    model="gpt-4.1",
    input=[{"role": "user", "content": "What's the weather like in Paris today?"}],
    tools=tools,
    stream=True
)

for event in stream:
    print(event)
```

```javascript
import { OpenAI } from "openai";

const openai = new OpenAI();

const tools = [{
    type: "function",
    name: "get_weather",
    description: "Get current temperature for provided coordinates in celsius.",
    parameters: {
        type: "object",
        properties: {
            latitude: { type: "number" },
            longitude: { type: "number" }
        },
        required: ["latitude", "longitude"],
        additionalProperties: false
    },
    strict: true
}];

const stream = await openai.responses.create({
    model: "gpt-4.1",
    input: [{ role: "user", content: "What's the weather like in Paris today?" }],
    tools,
    stream: true,
    store: true,
});

for await (const event of stream) {
    console.log(event)
}
```

Output events

```json
{"type":"response.output_item.added","response_id":"resp_1234xyz","output_index":0,"item":{"type":"function_call","id":"fc_1234xyz","call_id":"call_1234xyz","name":"get_weather","arguments":""}}
{"type":"response.function_call_arguments.delta","response_id":"resp_1234xyz","item_id":"fc_1234xyz","output_index":0,"delta":"{\""}
{"type":"response.function_call_arguments.delta","response_id":"resp_1234xyz","item_id":"fc_1234xyz","output_index":0,"delta":"location"}
{"type":"response.function_call_arguments.delta","response_id":"resp_1234xyz","item_id":"fc_1234xyz","output_index":0,"delta":"\":\""}
{"type":"response.function_call_arguments.delta","response_id":"resp_1234xyz","item_id":"fc_1234xyz","output_index":0,"delta":"Paris"}
{"type":"response.function_call_arguments.delta","response_id":"resp_1234xyz","item_id":"fc_1234xyz","output_index":0,"delta":","}
{"type":"response.function_call_arguments.delta","response_id":"resp_1234xyz","item_id":"fc_1234xyz","output_index":0,"delta":" France"}
{"type":"response.function_call_arguments.delta","response_id":"resp_1234xyz","item_id":"fc_1234xyz","output_index":0,"delta":"\"}"}
{"type":"response.function_call_arguments.done","response_id":"resp_1234xyz","item_id":"fc_1234xyz","output_index":0,"arguments":"{\"location\":\"Paris, France\"}"}
{"type":"response.output_item.done","response_id":"resp_1234xyz","output_index":0,"item":{"type":"function_call","id":"fc_1234xyz","call_id":"call_2345abc","name":"get_weather","arguments":"{\"location\":\"Paris, France\"}"}}
```

Instead of aggregating chunks into a single `content` string, however, you're aggregating chunks into an encoded `arguments` JSON object.

When the model calls one or more functions an event of type `response.output_item.added` will be emitted for each function call that contains the following fields:

| Field | Description |
| --- | --- |
| `response_id` | The id of the response that the function call belongs to |
| `output_index` | The index of the output item in the response. This respresents the individual function calls in the response. |
| `item` | The in-progress function call item that includes a `name`, `arguments` and `id` field |

Afterwards you will receive a series of events of type `response.function_call_arguments.delta` which will contain the `delta` of the `arguments` field. These events contain the following fields:

| Field | Description |
| --- | --- |
| `response_id` | The id of the response that the function call belongs to |
| `item_id` | The id of the function call item that the delta belongs to |
| `output_index` | The index of the output item in the response. This respresents the individual function calls in the response. |
| `delta` | The delta of the `arguments` field. |

Below is a code snippet demonstrating how to aggregate the `delta` s into a final `tool_call` object.

Accumulating tool\_call deltas

python

```python
final_tool_calls = {}

for event in stream:
    if event.type === 'response.output_item.added':
        final_tool_calls[event.output_index] = event.item;
    elif event.type === 'response.function_call_arguments.delta':
        index = event.output_index

        if final_tool_calls[index]:
            final_tool_calls[index].arguments += event.delta
```

```javascript
const finalToolCalls = {};

for await (const event of stream) {
    if (event.type === 'response.output_item.added') {
        finalToolCalls[event.output_index] = event.item;
    } else if (event.type === 'response.function_call_arguments.delta') {
        const index = event.output_index;

        if (finalToolCalls[index]) {
            finalToolCalls[index].arguments += event.delta;
        }
    }
}
```

Accumulated final\_tool\_calls\[0\]

```json
{
    "type": "function_call",
    "id": "fc_1234xyz",
    "call_id": "call_2345abc",
    "name": "get_weather",
    "arguments": "{\"location\":\"Paris, France\"}"
}
```

When the model has finished calling the functions an event of type `response.function_call_arguments.done` will be emitted. This event contains the entire function call including the following fields:

| Field | Description |
| --- | --- |
| `response_id` | The id of the response that the function call belongs to |
| `output_index` | The index of the output item in the response. This respresents the individual function calls in the response. |
| `item` | The function call item that includes a `name`, `arguments` and `id` field. |

</details>

<details>
<summary>function-calling-with-the-gemini-api-google-ai-for-developer</summary>

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
in a single turn (parallel function calling) and in
sequence (compositional function calling).

### Step 1: Define a function declaration

Define a function and its declaration within your application code that allows
users to set light values and make an API request. This function could call
external services or APIs.

```python
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

### Step 2: Call the model with function declarations

Once you have defined your function declarations, you can prompt the model to
use them. It analyzes the prompt and function declarations and decides whether
to respond directly or to call a function. If a function is called, the response
object will contain a function call suggestion.

```python
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

The model then returns a `functionCall` object in an OpenAPI compatible
schema specifying how to call one or more of the declared functions in order to
respond to the user's question.

```python
id=None args={'color_temp': 'warm', 'brightness': 25} name='set_light_values'
```

### Step 3: Execute set\_light\_values function code

Extract the function call details from the model's response, parse the arguments
, and execute the `set_light_values` function.

```python
# Extract tool call details, it may not be in the first part.
tool_call = response.candidates[0].content.parts[0].function_call

if tool_call.name == "set_light_values":
    result = set_light_values(**tool_call.args)
    print(f"Function execution result: {result}")
```

### Step 4: Create user friendly response with function result and call the model again

Finally, send the result of the function execution back to the model so it can
incorporate this information into its final response to the user.

```python
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

## Parallel function calling

In addition to single turn function calling, you can also call multiple
functions at once. Parallel function calling lets you execute multiple functions
at once and is used when the functions are not dependent on each other. This is
useful in scenarios like gathering data from multiple independent sources, such
as retrieving customer details from different databases or checking inventory
levels across various warehouses or performing multiple actions such as
converting your apartment into a disco.

```python
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

Configure the function calling mode to allow using all of the specified tools.
To learn more, you can read about
[configuring function calling](https://ai.google.dev/gemini-api/docs/function-calling#function_calling_modes).

```python
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

Each of the printed results reflects a single function call that the model has
requested. To send the results back, include the responses in the same order as
they were requested.

The Python SDK supports [automatic function calling](https://ai.google.dev/gemini-api/docs/function-calling#automatic_function_calling_python_only),
which automatically converts Python functions to declarations, handles the
function call execution and response cycle for you. Following is an example for
the disco use case.

```python
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
"Get the temperature in a given location", the Gemini API might first invoke
a `get_current_location()` function followed by a `get_weather()` function that
takes the location as a parameter.

The following example demonstrates how to implement compositional function
calling using the Python SDK and automatic function calling.

This example uses the automatic function calling feature of the
`google-genai` Python SDK. The SDK automatically converts the Python
functions to the required schema, executes the function calls when requested
by the model, and sends the results back to the model to complete the task.

```python
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

```python
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

```python
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

```python
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

```python
AllowedType = (
  int | float | bool | str | list['AllowedType'] | pydantic.BaseModel)
```

To see what the inferred schema looks like, you can convert it using
[`from_callable`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionDeclaration.from_callable):

```python
def multiply(a: float, b: float):
    """Returns a * b."""
    return a * b

fn_decl = types.FunctionDeclaration.from_callable(callable=multiply, client=client)

# to_json_dict() provides a clean JSON representation.
print(fn_decl.to_json_dict())
```

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
<summary>react-vs-plan-and-execute-a-practical-comparison-of-llm-agen</summary>

I have analyzed the provided markdown and the article guidelines. The markdown content is a detailed comparison of the ReAct and Plan-and-Execute agent patterns. However, the article guidelines for this lesson state that ReAct is a concept to be introduced in a future lesson (Lesson 7), and the current lesson should only briefly mention it as a solution to the limitations of simple tool-calling loops.

Therefore, the entire body of the provided markdown content is irrelevant to the specified lesson plan and falls outside the scope defined by the guidelines. As the task is to *only remove* irrelevant content, and all the core content of the provided markdown is irrelevant, the correct output is an empty string.

</details>
