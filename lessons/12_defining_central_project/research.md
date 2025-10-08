# Research

## Research Results

<details>
<summary>What is the Model-Context Protocol (MCP) and how does it facilitate steerable, human-in-the-loop AI agent workflows?</summary>

### Source [1]: https://www.solo.io/topics/ai-infrastructure/what-is-mcp

Query: What is the Model-Context Protocol (MCP) and how does it facilitate steerable, human-in-the-loop AI agent workflows?

Answer: The **Model Context Protocol (MCP)** is an open standard introduced by Anthropic in November 2024 to provide a universal, model-agnostic interface for connecting large language models (LLMs) to external tools, systems, and data sources. MCP enables secure, bi-directional context exchange between AI applications and data repositories through a well-defined protocol, allowing developers to expose data via MCP servers or build MCP clients that access those servers. This architecture lets AI agents access structured, relevant data from multiple tools and sources using pre-defined templates and slash commands, supporting custom logic and error handling.

MCP facilitates steerable, human-in-the-loop workflows by maintaining explicit context, addressing issues like repetitive queries and irrelevant responses. It also supports integration with business tools, development environments, and hardware, enabling richer, more accurate, and context-aware AI experiences. The protocol includes robust security features, such as JWT and OIDC authentication, supporting secure, policy-driven access to enterprise data.

Organizations like Block and Apollo use MCP to allow AI agents to securely retrieve proprietary information, while development tool providers such as Replit and Zed leverage MCP for context-aware coding assistants. However, adopting MCP increases the operational and security risk surface, introducing challenges in access control, schema management, and observability. Each MCP-enabled endpoint requires robust security and operational management to prevent issues like data leakage, credential theft, and increased latency.

In summary, MCP streamlines the integration of AI models with external systems, making workflows more interactive and context-rich, but also introduces new security and operational complexities that must be managed carefully.

-----

-----

-----

### Source [2]: https://en.wikipedia.org/wiki/Model_Context_Protocol

Query: What is the Model-Context Protocol (MCP) and how does it facilitate steerable, human-in-the-loop AI agent workflows?

Answer: The **Model Context Protocol (MCP)** is an open standard and open-source framework introduced by Anthropic in November 2024 to standardize how AI systems, including LLMs, integrate and share data with external tools and sources. MCP offers a universal interface for reading files, executing functions, and handling contextual prompts, and has been widely adopted by major AI providers such as OpenAI and Google DeepMind.

The protocol was designed to overcome the "N×M" integration challenge, where each data source previously required custom connectors, resulting in significant complexity. MCP draws inspiration from the Language Server Protocol (LSP), reusing message flow concepts and specifying JSON-RPC 2.0 as its underlying message format. It formally supports stdio and HTTP (optionally with Server-Sent Events) as standard transport mechanisms.

By providing these standardized mechanisms, MCP enables steerable, human-in-the-loop workflows by letting AI assistants interact with content repositories, business management tools, and development environments in a seamless and vendor-neutral way. This structure helps break down information silos and simplifies the development of AI agents that can dynamically interact with diverse, real-world data and tools.

-----

-----

-----

### Source [3]: https://www.cloudflare.com/it-it/learning/ai/what-is-model-context-protocol-mcp/

Query: What is the Model-Context Protocol (MCP) and how does it facilitate steerable, human-in-the-loop AI agent workflows?

Answer: The **Model Context Protocol (MCP)** is a standardized way for AI agents to access external tools and data sources, allowing them to act beyond their original training by incorporating new information into their decision-making and outputs. MCP functions similarly to an API but is specific to the needs of AI, offering a documented, universal approach for integrating external services.

MCP supports "agentic AI"—intelligent programs capable of autonomous action—by providing the equivalent of a "phone number" for AI agents to obtain the information needed to complete tasks. For example, an AI assistant could use MCP to access a restaurant booking system, retrieve availability, and make reservations, all via standardized integrations.

Developed by Anthropic and open-sourced in late 2024, MCP has rapidly become an industry standard for connecting AI agents to external systems. Its adoption enables more advanced, context-aware, and steerable workflows, where humans and AI agents can collaborate effectively, leveraging up-to-date and relevant data from multiple sources.

-----

-----

-----

### Source [4]: https://www.descope.com/learn/post/mcp

Query: What is the Model-Context Protocol (MCP) and how does it facilitate steerable, human-in-the-loop AI agent workflows?

Answer: The **Model Context Protocol (MCP)** uses a client-server architecture, partially inspired by the Language Server Protocol (LSP), to enable standardized interaction between AI applications and external systems. MCP's architecture consists of four core components:

- **Host application:** The LLM or AI application that interacts with users and initiates connections (e.g., Claude Desktop, AI-enhanced IDEs).
- **MCP client:** Embedded within the host application, it manages connections to MCP servers and translates requirements between the host and the MCP protocol.
- **MCP server:** Exposes context and functions to AI applications, typically focusing on a specific integration (e.g., accessing a code repository or database).
- **Transport layer:** Handles communication between clients and servers, supporting both STDIO (for local) and HTTP+SSE (for remote) connections.

All MCP communication uses **JSON-RPC 2.0** for standardized requests, responses, and notifications.

MCP facilitates steerable, human-in-the-loop workflows by allowing LLMs to interactively and programmatically retrieve, update, or act on external data and services. This makes it possible for users to guide or correct AI agent behavior in real time, while ensuring that context and actions remain consistent and auditable across integrations.

-----

-----

-----

### Source [5]: https://developers.openai.com/apps-sdk/concepts/mcp-server/

Query: What is the Model-Context Protocol (MCP) and how does it facilitate steerable, human-in-the-loop AI agent workflows?

Answer: The **Model Context Protocol (MCP)** is an open specification that allows large language model clients to connect to external tools and resources. An MCP server exposes **tools** that a model can call during a conversation, returning results based on specified parameters. The protocol supports returning additional metadata, such as inline HTML, which can be used to render interfaces in applications like ChatGPT.

The MCP protocol standardizes the wire format, authentication, and metadata exchange, enabling the model, server, and user interface to stay in sync. This consistency allows models to reason about external tools in the same way as built-in tools, facilitating seamless, steerable workflows where human users can direct or intervene in the agent's actions as needed.

A minimal MCP server implements three capabilities:
- **List tools:** The server advertises available tools, their input/output schemas, and annotations.
- **Call tools:** The model sends a request to execute a tool with specific arguments.
- **Return components:** Tools can point to UI resources for embedding in the client interface.

The protocol is transport agnostic, supporting both Server-Sent Events and Streamable HTTP, and is the backbone for connecting models to user-facing applications and custom workflows.

-----

-----

</details>

<details>
<summary>What are the key features of LangGraph for building reliable and stateful LLM workflows?</summary>

### Source [6]: https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-frameworks/langchain-langgraph.html

Query: What are the key features of LangGraph for building reliable and stateful LLM workflows?

Answer: LangGraph extends LangChain's capabilities to build complex, **stateful agent workflows** with rich orchestration. Key features include:

- **Component ecosystem**: Extensive library of pre-built components for agent development, accelerating the creation of specialized autonomous agents.
- **Foundation model selection**: Supports a wide range of models (Anthropic Claude, Amazon Nova, OpenAI, etc.), allowing selection based on reasoning needs.
- **LLM API integration**: Standardized interfaces for various large language model providers, enabling flexible deployment.
- **Multimodal processing**: Built-in support for text, image, and audio inputs, empowering agents to handle diverse data types.
- **Graph-based workflows**: Enables definition of agent behaviors as state machines, supporting sophisticated decision logic and stateful interaction.
- **Memory abstractions**: Multiple options for managing short-term and long-term memory, which are essential for maintaining context over extended operations.
- **Tool integration**: Rich integrations with services and APIs to expand agent capabilities.
- **LangGraph platform**: Offers managed deployment and monitoring for production, supporting long-running autonomous agents with reliable operation.

LangGraph is designed for building reliable, stateful LLM workflows with deep support for context, multimodal data, and orchestration[1].

-----

-----

-----

### Source [7]: https://langchain-ai.github.io/langgraph/concepts/why-langgraph/

Query: What are the key features of LangGraph for building reliable and stateful LLM workflows?

Answer: LangGraph’s key features for building reliable and stateful LLM workflows include:

- **Reliability and controllability**: Agents’ actions can be steered using moderation checks and human-in-the-loop approvals. Context is persisted for long-running workflows, ensuring consistent operation.
- **Low-level and extensible primitives**: Developers have granular control without rigid abstractions, enabling highly customizable agents and multi-agent systems tailored to specific roles.
- **First-class streaming support**: Provides token-by-token streaming and streaming of intermediate steps, offering real-time visibility into agent reasoning and actions.
- Tutorials demonstrate how to:
  - Build chatbots that answer questions via web search,
  - Maintain conversation state across calls,
  - Route complex queries to human reviewers,
  - Use custom state for behavior control,
  - “Rewind” and explore alternative conversation paths.

LangGraph emphasizes reliability, transparency, and extensibility for robust stateful agent workflows[2].

-----

-----

-----

### Source [8]: https://langchain-ai.github.io/langgraph/

Query: What are the key features of LangGraph for building reliable and stateful LLM workflows?

Answer: LangGraph is a **low-level orchestration framework** for building, managing, and deploying long-running, stateful agents. Core benefits include:

- **Durable execution**: Agents persist through failures and can run for extended periods, resuming exactly where they left off.
- **Human-in-the-loop**: Enables inspection and modification of agent state at any point, allowing for oversight and intervention.
- **Comprehensive memory**: Supports both short-term working memory and long-term persistent memory for truly stateful agents.
- **Debugging tools**: Integration with LangSmith provides visualization and tracing of execution paths, state transitions, and runtime metrics.
- **Production-ready deployment**: Infrastructure is designed for scalable, stateful, long-running workflows.

LangGraph avoids rigid abstractions, allowing developers full control over agent prompts and architecture for reliability and customization[3].

-----

-----

-----

### Source [9]: https://docs.langchain.com/oss/python/langgraph/overview

Query: What are the key features of LangGraph for building reliable and stateful LLM workflows?

Answer: LangGraph focuses on critical agent orchestration capabilities:

- **Durable execution**: Agents persist and resume after interruptions.
- **Human-in-the-loop**: Human oversight is integral, with the ability to inspect and modify agent state at any execution point.
- **Comprehensive memory**: Agents maintain both short-term and long-term memory for stateful operation.
- **Debugging with LangSmith**: Visualization of execution paths, state transitions, and runtime metrics for deep observability.
- **Production-ready deployment**: Scalable infrastructure supports deployment of sophisticated stateful workflows.

Additionally, LangGraph integrates seamlessly with LangChain, LangSmith, and LangGraph Platform, providing a suite of tools for agent evaluation, observability, deployment, and rapid prototyping[4].

-----

-----

-----

### Source [10]: https://www.langchain.com/langgraph

Query: What are the key features of LangGraph for building reliable and stateful LLM workflows?

Answer: LangGraph provides essential features for **reliable, stateful LLM workflows**:

- **Persistent context for long-term interactions**: Built-in memory stores conversation history and maintains context across sessions, enabling personalized and coherent agent behavior.
- **First-class streaming support**: Native token-by-token streaming and streaming of intermediate steps offer real-time feedback of agent reasoning and actions.
- **API for dynamic, interactive experiences**: The LangGraph Platform API enables delivery of dynamic user experiences, especially for conversational agents and complex automation.

LangGraph’s features support robust context management, interactive user experiences, and reliable long-term agent operation[5].

-----

-----

</details>

<details>
<summary>Best practices for designing agentic systems with distinct exploratory research agents and reliable production workflows.</summary>

### Source [11]: https://www.coalitionforsecureai.org/announcing-the-cosai-principles-for-secure-by-design-agentic-systems/

Query: Best practices for designing agentic systems with distinct exploratory research agents and reliable production workflows.

Answer: The CoSAI Principles for Secure-by-Design Agentic Systems emphasize three foundational best practices for designing agentic systems with both exploratory research agents and reliable production workflows:

- **Enable Human Oversight and Shared Accountability:** Architect agentic systems for meaningful human control, ensuring clear and shared accountability throughout the system lifecycle. Employ risk-based actionable controls and oversight aligned with business outcomes and risk tolerance. Focus on defining rules of engagement and data governance rather than micromanaging agent actions. Human intervention should be reserved for high-risk or anomalous events, and shared accountability must be established among all parties—model producers, agent developers, service implementers, and human principals—before system development begins.

- **Maximize Oversight While Minimizing Intervention:** Implement clear data governance policies, operational no-go zones, and risk-based thresholds that trigger human review. Use strategic points of control to ensure governance without sacrificing agentic system speed and efficiency.

- **Implement Agile, Risk-Based Governance:** Apply governance measures proportional to task risk, such as simple logging for low-risk tasks and multi-party approval for high-risk ones. Different workloads (e.g., research vs. production) require different controls.

- **Bounded and Resilient Design:** Limit agent capabilities and resource access with strict, purpose-specific entitlements. Employ robust cybersecurity controls and continuous validation of alignment with intended outcomes. Define failure modes in advance (e.g., fail-safe, degrade gracefully, fail-fast) to handle both expected and adversarial failures.

- **Adapt Zero Trust Security:** Apply micro-segmentation and continuous monitoring of agent actions. Treat each agent action as a potential threat vector, enforcing automated policy boundaries in real time.

- **Granular Entitlements and Identity Management:** Clearly define and enforce what data and tools agents can access, and manage both agent and human identities accordingly.

- **Design for Failure:** Architect explicit failure modes and strategies to minimize impact, including handling Byzantine (unexpected or adversarial) failures.

-----

-----

-----

### Source [12]: https://azure.microsoft.com/en-us/blog/agent-factory-the-new-era-of-agentic-ai-common-use-cases-and-design-patterns/

Query: Best practices for designing agentic systems with distinct exploratory research agents and reliable production workflows.

Answer: Best practices for agentic system design highlighted in the Agent Factory series include:

- **Planning Pattern:** Use planning agents to break complex, high-level goals into actionable tasks and adapt dynamically as requirements change. Planning agents ensure robustness by tracking progress, executing in phases, and verifying completion before advancing, which helps bridge the gap between exploratory research and reliable production.

- **Tool Use and Reflection:** Combine planning with tool utilization and reflective checks to generate flexible workflows. Plans can be dynamically created by large language models (LLMs) or follow predefined sequences, depending on reliability needs.

- **Multi-Agent Pattern:** Employ specialized agents for different workflow stages, coordinated by an orchestrator. This modular, team-based design promotes agility, scalability, and clear governance, with each agent responsible for a distinct workflow component. Common orchestration patterns include:
  - Sequential orchestration: Agents refine outputs stepwise.
  - Concurrent orchestration: Agents operate in parallel, merging results.
  - Group chat/maker-checker: Agents debate and validate outputs collaboratively.
  - Dynamic handoff: Agents triage or route tasks in real time.
  - Manager agent orchestration: A manager coordinates subtasks until completion.

- **Productionization and Automation:** Automate routine processes and standardize development cycles, reducing manual overhead and increasing reliability. For example, orchestrators can cut requirements and QA timelines by automating task coordination and validation.

These patterns enable distinct boundaries between exploratory (flexible, adaptive) research agents and reliable, standardized production workflows, while facilitating collaboration and robust oversight.

-----

-----

-----

### Source [13]: https://akka.io/blog/agentic-ai-architecture

Query: Best practices for designing agentic systems with distinct exploratory research agents and reliable production workflows.

Answer: Akka's guide to agentic AI architecture stresses several layered design practices for balancing research exploration with production reliability:

- **Orchestration Components:** Use workflow engines (e.g., Temporal, Airflow) to manage how agents interconnect and how their lifecycle is handled, including data input and updates. Orchestration also addresses cost considerations and system evolution.

- **Observability Layer:** Implement robust monitoring, tracing, and logging at every stage (perception, reasoning, action). Maintain detailed attribution of decisions to specific agents, versions, or contexts. Real-time anomaly detection (e.g., hallucinations, misuse) is essential, and observability must not compromise privacy or compliance.

- **Security Layer:** Protect sensitive data throughout the agentic system, both at rest and in transit. Employ sandboxing, rate-limiting, and runtime constraints to prevent misuse. Enforce strict policies for agent actions, with strong authentication, authorization, and revocation for all agent-driven API calls.

- **Governance Layer:** Ensure compliance with both internal and external policies (GDPR, DORA, SOC 2). Enforce policies across models, memory, orchestration, and logging. Include approval workflows, audit trails, and explainability requirements for agent behaviors, and clearly define governance roles and system visibility.

- **Challenges:** Key implementation challenges include selecting suitable LLMs and integrating all these layers to balance flexibility for research with the robustness and compliance needed in production.

These layered practices provide a blueprint for creating agentic systems where exploratory research agents can be safely developed and tested, while production workflows remain secure, observable, and governed.

-----

-----

</details>

<details>
<summary>How to implement state persistence and checkpointing in multi-stage LLM agent systems for auditability and reliability.</summary>

### Source [14]: https://docs.koog.ai/agent-persistence/

Query: How to implement state persistence and checkpointing in multi-stage LLM agent systems for auditability and reliability.

Answer: **Koog Agent Persistence** provides checkpoint functionality for multi-stage LLM agent systems by allowing you to save and restore an agent's state at specific execution points. This is essential for auditability and reliability, as it enables:

- **Resuming execution:** The agent can continue from a saved state after interruption.
- **Rolling back:** You can revert to previous states if errors are detected.
- **Session persistence:** States can be retained across different sessions.

**Key checkpoint contents**:
- Message history (all user, system, assistant, and tool interactions)
- The current node being executed
- Input data for the current node
- Timestamp of checkpoint creation
- Each checkpoint has a unique ID, linked to the agent

**Configuration and Usage**:
- Each node in the agent’s strategy must have a unique name for checkpointing to work.
- The persistence feature is installed via configuration, where you select a storage provider:
  - `InMemoryPersistenceStorageProvider`: For in-memory, ephemeral checkpoint storage.
  - `FilePersistenceStorageProvider`: For durable, file-based storage.
  - `NoPersistenceStorageProvider`: Disables persistence.
  - **Custom providers** can be implemented for storage in databases or remote systems.

**Continuous persistence** can be enabled so checkpoints are automatically created after each node execution, ensuring granular recoverability.

**Creating/restoring checkpoints**:
- Checkpoints are created programmatically via the API, specifying node ID, input data, and checkpoint ID.
- Extension functions and advanced APIs allow setting the execution point directly, enabling fine-grained control over recovery, rollback, and time travel.

**Custom storage providers** are supported by implementing the `PersistenceStorageProvider` interface, allowing integration into enterprise storage systems for audit logs and compliance.

-----

-----

-----

### Source [15]: https://docs.databricks.com/aws/en/generative-ai/agent-framework/stateful-agents

Query: How to implement state persistence and checkpointing in multi-stage LLM agent systems for auditability and reliability.

Answer: **Databricks Mosaic AI Agent Framework** supports stateful agents with checkpointing for auditability and reliability in multi-stage LLM systems.

- **Stateful agents** use thread IDs to maintain context, and **checkpointing** saves the agent’s state at specific points.
- **Time travel** enables replaying or forking conversation history, which is critical for auditing, debugging, and exploring alternative reasoning paths.

**Key features:**
- **Checkpointing and time travel** allow:
  - Observing exactly what the agent knew and did at each step.
  - Debugging mistakes by identifying where errors occurred.
  - Exploring alternative conversation paths while preserving the original state.
- **Lakebase** is used as the persistent memory store.

**Implementation**:
- The agent retrieves and updates checkpoint history using functions in the agent class (example: `LangGraphResponsesAgent`), which interface with a persistent backend (e.g., Postgres or Lakebase).
- When resuming from a checkpoint, a new conversation fork is created, preserving audit history.

**Typical checkpoint data includes**:
- Checkpoint ID
- Timestamp
- Thread ID
- Next nodes in the conversation graph
- Message count and content

This model enables robust audit trails and reliable restoration for complex, multi-stage LLM workflows.

-----

-----

-----

### Source [16]: https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/stateful-agents

Query: How to implement state persistence and checkpointing in multi-stage LLM agent systems for auditability and reliability.

Answer: **Azure Databricks** (via Mosaic AI Agent Framework and LangGraph) provides formal mechanisms for state persistence and checkpointing in LLM agent systems.

- **Stateful agents** maintain context across interactions using thread IDs.
- **Checkpointing** allows saving the agent’s state at any point for future restoration.
- **Time travel** enables replaying and exploring conversations from saved states, supporting traceability and auditability.

**Benefits for auditability and reliability:**
- Audit trails: Analyze agent knowledge and actions at each step.
- Debugging: Pinpoint errors and their causes within the execution flow.
- Alternative exploration: Test different conversation paths from the same state.

**Implementation requirements:**
- A persistent storage backend (Lakebase).
- Configuration of agent memory and thread management.

**Workflow:**
- Checkpoints are created and stored at key stages.
- Agents can be rolled back or resumed from any checkpoint, maintaining a complete execution history for compliance and debugging.

This approach supports enterprise requirements for explainability, repeatability, and resilience in production LLM agent deployments.

-----

-----

-----

### Source [17]: https://slack-chats.kotlinlang.org/t/29749269/looking-at-implementing-my-own-https-docs-koog-ai-agent-pers

Query: How to implement state persistence and checkpointing in multi-stage LLM agent systems for auditability and reliability.

Answer: The **PersistencyStorageProvider** interface (as referenced in Koog) supports custom implementations for agent state persistence and checkpoint management:

- **Methods**:
  - `getCheckpoints(filters: Map<String, Any>? = null)`: Retrieve checkpoints with filtering (e.g., by date, status, priority).
  - `saveCheckpoint(agentCheckpointData, filters: Map<String, Any>? = null)`: Save a checkpoint, with optional overwrite or other flags.
  - `getLatestCheckpoint(filters: Map<String, Any>? = null)`: Retrieve the most recent checkpoint, with filtering (e.g., by agentId).

**Usage example:**
- Filtering checkpoints by date, status, or agent attributes.
- Saving checkpoints with overwrite options or other custom flags.
- Retrieving the latest checkpoint for a specific agent for fast recovery or audit.

This interface enables flexible, fine-grained control over checkpoint storage and retrieval, essential for both reliability and auditability in complex LLM agent systems.

-----

-----

</details>

<details>
<summary>What are the differences between agentic AI frameworks like AutoGen or CrewAI versus workflow orchestration frameworks like LangGraph?</summary>

### Source [18]: https://www.ibm.com/think/tutorials/build-agentic-workflows-langgraph-granite

Query: What are the differences between agentic AI frameworks like AutoGen or CrewAI versus workflow orchestration frameworks like LangGraph?

Answer: LangGraph is described as a framework for developing **AI-driven workflows** by representing AI models as *stateful agents* within a computational graph. Its main distinction from traditional automation is its ability to perform **dynamic, context-aware task execution** using real-time logic and memory. In LangGraph, each step (node) is handled by an agent, often powered by an LLM, and nodes move between states based on model outputs or conditional logic. This allows for dynamic, decision-driven workflows rather than rigid, linear pipelines.

Key features highlighted include:
- Each agent behavior is a distinct node in the graph.
- Algorithms or model outputs determine the next step, allowing for adaptive flows.
- Shared state between nodes enables the system to preserve and use context.
- The graph structure allows for visualization, debugging, and modularity.
- **Cyclical graphs** (conditional edges) enable loops and conditional branching, which are crucial for iterative or adaptive tasks.

LangGraph is positioned as a **workflow orchestration framework** designed specifically for scalable, modular, and dynamic AI workflows, distinct from traditional agentic frameworks by its focus on graph-based, stateful task execution.

-----

-----

-----

### Source [19]: https://duplocloud.com/blog/langchain-vs-langgraph/

Query: What are the differences between agentic AI frameworks like AutoGen or CrewAI versus workflow orchestration frameworks like LangGraph?

Answer: LangGraph is contrasted with linear frameworks like LangChain and is described as ideal for **complex, multi-agent systems with dynamic, non-linear workflows**. LangGraph models workflows as cyclical graphs of nodes and edges, where nodes can be LLM calls, tools, or custom functions, and edges can be conditional. This structure supports loops, conditional branching, and multi-agent collaboration—capabilities not native to linear, chain-based orchestration frameworks.

Key differences highlighted:
- **LangChain** (linear): Best for straightforward, sequential flows (DAGs). Limited support for complex branching and iteration.
- **LangGraph** (dynamic): Built for stateful, multi-agent systems with cycles, conditional logic, and dynamic routing based on runtime conditions.

LangGraph is presented as a **workflow orchestration framework** designed to handle non-linear, adaptive, and collaborative AI workflows, distinguishing it from simpler, agentic frameworks built around sequential logic.

-----

-----

-----

### Source [20]: https://langchain-ai.github.io/langgraph/

Query: What are the differences between agentic AI frameworks like AutoGen or CrewAI versus workflow orchestration frameworks like LangGraph?

Answer: LangGraph is described as a **low-level orchestration framework** for building, managing, and deploying long-running, stateful agents. It provides infrastructure for durable execution, enabling agents to persist through failures and run over extended periods, resuming from the exact point of interruption. LangGraph also offers human-in-the-loop capabilities, comprehensive memory (both short-term and long-term), and robust debugging and production deployment features.

Notably, LangGraph does not abstract away prompts or architecture, giving developers full control over agent design and workflow logic. Its focus is on supporting *any* long-running, stateful workflow or agent, making it suitable for both agentic systems and broader workflow orchestration tasks.

The emphasis is on building **stateful, persistent, and debuggable workflows and agents**, which distinguishes it from frameworks that solely focus on agent autonomy or prompt chaining.

-----

-----

-----

### Source [21]: https://docs.langchain.com/langgraph-platform/faq

Query: What are the differences between agentic AI frameworks like AutoGen or CrewAI versus workflow orchestration frameworks like LangGraph?

Answer: According to the official FAQ, **LangGraph is an orchestration framework for complex agentic systems** and is characterized as more low-level and controllable than LangChain agents. This means LangGraph is used to orchestrate the interactions of multiple agents or workflow steps, rather than just defining the behavior of a single autonomous agent. Its design provides greater flexibility and explicit control for developers building sophisticated, multi-agent, or multi-step systems.

-----

-----

-----

### Source [22]: https://blog.langchain.com/how-to-think-about-agent-frameworks/

Query: What are the differences between agentic AI frameworks like AutoGen or CrewAI versus workflow orchestration frameworks like LangGraph?

Answer: LangGraph is described as a **low-level orchestration framework for building agentic systems**. It supports the construction of workflows, agents, and hybrid systems that fall between classic agentic and workflow paradigms. This flexibility allows developers to implement both agentic behavior (autonomous, decision-making agents) and orchestrated, multi-step workflows within the same infrastructure.

LangGraph's key strength is orchestrating **stateful, long-running, and complex workflows** that may involve multiple agents, tools, or conditional logic, as opposed to frameworks that confine developers to a specific agentic or linear workflow model.

-----

-----

</details>

<details>
<summary>Limitations of off-the-shelf deep research tools for developers requiring custom agent behavior.</summary>

### Source [23]: https://sep.com/blog/when-to-reach-for-custom-artificial-intelligence-solutions-vs-off-the-shelf/

Query: Limitations of off-the-shelf deep research tools for developers requiring custom agent behavior.

Answer: Off-the-shelf deep research tools often present significant **limitations for developers requiring custom agent behavior** due to several types of misalignments:

- **Regulatory & Security Misalignment:** Off-the-shelf AI solutions may not align with strict corporate security, intellectual property (IP), or regulatory requirements. For example, if an organization needs to operate within an air-gapped environment or has specific IP concerns, externally hosted or generative AI solutions cannot be used.

- **AI Approach Mismatch:** These tools may not support the specific type of AI approach required by the project, such as generative AI, machine learning, or computer vision, especially if the problem falls outside the typical use cases these tools target.

- **Input/Output Format Constraints:** Off-the-shelf systems often expect data in certain formats and produce outputs in fixed formats (e.g., prose instead of structured text), causing additional development overhead to transform data or adapt workflows.

- **Accuracy and Adaptability:** If the system’s accuracy does not meet project requirements, developers may need to build extra tools to filter or validate output, or train end-users to watch for errors, increasing ongoing costs and complexity.

- **Cost of Adaptation:** While some mismatches can be overcome (with retraining, extra tooling, or process changes), each adaptation adds development time, operational cost, or manual effort, potentially negating the advantages of using off-the-shelf solutions for custom agent behavior.

These factors collectively indicate that off-the-shelf tools frequently fall short when highly customized agent behavior or strict compliance is required.

-----

-----

-----

### Source [24]: https://dlabs.ai/blog/machine-learning-off-the-shelf-models-or-custom-build-pros-and-cons/

Query: Limitations of off-the-shelf deep research tools for developers requiring custom agent behavior.

Answer: Off-the-shelf deep research tools, such as third-party ML services and APIs, have several **limitations for developers needing custom agent behavior**:

- **Repetitiveness and Manual Work:** Off-the-shelf tools typically require users to upload data and accept results as provided. For repetitive or complex workflows, this manual process can become tedious, especially if the tool lacks bulk processing features.

- **Limited Workflow Integration:** If the results from an off-the-shelf service are just one step in a multi-stage process, integrating these results can become cumbersome and manual, hindering automation and seamless workflow integration.

- **Lack of Flexibility:** These solutions are designed for general use cases and often cannot be tailored to perform nuanced or domain-specific tasks, limiting their utility for specialized research agent behavior.

- **Cost Considerations:** Off-the-shelf tools may be cost-prohibitive for high-frequency or large-scale use cases, as many operate on a subscription or pay-per-use basis with variable costs.

These limitations highlight why developers with specialized requirements may find off-the-shelf research tools inadequate for creating custom agent behaviors.

-----

-----

-----

### Source [25]: https://botscrew.com/blog/custom-ai-development-vs-off-the-shelf-ai/

Query: Limitations of off-the-shelf deep research tools for developers requiring custom agent behavior.

Answer: Off-the-shelf AI solutions impose **several integration, performance, and customization limitations** relevant to developers seeking custom agent behavior:

- **Integration Restrictions:** These tools offer standard integration (APIs, plugins), but only in ways predefined by the vendor. If a developer’s systems or workflows are complex or unique (e.g., legacy systems, custom data flows), the off-the-shelf tool may not integrate deeply or flexibly, potentially requiring process adjustments or resulting in siloed systems.

- **Data Control and Security:** Data handling is managed by the vendor, introducing potential compliance and privacy issues, especially if sensitive data must be sent to external servers and handled under generic contract terms.

- **Limited Customization:** Off-the-shelf tools provide a fixed set of features and algorithms. Their models are trained for broad, common scenarios, and customization for niche use cases or the incorporation of unique business logic is limited or impossible.

- **Performance on Specialized Tasks:** While performance may be strong on general tasks, effectiveness drops for domain-specific problems where models are not fine-tuned to the organization’s data or needs. For instance, generic language models may misunderstand industry-specific terminology or fail to support proprietary workflows essential for custom agent behavior.

These constraints make off-the-shelf solutions less suitable when precise control, integration, and adaptation are critical for research agent development.

-----

-----

-----

### Source [26]: https://marutitech.com/custom-ai-vs-off-the-shelf-for-business/

Query: Limitations of off-the-shelf deep research tools for developers requiring custom agent behavior.

Answer: Off-the-shelf AI tools often fail to meet **latency, accuracy, and integration requirements** for custom agent behavior:

- **Latency and Accuracy:** APIs might have higher response times or lower accuracy, especially if the models are trained on data unrelated to the target domain, resulting in poor performance for specialized research tasks.

- **Integration Challenges:** These solutions may lack essential integrations, requiring significant setup and customization effort. If an organization’s data is different from the datasets used to train the model, performance suffers, and developers may need to undertake extensive retraining or annotation, which undermines the convenience of off-the-shelf tools.

- **Domain Mismatch:** For niche or emerging applications, off-the-shelf AI may not exist, or existing models lack the required variables and logic (e.g., unique environmental factors in logistics or retail operations outside the region where the model was developed).

- **Resource Overhead:** Using off-the-shelf solutions may require substantial manual labeling, annotation, and collaboration with consultants to adapt the model, leading to increased resource expenditure.

These issues contribute to the inadequacy of off-the-shelf solutions for developers building agents with highly specialized or domain-specific behaviors.

-----

-----

-----

### Source [27]: https://artoonsolutions.com/custom-ai-solutions-vs-off-the-shelf-ai-tools/

Query: Limitations of off-the-shelf deep research tools for developers requiring custom agent behavior.

Answer: Off-the-shelf AI tools are generally **not flexible enough to address specific or niche requirements**, which is a major barrier for developers needing custom agent behaviors:

- **Lack of Flexibility:** These tools are designed to serve general business needs and may not provide the customization necessary for specialized applications or industries.

- **Inadequacy for Niche Markets:** If a business or research requirement is highly specialized, off-the-shelf solutions usually lack the capacity to adapt, necessitating the development of custom AI to achieve the desired outcomes.

This lack of adaptability means off-the-shelf tools frequently cannot support the development of agents that must operate in unique, highly controlled, or innovative ways.

-----

</details>

<details>
<summary>Techniques for automated context ingestion from diverse sources like GitHub, YouTube, and local files for LLM agents.</summary>

### Source [28]: https://arxiv.org/html/2507.13334v1

Query: Techniques for automated context ingestion from diverse sources like GitHub, YouTube, and local files for LLM agents.

Answer: Automated context ingestion for LLM agents is situated within the emerging discipline of **Context Engineering**, which systematically optimizes the information provided to language models. The process is composed of three foundational components:

- **Context Retrieval and Generation:** This involves prompt-based generation, retrieval of external knowledge (such as code from GitHub, transcripts from YouTube, or local files), and dynamic assembly of context. For GitHub, automated retrieval techniques may include using APIs or scrapers to fetch code snippets, issues, or documentation. For YouTube, context ingestion commonly involves transcript extraction via APIs or speech-to-text tools. Local files require directory crawling and parsing based on file type (text, code, etc.).

- **Context Processing:** Once retrieved, data undergoes transformation and optimization. This includes handling long sequences (splitting, chunking), self-refinement (iterative summarization or filtering), and integrating multimodal or structured information (like combining code, text, and multimedia content into coherent context windows).

- **Context Management:** Efficient organization is crucial due to LLM context window limits. Techniques include memory hierarchies (short-term vs. long-term memory), compression, deduplication, and context prioritization. For multi-source ingestion, context engineering systems coordinate these processes, often using modular or agentic architectures that route, compress, or summarize inputs from GitHub, YouTube, and local files before feeding them to the LLM.

These components are implemented in various architectures such as Retrieval-Augmented Generation (RAG), memory systems, and tool-integrated reasoning modules. Each architecture employs standardized pipelines for ingesting, processing, and managing heterogeneous data sources to maximize LLM utility and performance.

-----

-----

-----

### Source [29]: https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/

Query: Techniques for automated context ingestion from diverse sources like GitHub, YouTube, and local files for LLM agents.

Answer: The NVIDIA blog describes LLM inference as a process involving two main phases—**prefill** (processing input) and **decode** (generating output). For automated context ingestion, the key technical considerations are:

- In the prefill phase, all **input tokens** (from sources like GitHub, YouTube transcripts, or local files) are processed in parallel, which is highly efficient for large input batches.
- Each **context source** must be tokenized before ingestion. This involves converting raw text/code into the atomic units that LLMs understand.
- Handling different tokenizers is important, especially when ingesting diverse sources—GitHub code, YouTube transcript text, and local file data may be encoded differently.
- The **decode phase** is memory-bound, requiring efficient management of keys and values (internal LLM states) as output is generated token by token.

Optimizing ingestion from heterogeneous sources thus involves:
- Standardizing tokenization across source types.
- Efficient batching and caching of input data.
- Managing memory and attention span for large or multimodal contexts.

-----

-----

-----

### Source [30]: https://rpc.cfainstitute.org/research/the-automation-ahead-content-series/retrieval-augmented-generation

Query: Techniques for automated context ingestion from diverse sources like GitHub, YouTube, and local files for LLM agents.

Answer: Retrieval-Augmented Generation (RAG) is a core technique for automated **context ingestion** from diverse data sources:

- **RAG pipelines** allow the integration of external, custom documents (such as code from GitHub, YouTube transcripts, or local files) into the LLM's context window.
- The process relies on **embedding models**: source data is converted into vector representations that capture semantic meaning. For example, a GitHub README, a YouTube transcript, or a local Markdown file are all embedded, indexed, and stored.
- At query time, the most relevant vectors (chunks of text/code) are retrieved based on similarity to the user query, then concatenated and injected into the LLM context.
- This approach allows **real-time, relevant information** (including recent code changes or new videos) to be surfaced, addressing the static knowledge limitations of LLMs.

Key requirements for ingestion:
- Structuring and chunking source documents appropriately.
- Extracting and indexing metadata (e.g., file paths, timestamps, authorship).
- Optimizing retrieval algorithms to surface the most contextually useful segments for each query.

-----

-----

</details>

<details>
<summary>Best practices for designing a hybrid AI system with an exploratory research agent and a separate, reliable writing workflow.</summary>

### Source [31]: https://www.geeksforgeeks.org/artificial-intelligence/what-is-hybrid-ai-and-its-architecture/

Query: Best practices for designing a hybrid AI system with an exploratory research agent and a separate, reliable writing workflow.

Answer: A hybrid AI system integrates both rule-based (symbolic) and learning-based (subsymbolic) components to maximize adaptability, explainability, and efficiency. The key architectural components include:

- **Data**: Quality and relevance of input data are critical for the growth and accuracy of the machine learning elements.
- **Knowledge Base**: This structured repository contains facts, rules, and relationships, serving as the foundation for rule-based logic and reasoning.
- **Rule Engine**: Executes predefined logic by processing incoming data and applying rules to generate actions or outputs, supporting system reliability.
- **Machine Learning Algorithms**: Decision trees, neural networks, and reinforcement learning models enable the system to learn patterns and make predictions or decisions from data, supporting exploratory research functions.
- **User Interface**: Facilitates human interaction, allowing users to input data, review results, and provide feedback, essential for collaborative workflows.
- **Integration Layer**: Links all components, enabling communication and data exchange between rule-based and machine learning modules.
- **Feedback Mechanisms**: Allow the system to update its knowledge base and adapt rules and models based on new data or human feedback.

Hybrid AI systems can be structured as expert systems augmented with machine learning, or as learning systems constrained by explicit rules. For research agents, the exploratory component leverages machine learning for discovery, while the writing workflow benefits from rule-based logic for reliability and transparency. Combining symbolic and subsymbolic reasoning ensures broader comprehension and mitigates risks like bias. Hybrid architectures use ensemble approaches, knowledge graphs, and modular integration to optimize both exploratory and reliable outputs.

-----

-----

-----

### Source [32]: https://www.fim-rc.de/Paperbibliothek/Veroeffentlicht/4961/id-4961.pdf

Query: Best practices for designing a hybrid AI system with an exploratory research agent and a separate, reliable writing workflow.

Answer: Designing human-AI hybrid systems involves two main phases: construction (defining requirements, selecting algorithms, integrating agents) and execution (operational collaboration between humans and AI). Best practices identified for hybrid AI construction include:

- **Rapid Testing of Use Cases**: Establish a project environment that allows for quick prototyping and evaluation of multiple hybrid use cases to assess their strategic fit and usefulness. Early feedback drives iterative refinement of system design.
- **Alignment with Organizational Strategy**: Ensure that the technical development aligns with the intended organizational use case, addressing both technical and operational requirements.
- **Training Human Agents**: Develop AI-specific competencies among human users (e.g., prompting, critical assessment of outputs) to support effective collaboration.
- **Code-Free Communication Interfaces**: Implement user-friendly interfaces (such as conversational AI) to facilitate seamless interaction between human users and AI agents.

During execution, maintaining alignment between human and AI mental models is crucial. Technical implementers must understand both the algorithm's logic and user expectations to foster effective collaboration. Organizational structures must support ongoing adaptation and integration of human and AI capabilities throughout the system's lifecycle.

-----

-----

-----

### Source [33]: https://www.mantech.com/blog/best-practices-for-architecting-ai-systems/

Query: Best practices for designing a hybrid AI system with an exploratory research agent and a separate, reliable writing workflow.

Answer: Architecting robust AI systems requires adherence to several foundational best practices:

- **Continuous Integration and Delivery (CI/CD)**: Use CI/CD pipelines to automate testing and deployment of AI models and system components, ensuring rapid, reliable updates and integration.
- **Robust System Integration**: Design for seamless interoperability with existing software, databases, and hardware. This is essential for hybrid systems that need to merge exploratory AI agents with reliable writing workflows, ensuring consistent communication and data exchange.
- **Modular Architecture**: Structure the system with independent, service-oriented modules connected via well-defined interfaces. This enables the separate development and maintenance of the research agent and writing workflow, supporting scalability and reliability.
- **Standardized Protocols and APIs**: Implement standardized communication interfaces to reduce compatibility issues and simplify integration between AI modules and other system components.
- **Data Management and Governance**: Enforce strong data governance and management practices to ensure data quality, consistency, and security across the hybrid system, meeting regulatory and operational requirements.

By following these principles, hybrid AI systems can achieve both exploratory flexibility and reliable, maintainable workflows.

-----

-----

</details>

<details>
<summary>Implementing self-critique and reflection loops in LangGraph to reduce 'AI slop' and enforce stylistic rules.</summary>

### Source [34]: https://cognitiveclass.ai/courses/build-a-self-reflective-deep-research-agent-using-langgraph

Query: Implementing self-critique and reflection loops in LangGraph to reduce 'AI slop' and enforce stylistic rules.

Answer: The course describes how to use LangGraph to build a self-reflective deep research agent that integrates self-critique mechanisms for continuous learning and output refinement. It focuses on creating workflows where the agent critiques its responses using **structured feedback** and external search, supporting research-backed validation and graph-based iterative workflows. The agent’s design includes:

- **Self-critique mechanisms:** Reflexion agents analyze their own outputs, using structured feedback loops to identify errors, inaccuracies, or stylistic deviations.
- **External validation tools:** These tools are incorporated to enhance factual accuracy and ensure that the agent's responses are backed by research.
- **Graph-based workflows:** LangGraph is leveraged to represent multi-step tasks and decision points, enabling cycles where the agent can re-evaluate and improve prior outputs.
- **Automated performance optimization:** The agent adapts and improves its performance through iterative refinement, learning from mistakes and external feedback.
- **Error handling and schema validation:** Robustness is ensured by validating outputs against expected schemas and limiting iteration counts to prevent infinite loops.

These components collectively enable the agent to reduce "AI slop"—that is, low-quality or off-target outputs—and enforce desired stylistic and factual standards by continuously critiquing and adjusting its responses based on both internal and external criteria.

-----

-----

-----

### Source [35]: https://learnopencv.com/langgraph-self-correcting-agent-code-generation/

Query: Implementing self-critique and reflection loops in LangGraph to reduce 'AI slop' and enforce stylistic rules.

Answer: This source details the implementation of a self-correcting agent using LangGraph for code generation, providing a concrete example of self-critique and reflection loops. The agent operates as follows:

- **Generation:** The LLM generates an initial output (e.g., code).
- **Execution/Testing:** The output is executed or tested to check for errors or undesired behaviors.
- **Evaluation and Reflection:** If the output fails, the agent reflects on what went wrong, using errors or feedback as input for the next iteration.
- **Regeneration:** Based on the reflection, the agent produces a revised solution. This loop continues until the output meets the desired criteria or a maximum number of iterations is reached.

The agent’s state is tracked using a custom `AgentState` structure containing fields such as error flags, messages (maintaining full conversation history), output, explanations, and iteration counts. This stateful approach allows the agent to:

- Maintain a history of critiques and revisions.
- Enforce iteration limits to avoid infinite loops.
- Use conditional graph edges in LangGraph to route control flow based on success or failure of outputs.

This cycle effectively reduces poor-quality AI outputs and can be extended to enforce specific stylistic rules by integrating additional reflective checks in the evaluation phase.

-----

-----

-----

### Source [36]: https://blog.langchain.com/agentic-rag-with-langgraph/

Query: Implementing self-critique and reflection loops in LangGraph to reduce 'AI slop' and enforce stylistic rules.

Answer: This LangChain blog post discusses implementing **self-reflective retrieval-augmented generation (RAG)** with LangGraph, emphasizing corrective and reflective loops to enhance output quality. The approach applies ideas from self-reflective RAG research, such as:

- **Retrieval evaluation:** The agent assesses the quality and relevance of retrieved documents, assigning confidence scores.
- **Conditional logic:** If retrieval is ambiguous or irrelevant, the agent supplements with additional web search, demonstrating a decision point and a corrective loop in the workflow.
- **Knowledge refinement:** Retrieved documents can be partitioned, graded, and filtered to ensure only relevant content is used, though this is presented as an optional enhancement.

LangGraph’s state machine approach supports the design of such flows with:

- **Explicit decision points:** For example, using Pydantic models to structure outputs and enforce consistent binary logic in conditional routing of graph edges.
- **Loops and retries:** The agent can be configured to revisit prior steps (e.g., retrieval, generation) based on self-assessment, thus enabling iterative self-correction and output improvement.

While the main focus here is RAG, the same principles—structured self-assessment, conditional re-routing, and iterative refinement—are applicable to enforcing stylistic rules and reducing off-target outputs in any multi-step AI workflow built with LangGraph.

-----

-----

-----

### Source [37]: https://activewizards.com/blog/a-deep-dive-into-langgraph-for-self-correcting-ai-agents

Query: Implementing self-critique and reflection loops in LangGraph to reduce 'AI slop' and enforce stylistic rules.

Answer: This article explains the "Generator-Critic" loop as an architectural pattern for implementing self-correcting agents with LangGraph. The key steps include:

- **State definition:** The agent maintains a shared state containing the problem statement, current answer, critique history, and revision count.
- **Generator node:** This node produces an answer using the problem statement and critique history.
- **Critic node:** This node evaluates the current answer for correctness, relevance, or adherence to stylistic rules. If the answer is unsatisfactory, it adds feedback to the critique history.
- **Looping:** If the critic is not satisfied, the flow returns to the generator node for another attempt, using the updated critique history for improvement.

The critique history allows for structured, cumulative feedback, helping the agent learn from prior mistakes and iteratively enforce both factual accuracy and stylistic requirements. The revision number (iteration count) is used to prevent infinite loops.

By separating generation and critique into distinct nodes and structuring their interaction as a loop, LangGraph enables robust, self-correcting workflows suitable for reducing undesirable AI behaviors (“AI slop”) and enforcing custom output standards.

-----

-----

</details>

<details>
<summary>Using Perplexity API for iterative, multi-round research cycles in a custom AI agent.</summary>

### Source [38]: https://zuplo.com/learning-center/perplexity-api

Query: Using Perplexity API for iterative, multi-round research cycles in a custom AI agent.

Answer: The Perplexity API enables integration of sophisticated conversational AI features, including real-time online searches with source citations, into custom applications. For iterative, multi-round research cycles in a custom AI agent, the API offers several important capabilities:

- **Model Selection:** Choose from models like sonar-pro (advanced search and comprehensive answers), sonar-small/medium (for basic queries), mistral-7b, codellama-34b (for code), and llama-2-70b.
- **Real-Time Research:** Unlike many competitors, Perplexity incorporates real-time search and can return up-to-date, attributed information, which is essential for multi-step research workflows.
- **Conversation Management:** The API follows conventions similar to OpenAI's chat format, supporting multi-turn conversations by managing a message history (system, user, assistant roles). This enables agents to maintain context over several research rounds.
- **Integration and Error Handling:** The API is compatible with OpenAI’s client libraries, making it straightforward to integrate into existing agent architectures. It supports robust error handling, including retry logic for transient errors (such as rate limiting), which is useful for agents running iterative research cycles.
- **Best Practices:** Use environment variables or secure storage for API keys, implement logging and monitoring to track usage and agent performance, and follow practices for key rotation and anomaly detection.
- **Example Workflow:** By sending a series of messages (each representing a round of research), agents can iteratively refine their queries, accumulate context, and build complex research chains, all while receiving cited, current data at each step.

These features make Perplexity API well-suited for iterative, multi-round research in custom AI agents, emphasizing reliability, up-to-date information, and extensible integration.

-----

-----

-----

### Source [39]: https://www.perplexity.ai/hub/blog/introducing-the-perplexity-search-api

Query: Using Perplexity API for iterative, multi-round research cycles in a custom AI agent.

Answer: The Perplexity Search API grants developers programmatic access to the same global-scale infrastructure used by Perplexity’s public answer engine. Key aspects relevant to iterative, multi-round research cycles include:

- **Web-Scale Indexing:** The API leverages an index covering hundreds of billions of webpages, enabling access to up-to-date information for each research iteration.
- **Structured Responses:** Returns are rich and structured, optimized for both AI and traditional applications, which facilitates parsing and follow-up queries in multi-step research agents.
- **Fine-Grained Retrieval:** The infrastructure divides documents into sub-document units, surfacing and scoring these units against queries. This means agents receive the most relevant snippets pre-ranked, reducing need for additional preprocessing between research rounds.
- **Integration Readiness:** The API is designed to streamline integration, allowing real-time, high-throughput research cycles with minimal setup.
- **SDK and Evaluation Tools:** Perplexity provides an SDK and open-source evaluation framework, supporting more robust agent development and iterative experimentation.

This architecture is particularly advantageous for custom AI agents that need to iteratively retrieve, refine, and reason over web-scale information in multiple cycles.

-----

-----

-----

### Source [40]: https://www.perplexity.ai/help-center/en/articles/10352995-api-settings

Query: Using Perplexity API for iterative, multi-round research cycles in a custom AI agent.

Answer: To use the Perplexity API in iterative, multi-round research cycles:

- **API Key Management:** Access API keys via the API settings tab under Settings. Generate a key, store it securely, and use it in your iterative agent’s requests.
- **Usage Monitoring:** The API settings page also enables monitoring of usage patterns, which is helpful when running agents that perform multi-step, high-volume research cycles.
- **Integration Simplicity:** The process is streamlined for easy integration into workflows, facilitating repeated, structured queries typical of multi-round research.

While this source focuses primarily on setup and management, these elements are foundational for building and maintaining robust, iterative research agents.

-----

-----

-----

### Source [41]: https://www.perplexity.ai/hub/blog/introducing-the-sonar-pro-api

Query: Using Perplexity API for iterative, multi-round research cycles in a custom AI agent.

Answer: The Sonar Pro API by Perplexity is tailored for advanced generative search applications, with features aimed at enabling iterative, multi-round research cycles in custom AI agents:

- **Real-Time Web Research:** Provides real-time, internet-wide research capabilities with source citations, ensuring each research round is informed by current, authoritative information.
- **Citations and Source Customization:** Supports citations and even custom source selection, enhancing trust and targeted research across multiple rounds.
- **Extended Context Window:** Sonar Pro offers a larger context window than standard models, allowing agents to handle longer, more complex, and multi-turn queries with better continuity.
- **Support for Follow-Up Questions:** Designed to process in-depth, multi-step queries. Agents can chain queries, refine research, and maintain context over several interactions.
- **Advanced Extensibility:** The API can double the number of citations per search versus the basic Sonar API and is optimized for speed and cost-effectiveness, which is critical for iterative, high-frequency agents.

These features collectively enable the design of AI agents capable of multi-round, iterative research cycles that leverage real-time, attributed data and maintain context across complex workflows.

-----

-----

-----

### Source [42]: https://docs.perplexity.ai

Query: Using Perplexity API for iterative, multi-round research cycles in a custom AI agent.

Answer: The Perplexity API documentation provides technical detail for powering products with real-time, web-wide research and Q&A:

- **Quickstart and Endpoints:** The API is REST-based, with endpoints that support sending and receiving message chains, facilitating multi-turn conversations typical of iterative agents.
- **Model Options:** Documentation outlines different models and their capacities, allowing developers to select the best for their iterative research process.
- **Parameterization:** Developers can control conversation history, context length, and response format, vital for agents that must track and evolve research threads over multiple rounds.
- **Response Structure:** The API returns data in structured formats (typically JSON), which is ideal for parsing and chaining responses in iterative agent workflows.
- **Authentication:** Secured via API keys, and the documentation offers guidance on key management for production agents.

The documentation is essential for designing, implementing, and scaling AI agents that perform iterative, multi-turn research cycles using the Perplexity API.

-----

-----

</details>

<details>
<summary>What are the practical differences and trade-offs when implementing human-in-the-loop for steerable agents versus reliable workflows?</summary>

### Source [43]: https://www.permit.io/blog/human-in-the-loop-for-ai-agents-best-practices-frameworks-use-cases-and-demo

Query: What are the practical differences and trade-offs when implementing human-in-the-loop for steerable agents versus reliable workflows?

Answer: Implementing **human-in-the-loop (HITL) for steerable agents** primarily addresses the risks associated with agent autonomy, such as hallucinated actions, misused permissions, overreach, and lack of traceability. HITL enables developers to delegate final decisions to humans, combining automation with human judgment. The practical difference is that steerable agents, empowered to take actions (not just generate text), pose significant risks if left unchecked, especially in sensitive contexts like access control or financial operations. HITL is deployed at key decision points to prevent irreversible mistakes, ensure accountability, comply with audit requirements, and build trust through transparent supervision.

For **reliable workflows**, HITL is less about steering agent behavior and more about ensuring that each step in an automated process is subject to human oversight where necessary. The trade-off is between operational efficiency (automation) and safety/control (human review). Reliable workflows use HITL to assure compliance and correctness, but the agents themselves are less "steerable" and more constrained by workflow logic.

In summary, **steerable agents require HITL for active control and risk mitigation**, whereas **reliable workflows use HITL to ensure process integrity and compliance**. The primary trade-off is between speed (automation) and safety (human oversight), with HITL serving as the responsible mechanism for both agentic autonomy and workflow reliability.

-----

-----

-----

### Source [44]: https://www.creatio.com/glossary/human-in-the-loop-ai-agents

Query: What are the practical differences and trade-offs when implementing human-in-the-loop for steerable agents versus reliable workflows?

Answer: **Human-in-the-loop (HITL) for AI agents** is designed to combine automation with human oversight, particularly enhancing reliability, safety, and accuracy while providing clear accountability. Steerable agents with HITL can independently perform tasks such as data collection and analysis but will pause and escalate to human experts for ambiguous, complex, or high-stakes decisions. This approach enables agents to learn from human feedback and continuously improve, making them more precise over time.

For **reliable workflows**, HITL is implemented to ensure higher accuracy and compliance, especially in regulated industries like finance, healthcare, and law. HITL helps organizations meet regulatory requirements by embedding human accountability into automated processes while still leveraging AI efficiency. The practical difference is that steerable agents leverage human context for decision-making, whereas reliable workflows focus on process-level oversight.

The central trade-off is operational efficiency versus risk mitigation. HITL in steerable agents enables strategic oversight, allowing AI to handle routine tasks but deferring critical decisions to humans. In reliable workflows, HITL acts as a safety net to prevent errors and maintain compliance, which may slow down automation but increases trustworthiness and correctness.

-----

-----

-----

### Source [45]: https://orkes.io/blog/human-in-the-loop/

Query: What are the practical differences and trade-offs when implementing human-in-the-loop for steerable agents versus reliable workflows?

Answer: In agentic workflows, **human-in-the-loop (HITL)** means pausing automated processes at key points for human review. Steerable agents benefit from HITL by maintaining the speed of automation for straightforward cases while routing complex, ambiguous, or risky cases to humans for decision-making. This hybrid approach allows organizations to harness both the scale of AI and the judgment of human reviewers.

For **reliable workflows**, HITL is about introducing a controlled "pause" in the process until a human makes a decision. While automation continues for routine operations, tasks requiring human input are held until an outcome is determined, with service-level agreements (SLAs) and deadlines guiding escalation and resolution. This ensures that no request (e.g., loan application, fraud alert) is stuck indefinitely and that oversight is systematic and accountable.

The practical difference is that steerable agents with HITL provide *dynamic control and adaptability*, while reliable workflows with HITL focus on *systematic review and compliance*. The trade-off is between maximizing automation and maintaining human oversight where necessary; HITL adds reliability and accountability but may introduce delays and operational complexity.

-----

-----

-----

### Source [46]: https://www.zenml.io/blog/steerable-deep-research-building-production-ready-agentic-workflows-with-controlled-autonomy

Query: What are the practical differences and trade-offs when implementing human-in-the-loop for steerable agents versus reliable workflows?

Answer: In the context of **steerable agents**, human-in-the-loop approval points can be designed as sophisticated quality gates where domain experts actively steer the direction of research or agent actions. This allows for controlled autonomy in agentic workflows, ensuring that agents remain aligned with organizational standards and objectives. Steerable agents benefit from HITL by enabling real-time intervention, course correction, and expert validation, particularly in research or innovation-driven environments.

In contrast, **reliable workflows** with HITL tend to focus on predefined checkpoints for human approval, emphasizing process reliability over adaptability. These workflows are less about steering agents and more about ensuring that outcomes meet quality and regulatory standards.

The trade-off centers on flexibility versus predictability. HITL for steerable agents provides high flexibility, enabling experts to intervene and influence outcomes as needed. Reliable workflows with HITL ensure consistent, predictable results but may restrict dynamic intervention and slow down innovation due to rigid process requirements.

-----

-----

</details>

<details>
<summary>What is the Model Context Protocol (MCP) and what are best practices for implementing an MCP server to expose agentic tools and workflows?</summary>

### Source [47]: https://www.descope.com/learn/post/mcp

Query: What is the Model Context Protocol (MCP) and what are best practices for implementing an MCP server to expose agentic tools and workflows?

Answer: The Model Context Protocol (MCP) is a standardized method for large language models (LLMs) to connect with external data and tools, functioning as a "universal remote" for AI. MCP uses a client-server architecture inspired by the Language Server Protocol (LSP). Its primary elements are:
- **Host application:** The LLM or interface initiating connections (such as Claude Desktop or AI-powered IDEs).
- **MCP client:** Embedded in the host application, this component manages the protocol connection, translating between the host's needs and MCP.
- **MCP server:** This server exposes specific functions or data integrations (e.g., a GitHub integration or a database connection), making them available to AI applications via MCP.
- **Transport layer:** Communication between clients and servers occurs over either STDIO (for local integrations) or HTTP+SSE (for remote and streaming interactions).

All communication uses **JSON-RPC 2.0**, ensuring standardized message structures for requests, responses, and notifications. The MCP server is responsible for exposing specific capabilities, such as functions or data, to the AI application. Best practices include structuring functionality clearly, supporting both local and remote transports as needed, and ensuring robust JSON-RPC compliance.

-----

-----

-----

### Source [48]: https://developers.openai.com/apps-sdk/concepts/mcp-server/

Query: What is the Model Context Protocol (MCP) and what are best practices for implementing an MCP server to expose agentic tools and workflows?

Answer: The Model Context Protocol (MCP) is an open specification for connecting LLM clients to external tools and resources. An MCP server exposes **tools** that models can call during a conversation and returns structured results based on provided parameters. Additionally, servers can return metadata and UI components (such as HTML for rendering interfaces) alongside tool results.

Key protocol building blocks for an MCP server:
- **List tools:** The server advertises supported tools, including their JSON Schema-based input/output contracts and optional annotations.
- **Call tools:** When a model selects a tool, it issues a `call_tool` request with relevant arguments. The server executes the action and returns structured, model-parsable content.
- **Return components:** Each tool can specify an embedded resource for rendering a custom interface in the client.

The protocol is **transport agnostic**—MCP servers can be hosted over Server-Sent Events (SSE) or Streamable HTTP, with Streamable HTTP recommended for most applications. Authentication, wire format, and metadata are standardized, enabling seamless reasoning and consistent tool invocation by LLMs.

-----

-----

-----

### Source [49]: https://stytch.com/blog/model-context-protocol-introduction/

Query: What is the Model Context Protocol (MCP) and what are best practices for implementing an MCP server to expose agentic tools and workflows?

Answer: MCP is an open standard serving as a **universal adapter** between AI models and external tools or services. It allows LLMs to make structured API calls consistently and securely, eliminating the need for bespoke integration code for each external function or data source.

MCP defines a **common protocol, built on JSON-RPC 2.0**, enabling AI assistants to invoke functions, retrieve data, or use predefined prompts in a structured way. Applications act as hosts, running an MCP client, while each integration (database, API, etc.) is implemented as an MCP server. The server exposes its capabilities (functions, data, prompts) through the protocol, and the client handles negotiation and structured communication. This design means the LLM never interacts with APIs directly, but through the MCP handshake, ensuring security and structured data exchange.

-----

-----

-----

### Source [50]: https://modelcontextprotocol.info/docs/

Query: What is the Model Context Protocol (MCP) and what are best practices for implementing an MCP server to expose agentic tools and workflows?

Answer: The Model Context Protocol (MCP) is a standardized protocol for enhancing interaction between LLMs and applications via structured context management. MCP’s documentation highlights the following core concepts:
- **Sampling:** Mechanisms for context sampling and strategy selection.
- **Transport:** Supported communication protocols and data transports.
- **Tools:** Guidelines for exposing and using tools via MCP.
- **Architecture:** Design principles and architecture of MCP-based systems.
- **Prompts:** Best practices for prompt engineering and usage.
- **Resources:** References and related technical resources.

MCP is designed for **standardization, efficiency, extensibility, and ease of use**, providing unified interfaces, optimized context management, support for custom extensions, and simple APIs.

-----

-----

-----

### Source [51]: https://modelcontextprotocol.io/specification/latest

Query: What is the Model Context Protocol (MCP) and what are best practices for implementing an MCP server to expose agentic tools and workflows?

Answer: MCP is an open protocol enabling seamless integration between LLM applications and external tools or data sources. Its goals are to:
- Share contextual information with LLMs.
- Expose tools and capabilities to AI systems.
- Build composable integrations and workflows.

Key protocol details:
- **Base protocol:** JSON-RPC message format, stateful connections, and capability negotiation between server and client.
- **Server features:** 
  - **Resources:** Data/context for use by AI or users.
  - **Prompts:** Templated messages and workflows.
  - **Tools:** Executable functions for LLMs.
  - **Sampling:** Agentic behaviors and recursive LLM interactions.
  - **Roots:** Inquiries into filesystem or URI spaces.
  - **Elicitation:** Server can request additional user information.

Additional utilities include configuration, progress tracking, cancellation, error reporting, and logging, supporting robust agentic workflows and integrations.

-----

-----

-----

### Source [52]: https://cloud.google.com/discover/what-is-model-context-protocol

Query: What is the Model Context Protocol (MCP) and what are best practices for implementing an MCP server to expose agentic tools and workflows?

Answer: MCP is an open standard (introduced by Anthropic in November 2024) that allows LLMs to safely access external data and tools, overcoming static knowledge limitations. It establishes a secure, standardized "language" for LLMs to interact with data, apps, and services. MCP standardizes two-way connections, enabling LLMs to:
- Retrieve up-to-date information.
- Perform actions (e.g., booking meetings, updating records).
- Use specialized features and current data.

MCP builds upon concepts like tool use and function calling, but offers a unified protocol for connecting LLMs with diverse external systems, reducing custom integration requirements and increasing AI reliability and automation.

-----

-----

</details>

<details>
<summary>How can LangGraph be used to enforce stylistic rules and reduce 'AI slop' through reflection and self-critique loops?</summary>

### Source [53]: https://www.langchain.com/langgraph

Query: How can LangGraph be used to enforce stylistic rules and reduce 'AI slop' through reflection and self-critique loops?

Answer: LangGraph provides a **controllable cognitive architecture** for creating agent workflows, allowing developers to enforce stylistic rules and reduce 'AI slop' by integrating moderation and quality control loops directly into the agent’s logic. The framework enables **easy-to-add moderation and quality loops**, which prevent agents from veering off course by enabling checkpoints for human-in-the-loop review and approval[1]. Developers can inspect agent actions, roll back to previous states, and correct course if the output does not meet stylistic or quality standards. LangGraph’s **statefulness and workflow customizability** allow for the design of complex control flows—including reflection steps and self-critique loops—so agents can be programmed to self-review their outputs for style, coherence, and compliance before finalizing them. This makes it possible to systematically enforce stylistic rules, such as maintaining tone, structure, or formatting, and to reduce undesired AI behaviors ('AI slop') through iterative self-assessment[1].

-----

-----

-----

### Source [54]: https://www.truefoundry.com/blog/langchain-vs-langgraph

Query: How can LangGraph be used to enforce stylistic rules and reduce 'AI slop' through reflection and self-critique loops?

Answer: LangGraph excels in **agent-style workflows** that require adaptation, backtracking, and state tracking. It is designed for scenarios where workflows may loop through steps until a condition is met, making it well-suited for implementing **reflection and self-critique loops**. With LangGraph, agents can **pause for verification or adjustment** at human checkpoints, allowing external enforcement of stylistic rules and reduction of AI slop through repeated review cycles[2]. The framework’s ability to **pause and resume** workflows precisely supports the insertion of self-assessment or moderation steps after each output. This enables developers to build complex systems where agents must review and possibly revise their outputs until they comply with specific stylistic requirements, leveraging both automated and human-in-the-loop critiques[2].

-----

-----

-----

### Source [55]: https://www.ema.co/additional-blogs/addition-blogs/building-ai-agents-langgraph

Query: How can LangGraph be used to enforce stylistic rules and reduce 'AI slop' through reflection and self-critique loops?

Answer: LangGraph lets developers **map AI workflows as graphs**, so they can design nodes for reflection, self-critique, moderation, and approval, as part of the agent’s decision-making process[3]. Each node in the graph can represent a distinct function, such as generating a draft response, performing a style check, or invoking a self-critique algorithm. The graph architecture supports **conditional paths and loops**, allowing agents to revisit previous steps if outputs do not meet stylistic or quality criteria. LangGraph’s built-in **state management** ensures agents retain context and memory across sessions, which is crucial for consistent stylistic enforcement[3]. Moderation and quality control can be embedded directly into the workflow, using approval gates or red-flag checks to catch and correct AI slop before results are finalized. **Human-in-the-loop controls** further strengthen this process by incorporating manual review and feedback loops, while **visual debugging tools** allow developers to inspect, trace, and roll back agent actions to continually refine stylistic adherence[3].

-----

-----

-----

### Source [56]: https://duplocloud.com/blog/langchain-vs-langgraph/

Query: How can LangGraph be used to enforce stylistic rules and reduce 'AI slop' through reflection and self-critique loops?

Answer: LangGraph’s **graph-based architecture** enables dynamic, non-linear workflows with explicit state management, supporting cycles and iterative processes essential for enforcing stylistic rules and reducing AI slop[4]. Developers can create workflows where agents **loop back to critique or revise outputs** based on specified style guidelines. Each node can modify the agent’s state, and conditional branching allows for repeated self-assessment or external review before proceeding. This facilitates **reflection and self-critique loops**, as agents can iteratively refine outputs until they meet desired stylistic standards. The explicit state object lets developers track changes, maintain history, and ensure compliance across sessions. LangGraph’s support for **checkpointers** and persistent state ensures that even long-running or multi-agent systems maintain stylistic consistency and can recover from missteps, further reducing AI slop[4].

-----

-----

-----

### Source [57]: https://langchain-ai.github.io/langgraph/concepts/why-langgraph/

Query: How can LangGraph be used to enforce stylistic rules and reduce 'AI slop' through reflection and self-critique loops?

Answer: LangGraph is built for developers aiming to create powerful, adaptable AI agents. It emphasizes **adaptability and control**, enabling robust agentic architectures where developers can **enforce rules, handle exceptions, and implement quality assurance** directly within the agent’s workflow[5]. The design supports **flexible, non-linear control flows**, which are ideal for embedding reflection and self-critique loops—agents can be made to re-evaluate their outputs, check for stylistic conformity, and revise as needed before finalizing responses. This architecture makes it practical to systematically enforce stylistic rules and reduce AI slop, as agents can be programmed to follow strict review and moderation protocols at multiple points in the workflow[5].

-----

-----

</details>

<details>
<summary>What are robust techniques for automated context ingestion from diverse sources like GitHub repositories, YouTube transcripts, and local files for LLM agents?</summary>

### Source [58]: https://galileo.ai/blog/multi-context-processing-llms

Query: What are robust techniques for automated context ingestion from diverse sources like GitHub repositories, YouTube transcripts, and local files for LLM agents?

Answer: Robust techniques for automated context ingestion in LLM agents focus on **multi-context processing**, enabling LLMs to synthesize information from various sources within a single inference cycle. Key strategies include:

- **Context Window Segmentation**: Modern transformer models split the context window into discrete attention blocks, each handling a different information stream, while maintaining pathways for cross-context communication. This allows for parallel processing of diverse sources such as code repositories, transcripts, or files.
  
- **Hierarchical Attention Patterns**: Advanced frameworks implement attention at both global and local context levels, efficiently handling simultaneous inputs and preventing less relevant information from dominating the context.

- **Dynamic Memory Allocation**: Memory strategies dynamically adjust context allocation based on relevance scores, so high-importance data from sources like GitHub or YouTube is prioritized.

- **Segmentation Algorithms**: Sophisticated algorithms create logical separations between contexts (e.g., code vs. transcript) while preserving semantic links for information that spans sources.

- **Context-Aware Prompt Engineering**: Use dynamic template architectures and context labeling systems to clearly delineate sources in the prompt, preventing information blending and maintaining source attribution. Conditional prompt generation activates template sections depending on context presence.

- **Token Allocation Strategies**: Adjust token use dynamically to balance comprehensive context coverage with computational efficiency, adapting to query complexity and available resources.

These approaches collectively ensure that LLM agents can robustly ingest, organize, and leverage context from diverse data sources in an automated fashion.

-----

-----

-----

### Source [59]: https://arxiv.org/html/2508.08322v1

Query: What are robust techniques for automated context ingestion from diverse sources like GitHub repositories, YouTube transcripts, and local files for LLM agents?

Answer: This research introduces a **multi-agent, context engineering workflow** for LLM code assistants working with complex, multi-file projects (e.g., large GitHub repositories). The workflow includes:

- **Intent Translator**: Uses an advanced LLM (e.g., GPT-5) to clarify user requirements, ensuring that agent tasks are grounded in correct objectives.
  
- **Semantic Literature Retrieval**: Employs tools like Elicit to automatically fetch domain knowledge relevant to the context, such as technical documentation or related code examples.

- **Document Synthesis**: Via systems like NotebookLM, agents generate synthesized, context-rich summaries of large codebases or documentation, transforming raw text into actionable context.

- **Multi-Agent Orchestration**: Claude Code and similar frameworks coordinate specialized sub-agents for different ingestion and analysis tasks (e.g., code generation, validation, retrieval), ensuring coverage of all necessary context slices.

- **Retrieval-Augmented Generation (RAG)**: Integrates external knowledge dynamically, so LLMs can access up-to-date information from repositories or documents as needed.

Results show this multi-component workflow significantly improves code assistant accuracy, reliability, and adherence to project context compared to monolithic, single-agent approaches. The approach emphasizes that **supplying the right information in the right form**, via targeted context injection and agent role decomposition, is crucial for robust ingestion from complex, heterogeneous sources.

-----

-----

-----

### Source [60]: https://www.llamaindex.ai/blog/context-engineering-what-it-is-and-techniques-to-consider

Query: What are robust techniques for automated context ingestion from diverse sources like GitHub repositories, YouTube transcripts, and local files for LLM agents?

Answer: Key context engineering techniques for robust context ingestion include:

- **Context Summarization**: After retrieving content (e.g., from YouTube transcripts or repository files), summarization algorithms condense information before passing it to the LLM, maximizing use of the limited context window.

- **Context Ordering and Ranking**: Retrieved data can be ranked (e.g., by recency or relevance), with only the most critical or recent items included in the context. This is particularly useful when ingesting files with time-sensitive or prioritized information.

- **Filtering**: Apply filters to exclude outdated or irrelevant data, ensuring only valuable information is presented to the LLM.

- **Code Example**: The article provides a Python pseudocode function which retrieves knowledge items, filters them by date, sorts them, and then concatenates the relevant text for LLM input.

These techniques help manage the **context limit** (the maximum token window for LLMs) and ensure that context ingestion is both efficient and effective, regardless of the data source type.

-----

-----

-----

### Source [61]: https://unstructured.io/blog/understanding-what-matters-for-llm-ingestion-and-preprocessing

Query: What are robust techniques for automated context ingestion from diverse sources like GitHub repositories, YouTube transcripts, and local files for LLM agents?

Answer: Effective LLM context ingestion from diverse sources (like code repositories, transcripts, local files) requires a robust pipeline:

- **Transform**: Extract and structure data from raw sources (e.g., scraping GitHub, parsing video transcripts, or reading local documents).
  
- **Clean**: Remove unwanted content such as boilerplate, irrelevant metadata, or noise, ensuring only high-quality data is ingested.

- **Chunk**: Break large documents or files into manageable segments, with chunking strategies tailored to the data type (e.g., function-level for code, paragraph-level for text).

- **Summarize**: Summarize long or dense content, reducing redundancy and fitting more salient information into the LLM’s context window.

- **Metadata Enrichment**: Add source metadata (such as file path, speaker, or timestamp) to preserve attribution and facilitate downstream processing.

This structured approach ensures the LLM receives relevant, high-quality, and well-labeled context, supporting accurate and context-aware generation.

-----

-----

-----

### Source [62]: https://community.openai.com/t/prompt-engineering-is-dead-and-context-engineering-is-already-obsolete-why-the-future-is-automated-workflow-architecture-with-llms/1314011

Query: What are robust techniques for automated context ingestion from diverse sources like GitHub repositories, YouTube transcripts, and local files for LLM agents?

Answer: Automated context ingestion is moving toward **workflow-driven architectures** where context is emitted and structured automatically by the data sources themselves:

- **Schema Analysis**: Automatically extract data schemas from codebases or files, providing structured summaries for LLM input.

- **Documentation Generators**: Use tools that auto-generate documentation from repositories or code, creating context blocks ready for LLM ingestion.

- **Workflow Compilers**: Compile high-level workflows from disparate sources (e.g., code, requirements documents, transcripts) into unified, structured prompts or context windows.

Such architectures automate the **generation, structuring, and presentation of context** for LLMs, reducing manual engineering and enabling scalable ingestion from a wide variety of source types.

-----

-----

-----

### Source [63]: https://www.newline.co/@zaoyang/dynamic-context-injection-with-retrieval-augmented-generation--68b80921

Query: What are robust techniques for automated context ingestion from diverse sources like GitHub repositories, YouTube transcripts, and local files for LLM agents?

Answer: **Dynamic context injection** and **Retrieval-Augmented Generation (RAG)** are robust methods for ingesting context from diverse external sources:

- **Dynamic Context Injection**: External data (e.g., from GitHub, YouTube, local files) is retrieved and injected into the LLM prompt in real time, ensuring the model has access to the latest and most relevant information.

- **RAG Pipelines**: LLMs fetch context from external databases or search indexes on demand, integrating up-to-date content seamlessly into prompts.

These techniques allow LLM agents to remain current with external knowledge and flexibly incorporate heterogeneous data sources into their reasoning and generation processes, supporting robust, automated context ingestion.

-----

-----

</details>

<details>
<summary>What are the key architectural differences between off-the-shelf 'deep research' tools like Perplexity and custom-built, multi-agent systems for research and writing?</summary>

### Source [64]: https://www.graphapp.ai/blog/perplexity-technical-deep-dive-understanding-the-complexities

Query: What are the key architectural differences between off-the-shelf 'deep research' tools like Perplexity and custom-built, multi-agent systems for research and writing?

Answer: Perplexity's architecture is composed of distinct, specialized components: a user-friendly interface that handles both text and voice input, a **query processing module** that interprets user intent, a **data retrieval system** employing advanced algorithms to fetch information from diverse sources, and a **machine learning engine** that improves over time by learning from user interactions. This architecture is designed for **scalability** through a distributed system capable of load balancing, allowing Perplexity to efficiently accommodate growth in users and queries. Security is integral, with robust encryption and access controls ensuring data privacy and compliance with regulations. The architecture is holistic, emphasizing both **functionality** (speed, accuracy, up-to-date knowledge) and **security**, distinguishing it from traditional single-agent or less integrated research tools.

-----

-----

-----

### Source [65]: https://www.xfunnel.ai/blog/inside-perplexity-ai

Query: What are the key architectural differences between off-the-shelf 'deep research' tools like Perplexity and custom-built, multi-agent systems for research and writing?

Answer: Perplexity leverages a **Retrieval-Augmented Generation (RAG)** paradigm, integrating its own web crawlers and external search APIs to collect and index context-rich web content. The process involves converting web data into numerical vectors (embeddings), retrieving relevant documents via similarity search, and then **augmenting** the LLM's context with these documents to ground responses in current facts. The LLM then generates answers that blend its pre-trained knowledge with real-time, retrieved data. The system is continuously improved, with ongoing refinement of document ranking, integration of new models, and adaptation to user behavior. This tightly coupled pipeline is optimized for **answer accuracy and recency**, but is fundamentally a single-agent system orchestrating retrieval and generation.

-----

-----

-----

### Source [66]: https://developer.nvidia.com/blog/spotlight-perplexity-ai-serves-400-million-search-queries-a-month-using-nvidia-inference-stack/

Query: What are the key architectural differences between off-the-shelf 'deep research' tools like Perplexity and custom-built, multi-agent systems for research and writing?

Answer: Perplexity serves over 20 AI models simultaneously using NVIDIA H100 GPU pods and NVIDIA Triton Inference Server, orchestrated via Kubernetes for scalability and reliability. A **front-end scheduler** directs user tasks to appropriate models based on intent, optimizing for both cost and latency. The architecture supports **multiple models and task types** (search, summarization, Q&A), with classifier models determining routing. The scheduler also dynamically scales resources and manages load, ensuring consistent performance. This highly modular, production-grade infrastructure is designed for large-scale, real-time user interaction, with robust scheduling and deployment mechanisms, but does not inherently support **multi-agent collaboration** or user-defined workflow customization.

-----

-----

-----

### Source [67]: https://www.m1-project.com/blog/what-is-perplexity-ai-and-how-it-works

Query: What are the key architectural differences between off-the-shelf 'deep research' tools like Perplexity and custom-built, multi-agent systems for research and writing?

Answer: Perplexity is positioned as an **end-to-end solution** that integrates research, content creation, and analysis using **state-of-the-art language models** (e.g., GPT-4) and proprietary search technologies. It interprets queries contextually with LLMs, then draws on a continually updated live search engine to extract the latest, most accurate information from a variety of web sources. Unlike traditional search engines that return link lists, Perplexity provides **summarized, actionable answers** grounded in current, authoritative sources. The design focuses on **real-time accuracy, contextual understanding, and response synthesis**, with minimal user configuration or workflow customization, reflecting a tightly integrated, single-agent approach.

-----

-----

-----

### Source [68]: https://annjose.com/post/perplexity-ai/

Query: What are the key architectural differences between off-the-shelf 'deep research' tools like Perplexity and custom-built, multi-agent systems for research and writing?

Answer: Perplexity employs a **RAG (Retrieval-Augmented Generation)** approach but extends it by strictly adhering to information retrieved from external sources. It prioritizes **grounded factuality**, refusing to generate content for which it lacks verifiable source material. The architecture consists of a dedicated **search component** for retrieval and an **LLM component** for crafting answers, with models fine-tuned for citation, summarization, and referencing. Perplexity is **model-agnostic**, supporting its own and third-party LLMs. The workflow is linear: understand the question, retrieve current web data, and synthesize a concise, cited answer. This differs from multi-agent systems, which typically feature **modular, orchestratable agents** that can be customized or extended to perform subtasks with more granular control.

-----

-----

-----

### Source [69]: https://www.perplexity.ai/api-platform/resources/architecting-and-evaluating-an-ai-first-search-api

Query: What are the key architectural differences between off-the-shelf 'deep research' tools like Perplexity and custom-built, multi-agent systems for research and writing?

Answer: Perplexity's API architecture combines **hybrid retrieval mechanisms** (mixing dense and sparse retrieval), multi-stage ranking pipelines, distributed indexing, and dynamic parsing. The system is built for **speed, scalability, and accuracy**, using distributed infrastructure to efficiently process and deliver answers. The architecture is optimized for **search and retrieval tasks** with real-time responsiveness, but is not designed for **complex, user-defined multi-agent workflows** where independent agents collaborate or negotiate on subtasks or strategies.

-----

-----

</details>


## Sources Scraped From Research Results

<details>
<summary>Instead of simply delivering information, agents reason, act, and collaborate—bridging the gap between knowledge and outcomes.</summary>

Instead of simply delivering information, agents reason, act, and collaborate—bridging the gap between knowledge and outcomes.

_This blog post is the first out of a six-part blog series called [Agent Factory](https://azure.microsoft.com/en-us/blog/tag/agent-factory/) which will share best practices, design patterns, and tools to help guide you through adopting and building agentic AI._

## Beyond knowledge: Why enterprises need agentic AI

Retrieval-augmented generation (RAG) marked a breakthrough for enterprise AI—helping teams surface insights and answer questions at unprecedented speed. For many, it was a launchpad: copilots and chatbots that streamlined support and reduced the time spent searching for information.

However, answers alone rarely drive real business impact. Most enterprise workflows demand action: submitting forms, updating records, or orchestrating multi-step processes across diverse systems. Traditional automation tools—scripts, Robotic Process Automation (RPA) bots, manual handoffs—often struggle with change and scale, leaving teams frustrated by gaps and inefficiencies.

This is where agentic AI emerges as a game-changer. Instead of simply delivering information, agents reason, act, and collaborate—bridging the gap between knowledge and outcomes and enabling a new era of enterprise automation.

## Patterns of agentic AI: Building blocks for enterprise automation

While the shift from retrieval to real-world action often begins with agents that can use tools, enterprise needs don’t stop there. Reliable automation requires agents that reflect on their work, plan multi-step processes, collaborate across specialties, and adapt in real time—not just execute single calls.

The five patterns below are foundational building blocks seen in production today. They’re designed to be combined and together unlock transformative automation.

### 1. Tool use pattern—from advisor to operator

Modern agents stand out by driving real outcomes. Today’s agents interact directly with enterprise systems—retrieving data, calling Application Programming Interface (APIs), triggering workflows, and executing transactions. Agents now surface answers and also complete tasks, update records, and orchestrate workflows end-to-end.

[**Fujitsu**](https://www.microsoft.com/en/customers/story/21885-fujitsu-azure-ai-foundry) transformed its sales proposal process using specialized agents for data analysis, market research, and document creation—each invoking specific APIs and tools. Instead of simply answering “what should we pitch,” agents built and assembled entire proposal packages, reducing production time by 67%.https://azure.microsoft.com/en-us/blog/wp-content/uploads/2025/07/image-7.webp

### 2. Reflection pattern—self-improvement for reliability

Once agents can act, the next step is reflection—the ability to assess and improve their own outputs. Reflection lets agents catch errors and iterate for quality without always depending on humans.

In high-stakes fields like compliance and finance, a single error can be costly. With self-checks and review loops, agents can auto-correct missing details, double-check calculations, or ensure messages meet standards. Even code assistants, like [GitHub Copilot](https://github.com/features/copilot), rely on internal testing and refinement before sharing outputs. This self-improving loop reduces errors and gives enterprises confidence that AI-driven processes are safe, consistent, and auditable.https://azure.microsoft.com/en-us/blog/wp-content/uploads/2025/07/image-9.webp

### 3. Planning pattern—decomposing complexity for robustness

Most real business processes aren’t single steps—they’re complex journeys with dependencies and branching paths. Planning agents address this by breaking high-level goals into actionable tasks, tracking progress, and adapting as requirements shift.

**ContraForce’s** Agentic Security Delivery Platform (ASDP) automated its partner’s security service delivery with security service agents using planning agents that break down incidents into intake, impact assessment, playbook execution, and escalation. As each phase completes, the agent checks for next steps, ensuring nothing gets missed. The result: 80% of incident investigation and response is now automated and full incident investigation can be processed for less than $1 per incident.

Planning often combines tool use and reflection, showing how these patterns reinforce each other. A key strength is flexibility: plans can be generated dynamically by an LLM or follow a predefined sequence, whichever fits the need.https://azure.microsoft.com/en-us/blog/wp-content/uploads/2025/08/image-1.webp

### 4. Multi-agent pattern—collaboration at machine speed

No single agent can do it all. Enterprises create value through teams of specialists, and the multi-agent pattern mirrors this by connecting networks of specialized agents—each focused on different workflow stages—under an orchestrator. This modular design enables agility, scalability, and easy evolution, while keeping responsibilities and governance clear.

Modern multi-agent solutions use [several orchestration patterns](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)—often in combination—to address real enterprise needs. These can be LLM-driven or deterministic: **sequential orchestration** (such as agents refine a document step by step), **concurrent orchestration** (agents run in parallel and merge results), **group chat/maker-checker** (agents debate and validate outputs together), **dynamic handoff** (real-time triage or routing), and **magentic orchestration** (a manager agent coordinates all subtasks until completion).

[JM Family](https://news.microsoft.com/source/features/ai/meet-4-developers-leading-the-way-with-ai-agents/) adopted this approach with business analyst/quality assurance (BAQA) Genie, deploying agents for requirements, story writing, coding, documentation, and Quality Assurance (QA). Coordinated by an orchestrator, their development cycles became standardized and automated—cutting requirements and test design from weeks to days and saving up to 60% of QA time.https://azure.microsoft.com/en-us/blog/wp-content/uploads/2025/08/image-2-1024x644.webp

### 5. ReAct (Reason + Act) pattern—adaptive problem solving in real time

The ReAct pattern enables agents to solve problems in real time, especially when static plans fall short. Instead of a fixed script, ReAct agents alternate between reasoning and action—taking a step, observing results, and deciding what to do next. This allows agents to adapt to ambiguity, evolving requirements, and situations where the best path forward isn’t clear.

For example, in enterprise IT support, a virtual agent powered by the ReAct pattern can diagnose issues in real time: it asks clarifying questions, checks system logs, tests possible solutions, and adjusts its strategy as new information becomes available. If the issue grows more complex or falls outside its scope, the agent can escalate the case to a human specialist with a detailed summary of what’s been attempted.https://azure.microsoft.com/en-us/blog/wp-content/uploads/2025/07/image-8.webp

These patterns are meant to be combined. The most effective agentic solutions weave together tool use, reflection, planning, multi-agent collaboration, and adaptive reasoning—enabling automation that is faster, smarter, safer, and ready for the real world.

## Why a unified agent platform is essential

Building intelligent agents goes far beyond prompting a language model. When moving from demo to real-world use, teams quickly encounter challenges:

- **How do I chain multiple steps together reliably?**
- **How do I give agents access to business data—securely and responsibly?**
- **How do I monitor, evaluate, and improve agent behavior?**
- **How do I ensure security and identity across different agent components?**
- **How do I scale from a single agent to a team of agents—or connect to others?**

Many teams end up building custom scaffolding—DIY orchestrators, logging, tool managers, and access controls. This slows time-to-value, creates risks, and leads to fragile solutions.

</details>

<details>
<summary>(no content)</summary>

(no content)

</details>

<details>
<summary>Trusted by companies shaping the future of agents - including Klarna, Replit, Elastic, and more - LangGraph is a low-level orchestration framework for building, managing, and deploying long-running, stateful agents.LangGraph is very low-level, and focused entirely on agent **orchestration**. Before using LangGraph, it is recommended you familiarize yourself with some of the components used to build agents, starting with [models](https://docs.langchain.com/oss/python/langchain/models) and [tools](https://docs.langchain.com/oss/python/langchain/tools). We will commonly use [LangChain](https://docs.langchain.com/oss/python/langchain/overview) components throughout the documentation, but you don’t need to use LangChain to use LangGraph.If you are just getting started with agents, or want a higher level abstraction, it is recommended that you use LangChain’s [agents](https://docs.langchain.com/oss/python/langchain/agents).LangGraph is focused on the underlying capabilties important for agent orchestration: durable execution, streaming, human-in-the-loop, etc. We expose two different APIs for consuming these capabilities: a Graph API and a functional API. We largely use the Graph API throughout the documentation, but feel free to use the functional API if you’d prefer.</summary>

Trusted by companies shaping the future of agents - including Klarna, Replit, Elastic, and more - LangGraph is a low-level orchestration framework for building, managing, and deploying long-running, stateful agents.LangGraph is very low-level, and focused entirely on agent **orchestration**. Before using LangGraph, it is recommended you familiarize yourself with some of the components used to build agents, starting with [models](https://docs.langchain.com/oss/python/langchain/models) and [tools](https://docs.langchain.com/oss/python/langchain/tools). We will commonly use [LangChain](https://docs.langchain.com/oss/python/langchain/overview) components throughout the documentation, but you don’t need to use LangChain to use LangGraph.If you are just getting started with agents, or want a higher level abstraction, it is recommended that you use LangChain’s [agents](https://docs.langchain.com/oss/python/langchain/agents).LangGraph is focused on the underlying capabilties important for agent orchestration: durable execution, streaming, human-in-the-loop, etc. We expose two different APIs for consuming these capabilities: a Graph API and a functional API. We largely use the Graph API throughout the documentation, but feel free to use the functional API if you’d prefer.

## Install

pip

uv

```
pip install --pre -U langgraph

```

Then, create a simple hello world example:

```
from langgraph.graph import StateGraph, MessagesState, START, END

def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

graph = StateGraph(MessagesState)
graph.add_node(mock_llm)
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)
graph = graph.compile()

graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})

```

## Core benefits

LangGraph provides low-level supporting infrastructure for _any_ long-running, stateful workflow or agent. LangGraph does not abstract prompts or architecture, and provides the following central benefits:

- [Durable execution](https://docs.langchain.com/oss/python/langgraph/durable-execution): Build agents that persist through failures and can run for extended periods, resuming from where they left off.
- [Human-in-the-loop](https://docs.langchain.com/oss/python/langgraph/add-human-in-the-loop): Incorporate human oversight by inspecting and modifying agent state at any point.
- [Comprehensive memory](https://docs.langchain.com/oss/python/concepts/memory): Create stateful agents with both short-term working memory for ongoing reasoning and long-term memory across sessions.
- [Debugging with LangSmith](https://docs.langchain.com/langsmith/home): Gain deep visibility into complex agent behavior with visualization tools that trace execution paths, capture state transitions, and provide detailed runtime metrics.
- [Production-ready deployment](https://docs.langchain.com/langgraph-platform/deployment-options): Deploy sophisticated agent systems confidently with scalable infrastructure designed to handle the unique challenges of stateful, long-running workflows.

## LangGraph ecosystem

While LangGraph can be used standalone, it also integrates seamlessly with any LangChain product, giving developers a full suite of tools for building agents. To improve your LLM application development, pair LangGraph with:

- [LangSmith](http://www.langchain.com/langsmith) — Helpful for agent evals and observability. Debug poor-performing LLM app runs, evaluate agent trajectories, gain visibility in production, and improve performance over time.
- [LangGraph Platform](https://docs.langchain.com/langgraph-platform) — Deploy and scale agents effortlessly with a purpose-built deployment platform for long running, stateful workflows. Discover, reuse, configure, and share agents across teams — and iterate quickly with visual prototyping in [LangGraph Studio](https://docs.langchain.com/langgraph-platform/langgraph-studio).
- [LangChain](https://docs.langchain.com/oss/python/langchain/overview) \- Provides integrations and composable components to streamline LLM application development. Contains agent abstractions built on top of LangGraph.

## Acknowledgements

LangGraph is inspired by [Pregel](https://research.google/pubs/pub37252/) and [Apache Beam](https://beam.apache.org/). The public interface draws inspiration from [NetworkX](https://networkx.org/documentation/latest/). LangGraph is built by LangChain Inc, the creators of LangChain, but can be used without LangChain.

</details>

<details>
<summary>Overview</summary>

# Overview

LangGraph is built for developers who want to build powerful, adaptable AI agents. Developers choose LangGraph for:

- **Reliability and controllability.** Steer agent actions with moderation checks and human-in-the-loop approvals. LangGraph persists context for long-running workflows, keeping your agents on course.
- **Low-level and extensible.** Build custom agents with fully descriptive, low-level primitives free from rigid abstractions that limit customization. Design scalable multi-agent systems, with each agent serving a specific role tailored to your use case.
- **First-class streaming support.** With token-by-token streaming and streaming of intermediate steps, LangGraph gives users clear visibility into agent reasoning and actions as they unfold in real time.

</details>

<details>
<summary>[Model Context Protocol](https://modelcontextprotocol.io/) (MCP) is an open protocol that</summary>

[Model Context Protocol](https://modelcontextprotocol.io/) (MCP) is an open protocol that
enables seamless integration between LLM applications and external data sources and
tools. Whether you’re building an AI-powered IDE, enhancing a chat interface, or creating
custom AI workflows, MCP provides a standardized way to connect LLMs with the context
they need.This specification defines the authoritative protocol requirements, based on the
TypeScript schema in
[schema.ts](https://github.com/modelcontextprotocol/specification/blob/main/schema/2025-06-18/schema.ts).For implementation guides and examples, visit
[modelcontextprotocol.io](https://modelcontextprotocol.io/).The key words “MUST”, “MUST NOT”, “REQUIRED”, “SHALL”, “SHALL NOT”, “SHOULD”, “SHOULD
NOT”, “RECOMMENDED”, “NOT RECOMMENDED”, “MAY”, and “OPTIONAL” in this document are to be
interpreted as described in [BCP 14](https://datatracker.ietf.org/doc/html/bcp14)
\[ [RFC2119](https://datatracker.ietf.org/doc/html/rfc2119)\]
\[ [RFC8174](https://datatracker.ietf.org/doc/html/rfc8174)\] when, and only when, they
appear in all capitals, as shown here.

## Overview

MCP provides a standardized way for applications to:

- Share contextual information with language models
- Expose tools and capabilities to AI systems
- Build composable integrations and workflows

The protocol uses [JSON-RPC](https://www.jsonrpc.org/) 2.0 messages to establish
communication between:

- **Hosts**: LLM applications that initiate connections
- **Clients**: Connectors within the host application
- **Servers**: Services that provide context and capabilities

MCP takes some inspiration from the
[Language Server Protocol](https://microsoft.github.io/language-server-protocol/), which
standardizes how to add support for programming languages across a whole ecosystem of
development tools. In a similar way, MCP standardizes how to integrate additional context
and tools into the ecosystem of AI applications.

## Key Details

### Base Protocol

- [JSON-RPC](https://www.jsonrpc.org/) message format
- Stateful connections
- Server and client capability negotiation

### Features

Servers offer any of the following features to clients:

- **Resources**: Context and data, for the user or the AI model to use
- **Prompts**: Templated messages and workflows for users
- **Tools**: Functions for the AI model to execute

Clients may offer the following features to servers:

- **Sampling**: Server-initiated agentic behaviors and recursive LLM interactions
- **Roots**: Server-initiated inquiries into uri or filesystem boundaries to operate in
- **Elicitation**: Server-initiated requests for additional information from users

### Additional Utilities

- Configuration
- Progress tracking
- Cancellation
- Error reporting
- Logging

## Security and Trust & Safety

The Model Context Protocol enables powerful capabilities through arbitrary data access
and code execution paths. With this power comes important security and trust
considerations that all implementors must carefully address.

### Key Principles

1. **User Consent and Control**   - Users must explicitly consent to and understand all data access and operations
   - Users must retain control over what data is shared and what actions are taken
   - Implementors should provide clear UIs for reviewing and authorizing activities
2. **Data Privacy**   - Hosts must obtain explicit user consent before exposing user data to servers
   - Hosts must not transmit resource data elsewhere without user consent
   - User data should be protected with appropriate access controls
3. **Tool Safety**   - Tools represent arbitrary code execution and must be treated with appropriate
        caution.
     - In particular, descriptions of tool behavior such as annotations should be
       considered untrusted, unless obtained from a trusted server.
   - Hosts must obtain explicit user consent before invoking any tool
   - Users should understand what each tool does before authorizing its use
4. **LLM Sampling Controls**   - Users must explicitly approve any LLM sampling requests
   - Users should control:
     - Whether sampling occurs at all
     - The actual prompt that will be sent
     - What results the server can see
   - The protocol intentionally limits server visibility into prompts

### Implementation Guidelines

While MCP itself cannot enforce these security principles at the protocol level,
implementors **SHOULD**:

1. Build robust consent and authorization flows into their applications
2. Provide clear documentation of security implications
3. Implement appropriate access controls and data protections
4. Follow security best practices in their integrations
5. Consider privacy implications in their feature designs

</details>


## Code Sources

_No code sources found._


## YouTube Video Transcripts

_No YouTube video transcripts found._


## Additional Sources Scraped

<details>
<summary>2412-11385-why-does-chatgpt-delve-so-much-exploring-the-sour</summary>

# Why Does ChatGPT "Delve" So Much? Exploring the Sources of Lexical Overrepresentation in Large Language Models

Authors: [Tom S. Juzek](https://arxiv.org/search/cs?searchtype=author&query=Juzek,+T+S), [Zina B. Ward](https://arxiv.org/search/cs?searchtype=author&query=Ward,+Z+B)

> Abstract:Scientific English is currently undergoing rapid change, with words like "delve," "intricate," and "underscore" appearing far more frequently than just a few years ago. It is widely assumed that scientists' use of large language models (LLMs) is responsible for such trends. We develop a formal, transferable method to characterize these linguistic changes. Application of our method yields 21 focal words whose increased occurrence in scientific abstracts is likely the result of LLM usage. We then pose "the puzzle of lexical overrepresentation": WHY are such words overused by LLMs? We fail to find evidence that lexical overrepresentation is caused by model architecture, algorithm choices, or training data. To assess whether reinforcement learning from human feedback (RLHF) contributes to the overuse of focal words, we undertake comparative model testing and conduct an exploratory online study. While the model testing is consistent with RLHF playing a role, our experimental results suggest that participants may be reacting differently to "delve" than to other focal words. With LLMs quickly becoming a driver of global language change, investigating these potential sources of lexical overrepresentation is important. We note that while insights into the workings of LLMs are within reach, a lack of transparency surrounding model development remains an obstacle to such research.

</details>

<details>
<summary>chatbots-have-thoroughly-infiltrated-scientific-publishing-s</summary>

# AI Chatbots Have Thoroughly Infiltrated Scientific Publishing

One percent of scientific articles published in 2023 showed signs of generative AI’s potential involvement, according to a recent analysishttps://static.scientificamerican.com/dam/m/2161eea18fe7dd06/original/AIsciencePubs_graphic_leadImage.png?m=1714411499.322&w=600

Amanda Montañez; Source: Andrew Gray

Researchers are misusing ChatGPT and other artificial intelligence chatbots to produce scientific literature. At least, that’s a new fear that some scientists have raised, citing a stark rise in suspicious AI shibboleths showing up in published papers.

Some of these tells—such as the [inadvertent inclusion](https://twitter.com/gcabanac/status/1767574447337124290?) of “certainly, here is a possible introduction for your topic” in a recent paper in _Surfaces and Interfaces_, a journal published by Elsevier—are reasonably obvious evidence that a scientist used an AI chatbot known as a large language model (LLM). But “that’s probably only the tip of the iceberg,” says scientific integrity consultant Elisabeth Bik. (A representative of Elsevier told _Scientific American_ that the publisher regrets the situation and is investigating how it could have “slipped through” the manuscript evaluation process.) In most other cases AI involvement isn’t as clear-cut, and automated AI text detectors are [unreliable tools](https://www.scientificamerican.com/article/tech-companies-new-favorite-solution-for-the-ai-content-crisis-isnt-enough/) for analyzing a paper.

> At least 60,000 papers may have used text generated by a large language model, according to librarian Andrew Gary's analysis.

Researchers from several fields have, however, identified a few key words and phrases (such as “ [complex and multifaceted](https://blog.j11y.io/2023-11-22_multifaceted/)”) that tend to appear more often in AI-generated sentences than in typical human writing. “When you’ve looked at this stuff long enough, you get a feel for the style,” says Andrew Gray, a librarian and researcher at University College London.

LLMs are designed to generate text—but what they produce may or may not be factually accurate. “The problem is that these tools are not good enough yet to trust,” Bik says. They succumb to what computer scientists call [hallucination](https://www.scientificamerican.com/article/chatbot-hallucinations-inevitable/): simply put, they make stuff up. “Specifically, for scientific papers,” Bik notes, an AI “will generate citation references that don’t exist.” So if scientists place too much confidence in LLMs, study authors risk inserting AI-fabricated flaws into their work, mixing more potential for error into the already messy reality of scientific publishing.

Gray recently hunted for AI buzzwords in scientific papers using Dimensions, a data analytics platform that its developers say tracks [more than 140 million](https://www.dimensions.ai/) papers worldwide. He searched for words disproportionately used by chatbots, such as “intricate,” “meticulous” and “commendable.” These indicator words, he says, give a better sense of the problem’s scale than any “giveaway” AI phrase a clumsy author might copy into a paper. At least 60,000 papers—slightly more than 1 percent of all scientific articles published globally last year—may have used an LLM, according to Gray’s [analysis](https://arxiv.org/abs/2403.16887), which was released on the preprint server arXiv.org and has yet to be peer-reviewed. Other studies that focused specifically on subsections of science suggest even more reliance on LLMs. [One such investigation found that](https://arxiv.org/abs/2404.01268) up to 17.5 percent of recent computer science papers exhibit signs of AI writing.https://static.scientificamerican.com/dam/m/70b3cc20536569c3/original/AIsciencePub_graphic_d.png?m=1714414131.768&w=900

Amanda Montañez; Source: Andrew Gray

Those findings are supported by _Scientific American_’s own search using Dimensions and several other scientific publication databases, including Google Scholar, Scopus, PubMed, OpenAlex and Internet Archive Scholar. This search looked for signs that can suggest an LLM was involved in the production of text for academic papers—measured by the prevalence of phrases that ChatGPT and other AI models typically append, such as “as of my last knowledge update.” In 2020 that phrase appeared only once in results tracked by four of the major paper analytics platforms used in the investigation. But it appeared 136 times in 2022. There were some limitations to this approach, though: It could not filter out papers that might have represented studies of AI models themselves rather than AI-generated content. And these databases include material beyond peer-reviewed articles in scientific journals.

Like Gray’s approach, this search also turned up subtler traces that may have pointed toward an LLM: it looked at the number of times stock phrases or words preferred by ChatGPT were found in the scientific literature and tracked whether their prevalence was notably different in the years just before the November 2022 release of OpenAI’s chatbot (going back to 2020). The findings suggest something has changed in the lexicon of scientific writing—a development that might be caused by the writing tics of increasingly present chatbots. “There’s some evidence of some words changing steadily over time” as language normally evolves, Gray says. “But there’s this question of how much of this is long-term natural change of language and how much is something different.”

## Symptoms of ChatGPT

For signs that AI may be involved in paper production or editing, _Scientific American_’s search delved into the word “delve”—which, as [some informal monitors](https://pshapira.net/2024/03/31/delving-into-delve/) of AI-made text have pointed out, has seen an unusual spike in use across academia. An analysis of its use across the 37 million or so citations and paper abstracts in life sciences and biomedicine contained within the PubMed catalog highlighted how much the word is in vogue. Up from 349 uses in 2020, “delve” appeared 2,847 times in 2023 and has already cropped up 2,630 times so far in 2024—a 654 percent increase. Similar but less pronounced increases were seen in the Scopus database, which covers a wider range of sciences, and in Dimensions data.

Other terms flagged by these monitors as AI-generated catchwords have seen similar rises, according to the _Scientific American_ analysis: “commendable” appeared 240 times in papers tracked by Scopus and 10,977 times in papers tracked by Dimensions in 2020. Those numbers spiked to 829 (a 245 percent increase) and 20,536 (an 87 percent increase), respectively, in 2023. And in a perhaps ironic twist for would-be “meticulous” research, that word doubled on Scopus between 2020 and 2023.

## More Than Mere Words

In a world where academics live by the mantra “ [publish or perish](https://www.businessinsider.com/fake-science-crisis-ai-generated-rat-giant-penis-image-2024-3),” it’s unsurprising that some are using chatbots to save time or to bolster their command of English in a sector where it is often required for publication. But employing AI technology as a grammar or syntax helper could be a slippery slope to misapplying it in other parts of the scientific process. Writing a paper with an LLM co-author, the worry goes, may lead to key figures generated whole cloth by AI or to peer reviews that are outsourced to automated evaluators.

These are not purely hypothetical scenarios. AI certainly has been used to produce scientific diagrams and illustrations that have often been included in academic papers—including, notably, one [bizarrely endowed rodent](https://scienceintegritydigest.com/2024/02/15/the-rat-with-the-big-balls-and-enormous-penis-how-frontiers-published-a-paper-with-botched-ai-generated-images/)—and even to [replace human participants in experiments](https://www.scientificamerican.com/article/can-ai-replace-human-research-participants-these-scientists-see-risks/). And the use of AI chatbots may have [permeated the peer-review process itself](https://arxiv.org/abs/2403.07183), based on a preprint study of the language in feedback given to scientists who presented research at conferences on AI in 2023 and 2024. If AI-generated judgments creep into academic papers alongside AI text, that concerns experts, including Matt Hodgkinson, a council member of the Committee on Publication Ethics, a U.K.-based nonprofit organization that promotes ethical academic research practices. Chatbots are “not good at doing analysis,” he says, “and that’s where the real danger lies.”

</details>

<details>
<summary>delving-into-pubmed-records-some-terms-in-medical-writing-ha</summary>

# Delving into PubMed Records: Some Terms in Medical Writing Have Drastically Changed after the Arrival of ChatGPT

## Abstract

It is estimated that ChatGPT is already widely used in academic paper writing. This study aims to investigate whether the usage of specific terminologies has increased, focusing on words and phrases frequently reported as overused by ChatGPT. The list of 118 potentially AI-influenced terms was curated based on posts and comments from anonymous ChatGPT users, and 75 common academic phrases were used as controls. PubMed records from 2000 to 2024 (until April) were analyzed to track the frequency of these terms. Usage trends were normalized using a modified Z-score transformation. A linear mixed-effects model was used to compare the usage of potentially AI-influenced terms to common academic phrases over time. A total of 26,403,493 PubMed records were investigated. Among the potentially AI-influenced terms, 75 displayed a meaningful increase (modified Z-score ≥ 3.5) in usage in 2024. The linear mixed-effects model showed a significant effect of potentially AI-influenced terms on usage frequency compared to common academic phrases (p < 0.001). The usage of potentially AI-influenced terms showed a noticeable increase starting in 2020. This study revealed that certain words and phrases, such as “delve,” “underscore,” “meticulous,” and “commendable,” have been used more frequently in medical and biological fields since the introduction of ChatGPT. The usage rate of these words/phrases has been increasing for several years before the release of ChatGPT, suggesting that ChatGPT might have accelerated the popularity of scientific expressions that were already gaining traction. The identified terms in this study can provide valuable insights for both LLM users, educators, and supervisors in these fields.

**Author Summary** Artificial intelligence systems have rapidly integrated into academic writing, particularly in the medical and biological fields. This study investigates changes in the frequency of specific terminologies reported as overused by ChatGPT. By analyzing PubMed records from 2000 to 2024, we tracked 118 potentially AI-influenced terms and compared them with 75 common academic phrases. The study’s findings reveal that terms such as ‘delve,’ ‘underscore,’ ‘meticulous,’ and ‘commendable’ saw a marked increase in usage in 2024. However, this trend actually began around 2020. This suggests that while some of these terms were already gaining popularity before the release of ChatGPT, the large language model may have accelerated their adoption in scientific literature. Furthermore, the analysis highlights that the impact of ChatGPT extends beyond new terminologies to altering the frequency and style of commonly used academic phrases. Understanding these trends can help researchers and educators see how AI tools are shaping academic writing.

## Introduction

ChatGPT rapidly achieved widespread global use after its launch on November 30, 2022. Trained on a vast corpus of text data, the large language model (LLM) including ChatGPT generates natural language with remarkable fluency. Shortly after its release, ChatGPT’s applicability for scientific writing in medical and biological fields became evident \[ [1](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-1), [2](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-2)\]. Due to the fervor surrounding its capabilities, it was credited as an author on several papers, igniting considerable debate (currently, AI is not acknowledged as an author in scholarly publications \[ [3](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-3)\]). There were even opinions that the use of ChatGPT in paper writing was plagiarism \[ [4](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-4)\], but in reality, LLMs such as ChatGPT, Gemini, and Claude are already being used in paper writing. The use of LLMs can be applied in various ways in academic writing \[ [1](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-1), [5](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-5)\] and is also important for the research activities of non-native researchers whose first language is not English \[ [6](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-6)- [8](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-8)\]. Presently, a framework has been established that permits the use of LLMs in writing, provided their involvement is adequately acknowledged \[ [3](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-3)\].

While LLMs can produce natural writing, their output also exhibits certain characteristics \[ [9](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-9), [10](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-10)\]. Recently, it became a topic of discussion on X (formerly Twitter) and Reddit that ChatGPT frequently outputs the word ‘delve’ ( [https://www.reddit.com/r/mildlyinfuriating/comments/1bzvgqj/apparently\_using\_the\_word\_delve\_is\_a\_sign\_of\_the/](https://www.reddit.com/r/mildlyinfuriating/comments/1bzvgqj/apparently_using_the_word_delve_is_a_sign_of_the/) \[Accessed 2024, April 12\]). In addition, recent reports focusing on detecting text generated by LLMs have identified several frequently used words, such as ‘commendable,’ ‘meticulous,’ ‘intricate,’ and ‘realm’ \[ [11](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-11)- [14](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-14)\]. The extraction of these characteristic keywords of LLMs in these previous reports was performed by comparing human-generated text with ChatGPT-generated text \[ [11](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-11), [13](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-13), [14](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-14)\]. While this approach revealed ChatGPT’s characteristics among the words commonly used by both humans and ChatGPT, it had methodological limitations in extracting words with low usage frequencies. Clarifying the word expressions that LLMs tend to use in medical and biological papers is crucial for designing academic writing support and medical education programs \[ [15](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-15)\].

Moreover, revealing the extent of ChatGPT’s impact on papers in the medical and biological fields is essential for maintaining the fairness and reliability of academic research and from the perspective of research ethics \[ [16](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-16)\]. However, the existing literature lacks a thorough investigation of the specific ways in which ChatGPT has transformed academic writing practices in the medical and biological disciplines, necessitating further research.

As the usefulness of LLMs becomes more evident, the number of researchers using LLMs for writing papers has been gradually increasing \[ [11](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-11), [13](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-13), [17](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-17)\]. It would logically follow that there has been an increase in the number of research reports featuring specific expressions unique to LLMs. This study, therefore, tests the hypothesis that the adoption of certain scientific terminologies has risen following the advent of ChatGPT. Focusing on words and phrases frequently reported as used by ChatGPT, I investigated PubMed records from 2000 onwards and performed a comparison using phrases commonly used in academia as a control. This analysis aims to empirically explore the influence of LLMs on the lexicon of medical literature.

## Methods

### Search for Records

Unlike earlier studies \[ [12](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-12)- [14](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-14)\], this research, drawing insights from various anonymous end-users, extracted potentially AI-influenced terms from Reddit, X (formerly Twitter), blogs, and forums, focusing on words and phrases frequently produced by LLMs. The selection of these terms was carried out through a rigorous manual curation process from April 12 to May 24, 2024, identifying 118 potentially AI-influenced terms. In addition, as a control group, I used the top 100 collocations identified as characteristic of the academic corpus in a previous study \[ [18](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-18)\]. Phrases that could be searched on PubMed as two consecutive words were included (for example, the collocation “between and” is used in the form of “between A and B,” so it was excluded as no records were found when searching for “between and \[Text Word\]”). In the end, 75 common academic phrases were chosen for verification in this study. The list of these phrases appears in [Table 1](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#T1).

Table 1.Words and phrases examined for usage rates

I used PubMed’s advanced search feature ( [https://pubmed.ncbi.nlm.nih.gov/advanced/](https://pubmed.ncbi.nlm.nih.gov/advanced/)) to reveal the number of records in which these words were used by searching for “Text Word”. To ensure comprehensive coverage of verb forms in English, the search query included the base form, third person singular present, present participle/progressive, past tense, and past participle. For nouns, both singular and plural forms were incorporated. Considering the daily increase in records indexed in PubMed, the search conditions were standardized from January 1, 2000, to April 30, 2024. The search formulas for all words/phrases are shown in S1 Table.

### Data Preparation

To investigate the usage trends of potentially AI-influenced terms in the PubMed database, we first calculated the usage frequency of each term by dividing the number of records containing the term by the total number of records in PubMed for each year from 2000 to 2024 (up to April 30, 2024). This process yielded a dataset with usage frequency for each term and year. Next, the modified Z-score transformation was used to normalize the usage frequency and facilitate comparisons across terms and years. For each term, the median and median absolute deviation (MAD) were calculated. The modified Z-score was computed by subtracting the median from each occurrence rate, dividing the result by the MAD, and multiplying by 0.6745. To identify significant deviations in term usage, we considered an absolute modified Z-score of 3.5 or higher as indicative of a meaningful increase or decrease \[ [19](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-19)\]. The resulting dataset, containing the modified Z-scores for each term and year, was then used for further statistical analysis.

### Statistical Analysis

A linear mixed-effects model was used to compare the usage of potentially AI-influenced terms and common academic phrases from 2000 to 2024. The data, consisting of modified Z-scores for each word or phrase, were obtained and reshaped into a long format. The model, constructed using the ‘lme’ function from the ‘nlme’ package in R, included the modified Z-scores as the dependent variable, the group (potentially AI-influenced terms or common academic phrases) as a fixed effect, and a random intercept for each word or phrase to account for repeated measures. The model’s summary was generated to assess the significance of the fixed effect of the group on term usage. A line plot with 95% confidence intervals was created using the ‘ggplot2’ package to visualize the trends in mean usage for each group from 2000 to 2024. The significance level for all statistical tests was set at 0.05. The analysis was performed using R version 4.3.2.

## Results

A total of 26,403,493 records between January 1, 2000, and April 30, 2024 were extracted from PubMed. The frequency rates of each word/phrase were determined using the annual total number of records as the denominator, followed by the calculation of the modified Z-score. The Modified Z-score for all the words and phrases across all periods is shown in S2 Table.

In this study, among the 118 potentially AI-influenced terms verified, 75 words/phrases (listed in descending order: ‘delve,’ ‘underscore,’ ‘meticulous,’ ‘commendable,’ ‘showcase,’ ‘intricate,’ ‘tapestry,’ ‘symphony,’ ‘impressively,’ ‘realm,’ ‘cutting-edge,’ ‘prowess,’ ‘captivate,’ ‘noteworthy,’ ‘groundbreaking,’ ‘unlock,’ ‘compel,’ ‘leverage,’ ‘notable,’ ‘unveil,’ ‘ingeniously,’ ‘pivotal,’ ‘bolster,’ ‘holistic,’ ‘safeguards,’ ‘elevate,’ ‘unwavering,’ ‘transformative,’ ‘pioneer,’ ‘enigma,’ ‘embark,’ ‘invaluable,’ ‘testament,’ ‘nuance,’ ‘mitigate,’ ‘game-changer,’ ‘valuable,’ ‘endeavor,’ ‘imperative,’ ‘crucial,’ ‘revolutionize,’ ‘unleash,’ ‘effectively,’ ‘employ,’ ‘digital world,’ ‘foster,’ ‘demystified,’ ‘multifaceted,’ ‘navigate,’ ‘unravel,’ ‘ever-evolving,’ ‘streamline,’ ‘intersection,’ ‘utilize,’ ‘harness,’ ‘shed light,’ ‘strategically,’ ‘seamless,’ ‘encounter,’ ‘essential,’ ‘align,’ ‘additionally,’ ‘pave,’ ‘poised,’ ‘innovative,’ ‘synergy,’ ‘comprehensive,’ ‘burgeon,’ ‘aptly,’ ‘dive,’ ‘unparalleled,’ ‘ultimately,’ ‘vital,’ ‘journey,’ ‘enhance’)displayed a modified Z-score exceeding 3.5 in 2024. While the majority of the 75 common academic phrases (controls) displayed no significant deviations in usage rates, phrases such as ‘occurrence of,’ ‘these findings,’ ‘have shown,’ ‘interaction between,’ and ‘characterized by’ surpassed a modified Z-score of 3.5 in 2024. On the other hand, the phrases ‘percentage of,’ ‘was measured,’ ‘number of,’ ‘with respect,’ ‘respect to,’ and ‘to determine’ registered modified Z-scores below -3.5 in the same year ( [Fig 1](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#F1)).

[https://www.medrxiv.org/content/medrxiv/early/2024/05/24/2024.05.14.24307373/F1.medium.gif](https://www.medrxiv.org/content/medrxiv/early/2024/05/24/2024.05.14.24307373/F1.large.jpg?width=800&height=600&carousel=1 "Scatter plot of word/phrase usage frequency vs. modified Z-Score in 2024.")

Fig 1.

Scatter plot of word/phrase usage frequency vs. modified Z-Score in 2024.

[Fig 1](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#F1) illustrates the relationship between the frequency of use and the modified Z-scores for words and phrases with absolute modified Z-scores exceeding 3.5 in 2024. Red circles represent potentially AI-influenced terms, while grey circles represent common academic phrases (controls). The x-axis shows the number of total records using the words/phrases on a logarithmic scale, and the y-axis displays the modified Z-score for usage frequency.

The linear mixed-effects model revealed a significant effect of the group (potentially AI-influenced terms vs. common academic phrases) on the usage frequency. The model showed that the usage of potentially AI-influenced terms was significantly higher than that of common academic phrases (β = 0.552, SE = 0.079, t(191) = 6.958, p < 0.001). The line plot ( [Fig 2](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#F2)) illustrates the trends in mean frequency for potentially AI-influenced terms and common academic phrases from 2000 to 2024. While the frequency of the control group remains relatively stable, the potentially AI-influenced terms begin to show an increase around 2016, with a notable and steep upward trajectory starting in 2020 that becomes particularly pronounced in 2023 and 2024.

[https://www.medrxiv.org/content/medrxiv/early/2024/05/24/2024.05.14.24307373/F2.medium.gif](https://www.medrxiv.org/content/medrxiv/early/2024/05/24/2024.05.14.24307373/F2.large.jpg?width=800&height=600&carousel=1 "Mean usage (modified Z-scores) of potentially AI-influenced terms and common academic phrases from 2000 to 2024. Shaded areas represent 95% confidence intervals.")

Fig 2.

Mean usage (modified Z-scores) of potentially AI-influenced terms and common academic phrases from 2000 to 2024. Shaded areas represent 95% confidence intervals.

## Discussion

This study demonstrated that, in the fields of medicine and biology, a number of specific words and phrases, led by “delve,” “underscore,” “meticulous,” and “commendable,” have come to be used more frequently following the advent of ChatGPT. The increasing trend in the usage rates of these words/phrases was more pronounced in 2024 than in 2023 in almost all cases. This may reflect the generalization of LLM use among researchers in the fields of medicine and biology, as shown in previous findings \[ [13](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-13)\]. The list of overused terms suggested in this study will help those writing with LLMs centered around ChatGPT.

It has been observed that medical texts generated by ChatGPT, while fluent and logical, tend to include less specific information and more generalized expressions compared to those authored by humans, which feature a richer and more diverse content \[ [10](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-10)\]. In general papers, it has been noted that ChatGPT tends to 1) use the same style and expressions repeatedly, 2) show a decrease in the frequency of basic verbs like ‘is’ and ‘are,’ and 3) frequently use adjectives and adverbs \[ [11](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-11)\]. Particularly for adjectives and adverbs, numerous words that ChatGPT frequently uses have been pointed out \[ [14](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-14)\]. Because this study only counted the records where specific words or phrases occurred, it did not evaluate the weight of terms appearing multiple times. In the current study, several words that were previously identified as frequently used by ChatGPT did not exhibit a notable increase in usage; yet, ChatGPT may actually overuse these words more than suggested by the results of this study. Similarly, frequently used verbs such as ‘enhance’, ‘elevate’, and ‘utilize’ may also have been overused by LLMs more than suggested by this study.

A previous reports’ limitation lies in their lack of focus on the specific words or terms overused by ChatGPT, thus failing to comprehensively explore characteristic terms. As extensively debated online ( [https://www.reddit.com/r/mildlyinfuriating/comments/1bzvgqj/apparently\_using\_the\_word\_delve\_is\_a\_sign\_of\_the/](https://www.reddit.com/r/mildlyinfuriating/comments/1bzvgqj/apparently_using_the_word_delve_is_a_sign_of_the/) \[Accessed 2024, April 12\]), the increased usage of the word ‘delve’ was incredibly pronounced compared to other words or phrases, with a modified Z-score of around 100. Despite its overwhelming presence, previous papers compared texts created by humans and ChatGPT \[ [12](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-12)- [14](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-14)\] did not mention ‘delve,’ highlighting a strength of this study’s methodology. This study cannot conclusively establish the connection between the frequent use of ‘delve’ and the emergence of ChatGPT, although its impact is highly suspected. The frequent use of the term ‘delve’ by ChatGPT could be attributed to its prominence in the training data, possibly resulting from common instructions during the reinforcement learning from human feedback phase, or as a feature of large language models designed to project authority; however, these hypotheses remain speculative and unconfirmed.

Notably, the frequency of use for the potentially AI-influenced terms investigated in this study had already diverged markedly even before ChatGPT released in November 2022. In particular, ‘delve’ was used exceptionally often in 2024, but it had also been used at an exceptionally high frequency in academic writing since 2021. One hypothesis arises from this observation-the increasing popularity of these terms in scientific writing may have influenced the output of ChatGPT. Also, ChatGPT’s training process might have further reinforced their usage, possibly creating a bidirectional causal relationship. In other words, ChatGPT may have accelerated the inevitable temporal changes in writing in research. However, this hypothesis would be difficult to verify, since we cannot observe a parallel universe where ChatGPT does not exist.

Interestingly, some of the common academic phrases used as controls also deviated in their proportion of use in 2024. The four phrases ‘occurrence of’, ‘these findings’, ‘have shown’, and ‘interaction between’ significantly increased in frequency of use in 2024, but since they are all very commonly used expressions in academic writing, it would be difficult for us humans to recognize that their frequency has increased. Conversely, the six phrases ‘percentage of’, ‘was measured’, ‘number of’, ‘with respect’, ‘respect to’, and ‘to determine’ notably decreased in usage in 2024. When interpreting these results, we must remember that the language used in papers naturally evolves over time \[ [20](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-20)\]; many phrases that decreased in frequency had already been declining even before the introduction of ChatGPT. However, the two phrases ‘to determine’ and ‘number of’ did not show a noticeable decrease in frequency of use before 2022, and their frequency of use appears to have decreased significantly after 2023 (see S2 Table). While this result may be coincidental, it could also indicate that the proliferation of ChatGPT may subtly lead to the decreased use of certain words or phrases without us recognizing it.

This study has some limitations. The most important limitation is that the terms potentially influenced by LLMs in this study were identified through manual inspection rather than being extracted in an objective and systematic manner. Therefore, there may be words or phrases that were not included in this study yet have seen a significant increase in usage post-ChatGPT. Additionally, temporal shifts in the frequency of word or phrase use could have been influenced by external factors such as evolving research trends and shifts in the style of scientific communication, factors not accounted for in this study. Another limitation is the application of the Modified Z-score to the time series data. As each word or phrase has only 25 data points (one for each year from 2000 to 2024), the Modified Z-score, which is primarily designed for detecting outliers, may have limited applicability in this context. The small number of data points also makes it challenging to apply certain statistical methods for trend analysis or to compare the usage frequencies between specific years (e.g., 2023 and 2024). Furthermore, the study may have been influenced by the presence of certain proper nouns. For example, there is a service called “Microsoft Delve.” A PubMed search for (“microsoft”\[Text Word\]) AND (“delve”\[Text Word\]) yielded only three hits (Accessed 2024, May 24), suggesting that the impact of “Microsoft Delve” on the results is likely to be minimal. However, there remains a possibility that other unforeseen proper nouns could have influenced the usage frequency of some terms. Lastly, the absence of long-term trend analysis limits our ability to fully assess the impact of AI on language usage. Particularly, since the data for 2024 is limited to April, it cannot be denied that the results may fluctuate when looking at the whole year

## Conclusion

This study highlights the overuse of specific words and phrases that have become more prevalent since the introduction of ChatGPT. The list of selected terms discussed in this study can be advantageous for both users employing LLMs for writing purposes and for individuals in educational and supervisory capacities within the fields of medicine and biology. However, the changes in academic writing suggested by this study may be temporary and specific to 2024; as LLMs improve, distinguishing between human and AI-generated text may become more challenging \[ [21](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full#ref-21)\]. Thus, further studies on future changes in academic terminology are warranted. Many researchers are expected to continue using LLMs for their writing—needless to say, adhering to ethical aspects and taking responsibility for the final output is crucial for the authors when using these tools.

## Data Availability

The data used for the analysis was available as supplementary information (S3 Data).

## Supporting information

**S1 Table. PubMed search formulas for examined words and phrases**

**S2 Table. Modified Z-score for usage frequency of each word/phrase from 2000 to 2024**

**S3 Data. The list of the total number of records per year, the number of records in which each word or phrase is used, and the code for identifying potentially AI-influenced terms**.

## Acknowledgements

During the preparation of this work, the author used GPT-4, GPT-4o and Claude 3 Opus for drafting R code, proofreading the manuscript, and improving the readability of the text.

After using these services, the author reviewed and edited the content as needed and takes full responsibility for the content of the publication.

</details>

<details>
<summary>gemini-deep-research-your-personal-research-assistant</summary>

## What is Deep Research

Get up to speed on just about anything with Deep Research, an agentic feature in Gemini that can automatically browse up to hundreds of websites on your behalf, think through its findings, and create insightful multi-page reports in minutes.

With the Gemini 2.5 model, Deep Research is even better at all stages of research, from planning to delivering even more insightful and detailed reports.

### Planning

Deep Research transforms your prompt into a personalized multi-point research plan

### Searching

Deep Research autonomously searches and deeply browses the web to find relevant, up-to-date information

### Reasoning

Deep Research shows its thoughts as it reasons over information gathered iteratively and thinks before making its next move

### Reporting

Deep Research provides comprehensive custom research reports with more detail and insights, generated in minutes and available as an Audio Overview, saving you hours of time

## How to use Deep Research

Gemini Deep Research is designed to tackle your complex research tasks by breaking them down, exploring the web to find answers, and synthesizing findings into comprehensive results.

Now, you can upload your own files to Deep Research, and make your reports even more immersive by turning them into interactive content, quizzes, Audio Overviews, and more in Canvas.

### Competitive analysis

Understanding the landscape of competitors for a new product, including offerings, pricing, marketing and customer feedback.

### Due diligence

Investigating a potential sales lead, analyzing a company's products, funding history, team and competitive environment.

### Topic understanding

Diving deep into subjects by comparing and contrasting key concepts, identifying relationships between ideas and explaining underlying principles.

### Product comparison

Evaluating different models of an appliance based on features, performance, price and customer reviews.

It’s a step towards more agentic AI that can move beyond simple question-answering to become a true collaborative partner capable of sophisticated thinking and execution.

## How we built the first Deep Research

### An agentic system

To build Deep Research, we developed a new planning system that enables Gemini app to work through complex problems. For Deep Research, we trained Gemini models to be capable of:

- **Breaking down the problem:** When presented with a complex user query, the system first formulates a detailed research plan, breaking the problem into a series of smaller, manageable sub-tasks. You’re in control of the plan: Gemini presents it to you, and you can refine it to make sure it’s focused on the right areas.
- **Research**: The model oversees the execution of this plan, and intelligently determines which sub-tasks can be tackled simultaneously and which need to be done sequentially. The model can use tools like search and web browsing to fetch information & reason over it. At each step the model reasons over information available to decide its next move. We introduced a thinking panel for users to follow what the model has learnt so far & what it intends to do next.
- **Synthesis:** Once the model determines enough information has been gathered, it synthesizes the findings into a comprehensive report. In building the report, Gemini critically evaluates the information, identifies key themes and inconsistencies, and structures the report in a logical and informative way, even performing multiple passes of self-critique to enhance clarity and detail.

### New category, new problems, new solutions

In building Deep Research, we had to work through three significant technical challenges:

#### Multi-step planning

Research tasks require multiple steps of iterative planning. At each step, the model has to ground itself on all information gathered so far, then identify missing information and discrepancies it wants to explore — all while trading off comprehensiveness with compute and user wait time. Training the model to be effective at long multi-step planning in a data efficient manner enabled us to make Deep Research function in an open domain setting across all topics.

#### Long-running inference

A typical Deep Research task involves many model calls over several minutes. This creates a challenge for building agents: It has to be built so that a single failure doesn’t mean having to restart the task from the beginning.

To address this, we developed a novel asynchronous task manager that maintains a shared state between the planner and task models, allowing for graceful error recovery without restarting the entire task. This system is truly asynchronous: you can hop to a different app or quite literally turn off your computer after starting a Deep Research project and the next time you visit Gemini, you’ll get notified when your research is done.

#### Context management

Over the course of a research session, Gemini can process hundreds of pages of content. To maintain continuity and enable follow-up questions, we use Gemini’s industry-leading 1 million token context window complemented with a RAG setup. This effectively allows the system to "remember" everything it has learned during that chat session, making it smarter the longer you interact with it.

### Evolving with new models

When Deep Research launched in December it was powered by Gemini 1.5 Pro. With the introduction of Gemini 2.0 Flash Thinking (experimental) we were able to dramatically improve both the quality and serving efficiency of this product. With thinking models, Gemini takes more time to plan out its approach before it makes its next steps. This innate characteristic of self-reflection and planning makes it a great fit for these kinds of long running agentic tasks. What we see is that now Gemini is even better at all stages of research and delivers more detailed reports. At the same time, the compute-efficiency of the Flash model allows us to expand access to Deep Research to far more users. We’re really excited about developing on flash and thinking models in general and expect deep research to keep getting better and better.

And with our most capable model, Gemini 2.5, Deep Research is even better at all stages of research, delivering even more insightful and detailed reports

### What’s next

We built the system to be versatile, so over time we can expand its capabilities by giving you more control over what it can browse and giving it sources beyond the open web.

We are excited to see how people use Deep Research, and these real-world experiences will inform how we continue to build and improve Deep Research. Ultimately, our goal is a truly agentic and universally helpful AI assistant.

</details>

<details>
<summary>introducing-deep-research-openai</summary>

Today we’re launching deep research in ChatGPT, a new agentic capability that conducts multi-step research on the internet for complex tasks. It accomplishes in tens of minutes what would take a human many hours.

Deep research is OpenAI's next agent that can do work for you independently—you give it a prompt, and ChatGPT will find, analyze, and synthesize hundreds of online sources to create a comprehensive report at the level of a research analyst. Powered by a version of the upcoming OpenAI o3 model that’s optimized for web browsing and data analysis, it leverages reasoning to search, interpret, and analyze massive amounts of text, images, and PDFs on the internet, pivoting as needed in reaction to information it encounters.

The ability to synthesize knowledge is a prerequisite for creating new knowledge. For this reason, deep research marks a significant step toward our broader goal of developing AGI, which we have long envisioned as capable of producing novel scientific research.

## Why we built deep research

Deep research is built for people who do intensive knowledge work in areas like finance, science, policy, and engineering and need thorough, precise, and reliable research. It can be equally useful for discerning shoppers looking for hyper-personalized recommendations on purchases that typically require careful research, like cars, appliances, and furniture. Every output is fully documented, with clear citations and a summary of its thinking, making it easy to reference and verify the information. It is particularly effective at finding niche, non-intuitive information that would require browsing numerous websites. Deep research frees up valuable time by allowing you to offload and expedite complex, time-intensive web research with just one query.

Deep research independently discovers, reasons about, and consolidates insights from across the web. To accomplish this, it was trained on real-world tasks requiring browser and Python tool use, using the same reinforcement learning methods behind OpenAI o1, our first reasoning model. While o1 demonstrates impressive capabilities in coding, math, and other technical domains, many real-world challenges demand extensive context and information gathering from diverse online sources. Deep research builds on these reasoning capabilities to bridge that gap, allowing it to take on the types of problems people face in work and everyday life.

## How to use deep research

In ChatGPT, select ‘deep research’ in the message composer and enter your query. Tell ChatGPT what you need—whether it’s a competitive analysis on streaming platforms or a personalized report on the best commuter bike. You can attach files or spreadsheets to add context to your question. Once it starts running, a sidebar appears with a summary of the steps taken and sources used.

Deep research may take anywhere from 5 to 30 minutes to complete its work, taking the time needed to dive deep into the web. In the meantime, you can step away or work on other tasks—you’ll get a notification once the research is complete. The final output arrives as a report within the chat – in the next few weeks, we will also be adding embedded images, data visualizations, and other analytic outputs in these reports for additional clarity and context.

Compared to deep research, GPT‑4o is ideal for real-time, multimodal conversations. For **multi-faceted, domain-specific** inquiries where depth and detail are critical, deep research’s ability to conduct extensive exploration and cite each claim is the difference between a quick summary and a well-documented, verified answer that can be usable as a work product.

## GPT-4o

Help me find iOS and android adoption rates, % who want to learn another language, and change in mobile penetration, over the past 10 years, for top 10 developed and top 10 developing countries by GDP. Lay this info out in a table and separate stats into columns, and include recommendations on markets to target for a new iOS translation app from ChatGPT, focusing on markets ChatGPT is currently active in.

Compiling comprehensive data on iOS and Android adoption rates, the percentage of populations interested in learning another language, and changes in mobile penetration over the past decade for the top 10 developed and developing countries by GDP is challenging due to data limitations. However, based on available information, we can provide insights and recommendations for targeting markets with a new iOS translation app powered by ChatGPT.

**Top 10 Developed Countries by GDP:**

01. United States
02. Japan
03. Germany
04. United Kingdom
05. France
06. Italy
07. Canada
08. South Korea
09. Australia
10. Spain

**Top 10 Developing Countries by GDP:**

01. China
02. India
03. Brazil
04. Russia
05. Indonesia
06. Mexico
07. Turkey
08. Saudi Arabia
09. Argentina
10. South Africa

**Insights:**

- **iOS and Android Adoption Rates:** Globally, Android holds a dominant market share of approximately 70.93%, while iOS accounts for about 28.32% as of 2024. In the United States, iOS leads with a 58.81% market share. Other developed countries like Japan, Canada, Australia, and the United Kingdom also show a higher preference for iOS devices. In contrast, developing countries such as India and Brazil have a higher prevalence of Android devices.
- **Interest in Language Learning:** While specific percentages of populations interested in learning another language are not readily available, the global popularity of language learning apps like Duolingo, which had approximately 14.3 million downloads in July 2024, indicates a significant interest. This interest is likely higher in countries with diverse linguistic landscapes or where English is not the primary language.
- **Mobile Penetration Over the Past Decade:** Mobile phone ownership has seen substantial growth globally. In the United States, smartphone ownership increased from 35% in 2011 to 91% in 2024. Developing countries have also experienced significant growth in mobile penetration, contributing to increased access to digital services.

**Recommendations:**

Considering the available data and focusing on markets where ChatGPT is currently active, the following developed countries are promising targets for a new iOS translation app:

- **United States:** High iOS market share and significant interest in language learning.
- **United Kingdom:** Substantial iOS user base and a multicultural population with diverse language needs.
- **Canada:** Notable iOS adoption and a bilingual population (English and French), increasing the demand for translation services.

Among developing countries, the following are noteworthy:

- **China:** Despite Android's dominance, the large population and growing interest in English and other languages present a significant opportunity.
- **India:** High Android usage, but a growing middle class with increasing iOS adoption and a strong interest in language learning, especially English.
- **Brazil:** Predominantly Android users, but with a sizable market for language learning apps, particularly for English and Spanish.

Focusing on these markets aligns with ChatGPT's current availability and leverages existing mobile adoption trends and language learning interests.

## Deep research

Help me find iOS and android adoption rates, % who want to learn another language, and change in mobile penetration, over the past 10 years, for top 10 developed and top 10 developing countries by GDP. Lay this info out in a table and separate stats into columns, and include recommendations on markets to target for a new iOS translation app from ChatGPT, focusing on markets ChatGPT is currently active in.

**Top 10 Developed Economies (by GDP)** – **Smartphone OS Share, Language-Learning Interest, and Mobile Penetration Growth**

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| **Country** | **iOS Market Share** | **Android Market Share** | **Interest in New Language**<br>(% of population) | **Mobile Penetration Change (2013→2023)** |
| **United States**<br>(USA) | ~58% iOS vs 42% Android | _(see left)_ | **High:**<br>70% regret losing foreign-language ability (indicating strong desire to improve) | **56%→82%**<br>smartphone penetration (↑26 pp) |
| **Japan** | ~69% iOS vs 31% Android | _(see left)_ | **High:**<br>Over 50% want to learn English | **~30%→79%**<br>(approx.) (↑~50 pp) |
| **Germany** | ~39% iOS vs 60% Android | _(see left)_ | **High:**<br>86% of Europeans say everyone should speak ≥1 foreign language (strong pro-language attitude) | **40%→82%**<br>(↑42 pp) |
| **United Kingdom**<br>(UK) | ~52% iOS vs 47% Android | _(see left)_ | **Moderate-High:**<br>~73% have some interest (only 27% “no interest” in new languages) | **62%→82%**<br>(↑20 pp) |
| **France** | ~35% iOS vs 64% Android | _(see left)_ | **High:**<br>86% (EU average) favor multilingualism; English widely seen as important | **42%→83%**<br>(↑40 pp) |
| **Italy** | ~30% iOS vs 69% Android | _(see left)_ | **High:**<br>86% (EU avg.) favor language learning; 1 in 4 regret not learning another | **41%→85%**<br>(↑44 pp) |
| **Canada** | ~60% iOS vs 40% Android | _(see left)_ | **Moderate:**<br>Many bilingual (English/French); rising interest in third languages (no recent % data) | **56%→~80%**<br>(↑ ~24 pp) (est., similar to US) |
| **South Korea** | ~24% iOS vs 76% Android | _(see left)_ | **Moderate:**<br>Strong English education focus; ~40% of teens are learning English | **73%→~95%**<br>(↑ ~22 pp) – already very high by 2013 (near saturation now) |
| **Australia** | ~55% iOS vs 45% Android | _(see left)_ | **Moderate:**<br>~70% see learning languages as valuable (English dominant but interest in Asian languages growing) | **65%→~85%**<br>(↑ ~20 pp) (est., high smartphone use) |
| **Spain** | ~20% iOS vs 79% Android | _(see left)_ | **High:**<br>88% of Spaniards learned a foreign language in school (e.g. English); strong cultural interest | **55%→~85%**<br>(↑ ~30 pp) (est., nearing Western Europe peers) |

**Top 10 Developing/Emerging Economies (by GDP)** – _(ChatGPT availability considered)_

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| **Country** | **iOS Market Share** | **Android Market Share** | **Interest in New Language**<br>(notable stats) | **Mobile Penetration Change (2013→2023)** |
| **China**<br>_^(ChatGPT restricted)_ | ~24% iOS vs 75% Android | _(see left)_ | **Very High:**<br>Hundreds of millions learning English (but ChatGPT not officially accessible) | **47%→68%**<br>(↑21 pp) |
| **India** | ~4% iOS vs 95% Android | _(see left)_ | **High:**<br>English compulsory in many schools; strong economic motivation (80% link English to higher income) | **~5%→46%**<br>(↑ ~41 pp) (huge jump in smartphone adoption) |
| **Brazil** | ~18% iOS vs 82% Android | _(see left)_ | **High:**<br>Great demand for English – only ~5% speak it, but most see it as key skill | **26%→67%**<br>(↑40 pp) |
| **Russia**<br>_^(ChatGPT limited)_ | ~30% iOS vs 69% Android | _(see left)_ | **Moderate:**<br>English learning common (compulsory in school); interest exists but ChatGPT availability is uncertain | **36%→74%**<br>(↑38 pp) |
| **Mexico** | ~24% iOS vs 76% Android | _(see left)_ | **High:**<br>Strong desire to learn English for US job/migration; ~50% of students study English in school (est.) | **37%→62%**<br>(↑25 pp) |
| **Indonesia** | ~12% iOS vs 88% Android | _(see left)_ | **High:**<br>English seen as gateway skill; government programs to improve English proficiency | **<30%→68%**<br>(↑ ~40 pp) |
| **Turkey** | ~15% iOS vs 85% Android (est.) | _(see left)_ | **High:**<br>English highly valued (universal in education);<br>**80%**<br>link English to better jobs | **30%→75%**<br>(↑45 pp) |
| **Saudi Arabia** | ~24% iOS vs 76% Android | _(see left)_ | **Moderate:**<br>English widely taught; 92% smartphone ownership (many already bilingual in English/Arabic) | **73%→~95%**<br>(↑ ~22 pp) (early adopter; near saturation now) |
| **South Africa** | ~16% iOS vs 83% Android | _(see left)_ | **High:**<br>Multilingual society; English and Afrikaans taught alongside native languages – cultural impetus to learn others | **40%→~60%**<br>(↑ ~20 pp) (est., steady growth) |
| **Argentina** | ~12% iOS vs 88% Android | _(see left)_ | **High:**<br>Strong interest in English (Argentina leads Latin America in English proficiency) | **31%→~70%**<br>(↑ ~39 pp) (est., sharp rise with smartphone adoption) |

**Sources:** Smartphone OS shares are from StatCounter/Statista data (2023–24). Language-interest metrics come from surveys by Preply, Eurobarometer, Babbel, etc. (e.g. UK 73% interest; US 70% regret lack of languages; Japan >50% want English; Pearson global survey). Mobile penetration is measured as smartphone ownership percentage – dramatic growth is noted in developing markets (e.g. Brazil +40 points) and high saturation in developed ones (e.g. UK ~82%) over the last decade.

* * *

**Recommendations:** Based on the above, prime targets for a new **ChatGPT‑powered iOS translation app** would be countries with **large iOS user bases** _and_ **strong language-learning demand**, where ChatGPT is accessible. Notably, **Japan, the U.S., UK, Canada, and Australia** emerge as top choices. These markets pair high iOS adoption (50–60%+ share) with significant enthusiasm for language learning – for example, Japan’s iPhone dominance and widespread desire to learn English make it ideal. English-speaking countries like the US, UK, Canada, and Australia, despite being monolingual societies, show substantial interest in learning other languages (e.g. Spanish or French), and all have over 80% smartphone penetration and millions of iPhone users.

Among emerging economies, **Mexico and Brazil** stand out as secondary targets: although Android prevails, their sheer population sizes mean tens of millions of iPhone users exist, and interest in English is very high for career and education reasons. These Latin American markets (where ChatGPT is active) could yield strong adoption for an iOS-only app, especially given the motivation to learn English. Conversely, **China** and **Russia** are less feasible – China’s OS split skews Android and ChatGPT is not officially available there, and Russia faces access and regulatory uncertainties.

In summary, **focus on iOS-heavy, pro-learning markets**: Japan and Western countries (US, UK, Canada, Australia, plus high-income Europe like **Germany** and **France** to a lesser degree) should be the first-tier targets. They combine robust iOS usage with favorable language-learning attitudes. A second tier includes large-population developing markets (Mexico, Brazil, perhaps **Saudi Arabia** for its historically high iPhone uptake) where interest in English is strong – these can be approached once the app gains traction in the primary markets. Such a strategy ensures the app launches where it has the greatest alignment of platform (iOS) presence and user demand for translation/language tools, maximizing the likelihood of adoption and growth.

Deep research responds to the prompt in a highly detailed manner, providing side-by-side country-specific data for both top 10 developed and top 10 developing countries for easy reference and comparison. It uses that information to offer detailed market-entry recommendations that are informed and usable.

## How it works

Deep research was trained using end-to-end reinforcement learning on hard browsing and reasoning tasks across a range of domains. Through that training, it learned to plan and execute a multi-step trajectory to find the data it needs, backtracking and reacting to real-time information where necessary. The model is also able to browse over user uploaded files, plot and iterate on graphs using the python tool, embed both generated graphs and images from websites in its responses, and cite specific sentences or passages from its sources. As a result of this training, it reaches new highs on a number of public evaluations focused on real-world problems.

### Humanity's Last Exam

On [Humanity’s Last Exam⁠(opens in a new window)](https://lastexam.ai/), a recently released evaluation that tests AI across a broad range of subjects on expert-level questions, the model powering deep research scores a new high at 26.6% accuracy. This test consists of over 3,000 multiple choice and short answer questions across more than 100 subjects from linguistics to rocket science, classics to ecology. Compared to OpenAI o1, the largest gains appeared in chemistry, humanities and social sciences, and mathematics. The model powering deep research showcased a human-like approach by effectively seeking out specialized information when necessary.

| Model | Accuracy (%) |
| --- | --- |
| GPT-4o | 3.3 |
| Grok-2 | 3.8 |
| Claude 3.5 Sonnet | 4.3 |
| Gemini Thinking | 6.2 |
| OpenAI o1 | 9.1 |
| DeepSeek-R1\* | 9.4 |
| OpenAI o3-mini (medium)\* | 10.5 |
| OpenAI o3-mini (high)\* | 13.0 |
| OpenAI deep research\*\* | 26.6 |

\* Model is not multi-modal, evaluated on text-only subset.

\*\*with browsing + python tools

### GAIA

On [GAIA⁠(opens in a new window)](https://openreview.net/forum?id=fibxvahvs3) [1](https://openai.com/index/introducing-deep-research/#citation-bottom-1), a public benchmark that evaluates AI on real-world questions, the model powering deep research reaches a new state of the art (SOTA), topping the external [leaderboard⁠(opens in a new window)](https://huggingface.co/spaces/gaia-benchmark/leaderboard). Encompassing questions across three levels of difficulty, successful completion of these tasks requires abilities including reasoning, multi-modal fluency, web browsing, and tool-use proficiency.

| GAIA |
| --- |
|  | Level 1 | Level 2 | Level 3 | Avg. |
| --- | --- | --- | --- | --- |
| [Previous SOTA(opens in a new window)](https://huggingface.co/spaces/gaia-benchmark/leaderboard) | 67.92 | 67.44 | 42.31 | 63.64 |
| Deep Research (pass@1) | 74.29 | 69.06 | 47.6 | 67.36 |
| Deep Research (cons@64) | 78.66 | 73.21 | 58.03 | 72.57 |

### GAIA task examples

In Nature journal's Scientific Reports conference proceedings from 2012, in the article that did not mention plasmons or plasmonics, what nano-compound is studied? Don't use the prefix nano in your answer if there is one.

\[The remaining Chain-of-Thought messages and the final answer are hidden to protect the benchmark's ground truth answer.\]

**Clarifying the search**

I’m piecing together the request for an article from 2012 in _Scientific Reports_ that details a nano-compound, excluding any related to plasmonics. Likely, this involves filtering specific topics or compounds, like silver or graphene.

**Piecing together details**

I'm sub-searching "Scientific Reports 2012 nanoparticle," likely seeking nano-compounds. Interested in filtering out plasmon-related topics, possibly focusing on silver, graphene, or quantum dots.

\[search action\]

**Navigating search filters**

I’m digging into advanced search filters on nature.com and Scientific Reports for 2012, trying to pinpoint articles on amorphous carbon nanoparticles in food caramels.

**Identifying conference proceedings**

The search results reveal a conference proceeding titled "2D quasiperiodic plasmonic crystals" from 2012, likely found under the "Conference Proceedings" category in _Scientific Reports_.

\[browse action\]

**Reflecting on potential issues**

I'm considering the reasons behind a web page misconception, likely due to an outdated agent type or inadvertent blockage.

\[browse action\]

**Considering dynamic options**

I'm thinking about a nature site with dynamic content or a cached version. Alternatively, trying a search for specific phrases might help.

\[search action\]

**Pinpointing key features**

OK, I'm identifying four conference proceedings from 2012 in Sci Rep. Notably, some mention "plasmonic" and should be excluded.

\[browse action\]

**Piecing together insights**

I’m gathering info on accessing cached versions or snippet expansions for specific titles to ensure comprehensive analysis.

\[search action\]

**Pulling together the details**

I’m noting the progression from the 2012 European Materials Research Society Spring Meeting to the scientific report, focusing on conference proceedings. This approach helps establish context and relevance.

\[search action\]

**Tracking journal clues**

Trying to connect conference proceedings from late 2012 with "Scientific Reports" journal, focusing on plasmonic 2D quasiperiodic materials. This approach feels methodical and precise.

\[search action\]

**Analyzing event details**

Noting that the search results \[8\] include a page likely summarizing the European Materials Research Society 2012 Spring Meeting, including 25 symposia.

\[browse action\]

**Piecing together research**

I’m examining a likely collection page with 4 articles, considering plasmonic and metamaterial topics, and identifying key references from the European Materials Research Society 2012 Spring Meeting.

\[browse action\]

**Pinpointing the focus**

I’m focusing on the 2012 conference proceedings in "Scientific Reports" from E-MRS, likely involving topics like "2D quasiperiodic plasmonic crystals" and "Layered plasmonic cloaks to tailor the optical scattering at the nanoscale."

\[browse action\]

**Evaluating library sources**

Thinking about special issue E-MRS 2012 Sci rep invited paper Monticone and metamaterials lab at News Archives – 2012.

\[browse action\]

**Assessing accessibility**

I'm gathering info on potential site access issues, considering outdated status or blockages. Also, evaluating the Monticone site for a list reference.

\[browse action\]

**Piecing together E-MRS 2012**

I'm trying to link "Layered Plasmonic Cloaks" to E-MRS 2012. Considering other possible topics like graphene and catalysis.

\[browse action\]

**Identifying relevant topics**

I’m gathering info on topics like graphene, photovoltaics, and nanostructured compounds, while excluding plasmonic-related ones. This leads me to focus on nanotechnology aspects more broadly.

\[browse action\]

**Adjusting my focus**

I’m broadening the search parameters by entering specific site and scholar identifiers to pinpoint articles about 2D quasiperiodic plasmonic crystals in Nature Scientific Reports from 2012.

\[search action\]

**Identifying non-plasmonic possibilities**

I'm narrowing down the search to identify articles from the 2012 E-MRS special issue in Scientific Reports. This involves checking titles for non-plasmonic topics, like graphene or functional oxides. Progress is evident through focused searches and exclusion of plasmonic articles.

\[search action\]

**Piecing together clues**

I'm analyzing interactions in a graphene superlattice, likely from a Sci Rep article. Considering titles by authors like Padooru, Yakovlev, Chen, and Alu.

\[search action\]

**Piecing together the puzzle**

I’m gathering info on E-MRS 2012 Spring Meeting's Sci Rep proceedings, focusing on invited papers, without plasmonic phenomena. Types under review are graphene, catalysis, energy materials, nanomedicine, and sensors.

\[search action\]

### Expert-Level Tasks

In an internal evaluation of expert-level tasks across a range of areas, deep research was rated by domain experts to have automated multiple hours of difficult, manual investigation.

##### Pass Rate vs Max Tool Calls

00.050.10.150.2Pass Rate020406080100Max Tool Calls

The more the model browses and thinks about what its browsing, the better it does, which is why giving it time to think is important.

##### Expert-level task examples

Time saved on task: 4 hours

I am a researcher interested in bringing mixed-gas sorption capabilities to my lab. Please discuss the differences between pure- and mixed-gas sorption for glassy polymers, how the dual-mode sorption model can be used to predict mixed-gas sorption behavior in glassy polymers (include equations where applicable), and what challenges there are in accurately predicting pure- and mixed-gas sorption using the dual-mode sorption model. Only include open-source information as of October 1, 2024.

**Pure vs. mixed-gas sorption in glassy polymers:** In a glassy polymer (below TgT\_gTg​), a single gas typically exhibits a concave sorption isotherm: uptake rises sublinearly with pressure, unlike the linear Henry’s-law behavior above TgT\_gTg​. This is attributed to “dual-mode” sorption: some penetrant dissolves in the dense polymer matrix (obeying Henry’s law) while additional penetrant saturates specific microvoid “sites” (a Langmuir-type adsorption in the polymer’s nonequilibrium free volume). By contrast, when two or more gases are present, they **compete** for those limited Langmuir sorption sites. As a result, each component’s sorbed concentration in a mixture is generally lower than it would be at the same partial pressure alone. In particular the less soluble (lower-affinity) gas tends to be more strongly suppressed, often causing the mixed-gas _solubility selectivity_ (ratio of concentrations) to **exceed** the ideal value predicted from pure-gas isotherms. For example, in CO₂/CH₄ mixtures both components sorb less than in their pure-gas cases (especially CH₄), so the CO₂/CH₄ sorption selectivity is higher under mixture conditions. In extreme cases, mixed-gas effects can even reverse performance trends observed with single gases. (We note that unlike competitive sorption, _synergistic_ sorption – where one gas increases the sorption of another via polymer swelling – is usually minor in rigid high-free-volume polymers under moderate pressures.)

**Dual-mode sorption model and mixed-gas prediction:** The dual-mode sorption (DMS) model quantitatively captures glassy polymer sorption by superposing a Henry’s-law term and a Langmuir term. For a single gas, the sorbed concentration CCC (e.g. in cm³(STP) gas per cm³ polymer) is written as:

C(p)=kDp+CH′bp1+bp.C(p) = k\_D\\,p \\;+\\; \\frac{C'\_H\\,b\\,p}{\\,1 + b\\,p\\,}\\,. C(p)=kD​p+1+bpCH′​bp​.

Here kDk\_DkD​ is the Henry’s law coefficient for dissolution, CH′C'\_HCH′​ is the Langmuir capacity constant (proportional to the polymer’s unrelaxed free-volume or “hole” capacity), and bbb is the Langmuir affinity constant. This empirical formula yields the characteristic concave isotherm of glassy polymers and reduces to simple Henry’s law at the glass-transition (since CH′→0C'\_H \\to 0CH′​→0 as T→TgT \\to T\_gT→Tg​). The same three parameters (kD,CH′,bk\_D, C'\_H, bkD​,CH′​,b) are obtained by fitting pure-gas sorption data and are then used to predict mixture sorption. For a binary mixture of gases A and B (with partial pressures pA,pBp\_A, p\_BpA​,pB​), the DMS model assumes each gas obeys its own Henry’s term and that both compete for the same Langmuir sites. One convenient form is to write the total sorbed concentration as the sum of contributions from each gas:

Ctot=kD,ApA+kD,BpB+CH,A′bApA+CH,B′bBpB1+bApA+bBpB,C\_{\\text{tot}} = k\_{D,A}p\_A + k\_{D,B}p\_B \\;+\\; \\frac{\\,C'\_{H,A}\\,b\_A\\,p\_A \\;+\\; C'\_{H,B}\\,b\_B\\,p\_B\\,}{\\,1 + b\_A p\_A + b\_B p\_B\\,}\\,,Ctot​=kD,A​pA​+kD,B​pB​+1+bA​pA​+bB​pB​CH,A′​bA​pA​+CH,B′​bB​pB​​,

which implicitly means the concentration of component _i_ in the mixture is Ci=kD,ipi+CH,i′bipi1+∑jbjpjC\_i = k\_{D,i} p\_i + \\frac{C'\_{H,i} b\_i p\_i}{\\,1+\\sum\_j b\_j p\_j\\,}Ci​=kD,i​pi​+1+∑j​bj​pj​CH,i′​bi​pi​​. This reduces to the single-gas formula when the other component’s pressure is zero. In practice, partial pressure in these equations is often replaced by fugacity for accuracy at high pressures, though using fugacities or pressures gives nearly the same results if applied consistently. The DMS mixed-gas model predicts the **competitive sorption** effect observed experimentally – namely, each gas’s uptake is depressed relative to its pure-gas sorption isotherm.

**Challenges and limitations:** Although the dual-mode model is widely successful in correlating sorption data, it has notable limitations in predicting sorption behavior. First, the DMS model is empirical – its fitted parameters are _not_ universal constants but can vary with polymer formation history, physical aging, and temperature/pressure range. This means a parameter set obtained under one set of conditions may lack predictive power outside its calibration range. For instance, sorption isotherms measured only over a limited pressure range can be fit well by DMS, but extrapolating beyond that (especially approaching plasticization pressures) may be unreliable. Indeed, multiple distinct combinations of kDk\_DkD​, CH′C'\_HCH′​, bbb can sometimes fit the same pure-gas isotherm almost equally well – yet they yield markedly different mixed-gas predictions. This sensitivity undermines confidence in mixture forecasts, as one study showed that using different pure-gas fit sets led to only _qualitative_ agreement with some mixed-CO₂/CH₄ data.

Furthermore, DMS inherently assumes a fixed population of Langmuir sites (microvoids) and ignores polymer relaxation or swelling induced by sorbate. In reality, penetrant-induced swelling (plasticization) can increase free volume and sorption capacity, especially for highly condensable gases at high activity – an effect the basic DMS model cannot capture. In binary sorption, this means any **synergistic** increase in sorption due to a second gas (or due to polymer dilation) is unaccounted for in the standard model. Conversely, the model also cannot describe sorption isotherm shapes that deviate from concave Langmuir-type behavior (e.g. sigmoidal isotherms caused by specific polymer-penetrant interactions or clustering). For example, glassy polymers absorbing polar molecules like alcohols can show sigmoidal sorption isotherms, which the simple dual-mode framework fails to represent. Finally, accurately measuring mixed-gas sorption itself is challenging, and discrepancies between idealized model assumptions (e.g. truly zero downstream pressure, no competitive diffusion limitations) and experimental realities can add to prediction error. Despite these challenges, the dual-mode sorption model – with appropriate care – remains a valuable first approximation for interpreting sorption in glassy polymers, and it highlights the key role of competitive site occupancy in mixed-gas environments. Researchers often turn to more elaborate models (e.g. non-equilibrium lattice fluid or thermodynamic theories) when the dual-mode model’s assumptions break down, but such models come at the cost of additional parameters or complexity. In summary, dual-mode sorption provides a useful baseline for pure- and mixed-gas sorption in glassy polymers, but one must be mindful of its empirical nature and the above limitations when predicting multicomponent sorption behavior in practice.

**Sources:** Dual-mode sorption model and mixed-gas extension; competitive sorption effects in mixtures; model equations and parameters; limitations of DMS model; parameter sensitivity and predictivity issues.

**Understanding sorption models**

I’m gathering info on pure and mixed-gas sorption in glassy polymers, focusing on the dual-mode sorption model and addressing challenges in modeling predictions.

\[search action\]

**Assessing open access**

OK, let me see. The result from MDPI on mixed-gas sorption in glassy polymers is probably open access.

\[browse action\]

\[browse action\]

**Examining sections**

I’m pulling together insights from the Dual Mode Sorption Model and its extended analysis within the MDPI paper.

\[browse action\]

**Clarifying key properties**

I’m examining the lab report to understand Henry's law constant, Langmuir affinity constant, and Langmuir capacity constant, which describe gas sorption.

\[browse action\]

**Piecing together**

I'm working through the possibility that "Equation (3)" and "Equation (4)" might be images or formula displays and considering consulting the PDF or searching for earlier references to understand them better.

\[browse action\]

**Confirming title relevance**

I’m verifying if the document from Freeman and Koros (1980) specifically matches the reference "Model for sorption of mixed gases in glassy polymers" by checking its abstract and title.

\[search action\]

**Examining article connections**

I noticed that the result from "pmc.ncbi.nlm.nih.gov" is likely the same MDPI article, and I'm curious to see if it presents the text or a simplified version.

\[browse action\]

\[browse action\]

\[browse action\]

**Gathering info**

I'm gathering information about pure and mixed gas sorption in glassy polymers. This is a critical step in understanding the material's behavior under different gas environments.

\[search action\]

**Comparing gas sorption**

I noticed that carbon dioxide's presence in a mixture affects methane sorption in polymer, reducing methane's sorption compared to a pure gas scenario.

\[browse action\]

**Linking concepts**

I’m thinking through how the term "extended dual mode equation mixture" might relate to dual mode superconducting cable equations. This shows potential for connecting concepts across fields.

\[search action\]

**Requesting review details**

I'm noting the request to review "Theories of Sorption and Transport in Polymer Membrane."

\[browse action\]

**Checking PDF link**

I’m curious about Result \[2\] and exploring the snippet, which involves extending a dual-mode sorption model to a binary mixture and expressing concentration of A in terms of partial pressures and constants.

\[search action\]

**Predicting sorption behavior**

The dual-mode model anticipates that adding a second gas reduces sorption at a constant partial pressure, unlike pure gas scenarios.

\[browse action\]

**Fetching a PDF**

I'm gathering info on a nature.com PDF, possibly an older Yampol'ski article, but thinking about alternative sources like the University of Texas at Austin repository for a text snippet.

\[browse action\]

**Evaluating repository options**

I’ve discovered that the UT repository result might not display correctly. Searching within it is uncertain. Alternatively, result \[3\] contains a snippet describing the dual mode mixture sorption model, explaining competitive effects and its adequacy in modeling experimental data.

\[search action\]

**Refining the focus**

I'm adjusting my approach to ensure the search targets the right context.

\[search action\]

**Understanding sorption models**

I’m detailing the dual-mode sorption model for pure gases: Henry dissolving and Langmuir microvoid filling. For mixed gases, competitive sorption and non-idealities are emphasized.

\[search action\]

**Exploring hero.epa.gov**

Thinking through hero.epa.gov's potential abstract or content.

\[browse action\]

**Refining the model**

I'm gathering insights on pure and mixed gas sorption, considering factors like competition, plasticization, and parameter stability. Incorporating mixed-gas data for more accurate predictions.

\[browse action\]

**Addressing dual sorption model challenges**

Parameter coupling and non-uniqueness impact predictions. Absence of polymer condition changes might lead to oversights.

\[browse action\]

**Quantifying deviation**

I’m quantifying nonideal CO2-CH4 solubility in 6FDA-mPDA, showing mixed gas effects on solubility. Competitive sorption isn’t the sole cause of permeability selectivity decline.

\[browse action\]

**Incorporating references**

I'm gathering citations, focusing on open-source materials like MDPI 2019 and Vopicka 2014, to support the discussion on differences and challenges in the dual-mode model for gas sorption in polymeric membranes.

\[browse action\]

**Proposing sections**

I'm organizing sections on pure vs mixed-gas sorption, exploring the dual-mode sorption model, and detailing challenges like parameter uncertainty and mixed gas prediction.

**Organizing content**

I’m focusing on the dual-mode sorption model, detailing pure and mixed-gas sorption behaviors, and highlighting key differences and challenges with bullet points and structured sections for clarity.

\[search action\]

**Choosing citations**

OK, let me see. I’m leaning on \[3\], \[20\], \[22\], and \[21\] for their clear insights into dual-mode sorption and gas mixtures.

**Mapping out parameters**

I’m piecing together parameter names to avoid confusion. Sticking with (k), (C\_H'), and (b) for clarity. The equations for pure gas and mixture are detailed and clearly formatted.

**Addressing challenges**

I'm tackling multi-solution parameter fits, polymer plasticization, and temporal effects. These intricacies make predicting pure and mixed gas sorption tricky.

**Piecing together**

I’m structuring sections on sorption differences, the dual-mode model for pure gases, its extension to mixed gases, and the associated challenges. This progression seems logical and effective.

##### Pass Rate on Expert-Level Tasks by Estimated Economic Value

0510152025Pass RateLowMediumHighVery HighEstimated Economic Value

##### Pass Rate on Expert-Level Tasks by Estimated Hours

0510152025Pass Rate1-3 hour(s)4-6 hours7-9 hours10+ hoursEstimated Hours

Estimated economic value of task is more correlated with pass rate than # of hours it would take a human – the things that models find difficult are different to what humans find time-consuming.

## Limitations

Deep research unlocks significant new capabilities, but it’s still early and has limitations. It can sometimes hallucinate facts in responses or make incorrect inferences, though at a notably lower rate than existing ChatGPT models, according to internal evaluations. It may struggle with distinguishing authoritative information from rumors, and currently shows weakness in confidence calibration, often failing to convey uncertainty accurately. At launch, there may be minor formatting errors in reports and citations, and tasks may take longer to kick off. We expect all these issues to quickly improve with more usage and time.

</details>

<details>
<summary>introducing-perplexity-deep-research</summary>

Written by

Perplexity Team

Published on

Feb 14, 2025

# Introducing Perplexity Deep Research

**Today we’re launching Deep Research** to save you hours of time by conducting in-depth research and analysis on your behalf. When you ask a Deep Research question, Perplexity performs dozens of searches, reads hundreds of sources, and reasons through the material to autonomously deliver a comprehensive report. It excels at a range of expert-level tasks—from finance and marketing to product research—and attains high benchmarks on Humanity’s Last Exam.

We believe everyone should have access to powerful research tools. That’s why we’re making Deep Research free for all. Pro subscribers get unlimited Deep Research queries, while non-subscribers will have access to a limited number of answers per day. Deep Research is available on Web starting today and will soon be rolling out to iOS, Android, and Mac. (Be sure update your apps to the latest version.)

To give it a try, go to [perplexity.ai and select “Deep Research”](https://www.perplexity.ai/?model_id=deep_research) from the mode selector in the search box before submitting your query.

### How It Works

Perplexity already excels at answering questions. Deep Research takes question answering to the next level by spending 2-4 minutes doing the work it would take a human expert many hours to perform. Here’s how it works:

- **Research with reasoning** \- Equipped with search and coding capabilities, Perplexity’s Deep Research mode iteratively searches, reads documents, and reasons about what to do next, refining its research plan as it learns more about the subject areas. This is similar to how a human might research a new topic, refining one’s understanding throughout the process.

- **Report writing** \- Once the source materials have been fully evaluated, the agent then synthesizes all the research into a clear and comprehensive report.

- **Export & Share** \- You can then export the final report to a PDF or document, or convert it into a Perplexity Page and share it with colleagues or friends.

### When to Use Deep Research

We built Deep Research to empower everyone to conduct expert-level analysis across a range of complex subject matters. Deep Research excels at creating work artifacts in domains including finance, marketing, and technology, and is equally useful as a personal consultant in areas such as health, product research, and travel planning. Here are a a few examples of how you might use Deep Research on Perplexity.

#### Finance

#### Marketing

#### Technology

#### Current Affairs

#### Health

#### Biography

#### Travel

### Humanity’s Last Exam

Deep Research on Perplexity attains a 21.1% accuracy score on Humanity’s Last Exam, significantly higher than Gemini Thinking, o3-mini, o1, DeepSeek-R1, and many other leading models. [Humanity’s Last Exam⁠](https://lastexam.ai/) is a comprehensive benchmark for AI systems consisting of over 3,000 questions across 100+ subjects ranging from mathematics and science to history and literature.

### SimpleQA

Scoring 93.9% accuracy on the [SimpleQA](https://arxiv.org/html/2411.04368v1) benchmark — a bank of several thousand questions that test for factuality — Perplexity Deep Research far exceeds the performance of leading models.

### Runtime Stats

Deep Research on Perplexity not only attains high scores on industry benchmarks, but it does so while completing most research tasks in under 3 minutes — which we’re working to make even faster in the future.

</details>

<details>
<summary>opinion-people-are-starting-to-talk-like-chatgpt-the-washing</summary>

If you use ChatGPT, Claude, Gemini or another artificial-intelligence-powered chatbot, you’re probably operating under the assumption that you’re both speaking the same language. You input English, it outputs English. Simple, right? Except that’s a misconception: You’ve actually been speaking different languages.

</details>


## Local Files

<details>
<summary>_Users_omar_Documents_ai_repos_course-ai-agents_lessons_12_central_project_l12_notes</summary>

This lesson is the first lesson of part 2 of a course where we're teaching how to build AI agents for production.

The lesson is titled "Central Project: Scope & Design".
Basically, this course has a part 1 where it teaches basic AI agent concepts.
Then, in part 2, the course goes more into practice and, step by step, teaches how to build two agents: a research agent and a writer agent.
The research agent is an actual agent that works with an MCP server and an MCP client, using `fastmcp`.
The writer agent is more like a workflow agent that works with `langgraph`.

So, in the two projects, we see the two main types of agents that are used in production:
- An adaptable and steerbable research agent that works with an MCP server, which can easily diverge from its main workflow to research different topics and adapt.
- A reliable writer agent whose task is better defined but harder, which needs less adaptation but more reliability.

In this lesson, we provide the high-level architecture and reasoning behind the two agents, and we explain the experience that we had while learning about the topic and building the agents.
- When we started working on the course, we weren’t sure about whether to use agents frameworks/libraries (and which ones) or not. The field is evolving very quickly, so frameworks/libraries can rise up and down just as quickly. It’s to early to predict which will be the ones that will last for multiple years and will become the standard of the field. We're not sure that it’s possible to say that there’s a standard in the field right now, it’s too early.
- So, we started reading the documentations of the existing frameworks, trying to understand which ones are good for quick experimentation and demoing, and which ones are good for production. Also, we considered the current adoptions from the AI developers.
- LangGraph is currently doing great in both adoption and production readiness. It has features to save the state of the computations of the agent/workflow, allowing to resuming it at different points. It allows easy monitoring. And other stuff not implemented by other frameworks/libraries. In contrast, LangGraph has a steeper learning curve. LangGraph seems to be very good for workflows.
- So, we decided to give a try to LangGraph and start using it for the research agent and the writing agent.
- Next, we reasoned about the structure of the research agent and the writing agent. Should they both be agents? Or workflows? What are the pros and cons? We made some assumptions but then we preferred starting working on them and trying them, to get a feel of what would work best.
- We started the reasoning agent as mainly a workflow. Then, by trying it, we noticed that it is a rather interactive process where human feedback is useful, and so that it's possible to stop its execution and resume it from a certain point. So, to make the workflow more flexible, we moved some steps of it to tools, and, over time, the overall structure became that of a full agent. Now, the research agent is simply an MCP server that provides ~10 MCP tools and 1 MCP prompt that describes an “agentic workflow” with them, that is, a recipe of how to use the tools to do a research. We took this opportunity to create also a simple and generic MCP client for it, to teach how to create both MCP servers and MCP clients. However, the MCP server can be used with any MCP client (e.g. Cursor, Claude Code, etc).
- Then, we looked at the current best libraries for structuring MCP clients and servers, and FastMCP has both a great adoption and a lot of features. For structuring MCP servers and clients, the FastMCP python library is currently the standard of the industry, for this reason we decided to use it for the research agent (so, no more LangGraph here).
- Meanwhile, we also worked on the "writing agent", but implementing all of it as a workflow (with LangGraph). Here, we noticed that the "process of writing sections, introductions, conclusions, etc by checking if the text is adhering to style guidelines and following the provided article script" is more prone to be exactly a workflow. The process is always the same, there’s not a lot of adaptation involved. The workflow is rather complex and long, but LangGraph allows to keep it well organized.
- It's hard to decide what framework to choose. Each one has its philosophy. It's a field that is evolving a lot and many libraries and frameworks can appear and get quick traction. It's hard to predict which one will win over time. With hindsight, it seems always easy to explain why a particular framework has succeeded instead of another one, but it's hard to predict it in advance. For this reason, we try not to lose ourselves in the details of the frameworks, but rather focus on the concepts and the ideas. Also for this reason, we were in favour of building a project with two agents with the two main types (which are currently better covered by different frameworks): an adaptable and steerable research agent (better served by MCP servers and `fastmcp`) and a reliable writer agent (better served by workflows and `langgraph`). But again, while the choice of the frameworks seems to be a good one today, it's not the only one and we may change our minds in the future. Our advice is to read the main concepts and philosophies of the frameworks and libraries and choose the one that seem to better fit your needs, and experiment with a lot of them. Indeed, also for these two agents, their final design was not obvious in advance. We tried implementing the research agent as workflows at first, only to find out (while testing it) that it needed more steerability. We wanted to teach only a single framework in the course (langgraph, as it's currently the most popular framework for workflows and ready for production), but we ended up using two frameworks for the two agents as a consequence. This shows that there isn't a clear winner yet in terms of the best framework/library for AI agents.
- The next lesson will talk more about the specific frameworks and libraries that have the most traction today and their philosophies, so no need to talk a lot about them in this lesson.
- For the task of the project (building an article about a topic), we considered the "build vs buy" tradeoff. Indeed, there are already several "deep research" tools available. This lesson should acknowledge the strengths of tools from OpenAI, Google, or Perplexity—fast, polished and convenient. However, we found that we needed something more steerable and interactive for the research part: we want to be able to change topics if the agent is researching not useful ones, and give feedback. The "deep research" tools are not very good at this. Moreover, as part of the research, we want to be able to read GitHub repositories and their code, and transcribe YouTube videos, and use some local files, and this is not easy to do with the "deep research" tools. Also for the writing task, we found that we needed something more deterministic and reliable than the "deep research" tools. The "deep research" tools are not very good at this. Moreover, as part of the writing, we want to be able to check if the text is adhering to style guidelines and following the provided article script, and give feedback. The "deep research" tools are not very good at this. So, we decided to build a custom system for the research and writing tasks, which is the main project of this course, which you'll see in the next lessons. We learned this lesson the hard way. “At first, we thought we could bend off-the-shelf tools into our process. But every time we needed guarantees—like enforcing a style rule or piping results into our CI/CD pipeline—they broke. These failures shaped our decision to build a custom system.”
- Then, this lesson should explain the overall design of the two agents, and the reasoning behind it.
- The research agent is organized as a collection of MCP tools in an MCP server, and an MCP prompt that describes its whole agentic workflow, that is, a recipe of how to use the tools to do a research. As input, the research agent expects a research directory, which contains an article guideline file. The article guideline file basically contains a description of what the final lesson should be about, what is the expected audience, the expected length of the article, the expected outline, some good sources for it, etc. Using this file, the goal of the research agent is to do a research about the topic, and to produce a research file that contains the research data. This research file will then used by the writing agent, together with the original article guideline file, to write the final article. Here, it would be great to show a simple Mermaid diagram showing this: that the research agent wants as input the article guideline and produces a research file, and the writing agent wants as input the article guideline and the research file (and some style guidelines) and produces a final article.
- This is how the research agent works, as described in its MCP prompt, which defines its whole agentic workflow. The lesson should show this prompt, and describe each tool and its purpose. It would be great to show also a simple Mermaid diagram showing this workflow.
<research_agent_architecture>
Your job is to execute the workflow below.

All the tools require a research directory as input.
If the user doesn't provide a research directory, you should ask for it before executing any tool.

**Workflow:**

1. Setup:

    1.1. Explain to the user the numbered steps of the workflow. Be concise. Keep them numbered so that the user
    can easily refer to them later.
    
    1.2. Ask the user for the research directory, if not provided. Ask the user if any modification is needed for the
    workflow (e.g. running from a specific step, or adding user feedback to specific steps).

    1.3 Extract the URLs from the ARTICLE_GUIDELINE_FILE with the "extract_guidelines_urls" tool. This tool reads the
    ARTICLE_GUIDELINE_FILE and extracts three groups of references from the guidelines:
    • "github_urls" - all GitHub links;
    • "youtube_videos_urls" - all YouTube video links;
    • "other_urls" - all remaining HTTP/HTTPS links;
    • "local_files" - relative paths to local files mentioned in the guidelines (e.g. "code.py", "src/main.py").
    Only extensions allowed are: ".py", ".ipynb", and ".md".
    The extracted data is saved to the GUIDELINES_FILENAMES_FILE within the NOVA_FOLDER directory.

2. Process the extracted resources in parallel:

    You can run the following sub-steps (2.1 to 2.4) in parallel. In a single turn, you can call all the
    necessary tools for these steps.

    2.1 Local files - run the "process_local_files" tool to read every file path listed under "local_files" in the
    GUIDELINES_FILENAMES_FILE and copy its content into the LOCAL_FILES_FROM_RESEARCH_FOLDER subfolder within
    NOVA_FOLDER, giving each copy an appropriate filename (path separators are replaced with underscores).

    2.2 Other URL links - run the "scrape_and_clean_other_urls" tool to read the `other_urls` list from the
    GUIDELINES_FILENAMES_FILE and scrape/clean them. The tool writes the cleaned markdown files inside the
    URLS_FROM_GUIDELINES_FOLDER subfolder within NOVA_FOLDER.

    2.3 GitHub URLs - run the "process_github_urls" tool to process the `github_urls` list from the
    GUIDELINES_FILENAMES_FILE with gitingest and save a Markdown summary for each URL inside the
    URLS_FROM_GUIDELINES_CODE_FOLDER subfolder within NOVA_FOLDER.

    2.4 YouTube URLs - run the "transcribe_youtube_urls" tool to process the `youtube_videos_urls` list from the
    GUIDELINES_FILENAMES_FILE, transcribe each video, and save the transcript as a Markdown file inside the
    URLS_FROM_GUIDELINES_YOUTUBE_FOLDER subfolder within NOVA_FOLDER.
        Note: Please be aware that video transcription can be a time-consuming process. For reference,
        transcribing a 39-minute video can take approximately 4.5 minutes.

3. Repeat the following research loop for 3 rounds:

    3.1. Run the "generate_next_queries" tool to analyze the ARTICLE_GUIDELINE_FILE, the already-scraped guideline
    URLs, and the existing PERPLEXITY_RESULTS_FILE. The tool identifies knowledge gaps, proposes new web-search
    questions, and writes them - together with a short justification for each - to the NEXT_QUERIES_FILE within
    NOVA_FOLDER.

    3.2. Run the "run_perplexity_research" tool with the new queries. This tool executes the queries with
    Perplexity and appends the results to the PERPLEXITY_RESULTS_FILE within NOVA_FOLDER.

4. Filter Perplexity results by quality:

    4.1 Run the "select_research_sources_to_keep" tool. The tool reads the ARTICLE_GUIDELINE_FILE and the
    PERPLEXITY_RESULTS_FILE, automatically evaluates each source for trustworthiness, authority and relevance,
    writes the comma-separated IDs of the accepted sources to the PERPLEXITY_SOURCES_SELECTED_FILE **and** saves a
    filtered markdown file PERPLEXITY_RESULTS_SELECTED_FILE that contains only the full content blocks of the accepted
    sources. Both files are saved within NOVA_FOLDER.

5. Identify which of the accepted sources deserve a *full* scrape:

    5.1 Run the "select_research_sources_to_scrape" tool. It analyses the PERPLEXITY_RESULTS_SELECTED_FILE together
    with the ARTICLE_GUIDELINE_FILE and the material already scraped from guideline URLs, then chooses up to 5 diverse,
    authoritative sources whose full content will add most value. The chosen URLs are written (one per line) to the
    URLS_TO_SCRAPE_FROM_RESEARCH_FILE within NOVA_FOLDER.

    5.2 Run the "scrape_research_urls" tool. The tool reads the URLs from URLS_TO_SCRAPE_FROM_RESEARCH_FILE and
    scrapes/cleans each URL's full content. The cleaned markdown files are saved to the
    URLS_FROM_RESEARCH_FOLDER subfolder within NOVA_FOLDER with appropriate filenames.

6. Write final research file:

    6.1 Run the "create_research_file" tool. The tool combines all research data including filtered Perplexity results
    from PERPLEXITY_RESULTS_SELECTED_FILE, scraped guideline sources from URLS_FROM_GUIDELINES_FOLDER,
    URLS_FROM_GUIDELINES_CODE_FOLDER, and URLS_FROM_GUIDELINES_YOUTUBE_FOLDER, and full research sources from
    URLS_FROM_RESEARCH_FOLDER into a comprehensive RESEARCH_MD_FILE organized into sections with collapsible blocks
    for easy navigation. The final RESEARCH_MD_FILE is saved in the root of the research directory.

Depending on the results of previous steps, you may want to skip running a tool if not necessary.

**Critical Failure Policy:**

If a tool reports a complete failure, you are required to halt the entire workflow immediately. A complete failure
is defined as processing zero items successfully (e.g., scraped 0/7 URLs, processed 0 files).

If this occurs, your immediate and only action is to:
    1. State the exact tool that failed and quote the output message.
    2. Announce that you are stopping the workflow as per your instructions.
    3. Ask the user for guidance on how to proceed.

**File and Folder Structure:**

After running the complete workflow, the research directory will contain the following structure:

```
research_directory/
├── ARTICLE_GUIDELINE_FILE                           # Input: Article guidelines and requirements
├── NOVA_FOLDER/                                     # Hidden directory containing all research data
│   ├── GUIDELINES_FILENAMES_FILE                    # Extracted URLs and local files from guidelines
│   ├── LOCAL_FILES_FROM_RESEARCH_FOLDER/           # Copied local files referenced in guidelines
│   │   └── [processed_local_files...]
│   ├── URLS_FROM_GUIDELINES_FOLDER/               # Scraped content from other URLs in guidelines
│   │   └── [scraped_web_pages...]
│   ├── URLS_FROM_GUIDELINES_CODE_FOLDER/          # GitHub repository summaries and code analysis
│   │   └── [github_repo_summaries...]
│   ├── URLS_FROM_GUIDELINES_YOUTUBE_FOLDER/       # YouTube video transcripts
│   │   └── [youtube_transcripts...]
│   ├── NEXT_QUERIES_FILE                           # Generated web-search queries with justifications
│   ├── PERPLEXITY_RESULTS_FILE                     # Complete results from all Perplexity research rounds
│   ├── PERPLEXITY_SOURCES_SELECTED_FILE            # Comma-separated IDs of quality sources selected
│   ├── PERPLEXITY_RESULTS_SELECTED_FILE            # Filtered Perplexity results (only selected sources)
│   ├── URLS_TO_SCRAPE_FROM_RESEARCH_FILE          # URLs selected for full content scraping
│   └── URLS_FROM_RESEARCH_FOLDER/                 # Fully scraped content from selected research URLs
│       └── [full_research_sources...]
└── RESEARCH_MD_FILE                                 # Final comprehensive research compilation
```

This organized structure ensures all research artifacts are systematically collected, processed, and made easily
accessible for article writing and future reference.
</research_agent_architecture>
- Now we can describe the writing agent. The writing agent is organized as a LangGraph workflow, which is a collection of nodes and edges that define the flow of the workflow. The workflow is broken into stages that outline, draft, reflect, iteratively edit, globally refine, and finalize the article.
- The main problem with the writing task is that "An LLM draft often reads like it was written by a brilliant but undisciplined junior writer—knowledgeable, but verbose, generic, and prone to hedging. You can always hear the ‘AI sound.’", and that it's very hard to steer the writing style and quality of an LLM. We learned this the hard way: our early drafts looked polished at first glance, but on reread were full of fluff, clichés, and missing citations. These failures forced us to adopt a checklist of manual editorial tricks—SOPs we now treat as automation targets. We'll talk more about what we specifically did for managing this in the next lessons.
- As support for this, we can add a vivid example of “AI slop.” Use the word “delve” as an example: after 2023, its frequency in professional and scientific writing spiked—not because humans suddenly loved it, but because LLMs overproduced it. Explain that this is a real symptom of a feedback loop gone wrong (i.e. an unwanted consequence of the reinforcement learning finetuning process of LLMs).
- Here follows the architecture of the writing agent.
<writing_agent_architecture>
## Brown — LangGraph-based AI Article Writing Agent

Brown is a multi-stage AI writing agent built with LangGraph and LangChain. It takes a structured guideline and research as inputs and produces a publish-ready technical article with title, SEO metadata, and reflection scores. The workflow is broken into stages that outline, draft, reflect, iteratively edit, globally refine, and finalize the article.

### What Brown Produces
- A complete technical article in Markdown with: title, introduction, body sections, conclusion, and optional references
- Iterative stage artifacts for traceability (Stage 1/2/3 Markdown files)
- Title and SEO metadata
- Reflection scores table (per stage, per unit)
- Style guideline materialized for the run
- Optional graph diagram of the workflow

## Workflow Stages and Nodes

Below is a concise walk-through of the main nodes.

### Stage 0 — Context Gathering
1) Parse all inputs (guideline, research, style guideline, examples, evaluation rules, writer profile)

### Stage 1 — Outline and One-shot Draft
2) Plan introduction, sections, conclusion (respect pre-defined sections in guideline)
3) Draft an end-to-end article using outline, research, style, profile
4) Evaluate draft against evaluation rules; apply targeted one-shot edits
5) Parse Markdown into introduction, sections, conclusion, references; render `article_stage_1.md`

### Stage 2 — Iterative Section Editing Loop
6) Score each section against evaluation rules
7) If less than 90% of the section checks are ok, then apply targeted changes guided by reflection results. Iterate this step until at least 90% of the section checks are ok, or until the maximum number of iterations is reached.

### Stage 3 — Global Reflection and Finalization
8) Evaluate entire article and log Stage 3 reflection scores
9) Apply global edits guided by reflection results
10) Produce final title/subtitle
11) Produce SEO title/description (requires title)
12) Save final article, metadata, style guideline, and reflection scores
</writing_agent_architecture>
- Explain that we reached the architecture/workflow above after trying a lot of different approaches, with the goal of finding the best one that allowed for flexibility and steerability in the writing style and quality, which we found to be very hard using LLMs.

</details>
