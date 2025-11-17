# Research

## Research Results

<details>
<summary>What is the relationship between context engineering and fine-tuning in large language models, and how do you decide which to use?</summary>

### Source [1]: https://huggingface.co/papers/2507.13334

Query: What is the relationship between context engineering and fine-tuning in large language models, and how do you decide which to use?

Answer: Context engineering is presented as a systematic discipline focused on optimizing the information provided to large language models (LLMs), moving beyond simple prompt design. It encompasses context retrieval and generation, context processing, and context management, which are foundational to advanced system implementations such as retrieval-augmented generation (RAG), memory systems, and tool-integrated reasoning. The survey identifies a critical gap: although advanced context engineering can enable models to understand complex contexts, LLMs still perform poorly in generating sophisticated, long-form outputs compared to their ability to process context. This suggests that context engineering is typically applied at inference time (when the model is used) to improve output quality and task performance, whereas fine-tuning is a model training process that changes model parameters to adapt to specific tasks or domains. The choice between the two depends on the nature of the problem: context engineering is suitable for tasks where inference-time flexibility and rapid iteration are needed, while fine-tuning is better for persistent, specialized behavior across many uses[1].

-----

-----

-----

### Source [2]: https://arize.com/docs/phoenix/learn/context-engineering/context-engineering-concepts

Query: What is the relationship between context engineering and fine-tuning in large language models, and how do you decide which to use?

Answer: Context engineering is defined as selecting, organizing, and framing the information that an LLM should process for a given task. This includes the data shown to the model, the session state, external tools available, and the format in which prompts and outputs are structured. The practice involves managing these components like software artifacts—testing, versioning, and iterating to improve predictability and reliability. Context engineering is essential when using LLM agents in complex, real-world systems where context drift, bandwidth overload, and tool blindness can cause failures. While not explicitly contrasted with fine-tuning, the focus here is on managing model inputs and session information dynamically, indicating context engineering is ideal for systems needing adaptable, real-time context management rather than static, model-level adaptation[2].

-----

-----

-----

### Source [3]: https://ramp.com/blog/what-is-context-engineering

Query: What is the relationship between context engineering and fine-tuning in large language models, and how do you decide which to use?

Answer: Context engineering involves deliberately crafting the inputs to an AI system, especially prompts, to guide the model’s output toward specific objectives. It considers how wording, background information, tone, and intent affect model behavior. The discipline emerged as practitioners recognized that LLM outputs are highly sensitive to input design, making context engineering a critical skill for maximizing accuracy and relevance. Unlike fine-tuning—which changes the model’s internal weights and requires retraining—context engineering operates at inference time, allowing rapid experimentation and adjustment without altering the model itself. This makes it preferable for scenarios where flexibility, safety, and immediate feedback are required, whereas fine-tuning is suited to embedding domain knowledge or specialized behavior deeply in the model[3].

-----

-----

-----

### Source [4]: https://blog.getzep.com/what-is-context-engineering/

Query: What is the relationship between context engineering and fine-tuning in large language models, and how do you decide which to use?

Answer: Context engineering marks a departure from traditional prompt engineering, emphasizing the assembly of all necessary information, instructions, and tools to enable reliable model performance. Rather than focusing on the clever formulation of individual prompts (prompt-tuning), context engineering builds systems that dynamically provide LLMs with the precise and relevant context they need for each scenario. This shift is driven by the need for robust, scalable solutions in real-world deployments. In contrast, fine-tuning is not discussed directly, but the emphasis on dynamic, context-driven systems implies that context engineering is best used when models must adapt to varied, evolving requirements without retraining, while fine-tuning is suited for static, repeatable tasks[4].

-----

-----

</details>

<details>
<summary>What are the best practices for structuring and formatting different types of context, such as using XML or YAML, to optimize LLM performance?</summary>

### Source [5]: https://milvus.io/ai-quick-reference/what-modifications-might-be-needed-to-the-llms-input-formatting-or-architecture-to-best-take-advantage-of-retrieved-documents-for-example-adding-special-tokens-or-segments-to-separate-context

Query: What are the best practices for structuring and formatting different types of context, such as using XML or YAML, to optimize LLM performance?

Answer: To optimize LLM performance when leveraging different types of context—including structured formats like XML or YAML—input formatting is critical. Best practices include clearly separating context segments using special tokens such as `[DOC]`, `[CONTEXT]`, or `[SEP]` to delineate document boundaries within the input. For example, a query can be formatted as `[QUERY] ... [CONTEXT] [DOC1] ... [DOC2] ...`, making it explicit where each information source begins and ends. This clarity helps the LLM distinguish between the original query and supplementary context. For lengthy documents, strategies such as chunking or using sliding windows prevent truncation losses. Models can also benefit from techniques like hierarchical summarization, where each document is compressed into a vector before full processing. Architecturally, enhancements such as cross-attention mechanisms (e.g., as used in Fusion-in-Decoder models) allow for parallel processing of queries and multiple documents, with their representations merged during decoding. Sparse or blockwise attention can efficiently process long contexts, and adapter layers can be added to transformer architectures to specialize in integrating external context. Adjustments to multi-head attention, such as biasing attention scores toward document tokens, also improve the handling of structured and multi-source inputs.

-----

-----

-----

### Source [7]: https://swimm.io/learn/large-language-models/llm-context-windows-basics-examples-and-prompting-best-practices

Query: What are the best practices for structuring and formatting different types of context, such as using XML or YAML, to optimize LLM performance?

Answer: For optimal LLM performance with large and structured contexts (such as XML, YAML, or multi-section text), several best practices are recommended. Provide clear, specific instructions to minimize ambiguity and ensure the model understands the task. When input is complex or lengthy, break down the task into smaller, manageable parts, and structure the prompt so the model can address each segment sequentially. Segment the context into meaningful units and, where possible, summarize each segment to highlight essential information. Tailor the context window dynamically to the requirements of the query—this is referred to as query-aware contextualization. This approach minimizes irrelevant information and allows the model to focus processing power on the most relevant segments of the input. These practices enable efficient handling of long or complex structured data, reducing noise and improving both output quality and model efficiency.

-----

-----

</details>

<details>
<summary>What are the latest benchmarks and studies on the "lost-in-the-middle" problem and how do different context engineering strategies mitigate it?</summary>

### Source [8]: https://openreview.net/forum?id=5sB6cSblDR

Query: What are the latest benchmarks and studies on the "lost-in-the-middle" problem and how do different context engineering strategies mitigate it?

Answer: This study investigates the "lost-in-the-middle" problem in long-context language models, particularly within multi-hop question answering tasks. Prior research established that models often focus on information at the beginning and end of the context, neglecting content in the middle. This work extends the analysis to cases where multiple necessary pieces of information are distributed throughout the input. The findings show that model performance not only drops as information is placed farther from the context edges but also between multiple pieces of relevant information spread across the context. To address this, the authors experiment with strategies such as: (1) reducing irrelevant content via knowledge graph triple extraction and summarization, which helps concentrate the model's attention on key facts; and (2) employing chain-of-thought prompting to encourage more thorough and explicit reasoning across the input. These interventions show promise in lessening the "lost-in-the-middle" effect by improving the model's ability to utilize central and distributed information more equitably.

-----

-----

### Source [9]: https://www.marktechpost.com/2024/06/27/solving-the-lost-in-the-middle-problem-in-large-language-models-a-breakthrough-in-attention-calibration/

Query: What are the latest benchmarks and studies on the "lost-in-the-middle" problem and how do different context engineering strategies mitigate it?

Answer: This article reports on recent research led by the University of Washington, MIT, Google Cloud AI Research, and Google, highlighting that even advanced large language models exhibit a strong attention bias toward the beginning and end of input sequences. As a result, their accuracy drops when critical information appears in the middle. Traditional mitigation strategies have included re-ranking documents or repositioning the most relevant content to the start or end of the input. However, these approaches often require extra supervision or fine-tuning and do not fundamentally change the model’s ability to process mid-sequence information. The researchers introduce a new calibration mechanism called “found-in-the-middle,” designed to reduce positional bias by making the model attend to context based on relevance rather than sequence location. This technique aims to improve LLMs’ utilization of information regardless of where it appears, representing a notable departure from earlier, location-based strategies.

-----

</details>

<details>
<summary>What are some real-world case studies or examples of using a multi-agent architecture, like the orchestrator-worker pattern, to isolate context?</summary>

### Source [12]: https://www.confluent.io/blog/event-driven-multi-agent-systems/

Query: What are some real-world case studies or examples of using a multi-agent architecture, like the orchestrator-worker pattern, to isolate context?

Answer: The orchestrator-worker pattern is widely used in event-driven multi-agent systems, particularly for isolating context and managing scalability. In this pattern, a central orchestrator assigns tasks to multiple worker agents and oversees execution. The orchestrator does not need to manage direct connections or the lifecycle of worker agents; instead, it uses a keying strategy to distribute tasks, frequently leveraging Kafka partitions. This ensures that related events are processed by the same worker, allowing for context isolation per partition. Worker agents operate as Kafka consumers, benefiting from built-in coordination, dynamic scaling, and fault recovery. When a worker fails, its tasks can be replayed from a log, ensuring continuity. The pattern is especially effective for breaking down large problems, as agents can be organized hierarchically, with each non-leaf agent acting as an orchestrator for its subtree, recursively decomposing work and maintaining context isolation at each layer[1].

-----

-----

-----

### Source [13]: https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns

Query: What are some real-world case studies or examples of using a multi-agent architecture, like the orchestrator-worker pattern, to isolate context?

Answer: Microsoft's Azure Architecture Center identifies the "handoff orchestration" pattern as a real-world multi-agent architecture for context isolation. In this pattern, specialized agents dynamically delegate tasks among themselves based on contextual analysis and task requirements. Each agent evaluates whether it has the necessary expertise or resources to handle a given task. If not, it transfers control and context to another, more suitable agent. This approach is particularly suited for scenarios where the correct agent cannot be predetermined or where requirements emerge only during task execution. This design ensures that context is encapsulated and transferred between agents only as needed, preventing context leakage and allowing each agent to focus on its specialized subdomain. Examples include complex customer support systems, where input is routed through various agents (including human agents) until the most capable entity is found[2].

-----

-----

-----

### Source [14]: https://www.dailydoseofds.com/ai-agents-crash-course-part-12-with-implementation/

Query: What are some real-world case studies or examples of using a multi-agent architecture, like the orchestrator-worker pattern, to isolate context?

Answer: A practical case study involves implementing a multi-agent pipeline, where each agent is responsible for a specific sub-task and receives only the context relevant to its function. Instead of a single agent solving a complex problem end-to-end, the orchestrator ensures agents act in sequence, handing off context and results in a controlled fashion. For instance, in a data processing workflow, one agent may handle database queries, another may format results, and a third may interact with the user. Each agent’s scope is deliberately narrow, which simplifies their prompts and increases reliability. By limiting the context and toolset available to each agent, the architecture prevents unnecessary information sharing and maintains clear boundaries. This compartmentalization is compared to an assembly line, where each station (agent) performs a targeted function without access to the entire workflow’s context, thereby isolating context at each stage[3].

-----

-----

-----

### Source [15]: https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/mixture-of-agents.html

Query: What are some real-world case studies or examples of using a multi-agent architecture, like the orchestrator-worker pattern, to isolate context?

Answer: In the "Mixture of Agents" pattern, as implemented in Microsoft's AutoGen framework, a single orchestrator agent coordinates multiple layers of worker agents, each capable of running different models or configurations. The orchestrator dispatches tasks to worker agents, who are instantiated only as needed, preserving context isolation between them. For example, in a math problem-solving scenario, the orchestrator assigns subtasks to worker agents in a structured hierarchy. Each worker operates independently, receives only the relevant context for its assigned subproblem, and returns results to the orchestrator. This structure allows for dynamic scalability and strict separation of context, as each worker processes only its assigned portion of the task, and the orchestrator manages the flow and aggregation of results. This pattern is extensible to more complex, multi-domain tasks by registering different worker agent types and maintaining context isolation through the orchestrator’s task dispatch logic[4].

-----

-----

</details>

<details>
<summary>How do different memory types (procedural, episodic, semantic) map to the components of an LLM's context, and what are the common architectural patterns for managing them?</summary>

### Source [16]: https://arxiv.org/html/2504.15965v1

Query: How do different memory types (procedural, episodic, semantic) map to the components of an LLM's context, and what are the common architectural patterns for managing them?

Answer: This survey provides a detailed mapping between human memory types (procedural, episodic, semantic) and memory systems in LLMs. It explains that human memory is typically divided into short-term (sensory and working memory) and long-term types (explicit and implicit memory). Explicit memory encompasses episodic (personal events) and semantic (facts and knowledge) memory, while implicit memory covers procedural memory (skills and habits). In LLMs, these map as follows:
- **Procedural memory** parallels the model’s parametric memory—the weights and structure of the neural network where learned procedures and language patterns are encoded and accessed during inference.
- **Episodic memory** is analogous to personal or session-based non-parametric memory—external stores that retain information from specific past interactions or sessions, retrievable via retrieval-augmented mechanisms to maintain continuity across conversations or tasks.
- **Semantic memory** corresponds to system-level knowledge bases or non-parametric memory—external databases or retrieval systems storing factual information, accessible by the LLM during inference.
The paper introduces a three-dimension classification: object (personal/system), form (parametric/non-parametric), and time (short-term/long-term), resulting in eight quadrants. This provides a systematic way to analyze memory architectures in LLM-driven AI. Common architectural patterns include:
- **Parametric memory** (procedural/semantic): knowledge and skills encoded directly in model weights.
- **Non-parametric memory** (episodic/semantic): external memory modules (vector databases, long-term storage) for storing and retrieving interaction or factual data.
- **Short-term memory**: the immediate context window (current prompt/session).
- **Long-term memory**: persistent external stores or knowledge bases that survive across sessions.
The survey also discusses challenges in integrating and synchronizing these memory types, and highlights future directions such as more robust personal memory systems and better integration of parametric and non-parametric memory[1].

-----

-----

-----

### Source [17]: https://www.cognee.ai/blog/deep-dives/model-context-protocol-cognee-llm-memory-made-simple

Query: How do different memory types (procedural, episodic, semantic) map to the components of an LLM's context, and what are the common architectural patterns for managing them?

Answer: This source describes how external memory systems, such as those implemented via the Model Context Protocol (MCP) and Cognee’s pipelines, can extend LLM memory beyond their native context window. It explains that:
- **Procedural memory** in LLMs is managed within model parameters, reflecting the model’s learned ability to generate coherent language and perform tasks.
- **Episodic memory** is implemented through external context repositories (like vector or graph databases), where records of specific user interactions or sessions are stored and retrieved to maintain context across conversations. These systems enable the LLM to recall prior exchanges, thus supporting continuity and personalization.
- **Semantic memory** is supported by integrating LLMs with structured knowledge bases or data repositories, allowing them to retrieve and utilize factual or general knowledge as needed during inference.
The architectural pattern described involves an external memory layer that interfaces with the LLM using open protocols (such as MCP). This layer manages both episodic (personalized, session-based) and semantic (general knowledge) memory by storing, indexing, and retrieving relevant information on demand. The system thus augments the LLM’s built-in procedural memory (in weights) with flexible, scalable access to both episodic and semantic content, facilitating context-aware and knowledge-rich interactions[2].

-----

-----

-----

### Source [18]: https://arxiv.org/html/2504.02441v1

Query: How do different memory types (procedural, episodic, semantic) map to the components of an LLM's context, and what are the common architectural patterns for managing them?

Answer: This source provides a cognitive framework for mapping human memory types to LLM components:
- **Sensory memory** in LLMs is likened to the immediate API request or prompt input—the transient, per-interaction context.
- **Short-term memory** is defined as the context window (tokens/embeddings) actively processed by the model during a session. This is analogous to working memory in humans, crucial for multi-step reasoning and immediate sequence processing.
- **Episodic memory** is represented by mechanisms that allow LLMs to access and recall information from past sessions or user interactions. This typically requires external memory modules or retrieval-augmented architectures that store and fetch session-specific data.
- **Semantic memory** is mapped to the model’s general knowledge—either encoded in model parameters (parametric memory) or accessed via external knowledge stores (non-parametric memory).
- **Procedural memory** corresponds to the learned capabilities embedded in the model’s weights, enabling language skills and task execution.
The paper emphasizes that short-term/working memory is limited by the context window size, while long-term/episodic memory depends on external memory integration. Common architectural patterns include retrieval-augmented generation (RAG), memory-augmented networks, and hybrid systems employing both parametric (in-weights) and non-parametric (external store) approaches to manage these diverse memory types[3].

-----

-----

</details>

<details>
<summary>What are some detailed case studies or architectural blueprints for implementing context engineering in enterprise AI applications, particularly in healthcare or financial services?</summary>

### Source [20]: https://www.arionresearch.com/blog/67uxqj096in5m3qkksco4lktqmwyzw

Query: What are some detailed case studies or architectural blueprints for implementing context engineering in enterprise AI applications, particularly in healthcare or financial services?

Answer: Context engineering in enterprise AI applications is illustrated through use cases like customer support, sales enablement, and knowledge management. In customer support, AI agents leverage context engineering by accessing customer history, product information, previous interactions, and real-time system status to deliver personalized and accurate resolutions. For sales enablement, AI systems brief representatives with account history, recent engagements, and competitive intelligence, ensuring meetings are contextually informed. Knowledge management systems use context engineering to tailor document retrieval and answers based on the user’s role, project, and current objectives.

Autonomous AI agents utilize context engineering for adaptive task planning, drawing from multi-modal context such as documents, images, system states, and user preferences. This allows dynamic workflow modification and continuous learning from environmental and user feedback. The architecture distinguishes between memory-based agents, which retain accumulated context, and stateless assistants, which do not, depending on application needs.

Developer frameworks like LangChain, Semantic Kernel, and LlamaIndex enable sophisticated context routing and management. Retrieval-Augmented Generation (RAG) pipelines now incorporate multiple retrieval strategies, reranking algorithms, and context synthesis techniques. These tools are critical for building robust enterprise AI applications—such as copilots and chatbots—capable of handling complex, context-rich scenarios.

-----

-----

-----

### Source [21]: https://kanerika.com/blogs/context-engineering/

Query: What are some detailed case studies or architectural blueprints for implementing context engineering in enterprise AI applications, particularly in healthcare or financial services?

Answer: This source provides detailed case studies of context engineering in enterprise AI, though not specifically in healthcare or financial services, but the principles are directly relevant. One case involves a global expert consultation leader who automated the identification of subject-matter experts for niche requests. Previously, the process was manual, fragmented across multiple systems, and led to poor matches and high support ticket volumes. Through context engineering, they integrated disparate data sources, automated expert identification, and improved match accuracy and response times.

Another example details a membership services provider with an AI chatbot that initially lacked access to critical context such as member histories, policy documents, and escalation protocols. Context engineering enabled the chatbot to resolve queries contextually, ensure compliance with policies, and improve support quality and efficiency.

The architectural blueprint involves integrating context as a foundational layer—moving away from static prompts to systems that can remember, comply with organizational policies, and adapt to evolving business needs. This includes continuous context integration across systems, supporting smarter decisions, better outcomes, and responsible generative AI deployment.

-----

-----

-----

### Source [22]: https://shellypalmer.com/2025/06/context-engineering-a-framework-for-enterprise-ai-operations/

Query: What are some detailed case studies or architectural blueprints for implementing context engineering in enterprise AI applications, particularly in healthcare or financial services?

Answer: This source offers a step-by-step architectural framework for implementing context engineering in enterprise AI:

- **Phase 1: Context Inventory**—Map all data sources, their owners, update frequencies, and business criticality to create a comprehensive context map.
- **Phase 2: Integration Architecture**—Design infrastructure to access and process context, including API development, data pipelines, and security frameworks.
- **Phase 3: Context Orchestration**—Develop intelligence to determine which context to retrieve for each query using semantic mappings, relevance algorithms, and performance optimization.
- **Phase 4: Continuous Optimization**—Establish ongoing monitoring, feedback, and expansion of context sources for operational excellence.

A well-context-engineered AI system anticipates information needs, maintains institutional memory, applies business-specific logic, respects governance and compliance, learns from interactions, and scales with complexity. Such systems deliver faster decision-making, operational efficiency, improved compliance, and competitive differentiation by offering business-outcome-oriented AI solutions.

-----

-----

-----

### Source [23]: https://www.clearpeople.com/blog/context-engineering-ai-differentiator

Query: What are some detailed case studies or architectural blueprints for implementing context engineering in enterprise AI applications, particularly in healthcare or financial services?

Answer: This source highlights practical approaches and blueprints for embedding context into AI, with relevant use cases in legal, customer service, and e-commerce. For instance, in legal AI applications, embedding domain-specific taxonomies, jurisdiction filters, and document metadata ensures relevant and accurate responses, avoiding irrelevant or outdated references. In customer service, context engineering lets AI access real-time product catalogues, customer histories, and intent classification, greatly improving the accuracy and quality of responses and reducing escalations.

The architecture involves systematic embedding of context—such as user roles and workflows—into every AI interaction. The Atlas Fuse platform exemplifies this by ensuring generative AI has structured and relevant knowledge available within users’ daily tools, enhancing both trustworthiness and utility.

Context engineering is also crucial for agent-to-agent interactions and smooth workflow transitions, ensuring continuity and context preservation across enterprise AI processes.

-----

-----

</details>

<details>
<summary>What are the most effective techniques for context compression in large language models, including summarization, deduplication, and moving working memory to long-term memory?</summary>

### Source [24]: https://aclanthology.org/2024.findings-emnlp.138.pdf

Query: What are the most effective techniques for context compression in large language models, including summarization, deduplication, and moving working memory to long-term memory?

Answer: This source introduces the In-Context Former (IC-Former), a novel context compression model for large language models (LLMs). The IC-Former can compress input context to a quarter of its original length, packaging it as a soft prompt while preserving most of the contextual information. The approach is both lightweight and efficient, using only 9% of the target LLM’s parameter size, and achieves compression speeds 68 to 112 times faster than the baseline, with over 90% retention of baseline performance. The IC-Former enhances interpretability by analyzing how it interacts with the context during compression. The paper also reviews related work on soft prompt compression, where a compact soft prompt is learned to represent the original prompt, aligning model predictions between the original and compressed prompts by minimizing KL divergence. This ensures that compressed representations still enable the LLM to function effectively, supporting summarization and deduplication as core context compression strategies[1].

-----

-----

-----

### Source [27]: https://openreview.net/forum?id=GYk0thSY1M

Query: What are the most effective techniques for context compression in large language models, including summarization, deduplication, and moving working memory to long-term memory?

Answer: This paper proposes Recurrent Context Compression (RCC), which is designed to efficiently extend the context length that large language models can handle. RCC is trained to compress information from past contexts into a recurrent memory state. This enables LLMs to summarize and condense prior context into a compact form that can be recalled later, effectively simulating a form of moving working memory into a longer-term memory buffer. The method supports efficient memory usage by allowing the model to revisit and retrieve compressed context segments as needed, thereby improving the model's ability to handle tasks that require remembering long or complex sequences. The approach directly addresses the challenge of fitting large or repetitive contexts within limited attention windows and can be combined with deduplication to further optimize context usage[4].

-----

-----

</details>

<details>
<summary>How is 'context drift' defined in AI systems, and what are the primary strategies to manage or mitigate conflicting information over time in an LLM's context?</summary>

### Source [28]: https://viso.ai/deep-learning/concept-drift-vs-data-drift/

Query: How is 'context drift' defined in AI systems, and what are the primary strategies to manage or mitigate conflicting information over time in an LLM's context?

Answer: Concept drift is defined as a change in the statistical properties of the target variable or the relationship between inputs and outputs over time, which undermines the assumption of stationary data distributions that most predictive models rely on. This can significantly degrade AI model accuracy because the trained patterns become less representative of new, real-world data. Causes of concept drift include evolving user behaviors, market shifts, or emergent trends in the domain of application. To manage or mitigate concept drift and conflicting information, adaptive learning algorithms and drift detection techniques are recommended. Notable strategies include online learning, where models are continuously updated with new data, and ensemble methods, which blend multiple models to adapt more robustly to changes. Tools like ADWIN (Adaptive Windowing) dynamically adjust to data changes to maintain model accuracy as the context evolves. Failing to incorporate such adaptive mechanisms can quickly render models obsolete and lead to erroneous decision-making[1].

-----

-----

-----

### Source [31]: https://erikjlarson.substack.com/p/context-drift-and-the-illusion-of

Query: How is 'context drift' defined in AI systems, and what are the primary strategies to manage or mitigate conflicting information over time in an LLM's context?

Answer: This source frames context drift in AI systems as the challenge of maintaining meaningful, persistent context over time, particularly in large language models (LLMs) that do not inherently simulate belief, memory, or intent. Traditional systems attempt to maintain continuity by layering logic around the model, which can become brittle and fail to handle evolving or conflicting context robustly. The AHI (Augmented Human Intelligence) approach treats context as a system-level object—separate from the model and explicitly defined, tracked, and managed by the user or system, rather than relying on implicit context within prompts or short-term memory. In this architecture, rules and context are first-class objects that can be adapted and updated as needed, which helps manage conflicting information and reduces context drift over time. This approach supports deeper integration and more reliable adaptation to changing circumstances by making context management explicit and external to the core model[4].

-----

-----

-----

### Source [32]: https://www.evidentlyai.com/ml-in-production/concept-drift

Query: How is 'context drift' defined in AI systems, and what are the primary strategies to manage or mitigate conflicting information over time in an LLM's context?

Answer: Concept drift is defined as a change in the relationship between input data and the model’s target variable, reflecting an evolution in the underlying problem the model addresses. It can be detected through shifts in feature distributions or changes in the distribution of target values (label drift). The source distinguishes target drift (change in the function mapping inputs to outputs) from label drift (change in frequency of observed outcomes), noting that both can impact model reliability. Effective management of concept drift includes monitoring feature and prediction distributions, frequent model retraining, and implementing drift detection mechanisms to trigger retraining or adaptation when significant changes are observed. This ensures that the model remains relevant as the context and relationships in the data evolve over time[5].

-----

</details>

<details>
<summary>How does context engineering integrate with the broader roles of software engineering, data engineering, and MLOps in production AI systems?</summary>

### Source [33]: https://www.charterglobal.com/context-engineering/

Query: How does context engineering integrate with the broader roles of software engineering, data engineering, and MLOps in production AI systems?

Answer: Context engineering is defined as the strategic design and structuring of the environment, input data, and interaction flows that influence how an AI system interprets and responds to information. Unlike traditional software engineering, which relies on hard-coded logic, context engineering is crucial for probabilistic AI models such as large language models (LLMs), which are highly sensitive to the context in which tasks are presented. This discipline involves managing user metadata, task instructions, data schemas, user intent, role-based behaviors, and environmental cues to ensure AI outcomes are relevant, trustworthy, and aligned with business goals. Context engineering is emerging as a foundational discipline alongside software engineering, data engineering, and MLOps, particularly as generative AI adoption accelerates in production environments. It extends beyond prompt engineering by addressing the limitations of static prompts and ensuring robust, scalable, and reliable AI applications through careful management of context at all layers of interaction[1].

-----

-----

-----

### Source [34]: https://www.philschmid.de/context-engineering

Query: How does context engineering integrate with the broader roles of software engineering, data engineering, and MLOps in production AI systems?

Answer: Context engineering is described as the discipline of designing and building dynamic systems that provide the right information and tools, in the right format, at the right time, to enable LLMs to accomplish tasks effectively. Unlike prompt engineering—which is limited to crafting static input strings—context engineering is about orchestrating dynamic, system-level processes that prepare and format all information required by the AI. This may include real-time retrieval of data such as calendar entries, emails, or web search results. The role of context engineering is inherently cross-functional, requiring an understanding of both technical and business domains. It integrates with software engineering by informing system design, with data engineering by structuring and delivering relevant data, and with MLOps by ensuring operationalization and monitoring of context flows in production AI systems. The ultimate goal is to minimize missing or irrelevant information and maximize AI task performance by engineering context as a first-class system component[2].

-----

-----

-----

### Source [35]: https://www.llamaindex.ai/blog/context-engineering-what-it-is-and-techniques-to-consider

Query: How does context engineering integrate with the broader roles of software engineering, data engineering, and MLOps in production AI systems?

Answer: Context engineering is portrayed as the "delicate art and science" of filling the LLM's context window with the most relevant information for a given task. While prompt engineering is about crafting instructions, context engineering focuses on curating the entirety of what goes into the context window, which may include data from multiple sources, retrieval mechanisms, and dynamically assembled knowledge. This approach is closely related to, but broader than, retrieval-augmented generation (RAG), as it encompasses not only document retrieval but also the orchestration and curation of all contextual signals within the system's operational constraints (such as context window limits). In practical terms, context engineering operates at the intersection of software engineering (system design and orchestration), data engineering (retrieval, transformation, and delivery of structured data), and MLOps (managing, monitoring, and updating context pipelines in production). This integration ensures that AI agents in production systems receive precisely the information they need for reliable, high-quality decision-making[3].

-----

-----

-----

### Source [36]: https://addyo.substack.com/p/context-engineering-bringing-engineering

Query: How does context engineering integrate with the broader roles of software engineering, data engineering, and MLOps in production AI systems?

Answer: Context engineering is about dynamically providing an AI system—particularly LLMs—with all the information and tools needed to successfully complete a task. This includes instructions, data, examples, tools, and task history, all orchestrated and packaged into the model’s context at runtime. The analogy used is to treat the LLM’s context window like a CPU’s RAM, where the role of the context engineer is similar to an operating system, responsible for loading the right "code and data" for each task. Inputs for context can come from a variety of sources, such as user queries, system instructions, retrieved documents, tool outputs, and summaries of previous interactions. Context engineering thus requires close integration with software engineering (for system logic and architecture), data engineering (for reliable data pipelines and retrieval), and MLOps (for monitoring, scaling, and continuous improvement of context delivery in production environments). The discipline brings engineering rigor to the process of assembling AI inputs, transforming context management from a manual art into a systematic, cross-functional engineering practice[4].

-----

-----

### Source [58]: https://www.charterglobal.com/context-engineering/

Query: How does context engineering integrate with the broader roles of software engineering, data engineering, and MLOps in production AI systems?

Answer: Context engineering is defined as the strategic design and structuring of the environment, input data, and interaction flows that shape how an AI system interprets and responds to information. Unlike traditional software engineering, which relies on deterministic logic, AI systems—especially those using large language models (LLMs)—are highly sensitive to the context in which tasks are presented. Context engineering manages aspects such as user metadata, task instructions, data schemas, user intent, role-based behaviors, and environmental cues. This ensures that AI-generated responses are relevant, trustworthy, and aligned with organizational objectives. As generative AI adoption increases, context engineering is becoming foundational for AI reliability, safety, and scalability. It is now considered essential for developers, AI product managers, data scientists, and business leaders building robust AI applications. While prompt engineering focuses on crafting effective inputs for generative AI, it is limited for production systems; context engineering expands this by addressing the broader system-level integration required in real-world AI deployments[1].

-----

-----

### Source [59]: https://www.philschmid.de/context-engineering

Query: How does context engineering integrate with the broader roles of software engineering, data engineering, and MLOps in production AI systems?

Answer: Context engineering is described as the discipline of designing and building dynamic systems that supply the right information and tools, in the right format, at the right time, enabling LLMs to accomplish their tasks effectively. It is not limited to the creation of prompt templates but involves the development of a system that dynamically generates context before the main LLM invocation. This context may include data such as calendar entries, emails, or web search results, tailored to the current task. The approach emphasizes the importance of providing concise, well-formatted information and tool schemas, rather than overwhelming the model with raw data. The engineering of context is framed as a cross-functional challenge that requires understanding the business use case, defining desired outputs, and structuring all necessary information for the LLM to perform the required function. This integration is central to modern software engineering, data engineering, and MLOps practices in production AI systems, as it ensures that AI agents are reliable and aligned with business objectives[2].

-----

-----

### Source [60]: https://www.llamaindex.ai/blog/context-engineering-what-it-is-and-techniques-to-consider

Query: How does context engineering integrate with the broader roles of software engineering, data engineering, and MLOps in production AI systems?

Answer: The article explains that context engineering is about providing AI agents with the relevant context needed to perform tasks effectively. Unlike prompt engineering, which focuses on crafting single task instructions, context engineering is concerned with carefully curating the contents of the LLM's context window using information drawn from multiple sources. This process includes, but is not limited to, retrieval-augmented generation (RAG) techniques. Context engineering encompasses the orchestration of dynamic, relevant information sources—such as databases, documentation, and recent interactions—while considering limitations like the model’s context window size. The discipline is crucial for building reliable, effective AI systems in production, as it addresses the broader system requirements for context assembly and integration, going beyond prompt engineering’s narrower focus[3].

-----

-----

### Source [61]: https://addyo.substack.com/p/context-engineering-bringing-engineering

Query: How does context engineering integrate with the broader roles of software engineering, data engineering, and MLOps in production AI systems?

Answer: Context engineering is described as the process of dynamically providing an AI model—such as an LLM—with all the information and tools needed to complete a task. This includes instructions, data, examples, tools, and historical context, all packaged into the model’s input context at runtime. The article presents a mental model likening an LLM to a CPU and its context window to RAM, with the context engineer acting as an operating system: responsible for loading working memory with the appropriate code and data for the task at hand. In practice, this assembled context may be sourced from user queries, system instructions, retrieved knowledge from databases or documentation, outputs from external tools, and summaries of previous interactions. Context engineering orchestrates these components into a coherent prompt for the model. This approach integrates with software engineering (system design and orchestration), data engineering (retrieval, formatting, and transformation of relevant data), and MLOps (managing context pipelines, quality assurance, and deployment processes) to ensure robust, reliable AI deployments in production environments[4].

-----

</details>

<details>
<summary>What are the best practices for managing and selecting from a large number of tools (100+) in an AI agent to avoid 'tool confusion' and optimize performance?</summary>

### Source [37]: https://shelf.io/blog/ai-agent-deployment/

Query: What are the best practices for managing and selecting from a large number of tools (100+) in an AI agent to avoid 'tool confusion' and optimize performance?

Answer: This source emphasizes the importance of a comprehensive and strategic approach when deploying AI agents, especially in environments with many tools. Key best practices include:

- Strategic Planning with Clear Objectives: Align the integration and use of AI tools with broader business goals and operational needs to avoid unnecessary complexity or redundancy.
- Flexibility and Scalability: Use a modular design for your AI agent system, allowing for easy adaptation and growth as organizational requirements change. This helps manage the addition or removal of tools without confusing the agent.
- Seamless Integration: Assess and prepare existing systems before integrating new tools to ensure that the AI agent enhances workflows rather than disrupts them. Technologies like vector search can help efficiently process large toolsets and data.
- Continuous Improvement: Employ robust monitoring and feedback mechanisms to continually optimize agent performance and adapt tool selection as needs evolve.
- Ethical Considerations: Implement responsible AI practices to build trust and mitigate risks, particularly as the number and diversity of tools increase.

These principles collectively help organizations avoid 'tool confusion' and optimize performance when managing a large number of tools within AI agents[1].

-----

-----

-----

### Source [38]: https://support.talkdesk.com/hc/en-us/articles/39096730105115--Preview-AI-Agent-Platform-Best-Practices

Query: What are the best practices for managing and selecting from a large number of tools (100+) in an AI agent to avoid 'tool confusion' and optimize performance?

Answer: The Talkdesk AI Agent Platform recommends several best practices for managing and optimizing AI agent orchestration when multiple tools are involved:

- Write clear, specific instructions for each tool or agent, avoiding ambiguous or conflicting guidance. This reduces the risk of agents misusing or misunderstanding which tool to use.
- Use structured task decomposition: split complex workflows into simpler, well-defined subtasks, each mapped to the most appropriate tool.
- Avoid reusing the same tool across multiple agents unless absolutely necessary, as this can lead to conflicts and confusion in tool selection logic.
- Provide relevant examples and maintain language consistency in prompts to reinforce correct tool usage.
- Use structured “Skills” and variables to manage agent context, ensuring that only relevant information is shared with each tool or agent, and that variable assignment is handled in a structured way.
- Implement systematic testing and regular monitoring to iteratively refine tool assignment and agent behavior based on real-world usage and outcomes.

These practices are designed to improve predictability, reduce confusion, and ensure that each tool is used optimally by the AI agent[2].

-----

-----

-----

### Source [39]: https://docs.wayfound.ai/agent-management-best-practices

Query: What are the best practices for managing and selecting from a large number of tools (100+) in an AI agent to avoid 'tool confusion' and optimize performance?

Answer: Wayfound suggests that ongoing, systematic management is crucial for effective AI agent operation, particularly when many tools are in use:

- Continuous Monitoring: Use daily alerts to review and address issues related to agent actions and tool performance. This helps quickly identify and resolve sources of confusion or suboptimal tool selection.
- User Feedback Integration: When the agent misapplies a tool or violates a guideline, provide direct feedback to help the system learn and refine its tool selection logic.
- Guideline Refinement: Adjust agent guidelines over time, providing more context or changing priority levels based on observed business impact, to ensure that tool selection remains aligned with organizational goals.
- Knowledge Gap Resolution: Regularly review transcripts where the agent failed to use the correct tool, and update knowledge bases to close common gaps.
- Action Performance Improvement: Monitor failure rates and error patterns with specific tools, optimizing or replacing tools that consistently underperform.

The focus is on continuous iteration and feedback-driven improvement to maintain optimal performance and minimize tool confusion[3].

-----

-----

-----

### Source [40]: https://help.make.com/ai-agent-best-practices

Query: What are the best practices for managing and selecting from a large number of tools (100+) in an AI agent to avoid 'tool confusion' and optimize performance?

Answer: Make.com highlights technical and operational controls for managing large toolsets in AI agents:

- Debugging Tool-Specific Logic: If an agent performs poorly, first review and adjust the logic associated with individual tools to ensure correct operation and avoid confusion.
- Data Access Constraints: Limit agent access to only the data and tools necessary for its tasks, reducing the surface area for confusion and potential security issues.
- Explicit Limitations and Constraints: Use the agent's system prompt to define clear rules about tool use, such as which tools are preferred or off-limits for certain tasks.
- Model Configuration: Set limits on output tokens, execution steps, and thread history to prevent runaway processes or infinite loops caused by misapplied tools.
- Human-in-the-Loop Safeguards: For critical outputs or when ambiguity in tool selection is possible, prompt a human to validate the agent’s choices.
- Prioritize Internal Knowledge: Guide the agent to prefer internal or vetted tools and references before resorting to external or less-reliable options.

These controls collectively reduce the risk of tool confusion and help optimize agent performance when managing a large and diverse toolset[4].

-----

</details>

<details>
<summary>What are the primary trade-offs between context engineering and model fine-tuning in terms of cost, development speed, and adaptability for enterprise AI applications?</summary>

### Source [41]: https://www.tabnine.com/blog/your-ai-doesnt-need-more-training-it-needs-context/

Query: What are the primary trade-offs between context engineering and model fine-tuning in terms of cost, development speed, and adaptability for enterprise AI applications?

Answer: The article emphasizes that context engineering—particularly through Retrieval-Augmented Generation (RAG)—delivers superior results in enterprise settings compared to model fine-tuning. RAG methods, such as vector and semantic retrieval, allow AI systems to integrate up-to-date technical knowledge, structured system context, and reasoning abilities in real time, all without retraining the underlying model. This approach enhances adaptability, as it enables rapid adjustments to changing information and business needs. In contrast, fine-tuning is limited in its ability to encode specific security policies, internal codebases, or unique organizational knowledge. Additionally, RAG-based context engineering is presented as more cost-effective, as it avoids the need for continual retraining pipelines and delivers immediate accuracy and relevance by grounding responses in current, enterprise-specific data. The article argues that for enterprises, especially those needing AI to support complex, evolving workflows, investing in context engineering yields more accurate, adaptable, and secure solutions with less ongoing cost and development overhead.

-----

-----

-----

### Source [42]: https://ai-pro.org/learn-ai/articles/optimal-strategies-for-ai-performance-fine-tune-vs-incontext-learning/

Query: What are the primary trade-offs between context engineering and model fine-tuning in terms of cost, development speed, and adaptability for enterprise AI applications?

Answer: This source compares fine-tuning and in-context learning (a form of context engineering) across several criteria relevant to enterprise use. Fine-tuning is characterized as enhancing task-specific accuracy and robustness, enabling the model to generalize well within a defined area and allowing for long-term improvements as new data arrives. It also provides strong control over model outputs, aligning them closely with business requirements. However, fine-tuning is resource-intensive—requiring significant computational power and high-quality labeled data—and poses risks of overfitting and the need for specialized machine learning expertise. In contrast, in-context learning excels in flexibility and rapid adaptation: it allows the model to respond to new tasks or information on the fly, without retraining. This makes it well-suited for dynamic environments (like customer support) or creative tasks, and it can be implemented more quickly than fine-tuning. The trade-off is that in-context learning may offer less precise control or accuracy for highly specialized, static tasks.

-----

-----

-----

### Source [43]: https://www.tribe.ai/applied-ai/fine-tuning-vs-prompt-engineering

Query: What are the primary trade-offs between context engineering and model fine-tuning in terms of cost, development speed, and adaptability for enterprise AI applications?

Answer: This article frames fine-tuning as akin to hiring a specialist—excellent for deeply embedding domain-specific expertise into a model for critical, high-accuracy tasks. Fine-tuning alters the model’s internal parameters through further training on proprietary or specialized data, achieving strong alignment with enterprise needs but requiring significant time, budget, and technical resources. In contrast, prompt engineering (context engineering) is likened to working with a savvy generalist, enabling rapid adaptation across new tasks by simply modifying prompts or the contextual input provided to the model. This approach is faster to implement and less costly, as it avoids the need for retraining, making it ideal for scenarios where enterprise requirements change frequently or where broad adaptability is more valuable than deep specialization. Thus, the primary trade-off is between the high up-front investment and specificity of fine-tuning versus the speed, flexibility, and lower cost of prompt/context engineering.

-----

-----

-----

### Source [44]: https://nexla.com/ai-infrastructure/prompt-engineering-vs-fine-tuning/

Query: What are the primary trade-offs between context engineering and model fine-tuning in terms of cost, development speed, and adaptability for enterprise AI applications?

Answer: According to this source, fine-tuning is preferred for enterprise AI applications demanding high accuracy and deep domain expertise, as it directly optimizes the model’s parameters for specialized tasks. The accuracy and precision achieved through fine-tuning are heavily dependent on the quality of the training data, and the process requires substantial resources and time. Once a model is fine-tuned, adapting it to new domains or tasks is costly and slow, since each new requirement may necessitate additional retraining. Prompt engineering, by contrast, offers much greater flexibility and adaptability: tasks can be changed simply by altering the prompt, allowing a single base model to serve many purposes with minimal additional cost or delay. The trade-off is that prompt engineering may not reach the same level of accuracy for highly specialized needs, particularly if those needs were not well-represented in the model’s original pre-training data.

-----

-----

</details>

<details>
<summary>What are the most effective strategies for managing 'tool confusion' in AI agents when the number of available tools is large, specifically regarding tool description optimization and dynamic selection?</summary>

### Source [45]: https://productschool.com/blog/artificial-intelligence/ai-agents-product-managers

Query: What are the most effective strategies for managing 'tool confusion' in AI agents when the number of available tools is large, specifically regarding tool description optimization and dynamic selection?

Answer: This source emphasizes that effective use of AI agents, especially when interacting with numerous tools, depends on clear workflows and oversight. Key strategies include:
- Start with clear and specific use cases for each tool, ensuring the agent’s role is well-defined and the toolset is purpose-driven. This helps prevent confusion about which tool to use for which task.
- Human oversight is critical: treat AI agent outputs as drafts and review them before finalization, especially for strategic or customer-facing outputs.
- Integrate agents directly with existing tools and platforms to streamline workflows and minimize redundant tool options, which reduces confusion.
- Begin with a small set of tools and expand only as needed, incrementally introducing more complexity as the agent’s reliability and your confidence grow.
- Regularly measure the agent’s impact and adjust the toolset to maximize value, pruning unnecessary or confusing options.
These approaches indirectly address tool confusion by optimizing descriptions (clear use cases) and dynamically controlling the available tool set (starting small and expanding thoughtfully).

-----

-----

-----

### Source [46]: https://www.hashicorp.com/en/blog/before-you-build-agentic-ai-understand-the-confused-deputy-problem

Query: What are the most effective strategies for managing 'tool confusion' in AI agents when the number of available tools is large, specifically regarding tool description optimization and dynamic selection?

Answer: This source discusses the "confused deputy problem" in agentic AI, a security risk where multiple agents (and tools) interact in ways that can lead to unauthorized or unintended actions. Key management strategies relevant to tool confusion include:
- Implementing strict consent and permission boundaries for each tool or agent, ensuring that agents can only access the tools and data explicitly authorized for a given context.
- Minimizing the number of agents and tools in a workflow when possible to reduce the complexity and risk of confusion.
- Carefully designing tool descriptions and permissions so each agent or tool has a clear, non-overlapping scope of action, which reduces ambiguity about their usage.
- Using architectural safeguards (e.g., chaining systems with clear boundaries, RBAC—Role-Based Access Control) to ensure that even in multi-agent environments, each agent’s tool access is tightly scoped and auditable.
While this source focuses on security, these measures also help manage tool confusion by making tool selection and usage more explicit and controlled.

-----

-----

-----

### Source [47]: https://www.uctoday.com/collaboration/ai-agent-confusion-from-digital-assistants-to-autonomous-co-workers/

Query: What are the most effective strategies for managing 'tool confusion' in AI agents when the number of available tools is large, specifically regarding tool description optimization and dynamic selection?

Answer: This source highlights the challenge of defining true AI agent autonomy, noting that many so-called "agents" are sophisticated automation tools rather than genuinely autonomous entities. In the context of tool confusion:
- The lack of clear definitions for agent roles and tool capabilities leads to user confusion, especially as the number of tools grows.
- To address this, the industry is moving toward viewing AI agents as "digital co-workers" with more clearly delineated responsibilities and proactive task management capabilities.
- Optimizing tool descriptions to reflect autonomous, end-to-end task handling—rather than simple automation—helps clarify when and why an agent should use a particular tool.
- The ongoing evolution toward true autonomy in agents implies that dynamic tool selection should be based on context-aware decision-making and negotiation between agents, rather than static assignments.
These insights suggest that reducing tool confusion depends on both better definitions in tool descriptions and more advanced dynamic selection capabilities in the agent architecture.

-----

-----

-----

### Source [49]: https://www.forrester.com/blogs/the-state-of-ai-agents-lots-of-potential-and-confusion/

Query: What are the most effective strategies for managing 'tool confusion' in AI agents when the number of available tools is large, specifically regarding tool description optimization and dynamic selection?

Answer: This source discusses the widespread confusion around GenAI agents and recommends leading with a robust content decision framework rather than reacting ad hoc to tool requests. Key strategies include:
- Developing a content decision framework that governs how and when agents select tools, based on content type, task requirements, and business logic. This framework helps agents choose the most appropriate tool dynamically from a large set.
- Avoiding reactive, on-the-fly tool assignments, which often leads to confusion and inefficiency as the number of available tools grows.
- Ensuring tool descriptions are structured around clear, actionable criteria within the decision framework, so agents can accurately match tasks to tools.
This approach addresses both description optimization (by standardizing how tools are presented to the agent) and dynamic selection (by providing a systematic selection logic).

-----

</details>

<details>
<summary>Can you provide detailed architectural blueprints or case studies for implementing context engineering in the financial services or healthcare sectors?</summary>

### Source [50]: https://66degrees.com/building-a-business-case-for-ai-in-financial-services/

Query: Can you provide detailed architectural blueprints or case studies for implementing context engineering in the financial services or healthcare sectors?

Answer: This case study describes the implementation of AI-driven context engineering within a mid-sized regional bank aiming to scale its operations and improve customer service. The bank partnered with 66degrees, an AI solutions provider, to develop a comprehensive strategy in three domains:

- **Customer Insights and Personalization:** Machine learning algorithms analyzed customer data and behavior to enable personalized product recommendations and targeted marketing.
- **Risk Assessment and Fraud Detection:** Advanced AI models were integrated to enhance credit scoring, detect fraud, and improve risk management.
- **Operational Efficiency:** AI-powered automation tools streamlined back-office operations, reduced manual errors, and accelerated processing times.

The implementation followed a **phased approach**: pilot projects were launched in each focus area, then scaled organization-wide based on successful outcomes. This iterative strategy allowed for continuous learning and minimized disruption. The result was significant improvement in business operations through data-driven decision-making, alignment of AI initiatives with business challenges, and measurable ROI.

Key architectural considerations highlighted included data integration, iterative deployment, and the alignment of AI systems with critical financial sector requirements such as compliance and customer trust.

-----

-----

-----

### Source [51]: https://shellypalmer.com/2025/06/context-engineering-a-framework-for-enterprise-ai-operations/

Query: Can you provide detailed architectural blueprints or case studies for implementing context engineering in the financial services or healthcare sectors?

Answer: This source presents a detailed architectural blueprint for context engineering in the enterprise, specifically referencing financial services:

- **Phase 1: Context Inventory**
  - Map the organization's complete context landscape, cataloging all data sources, ownership, update frequency, and business criticality.
  - Deliverable: A context map for all critical information assets.

- **Phase 2: Integration Architecture**
  - Design technical infrastructure to access/process context sources: APIs, data pipelines, and security frameworks.
  - Deliverable: Technical architecture for dynamic context assembly with governance controls.

- **Phase 3: Context Orchestration**
  - Build an intelligence layer to retrieve and assemble relevant context for queries using semantic mappings, relevance algorithms, and performance optimization.
  - Deliverable: A fully functioning context orchestration system.

- **Phase 4: Continuous Optimization**
  - Establish ongoing processes for monitoring context quality, incorporating user feedback, and expanding data sources.
  - Deliverable: An operational excellence framework for context engineering.

The blueprint emphasizes that context engineering enables AI systems to maintain institutional memory, apply business logic, respect compliance, learn from usage patterns, and scale with business needs. Benefits include faster decisions, reduced costs, improved compliance, and competitive advantage.

-----

-----

-----

### Source [52]: https://www.akira.ai/blog/context-engineering

Query: Can you provide detailed architectural blueprints or case studies for implementing context engineering in the financial services or healthcare sectors?

Answer: This comprehensive guide discusses metrics, implementation, and case studies of context engineering across financial and healthcare sectors:

- **Key Performance Indicators (KPIs):** Response accuracy/relevance, user satisfaction, processing time, and resource usage.
- **Evaluation Metrics:** Use BLEU (for text quality), F1-score (for classification), and A/B testing to optimize context strategies.
- **Case Studies:**
  - A FinTech firm increased financial advice accuracy by 30% using Retrieval-Augmented Generation (RAG).
  - A healthcare AI system reduced diagnostic errors by leveraging patient history context.
- **ROI and Impact:** Success is measured by cost savings, efficiency gains, and user engagement.
- **Emerging Trends:** Adoption of advanced RAG systems for real-time data and multimodal models for richer context, with ongoing research into reinforcement learning for context optimization.

The source stresses the importance of fairness, transparency, and privacy in designing context-aware systems—essential for trust in sensitive sectors like finance and healthcare.

-----

-----

-----

### Source [53]: https://www.finextra.com/blogposting/28834/context-engineering-for-financial-services

Query: Can you provide detailed architectural blueprints or case studies for implementing context engineering in the financial services or healthcare sectors?

Answer: This article frames context engineering as the discipline of shaping the data, metadata, and relationships that feed AI, highlighting its criticality in fintech:

- **Technical Approaches:** Explores the utility of various platforms—technical computing, vector-based systems, time-series databases, graph, and geospatial platforms—for implementing context engineering in financial services.
- **Application Examples:** While not a full architectural blueprint, the source emphasizes the importance of parameterizing data with technical computing tools (e.g., R, Julia) and leveraging advanced data engineering architectures to build contextualized AI systems.
- **Strategic Value:** Context engineering enhances the relevance, accuracy, and affordability of AI systems in financial services by grounding machine learning in domain-specific context.

The article underscores that context engineering is essential for moving beyond generic AI to solutions that are tuned for the specific challenges and regulatory requirements of the financial sector.

-----

-----

</details>

<details>
<summary>What are the latest research findings and benchmarks on the 'lost-in-the-middle' problem, and how do techniques like instruction repetition and context reordering mitigate it?</summary>

### Source [54]: https://www.marktechpost.com/2024/06/27/solving-the-lost-in-the-middle-problem-in-large-language-models-a-breakthrough-in-attention-calibration/

Query: What are the latest research findings and benchmarks on the 'lost-in-the-middle' problem, and how do techniques like instruction repetition and context reordering mitigate it?

Answer: Researchers from the University of Washington, MIT, Google Cloud AI Research, and Google identified that large language models (LLMs) exhibit an inherent attention bias: they preferentially attend to tokens at the beginning and end of input sequences, causing a drop in accuracy when essential information appears in the middle. This "lost-in-the-middle" phenomenon is tied to intrinsic positional attention bias within LLMs. To address it, the team proposed mechanisms that calibrate attention based on relevance rather than position. Their "found-in-the-middle" mechanisms allow the model to attend more uniformly and meaningfully across the context, regardless of position, significantly improving LLM performance on long-context tasks. This solution directly targets positional bias and opens new approaches for enhancing the attention mechanisms in LLMs for better practical application, particularly in user-facing scenarios.

-----

-----

-----

### Source [55]: https://www.unite.ai/why-large-language-models-forget-the-middle-uncovering-ais-hidden-blind-spot/

Query: What are the latest research findings and benchmarks on the 'lost-in-the-middle' problem, and how do techniques like instruction repetition and context reordering mitigate it?

Answer: The "lost-in-the-middle" problem in LLMs arises due to architectural biases, especially the interplay between causal masking and relative positional encoding in Transformers. This bias leads models to focus on the beginning and end of text sequences, overlooking critical information in the middle. The research emphasizes that this issue is structural and cannot be resolved by simply increasing training data. The practical impact is substantial: for tasks like summarization or question answering on long documents, models may miss or misinterpret important middle content, which is especially problematic in domains like law and medicine. To mitigate this, strategies such as breaking long texts into smaller, manageable chunks or designing models that explicitly direct attention to all parts of the text are suggested. The study also stresses the need for rigorous, position-aware testing of LLMs to ensure reliability in handling long, complex inputs.

-----

-----

-----

### Source [56]: https://promptmetheus.com/resources/llm-knowledge-base/lost-in-the-middle-effect

Query: What are the latest research findings and benchmarks on the 'lost-in-the-middle' problem, and how do techniques like instruction repetition and context reordering mitigate it?

Answer: The "Lost-in-the-Middle Effect" refers to the tendency of LLMs to pay less attention to information located in the middle of a prompt, resulting in suboptimal performance for tasks that require processing such information. Addressing this at the development level involves optimizing attention mechanisms and training strategies for more balanced processing across the input. From a prompt engineering perspective, mitigation techniques include strategically placing critical details at the beginning or end of prompts, or repeating key information at the end to reinforce its relevance. These practical prompt design strategies help counteract the model's tendency to overlook middle content during inference.

-----

-----

-----

### Source [57]: https://news.mit.edu/2025/unpacking-large-language-model-bias-0617

Query: What are the latest research findings and benchmarks on the 'lost-in-the-middle' problem, and how do techniques like instruction repetition and context reordering mitigate it?

Answer: MIT researchers discovered that positional bias, where LLMs prefer information at the start and end of sequences, becomes amplified as models grow larger and acquire more attention layers. Experiments varying answer positions in information retrieval tasks revealed a U-shaped pattern in retrieval accuracy: models performed best when the answer was at the beginning, worst in the middle, and slightly better at the end. The study found that using positional encodings to make the model link words more strongly to their neighbors can help mitigate this position bias. However, the effect of such encodings diminishes as model complexity increases. Beyond architecture, bias can also stem from training data, suggesting that both model design and fine-tuning on de-biased data are important for addressing "lost-in-the-middle."

-----

-----

</details>


## Sources Scraped From Research Results

<details>
<summary>As architects and developers design their workload to take full advantage of language model capabilities, AI agent systems become increasingly complex. These systems often exceed the abilities of a single agent that has access to many tools and knowledge sources. Instead, these systems use multi-agent orchestrations to handle complex, collaborative tasks reliably. This guide covers fundamental orchestration patterns for multi-agent architectures and helps you choose the approach that fits your specific requirements.</summary>

As architects and developers design their workload to take full advantage of language model capabilities, AI agent systems become increasingly complex. These systems often exceed the abilities of a single agent that has access to many tools and knowledge sources. Instead, these systems use multi-agent orchestrations to handle complex, collaborative tasks reliably. This guide covers fundamental orchestration patterns for multi-agent architectures and helps you choose the approach that fits your specific requirements.

## Overview

When you use multiple AI agents, you can break down complex problems into specialized units of work or knowledge. You assign each task to dedicated AI agents that have specific capabilities. These approaches mirror strategies found in human teamwork. Using multiple agents provides several advantages compared to monolithic single-agent solutions.

- **Specialization:** Individual agents can focus on a specific domain or capability, which reduces code and prompt complexity.

- **Scalability:** Agents can be added or modified without redesigning the entire system.

- **Maintainability:** Testing and debugging can be focused on individual agents, which reduces the complexity of these tasks.

- **Optimization:** Each agent can use distinct models, task-solving approaches, knowledge, tools, and compute to achieve its outcomes.

The patterns in this guide show proven approaches for orchestrating multiple agents to work together and accomplish an outcome. Each pattern is optimized for different types of coordination requirements. These AI agent orchestration patterns complement and extend traditional [cloud design patterns](https://learn.microsoft.com/en-us/azure/architecture/patterns/) by addressing the unique challenges of coordinating autonomous components in AI-driven workload capabilities.

## Sequential orchestration

The sequential orchestration pattern chains AI agents in a predefined, linear order. Each agent processes the output from the previous agent in the sequence, which creates a pipeline of specialized transformations.

[https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/sequential-pattern.svg](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/sequential-pattern.svg#lightbox)

[The image shows several sections that have arrows and connecting lines. An arrow points from Input to Agent 1. A line connects Agent 1 to a section that reads Model, knowledge, and tools. An arrow points from Agent 1 to Agent 2. A line connects Agent 2 to a section that reads Model, knowledge, and tools. An arrow points from Agent 2 to a box that has ellipses. An arrow points from this box to Agent n. A line connects Agent n to a section that reads Model, knowledge, and tools. An arrow points from Agent n to Result. A section that reads Common state spans the Agent 1 section through the Agent n section.](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/sequential-pattern.svg#lightbox)

The sequential orchestration pattern solves problems that require step-by-step processing, where each stage builds on the previous stage. It suits workflows that have clear dependencies and improve output quality through progressive refinement. This pattern resembles the [Pipes and Filters](https://learn.microsoft.com/en-us/azure/architecture/patterns/pipes-and-filters) cloud design pattern, but it uses AI agents instead of custom-coded processing components. The choice of which agent gets invoked next is deterministically defined as part of the workflow and isn't a choice given to agents in the process.

### When to use sequential orchestration

Consider the sequential orchestration pattern in the following scenarios:

- Multistage processes that have clear linear dependencies and predictable workflow progression

- Data transformation pipelines, where each stage adds specific value that the next stage depends on

- Workflow stages that can't be parallelized

- Progressive refinement requirements, such as _draft, review, polish_ workflows

- Systems where you understand the availability and performance characteristics of every AI agent in the pipeline, and where failures or delays in one AI agent's processing are tolerable for the overall task to be accomplished

### When to avoid sequential orchestration

Avoid this pattern in the following scenarios:

- Stages are [embarrassingly parallel](https://wikipedia.org/wiki/Embarrassingly_parallel). You can parallelize them without compromising quality or creating shared state contention.

- Processes that include only a few stages that a single AI agent can accomplish effectively.

- Early stages might fail or produce low-quality output, and there's no reasonable way to prevent later steps from processing by using accumulated error output.

- AI agents need to collaborate rather than hand off work.

- The workflow requires backtracking or iteration.

- You need dynamic routing based on intermediate results.

### Sequential orchestration example

A law firm's document management software uses sequential agents for contract generation. The intelligent application processes requests through a pipeline of four specialized agents. The sequential and predefined pipeline steps ensure that each agent works with the complete output from the previous stage.

[https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/sequential-pattern-example.svg](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/sequential-pattern-example.svg#lightbox)

[The image shows several sections that have arrows and connecting lines. An arrow points from Document creation requirements to Template selection agent. A line connects the Template section agent to a section that reads Model, template library, and research tools. An arrow points from the Template selection agent to the Clause customization agent. A line connects the Clause customization agent to a section that reads Fine-tuned model. An arrow points from the Clause customization agent to the Regulatory compliance agent. A line connects the Regulatory compliance agent to a section that reads Model, regulatory knowledge. An arrow points from the Regulatory compliance agent to the Risk assessment agent. A line connects the Risk assessment agent to a section that reads Model, liability knowledge, and persistence tools. An arrow points from the Risk assessment agent to a section that reads Proposed document. A section that reads Document state spans the Clause customization agent to the Proposed document section.](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/sequential-pattern-example.svg#lightbox)

1. The _template selection agent_ receives client specifications, like contract type, jurisdiction, and parties involved, and selects the appropriate base template from the firm's library.

2. The _clause customization agent_ takes the selected template and modifies standard clauses based on negotiated business terms, including payment schedules and liability limitations.

3. The _regulatory compliance agent_ reviews the customized contract against applicable laws and industry-specific regulations.

4. The _risk assessment agent_ performs comprehensive analysis of the complete contract. It evaluates liability exposure and dispute resolution mechanisms while providing risk ratings and protective language recommendations.

## Concurrent orchestration

The concurrent orchestration pattern runs multiple AI agents simultaneously on the same task. This approach allows each agent to provide independent analysis or processing from its unique perspective or specialization.

[https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/concurrent-pattern.svg](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/concurrent-pattern.svg#lightbox)

[The image contains three key sections. In the top section, an arrow points from Input to the Initiator and collector agent. An arrow points from the Initiator and collector agent to a section that reads Aggregated results based on combined, compared, and selected results. A line connects the Initiator and collector agent to a line that connects to four sections via arrows. These sections are Agent 1, Agent 2, an unlabeled section that has ellipses, and Agent n. An arrow points from Agent 1 to Intermediate result. A line points from Agent 1 and splits into two flows. The first flow shows a Sub agent 1.1 section and a section that reads Model, knowledge, and tools. The second flow shows a Sub agent 1.2 and a section that reads Model, knowledge and tools. An arrow points from Agent 2 to Intermediate result. A line connects Agent 2 to a section that reads Model, knowledge, and tools. An arrow points from the unlabeled section that has ellipses to Intermediate results. An arrow points from Agent n to Intermediate result. A line connects Agent n to a section that reads Model, knowledge, and tools.](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/concurrent-pattern.svg#lightbox)

This pattern addresses scenarios where you need diverse insights or approaches to the same problem. Instead of sequential processing, all agents work in parallel, which reduces overall run time and provides comprehensive coverage of the problem space. This orchestration pattern resembles the Fan-out/Fan-in cloud design pattern. The results from each agent are often aggregated to return a final result, but that's not required. Each agent can independently produce its own results within the workload, such as invoking tools to accomplish tasks or updating different data stores in parallel.

Agents operate independently and don't hand off results to each other. An agent might invoke extra AI agents by using its own orchestration approach as part of its independent processing. The available agents must know which agents are available for processing. This pattern supports both deterministic calls to all registered agents and dynamic selection of which agents to invoke based on the task requirements.

### When to use concurrent orchestration

Consider the concurrent orchestration pattern in the following scenarios:

- Tasks that you can run in parallel, either by using a fixed set of agents or by dynamically choosing AI agents based on specific task requirements.

- Tasks that benefit from multiple independent perspectives or different specializations, such as technical, business, and creative approaches, that can all contribute to the same problem. This collaboration typically occurs in scenarios that feature the following multi-agent decision-making techniques:

  - Brainstorming

  - Ensemble reasoning

  - Quorum and voting-based decisions
- Time-sensitive scenarios where parallel processing reduces latency.

### When to avoid concurrent orchestration

Avoid this orchestration pattern in the following scenarios:

- Agents need to build on each other's work or require cumulative context in a specific sequence.

- The task requires a specific order of operations or deterministic, reproducible results from running in a defined sequence.

- Resource constraints, such as model quota, make parallel processing inefficient or impossible.

- Agents can't reliably coordinate changes to shared state or external systems while running simultaneously.

- There's no clear conflict resolution strategy to handle contradictory or conflicting results from each agent.

- Result aggregation logic is too complex or lowers the quality of the results.

### Concurrent orchestration example

A financial services firm built an intelligent application that uses concurrent agents that specialize in different types of analysis to evaluate the same stock simultaneously. Each agent contributes insights from its specialized perspective, which provides diverse, time-sensitive input for rapid investment decisions.

[https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/concurrent-pattern-example.svg](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/concurrent-pattern-example.svg#lightbox)

[The image contains three key sections. In the top section, an arrow points from Ticker symbol to the Stock analysis agent. A line connects Model, exchange symbol mapping knowledge to the Stock analysis agent. An arrow points from the Stock analysis agent to a section that reads Decision with supporting evidence based on combined intermediate results. A line connects Stock analysis agent to a line that points to four separate sections. These sections are four separate flows: Fundamental analysis agent, Technical analysis agent, Sentiment analysis agent, and ESG agent. A line connects Model to the Fundamental analysis agent flow. An arrow points from Fundamental analysis agent flow to Intermediate result. A line points from the Fundamental analysis agent flow and splits into two flows: Financials and revenue analysis agent and Competitive analysis agent. A line connects Financials and revenue analysis agent to a section that reads Model, reported financials knowledge. A line connects Competitive analysis agent to a section that reads Model, competitive knowledge. An arrow points from Technical analysis agent to Intermediate result. A line connects Technical analysis agent to a section that reads Fine-tuned model, market APIs. An arrow points from Sentiment analysis agent to Intermediate result. A line connects Sentiment analysis agent to a section that reads Model, social APIs, news APIs. An arrow points from the ESG agent to Intermediate result. A line connects the ESG agent to a section that reads Model, ESG knowledge.](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/concurrent-pattern-example.svg#lightbox)

The system processes stock analysis requests by dispatching the same ticker symbol to four specialized agents that run in parallel.

- The _fundamental analysis agent_ evaluates financial statements, revenue trends, and competitive positioning to assess intrinsic value.

- The _technical analysis agent_ examines price patterns, volume indicators, and momentum signals to identify trading opportunities.

- The _sentiment analysis agent_ processes news articles, social media mentions, and analyst reports to gauge market sentiment and investor confidence.

- The _environmental, social, and governance (ESG) agent_ reviews environmental impact, social responsibility, and governance practice reports to evaluate sustainability risks and opportunities.

These independent results are then combined into a comprehensive investment recommendation, which enables portfolio managers to make informed decisions quickly.

## Group chat orchestration

The group chat orchestration pattern enables multiple agents to solve problems, make decisions, or validate work by participating in a shared conversation thread where they collaborate through discussion. A chat manager coordinates the flow by determining which agents can respond next and by managing different interaction modes, from collaborative brainstorming to structured quality gates.

[https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/group-chat-pattern.svg](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/group-chat-pattern.svg#lightbox)

[The image shows several sections that have arrows and connecting lines. An arrow points from Input to Group chat manager. An arrow starts at Model, goes through Group chat manager, and points to Accumulating chat thread. A section below this line reads New group instructions based on accumulated context. A line connects to a section that reads Human chat participant or observer. An arrow points from Group chat manager to Agent 2. A double-sided arrow connects Agent 1, an unlabeled box that has ellipses, and Agent n. A line connects Agent 1, Agent 2, the unlabeled box, and Agent n. A line connects Agent 1 to Model and knowledge. A line connects Agent 2 to Model and knowledge. A line connects Agent n to Model and knowledge. An arrow points from a section that reads Chat output from agents to Accumulating chat thread. A line connects Accumulating chat thread to Result.](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/group-chat-pattern.svg#lightbox)

This pattern addresses scenarios that are best accomplished through group discussion to reach decisions. These scenarios might include collaborative ideation, structured validation, or quality control processes. The pattern supports various interaction modes, from free-flowing brainstorming to formal review workflows that have fixed roles and approval gates.

This pattern works well for human-in-the-loop scenarios where humans can optionally take on dynamic chat manager responsibilities and guide conversations toward productive outcomes. In this orchestration pattern, agents are typically in a _read-only_ mode. They don't use tools to make changes in running systems.

### When to use group chat orchestration

Consider group chat orchestration when your scenario can be solved through spontaneous or guided collaboration or iterative maker-checker loops. All of these approaches support real-time human oversight or participation. Because all agents and humans in the loop emit output into a single accumulating thread, this pattern provides transparency and auditability.

#### Collaborative scenarios

- Creative brainstorming sessions where agents that have different perspectives and knowledge sources build on each other's contributions to the chat

- Decision-making processes that benefit from debate and consensus-building

- Decision-making scenarios that require iterative refinement through discussion

- Multidisciplinary problems that require cross-functional dialogue

#### Validation and quality control scenarios

- Quality assurance requirements that involve structured review processes and iteration

- Compliance and regulatory validation that requires multiple expert perspectives

- Content creation workflows that require editorial review with a clear separation of concerns between creation and validation

### When to avoid group chat orchestration

Avoid this pattern in the following scenarios:

- Simple task delegation or linear pipeline processing is sufficient.

- Real-time processing requirements make discussion overhead unacceptable.

- Clear hierarchical decision-making or deterministic workflows without discussion are more appropriate.

- The chat manager has no objective way to determine whether the task is complete.

Managing conversation flow and preventing infinite loops require careful attention, especially as more agents make control more difficult to maintain. To maintain effective control, consider limiting group chat orchestration to three or fewer agents.

### Maker-checker loops

The maker-checker loop is a specific type of group chat orchestration where one agent, the _maker_, creates or proposes something. Another agent, the _checker_, provides a critique of the result. This pattern is iterative, with the checker agent pushing the conversation back to the maker agent to make updates and repeat the process. Although the group chat pattern doesn't require agents to _take turns_ chatting, the maker-checker loop requires a formal turn-based sequence that the chat manager drives.

### Group chat orchestration example

A city parks and recreation department uses software that includes group chat orchestration to evaluate new park development proposals. The software reads the draft proposal, and multiple specialist agents debate different community impact perspectives and work toward consensus on the proposal. This process occurs before the proposal opens for community review to help anticipate the feedback that it might receive.

[https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/group-chat-pattern-example.svg](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/group-chat-pattern-example.svg#lightbox)

[The image shows several sections that have arrows and connecting lines. An arrow points from Park development proposal to Group chat manager. A line starts at Model, goes through Group chat manager, and points to Accumulating conversation. A line connects Parks department employee to this line. A section that reads Instructions based on accumulated context and fresh insight is beneath this section. An arrow points from Group chat manager to the Environmental planning agent. A double-sided arrow connects the Community engagement agent and the Parks budget and operations agent. A line connects the Community engagement agent to the Environmental planning agent and the Parks budget and operations agent. A line connects the Community engagement agent to a section that reads Model and civic knowledge. A line connects the Environmental planning agent to a section that reads Model and local environmental knowledge. An arrow connects a section that reads Chat output from civic agents to Accumulating conversation. A line connects Accumulating conversation to Park proposal consensus. A line connects the Parks budget and operations agent to a section that reads Model and city knowledge.](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/group-chat-pattern-example.svg#lightbox)

The system processes park development proposals by initiating a group consultation with specialized municipal agents that engage in the task from multiple civic perspectives.

- The _community engagement agent_ evaluates accessibility requirements, anticipated resident feedback, and usage patterns to ensure equitable community access.

- The _environmental planning agent_ assesses ecological impact, sustainability measures, native vegetation displacement, and compliance with environmental regulations.

- The _budget and operations agent_ analyzes construction costs, ongoing maintenance expenses, staffing requirements, and long-term operational sustainability.

The chat manager facilitates structured debate where agents challenge each other's recommendations and defend their reasoning. A parks department employee participates in the chat thread to add insight and respond to agents' knowledge requests in real time. This process enables the employee to update the original proposal to address identified concerns and better prepare for community feedback.

## Handoff orchestration

The handoff orchestration pattern enables dynamic delegation of tasks between specialized agents. Each agent can assess the task at hand and decide whether to handle it directly or transfer it to a more appropriate agent based on the context and requirements.

[https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/handoff-pattern.svg](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/handoff-pattern.svg#lightbox)

[The image shows five key sections. The Agent 1 section includes input, a model and general knowledge section, and a result. The Agent 2 section includes a result and model and knowledge section. The Agent 3 section includes the model, knowledge, and tools section, a result, and an unlabeled section that connects to a result. The Agent n section includes a model and knowledge section and a result. The Customer support employee section includes a result. Curved arrows flow from agent to agent and to the customer support employee.](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/handoff-pattern.svg#lightbox)

This pattern addresses scenarios where the optimal agent for a task isn't known upfront or where the task requirements become clear only during processing. It enables intelligent routing and ensures that tasks reach the most capable agent. Agents in this pattern don't typically work in parallel. Full control transfers from one agent to another agent.

### When to use handoff orchestration

Consider the agent handoff pattern in the following scenarios:

- Tasks that require specialized knowledge or tools, but where the number of agents needed or their order can't be predetermined

- Scenarios where expertise requirements emerge during processing, resulting in dynamic task routing based on content analysis

- Multiple-domain problems that require different specialists who operate one at a time

- Logical relationships and signals that you can predetermine to indicate when one agent reaches its capability limit and which agent should handle the task next

### When to avoid handoff orchestration

Avoid this pattern in the following scenarios:

- The appropriate agents and their order are always known upfront.

- Task routing is simple and deterministically rule-based, not based on dynamic context window or dynamic interpretation.

- Suboptimal routing decisions might lead to a poor or frustrating user experience.

- Multiple operations should run concurrently to address the task.

- Avoiding an infinite handoff loop or avoiding excessive bouncing between agents is challenging.

### Agent handoff pattern example

A telecommunications customer relationship management (CRM) solution uses handoff agents in its customer support web portal. An initial agent begins helping customers but discovers that it needs specialized expertise during the conversation. The initial agent passes the task to the most appropriate agent to address the customer's concern. Only one agent at a time operates on the original input, and the handoff chain results in a single result.

[https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/handoff-pattern-example.svg](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/handoff-pattern-example.svg#lightbox)

[The image includes five key sections. The Triage support agent section includes a model and general knowledge section, input, and a result. The Technical infrastructure agent section includes a result and a model, infrastructure knowledge, and tools section. The Financial resolution agent section includes a model, billing account knowledge, and billing API access section, and a result. The Account access agent section includes a result and a model and customer knowledge section. The Customer support employee section includes a result. Curved arrows flow from agent to agent and to the Customer support employee.](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/handoff-pattern-example.svg#lightbox)

In this system, the _triage support agent_ interprets the request and tries to handle common problems directly. When it reaches its limits, it hands network problems to a _technical infrastructure agent_, billing disputes to a _financial resolution agent_, and so on. Further handoffs occur within those agents when the current agent recognizes its own capability limits and knows another agent can better support the scenario.

Each agent is capable of completing the conversation if it determines that customer success has been achieved or that no other agent can further benefit the customer. Some agents are also designed to hand off the user experience to a human support agent when the problem is important to solve but no AI agent currently has the capabilities to address it.

One example of a handoff instance is highlighted in the diagram. It begins with the triage agent that hands off the task to the technical infrastructure agent. The technical infrastructure agent then decides to hand off the task to the financial resolution agent, which ultimately redirects the task to customer support.

## Magentic orchestration

The magentic orchestration pattern is designed for open-ended and complex problems that don't have a predetermined plan of approach. Agents in this pattern typically have tools that allow them to make direct changes in external systems. The focus is as much on building and documenting the approach to solve the problem as it is on implementing that approach. The task list is dynamically built and refined as part of the workflow through collaboration between specialized agents and a magentic manager agent. As the context evolves, the magentic manager agent builds a task ledger to develop the approach plan with goals and subgoals, which is eventually finalized, followed, and tracked to complete the desired outcome.

[https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/magentic-pattern.svg](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/magentic-pattern.svg#lightbox)

[The image shows a Manager agent section. It includes the input and a model. An arrow labeled Invoke agents points from the Manager agent to Agent 2. An arrow labeled Evaluate goal loop points to the Task complete section. An arrow labeled Yes points to the Results section, and an arrow labeled No points back to the Manager agent. An arrow points from the Manager agent to the Task and progress ledger section. A line connects the Task and progress ledger section to the Human participant section. A line that has three arrows points to Agent 1, Agent 2, an unlabeled section, and Agent n. A line connects Agent 1 to a section that reads Model and knowledge. A line connects Agent 2 to a section that reads Model, knowledge, and tools. A line connects Agent n to Model and tools. An arrow points from the section that reads Model, knowledge, and tools to External systems and from the Model and tools section to External systems.](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/magentic-pattern.svg#lightbox)

The manager agent communicates directly with specialized agents to gather information as it builds and refines the task ledger. It iterates, backtracks, and delegates as many times as needed to build a complete plan that it can successfully carry out. The manager agent frequently evaluates whether the original request is fully satisfied or stalled. It updates the ledger to adjust the plan.

In some ways, this orchestration pattern is an extension of the [group chat](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns#group-chat-orchestration) pattern. The magentic orchestration pattern focuses on an agent that builds a plan of approach, while other agents use tools to make changes in external systems instead of only using their knowledge stores to reach an outcome.

### When to use magentic orchestration

Consider the magentic pattern in the following scenarios:

- A complex or open-ended use case that has no predetermined solution path.

- A requirement to consider input and feedback from multiple specialized agents to develop a valid solution path.

- A requirement for the AI system to generate a fully developed plan of approach that a human can review before or after implementation.

- Agents equipped with tools that interact with external systems, consume external resources, or can induce changes in running systems. A documented plan that shows how those agents are sequenced can be presented to a user before allowing the agents to follow the tasks.

### When to avoid magentic orchestration

Avoid this pattern in the following scenarios:

- The solution path is developed or should be approached in a deterministic way.

- There's no requirement to produce a ledger.

- The task has low complexity and a simpler pattern can solve it.

- The work is time-sensitive, as the pattern focuses on building and debating viable plans, not optimizing for end results.

- You anticipate frequent stalls or infinite loops that don't have a clear path to resolution.

### Magentic orchestration example

A site reliability engineering (SRE) team built automation that uses magentic orchestration to handle low-risk incident response scenarios. When a service outage occurs within the scope of the automation, the system must dynamically create and implement a remediation plan. It does this without knowing the specific steps needed upfront.

[https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/magentic-pattern-example.svg](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/magentic-pattern-example.svg#lightbox)

[The image shows the SRE automation manager agent section that includes input and a model. An arrow points from the SRE automation manager agent to the Task and progress ledger section. An arrow labeled Invoke knowledge and action agents points to a line that points to the Infrastructure, Diagnostics, Rollback, and Communication agents. An arrow labeled Evaluate goal loop points from the SRE automation manager agent to the Live-site issue resolved section. An arrow labeled Yes points from Live-site issue resolved to Result. The Task and progress ledger section includes a Resolution approach plan, Resolution task statuses, and the Live-site issue resolved section. An arrow labeled No points from the Live-site issue to the SRE automation manager agent. A line starts at the Diagnostic agent, goes through the Model and log and metrics knowledge section, and points to Workload systems. A line starts at the Infrastructure agent, goes through the model, graph knowledge, and CLI tools section, and joins the line that points to Workload systems. A line starts at the Rollback agent, goes through the model, Git access, CLI tools section, and points to Workload systems. A line starts at the Communication agent, goes through the Model and communication API access section, and points to the Human participant section.](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/_images/magentic-pattern-example.svg#lightbox)

When the automation detects a qualifying incident, the _magentic manager agent_ begins by creating an initial task ledger with high-level goals such as restoring service availability and identifying the root cause. The manager agent then consults with specialized agents to gather information and refine the remediation plan.

1. The _diagnostics agent_ analyzes system logs, performance metrics, and error patterns to identify potential causes. It reports findings back to the manager agent.

2. Based on diagnostic results, the manager agent updates the task ledger with specific investigation steps and consults the _infrastructure agent_ to understand current system state and available recovery options.

3. The _communication agent_ provides stakeholder notification capabilities, and the manager agent incorporates communication checkpoints and approval gates into the evolving plan according to the SRE team's escalation procedures.

4. As the scenario becomes clearer, the manager agent might add the _rollback agent_ to the plan if deployment reversion is needed, or escalate to human SRE engineers if the incident exceeds the automation's scope.

Throughout this process, the manager agent continuously refines the task ledger based on new information. It adds, removes, or reorders tasks as the incident evolves. For example, if the diagnostics agent discovers a database connection problem, the manager agent might switch the entire plan from a deployment rollback strategy to a plan that focuses on restoring database connectivity.

The manager agent watches for excessive stalls in restoring service and guards against infinite remediation loops. It maintains a complete audit trail of the evolving plan and the implementation steps, which provides transparency for post-incident review. This transparency ensures that the SRE team can improve both the workload and the automation based on lessons learned.

## Implementation considerations

When you implement any of these agent design patterns, several considerations must be addressed. Reviewing these considerations helps you avoid common pitfalls and ensures that your agent orchestration is robust, secure, and maintainable.

### Single agent, multitool

You can address some problems with a single agent if you give it sufficient access to a single agent if you give it sufficient access to tools and knowledge sources. As the number of knowledge sources and tools increases, it becomes difficult to provide a predictable agent experience. If a single agent can reliably solve your scenario, consider adopting that approach. Decision-making and flow-control overhead often exceed the benefits of breaking the task into multiple agents. However, security boundaries, network line of sight, and other factors can still render a single-agent approach infeasible.

### Deterministic routing

Some patterns require you to route flow between agents deterministically. Others rely on agents to choose their own routes. If your agents are defined in a no-code or low-code environment, you might not control those behaviors. If you define your agents in code by using SDKs like Semantic Kernel, you have more control.

### Context window

AI agents often have limited context windows. This constraint can affect their ability to process complex tasks. When you implement these patterns, decide what context the next agent requires to be effective. In some scenarios, you need the full, raw context gathered so far. In other scenarios, a summarized or truncated version is more appropriate. If your agent can work without accumulated context and only requires a new instruction set, take that approach instead of providing context that doesn't help accomplish the agent's task.

### Reliability

These patterns require properly functioning agents and reliable transitions between them. They often result in classical distributed systems problems such as node failures, network partitions, message loss, and cascading errors. Mitigation strategies should be in place to address these challenges. Agents and their orchestrators should do the following steps.

- Implement timeout and retry mechanisms.

- Include a graceful degradation implementation to handle one or more agents within a pattern faulting.

- Surface errors instead of hiding them, so downstream agents and orchestrator logic can respond appropriately.

- Consider circuit breaker patterns for agent dependencies.

- Design agents to be as isolated as is practical from each other, with single points of failure not shared between agents. For example:

  - Ensure compute isolation between agents.

  - Evaluate how using a single models as a service (MaaS) model or a shared knowledge store can result in rate limiting when agents run concurrently.
- Use checkpoint features available in your SDK to help recover from an interrupted orchestration, such as from a fault or a new code deployment.

### Security

Implementing proper security mechanisms in these design patterns minimizes the risk of exposing your AI system to attacks or data leakage. Securing communication between agents and limiting each agent's access to sensitive data are key security design strategies. Consider the following security measures:

- Implement authentication and use secure networking between agents.

- Consider data privacy implications of agent communications.

- Design audit trails to meet compliance requirements.

- Design agents and their orchestrators to follow the principle of least privilege.

- Consider how to handle the user's identity across agents. Agents must have broad access to knowledge stores to handle requests from all users, but they must not return data that's inaccessible to the user. Security trimming must be implemented in every agent in the pattern.

### Observability and testing

Distributing your AI system across multiple agents requires monitoring and testing each agent individually, as well as the system as a whole, to ensure proper functionality. When you design your observability and testing strategies, consider the following recommendations:

- Instrument all agent operations and handoffs. Troubleshooting distributed systems is a computer science challenge, and orchestrated AI agents are no exception.

- Track performance and resource usage metrics for each agent so that you can establish a baseline, find bottlenecks, and optimize.

- Design testable interfaces for individual agents.

- Implement integration tests for multi-agent workflows.

### Common pitfalls and anti-patterns

Avoid these common mistakes when you implement agent orchestration patterns:

- Creating unnecessary coordination complexity by using a complex pattern when simple sequential or concurrent orchestration would suffice.

- Adding agents that don't provide meaningful specialization.

- Overlooking latency impacts of multiple-hop communication.

- Sharing mutable state between concurrent agents, which can result in transactionally inconsistent data because of assuming synchronous updates across agent boundaries.

- Using deterministic patterns for workflows that are inherently nondeterministic.

- Using nondeterministic patterns for workflows that are inherently deterministic.

- Ignoring resource constraints when you choose concurrent orchestration.

- Consuming excessive model resources because context windows grow as agents accumulate more information and consult their model to make progress on their task.

### Combining orchestration patterns

Applications sometimes require you to combine multiple orchestration patterns to address their requirements. For example, you might use sequential orchestration for the initial data processing stages and then switch to concurrent orchestration for parallelizable analysis tasks. Don't try to make one workflow fit into a single pattern when different stages of your workload have different characteristics and can benefit from each stage using a different pattern.

## Relationship to cloud design patterns

AI agent orchestration patterns extend and complement traditional [cloud design patterns](https://learn.microsoft.com/en-us/azure/architecture/patterns/) by addressing the unique challenges of coordinating intelligent, autonomous components. Cloud design patterns focus on structural and behavioral concerns in distributed systems, but AI agent orchestration patterns specifically address the coordination of components with reasoning capabilities, learning behaviors, and nondeterministic outputs.

## Implementations in Microsoft Semantic Kernel

Many of these patterns rely on a code-based implementation to address the orchestration logic. The Agent Framework within Semantic Kernel provides support for many of the following [Agent Orchestration patterns](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-orchestration/):

- [Sequential orchestration](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-orchestration/sequential)
- [Concurrent orchestration](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-orchestration/concurrent)
- [Group Chat orchestration](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-orchestration/group-chat)
- [Handoff orchestration](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-orchestration/handoff)
- [Magentic orchestration](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/agent-orchestration/magentic)

For hands-on implementation, explore [Semantic Kernel multi-agent orchestration samples](https://github.com/microsoft/semantic-kernel/tree/main/python/samples/getting_started_with_agents) on GitHub that demonstrate these patterns in practice. You can also find many of these patterns in [AutoGen](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/intro.html), such as [Magentic-One](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/magentic-one.html).

## Implementations in Azure AI Foundry Agent Service

You can also use the [Azure AI Foundry Agent Service](https://learn.microsoft.com/en-us/azure/ai-foundry/agents/overview) to chain agents together in relatively simple workflows by using its [connected agents](https://learn.microsoft.com/en-us/azure/ai-foundry/agents/how-to/connected-agents) functionality. The workflows that you implement by using this service are primarily nondeterministic, which limits which patterns can be fully implemented in this no-code environment.

</details>

<details>
<summary>We’ve spent two years teaching everyone about “prompt engineering,” which has been great. But crafting clever questions represents perhaps 5% of what makes enterprise AI successful. Now, there’s a new term being added to the buzzword bingo lexicon: “context engineering.” I want to make fun of it, but I really like it. Context engineering isn’t about the evolution of end user behavior, it’s a nice way to describe the components you need to get the most out of the current crop of LLMs and Reasoning Engines.</summary>

We’ve spent two years teaching everyone about “prompt engineering,” which has been great. But crafting clever questions represents perhaps 5% of what makes enterprise AI successful. Now, there’s a new term being added to the buzzword bingo lexicon: “context engineering.” I want to make fun of it, but I really like it. Context engineering isn’t about the evolution of end user behavior, it’s a nice way to describe the components you need to get the most out of the current crop of LLMs and Reasoning Engines.

## What Is Context Engineering?

Andrej Karpathy, a founding member of OpenAI and former Director of AI at Tesla, says: “+1 for ‘context engineering’ over ‘prompt engineering’… In every industrial-strength LLM app, context engineering is the delicate art and science of filling the context window with just the right information for the next step.”

## AI Systems Thinking

AI systems don’t just store and retrieve; they synthesize and reason across vast, heterogeneous data sources.

Consider a seemingly simple request: “What’s our exposure to the XYZ market given current conditions?”

A traditional system would query predefined reports. A context-engineered AI system would:

1. Pull current positions from trading systems
2. Analyze recent market movements from external feeds
3. Review internal research reports
4. Check compliance limits and risk parameters
5. Consider historical patterns from similar conditions
6. Synthesize insights from news and analyst reports
7. Generate a comprehensive analysis with specific recommendations

The difference between these approaches is the difference between a 20-minute manual analysis and a 20-second comprehensive assessment that considers factors your analysts might miss.

## Context Engineering in Action

One of our financial services clients recently implemented context engineering for their wealth management division. By connecting market data, client portfolios, regulatory requirements, and relationship history, their advisors now receive AI-generated insights that would have required hours of cross-functional meetings to compile. The result? 40% reduction in prep time and significantly more personalized client strategies.

## The Business-Technology Alignment Challenge

The biggest obstacle to effective context engineering isn’t technical, it’s organizational. Business units own the context (data, procedures, expertise) while IT owns the infrastructure. Context engineering requires unprecedented collaboration between these traditionally separate domains.

### Business Units Must:

- Identify which context sources matter for their decisions
- Define quality standards for different types of information
- Specify how different contexts relate and interact
- Determine acceptable latency for different use cases
- Establish governance rules for sensitive information

### Technology Teams Must:

- Build robust integration architectures
- Ensure real-time data synchronization
- Implement sophisticated access controls
- Optimize for cost and performance
- Maintain system reliability and scalability

### Together They Must:

- Map business processes to context requirements
- Design feedback loops for continuous improvement
- Establish metrics for context quality and completeness
- Create governance frameworks for AI decision-making
- Build change management processes for context evolution

Here’s what a context-engineered system architecture might include:

### Internal Context Sources:

- Enterprise data warehouses and lakes
- CRM and ERP systems
- Document management platforms
- Internal knowledge bases and wikis
- Email and communication archives
- Proprietary databases
- Historical transaction data
- Policy and procedure documentation

### External Context Sources:

- Real-time market data feeds
- Regulatory databases
- Industry intelligence platforms
- News and social media monitoring
- Weather and logistics data
- Competitive intelligence systems
- Third-party APIs
- Public records and filings

### Context Processing Layers:

- Data integration and ETL pipelines
- Embedding and vector databases
- Semantic search capabilities
- Entity resolution systems
- Memory management infrastructure
- Privacy and access controls
- Quality assurance mechanisms
- Performance optimization

## A Practical Context Engineering Roadmap

**Phase 1: Context Inventory** – Start by mapping your context landscape. What information do your teams use to make decisions? Where does it live? How current is it? How reliable?

Key deliverable: A comprehensive context map showing all data sources, their owners, update frequencies, and business criticality.

**Phase 2: Integration Architecture** – Design the technical infrastructure to access and process identified context sources. This includes API development, data pipeline construction, and security framework implementation.

Key deliverable: Technical architecture supporting dynamic context assembly with appropriate governance controls.

**Phase 3: Context Orchestration** – Build the intelligence layer that determines which context to retrieve for different queries. This involves creating semantic mappings, relevance algorithms, and performance optimization strategies.

Key deliverable: Functioning context orchestration system that dynamically assembles relevant information.

**Phase 4: Continuous Optimization** – Context engineering isn’t a project, it’s a discipline. Establish processes for monitoring context quality, gathering user feedback, and continuously expanding context sources.

Key deliverable: Operational excellence framework for context engineering.

## The Competitive Imperative

Organizations that master context engineering will have AI systems that truly understand their businesses. A well-context-engineered system won’t just answer questions, it will anticipate information needs before they’re articulated, maintain institutional memory across personnel changes, apply your company-specific logic and business rules, respect governance requirements and compliance frameworks, learn from usage patterns to improve over time, and scale seamlessly with business complexity.

This translates directly to measurable outcomes: faster decision-making, reduced operational costs, improved compliance, and the ability to identify opportunities your competitors miss because their AI lacks context.

Said differently, it’s the difference between using off-the-shelf solutions (even really good 3rd-party solutions) and building a business-outcome-oriented AI solution.

## But Isn’t This Just System Integration?

Skeptics might argue this is just traditional system integration with an AI wrapper. They’re missing the point. Traditional integration moves data. Context engineering creates understanding. It’s the difference between giving someone a library card and giving them a research assistant who’s read every book and knows exactly where to find what you’re looking for.

## Common Pitfalls to Avoid

**1. The Data Dump Fallacy** Simply connecting AI to all available data sources doesn’t create context, it creates noise. Context engineering requires intelligent curation and relevance filtering.

**2. The Silo Trap** Building separate context systems for different departments defeats the purpose. Context engineering should create unified intelligence across the enterprise.

**3. The Static Context Mistake** Business context evolves constantly. Systems must be designed for continuous context updates, not one-time configurations.

**4. The Security Afterthought** Context engineering amplifies both capabilities and risks. Security and governance must be foundational, not bolted on.

## Questions for Your Leadership Team

1. Who owns context engineering in your organization? If the answer is unclear, you have an organizational challenge before you have a technical one.
2. What percentage of your critical business context is AI-accessible today? Most enterprises discover it’s less than 20%.
3. How do you measure context quality? Without metrics, you can’t improve.
4. What’s your context refresh strategy? Static context leads to stale AI.
5. How does context engineering fit your AI governance framework? This isn’t optional in regulated industries.

## The Path Forward

Context engineering represents the maturation of enterprise AI from experimental technology to operational capability. It’s not about teaching employees new ways to interact with AI, it’s about building AI systems that deeply understand your business.

Context engineering leverages existing enterprise capabilities: data management, system integration, governance frameworks. The challenge is orchestrating these capabilities in service of AI rather than traditional applications.

Start by bringing your business and technology leaders together around this question: “What context would transform our AI from a smart assistant into a knowledgeable partner?” The answer will drive your context engineering strategy.

Is “context engineering” another buzzword? Perhaps. But it’s a useful one that captures a critical need: orchestrating your entire information ecosystem to make AI truly intelligent about your business. In an era where competitive advantage comes from decision speed and quality, context isn’t just king… it’s the entire kingdom.

</details>

<details>
<summary>In neuroscience, human memory refers to the brain’s ability to store, retain, and recall information \[ [9](https://arxiv.org/html/2504.15965v1#bib.bib9 ""), [10](https://arxiv.org/html/2504.15965v1#bib.bib10 "")\].</summary>

In neuroscience, human memory refers to the brain’s ability to store, retain, and recall information \[ [9](https://arxiv.org/html/2504.15965v1#bib.bib9 ""), [10](https://arxiv.org/html/2504.15965v1#bib.bib10 "")\].
Human memory serves as the foundation for understanding the world, learning new knowledge, adapting to the environment, and making decisions, allowing us to preserve past experiences, skills, and knowledge, and helping us form our personal identity and behavior patterns \[ [11](https://arxiv.org/html/2504.15965v1#bib.bib11 "")\].
Human memory can be broadly classified into short-term memory and long-term memory based on the duration of new memory formation \[ [12](https://arxiv.org/html/2504.15965v1#bib.bib12 "")\].
Short-term memory refers to the information we temporarily store and process, typically lasting from a few seconds to a few minutes, and includes sensory memory and working memory \[ [11](https://arxiv.org/html/2504.15965v1#bib.bib11 "")\].
Long-term memory refers to the information we can store for extended periods, ranging from minutes to years, and includes declarative explicit memory (such as episodic and semantic memory) and non-declarative implicit memory (such as conditioned reflexes and procedural memory) \[ [11](https://arxiv.org/html/2504.15965v1#bib.bib11 "")\].
Human memory is a complex and dynamic process that relies on different memory systems to process information for various purposes, influencing how we understand and respond to the world.
The different types of human memory and their working mechanisms can greatly inspire us to develop more scientific and reasonable memory-enhanced AI systems \[ [13](https://arxiv.org/html/2504.15965v1#bib.bib13 ""), [14](https://arxiv.org/html/2504.15965v1#bib.bib14 ""), [15](https://arxiv.org/html/2504.15965v1#bib.bib15 ""), [16](https://arxiv.org/html/2504.15965v1#bib.bib16 "")\].

In the era of large language models (LLMs), the most typical memory-enhanced AI system is the LLM-powered autonomous agent system \[ [10](https://arxiv.org/html/2504.15965v1#bib.bib10 "")\].
Large language model (LLM) powered agents are AI systems that can perform complex tasks using natural language, incorporating capabilities like planning, tool use, memory, and multi-step reasoning to enhance interactions and problem-solving \[ [1](https://arxiv.org/html/2504.15965v1#bib.bib1 ""), [2](https://arxiv.org/html/2504.15965v1#bib.bib2 ""), [10](https://arxiv.org/html/2504.15965v1#bib.bib10 "")\].
This memory-enhanced AI system is capable of autonomously decomposing complex tasks, remembering interaction history, and invoking and executing tools, thereby efficiently completing a series of intricate tasks.
In particular, memory, as a key component of the LLM-powered agent, can be defined as the process of acquiring, storing, retaining, and subsequently retrieving information \[ [10](https://arxiv.org/html/2504.15965v1#bib.bib10 "")\].
It enables the large language model to overcome the limitation of LLM’s context window, allowing the agent to recall interaction history and make more accurate and intelligent decisions.
For instance, MemoryBank \[ [17](https://arxiv.org/html/2504.15965v1#bib.bib17 "")\] proposed a long-term memory mechanism to allow LLMs for retrieving relevant memories, continuously evolving through continuous updates, and understanding and adapting to a user’s personality by integrating information from previous interactions.
In addition, many commercial and open-source AI systems have also integrated memory systems to enhance the personalization capabilities of the system, such as OpenAI ChatGPT Memory \[ [18](https://arxiv.org/html/2504.15965v1#bib.bib18 "")\], Apple Personal Context \[ [19](https://arxiv.org/html/2504.15965v1#bib.bib19 "")\], mem0 \[ [20](https://arxiv.org/html/2504.15965v1#bib.bib20 "")\], MemoryScope \[ [21](https://arxiv.org/html/2504.15965v1#bib.bib21 "")\], etc.

Although previous studies and reviews have provided detailed explanations of memory mechanisms, most of the existing work focuses on analyzing and explaining memory from the temporal (time) dimension, specifically in terms of short-term and long-term memory \[ [8](https://arxiv.org/html/2504.15965v1#bib.bib8 ""), [7](https://arxiv.org/html/2504.15965v1#bib.bib7 ""), [17](https://arxiv.org/html/2504.15965v1#bib.bib17 "")\].
We believe that categorizing memory solely based on the time dimension is insufficient, as there are many other aspects (such as object and form) to memory classification in AI systems.
For example, from the object dimension, since AI systems often interact with humans, they need to perceive, store, recall, and use memories related to individual users, thus generating personal memory. Meanwhile, when AI systems perform complex tasks, they generate intermediate results (such as reasoning and planning processes, internet search results, etc.), which form system memory.
In addition, from the form dimension, since AI systems are powered by large language models (LLMs), they can store memories through the parametric memory encoded within the model parameters, as well as through non-parametric memory in the form of external memory documents that are stored and managed outside the model.
Therefore, insights that consider memory from the perspectives of object (personal and system), form (parametric and non-parametric), and time (short-term and long-term) are still lacking in the current era of large language models.
There is still no comprehensive review that systematically analyzes the relationship between memory in LLM-driven AI systems and human memory, and how insights from human memory can be leveraged to build more efficient and powerful memory systems.

To fill this gap, this paper presents a comprehensive review of the memory mechanisms in LLM-driven AI systems.
First, we provide a detailed analysis of the categories of human memory and relate them to the memory systems in AI.
In particular, we explore how human memory types — short-term memory (including sensory memory and working memory) and long-term memory (including explicit memory and implicit memory) — correspond to personal and system memory, parametric and non-parametric memory, and short-term and long-term memory in LLM-driven AI systems.
Next, we systematically organize the existing work related to memory and propose a classification method based on three dimensions (object, form, and time) with eight quadrants.
In the object dimension, memory can be divided into personal memory and system memory; in the form dimension, it can be classified into parametric memory and non-parametric memory; in the time dimension, memory can be categorized into short-term memory and long-term memory.
Finally, based on the classification results from the three dimensions and eight quadrants mentioned above, we analyze some open issues in the memory of current AI systems and outline potential future directions for memory development in the era of large language models.

The main contributions of this paper are summarized as follows:
(1) We systematically and comprehensively define LLM-driven AI systems’ memory and establish corresponding relationships with human memory.
(2) We propose a classification method for memory based on three dimensions (object, form, and time) and eight quadrants, which facilitates a more systematic exploration of memory in the era of large language models.
(3) From the perspective of enhancing personalized capabilities, we analyze and summarize research related to personal memory.
(4) From the perspective of AI system’s ability to perform complex tasks, we analyze and summarize research related to system memory.
(5) We identify the existing issues and challenges in current memory research and point out potential future directions for development.

The remainder of the paper is organized as follows:
In Section 2, we present a detailed description of human memory and AI systems’ memory, comparing their differences and relationships, and introduce the classification method for memory based on three dimensions (object, form, and time) and eight quadrants.
In Section 3, we summarize research related to personal memory, aimed at enhancing the personalized response capabilities of AI systems.
In Section 4, we summarize research related to system memory, aimed at improving AI systems’ ability to perform complex tasks.
In Section 5, we analyze some open issues related to memory and point out potential future directions for development.
Finally, in Section 6, we conclude the survey.

## 2 Overview

The human brain has evolved complex yet efficient memory mechanisms over a long period, enabling it to encode, store, and recall information effectively \[ [9](https://arxiv.org/html/2504.15965v1#bib.bib9 "")\].
Accordingly, in the development of AI systems, we can draw insights from human memory to design effective & efficient memory mechanisms or systems.
In this section, we will first describe in detail the complex memory mechanisms and related memory systems of the human brain from the perspective of memory neuroscience.
Then, we will discuss the memory mechanisms and types specific to LLM-driven AI systems.
Finally, based on the memory features of LLM-driven AI systems, we will systematically review and categorize existing work from different dimensions.

### 2.1 Human Memory

Human memory typically relies on different memory systems to process information for various purposes, such as working memory for temporarily storing and processing information to support ongoing cognitive activities, and episodic memory for recording personal experiences and events for a long time \[ [11](https://arxiv.org/html/2504.15965v1#bib.bib11 "")\].

#### 2.1.1 Short-Term and Long-Term Memory

Based on the time range, human memory can be roughly divided into short-term memory and long-term memory according to the well-known Multi-Store Model (or Atkinson-Shiffrin Memory Model) \[ [22](https://arxiv.org/html/2504.15965v1#bib.bib22 "")\].

##### Short-Term Memory

Short-term memory is a temporary storage system that holds small amounts of information for brief periods, typically ranging from seconds to minutes.
It includes sensory memory, which briefly captures raw sensory information from the environment (like sights or sounds), and working memory, which actively processes and manipulates information to complete tasks such as problem-solving or learning.
Together, these components allow humans to temporarily hold and work with information before either discarding it or transferring it to long-term memory.

- •


Sensory memory: Sensory memory is the brief storage of sensory information we acquire from the external world, including iconic memory (visual), echoic memory (auditory), haptic memory (touch), and other sensory data. It typically lasts only a few milliseconds to a few seconds. Some sensory memories are transferred to working memory, while others are eventually stored in long-term memory (such as episodic memory).

- •


Working memory: Working memory is the system we use to temporarily store and process information. It not only helps us maintain current thoughts but also plays a role in decision-making and problem-solving. For example, when solving a math problem, it allows us to keep track of both the problem and the steps involved in finding the solution.

##### Long-Term Memory

Long-term memory is a storage system that holds information for extended periods, ranging from minutes to a lifetime.
It includes explicit memory, which involves conscious recall of facts and events, and implicit memory, which involves unconscious skills and habits, like riding a bike.
These two types work together to help humans retain knowledge, experiences, and learned abilities over time.

- •


Explicit memory: Explicit memory, also known as declarative memory, refers to memories that we can easily verbalize or declare. It can be further divided into episodic memory and semantic memory. Episodic memory refers to memories related to personal experiences and events, such as what you had for lunch. This type of memory is typically broken down into stages like encoding, storage, and retrieval. Semantic memory, on the other hand, refers to memories related to facts and knowledge, such as knowing that the Earth is round or that the Earth orbits the Sun.

- •


Implicit memory: Implicit memory, also known as non-declarative memory, refers to memories that are difficult to describe in words. It is associated with habits, skills, and procedures, and does not require conscious recall. Procedural memory (or "muscle memory") is a typical form of implicit memory. It refers to memories gained through actions, such as riding a bicycle or playing the piano. The planning and coordination of movements are key components of procedural memory.

Multiple memory systems typically operate simultaneously, storing information in various ways across different brain regions. These memory systems are not completely independent; they interact with each other and, in many cases, depend on one another.
For example, when you hear a new song, the sensory memory in your ears and the brain regions responsible for processing sound will become active, storing the sound of the song for a few seconds. This sound is then transferred to your working memory system.
As you use your working memory and consciously think about the song, your episodic memory will automatically activate, recalling where you heard the song and what you were doing at the time.
As you hear the song in different places and at different times, a new semantic memory gradually forms, linking the melody of the song with its title. So, when you hear the song again, you’ll remember the song’s title, rather than a specific instance from your multiple listening experiences.
When you practice playing the song on the guitar, your procedural memory will remember the finger movements involved in playing the song.

#### 2.1.2 Memory Mechanisms

Memory is the ability to encode, store and recall information.
The three main processes involved in human memory are therefore encoding (the process of acquiring and processing information into a form that can be stored), storage (the retention of encoded information over time in short-term or long-term memory), and retrieval (recall, the process of accessing and bringing stored information back into conscious awareness when needed).

- •


Encoding Memory encoding is the process of changing sensory information into a form that our brain can cope with and store effectively. In particular, there are different types of encoding in terms of how information is processed, such as visual encoding, which involves processing information based on its visual features like color, shape, or texture; acoustic encoding, which focuses on the auditory characteristics of information, such as pitch, tone, or rhythm; and semantic encoding, which is based on the meaning of the information, making it easier to structure and remember. In addition, there are many approaches to make our brain better at encoding memory, such as mnemonics, which involve using acronyms or peg-word systems to aid recall, chunking, where information is broken down into smaller, meaningful units to enhance retention, imagination, which strengthens encoding by linking images to words, and association, where new information is connected to prior knowledge to improve understanding and long-term memory storage.

- •


Storage The storage of memory involves the coordinated activity of multiple brain regions, with key areas including: the prefrontal cortex, which is associated with working memory and decision-making, helping us maintain and process information in the short term; the hippocampus, which helps organize and consolidate information to form new explicit memories (such as episodic memory); the cerebral cortex, which is involved in the storage and retrieval of semantic memory, allowing us to retain facts, concepts, and general knowledge over time; and the cerebellum, which is primarily responsible for procedural memory formed through repetition.

- •


Retrieval Memory retrieval is the ability to access information and get it out of the memory storage. When we recall something, the brain reactivates neural pathways (also called synapses) linked to that memory. The prefrontal cortex helps in bringing memories back to awareness. Similarly, there are different types of memory retrieval, including recognition, where we identify previously encountered information or stimuli, such as recognizing a familiar face or a fact we have learned before; recall, which is the ability to retrieve information from memory without external cues, like remembering a phone number or address from memory; and relearning, a process in which we reacquire previously learned but forgotten information, often at a faster pace than initial learning due to the residual memory traces that still exist.

In addition to the fundamental memory processing stages of encoding, storage, and retrieval, human memory also includes consolidation (the process of stabilizing and strengthening memories to facilitate long-term storage), reconsolidation (the modification or updating of previously stored memories when they are reactivated, allowing them to adapt to new information or contexts), reflection (the active review and evaluation of one’s memories to enhance self-awareness, improve learning strategies, and optimize decision-making), and forgetting (the process by which information becomes inaccessible).

- •


Consolidation Memory consolidation refers to the process of converting short-term memory into long-term memory, allowing information to be stably stored in the brain and reducing the likelihood of forgetting. It primarily involves the hippocampus and strengthens neural connections through synaptic plasticity (strengthening of connections between neurons) and systems consolidation (the gradual transfer and reorganization of memories from the hippocampus to the neocortex for long-term storage).

- •


Reconsolidation Memory reconsolidation refers to the process in which a previously stored memory is reactivated, entering an unstable state and requiring reconsolidation to maintain its storage. This process allows for the modification or updating of existing memories to adapt to new information or contexts, potentially leading to memory enhancement, weakening, or distortion. Once a memory is reactivated, it involves the hippocampus and amygdala and may be influenced by emotions, cognitive biases, or new information, resulting in memory adjustment or reshaping.

- •


Reflection Memory reflection refers to the process in which an individual actively reviews, evaluates, and examines their own memory content and processes to enhance self-awareness, adjust learning strategies, or optimize decision-making. It helps improve metacognitive ability, correct memory biases, facilitate deep learning, and regulate emotions. This process primarily relies on the brain’s metacognitive ability (Metacognition) and involves the prefrontal cortex, which monitors and regulates memory functions.

- •


Forgetting Forgetting is a natural process that occurs when the brain fails to retrieve or retain information, which can result from encoding failure (when information is not properly encoded due to lack of attention or meaningful connection), memory decay (when memories fade over time without reinforcement as neural connections weaken), interference (when similar or new memories compete with or overwrite existing ones), retrieval failure (when information is inaccessible due to missing contextual cues despite being stored), or motivated forgetting (when individuals consciously suppress or unconsciously repress traumatic or distressing memories). However, forgetting is a natural and necessary process that enables our brains to filter out irrelevant and outdated information, allowing us to prioritize what is most important for our current needs.

### 2.2 Memory of LLM-driven AI Systems

Similar to humans, LLM-driven AI systems also rely on memory systems to encode, store and recall information for future use.
A typical example is the LLM-driven agent system, which leverages memory to enhance the agent system’s abilities in reasoning, planning, personalization, and more \[ [10](https://arxiv.org/html/2504.15965v1#bib.bib10 "")\].

#### 2.2.1 Fundamental Dimensions of AI Memory

The memory of an LLM-driven AI system is closely related to the features of the LLM, that define how information is processed, stored, and retrieved based on its architecture and capabilities.
We primarily categorize and organize memory based on three dimensions: object (personal and system memory), form (non-parametric and parametric memory), and time (short-term and long-term memory).
These three dimensions comprehensively capture what type of information is retained (object), how information is stored (form), and how long it is preserved (time), aligning with both the functional structure of LLMs and practical requirements for efficient recall and adaptability.

##### Object Dimension

The object dimension is closely tied to the interaction between LLM-driven AI systems and humans, as it defines how information is categorized based on its source and purpose. On one hand, the system receives human input and feedback (i.e., personal memory); on the other hand, it generates a series of intermediate output results during task execution (i.e., system memory). Personal memory helps the system improve its understanding of user behavior and enhances its personalization capabilities, while system memory can strengthen the system’s reasoning ability, such as in approaches like CoT (Chain-of-Thought) \[ [23](https://arxiv.org/html/2504.15965v1#bib.bib23 "")\] and ReAct \[ [24](https://arxiv.org/html/2504.15965v1#bib.bib24 "")\].

##### Form Dimension

The form dimension focuses on how memory is represented and stored in LLM-driven AI systems, shaping how information is encoded and retrieved. Some memory is embedded within the model’s parameters through training, forming parametric memory, while other memory exists externally in structured databases or retrieval mechanisms, constituting non-parametric memory. Non-parametric memory serves as a supplementary knowledge source that can be dynamically accessed by the large language model, enhancing its ability to retrieve relevant information in real-time, as seen in retrieval-augmented generation (RAG) \[ [25](https://arxiv.org/html/2504.15965v1#bib.bib25 "")\].

##### Time Dimension

The time dimension defines how long memory is retained and how it influences the LLM’s interactions over different timescales. Short-term memory refers to contextual information temporarily maintained within the current conversation, enabling coherence and continuity in multi-turn dialogues. In contrast, long-term memory consists of information from past interactions that is stored in an external database and retrieved when needed, allowing the model to retain user-specific knowledge and improve personalization over time. This distinction ensures that the system can balance real-time responsiveness with accumulated learning for enhanced adaptability.

In addition to the three primary dimensions discussed above, memory can also be classified based on other criteria, such as modality, which distinguishes between unimodal memory (single data type) and multimodal memory (integrating multiple data types, such as text, images, and audio), or dynamics, which differentiates between static memory (fixed and unchanging) and streaming memory (dynamically updated in real-time). However, these alternative classifications are not considered the primary criteria here, as our focus is on the core structural aspects that most directly influence memory organization and retrieval in LLM-driven AI systems.

#### 2.2.2 Parallels Between Human and AI Memoryhttps://arxiv.org/html/2504.15965v1/extracted/6380911/memory-human-ai.pngFigure 1: Illustrating the parallels between human and AI memory.
The memory of LLM-driven AI system exhibits similarities to human memory in terms of structure and function. Human memory is generally categorized into short-term memory and long-term memory, a distinction that also applies to AI memory systems. Below, we draw a direct comparison between these categories, mapping human cognitive memory processes to their counterparts in intelligent AI systems.
Figure [1](https://arxiv.org/html/2504.15965v1#S2.F1 "Figure 1 ‣ 2.2.2 Parallels Between Human and AI Memory ‣ 2.2 Memory of LLM-driven AI Systems ‣ 2 Overview ‣ From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs") illustrates the parallels between human and AI memory.

- •


Sensory Memory: When an LLM-driven AI system perceives external information, it converts inputs such as text, images, speech, and video into machine-processable signals. This initial stage of information processing is analogous to human sensory memory, where raw data is briefly held before further cognitive processing. If these signals undergo additional processing, they transition into working memory, facilitating reasoning and decision-making. However, if no further processing or storage occurs, the information is quickly discarded, mirroring the transient nature of human sensory memory.

- •


Working Memory: The working memory of an AI system serves as a temporary storage and processing mechanism, enabling real-time reasoning and decision-making. It encompasses personal memory, such as contextual information retained during multi-turn dialogues, and system memory, including the chain of thoughts generated during task execution. As a form of short-term memory, working memory can undergo further processing and consolidation, eventually transitioning into long-term memory (e.g., episodic memory) that can be retrieved for future use. Additionally, during inference, large language models generate intermediate computational results, such as KV-Caches, which act as a form of parametric short-term memory that enhances efficiency by accelerating the inference process.

- •


Explicit Memory: The explicit memory of an AI system can be categorized into two distinct components. The first is non-parametric long-term memory, which involves the storage and retrieval of user-specific information, allowing the system to retain and utilize personalized data—analogous to episodic memory in humans. The second is parametric long-term memory, where factual knowledge and learned information are embedded within the model’s parameters, forming an internalized knowledge base—corresponding to semantic memory in human cognition. Together, these components enable the system to recall past interactions and apply acquired knowledge effectively.

- •


Implicit Memory: The implicit memory of an AI system encompasses the learned processes and patterns involved in task execution, enabling the development of specialized skills for specific tasks—analogous to procedural memory in humans. This form of memory is typically encoded within the model’s parameters, allowing the system to internalize task-related knowledge and perform operations efficiently without explicit recall.

Beyond these parallels, insights from human memory can further guide the design of more effective and efficient AI memory systems, enhancing their ability to process, store, and retrieve information in a more structured and adaptive manner.

#### 2.2.3 3D-8Q Memory Taxonomy

Building upon the three fundamental memory dimensions—object (personal & system), form (non-parametric & parametric), and time (short-term & long-term)—as well as the established parallels between human and AI memory, we propose a three-dimensional, eight-quadrant (3D-8Q) memory taxonomy for AI memory.
This memory taxonomy systematically categorizes AI memory based on its function, storage mechanism, and retention duration, providing a structured approach to understanding and optimizing AI memory systems.
Table [1](https://arxiv.org/html/2504.15965v1#S2.T1 "Table 1 ‣ 2.2.3 3D-8Q Memory Taxonomy ‣ 2.2 Memory of LLM-driven AI Systems ‣ 2 Overview ‣ From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs") presents the eight quadrants and their respective roles and functions.

| Object | Form | Time | Quadrant | Role | Function |
| --- | --- | --- | --- | --- | --- |
| Personal | Non-Parametric | Short-Term | I | Working Memory | Supports real-time context supplementation, enhancing the AI’s ability to maintain coherent interactions within a session. |
| Long-Term | II | Episodic Memory | Enables memory retention beyond session limits, allowing the system to recall and retrieve past user interactions for personalization. |
| Parametric | Short-Term | III | Working Memory | Temporarily enhances contextual understanding in ongoing interactions, improving response relevance and coherence. |
| Long-Term | IV | Semantic Memory | Facilitates the continuous integration of newly acquired knowledge into the model, improving adaptability and personalization |
| System | Non-Parametric | Short-Term | V | Working Memory | Assists in complex reasoning and decision-making by storing intermediate outputs such as chain-of-thought prompts. |
| Long-Term | VI | Procedural Memory | Captures historical experiences and self-reflection insights, enabling the AI to refine its reasoning and problem-solving skills over time. |
| Parametric | Short-Term | VII | Working Memory | Enhances computational efficiency through temporary parametric storage mechanisms such as KV-Caches, optimizing inference speed and reducing resource consumption. |
| Long-Term | VIII | Semantic Memory | Forms a foundational knowledge base encoded in the model’s parameters, serving as a long-term repository of factual and conceptual knowledge. |

Table 1: Three-dimensional, eight-quadrant (3D-8Q) memory taxonomy for LLM-driven AI systems.
Next, we will provide insights and descriptions of existing works from the perspectives of personal memory (in Section [3](https://arxiv.org/html/2504.15965v1#S3 "3 Personal Memory ‣ From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs")) and system memory (in Section [4](https://arxiv.org/html/2504.15965v1#S4 "4 System Memory ‣ From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs")). In particualr, personal memory focuses more on the individual data perceived and observed by the model from the environment, while system memory emphasizes the system’s internal or endogenous memory, such as the intermediate memory generated during task execution.

## 3 Personal Memory

Personal memory refers to the process of storing and utilizing human input and response data during interactions with an LLM-driven AI system.
The development and application of personal memory play a crucial role in enhancing AI systems’ personalization capabilities and improving user experience.
In this section, we explore the concept of personal memory and relevant research, examining both non-parametric and parametric approaches to its construction and implementation.
Table [2](https://arxiv.org/html/2504.15965v1#S3.T2 "Table 2 ‣ 3 Personal Memory ‣ From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs") shows the categories, features, and related research work of personal memory.

| Quadrant | Dimension | Feature | Models |
| --- | --- | --- | --- |
| I | Personal Non-Parametric Short-Term | Multi-Turn Dialogue | ChatGPT \[ [26](https://arxiv.org/html/2504.15965v1#bib.bib26 "")\], DeepSeek-Chat \[ [27](https://arxiv.org/html/2504.15965v1#bib.bib27 "")\], Claude \[ [28](https://arxiv.org/html/2504.15965v1#bib.bib28 "")\], QWEN-CHAT \[ [29](https://arxiv.org/html/2504.15965v1#bib.bib29 "")\], Llama 2-Chat \[ [30](https://arxiv.org/html/2504.15965v1#bib.bib30 "")\], Gemini \[ [31](https://arxiv.org/html/2504.15965v1#bib.bib31 "")\], PANGU-BOT \[ [32](https://arxiv.org/html/2504.15965v1#bib.bib32 "")\], ChatGLM \[ [33](https://arxiv.org/html/2504.15965v1#bib.bib33 "")\], OpenAssistant \[ [34](https://arxiv.org/html/2504.15965v1#bib.bib34 "")\] |
| II | Personal Non-Parametric Long-Term | Personal Assistant | ChatGPT Memory \[ [18](https://arxiv.org/html/2504.15965v1#bib.bib18 "")\], Apple Intelligence \[ [19](https://arxiv.org/html/2504.15965v1#bib.bib19 "")\], Microsoft Recall \[ [35](https://arxiv.org/html/2504.15965v1#bib.bib35 "")\], Me.bot \[ [36](https://arxiv.org/html/2504.15965v1#bib.bib36 "")\] |
|  |  | Open-Source Framework | MemoryScope \[ [21](https://arxiv.org/html/2504.15965v1#bib.bib21 "")\], mem0 \[ [20](https://arxiv.org/html/2504.15965v1#bib.bib20 "")\], Memary \[ [37](https://arxiv.org/html/2504.15965v1#bib.bib37 "")\], LangGraph Memory \[ [38](https://arxiv.org/html/2504.15965v1#bib.bib38 "")\], Charlie Mnemonic \[ [39](https://arxiv.org/html/2504.15965v1#bib.bib39 "")\], Memobase \[ [40](https://arxiv.org/html/2504.15965v1#bib.bib40 "")\], Letta \[ [41](https://arxiv.org/html/2504.15965v1#bib.bib41 "")\], Cognee \[ [42](https://arxiv.org/html/2504.15965v1#bib.bib42 "")\] |
|  |  | Construction | MPC \[ [43](https://arxiv.org/html/2504.15965v1#bib.bib43 "")\], RET-LLM \[ [44](https://arxiv.org/html/2504.15965v1#bib.bib44 "")\], MemoryBank \[ [17](https://arxiv.org/html/2504.15965v1#bib.bib17 "")\], MemGPT \[ [45](https://arxiv.org/html/2504.15965v1#bib.bib45 "")\], KGT \[ [46](https://arxiv.org/html/2504.15965v1#bib.bib46 "")\], Evolving Conditional Memory \[ [47](https://arxiv.org/html/2504.15965v1#bib.bib47 "")\], SECOM \[ [48](https://arxiv.org/html/2504.15965v1#bib.bib48 "")\], Memory3\[ [49](https://arxiv.org/html/2504.15965v1#bib.bib49 "")\], MemInsight \[ [50](https://arxiv.org/html/2504.15965v1#bib.bib50 "")\] |
|  |  | Management | MemoChat \[ [51](https://arxiv.org/html/2504.15965v1#bib.bib51 "")\], MemoryBank \[ [17](https://arxiv.org/html/2504.15965v1#bib.bib17 "")\], RMM \[ [52](https://arxiv.org/html/2504.15965v1#bib.bib52 "")\], LD-Agent \[ [53](https://arxiv.org/html/2504.15965v1#bib.bib53 "")\], A-MEM \[ [54](https://arxiv.org/html/2504.15965v1#bib.bib54 "")\], Generative Agents \[ [55](https://arxiv.org/html/2504.15965v1#bib.bib55 "")\], EMG-RAG \[ [56](https://arxiv.org/html/2504.15965v1#bib.bib56 "")\], KGT \[ [46](https://arxiv.org/html/2504.15965v1#bib.bib46 "")\], LLM-Rsum \[ [57](https://arxiv.org/html/2504.15965v1#bib.bib57 "")\], COMEDY \[ [58](https://arxiv.org/html/2504.15965v1#bib.bib58 "")\] |
|  |  | Retrieval | RET-LLM \[ [44](https://arxiv.org/html/2504.15965v1#bib.bib44 "")\], ChatDB \[ [59](https://arxiv.org/html/2504.15965v1#bib.bib59 "")\], Human-like Memory \[ [60](https://arxiv.org/html/2504.15965v1#bib.bib60 "")\], HippoRAG \[ [13](https://arxiv.org/html/2504.15965v1#bib.bib13 "")\], HippoRAG 2 \[ [61](https://arxiv.org/html/2504.15965v1#bib.bib61 "")\], EgoRAG \[ [62](https://arxiv.org/html/2504.15965v1#bib.bib62 "")\], MemInsight \[ [50](https://arxiv.org/html/2504.15965v1#bib.bib50 "")\] |
|  |  | Usage | MemoCRS \[ [63](https://arxiv.org/html/2504.15965v1#bib.bib63 "")\], RecMind \[ [64](https://arxiv.org/html/2504.15965v1#bib.bib64 "")\], RecAgent \[ [65](https://arxiv.org/html/2504.15965v1#bib.bib65 "")\], InteRecAgent \[ [66](https://arxiv.org/html/2504.15965v1#bib.bib66 "")\], SCM \[ [67](https://arxiv.org/html/2504.15965v1#bib.bib67 "")\], ChatDev \[ [68](https://arxiv.org/html/2504.15965v1#bib.bib68 "")\], MetaAgents \[ [69](https://arxiv.org/html/2504.15965v1#bib.bib69 "")\], S3\[ [70](https://arxiv.org/html/2504.15965v1#bib.bib70 "")\], TradingGPT \[ [71](https://arxiv.org/html/2504.15965v1#bib.bib71 "")\], Memolet \[ [72](https://arxiv.org/html/2504.15965v1#bib.bib72 "")\], Synaptic Resonance \[ [14](https://arxiv.org/html/2504.15965v1#bib.bib14 "")\], MemReasoner \[ [73](https://arxiv.org/html/2504.15965v1#bib.bib73 "")\] |
|  |  | Benchmark | MADial-Bench \[ [74](https://arxiv.org/html/2504.15965v1#bib.bib74 "")\], LOCOMO \[ [75](https://arxiv.org/html/2504.15965v1#bib.bib75 "")\], MemDaily \[ [76](https://arxiv.org/html/2504.15965v1#bib.bib76 "")\], ChMapData \[ [77](https://arxiv.org/html/2504.15965v1#bib.bib77 "")\], MSC \[ [78](https://arxiv.org/html/2504.15965v1#bib.bib78 "")\], MMRC \[ [79](https://arxiv.org/html/2504.15965v1#bib.bib79 "")\], Ego4D \[ [80](https://arxiv.org/html/2504.15965v1#bib.bib80 "")\], EgoLife \[ [62](https://arxiv.org/html/2504.15965v1#bib.bib62 "")\], BABILong \[ [81](https://arxiv.org/html/2504.15965v1#bib.bib81 ""), [82](https://arxiv.org/html/2504.15965v1#bib.bib82 "")\] |
| III | Personal Parametric Short-Term | Caching for Acceleration | Prompt Cache \[ [83](https://arxiv.org/html/2504.15965v1#bib.bib83 "")\], Contextual Retrieval \[ [84](https://arxiv.org/html/2504.15965v1#bib.bib84 "")\] |
| IV | Personal Parametric Long-Term | Knowledge Editing | Character-LLM \[ [85](https://arxiv.org/html/2504.15965v1#bib.bib85 "")\], AI-Native Memory \[ [36](https://arxiv.org/html/2504.15965v1#bib.bib36 "")\], MemoRAG \[ [86](https://arxiv.org/html/2504.15965v1#bib.bib86 "")\], Echo \[ [87](https://arxiv.org/html/2504.15965v1#bib.bib87 "")\] |

Table 2: Personal Memory
### 3.1 Contextual Personal Memory

In personal memory, the non-parametric contextual memory that can be loaded is generally divided into two categories: the short-term memory of the current session’s multi-turn dialogue and the long-term memory of historical dialogues across sessions.
The former can effectively supplement contextual information, while the latter can effectively fill in missing information and overcome the limitations of context length.

#### 3.1.1 Loading Multi-Turn Dialogue (Quadrant-I)

In multi-turn dialogue scenarios, the conversation history of the current session can significantly enhance the LLM-driven AI system’s understanding of the user’s real-time intent, leading to more relevant and contextually appropriate responses.
Many modern dialogue systems are capable of handling multi-turn conversations and fully consider the current dialogue context in their responses.
Notable examples include ChatGPT \[ [26](https://arxiv.org/html/2504.15965v1#bib.bib26 "")\], DeepSeek-Chat \[ [27](https://arxiv.org/html/2504.15965v1#bib.bib27 "")\], and Claude \[ [28](https://arxiv.org/html/2504.15965v1#bib.bib28 "")\], which excel at maintaining coherence and relevance over extended interactions.

For instance, ChatGPT \[ [26](https://arxiv.org/html/2504.15965v1#bib.bib26 "")\] is a prime example of a multi-turn dialogue system where the conversation history of the current session serves as short-term memory, helping to supplement the contextual information of the dialogue.
In ChatGPT, the dialogue memory is encoded in a role-content format, with distinct roles such as “User” and “Assistant”.
This encoding allows the system to maintain clarity regarding the speaker and the flow of the conversation.

Through effective dialogue management at different levels, including “Assistant”, “Threads”, “Messages”, and “Runs”, the system can precisely track the state of each turn and each step of the conversation, ensuring continuity and consistency in interactions.
Additionally, when the conversation length becomes too extensive, the dialogue system manages the conversation’s input by truncating the number of turns, thereby preventing the input from exceeding the model’s length limitations.
This ensures that the system can continue processing the dialogue without losing track of essential context, maintaining the effectiveness of multi-turn interactions.

#### 3.1.2 Memory Retrieval-Augmented Generation (Quadrant-II)

In cross-session dialogue scenarios, retrieving relevant user long-term memories from historical conversations can effectively supplement missing information in the current session, such as personal preferences and character relationships.
The advantage of memory retrieval-augmented generation is that large language models (LLMs) do not need to load all multi-session conversations.
Given the limited length of LLMs’ context windows—even when extended to millions of tokens—retrieving relevant information from historical sessions is also more efficient and cost-effective in terms of computation.
In addition to multi-session conversations, long-term personal memory also encompasses users’ behavioral history, preferences, and interaction records with AI agents over an extended period of time.

By leveraging retrieval-augmented generation from long-term memory, LLM-driven AI systems can better tailor their responses and behaviors, thereby improving user satisfaction and engagement.
For instance, a personal assistant that remembers a user’s preferred news sources can prioritize those outlets in daily briefings, while a recommendation system that understands past viewing habits can suggest content more aligned with the user’s tastes.
Currently, many commercial and open-source platforms are striving to construct and utilize long-term memory for personalized AI systems—for example, ChatGPT Memory \[ [18](https://arxiv.org/html/2504.15965v1#bib.bib18 "")\] and Me.bot \[ [36](https://arxiv.org/html/2504.15965v1#bib.bib36 "")\] for personal assistants, and MemoryScope \[ [21](https://arxiv.org/html/2504.15965v1#bib.bib21 "")\] and mem0 \[ [20](https://arxiv.org/html/2504.15965v1#bib.bib20 "")\] as open-source frameworks.
Long-term personal memory typically follow four core processing stages: construction, management, retrieval, and usage.
The second section of Table [2](https://arxiv.org/html/2504.15965v1#S3.T2 "Table 2 ‣ 3 Personal Memory ‣ From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs") (organized by rows) provides an overview of existing work on personal non-parametric long-term memory, classified based on their primary contributions.

##### Construction

The construction of user memory requires extraction and refinement from raw memory data, such as multi-turn conversations. This process is analogous to human memory consolidation—the process of stabilizing and strengthening memories to facilitate their long-term storage.
Well-organized long-term memory enhances both the efficiency of storage and the effectiveness of retrieval in user memory.
For example, MemoryBank \[ [17](https://arxiv.org/html/2504.15965v1#bib.bib17 "")\] leverages a memory module to store conversation histories and summaries of key events, enabling the construction of a long-term user profile.
Similarly, RET-LLM \[ [44](https://arxiv.org/html/2504.15965v1#bib.bib44 "")\] uses its memory module to retain essential factual knowledge about the external world, allowing the agent to monitor and update real-time environmental context relevant to the user.
In addition, to accommodate different types of memory, a variety of storage formats have been developed, including key-value, graph, and vector representations.
Specifically, key-value formats \[ [44](https://arxiv.org/html/2504.15965v1#bib.bib44 ""), [50](https://arxiv.org/html/2504.15965v1#bib.bib50 ""), [63](https://arxiv.org/html/2504.15965v1#bib.bib63 "")\] enable efficient access to structured information such as user facts and preferences.
Graph-based formats \[ [46](https://arxiv.org/html/2504.15965v1#bib.bib46 ""), [13](https://arxiv.org/html/2504.15965v1#bib.bib13 ""), [61](https://arxiv.org/html/2504.15965v1#bib.bib61 ""), [20](https://arxiv.org/html/2504.15965v1#bib.bib20 "")\] are designed to capture and represent relationships among entities, such as individuals and events.
Meanwhile, vector formats \[ [17](https://arxiv.org/html/2504.15965v1#bib.bib17 ""), [48](https://arxiv.org/html/2504.15965v1#bib.bib48 ""), [20](https://arxiv.org/html/2504.15965v1#bib.bib20 "")\], which are typically derived from textual, visual, or audio memory representations, are utilized to encode the semantic meaning and contextual information of conversations.

##### Management

The management of user memory involves further processing and refinement of previously constructed memories, such as deduplication, merging, and conflict resolution. This process is analogous to human memory reconsolidation and reflection, where existing memories are reactivated, updated, and integrated to maintain coherence and relevance over time.
For instance, Reflective Memory Management (RMM) \[ [52](https://arxiv.org/html/2504.15965v1#bib.bib52 "")\] is a user long-term memory management framework that combines Prospective Reflection for dynamic summarization with Retrospective Reflection for retrieval optimization via reinforcement learning.
This dual-process approach addresses limitations such as rigid memory granularity and fixed retrieval mechanisms, enhancing the accuracy and flexibility of long-term memory management.
LD-Agent \[ [53](https://arxiv.org/html/2504.15965v1#bib.bib53 "")\] enhances long-term dialogue personalization and consistency by constructing personalized persona information for both users and agents through a dynamic persona modeling module, while integrating retrieved memories to optimize response generation.
A-MEM \[ [54](https://arxiv.org/html/2504.15965v1#bib.bib54 "")\] introduces a self-organizing memory system inspired by the Zettelkasten method \[ [88](https://arxiv.org/html/2504.15965v1#bib.bib88 "")\], which constructs interconnected knowledge networks through dynamic indexing, linking, and memory evolution, enabling LLM agents to more flexibly organize, update, and retrieve long-term memories, thereby enhancing task adaptability and contextual awareness.
In addition, MemoryBank \[ [17](https://arxiv.org/html/2504.15965v1#bib.bib17 "")\] incorporates a memory updating mechanism inspired by the Ebbinghaus Forgetting Curve \[ [89](https://arxiv.org/html/2504.15965v1#bib.bib89 "")\], allowing the AI to forget or reinforce memories based on the time elapsed and their relative importance, thereby enabling a more human-like memory system and enhancing the user experience.

##### Retrieval

Retrieving personal memory involves identifying memory entries relevant to the user’s current request, and the retrieval method is closely tied to how the memory is stored.
For key-value memory, ChatDB \[ [59](https://arxiv.org/html/2504.15965v1#bib.bib59 "")\] performs retrieval using SQL queries over structured databases.
RET-LLM \[ [44](https://arxiv.org/html/2504.15965v1#bib.bib44 "")\], on the other hand, employs a fuzzy search to retrieve triplet-structured memories, where information is stored as relationships between two entities connected by a predefined relation.
For graph-based memory, HippoRAG \[ [13](https://arxiv.org/html/2504.15965v1#bib.bib13 "")\] constructs knowledge graphs over entities, phrases, and summarization to recall more relative and comprehensive memories, while HippoRAG 2 \[ [61](https://arxiv.org/html/2504.15965v1#bib.bib61 "")\] further combines original passages with phrase-based knowledge graphs to incorporate both conceptual and contextual information.
For vector memory, MemoryBank \[ [17](https://arxiv.org/html/2504.15965v1#bib.bib17 "")\] adopts a dual-tower dense retrieval model, similar to Dense Passage Retrieval \[ [90](https://arxiv.org/html/2504.15965v1#bib.bib90 "")\], to accurately identify relevant memories. The resulting vector representations are then indexed using FAISS \[ [91](https://arxiv.org/html/2504.15965v1#bib.bib91 "")\] for efficient similarity-based retrieval.

##### Usage

The use of personal memory can effectively empower downstream applications with personalization, enhancing the user’s individualized experience.
For instance, the recalled relevant memory is used as contextual information to enhance the personalized recommendation and response capability of the conversational recommender agents \[ [63](https://arxiv.org/html/2504.15965v1#bib.bib63 ""), [64](https://arxiv.org/html/2504.15965v1#bib.bib64 ""), [65](https://arxiv.org/html/2504.15965v1#bib.bib65 ""), [66](https://arxiv.org/html/2504.15965v1#bib.bib66 "")\], improving the personalized user experience.
In addition to memory-augmented personalized dialogue and recommendation, personal memory can also be leveraged to enhance a wide range of applications, including software development \[ [68](https://arxiv.org/html/2504.15965v1#bib.bib68 "")\], social-network simulation \[ [69](https://arxiv.org/html/2504.15965v1#bib.bib69 ""), [70](https://arxiv.org/html/2504.15965v1#bib.bib70 "")\], and financial trading \[ [71](https://arxiv.org/html/2504.15965v1#bib.bib71 "")\].

To facilitate in-depth research on personal memory, a variety of memory-related benchmarks have emerged in recent years, including long-term conversational memory (MADial-Bench \[ [74](https://arxiv.org/html/2504.15965v1#bib.bib74 "")\], LOCOMO \[ [75](https://arxiv.org/html/2504.15965v1#bib.bib75 "")\], MSC \[ [78](https://arxiv.org/html/2504.15965v1#bib.bib78 "")\]), everyday life memory (MemDaily \[ [76](https://arxiv.org/html/2504.15965v1#bib.bib76 "")\]), memory-aware proactive dialogue (ChMapData \[ [77](https://arxiv.org/html/2504.15965v1#bib.bib77 "")\]), multimodal dialogue memory (MMRC \[ [79](https://arxiv.org/html/2504.15965v1#bib.bib79 "")\]), egocentric video understanding (Ego4D \[ [80](https://arxiv.org/html/2504.15965v1#bib.bib80 "")\], EgoLife \[ [62](https://arxiv.org/html/2504.15965v1#bib.bib62 "")\]), and long-context reasoning-in-a-haystack (BABILong \[ [81](https://arxiv.org/html/2504.15965v1#bib.bib81 ""), [82](https://arxiv.org/html/2504.15965v1#bib.bib82 "")\]).

### 3.2 Parametric Personal Memory

In addition to external non-parametric memory, a user’s personal memory can also be stored parametrically. Specifically, personal data can be used to fine-tune an LLM, embedding the memory directly into its parameters (i.e., parametric long-term memory) to create a personalized LLM . Alternatively, historical dialogues can be cached as prompts during inference (i.e., parametric short-term memory), enabling quick reuse in future interactions.

#### 3.2.1 Memory Caching For Acceleration (Quadrant-III)

Personal parametric short-term memory typically refers to intermediate attention states produced by the LLM when processing personal data, which is usually utilized as memory caches to accelerate inference.
Specifically, prompt caching \[ [83](https://arxiv.org/html/2504.15965v1#bib.bib83 "")\] is usually used as an efficient data management technique that allows for the pre-storage of large amounts of personal data or information that may be frequently requested, such as a user’s conversational history.
For instance, during multi-turn dialogues, the dialogue system can quickly provide the personal context information directly from the parametric memory cache, avoiding the need to recalculate or retrieve it from the original data source, saving both time and resources.
Major platforms such as DeepSeek, Anthropic, OpenAI, and Google employ prompt caching to reduce API call costs and improve response speed in dialogue scenarios.
Moreover, personal parametric short-term memory can enhance the performance of retrieval-augmented generation (RAG) through Contextual Retrieval \[ [84](https://arxiv.org/html/2504.15965v1#bib.bib84 "")\], where prompt caching helps reduce the overhead of generating contextualized chunks.
At present, research specifically targeting caching techniques for personal memory data remains limited. Instead, most existing work considers caching as a fundamental capability of system memory, particularly in the context of key-value (KV) management and KV reuse. A more detailed discussion of these aspects is provided in Section [4](https://arxiv.org/html/2504.15965v1#S4 "4 System Memory ‣ From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs").

#### 3.2.2 Personalized Knowledge Editing (Quadrant-IV)

Personal parametric long-term memory utilizes personalized Knowledge Editing technology \[ [92](https://arxiv.org/html/2504.15965v1#bib.bib92 "")\], such as Parameter-Efficient Fine-Tuning (PEFT) \[ [93](https://arxiv.org/html/2504.15965v1#bib.bib93 "")\], to encode personal data into the LLM’s parameters in a parametric manner, thereby facilitating the long-term, parameterized storage of memory.
For instance, Character-LLM \[ [85](https://arxiv.org/html/2504.15965v1#bib.bib85 "")\] enables the role-playing of specific characters, such as Beethoven, Queen Cleopatra, Julius Caesar, etc., by training large language models to remember the roles and experiences of these characters.
AI-Native Memory \[ [36](https://arxiv.org/html/2504.15965v1#bib.bib36 "")\] proposes using deep neural network models, specifically large language models (LLMs), as Lifelong Personal Models (LPMs) to parameterize, compress, and continuously evolve personal memory through user interactions, enabling a more comprehensive understanding of the user.
MemoRAG \[ [86](https://arxiv.org/html/2504.15965v1#bib.bib86 "")\] utilizes LLM parametric memory to store user conversation history and preferences, forming a personalized global memory that enhances personalization and enables tailored recommendations.
Echo \[ [87](https://arxiv.org/html/2504.15965v1#bib.bib87 "")\] is a large language model enhanced with temporal episodic memory, designed to improve performance in applications requiring multi-turn, complex memory-based dialogues.
The parameterization of personal long-term memory presents several challenges, notably the need to fine-tune models on individual user data, which demands substantial computational resources. This requirement significantly hinders the scalability and practical deployment of parametric approaches to long-term personal memory.

### 3.3 Discussion

In this section, we describe personal memory and related work from the perspectives of non-parametric and parametric approaches.
Specifically, personal non-parametric short-term memory necessitates efficient mechanisms for memory encoding and management. Existing literature predominantly emphasizes the design and implementation of systems that facilitate the construction, management, retrieval, and effective utilization of a user’s personal non-parametric long-term memory.
In contrast, personal parametric short-term memory can employ techniques such as prompt caching to reduce computational costs and enhance efficiency.
Parametric long-term memory offers advantages in memory compression, thereby supporting a more comprehensive and global representation of the user’s accumulated experiences.
Recent trends in the field indicate a growing interest in integrating both short-term and long-term memory paradigms, wherein parametric and non-parametric memory components complement and reinforce one another.
The subsequent section will present a detailed discussion of system memory and its associated research developments.

## 4 System Memory

System memory constitutes a critical component of LLM-driven AI systems.
It encompasses a sequence of intermediate representations or results generated throughout the task execution process.
By leveraging system memory, LLM-driven AI systems can enhance their capabilities in reasoning, planning, and other higher-order cognitive functions.
Moreover, the effective use of system memory contributes to the system’s capacity for self-evolution and continual improvement.
In this section, we examine system memory and its associated research from both non-parametric and parametric perspectives.

|     |     |     |     |
| --- | --- | --- | --- |
| Quadrant | Dimension | Feature | Models |
| V | System Non-Parametric Short-Term | Reasoning & Planning Enhancement | ReAct \[ [24](https://arxiv.org/html/2504.15965v1#bib.bib24 "")\], RAP \[ [94](https://arxiv.org/html/2504.15965v1#bib.bib94 "")\], Reflexion \[ [95](https://arxiv.org/html/2504.15965v1#bib.bib95 "")\], Talker-Reasoner \[ [96](https://arxiv.org/html/2504.15965v1#bib.bib96 "")\], TPTU \[ [97](https://arxiv.org/html/2504.15965v1#bib.bib97 "")\] |
| VI | System Non-Parametric Long-Term | Reflection & Refinement | Buffer of Thoughts \[ [98](https://arxiv.org/html/2504.15965v1#bib.bib98 "")\], AWM \[ [99](https://arxiv.org/html/2504.15965v1#bib.bib99 "")\], Think-in-Memory \[ [100](https://arxiv.org/html/2504.15965v1#bib.bib100 "")\], GITM \[ [101](https://arxiv.org/html/2504.15965v1#bib.bib101 "")\], Voyager \[ [102](https://arxiv.org/html/2504.15965v1#bib.bib102 "")\], Retroformer \[ [103](https://arxiv.org/html/2504.15965v1#bib.bib103 "")\], Expel \[ [104](https://arxiv.org/html/2504.15965v1#bib.bib104 "")\], Synapse \[ [105](https://arxiv.org/html/2504.15965v1#bib.bib105 "")\], MetaGPT \[ [106](https://arxiv.org/html/2504.15965v1#bib.bib106 "")\], Learned Memory Bank \[ [107](https://arxiv.org/html/2504.15965v1#bib.bib107 "")\], M+ \[ [108](https://arxiv.org/html/2504.15965v1#bib.bib108 "")\] |
| VII | System Parametric Short-Term | KV Management | LookupFFN \[ [109](https://arxiv.org/html/2504.15965v1#bib.bib109 "")\], ChunkKV \[ [110](https://arxiv.org/html/2504.15965v1#bib.bib110 "")\], vLLM \[ [111](https://arxiv.org/html/2504.15965v1#bib.bib111 "")\], FastServe \[ [112](https://arxiv.org/html/2504.15965v1#bib.bib112 "")\], StreamingLLM \[ [113](https://arxiv.org/html/2504.15965v1#bib.bib113 "")\], Orca \[ [114](https://arxiv.org/html/2504.15965v1#bib.bib114 "")\], DistServe \[ [115](https://arxiv.org/html/2504.15965v1#bib.bib115 "")\], LLM.int8() \[ [116](https://arxiv.org/html/2504.15965v1#bib.bib116 "")\], FastGen \[ [117](https://arxiv.org/html/2504.15965v1#bib.bib117 "")\], Train Large, Then Compress \[ [118](https://arxiv.org/html/2504.15965v1#bib.bib118 "")\], Scissorhands \[ [119](https://arxiv.org/html/2504.15965v1#bib.bib119 "")\], H2O \[ [120](https://arxiv.org/html/2504.15965v1#bib.bib120 "")\], Mooncake \[ [121](https://arxiv.org/html/2504.15965v1#bib.bib121 "")\], MemServe \[ [122](https://arxiv.org/html/2504.15965v1#bib.bib122 "")\], SLM Serving \[ [123](https://arxiv.org/html/2504.15965v1#bib.bib123 "")\], IMPRESS \[ [124](https://arxiv.org/html/2504.15965v1#bib.bib124 "")\], AdaServe \[ [125](https://arxiv.org/html/2504.15965v1#bib.bib125 "")\], MPIC \[ [126](https://arxiv.org/html/2504.15965v1#bib.bib126 "")\], IntelLLM \[ [127](https://arxiv.org/html/2504.15965v1#bib.bib127 "")\] |
|  |  | KV Reuse | KV Cache \[ [128](https://arxiv.org/html/2504.15965v1#bib.bib128 "")\], Prompt Cache \[ [83](https://arxiv.org/html/2504.15965v1#bib.bib83 "")\], Contextual Retrieval \[ [84](https://arxiv.org/html/2504.15965v1#bib.bib84 "")\], CacheGen \[ [129](https://arxiv.org/html/2504.15965v1#bib.bib129 "")\], ChunkAttention \[ [130](https://arxiv.org/html/2504.15965v1#bib.bib130 "")\], RAGCache \[ [131](https://arxiv.org/html/2504.15965v1#bib.bib131 "")\], SGLang \[ [132](https://arxiv.org/html/2504.15965v1#bib.bib132 "")\], Ada-KV \[ [133](https://arxiv.org/html/2504.15965v1#bib.bib133 "")\], HCache \[ [134](https://arxiv.org/html/2504.15965v1#bib.bib134 "")\], Cake \[ [135](https://arxiv.org/html/2504.15965v1#bib.bib135 "")\], EPIC \[ [136](https://arxiv.org/html/2504.15965v1#bib.bib136 "")\], RelayAttention \[ [137](https://arxiv.org/html/2504.15965v1#bib.bib137 "")\], Marconi \[ [138](https://arxiv.org/html/2504.15965v1#bib.bib138 "")\], IKS \[ [139](https://arxiv.org/html/2504.15965v1#bib.bib139 "")\], FastCache \[ [140](https://arxiv.org/html/2504.15965v1#bib.bib140 "")\], Cache-Craft \[ [141](https://arxiv.org/html/2504.15965v1#bib.bib141 "")\], KVLink \[ [142](https://arxiv.org/html/2504.15965v1#bib.bib142 "")\], RAGServe \[ [143](https://arxiv.org/html/2504.15965v1#bib.bib143 "")\], BumbleBee \[ [144](https://arxiv.org/html/2504.15965v1#bib.bib144 "")\] |
| VIII | System Parametric Long-Term | Parametric Memory Structures | Memorizing Transformer \[ [145](https://arxiv.org/html/2504.15965v1#bib.bib145 "")\], Focused Transformer \[ [146](https://arxiv.org/html/2504.15965v1#bib.bib146 "")\], MAC \[ [147](https://arxiv.org/html/2504.15965v1#bib.bib147 "")\], MemoryLLM \[ [148](https://arxiv.org/html/2504.15965v1#bib.bib148 "")\], WISE \[ [149](https://arxiv.org/html/2504.15965v1#bib.bib149 "")\], LongMem \[ [150](https://arxiv.org/html/2504.15965v1#bib.bib150 "")\], LM2 \[ [151](https://arxiv.org/html/2504.15965v1#bib.bib151 "")\], Titans \[ [152](https://arxiv.org/html/2504.15965v1#bib.bib152 "")\] |

Table 3: System Memory
### 4.1 Contextual System Memory

From a temporal perspective, non-parametric short-term system memory refers to a series of reasoning and action results generated by large language models during task execution.
This form of memory supports enhanced reasoning and planning within the context of the current task, thereby contributing to improved task accuracy, efficiency, and overall completion rates.
In contrast, non-parametric long-term system memory represents a more abstracted and generalized form of short-term memory.
It encompasses the consolidation of prior successful experiences and mechanisms of self-reflection based on historical interactions, which collectively facilitate the continual evolution and adaptive enhancement of LLM-driven AI systems.

#### 4.1.1 Reasoning & Planning Enhancement (Quadrant-V)

Analogous to human cognition, the reasoning and planning processes of large language models (LLMs) give rise to a sequence of short-term intermediate outputs. These outputs may reflect task-related attempts, which can be either successful or erroneous. Regardless of their correctness, such intermediate results serve as informative and constructive references that can guide subsequent task execution. This form of system non-parametric short-term memory plays a pivotal role in LLM-driven AI systems. Empirical evidence demonstrates that leveraging this memory structure significantly enhances the reasoning and planning capabilities of LLMs.
For instance, ReAct \[ [24](https://arxiv.org/html/2504.15965v1#bib.bib24 "")\] integrates reasoning and action by generating intermediate reasoning steps alongside corresponding actions, enabling the model to alternate between thought and execution. This approach facilitates intelligent planning and adaptive decision-making in complex problem-solving scenarios. Similarly, Reflexion \[ [95](https://arxiv.org/html/2504.15965v1#bib.bib95 "")\] introduces mechanisms for dynamic memory and self-reflection, allowing the LLM to self-evaluate and iteratively refine its behavior based on prior errors or limitations. This self-improvement loop promotes enhanced performance in future tasks, resembling a continuous learning and optimization process.

#### 4.1.2 Reflection & Refinement (Quadrant-VI)

The development of system non-parametric long-term memory parallels the human process of learning from both successes and failures.
It involves the reflection upon and refinement of accumulated short-term memory traces.
This memory mechanism enables the system not only to retain and replicate effective strategies from past experiences but also to extract valuable lessons from failures, thereby minimizing the likelihood of repeated errors.
Through continuous updating and optimization, the system incrementally enhances its decision-making capabilities and improves its responsiveness to novel challenges.
Moreover, the progressive accumulation of long-term memory empowers the system to address increasingly complex tasks with greater adaptability and resilience.
For instance, Buffer of Thoughts (BoT) \[ [98](https://arxiv.org/html/2504.15965v1#bib.bib98 "")\] refines the chain of thoughts from historical tasks to form thought templates, which are then stored in a memory repository, guiding future reasoning and decision-making processes.
Agent Workflow Memory (AWM) \[ [99](https://arxiv.org/html/2504.15965v1#bib.bib99 "")\] introduces reusable paths, called workflows, and guides subsequent task generation by selecting different workflows.
Think-in-Memory (TiM) \[ [100](https://arxiv.org/html/2504.15965v1#bib.bib100 "")\] continuously generates new thoughts based on conversation history, which is more conducive to reasoning and computation compared to raw observational data.
Ghost in the Minecraft (GITM) \[ [101](https://arxiv.org/html/2504.15965v1#bib.bib101 "")\] uses reference plans recorded in memory, allowing the agent planner to more efficiently handle encountered tasks, thereby improving task execution success rates.
Voyager \[ [102](https://arxiv.org/html/2504.15965v1#bib.bib102 "")\] refines skills based on environmental feedback and stores acquired skills in memory, forming a skill library for future reuse in similar situations (e.g., fighting zombies vs. fighting spiders).
Retroformer \[ [103](https://arxiv.org/html/2504.15965v1#bib.bib103 "")\] leverages recent interaction trajectories as short-term memory and reflective feedback from past failures as long-term memory to guide decision-making and reasoning.
ExpeL \[ [104](https://arxiv.org/html/2504.15965v1#bib.bib104 "")\] enhances task resolution by drawing on contextualized successful examples and abstracting insights from both successes and failures through comparative and pattern-based analysis of past experiences.

### 4.2 Parametric System Memory

The parametric system memory refers to the temporary storage of knowledge information in parametric forms, such as KV Cache \[ [128](https://arxiv.org/html/2504.15965v1#bib.bib128 "")\], during the inference process (short-term memory), or the long-term editing and storage of knowledge information in the model parameters (long-term memory).
The former, parametric short-term system memory, corresponds to human working memory, enabling cost reduction and efficiency improvement in large language model inference.
The latter, parametric long-term system memory, corresponds to human semantic memory, facilitating the efficient integration of new knowledge.

#### 4.2.1 KV Management & Reuse (Quadrant-VII)

Parametric short-term system memory primarily focuses on the management and reuse of attention keys (Key) and values (Value) in LLMs, aiming to address issues such as high inference costs and latency during the reasoning process.
KV management optimizes memory efficiency and inference performance through techniques such as KV cache organization \[ [111](https://arxiv.org/html/2504.15965v1#bib.bib111 "")\], compression \[ [110](https://arxiv.org/html/2504.15965v1#bib.bib110 "")\], and quantization \[ [116](https://arxiv.org/html/2504.15965v1#bib.bib116 "")\].
In particular, vLLM \[ [111](https://arxiv.org/html/2504.15965v1#bib.bib111 "")\] is a high-efficiency LLM serving system built on PagedAttention, a virtual memory-inspired attention mechanism that enables near-zero KV cache waste and flexible sharing across requests, substantially improving batching efficiency and inference throughput.
ChunkKV \[ [110](https://arxiv.org/html/2504.15965v1#bib.bib110 "")\] is a method for compressing the key-value cache in long-context inference with LLMs by grouping tokens into semantic chunks, retaining the most informative ones, and enabling layer-wise index reuse, thereby reducing memory and computational costs while outperforming existing approaches on several benchmarks.
LLM.int8() \[ [116](https://arxiv.org/html/2504.15965v1#bib.bib116 "")\] is a mixed-precision quantization method that combines vector-wise Int8 quantization with selective 16-bit handling of emergent outlier features, enabling memory-efficient inference of large language models (up to 175B parameters) without performance degradation.

Meanwhile, KV reuse focuses on reusing inference-related parameters through token-level KV Cache \[ [128](https://arxiv.org/html/2504.15965v1#bib.bib128 "")\] and sentence-level Prompt Cache \[ [83](https://arxiv.org/html/2504.15965v1#bib.bib83 "")\], which helps reduce computational costs and improve the efficiency of large language model (LLM) usage.
Specifically, KV Cache \[ [128](https://arxiv.org/html/2504.15965v1#bib.bib128 "")\] stores the attention keys (Key) and values (Value) generated by the neural network during sequence generation, allowing them to be reused in subsequent inference steps. This reuse accelerates attention computation in long-text generation and reduces redundant computation.
In contrast, Prompt Cache \[ [83](https://arxiv.org/html/2504.15965v1#bib.bib83 "")\] operates at the sentence level by caching previous input prompts along with their corresponding output results. When similar prompts are encountered, the LLM can retrieve and return cached responses directly, saving computation and accelerating response generation.
By avoiding frequent recomputation of identical or similar contexts, KV reuse enables more efficient inference and significantly reduces computational overhead.
Additionally, it enhances the flexibility and responsiveness of LLM-based systems in handling continuous or interactive tasks.
Building on these ideas, RAGCache \[ [131](https://arxiv.org/html/2504.15965v1#bib.bib131 "")\] introduces a multilevel dynamic caching system tailored for Retrieval-Augmented Generation (RAG), which caches intermediate knowledge states, optimizes memory replacement policies based on LLM inference and retrieval patterns, and overlaps retrieval with inference to significantly reduce latency and improve throughput.

Parametric short-term system memory overlaps somewhat with the previously mentioned parametric short-term personal memory in terms of technical approach.
The difference lies in their focus: parametric short-term personal memory is more concerned with improving the processing of individual input data, while parametric short-term system memory focuses on optimizing the storage and reuse of system-level context during task execution.
The former primarily addresses how to quickly process and adapt to an individual’s input information, whereas the latter aims to reduce inference costs in multi-turn reasoning and enhance the consistency and efficiency of global tasks.

#### 4.2.2 Parametric Memory Structures (Quadrant-VIII)

From the perspective of large language models (LLM) as long-term parametric memory, LLMs are not merely tools that provide immediate responses based on input and output; they can also store and integrate information over long time spans, forming an ever-evolving knowledge system.
LLMs based on the Transformer \[ [153](https://arxiv.org/html/2504.15965v1#bib.bib153 "")\] architecture are capable of memorizing knowledge information, primarily due to the self-attention mechanism in the Transformer-based model and the large-scale parameterized training approach.
By training on vast corpora, LLMs learn extensive world knowledge, language patterns, and solutions to various tasks. Additionally, LLMs can modify, update, or refine the internal knowledge through parameterized knowledge editing, allowing for more precise task handling or responses that better align with user needs.
MemoryLLM \[ [148](https://arxiv.org/html/2504.15965v1#bib.bib148 "")\] has the ability to self-update and inject memory with new knowledge, effectively integrating new information and demonstrating excellent model editing performance and long-term information retention capabilities.
WISE \[ [149](https://arxiv.org/html/2504.15965v1#bib.bib149 "")\] is a lifelong editing framework for large language models that employs a dual-parametric memory design, with the main memory preserving pretrained knowledge and the side memory storing edited information.
It leverages a routing mechanism to dynamically access the appropriate memory during inference and uses knowledge sharding to distribute and integrate edits efficiently, ensuring reliability, generalization, and locality throughout continual updates.
The core function of parameterized knowledge editing \[ [92](https://arxiv.org/html/2504.15965v1#bib.bib92 "")\] is to enable large language models (LLMs) with dynamic and flexible knowledge updating capabilities, allowing them to respond to constantly changing task requirements, domain knowledge, and new information from the real world.
This allows LLMs to remain efficient and accurate across various application scenarios and be customized and optimized according to user or environmental needs.

### 4.3 Discussion

In this section, we describe system memory and related work from the perspectives of non-parametric and parametric approaches.
Non-parametric short-term system memory can enhance the reasoning and planning abilities for current tasks, while non-parametric long-term system memory enables the reuse of successful experiences and the self-reflection based on historical experience, facilitating the evolution of LLM-driven AI system capabilities.
On the other hand, parametric short-term system memory can reduce costs and improve efficiency in large language model inference, and long-term parametric system memory can store and integrate information over long time spans, forming a continuously evolving knowledge system.
In the next section, we will summarize the issues and challenges in memory research in the era of large language models and point out potential future directions for development.

## 5 Open Problems and Future Directions

Although substantial progress has been made in current memory research across the three dimensions—object, form, and time—as well as within the eight corresponding quadrants, numerous open issues and challenges remain.
Building upon recent advancements and recognizing existing limitations, we outline the following promising directions for future research:

##### From Unimodal Memory to Multimodal Memory

In the era of large language models, LLM-driven AI systems are gradually expanding from being able to process only a single type of data (such as text) to handle multiple types of data simultaneously (such as text, images, audio, video, and even sensor data).
This transition enhances perceptual capabilities and enables robust performance in complex real-world tasks.
For example, in the medical field, by combining text (medical records), images (medical imaging), and speech (doctor-patient conversations), AI systems can more accurately understand and diagnose medical conditions.
Multimodal memory systems can integrate information from different sensory channels into a unified understanding, thereby approaching human cognitive processes more closely.
Moreover, the expansion of multimodal memory also opens up possibilities for more personalized and interactive AI applications \[ [154](https://arxiv.org/html/2504.15965v1#bib.bib154 "")\].
For instance, personal AI assistants can not only communicate with users through text but also interpret users’ emotions by recognizing facial expressions, voice intonations, or body language, thus providing more personalized and empathetic responses.

##### From Static Memory to Stream Memory

Static memory can be viewed as a batch-processing approach to memory storage. It accumulates information or experiences in discrete batches, typically processing, storing, and retrieving them at specific intervals or predetermined points in time. As an offline memory model, static memory emphasizes the systematic organization and consolidation of large volumes of information, making it well-suited for long-term knowledge retention and structured learning.
In contrast, stream memory operates in a continuous, real-time manner. Analogous to data stream processing, it handles information as it arrives, prioritizing immediacy and adaptability. As an online or real-time memory model, stream memory focuses on the dynamic updating of information and rapid responsiveness to evolving contexts.
These two memory paradigms are not mutually exclusive and often function complementarily: while static memory supports the accumulation of stable, long-term knowledge, stream memory enables agile adaptation to ongoing tasks and real-time information demands.

##### From Specific Memory to Comprehensive Memory

The human memory system comprises multiple interconnected subsystems—such as sensory memory, working memory, explicit memory, and implicit memory—each fulfilling distinct functions and contributing to the overall cognitive process.
In the context of large language models (LLMs), current memory architectures often concentrate on narrow or task-specific components, such as short-term memory for immediate inference or domain-specific knowledge storage.
While such targeted memory mechanisms can enhance performance in specific scenarios, their limited scope constrains the system’s overall flexibility, generalization, and adaptability.
Looking forward, the development of comprehensive and collaborative memory systems is essential. These systems should integrate diverse memory types and support efficient interaction, self-organization, and continual updating, enabling LLMs to manage increasingly complex and dynamic tasks.
By more closely emulating the multi-layered, multi-dimensional, and adaptive characteristics of human memory, such architectures have the potential to significantly advance the general intelligence and autonomy of LLM-based AI systems.

##### From Exclusive Memory to Shared Memory

At present, the memory of each LLM-driven AI system operates independently, typically confined to a specific domain and tailored to processing isolated tasks or environments.
However, as AI technologies continue to evolve, memory systems are expected to become increasingly interconnected, transcending domain boundaries and enabling enhanced collaboration among models.
For instance, a large language model specialized in the medical domain could share its memory or knowledge base with another model focused on finance, facilitating cross-domain knowledge transfer and cooperative task solving.
Such a shared memory paradigm would not only improve the efficiency and adaptability of individual systems but also empower multiple LLMs to dynamically access and leverage one another’s domain-specific expertise.
This shift toward collaborative memory architectures could give rise to a more intelligent, resource-efficient network of AI systems capable of addressing complex, multi-domain challenges.
Ultimately, shared memory is poised to broaden the scope of AI applications and accelerate its integration into increasingly diverse and demanding real-world scenarios.

##### From Individual Privacy to Collective Privacy

With the increasing prevalence of data sharing in the AI era, the focus of privacy protection is gradually shifting from the traditional notion of individual privacy to the broader and emerging concept of collective privacy.
Conventional privacy frameworks primarily aim to safeguard personal data, preventing unauthorized access, leakage, or misuse of individually identifiable information.
However, in the context of large language models, individual data is often aggregated into group-level datasets for large-scale analysis and prediction.
Collective privacy concerns the protection of the rights and interests of groups or communities whose data is used collectively, raising questions about how to prevent misuse, profiling, or excessive surveillance at the group level.
As memory systems in AI become more advanced and interconnected, ensuring collective privacy will emerge as a critical challenge.
Addressing this issue will require innovative techniques that can effectively balance the trade-off between data utility and privacy preservation \[ [155](https://arxiv.org/html/2504.15965v1#bib.bib155 "")\].

##### From Rule-Based Evolution to Automated Evolution

Traditional AI systems evolve by reflecting on past experiences—such as reusing successful strategies—based on accumulated knowledge and historical data.
However, this evolutionary process often depends on manually crafted rules and heuristic adjustments to enable such self-reflection.
While rule-based evolution can be effective, it inherently limits the system’s flexibility, scalability, and efficiency, with the quality and generalizability of the rules directly constraining the system’s adaptive capabilities.
Looking ahead, AI systems are expected to achieve automated evolution, dynamically adjusting and optimizing themselves by leveraging both personal and system-level memories in response to changing data and environmental contexts.
Such systems will be capable of autonomously identifying performance bottlenecks and initiating self-improvement without relying on explicit, human-defined rules.
This transition toward self-directed adaptation will significantly enhance system responsiveness, reduce the need for human intervention, and enable a more intelligent, dynamic, and continuously self-evolving paradigm.

## 6 Conclusion

Memory plays a pivotal role in the advancement of AI systems in the era of large language models (LLMs). It not only shapes the degree of personalization in AI behavior but also influences key capabilities such as adaptability, reasoning, planning, and self-evolution.
This article systematically examines the relationship between human memory and memory mechanisms in LLM-driven AI systems, exploring how principles of human cognition can inspire the design of more efficient and flexible memory architectures.
We begin by analyzing various categories of human memory—including perceptual memory, working memory, and long-term memory—and compare them with existing memory models in AI. Building upon this, we propose an eight-quadrant classification framework grounded in three dimensions: object, form, and time, offering a theoretical foundation for the construction of multi-level and comprehensive memory systems.
Furthermore, we review the current state of memory development in AI from both personal memory and system memory perspectives.
Finally, we identify key open challenges in contemporary AI memory design and outline promising directions for future research in the LLM era.
We believe that, with continued technological progress, AI systems will increasingly adopt more dynamic, adaptive, and intelligent memory architectures, thereby enabling more robust applications across complex, real-world tasks.

</details>

<details>
<summary>⚠️ Error scraping https://aclanthology.org/2024.findings-emnlp.138.pdf after 3 attempts: Request Timeout: Failed to make POST request as the request timed out. Request timed out - No additional error details provided.</summary>

⚠️ Error scraping https://aclanthology.org/2024.findings-emnlp.138.pdf after 3 attempts: Request Timeout: Failed to make POST request as the request timed out. Request timed out - No additional error details provided.

</details>

<details>
<summary>Despite the significant advancement in large language models (LLMs), LLMs often need help with long contexts, especially where information is spread across the complete text. LLMs can now handle long stretches of text as input, but they still face the “lost in the middle” problem. The ability of LLMs to accurately find and use information within that context weakens as the relevant information gets further away from the beginning or end. In other words, they tend to focus on the information at the beginning and end, neglecting what’s sandwiched in between.</summary>

Despite the significant advancement in large language models (LLMs), LLMs often need help with long contexts, especially where information is spread across the complete text. LLMs can now handle long stretches of text as input, but they still face the “lost in the middle” problem. The ability of LLMs to accurately find and use information within that context weakens as the relevant information gets further away from the beginning or end. In other words, they tend to focus on the information at the beginning and end, neglecting what’s sandwiched in between.

Researchers from the University of Washington, MIT, Google Cloud AI Research, and Google collaborated to address the “lost-in-the-middle” issue. Despite being trained to handle large input contexts, LLMs exhibit an inherent attention bias that results in higher attention to tokens at the beginning and end of the input. This leads to reduced accuracy when critical information is situated in the middle. The study aims to mitigate the positional bias by allowing the model to attend to contexts based on their relevance, regardless of their position within the input sequence.

Current methods to tackle the lost-in-the-middle problem often involve re-ranking the relevance of documents and repositioning the most pertinent ones at the beginning or end of the input sequence. However, these methods usually require additional supervision or fine-tuning and do not fundamentally address the LLMs’ ability to utilize mid-sequence information effectively. To overcome this limitation, the researchers propose a novel calibration mechanism called “found-in-the-middle.”

The researchers first establish that the lost-in-the-middle issue is linked to a U-shaped attention bias. The inherent bias persists even when the order of documents is randomized. To verify their hypothesis, the authors intervene by adjusting the attention distribution to reflect relevance rather than position. They quantify this positional bias by measuring changes in attention as they vary the position of a fixed context within the input prompt.

The proposed “found-in-the-middle” mechanism disentangles positional bias from the attention scores, enabling a more accurate reflection of the documents’ relevance. This calibration involves estimating the bias and adjusting attention scores accordingly. Experiments demonstrate that the calibrated attention significantly improves the model’s ability to locate relevant information within long contexts, leading to better performance in retrieval-augmented generation (RAG) tasks.

The researchers operationalize this calibration mechanism to improve overall RAG performance. The attention calibration method consistently outperforms uncalibrated models across various tasks and models, including those with different context window lengths. The approach yields improvements of up to 15 percentage points on the NaturalQuestions dataset. Additionally, combining attention calibration with existing reordering methods further enhances model performance, demonstrating the effectiveness and complementarity of the proposed solution.

In conclusion, the proposed mechanism effectively identifies and addresses the lost-in-the-middle phenomenon by linking it to intrinsic positional attention bias in LLMs. The found-in-the-middle mechanism successfully mitigates this bias, enabling the models to attend to relevant contexts more faithfully and significantly improving performance in long-context utilization tasks. This advancement opens new ways for enhancing LLM attention mechanisms and their application in various user-facing applications.

</details>


## Code Sources

<details>
<summary>Repository analysis for https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-03-own-your-context-window.md</summary>

# Repository analysis for https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-03-own-your-context-window.md

## Summary
Repository: humanlayer/12-factor-agents
File: factor-03-own-your-context-window.md
Lines: 260

Estimated tokens: 2.5k

## File tree
```Directory structure:
└── factor-03-own-your-context-window.md

```

## Extracted content
================================================
FILE: content/factor-03-own-your-context-window.md
================================================
[← Back to README](https://github.com/humanlayer/12-factor-agents/blob/main/README.md)

### 3. Own your context window

You don't necessarily need to use standard message-based formats for conveying context to an LLM.

> #### At any given point, your input to an LLM in an agent is "here's what's happened so far, what's the next step"

<!-- todo syntax highlighting -->
<!-- ![130-own-your-context-building](https://github.com/humanlayer/12-factor-agents/blob/main/img/130-own-your-context-building.png) -->

Everything is context engineering. [LLMs are stateless functions](https://thedataexchange.media/baml-revolution-in-ai-engineering/) that turn inputs into outputs. To get the best outputs, you need to give them the best inputs.

Creating great context means:

- The prompt and instructions you give to the model
- Any documents or external data you retrieve (e.g. RAG)
- Any past state, tool calls, results, or other history 
- Any past messages or events from related but separate histories/conversations (Memory)
- Instructions about what sorts of structured data to output

![image](https://github.com/user-attachments/assets/0f1f193f-8e94-4044-a276-576bd7764fd0)


### on context engineering

This guide is all about getting as much as possible out of today's models. Notably not mentioned are:

- Changes to models parameters like temperature, top_p, frequency_penalty, presence_penalty, etc.
- Training your own completion or embedding models
- Fine-tuning existing models

Again, I don't know what's the best way to hand context to an LLM, but I know you want the flexibility to be able to try EVERYTHING.

#### Standard vs Custom Context Formats

Most LLM clients use a standard message-based format like this:

```yaml
[
  {
    "role": "system",
    "content": "You are a helpful assistant..."
  },
  {
    "role": "user",
    "content": "Can you deploy the backend?"
  },
  {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {
        "id": "1",
        "name": "list_git_tags",
        "arguments": "{}"
      }
    ]
  },
  {
    "role": "tool",
    "name": "list_git_tags",
    "content": "{\"tags\": [{\"name\": \"v1.2.3\", \"commit\": \"abc123\", \"date\": \"2024-03-15T10:00:00Z\"}, {\"name\": \"v1.2.2\", \"commit\": \"def456\", \"date\": \"2024-03-14T15:30:00Z\"}, {\"name\": \"v1.2.1\", \"commit\": \"abe033d\", \"date\": \"2024-03-13T09:15:00Z\"}]}",
    "tool_call_id": "1"
  }
]
```

While this works great for most use cases, if you want to really get THE MOST out of today's LLMs, you need to get your context into the LLM in the most token- and attention-efficient way you can.

As an alternative to the standard message-based format, you can build your own context format that's optimized for your use case. For example, you can use custom objects and pack/spread them into one or more user, system, assistant, or tool messages as makes sense.

Here's an example of putting the whole context window into a single user message:
```yaml

[
  {
    "role": "system",
    "content": "You are a helpful assistant..."
  },
  {
    "role": "user",
    "content": |
            Here's everything that happened so far:
        
        <slack_message>
            From: @alex
            Channel: #deployments
            Text: Can you deploy the backend?
        </slack_message>
        
        <list_git_tags>
            intent: "list_git_tags"
        </list_git_tags>
        
        <list_git_tags_result>
            tags:
              - name: "v1.2.3"
                commit: "abc123"
                date: "2024-03-15T10:00:00Z"
              - name: "v1.2.2"
                commit: "def456"
                date: "2024-03-14T15:30:00Z"
              - name: "v1.2.1"
                commit: "ghi789"
                date: "2024-03-13T09:15:00Z"
        </list_git_tags_result>
        
        what's the next step?
    }
]
```

The model may infer that you're asking it `what's the next step` by the tool schemas you supply, but it never hurts to roll it into your prompt template.

### code example

We can build this with something like: 

```python

class Thread:
  events: List[Event]

class Event:
  # could just use string, or could be explicit - up to you
  type: Literal["list_git_tags", "deploy_backend", "deploy_frontend", "request_more_information", "done_for_now", "list_git_tags_result", "deploy_backend_result", "deploy_frontend_result", "request_more_information_result", "done_for_now_result", "error"]
  data: ListGitTags | DeployBackend | DeployFrontend | RequestMoreInformation |  
        ListGitTagsResult | DeployBackendResult | DeployFrontendResult | RequestMoreInformationResult | string

def event_to_prompt(event: Event) -> str:
    data = event.data if isinstance(event.data, str) \
           else stringifyToYaml(event.data)

    return f"<{event.type}>\n{data}\n</{event.type}>"


def thread_to_prompt(thread: Thread) -> str:
  return '\n\n'.join(event_to_prompt(event) for event in thread.events)
```

#### Example Context Windows

Here's how context windows might look with this approach:

**Initial Slack Request:**
```xml
<slack_message>
    From: @alex
    Channel: #deployments
    Text: Can you deploy the latest backend to production?
</slack_message>
```

**After Listing Git Tags:**
```xml
<slack_message>
    From: @alex
    Channel: #deployments
    Text: Can you deploy the latest backend to production?
    Thread: []
</slack_message>

<list_git_tags>
    intent: "list_git_tags"
</list_git_tags>

<list_git_tags_result>
    tags:
      - name: "v1.2.3"
        commit: "abc123"
        date: "2024-03-15T10:00:00Z"
      - name: "v1.2.2"
        commit: "def456"
        date: "2024-03-14T15:30:00Z"
      - name: "v1.2.1"
        commit: "ghi789"
        date: "2024-03-13T09:15:00Z"
</list_git_tags_result>
```

**After Error and Recovery:**
```xml
<slack_message>
    From: @alex
    Channel: #deployments
    Text: Can you deploy the latest backend to production?
    Thread: []
</slack_message>

<deploy_backend>
    intent: "deploy_backend"
    tag: "v1.2.3"
    environment: "production"
</deploy_backend>

<error>
    error running deploy_backend: Failed to connect to deployment service
</error>

<request_more_information>
    intent: "request_more_information_from_human"
    question: "I had trouble connecting to the deployment service, can you provide more details and/or check on the status of the service?"
</request_more_information>

<human_response>
    data:
      response: "I'm not sure what's going on, can you check on the status of the latest workflow?"
</human_response>
```

From here your next step might be: 

```python
nextStep = await determine_next_step(thread_to_prompt(thread))
```

```python
{
  "intent": "get_workflow_status",
  "workflow_name": "tag_push_prod.yaml",
}
```

The XML-style format is just one example - the point is you can build your own format that makes sense for your application. You'll get better quality if you have the flexibility to experiment with different context structures and what you store vs. what you pass to the LLM. 

Key benefits of owning your context window:

1. **Information Density**: Structure information in ways that maximize the LLM's understanding
2. **Error Handling**: Include error information in a format that helps the LLM recover. Consider hiding errors and failed calls from context window once they are resolved.
3. **Safety**: Control what information gets passed to the LLM, filtering out sensitive data
4. **Flexibility**: Adapt the format as you learn what works best for your use case
5. **Token Efficiency**: Optimize context format for token efficiency and LLM understanding

Context includes: prompts, instructions, RAG documents, history, tool calls, memory


Remember: The context window is your primary interface with the LLM. Taking control of how you structure and present information can dramatically improve your agent's performance.

Example - information density - same message, fewer tokens:

![Loom Screenshot 2025-04-22 at 09 00 56](https://github.com/user-attachments/assets/5cf041c6-72da-4943-be8a-99c73162b12a)


### Don't take it from me

About 2 months after 12-factor agents was published, context engineering started to become a pretty popular term.

<a href="https://x.com/karpathy/status/1937902205765607626"><img width="378" alt="Screenshot 2025-06-25 at 4 11 45 PM" src="https://github.com/user-attachments/assets/97e6e667-c35f-4855-8233-af40f05d6bce" /></a> <a href="https://x.com/tobi/status/1935533422589399127"><img width="378" alt="Screenshot 2025-06-25 at 4 12 59 PM" src="https://github.com/user-attachments/assets/7e6f5738-0d38-4910-82d1-7f5785b82b99" /></a>

There's also a quite good [Context Engineering Cheat Sheet](https://x.com/lenadroid/status/1943685060785524824) from [@lenadroid](https://x.com/lenadroid) from July 2025.

<a href="https://x.com/lenadroid/status/1943685060785524824"><img width="256" alt="image" src="https://github.com/user-attachments/assets/cac88aa3-8faf-440b-9736-cab95a9de477" /></a>



Recurring theme here: I don't know what's the best approach, but I know you want the flexibility to be able to try EVERYTHING.


[← Own Your Prompts](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-02-own-your-prompts.md) | [Tools Are Structured Outputs →](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-04-tools-are-structured-outputs.md)

</details>


## YouTube Video Transcripts

_No YouTube video transcripts found._


## Additional Sources Scraped

<details>
<summary>context-engineering-a-guide-with-examples-datacamp</summary>

You may be a master prompt engineer, but as the conversation goes on, your chatbot often forgets the earliest and most important pieces of your instructions, your code assistant loses track of project architecture, and your RAG tool can’t connect information across complex documents and domains.

As AI use cases grow more complex, writing a clever prompt is just one small part of a much larger challenge: **context engineerin** **g**.

In this tutorial, I will explain what context engineering is, how it works, when to use it instead of regular prompt engineering, and the practical techniques that make AI systems smarter and more context-aware.

## What Is Context Engineering?

Context engineering is the practice of designing systems that decide what information an AI model sees before it generates a response.

Even though the term is new, the principles behind context engineering have existed for quite a while. This new abstraction allows us to reason about the most and ever-present issue of designing the information flow that goes in and out of AI systems.

Instead of writing perfect prompts for individual requests, you create systems that gather relevant details from multiple sources and organize them within the model’s context window. This means your system pulls together conversation history, user data, external documents, and available tools, then formats them so the model can work with them.https://media.datacamp.com/cms/ad_4nxcdalepxi_aheoksazdeushsfbbtychlv2ocecq4yyglsbeyz9je2dq-ifk2gne_dx8v-4gun0oedhjo12iviw8hgndp0_ibd0y0prfbb4vhaq5r5h3y2wn_vkhrp-qqqxw3d-9bg.png

Source: [12-factor-agents](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-03-own-your-context-window.md)

This approach requires managing several different types of information that make up the full context:

- System instructions that set behavior and rules
- Conversation history and user preferences
- Retrieved information from documents or databases
- Available tools and their definitions
- Structured output formats and schemas
- Real-time data and external API responses

The main challenge is working within context window limitations while maintaining coherent conversations over time. Your system needs to decide what’s most relevant for each request, which usually means building retrieval systems that find the right details when you need them.

This involves creating memory systems that track both short-term conversation flow and long-term user preferences, plus removing outdated information to make space for current needs.

The real benefit comes when different types of context work together to create AI systems that feel more intelligent and aware. When your AI assistant can reference previous conversations, access your calendar, and understand your communication style all at once, interactions stop feeling repetitive and start feeling like you’re working with something that remembers you.

## Context Engineering vs. Prompt Engineering

If you ask ChatGPT to “write a professional email,” that’s prompt engineering — you’re writing instructions for a single task. But if you’re building a customer service bot that needs to remember previous tickets, access user account details, and maintain conversation history across multiple interactions, that’s context engineering.

Andrej Karpathy [explains this well](https://x.com/karpathy/status/1937902205765607626):

> ### **People associate prompts with short task descriptions you’d give an LLM in your day-to-day use. When in every industrial-strength LLM app, context engineering is the delicate art and science of filling the context window with just the right information for the next step.**
>
> Andrej Karpathy

Most AI applications use both prompt engineering and context engineering. You still need well-written prompts within your context engineering system. The difference is that those prompts now work with carefully managed background information instead of starting fresh each time.

|     |     |
| --- | --- |
| Approach | Best Used For |
| **Prompt Engineering** | One-off tasks, content generation, format-specific outputs |
| **Context Engineering** | Conversational AI, document analysis tools, coding assistants |
| **Both Together** | Production AI applications that need consistent, reliable performance |

## Context Engineering in Practice

Context engineering moves from theory to reality when you start [building AI applications](https://www.datacamp.com/tracks/developing-ai-applications) that need to work with complex, interconnected information. Consider a customer service bot that needs to access previous support tickets, check account status, and reference product documentation, all while maintaining a helpful conversation tone. This is where traditional prompting breaks down and context engineering becomes necessary.

### RAG systems

Context engineering arguably started with [retrieval augmented generation (RAG)](https://www.datacamp.com/courses/retrieval-augmented-generation-rag-with-langchain) systems. RAG was one of the first techniques that let you introduce LLMs to information that wasn’t part of their original training data.

RAG systems use advanced context engineering techniques to organize and present information more effectively. They break documents into meaningful pieces, rank information by relevance, and fit the most useful details within token limits.

Before RAG, if you wanted an AI to answer questions about your company’s internal documents, you’d have to retrain or [fine-tune](https://www.datacamp.com/tutorial/fine-tuning-large-language-models) the entire model. RAG changed this by building systems that could search through your documents, find relevant chunks, and include them in the context window alongside your question.

This meant LLMs could suddenly analyze multiple documents and sources to answer complex questions that would normally require a human to read through hundreds of pages.

### AI agents

RAG systems opened the door to external information, but AI agents took this further by making context dynamic and responsive. Instead of just retrieving static documents, agents use external tools during conversations.

The AI decides which tool will best solve the current problem. An agent can start a conversation, realize it needs current stock data, call a financial API, and then use that fresh information to continue the conversation.

The decreasing cost of LLM tokens also made multi-agent systems possible. Instead of cramming everything into a single model’s context window, you can have specialized agents that handle different aspects of a problem and share information between them via protocols like [A2A](https://www.datacamp.com/blog/a2a-agent2agent) or [MCP](https://www.datacamp.com/tutorial/mcp-model-context-protocol).

### AI coding assistants

AI coding assistants—like [Cursor](https://www.datacamp.com/tutorial/cursor-ai-code-editor) or [Windsurf](https://www.datacamp.com/tutorial/windsurf-ai-agentic-code-editor)—represent one of the most advanced applications of context engineering because they combine both RAG and agent principles while working with highly structured, interconnected information.

These systems need to understand not just individual files, but entire project architectures, dependencies between modules, and coding patterns across your codebase.

When you ask a coding assistant to refactor a function, it needs context about where that function is used, what data types it expects, and how changes might affect other parts of your project.

Context engineering becomes critical here because code has relationships that span multiple files and even multiple repositories. A good coding assistant maintains context about your project structure, recent changes you’ve made, your coding style, and the frameworks you’re using.

This is why tools like Cursor work better the longer you use them in a project. They build up context about your specific codebase and can make more relevant suggestions based on your patterns and preferences.

## Context Failures And Techniques to Mitigate Them

As you read through the article, you may think that context engineering is unnecessary or will be unnecessary in the near future as context windows of frontier models continue to grow. This would be a natural assumption because if the context is large enough, you could throw everything into a prompt (tools, documents, instructions, and more) and let the model take care of the rest.

However, [this excellent article](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html) written by Drew Breunig shows four surprising ways the context can get out of hand, even when the model in question supports 1 million token context windows. In this section, I will quickly describe the issues described by Drew Breunig and the context engineering patterns that solve them—I strongly recommend reading Breunig’s article for more details.

### Context poisoning

Context poisoning happens when a [hallucination](https://www.datacamp.com/blog/ai-hallucination) or error ends up in your AI system’s context and then gets referenced over and over in future responses. The DeepMind team identified this problem in their [Gemini 2.5 technical report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf) while building a Pokémon-playing agent. When the agent would sometimes hallucinate about the game state, this false information would poison the “goals” section of its context, causing the agent to develop nonsense strategies and pursue impossible objectives for a long time.

This problem becomes really bad in agent workflows where information builds up. Once a poisoned context gets established, it can take forever to fix because the model keeps referencing the false information as if it were true.

The best fix is context validation and quarantine. You can isolate different types of context in separate threads and validate information before it gets added to long-term memory. Context quarantine means starting fresh threads when you detect potential poisoning, which prevents bad information from spreading to future interactions.

### Context distraction

Context distraction happens when your context grows so large that the model starts focusing too much on the accumulated history instead of using what it learned during training. The Gemini agent playing Pokémon showed this — once the context grew beyond 100,000 tokens, the agent began repeating actions from its vast history rather than developing new strategies.

A [Databricks study](https://www.databricks.com/blog/long-context-rag-performance-llms) (very interesting study; definitely worth a read) found that model correctness began dropping around 32,000 tokens for [Llama 3.1 405b](https://www.datacamp.com/blog/llama-3-1-405b-meta-ai), with smaller models hitting their limit much earlier. This means models start making mistakes long before their context windows are actually full, which makes you wonder about the real value of very large context windows for complex reasoning tasks.https://media.datacamp.com/cms/ad_4nxfhixqlsmlfsmanfddtu14_x440vudfotpsszlmym6ueghlfz-d2p39fwa8wpordmnq6xh9v2vneamqlyijvkcyl8srrvg3qioxe42tdzdtwlc8dxjdfk9p8amknimgcis0e8lh1a.png

Source: [Databricks](https://www.databricks.com/blog/long-context-rag-performance-llms)

The best approach is context summarization. Instead of letting context grow forever, you can compress accumulated information into shorter summaries that keep important details while removing redundant history. This helps when you hit the distraction ceiling — you can summarize the conversation so far and start fresh while keeping things consistent.

### Context confusion

Context confusion happens when you include extra information in your context that the model uses to generate bad responses, even when that information isn’t relevant to the current task. The [Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) shows this — every model performs worse when given more than one tool, and models will sometimes call tools that have nothing to do with the task.

The problem gets worse with smaller models and more tools. A recent study found that a quantized Llama 3.1 8b failed on the GeoEngine benchmark when given all 46 available tools, even though the context was well within the 16k window limit. But when researchers gave the same model only 19 tools, it worked fine.

The solution is tool loadout management using RAG techniques. [Research by Tiantian Gan and Qiyao Sun](https://arxiv.org/abs/2505.03275) showed that applying RAG to tool descriptions can really improve performance. By storing tool descriptions in a [vector database](https://www.datacamp.com/blog/the-top-5-vector-databases), you can select only the most relevant tools for each task. Their study found that keeping tool selections under 30 tools gave three times better tool selection accuracy and much shorter prompts.

### Context clash

Context clash happens when you gather information and tools in your context that directly conflict with other information already there. A Microsoft and Salesforce study showed this by taking benchmark prompts and “sharding” their information across multiple conversational turns instead of providing everything at once. The results were huge — an average performance drop of 39%, with [OpenAI’s o3](https://www.datacamp.com/blog/o3-openai) model dropping from 98.1 to 64.1.https://media.datacamp.com/cms/ad_4nxep3if9fetk_gcocfoo2qoqddl3w7nss64iqgaqrya-yqkzqt8v4gqxbw97yz8mhotyrxs7dddjy5kq1yp5k7awjaob2hc8zerxrel6ds-wj4uszdk6pej6io4uvwy4d4jvpwkz.png

Source: [Laban et. al, 2025](https://arxiv.org/pdf/2505.06120)

The problem happens because when information comes in stages, the assembled context contains early attempts by the model to answer questions before it has all the information. These incorrect early answers stay in the context and affect the model when it generates final responses.

The best fixes are context pruning and offloading. Context pruning means removing outdated or conflicting information as new details arrive. Context offloading, like [Anthropic’s “think” tool](https://docs.anthropic.com/en/docs/build-with-claude/tool-use), gives models a separate workspace to process information without cluttering the main context. This scratchpad approach can give up to 54% improvement in specialized agent benchmarks by preventing internal contradictions from messing up reasoning.

## Conclusion

Context engineering represents the next phase of AI development, where the focus shifts from crafting perfect prompts to building systems that manage information flow over time. The ability to maintain relevant context across multiple interactions determines whether your AI feels intelligent or just gives good one-off responses.

The techniques covered in this tutorial — from RAG systems to context validation and tool management — are already being used in production systems that handle millions of users.

If you’re building anything more complex than a simple content generator, you’ll likely need context engineering techniques. The good news is that you can start small with basic RAG implementations and gradually add more sophisticated memory and tool management as your needs grow.

## FAQs

### When should I start using context engineering instead of just prompts?

**Start using context engineering when your AI needs to remember things between conversations, work with multiple information sources, or maintain long-running tasks. If you're building anything more complex than a simple content generator, you'll likely need these techniques.**

### What's the main difference between context engineering and prompt engineering?

**Prompt engineering focuses on writing instructions for single tasks, while context engineering designs systems that manage information flow across multiple interactions. Context engineering builds memory and retrieval systems, while prompt engineering crafts individual requests.**

### Can I use larger context windows instead of context engineering?

**Larger context windows don't solve the core problems. Research shows model performance drops around 32,000 tokens, even with million-token windows, due to context distraction and confusion. You still need techniques like summarization, pruning, and smart information selection regardless of context size.**

### Why do AI models perform worse when I give them more tools or information?

**This is called context confusion—models get distracted by irrelevant information and may use tools that don't match the task. The solution is tool loadout management: use RAG techniques to select only the most relevant tools for each specific task, keeping selections under 30 tools.**

</details>

<details>
<summary>context-engineering-guide-by-elvis-ai-newsletter</summary>

A few years ago, many, even top AI researchers, claimed that prompt engineering would be dead by now.

Obviously, they were very wrong, and in fact, prompt engineering is now even more important than ever. It is so important that it is now being rebranded as _**context engineering**_.

Yes, another fancy term to describe the important process of tuning the instructions and relevant context that an LLM needs to perform its tasks effectively.

Much has been written already about context engineering ( [Ankur Goyal](https://x.com/ankrgyl/status/1913766591910842619), [Walden Yan](https://cognition.ai/blog/dont-build-multi-agents), [Tobi Lutke](https://x.com/tobi/status/1935533422589399127), and [Andrej Karpathy](https://x.com/karpathy/status/1937902205765607626)), but I wanted to write about my thoughts on the topic and show you a concrete step-by-step guide putting context engineering into action in developing an AI agent workflow.

I am not entirely sure who coined context engineering, but we will build on this figure from [Dex Horthy](https://x.com/dexhorthy/status/1933283008863482067) that briefly explains a bit about what context engineering is.

[https://substackcdn.com/image/fetch/$s_!2kzL!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3a0a3a15-4327-4094-ad24-19ed97e184cd_680x383.jpeg](https://substackcdn.com/image/fetch/$s_!2kzL!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3a0a3a15-4327-4094-ad24-19ed97e184cd_680x383.jpeg)

I like the term context engineering as it feels like a broader term that better explains most of the work that goes into prompt engineering, including other related tasks.

The doubt about prompt engineering being a serious skill is that many confuse it with blind prompting (a short task description you use in an LLM like ChatGPT). In blind prompting, you are just asking the system a question. In prompt engineering, you have to think more carefully about the context and structure of your prompt. Perhaps it should have been called context engineering from early on.

Context engineering is the next phase, where you architect the full context, which in many cases requires going beyond simple prompting and into more rigorous methods to obtain, enhance, and optimize knowledge for the system.

From a developer's point of view, context engineering involves an iterative process to optimize instructions and the context you provide an LLM to achieve a desired result. This includes having formal processes (e.g., eval pipelines) to measure whether your tactics are working.

Given the fast evolution of the AI field, I suggest a broader definition of context engineering: _**the process of designing and optimizing instructions and relevant context for the LLMs and advanced AI models to perform their tasks effectively.**_ This encompasses not only text-based LLMs but also optimizing context for multimodal models, which are becoming more widespread. This can include all the prompt engineering efforts and the related processes such as:

- Designing and managing prompt chains (when applicable)

- Tuning instructions/system prompts

- Managing dynamic elements of the prompt (e.g., user inputs, date/time, etc.)

- Searching and preparing relevant knowledge (i.e., RAG)

- Query augmentation

- Tool definitions and instructions (in the case of agentic systems)

- Preparing and optimizing few-shot demonstrations

- Structuring inputs and outputs (e.g., delimiters, JSON schema)

- Short-term memory (i.e., managing state/historical context) and long-term memory (e.g., retrieving relevant knowledge from a vector store)

- And the many other tricks that are useful to optimize the LLM system prompt to achieve the desired tasks.


In other words, what you are trying to achieve in context engineering is optimizing the information you are providing in the context window of the LLM. This also means filtering out noisy information, which is a science on its own, as it requires systematically measuring the performance of the LLM.

Everyone is writing about context engineering, but here we are going to walk you through a concrete example of what context engineering looks like when building AI agents.

* * *

## Context Engineering in Action

Let’s look at a concrete example of some recent context engineering work I did for a multi-agent deep research application I built for personal use.

I built the agentic workflow inside of n8n, but the tool doesn’t matter. The complete agent architecture I built looks like the following:

[https://substackcdn.com/image/fetch/$s_!uSvA!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc937706e-a25f-427e-81ec-002606966b2a_2822x958.png](https://substackcdn.com/image/fetch/$s_!uSvA!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc937706e-a25f-427e-81ec-002606966b2a_2822x958.png)

The Search Planner agent in my workflow is in charge of generating a search plan based on the user query.

Below is the system prompt I have put together for this subagent:

```
You are an expert research planner. Your task is to break down a complex research query (delimited by <user_query></user_query>) into specific search subtasks, each focusing on a different aspect or source type.

The current date and time is: {{ $now.toISO() }}

For each subtask, provide:
1. A unique string ID for the subtask (e.g., 'subtask_1', 'news_update')
2. A specific search query that focuses on one aspect of the main query
3. The source type to search (web, news, academic, specialized)
4. Time period relevance (today, last week, recent, past_year, all_time)
5. Domain focus if applicable (technology, science, health, etc.)
6. Priority level (1-highest to 5-lowest)

All fields (id, query, source_type, time_period, domain_focus, priority) are required for each subtask, except time_period and domain_focus which can be null if not applicable.

Create 2 subtasks that together will provide comprehensive coverage of the topic. Focus on different aspects, perspectives, or sources of information.

Each substask will include the following information:

id: str
query: str
source_type: str  # e.g., "web", "news", "academic", "specialized"
time_period: Optional[str] = None  # e.g., "today", "last week", "recent", "past_year", "all_time"
domain_focus: Optional[str] = None  # e.g., "technology", "science", "health"
priority: int  # 1 (highest) to 5 (lowest)

After obtaining the above subtasks information, you will add two extra fields. Those correspond to start_date and end_date. Infer this information given the current date and the time_period selected. start_date and end_date should use the format as in the example below:

"start_date": "2024-06-03T06:00:00.000Z",
"end_date": "2024-06-11T05:59:59.999Z",
```

There are many parts to this prompt that require careful consideration about what exact context we are providing the planning agent to carry out the task effectively. As you can see, it’s not just about designing a simple prompt or instruction; this process requires experimentation and providing important context for the model to perform the task optimally.

Let’s break down the problem into core components that are key to effective context engineering.

### **Instructions**

The instruction is the high-level instructions provided to the system to instruct it exactly what to do.

```
You are an expert research planner. Your task is to break down a complex research query (delimited by <user_query></user_query>) into specific search subtasks, each focusing on a different aspect or source type.
```

Many beginners and even experienced AI developers would stop here. Given that I shared the full prompt above, you can appreciate how much more context we need to give the system for it to work as we want. That’s what context engineering is all about; it informs the system more about the problem scope and the specifics of what exactly we desire from it.

### **User Input**

The user input wasn’t shown in the system prompt, but below is an example of how it would look.

```
<user_query> What's the latest dev news from OpenAI? </user_query>
```

Notice the use of the delimiters, which is about structuring the prompt better. This is important to avoid confusion and adds clarity about what the user input is and what things we want the system to generate. Sometimes, the type of information we are inputting is related to what we want the model to output (e.g., the query is the input, and subqueries are the outputs).

### **Structured Inputs and Outputs**

In addition to the high-level instruction and the user input, you might have noticed that I spent a considerable amount of effort on the details related to the subtasks the planning agent needs to produce. Below are the detailed instructions I have provided to the planning agent to create the subtasks given the user query.

```
For each subtask, provide:
1. A unique string ID for the subtask (e.g., 'subtask_1', 'news_update')
2. A specific search query that focuses on one aspect of the main query
3. The source type to search (web, news, academic, specialized)
4. Time period relevance (today, last week, recent, past_year, all_time)
5. Domain focus if applicable (technology, science, health, etc.)
6. Priority level (1-highest to 5-lowest)

All fields (id, query, source_type, time_period, domain_focus, priority) are required for each subtask, except time_period and domain_focus which can be null if not applicable.

Create 2 subtasks that together will provide comprehensive coverage of the topic. Focus on different aspects, perspectives, or sources of information.
```

If you look closely at the instructions above, I have decided to structure a list of the required information I want the planning agent to generate, along with some hints/examples to better help steer the data generation process. This is crucial to give the agent additional context on what is expected. As an example, if you don’t tell it that you want the priority level to be on a scale of 1-5, then the system might prefer to use a scale of 1-10. Again, this context matters a lot!

Next, let’s talk about structured outputs. In order to get consistent outputs from the planning agent, we are also providing some context on the subtask format and field types that we expect. Below is the example we are passing as additional context to the agent. This will provide the agent with hints and clues on what we expect as the output:

```
Each substask will include the following information:

id: str
query: str
source_type: str  # e.g., "web", "news", "academic", "specialized"
time_period: Optional[str] = None  # e.g., "today", "last week", "recent", "past_year", "all_time"
domain_focus: Optional[str] = None  # e.g., "technology", "science", "health"
priority: int  # 1 (highest) to 5 (lowest)
```

In addition to this, inside of n8n, you can also use a tool output parser, which essentially is going to be used to structure the final outputs. The option I am using is providing a JSON example as follows:

```
{
  "subtasks": [
    {
      "id": "openai_latest_news",
      "query": "latest OpenAI announcements and news",
      "source_type": "news",
      "time_period": "recent",
      "domain_focus": "technology",
      "priority": 1,
      "start_date": "2025-06-03T06:00:00.000Z",
      "end_date": "2025-06-11T05:59:59.999Z"
    },
    {
      "id": "openai_official_blog",
      "query": "OpenAI official blog recent posts",
      "source_type": "web",
      "time_period": "recent",
      "domain_focus": "technology",
      "priority": 2,
      "start_date": "2025-06-03T06:00:00.000Z",
      "end_date": "2025-06-11T05:59:59.999Z"
    },
...
}
```
Then the tool will automatically generate the schema from these examples, which in turn allows the system to parse and generate proper structured outputs, as shown in the example below:
```
[
  {
    "action": "parse",
    "response": {
      "output": {
        "subtasks": [
          {
            "id": "subtask_1",
            "query": "OpenAI recent announcements OR news OR updates",
            "source_type": "news",
            "time_period": "recent",
            "domain_focus": "technology",
            "priority": 1,
            "start_date": "2025-06-24T16:35:26.901Z",
            "end_date": "2025-07-01T16:35:26.901Z"
          },
          {
            "id": "subtask_2",
            "query": "OpenAI official blog OR press releases",
            "source_type": "web",
            "time_period": "recent",
            "domain_focus": "technology",
            "priority": 1.2,
            "start_date": "2025-06-24T16:35:26.901Z",
            "end_date": "2025-07-01T16:35:26.901Z"
          }
        ]
      }
    }
  }
]
```
This stuff looks complicated, but many tools today enable structured output functionalities out of the box, so it’s likely you won’t need to implement it yourself. n8n makes this part of context engineering a breeze. This is one underrated aspect of context engineering that I see many AI devs ignore for some reason. Hopefully, context engineering sheds more light on these important techniques. This is a really powerful approach, especially when your agent is getting inconsistent outputs that need to be passed in a special format to the next component in the workflow.
### **Tools**
We are using n8n to build our agent, so it’s easy to put in the context the current date and time. You can do it like so:
```
The current date and time is: {{ $now.toISO() }}
```
This is a simple, handy function that’s being called in n8n, but it’s typical to build this as a dedicated tool that can help with making things more dynamic (i.e., only get the date and time if the query requires it). That’s what context engineering is about. It forces you, the builder, to make concrete decisions about what context to pass and when to pass it to the LLM. This is great because it eliminates assumptions and inaccuracies from your application.
The date and time are important context for the system; otherwise, it tends not to perform well with queries that require knowledge of the current date and time. For instance, if I asked the system to search for the latest dev news from OpenAI that happened last week, it would just guess the dates and time, which would lead to suboptimal queries and, as a result, inaccurate web searches. When the system has the correct date and time, it can better infer date ranges, which are important for the search agent and tools. I added this as part of the context to allow the LLM to generate the date range:
```
After obtaining the above subtasks information, you will add two extra fields. Those correspond to start_date and end_date. Infer this information given the current date and the time_period selected. start_date and end_date should use the format as in the example below:

"start_date": "2024-06-03T06:00:00.000Z",
"end_date": "2024-06-11T05:59:59.999Z",
```
We are focusing on the planning agent of our architecture, so there aren’t too many tools we need to add here. The only other tool that would make sense to add is a retrieval tool that retrieves relevant subtasks given a query. Let’s discuss this idea below.
### **RAG & Memory**
This first version of the deep research application I have built doesn’t require the use of short-term memory, but we have built a version of it that caches subqueries for different user queries. This is useful to achieve some speed-ups/optimizations in the workflow. If a similar query was already used by a user before, it is possible to store those results in a vector store and search over them to avoid the need to create a new set of subqueries for a plan that we already generated and exists in the vector store. Remember, every time you call the LLM APIs, you are increasing latency and costs.
This is clever context engineering as it makes your application more dynamic, cheaper, and efficient. You see, context engineering is not just about optimizing your prompt; it’s about choosing the right context for the goals you are targeting. You can also get more creative about how you are maintaining that vector store and how you pull those existing subtasks into context. Creative and novel context engineering is the moat!
### **States & Historical Context**
We are not showing it in v1 of our deep research agent, but an important part of this project was to optimize the results to generate the final report. In many cases, the agentic system might need to revise all or a subset of the queries, subtasks, and potentially the data it’s pulling from the web search APIs. This means that the system will take multiple shots at the problem and needs access to the previous states and potentially all the historical context of the system.
What does this mean in the context of our use case? In our example, it could be giving the agent access to the state of the subtasks, the revisions (if any), the past results from each agent in the workflow, and whatever other context is necessary to help in the revision phase. For this type of context, what we are passing would depend on what you are optimizing for. Lots of decision-making will happen here. Context engineering isn’t always straightforward, and I think you can start to imagine how many iterations this component will require. This is why I continue to emphasize the importance of other areas, such as evaluation. If you are not measuring all these things, how do you know whether your context engineering efforts are working?

</details>

<details>
<summary>context-engineering-what-it-is-and-techniques-to-consider-ll</summary>

Although the principles behind the term ‘context engineering’ are not new, the wording is a useful abstraction that allows us to reason about the most pressing challenges when it comes to building effective AI agents. So let’s break it down. In this article, I want to cover three things: what we mean by context engineering, how it’s different from “prompt engineering”, and how you can use LlamaIndex and LlamaCloud to design agentic systems that adhere to context engineering principles.

### What is Context Engineering

AI agents require the relevant context for a task, to perform that task in a reasonable way. We’ve known this for a while, but given the speed and fresh nature of everything AI, we are continuously coming up with new abstractions that allow us to reason about best practices and new approaches in easy to understand terms.

[Andrey Karpathy’s post](https://x.com/karpathy/status/1937902205765607626) about this is a great summary:

> People associate prompts with short task descriptions you'd give an LLM in your day-to-day use. When in every industrial-strength LLM app, context engineering is the delicate art and science of filling the context window with just the right information for the next step.

While the term “prompt engineering” focused on the art of providing the right instructions to an LLM at the forefront, although these two terms may seem very similar, “context engineering” puts _a lot_ more focus on filling the context window of an LLM with the most relevant information, wherever that information may come from.

You may ask “isn’t this just RAG? This seems a lot like focusing on retrieval”. And you’d be correct to ask that question. But the term context engineering allows us to think beyond the retrieval step and think about the context window as something that we have to carefully curate, taking into account its limitations as well: quite literally, the context window limit.

### What Makes Up Context

Before writing this blog, we read [“The New Skill in AI is Not Prompting, It’s Context Engineering”](https://www.philschmid.de/context-engineering), by [Philipp Schmid](https://www.linkedin.com/in/philipp-schmid-a6a2bb196/), where he does a great job of breaking down what makes up the context of an AI Agent or LLM. So, here’s what we narrow down as “context” based on both his list, and a few additions from our side:

- **The system prompt/instruction:** sets the scene for the agent about what sort of tasks we want it to perform
- **The user input:** can be anything from a question to a request for a task to be completed.
- **Short term memory or chat history:** provides the LLM context about the ongoing chat.
- **Long-term memory:** can be used to store and retrieve both long-term chat history or other relevant information.
- **Information retrieved from a knowledge base**: this could still be retrieval based on vector search over a database, but could also entail relevant information retrieved from any external knowledge base behind API calls, MCP tools or other sources.
- **Tools and their definitions:** provide additional context to the LLM as to what tools it has access to.
- **Responses from tools:** provide the responses from tool runs back to the LLM as additional context to work with.
- **Structured Outputs:** provide context on what kind of information we are after from the LLM. But can also go the other way in providing condensed, structured information as context for specific tasks.
- **Global State/Context:** especially relevant to agents built with LlamaIndex, allowing us to use workflow [`Context`](https://docs.llamaindex.ai/en/stable/api_reference/workflow/context/) as a sort of scratchpad that we can store and retrieve global information across agent steps.

Some combination of the above make up the context for the underlying LLM in practically all agentic AI applications now. Which brings us to the main point: thinking about precisely which of the above should make up your agent context, and _in what manner_ is exactly what context engineering calls for. So with that, let’s look at some examples of situations in which we might want to think about our context strategy, and how you may implement these with LlamaIndex and LlamaCloud.

## Techniques and Strategies to Consider for Context Engineering

A quick glance at the list above and you may already notice that there’s a lot that _could_ make up our context. Which means we have 2 main challenges: selecting the right context, and making that context fit the context window. While I’m fully aware that this list could grow and grow, let’s look at a few architectural choices that will be top of mind when curating the right context for an agent:

### Knowledge base or tool selection

When we think of RAG, we are mostly talking about AI applications that are designed to do question answering over a single knowledge base, often a vector store. But, for most agentic applications today, this is no longer the case. We now see applications that need to have access to multiple knowledge bases, maybe with the addition of tools that can either return more context or perform certain tasks.

Before we retrieve additional context from a knowledge base or tool though, the first context the LLM has is information _about_ the available tools or knowledge bases in the first place. This is context that allows us to ensure that our agentic ai application is choosing the right resource.https://www.llamaindex.ai/_next/image?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2F7m9jw85w%2Fproduction%2F7681afcfedcbd9618b12adf31c3a2fa77703dedd-1668x866.png%3Ffit%3Dmax%26auto%3Dformat&w=1920&q=75

### Context ordering or compression

Another important consideration when it comes to context engineering is the limitations we have when it comes to the context limit. We simply have a limited space to work with. This has lead to some implementations where we try to make the most out of that space by employing techniques such as context summarization where after a given retrieval step, we summarize the results before adding it to the LLM context.

In some other cases, it’s not only the content of the context that matters, but also the order in which it appears. Consider a use-case where we not only need to retrieve data, but the date of the information is also highly relevant. In that situation, incorporating a ranking step which allows the LLM to receive the most relevant information in terms of ordering can also be quite effective.

```
def search_knowledge(
  query: Annotated[str, “A natural language query or question.”]
) → str:
  """Useful for retrieving knowledge from a database containing information about""" XYZ. Each query should be a pointed and specific natural language question or query.”””

  nodes = retriever.retrieve(query)
	sorted_and_filtered_nodes = sorted(
    [item for item in data if datetime.strptime(item['date'], '%Y-%m-%d') > cutoff_date],
    key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d')
  )
  return "\\n----\\n".join([n.text for n in sorted_and_filtered_nodes])
```

### Choices for Long-term memory storage and retrieval

If we have an application where we need ongoing conversations with an LLM, the history of that conversation becomes context in itself. In LlamaIndex, we’ve provided an array of long-term memory implementations for this exact reason, as well as providing a Base Memory Block that can be extended to implement any unique memory requirements you may have.

For example, some of the pre-built memory blocks we provide are:

- `VectorMemoryBlock`: A memory block that stores and retrieves batches of chat messages from a vector database.
- `FactExtractionMemoryBlock`: A memory block that extracts facts from the chat history.
- `StaticMemoryBlock`: A memory block that stores a static piece of information.

With each iteration we have with an agent, if long-term memory is important to the use case, the agent will be retrieving additional context from it before deciding on the next best step. This makes deciding on what _kind_ of long-term memory we need and just how much context it should return a pretty significant decision. In LlamaIndex, we’ve made it so that you can use any combination of the long-term memory blocks above.

### Structured Information

A common mistake we see people make when creating agentic AI systems is often providing _all_ the context when it simply isn’t required; it can potentially overcrowd the context limit when it’s not necessary.

Structured outputs have been one of my absolute favorite features introduced to LLMs in recent years for this reason. They can have a significant impact on providing the _most_ relevant context to LLMs. And it goes both ways:

- The requested structure: this is a schema that we can provide an LLM, to ask for output that matches that schema.
- Structured data provided as additional context: which is a way we can provide relevant context to an LLM without overcrowding it with additional, unnecessary context.

[LlamaExtract](https://docs.cloud.llamaindex.ai/llamaextract/getting_started) is a LlamaCloud tool that allows you to make use of the structured output functionality of LLMs to extract the most relevant data from complex and long files and sources. Once extracted, these structured outputs can be used as condensed context for downstream agentic applications.

### Workflow Engineering

While context engineering focuses on optimizing what information goes into each LLM call, workflow engineering takes a step back to ask: _what sequence of LLM calls and non-LLM steps do we need to reliably complete this work?_ Ultimately this allows us to optimize the context as well. [LlamaIndex Workflows](https://docs.llamaindex.ai/en/stable/module_guides/workflow/) provides an event-driven framework that lets you:

- **Define explicit step sequences**: Map out the exact progression of tasks needed to complete complex work
- **Control context strategically**: Decide precisely when to engage the LLM versus when to use deterministic logic or external tools
- **Ensure reliability**: Build in validation, error handling, and fallback mechanisms that simple agents can't provide
- **Optimize for specific outcomes**: Create specialized workflows that consistently deliver the results your business needs

From a context engineering perspective, workflows are crucial because they prevent context overload. Instead of cramming everything into a single LLM call and hoping for the best, you can break complex tasks into focused steps, each with its own optimized context window.

The strategic insight here is that every AI builder is ultimately building specialized workflows - whether they realize it or not. Document processing workflows, customer support workflows, coding workflows - these are the building blocks of practical AI applications.

</details>

<details>
<summary>context-engineering</summary>

As Andrej Karpathy puts it, LLMs are like a [new kind of operating system](https://www.youtube.com/watch?si=-aKY-x57ILAmWTdw&t=620&v=LCEmiRjPEtQ&feature=youtu.be&ref=blog.langchain.com). The LLM is like the CPU and its [context window](https://docs.anthropic.com/en/docs/build-with-claude/context-windows?ref=blog.langchain.com) is like the RAM, serving as the model’s working memory. Just like RAM, the LLM context window has limited [capacity](https://lilianweng.github.io/posts/2023-06-23-agent/?ref=blog.langchain.com) to handle various sources of context. And just as an operating system curates what fits into a CPU’s RAM, we can think about “context engineering” playing a similar role. [Karpathy summarizes this well](https://x.com/karpathy/status/1937902205765607626?ref=blog.langchain.com):

> _\[Context engineering is the\] ”…delicate art and science of filling the context window with just the right information for the next step.”_https://blog.langchain.com/content/images/2025/07/image-1.pngContext types commonly used in LLM applications

What are the types of context that we need to manage when building LLM applications? Context engineering as an [umbrella](https://x.com/dexhorthy/status/1933283008863482067?ref=blog.langchain.com) that applies across a few different context types:

- **Instructions** – prompts, memories, few‑shot examples, tool descriptions, etc
- **Knowledge** – facts, memories, etc
- **Tools** – feedback from tool calls

### Context Engineering for Agents

This year, interest in [agents](https://www.anthropic.com/engineering/building-effective-agents?ref=blog.langchain.com) has grown tremendously as LLMs get better at [reasoning](https://platform.openai.com/docs/guides/reasoning?api-mode=responses&ref=blog.langchain.com) and [tool calling](https://www.anthropic.com/engineering/building-effective-agents?ref=blog.langchain.com). [Agents](https://www.anthropic.com/engineering/building-effective-agents?ref=blog.langchain.com) interleave [LLM invocations and tool calls](https://www.anthropic.com/engineering/building-effective-agents?ref=blog.langchain.com), often for [long-running tasks](https://blog.langchain.com/introducing-ambient-agents/). Agents interleave [LLM calls and tool calls](https://www.anthropic.com/engineering/building-effective-agents?ref=blog.langchain.com), using tool feedback to decide the next step.https://blog.langchain.com/content/images/2025/07/image-2.pngAgents interleave [LLM calls and](https://www.anthropic.com/engineering/building-effective-agents?ref=blog.langchain.com) [tool calls](https://www.anthropic.com/engineering/building-effective-agents?ref=blog.langchain.com), using tool feedback to decide the next step

However, long-running tasks and accumulating feedback from tool calls mean that agents often utilize a large number of tokens. This can cause numerous problems: it can [exceed the size of the context window](https://cognition.ai/blog/kevin-32b?ref=blog.langchain.com), balloon cost / latency, or degrade agent performance. Drew Breunig [nicely outlined](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html?ref=blog.langchain.com) a number of specific ways that longer context can cause perform problems, including:

- [Context Poisoning: When a hallucination makes it into the context](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html?ref=blog.langchain.com#context-poisoning)
- [Context Distraction: When the context overwhelms the training](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html?ref=blog.langchain.com#context-distraction)
- [Context Confusion: When superfluous context influences the response](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html?ref=blog.langchain.com#context-confusion)
- [Context Clash: When parts of the context disagree](https://www.dbreunig.com/2025/06/22/how-contexts-fail-and-how-to-fix-them.html?ref=blog.langchain.com#context-clash)https://blog.langchain.com/content/images/2025/07/image-3.pngContext from tool calls accumulates over multiple agent turns

With this in mind, [Cognition](https://cognition.ai/blog/dont-build-multi-agents?ref=blog.langchain.com) called out the importance of context engineering:

> _“Context engineering” … is effectively the #1 job of engineers building AI agents._

[Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system?ref=blog.langchain.com) also laid it out clearly:

> _Agents often engage in conversations spanning hundreds of turns, requiring careful context management strategies._

So, how are people tackling this challenge today? We group common strategies for agent context engineering into four buckets — **write, select, compress, and isolate —** and give examples of each from review of some popular agent products and papers.https://blog.langchain.com/content/images/2025/07/image-4.pngGeneral categories of context engineering

### Write Context

_Writing context means saving it outside the context window to help an agent perform a task._

**Scratchpads**

When humans solve tasks, we take notes and remember things for future, related tasks. Agents are also gaining these capabilities! Note-taking via a “ [scratchpad](https://www.anthropic.com/engineering/claude-think-tool?ref=blog.langchain.com)” is one approach to persist information while an agent is performing a task. The idea is to save information outside of the context window so that it’s available to the agent. [Anthropic’s multi-agent researcher](https://www.anthropic.com/engineering/built-multi-agent-research-system?ref=blog.langchain.com) illustrates a clear example of this:

> _The LeadResearcher begins by thinking through the approach and saving its plan to Memory to persist the context, since if the context window exceeds 200,000 tokens it will be truncated and it is important to retain the plan._

Scratchpads can be implemented in a few different ways. They can be a [tool call](https://www.anthropic.com/engineering/claude-think-tool?ref=blog.langchain.com) that simply [writes to a file](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem?ref=blog.langchain.com). They can also be a field in a runtime [state object](https://langchain-ai.github.io/langgraph/concepts/low_level/?ref=blog.langchain.com#state) that persists during the session. In either case, scratchpads let agents save useful information to help them accomplish a task.

**Memories**

Scratchpads help agents solve a task within a given session (or [thread](https://langchain-ai.github.io/langgraph/concepts/persistence/?ref=blog.langchain.com#threads)), but sometimes agents benefit from remembering things across _many_ sessions! [Reflexion](https://arxiv.org/abs/2303.11366?ref=blog.langchain.com) introduced the idea of reflection following each agent turn and re-using these self-generated memories. [Generative Agents](https://ar5iv.labs.arxiv.org/html/2304.03442?ref=blog.langchain.com) created memories synthesized periodically from collections of past agent feedback.https://blog.langchain.com/content/images/2025/07/image-5.pngAn LLM can be used to update or create memories

These concepts made their way into popular products like [ChatGPT](https://help.openai.com/en/articles/8590148-memory-faq?ref=blog.langchain.com), [Cursor](https://forum.cursor.com/t/0-51-memories-feature/98509?ref=blog.langchain.com), and [Windsurf](https://docs.windsurf.com/windsurf/cascade/memories?ref=blog.langchain.com), which all have mechanisms to auto-generate long-term memories that can persist across sessions based on user-agent interactions.

### Select Context

_Selecting context means pulling it into the context window to help an agent perform a task._

**Scratchpad**

The mechanism for selecting context from a scratchpad depends upon how the scratchpad is implemented. If it’s a [tool](https://www.anthropic.com/engineering/claude-think-tool?ref=blog.langchain.com), then an agent can simply read it by making a tool call. If it’s part of the agent’s runtime state, then the developer can choose what parts of state to expose to an agent each step. This provides a fine-grained level of control for exposing scratchpad context to the LLM at later turns.

**Memories**

If agents have the ability to save memories, they also need the ability to select memories relevant to the task they are performing. This can be useful for a few reasons. Agents might select few-shot examples ( [episodic](https://langchain-ai.github.io/langgraph/concepts/memory/?ref=blog.langchain.com#memory-types) [memories](https://arxiv.org/pdf/2309.02427?ref=blog.langchain.com)) for examples of desired behavior, instructions ( [procedural](https://langchain-ai.github.io/langgraph/concepts/memory/?ref=blog.langchain.com#memory-types) [memories](https://arxiv.org/pdf/2309.02427?ref=blog.langchain.com)) to steer behavior, or facts ( [semantic](https://langchain-ai.github.io/langgraph/concepts/memory/?ref=blog.langchain.com#memory-types) [memories](https://arxiv.org/pdf/2309.02427?ref=blog.langchain.com)) for task-relevant context.https://blog.langchain.com/content/images/2025/07/image-6.png

One challenge is ensuring that relevant memories are selected. Some popular agents simply use a narrow set of files that are _always_ pulled into context. For example, many code agent use specific files to save instructions (”procedural” memories) or, in some cases, examples (”episodic” memories). Claude Code uses [`CLAUDE.md`](http://claude.md/?ref=blog.langchain.com). [Cursor](https://docs.cursor.com/context/rules?ref=blog.langchain.com) and [Windsurf](https://windsurf.com/editor/directory?ref=blog.langchain.com) use rules files.

But, if an agent is storing a larger [collection](https://langchain-ai.github.io/langgraph/concepts/memory/?ref=blog.langchain.com#collection) of facts and / or relationships (e.g., [semantic](https://langchain-ai.github.io/langgraph/concepts/memory/?ref=blog.langchain.com#memory-types) memories), selection is harder. [ChatGPT](https://help.openai.com/en/articles/8590148-memory-faq?ref=blog.langchain.com) is a good example of a popular product that stores and selects from a large collection of user-specific memories.

Embeddings and / or [knowledge](https://arxiv.org/html/2501.13956v1?ref=blog.langchain.com#:~:text=In%20Zep%2C%20memory%20is%20powered,subgraph%2C%20and%20a%20community%20subgraph) [graphs](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/?ref=blog.langchain.com#:~:text=changes%20since%20updates%20can%20trigger,and%20holistic%20memory%20for%20agentic) for memory indexing are commonly used to assist with selection. Still, memory selection is challenging. At the AIEngineer World’s Fair, [Simon Willison shared](https://simonwillison.net/2025/Jun/6/six-months-in-llms/?ref=blog.langchain.com) an example of selection gone wrong: ChatGPT fetched his location from memories and unexpectedly injected it into a requested image. This type of unexpected or undesired memory retrieval can make some users feel like the context window “ _no longer belongs to them_”!

**Tools**

Agents use tools, but can become overloaded if they are provided with too many. This is often because the tool descriptions overlap, causing model confusion about which tool to use. One approach is [to apply RAG (retrieval augmented generation) to tool descriptions](https://arxiv.org/abs/2410.14594?ref=blog.langchain.com) in order to fetch only the most relevant tools for a task. Some [recent papers](https://arxiv.org/abs/2505.03275?ref=blog.langchain.com) have shown that this improve tool selection accuracy by 3-fold.

**Knowledge**

[RAG](https://github.com/langchain-ai/rag-from-scratch?ref=blog.langchain.com) is a rich topic and it [can be a central context engineering challenge](https://x.com/_mohansolo/status/1899630246862966837?ref=blog.langchain.com). Code agents are some of the best examples of RAG in large-scale production. Varun from Windsurf captures some of these challenges well:

> _Indexing code ≠ context retrieval … \[We are doing indexing & embedding search … \[with\] AST parsing code and chunking along semantically meaningful boundaries … embedding search becomes unreliable as a retrieval heuristic as the size of the codebase grows … we must rely on a combination of techniques like grep/file search, knowledge graph based retrieval, and … a re-ranking step where \[context\] is ranked in order of relevance._

### Compressing Context

_Compressing context involves retaining only the tokens required to perform a task._

**Context Summarization**

Agent interactions can span [hundreds of turns](https://www.anthropic.com/engineering/built-multi-agent-research-system?ref=blog.langchain.com) and use token-heavy tool calls. Summarization is one common way to manage these challenges. If you’ve used Claude Code, you’ve seen this in action. Claude Code runs “ [auto-compact](https://docs.anthropic.com/en/docs/claude-code/costs?ref=blog.langchain.com)” after you exceed 95% of the context window and it will summarize the full trajectory of user-agent interactions. This type of compression across an [agent trajectory](https://langchain-ai.github.io/langgraph/concepts/memory/?ref=blog.langchain.com#manage-short-term-memory) can use various strategies such as [recursive](https://arxiv.org/pdf/2308.15022?ref=blog.langchain.com#:~:text=the%20retrieved%20utterances%20capture%20the,based%203) or [hierarchical](https://alignment.anthropic.com/2025/summarization-for-monitoring/?ref=blog.langchain.com#:~:text=We%20addressed%20these%20issues%20by,of%20our%20computer%20use%20capability) summarization.https://blog.langchain.com/content/images/2025/07/image-7.pngA few places where summarization can be applied

It can also be useful to [add summarization](https://github.com/langchain-ai/open_deep_research/blob/e5a5160a398a3699857d00d8569cb7fd0ac48a4f/src/open_deep_research/utils.py?ref=blog.langchain.com#L1407) at specific points in an agent’s design. For example, it can be used to post-process certain tool calls (e.g., token-heavy search tools). As a second example, [Cognition](https://cognition.ai/blog/dont-build-multi-agents?ref=blog.langchain.com#a-theory-of-building-long-running-agents) mentioned summarization at agent-agent boundaries to reduce tokens during knowledge hand-off. Summarization can be a challenge if specific events or decisions need to be captured. [Cognition](https://cognition.ai/blog/dont-build-multi-agents?ref=blog.langchain.com#a-theory-of-building-long-running-agents) uses a fine-tuned model for this, which underscores how much work can go into this step.

**Context Trimming**

Whereas summarization typically uses an LLM to distill the most relevant pieces of context, trimming can often filter or, as Drew Breunig points out, “ [prune](https://www.dbreunig.com/2025/06/26/how-to-fix-your-context.html?ref=blog.langchain.com)” context. This can use hard-coded heuristics like removing [older messages](https://python.langchain.com/docs/how_to/trim_messages/?ref=blog.langchain.com) from a list. Drew also mentions [Provence](https://arxiv.org/abs/2501.16214?ref=blog.langchain.com), a trained context pruner for Question-Answering.

### Isolating Context

_Isolating context involves splitting it up to help an agent perform a task._

**Multi-agent**

One of the most popular ways to isolate context is to split it across sub-agents. A motivation for the OpenAI [Swarm](https://github.com/openai/swarm?ref=blog.langchain.com) library was [separation of concerns](https://openai.github.io/openai-agents-python/ref/agent/?ref=blog.langchain.com), where a team of agents can handle specific sub-tasks. Each agent has a specific set of tools, instructions, and its own context window.https://blog.langchain.com/content/images/2025/07/image-8.pngSplit context across multiple agents

Anthropic’s [multi-agent researcher](https://www.anthropic.com/engineering/built-multi-agent-research-system?ref=blog.langchain.com) makes a case for this: many agents with isolated contexts outperformed single-agent, largely because each subagent context window can be allocated to a more narrow sub-task. As the blog said:

> _\[Subagents operate\] in parallel with their own context windows, exploring different aspects of the question simultaneously._

Of course, the challenges with multi-agent include token use (e.g., up to [15× more tokens](https://www.anthropic.com/engineering/built-multi-agent-research-system?ref=blog.langchain.com) than chat as reported by Anthropic), the need for careful [prompt engineering](https://www.anthropic.com/engineering/built-multi-agent-research-system?ref=blog.langchain.com) to plan sub-agent work, and coordination of sub-agents.

**Context Isolation with Environments**

HuggingFace’s [deep researcher](https://huggingface.co/blog/open-deep-research?ref=blog.langchain.com#:~:text=From%20building%20,it%20can%20still%20use%20it) shows another interesting example of context isolation. Most agents use [tool calling APIs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview?ref=blog.langchain.com), which return JSON objects (tool arguments) that can be passed to tools (e.g., a search API) to get tool feedback (e.g., search results). HuggingFace uses a [CodeAgent](https://huggingface.co/papers/2402.01030?ref=blog.langchain.com), which outputs that contains the desired tool calls. The code then runs in a [sandbox](https://e2b.dev/?ref=blog.langchain.com). Selected context (e.g., return values) from the tool calls is then passed back to the LLM.https://blog.langchain.com/content/images/2025/07/image-9.pngSandboxes can isolate context from the LLM.

This allows context to be isolated from the LLM in the environment. Hugging Face noted that this is a great way to isolate token-heavy objects in particular:

> _\[Code Agents allow for\] a better handling of state … Need to store this image / audio / other for later use? No problem, just assign it as a variable_ [_in your state and you \[use it later\]_](https://deepwiki.com/search/i-am-wondering-if-state-that-i_0e153539-282a-437c-b2b0-d2d68e51b873?ref=blog.langchain.com) _._

**State**

It’s worth calling out that an agent’s runtime [state object](https://langchain-ai.github.io/langgraph/concepts/low_level/?ref=blog.langchain.com#state) can also be a great way to isolate context. This can serve the same purpose as sandboxing. A state object can be designed with a [schema](https://langchain-ai.github.io/langgraph/concepts/low_level/?ref=blog.langchain.com#schema) that has fields that context can be written to. One field of the schema (e.g., `messages`) can be exposed to the LLM at each turn of the agent, but the schema can isolate information in other fields for more selective use.

</details>

<details>
<summary>scraping-failed</summary>

⚠️ Error scraping https://arxiv.org/pdf/2507.13334 after 3 attempts: Request Timeout: Failed to make POST request as the request timed out. Request timed out - No additional error details provided.

</details>

<details>
<summary>the-rise-of-context-engineering</summary>

Context engineering is building dynamic systems to provide the right information and tools in the right format such that the LLM can plausibly accomplish the task.

Most of the time when an agent is not performing reliably the underlying cause is that the appropriate context, instructions and tools have not been communicated to the model.

LLM applications are evolving from single prompts to more complex, dynamic agentic systems. As such, context engineering is becoming the [most important skill an AI engineer can develop](https://cognition.ai/blog/dont-build-multi-agents?ref=blog.langchain.com#a-theory-of-building-long-running-agents).

## What is context engineering?

Context engineering is building dynamic systems to provide the right information and tools in the right format such that the LLM can plausibly accomplish the task.

Let’s break it down.

**Context engineering is a system**

Complex agents likely get context from many sources. Context can come from the developer of the application, the user, previous interactions, tool calls, or other external data. Pulling these all together involves a complex system.

**This system is dynamic**

Many of these pieces of context can come in dynamically. As such, the logic for constructing the final prompt needs to be dynamic as well. It is not just a static prompt.

**You need the right information**

A common reason agentic systems don’t perform is they just don’t have the right context. LLMs cannot read minds - you need to give them the right information. Garbage in, garbage out.

**You need the right tools**

It may not always be the case that the LLM will be able to solve the task just based solely on the inputs. In these situations, if you want to empower the LLM to do so, you will want to make sure that it has the right tools. These could be tools to look up more information, take actions, or anything in between. Giving the LLM the right tools is just as important as giving it the right information.

**The format matters**

Just like communicating with humans, how you communicate with LLMs matters. A short but descriptive error message will go a lot further a large JSON blob. This also applies to tools. What the input parameters to your tools are matters a lot when making sure that LLMs can use them.

**Can it plausibly accomplish the task?**

This is a great question to be asking as you think about context engineering. It reinforces that LLMs are not mind readers - you need to set them up for success. It also helps separate the failure modes. Is it failing because you haven’t given it the right information or tools? Or does it have all the right information and it just messed up? These failure modes have very different ways to fix them.

## Why is context engineering important

When agentic systems mess up, it’s largely because an LLM messes. Thinking from first principles, LLMs can mess up for two reasons:

1. The underlying model just messed up, it isn’t good enough
2. The underlying model was not passed the appropriate context to make a good output

More often than not (especially as the models get better) model mistakes are caused more by the second reason. The context passed to the model may be bad for a few reasons:

- There is just missing context that the model would need to make the right decision. Models are not mind readers. If you do not give them the right context, they won’t know it exists.
- The context is formatted poorly. Just like humans, communication is important! How you format data when passing into a model absolutely affects how it responds

## How is context engineering different from prompt engineering?

Why the shift from “prompts” to “context”? Early on, developers focused on phrasing prompts cleverly to coax better answers. But as applications grow more complex, it’s becoming clear that **providing complete and structured context** to the AI is far more important than any magic wording.

I would also argue that prompt engineering is a subset of context engineering. Even if you have all the context, how you assemble it in the prompt still absolutely matters. The difference is that you are not architecting your prompt to work well with a single set of input data, but rather to take a set of dynamic data and format it properly.

I would also highlight that a key part of context is often core instructions for how the LLM should behave. This is often a key part of prompt engineering. Would you say that providing clear and detailed instructions for how the agent should behave is context engineering or prompt engineering? I think it’s a bit of both.

## Examples of context engineering

Some basic examples of good context engineering include:

- Tool use: Making sure that if an agent needs access to external information, it has tools that can access it. When tools return information, they are formatted in a way that is maximally digestable for LLMs
- Short term memory: If a conversation is going on for a while, creating a summary of the conversation and using that in the future.
- Long term memory: If a user has expressed preferences in a previous conversation, being able to fetch that information.
- Prompt Engineering: Instructions for how an agent should behave are clearly enumerated in the prompt.
- Retrieval: Fetching information dynamically and inserting it into the prompt before calling the LLM.

</details>

<details>
<summary>what-is-context-engineering-pinecone</summary>

LLMs are getting better, faster, and smarter, and as they do, we need new ways to use them.

Applications people build with them have transitioned from asking LLMs to write to letting LLMs drive actions. With that, comes new challenges in developing what are called agentic applications.

**Context engineering** is a term that attempts to describe the architecting necessary to support building accurate LLM applications. But what does context engineering involve?

## Hallucinations Constrain AI Applications

Much has been made of the potential of agents to complete tasks and revolutionize industries. Still, if there’s one thing that has passed the test of time, it’s that LLM applications will always fail without the relevant information. And in those failures, come hallucinations.

Multiple tool calls, messages, and competing objectives blur instructions in agentic applications. Due to these diverse integrations all competing for a fixed (literal!) attention span for a model, a need arises for _engineering their integration._ Absent this, models default to their world knowledge and information to generate results, which can result in unintended consequences.

Context engineering is an umbrella term for a series of techniques to maintain the necessary information needed for an agent to complete tasks successfully. [Harrison Chase from LangChain](https://blog.langchain.com/the-rise-of-context-engineering/) breaks down context engineering into a few parts:

- actions the LLM can take (tool use)
- instructions from the user (prompt engineering)
- data related to the task at hand, like code, documents, produced artifacts, etc (retrieval)
- historical artifacts like conversation memory or user facts (long and short term memory)
- data produced by subagents, or other intermediate task or tool outputs (agentic architectures)https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fe5b53eff8128606a7432ceb85a46b0fee9052c21-2840x1530.png&w=3840&q=75

Context Engineering requires putting together many building blocks of context generated from various resources, into a finite context window

All of these must fit into a finite context window for applications to succeed.

Retrieval and vector databases are uniquely impactful for these applications, as they help retrieve the external information in various modalities and representations necessary to ground responses with context. But just having the context isn’t enough.

Organizing, filtering, deleting, and processing this information so that an LLM can continue to focus on the task at hand is context engineering.

## Applying Lessons from Retrieval-augmented Generation to Context Engineering

Now if you’re reading this far, you might think, oh no!! Another technique for the aspiring AI engineer to learn, the horror! How will you ever catch up!?!

Not to fear. If you’ve built any search or retrieval-augmented generation application before, you already know a lot of the principles for context engineering! In fact, we can make the argument that **context engineering is just a step-up abstraction of prompt engineering for RAG applications**.

How, you ask?

Imagine you’ve built an application for helping answer incoming customer support tickets. It’s architected as follows:

1. Take an incoming user query, and query your semantic search which indexes documents from your company
2. pass the retrieved context to an LLM, like Claude or OpenAI
3. Answer user queries using the context

Accordingly, the application has access to a knowledge base of information that might include previous support tickets, company documentation, and other information critical to respond to users.

You might use a prompt like this:

```text
You are a customer support agent tasked with helping users solve their problems.

You have access to a knowledge base containing documentation, FAQs, and previous support tickets.

Given the information below, please help the user with their query.

If you don't know the answer, say so and offer to create a support ticket.

INSTRUCTIONS:

Always be polite and professional

Use the provided context to answer questions accurately

If the information needed is not in the context, acknowledge this and offer to create a support ticket

If creating a ticket, collect: user name, email, issue description, and priority level

For technical questions, provide step-by-step instructions when possible

CONTEXT: <retrieved docs>

USER QUERY: <user query>

Please respond in a helpful, conversational manner while remaining factual and concise.
```

In that prompt, you’d balance how to drive the LLM’s behavior, manage the documents retrieved from the user query, and provide any additional information necessary for the task at hand.

It’s a great proof-of-concept that quickly delivers answers to frustrated users. But, you have a new requirement now:

> Build a chatbot that can manage support tickets given user queries

Specifically, the chatbot must be turned into an agent that can:

- Maintain a conversation with users and extract key information from them for the tickets
- Open, write to, update, and close support tickets
- Answer tickets that are in-domain or available in a knowledge base or previous tickets
- Route the tickets to an appropriate customer support personnel for follow-up

The LLM must reason and act instead of just responding. It must also maintain information about a given set of tickets over time to provide a personalized user experience.

So, how do we go about doing this?

We might need some of the following:

- Tool Use, to enable writing and closing tickets
- Memory, to understand user needs and maintain key information over time, as well as to summarize and manage information over time
- Retrieval, to modify user queries to find documentation and information over time
- Structured Generation, to properly extract information for tickets, or to classify and route tickets to employees
- Compaction, Deletion, and Scratchpads to maintain, remove, and persist temporary information over time

All of these additional capabilities consume significant context over time, and warrant additional data structures, mechanisms, programming, and prompt engineering to smooth out capabilities.

Fortunately, prompt engineering for RAG incorporates many lessons you’d need to help tackle this problem.

We know that all embedding models and LLMs have limits to the amount of information they can process in their context window, and that the best way to [budget this window is via **chunking**](https://www.pinecone.io/learn/chunking-strategies/).

Furthermore, you may be [familiar with reranking](https://www.pinecone.io/learn/refine-with-rerank/), which allows you to refine relevant documents sets down to more manageable sizes, to reduce cost, latency and hallucination rates.

Here, we can see how summarization and reranking can prune context down for future conversational turns.https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fecb752e2dbf9ed122712656efcb392218d767509-2983x2900.png&w=3840&q=75

And, if you are building agents, you might even know about the importance of letting your agent control queries to an [**external vector database via a tool or MCP server**,](https://www.pinecone.io/blog/first-MCPs/) which lets it determine the appropriate questions to ask for the task at hand.

All of these techniques help you generate accurate responses given a user’s query. For more examples of how this is achieved in practice, read Dexter Horthy’s [great writeup](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-03-own-your-context-window.md) on context engineering in prompts, or Drew Breunig’s write up on fixing [context issues here](https://www.dbreunig.com/2025/06/26/how-to-fix-your-context.html).

But, user’s might make multiple queries. They might ask for revisions on existing information, or for you to get new information for the current task. They want their problems solved, not just explained. This is where an agentic architecture becomes necessary, and context engineering starts to become a useful concept.

### How Context Engineering informs Agentic Architectures

As you build this system, you get some feedback from your coworkers:

> Your current implementation relies on a single agent interacting with the user. This creates a bottleneck where the agent must wait on tool calls or user input to do certain things. What if we implemented a subagent architecture instead?

In other words, instead of having a single LLM instance make tickets, route requests, and maintain a conversation with users, our LLM could delegate tasks to other agents to complete asynchronously.

This would free up our “driving” LLM instance to continue conversing with our frustrated customer, ensuring lower latencies in a domain where every second matters.

Great idea! But, context engineering gives us a framework to think about the benefits of these kinds of parallelized architectures versus sequential ones.

Anthropic and Cognition both wrote about the tradeoffs that come with these, concluding that for read-heavy applications (l [ike research agents](https://www.anthropic.com/engineering/built-multi-agent-research-system)) or certain technical ones ( [code agents](https://cognition.ai/blog/dont-build-multi-agents)), a sequential agentic architecture may be easier to maintain context with than one that involves subagents. This mostly comes down to engineering the context gained and lost over the course of the agent’s work, as well as eschewing multi-agent architectures due to the difficulty of maintaining context over multiple agent runs.

</details>
