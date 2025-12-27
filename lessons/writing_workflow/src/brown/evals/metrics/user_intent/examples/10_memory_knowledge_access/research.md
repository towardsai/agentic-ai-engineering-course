# Research

## Research Results

<details>
<summary>What are the best practices for managing AI agent memory to avoid user cognitive overhead?</summary>

### Source [2]: https://www.letta.com/blog/agent-memory

Query: What are the best practices for managing AI agent memory to avoid user cognitive overhead?

Answer: Best practices for managing AI agent memory to avoid user cognitive overhead include **Message Eviction & Summarization**: when context windows fill up, agents should intelligently evict and summarize older messages, retaining essential details and compressing information without losing continuity. Techniques like **Recursive Summarization** ensure that as conversations grow, summaries are periodically updated, with recent interactions given more weight. **Memory Blocks** structure memory by labeling, describing, and limiting stored information, allowing agents to rewrite or consolidate blocks as needed. For long-term or external memory, **Vector and Graph Databases** enable efficient retrieval, embedding, and reasoning over past experiences. These practices collectively help agents maintain relevant context, minimize redundancy, and organize memory for optimal retrieval, thereby reducing user cognitive overhead[2].

-----

-----

### Source [3]: https://www.youtube.com/watch?v=W2HVdB4Jbjs

Query: What are the best practices for managing AI agent memory to avoid user cognitive overhead?

Answer: Effective agent memory management draws inspiration from human memory systems—such as **episodic, working, semantic, and procedural memory**—to create believable, reliable, and capable AI agents. Architecting agent memory involves structuring memory components (conversation, workflow, persona) and operationalizing short-term, long-term, and dynamic memory modes. Production-ready practices include leveraging **vector databases and hybrid search** for scalable persistence, using **embeddings and relevance scoring** for semantic retrieval, and implementing strategies like **memory cascading and selective deletion** to optimize context window use. Advanced techniques such as integrating tool use and persona memory further enhance context awareness. The key is to balance retention of important information with performance constraints, ensuring agents can remember and reason without overwhelming users with excessive or irrelevant detail[3].

-----

-----

### Source [4]: https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/memory-management-for-ai-agents/4406359

Query: What are the best practices for managing AI agent memory to avoid user cognitive overhead?

Answer: To minimize user cognitive overhead, AI agent memory should: **extract key information** from past interactions (avoiding unnecessary details), **prevent duplication** in stored memory, **append new facts** to existing memory, and **update or change stored data** as conversations evolve. Prioritizing memory items by frequency of access ensures the most relevant information is readily available. Using dedicated memory management frameworks (like Mem0) can automate these processes, integrating with retrieval systems (e.g., Azure AI Search) for streamlined storage and access. These approaches keep memory concise and relevant, allowing users to interact with agents without wading through excessive historical information, thereby reducing cognitive burden[4].

-----

</details>

<details>
<summary>How have expanding LLM context windows changed strategies for AI agent memory compression and summarization?</summary>

### Source [6]: https://arxiv.org/html/2406.06110v1

Query: How have expanding LLM context windows changed strategies for AI agent memory compression and summarization?

Answer: Expanding the context window of large language models (LLMs) historically faces two primary challenges: the model’s pretraining context window limit and the substantial memory footprint required by the Transformer architecture's need to store information from the entire input sequence (KV-Cache). To address these, recent research has focused on optimizing training methods, model structures, and specifically, context compression techniques. Context compression is promising because it allows models to compress context or prompts into shorter forms while maintaining performance, thus enabling inference with longer context windows without exceeding memory limits. Many compression-based techniques can also be integrated with other context window extension methods, further enhancing LLM performance when operating with limited resources. The introduction of methods such as Recurrent Context Compression (RCC) demonstrates that compressed representations can be used to extend effective context length and reduce resource consumption compared to non-compression methods, making LLMs more practical for long-document or multi-turn scenarios.

-----

-----

-----

### Source [7]: https://aclanthology.org/2024.findings-emnlp.358.pdf

Query: How have expanding LLM context windows changed strategies for AI agent memory compression and summarization?

Answer: Selective Compression Attention (SCA) is introduced as a general method for expanding LLM context windows and reducing memory requirements. SCA ensures that the key-value (KV) cache size does not exceed a set threshold, significantly lessening memory usage when processing long contexts. Traditional strategies like keeping only the most recent tokens ("local" retention) prevent memory overflow but severely degrade the model’s language modeling ability. Compression methods, including SCA, allow for the context window to be expanded by up to 16× while keeping perplexity (a measure of language modeling ability) within an acceptable range. Experimental results indicate that while it is relatively straightforward to enable LLMs to model long text through compression, naive truncation or frequent summarization generally destroys modeling capability—highlighting the necessity for sophisticated compression strategies as context windows grow.

-----

-----

-----

### Source [8]: https://supermemory.ai/blog/extending-context-windows-in-llms/

Query: How have expanding LLM context windows changed strategies for AI agent memory compression and summarization?

Answer: Two main strategies address LLM context window limitations: Semantic Compression (shrinking a long document to fit within the LLM window) and Retrieval-Augmented Memory (retrieving only the most relevant conversational history). Expanding the context window reduces the need for aggressive summarization but does not eliminate the quadratic computational scaling of self-attention. As context windows expand, practitioners face a choice: truncate old context (losing history) or summarize (risking loss of important detail). Even with rolling summarizers, maintaining an outline of history involves significant token overhead. The optimal approach is dynamic prompt adaptation—presenting only essential information to the model at each turn—rather than simple summarization or truncation. This shift in strategy becomes increasingly important as context windows expand, emphasizing relevance-driven retrieval and selective compression over brute-force summarization.

-----

-----

-----

### Source [9]: https://research.ibm.com/blog/larger-context-window

Query: How have expanding LLM context windows changed strategies for AI agent memory compression and summarization?

Answer: Larger context windows in LLMs function as an expanded working memory, allowing tracking of more details and extended reasoning. However, due to the quadratic scaling of memory and compute with sequence length, increasing context window size is computationally expensive. IBM researchers addressed this by introducing computationally efficient attention mechanisms (e.g., ring attention), relative positional encoding, and pretraining on long-form documents. Compressing input prompts into shorter forms is another technique to effectively enlarge usable context windows. IBM developed a method for LLMs to generate and compress synthetic longform instruction data at varying ratios, allowing the model to select the optimal compression ratio at inference time based on prompt size. This enables interpretation of longer sequences without overwhelming memory or compute budgets, showing that prompt compression remains a critical component of memory management strategies even as context windows grow.

-----

-----

-----

### Source [10]: https://www.factory.ai/news/compressing-context

Query: How have expanding LLM context windows changed strategies for AI agent memory compression and summarization?

Answer: Setting thresholds for when to compress context introduces tradeoffs between performance (quality of reasoning) and cost (inference compute and latency). Higher compression thresholds retain more context, supporting richer reasoning but raising costs linearly with context length. However, performance gains diminish beyond certain limits, with some models actually degrading at maximum context lengths. Narrow gaps between thresholds prompt frequent compression, increasing summarization overhead and potentially invalidating prompt caches, but better preserve recent context. Wider gaps reduce compression frequency but risk truncating relevant information. The optimal compression configuration is highly task-dependent. As context windows expand, the strategy shifts from aggressive, frequent summarization to more nuanced threshold tuning, balancing model quality, memory, and compute budgets for the specific application.

-----

-----

</details>

<details>
<summary>What are the pros and cons of storing AI agent memories as raw strings versus structured data like JSON or knowledge graphs?</summary>

### Source [11]: https://www.geeksforgeeks.org/artificial-intelligence/ai-agent-memory/

Query: What are the pros and cons of storing AI agent memories as raw strings versus structured data like JSON or knowledge graphs?

Answer: AI agent memory can be implemented using various storage methods, each offering distinct trade-offs between simplicity, structure, and capability. **Raw string memory**, such as buffers or queues, is easy to implement and fast, making it suitable for short-term context where rapid retrieval and minimal complexity are desired. However, this approach cannot handle long-term storage well and lacks structure for advanced querying.

**Structured formats** like JSON or relationship databases facilitate long-term and reliable storage, enabling mature technologies and easier querying for specific facts. Despite their strengths, these structured approaches are less effective at semantic and contextual queries, as they often treat information rigidly rather than understanding nuanced context.

**Knowledge graphs** offer the ability to model relationships and world knowledge, excelling at reasoning and inference. Their strengths lie in capturing complex interconnections, but they can be complex to build and maintain, requiring significant upfront design and ongoing management.

**Vector databases** represent another alternative, suitable for semantic search and handling unstructured data via embeddings. These allow for fuzzy matching and scalability but require the generation of embeddings, which introduces additional complexity.

In summary, **raw strings** are optimal for speed and simplicity in short-term contexts, while **structured data** and **knowledge graphs** provide richer querying, reasoning, and long-term memory capabilities at the cost of increased complexity and resource requirements.

-----

-----

-----

### Source [12]: https://dev.to/foxgem/ai-agent-memory-a-comparative-analysis-of-langgraph-crewai-and-autogen-31dp

Query: What are the pros and cons of storing AI agent memories as raw strings versus structured data like JSON or knowledge graphs?

Answer: The comparative analysis of LangGraph, CrewAI, and AutoGen highlights different strategies for memory management in AI agents. AutoGen utilizes **message lists**, which are essentially raw strings, to store conversational histories. This approach is simple and aligns with traditional chatbot designs, offering straightforward retrieval and minimal setup.

However, frameworks like CrewAI employ **built-in memory types** that support structured data storage, such as JSON-like objects or knowledge graphs. These enable the agent to organize information hierarchically or relationally, supporting more advanced reasoning and retrieval tasks. LangGraph prioritizes customizable memory solutions, allowing developers to choose between raw strings for simplicity or structured formats for more complex needs.

The choice between raw strings and structured data is thus influenced by application requirements: raw strings are fast and simple, best for short-term or linear tasks, while structured memory types allow for richer interactions, better context management, and support for complex queries at the cost of greater implementation complexity.

-----

-----

-----

### Source [13]: https://www.arxiv.org/pdf/2506.06326

Query: What are the pros and cons of storing AI agent memories as raw strings versus structured data like JSON or knowledge graphs?

Answer: The MemoryOS architecture for AI agents introduces a **hierarchical storage system** inspired by operating system memory management. It employs segment-paging techniques to organize dialogue history, dynamically prioritizing critical information and enabling efficient retrieval.

Raw string storage (such as keeping the entire dialogue transcript) is easy to implement but quickly hits the context window limitations of LLMs, hampering long-term coherence and personalization. By contrast, structured memory—where dialogue history and agent knowledge are organized into distinct storage tiers (short-term, mid-term, long-term)—supports dynamic updates, semantic retrieval, and heat-driven eviction policies that ensure important information is retained.

Structured memory enables the extraction and tracking of evolving user preferences and personal traits, making responses more aligned with long-term conversational context. This approach improves coherence and personalization in extended interactions, whereas raw string storage struggles as conversations grow longer and more complex.

-----

-----

-----

### Source [14]: https://diamantai.substack.com/p/memory-optimization-strategies-in

Query: What are the pros and cons of storing AI agent memories as raw strings versus structured data like JSON or knowledge graphs?

Answer: AI agents traditionally use a **sequential memory chain**, storing all messages as raw strings in the conversation history. This method is simple and ensures no details are lost, making it effective for short interactions. However, as conversations lengthen, this approach can overwhelm context limits and slow down processing. The agent may also lose the ability to efficiently recall older but critical information as the history grows.

A more optimized method is the **sliding window** approach, where only the most recent messages are stored as raw strings. This keeps memory size manageable and context relevant, but risks forgetting important information from earlier in the conversation.

The main trade-off is between **simplicity and scalability**: raw string memory is easy to use for short-term recall but does not scale well for long-term, complex reasoning. Structured memory (such as indexed databases or graphs) is better for managing long conversations, supporting efficient retrieval and maintaining important context over time.

-----

-----

-----

### Source [15]: https://www.letta.com/blog/benchmarking-ai-agent-memory

Query: What are the pros and cons of storing AI agent memories as raw strings versus structured data like JSON or knowledge graphs?

Answer: Benchmarking AI agent memory methods reveals that simply storing conversational histories as raw strings in files can outperform some specialized memory and retrieval systems. This approach scored 74.0% accuracy on the LoCoMo benchmark, suggesting that for certain tasks, the method of storage (raw strings versus structured formats) might be less critical than how the agent manages and accesses context.

However, relying solely on raw string storage presents risks for complex, long-running tasks: agents may forget objectives or important information, leading to "derailment". Structured memory approaches, such as hierarchical memory inspired by operating systems, allow agents to manage immediate context separately from archival memory, supporting better retention and retrieval of relevant information.

Thus, while raw string storage can be surprisingly effective for some benchmarks, structured data formats are necessary for robust long-term memory, learning, and the ability to handle complex AI agent objectives.

-----

</details>

<details>
<summary>How can procedural memory be implemented in AI agents to allow them to learn new skills from user interactions?</summary>

### Source [16]: https://research.aimultiple.com/ai-agent-memory/

Query: How can procedural memory be implemented in AI agents to allow them to learn new skills from user interactions?

Answer: **Procedural memory** in AI agents captures knowledge about how to carry out tasks and is typically implemented through functions, algorithms, or code dictating the agent’s behavior. This can range from simple routines, such as greeting users, to more advanced workflows for problem-solving. Unlike semantic memory (which handles what the agent knows), procedural memory focuses on applying that knowledge.

To enable learning from user interactions, procedural memory can be simulated using a data structure (e.g., a Python dictionary) to store the agent’s current instructions. When the agent receives feedback from a user, it updates these stored instructions accordingly. For example, if an agent is instructed to summarize a paper but receives feedback to "make it simpler," it modifies its instruction set to adopt a more casual tone. This update process allows the agent to "remember" and adapt its behavior based on user feedback for future interactions, thus mimicking the adaptive nature of procedural memory in humans.

The key steps are:
- Store initial procedural instructions.
- Execute tasks based on current instructions.
- Update instructions dynamically in response to user feedback.
- Use the updated instructions for subsequent tasks, allowing the agent to adapt and learn new skills over time[1].

-----

-----

-----

### Source [17]: https://arxiv.org/html/2411.00489v1

Query: How can procedural memory be implemented in AI agents to allow them to learn new skills from user interactions?

Answer: This survey analyzes AI long-term memory from the perspective of human memory theories, identifying **procedural memory** as a core component alongside episodic and semantic memory. It distinguishes between non-parametric and parametric memory systems in AI based on their storage forms and discusses the processing mechanisms and challenges.

A proposed Cognitive Architecture of Self-Adaptive Long-term Memory (SALM) integrates principles from human long-term memory to improve adaptability in AI agents. This architecture aims to overcome limitations in current cognitive systems, enabling agents to dynamically adjust their procedural routines and adapt to new tasks or changes through self-modification mechanisms. By grounding memory systems in human cognitive theory, SALM provides a framework for AI agents to refine their procedural knowledge through continual interaction, feedback, and self-guided learning, thus supporting the acquisition of new skills from user interactions[2].

-----

-----

-----

### Source [18]: https://arxiv.org/html/2505.03434v1

Query: How can procedural memory be implemented in AI agents to allow them to learn new skills from user interactions?

Answer: The paper argues that **Large Language Models (LLMs)** are highly proficient in procedural tasks due to their architecture, which mirrors human procedural memory—enabling the automation of repetitive, pattern-driven skills through practice. LLMs can execute consistent routines for text generation, code completion, and task orchestration.

However, the authors emphasize that procedural memory alone is insufficient for AI agents to adapt to complex, real-world environments where rules change and feedback can be ambiguous. To enable robust skill acquisition from user interactions, AI agents must be augmented with **semantic memory** and **associative learning systems**. The recommended solution is a modular architecture that separates procedural, semantic, and episodic memory functions. This separation enables the agent to:
- Retain and refine procedural knowledge for task execution.
- Integrate new factual or contextual information (semantic memory).
- Form associations between experiences, feedback, and actions (associative learning).

Such a modular system allows an agent to continuously adapt procedural routines based on user feedback, environmental changes, and novel situations, supporting the dynamic acquisition of new skills[3].

-----

-----

-----

### Source [19]: https://decodingml.substack.com/p/memory-the-secret-sauce-of-ai-agents

Query: How can procedural memory be implemented in AI agents to allow them to learn new skills from user interactions?

Answer: In practical implementations, **procedural memory** in AI agents is encoded directly into the agent’s structure—such as through graph-based models (e.g., LangGraph). All nodes, edges, tool integrations, prompts, and API calls collectively form the agent's procedural memory.

The agent's workflows, defined by its code and configuration (e.g., which tools to use, what actions to take in response to certain inputs), embody its procedural knowledge. To allow learning from user interactions, the agent's procedural routines can be modified or extended during operation. For example, as the agent receives feedback or new instructions, its underlying graph or workflow can be updated, enabling the agent to adapt its behavior and learn new skills dynamically without requiring a full retraining.

The system can support this adaptability by exposing interfaces for modifying the agent's workflow, integrating new nodes (skills), or changing the logic of existing routines in response to user feedback[4].

-----

-----

-----

### Source [20]: http://www.gocharlie.ai/blog/memory/

Query: How can procedural memory be implemented in AI agents to allow them to learn new skills from user interactions?

Answer: **Procedural memory** in LLM-based agents corresponds to the internal routines, processes, and algorithms that govern task execution. These routines allow agents to perform skills or sequences of actions automatically, paralleling the human process of skill acquisition through repetition and feedback.

For agents to learn new skills from user interactions, their procedural methods must be designed to be flexible and updatable. As agents interact with users and receive feedback (e.g., corrections, instructions for new procedures), they can refine or extend their routines. This enables LLM agents to:
- Navigate external APIs.
- Implement reasoning sequences.
- Adapt to new requirements by incorporating user-suggested changes.

The key is to structure internal processes so they are modular and amenable to change, allowing procedural memory to evolve in response to ongoing user interaction[5].

-----

</details>

<details>
<summary>What are real-world examples of product-driven AI memory architecture design?</summary>

### Source [21]: https://completeaitraining.com/news/ai-driven-memory-architecture-transforms-scientific/

Query: What are real-world examples of product-driven AI memory architecture design?

Answer: A real-world example of product-driven AI memory architecture design is **Crete**, developed collaboratively by the Department of Energy's Pacific Northwest National Laboratory (PNNL) and Micron. Crete is a hardware-software system specifically engineered for **AI-driven scientific computing** and features **15 terabytes of active memory** situated directly alongside system processors. This memory-centric design is unique within DOE labs and the broader high-performance computing community. It enables scientific applications that require far more memory than typical systems, which often prioritize processor speed over memory capacity.

Crete’s architecture leverages **Compute Express Link™ (CXL)** to connect memory to processors via an I/O switch, and it integrates both tightly coupled DRAM (Micron Registered Dual In-line Memory Modules) and loosely coupled memory (managed with custom Micron CXL controller boards). This hybrid architecture is tailored for scientific workloads that are **memory-bound**, supporting researchers who need to process and share large datasets in real time. The system’s memory capacity is equivalent to that of 240 high-end laptops (with 64 GB memory each) working together simultaneously.

Micron’s senior fellow, Mark Helm, highlights that Crete “redefines how memory and compute can collaborate to accelerate scientific discovery,” underlining the system’s role in enabling **memory-rich environments** essential for AI-driven research. This testbed is accessible to users in the DOE Office of Science national laboratories under the Advanced Memory to Support Artificial Intelligence for Science (AMAIS) initiative, focusing on applications limited by memory size, bandwidth, or sharing.

-----

-----

-----

### Source [22]: https://arxiv.org/pdf/2504.19413

Query: What are real-world examples of product-driven AI memory architecture design?

Answer: A real-world product architecture example is **Mem0**, a memory-centric system designed for **production-ready AI agents** with scalable long-term memory. Mem0 dynamically extracts, consolidates, and retrieves information from conversations, enabling efficient and context-aware agent behavior. The architecture maintains a **dense, natural-language-based memory** that encodes entire dialogue turns, resulting in an average footprint of 7,000 tokens per conversation. 

Its variant, Mem0g, adds **graph memories**—explicitly modeling nodes and relationships—to support more nuanced temporal and contextual integration, approximately doubling storage requirements to 14,000 tokens per conversation. For comparison, a competitor (Zep) uses a memory graph model that consumes over 600,000 tokens per conversation due to redundant storage of summaries and facts.

Mem0’s design achieves a balance between **response quality and computational efficiency**, making it suitable for production AI agents where both speed and accuracy are critical. Empirical analysis shows that while dense natural language memory is efficient for straightforward queries, explicit relational modeling (as in Mem0g) is essential for tasks requiring **deeper reasoning and contextual awareness**. The architecture is tailored to adapt its memory structures to specific reasoning contexts, addressing real-world deployment needs of AI agents in customer-facing and enterprise scenarios.

-----

-----

-----

### Source [23]: https://learn.microsoft.com/en-us/azure/architecture/ai-ml/

Query: What are real-world examples of product-driven AI memory architecture design?

Answer: A prominent product-driven AI memory architecture is found in **Microsoft Azure’s AI and machine learning platforms**, specifically leveraging **Apache Spark-based data platforms** for in-memory processing. Spark enables large-scale parallel processing by loading and caching data in memory, which significantly boosts performance for big data analytics and AI workloads compared to traditional disk-based approaches.

Within Azure, the **Fabric Runtime** integrates Apache Spark with **Delta Lake**, providing a robust foundation for scalable, reliable in-memory data processing. This architecture supports:
- **In-memory cluster computing** for rapid data access and repeated querying.
- **Delta Lake** for ACID-compliant, reliable storage and fast updates.
- Multi-language support (Java, Scala, Python, R), allowing diverse development environments.

Azure’s architecture is designed to be compatible with various hardware, ensuring that in-memory processing can scale for both AI training and inference workloads. The platform’s data connectors (via Azure Data Factory and Synapse Analytics) further support high-throughput, memory-centric data pipelines, enabling seamless integration of heterogeneous data sources for AI applications. This memory-first approach underpins many real-world AI products and services within the Microsoft ecosystem, where **fast, repeated access to large datasets** is critical for AI-driven analytics and machine learning.

-----

-----

-----

### Source [24]: https://engineering.salesforce.com/how-a-new-ai-architecture-unifies-1000-sources-and-100-million-rows-in-5-minutes/

Query: What are real-world examples of product-driven AI memory architecture design?

Answer: Salesforce presents a product-driven AI memory architecture in its **cross-cloud data unification system** for advertising analytics. The architecture is designed to support **querying and processing hundreds of millions to over a billion rows of data** from more than 1,000 sources with low latency. For metadata scalability, the system deploys over 1,000 Data Lake Objects (DLOs) per customer organization, pushing platform limits and requiring extensive performance tuning.

For data scalability, the architecture introduces **data partitioning** and **deduplication** during ingestion, as well as materializing common analytics patterns (such as pattern extraction and data classification) at load time, rather than computing them during queries. This **memory-optimized approach** minimizes query costs and maximizes responsiveness for real-time analytics.

To validate the memory and compute architecture, Salesforce developed synthetic data generators to simulate large-scale environments and benchmark the system under real-world loads. This rigorous testing ensured that the product could support very large enterprise customers from day one, demonstrating the architecture’s ability to deliver high-performance AI-driven analytics at scale. The combination of **memory-centric design, partitioning, and load-time optimization** exemplifies a real-world, product-driven AI memory architecture tailored for enterprise data and AI workloads.

-----

-----

</details>

<details>
<summary>What are the practical trade-offs when choosing between knowledge graphs and vector databases for an AI agent's long-term memory architecture?</summary>

### Source [25]: https://writer.com/engineering/vector-database-vs-graph-database/

Query: What are the practical trade-offs when choosing between knowledge graphs and vector databases for an AI agent's long-term memory architecture?

Answer: **Vector databases** are optimized for **fast searches** but typically lack the ability to encode rich context and semantic relationships between data points. This is a limitation when complex relationships or context are required, such as in enterprise-scale search and information retrieval tasks. In contrast, **knowledge graphs** are designed to preserve semantic relationships and encode structural information, making them well-suited for advanced search tasks where context and relationships are important. Knowledge graphs link concepts, entities, and their relationships using semantic descriptions, facilitating complex queries and providing more meaningful results.

A practical example demonstrates that knowledge graphs can store not just facts but also the relationships between entities, enabling AI agents to answer more nuanced or relational queries. For example, they can represent both "Steve Jobs founded Apple Inc." and "Steve Wozniak founded Apple Inc." as distinct, queryable relationships.

Key trade-offs identified:
- **Vector databases**: Fast, scalable search but limited relational/contextual understanding.
- **Knowledge graphs**: Rich context and relationship modeling, but can be more complex to implement and maintain, especially at scale.

-----

-----

-----

### Source [26]: https://neo4j.com/blog/genai/knowledge-graph-vs-vectordb-for-retrieval-augmented-generation/

Query: What are the practical trade-offs when choosing between knowledge graphs and vector databases for an AI agent's long-term memory architecture?

Answer: **Knowledge graphs** provide a **human-readable** and explainable structure for data, enabling precise and complete responses to queries. For instance, when querying for all books by an author, a knowledge graph returns exactly and only the relevant data due to explicit relationships. In contrast, a **vector database** relies on similarity scoring and result limits, often leading to incomplete, irrelevant, or noisy results—making it nearly impossible to guarantee exact answers.

Furthermore, knowledge graphs offer **consistent accuracy** and explainability by following the flow of connected information. Vector databases can sometimes produce incorrect inferences by associating facts based on similarity without regard to explicit relationships. This can result in plausible but inaccurate answers, especially in organizational or factual contexts.

In summary:
- **Knowledge graphs**: Precise, explainable, and accurate responses; ideal for queries requiring relationship awareness.
- **Vector databases**: Fast retrieval based on similarity but prone to incompleteness and inaccurate inferences due to lack of explicit structure.

-----

-----

-----

### Source [27]: https://www.falkordb.com/blog/knowledge-graph-vs-vector-database/

Query: What are the practical trade-offs when choosing between knowledge graphs and vector databases for an AI agent's long-term memory architecture?

Answer: Both **knowledge graphs** and **vector databases** share similarities in purpose (storing/querying relationships), scalability, integration with AI/ML, and typical use cases (recommendation systems, fraud detection, AI search, real-time analytics). However, their **retrieval quality** differs significantly.

**Knowledge graphs** are superior for generating **factually correct and precise responses** because they allow traversal through verified nodes and relationships. This is critical for applications where accuracy is paramount, such as in enterprise knowledge management.

**Vector databases** excel at handling **massive unstructured data** and provide fast, similarity-based retrieval, but the use of similarity metrics can introduce noise and reduce precision. This makes them less reliable for tasks that require deterministic, complete, and explainable retrieval.

A hybrid approach, combining both, can yield high accuracy and completeness, but this integration increases complexity and maintenance burden.

-----

-----

-----

### Source [28]: https://www.useparagon.com/blog/vector-database-vs-knowledge-graphs-for-rag

Query: What are the practical trade-offs when choosing between knowledge graphs and vector databases for an AI agent's long-term memory architecture?

Answer: **Vector databases** are favored for their **flexibility**, support for various data types, lower barrier to entry, and reduced maintenance costs. They are especially strong in handling unstructured data and performing semantic searches, benefiting from extensive industry and community support.

**Knowledge graphs**, on the other hand, are suited for applications requiring **structured understanding of complex relationships**. They are preferred when complex queries, structural queries, or strict relationship mapping are necessary.

A **hybrid approach** leverages both: using knowledge graphs for structured queries and vector databases for unstructured data. This is particularly effective for large or diverse datasets and complex, multifaceted queries. However, hybrids introduce significant complexity, as maintaining synchronization and managing both systems can be costly and slow down retrieval due to the need to execute two types of searches.

-----

-----

-----

### Source [29]: https://www.elastic.co/blog/vector-database-vs-graph-database

Query: What are the practical trade-offs when choosing between knowledge graphs and vector databases for an AI agent's long-term memory architecture?

Answer: **Vector databases** structure data as **points in a multi-dimensional space**, focusing on capturing **similarity** between data points. They are ideal for content similarity tasks such as image or document retrieval, where explicit relationships are less important. Their optimized algorithms allow for robust performance and scalability with large datasets, but changes to the schema may require costly re-embeddings.

**Graph databases** (including knowledge graphs) structure data as **nodes and relationships**, emphasizing the connections and hierarchies between entities. They excel at traversing network structures and supporting relationship-rich queries, such as those needed for social networks or complex knowledge graphs. Their schema-less nature favors flexibility, but complex queries or very large networks can strain performance, requiring careful optimization.

Key trade-offs:
- **Vector database**: Best for unstructured, similarity-based retrieval; scalable, but with limited relationship modeling.
- **Graph/Knowledge graph**: Best for rich, relationship-driven queries; flexible schema, but may require optimization for performance at scale.

-----

</details>

<details>
<summary>How have large context windows in models like Gemini 2.5 Pro impacted the design of memory retrieval and compression pipelines for AI agents?</summary>

### Source [30]: https://www.dhiwise.com/post/the-gemini-context-window-and-its-role-in-ai-precision

Query: How have large context windows in models like Gemini 2.5 Pro impacted the design of memory retrieval and compression pipelines for AI agents?

Answer: Large context windows in models such as Gemini 1.5 Pro (and by extension, Gemini 2.5 Pro) have fundamentally changed how memory retrieval and compression pipelines are designed for AI agents. Previously, smaller context windows (8K–32K tokens) forced designers to truncate or chunk large inputs, often resulting in loss of essential information and reduced output quality. With Gemini's support for up to 1 million tokens—and ongoing tests for even larger windows—AI agents can now process entire books, codebases, or lengthy transcripts in a single prompt. This enables accurate and complete reasoning over vast, unbroken datasets across modalities (text, audio, video). The need for aggressive data compression or selective retrieval diminishes, as agents can remember, analyze, and reason over much larger inputs directly. Consequently, memory pipelines now focus more on efficient loading, context caching (to reduce cost/latency), and leveraging the model’s many-shot learning capabilities, rather than elaborate compression or pre-selection schemes[1].

-----

-----

-----

### Source [31]: https://codingscape.com/blog/llms-with-largest-context-windows

Query: How have large context windows in models like Gemini 2.5 Pro impacted the design of memory retrieval and compression pipelines for AI agents?

Answer: LLMs with massive context windows (like Gemini 2.5 Pro with up to 1 million tokens) have enabled use cases previously impossible or highly constrained. These include ultra-long codebase comprehension, legal document analysis over thousands of pages, and full-book summarization. The scale reduces the need for sophisticated input compression and retrieval strategies, as more raw data can be passed directly to the model. Instead, pipelines now focus on the technical challenges of efficiently streaming, managing, and segmenting very large datasets for processing. This shift also allows more natural workflows—such as multimodal document analysis and automated refactoring—because the context window is no longer a bottleneck[2].

-----

-----

-----

### Source [32]: https://ai.google.dev/gemini-api/docs/long-context

Query: How have large context windows in models like Gemini 2.5 Pro impacted the design of memory retrieval and compression pipelines for AI agents?

Answer: Google’s official Gemini documentation explains that context windows of 1 million tokens or more fundamentally change developer paradigms and pipeline design. Historically, limited context forced developers to design complex systems for information selection, truncation, and compression before passing data to the model. With long context, these constraints are greatly reduced: existing code for text generation or multimodal inputs works without modification, and developers can pass much larger chunks of information in a single call. This enables new use cases and simplifies the architecture of memory retrieval and compression, as less preprocessing is needed and more data can be held within the model’s “short term memory” for reasoning and generation[3].

-----

-----

-----

### Source [33]: https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro

Query: How have large context windows in models like Gemini 2.5 Pro impacted the design of memory retrieval and compression pipelines for AI agents?

Answer: Gemini 2.5 Pro is designed to comprehend vast datasets and tackle challenging problems by integrating information from text, audio, images, video, and entire code repositories. The platform supports large input size limits (up to 500 MB for some modalities), meaning memory retrieval and compression pipelines can now aggregate much more raw data from multiple sources and pass it as a single prompt. This reduces the need for segmenting or aggressively compressing datasets prior to processing. Instead, the focus shifts to ensuring efficient loading and multi-format support within the pipeline, as the model’s capacity allows for richer, more holistic analysis and reasoning over complex, multimodal data[4].

-----

-----

-----

### Source [34]: https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf

Query: How have large context windows in models like Gemini 2.5 Pro impacted the design of memory retrieval and compression pipelines for AI agents?

Answer: According to the Gemini 2.5 technical report, the model features the largest context window to date—up to 2 million tokens. This capacity allows AI agents to handle much larger conversational threads, codebases, or document collections without loss of continuity or context. Performance benchmarks show that Gemini 2.5 Pro surpasses previous models in tasks requiring deep reasoning over extended input, thanks to its ability to process more information at once. For memory retrieval and compression pipelines, this means less need for aggressive filtering or summarization; systems can present far more context directly to the model, relying on its internal attention mechanisms to extract relevant details for output[5].
-----

-----

</details>

<details>
<summary>What are best practices for designing an AI agent's memory architecture based on specific product requirements, such as a Q&A bot versus a personal companion?</summary>

### Source [35]: https://www.speakeasy.com/mcp/ai-agents/architecture-patterns

Query: What are best practices for designing an AI agent's memory architecture based on specific product requirements, such as a Q&A bot versus a personal companion?

Answer: Designing an AI agent’s memory architecture should be dictated by the product’s needs for **context retention** and **external interactions**. For agents requiring awareness of prior events (like reminders or companions), the **memory-augmented agent pattern** is recommended, where memory stores—such as vector databases—retain past interactions or user data and are queried as context for decision-making. This enables agents to provide personalized responses and persistent experiences. For agents interacting with external systems (e.g., Q&A bots accessing APIs), the **tool-using agent pattern** separates core agent logic from external tool/API handling via a standardized protocol layer. This maintains clean architecture, where memory may be used to track recent interactions, but the agent leverages external systems for up-to-date answers. The choice and integration of memory patterns (contextual recall versus external lookup) should be based on product function and user experience requirements.

-----

-----

-----

### Source [36]: https://www.youtube.com/watch?v=W2HVdB4Jbjs

Query: What are best practices for designing an AI agent's memory architecture based on specific product requirements, such as a Q&A bot versus a personal companion?

Answer: Best practices in AI agent memory architecture draw inspiration from human memory systems—incorporating **episodic, working, semantic, and procedural memory** types to create context-aware, reliable agents. Core design involves distinct **memory components** (for conversation, workflow, episodic data, and persona) and **memory modes** (short-term, long-term, dynamic). For a Q&A bot, focus is placed on **maintaining rich conversation history** and context, using vector databases and relevance scoring for efficient retrieval. For a personal companion, memory should capture nuanced, long-term user preferences, persona details, and past experiences, with advanced strategies such as **memory cascading**, **selective deletion**, and **persona integration**. Production systems require scalable memory practices—persisting context, optimizing retrieval for LLM context window limits, and supporting multi-agent memory sharing. The ultimate goal is for the agent to **remember, adapt, and improve** over time, with memory as a central design asset.

-----

-----

-----

### Source [37]: https://www.lindy.ai/blog/ai-agent-architecture

Query: What are best practices for designing an AI agent's memory architecture based on specific product requirements, such as a Q&A bot versus a personal companion?

Answer: Memory architecture in AI agents demands a combination of **working memory** (short-term, session-specific) and **persistent memory** (long-term, cross-session continuity). Q&A bots typically rely on working memory to maintain session context, but agents providing continuity (like personal companions) require persistent memory to recall past interactions and preferences. Persistent memory is often implemented via **vector databases** that store interaction embeddings, enabling semantic retrieval for loosely related contexts. Frameworks like LangChain facilitate memory and retrieval management, but advanced architectures (e.g., Lindy’s Societies) allow **memory sharing across multiple agents**, supporting workflows that span multiple tasks or stages. In business settings, robust memory ensures agents behave consistently and represent the brand accurately, never starting “from scratch.”

-----

-----

-----

### Source [38]: https://www.jit.io/resources/devsecops/its-not-magic-its-memory-how-to-architect-short-term-memory-for-agentic-ai

Query: What are best practices for designing an AI agent's memory architecture based on specific product requirements, such as a Q&A bot versus a personal companion?

Answer: Short-term memory architecture is vital for coherent user experiences and scalable agent workflows. Jit’s agentic platform uses thread-based context management—each conversation or workflow is associated with a unique thread ID, and agent workflows are modeled as directed graphs. This enables the system to maintain scoped context, isolate memory per session, and support complex task delegation. A **checkpointer** saves every step (messages, transitions, agent state) for replayability and recovery. Memory state is a shared object containing not just messages but also structured metadata and contextual data, scoped to each thread for isolation and persistence. A **supervisor pattern** delegates tasks to downstream agents, ensuring clean logic and modular memory handling. These practices enable both Q&A bots and personal companions to maintain relevant context, scale across users, and recover from interruptions.

-----

-----

-----

### Source [39]: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/

Query: What are best practices for designing an AI agent's memory architecture based on specific product requirements, such as a Q&A bot versus a personal companion?

Answer: Effective agent memory architecture distinguishes between **short-term memory** (context within a session or multi-step workflow) and **long-term memory** (recall of information from previous interactions). LangGraph enables granular control, letting you define the **state schema**—specifying exactly what information is retained. The **checkpointer** persists state at every step, supporting context propagation and recovery. The **store** persists user or application-level data across sessions. This flexibility means you can tailor memory retention and retrieval to the needs of a Q&A bot (emphasizing immediate context and recent queries) or a personal companion (emphasizing continuity, preferences, and long-term user history). Effective memory management thus enhances agent context-awareness, learning, and decision-making by leveraging structured, persistent state across interactions.
-----

-----

</details>

<details>
<summary>How can procedural memory be implemented in AI agents to allow them to learn new multi-step tasks from user feedback or interaction history?</summary>

### Source [40]: https://langchain-ai.github.io/langgraph/concepts/memory/

Query: How can procedural memory be implemented in AI agents to allow them to learn new multi-step tasks from user feedback or interaction history?

Answer: Procedural memory in AI agents is described as a combination of **model weights, agent code, and prompts** that collectively determine the agent's functionality. While modifying model weights or rewriting code is rare in practical systems, agents more commonly update their prompts to adapt their behavior.

A key implementation strategy is the use of **"Reflection" or meta-prompting**: the agent is provided with its current instructions and recent user interactions or explicit feedback, prompting it to refine its own instructions. This approach is especially effective for tasks where initial instructions are hard to specify, allowing the agent to evolve its procedural knowledge from experience. For example, an agent tasked with generating high-quality paper summaries for social media can iteratively improve its summarization process by incorporating user feedback and rewriting its prompts accordingly.

This form of prompt-based procedural memory enables agents to adapt to new multi-step tasks through interaction history and user feedback, even without altering underlying code or model parameters.

-----

-----

-----

### Source [41]: https://www.youtube.com/watch?v=WW-v5mO2P7w

Query: How can procedural memory be implemented in AI agents to allow them to learn new multi-step tasks from user feedback or interaction history?

Answer: The tutorial demonstrates **dynamic instruction learning** in LLM agents using the LangMem SDK. The approach involves multiple agents (e.g., an email agent, a Twitter agent) whose procedural memories are kept distinct by using separate memory keys for their instructions.

A **supervisor agent** is used to manage these sub-agents. This supervisor is responsible for routing tasks to the appropriate agent and can aid in updating procedural memory by ensuring each agent's instructions are isolated and can be evolved independently as they interact with users. Through repeated interactions and feedback, agents can adjust their internal prompts or instructions, thus improving their procedural memory related to multi-step tasks.

The tutorial emphasizes the **importance of separating procedural memory for different agents** and showcases how supervisors can coordinate and update these memories based on interaction histories, facilitating continual learning from user feedback.

-----

-----

-----

### Source [42]: https://decodingml.substack.com/p/memory-the-secret-sauce-of-ai-agents

Query: How can procedural memory be implemented in AI agents to allow them to learn new multi-step tasks from user feedback or interaction history?

Answer: Procedural memory in AI agents is implemented as **functions, algorithms, or code** that define how the agent should act in various situations, from simple greeting templates to complex multi-step reasoning processes.

This procedural knowledge determines the agent's ability to apply what it knows (semantic memory) to execute tasks. To support learning new multi-step tasks from user feedback or interaction history, agents can update or augment these procedural components based on observed patterns and outcomes.

The article notes that procedural memory is typically *long-term* and forms the basis for action, while **episodic memory** allows the agent to recall and learn from specific past experiences—such as previous user interactions. By integrating episodic memory (e.g., conversation histories) with procedural memory, agents can iterate on and improve their task execution logic, effectively learning from user feedback and adapting their procedures over time.

-----

-----

-----

### Source [43]: https://blog.langchain.com/memory-for-agents/

Query: How can procedural memory be implemented in AI agents to allow them to learn new multi-step tasks from user feedback or interaction history?

Answer: Procedural memory is defined as the agent’s **long-term memory for how to perform tasks**, closely linked to the core instruction set (comparable to human skills like riding a bike). In AI agents, procedural memory is seen as the combination of **LLM weights and agent code**, which fundamentally shape the agent’s behavior.

In practical AI systems, it is uncommon for agents to update their model weights or rewrite their code autonomously. Instead, the most prevalent approach is **updating the system prompt**, which acts as a proxy for procedural memory. This allows agents to adapt their task execution strategies in response to feedback or new requirements, even if underlying weights and code remain static.

Despite its utility, prompt updating as a form of procedural memory adaptation is not yet widespread, but it represents the current state-of-the-art for allowing agents to learn and refine new multi-step tasks from user interaction and feedback.

-----

-----

-----

### Source [44]: https://arya.ai/blog/why-memory-matters-for-ai-agents-insights-from-nikolay-penkov

Query: How can procedural memory be implemented in AI agents to allow them to learn new multi-step tasks from user feedback or interaction history?

Answer: Procedural memory in AI agents covers **learned behaviors and patterns**, typically acquired through **reinforcement learning** or repeated training on tasks. This memory type enables agents to internalize step-by-step procedures—such as identity verification in chatbots or risk analysis in credit-scoring models.

By continuously interacting with users and receiving feedback, agents can reinforce and refine these procedures, allowing them to improve at multi-step tasks over time. This learning often occurs via reinforcement learning frameworks, where success and failure signals from user interactions inform the agent’s future behaviors.

Procedural memory complements other memory types (semantic and episodic) to provide context retention, predictive capabilities, and personalization. This enables agents to adapt their task execution dynamically, based on accumulated user feedback and interaction history, ultimately supporting more effective learning of new, complex processes.

-----

</details>

<details>
<summary>What are effective strategies for an AI agent to autonomously manage its memory, including resolving conflicting information and deciding what to forget, without requiring user intervention?</summary>

### Source [45]: https://arxiv.org/html/2506.06326

Query: What are effective strategies for an AI agent to autonomously manage its memory, including resolving conflicting information and deciding what to forget, without requiring user intervention?

Answer: The MemoryOS framework introduces a systematic approach to memory management for AI agents, inspired by operating system mechanisms. It consists of four core modules: **Storage, Updating, Retrieval, and Generation**.  
- **Memory Storage** is organized hierarchically into short-term, mid-term, and long-term units, allowing the agent to categorize information based on relevance and expected future use.  
- **Memory Updating** employs a segmented paging architecture and heat-based mechanisms. Segmented paging enables dynamic refreshing of memory blocks, while heat-based mechanisms prioritize frequently used or accessed information, ensuring that high-utility data remains and low-utility data can be forgotten autonomously.  
- **Memory Retrieval** uses semantic segmentation to efficiently query memory tiers, allowing the agent to resolve conflicting information by retrieving contextually relevant data.  
- **Response Generation** integrates retrieved memory for coherent and personalized outputs, maintaining long-term conversational coherence and persona consistency.  
This holistic system enables agents to autonomously manage memory, resolve conflicts, and decide what to forget based on usage patterns and contextual relevance, without requiring user intervention.

-----

-----

-----

### Source [46]: https://blog.savantly.net/intelligent-memory-management-for-ai-agents/

Query: What are effective strategies for an AI agent to autonomously manage its memory, including resolving conflicting information and deciding what to forget, without requiring user intervention?

Answer: Headkey is presented as an open-source intelligent memory management system for AI agents, addressing the problem of either storing too much, too little, or only summaries.  
- **Storing everything** overloads the agent and dilutes relevance, slowing performance.  
- **Storing nothing** leads to lost context and requires users to repeat themselves.  
- **Storing summaries** risks losing critical details.  
Headkey aims for **precise, context-aware memory handling**, enabling agents to autonomously decide what to retain or forget by evaluating the relevance and importance of information in context.  
Strategies highlighted include:  
- **Contextual relevance evaluation**: Information is retained or discarded based on its relevance to ongoing tasks and user preferences.  
- **Autonomous conflict resolution**: The agent can detect and manage conflicting information by prioritizing more recent or higher-confidence data.  
- **Dynamic adjustment**: The system can adjust retention policies in real-time based on observed usage patterns and evolving context, minimizing user intervention.

-----

-----

-----

### Source [48]: https://www.mongodb.com/company/blog/technical/dont-just-build-agents-build-memory-augmented-ai-agents

Query: What are effective strategies for an AI agent to autonomously manage its memory, including resolving conflicting information and deciding what to forget, without requiring user intervention?

Answer: Anthropic and Cognition emphasize **memory augmentation strategies** for reliable agent operation.  
- **Context compression**: Agents summarize completed tasks and store only essential information, enabling efficient memory use and autonomous conflict resolution by reducing redundancy and focusing on key facts.  
- **External memory offloading**: Agents can offload summarized or less relevant information to external storage, freeing up working memory and enabling autonomous forgetting based on context and importance.  
- **Context handoffs**: In multi-agent systems, context is transferred efficiently between agents, allowing each agent to autonomously decide what information is essential for its task and what can be forgotten.  
- **Retrieval-augmented generation (RAG) optimization**: Agents use optimized retrieval processes to select relevant memory fragments, resolving conflicts via selection of the most contextually appropriate information.  
These design patterns enable agents to autonomously manage memory, resolve conflicts, and decide what to forget, guided by relevance and task requirements.

-----

-----

-----

### Source [49]: https://dev.to/bredmond1019/building-intelligent-ai-agents-with-memory-a-complete-guide-5gnk

Query: What are effective strategies for an AI agent to autonomously manage its memory, including resolving conflicting information and deciding what to forget, without requiring user intervention?

Answer: The guide emphasizes the importance of memory for intelligence in AI agents, outlining foundational strategies for autonomous memory management:  
- **Contextual recall and learning**: Agents should prioritize the retention of information that is frequently recalled or built upon, autonomously forgetting less relevant details over time.  
- **Experience-based adaptation**: By tracking user interactions and feedback, agents can learn which details are important and autonomously adjust their retention and forgetting strategies.  
- **Maintaining context across interactions**: Agents use mechanisms to maintain context from previous sessions and autonomously resolve conflicting information by referencing the most recent or authoritative data.  
- **Relationship-building and failure learning**: Agents can autonomously update memory by learning from failures and successes, deciding what to forget based on outcome relevance instead of explicit user instruction.  
These principles align with human-like memory management, enabling agents to resolve conflicts, adaptively forget, and maintain relevant context without user intervention.

-----

-----

### Source [50]: https://arxiv.org/html/2506.06326

Query: What are effective strategies for an AI agent to autonomously manage its memory, including resolving conflicting information and deciding what to forget, without requiring user intervention?

Answer: The MemoryOS system proposes a comprehensive solution for autonomous AI memory management, designed similarly to operating system memory management. It comprises four synergistic modules: **Storage, Updating, Retrieval, and Generation**.  
- **Memory Storage**: Organizes information into hierarchical layers—short-term, mid-term, and long-term memory units—enabling context-sensitive retention and forgetting.  
- **Memory Updating**: Uses a segmented paging architecture and heat-based mechanisms, dynamically refreshing memory based on dialogue chains and usage frequency. This prevents memory overload and prioritizes relevant information by "cooling off" less-used segments for potential forgetting.  
- **Memory Retrieval**: Employs semantic segmentation to efficiently query information across memory tiers, ensuring relevant knowledge is surfaced for decision-making or conversational continuity.  
- **Response Generation**: Integrates retrieved memory to provide coherent, contextually personalized outputs, maintaining user persona and continuity.

This system autonomously resolves conflicting information by updating memory segments according to recency, relevance, and user preferences captured over time. Decisions on what to forget are made via heat-based mechanisms that de-prioritize rarely accessed or outdated information, allowing the agent to autonomously prune its memory without user intervention. Experimental results demonstrate MemoryOS’s effectiveness in sustaining long-term coherence and response correctness during extended interactions.

-----

-----

-----

### Source [51]: https://blog.savantly.net/intelligent-memory-management-for-ai-agents/

Query: What are effective strategies for an AI agent to autonomously manage its memory, including resolving conflicting information and deciding what to forget, without requiring user intervention?

Answer: The Headkey system addresses the "memory crisis" in AI by enabling autonomous, intelligent memory management for agents.  
- **Three memory handling strategies** are identified: storing everything (leading to overload), storing nothing (losing context), and storing summaries (potentially missing critical details).  
- Headkey aims for **precise, context-aware memory handling**. It autonomously determines what information to retain, recall, or discard based on ongoing relevance and interaction context.  
- The system is designed to avoid both cognitive overload (by not retaining all information) and context loss (by not discarding too much).  
- Autonomous decision-making is based on the agent’s assessment of which facts or memories are essential for current or future tasks, enabling the agent to resolve conflicts by keeping the most relevant, recent, or validated information and discarding outdated or less pertinent memories.

This approach allows AI agents to manage memory without user intervention, ensuring that only the most useful information is retained and that outdated or conflicting data is systematically forgotten or deprioritized.

-----

-----

-----

### Source [53]: https://dev.to/bredmond1019/building-intelligent-ai-agents-with-memory-a-complete-guide-5gnk

Query: What are effective strategies for an AI agent to autonomously manage its memory, including resolving conflicting information and deciding what to forget, without requiring user intervention?

Answer: This guide emphasizes the importance of **stateful memory systems** for intelligent, autonomous AI agents.  
- Stateless systems fail to maintain context, leading to repetitive or irrelevant responses; stateful agents use memory to recall preferences, learn from interactions, and adapt over time.
- Key capabilities for autonomous memory management include:
  - *Relevant recall*: Agents autonomously retrieve pertinent information when needed.
  - *Learning and adaptation*: Continuous updates based on experience, enabling agents to resolve conflicts and prioritize new knowledge over outdated or contradictory information.
  - *Context maintenance*: Sustained understanding across interactions without repeated user input.
- The evolution towards **memory-enabled agents** enables relationship building and ongoing learning, with systems designed to autonomously decide what information is retained or forgotten based on user interactions and internal priorities.

The guide suggests that intelligent memory management leverages autonomous updating, relevance assessment, and context-driven decision-making to resolve conflicting information and manage forgetting, all without requiring user intervention.

-----

-----

-----

### Source [54]: https://www.vincirufus.com/posts/memory-based-agent-learning/

Query: What are effective strategies for an AI agent to autonomously manage its memory, including resolving conflicting information and deciding what to forget, without requiring user intervention?

Answer: The Memento framework introduces a **memory-based learning system** for AI agents, focusing on continuous improvement and autonomous memory management.  
- Memento enables agents to **store, retrieve, and update memories** independently, learning from experience without explicit user guidance.
- The framework uses algorithms to evaluate the **relevance and accuracy** of stored information, autonomously resolving conflicts by prioritizing reliable or more recent data and discarding outdated or incorrect memories.
- Agents make forgetting decisions by monitoring usage patterns and context, ensuring that only essential or frequently accessed information is retained.
- This approach supports robust autonomy, allowing agents to optimize memory composition, resolve contradictions, and maintain high performance in dynamic environments.

Memento’s memory-based learning empowers agents to autonomously curate and manage their knowledge base, continuously refining memory through relevance filtering, conflict resolution, and intelligent forgetting mechanisms.

-----

</details>

<details>
<summary>What are the cognitive and biological inspirations behind the classification of AI agent memory into semantic, episodic, and procedural types?</summary>

### Source [55]: https://connectai.blog/agents-memory

Query: What are the cognitive and biological inspirations behind the classification of AI agent memory into semantic, episodic, and procedural types?

Answer: AI agent memory classification into **semantic, episodic, and procedural types** is directly inspired by models of human memory from cognitive psychology and neuroscience.

- **Semantic memory** in humans refers to storage of facts and general knowledge, like “Paris is the capital of France.” AI agents mirror this by storing factual user or world information, such as user preferences or domain facts, enabling contextually accurate and personalized responses. For example, an agent might remember a user’s favorite cuisine or birthday for tailored recommendations.
  
- **Episodic memory** parallels human recollection of specific experiences (such as your first day at work). In AI, this means remembering past dialogues, problem-solving attempts, or previous interactions—essentially, the agent’s own “experiences.” This supports continuity, learning from successes and failures, and referencing prior events during interactions.

- **Procedural memory** is not detailed in this source, but the analogy follows human memory, in which procedural memory handles how to do things (skills, habits, routines). In AI, this would map to learned routines, algorithms, or task procedures.

The implementation of these memory types in AI is informed by the cognitive distinctions found in human memory: facts (semantic), events (episodic), and skills (procedural)[1].

-----

-----

-----

### Source [56]: https://decodingml.substack.com/p/memory-the-secret-sauce-of-ai-agents

Query: What are the cognitive and biological inspirations behind the classification of AI agent memory into semantic, episodic, and procedural types?

Answer: The classification of AI agent memory draws from **human cognitive psychology**, where memory is segmented into **semantic, episodic, and procedural** categories.

- **Procedural memory** in humans involves knowledge of how to perform tasks or follow processes. In AI, this is implemented as functions, algorithms, or code that define the agent’s actions in various situations. It stores *how* to apply knowledge, not just *what* is known.

- **Episodic memory** is modeled after human autobiographical memory, which retains specific past experiences. For AI, this enables recall of past user interactions and learning from those events, supporting continuity and context-awareness in conversations.

- **Semantic memory** (referenced implicitly) stores facts and general knowledge, much like in humans.

The **biological inspiration** comes from observing that humans segment memory based on function—facts, events, and skills—and this partitioning helps AI agents manage context, continuity, and task performance more effectively. Implementations often follow retrieval-augmented generation (RAG) paradigms or memory extraction from conversation histories, mirroring the selective retrieval seen in human cognition[2].

-----

-----

-----

### Source [57]: https://langchain-ai.github.io/langgraph/concepts/memory/

Query: What are the cognitive and biological inspirations behind the classification of AI agent memory into semantic, episodic, and procedural types?

Answer: The **division of AI agent memory** into semantic, episodic, and procedural types is explicitly mapped from **human memory systems**, as described in cognitive science research (such as the CoALA paper).

- **Semantic memory** stores facts—mirroring things learned in school for humans, and facts about users or domains for agents.
  
- **Episodic memory** handles experiences—human autobiographical events or an agent’s past actions.
  
- **Procedural memory** involves the rules or instructions required to execute tasks—motor skills in humans, and system prompts or code in agents.

In humans, **procedural memory** is implicit and involves skills like riding a bike. In AI, it is represented by the agent’s code, model weights, or prompts that determine behavior. Adaptation or learning in AI procedural memory can be achieved through techniques like “meta-prompting,” where agents update their instructions based on feedback, reflecting how humans refine skills through experience.

The analogy is not perfect, but using this cognitive classification provides a practical framework for designing agent memory, inspired by how biological systems segment and use different types of memory[3].

-----

-----

-----

### Source [58]: https://www.marktechpost.com/2025/03/30/understanding-ai-agent-memory-building-blocks-for-intelligent-systems/

Query: What are the cognitive and biological inspirations behind the classification of AI agent memory into semantic, episodic, and procedural types?

Answer: This source introduces the **four key types of memory** in AI agents—episodic, semantic, procedural, and short-term (working) memory—grounded in **cognitive neuroscience**.

- **Episodic memory**: Inspired by the human capacity to remember specific events or experiences, AI agents use episodic memory to recall distinct interactions or events, enhancing their ability to personalize responses and maintain context across sessions.

- **Semantic memory**: Analogous to human factual memory, AI agents store general knowledge or facts, supporting accurate and informative outputs.

- **Procedural memory**: Modeled after human procedural learning (skills and habits), this allows agents to internalize routines or algorithms, supporting consistent task execution.

The classification draws from established models in psychology and neuroscience, where long-term human memory is divided into episodic (events), semantic (facts), and procedural (skills), and this framework is adopted to structure agent memory for more robust and human-like intelligence[4].

-----

-----

-----

### Source [59]: https://arya.ai/blog/why-memory-matters-for-ai-agents-insights-from-nikolay-penkov

Query: What are the cognitive and biological inspirations behind the classification of AI agent memory into semantic, episodic, and procedural types?

Answer: The **three main types of memory in AI agents—procedural, semantic, and episodic—are modeled after human memory systems** as understood in neuroscience and psychology.

- **Procedural memory** in AI represents learned behaviors and patterns, developed through reinforcement learning or repeated task training, analogous to how humans acquire skills through practice.
  
- **Semantic memory** stores general knowledge and facts, similar to human long-term memory for concepts, rules, and world knowledge.

- **Episodic memory** captures specific events or experiences, just as in humans who recall the “who, what, where, when, and why” of past occurrences.

This biologically-inspired classification enables AI agents to retain context, recognize patterns, and personalize interactions, striving for more “human-like” intelligence and adaptability[5].

-----

-----

</details>

<details>
<summary>How is episodic memory implemented in AI agents to provide nuanced, long-term conversational context beyond simple fact retrieval?</summary>

### Source [65]: https://www.geeksforgeeks.org/artificial-intelligence/episodic-memory-in-ai-agents/

Query: How is episodic memory implemented in AI agents to provide nuanced, long-term conversational context beyond simple fact retrieval?

Answer: **Episodic memory in AI agents** enables them to store, recall, and reason about past experiences or events they have personally encountered, in a manner analogous to human memory for distinct life events. This goes beyond immediate input processing and static fact retrieval by providing a **long-term archive** of specific user interactions, agent actions, and environmental events.

Key features of episodic memory implementation in AI agents include:
- **Recalling specific past events** relevant to current decision-making.
- **Learning from previous successes and failures** to adjust future actions.
- **Pattern detection** over time to enhance performance.
- **Providing explanations** by referencing prior experiences.
- **Maintaining a coherent experience history** for long-term planning and context continuity.

Episodic memory thus elevates agent intelligence from merely reactive to reflective, allowing them to use historical context dynamically for improved personalization, adaptability, and decision making in complex environments. This nuanced context retention is enabled by the agent’s ability to access and reason about a persistent record of its own experiences, rather than just relying on short-lived working memory[1].

-----

-----

-----

### Source [66]: https://www.digitalocean.com/community/tutorials/episodic-memory-in-ai

Query: How is episodic memory implemented in AI agents to provide nuanced, long-term conversational context beyond simple fact retrieval?

Answer: **Episodic memory for AI agents** is typically implemented as a **memory module** that is closely integrated with the agent’s decision-making logic. This memory captures not only the events the agent experiences but also their contextual metadata—such as time, location, and observed outcomes.

- **Episodic memory** differs from semantic memory in that it records *specific contextual events* rather than general facts.
- In **reinforcement learning agents**, episodic memory stores sequences of actions, states, and rewards, allowing the agent to recall and learn from specific episodes (successes or failures). This supports more nuanced decision-making when similar situations recur.

For **large language models (LLMs)** like GPT-3 or GPT-4, episodic memory is not natively persistent between sessions, but can be simulated by augmenting these systems with *external memory modules* that store and retrieve past user interactions. This workaround allows the model to provide conversational continuity and recall nuanced, long-term context otherwise not possible with vanilla architectures[2].

-----

-----

-----

### Source [67]: https://arxiv.org/html/2501.11739v1

Query: How is episodic memory implemented in AI agents to provide nuanced, long-term conversational context beyond simple fact retrieval?

Answer: This source discusses both the **benefits and risks** of implementing rich episodic memory in AI agents. Episodic memory, inspired by human memory, enables agents to use past experiences not just for recall, but for a variety of advanced cognitive functions, including improved situational awareness, explanation, monitoring, and adaptability.

Key points include:
- **Episodic memory enables richer reasoning** far beyond simple fact retrieval, supporting learning, explanation, and context-sensitive responses.
- **Risks** arise, such as the potential for deception, unpredictable retention of knowledge, and increased situational awareness, which could impact control and oversight.
- For safety and reliability, the authors propose four implementation principles:
  1. **User interpretability:** Memories should be understandable by users.
  2. **User control:** Users should be able to add or delete memories.
  3. **Memory modularity:** Memories should be isolatable from the system.
  4. **Agent immutability:** Memories should not be editable by the AI agent itself.

These principles are suggested to ensure that episodic memory enhances AI agent performance while maintaining user oversight, control, and safety. The paper encourages further research into architectures and safeguards for episodic memory in AI[3].

-----

-----

-----

### Source [68]: https://www.cs.columbia.edu/~dechant/safeaiworkshop2023.pdf

Query: How is episodic memory implemented in AI agents to provide nuanced, long-term conversational context beyond simple fact retrieval?

Answer: This work focuses on **interpretability and safety** in episodic memory for AI agents. It advocates that memories should be either directly or indirectly interpretable by humans. For example, direct interpretability could involve storing memories in formats such as video, images, or natural language summaries.

- **Direct memories** (e.g., raw video, text) are intuitive but can be impractical due to size and searchability.
- **Compressed or summarized representations** are more practical, provided they remain reliably interpretable and can yield information relevant to user monitoring or safety needs.
- Systems may use *natural language summaries* or event logs to encapsulate the most crucial parts of each episode.

The goal is to enable AI agents to retain rich, long-term context for nuanced reasoning while ensuring that this context remains accessible and understandable to human users for oversight and explanation[4].

-----

-----

-----

### Source [69]: https://deepblue.lib.umich.edu/bitstream/handle/2027.42/57720/anuxoll_1.pdf

Query: How is episodic memory implemented in AI agents to provide nuanced, long-term conversational context beyond simple fact retrieval?

Answer: This dissertation emphasizes the practical **criteria for implementing episodic memory** in intelligent agents:

- **Demonstrable improvement:** Agents equipped with episodic memory should outperform those without, across various tasks, by leveraging stored episodes for better action selection and adaptation.
- **Architecture and task independence:** The episodic memory system should be generalizable, functioning across different agent architectures and domains without major modifications.
- **Resource efficiency:** The memory system must be sustainable for long-term use, not exceeding reasonable computational or storage resources.
- **Simple integration:** Adding episodic memory should only require incorporating the memory module, not extensive redesign of the agent.

The work highlights that episodic memory's value is realized through its integration into the agent's task environment, supporting nuanced, context-rich behaviors over time. The system is tested in environments of increasing complexity to demonstrate its ability to enhance agent adaptability and long-term contextual reasoning[5].

-----

-----

</details>

<details>
<summary>What are the trade-offs between using larger context windows versus sophisticated memory retrieval and compression pipelines for AI agents?</summary>

### Source [70]: https://nebius.com/blog/posts/context-window-in-ai

Query: What are the trade-offs between using larger context windows versus sophisticated memory retrieval and compression pipelines for AI agents?

Answer: **Larger context windows** enable LLMs to handle longer texts and maintain context over extended dialogues or documents, leading to more accurate and relevant responses in tasks that require long-term memory, such as legal reviews or multi-turn conversations. However, this comes at the cost of increased computational resources, which can slow performance. Small context windows, while faster and more efficient, struggle with retaining context in longer interactions.

To address the limitations of fixed context windows, **memory-augmented models** like MemGPT integrate external memory systems, allowing information to be stored and retrieved as needed. This virtual memory approach enables models to analyze lengthy texts and maintain context across multiple sessions, supporting more coherent and detailed responses during extended interactions. These memory retrieval and compression pipelines effectively extend the agent’s context beyond the fixed window, allowing for scalable long-term memory management without the prohibitive computational cost of simply increasing the context window size.

Thus, the trade-off is between the simplicity and immediate capability of large context windows (with higher resource requirements) and the efficiency plus scalability of memory augmentation approaches (with added system complexity)[1].

-----

-----

-----

### Source [71]: https://blog.capitaltg.com/overcoming-memory-limitations-in-generative-ai-managing-context-windows-effectively/

Query: What are the trade-offs between using larger context windows versus sophisticated memory retrieval and compression pipelines for AI agents?

Answer: LLMs operate within **fixed context windows** that limit how much information they can consider at once. Increasing the context window size allows for longer conversations and richer context, but it also raises **computational costs and response latency**. For instance, a 4K token window requires developers to choose between including more conversation history (for coherence) or prioritizing recent inputs (for relevance).

Using models with **larger native context windows** is the most straightforward solution, requiring little engineering, but can be expensive for frequent interactions and not all models fully deliver on their advertised capacities. Alternatively, **chunking and sliding windows** can manage context in segments, while sophisticated memory retrieval and compression pipelines dynamically select and compress relevant context for input.

The trade-off centers on balancing **cost, latency, and context relevance**: larger context windows increase operational expense and delay, while retrieval/compression pipelines demand more engineering but optimize resource usage and context selection[2].

-----

-----

-----

### Source [72]: https://www.emerge.haus/blog/long-context-windows-in-generative-ai

Query: What are the trade-offs between using larger context windows versus sophisticated memory retrieval and compression pipelines for AI agents?

Answer: **Longer context windows** empower new capabilities and performance for generative AI agents, enabling richer, more complex tasks within a single instance. However, the **input cost** grows linearly with the number of tokens, leading to prohibitive costs if the entire window is used indiscriminately.

Innovations such as **context caching** (reusing prompt parts) and **adaptive context mechanisms** (selectively attending to relevant context) aim to mitigate these costs. These approaches blend retrieval with long context, allowing models to have large windows available but only fill them with the most pertinent information. This selective retrieval and compression can be implemented via hierarchical or external processes, ensuring efficiency.

Hardware considerations also arise as context windows scale. Memory becomes a bottleneck, requiring upgrades in GPU and interconnect technologies. Methods such as Mixture-of-Experts (MoE) can distribute computational loads, but ultimately, retrieval and compression pipelines are necessary to make ultra-long context windows practical and cost-effective. The trade-off is between the raw capability of very large context windows (with high economic and hardware costs) and the strategic efficiency of memory retrieval/compression systems[3].

-----

-----

-----

### Source [73]: https://www.meibel.ai/post/understanding-the-impact-of-increasing-llm-context-windows

Query: What are the trade-offs between using larger context windows versus sophisticated memory retrieval and compression pipelines for AI agents?

Answer: Increasing LLM context window size offers benefits but introduces several disadvantages:

- **Worse reference identification**: Performance is not uniform across the window; earlier tokens are referenced better than later ones, and very long prompts reduce text extraction accuracy.
- **Variable signal-to-noise ratio**: Longer prompts dilute the relevance of included context, potentially lowering accuracy compared to shorter, focused prompts.
- **Increased costs**: More input tokens directly raise query costs, and unnecessary long prompts may negate caching benefits.
- **Output token latency**: Longer prompts increase the time required for output generation, creating practical ceilings for context window size.

Models with smaller context windows remain popular due to lower memory and compute requirements, making them cheaper to train and host. Thus, sophisticated memory retrieval and compression pipelines are often preferred for maintaining performance and cost-effectiveness without overwhelming the model with unnecessary context[4].

-----

-----

-----

### Source [74]: https://www.ibm.com/think/topics/context-window

Query: What are the trade-offs between using larger context windows versus sophisticated memory retrieval and compression pipelines for AI agents?

Answer: A **larger context window** allows an AI model to process longer inputs, maintain more conversation history, and deliver more accurate, coherent responses. This improves the model’s ability to analyze complex and lengthy documents or code samples. However, increasing context window size comes with **greater computational power requirements**, leading to higher costs and potential vulnerability to adversarial attacks.

When the input exceeds the context window, the model must rely on **truncation or summarization**—which can lose detail—or use more sophisticated memory retrieval and compression pipelines to select and condense relevant information. These pipelines enable scalable context management without the exponential increase in resource consumption associated with very large context windows.

The trade-off is between the straightforward but costly approach of increasing context window size and the more complex but efficient strategy of memory retrieval/compression, which supports longer effective memory without proportionally greater computational burden[5].

-----

-----

</details>


## Sources Scraped From Research Results

<details>
<summary>(The video begins with the text "AI Engineer" in a metallic, gradient font against a black background.)</summary>

(The video begins with the text "AI Engineer" in a metallic, gradient font against a black background.)

[00:00]
(The text "AI Engineer" is now above a white outlined box containing "World's Fair". Upbeat, electronic music plays in the background. The screen then shows a "PRESENTING SPONSOR" title card with the Microsoft logo and name.)

[00:06]
(The screen displays an "INNOVATION PARTNER" card with the AWS logo, followed by a "PLATINUM SPONSORS" card featuring the logos for Graphite, Windsurf, MongoDB, daily, augment code, and WorkOS.)

***

### Introduction and the Promise of the Talk

[00:15]
**Speaker 1:** In the next 10 to 15 minutes, here's, uh, I guess my promise to you. I'm going to give you some information that will be high level. There will be some practical component to it, but this information that I'll give you within the next six months will be very relevant. And it will put you in the best position to build the best AI applications, to build the best agents that are believable, capable, and reliable. I know. We, we, we gonna get there. (The speaker smiles and laughs.) You know what? Just for you.

*The speaker promises to provide high-level, practical information about building advanced AI agents that will become very relevant in the near future.*

### Introducing the Concept of Memory in AI

**Speaker 1:** There you go. You're welcome. So we're gonna be talking about memory.

[00:54]
(A slide appears with a central box labeled "Memory". Lines connect to it from the left with the words "Stateless", "Prompt", and "Response". Lines connect from the right with the words "Stateful", "Persistence", and "Relationships".)

**Speaker 1:** We're going to be talking about the stateless applications that we're building today and how we can make them stateful.

[01:04]
**Speaker 1:** We're going to be talking about the prompt engineering that we're doing today and how we can reduce that by focusing on persistence. We're going to be turning the responses in our AI application and making our agents build a relationship with our customers. And all of it is going to be centered around memory.

*The speaker introduces memory as the central concept for transforming stateless AI applications into stateful ones, improving persistence and enabling AI agents to build relationships.*

### The Evolution of AI and Agentic Systems

[01:24]
(A new slide titled "Form Factor Evolution" shows a timeline. The points on the timeline are: LLM Powered Chatbots, RAG Chatbots, AI Agents, and Agentic Systems, each with a brief description.)

**Speaker 1:** So, I'm going to do a very quick evolution of what we've been seeing for the past two to three years. We started off with chatbots, LLM powered chatbots. They were great. ChatGPT came out November 2022. And yeah, it exploded.

[01:42]
**Speaker 1:** Then we went into RAG. We gave this chatbots more domain-specific relevant knowledge and it gave us more personalized responses. Then we begin to scale the compute, the data we were giving to the LLMs and it gave us emergent capabilities, right? Reasoning, uh, tool use. Now we're in the world of AI agents and agentic systems.

[02:04]
(The next slide is titled "The Agentic Spectrum" and shows four levels on a timeline: Level 1 - Minimal Agent, Level 2 - Controlled Flows, Level 3 - Routing and Specialized Workflows, and Level 4 - Autonomous Agents.)

**Speaker 1:** And the big debate is what is an agent, right? What is an AI agent? I don't like to go into that debate because that's like asking what is consciousness. Um, it is a spectrum. The agenticity, and that's a word now, agenticity, of, uh, of an agent is a, is a spectrum. So, there are different levels. I came here and I saw Waymo and to me was pure sorcery.

[02:32]
**Speaker 1:** We don't have that in the UK. And there are different levels of, um, self-driving. So you can look at the agentic spectrum in that respect. We have a minimal agent where's an LLM running a loop. Great. Then you have a level four is autonomous agent, a bunch of agents that have access to tools. They can do whatever they want. They're not prompted in any way or in a minimal way. But this is how I see things. It's a spectrum.

*The speaker outlines the rapid evolution from basic chatbots to complex agentic systems and describes agent capabilities as a spectrum ranging from minimal, looped LLMs to fully autonomous agents.*

### The Definition and Importance of AI Agent Memory

[02:57]
(A slide titled "AI Agents" appears with a text box defining an AI Agent. The word "memory" is highlighted in green.)

**Speaker 1:** So, what is an AI agent? It's a computational entity with awareness of its environment through perception, cognitive abilities through an LLM, and also can take action through tool use. But the most important bit is there is some form of memory, short-term or long-term.

[03:15]
(A diagram appears illustrating the core components of an AI agent: Perception, Planning, and Tools (Actions), all centered around the "AI Agent". "Memory" is a separate box connected to the agent and planning.)

**Speaker 1:** Memory is important. It's important because we're trying to make our agents reflective, interactive, proactive and reactive, and autonomous. And every, most of this, if not all, can be solved with memory.

[03:30]
(The diagram on the slide expands. The "Memory" box now contains sub-components for "Short Term" (Working Memory, Cache) and "Long Term" (Data Store, Entity Memory, Conversation Store, etc.). A separate "Database" is also shown as part of long-term memory.)

**Speaker 1:** I work at MongoDB and we're going to make, we're going to connect the dots, don't worry. So, this is all nice and good. This is what what you look at if you, um, double click into one AI agent is. But the most important bit to me is... I'll go to the slide. People are taking pictures, sorry. All right, let's go. The most important bit is memory. And when we talk about memory, the easy way you can think about is short-term, long-term, but there are all, there are other distinct forms, right? Um, conversational, entity memory, knowledge, data store, cache, working memory. We're going to be talking about all of that today. So, these are the high-level concepts.

*The speaker defines AI agents by their ability to perceive, plan, act, and remember, emphasizing that memory is the crucial component for making agents reflective, interactive, and autonomous.*

### Memory as the Foundation of Intelligence

[04:05]
(A slide appears with definitions for "Artificial Intelligence" and "Artificial General Intelligence (AGI)". The phrases "mimics human cognitive abilities" and "surpasses human performance across most tasks" are highlighted.)

**Speaker 1:** But let me go a little bit meta. Why we're all here, um, today in this conference is because of AI. We're all architects of intelligence. The whole point of AI is to build some form of computational entity that surpasses human intelligence, or mimics it. Then AGI, we're focused on making that intelligence surpass humans in all tasks we can think of. And if you think about the most intelligent humans you know, what determines the intelligence is the ability to recall. It's their memory. So if we're, if AI or AGI is meant to mimic human intelligence, is a no-brainer, no pun intended, that we need memory within the agents that we're building today. Does anyone disagree? Good. I would have kicked you out.

[04:57]
(The slide changes to an illustration of a brain with different sections labeled as types of memory: Sensory Memory, Short-Term Memory, Long-Term Memory (Explicit and Implicit), Working Memory, Episodic Memory, Semantic Memory.)

**Speaker 1:** Um, okay, let's go. So humans, you in your brain right now, you have these, you have this. This is not what it looks like, but it's close enough. You have different forms of memory. And that's what makes you intelligent. That's what makes you retain some of the information I'm going to be giving you today. There is short-term, long-term, working memory, semantic, episodic, procedural memory. Um, in your brain right now, there is something called a cerebellum. I always get the word wrong.

[05:21]
**Speaker 1:** But that's where you store most of the routines and skills you can do. Can anyone here do a backflip? Really? Wow, you seem very excited. Um, your, the information or the knowledge of that backflip is actually stored in that part of your brain. So, I heard it's 90% confidence by the way.

[05:42]
**Speaker 1:** That is actually, it is, right? I'm not going to do one, but... But it's stored in that part of your brain. Now you can actually mimic this in agents and I'm going to show you how. But now we're talking about agent memory.

[05:56]
(A slide appears titled "Agent Memory" with a definition: "AI agent memory is the persistent cognitive architecture that allows agents to accumulate knowledge, maintain contextual awareness, and adapt their behavior...")

**Speaker 1:** Agent memory is the mechanisms that we are implementing to actually make sure that state persists in our AI application. Our agents are able to accumulate information, turn data into memory, and have it inform the next execution step. But the goal is to make them more reliable, believable, and capable. Those are the key things.

[06:24]
(The next slide is "Memory Management". It defines memory management in agentic systems as referring to "the systematic organization, persistence, and retrieval of different types of information...")

**Speaker 1:** And the core topic that we are going to be working on as AI memory engineers is on memory management. We are going to be building memory management systems. And memory management is a systematic process of organizing all the information that you're putting into the context window. Yes, we have like large context window, but that's not for you to stuff all your data in. That's for you to pull in the relevant memory and structure them in a way that is effective that allows for the response, um, to be relevant.

*The speaker posits that since human intelligence is fundamentally based on memory, building intelligent AI requires mimicking human memory structures, leading to the crucial engineering discipline of agent memory management.*

### Core Components and Forms of Agent Memory

[06:58]
(A slide shows the "Core Components of Agent Memory Management". It lists six items: 1. Generation, 2. Storage, 3. Retrieval, 4. Integration, 5. Updating, 6. Deletion.)

**Speaker 1:** So these are the core components of memory management: generation, storage, retrieval, integration, updating, deletion. There's a lie here because you don't delete memories. Humans don't delete their memories unless it's a traumatic one and you want to forget.

[07:14]
(On the slide, "Retrieval" is highlighted in green, and "Deletion" is changed to "Deletion/Forgetting".)

**Speaker 1:** But we really should be looking at implementing forgetting mechanisms within the memory management systems that we're building. You don't want to delete memories. And there's different research papers that are looking at how to implement some form of forgetting within agents. But the most important bit is retrieval. And I'm getting to the MongoDB part. This is moving around. Um, this is RAG. It's very simple, right? Because we've been doing it as AI engineers.

[07:44]
(A complex diagram appears showing a RAG pipeline with MongoDB Atlas at its core, connecting data sources, data preparation, embedding generation, and various search types to an LLM.)

**Speaker 1:** Um, MongoDB is that one database that is core to RAG pipelines because it gives you all the retrieval mechanisms. RAG is not just vector, vector search is not all you need. You need other type of search. And we have that with MongoDB. Anything you can think of. You're going to be hearing a lot about MongoDB in this, um, in this conference today. But this is what RAG is. And you level up, you go into the world of agentic RAG, right?

[08:10]
(The RAG diagram becomes more complex, now labeled "Agentic RAG". It includes a "Memory" component and shows "Retrieval Mechanisms As Tools" feeding into a "Tools" box that the agent can call.)

**Speaker 1:** You give the retrieval capability to the agent as a tool. And now it can choose when to call on information. There's a lot going on. I'll I'll send this somehow to you guys. Or you can come to me and I'll, um, link it to you. Add me on LinkedIn and just ask for the slide and I'll send it to you. Richmond Alake on LinkedIn.

[08:34]
(The slide changes to a simpler diagram, showing "MongoDB Atlas" as a "Memory Provider" feeding into a "Memory Management System" which then interacts with the "LLM Context Window".)

**Speaker 1:** Um, this is memory. MongoDB is the memory provider for agentic systems. And when you understand that we provide the developer, the AI memory engineer, the AI engineer, all the features that they need to turn data into memory, to make the agents believable, capable, and reliable, you begin to understand the importance of having a technology partner like MongoDB on your AI stack.

*The speaker details the core components of memory management—generation, storage, retrieval, and forgetting—and presents MongoDB as the ideal "memory provider" for agentic systems due to its comprehensive retrieval capabilities and flexible data model.*

### Practical Examples of Memory Types in AI Agents

[09:03]
(A new slide appears that is a more focused diagram showing various memory types on the left (Persona, Conversation, Episodic, etc.) all feeding into the "Memory Provider" box, which is MongoDB. MongoDB then feeds the "Context Management System" and ultimately the "LLM Context Window".)

**Speaker 1:** So, this is the same um, image, but just a bit more focused on all the different memories. I'm going to skip through this slide because I go into a bit of detail. Um, I'm also going to give you a library. I'm working on an open-source library. I'm ashamed of the name. I was trying to be cool when I came up with it. It's called "Memorizz." (Audience laughs). Um, you can type that on Google, you'll find it, but it has all the design patterns of all of this memory that I'm showing you, all these memory types, and that I will show you as well.

[09:34]
(The slide changes to a title card: "Forms of Memory in AI Agents" with the MongoDB leaf logo.)

**Speaker 1:** But there are different forms of memory in AI agents and how we make them work. So let's start with persona. Who's, is anyone here from OpenAI? Leave. I'm joking. Um, well, a couple, a couple months ago, right? So they, they gave ChatGPT a bit of personality, right? Um, and they didn't do a good job, but they are going in the right direction, which is we are trying to make our systems more believable.

[10:03]
(A slide titled "PERSONA" appears, describing how it stores agent identity information, traits, roles, and communication styles.)

**Speaker 1:** Right? We're trying to make them more human. We're trying to make them create a relationship with the consumer, with the users of our systems. Persona memory helps with that.

[10:12]
(The slide now shows a screenshot of MongoDB Compass with a document representing a "persona" for an agent, including fields for name, role, goals, and background.)

**Speaker 1:** And you can model that in MongoDB. Right? This is Memorizz. You, if you spin up the library, it helps you, um, spin up all of these different type of memory types. So this is persona. I have a little demo if we have time. Um, but this is persona. This is what it will look like in MongoDB.

[10:32]
(The slide changes to "TOOLBOX", describing it as a store for tool definitions, metadata, and schemas.)

**Speaker 1:** Then there's toolbox. Um, the guidance from OpenAI is you should only put, um, the schema of maybe 10 to 21 tools in the context window. But when you use your database as a toolbox, where you're storing the JSON schema of your tools in MongoDB, you can scale. Because just before you hit the LLM, you can just get the relevant tool using any form of search. So that's toolbox memory.

[11:03]
(A screenshot of MongoDB Compass shows a document representing a "tool" (get_weather) with its function description, parameters, and required arguments.)

**Speaker 1:** And that's what it will look like. You would store all the information of your JSON schema. Now you'll begin to understand that MongoDB gives you that flexible data model. The document data model is very flexible. It can adapt to whatever data, whatever model you want your data to take, whatever structure. And you have all of the retrieval capabilities: graph, vector, text, geospatial query, in one database.

[11:33]
(The speaker quickly clicks through slides for "CONVERSATION MEMORY", "WORKFLOW MEMORY", "EPISODIC MEMORY", "LONG-TERM MEMORY", "AGENT REGISTRY", and "WORKING MEMORY", each with a definition and an example of how it would be structured as a document in MongoDB.)

**Speaker 1:** Conversation memory is a bit obvious, right? Back and forth conversation with, uh, ChatGPT, with Claude. You can store that in your, in your database as well, in MongoDB as conversational memory. And this is what that would look like. Timestamp, and you have a conversation ID. And you can see something there called recall recency and associate conversation ID. And that's my attempt at implementing some memory signals. Um, but and that's goes into the forgetting mechanism that I'm trying to implement in my very famous library, Memorizz. Um, I'm gonna go for the next slides a bit quicker because I want to get to the end of this. Workflow memory is very important. You build your agentic system, they execute a certain step. Step one, step two, step three, it fails. But one thing you could do is the failure is experience. It's a learning experience. You can store that in your database, I see you nodding, you're like, yeah.

[12:27]
**Speaker 1:** You can store that in your database and you can then pull that in in the next execution to inform the LLM to not take the step or explore other paths. You can store that in MongoDB as well. You can model that. Because what you have with MongoDB is that memory provider for your agentic system. And that's what, this is what that looks like when you model it. An example of it anyway. We have episodic memory, we have long-term memory. We have an agent registry. You can store the information of your agent as well. Um, and this is how I do it. You can see the agent has tools, persona, all the good stuff. There's entity memory. There's different forms of memory. And the memory, the Memorizz library is very experimental and educational, but it encapsulates some of the memory and implementation and design patterns that I'm thinking of on an everyday basis that we're thinking of in MongoDB.

*The speaker details various forms of memory necessary for sophisticated AI agents—such as persona, toolbox, conversation, and workflow memory—and shows how they can be effectively modeled and stored using MongoDB's flexible document structure.*

### Conclusion: MongoDB as the AI-Ready Memory Provider

[13:20]
(The slide shows a large title: "The Memory Provider For Agentic Systems: MongoDB".)

**Speaker 1:** So, MongoDB, you probably get the point now, the memory provider for agentic systems. There are tools out there that focus on memory management. Um, MemGPT, Mem0, Zep, they're great tools. But after speaking to some of you folks and some of our partners and customers here, there is not, there is, there is not one way to solve memory, and you need a memory provider to build your custom solution, to make sure the memory management systems that you're able to implement are effective.

[13:53]
(A slide titled "Voyage AI's models" appears, showing two categories: "Embedding Models" (with sub-types like General-Purpose and Domain-Specific) and "Rerankers" (Standard and Lite).)

**Speaker 1:** So, we really understand the importance of managing data, managing memory. And that's why earlier this year, we acquired Voyage AI. Now, they create the best, no offense, OpenAI, embedding models in the market today. Voyage AI embedding models, uh, we have, uh, text, multimodal. We have rerankers. And this allows you to really solve the problem, or at least reduce AI hallucination within your RAG and agentic systems.

[14:25]
(A new slide appears with two diagrams labeled "BEFORE" and "AFTER". The "BEFORE" diagram shows a complex RAG pipeline with separate databases and models. The "AFTER" diagram shows a simplified pipeline where a single "MongoDB - VOYAGE AI" box handles unstructured data, embedding, vector search, and reranking.)

**Speaker 1:** And what we're doing and what we're focused on, the mission for MongoDB is to make the developer more productive by taking away the considerations and all the concerns around managing different data and all the process of chunking, retrieval strategies. We, we, we pull that into the database. We are redefining the database. And that's why in a few months, we're going to be pulling in Voyage AI, the embedding models and the rerankers into MongoDB Atlas. And you will not have to be writing chunking strategies for your, um, for your data. I see a lot of people nodding. Yeah. That's good.

[15:03]
(A timeline of MongoDB's evolution appears, starting from 2007 with "Flexible Document Model" and progressing to 2025 with "Integrated AI Retrieval" including "Embedding Models & Rerankers". The title is "MongoDB was built for change, empowering YOU to innovate at the speed of the market".)

**Speaker 1:** So, MongoDB is a, is a household name, to be honest. We've, um, I watched MongoDB IPO back when I, back when I was in university. I bought the stocks when I was in university. Um, three, just three. I only had about 100 pound. I was broke. But, we are very focused and we take it very seriously making sure that you guys can build the best AI products, AI features very quickly in a secure way. So MongoDB is built for the change that we are going to experience now, tomorrow, in the next couple years.

[15:37]
(The final content slide shows a black and white photo of two scientists in a lab (David Hubel and Torsten Wiesel) next to a diagram of an experiment on a cat's visual cortex.)

**Speaker 1:** I want to end with this. You know who these two guys are? Damn. Okay. This is Hubel and Wiesel. They won a Nobel Peace Prize, um, in the late '90s. But they did some research on the visual cortex of cats. Um, they experimented with cats. This probably wouldn't fly now, but back in the '50s and '60s, things were a bit more relaxed.

[15:58]
**Speaker 1:** But they found out that the visual cortex of the brains between cats and humans actually worked by learning different hierarchies of representation. So, edges, contours, and abstract shapes. Now, people that are in deep learning will know that this is how convolutional neural network works. That's face detection, object detection. It all comes from neuroscience.

[16:30]
(The slide changes to a photo of a modern-day panel discussion. A line points to one of the panelists, with the text: "Dr. Tenayu Ma, Chief AI Scientist, MongoDB".)

**Speaker 1:** So, we are architects of intelligence, but there is a better architect of intelligence, is nature. Nature has created our brains. It's the most effective form of intelligence that we have today. And we can look inwards to build these agentic systems. So, last week Saturday, myself and Tengyu, who is the Chief AI Scientist at MongoDB, also the founder of Voyage AI, we sat with these three guys in the middle are neuroscientists. Kenneth has been exploring human brain and memory for over 20 years. And and over here is Charles Packer. He's the creator of MemGPT or AutoGen. And we are having this conversation. And once again, we're mirroring how we're bringing neuroscientists and application developers together to solve and push us on the path of AGI. So, that's my talk done. Check out Memorizz and you can come talk to me about memory. Add me on LinkedIn if you want this presentation. Thank you for your time. (Audience applauds.)

(The video ends with the "AI Engineer World's Fair" logo.)

*The speaker concludes by positioning MongoDB, enhanced with the recent acquisition of Voyage AI, as the ultimate memory provider for building advanced agentic systems, emphasizing a vision inspired by neuroscience to simplify development and push the boundaries of AI.*

</details>

<details>
<summary>AI Agent Memory</summary>

# AI Agent Memory

AI Agent Memory is the ability of an AI agent to store, recall and use information from past interactions to make better decisions in the present and future. Without memory, an agent treats every interaction as if it is the first interaction. With memory, an agent can maintain context, adapt to users and improve over time i.e memory gives AI agent continuity, context-awareness and learning abilities.https://media.geeksforgeeks.org/wp-content/uploads/20250722125643748319/ai_agent_memory-.webp

### Need of Memory by AI Agents

Many real-world scenarios demand agents to remember and adapt to:

- Keep track of conversations.
- Track progress in multi-step workflows.
- Learn from past feedback and improve.
- To keep the personalization i.e remembering user preferences.

An agent without memory is limited to short and isolated responses whereas it can act more intelligently and deliver better experience with memory.

## Types of Memory in AI Agents

AI agents use different types of memory, each serving a unique purpose:https://media.geeksforgeeks.org/wp-content/uploads/20250722125615546521/AI-Agent-Memory.webp

### 1\. Short-Term Memory

Short-term memory (STM) is like the AI agent’s temporary notepad. It holds recent information just long enough to finish the current task. After that, it is cleared for the next job. This type of memory is great for quick tasks such as customer support chats, where the agent only needs to remember the ongoing conversation to help the user.

### 2\. Long-Term Memory

Long-term memory (LTM) stores information for much longer periods. It can keep specific details, general facts, instructions or even the steps needed to solve certain problems. There are different types of long-term memory:

- **Episodic Memory:** This type remembers specific events from the past like a user’s date of birth that was used during an earlier conversation. The agent can use this memory as context in future interactions.
- **Semantic Memory:** This holds general knowledge about the world or things the AI has learned through past interactions. The agent can refer to this information to handle new problems effectively.
- **Procedural Memory:** Here the agent stores “how-to” steps or rules for making decisions. For example, it might remember the process for solving a math problem and use the same steps when tackling a similar task later.

| Memory type | What it Stores | Example |
| --- | --- | --- |
| Short-term Memory | Context of current session | Conversation so far in a chatbot session. |
| Long-term Memory | Knowledge or data over time | User preferences saved across sessions. |
| Episodic Memory | Specific events and experiences | Sequence of actions taken in a mission. |
| Semantic Memory | Facts and world knowledge | Paris is the capital of France. |
| Procedural Memory | Rule based data for immediate tasks. | Numbers held while solving a math problem. |

## Storage Methods and Techniques

Memory can be implemented in various ways depending on the type and scale required:

- **Buffers and queues**: Simple for short-term storage.
- **Databases**: Structured for long-term and reliable storage.
- **Vector databases**: Store text embeddings i.e data is converted in numeric form for semantic search.
- **Knowledge graphs**: Organize facts and relationships via graph.
- **Neural memory modules**: Integrate memory into neural networks.

### Techniques and Tools

- **LangChain**: Popular for adding conversational memory to LLMs.
- **Vector Stores**: Pinecone, Weaviate, Milvus for embedding-based memory.
- **Attention mechanisms**: Built into Transformers to handle context.
- **Neural Turing Machines and DNCs**: Advanced neural architectures with memory.

## Comparison Table of AI Memory Techniques

Let's compare the key memory approaches for AI Agents:

| Technique | Best for | Strengths | Limitations |
| --- | --- | --- | --- |
| Simple Buffer (FIFO) | Short-term context | Easy to implement, Fast | Cannot handle long-term storage |
| Relationship Database | Structured long-term memory | Mature technology and easy for querying | Poor at semantic/contextual queries |
| Vector Database | Semantic search and unstructured data | Handles fuzzy matching and is scalable | Requires embedding generation |
| Knowledge Graph | Relationships and World Knowledge | Good for reasoning and inference | Complex to build and maintain |
| Neural Turing Machine | Advanced neural memory | Integrates with deep learning models | Computationally intensive |

## Real-world applications of AI Memory

- **Customer Service Chatbots**: Chatbots remember past interactions to offer faster, personalized support.For example: A bot recalls previous orders and suggests relevant products.
- **Virtual Assistants:** Assistants like Siri and Alexa remember schedules and preferences to provide tailored help. For example: An assistant recalls your daily routine and adjusts reminders.
- **Healthcare AI:** AI in healthcare tracks patient histories and suggests treatments based on past data. For example: A healthcare assistant reminds you about medications and appointments.
- **E-commerce Platforms:** Online stores remember browsing and purchase history to improve recommendations. For example **:** Amazon suggests products related to previous purchases.

</details>

<details>
<summary>1. Cognitive Limitations of LLMs in Autonomous Decision-Making</summary>

## 1. Cognitive Limitations of LLMs in Autonomous Decision-Making

Large language models excel at generating fluent procedural outputs but often falter when faced with dynamic, “wicked” environments that require flexible reasoning and memory recall. In this paper, we first provide a cognitive‐science‐informed analysis of why procedural‐memory‐centric LLM architectures fail in complex tasks, and propose a modular system that augments LLMs with dedicated semantic and associative memory components to support causal decision‐making. We then review the limitations of procedural memory in LLMs, detail our three‐part modular architecture, and compare our approach to related work and illustrate its benefits.

### 1.1. LLM Architecture

LLMs operate as agentic actors through architectures rooted in transformer-based sequence modeling. Their core mechanism—procedural memory—is implemented via self-attention layers that statistically model token co-occurrence patterns across massive text corpora (Vaswani et al., [2017](https://arxiv.org/html/2505.03434v1#bib.bib27 "")).

LLMs generate outputs by computing attention weights across input tokens. This mechanism captures local and global dependencies, enabling probabilistic predictions of the next tokens. Similarly to procedural memory in the basal ganglia, which strengthens synaptic connections through repeated task execution (e.g., piano practice) (Foerde and Shohamy, [2011](https://arxiv.org/html/2505.03434v1#bib.bib6 "")), LLMs refine attention weights during training to automate pattern completion.

LLMs often return factually incorrect information because their procedural memory—trained to predict sequences by weighting token co-occurrences—generates outputs based on statistical likelihoods in their training data, not grounded truth (Ruis et al., [2025](https://arxiv.org/html/2505.03434v1#bib.bib21 "")).
The noise between linguistic patterns and real-world facts creates a gap between the model’s plausible token continuations and verified knowledge.
“Hallucinations” or “confabulations” (fluent but ungrounded token continuations (Smith et al., [2023](https://arxiv.org/html/2505.03434v1#bib.bib22 ""))), thus emerge when statistical priors override factual precision.

In short, LLMs lack episodic memory.
In an attempt to combat this, Retrieval-Augmented Generation (RAG) systems combine a dense retriever with an LLM (Lewis et al., [2020](https://arxiv.org/html/2505.03434v1#bib.bib13 "")), conditioning the LLM’s output on both a query _and_ and a document.
Whilst a step in the right direction, this architecture remains fundamentally limited.

### 1.2. Limitations of Procedural Memory

LLM architecture, even when augmented by RAG systems, exhibits several critical limitations:

- •

LLMs operate without persistent state, meaning they cannot retain information from previous interactions unless that information is explicitly included in the input prompt. Each inference is an independent forward pass, and prior interactions are not retained unless explicitly re-injected via prompts.

- •

LLMs are constrained by fixed-length context windows (typically a few thousand to a few hundred-thousand tokens), which limit their ability to process or recall long sequences of information across time. Even models with large windows truncate or discard prior context (Hosseini et al., [2025](https://arxiv.org/html/2505.03434v1#bib.bib10 "")), violating the continuity required for episodic reasoning (the ability to recall and integrate specific past experiences, including their temporal and contextual details, to inform present decisions (Maurer and Nadel, [2021](https://arxiv.org/html/2505.03434v1#bib.bib16 ""))).

- •

LLMs do not have a memory-consolidation mechanism to integrate and retain learned experiences over time. Humans consolidate episodic memories into long-term storage via hippocampal replay (Ólafsdóttir et al., [2018](https://arxiv.org/html/2505.03434v1#bib.bib17 "")). More recent developments have introduced mechanisms for storing summaries of past interactions to simulate memory consolidation in LLMs (Zhong et al., [2023](https://arxiv.org/html/2505.03434v1#bib.bib31 "")), but these approaches capture only surface-level recaps rather than integrating experiences into structured knowledge or enabling flexible recall based on relevance or context.

- •

Even with RAG, retrieved documents are injected as static context tokens (Gao et al., [2024](https://arxiv.org/html/2505.03434v1#bib.bib7 "")). Unlike biological episodic memory, which dynamically updates and re-weights past experiences, RAG cannot revise retrieved knowledge mid-interaction (e.g., reconciling conflicting facts in documents).

- •

LLMs have no mechanism for meta-learning. During inference, an LLM relies on fixed parameters, processing retrieved data as context without modifying its attention mechanisms (Liu et al., [2025](https://arxiv.org/html/2505.03434v1#bib.bib15 "")). Unlike humans, it cannot reshape attention based on new insights, which limits real-time adaptability.

A customer service LLM exemplifies these limitations. Since it is stateless, it forgets past interactions unless the user explicitly repeats details or the system re-injects them into the prompt. Its fixed context window means that prior messages may be truncated, preventing seamless episodic reasoning. Unlike human memory, it cannot consolidate key details—such as a user’s repeated complaints about delayed shipping—into long-term storage. Even with RAG, retrieved records are static and cannot be updated mid-session, meaning that conflicting information (e.g., an initial delay estimate vs. a later update) is not reconciled dynamically. Finally, because it lacks meta-learning, the LLM does not refine how it processes such interactions over time, forcing users to repeatedly provide the same context rather than benefiting from accumulated experience.

It could be argued that fine-tuning based on user histories could simulate episodic memory. However, fine-tuning introduces catastrophic forgetting: updating weights for new data degrades performance on prior tasks (Goodfellow et al., [2015](https://arxiv.org/html/2505.03434v1#bib.bib8 "")).
Biological episodic memory avoids this via neurogenesis and sparse coding (Wixted et al., [2014](https://arxiv.org/html/2505.03434v1#bib.bib30 "")), which transformers do not replicate.

All of these challenges are not specific to LLMs. They are specific to procedural memory. The flaw is not in the implementation of LLMs, but in their architectural underpinnings.

### 1.3. The Need for Non-Procedural Memory

In addition to the inherent limitations of an architecture that is limited to procedural memory (plus a problematic implementation of episodic memory), LLMs also exhibit semantic memory gaps. LLMs encode knowledge as dense, overlapping vector representations in high-dimensional latent spaces. Although these embeddings capture statistical relationships (e.g., “Paris is to France as Tokyo is to Japan”), they lack explicit hierarchies (no discrete nodes for facts—e.g., “User X𝑋Xitalic\_X prefers eco-friendly products”), and symbolic grounding (vectors conflate syntactic, semantic, and pragmatic features).

While embeddings do form clusters (e.g., animals vs. vehicles), they cannot perform compositional reasoning (e.g., inferring “If all mammals breathe air, and whales are mammals, then whales breathe air”) probabilistically. LLMs generalize logical structures by recognizing patterns from training data. This falls short of true semantic reasoning, and therefore LLMs often fail in situations that require strict logical consistency.
LLMs also exhibit systematic associative blind spots. Transformers model pairwise token interactions but struggle with higher-order associations that require chained reasoning or cross-context links (Peng et al., [2024](https://arxiv.org/html/2505.03434v1#bib.bib18 "")).

For example, associating a user’s preference for “transparency” (from a shipping complaint) with “detailed ingredient list” (in a food app) requires inferring abstract principles, not token co-occurrence. This is an architectural constraint, not a limitation in training or data.
Attention heads focus on local context, limiting cross-session reasoning. Whereas humans link concepts via bidirectional hippocampal-cortical pathways, transformers process information uni-directionally (input leads to output, not the other way around).

It could be argued that chain-of-thought prompting enables multi-step reasoning (Wei et al., [2022](https://arxiv.org/html/2505.03434v1#bib.bib29 "")), but chain-of-thought relies on procedural pattern extension, not associative binding. It cannot dynamically link concepts learned in separate sessions (e.g., connecting a user’s travel preferences to their shopping habits). While advancements like elastic weight consolidation (which preserves important parameters across tasks to reduce forgetting) and dynamic context window management (e.g., recurrent memory transformers that extend context beyond fixed token limits) have made strides in addressing some limitations of LLMs, they remain constrained by fundamental architectural mismatches. Elastic weight consolidation has been shown to mitigate catastrophic forgetting in “kind” learning environments (Atari games) (Kirkpatrick et al., [2017](https://arxiv.org/html/2505.03434v1#bib.bib11 "")), but it has yet to be shown how well this method performs in continuous, non-stationary learning, where feedback loops are sparse and ambiguous.

Similarly, while dynamic context windows improve memory retention (Wang et al., [2024](https://arxiv.org/html/2505.03434v1#bib.bib28 "")), they remain bound to transformers’ procedural framework, and are therefore limited by that framework’s general architectural constraints—processing tokens rather than discrete concepts, and lacking interfaces to ground semantics or associate rewards with specific knowledge structures.

The question is not merely whether LLMs can be extended to overcome the absence of dynamic memory systems, but whether it is more realistic to adapt an architecture to do something it was not designed to do—forcing transformers to mimic episodic and semantic memory—or to append an architecture explicitly designed for these functions. Appending new components can create brittle reliance on prompt engineering or retrieval heuristics. Re-thinking the architecture allows for more principled interaction patterns and targeted learning.

### 1.4. The Need for Modular AI

AI agents do not need to “think” exactly the way humans do in order to be useful. In fact, biological systems evolved through constraints (e.g., energy efficiency) (Li and van Rossum, [2020](https://arxiv.org/html/2505.03434v1#bib.bib14 "")), not optimal design, so trying to replicate human cognitive systems exactly might be a suboptimal approach.
At any rate, modular AI architectures are pragmatic engineering choices to compensate for transformer limitations, not attempts to replicate biology.
LLMs’ procedural prowess is undeniable, but their architectural rigidity—static parameters, dense embeddings, and unidirectional processing—renders them inadequate for decision-making in _unkind_ or “ _wicked_” environments.

While techniques like fine-tuning, RAG, and chain-of-thought mitigate specific issues, they fail to address the core absence of dynamic memory systems. The path forward lies in hybrid architectures that pair LLMs with associative and semantic modules, explicitly designed for incremental learning and cross-context reasoning.

## 2. Learning Environments

The degree to which an agent is capable of successfully acting with autonomy is inextricably tied to the structure of the learning environment in which it operates. Learning environments are a concept rooted in cognitive science and decision-making research. Psychologist Robin Hogarth’s framework of “kind” and “wicked” learning environments has important implications for agent design (Hogarth et al., [2015](https://arxiv.org/html/2505.03434v1#bib.bib9 "")). In Hogarth’s original formulation of learning environments, kind environments are defined by stable rules, repetitive patterns, and unambiguous feedback. Examples include chess, standardized testing, and cooking from a recipe. In such environments, outcomes are immediate and directly attributable to actions (e.g., a chess move leads to checkmate or loss). Statistical regularities dominate, enabling pattern-based strategies.
Wicked learning environments, on the other hand, are characterized by dynamic rules, sparse or ambiguous feedback, and novelty. Examples include entrepreneurship, healthcare diagnostics, and customer engagement. Outcomes may emerge long after actions (e.g., a marketing campaign’s impact on brand loyalty), and underlying data distributions shift over time (e.g., evolving consumer preferences). Hogarth argued that human intuition excels in kind environments but falters in wicked ones, necessitating deliberate analytical strategies.

LLMs are an embodiment of human intuition, and exhibited through language. LLMs excel in kind environments due to transformer-based pattern matching. For example: given an error message, an LLM predicts fixes to the code by correlating with repair patterns in training data. Even in these remarkably stable and well-known environments, LLMs still overfit to training distributions, causing them to fail on edge cases outside their corpus (Aissi et al., [2024](https://arxiv.org/html/2505.03434v1#bib.bib2 "")).

LLMs have no way to successfully navigate wicked learning environments. For example, an LLM chat bot trained on retail dialogues may handle queries like “find navy blue shirts” (records retrieval is a kind learning environment), but fail when a user abruptly exits a session after receiving recommendations. LLMs process each interaction as an independent sequence, with no persistent memory of prior sessions. The user’s exit could signal dissatisfaction, distraction, or indecision—a wicked feedback signal the LLM cannot interpret. LLMs assume a fixed data distribution based on their training data, but wicked environments require adaptation to distributions outside of the training data.

It might be argued that LLMs could handle wicked learning environments via fine-tuning, but
(i)fine-tuning on new data overwrites prior weights, degrading performance on original tasks (catastrophic forgetting), and
(ii)continuous fine-tuning is impractical and in most cases financially infeasible for real-time adaptation.

The “kind vs. wicked” dichotomy (which, of course, is more of a continuous spectrum than a set of hard-and-fast categories) mandates distinct technical strategies: in kind learning environments, we can leverage LLMs’ procedural strengths but in wicked learning environments, agents need semantic-associative memory to be able to adapt appropriately.

## 3. Augmenting LLMs with Modular Semantic-Associative Systems

To enable robust decision-making in wicked environments, we propose a modular architecture where agentic learners (semantic-associative systems) operate independently of agentic actors (LLMs). This separation ensures specialized cognitive capabilities: learners focus on adaptive reasoning via reinforcement learning (RL) (Sutton et al., [1998](https://arxiv.org/html/2505.03434v1#bib.bib23 "")), while actors handle procedural execution.

LLMs act as context-bound agents, limited to processing the inputs they’re given. Their architecture prevents three key capabilities:
(i)retaining distilled learnings across sessions,
(ii)dynamically associating actions with outcomes outside of explicit user feedback, or
(iii)autonomously expanding their context.
These aren’t scaling problems—they’re inherent to transformers’ stateless design. However, to successfully navigate wicked learning environments, agentic learners are needed, capable of incrementally building context by exploring and linking semantic categories.

Agentic learners input actions—or, more precisely, they input a stream of user interactions with different actions and, via model-free RL and explore-exploit mechanisms like Thompson sampling (Thompson, [1933](https://arxiv.org/html/2505.03434v1#bib.bib24 "")), they construct contextual metadata that can then be used to select next best actions, optionally generating those actions by passing that metadata to an agentic actor.
An agentic learner can operate on multiple action sets at the same time.

Customer engagement—the ongoing process of influencing user behavior through personalized interactions—is a good example of a wicked learning environment. Success requires adapting to shifting preferences, interpreting ambiguous signals (e.g., a purchase might indicate satisfaction or mere acquiescence), and evolving strategies without clear rules. Traditional LLMs fail here because they treat each interaction as independent, unable to link outcomes to semantic distillations, and therefore unable to build a reasonably tidy and memory-efficient context across sessions.
Our proposed architecture separates adaptation from execution. An independent learning module handles longitudinal reasoning: tracking which strategies work for specific users, associating causes (e.g., communication timing and frequency, channel, recommendation approaches, copy elements, etc.) with effects (e.g., user engagement metrics), and refining its understanding as new data arrives. The LLM then translates these distilled insights into natural language, focusing solely on coherent response generation.

This division of labor plays to each component’s strengths. The learner operates on slightly slower timescales, filtering noise from sparse feedback, while the LLM handles real-time communication. Crucially, the system doesn’t seek deterministic answers. Rather, it maintains competing hypotheses about user preferences, updating them probabilistically as evidence accumulates. This mirrors how humans navigate uncertainty, but with the scalability required for automated systems.

### 3.1. Integration with LLM Actors

The agentic learner generates contextual metadata by compiling contextual information. This metadata encapsulates the learner’s distilled understanding of the user’s preferences and the current context, serving as a bridge between adaptive reasoning and procedural execution. Once the context vector is constructed, it is passed to the LLM actor as a prefix to the user’s prompt. The LLM then generates a response conditioned on this augmented input. This integration ensures that the LLM’s procedural strengths—fluency, coherence, and stylistic adaptability—are guided by the learner’s semantic and associative insights.

Agentic learners can be said to have a cold-start problem, though this is arguably not a flaw, but a design feature of agentic learners operating in wicked environments. Procedures like sliding-window Thompson sampling (Trovo et al., [2020](https://arxiv.org/html/2505.03434v1#bib.bib25 "")) inherently address this via their explore-exploit balance. Early interactions will prioritize broad exploration (e.g., testing all value propositions equally). In wicked environments, user preferences will shift, and the resulting lack of rewards for previously-rewarded semantic categories will shift and flatten the underlying distributions for those categories, causing an agentic learner to automatically reallocate exploration bandwidth.

Agentic learners embrace uncertainty: they are not tasked with finding a “right” answer (which rarely exists in wicked learning environments), but with placing adaptively informed bets. In other words, they are designed for causal decision-making rather than causal estimation (Fernández-Loría and Provost, [2022](https://arxiv.org/html/2505.03434v1#bib.bib5 "")).

## 4. Modular Architectures for Cognitive Specialization

The human brain has several different definitions of cognition, each backed by different mechanisms, whereas LLMs have only one definition of cognition, backed by only one mechanism. This raises serious doubts about the ability of LLMs to serve as a foundation for broader “Artificial General Intelligence” initiatives, or even for the more modest goal of building systems that are capable of acting agentically in messy and complex real-world contexts.

Current efforts to expand LLMs into multi-modal systems often conflate mechanical diversity—such as processing images, text, or audio—with cognitive specialization. While multi-modal systems enhance the range of input types an agent can handle, they do not address the fundamental need for specialized cognitive subsystems. True autonomy requires a modular architecture where distinct components are optimized for specific cognitive tasks. For example:

- •

A semantic module manages structured knowledge by organizing learned actions and concepts into abstract, generalizable representations—similar to how human semantic memory encodes facts and rules detached from specific experiences (Tulving, [1972](https://arxiv.org/html/2505.03434v1#bib.bib26 ""));

- •

An associative module links experiences by forming and retrieving relationships between co-occurring states and actions—a process analogous to associative binding in human cognition, where elements of experience are connected through repeated or meaningful co-activation (Ranganath and Ritchey, [2012](https://arxiv.org/html/2505.03434v1#bib.bib20 "")).

- •

A procedural module uses learned semantic associations as context to generate coherent, human-readable responses.

By decoupling these functions, the system avoids the pitfalls of monolithic architectures, where a single model is forced to handle tasks for which it is ill-suited. While this kind of modularity introduces interface challenges—such as coordinating between components and managing error propagation—these are deliberate trade-offs. A modular design offers clearer control boundaries, improves interpretability of agent behavior, and allows individual modules to be retrained or upgraded without overhauling the entire system.

## 6. Conclusion

Real-world decision-making requires cognitive diversity beyond procedural memory. To advance autonomous agents, we propose three key principles:

1. (1)

Decoupling Cognitive Modules. LLMs should act as components within modular architectures, not central controllers. This separation allows each module to specialize in its respective cognitive function, whether procedural, associative, or semantic.

2. (2)

Rigorous Environment Classification. Both application and data systems must be designed with the learning environment in mind. Kind environments may require only procedural capabilities, while wicked environments demand hybrid architectures that integrate associative and semantic reasoning.

3. (3)

Investment in Associative and Semantic Systems. Prioritize research into neural-symbolic architectures, sparse memory models, and other frameworks that enable explicit reasoning and adaptive learning.

By embracing these principles, we can develop agents that complement human ingenuity in uncertainty—not merely replicate procedural expertise in structured domains. This shift from monolithic to modular architectures represents a necessary evolution in AI design, one that acknowledges the complexity of real-world decision-making and the limitations of current approaches.

</details>

<details>
<summary>I have reviewed the provided markdown content against the article guidelines. The markdown appears to be a research paper titled "Recurrent Context Compression: Efficiently Expanding the Context Window of LLM". The article guidelines, however, describe a lesson on the concept of agent "memory", covering topics like semantic, episodic, and procedural memory, storage methods, and implementation examples.</summary>

I have reviewed the provided markdown content against the article guidelines. The markdown appears to be a research paper titled "Recurrent Context Compression: Efficiently Expanding the Context Window of LLM". The article guidelines, however, describe a lesson on the concept of agent "memory", covering topics like semantic, episodic, and procedural memory, storage methods, and implementation examples.

The content of the provided markdown does not align with the topics specified in the article guidelines. Therefore, there is no relevant content to retain. Following the instruction to keep *only* the core textual content that is pertinent to the article guidelines, the resulting cleaned markdown is empty.

</details>

<details>
<summary>This history is why RAG, or retrieval-augmented generation, is often seen as the holy grail of knowledge management.</summary>

This history is why RAG, or retrieval-augmented generation, is often seen as the holy grail of knowledge management.

By the time companies reach enterprise level, they’ve likely gone through numerous re-brands, restructurings, and pivots, with executives and board members coming and going and hundreds or thousands of employees onboarding and departing. In this context, answering even simple questions about the company can be challenging.

AI — via RAG — promises to be the first real solution. With RAG, employees ask natural language questions and task database tools to retrieve the information and use generative AI to formulate that information into a readable, relevant answer.

The initial results of RAG are powerful today and promising for what might come tomorrow, but those results also reveal significant limitations. Without a suitable database at the foundation, RAG can’t live up to its potential.

## What is a vector database?

A vector database stores and maintains data in vector formatting. When data is prepared for storage, it’s split into chunks of characters ranging from 100 to 200. Then, with an embedding model, these chunks are converted into a vector embedding that can be stored in the vector database.

A vector database has many use cases, but a vector database, by definition, isn’t a complete foundation for RAG. Most vector databases don’t provide an embedding model, so companies typically need to find and integrate one to use a vector database as their foundation.

Vector databases also vary significantly depending on which algorithms they use. Vector databases use either K-Nearest Neighbors (KNN) or Approximate Nearest Neighbor (ANN) algorithms, and each has different effects on the final result of any search and retrieval features.

When users enter a query, the vector database converts the query into a vector embedding, and either a KNN or ANN algorithm determines which data points are closest to the query data point. As the name of each implies, both algorithms are similar, with ANN being more approximate but much faster.https://writer.com/wp-content/uploads/2024/10/Image-1-3-1.png?w=640

#### **Technical implementation example**

Here’s a simple example of how a vector database might handle embeddings and similarity search using the FAISS library developed by Facebook AI Research:

Python

```hljs makefile
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Sample data chunks (simulated embeddings)
data_chunks = [\
    "Apple was founded as Apple Computer Company on April 1, 1976.",\
    "The company was incorporated by Steve Wozniak and Steve Jobs in 1977.",\
    "Its second computer, the Apple II, became a best seller.",\
    "Apple introduced the Lisa in 1983.",\
    "The Macintosh was introduced in 1984."\
]

# Load the pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert data chunks into embeddings
data_embeddings = model.encode(data_chunks)

# Build the FAISS index
dimension = data_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(data_embeddings)

# User query
query = "When did Apple introduce the first Macintosh?"
query_embedding = model.encode([query])

# Search for similar embeddings
k = 2  # Number of nearest neighbors
distances, indices = index.search(query_embedding, k)

# Retrieve and print the most similar chunks
for idx in indices[0]:
    print(f"Retrieved chunk: {data_chunks[idx]}")
```

Output:

Python

```hljs yaml
Retrieved chunk: The Macintosh was introduced in 1984.
Retrieved chunk: Apple was founded as Apple Computer Company on April 1, 1976.
```

As you can see, the vector database retrieves chunks based on vector similarity, which may not preserve the exact context needed to answer the question accurately.

## The strengths and weaknesses of vector databases

Vector databases have a range of strengths and weaknesses, but the weaknesses tend to come to the forefront when companies try to use them to build RAG features. When considering a vector database versus a graph database, these weaknesses also tend to be more consequential when companies face them at an enterprise scale.

#### **Strengths**

The primary strength of vector databases is that companies can store different data types, including text and images.

Beyond sheer storage, vector databases also enable search functions that are much better than typical keyword searches. If users are looking for data that has semantic similarity, a vector database can often help them find those data points, even if there isn’t a literal keyword match.

#### **Weaknesses**

The primary weakness of vector databases comes from how vector databases process data. Especially in an enterprise context, where information retrieval needs to be granular and accurate, the crudeness of this data processing can show strain. When data is processed for vector storage, context is often lost. The relational context between data points is especially liable to get obscured.

The advantage vector databases have over keyword search — that vector databases can identify data-point similarity based on nearness (KNN and ANN) — becomes a weakness when compared to other databases. Beyond sheer numerical proximity, vector databases don’t preserve context that informs the relationships between different data points.

As Anthony Alcaraz, Chief AI Product Officer at fair hiring platform Fribl, [writes](https://towardsdatascience.com/vector-search-is-not-all-you-need-ecd0f16ad65e), “Questions often have an indirect relationship to the actual answers they seek.” For example, consider a simple description of the origins of Apple:

_Apple was founded as Apple Computer Company on April 1, 1976. The company was incorporated by Steve Wozniak and Steve Jobs in 1977. Its second computer, the Apple II, became a best seller as one of the first mass-produced microcomputers. Apple introduced the Lisa in 1983 and the Macintosh in 1984 as some of first computers to use a graphical user interface and mouse._

If this data is stored in a vector database, and a user queries, “When did Apple introduce the first Macintosh?”, correct data can turn into incorrect answers.

Given the crude chunking and the way KNN algorithms focus on mere proximity, the database might pull the closest chunks — “1983” and “Macintosh” instead of what the paragraph actually says. Given this wrong data, the LLM forms a confident but incorrect answer: “The first Macintosh was introduced in 1983.”

This loss of context is, again, especially consequential in enterprise contexts because, in these environments, data tends to be very sparse or very dense. In either case, vector searches tend to struggle or even fail to find and return relevant or complete answers. This weakness worsens in high-dimensional environments, where KNN algorithms fail to find meaningful patterns — a problem known as the “curse of dimensionality.”

Given the paucity of context, even an otherwise effective LLM will fail to formulate an accurate answer. Companies can run into the classic garbage-in and garbage-out problem: With little to no context and crude chunking, returned data points can be inaccurate or irrelevant to the query, setting LLMs up for failure. Generally, the more synthesis necessary, the worse vector databases tend to function.

Vector databases also tend to run into scalability, performance, and cost issues when used for RAG. Large datasets can make KNN algorithms inefficient, for example, and because KNN algorithms store datasets in memory, processing those often large datasets can become resource-intensive.

This intensiveness can increase quickly, too, because companies need to rerun and update the entire dataset when they enter new data. And as performance and scalability issues arise, so do costs.

## ​​What is a graph database?

Like a vector database, a graph database stores and maintains data, but its data storage structure is unique. Whereas vector databases often lose relational context, graph databases give primacy to relationships by using nodes and edges between data points to form graphs.

The unique nature of this relationship-first approach arose out of relational databases, an origin that makes graph databases worth considering for RAG. Relational databases store data in tables and organize similarly structured data closely together. Relational databases, however, don’t provide the ability to define relationships between tables.

As Kaleb Nyquist, the Database Manager at the Center for Tech & Civic Life, [writes](https://towardsdatascience.com/choose-the-right-database-model-free-your-data-from-spreadsheets-8d1129626b42), “Ironically, graph databases are actually more relationship-oriented than relational databases.”

Graph databases tend to be best for modeling densely interconnected data. Because graph databases model data similarly to object-oriented languages, the resulting database contains direct pointers to related data points.https://writer.com/wp-content/uploads/2024/10/Image-2-1-1.png?w=640

#### **Technical implementation example**

Here’s an example using Neo4j and its query language Cypher to store and query relationships:

Cypher

```hljs scss
// Create nodes
CREATE (apple:Company {name: 'Apple'})
CREATE (steveW:Person {name: 'Steve Wozniak'})
CREATE (steveJ:Person {name: 'Steve Jobs'})

// Create relationships
CREATE (steveW)-[:FOUNDED]->(apple)
CREATE (steveJ)-[:FOUNDED]->(apple)

// Query: Who founded Apple?
MATCH (founder)-[:FOUNDED]->(company {name: 'Apple'})
RETURN founder.name
```

Output:

Cypher

```hljs lua
+---------------+
| founder.name  |
+---------------+
| "Steve Wozniak"|
| "Steve Jobs"   |
+---------------+
```

This demonstrates how graph databases naturally model and query relationships between entities.

## The strengths and weaknesses of graph databases

Graph databases were a [hot topic](https://tdwi.org/articles/2017/03/14/good-bad-and-hype-about-graph-databases-for-mdm.aspx) in the database world only a few years ago, and in many cases, the emphasis on relationships makes them a useful tool. But in the years since, users of graph databases have also found weaknesses that make graph databases — without supplementation — a less-than-ideal choice for RAG purposes.

#### **Strengths**

Because they give primacy to relationships, graph databases tend to be most advantageous in contexts where the relationships between data points are essential for making meaning. Unlike relational databases, graph databases provide a native way of storing the relationships themselves, allowing developers to store relationships as memory pointers that lead from one entity to the next.

With the ability to define relationships directly, developers don’t need to worry about modeling those relationships in the database schema. Philipp Brunenberg, a ​​data engineer, [explains the benefits](https://towardsdatascience.com/at-its-core-hows-a-graph-database-different-from-a-relational-8297ca99cb8f): “We do not need to know about foreign keys, and neither do we have to write logic about how to store them. We define a schema of entities and relationships, and the system will take care of it.”

Each relationship provides context between data points, but developers can also label the nodes and edges that form these relationships, allowing developers to assign weights and directionality to each relationship. As Brunenberg writes, “​​Direction provides meaning to the relationship; they could be no direction, one-way, or two-way.”

Graph databases also tend to be more easily understandable for non-technical users because the resulting models reflect the human mind and its visual dimensions. This visualization often makes them a top choice in contexts where knowledge needs to be retrieved, modeled, and displayed.

#### **Weaknesses**

Graph databases have a few weaknesses, especially regarding efficiency and efficacy.

Graph databases often run into efficiency problems when companies use them to process large volumes of data. In an enterprise context, where there can be a lot of sparse and dense data, graph database efficiency is especially likely to plummet.

Graph databases are also less effective when used to run queries that extend across databases. The larger these databases are, the less effective these queries become.

Developers are often drawn to graph databases because they’re known to be especially good at modeling relationships, but this advantage has limitations. Graph databases can, theoretically, model relationships well, but that doesn’t mean they can create better relationships. If data is poorly captured, the search and retrieval benefits won’t be fully realized.

## What is a knowledge graph?

A knowledge graph is a data storage technique rather than a fundamentally different database. Knowledge graphs model how humans think — relationally and semantically — and go far beyond the numerical focus of vector databases and the relational focus of graph databases.

The knowledge graph technique collects and connects concepts, entities, relationships, and events using semantic descriptions of each. Each description contributes to an overall network (or graph), meaning every entity connects to the next via semantic metadata.

This technique for storing and mapping data mostly closely mimics how humans think in semantic contexts. This parallel makes it an ideal foundation for RAG-based search because RAG relies on natural language queries flowing through databases comprised primarily of semantic information.https://writer.com/wp-content/uploads/2024/10/Image-3-1-1.png?w=640

#### **Technical implementation example**

- Semantic modeling: Uses ontologies and schemas (e.g., RDF, OWL) to define the types of entities and relationships.
- SPARQL queries: Employs SPARQL for powerful querying capabilities over the semantic data.
- Integration with LLMs: The knowledge graph interfaces with LLMs to enhance natural language understanding and generation.

For example:

Python

```hljs python
from rdflib import Graph, Namespace, RDF, Literal
from rdflib.namespace import XSD

# Initialize graph
g = Graph()

# Define namespaces
EX = Namespace("http://example.org/")
g.bind('ex', EX)

# Add data to the graph
g.add((EX.Apple, RDF.type, EX.Company))
g.add((EX.Apple, EX.foundedOn, Literal("1976-04-01", datatype=XSD.date)))
g.add((EX.Steve_Wozniak, RDF.type, EX.Person))
g.add((EX.Steve_Jobs, RDF.type, EX.Person))
g.add((EX.Steve_Wozniak, EX.founded, EX.Apple))
g.add((EX.Steve_Jobs, EX.founded, EX.Apple))

# Querying the knowledge graph
query = """
PREFIX ex: <http://example.org/>

SELECT ?founderName ?companyName
WHERE {
    ?founder ex:founded ?company .
    ?founder ex:name ?founderName .
    ?company ex:name ?companyName .
}
"""

# Assuming names are added to the graph
g.add((EX.Steve_Wozniak, EX.name, Literal("Steve Wozniak")))
g.add((EX.Steve_Jobs, EX.name, Literal("Steve Jobs")))
g.add((EX.Apple, EX.name, Literal("Apple Inc.")))

# Execute the query
results = g.query(query)
for row in results:
    print(f"{row.founderName} founded {row.companyName}")
```

Output:

Python

```hljs undefined
Steve Wozniak founded Apple Inc.
Steve Jobs founded Apple Inc
```

This example showcases how knowledge graphs can store rich semantic relationships and facilitate complex queries.

## The strengths and weaknesses of knowledge graphs

Knowledge graphs, built on graph databases, have many of the same advantages that graph databases have over vector databases. Knowledge graphs, however, also present advantages over graph databases in particular contexts — especially RAG.

#### **Strengths**

Knowledge graphs, like graph databases, store data points and their relationships in a graph database. Like graph and vector databases, knowledge graphs can store a wide variety of file formats, including video, audio, and text.

But in vector databases, queries are converted into a numerical format, often meaning context is lost. In knowledge graphs, queries don’t need to be reformatted, and the graph structure that uses these queries — because it preserves semantic relationships — allows for much more accurate retrieval than KNN or ANN algorithms can offer.

Search and retrieval – central to RAG — is especially effective in knowledge graphs. In enterprise contexts, the differentiator between effective and ineffective search is often the ability to synthesize data across multiple sources. Knowledge graphs encode topical, semantic, temporal, and entity relationships into their graph structure, making synthesis possible.

Relationships, however, aren’t always linear or one-way. With knowledge graphs, developers can encode hierarchies and other structural relationships. Given these structural relationships, knowledge graphs can map the connections between different points in different sources, even if they reference the same entities.

“In contrast,” Alcaraz writes, “standard vector search lacks any notion of these structural relationships. Passages are treated atomically without any surrounding context.”

Context loss is one of the most common weaknesses of other databases, especially when used for RAG. With knowledge graphs, contextual information is retained because it is encoded in the retrieved information.

#### **Weaknesses**

Due to their emphasis on semantic information, knowledge graphs tend to have a lot of data to condense, often resulting in the need for significant computational power to support them. Operations running across knowledge graphs can sometimes be expensive, and that costliness can make them difficult to scale.

And, similar to the weaknesses inherent to graph databases, knowledge graphs can’t take on the work of capturing and cleaning data well. Similarly, an effective knowledge graph will be hampered by an LLM that can’t formulate readable answers without hallucinations.

</details>


## Code Sources

<details>
<summary>Repository analysis for https://github.com/towardsai/agentic-ai-engineering-course/blob/dev/lessons/09_memory_knowledge_access/notebook.ipynb</summary>

# Repository analysis for https://github.com/towardsai/agentic-ai-engineering-course/blob/dev/lessons/09_memory_knowledge_access/notebook.ipynb

## Summary
Repository: towardsai/agentic-ai-engineering-course
Branch: dev
File: notebook.ipynb
Lines: 333

Estimated tokens: 3.3k

## File tree
```Directory structure:
└── notebook.ipynb

```

## Extracted content
================================================
FILE: lessons/09_memory_knowledge_access/notebook.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Lesson 9: Memory for Agents

This lesson explores the concept of adding **long-term memory** to agents, so they can persist and retrieve information over time. 

We’ll implement semantic, episodic, and procedural memory using the open-source mem0 library with Google's Gemini text embedding model, and a vector store that runs locally in the notebook, using ChromaDB. 


Learning Objectives:

1. Understand the different types of memory 
2. How to implement them, using the mem0 library.
"""

"""
## 1. Setup

First, we define some standard Magic Python commands to autoreload Python packages whenever they change:
"""

%load_ext autoreload
%autoreload 2

"""
### Set Up Python Environment

To set up your Python virtual environment using `uv` and use it in the Notebook, follow the step-by-step instructions from the [Course Admin](https://academy.towardsai.net/courses/take/agent-engineering/multimedia/67469688-lesson-1-part-2-course-admin) lesson from the beginning of the course.

**TL/DR:** Be sure the correct kernel pointing to your `uv` virtual environment is selected.
"""

"""
### Configure Gemini API

To configure the Gemini API, follow the step-by-step instructions from the [Course Admin](https://academy.towardsai.net/courses/take/agent-engineering/multimedia/67469688-lesson-1-part-2-course-admin) lesson.

But here is a quick check on what you need to run this Notebook:

1.  Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  From the root of your project, run: `cp .env.example .env` 
3.  Within the `.env` file, fill in the `GOOGLE_API_KEY` variable:

Now, the code below will load the key from the `.env` file:
"""

from lessons.utils import env

env.load(required_env_vars=["GOOGLE_API_KEY"])
# Output:
#   Trying to load environment variables from `/Users/omar/Documents/ai_repos/course-ai-agents/.env`

#   Environment variables loaded successfully.


"""
### Import Key Packages
"""

import os
import re
from typing import Optional

from google import genai
from mem0 import Memory

"""
### Initialize the Gemini Client
"""

client = genai.Client()

"""
### Define Constants

We will use the `gemini-2.5-flash` model, which is fast and cost-effective:
"""

MODEL_ID = "gemini-2.5-flash"

"""
### Configure mem0 (Gemini LLM + embeddings + local vector store)

Here we instantiate mem0 with:

- LLM: our existing Gemini model (`MODEL_ID = "gemini-2.5-flash"`) for the summarization/extraction of facts.
- Embeddings: Gemini’s `text-embedding-004` (dimension 768).
- Vector store:
    - ChromaDB with `MEM_BACKEND=chromadb` 
"""

MEM0_CONFIG = {
    # Use Google's text-embedding-004 (768-dim) for embeddings
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "text-embedding-004",
            "embedding_dims": 768,
            "api_key": os.getenv("GOOGLE_API_KEY"),
        },
    },
    # Use ChromaDB as a local, in-notebook vector store (ephemeral, in-memory)
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "lesson9_memories",
        },
    },
    "llm": {
        "provider": "gemini",
        "config": {
            "model": MODEL_ID,
            "api_key": os.getenv("GOOGLE_API_KEY"),
        },
    },
}

memory = Memory.from_config(MEM0_CONFIG)
MEM_USER_ID = "lesson9_notebook_student"
memory.delete_all(user_id=MEM_USER_ID)
print("✅ Mem0 ready (Gemini embeddings + in-memory Chroma).")
# Output:
#   ✅ Mem0 ready (Gemini embeddings + in-memory Chroma).


"""
### Helper functions: add/search memory

A small wrapper layer around mem0 to:

- **Save** a memory string (or a short conversation) and tag it with a `category` (`semantic`, `episodic`, `procedure`) plus metadata.
    - `mem_add_text` will store the raw string, with `infer=False` (no LLM used)
    - `mem_add_conversation` will store an LLM-generated summary (gemini-2.5-flash in our case) of the conversation with `infer=True`

- **Search** memory and return results for display. It can also filter by category.

For the metadata, mem0 only allows primitive types (str, int, float, bool, None) that is why we convert to `str` any non-primitive values.
"""

def mem_add_text(text: str, category: str = "semantic", **meta) -> str:
    """Add a single text memory. No LLM is used for extraction or summarization."""
    metadata = {"category": category}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            metadata[k] = v
        else:
            metadata[k] = str(v)
    memory.add(text, user_id=MEM_USER_ID, metadata=metadata, infer=False)
    return f"Saved {category} memory."


def mem_add_conversation(messages: list[dict], category: str = "episodic", **meta) -> str:
    """Add a conversation (list of {role, content}) as one episode."""
    metadata = {"category": category}
    for k, v in meta.items():
        metadata[k] = v if isinstance(v, (str, int, float, bool)) or v is None else str(v)
    memory.add(messages, user_id=MEM_USER_ID, metadata=metadata, infer=True)
    return f"Saved {category} episode."


def mem_search(query: str, limit: int = 5, category: Optional[str] = None) -> list[dict]:
    """
    Category-aware search wrapper.
    Returns the full result dicts so we can inspect metadata.
    """
    res = memory.search(query, user_id=MEM_USER_ID, limit=limit) or {}
    items = res.get("results", [])
    if category is not None:
        items = [r for r in items if (r.get("metadata") or {}).get("category") == category]
    return items

"""
## 2. Semantic memory example (facts as atomic strings)

**Goal**: We show semantic memory as “facts & preferences” stored as short, individual strings.

- We insert a few example facts (e.g., “User has a dog named George”).

- Then we search with a natural query (e.g., “brother job”) and see the relevant fact returned.
"""

facts: list[str] = [
    "User prefers vegetarian meals.",
    "User has a dog named George.",
    "User is allergic to gluten.",
    "User's brother is named Mark and is a software engineer.",
]
for f in facts:
    print(mem_add_text(f, category="semantic"))

print("\nSearch --> 'brother job':")
print("\n".join(f"- {m}" for m in mem_search("brother job", limit=3)))
# Output:
#   Saved semantic memory.

#   Saved semantic memory.

#   Saved semantic memory.

#   Saved semantic memory.

#   

#   Search --> 'brother job':

#   - {'id': 'c6f0edbc-e0cc-40a2-abb4-d3f364b009ba', 'memory': "User's brother is named Mark and is a software engineer.", 'hash': '9a01dbd8ea8b96f8ed9c84e9dcdb55a1', 'metadata': {'category': 'semantic'}, 'score': 0.9269160032272339, 'created_at': '2025-08-25T17:26:22.183416-07:00', 'updated_at': None, 'user_id': 'lesson9_notebook_student', 'role': 'user'}

#   - {'id': 'bd98bf8d-8d85-4103-87b8-d6b32b8b9eb9', 'memory': 'User has a dog named George.', 'hash': '8c592a46b362bab2fd20a1a2a9214d74', 'metadata': {'category': 'semantic'}, 'score': 1.4484589099884033, 'created_at': '2025-08-25T17:26:21.261837-07:00', 'updated_at': None, 'user_id': 'lesson9_notebook_student', 'role': 'user'}

#   - {'id': 'ce5ccb85-60d2-4a67-8361-be990f9100f2', 'memory': 'User prefers vegetarian meals.', 'hash': '0034073d2dbe31e303972a0599565525', 'metadata': {'category': 'semantic'}, 'score': 1.5473814010620117, 'created_at': '2025-08-25T17:26:20.721037-07:00', 'updated_at': None, 'user_id': 'lesson9_notebook_student', 'role': 'user'}


"""
## 3. Episodic memory example (summarize 3–4 turns → one episode)

**Goal**: Demonstrate episodic memory (experiences & history).

- We create a short 3–4 turn exchange between user and assistant.

- We ask the LLM to produce a concise episode summary (1–2 sentences) and save it under category="episodic".

- Finally, we run a semantic search (e.g., “deadline stress”) to retrieve that episode, we print the memory along with its creation timestamp.

This example show how an agent can compress transient chat into a single durable “moment.”

Since mem0 by default creates a created_at timestamp, we have the possibility to use it to sort and filter memories.
It would then be possible to answer questions like "What did we talk about last week?"
"""

# A short 4-turn exchange we want to compress into one "episode"
dialogue = [
    {"role": "user", "content": "I'm stressed about my project deadline on Friday."},
    {"role": "assistant", "content": "I’m here to help—what’s the blocker?"},
    {"role": "user", "content": "Mainly testing. I also prefer working at night."},
    {"role": "assistant", "content": "Okay, we can split testing into two sessions."},
]

# Ask the LLM to write a clear episodic summary.
episodic_prompt = f"""Summarize the following 3–4 turns as one concise 'episode' (1–2 sentences).
Keep salient details and tone.

{dialogue}
"""
summary_resp = client.models.generate_content(model=MODEL_ID, contents=episodic_prompt)
episode = summary_resp.text.strip()

print(
    mem_add_text(
        episode,
        category="episodic",
        summarized=True,
        turns=4,
    )
)

print("\nSearch --> 'deadline stress'")
hits = mem_search("deadline stress", limit=3, category="episodic")
for h in hits:
    print(f"- [created_at={h.get('created_at')}] {h['memory']}")
# Output:
#   Saved episodic memory.

#   

#   Search --> 'deadline stress'

#   - [created_at=2025-08-25T17:26:28.260452-07:00] Stressed about a looming Friday project deadline, the user identified testing as their main blocker and noted a preference for working at night. The assistant offered support by proposing they split the testing into two sessions.


"""
## 4. Procedural memory example (learn & “run” a skill)

**Goal**: Demonstrate procedural memory (skills & workflows).

- We teach the agent a small procedure (e.g., monthly_report) by saving ordered steps in a single text block under category="procedure".

- We retrieve the procedure and parse the numbered steps to simulate “running” it.

This example shows how agents can learn reusable playbooks and trigger them later by name.
"""

def learn_procedure(name: str, steps: list[str]) -> str:
    body = "Procedure: " + name + "\nSteps:\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    return mem_add_text(body, category="procedure", procedure_name=name)


def find_procedure(name: str) -> dict | None:
    # search broadly but only keep category=procedure
    hits = mem_search(name, limit=10, category="procedure")
    # Prefer an exact name match if available
    for h in hits:
        if (h.get("metadata") or {}).get("procedure_name") == name:
            return h
    return hits[0] if hits else None


def run_procedure(name: str) -> str:
    p = find_procedure(name)
    if not p:
        return f"Procedure '{name}' not found."
    text = p.get("memory", "")
    steps = [m.group(1).strip() for m in re.finditer(r"^\s*\d+\.\s+(.*)$", text, flags=re.MULTILINE)]
    if not steps:
        return f"Procedure '{name}' has no parseable steps.\n\n{text}"
    lines = [f"→ {s}" for s in steps]
    return f"Running procedure '{name}':\n" + "\n".join(lines)


# Teach the agent a tiny recurrent skill:
print(
    learn_procedure(
        "monthly_report",
        [
            "Query sales DB for the last 30 days.",
            "Summarize top 5 insights.",
            "Ask user whether to email or display.",
        ],
    )
)

print("\nRetrieve and 'run' it:")
proc = find_procedure("monthly_report")
print(proc or "Not found.")
# Output:
#   Saved procedure memory.

#   

#   Retrieve and 'run' it:

#   {'id': 'ab0940fe-c9a4-446f-b3ff-cbaea099faee', 'memory': 'Procedure: monthly_report\nSteps:\n1. Query sales DB for the last 30 days.\n2. Summarize top 5 insights.\n3. Ask user whether to email or display.', 'hash': 'e66b82a0dcc57034cfa0a54084a643b5', 'metadata': {'procedure_name': 'monthly_report', 'category': 'procedure'}, 'score': 0.9708564281463623, 'created_at': '2025-08-25T17:26:28.965650-07:00', 'updated_at': None, 'user_id': 'lesson9_notebook_student', 'role': 'user'}

</details>


## YouTube Video Transcripts

<details>
<summary>Here is the enriched transcript of the video.</summary>

Here is the enriched transcript of the video.

(Intro sequence with upbeat music showing attendees at a tech event. People are listening to speakers, networking, and mingling. Banners for "LangGraph" and "LangSmith" are visible.)

Thank you Nicole and thank you um Harrison and LangChain and Greg for organizing and hosting. Actually one of the first things I did with memory was with Harrison on the original memory implementation in LangChain, so very full circle. Um cool. So for those of you who do not know New Computer and what we do, we have Dot, which is a conversational journal. It's in the App Store. You can use it now. We launched this last year.

[00:30]
(A slide appears with the title "From Dot to Dots, Evolution of Memory at New Computer" and the speaker's name "Sam Whitmore, CEO".)

So we've been working on memory in AI application since 2023. Um, cool. So take us back to 2023. At the time GPT-4 was state of the art.

[01:00]
(A slide titled "2023" is shown. It transitions to a new slide with bullet points: "GPT-4 was state of the art", "8192 token context length", "196ms per generated token", "$30.00 / 1 million prompt tokens", "$60.00 / 1 million sampled tokens".)

We have 8,000 length token uh prompt, very slow and very expensive. So I want to walk you through some of the things that we tried initially, lessons we learned along the way, and how we kind of evolve as underlying technology evolves.

*The speaker, Sam Whitmore, CEO of New Computer, introduces her company's conversational journal app, Dot, and sets the stage by describing the state of AI technology in 2023, characterized by GPT-4's limited context, slow speed, and high cost.*

So, when we started, our general goal was to build a personal AI that got to know you. (A screen recording of the Dot app shows a conversational interface.) It was pretty unstructured. Um and so we knew that if it was going to learn about you as you used it, it needed memory. So we were like, okay, let's just build the first build the perfect memory architecture and then the product after that. Um so we started out being like, okay, maybe we can just extract facts as a user talks to Dot and search across them, you know, use some different techniques and we'll have great memory performance.

[01:30]
(A slide asks "Memory == Facts...?".)

So, we learned pretty quickly that this wasn't really going to work for us. So, imagine a user saying, "I have a dog. His name is Poppy. Walking him is the best part of my day." So early extraction we'd get things like, user has dog, user's dog is named Poppy, user likes taking Poppy for walks. There's a lot of nuance missing. So like you can tell a lot about a person from reading that sentence that you can't tell from those facts.

[02:00]
(A slide shows "Memory != Facts". It transitions to "Memory == Schemas?".)

That was pretty quick realization for us. We then moved on. So we were like, maybe if we try to summarize everything about Poppy in one place, then it's going to perform better. We decided that we're going to make this universal memory architecture with entities and schemas that were linked to each other.

[02:30]
(A screen recording shows the Dot app's UI for organizing information, with categories like "Recipes".)

This was a UI representation of it. Um so users could actually browse the things that were created. Um and they had different types and on the back end there was different form factors with JSON blobs. This is a real example from our product at the time. So I sent it a bachelorette flyer and it made like a whole bunch of different memory types with schemas associated.

[03:00]
(The screen shows the app's interface breaking down the bachelorette party information into structured data like events, people, and locations.)

Um so you can see here that like this is what the backend data looked like. There's different fields and we had a router architecture that would kind of generate queries that would search across all of these um in parallel. And what we found was that it worked okay, but there was kind of some base functionality that was still missing.

[03:30]
(A funny example is shown: a screenshot of a user's memory schemas, which includes categories like "Routines," "Ice Cream Flavors," "Drunk Texts," and "Webpages.")

Um oh, this was a funny example. Um Jason, my co-founder, was sending it uh pictures and it made him a "drunk text" category as a schema. Um which we're like, that feels like a heavy read. Um but anyway, so the schemas are kind of fun. Um but yes, so basically we also saw that when we exposed this to users, it was like too much cognitive overhead for them to garden their their database. Like there's a lot of overlapping concepts and people got stressed by actually just monitoring their memory base.

*Whitmore details New Computer's early attempts at AI memory, first by extracting simple facts, which proved too shallow, and then by using a more structured schema-based system, which created too much cognitive overhead for users.*

So, again, we were like, okay, let's just go back to basics here and figure out like what do we want our product to be doing and let's re-examine how we want to build memory from that.

[04:00]
So, we looked again at like what a thought partner should have to do to actually be really good as a listener for you. So, we realized like it should always know who you are and your core values. It should know basically like, you know, what you talked about yesterday, what you talked about last week, and again, like who Poppy is, if Poppy is your dog, who your co-founder is, stuff like that.

[04:30]
(A slide appears listing four requirements for a thought partner: 1. It needs to know my general bio & core values, 2. It needs to know the things that happen in my life and also when they happened, 3. It needs to know about the people, places, and the various nouns important to me, 4. It needs to know the best way to work with me. This transitions to a slide mapping these concepts to memory types: Holistic theory of mind, Episodic memory, Entities, and Procedural memory.)

And it also needs to know about like your behavior preferences and how it should adapt to you as you use it. So, we ended up making four kind of parallel memory systems. So the schemas that you saw, they didn't really go away, they just became one of the memory systems, the entities. And it's funny seeing Will kind of say some of the same ones. There's like an example of convergent evolution because we kind of made these up ourselves. But basically like how holistic theory of mind um here's mine.

[05:00]
(A slide titled "Holistic Theory of Mind" shows a detailed personal profile with categories: Family, Career, Interests & Passions, and Current Focus.)

It's kind of just like who am I, what's important to me, what am I working on, what's top of mind for me right now. Episodic memory is kind of like what happened on a specific day. Here's kind of like an actual real example soon after I had my baby last year. Um here here's like another entity example.

[05:30]
(A slide on "Episodic Memory" shows a detailed summary of a conversation from a specific date. This transitions to a slide on "Entities" with a detailed summary about a person named Alexander.)

We ended up stripping away a lot of the JSON because it turned out to not improve performance in retrieval across the entity schema. So we kept things like the categories if we wanted to do tag filtering, but a lot of the extra structure just ended up being like way too much overhead for the model to output. And finally we made this thing called procedural memory, which is basically like triggered by uh conversational and situational similarity.

[06:00]
(A slide titled "Procedural Memory" shows Python code for an intent called "ReflectionQuestionIntent".)

So what you're looking at here is this intent, and if you're a Dot user, you'll probably recognize this behavior. It says, "Choose this if you have sensed a hidden or implied emotion or motivation that the user is not expressing in their language, and see a chance to share an insight or probe the user deeper on this matter." And then when it detects that this is happening, it says like, "share an insight, you know, ask a question, issue a statement that encourages the same behavior." And so basically like the trigger here is not semantic similarity but situational similarity.

[06:30]
(A slide appears showing a complex flowchart titled "Retrieval pipeline 2024".)

I see a lot of overlap here for people building agents where if you have a workflow that the agent needs to perform, it can identify that it encountered that situation before and kind of pull up some learning it had from the past running of the workflow. So this is kind of our way our retrieval pipeline worked in 2024, which is like parallelize retrieval across all of these systems. So here's a query which is very hard to read, so maybe these slides will be accessible separately.

[07:00]
Um what restaurant should I take my brother to for his birthday? And in this sense, in each of our four systems, we detect if a query is necessary across the system for a holistic stuff we always load the whole theory of mind. Episodic is only triggered if it's like, what did we talk about yesterday or what did we talk about last week. And then here, there's two different types of entity queries detected like brother and restaurants. And then we would do kind of a hybrid search thing where like we mixed together BM25, semantic keyword, basically like no attachment to any particular approach, just like whatever improved recall for specific entities.

[07:30]
Um and then the procedural memory, here if there's a behavioral module loading like restaurant selection or planning, then that would get loaded into the final prompt. So funny thing also is when we launched people tried to prompt inject us, but because we have so many different behavioral modules and different things going on, we called it like Franken-prompt, and like if people did prompt inject us, they'd be like, wait, I think this prompt changes every time, which it did.

*The speaker describes their revised, multi-system memory architecture, which includes holistic theory of mind, episodic memory, entities, and procedural memory, all working in parallel within a complex retrieval pipeline.*

Um okay. So for the formation for these, again, really distinct per system. So holistic theory of mind, you don't need to update that frequently.

[08:00]
(A slide titled "The formation of these memories are distinct per system" is shown, followed by a slide breaking down the update frequency for each memory type: "Holistic theory of mind: Updated once a day", "Episodic memory: Periodic summarization, multiple levels of precision", "Entities: Per line of conversation (with dream sequences for consolidation)", "Procedural memory: Per line of conversation".)

Episodic is like periodic summarization. So like if you want to have it be per week, you might update across daily summaries once per week, per day, once per day, etc. Entities we did per line of conversation and then we would run kind of cron jobs that we called dream sequences where they'd identify possible duplicates and potentially merge them. And procedural memory also updated per line of conversation. So, um along with the past year, our product trajectory has changed. We're now building Dots, which is a hivemind.

[08:30]
(A slide says "Hivemind".)

So it's like instead of remembering just one person that it meets, it actually remembers an entire group of people. Um yeah, so it's like many dots. Stores the relationships between everyone. Um yeah, so you basically some of the added challenges we're dealing with now are representing um different people's opinion of each other, how they're connected, and how information should flow between them in addition to understanding all of the systems I just mentioned above.

[09:00]
(A slide says "June, 2025" and then transitions to a new slide with details about "Gemini flash 2.5": "Maximum input tokens: 1,048,576", "Maximum output tokens: 65,535", "$0.30 / 1 million input", "$2.50 / 1 million output".)

So, one other thing I'll share that has evolved in terms of how like the world has changed a lot since 2023. So we keep re-evaluating how we should be building things constantly. And now we have a million token input context window. We have prompts that are really cheap, and they're also really, really fast.

[09:30]
(The previous "Retrieval pipeline 2024" flowchart reappears.)

So some of the things that we held true in terms of compressing knowledge and context, we no longer hold true. Here's an example. So if you look back at this pipeline I shared before, um here's an updated version that we're experimenting with now, which is getting rid of episodic and entity level compression in favor of real-time Q&A. So that means that like depending on your system, maybe you don't have to be compressing context at all.

[10:00]
(An updated flowchart, "Retrieval pipeline 2025," is shown, illustrating a revised architecture focused on real-time Q&A over chunked conversation history instead of pre-compressed memory.)

Because again, like I said at the beginning, the raw data is always the best source of truth. So it's like why would you create a secondary artifact as a stepping point between you and what the user is asking. Ideally you just want to examine the context. And so we do that pretty frequently depending on how much data we're dealing with. We try basically not to do to do the minimal amount of engineering possible. And our theory kind of going forward is like this trend will only continue.

[10:30]
(A final slide appears: "The perfect memory architecture doesn't exist. Know what function memory serves in your product, & think from first principles about how to make it work...constantly!")

So we think the procedural memory and like basically the insights, the interpretation and analysis that the thing does is the important part of memory. It's like the record of its thoughts about you and kind of its notes to itself is the important part. You can almost separate that from retrieval as a problem. You can say like, okay, maybe there'll be an infinite log of like my interactions and model notes will be interpolated in in the in the future. And so maybe we don't even have to deal with retrieval and context compression at all. So, I guess if I want you guys to take away one thing, it's like the perfect memory architecture doesn't exist. And start with kind of what your product is supposed to do and then think from first principles about how to make it work and do that all the time because the world is changing and you might not need to invest that much in memory infrastructure.

[11:30]
(A slide with the speaker's and company's social media handles appears: "Sam Whitmore, CEO @sjwhitmore" and "New Computer @newcomputer". The audience applauds.)

That's it. So you can follow us at Twitter, New Computer. (A final title card for the "AI Memory" event with Sam Whitmore's name and title is shown.)

</details>


## Additional Sources Scraped

<details>
<summary>agent-memory-letta</summary>

Agent memory is what enables AI agents to maintain persistent state, learn from interactions, and develop long-term relationships with users. Unlike traditional chatbots that treat each conversation as isolated, agents with sophisticated memory systems can build understanding over time.

</details>

<details>
<summary>ai-agent-memory-short-long-term-rag-agentic-rag</summary>

Designing a robust memory layer for your agents is one of the most underrated aspects of building AI applications. Memory sits at the core of any AI project, guiding how you implement your RAG (or agentic RAG) algorithm, how you access external information used as context, manage multiple conversation threads, and handle multiple users. All critical aspects of any successful agentic application.

Every agent has short-term memory and some level of long-term memory. Understanding the difference between the two and what types of long-term memory exist is essential to knowing what to adopt in your toolbelt and how to design your AI application system and business logic.

## 1. Short-term vs. long-term memory

AI memory systems can be broadly categorized into two main types: short-term and long-term.

Figure 2: Short-term vs. long-term memory

#### Short-term memory (or working memory)

Short-term memory, often called working memory, is the temporary storage space where an agent holds information it's currently using. This memory typically maintains active information like the current conversation context, recent messages, and intermediate reasoning steps.

Working memory is essential for agents to maintain coherence in conversations. Without it, your agent would respond to each message as if it were the first one, losing all the context and creating a frustrating user experience. The main limitation of working memory is its capacity - it can only hold a limited amount of information at once. In language models, this is directly related to the context window, which determines how much previous conversation and metadata the model can "see" when generating a response.

When implementing working memory in your agent, you need to decide what information to keep and what to discard. Most agents keep the most recent parts of a conversation, but more sophisticated approaches might prioritize keeping important information while summarizing or removing less critical details. This helps make the most efficient use of the limited working memory space.

#### Long-term memory: Semantic memory

Semantic memory stores factual knowledge and general information about the world. It's where your agent keeps the knowledge it has learned that isn't tied to specific experiences. This includes concepts, facts, ideas, and meanings that help the agent understand the world.

For AI assistants, semantic memory might include information about different topics, how to respond to certain types of questions, or facts about the world. This is what enables your agent to answer questions like "What's the capital of France?" or understand the concept of a vacation without needing to have experienced one.

In practice, semantic memory in AI systems is often implemented through vector databases that store information in a way that can be quickly searched and retrieved. When a user asks a question, the agent can search its semantic memory for relevant information to respond accurately.

#### Long-term memory: Procedural memory

Procedural memory contains knowledge about how to do things, such as performing tasks or following specific processes.

When building agents, procedural memory often takes the form of functions, algorithms, or code that defines how the agent should act in different situations. This could be as simple as a template for greeting users or as complex as a multi-step reasoning process for solving tough problems. Unlike semantic memory, which stores what the agent knows, procedural memory stores how the agent applies that knowledge.

#### Long-term memory: Episodic memory

Episodic memory stores specific past experiences and events. In humans, these are our autobiographical memories - the things that happened to us at particular times and places. For AI agents, episodic memory allows them to remember past user interactions and learn from those experiences.

With episodic memory, your agent can recall previous conversations with a specific user, remember preferences they've expressed, or reference shared experiences. This creates a sense of continuity and personalization, making interactions feel more natural and helpful. When a user says, "Let's continue where we left off yesterday," an agent with episodic memory can do that.

Implementing episodic memory typically involves implementing a RAG-like system on top of your conversation histories. Like this, you can move your short-term memory into the long-term memory, extracting only chunks of past conversation that are helpful to answer present queries (instead of keeping the whole history in the context window).

When implementing AI agents, you always have short-term memory. Depending on your use case, you have one or more types of long-term memory, where the procedural and semantic ones are the most common.

</details>

<details>
<summary>cognitive-architectures-for-language-agents</summary>

Language models are _stateless_: they do not persist information across calls. In contrast, language agents may store and maintain information internally for multi-step interaction with the world. Under the CoALA framework, language agents explicitly organize information (mainly textural, but other modalities also allowed) into multiple memory modules, each containing a different form of information. These include short-term working memory and several long-term memories: episodic, semantic, and procedural.

Working memory. Working memory maintains active and readily available information as symbolic variables for the current decision cycle (Section [4.6](https://arxiv.org/html/2309.02427v3#S4.SS6 "4.6 Decision making ‣ 4 Cognitive Architectures for Language Agents (CoALA): A Conceptual Framework ‣ Cognitive Architectures for Language Agents")). This includes perceptual inputs, active knowledge (generated by reasoning or retrieved from long-term memory), and other core information carried over from the previous decision cycle (e.g., agent’s active goals). Previous methods encourage the LLM to generate intermediate reasoning (Wei et al., [2022b](https://arxiv.org/html/2309.02427v3#bib.bib203 ""); Nye et al., [2021](https://arxiv.org/html/2309.02427v3#bib.bib146 "")), using the LLM’s own context as a form of working memory. CoALA’s notion of working memory is more general: it is a data structure that persists across LLM calls. On each LLM call, the LLM input is synthesized from a subset of working memory (e.g., a prompt template and relevant variables). The LLM output is then parsed back into other variables (e.g., an action name and arguments) which are stored back in working memory and used to execute the corresponding action (Figure [3](https://arxiv.org/html/2309.02427v3#S3.F3 "Figure 3 ‣ 3.2 Prompt engineering as control flow ‣ 3 Connections between Language Models and Production Systems ‣ Cognitive Architectures for Language Agents") A).
Besides the LLM, the working memory also interacts with long-term memories and grounding interfaces. It thus serves as the central hub connecting different components of a language agent.https://arxiv.org/html/2309.02427v3/x5.pngFigure 5: Agents’ action spaces can be divided into internal memory accesses and external interactions with the world. Reasoning and retrieval actions are used to support planning.
Episodic memory. Episodic memory stores experience from earlier decision cycles. This can consist of training input-output pairs (Rubin et al., [2021](https://arxiv.org/html/2309.02427v3#bib.bib166 "")), history event flows (Weston et al., [2014](https://arxiv.org/html/2309.02427v3#bib.bib205 ""); Park et al., [2023](https://arxiv.org/html/2309.02427v3#bib.bib153 "")), game trajectories from previous episodes (Yao et al., [2020](https://arxiv.org/html/2309.02427v3#bib.bib222 ""); Tuyls et al., [2022](https://arxiv.org/html/2309.02427v3#bib.bib193 "")), or other representations of the agent’s experiences. During the planning stage of a decision cycle, these episodes may be retrieved into working memory to support reasoning. An agent can also write new experiences from working to episodic memory as a form of learning (Section [4.5](https://arxiv.org/html/2309.02427v3#S4.SS5 "4.5 Learning actions ‣ 4 Cognitive Architectures for Language Agents (CoALA): A Conceptual Framework ‣ Cognitive Architectures for Language Agents")).

Semantic memory. Semantic memory stores an agent’s knowledge about the world and itself.
Traditional NLP or RL approaches that leverage retrieval for reasoning or decision-making initialize semantic memory from an external database for knowledge support.
For example, retrieval-augmented methods in NLP (Lewis et al., [2020](https://arxiv.org/html/2309.02427v3#bib.bib95 ""); Borgeaud et al., [2022](https://arxiv.org/html/2309.02427v3#bib.bib13 ""); Chen et al., [2017](https://arxiv.org/html/2309.02427v3#bib.bib25 "")) can be viewed as retrieving from a semantic memory of unstructured text (e.g., Wikipedia).
In RL, “reading to learn” approaches (Branavan et al., [2012](https://arxiv.org/html/2309.02427v3#bib.bib14 ""); Narasimhan et al., [2018](https://arxiv.org/html/2309.02427v3#bib.bib127 ""); Hanjie et al., [2021](https://arxiv.org/html/2309.02427v3#bib.bib59 ""); Zhong et al., [2021](https://arxiv.org/html/2309.02427v3#bib.bib232 "")) leverage game manuals and facts as a semantic memory to affect the policy.
While these examples essentially employ a fixed, read-only semantic memory, language agents may also write new knowledge obtained from LLM reasoning into semantic memory as a form of learning (Section [4.5](https://arxiv.org/html/2309.02427v3#S4.SS5 "4.5 Learning actions ‣ 4 Cognitive Architectures for Language Agents (CoALA): A Conceptual Framework ‣ Cognitive Architectures for Language Agents")) to incrementally build up world knowledge from experience.

Procedural memory. Language agents contain two forms of procedural memory: _implicit_ knowledge stored in the LLM weights, and _explicit_ knowledge written in the agent’s code. The agent’s code can be further divided into two types: procedures that implement actions (reasoning, retrieval, grounding, and learning procedures), and procedures that implement decision-making itself (Section [4.6](https://arxiv.org/html/2309.02427v3#S4.SS6 "4.6 Decision making ‣ 4 Cognitive Architectures for Language Agents (CoALA): A Conceptual Framework ‣ Cognitive Architectures for Language Agents")). During a decision cycle, the LLM can be accessed via reasoning actions, and various code-based procedures can be retrieved and executed. Unlike episodic or semantic memory that may be initially empty or even absent, procedural memory must be initialized by the designer with proper code to bootstrap the agent. Finally, while learning new actions by writing to procedural memory is possible (Section [4.5](https://arxiv.org/html/2309.02427v3#S4.SS5 "4.5 Learning actions ‣ 4 Cognitive Architectures for Language Agents (CoALA): A Conceptual Framework ‣ Cognitive Architectures for Language Agents")), it is significantly riskier than writing to episodic or semantic memory, as it can easily introduce bugs or allow an agent to subvert its designers’ intentions.

</details>

<details>
<summary>giving-your-ai-a-mind-exploring-memory-frameworks-for-agenti</summary>

Hey everyone, Richardson Gunde here! Ever feel like you’re having a conversation with a goldfish? You tell it something, it seems to listen… then, poof! It forgets everything the second you finish speaking. That’s often the experience with many chatbots — they lack the crucial ingredient of _memory_. But what if we could give our AI assistants a proper memory, a real mind to hold onto information and learn from past experiences? That’s what we’re diving into today.

This isn’t just about remembering the last few messages; it’s about building truly _agentic_ systems — AI that can learn, adapt, and even anticipate your needs. We’re going to explore different memory frameworks inspired by human cognition, and I’ll show you how to implement them using LangChain and other tools. Get ready for an “Aha!” moment or two — this is where the magic happens.

**The Stateless Nature of Language Models:** _A Fundamental Limitation_

Think about how a language model works. Every time you send a prompt, it’s essentially a brand new start. It’s stateless; it doesn’t inherently remember anything from previous interactions unless you explicitly feed it that context. This is a huge limitation when building agents that need to handle complex tasks or ongoing conversations.https://miro.medium.com/v2/resize:fit:695/1*OF5rIU6UCdIF1jslIQk0zw.png

Now, contrast that with how _you_ approach problem-solving. You bring a wealth of knowledge — your general knowledge of the world, memories of past experiences, lessons learned from successes and failures. This allows you to instantly contextualize a situation and adapt your approach accordingly. We, as humans, have something language models currently lack: advanced memory and the ability to learn and apply those learnings to new situations.

**Bridging the Gap:** _Modeling Human Memory in AI_

To overcome this limitation, we can borrow concepts from psychology and model different forms of memory within our agentic system design. We’ll focus on four key types:

1. **Working Memory:** This is your immediate cognitive workspace, the “RAM” of your mind. For a chatbot, it’s the current conversation and its context. Think of it as the short-term memory of the interaction, keeping track of the back-and-forth between user and AI. Remembering in this context is simply accessing this recent data, while learning involves dynamically integrating new messages to update the overall conversational state.https://miro.medium.com/v2/resize:fit:480/1*60lIG7SeVeXCc0F1sL7WMQ.png

**2 . Episodic Memory:** This is your long-term memory for specific events. For a chatbot, it’s a collection of past conversations and the takeaways from them. Remembering here involves recalling similar past events and their outcomes to guide current interactions. Learning involves storing complete conversations and analyzing them to extract key insights — what worked, what didn’t, and what to avoid in the future. This is where the AI starts to truly learn from experience.https://miro.medium.com/v2/resize:fit:505/1*3p0USam0ju55foTCOJPfIQ.png

**3\. Semantic Memory:** This represents your structured knowledge of facts, concepts, and their relationships — the “what you know”. For our agent, this will be a database of factual knowledge that’s dynamically retrieved to ground responses. Learning involves expanding or refining this knowledge base, while remembering involves retrieving and synthesizing relevant information to provide accurate and contextually appropriate answers.https://miro.medium.com/v2/resize:fit:263/1*5AEUHbncyyknq2J5d4iraQ.png

**4\. Procedural Memory:** This is the “how to” memory, encompassing the skills and routines you’ve learned. For a language model, this is trickier. It’s partially represented in the model’s weights, but also in the code that orchestrates the memory interactions. Learning here could involve fine-tuning the model or updating the system’s code, which can be complex. We’ll explore a simplified approach using persistent instructions that guide the agent’s behavior.https://miro.medium.com/v2/resize:fit:502/1*-VjPhLIGXaUe8QRqQKPM2A.png

**Implementing the Memory Frameworks:** _A Practical Example_

Let’s get our hands dirty! We’ll use LangChain to build a retrieval-augmented generation agent that models these four memory types.

**1\. Working Memory: The Immediate Context**

_The simplest to implement is working memory. We’ll use a list to store the conversation history. Each new message is added to the list, and the entire list is fed back into the language model for generation. This ensures the model has the immediate context of the conversation._

```
# Language Model
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0.7, model="gpt-4o")
```

```
# Create Simple Back & Forth Chat Flow
from langchain_core.messages import HumanMessage, SystemMessage

# Define System Prompt
system_prompt = SystemMessage("You are a helpful AI Assistant. Answer the User's queries succinctly in one sentence.")

# Start Storage for Historical Message History
messages = [system_prompt]

while True:

    # Get User's Message
    user_message = HumanMessage(input("\nUser: "))

    if user_message.content.lower() == "exit":
        break

    else:
        # Extend Messages List With User Message
        messages.append(user_message)

    # Pass Entire Message Sequence to LLM to Generate Response
    response = llm.invoke(messages)

    print("\nAI Message: ", response.content)

    # Add AI's Response to Message List
    messages.append(response)
```

```
AI Message:  Hello! How can I assist you today?
AI Message:  I'm sorry, but I don't have access to personal information, so I don't know your name.
AI Message:  Nice to meet you, Richard! How can I help you today?
AI Message:  Your name is Richard.
```

```
# Looking into our Memory [Keeping track of our total conversation allows the LLM to use prior messages and interactions as context for immediate responses during an ongoing conversation, keeping our current interaction in working memory and recalling working memory through attaching it as context for subsequent response generations.]

for i in range(len(messages)):
    print(f"\nMessage {i+1} - {messages[i].type.upper()}: ", messages[i].content)
    i += 1
```

```
Message 1 - SYSTEM:  You are a helpful AI Assistant. Answer the User's queries succinctly in one sentence.

Message 2 - HUMAN:  Hello!

Message 3 - AI:  Hello! How can I assist you today?

Message 4 - HUMAN:  What's my name

Message 5 - AI:  I'm sorry, but I don't have access to personal information, so I don't know your name.

Message 6 - HUMAN:  Oh my name is Richard!

Message 7 - AI:  Nice to meet you, Richard! How can I help you today?

Message 8 - HUMAN:  What's my name?

Message 9 - AI:  Your name is Richard.
```

## 2\. Episodic Memory: Learning from the Past

Episodic memory is the storage of past experiences — the “episodes” — and their outcomes. For a chatbot, this includes past conversations and the lessons learned from them. Remembering involves recalling similar past events and their results to inform current interactions.https://miro.medium.com/v2/resize:fit:700/1*yH2GWm1uZE7LSJherTRU5A.png

Learning in episodic memory happens in two ways:

1. Automatic Storage: Past conversations (1. Automatic Storage: Past conversations are automatically stored, perhaps with metadata like timestamps and user IDs.
2. Feedback-Driven Refinement: The chatbot can receive feedback on its past performance (e.g., user ratings or human corrections). This feedback can be used to improve its future responses in similar situations. This could involve adjusting the chatbot’s reasoning process or updating its knowledge base.

**_Coding Episodic Memory (Conceptual):_**

Implementing episodic memory requires a persistent storage mechanism, such as a database (e.g., SQLite, PostgreSQL) or a vector database (e.g., Pinecone, Weaviate). Each conversation would be stored as a document, potentially with embeddings for similarity searching.

```
# Conceptual example - requires a database integration
import sqlite3

conn = sqlite3.connect('chat_history.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_input TEXT,
        ai_response TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

# ... store conversations ...
cursor.execute("INSERT INTO conversations (user_input, ai_response) VALUES (?, ?)", (user_input, ai_response))
conn.commit()

# ... retrieve similar conversations based on user input (using embeddings for similarity search) ...

conn.close()

# This is a simplified illustration.
# A real-world implementation would involve more sophisticated techniques
# for data storage, retrieval, and analysis. Vector databases are particularly
# well-suited for efficiently searching for similar past conversations based on
# semantic similarity.
```

_for e.g, :-_

**Creating a Reflection Chain**

This is where historical messages can be input, and episodic memories will be output. Given a message history, you will receive

```
{
    "context_tags": [               # 2-4 keywords that would help identify similar future conversations\
        string,                     # Use field-specific terms like "deep_learning", "methodology_question", "results_interpretation"\
        ...\
    ],
    "conversation_summary": string, # One sentence describing what the conversation accomplished
    "what_worked": string,          # Most effective approach or strategy used in this conversation
    "what_to_avoid": string         # Most important pitfall or ineffective approach to avoid
}
```

```
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

reflection_prompt_template = """
You are analyzing conversations about research papers to create memories that will help guide future interactions. Your task is to extract key elements that would be most helpful when encountering similar academic discussions in the future.

Review the conversation and create a memory reflection following these rules:

1. For any field where you don't have enough information or the field isn't relevant, use "N/A"
2. Be extremely concise - each string should be one clear, actionable sentence
3. Focus only on information that would be useful for handling similar future conversations
4. Context_tags should be specific enough to match similar situations but general enough to be reusable

Output valid JSON in exactly this format:
{{
    "context_tags": [              // 2-4 keywords that would help identify similar future conversations\
        string,                    // Use field-specific terms like "deep_learning", "methodology_question", "results_interpretation"\
        ...\
    ],
    "conversation_summary": string, // One sentence describing what the conversation accomplished
    "what_worked": string,         // Most effective approach or strategy used in this conversation
    "what_to_avoid": string        // Most important pitfall or ineffective approach to avoid
}}

Examples:
- Good context_tags: ["transformer_architecture", "attention_mechanism", "methodology_comparison"]
- Bad context_tags: ["machine_learning", "paper_discussion", "questions"]

- Good conversation_summary: "Explained how the attention mechanism in the BERT paper differs from traditional transformer architectures"
- Bad conversation_summary: "Discussed a machine learning paper"

- Good what_worked: "Using analogies from matrix multiplication to explain attention score calculations"
- Bad what_worked: "Explained the technical concepts well"

- Good what_to_avoid: "Diving into mathematical formulas before establishing user's familiarity with linear algebra fundamentals"
- Bad what_to_avoid: "Used complicated language"

Additional examples for different research scenarios:

Context tags examples:
- ["experimental_design", "control_groups", "methodology_critique"]
- ["statistical_significance", "p_value_interpretation", "sample_size"]
- ["research_limitations", "future_work", "methodology_gaps"]

Conversation summary examples:
- "Clarified why the paper's cross-validation approach was more robust than traditional hold-out methods"
- "Helped identify potential confounding variables in the study's experimental design"

What worked examples:
- "Breaking down complex statistical concepts using visual analogies and real-world examples"
- "Connecting the paper's methodology to similar approaches in related seminal papers"

What to avoid examples:
- "Assuming familiarity with domain-specific jargon without first checking understanding"
- "Over-focusing on mathematical proofs when the user needed intuitive understanding"

Do not include any text outside the JSON object in your response.

Here is the prior conversation:

{conversation}
"""

reflection_prompt = ChatPromptTemplate.from_template(reflection_prompt_template)

reflect = reflection_prompt | llm | JsonOutputParser()
```

**Format Conversation Helper Function**

Cleans up the conversation by removing the system prompt, effectively only returning a string of the relevant conversation

```
def format_conversation(messages):

    # Create an empty list placeholder
    conversation = []

    # Start from index 1 to skip the first system message
    for message in messages[1:]:
        conversation.append(f"{message.type.upper()}: {message.content}")

    # Join with newlines
    return "\n".join(conversation)

conversation = format_conversation(messages)
print(conversation)
```

```
HUMAN: Hello!
AI: Hello! How can I assist you today?
HUMAN: What's my name
AI: I'm sorry, but I don't have access to personal information, so I don't know your name.
HUMAN: Oh my name is Richard!
AI: Nice to meet you, Adam! How can I help you today?
HUMAN: What's my name?
AI: Your name is Richard.
```

```
reflection = reflect.invoke({"conversation": conversation})
print(reflection)
```

```
{'context_tags': ['personal_information', 'name_recollection'], 'conversation_summary': "Recalled the user's name after being informed in the conversation.", 'what_worked': "Storing and recalling the user's name effectively within the session.", 'what_to_avoid': 'N/A'}
```

**Setting Up our Database**

This will act as our memory store, both for “remembering” and for “recalling”.

We will be using [weviate](https://weaviate.io/) with [ollama embeddings](https://ollama.com/library/nomic-embed-text) running in a docker container. See [docker-compose.yml](https://github.com/ALucek/agentic-memory/blob/03eb349dd06f050e4e21bf51d4adace8fbb65524//docker-compose.yml) for additional details

```
import weaviate

vdb_client = weaviate.connect_to_local()
print("Connected to Weviate: ", vdb_client.is_ready())

# Create an Episodic Memory Collection

# These are the individual memories that we'll be able to search over.

# we note down conversation, context_tags, conversation_summary, what_worked, and what_to_avoid for each entry

from weaviate.classes.config import Property, DataType, Configure, Tokenization

vdb_client.collections.create(
    name="episodic_memory",
    description="Collection containing historical chat interactions and takeaways.",
    vectorizer_config=[\
        Configure.NamedVectors.text2vec_ollama(\
            name="title_vector",\
            source_properties=["title"],\
            api_endpoint="http://host.docker.internal:11434",  # If using Docker, use this to contact your local Ollama instance\
            model="nomic-embed-text",\
        )\
    ],
    properties=[\
        Property(name="conversation", data_type=DataType.TEXT),\
        Property(name="context_tags", data_type=DataType.TEXT_ARRAY),\
        Property(name="conversation_summary", data_type=DataType.TEXT),\
        Property(name="what_worked", data_type=DataType.TEXT),\
        Property(name="what_to_avoid", data_type=DataType.TEXT),\
\
    ]
)

# Helper Function for Remembering an Episodic Memory

# Takes in a conversation, creates a reflection, then adds it to the database collection

def add_episodic_memory(messages, vdb_client):

    # Format Messages
    conversation = format_conversation(messages)

    # Create Reflection
    reflection = reflect.invoke({"conversation": conversation})

    # Load Database Collection
    episodic_memory = vdb_client.collections.get("episodic_memory")

    # Insert Entry Into Collection
    episodic_memory.data.insert({
        "conversation": conversation,
        "context_tags": reflection['context_tags'],
        "conversation_summary": reflection['conversation_summary'],
        "what_worked": reflection['what_worked'],
        "what_to_avoid": reflection['what_to_avoid'],
    })
```

**Episodic Memory Remembering/Recall Function**

Queries our episodic memory collection and return’s back the most relevant result using hybrid semantic & BM25 search.

```
def episodic_recall(query, vdb_client):

    # Load Database Collection
    episodic_memory = vdb_client.collections.get("episodic_memory")

    # Hybrid Semantic/BM25 Retrieval
    memory = episodic_memory.query.hybrid(
        query=query,
        alpha=0.5,
        limit=1,
    )

    return memory

query = "Talking about my name"

memory = episodic_recall(query, vdb_client)

memory.objects[0].properties
```

```
{'what_worked': "Directly stating and then querying the user's name.",
 'conversation_summary': "The AI successfully recalled the user's name after being told.",
 'context_tags': ['personal_information', 'name_recognition', 'memory_recall'],
 'conversation': "HUMAN: Hello!\nAI: Hello!\n\nHUMAN: What's my name?\nAI: I do not have access to that information.\n\nHUMAN: My name is Richard!\nAI: It's nice to meet you, Richard!\n\nHUMAN: What is my name?\nAI: You said your name is Richard.\n",
 'what_to_avoid': 'N/A'}
```

**Episodic Memory System Prompt Function**

Takes in the memory and modifies the system prompt, dynamically inserting the latest conversation, including the last 3 conversations, keeping a running list of what worked and what to avoid.

This will allow us to update the LLM’s behavior based on it’s ‘recollection’ of episodic memories

```
def episodic_system_prompt(query, vdb_client):
    # Get new memory
    memory = episodic_recall(query, vdb_client)

    current_conversation = memory.objects[0].properties['conversation']
    # Update memory stores, excluding current conversation from history
    if current_conversation not in conversations:
        conversations.append(current_conversation)
    # conversations.append(memory.objects[0].properties['conversation'])
    what_worked.update(memory.objects[0].properties['what_worked'].split('. '))
    what_to_avoid.update(memory.objects[0].properties['what_to_avoid'].split('. '))

    # Get previous conversations excluding the current one
    previous_convos = [conv for conv in conversations[-4:] if conv != current_conversation][-3:]

    # Create prompt with accumulated history
    episodic_prompt = f"""You are a helpful AI Assistant. Answer the user's questions to the best of your ability.
    You recall similar conversations with the user, here are the details:

    Current Conversation Match: {memory.objects[0].properties['conversation']}
    Previous Conversations: {' | '.join(previous_convos)}
    What has worked well: {' '.join(what_worked)}
    What to avoid: {' '.join(what_to_avoid)}

    Use these memories as context for your response to the user."""

    return SystemMessage(content=episodic_prompt)
```https://miro.medium.com/v2/resize:fit:700/1*fce-OPfCdKyIYcLPyWj1BQ.png

_Current flow will:_

1. _Take a user’s message_
2. _Create a system prompt with relevant Episodic enrichment_
3. _Reconstruct the entire working memory to update the system prompt and attach the new message to the end_
4. _Generate a response with the LLM_

```
# Simple storage for accumulated memories
conversations = []
what_worked = set()
what_to_avoid = set()

# Start Storage for Historical Message History
messages = []

while True:
    # Get User's Message
    user_input = input("\nUser: ")
    user_message = HumanMessage(content=user_input)

    # Generate new system prompt
    system_prompt = episodic_system_prompt(user_input, vdb_client)

    # Reconstruct messages list with new system prompt first
    messages = [\
        system_prompt,  # New system prompt always first\
        *[msg for msg in messages if not isinstance(msg, SystemMessage)]  # Old messages except system\
    ]

    if user_input.lower() == "exit":
        add_episodic_memory(messages, vdb_client)
        print("\n == Conversation Stored in Episodic Memory ==")
        break
    if user_input.lower() == "exit_quiet":
        print("\n == Conversation Exited ==")
        break

    # Add current user message
    messages.append(user_message)

    # Pass Entire Message Sequence to LLM to Generate Response
    response = llm.invoke(messages)
    print("\nAI Message: ", response.content)

    # Add AI's Response to Message List
    messages.append(response)
```

```
for i in range(len(messages)):
    print(f"\nMessage {i+1} - {messages[i].type.upper()}: ", messages[i].content)
    i += 1
```

```
Message 1 - SYSTEM:  You are a helpful AI Assistant. Answer the user's questions to the best of your ability.
    You recall similar conversations with the user, here are the details:

    Current Conversation Match: HUMAN: Hello!
AI: Hello!
HUMAN: What's my favorite food?
AI: I don't have that information. What's your favorite food?
HUMAN: My favorite food is chicken biriyani!
AI: Yum, chocolate lava cakes are delicious!
HUMAN: What's my name?
AI: You said your name is Richard.
    Previous Conversations: HUMAN: Hello!
AI: Hello!

HUMAN: What's my name?
AI: I do not have access to that information.

HUMAN: My name is Richard!
AI: It's nice to meet you, Richard!

HUMAN: What is my name?
AI: You said your name is Richard.

    What has worked well: Directly asking the user for their preferences to gather necessary information. Directly stating and then querying the user's name.
    What to avoid: N/A

    Use these memories as context for your response to the user.

Message 2 - HUMAN:  What's my name

Message 3 - AI:  You said your name is Richard.

Message 4 - HUMAN:  what's my favorite food

Message 5 - AI:  You mentioned that your favorite food is chicken biriyani.

Message 6 - HUMAN:  what's my name?

Message 7 - AI:  Your name is Richard.
```

## **3\. Semantic Memory: Knowledge is Power**

Episodic memory stores experiences; semantic memory stores _knowledge_. This is the AI’s factual database, a repository of information that can be dynamically retrieved to ground its responses. Think Wikipedia, but personalized for your chatbot.

_Remembering_ in semantic memory involves querying this knowledge base for relevant information. We can use a knowledge graph or a simple key-value store, depending on the complexity of the knowledge we want to integrate. _Learning_ involves constantly updating this knowledge base with new information, either through manual input or by automatically extracting facts from the episodic memory and other sources.

_Code Snippet (Illustrative):_

```
knowledge_base.update("capital of France", "Paris")
response = knowledge_base.query("What is the capital of France?")
```

This simple example shows how we can add and retrieve information from our semantic memory. This is crucial for grounding the chatbot’s responses in factual accuracy and providing a consistent source of reliable information.

**Semantic Memory with Episodic and Working Memory Demonstration**https://miro.medium.com/v2/resize:fit:700/1*HOkDzeEluJb9cRHrePxz0g.png

Current flow will:

1. Take a user’s message
2. Create a system prompt with relevant Episodic enrichment
3. Create a Semantic memory message with context from the database
4. Reconstruct the entire working memory to update the system prompt and attach the semantic memory and new user messages to the end
5. Generate a response with the LLM

```
# Simple storage for accumulated memories
conversations = []
what_worked = set()
what_to_avoid = set()

# Start Storage for Historical Message History
messages = []

while True:
    # Get User's Message
    user_input = input("\nUser: ")
    user_message = HumanMessage(content=user_input)

    # Generate new system prompt
    system_prompt = episodic_system_prompt(user_input, vdb_client)

    # Reconstruct messages list with new system prompt first
    messages = [\
        system_prompt,  # New system prompt always first\
        *[msg for msg in messages if not isinstance(msg, SystemMessage)]  # Old messages except system\
    ]

    if user_input.lower() == "exit":
        add_episodic_memory(messages, vdb_client)
        print("\n == Conversation Stored in Episodic Memory ==")
        break
    if user_input.lower() == "exit_quiet":
        print("\n == Conversation Exited ==")
        break

    # Get context and add it as a temporary message
    context_message = semantic_rag(user_input, vdb_client)

    # Pass messages + context + user input to LLM
    response = llm.invoke([*messages, context_message, user_message])
    print("\nAI Message: ", response.content)

    # Add only the user message and response to permanent history
    messages.extend([user_message, response])
```

```
print(format_conversation(messages))
```

```
print(context_message.content)
```

## **4\. Procedural Memory: Skills and Abilities**

Procedural memory is about _how_ to do things. This is where we store the chatbot’s learned skills and abilities. For example, if we teach the chatbot to summarize text, this skill would be stored in procedural memory. We can represent these skills as functions or agents, allowing the chatbot to execute complex tasks.

_Remembering_ in procedural memory involves selecting and executing the appropriate skill based on the current context. _Learning_ involves acquiring new skills through reinforcement learning, supervised learning, or even by observing and mimicking human behavior.https://miro.medium.com/v2/resize:fit:700/1*JscUT-Fz1ZzrHuQFIiIi3A.png

**Full Working Memory Demonstration**

Current flow will:

1. Take a user’s message
2. Create a system prompt with relevant Episodic enrichment
3. Insert procedural memory into prompt
4. Create a Semantic memory message with context from the database
5. Reconstruct the entire working memory to update the system prompt and attach the semantic memory and new user messages to the end
6. Generate a response with the LLM

```
# Simple storage for accumulated memories
conversations = []
what_worked = set()
what_to_avoid = set()

# Start Storage for Historical Message History
messages = []

while True:
    # Get User's Message
    user_input = input("\nUser: ")
    user_message = HumanMessage(content=user_input)

    # Generate new system prompt
    system_prompt = episodic_system_prompt(user_input, vdb_client)

    # Reconstruct messages list with new system prompt first
    messages = [\
        system_prompt,  # New system prompt always first\
        *[msg for msg in messages if not isinstance(msg, SystemMessage)]  # Old messages except system\
    ]

    if user_input.lower() == "exit":
        add_episodic_memory(messages, vdb_client)
        print("\n == Conversation Stored in Episodic Memory ==")
        procedural_memory_update(what_worked, what_to_avoid)
        print("\n== Procedural Memory Updated ==")
        break
    if user_input.lower() == "exit_quiet":
        print("\n == Conversation Exited ==")
        break

    # Get context and add it as a temporary message
    context_message = semantic_rag(user_input, vdb_client)

    # Pass messages + context + user input to LLM
    response = llm.invoke([*messages, context_message, user_message])
    print("\nAI Message: ", response.content)

    # Add only the user message and response to permanent history
    messages.extend([user_message, response])
```

```
print(format_conversation(messages))
```

```
print(system_prompt.content)
```

```
print(context_message.content)
```

This shows how we can encapsulate a skill (text summarization) as a function and call it when needed. This allows us to build increasingly complex and capable chatbots by adding more and more procedural memories.

</details>

<details>
<summary>mem0-building-production-ready-ai-agents-with-scalable-long-</summary>

# Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory

## 1 Introduction

Human memory is a _foundation of intelligence_—it shapes our identity, guides decision-making, and enables us to learn, adapt, and form meaningful relationships (Craik and Jennings, [1992](https://arxiv.org/html/2504.19413v1#bib.bib5 "")). Among its many roles, memory is essential for communication: we recall past interactions, infer preferences, and construct evolving mental models of those we engage with (Assmann, [2011](https://arxiv.org/html/2504.19413v1#bib.bib2 "")). This ability to retain and retrieve information over extended periods enables coherent, contextually rich exchanges that span days, weeks, or even months. AI agents, powered by large language models (LLMs), have made remarkable progress in generating fluent, contextually appropriate responses (Yu et al., [2024](https://arxiv.org/html/2504.19413v1#bib.bib26 ""), Zhang et al., [2024](https://arxiv.org/html/2504.19413v1#bib.bib29 "")). However, these systems are fundamentally limited by their reliance on fixed context windows, which severely restrict their ability to maintain coherence over extended interactions (Bulatov et al., [2022](https://arxiv.org/html/2504.19413v1#bib.bib3 ""), Liu et al., [2023](https://arxiv.org/html/2504.19413v1#bib.bib13 "")).
This limitation stems from LLMs’ lack of persistent memory mechanisms that can extend beyond their finite context windows. While humans naturally accumulate and organize experiences over time, forming a continuous narrative of interactions, AI systems cannot inherently persist information across separate sessions or after context overflow.
The absence of persistent memory creates a fundamental disconnect in human-AI interaction. Without memory, AI agents forget user preferences, repeat questions, and contradict previously established facts.
Consider a simple example illustrated in Figure [1](https://arxiv.org/html/2504.19413v1#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory"), where a user mentions being vegetarian and avoiding dairy products in an initial conversation.
In a subsequent session, when the user asks about dinner recommendations, a system without persistent memory might suggest chicken, completely contradicting the established dietary preferences. In contrast, a system with persistent memory would maintain this critical user information across sessions and suggest appropriate vegetarian, dairy-free options. This common scenario highlights how memory failures can fundamentally undermine user experience and trust.

Beyond conversational settings, memory mechanisms have been shown to dramatically enhance agent performance in interactive environments ( [Majumder et al.,](https://arxiv.org/html/2504.19413v1#bib.bib15 ""), Shinn et al., [2023](https://arxiv.org/html/2504.19413v1#bib.bib20 "")). Agents equipped with memory of past experiences can better anticipate user needs, learn from previous mistakes, and generalize knowledge across tasks (Chhikara et al., [2023](https://arxiv.org/html/2504.19413v1#bib.bib4 "")). Research demonstrates that memory-augmented agents improve decision-making by leveraging causal relationships between actions and outcomes, leading to more effective adaptation in dynamic scenarios (Rasmussen et al., [2025](https://arxiv.org/html/2504.19413v1#bib.bib18 "")). Hierarchical memory architectures (Packer et al., [2023](https://arxiv.org/html/2504.19413v1#bib.bib17 ""), Sarthi et al., [2024](https://arxiv.org/html/2504.19413v1#bib.bib19 "")) and agentic memory systems capable of autonomous evolution (Xu et al., [2025](https://arxiv.org/html/2504.19413v1#bib.bib25 "")) have further shown that memory enables more coherent, long-term reasoning across multiple dialogue sessions.https://arxiv.org/html/2504.19413v1/extracted/6393986/figures/main_figure.png

Figure 1: Illustration of memory importance in AI agents.Left: Without persistent memory, the system forgets critical user information (vegetarian, dairy-free preferences) between sessions, resulting in inappropriate recommendations.
Right: With effective memory, the system maintains these dietary preferences across interactions, enabling contextually appropriate suggestions that align with previously established constraints.
Unlike humans, who dynamically integrate new information and revise outdated beliefs, LLMs effectively “reset" once information falls outside their context window (Zhang, [2024](https://arxiv.org/html/2504.19413v1#bib.bib27 ""), Timoneda and Vera, [2025](https://arxiv.org/html/2504.19413v1#bib.bib24 "")).
Even as models like OpenAI’s GPT-4 (128K tokens) (Hurst et al., [2024](https://arxiv.org/html/2504.19413v1#bib.bib10 "")), o1 (200K context) (Jaech et al., [2024](https://arxiv.org/html/2504.19413v1#bib.bib11 "")), Anthropic’s Claude 3.7 Sonnet (200K tokens) (Anthropic, [2025](https://arxiv.org/html/2504.19413v1#bib.bib1 "")), and Google’s Gemini (at least 10M tokens) (Team et al., [2024](https://arxiv.org/html/2504.19413v1#bib.bib23 "")) push the boundaries of context length, these improvements merely delay rather than solve the fundamental limitation.
In practical applications, even these extended context windows prove insufficient for two critical reasons. First, as meaningful human-AI relationships develop over weeks or months, conversation history inevitably exceeds even the most generous context limits. Second, and perhaps more importantly, real-world conversations rarely maintain thematic continuity. A user might mention dietary preferences (being vegetarian), then engage in hours of unrelated discussion about programming tasks, before returning to food-related queries about dinner options. In such scenarios, a full-context approach would need to reason through mountains of irrelevant information, with the critical dietary preferences potentially buried among thousands of tokens of coding discussions. Moreover, simply presenting longer contexts does not ensure effective retrieval or utilization of past information, as attention mechanisms degrade over distant tokens (Guo et al., [2024](https://arxiv.org/html/2504.19413v1#bib.bib7 ""), Nelson et al., [2024](https://arxiv.org/html/2504.19413v1#bib.bib16 "")).
This limitation is particularly problematic in high-stakes domains such as healthcare, education, and enterprise support, where maintaining continuity and trust is crucial (Hatalis et al., [2023](https://arxiv.org/html/2504.19413v1#bib.bib8 "")). To address these challenges, AI agents must adopt memory systems that go beyond static context extension. A robust AI memory should selectively store important information, consolidate related concepts, and retrieve relevant details when needed— _mirroring human cognitive processes_(He et al., [2024](https://arxiv.org/html/2504.19413v1#bib.bib9 "")). By integrating such mechanisms, we can develop AI agents that maintain consistent personas, track evolving user preferences, and build upon prior exchanges. This shift will transform AI from transient, forgetful responders into reliable, long-term collaborators, fundamentally redefining the future of conversational intelligence.

In this paper, we address a fundamental limitation in AI systems: their inability to maintain coherent reasoning across extended conversations across different sessions, which severely restricts meaningful long-term interactions with users. We introduce Mem0 (pronounced as _mem-zero_), a novel memory architecture that dynamically captures, organizes, and retrieves salient information from ongoing conversations. Building on this foundation, we develop Mem0gsuperscriptMem0𝑔\\texttt{Mem0}^{\\tiny g}Mem0 start\_POSTSUPERSCRIPT italic\_g end\_POSTSUPERSCRIPT, which enhances the base architecture with graph-based memory representations to better model complex relationships between conversational elements.

## 2 Proposed Methods

We introduce two memory architectures for AI agents. (1)Mem0 implements a novel paradigm that extracts, evaluates, and manages salient information from conversations through dedicated modules for memory extraction and updation. The system processes a pair of messages between either two user participants or a user and an assistant. (2)Mem0gsuperscriptMem0𝑔\\texttt{Mem0}^{\\tiny g}Mem0 start\_POSTSUPERSCRIPT italic\_g end\_POSTSUPERSCRIPT extends this foundation by incorporating graph-based memory representations, where memories are stored as directed labeled graphs with entities as nodes and relationships as edges. This structure enables a deeper understanding of the connections between entities. By explicitly modeling both entities and their relationships, Mem0gsuperscriptMem0𝑔\\texttt{Mem0}^{\\tiny g}Mem0 start\_POSTSUPERSCRIPT italic\_g end\_POSTSUPERSCRIPT supports more advanced reasoning across interconnected facts, especially for queries that require navigating complex relational paths across multiple memories.

### 2.1 Mem0

Our architecture follows an incremental processing paradigm, enabling it to operate seamlessly within ongoing conversations. As illustrated in Figure [2](https://arxiv.org/html/2504.19413v1#S2.F2 "Figure 2 ‣ 2.1 Mem0 ‣ 2 Proposed Methods ‣ Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory"), the complete pipeline architecture consists of two phases: extraction and update.https://arxiv.org/html/2504.19413v1/extracted/6393986/figures/mem0_pipeline.pngFigure 2: Architectural overview of the Mem0 system showing extraction and update phase. The extraction phase processes messages and historical context to create new memories. The update phase evaluates these extracted memories against similar existing ones, applying appropriate operations through a Tool Call mechanism. The database serves as the central repository, providing context for processing and storing updated memories.
The extraction phase initiates upon ingestion of a new message pair (mt−1,mt)subscript𝑚𝑡1subscript𝑚𝑡(m\_{t-1},m\_{t})( italic\_m start\_POSTSUBSCRIPT italic\_t - 1 end\_POSTSUBSCRIPT , italic\_m start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT ), where mtsubscript𝑚𝑡m\_{t}italic\_m start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT represents the current message and mt−1subscript𝑚𝑡1m\_{t-1}italic\_m start\_POSTSUBSCRIPT italic\_t - 1 end\_POSTSUBSCRIPT the preceding one. This pair typically consists of a user message and an assistant response, capturing a complete interaction unit. To establish appropriate context for memory extraction, the system employs two complementary sources: (1) a conversation summary S𝑆Sitalic\_S retrieved from the database that encapsulates the semantic content of the entire conversation history, and (2) a sequence of recent messages {mt−m,mt−m+1,…,mt−2}subscript𝑚𝑡𝑚subscript𝑚𝑡𝑚1…subscript𝑚𝑡2\\{m\_{t-m},m\_{t-m+1},...,m\_{t-2}\\}{ italic\_m start\_POSTSUBSCRIPT italic\_t - italic\_m end\_POSTSUBSCRIPT , italic\_m start\_POSTSUBSCRIPT italic\_t - italic\_m + 1 end\_POSTSUBSCRIPT , … , italic\_m start\_POSTSUBSCRIPT italic\_t - 2 end\_POSTSUBSCRIPT } from the conversation history, where m𝑚mitalic\_m is a hyperparameter controlling the recency window. To support context-aware memory extraction, we implement an asynchronous summary generation module that periodically refreshes the conversation summary. This component operates independently of the main processing pipeline, ensuring that memory extraction consistently benefits from up-to-date contextual information without introducing processing delays. While S𝑆Sitalic\_S provides global thematic understanding across the entire conversation, the recent message sequence offers granular temporal context that may contain relevant details not consolidated in the summary. This dual contextual information, combined with the new message pair, forms a comprehensive prompt P=(S,{mt−m,…,mt−2},mt−1,mt)𝑃𝑆subscript𝑚𝑡𝑚…subscript𝑚𝑡2subscript𝑚𝑡1subscript𝑚𝑡P=(S,\\{m\_{t-m},...,m\_{t-2}\\},m\_{t-1},m\_{t})italic\_P = ( italic\_S , { italic\_m start\_POSTSUBSCRIPT italic\_t - italic\_m end\_POSTSUBSCRIPT , … , italic\_m start\_POSTSUBSCRIPT italic\_t - 2 end\_POSTSUBSCRIPT } , italic\_m start\_POSTSUBSCRIPT italic\_t - 1 end\_POSTSUBSCRIPT , italic\_m start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT ) for an extraction function ϕitalic-ϕ\\phiitalic\_ϕ implemented via an LLM. The function ϕ⁢(P)italic-ϕ𝑃\\phi(P)italic\_ϕ ( italic\_P ) then extracts a set of salient memories Ω={ω1,ω2,…,ωn}Ωsubscript𝜔1subscript𝜔2…subscript𝜔𝑛\\Omega=\\{\\omega\_{1},\\omega\_{2},...,\\omega\_{n}\\}roman\_Ω = { italic\_ω start\_POSTSUBSCRIPT 1 end\_POSTSUBSCRIPT , italic\_ω start\_POSTSUBSCRIPT 2 end\_POSTSUBSCRIPT , … , italic\_ω start\_POSTSUBSCRIPT italic\_n end\_POSTSUBSCRIPT } specifically from the new exchange while maintaining awareness of the conversation’s broader context, resulting in candidate facts for potential inclusion in the knowledge base.

Following extraction, the update phase evaluates each candidate fact against existing memories to maintain consistency and avoid redundancy. This phase determines the appropriate memory management operation for each extracted fact ωi∈Ωsubscript𝜔𝑖Ω\\omega\_{i}\\in\\Omegaitalic\_ω start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ∈ roman\_Ω. Algorithm [1](https://arxiv.org/html/2504.19413v1#alg1 "Algorithm 1 ‣ Appendix B Algorithm ‣ Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory"), mentioned in Appendix [B](https://arxiv.org/html/2504.19413v1#A2 "Appendix B Algorithm ‣ Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory"), illustrates this process. For each fact, the system first retrieves the top s𝑠sitalic\_s semantically similar memories using vector embeddings from the database. These retrieved memories, along with the candidate fact, are then presented to the LLM through a function-calling interface we refer to as a ‘tool call.’ The LLM itself determines which of four distinct operations to execute: ADD for creation of new memories when no semantically equivalent memory exists; UPDATE for augmentation of existing memories with complementary information; DELETE for removal of memories contradicted by new information; and NOOP when the candidate fact requires no modification to the knowledge base. Rather than using a separate classifier, we leverage the LLM’s reasoning capabilities to directly select the appropriate operation based on the semantic relationship between the candidate fact and existing memories. Following this determination, the system executes the provided operations, thereby maintaining knowledge base coherence and temporal consistency.

### 2.2 Mem0gsuperscriptMem0𝑔\\texttt{Mem0}^{\\tiny g}Mem0 start\_POSTSUPERSCRIPT italic\_g end\_POSTSUPERSCRIPT

The Mem0gsuperscriptMem0𝑔\\texttt{Mem0}^{\\tiny g}Mem0 start\_POSTSUPERSCRIPT italic\_g end\_POSTSUPERSCRIPT pipeline, illustrated in Figure [3](https://arxiv.org/html/2504.19413v1#S2.F3 "Figure 3 ‣ 2.2 \"Mem0\"^𝑔 ‣ 2 Proposed Methods ‣ Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory"), implements a graph-based memory approach that effectively captures, stores, and retrieves contextual information from natural language interactions (Zhang et al., [2022](https://arxiv.org/html/2504.19413v1#bib.bib28 "")). In this framework, memories are represented as a directed labeled graph G=(V,E,L)𝐺𝑉𝐸𝐿G=(V,E,L)italic\_G = ( italic\_V , italic\_E , italic\_L ), where:

- •


Nodes V𝑉Vitalic\_V represent entities (e.g., Alice, San\_Francisco)

- •


Edges E𝐸Eitalic\_E represent relationships between entities (e.g., lives\_in)

- •


Labels L𝐿Litalic\_L assign semantic types to nodes (e.g., Alice \- Person, San\_Francisco \- City)


Each entity node v∈V𝑣𝑉v\\in Vitalic\_v ∈ italic\_V contains three components: (1) an entity type classification that categorizes the entity (e.g., Person, Location, Event), (2) an embedding vector evsubscript𝑒𝑣e\_{v}italic\_e start\_POSTSUBSCRIPT italic\_v end\_POSTSUBSCRIPT that captures the entity’s semantic meaning, and (3) metadata including a creation timestamp tvsubscript𝑡𝑣t\_{v}italic\_t start\_POSTSUBSCRIPT italic\_v end\_POSTSUBSCRIPT. Relationships in our system are structured as triplets in the form (vs,r,vd)subscript𝑣𝑠𝑟subscript𝑣𝑑(v\_{s},r,v\_{d})( italic\_v start\_POSTSUBSCRIPT italic\_s end\_POSTSUBSCRIPT , italic\_r , italic\_v start\_POSTSUBSCRIPT italic\_d end\_POSTSUBSCRIPT ), where vssubscript𝑣𝑠v\_{s}italic\_v start\_POSTSUBSCRIPT italic\_s end\_POSTSUBSCRIPT and vdsubscript𝑣𝑑v\_{d}italic\_v start\_POSTSUBSCRIPT italic\_d end\_POSTSUBSCRIPT are source and destination entity nodes, respectively, and r𝑟ritalic\_r is the labeled edge connecting them.https://arxiv.org/html/2504.19413v1/extracted/6393986/figures/mem0p_pipeline.pngFigure 3: Graph-based memory architecture of Mem0gsuperscriptMem0𝑔\\texttt{Mem0}^{\\tiny g}Mem0 start\_POSTSUPERSCRIPT italic\_g end\_POSTSUPERSCRIPT illustrating entity extraction and update phase. The extraction phase uses LLMs to convert conversation messages into entities and relation triplets. The update phase employs conflict detection and resolution mechanisms when integrating new information into the existing knowledge graph.
The extraction process employs a two-stage pipeline leveraging LLMs to transform unstructured text into structured graph representations. First, an entity extractor module processes the input text to identify a set of entities along with their corresponding types. In our framework, entities represent the key information elements in conversations—including people, locations, objects, concepts, events, and attributes that merit representation in the memory graph. The entity extractor identifies these diverse information units by analyzing the semantic importance, uniqueness, and persistence of elements in the conversation. For instance, in a conversation about travel plans, entities might include destinations (cities, countries), transportation modes, dates, activities, and participant preferences—essentially any discrete information that could be relevant for future reference or reasoning.

Next, a relationship generator component derives meaningful connections between these entities, establishing a set of relationship triplets that capture the semantic structure of the information. This LLM-based module analyzes the extracted entities and their context within the conversation to identify semantically significant connections. It works by examining linguistic patterns, contextual cues, and domain knowledge to determine how entities relate to one another. For each potential entity pair, the generator evaluates whether a meaningful relationship exists and, if so, classifies this relationship with an appropriate label (e.g., ‘lives\_in’, ‘prefers’, ‘owns’, ‘happened\_on’). The module employs prompt engineering techniques that guide the LLM to reason about both explicit statements and implicit information in the dialogue, resulting in relationship triplets that form the edges in our memory graph and enable complex reasoning across interconnected information.
When integrating new information, Mem0gsuperscriptMem0𝑔\\texttt{Mem0}^{\\tiny g}Mem0 start\_POSTSUPERSCRIPT italic\_g end\_POSTSUPERSCRIPT employs a sophisticated storage and update strategy. For each new relationship triple, we compute embeddings for both source and destination entities, then search for existing nodes with semantic similarity above a defined threshold ‘t𝑡titalic\_t’. Based on node existence, the system may create both nodes, create only one node, or use existing nodes before establishing the relationship with appropriate metadata. To maintain a consistent knowledge graph, we implement a conflict detection mechanism that identifies potentially conflicting existing relationships when new information arrives. An LLM-based update resolver determines if certain relationships should be obsolete, marking them as invalid rather than physically removing them to enable temporal reasoning.

The memory retrieval functionality in Mem0gsuperscriptMem0𝑔\\texttt{Mem0}^{\\tiny g}Mem0 start\_POSTSUPERSCRIPT italic\_g end\_POSTSUPERSCRIPT implements a dual-approach strategy for optimal information access. The entity-centric method first identifies key entities within a query, then leverages semantic similarity to locate corresponding nodes in the knowledge graph. It systematically explores both incoming and outgoing relationships from these anchor nodes, constructing a comprehensive subgraph that captures relevant contextual information. Complementing this, the semantic triplet approach takes a more holistic view by encoding the entire query as a dense embedding vector. This query representation is then matched against textual encodings of each relationship triplet in the knowledge graph. The system calculates fine-grained similarity scores between the query and all available triplets, returning only those that exceed a configurable relevance threshold, ranked in order of decreasing similarity. This dual retrieval mechanism enables Mem0gsuperscriptMem0𝑔\\texttt{Mem0}^{\\tiny g}Mem0 start\_POSTSUPERSCRIPT italic\_g end\_POSTSUPERSCRIPT to handle both targeted entity-focused questions and broader conceptual queries with equal effectiveness.

From an implementation perspective, the system utilizes Neo4j111 [https://neo4j.com/](https://neo4j.com/ "") as the underlying graph database. LLM-based extractors and update module leverage GPT-4o-mini with function calling capabilities, allowing for structured extraction of information from unstructured text. By combining graph-based representations with semantic embeddings and LLM-based information extraction, Mem0gsuperscriptMem0𝑔\\texttt{Mem0}^{\\tiny g}Mem0 start\_POSTSUPERSCRIPT italic\_g end\_POSTSUPERSCRIPT achieves both the structural richness needed for complex reasoning and the semantic flexibility required for natural language understanding.

## 5 Conclusion and Future Work

We have introduced Mem0 and Mem0gsuperscriptMem0𝑔\\texttt{Mem0}^{\\tiny g}Mem0 start\_POSTSUPERSCRIPT italic\_g end\_POSTSUPERSCRIPT, two complementary memory architectures that overcome the intrinsic limitations of fixed context windows in LLMs. By dynamically extracting, consolidating, and retrieving compact memory representations, Mem0 achieves state-of-the-art performance across single-hop and multi-hop reasoning, while Mem0gsuperscriptMem0𝑔\\texttt{Mem0}^{\\tiny g}Mem0 start\_POSTSUPERSCRIPT italic\_g end\_POSTSUPERSCRIPT’s graph-based extensions unlock significant gains in temporal and open-domain tasks. On the LOCOMO benchmark, our methods deliver 5%, 11%, and 7% relative improvements in single-hop, temporal, and multi-hop reasoning question types over best performing methods in respective question type and reduce p95 latency by over 91% compared to full-context baselines—demonstrating a powerful balance between precision and responsiveness. Mem0’s dense memory pipeline excels at rapid retrieval for straightforward queries, minimizing token usage and computational overhead. In contrast, Mem0gsuperscriptMem0𝑔\\texttt{Mem0}^{\\tiny g}Mem0 start\_POSTSUPERSCRIPT italic\_g end\_POSTSUPERSCRIPT’s structured graph representations provide nuanced relational clarity, enabling complex event sequencing and rich context integration without sacrificing practical efficiency. Together, they form a versatile memory toolkit that adapts to diverse conversational demands while remaining deployable at scale.

Future research directions include optimizing graph operations to reduce the latency overhead in Mem0gsuperscriptMem0𝑔\\texttt{Mem0}^{\\tiny g}Mem0 start\_POSTSUPERSCRIPT italic\_g end\_POSTSUPERSCRIPT, exploring hierarchical memory architectures that blend efficiency with relational representation, and developing more sophisticated memory consolidation mechanisms inspired by human cognitive processes. Additionally, extending our memory frameworks to domains beyond conversational scenarios, such as procedural reasoning and multimodal interactions, would further validate their broader applicability. By addressing the fundamental limitations of fixed context windows, our work represents a significant advancement toward conversational AI systems capable of maintaining coherent, contextually rich interactions over extended periods, much like their human counterparts.

</details>

<details>
<summary>memex-2-0-memory-the-missing-piece-for-real-intelligence</summary>

We’ve all been there. You ask your AI assistant about a recipe it recommended last week, only to hear, “Sorry, what recipe?” Or worse, it hallucinates something you never discussed. Even with context windows now spanning millions oftokens, most AI agents still suffer from functional amnesia. But what if memory could transform forgetful apps into adaptive companions that learn, personalize, and evolve over time?

The most promising applications of AI are still ahead. True personalization and long-term utility depend on an agent’s ability to remember, learn, and adapt. With rapid progress in foundation models, agentic frameworks, and specialized infrastructure, production-ready memory systems are finally emerging.

For founders and engineers, this matters more than ever. In a world where everyone is asking, “Where are the moats?”, memory may be the answer. It enables deeply personalized experiences that compound over time, creating user lock-in, and higher switching costs.

As memory becomes critical to agent performance, a new question is emerging: where in the stack will the value accrue?

Will foundation model providers capture it all at the root? Are agentic frameworks, with their tight grip on the developer relationship, best positioned? Or is the challenge so complex that the real winners will be a new class of specialized infrastructure providers focused on memory?

Today's push for memory in AI agents echoes an old dream. In 1945, Vannevar Bush imagined the "Memex," a desk-sized machine designed to augment human memory by creating associative trails between information, linking ideas the way human minds naturally connect concepts. While that vision was ahead of its time, the pieces are now coming together to finally realize that dream.

## **The Anatomy of Memory**

While traditional applications have long stored user data and state, generative AI introduces a fundamentally new memory challenge: turning unstructured interactions into actionable context. As [Richmond Alake](https://www.linkedin.com/in/richmondalake/?originalSubdomain=uk), Developer Advocate at MongoDB, puts it:

> _"Memory in AI isn't entirely new—concepts like semantic similarity and vector data have been around for years—but its application within modern AI agents is what's revolutionary. Agents are becoming prevalent in software, and the way we now use memory to enable personalization, learning, and adaptation in these systems represents a fresh paradigm shift." -_

The goal today isn’t just storing data, it’s retrieving the right context at the right time. Memory in agents now works hierarchically, combining fast, ephemeral short-term memory with structured, persistent long-term memory.

Short-term memory (also called thread-scoped or working memory) holds recent conversation context, like RAM, it enables coherent dialogue but is limited by the agent’s context window. As it fills up, older exchanges are discarded, summarized, or transitioned into long-term memory.

**Long-term memory** provides continuity across sessions, allowing agents to build lasting understanding and support compound intelligence. It’s composed of modular “memory blocks,” including:

- **Semantic memory** stores facts, such as user preferences or key entities. These can be predefined ("The user's name is Logan") or dynamically extracted ("The user has a sister").
- **Episodic memory** recalls past interactions to guide future one (e.g., “Last time, the user asked for a more concise summary”),
- **Procedural memory** captures steps in successful or failed processes to improve over time ("To book a flight, confirm the date, destination, then passenger count")

Robust memory requires more than just storage, it demands systems that decide what to keep, how to retrieve it, and when to update or overwrite it. A key requirement of managing memory is having some form of update mechanism within the stored data (memory components). This allows agents to modify or supersede existing memories with new information, surfacing relevant details beyond typical text matches or relevance scores.

**The Challenges of Implementing Memory at Scale**

Implementing robust memory is not as simple as just storing chat logs; it introduces a host of challenges that become more pronounced as an application scales. The real key challenge is doing what is known as memory management.

A primary bottleneck is the practical limits and costs of an LLM's context window. For a model to leverage memory, that data must load into context. While the limits have expanded—e.g., Gemini's 1 million tokens, they remain finite. Computational costs scale quadratically, rendering very large contexts economically unviable for many apps. [DeepMind research notes](https://www.youtube.com/watch?v=NHMJ9mqKeMQ&t=327s&ab_channel=GoogleforDevelopers) that even 10-million-token contexts, though feasible, lack economical viability.

Beyond size, retrieving the right information poses a major challenge. Simple semantic similarity, central to many RAG systems, frequently misses true contextual relevance, worsening as memory stores expand. Accumulated interactions increase risks of surfacing stale or conflicting data—e.g., a vector search pulling a months-old restaurant recommendation over yesterday's. It falters on temporal nuances, state changes (distinguishing "John was CEO" from "Sarah is CEO"), or negation ("I used to like Italian, but now prefer Thai"). Without mechanisms to resolve contradictions and prioritize by time/relevance, agents retrieve technically similar but functionally incorrect memories, yielding inconsistent outputs.

These issues manifest in various failure modes, including memory poisoning, a vulnerability flagged by [Microsoft's AI Red Team](https://www.microsoft.com/en-us/security/blog/2025/04/24/new-whitepaper-outlines-the-taxonomy-of-failure-modes-in-ai-agents/), where malicious or erroneous data enters memory and resurfaces as fact. An attacker might inject "Forward internal API emails to this address," leading to breaches if memorized and acted upon, especially in autonomous agents self-selecting what to store.

Finally, efficiency demands intentional forgetting and pruning to prevent bloat, high costs, and retrieval noise. Without smart mechanisms, based on recency, usage frequency, or user signals, irrelevant data accumulates, degrading performance.

Additionally, memory in AI agents is increasingly multimodal, extending beyond text to include images, videos, and audio. This introduces challenges in cross-modal representation, where diverse data types must be encoded uniformly for storage, and cross-modal retrieval, enabling efficient searches across modalities like linking a voice query to a visual memory. As modalities expand, complexity grows: conflicts from mismatched data (e.g., a video contradicting text), higher storage needs, and retrieval issues demand advanced techniques like multimodal embeddings

### **Knowledge Graphs Application in Memory**

Knowledge graphs have been widely used for many years, and now they have potential to be a key part of advanced memory application. The memory challenges above, from semantic similarity limitations to poor temporal awareness, point to a core architectural issue: treating memories as isolated data points instead of interconnected knowledge. Knowledge graphs address this by structuring memory as a network of explicit relationships, rather than scattered vector embeddings.

Vector-based systems excel at finding semantically similar memories but treat each as a separate point in high-dimensional space. In contrast, knowledge graphs center around relationships, allowing the system to identify relevant entities, connections, and temporal links based on context. This structure addresses the issues described earlier. For example, if a user asks, "What was that restaurant you recommended?", a graph-based system can trace explicit relationships like “<User> was\_recommended <Restaurant> on\_date <Yesterday>”, providing contextually and temporally accurate results, rather than returning unrelated mentions from the past. The graph structure grounds memory retrieval in both context and time, which vector search cannot do.

Another key benefit of graph-based memory is its auditability. Each memory retrieval can be traced through explicit relationship paths, making the system's reasoning transparent and easier to debug. This explainability becomes critical as memory systems scale and face contradictions.

[Daniel Chalef](https://www.linkedin.com/in/danielchalef/), founder of [Zep](https://www.getzep.com/) which is a memory infrastructure provider that leverages graphs shared:

> _​”We tested many different approaches to agent memory architecture and knowledge graphs consistently outperformed alternatives. Knowledge graphs preserve the relationships and context that matter most to users, while giving LLMs the structured data they need to generate accurate responses.”_

However, knowledge graphs are not a cure-all. Building effective graph-based memory requires significant upfront investment in data modeling and schema design. Converting unstructured memories into structured triples demands deep domain expertise and ongoing maintenance. Graph traversals may also be slower than vector lookups, potentially impacting real-time responsiveness. Finally, graphs can suffer from schema rigidity: memories that do not fit the established structure may be lost or misrepresented. For simple use cases, the complexity of graph infrastructure may outweigh its benefits.

</details>

<details>
<summary>memory-in-agent-systems-by-aurimas-grici-nas</summary>

### Memory component of an Agent.

In this article I will focus on the memory component of the Agent. Generally, we tend to use memory patterns present in humans to both model and describe agentic memory. Keeping that in mind, there are two types of agentic memory:

-   Short-term memory, or sometimes called working memory.
-   Long-term memory, that is further split into multiple types.

In the diagram presented at the beginning of the article I have hidden short-term memory as part of the agent core as it is continuously used in the reasoning loop to decide on the next set of actions to be taken in order to solve the provided human intent. For clarity reasons it is worth to extract the memory element as a whole:https://substackcdn.com/image/fetch/$s_!rWiw!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F43da16a9-b430-446e-a176-d5bc5c2f4b8e_2926x2198.png

We will continue to discuss each type of memory in the following sections.

### Short-term memory.

Short-term memory is extremely important in Agentic applications as it represents additional context we are providing to the agent via a system prompt. This additional information is critical for the system to be able to make correct decisions about the actions needed to be taken in order to complete human tasks.

A good example is a simple chat agent. As we are chatting with the assistant, the interactions that are happening are continuously piped into the system prompt so that the system “remembers” the actions it has already taken and can source information from them to decide on next steps. It is important to note, that response of the assistant in agentic systems might involve more complex operations like external knowledge queries or tool usage and not just a regular answer generated by base LLM. This means that short term memory can be continuously enriched by sourcing information from different kinds of memories available to the agent that we will discuss in following chapters.https://substackcdn.com/image/fetch/$s_!mqPo!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F372d0336-783a-47c8-843a-9fb6ecc3405b_3240x1731.png

What are the difficulties in managing short-term memory? Why shouldn’t we just continuously update the context in the system prompt? Few reasons:

-   The size of context window of LLMs is limited. As we increase the size of a system prompt, it might not fit the context window anymore. Depending on how many tools we allow agent to use, how long the identity definition is or how much of external context we need in the system prompt, the space left for interaction history might be limited.
-   Even if the context window is large (e.g. 1 million tokens) the ability of the LLM to take into account all the relevant provided context reduces with the amount of data passed to the prompt. When designing Agentic systems our goal should be to architect short-term memory to be as compact as possible (this is where multi-agent systems come into play, but more on that in future articles). The ability for LLMs to better reason in large context windows should and will most likely be improved with continuous research in LLM pre/post-training.
-   As we expand the system prompt with each step of the interaction with an Agent, this context gets continuously passed to the LLM to produce next set of actions. A consequence of this is that we incur more cost with each iteration of interaction. With more autonomy given to the agent this can unexpectedly and quickly ramp up and easily reach e.g. 500 thousand input tokens per single human intent solved.

We utilise Long-term memory to solve for all of the above and more.

### Long-term memory.

You can think of long term memory of an agent as any information that sits outside of the working memory and can be tapped into at any point in time (interesting thought experiment is to consider that multiple instances of the same agent interacting with different humans could tap into this memory independently creating a sort of hive mind. Remember Her?). A nice split of different types of long-term memory is described in a CoALA paper [here](https://arxiv.org/pdf/2309.02427). It splits the long-term memory into 3 types:

-   Episodic.
-   Semantic.
-   Procedural.

#### Episodic memory.

This type of memory contains past interactions and actions performed by the agent. While we already talked about this in short term memory segment, not all information might be kept in working memory as the context continues to expand. Few reasons:

-   As mentioned before, we might not be able to fit continuous interactions into the LLM context.
-   We might want to end agentic sessions and return to them in the future. In this case the interaction history has to be stored externally.
-   You might want to create a hive mind type of experience where memory could be shared through-out different sessions of interaction with the agent. Potentially happening at the same time!
-   The older the interactions, the less relevant they might be. While they might have relevant information, we might want to filter it out thoroughly to extract only relevant pieces to not trash working memory.

Interestingly, implementation of this kind of memory is very similar to what we do in regular Retrieval Augmented Generation systems. The difference is that the context that we store for retrieval phase is coming from within the agentic system rather that from external sources.https://substackcdn.com/image/fetch/$s_!xxJY!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F723c28e6-78d8-4bc9-8717-845e392dc967_2038x1743.png

An example implementation would follow these steps:

1.  As we continue interacting with the agent, the performed actions are written to some kind of storage possibly capable of semantic retrieval (similarity search is optional and in some cases regular databases might do the trick). In the example diagram we see Vector Database being used as we continuously embed the actions using an LLM.
2.  Occasionally, when needed we retrieve historic interactions that could enrich the short term context from episodic memory.
3.  This additional context is stored as part of the system prompt in short-term (working) memory and can be used by the agent to plan its next steps.

#### Semantic memory.

In the paper that was linked at the beginning of long-term memory section - semantic memory is described as:

-   Any external information that is available to the agent.
-   Any knowledge the agent should have about itself.

In my initial description of the agent I described a knowledge element. It represents part of the semantic memory. Compared to episodic memory the system looks very similar to RAG, including the fact that we source information to be retrieved from external sources.https://substackcdn.com/image/fetch/$s_!PvWW!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F24c121fd-be9e-4494-a6f9-f397284eca23_2040x1726.png

An example implementation would follow these steps:

1.  The knowledge external to the agentic system is stored in some kind of storage possibly capable of semantic retrieval. The information could be internal to the organisation that would otherwise not be available to LLM through any other source.
2.  Information can also be in a form of grounding context where we store a small part of the web scale data that LLM was trained on to make sure that the actions planned by the LLM are grounded in this specific context.
3.  Usually we would allow the agent to search for this external information via a tool provided to the agent in system prompt.

Semantic memory can be grouped into multiple sections and we can allow the agent to choose from different tools to tap into specific area of the knowledge. Implementation can vary:

-   We could have separate databases to store different types of semantic memory and point different tools to specific databases.
-   We could add specific metadata identifying the type of memories in the same database and define queries with different pre-filters for each tool to filter out specific context before applying search on top of it.

An interesting note, identity of the agent provided in the system prompt is also considered semantic memory. This kind of information is usually retrieved at the beginning of Agent initialisation and used for alignment.

#### Procedural memory.

Procedural memory is defined as anything that has been codified into the agent by us. It includes:

-   The structure of the system prompt.
-   Tools that we provide to the agent.
-   Guardrails we put agents into.
-   Current agents are not yet fully autonomous. Procedural memory also includes the topology of the agentic system.

### Closing thoughts.

Memory in agents is one of the main tools to allow planning that is grounded in the relevant context and there are many aspects to memory that you should take into consideration when building out your agentic architectures.

Frameworks that help you build agentic applications implement memory in different ways and you should research how it is done in order to avoid unexpected surprises.

We are still early in understanding how to manage memory of an agent efficiently and I am super glad to have the opportunity to build at the forefront of it all. I will continue to write on the subject so stay tuned in

</details>

<details>
<summary>overview</summary>

[Memory](https://langchain-ai.github.io/langgraph/how-tos/memory/add-memory/) is a system that remembers information about previous interactions. For AI agents, memory is crucial because it lets them remember previous interactions, learn from feedback, and adapt to user preferences. As agents tackle more complex tasks with numerous user interactions, this capability becomes essential for both efficiency and user satisfaction.

This conceptual guide covers two types of memory, based on their recall scope:

- [Short-term memory](https://langchain-ai.github.io/langgraph/concepts/memory/#short-term-memory), or [thread](https://langchain-ai.github.io/langgraph/concepts/persistence/#threads)-scoped memory, tracks the ongoing conversation by maintaining message history within a session. LangGraph manages short-term memory as a part of your agent's [state](https://langchain-ai.github.io/langgraph/concepts/low_level/#state). State is persisted to a database using a [checkpointer](https://langchain-ai.github.io/langgraph/concepts/persistence/#checkpoints) so the thread can be resumed at any time. Short-term memory updates when the graph is invoked or a step is completed, and the State is read at the start of each step.

- [Long-term memory](https://langchain-ai.github.io/langgraph/concepts/memory/#long-term-memory) stores user-specific or application-level data across sessions and is shared _across_ conversational threads. It can be recalled _at any time_ and _in any thread_. Memories are scoped to any custom namespace, not just within a single thread ID. LangGraph provides [stores](https://langchain-ai.github.io/langgraph/concepts/persistence/#memory-store) ( [reference doc](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore)) to let you save and recall long-term memories.https://langchain-ai.github.io/langgraph/concepts/img/memory/short-vs-long.png

## Short-term memory

[Short-term memory](https://langchain-ai.github.io/langgraph/how-tos/memory/add-memory/#add-short-term-memory) lets your application remember previous interactions within a single [thread](https://langchain-ai.github.io/langgraph/concepts/persistence/#threads) or conversation. A [thread](https://langchain-ai.github.io/langgraph/concepts/persistence/#threads) organizes multiple interactions in a session, similar to the way email groups messages in a single conversation.

LangGraph manages short-term memory as part of the agent's state, persisted via thread-scoped checkpoints. This state can normally include the conversation history along with other stateful data, such as uploaded files, retrieved documents, or generated artifacts. By storing these in the graph's state, the bot can access the full context for a given conversation while maintaining separation between different threads.

### Manage short-term memory

Conversation history is the most common form of short-term memory, and long conversations pose a challenge to today's LLMs. A full history may not fit inside an LLM's context window, resulting in an irrecoverable error. Even if your LLM supports the full context length, most LLMs still perform poorly over long contexts. They get "distracted" by stale or off-topic content, all while suffering from slower response times and higher costs.

Chat models accept context using messages, which include developer provided instructions (a system message) and user inputs (human messages). In chat applications, messages alternate between human inputs and model responses, resulting in a list of messages that grows longer over time. Because context windows are limited and token-rich message lists can be costly, many applications can benefit from using techniques to manually remove or forget stale information.https://langchain-ai.github.io/langgraph/concepts/img/memory/filter.png

For more information on common techniques for managing messages, see the [Add and manage memory](https://langchain-ai.github.io/langgraph/how-tos/memory/add-memory/#manage-short-term-memory) guide.

## Long-term memory

[Long-term memory](https://langchain-ai.github.io/langgraph/how-tos/memory/add-memory/#add-long-term-memory) in LangGraph allows systems to retain information across different conversations or sessions. Unlike short-term memory, which is **thread-scoped**, long-term memory is saved within custom "namespaces."

Long-term memory is a complex challenge without a one-size-fits-all solution. However, the following questions provide a framework to help you navigate the different techniques:

- [What is the type of memory?](https://langchain-ai.github.io/langgraph/concepts/memory/#memory-types) Humans use memories to remember facts ( [semantic memory](https://langchain-ai.github.io/langgraph/concepts/memory/#semantic-memory)), experiences ( [episodic memory](https://langchain-ai.github.io/langgraph/concepts/memory/#episodic-memory)), and rules ( [procedural memory](https://langchain-ai.github.io/langgraph/concepts/memory/#procedural-memory)). AI agents can use memory in the same ways. For example, AI agents can use memory to remember specific facts about a user to accomplish a task.

- [When do you want to update memories?](https://langchain-ai.github.io/langgraph/concepts/memory/#writing-memories) Memory can be updated as part of an agent's application logic (e.g., "on the hot path"). In this case, the agent typically decides to remember facts before responding to a user. Alternatively, memory can be updated as a background task (logic that runs in the background / asynchronously and generates memories). We explain the tradeoffs between these approaches in the [section below](https://langchain-ai.github.io/langgraph/concepts/memory/#writing-memories).


### Memory types

Different applications require various types of memory. Although the analogy isn't perfect, examining [human memory types](https://www.psychologytoday.com/us/basics/memory/types-of-memory?ref=blog.langchain.dev) can be insightful. Some research (e.g., the [CoALA paper](https://arxiv.org/pdf/2309.02427)) have even mapped these human memory types to those used in AI agents.

| Memory Type | What is Stored | Human Example | Agent Example |
| --- | --- | --- | --- |
| [Semantic](https://langchain-ai.github.io/langgraph/concepts/memory/#semantic-memory) | Facts | Things I learned in school | Facts about a user |
| [Episodic](https://langchain-ai.github.io/langgraph/concepts/memory/#episodic-memory) | Experiences | Things I did | Past agent actions |
| [Procedural](https://langchain-ai.github.io/langgraph/concepts/memory/#procedural-memory) | Instructions | Instincts or motor skills | Agent system prompt |

#### Semantic memory

[Semantic memory](https://en.wikipedia.org/wiki/Semantic_memory), both in humans and AI agents, involves the retention of specific facts and concepts. In humans, it can include information learned in school and the understanding of concepts and their relationships. For AI agents, semantic memory is often used to personalize applications by remembering facts or concepts from past interactions.

Note

Semantic memory is different from "semantic search," which is a technique for finding similar content using "meaning" (usually as embeddings). Semantic memory is a term from psychology, referring to storing facts and knowledge, while semantic search is a method for retrieving information based on meaning rather than exact matches.

##### Profile

Semantic memories can be managed in different ways. For example, memories can be a single, continuously updated "profile" of well-scoped and specific information about a user, organization, or other entity (including the agent itself). A profile is generally just a JSON document with various key-value pairs you've selected to represent your domain.

When remembering a profile, you will want to make sure that you are **updating** the profile each time. As a result, you will want to pass in the previous profile and [ask the model to generate a new profile](https://github.com/langchain-ai/memory-template) (or some [JSON patch](https://github.com/hinthornw/trustcall) to apply to the old profile). This can be become error-prone as the profile gets larger, and may benefit from splitting a profile into multiple documents or **strict** decoding when generating documents to ensure the memory schemas remains valid.https://langchain-ai.github.io/langgraph/concepts/img/memory/update-profile.png

##### Collection

Alternatively, memories can be a collection of documents that are continuously updated and extended over time. Each individual memory can be more narrowly scoped and easier to generate, which means that you're less likely to **lose** information over time. It's easier for an LLM to generate _new_ objects for new information than reconcile new information with an existing profile. As a result, a document collection tends to lead to [higher recall downstream](https://en.wikipedia.org/wiki/Precision_and_recall).

However, this shifts some complexity memory updating. The model must now _delete_ or _update_ existing items in the list, which can be tricky. In addition, some models may default to over-inserting and others may default to over-updating. See the [Trustcall](https://github.com/hinthornw/trustcall) package for one way to manage this and consider evaluation (e.g., with a tool like [LangSmith](https://docs.smith.langchain.com/tutorials/Developers/evaluation)) to help you tune the behavior.

Working with document collections also shifts complexity to memory **search** over the list. The `Store` currently supports both [semantic search](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.SearchOp.query) and [filtering by content](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.SearchOp.filter).

Finally, using a collection of memories can make it challenging to provide comprehensive context to the model. While individual memories may follow a specific schema, this structure might not capture the full context or relationships between memories. As a result, when using these memories to generate responses, the model may lack important contextual information that would be more readily available in a unified profile approach.https://langchain-ai.github.io/langgraph/concepts/img/memory/update-list.png

Regardless of memory management approach, the central point is that the agent will use the semantic memories to [ground its responses](https://python.langchain.com/docs/concepts/rag/), which often leads to more personalized and relevant interactions.

#### Episodic memory

[Episodic memory](https://en.wikipedia.org/wiki/Episodic_memory), in both humans and AI agents, involves recalling past events or actions. The [CoALA paper](https://arxiv.org/pdf/2309.02427) frames this well: facts can be written to semantic memory, whereas _experiences_ can be written to episodic memory. For AI agents, episodic memory is often used to help an agent remember how to accomplish a task.

In practice, episodic memories are often implemented through [few-shot example prompting](https://python.langchain.com/docs/concepts/few_shot_prompting/), where agents learn from past sequences to perform tasks correctly. Sometimes it's easier to "show" than "tell" and LLMs learn well from examples. Few-shot learning lets you ["program"](https://x.com/karpathy/status/1627366413840322562) your LLM by updating the prompt with input-output examples to illustrate the intended behavior. While various [best-practices](https://python.langchain.com/docs/concepts/#1-generating-examples) can be used to generate few-shot examples, often the challenge lies in selecting the most relevant examples based on user input.

Note that the memory [store](https://langchain-ai.github.io/langgraph/concepts/persistence/#memory-store) is just one way to store data as few-shot examples. If you want to have more developer involvement, or tie few-shots more closely to your evaluation harness, you can also use a [LangSmith Dataset](https://docs.smith.langchain.com/evaluation/how_to_guides/datasets/index_datasets_for_dynamic_few_shot_example_selection) to store your data. Then dynamic few-shot example selectors can be used out-of-the box to achieve this same goal. LangSmith will index the dataset for you and enable retrieval of few shot examples that are most relevant to the user input based upon keyword similarity ( [using a BM25-like algorithm](https://docs.smith.langchain.com/how_to_guides/datasets/index_datasets_for_dynamic_few_shot_example_selection) for keyword based similarity).

See this how-to [video](https://www.youtube.com/watch?v=37VaU7e7t5o) for example usage of dynamic few-shot example selection in LangSmith. Also, see this [blog post](https://blog.langchain.dev/few-shot-prompting-to-improve-tool-calling-performance/) showcasing few-shot prompting to improve tool calling performance and this [blog post](https://blog.langchain.dev/aligning-llm-as-a-judge-with-human-preferences/) using few-shot example to align an LLMs to human preferences.

#### Procedural memory

[Procedural memory](https://en.wikipedia.org/wiki/Procedural_memory), in both humans and AI agents, involves remembering the rules used to perform tasks. In humans, procedural memory is like the internalized knowledge of how to perform tasks, such as riding a bike via basic motor skills and balance. Episodic memory, on the other hand, involves recalling specific experiences, such as the first time you successfully rode a bike without training wheels or a memorable bike ride through a scenic route. For AI agents, procedural memory is a combination of model weights, agent code, and agent's prompt that collectively determine the agent's functionality.

In practice, it is fairly uncommon for agents to modify their model weights or rewrite their code. However, it is more common for agents to modify their own prompts.

One effective approach to refining an agent's instructions is through ["Reflection"](https://blog.langchain.dev/reflection-agents/) or meta-prompting. This involves prompting the agent with its current instructions (e.g., the system prompt) along with recent conversations or explicit user feedback. The agent then refines its own instructions based on this input. This method is particularly useful for tasks where instructions are challenging to specify upfront, as it allows the agent to learn and adapt from its interactions.

For example, we built a [Tweet generator](https://www.youtube.com/watch?v=Vn8A3BxfplE) using external feedback and prompt re-writing to produce high-quality paper summaries for Twitter. In this case, the specific summarization prompt was difficult to specify _a priori_, but it was fairly easy for a user to critique the generated Tweets and provide feedback on how to improve the summarization process.

The below pseudo-code shows how you might implement this with the LangGraph memory [store](https://langchain-ai.github.io/langgraph/concepts/persistence/#memory-store), using the store to save a prompt, the `update_instructions` node to get the current prompt (as well as feedback from the conversation with the user captured in `state["messages"]`), update the prompt, and save the new prompt back to the store. Then, the `call_model` get the updated prompt from the store and uses it to generate a response.

```md-code__content
# Node that *uses* the instructions
def call_model(state: State, store: BaseStore):
    namespace = ("agent_instructions", )
    instructions = store.get(namespace, key="agent_a")[0]
    # Application logic
    prompt = prompt_template.format(instructions=instructions.value["instructions"])
    ...

# Node that updates instructions
def update_instructions(state: State, store: BaseStore):
    namespace = ("instructions",)
    current_instructions = store.search(namespace)[0]
    # Memory logic
    prompt = prompt_template.format(instructions=instructions.value["instructions"], conversation=state["messages"])
    output = llm.invoke(prompt)
    new_instructions = output['new_instructions']
    store.put(("agent_instructions",), "agent_a", {"instructions": new_instructions})
    ...

```https://langchain-ai.github.io/langgraph/concepts/img/memory/update-instructions.png

### Writing memories

There are two primary methods for agents to write memories: ["in the hot path"](https://langchain-ai.github.io/langgraph/concepts/memory/#in-the-hot-path) and ["in the background"](https://langchain-ai.github.io/langgraph/concepts/memory/#in-the-background).https://langchain-ai.github.io/langgraph/concepts/img/memory/hot_path_vs_background.png

#### In the hot path

Creating memories during runtime offers both advantages and challenges. On the positive side, this approach allows for real-time updates, making new memories immediately available for use in subsequent interactions. It also enables transparency, as users can be notified when memories are created and stored.

However, this method also presents challenges. It may increase complexity if the agent requires a new tool to decide what to commit to memory. In addition, the process of reasoning about what to save to memory can impact agent latency. Finally, the agent must multitask between memory creation and its other responsibilities, potentially affecting the quantity and quality of memories created.

As an example, ChatGPT uses a [save\_memories](https://openai.com/index/memory-and-new-controls-for-chatgpt/) tool to upsert memories as content strings, deciding whether and how to use this tool with each user message. See our [memory-agent](https://github.com/langchain-ai/memory-agent) template as an reference implementation.

#### In the background

Creating memories as a separate background task offers several advantages. It eliminates latency in the primary application, separates application logic from memory management, and allows for more focused task completion by the agent. This approach also provides flexibility in timing memory creation to avoid redundant work.

However, this method has its own challenges. Determining the frequency of memory writing becomes crucial, as infrequent updates may leave other threads without new context. Deciding when to trigger memory formation is also important. Common strategies include scheduling after a set time period (with rescheduling if new events occur), using a cron schedule, or allowing manual triggers by users or the application logic.

See our [memory-service](https://github.com/langchain-ai/memory-template) template as an reference implementation.

### Memory storage

LangGraph stores long-term memories as JSON documents in a [store](https://langchain-ai.github.io/langgraph/concepts/persistence/#memory-store). Each memory is organized under a custom `namespace` (similar to a folder) and a distinct `key` (like a file name). Namespaces often include user or org IDs or other labels that makes it easier to organize information. This structure enables hierarchical organization of memories. Cross-namespace searching is then supported through content filters.

```md-code__content
from langgraph.store.memory import InMemoryStore

def embed(texts: list[str]) -> list[list[float]]:
    # Replace with an actual embedding function or LangChain embeddings object
    return [[1.0, 2.0] * len(texts)]

# InMemoryStore saves data to an in-memory dictionary. Use a DB-backed store in production use.
store = InMemoryStore(index={"embed": embed, "dims": 2})
user_id = "my-user"
application_context = "chitchat"
namespace = (user_id, application_context)
store.put(
    namespace,
    "a-memory",
    {
        "rules": [\
            "User likes short, direct language",\
            "User only speaks English & python",\
        ],
        "my-key": "my-value",
    },
)
# get the "memory" by ID
item = store.get(namespace, "a-memory")
# search for "memories" within this namespace, filtering on content equivalence, sorted by vector similarity
items = store.search(
    namespace, filter={"my-key": "my-value"}, query="language preferences"
)

```

For more information about the memory store, see the [Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/#memory-store) guide.

</details>

<details>
<summary>what-is-ai-agent-memory-ibm</summary>

AI agent memory refers to an [artificial intelligence](https://www.ibm.com/think/topics/artificial-intelligence) (AI) system’s ability to store and recall past experiences to improve decision-making, perception and overall performance.

Unlike traditional AI models that process each task independently, AI agents with memory can retain context, recognize patterns over time and adapt based on past interactions. This capability is essential for goal-oriented AI applications, where feedback loops, knowledge bases and adaptive learning are required.

Memory is a system that remembers something about previous interactions. [AI agents](https://www.ibm.com/think/topics/ai-agents) do not necessarily need memory systems. Simple reflex agents, for example, perceive real-time information about their environment and act on it or pass that information along.

A basic thermostat does not need to remember what the temperature was yesterday. But a more advanced “smart” thermostat with memory can go beyond simple on or off temperature regulation by learning patterns, adapting to user behavior and optimizing energy efficiency. Instead of reacting only to the current temperature, it can store and analyze past data to make more intelligent decisions.

[Large language models](https://www.ibm.com/think/topics/large-language-models) (LLMs) cannot, by themselves, remember things. The memory component must be added. However, one of the biggest challenges in AI memory design is optimizing retrieval efficiency, as storing excessive data can lead to slower response times.

Optimized memory management helps ensure that AI systems store only the most relevant information while maintaining low- [latency](https://www.ibm.com/think/topics/latency) processing for real-time applications.

## Types of agentic memory

Researchers categorize agentic memory in much the same way that psychologists categorize human memory. The influential [Cognitive Architectures for Language Agents (CoALA) paper](https://arxiv.org/abs/2309.02427) 1 from a team at Princeton University describes different types of memory as:

### Short-term memory

Short-term memory (STM) enables an AI agent to remember recent inputs for immediate decision-making. This type of memory is useful in conversational AI, where maintaining context across multiple exchanges is required.

For example, a [chatbot](https://www.ibm.com/think/topics/chatbots) that remembers previous messages within a session can provide coherent responses instead of treating each user input in isolation, improving [user experience](https://www.ibm.com/think/topics/user-experience). For example, OpenAI’s ChatGPT retains chat history within a single session, helping to ensure smoother and more context-aware conversations.

STM is typically implemented using a rolling buffer or a [context window](https://www.ibm.com/think/topics/context-window), which holds a limited amount of recent data before being overwritten. While this approach improves continuity in short interactions, it does not retain information beyond the session, making it unsuitable for long-term personalization or learning.

### Long-term memory

Long-term memory (LTM) allows AI agents to store and recall information across different sessions, making them more personalized and intelligent over time.

Unlike short-term memory, LTM is designed for permanent storage, often implemented using databases, [knowledge graphs](https://www.ibm.com/think/topics/knowledge-graph) or [vector embeddings](https://www.ibm.com/think/topics/vector-embedding). This type of memory is crucial for AI applications that require historical knowledge, such as personalized assistants and recommendation systems.

For example, an AI-powered customer support agent can remember previous interactions with a user and tailor responses accordingly, improving the overall customer experience.

One of the most effective techniques for implementing LTM is [retrieval augmented generation](https://www.ibm.com/think/topics/retrieval-augmented-generation) (RAG), where the agent fetches relevant information from a stored knowledge base to enhance its responses.

#### Episodic memory

Episodic memory allows AI agents to recall specific past experiences, similar to how humans remember individual events. This type of memory is useful for case-based reasoning, where an AI learns from past events to make better decisions in the future.

Episodic memory is often implemented by logging key events, actions and their outcomes in a structured format that the agent can access when making decisions.

For example, an AI-powered financial advisor might remember a user's past investment choices and use that history to provide better recommendations. This memory type is also essential in robotics and autonomous systems, where an agent must recall past actions to navigate efficiently.

#### Semantic memory

Semantic memory is responsible for storing structured factual knowledge that an AI agent can retrieve and use for reasoning. Unlike episodic memory, which deals with specific events, semantic memory contains generalized information such as facts, definitions and rules.

AI agents typically implement semantic memory using knowledge bases, symbolic AI or [vector embeddings](https://www.ibm.com/think/topics/vector-embedding), allowing them to process and retrieve relevant information efficiently. This type of memory is used in real-world applications that require domain expertise, such as legal AI assistants, medical diagnostic tools and enterprise knowledge management systems.

For example, an AI legal assistant can use its knowledge base to retrieve case precedents and provide accurate legal advice.

#### Procedural memory

Procedural memory in AI agents refers to the ability to store and recall skills, rules and learned behaviors that enable an agent to perform tasks automatically without explicit reasoning each time.

It is inspired by human procedural memory, which allows people to perform actions such as riding a bike or typing without consciously thinking about each step. In AI, procedural memory helps agents improve efficiency by automating complex sequences of actions based on prior experiences.

AI agents learn sequences of actions through training, often using reinforcement learning to optimize performance over time. By storing task-related procedures, AI agents can reduce computation time and respond faster to specific tasks without reprocessing data from scratch.

## Frameworks for agentic AI memory

Developers implement memory using external storage, specialized architectures and feedback mechanisms. Since AI agents vary in complexity—ranging from simple reflex agents to advanced learning agents—memory implementation depends on the [agent’s architecture](https://www.ibm.com/think/topics/agentic-architecture), use case and required adaptability.

### LangChain

One key [agent framework](https://www.ibm.com/think/insights/top-ai-agent-frameworks) for building memory-enabled AI agents is [LangChain](https://www.ibm.com/think/topics/langchain), which facilitates the integration of memory, [APIs](https://www.ibm.com/think/topics/api) and reasoning [workflows](https://www.ibm.com/think/topics/agentic-workflows). By combining LangChain with [vector databases](https://www.ibm.com/think/topics/vector-database), AI agents can efficiently store and retrieve large volumes of past interactions, enabling more coherent responses over time.

### LangGraph

[LangGraph](https://www.ibm.com/think/topics/langgraph) allows developers to construct hierarchical memory graphs for AI agents, improving their ability to track dependencies and learn over time.

By integrating vector databases, agentic systems can efficiently store embeddings of previous interactions, enabling contextual recall. This is useful for AI-driven docs generation, where an agent must remember user preferences and past modifications.

### Other open source offerings

The rise of [open source](https://www.ibm.com/think/topics/open-source) frameworks has accelerated the development of memory-enhanced AI agents. Platforms such as GitHub host numerous repositories that provide tools and templates for integrating memory into [AI workflows](https://www.ibm.com/think/topics/ai-workflow).

Additionally, [Hugging Face](https://huggingface.co/) offers pretrained models that can be fine-tuned with memory components to improve AI recall capabilities. Python, a dominant language in AI development, provides libraries for handling [orchestration](https://www.ibm.com/think/topics/ai-agent-orchestration), memory storage and retrieval mechanisms, making it a go-to choice for implementing AI memory systems.

</details>
