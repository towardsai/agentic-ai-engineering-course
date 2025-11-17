# Research

## Research Results

<details>
<summary>What are the fundamental architectural differences between a standard RAG pipeline and an agentic RAG system?</summary>

### Source [2]: https://weaviate.io/blog/what-is-agentic-rag

Query: What are the fundamental architectural differences between a standard RAG pipeline and an agentic RAG system?

Answer: The architecture of agentic RAG is characterized by the incorporation of AI agents that can act as routers or coordinators, making dynamic decisions about information retrieval and task execution:

- **Single-Agent vs. Multi-Agent**: In its simplest form, agentic RAG may use a single agent to decide from which external knowledge source to retrieve information, potentially using APIs, databases, or web searches. In contrast, more advanced systems chain multiple agents together, each specializing in different retrieval or processing tasks.
- **Complex Coordination**: A multi-agent system might include a master agent that delegates retrieval to domain-specific agents, each responsible for different data domains (internal data, personal accounts, public web).
- **Beyond Retrieval**: Agents in agentic RAG can perform functions beyond retrieval, such as reasoning, tool usage, or workflow execution, supporting greater flexibility and extensibility compared to the fixed retrieval-generation sequence of standard RAG.
- **Implementation Flexibility**: Agentic RAG can be implemented using language models with function calling or agent frameworks, offering more control and adaptability than standard RAG pipelines.

Agentic RAG thus moves from a rigid, sequential architecture to a flexible, agent-driven system capable of dynamic orchestration and task specialization.

-----

-----

-----

### Source [3]: https://galileo.ai/blog/agentic-rag-integration-ai-architecture

Query: What are the fundamental architectural differences between a standard RAG pipeline and an agentic RAG system?

Answer: Agentic RAG systems distinguish themselves by integrating autonomous decision-making agents into the retrieval-augmented generation framework:

- **Autonomous Evaluation and Orchestration**: Agents independently assess information needs, orchestrate multi-step retrieval processes, and refine generated outputs through feedback and reasoning loops. Standard RAG, in contrast, typically follows a static, two-step retrieval-then-generation process.
- **Multi-hop Reasoning**: Agentic RAG enables complex, multi-hop reasoning, where agents can decompose problems, perform query criticism, and continuously improve the response by iteratively refining their approach.
- **Minimal Human Intervention**: The architecture aims for high autonomy, reducing the need for human-guided correction or intervention.
- **Strategic Planning**: Agents can plan, strategize, and adapt their behavior dynamically to meet the demands of complex queries, a capability not present in standard RAG pipelines.

These architectural advances enable agentic RAG systems to deliver superior accuracy, contextual relevance, and reasoning capability.

-----

-----

-----

### Source [4]: https://arize.com/blog/understanding-agentic-rag/

Query: What are the fundamental architectural differences between a standard RAG pipeline and an agentic RAG system?

Answer: Agentic RAG introduces several distinctive architectural features that separate it from standard RAG pipelines:

- **Dynamic Retrieval**: Agents adapt their retrieval strategy based on query complexity, context, and user intent, rather than relying on a fixed embedding and similarity search pipeline.
- **Conditional Retrieval**: Agents can decide when retrieval is necessary, sometimes using only internal model knowledge without external data, unlike standard RAG which always performs retrieval.
- **Adaptive Granularity**: Retrieval depth and specificity can be modulated by agents, retrieving summaries for broad queries or granular details for specific ones.
- **Iterative Refinement**: Agents can iteratively adjust search parameters and methods if initial results are inadequate, broadening or narrowing the search as needed.
- **Source Prioritization**: Agents can prioritize data sources based on query context or historical performance, optimizing relevance and accuracy.

These capabilities reflect a shift from static, deterministic workflows to adaptive, agent-driven architectures.

-----

-----

-----

### Source [5]: https://www.moveworks.com/us/en/resources/blog/what-is-agentic-rag

Query: What are the fundamental architectural differences between a standard RAG pipeline and an agentic RAG system?

Answer: Agentic RAG represents an evolution of the standard RAG pipeline by introducing intelligent, autonomous agents with real-time planning, execution, and optimization abilities:

- **Strategic Decision-Making**: Agents autonomously analyze input, make strategic decisions, and perform multi-step reasoning to handle complex queries across diverse data sources, surpassing the rule-based, linear processing of standard RAG.
- **Enhanced Retrieval**: Advanced retrieval strategies, such as reranking and hybrid search, are employed by agents; multiple vectors per document and semantic caching further refine retrieval and response consistency.
- **Multimodal Integration**: Agentic RAG systems can incorporate non-textual data (e.g., images), supporting richer and more versatile interactions.
- **Specialized Tools**: Agents utilize a variety of tools for functions such as entity recognition, summarization, translation, and code generation, enabling more specialized and efficient task execution compared to standard RAG’s general-purpose retrieval and generation.

Overall, agentic RAG systems are characterized by adaptivity, autonomy, and the ability to integrate specialized tools and strategies for superior performance in complex environments.

-----

-----

</details>

<details>
<summary>What are the best practices and common pitfalls when designing the offline ingestion and online retrieval phases of a RAG pipeline for production systems?</summary>

### Source [7]: https://www.ragie.ai/blog/the-architects-guide-to-production-rag-navigating-challenges-and-building-scalable-ai

Query: What are the best practices and common pitfalls when designing the offline ingestion and online retrieval phases of a RAG pipeline for production systems?

Answer: The design of a production RAG system depends on careful architectural decisions involving the **knowledge base, retriever, and generator**. Best practices include fine-tuning **chunking logic** (how documents are split for retrieval), implementing **reranking models** to improve retrieval accuracy, and optimizing orchestration layers for performance and scalability. The system should be designed to retrieve only the most relevant context, delivering results with **low latency**, cost-effectiveness, and security. Major pitfalls are improper chunking, insufficient retrieval accuracy, high latency, and overlooking security or cost controls. Building from scratch allows deep customization but requires significant expertise. Using a managed RAG service can mitigate operational risks and accelerate deployment, as these platforms often embed best practices and handle complexity internally.

-----

-----

-----

### Source [8]: https://orkes.io/blog/rag-best-practices/

Query: What are the best practices and common pitfalls when designing the offline ingestion and online retrieval phases of a RAG pipeline for production systems?

Answer: Key best practices for production-scale RAG systems include focusing on **retrieval quality**, **search speed**, and **response quality**. Custom workers or microservices can be used for specialized tasks such as reranking search results using advanced models (e.g., Cohere’s rerank API). The use of orchestrators (such as Conductor) allows for modular, scalable, and maintainable pipelines, where each component—retrieval, reranking, and generation—can be independently optimized or replaced. Prompt engineering is highlighted as crucial for effective LLM output, with tools to manage and test prompts systematically. Common pitfalls are neglecting to rerank retrieved results, underestimating the importance of prompt templates, or failing to orchestrate and monitor the pipeline effectively, which can result in irrelevant responses, lower system reliability, and difficulty scaling.

-----

-----

-----

### Source [9]: https://docs.llamaindex.ai/en/stable/optimizing/production_rag/

Query: What are the best practices and common pitfalls when designing the offline ingestion and online retrieval phases of a RAG pipeline for production systems?

Answer: For performant production RAG applications, **structured retrieval** techniques are important, especially as document volume grows. One best practice is **tagging documents with metadata** and using metadata filters at retrieval time, which helps the system narrow down candidates before semantic search. This is supported by major vector databases. Another approach is to **store document hierarchies**, where summaries are embedded and linked to raw chunks; retrieval first happens at the summary level, then drills down to the chunk. This enables more scalable and precise retrieval. Pitfalls include difficulty in defining effective metadata tags, the risk of tags being too coarse for precise filtering, and the computational cost of auto-generating summaries. Failing to implement structured retrieval can lead to poor relevance as the dataset scales, and ignoring these strategies may result in inefficiency or diminished accuracy in large-scale deployments.

-----

-----

-----

### Source [10]: https://resources.nvidia.com/en-us-nim/rag-application-pilot

Query: What are the best practices and common pitfalls when designing the offline ingestion and online retrieval phases of a RAG pipeline for production systems?

Answer: Although the full content is not displayed, the NVIDIA source focuses on moving RAG applications from pilot to production. Typical best practices referenced by NVIDIA include robust **monitoring and evaluation** (to track model performance and data drift), optimizing **latency and throughput** (critical for user experience), and automating the **deployment pipeline** for consistency and reliability. Common pitfalls are insufficient monitoring, overlooking scalability requirements, and not rigorously testing the system under production loads, which can lead to degraded performance or unexpected failures as usage grows.

-----

-----

</details>

<details>
<summary>How does GraphRAG specifically improve retrieval for queries that require understanding complex relationships across multiple documents, compared to vector-based hybrid search?</summary>

### Source [11]: https://www.chitika.com/graph-rag-vs-vector-rag/

Query: How does GraphRAG specifically improve retrieval for queries that require understanding complex relationships across multiple documents, compared to vector-based hybrid search?

Answer: **Graph-Based RAG** excels in scenarios where understanding and reasoning over **highly interconnected datasets** is required, such as healthcare, because it supports **multi-hop queries**—queries that demand traversing complex relationships or chains of associated facts across multiple documents. This deep reasoning capability provides **precise insights** but comes with increased **computational overhead** due to the complexity of graph traversal algorithms. The result is more accurate retrieval for queries involving nuanced or indirect relationships, but it can be less scalable and slower, especially as the data grows.

In contrast, **Vector-Based RAG** is optimized for **scalability and speed**, performing well in environments where semantic similarity is sufficient (such as e-commerce). However, it can struggle in domains requiring **contextual nuance** or when critical facts are distributed across multiple documents and not directly captured in the closest vector matches.

A **hybrid approach** can mitigate these trade-offs by using vector search for initial broad semantic filtering and graph traversal for deeper reasoning, effectively balancing speed and relationship understanding in retrieval tasks.

-----

-----

-----

### Source [12]: https://arxiv.org/html/2507.03608v1

Query: How does GraphRAG specifically improve retrieval for queries that require understanding complex relationships across multiple documents, compared to vector-based hybrid search?

Answer: Empirical benchmarks demonstrate that **GraphRAG and Hybrid GraphRAG outperform traditional vector-based RAG** on queries requiring complex reasoning, particularly as question difficulty increases. For medium and hard questions—where answers require integrating facts spread across multiple documents—**graph-based pipelines show higher faithfulness and factual correctness**. Specifically, GraphRAG and Hybrid GraphRAG achieve faithfulness scores of 0.59 versus 0.55 for vector RAG, and Hybrid GraphRAG attains the highest factual correctness (0.58), reflecting its strength in complex retrieval scenarios.

Vector RAG performs best for straightforward queries but its effectiveness drops when semantic similarity alone is insufficient—such as when queries require synthesizing indirect or multi-hop relationships not co-located in the same document chunk.

Hybrid GraphRAG further improves performance by **compensating for gaps** in either method: it leverages vector retrieval when graph context is missing, and graph traversal when relationships must be pieced together from multiple sources.

-----

-----

-----

### Source [13]: https://weaviate.io/blog/graph-rag

Query: How does GraphRAG specifically improve retrieval for queries that require understanding complex relationships across multiple documents, compared to vector-based hybrid search?

Answer: The **local search** approach in GraphRAG combines knowledge graph navigation with text-based retrieval. When a query is received, the system:

- Recognizes key entities within the query.
- Uses those entities as **entry points** into the knowledge graph.
- Navigates relationships to collect connected entities and attributes.
- Pulls in context from the source documents.

This process allows the system to **uncover direct and indirect relationships** that are often **missed by vector search alone**, which focuses on semantic similarity rather than explicit connections. By **integrating graph traversal with vector semantic search**, GraphRAG can provide more comprehensive answers to queries that require understanding the **interplay between multiple entities and facts** scattered across documents.

-----

-----

-----

### Source [14]: https://www.useparagon.com/blog/vector-database-vs-knowledge-graphs-for-rag

Query: How does GraphRAG specifically improve retrieval for queries that require understanding complex relationships across multiple documents, compared to vector-based hybrid search?

Answer: A **hybrid approach** uses a knowledge graph to represent and query complex relationships, while a vector database handles unstructured data for semantic search. The retrieval process involves:

- **Entity extraction:** Identifying key entities and relationships in the query.
- **Vector search:** Narrowing down relevant graph nodes based on semantic similarity.
- **Graph traversal:** Extracting related entities, relationships, and metadata relevant to the query.
- **Response generation:** Using the combined context to answer the query.

This method allows for both **deep relationship reasoning** (via the knowledge graph) and **broad semantic coverage** (via vector embeddings), resulting in more accurate retrieval for queries that span multiple documents and require an understanding of how entities are interrelated.

-----

-----

-----

### Source [15]: https://www.falkordb.com/blog/graph-rag-vs-vector-rag-solving-gartner-challenges/

Query: How does GraphRAG specifically improve retrieval for queries that require understanding complex relationships across multiple documents, compared to vector-based hybrid search?

Answer: **Graph RAG** inherently combines multiple retrieval methods by leveraging **graph structures** that explicitly model relationships, making it particularly effective for complex queries where understanding **the context and connections among data points** is crucial. This approach addresses concerns about the limitations of pure vector search, which relies on similarity and may miss nuanced relationships.

For example, in a healthcare setting, a question like “What are the potential side effects of [specific medication] for a patient with [relevant medical history]?” can be transformed and disambiguated through the graph structure, facilitating more precise retrieval across multiple interconnected documents.

Graph RAG’s **structured retrieval** is more adept at handling queries requiring cross-document reasoning and relationship synthesis, as opposed to vector RAG’s focus on direct similarity.

-----

-----

</details>

<details>
<summary>What are the most effective query transformation techniques, such as HyDE and multi-query decomposition, for improving retrieval relevance in RAG systems?</summary>

### Source [16]: https://www.pondhouse-data.com/blog/advanced-rag-hypothetical-document-embeddings

Query: What are the most effective query transformation techniques, such as HyDE and multi-query decomposition, for improving retrieval relevance in RAG systems?

Answer: **HyDE (Hypothetical Document Embeddings)** is an advanced technique for improving the retrieval step in Retrieval-Augmented Generation (RAG) systems. Traditional RAG relies on direct similarity between the user's query and document embeddings, but this often fails to capture the nuanced intent behind queries. HyDE addresses this by generating a **hypothetical document embedding** representing the ideal document to answer the user's query. This embedding then guides the retrieval process, increasing the likelihood of retrieving documents that are actually relevant to the underlying intent, not just keyword matches. The integration of HyDE involves setting up a vector store index and configuring the query engine to generate and embed these hypothetical documents. The blog provides a practical tutorial on implementing HyDE, emphasizing its role in bridging the gap between user queries and the most relevant documents by focusing retrieval not on the literal query, but on a richer, LLM-generated hypothetical answer[1].

-----

-----

-----

### Source [17]: https://www.chitika.com/hyde-query-expansion-rag/

Query: What are the most effective query transformation techniques, such as HyDE and multi-query decomposition, for improving retrieval relevance in RAG systems?

Answer: **Query expansion via HyDE** transforms vague or ambiguous queries into semantically rich hypothetical documents, significantly improving retrieval relevance in RAG systems. HyDE works by capturing the user's intent rather than just matching keywords, embedding the generated hypothetical document into a dense vector space for retrieval. This technique is especially effective for zero-shot retrieval (i.e., when no labeled data is available) and for handling complex or poorly phrased queries. HyDE has proven particularly valuable in specialized domains, such as legal research, where it enables the system to generate nuanced hypothetical documents that reflect intricate argumentation. A key limitation is that the quality of retrieval depends on the underlying language model's domain understanding, so domain-specific fine-tuning may be necessary. Overall, HyDE bridges the semantic gap between user intent and document representation, improving accuracy, handling ambiguous queries, and boosting performance in zero-shot and domain-specific scenarios[2].

-----

-----

-----

### Source [18]: https://blog.gopenai.com/part-5-advanced-rag-techniques-llm-based-query-rewriting-and-hyde-dbcadb2f20d1

Query: What are the most effective query transformation techniques, such as HyDE and multi-query decomposition, for improving retrieval relevance in RAG systems?

Answer: This source discusses **LLM-based query rewriting** and **HyDE** as two incremental improvements to basic RAG systems. Standard RAG systems often use a single prompt for both document retrieval and answer generation, which can be inefficient. Query transformation techniques like LLM-based rewriting allow the query to be optimized specifically for retrieval, while HyDE generates a hypothetical ideal answer, which is then embedded and used for retrieval. Both methods help bridge the gap between user queries and document content, improving retrieval precision and overall RAG system performance. The article provides practical guidance on integrating these techniques into RAG chatbots, highlighting their importance for anyone aiming to optimize advanced retrieval workflows[3].

-----

-----

-----

### Source [19]: https://wandb.ai/site/articles/rag-techniques/

Query: What are the most effective query transformation techniques, such as HyDE and multi-query decomposition, for improving retrieval relevance in RAG systems?

Answer: This article outlines several **query transformation techniques** crucial for improving retrieval relevance in RAG systems:

- **Query rewriting:** Expands or clarifies the original query by adding details, expanding keywords, or resolving acronyms, making it more suitable for the retrieval system.
- **Query decomposition:** Breaks down complex or multi-faceted queries into simpler sub-queries. For example, a complex comparison is split into separate, focused questions, and retrieval is performed for each sub-query. The results are then combined to generate a comprehensive answer.

These techniques are essential for handling brief, ambiguous, or compound queries—ensuring that the retrieval component can access more relevant and specific information chunks from the document corpus. This results in more accurate and context-aware RAG responses[4].

-----

-----

-----

### Source [20]: https://docs.llamaindex.ai/en/stable/examples/query_transformations/HyDEQueryTransformDemo/

Query: What are the most effective query transformation techniques, such as HyDE and multi-query decomposition, for improving retrieval relevance in RAG systems?

Answer: The official LlamaIndex documentation demonstrates how **HyDE** can substantially improve retrieval and output quality in RAG systems. In the example, HyDE generates a hypothetical document that accurately represents what an expert answer would look like for a given query. This hypothetical document is then embedded, and the retrieval step uses this embedding to find the most relevant information in the corpus. The example shows that HyDE can generate highly contextually relevant hypothetical answers, leading to better embedding quality and more accurate final outputs, especially for queries where the original phrasing may not directly match the relevant documents. The documentation provides code snippets for integrating HyDE into a LlamaIndex-powered RAG pipeline and shows its impact through practical use cases[5].

-----

-----

-----

### Source [96]: https://www.pondhouse-data.com/blog/advanced-rag-hypothetical-document-embeddings

Query: What are the most effective query transformation techniques, such as HyDE and multi-query decomposition, for improving retrieval relevance in RAG systems?

Answer: **HyDE (Hypothetical Document Embeddings)** is highlighted as a significant query transformation technique for improving retrieval in RAG systems. Instead of relying solely on direct query-document embedding similarity, HyDE generates hypothetical document embeddings that represent what the ideal answer document would look like for a given query. By using these “ideal” embeddings as the retrieval target, HyDE helps the system find documents that are more semantically relevant and closely aligned with the user’s intent. This approach bridges the gap between queries and actual documents, especially when the query is nuanced or when the corpus is heterogeneous.

The source also provides practical insights into integrating HyDE: by configuring the vector store and query engine accordingly, HyDE can be incorporated into RAG pipelines. The result is enhanced retrieval relevance, especially for complex or ambiguous queries, as HyDE steers retrieval toward the most contextually appropriate content.

-----

-----

-----

### Source [97]: https://www.chitika.com/hyde-query-expansion-rag/

Query: What are the most effective query transformation techniques, such as HyDE and multi-query decomposition, for improving retrieval relevance in RAG systems?

Answer: **Query expansion using HyDE** transforms retrieval in RAG by creating richer, intent-focused queries. HyDE generates hypothetical documents that encapsulate the user’s underlying intent, effectively translating ambiguous or poorly phrased queries into semantically enriched representations. This enables retrieval systems to operate within a more meaningful framework, improving precision and relevance.

A unique strength of HyDE is its zero-shot retrieval capability, which removes the need for labeled data and adapts well to complex, open-ended, or domain-specific queries. By embedding these hypothetical documents into a dense vector space, the system can close the semantic gap between user intent and document representation, outperforming traditional keyword or direct embedding matching.

The source also notes that in specialized domains, the quality of generated hypothetical documents depends on the LLM’s training, and fine-tuning may be necessary for optimal results.

-----

-----

-----

### Source [98]: https://blog.gopenai.com/part-5-advanced-rag-techniques-llm-based-query-rewriting-and-hyde-dbcadb2f20d1

Query: What are the most effective query transformation techniques, such as HyDE and multi-query decomposition, for improving retrieval relevance in RAG systems?

Answer: This source describes two major query transformation techniques for RAG:

- **LLM-Based Query Rewriting:** This method uses large language models to rewrite user queries, making them more suitable for retrieval. By separating the retrieval prompt from the generation prompt, the system can optimize each for its distinct purpose, leading to more effective document search and answer generation.

- **HyDE:** HyDE is discussed as an alternative or complement to rewriting, where the LLM generates a hypothetical answer (or document) that represents what the user is seeking. The system then uses this hypothetical document as the query for retrieval, placing the search in a semantically richer context.

Both techniques address the inefficiency of using the same prompt for retrieval and generation, leading to more relevant results and improved RAG system performance.

-----

-----

-----

### Source [99]: https://wandb.ai/site/articles/rag-techniques/

Query: What are the most effective query transformation techniques, such as HyDE and multi-query decomposition, for improving retrieval relevance in RAG systems?

Answer: This source summarizes **query transformation techniques** for RAG, focusing on:

- **Query rewriting:** Expands or clarifies the original query (e.g., translating “AI latest” to “Recent advancements in artificial intelligence technology”), making it more precise for retrieval.
- **Query decomposition:** Breaks down complex queries into simpler sub-queries. Each sub-query is retrieved separately, and the results are combined to form a comprehensive answer. For example, a multi-part health comparison query is decomposed into individual health benefit questions for each item, allowing targeted retrieval from the corpus.

These transformations make user inputs more digestible for retrieval systems, especially when queries are brief, ambiguous, or complex.

-----

-----

-----

### Source [100]: https://docs.llamaindex.ai/en/stable/examples/query_transformations/HyDEQueryTransformDemo/

Query: What are the most effective query transformation techniques, such as HyDE and multi-query decomposition, for improving retrieval relevance in RAG systems?

Answer: This documentation provides an example of **HyDE** in practice. HyDE “hallucinates” (generates) a plausible answer to the query, then uses the embedding of this hypothetical answer to retrieve relevant documents. The example shows that by hypothesizing what an ideal answer would look like, the system creates higher-quality embeddings, leading to significantly improved retrieval and final output.

The process can include the original query along with the hypothetical document, further enhancing retrieval performance. This technique is especially powerful when the original query is vague or lacks context, as the hypothetical document serves to bridge the gap between query and the latent structure of the corpus.

-----

-----

</details>

<details>
<summary>How does a ReAct agent decide when to use a RAG retrieval tool versus other available tools like a web search or code interpreter?</summary>

### Source [21]: https://arize.com/blog/understanding-agentic-rag/

Query: How does a ReAct agent decide when to use a RAG retrieval tool versus other available tools like a web search or code interpreter?

Answer: A ReAct agent within an agentic RAG system dynamically decides when and how to use a RAG retrieval tool versus other tools by analyzing the specifics of the user query and context. The agent can perform **conditional retrieval**, determining if external retrieval is necessary or if its internal knowledge suffices for simple questions. It can also employ **adaptive granularity**, adjusting the detail level of information retrieved based on query complexity. When initial retrieval results are inadequate, the agent engages in **iterative refinement**, modifying retrieval parameters, changing retrieval methods, or querying related concepts to enhance results. Importantly, the agent can perform **source prioritization**, selecting data sources based on the query’s context or previous success rates. For instance, if a query relates to recent events, the agent may select a real-time web search or news API over a static RAG document store[1].

-----

-----

-----

### Source [22]: https://workativ.com/ai-agent/blog/agentic-rag

Query: How does a ReAct agent decide when to use a RAG retrieval tool versus other available tools like a web search or code interpreter?

Answer: ReAct agents in an agentic RAG pipeline combine **routing, query planning, and tool use** to address queries. The **routing agent** analyzes the user query using LLM-based reasoning to determine which RAG pipeline or tool (such as a retrieval system, web search, or code interpreter) is most suitable. **Query planning agents** decompose complex queries into sub-queries and assign them to specialized agents or tools, later consolidating the outputs. The agent’s dynamic planning process allows it to adapt its tool selection based on the query’s demands and real-time data availability, iteratively executing steps until the complete task is resolved. This means a ReAct agent will choose a RAG retrieval tool when a query requires access to specific indexed documents, while a web search or code interpreter might be used for real-time data needs or computational tasks, respectively[2].

-----

-----

-----

### Source [23]: https://airbyte.com/data-engineering-resources/using-langchain-react-agents

Query: How does a ReAct agent decide when to use a RAG retrieval tool versus other available tools like a web search or code interpreter?

Answer: LangChain’s ReAct agents select tools through a reasoning-and-action loop. The agent maintains a **scratchpad** of its thought process, alternating between reasoning steps and tool invocations (such as RAG retrieval, web search, or code execution). The agent assesses the sufficiency of information from each tool and can recognize when additional or alternative data sources are needed. ReAct agents are particularly effective for complex queries that require multi-step reasoning or validation, dynamically seeking further external data if earlier tool outputs are inadequate. This iterative, evaluative process underpins tool selection, with the agent switching between RAG, web search, or code interpreter based on the evolving informational requirements of the query[3].

-----

-----

-----

### Source [24]: https://docs.llamaindex.ai/en/stable/examples/agent/react_agent_with_query_engine/

Query: How does a ReAct agent decide when to use a RAG retrieval tool versus other available tools like a web search or code interpreter?

Answer: In LlamaIndex’s implementation, a ReAct agent is equipped with multiple **query engine (tool) options**, such as querying specific document stores (e.g., company financial reports). The agent decides which tool to use based on the content and requirements of the user query. The ReAct agent’s reasoning loop (thoughts and actions) is visible in its streamed output, showing how it evaluates which tool (RAG retrieval or otherwise) is likely to provide the needed information. If a question pertains to a particular document, the agent will select the corresponding retrieval tool; for other types of queries, it may choose different tools configured in its environment[4].

-----

-----

-----

### Source [25]: https://docs.smith.langchain.com/evaluation/tutorials/rag

Query: How does a ReAct agent decide when to use a RAG retrieval tool versus other available tools like a web search or code interpreter?

Answer: The ReAct pattern, as used in LangSmith’s evaluation framework, integrates **reasoning and acting**, allowing the agent to select and invoke tools (e.g., Wikipedia search API, RAG retrieval, or others), then reason about the outputs to decide the next action. The agent’s tool selection is guided by the initial analysis of the question and iterative assessment of the helpfulness and relevance of each tool’s results. If a tool’s output does not sufficiently address the query, the agent can switch tools or refine its approach. This cycle continues until the agent is confident in its answer, ensuring that the most appropriate tool (RAG, web search, code interpreter, etc.) is used at each decision point[5].
-----

-----

-----

### Source [71]: https://arize.com/blog/understanding-agentic-rag/

Query: How does a ReAct agent decide when to use a RAG retrieval tool versus other available tools like a web search or code interpreter?

Answer: A ReAct agent operating within an Agentic RAG system decides when to use a RAG retrieval tool versus other tools through several adaptive mechanisms:

- **Conditional Retrieval:** The agent first determines if external retrieval is necessary at all. For straightforward questions answerable from internal knowledge, it may skip RAG or any external tool.
- **Adaptive Granularity:** Based on the complexity of the query, the agent adjusts how much detail it retrieves. For broad queries, it may fetch summaries; for specific questions, it retrieves granular details from RAG or other sources.
- **Iterative Refinement:** If the initial retrieval (from RAG or otherwise) is insufficient, the agent can refine its search parameters, switch retrieval methods, or query related concepts to expand or narrow the search.
- **Source Prioritization:** The agent can learn or be configured to prioritize certain sources. For example, for recent events, it may use a web search tool or real-time API instead of RAG, which often relies on static document repositories.

This decision-making is dynamic and context-dependent, allowing the agent to select the most appropriate tool—including RAG, web search, or code interpreters—based on query type, available data, and required specificity[1].

-----

-----

-----

### Source [72]: https://workativ.com/ai-agent/blog/agentic-rag

Query: How does a ReAct agent decide when to use a RAG retrieval tool versus other available tools like a web search or code interpreter?

Answer: ReAct agents within agentic RAG pipelines employ a combination of routing, query planning, and tool selection:

- **Routing Agent:** Uses LLM-based reasoning to determine which RAG pipeline or tool is most relevant for a given query.
- **Query Planning:** For complex queries, the agent breaks them into sub-queries and assigns them to agents with access to specific tools (e.g., RAG, web search, code interpreter).
- **Iterative Reasoning:** The agent dynamically adapts its plan based on real-time data and user interactions. It can switch between tools as necessary, integrating results from RAG retrieval, web search, or code execution to construct a comprehensive answer.

The agent's planner and executor components ensure that each step, including tool selection, is optimized according to the context and requirements of the user's request. This allows the agent to flexibly decide when a RAG retrieval tool is appropriate versus other available tools[2].

-----

-----

-----

### Source [73]: https://airbyte.com/data-engineering-resources/using-langchain-react-agents

Query: How does a ReAct agent decide when to use a RAG retrieval tool versus other available tools like a web search or code interpreter?

Answer: LangChain’s ReAct agents decide among RAG, web search, and code interpreter tools through iterative reasoning and tool interaction:

- **Tool Selection Framework:** Agents use structured reasoning cycles, alternating between thinking ("thoughts") and acting ("actions"). Each action may involve invoking a different tool, such as a RAG retriever, a web search API, or a code interpreter.
- **Dynamic Tool Use:** If initial tool use (for example, a RAG retrieval) does not yield sufficient information, the agent can recognize the need to try alternative tools, such as performing a web search or running a code snippet.
- **State Management:** The agent maintains a scratchpad of its reasoning and evidence, allowing it to re-evaluate previous steps and switch tools as new information emerges.

This iterative, tool-agnostic approach enables ReAct agents to handle complex queries by selecting the most effective tool at each decision point[3].

-----

-----

-----

### Source [74]: https://docs.llamaindex.ai/en/stable/examples/agent/react_agent_with_query_engine/

Query: How does a ReAct agent decide when to use a RAG retrieval tool versus other available tools like a web search or code interpreter?

Answer: In LlamaIndex's implementation, a ReAct agent can be configured with multiple tools (such as separate RAG query engines for different document sets):

- The agent's decision to use a specific RAG retrieval tool (e.g., querying the "Lyft 10-K" vs. "Uber 10-K") is governed by its reasoning process about which tool is relevant to the user's question.
- The agent streams its reasoning process and tool calls, providing transparency into why a particular retrieval engine was chosen.
- This reasoning can be extended to include other tools (like web search or code interpretation), with the agent selecting based on the context of the question and the specificity of information required[4].

-----

-----

-----

### Source [75]: https://docs.smith.langchain.com/evaluation/tutorials/rag

Query: How does a ReAct agent decide when to use a RAG retrieval tool versus other available tools like a web search or code interpreter?

Answer: LangChain’s evaluation documentation highlights that ReAct agents interleave reasoning and tool usage, such as calling a Wikipedia search API or other retrieval mechanisms:

- **Reasoning Before Acting:** The agent analyzes whether the current question requires information beyond its internal context and, if so, selects the most relevant tool.
- **Observation and Adaptation:** After using a tool, the agent observes the results and decides whether additional actions (such as trying a different retrieval method or switching to a code interpreter) are needed to answer the question fully.
- **Iterative Process:** This cycle continues until the agent is confident it has enough information to answer the user’s query.

The decision to use a RAG retrieval tool versus alternatives is thus an outcome of this ongoing, context-driven reasoning and observation loop[5].
-----

-----

</details>

<details>
<summary>What are the fundamental differences and trade-offs between various document chunking strategies like fixed-size, recursive character splitting, and semantic chunking for RAG pipelines?</summary>

### Source [26]: https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/

Query: What are the fundamental differences and trade-offs between various document chunking strategies like fixed-size, recursive character splitting, and semantic chunking for RAG pipelines?

Answer: **Document chunking** is a crucial preprocessing step in RAG pipelines, as it determines how documents are divided into smaller, manageable units for indexing and retrieval. The chunking strategy directly affects retrieval precision, efficiency, and the overall quality of AI-generated responses. Poorly chosen chunk sizes or methods can scatter related information, resulting in irrelevant or incomplete outputs and excessive computational burden. Effective chunking improves retrieval precision and context coherence, which leads to more accurate and helpful results for users, as well as operational efficiency for businesses.

The source describes several chunking approaches:
- **Page-level, section-level, and token-based chunking**: Each varies in how much context is preserved and how precisely relevant information can be retrieved. For example, larger chunks (such as whole pages or sections) provide more context but may include unrelated material, while smaller chunks (like token-based) improve retrieval specificity but risk losing necessary context.
- The **trade-off** is thus between *contextual richness* and *retrieval precision*: large chunks offer more context but less precise retrieval, while small chunks are more precise but may not provide enough context for the AI to generate accurate answers.

The optimal strategy is highly dataset- and use-case-dependent, requiring empirical evaluation to balance these trade-offs for specific RAG applications.

-----

-----

-----

### Source [27]: https://infohub.delltechnologies.com/es-es/p/chunk-twice-retrieve-once-rag-chunking-strategies-optimized-for-different-content-types/

Query: What are the fundamental differences and trade-offs between various document chunking strategies like fixed-size, recursive character splitting, and semantic chunking for RAG pipelines?

Answer: This source emphasizes that the choice of chunking strategy in RAG systems is essential for preserving semantic relationships and ensuring that chunks represent the smallest meaningful units of knowledge for LLMs. Poor chunking can fragment related concepts across chunks, harming retrieval accuracy. Smart chunking, by contrast, maintains semantic integrity and improves retrieval.

Key points about different strategies:
- The chunking method should align with the *inherent structure* of the document (e.g., sections, paragraphs, or logical units).
- The strategy should strive to make each chunk a *self-contained unit of knowledge* that matches how LLMs process information.
- Trade-offs involve balancing the preservation of context with the need for precise, relevant retrieval. Overly broad chunks may dilute the relevance of retrieved information, while overly narrow chunks may strip away essential context.

The chunking strategy should be chosen based on document type and retrieval requirements, not as a one-size-fits-all solution.

-----

-----

-----

### Source [28]: https://www.snowflake.com/en/engineering-blog/impact-retrieval-chunking-finance-rag/

Query: What are the fundamental differences and trade-offs between various document chunking strategies like fixed-size, recursive character splitting, and semantic chunking for RAG pipelines?

Answer: This source compares three main chunking strategies:

- **Recursive chunking**: Splits text hierarchically using delimiters, starting with broader semantic boundaries (e.g., paragraphs via double newlines) and proceeding to finer splits if needed. This approach preserves natural language boundaries and adapts flexibly to document structure, making it robust across varied content types.
- **Semantic chunking**: Uses sentence embeddings to find natural breakpoints based on semantic shifts. By calculating cosine distances between consecutive sentence embeddings, it splits where meaning changes significantly. This method is particularly suited to preserving semantic integrity but may increase computational complexity due to embedding calculations.
- **Markdown-header-based chunking**: Splits documents at markdown headers, aligning chunks with structurally significant sections (e.g., "Risk Factors"). This leverages document organization to maintain context and is especially useful for structured documents.

**Trade-offs**:
- Recursive chunking is flexible and robust but may sometimes split at suboptimal points if the delimiter hierarchy does not perfectly match semantic transitions.
- Semantic chunking often yields the most contextually coherent chunks but is computationally more expensive.
- Header-based chunking ties chunks to human-understandable sections but may miss fine-grained context changes within sections.

-----

-----

-----

### Source [29]: https://www.superteams.ai/blog/a-deep-dive-into-chunking-strategy-chunking-methods-and-precision-in-rag-applications

Query: What are the fundamental differences and trade-offs between various document chunking strategies like fixed-size, recursive character splitting, and semantic chunking for RAG pipelines?

Answer: This article explains the foundational role of chunking in breaking down large documents to enable efficient retrieval and generation. It discusses how chunking methods impact the quality of context retrieved and, consequently, the accuracy of generated answers.

It advocates for empirically evaluating different chunking strategies for a given application using pipelines that compare the retrieved contexts and model responses. Different strategies may perform better depending on the type and structure of the document and the nature of queries.

Challenges and pitfalls highlighted include:
- *Too-large chunks*: Can overwhelm retrieval models and include irrelevant information.
- *Too-small chunks*: Risk losing necessary context, leading to incomplete answers.
- *Advanced strategies* (e.g., semantic or recursive splitting): Can help optimize for both precision and context retention, but may require more sophisticated implementation and tuning.

-----

-----

-----

### Source [30]: https://www.tigerdata.com/blog/which-rag-chunking-and-formatting-strategy-is-best

Query: What are the fundamental differences and trade-offs between various document chunking strategies like fixed-size, recursive character splitting, and semantic chunking for RAG pipelines?

Answer: This source details the **recursive character text splitter**, which is customizable for various document types (HTML, Markdown, code). It works by trying a hierarchy of separators (from larger structural elements down to sentences or even spaces) to preserve the underlying document structure.

Key points:
- The recursive approach adapts to document structure, splitting at the highest available structural boundary and only falling back to finer splits as needed.
- The system tracks which chunking and formatting strategies were used for embeddings, facilitating A/B testing and enabling controlled rollouts of new chunking strategies.
- Monitoring and recording chunking strategies is essential for managing changes and ensuring reproducibility in RAG pipelines.

The trade-off with recursive character splitting is between structure preservation and chunk granularity: it excels at maintaining structural coherence, but may still include context boundaries that are not optimally aligned with semantic meaning if structural cues are weak.

-----

</details>

<details>
<summary>How do hybrid search systems in RAG combine keyword-based (like BM25) and vector-based search, and what are the primary benefits of this approach for retrieval accuracy?</summary>

### Source [31]: https://blog.vectorchord.ai/hybrid-search-with-postgres-native-bm25-and-vectorchord

Query: How do hybrid search systems in RAG combine keyword-based (like BM25) and vector-based search, and what are the primary benefits of this approach for retrieval accuracy?

Answer: Hybrid search systems in Retrieval-Augmented Generation (RAG) combine **keyword-based retrieval** (using BM25) and **vector-based (semantic) search** to address the limitations of each individual approach. BM25 excels at ranking documents for queries with clear, exact keywords but struggles to understand synonyms, context, or semantic intent. Conversely, vector-based search, such as that powered by advanced embedding models, captures deep contextual meaning and generalizes well across diverse queries but may miss precision in exact keyword matches.

By integrating BM25’s precise keyword matching with the semantic capabilities of vector search, hybrid systems deliver results that are both contextually relevant and precise. In practical implementation (such as in PostgreSQL with VectorChord-bm25 and VectorChord), the process typically involves:
- Running the query through both BM25 for keyword scoring and a vector similarity algorithm for semantic matching.
- Using algorithms (like **Block-WeakAnd**) to efficiently combine scores from both retrieval methods.
- Producing a merged, re-ranked results list that balances exactness and contextual meaning.

This approach is especially effective for applications like recommendation engines, document retrieval systems, and enterprise search, as it leads to **faster, more accurate, and semantically aware results**[1].

-----

-----

-----

### Source [32]: https://www.elastic.co/what-is/hybrid-search

Query: How do hybrid search systems in RAG combine keyword-based (like BM25) and vector-based search, and what are the primary benefits of this approach for retrieval accuracy?

Answer: Hybrid search combines **keyword or lexical search (such as BM25)** and **semantic/vector search** within a single retrieval pipeline. **BM25** focuses on matching the exact words in a user’s query, making it powerful for structured queries and keyword-heavy content. However, it lacks the ability to interpret meaning or context.

**Semantic search**, driven by vector representations, aims to understand the intent and context behind queries. It uses technologies like NLP and machine learning to generate embeddings that capture meaning, allowing the system to recognize synonyms, related concepts, and user intent—even when the query wording differs from the text in the database.

In practice, hybrid search works by:
- Transforming queries into both keyword and vector representations.
- Retrieving results using both approaches (e.g., k-nearest neighbor for vectors and BM25 for keywords).
- Combining the ranked lists, often by scoring or blending, to address both literal matching and conceptual similarity.

The primary benefit is that hybrid systems can handle complex, ambiguous, or multi-language queries, and are robust against the variability of natural language, often producing more relevant results than either method alone[2].

-----

-----

-----

### Source [33]: https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking

Query: How do hybrid search systems in RAG combine keyword-based (like BM25) and vector-based search, and what are the primary benefits of this approach for retrieval accuracy?

Answer: BM25, a form of **sparse vector search**, calculates document relevance based on the frequency and distinctiveness of query terms (see the provided BM25 formula for details). This method is highly effective for exact term matches. In contrast, **dense vector search** (semantic search) uses high-dimensional embeddings to represent text, capturing semantic relationships and enabling retrieval based on meaning rather than exact wording.

A common hybrid retrieval approach is the **ensemble retriever**, which combines the outputs of keyword and vector retrievers. This is done by assigning weights to each retriever and blending their scores for each document. For example, if a document is highly relevant by BM25 but only moderately relevant by vector similarity, the combined score can reflect this nuanced assessment.

The key benefits of this hybrid strategy are:
- Improved recall and precision, as documents relevant either by exact match or by semantic similarity can be retrieved.
- Better handling of queries with synonyms, paraphrases, or ambiguous meanings.
- A more robust retrieval stage for RAG, boosting the quality and relevance of retrieved context for downstream generative models[3].

-----

-----

-----

### Source [34]: https://cloud.google.com/vertex-ai/docs/vector-search/about-hybrid-search

Query: How do hybrid search systems in RAG combine keyword-based (like BM25) and vector-based search, and what are the primary benefits of this approach for retrieval accuracy?

Answer: Hybrid search in Google Cloud’s Vertex AI Vector Search merges **token-based (keyword/BM25)** and **semantic/vector-based** approaches within a single search index. Token-based (sparse) embeddings represent text by the frequency of each word, using algorithms like BM25 or TF-IDF, and are excellent for capturing explicit keyword matches. Dense embeddings, on the other hand, encode semantic meaning.

When performing hybrid search:
- Both sparse (keyword) and dense (semantic) representations are created for each document and query.
- The retrieval system can be configured to weigh each type of search result according to application needs (e.g., equal weighting, or favoring one over the other).
- Results from both methods are merged and ranked according to these weights.

The benefit of this approach is that it enables the system to retrieve documents that are either an exact match for the query or are semantically relevant, thus **increasing overall search quality and relevance**. Google notes that hybrid search addresses both "search by meaning" and "search by keyword," and is foundational in systems like Google Search since the introduction of RankBrain[4].

-----

-----

-----

### Source [35]: https://www.fuzzylabs.ai/blog-post/improving-rag-performance-hybrid-search

Query: How do hybrid search systems in RAG combine keyword-based (like BM25) and vector-based search, and what are the primary benefits of this approach for retrieval accuracy?

Answer: In RAG systems, hybrid search is built by indexing data with both **dense vector embeddings** (for semantic similarity) and storing the original text for **keyword-based retrieval**. At retrieval time, the user query is converted into an embedding (for semantic search) and is also run as a keyword query (for BM25 or similar methods).

The results from both retrievals are then merged, either by combining the top results from each or by reranking according to a blended score. This allows the system to surface documents that are relevant either because they contain the right keywords or because they are semantically similar, thus addressing the limitations of both standalone approaches.

The primary benefit here is **improved performance for the retrieval stage in RAG**, leading to better downstream answers from the LLM, particularly for queries that are ambiguous, use synonyms, or require contextual understanding[5].

-----

-----

</details>

<details>
<summary>What is the role of a re-ranker, such as a cross-encoder model, in a RAG pipeline, and what are the performance trade-offs compared to relying solely on initial retrieval scores?</summary>

### Source [36]: https://adasci.org/a-hands-on-guide-to-enhance-rag-with-re-ranking/

Query: What is the role of a re-ranker, such as a cross-encoder model, in a RAG pipeline, and what are the performance trade-offs compared to relying solely on initial retrieval scores?

Answer: A **re-ranker** such as a cross-encoder model plays a vital role in a RAG (Retrieval-Augmented Generation) pipeline by significantly enhancing the quality of search results through prioritizing the most relevant documents. Initially, RAG retrieves a broad set of candidate responses based on the input query. The re-ranker (e.g., a cross-encoder) then ranks these candidates by scoring their relevance, ensuring that the highest-priority and most contextually appropriate documents are presented to the language model for answer generation. This two-stage approach increases the accuracy and dependability of the final output by reducing irrelevant or hallucinated responses. The main trade-off is computational: adding a re-ranker step introduces additional processing time and resource requirements, so choosing an optimal re-ranking model is necessary to balance improved answer quality with system efficiency.

-----

-----

-----

### Source [37]: https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/

Query: What is the role of a re-ranker, such as a cross-encoder model, in a RAG pipeline, and what are the performance trade-offs compared to relying solely on initial retrieval scores?

Answer: In a RAG pipeline, integrating re-ranking means augmenting the pipeline so that the most relevant chunks retrieved are those used to generate the final response. The re-ranking component can use advanced language understanding capabilities (such as those in LLMs or cross-encoders) to score and select the most contextually relevant results from an initial pool of candidates. This ensures that the information provided to the language model is highly pertinent, which in turn improves the accuracy and informativeness of the generated answers. While this method enhances the relevance of responses, it does introduce extra computational complexity due to the need to evaluate each candidate's relevance with a more sophisticated model.

-----

-----

-----

### Source [38]: https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html

Query: What is the role of a re-ranker, such as a cross-encoder model, in a RAG pipeline, and what are the performance trade-offs compared to relying solely on initial retrieval scores?

Answer: The **cross-encoder re-ranker** is used after a fast initial retrieval step (often using a bi-encoder for efficiency) to substantially improve the final relevance of search results. The cross-encoder processes the query and each candidate document together, computing an attention-based relevance score (typically between 0 and 1). The key benefit is higher performance and improved quality: the cross-encoder examines the relationship between the query and candidate in detail, leading to better ranking accuracy. However, this approach is computationally expensive and not suitable for scoring millions of candidates. Therefore, it is standard to first retrieve a manageable set (e.g., 100) of candidates using the fast retriever and then re-rank them using the cross-encoder, selecting only the top results for presentation to the user or language model.

-----

-----

-----

### Source [39]: https://zilliz.com/learn/optimize-rag-with-rerankers-the-role-and-tradeoffs

Query: What is the role of a re-ranker, such as a cross-encoder model, in a RAG pipeline, and what are the performance trade-offs compared to relying solely on initial retrieval scores?

Answer: A reranker-enhanced RAG architecture uses a **two-stage retrieval system**: first, a vector database retrieves initial candidates using fast, inexpensive methods (e.g., Approximate Nearest Neighbor search). Next, a reranker (often a cross-encoder) deeply analyzes and prioritizes these candidates according to their relevance to the input query. This leads to more relevant and accurate answers for the user. The trade-off is significant: rerankers, especially cross-encoders, require running a deep neural network inference for each candidate, increasing latency from milliseconds (for the initial retrieval) to hundreds of milliseconds or even seconds (for reranking), depending on the number of candidates and system hardware. Organizations must balance the improved answer quality against these higher computational and latency costs.

-----

-----

-----

### Source [40]: https://www.pinecone.io/learn/series/rag/rerankers/

Query: What is the role of a re-ranker, such as a cross-encoder model, in a RAG pipeline, and what are the performance trade-offs compared to relying solely on initial retrieval scores?

Answer: **Reranking** in RAG pipelines dramatically improves recall and overall answer relevance. The process involves retrieving a larger pool of candidate documents (e.g., top 25) and using a reranker to reorder these candidates, selecting the most relevant ones (e.g., top 3) for the language model. The result is a significant increase in relevant information and a reduction in noise, which enhances answer quality for users. The performance trade-off is that reranking adds an extra computational step, increasing processing time and system resource consumption compared to relying solely on the initial retrieval scores, which are fast but less precise in terms of semantic relevance.

-----

</details>

<details>
<summary>How does GraphRAG leverage knowledge graphs to answer complex, multi-hop questions that standard vector-based RAG struggles with?</summary>

### Source [41]: https://www.zyphra.com/post/understanding-graph-based-rag-and-multi-hop-question-answering

Query: How does GraphRAG leverage knowledge graphs to answer complex, multi-hop questions that standard vector-based RAG struggles with?

Answer: Graph-based RAG systems address the limitations of vector-based RAG in answering **multi-hop questions** by leveraging the structure and relationships inherent in knowledge graphs. While standard vector-based RAG retrieves chunks of text based primarily on similarity to the question, it struggles with questions that require reasoning across multiple entities or steps—known as multi-hop reasoning. In such cases, a chain of retrievals is necessary, where each step depends on the result of the previous one. Knowledge graphs provide explicit links between entities, allowing graph-based RAG to sequentially traverse these relationships. For example, to answer "Where did the most decorated Olympian of all time get their undergraduate degree?", a graph-based RAG system would first identify the entity 'most decorated Olympian of all time', find the relationship to 'Michael Phelps', and then retrieve information about his undergraduate degree. This structured traversal enables **reliable answers to complex, multi-hop questions** requiring intermediate reasoning, which vector-based approaches often fail to resolve due to lack of explicit entity connectivity[1].

-----

-----

-----

### Source [42]: https://arxiv.org/html/2506.19967v1

Query: How does GraphRAG leverage knowledge graphs to answer complex, multi-hop questions that standard vector-based RAG struggles with?

Answer: Inference Scaled GraphRAG enhances multi-hop reasoning over knowledge graphs by applying **inference-time scaling**, improving performance on complex, multi-hop question answering tasks. Evaluations on the GRBench benchmark demonstrate its effectiveness, particularly for medium and hard questions that require traversing several connected entities or synthesizing multiple paths within the graph. The approach systematically retrieves and reasons over nodes and edges, enabling answers that require deep traversal and inductive or abstract reasoning. Inference-time scaling further boosts the system’s ability to handle both shallow and deep reasoning tasks by dynamically adjusting the retrieval and reasoning process based on question complexity. This leads to **improved accuracy and adaptability** in answering questions that standard vector-based RAG systems find challenging due to their limited ability to follow multi-step relational chains in structured data[2].

-----

-----

-----

### Source [43]: https://aclanthology.org/2024.icnlsp-1.45.pdf

Query: How does GraphRAG leverage knowledge graphs to answer complex, multi-hop questions that standard vector-based RAG struggles with?

Answer: SG-RAG is a **zero-shot GraphRAG method** that leverages knowledge graphs for multi-hop question answering by retrieving relevant subgraphs and using them as context for large language models (LLMs). The process involves two main steps: subgraph retrieval and response generation. For a given question, SG-RAG formulates a Cypher query to extract a subgraph containing all necessary entities and relationships from the knowledge graph. This subgraph is then transformed into a set of textual triplets, which are included in the prompt to the LLM. This approach allows the LLM to reason over multiple connected facts, effectively handling multi-hop questions by presenting all relevant relational information explicitly. By using structured graph queries and providing rich, interconnected context, SG-RAG enables LLMs to answer questions that would otherwise require multiple, sequential retrieval steps, thus overcoming the limitations of standard vector-based RAG[3].

-----

-----

-----

### Source [44]: https://neo4j.com/blog/genai/knowledge-graph-llm-multi-hop-reasoning/

Query: How does GraphRAG leverage knowledge graphs to answer complex, multi-hop questions that standard vector-based RAG struggles with?

Answer: GraphRAG combines RAG applications with knowledge graphs to improve the ability to answer **complex, multi-part questions**. Unlike vector-based RAG, which often fails to retrieve all necessary information for multi-hop reasoning, GraphRAG utilizes the networked structure of knowledge graphs—nodes representing entities and edges representing relationships—to efficiently traverse from one piece of information to another. This structure enables RAG applications to "connect the dots" across several related facts, making it possible to answer questions that require reasoning over multiple entities and relationships. The process involves modeling the domain to define entities and relationships, constructing the graph by importing or extracting structured data, and iteratively refining the graph. This ensures that the RAG system can access and traverse relevant data paths for complex questions, directly addressing the weaknesses of vector-based retrieval in multi-hop scenarios[4].

-----

-----

-----

### Source [45]: https://arxiv.org/abs/2506.19967

Query: How does GraphRAG leverage knowledge graphs to answer complex, multi-hop questions that standard vector-based RAG struggles with?

Answer: Inference Scaled GraphRAG is designed to improve multi-hop question answering on knowledge graphs by enhancing reasoning capabilities during inference. It demonstrates superior performance on benchmark datasets that explicitly test multi-hop reasoning, outperforming both traditional GraphRAG and previous graph traversal approaches. The framework systematically retrieves and processes connected graph entities, enabling answers to questions that require deep, multi-step reasoning across the graph’s structure. By leveraging inference-time scaling, the system adapts its reasoning depth to match the complexity of the question, ensuring robustness and accuracy in scenarios where standard vector-based RAG methods typically struggle[5].

-----

-----

-----

### Source [61]: https://www.zyphra.com/post/understanding-graph-based-rag-and-multi-hop-question-answering

Query: How does GraphRAG leverage knowledge graphs to answer complex, multi-hop questions that standard vector-based RAG struggles with?

Answer: Graph-based RAG systems are particularly effective at answering **multi-hop questions**, which require a sequence of reasoning steps rather than retrieval of a single fact. In standard vector-based RAG, retrieval is based on semantic similarity, which often fails when the answer involves connecting multiple, indirectly related facts. For example, a question like "Where did the most decorated Olympian of all time get their undergraduate degree?" requires first identifying the person (Michael Phelps), and then retrieving his educational background. Graph-based RAG leverages the **explicit connections in a knowledge graph**, enabling sequential retrieval steps—first retrieving the relevant entity through its relationships, and then following further edges for additional facts. This approach supports **sequential, multi-step reasoning** by traversing paths in the knowledge graph, making it superior to standard vector search for questions that require combining facts across multiple entities or relationships.

-----

-----

-----

### Source [62]: https://arxiv.org/html/2506.19967v1

Query: How does GraphRAG leverage knowledge graphs to answer complex, multi-hop questions that standard vector-based RAG struggles with?

Answer: Inference Scaled GraphRAG is designed to enhance **multi-hop question answering** over knowledge graphs by applying inference-time scaling methods, directly addressing the challenges faced by standard RAG approaches. The framework is evaluated on GRBench—a benchmark specifically crafted for multi-hop reasoning over domain-specific knowledge graphs. Questions are categorized by reasoning complexity: 
- "Easy" questions require only single-hop (direct) retrieval.
- "Medium" and "Hard" questions require reasoning across multiple connected entities or synthesizing information from multiple paths.

Inference Scaled GraphRAG demonstrates improved performance, particularly for questions requiring **multiple hops**—that is, questions where the answer is not directly connected to the query node but is accessible via a path through the graph. The knowledge graph structure, with its nodes and edges, allows the system to **systematically traverse relationships** and aggregate the necessary information step by step, which standard vector-based approaches cannot do reliably.

-----

-----

-----

### Source [63]: https://aclanthology.org/2024.icnlsp-1.45.pdf

Query: How does GraphRAG leverage knowledge graphs to answer complex, multi-hop questions that standard vector-based RAG struggles with?

Answer: SG-RAG (SubGraph Retrieval Augmented Generation) is a **zero-shot Graph RAG method** that enables LLMs to answer multi-hop questions using knowledge graphs. Its process is twofold:
- **Subgraph Retrieval:** The system constructs a Cypher query (Neo4j's query language) based on the input question, which is then executed against the knowledge graph. This query retrieves **subgraphs** containing all relevant nodes and relationships needed to answer the multi-hop question.
- **Response Generation:** The retrieved subgraph is transformed into a set of triplets (subject-predicate-object), which are then formatted into a textual prompt for the LLM. The LLM uses this context to generate an answer.

By retrieving entire **subgraphs** instead of isolated nodes, SG-RAG ensures that all intermediate entities and relationships required for multi-step reasoning are present in the LLM’s context. This explicit, structured context enables the LLM to perform **reasoning across multiple hops**—something that vector-based retrieval, which relies solely on semantic similarity, is not equipped to do.

-----

-----

-----

### Source [64]: https://neo4j.com/blog/genai/knowledge-graph-llm-multi-hop-reasoning/

Query: How does GraphRAG leverage knowledge graphs to answer complex, multi-hop questions that standard vector-based RAG struggles with?

Answer: Knowledge graphs underpin GraphRAG’s ability to answer **complex, multi-part questions** by modeling data as a network of nodes (entities) and edges (relationships). RAG applications typically struggle with such questions because vector-based retrieval does not efficiently connect disparate pieces of information. By contrast, knowledge graphs allow RAG systems to **navigate from one fact to the next** using explicit relationships, efficiently aggregating all required data points.

To leverage this, GraphRAG involves:
- **Modeling the domain** by defining entities and their relationships.
- **Constructing the knowledge graph** from structured and unstructured data sources.

When a multi-hop question is posed, GraphRAG queries the graph to follow the necessary relationships—retrieving sequences of connected information that collectively answer the question. This connected data structure is what gives GraphRAG its edge over standard RAG, which cannot easily "chain" together multiple facts required for complex reasoning.

-----

-----

-----

### Source [65]: https://arxiv.org/abs/2506.19967

Query: How does GraphRAG leverage knowledge graphs to answer complex, multi-hop questions that standard vector-based RAG struggles with?

Answer: Inference Scaled GraphRAG is presented as a solution to **multi-hop question answering on knowledge graphs**, which standard vector-based RAG fails to solve consistently. This approach leverages the graph’s structured relationships to traverse multiple hops at inference time, aggregating evidence from different parts of the graph. Experimental results show that this method outperforms both traditional GraphRAG and previous graph traversal techniques, especially on questions that require combining information from several connected entities or synthesizing abstract patterns across the graph. This demonstrates the importance of **explicit graph traversal** and structured retrieval, which are inherently lacking in vector-based systems.

-----

-----

### Source [91]: https://www.zyphra.com/post/understanding-graph-based-rag-and-multi-hop-question-answering

Query: How does GraphRAG leverage knowledge graphs to answer complex, multi-hop questions that standard vector-based RAG struggles with?

Answer: Graph-based RAG systems are particularly advantageous for **multi-hop question answering** because they align with the need for sequential, chain-like retrievals that multi-hop questions demand. In standard vector-based RAG, retrieval is typically limited to single-hop reasoning: the system finds passages most similar to the query but may miss relevant intermediate connections. For questions that require multiple inferential steps—such as first identifying an entity and then retrieving a property about that entity—vector-based retrieval often fails, since all the information may not be present in any single chunk.

Graph-based RAG leverages the explicit structure of **knowledge graphs**: entities (nodes) and relationships (edges) allow the system to perform stepwise traversals. For a multi-hop question, the system can:
- First, retrieve relevant nodes (e.g., "most decorated Olympian of all time" → Michael Phelps).
- Then, traverse to connected nodes/attributes (e.g., Michael Phelps → undergraduate degree).

Each "hop" corresponds to a traversal along the graph, allowing for **sequential reasoning** that reflects the intermediate logic needed. This structured, stepwise retrieval process enables GraphRAG to answer complex questions that would be difficult or unreliable for standard vector-based RAG, which lacks awareness of explicit entity relationships and thus cannot chain together facts over multiple steps.

-----

-----

-----

### Source [92]: https://arxiv.org/html/2506.19967v1

Query: How does GraphRAG leverage knowledge graphs to answer complex, multi-hop questions that standard vector-based RAG struggles with?

Answer: Inference Scaled GraphRAG, an advanced framework, enhances **multi-hop reasoning over knowledge graphs** by applying inference-time scaling. The approach is evaluated on GRBench, a benchmark specifically designed for multi-hop question answering over knowledge graphs, which categorizes questions by the complexity of graph traversal needed:
- Easy: one-hop queries
- Medium: multi-hop reasoning across connected entities
- Hard: abstract reasoning requiring synthesis across multiple graph paths

The method demonstrates significant improvements, particularly for questions requiring multiple hops—where retrieving and reasoning over **chains of nodes and edges** is essential. Unlike traditional vector-based RAG, which may retrieve isolated chunks of text, GraphRAG leverages the graph’s structure to traverse connections and aggregate information over several steps, directly reflecting the logic needed for multi-hop inference.

-----

-----

-----

### Source [93]: https://aclanthology.org/2024.icnlsp-1.45.pdf

Query: How does GraphRAG leverage knowledge graphs to answer complex, multi-hop questions that standard vector-based RAG struggles with?

Answer: SG-RAG (SubGraph Retrieval Augmented Generation) is a **zero-shot GraphRAG method** for answering domain-specific multihop questions using a knowledge graph (KG) and an LLM. Its operation involves two key steps:
- **Subgraph Retrieval:** The system queries the knowledge graph using a Cypher statement that represents the natural language question. Instead of retrieving isolated pieces of information, it fetches subgraphs that contain all the nodes and relationships necessary to answer the question.
- **Response Generation:** The retrieved subgraphs are converted into a set of textual triplets and presented as context within a prompt for the LLM. The LLM then generates an answer using both the question and the structured, multi-hop context.

This approach allows SG-RAG to explicitly collect and chain together the pieces of information needed for multi-hop reasoning—something vector-based retrieval cannot do because it lacks the ability to traverse relationships or retrieve structured, intermediate steps.

-----

-----

-----

### Source [94]: https://neo4j.com/blog/genai/knowledge-graph-llm-multi-hop-reasoning/

Query: How does GraphRAG leverage knowledge graphs to answer complex, multi-hop questions that standard vector-based RAG struggles with?

Answer: Knowledge graphs underpin GraphRAG’s ability to **answer complex, multi-part questions** by representing information as interconnected nodes (entities) and edges (relationships). This networked data structure allows RAG applications to:
- Efficiently **navigate from one piece of information to another**, following relationships to gather all connected facts necessary for multi-hop reasoning.
- Model domains by defining explicit entity types and relationships, enabling structured traversal and retrieval far beyond what vector search can provide.

Because of this, GraphRAG can retrieve and synthesize information across several hops, connecting data points that would remain unlinked in a standard vector database. This is especially valuable for questions where the answer is not contained in a single context but distributed across related entities—requiring the model to "connect the dots" step by step.

-----

-----

-----

### Source [95]: https://arxiv.org/abs/2506.19967

Query: How does GraphRAG leverage knowledge graphs to answer complex, multi-hop questions that standard vector-based RAG struggles with?

Answer: Inference Scaled GraphRAG improves **multi-hop question answering on knowledge graphs** by leveraging the graph’s structure to perform sequential, inference-time reasoning. The approach is benchmarked on datasets specifically designed to challenge multi-hop capabilities, showing that the method yields better performance as the number of hops required increases. The explicit modeling of entity relationships and the ability to traverse multiple connected nodes are key factors in GraphRAG’s superior handling of complex, multi-hop questions, compared to standard vector-based RAG approaches.

-----

</details>

<details>
<summary>What are the core architectural and behavioral distinctions between a standard, linear RAG pipeline and an "Agentic RAG" system that uses a ReAct-style agent?</summary>

### Source [46]: https://weaviate.io/blog/what-is-agentic-rag

Query: What are the core architectural and behavioral distinctions between a standard, linear RAG pipeline and an "Agentic RAG" system that uses a ReAct-style agent?

Answer: A **standard, linear RAG pipeline** operates sequentially: a user query is passed to a retriever (often a vector database), which fetches relevant documents, and these are then passed to a language model for answer generation. There is no internal decision-making or adaptive routing; the pipeline is fixed.

An **Agentic RAG system**, by contrast, is centered on one or more agents that can reason about which external knowledge sources or tools to consult. In its simplest form (single-agent), the agent acts as a router, choosing among multiple knowledge sources (not just vector databases but also APIs, web search, or internal tools) based on the query. More advanced, multi-agent architectures can involve a "master" agent coordinating specialized retrieval agents, each targeting different data domains (e.g., internal docs, emails, web, etc.). Agents may also be used for non-retrieval tasks, such as validation or synthesis.

Architecturally, this means Agentic RAG introduces:
- **Decision-making logic**: Agents use reasoning to select sources and actions.
- **Tool and data source flexibility**: Agents can access a broader range of external tools and APIs.
- **Multi-agent collaboration**: Multiple agents can work together, each specializing in certain tasks or data sources.

Behaviorally, Agentic RAG enables dynamic, context-aware retrieval, adaptation, and orchestration, rather than rigid, single-path information flow.

-----

-----

-----

### Source [47]: https://empathyfirstmedia.com/building-multi-agent-rag-systems-step-by-step-implementation-guide/

Query: What are the core architectural and behavioral distinctions between a standard, linear RAG pipeline and an "Agentic RAG" system that uses a ReAct-style agent?

Answer: In standard RAG, the pipeline is generally linear: a single query triggers document retrieval and then generation, with no intermediate reasoning or inter-agent collaboration.

In a multi-agent (Agentic RAG) system, you explicitly define roles: e.g., a master orchestrator, retrieval agents for various sources (knowledge base, web, database), a validation agent for fact-checking, and a synthesis agent for final response generation. Each agent is implemented as a modular component (typically a class), capable of communicating with other agents and maintaining its own memory or context.

Key architectural distinctions include:
- **Agent modularity**: Each agent inherits from a common base, enabling extensibility and specialization.
- **Inter-agent communication**: Agents can message each other, share intermediate results, and collaborate.
- **Task orchestration**: A top-level orchestrator coordinates the workflow, delegating subtasks to the appropriate agents.

Behaviorally, this architecture supports:
- **Iterative reasoning**: Agents can break down tasks, ask follow-up questions, and validate information.
- **Dynamic task allocation**: The system can adaptively assign subtasks based on query complexity and required data sources.

-----

-----

-----

### Source [48]: https://www.eyelevel.ai/post/agentic-rag

Query: What are the core architectural and behavioral distinctions between a standard, linear RAG pipeline and an "Agentic RAG" system that uses a ReAct-style agent?

Answer: The **ReAct architecture** (Reasoning + Acting) is fundamental to Agentic RAG systems. In ReAct-style agentic RAG, instead of a simple retrieve-then-generate flow, the agent alternates between internal reasoning steps and external actions (retrievals, tool calls, etc.).

Architectural differences:
- **Structured prompt design**: The agent is prompted to explicitly reason about the query (e.g., break it into sections, plan actions).
- **Action-interleaving**: The agent can perform multiple, context-driven retrievals and tool calls in sequence, refining its approach based on intermediate findings.
- **Autonomy**: The agent can decide not just what to retrieve, but how to process, validate, and synthesize information.

Behaviorally, a ReAct-style Agentic RAG system:
- Engages in multi-step, context-aware reasoning.
- Adapts its retrieval and response generation dynamically, based on ongoing analysis of the task.
- Provides more structured, explainable outputs by exposing its reasoning process alongside its actions.

-----

-----

-----

### Source [49]: https://www.digitalocean.com/community/conceptual-articles/rag-ai-agents-agentic-rag-comparative-analysis

Query: What are the core architectural and behavioral distinctions between a standard, linear RAG pipeline and an "Agentic RAG" system that uses a ReAct-style agent?

Answer: Agentic RAG systems address the limitations of static, linear RAG by supporting **multi-step, adaptive, and autonomous problem-solving**.

Core distinctions:
- **Standard RAG**: Designed for one-shot, factually grounded answers. The retriever module (sparse or neural) finds relevant documents, and the generator produces a response—no iterative adaptation or multi-step decision-making.
- **Agentic RAG**: Embeds an agent controller that can plan, analyze retrieved information, assess its value, and determine next actions—creating a feedback loop for continuous improvement.

Architecturally, Agentic RAG incorporates:
- **Agent controllers**: Oversee workflow, plan actions, and adapt strategies in real-time.
- **Iterative modules**: Retrieval and generation can occur multiple times in a loop, with each cycle informed by prior context.

Behaviorally, Agentic RAG systems:
- Maintain ongoing context, adapt responses as new information is gathered, and can execute real-time actions (e.g., triggering API calls, updating knowledge).
- Excel in complex, iterative, or multi-turn tasks where static RAG pipelines struggle.

-----

-----

-----

### Source [50]: https://www.k2view.com/blog/react-agent-llm/

Query: What are the core architectural and behavioral distinctions between a standard, linear RAG pipeline and an "Agentic RAG" system that uses a ReAct-style agent?

Answer: A **ReAct agent LLM** combines reasoning with action, using a RAG architecture to access external, often trusted, data beyond its training corpus.

Architectural distinctions:
- **Integrated reasoning and retrieval**: The agent alternates between "thought" (reasoning about the task, planning next steps) and "action" (retrieving documents, querying APIs).
- **Dynamic task decomposition**: The agent can break complex queries into manageable subtasks, each potentially triggering new retrievals or tool calls.
- **Flexible data access**: Beyond static document retrieval, the agent can fetch structured and unstructured data, and interact with enterprise systems.

Behaviorally, ReAct agent LLMs in an Agentic RAG framework:
- Iteratively refine their approach, using new information to inform subsequent reasoning and retrievals.
- Deliver more reliable, up-to-date, and context-aware outputs compared to linear RAG, which lacks such iterative refinement.
- Are well-suited for tasks requiring both critical thinking and the integration of current, external knowledge.

-----

-----

### Source [66]: https://weaviate.io/blog/what-is-agentic-rag

Query: What are the core architectural and behavioral distinctions between a standard, linear RAG pipeline and an "Agentic RAG" system that uses a ReAct-style agent?

Answer: A **standard, linear RAG pipeline** operates sequentially: it retrieves information from a predefined external knowledge source (often a vector database), passes this to a language model for generation, and outputs the result. In contrast, **Agentic RAG** introduces an agent (or multiple agents) as the core orchestrator. 

- In **single-agent RAG**, the agent acts as a router, selecting among multiple external knowledge sources or tools (such as databases, web search APIs, or enterprise systems) for context retrieval. This agent is responsible for reasoning about which source best fits the query before retrieving information.
- **Multi-agent RAG systems** further expand this architecture by chaining agents with specialized roles: a master agent coordinates, while retrieval agents target specific data sources (e.g., proprietary databases, email, public web). Agents can also serve purposes beyond retrieval, such as synthesis or validation.
- Implementation can use either a language model with function calling or an agent framework, offering different levels of control and flexibility.

The behavioral distinction is the agent’s autonomy in reasoning, selecting sources, and orchestrating multi-step processes, versus the linear, fixed steps of standard RAG.

-----

-----

-----

### Source [67]: https://empathyfirstmedia.com/building-multi-agent-rag-systems-step-by-step-implementation-guide/

Query: What are the core architectural and behavioral distinctions between a standard, linear RAG pipeline and an "Agentic RAG" system that uses a ReAct-style agent?

Answer: A **multi-agent RAG system** is architected around specialized agents with defined roles, coordinated by a master orchestrator. 

- The architecture is mapped out with agent roles such as Document Retrieval Agent (internal knowledge), Web Search Agent (market data), Database Query Agent (CRM/sales), Validation Agent (fact-checking), and Synthesis Agent (response generation).
- Each agent is built from a base class, supporting asynchronous processing and inter-agent communication. This enables collaboration and division of labor among agents, contrasting with the single-pass behavior of linear RAG.
- The agents communicate to pass tasks and results, allowing iterative refinement and validation, which is absent in standard RAG.
- The deployment environment incorporates frameworks and libraries for retrieval, orchestration, and API exposure, supporting the more complex, dynamic workflows required for agentic RAG systems.

-----

-----

-----

### Source [68]: https://www.eyelevel.ai/post/agentic-rag

Query: What are the core architectural and behavioral distinctions between a standard, linear RAG pipeline and an "Agentic RAG" system that uses a ReAct-style agent?

Answer: The **ReAct architecture** (Reasoning and Act) forms the backbone of many Agentic RAG systems. Unlike linear RAG, ReAct-style agents are instructed to process a question by decomposing it into sections and explicitly reasoning about each part.

- The system alternates between reasoning steps ("think") and actionable steps ("act"), iteratively gathering information, evaluating, and synthesizing answers.
- This explicit, multi-step reasoning and action cycle enables the agent to handle complex queries, adapt to evolving contexts, and provide detailed, context-aware responses.
- In a linear RAG pipeline, by contrast, the process is rigid: retrieve relevant documents, generate an answer, and stop. ReAct-style agents continue reasoning and acting until they reach a satisfactory answer.

-----

-----

-----

### Source [69]: https://www.digitalocean.com/community/conceptual-articles/rag-ai-agents-agentic-rag-comparative-analysis

Query: What are the core architectural and behavioral distinctions between a standard, linear RAG pipeline and an "Agentic RAG" system that uses a ReAct-style agent?

Answer: **Agentic RAG** advances RAG from static, single-turn interactions to dynamic, multi-step tasks managed by autonomous agents.

- Standard RAG focuses on factual grounding: it retrieves documents and generates a response in a single, linear process.
- Agentic RAG introduces agents with planning and adaptability, enabling iterative decision-making, evaluation, and action. The agent analyzes, selects information, determines responses, and may execute actions in a feedback loop.
- The architecture relies on effective retriever modules (sparse or neural), generator models, and adaptive agent controllers—components which are orchestrated rather than sequentially chained.
- Use cases requiring contextual awareness and real-time action (robotics, legal, healthcare, customer service) benefit from agentic RAG, as mere retrieval is insufficient for complex tasks.

-----

-----

-----

### Source [70]: https://www.k2view.com/blog/react-agent-llm/

Query: What are the core architectural and behavioral distinctions between a standard, linear RAG pipeline and an "Agentic RAG" system that uses a ReAct-style agent?

Answer: A **ReAct agent LLM** exemplifies Agentic RAG by combining reasoning and action.

- Unlike traditional LLMs or standard RAG, which passively retrieve and generate, a ReAct agent LLM actively plans, interacts, and iterates through task steps.
- The agent may break complex tasks into smaller parts, search various sources (enterprise systems, online, knowledge bases), and alternate between "thought" (reasoning) and "action" (retrieval, task execution).
- This cycle continues until it arrives at a well-supported answer. The process is dynamic and responsive, reducing errors and supporting critical thinking with up-to-date information.
- Agentic RAG, via ReAct agents, delivers more reliable, context-aware, and understandable responses, especially for complex, multi-faceted queries.

-----

-----

</details>

<details>
<summary>What are the key architectural components and data flows in a production-ready RAG system, from offline ingestion to online retrieval?</summary>

### Source [51]: https://galileo.ai/blog/rag-architecture

Query: What are the key architectural components and data flows in a production-ready RAG system, from offline ingestion to online retrieval?

Answer: A production-ready Retrieval-Augmented Generation (RAG) system consists of two central architectural components: 

- **Retriever Component**: This module fetches relevant information from a predefined knowledge base, which may include documentation, enterprise data, or real-time information. The retriever uses various retrieval technologies to ensure the information used by downstream systems is both up-to-date and accurate.

- **Generation Component**: This module consumes the information retrieved and generates human-like, contextually appropriate responses. It leverages advanced language models (LLMs), using the retrieved documents as context for content generation.

The data flow starts with the retriever, which ensures the generation component always has access to the most relevant information. The generation component then produces coherent responses, integrating both static (pre-ingested) and dynamic (real-time) knowledge. This tight integration is crucial for scaling reliable, enterprise-grade AI applications, as it bridges the gap between static model knowledge and the latest organizational data.[1]

-----

-----

-----

### Source [52]: https://python.langchain.com/docs/tutorials/rag/

Query: What are the key architectural components and data flows in a production-ready RAG system, from offline ingestion to online retrieval?

Answer: A typical RAG system is divided into two main phases:

- **Indexing (Offline Ingestion)**:
    - **Load**: Data is loaded from sources using document loaders.
    - **Split**: Large documents are split into smaller, manageable chunks, which are easier to index and search, and fit into the context window of LLMs.
    - **Store**: These chunks are stored and indexed, commonly in a vector store, after embedding them using an embeddings model.

- **Retrieval and Generation (Online Retrieval)**:
    - **Retrieve**: At query time, the system uses a retriever to fetch relevant chunks from the vector store based on the user’s query.
    - **Generate**: The language model then generates an answer, using a prompt that includes both the question and the retrieved data.

This structure ensures fast, relevant retrieval and coherent generation, with offline processes focusing on efficient indexing and online pipelines dedicated to low-latency, context-enhanced generation.[2]

-----

-----

-----

### Source [53]: https://developer.nvidia.com/blog/rag-101-demystifying-retrieval-augmented-generation-pipelines/

Query: What are the key architectural components and data flows in a production-ready RAG system, from offline ingestion to online retrieval?

Answer: The RAG pipeline is split into two major phases:

- **Offline (Document Ingestion)**:
    - Raw data is ingested from various sources (databases, documents, live feeds).
    - Data is pre-processed and embedded—meaning it is transformed into vector representations for efficient similarity search.
    - Data is loaded using document loaders, which can handle diverse formats (PDFs, CSVs, emails, etc.).

- **Online (Inference Pipeline)**:
    - When a user query arrives, the retrieval system searches the embedded/document index for relevant information.
    - Retrieved documents are then fed to a generative LLM, which produces a user-facing response.

Each phase can be structured as microservices, allowing for modularity and scalability. The decoupling of ingestion and inference supports continuous updates to the knowledge base without disrupting online query processing.[3]

-----

-----

-----

### Source [54]: https://www.ibm.com/think/topics/retrieval-augmented-generation

Query: What are the key architectural components and data flows in a production-ready RAG system, from offline ingestion to online retrieval?

Answer: A production RAG system comprises four primary components and follows a five-stage data flow:

- **Components**:
    - **Knowledge Base**: A repository containing external, often unstructured, data (PDFs, guides, websites, audio files, etc.).
    - **Retrieval Model**: Queries the knowledge base for relevant information.
    - **Integration Layer**: Receives retrieved information and constructs an augmented prompt.
    - **Generative Model**: Produces the final output using the augmented prompt.

- **Data Flow**:
    1. User submits a prompt.
    2. Retrieval model queries the knowledge base.
    3. Relevant information is returned to the integration layer.
    4. Augmented prompt (with context) is sent to the LLM.
    5. LLM generates and returns output to the user.

Additional components can include a **ranker** (to order retrieved results by relevance) and an **output handler** (for formatting responses). The process starts with building a queryable knowledge base from diverse, often unstructured, data sources.[4]

-----

-----

-----

### Source [55]: https://docs.spring.io/spring-ai/reference/api/retrieval-augmented-generation.html

Query: What are the key architectural components and data flows in a production-ready RAG system, from offline ingestion to online retrieval?

Answer: Spring AI provides a modular approach to RAG systems, supporting custom or out-of-the-box flows via the `RetrievalAugmentationAdvisor`. The architecture includes:

- **Query Transformers**: Rewrite or modify user queries for optimal retrieval.
- **Document Retriever**: Uses a vector store and similarity thresholds to fetch relevant documents.
- **Document Post-Processor**: Allows post-processing of retrieved documents (re-ranking, redundancy removal, content compression) before they are passed to the LLM.
- **Chat Client**: Orchestrates user question handling, advisor application, and content retrieval/generation.

This modular architecture enables customization and reconfiguration of each pipeline component. For example, you can swap retrieval strategies, add post-processing, or replace the generation model without disrupting the entire system. This flexibility is critical for production systems needing to adapt to evolving requirements and data sources.[5]
-----

-----

</details>

<details>
<summary>What are the most effective document chunking strategies (e.g., fixed-size, recursive, semantic) for different types of unstructured data, and what are their trade-offs?</summary>

### Source [56]: https://www.f22labs.com/blogs/7-chunking-strategies-in-rag-you-need-to-know/

Query: What are the most effective document chunking strategies (e.g., fixed-size, recursive, semantic) for different types of unstructured data, and what are their trade-offs?

Answer: **Fixed-size chunking** divides documents based on a set number of tokens, words, or characters. This approach is simple and computationally efficient, but may split contextually related information across chunk boundaries, which can degrade downstream performance for tasks requiring deep understanding of text.

**Recursive-based chunking** splits text by hierarchical boundaries—such as paragraphs, sentences, or sub-sections—recursively applying chunking rules to preserve more context. This method retains logical structure better than fixed-size chunking but can still break semantic units if document structure is inconsistent.

**Document-based chunking** treats the entire document as one chunk or divides it minimally, preserving full context and structure. It is best for legal, medical, or scientific documents where maintaining context is critical. However, this can result in very large chunks, which may be inefficient for retrieval or processing.

**Semantic chunking** splits text at points where semantic shifts occur, often detected via sentence embeddings. Each chunk is designed to be internally coherent and thematically unified. This approach is ideal for data where meaning and context must be preserved—such as narrative text or topic-diverse documents. The trade-off is increased computational cost and complexity, especially in setting optimal thresholds for chunking.

**Token-based chunking** uses a fixed number of tokens per chunk. It is fast and suitable for uniform data, but risks splitting sentences or concepts, leading to loss of meaning in some cases.

Each strategy has trade-offs related to context preservation, chunk size consistency, computational complexity, and suitability for specific document types.

-----

-----

-----

### Source [57]: https://www.pinecone.io/learn/chunking-strategies/

Query: What are the most effective document chunking strategies (e.g., fixed-size, recursive, semantic) for different types of unstructured data, and what are their trade-offs?

Answer: **Semantic chunking** involves grouping sentences based on surrounding semantic context. By analyzing sentence embeddings, semantically coherent groups are formed, which can then be used for more meaningful retrieval and downstream tasks. This method is particularly effective for unstructured text that does not have clear organizational boundaries, as it ensures each chunk is contextually relevant.

**Document structure-based chunking** leverages the inherent structure of complex documents (PDFs, DOCX, HTML, Markdown, LaTeX) to inform chunk boundaries. For example:
- **PDFs**: Chunking may be guided by headers, paragraphs, or tables, using preprocessing tools to identify structural elements.
- **HTML**: Tags like `<p>` and `<title>` are used to segment text, preserving logical divisions found in web content.
- **Markdown**: Syntax elements (headings, lists, code blocks) direct chunking, maintaining semantic and hierarchical organization.
- **LaTeX**: Chunking is based on commands and environments (sections, equations), preserving technical and academic document structure.

These structure-aware approaches maintain semantic integrity and logical flow, improving retrieval and comprehension for users interacting with complex, multi-format documents.

-----

-----

-----

### Source [58]: https://bitpeak.com/chunking-methods-in-rag-methods-comparison/

Query: What are the most effective document chunking strategies (e.g., fixed-size, recursive, semantic) for different types of unstructured data, and what are their trade-offs?

Answer: A comparative study of **semantic chunking methods** (percentile-based, double-pass merging, proposition-based) shows marked differences in effectiveness and computational requirements:
- **Clustering-based chunking** is unsuitable for texts with distinct topic segments; it loses sentence order and thematic coherence.
- **Classical methods** (sentence-based, token-based) struggle to divide texts with uneven segment lengths or mixed topics, sometimes splitting related information.
- **Classical semantic chunking** (using semantic breakpoints) performs better but is not perfect, occasionally splitting coherent units.
- **Semantic double-pass merging chunking** and **proposition-based chunking** (using models like GPT-4) produce thematically coherent chunks and handle complex texts well.
- For texts with code or pseudocode, percentile-based semantic chunking may fragment logical units, while double-pass and proposition-based methods excel.

**Performance metrics**:
- Longest, most coherent chunks: double-pass merging and proposition-based methods.
- Fastest: classical token-based and sentence-based chunking, due to lower computational overhead.
- Most computationally expensive: proposition-based chunking (especially with large language models like GPT-4), which can also be costly in terms of API usage.

Trade-offs center on chunk coherence versus speed and cost: advanced semantic methods yield better chunks for complex data but require more resources.

-----

-----

-----

### Source [59]: https://infohub.delltechnologies.com/es-es/p/chunk-twice-retrieve-once-rag-chunking-strategies-optimized-for-different-content-types/

Query: What are the most effective document chunking strategies (e.g., fixed-size, recursive, semantic) for different types of unstructured data, and what are their trade-offs?

Answer: **Multi-modal chunking** is recommended for complex documents such as PDFs containing text, tables, and images. This strategy separates each content type for specialized handling:
- **Text blocks**: Chunked by semantic boundaries to preserve narrative flow.
- **Tables**: Converted to structured markdown, ensuring relational data integrity.
- **Images**: Extracted and encoded (e.g., base64) with metadata retaining page and positional information.

Each chunk is enriched with metadata (page number, bounding box, content type), allowing retrieval systems to access both content and original presentation context. This method ensures:
- Tabular data remains queryable.
- Images retain visual context.
- Text maintains logical order and meaning.

Multi-modal chunking is optimal for documents with heterogeneous content types and complex structure, enhancing retrieval accuracy and relevance.

-----

-----

-----

### Source [60]: https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/

Query: What are the most effective document chunking strategies (e.g., fixed-size, recursive, semantic) for different types of unstructured data, and what are their trade-offs?

Answer: **Page-level chunking** is recommended as the default strategy for most document types, providing consistent performance and reliable citation/reference capabilities. Unlike token-based chunking—where references may shift with different chunk sizes—page boundaries are static, aiding reproducibility and traceability.

For refinement or specialized cases:
- **Financial documents**: Token-based chunking (512 or 1,024 tokens) or section-level chunking may outperform page-level chunking, especially for structured financial data.
- **Diverse documents**: Smaller token-sized chunks (256–512 tokens) work well for collections with high variability.
- **Query characteristics**: For factoid queries (seeking specific facts), page-level or smaller chunks (256–512 tokens) offer optimal retrieval performance.

The main trade-off is that page-level chunking may not be optimal for all datasets, but it provides high average accuracy and stability. Token-based chunking offers granularity but can fragment context, while section-level chunking can improve relevance for structured documents.

-----

</details>

<details>
<summary>What are the practical trade-offs and best practices when choosing a document chunking strategy, such as semantic, layout-aware, or context-enriched, for different types of unstructured data in RAG?</summary>

### Source [76]: https://www.f22labs.com/blogs/7-chunking-strategies-in-rag-you-need-to-know/

Query: What are the practical trade-offs and best practices when choosing a document chunking strategy, such as semantic, layout-aware, or context-enriched, for different types of unstructured data in RAG?

Answer: Chunking strategies like **fixed-size chunking** and **recursive character text splitting** each offer trade-offs for RAG. Fixed-size chunking is simple, splitting documents into uniform sizes (e.g., 500 tokens), but may divide sentences and disrupt meaning. To address this, chunk overlap is used, repeating tokens between chunks to preserve context, though this increases storage and compute costs. Recursive character text splitting is more adaptive, using a hierarchy of separators (e.g., paragraphs, sentences, code blocks) to create meaningful chunks based on the document’s logical structure, which improves semantic coherence but can be more computationally intensive to implement. The choice should be aligned with the document type and the downstream application: code, for example, benefits from logical splits by class or function, while prose may favor sentence or paragraph boundaries.

-----

-----

-----

### Source [77]: https://infohub.delltechnologies.com/es-es/p/chunk-twice-retrieve-once-rag-chunking-strategies-optimized-for-different-content-types/

Query: What are the practical trade-offs and best practices when choosing a document chunking strategy, such as semantic, layout-aware, or context-enriched, for different types of unstructured data in RAG?

Answer: The chunking strategy directly determines a RAG system’s retrieval accuracy and its ability to maintain semantic relationships. Poor chunking can scatter related concepts, harming context and retrieval. Thoughtful chunking aims to make each chunk the smallest unit of knowledge processable by an LLM, tailored to the content’s inherent organization. For example, chunking strategies should differ based on whether the document is prose, code, or tabular data—each requiring a nuanced approach to preserve structure and meaning. Smart chunking strategies maximize retrieval quality by keeping related information together, whereas generic or naive splits can fragment knowledge and reduce RAG system effectiveness.

-----

-----

-----

### Source [78]: https://www.ibm.com/think/tutorials/chunking-strategies-for-rag-with-langchain-watsonx-ai

Query: What are the practical trade-offs and best practices when choosing a document chunking strategy, such as semantic, layout-aware, or context-enriched, for different types of unstructured data in RAG?

Answer: Effective chunking breaks large text into smaller, meaningful chunks for embedding and retrieval in RAG. Smaller chunks generally outperform larger ones, especially for models with limited context windows, as they are more manageable and less likely to overwhelm the model. However, the most effective chunking technique depends on the LLM application’s specific needs. Common chunking processes include splitting by sentences, paragraphs, or using semantic boundaries. Selecting the right chunking strategy is crucial—too small chunks may lose context, while too large chunks can be inefficient and less precise during retrieval.

-----

-----

-----

### Source [79]: https://bitpeak.com/chunking-methods-in-rag-methods-comparison/

Query: What are the practical trade-offs and best practices when choosing a document chunking strategy, such as semantic, layout-aware, or context-enriched, for different types of unstructured data in RAG?

Answer: Comparative analysis shows that **semantic double-pass merging chunking** outperforms other methods in most metrics, as it preserves semantic relationships while optimizing chunk size. Traditional methods like **sentence-based** or **token-based chunking** are inexpensive but may not capture meaning as effectively for all document types. Proposition-based chunking performs poorly for long prose, being both inefficient and expensive. The optimal chunk length varies by text and use-case, so method selection should be data-driven. Enhanced semantic techniques, though costlier, yield better retrieval results by aligning chunks with meaningful content boundaries.

-----

-----

-----

### Source [80]: https://www.tigerdata.com/blog/which-rag-chunking-and-formatting-strategy-is-best

Query: What are the practical trade-offs and best practices when choosing a document chunking strategy, such as semantic, layout-aware, or context-enriched, for different types of unstructured data in RAG?

Answer: **Recursive character text splitting** can be tailored for different document types (e.g., HTML, Markdown, code) by configuring a hierarchy of split points, such as HTML articles, sections, paragraphs, and sentences. This preserves logical document structure and semantic relationships while accommodating various unstructured data formats. Tracking chunking and formatting strategies for each embedding supports systematic experimentation, A/B testing, and robust rollout management. This approach allows teams to refine chunking strategies based on observed performance, rather than relying on ad hoc or undocumented changes.

-----

-----

</details>

<details>
<summary>What are the key architectural patterns and implementation challenges when building a production-ready "Agentic RAG" system that can dynamically choose between retrieval and other tools?</summary>

### Source [82]: https://workativ.com/ai-agent/blog/agentic-rag

Query: What are the key architectural patterns and implementation challenges when building a production-ready "Agentic RAG" system that can dynamically choose between retrieval and other tools?

Answer: Key architectural patterns discussed:
- **Agent-based pipeline orchestration**: Embedding agents in the RAG pipeline, each responsible for retrieval, reasoning, or task execution.
- **Routing agents**: Use LLMs to analyze and route user queries to the relevant RAG pipeline, dynamically selecting tools or sources.
- **Query planning agents**: Decompose complex queries into sub-queries and assign them to specialized agents or pipelines, later merging the results for comprehensive responses.
- **ReAct and dynamic planning agents**: Integrate routing, planning, and tool use into a sequential, context-aware workflow, enabling multi-step and iterative actions based on real-time inputs.

Implementation challenges include:
- **Workflow complexity**: Handling multi-stage, sequential, and iterative workflows requires agents to coordinate and maintain context across steps.
- **Adaptability**: The system must dynamically adjust to changes in data, user needs, and available tools during execution.
- **Component integration**: Seamlessly connecting reasoning, routing, and execution agents is non-trivial and demands careful architectural planning.

-----

-----

-----

### Source [83]: https://www.moveworks.com/us/en/resources/blog/what-is-agentic-rag

Query: What are the key architectural patterns and implementation challenges when building a production-ready "Agentic RAG" system that can dynamically choose between retrieval and other tools?

Answer: Key implementation steps and challenges:
- **Initial assessment and planning**: Evaluate existing IT systems, define clear goals, and identify necessary data sources and tools.
- **Resource and team setup**: Assemble a skilled team and allocate resources for development, testing, and deployment.
- **Integration with legacy systems**: Develop a plan for smooth integration with current infrastructure, addressing compatibility and data format issues.

Implementation challenges include:
- **Data quality and curation**: The effectiveness of agentic RAG depends heavily on the quality, completeness, and relevance of underlying data.
- **Interpretability and explainability**: Agents’ decision-making must be transparent; models must explain their reasoning and data provenance to foster trust.
- **Privacy and security**: Robust data protection, access controls, and secure communication are essential to safeguard sensitive information.

-----

-----

-----

### Source [84]: https://aws.amazon.com/blogs/machine-learning/create-an-agentic-rag-application-for-advanced-knowledge-discovery-with-llamaindex-and-mistral-in-amazon-bedrock/

Query: What are the key architectural patterns and implementation challenges when building a production-ready "Agentic RAG" system that can dynamically choose between retrieval and other tools?

Answer: The AWS example architecture features:
- **Agent orchestration**: The AgentRunner orchestrates conversation history, manages tasks, and coordinates agent workflows, while AgentWorker performs stepwise reasoning and execution.
- **Modular retrieval integration**: The system uses LlamaIndex for document processing, chunking, and storage in an Amazon OpenSearch Serverless vector store, or alternatively leverages Amazon Bedrock Knowledge Bases for managed vector storage.
- **API tool integration**: Agents can dynamically invoke external APIs (e.g., GitHub, arXiv, DuckDuckGo) alongside internal RAG-backed document retrieval.
- **Reranking and relevance**: Incorporates retrieval reranking to improve the contextual relevance of responses.

Implementation challenges:
- **Component orchestration**: Managing state, history, and execution flow across agent components.
- **Flexible integration**: Supporting both custom pipelines (LlamaIndex) and managed solutions (Bedrock Knowledge Bases), with configuration tailored to developer expertise and infrastructure needs.
- **Scalability and real-time response**: Ensuring the system can handle concurrent queries and maintain low latency.

-----

-----

-----

### Source [85]: https://www.codiste.com/implementing-agentic-rag-enterprise-guide

Query: What are the key architectural patterns and implementation challenges when building a production-ready "Agentic RAG" system that can dynamically choose between retrieval and other tools?

Answer: Key architectural elements and steps:
- **Data aggregation and preprocessing**: Gather, clean, and preprocess data from multiple sources, ensuring high quality and relevance.
- **Indexing and embedding**: Convert processed text into embeddings for efficient vector-based retrieval.
- **Dynamic retrieval configuration**: Design agents to dynamically fetch context from databases, APIs, and repositories as needed by each query.
- **Tooling and framework selection**: Use frameworks like LlamaIndex to connect large language models with enterprise data sources.

Implementation challenges:
- **Data preparation rigor**: The success of agentic RAG hinges on meticulous data gathering, cleaning, and indexing.
- **Integration complexity**: Dynamically connecting agents with diverse sources and tools increases integration and maintenance complexity.
- **Scalability and extensibility**: Building for growth and change requires careful architectural planning from the outset.

-----

-----

</details>

<details>
<summary>How do re-ranking models, such as cross-encoders, improve retrieval relevance in a RAG pipeline, and what are the associated performance trade-offs in terms of latency and computational cost?</summary>

### Source [86]: https://haystack.deepset.ai/blog/optimize-rag-with-nvidia-nemo

Query: How do re-ranking models, such as cross-encoders, improve retrieval relevance in a RAG pipeline, and what are the associated performance trade-offs in terms of latency and computational cost?

Answer: Adding a reranking component—often implemented as a cross-encoder or fine-tuned large language model (LLM)—to a RAG pipeline improves both **recall** (ability to retrieve all relevant documents) and **precision** (ranking the most relevant documents at the top). The reranker reorders the initially retrieved document chunks so the **most relevant ones appear first**, thus enhancing the likelihood that the LLM receives the best context for answer generation. This is especially critical when the LLM has a limited context window or when inference speed and cost-efficiency are priorities. Reranking is particularly valuable in **hybrid retrieval** scenarios, where multiple retrieval methods are combined—ensuring consistent, semantically relevant document selection regardless of the retrieval method used. This process helps counteract biases of individual retrievers and raises overall output quality.

-----

-----

### Source [87]: https://www.chatbase.co/blog/reranking

Query: How do re-ranking models, such as cross-encoders, improve retrieval relevance in a RAG pipeline, and what are the associated performance trade-offs in terms of latency and computational cost?

Answer: Reranking is described as a **crucial technique** for dramatically improving the relevance and quality of retrieved results in RAG pipelines. While standard retrieval steps may surface a set of candidates, they often fail to consistently identify the most pertinent documents due to limited retrieval model expressivity. By applying more sophisticated relevance judgments—such as cross-encoder-based scoring—rerankers **ensure that the most salient information is passed to the language model** for generation. The source emphasizes that mastering reranking substantially boosts retrieval capabilities and is essential for advanced RAG systems, highlighting its impact on both quality and relevance of outputs.

-----

-----

### Source [88]: https://www.pinecone.io/learn/series/rag/rerankers/

Query: How do re-ranking models, such as cross-encoders, improve retrieval relevance in a RAG pipeline, and what are the associated performance trade-offs in terms of latency and computational cost?

Answer: After implementing reranking, the information sent to the LLM is **far more relevant**, resulting in significantly improved RAG performance. Rerankers maximize the presence of relevant content while minimizing noise, leading to better recall and reduced irrelevant input for the LLM. The two-stage retrieval process—retrieving a broader set of candidates, then reranking—enables scalable search while maintaining high-quality results. The source highlights reranking as one of the simplest and most effective methods for **dramatically improving recall** in RAG or similar retrieval-based pipelines, explaining that the reranker often outperforms the initial embedding-based retriever in selecting the best candidates.

-----

-----

### Source [89]: https://galileo.ai/blog/mastering-rag-how-to-select-a-reranking-model

Query: How do re-ranking models, such as cross-encoders, improve retrieval relevance in a RAG pipeline, and what are the associated performance trade-offs in terms of latency and computational cost?

Answer: A reranker acts as a **second-pass document filter** in information retrieval (IR) systems by reordering the initial candidate set based on more sophisticated query-document relevance metrics. This boosts the **quality of ranked search results** and mitigates risks like hallucinations. Cross-encoders and multi-vector rerankers are specifically noted for their ability to deeply augment search precision. The reranker’s main objective is to enhance search result relevance, but this comes with a trade-off: rerankers, especially cross-encoders, are more **computationally intensive** than simple vector-based retrieval, potentially introducing **latency** and increasing **computational cost**. Thus, while they improve effectiveness, they require careful tuning and observability to maintain practical performance in production RAG systems.

-----

-----

### Source [90]: https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/

Query: How do re-ranking models, such as cross-encoders, improve retrieval relevance in a RAG pipeline, and what are the associated performance trade-offs in terms of latency and computational cost?

Answer: Re-ranking leverages the advanced language understanding capabilities of LLMs to **enhance the relevance of search results**. In a RAG pipeline, integrating a reranker ensures that only the **most relevant chunks** are used to augment the original query, producing more accurate and contextually appropriate responses. This process involves using LLM-based models to assess and reorder candidate documents, thereby improving the quality of the final answer. While the source demonstrates practical integration of reranking into a RAG pipeline, it also implies that reranking with LLMs is resource-intensive, as it requires applying complex models to multiple candidate documents, which increases computational workload and can result in higher latency compared to simpler retrieval methods.
-----

-----

</details>


## Sources Scraped From Research Results

<details>
<summary>Despite the rapid advancements in AI, nearly 70% of organizations still struggle to extract actionable insights from their data. Why? Because the tools they rely on often fail to bridge the gap between unstructured information and meaningful context. This is where Retrieval-Augmented Generation (RAG) systems come into play, offering a lifeline to businesses drowning in data. But here’s the twist—choosing the right RAG approach isn’t as straightforward as it seems.</summary>

Despite the rapid advancements in AI, nearly 70% of organizations still struggle to extract actionable insights from their data. Why? Because the tools they rely on often fail to bridge the gap between unstructured information and meaningful context. This is where Retrieval-Augmented Generation (RAG) systems come into play, offering a lifeline to businesses drowning in data. But here’s the twist—choosing the right RAG approach isn’t as straightforward as it seems.

Graph-based RAG and Vector-based RAG, two dominant methodologies, promise to revolutionize how we retrieve and process information. Yet, their differences go far beyond technical nuances; they represent fundamentally distinct philosophies in data handling. Which one aligns with your needs? And more importantly, how do these choices shape the future of AI-driven decision-making?

This guide unpacks these questions, exploring the untapped potential and trade-offs of each approach.

## Overview of Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) transforms how AI systems interact with external knowledge, but its true power lies in its adaptability to diverse data landscapes. Unlike traditional models that rely solely on pre-trained knowledge, RAG dynamically retrieves domain-specific information, enabling it to generate responses that are both contextually relevant and factually grounded. This dual-layered approach—retrieval and generation—bridges the gap between static knowledge and real-time adaptability.

Consider the healthcare sector: a Graph-based RAG system can traverse complex relationships in patient data, linking symptoms, treatments, and outcomes for precise diagnostics. Meanwhile, a Vector-based RAG excels in unstructured environments, such as analyzing vast medical literature to identify emerging treatment trends.

What’s often overlooked is the role of _retrieval quality_. Poorly indexed data or irrelevant retrieval can undermine even the most advanced generative models. By integrating hybrid techniques, such as reranking algorithms or domain-specific ontologies, organizations can significantly enhance RAG’s performance, unlocking its full potential.

### The Role of Data Structures in RAG Systems

Data structures are the backbone of Retrieval-Augmented Generation (RAG) systems, dictating how information is stored, retrieved, and utilized. Graph RAG leverages _knowledge graphs_, which excel at representing structured relationships between entities. This enables advanced reasoning, such as tracing causal links in medical diagnostics or mapping financial fraud networks. In contrast, Vector RAG relies on _dense vector embeddings_, optimized for semantic similarity searches across unstructured data, like retrieving relevant customer reviews or research papers.

A key consideration is the trade-off between explainability and efficiency. Graph RAG offers transparency by visualizing relationships, but it demands meticulous data curation. Vector RAG, while scalable, often sacrifices interpretability due to its reliance on abstract vector spaces.

To maximize outcomes, hybrid systems are emerging. These combine graph-based reasoning with vector-based retrieval, creating a framework that balances depth and speed. For developers, this means prioritizing use-case-specific data structuring to unlock RAG’s full potential.

## Understanding Graph-Based RAG

Graph-Based RAG thrives on _knowledge graphs_, which organize data into nodes (entities) and edges (relationships). Think of it as a city map: nodes are landmarks, and edges are roads connecting them. This structure allows Graph RAG to perform deep reasoning, such as identifying drug interactions in healthcare or tracing supply chain dependencies in logistics.

A common misconception is that knowledge graphs are static. In reality, they evolve dynamically, integrating new data to reflect real-world changes. For instance, in financial fraud detection, updating the graph with new transaction patterns can reveal hidden networks of fraudulent activity.

What sets Graph RAG apart is its _explainability_. By visualizing subgraphs, users can trace the logic behind AI decisions, fostering trust. However, this comes at a cost—building and maintaining these graphs requires significant effort. Experts suggest hybridizing with vector methods to balance transparency and scalability, unlocking broader applications across industries.

### Fundamentals of Graph Theory in AI

Graph theory enables AI to model _relationships_ rather than isolated data points. This is a game-changer for domains like recommendation systems. For example, Netflix doesn’t just suggest movies based on your viewing history—it maps relationships between genres, actors, and user preferences to predict what you’ll love next.

One overlooked factor is the role of **graph traversal algorithms**. Techniques like Breadth-First Search (BFS) and Depth-First Search (DFS) allow AI to uncover hidden patterns, such as identifying influencers in social networks or tracing supply chain disruptions. These algorithms excel in navigating complex, interconnected datasets.

Interestingly, graph theory also intersects with _game theory_. Auction platforms like eBay use graph-based models to optimize bidding strategies, a concept rooted in Nash equilibrium. Graph theory isn’t just about data—it’s about understanding _interactions_. For AI practitioners, mastering these fundamentals unlocks smarter, more context-aware systems.

### How Graphs Enhance Knowledge Retrieval

Graphs excel at **contextualizing relationships**, making them indispensable for knowledge retrieval. Unlike flat databases, a graph structure connects entities (nodes) through meaningful relationships (edges). This allows systems to infer new knowledge, such as identifying indirect links between concepts. For instance, in healthcare, a graph can connect symptoms to diseases and treatments, enabling precise diagnostic support.

Graphs can organize data categorically, such as grouping products by type, brand, or price range in eCommerce. This layered approach ensures that queries retrieve not just relevant results but also logically grouped insights, improving user experience.

Conventional wisdom often favors speed over depth, but graphs challenge this by balancing **efficiency with explainability**. Traversal algorithms like Dijkstra’s can retrieve optimal paths while maintaining interpretability. For practitioners, integrating graph-based retrieval with domain-specific ontologies offers a framework to enhance both accuracy and transparency in AI-driven systems.

## Exploring Vector-Based RAG

Vector-Based RAG thrives in scenarios involving **vast,** **unstructured data**. By encoding textual information into dense vector embeddings, it enables rapid similarity searches. For instance, in e-commerce, this approach powers product recommendation systems by matching user preferences with semantically similar items, enhancing personalization at scale.

A common misconception is that Vector RAG lacks precision. However, advancements like **learning-to-rank algorithms** and **query expansion techniques** have significantly improved its accuracy. These methods refine retrieval by prioritizing contextually relevant results, even in noisy datasets.

Unexpectedly, Vector RAG intersects with **natural language processing (NLP)**. Embedding models like BERT or GPT capture nuanced semantic relationships, making Vector RAG ideal for applications like literature search or adaptive learning platforms. Think of it as a librarian who not only finds books by title but also understands themes and genres.

Looking ahead, integrating **domain-specific ontologies** into vector spaces could further enhance retrieval quality, bridging gaps in semantic understanding.

### Introduction to Vector Representations

Vector representation transforms data into **high-dimensional numerical embeddings**, capturing semantic relationships. Unlike traditional keyword matching, vectors encode context, enabling systems to understand nuances like synonyms or idiomatic expressions. For example, in **legal document retrieval**, vector embeddings can identify cases with similar precedents, even if phrased differently.

While higher dimensions often improve precision, they can introduce computational overhead. Techniques like **Principal Component Analysis (PCA)** help balance accuracy and efficiency by reducing dimensions without losing critical information.

Interestingly, vector representations draw heavily from **linear algebra**. Operations like cosine similarity measure the “angle” between vectors, revealing how closely two concepts align. This principle extends beyond text—vectors now encode images, audio, and even molecular structures, revolutionizing fields like **drug discovery**.

To maximize impact, organizations should invest in **domain-specific embedding models**, ensuring vectors reflect industry-specific nuances for superior retrieval outcomes.

### Utilizing Vectors for Information Retrieval

Vectors excel in **semantic similarity searches**, enabling retrieval systems to go beyond exact matches. By encoding data into dense embeddings, they capture subtle relationships, such as synonyms or contextual relevance. For instance, in **e-commerce**, vector-based retrieval can recommend products based on user intent, even if the query lacks precise keywords.

Models like BERT or Sentence Transformers perform best when fine-tuned on domain-specific corpora. Without this, retrieval accuracy can suffer, especially in specialized fields like **medical diagnostics** or **legal research**.

Interestingly, vector retrieval intersects with **probabilistic models**. Techniques like **learning-to-rank** refine results by prioritizing embeddings most likely to satisfy user intent. This approach is particularly effective in **personalized content delivery**, where relevance is paramount.

To optimize outcomes, organizations should combine **vector search with reranking models**, ensuring both semantic depth and contextual precision in retrieval pipelines.

### Case Studies: Implementations of Vector-Based RAG

One standout application of Vector-Based RAG is in **academic research platforms**. By leveraging dense embeddings, these systems retrieve semantically relevant papers, even when user queries are vague or imprecise. For example, platforms like Semantic Scholar use vector search to connect researchers with studies that align with their intent, not just their keywords.

In multilingual research, embeddings trained on cross-lingual datasets ensure that papers in different languages are retrieved with equal precision. This approach has revolutionized **global collaboration**, breaking down language barriers in scientific discovery.

Interestingly, **vector-based retrieval** also integrates well with **citation networks**. By combining vector embeddings with graph-based citation data, platforms can recommend not only relevant papers but also influential works within a field. This hybrid approach enhances both **relevance** and **contextual depth**, offering a blueprint for future implementations in other knowledge-intensive domains.

## Core Differences Between Graph-Based and Vector-Based RAG

Graph-Based RAG thrives on **structured relationships**, mapping entities and their connections in a way that mirrors real-world hierarchies. Think of it as a subway map: nodes are stations (entities), and edges are the tracks (relationships). This makes it ideal for domains like **healthcare**, where understanding drug interactions or biological pathways requires deep reasoning over interconnected data.

Vector-Based RAG, by contrast, excels in **semantic similarity**. It’s like a search engine that doesn’t just match words but understands intent. For instance, in **customer support**, it can retrieve relevant FAQs or manuals even when queries are phrased differently. However, its reliance on embeddings means it struggles with multi-step reasoning tasks.

**Hybrid systems** are emerging as the sweet spot. By combining graph reasoning with vector efficiency, they balance **explainability** and **scalability**, offering a glimpse into the future of Retrieval-Augmented Generation.

### Structural Variations and Their Implications

The structural foundation of Graph-Based RAG lies in **knowledge graphs**, where entities and their relationships are explicitly defined. This structure enables **multi-hop reasoning**, making it indispensable in fields like **legal research**, where understanding the interplay between case law, statutes, and precedents is critical. However, maintaining these graphs demands meticulous curation, which can be resource-intensive.

Vector-Based RAG, on the other hand, leverages **dense embeddings** to encode semantic meaning. This approach shines in **unstructured data environments**, such as customer feedback analysis, where rapid retrieval of contextually relevant insights is key. Yet, its reliance on high-dimensional spaces can lead to **overfitting** or **semantic drift**, especially in niche domains.

**Data sparsity** impacts both systems differently. While sparse graphs lose connectivity, sparse vector datasets risk reduced retrieval accuracy. Addressing this requires **hybrid models** that integrate graph structures for reasoning and vectors for scalability, paving the way for more robust RAG systems.

### Performance Analysis Across Different Scenarios

Graph-Based RAG thrives in **highly interconnected datasets**, such as healthcare systems where relationships between symptoms, treatments, and outcomes must be deeply reasoned. Its ability to perform **multi-hop queries** ensures precise insights, but this comes at the cost of **computational overhead**, especially in real-time applications.

Vector-Based RAG, by contrast, excels in **scalability and speed**, making it ideal for **e-commerce platforms** that require rapid semantic searches across millions of product descriptions. However, its reliance on **dense embeddings** can falter in domains where **contextual nuance** is critical, such as legal or scientific research.

**Hybrid retrieval systems** outperform both in dynamic environments. For instance, a **fraud detection system** could use graph reasoning to map suspicious connections while leveraging vector embeddings for semantic anomaly detection. This dual approach not only enhances accuracy but also reduces latency, offering a practical framework for balancing depth and efficiency in diverse scenarios.

### Scalability and Computational Efficiency

Vector-Based RAG shines in **horizontal scalability**, leveraging distributed vector databases to handle billions of embeddings with minimal latency. For example, **e-commerce platforms** like Amazon use vector search to deliver personalized recommendations in milliseconds, even during peak traffic. This efficiency stems from **approximate nearest neighbor (ANN) algorithms**, which reduce computational load by prioritizing approximate matches over exact ones.

Graph-Based RAG, however, faces challenges in scaling due to the **complexity of graph traversal algorithms**. While it excels in domains like **healthcare**, where multi-hop reasoning is critical, its reliance on **centralized graph databases** can bottleneck performance as data grows. Techniques like **graph partitioning** mitigate this but often introduce trade-offs in query accuracy.

A hybrid approach could redefine scalability. By integrating **vector embeddings for initial filtering** and graph traversal for deeper reasoning, systems can achieve both speed and depth. This framework is particularly promising for **fraud detection**, where rapid yet nuanced analysis is essential.

## Practical Implementation Strategies

When implementing Graph-Based RAG, **data preparation is everything**. Start by constructing a robust knowledge graph that captures domain-specific relationships. For instance, in healthcare, mapping diseases, symptoms, and treatments into a graph ensures accurate multi-hop reasoning. Tools like **GraphRAG-SDK** simplify this process by automating graph creation and traversal optimization.

Vector-Based RAG, on the other hand, thrives on **high-quality embeddings**. Training models on domain-specific datasets—like customer reviews for e-commerce—boosts semantic accuracy. Using generic embeddings can dilute precision. Instead, frameworks like **FAISS** enable scalable, efficient similarity searches tailored to your data.

Hybrid systems unlock new possibilities. Picture this: a fraud detection system uses vectors for rapid anomaly detection, then switches to graph traversal for deeper investigation. This layered approach balances speed with depth, addressing both scalability and complexity..

### Selecting the Right Approach for Your Needs

Choosing between Graph-Based RAG and Vector-Based RAG hinges on **data structure and query complexity**. If your domain involves intricate relationships—like mapping supply chain dependencies—Graph RAG excels by leveraging structured knowledge graphs. For example, a logistics company can model routes, delays, and costs as interconnected nodes, enabling multi-step reasoning to optimize delivery times.

Vector RAG, however, shines in **unstructured data environments**. Think of a customer support chatbot that retrieves answers from vast, unorganized FAQs. By encoding semantic meaning into vectors, it ensures quick, relevant responses, even when user queries are vague or imprecise.

In fraud detection, vectors can flag anomalies in real-time, while graphs trace the relationships between flagged entities for deeper insights.

Start with your data’s nature. Structured? Go Graph. Unstructured? Vector. Mixed? Hybrid systems offer the best of both worlds.

### Integration with Existing Systems

Integrating RAG systems with existing infrastructure requires **compatibility with data pipelines and query workflows**. For instance, Graph RAG often demands pre-built knowledge graphs, which can be challenging to align with legacy databases. However, tools like Neo4j or GraphRAG-SDK simplify this by offering APIs that map relational data into graph structures, enabling seamless integration.

Vector RAG integrates more easily with modern vector databases like Pinecone or Weaviate. These systems support high-dimensional embeddings, making them ideal for unstructured data environments such as customer support platforms or recommendation engines.

Hybrid systems combining graph and vector approaches can introduce delays if not optimized. Techniques like caching frequently accessed subgraphs or precomputing embeddings mitigate this issue.

Assess your system’s data flow. For structured data, prioritize graph compatibility. For unstructured data, focus on vector database integration. Optimize hybrid setups with latency-reduction strategies.

### Tools and Frameworks for Development

When building RAG systems, **GraphRAG-SDK** stands out for its ability to streamline graph-based implementations. It simplifies the creation of knowledge graphs by automating entity extraction and relationship mapping, reducing the manual effort required. For example, in healthcare, GraphRAG-SDK can map patient data, symptoms, and treatments into a graph, enabling advanced clinical decision support.

On the vector side, **FAISS** by Facebook excels in high-speed similarity searches. Its GPU-accelerated indexing makes it ideal for large-scale applications like e-commerce, where rapid retrieval of semantically similar products is critical.

**LangChain** bridges graph and vector approaches. It enables hybrid retrieval by chaining graph traversal with vector similarity, offering flexibility for mixed data environments.

Use GraphRAG-SDK for structured domains like finance or healthcare. Leverage FAISS for unstructured, high-volume tasks. For hybrid needs, LangChain provides a scalable, adaptable framework.

## Advanced Applications and Future Developments

The future of RAG systems lies in **hybrid architectures** that seamlessly combine Graph RAG’s structured reasoning with Vector RAG’s semantic agility. For instance, in drug discovery, researchers are exploring joint embedding spaces where molecular interactions (graphs) and scientific literature (vectors) coexist, enabling breakthroughs in identifying novel compounds.

**Graph embeddings** are evolving to mimic vector representations, narrowing the gap between the two approaches. This could revolutionize industries like finance, where fraud detection demands both explainability (graphs) and speed (vectors).

Graph RAG is too rigid for dynamic data. Emerging techniques like **dynamic graph embeddings** challenge this, allowing real-time updates without sacrificing structure.

Looking ahead, **domain-specific ontologies** integrated into vector spaces could redefine semantic retrieval. Imagine personalized education platforms that adapt to a student’s learning style by blending structured knowledge graphs with unstructured content recommendations.

Invest in hybrid RAG research to future-proof AI systems.

### Hybrid Models Combining Graphs and Vectors

Hybrid RAG models excel by leveraging **graph-based reasoning** for structured insights and **vector-based retrieval** for semantic flexibility. A standout example is in **supply chain optimization**, where knowledge graphs map relationships between suppliers, products, and logistics, while vector embeddings analyze unstructured data like market trends or customer reviews. Together, they enable real-time, data-driven decisions.

Graphs provide **explainability**—critical for industries like healthcare—while vectors ensure **scalability** across vast datasets. For instance, a hybrid system in clinical trials could link patient histories (graphs) with emerging research (vectors), uncovering personalized treatment options.

**Query transformation** bridges the gap between these systems. By rephrasing user queries into graph-compatible and vector-compatible formats, hybrid models achieve higher precision.

Start with **domain-specific graph construction** and pair it with pre-trained vector models. This layered approach ensures both depth and breadth in retrieval, unlocking new possibilities for AI-driven solutions.

### Emerging Trends in RAG Technologies

One emerging trend in RAG technologies is the **integration of domain-specific ontologies** into vector spaces. By embedding structured ontologies alongside unstructured data, systems achieve a richer semantic understanding. For example, in **legal tech**, combining case law ontologies with vector embeddings enables nuanced retrieval of precedents, factoring in both legal hierarchies and semantic context.

This approach works because ontologies provide **conceptual scaffolding**, ensuring retrieval aligns with domain-specific logic. Meanwhile, vectors enhance scalability, allowing systems to process vast legal corpora efficiently. **Cross-modal embeddings**—mapping text, images, and graphs into a unified space—further enrich retrieval, especially in multimedia-heavy domains like e-commerce.

Conventional wisdom suggests ontologies are too rigid for dynamic systems. However, **dynamic ontology updates** challenge this, enabling real-time adaptability.

Invest in **ontology-driven vector training pipelines**. This hybrid strategy ensures precision without sacrificing scalability, paving the way for more intelligent, domain-aware RAG systems.

### Cross-Domain Perspectives and Innovations

A key innovation in cross-domain RAG systems is the **use of transfer learning to bridge domain gaps**. By pre-training models on general datasets and fine-tuning them with domain-specific knowledge, systems can adapt to diverse fields like healthcare and e-commerce. For instance, a Graph RAG model trained on biomedical ontologies can be repurposed for supply chain optimization by integrating logistics-specific graphs.

This works because transfer learning leverages shared patterns across domains, reducing the need for extensive retraining. **Domain adaptation techniques**, such as adversarial training, help models retain generalization while aligning with new domain-specific nuances.

Conventional wisdom assumes cross-domain systems sacrifice precision for flexibility. However, **hybrid RAG architectures**—combining graph reasoning with vector scalability—prove otherwise, delivering both adaptability and accuracy.

Develop **modular RAG pipelines** that allow seamless integration of domain-specific components. This ensures scalability while maintaining relevance across industries.

## FAQ

#### 1\. What are the fundamental differences between Graph-Based RAG and Vector-Based RAG?

Graph-Based RAG and Vector-Based RAG differ in data representation and retrieval. Graph-Based RAG organizes information as knowledge graphs with nodes and edges for deep reasoning and explainability, excelling in complex domains like fraud detection but requiring significant computational effort.

Vector-Based RAG uses high-dimensional embeddings for efficient semantic searches, ideal for large-scale unstructured data like document retrieval, offering scalability and speed but lacking multi-step reasoning and explainability. Hybrid systems often combine their strengths for balanced performance.

#### 2\. In which scenarios is Graph-Based RAG more effective than Vector-Based RAG?

Graph-Based RAG is more effective in scenarios that require modeling and reasoning over complex, structured relationships. For example, in domains like healthcare and legal systems, where understanding intricate connections between entities such as patient histories, legal precedents, or regulatory frameworks is critical, Graph-Based RAG excels by leveraging knowledge graphs. Its ability to perform multi-hop reasoning and provide explainable insights makes it particularly valuable in these contexts.

Additionally, applications like fraud detection and scientific research benefit from the structured nature of Graph-Based RAG. By mapping transactions or scientific entities into a graph, it can uncover hidden patterns and relationships that are not immediately apparent in unstructured data. This structured approach ensures a deeper contextual understanding, making it indispensable for tasks requiring high fidelity and interpretability.

#### 3\. How does the explainability of Graph-Based RAG compare to the efficiency of Vector-Based RAG?

Graph-Based RAG offers superior explainability due to its structured representation of data in the form of knowledge graphs. By explicitly mapping entities and their relationships, it allows users to trace the reasoning process through graph traversal, providing clear and interpretable insights. This level of transparency is particularly advantageous in domains like finance or healthcare, where trust and accountability are paramount.

On the other hand, Vector-Based RAG prioritizes efficiency, leveraging high-dimensional embeddings and similarity searches to retrieve relevant information quickly. Its ability to handle large-scale, unstructured datasets with minimal latency makes it ideal for applications like customer support or content recommendation.

However, this efficiency comes at the cost of reduced explainability, as the retrieval process relies on semantic similarity rather than explicit relationships, making it less transparent compared to Graph-Based RAG.

#### 4\. What are the key challenges in implementing hybrid RAG systems combining graphs and vectors?

Implementing hybrid RAG systems combining graphs and vectors faces several challenges. Synchronizing knowledge graphs and vector databases as data evolves requires robust pipelines and high computational resources. Integrating graph reasoning with vector-based retrieval complicates query processing and design, often increasing latency due to dual retrieval processes.

Cost and scalability are also concerns, as knowledge graphs demand intensive maintenance, while vector databases require significant storage and computing power for high-dimensional embeddings. Despite these challenges, hybrid systems balance explainability and efficiency, making them valuable for complex, data-rich applications.

#### 5\. How do Graph-Based and Vector-Based RAG approaches impact real-world applications like healthcare and finance?

Graph-Based and Vector-Based RAG approaches address distinct needs in real-world applications like healthcare and finance. In healthcare, Graph-Based RAG leverages knowledge graphs to model relationships between diseases, symptoms, and treatments, improving diagnostics, workflows, and error reduction with explainable insights. In finance, it aids fraud detection by mapping transactions and identifying hidden patterns through multi-hop reasoning, uncovering complex fraud networks.

Vector-Based RAG offers scalability and efficiency, excelling in handling unstructured data. In healthcare, it enables fast retrieval of semantically similar medical literature or patient records, supporting research and personalized care. In finance, it powers customer support and recommendation systems by efficiently processing large volumes of queries or transaction data. Graph-Based RAG emphasizes depth and explainability, while Vector-Based RAG focuses on speed and flexibility, making both essential for their respective domains.

## Conclusion

Choosing between Graph-Based RAG and Vector-Based RAG is not a matter of superiority but of alignment with specific application needs. Graph-Based RAG thrives in domains like healthcare, where explainability is non-negotiable. For instance, a hospital system using knowledge graphs can trace the relationship between symptoms and treatments, ensuring decisions are transparent and evidence-based. However, this comes with the challenge of maintaining graph structures, akin to curating a vast, ever-evolving library.

Vector-Based RAG, by contrast, shines in unstructured environments like e-commerce. Imagine a platform recommending products based on semantic similarity—quick, scalable, and intuitive. Yet, its “black-box” nature often leaves users questioning the “why” behind results, a trade-off for speed.

A hybrid approach may seem ideal, but integrating these systems is like merging two languages—powerful yet complex. Ultimately, the choice depends on whether your priority is _clarity_ or _velocity_.

</details>

<details>
<summary>Retrieval-Augmented Generation (RAG) has become a cornerstone in AI applications, and as our needs grow, more complex, traditional RAG approaches are showing their limitations. Enter Agentic RAG, which introduces intelligent agents into the retrieval process.</summary>

Retrieval-Augmented Generation (RAG) has become a cornerstone in AI applications, and as our needs grow, more complex, traditional RAG approaches are showing their limitations. Enter Agentic RAG, which introduces intelligent agents into the retrieval process.

Let’s talk about what it is, how it works, and why monitoring and observability are key parts of the process.

## How RAG Works: A Quick Recaphttps://arize.com/wp-content/uploads/2024/03/how-retrieval-augmented-generation-llm-works-1024x399.pngHow RAG works

Let’s start with a quick refresh on traditional RAG, which is kind of like a librarian finding the perfect book for you.

RAG implements a vector-based retrieval process that begins with the transformation of documents into dense vector embeddings. These are then indexed in a vector store. When processing a user query, the system computes an embedding for the input and performs semantic similarity computations (typically using cosine similarity metrics) against the stored document embeddings.

The highest-scoring documents are then retrieved and concatenated into the context window of the prompt, providing the foundation for the language model’s response generation. While this architecture has proven effective for straightforward retrieval tasks, it presents limitations when dealing with heterogeneous data sources or complex, multi-step queries that require more nuanced retrieval strategies.

While this approach works well for simple use cases, it faces challenges when dealing with multiple data sources or complex queries. Traditional RAG often struggles with multi-hop questions that require retrieving information from different parts of the knowledge base or even different data sources sequentially. For instance, a user might ask, ‘What is our return policy for items bought with a discount code I received last month?’ This requires first identifying the general return policy and then applying the specifics related to discount codes and potentially the timeframe of the promotion – a multi-step process that traditional RAG often handles poorly.

## Agentic RAG: Adding Intelligence to Retrieval

At its core, Agentic RAG can be defined as a retrieval-augmented generation framework that leverages autonomous agents to dynamically orchestrate the retrieval of relevant context based on the complexity and nuances of the user query. These agents employ reasoning and decision-making capabilities to select appropriate retrieval tools and strategies, going beyond simple vector similarity searches.

Agentic RAG introduces AI agents into the retrieval process, acting as intelligent intermediaries between user queries and data sources.

These agents can:

- Determine if external knowledge sources are needed at all
- Choose which specific data sources to query based on the question
- Evaluate if the retrieved context actually helps answer the user’s question
- Decide whether to try alternative retrieval strategies if initial results are inadequate

### What are the Key Characteristics of Agentic RAG?

Here is an overview of some key characteristics of agentic RAG.

#### **Dynamic Retrieval**

In contrast to traditional RAG, where the retrieval process is often a fixed sequence of embedding and similarity search, Agentic RAG empowers intelligent agents to adapt their retrieval strategy on the fly based on the nuances of the user query. This adaptability manifests in several ways:

- **Conditional Retrieval:** Agents can determine if retrieval is even necessary. For simple questions that can be answered from the model’s internal knowledge, the agent might bypass external data sources entirely.
- **Adaptive Granularity:** Depending on the query’s complexity, agents can adjust the granularity of the retrieved information. For a broad question, they might initially retrieve high-level summaries, while a specific question might trigger the retrieval of very granular document sections or individual data points.
- **Iterative Refinement:** If the initial retrieval doesn’t yield satisfactory results, agents can iteratively refine their search parameters, try different retrieval methods, or even query related concepts to broaden or narrow the search space.
- **Source Prioritization:** Agents can learn or be configured to prioritize certain data sources based on the query’s context or the historical success rate of those sources for similar queries. For instance, for a question about recent events, an agent might prioritize a real-time news API over a static document repository.

#### Tool Usage

A defining feature of Agentic RAG is the agent’s ability to leverage a diverse set of “tools” to access and retrieve information. These tools extend beyond simple vector stores and enable a much richer and more versatile retrieval process:

- **Vector Stores:** Agents can utilize vector databases (like Chroma, Pinecone, FAISS) for semantic similarity search, just like in traditional RAG. However, they can strategically choose which vector store to query based on the topic of the question.
- **SQL Databases:** Agents equipped with natural language to SQL translation capabilities can query structured data stored in relational databases (like PostgreSQL, MySQL). This allows them to retrieve specific facts and figures based on semantic understanding.
- **APIs (Application Programming Interfaces):** Agents can interact with external APIs to fetch real-time data, such as weather information, stock prices, news feeds, or data from specialized services. This significantly expands the scope of information the system can access.
- **Web Search:** Agents can be equipped with the ability to perform web searches to gather information that might not be present in internal knowledge bases. This is particularly useful for open-ended or exploratory queries.
- **Specialized Tools:** The “tool” concept can be extended to include custom functions or modules designed for specific retrieval tasks, such as accessing file systems, querying knowledge graphs, or interacting with specific enterprise systems. The agent acts as an orchestrator, deciding which tool is most appropriate for each part of the information-gathering process.

#### Reasoning and Planning

Agentic RAG goes beyond simply retrieving relevant documents; the agents exhibit a degree of reasoning and planning to fulfill the user’s request effectively.

- **Intent Recognition:** Agents analyze the user’s query to understand the underlying intent and the specific information being sought. This is crucial for selecting the right tools and retrieval strategies.
- **Decomposition of Complex Queries:** For multi-faceted questions, agents can break down the query into smaller, more manageable sub-tasks. Each sub-task might require a different retrieval strategy or tool.
- **Step-by-Step Planning:** Agents can formulate a plan outlining the sequence of retrieval steps needed to gather all the necessary information. This might involve querying multiple sources in a specific order or iteratively refining the search based on intermediate results.
- **Conditional Logic:** Agents can employ conditional logic (if-then-else rules or more complex decision-making processes) to determine the next course of action based on the outcome of previous retrieval steps. For example, if the initial vector search yields low-confidence results, the agent might decide to try a keyword-based search or consult a different data source.

#### Context Evaluation

After retrieving information from various sources, Agentic RAG agents play a crucial role in evaluating the relevance and quality of the retrieved context:

- **Relevance Scoring:** Agents can employ various techniques (potentially leveraging language models themselves) to score the retrieved documents or data snippets based on their relevance to the original user query and the specific sub-task they were intended to address.
- **Factuality and Reliability Assessment:** In more sophisticated systems, agents might attempt to assess the factuality and reliability of the retrieved information, potentially by cross-referencing information from multiple sources or using external knowledge.
- **Redundancy Detection:** Agents can identify and filter out redundant information retrieved from different sources, ensuring that the context passed to the generation model is concise and focused.
- **Contextual Coherence:** When retrieving information from multiple steps or sources, agents can evaluate the coherence and consistency of the combined context to ensure it provides a unified and logical foundation for the language model’s response.
- **Sufficiency Check:** Agents can determine if the retrieved context is sufficient to answer the user’s query comprehensively. If not, they might initiate further retrieval steps or inform the user about the limitations.

By incorporating these key characteristics, Agentic RAG systems can handle a wider range of complex queries, integrate diverse data sources more effectively, and ultimately provide more accurate and helpful responses compared to traditional RAG approaches.

## What’s the Difference Between Single and Multi-Agent RAG?

The core distinction between single and multi-agent Agentic RAG lies in how the responsibility for the intelligent retrieval process is distributed.

**Single Agent:** One agent handles all aspects of retrieval: query analysis, tool selection, execution, and context evaluation. It’s simpler to implement initially but can become a bottleneck for complex tasks and diverse data. Think of a versatile individual handling everything.

**Multi-Agent:** Multiple specialized agents collaborate, each focusing on specific tasks (e.g., querying a specific database type, handling API calls). A central agent might coordinate. This offers better specialization, scalability, and modularity for complex scenarios but introduces more implementation and coordination challenges. Think of a team of experts working together. So multi-agent RAG might look like this:

- One agent for internal knowledge base queries
- Another agent for external API calls
- Additional agents for specialized tools and operations

|  |
| Feature | Single Agent | Multi-Agent |
| **Complexity** | Lower initial complexity | Higher implementation complexity |
| **Specialization** | Limited | High, optimized per task |
| **Scalability** | Potential bottleneck | Better for high query loads |
| **Modularity** | Less flexible for new tools | More modular and maintainable |
| **Coordination** | Simple (internal) | Requires careful management |
| **Debugging** | Easier to trace initially | More challenging to trace |

## Practical Implementation: A Real-World Example

Let’s look at a practical implementation of Agentic RAG using LlamaIndex. Consider an internal company system that needs to handle both employee information and company policies.

### Architecture Components

The implementation’s foundation rests on a dual-database architecture that leverages both vector and relational paradigms. The system employs Chroma as the vector store for managing company policy documents, while PostgreSQL serves as the relational backbone for structured employee data.

This data architecture means we need specialized query engines: a natural language SQL query engine interfaces with PostgreSQL, translating semantic queries into structured SQL, while a vector query engine handles document retrieval operations through Chroma.

The agent layer sits on top of this infrastructure, configured with specific context parameters that define its operational boundaries and decision-making capabilities. The agent’s architecture incorporates detailed tool descriptions that serve as a decision framework for selecting appropriate data sources, complemented by integration with GPT-3.5 Turbo for sophisticated reasoning capabilities. This configuration enables the agent to dynamically select between the vector and relational query engines based on the semantic requirements of incoming queries.

## Best Practices and Considerations

When implementing Agentic RAG, consider these key points:

1. **Clear Tool Descriptions:** Provide detailed descriptions of each tool’s capabilities to help the agent make informed decisions
2. **Robust Testing:** Verify that agents are selecting the correct tools and retrieving appropriate documents
3. **Document Quality:** Ensure your knowledge base documents contain sufficient context for accurate retrieval
4. **Monitoring Strategy:** Implement comprehensive observability to track and improve system performance

Agentic RAG represents a significant advancement in how we approach information retrieval and question-answering systems. By introducing intelligent agents into the retrieval process, we can handle more complex queries across multiple data sources while maintaining accuracy and relevance. The combination of traditional RAG capabilities with agent-based decision-making opens up new possibilities for building more sophisticated AI applications. As this technology continues to evolve, we can expect to see even more innovative implementations and use cases emerge.

</details>


## Code Sources

_No code sources found._


## YouTube Video Transcripts

_No YouTube video transcripts found._


## Additional Sources Scraped

<details>
<summary>a-complete-guide-to-rag-towards-ai</summary>

If you haven’t heard about RAG from your refrigerator yet, you surely will very soon, so popular this technique has become. Surprisingly, there is a lack of complete guides that consider all the nuances (like relevance assessment, combating hallucinations, etc.), instead of just fragmented pieces. Based on our experience, I have compiled a guide that covers this topic thoroughly.

**So, why do we need RAG?**

You could use LLM models like ChatGPT to create horoscopes (which it does quite successfully), or for something more practical (like work). However, there is a problem: companies typically have a multitude of documents, rules, regulations, etc., about which ChatGPT knows nothing, of course.

**What can be done?**

There are two options: retrain the model with your data or use RAG.

Retraining is long, expensive, and most likely **will not succeed** (don’t worry, it’s not because you’re a bad parent; it’s just that few people can and know how to do it).

The second option is **Retrieval-Augmented Generation** (also known as RAG). Essentially, the idea is simple: take a good existing model (like OpenAI’s), and attach a company information search to it. The model still knows little about your company, but now it has somewhere to look. While not as effective as if it knew everything, it’s sufficient for most tasks.

Here is a basic overview of the RAG structure:https://miro.medium.com/v2/resize:fit:700/0*4FnEj61Crx9YA-dh

**The Retriever** is part of the system that searches for information relevant to your query (similarly to how you would search in your own wiki, company documents, or on Google). Typically, a vector database like Qdrant, where all the company’s indexed documents are stored, is used for this purpose, but essentially anything can be used.

**The Generator** receives the data found by the Retriever and uses it (combines, condenses, and extracts only the important information) to provide an answer to the user. This part is usually done using an LLM like OpenAI. It simply takes all (or part) of the found information and asks to make sense of it and provide an answer.

Here is an example of the simplest implementation of RAG in Python and LangChain.

```
import os
import wget
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain import OpenAI
from langchain_community.document_loaders import BSHTMLLoader
from langchain.chains import RetrievalQA

#download War and Peace by Tolstoy
wget.download("http://az.lib.ru/t/tolstoj_lew_nikolaewich/text_0073.shtml")

#load text from html
loader = BSHTMLLoader("text_0073.shtml", open_encoding='ISO-8859-1')
war_and_peace = loader.load()

#init Vector DB
embeddings = OpenAIEmbeddings()

doc_store = Qdrant.from_documents(
 war_and_peace,
 embeddings,
 location=":memory:",
 collection_name="docs",
)

llm = OpenAI()
# ask questions

while True:
 question = input('Your question: ')
 qa = RetrievalQA.from_chain_type(
 llm=llm,
 chain_type="stuff",
 retriever=doc_store.as_retriever(),
 return_source_documents=False,
 )

 result = qa(question)
 print(f"Answer: {result}")
```

It sounds simple, but there’s a **nuance**:

Since the knowledge isn’t hardcoded into the model, the quality of the answers depends heavily on what the Retriever finds and in what form. It’s not a trivial task, as in the typical chaos of company documents, even people usually have a hard time understanding them. Documents and knowledge are generally stored in poorly structured forms, in different places, sometimes as images, charts, handwritten notes, etc. Often, information in one place contradicts information in another, and one has to make sense of all this mess.

Part of the information simply makes no sense without context, such as abbreviations, acronyms adopted by the company, and names and surnames.

What to do?

This is where various search optimizations (aka hacks) come into play. They are applied at different stages of the search. Broadly, the search can be divided into:

- Initial processing and cleaning of the user’s question
- Data searching in repositories
- Ranking of the results obtained from the repositories
- Processing and combining results into an answer
- Evaluating the response
- Applying formatting, stylistic, and tone

Let’s take a detailed look at each stage:

Initial processing of the user’s question

You wouldn’t believe what users write as questions. You can’t count on them being reasonable — the question could be phrased as a demand, statement, complaint, threat, just a single letter, or AN ENTIRE essay the size of “War and Peace.” For example:https://miro.medium.com/v2/resize:fit:700/0*_zzJBaDyWGgpPL0g

What “and”?

orhttps://miro.medium.com/v2/resize:fit:700/0*__thyuBVknlrDrpB

The input needs to be processed, turning it into a query that can be used to search for information. To solve this problem, we need a translator from the user language to the human language. Who could do this? Of course, an **LLM**. Basically, it might look like this:https://miro.medium.com/v2/resize:fit:700/0*zrXahDMAXECjFNbg

The simplest option — ask the LLM to reformulate the user’s request. But, depending on your audience, this might not be enough!!!!!1111

Then a slightly more complex technique comes into play — **RAG Fusion**.

**RAG Fusion**

The idea is to ask the LLM to provide several versions of the user’s question, conduct a search based on them, and then combine the results, having ranked them beforehand using some clever algorithm, such as a **Cross-Encoder**. Cross Encoder works quite slowly, but it provides more relevant results, so it’s not practical to use it for information retrieval — however, for ranking a list of found results, it’s quite suitable.

**Remarks about Cross and Bi Encoders**

Vector databases use **Bi-encoder** models to compute the similarity of two concepts in vector space. These models are trained to represent data in vector form and, accordingly, during a search, the user’s query is also turned into a vector, and vectors closest to the query are returned. However, this proximity does not guarantee that it is the best answer.https://miro.medium.com/v2/resize:fit:700/0*N_wng9uYMv5zi3ge

**Cross-Encoder** works differently. It takes two objects (texts, images, etc.) and returns their relevance (similarity) relative to each other. Its accuracy is usually [better](https://arxiv.org/abs/1908.10084) than that of a Bi-Encoder. Typically, more results than necessary are returned from the vector database (just in case, say 30) and then they are ranked using a Cross-Encoder or similar techniques, with the top 3 being returned.https://miro.medium.com/v2/resize:fit:700/0*h98JJPZ-YSiOWwR1

User request preprocessing also includes its classification. For example, requests can be subdivided into questions, complaints, requests, etc. Requests can further be classified as urgent, non-urgent, spam, or fraud. They can be classified by departments (e.g., accounting, production, HR), etc. All this helps narrow down the search for information and, consequently, increases the speed and quality of the response.

For classification, an LLM model or a specially trained neural network classifier can be used again.

**Data Search in Repositories**

The so-called **retriever**(the first letter in RAG) is responsible for the search.

Usually, a vector database serves as the repository where company data from various sources (document storage, databases, wikis, CRM, etc.) are indexed. However, it’s not mandatory and anything can be used, such as Elasticsearch or even Google search.

I will not discuss non-vector base searches here, as the principle is the same everywhere.

**Digression about Vector Databases**

A vector database (or vector storage. I use these terms interchangeably, although technically they are not the same) is a type of data storage optimized for storing and processing vectors (which are essentially arrays of numbers). These vectors are used to represent complex objects, such as images, texts, or sounds, as vectors in vector spaces for machine learning and data analysis tasks. In a vector database (or, more precisely, in vector space), concepts that are semantically similar are located close to each other, regardless of their representation. For example, the words “dog” and “bulldog” will be close, whereas the words “lock” (as in a door lock) and “lock” (as in a castle) will be far apart. Therefore, vector databases are well suited for semantic data search.

Most Popular Vector Databases (as of now):

- **QDrant**— open-source database
- **Pinecone**— cloud-native (i.e., they will charge you a lot) database
- **Chroma**— another open-source database (Apache-2.0 license)
- **Weaviate**— open under BSD-3-Clause license
- **Milvus**— open under Apache-2.0 license
- **FAISS**— a separate beast, not a database but a framework from Meta

Also, some popular non-vector databases have started offering vector capabilities:

- **Pgvector** for PostgreSQL
- **Atlas** for Mongo

To improve results, several main techniques are used:

**Ensemble of retrievers and/or data sources —** a simple but effective idea, which involves asking several experts the same question and then somehow aggregating their answers (even just averaging) — the result on average turns out better. In some sense, this is analogous to “Ask the Crowd.”

As an example — the use of multiple types of retrievers from Langchain. Ensembling is particularly useful when combining sparse retrievers (like BM25) and dense retrievers (working based on embedding similarities, such as the same vector databases) because they complement each other well.

**Dense Retriever** — typically uses transformers, such as BERT, to encode both queries and documents into vectors in a multidimensional space. The similarity between a query and a document is measured by the proximity of their vectors in this space, often using cosine similarity to assess their closeness. This is the basis on which vector databases are built. Such a model better understands the semantic (meaningful) value of queries and documents, leading to more accurate and relevant results, especially for complex queries. Because the model operates at the level of meaning (semantics), it handles paraphrasing and semantic similarities well.

**Sparse Retriever** — uses traditional information retrieval methods, such as TF-IDF (Term Frequency) or BM25. These methods create sparse vectors, where each dimension corresponds to a specific term from a predefined dictionary. The relevance of a document to a user’s query is calculated based on the presence and frequency of the terms (or words, let’s say) of the query in the document. It is effective for keyword-based queries and when the terms of the query are expected to be directly present in the relevant documents. They don’t always work as accurately as dense retrievers, but are faster and require fewer resources for searching and training.

The EnsembleRetriever then ranks and combines results using, for example, [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf):

_Example of an ensemble:_

```
!pip install rank_bm25
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import Chroma

embedding = OpenAIEmbeddings()
documents = "/all_tolstoy_novels.txt"
bm25_retriever = BM25Retriever.from_texts(doc_list)
bm25_retriever.k = 2

vectorstore = Chroma.from_texts(doc_list, embedding)
vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vectorstore_retriever ], weights=[0.4, 0.6])
docs = ensemble_retriever.get_relevant_documents("War and Peace")
```

How to choose the right strategy from all this zoo? Experiment. Or use a framework, for example, [https://github.com/Marker-Inc-Korea/AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG).

By the way, it’s also possible to ensemble several LLMs, which also improves the result. See “ [More agents is all you need](https://arxiv.org/abs/2402.05120).”

**RELP**

This is another method for data retrieval, Retrieval Augmented Language Model based Prediction. The distinction here is in the search step — after we find information in the vector storage, including using the techniques mentioned above, we use it not to generate an answer using an LLM but to generate example answers (via [few-shot prompting](https://en.wikipedia.org/wiki/Few-shot_learning)) for the LLM. Based on these examples, the LLM effectively learns and responds based on this mini-training to the posed question. This technique is a form of dynamic learning, which is much less costly than re-training the model using standard methods.https://miro.medium.com/v2/resize:fit:451/0*0fpWX48eZ6Lw7nJR

**Remarks about few-shot (learning) prompting**

There are two similar LLM prompting techniques: zero-shot and few-shot. Zero-shot is when you ask your LLM a question without providing any examples. For instance:https://miro.medium.com/v2/resize:fit:700/0*AlG7hPApACw63e2-

**Few-shot** is when you first give the LLM several examples on which it trains. This significantly increases the likelihood of getting a relevant answer in the relevant form. For example:https://miro.medium.com/v2/resize:fit:700/0*hyKm2xkJEW17gXDM

As you can see, not everything is always obvious, and examples help to understand.

**Ranking, Combining, and Evaluating the Obtained Results**

We have already partially touched on this topic as part of RAG Fusion and the ensembling of retrievers. When we extract results from a (vector) storage, before sending this data to an LLM for answer generation, we need to rank the results, and possibly discard the irrelevant ones. The order in which you present the search results to the LLM for answer formulation matters. What the LLM sees first will have a stronger influence on the final outcome (more details here).

Different approaches are used for ranking. The most common include:

1. Using a **Cross-Encoder** (described above) for re-ranking the obtained results and discarding the least relevant (for example, pulling the top 30 results from a vector database (top k), ranking them with a Cross-Encoder, and taking the first 10).

There are already ready-made solutions for these purposes, for example from [Cohere](https://cohere.com/rerankhttps:/cohere.com/rerank).

[2\. Reciprocal Rank Fusion.](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) The main idea of RRF is to give greater importance to elements occupying higher positions in each set of search results. In RRF, the score of each element is calculated based on its position in the individual search results. This is usually done using the formula 1/(k + rank), where “rank” is the position of the element in a particular set of search results, and “k” is a constant (often set around 60). This formula provides a higher score for elements with a higher rank.

Scores for each element in different sets of results are then summed to obtain a final score. Elements are sorted by these final scores to form a combined list of results.

RRF is particularly useful because it does not depend on the absolute scores assigned by individual search systems, which can vary significantly in their scale and distribution. RRF effectively combines results from different systems in a way that highlights the most consistently highly ranked elements.

3\. LLM-based ranking and evaluation: you can relax and simply ask an LLM to rank and evaluate the result 🙂. The latest versions of OpenAI handle this quite well. However, using them for this purpose is costly.

**Evaluation of search results in the Vector Store:**

Suppose you have made reranking or other changes — how do you determine whether it was beneficial? Did the relevance increase or not? And in general, how well does the system work? This is a quality metric for the information found. It is used to understand how relevant the information your system finds is and to make decisions about further refinement.

You can assess how relevant the results are to the query using the following metrics: P@K, MAP@K, NDCG@K (and similar). These usually return a number from 0 to 1, where 1 is the highest accuracy. They are similar in meaning, with differences in details:

**P@K** means precision at K, i.e., accuracy for K elements. Suppose for a query about rabbits, the system found 4 documents:

_\[“Wild Rabbits”, “Task Rabbit: modern jobs platform”, “Treatise on Carrots”, “My Bunny: Memoirs by Walter Issac”\]_

Since Walter Issac's biography or jobs platforms have nothing to do with rabbits, these positions would be rated 0, and the overall accuracy would be calculated like this:https://miro.medium.com/v2/resize:fit:700/0*fkdMKuqKYbn5odPj

P@K at K=4, or P@4 = 2 relevant / (2 relevant + 2 irrelevant) = ½ = 0.5.

However, this does not take into account the order. What if the returned list looks like this:

_\[“Task Rabbit: modern jobs platform”, “My Bunny: Memoirs by Walter Issac”, “Wild Rabbits”, “Treatise on Carrots”\]_

P@K is still 0.5, but as we know, the order of relevant and irrelevant results matters! (both for people and for the LLM that will use them).

Therefore, we use **AP@K** or average precision at K. The idea is simple: we need to modify the formula so that the order is taken into account and relevant results at the end of the list do not increase the overall score less than those at the beginning of the list:https://miro.medium.com/v2/resize:fit:700/0*kQLsLwQqidjxR6sA

Or for our example above:

AP@4 = (0 \* 0 + 0 \*½ + 1 \* ⅓ + 1 + 1 \* 2/4) .2 = (⅓ + 2/4) / 2 = 0.41

Here are a few questions that arise: how did we assess the relevance of individual elements to calculate these metrics? This is a detective question, a very good one indeed.

In the context of RAG, we often ask an LLM or another model to make an assessment. That is, we query the LLM about each element — this document we found in the vector storage — is it relevant to this query at all?

Now, the second question: is it sufficient to ask just this way? The answer is no. We need more specific questions for the LLM that ask it to assess relevance according to certain parameters. For example, for the sample above, the questions might be:

Does this document relate to the animal type “rabbit”?

Is the rabbit in this document real or metaphorical?

Etc. There can be many questions (from two to hundreds), and they depend on how you assess relevance. This needs to be aggregated, and that’s where:

MAP@K (Mean Average Precision at K) comes in — it’s the average of the sum of AP@K for all questions.

NDCG@K stands for normalized discounted cumulative gain at K, and I won’t even translate that 🙂. Look it up online yourself.

**Evaluating the results of the LLM response**

Not everyone knows this, but you can ask an LLM (including Llama and OpenAI) not just to return tokens (text) but logits. That is, you can actually ask it to return a distribution of tokens with their probabilities, and see — how confident is the model really in what it has concocted (calculating token-level uncertainty)? If the probabilities in the distribution are low (what is considered low depends on the task), then most likely, the model has started to fabricate (hallucinate) and is not at all confident in its response. This can be used to evaluate the response and to return an honest “I don’t know” to the user.

**Using formatting, style, and tone**

The easiest item 🙂. Just ask the LLM to format the answer in a certain way and use a specific tone. It’s better to give the model an example, as it then follows instructions better. For instance, you could set the tone like this:https://miro.medium.com/v2/resize:fit:700/0*lsnHytCsXujPJZrv

Formatting and stylistics can be programmatically set in the last step of RAG — requesting the LLM to generate the final answer, for example:

```
question = input('Your question: ')
style = 'Users have become very very impudent lately. Answer as a gangster from a ghetto'
qa = RetrievalQA.from_chain_type(
 llm=llm,
 chain_type="stuff",
 retriever=doc_store.as_retriever()
)

result = qa(style + " user question: " + question)
print(f"Answer: {result}")
```

## Fine-tuning models

Sometimes you might indeed need further training. Yes, initially I said that most likely you won’t succeed, but there are cases when it is justified. If your company uses acronyms, names/surnames, and terms that the model does not and cannot know about, RAG may perform poorly. For example, it might struggle with searching data by Russian surnames, especially their declensions. Here, a light fine-tuning of the model using [LORA](https://arxiv.org/abs/2106.09685) can help, to train the model to understand such specific cases. You can use frameworks like [https://github.com/bclavie/RAGatouille](https://github.com/bclavie/RAGatouille).

Such fine-tuning is beyond the scope of this article, but if there is interest, I will describe it separately.

**Systems based on RAG**

There are several more advanced options based on RAG. In fact, new variants are emerging almost every day, and their creators claim that they have become increasingly better…

Nevertheless, one variation stands out — [FLARE](https://arxiv.org/abs/2305.06983)(Forward Looking Active REtrieval Augmented Generation).

It’s a very interesting idea based on the principle that RAG should not be used haphazardly but only when the LLM itself wants to. If the LLM confidently answers without RAG, then please proceed. However, if it starts to doubt, that’s when more contextual data needs to be searched for. This should not be done just once but as many times as necessary. When, during the response process, the LLM feels it needs more data, it performs a RAG search.

In some ways, this is similar to how people operate. We often do not know what we do not know and realize it only during the search process itself.

I will not go into details here; that is a topic for another article.

## Summary

In this article, I’ve provided a comprehensive guide to Retrieval-Augmented Generation (RAG). Here’s a quick recap of our journey:

**Need and Advantages:** I started by discussing why RAG is needed and its benefits over retraining models with custom data.

**RAG Structure:** Then, I explained the basic structure of RAG, highlighting the roles of the Retriever and Generator components.

**Implementation:** I walked through an example implementation using Python and LangСhain.

**User Query Processing:** I delved into processing user queries, including RAG Fusion and Cross-Encoders.

**Data Search Techniques:** Next, I explored various data search techniques, such as vector databases and ensembling retrievers.

**Ranking and Evaluating:** I covered the importance of ranking, combining, and evaluating retrieved results to improve response quality.

**Advanced Methods:** Finally, I discussed optimizations and advanced methods like RELP and FLARE, as well as considerations for fine-tuning models and maintaining response formatting, style, and tone.

</details>

<details>
<summary>advanced-rag-blueprint-optimize-llm-retrieval-systems</summary>

The vanilla RAG framework doesn’t address many fundamental aspects that impact the quality of the retrieval and answer generation, such as:

- Are the retrieved documents relevant to the user’s question?
- Is the retrieved context enough to answer the user’s question?
- Is there any redundant information that only adds noise to the augmented prompt?
- Does the latency of the retrieval step match our requirements?
- What do we do if we can’t generate a valid answer using the retrieved information?

From the questions above, we can draw two conclusions.

**The first one** is that we need a robust evaluation module for our RAG system that can quantify and measure the quality of the retrieved data and generate answers relative to the user’s question.

**The second conclusion** is that we must improve our RAG framework to address the retrieval limitations directly in the algorithm. These improvements are known as advanced RAG.

This article will focus on the second conclusion, answering the question: _“How can I optimize an RAG system?”_.

[https://substackcdn.com/image/fetch/$s_!yOzH!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F19d2780b-7e3f-48aa-8e67-c4107ef8f0c7_792x792.png](https://substackcdn.com/image/fetch/$s_!yOzH!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F19d2780b-7e3f-48aa-8e67-c4107ef8f0c7_792x792.png) Figure 1: The three stages of advanced RAG

The vanilla RAG design can be **optimized** **at** **three different stages**:

1. **Pre-retrieval:** This stage focuses on structuring and preprocessing your data for data indexing and query optimizations.
2. **Retrieval:** This stage revolves around improving the embedding models and metadata filtering to improve the vector search step.
3. **Post-retrieval:** This stage mainly targets different ways to filter out noise from the retrieved documents and compress the prompt before feeding it to an LLM for answer generation.

* * *

## 1. Pre-retrieval

The pre-retrieval steps are performed in two different ways:

- **Data indexing:** It is part of the RAG ingestion pipeline. It is mainly implemented within the cleaning or chunking modules to preprocess the data for better indexing.
- **Query optimization:** The algorithm is performed directly on the user’s query before embedding it and retrieving the chunks from the vector DB.

As we index our data using embeddings that semantically represent the content of a chunked document, most of the data indexing techniques focus on better preprocessing and structuring the data to improve retrieval efficiency.

Here are a few popular methods for **optimizing data indexing**.

#### 1. Sliding window

The sliding window technique introduces overlap between text chunks, ensuring that important context near chunk boundaries is retained, which enhances retrieval accuracy.

This is particularly beneficial in domains like legal documents, scientific papers, customer support logs, and medical records, where critical information often spans multiple sections.

The embedding is computed on the chunk along with the overlapping portion. Hence, the sliding window improves the system’s ability to retrieve relevant and coherent information by maintaining context across boundaries.

#### 2. Enhancing data granularity

This involves data cleaning techniques like removing irrelevant details, verifying factual accuracy, and updating outdated information. A clean and accurate dataset allows for sharper retrieval.

#### 3. Metadata

Adding metadata tags like dates, URLs, external IDs, or chapter markers helps filter results efficiently during retrieval.

#### 4. Optimizing index structures

It is based on different data index methods, such as various chunk sizes and multi-indexing strategies.

#### 5. Small-to-big

The algorithm decouples the chunks used for retrieval and the context used in the prompt for the final answer generation.

The algorithm uses a small sequence of text to compute the embedding while preserving the sequence itself and a wider window around it in the metadata. Thus, using smaller chunks enhances the retrieval’s accuracy, while the larger context adds more contextual information to the LLM.

The intuition behind this is that if we use the whole text for computing the embedding, we might introduce too much noise, or the text could contain multiple topics, which results in a poor overall semantic representation of the embedding.

**On the** **query optimization side**, we can leverage techniques such as query routing, query rewriting, and query expansion to refine the retrieved information for the LLM further.

#### 1. Query routing

Based on the user’s input, we might have to interact with different categories of data and query each category differently.

Query rooting is used to decide what action to take based on the user’s input, similar to if/else statements. Still, the decisions are made solely using natural language instead of logical statements.

As illustrated in Figure 2, let’s assume that, based on the user’s input, to do RAG, we can retrieve additional context from a vector DB using vector search queries, a standard SQL DB by translating the user query to an SQL command, or the internet by leveraging REST API calls.

The query router can also detect whether a context is required, helping us avoid making redundant calls to external data storage. Also, a query router can pick the best prompt template for a given input.

The routing usually uses an LLM to decide what route to take or embeddings by picking the path with the most similar vectors.

To summarize, query routing is identical to an if/else statement but much more versatile as it works directly with natural language.

[https://substackcdn.com/image/fetch/$s_!SdZP!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fab3d3265-49a9-4bd6-a61a-7a3bf019ef25_792x792.png](https://substackcdn.com/image/fetch/$s_!SdZP!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fab3d3265-49a9-4bd6-a61a-7a3bf019ef25_792x792.png) Figure 2: Query routing

#### 2. Query rewriting

Sometimes, the user’s initial query might not perfectly align with how your data is structured. Query rewriting tackles this by reformulating the question to match the indexed information better.

This can involve techniques like:

- **Paraphrasing:** Rephrasing the user’s query while preserving its meaning (e.g., “What are the causes of climate change?” could be rewritten as “Factors contributing to global warming”).
- **Synonym substitution:** Replacing less common words with synonyms to broaden the search scope (e.g., “ joyful” could be rewritten as “happy”).
- **Sub-queries:** For longer queries, we can break them down into multiple shorter and more focused sub-queries. This can help the retrieval stage identify relevant documents more precisely.

#### 3. Hypothetical document embeddings (HyDE)

This technique involves having an LLM create a hypothetical response to the query. Then, both the original query and the LLM’s response are fed into the retrieval stage.

#### 4. Query expansion

This approach aims to enrich the user’s question by adding additional terms or concepts, resulting in different perspectives of the same initial question. For example, when searching for “disease,” you can leverage synonyms and related terms associated with the original query words and also include “illnesses” or “ailments.”

#### 5. Self-query

The core idea is to map unstructured queries into structured ones. An LLM identifies key entities, events, and relationships within the input text. These identities are used as filtering parameters to reduce the vector search space (e.g., identify cities within the query, for example, “Paris,” and add it to your filter to reduce your vector search space).

Both data indexing and query optimization pre-retrieval optimization techniques depend highly on your data type, structure, and source. Thus, as with any data processing pipeline, no method always works, as every use case has its own particularities and gotchas.

Optimizing your pre-retrieval RAG layer is experimental. Thus, what is essential is to try multiple methods (such as the ones enumerated in this section), reiterate, and observe what works best.

## 2. Retrieval

The retrieval step can be optimized in two fundamental ways:

- Improving the embedding models used in the RAG ingestion pipeline to encode the chunked documents and, at inference time, transform the user’s input.
- Leveraging the DB’s filter and search features. This step will be used solely at inference time when you have to retrieve the most similar chunks based on user input.

Both strategies are aligned with our ultimate goal: to enhance the vector search step by leveraging the semantic similarity between the query and the indexed data.

When improving the **embedding models**, you usually have to fine-tune the pre-trained embedding models to tailor them to specific jargon and nuances of your domain, especially for areas with evolving terminology or rare terms.

Instead of fine-tuning the embedding model, you can leverage instructor models, such as **[instructor-xl](https://huggingface.co/hkunlp/instructor-xl)**, to guide the embedding generation process with an instruction/prompt aimed at your domain. Tailoring your embedding network to your data using such a model can be a good option, as fine-tuning a model consumes more computing and human resources.

In the code snippet below, you can see an example of an Instructor model that embeds article titles about AI:

```
from InstructorEmbedding import INSTRUCTOR

model = INSTRUCTOR(“hkunlp/instructor-base”)

sentence = “RAG Fundamentals First”

instruction = “Represent the title of an article about AI:”

embeddings = model.encode([[instruction, sentence]])

print(embeddings.shape) # noqa

# Output: (1, 768)
```

On the other side of the spectrum, here is how you can **improve your retrieval** by leveraging classic filter and search DB features.

#### **Hybrid search**

This is a vector and keyword-based search blend.

Keyword-based search excels at identifying documents containing specific keywords. When your task demands pinpoint accuracy, and the retrieved information must include exact keyword matches, hybrid search shines. Vector search, while powerful, can sometimes struggle with finding exact matches, but it excels at finding more general semantic similarities.

You leverage both keyword matching and semantic similarities by combining the two methods. You have a parameter, usually called alpha, that controls the weight between the two methods. The algorithm has two independent searches, which are later normalized and unified.

#### Filtered vector search

This type of search leverages the metadata index to filter for specific keywords within the metadata. It differs from a hybrid search in that you retrieve the data once using only the vector index and perform the filtering step before or after the vector search to reduce your search space.

In practice, you usually start with filtered vector or hybrid search on the retrieval side, as they are fairly quick to implement. This approach gives you the flexibility to adjust your strategy based on performance.

If the results are unexpected, you can always fine-tune your embedding model.

## 3. Post-retrieval

The post-retrieval optimizations are solely performed on the retrieved data to ensure that the LLM’s performance is not compromised by issues such as limited context windows or noisy data.

This is because the retrieved context can sometimes be too large or contain irrelevant information, both of which can distract the LLM.

Two popular methods performed at the post-retrieval step are the following.

#### **Prompt compression**

Eliminate unnecessary details while keeping the essence of the data.

#### Re-ranking

Use a cross-encoder ML model to give a matching score between the user’s input and every retrieved chunk.

[https://substackcdn.com/image/fetch/$s_!RjK7!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fff797810-7274-4a51-b487-ee06e881efe6_792x792.png](https://substackcdn.com/image/fetch/$s_!RjK7!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fff797810-7274-4a51-b487-ee06e881efe6_792x792.png) Figure 3: Bi-encoder (the standard embedding model) versus cross-encoder

The retrieved items are sorted based on this score. Only the top N results are kept as the most relevant. As you can see in Figure 3, this works because the re-ranking model can find more complex relationships between the user input and some content than a simple similarity search.

However, we can’t apply this model at the initial retrieval step because it is costly. That is why a popular strategy is to retrieve the data using a similarity distance between the embeddings and refine the retrieved information using a re-raking model, as illustrated in Figure 4.

[https://substackcdn.com/image/fetch/$s_!2OdZ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F33ab6874-b3ea-4cd4-9a66-9665afe27de5_792x792.png](https://substackcdn.com/image/fetch/$s_!2OdZ!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F33ab6874-b3ea-4cd4-9a66-9665afe27de5_792x792.png) Figure 4: The re-ranking algorithm

* * *

## Conclusion

The abovementioned techniques are far from an exhaustive list of all potential solutions. We used them as examples to get an intuition on what you can (and should) optimize at each step in your RAG workflow.

The truth is that these techniques can vary tremendously by the type of data you work with. For example, if you work with multi-modal data such as text and images, most of the techniques from earlier won’t work as they are designed for text only.

To summarize, the primary goal of these optimizations is to enhance the RAG algorithm at three key stages: pre-retrieval, retrieval, and post-retrieval.

This involves preprocessing data for improved vector indexing, adjusting user queries for more accurate searches, enhancing the embedding model, utilizing classic filtering DB operations, and removing noisy data.

By keeping these goals in mind, you can optimize your RAG workflow for data processing and retrieval.

</details>

<details>
<summary>build-advanced-retrieval-augmented-generation-systems-micros</summary>

# Build advanced retrieval-augmented generation systems

This article explains retrieval-augmented generation (RAG) and what developers need to build a production-ready RAG solution.

The following diagram shows the main steps of RAG:https://learn.microsoft.com/en-us/azure/developer/ai/media/naive-rag-inference-pipeline-highres.png

This process is called _naive RAG_. It helps you understand the basic parts and roles in a RAG-based chat system.

Real-world RAG systems need more preprocessing and post-processing to handle articles, queries, and responses. The next diagram shows a more realistic setup, called _advanced RAG_:https://learn.microsoft.com/en-us/azure/developer/ai/media/advanced-rag-inference-pipeline-highres.png

This article gives you a simple framework to understand the main phases in a real-world RAG-based chat system:

- Ingestion phase
- Inference pipeline phase
- Evaluation phase

## Ingestion

Ingestion means saving your organization's documents so you can quickly find answers for users. The main challenge is to find and use the parts of documents that best match each question. Most systems use vector embeddings and cosine similarity search to match questions to content. You get better results when you understand the content type (like patterns and format) and organize your data well in the vector database.

When setting up ingestion, focus on these steps:

- Content preprocessing and extraction
- Chunking strategy
- Chunking organization
- Update strategy

### Content preprocessing and extraction

The first step in the ingestion phase is to preprocess and extract the content from your documents. This step is crucial because it ensures that the text is clean, structured, and ready for indexing and retrieval.

Clean and accurate content makes a RAG-based chat system work better. Start by looking at the shape and style of the documents you want to index. Do they follow a set pattern, like documentation? If not, what questions could these documents answer?

At a minimum, set up your ingestion pipeline to:

- Standardize text formats
- Handle special characters
- Remove unrelated or old content
- Track different versions of content
- Handle content with tabs, images, or tables
- Extract metadata

Some of this information, like metadata, can help during retrieval and evaluation if you keep it with the document in the vector database. You can also combine it with the text chunk to improve the chunk's vector embedding.

### Chunking strategy

As a developer, decide how to break up large documents into smaller chunks. Chunking helps send the most relevant content to the LLM so it can answer user questions better. Also, think about how you'll use the chunks after you get them. Try out common industry methods and test your chunking strategy in your organization.

When chunking, think about:

- **Chunk size optimization**: Pick the best chunk size and how to split it—by section, paragraph, or sentence.
- **Overlapping and sliding window chunks**: Decide if chunks should be separate or overlap. You can also use a sliding window approach.
- **Small2Big**: If you split by sentence, organize the content so you can find nearby sentences or the full paragraph. Giving this extra context to the LLM can help it answer better. For more, see the next section.

### Chunking organization

In a RAG system, how you organize your data in the vector database makes it easier and faster to find the right information. Here are some ways to set up your indexes and searches:

- **Hierarchical indexes**: Use layers of indexes. A top-level summary index quickly finds a small set of likely chunks. A second-level index points to the exact data. This setup speeds up searches by narrowing down the options before looking in detail.
- **Specialized indexes**: Pick indexes that fit your data. For example, use graph-based indexes if your chunks connect to each other, like in citation networks or knowledge graphs. Use relational databases if your data is in tables, and filter with SQL queries.
- **Hybrid indexes**: Combine different indexing methods. For example, use a summary index first, then a graph-based index to explore connections between chunks.

### Alignment optimization

Make retrieved chunks more relevant and accurate by matching them to the types of questions they answer. One way is to create a sample question for each chunk that shows what question it answers best. This approach helps in several ways:

- **Improved matching**: During retrieval, the system compares the user’s question to these sample questions to find the best chunk. This technique improves the relevance of the results.
- **Training data for machine learning models**: These question-chunk pairs help train the machine learning models in the RAG system. The models learn which chunks answer which types of questions.
- **Direct query handling**: If a user’s question matches a sample question, the system can quickly find and use the right chunk, speeding up the response.

Each chunk’s sample question acts as a label that guides the retrieval algorithm. The search becomes more focused and aware of context. This method works well when chunks cover many different topics or types of information.

### Update strategies

If your organization updates documents often, you need to keep your database current so the retriever can always find the latest information. The _retriever component_ is the part of the system that searches the vector database and returns results. Here are some ways to keep your vector database up to date:

- **Incremental updates**:

  - **Regular intervals**: Set updates to run on a schedule (like daily or weekly) based on how often documents change. This action keeps the database fresh.
  - **Trigger-based updates**: Set up automatic updates when someone adds or changes a document. The system reindexes only the affected parts.
- **Partial updates**:

  - **Selective reindexing**: Update only the parts of the database that changed, not the whole thing. This technique saves time and resources, especially for large datasets.
  - **Delta encoding**: Store just the changes between old and new documents, which reduces the amount of data to process.
- **Versioning**:

  - **Snapshotting**: Save versions of your document set at different times. This action lets you go back or restore earlier versions if needed.
  - **Document version control**: Use a version control system to track changes and keep a history of your documents.
- **Real-time updates**:

  - **Stream processing**: Use stream processing to update the vector database in real time as documents change.
  - **Live querying**: Use live queries to get up-to-date answers, sometimes mixing live data with cached results for speed.
- **Optimization techniques**:

  - **Batch processing**: Group changes and apply them together to save resources and reduce overhead.
  - **Hybrid approaches**: Mix different strategies:

    - Use incremental updates for small changes.
    - Use full reindexing for significant updates.
    - Track and document major changes to your data.

Pick the update strategy or mix that fits your needs. Think about:

- Document corpus size
- Update frequency
- Real-time data needs
- Available resources

Review these factors for your application. Each method has trade-offs in complexity, cost, and how quickly updates show up.

## Inference pipeline

Your articles are now chunked, vectorized, and stored in a vector database. Next, focus on getting the best answers from your system.

To get accurate and fast results, think about these key questions:

- Is the user's question clear and likely to get the right answer?
- Does the question break any company rules?
- Can you rewrite the question to help the system find better matches?
- Do the results from the database match the question?
- Should you change the results before sending them to the LLM to make sure the answer is relevant?
- Does the LLM's answer fully address the user's question?
- Does the answer follow your organization's rules?

The whole inference pipeline works in real time. There’s no single right way to set up your preprocessing and post-processing steps. You use a mix of code and LLM calls. One of the biggest trade-offs is balancing accuracy and compliance with cost and speed.

Let’s look at strategies for each stage of the inference pipeline.

### Query preprocessing steps

Query preprocessing starts right after the user sends a question:https://learn.microsoft.com/en-us/azure/developer/ai/media/advanced-rag-query-processing-steps-highres.png

These steps help make sure the user’s question fits your system and is ready to find the best article chunks using cosine similarity or "nearest neighbor" search.

**Policy check**: Use logic to spot and remove or flag unwanted content, like personal data, bad language, or attempts to break safety rules (called "jailbreaking").

**Query rewriting**: Change the question if needed—expand acronyms, remove slang, or rephrase it to focus on bigger ideas (step-back prompting).

A special version of step-back prompting is _Hypothetical Document Embeddings (HyDE)_. HyDE has the LLM answer the question, makes an embedding from that answer, and then searches the vector database with it.

### Subqueries

Subqueries break a long or complex question into smaller, easier questions. The system answers each small question, then combines the answers.

For example, if someone asks, "Who made more important contributions to modern physics, Albert Einstein or Niels Bohr?" you can split it into:

- **Subquery 1**: "What did Albert Einstein contribute to modern physics?"
- **Subquery 2**: "What did Niels Bohr contribute to modern physics?"

The answers might include:

- For Einstein: the theory of relativity, the photoelectric effect, and _E=mc^2_.
- For Bohr: the hydrogen atom model, work on quantum mechanics, and the principle of complementarity.

You can then ask follow-up questions:

- **Subquery 3**: "How did Einstein's theories change modern physics?"
- **Subquery 4**: "How did Bohr's theories change modern physics?"

These follow-ups look at each scientist’s effect, like:

- How Einstein’s work led to new ideas in cosmology and quantum theory
- How Bohr’s work helped us understand atoms and quantum mechanics

The system combines the answers to give a full response to the original question. This method makes complex questions easier to answer by breaking them into clear, smaller parts.

### Query router

Sometimes, your content lives in several databases or search systems. In these cases, use a query router. A _query router_ picks the best database or index to answer each question.

A query router works after the user asks a question but before the system searches for answers.

Here’s how a query router works:

1. **Query analysis**: The LLM or another tool looks at the question to figure out what kind of answer is needed.
2. **Index selection**: The router picks one or more indexes that fit the question. Some indexes are better for facts, others for opinions or special topics.
3. **Query dispatch**: The router sends the question to the chosen index or indexes.
4. **Results aggregation**: The system collects and combines the answers from the indexes.
5. **Answer generation**: The system creates a clear answer using the information it found.

Use different indexes or search engines for:

- **Data type specialization**: Some indexes focus on news, others on academic papers, or on special databases like medical or legal info.
- **Query type optimization**: Some indexes are fast for simple facts (like dates), while others handle complex or expert questions.
- **Algorithmic differences**: Different search engines use different methods, like vector search, keyword search, or advanced semantic search.

For example, in a medical advice system, you might have:

- A research paper index for technical details
- A case study index for real-world examples
- A general health index for basic questions

If someone asks about the effects of a new drug, the router sends the question to the research paper index. If the question is about common symptoms, it uses the general health index for a simple answer.

### Post-retrieval processing steps

Post-retrieval processing happens after the system finds content chunks in the vector database:https://learn.microsoft.com/en-us/azure/developer/ai/media/advanced-rag-post-retrieval-processing-steps-highres.png

Next, check if these chunks are useful for the LLM prompt before sending them to the LLM.

Keep these things in mind:

- Extra information can hide the most important details.
- Irrelevant information can make the answer worse.

Watch out for the _needle in a haystack_ problem: LLMs often pay more attention to the start and end of a prompt than the middle.

Also, remember the LLM’s maximum context window and the number of tokens needed for long prompts, especially at scale.

To handle these issues, use a post-retrieval processing pipeline with steps like:

- **Filtering results**: Only keep chunks that match the query. Ignore the rest when building the LLM prompt.
- **Re-ranking**: Put the most relevant chunks at the beginning and end of the prompt.
- **Prompt compression**: Use a small, cheap model to summarize and combine chunks into a single prompt before sending it to the LLM.

### Post-completion processing steps

Post-completion processing happens after the user’s question and all content chunks go to the LLM:https://learn.microsoft.com/en-us/azure/developer/ai/media/advanced-rag-post-completion-processing-steps-highres.png

After the LLM gives an answer, check its accuracy. A post-completion processing pipeline can include:

- **Fact check**: Look for statements in the answer that claim to be facts, then check if they’re true. If a fact check fails, you can ask the LLM again or show an error message.
- **Policy check**: Make sure the answer doesn’t include harmful content for the user or your organization.

</details>

<details>
<summary>from-local-to-global-a-graphrag-approach-to-query-focused-su</summary>

Retrieval augmented generation (RAG) (Lewis et al.,, [2020](https://arxiv.org/html/2404.16130v2#bib.bib32 "")) is an established approach to using LLMs to answer queries based on data that is too large to contain in a language model’s _context window_, meaning the maximum number of _tokens_ (units of text) that can be processed by the LLM at once  (Liu et al.,, [2023](https://arxiv.org/html/2404.16130v2#bib.bib33 ""); Kuratov et al.,, [2024](https://arxiv.org/html/2404.16130v2#bib.bib29 "")).
In the canonical RAG setup, the system has access to a large external corpus of text records and retrieves a subset of records that are individually relevant to the query and collectively small enough to fit into the context window of the LLM. The LLM then generates a response based on both the query and the retrieved records (Dang,, [2006](https://arxiv.org/html/2404.16130v2#bib.bib12 ""); Yao et al.,, [2017](https://arxiv.org/html/2404.16130v2#bib.bib71 ""); Baumel et al.,, [2018](https://arxiv.org/html/2404.16130v2#bib.bib6 ""); Laskar et al.,, [2020](https://arxiv.org/html/2404.16130v2#bib.bib31 "")).
This conventional approach, which we collectively call _vector RAG_, works well for queries that can be answered with information localized within a small set of records.
However, vector RAG approaches do not support sensemaking queries, meaning queries that require global understanding of the entire dataset, such as ” _What are the key trends in how scientific discoveries are influenced by interdisciplinary research over the past decade?_”

Sensemaking tasks require reasoning over “ _connections_
_(which can be among people, places, and events) in order to anticipate their trajectories and act effectively_” (Klein et al.,, [2006](https://arxiv.org/html/2404.16130v2#bib.bib27 "")).
LLMs such as GPT (Brown et al.,, [2020](https://arxiv.org/html/2404.16130v2#bib.bib8 ""); Achiam et al.,, [2023](https://arxiv.org/html/2404.16130v2#bib.bib1 "")), Llama (Touvron et al.,, [2023](https://arxiv.org/html/2404.16130v2#bib.bib61 "")), and Gemini (Anil et al.,, [2023](https://arxiv.org/html/2404.16130v2#bib.bib2 "")) excel at sensemaking in complex domains like scientific discovery (Microsoft,, [2023](https://arxiv.org/html/2404.16130v2#bib.bib41 "")) and intelligence analysis (Ranade and Joshi,, [2023](https://arxiv.org/html/2404.16130v2#bib.bib51 "")).
Given a sensemaking query and a text with an implicit and interconnected set of concepts, an LLM can generate a summary that answers the query.
The challenge, however, arises when the volume of data requires a RAG approach, since vector RAG approaches are unable to support sensemaking over an entire corpus.

In this paper, we present GraphRAG – a graph-based RAG approach that enables sensemaking over the entirety of a large text corpus.
GraphRAG first uses an LLM to construct a knowledge graph, where nodes correspond to key entities in the corpus and edges represent relationships between those entities.
Next, it partitions the graph into a hierarchy of communities of closely related entities, before using an LLM to generate community-level summaries. These summaries are generated in a bottom-up manner following the hierarchical structure of extracted communities, with summaries at higher levels of the hierarchy recursively incorporating lower-level summaries.
Together, these community summaries provide global descriptions and insights over the corpus.
Finally, GraphRAG answers queries through map-reduce processing of community summaries; in the map step, the summaries are used to provide partial answers to the query independently and in parallel, then in the reduce step, the partial answers are combined and used to generate a final global answer.

The GraphRAG method and its ability to perform global sensemaking over an entire corpus form the main contribution of this work. To demonstrate this ability, we developed a novel application of the LLM-as-a-judge technique  (Zheng et al.,, [2024](https://arxiv.org/html/2404.16130v2#bib.bib78 "")) suitable for questions targeting broad issues and themes where there is no ground-truth answer.
This approach first uses one LLM to generate a diverse set of global sensemaking questions based on corpus-specific use cases, before using a second LLM to judge the answers of two different RAG systems using predefined criteria (defined in [Section 3.3](https://arxiv.org/html/2404.16130v2#S3.SS3 "3.3 Criteria for Evaluating Global Sensemaking ‣ 3 Methods ‣ From Local to Global: A GraphRAG Approach to Query-Focused Summarization")).
We use this approach to compare GraphRAG to vector RAG on two representative real-world text datasets.
Results show GraphRAG strongly outperforms vector RAG when using GPT-4 as the LLM.

GraphRAG is available as open-source software at [https://github.com/microsoft/graphrag](https://github.com/microsoft/graphrag "").
In addition, versions of the GraphRAG approach are also available as extensions to multiple open-source libraries, including LangChain (LangChain,, [2024](https://arxiv.org/html/2404.16130v2#bib.bib30 "")), LlamaIndex (LlamaIndex,, [2024](https://arxiv.org/html/2404.16130v2#bib.bib34 "")), NebulaGraph (NebulaGraph,, [2024](https://arxiv.org/html/2404.16130v2#bib.bib43 "")), and Neo4J (Neo4J,, [2024](https://arxiv.org/html/2404.16130v2#bib.bib44 "")).

## 2 Background

### 2.1 RAG Approaches and Systems

RAG generally refers to any system where a user query is used to retrieve relevant information from external data sources, whereupon this information is incorporated into the generation of a response to the query by an LLM (or other generative AI model, such as a multi-media model).
The query and retrieved records populate a prompt template, which is then passed to the LLM (Ram et al.,, [2023](https://arxiv.org/html/2404.16130v2#bib.bib50 "")).
RAG is ideal when the total number of records in a data source is too large to include in a single prompt to the LLM, i.e. the amount of text in the data source exceeds the LLM’s context window.

In canonical RAG approaches, the retrieval process returns a set number of records that are semantically similar to the query and the generated answer uses only the information in those retrieved records.
A common approach to conventional RAG is to use text embeddings, retrieving records closest to the query in vector space where closeness corresponds to semantic similarity (Gao et al.,, [2023](https://arxiv.org/html/2404.16130v2#bib.bib18 "")).
While some RAG approaches may use alternative retrieval mechanisms, we collectively refer to the family of conventional approaches as _vector RAG_.
GraphRAG contrasts with vector RAG in its ability to answer queries that require global sensemaking over the entire data corpus.

GraphRAG builds upon prior work on advanced RAG strategies.
GraphRAG leverages summaries over large sections of the data source as a form of ”self-memory” (described in Cheng et al., [2024](https://arxiv.org/html/2404.16130v2#bib.bib9 "")), which are later used to answer queries as in Mao et al., [2020](https://arxiv.org/html/2404.16130v2#bib.bib37 "")). These summaries are generated in parallel and iteratively aggregated into global summaries, similar to prior techniques (Shao et al.,, [2023](https://arxiv.org/html/2404.16130v2#bib.bib55 ""); Wang et al.,, [2024](https://arxiv.org/html/2404.16130v2#bib.bib66 ""); Su et al.,, [2020](https://arxiv.org/html/2404.16130v2#bib.bib58 ""); Feng et al.,, [2023](https://arxiv.org/html/2404.16130v2#bib.bib16 ""); Trivedi et al.,, [2022](https://arxiv.org/html/2404.16130v2#bib.bib64 ""); Khattab et al.,, [2022](https://arxiv.org/html/2404.16130v2#bib.bib24 ""); Gao et al.,, [2023](https://arxiv.org/html/2404.16130v2#bib.bib18 "")).
In particular, GraphRAG is similar to other approaches that use hierarchical indexing to create summaries (similar to Kim et al., [2023](https://arxiv.org/html/2404.16130v2#bib.bib26 ""); Sarthi et al., [2024](https://arxiv.org/html/2404.16130v2#bib.bib53 "")).
GraphRAG contrasts with these approaches by generating a graph index from the source data, then applying graph-based community detection to create a thematic partitioning of the data.

### 2.2 Using Knowledge Graphs with LLMs and RAG

Approaches to knowledge graph extraction from natural language text corpora include rule-matching, statistical pattern recognition, clustering, and embeddings (Yates et al.,, [2007](https://arxiv.org/html/2404.16130v2#bib.bib73 ""); Mooney and Bunescu,, [2005](https://arxiv.org/html/2404.16130v2#bib.bib42 ""); Kim et al.,, [2016](https://arxiv.org/html/2404.16130v2#bib.bib25 ""); Etzioni et al.,, [2004](https://arxiv.org/html/2404.16130v2#bib.bib15 "")).
GraphRAG falls into a more recent body of research that use of LLMs for knowledge graph extraction (Melnyk et al.,, [2022](https://arxiv.org/html/2404.16130v2#bib.bib39 ""); Tan et al.,, [2017](https://arxiv.org/html/2404.16130v2#bib.bib59 ""); OpenAI,, [2023](https://arxiv.org/html/2404.16130v2#bib.bib47 ""); Ban et al.,, [2023](https://arxiv.org/html/2404.16130v2#bib.bib4 ""); [Zhang et al., 2024a,](https://arxiv.org/html/2404.16130v2#bib.bib76 ""); Trajanoska et al.,, [2023](https://arxiv.org/html/2404.16130v2#bib.bib63 ""); Yao et al.,, [2023](https://arxiv.org/html/2404.16130v2#bib.bib72 ""); Yates et al.,, [2007](https://arxiv.org/html/2404.16130v2#bib.bib73 "")).
It also adds to a growing body of RAG approaches that use a knowledge graph as an index (Gao et al.,, [2023](https://arxiv.org/html/2404.16130v2#bib.bib18 "")).
Some techniques use subgraphs, elements of the graph, or properties of the graph structure directly in the prompt  (Baek et al.,, [2023](https://arxiv.org/html/2404.16130v2#bib.bib3 ""); He et al.,, [2024](https://arxiv.org/html/2404.16130v2#bib.bib19 ""); Zhang,, [2023](https://arxiv.org/html/2404.16130v2#bib.bib75 "")) or as factual grounding for generated outputs (Kang et al.,, [2023](https://arxiv.org/html/2404.16130v2#bib.bib23 ""); Ranade and Joshi,, [2023](https://arxiv.org/html/2404.16130v2#bib.bib51 "")).
Other techniques ( [Wang et al., 2023b,](https://arxiv.org/html/2404.16130v2#bib.bib68 "")) use the knowledge graph to enhance retrieval, where at query time an LLM-based agent dynamically traverses a graph with nodes representing document elements (e.g., passages, tables) and edges encoding lexical and semantical similarity or structural relationships.
GraphRAG contrasts with these approaches by focusing on a previously unexplored quality of graphs in this context: their inherent _modularity_(Newman,, [2006](https://arxiv.org/html/2404.16130v2#bib.bib45 "")) and the ability to partition graphs into nested modular communities of closely related nodes (e.g., Louvain, Blondel et al., [2008](https://arxiv.org/html/2404.16130v2#bib.bib7 ""); Leiden, Traag et al., [2019](https://arxiv.org/html/2404.16130v2#bib.bib62 "")).
Specifically, GraphRAG recursively creates increasingly global summaries by using the LLM to create summaries spanning this community hierarchy.

</details>

<details>
<summary>introducing-contextual-retrieval-anthropic</summary>

For an AI model to be useful in specific contexts, it often needs access to background knowledge. For example, customer support chatbots need knowledge about the specific business they're being used for, and legal analyst bots need to know about a vast array of past cases.

Developers typically enhance an AI model's knowledge using Retrieval-Augmented Generation (RAG). RAG is a method that retrieves relevant information from a knowledge base and appends it to the user's prompt, significantly enhancing the model's response. The problem is that traditional RAG solutions remove context when encoding information, which often results in the system failing to retrieve the relevant information from the knowledge base.

In this post, we outline a method that dramatically improves the retrieval step in RAG. The method is called “Contextual Retrieval” and uses two sub-techniques: Contextual Embeddings and Contextual BM25. This method can reduce the number of failed retrievals by 49% and, when combined with reranking, by 67%. These represent significant improvements in retrieval accuracy, which directly translates to better performance in downstream tasks.

### A note on simply using a longer prompt

Sometimes the simplest solution is the best. If your knowledge base is smaller than 200,000 tokens (about 500 pages of material), you can just include the entire knowledge base in the prompt that you give the model, with no need for RAG or similar methods.

However, as your knowledge base grows, you'll need a more scalable solution. That’s where Contextual Retrieval comes in.

## A primer on RAG: scaling to larger knowledge bases

For larger knowledge bases that don't fit within the context window, RAG is the typical solution. RAG works by preprocessing a knowledge base using the following steps:

1. Break down the knowledge base (the “corpus” of documents) into smaller chunks of text, usually no more than a few hundred tokens;
2. Use an embedding model to convert these chunks into vector embeddings that encode meaning;
3. Store these embeddings in a vector database that allows for searching by semantic similarity.

At runtime, when a user inputs a query to the model, the vector database is used to find the most relevant chunks based on semantic similarity to the query. Then, the most relevant chunks are added to the prompt sent to the generative model.

While embedding models excel at capturing semantic relationships, they can miss crucial exact matches. Fortunately, there’s an older technique that can assist in these situations. BM25 (Best Matching 25) is a ranking function that uses lexical matching to find precise word or phrase matches. It's particularly effective for queries that include unique identifiers or technical terms.

BM25 works by building upon the TF-IDF (Term Frequency-Inverse Document Frequency) concept. TF-IDF measures how important a word is to a document in a collection. BM25 refines this by considering document length and applying a saturation function to term frequency, which helps prevent common words from dominating the results.

Here’s how BM25 can succeed where semantic embeddings fail: Suppose a user queries "Error code TS-999" in a technical support database. An embedding model might find content about error codes in general, but could miss the exact "TS-999" match. BM25 looks for this specific text string to identify the relevant documentation.

RAG solutions can more accurately retrieve the most applicable chunks by combining the embeddings and BM25 techniques using the following steps:

1. Break down the knowledge base (the "corpus" of documents) into smaller chunks of text, usually no more than a few hundred tokens;
2. Create TF-IDF encodings and semantic embeddings for these chunks;
3. Use BM25 to find top chunks based on exact matches;
4. Use embeddings to find top chunks based on semantic similarity;
5. Combine and deduplicate results from (3) and (4) using rank fusion techniques;
6. Add the top-K chunks to the prompt to generate the response.

By leveraging both BM25 and embedding models, traditional RAG systems can provide more comprehensive and accurate results, balancing precise term matching with broader semantic understanding.https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F45603646e979c62349ce27744a940abf30200d57-3840x2160.png&w=3840&q=75A Standard Retrieval-Augmented Generation (RAG) system that uses both embeddings and Best Match 25 (BM25) to retrieve information. TF-IDF (term frequency-inverse document frequency) measures word importance and forms the basis for BM25.

This approach allows you to cost-effectively scale to enormous knowledge bases, far beyond what could fit in a single prompt. But these traditional RAG systems have a significant limitation: they often destroy context.

### The context conundrum in traditional RAG

In traditional RAG, documents are typically split into smaller chunks for efficient retrieval. While this approach works well for many applications, it can lead to problems when individual chunks lack sufficient context.

For example, imagine you had a collection of financial information (say, U.S. SEC filings) embedded in your knowledge base, and you received the following question: _"What was the revenue growth for ACME Corp in Q2 2023?"_

A relevant chunk might contain the text: _"The company's revenue grew by 3% over the previous quarter."_ However, this chunk on its own doesn't specify which company it's referring to or the relevant time period, making it difficult to retrieve the right information or use the information effectively.

## Introducing Contextual Retrieval

Contextual Retrieval solves this problem by prepending chunk-specific explanatory context to each chunk before embedding (“Contextual Embeddings”) and creating the BM25 index (“Contextual BM25”).

Let’s return to our SEC filings collection example. Here's an example of how a chunk might be transformed:

```plaintext
original_chunk = "The company's revenue grew by 3% over the previous quarter."

contextualized_chunk = "This chunk is from an SEC filing on ACME corp's performance in Q2 2023; the previous quarter's revenue was $314 million. The company's revenue grew by 3% over the previous quarter."
```

It is worth noting that other approaches to using context to improve retrieval have been proposed in the past. Other proposals include: [adding generic document summaries to chunks](https://aclanthology.org/W02-0405.pdf) (we experimented and saw very limited gains), [hypothetical document embedding](https://arxiv.org/abs/2212.10496), and [summary-based indexing](https://www.llamaindex.ai/blog/a-new-document-summary-index-for-llm-powered-qa-systems-9a32ece2f9ec) (we evaluated and saw low performance). These methods differ from what is proposed in this post.

### Implementing Contextual Retrieval

Of course, it would be far too much work to manually annotate the thousands or even millions of chunks in a knowledge base. To implement Contextual Retrieval, we turn to Claude. We’ve written a prompt that instructs the model to provide concise, chunk-specific context that explains the chunk using the context of the overall document. We used the following Claude 3 Haiku prompt to generate context for each chunk:

```plaintext
<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
```

The resulting contextual text, usually 50-100 tokens, is prepended to the chunk before embedding it and before creating the BM25 index.

Here’s what the preprocessing flow looks like in practice:https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F2496e7c6fedd7ffaa043895c23a4089638b0c21b-3840x2160.png&w=3840&q=75_Contextual Retrieval is a preprocessing technique that improves retrieval accuracy._

### Using Prompt Caching to reduce the costs of Contextual Retrieval

Contextual Retrieval is uniquely possible at low cost with Claude, thanks to the special prompt caching feature we mentioned above. With prompt caching, you don’t need to pass in the reference document for every chunk. You simply load the document into the cache once and then reference the previously cached content. Assuming 800 token chunks, 8k token documents, 50 token context instructions, and 100 tokens of context per chunk, **the one-time cost to generate contextualized chunks is $1.02 per million document tokens**.

#### Methodology

We experimented across various knowledge domains (codebases, fiction, ArXiv papers, Science Papers), embedding models, retrieval strategies, and evaluation metrics.

The graphs below show the average performance across all knowledge domains with the top-performing embedding configuration (Gemini Text 004) and retrieving the top-20-chunks. We use 1 minus recall@20 as our evaluation metric, which measures the percentage of relevant documents that fail to be retrieved within the top 20 chunks. You can see the full results in the appendix - contextualizing improves performance in every embedding-source combination we evaluated.

#### Performance improvements

Our experiments showed that:

- **Contextual Embeddings reduced the top-20-chunk retrieval failure rate by 35%** (5.7% → 3.7%).
- **Combining Contextual Embeddings and Contextual BM25 reduced the top-20-chunk retrieval failure rate by 49%** (5.7% → 2.9%).https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F7f8d739e491fe6b3ba0e6a9c74e4083d760b88c9-3840x2160.png&w=3840&q=75_Combining Contextual Embedding and Contextual BM25 reduce the top-20-chunk retrieval failure rate by 49%._

#### Implementation considerations

When implementing Contextual Retrieval, there are a few considerations to keep in mind:

1. **Chunk boundaries:** Consider how you split your documents into chunks. The choice of chunk size, chunk boundary, and chunk overlap can affect retrieval performance1.
2. **Embedding model:** Whereas Contextual Retrieval improves performance across all embedding models we tested, some models may benefit more than others. We found [Gemini](https://ai.google.dev/gemini-api/docs/embeddings) and [Voyage](https://www.voyageai.com/) embeddings to be particularly effective.
3. **Custom contextualizer prompts:** While the generic prompt we provided works well, you may be able to achieve even better results with prompts tailored to your specific domain or use case (for example, including a glossary of key terms that might only be defined in other documents in the knowledge base).
4. **Number of chunks:** Adding more chunks into the context window increases the chances that you include the relevant information. However, more information can be distracting for models so there's a limit to this. We tried delivering 5, 10, and 20 chunks, and found using 20 to be the most performant of these options (see appendix for comparisons) but it’s worth experimenting on your use case.

**Always run evals:** Response generation may be improved by passing it the contextualized chunk and distinguishing between what is context and what is the chunk.

## Further boosting performance with Reranking

In a final step, we can combine Contextual Retrieval with another technique to give even more performance improvements. In traditional RAG, the AI system searches its knowledge base to find the potentially relevant information chunks. With large knowledge bases, this initial retrieval often returns a lot of chunks—sometimes hundreds—of varying relevance and importance.

Reranking is a commonly used filtering technique to ensure that only the most relevant chunks are passed to the model. Reranking provides better responses and reduces cost and latency because the model is processing less information. The key steps are:

1. Perform initial retrieval to get the top potentially relevant chunks (we used the top 150);
2. Pass the top-N chunks, along with the user's query, through the reranking model;
3. Using a reranking model, give each chunk a score based on its relevance and importance to the prompt, then select the top-K chunks (we used the top 20);
4. Pass the top-K chunks into the model as context to generate the final result.https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F8f82c6175a64442ceff4334b54fac2ab3436a1d1-3840x2160.png&w=3840&q=75_Combine Contextual Retrieva and Reranking to maximize retrieval accuracy._

### Performance improvements

There are several reranking models on the market. We ran our tests with the [Cohere reranker](https://cohere.com/rerank). Voyage [also offers a reranker](https://docs.voyageai.com/docs/reranker), though we did not have time to test it. Our experiments showed that, across various domains, adding a reranking step further optimizes retrieval.

Specifically, we found that Reranked Contextual Embedding and Contextual BM25 reduced the top-20-chunk retrieval failure rate by 67% (5.7% → 1.9%).https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F93a70cfbb7cca35bb8d86ea0a23bdeeb699e8e58-3840x2160.png&w=3840&q=75_Reranked Contextual Embedding and Contextual BM25 reduces the top-20-chunk retrieval failure rate by 67%._

#### Cost and latency considerations

One important consideration with reranking is the impact on latency and cost, especially when reranking a large number of chunks. Because reranking adds an extra step at runtime, it inevitably adds a small amount of latency, even though the reranker scores all the chunks in parallel. There is an inherent trade-off between reranking more chunks for better performance vs. reranking fewer for lower latency and cost. We recommend experimenting with different settings on your specific use case to find the right balance.

## Conclusion

We ran a large number of tests, comparing different combinations of all the techniques described above (embedding model, use of BM25, use of contextual retrieval, use of a reranker, and total # of top-K results retrieved), all across a variety of different dataset types. Here’s a summary of what we found:

1. Embeddings+BM25 is better than embeddings on their own;
2. Voyage and Gemini have the best embeddings of the ones we tested;
3. Passing the top-20 chunks to the model is more effective than just the top-10 or top-5;
4. Adding context to chunks improves retrieval accuracy a lot;
5. Reranking is better than no reranking;
6. **All these benefits stack**: to maximize performance improvements, we can combine contextual embeddings (from Voyage or Gemini) with contextual BM25, plus a reranking step, and adding the 20 chunks to the prompt.

#### Footnotes

1\. For additional reading on chunking strategies, check out [this link](https://www.pinecone.io/learn/chunking-strategies/) and [this link](https://research.trychroma.com/evaluating-chunking).

</details>

<details>
<summary>rag-fundamentals-first-by-paul-iusztin-decoding-ml</summary>

To build successful and complex RAG applications, you must first deeply understand the fundamentals behind them. _In this article, we will learn why we use RAG and how to design the architecture of your RAG layer._

Retrieval-augmented generation (RAG) enhances the accuracy and reliability of generative AI models with information fetched from external sources. It is a technique complementary to the internal knowledge of the LLMs. Before going into the details, let’s understand what RAG stands for:

- **Retrieval:** search for relevant data;

- **Augmented:** add the data as context to the prompt;

- **Generation:** use the augmented prompt with an LLM for generation.


Any LLM is bound to understand the data it was trained on, sometimes called parameterized knowledge. Thus, even if the LLM can perfectly answer what happened in the past, it won’t have access to the newest data or any other external sources on which it wasn’t trained.

Let’s take the most powerful model from OpenAI as an example, which in the summer of 2024 is GPT-4o. The model is trained on data up to Oct 2023. Thus, if we ask what happened during the 2020 pandemic, it can be answered perfectly due to its parametrized knowledge. However, it will not know the answer if we ask about the 2024 soccer EURO cup results due to its bounded parametrized knowledge. Another scenario is that it will start confidently hallucinating and provide a faulty answer.

RAG overcomes these two limitations of LLMs. It provides access to external or latest data and prevents hallucinations, enhancing generative AI models’ accuracy and reliability.

* * *

## **Why use RAG?**

We briefly explained the importance of using RAG in generative AI applications above. Now, we will dig deeper into the “why”. Next, we will focus on what a naïve RAG framework looks like.

For now, to get an intuition about RAG, you have to know that when using RAG, we inject the necessary information into the prompt to answer the initial user question. After, we pass the augmented prompt to the LLM for the final answer. Now the LLM will use the additional context to answer the user question.

There are two fundamental problems that RAG solves:

1\. Hallucinations

2\. Old or private information

### **Hallucinations**

If a chatbot without RAG is asked a question about something it wasn’t trained on, there are big changes that will give you a confident answer about something that isn’t true. Let’s take the 2024 soccer EURO Cup as an example. If the model is trained up to Oct 2023 and we ask something about the tournament, it will most likely come up with some random answer that is hard to differentiate from reality.

Even if the LLM doesn’t hallucinate all the time, it raises the concern of trust in its answers. Thus, we have to ask ourselves: “When can we trust the LLM’s answers?” and “How can we evaluate if the answers are correct?”

By introducing RAG, we will enforce the LLM to always answer solely based on the introduced context. The LLM will act as the reasoning engine, while the additional information added through RAG will act as the single source of truth for the generated answer.

By doing so, we can quickly evaluate if the LLM’s answer is based on the external data or not.

### **Old information**

Any LLM is trained or fine-tuned on a subset of the total world knowledge dataset. This is due to three main issues:

- **Private data:** You cannot train your model on data you don’t own or have the right to use.

- **New data**: New data is generated every second. Thus, you would have to constantly train your LLM to keep up.

- **Costs:** Training or fine-tuning an LLM is an extremely costly operation. Hence, it is not feasible to do it on an hourly or daily basis.


RAG solved these issues, as you no longer have to constantly fine-tune your LLM on new data (or even private data). Directly injecting the necessary data to respond to user questions into the prompts that are fed to the LLM is enough to generate correct and valuable answers.

## **The vanilla RAG framework**

Every RAG system is similar at its roots. We will first focus on understanding RAG in its simplest form. Later, we will gradually introduce more advanced RAG techniques to improve the system’s accuracy.

A RAG system is composed of three main modules independent of each other:

1. **Ingestion pipeline:** A batch or streaming pipeline used to populate the vector DB.

2. **Retrieval pipeline:** A module that queries the vector DB and retrieves relevant entries to the user’s input.

3. **Generation pipeline:** The layer that uses the retrieved data to augment the prompt and an LLM to generate answers.


As these three components are classes or services of their own, we will dig into each separately.

_But how are these three modules connected?_ Here is a very simplistic overview:

1. On the backend side, the ingestion pipeline runs on a schedule or constantly to populate the vector DB with external data.

2. On the client side, the user asks a question.

3. The question is passed to the retrieval module, which pre-processes the user’s input and queries the vector DB.

4. The generation pipelines use a prompt template, user input, and retrieved context to create the prompt.

5. The prompt is passed to an LLM to generate the answer.

6. The answer is shown to the user.


[https://substackcdn.com/image/fetch/$s_!nn9L!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F77a6bf20-e217-4a8c-8df4-f00caa5c51ca_933x933.png](https://substackcdn.com/image/fetch/$s_!nn9L!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F77a6bf20-e217-4a8c-8df4-f00caa5c51ca_933x933.png) Vanilla RAG architecture

### **Ingestion pipeline**

The RAG ingestion pipeline extracts raw documents from various data sources (e.g., data warehouse, data lake, web pages, etc.). Then, it cleans, chunks and embeds the documents. Ultimately, it loads the embedded chunks to a vector DB (or other similar vector storage).

Thus, the RAG ingestion pipeline is split again into the following:

- The **data extraction module** gathers all necessary data from various sources such as databases, APIs, or web pages. This module is highly dependent on your data. It can be as easy as querying your data warehouse or something more complex, such as crawling Wikipedia.

- A **cleaning layer** that standardizes and removes unwanted characters from the extracted data.

- The **chunking module** splits the cleaned documents into smaller ones. As we want to pass the document’s content to an embedding model, this is necessary to ensure it doesn’t exceed the model’s input maximum size. Also, chunking is required to separate specific regions that are semantically related. For example, when chunking a book chapter, the most optimal way is to group similar paragraphs into the same chunk. By doing so, at the retrieval time, you will add only the essential data to the prompt.

- The **embedding component** usesanembedding model to take the chunk’s content (text, images, audio, etc.) and project it into a dense vector packed with semantic value — more on embeddings in the Embeddings models section below.


The **loading module** takes the embedded chunks along with a metadata document. The metadata will contain essential information such as the embedded content, the URL to the source of the chunk, when the content was published on the web, etc. The embedding is used as an index to query similar chunks, while the metadata is used to access the information added to augment the prompt.

At this point, we have an RAG ingestion pipeline that takes raw documents as input, processes them, and populates a vector DB. The next step is to retrieve relevant data from the vector store correctly.

### **Retrieval pipeline**

The retrieval components take the user’s input (text, image, audio, etc.), embed it, and query the vector DB for similar vectors to the user’s input.

The primary function of the retrieval step is to project the user’s input into the same vector space as the embeddings used as an index in the vector DB. This allows us to find the top K’s most similar entries by comparing the embeddings from the vector storage with the user’s input vector. These entries then serve as content to augment the prompt that is passed to the LLM to generate the answer.

You must use a distance metric to compare two vectors, such as the Euclidean or Manhattan distance. But the most popular one is the [cosine distance](https://en.wikipedia.org/wiki/Cosine_similarity) \[1\], which is equal to 1 minus the cosine of the angle between two vectors as follows:

[https://substackcdn.com/image/fetch/$s_!LkC-!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd6d16dbc-de3b-4b57-9f31-07f06799ba57_397x46.jpeg](https://substackcdn.com/image/fetch/$s_!LkC-!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd6d16dbc-de3b-4b57-9f31-07f06799ba57_397x46.jpeg) Cosine distance formula

It ranges from -1 to 1, with a value of -1 when vectors A and B are in opposite directions, 0 if they are orthogonal and 1 if they point in the same direction.

Most of the time, the cosine distance works well in non-linear complex vector spaces. However, it is essential to notice that choosing the proper distance between two vectors depends on your data and the embedding model you use.

One critical factor to highlight is that the user’s input and embeddings must be in the same vector space. Otherwise, you cannot compute the distance between them. To do so, it is essential to pre-process the user input in the same way you processed the raw documents in the RAG ingestion pipeline. It means you must clean, chunk (if necessary), and embed the user’s input using the same functions, models, and hyperparameters. Similar to how you have to pre-process the data into features in the same way between training and inference, otherwise the inference will yield inaccurate results — a phenomenon also known as the training-serving skew.

### **Generation pipeline**

The last step of the RAG system is to take the user’s input and the retrieved data, pass them to an LLM of your choice and generate the answer.

The final prompt results from a prompt template populated with the user’s query and retrieved context. You might have a single or multiple prompt templates depending on your application. Usually, all the prompt engineering is done at the prompt template level.

Each prompt template and LLM should be tracked and versioned using MLOps best practices. Thus, you always know that a given answer was generated by a specific version of the LLM and prompt template(s).

* * *

## **Conclusion**

In this article, we have looked over the fundamentals of RAG.

First, we understood why RAG is so powerful and why many AI applications implement it, as it overcomes challenges such as hallucinations and outdated data.

Secondly, we examined the architecture of a naive RAG system, which consists of an ingestion, retrieval and generation pipeline.

</details>

<details>
<summary>rag-is-dead-long-live-agentic-retrieval-llamaindex-build-kno</summary>

Nowadays, an AI engineer has to be aware of a plethora of techniques and terminology that encompass the data-retrieval aspects of agentic systems: hybrid search, CRAG, Self-RAG, HyDE, deep research, reranking, multi-modal embeddings, and RAPTOR just to name a few.

## Starting with the basics

You can’t talk about RAG without talking about “naive top-k retrieval”. In this basic approach, document chunks are stored in a vector database, and query embeddings are matched with the `k` most similar chunk embeddings.https://www.llamaindex.ai/_next/image?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2F7m9jw85w%2Fproduction%2F188534199027c3232c2c49248b0099295ddce603-576x1210.png%3Ffit%3Dmax%26auto%3Dformat&w=640&q=75Naive top-k RAG

Going slightly beyond this naive `chunk` retrieval mode, there are also two more modes if you want to retrieve the entire contents of relevant files:

- `files_via_metadata` \- use this mode when you want to handle queries where a specific filename or pathname is mentioned e.g. “What does the 2024\_MSFT\_10K.pdf file say about the financial outlook of MSFT?”.
- `files_via_content` \- use this mode when you want to handle queries that are asking general questions about a topic but not a particular set of files e.g. “What is the financial outlook of MSFT?”.https://www.llamaindex.ai/_next/image?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2F7m9jw85w%2Fproduction%2F3e6e040367e14d6b7133820e26531396fa8c1fdd-1072x624.png%3Ffit%3Dmax%26auto%3Dformat&w=1080&q=75Multiple retrieval modes

## Level up: Auto Mode

Now that we have an understanding how and when to use each of our retrieval modes, you’re now equipped with the power to answer any and all of types of questions about your knowledge base!

However, many applications will not know which type of question is being asked beforehand. Most of the time, these questions are being asked by your end user. You will need a way to know which retrieval mode would be most appropriate for the given query.

Enter the 4th retrieval mode - `auto_routed` mode! As the name suggests, this mode uses a lightweight agent to determine which of the other 3 retrieval modes to use for a given query.https://www.llamaindex.ai/_next/image?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2F7m9jw85w%2Fproduction%2Fc2d0bd793627f4d38e93491cfe51ba0f8bad09a5-1316x1050.png%3Ffit%3Dmax%26auto%3Dformat&w=1920&q=75Agentically auto-routed retrieval

## Expanding Beyond a single knowledge base

With the use of `auto_routed` mode, we have a lightweight agentic system that is capable of competently answering a variety of questions. However, this system is somewhat restricted in terms of its search space - it is only able to retrieve data that has been ingested in a single index.

If all of your documents are of the same format (e.g. they’re all just SEC 10K filings), it may be actually be appropriate for you to just ingest all your documents through a single index. The parsing and chunking configurations on that single index can be highly optimized to fit the formatting of this homogenous set of documents. However, your overall knowledge base will surely encompass a wide variety of file formats - SEC Filings, Meeting notes, Customer Service requests, etc. These other formats will necessitate the setup of separate indices whose parsing & chunking settings are optimized to each subset of documents.

## Piecing Together a Knowledge Agent

Now that we know how to use agents for both individual and multi-index level, we can put together a single system that does agentic retrieval at every step of retrieval! Doing so will enable the use of an LLM to optimize every layer of our search path.

The system works like this:

1. At the top layer, the composite retriever uses LLM-based classification to decide which sub-index (or indices) are relevant for the given query.
2. At the sub-index level, the `auto_routed` retrieval mode determines the most appropriate retrieval method (e.g., `chunk`, `files_via_metadata`, or `files_via_content`) for the query.https://www.llamaindex.ai/_next/image?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2F7m9jw85w%2Fproduction%2F9fdb15bafdf8c0921f36c6cd8cdac43c8ca87e27-2232x1562.png%3Ffit%3Dmax%26auto%3Dformat&w=3840&q=75Retrieval routed agentically across multiple auto-routed indexes

This setup ensures that retrieval decisions are intelligently routed at each layer, using LLM-based classification to handle complex queries across multiple indices and retrieval modes. The result is a fully agentic retrieval system capable of adapting dynamically to diverse user queries.

### Naive RAG is dead, agentic retrieval is the future

Agents have become an essential part of modern applications. For these agents to operate effectively and autonomously, they need precise and relevant context at their fingertips. This is why sophisticated data retrieval is crucial for any agent-based system.

</details>

<details>
<summary>the-rise-of-rag</summary>

# The Rise of RAG

### Making LLMs More Reliable Through Additional Information

Good morning everyone! In this iteration, we go back to the basics and explore the popular method called retrieval augmented generation (RAG), introduced by a Meta [paper](https://arxiv.org/abs/2005.11401) in 2020. In one line: RAG answers the known limitations of LLMs, such as non-access to up-to-date information and hallucinations.

Let’s dive into what it really is (simpler than you think), how it works, and when to use it (or not)!

## Understanding RAG

RAG is conceptually simple: It improves LLM responses by adding relevant information to the context window—something the model wouldn't otherwise know. Think of it as giving the LLM access to a vast, curated library of information it can reference in real-time.

A RAG system has two parts. The first is **retrieval**, where we find the most relevant information given a context (the user prompt). The second part is **generation**, where an LLM resolves the user’s request by leveraging the retrieved information as the source. With a good enough LLM, this generation step is straightforward: you combine the retrieved information with the user prompt, and voilà—the LLM produces the response. RAG's “tricky” part is retrieving correct and useful information.

By the way, this concept of search/retrieval is nothing new. There’s been extensive research on retrieval techniques over the years, and you're probably already familiar with this idea—you’ve likely used it every time you search on Google!

A satisfactory RAG system must analyze user prompts and retrieve precisely pertinent information from external sources, such as documents, databases, or multimedia content. This retrieval challenge differentiates a bad RAG implementation from one that provides value to users.

[https://substackcdn.com/image/fetch/$s_!vT0-!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffb26de1d-ca6f-4722-adc3-87c0b18e9c2c_2934x1524.png](https://substackcdn.com/image/fetch/$s_!vT0-!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffb26de1d-ca6f-4722-adc3-87c0b18e9c2c_2934x1524.png)

## Why RAG Matters

By providing LLMs with accurate, relevant information, we improve their ability to generate helpful responses. LLMs are powerful and somewhat intelligent, but they are not accurate sources of knowledge we can blindly trust. At the beginning of the ChatGPT release, there was a [case](https://www.forbes.com/sites/mattnovak/2023/05/27/lawyer-uses-chatgpt-in-federal-court-and-it-goes-horribly-wrong/) where a lawyer cited fictive instances generated by the model in court. Now we know (hopefully) that this is bad!

The RAG approach effectively addresses the two critical limitations of traditional LLMs: knowledge cutoff dates and hallucinations. When an LLM needs to discuss recent events or specific documents, RAG ensures it's working with factual, verifiable information rather than potentially outdated or incorrect knowledge from its training data. Using RAG allows us to leverage an LLM best to manipulate and use language vs. use it to memorize facts. Moreover, RAG enables something important in professional contexts: the ability to cite sources, making responses both traceable and credible.

## Under the Hood: How RAG Works

Much of the work happens in the retrieval component of RAG. The most common approach for unstructured data like PDFs or web content is chunk-based retrieval using semantic similarity—where documents are split into smaller pieces (chunks), converted into numerical representations (embeddings), and then compared mathematically with the user's query embedding to find the most relevant pieces mathematically (semantic similarity).

This can be enhanced through various sophisticated techniques:

- Hierarchical chunking for better context preservation- creates a multi-level structure of chunks (e.g., paragraphs within sections) to maintain broader document context

- Metadata filtering for more precise results - uses document properties like date, author, or category to narrow down the search space before semantic comparison.

- Hybrid search combining traditional methods such as BM25 with semantic similarity - merges keyword-based search with semantic search to leverage the strengths of both approaches

- [Contextual retrieval](https://www.anthropic.com/news/contextual-retrieval) that considers the full query context - enhances each chunk by automatically generating and appending additional context about how the chunk fits within the broader document before creating its embedding.

- Custom-trained embedding models and rerankers - fine-tunes embedding models on domain-specific data and uses additional models to re-score initial search results for better accuracy

- [GraphRAG](https://highlearningrate.substack.com/p/should-you-be-using-graphrag) - introduces knowledge graphs into the retrieval process. Instead of relying solely on vector similarity (comparing numbers to find the most relevant ‘similar’ matches), GraphRAG extracts entities and relationships from your data, creating a structured representation that captures semantic connections.

For structured data, RAG can leverage text-to-SQL approaches, ranging from constrained query generation to full-fledged SQL synthesis. Semantic similarity search, implemented through tools like PGVector, enables efficient retrieval from structured databases.

The effectiveness of these retrieval systems isn't left to chance. We must measure performance through metrics like recall, hit rate, and mean reciprocal rank. Read our iteration on [RAG Evaluation](https://highlearningrate.substack.com/p/rag-evaluation) to learn more about them. However, the actual test lies in user satisfaction – leading organizations increasingly track user feedback to ensure their RAG systems truly enhance the user experience.

## RAG vs. Long Context

As we addressed in a [previous iteration](https://highlearningrate.substack.com/p/the-death-of-rag), while some argue that increasing context windows (like Gemini's 1.5 Flash and Pro, 2M tokens context window) might obviate the need for RAG, the reality is more nuanced. RAG offers several advantages over simply increasing context length by retrieving specific information rather than processing vast amounts of text and enhancing efficiency and speed.

RAG shines when (1) dealing with **large datasets**, (2) where **processing time** is critical, and (3) when the application needs to be **cost-effective**. This is especially important when utilizing an LLM through an API, as sending millions of tokens for every request to provide context can be expensive. In contrast, retrieval is highly efficient and cheap, allowing you to send just the right bits of information to your LLM.

## The Future of RAG: Toward Intelligent Information Retrieval

The next frontier in RAG development lies in handling complex queries through what we might call "agentic" LLM applications. These systems can decompose complex queries into smaller components, launching multiple retrieval processes in parallel or sequence to gather comprehensive information. Ideally, automatically identifying if relevant information is missing before responding to reduce hallucinations. This approach mimics how a human researcher might break down a complex question into smaller, manageable research tasks.

Such systems represent a significant step toward more sophisticated information retrieval and synthesis, potentially revolutionizing how we interact with knowledge bases and document collections.

## Conclusion

RAG isn't just another technical innovation—it's a fundamental shift in approaching AI-powered information coupling retrieval and generation. By bridging the gap between LLMs and real-world knowledge bases, RAG is helping create more reliable, trustworthy, and useful AI systems. As we continue to push the boundaries of what's possible with LLMs, RAG will undoubtedly play a crucial role in shaping the future of AI-powered information systems.

</details>

<details>
<summary>what-is-agentic-rag-ibm</summary>

# What is agentic RAG?

Agentic RAG is the use of [AI agents](https://www.ibm.com/think/topics/ai-agents) to facilitate [retrieval augmented generation (RAG)](https://www.ibm.com/think/topics/retrieval-augmented-generation). Agentic RAG systems add AI agents to the RAG pipeline to increase adaptability and accuracy. Compared to traditional RAG systems, agentic RAG allows [large language models (LLMs)](https://www.ibm.com/think/topics/large-language-models) to conduct [information retrieval](https://www.ibm.com/think/topics/information-retrieval) from multiple sources and handle more complex [workflows](https://www.ibm.com/think/topics/workflow).

### What is RAG?

Retrieval augmented generation is an [artificial intelligence (AI)](https://www.ibm.com/think/topics/artificial-intelligence) application that connects a [generative AI](https://www.ibm.com/think/topics/generative-ai) model with an external knowledge base. The data in the knowledge base augments user queries with more context so the LLM can generate more accurate responses. RAG enables LLMs to be more accurate in domain-specific contexts without needing [fine-tuning](https://www.ibm.com/think/topics/fine-tuning).

Rather than rely solely on training data, RAG-enabled [AI models](https://www.ibm.com/think/topics/ai-model) can access current data in real time through [APIs](https://www.ibm.com/think/topics/api) and other connections to data sources. A standard RAG pipeline comprises two AI models:

- The information retrieval component, typically an [embedding](https://www.ibm.com/think/topics/embedding) model paired with a [vector database](https://www.ibm.com/think/topics/vector-database) containing the data to be retrieved.


- The [generative AI](https://www.ibm.com/think/topics/generative-ai) component, usually an LLM.


In response to [natural language](https://www.ibm.com/think/topics/natural-language-processing) user queries, the embedding model converts the query to a vector embedding, then retrieves similar data from the knowledge base. The AI system combines the retrieved data with the user query for context-aware response generation.

### What is agentic AI?

Agentic AI is a type of AI that can determine and carry out a course of action by itself. Most agents available at the time of publishing are LLMs with function-calling capabilities, meaning that they can call tools to perform tasks. In theory, AI agents are LLMs with three significant characteristics:

- They have **memory**, both short and long term, which enables them to plan and execute complex tasks. Memory also allows agents to refer to previous tasks and use that data to inform future workflows. Agentic RAG systems use semantic caching to store and refer to previous sets of queries, context and results.


- They are capable of query **routing**, step-by-step **planning** and decision-making. Agents use their memory capabilities to retain information and plot an appropriate course of action in response to complex queries and prompts.


- They can perform **tool calling** through APIs. More capable agents can choose which tools to use for the workflow they generate in response to user interactions.


Agentic workflows can consist of either one AI agent or multiagent systems that combine several agents together.

## Agentic RAG vs. traditional RAG systems

Agentic RAG brings several significant improvements over traditional RAG implementation:

- **Flexibility:** Agentic RAG applications pull data from multiple external knowledge bases and allow for external tool use. Standard RAG pipelines connect an LLM to a single external dataset. For example, many enterprise RAG systems pair a [chatbot](https://www.ibm.com/think/topics/chatbots) with a knowledge base containing proprietary organization data.

- **Adaptability:** Traditional RAG systems are reactive data retrieval tools that find relevant information in response to specific queries. There is no ability for the RAG system to adapt to changing contexts or access other data. Optimal results often require extensive [prompt engineering](https://www.ibm.com/think/topics/prompt-engineering).

Meanwhile, agentic RAG is a transition from static rule-based querying to adaptive, intelligent problem-solving. Multiagent systems encourage multiple AI models to collaborate and check each other’s work.

- **Accuracy:** Traditional RAG systems do not validate or optimize their own results. People must discern whether the system is performing at an acceptable standard. The system itself has no way of knowing whether it is finding the right data or successfully incorporating it to facilitate context-aware generation. However, AI agents can iterate on previous processes to optimize results over time.

- **Scalability:** With networks of RAG agents working together, tapping into multiple external data sources and using tool-calling and planning capabilities, agentic RAG has greater scalability. Developers can construct flexible and scalable RAG systems that can handle a wide range of user queries.

- **Multimodality:** Agentic RAG systems benefit from recent advancements in multimodal LLMs to work with a greater range of data types, such as images and audio files. Multimodal models process multiple types of [structured, semistructured and unstructured data](https://www.ibm.com/think/topics/structured-vs-unstructured-data). For example, several recent [GPT](https://www.ibm.com/think/topics/gpt) models can generate visual and audio content in addition to standard text generation.


Consider several employees working in an office. A traditional RAG system is the employee who performs well when given specific tasks and told how to accomplish them. They are reluctant to take initiative and feel uncomfortable going outside explicit instructions.

In comparison, an agentic RAG system is a proactive and creative team. They are also good at following directions but love to take initiative and solve challenges on their own. They are unafraid to come up with their own solutions to complex tasks that might stump or intimidate their coworkers.

### Is agentic RAG better than traditional RAG?

While agentic RAG optimizes results with function calling, multistep reasoning and multiagent systems, it isn’t always the better choice. More agents at work mean greater expenses, and an agentic RAG system usually require paying for more tokens. While agentic RAG can increase speed over traditional RAG, LLMs also introduce latency because it can take more time for the model to generate its outputs.

Lastly, agents are not always reliable. They might struggle and even fail to complete tasks, depending on the complexity and the agents used. Agents do not always collaborate smoothly and can compete over resources. The more agents in a system, the more complex the collaboration becomes, with a higher chance for complications. And even the most airtight RAG system cannot eliminate the potential for hallucinations entirely.

## How does agentic RAG work?

Agentic RAG works by incorporating one or more types of AI agents into RAG systems. For example, an agentic RAG system might combine multiple information retrieval agents, each specialized in a certain domain or type of data source. One agent queries external databases while another can comb through emails and web results.

[Agentic AI frameworks](https://www.ibm.com/think/insights/top-ai-agent-frameworks), such as [LangChain](https://www.ibm.com/think/topics/langchain) and [LlamaIndex](https://www.ibm.com/think/topics/llamaindex), and the orchestration framework [LangGraph](https://www.ibm.com/think/topics/langgraph) can be found on GitHub. With them, it is possible to experiment with [agentic architectures](https://www.ibm.com/think/topics/agentic-architecture) for RAG at minimal costs. If using [open source](https://www.ibm.com/think/topics/open-source) models such as [Granite](https://www.ibm.com/granite) ™ or Llama-3, RAG system designers can also mitigate the fees demanded by other providers such as OpenAI while enjoying greater [observability](https://www.ibm.com/think/topics/observability).

Agentic RAG systems can contain one or more types of AI agents, such as:

- Routing agents


- Query planning agents


- ReAct agents


- Plan-and-execute agents


### Routing agents

Routing agents determine which external knowledge sources and tools are used to address a user query. They process user prompts and identify the RAG pipeline most likely to result in optimal response generation. In a single-agent RAG system, a routing agent chooses which data source to query.

### Query planning agents

Query planning agents are the task managers of the RAG pipeline. They process complex user queries to break them down into step-by-step processes. They submit the resulting subqueries to the other agents in the RAG system, then combine the responses for a cohesive overall response. The process of using one agent to manage other AI models is a type of [AI orchestration](https://www.ibm.com/think/topics/ai-orchestration).

### ReAct agents

ReAct (reasoning and action) is an agent framework that creates [multiagent systems](https://www.ibm.com/think/topics/multiagent-system) that can create and then act on step-by-step solutions. They can also identify appropriate tools that can help. Based on the results of each step, ReAct agents can dynamically adjust subsequent stages of the generated workflow.

### Plan-and-execute agents

Plan-and-execute agent frameworks are a progression from ReAct agents. They can execute multistep workflows without calling back to the primary agent, reducing costs and increasing efficiency. And because the planning agent must reason through all the steps needed for a task, completion rates and quality tend to be higher.

## Agentic RAG use cases

While agentic RAG can suit any traditional RAG application, the greater compute demands make it more appropriate for situations that require querying multiple data sources. Agentic RAG applications include:

- **Real-time question-answering:** Enterprises can deploy RAG-powered chatbots and FAQs to provide employees and customers with current, accurate information.


- **Automated support:** Businesses wanting to streamline customer support services can use automated RAG systems to handle simpler customer inquiries. The agentic RAG system can escalate more demanding support requests to human personnel.


- **Data management:** RAG systems make it easier to find information within proprietary data stores. Employees can quickly get the data they need without having to sort through databases themselves.

</details>

<details>
<summary>what-is-agentic-rag-weaviate</summary>

While Retrieval-Augmented Generation (RAG) dominated 2023, [agentic workflows are driving massive progress in 2024](https://x.com/AndrewYNg/status/1770897666702233815?lang=en). The usage of AI agents opens up new possibilities for building more powerful, robust, and versatile Large Language Model(LLM)-powered applications. One possibility is enhancing RAG pipelines with AI agents in agentic RAG pipelines.

This article introduces you to the concept of agentic RAG, its implementation, and its benefits and limitations.

## Fundamentals of Agentic RAG

Agentic RAG describes an AI agent-based implementation of RAG. Before we go any further, let’s quickly recap the fundamental concepts of RAG and AI agents.

### What is Retrieval-Augmented Generation (RAG)

[Retrieval-Augmented Generation (RAG)](https://weaviate.io/blog/introduction-to-rag) is a technique for building LLM-powered applications. It leverages an external knowledge source to provide the LLM with relevant context and reduce hallucinations.

A naive RAG pipeline consists of a retrieval component (typically composed of an embedding model and a vector database) and a generative component (an LLM). At inference time, the user query is used to run a similarity search over the indexed documents to retrieve the most similar documents to the query and provide the LLM with additional context.https://weaviate.io/assets/images/Vanilla_RAG-697535e2d5b9ae64ccfd6415a79965c7.png

Typical RAG applications have two considerable limitations:

1. The naive RAG pipeline only considers one external knowledge source. However, some solutions might require two external knowledge sources, and some solutions might require external tools and APIs, such as web searches.
2. They are a one-shot solution, which means that context is retrieved once. There is no reasoning or validation over the quality of the retrieved context.

### What are Agents in AI Systems

With the popularity of LLMs, new paradigms of AI agents and multi-agent systems have emerged. AI agents are LLMs with a role and task that have access to memory and external tools. The reasoning capabilities of LLMs help the agent plan the required steps and take action to complete the task at hand.

Thus, the core components of an AI agent are:

- LLM (with a role and a task)
- Memory (short-term and long-term)
- Planning (e.g., reflection, self-critics, query routing, etc.)
- Tools (e.g., calculator, web search, etc.)https://weaviate.io/assets/images/Components_of_an_AI_agent-2f1846374720471d6b11169203ccb865.png

One popular framework is the [ReAct framework](https://arxiv.org/abs/2210.03629). A ReAct agent can handle sequential multi-part queries while maintaining state (in memory) by combining routing, query planning, and tool use into a single entity.

> ReAct = Reason + Act (with LLMs)

The process involves the following steps:

1. Thought: Upon receiving the user query, the agent reasons about the next action to take

2. Action: the agent decides on an action and executes it (e.g., tool use)

3. Observation: the agent observes the feedback from the action

4. This process iterates until the agent completes the task and responds to the user.https://weaviate.io/assets/images/ReAct-6d7ea59cdc1c7f882f860e7015228302.png

## What is Agentic RAG?

Agentic RAG describes an AI agent-based implementation of RAG. Specifically, it incorporates AI agents into the RAG pipeline to orchestrate its components and perform additional actions beyond simple information retrieval and generation to overcome the limitations of the non-agentic pipeline.

> Agentic RAG describes an AI agent-based implementation of RAG.

### How does Agentic RAG work?

Although agents can be incorporated in different stages of the RAG pipeline, agentic RAG most commonly refers to the use of agents in the retrieval component.

Specifically, the retrieval component becomes agentic through the use of retrieval agents with access to different retriever tools, such as:

- Vector search engine (also called a query engine) that performs vector search over a vector index (like in typical RAG pipelines)
- Web search
- Calculator
- Any API to access software programmatically, such as email or chat programs
- and many more.

The RAG agent can then reason and act over the following example retrieval scenarios:

1. Decide whether to retrieve information or not
2. Decide which tool to use to retrieve relevant information
3. Formulate the query itself
4. Evaluate the retrieved context and decide whether it needs to re-retrieve.

### Agentic RAG Architecture

In contrast to the sequential naive RAG architecture, the core of the agentic RAG architecture is the agent. Agentic RAG architectures can have various levels of complexity. In the simplest form, a single-agent RAG architecture is a simple router. However, you can also add multiple agents into a multi-agent RAG architecture. This section discusses the two fundamental RAG architectures.

**Single-Agent RAG (Router)**

In its simplest form, agentic RAG is a router. This means you have at least two external knowledge sources, and the agent decides which one to retrieve additional context from. However, the external knowledge sources don't have to be limited to (vector) databases. You can retrieve further information from tools as well. For example, you can conduct a web search, or you could use an API to retrieve additional information from Slack channels or your email accounts.https://weaviate.io/assets/images/Single_Agent_RAG_System_(Router-ae2ec18616941504070d6b2a7210a358.png)

**Multi-agent RAG Systems**

As you can guess, the single-agent system also has its limitations because it's limited to only one agent with reasoning, retrieval, and answer generation in one. Therefore, it is beneficial to chain multiple agents into a multi-agent RAG application.

For example, you can have one master agent who coordinates information retrieval among multiple specialized retrieval agents. For instance, one agent could retrieve information from proprietary internal data sources. Another agent could specialize in retrieving information from your personal accounts, such as email or chat. Another agent could also specialize in retrieving public information from web searches.https://weaviate.io/assets/images/Multi_Agent_RAG_System-73e480f62a52e172a78a0ac344dcdcb5.png

### Beyond Retrieval Agents

The above example shows the usage of different retrieval agents. However, you could also use agents for purposes other than retrieval. The possibilities of agents in the RAG system are manifold.

## Agentic RAG vs. (Vanilla) RAG

While the fundamental concept of RAG (sending a query, retrieving information, and generating a response) remains the same, **tool use generalizes it,** making it more flexible and powerful.

Think of it this way: Common (vanilla) RAG is like being at the library (before smartphones existed) to answer a specific question. Agentic RAG, on the other hand, is like having a smartphone in your hand with a web browser, a calculator, your emails, etc.

|  | Vanilla RAG | Agentic RAG |
| --- | --- | --- |
| Access to external tools | No | Yes |
| Query pre-processing | No | Yes |
| Multi-step retrieval | No | Yes |
| Validation of retrieved information | No | Yes |

## Implementing Agentic RAG

As outlined earlier, agents are comprised of multiple components. To build an agentic RAG pipeline, there are two options: a language model with function calling or an agent framework. Both implementations get to the same result, it will just depend on the control and flexibility you want.

### Language Models with Function Calling

Language models are the main component of agentic RAG systems. The other component is tools, which enable the language model access to external services. Language models with function calling offer a way to build an agentic system by allowing the model to interact with predefined tools. Language model providers have added this feature to their clients.

In June 2023, [OpenAI released function calling](https://platform.openai.com/docs/assistants/tools/function-calling) for `gpt-3.5-turbo` and `gpt-4`. It enabled these models to reliably connect GPT’s capabilities with external tools and APIs. Developers quickly started building applications that plugged `gpt-4` into code executors, databases, calculators, and more.

[Cohere](https://docs.cohere.com/v2/docs/tool-use) further launched their connectors API to add tools to the Command-R suite of models. Additionally, [Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) and [Google](https://ai.google.dev/gemini-api/docs/function-calling) launched function calling for Claude and Gemini. By powering these models with external services, it can access and cite web resources, execute code and more.

Function calling isn’t only for proprietary models. Ollama introduced tool support for popular open-source models like `Llama3.2`, `nemotron-mini`, and [others](https://ollama.com/search?c=tools).

To build a tool, you first need to define the function. In this snippet, we’re writing a function that is using Weaviate’s [hybrid search](https://docs.weaviate.io/weaviate/search/hybrid) to retrieve objects from the database:

```codeBlockLines_e6Vv
def get_search_results(query: str) -> str:
    """Sends a query to Weaviate's Hybrid Search. Parses the response into a {k}:{v} string."""

    response = blogs.query.hybrid(query, limit=5)

    stringified_response = ""
    for idx, o in enumerate(response.objects):
        stringified_response += f"Search Result: {idx+1}:\n"
        for prop in o.properties:
            stringified_response += f"{prop}:{o.properties[prop]}"
        stringified_response += "\n"

    return stringified_response

```

We will then pass the function to the language model via a `tools_schema`. The schema is then used in the prompt to the language model:

```codeBlockLines_e6Vv
tools_schema=[{\
    'type': 'function',\
    'function': {\
        'name': 'get_search_results',\
        'description': 'Get search results for a provided query.',\
        'parameters': {\
          'type': 'object',\
          'properties': {\
            'query': {\
              'type': 'string',\
              'description': 'The search query.',\
            },\
          },\
          'required': ['query'],\
        },\
    },\
}]

```

Since you’re connecting to the language model API directly, you’ll need to write a loop that routes between the language model and tools:

```codeBlockLines_e6Vv
def ollama_generation_with_tools(user_message: str,
                                 tools_schema: List, tool_mapping: Dict,
                                 model_name: str = "llama3.1") -> str:
    messages=[{\
        "role": "user",\
        "content": user_message\
    }]
    response = ollama.chat(
        model=model_name,
        messages=messages,
        tools=tools_schema
    )
    if not response["message"].get("tool_calls"):
        return response["message"]["content"]
    else:
        for tool in response["message"]["tool_calls"]:
            function_to_call = tool_mapping[tool["function"]["name"]]
            print(f"Calling function {function_to_call}...")
            function_response = function_to_call(tool["function"]["arguments"]["query"])
            messages.append({
                "role": "tool",
                "content": function_response,
            })

    final_response = ollama.chat(model=model_name, messages=messages)
    return final_response["message"]["content"]

```

Your query will then look like this:

```codeBlockLines_e6Vv
ollama_generation_with_tools("How is HNSW different from DiskANN?",
                            tools_schema=tools_schema, tool_mapping=tool_mapping)

```

You can follow along [this recipe](https://github.com/weaviate/recipes/blob/main/integrations/llm-agent-frameworks/function-calling/ollama/ollama-weaviate-agents.ipynb) to recreate the above.

### Agent Frameworks

Agent Frameworks such as DSPy, LangChain, CrewAI, LlamaIndex, and Letta have emerged to facilitate building applications with language models. These frameworks simplify building agentic RAG systems by plugging pre-built templates together.

- DSPy supports [ReAct](https://dspy-docs.vercel.app/deep-dive/modules/react/) agents and [Avatar](https://github.com/stanfordnlp/dspy/blob/main/examples/agents/avatar_langchain_tools.ipynb) optimization. Avatar optimization describes the use of automated prompt engineering for the descriptions of each tool.
- [LangChain](https://www.langchain.com/) provides many services for working with tools. LangChain’s [LCEL](https://python.langchain.com/v0.1/docs/expression_language/) and [LangGraph](https://www.langchain.com/langgraph) frameworks further offer built-in tools.
- [LlamaIndex](https://www.llamaindex.ai/) further introduces the QueryEngineTool, a collection of templates for retrieval tools.
- [CrewAI](https://www.crewai.com/) is one of the leading frameworks for developing multi-agent systems. One of the key concepts utilized for tool use is sharing tools amongst agents.
- [Swarm](https://github.com/openai/swarm) is a framework built by OpenAI for multi-agent orchestration. Swarm similarly focuses on how tools are shared amongst agents.
- [Letta](https://docs.letta.com/introduction) interfaces reflecting and refining an internal world model as functions. This entails potentially using search results to update the agent’s memory of the chatbot user, in addition to responding to the question.

## Why are Enterprises Adopting Agentic RAG

Enterprises are moving on from vanilla RAG to building agentic RAG applications. [Replit released an agent](https://docs.replit.com/replitai/agent) that helps developers build and debug software. Additionally, [Microsoft announced copilots](https://blogs.microsoft.com/blog/2024/10/21/new-autonomous-agents-scale-your-team-like-never-before/) that work alongside users to provide suggestions in completing tasks. These are only a few examples of agents in production and the possibilities are endless.

### Benefits of Agentic RAG

The shift from vanilla RAG to agentic RAG allows these systems to produce more accurate responses, perform tasks autonomously, and better collaborate with humans.

The benefit of agentic RAG lies primarily in the improved quality of retrieved additional information. By adding agents with access to tool use, the retrieval agent can route queries to specialized knowledge sources. Furthermore, the reasoning capabilities of the agent enable a layer of validation of the retrieved context before it is used for further processing. As a result, agentic RAG pipelines can lead to more robust and accurate responses.

### Limitations of Agentic RAG

However, there are always two sides to every coin. Using an AI agent a for subtask means incorporating an LLM to do a task. This comes with the limitations of using LLMs in any application, such as added latency and unreliability. Depending on the reasoning capabilities of the LLM, an agent may fail to complete a task sufficiently (or even at all). It is important to incorporate proper failure modes to help an AI agent get unstuck when they are unable to complete a task.

## Summary

This blog discussed the concept of agentic RAG, which involves incorporating agents into the RAG pipeline. Although agents can be used for many different purposes in a RAG pipeline, agentic RAG most often involves using retrieval agents with access to tools to generalize retrieval.

This article discussed agentic RAG architectures using single-agent and multi-agent systems and their differences from vanilla RAG pipelines.

With the rise and popularity of AI agent systems, many different frameworks are evolving for implementing agentic RAG, such as LlamaIndex, LangGraph, or CrewAI.

Finally, this article discussed the benefits and limitations of agentic RAG pipelines.

</details>

<details>
<summary>what-is-retrieval-augmented-generation-aka-rag-nvidia-blogs</summary>

To understand the latest advancements in generative AI, imagine a courtroom.

Judges hear and decide cases based on their general understanding of the law. Sometimes a case — like a malpractice suit or a labor dispute — requires special expertise, so judges send court clerks to a law library, looking for precedents and specific cases they can cite.

Like a good judge, large language models (LLMs) can respond to a wide variety of human queries. But to deliver authoritative answers — grounded in specific court proceedings or similar ones  — the model needs to be provided that information.

The court clerk of AI is a process called retrieval-augmented generation, or RAG for short.

## **So, What Is Retrieval-Augmented Generation (RAG)?**

Retrieval-augmented generation is a technique for enhancing the accuracy and reliability of generative AI models with information fetched from specific and relevant data sources.

In other words, it fills a gap in how LLMs work. Under the hood, LLMs are neural networks, typically measured by how many parameters they contain. An LLM’s parameters essentially represent the general patterns of how humans use words to form sentences.

That deep understanding, sometimes called parameterized knowledge, makes LLMs useful in responding to general prompts. However, it doesn’t serve users who want a deeper dive into a specific type of information.

## **Combining Internal, External Resources**

Lewis and colleagues developed retrieval-augmented generation to link generative AI services to external resources, especially ones rich in the latest technical details.

The paper, with coauthors from the former Facebook AI Research (now Meta AI), University College London and New York University, called RAG “a general-purpose fine-tuning recipe” because it can be used by nearly any LLM to connect with practically any external resource.

## **Building User Trust**

Retrieval-augmented generation gives models sources they can cite, like footnotes in a research paper, so users can check any claims. That builds trust.

What’s more, the technique can help models clear up ambiguity in a user query. It also reduces the possibility that a model will give a very plausible but incorrect answer, a phenomenon called hallucination.

Another great advantage of RAG is it’s relatively easy. A blog by Lewis and three of the paper’s coauthors said developers can implement the process with as few as five lines of code.

That makes the method faster and less expensive than retraining a model with additional datasets. And it lets users hot-swap new sources on the fly.

## **How People Are Using RAG**

With retrieval-augmented generation, users can essentially have conversations with data repositories, opening up new kinds of experiences. This means the applications for RAG could be multiple times the number of available datasets.

For example, a generative AI model supplemented with a medical index could be a great assistant for a doctor or nurse. Financial analysts would benefit from an assistant linked to market data.

In fact, almost any business can turn its technical or policy manuals, videos or logs into resources called knowledge bases that can enhance LLMs. These sources can enable use cases such as customer or field support, employee training and developer productivity.

## **The History of RAG**

The roots of the technique go back at least to the early 1970s. That’s when researchers in information retrieval prototyped what they called question-answering systems, apps that use natural language processing (NLP) to access text, initially in narrow topics such as baseball.

The concepts behind this kind of text mining have remained fairly constant over the years. But the machine learning engines driving them have grown significantly, increasing their usefulness and popularity.

In the mid-1990s, the Ask Jeeves service, now Ask.com, popularized question answering with its mascot of a well-dressed valet. IBM’s Watson became a TV celebrity in 2011 when it handily beat two human champions on the _Jeopardy!_ game show.

Today, LLMs are taking question-answering systems to a whole new level.

## **How Retrieval-Augmented Generation Works**

At a high level, here’s how retrieval-augmented generation works.

When users ask an LLM a question, the AI model sends the query to another model that converts it into a numeric format so machines can read it. The numeric version of the query is sometimes called an embedding or a vector.

The embedding model then compares these numeric values to vectors in a machine-readable index of an available knowledge base. When it finds a match or multiple matches, it retrieves the related data, converts it to human-readable words and passes it back to the LLM.

Finally, the LLM combines the retrieved words and its own response to the query into a final answer it presents to the user, potentially citing sources the embedding model found.

## **Keeping Sources Current**

In the background, the embedding model continuously creates and updates machine-readable indices, sometimes called vector databases, for new and updated knowledge bases as they become available.

Many developers find LangChain, an open-source library, can be particularly useful in chaining together LLMs, embedding models and knowledge bases.

The future of generative AI lies in agentic AI — where LLMs and knowledge bases are dynamically orchestrated to create autonomous assistants. These AI-driven agents can enhance decision-making, adapt to complex tasks and deliver authoritative, verifiable results for users.

</details>
