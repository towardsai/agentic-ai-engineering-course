# Research

## Research Results

<details>
<summary>What are the performance benchmarks and error analysis for traditional OCR systems on complex documents like financial reports and technical diagrams?</summary>

### Source [1]: https://www.dataunboxed.io/blog/ocr-vs-vlm-ocr-naive-benchmarking-accuracy-for-scanned-documents

Query: What are the performance benchmarks and error analysis for traditional OCR systems on complex documents like financial reports and technical diagrams?

Answer: This source presents a comparative benchmark between **traditional OCR methods** and **Vision Language Models (VLMs)**, focusing on noisy, complex scanned documents using the FUNSD dataset. The evaluation covers multiple metrics such as **text similarity, word error rate, character error rate, and processing time**. Results indicate that VLMs (notably Qwen and Mistral) **outperform traditional OCR in accuracy** for documents with complex layouts and poor scan quality, though they require longer processing times. The analysis highlights that traditional OCR struggles particularly with challenging layouts and low-quality scans, which are common in financial reports and technical diagrams. The study did not utilize any pre-processing, emphasizing the raw capability of OCR methods. Recommendations suggest traditional OCR may be preferable for simpler documents or high-throughput scenarios where speed and cost are prioritized over maximum accuracy.

-----

-----

-----

### Source [2]: https://www.mixedbread.com/blog/the-hidden-ceiling

Query: What are the performance benchmarks and error analysis for traditional OCR systems on complex documents like financial reports and technical diagrams?

Answer: This source describes the "OHR (OCR hinders RAG) Benchmark v2," a comprehensive evaluation of OCR systems on **8,500+ PDF pages** from enterprise domains including finance and technical manuals—documents with **complex layouts, tables, formulas, charts, and diagrams**. All tested OCR solutions performed **below ground truth benchmarks**, establishing an "OCR ceiling" that limits downstream retrieval and question-answering (QA) accuracy. Error analysis revealed specific weaknesses:
- OCR struggled with **non-standard reading orders**, dense tables, and embedded diagrams.
- Even state-of-the-art commercial and open-source OCR solutions (such as Azure Document Intelligence and MinerU) **could not match human-verified extraction**.
- Retrieval and QA tasks were directly impacted by OCR errors, especially in extracting structured data from financial reports and technical diagrams.
- Multimodal models that bypass traditional OCR (retrieving directly from page images) showed superior performance on these complex documents.

-----

-----

-----

### Source [3]: https://www.3rdaiautomation.com/blog/benchmarking-QA-on-complex-indestrial-PDFs

Query: What are the performance benchmarks and error analysis for traditional OCR systems on complex documents like financial reports and technical diagrams?

Answer: This source critiques traditional OCR benchmarks, which typically use **word/character accuracy metrics** in isolation, arguing these do not reflect real-world performance when processing **complex, semi-structured industrial PDFs**. The TIA-pdf-QA-Bench was developed to measure **end-to-end QA performance** over such documents. Key findings include:
- **OCR mistakes in technical diagrams and financial tables** can have outsized effects on downstream applications, such as QA systems, where a single misrecognized term may disrupt answer extraction.
- **Chunking and representation** of extracted text is a major challenge: poor chunking leads to missed answers, irrelevant retrievals, and hallucinated outputs.
- Industrial and technical documents pose unique challenges: heterogeneous formatting, tabular and graphical data, implicit cross-references, and domain-specific language—all of which exacerbate OCR error rates and reduce overall utility.

-----

-----

-----

### Source [4]: https://getomni.ai/blog/ocr-benchmark

Query: What are the performance benchmarks and error analysis for traditional OCR systems on complex documents like financial reports and technical diagrams?

Answer: This benchmark assesses **10 leading OCR providers** and Vision Language Models (VLMs) across 1,000 real-world documents, including those with **charts, infographics, handwriting, and complex input fields**. Key findings:
- **VLMs often match or exceed traditional OCR accuracy** for documents with charts, handwriting, and noisy inputs (e.g., creases, watermarks).
- **Traditional OCR is superior for high-density text pages** (textbooks, research papers) and standardized formats (tax forms).
- Accuracy was measured by JSON field-level difference from ground truth, reflecting practical extraction quality.
- Error analysis shows traditional OCR systems struggle with **complex graphical layouts** and degraded scans but are reliable for dense, regular text.
- VLMs may refuse to process certain sensitive documents due to content policy, introducing unpredictability in supported scenarios.
- Processing time for traditional OCR is generally lower per page compared to VLMs, which are slower due to token output generation.

-----

-----

-----

### Source [5]: https://arxiv.org/html/2412.02210v2

Query: What are the performance benchmarks and error analysis for traditional OCR systems on complex documents like financial reports and technical diagrams?

Answer: This academic source describes the **CC-OCR benchmark**, which evaluates OCR performance across four tracks: Conventional OCR, Multilingual OCR, Document Parsing, and Key Information Extraction. For **complex documents** (including those with formulas, tables, charts, and diverse layouts), challenges highlighted include:
- **Noise factors** such as shadows, lighting variations, folds, material textures, and complex backgrounds severely impact OCR accuracy.
- Structural recognition (such as table and chart analysis, formula recognition) is limited by these factors, reducing robustness of traditional models.
- Most models are primarily tested on scanned documents, but real-world documents (e.g., financial reports, technical diagrams) often have additional noise and layout complexity.
- The benchmark stresses that **current OCR systems are not robust enough** for reliable parsing and extraction from these challenging documents, and further research is needed to handle such scenarios.

-----

-----

</details>

<details>
<summary>How do multimodal LLM architectures like the Unified Embedding Decoder and Cross-modality Attention compare in terms of performance, efficiency, and training complexity?</summary>

### Source [6]: https://magazine.sebastianraschka.com/p/understanding-multimodal-llms

Query: How do multimodal LLM architectures like the Unified Embedding Decoder and Cross-modality Attention compare in terms of performance, efficiency, and training complexity?

Answer: The Unified Embedding Decoder Architecture is typically easier to implement because it does not require modifications to the LLM architecture itself. In this method, images are converted into tokens that match the embedding size of text tokens, allowing the model to process both modalities as a single sequence. This approach leverages a single decoder, similar to unmodified LLMs like GPT-2 or Llama 3.2, and provides straightforward integration by simply concatenating the text and image tokens.

The Cross-modality Attention Architecture, in contrast, uses a cross-attention mechanism to directly integrate image and text embeddings within the attention layer. This requires architectural changes, specifically the addition of cross-attention modules to the transformer blocks, which increases implementation complexity.

In terms of **efficiency**, the Unified Embedding Decoder is generally more scalable and faster due to its simpler architecture. The Cross-modality Attention approach, while potentially offering better alignment between modalities, tends to be more computationally intensive due to the additional attention operations.

For **performance**, the Cross-modality Attention method often achieves higher accuracy on tasks that demand fine-grained understanding between modalities, since it allows for direct modeling of interactions between image and text tokens. However, this comes at the cost of higher **training complexity** and resource requirements.

In summary:
- **Unified Embedding Decoder:** Easier to implement, more efficient, but may lack deep cross-modal alignment.
- **Cross-modality Attention:** More complex and resource-intensive, but can deliver superior performance on tasks needing tight integration between modalities[1].

-----

-----

</details>

<details>
<summary>What are the best practices for handling and storing multimodal data (bytes, Base64, URLs) in enterprise AI applications?</summary>

### Source [11]: https://www.xenonstack.com/blog/multimodal-ai-models-snowflake

Query: What are the best practices for handling and storing multimodal data (bytes, Base64, URLs) in enterprise AI applications?

Answer: Best practices for handling and storing multimodal data in enterprise AI applications using Snowflake include:

- **Centralized Data Lake/Warehouse:** Use Snowflake as the main repository for all multimodal data (such as bytes, Base64, URLs) to eliminate data silos and create a unified data view.
- **Schema-on-Read Approach:** Take advantage of schema-on-read to flexibly ingest multimodal and semi-structured data, defining schemas after ingestion to adapt to evolving formats.
- **Data Virtualization:** For data sources that cannot be physically moved into Snowflake (such as large image or video files), leverage data virtualization to query data in place, which is useful for handling data referenced by URLs.
- **Metadata Management:** Implement robust metadata management by cataloging and documenting all data sources, modalities, features, and pipelines. This improves data lineage, discoverability, and maintainability.
- **Cross-Modal Feature Engineering:** Go beyond simple concatenation of modality-specific features by designing features that capture relationships and interactions between modalities.
- **Use of Transformer Networks and Pre-trained Embeddings:** Prepare data in Snowflake for transformer-based models and multimodal embeddings, which can be stored and accessed efficiently within the platform for downstream AI tasks[1].

-----

-----

-----

### Source [12]: https://www.tribe.ai/applied-ai/multi-modal-ai

Query: What are the best practices for handling and storing multimodal data (bytes, Base64, URLs) in enterprise AI applications?

Answer: Multimodal AI systems in enterprises should be designed to integrate and process diverse data types (including bytes, Base64, URLs) holistically to reveal patterns and relationships that isolated data streams cannot. By combining these sources, the system gains a comprehensive, 360-degree view for decision-making. For example, in healthcare or customer experience, integrating structured and unstructured data types (text, images, audio, video) enables more accurate, context-aware insights.

Enterprises adopting multimodal AI benefit from:
- Enhanced decision-making through cross-referencing information from multiple modalities.
- Greater adaptability of AI solutions to complex, real-world scenarios by processing data in a unified pipeline.
- Competitive advantage through automation and the ability to surface insights traditional siloed approaches would miss[2].

-----

-----

-----

### Source [13]: https://zilliz.com/blog/multimodal-pipelines-for-ai-applications

Query: What are the best practices for handling and storing multimodal data (bytes, Base64, URLs) in enterprise AI applications?

Answer: Best practices for managing multimodal data pipelines in AI applications include:

- **Scalable and Secure Data Management:** Employ advanced data pipeline platforms and vector databases (such as Zilliz and Milvus) to handle diverse data types at scale.
- **Automation of Preprocessing and Metadata Enrichment:** Automate tasks like data preprocessing, embedding generation, and metadata enrichment to reduce manual effort while maintaining accuracy and scalability.
- **Continuous and Automated Data Pipelines:** Ensure your pipelines support real-time synchronization and transformation, allowing for ongoing assessment and improvement of application performance.
- **Integration with Vector Databases:** Storing multimodal data as vector embeddings in specialized databases enables efficient search, retrieval, and downstream AI workflows, regardless of the original storage format (bytes, Base64, URLs)[3].

-----

-----

-----

### Source [14]: https://intervision.com/blog-what-is-multimodal-gen-ai-and-how-can-it-evolve-your-business-strategy/

Query: What are the best practices for handling and storing multimodal data (bytes, Base64, URLs) in enterprise AI applications?

Answer: Implementing multimodal AI in enterprises requires addressing several challenges and following best practices:

- **Data Complexity and Integration:** Use robust strategies for metadata tagging, vector embedding creation, and appropriate storage solutions to manage the complexity of multimodal data sources (including binary files, Base64-encoded data, and URLs).
- **Infrastructure Modernization:** Deploy scalable, high-performance compute environments—often leveraging cloud or hybrid solutions—to efficiently handle demanding multimodal AI workloads.
- **Governance and Compliance:** Establish governance frameworks to ensure privacy, compliance with regulations, and ethical management of multimodal data, particularly when dealing with sensitive or personally identifiable information.
- **Strategic Alignment:** Ensure that AI solutions and associated data management practices are aligned with clearly defined business objectives and prioritized use cases[4].

-----

-----

-----

### Source [15]: https://www.domo.com/learn/article/ai-in-data-management

Query: What are the best practices for handling and storing multimodal data (bytes, Base64, URLs) in enterprise AI applications?

Answer: Best practices for implementing AI in data management, relevant to multimodal data handling, include:

- **Clear Use Cases:** Begin with well-defined business challenges to ensure focused, measurable deployment of AI with multimodal data.
- **Data Quality:** Invest heavily in ensuring data cleanliness and reliability—remove duplicates, standardize formats (including consistent encoding for bytes and Base64), and resolve inconsistencies across data sources.
- **Human Oversight:** Balance automation with human review, especially in handling complex or sensitive data types, ensuring that context and strategy remain aligned with business needs[5].
-----

-----

</details>

<details>
<summary>What is the difference between multimodal LLMs for understanding (like Gemini) versus generative diffusion models for image creation (like Midjourney or Stable Diffusion)?</summary>

### Source [16]: https://blog.google/technology/ai/google-gemini-ai/

Query: What is the difference between multimodal LLMs for understanding (like Gemini) versus generative diffusion models for image creation (like Midjourney or Stable Diffusion)?

Answer: **Gemini** is a multimodal large language model (LLM) designed to *understand* and *reason* across multiple modalities (text, images, audio) natively. Unlike earlier multimodal models that relied on stitching together separate components for each modality, Gemini was trained from the start on diverse data types, enabling seamless cross-modal understanding. Its core capability is sophisticated reasoning over complex written and visual information, extracting insights from vast datasets and providing explanations, especially in subjects like math and physics. Gemini is optimized for *understanding* and *explaining* inputs, not primarily for generating new images or media. In contrast, generative diffusion models like Stable Diffusion and Midjourney are specialized for *creating* images from text prompts, focusing on synthesis rather than comprehension.

-----

-----

-----

### Source [17]: https://arxiv.org/abs/2312.11805

Query: What is the difference between multimodal LLMs for understanding (like Gemini) versus generative diffusion models for image creation (like Midjourney or Stable Diffusion)?

Answer: The **Gemini family** (Ultra, Pro, Nano) represents a new class of highly capable multimodal models excelling at *image, audio, video, and text understanding*. In benchmark evaluations, Gemini Ultra advances the state of the art in most tasks, notably achieving human-expert performance on exams and improving results across 20 multimodal benchmarks. The model's strength lies in *cross-modal reasoning*—it can interpret, relate, and explain content involving multiple input types. Its primary use cases center on *comprehension, reasoning, and language understanding* rather than generative synthesis. This contrasts with generative diffusion models, which are designed to *produce* new images from text prompts using probabilistic sampling and iterative refinement, not advanced understanding or reasoning across modalities.

-----

-----

-----

### Source [18]: https://cloud.google.com/vertex-ai/generative-ai/docs/models

Query: What is the difference between multimodal LLMs for understanding (like Gemini) versus generative diffusion models for image creation (like Midjourney or Stable Diffusion)?

Answer: **Gemini models** in Google’s Vertex AI platform are described as best for *multimodal understanding* and solving complex problems involving various input types (text, images, etc.). The documentation emphasizes Gemini’s strength in processing complex prompts, providing well-rounded responses, and excelling at tasks like coding and web development. There is no mention of image generation capabilities, highlighting that Gemini is intended for *analysis and comprehension*, not creative synthesis. This is distinct from generative models like Imagen, which are designed for *image creation* rather than multimodal reasoning.

-----

-----

-----

### Source [19]: https://gemini.google/overview/

Query: What is the difference between multimodal LLMs for understanding (like Gemini) versus generative diffusion models for image creation (like Midjourney or Stable Diffusion)?

Answer: The **Gemini app** is an interface to a multimodal LLM that can handle text, audio, images, and more, focusing on *understanding* and *responding* to user prompts. The documentation outlines Gemini’s limitations, such as accuracy and bias, which stem from its reliance on predictive text modeling. Importantly, Gemini is described as generating responses relevant to context, but it does not inherently distinguish between accurate and inaccurate information. The app’s functionality centers on *interpreting and explaining* multimodal data, not generating images. Features like “double check” are aimed at verifying Gemini’s responses, further reinforcing its focus on comprehension rather than creative image synthesis as seen in diffusion models.

-----

-----

-----

### Source [20]: https://docs.litellm.ai/docs/providers/gemini

Query: What is the difference between multimodal LLMs for understanding (like Gemini) versus generative diffusion models for image creation (like Midjourney or Stable Diffusion)?

Answer: **Google AI Studio’s Gemini** integration focuses on providing generative AI capabilities through APIs, supporting modalities like text and audio (TTS). The supported parameters and endpoints are aligned with tasks such as chat completions, embeddings, and reasoning content generated from user input. While Gemini can handle multimodal inputs, the documentation does not indicate any image synthesis or generative art features. Instead, it highlights *reasoning and understanding* across supported modalities. For direct image creation, other specialized generative models (like diffusion models) would be used.

-----

</details>

<details>
<summary>How are multimodal RAG systems being integrated as tools into agentic frameworks like LangGraph?</summary>

### Source [21]: https://blog.futuresmart.ai/langgraph-rag-agent-tutorial-basics-to-advanced-multi-agent-ai-chatbot

Query: How are multimodal RAG systems being integrated as tools into agentic frameworks like LangGraph?

Answer: This tutorial details the evolution from basic LLM calls to advanced, **tool-aware agents** using LangGraph, incorporating RAG (Retrieval-Augmented Generation) as a key component. In LangGraph, RAG is integrated as a tool within an agentic framework. The framework enables the agent to:
- **Decide between internal knowledge and external tools** (such as RAG or web search) using a router mechanism.
- **Pull trusted data from user-provided documents** through RAG.
- **Fallback to real-time web search** if the internal knowledge base is insufficient.
LangGraph supports memory for multi-turn conversations and allows for **streaming partial answers** using async generators. The modular design means developers can swap in their own data sources and deploy the agent as an API or serverless function. RAG, as a tool, is orchestrated by the LangGraph agent, which determines when and how to employ it for optimal results.

-----

-----

-----

### Source [22]: https://qdrant.tech/documentation/agentic-rag-langgraph/

Query: How are multimodal RAG systems being integrated as tools into agentic frameworks like LangGraph?

Answer: This tutorial explains how **Agentic RAG** systems leverage LangGraph’s **state management** to orchestrate multiple tools, including RAG, vector databases (like Qdrant), and web search. In this setup:
- RAG is one of several tools available to the agent, not the only retrieval mechanism.
- The LangGraph agent evaluates each query and **dynamically chooses** whether to use RAG (via one of multiple vector stores) or other sources like live web search.
- The agent’s workflow is not linear; it can perform **multi-step, multi-source retrieval** and decision-making depending on the complexity of the query.
This selective and flexible orchestration is central to agentic frameworks, enabling richer and more context-aware responses than traditional (static) RAG pipelines.

-----

-----

-----

### Source [23]: https://python.langchain.com/docs/tutorials/rag/

Query: How are multimodal RAG systems being integrated as tools into agentic frameworks like LangGraph?

Answer: The LangChain documentation highlights that **LangGraph is not required** for RAG but offers significant advantages when integrating RAG as a tool in more complex, agentic applications. Benefits include:
- **Multiple invocation modes** (sync, async, streaming).
- **Stateful conversation management** for context-informed answers.
- **Persistence and human-in-the-loop features**.
LangGraph manages and persists state, simplifying the integration of RAG within a conversational agent. Developers can define RAG as a callable tool within the LangGraph framework, which can be invoked by the agent as part of broader workflows.

-----

-----

-----

### Source [24]: https://ai.gopubby.com/building-rag-research-multi-agent-with-langgraph-1bd47acac69f

Query: How are multimodal RAG systems being integrated as tools into agentic frameworks like LangGraph?

Answer: This article describes a **RAG Research Multi-Agent** tool built with LangGraph, designed for complex, multi-step, multi-source question answering. The system:
- Uses a **hybrid search** (including RAG) and a **Cohere reranking step** to retrieve and prioritize relevant documents.
- Incorporates a **self-corrective mechanism**, such as hallucination checks, to enhance reliability.
- Tracks execution state via a structured message history between user and agent, enabling iterative reasoning.
RAG is treated as one tool among several, used selectively and sometimes in combination with others, under the orchestration of LangGraph’s agentic architecture.

-----

-----

-----

### Source [25]: https://www.youtube.com/watch?v=qj5WTJaE7r0

Query: How are multimodal RAG systems being integrated as tools into agentic frameworks like LangGraph?

Answer: In this video tutorial, a **multimodal agent** is constructed using LangGraph, demonstrating how agents can employ tools—including RAG—and memory to manage user interactions. The agent:
- Has access to a database through tool integrations, which can include RAG-based retrieval.
- Handles multiple users and conversations by maintaining state and context.
- Can process multimodal inputs (e.g., text, images) and use tools conditionally, based on input type and content.
The LangGraph framework enables the agent to invoke RAG as a tool when textual knowledge retrieval is needed, alongside other modalities and tools, within a unified agentic workflow.

-----

-----

</details>

<details>
<summary>How do multimodal embedding models like CLIP and ColPali create a shared vector space for text and images?</summary>

### Source [31]: https://milvus.io/docs/use_ColPali_with_milvus.md

Query: How do multimodal embedding models like CLIP and ColPali create a shared vector space for text and images?

Answer: ColPali combines the multi-vector embedding approach of ColBERT with PaliGemma, a multimodal large language model, to represent both text and images within a unified multi-vector embedding space. ColBERT, originally designed for text, generates a list of embeddings for each data instance. When comparing a query and a document, it uses the MaxSim operation: for each word in the query, it selects the most similar embedding from the document (using cosine similarity or squared L2 distance) and sums these maximum similarities. ColPali extends this methodology to multimodal data, enabling pages containing both text and images to be encoded as unified multi-vector embeddings. This approach captures fine-grained information from both modalities, supporting more effective retrieval-augmented generation for multimodal content.

-----

-----

-----

### Source [32]: https://dev.to/aws/beyond-text-building-intelligent-document-agents-with-vision-language-models-and-colpali-and-oc

Query: How do multimodal embedding models like CLIP and ColPali create a shared vector space for text and images?

Answer: ColPali leverages the architecture of Vision-Language Models (VLMs), which feature two main encoders: an image encoder and a text encoder. The image encoder divides an image into patches and encodes each patch into separate embeddings, capturing detailed visual information. The text encoder processes accompanying text into its own set of embeddings. After encoding, the image embeddings pass through an adapter layer that transforms them into a numerical format suitable for integration with text embeddings. This structure allows text and image embeddings to be compared or combined in a shared space, facilitating the creation of unified representations across modalities.

-----

-----

-----

### Source [33]: https://www.nomic.ai/blog/posts/nomic-embed-multimodal

Query: How do multimodal embedding models like CLIP and ColPali create a shared vector space for text and images?

Answer: Multimodal embedding models like ColPali and ColNomic use a multi-vector late interaction mechanism, generating multiple embeddings per document or query rather than a single embedding. This late interaction enables more precise cross-modal matching. For training, these models use strategies to prevent shortcut learning—for example, sampling from the same data source and employing hard negative mining, which encourages the model to learn meaningful relationships between text and images. By using these advanced training and retrieval techniques, ColPali and similar models achieve higher performance than CLIP-style architectures, especially in bridging the modality gap between text and images.

-----

-----

-----

### Source [34]: https://blog.vespa.ai/retrieval-with-vision-language-models-colpali/

Query: How do multimodal embedding models like CLIP and ColPali create a shared vector space for text and images?

Answer: The ColPali model is trained end-to-end for the task of page retrieval, projecting both image patches and text tokens into a common 128-dimensional embedding space. For images, each grid cell (patch) is encoded as a 128D vector; for text, each token is similarly projected into the same space. The model then applies a late-interaction scoring mechanism (MaxSim), calculating the similarity between each query token vector and each image patch vector to determine relevance. This design supports cross-modal retrieval, as both image and text data are represented in the same vector space, enabling direct comparison. The base model for ColPali is PaliGemma 3, a powerful vision-language model released by Google.

-----

-----

-----

### Source [35]: https://learnopencv.com/multimodal-rag-with-colpali/

Query: How do multimodal embedding models like CLIP and ColPali create a shared vector space for text and images?

Answer: ColPali integrates ColBERT’s late interaction mechanism with PaliGemma, a vision large language model (LLM), to extract and represent multimodal content. ColBERT, by design, generates contextualized embeddings for each token in a query, and each token interacts with all document embeddings through a late interaction (MaxSim) operation. This preserves granular semantic relationships between query and document. ColPali extends this to document pages by treating each page as an image and extracting embeddings for both the visual and textual elements. The result is a multimodal embedding space where detailed content from images, tables, charts, and text can be efficiently compared and retrieved, supporting fine-grained multimodal retrieval tasks.

-----

-----

</details>

<details>
<summary>What are the enterprise-level architectural patterns for processing and storing multimodal data as raw bytes, Base64, and private data lake URLs?</summary>

### Source [36]: https://airbyte.com/data-engineering-resources/enterprise-data-architecture

Query: What are the enterprise-level architectural patterns for processing and storing multimodal data as raw bytes, Base64, and private data lake URLs?

Answer: Enterprise data architecture leverages advanced patterns such as **data mesh** and **data fabric** to process and store multimodal data. 

- **Data Mesh** promotes decentralized, domain-oriented ownership of data, allowing business domains to treat data—regardless of format (raw bytes, Base64, private data lake URLs)—as products. This requires self-serve infrastructure for data ingestion, transformation, and storage, with standardized tools ensuring discoverability and interoperability. Federated governance balances enterprise standards with domain autonomy, enabling flexible handling of different data types and formats.

- **Data Fabric** provides a unified access layer spanning distributed environments, integrating diverse storage types (on-premises, cloud) and formats. It automates data discovery and flow orchestration using active metadata management and AI, making it possible to manage multimodal data seamlessly. Data fabric can automatically provision access to data stored as raw bytes, encoded formats (Base64), or referenced by data lake URLs, optimizing operational efficiency and reducing silos.

These architectures enable organizations to adopt scalable, flexible solutions for processing and storing various multimodal data representations, supporting both operational and analytical workloads across the enterprise.

-----

-----

-----

### Source [37]: https://www.rishabhsoft.com/blog/enterprise-software-architecture-patterns

Query: What are the enterprise-level architectural patterns for processing and storing multimodal data as raw bytes, Base64, and private data lake URLs?

Answer: The **event-driven architecture pattern** is widely used in enterprise systems for processing diverse data, including multimodal sources. 

- It consists of **decoupled, single-purpose modules** that process and respond to events asynchronously, which can include data arriving in raw bytes, Base64 strings, or as references (such as URLs to data lakes).
- The pattern supports two main topologies: *broker* (linking events without a central mediator) and *mediator* (orchestrating operations via an event bus).
- This approach is highly adaptable to real-time changes and enables scalable, responsive handling of multimodal data blocks that interact with specific modules. It is particularly suited for asynchronous systems, applications requiring instant data communication, and those that process unstructured or rapidly changing data.

However, building a unified data structure is challenging if events differ in requirements, and error handling can be complex when multiple modules process the same event type. 

-----

-----

-----

### Source [38]: https://www.informatica.com/blogs/data-integration-architecture-why-there-is-a-shift-in-common-patterns.html.html.html.html

Query: What are the enterprise-level architectural patterns for processing and storing multimodal data as raw bytes, Base64, and private data lake URLs?

Answer: Modern data integration architectures emphasize **ETL (Extract, Transform, Load)** and **multi-cloud patterns** to support multimodal data processing and storage.

- **ETL architectures** use dedicated processing servers for transforming and enriching data from multiple sources, whether raw bytes, Base64-encoded, or URLs, without overloading source or target applications. Modern ETL can distribute transformation responsibilities across applications and supports diverse integration patterns.
- **Multi-cloud architectures** enable cloud-agnostic, scalable management of multimodal data, providing unified views regardless of where or how the data is stored (e.g., as raw bytes in object storage, Base64-encoded documents, or URLs referencing private data lakes).
- **Data fabric and data mesh** are recommended for unified data experiences and governance, ensuring cataloging and compliance for all data types.
- **Autonomous data integration** using AI and machine learning further optimize processing, supporting digital transformation and advanced analytics for multimodal data.

-----

-----

-----

### Source [39]: https://www.myriadventures.com/perspectives/architectural-paradigms-for-scalable-unstructured-data-processing-in-enterprise

Query: What are the enterprise-level architectural patterns for processing and storing multimodal data as raw bytes, Base64, and private data lake URLs?

Answer: A **hybrid storage architecture** is recommended for scalable processing of multimodal, unstructured data.

- **Raw data** is typically stored in object storage.
- **Processed information** may reside in document databases.
- **Vector representations** (for ML tasks) are kept in vector databases.
- **Extracted relationships** are managed in graph databases.

This multi-modal strategy efficiently supports diverse query patterns and use cases, whether data is stored as raw bytes, Base64, or referenced by private data lake URLs.

For data processing:
- **Distributed computing frameworks** (e.g., Apache Spark, Ray) handle large-scale data transformations.
- **Stream processing frameworks** (Apache Flink, Kafka Streams) enable real-time analytics and event-driven architectures.
- **Analytics engines** (Elasticsearch, Apache Solr) facilitate search and analysis of textual data.
- **Machine learning frameworks** (TensorFlow, PyTorch) process and analyze multimodal data.
- **Graph analytics** are used for data stored in graph databases to reveal complex relationships.

This architecture enables flexible, scalable handling and analytics of multimodal data representations in enterprise environments.

-----

-----

-----

### Source [40]: https://www.tiledb.com/multimodal-data

Query: What are the enterprise-level architectural patterns for processing and storing multimodal data as raw bytes, Base64, and private data lake URLs?

Answer: **TileDB's universal database platform** provides an enterprise-ready solution for multimodal data storage and processing.

- It supports diverse data types (raw bytes, Base64, data lake URLs) within a single, unified system, eliminating silos and enabling high performance and scalability.
- In life sciences, TileDB manages massive datasets (e.g., sequencing data, clinical metadata) using a **sparse array technology** that efficiently represents complex, multimodal information.
- Its **cloud-native architecture** enables distributed processing without unnecessary data movement, supporting operational and analytical workloads at scale.
- TileDB's **flexible data model** adapts to new and emerging modalities, making it future-proof for expanding enterprise requirements.
- **Collaboration infrastructure** supports secure sharing and integration of multimodal data across institutions, addressing governance and privacy needs.

TileDB exemplifies how modern enterprise architectures can unify storage and processing for multimodal data, regardless of format or origin.

-----

-----

</details>

<details>
<summary>What is the technical architecture of the ColPali model for multimodal Retrieval-Augmented Generation (RAG)?</summary>

### Source [41]: https://dev.to/aws/beyond-text-building-intelligent-document-agents-with-vision-language-models-and-colpali-and-oc

Query: What is the technical architecture of the ColPali model for multimodal Retrieval-Augmented Generation (RAG)?

Answer: ColPali is designed to address the limitations of traditional Retrieval-Augmented Generation (RAG) systems in handling multimodal documents containing text, images, tables, and complex layouts. Traditional RAG pipelines rely on extracting and separately processing text and visual elements, which often leads to loss of vital context, especially where the visual layout or diagrams are essential to understanding. ColPali overcomes this by leveraging vision-language models (VLMs) capable of processing raw visual documents directly. This approach enables retrieval based on visual relevance rather than isolated text, maintaining the structural and contextual integrity of the original document, and supporting more accurate and context-aware retrieval for RAG applications.

-----

-----

-----

### Source [42]: https://decodingml.substack.com/p/the-king-of-multi-modal-rag-colpali

Query: What is the technical architecture of the ColPali model for multimodal Retrieval-Augmented Generation (RAG)?

Answer: The technical architecture of a ColPali-based RAG system comprises five core components:
- **PDF to Image Converter:** Converts documents into high-quality images, ensuring the preservation of visual structure, such as tables, diagrams, and layout.
- **Storage Service:** Manages persistent storage of these images for efficient retrieval.
- **ColPali Model (ColQwen 2.5):** Acts as the visual understanding engine that generates embeddings from document images. This model captures not only textual content but also spatial relationships, formatting, and the interplay between text and visuals, unlike traditional text-based RAG systems.
- **Vector Database (e.g., Qdrant):** Stores the generated visual embeddings, allowing fast similarity search and retrieval at the page level, with support for metadata.
- **Multimodal LLM (e.g., Claude Sonnet 3.7):** Interprets retrieved images and generates responses, leveraging the actual document visuals to reference tables, figures, and original formatting.

This architecture allows ColPali to maintain the document's visual and contextual information throughout the retrieval and generation process, resulting in more accurate and contextually relevant answers.

-----

-----

-----

### Source [43]: https://learnopencv.com/multimodal-rag-with-colpali/

Query: What is the technical architecture of the ColPali model for multimodal Retrieval-Augmented Generation (RAG)?

Answer: ColPali’s multimodal RAG architecture treats each document page as an image, facilitating efficient retrieval of images, tables, charts, and text in complex documents such as financial reports and legal contracts. By leveraging Vision Language Models (VLMs), the system can understand detailed and nuanced information, including spatial and visual relationships, which are often critical in professional documents. The approach bypasses the challenges of processing unstructured elements by enabling direct analysis of visual content, making it especially suitable for domains where the accuracy and contextual integrity of source data are paramount.

-----

-----

-----

### Source [44]: https://blog.vespa.ai/Transforming-the-Future-of-Information-Retrieval-with-ColPali/

Query: What is the technical architecture of the ColPali model for multimodal Retrieval-Augmented Generation (RAG)?

Answer: ColPali’s architecture is built specifically for Retrieval-Augmented Generation (RAG) and is distinguished by its ability to capture both textual and visual content using vector representations of the entire rendered document. The system integrates the entire document—including images, tables, and layout—into its embedding process, ensuring that context from visual elements is not lost. This reduces preprocessing complexity (such as OCR, table extraction, and layout analysis) and enhances the relevance and accuracy of retrieved information. ColPali’s approach streamlines document preparation and enables more robust retrieval in visually rich, document-heavy contexts like healthcare and financial services.

-----

-----

-----

### Source [45]: https://www.activeloop.ai/resources/col-palis-vision-rag-and-max-sim-for-multi-modal-ai-search-on-documents/

Query: What is the technical architecture of the ColPali model for multimodal Retrieval-Augmented Generation (RAG)?

Answer: ColPali is a document retrieval model that combines Vision Language Models (VLMs) with late interaction mechanisms, allowing it to process entire documents as images and produce multi-vector embeddings that capture both textual and visual content. The model eliminates the need for traditional OCR pipelines and leverages a multi-vector retrieval mechanism inspired by ColBERT’s late interaction. However, this innovation results in a significant storage requirement—ColPali embeddings require 256 KB per page, which is about 30 times larger than traditional BM25-based methods. Additionally, the multi-vector mechanism is not widely supported in existing vector retrieval frameworks, introducing additional engineering complexity for large-scale deployment.

-----

-----

-----

### Source [86]: https://decodingml.substack.com/p/the-king-of-multi-modal-rag-colpali

Query: What is the technical architecture of the ColPali model for multimodal Retrieval-Augmented Generation (RAG)?

Answer: The ColPali RAG system architecture is composed of five core components:

- **PDF to Image Converter:** Uses tools such as pdf2image to convert PDF documents into high-quality images, preserving the original visual layout, including tables, diagrams, and formatting.
- **Storage Service:** Stores the converted images for later retrieval and ensures persistent availability during query processing.
- **ColPali Model (ColQwen 2.5):** Acts as a visual understanding engine, generating embeddings directly from document images. This allows ColPali to capture not only textual information but also spatial relationships, formatting, and the interplay between visual and textual elements—capabilities beyond traditional text-based RAG models.
- **Vector Database (e.g., Qdrant):** Indexes these visual embeddings for efficient similarity search, supporting multi-vector storage and cosine similarity. This enables precise document or page-level retrieval with associated metadata.
- **Multimodal LLM (e.g., Claude Sonnet 3.7):** Processes the retrieved document images to generate final responses. By interpreting actual images, the LLM can accurately reference tables, interpret figures, and maintain the original document context.

This architecture allows ColPali to support a visual document understanding pipeline that operates on rendered images rather than extracted text, resulting in richer and more accurate retrieval for multimodal content.

-----

-----

-----

### Source [87]: https://learnopencv.com/multimodal-rag-with-colpali/

Query: What is the technical architecture of the ColPali model for multimodal Retrieval-Augmented Generation (RAG)?

Answer: ColPali’s multimodal RAG approach treats each document page as an image, enabling efficient retrieval of not only text but also elements like images, tables, and charts. By leveraging Vision Language Models (VLMs), ColPali can understand detailed relationships and structures in complex documents—such as financial reports and legal contracts—where factual accuracy is crucial.

The architecture is designed to handle challenges in processing unstructured elements and is suitable for industrial-scale applications, such as the analysis of SEC 10-Q financial reports. By using VLMs, ColPali can parse and comprehend intricate visual and textual components, providing a robust solution for organizations requiring precise document analysis.

The article emphasizes ColPali’s advantage over traditional retrieval systems, especially in scenarios where visual context is essential for fact extraction and decision-making.

-----

-----

-----

### Source [88]: https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introduction-to-ocr-free-vision-rag-using-colpali-for-complex-documents/4276357

Query: What is the technical architecture of the ColPali model for multimodal Retrieval-Augmented Generation (RAG)?

Answer: ColPali’s architecture integrates Vision Language Models (VLMs) such as PaliGemma to interpret document images. These VLMs are trained on datasets containing text, images, diagrams, and layouts, enabling the model to understand both visual and textual features.

Key features include:

- **Integrated VLMs:** The architecture leverages VLMs for interpreting the complex interplay between document visuals and text.
- **Enhanced Contextual Understanding:** Unlike OCR-based systems, ColPali analyzes the full document layout, recognizing the relationships between tables, surrounding text, and diagrams, resulting in more accurate comprehension.
- **Dynamic RAG Integration:** ColPali is built for seamless integration into Retrieval-Augmented Generation frameworks, allowing real-time, contextually rich information retrieval based on user queries.
- **Simplified Indexing:** The system eliminates complex preprocessing steps, streamlining document indexing.
- **Low Query Latency:** The end-to-end trainable model maintains low latency during queries, supporting real-time applications.

By feeding image patch embeddings into a language model, ColPali maps visual features into a latent space that aligns closely with textual content, enhancing retrieval accuracy and response relevance.

-----

-----

-----

### Source [89]: https://blog.vespa.ai/Transforming-the-Future-of-Information-Retrieval-with-ColPali/

Query: What is the technical architecture of the ColPali model for multimodal Retrieval-Augmented Generation (RAG)?

Answer: ColPali’s RAG architecture is designed to process visually rich, multimodal documents (such as PDFs) by embedding the entire rendered document, including visual elements, into vector representations for retrieval. This approach eliminates the need for traditional multi-step preprocessing—such as text extraction, OCR, and layout analysis—by directly creating embeddings that capture both textual and visual information.

The architecture is optimized for document-heavy contexts where visual features (like tables, charts, and layouts) are crucial for understanding content. By embedding the full visual context, ColPali enables retrieval systems to return more relevant and contextually appropriate information, especially in domains like healthcare and finance where accuracy and context are critical.

The solution streamlines document preparation for RAG, making the retrieval process more accurate and efficient by maintaining the integrity of both visual and textual content in the embedding process.

-----

-----

-----

### Source [90]: https://www.activeloop.ai/resources/col-palis-vision-rag-and-max-sim-for-multi-modal-ai-search-on-documents/

Query: What is the technical architecture of the ColPali model for multimodal Retrieval-Augmented Generation (RAG)?

Answer: ColPali combines Vision Language Models (VLMs) with late interaction mechanisms to process documents. Unlike traditional OCR-based pipelines, ColPali processes entire documents as images and generates multi-vector embeddings that capture both textual and visual content.

Key architectural elements include:

- **Multi-vector Embeddings:** Inspired by ColBERT’s late interaction architecture, ColPali creates multiple vectors per document, enabling fine-grained semantic matching during retrieval.
- **Vision-language Processing:** The model captures nuanced contextual and visual cues, supporting richer understanding of complex documents.
- **Storage Considerations:** Each ColPali embedding requires 256 KB per page, which is significantly larger than traditional text-based methods. This has implications for scalability and engineering complexity, as many vector databases do not natively support multi-vector retrieval.
- **No Standard OCR:** ColPali bypasses OCR entirely, allowing for richer feature extraction directly from rendered images.

This architecture enables advanced search and retrieval in multimodal, visually rich document collections, but also introduces new storage and deployment challenges due to its large memory requirements and specialized retrieval mechanisms.

-----

-----

</details>

<details>
<summary>How can multimodal capabilities be integrated into AI agent frameworks like LangGraph to process and reason over combined text and image inputs?</summary>

### Source [46]: https://python.langchain.com/docs/how_to/multimodal_inputs/

Query: How can multimodal capabilities be integrated into AI agent frameworks like LangGraph to process and reason over combined text and image inputs?

Answer: LangChain supports integrating **multimodal capabilities** into agent frameworks by allowing models to receive both text and image inputs in a unified format. The framework provides:
- **Provider-specific and cross-provider standards** for passing multimodal data, particularly in chat models.
- For images, models can accept data either as URLs or as base64-encoded content blocks. The standard format for in-line images looks like:
  ```json
  {
    "type": "image",
    "source_type": "base64",
    "mime_type": "image/jpeg", // or image/png, etc.
    "data": "<base64 data string>"
  }
  ```
- This approach enables the direct inclusion of images alongside text input, which is essential for agents needing to reason over both modalities in a single prompt.
- The documentation provides examples of fetching image data, encoding it in base64, and passing it to supported models, ensuring compatibility with various providers that accept multimodal input.

This method allows agent frameworks like LangGraph (which builds upon LangChain) to process and reason over combined text and image content by formatting and routing multimodal payloads to compatible models.

-----

-----

-----

### Source [48]: https://langchain-ai.github.io/langgraph/concepts/multi_agent/

Query: How can multimodal capabilities be integrated into AI agent frameworks like LangGraph to process and reason over combined text and image inputs?

Answer: LangGraph provides a flexible framework for building **multi-agent systems**, and it enables agents to have different input and output schemas, which is foundational for multimodal processing:
- Agents can define private state schemas, distinct from the overall graph state, allowing them to process specialized data—such as images or text—relevant to their function.
- The framework supports the **handoff of data between agents** using tool calls, with the ability to update the graph state and control flow. This mechanism can be extended to carry multimodal data (text, images, etc.) between agents.
- For effective multimodal integration, input/output transformations can be defined so that agents or subgraphs using different modalities can communicate within the broader workflow, maintaining compatibility and data integrity across the system.

-----

-----

-----

### Source [49]: https://www.langchain.com/langgraph

Query: How can multimodal capabilities be integrated into AI agent frameworks like LangGraph to process and reason over combined text and image inputs?

Answer: LangGraph is designed to support **expressive, customizable agent workflows** and robust control flows, which are prerequisites for multimodal agent architectures:
- The framework’s low-level primitives and stateful design allow developers to create agents that can process complex, structured data—including multiple modalities—by defining custom states and agent functions.
- LangGraph’s architecture facilitates **human-agent collaboration**, state inspection, and quality control, all of which can be leveraged to manage multimodal reasoning tasks (e.g., verifying model outputs that combine text and image analysis).
- Although this source does not detail explicit multimodal API integrations, it emphasizes that the flexibility and customizability of LangGraph make it suitable for implementing multimodal agent pipelines where images and text are handled together.

-----

-----

-----

### Source [50]: https://cloud.google.com/blog/products/ai-machine-learning/build-multimodal-agents-using-gemini-langchain-and-langgraph

Query: How can multimodal capabilities be integrated into AI agent frameworks like LangGraph to process and reason over combined text and image inputs?

Answer: This source provides a practical overview of **building multimodal agents** with Gemini, LangChain, and LangGraph:
- Gemini models support multimodal input, enabling use cases such as **object identification** and **content moderation** that require reasoning over both text and images.
- LangChain provides the tools to chain LLM calls and external data sources, while LangGraph supplies the graph-based structure for orchestrating complex, controlled, multi-agent workflows.
- To build a multimodal agent, developers must decide between no-code/low-code tools or custom agent programming. For advanced use cases (like enterprise-level object detection), combining Gemini (for multimodal understanding), LangChain (for chaining/model calls), and LangGraph (for workflow control) is recommended.
- The integration process involves routing both text and image data through a unified workflow, allowing agents to query, process, and reason over combined modalities—enabling sophisticated applications like multimedia search and verification.

-----

-----

</details>

<details>
<summary>What are the architectural patterns for extending multimodal LLMs to new data types like audio and video by integrating different encoders?</summary>

### Source [51]: https://blog.premai.io/multimodal-llms-architecture-techniques-and-use-cases/

Query: What are the architectural patterns for extending multimodal LLMs to new data types like audio and video by integrating different encoders?

Answer: **Architectural Patterns for Extending Multimodal LLMs**

- **Unified Embedding Decoder Architecture**: This pattern utilizes a single decoder model that processes multiple modalities. Non-textual data (such as audio or video, once encoded) are transformed into embedding vectors with the same dimensionality as text token embeddings, enabling concatenation and unified processing by the language model. This approach is exemplified by models like Llama 3.2 and GPT-2.
- **Cross-Modality Attention Architecture**: This pattern introduces cross-attention mechanisms that connect embeddings from different modalities (e.g., visual and textual). For instance, image or audio embeddings are connected to text embeddings in the multi-head attention layers, allowing direct and flexible integration.
- **Training Strategies**:
  - **Pretraining and Fine-Tuning**: Models are initially pretrained with modality-specific encoders and adapters, then fine-tuned for specific multimodal tasks.
  - **Parameter-Efficient Fine-Tuning**: Methods like LoRA (Low-Rank Adaptation) and QLoRA enable efficient adaptation to new modalities by updating only small sets of parameters rather than the entire model, which is especially important for large models.
- These patterns allow new data types such as audio and video to be integrated by developing or selecting appropriate encoders, projecting their outputs to a shared embedding space, and employing architectures like cross-attention or unified decoders to combine and process the information.

-----

-----

-----

### Source [52]: https://arxiv.org/html/2405.17927v1

Query: What are the architectural patterns for extending multimodal LLMs to new data types like audio and video by integrating different encoders?

Answer: **Taxonomy of Multimodal Model Architectures**

- **Two Main Categories**:
  - **Deep Fusion**: Fusion of modalities occurs within the internal layers of the model.
    - *Type-A*: Employs standard cross-attention layers for integration.
    - *Type-B*: Utilizes custom-designed layers specifically tailored for cross-modal interactions.
  - **Early Fusion**: Fusion of modalities happens at the model's input.
    - *Type-C*: Ingests non-tokenized multimodal inputs directly into the model.
    - *Type-D*: Uses discretely tokenized multimodal inputs, supplied directly to the transformer (decoder-only or encoder-decoder style).
- **Integration of New Data Types**:
  - Introducing new modalities (like audio or video) typically involves designing or adapting modality-specific encoders.
  - The outputs from these encoders are fused with text representations either early (input-level) or deep (internal layers) in the model, depending on the chosen architecture type.
- **Advantages/Disadvantages**:
  - Each architecture type comes with trade-offs in terms of training data requirements, computational complexity, and flexibility for supporting new modalities.

-----

-----

-----

### Source [53]: https://www.ionio.ai/blog/a-comprehensive-guide-to-multimodal-llms-and-how-they-work

Query: What are the architectural patterns for extending multimodal LLMs to new data types like audio and video by integrating different encoders?

Answer: **Core Components and Integration of Modality Encoders**

- **Modality Encoders**: Each data type (text, audio, video, image) requires a dedicated encoder tailored to extract relevant features. For example, a text encoder captures semantics, while an audio encoder extracts acoustic and temporal features.
- **Input Projector**: After encoding, all modality-specific representations are unified—typically via concatenation or projection into a shared embedding space—so the backbone LLM can process them together.
- **Shared Embedding Space**: The projection step ensures that the outputs of different encoders are compatible and can be jointly processed, enabling the model to reason over and generate multimodal content.
- This modular structure allows straightforward extension to new data types by implementing new encoders and updating the projection mechanism to accept the new modality.

-----

-----

-----

### Source [54]: https://www.nvidia.com/en-us/glossary/multimodal-large-language-models/

Query: What are the architectural patterns for extending multimodal LLMs to new data types like audio and video by integrating different encoders?

Answer: **Summary Table of Multimodal LLM Architecture**

- **Separate Modality Encoders**: Each modality (text, image, audio, video) uses a specialized encoder to convert raw inputs into embeddings capturing their meaning.
- **Fusion Module (Input Projector)**: Encoded representations from different modalities are aligned and projected into a unified embedding space.
- **LLM Backbone**: Processes the fused, multimodal embeddings using pretrained knowledge to perform complex reasoning and generation tasks.
- **Contrastive Learning Objectives**: Training often involves aligning representations across modalities (e.g., making sure an image and its caption are close in embedding space), which supports effective integration.
- **Extending to New Modalities**: Adding support for audio or video involves developing new encoders for those data types and updating the fusion module to include their embeddings in the unified representation.

-----

-----

-----

### Source [55]: https://arxiv.org/html/2411.06284v2

Query: What are the architectural patterns for extending multimodal LLMs to new data types like audio and video by integrating different encoders?

Answer: **Unified Representation and Cross-Modal Integration**

- **Unified Codebooks & Joint Embedding Spaces**: Multimodal LLMs build integrated representations by mapping different data types (text, image, audio, video) into a shared embedding space using unified codebooks or projection methods.
- **Seamless Cross-Modal Processing**: This unified representation allows models to perform tasks such as translating between modalities, cross-modal retrieval, and context-aware reasoning.
- **Extending to New Modalities**: To add a new data type (e.g., audio), a suitable encoder is introduced, and its outputs are projected into the joint embedding space, enabling seamless fusion and processing alongside other modalities.
- **Applications**: Unified architectures enable complex tasks such as describing images, matching sounds with visuals, and interpreting multimodal inputs for domains like healthcare, security, and commerce.

-----

-----

-----

### Source [76]: https://blog.premai.io/multimodal-llms-architecture-techniques-and-use-cases/

Query: What are the architectural patterns for extending multimodal LLMs to new data types like audio and video by integrating different encoders?

Answer: Multimodal large language models (LLMs) typically extend to new data types like audio and video through three main architectural approaches for integrating different encoders:

- **Unified Embedding Decoder Architecture**: In this approach, each non-text modality (such as images, audio, or video) is processed with its own encoder, producing embedding vectors with the same dimensions as text tokens. These embeddings are then concatenated and fed into a single decoder (often a transformer language model) capable of handling the multimodal input seamlessly. This enables the model to treat all modalities in a consistent manner, as seen in models like Llama 3.2 and GPT-2.
- **Cross-Modality Attention Architecture**: Here, a cross-attention mechanism allows the model to directly connect visual (or other non-text) embeddings with text embeddings in the multi-head attention layers. This enables more dynamic integration between modalities, with the attention mechanism learning the relationships and dependencies across different data types, inspired by the original transformer design.
- **Training Strategies**: Extension to new modalities often involves a two-phase training procedure:
  - **Pretraining**: A pretrained language model is augmented by adding adapters and modality-specific encoders, then aligned with the new modality.
  - **Fine-tuning**: The full model is adapted for specific multimodal tasks (e.g., visual question answering, captioning). Techniques like Low-Rank Adaptation (LoRA) and QLoRA enable parameter-efficient fine-tuning, allowing new encoders to be added and trained without needing to retrain the entire model from scratch.

These architectures provide flexibility for integrating new encoders for modalities like audio and video, facilitating efficient expansion of LLM capabilities.

-----

-----

-----

### Source [77]: https://arxiv.org/html/2405.17927v1

Query: What are the architectural patterns for extending multimodal LLMs to new data types like audio and video by integrating different encoders?

Answer: This source identifies and characterizes four prevalent architectural patterns for extending multimodal models to new data types by integrating different encoders:

- **Deep Fusion (Internal Layer Fusion)**
  - **Type-A**: Employs *standard cross-attention layers* for integrating different modalities. Each modality is first encoded independently, and their representations are fused within the internal layers of the model using cross-attention.
  - **Type-B**: Uses *custom-designed fusion layers* (beyond standard cross-attention) for integrating modality representations at deeper levels, allowing for tailored interactions between modalities.

- **Early Fusion (Input-Level Fusion)**
  - **Type-C**: Involves *non-tokenized multimodal inputs*, where raw or minimally processed representations from different encoders are directly fed into the model at the input.
  - **Type-D**: Uses *discretely tokenized multimodal inputs*, where outputs from each modality's encoder are tokenized into a unified format and concatenated at the input stage, suitable for decoder-only or encoder-decoder transformers.

Each of these patterns allows for the integration of new data types by designing or selecting encoders appropriate for the modality (e.g., audio, video) and adopting a fusion strategy—either at the input or within the internal layers. The choice of architectural pattern impacts training data requirements, computational cost, and the flexibility to add new modalities.

-----

-----

-----

### Source [78]: https://www.ionio.ai/blog/a-comprehensive-guide-to-multimodal-llms-and-how-they-work

Query: What are the architectural patterns for extending multimodal LLMs to new data types like audio and video by integrating different encoders?

Answer: A multimodal LLM architecture generally includes several key components to extend to new data types:

- **Modality Encoders**: Each data type (audio, video, image, text) is processed by its own encoder. For instance, an audio encoder extracts temporal and frequency features, while a video encoder might extract spatiotemporal features.
- **Input Projector**: After encoding, the representations from each modality are projected or mapped into a common embedding space. This is typically achieved by concatenation or via a learned projection, ensuring that all modalities can be jointly processed by the LLM.
- **LLM Backbone**: The unified embeddings are then fed into the LLM core (often a transformer), which performs reasoning, generation, or classification across the combined modalities.

This modular design allows integration of new encoders for additional modalities (like audio or video) by developing appropriate encoders and ensuring the output can be projected into the shared embedding space for joint processing.

-----

-----

-----

### Source [79]: https://www.nvidia.com/en-us/glossary/multimodal-large-language-models/

Query: What are the architectural patterns for extending multimodal LLMs to new data types like audio and video by integrating different encoders?

Answer: Multimodal LLM architectures are characterized by the following extensions to support new data types:

- **Separate Encoders for Each Modality**: Each data type (e.g., audio, video, image) is processed by a dedicated encoder that converts its input into an embedding representing its semantic content.
- **Fusion Module (Input Projector)**: The outputs of the different modality encoders are aligned and projected into a unified embedding space. This enables the model to combine multimodal information effectively.
- **LLM Backbone**: The unified embedding is then processed by the main language model, which handles tasks such as reasoning, comprehension, and generation across modalities.

The architecture supports the addition of new encoders for novel modalities, provided their outputs can be projected to the shared space, and training objectives (such as contrastive learning) are used to align representations across modalities.

-----

-----

-----

### Source [80]: https://arxiv.org/html/2411.06284v2

Query: What are the architectural patterns for extending multimodal LLMs to new data types like audio and video by integrating different encoders?

Answer: A unified representation approach is a key architectural pattern in multimodal LLMs for supporting new data types:

- **Unified Codebooks and Joint Embedding Spaces**: All modality encoders (for text, image, audio, video, etc.) are designed to map their inputs into a joint embedding space or codebook. This unified representation allows seamless integration of new modalities, as each encoder simply needs to project its output into the shared space.
- **Cross-Modal Capabilities**: With unified embeddings, the model can perform tasks such as cross-modal retrieval or translation (e.g., describing a video with text or matching an audio clip to a video).
- **Enhanced Contextual Understanding**: The joint embedding enables the model to reason about relationships between different modalities, improving its ability to generate contextually accurate and relevant responses.

This architectural pattern facilitates the addition of new encoders for audio, video, or other data types, provided they adhere to the unified embedding design.

-----

</details>

<details>
<summary>What are the practical trade-offs and best practices when passing multimodal data to LLM APIs as raw bytes versus Base64 strings versus private data lake URLs?</summary>

### Source [56]: https://www.ibm.com/think/insights/llm-apis

Query: What are the practical trade-offs and best practices when passing multimodal data to LLM APIs as raw bytes versus Base64 strings versus private data lake URLs?

Answer: LLM APIs operate over standard protocols, usually using HTTP requests with JSON data payloads. Before transmission, applications must convert their data into the API’s required format—typically, this is structured as JSON, which is well-suited for text and simple structured data. When dealing with **multimodal data** (such as images, audio, or video), raw binary formats are generally not directly supported in these JSON payloads. Instead, data is often encoded or referenced:
- **Raw bytes**: Not directly supported in JSON; transmitting raw bytes may require alternative encoding.
- **Base64 encoding**: Frequently used to embed binary data (like images or audio) within JSON as ASCII strings, enabling seamless transfer through APIs.
- **URLs (private data lakes)**: For large or sensitive datasets, JSON payloads commonly reference data via URLs, allowing the API to fetch data from secure storage rather than transmitting the data itself.

When choosing a method, it is crucial to implement secure protocols (such as HTTPS), encrypt data in transit, and enforce strict access controls—especially when using URLs that reference private datasets. This safeguards sensitive information and complies with data protection standards.

-----

-----

-----

### Source [57]: https://www.turing.com/resources/building-high-quality-multimodal-data-pipelines-for-llms

Query: What are the practical trade-offs and best practices when passing multimodal data to LLM APIs as raw bytes versus Base64 strings versus private data lake URLs?

Answer: A robust multimodal data pipeline must address **quality, diversity, and efficiency** when preparing and delivering data to LLM APIs:
- **Data quality**: Consistency in annotation, alignment between modalities (e.g., image-text pairs), and contextual relevance are essential. Encoding choices can impact this—poorly aligned or improperly encoded data will lead to noisy LLM outputs.
- **Diversity**: Coverage across various formats and real-world conditions demands flexible encoding strategies. For example, Base64 is simple and portable but increases data size (~33% overhead), which can slow transmission, whereas URLs referencing data lakes may better handle large datasets and varied formats.
- **Efficiency**: Manual handling of large binary files (raw bytes or Base64) is slow and prone to inconsistency. Automated pipelines that generate and upload data to secure storage, then pass access URLs to LLM APIs, scale better and support real-time annotation, review, and iteration.

Best practices include:
- Automating data ingestion and conversion (e.g., to Base64 or URL reference).
- Embedding reviewers and automated quality checks to catch encoding or alignment errors.
- Using URLs for large or sensitive files, with access controls to prevent unauthorized retrieval.

-----

-----

-----

### Source [58]: https://arxiv.org/html/2506.06579v1

Query: What are the practical trade-offs and best practices when passing multimodal data to LLM APIs as raw bytes versus Base64 strings versus private data lake URLs?

Answer: Efficient inference and deployment of LLMs in resource-constrained settings (e.g., edge or mobile) require careful trade-offs in **data transmission and handling**:
- **Raw bytes**: Transmitting raw bytes is bandwidth-efficient but often incompatible with standard API (JSON/HTTP) interfaces, requiring specialized handling and increasing integration complexity.
- **Base64 strings**: Universally supported in JSON APIs, but at the cost of increased payload size. For large files or streaming scenarios, the overhead can be substantial, potentially impacting latency and cost.
- **Private data lake URLs**: Offloads transmission cost from the API call, allowing the model server to fetch only when needed, reducing client bandwidth and memory usage. However, this requires reliable, secure storage and robust authorization mechanisms to prevent unauthorized access or data leakage.

For efficient, scalable deployment:
- Use Base64 for small, infrequent binary payloads where latency and payload size are not critical.
- Use secure URLs for large files or when integrating with cloud data lakes or distributed storage, ensuring proper authentication and access expiration.

-----

-----

-----

### Source [59]: https://hatchworks.com/blog/gen-ai/llm-integration-guide/

Query: What are the practical trade-offs and best practices when passing multimodal data to LLM APIs as raw bytes versus Base64 strings versus private data lake URLs?

Answer: Proper data preparation is central to successful LLM integration:
- Breaking down complex multimodal requests into manageable pieces improves model comprehension and output quality.
- Setting clear expectations for the format and type of multimodal input—whether as embedded Base64 blobs in JSON, or as URLs to data storage—ensures the LLM processes the data correctly.
- When using URLs, clearly specify the expected access and format, and ensure the storage location is reliable and secure.

Iterative prompt and data format refinement is recommended: start with a simple approach (e.g., Base64 for small files), and transition to URL-based references as data volume or complexity grows.

-----

-----

-----

### Source [60]: https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models

Query: What are the practical trade-offs and best practices when passing multimodal data to LLM APIs as raw bytes versus Base64 strings versus private data lake URLs?

Answer: Different multimodal models support different data input modalities:
- For models designed to handle both text and images (e.g., VLMs), input data is often provided as Base64-encoded images within JSON payloads for simplicity and compatibility.
- Some specialized models or frameworks can fetch data from cloud storage via URLs, but this requires tighter integration with a data lake and secure access management.
- The choice of encoding (Base64 vs. URL) can impact both model performance and integration complexity—Base64 is easier for small data, while URLs are more scalable for large datasets.

When choosing between formats, consider:
- The typical data size and frequency.
- The LLM or VLM’s native support for input formats.
- Security and compliance requirements for data in transit and at rest.

-----

-----

-----

### Source [91]: https://www.ibm.com/think/insights/llm-apis

Query: What are the practical trade-offs and best practices when passing multimodal data to LLM APIs as raw bytes versus Base64 strings versus private data lake URLs?

Answer: IBM emphasizes the importance of **secure transmission and proper data formatting** when interacting with LLM APIs. Data sent to LLM APIs is usually packaged in **JSON format**, which standardizes how information—including prompts and parameters—is structured for API consumption. While the article does not specifically address multimodal data encoding (raw bytes vs. Base64 vs. URLs), it does highlight that secure protocols should be implemented to **protect data in transit**, suggesting that any data—regardless of format—should be encrypted during transmission. IBM also stresses the need for **access controls**, ensuring only authorized users or services can submit or retrieve data through the API. These practices are fundamental regardless of how multimodal data is passed, and they mitigate risks associated with exposing raw data or sensitive links in API payloads[1].

-----

-----

-----

### Source [92]: https://www.turing.com/resources/building-high-quality-multimodal-data-pipelines-for-llms

Query: What are the practical trade-offs and best practices when passing multimodal data to LLM APIs as raw bytes versus Base64 strings versus private data lake URLs?

Answer: Turing AGI Advancement discusses operational and strategic challenges in **building multimodal data pipelines** for LLMs, which directly inform best practices for passing data to APIs. The report notes that **quality, diversity, and efficiency** are critical, especially when handling vision, audio, and text data. While it does not explicitly compare raw bytes, Base64 strings, and URLs, the emphasis on **annotation consistency and contextual relevance** suggests that:
- Whatever encoding is chosen, it must preserve metadata and alignment to avoid annotation drift and noisy outputs.
- To maintain efficiency, **manual handling of large raw byte streams is discouraged**; structured formats (such as Base64 or URLs referencing objects in a managed data lake) help automate and streamline data delivery and annotation.
- Data provenance and taxonomy tracking are crucial; referencing data via **URLs to a private data lake** allows for auditability and version control, whereas passing raw bytes or Base64 might complicate traceability if not carefully managed.

The report advocates for **intelligent tooling** and structured flows, which are generally easier to implement when multimodal data is referenced via **URLs to managed storage** rather than embedded directly in payloads[2].

-----

-----

-----

### Source [93]: https://arxiv.org/html/2506.06579v1

Query: What are the practical trade-offs and best practices when passing multimodal data to LLM APIs as raw bytes versus Base64 strings versus private data lake URLs?

Answer: This academic survey on efficient multi-LLM inference reviews strategies for scalable deployment, especially in resource-constrained environments. It does not focus directly on multimodal data encoding practices but provides context for practical trade-offs:
- **Bandwidth and computational constraints**: Passing large multimodal data as raw bytes can be resource-intensive, affecting latency and throughput. Encoding data as Base64 increases payload size (~33%) but provides safer transmission over text-based protocols.
- **Hierarchical inference and offloading**: If data is referenced by **URLs (e.g., pointing to a private data lake)**, models can defer loading large resources until needed—optimizing resource allocation and reducing unnecessary data transfer.
- **Scalability**: Referencing data by URL supports model routing and hierarchical inference systems, which dynamically choose when and which resources to load, rather than transmitting all data upfront.

Efficiency and adaptability are improved when **large multimodal datasets are accessed by reference** rather than embedded in API calls, especially at scale or in bandwidth-restricted scenarios[3].

-----

-----

-----

### Source [94]: https://hatchworks.com/blog/gen-ai/llm-integration-guide/

Query: What are the practical trade-offs and best practices when passing multimodal data to LLM APIs as raw bytes versus Base64 strings versus private data lake URLs?

Answer: Hatchworks provides practical LLM integration guidance, focusing on **data preparation and prompt design**. While the article does not address multimodal data transmission formats directly, it describes essential practices for **ensuring high-quality input**:
- **Pre-process and structure data** before sending to the model to improve interpretability and output quality.
- **Break down complex queries** and set clear expectations for output format, which is especially relevant when integrating multimodal data (e.g., images, text, audio).
- Although not explicit about encoding formats, the guidance implies that **data should be structured in a way that the LLM can process reliably**, suggesting that embedding raw bytes may lead to misinterpretation, while structured formats (Base64 for inline data, URLs for remote data) offer clarity and control.

Iterative prompt refinement and setting parameters are recommended to ensure the LLM understands the modality and content of the input, regardless of the actual data encoding method used[4].

-----

-----

-----

### Source [95]: https://www.databricks.com/blog/unite-your-patients-data-multi-modal-rag

Query: What are the practical trade-offs and best practices when passing multimodal data to LLM APIs as raw bytes versus Base64 strings versus private data lake URLs?

Answer: Databricks discusses best practices for **multi-modal retrieval augmented generation (RAG)**, focusing on uniting patient data across modalities such as images, text, and structured records. Their recommended approach involves:
- **Creating a vector search index** where multimodal data is referenced by **unique identifiers or URLs** rather than transmitted as raw bytes or Base64-encoded blobs.
- Using an **agentic workflow** that modularizes logic and keeps data retrieval separate from inference, enabling flexible and efficient querying.
- The evaluation framework relies on retrieving relevant data from the index (e.g., via URL or ID), and the agent only loads the actual data when needed, reducing bandwidth and memory pressure.

This approach enables **auditability, modularity, and efficient resource utilization**, making referencing data by **private data lake URLs** the preferred method in scalable, production-grade multimodal LLM applications[5].

-----

-----

</details>

<details>
<summary>How do multimodal embedding models like CLIP create a shared vector space that allows for semantic similarity searches between text, images, and documents?</summary>

### Source [61]: https://huggingface.co/docs/transformers/v4.22.1/en/model_doc/clip

Query: How do multimodal embedding models like CLIP create a shared vector space that allows for semantic similarity searches between text, images, and documents?

Answer: CLIP is a **multi-modal vision and language model** used for tasks such as image-text similarity and zero-shot image classification. It operates by taking both images and text as input, processing them through respective encoders, and projecting both modalities into a **shared embedding space** of configurable dimensionality (default is 512). The model consists of a vision encoder (for images) and a text encoder (for text data), each producing embeddings of the same size. These embeddings can be compared directly using similarity metrics (such as cosine similarity) to assess the semantic relationship between images and text. This shared vector space enables semantic similarity searches across modalities by ensuring that semantically similar text and images have similar embedding vectors.

-----

-----

-----

### Source [62]: https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/onnx-pipeline-models-multi-modal-embedding.html

Query: How do multimodal embedding models like CLIP create a shared vector space that allows for semantic similarity searches between text, images, and documents?

Answer: CLIP models provide **multi-modal embedding pipelines** that accept both image and text inputs and generate embeddings as output. The architecture includes two separate pipelines: one for image embedding and one for text embedding. Both pipelines preprocess their respective inputs and then use the CLIP model to map inputs into the **same vector space**. The embeddings for text and images are directly comparable, which allows semantic similarity tasks, such as finding which text best describes a given image or vice versa. The ONNX pipeline exports two models, one for each modality, but both generate embeddings that exist in the same latent space, facilitating cross-modal retrieval and similarity search.

-----

-----

-----

### Source [63]: https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/generate-multi-modal-embeddings-using-clip.html

Query: How do multimodal embedding models like CLIP create a shared vector space that allows for semantic similarity searches between text, images, and documents?

Answer: To generate multi-modal embeddings using CLIP, the process involves setting up the required environment and dependencies, then using the CLIP model to process both text and image data. The resulting embeddings are stored and can be used for downstream tasks, such as semantic search or similarity comparison. The key capability of CLIP in this context is that it encodes both text and images into **vectors of the same dimensionality within a shared space**, making it straightforward to compute similarity scores between them, enabling rich multi-modal search and retrieval applications.

-----

-----

-----

### Source [64]: https://docs.weaviate.io/weaviate/model-providers/transformers/embeddings-multimodal

Query: How do multimodal embedding models like CLIP create a shared vector space that allows for semantic similarity searches between text, images, and documents?

Answer: Weaviate's integration with CLIP allows users to generate **multimodal object embeddings** for both images and text, saving them into a unified vector index. When performing search, Weaviate converts queries (whether text, image, or both) into embeddings using the CLIP model and compares these embeddings in the shared vector space. This enables **multimodal and hybrid search operations**: users can, for example, search for images using text queries or find text descriptions that match a given image, all using the semantic similarity of their embeddings. The shared vectorizer ensures that semantically similar objects, regardless of modality, are close in the embedding space.

-----

-----

-----

### Source [65]: https://huggingface.co/docs/transformers/en/model_doc/clip

Query: How do multimodal embedding models like CLIP create a shared vector space that allows for semantic similarity searches between text, images, and documents?

Answer: CLIP is designed to **overcome the limitations of fixed object categories** in traditional vision models by enabling open-ended recognition tasks. It achieves this by projecting both visual and textual inputs through their respective encoders and then **mapping them into a common embedding space** using projection layers (default dimensionality is 512). Both the image and text encoders have corresponding projection layers to ensure their outputs are directly comparable. The result is that both modalities' embeddings can be used for **semantic similarity comparisons**, enabling efficient cross-modal search and retrieval based on their proximity in the shared vector space.

-----

-----

### Source [96]: https://huggingface.co/docs/transformers/v4.22.1/en/model_doc/clip

Query: How do multimodal embedding models like CLIP create a shared vector space that allows for semantic similarity searches between text, images, and documents?

Answer: CLIP is a multi-modal vision and language model designed for **image-text similarity** and zero-shot image classification. The model consists of two parallel encoders: one for images and one for text. Each encoder transforms its respective input (an image or a text prompt) into a **vector embedding** of the same dimensionality (default hidden size is 512). These embeddings are produced by a Transformer-based architecture, with configurable encoder layers, attention heads, and intermediate sizes. 

When both an image and text are input, CLIP encodes each into its embedding and computes the **cosine similarity** between the two vectors. This shared vector space enables semantic similarity searches: images and texts that are semantically related are mapped to nearby points in the embedding space, regardless of their modality. For practical usage, CLIP’s processors handle pre-processing and batching for both images and text, ensuring consistent embedding generation for similarity comparison tasks.

-----

-----

-----

### Source [97]: https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/onnx-pipeline-models-multi-modal-embedding.html

Query: How do multimodal embedding models like CLIP create a shared vector space that allows for semantic similarity searches between text, images, and documents?

Answer: CLIP models use two distinct pipelines: an **image embedding pipeline** and a **text embedding pipeline**. Both pipelines are pretrained so that, after appropriate pre-processing, they output **embeddings for images and texts in a common vector space**. This shared space allows direct comparison of vectors from different modalities, enabling **image-text similarity** tasks. For example, you can compare multiple text embeddings with an image embedding to determine which text best describes the image.

At inference, both the text and image pipelines are used to generate embeddings, which are then compared using vector similarity (such as cosine similarity). The ONNX pipeline exposes these steps, allowing seamless integration for multimodal searches and retrieval, where embeddings from either modality can be indexed and queried together.

-----

-----

-----

### Source [98]: https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/generate-multi-modal-embeddings-using-clip.html

Query: How do multimodal embedding models like CLIP create a shared vector space that allows for semantic similarity searches between text, images, and documents?

Answer: This source provides a step-by-step guide for generating **multi-modal embeddings using CLIP**. After installing the required dependencies and setting up the environment, you can generate embeddings from both text and images using the CLIP model. The embeddings are created by running each modality through its respective encoder. These embeddings are then suitable for **semantic similarity search**, as both image and text data are embedded into the same vector space.

The process supports end-to-end workflows, from environment setup to embedding generation and downstream similarity search operations, demonstrating the model’s ability to facilitate semantic comparisons between images, texts, or even documents.

-----

-----

-----

### Source [99]: https://github.com/openai/CLIP

Query: How do multimodal embedding models like CLIP create a shared vector space that allows for semantic similarity searches between text, images, and documents?

Answer: CLIP is trained on a large dataset of **(image, text) pairs** to align the representations of both modalities. The model exposes APIs to encode images and texts separately:
- `model.encode_image(image)` encodes images into feature vectors.
- `model.encode_text(text)` encodes texts into feature vectors.

Both types of embeddings are produced in the **same vector space**, enabling direct comparison. The main inference method of CLIP takes both image and text inputs and returns **cosine similarity scores** (scaled by 100) between all pairs. This approach means that an image and a semantically related text will have a high similarity score, reflecting proximity in the vector space. The model is thus capable of **cross-modal retrieval**, such as finding the most relevant caption for an image or vice versa.

-----

-----

-----

### Source [100]: https://docs.weaviate.io/weaviate/model-providers/transformers/embeddings-multimodal

Query: How do multimodal embedding models like CLIP create a shared vector space that allows for semantic similarity searches between text, images, and documents?

Answer: Weaviate’s CLIP integration uses the Hugging Face CLIP models to generate **multimodal object embeddings** for both images and texts at import, storing them in a vector index. When performing a search, queries from any supported modality are converted into embeddings using the same model. Since all embeddings live in the **shared CLIP vector space**, Weaviate supports **multimodal and hybrid search operations**: users can search for relevant images using text, or vice versa, based on semantic similarity.

This architecture allows efficient and scalable semantic search across large datasets that include images, text, or documents, leveraging the shared embedding space constructed by CLIP’s dual encoders.

-----

</details>

<details>
<summary>How are multimodal RAG systems integrated as tools into agentic frameworks like LangGraph to enable reasoning over private visual and textual data?</summary>

### Source [66]: https://python.langchain.com/docs/tutorials/rag/

Query: How are multimodal RAG systems integrated as tools into agentic frameworks like LangGraph to enable reasoning over private visual and textual data?

Answer: LangChain demonstrates how Retrieval Augmented Generation (RAG) systems are structured within agentic frameworks using LangGraph. The **application state** (often a TypedDict or Pydantic BaseModel) encapsulates inputs, retrieved context, and the generated answer, allowing the agent to pass and update data across nodes. For multimodal RAG, this state can be extended to include both textual and visual (image) data, enabling reasoning over private documents of various modalities.

**Nodes** represent steps in the workflow—such as retrieval (fetching relevant documents from private stores using similarity search) and generation (prompting the model with retrieved context and the question). Each node operates on the application’s state, ensuring modularity and extensibility for multimodal inputs.

The **control flow** is defined with LangGraph's StateGraph, connecting nodes in sequences or more complex flows. This orchestration allows the agent to reason over data iteratively, invoke different tools for different modalities (e.g., visual search for images, text retrieval for documents), and perform chained or branched reasoning tasks.

LangGraph also provides utilities to **visualize the control flow**, facilitating debugging and iterative design of agentic workflows that incorporate RAG as a tool among others. This makes it straightforward to integrate RAG modules that process private visual and textual data within larger agentic frameworks.

-----

-----

-----

### Source [67]: https://js.langchain.com/docs/tutorials/rag/

Query: How are multimodal RAG systems integrated as tools into agentic frameworks like LangGraph to enable reasoning over private visual and textual data?

Answer: LangChain.js explains that RAG systems are implemented by tying **retrieval** and **generation** steps into a single application using LangGraph. The application is organized around a clearly defined **state** that can handle both questions and retrieved context (which may include visual/text chunks, depending on the data). The **prompting mechanism** is adaptable, allowing context of any modality (text or image descriptions) to be passed to the model.

LangGraph’s orchestration brings several advantages:
- It supports multiple invocation modes (streaming, async, batch), which is helpful for multimodal data where processing times may vary.
- The modular state and step definitions enable easy integration of additional features, such as human-in-the-loop approval (e.g., for sensitive visual data), persistence, and tracing.
- The control flow is explicit, so it can be extended to include visual data retrieval (such as embeddings for images) and reasoning steps (e.g., using a vision-language model for generation).

In effect, LangGraph enables agentic workflows where RAG is one tool among many, capable of orchestrating reasoning over both private visual and textual data in a unified, traceable manner.

-----

-----

-----

### Source [68]: https://qdrant.tech/documentation/agentic-rag-langgraph/

Query: How are multimodal RAG systems integrated as tools into agentic frameworks like LangGraph to enable reasoning over private visual and textual data?

Answer: Qdrant’s documentation on Agentic RAG with LangGraph clarifies that agentic frameworks enhance traditional RAG by enabling **AI agents** to orchestrate multiple retrieval steps and dynamically choose data sources—such as separate vector stores for text and images, or web search.

LangGraph’s **state management** is central, allowing the agent to track the reasoning process, results from each retrieval tool (text, images, web), and context needed for the next decision. The agent evaluates each query, determines the best retrieval strategy (e.g., image vs. text search), and then routes the result to the generation step, which can use multimodal models if necessary.

This selective, tool-based approach enables the system to flexibly reason over **private visual and textual data** by:
- Maintaining a state that includes all relevant modalities.
- Integrating multiple retrieval tools (vector stores, web search, etc.) as graph nodes.
- Orchestrating the flow so the agent decides which retrieval/generation path is optimal for a given query.

While the tutorial focuses on text, the same pattern extends to multimodal data, laying the groundwork for agents that reason over images, documents, and other private information sources.

-----

-----

-----

### Source [69]: https://learnopencv.com/langgraph-self-correcting-agent-code-generation/

Query: How are multimodal RAG systems integrated as tools into agentic frameworks like LangGraph to enable reasoning over private visual and textual data?

Answer: LearnOpenCV details how LangGraph’s agent state can be extended for advanced agentic workflows, such as self-correcting code generation, by tracking structured outputs, error statuses, and conversation history. For multimodal RAG, such a state would naturally include fields for **visual inputs (e.g., image embeddings, OCR results)** and **textual context**.

LangGraph’s **message reducer** ensures that all data—whether from human, AI, or tool sources—are appended to the state, enabling the agent to maintain a full, multimodal conversational history. This is crucial for reasoning over complex or private data, as the agent can reference past visual or textual context at any point.

Conditional control flow (edges) can be triggered based on the state, allowing the agent to:
- Retry or refine queries based on retrieval/generation errors (including from visual processing steps).
- Iterate over different modalities as needed, or escalate to human review for ambiguous visual content.

This robustness and extensibility are key for integrating multimodal RAG tools into agentic frameworks like LangGraph.

-----

-----

-----

### Source [70]: https://www.langchain.com/langgraph

Query: How are multimodal RAG systems integrated as tools into agentic frameworks like LangGraph to enable reasoning over private visual and textual data?

Answer: The official LangGraph platform overview emphasizes its **controllable cognitive architecture** and support for diverse, complex agentic workflows. LangGraph enables easy configuration of tools—including RAG modules for both text and image data—within flexible, stateful agents.

Key features for multimodal RAG integration include:
- **Modular, easily configurable tools** for retrieval and generation, allowing agents to switch between or combine visual and textual reasoning as required.
- Built-in support for collaboration and human-agent review, which is especially important when reasoning over sensitive private data (e.g., images or confidential documents).
- **Visualization and inspection tools** for development and debugging, making it easier to understand and optimize how multimodal RAG steps fit into the overall reasoning graph.

LangGraph’s design ensures that multimodal RAG can be integrated as one of many tools, with agents orchestrating complex, multi-tool reasoning processes over private, multi-modal data.

-----

-----

### Source [81]: https://www.youtube.com/watch?v=uLrReyH5cu0

Query: How are multimodal RAG systems integrated as tools into agentic frameworks like LangGraph to enable reasoning over private visual and textual data?

Answer: This tutorial demonstrates how to build a **multimodal Retrieval-Augmented Generation (RAG) pipeline** using LangChain and the Unstructured library, providing practical steps for integrating both visual and textual data into agentic frameworks. The process starts by using the Unstructured library to **parse and pre-process various document types**, including PDFs with text, images, tables, and plots. This parsed multimodal content is then indexed so it can be retrieved effectively based on user queries.

LangChain is used to **set up a document retrieval system** that can access both text and visual elements. When a query is received, the system leverages a **multimodal LLM (like GPT-4 with vision)** to analyze and answer questions about the retrieved content, enabling reasoning that combines insights from both formats. This approach allows agents to provide responses grounded in the actual content of complex documents, expanding the system's applicability to scenarios such as technical documentation, scientific papers, or presentations that mix text and images.

The integration essentially involves:
- Using specialized parsing tools (Unstructured) to extract and store both visual and textual features.
- Building retrieval logic with LangChain, ensuring both modalities can be surfaced as context for the LLM.
- Employing a multimodal LLM as the reasoning engine that can interpret and generate answers based on the combined input.

This framework is suitable for private data, as all processing—including parsing, retrieval, and generation—can be kept on-premise or within a secure environment, giving organizations control over sensitive information.

-----

-----

-----

### Source [82]: https://python.langchain.com/docs/tutorials/rag/

Query: How are multimodal RAG systems integrated as tools into agentic frameworks like LangGraph to enable reasoning over private visual and textual data?

Answer: The official LangChain RAG tutorial describes how **agentic frameworks like LangGraph** orchestrate the sequence of retrieval and generation steps for RAG systems. The application's **state** holds all necessary information, such as the user query, retrieved context, and the final answer. Each step (node) in the workflow—like retrieving documents or generating answers—is defined as a function operating on this state.

When a user submits a question, the **retrieve** node searches for relevant documents (these could include both text and images, depending on the retrieval setup). The **generate** node then passes this context, along with the question, to a chat model (potentially a multimodal LLM), which produces the answer.

The entire process is compiled into a **graph object** using LangGraph, making it easy to define, visualize, and extend the workflow. This modular design allows for easy substitution or augmentation of nodes, such as introducing specialized visual parsing or reasoning steps. The agentic framework thus enables complex, customizable, and auditable reasoning over multimodal (textual and visual) private data by chaining together the appropriate tools and models within a clear, inspectable state machine.

-----

-----

-----

### Source [83]: https://www.langchain.com/langgraph

Query: How are multimodal RAG systems integrated as tools into agentic frameworks like LangGraph to enable reasoning over private visual and textual data?

Answer: LangGraph is positioned as a **controllable cognitive architecture** for building agentic applications, including those that require reasoning over multimodal (text and visual) private data. The framework supports diverse control flows—like single agent, multi-agent, or hierarchical—and is designed for **integration of tools, prompts, and models** as configurable components.

LangGraph’s strengths in this context include:
- **Statefulness:** Each agent maintains a persistent state throughout the workflow, which is essential for tracking multimodal data as it moves between tools (e.g., from visual parsers to retrieval engines to LLMs).
- **Expressive control flows:** Developers can build workflows that route queries through different tools depending on the data type or required reasoning (e.g., text-only vs. visual-and-text queries).
- **Human-in-the-loop:** The platform supports human review and correction, which is valuable for sensitive or ambiguous multimodal reasoning tasks involving private data.
- **Scalability and Monitoring:** The platform offers deployment and monitoring tools for scaling agentic applications that process private multimodal data securely.

By abstracting away infrastructure and providing primitives for tool integration, LangGraph enables developers to focus on the logic of **combining multimodal RAG components** (like visual parsers, retrieval systems, and multimodal LLMs) within flexible, agent-driven workflows.

-----

-----

-----

### Source [84]: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag_local/

Query: How are multimodal RAG systems integrated as tools into agentic frameworks like LangGraph to enable reasoning over private visual and textual data?

Answer: This tutorial illustrates how to use LangGraph to construct a **local RAG agent** that can be adapted for multimodal workflows. The framework combines several advanced RAG strategies:
- **Routing:** Directs queries to different retrieval approaches based on the question type, which is essential for multimodal data (e.g., routing image-based questions to visual parsers).
- **Fallback and Self-correction:** Includes mechanisms to handle irrelevant or hallucinated results, which is particularly important when integrating multiple data modalities.

The tutorial demonstrates integration with local models (like LLaMA3) and embedding tools, showing how retrieval and generation steps can be customized for private, on-premise deployment—key for handling confidential visual and textual data.

LangGraph's structure allows each step to be independently configured, so developers can introduce specialized parsing, retrieval, or reasoning modules for handling images and text. This flexibility is central to enabling **agentic, auditable, and privacy-preserving multimodal RAG workflows**.

-----

-----

-----

### Source [85]: https://python.langchain.com/docs/concepts/multimodality/

Query: How are multimodal RAG systems integrated as tools into agentic frameworks like LangGraph to enable reasoning over private visual and textual data?

Answer: The official LangChain multimodality documentation outlines support for **multimodal inputs and outputs** in chat models integrated via LangChain. Developers can identify which models (e.g., GPT-4 Vision) support multimodal reasoning and reference provider-specific formats or cross-provider standards for input.

While no chat model currently works **directly** with multimodal data as part of tool call requests, the documentation suggests that multimodal support is achieved by:
- Formatting and passing both textual and visual data to compatible chat models.
- Receiving and processing **multimodal outputs** (such as text responses or generated images) as part of the AIMessage object.

This approach is compatible with agentic frameworks like LangGraph, where multimodal RAG systems are treated as modular tools within the workflow, enabling **reasoning over both private visual and textual data** by leveraging the underlying model’s capabilities.

-----

</details>

<details>
<summary>What are the performance benchmarks and error analyses for traditional OCR systems on complex documents like financial reports and technical diagrams?</summary>

### Source [71]: https://www.dataunboxed.io/blog/ocr-vs-vlm-ocr-naive-benchmarking-accuracy-for-scanned-documents

Query: What are the performance benchmarks and error analyses for traditional OCR systems on complex documents like financial reports and technical diagrams?

Answer: This source presents a benchmark comparing traditional OCR technologies with Vision Language Models (VLMs) using the FUNSD dataset of noisy scanned forms, which includes complex document layouts. The evaluation covers metrics such as **text similarity**, **word error rate**, **character error rate**, and **processing time**. Key findings include:
- **VLMs (Qwen, Mistral) greatly outperform traditional OCR** in accuracy for documents with complex layouts and poor scan quality; however, they require longer processing times.
- The benchmark did not involve pre-processing, which is often necessary for optimal OCR performance but can introduce variability in real-world scenarios.
- Practical recommendations are provided: traditional OCR may suffice for simple layouts or high-volume, cost-sensitive tasks, whereas VLMs are preferable for highly complex documents where accuracy is paramount.
- The study highlights a lack of comprehensive public benchmarks covering traditional OCR versus VLMs specifically for complex scanned documents, motivating further research.

-----

-----

-----

### Source [72]: https://www.mixedbread.com/blog/the-hidden-ceiling

Query: What are the performance benchmarks and error analyses for traditional OCR systems on complex documents like financial reports and technical diagrams?

Answer: This source details the **OHR (OCR hinders RAG) Benchmark v2**, designed to measure OCR performance on challenging real-world documents (8,500+ pages from seven enterprise domains, including finance and technical manuals). It features:
- Documents with **complex layouts, tables, formulas, charts, diagrams**, and non-standard reading orders, all known to challenge OCR.
- **8,498 targeted QA pairs** to probe retrieval and understanding, grounded in human-verified ground truth.
- All tested OCR solutions (traditional, open-source, and commercial) **underperform compared to perfect text ground truth**—establishing a persistent "performance ceiling."
- The benchmark isolates the impact of OCR errors on downstream retrieval and QA tasks, showing that even small recognition mistakes can severely limit system performance on information extraction and question answering for complex documents.

-----

-----

-----

### Source [73]: https://www.3rdaiautomation.com/blog/benchmarking-QA-on-complex-indestrial-PDFs

Query: What are the performance benchmarks and error analyses for traditional OCR systems on complex documents like financial reports and technical diagrams?

Answer: This source argues that traditional OCR benchmarks—focusing on **word/character accuracy metrics**—do not fully capture performance on complex industrial documents. It introduces the **TIA-pdf-QA-Bench** for evaluating end-to-end QA pipelines over semi-structured PDFs. Insights include:
- OCR errors can have disproportionate downstream effects: a single misrecognized term may not impact raw OCR scores much but can derail retrieval or QA if the term is critical (e.g., a parameter in a financial report or technical diagram).
- Real challenge lies in **retrieval and semantic understanding**, not merely text extraction.
- Industrial and technical documents often have long, heterogeneous formatting, intricate tables and figures, implicit references, and dense hierarchical structures. These features compound OCR errors and complicate chunking and indexing.
- Poor text chunking and representation significantly degrade QA performance, resulting in missed answers, irrelevant retrievals, and model hallucinations.

-----

-----

-----

### Source [74]: https://getomni.ai/blog/ocr-benchmark

Query: What are the performance benchmarks and error analyses for traditional OCR systems on complex documents like financial reports and technical diagrams?

Answer: This benchmark evaluates **10 popular OCR providers** and Vision Language Models (VLMs) on 1,000 real-world documents, including those with **charts, infographics, handwriting, and complex input fields**. Key points:
- **VLMs outperform traditional OCR** on documents with noisy backgrounds, charts, handwriting, and complex fields, handling scan artifacts more robustly.
- **Traditional OCR performs better on high-density text pages** (e.g., textbooks, research papers) and standardized forms (e.g., tax forms).
- Error analysis uses **JSON accuracy**: the number of differences between OCR output and ground truth JSON fields.
- **Processing time** and **cost per page** are tracked, showing that VLMs are slower and potentially more expensive (due to output token costs), but their accuracy gains may justify the tradeoff on complex documents.
- VLMs may refuse to process certain document types due to content policies (e.g., photo IDs, passports), introducing unpredictable error modes.

-----

-----

-----

### Source [75]: https://arxiv.org/html/2412.02210v2

Query: What are the performance benchmarks and error analyses for traditional OCR systems on complex documents like financial reports and technical diagrams?

Answer: This comprehensive benchmark (CC-OCR) covers four tracks: **Conventional OCR, Multilingual OCR, Document Parsing, and Key Information Extraction**, with a focus on **formula recognition, table and chart analysis, element detection, and layout analysis**. Relevant points:
- Most Large Multimodal Models (LMMs) and OCR systems are evaluated primarily on scanned documents, but real-world documents introduce significant noise (shadows, folds, material textures, diverse backgrounds) that **limit robustness** in both text and structural recognition.
- The benchmark emphasizes scenarios that challenge current models, including **financial and technical documents with complex visual and semantic structures**.
- Error analysis reveals that **structural and environmental noise**—not just text distortion—significantly hampers OCR and parsing accuracy on financial reports and technical diagrams.
- Benchmark results indicate that **current OCR systems struggle with layout analysis, formula parsing, and key information extraction** when documents deviate from clean, standardized formats.

-----

-----

</details>


## Sources Scraped From Research Results

<details>
<summary>Unstructured data encompasses a wide array of information types that do not conform to predefined data models or organized in traditional relational databases. This includes text documents, emails, social media posts, images, audio files, videos, and sensor data. The inherent lack of structure makes this data difficult to process using conventional methods, yet it often contains valuable insights that can drive innovation, improve decision-making, and enhance customer experiences.</summary>

##### Unstructured data encompasses a wide array of information types that do not conform to predefined data models or organized in traditional relational databases. This includes text documents, emails, social media posts, images, audio files, videos, and sensor data. The inherent lack of structure makes this data difficult to process using conventional methods, yet it often contains valuable insights that can drive innovation, improve decision-making, and enhance customer experiences.

The rise of generative AI and large language models (LLMs) has further emphasized the importance of effectively managing unstructured data. These models require vast amounts of diverse, high-quality data for training and fine-tuning. Additionally, techniques like retrieval-augmented generation (RAG) rely on the ability to efficiently search and retrieve relevant information from large unstructured datasets.

###### Architectural Considerations for Unstructured Data Systems In Enterprise

Data Ingestion and Processing Architecture. The first challenge in dealing with unstructured data is ingestion. Unlike structured data, which can be easily loaded into relational databases, unstructured data requires specialized processing pipelines. These pipelines must be capable of handling a variety of data formats and sources, often in real-time or near-real-time, and at massive scale. For modern global enterprises, it’s crucial to design the ingestion architecture with global distribution in mind. **‍**

- Text-based Data. Natural language processing (NLP) techniques are essential for processing text-based data. This includes tokenization, part-of-speech tagging, named entity recognition, and sentiment analysis. Modern NLP pipelines often leverage deep learning models, such as BERT or GPT, which can capture complex linguistic patterns and context. At enterprise scale, these models may need to be deployed across distributed clusters to handle the volume of incoming data. Startups like [Hugging Face](https://huggingface.co/) provide transformer-based models that can be fine-tuned for specific enterprise needs, enabling sophisticated text analysis and generation capabilities.

- Image and Video Data. Computer vision algorithms are necessary for processing image and video data. These may include convolutional neural networks (CNNs) for image classification and object detection, or more advanced architectures like Vision Transformers (ViT) for tasks requiring understanding of spatial relationships. Processing video data, in particular, requires significant computational resources and may benefit from GPU acceleration. Notable startups such as [OpenCV.ai](https://www.opencv.ai/) are innovating in this space by providing open-source computer vision libraries and tools that can be integrated into enterprise workflows. Companies like [Roboflow](https://roboflow.com/) and [Encord](https://encord.com/) offer an end-to-end computer vision platform providing tools for data labeling, augmentation, and model training, making it easier for enterprises to build custom computer vision models. Their open-source YOLOv5 implementation has gained significant traction in the developer community. [Voxel51](https://voxel51.com/) is tackling unstructured data retrieval in computer vision with their open-source [FiftyOne](https://docs.voxel51.com/index.html) platform, which enables efficient management, curation, and analysis of large-scale image and video datasets. [Coactive](https://coactive.ai/) is leveraging unstructured data retrieval across multiple modalities with their neural database technology, designed to efficiently store and query diverse data types including text, images, and sensor data.
- Audio Data. Audio data presents its own set of challenges, requiring speech-to-text conversion for spoken content and specialized audio analysis techniques for non-speech sounds. Deep learning models like wav2vec and HuBERT have shown promising results in this domain. For enterprises dealing with large volumes of audio data, such as call center recordings, implementing a distributed audio processing pipeline is crucial. Companies like [Deepgram](https://deepgram.com/) and [AssemblyAI](https://www.assemblyai.com/) are leveraging end-to-end deep learning models to provide accurate and scalable speech recognition solutions.

To handle the diverse nature of unstructured data, organizations should consider implementing a modular, event-driven ingestion architecture. This could involve using Apache Kafka or Apache Pulsar for real-time data streaming, coupled with specialized processors for each data type. [RedPanda](https://redpanda.com/what-is-redpanda) built an open-source data streaming platform designed to replace Apache Kafka with lower latency and higher throughput. Containerization technologies like Docker and orchestration platforms like Kubernetes can provide the flexibility needed to scale and manage these diverse processing pipelines. [Graphlit](https://www.graphlit.com/) build a data platform designed for spatial and unstructured data files automating complex data workflows, including data ingestion, knowledge extraction, LLM conversations, semantic search, and application integrations.

Data Storage and Retrieval. Traditional relational databases are ill-suited for storing and querying large volumes of unstructured data. Instead, organizations must consider a range of specialized storage solutions. For raw unstructured data, object storage systems like Amazon S3, Google Cloud Storage, or Azure Blob Storage provide scalable and cost-effective options. These systems can handle petabytes of data and support features like versioning and lifecycle management. [MinIO](https://min.io/) developed an open-source, high-performance, distributed object storage system designed for large-scale unstructured data. For semi-structured data, document databases like MongoDB or Couchbase offer flexible schemas and efficient querying capabilities. These are particularly useful for storing JSON-like data structures extracted from unstructured sources. [SurrealDB](https://surrealdb.com/) is a multi-model, cloud-ready database allows developers and organizations to meet the needs of their applications, without needing to worry about scalability or keeping data consistent across multiple different database platforms, making it suitable for modern and traditional applications. As machine learning models increasingly represent data as high-dimensional vectors, vector databases have emerged as a crucial component of the unstructured data stack. Systems like [LanceDB](https://lancedb.com/), [Marqo](https://www.marqo.ai/), [Milvus](https://milvus.io/), and [Vespa](https://vespa.ai/) are designed to efficiently store and query these vector representations, enabling semantic search and similarity-based retrieval. For data with complex relationships, graph databases like Neo4j or Amazon Neptune can be valuable. These are particularly useful for representing knowledge extracted from unstructured text, allowing for efficient traversal of relationships between entities. [TerminusDB](https://terminusdb.com/), an open-source graph database, can be used for representing and querying complex relationships extracted from unstructured text. This approach is particularly useful for enterprises needing to traverse relationships between entities efficiently. [Kumo AI](https://kumo.ai/) developed graph machine learning-centered AI platform that uses LLMs and graph neural networks (GNNs) designed to manage large-scale data warehouses, integrating ML between modern cloud data warehouses and AI algorithms infrastructure to simplify the training and deployment of models on both structured and unstructured data, enabling businesses to make faster, simpler, and more accurate predictions. [Roe AI](https://www.getroe.ai/) has built AI-powered data warehouse to store, process, and query unstructured data like documents, websites, images, videos, and audio by providing multi-modal data extraction, data classification and multi-modal RAG via Roe’s SQL engine.

When designing the storage architecture, it’s important to consider a hybrid approach that combines these different storage types. For example, raw data might be stored in object storage, processed information in document databases, vector representations in vector databases, and extracted relationships in graph databases. This multi-modal storage approach allows for efficient handling of different query patterns and use cases.

Data Processing and Analytics. Processing unstructured data at scale requires distributed computing frameworks capable of handling large volumes of data. Apache Spark remains a popular choice due to its versatility and extensive ecosystem. For more specialized workloads, frameworks like [Ray](https://www.ray.io/) are gaining traction, particularly for distributed machine learning tasks. For real-time processing, stream processing frameworks like Apache Flink or Kafka Streams can be employed. These allow for continuous processing of incoming unstructured data, enabling real-time analytics and event-driven architectures. When it comes to analytics, traditional SQL-based approaches are often insufficient for unstructured data. Instead, architecture teams should consider implementing a combination of techniques including (i) engines like [Elasticsearch](https://www.elastic.co/elasticsearch) or Apache Solr provide powerful capabilities for searching and analyzing text-based unstructured data; (ii) for tasks like classification, clustering, and anomaly detection, machine learning models can be deployed on processed unstructured data. Frameworks like TensorFlow and PyTorch, along with managed services like Google Cloud AI Platform or Amazon SageMaker, can be used to train and deploy these models at scale; (iii) for data stored in graph databases, specialized graph analytics algorithms can uncover complex patterns and relationships. [OmniAI](https://getomni.ai/) developed a data transformation platform designed to convert unstructured data into accurate, tabular insights while maintaining control over their data and infrastructure. Roe AI

To enable flexible analytics across different data types and storage systems, architects should consider implementing a data virtualization layer. Technologies like [Presto](https://prestodb.io/) or [Dremio](https://www.dremio.com/) can provide a unified SQL interface across diverse data sources, simplifying analytics workflows. [Vectorize](https://vectorize.io/) is developing a streaming database for real-time AI applications to bridge the gap between traditional databases and the needs of modern AI systems, enabling real-time feature engineering and inference.

Data Governance and Security. Unstructured data often contains sensitive information, making data governance and security critical considerations. Organizations must implement robust mechanisms for data discovery, classification, and access control. Automated data discovery and classification tools such as [Sentra Security](https://www.sentra.io/), powered by machine learning, can scan unstructured data to identify sensitive information and apply appropriate tags. These tags can then be used to enforce access policies and data retention rules. For access control, attribute-based access control (ABAC) systems are well-suited to the complex nature of unstructured data. ABAC allows for fine-grained access policies based on attributes of the data, the user, and the environment. Encryption is another critical component of securing unstructured data. This includes both encryption at rest and in transit. For particularly sensitive data, consider implementing field-level encryption, where individual elements within unstructured documents are encrypted separately.

###### Emerging Technologies and Approaches

LLMs like GPT-3 and its successors have demonstrated remarkable capabilities in understanding and generating human-like text. These models can be leveraged for a wide range of tasks, from text classification and summarization to question answering and content generation. For enterprises, the key challenge remains adapting these models to domain-specific tasks and data. Techniques like fine-tuning and prompt engineering allow for customization of pre-trained models. Additionally, approaches like retrieval-augmented generation (RAG) enable these models to leverage enterprise-specific knowledge bases, improving their accuracy and relevance. Implementing a modular architecture that allows for easy integration of different LLMs and fine-tuned variants might involve setting up model serving infrastructure using frameworks like TensorFlow Serving or Triton Inference Server, coupled with a caching layer to improve response times. Companies like [Unstructured](https://unstructured.io/) use open-source libraries and application programming interfaces to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines, enabling clients to transform simple data into language data and write it to a destination (vector database or otherwise).

Multi-modal AI Models. As enterprises deal with diverse types of unstructured data, multi-modal AI models that can process and understand different data types simultaneously are becoming increasingly important. Models like CLIP (Contrastive Language-Image Pre-training) demonstrate the potential of combining text and image understanding. To future proof organizational agility, systems need to be designed to handle multi-modal data inputs and outputs, potentially leveraging specialized hardware like GPUs or TPUs for efficient processing as well as implementing a pipeline architecture that allows for parallel processing of different modalities, with a fusion layer that combines the results. [Adept AI](https://www.adept.ai/) is working on AI models that can interact with software interfaces, potentially changing how enterprises interact with their digital tools, combining language understanding with the ability to take actions in software environments. In the defense sector, [Helsing AI](https://helsing.ai/) is developing advanced AI systems for defense and national security applications that process and analyze vast amounts of unstructured sensor data in real-time, integrating information from diverse sources such as radar, electro-optical sensors, and signals intelligence to provide actionable insights in complex operational environments. In industrial and manufacturing sectors, [Archetype AI](https://www.archetypeai.io/) offers a multimodal AI foundation model that fuses real-time sensor data with natural language, enabling individuals and organizations to ask open-ended questions about their surroundings and take informed action for improvement.

Federated Learning. For enterprises dealing with sensitive or distributed unstructured data, federated learning offers a way to train models without centralizing the data. This approach allows models to be trained across multiple decentralized devices or servers holding local data samples, without exchanging them. Implementing federated learning however requires careful design, including mechanisms for model aggregation, secure communication, and differential privacy to protect individual data points. Frameworks like TensorFlow Federated or PySyft can be used to implement federated learning systems. For example, in the space of federated learning for healthcare and life sciences, [Owkin](https://www.owkin.com/) enables collaborative research on sensitive medical data without compromising privacy.

Synthetic Data Generation. The scarcity of labeled unstructured data for specific domains or tasks can be a significant challenge. Synthetic data generation, often powered by generative adversarial networks (GANs) or other generative models, may offer a solution to this problem. Incorporating synthetic data generation pipelines into machine learning workflows might involve setting up separate infrastructure for data generation and validation, ensuring that synthetic data matches the characteristics of real data while avoiding potential biases. [RAIC Labs](https://raiclabs.com/) is developing technology for rapid AI modeling with minimal data. Their RAIC (Rapid Automatic Image Categorization) platform can generate and categorize synthetic data, potentially solving the cold start problem for many machine learning applications.

Knowledge Graphs. Knowledge graphs offer a powerful way to represent and reason about information extracted from unstructured data. Startups like [Diffbot](https://www.diffbot.com/) are developing automated knowledge graph construction tools that use natural language processing, entity resolution, and relationship extraction techniques to build rich knowledge graphs. These graphs capture the semantics of unstructured data, enabling efficient querying and reasoning about the relationships between entities. Implementing knowledge graphs involves (i) entity extraction and linking to identify and disambiguate entities mentioned in unstructured text; (ii) relationship extraction to determine the relationships between entities; (iii) ontology management to define and maintain the structure of the knowledge graph; and (iv) graph storage and querying for efficiently storing and querying the resulting graph structure.

Businesses should consider using a combination of machine learning models for entity and relationship extraction, coupled with specialized graph databases for storage. Technologies like RDF (Resource Description Framework) and SPARQL can be used for semantic representation and querying.

While the potential of unstructured data is significant, several challenges must be addressed with most important are scalability, data quality and cost. Processing and analyzing large volumes of unstructured data requires significant computational resources. Systems must be designed that can scale horizontally, leveraging cloud resources and distributed computing frameworks. Unstructured data often contains noise, inconsistencies, and errors. Implementing robust data cleaning and validation pipelines is crucial for ensuring the quality of insights derived from this data. [Galileo](http://www.rungalileo.io/) developed an engine that processes unlabeled data to automatically identify error patterns and data gaps in the model, enabling organizations to improve efficiencies, reduce costs, and mitigate data biases. [Cleanlab](https://cleanlab.ai/) developed an automated data-centric platform designed to help enterprises improve the quality of datasets, diagnose or fix issues and produce more reliable machine learning models by cleaning labels and supporting finding, quantifying, and learning data issues. Processing and storing large volumes of unstructured data can be expensive. Implementing data lifecycle management, tiered storage solutions, and cost optimization strategies is crucial for managing long-term costs. For example, [Bem](https://www.bem.ai/)’s data interface transforms any input into ready-to-use data, eliminating the need for costly and time-consuming manual processes. Lastly, as machine learning models become more complex, ensuring interpretability of results becomes challenging. Techniques like SHAP (SHapley Additive exPlanations) values or LIME (Local Interpretable Model-agnostic Explanations) can be incorporated into model serving pipelines to provide explanations for model predictions. Unstructured data also often contains sensitive information, and AI models trained on this data can perpetuate biases. Architects must implement mechanisms for bias detection and mitigation, as well as ensure compliance with data protection regulations.

Unstructured data presents both significant challenges and opportunities for enterprises. By implementing a robust architecture that can ingest, store, process, and analyze diverse types of unstructured data, enterprises can unlock valuable insights and drive innovation. Businesses must stay abreast of emerging technologies and approaches, continuously evolving their data infrastructure to handle the growing volume and complexity of unstructured data. By combining traditional data management techniques with cutting-edge AI and machine learning approaches, enterprises can build systems capable of extracting maximum value from their unstructured data assets. As the field continues to evolve rapidly, flexibility and adaptability should be key principles in any unstructured data architecture. By building modular, scalable systems that can incorporate new technologies and handle diverse data types, enterprises can position themselves to leverage the full potential of unstructured data in the years to come.

</details>

<details>
<summary>Here’s a common scenario when building AI agents that might feel confusing: How can you use the latest Gemini models and an open-source framework like LangChain and LangGraph to create multimodal agents that can detect objects?</summary>

Here’s a common scenario when building AI agents that might feel confusing: How can you use the latest Gemini models and an open-source framework like LangChain and LangGraph to create multimodal agents that can detect objects?

Detecting objects is critically important for use cases from content moderation to multimedia search and retrieval. Langchain provides tools to chain together LLM calls and external data. LangGraph provides a graph structure to build more controlled and complex multiagents apps.

In this post, we’ll show you which decisions you need to make to combine Gemini, LangChain and LangGraph to build multimodal agents that can identify objects. This will provide a foundation for you to start building enterprise use cases like:

- **Content moderation**: [Advertising policies](https://support.google.com/adspolicy/answer/13584894?hl=en), movie ratings, brand infringement
- **Object identification**: Using different sources of data to verify if an object exist on a map
- **Multimedia search and retrieva** l: Finding files that contains a specific object

### **First decision: No-code/low-code, or custom agents?**

The first decision enterprises have to decide is: no-code/low-code options or build custom agents? If you are building a simple agent like a customer service chat bot, you can use Google’s Vertex AI [Agent Builder](https://cloud.google.com/products/agent-builder?utm_source=google&utm_medium=cpc&utm_campaign=na-US-all-en-dr-bkws-all-all-trial-e-dr-1710134&utm_content=text-ad-none-any-DEV_c-CRE_706196044739-ADGP_Hybrid+%7C+BKWS+-+MIX+%7C+Txt-AI+and+Machine+Learning-Agent+Builder-KWID_43700080502290554-kwd-2327285297755&utm_term=KW_google+vertex+ai+agent+builder-ST_google+vertex+ai+agent+builder&gad_source=1&gad_campaignid=20363681466&gclid=Cj0KCQjwxdXBBhDEARIsAAUkP6ht8a3k7Vc6Y9eGdb4xkfx0MQ3_XIWcXPyrdFc3TYCL4zaL4CzvtwwaAmEjEALw_wcB&gclsrc=aw.ds&e=48754805&hl=en) to build a simple agent in a few minutes or start from pre-built agents that are available in [Google Agentspace Agent Gallery](https://cloud.google.com/agentspace/agentspace-enterprise/docs/agents-gallery).

But if your use case requires orchestration of multiple agents and integration with custom tooling, you would have to build custom agents which leads to the next question.

### **Second decision: What agentic framework to use?**

It’s hard to keep up with so many agentic frameworks out there releasing new features every week. Top contenders include CrewAI, Autogen, LangGraph and Google’s ADK. Some of them, like ADK and CrewAI, have higher levels of abstraction while others like LangGraph allow higher degree of control.

That’s why in this blog, we center the discussion on building a custom agent using the open-sourced LangChain, LangGraph as an agentic framework, and Gemini 2.0 Flash as the LLM brain.

### **Code deep dive**

This [example code](https://github.com/Mayshinlyan/generative-ai) identifies an object in an image, in an audio file, and in a video. In this case we will use a dog as the object to be identified. We have different agents (image analysis agent, audio analysis agent, and a video analysis agent) performing different tasks but all working together towards a common goal, object identification.https://storage.googleapis.com/gweb-cloudblog-publish/images/1_Ai6ddoG.max-1800x1800.png

Generative AI workflow for object detection

This gen AI workflow entails a user asking the agent to verify if a specific object exists in the provided files. The Orchestrator Agent will call relevant worker agents: image\_agent, audio\_agent, and video\_agent while passing the user question and the relevant files. Each worker agent will call respective tooling to convert the provided file to base64 encoding. The final finding of each agent is then passed back to the Orchestrator Agent. The Orchestrator Agent then synthesizes the findings and makes the final determination. This code can be used as a starting point template where you need to ask an agent to reason and make a decision or generate conclusions from different sources.

If you want to create multiagent systems with ADK, here is a [video production agent](https://github.com/byronwhitlock-google/adk-playground/) built by a Googler which generates video commercials from user prompts and utilizes Veo for video content generation, Lyria for composing music, and Google Text-to-Speech for narration. This example demonstrates the fact that many ingredients can be used to meet your agentic goals, in this case an [AI agent as a production studio](https://medium.com/google-cloud/did-your-next-viral-commercial-just-get-built-by-googles-adk-ai-seriously-it-did-620818d54237). If you want to try ADK, here is an [ADK Quickstart](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-development-kit/quickstart) to help you kick things off.

### **Third decision: Where to deploy the agents?**

If you are building a simple app that needs to go live quickly, Cloud Run is an easy way to deploy your app. Just like any serverless web app, you can follow the same instructions to deploy on Cloud Run. Watch this video of [building AI agents on Cloud Run](https://www.youtube.com/watch?v=GwL8e5Z1tl4). However, if you want more enterprise grade managed runtime, quality and evaluation, managing context and monitoring, Agent Engine is the way to go. Here is [a quick start for Agent Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/quickstart). Agent Engine is a fully managed runtime which you can integrate with many of the previously mentioned frameworks – ADK, LangGraph, Crew.ai, etc (see the image below, from the official [Google Cloud Docs](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview)).https://storage.googleapis.com/gweb-cloudblog-publish/images/2_MbjVrzs.max-1900x1900.png

### **Get started**

Building intelligent agents with generative AI, especially those capable of multimodal understanding, is akin to solving a complex puzzle. Many developers are finding that a prototypical agentic build involves a LangChain agent with Gemini Flash as the LLM. This post explored how to combine the power of **Gemini models** with open-source frameworks like **LangChain** and **LangGraph.** To get started right away, use this [ADK Quickstart](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-development-kit/quickstart) and or visit our [Agent Development GitHub](https://google.github.io/adk-docs/).

</details>

<details>
<summary>This brief shares practical insights from Turing AGI Advancement’s recent collaboration with a leading AI lab to build large-scale multimodal datasets, spanning vision, language, and audio, for instruction tuning, alignment, and evaluation. The work involved scaling human-in-the-loop (HITL) systems, evolving taxonomies, and navigating ethical and operational complexities in multimodal data generation.</summary>

This brief shares practical insights from Turing AGI Advancement’s recent collaboration with a leading AI lab to build large-scale multimodal datasets, spanning vision, language, and audio, for instruction tuning, alignment, and evaluation. The work involved scaling human-in-the-loop (HITL) systems, evolving taxonomies, and navigating ethical and operational complexities in multimodal data generation.

## The multimodal data challenge: Getting beyond volume

As organizations push toward more capable and generalizable multimodal LLMs, data becomes a major bottleneck, not just in terms of quantity, but in strategic composition and delivery. Three interlinked challenges consistently emerge:

1. **Quality**: Multimodal data often suffers from weak alignment, poor annotation consistency, or low contextual relevance, especially when scaling across languages, formats, or domains. Without clear calibration, even sophisticated prompts can generate noisy or misaligned outputs. Projects that lack robust quality-control pipelines frequently encounter slow feedback loops, annotation drift, and uneven performance across modalities.



_**In a high-volume vision understanding project, our teams had to adapt to evolving taxonomies midstream, quickly recalibrating on new categories without compromising annotation quality.**_
2. **Diversity**: It’s not enough to gather a large dataset; the data needs to reflect a diverse set of capabilities, users, and real-world contexts. Rigid or outdated taxonomies often prevent full-spectrum coverage. Diversity needs to be actively designed: across demographics, domains, modalities, and intent types.



_**Recently, a medical and safety-critical project required sourcing varied data like imaging modalities, accents, or multilingual prompts, sometimes requiring new sourcing strategies to cover underrepresented cases.**_
3. **Efficiency**: Manual processes quickly become a bottleneck. Without intelligent tooling and structured agentic flows, annotation becomes slow and inconsistent, especially when working across vision, audio, and text.



_**Our Pod-based HITL teams with embedded reviewers and automated support systems enabled fast adaptation to delivery goals, even as category definitions changed midstream.**_

## Strategic tradeoffs in multimodal data development

Building high-quality multimodal datasets isn’t just about applying best practices; it’s about navigating tensions between quality, diversity, and efficiency. Teams frequently encounter tradeoffs such as:

- **Speed vs. quality:** Automation can accelerate throughput but may increase label noise if not paired with rigorous validation.
- **Diversity vs. efficiency:** Expanding taxonomy coverage or sourcing rare data types slows delivery unless supported by adaptable workflows.
- **Consistency vs. flexibility:** Midstream updates to category definitions or prompt formats can improve coverage but require recalibration and retraining.

_**When a vision RLHF project introduced new label categories mid-delivery, our team avoided disruption by using calibration gold sets and a pod-based knowledge transfer model.**_

At Turing, we work closely with clients to strike the right balance for their model goals, customizing solutions based on the tradeoffs that matter most for their use case.

## The real-world playbook for building better multimodal datasets

Solving for scale, quality, and diversity in multimodal datasets requires flexible frameworks, evolving taxonomies, and agentic HITL systems that adapt to the pace of model development. Below is a breakdown of how each core challenge can be tackled, with lessons drawn from real-world deployments.

### 1\. Improving quality with AI-driven calibration and QA systems

Ensuring consistent, high-quality data across modalities means actively aligning annotators, prompts, models, and review mechanisms.

**Key practices:**

- **Gold sets for calibration** Standardized datasets help benchmark annotator accuracy and consistency. These are essential for onboarding, drift detection, and grounding feedback discussions.



_**Our projects with early-stage gold batches established strong baselines that helped us maintain quality even when category definitions shifted.**_
- **Iterative refinement loops** Structured retrospectives across training cycles uncover prompt failure patterns or annotation mismatches. Teams revise annotation guidelines and evolve processes continuously, avoiding the “set and forget” trap.



_**Instruction tuning workflows helped us regularly refine prompt formats after observing reward model confusion, limiting misalignment downstream.**_
- **AI-driven calibration techniques** Models can help calibrate themselves. Methods like uncertainty quantification and expected calibration error (ECE) reduction serve as proxies for annotation quality, especially when paired with real-time dashboards that track drift and gaps. “Garbage in, garbage out” becomes real when misaligned annotations ripple into RLHF confusion or degraded eval performance.
- **QA pipelines as continuous systems** Quality assurance is layered and modality-specific. Tiered QA, including golden comparisons, consensus checks, and post-model evaluation, keeps the loop active and actionable.



_**We created vision datasets that passed through a two-stage QA process combining Gemini-based LLM screening with final human review.**_
- **Knowledge transfer through pod-based structures** To ensure annotation consistency at scale, teams often use pod-based systems, where experienced leads guide sub-groups, reinforce standards, and resolve edge cases. This structure supports fast onboarding and helps maintain quality under shifting conditions.



_**When working with large, distributed teams, this structure ensured that domain knowledge and updated calibration rules were quickly propagated without introducing drift.**_

### 2\. Designing for diversity with flexible, evolving taxonomies

Multimodal model robustness depends on exposure to diverse tasks, formats, and viewpoints, but that only happens when taxonomies are built to evolve and capture complexity.

**Key practices:**

- **Clear taxonomy definitions and evolution plans** Categories must come with detailed definitions, edge cases, and intended use documentation. As projects grow, taxonomies often expand or shift, and systems need to absorb those changes without breaking.



_**When new categories were introduced midstream in a vision RL project, strong calibration protocols allowed the change to propagate in hours, not days.**_
- **Embedding diversity in the design layer** Structured prompt templates, like Anthropic’s “helpful, honest, harmless” criteria, help steer data collection toward broader representations and reduce demographic blind spots.



Another example is Hugging Face Dataset Cards that have become a common tool to explicitly state labeling rationales, ethical considerations, and representational gaps.
- **Coverage-aware monitoring** Real-time dashboards that track label distribution help avoid imbalance. If a taxonomy includes 20 categories but 80% of the data falls into 3, the model’s ability to generalize suffers.



_**Trade-off:**_ Coverage versus efficiency often emerges and teams must decide when to slow down to intentionally rebalance data flows.

### 3\. Driving efficiency with agentic HITL workflows

Manual annotation at scale is resource-intensive and slow. Teams are increasingly adopting [agentic workflows](https://www.turing.com/services/llm-agentic-workflows), where AI systems augment or automate parts of the workflow while humans remain in control.

**Key practices:**

- **HITL pods with embedded agents** Structured pods pair human annotators with AI agents that handle tasks like pre-labeling, ranking model responses, or routing ambiguous cases for human review.



_**These pods allowed rapid response when annotation criteria changed, without affecting delivery timelines.**_
- **Multi-LLM consensus frameworks** In many workflows, multiple LLMs provide candidate responses, which are then ranked or filtered using heuristic rules or crowd judgment. Disagreements trigger human review, improving both efficiency and reliability.



For example, [DeepMind’s approach to building RLHF reward data](https://arxiv.org/pdf/2209.14375) used multi-turn ranking tasks that distilled label consensus through preference-based voting.
- **Synthetic-human blends for annotation** High-volume tasks increasingly rely on synthetic content, especially in long-tail or sensitive domains, but require human review to maintain trust and safety.



_**Operationally**,_ these flows reduce annotation fatigue and increase consistency, while keeping a human-in-the-loop for final validation.
- **Real-time feedback infrastructure** Dashboards and coverage monitoring tools help teams detect drift, imbalances, or bottlenecks early, enabling faster iteration and resource shifts.

## Ethical considerations: Safety, fairness, and consent

Multimodal systems raise complex ethical challenges: visual bias, speech biometrics, data provenance, and more. Embedding ethical frameworks into pipeline design is critical—not just as a compliance step, but as part of scalable data governance.

- **Bias and fairness**

Detecting and mitigating bias requires both technical tooling (e.g., WEAT, iEAT) and procedural methods (e.g., counterfactual augmentation, diverse reviewer pools, taxonomy balancing). Teams must define what “fair” means for a given use case and constantly monitor for drift.



_**In language safety tasks, our teams often experienced annotator fatigue due to adversarial or disturbing prompts, highlighting the need for mental wellness guidelines and task rotation.**_
- **Privacy and consent**

Audio and image data introduce heightened sensitivity. Regulatory requirements differ by region, and what’s legal may not always align with ethical best practices. Data pipelines must integrate consent capture, differential privacy, and opt-out tooling from day one.



_**In voice capture projects, Turing teams designed collection flows that respected regional voice biometric laws, using synthetic noise overlays and in-field recordings for realism while protecting identity.**_

## Looking ahead: Self-improving pipelines and dynamic standards

The future of multimodal data development isn’t just about getting bigger, it’s about getting smarter. Emerging approaches like self-improving pipelines, where model feedback dynamically guides data generation and selection, promise to close the loop between modeling and data ops.

Just as importantly, teams must co-create dynamic standards, for diversity, quality, and fairness, that evolve with the landscape and are built with the same agility as the models they support.

</details>

<details>
<summary>The King of Multi-Modal RAG: ColPali</summary>

# The King of Multi-Modal RAG: ColPali

### Query your rich visual PDFs under a RAG web app

**What exactly is ColPali in the document retrieval world?**

Traditional RAG systems have always struggled with complex documents - losing formatting, missing tables, and fumbling with figures. They treat documents as plain text, throwing away the visual richness that makes them meaningful.

Instead of extracting text and hoping for the best, [ColPali](https://arxiv.org/pdf/2407.01449) treats documents as images, preserving every table, figure, and layout element. Like having a photographic memory for documents, it understands not just what's written, but how it's presented.

[https://substackcdn.com/image/fetch/$s_!LyEV!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F45e112b5-db9c-420c-9947-05e5593b1621_2044x1236.png](https://substackcdn.com/image/fetch/$s_!LyEV!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F45e112b5-db9c-420c-9947-05e5593b1621_2044x1236.png) Taken from [ColPali: EFFICIENT DOCUMENT RETRIEVAL WITH VISION LANGUAGE MODELS](https://arxiv.org/pdf/2407.01449)

In this tutorial, **we'll create a web API that uses ColPali for visual document retrieval**. Users can upload PDFs through REST endpoints, and the system will answer questions about those documents by understanding their visual layout - tables, figures, formatting and all.

ColPali represents a fundamental shift in how we approach document understanding. The original research introduced a clever technique: instead of extracting text from documents, they fed document images directly to a Vision-Language Model (PaliGemma-3B). The model splits each image into patches, processes them through a vision transformer, and generates contextualized embeddings in the language model space, essentially creating a visual understanding of the content.

Here's what makes this approach powerful: during indexing, the vision transformer encodes document images by splitting them into patches, which become "soft" tokens for the language model. This produces high-quality contextualized patch embeddings that capture both textual content and spatial relationships. These embeddings are then projected to a lower dimension for efficient storage, creating what the researchers call "multi-vector document representations."

The technique has evolved since the original paper. While ColPali established the methodology, newer implementations have experimented with different vision-language models as the backbone. The version we're using in this tutorial (ColQwen 2.5) leverages Qwen2.5-VL, a more efficient vision-language model that maintains the same core approach while offering improved performance.

## Understanding the Core Components of a ColPali RAG System

The backbone of a ColPali-based RAG application consists of five essential components that work together to create a visual document understanding pipeline:

1. **PDF to Image Converter**: Transforms PDF documents into high-quality images using pdf2image with optimized DPI settings. This preserves the document's visual integrity—every table border, diagram annotation, and layout relationship that would be lost in traditional text extraction.

2. **Storage Service:** Persists the converted images for retrieval during query processing. This component ensures document images remain accessible throughout the system's lifetime.

3. **ColPali Model (ColQwen 2.5)**: The visual understanding engine that generates embeddings directly from document images. Unlike traditional RAG models that process extracted text, ColQwen 2.5 captures spatial relationships, formatting context, and the interplay between text and visual elements.

4. **Vector Database (Qdrant)**: Stores and indexes the visual embeddings for efficient similarity search. Qdrant's multi-vector storage and cosine similarity support make it ideal for ColPali's embedding structure, enabling precise page-level retrieval with metadata.

5. **Multimodal LLM (Claude Sonnet 3.7)**: Interprets retrieved document images and generates responses. By working with actual images rather than extracted text, it can accurately reference tables, interpret figures, and maintain the document's original visual context.


#### How Data Flows Through the System

[https://substackcdn.com/image/fetch/$s_!XDu1!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff9ce6cf1-ff56-491f-9336-09a465cf9d99_3016x712.png](https://substackcdn.com/image/fetch/$s_!XDu1!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff9ce6cf1-ff56-491f-9336-09a465cf9d99_3016x712.png)

**Ingestion Pipeline:** During document ingestion, PDFs are first converted into JPEG images which are then uploaded to the storage service. ColQwen 2.5 processes each page image to generate multi-vector embeddings that capture both textual content and spatial relationships. These embeddings are stored in Qdrant along with metadata including session ID, document name, and page number for efficient retrieval.

[https://substackcdn.com/image/fetch/$s_!q5Nu!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4cf628c9-c8ad-46d4-90cd-a58a60f12cb3_3016x712.png](https://substackcdn.com/image/fetch/$s_!q5Nu!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4cf628c9-c8ad-46d4-90cd-a58a60f12cb3_3016x712.png)

**Inference Pipeline**: When processing user queries, the system converts the query text into an embedding using the same ColQwen 2.5 model to ensure compatibility. Qdrant performs a similarity search to find the most relevant document pages, then the storage service retrieves the actual page images for these matches. Finally, Claude Sonnet 3.7 receives both the original query and the retrieved images to generate a contextually accurate response.

> _This architecture is provider-agnostic—you could substitute Qdrant with Weaviate, Supabase with S3, or Claude with GPT-4o. The key principle is maintaining the visual pipeline throughout, ensuring no visual information is lost in translation._

**💻 Enough theory, let’s get into the implementation ↓**

* * *

## Service Providers

You're free to choose any providers for the five main components. In this tutorial, I use:

- **Document Processing**: `pdf2image` for PDF conversion

- **Object Storage**: Supabase for image storage

- **Visual Embeddings**: ColQwen 2.5 (v0.2) from Hugging Face

- **Vector Storage**: Qdrant Cloud

- **Response Generation**: Claude Sonnet 3.7 from Anthropic


_Note: Why Supabase for storage? While you could use S3, Google Cloud Storage, or even local storage, Supabase offers a convenient free storage tier._

#### Dependencies

Before we start building our ColPali RAG application, let's ensure we have all the necessary Python packages installed.

- `pdf2image`: Converts PDF documents into images

- `colpali-engine`: Provides the ColPali models and processors

- `qdrant-client`: Handles vector database operations and similarity search

- `supabase`: Manages object storage for document images

- `fastapi[standard]`: Powers our web API

- `instructor[anthropic]`: Enables structured output parsing from Claude

- `pydantic-settings`: Manages environment variables and configuration


> _Note: The_ `pdf2image` _package requires Poppler. Check the [official documentation](https://pdf2image.readthedocs.io/en/latest/installation.html#installing-poppler) and follow the appropriate instructions for your operating system._

## Environment Variables

To connect to these services, you'll need API keys and configuration variables.

**Set up Anthropic:** Create an an [Anthropic](https://docs.anthropic.com/en/docs/get-started) ccount and generate an API key to use Claude Sonnet 3.7.

**Set up Qdrant:** Create a [Qdrant Cloud](https://qdrant.to/cloud) account and set up a free cluster for the vector store. This will provide both an API key and host URL. You'll also need to choose a name for the collection that will be created in the next section.

[https://substackcdn.com/image/fetch/$s_!_lHW!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff7062633-fc54-4a18-8c70-ac092ee92d8d_1598x1546.png](https://substackcdn.com/image/fetch/$s_!_lHW!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff7062633-fc54-4a18-8c70-ac092ee92d8d_1598x1546.png)

**Set up Supabase:** Create a [Supabase](https://supabase.com/dashboard) account and project to get your project URL and API key. Then create a bucket in Supabase Storage and note the bucket name.

[https://substackcdn.com/image/fetch/$s_!AqR3!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F769b566f-877b-44d6-bacb-dd6e081978cd_1646x1086.png](https://substackcdn.com/image/fetch/$s_!AqR3!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F769b566f-877b-44d6-bacb-dd6e081978cd_1646x1086.png)

Once you have all the keys, create a `.env` file that looks like this:

```
# Vector Database
QDRANT_URL=<your-qdrant-url>
QDRANT_API_KEY=<your-qdrant-api-key>
COLLECTION_NAME=<your-collection-name>

# Storage
SUPABASE_URL=<your-supabase-url>
SUPABASE_KEY=<your-supabase-key>

# AI Model
ANTHROPIC_API_KEY=<your-anthropic-api-key>
```

💡 _**Note:** Always add_ `.env` _to your_ `.gitignore` _to protect your credentials._ Now we'll use `pydantic-settings` to manage these securely:

```
import os
from functools import lru_cache

from pydantic_settings import BaseSettings

class QdrantSettings(BaseSettings):
    collection_name: str = os.environ.get("QDRANT_COLLECTION_NAME", "")
    qdrant_url: str = os.environ.get("QDRANT_URL", "")
    qdrant_api_key: str = os.environ.get("QDRANT_API_KEY", "")

class ColpaliSettings(BaseSettings):
    colpali_model_name: str = "vidore/colqwen2.5-v0.2"

class SupabaseSettings(BaseSettings):
    supabase_key: str = os.environ.get("SUPABASE_KEY", "")
    supabase_url: str = os.environ.get("SUPABASE_URL", "")
    bucket: str = "colpali"

class AnthropicSettings(BaseSettings):
    api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")

class Settings(BaseSettings):
    qdrant: QdrantSettings = QdrantSettings()
    colpali: ColpaliSettings = ColpaliSettings()
    supabase: SupabaseSettings = SupabaseSettings()
    anthropic: AnthropicSettings = AnthropicSettings()

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

## Set up Qdrant Collection

A collection in Qdrant serves as a container for vectors with the same dimensionality. In our ColPali system, it stores the multi-vector embeddings representing each document image.

You can create a collection through Qdrant Cloud's UI or programmatically. Either way, the configuration follows [ColPali's](https://arxiv.org/pdf/2407.01449) specifications:

- **Embedding dimension**: 128 (as specified in the ColPali paper)

- **Distance metric**: Cosine similarity for comparing embeddings

- **Multi-vector comparison**: Max similarity finds the best matching spots on each page for your query


Here's a simplified version of the programmatic approach:

```
# Full version available at https://github.com/jjovalle99/colpali-rag-app/blob/main/scripts/create_collection.py

qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=128,
        distance=models.Distance.COSINE,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
    )
)
```

## Loading the ColPali model

If you've worked with Hugging Face models before, this step will feel familiar. The ColPali team provides a Python library that makes it straightforward to interact with their models. You can find their repository at [https://github.com/illuin-tech/colpali](https://github.com/illuin-tech/colpali).

When loading ColPali models, we need to consider a few key aspects:

- **Device Selection**: The model can run on CUDA (NVIDIA GPUs), MPS (Apple Silicon), or CPU

- **Precision**: Using bfloat16 or float16 reduces memory usage while maintaining quality

- **Attention Implementation**: Flash Attention 2 significantly speeds up inference when available


Here's how to load the model and processor:

```
import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2, ColQwen2Processor

model_name = "vidore/colqwen2-v1.0"

model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # or "mps" if on Apple Silicon
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()

processor = ColQwen2Processor.from_pretrained(model_name)

# Your inputs
images = [\
    Image.new("RGB", (128, 128), color="white"),\
    Image.new("RGB", (64, 32), color="black"),\
]
queries = [\
    "What is the organizational structure for our R&D department?",\
    "Can you provide a breakdown of last year’s financial performance?",\
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)
```

To make our code more maintainable and avoid repeated initialization, let's encapsulate the model loading logic into a dedicated class. This approach allows us to:

- Automatically detect the best available hardware (GPU, Apple Silicon, or CPU)

- Select the optimal precision for your system

- Handle the attention implementation gracefully


```
class ColQwen2_5Loader:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self._dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported()
            else torch.float16
        )

    def load(self):
        model = ColQwen2_5.from_pretrained(
            self.model_name,
            device_map=self._device,
            torch_dtype=self._dtype,
            attn_implementation="flash_attention_2" if available else None
        ).eval()

        processor = ColQwen2_5_Processor.from_pretrained(self.model_name)

        return model, processor
```

> _Note: The_ `.eval()` _call is crucial here as it ensures the model runs in evaluation mode, disabling dropout and other operations that would otherwise affect our embeddings' consistency._

This loader class provides a clean interface for initializing ColPali models throughout our application. We'll use this in the next sections when building our ingestion pipeline and query functionality.

## Building the Document Ingestion Pipeline

Now let's build the document ingestion pipeline - the component that transforms PDFs into searchable visual embeddings.

If you look at traditional RAG pipelines, they typically extract text from PDFs, chunk it, and generate embeddings. But this approach loses crucial information - table structures, figure placements, and visual relationships that give documents their meaning. ColPali takes a fundamentally different approach.

Instead of text extraction, we'll convert each PDF page into a high-resolution image and process it visually. This preserves everything - from complex tables to annotated diagrams.

```
class PDFIngestController:
    def __init__(self, model, processor, uploader, qdrant_client, collection_name):
        self.model = model
        self.processor = processor
        self.uploader = uploader
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name

    async def ingest(self, files: list[UploadFile], session_id: UUID4):
        results = []

        for file in files:
            # Convert PDF to high-resolution images
            pdf_bytes = await file.read()
            images = await run_in_threadpool(
                convert_from_bytes,
                pdf_file=pdf_bytes,
                dpi=300,  # High DPI for clear text
                thread_count=4,
                fmt="jpeg"
            )

            # Process images with ColPali
            for page_num, image in enumerate(images, 1):
                with torch.inference_mode():
                    processed = self.processor.process_images([image])
                    embedding = self.model(**processed.to(self.model.device))

                # Store in vector database
                point = models.PointStruct(
                    id=str(uuid4()),
                    vector=embedding.cpu().float().numpy().tolist(),
                    payload={
                        "session_id": session_id,
                        "document": file.filename,
                        "page": page_num
                    }
                )
                await self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[point]
                )

                # Upload image to storage
                await self.uploader.upload_image(
                    session_id=session_id,
                    file_name=file.filename,
                    page=page_num,
                    image=image
                )
```

In this code snippet, we:

1. Receive the user file

2. Split each file into individual pages

3. Convert the pages into images

4. Persist the images

5. Generate the multi-vector embeddings using ColPali model

6. Store the vectors in the collection


## Building the Inference Pipeline

Traditional RAG systems search through text chunks and concatenate them for the LLM. But since we're working with visual documents, our pipeline needs to handle images throughout the entire process. This means searching with visual embeddings, retrieving actual document images, and feeding them directly to a multimodal LLM.

```
class QueryController:
    def __init__(self, model, processor, downloader, instructor_client,
                 qdrant_client, collection_name, prompts):
        self.model = model
        self.processor = processor
        self.downloader = downloader
        self.instructor_client = instructor_client
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.prompts = prompts

    async def query(self, query: str, top_k: int, session_id: UUID4):
        # Generate query embedding
        with torch.inference_mode():
            processed_query = self.processor.process_queries([query])
            query_embedding = self.model(**processed_query.to(self.model.device))

        # Search for similar document pages
        search_results = await self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding[0].cpu().float().tolist(),
            limit=top_k,
            query_filter=models.Filter(
                must=[models.FieldCondition(\
                    key="session_id",\
                    match=models.MatchValue(value=str(session_id))\
                )]
            )
        )

        # Download retrieved images
        filenames = [\
            f"{point['session_id']}/{point['document']}/{point['page']}.jpeg"\
            for point in search_results.points\
        ]
        images = await self.downloader.download_instructor_images(filenames)

        # Generate response with multimodal LLM
        prompt_content = [self.prompts["prompt1"]]
        for filename, image in zip(filenames, images):
            prompt_content.extend([\
                f'<image file="{filename}">',\
                image,\
                '</image>'\
            ])
        prompt_content.append(self.prompts["prompt2"])

        # Stream the response
        stream = self.instructor_client.completions.create_partial(
            model="claude-3-7-sonnet-latest",
            messages=[{"role": "user", "content": prompt_content}],
            context={"query": query},
            temperature=0.0,
            max_tokens=8192
        )

        async for partial in stream:
            yield partial.model_dump_json() + "\n"
```

Let's break down what happens in this pipeline:

1. **Query Embedding Generation**: We convert the user's question into a embedding using the same ColPali model. This ensures we're searching in the same embedding space as our document images.

2. **Vector Similarity Search**: Using Qdrant's query capabilities, we find the most visually similar document pages. The multi-vector comparison (remember the MAX\_SIM setting?) ensures we're matching against the best regions within each page.

3. **Session Filtering**: We filter results to only include documents from the current session, maintaining data isolation between users.

4. **Image Retrieval**: Instead of retrieving text chunks, we download the actual document images from our storage service. This preserves the full visual context.

5. **Multimodal Response Generation**: We construct a prompt that includes both the query and the retrieved images, then stream the response from Claude. The LLM can now reference specific tables, interpret figures, and understand the document's visual layout.


> _Note: The streaming approach using_ `create_partial` _from the_ `Instructor` _library allows for real-time response generation while maintaining structured answer and reference formatting._

This pipeline completes our visual RAG system. Unlike traditional approaches that would struggle with "find the revenue table on page 3," our system understands both the semantic meaning and visual structure of documents, delivering accurate, contextual responses.

## Bringing It All Together: Building the API

Now that we have our components ready, let's bring everything together into a FastAPI application that can handle document ingestion and queries.

With our ColPali model, storage service, and vector database configured, we need to wire these components into a cohesive API. We'll use FastAPI's dependency injection system to manage our resources efficiently and ensure they're properly initialized and cleaned up.

### Application Lifespan

```
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Initialize clients
    qdrant_client = create_qdrant_client(settings)
    anthropic_client = create_anthropic_client(settings)
    instructor_client = instructor.from_anthropic(anthropic_client)
    supabase_client = create_supabase_client(settings)

    # Initialize services
    supabase_uploader = SupabaseJPEGUploader(
        client=supabase_client,
        bucket_name=settings.supabase.bucket
    )
    supabase_downloader = SupabaseJPEGDownloader(
        client=supabase_client,
        bucket_name=settings.supabase.bucket
    )

    # Load ColPali model
    loader = ColQwen2_5Loader(model_name="vidore/colqwen2.5-v0.2")
    model, processor = loader.load()

    yield {
        "model": model,
        "processor": processor,
        "supabase_uploader": supabase_uploader,
        "supabase_downloader": supabase_downloader,
        "instructor_client": instructor_client,
        "qdrant_client": qdrant_client,
        "collection_name": settings.qdrant.collection_name
    }

    # Cleanup
    await qdrant_client.close()
    await anthropic_client.close()
```

In this lifespan handler, we:

1. **Initialize all clients**: Set up connections to Qdrant, Anthropic, and Supabase

2. **Load the ColPali model**: This is the most resource-intensive step, loading the model into GPU memory

3. **Yield the resources**: Make them available to our application through FastAPI's state

4. **Clean up on shutdown**: Properly close all connections to prevent resource leaks


💡 _**Note:** Loading the ColPali model can take 10-30 seconds depending on your hardware. The lifespan approach ensures this happens only once when your server starts._

### Setting Up the API Routes

With our resources initialized, we need to create endpoints for document ingestion and querying. FastAPI's dependency injection makes it elegant to access our shared resources in each endpoint.

**Document Ingestion Endpoint** ( `/ingest-pdfs/`):

- Accepts multiple PDF files and a session ID

- Converts each PDF page to an image

- Generates ColPali embeddings for each page

- Stores the embeddings in Qdrant with metadata

- Uploads the images to Supabase for later retrieval


```
@router.post("/ingest-pdfs/")
async def ingest_pdf(
    files: list[UploadFile],
    session_id: UUID4,
    model: Annotated[ColQwen2_5, Depends(get_colpali_model)],
    processor: Annotated[ColQwen2_5_Processor, Depends(get_colpali_processor)],
    uploader: Annotated[SupabaseJPEGUploader, Depends(get_supabase_uploader)],
    qdrant_client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    collection_name: Annotated[str, Depends(get_collection_name)],
):
    controller = PDFIngestController(
        model=model,
        processor=processor,
        uploader=uploader,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
    )
    return await controller.ingest(files=files, session_id=session_id)
```

Let's examine what each endpoint does:

**Query Endpoint** ( `/query/`):

- Takes a natural language query and session ID

- Converts the query to a ColPali embedding

- Searches for the most relevant document pages

- Retrieves the actual page images

- Streams the multimodal LLM's response


```
@router.post("/query/")
async def query_endpoint(
    query: str,
    top_k: int,
    session_id: UUID4,
    model: Annotated[ColQwen2_5, Depends(get_colpali_model)],
    processor: Annotated[ColQwen2_5_Processor, Depends(get_colpali_processor)],
    downloader: Annotated[\
        SupabaseJPEGDownloader, Depends(get_supabase_downloader)\
    ],
    instructor_client: Annotated[\
        AsyncInstructor, Depends(get_instructor_client)\
    ],
    qdrant_client: Annotated[AsyncQdrantClient, Depends(get_qdrant_client)],
    collection_name: Annotated[str, Depends(get_collection_name)],
    prompts: Annotated[dict[str, str], Depends(get_prompts)],
):
    controller = QueryController(
        model=model,
        processor=processor,
        downloader=downloader,
        instructor_client=instructor_client,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        prompts=prompts,
    )
    return StreamingResponse(
        controller.query(query, top_k, session_id),
        media_type="text/event-stream",
    )
```

The use of `StreamingResponse` in the query endpoint is crucial - it allows users to see results as they're generated rather than waiting for the complete response.

### Configuring the FastAPI Application

Finally, we need to configure our FastAPI application with the necessary middleware and include our routers:

```
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(pdf_ingest.router)
app.include_router(query.router)
```

### Testing the application

With everything set up, let's launch our ColPali RAG system:

```
uvicorn server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload
```

Once your server is running, you can test the system through these steps:

1. **Upload documents**: Send PDFs to `/ingest-pdfs/` with a session ID

2. **Query the system**: Post queries to `/query/` with the same session ID

3. **Stream responses**: The system returns a stream of partial responses


Programmatically, it would look something like this:

```
import json
from pathlib import Path
from uuid import uuid4

import requests

# Config
session_id = str(uuid4())
port = 8000
host = "localhost"

# PDF Upload
url = f"http://{host}:{port}/ingest-pdfs/?session_id={session_id}"
pdf_path = Path("colpali.pdf")
headers = {
    "accept": "application/json",
}
with pdf_path.open("rb") as f:
    files = {"files": (pdf_path.name, f, "application/pdf")}
    response = requests.post(url, headers=headers, files=files)

# Query
query = "What are the advantages of Colpali? and what disadvantages does it have?"
top_k = 5
url = f"http://{host}:{port}/query/?query={query}&top_k={top_k}&session_id={session_id}"

with requests.post(url, stream=True) as response:
    for chunk in response.iter_content(chunk_size=4096, decode_unicode=True):
        parsed_json = json.loads(chunk)
        formatted_json = json.dumps(parsed_json, indent=4)
        print(formatted_json)

```

* * *

## Wrap-Up

Congratulations! You now have a fully functional ColPali RAG system that can understand documents the way humans do - visually.

Throughout this tutorial, we've built a system that preserves the rich visual information in documents rather than discarding it. From converting PDFs to images, generating visual embeddings with ColPali, to having a multimodal LLM interpret the retrieved pages - every step maintains the visual pipeline.

To keep this tutorial focused and readable, I've shown you the core components and their interactions. However, a production system needs additional pieces like error handling, input validation, and proper logging.

> _If you're interested in the **complete implementation**, including all supporting code and best practices, I encourage you to check out the **[GitHub repository](https://github.com/jjovalle99/colpali-rag-app)**._

</details>

<details>
<summary>CLIP</summary>

# CLIP

[\[Blog\]](https://openai.com/blog/clip/) [\[Paper\]](https://arxiv.org/abs/2103.00020) [\[Model Card\]](https://github.com/openai/CLIP/blob/main/model-card.md) [\[Colab\]](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb)

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.

## Approach

[https://github.com/openai/CLIP/raw/main/CLIP.png](https://github.com/openai/CLIP/blob/main/CLIP.png)

## Usage

First, [install PyTorch 1.7.1](https://pytorch.org/get-started/locally/) (or later) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:

```
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

Replace `cudatoolkit=11.0` above with the appropriate CUDA version on your machine or `cpuonly` when installing on a machine without a GPU.

```
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
```

## API

The CLIP module `clip` provides the following methods:

#### `clip.available_models()`

Returns the names of the available CLIP models.

#### `clip.load(name, device=..., jit=False)`

Returns the model and the TorchVision transform needed by the model, specified by the model name returned by `clip.available_models()`. It will download the model as necessary. The `name` argument can also be a path to a local checkpoint.

The device to run the model can be optionally specified, and the default is to use the first CUDA device if there is any, otherwise the CPU. When `jit` is `False`, a non-JIT version of the model will be loaded.

#### `clip.tokenize(text: Union[str, List[str]], context_length=77)`

Returns a LongTensor containing tokenized sequences of given text input(s). This can be used as the input to the model

* * *

The model returned by `clip.load()` supports the following methods:

#### `model.encode_image(image: Tensor)`

Given a batch of images, returns the image features encoded by the vision portion of the CLIP model.

#### `model.encode_text(text: Tensor)`

Given a batch of text tokens, returns the text features encoded by the language portion of the CLIP model.

#### `model(image: Tensor, text: Tensor)`

Given a batch of images and a batch of text tokens, returns two Tensors, containing the logit scores corresponding to each image and text input. The values are cosine similarities between the corresponding image and text features, times 100.

## More Examples

### Zero-Shot Prediction

The code below performs zero-shot prediction using CLIP, as shown in Appendix B in the paper. This example takes an image from the [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), and predicts the most likely labels among the 100 textual labels from the dataset.

```
import os
import clip
import torch
from torchvision.datasets import CIFAR100

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
image, class_id = cifar100[3637]
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
```

The output will look like the following (the exact numbers may be slightly different depending on the compute device):

```
Top predictions:

           snake: 65.31%
          turtle: 12.29%
    sweet_pepper: 3.83%
          lizard: 1.88%
       crocodile: 1.75%

```

Note that this example uses the `encode_image()` and `encode_text()` methods that return the encoded features of given inputs.

### Linear-probe evaluation

The example below uses [scikit-learn](https://scikit-learn.org/) to perform logistic regression on image features.

```
import os
import clip
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Load the dataset
root = os.path.expanduser("~/.cache")
train = CIFAR100(root, download=True, train=True, transform=preprocess)
test = CIFAR100(root, download=True, train=False, transform=preprocess)

def get_features(dataset):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Calculate the image features
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")
```

Note that the `C` value should be determined via a hyperparameter sweep using a validation split.

## See Also

- [OpenCLIP](https://github.com/mlfoundations/open_clip): includes larger and independently trained CLIP models up to ViT-G/14
- [Hugging Face implementation of CLIP](https://huggingface.co/docs/transformers/model_doc/clip): for easier integration with the HF ecosystem

</details>

<details>
<summary>An [agent](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#agent-architectures) is _a system that uses an LLM to decide the control flow of an application_. As you develop these systems, they might grow more complex over time, making them harder to manage and scale. For example, you might run into the following problems:</summary>

An [agent](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#agent-architectures) is _a system that uses an LLM to decide the control flow of an application_. As you develop these systems, they might grow more complex over time, making them harder to manage and scale. For example, you might run into the following problems:

- agent has too many tools at its disposal and makes poor decisions about which tool to call next
- context grows too complex for a single agent to keep track of
- there is a need for multiple specialization areas in the system (e.g. planner, researcher, math expert, etc.)

To tackle these, you might consider breaking your application into multiple smaller, independent agents and composing them into a **multi-agent system**. These independent agents can be as simple as a prompt and an LLM call, or as complex as a [ReAct](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#tool-calling-agent) agent (and more!).

The primary benefits of using multi-agent systems are:

- **Modularity**: Separate agents make it easier to develop, test, and maintain agentic systems.
- **Specialization**: You can create expert agents focused on specific domains, which helps with the overall system performance.
- **Control**: You can explicitly control how agents communicate (as opposed to relying on function calling).

## Multi-agent architectureshttps://langchain-ai.github.io/langgraph/concepts/img/multi_agent/architectures.png

There are several ways to connect agents in a multi-agent system:

- **Network**: each agent can communicate with [every other agent](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/). Any agent can decide which other agent to call next.
- **Supervisor**: each agent communicates with a single [supervisor](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/) agent. Supervisor agent makes decisions on which agent should be called next.
- **Supervisor (tool-calling)**: this is a special case of supervisor architecture. Individual agents can be represented as tools. In this case, a supervisor agent uses a tool-calling LLM to decide which of the agent tools to call, as well as the arguments to pass to those agents.
- **Hierarchical**: you can define a multi-agent system with [a supervisor of supervisors](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/). This is a generalization of the supervisor architecture and allows for more complex control flows.
- **Custom multi-agent workflow**: each agent communicates with only a subset of agents. Parts of the flow are deterministic, and only some agents can decide which other agents to call next.

### Handoffs

In multi-agent architectures, agents can be represented as graph nodes. Each agent node executes its step(s) and decides whether to finish execution or route to another agent, including potentially routing to itself (e.g., running in a loop). A common pattern in multi-agent interactions is **handoffs**, where one agent _hands off_ control to another. Handoffs allow you to specify:

- **destination**: target agent to navigate to (e.g., name of the node to go to)
- **payload**: [information to pass to that agent](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#communication-and-state-management) (e.g., state update)

To implement handoffs in LangGraph, agent nodes can return [`Command`](https://langchain-ai.github.io/langgraph/concepts/low_level/#command) object that allows you to combine both control flow and state updates:

```python
def agent(state) -> Command[Literal["agent", "another_agent"]]:
    # the condition for routing/halting can be anything, e.g. LLM tool call / structured output, etc.
    goto = get_next_agent(...)  # 'agent' / 'another_agent'
    return Command(
        # Specify which agent to call next
        goto=goto,
        # Update the graph state
        update={"my_state_key": "my_state_value"}
    )

```

In a more complex scenario where each agent node is itself a graph (i.e., a [subgraph](https://langchain-ai.github.io/langgraph/concepts/subgraphs/)), a node in one of the agent subgraphs might want to navigate to a different agent. For example, if you have two agents, `alice` and `bob` (subgraph nodes in a parent graph), and `alice` needs to navigate to `bob`, you can set `graph=Command.PARENT` in the `Command` object:

```python
def some_node_inside_alice(state):
    return Command(
        goto="bob",
        update={"my_state_key": "my_state_value"},
        # specify which graph to navigate to (defaults to the current graph)
        graph=Command.PARENT,
    )

```

Note

If you need to support visualization for subgraphs communicating using `Command(graph=Command.PARENT)` you would need to wrap them in a node function with `Command` annotation:
Instead of this:

```python
builder.add_node(alice)

```

you would need to do this:

```python
def call_alice(state) -> Command[Literal["bob"]]:
    return alice.invoke(state)

builder.add_node("alice", call_alice)

```

#### Handoffs as tools

One of the most common agent types is a [tool-calling agent](https://langchain-ai.github.io/langgraph/agents/overview/). For those types of agents, a common pattern is wrapping a handoff in a tool call:

```python
from langchain_core.tools import tool

@tool
def transfer_to_bob():
    """Transfer to bob."""
    return Command(
        # name of the agent (node) to go to
        goto="bob",
        # data to send to the agent
        update={"my_state_key": "my_state_value"},
        # indicate to LangGraph that we need to navigate to
        # agent node in a parent graph
        graph=Command.PARENT,
    )

```

This is a special case of updating the graph state from tools where, in addition to the state update, the control flow is included as well.

Important

If you want to use tools that return `Command`, you can use the prebuilt [`create_react_agent`](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent) / [`ToolNode`](https://langchain-ai.github.io/langgraph/reference/agents/#langgraph.prebuilt.tool_node.ToolNode) components, or else implement your own logic:

```python
def call_tools(state):
    ...
    commands = [tools_by_name[tool_call["name"]].invoke(tool_call) for tool_call in tool_calls]
    return commands

```

Let's now take a closer look at the different multi-agent architectures.

### Network

In this architecture, agents are defined as graph nodes. Each agent can communicate with every other agent (many-to-many connections) and can decide which agent to call next. This architecture is good for problems that do not have a clear hierarchy of agents or a specific sequence in which agents should be called.

```python
from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langgraph.graph import StateGraph, MessagesState, START, END

model = ChatOpenAI()

def agent_1(state: MessagesState) -> Command[Literal["agent_2", "agent_3", END]]:
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # to determine which agent to call next. a common pattern is to call the model
    # with a structured output (e.g. force it to return an output with a "next_agent" field)
    response = model.invoke(...)
    # route to one of the agents or exit based on the LLM's decision
    # if the LLM returns "__end__", the graph will finish execution
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )

def agent_2(state: MessagesState) -> Command[Literal["agent_1", "agent_3", END]]:
    response = model.invoke(...)
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )

def agent_3(state: MessagesState) -> Command[Literal["agent_1", "agent_2", END]]:
    ...
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )

builder = StateGraph(MessagesState)
builder.add_node(agent_1)
builder.add_node(agent_2)
builder.add_node(agent_3)

builder.add_edge(START, "agent_1")
network = builder.compile()

```

### Supervisor

In this architecture, we define agents as nodes and add a supervisor node (LLM) that decides which agent nodes should be called next. We use [`Command`](https://langchain-ai.github.io/langgraph/concepts/low_level/#command) to route execution to the appropriate agent node based on supervisor's decision. This architecture also lends itself well to running multiple agents in parallel or using [map-reduce](https://langchain-ai.github.io/langgraph/how-tos/graph-api/#map-reduce-and-the-send-api) pattern.

```python
from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langgraph.graph import StateGraph, MessagesState, START, END

model = ChatOpenAI()

def supervisor(state: MessagesState) -> Command[Literal["agent_1", "agent_2", END]]:
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # to determine which agent to call next. a common pattern is to call the model
    # with a structured output (e.g. force it to return an output with a "next_agent" field)
    response = model.invoke(...)
    # route to one of the agents or exit based on the supervisor's decision
    # if the supervisor returns "__end__", the graph will finish execution
    return Command(goto=response["next_agent"])

def agent_1(state: MessagesState) -> Command[Literal["supervisor"]]:
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # and add any additional logic (different models, custom prompts, structured output, etc.)
    response = model.invoke(...)
    return Command(
        goto="supervisor",
        update={"messages": [response]},
    )

def agent_2(state: MessagesState) -> Command[Literal["supervisor"]]:
    response = model.invoke(...)
    return Command(
        goto="supervisor",
        update={"messages": [response]},
    )

builder = StateGraph(MessagesState)
builder.add_node(supervisor)
builder.add_node(agent_1)
builder.add_node(agent_2)

builder.add_edge(START, "supervisor")

supervisor = builder.compile()

```

### Supervisor (tool-calling)

In this variant of the [supervisor](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#supervisor) architecture, we define a supervisor [agent](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#agent-architectures) which is responsible for calling sub-agents. The sub-agents are exposed to the supervisor as tools, and the supervisor agent decides which tool to call next. The supervisor agent follows a [standard implementation](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#tool-calling-agent) as an LLM running in a while loop calling tools until it decides to stop.

```python
from typing import Annotated
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState, create_react_agent

model = ChatOpenAI()

# this is the agent function that will be called as tool
# notice that you can pass the state to the tool via InjectedState annotation
def agent_1(state: Annotated[dict, InjectedState]):
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # and add any additional logic (different models, custom prompts, structured output, etc.)
    response = model.invoke(...)
    # return the LLM response as a string (expected tool response format)
    # this will be automatically turned to ToolMessage
    # by the prebuilt create_react_agent (supervisor)
    return response.content

def agent_2(state: Annotated[dict, InjectedState]):
    response = model.invoke(...)
    return response.content

tools = [agent_1, agent_2]
# the simplest way to build a supervisor w/ tool-calling is to use prebuilt ReAct agent graph
# that consists of a tool-calling LLM node (i.e. supervisor) and a tool-executing node
supervisor = create_react_agent(model, tools)

```

### Hierarchical

As you add more agents to your system, it might become too hard for the supervisor to manage all of them. The supervisor might start making poor decisions about which agent to call next, or the context might become too complex for a single supervisor to keep track of. In other words, you end up with the same problems that motivated the multi-agent architecture in the first place.

To address this, you can design your system _hierarchically_. For example, you can create separate, specialized teams of agents managed by individual supervisors, and a top-level supervisor to manage the teams.

```python
from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
model = ChatOpenAI()

# define team 1 (same as the single supervisor example above)

def team_1_supervisor(state: MessagesState) -> Command[Literal["team_1_agent_1", "team_1_agent_2", END]]:
    response = model.invoke(...)
    return Command(goto=response["next_agent"])

def team_1_agent_1(state: MessagesState) -> Command[Literal["team_1_supervisor"]]:
    response = model.invoke(...)
    return Command(goto="team_1_supervisor", update={"messages": [response]})

def team_1_agent_2(state: MessagesState) -> Command[Literal["team_1_supervisor"]]:
    response = model.invoke(...)
    return Command(goto="team_1_supervisor", update={"messages": [response]})

team_1_builder = StateGraph(Team1State)
team_1_builder.add_node(team_1_supervisor)
team_1_builder.add_node(team_1_agent_1)
team_1_builder.add_node(team_1_agent_2)
team_1_builder.add_edge(START, "team_1_supervisor")
team_1_graph = team_1_builder.compile()

# define team 2 (same as the single supervisor example above)
class Team2State(MessagesState):
    next: Literal["team_2_agent_1", "team_2_agent_2", "__end__"]

def team_2_supervisor(state: Team2State):
    ...

def team_2_agent_1(state: Team2State):
    ...

def team_2_agent_2(state: Team2State):
    ...

team_2_builder = StateGraph(Team2State)
...
team_2_graph = team_2_builder.compile()

# define top-level supervisor

builder = StateGraph(MessagesState)
def top_level_supervisor(state: MessagesState) -> Command[Literal["team_1_graph", "team_2_graph", END]]:
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # to determine which team to call next. a common pattern is to call the model
    # with a structured output (e.g. force it to return an output with a "next_team" field)
    response = model.invoke(...)
    # route to one of the teams or exit based on the supervisor's decision
    # if the supervisor returns "__end__", the graph will finish execution
    return Command(goto=response["next_team"])

builder = StateGraph(MessagesState)
builder.add_node(top_level_supervisor)
builder.add_node("team_1_graph", team_1_graph)
builder.add_node("team_2_graph", team_2_graph)
builder.add_edge(START, "top_level_supervisor")
builder.add_edge("team_1_graph", "top_level_supervisor")
builder.add_edge("team_2_graph", "top_level_supervisor")
graph = builder.compile()

```

### Custom multi-agent workflow

In this architecture we add individual agents as graph nodes and define the order in which agents are called ahead of time, in a custom workflow. In LangGraph the workflow can be defined in two ways:

- **Explicit control flow (normal edges)**: LangGraph allows you to explicitly define the control flow of your application (i.e. the sequence of how agents communicate) explicitly, via [normal graph edges](https://langchain-ai.github.io/langgraph/concepts/low_level/#normal-edges). This is the most deterministic variant of this architecture above — we always know which agent will be called next ahead of time.

- **Dynamic control flow (Command)**: in LangGraph you can allow LLMs to decide parts of your application control flow. This can be achieved by using [`Command`](https://langchain-ai.github.io/langgraph/concepts/low_level/#command). A special case of this is a [supervisor tool-calling](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#supervisor-tool-calling) architecture. In that case, the tool-calling LLM powering the supervisor agent will make decisions about the order in which the tools (agents) are being called.

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START

model = ChatOpenAI()

def agent_1(state: MessagesState):
    response = model.invoke(...)
    return {"messages": [response]}

def agent_2(state: MessagesState):
    response = model.invoke(...)
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node(agent_1)
builder.add_node(agent_2)
# define the flow explicitly
builder.add_edge(START, "agent_1")
builder.add_edge("agent_1", "agent_2")

```

## Communication and state management

The most important thing when building multi-agent systems is figuring out how the agents communicate.

A common, generic way for agents to communicate is via a list of messages. This opens up the following questions:

- Do agents communicate [**via handoffs or via tool calls**](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#handoffs-vs-tool-calls)?
- What messages are [**passed from one agent to the next**](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#message-passing-between-agents)?
- How are [**handoffs represented in the list of messages**](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#representing-handoffs-in-message-history)?
- How do you [**manage state for subagents**](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#state-management-for-subagents)?

Additionally, if you are dealing with more complex agents or wish to keep individual agent state separate from the multi-agent system state, you may need to use [**different state schemas**](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#using-different-state-schemas).

### Handoffs vs tool calls

What is the "payload" that is being passed around between agents? In most of the architectures discussed above, the agents communicate via [handoffs](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#handoffs) and pass the [graph state](https://langchain-ai.github.io/langgraph/concepts/low_level/#state) as part of the handoff payload. Specifically, agents pass around lists of messages as part of the graph state. In the case of the [supervisor with tool-calling](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#supervisor-tool-calling), the payloads are tool call arguments.https://langchain-ai.github.io/langgraph/concepts/img/multi_agent/request.png

### Message passing between agents

The most common way for agents to communicate is via a shared state channel, typically a list of messages. This assumes that there is always at least a single channel (key) in the state that is shared by the agents (e.g., `messages`). When communicating via a shared message list, there is an additional consideration: should the agents [share the full history](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#sharing-full-thought-process) of their thought process or only [the final result](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#sharing-only-final-results)?https://langchain-ai.github.io/langgraph/concepts/img/multi_agent/response.png

#### Sharing full thought process

Agents can **share the full history** of their thought process (i.e., "scratchpad") with all other agents. This "scratchpad" would typically look like a [list of messages](https://langchain-ai.github.io/langgraph/concepts/low_level/#why-use-messages). The benefit of sharing the full thought process is that it might help other agents make better decisions and improve reasoning ability for the system as a whole. The downside is that as the number of agents and their complexity grows, the "scratchpad" will grow quickly and might require additional strategies for [memory management](https://langchain-ai.github.io/langgraph/how-tos/memory/add-memory/).

#### Sharing only final results

Agents can have their own private "scratchpad" and only **share the final result** with the rest of the agents. This approach might work better for systems with many agents or agents that are more complex. In this case, you would need to define agents with [different state schemas](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#using-different-state-schemas).

For agents called as tools, the supervisor determines the inputs based on the tool schema. Additionally, LangGraph allows [passing state](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/#short-term-memory) to individual tools at runtime, so subordinate agents can access parent state, if needed.

#### Indicating agent name in messages

It can be helpful to indicate which agent a particular AI message is from, especially for long message histories. Some LLM providers (like OpenAI) support adding a `name` parameter to messages — you can use that to attach the agent name to the message. If that is not supported, you can consider manually injecting the agent name into the message content, e.g., `<agent>alice</agent><message>message from alice</message>`.

### Representing handoffs in message history

Handoffs are typically done via the LLM calling a dedicated [handoff tool](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#handoffs-as-tools). This is represented as an [AI message](https://python.langchain.com/docs/concepts/messages/#aimessage) with tool calls that is passed to the next agent (LLM). Most LLM providers don't support receiving AI messages with tool calls **without** corresponding tool messages.

You therefore have two options:

1. Add an extra [tool message](https://python.langchain.com/docs/concepts/messages/#toolmessage) to the message list, e.g., "Successfully transferred to agent X"
2. Remove the AI message with the tool calls

In practice, we see that most developers opt for option (1).

### State management for subagents

A common practice is to have multiple agents communicating on a shared message list, but only [adding their final messages to the list](https://langchain-ai.github.io/langgraph/concepts/multi_agent/#sharing-only-final-results). This means that any intermediate messages (e.g., tool calls) are not saved in this list.

What if you **do** want to save these messages so that if this particular subagent is invoked in the future you can pass those back in?

There are two high-level approaches to achieve that:

1. Store these messages in the shared message list, but filter the list before passing it to the subagent LLM. For example, you can choose to filter out all tool calls from **other** agents.
2. Store a separate message list for each agent (e.g., `alice_messages`) in the subagent's graph state. This would be their "view" of what the message history looks like.

### Using different state schemas

An agent might need to have a different state schema from the rest of the agents. For example, a search agent might only need to keep track of queries and retrieved documents. There are two ways to achieve this in LangGraph:

- Define [subgraph](https://langchain-ai.github.io/langgraph/concepts/subgraphs/) agents with a separate state schema. If there are no shared state keys (channels) between the subgraph and the parent graph, it's important to [add input / output transformations](https://langchain-ai.github.io/langgraph/concepts/how-tos/subgraph.ipynb#different-state-schemas) so that the parent graph knows how to communicate with the subgraphs.
- Define agent node functions with a [private input state schema](https://langchain-ai.github.io/langgraph/concepts/how-tos/graph-api.ipynb#pass-private-state-between-nodes) that is distinct from the overall graph state schema. This allows passing information that is only needed for executing that particular agent.

</details>

<details>
<summary>Retrieval-Augmented Generation (RAG) has become the default way to connect Large Language Models (LLMs) with enterprise data. However, there's a critical flaw in this approach that's rarely discussed: nearly all production RAG pipelines rely on Optical Character Recognition (OCR) to process PDFs, scans, presentations, and other documents, with the silent assumption that the extracted text is "good enough" for downstream AI tasks.</summary>

Retrieval-Augmented Generation (RAG) has become the default way to connect Large Language Models (LLMs) with enterprise data. However, there's a critical flaw in this approach that's rarely discussed: nearly all production RAG pipelines rely on Optical Character Recognition (OCR) to process PDFs, scans, presentations, and other documents, with the silent assumption that the extracted text is "good enough" for downstream AI tasks.

Our comprehensive analysis shows that this assumption is fundamentally flawed. OCR quality creates an invisible ceiling that limits the performance of even the most advanced RAG systems. The gap between what's possible with perfect text extraction and what's achieved with current OCR technology represents one of the most significant yet overlooked challenges in enterprise AI today.

_**TLDR:**_

- **OCR creates an invisible performance ceiling.** Text extraction errors significantly limit both retrieval accuracy and generation quality in RAG systems.
- **Benchmarks reveal a substantial gap.** Even leading OCR solutions fall **~4.5%** short (NDCG@5) of ground-truth text performance, particularly with complex document layouts.
- **Vision-only generation is not ready yet.** Despite rapid improvements, multimodal models still cannot reliably generate precise answers directly from multiple document images.
- **Multimodal retrieval beats perfect text.** Our vector stores outperform even _perfect text_ by **~12%** on retrieval accuracy (NDCG@5) and recover **70%** of generation quality lost to OCR errors, while simultaneously simplifying architecture and enhancing future compatibility.

## [Why OCR is still critical for AI systems](https://www.mixedbread.com/blog/the-hidden-ceiling\#why-ocr-is-still-critical-for-ai-systems)

Most enterprise knowledge is locked in unstructured formats like PDFs, scanned documents, invoices, presentations, images, and a plethora of other formats. Before a Large Language Model (LLM) can reason over this knowledge, it needs to be converted from its original visual or semi-structured format into plain text.

This text conversion step, typically handled by OCR engines, is crucial because it feeds two core components of a RAG system:

1. **The Retrieval System:** Most retrieval systems depend on extracted text as their main search input. When OCR quality is poor, it produces inaccurate or "corrupted" text representations of your documents. This results in flawed text representations, making it difficult or impossible for the retrieval system to locate the relevant documents when a user asks a question. If the text doesn't accurately reflect the content, the search fails before it even begins.
2. **The Generation Model (LLM):** LLMs generate answers based _only_ on the context they are given. If the retrieved document snippets contain OCR errors (missing words, jumbled tables, incorrect numbers), the LLM receives flawed information. This directly leads to incomplete, nonsensical, or factually incorrect answers, even if the retrieval system managed to find the _correct_ document pages.

In short, errors introduced by OCR don't just stay in the text; they cascade through the entire RAG pipeline, impacting both the ability to _find_ information and the ability to _generate accurate answers_ from it.

## [Putting OCR to the Test: Our Benchmark Setup](https://www.mixedbread.com/blog/the-hidden-ceiling\#putting-ocr-to-the-test-our-benchmark-setup)

To quantify this "OCR ceiling" and understand its real-world impact, we needed a robust way to measure performance across diverse and challenging documents. We conducted extensive testing using the **[OHR (OCR hinders RAG) Benchmark v2](https://arxiv.org/abs/2412.02592)**.

This benchmark is specifically designed to evaluate how OCR performance affects RAG tasks and includes:

- **Diverse & Challenging Documents:** **8,500+** PDF pages across seven enterprise domains (Textbooks, Law, Finance, Newspapers, Manuals, Academic Papers, Administrative) featuring complex layouts, tables, formulas, charts, diagrams, and non-standard reading orders that are known to challenge OCR systems.
- **Targeted Questions:** **8,498** question-answer pairs specifically designed to test retrieval and understanding of information related to these OCR challenges. Each answer is grounded in specific evidence pages within the documents.
- **Verified Ground Truth:** Human-verified, perfect text extraction and curated answers provide a reliable "gold standard" for comparison.

Against this benchmark, we evaluated a range of OCR and retrieval approaches:

- **[Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs/models):** A frontier closed-source multimodal model capable of OCR.
- **[MinerU](https://github.com/opendatalab/MinerU):** A popular open-source library implementing state-of-the-art OCR methods from academic literature.
- **[Azure Document Intelligence](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/overview?view=doc-intel-4.0.0):** A widely used commercial OCR solution in the industry.
- **[Qwen-2.5-VL](https://github.com/QwenLM/Qwen-VL):** A frontier open-source multimodal model capable of OCR.
- **[Unstructured](https://github.com/Unstructured-IO/unstructured):** A popular open-source library with broad adoption for document parsing.
- **[Mixedbread Vector Store](https://www.mixedbread.com/docs/vector-stores/overview):** Our core offering, using native multimodal retrieval (treating pages as _images_, not just text) powered by our internal multimodal models ( `mxbai-omni-v0.1`). It bypasses traditional reliance on OCR for retrieval.

This comprehensive setup allowed us to isolate the impact of different OCR qualities and compare text-based approaches directly against our multimodal retrieval system.

## [Testing Retrieval: Setup and Results](https://www.mixedbread.com/blog/the-hidden-ceiling\#testing-retrieval-setup-and-results)

First, we focused on retrieval - the task of finding the _right_ information within the vast document set. If your RAG system can't surface the correct documents, the LLM has no chance of answering the user's query accurately.

### [Retrieval Setup](https://www.mixedbread.com/blog/the-hidden-ceiling\#retrieval-setup)

We transformed the OHR benchmark's question-answer pairs into a retrieval task: the question became the query, and the associated evidence pages were the target documents to retrieve.

For the text-based OCR methods, we used **[BM25](https://en.wikipedia.org/wiki/Okapi_BM25)**, a standard and robust keyword-based ranking algorithm commonly used in search engines. (We tested embedding-based retrieval and rerankers too, but found they often degraded performance on this benchmark compared to the strong BM25 baseline, likely due to OCR noise corrupting the embeddings. You can find more details [here](https://docs.google.com/spreadsheets/d/1zBGOIOCzZZjw1HXBGGI8BzNx_kYj34LlYaFteZTU7Bg/edit?usp=sharing).)

For the Mixedbread Vector Store, we leveraged our multimodal embedding model ( `mxbai-omni-v0.1`), which directly processes **screenshots** of the document pages. This approach is inherently resilient to OCR errors because it "sees" the page layout, structure, and visual elements alongside the text.

We measured retrieval performance using two standard metrics:

- **NDCG@5 (Normalized Discounted Cumulative Gain @ 5):** This metric evaluates the quality of the top 5 retrieved documents. It cares not only _if_ the correct documents are found but also _how highly ranked_ they are. Relevant documents ranked higher get more points. We chose K=5 because research shows LLMs are heavily influenced by the order of documents in their context window, with earlier documents having more impact.
- **Recall@5:** This metric measures whether at least one of the _correct_ evidence pages was retrieved within the top 5 results. It tells us if the necessary information was surfaced at all, regardless of its exact ranking.

### [Retrieval Results: The OCR Ceiling is Real](https://www.mixedbread.com/blog/the-hidden-ceiling\#retrieval-results-the-ocr-ceiling-is-real)

Our retrieval benchmarks revealed stark differences between traditional OCR-dependent methods and our multimodal approach.

* * *

**NDCG@5 Performance**(Average across all 7 document domains)https://www.mixedbread.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fretrieval-ndcg.7637e275.webp&w=3840&q=75&dpl=dpl_6D5g8JYtmzhd4sCHskMoarzC6kVS

This chart shows NDCG@5 scores for each retrieval method, averaged across seven document domains. NDCG@5 measures both the presence and ranking of relevant documents in the top 5—higher values mean more accurate retrieval, with extra weight for top-ranked relevant pages.

| Domain | Gemini 2.5 Flash | MinerU | Mixedbread OCR | Qwen-2.5-VL | Azure | Unstructured | **Mixedbread Vector Store** | Ground Truth OCR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| academic | 0.805 | 0.786 | 0.795 | 0.822 | 0.797 | 0.693 | **0.923** | 0.845 |
| administration | 0.861 | 0.776 | 0.842 | 0.853 | 0.854 | 0.672 | **0.920** | 0.895 |
| finance | 0.656 | 0.576 | 0.636 | 0.666 | 0.664 | 0.517 | **0.773** | 0.722 |
| law | 0.876 | 0.829 | 0.871 | 0.873 | 0.889 | 0.724 | **0.913** | 0.897 |
| manual | 0.800 | 0.756 | 0.820 | 0.834 | 0.828 | 0.721 | **0.923** | 0.861 |
| news | 0.442 | 0.438 | 0.454 | 0.415 | 0.460 | 0.111 | **0.686** | 0.467 |
| textbook | 0.624 | 0.572 | 0.673 | 0.698 | 0.671 | 0.159 | **0.915** | 0.720 |
| **avg** | **0.723** | **0.676** | **0.727** | **0.737** | **0.738** | **0.514** | **0.865** | **0.773** |

* * *

**Recall@5 Performance** _(Average across all 7 document domains)_https://www.mixedbread.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fretrieval-recall.351db88a.webp&w=3840&q=75&dpl=dpl_6D5g8JYtmzhd4sCHskMoarzC6kVS

This chart shows Recall@5 for each method, averaged across domains. Recall@5 is the percentage of questions where at least one correct evidence page appeared in the top 5—higher is better.

| Domain | Gemini 2.5 Flash | MinerU | Mixedbread OCR | Qwen-2.5-VL | Azure | Unstructured | **Mixedbread Vector Store** | Ground Truth OCR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| academic | 0.902 | 0.885 | 0.896 | 0.911 | 0.902 | 0.789 | **0.982** | 0.937 |
| administration | 0.930 | 0.857 | 0.920 | 0.930 | 0.931 | 0.735 | **0.967** | 0.959 |
| finance | 0.778 | 0.677 | 0.760 | 0.781 | 0.783 | 0.625 | **0.883** | 0.836 |
| law | 0.933 | 0.890 | 0.929 | 0.932 | 0.948 | 0.775 | **0.968** | 0.951 |
| manual | 0.874 | 0.844 | 0.904 | 0.912 | 0.915 | 0.802 | **0.971** | 0.932 |
| news | 0.479 | 0.468 | 0.489 | 0.458 | 0.493 | 0.115 | **0.767** | 0.499 |
| textbook | 0.644 | 0.600 | 0.700 | 0.728 | 0.702 | 0.168 | **0.936** | 0.746 |
| **avg** | **0.791** | **0.746** | **0.800** | **0.807** | **0.811** | **0.573** | **0.925** | **0.837** |

* * *

These results reveal several critical insights:

1. **OCR Creates a Performance Ceiling:** Every single OCR solution tested underperformed compared to the Ground Truth benchmark using perfect text. The best OCR methods plateaued around 0.74 average NDCG@5, a **~4.5%** absolute gap below the Ground Truth's 0.773. This confirms that OCR errors inherently limit retrieval effectiveness.
2. **Complexity Magnifies OCR Issues:** The performance gap widens for documents with complex layouts (finance, textbooks, news). These domains often feature tables, formulas, multi-column text, etc., that challenge OCR.
3. **Multimodal Excels by Seeing the Whole Picture:** Mixedbread Vector Store consistently outperformed _all_ other methods, including the perfect text Ground Truth benchmark. Its average NDCG@5 of **0.865** is nearly **12% higher** than Ground Truth text because it understands the _visual context_ (layout, tables, charts) directly from the image, providing richer relevance cues.

The Recall@5 increases from 0.84 using Ground Truth text to 0.92 using the Mixedbread Vector Store. Let's put this in perspective:

- With Ground Truth (perfect OCR): Recall@5 = 84% → 84 out of every 100 truly relevant documents are retrieved in the top 5.
- With Mixedbread Vector Store: Recall@5 = 92% → 92 out of every 100 truly relevant documents make it into the top 5.

This 8% absolute improvement (or ~9.5% relative improvement) in recall represents a substantial gain in retrieval performance. These retrieval benchmarks quantify the hidden ceiling imposed by relying solely on OCR. While better OCR helps, the results strongly indicate that a multimodal approach represents a fundamental leap forward.

## [Testing Generation: Setup and Results](https://www.mixedbread.com/blog/the-hidden-ceiling\#testing-generation-setup-and-results)

Okay, so multimodal retrieval finds better documents, overcoming the OCR ceiling. _But does this improved retrieval actually translate into more accurate final answers from the LLM?_ To find out, we tested the end-to-end RAG performance.

### [Generation Setup](https://www.mixedbread.com/blog/the-hidden-ceiling\#generation-setup)

We set up three scenarios, feeding the top 5 retrieved documents from each into the same powerful LLM ( **`gemini-2.5-flash-preview-04-17`**) for answer generation:

1. **Perfect OCR & Perfect Retrieval (Ground Truth):** Using the human-verified text for generation and the true evidence pages as an input ('Perfect Retrieval'). This represents the theoretical maximum performance achievable with the current models if they would have the correct context and perfect extraction.
2. **Perfect OCR & Retrieval**: Using the human-verified text for both BM25 retrieval and for the top 5 passages and generation context. This is the quality you would get if your OCR would be perfect with the current technology.
3. **Mixedbread OCR (Text-Based RAG):** Using text extracted by our high-quality OCR engine for both BM25 retrieval for the top 5 passages and generation context. This mirrors a standard, good-quality text-only RAG pipeline.
4. **Mixedbread Vector Store (Multimodal Retrieval):** Using our multimodal model to retrieve the top 5 _page images_, but then using the corresponding clean text extracted by Mixedbread OCR as the generation context. This isolates the benefit of _visual retrieval_ while keeping the generation input modality (text) consistent.

To measure success, we focused on the **Correct Answers** rate. We used **GPT-4.1** as an impartial judge, providing it with the original question, the ground truth answer, the ground truth _evidence text_, and the answer generated by `gemini-2.5-flash-preview-04-17` in each scenario. The final score is simply the number of correct answers divided by the total number of questions.

### [Generation Results: Better Retrieval = Better Answers](https://www.mixedbread.com/blog/the-hidden-ceiling\#generation-results-better-retrieval--better-answers)

The generation tests confirmed our hypothesis: superior retrieval leads directly to more accurate answers.

**Correct Answers Rate**https://www.mixedbread.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fgeneration-accuracy.af83e122.webp&w=3840&q=75&dpl=dpl_6D5g8JYtmzhd4sCHskMoarzC6kVS

This chart shows the percentage of correct answers from each generation method, averaged across 7 domains and judged by GPT-4.1. Higher values mean the LLM produced more accurate, ground-truth answers.

| Domain | Mixedbread OCR (ret & gen) | Perfect OCR + ret. | Mixedbread Vector Store (ret) + Mixedbread OCR (gen) | Perfect OCR + Perfect ret. |
| --- | --- | --- | --- | --- |
| academic | 0.711 | 0.797 | **0.876** | 0.904 |
| administration | 0.714 | 0.812 | **0.846** | 0.896 |
| finance | 0.618 | 0.686 | **0.742** | 0.877 |
| law | 0.866 | 0.898 | **0.909** | 0.950 |
| manual | 0.782 | 0.825 | **0.888** | 0.914 |
| news | 0.435 | 0.447 | **0.753** | 0.951 |
| textbook | 0.607 | 0.715 | **0.885** | 0.896 |
| **avg** | 0.676 | 0.740 | **0.843** | 0.912 |

Key takeaways from the generation tests:

1. **OCR Flaws Amplify During Generation:** Relying on standard OCR for both retrieval and generation resulted in a **25.8% decrease** in correct answers compared to using perfect text (0.677 vs 0.913). Flawed input context significantly degrades the LLM's ability to generate accurate answers.
2. **Better Retrieval Dramatically Boosts Correct Answers:** Simply swapping standard OCR-based _retrieval_ for Mixedbread Vector Store's _multimodal retrieval_ – while still using the _same potentially imperfect OCR text_ for generation – caused the average correct answer rate to jump massively from 0.677 to **0.843**. This single change **recovered 70%** of the accuracy lost due to the limitations of a standard OCR-based pipeline.
3. **Finding the Right Pages is Paramount:** The _quality of retrieval_ is often more critical than _perfect text_ in the generation context. Getting the _correct_ documents into the LLM's view, even with minor OCR imperfections, is far more beneficial than feeding the LLM slightly cleaner text from the _wrong_ documents.

These generation benchmarks confirm that state-of-the-art multimodal _retrieval_ can mitigate a large portion of the negative downstream effects of OCR errors.

## [Direct Image Generation: Is Vision-Only RAG Ready?](https://www.mixedbread.com/blog/the-hidden-ceiling\#direct-image-generation-is-vision-only-rag-ready)

Given the success of using visual information for retrieval, a natural question arises: can we skip OCR entirely, even for the generation step? What if we feed the _images_ of the retrieved pages directly to a powerful multimodal LLM like **Gemini 2.5 Flash** and ask it to generate the answer by "reading" the images? We tested this "Direct Image Understanding" approach:

**Correct Answers Rate** _(Average across 3 document domains)_

| Retrieval Method | Generation Input | Avg. Correct Answers | Performance vs. Perfect OCR |
| --- | --- | --- | --- |
| Perfect OCR (Ground Truth) | Perfect OCR Text | **0.899** | ±0.0% (Baseline) |
| **Mixedbread Vector Store** | **Mixedbread OCR Text** | **0.869** | **-3.3%** |
| Mixedbread OCR | Mixedbread OCR Text | 0.678 | -24.6% |
| Mixedbread Vector Store | Direct Image Input | 0.627 | -30.3% |

| Domain | Mixedbread OCR (ret. & gen.) | **Mixedbread Vector Store (ret.) + Mixedbread OCR (gen.)** | Mixedbread Vector Store (ret.) + Direct Image Input (gen.) | Perfect OCR + Retrieval |
| --- | --- | --- | --- | --- |
| academic | 0.712 | **0.876** | 0.534 | 0.904 |
| administration | 0.715 | **0.846** | 0.672 | 0.896 |
| textbook | 0.607 | **0.885** | 0.675 | 0.896 |
| **avg** | **0.678** | **0.869** | **0.627** | **0.899** |

The results were surprising:

- **Direct Image Input Lags Significantly:** Feeding page images directly to the LLM for generation yielded the _lowest_ average correct answers (0.627).
- **Visual Retrieval vs. Visual Generation:** Multimodal models excel at using visual cues for _retrieval_, but current models still struggle with fine-grained _extraction_ directly from pixels _across multiple documents_ during generation, compared to working with pre-processed text.
- **Quality OCR Text Still Best for Generation (For Now):** Providing clean, explicit text to the LLM currently leads to the most accurate answers.

**In essence: While fully visual RAG is an exciting possibility, today's reality is that combining the strengths of multimodal retrieval with high-quality OCR text for generation provides the best overall performance.**

## [Illustrative Examples: Where Standard OCR Falters](https://www.mixedbread.com/blog/the-hidden-ceiling\#illustrative-examples-where-standard-ocr-falters)

To make the impact of OCR limitations more concrete, let's examine a few specific scenarios from our benchmark data. These examples highlight common situations where traditional OCR-based systems can struggle and demonstrate how a multimodal approach to retrieval can lead to more accurate document interpretation.

### [Example 1: The Challenge of Handwritten Data in Regulatory Filings](https://www.mixedbread.com/blog/the-hidden-ceiling\#example-1-the-challenge-of-handwritten-data-in-regulatory-filings)

**The Scenario:** Regulatory filings, such as a telecommunications company's PUCO annual report, frequently combine structured typed content with critical handwritten financial figures. This mixture presents a significant **OCR challenge**, as traditional systems often fail to accurately recognize handwritten entries, leading to potential compliance and analysis issues.https://www.mixedbread.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fexample-1.8c2adea3.png&w=3840&q=75&dpl=dpl_6D5g8JYtmzhd4sCHskMoarzC6kVS

**Typical OCR Output & Its Limitations:**
When processed by a standard OCR engine, the crucial handwritten financial data is often missed entirely or garbled:

```
Annual Report of TSC Communications, Inc. Year Ended December 31, 2026

Instructions:
Schedule 2 is used for PUCO annual assessment purposes pursuant to Section 4905.10, RC...

STATEMENT OF INTRASTATE GROSS EARNINGS (REVENUE)
                                                       Amount
Line                                                   Ohio
No.        Item                                         Intrastate
1    Operating and Miscellaneous Revenue - Wholesale    [???????]
     Cellular Communications, Radio Common Carrier...
2    Other Revenue, Dividend and Interest Income...      [???????]
3    SUBTOTAL                  (1) + (2)                [???????]
4    Earnings or receipts from sales to other public    (          )
     utilities for resale
5    TOTAL                     (3) + (4)                [???????]

     [???????]
     [???????]
     [???????]
```

**Impact on RAG Systems:**
Consequently, if a query such as, _"What is the total revenue of TSC Communications?"_ is posed, a RAG system relying on this flawed OCR output would likely respond: _"Unable to determine revenue figures from the available document."_ This necessitates manual data review, delaying important reporting and analytical tasks.

**The Multimodal Approach:**
In contrast, the multimodal system processes both the structured form and the handwritten financial figures by analyzing the document's visual layout and handwriting patterns. This holistic understanding allows it to correctly identify the total revenue as **$2,775,060**, along with component values ($2,325,472 for operating revenue and $449,588 for other revenue). This capability enables accurate, automated responses regarding the company's financial standing and regulatory obligations.

* * *

### [Example 2: Deciphering Visual Trends in Financial Charts](https://www.mixedbread.com/blog/the-hidden-ceiling\#example-2-deciphering-visual-trends-in-financial-charts)

**The Scenario:** Quarterly investment reports often feature charts, like stacked area charts showing portfolio allocation, to convey critical trends. The **OCR challenge** here is that traditional OCR primarily extracts textual elements (titles, labels) but fails to capture the actual visual data representing the trends themselves.https://www.mixedbread.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fexample-2.14a5258e.png&w=3840&q=75&dpl=dpl_6D5g8JYtmzhd4sCHskMoarzC6kVS

**Typical OCR Output & Its Limitations:**
A standard OCR tool might only extract the labels and title, leaving out the core data:

```
Portfolio Allocation Trends (Q1 2023 - Q4 2024)
Percentage (%)
100
75
50
25
0
Q1 2023, Q2 2023, Q3 2023, Q4 2023, Q1 2024, Q2 2024, Q3 2024, Q4 2024
Cash, Commodities,Real Estate,Fixed Income, Equities
```

**Impact on RAG Systems:**
When a client asks, _"How has my equity exposure changed over the past year?"_, a RAG system using this limited OCR output might provide only generic information about portfolio components. It would completely miss the crucial visual trend, such as a 13 percentage point increase in equity exposure, which is essential for understanding investment risk.

**The Multimodal Approach:**
The multimodal system, by directly analyzing the chart visually, recognizes both the allocation percentages at each time point and the overall trend patterns. This allows it to accurately respond: _"Your equity allocation has increased significantly from 45% to 58% over the past year, representing the largest shift in your portfolio composition."_ The system can even extract specific quarterly changes to illustrate the gradual increase.

* * *

### [Example 3: Navigating Complex Financial Tables](https://www.mixedbread.com/blog/the-hidden-ceiling\#example-3-navigating-complex-financial-tables)

**The Scenario:** Financial reports frequently contain multi-column tables detailing revenue breakdowns and operating expenses. The **OCR challenge** with such complex table structures lies in maintaining correct column and row alignment; failures here can lead to financial figures being associated with incorrect business units or categories.https://www.mixedbread.com/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fexample-3.5f1ccfef.png&w=3840&q=75&dpl=dpl_6D5g8JYtmzhd4sCHskMoarzC6kVS

**Typical OCR Output & Its Limitations:**
Even if text is extracted, subtle misalignments or parsing errors by the OCR can corrupt the table's structure:

```
Operating Expenses
                                          Year Ended
                          Jan 26, 2025     Jan 28, 2024        $           %
                                                              Change       Change
                                           ($ in millions)
Research and development expenses    $        12,914    $         8,675    $    4,239        49 %
% of net revenue                                9.9 %              14.2 %
Sales, general and administrative expenses      3,491              2,654         837         32 %
% of net revenue                                2.7 %               4.4 %
  Total operating expenses           $        16,405    $        11,329    $   5,076         45 %
% of net revenue                               12.6 %              18.6 %
```

**Impact on RAG Systems:**
If a financial analyst asks, _"What percentage of revenue did R&D represent in 2025 compared to 2024?"_, a RAG system relying on poorly structured OCR output might misinterpret the relationships between figures. An erroneous response could be: _"R&D was 49% of revenue in 2025 compared to 8,675% in 2024."_ Such nonsensical answers arise from the system's inability to correctly understand the visual and semantic structure of the table.

**The Multimodal Approach:**
The multimodal system analyzes the visual structure of the table, correctly understanding the complex alignments and relationships between headers, dollar amounts, and percentage figures. This enables an accurate response: _"R&D expenses represented 9.9% of net revenue in 2025, down from 14.2% in 2024, despite a 49% increase in absolute R&D spending."_ The system properly interprets both the spatial layout and the semantic connections within the financial data.

## [The Mixedbread Vector Store Approach: Functionality and Implications](https://www.mixedbread.com/blog/the-hidden-ceiling\#the-mixedbread-vector-store-approach-functionality-and-implications)

The Vector Store is designed to address the observed limitations of OCR-dependent RAG systems. Its architecture is centered on leveraging multimodal information for retrieval through our `mxbai-omni-v0.1` model. This model directly analyzes and creates embeddings from the visual content of page screenshots, videos, and other multimodal data, enabling an understanding of layout, structure, tables, and charts in their original context. As shown in our benchmarks, this improved retrieval accuracy (NDCG@5) by approximately 12% compared to even perfect text extraction.

Concurrently with visual analysis, documents are processed by our OCR engine. The extracted text is stored and made available alongside the visual embeddings. This dual-modality approach offers distinct advantages for RAG pipelines:

- **Better Retrieval:** Visual analysis helps locate the most relevant documents, particularly in cases where text-only search might falter due to OCR errors or the nature of the content (e.g., charts, complex tables).
- **Optimized Generation Context:** High-quality OCRed text remains available, which is beneficial for current Large Language Models that primarily operate on textual input for generation.
- **Integrated Document Processing:** The system handles both visual embedding and text extraction automatically, so users don't have to worry about anything during data ingestion and preparation for RAG.
- **Adaptability for Future LLMs:** By storing both visual representations and text, systems are better prepared for future advancements in multimodal LLMs that might directly leverage richer image data for generation.

This integrated system design aims to improve overall RAG performance, as evidenced by the benchmarked retrieval gains and the recovery of 70% of generation accuracy typically diminished by OCR issues in conventional pipelines, all within a unified framework.

## [Conclusion: Navigating the OCR Bottleneck with Multimodal Retrieval](https://www.mixedbread.com/blog/the-hidden-ceiling\#conclusion-navigating-the-ocr-bottleneck-with-multimodal-retrieval)

The benchmark results presented indicate that Optical Character Recognition quality can be a significant limiting factor for RAG system performance, particularly with complex, real-world documents. Errors and omissions in text extraction can restrict both the ability to accurately retrieve relevant information and the quality of the final answers generated by an LLM.

An approach incorporating multimodal analysis for retrieval, such as that employed by the Mixedbread Vector Store, addresses some of these limitations. By directly interpreting visual information from page images, this method improved retrieval accuracy by approximately 12% (NDCG@5) compared to even perfect text extraction in our tests. This enhancement in retrieval subsequently contributed to recovering 70% of the generation accuracy that was otherwise diminished by OCR errors in more conventional pipelines.

While current Large Language Models generally perform optimally with high-quality text for the generation phase, the strong retrieval performance of multimodal systems highlights a path towards more robust document understanding. An integrated system that provides both visually-driven retrieval and high-quality OCR text offers a practical solution for current application needs. Furthermore, it establishes a foundation for adapting to future advancements in LLMs that may more directly leverage rich image data for generation tasks.

The findings suggest that for applications involving diverse and structurally complex documents, incorporating multimodal understanding into the retrieval process is a key consideration for improving the accuracy and reliability of RAG systems.

</details>


## Code Sources

<details>
<summary>Repository analysis for https://github.com/towardsai/course-ai-agents/blob/dev/lessons/11_multimodal/notebook.ipynb</summary>

# Repository analysis for https://github.com/towardsai/course-ai-agents/blob/dev/lessons/11_multimodal/notebook.ipynb

## Summary
Repository: towardsai/course-ai-agents
Branch: dev
File: notebook.ipynb
Lines: 1,402

Estimated tokens: 11.4k

## File tree
```Directory structure:
└── notebook.ipynb

```

## Extracted content
================================================
FILE: lessons/11_multimodal/notebook.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Lesson 11: Multimodal

This notebook demonstrates how to build multimodal AI systems that can process and understand multimodal data such text, images and documents using Google's Gemini models.

We will use the `google-genai` library to interact with Google's Gemini models.

**Learning Objectives:**

1. **Process multimodal content**: Learn to handle images and PDFs in different formats (bytes, base64, URLs) with Gemini models
2. **Implement object detection**: Use multimodal LLMs for visual analysis and structured output generation
3. **Build multimodal RAG systems**: Create and index embeddings for images, documents and text to enable semantic search across multimodal content
4. **Develop multimodal AI agents**: Construct ReAct agents that can search through and reason about multimodal information
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
#   Trying to load environment variables from `/Users/pauliusztin/Documents/01_projects/TAI/course-ai-agents/.env`

#   Environment variables loaded successfully.


"""
### Import Key Packages
"""

import base64
import io
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types
from IPython.display import Image as IPythonImage
from PIL import Image as PILImage

from lessons.utils import pretty_print

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
## 2. Applying multimodal LLMs to images and PDFs

There are three core ways we can process images and PDFs with multimodal LLMs:
1. As raw bytes
2. As base64 encoded strings
3. As URLs

We will first look into how we can process images and then PDFs.

Now, let's look at our test image:

"""

def display_image(image_path: Path) -> None:
    """
    Display an image from a file path in the notebook.

    Args:
        image_path: Path to the image file to display

    Returns:
        None
    """

    image = IPythonImage(filename=image_path, width=400)
    display(image)


display_image(Path("images") / "image_1.jpeg")
# Output:
#   <IPython.core.display.Image object>

"""
### 2.1 As raw bytes
"""

def load_image_as_bytes(
    image_path: Path, format: Literal["WEBP", "JPEG", "PNG"] = "WEBP", max_width: int = 600, return_size: bool = False
) -> bytes | tuple[bytes, tuple[int, int]]:
    """
    Load an image from file path and convert it to bytes with optional resizing.

    Args:
        image_path: Path to the image file to load
        format: Output image format (WEBP, JPEG, or PNG). Defaults to "WEBP"
        max_width: Maximum width for resizing. If image width exceeds this, it will be resized proportionally. Defaults to 600
        return_size: If True, returns both bytes and image size tuple. Defaults to False

    Returns:
        bytes: Image data as bytes, or tuple of (bytes, (width, height)) if return_size is True
    """

    image = PILImage.open(image_path)
    if image.width > max_width:
        ratio = max_width / image.width
        new_size = (max_width, int(image.height * ratio))
        image = image.resize(new_size)

    byte_stream = io.BytesIO()
    image.save(byte_stream, format=format)

    if return_size:
        return byte_stream.getvalue(), image.size

    return byte_stream.getvalue()

"""
Load image:
"""

image_bytes = load_image_as_bytes(image_path=Path("images") / "image_1.jpeg", format="WEBP")
pretty_print.wrapped([f"Bytes `{image_bytes[:30]}...`", f"Size: {len(image_bytes)} bytes"], title="Image as Bytes")
# Output:
#   [93m------------------------------------------ Image as Bytes ------------------------------------------[0m

#     Bytes `b'RIFF`\xad\x00\x00WEBPVP8 T\xad\x00\x00P\xec\x02\x9d\x01*X\x02X\x02'...`

#   [93m----------------------------------------------------------------------------------------------------[0m

#     Size: 44392 bytes

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
Compute captions:
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/webp",
        ),
        "Tell me what is in this image in one paragraph.",
    ],
)
pretty_print.wrapped(response.text, title="Image 1 Caption")

# Output:
#   [93m----------------------------------------- Image 1 Caption -----------------------------------------[0m

#     This striking image features a massive, dark metallic robot, its powerful form detailed with intricate circuit patterns on its head and piercing red glowing eyes. Perched playfully on its right arm is a small, fluffy grey tabby kitten, its front paw raised as if exploring or batting at the robot's armored limb, while its gaze is directed slightly off-frame. The robot's large, segmented hand is visible beneath the kitten. The background suggests an industrial or workshop environment, with hints of metal structures and natural light filtering in from an unseen window, creating a dramatic contrast between the soft, vulnerable kitten and the formidable, mechanical sentinel.

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
Using the same approach, we can easily pass multiple images simultaneously. For example, the previous one plus the one below, and compare them:
"""

display_image(Path("images") / "image_2.jpeg")
# Output:
#   <IPython.core.display.Image object>

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        types.Part.from_bytes(
            data=load_image_as_bytes(image_path=Path("images") / "image_1.jpeg", format="WEBP"),
            mime_type="image/webp",
        ),
        types.Part.from_bytes(
            data=load_image_as_bytes(image_path=Path("images") / "image_2.jpeg", format="WEBP"),
            mime_type="image/webp",
        ),
        "What's the difference between these two images? Describe it in one paragraph.",
    ],
)
pretty_print.wrapped(response.text, title="Differences between images")

# Output:
#   [93m------------------------------------ Differences between images ------------------------------------[0m

#     The primary difference between the two images lies in the nature of the interaction depicted and their respective settings. In the first image, a small, grey kitten is shown curiously interacting with a large, metallic robot, gently perched on its arm within what appears to be a clean, well-lit workshop or industrial space. Conversely, the second image portrays a tense and aggressive confrontation between a fluffy white dog and a sleek black robot, both in combative stances, amidst a cluttered and grimy urban alleyway filled with trash and graffiti.

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
### 2.2 As base64 encoded strings

Now, let's load the same image as base64:
"""

from typing import cast


def load_image_as_base64(
    image_path: Path, format: Literal["WEBP", "JPEG", "PNG"] = "WEBP", max_width: int = 600, return_size: bool = False
) -> str:
    """
    Load an image and convert it to base64 encoded string.

    Args:
        image_path: Path to the image file to load
        format: Output image format (WEBP, JPEG, or PNG). Defaults to "WEBP"
        max_width: Maximum width for resizing. If image width exceeds this, it will be resized proportionally. Defaults to 600
        return_size: Parameter passed to load_image_as_bytes function. Defaults to False

    Returns:
        str: Base64 encoded string representation of the image
    """

    image_bytes = load_image_as_bytes(image_path=image_path, format=format, max_width=max_width, return_size=False)

    return base64.b64encode(cast(bytes, image_bytes)).decode("utf-8")

image_base64 = load_image_as_base64(image_path=Path("images") / "image_1.jpeg", format="WEBP")
pretty_print.wrapped(
    [f"Base64: {image_base64[:100]}...`", f"Size: {len(image_base64)} characters"], title="Image as Base64"
)
# Output:
#   [93m----------------------------------------- Image as Base64 -----------------------------------------[0m

#     Base64: UklGRmCtAABXRUJQVlA4IFStAABQ7AKdASpYAlgCPm0ylEekIqInJnQ7gOANiWdtk7FnEo2gDknjPixW9SNSb5P7IbBNhLn87Vtp...`

#   [93m----------------------------------------------------------------------------------------------------[0m

#     Size: 59192 characters

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
On average base64 format is 33% larger than raw bytes. As we can see in this use case as well:
"""

print(f"Image as Base64 is {(len(image_base64) - len(image_bytes)) / len(image_bytes) * 100:.2f}% larger than as bytes")
# Output:
#   Image as Base64 is 33.34% larger than as bytes


"""
Now, let's recompute the image caption using this method:
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        types.Part.from_bytes(data=image_base64, mime_type="image/webp"),
        "Tell me what is in this image in one paragraph.",
    ],
)
response.text
# Output:
#   "The image features a striking contrast between a large, formidable robot and a small, adorable kitten. The robot, crafted from dark, sleek metallic armor with intricate circuitry patterns on its head, possesses piercing red glowing eyes that appear to be focused on its tiny companion. A fluffy, gray tabby kitten is playfully perched on the robot's massive metallic arm and shoulder, its small paws resting gently on the armored surface as it looks up with curiosity. The scene is set in what looks like an industrial or workshop environment, with warm light filtering in from the background, highlighting this unexpected and endearing interaction between advanced technology and natural innocence."

"""
### 2.3 As public URLs

Using Gemini `url_context` out-of-the-box tool, we can automatically visit and parse webpages, PDFs, and images from the open internet. You only have to provide the direct URL in the prompt and configure the `url_context` tool. This makes it a no-brainer to parse multiple data formats when available online:
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents="Based on the provided paper as a PDF, tell me how ReAct works: https://arxiv.org/pdf/2210.03629",
    config=types.GenerateContentConfig(tools=[{"url_context": {}}]),
)
pretty_print.wrapped(response.text, title="How ReAct works")
# Output:
#   [93m----------------------------------------- How ReAct works -----------------------------------------[0m

#     

#   

#   ReAct is a novel paradigm for large language models (LLMs) that combines reasoning (Thought) and acting (Action) in an interleaved manner to solve diverse language and decision-making tasks. This approach allows the model to:

#   

#   *   **Reason to Act:** Generate verbal reasoning traces to induce, track, and update action plans, and handle exceptions.

#   *   **Act to Reason:** Interface with and gather additional information from external sources (like knowledge bases or environments) to incorporate into its reasoning.

#   

#   **How it works:**

#   

#   Instead of just generating a direct answer (Standard prompting) or a chain of thought without external interaction (CoT), or only actions (Act-only), ReAct augments the LLM's action space to include a "language space" for generating "thoughts" or reasoning traces.

#   

#   1.  **Thought:** The model explicitly generates a thought, which is a verbal reasoning trace. This thought helps the model to:

#       *   Decompose task goals and create action plans.

#       *   Inject commonsense knowledge.

#       *   Extract important information from observations.

#       *   Track progress and adjust action plans.

#       *   Handle exceptions.

#   2.  **Action:** Based on the current thought and context, the model performs a task-specific action. This could involve:

#       *   Searching external databases (e.g., Wikipedia API using `search[entity]` or `lookup[string]`).

#       *   Interacting with an environment (e.g., `go to cabinet 1`, `take pepper shaker 1`).

#       *   Finishing the task with an answer (`finish[answer]`).

#   3.  **Observation:** The environment provides an observation feedback based on the executed action.

#   

#   This cycle of Thought, Action, and Observation continues until the task is completed.

#   

#   **Benefits of ReAct:**

#   

#   *   **Improved Performance:** ReAct consistently outperforms baselines that only perform reasoning or acting in isolation on tasks like question answering (HotpotQA), fact verification (FEVER), text-based games (ALFWorld), and webpage navigation (WebShop).

#   *   **Reduced Hallucination and Error Propagation:** By interacting with external sources, ReAct can overcome issues of hallucination and error propagation common in chain-of-thought reasoning that relies solely on internal knowledge.

#   *   **Human Interpretability and Trustworthiness:** The interleaved reasoning traces make the model's decision-making process more interpretable and trustworthy, as humans can inspect the thoughts and actions.

#   *   **Flexibility and Generalizability:** ReAct is flexible enough to be applied to diverse tasks with different action spaces and reasoning needs, and it shows strong generalization with only a few in-context examples.

#   *   **Human Alignment and Controllability:** Humans can control or correct the agent's behavior by editing its thoughts, enabling new forms of human-machine collaboration.

#   

#   For example, in a question-answering task, ReAct might first *think* about what to search, then *act* by searching Wikipedia, *observe* the results, *think* about what the results mean and what to search next, and so on, until it can *think* of the final answer and *act* to finish the task.

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
### 2.4 As URLs from private data lakes

At the time of writing this notebook, Gemini works well primarily with GCP Cloud Storage links and not with other buckets such as S3. Buckets are excellent for production use cases, but they complicate our simple demonstration. Therefore, we will show you a mocked example.

The code would look like this, where you have to change the `uri` and ensure the LLM has the right permissions to your GCS bucket:
```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        types.Part.from_uri(uri="gs://gemini-images/image_1.jpeg", mime_type="image/webp"),
        "Tell me what is in this image in one paragraph.",
    ],
)
```
"""

"""
### 2.5 Object detection with LLMs

As a more exciting example, let's do object detection with multimodal LLMs.

First, let's define the output Pydantic models:
"""

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    ymin: float
    xmin: float
    ymax: float
    xmax: float
    label: str = Field(
        default="The category of the object found within the bounding box. For example: cat, dog, diagram, robot."
    )


class Detections(BaseModel):
    bounding_boxes: list[BoundingBox]

"""
Then the prompt and image:
"""

prompt = """
Detect all of the prominent items in the image. 
The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.
Also, output the label of the object found within the bounding box.
"""

image_bytes, image_size = load_image_as_bytes(
    image_path=Path("images") / "image_1.jpeg", format="WEBP", return_size=True
)

"""
Now, let's call the LLM:
"""

config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=Detections,
)

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/webp",
        ),
        prompt,
    ],
    config=config,
)

detections = cast(Detections, response.parsed)
pretty_print.wrapped([f"Image size: {image_size}", *detections.bounding_boxes], title="Detections")
# Output:
#   [93m-------------------------------------------- Detections --------------------------------------------[0m

#     Image size: (600, 600)

#   [93m----------------------------------------------------------------------------------------------------[0m

#     ymin=1.0 xmin=450.0 ymax=997.0 xmax=1000.0 label='robot'

#   [93m----------------------------------------------------------------------------------------------------[0m

#     ymin=269.0 xmin=39.0 ymax=782.0 xmax=530.0 label='kitten'

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
Let's also visualize the bounding boxes: 
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def visualize_detections(detections: Detections, image_path: Path) -> None:
    """
    Visualize detected bounding boxes on an image with red rectangles and labels.

    Args:
        detections: Detections object containing bounding boxes in [ymin, xmin, ymax, xmax] format normalized to 0-1000
        image_path: Path to the image file to visualize

    Returns:
        None: Displays the image with bounding boxes in the notebook
    """

    # Clear any existing plots to prevent overlapping
    plt.clf()

    image = PILImage.open(image_path)
    image_array = np.array(image)
    img_height, img_width = image_array.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(image_array)

    for bbox in detections.bounding_boxes:
        # Convert normalized coordinates (0-1000) to pixel coordinates
        xmin = (bbox.xmin / 1000) * img_width
        ymin = (bbox.ymin / 1000) * img_height
        xmax = (bbox.xmax / 1000) * img_width
        ymax = (bbox.ymax / 1000) * img_height

        # Calculate box dimensions (matplotlib uses bottom-left corner + width/height)
        width = xmax - xmin
        height = ymax - ymin

        # Create rectangle patch (x, y is bottom-left corner)
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=3, edgecolor="red", facecolor="none")

        # Add rectangle to the plot
        ax.add_patch(rect)

        # Add label text (positioned at top-left of bounding box)
        ax.text(
            xmin,
            ymin + 5,  # Slightly above the box
            bbox.label[:15],
            fontsize=12,
            color="red",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Remove axis ticks and labels for cleaner display
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Object Detection Results: {image_path.name}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.show()

visualize_detections(detections, Path("images") / "image_1.jpeg")
# Output:
#   <Figure size 640x480 with 0 Axes>
#   <Figure size 800x600 with 1 Axes>

"""
### 2.6 Working with PDFs

Ultimately, let's see how we can work with PDFs. We will use the legendary `Attention Is All You Need` Paper as an example. 

To display it, we extracted the first 3 pages of the PDF as images. For example, this is how the page looks:

"""

display_image(Path("images") / "attention_is_all_you_need_0.jpeg")
# Output:
#   <IPython.core.display.Image object>

"""
We can treat PDFs similarly to images. Therefore, we can pass PDFs as bytes:
"""

pdf_bytes = (Path("pdfs") / "attention_is_all_you_need_paper.pdf").read_bytes()
pretty_print.wrapped(f"Bytes: {pdf_bytes[:40]}...", title="PDF bytes")
# Output:
#   [93m-------------------------------------------- PDF bytes --------------------------------------------[0m

#     Bytes: b'%PDF-1.7\n%\xe2\xe3\xcf\xd3\n24 0 obj\n<<\n/Filter /Flat'...

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
Call the LLM:
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
        "What is this document about? Provide a brief summary of the main topics.",
    ],
)
pretty_print.wrapped(response.text, title="PDF Summary (as bytes)")
# Output:
#   [93m-------------------------------------- PDF Summary (as bytes) --------------------------------------[0m

#     This document introduces the **Transformer**, a novel neural network architecture designed for **sequence transduction tasks** (like machine translation).

#   

#   Its main topics include:

#   

#   1.  **Dispensing with Recurrence and Convolutions**: Unlike previous dominant models (RNNs and CNNs), the Transformer relies *solely* on **attention mechanisms**, eliminating the need for sequential computation.

#   2.  **Attention Mechanisms**: It details the **Scaled Dot-Product Attention** and **Multi-Head Attention** as its core building blocks, explaining how they allow the model to weigh different parts of the input sequence.

#   3.  **Parallelization and Efficiency**: The paper highlights that the Transformer's architecture allows for significantly more parallelization during training, leading to **faster training times** compared to prior models.

#   4.  **Superior Performance**: It demonstrates that the Transformer achieves **state-of-the-art results** on machine translation tasks (English-to-German and English-to-French) and generalizes well to other tasks like English constituency parsing.

#   5.  **Positional Encoding**: Since the model lacks recurrence or convolution, it introduces positional encodings to inject information about the relative or absolute position of tokens in the sequence.

#   

#   In essence, the document proposes and validates that **attention alone is sufficient** for building high-quality, efficient, and parallelizable sequence transduction models.

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
Alternatively, as base64 encoded strings:
"""

def load_pdf_as_base64(pdf_path: Path) -> str:
    """
    Load a PDF file and convert it to base64 encoded string.

    Args:
        pdf_path: Path to the PDF file to load

    Returns:
        str: Base64 encoded string representation of the PDF
    """

    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

"""
Load the PDF:
"""

pdf_base64 = load_pdf_as_base64(pdf_path=Path("pdfs") / "attention_is_all_you_need_paper.pdf")
pretty_print.wrapped(f"Base64: {pdf_base64[:40]}...", title="PDF as Base64")
# Output:
#   [93m------------------------------------------ PDF as Base64 ------------------------------------------[0m

#     Base64: JVBERi0xLjcKJeLjz9MKMjQgMCBvYmoKPDwKL0Zp...

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
Call the LLM:
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        "What is this document about? Provide a brief summary of the main topics.",
        types.Part.from_bytes(data=pdf_base64, mime_type="application/pdf"),
    ],
)

pretty_print.wrapped(response.text, title="PDF Summary (as base64)")
# Output:
#   [93m------------------------------------- PDF Summary (as base64) -------------------------------------[0m

#     This document introduces the **Transformer**, a novel neural network architecture for **sequence transduction models**, primarily applied to **machine translation**.

#   

#   Here's a brief summary of the main topics:

#   

#   *   **Core Innovation:** The Transformer proposes to completely abandon recurrent neural networks (RNNs) and convolutional neural networks (CNNs), relying *solely on attention mechanisms* (specifically "multi-head self-attention") for learning dependencies between input and output sequences.

#   *   **Problem Addressed:** Traditional RNNs/CNNs suffer from inherent sequential computation, which limits parallelization and makes it difficult to efficiently learn long-range dependencies. The Transformer addresses this by allowing constant-time operations for relating any two positions in a sequence.

#   *   **Architecture:** It maintains an encoder-decoder structure, where both the encoder and decoder are composed of stacks of self-attention and point-wise fully connected layers. Positional encodings are added to input embeddings to inject information about the order of the sequence.

#   *   **Key Advantages:** The Transformer is significantly more parallelizable and requires substantially less training time compared to previous state-of-the-art models.

#   *   **Performance:** It achieves new state-of-the-art results on major machine translation benchmarks (WMT 2014 English-to-German and English-to-French) and demonstrates strong generalization to other tasks, such as English constituency parsing.

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
Now, let's do a more interesting example and detect the diagrams from a page of the transformers paper, such as the one below:
"""

display_image(Path("images") / "attention_is_all_you_need_1.jpeg")
# Output:
#   <IPython.core.display.Image object>

"""
Define the object detection prompt to detect diagrams (similar to how we did for images):
"""

prompt = """
Detect all the diagrams from the provided image as 2d bounding boxes. 
The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.
Also, output the label of the object found within the bounding box.
"""

image_bytes, image_size = load_image_as_bytes(
    image_path=Path("images") / "attention_is_all_you_need_1.jpeg", format="WEBP", return_size=True
)

"""
Call the LLM:
"""

config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=Detections,
)
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/webp",
        ),
        prompt,
    ],
    config=config,
)
detections = cast(Detections, response.parsed)
pretty_print.wrapped([f"Image size: {image_size}", *detections.bounding_boxes], title="Detections")
# Output:
#   [93m-------------------------------------------- Detections --------------------------------------------[0m

#     Image size: (600, 776)

#   [93m----------------------------------------------------------------------------------------------------[0m

#     ymin=88.0 xmin=309.0 ymax=515.0 xmax=681.0 label='diagram'

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
Visualize the detections:
"""

visualize_detections(detections, Path("images") / "attention_is_all_you_need_1.jpeg")
# Output:
#   <Figure size 640x480 with 0 Axes>
#   <Figure size 800x600 with 1 Axes>

"""
## 3. Implementing multimodal RAG for images, PDFs and text

To bring everything we did in this course together, let's implement a multimodal RAG system that works with text, images, and PDFs.

These are the images and PDF pages (as images) we will index for semantic search:
"""

def display_image_grid(image_paths: list[Path], rows: int = 2, cols: int = 2, figsize: tuple = (8, 6)) -> None:
    """
    Display a grid of images.

    Args:
        image_paths: List of paths to images to display
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        figsize: Figure size as (width, height)
    """

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.ravel()

    for idx, img_path in enumerate(image_paths[: rows * cols]):
        img = PILImage.open(img_path)
        axes[idx].imshow(img)
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


display_image_grid(
    image_paths=[
        Path("images") / "image_1.jpeg",
        Path("images") / "image_2.jpeg",
        Path("images") / "image_3.jpeg",
        Path("images") / "image_4.jpeg",
        Path("images") / "attention_is_all_you_need_1.jpeg",
        Path("images") / "attention_is_all_you_need_2.jpeg",
    ],
    rows=2,
    cols=3,
)
# Output:
#   <Figure size 800x600 with 6 Axes>

"""
Now, let's define the core functions.

First, one that creates image descriptions:
"""

from io import BytesIO
from typing import Any

import numpy as np


def generate_image_description(image_bytes: bytes) -> str:
    """
    Generate a detailed description of an image using Gemini Vision model.

    Args:
        image_bytes: Image data as bytes

    Returns:
        str: Generated description of the image
    """

    try:
        # Convert bytes back to PIL Image for vision model
        img = PILImage.open(BytesIO(image_bytes))

        # Use Gemini Vision model to describe the image
        prompt = """
        Describe this image in detail for semantic search purposes. 
        Include objects, scenery, colors, composition, text, and any other visual elements that would help someone find 
        this image through text queries.
        """

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[prompt, img],
        )

        if response and response.text:
            description = response.text.strip()

            return description
        else:
            print("❌ No description generated from vision model")

            return ""

    except Exception as e:
        print(f"❌ Failed to generate image description: {e}")

        return ""


"""
Another one that creates embedding using `gemini_embedding-001`, based on the given input:
"""

def embed_text_with_gemini(content: str) -> np.ndarray | None:
    """
    Embed text content using Gemini's text embedding model.

    Args:
        content: Text string to embed

    Returns:
        np.ndarray | None: Embedding vector as numpy array or None if failed
    """

    try:
        result = client.models.embed_content(
            model="gemini-embedding-001",  # Gemini's text embedding model
            contents=[content],
        )
        if not result or not result.embeddings:
            print("❌ No embedding data found in response")
            return None

        return np.array(result.embeddings[0].values)

    except Exception as e:
        print(f"❌ Failed to embed text: {e}")
        return None

"""
Let's see how this works:
"""

embedding = embed_text_with_gemini("This is a test")
embedding
# Output:
#   array([-0.02252334, -0.00076438,  0.00240217, ..., -0.00574729,

#          -0.00052345, -0.00213343], shape=(3072,))

"""
As we can see below, it creates a 3072 embedding from the input text:
"""

embedding.shape
# Output:
#   (3072,)

"""
Let's glue these functions and create the vector index out of our test images and PDF pages:
"""

from typing import cast


def create_vector_index(image_paths: list[Path]) -> list[dict]:
    """
    Create embeddings for images by generating descriptions and embedding them.

    This function processes a list of image paths by:
    1. Loading each image as bytes
    2. Generating a text description using Gemini Vision
    3. Creating an embedding of that description using Gemini Embeddings

    Args:
        image_paths (list[Path]): List of paths to image files to process

    Returns:
        list[dict]: List of dictionaries with the following keys:
            - content (bytes): Raw image bytes
            - type (str): Always "image"
            - filename (Path): Original image path
            - description (str): Generated image description
            - embedding (np.ndarray): Vector embedding of the description
    """

    vector_index = []
    for image_path in image_paths:
        image_bytes = cast(bytes, load_image_as_bytes(image_path, format="WEBP", return_size=False))

        image_description = generate_image_description(image_bytes)
        pretty_print.wrapped(f"`{image_description[:500]}...`", title="Generated image description:")

        # IMPORTANT NOTE: When working with multimodal embedding models, we can directly embed the
        # `image_bytes` instead of generating and embedding the description. Otherwise, everything
        # else remains the same within the whole RAG system.
        image_embedding = embed_text_with_gemini(image_description)

        vector_index.append(
            {
                "content": image_bytes,
                "type": "image",
                "filename": image_path,
                "description": image_description,
                "embedding": image_embedding,
            }
        )

    return vector_index

"""
We call the `create_vector_index` function on all the images from the `images` dir:
"""

image_paths = list(Path("images").glob("*.jpeg"))
vector_index = create_vector_index(image_paths)
# Output:
#   [93m----------------------------------- Generated image description: -----------------------------------[0m

#     `This image is a page from a technical or scientific document, likely a research paper, textbook, or dissertation related to machine learning, deep learning, or artificial intelligence.

#   

#   **Overall Composition & Scenery:**

#   The image is a vertically oriented page (A4 or similar size) with a clean, academic layout. The dominant colors are black text on a white background. The page is filled with text and features two prominent block diagrams at the top, along with a mathematical equation in the lowe...`

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m----------------------------------- Generated image description: -----------------------------------[0m

#     `This image is a detailed, photorealistic digital rendering or illustration depicting an unlikely interaction between a large, imposing robot and a small, delicate kitten in an industrial setting.

#   

#   **Objects:**

#   *   **Robot:** The dominant figure is a large, humanoid robot, occupying the right side of the frame. Its body is constructed from dark, metallic armored plates in shades of charcoal, gunmetal, and dark grey, with visible bolts, rivets, and segmented joints suggesting a heavy, industrial d...`

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m----------------------------------- Generated image description: -----------------------------------[0m

#     `This image depicts a dramatic and tense confrontation between a large, fluffy white dog and a sleek, dark humanoid robot in a desolate urban alleyway.

#   

#   **Objects and Characters:**

#   

#   *   **White Dog:** Positioned on the left, a large, fluffy white dog, strongly resembling a Samoyed or other Spitz breed (like a white husky or malamute), is captured mid-lunge. Its mouth is wide open, baring sharp teeth, indicative of barking, snarling, or attacking. Its ears are forward, and its tail is high and cur...`

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m----------------------------------- Generated image description: -----------------------------------[0m

#     `This image is a detailed, close-up shot of an African American man intently working on the internal components of an open desktop computer tower.

#   

#   **Objects:**

#   *   **Person:** An adult African American male with a neatly trimmed beard (streaked with some grey) and black-rimmed glasses is positioned on the left side, looking down with a focused expression into the computer case. His dark-skinned hands are prominent, one holding a screwdriver and the other steadying a component or pointing. He wea...`

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m----------------------------------- Generated image description: -----------------------------------[0m

#     `This image is a detailed technical document, likely from a research paper or academic publication, featuring a prominent diagram of the Transformer model architecture alongside explanatory text.

#   

#   **Overall Composition & Scenery:**

#   The image is set against a clean white background. The top half is dominated by a multi-colored block diagram, while the bottom half contains black text organized into sections and paragraphs. A page number "3" is centered at the very bottom.

#   

#   **Objects & Diagram Eleme...`

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m----------------------------------- Generated image description: -----------------------------------[0m

#     `This image depicts a dynamic, high-energy futuristic battle scene between two humanoid robots or mechs.

#   

#   **Objects:**

#   *   **Two Robots/Mechs:**

#       *   **Left Robot:** Appears sleek and agile, made of highly reflective, polished silver or chrome metal. Its head, chest, and arms feature prominent electric blue glowing lines and accents, including a bright blue visor or eye piece. It is in the process of delivering a powerful punch with its right fist into the other robot. Its posture suggests for...`

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m----------------------------------- Generated image description: -----------------------------------[0m

#     `This image is a digital scan or representation of the first page of a widely recognized academic research paper. The dominant visual element is text, set against a plain white background, simulating a printed document.

#   

#   **Overall Composition & Layout:**

#   The page is organized in a standard academic paper format with a title, author list, abstract, and footnotes. Text is primarily black, with a small section of red text at the very top. A vertical, faint grey text string (likely a watermark or ide...`

#   [93m----------------------------------------------------------------------------------------------------[0m


if len(vector_index) == 0:
    pretty_print.wrapped("Could not create the vector index.", title="❌")
else:
    pretty_print.wrapped(f"Successfully created {len(vector_index)} embeddings under the `vector_index` variable", title="✅")
# Output:
#   [93m------------------------------------------------ ✅ ------------------------------------------------[0m

#     Successfully created 7 embeddings under the `vector_index` variable

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
This is how an element from the `vector_index` looks like:
"""

vector_index[0].keys()
# Output:
#   dict_keys(['content', 'type', 'filename', 'description', 'embedding'])

vector_index[0]["embedding"].shape
# Output:
#   (3072,)

print(f"{vector_index[0]['description'][:150]}...")
# Output:
#   This image is a page from a technical or scientific document, likely a research paper, textbook, or dissertation related to machine learning, deep lea...


"""
Now let's define a function that finds `top_k` most similar items from the vector_index based on a user query:
"""

from sklearn.metrics.pairwise import cosine_similarity


def search_multimodal(query_text: str, vector_index: list[dict], top_k: int = 3) -> list[Any]:
    """
    Search for most similar documents to query using direct Gemini client.

    This function embeds the query text and compares it against pre-computed embeddings
    of document descriptions to find the most semantically similar matches.

    Args:
        query_text: Text query to search for
        docs: List of document dictionaries containing embeddings and metadata
        top_k: Number of top results to return. Defaults to 3

    Returns:
        list[Any]: List of document dictionaries with similarity scores, sorted by relevance
    """

    print(f"\n🔍 Embedding query: '{query_text}'")

    query_embedding = embed_text_with_gemini(query_text)

    if query_embedding is None:
        print("❌ Failed to embed query")
        return []
    else:
        print("✅ Query embedded successfully")

    # Calculate similarities using our custom function
    embeddings = [doc["embedding"] for doc in vector_index]
    similarities = cosine_similarity([query_embedding], embeddings).flatten()

    # Get top results
    top_indices = np.argsort(similarities)[::-1][:top_k]  # type: ignore

    results = []
    for idx in top_indices.tolist():
        results.append({**vector_index[idx], "similarity": similarities[idx]})

    return results

"""
Let's test this with an example:
"""

query = "what is the architecture of the transformer neural network?"
results = search_multimodal(query, vector_index, top_k=1)

if not results:
    pretty_print.wrapped("❌ No results found", title="❌")
else:
    result = results[0]

    pretty_print.wrapped(
        [
            f"Similarity {result['similarity']:.3f}",
            f"Filename {result['filename']}",
            f"Description `{result['description'][:1000]}...`",
        ],
        title=f"Results for query = {query}",
    )
    display_image(Path(result["filename"]))
# Output:
#   

#   🔍 Embedding query: 'what is the architecture of the transformer neural network?'

#   ✅ Query embedded successfully

#   [93m--------- Results for query = what is the architecture of the transformer neural network? ---------[0m

#     Similarity 0.744

#   [93m----------------------------------------------------------------------------------------------------[0m

#     Filename images/attention_is_all_you_need_1.jpeg

#   [93m----------------------------------------------------------------------------------------------------[0m

#     Description `This image is a detailed technical document, likely from a research paper or academic publication, featuring a prominent diagram of the Transformer model architecture alongside explanatory text.

#   

#   **Overall Composition & Scenery:**

#   The image is set against a clean white background. The top half is dominated by a multi-colored block diagram, while the bottom half contains black text organized into sections and paragraphs. A page number "3" is centered at the very bottom.

#   

#   **Objects & Diagram Elements:**

#   

#   *   **Main Diagram:** Titled "Figure 1: The Transformer - model architecture," it is a flowchart or block diagram illustrating a neural network architecture. It's broadly divided into two main vertical stacks: an **Encoder** on the left and a **Decoder** on the right.

#   *   **Encoder (Left Stack):**

#       *   Starts with "Inputs" at the bottom, receiving combined data from a pink "Input Embedding" rectangular block and a circular "Positional Encoding" icon.

#       *   Above the input, a vertica...`

#   [93m----------------------------------------------------------------------------------------------------[0m

#   <IPython.core.display.Image object>

"""
...and another example:
"""

query = "a kitten with a robot"
results = search_multimodal(query, vector_index, top_k=1)

if not results:
    pretty_print.wrapped("❌ No results found", title="❌")
else:
    result = results[0]

    pretty_print.wrapped(
        [
            f"Similarity {result['similarity']:.3f}",
            f"Filename {result['filename']}",
            f"Description `{result['description'][:1000]}...`",
        ],
        title=f"Results for query = {query}",
    )
    display_image(Path(result["filename"]))
# Output:
#   

#   🔍 Embedding query: 'a kitten with a robot'

#   ✅ Query embedded successfully

#   [93m---------------------------- Results for query = a kitten with a robot ----------------------------[0m

#     Similarity 0.811

#   [93m----------------------------------------------------------------------------------------------------[0m

#     Filename images/image_1.jpeg

#   [93m----------------------------------------------------------------------------------------------------[0m

#     Description `This image is a detailed, photorealistic digital rendering or illustration depicting an unlikely interaction between a large, imposing robot and a small, delicate kitten in an industrial setting.

#   

#   **Objects:**

#   *   **Robot:** The dominant figure is a large, humanoid robot, occupying the right side of the frame. Its body is constructed from dark, metallic armored plates in shades of charcoal, gunmetal, and dark grey, with visible bolts, rivets, and segmented joints suggesting a heavy, industrial design.

#       *   **Head/Face:** The robot's head is highly detailed, featuring intricate circuit board patterns or etched lines across its dark surface, implying advanced technology or artificial intelligence. Its most striking feature is its eyes, which are large, glowing red lights, casting a subtle red ambient glow. The face design is angular and segmented, reminiscent of a protective helmet or mask, with no visible mouth.

#       *   **Body:** Parts of its robust shoulder, upper arm, and a large, ...`

#   [93m----------------------------------------------------------------------------------------------------[0m

#   <IPython.core.display.Image object>

"""
## 4. Building multimodal AI agents
"""

"""
The last step is to hook our RAG `search_multimodal` function to a ReAct agent to create an agentic RAG system.

First, we define the `multimodal_search_tool` using LangGraph:
"""

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent


@tool
def multimodal_search_tool(query: str) -> dict[str, Any]:
    """
    Search through a collection of images and their text descriptions to find relevant content.

    This tool searches through a pre-indexed collection of image-text pairs using the query
    and returns the most relevant match. The search uses multimodal embeddings to find
    semantic matches between the query and the content.

    Args:
        query: Text query describing what to search for (e.g., "cat", "kitten with robot")

    Returns:
        A formatted string containing the search result with description and similarity score
    """

    pretty_print.wrapped(query, title="🔍 Tool executing search for:")

    results = search_multimodal(query, vector_index, top_k=1)

    if not results:
        return {"role": "tool_result", "content": "No relevant content found for your query."}
    else:
        pretty_print.wrapped(str(results[0]["filename"]), title="🔍 Found results:")
    result = results[0]

    content = [
        {
            "type": "text",
            "text": f"Image description: {result['description']}",
        },
        types.Part.from_bytes(
            data=result["content"],
            mime_type="image/jpeg",
        ),
    ]

    return {
        "role": "tool_result",
        "content": content,
    }

"""
Next, we create a ReAct agent using LangGraph's `create_react_agent` function and the RAG tool defined above:
"""

def build_react_agent() -> Any:
    """
    Build a ReAct agent with multimodal search capabilities.

    This function creates a LangGraph ReAct agent that can search through images
    and text using the multimodal_search_tool. The agent uses Gemini 2.5 Pro
    for reasoning and tool execution.

    Returns:
        Any: A LangGraph ReAct agent instance configured with multimodal search tools
    """

    tools = [multimodal_search_tool]

    system_prompt = """You are a helpful AI assistant that can search through images and text to answer questions.
    
    When asked about visual content like animals, objects, or scenes:
    1. Use the multimodal_search_tool to find relevant images and descriptions
    2. Carefully analyze the image or image descriptions from the search results
    3. Look for specific details like colors, features, objects, or characteristics
    4. Provide a clear, direct answer based on the search results
    5. If you can't find the specific information requested, be honest about

[... Content truncated due to length ...]

</details>


## YouTube Video Transcripts

<details>
<summary>An enriched transcript of the video:</summary>

An enriched transcript of the video:

Although AI research is traditionally split into distinct fields like NLP and computer vision, countless real-world problems require solutions that integrate information across these modalities. In this video, I'll discuss how we can solve such problems using multimodal embeddings. (A graphic appears with the text "Multimodal Embeddings"). Then show how to use them to do things like zero-shot image classification (A graphic shows a banana image being correctly classified as "Banana" from a list) and image search. (A second graphic shows an image search for "Papaya" returning relevant images). And if you're new here, welcome. I'm Shaw. I make videos about the things I'm learning about and building in AI. And if you enjoyed this content, please consider clicking the subscribe button. That's a great no-cost way you can support me in all the videos that I make.

*The speaker introduces the topic of multimodal embeddings, which integrate information across different data modalities like text and images for tasks such as classification and search.*

[00:44]
(A slide titled "Multimodal Embeddings" appears with the subtitle "An introduction with example code (ft. CLIP)". A diagram shows icons for a book (text) and a picture (image) feeding into a block labeled "CLIP", which outputs two vectors representing text and image embeddings.)
Here we're going to talk about multimodal embeddings. Although the discussion here will focus around CLIP, which works with text and image data, this is a much more general idea that can be extended to many other modalities.

*Multimodal embeddings, exemplified by CLIP for text and images, represent a general concept applicable to various data types.*

[01:01]
(The slide changes to a new one titled "What are Embeddings?" with the subtitle "Useful numerical representations of data learned via model training".)
Before talking about multimodal embeddings, it's worth answering the question, "What are embeddings?" The way I'll define embeddings here are useful numerical representations of data learned through model training. A classic example of this is BERT, which is a popular language model before the era of GPT-3 and all the modern large language models. BERT used to be state of the art, and one of the things that it does is masked language modeling. In other words, you can give it a sequence of text where one of the tokens in that sequence is masked, meaning that it's not visible, and BERT will predict the most likely token that goes in the place of that mask. So if you pass in the sequence, "Listen to your [MASK]", the most likely token that goes in the sequence is "instincts." So it turns out that through learning how to do this prediction, BERT learns useful representations of text which can be extended to other NLP tasks.

[02:08]
(An animation on the slide shows the BERT model. An input "Listen to your [MASK]." goes in, and the output "Listen to your instincts." comes out. Below the BERT model, an arrow points down with the text "Drop head". This leads to a mutilated version of the BERT model, which now takes "Listen to your instincts." as input and outputs an "n x d" matrix.)
The basic idea here is that you'll take BERT and you'll drop its head, so the classification head, which is doing this masked language modeling. And you'll have this mutilated version of BERT, which instead of doing this token prediction, it'll take an input sequence of text and return a numerical representation of it, where each row in this matrix corresponds to each token in this text sequence. And then each of these columns corresponds to the embedding dimension, the dimension of this internal representation of text that BERT uses in order to do masked language modeling. We can go one step further and go from token-level representations to sequence-level representations. So we could do something like take the average across all these tokens in the sequence, and we're left with a one-by-d matrix, which represents the entire sequence. (The n x d matrix is shown transforming into a 1 x d vector). And of course, to get these embeddings to be a bit more practical, people will often do additional fine-tuning on top of these embeddings. But this is the basic idea of where they are coming from.

[03:20]
(The slide changes. On the left is a 2D plot titled "Text Embeddings". Various text phrases like "A cute puppy", "A good boy", and "A cute cat" are plotted as points, with similar phrases clustered together.)
A key point about embeddings is that these aren't just arbitrary numerical representations. They are typically semantically meaningful, such that if we were to look at how text were organized in this embedding space, similar concepts would tend to be located close together while dissimilar concepts will tend to be far apart. For example, the sequence "a cute puppy" might be relatively close to the sequence "a good boy," while that same sequence "a cute puppy" might be relatively far from a sequence like "funny cat meme."

[03:53]
(A second 2D plot appears on the right titled "Image Embeddings". It shows various images of cats, dogs, and a goat plotted as points, with images of cats clustered in one area and dogs in another.)
However, this isn't limited to just text. We can generate embeddings for any type of data. Another popular type of data we might work with are images. So if we had some image embeddings, the space might be structured like this, where we tend to have cats in the top left part, the dogs tend to be in the bottom right part, and then we have a goat further away from these. Although text embeddings and image embeddings are super helpful in that they can be adapted and repurposed to solve other, either NLP tasks or computer vision tasks, one major limitation here is that any random text embedding space we might be working with and any random image embedding space we might be working in don't have any relationship to one another. There's no way out of the box to directly map this text embedding space to this image embedding space and vice versa, even if they are semantically meaningful in of themselves. (Text appears at the bottom: "Text and image embedding spaces are not aligned!"). And that's something we can plainly see here in that the text and image embedding spaces are not aligned because for this text embedding space, the puppies tend to be in the top right, the cats tend to be at the bottom, while in our image embedding space, the cats tend to be up here and the dogs tend to be down here.

*Embeddings are semantically meaningful numerical representations of data, but different modalities like text and images exist in separate, unaligned embedding spaces, making direct comparison difficult.*

[05:04]
(The slide changes to one titled "Multimodal Embeddings" with the subtitle "Embeddings which align representations of different data modalities". A 2D plot shows both images and their corresponding text descriptions ("a cute cat", "a cute puppy") clustered together.)
But what if there was a way we could merge these two embedding spaces together? That's exactly the key idea behind multimodal embeddings, which are embeddings which align representations of different data modalities. And the basic intuition here is that if we had a multimodal embedding space, we could represent text and images in the same vector space. So now, indeed, text like "a cute puppy" will be close to images of cute puppies. The text "a cute cat" will be close to images of a cute cat, and the same thing will hold for other concepts.

[05:39]
(On the same slide, audio speaker icons are added to the plot next to the corresponding animal images and text.)
However, this idea is not limited to just images and text. We could just as easily embed audio and images together. Maybe this is an audio file that is a cat meowing, this is a goat making goat noises, we have a puppy with a cute bark, and then maybe we have like a funny shrieking sound associated with this cat meme here. (On the same slide, brain scan images are added to the plot). Another application of this is aligning representations of brain signals with images and text. What this means is if we were to record someone's brain activity and then represent it in this embedding space, we could, in principle, decode the brain information to generate images and text. So in essence, reading people's thoughts. And actually, in reference number four, they are aiming to do exactly this with large language models.

*Multimodal embeddings solve the alignment problem by creating a shared vector space where different data types—like text, images, audio, or even brain signals—can be represented and compared based on semantic similarity.*

[06:55]
(The slide changes to "Contrastive Learning" with the subtitle "Learning approach that seeks to represent different views of the same information similarly".)
Intuitively, this idea of multimodal embeddings is pretty simple to understand. We have this embedding space, which is agnostic to modality, so it doesn't matter if it's an image of a cat, a text description of a cat, or the brain signals of someone looking at a picture of a cat, these numerical representations will be relatively similar. But how do we create these aligned numerical representations? In other words, how does this work under the hood? So the key technique behind creating these types of embeddings is contrastive learning, which is an approach that seeks to represent different views of the same underlying information similarly.

[07:12]
(The text "Positive Pairs" and "Negative Pairs" appears on the slide.)
And the way this works is that we'll train a model on two things. One, positive pairs of data, and two, negative pairs. So in the case of aligning image and text representations, positive pairs might be a picture of a cute cat and a textual caption for this image. And then we might have the text and an image of a cute puppy, and then we might have the same thing for a baby goat. On the other hand, negative captions might look something like this, where you have the image of a cat, but the caption is "a cute puppy", image of a puppy and the caption is a goat, and you have a goat and the caption is a cat. (Examples of positive and negative image-text pairs are shown). So the intuition here is that we train a model to maximize the similarity between these positive pairs and minimize the similarity between these negative pairs. That's the key intuition. In the following slides, we're going to go one level deeper and look at the loss function and the math behind how this is accomplished. If you don't care about the math and how this is working under the hood, feel free to skip ahead to the example code. But if you're interested in the math, we're about to jump right into it.

*Contrastive learning creates multimodal embeddings by training a model to maximize the similarity between corresponding data pairs (e.g., an image and its caption) while minimizing similarity between non-corresponding pairs.*

[08:17]
(The slide shows three images on the left: a cat, a dog, and a goat. Each image is shown being converted into a vector representation. These vectors are then combined into an n x d matrix labeled Ie.)
The way we can use contrastive learning to align image and text representations is we can take images, generate image embeddings using any image encoder. So basically we take our images and generate a single numerical representation for them. And then we can take all these image embeddings and we can concatenate them into a matrix that I'll call I_e. So this will be an n x d matrix where n is the number of images. So if you have three images here, it'll be one, two, three. And then d, so the number of columns will be the embedding dimension. (The slide now shows the corresponding text captions on the right, also being converted into vectors and combined into a matrix Te.) Then we can do a similar thing for text, so we can get a text encoder from off the shelf. We can generate these text embeddings, and then we can concatenate them into a matrix that I'll call T_e, and that will have the same shape. So we'll have n captions, and then they'll have some embedding dimension d. Just to point out here that the representations that we would put into these matrices won't directly come from an image encoder and a text encoder. Instead, these will be multiplied by some learnable weight matrix and then normalized before being organized in this matrix. So that weight matrix that we multiply the original embeddings by are the learnable parameters.

[09:37]
(A new slide shows the formula for logits, calculated as the similarity between an image embedding and a text embedding, divided by a temperature parameter τ.)
Once we have these matrices I and T, we can construct this logits matrix. Basically what that means is we're going to take each image in our image embedding matrix and then each text sequence in our text embedding matrix and we're going to compute their similarity. Typically this is just a cosine similarity, so you do the dot product between these two matrices and then you'll divide it by a temperature parameter. That's what this tau parameter is representing. And the reason we call them logits is because at some point it's going to be the argument in an exponential function, and we'll see that in a little bit here. (The slide now shows a 3x3 grid representing the logits matrix, with images as rows and text captions as columns.) So taking just those three examples from the previous slide, the similarity between the first image and the first text sequence will be in this 1,1 position of the matrix. And then the similarity between this cat image and the sequence "a cute puppy" will be represented by this value here, and the similarity between this cat image and the text sequence "a cute baby goat" will be represented by this value here, and so on and so forth. Just looking at this, we can see that what we want is to make the logits on the diagonal of this matrix as big as possible. So in other words, we want to maximize the similarity between the positive pairs and we want to minimize the off-diagonal elements, which correspond to negative pairs.

[11:02]
(An equation for "Contrastive Loss (Images)" appears, showing a negative log of a fraction. The numerator is the exponentiated logit for a positive pair, and the denominator is the sum of exponentiated logits for that image against all text captions.)
One way we can do this is via the contrastive loss. So this might take slightly different forms depending on the context or the paper that you're reading. But here I'm going to follow what was done in developing CLIP, which is reference number three here. And so basically one way we can achieve this goal of maximizing the similarity of these on-diagonal elements and minimizing the similarity between these off-diagonal elements is via this equation here, which is basically saying for the i-th image, so let's say this cat image here, we want the numerator to be as big as possible, so the numerator will be the i,i element. So this will be either 1,1 or 2,2 or 3,3. And then we want the denominator to be as small as possible. So if the numerator is big, the denominator is small, that means this fraction becomes big. And then if we take the log of that, we'll still have a big number. And then we want this number to be as big as possible because the goal of training is to minimize the loss. And then if this number is big and we have a minus sign next to it, then this will be as minimal as possible. That was probably a bit abstract, so let's walk through this step by step. Let's look at just the first image first. With this notation, I call the loss associated with the first image L1. This will consist of taking this 1,1 logit and then summing over all the logits in this first row. (The first row of the logits matrix is highlighted). So we're basically taking this image and comparing it to every single caption. Then we do the same thing for the second image. We have the positive pair similarity here, and then we sum over all the logits in this row. And then we do a similar thing for image number three. So we look at the positive pair similarity, and then we sum over all the logits or similarities in this row. We can do this for every single image in our batch, or even in our whole training data set. And then we can aggregate them to get the final contrastive loss. What that'll look like is we'll take the loss according to the first image, the loss according to the second image, and the loss corresponding to the third image, and then we can just take their average, and that'll give us the contrastive loss for the images.

[13:15]
(A new equation appears for "Contrastive Loss (Text)", which is symmetric to the image loss but iterates over text captions and compares them against all images.)
But we can do this exact same thing for text. This is how I'm notating contrastive loss for the text. I've switched the index here from J to I, and then I've changed the summation here to I. I feel this notation might be a bit too subtle, but hopefully explaining it step by step, it makes sense what I mean here. So let's see what this looks like for the first text sequence. We're going to be evaluating "a cute cat," so we'll look at logits 1,1 here, and then we'll sum over the logits in this first column. Then we'll do the same thing for this second text sequence, "a cute puppy," and we'll sum over all the logits in this column, and then finally we do it for the final text sequence. It's important to note here that generally this logits matrix is asymmetric because the similarity between the text "a cute puppy" and this image of a cat is in general different than the similarity between this image of a puppy and the text sequence "a cute cat." That's an important thing to note here, and that's the reason why we go through this whole procedure for the images and the text sequences separately. And then we can aggregate the loss over all the text examples just like we did for the images like this. And then we'll get a total text loss by taking the average of all the examples in our minibatch.

[14:35]
(A slide shows the Final Loss equation: L = (LI + LT) / 2. A more complex equation expands this out, showing the Image term and the Text term.)
We can then combine the image loss and text loss together by taking their average. Then we can write it all out to have this big monstrosity all on one page. But basically, this first term here corresponds to the image loss. This second term here corresponds to the text loss. And this is how we train the weights which translate the raw image and text encodings into our multimodal embedding space. This will give us a training signal which we can use to update the weights of these projection matrices and just keep doing that until we're satisfied with the loss.

*The contrastive learning process is mathematically defined by a loss function that averages the image loss and text loss, where each is calculated to maximize similarity for correct pairs and minimize it for incorrect pairs across a batch of data.*

[15:13]
(The slide changes to "Example 1: Using CLIP for 0-shot Image Classification". It shows a hugging face emoji and a Python logo.)
So if that was much more math than you were hoping to get out of this video, I apologize for that, but let's jump to practical applications of multimodal embeddings. Here I'm going to use CLIP for two different use cases. This first use case is zero-shot image classification. The meaning of this is we're going to do image classification without explicitly training CLIP to distinguish between the different image classes that we're considering. The first step is to import transformers. Then I'm going to bring in these two things. And then I'm importing this PIL library, which will allow us to work with images in Python. (The slide shows python code for imports: `from transformers import CLIPProcessor, CLIPModel` and `from PIL import Image`). Next we'll load in the model and the data processor. (The slide shows code for loading the CLIP model and processor from Hugging Face). The image preprocessing is important because images could be any type of size and shape and all that. The CLIP processor is an abstraction that ensures the data are in a suitable format to be passed through the model.

[16:07]
(The slide now shows code to load an image of a cat and define two text classes: "a photo of a cat" and "a photo of a dog". An image of the cute cat appears on the right.)
Next, we're going to load in our image. So I'm going to load in this image of a cute cat, so it's the same one we've seen so far. Then I'm going to define the text classes. So this is a really interesting aspect of using CLIP for zero-shot image classification because before, if you wanted to do image classification, traditionally that was something that was set at model training. It was implicitly coded into the architecture of the model in that you had this classification head, and each value in the output layer corresponded to the probability of class one versus class two versus class three, so on and so forth. But now, when using CLIP, which is trained via contrastive learning, we actually pass the classes as text inputs.

[17:00]
(The next slide shows code for passing the image and text to the CLIP processor and then to the model.)
So with our text and image inputs defined, we can pass these through our processor to put them in the appropriate format for CLIP, and then we can just pass it to the model. Then with this one line of code, we'll generate these outputs. We can extract the logits per image. Recall the logits matrix that we saw a few slides ago where we had an image, and then we had logit values or similarity values between that image and every single piece of text that we passed into the model. That's exactly what we're extracting here. We're extracting the similarity score of the input image to both the text inputs. Then we can convert these logits to probabilities via the softmax.

[17:36]
(The final code snippet on the slide shows how to get the predicted class and print the results. The output shows: ">> a photo of a cat | Probability = 0.9979".)
And then this will give us a prediction. What I'm doing here is I'm just doing argmax of the probabilities tensor, and using that to pick out the predicted class from this original list that I created. And then I'm just going to print everything like this. So I'll print the predicted class as well as a rounded probability corresponding to that class. With that, the most probable class is "a photo of a cat" with an associated probability of 99.79%. So it basically nails the classification of this image.

[18:11]
(Two more classification examples appear on the slide. The first compares classes "ugly cat" vs "cute cat", correctly predicting "cute cat" with 97.03% probability. The second compares "cat meme" vs "not cat meme", predicting "not cat meme" with 54.64% probability.)
But let's see what happens when we use different text classes. Instead of passing in "a photo of a cat" and "a photo of a dog," which are pretty easy classes to distinguish between, let's try something a bit more nuanced, like an "ugly cat" versus "cute cat". And again here, the model basically nails it with a 97% probability of this "cute cat" class. And then we can try something even more challenging, like trying to distinguish if this is a "cat meme" or "not cat meme". And it indeed gets that it's not a cat meme, but we can see that the probability dropped significantly.

[18:45]
(The slide now shows the famous "woman yelling at a cat" meme image.)
Then as a final test of this model, let's see what happens when we pass in an actual cat meme and give it the class choices of "cat meme" versus "not cat meme." And so here, the model again nails it. It correctly classifies this as a cat meme with a probability of 83%. And so again, what we're doing here, using CLIP, is we're taking these three entities. We're taking the text sequence of "cat meme," the text sequence of "not cat meme," and this image of a cat, encoding them in a shared embedding space. And we're evaluating the similarity between this image of a cat and the text sequence "cat meme" and the similarity between this image of a cat and the text sequence "not cat meme." And then we can convert that similarity into a probability as well as a class prediction. The key unlock here is that you are not restricted or limited in the different class labels you can use for image classification. You can be as detailed or vague as you like. You can adapt this to endless different use cases, which is pretty amazing.

*CLIP's multimodal embeddings enable powerful zero-shot image classification by dynamically comparing an image to arbitrary text-based class labels, calculating similarity scores in a shared space to determine the best match without explicit training on those specific classes.*

[19:50]
(The slide changes to "Example 2: Using CLIP for Image Search".)
This second example is basically the inverse of zero-shot image classification. There, we had an input image, and we wanted to match it with one of the input text sequences. Here in example two, we're going to do the exact opposite. So instead of starting with an image, we're going to start with a piece of text, in other words, a search query, and then we're going to match it to a set of images. So essentially, what we're doing is we're doing a search over a set of images.

[20:20]
(The slide shows code for creating a list of images to search over, including a cat, a dog, and a goat. The three images are displayed below the code.)
The way this looks is we'll first load in our images. Here we have a picture of a cute cat, a picture of a dog, and a picture of a goat. We'll store them in this image list using the PIL library to open the images and just store them in this list.

[20:35]
(The next slide shows code for defining a query, "a cute dog", and processing both the text query and the list of images with the CLIP processor.)
Then we're going to define a query and process the inputs. Here our query will be "a cute dog," and then we'll pass this query along with our image list through the processor so it's in the appropriate format for CLIP. Then we'll run these inputs through our model, get these outputs, extract the logits per text now. Before we did logits per image, now we're doing logits per text. So these are going to be the similarity scores between the input text and all the images that we inputted. And then we'll convert these logits into probabilities.

[21:10]
(The final code snippet for this example shows evaluating the best match by finding the argmax of the probabilities and displaying the corresponding image and its match probability. The output shows the dog image with a probability of 0.9817.)
So with that, we can evaluate the best match. So I'm doing that again in a similar way. So we have these probabilities doing argmax, which will give us an integer 0, 1 or 2. We can use that to pick out the best matched image, and we can take the probability associated with that image, and then we're just going to print everything. So again, the query here was "a cute dog," and this is the best matched image with a probability of about 98%. But again, that was a super easy example. (Two more search queries are shown. "something cute but metal" returns the goat image with 77.15% probability. "a good boy" returns the dog image with 82.48% probability.) So let's try a trickier query like "something cute but metal." In this case, the model returns the goat, which is indeed cute, but also goats are associated with heavy metal music, and it got a 77% match probability. Reading this, "a good boy," the text itself doesn't have anything to do with animals. You know, maybe it's a human boy and he's well-behaved. But "a good boy" is a colloquialism for dogs that we use often. And the model can pick that up quite easily. So it matches it with a dog with 82% probability. It would be interesting to see if we threw in a picture of a human boy to see how the model would handle that case. This could be something that you do with the example code from the GitHub.

[22:25]
(A final query is shown: "the best pet in the world". The result is the cat image with a match probability of 0.5664.)
And then we can try an extremely controversial query like "the best pet in the world". For this, the model returns a cat with a 56% match probability. This is likely indicating that on average, people on the internet love cats more than they love dogs. Nevertheless, it's super interesting how we can use this model in order to do search like this.

*Multimodal embeddings also power text-to-image search by taking a text query, embedding it into the shared space, and finding the images with the highest similarity scores, effectively retrieving the most relevant visuals from a collection.*

[22:47]
(The slide changes to a new section titled "What's Next?")
So those were the two examples. Code is on the GitHub, link in the description below. Let's look ahead to the next video of this series. (An animation appears showing two parts. "Part 1" shows a multimodal LLM taking text, image, and audio inputs and producing a text output. "Part 2" shows a multimodal embedding model taking text and image inputs to produce an image output.) In the previous video, so part one, we talked about multimodal large language models. So basically, large language models that can process or generate multiple data modalities. In this video, we talked about multimodal embeddings like those generated by CLIP, which can be used to do things like image search. So we pass in a query and a set of potential images, and then it'll spit out the best matched image.

[23:21]
(A new slide appears titled "What's Next? Multimodal RAG" with a diagram showing a user query "What's there to do in Bali?" going into a retrieval system that uses Multimodal Embeddings (MME) to retrieve relevant text and image context. This context is then used to create a prompt for a Multimodal LLM, which generates a model response.)
In the next video of this series, we're going to bring these two ideas together to create a multimodal RAG system. The basic flow will be to take a user query like "what's there to do in Bali?" We'll pass the query into a multimodal retrieval system, which involves using a multimodal embedding model to pick out the documents and images that are most relevant to this query. We'll take the user query and relevant documents and images to generate a prompt, and then we'll pass that prompt into a multimodal large language model, which can process the user query, relevant text documents, and relevant images to generate a helpful response.

[24:10]
(The slide changes to show a screenshot of a blog post titled "Multimodal Embeddings: An introduction".)
And as a final note, if you enjoyed this video and you want to learn more, check out the blog published in Towards Data Science. There I went into some details that I probably missed here. And as always, even though this is going to be a member-only story, you can access it completely for free using the friend link in the description below. And with that, thank you so much for your time and thanks for watching. (A final screen appears with a logo and the text "Thanks for watching.")

*The next step in this series will combine multimodal large language models and multimodal embeddings to build a multimodal Retrieval-Augmented Generation (RAG) system, which can answer user queries by retrieving and processing relevant text and images.*

</details>


## Additional Sources Scraped

<details>
<summary>arxiv-org</summary>

Documents are visually rich structures that convey information through text, but also figures, page layouts, tables, or even fonts. Since modern retrieval systems mainly rely on the textual information they extract from document pages to index documents -often through lengthy and brittle processes-, they struggle to exploit key visual cues efficiently. This limits their capabilities in many practical document retrieval applications such as Retrieval Augmented Generation (RAG). To benchmark current systems on visually rich document retrieval, we introduce the Visual Document Retrieval Benchmark ViDoRe, composed of various page-level retrieval tasks spanning multiple domains, languages, and practical settings. The inherent complexity and performance shortcomings of modern systems motivate a new concept; doing document retrieval by directly embedding the images of the document pages. We release ColPali, a Vision Language Model trained to produce high-quality multi-vector embeddings from images of document pages. Combined with a late interaction matching mechanism, ColPali largely outperforms modern document retrieval pipelines while being drastically simpler, faster and end-to-end trainable. We release models, data, code and benchmarks under open licenses at [https://hf.co/vidore](https://hf.co/vidore).

# 1 INTRODUCTION

Document Retrieval consists of matching a user query to relevant documents in a given corpus. It is central to many widespread industrial applications, either as a standalone ranking system (search engines) or as part of more complex information extraction or Retrieval Augmented Generation (RAG) pipelines.

Over recent years, pretrained language models have enabled large improvements in text embedding models. In practical industrial settings, however, the primary performance bottleneck for efficient document retrieval stems not from embedding model performance but from the prior data ingestion pipeline. Indexing a standard PDF document involves several steps. First, PDF parsers or Optical Character Recognition (OCR) systems are used to extract words from the pages. Document layout detection models can then be run to segment paragraphs, titles, and other page objects such as tables, figures, and headers. A chunking strategy is then defined to group text passages with some semantical coherence, and modern retrieval setups may even integrate a captioning step to describe visually rich elements in a natural language form, more suitable for embedding models. In our experiments (Table 2), we typically find that optimizing the ingestion pipeline yields much better performance on visually rich document retrieval than optimizing the text embedding model.https://arxiv.org/pdf/images/391f32efa12ee5d8c95be1f641c318cd9e821711f7d2f28610e9c54a25d13db6.jpg

Figure 1: ColPali simplifies document retrieval w.r.t. standard retrieval methods while achieving stronger performances with better latencies. Latencies and results are detailed in section 5 and subsection B.4.

We propose a novel concept and model architecture based on Vision Language Models (VLMs) to efficiently index documents purely from their visual features, allowing for subsequent fast query matching with late interaction mechanisms (Khattab & Zaharia, 2020). Our method, ColPali, significantly outperforms all other retrieval systems on $V i D o R e$ while being fast and end-to-end trainable. These results demonstrate the potential and the many benefits of this novel Retrieval in Vision Space concept, which could significantly alter the way document retrieval is approached in the industry moving forward. We release all resources at [https://hf.co/vidore](https://hf.co/vidore).

# 2 PROBLEM FORMULATION & RELATED WORK

Problem Setting. In our setting, a retrieval system scores how relevant a document $d$ from corpus $\\mathcal { D }$ is with respect to a query $q$ . Computing the similarity score $s ( q , d ) \\in \\mathbb { R }$ for each of the $\| \\mathcal { D } \|$ documents in the corpus creates a ranking we can use to extract the most relevant documents. In this work, we focus on page-level retrieval: given a query, is the correct document page retrieved by the system? For coherence with existing literature, we further use the term document to refer to individual pages, i.e. the atomic retrieved elements in our setting. As we focus on practical industrial retrieval applications (RAG, search engines) with potentially large corpora sizes, latency constraints are imposed on scoring systems. Most current retrieval systems can be decomposed into (1) an offline indexation phase in which a document index is built and (2) an online querying phase in which a query is matched to documents from the index and where low latency is vital to the user experience.

Under these industrial constraints, we identify three main properties an efficient document retrieval systems should exhibit: (R1) strong retrieval performance, as measured by standard retrieval metrics; $( R 2 )$ fast online querying, measured through average latencies; (R3) high throughput corpus indexation, ie. the number of pages that can be embedded in a given timeframe.

2.1 TEXTUAL RETRIEVAL METHODS

# Document Retrieval in Text Space.

Statistical methods based on word frequency like TF-IDF (Sparck Jones, 1972) and BM25 (Robertson et al., 1994) are still widely used due to their simplicity and efficiency. More recently, neural embedding models based on fine-tuned large language models display state-of-the-art performance on a variety of text embedding tasks and top the retrieval leaderboards (Muennighoff et al., 2022).

Neural Retrievers. In bi-encoder models (Reimers & Gurevych, 2019; Karpukhin et al., 2020; Wang et al., 2022), documents are independently mapped offline to a dense vector space. Queries are embedded online and matched to documents through a fast cosine distance computation. A slower, but slightly more performant alternative, cross-encoder systems (Wang et al., 2020; Cohere, 2024) concatenate query and document as a single input sequence and iteratively attribute matching scores to each possible combination. This enables full attention computation between query and document terms but comes at the cost of computational efficiency, as $\| \\mathcal D \|$ encoding passes must be done online.

Multi-Vector retrieval via late interaction. In the late interaction paradigm introduced by ColBERT (Khattab & Zaharia, 2020), an embedding is pre-computed and indexed per document token. At runtime, similarity can be computed with individual query token embeddings. The idea is to benefit from the rich interaction between individual query and document terms while taking advantage of the offline computation and fast query matching enabled by bi-encoders. See section E for more details.

While documents often rely on visual elements to more efficiently convey information to human readers, text-only systems barely tap into these visual cues. Other work has also independently studied table or chart retrieval systems through repurposed Question Answering datasets (Zhang et al., 2019; Nowak et al., 2024) but only assessing specialized methods for each task.

# 2.2 INTEGRATING VISUAL FEATURES

Contrastive Vision Language Models. Mapping latent representations of textual content to corresponding representations of visual content has been done by aligning disjoint visual and text encoders through contrastive losses (Radford et al., 2021; Zhai et al., 2023). While some OCR capabilities exist in these models, the visual component is often not optimized for text understanding.

The Fine-grained Interactive Language-Image Pre-training (Yao et al., 2021) framework extends the late interaction mechanism to cross-modal Vision Language Models, relying on max similarity operations between text tokens and image patches.

Visually Rich Document Understanding. To go beyond text, some document-focused models jointly encode text tokens alongside visual or document layout features (Appalaraju et al., 2021; Kim et al., 2021; Huang et al., 2022; Tang et al., 2022). Large Language transformer Models (LLMs) with strong reasoning capabilities have recently been combined with Vision Transformers (ViTs) (Dosovitskiy et al., 2020) to create VLMs (Alayrac et al., 2022; Liu et al., 2023; Bai et al., 2023; Lauren¸con et al., 2024b) where image patch vectors from contrastively trained ViT models (Zhai et al., 2023) are fed as input embeddings to the LLM and concatenated with the text-token embeddings.

PaliGemma. The PaliGemma-3B model (Beyer et al., 2024) extends concepts from Pali3 (Chen et al., 2023), and projects SigLIP-So400m/14 (Alabdulmohsin et al., 2023) patch embeddings into Gemma-2B’s text vector space (Gemma Team et al., 2024). Along with its reasonable size w.r.t. other performant VLMs, an interesting property of PaliGemma’s text model is that it is fine-tuned with full-block attention on the prefix (instruction text and image tokens). See Appendix E for more details.

VLMs display enhanced capabilities in Visual Question Answering, captioning, and document understanding (Yue et al., 2023), but are not optimized for retrieval tasks.

# 3.2 ASSESSING CURRENT SYSTEMS

Unstructured. We evaluate retrieval systems representative of those found in standard industrial RAG pipelines. As is common practice, we rely on the Unstructured4 off-the-shelf tool in the highest resolution settings to construct high-quality text chunks from PDF documents. Unstructured orchestrates the document parsing pipeline, relying on deep learning vision models to detect titles and document layouts (Ge et al., 2021), OCR engines (Smith, 2007) to extract text in non-native PDFs, specialized methods or models to detect and reconstruct tables, and implements a chunking strategy (by-title) that leverages the detected document structure to preserve section boundaries when concatenating texts. As is common practice, in our simplest Unstructured configuration (text-only), only textual elements are kept and figures, images, and tables are considered noisy information and are filtered out.

Unstructured $+ \\textbf { X }$ . While Unstructured is a strong baseline by itself, we further augment Unstructured’s output by integrating the visual elements. In $( + \ O C R )$ , tables, charts, and images are run through an OCR engine, processed by Unstructured, and chunked independently. In ( $+$ Captioning), we set up a fully-fledged captioning strategy (Zhao et al., 2023), in which we feed visual elements to a strong proprietary Vision Language Model (Claude-3 Sonnet (Anthropic, 2024)) to obtain highly detailed textual descriptions of the elements. Both strategies aim to integrate visual elements in the retrieval pipeline but incur significant latency and resource costs (subsection 5.2).

Embedding Model. To embed textual chunks, we evaluate Okapi BM25, the de facto standard sparse statistical retrieval method, and the dense encoder of BGE-M3 (Chen et al., 2024), a multilingual neural method with SOTA performance in its size category. Chunks are embedded and scored independently, and page-level scores are obtained by max-pooling over the page’s chunk scores.5

Contrastive VLMs. We also evaluate the strongest available vision-language embedding models; Jina CLIP (Koukounas et al., 2024), Nomic Embed Vision (Nomic, 2024), and SigLIP-So400m/14 (Alabdulmohsin et al., 2023).

Results. From a performance perspective, best results are obtained by combining the Unstructured parser with visual information, either from captioning strategies or by running OCR on the visual elements (Table 2). Little difference is seen between BM25 and BGE-M3 embeddings highlighting the visual information bottleneck. Contrastive VLMs lag behind. Beyond retrieval performance (R1), the indexing latencies (R2) reported in Figure 2 illustrate that PDF parsing pipelines can be very lengthy, especially when incorporating OCR or captioning strategies. Querying latencies at runtime (R3) are very good for all evaluated systems $\\leq 2 2 \\mathrm { m s }$ on a NVIDIA L4) due to fast query encoding and cosine similarity matching.https://arxiv.org/pdf/images/9b42e697ca163d7e29300ce3893bfbb565058abb5496f1a4433ddbc0420c9cb1.jpg

Figure 2: Offline document indexing with ColPali is much simpler and faster compared to standard retrieval methods. The PDF Parser results are obtained following the Unstructured settings with BGE-M3 detailed in subsection 3.2. All indexing speeds are averaged per-page latencies. More details in subsection B.4

# 4 LATE INTERACTION BASED VISION RETRIEVAL

# 4.1 ARCHITECTURE

Vision-Language Models. Encouraged by their strong document understanding capabilities, we propose adapting recent VLMs for retrieval. The key concept is to leverage the alignment between output embeddings of text and image tokens acquired during multi-modal fine-tuning. To this extent, we introduce ColPali, a Paligemma-3B extension that is capable of generating ColBERT-style multivector representations of text and images (Figure 1). PaliGemma-3B is a strong candidate due to its small size, the many released checkpoints fine-tuned for different image resolutions and tasks, and the promising performances on various document understanding benchmarks. We add a projection layer to map each of the language model’s output token embeddings (whether from text or image tokens) to a vector space of reduced dimension $D = 1 2 8$ as used in the ColBERT paper (Khattab & Zaharia, 2020) to keep lightweight bag-of-embedding representations.

Late Interaction. Given query $q$ and document $d$ , we denote as $\\mathbf { E \_ { q } } \\in \\mathbb { R } ^ { N \_ { q } \\times D }$ and $\\mathbf { E \_ { d } } \\in \\mathbb { R } ^ { N \_ { d } \\times D }$ their respective multi-vector representation in the common embedding space $\\mathbb { R } ^ { D }$ , where $N \_ { q }$ and $N \_ { d }$ are respectively the number of vectors in the query and in the document page embeddings. The late interaction operator, $\\operatorname { L I } \\left( q , d \\right)$ , is the sum over all query vectors $\\mathbf { E \_ { q } } ^ { ( j ) }$ , of its maximum dot product $\\langle \\cdot \| \\cdot \\rangle$ with each of the $N \_ { d }$ document embedding vectors $\\mathbf { E \_ { d \\left( 1 : N \_ { d } \\right) } }$ .

$$
\\mathbf { L } \\left( q , d \\right) = \\sum \_ { i \\in \\left\[ \\left\| 1 , N \_ { q } \\right\| \\right\] } \\operatorname\* { m a x } \_ { j \\in \\left\[ \\left\| 1 , N \_ { d } \\right\| \\right\] } \\langle \\mathbf { E \_ { q } } ^ { ( i ) } \| \\mathbf { E \_ { d } } ^ { ( j ) } \\rangle
$$

Contrastive Loss. The Late Interaction operation is fully differentiable, enabling backpropagation. Let a batch ${ q \_ { k } , d \_ { k } } \_ { k \\in \[ \| 1 , b \| \] }$ composed of $b$ query-page pairs, where for all $k \\in \[ \| \\bar { 1 , \\upsilon } \| \]$ , the document page $d \_ { k }$ is the document corresponding to query $q \_ { k }$ . Following Khattab & Zaharia (2020), we define our in-batch contrastive loss $\\mathcal { L }$ as the softmaxed cross-entropy of the positive scores $s \_ { k } ^ { + } = \\mathrm { L I } \\left( q \_ { k } , d \_ { k } \\right)$ w.r.t. to the maximal in-batch negative scores $s \_ { k } ^ { - } = \\operatorname\* { m a x } \_ { l , l \\neq k } \\quad \\mathrm { L I } ( q \_ { k } , d \_ { l } ) ^ { 6 }$ :

$$
\\mathcal { L } = - \\frac { 1 } { b } \\sum \_ { k = 1 } ^ { b } \\log \\left\[ \\frac { \\exp \\left( s \_ { k } ^ { + } \\right) } { \\exp \\left( s \_ { k } ^ { + } \\right) + \\exp \\left( s \_ { k } ^ { - } \\right) } \\right\] = \\frac { 1 } { b } \\sum \_ { k = 1 } ^ { b } \\log \\left( 1 + \\exp \\left( s \_ { k } ^ { - } - s \_ { k } ^ { + } \\right) \\right)
$$

# 5 RESULTS

Table 2: Comprehensive evaluation of baseline models and our proposed method on ViDoRe. Results are presented using $\\mathrm { n D C G } @ 5$ metrics, and illustrate the impact of different components. Text-only metrics are not computed for benchmarks with only visual elements.

|     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | ArxivQ DocQ |  | InfoQ | TabF | TATQ | Shift | AI | Energy | Gov. | Health. | Avg. |
| Unstructured text-only |  |  |  |  |  |  |  |  |  |  |  |
| \- BM25 |  | 34.1 |  |  | 44.0 | 59.6 | 90.4 | 78.3 | 78.8 | 82.6 |  |
| \- BGE-M3 |  | 28.4↓5.7 |  |  | 36.1↓7.9 | 68.5†8.9 | 88.4↓2.0 | 76.8↓1.5 | 77.7↓1.1 | 84.6↑2.0 |  |
| Unstructured +oCR |  |  |  |  |  |  |  |  |  |  |  |
| \- BM25 | 31.6 | 36.8 | 62.9 | 46.5 | 62.7 | 64.3 | 92.8 | 85.9 | 83.9 | 87.2 | 65.5 |
| \- BGE-M3 | 31.4↓0.2 | 25.7↓11.1 60.1↓2.8 |  | 70.8↑24.3 50.5↓12.2 73.2↑8.9 |  |  | 90.2↓2.6 | 83.6↓2.3 | 84.9↑1.0 | 91.1↑3.9 | 66.1↑0.6 |
| Unstructured + Captioning |  |  |  |  |  |  |  |  |  |  |  |
| \- BM25 | 40.1 | 38.4 | 70.0 | 35.4 | 61.5 | 60.9 | 88.0 | 84.7 | 82.7 | 89.2 | 65.1 |
| \- BGE-M3 | 35.7↓4.4 | 32.9↓5.4 | 71.9†1.9 |  |  |  | 69.1†↑33.7 43.8↓17.7 73.1↑12.2 88.810.8 | 83.3↓1.4 | 80.4↓2.3 | 91.3↑2.1 | 67.0个1.9 |
| ContrastiveVLMs |  |  |  |  |  |  |  |  |  |  |  |
| Jina-CLIP | 25.4 | 11.9 | 35.5 | 20.2 | 3.3 | 3.8 | 15.2 | 19.7 | 21.4 | 20.8 | 17.7 |
| Nomic-vision | 17.1 | 10.7 | 30.1 | 16.3 | 2.7 | 1.1 | 12.9 | 10.9 | 11.4 | 15.7 | 12.9 |
| SigLIP (Vanilla) | 43.2 | 30.3 | 64.1 | 58.1 | 26.2 | 18.7 | 62.5 | 65.7 | 66.1 | 79.1 | 51.4 |
| Ours |  |  |  |  |  |  |  |  |  |  |  |
| SigLIP (vanilla) | 43.2 | 30.3 | 64.1 | 58.1 | 26.2 | 18.7 | 62.5 | 65.7 | 66.1 | 79.1 | 51.4 |
| BiSigLIP (+fine-tuning) |  | 58.5↑15.3 32.9↑2.6 | 70.5↑6.4 | 62.7↑4.6 | 30.5↑4.3 | 26.5↑7.8 | 74.3†11.8 73.7†8.0 |  | 74.2†8.1 | 82.3↑3.2 | 58.6↑7.2 |
| BiPali (+LLM) |  | 56.5↓-2.0 30.0↓-2.9 67.4↓-3.1 |  | 76.9↑14.2 | 33.4↑2.9 |  |  | 43.7↑17.2 71.2↓-3.1 61.9↓-11.7 | 73.8↓-0.4 | 73.6↓-8.8 | 58.8↑0.2 |
| ColPali (+Late Iter.) |  | 79.1↑22.6 54.424.5 81.8个14.4 83.9†7.0 |  |  |  |  | 65.8†32.4 73.2↑29.5 96.2↑25.0 91.0个29.1 |  |  | 92.7†18.9 94.4个20.8 | 81.3†22.5 |

# 5.1 PERFORMANCE (R1)

We show performance is achieved iteratively through the combination of three factors; (1) a carefully crafted task-specific dataset, (2) pairing a pretrained LLM to a vision model to better leverage text semantics from the image, and (3) using multi-vector embeddings rather than a single vector representation to better capture the vast amount of visual information present in a document.

Leveraging Multi-Vector Embeddings through Late Interaction: ColPali. One benefit of inputting image patch embeddings through a language model is that they are natively mapped to a latent space similar to the textual input (query). This enables leveraging the ColBERT strategy to construct one embedding per image patch token, and at inference compute all interactions between text tokens and image patches, resulting in a step-change improvement in performance compared to BiPali. Results in Table 2 show that our ColPali model also largely outperforms the strong baselines based on Unstructured and captioning, as well as all evaluated text-image embedding models. The difference is particularly stark on the more visually complex benchmark tasks, such as InfographicVQA, ArxivQA, and TabFQuAD, respectively representing infographics, figures, and tables. However, text-centric documents are also better retrieved by the ColPali models across all evaluated domains and languages, making our approach the overall best-performing document-retrieval model.

# 5.2 LATENCIES & MEMORY FOOTPRINT

Online Querying. (R2) Logically, querying latencies differ between ColPali and a BGE-M3 embedding model. For BGE, encoding takes about $2 2 ~ \\mathrm { m s }$ for 15 tokens, while encoding a query with ColPali’s language model takes about $\\mathrm { 3 0 ~ m s ^ { 12 } }$ . For smaller corpus sizes, computing the late interaction operation induces marginally small overheads $\\approx 1$ ms per 1000 pages in the corpus), and the cosine similarity computation between bi-encoder vectors is even faster. Optimized late interaction engines (Santhanam et al., 2022; Lee et al., 2023) enable to easily scale corpus sizes to millions of documents with reduced latency degradations.

Offline Indexing. (R3) Standard retrieval methods using bi-encoders represent each chunk as a single vector embedding, which is easy to store and fast to compute. However, processing a PDF to get the different chunks is the most time-consuming part (layout detection, OCR, chunking), and using captioning to handle multimodal data will only exacerbate this already lengthy process. On the other hand, ColPali directly encodes pages from their image representation. Although the model is larger than standard retrieval encoders, skipping the preprocessing allows large speedups at indexing13 (Figure 2). As pages are embedded end-to-end in single forward pass, the VRAM usage depends exclusively on the sequence length (number of patches per image) which is fixed as well, enabling efficient batching strategies to fully leverage hardware acceleration. ColPali also benefits from most LLM efficiency improvements introduced in the ecosystem such as Flash Attention (Dao, 2023).

Storage Footprint. Our method requires storing a vector per image patch, along with 6 extra text tokens “Describe the image” concatenated to image patches. We project each PaliGemma vector to a lower dimensional space $D = 1 2 8 ,$ ) to maximize efficiency, leading to a memory footprint of 257.5 KB per page (subsection B.3). Importantly, the memory footprint of the naive ColBERT indexing strategy can be drastically improved through compression and clustering mechanisms (Santhanam et al., 2022; Clavie´ et al., 2024).

Token pooling. Token pooling (Clavi´e et al., 2024) is a CRUDE-compliant method (document addition/deletion-friendly) that aims amountto reduce the amount of multi-vector embeddings. For ColPali, many image patches share redundant information, e.g. white background patches. By pooling these patches together, we can reduce the amount of embeddings while retaining most information. Retrieval performance with hierarchical mean token pooling on image embeddings is shown in Figure 3 (left). With a pool factor of 3, the total number of vectors is reduced by $6 6 . 7 %$ while $9 7 . 8 %$ of the original performance is maintained. We note that the Shift dataset—composed of the most text-dense documents—is a clear outlier, showcasing more information dense documents contain less redundant patches and may be prone to worse performance degradation with such pooling techniques.https://arxiv.org/pdf/images/8293b990ecdbbef3709cfbff1889d6bbaef5395f6aee03741bd76b28b993209f.jpg

Figure 3: (Left: Token Pooling) Relative performance degradation when reducing the number of stored embeddings per document. (Right: Interpretability) For each term in a user query, ColPali identifies the most relevant document image patches (highlighted zones) and computes a query-topage matching score.

# 5.3 INTERPRETABILITY

By superimposing the late interaction heatmap on top of the original image, we can visualize the most salient image patches with respect to each term of the query, yielding interpretable insights into model focus zones. As epitomized in Figure 3 (right), we observe ColPali exhibits strong OCR capabilities as both the words “hourly” and “hours” present a high similarity score with the query token < hour>. We also note particular focus on other non-trivial image features such as the x-axis representing hours being salient. Other visualization examples are shown in Appendix D.

# 7 CONCLUSIONS

In this work, we introduced the Visual Document Retrieval Benchmark (ViDoRe), which evaluates document retrieval systems in realistic settings involving visually complex documents. We demonstrated that current retrieval pipelines and contrastive vision-language models struggle to efficiently exploit visual information embedded in documents, leading to suboptimal performance. To address this, we presented ColPali, a novel retrieval method that leverages Vision-Language Models to create high-quality, multi-vector embeddings purely from visual document features. ColPali largely outperforms the best existing document retrieval methods while enabling faster corpus indexing times and maintaining low querying latencies, thus circumventing many pain points of modern document retrieval applications. We hope to drive industrial adoption, and to encourage future work by publicly releasing the ViDoRe benchmark, the data, the codebase, and all models and baselines from our work.

</details>

<details>
<summary>complex-document-recognition-ocr-doesn-t-work-and-here-s-how</summary>

# Complex Document Recognition: OCR Doesn’t Work and Here’s How You Fix It

In this article, I will dive into a complex world of complex document recognition using AI and OCR.

Document recognition nowadays is not a complex task.

Modern OCR solutions are able to detect both typed and written text in many languages. One can find dedicated solutions for the detection of specific documents like passports and driver’s licenses.

But where out-of-the-box [AI](https://hackernoon.com/c/ai?ref=hackernoon.com) tends to struggle is when a document includes special symbols or tilted text.

Technical drawings are among the ‘trouble children’ that cause ready-made OCR solutions to struggle: they are nothing but a collection of weird symbols and weirdly placed text.

Having worked on an AI solution for technical drawing recognition, I have insights into the world of modern OCR that I will share in this article.

## Why OCR is bad for OCR

The ‘digital first’ approach, at the forefront of many businesses, has motivated many to convert physical documents into a digital format. This process usually involves the implementation of OCR — optical character recognition — which converts physical documents into PDF files.

Morel OCR tools are capable of recognizing more than just text. In many cases, OCR tools can detect special symbols, written text, signatures, images, and more.

Many of these tools come ready to use: all you need to do is install the tool (or, if you are working on a custom solution, use an API) to scan the documents in question.

Despite all this, OCR tools have certain limitations. They don’t work well for irregular text, also called wild text, like low-quality scanned documents with no predefined structure, car license plates, text on advertisement billboards, etc.

### Low-quality scans

The quality of text recognition depends highly on the quality of the document itself. Warping, scratches, faded ink, and more have a detrimental effect on the recognition quality.

### Symbol mixups

Even the best OCR tools have trouble distinguishing between certain similar-looking letters and numbers, like ‘3’ and ‘8’ or ‘O’ and ‘D.’ The very challenges OCR is supposed to solve often become the stumbling block of document digitization.

### Special symbols

Documents that feature any special symbols, from letters specific to a certain language to symbols denominating certain objects, like symbols used in technical drawings, e.g., diameter ‘Ø,’ square ‘□.’

## AI to the rescue

Using artificial intelligence, OCR tools can be improved and augmented to better handle complex documents, and often even replaced by a custom [AI neural network](https://hackernoon.com/enhancing-neural-network-reasoning-the-promise-of-contrastive-decoding-for-llms?ref=hackernoon.com).

Model-based OCR, or intelligent OCR, is the result of using deep learning for text document recognition.

Neural networks can be trained to recognize text regular OCR tools have trouble with. Intelligent OCR provides superior text recognition results in document recognition applications by improving recognition speed and reducing errors.

## Recognition of complex documents

Despite the widespread digitization, some paperwork remains offline. This usually applies to complex documents that are impossible to digitize due to their complex layouts, the use of special symbols, and unconventional formatting.

Technical drawings are the perfect example of a complex document: their layouts change from one document to another; they include a bunch of symbols specific to technical drawings only, and the text is often formatted in odd ways. All of the above makes technical drawings the perfect candidate for model-based OCR.

While working on a similar project, I’ve developed an understanding of the best strategies to apply when working on digitizing technical drawings. I have had experience with working on an AI for floor plan detection, so that’s what I’ll be using as an example.

I’ve broken the process down into sections, as this is exactly how one should approach the development of AI-based OCR solutions for complex document recognition.

## Stage 1: Detection of text

Recognition of plain text is the most simple part of this entire ordeal. When it comes to technical drawings, plain text is used to specify the drawing type, dimensions, floor plan type, building type, etc. While the detection of plain text is a simple task, detecting text on a technical drawing is far more complex.

The text can come in a variety of fonts, sizes, and colors, can be rotated or upside down, and contains special symbols. Ready-made OCR software like iText and OCRSpace can detect simple text with high accuracy, but they fail spectacularly when it comes to technical drawings (or any other complex document, for that matter). For example, these tools struggle to detect rotated text.https://hackernoon.imgix.net/images/2DFAaGGO5cfymtBKn4bFFAoT6sg2-v993xj8.jpeg?w=1200

OCR tools often have trouble detecting rotated text | Image by author

Most OCR tools can be fine-tuned to handle problematic text better. The best approach to recognizing complex text is to use multiple fine-tuned OCR tools along with a balancer that compares the results of each tool and chooses the one that produces the most accurate results.

Another benefit of using fine-tuned OCR software is the increase in recognition speed.https://hackernoon.imgix.net/images/2DFAaGGO5cfymtBKn4bFFAoT6sg2-v9a3xhv.jpeg?w=1200

Fine-tuning of OCR software leads to better results | Image by author

By fine-tuning these tools alone, we’ve seen a 200 times decrease in document processing speed.If you add an OCR engine into the equation, like Tesseract, the text recognition quality can be increased up to 99.9%.

## Stage 2: Recognition of special symbols

Each technical drawing includes special symbols of some sort. In the case of floor plan technical drawings, the documents include symbols designating doors, windows, electrical outlets, etc.

These symbols, or labels, look like geometric figures with text inside. They can be difficult to distinguish from their surroundings due to their shape, which blends in perfectly with the rest of the drawing.

In addition, there can be multiple labels representing the same object due to inconsistencies in document design.https://hackernoon.imgix.net/images/2DFAaGGO5cfymtBKn4bFFAoT6sg2-efb3xu6.jpeg?w=1200

Similar looking objects are often detected as the same one | Image by author

Pre-trained computer vision solutions, like OpenCV libraries for symbol detection, work best with photographs of real-life objects. Technical drawings are quite a bit different: they are almost always in black and white and mostly consist of geometric shapes.

We’ve tested multiple OpenCV libraries, each of which resulted in albeit different, yet insufficiently low recognition quality. Unless you develop your own neural network from scratch, any pre-trained computer vision model needs to be built upon to achieve decent recognition quality.

One of the main problems with using [pre-trained CV models](https://hackernoon.com/creating-computer-vision-apps-without-building-media-pipelines?ref=hackernoon.com) is the amount of false positive results they produce. Technical drawings consist of simple geometric shapes, but so do special symbols and labels, which results in CV models detecting random parts of the drawings as labels.

The best way of mitigating this issue is to implement deep learning to detect false positive results and remove them from the final detection results.https://hackernoon.imgix.net/images/2DFAaGGO5cfymtBKn4bFFAoT6sg2-mjc3x1z.jpeg?w=1200

Deep learning can be used to remove false positive results | Image by author

## Stage 3: Spreadsheets

Technical drawings often include large spreadsheets with merged cells and complex structures stretching across multiple pages. While spreadsheets are generally easy to detect, the complex nature of these spreadsheets makes them difficult to crack.

Going a custom software route is the best way to achieve satisfactory results. Here’s how we’ve done it:

### Recognition of text in a spreadsheet

Solutions like Amazon Textract work very well and can extract text with very high accuracy as long as the document scan is of high quality. Documents with 300 DPI result in 100% recognition accuracy and 100 DPI results in ~90% accuracy.

### Recognition of spreadsheet structure

First, you need to detect the spreadsheet structure by detecting vertical and horizontal lines.

Using OpenCV, create a binary matrix by converting the document into black and white, defining its threshold in a way that results in all horizontal and vertical lines being one and the rest — a zero. The binary matrix will then contain the spreadsheet structure.

Using the extracted text and spreadsheet structure, the spreadsheet itself can be extracted in an editable format like Excel.

## Summing Up

Digitizing any complex document comes with its own set of problems. The best approach to solving them is to approach them one by one, researching the best tools for the job, testing them, and comparing results.

The approaches I’ve described work on any document type despite its type, as individual challenges can be similar despite the document type being completely different.

For example, I have experience in working on a passport detection solution where the text recognition challenges were very similar, and we’ve used some of the same techniques.

Knowing your OCR tools, being well-versed in coding neural networks and having decent experience in the field of custom AI development will help overcome any document digitization challenges.

</details>

<details>
<summary>google-generative-ai-embeddings-ai-studio-gemini-api-langcha</summary>

Connect to Google's generative AI embeddings service using the `GoogleGenerativeAIEmbeddings` class, found in the [langchain-google-genai](https://pypi.org/project/langchain-google-genai/) package.

This will help you get started with Google's Generative AI embedding models (like Gemini) using LangChain. For detailed documentation on `GoogleGenerativeAIEmbeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/v0.2/api_reference/google_genai/embeddings/langchain_google_genai.embeddings.GoogleGenerativeAIEmbeddings.html).

## Overview [​](https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai/\#overview "Direct link to Overview")

### Integration details [​](https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai/\#integration-details "Direct link to Integration details")

| Provider | Package |
| --- | --- |
| [Google Gemini](https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai) | [langchain-google-genai](https://python.langchain.com/api_reference/google_genai/embeddings/langchain_google_genai.embeddings.GoogleGenerativeAIEmbeddings.html) |

## Setup [​](https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai/\#setup "Direct link to Setup")

To access Google Generative AI embedding models you'll need to create a Google Cloud project, enable the Generative Language API, get an API key, and install the `langchain-google-genai` integration package.

### Credentials [​](https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai/\#credentials "Direct link to Credentials")

To use Google Generative AI models, you must have an API key. You can create one in Google AI Studio. See the [Google documentation](https://ai.google.dev/gemini-api/docs/api-key) for instructions.

Once you have a key, set it as an environment variable `GOOGLE_API_KEY`:

```codeBlockLines_e6Vv
import getpass
import os

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")

```

To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:

```codeBlockLines_e6Vv
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")

```

## Installation [​](https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai/\#installation "Direct link to Installation")

```codeBlockLines_e6Vv
%pip install --upgrade --quiet  langchain-google-genai

```

## Usage [​](https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai/\#usage "Direct link to Usage")

```codeBlockLines_e6Vv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector = embeddings.embed_query("hello, world!")
vector[:5]

```

```codeBlockLines_e6Vv
[-0.024917153641581535,\
 0.012005362659692764,\
 -0.003886754624545574,\
 -0.05774897709488869,\
 0.0020742062479257584]

```

## Batch [​](https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai/\#batch "Direct link to Batch")

You can also embed multiple strings at once for a processing speedup:

```codeBlockLines_e6Vv
vectors = embeddings.embed_documents(
    [\
        "Today is Monday",\
        "Today is Tuesday",\
        "Today is April Fools day",\
    ]
)
len(vectors), len(vectors[0])

```

```codeBlockLines_e6Vv
(3, 3072)

```

## Indexing and Retrieval [​](https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai/\#indexing-and-retrieval "Direct link to Indexing and Retrieval")

Embedding models are often used in retrieval-augmented generation (RAG) flows, both as part of indexing data as well as later retrieving it. For more detailed instructions, please see our [RAG tutorials](https://python.langchain.com/docs/tutorials/rag/).

Below, see how to index and retrieve data using the `embeddings` object we initialized above. In this example, we will index and retrieve a sample document in the `InMemoryVectorStore`.

```codeBlockLines_e6Vv
# Create a vector store with a sample text
from langchain_core.vectorstores import InMemoryVectorStore

text = "LangChain is the framework for building context-aware reasoning applications"

vectorstore = InMemoryVectorStore.from_texts(
    [text],
    embedding=embeddings,
)

# Use the vectorstore as a retriever
retriever = vectorstore.as_retriever()

# Retrieve the most similar text
retrieved_documents = retriever.invoke("What is LangChain?")

# show the retrieved document's content
retrieved_documents[0].page_content

```

**API Reference:** [InMemoryVectorStore](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.in_memory.InMemoryVectorStore.html)

```codeBlockLines_e6Vv
'LangChain is the framework for building context-aware reasoning applications'

```

## Task type [​](https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai/\#task-type "Direct link to Task type")

`GoogleGenerativeAIEmbeddings` optionally support a `task_type`, which currently must be one of:

- `SEMANTIC_SIMILARITY`: Used to generate embeddings that are optimized to assess text similarity.
- `CLASSIFICATION`: Used to generate embeddings that are optimized to classify texts according to preset labels.
- `CLUSTERING`: Used to generate embeddings that are optimized to cluster texts based on their similarities.
- `RETRIEVAL_DOCUMENT`, `RETRIEVAL_QUERY`, `QUESTION_ANSWERING`, and `FACT_VERIFICATION`: Used to generate embeddings that are optimized for document search or information retrieval.
- `CODE_RETRIEVAL_QUERY`: Used to retrieve a code block based on a natural language query, such as sort an array or reverse a linked list. Embeddings of the code blocks are computed using `RETRIEVAL_DOCUMENT`.

By default, we use `RETRIEVAL_DOCUMENT` in the `embed_documents` method and `RETRIEVAL_QUERY` in the `embed_query` method. If you provide a task type, we will use that for all methods.

```codeBlockLines_e6Vv
%pip install --upgrade --quiet  matplotlib scikit-learn

```

```codeBlockLines_e6Vv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

query_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", task_type="RETRIEVAL_QUERY"
)
doc_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", task_type="RETRIEVAL_DOCUMENT"
)

q_embed = query_embeddings.embed_query("What is the capital of France?")
d_embed = doc_embeddings.embed_documents(
    ["The capital of France is Paris.", "Philipp is likes to eat pizza."]
)

for i, d in enumerate(d_embed):
    print(f"Document {i + 1}:")
    print(f"Cosine similarity with query: {cosine_similarity([q_embed], [d])[0][0]}")
    print("---")

```

```codeBlockLines_e6Vv
Document 1
Cosine similarity with query: 0.7892893360164779
---
Document 2
Cosine similarity with query: 0.5438283285204146
---

```

## API Reference [​](https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai/\#api-reference "Direct link to API Reference")

For detailed documentation on `GoogleGenerativeAIEmbeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/api_reference/google_genai/embeddings/langchain_google_genai.embeddings.GoogleGenerativeAIEmbeddings.html).

## Additional Configuration [​](https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai/\#additional-configuration "Direct link to Additional Configuration")

You can pass the following parameters to ChatGoogleGenerativeAI in order to customize the SDK's behavior:

- `client_options`: [Client Options](https://googleapis.dev/python/google-api-core/latest/client_options.html#module-google.api_core.client_options) to pass to the Google API Client, such as a custom `client_options["api_endpoint"]`
- `transport`: The transport method to use, such as `rest`, `grpc`, or `grpc_asyncio`.

</details>

<details>
<summary>image-understanding-gemini-api-google-ai-for-developers</summary>

Gemini models are built to be multimodal from the ground up, unlocking a wide range of image processing and computer vision tasks including but not limited to image captioning, classification, and visual question answering without having to train specialized ML models.

## Passing images to Gemini

You can provide images as input to Gemini using two methods:

- [Passing inline image data](#inline-image): Ideal for smaller files (total request
size less than 20MB, including prompts).
- [Uploading images using the File API](#upload-image): Recommended for larger files or for
reusing images across multiple requests.

### Passing inline image data

You can pass inline image data in the
request to `generateContent`. You can provide image data as Base64 encoded
strings or by reading local files directly (depending on the language).

The following example shows how to read an image from a local file and pass
it to `generateContent` API for processing.

```
  from google.genai import types

  with open('path/to/small-sample.jpg', 'rb') as f:
      image_bytes = f.read()

  response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[\
      types.Part.from_bytes(\
        data=image_bytes,\
        mime_type='image/jpeg',\
      ),\
      'Caption this image.'\
    ]
  )

  print(response.text)

```

You can also fetch an image from a URL, convert it to bytes, and pass it to
`generateContent` as shown in the following examples.

```
from google import genai
from google.genai import types

import requests

image_path = "https://goo.gle/instrument-img"
image_bytes = requests.get(image_path).content
image = types.Part.from_bytes(
  data=image_bytes, mime_type="image/jpeg"
)

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=["What is this image?", image],
)

print(response.text)

```

### Uploading images using the File API

For large files or to be able to use the same image file repeatedly, use the
Files API. The following code uploads an image file and then uses the file in a
call to `generateContent`. See the [Files API guide](https://ai.google.dev/gemini-api/docs/files) for
more information and examples.

```
from google import genai

client = genai.Client()

my_file = client.files.upload(file="path/to/sample.jpg")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[my_file, "Caption this image."],
)

print(response.text)

```

## Prompting with multiple images

You can provide multiple images in a single prompt by including multiple image
`Part` objects in the `contents` array. These can be a mix of inline data
(local files or URLs) and File API references.

```
from google import genai
from google.genai import types

client = genai.Client()

# Upload the first image
image1_path = "path/to/image1.jpg"
uploaded_file = client.files.upload(file=image1_path)

# Prepare the second image as inline data
image2_path = "path/to/image2.png"
with open(image2_path, 'rb') as f:
    img2_bytes = f.read()

# Create the prompt with text and multiple images
response = client.models.generate_content(

    model="gemini-2.5-flash",
    contents=[\
        "What is different between these two images?",\
        uploaded_file,  # Use the uploaded file reference\
        types.Part.from_bytes(\
            data=img2_bytes,\
            mime_type='image/png'\
        )\
    ]
)

print(response.text)

```

## Object detection

From Gemini 2.0 onwards, models are further trained to detect objects in an
image and get their bounding box coordinates. The coordinates, relative to image
dimensions, scale to \[0, 1000\]. You need to descale these coordinates based on
your original image size.

```
from google import genai
from google.genai import types
from PIL import Image
import json

client = genai.Client()
prompt = "Detect the all of the prominent items in the image. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000."

image = Image.open("/path/to/image.png")

config = types.GenerateContentConfig(
  response_mime_type="application/json"
  )

response = client.models.generate_content(model="gemini-2.5-flash",
                                          contents=[image, prompt],
                                          config=config
                                          )

width, height = image.size
bounding_boxes = json.loads(response.text)

converted_bounding_boxes = []
for bounding_box in bounding_boxes:
    abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
    abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
    abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
    abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)
    converted_bounding_boxes.append([abs_x1, abs_y1, abs_x2, abs_y2])

print("Image size: ", width, height)
print("Bounding boxes:", converted_bounding_boxes)

```

## Supported image formats

Gemini supports the following image format MIME types:

- PNG - `image/png`
- JPEG - `image/jpeg`
- WEBP - `image/webp`
- HEIC - `image/heic`
- HEIF - `image/heif`

</details>

<details>
<summary>multi-modal-ml-with-openai-s-clip-pinecone</summary>

Language models (LMs) can not rely on language alone. That is the idea behind the “Experience Grounds Language” paper, that proposes a framework to measure LMs' current and future progress. A key idea is that, beyond a certain threshold LMs need other forms of data, such as visual input \[1\] \[2\].https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F25e7f2f54b543af8c34c143448a4b0c55f77c6b5-2360x854.png&w=3840&q=75

World Scopes (WS), as datasets become larger in scope and span multiple modalities, the capabilities of models trained with them increase.

The next step beyond well-known language models; BERT, GPT-3, and T5 is _”World Scope 3”_. In World Scope 3, we move from large text-only datasets to large multi-modal datasets. That is, datasets containing information from multiple forms of media, like _both_ images and text.

The world, both digital and real, is multi-modal. We perceive the world as an orchestra of language, imagery, video, smell, touch, and more. This chaotic ensemble produces an inner state, our “model” of the outside world.

AI must move in the same direction. Even specialist models that focus on language or vision must, at some point, have input from the other modalities. How can a model fully understand the concept of the word “person” without _seeing_ a person?

OpenAI **C** ontrastive **L** earning **I** n **P** retraining (CLIP) is a world scope three model. It can comprehend concepts in both text and image and even connect concepts between the two modalities. In this chapter we will learn about multi-modality, how CLIP works, and how to use CLIP for different use cases like encoding, classification, and object detection.

## Multi-modality

The multi-modal nature of CLIP is powered by two encoder models trained to “speak the same language”. Text inputs are passed to a text encoder, and image inputs to an image encoder \[3\]. These models then create a _vector representation_ of the respective input.

Both models “speak the same language” by encoding similar concepts in text and images into similar vectors. That means that the text “two dogs running across a frosty field” would output a vector similar to an _image_ of two dogs running across a frosty field.https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fa54a2f1fa0aeac03748c09df0fdfbb42aadc96b7-2430x1278.png&w=3840&q=75

Similar text and images will be encoded into a similar vector space. Dissimilar text and images do not share a similar vector space.

We can think of the language these models speak as the vector space in which they encode vectors. These two models can express nuanced information about text and images through this vector space. However, this “vector language” is far too abstract for us to directly understand.

Rather than directly reading this “language”, we can train other simple neural networks to understand it and make predictions that we can understand. Or we use vector search to identify similar concepts and patterns across text and image domains.

Let’s take a look at an example of CLIP in action.

### Text-to-Image Search

Entering a prompt in the search bar above allows us to search through images based on their _content_ rather than any attached textual metadata. We call this **C** ontent **B** ased **I** mage **R** etrieval (CBIR).

With CBIR, we can search for specific phrases such as “two dogs running across a frosty field”. We can even drop the word “dogs” and replace it with everyday slang for dogs like “good boy” or “mans best friend”, and we return the same images showing dogs running across fields.

CLIP can accurately understand language. It understands that _in the context_ of running across a field, we are likely referring to dogs and do not literally mean good children or someone’s “human” best friend.

Amusingly, the dataset contains no images of the food hot dogs (other than one). So, suppose we search for “hot dogs”. In that case, we first get an image containing a hot dog (and a dog), a dog looking toasty in a warm room, another dog looking warm with wooly clothing, and another dog posing for the camera. All of these portray a hot dog in one sense or another.

_After being processed by CLIP’s text or image encoder, we are left with vectors. That means we can search across_ **_any_** _modality with_ **_any_** _modality; we can search in either direction. We can also stick to a single modality, like text-to-text or image-to-image._

Now that we’ve seen what CLIP can do, let’s take a look at _how_ it can do this.

## CLIP

CLIP actually consists of two models trained in parallel. A 12-layer text transformer for building text embeddings and a ResNet or vision transformer (ViT) for building image embeddings \[3\].https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F539716ea1571e459908c1fdc5a898fea239d8243-2803x1672.png&w=3840&q=75

Architecture diagram of CLIP with the text encoder and ViT or ResNet as the image encoder.

The text encoder and image encoder (ResNet _or_ ViT) output single vector embeddings for each text/image record fed into the encoders. All vectors are 512 dimensional and can be represented in the same vector space, meaning similar images and text produce vectors that appear near each other.

### Contrastive Pretraining

Across both [**N** atural](https://www.pinecone.io/learn/series/nlp/) [**L** anguage](https://www.pinecone.io/learn/series/nlp/) [**P** rocessing (NLP)](https://www.pinecone.io/learn/series/nlp/) and computer vision (CV), large pretrained models dominate the SotA. The idea is that by giving a big model a lot of data, they can learn general patterns from the dataset.

For language models, that may be the general rules and patterns in the English language. For vision models, that may be the characteristics of different scenes or objects.

The problem with multi-modality is that these models are trained separately and, by default, have no understanding of one another. CLIP solves this thanks to image-text _contrastive pretraining_. With CLIP, text and image encoders are trained while considering the other modality and context. Meaning that the text and image encoders share an “indirect understanding” of patterns in both modalities; language and vision.

Contrastive pretraining works by taking a _(text, image)_ pair – where the text describes the image – and learning to encode the pairs as closely as possible in vector space.

For this to work well, we also need negative pairs to provide a contrastive comparison. We need positive pairs that should output similar vectors and negative pairs that should output dissimilar vectors.

This is the general idea behind contrastive learning, which can be found in the training functions of many models, particularly those that produce embedding vectors.

The negative pairs can be extracted directly from positive pairs. If we have positive pairs (T1,I1)(T\_1,I\_1)(T1​,I1​) and (T2,I2)(T\_2,I\_2)(T2​,I2​), we simply swap the components, giving us the negative pairs (T1,I2)(T\_1,I\_2)(T1​,I2​) and (T2,I1)(T\_2,I\_1)(T2​,I1​).

With this, we can apply a loss function that maximizes the similarity between (T1,I1)(T\_1,I\_1)(T1​,I1​) and (T2,I2)(T\_2,I\_2)(T2​,I2​), and minimizes the similarity between (T1,I2)(T\_1,I\_2)(T1​,I2​) and (T2,I1)(T\_2,I\_1)(T2​,I1​). Altogether, this looks like this:https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fd6868e6dae721512fed8f1287fc9ffe6b6a2cddd-2332x1342.png&w=3840&q=75

Contrastive pretraining with CLIP.

In this image, we can see a single pretraining step on a single batch. The loss function assumes pairs in the diagonal should have a maximized dot product score, and all other pairs should have a minimized dot product score. Both text and image encoder models are optimized for this.

A fundamental assumption is that there are no other positive pairs within a single batch. For example, we assume that “two dogs running across a frosty field” is only relevant to the image it is paired with. We assume there are no other texts or images with similar meanings.

This assumption is possible because the datasets used for pretraining are diverse and large enough that the likelihood of two similar pairs appearing in a single batch is negligible. Therefore, rare enough to have a little-to-no negative impact on pretraining performance.

## Using CLIP

We have a good idea of what CLIP can be used for and how it is trained. With that, how can we get started with it?

OpenAI released a few implementations of CLIP via the Hugging Face library; this is the fastest way to get started. First, we need to install the necessary libraries.

`pip install transformers torch datasets`

Before we can do anything with CLIP, we need some text and images. The `jamescalam/image-text-demo` dataset contains a small number of image-text pairs we can use in our examples.

```python
from datasets import load_dataset

data = load_dataset(
    "jamescalam/image-text-demo",
    split="train"
)
```https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fa40f673ed52e07f497c7a39b032c27b33ce9f565-1128x761.png&w=3840&q=75

Example of text-image pair found in the dataset. Text is stored in the "text" feature and images in the "image" feature.

With these sample records ready, we can move on to initializing CLIP and an image/text preprocessor like so:

```python
from transformers import CLIPProcessor, CLIPModel
import torch

model_id = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

# move model to device if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.to(device)
```

The `model` is CLIP itself. Note that we use the ViT image encoder (the model is `clip-vit`). Text and image data cannot be fed directly into CLIP. The text must be preprocessed to create “tokens IDs”, and images must be resized and normalized. The `processor` handles both of these functions.

### Encoding Text

We will start with encoding text using the CLIP text transformer. Before feeding text into CLIP, it must be preprocessed and converted into token IDs. Let’s take a batch of sentences from the `unsplash` data and encode them.

```python
text = data['text']  # 21

tokens = processor(
    text=text,
    padding=True,
    images=None,
    return_tensors='pt'
).to(device)
tokens.keys()
```

```
dict_keys(['input_ids', 'attention_mask'])
```

This returns the typical text transformer inputs of `input_ids` and `attention_mask`.

The `input_ids` are token ID values where each token ID is an integer value ID that maps to a specific word or sub-word. For example the phrase _“multi-modality”_ may be split into tokens _\[“multi”, “-”, “modal”, “ity”\]_, which are then mapped to IDs _\[1021, 110, 2427, 425\]_.

A text transformer maps these token IDs to semantic vector embeddings that the model learned during pretraining.

The `attention_mask` is a tensor of 1s and 0s used by the model’s internal mechanisms to “pay attention” to real token IDs and ignore padding tokens.

_Padding tokens are a special type of token used by text transformers to create input sequences of a fixed length from sentences of varying length. They are appended to the end of shorter sentences, so “hello world” may become “hello world \[PAD\] \[PAD\] \[PAD\]”._

We then use CLIP to encode all of these text descriptions with `get_text_features` like so:

```python
text_emb = model.get_text_features(
    **tokens
)
```

One important thing to note here is that these embeddings are _not_ normalized. If we plan on using a similarity metric like the dot product, we must normalize the embeddings:

```python
print(text_emb.shape)
print(text_emb.min(), text_emb.max())
```

```
torch.Size([21, 512])
tensor(-1.1893, grad_fn=<MinBackward1>) tensor(4.8015, grad_fn=<MaxBackward1>)

```

```python
# IF using dot product similarity, must normalize vectors like so...
import numpy as np

# detach text emb from graph, move to CPU, and convert to numpy array
text_emb = text_emb.detach().cpu().numpy()

# calculate value to normalize each vector by
norm_factor = np.linalg.norm(text_emb, axis=1)
norm_factor.shape
```

```
(21,)
```

```python
text_emb = text_emb.T / norm_factor
# transpose back to (21, 512)
text_emb = text_emb.T
print(text_emb.shape)
print(text_emb.min(), text_emb.max())
```

```
(21, 512)
-0.1526844 0.53449875

```

Alternatively, we can use cosine similarity as our metric as this only considers angular similarity and not vector magnitude (like dot product). For our examples, we will normalize and use dot product similarity.

We now have our text embeddings; let’s see how to do the same for images.

### Encoding Images

Images will be encoded using the ViT portion of CLIP. Similar to text encoding, we need to preprocess these images using the `preprocessor` like so:

```python
data['image'][0].size
```

```
(6000, 3376)
```

```python
image_batch = data['image']

images = processor(
    text=None,
    images=image_batch,
    return_tensors='pt'
)['pixel_values'].to(device)
images.shape
```

```
torch.Size([21, 3, 224, 224])
```

Preprocessing images does _not_ produce token IDs like those we saw from preprocessing our text. Instead, preprocessing images consists of resizing the image to a 244x244 array with three color channels (red, green, and blue) and normalizing pixel values into a \[0,1\]\[0,1\] range.

After preprocessing our images, we get the image features with `get_image_features` and normalize them as before:

```python
img_emb = model.get_image_features(images)
print(img_emb.shape)
print(img_emb.min(), img_emb.max())
```

```
torch.Size([21, 512])
tensor(-8.6533, grad_fn=<MinBackward1>) tensor(2.6551, grad_fn=<MaxBackward1>)

```

```python
# NORMALIZE
# detach text emb from graph, move to CPU, and convert to numpy array
img_emb = img_emb.detach().cpu().numpy()

img_emb = img_emb.T / np.linalg.norm(img_emb, axis=1)
# transpose back to (21, 512)
img_emb = img_emb.T
print(img_emb.shape)
print(img_emb.min(), img_emb.max())
```

```
(21, 512)
-0.7275361 0.23383287

```

With this, we have created CLIP embeddings for both text and images. We can move on to comparing items across the two modalities.

### Calculating Similarity

CLIP embedding similarities are represented by their angular similarity. Meaning we can identify similar pairs using cosine similarity:

cossim(A,B)=A⋅B∣∣A∣∣∗∣∣B∣∣=∑inAiBi∑inAi2∑inBi2cossim(A, B) = \\frac{A \\cdot B}{\|\|A\|\| \* \|\|B\|\|} = \\frac{\\sum\_i^nA\_iB\_i}{\\sqrt{\\sum\_i^nA\_i^2} \\sqrt{\\sum\_i^nB\_i^2}}cossim(A,B)=∣∣A∣∣∗∣∣B∣∣A⋅B​=∑in​Ai2​​∑in​Bi2​​∑in​Ai​Bi​​

Or, if we have normalized the embeddings, we can use dot product similarity:

dotproduct(A,B)=A⋅B=∑i=0n−1AiBidotproduct(A, B) = A \\cdot B = \\sum\_{i=0}^{n-1}A\_iB\_idotproduct(A,B)=A⋅B=i=0∑n−1​Ai​Bi​

Let’s try both. First, for cosine similarity, we do:

```python
from numpy.linalg import norm

cos_sim = np.dot(text_emb, img_emb.T) / (
    norm(text_emb, axis=1) * norm(img_emb, axis=1)
)
cos_sim.shape
```

```
(21, 21)
```

```python
import matplotlib.pyplot as plt

plt.imshow(cos_sim)
plt.show()
```

And if we perform the same operation for dot product similarity, we should return the same results:

```python
dot_sim = np.dot(text_emb, img_emb.T)

plt.imshow(cos_sim)
plt.show()
```

Both of these similarity score arrays look the same, and if we check for the difference between the two arrays, we will see that the scores are the same. We see some slight differences due to floating point errors.

```python
diff = cos_sim - dot_sim
diff.min(), diff.max()
```

```
(0.0, 2.9802322e-08)
```

Using the embedding functions of CLIP in this way, we can perform a semantic search across the modalities of text and image in any direction. We can search for images with text, text with images, text with text, and images with images.

These use cases are great, but we can make slight modifications to this for many other tasks.

### Classification

One of the most impressive demonstrations of CLIP is its unparalleled zero-shot performance on various tasks. For example, given the `fragment/imagenette` dataset from Hugging Face _Datasets_, we can write a list of brief sentences that align with the ten class labels.https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Ff841984e7617686f5041ca95797498e2b0b085b5-1348x542.png&w=3840&q=75

We take the original imagenette labels and preappend "a photo of a ..." to each to create a set of CLIP-friendly sentence representations.

From this, we can calculate the cosine similarity between the text embeddings of these ten labels against an image we’d like to classify. The text that returns the highest similarity is our predicted class.

### Object Detection

Another compelling use case of zero-shot CLIP is object detection. We can do this by splitting our images into smaller patches and running each patch through the image encoder of CLIP. We then compare these patch embeddings to a text encoding describing what we are looking for. After calculating the similarity scores for all patches, we can collate them into a map of relevance.

For example, given an image of a butterfly and a cat, we could break it into many small patches. Given the prompt `"a fluffy cat"`, we will return an outline of the cat, whereas the prompt `"a butterfly"` will produce an outline of the butterfly.https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fbe4800918976efd9d974d9e5453985a5106f2558-2389x1455.png&w=3840&q=75

Zero-shot object detection with CLIP allows us to find specific objects with natural language prompts.

These are only a few of the use cases of CLIP and only scratch the surface of what is possible with this model and others in the scope of multi-modal ML.

That’s it for this introduction to multi-modal ML with OpenAI’s CLIP. The past years since the CLIP release have seen ever more fascinating applications of the model.

DALL-E 2 is a well-known example of CLIP. The incredible images generated by DALL-E 2 start by embedding the user’s text prompt with CLIP \[4\]. That text embedding is then passed to the diffusion model, which generates some mind-blowing images.

The fields of NLP and CV have mainly progressed independently of each other for the past decade. However, with the introduction of world scope three models, they’re becoming more entwined into a majestic multi-modal field of Machine Learning.

</details>

<details>
<summary>multimodal-embeddings-an-introduction-towards-data-science</summary>

# Multimodal Embeddings: An Introduction

Mapping text and images into a common space

This is the 2nd article in a [larger series](https://shawhin.medium.com/list/multimodal-ai-fe9521d0e77a) on multimodal AI. In the [previous post](https://towardsdatascience.com/multimodal-models-llms-that-can-see-and-hear-5c6737c981d3), we saw how to augment [large language models (LLMs)](https://shawhin.medium.com/list/large-language-models-llms-8e009ae3054c) to understand new data modalities (e.g., images, audio, video). One such approach relied on encoders that generate vector representations (i.e. embeddings) of non-text data. In this article, I will discuss _multimodal_ embeddings and share what they can do via two practical use cases.https://towardsdatascience.com/wp-content/uploads/2024/11/1a6BF-kEeo8rd7OW2a3JYGA.pngImage from Canva.

* * *

AI research is traditionally split into distinct fields: NLP, computer vision (CV), robotics, human-computer interface (HCI), etc. However, countless practical tasks require the **integration of these different research areas** e.g. autonomous vehicles (CV + robotics), AI agents (NLP + CV + HCI), personalized learning (NLP + HCI), etc.

Although these fields aim to solve different problems and work with different data types, they all share a fundamental process. Namely, **generating useful numerical representations of real-world phenomena**.

Historically, this was done by hand. This means that researchers and practitioners would use their (or other people’s) expertise to explicitly transform data into a more helpful form. Today, however, _these can be derived another way_.

## **Embeddings**

**Embeddings** are **(useful) numerical representations of data learned implicitly through model training**. For example, through learning how to predict text, BERT learned representations of text, which are helpful for many NLP tasks \[1\]. Another example is the Vision Transformer (ViT), trained for image classification on Image Net, which can be repurposed for other applications \[2\].

A key point here is that these learned embedding spaces will have some underlying structure so that **similar concepts are located close together**. As shown in the toy examples below.https://towardsdatascience.com/wp-content/uploads/2024/11/1jpmC6Kx7DxVeikEr15vooA.pngToy represetation of text and image embeddings, respectively. Image by author.

One **key limitation** of the previously mentioned models is they are restricted to a single data modality, e.g., text or images. Preventing cross-modal applications like image captioning, content moderation, image search, and more. _But what if we could merge these two representations?_

## **Multimodal Embeddings**

Although text and images may look very different to us, in a neural network, these are **represented via the same mathematical object**, i.e., a vector. Therefore, in principle, text, images, or any other data modality can processed by a single model.

This fact underlies **multimodal embeddings**, which **represent multiple data modalities in the same vector space** such that similar concepts are co-located (independent of their original representations).https://towardsdatascience.com/wp-content/uploads/2024/11/15d3HBNjNIXLy0oMIvJjxWw.pngToy representation of multimodal embedding space. Image by author.

For example, CLIP encodes text and images into a shared embedding space \[3\]. A key insight from CLIP is that by aligning text and image representations, the **model is capable of 0-shot image classification on an arbitrary set of target classes** since any input text can be treated as a class label (we will see a concrete example of this later).

However, this idea is not limited to text and images. Virtually any data modalities can be aligned in this way e.g., text-audio, audio-image, text-EEG, image-tabular, and text-video. Unlocking use cases such as video captioning, advanced OCR, audio transcription, video search, and EEG-to-text \[4\].

## **Contrastive Learning**

The standard approach to aligning disparate embedding spaces is **contrastive learning (CL)**. A key intuition of CL is to **represent different views of the same _information_ similarly** \[5\].

This consists of learning representations that **maximize the similarity between positive pairs** and **minimize the similarity of negative pairs**. In the case of an image-text model, a positive pair might be an image with an appropriate caption, while a negative pair would be an image with an irrelevant caption (as shown below).https://towardsdatascience.com/wp-content/uploads/2024/11/1AGHBVjzwjXapJSe4aUPrjg.pngExample positive and negative pairs used in contrastive training. Image by author.

**Two key aspects** **of CL** contribute to its effectiveness

1. Since positive and negative pairs can be curated from the data’s inherent structure (e.g., metadata from web images), CL training data **do not require manual labeling**, which unlocks larger-scale training and more powerful representations \[3\].
2. It simultaneously maximizes positive and minimizes negative pair similarity via a special loss function, as demonstrated by CLIP \[3\].![CLIP's contrastive loss for text-image representation alignment [3]. Image by author.](https://towardsdatascience.com/wp-content/uploads/2024/11/12X1aT8fzFsgbqn23zXmmAA.png)CLIP’s contrastive loss for text-image representation alignment \[3\]. Image by author.

## **Example Code:** Using CLIP for 0-shot classification and image search

With a high-level understanding of how multimodal embeddings work, let’s see two concrete examples of what they can do. Here, I will use the open-source [CLIP model](https://huggingface.co/openai/clip-vit-base-patch16) to perform two tasks: 0-shot image classification and image search.

The **code for these examples** is freely available on the [GitHub repository](https://github.com/ShawhinT/YouTube-Blog/tree/main/multimodal-ai/2-mm-embeddings).

* * *

### Use case 1: 0-shot Image Classification

The basic idea behind using CLIP for 0-shot image classification is to pass an image into the model along with a set of possible class labels. Then, a classification can be made by **evaluating which text input is most similar to the input image**.

We’ll start by importing the [Hugging Face Transformers library](https://huggingface.co/docs/transformers/en/installation) so that the CLIP model can be downloaded locally. Additionally, the PIL library is used to load images in Python.

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
```

Next, we can import a version of the clip model and its associated data processor. _Note: the processor handles tokenizing input text and image preparation._

```ini
# import model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

# import processor (handles text tokenization and image preprocessing)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
```

We load in the below image of a cat and create a list of two possible class labels: " _a photo of a cat_" or " _a photo of a dog_".

```ini
# load image
image = Image.open("images/cat_cute.png")

# define text classes
text_classes = ["a photo of a cat", "a photo of a dog"]
```https://towardsdatascience.com/wp-content/uploads/2024/11/1Nzo536sqahqm1Q24Ms2vmA.pngInput cat photo. Image from Canva.

Next, we’ll preprocess the image/text inputs and pass them into the model.

```ini
# pass image and text classes to processor
inputs = processor(text=text_classes, images=image, return_tensors="pt",
                                                    padding=True)

# pass inputs to CLIP
outputs = model(**inputs) # note: "**" unpacks dictionary items
```

To make a class prediction, we must extract the image logits and evaluate which class corresponds to the maximum.

```makefile
# image-text similarity score
logits_per_image = outputs.logits_per_image
# convert scores to probs via softmax
probs = logits_per_image.softmax(dim=1)

# print prediction
predicted_class = text_classes[probs.argmax()]
print(predicted_class, "| Probability = ",
                       round(float(probs[0][probs.argmax()]),4))
```

``` language-none
>> a photo of a cat | Probability =  0.9979
```

The model nailed it with a 99.79% probability that it’s a cat photo. However, this was a super easy one. Let’s see what happens when we change the class labels to: " _ugly cat_" and " _cute cat_" for the same image.

``` language-none
>> cute cat | Probability =  0.9703
```

The model easily identified that the image was indeed a cute cat. Let’s do something more challenging like the labels: " _cat meme_" or " _not cat meme_".

``` language-none
>> not cat meme | Probability =  0.5464
```

While the model is less confident about this prediction with a 54.64% probability, it correctly implies that the image is not a meme.

### Use case 2: Image Search

Another application of CLIP is essentially the inverse of Use Case 1. Rather than identifying which text label matches an input image, we can evaluate **which image (in a set) best matches a text input (i.e. query)**—in other words, performing a search over images.

We start by storing a set of images in a list. Here, I have three images of a cat, dog, and goat, respectively.

```python
# create list of images to search over
image_name_list = ["images/cat_cute.png", "images/dog.png", "images/goat.png"]

image_list = []
for image_name in image_name_list:
    image_list.append(Image.open(image_name))
```

Next, we can define a query like " _a cute dog_" and pass it and the images into CLIP.

```python
# define a query
query = "a cute dog"

# pass images and query to CLIP
inputs = processor(text=query, images=image_list, return_tensors="pt",
                                                  padding=True)
```

We can then match the best image to the input text by extracting the text logits and evaluating the image corresponding to the maximum.

```python
# compute logits and probabilities
outputs = model(**inputs)
logits_per_text = outputs.logits_per_text
probs = logits_per_text.softmax(dim=1)

# print best match
best_match = image_list[probs.argmax()]
prob_match = round(float(probs[0][probs.argmax()]),4)

print("Match probability: ",prob_match)
display(best_match)
```

``` language-none
>> Match probability:  0.9817
```https://towardsdatascience.com/wp-content/uploads/2024/11/14wnqr5p_7N3QD5EkXIQeew.pngBest match for query "a cute dog". Image from Canva.

We see that (again) the model nailed this simple example. But let’s try some trickier examples.

```python
query = "something cute but metal 🤘"
```

``` language-none
>> Match probability:  0.7715
```https://towardsdatascience.com/wp-content/uploads/2024/11/1tIY3_ONQQT_cracAPWm8NQ.pngBest match for query "something cute but metal 🤘". Image from Canva.

```python
query = "a good boy"
```

``` language-none
>> Match probability:  0.8248
```https://towardsdatascience.com/wp-content/uploads/2024/11/14wnqr5p_7N3QD5EkXIQeew.pngBest match for query "a good boy". Image from Canva.

```python
query = "the best pet in the world"
```

``` language-none
>> Match probability:  0.5664
```https://towardsdatascience.com/wp-content/uploads/2024/11/1Nzo536sqahqm1Q24Ms2vmA.pngBest match for query "the best pet in the world". Image from Canva.

Although this last prediction is quite controversial, all the other matches were spot on! This is likely since images like these are ubiquitous on the internet and thus were seen many times in CLIP’s pre-training.

</details>

<details>
<summary>multimodal-rag-with-colpali-milvus-and-vlms</summary>

In this post, we will see how to doIn this post, we will see how to do multimodal RAG with [colpali](https://arxiv.org/abs/2407.01449), [milvus](https://milvus.io/) and a visual language model (gemini/gpt-4o).

We will build an application to upload a PDF and then do Q&A queries on it. Q&A can be done on both text and visual elements of the PDF. We will not extract text from the PDF; instead, we will treat it as an image and use colpali to get embeddings for the PDF pages. These embeddings will be indexed to Milvus, and then we will use a VLM to do Q&A queries on the PDF pages.

## Problem

Let's say a company wants to build a Q&A/search interface for its internal documents, which include PDFs, word files, wikis, images, and text files. The traditional approach involves extracting text and media, detecting layout for structure, and indexing the information in a vector store for semantic search. However, this method often falls short for complex documents containing images, tables, and graphs. Let's look at an example below:

We have a [PDF with stats on covid](https://saumitra.me/2024/covid-slides.pdf) in the form of charts and tables. We want to answer the queries below:

```markdown
1. What is the correlation between the samples tested and the positivity rate?
2. When and what was the highest number of cases and TPR?
3. Which country had the highest omicron cases?

```

These queries can be answered by using data from following 3 pages:

**Page 4: A chart showing stats on samples and positivity rate**

[https://saumitra.me/2024/covid-page-4.png](https://saumitra.me/2024/covid-page-4.png)

**Page 8: A table showing cases and TPR**

[https://saumitra.me/2024/covid-page-8.png](https://saumitra.me/2024/covid-page-8.png)

**Page 9: A table showing cases by country**

[https://saumitra.me/2024/covid-page-9.png](https://saumitra.me/2024/covid-page-9.png)

It would be difficult to extract data from these pages as text in a manner which can be used for querying.
We want to show user the answer and source page(s) from the PDF which contains the answer, like below:

[https://saumitra.me/2024/rag-demo-screenshot.png](https://saumitra.me/2024/rag-demo-screenshot.png)

Let's understand how colpali can help us here.

## Why colpali?

Document retrieval has always been a key component of systems like search engines and information retrieval. Traditional document retrieval methods rely heavily on text-based methods (like OCR and text segmentation), often missing crucial visual cues like layouts, images, and tables.

Colpali addresses this by using Vision-Language Models (VLMs) to understand and retrieve visually rich documents, capturing both textual and visual information. Colpali's architecture allows direct encoding of document images into a common embedding space, eliminating the need for time-consuming text extraction and segmentation.

## Understanding how colpali works

Colpali works in the following steps:

### Step 1: Treating the Document as an Image

Imagine we have a PDF document. Normally, we would extract text from the document using OCR (Optical Character Recognition), segment it into different sections, and then use these segments for searching. colpali simplifies this process by treating the entire document page as an image, bypassing the need for complex text extraction, layout detection, or OCR.

### Step 2: Splitting the Image into Patches

Once colpali has this "image" of the document, it divides the page into small, uniform pieces called patches. Each patch captures a tiny portion of the page. It might contain a few words, a piece of a graph, or part of an image. This division helps the model focus on the document's small, detailed parts rather than trying to understand the whole page at once.

At first glance, it might seem like dividing an image into patches is similar to breaking text into chunks. However, these two methods have several key differences, especially in how they handle and preserve context. Let’s dive deeper into these differences to understand why patch-based processing in colpali is more effective for document retrieval compared to traditional text chunking.

#### Understanding Context Loss in Text Chunking

In traditional text chunking, text is split into smaller chunks based on certain tokens since many models limit the number of tokens they can process at once.

Problem with Context Loss:

- Chunking can split sentences or paragraphs midway, causing crucial context to be lost. It can also result in incomplete information in one chunk and missing context in another.
Chunking doesn't preserve visual or structural information, such as the relationship between headings and their corresponding content or the placement of text in tables or figures.

For example, If you have a document with a heading followed by a table, text chunking might separate the heading and the table, losing the context that the table belongs to that heading.

#### Patch-Based Image Processing in colpali

Colpali divides the document image into patches, much like dividing a photo into small squares. Each patch is a fixed-size portion of the image, like a mini-snapshot of that part of the page.

Patches are more effective due to the following reasons:

- **No Loss of Structure:** The patches retain the document's visual structure, preserving its spatial layout. For instance, if a page has two columns of text or a table with rows and columns, each patch maintains its relative position, ensuring that the model understands the overall arrangement of the elements.
- **Multi-Modal Context:** Patches capture both textual and visual information. This includes both visual features (e.g., font styles, colors, boldness) and non-text elements (e.g., figures and graphs).
- **Positional Awareness:** Each patch has a positional embedding that tells the model where it is located on the page, helping the model understand the overall layout.

### Step 3: Embedding Creation and **Aligning Visual and Textual Information**

Each patch is then passed through a Vision Transformer (ViT), which converts them into unique embeddings. Next, colpali aligns these visual embeddings with the text of the query by transforming the query into its own set of embeddings. colpali uses a process called `alignment` that aligns image path embeddings and text embeddings in the same vector space. Only then can we compare the similarity between query and document embeddings.

### Step 4: Scoring the Relevance - Late Interaction Mechanism

At this point, colpali has embeddings for both the query and the document. The next challenge is to identify the relevant parts of the document. colpali uses a process called the `Late Interaction Mechanism`, where each piece of the query is finely matched against every part of the document, scoring and ranking their relevance.

Colpali highlights the most relevant pieces of the document, focusing on the patches that best match the query. This approach enables colpali to efficiently retrieve relevant information from visually rich documents, capturing both visual and textual data without losing context.

* * *

## Code

### 1\. Add colpali processor

```python
model_name = "vidore/colpali-v1.2"
device = get_torch_device("cuda")

model = colpali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
).eval()

processor = cast(colpaliProcessor, colpaliProcessor.from_pretrained(model_name))

```

### 2\. Use colpali to get embeddings for image (pdf pages)

```python
def process_images(self, image_paths:list[str], batch_size=5):

    print(f"Processing {len(image_paths)} image_paths")

    images = self.get_images(image_paths)

    dataloader = DataLoader(
        dataset=ListDataset[str](images),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )

    ds: List[torch.Tensor] = []
    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to(device))))

    ds_np = [d.float().cpu().numpy() for d in ds]

    return ds_np

```

### 3\. Use colpali to get embeddings for text (user query)

```python
def process_text(self, texts: list[str]):
    print(f"Processing {len(texts)} texts")

    dataloader = DataLoader(
        dataset=ListDataset[str](texts),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: processor.process_queries(x),
    )

    qs: List[torch.Tensor] = []
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
            embeddings_query = model(**batch_query)

        qs.extend(list(torch.unbind(embeddings_query.to(device))))

    qs_np = [q.float().cpu().numpy() for q in qs]

    return qs_np

```

### 4\. Code to create collection, index and query in milvus

```python
class MilvusManager:
    def __init__(self, milvus_uri, collection_name, create_collection, dim=128):
        self.client = MilvusClient(uri=milvus_uri)
        self.collection_name = collection_name
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.load_collection(collection_name)
        self.dim = dim

        if create_collection:
            self.create_collection()
            self.create_index()

    def create_collection(self):
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_fields=True,
        )
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim
        )
        schema.add_field(field_name="seq_id", datatype=DataType.INT16)
        schema.add_field(field_name="doc_id", datatype=DataType.INT64)
        schema.add_field(field_name="doc", datatype=DataType.VARCHAR, max_length=65535)

        self.client.create_collection(
            collection_name=self.collection_name, schema=schema
        )

    def create_index(self):
        self.client.release_collection(collection_name=self.collection_name)
        self.client.drop_index(
            collection_name=self.collection_name, index_name="vector"
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_name="vector_index",
            index_type="HNSW",
            metric_type="IP",
            params={
                "M": 16,
                "efConstruction": 500,
            },
        )

        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )

    def create_scalar_index(self):
        self.client.release_collection(collection_name=self.collection_name)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="doc_id",
            index_name="int32_index",
            index_type="INVERTED",
        )

        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )

    def search(self, data, topk):
        search_params = {"metric_type": "IP", "params": {}}
        results = self.client.search(
            self.collection_name,
            data,
            limit=int(50),
            output_fields=["vector", "seq_id", "doc_id"],
            search_params=search_params,
        )
        doc_ids = set()
        for r_id in range(len(results)):
            for r in range(len(results[r_id])):
                doc_ids.add(results[r_id][r]["entity"]["doc_id"])

        scores = []

        def rerank_single_doc(doc_id, data, client, collection_name):
            doc_colbert_vecs = client.query(
                collection_name=collection_name,
                filter=f"doc_id in [{doc_id}, {doc_id + 1}]",
                output_fields=["seq_id", "vector", "doc"],
                limit=1000,
            )
            doc_vecs = np.vstack(
                [doc_colbert_vecs[i]["vector"] for i in range(len(doc_colbert_vecs))]
            )
            score = np.dot(data, doc_vecs.T).max(1).sum()
            return (score, doc_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
            futures = {
                executor.submit(
                    rerank_single_doc, doc_id, data, self.client, self.collection_name
                ): doc_id
                for doc_id in doc_ids
            }
            for future in concurrent.futures.as_completed(futures):
                score, doc_id = future.result()
                scores.append((score, doc_id))

        scores.sort(key=lambda x: x[0], reverse=True)
        if len(scores) >= topk:
            return scores[:topk]
        else:
            return scores

    def insert(self, data):
        colbert_vecs = [vec for vec in data["colbert_vecs"]]
        seq_length = len(colbert_vecs)
        doc_ids = [data["doc_id"] for i in range(seq_length)]
        seq_ids = list(range(seq_length))
        docs = [""] * seq_length
        docs[0] = data["filepath"]

        self.client.insert(
            self.collection_name,
            [\
                {\
                    "vector": colbert_vecs[i],\
                    "seq_id": seq_ids[i],\
                    "doc_id": doc_ids[i],\
                    "doc": docs[i],\
                }\
                for i in range(seq_length)\
            ],
        )

    def get_images_as_doc(self, images_with_vectors:list):

        images_data = []

        for i in range(len(images_with_vectors)):
            data = {
                "colbert_vecs": images_with_vectors[i]["colbert_vecs"],
                "doc_id": i,
                "filepath": images_with_vectors[i]["filepath"],
            }
            images_data.append(data)

        return images_data

    def insert_images_data(self, image_data):
        data = self.get_images_as_doc(image_data)

        for i in range(len(data)):
            self.insert(data[i])

```

### 5\. Save pdf as individual images

```python
class PdfManager:
    def __init__(self):
        pass

    def clear_and_recreate_dir(self, output_folder):
        print(f"Clearing output folder {output_folder}")

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        os.makedirs(output_folder)

    def save_images(self, id, pdf_path, max_pages, pages: list[int] = None) -> list[str]:
        output_folder = f"pages/{id}/"
        images = convert_from_path(pdf_path)

        print(f"Saving images from {pdf_path} to {output_folder}. Max pages: {max_pages}")

        self.clear_and_recreate_dir(output_folder)

        num_page_processed = 0

        for i, image in enumerate(images):
            if max_pages and num_page_processed >= max_pages:
                break

            if pages and i not in pages:
                continue

            full_save_path = f"{output_folder}/page_{i + 1}.png"

            image.save(full_save_path, "PNG")

            num_page_processed += 1

        return [f"{output_folder}/page_{i + 1}.png" for i in range(num_page_processed)]

```

### 6\. Middleware to index and search Milvus for embeddings generated from colpali

```python
class Middleware:
    def __init__(self, id:str, create_collection=True):
        hashed_id = hashlib.md5(id.encode()).hexdigest()[:8]
        milvus_db_name = f"milvus_{hashed_id}.db"
        self.milvus_manager = MilvusManager(milvus_db_name, "colpali", create_collection)

    def index(self, pdf_path: str, id:str, max_pages: int, pages: list[int] = None):

        print(f"Indexing {pdf_path}, id: {id}, max_pages: {max_pages}")

        image_paths = pdf_manager.save_images(id, pdf_path, max_pages)

        print(f"Saved {len(image_paths)} images")

        colbert_vecs = colpali_manager.process_images(image_paths)

        images_data = [{\
            "colbert_vecs": colbert_vecs[i],\
            "filepath": image_paths[i]\
        } for i in range(len(image_paths))]

        print(f"Inserting {len(images_data)} images data to Milvus")

        self.milvus_manager.insert_images_data(images_data)

        print("Indexing completed")

        return image_paths


    def search(self, search_queries: list[str]):
        print(f"Searching for {len(search_queries)} queries")

        final_res = []

        for query in search_queries:
            print(f"Searching for query: {query}")
            query_vec = colpali_manager.process_text([query])[0]
            search_res = self.milvus_manager.search(query_vec, topk=1)
            print(f"Search result: {search_res} for query: {query}")
            final_res.append(search_res)

        return final_res

```

### 7\. Use Gemini or gpt-4o to do Q&A on pdf page(s) matching user query

```python
class Rag:

    def get_answer_from_gemini(self, query, imagePaths):

        print(f"Querying Gemini for query={query}, imagePaths={imagePaths}")

        try:
            genai.configure(api_key=os.environ['GEMINI_API_KEY'])
            model = genai.GenerativeModel('gemini-1.5-flash')

            images = [Image.open(path) for path in imagePaths]

            chat = model.start_chat()

            response = chat.send_message([*images, query])

            answer = response.text

            print(answer)

            return answer

        except Exception as e:
            print(f"An error occurred while querying Gemini: {e}")
            return f"Error: {str(e)}"


    def get_answer_from_openai(self, query, imagesPaths):
        print(f"Querying OpenAI for query={query}, imagesPaths={imagesPaths}")

        try:
            payload = self.__get_openai_api_payload(query, imagesPaths)

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
            }

            response = requests.post(
                url="https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()  # Raise an HTTPError for bad responses

            answer = response.json()["choices"][0]["message"]["content"]

            print(answer)

            return answer

        except Exception as e:
            print(f"An error occurred while querying OpenAI: {e}")
            return None

    def __get_openai_api_payload(self, query:str, imagesPaths:List[str]):
        image_payload = []

        for imagePath in imagesPaths:
            base64_image = encode_image(imagePath)
            image_payload.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        payload = {
            "model": "gpt-4o",
            "messages": [\
                {\
                    "role": "user",\
                    "content": [\
                        {\
                            "type": "text",\
                            "text": query\
                        },\
                        *image_payload\
                    ]\
                }\
            ],
            "max_tokens": 1024
        }

        return payload


```

In the next post, we will understand the limitations of colpali and a workaround for them.

## References

1. [https://milvus.io/docs/use\_colpali\_with\_milvus.md](https://milvus.io/docs/use_colpali_with_milvus.md)
2. [https://arxiv.org/abs/2407.01449](https://arxiv.org/abs/2407.01449)

</details>

<details>
<summary>start-with-a-prebuilt-agent</summary>

```md-code__content
from langgraph.prebuilt import create_react_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    prompt="You are a helpful assistant"
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

```

</details>

<details>
<summary>the-8-best-ai-image-generators-in-2025-zapier</summary>

# The 8 best AI image generators in 2025

## Get the best AI-generated images using text-to-image AI.

By Harry Guinness · May 23, 2025https://images.ctfassets.net/lzny33ho1g45/2olcy4TVSWAjqy5dsxLNZd/09b4a18346af97076615d5f1d1407c39/best-ai-image-generator-hero.jpg?fm=jpg&q=31&fit=thumb&w=1520&h=760

AI image generators have been brewing (generating?) up a storm for the last couple of years. If you've been on social media, watched prime time news shows, or read a magazine, AI-generated images have been impossible to miss. These kinds of AI-generated images are everywhere, and sometimes you won't even realize. If you want to join in the fun, or [add some AI-powered features to your business workflows](https://zapier.com/blog/ai-image-examples-for-business/), the apps on this list will give you what you're looking for.

I've been writing about AI image generators [since Google Deep Dream in 2015](https://photography.tutsplus.com/articles/brave-new-camera-computational-photography--cms-23438). That's about as long as anyone outside of a computer science lab has realistically been thinking about these tools, and I'm really excited by how far they've come.

I'm going to try to avoid the thorny discussions around artistic merit, whether or not these tools are replacing or augmenting artists, and copyright infringement in training data, at least where I can. Instead, I'll focus on the fact that these AI image generators can now produce excellent results from a wide range of text and image prompts.

It's worth taking a few hours to play around with one of these text-to-image AI apps—even just so you can appreciate them from a technical perspective. Whether you like it or not, we're all seeing a lot of their output at the moment. And there will only be more to come.

## The best AI image generators

- [ChatGPT (GPT-4o)](https://zapier.com/blog/best-ai-image-generator/#gpt-4o) for the best AI image generator overall

- [Midjourney](https://zapier.com/blog/best-ai-image-generator/#midjourney) for artistic results

- [Reve](https://zapier.com/blog/best-ai-image-generator/#reve) for overall prompt adherence

- [Ideogram](https://zapier.com/blog/best-ai-image-generator/#ideogram) for accurate text

- [Stable Diffusion](https://zapier.com/blog/best-ai-image-generator/#stable-diffusion) for customization and control of your AI images

- [FLUX.1](https://zapier.com/blog/best-ai-image-generator/#flux) for a Stable Diffusion alternative

- [Adobe Firefly](https://zapier.com/blog/best-ai-image-generator/#firefly) for integrating AI-generated images into photos

- [Recraft](https://zapier.com/blog/best-ai-image-generator/#recraft) for graphic design


## How do AI image generators work?

All these AI image generators take a text prompt and then turn it—as best they can—into a matching image. This opens up some wild possibilities, since your prompt can be anything from "an impressionist oil painting of a Canadian man riding a moose through a forest of maple trees" to "a painting in the style of Vermeer of a large fluffy Irish wolfhound enjoying a pint of beer in a traditional pub" or "a photograph of a donkey on the moon."https://images.ctfassets.net/lzny33ho1g45/2udOp4paDgOh5HpqG5JRAQ/18abc9476c4705aacf3609edcec4f945/image8.jpeg?

I made this with Midjourney using the prompt "an impressionist oil painting of a Canadian man riding a moose through a forest of maple trees"

Seriously, the only real limits are your imagination, the AI image generator's ability to [comprehend your prompt](https://zapier.com/blog/natural-language-processing/), and any content filters put in place to stop plagiarism, copyright infringement, and bad actors flooding the internet with AI-generated violence or other NSFW content. (That Vermeer prompt used to work reliably, but some more restrictive image generators now block it because it uses a named artist.)

Most AI image generators work in a pretty similar way. [Millions or billions](https://laion.ai/blog/laion-5b/) of image-text pairs are used to train a neural network (basically, a very fancy computer algorithm [modeled loosely on the human brain](https://news.mit.edu/2017/explained-neural-networks-deep-learning-0414)) on _what things are_. By allowing it to process near-countless images, it learns what dogs, the color red, Vermeers, and everything else are. Once this is done, you have an AI that can interpret almost any prompt—though [there is a skill in setting things up](https://zapier.com/blog/ai-art-prompts/) so it can do so accurately.

The next step is to actually render the AI-generated image. The latest generation of AI image generators typically uses a [process called diffusion](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)—though OpenAI's latest foray into image generation uses a slightly different [process called autoregression](https://arxiv.org/abs/2404.02905). In essence, the image generators start with a random field of noise and then edit it in a series of steps to match their interpretation of the prompt. It's kind of like looking up at a cloudy sky, finding a cloud that looks kind of like a dog, and then being able to snap your fingers to keep making it more and more dog-like.https://images.ctfassets.net/lzny33ho1g45/1LHdvgxMxOKcgWqC2yzoKh/ff7194426828d81a2d8437f4f9c38132/ai-image-generator-dogs.png?

A dog-shaped cloud floating in a clear blue sky—from top-left, going clockwise, at 10 steps, 20 steps, 40 steps, and 120 steps.

Before we dive in: I don't want to oversell things. What these text-to-image generators can do is very impressive, but they aren't likely to save you from ever having to do a product photoshoot again. If you just need some weird or unique images, they can really help. But if you're looking for something super specific, you're better off hiring a photographer—or licensing the exact image you want. Similarly, trying to use one to [make a header image for a blog post](https://zapier.com/blog/generate-blog-images-with-dall-e-and-zapier/) can take a lot more time than just finding a header image for your blog through a stock photo site.

## What makes the best AI image generator?

There's a reason that AI image generators have become incredibly popular over the past few years: before that, they were pretty bad. The technology underlying them was incredibly cool and impressive, at least to research scientists, but [the images they could output](https://www.theguardian.com/artanddesign/2016/mar/28/google-deep-dream-art) were underwhelming. Even the original DALL·E was more of a fun novelty than a world-shaking revelation [when it launched in 2021](https://openai.com/research/dall-e).

Now that these text-to-image generators have been around for a while, there's some real competition between the different models. They've really increased in quality and can now even generate text somewhat accurately. If all you care about is the current "best" model, check out [Artificial Analysis's Image Arena](https://artificialanalysis.ai/text-to-image/arena?tab=Leaderboard). But we've reached the stage where the top dozen or more models are all excellent, so other features and usability matter more than they used to.

So, to find the best AI art generators, I set some pretty strict criteria:

- I was looking for apps that allowed you to generate AI images from a text prompt (and to a lesser degree, an image prompt). Tools that have you upload a dozen of your photos and then [spit out AI-generated portraits](https://land.prisma-ai.com/magic-avatars/) are fun (and normally built using Stable Diffusion), but they aren't the kind of general-purpose image generators I was considering.

- I was looking at the AI image generators themselves, not [tools built on top of them](https://zapier.com/blog/ai-art-generator/). For example, [NightCafe](https://nightcafe.studio/) is an AI picture generator that has a great community and app, but it just enables you to use [open source models](https://zapier.com/blog/open-source-ai/) like FLUX and Stable Diffusion, fine-tuned models based on various versions of them, the DALL·E 3 and Google Imagen APIs, as well as a handful of older generative models. It's worth checking out, but it doesn't meet my criteria for its own entry on this list.


Aside from all that, I also considered how easy each AI image creator is to use, what kinds of controls and customization options it provides (for things like AI image upscale), what pricing model it has, and most important of all: how good were the results? The best AI image generators are now far less likely to create weird or impossible-looking things.

I've been using and writing about text-to image generators since the original DALL·E launched, and about photography and art for more than a decade, so I'm pretty familiar with how all these tools work—and their various pros, cons, and bonkers behaviors. But writing this article was actually the first time I've put so many AI image generators head-to-head with the _same prompts_. The results were fascinating, and I'm delighted to say all the apps on the list offer genuine reasons to use them.

## The best AI image generators at a glance

|  | **Best for** | **Access options** | **Price** | **Parent company** |
| --- | --- | --- | --- | --- |
| [ChatGPT (GPT-4o)](https://zapier.com/blog/best-ai-image-generator/#gpt-4o) | Ease of use and overall quality | ChatGPT; API | Free with ChatGPT; fewer restrictions with ChatGPT Plus at $20/month | OpenAI |
| [Midjourney](https://zapier.com/blog/best-ai-image-generator/#midjourney) | Artistic results | Web app; Discord | From $10/month for ~200 images/month and commercial usage rights | Midjourney |
| [Reve](https://zapier.com/blog/best-ai-image-generator/#reve) | Adhering to prompts | Web app | 20 free credits/day; $5 for 500 credits | Reve |
| [Ideogram](https://zapier.com/blog/best-ai-image-generator/#ideogram) | Accurate text | Web app | Limited free plan; from $8/month for full-resolution download and 400 monthly priority credits | Ideogram AI |
| [Stable Diffusion](https://zapier.com/blog/best-ai-image-generator/#stable-diffusion) | Customization and control | NightCafe, Tensor.Art, Civitai, and lots of other apps; API; downloading it to a local server | Depends on the platform | Stability AI |
| [FLUX.1](https://zapier.com/blog/best-ai-image-generator/#flux) | Stable Diffusion alternative | NightCafe, Tensor.Art, Civitai, and lots of other apps; API; downloading it to a local server | Depends on the platform | Black Forest Labs |
| [Adobe Firefly](https://zapier.com/blog/best-ai-image-generator/#firefly) | Using AI-generated images in photos | firefly.adobe.com, Photoshop, Express, and other Adobe tools | Limited free credits; from $9.99 for 2,000 credits/month | Adobe |
| [Recraft](https://zapier.com/blog/best-ai-image-generator/#recraft) | Graphic design | Web app | Free for 50 credits/day; from $12/month for full features | Recraft |

## The best AI image generator overall

### [GPT-4o](https://chat.com/)(ChatGPT)https://images.ctfassets.net/lzny33ho1g45/75DSS8gsgXORvalbs3MCyE/e5c337007c370f28ba0e27584234c762/image13.jpg?

**GPT-4o pros:**

- Incredibly easy to use and a best-in-class model

- Included with ChatGPT Plus, so you get a lot of AI for your money

- Integrates with Zapier


**GPT-4o cons:**

- Very slow

- Controls can be hit and miss

- $20/month is pricey if you don't want the rest of ChatGPT with it


After OpenAI's [DALL·E](https://zapier.com/blog/dall-e-3/) model kickstarted the text-to-image boom, it seemed to take a backseat to the company's language models. DALL·E 2 and DALL·E 3 were good when they debuted, but were both quickly overtaken by other models. But now OpenAI is back with a bang. GPT-4o, the [multimodal](https://zapier.com/blog/multimodal-ai/) model that powers [ChatGPT](https://zapier.com/blog/how-to-use-chatgpt/), can now [natively generate images](https://zapier.com/blog/chatgpt-image-generation/).

GPT-4o is one of the best image generators available. It's also ridiculously easy to use: tell ChatGPT what you want to see, and it'll create the image. Unfortunately, because GPT-4o uses an autoregression model instead of diffusion, it's much slower than the other image generators on this list—and it only generates a single image. If you're only occasionally generating a few images, this isn't a big deal, but it's worth noting.

It's really solid across the board: accurate text rendering, easy editing, understanding of numbers and position, the list goes on. GPT-4o's best feature, though, is what's caused it to go viral. It's great adhering to image prompts (and it's pretty good at adhering to regular prompts, too). If you upload a photo and direct it to create the image in the style of Picasso, Vermeer, or, yes, Studio Ghibli, it will do an exceptional job. It's also pretty good at incorporating feedback—ask it to change just one element of your image and it generally will. Compared to DALL·E 3 (which you can still use as a [GPT](https://chatgpt.com/g/g-2fkFE8rbu-dall-e)), it's a huge improvement.

In addition to GPT-4o image generation through ChatGPT, [OpenAI offers an API](https://zapier.com/blog/openai-api/), which means you can [connect ChatGPT to Zapier](https://zapier.com/apps/chatgpt/integrations) to do things like automatically create images from Google Forms or HubSpot responses—or any other apps you use. Learn more about [how to automate ChatGPT](https://zapier.com/blog/automate-chatgpt/), or get started with one of these pre-made templates.

**GPT-4o pricing:** Free users can access it, but if you don't want to run into limits, GPT-4o image generation is included as part of ChatGPT Plus at $20/month.

## The best AI image generator for artistic results

### [**Midjourney**](https://www.midjourney.com/explore?tab=top)https://images.ctfassets.net/lzny33ho1g45/5c2lxK4vhLWzfata4t1eul/5037e39582914b8b1f4be36d945085e3/image12.jpg?

**Midjourney pros:**

- Consistently produces some of the best looking AI-generated images

- The community is a great way to get inspiration


**Midjourney cons:**

- Images you generate are public by default

- [Free trials are currently suspended](https://help.midjourney.com/en/articles/8150088-is-there-a-free-trial)


For a long time, [Midjourney](https://zapier.com/blog/how-to-use-midjourney/) produced my favorite results of all of the image generators on this list. Other apps have finally caught up with its quality, but it still produces some of the most coherent, visually appealing, and interesting results with great textures and colors. It's telling that it was [the first AI image generator to win an art competition](https://www.nytimes.com/2022/09/02/technology/ai-artificial-intelligence-artists.html).

Best of all, Midjourney now has an actual web app. You no longer have to access it through Discord—though you can if you want.

Still, as you can probably guess, Midjourney isn't totally free of quirks: by default, every image you generate is posted publicly on Midjourney's Explore page and can be viewed on your profile. It gives everything a cool community aspect, but it means that anyone who cares to look can see what you're creating. While not necessarily a problem for artists, this might be a dealbreaker if you're looking to use Midjourney for business purposes.

If things still sound a bit confusing, don't worry. [Midjourney's help docs](https://docs.midjourney.com/docs/quick-start) are really good and walk you through getting started with both the web app and Discord, and they show you how to control all its various features, from selecting model versions and upscaling to using character references and its personalization tools. Once you understand the different options, the results you can get are genuinely amazing.

Midjourney's free trials are currently suspended because of [the overwhelming number of people trying to use it](https://www.theverge.com/2023/3/30/23662940/deepfake-viral-ai-misinformation-midjourney-stops-free-trials), but they're occasionally reinstated for a few days. If you miss a free trial window, the Basic Plan starts at $10/month and comes with 3.3 hours of GPU time per month, or around 200 images. You also get the option to buy additional GPU time, and you can use your images commercially.

**Midjourney pricing:** From $10/month for the Basic Plan that allows you to generate ~200 images/month and provides commercial usage rights.

**Read more:** [Midjourney vs. DALL·E 3](https://zapier.com/blog/midjourney-vs-dalle/)

## The best AI image generator for adhering to prompts

### [**Reve**](https://preview.reve.art/)https://images.ctfassets.net/lzny33ho1g45/1rErUICKuzBtIoT0x1EmHf/4e9492bc64da35bec554e2eb16f4ca02/image7.jpg?

**Reve Image 1.0 pros:**

- Great prompt adherence

- Free plan plus affordable credit system


**Reve Image 1.0 cons:**

- Images you generate are public by default


Reve Image 1.0 is a new image model that essentially came out of nowhere in March 2025. It instantly jumped to the top of Artificial Analysis's leaderboard—until it was replaced by GPT-4o a few days later. Still, Image 1.0 is an incredibly powerful image generator with best-in-class prompt adherence.

In plain English, that means Reve Image 1.0 is able to stick closely to the prompt you give it. If you ask for, say, an image with a warrior holding a sword and a wizard holding a staff, that's what you'll get—not a warrior with a staff and a wizard with a sword. This kind of adherence has been a struggle for image generators, especially as prompts get longer and more complicated. I was pretty blown away by just how many details Image 1.0 could manage.

On top of that, Image 1.0 is great with text, different styles, and photorealism. Really, the only area it falls short is with editing. While you can edit a prompt or instruct the model to do something differently, it isn't as effective as GPT-4o or Midjourney at incorporating these changes.

Reve Image 1.0 also represents a return to credit-based pricing, which had fallen out of vogue. You get 100 free credits to start and 20 credits per day. Packs of 500 credits cost $5. Each credit is good for one image, though be warned: on the default settings, you generate four images for every prompt.

**Reve Image 1.0 pricing:** Free for 20 credits per day; additional credits are $5/500

## Best AI image generator for accurate text

### [Ideogram](https://ideogram.ai/)https://images.ctfassets.net/lzny33ho1g45/7xaiByWYInfO3qQnxkpn9O/05f734289aa1b0f517b3f43eb74f9680/image15-ideogram.jpg?

**Ideogram pros:**

- Great looking AI-generated images—and among the most accurate text of any app

- There's a free plan


**Ideogram cons:**

- Images you generate are public by default


Although they're getting better, most AI image generators still struggle to generate text correctly—the diffusion process just doesn't lend itself to precisely rendering letters. Ideogram, though, has cracked it. Its latest 3.0 algorithm is able to accurately and reliably include text along with any generated image.

What makes this more impressive is Ideogram is also one of the best image generators overall. It has an intuitive web app and some nice features like an [image editor](https://docs.ideogram.ai/using-ideogram/ideogram-features/ideogram-editor) and the ability to [use any image as the basis for a new one](https://docs.ideogram.ai/using-ideogram/ideogram-features/remix). There's a new Batch Generator that allows you to upload a spreadsheet with a list of prompts, and it's beta testing a canvas feature that allows for more complex designs. In my testing, it was up there with Midjourney in terms of quality.

Ideogram even has a free plan. With it, you're limited to 10 credits a week, you have to wait a few minutes for a generation to start, and you only get Ideogram's basic features, but it's still a great way to get a feel for one of the best AI image generators available.

**Ideogram pricing:** Limited free plan; from $8/month for full-resolution download and 400 monthly priority credits.

## Best AI image generator for customization and control

### [Stable Diffusion](https://stability.ai/)https://images.ctfassets.net/lzny33ho1g45/4Az7EJ5gtpVyQYpyk3J7AX/5077102edb0e3f841c0d3160aeae1bd0/image3.jpeg?

**Stable Diffusion pros:**

- Widely available across AI art generator platforms

- Affordable, customizable, and super powerful with generally great results


**Stable Diffusion cons:**

- The company behind it is maybe collapsing

- There's no one easy option for using it


Unlike Midjourney and Ideogram, [Stable Diffusion](https://zapier.com/blog/how-to-use-stable-diffusion) [has an open license](https://zapier.com/blog/open-source-ai/). This means anyone with the requisite technical skills can [download some versions of it](https://github.com/CompVis/stable-diffusion) and run them locally on their own computer. It also means that you can train and fine-tune the model for specific purposes. For the past couple of years, almost all the services that use AI to generate [artistic portraits](https://land.prisma-ai.com/magic-avatars/), [historical portraits](https://www.myheritage.com/ai-time-machine), [architectural renders](https://interiorai.com/), and everything else use Stable Diffusion this way.

But this kind of open setup can also mean chaos. And that's exactly what's happened with Stability.ai, the company that was formed by some of the researchers who developed Stable Diffusion. In 2024, it was [on the verge of collapse](https://futurism.com/the-byte/stability-ai-collapsing-considering-sale), its latest model and licensing terms [had been heavily criticized](https://www.zeniteq.com/blog/stability-ais-july-2024-license-update), and most of the research team had left to form a new company (which I'll talk about next).

While Stability AI seems to have weathered the crisis for now, all this puts Stable Diffusion in a weird place. The existing versions are still some of the best models available, there are countless fine-tuned versions that make it even better for specific uses, and it's wildly popular—but I'm not sure for how much longer any of this will remain true. The latest version, Stable Diffusion 3.5, is a great model, but it's not as popular or widely available as the earlier models.

The best (or at least most stable) way to use the most popular versions of Stable Diffusion is through an image generation tool like [NightCafe](https://creator.nightcafe.studio/), [Tensor.Art](https://tensor.art/), or [Civitai](https://civitai.com/)—though you can find [lots of other apps](https://zapier.com/blog/ai-art-generator/) that will give you access to it. A lot of these platforms even give you a few free credits so you can try it out before paying. One word of warning, though: some of these platforms don't have the kind of content moderation that's typical on larger social sites. You might see some weird and NSFW things.

If you want to avoid all that or have total control, you can always download Stable Diffusion and run it locally.

**Stable Diffusion pricing:** Depends on the platform, but many offer free credits so you can try them out.

**Read more:** [Stable Diffusion vs. DALL·E](https://zapier.com/blog/stable-diffusion-vs-dalle) and [Midjourney vs. Stable Diffusion](https://zapier.com/blog/midjourney-vs-stable-diffusion).

## Best Stable Diffusion alternative

### [**FLUX.1**](https://blackforestlabs.ai/)https://images.ctfassets.net/lzny33ho1g45/5xAzjYy11xVmiruodsWtSo/d7a171fd60b1ffd639829b40a87e8cc7/image5.jpg?

**FLUX.1 pros:**

- From the team behind Stable Diffusion—but without the drama

- Powerful and open


**FLUX.1 cons:**

- New and not as widely available as Stable Diffusion


As Stability.ai started collapsing, a significant portion of the team left the company to found [Black Forest Labs](https://blackforestlabs.ai/). Now, they've released their first series of text-to-image models: [FLUX.1](https://zapier.com/blog/flux-ai-image/).

In my testing, FLUX.1 is better than any version of Stable Diffusion that's widely available. It's also increasing in popularity and being embraced by the AI art community.

Right now, if you're looking to get into open AI image generation rather than just using one of the simpler text-to-image tools, I'd suggest experimenting with FLUX.1 over Stable Diffusion. FLUX.1 Schnell is released under an open Apache 2.0 license, while the larger FLUX.1 is open for non-commercial use.

Like Stable Diffusion, the simplest way to use FLUX.1 is through online AI art generators like NightCafe, Tensor.Art, and Civitai. Sign up for a free account, give it a go, and compare it side by side with some of the other models. But again, be warned that the content on these sites may not be entirely SFW.

**FLUX.1 pricing:** Depends on the platform, but many offer free credits so you can try them out.

## Best AI image generator for integrating AI-generated images into photos

### [Adobe Firefly](https://www.adobe.com/products/firefly/features/text-to-image.html)https://images.ctfassets.net/lzny33ho1g45/4awQwjmU6tXZ9zR8TvLFCm/3835c17c8b44c9a9396d57e77701fdd5/best-ai-image-generator-image2.jpeg?

**Adobe Firefly pros:**

- Integrates well with Adobe's apps, especially Photoshop

- Powerful when it's matching an image


**Adobe Firefly cons:**

- Not the best as a pure text-to-image model


Adobe has been building AI tools into its apps for more than 15 years, so it should be no surprise that it has one of the most powerful text-to-image generators—at least in terms of how it integrates with other tools. You can try its AI model, [Firefly](https://zapier.com/blog/adobe-firefly/), out on the web for free or through [Adobe Express](https://zapier.com/blog/adobe-express-ai), but it's at its best in the latest version of Photoshop.

Firefly has a few tricks up its sleeve. In addition to being capable of generating new images from a detailed text description, it can create text effects from a written prompt (think, the word "TOAST" written with letters that look like they're made from toast), recolor vector artwork, or add AI-generated elements to your images. You can test all these out through the web app, but it's that last feature where Firefly stands out.

Taken purely as a text-to-image generator, Firefly's results can be pretty hit and miss. It can match the best image generators like Midjourney for some prompts, but for others, I question what it was aiming to do. On the other hand, its integration with Photoshop, the industry standard image editor, is next level.

The two best features are Generative Fill and Generative Expand. With Generative Fill, you use Photoshop's regular tools to select an area of your image, and then, just by clicking a button and typing a prompt, you can replace it with something else. With Generative Expand, you can add to the outside of your image. Crucially, both tools understand the context of your image. In the screenshot above, you can see that Photoshop has matched the depth-of-field blur for the forest I added using Generative Fill. It looks cohesive.

As much as DALL·E and Stable Diffusion have started the conversation about image-generating AIs, Adobe's Firefly is the first implementation of an AI photo generator that really hints at what's to come. It isn't a party trick but a tool that's available to the millions of professionals who use Adobe apps every day.

**Firefly pricing:** Limited free credits; from $9.99 for Firefly Standard with 2,000 credits/month; Photoshop is available from $19.99/month as part of the Creative Cloud Photography Plan, which comes with 500 generative credits.

## The best AI image generator for graphic design

### [**Recraft**](https://www.recraft.ai/)https://images.ctfassets.net/lzny33ho1g45/3ZU5phnoABT9vgevnLFlcG/6eb9b110464e7ed161fe54816ef0f78c/image2.png?

**Recraft pros:**

- One of the most powerful and usable AI image generators

- Graphic design features are second to none


**Recraft cons:**

- More complicated to use than some of the other apps


Recraft is probably the most impressive app on this list. Its model is excellent and able to generate whatever you want, from photorealistic images to interesting logo designs. But it's the tools that Recraft has built around its model that really make it stand out.

Here's one example. Recraft allows you to create image sets that all fit the same style and color panel from a single set of prompts. You have all the style, color, and controls you need to dial things in, and it does an exceptional job right off the bat. Once you're happy with your images, you can export them as JPGs (fine), PNGs (better), or SVGs (amazing). Instead of being limited to small individual images, right from Recraft, you can create matching scalable design elements.

On top of that, you can use Recraft to create product mockups that combine multiple AI elements, in-paint and out-paint to add elements and combine images, adjust images and AI-generated work, remove backgrounds, and so much more. It's got collaboration tools, a great workspace, and you can export your work to other apps like Photoshop or Illustrator. It's a real continuation of what Adobe has done integrating Firefly into Photoshop.

**Recraft Pricing:** Free for 50 credits/day and limited features. From $12/month for Basic with 1,000 credits/month, commercial rights, and more artistic controls.

## Other AI image generators worth trying out

Over the past year, the overall standard of image generators has really improved. There are now a dozen different models that are almost equivalent in quality. I feel the seven above are the best choices for most people, but there are a handful of other apps that warrant mentioning:

- [Google Imagen 3](https://imagen.research.google/). Google's Imagen model is really great, and [if you already pay for Google Gemini](https://zapier.com/blog/gemini-for-workspace/), it's the first model you should look at.

- [Generative AI by Getty](https://www.gettyimages.ie/ai). Designed to generate commercially safe images, Generative AI by Getty is...fine. If you need images with zero commercial risk, it's worth a look—but the legal system doesn't seem to care about companies using images from Midjourney, Ideogram, or DALL·E.

- [Leonardo.Ai](http://leonardo.ai/). In addition to offering FLUX, image creation tool Leonardo.Ai has developed its own Phoenix model. It's a solid platform that just lacks a few features.

- [DALL·E 3](https://openai.com/index/dall-e-3/). DALL·E 3 is still available as a [GPT](https://zapier.com/blog/custom-chatgpt/). If you've got a soft spot for it, you can keep using it, but it's actually considered a legacy model now.

- [Luma Photon](https://lumalabs.ai/photon). Luma Photon is another great model, though I found the [Dream Machine](https://lumalabs.ai/dream-machine) app that uses it a bit too offbeat.

- [Playground](https://playground.com/design). Playground is great for creating designs, but its reliance on a template system meant I felt it was a little out of scope for the list.

- [MiniMax Image-01](https://www.minimax.io/news/image-01). Image-01 is doing really well in Artificial Analysis's leaderboards, though it's only available as an API. If you're a developer, it's worth a look.


If you want a laundry list of every AI image generator out there, including those that are built on top of all the models I've talked about, I made that too. It includes more than two dozen image generators: some are built into other tools, like [AI writing apps](https://zapier.com/blog/best-ai-writing-generator/), [photo editing apps](https://zapier.com/blog/best-ai-photo-editor/), or [stock photo sites](https://zapier.com/blog/best-free-stock-photos/); some let you [select from multiple models](https://zapier.com/blog/hugging-face/); and each one differs on how it approaches AI image generation. So if none of the apps on this list feel natural to you, check out my list of the [top AI art generators](https://zapier.com/blog/ai-art-generator/), and see if anything stands out.

## How to use an AI image generator

Ok, so you know what the best options are, but...now what? The team at Zapier has put together a bunch of resources to help you understand how to use these tools—and put them to work.

First, tutorials and walkthroughs for some of the best AI image generators:

- [How to use the ChatGPT image generator](https://zapier.com/blog/chatgpt-image-generation/)

- [How to use Midjourney](https://zapier.com/blog/how-to-use-midjourney/)

- [How to use Stable Diffusion](https://zapier.com/blog/how-to-use-stable-diffusion/)

- [How to use FLUX.1](https://zapier.com/blog/flux-ai-image/)

- [How to use Adobe Firefly](https://zapier.com/blog/adobe-firefly/)


Plus, a guide for [how to write effective AI art prompts](https://zapier.com/blog/ai-art-prompts/), so you can get what you're looking for faster (and better) when generating images.

Once you've got the basics down, it's time to use these tools for more than just creating wacky pictures. Here are some [tips for how to use AI image generators at work](https://zapier.com/blog/ai-image-examples-for-business/).

And finally, you can [automate your AI image generators](https://zapier.com/blog/automate-ai-images/), so they do their magic behind the scenes and connect to all the other apps you use.

## The legal and ethical implications of AI-generated images

AI-generated images are everywhere now, but that doesn't mean we shouldn't be asking questions about [how they should (or shouldn't) be used](https://zapier.com/blog/how-to-use-ai-badly/).

There aren't clear laws in place surrounding AI-generated images. And that goes for both sides of the coin: the U.S. Copyright Office [suggests that](https://fortune.com/2023/02/23/no-copyright-images-made-ai-artificial-intelligence/) AI-generated content isn't copyright-protected [without some kind of significant human input to the process](https://petapixel.com/2025/01/30/us-copyright-office-softens-its-stance-toward-registering-ai-generated-artworks/), and there aren't rules to protect artists whose work was scraped for AI training. (That's why Firefly was trained on licensed images and public domain content only.) They've [reaffirmed this stance](https://www.reuters.com/legal/litigation/us-copyright-office-denies-protection-another-ai-created-image-2023-09-06/), and the courts [have sided with their interpretation](https://www.reuters.com/world/us/us-appeals-court-rejects-copyrights-ai-generated-art-lacking-human-creator-2025-03-18/).

You're not likely to get into trouble for using AI-generated images for a few social media posts or blog hero images, but because there's no line drawn in the sand yet, it can be risky to develop an entire strategy around AI-generated art. (For what it's worth, [Hollywood seems to be quietly using it](https://nofilmschool.com/the-brutalist-ai).)

Then there's the [issue of bias](https://zapier.com/blog/ai-ethics/). As of now, AI has a lot of the same biases as humans, and that can lead to everything from the portrayal of stereotypes to harmful content. I experienced this myself with the outputs I got from some of the apps while testing them, though other tools take deliberate steps to add diversity to the images they generate. It's up to us as humans to avoid it by reviewing AI-generated content for bias and refining our prompts to eliminate that bias as much as possible.

## What's next for AI image generators?

AI image generating is a rapidly evolving space—and more powerful models are available each time I update this article. (I literally updated the article six weeks ago, and three new or upgraded image models dropped, so I had to update it again.) It's wild how good text-to-image models like GPT-4o, Reve, Midjourney, Ideogram, and FLUX.1 are getting at rendering tricky concepts repeatedly. While they're still a somewhat niche tool now, if they continue getting better at this pace, they could really shake things up.

</details>

<details>
<summary>understanding-multimodal-llms-by-sebastian-raschka-phd-1</summary>

# Understanding Multimodal LLMs

### An introduction to the main techniques and latest models

[https://substackcdn.com/image/fetch/$s_!Pq2z!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0d76dab1-362f-45b6-9b12-a12ac131edc5_1600x944.png](https://substackcdn.com/image/fetch/$s_!Pq2z!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0d76dab1-362f-45b6-9b12-a12ac131edc5_1600x944.png) _An illustration of a multimodal LLM that can accept different input modalities (audio, text, images, and videos) and returns text as the output modality._

* * *

# 1\. Use cases of multimodal LLMs

What are multimodal LLMs? As hinted at in the introduction, multimodal LLMs are large language models capable of processing multiple types of inputs, where each "modality" refers to a specific type of data—such as text (like in traditional LLMs), sound, images, videos, and more. For simplicity, we will primarily focus on the image modality alongside text inputs.

A classic and intuitive application of multimodal LLMs is image captioning: you provide an input image, and the model generates a description of the image, as shown in the figure below.

[https://substackcdn.com/image/fetch/$s_!8kaL!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F93884822-79f1-498d-a33a-8a367ba57134_1500x1222.png](https://substackcdn.com/image/fetch/$s_!8kaL!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F93884822-79f1-498d-a33a-8a367ba57134_1500x1222.png) _Example use of a multimodal LLM explaining [a meme](https://x.com/PainSci/status/1309570607458086914)._

Of course, there are many other use cases. For example, one of my favorites is extracting information from a PDF table and converting it into LaTeX or Markdown.

# 2\. Common approaches to building multimodal LLMs

There are two main approaches to building multimodal LLMs:

- Method A: Unified Embedding Decoder Architecture approach;

- Method B: Cross-modality Attention Architecture approach.


(By the way, I don’t believe official terms for these techniques exist yet, but let me know if you’ve come across any. For instance, briefer descriptions may be "decoder-only" and "cross-attention-based" approaches.)

[https://substackcdn.com/image/fetch/$s_!8miE!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F53956ae8-9cd8-474e-8c10-ef6bddb88164_1600x938.png](https://substackcdn.com/image/fetch/$s_!8miE!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F53956ae8-9cd8-474e-8c10-ef6bddb88164_1600x938.png) _The two main approaches to developing multimodal LLM architectures._

As shown in the figure above, the _**Unified Embedding-Decoder Architecture**_ utilizes a single decoder model, much like an unmodified LLM architecture such as GPT-2 or Llama 3.2. In this approach, images are converted into tokens with the same embedding size as the original text tokens, allowing the LLM to process both text and image input tokens together after concatenation.

The _**Cross-Modality Attention Architecture**_ employs a cross-attention mechanism to integrate image and text embeddings directly within the attention layer.

In the following sections, we will explore how these methods work on a conceptual level. Then, we will look at recent research papers on multimodal LLMs to see how they are applied in practice.

## **2.1 Method A: Unified Embedding Decoder Architecture**

Let’s begin with the unified embedding decoder architecture, illustrated again in the figure below.

[https://substackcdn.com/image/fetch/$s_!Ws6n!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F91955021-7da5-4bc4-840e-87d080152b18_1166x1400.png](https://substackcdn.com/image/fetch/$s_!Ws6n!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F91955021-7da5-4bc4-840e-87d080152b18_1166x1400.png) _Illustration of the unified embedding decoder architecture, which is an unmodified decoder-style LLM (like GPT-2, Phi-3, Gemma, or Llama 3.2) that receives inputs consisting of image token and text token embeddings._

In the unified embedding-decoder architecture, an image is converted into embedding vectors, similar to how input text is converted into embeddings in a standard text-only LLM.

For a typical text-only LLM that processes text, the text input is usually tokenized (e.g., using Byte-Pair Encoding) and then passed through an embedding layer, as shown in the figure below.

[https://substackcdn.com/image/fetch/$s_!dOba!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc97009dd-cee6-455f-87fe-64c33a868e9f_986x858.png](https://substackcdn.com/image/fetch/$s_!dOba!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc97009dd-cee6-455f-87fe-64c33a868e9f_986x858.png) _Illustration of the standard process for tokenizing text and converting it into token embedding vectors, which are subsequently passed to an LLM during training and inference._

### **2.1.1 Understanding Image encoders**

Analogous to the tokenization and embedding of text, image embeddings are generated using an image encoder module (instead of a tokenizer), as shown in the figure below.

[https://substackcdn.com/image/fetch/$s_!PlBh!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F15e9cc2f-95de-4723-9de5-9f2af7573aaa_790x750.png](https://substackcdn.com/image/fetch/$s_!PlBh!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F15e9cc2f-95de-4723-9de5-9f2af7573aaa_790x750.png) _Illustration of the process for encoding an image into image patch embeddings._

What happens inside the image encoder shown above? To process an image, we first divide it into smaller patches, much like breaking words into subwords during tokenization. These patches are then encoded by a pretrained vision transformer (ViT), as shown in the figure below.

[https://substackcdn.com/image/fetch/$s_!_DNf!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffef5f8cb-c76c-4c97-9771-7fdb87d7d8cd_1600x1135.png](https://substackcdn.com/image/fetch/$s_!_DNf!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffef5f8cb-c76c-4c97-9771-7fdb87d7d8cd_1600x1135.png) _Illustration of a classic vision transformer (ViT) setup, similar to the model proposed in [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (2020)._

Note that ViTs are often used for classification tasks, so I included the classification head in the figure above. However, in this case, we only need the image encoder part.

### **2.1.2 The role of the linear projection module**

The "linear projection" shown in the previous figure consists of a single linear layer (i.e., a fully connected layer). The purpose of this layer is to project the image patches, which are flattened into a vector, into an embedding size compatible with the transformer encoder. This linear projection is illustrated in the figure below. An image patch, flattened into a 256-dimensional vector, is up-projected to a 768-dimensional vector.

[https://substackcdn.com/image/fetch/$s_!i9i4!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fee32d720-92d7-48c2-b39d-adf61a870075_1600x681.png](https://substackcdn.com/image/fetch/$s_!i9i4!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fee32d720-92d7-48c2-b39d-adf61a870075_1600x681.png) _Illustration of a linear projection layer that projects flattened image patches from a 256-dimensional into a 768-dimensional embedding space._

For those who prefer seeing a code example, In PyTorch code, we could implement the linear projection for the image patches as follows:

```
import torch

class PatchProjectionLayer(torch.nn.Module):

    def __init__(self, patch_size, num_channels, embedding_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim
        self.projection = torch.nn.Linear(
            patch_size * patch_size * num_channels, embedding_dim
        )

    def forward(self, x):

        batch_size, num_patches, channels, height, width = x.size()
        x = x.view(batch_size, num_patches, -1)  # Flatten each patch
        x = self.projection(x)  # Project each flattened patch
        return x

# Example Usage:
batch_size = 1
num_patches = 9  # Total patches per image
patch_size = 16  # 16x16 pixels per patch
num_channels = 3  # RGB image
embedding_dim = 768  # Size of the embedding vector

projection_layer = PatchProjectionLayer(patch_size, num_channels, embedding_dim)

patches = torch.rand(
    batch_size, num_patches, num_channels, patch_size, patch_size
)

projected_embeddings = projection_layer(patches)
print(projected_embeddings.shape)

# This prints
# torch.Size([1, 9, 768])
```

If you have read my [Machine Learning Q and AI](https://www.amazon.com/Machine-Learning-AI-Essential-Questions/dp/1718503768/) book by chance, you may know there are ways to replace linear layers with convolution operations that can be implemented to be mathematically equivalent. Here, this can be especially handy as we can combine the creation of patches and projection into two lines of code:

```
layer = torch.nn.Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))

image = torch.rand(batch_size, 3, 48, 48)
projected_patches = layer(image)

print(projected_patches.flatten(-2).transpose(-1, -2).shape)
# This prints
# torch.Size([1, 9, 768])
```

### **2.1.3 Image vs text tokenization**

Now that we briefly discussed the purpose of the image encoder (and the linear projection that is part of the encoder), let's return to the text tokenization analogy from earlier and look at text and image tokenization and embedding side by side, as depicted in the figure below.

[https://substackcdn.com/image/fetch/$s_!zjmg!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0d56ea06-d202-4eb7-9e01-9aac492ee309_1522x1206.png](https://substackcdn.com/image/fetch/$s_!zjmg!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0d56ea06-d202-4eb7-9e01-9aac492ee309_1522x1206.png) _Image tokenization and embedding (left) and text tokenization and embedding (right) side by side._

As you can see in the figure above, I included an additional _**projector**_ module that follows the image encoder. This _projector_ is usually just another _**linear projection**_ layer that is similar to the one explained earlier. The purpose is to project the image encoder outputs into a dimension that matches the dimensions of the embedded text tokens, as illustrated in the figure below. (As we will see later, the projector is sometimes also called adapter, adaptor, or connector.)

[https://substackcdn.com/image/fetch/$s_!TaTW!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5d0be64c-da90-4193-86db-804f6a8a0abb_1542x1242.png](https://substackcdn.com/image/fetch/$s_!TaTW!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5d0be64c-da90-4193-86db-804f6a8a0abb_1542x1242.png) _Another side-by-side comparison between image tokenization and text tokenization, where the role of the projector is to match the text token embedding dimensions._

Now that the image patch embeddings have the same embedding dimension as the text token embeddings, we can simply concatenate them as input to the LLM, as shown in the figure at the beginning of this section. Below is the same figure again for easier reference.

[https://substackcdn.com/image/fetch/$s_!FTft!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa219f185-211b-4569-9398-2e080e2c5619_1166x1400.png](https://substackcdn.com/image/fetch/$s_!FTft!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa219f185-211b-4569-9398-2e080e2c5619_1166x1400.png) _After projecting the image patch tokens into the same dimension as the text token embeddings, we can simply concatenate them as input to a standard LLM._

By the way, the image encoder we discussed in this section is usually a pretrained vision transformer. A popular choice is [CLIP](https://github.com/openai/CLIP) or [OpenCLIP](https://github.com/mlfoundations/open_clip).

However, there are also versions of Method A that operate directly on patches, such as [Fuyu](https://www.adept.ai/blog/fuyu-8b), which is shown in the figure below.

[https://substackcdn.com/image/fetch/$s_!LB1L!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F28269d0d-b806-4ae7-bf96-b282affd7e93_1600x645.png](https://substackcdn.com/image/fetch/$s_!LB1L!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F28269d0d-b806-4ae7-bf96-b282affd7e93_1600x645.png) _Annotated figure of the Fuyu multimodal LLM that operates directly on the image patches without image encoder. (Annotated figure from [https://www.adept.ai/blog/fuyu-8b](https://www.adept.ai/blog/fuyu-8b).)_

As illustrated in the figure above, Fuyu passes the input patches directly into a linear projection (or embedding layer) to learn its own image patch embeddings rather than relying on an additional pretrained image encoder like other models and methods do. This greatly simplifies the architecture and training setup.

## **2.2 Method B: Cross-Modality Attention Architecture**

Now that we have discussed the unified embedding decoder architecture approach to building multimodal LLMs and understand the basic concept behind image encoding, let's talk about an alternative way of implementing multimodal LLMs via cross-attention, as summarized in the figure below.

[https://substackcdn.com/image/fetch/$s_!7Xvv!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd9c06055-b959-45d1-87b2-1f4e90ceaf2d_1296x1338.png](https://substackcdn.com/image/fetch/$s_!7Xvv!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd9c06055-b959-45d1-87b2-1f4e90ceaf2d_1296x1338.png) _An illustration of the Cross-Modality Attention Architecture approach to building multimodal LLMs._

In the Cross-Modality Attention Architecture method depicted in the figure above, we still use the same image encoder setup we discussed previously. However, instead of encoding the patches as input to the LLM, we connect the input patches in the multi-head attention layer via a cross-attention mechanism.

The idea is related and goes back to the original transformer architecture from the 2017 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper, highlighted in the figure below.

[https://substackcdn.com/image/fetch/$s_!JYyE!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5d028b95-7965-43e0-b8fc-350609a69377_1370x1582.png](https://substackcdn.com/image/fetch/$s_!JYyE!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5d028b95-7965-43e0-b8fc-350609a69377_1370x1582.png) _High-level illustration of the cross-attention mechanism used in the original transformer architecture. (Annotated figure from the "Attention Is All You Need" paper: https://arxiv.org/abs/1706.03762.)_

Note that the original "Attention Is All You Need" transformer depicted in the figure above was originally developed for language translation. So, it consists of a text **en** coder (left part of the figure) that takes the sentence to be translated and generates the translation via a text **de** coder (right part of the figure). In the context of multimodal LLM, the encoder is an image encoder instead of a text encoder, but the same idea applies.

How does cross-attention work? Let's have a look at a conceptual drawing of what happens inside the regular self-attention mechanism.

[https://substackcdn.com/image/fetch/$s_!HqoQ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff763532b-1eed-4f7d-ae2c-7783d4f4fc46_1440x1194.png](https://substackcdn.com/image/fetch/$s_!HqoQ!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff763532b-1eed-4f7d-ae2c-7783d4f4fc46_1440x1194.png) _Outline of the regular self-attention mechanism. (This flow depicts one of the heads in a regular multi-head attention module.)_

In the figure above, x is the input, and _Wq_ is a weight matrix used to generate the queries ( _Q_). Similarly, _K_ stands for keys, and _V_ stands for values. A represents the attention scores matrix, and _Z_ are the inputs (x) transformed into the output context vectors. (If this seems confusing, you may find a comprehensive introduction in Chapter 3 of my [Build a Large Language Model from Scratch book](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167/) helpful; alternatively, you may also find my article, [Understanding and Coding Self-Attention, Multi-Head Attention, Cross-Attention, and Causal-Attention in LLMs](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention) helpful here.)

In cross-attention, in contrast to self-attention, we have two different input sources, as illustrated in the following figure.

[https://substackcdn.com/image/fetch/$s_!3PZD!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffe4cc6f4-ca9a-431b-b572-95a1fda373a7_1508x1120.png](https://substackcdn.com/image/fetch/$s_!3PZD!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffe4cc6f4-ca9a-431b-b572-95a1fda373a7_1508x1120.png) _Illustration of cross attention, where there can be two different inputs x1 and x2_

As illustrated in the previous two figures, in self-attention, we work with the same input sequence. In cross-attention, we mix or combine two different input sequences.

In the case of the original transformer architecture in the _Attention Is All You Need_ paper, the two inputs _x1_ and _x2_ correspond to the sequence returned by the encoder module on the left ( _x2_) and the input sequence being processed by the decoder part on the right ( _x1_). In the context of a multimodal LLM, _x2_ is the output of an image encoder. (Note that the queries usually come from the decoder, and the keys and values typically come from the encoder.)

Note that in cross-attention, the two input sequences _x1_ and _x2_ can have different numbers of elements. However, their embedding dimensions must match. If we set _x1 = x2_, this is equivalent to self-attention.

# 3\. Unified decoder and cross-attention model training

Now that we have talked a bit about the two major multimodal design choices, let's briefly talk about how we deal with the three major components during model training, which are summarized in the figure below.

[https://substackcdn.com/image/fetch/$s_!e2P-!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F24a12032-d32e-41f6-b390-4e321e1ea29f_1600x770.png](https://substackcdn.com/image/fetch/$s_!e2P-!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F24a12032-d32e-41f6-b390-4e321e1ea29f_1600x770.png) _An overview of the different components in a multimodal LLM. The components numbered 1-3 can be frozen or unfrozen during the multimodal training process._

Similar to the development of traditional text-only LLMs, the training of multimodal LLMs also involves two phases: pretraining and instruction finetuning. However, unlike starting from scratch, multimodal LLM training typically begins with a pretrained, instruction-finetuned text-only LLM as the base model.

For the image encoder, CLIP is commonly used and often remains unchanged during the entire training process, though there are exceptions, as we will explore later. Keeping the LLM part frozen during the pretraining phase is also usual, focusing only on training the projector—a linear layer or a small multi-layer perceptron. Given the projector's limited learning capacity, usually comprising just one or two layers, the LLM is often unfrozen during multimodal instruction finetuning (stage 2) to allow for more comprehensive updates. However, note that in the cross-attention-based models (Method B), the cross-attention layers are unfrozen throughout the entire training process.

After introducing the two primary approaches (Method A: Unified Embedding Decoder Architecture and Method B: Cross-modality Attention Architecture), you might be wondering which is more effective. The answer depends on specific trade-offs.

The Unified Embedding Decoder Architecture (Method A) is typically easier to implement since it doesn't require any modifications to the LLM architecture itself.

The Cross-modality Attention Architecture (Method B) is often considered more computationally efficient because it doesn't overload the input context with additional image tokens, introducing them later in the cross-attention layers instead. Additionally, this approach maintains the text-only performance of the original LLM if the LLM parameters are kept frozen during training.

</details>

<details>
<summary>understanding-multimodal-llms-by-sebastian-raschka-phd</summary>

# Understanding Multimodal LLMs

### An introduction to the main techniques and latest models

[https://substackcdn.com/image/fetch/$s_!Pq2z!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0d76dab1-362f-45b6-9b12-a12ac131edc5_1600x944.png](https://substackcdn.com/image/fetch/$s_!Pq2z!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0d76dab1-362f-45b6-9b12-a12ac131edc5_1600x944.png) _An illustration of a multimodal LLM that can accept different input modalities (audio, text, images, and videos) and returns text as the output modality._

# 1. Use cases of multimodal LLMs

What are multimodal LLMs? As hinted at in the introduction, multimodal LLMs are large language models capable of processing multiple types of inputs, where each "modality" refers to a specific type of data—such as text (like in traditional LLMs), sound, images, videos, and more. For simplicity, we will primarily focus on the image modality alongside text inputs.

A classic and intuitive application of multimodal LLMs is image captioning: you provide an input image, and the model generates a description of the image, as shown in the figure below.

[https://substackcdn.com/image/fetch/$s_!8kaL!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F93884822-79f1-498d-a33a-8a367ba57134_1500x1222.png](https://substackcdn.com/image/fetch/$s_!8kaL!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F93884822-79f1-498d-a33a-8a367ba57134_1500x1222.png) _Example use of a multimodal LLM explaining [a meme](https://x.com/PainSci/status/1309570607458086914)._

Of course, there are many other use cases. For example, one of my favorites is extracting information from a PDF table and converting it into LaTeX or Markdown.

# 2. Common approaches to building multimodal LLMs

There are two main approaches to building multimodal LLMs:

- Method A: Unified Embedding Decoder Architecture approach;

- Method B: Cross-modality Attention Architecture approach.


(By the way, I don’t believe official terms for these techniques exist yet, but let me know if you’ve come across any. For instance, briefer descriptions may be "decoder-only" and "cross-attention-based" approaches.)

[https://substackcdn.com/image/fetch/$s_!8miE!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F53956ae8-9cd8-474e-8c10-ef6bddb88164_1600x938.png](https://substackcdn.com/image/fetch/$s_!8miE!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F53956ae8-9cd8-474e-8c10-ef6bddb88164_1600x938.png) _The two main approaches to developing multimodal LLM architectures._

As shown in the figure above, the _**Unified Embedding-Decoder Architecture**_ utilizes a single decoder model, much like an unmodified LLM architecture such as GPT-2 or Llama 3.2. In this approach, images are converted into tokens with the same embedding size as the original text tokens, allowing the LLM to process both text and image input tokens together after concatenation.

The _**Cross-Modality Attention Architecture**_ employs a cross-attention mechanism to integrate image and text embeddings directly within the attention layer.

In the following sections, we will explore how these methods work on a conceptual level. Then, we will look at recent research papers on multimodal LLMs to see how they are applied in practice.

## **2.1 Method A: Unified Embedding Decoder Architecture**

Let’s begin with the unified embedding decoder architecture, illustrated again in the figure below.

[https://substackcdn.com/image/fetch/$s_!Ws6n!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F91955021-7da5-4bc4-840e-87d080152b18_1166x1400.png](https://substackcdn.com/image/fetch/$s_!Ws6n!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F91955021-7da5-4bc4-840e-87d080152b18_1166x1400.png) _Illustration of the unified embedding decoder architecture, which is an unmodified decoder-style LLM (like GPT-2, Phi-3, Gemma, or Llama 3.2) that receives inputs consisting of image token and text token embeddings._

In the unified embedding-decoder architecture, an image is converted into embedding vectors, similar to how input text is converted into embeddings in a standard text-only LLM.

For a typical text-only LLM that processes text, the text input is usually tokenized (e.g., using Byte-Pair Encoding) and then passed through an embedding layer, as shown in the figure below.

[https://substackcdn.com/image/fetch/$s_!dOba!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc97009dd-cee6-455f-87fe-64c33a868e9f_986x858.png](https://substackcdn.com/image/fetch/$s_!dOba!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc97009dd-cee6-455f-87fe-64c33a868e9f_986x858.png) _Illustration of the standard process for tokenizing text and converting it into token embedding vectors, which are subsequently passed to an LLM during training and inference._

### **2.1.1 Understanding Image encoders**

Analogous to the tokenization and embedding of text, image embeddings are generated using an image encoder module (instead of a tokenizer), as shown in the figure below.

[https://substackcdn.com/image/fetch/$s_!PlBh!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F15e9cc2f-95de-4723-9de5-9f2af7573aaa_790x750.png](https://substackcdn.com/image/fetch/$s_!PlBh!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F15e9cc2f-95de-4723-9de5-9f2af7573aaa_790x750.png) _Illustration of the process for encoding an image into image patch embeddings._

What happens inside the image encoder shown above? To process an image, we first divide it into smaller patches, much like breaking words into subwords during tokenization. These patches are then encoded by a pretrained vision transformer (ViT), as shown in the figure below.

[https://substackcdn.com/image/fetch/$s_!_DNf!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffef5f8cb-c76c-4c97-9771-7fdb87d7d8cd_1600x1135.png](https://substackcdn.com/image/fetch/$s_!_DNf!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffef5f8cb-c76c-4c97-9771-7fdb87d7d8cd_1600x1135.png) _Illustration of a classic vision transformer (ViT) setup, similar to the model proposed in [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (2020)._

Note that ViTs are often used for classification tasks, so I included the classification head in the figure above. However, in this case, we only need the image encoder part.

### **2.1.2 The role of the linear projection module**

The "linear projection" shown in the previous figure consists of a single linear layer (i.e., a fully connected layer). The purpose of this layer is to project the image patches, which are flattened into a vector, into an embedding size compatible with the transformer encoder. This linear projection is illustrated in the figure below. An image patch, flattened into a 256-dimensional vector, is up-projected to a 768-dimensional vector.

[https://substackcdn.com/image/fetch/$s_!i9i4!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fee32d720-92d7-48c2-b39d-adf61a870075_1600x681.png](https://substackcdn.com/image/fetch/$s_!i9i4!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fee32d720-92d7-48c2-b39d-adf61a870075_1600x681.png) _Illustration of a linear projection layer that projects flattened image patches from a 256-dimensional into a 768-dimensional embedding space._

For those who prefer seeing a code example, In PyTorch code, we could implement the linear projection for the image patches as follows:

```
import torch

class PatchProjectionLayer(torch.nn.Module):

    def __init__(self, patch_size, num_channels, embedding_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim
        self.projection = torch.nn.Linear(
            patch_size * patch_size * num_channels, embedding_dim
        )

    def forward(self, x):

        batch_size, num_patches, channels, height, width = x.size()
        x = x.view(batch_size, num_patches, -1)  # Flatten each patch
        x = self.projection(x)  # Project each flattened patch
        return x

# Example Usage:
batch_size = 1
num_patches = 9  # Total patches per image
patch_size = 16  # 16x16 pixels per patch
num_channels = 3  # RGB image
embedding_dim = 768  # Size of the embedding vector

projection_layer = PatchProjectionLayer(patch_size, num_channels, embedding_dim)

patches = torch.rand(
    batch_size, num_patches, num_channels, patch_size, patch_size
)

projected_embeddings = projection_layer(patches)
print(projected_embeddings.shape)

# This prints
# torch.Size([1, 9, 768])
```

If you have read my [Machine Learning Q and AI](https://www.amazon.com/Machine-Learning-AI-Essential-Questions/dp/1718503768/) book by chance, you may know there are ways to replace linear layers with convolution operations that can be implemented to be mathematically equivalent. Here, this can be especially handy as we can combine the creation of patches and projection into two lines of code:

```
layer = torch.nn.Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))

image = torch.rand(batch_size, 3, 48, 48)
projected_patches = layer(image)

print(projected_patches.flatten(-2).transpose(-1, -2).shape)
# This prints
# torch.Size([1, 9, 768])
```

### **2.1.3 Image vs text tokenization**

Now that we briefly discussed the purpose of the image encoder (and the linear projection that is part of the encoder), let's return to the text tokenization analogy from earlier and look at text and image tokenization and embedding side by side, as depicted in the figure below.

[https://substackcdn.com/image/fetch/$s_!zjmg!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0d56ea06-d202-4eb7-9e01-9aac492ee309_1522x1206.png](https://substackcdn.com/image/fetch/$s_!zjmg!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0d56ea06-d202-4eb7-9e01-9aac492ee309_1522x1206.png) _Image tokenization and embedding (left) and text tokenization and embedding (right) side by side._

As you can see in the figure above, I included an additional _**projector**_ module that follows the image encoder. This _projector_ is usually just another _**linear projection**_ layer that is similar to the one explained earlier. The purpose is to project the image encoder outputs into a dimension that matches the dimensions of the embedded text tokens, as illustrated in the figure below. (As we will see later, the projector is sometimes also called adapter, adaptor, or connector.)

[https://substackcdn.com/image/fetch/$s_!TaTW!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5d0be64c-da90-4193-86db-804f6a8a0abb_1542x1242.png](https://substackcdn.com/image/fetch/$s_!TaTW!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5d0be64c-da90-4193-86db-804f6a8a0abb_1542x1242.png) _Another side-by-side comparison between image tokenization and text tokenization, where the role of the projector is to match the text token embedding dimensions._

Now that the image patch embeddings have the same embedding dimension as the text token embeddings, we can simply concatenate them as input to the LLM, as shown in the figure at the beginning of this section. Below is the same figure again for easier reference.

[https://substackcdn.com/image/fetch/$s_!FTft!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa219f185-211b-4569-9398-2e080e2c5619_1166x1400.png](https://substackcdn.com/image/fetch/$s_!FTft!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa219f185-211b-4569-9398-2e080e2c5619_1166x1400.png) _After projecting the image patch tokens into the same dimension as the text token embeddings, we can simply concatenate them as input to a standard LLM._

By the way, the image encoder we discussed in this section is usually a pretrained vision transformer. A popular choice is [CLIP](https://github.com/openai/CLIP) or [OpenCLIP](https://github.com/mlfoundations/open_clip).

However, there are also versions of Method A that operate directly on patches, such as [Fuyu](https://www.adept.ai/blog/fuyu-8b), which is shown in the figure below.

[https://substackcdn.com/image/fetch/$s_!LB1L!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F28269d0d-b806-4ae7-bf96-b282affd7e93_1600x645.png](https://substackcdn.com/image/fetch/$s_!LB1L!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F28269d0d-b806-4ae7-bf96-b282affd7e93_1600x645.png) _Annotated figure of the Fuyu multimodal LLM that operates directly on the image patches without image encoder. (Annotated figure from [https://www.adept.ai/blog/fuyu-8b](https://www.adept.ai/blog/fuyu-8b).)_

As illustrated in the figure above, Fuyu passes the input patches directly into a linear projection (or embedding layer) to learn its own image patch embeddings rather than relying on an additional pretrained image encoder like other models and methods do. This greatly simplifies the architecture and training setup.

## **2.2 Method B: Cross-Modality Attention Architecture**

Now that we have discussed the unified embedding decoder architecture approach to building multimodal LLMs and understand the basic concept behind image encoding, let's talk about an alternative way of implementing multimodal LLMs via cross-attention, as summarized in the figure below.

[https://substackcdn.com/image/fetch/$s_!7Xvv!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd9c06055-b959-45d1-87b2-1f4e90ceaf2d_1296x1338.png](https://substackcdn.com/image/fetch/$s_!7Xvv!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd9c06055-b959-45d1-87b2-1f4e90ceaf2d_1296x1338.png) _An illustration of the Cross-Modality Attention Architecture approach to building multimodal LLMs._

In the Cross-Modality Attention Architecture method depicted in the figure above, we still use the same image encoder setup we discussed previously. However, instead of encoding the patches as input to the LLM, we connect the input patches in the multi-head attention layer via a cross-attention mechanism.

The idea is related and goes back to the original transformer architecture from the 2017 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper, highlighted in the figure below.

[https://substackcdn.com/image/fetch/$s_!JYyE!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5d028b95-7965-43e0-b8fc-350609a69377_1370x1582.png](https://substackcdn.com/image/fetch/$s_!JYyE!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5d028b95-7965-43e0-b8fc-350609a69377_1370x1582.png) _High-level illustration of the cross-attention mechanism used in the original transformer architecture. (Annotated figure from the "Attention Is All You Need" paper: https://arxiv.org/abs/1706.03762.)_

Note that the original "Attention Is All You Need" transformer depicted in the figure above was originally developed for language translation. So, it consists of a text **en** coder (left part of the figure) that takes the sentence to be translated and generates the translation via a text **de** coder (right part of the figure). In the context of multimodal LLM, the encoder is an image encoder instead of a text encoder, but the same idea applies.

How does cross-attention work? Let's have a look at a conceptual drawing of what happens inside the regular self-attention mechanism.

[https://substackcdn.com/image/fetch/$s_!HqoQ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff763532b-1eed-4f7d-ae2c-7783d4f4fc46_1440x1194.png](https://substackcdn.com/image/fetch/$s_!HqoQ!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff763532b-1eed-4f7d-ae2c-7783d4f4fc46_1440x1194.png) _Outline of the regular self-attention mechanism. (This flow depicts one of the heads in a regular multi-head attention module.)_

In the figure above, x is the input, and _Wq_ is a weight matrix used to generate the queries ( _Q_). Similarly, _K_ stands for keys, and _V_ stands for values. A represents the attention scores matrix, and _Z_ are the inputs (x) transformed into the output context vectors. (If this seems confusing, you may find a comprehensive introduction in Chapter 3 of my [Build a Large Language Model from Scratch book](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167/) helpful; alternatively, you may also find my article, [Understanding and Coding Self-Attention, Multi-Head Attention, Cross-Attention, and Causal-Attention in LLMs](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention) helpful here.)

In cross-attention, in contrast to self-attention, we have two different input sources, as illustrated in the following figure.

[https://substackcdn.com/image/fetch/$s_!3PZD!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffe4cc6f4-ca9a-431b-b572-95a1fda373a7_1508x1120.png](https://substackcdn.com/image/fetch/$s_!3PZD!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffe4cc6f4-ca9a-431b-b572-95a1fda373a7_1508x1120.png) _Illustration of cross attention, where there can be two different inputs x1 and x2_

As illustrated in the previous two figures, in self-attention, we work with the same input sequence. In cross-attention, we mix or combine two different input sequences.

In the case of the original transformer architecture in the _Attention Is All You Need_ paper, the two inputs _x1_ and _x2_ correspond to the sequence returned by the encoder module on the left ( _x2_) and the input sequence being processed by the decoder part on the right ( _x1_). In the context of a multimodal LLM, _x2_ is the output of an image encoder. (Note that the queries usually come from the decoder, and the keys and values typically come from the encoder.)

Note that in cross-attention, the two input sequences _x1_ and _x2_ can have different numbers of elements. However, their embedding dimensions must match. If we set _x1 = x2_, this is equivalent to self-attention.

# 3. Unified decoder and cross-attention model training

Now that we have talked a bit about the two major multimodal design choices, let's briefly talk about how we deal with the three major components during model training, which are summarized in the figure below.

[https://substackcdn.com/image/fetch/$s_!e2P-!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F24a12032-d32e-41f6-b390-4e321e1ea29f_1600x770.png](https://substackcdn.com/image/fetch/$s_!e2P-!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F24a12032-d32e-41f6-b390-4e321e1ea29f_1600x770.png) _An overview of the different components in a multimodal LLM. The components numbered 1-3 can be frozen or unfrozen during the multimodal training process._

Similar to the development of traditional text-only LLMs, the training of multimodal LLMs also involves two phases: pretraining and instruction finetuning. However, unlike starting from scratch, multimodal LLM training typically begins with a pretrained, instruction-finetuned text-only LLM as the base model.

For the image encoder, CLIP is commonly used and often remains unchanged during the entire training process, though there are exceptions, as we will explore later. Keeping the LLM part frozen during the pretraining phase is also usual, focusing only on training the projector—a linear layer or a small multi-layer perceptron. Given the projector's limited learning capacity, usually comprising just one or two layers, the LLM is often unfrozen during multimodal instruction finetuning (stage 2) to allow for more comprehensive updates. However, note that in the cross-attention-based models (Method B), the cross-attention layers are unfrozen throughout the entire training process.

After introducing the two primary approaches (Method A: Unified Embedding Decoder Architecture and Method B: Cross-modality Attention Architecture), you might be wondering which is more effective. The answer depends on specific trade-offs.

The Unified Embedding Decoder Architecture (Method A) is typically easier to implement since it doesn't require any modifications to the LLM architecture itself.

The Cross-modality Attention Architecture (Method B) is often considered more computationally efficient because it doesn't overload the input context with additional image tokens, introducing them later in the cross-attention layers instead. Additionally, this approach maintains the text-only performance of the original LLM if the LLM parameters are kept frozen during training.

We will revisit the discussion on modeling performance and response quality in a later section, where we will discuss NVIDIA's NVLM paper.

This marks the end of what turned out to be a rather extensive introduction to multimodal LLMs. As I write this, I realize that the discussion has become lengthier than initially planned, which probably makes this a good place to conclude the article.

However, to provide a practical perspective, it would be nice to examine a few recent research papers that implement these approaches. So, we will explore these papers in the remaining sections of this article.

# 4. Recent multimodal models and methods

For the remainder of this article, I will review recent literature concerning multimodal LLMs, focusing specifically on works published in the last few weeks to maintain a reasonable scope.

Thus, this is not a historical overview or comprehensive review of multimodal LLMs but rather a brief look at the latest developments. I will also try to keep these summaries short and without too much fluff as there are 10 of them.

The conclusion section at the end of this has an overview that compares the methods used in these papers.

## **4.1 The Llama 3 Herd of Models**

_[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)_ paper (July 31, 2024) by Meta AI came out earlier this summer, which feels like ages ago in LLM terms. However, given that they only described but did not release their multimodal models until much later, I think it's fair to include Llama 3 in this list. (Llama 3.2 models were officially announced and made available on September 25.)

The multimodal Llama 3.2 models, which come in an 11-billion and 90-billion parameter version, are image-text models that use the previously described cross-attention-based approach, which is illustrated in the figure below.

[https://substackcdn.com/image/fetch/$s_!fTYU!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7c8578fa-70f2-474f-9e98-87621f2dce96_1600x833.png](https://substackcdn.com/image/fetch/$s_!fTYU!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7c8578fa-70f2-474f-9e98-87621f2dce96_1600x833.png) _Illustration of the multimodal LLM approach used by Llama 3.2. (Annotated figure from the Llama 3 paper: https://arxiv.org/abs/2407.21783.The video and speech parts are visually occluded to focus the attention on the image part.)_

Note that while the figure also depicts video and speech as possible modalities, the models that were released as of this writing focus only on image and text.

Llama 3.2 uses the cross-attention-based approach. However, it differs a bit from what I wrote about earlier, namely that in multimodal LLM development, we usually freeze the image encoder and only update the LLM parameters during pretraining.

Here, the researchers almost take the opposite approach: they update the image encoder but do not update the language model's parameters. They write that this is intentional and done to preserve the text-only capabilities so that the 11B and 90B multimodal models can be used as drop-in replacements for the Llama 3.1 8B and 70B text-only model on text tasks.

The training itself is done in multiple iterations, starting with the Llama 3.1 text models. After adding the image encoder and projection (here called "adapter") layers, they pretrain the model on image-text data. Then, similar to the Llama 3 model text-only training (I wrote about it in [an earlier article](https://magazine.sebastianraschka.com/i/147749119/llama-overview)), they follow up with instruction and preference finetuning.

Instead of adopting a pretrained model such as CLIP as an image encoder, the researchers used a vision transformer that they pretrained from scratch. Specifically, they adopted the  ViT-H/14 variant (630 million parameters) of the classic vision transformer architecture ( [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)). They then pretrained the ViT on a dataset of 2.5 billion image-text pairs over five epochs; this was done before connecting the image encoder to the LLM. (The image encoder takes 224×224 resolution images and divides them into a 14×14 grid of patches, with each patch sized at 16×16 pixels.)

As the cross-attention layers add a substantial amount of parameters, they are only added in every fourth transformer block. (For the 8B model, this adds 3B parameters, and for the 70B model, this adds 20 billion parameters.)

## **4.2 Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models**

_[The Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models](https://www.arxiv.org/abs/2409.17146)_ paper (September 25, 2024) is notable because it promises to open source not only the model weights but also the dataset and source code similar to the language-only OLMo LLM. (This is great for LLM research as it allows us to take a look at the exact training procedure and code and also lets us run ablation studies and reproduce results on the same dataset.)

If you are wondering why there are two names in the paper title, Molmo refers to the model (Multimodal Open Language Model), and PixMo (Pixels for Molmo) is the dataset.

[https://substackcdn.com/image/fetch/$s_!9P0w!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F73337002-8feb-4f1b-a109-1407096e32c5_1104x704.png](https://substackcdn.com/image/fetch/$s_!9P0w!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F73337002-8feb-4f1b-a109-1407096e32c5_1104x704.png) _Illustration of the Molmo decoder-only approach (Method A). Annotated figure adapted from the Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models paper: https://www.arxiv.org/abs/2409.17146._

As illustrated in the figure above, the image encoder employs an off-the-shelf vision transformer, specifically CLIP. The term "connector" here refers to a "projector" that aligns image features with the language model.

Molmo streamlines the training process by avoiding multiple pretraining stages, choosing instead a simple pipeline that updates all parameters in a unified approach—including those of the base LLM, the connector, and the image encoder.

The Molmo team offers several options for the base LLM:

- OLMo-7B-1024 (a fully open model backbone),

- OLMoE-1B-7B (a mixture-of-experts architecture; the most efficient model),

- Qwen2 7B (an open-weight model that performs better than OLMo-7B-1024),

- Qwen2 72B (an open-weight model and the best-performing model)


## **4.3 NVLM: Open Frontier-Class Multimodal LLMs**

NVIDIA's _[NVLM: Open Frontier-Class Multimodal LLMs](https://arxiv.org/abs/2409.11402)_ paper (September 17, 2024) is particularly interesting because, rather than focusing on a single approach, it explores both methods:

- Method A, the Unified Embedding Decoder Architecture ("decoder-only architecture," NVLM-D), and

- Method B, the Cross-Modality Attention Architecture ("cross-attention-based architecture," NVLM-X).


Additionally, they develop a hybrid approach (NVLM-H) and provide an apples-to-apples comparison of all three methods.

[https://substackcdn.com/image/fetch/$s_!6n6Y!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F45916952-b1ee-4972-a956-e45703e3fe36_1600x927.png](https://substackcdn.com/image/fetch/$s_!6n6Y!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F45916952-b1ee-4972-a956-e45703e3fe36_1600x927.png) _Overview of the three multimodal approaches. (Annotated figure from the NVLM: Open Frontier-Class Multimodal LLMs paper: https://arxiv.org/abs/2409.11402)_

As summarized in the figure below, NVLM-D corresponds to Method A, and NVLM-X corresponds to Method B, as discussed earlier. The concept behind the hybrid model (NVLM-H) is to combine the strengths of both methods: an image thumbnail is provided as input, followed by a dynamic number of patches passed through cross-attention to capture finer high-resolution details.

In short, the research team find that:

- NVLM-X demonstrates superior computational efficiency for high-resolution images.

- NVLM-D achieves higher accuracy in OCR-related tasks.

- NVLM-H combines the advantages of both methods.


Similar to Molmo and other approaches, they begin with a text-only LLM rather than pretraining a multimodal model from scratch (as this generally performs better). Additionally, they use an instruction-tuned LLM instead of a base LLM. Specifically, the backbone LLM is Qwen2-72B-Instruct (to my knowledge, Molmo used the Qwen2-72B base model).

While training all LLM parameters in the NVLM-D approach, they found that for NVLM-X, it works well to freeze the original LLM parameters and train only the cross-attention layers during both pretraining and instruction finetuning.

For the image encoder, instead of using a typical CLIP model, they use [InternViT-6B](https://arxiv.org/abs/2312.14238), which remains frozen throughout all stages.

The projector is a multilayer perceptron rather than a single linear layer.

## **4.4 Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution**

The previous two papers and models, Molmo and NVLM, were based on Qwen2-72B LLM. In this paper, the Qwen research team itself announces a multimodal LLM, _[Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)_ (October 3rd, 2024).

At the core of this work is their so-called "Naive Dynamic Resolution" mechanism (the term "naive" is intentional and not a typo for "native," though "native" could also be fitting). This mechanism allows the model to handle images of varying resolutions without simple downsampling, enabling the input of images in their original resolution.

[https://substackcdn.com/image/fetch/$s_!Zrt8!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2247e684-253a-462e-afb4-549411d5741a_1490x1068.png](https://substackcdn.com/image/fetch/$s_!Zrt8!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2247e684-253a-462e-afb4-549411d5741a_1490x1068.png) _An overview of the multimodal Qwen model, which can process input images with various different resolutions natively. (Annotated figure from the Qwen2-VL paper: https://arxiv.org/abs/2409.12191)_

The native resolution input is implemented via a modified ViT by removing the original absolute position embeddings and introducing 2D-RoPE.

They used a classic vision encoder with 675M parameters and LLM backbones of varying sizes, as shown in the table below.

[https://substackcdn.com/image/fetch/$s_!NdAJ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2ce9ce4a-d7ec-476d-91cb-29b6f5440b3b_1396x482.png](https://substackcdn.com/image/fetch/$s_!NdAJ!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2ce9ce4a-d7ec-476d-91cb-29b6f5440b3b_1396x482.png) The components of the different Qwen2-VL models. (Annotated figure from the Qwen2-VL paper: https://arxiv.org/abs/2409.12191)

The training itself consists of 3 stages: (1) pretraining only the image encoder, (2) unfreezing all parameters (including LLM), and (3) freezing the image encoder and instruction-finetuning only the LLM.

## **4.5 Pixtral 12B**

_[Pixtral 12B](https://mistral.ai/news/pixtral-12b/)_ (September 17, 2024), which uses the Method A: Unified Embedding Decoder Architecture approach, is the first multimodal model from Mistral AI. Unfortunately, there is no technical paper or report available, but the Mistral team shared a few interesting tidbits in their [blog post](https://mistral.ai/news/pixtral-12b/).

Interestingly, they chose not to use a pretrained image encoder, instead training one with 400 million parameters from scratch. For the LLM backbone, they used the 12-billion-parameter [Mistral NeMo](https://mistral.ai/news/mistral-nemo/) model.

Similar to Qwen2-VL, Pixtral also supports variable image sizes natively, as illustrated in the figure below.

[https://substackcdn.com/image/fetch/$s_!eW3C!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F37bb0f12-4533-4f44-8907-1da868006ff3_1144x726.png](https://substackcdn.com/image/fetch/$s_!eW3C!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F37bb0f12-4533-4f44-8907-1da868006ff3_1144x726.png) _Illustration of how Pixtral processes images of different sizes. (Annotated figure from the Pixtral blog  post: https://mistral.ai/news/pixtral-12b/)_

## **4.6 MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning**

The _[MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning](https://arxiv.org/abs/2409.20566)_ paper (September 30, 2024) provides practical tips and introduces a mixture-of-experts multimodal model alongside a dense model similar to Molmo. The models span a wide size range, from 1 billion to 30 billion parameters.

The models described in this paper focuse on Method A, a Unified Embedding Transformer Architecture, which structures inputs effectively for multimodal learning.

In addition, the paper has a series of interesting ablation studies looking into data mixtures and the effects of using coordinate tokens.

[https://substackcdn.com/image/fetch/$s_!fMsE!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F71b22b97-e901-4c5f-a9c2-67e32c867823_1402x1178.png](https://substackcdn.com/image/fetch/$s_!fMsE!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F71b22b97-e901-4c5f-a9c2-67e32c867823_1402x1178.png) _Illustration of the MM1.5 approach, which includes additional coordinate tokens to denote bounding boxes. (Annotated figure from the MM1.5 paper: https://arxiv.org/abs/2409.20566.)_

## **4.7 Aria: An Open Multimodal Native Mixture-of-Experts Model**

The _[Aria: An Open Multimodal Native Mixture-of-Experts Model](https://arxiv.org/abs/2410.05993)_ paper (October 8, 2024) introduces another mixture-of-experts model approach, similar to one of the variants in the Molmo and MM1.5 lineups.

The Aria model has 24.9 billion parameters, with 3.5 billion parameters allocated per text token. The image encoder ( [SigLIP](https://arxiv.org/abs/2303.15343)) has 438-million-parameters.

This model is based on a cross-attention approach with the following overall training procedure:

1. Training the LLM backbone entirely from scratch.

2. Pretraining both the LLM backbone and the vision encoder.


## **4.8 Baichuan-Omni**

The _[Baichuan-Omni Technical Report](https://arxiv.org/abs/2410.08565)_ (October 11, 2024) introduces Baichuan-Omni, a 7-billion-parameter multimodal LLM based on Method A: the Unified Embedding Decoder Architecture approach, as shown in the figure below.

[https://substackcdn.com/image/fetch/$s_!-IYi!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F142c39bd-2d3f-4813-9363-5ecf616cb784_2102x1326.png](https://substackcdn.com/image/fetch/$s_!-IYi!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F142c39bd-2d3f-4813-9363-5ecf616cb784_2102x1326.png) _An overview of the Baichuan-Omni model, which can handle various input modalities. (Annotated figure from the Baichuan-Omni paper: https://arxiv.org/abs/2410.08565)_

The training process for Baichuan-Omni involves a three-stage approach:

1. **Projector training**: Initially, only the projector is trained, while both the vision encoder and the language model (LLM) remain frozen.

2. **Vision encoder training**: Next, the vision encoder is unfrozen and trained, with the LLM still frozen.

3. **Full model training**: Finally, the LLM is unfrozen, allowing the entire model to be trained end-to-end.


The model utilizes the SigLIP vision encoder and incorporates the [AnyRes](https://arxiv.org/abs/2204.07156) module to handle high-resolution images through down-sampling techniques.

While the report does not explicitly specify the LLM backbone, it is likely based on the Baichuan 7B LLM, given the model's parameter size and the naming convention.

## **4.9 Emu3: Next-Token Prediction is All You Need**

The _Emu3: Next-Token Prediction is All You Need_ paper (September 27, 2024) presents a compelling alternative to diffusion models for image generation, which is solely based on a transformer-based decoder architecture. Although it's not a multimodal LLM in the classic sense (i.e., models focused on image understanding rather than generation), Emu3 is super interesting as it demonstrates that it's possible to use transformer decoders for image generation, which is a task typically dominated by diffusion methods. (However, note that there have been other similar approaches before, such as [Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation](https://arxiv.org/abs/2406.06525).)

[https://substackcdn.com/image/fetch/$s_!IWU7!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F775db9c7-662f-4314-a5c4-c3f5efe0238d_1056x904.png](https://substackcdn.com/image/fetch/$s_!IWU7!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F775db9c7-662f-4314-a5c4-c3f5efe0238d_1056x904.png) _Emu3 is primarily an LLM for image generation as an alternative to diffusion models. (Annotated figure from the Emu3 paper: https://arxiv.org/abs/2409.18869)_

The researchers trained Emu3 from scratch and then used [Direct Preference Optimization](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb) (DPO) to align the model with human preferences.

The architecture includes a vision tokenizer inspired by [SBER-MoVQGAN](https://arxiv.org/abs/2209.09002). The core LLM architecture is based on Llama 2, yet it is trained entirely from scratch.

## **4.10 Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation**

We previously focused on multimodal LLMs for image understanding and just saw one example for image generation with Emu 3 above. Now, the _[Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2410.13848)_ paper (October 17, 2024) introduces a framework that unifies multimodal understanding and generation tasks within a single LLM backbone.

A key feature of Janus is the decoupling of visual encoding pathways to address the distinct requirements of understanding and generation tasks. The researchers argue that image understanding tasks require high-dimensional semantic representations, while generation tasks require detailed local information and global consistency in images. By separating these pathways, Janus effectively manages these differing needs.

The model employs the SigLIP vision encoder, similar to that used in Baichuan-Omni, for processing visual inputs. For image generation, it utilizes a [Vector Quantized (VQ)](https://arxiv.org/abs/2406.06525) tokenizer to handle the generation process. The base LLM in Janus is the [DeepSeek-LLM](https://arxiv.org/abs/2401.02954) with 1.3 billion parameters.

[https://substackcdn.com/image/fetch/$s_!9UFg!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F89d62626-4386-4e73-8992-158550752ce2_1434x692.png](https://substackcdn.com/image/fetch/$s_!9UFg!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F89d62626-4386-4e73-8992-158550752ce2_1434x692.png) _An overview of the unified decoder-only framework used in Janus. (Annotated figure from the Janus paper: https://arxiv.org/abs/2410.13848.)_

The training process for the model in this image follows three stages, as shown in the figure below.

[https://substackcdn.com/image/fetch/$s_!Da5n!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2fb4f079-0771-4d21-8805-fded73134983_1536x648.png](https://substackcdn.com/image/fetch/$s_!Da5n!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2fb4f079-0771-4d21-8805-fded73134983_1536x648.png) Illustration of the 3-stage training process of the Janus model. (Annotated figure from the Janus paper: https://arxiv.org/abs/2410.13848)

In Stage I, only the projector layers and image output layer are trained while the LLM, understanding, and generation encoders remain frozen. In Stage II, the LLM backbone and text output layer are unfrozen, allowing for unified pretraining across understanding and generation tasks. Finally, in Stage III, the entire model, including the SigLIP image encoder, is unfrozen for supervised fine-tuning, enabling the model to fully integrate and refine its multimodal capabilities.

# Conclusion

As you may have noticed, I almost entirely skipped both the modeling and the computational performance comparisons. First, comparing the performance of LLMs and multimodal LLMs on public benchmarks is challenging due to prevalent data contamination, meaning that the test data may have been included in the training data.

Additionally, the architectural components vary so much that making an apples-to-apples comparison is difficult. So, big kudos to the NVIDIA team for developing NVLM in different flavors, which allowed for a comparison between the decoder-only and cross-attention approaches at least.

In any case, the main takeaway from this article is that multimodal LLMs can be built successfully in many different ways. Below is a figure that summarizes the different components of the models covered in this article.

[https://substackcdn.com/image/fetch/$s_!R_9Y!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb043e6d7-78e5-4628-987a-b333d3a58829_2224x1180.png](https://substackcdn.com/image/fetch/$s_!R_9Y!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb043e6d7-78e5-4628-987a-b333d3a58829_2224x1180.png) An overview of the different models covered in this article along with their subcomponents and training approaches.

I hope you found reading this article educational and now have a better understanding of how multimodal LLMs work

</details>

<details>
<summary>what-are-some-real-world-applications-of-multimodal-ai</summary>

# What are some real-world applications of multimodal AI?

Multimodal AI, which processes and combines different data types like text, images, audio, and sensor inputs, has practical applications across industries. By integrating multiple data sources, these systems improve accuracy and functionality in tasks that require contextual understanding. Below are three key areas where multimodal AI is being applied effectively today.

In healthcare, multimodal AI enhances diagnostics and patient care by merging medical imaging, electronic health records (EHRs), and sensor data. For example, a system might analyze a chest X-ray (image), a patient’s symptom descriptions (text), and vital signs from wearables (sensor data) to detect pneumonia. Models like Google’s **Med-PaLM 2** combine vision and language processing to interpret radiology images alongside clinical notes, reducing misdiagnosis risks. Another use case is monitoring postoperative recovery: wearable devices track movement and heart rate, while speech analysis detects pain or fatigue in a patient’s voice, enabling proactive interventions.

Autonomous vehicles rely heavily on multimodal AI to fuse data from cameras, LiDAR, radar, and GPS. A self-driving car processes road signs (visual data), pedestrian movements (video), and proximity sensor readings to navigate safely. Tesla’s Autopilot, for instance, uses neural networks to combine camera feeds with ultrasonic sensors, improving object detection in varied lighting or weather. Similarly, companies like Waymo train models to correlate map data with real-time sensor inputs, ensuring precise localization and path planning. This redundancy across modalities helps address limitations of single-sensor systems, such as camera failures in low light.

Customer service and content moderation also benefit from multimodal approaches. Virtual assistants like Amazon’s Alexa process voice commands while analyzing user history (text) to personalize responses. In moderation, platforms like YouTube use AI to flag harmful content by scanning video frames (images), audio for hate speech, and user comments (text) simultaneously. For example, a post containing violent imagery and threatening text would be detected faster than if each modality were analyzed separately. Tools like **OpenAI’s CLIP** enable cross-modal matching, such as linking inappropriate images to their descriptive captions, improving accuracy in filtering violations. These systems reduce reliance on manual review while scaling to handle large data volumes.

</details>

<details>
<summary>what-are-vision-language-models-nvidia-glossary</summary>

Vision language models (VLMs) are multimodal, generative AI models capable of understanding and processing video, image, and text.

## What Are Vision Language Models?

Vision language models are multimodal AI systems built by combining a large language model (LLM) with a vision encoder, giving the LLM the ability to “see.”

With this ability, VLMs can process and provide advanced understanding of video, image, and text inputs supplied in the prompt to generate text responses.https://www.nvidia.com/content/nvidiaGDC/us/en_US/glossary/vision-language-models/_jcr_content/root/responsivegrid/nv_container_copy/nv_image.coreimg.100.1290.png/1736201901571/metropolis-iva-diagram-vlm-glossary-ces25-3576177-r1--1-.png

Figure 1: Use cases for vision language models

Unlike traditional [computer vision](https://www.nvidia.com/en-us/glossary/computer-vision/) models, VLMs are not bound by a fixed set of classes or a specific task like classification or detection. Retrained on a vast corpus of text and image/video-caption pairs, VLMs can be instructed in natural language and used to handle many classic vision tasks plus new generative AI-powered tasks such as summarization and visual Q&A.

## Why Are Vision Language Models Important?

To understand the importance of VLMs, it’s helpful to know how past computer vision (CV) models work. Traditional convolutional neural network ( [CNN](https://www.nvidia.com/en-us/glossary/convolutional-neural-network/))-based CV models are trained for a specific task on a bounded set of classes. For example:

- A classification model that identifies whether an image contains a cat or a dog
- An optical character detection and recognition CV model that reads text in an image but doesn’t interpret the format or any visual data within a document

Previous CV models were trained for a specific purpose and did not have the ability to go beyond the task or set of classes they were developed for and trained on. If the use case changed at all or required a new class to be added to the model, a developer would have to collect and label a large number of images and retrain the model. This is an expensive, time-consuming process. Additionally, CV models don't have any natural language understanding.

VLMs bring a new class of capabilities by combining the power of foundation models, like [CLIP](https://github.com/openai/CLIP), and LLMs to have both vision and language capabilities. Out of the box, VLMs have strong zero-shot performance on a variety of vision tasks, like visual question-answering, classification, and optical character recognition. They are also extremely flexible and can be used not just on a fixed set of classes but for nearly any use case by simply changing a text prompt.

Using a VLM is very similar to interacting with an LLM. The user supplies text prompts that can be interleaved with images. The inputs are then used to generate text output. The input prompts are open-ended, allowing the user to instruct the VLM to answer questions, summarize, explain the content, or reason with the image. Users can chat back and forth with the VLM, with the ability to add images into the context of the conversation. VLMs can also be integrated into visual agents to autonomously perform vision tasks.

## How Do Vision Language Models Work?

Most VLMs follow an architecture with three parts:

- A vision encoder
- A projector
- An LLM

The vision encoder is typically a CLIP-based model with a transformer architecture that has been trained on millions of image-text pairs, giving it the ability to associate images and text. The projector is a set of layers that translates the output of the vision encoder into a form the LLM can understand, often interpreted as image tokens. This projector can be a simple line layer like LLaVA and VILA, or something more complex like the cross-attention layers used in Llama 3.2 Vision.

Any off-the-shelf LLM can be used to build a VLM. There are hundreds of VLM variants that combine various LLMs with vision encoders.https://www.nvidia.com/content/nvidiaGDC/us/en_US/glossary/vision-language-models/_jcr_content/root/responsivegrid/nv_container_copy_co_300503066/nv_image.coreimg.svg/1736168815674/vlm-architecture-diagram.svg

Figure 2: A common three-part architecture for vision language models

## How Are Vision Language Models Trained?

VLMs are trained in several stages that include pretraining, followed by supervised fine-tuning. Optionally, parameter efficient fine-tuning (PEFT) can be applied as a final stage to create a domain-specific VLM on custom data.

The pretraining stage aligns the vision encoder, projector, and LLM to essentially speak the same language when interpreting the text and image input. This is done using large corpora of text and images with image-caption pairs and interleaved image-text data. Once the three components have been aligned through pretraining, the VLM goes through a supervised fine-tuning stage to help it understand how to respond to user prompts.

The data used in this stage are a blend of example prompts with text and/or image input and the expected response of the model. For example, this data could be prompts telling the model to describe the image or to count all the objects in the frame with the expected correct response. After this round of training, the VLM will understand how to best interpret images and respond to user prompts.https://www.nvidia.com/content/nvidiaGDC/us/en_US/glossary/vision-language-models/_jcr_content/root/responsivegrid/nv_container_copy_co_1755415045/nv_image.coreimg.svg/1736168816034/vlm-training-process-diagram.svg

Figure 3: Training for VLMs is often done in several stages to target certain parts of the model

Once the VLM is trained, it can be used in the same way as an LLM by providing prompts that can also include images interleaved in text. The VLM will then generate a text response based on the inputs. VLMs are typically deployed with an OpenAI style REST API interface to make it easy to interact with the model.

More advanced techniques are currently being researched to enhance vision capabilities:

- Ensembling vision encoders to process image inputs
- Breaking apart high-resolution image inputs into smaller tiles for processing
- Increasing context length to improve long video understanding

All of these advancements are progressing the capabilities of VLMs from only understanding single-image input to being highly capable models that can compare and contrast images, accurately read text, understand long videos, and have strong spatial understanding.

## How Are Vision Language Models Benchmarked?

Several common benchmarks, such [MMMU](https://mmmu-benchmark.github.io/), [Video-MME](https://video-mme.github.io/home_page.html), [MathVista](https://mathvista.github.io/), [ChartQA](https://github.com/vis-nlp/ChartQA) , and [DocVQA](https://www.docvqa.org/), exist to determine how well vision-language models perform on a variety of tasks, such as:

- Visual question-answering
- Logic and reasoning
- Document understanding
- Multi-image comparisons
- Video understanding

Most benchmarks consist of a set of images with several associated questions, often posed as multiple-choice questions. The multiple-choice format is the easiest way to consistently benchmark and compare VLMs. These questions test the VLMs perception, knowledge, and reasoning capabilities. When running these benchmarks, the VLM is provided with the image, question, and several multiple-choice answers it must choose from.https://www.nvidia.com/content/nvidiaGDC/us/en_US/glossary/vision-language-models/_jcr_content/root/responsivegrid/nv_container_copy_co_42410027/nv_image.coreimg.100.1290.jpeg/1736168816436/vlm-mmmu-ari.jpeg

Figure 4: Example multiple-choice questions for VLMs used in the MMMU benchmark

Source ( [MMMU](https://mmmu-benchmark.github.io/))

The accuracy of the VLM is the number of correct choices over the set of multiple-choice questions. Some benchmarks also include numerical questions where the VLM must perform a specific calculation and be within a certain percentage of the answer to be considered correct. Often these questions and images come from academic sources, such as college-level textbooks.

## How Are Vision Language Models Used?

VLMs are quickly becoming the go-to tool for all types of vision-related tasks due to their flexibility and natural language understanding. VLMs can be easily instructed to perform a wide variety of tasks through natural language:

1. Visual questions-answering
2. Image and video summarization
3. Parsing text and handwritten documents

Previous applications that would have required a large ensemble of specially trained models can now be accomplished with just a single VLM.

VLMs are especially good at summarizing the contents of images and can be prompted to perform specific tasks based on the contents. Take for example, an education use case—a VLM could be given an image of a handwritten math problem, and it could use its optical character recognition and reasoning capabilities to interpret the problem and produce a step-by-step guide on how to solve it. VLMs can not only understand the content of the image but also reason and perform specific tasks.https://www.nvidia.com/content/nvidiaGDC/us/en_US/glossary/vision-language-models/_jcr_content/root/responsivegrid/nv_container_copy_co_531349501/nv_image.coreimg.svg/1736168816834/vlm-real-world-diagram.svg

Figure 5: video analytics AI agents transform video and image data into real-world insights

With vast amounts of video being produced every day, it's infeasible to review and extract insights from this volume of video that is produced by all industries. VLMs can be integrated into a larger system to build video analytics AI agents capable of detecting specific events when prompted. These systems could be used to detect malfunctioning robots in a warehouse or generate out-of-stock alerts when shelves are empty. Their general understanding goes beyond simple detection and could be used to generate automated reports. For example, an intelligent traffic system could detect, analyze, and produce reports of traffic hazards, such as fallen trees, stalled vehicles, or collisions.

VLMs can be used with technologies like graph databases to understand long videos. This helps them capture the complexity of objects and events in a video. Such systems could be used to summarize operations in a warehouse to find bottlenecks and inefficiencies or produce sports commentary for football, basketball, or soccer games.

## What Are the Challenges of Vision Language Models?

Vision language models are maturing quickly, but they still have some limitations, particularly around spatial understanding and long-context video understanding.

Most VLMs use CLIP-based models as the vision encoder, which are limited to 224x224 or 336x336 image input size. This relatively small input image makes it difficult for small objects and details to be detected. For example, an HD 1080x1920 frame from a video must be downsized or cropped to a much smaller input resolution, making it difficult to retain details for small objects or fine details. To fix this, VLMs are starting to use tiling methods that allow a big image to be broken into smaller pieces and then fed into the model. There's also ongoing research to explore the use of higher-resolution image encoders.

VLMs also have difficulty providing precise locations for objects. The training data for CLIP-based vision encoders consists mostly of short text descriptions of images, like captions. These descriptions don't include detailed, fine-grained object locations, and this limitation impacts CLIP’s spatial understanding. This is inherited by VLMs that use it as a vision encoder. New approaches are exploring the use of ensembling several vision encoders to address these limitations [2408.15998 (arxiv.org)](https://arxiv.org/pdf/2408.15998).

Long video understanding is a challenge due to the need to take into account visual information across potential hours of video to properly analyze or answer questions. Like LLMs, VLMs have limited context length meaning—only a certain number of frames from a video can be included to answer questions. Approaches to increase context length and train VLMs on more video-based data are being researched, such as LongVILA [2408.10188 (arxiv.org)](https://www.arxiv.org/pdf/2408.10188).

VLMs may not have seen enough data for very specific use cases, such as finding manufacturing defects in a specific product line. This limitation can be overcome by fine-tuning the VLM on domain-specific data or using multi-image VLMs with in-context learning to provide examples that can teach the model new information without explicitly training the model. Training the model on domain-specific data with PEFT is another technique that can be used to improve a VLM’s accuracy on custom data.

</details>

<details>
<summary>what-is-optical-character-recognition-ocr-explained</summary>

Have you ever wondered how a computer can understand the words on a photo, just like you do?  That's where Optical Character Recognition, or [OCR](https://roboflow.com/ocr?ref=blog.roboflow.com), steps in. OCR takes the text you see in images – be it from a book, a receipt, or an old letter – and turns it into something your computer can read, edit, and search.

OCR finds widespread applications in tasks such as automated data entry, document digitization, text extraction from images, invoice processing, form recognition, and enhancing accessibility for visually impaired individuals.

Let's explore the fundamentals of OCR, understanding its workings, the challenges it addresses, and why it remains a crucial component of present and future technology.

## What Is Optical Character Recognition?

Optical Character Recognition (OCR) involves converting both handwritten and typed text from various sources, including images, videos, and scanned documents like PDFs, into a digitally editable format.

The output from OCR can be used by a computer to make decisions. Common use cases of OCR include:

Using OCR to read product identifiers on an assembly line. When each identifier is read, a piece of software can update an inventory tracking system to note the package with the given identifier has arrived.

Using OCR for scanned document recognition. This involves scanning printed documents, after which OCR software converts them into searchable and editable text. This method is widely employed to automate the handling of legal documents, extract data from bank statements and invoices, and streamline tasks like invoice processing and financial record-keeping.

Using OCR for “scene text recognition”, wherein an OCR system recognizes text from natural scenes, such as street signs, storefronts, or license plates.

Using OCR for alphanumeric, printed text, such as text that was written on a typewriter, or text that was printed out. But, you can also use OCR on handwriting. This usually involves using a separate system due to the differences in handwriting compared to printed text.https://blog.roboflow.com/content/images/2024/04/image-733.webp_Application of OCR on the text of a book._ [_Source_](https://www.edenai.co/post/optical-character-recognition-ocr-which-solution-to-choose?ref=blog.roboflow.com).

## How Optical Character Recognition Works

Let's discuss the typical steps modern OCR software uses to read text:

1. **Image pre-processing**: After an image has been collected, the image undergoes pre-processing to enhance image quality, improving recognition. Pre-processing may involve resizing, contrast enhancement, binarization, noise reduction, and other techniques.
2. **Text Detection**: Using a specialized deep-learning model trained on large datasets of images and text, the computer vision model detects regions in the input image that likely contain text. This process is usually a crucial step.
3. **Layout Analysis**: After detecting text regions, the computer vision model conducts layout analysis to determine the structure and order of the text in the image. This step ensures the preservation of context and organizes the output for readability, but is not run by all OCR systems.
4. **Text Recognition**: Detected text regions pass through a deep learning-based text recognition model, utilizing a combination of convolutional neural networks (CNNs) and recurrent neural networks (RNNs). This model recognizes individual characters and words in the input image, converting them into machine-readable text.
5. **Language Model**: The final output undergoes post-processing to remove noise, correct spelling mistakes, and enhance overall accuracy. The predicted sequence of characters may contain errors, especially for long or uncommon words. Language models, acting as word processors, refine the output by predicting the probability of a sequence of words based on the input image. Statistical models and advanced methods, including deep learning, may be employed for this purpose.https://blog.roboflow.com/content/images/2024/04/image-738.webp_An example OCR system pipeline._

Having acquired an understanding of how OCR operates, let's examine its algorithms and investigate their operational mechanisms, covering the old and the new.

## Traditional Approaches to OCR

The first OCR algorithms rooted in image processing were typically rule-based systems. One well-known OCR that uses this approach is [Tesseract](https://github.com/tesseract-ocr/tesseract?ref=blog.roboflow.com). These systems relied on manually crafted features and heuristic rules to identify characters within images. The approach involved segmenting characters into individual units and applying a set of rules for character classification.

However, the accuracy and performance of these algorithms were often constrained due to the intricate process of developing and fine-tuning the necessary handcrafted features and rules for effective recognition.

### Tesseract

Tesseract, an open-source optical character recognition engine, originated at Hewlett-Packard Laboratories in the 1980s and subsequently became open-source in 2005.

Initially designed to recognize English text exclusively, Tesseract has evolved into a versatile OCR engine. Working from traditional image processing principles, which involves manual logic unlike the deep learning processes in modern systems, Tesseract analyzes images to identify patterns for character recognition.

First, Tesseract preprocesses the image to enhance input quality, a step which encompasses tasks like contrast improvement and noise removal. Following this, Tesseract employs feature extraction techniques, including edge detection and pattern recognition, to identify and recognize characters.https://blog.roboflow.com/content/images/2024/04/image-741.webp_Tesseract OCR engine pipeline._ [_Source_](https://www.researchgate.net/figure/Tesseract-OCR-engine-architecture_fig4_265087843?ref=blog.roboflow.com).

## Deep Learning Approaches to Optical Character Recognition

With the rise of deep learning, the integration of neural networks into OCR systems has gained substantial popularity. In particular, deep learning methodologies like [Convolutional Neural Networks](https://blog.roboflow.com/what-is-a-convolutional-neural-network/) and Long Short-Term Memory networks are leveraged, for precise text recognition. Neural networks regularly achieve better performance than traditional OCR techniques.

In recent years, there has also been a surge in novel approaches that leverage pre-trained image and text [Transformers](https://blog.roboflow.com/what-is-a-transformer/), a deep learning architecture. Transformers are ushering in a new era of end-to-end optical word recognition.

### PaddleOCR

[Paddle OCR](https://arxiv.org/abs/2009.09941?ref=blog.roboflow.com) is an open-source engine developed by Baidu's PaddlePaddle team. Leveraging deep learning techniques, including CNNs and recurrent neural networks, Paddle OCR excels in accurate text recognition. It comprises two key components: the detector and the extractor. The detector is tasked with pinpointing text within an image or document. It employs various algorithms, such as [EAST (Efficient and Accurate Scene Text)](https://paperswithcode.com/paper/east-an-efficient-and-accurate-scene-text?ref=blog.roboflow.com) or [DB (Differentiable Binarization)](https://arxiv.org/abs/1911.08947?ref=blog.roboflow.com) detectors, to identify text regions.https://blog.roboflow.com/content/images/2024/04/image-745.webp_DB (Differentiable Binarization) architecture._ [_Source_](https://arxiv.org/pdf/2009.09941.pdf?ref=blog.roboflow.com).

After the detector locates the text, the extractor comes into play, retrieving the text from the image. It employs a blend of Convolutional Neural Networks and Recurrent Neural Networks for precise text recognition. CNNs are utilized to extract features from the text, while RNNs play a crucial role in recognizing the sequence of characters.https://blog.roboflow.com/content/images/2024/04/image-748.webp_CRNN Extractor architecture._ [_Source_](https://arxiv.org/pdf/1507.05717.pdf?ref=blog.roboflow.com).

Paddle OCR stands out for its remarkable speed, making it among the swiftest OCR engines. Its efficiency is attributed to the utilization of parallel computing and GPU acceleration. This feature renders it particularly suitable for extensive OCR tasks, including document scanning and image recognition. Moreover, its adaptability shines through as it can be tailored and fine-tuned for specific tasks and datasets, enhancing its versatility and robustness in various OCR applications.

### TrOCR

[Transformer-based Optical Character Recognition (TrOCR)](https://arxiv.org/abs/2109.10282?ref=blog.roboflow.com) is one of many transformer-based [OCR models](https://blog.roboflow.com/best-ocr-models-text-recognition/). In contrast to traditional OCR systems, TrOCR adopts a methodology where both input image processing and the generation of corresponding text output occur within a single model.

The encoder segment of TrOCR employs a transformer-based architecture to handle the input image, segmenting it into a grid of patches and extracting visual features from each patch. Simultaneously, the decoder component utilizes a transformer-based model to produce the relevant text output, incorporating the visual features extracted from the image.https://blog.roboflow.com/content/images/2024/04/image-752.webp_TrOCR Architecture._ [_Source_](https://arxiv.org/pdf/2109.10282.pdf?ref=blog.roboflow.com).

This comprehensive and transformer-based methodology empowers TrOCR to attain strong performance across diverse OCR benchmarks, establishing the model as a highly dependable and effective tool for text recognition tasks.

## Advantages of Modern OCR Techniques

One of the primary advantages of OCR is its ability to automate the data entry process. Traditional manual data entry is not only time-consuming but also prone to errors. OCR technology streamlines this process by automatically extracting text from images or scanned documents, eliminating the need for human input. This automation significantly reduces the time required for tasks such as transcribing printed or handwritten text into digital formats.

In addition, OCR facilitates the digitization of documents, leading to improved efficiency in document management. By converting physical documents into digital formats, OCR enables easy storage, retrieval, and organization of information.

Digital documents are more accessible and can be quickly searched, eliminating the need for manual sorting through paper files. This advantage is particularly crucial in business settings where quick access to relevant information is essential.

## Limitations of Modern OCR Techniques

OCR systems, while proficient in recognizing printed text, often face challenges when it comes to accurately interpreting handwritten text. Handwriting is inherently diverse, varying in styles, shapes, and legibility. Unlike printed text, which follows standardized fonts and structures, handwritten text can exhibit significant variability, making it difficult for OCR algorithms to consistently and accurately recognize every nuance.

This limitation is particularly pronounced in scenarios where the handwriting is cursive, unconventional, or poorly formed. Overcoming this challenge requires more advanced techniques, such as integrating machine learning [models](https://blog.roboflow.com/best-ocr-models-text-recognition/) specifically trained on diverse handwritten datasets.

Furthermore,OCR systems can be sensitive to the quality of the input image and may struggle with images that have poor resolution, low contrast, or significant noise. Additionally, documents with complex layouts, multiple columns, or irregular text arrangements pose challenges for traditional OCR methods.

The image preprocessing steps performed by OCR engines, such as Tesseract, are crucial for improving recognition accuracy, but they may not always suffice for images with inherent complexities. Complex layouts can disrupt the OCR's ability to accurately segment text regions and extract meaningful content, leading to errors in character recognition.

To mitigate these issues, additional preprocessing techniques or more advanced OCR methods may be necessary, adding complexity to the implementation process.

## Optical Character Recognition

Optical Character Recognition (OCR) is the extraction of text from scanned documents or images, converting it into machine-readable data to enhance information accessibility.

OCR can reduce the time and resources needed for managing non-searchable or elusive data, eliminating manual data input, reducing errors, and boosting productivity. However, challenges such as handwritten text recognition and sensitivity to image quality persist in OCR systems.

Despite these challenges, OCR remains pivotal in present and future technology, automating data entry, improving document management, and enhancing accessibility. Its adaptability and multilingual support position OCR as a fundamental component in shaping technological advancements.

</details>
