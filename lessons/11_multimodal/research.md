# Research

## Research Results

<details>
<summary>What recent research compares OCR-based PDF retrieval pipelines with vision-language approaches like ColPali, and what specific failure modes and performance gaps have been documented?</summary>

### Source [1]: https://arxiv.org/html/2505.05666v1

Query: What recent research compares OCR-based PDF retrieval pipelines with vision-language approaches like ColPali, and what specific failure modes and performance gaps have been documented?

Answer: The study "Lost in OCR Translation? Vision-Based Approaches to Robust Document Retrieval" presents a **systematic comparison** between a **vision-based RAG system (ColPali)** and traditional **OCR-based pipelines** (using Llama 3.2 (90B) and Nougat OCR) across various document qualities.

Key findings relevant to the comparison:

- **Vision-based approaches like ColPali embed documents visually**, removing the need for OCR and thus avoiding OCR-induced errors, which can be significant in degraded or complex documents.
- The research introduces a **semantic answer evaluation benchmark**, going beyond standard retrieval accuracy to assess **end-to-end question-answering performance**.
- **Performance gaps and failure modes:**
  - **Vision-based RAG (ColPali) performs well on documents it has been fine-tuned on.**
  - **OCR-based RAG generalizes better to unseen documents of varying quality.** This indicates a potential overfitting or lack of robustness in vision-based models when facing truly novel document layouts or degradation types.
  - The study discusses a **trade-off between computational efficiency and semantic accuracy**. Vision-based models may achieve better semantic understanding on known formats, but at the cost of computational expense and with possible generalization issues.
- The paper provides **practical guidance**, suggesting that the selection between OCR-based and vision-based systems should consider the types of documents expected in production, as **vision-based systems may struggle with generalization**, while **OCR-based systems, though error-prone in specific cases, can handle a broader range of unseen document types**.

These findings document specific **failure modes** (generalization issues for vision models, OCR error propagation for OCR pipelines) and **performance gaps** (semantic accuracy in familiar domains for vision models, robustness to novel documents for OCR).

-----

-----

-----

### Source [4]: https://arxiv.org/abs/2505.05666

Query: What recent research compares OCR-based PDF retrieval pipelines with vision-language approaches like ColPali, and what specific failure modes and performance gaps have been documented?

Answer: This arXiv paper also provides a **systematic comparison between ColPali (a vision-based RAG system) and OCR-based pipelines**:

- The paper reiterates that **ColPali eliminates the need for OCR** by directly embedding document visuals.
- **Findings:**
  - Vision-based RAG systems like ColPali are **highly effective on documents they are fine-tuned on**, showing strong semantic QA performance.
  - **OCR-based pipelines, despite occasional errors, are more robust to unseen documents and varying qualities**, supporting better generalization.
- **Failure modes:**
  - ColPali and similar vision-based systems may be **less robust when encountering novel document types or unseen layouts**, indicating a gap in generalization compared to OCR-based pipelines.
- The research emphasizes the **trade-off between semantic accuracy (favoring vision-based models) and generalization/robustness (favoring OCR-based pipelines)**.

-----

-----

</details>

<details>
<summary>How does the late-interaction architecture used in ColBERT enable multi-vector retrieval, and what benchmarking results show its effectiveness over single-vector dense retrievers?</summary>

### Source [5]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/final-reports/final-report-168833820.pdf

Query: How does the late-interaction architecture used in ColBERT enable multi-vector retrieval, and what benchmarking results show its effectiveness over single-vector dense retrievers?

Answer: The ColBERT architecture, as introduced by Khattab & Zaharia in 2020, uses a **multi-vector late-interaction mechanism** for retrieval. In this system, both documents and queries are **encoded into sets of token vectors** rather than a single vector per item. This design allows **fine-grained interactions between query and document tokens**, supporting more nuanced matching compared to single-vector dense retrievers. Specifically, instead of pooling token embeddings into a single vector, ColBERT retains token-level representations and computes interactions at query time, typically using a MaxSim operator, which selects the maximum similarity for each query token across all document tokens.

ColBERTv2, an optimized version, further improves retrieval performance through training optimizations such as **supervision with n-way tuples** and **distilled scores from cross-encoder rerankers**. Benchmarking in this source indicates that retrieval performance improves with these enhancements, and that late-interaction multi-vector models like ColBERT outperform single-vector dense retrievers, especially in ranking accuracy and robustness to out-of-distribution queries.

The document also notes that while single-vector models, such as dual encoders, are efficient, larger encoders and multi-vector approaches (e.g., ColBERTv2) can surpass their performance by capturing richer interactions, as measured on both in-domain and out-of-domain retrieval benchmarks[1].

-----

-----

-----

### Source [6]: https://blog.lancedb.com/late-interaction-efficient-multi-modal-retrievers-need-more-than-just-a-vector-index/

Query: How does the late-interaction architecture used in ColBERT enable multi-vector retrieval, and what benchmarking results show its effectiveness over single-vector dense retrievers?

Answer: **Late interaction rankers** implement an architecture where **every query embedding interacts with all document embeddings using a MaxSim operator**. This operator computes the maximum similarity (like cosine similarity) between each query token and all document tokens, and the scalar outputs are summed across all query terms. This enables ColBERT to leverage **deep language model-based representations** effectively.

A key aspect of ColBERT is that it **builds a vector index for all documents offline**. At query time, it computes the MaxSim interactions between the query’s token embeddings and each document’s token embeddings. This approach allows the system to retain **token-level granularity** in matching, which is not possible in single-vector dense retrievers that compress the entire document into one vector. As a result, late interaction architectures like ColBERT achieve **better retrieval precision** because they can match queries to documents at a finer level of detail[2].

-----

-----

-----

### Source [7]: https://weaviate.io/blog/late-interaction-overview

Query: How does the late-interaction architecture used in ColBERT enable multi-vector retrieval, and what benchmarking results show its effectiveness over single-vector dense retrievers?

Answer: Late interaction retrieval models, such as ColBERT, **store embeddings for each token of a document rather than pooling them into a single vector**. At retrieval time, the model calculates the interaction between the **multi-vector (token-level) embeddings** of the query and the document, which preserves **contextual richness and fine-grained semantic information**.

This architecture is both **scalable** and **contextually rich**, offering a balance between the speed of no-interaction (dual encoder) models and the accuracy of all-to-all interaction (cross-encoder) models. However, the storage requirements are higher since each token embedding must be stored. ColBERTv2 addresses this by applying **aggressive quantization**, reducing vector size significantly while maintaining accuracy.

**Benchmarking results** referenced in this source indicate that late interaction models like ColBERT and ColBERTv2 **outperform single-vector dense retrievers** in ranking precision, as they can capture more detailed semantic relationships between queries and documents. This advantage is especially clear when deploying quantization and efficient storage solutions, which mitigate the higher storage cost[3].

-----

-----

-----

### Source [8]: https://www.pinecone.io/blog/cascading-retrieval-with-multi-vector-representations/

Query: How does the late-interaction architecture used in ColBERT enable multi-vector retrieval, and what benchmarking results show its effectiveness over single-vector dense retrievers?

Answer: Multi-vector models such as ColBERT are designed to **capture fine-grained interactions between queries and documents**, outperforming traditional single-vector dense retrievers in **ranking precision**. However, they are generally not optimized for the absolute fastest first-stage retrieval at very large scale, but rather excel as **second-stage rerankers** within multi-stage retrieval pipelines.

A benchmarking comparison in this source shows that late interaction models like ColBERT **significantly improve ranking precision over dense retrieval models** while avoiding the high compute cost of full cross-encoders. They represent a **middle ground**: offering much higher accuracy than sparse or dense single-vector retrievers, with efficiency that is still practical for many applications[4].

-----

-----

-----

### Source [9]: https://aclanthology.org/2025.coling-main.295.pdf

Query: How does the late-interaction architecture used in ColBERT enable multi-vector retrieval, and what benchmarking results show its effectiveness over single-vector dense retrievers?

Answer: ColBERT and its extensions adopt a **late interaction scoring mechanism** where **cosine similarities are computed for all pairs of query and passage embeddings**. For each query term, the maximum similarity across all passage tokens is selected (MaxSim), and these maximum values are summed over all query terms to produce the final relevance score.

This method allows ColBERT to **preserve fine-grained token-level interactions** which single-vector dense retrievers miss, as they compress all semantic information into a single vector. By keeping interactions "late" (i.e., after encoding but before scoring), ColBERT achieves **greater expressiveness and accuracy** in retrieval.

Empirical results cited in this paper indicate that such multi-vector late-interaction architectures **consistently outperform single-vector dense retrievers** in benchmarks, especially on tasks that require nuanced understanding of query-document relationships[5].

-----

</details>

<details>
<summary>What best-practice guidelines do cloud providers (Google, AWS, Azure) give for supplying images to multimodal LLM APIs, and how do they compare raw-byte, Base64, and signed-URL methods?</summary>

### Source [10]: https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-multimodal-embeddings

Query: What best-practice guidelines do cloud providers (Google, AWS, Azure) give for supplying images to multimodal LLM APIs, and how do they compare raw-byte, Base64, and signed-URL methods?

Answer: Google Vertex AI’s official documentation for multimodal embeddings states that **images can be supplied for multimodal LLM APIs via two main methods**:
- **Base64-encoded image data**: Images can be provided directly as Base64-encoded data within the API request body. The maximum accepted size is 20 MB when transcoded to PNG. Supported formats are BMP, GIF, JPG, and PNG. Google recommends using smaller images to avoid increased network latency, since the model resizes all images to 512x512 pixels. Providing images larger than this resolution offers no benefit, as they will be downscaled.
- **Cloud Storage URL**: Images can be referenced via a Google Cloud Storage URI. The maximum file size is also 20 MB (in the original format). 

**Best practices highlighted**:
- Use smaller images to reduce network latency.
- Since all images are resized to 512x512 pixels, avoid supplying higher-resolution images.
- Choose between Base64 or Cloud Storage URL based on convenience, performance, and expected image reuse[1].

-----

-----

-----

### Source [12]: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding

Query: What best-practice guidelines do cloud providers (Google, AWS, Azure) give for supplying images to multimodal LLM APIs, and how do they compare raw-byte, Base64, and signed-URL methods?

Answer: For Gemini on Vertex AI, Google provides **three main approaches** to supply images:
- **Cloud Storage bucket URI**: The image must be either publicly readable or reside in the same project as the API call. The maximum accepted file size is 2 GB for specific models.
- **HTTP URL**: The image must be publicly readable. You may supply up to 10 image files per request. The maximum file size for images via HTTP URL is 15 MB.
- **YouTube video URL**: For video analysis, not directly relevant for static images.

**Best practices**:
- Specify the correct MIME type when using file URIs.
- Use Cloud Storage or HTTP URLs for larger or shared images, while inline data (Base64) is suitable for smaller files.
- VPC Service Controls may restrict use of media URLs.

Google’s guidance leans toward using URLs (either Cloud Storage or HTTP) for larger images or when images will be reused, and inline data only for small files or one-off submissions[3].

-----

-----

-----

### Source [14]: https://ai.google.dev/gemini-api/docs/image-understanding

Query: What best-practice guidelines do cloud providers (Google, AWS, Azure) give for supplying images to multimodal LLM APIs, and how do they compare raw-byte, Base64, and signed-URL methods?

Answer: The Gemini API documentation for image understanding outlines two ways to provide images to the API:
- **Passing inline image data**: For smaller files (total request size, including text, less than 20 MB), images can be supplied as Base64-encoded strings or read from local files, then included directly in the API request.
- **Uploading via the File API**: For larger files or when images need to be reused across multiple requests, uploading the image via the File API is recommended. This typically results in a signed URL or reference that you provide to the model in your request.

**Best practices include**:
- Prefer **inline Base64** for small, one-off images where request size is manageable.
- Use **File API (signed URLs or cloud links)** for larger images or when images are reused, to reduce request payload and improve efficiency.
- Choose the method based on file size, expected reuse, and performance needs.

No explicit comparison is made with raw-byte uploads, but the methods reflect the typical practice of using Base64 for small/ephemeral uploads and URLs for larger or persistent image storage[5].
-----

-----

</details>

<details>
<summary>Which open-weight multimodal embedding models (e.g., CLIP, SigLIP, JinaCLIP, Nomic Embed Vision) are recommended for text-image similarity, and what published evaluations detail their capabilities and limitations?</summary>

### Source [15]: https://www.nomic.ai/blog/posts/nomic-embed-vision

Query: Which open-weight multimodal embedding models (e.g., CLIP, SigLIP, JinaCLIP, Nomic Embed Vision) are recommended for text-image similarity, and what published evaluations detail their capabilities and limitations?

Answer: **Nomic Embed Vision** v1 and v1.5 are open-weight, high-quality vision embedding models designed to share the same latent space as Nomic Embed Text models, enabling seamless multimodal embedding for both text and images. All Nomic Embed Text embeddings are now multimodal, allowing users to query vision embeddings with text and vice versa. According to Nomic, their unified embedding space outperforms OpenAI CLIP and OpenAI Text Embedding 3 Small on multimodal and text tasks. Nomic Embed Vision features a compact vision encoder (92M parameters), making it suitable for high-volume production. 

The training code and replication instructions for Nomic Embed Vision are open-sourced, ensuring replicability and transparency. The blog notes that while CLIP models have strong zero-shot multimodal capabilities, their text encoders underperform on purely text-based embedding benchmarks (such as MTEB). Nomic Embed Vision is specifically designed to address these limitations, aligning the vision encoder optimally to the multimodal embedding space. No direct comparison with SigLIP is provided in this announcement.

-----

-----

-----

### Source [16]: https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Document_Haystacks__Vision-Language_Reasoning_Over_Piles_of_1000_Documents_CVPR_2025_paper.pdf

Query: Which open-weight multimodal embedding models (e.g., CLIP, SigLIP, JinaCLIP, Nomic Embed Vision) are recommended for text-image similarity, and what published evaluations detail their capabilities and limitations?

Answer: This Vision-Language Reasoning benchmark evaluates several open-weight multimodal embedding models for text-to-image similarity, including **Jina-CLIP-v1**, **Nomic-Embed-Vision-v1.5**, **CLIP ViT-L/14@336**, and **SigLIP**. Performance is reported on recall metrics (R@1, R@3, R@5) across datasets of increasing size (100, 200, 1000 documents). Key findings include:

- **SigLIP** generally outperforms other open models at higher recall levels, particularly on larger datasets (e.g., R@1=33.0 for 1000 docs, higher than CLIP and Nomic).
- **CLIP** shows strong performance, especially for smaller datasets, but its recall drops with scale.
- **Nomic-Embed-Vision** and **Jina-CLIP** perform less well on these large retrieval tasks, with notably lower recall scores as dataset size increases.
- **BM25 (OCR)**, a non-neural baseline, outperforms all neural models on this particular document retrieval task, suggesting challenges for current multimodal models in large-scale, document-level text-image similarity.

This benchmark provides a direct, empirical comparison of open-weight models, detailing both capabilities (recall under scaling) and limitations (drop-off in large collections).

-----

-----

-----

### Source [17]: https://arxiv.org/html/2504.10471v1

Query: Which open-weight multimodal embedding models (e.g., CLIP, SigLIP, JinaCLIP, Nomic Embed Vision) are recommended for text-image similarity, and what published evaluations detail their capabilities and limitations?

Answer: The **Massive Image Embedding Benchmark (MIEB)** systematically evaluates open-weight multimodal embedding models, including various CLIP and SigLIP variants, on a wide range of image-text similarity tasks. The benchmark computes embedding similarity scores and compares them to human annotations using Spearman correlation as the main metric. Notable results:

- **CLIP-ViT-bigG-14-laion2B** and **EVA02-CLIP-bigE-14-plus** achieve the highest Spearman correlations on most image-text similarity datasets, indicating superior alignment with human judgments.
- **SigLIP** models (base-patch16-512 and base-patch16-384) rank slightly below the top CLIP variants but still deliver strong, consistent performance across benchmarks.
- **CLIP-ViT-L-14-DataComp.XL** achieves competitive results, though not as high as the largest CLIP/EVA02 models.
- The results highlight that while CLIP variants generally lead, SigLIP is a strong open alternative, especially for users prioritizing open weights and competitive performance.

The evaluation details both strengths (high correlation with human similarity judgments) and limitations (performance variance across datasets and modalities) for each model.

-----

-----

</details>

<details>
<summary>What example projects or documentation show how LangGraph’s create_react_agent can integrate a custom retrieval tool to build a multimodal ReAct agent workflow?</summary>

### Source [19]: https://python.langchain.com/docs/tutorials/agents/

Query: What example projects or documentation show how LangGraph’s create_react_agent can integrate a custom retrieval tool to build a multimodal ReAct agent workflow?

Answer: This official LangChain tutorial demonstrates how to build an agent that can interact with external tools, such as a search engine, using LangGraph’s high-level interface. The tutorial provides code for initializing an agent with a language model (LLM) and a list of tools. The `create_react_agent` function from `langgraph.prebuilt` is used to construct the agent, automatically handling the binding of tools to the model:

```python
from langgraph.prebuilt import create_react_agent

agent_executor = create_react_agent(model, tools)
```

The documentation notes that this high-level interface is backed by a more flexible, low-level API, allowing for further customization of agent logic. While the tutorial primarily showcases integration with a search tool, it emphasizes that additional tools—including custom retrieval tools—can be defined and passed into the agent. This lays the groundwork for building **multimodal ReAct agent workflows** by extending the list of tools to include, for example, retrieval-augmented generation or image processing capabilities.

The page encourages users to explore the LangGraph documentation for more advanced concepts, tutorials, and how-to guides, which are likely to provide further examples of integrating custom tools, including for multimodal use cases.

-----

-----

-----

### Source [20]: https://python.langchain.com/docs/how_to/migrate_agent/

Query: What example projects or documentation show how LangGraph’s create_react_agent can integrate a custom retrieval tool to build a multimodal ReAct agent workflow?

Answer: This migration guide details how to transition from legacy LangChain agents to LangGraph, explicitly using the `create_react_agent` helper method for constructing ReAct agents. It shows how to structure the agent’s state as a list of messages and how the agent processes these messages until no more tool calls are present in the output.

Key steps include:
- Creating a list of **tools** (which can include custom retrieval or multimodal tools).
- Initializing the agent executor with `create_react_agent(model, tools)`.
- Invoking the agent with a message history, allowing for continued conversation.

Example code:
```python
from langgraph.prebuilt import create_react_agent

langgraph_agent_executor = create_react_agent(model, tools)

messages = langgraph_agent_executor.invoke({"messages": [("human", query)]})
```

The agent’s state and conversation history are managed and returned as part of the output, making it straightforward to integrate **custom retrieval tools** by including them in the `tools` list. The agent can then call these tools as needed during execution, supporting **multimodal workflows** if the custom tool handles multimodal data.

-----

-----

-----

### Source [21]: https://python.langchain.com/api_reference/langchain/agents/langchain.agents.react.agent.create_react_agent.html

Query: What example projects or documentation show how LangGraph’s create_react_agent can integrate a custom retrieval tool to build a multimodal ReAct agent workflow?

Answer: The API reference for `create_react_agent` explains the function’s parameters and usage. It emphasizes that for robust, production-ready implementations, the `create_react_agent` function from the **LangGraph library** is preferred.

Key configuration options:
- **llm**: The language model used by the agent.
- **tools**: A sequence of tools that the agent can access. These tools can be **custom-built retrieval tools** or multimodal tools, provided they conform to the required interface.
- **prompt**: The prompt template for the agent.
- **output_parser** and **tools_renderer**: Advanced controls for customizing how outputs and tool descriptions are parsed and presented.

The documentation states:
> “Tools this agent has access to” can be any tool provided in the sequence, allowing for the integration of custom retrieval or multimodal tools.

This makes it possible to create a **multimodal ReAct agent workflow** by supplying the relevant tools during agent construction.

-----

-----

-----

### Source [22]: https://ai.google.dev/gemini-api/docs/langgraph-example

Query: What example projects or documentation show how LangGraph’s create_react_agent can integrate a custom retrieval tool to build a multimodal ReAct agent workflow?

Answer: This example demonstrates building a **ReAct agent from scratch with Gemini 2.5 and LangGraph**, showcasing the flexibility to define custom workflows. While it points out that LangGraph provides a prebuilt `create_react_agent`, the guide details how to build a fully customizable agent graph, which is highly relevant for integrating **custom retrieval tools and multimodal workflows**.

Workflow structure:
- **StateGraph** is used to define the agent’s state, which can include multimodal data.
- Nodes are created for different tasks, such as calling the LLM or invoking tools.
- Conditional logic can route execution to different tool nodes, depending on the output of the LLM.

This approach allows for the integration of **custom tools** (e.g., retrieval or multimodal tools like image captioning or document search) by adding them as nodes in the workflow and defining how and when they are called. This is suitable for users needing a highly tailored **multimodal ReAct agent** beyond standard prebuilt agents.

-----

-----

</details>

<details>
<summary>What research has quantified OCR accuracy on documents with complex layouts (e.g., tables, multi-column pages, handwritten notes) and how does it compare to plain-text pages?</summary>

### Source [23]: https://www.amazon.science/publications/docbed-a-multi-stage-ocr-solution-for-documents-with-complex-layouts

Query: What research has quantified OCR accuracy on documents with complex layouts (e.g., tables, multi-column pages, handwritten notes) and how does it compare to plain-text pages?

Answer: The DocBed study addresses the **challenge of digitizing newspapers with complex layouts**, such as multiple columns and text interrupted by images. The researchers released a dataset of 3,000 fully-annotated, real-world newspaper images from 21 different U.S. states, covering a wide variety of complex layouts. The study proposes that **layout segmentation should precede OCR** to improve results on such documents. Multiple state-of-the-art image segmentation models and post-processing methods are evaluated for layout segmentation, followed by end-to-end OCR evaluation.

The research provides a **structured evaluation protocol** for both isolated layout segmentation and complete OCR workflows. Although specific quantitative accuracy results are not quoted in the abstract, the study emphasizes that **complex layouts present significant challenges** for OCR engines compared to plain-text pages. By segmenting the layout before applying OCR, the system can better preserve the human read-order and improve recognition accuracy, highlighting the **necessity of layout analysis in handling complex documents**[1].

-----

-----

-----

### Source [25]: https://onlinelibrary.wiley.com/doi/full/10.1002/eng2.12832

Query: What research has quantified OCR accuracy on documents with complex layouts (e.g., tables, multi-column pages, handwritten notes) and how does it compare to plain-text pages?

Answer: This peer-reviewed article explores **document layout analysis (DLA)** and **text line detection (TLD)** as critical components for improving OCR systems, particularly for documents with complex layouts. The authors develop a method using **deep learning models and a voting system** to accurately extract both textual and non-textual regions. They introduce techniques such as **angle correction and line curvature elimination** to further boost OCR accuracy.

The study reports a **2.8% improvement in the accuracy of Tesseract-OCR 5.1.0** after applying their advanced DLA and TLD techniques, compared to standard approaches. This demonstrates that addressing layout complexities—such as separating tables, columns, and irregular text flows—**measurably improves OCR accuracy** when compared to baseline OCR performance, which typically assumes plain-text, single-column pages[3].

-----

-----

</details>

<details>
<summary>How does ColPali’s visual document retrieval performance compare to OCR-based RAG pipelines on benchmarks like ViDoRe, including metrics such as Recall@k or nDCG?</summary>

### Source [27]: https://arxiv.org/html/2407.01449v1

Query: How does ColPali’s visual document retrieval performance compare to OCR-based RAG pipelines on benchmarks like ViDoRe, including metrics such as Recall@k or nDCG?

Answer: ColPali is a retrieval model based on **Vision Language Models (VLMs)** that indexes documents **purely from visual features**, enabling fast query matching via late interaction mechanisms. On the **ViDoRe benchmark**, ColPali **outperforms all other retrieval systems**, including traditional OCR-based RAG (Retrieval Augmented Generation) pipelines, in both performance and speed. The paper states that ColPali "largely outperforms the best existing document retrieval methods" on ViDoRe, while also enabling faster indexing and maintaining low query latencies. This suggests significant improvements in metrics like **Recall@k** and **nDCG** (Normalized Discounted Cumulative Gain), although precise metric values are not provided in this section. The model's end-to-end trainability and reliance on visual features position it as a high-potential solution for industrial document retrieval applications, especially where document visuals and layouts are crucial[1].

-----

-----

-----

### Source [29]: https://qdrant.tech/blog/colpali-qdrant-optimization/

Query: How does ColPali’s visual document retrieval performance compare to OCR-based RAG pipelines on benchmarks like ViDoRe, including metrics such as Recall@k or nDCG?

Answer: In practical experiments optimizing ColPali for large-scale retrieval, a dataset of over 20,000 PDF pages—including the ViDoRe benchmark—was used. Retrieval quality was measured using **NDCG@20** and **Recall@20**. With mean pooling, ColPali achieved **NDCG@20 = 0.952** and **Recall@20 = 0.917**, which is described as "nearly identical quality to the original ColPali." These results highlight that ColPali maintains high retrieval quality even with optimizations for speed (13x faster retrieval). The source does not provide direct baseline values for OCR-based RAG pipelines, but the strong scores and qualitative statements reinforce ColPali's superior performance on ViDoRe relative to traditional methods[3].

-----

-----

-----

### Source [30]: https://arxiv.org/abs/2407.01449

Query: How does ColPali’s visual document retrieval performance compare to OCR-based RAG pipelines on benchmarks like ViDoRe, including metrics such as Recall@k or nDCG?

Answer: The abstract of the ColPali paper describes the limitations of OCR-based pipelines, noting that they "mainly rely on the textual information they extract from document pages," which makes them "lengthy and brittle." The ColPali approach, by contrast, leverages visual information—including **figures, page layouts, tables, and fonts**—which are often missed by OCR. The paper introduces ViDoRe specifically to benchmark such visually rich retrieval tasks and reports that ColPali "outperforms modern document retrieval pipelines" on this benchmark. This further substantiates ColPali's superiority in visual document retrieval performance using metrics like Recall@k and nDCG, though it does not give explicit metric values in the abstract[4].

-----

</details>

<details>
<summary>What best-practice recommendations do AWS, Azure, and Google give for supplying images to multimodal LLM APIs, and how do raw-byte, Base64, and signed-URL methods compare in performance and reliability?</summary>

### Source [31]: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding

Query: What best-practice recommendations do AWS, Azure, and Google give for supplying images to multimodal LLM APIs, and how do raw-byte, Base64, and signed-URL methods compare in performance and reliability?

Answer: Google Cloud Vertex AI's best-practice recommendations for supplying images to multimodal LLM APIs include several supported methods:

- **Base64-encoded image data**: You can directly include base64-encoded image data in your request, which is useful for quick prototyping or when the image size is small.
- **Cloud Storage bucket URI**: Images can be referenced via a Cloud Storage URI (`gs://`). The object must be publicly readable or reside in the same Google Cloud project as the requesting service. For certain models (e.g., gemini-2.0-flash), the limit is 2 GB per file.
- **HTTP URL**: You may use a publicly readable HTTP URL to reference images, with a limit of up to 10 image files per request. The file must be accessible without authentication.
- **By direct upload via UI**: For manual testing or simple use cases, you can upload the file through the Vertex AI console interface.

**Performance and reliability comparison:**
- **Base64 encoding** is straightforward but can increase request payload sizes (by about 33%), potentially causing timeouts or exceeding limits for large files.
- **Cloud Storage URI** is preferred for large files because it avoids payload bloat, is more reliable for production, and leverages Google’s infrastructure. It also supports large file sizes (up to 2 GB).
- **HTTP URLs** are suitable for small, public files. However, reliability depends on the uptime and speed of the external server. If VPC Service Controls are enabled, specifying a media file URL is not supported.
- **Signed URLs** are not specifically mentioned, but the requirement for public readability implies that signed URLs could be used if they provide public, time-limited access.

When specifying a file URI, you must also provide the media type (`mimeType`) of the file. Public accessibility is a requirement unless the file resides in the same project as the requesting service.

-----

-----

-----

### Source [32]: https://aws.amazon.com/blogs/machine-learning/build-an-image-to-text-generative-ai-application-using-multimodality-models-on-amazon-sagemaker/

Query: What best-practice recommendations do AWS, Azure, and Google give for supplying images to multimodal LLM APIs, and how do raw-byte, Base64, and signed-URL methods compare in performance and reliability?

Answer: Amazon SageMaker (AWS) documentation for multimodal LLM APIs primarily discusses model artifact handling via S3 but gives indirect guidance relevant to image inputs for inference endpoints:

- **S3 URL reference**: For model artifacts, AWS supports specifying Amazon S3 URLs, which is performant and reliable due to native integration. While the example focuses on models, this approach extends to data inputs in other AWS APIs.
- When invoking the endpoint, you can send an input image directly, but the blog does not explicitly specify whether images should be sent as raw bytes, base64 strings, or as URLs. However, in most AWS generative AI APIs, sending images as base64-encoded data or as S3 URLs is common practice.

**Performance and reliability notes:**
- **S3 URLs** are highly performant and reliable within AWS, especially for large image files, as data transfer stays within the AWS network.
- **Base64 encoding** is generally used for smaller images or when the API expects inline data. However, this can lead to payload size increases, which may affect latency and error rates for large files.
- **Raw bytes** are rarely used directly in REST APIs, as they are not JSON-safe, but could be used in lower-level SDKs or with multipart/form-data.
- **Signed URLs** (presigned S3 URLs) are commonly used for temporary, secure access to private files, offering a secure and scalable method for referencing images.

The blog emphasizes that S3-based workflows are preferred for performance, particularly with large files, due to efficient in-network transfer and storage handling.

-----

-----

-----

### Source [33]: https://langfuse.com/docs/tracing-features/multi-modality

Query: What best-practice recommendations do AWS, Azure, and Google give for supplying images to multimodal LLM APIs, and how do raw-byte, Base64, and signed-URL methods compare in performance and reliability?

Answer: Langfuse supports multi-modal LLM traces and provides insights into handling media attachments (including images):

- **Base64 encoding**: Langfuse SDKs automatically handle base64-encoded data URIs. These are extracted and uploaded to object storage, with links provided in the trace.
- **External URLs**: Media files can be referenced via external URLs, which the system fetches and stores as needed.
- **Customization via SDKs**: Users can customize media handling by referencing files directly or via the `LangfuseMedia` class.

**Performance and reliability comparison:**
- **Base64** is convenient but increases payload size and compute costs, especially for large files.
- **External URLs** allow for flexibility and reduced payload sizes but depend on the reliability and accessibility of the external server.
- **Direct uploads to object storage** (e.g., S3-compatible buckets) are recommended for large files, ensuring scalability and efficient access. The storage bucket must be publicly resolvable for direct uploads and browser access.

Langfuse’s approach mirrors best practices for cloud-based multi-modality: use base64 for small files or quick prototyping and object storage URLs (or signed URLs) for production and large files.

-----

-----

</details>

<details>
<summary>Which real-world industry case studies document multimodal AI agents processing financial reports, medical diagnostics, or complex technical diagrams, and what benefits were reported over text-only systems?</summary>

### Source [35]: https://www.multimodal.dev/customer-stories/financial-statement-processing-ai

Query: Which real-world industry case studies document multimodal AI agents processing financial reports, medical diagnostics, or complex technical diagrams, and what benefits were reported over text-only systems?

Answer: A Fortune 500 company faced challenges in processing **unstructured financial reports** due to document inconsistency and complexity. To address this, an **Unstructured AI Agent** was deployed specifically designed to handle unstructured, multi-format financial documents at scale. The system could **extract charts, tables, and freeform text** from various formats such as PDFs, DOCX, and HTML, transforming them into structured, query-ready JSON outputs.

Key technical advantages included:
- **Parallel processing**: The agent split long documents into overlapping page segments and reassembled them using GPT-based analysis, maintaining header continuity and accurate section mapping.
- **Improved data integrity**: This method avoided data loss often seen in standard extraction tools, especially when crucial context spanned multiple pages.

**Benefits reported over text-only systems**:
- **80% faster processing** of financial statements.
- Enabled a **seamless pipeline** where documents were automatically converted into structured insights.
- Credit analysts could quickly search, analyze, and act on structured data, leading to faster and more consistent downstream decisions, such as credit memo preparation.
- The automation significantly reduced the backlog and manual review workload, directly improving operational efficiency.

-----

-----

-----

### Source [36]: https://aws.amazon.com/blogs/machine-learning/generative-ai-and-multi-modal-agents-in-aws-the-key-to-unlocking-new-value-in-financial-markets/

Query: Which real-world industry case studies document multimodal AI agents processing financial reports, medical diagnostics, or complex technical diagrams, and what benefits were reported over text-only systems?

Answer: AWS highlights the use of **multi-modal agents** in financial markets, emphasizing their ability to process and summarize **lengthy financial reports** rapidly, which saves analysts substantial time and effort. These agents can analyze diverse data sources, including text, tables, and charts, to generate **market intelligence reports**.

Key use cases:
- **Smart reporting and market intelligence**: Multi-modal agents synthesize various financial information sources, producing comprehensive, up-to-date reports that help analysts and investors track trends.
- **Quantitative modeling and forecasting**: By integrating structured and unstructured data, multi-modal AI delivers more robust financial forecasts and risk assessments than text-only systems.
- **Compliance and fraud detection**: Multi-modal agents monitor a range of data types (calls, emails, chats, logs) to detect insider trading or market manipulation, which requires cross-modal analysis.

**Reported benefits over text-only systems**:
- **Productivity boost**: Automates repetitive tasks, freeing human analysts for higher-value work.
- **Faster, more accurate insights**: Enhanced ability to extract, combine, and analyze insights from heterogeneous data.
- **Scalability and security**: AWS's integration of generative AI and analytics services enables secure, large-scale, and efficient analysis of multi-modal financial data.

-----

-----

-----

### Source [37]: https://generativeaienterprise.ai/p/20-must-read-ai-agents-case-studies-bb2cf6f29ed87b92

Query: Which real-world industry case studies document multimodal AI agents processing financial reports, medical diagnostics, or complex technical diagrams, and what benefits were reported over text-only systems?

Answer: OpenAI published a case study on its collaboration with **Endex**, a company that developed a **financial analyst AI agent** leveraging OpenAI's advanced reasoning models. The agent processes a wide range of financial documents, including **investor presentations, internal decks, Excel models, and 8-Ks**, using both text and visual (vision) capabilities.

Key features and benefits:
- **Automated reconciliation and discrepancy detection**: The agent identifies inconsistencies and financial restatements, flagging issues with precise citations.
- **Multi-step workflows**: Endex uses OpenAI models to automate complex processes like financial model reconciliation, which previously required manual effort.
- **Vision capabilities**: OpenAI’s o1 model enables the agent to analyze not just text but also visuals from presentations and spreadsheets, broadening the scope of analyzable data.
- **Efficiency gains**: The system delivers 3x faster analysis, reduces manual report generation, and allows finance professionals to focus on strategic tasks instead of data formatting.
- **Referenceable reasoning**: Provides structured, referenceable outputs that support regulatory and auditing needs, a challenge for typical non-reasoning LLMs.

-----

-----

-----

### Source [38]: https://global.fujitsu/en-global/insight/tl-aiagents-financial-industry-20250418

Query: Which real-world industry case studies document multimodal AI agents processing financial reports, medical diagnostics, or complex technical diagrams, and what benefits were reported over text-only systems?

Answer: Fujitsu documents early adoption of **AI agents in the financial industry**, highlighting real-world deployments by companies like **Moody’s**. Moody’s created a multi-agent system with 35 specialized agents, including those dedicated to analyzing **SEC filings and industry data**.

Key aspects:
- **Multi-agent collaboration**: Agents specialize in tasks like project management and financial analysis, with supervising agents verifying outputs for higher reliability.
- **Complex workflow execution**: The system efficiently handles complex research tasks, including extracting and synthesizing information from various financial documents.
- **Enhanced research capabilities**: By automating extraction and verification, human analysts are freed to focus on higher-level analysis and decision-making.

**Benefits over text-only systems**:
- **Efficient handling of complex, multi-format data**: The agents’ ability to process both text and structured data from filings leads to faster and more reliable research.
- **Quality assurance**: Supervising agents verify results, reducing the risk of error.
- **Workforce transformation**: Human analysts collaborate with AI agents, shifting their focus from routine processing to analytical and strategic work.

-----

-----

</details>

<details>
<summary>What recent benchmarks evaluate open-weight multimodal embedding models (e.g., CLIP, SigLIP, JinaCLIP, Nomic Embed Vision, EVA-CLIP) on image-text retrieval tasks, and what performance rankings do they report?</summary>

### Source [39]: https://jina.ai/models/jina-clip-v1/

Query: What recent benchmarks evaluate open-weight multimodal embedding models (e.g., CLIP, SigLIP, JinaCLIP, Nomic Embed Vision, EVA-CLIP) on image-text retrieval tasks, and what performance rankings do they report?

Answer: Jina CLIP v1 is evaluated on multiple standard benchmarks for **image-text retrieval tasks**, including MTEB for text retrieval, CIFAR-100 for image tasks, and Flickr8k/30k and MSCOCO Captions for cross-modal performance. The model is reported to outperform OpenAI's original CLIP across all these benchmarks. 

- In **text-only retrieval**, Jina CLIP v1 achieves a score of **0.429** compared to CLIP's **0.162**.
- In **text-to-image retrieval**, it achieves **0.899**, which is 2% higher than CLIP.
- For **image-to-text retrieval**, the score is **0.803** (6% higher than CLIP).
- In **image-to-image retrieval**, it scores **0.916** (12% higher than CLIP).

The model is also evaluated on both short captions and detailed text descriptions, maintaining strong visual and textual understanding. It consistently outperforms specialized single-modality models while remaining competitive in cross-modal tasks.

-----

-----

-----

### Source [40]: https://jina.ai/models/jina-embeddings-v4/

Query: What recent benchmarks evaluate open-weight multimodal embedding models (e.g., CLIP, SigLIP, JinaCLIP, Nomic Embed Vision, EVA-CLIP) on image-text retrieval tasks, and what performance rankings do they report?

Answer: Jina Embeddings V4 is benchmarked for **cross-modal retrieval** and compared with other open-weight models:

- On the **CLIP Benchmark**, it scores **84.11**, outperforming jina-clip-v2 (**81.12**) and nllb-clip-large-siglip (**83.19**).
- It achieves **improved cross-modal alignment** with a score of **0.71**, whereas OpenAI CLIP scores **0.15**, addressing the modality gap issue seen in other multimodal models.
- For **visual document retrieval** (ViDoRe benchmark), it scores **84.11** on average, and in multi-vector mode, it achieves **90.17**.

The results indicate that Jina Embeddings V4 is a leading open-weight model for image-text retrieval, outperforming prior versions and SigLIP-based models in several cross-modal tasks.

-----

-----

-----

### Source [41]: https://arxiv.org/html/2412.08802v2

Query: What recent benchmarks evaluate open-weight multimodal embedding models (e.g., CLIP, SigLIP, JinaCLIP, Nomic Embed Vision, EVA-CLIP) on image-text retrieval tasks, and what performance rankings do they report?

Answer: The arXiv preprint for jina-clip-v2 provides **benchmark comparisons and detailed performance metrics**:

- On the **ViDoRe benchmark** (Faysse et al., 2024), jina-clip-v2 achieves **35% higher scores** than jina-clip-v1, indicating significant improvement for **visual document retrieval**.
- In **multilingual cross-modal performance**, jina-clip-v2 achieves up to **67% higher scores** than jina-clip-v1, and up to **60% higher on multilingual retrieval performance**.
- The paper includes a table showing **Recall@5** scores for zero-shot image retrieval, directly comparing jina-clip-v2, jina-clip-v1, and nllb-siglip models, but the exact numerical values for Recall@5 are not fully visible in the snippet.

The source establishes jina-clip-v2 as a state-of-the-art open-weight model for both monolingual and multilingual image-text retrieval tasks, surpassing both its previous version and SigLIP derivatives.

-----

-----

</details>

<details>
<summary>What peer-reviewed studies quantify OCR accuracy degradation on complex PDF layouts (e.g., multi-column tables, charts, handwritten notes) compared with plain single-column text documents?</summary>

### Source [42]: https://www.ijsr.net/archive/v14i6/SR25603211507.pdf

Query: What peer-reviewed studies quantify OCR accuracy degradation on complex PDF layouts (e.g., multi-column tables, charts, handwritten notes) compared with plain single-column text documents?

Answer: This peer-reviewed report provides a comparative analysis between **traditional OCR and GenAI-assisted OCR**, with a particular focus on complex layouts such as multi-column formats, tables, and graphical elements. The study highlights that **traditional OCR systems are significantly limited** when processing documents with intricate layouts. These systems often rely on rigid, template-based algorithms, causing them to struggle with documents containing **overlapping text, embedded images, or complex tables**. The report notes that traditional OCR has difficulty accurately extracting data from these challenging formats, leading to a marked **degradation in accuracy** compared to processing plain, single-column text documents. In contrast, GenAI-assisted OCR—leveraging transformer architectures and multimodal learning—shows improved adaptability and recognition of complex layouts, thereby reducing the accuracy gap. However, the report does not provide specific quantitative metrics for the degree of accuracy loss in traditional OCR on complex versus plain layouts, but it emphasizes traditional OCR’s consistent underperformance on complex documents.

-----

-----

-----

### Source [43]: https://onlinelibrary.wiley.com/doi/full/10.1002/eng2.12832

Query: What peer-reviewed studies quantify OCR accuracy degradation on complex PDF layouts (e.g., multi-column tables, charts, handwritten notes) compared with plain single-column text documents?

Answer: This article discusses **document layout analysis (DLA) and text line detection (TLD)** as critical factors influencing OCR accuracy, particularly in languages with complex scripts. The authors demonstrate that improving DLA and TLD directly enhances OCR performance on documents with **complex layouts**. In their comparative evaluation, they report a **2.8% improvement in OCR accuracy** (using Tesseract-OCR 5.1.0) when their advanced layout analysis techniques are applied, compared to baseline methods. While the article focuses on Persian and similar languages, the results underscore that documents with **irregular arrangements—such as multiple columns, skewed lines, and curved text—lead to lower OCR accuracy** unless sophisticated layout analysis is performed. The study thus quantifies the typical improvement margin (up to several percent) that can be achieved on complex layouts through targeted DLA and TLD enhancements, implying that baseline OCR on complex layouts is measurably worse than on plain, single-column documents.

-----

-----

-----

### Source [44]: https://graz.elsevierpure.com/en/publications/enhancing-ocr-in-historical-documents-with-complex-layouts-throug

Query: What peer-reviewed studies quantify OCR accuracy degradation on complex PDF layouts (e.g., multi-column tables, charts, handwritten notes) compared with plain single-column text documents?

Answer: This peer-reviewed paper investigates the impact of **machine learning-driven layout detection** on OCR accuracy for historical documents with complex layouts, specifically the 19th-century Hof- und Staatsschematismus records. The authors first train a Faster R-CNN model on synthetic and annotated data to detect intricate structural elements before applying OCR. Their **evaluation provides quantitative evidence**: integrating advanced layout detection and fine-tuned OCR yields a **15.68 percentage point reduction in Character Error Rate (CER)** and a **19.95 percentage point reduction in Word Error Rate (WER)** compared to baseline OCR without such preprocessing. This substantial improvement highlights how **complex layouts significantly degrade OCR accuracy**—unless specialized layout detection and model adaptation are used. The results demonstrate that OCR error rates on complex, multi-column, or highly structured documents can be much higher than those on simpler, single-column text, but can be mitigated by advanced preprocessing.

-----

-----

-----

### Source [45]: https://www.econstor.eu/bitstream/10419/319163/1/00799_2025_Article_413.pdf

Query: What peer-reviewed studies quantify OCR accuracy degradation on complex PDF layouts (e.g., multi-column tables, charts, handwritten notes) compared with plain single-column text documents?

Answer: This publication reports on the same study as [3], confirming the **quantitative improvements** achieved by combining structure detection with OCR fine-tuning. The authors observe a **drop in OCR error rates by 15.68 percentage points (CER) and 19.95 percentage points (WER)** when advanced machine learning techniques are used to preprocess complex historical documents prior to OCR. This underscores the **clear degradation in OCR accuracy** on documents with complex layouts and the effectiveness of modern machine learning approaches in reducing this degradation. The findings reinforce that **plain, single-column documents yield substantially lower error rates** than complex, multi-structured ones—unless dedicated layout analysis is performed.

-----

-----

-----

### Source [46]: https://ijireeice.com/wp-content/uploads/2025/04/IJIREEICE.2025.13435.pdf

Query: What peer-reviewed studies quantify OCR accuracy degradation on complex PDF layouts (e.g., multi-column tables, charts, handwritten notes) compared with plain single-column text documents?

Answer: This peer-reviewed article discusses OCR systems that utilize neural networks to enhance recognition of **complex document layouts**, including varied fonts, handwriting, noisy images, and low resolutions. It highlights that such systems offer **higher accuracy** and improved reliability for extracting text from documents with **multi-column layouts, tables, and handwritten notes**, compared to traditional OCR solutions. The article notes that these advancements allow for better handling of non-standard and degraded documents, but does not provide specific quantitative accuracy degradation figures. Instead, it emphasizes the *main advantages*—greater adaptability and accuracy—of advanced neural network-based OCR in complex scenarios, implicitly contrasting this with the known challenges and degraded accuracy of traditional OCR on such layouts.

-----

-----

</details>

<details>
<summary>What official AWS and Microsoft Azure guidelines compare Base64, raw-byte, and signed-URL methods for sending images to multimodal AI endpoints, including file-size limits and latency considerations?</summary>

### Source [47]: https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-multimodal

Query: What official AWS and Microsoft Azure guidelines compare Base64, raw-byte, and signed-URL methods for sending images to multimodal AI endpoints, including file-size limits and latency considerations?

Answer: Azure's official guidelines for multimodal endpoints allow you to send images using either **base64-encoded bytes or a blob URL**. The relevant request body parameter for the image is named `content` (for base64) or `blobUrl` (for signed URL). The endpoint will **refuse the request if both are provided simultaneously**.

Key points from the documentation:
- **File size and dimension limits:** The maximum allowed image size is **4 MB** and the maximum dimensions are **7,200 x 7,200 pixels**. The minimum image size is **50 x 50 pixels**.
- **Content submission methods:** 
  - `content`: Accepts a base64-encoded string of the image.
  - `blobUrl`: Accepts a URL (typically pointing to Azure Blob Storage) where the image can be accessed.
- **Latency considerations:** While not explicitly stated, using a `blobUrl` (i.e., signed URL) is generally more efficient for large images because the server fetches the image directly, avoiding the overhead of encoding large files into base64 and embedding them in the request payload.
- If both `content` and `blobUrl` are provided, the request fails.
- There is no mention of raw-byte submission; only base64 or URL is supported.

No explicit latency comparison is given, but the design suggests that for **large files, using a blob URL is recommended** to stay within size limits and minimize payload overhead[1].

-----

-----

-----

### Source [49]: https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/how-to/image-retrieval

Query: What official AWS and Microsoft Azure guidelines compare Base64, raw-byte, and signed-URL methods for sending images to multimodal AI endpoints, including file-size limits and latency considerations?

Answer: Azure Computer Vision's vectorization API accepts images via:
- **URL:** The `"url"` parameter is used to supply a remote image location (e.g., a signed URL).
- **Binary data:** For local images, the binary data is placed directly in the HTTP request body.

While the documentation does not specify limits or latency comparisons, it implies that for local images, *raw-byte* (binary) submission is supported, whereas for remote images, using a URL is standard. There is no mention of base64-encoding for this API; instead, binary upload is the alternative to URL submission.

The documentation notes that the Vision Studio (web interface) limits batch processing to 500 images but does not specify per-image file size limits.

-----

-----

</details>

<details>
<summary>Which published case studies document multimodal AI agents used in medical diagnostics or healthcare document analysis, and what improvements over text-only solutions were reported?</summary>

### Source [51]: https://www.puppyagent.com/blog/How-Multimodal-AI-Agents-Are-Revolutionizing-Healthcare-Diagnostics

Query: Which published case studies document multimodal AI agents used in medical diagnostics or healthcare document analysis, and what improvements over text-only solutions were reported?

Answer: This source documents **real-world applications of multimodal AI agents in healthcare diagnostics**. These systems integrate multiple data types—including images (such as X-rays and MRIs), patient records, and lab results—to produce more accurate and reliable diagnoses. During the COVID-19 pandemic, AI tools that combined chest X-rays, CT scans, and patient symptoms achieved over **90% diagnostic accuracy**, outperforming many text-only solutions. In MRI analysis, deep learning models highlighted abnormalities and provided preliminary assessments, reducing the risk of oversight and improving diagnostic reliability.

Multimodal AI agents were found to **reduce diagnostic errors** by analyzing diverse data types simultaneously, leading to a reported **20% improvement in predicting patient deterioration** compared to traditional or text-only systems. These agents also simulate the input of multiple medical specialists, aggregating insights for a holistic diagnosis. This multidisciplinary approach ensures that critical details are less likely to be overlooked, thus enhancing the quality and comprehensiveness of care.

-----

-----

-----

### Source [52]: https://arxiv.org/html/2503.18968v1

Query: Which published case studies document multimodal AI agents used in medical diagnostics or healthcare document analysis, and what improvements over text-only solutions were reported?

Answer: This source presents **MedAgent-Pro**, an evidence-based, multimodal reasoning agentic system designed for medical diagnostics. MedAgent-Pro integrates retrieved medical guidelines and expert tools to generate reliable and explainable diagnoses. The system features a hierarchical workflow: at the task level, multimodal large language models (MLLMs) plan diagnostic procedures by integrating clinical criteria; at the case level, expert models analyze both quantitative and qualitative indicators, including visual evidence.

MedAgent-Pro was evaluated on both **2D and 3D multimodal medical diagnostic tasks**, where it achieved **state-of-the-art performance**, surpassing both general MLLMs and task-specific solutions. Case studies demonstrated that MedAgent-Pro provided **improved interpretability and reliability** over text-only or unimodal approaches, as diagnoses are supported by both clinical literature and visual artifacts. The source emphasizes the system’s ability to deliver **accurate, explainable** diagnoses and highlights improved decision-making through the integration of multimodal data.

-----

-----

-----

### Source [53]: https://www.akira.ai/blog/multi-modal-in-healthcare

Query: Which published case studies document multimodal AI agents used in medical diagnostics or healthcare document analysis, and what improvements over text-only solutions were reported?

Answer: This source discusses the **integration of multimodal AI agents in healthcare diagnostics and document analysis**. Akira AI’s approach uses multiple specialized agents to process data from sources such as electronic health records (EHRs), medical imaging, genomic data, real-time monitoring from wearables, and patient surveys. Each type of data is analyzed by a domain-specific agent (e.g., Image Analysis Agent for scans, Genomic Analysis Agent for genetic markers), and insights are aggregated by a Master Orchestrator Agent.

This architecture allows for **more accurate diagnostics, predictive insights, and timely medical interventions** compared to text-only systems. The system’s ability to combine structured (e.g., lab results) and unstructured data (e.g., medical documents, images) leads to a **more comprehensive health analysis**. Predictive modeling agents use the aggregated data to assess health risks and predict disease progression, supporting earlier and more personalized interventions than traditional, single-modality approaches.

-----

-----

-----

### Source [54]: https://research.google/blog/amie-gains-vision-a-research-ai-agent-for-multi-modal-diagnostic-dialogue/

Query: Which published case studies document multimodal AI agents used in medical diagnostics or healthcare document analysis, and what improvements over text-only solutions were reported?

Answer: This source introduces **multimodal AMIE**, a conversational AI agent that can request, interpret, and reason about visual medical information in diagnostic dialogues. In a remote expert study involving 105 case scenarios with patient actors, AMIE handled multimodal consultations where patients could upload artifacts like skin photos, closely simulating real-world telemedicine interactions.

The evaluation showed that **AMIE matched or exceeded the performance of primary care physicians (PCPs)** in interpreting multimodal data. AMIE outperformed PCPs in diagnostic accuracy, management reasoning, and communication skills, and produced **more accurate and comprehensive differential diagnoses**. Both patient actors and specialist physicians (in dermatology, cardiology, and internal medicine) rated AMIE higher on most evaluation criteria. Compared to text-only or physician-only approaches, AMIE’s use of multimodal data led to **higher consultation quality** and better patient outcomes in the simulated setting.

-----

</details>

<details>
<summary>Where can I find benchmark results (Recall@k, nDCG) that directly compare ColPali’s visual document retrieval to OCR-based RAG pipelines on the ViDoRe dataset?</summary>

### Source [55]: https://arxiv.org/html/2407.01449v2

Query: Where can I find benchmark results (Recall@k, nDCG) that directly compare ColPali’s visual document retrieval to OCR-based RAG pipelines on the ViDoRe dataset?

Answer: The official ColPali paper introduces the **ViDoRe benchmark** to evaluate visually rich document retrieval systems, stating that ColPali "outperforms all other retrieval systems on ViDoRe while being fast and end-to-end trainable." The paper emphasizes that ColPali leverages visual features alone, in contrast to OCR-based retrieval augmented generation (RAG) pipelines, and demonstrates superior performance. The authors specifically mention:

> "ColPali largely outperforms the best existing document retrieval methods while enabling faster corpus indexing time and maintaining low querying latencies..."[1]

However, this source does **not provide explicit Recall@k or nDCG values** in the provided content, nor does it present a table or figure directly comparing ColPali to OCR-based RAG pipelines on the ViDoRe dataset. It does confirm that such benchmarking was performed and results were favorable for ColPali, but readers seeking precise numerical benchmarks or side-by-side comparisons would need to consult the full paper or its supplementary materials.

-----

-----

</details>

<details>
<summary>What technical resources explain and benchmark the ColBERT late-interaction architecture, highlighting why multi-vector retrieval outperforms single-vector dense retrievers in large-scale search?</summary>

### Source [59]: https://jina.ai/news/what-is-colbert-and-late-interaction-and-why-they-matter-in-search/

Query: What technical resources explain and benchmark the ColBERT late-interaction architecture, highlighting why multi-vector retrieval outperforms single-vector dense retrievers in large-scale search?

Answer: ColBERT’s **late interaction** architecture is characterized by the independent encoding of queries and documents, with their interaction occurring only after both have been processed. This is distinct from **early interaction** models, such as BERT and DPR, where embeddings interact during or before encoding, which increases computational complexity and reduces scalability for large-scale search. The late interaction approach lets ColBERT pre-compute document representations and perform lightweight, efficient interaction steps during retrieval, significantly improving retrieval speed and scalability for large document collections. This architecture underpins ColBERT’s ability to handle large-scale search tasks more efficiently than single-vector dense retrievers, which would require exhaustive pairwise computation between query and document embeddings.

-----

-----

-----

### Source [60]: https://people.eecs.berkeley.edu/~matei/papers/2020/sigir_colbert.pdf

Query: What technical resources explain and benchmark the ColBERT late-interaction architecture, highlighting why multi-vector retrieval outperforms single-vector dense retrievers in large-scale search?

Answer: The original **ColBERT** paper introduces a **late interaction architecture** where both the query and document are independently encoded using BERT. ColBERT then employs an efficient interaction step that models fine-grained similarity between the sets of token-level contextualized embeddings. This late interaction allows ColBERT to combine the expressiveness of deep language models with the ability to pre-compute and index document vectors offline, enabling rapid query processing. The model is designed to be pruning-friendly, allowing use of vector-similarity indexes to retrieve passages from millions of documents efficiently. Experimental benchmarks in the paper show that **ColBERT’s multi-vector approach outperforms non-BERT baselines and achieves effectiveness competitive with full cross-encoder models** but with significantly better scalability and speed, highlighting why multi-vector retrieval is superior to single-vector dense retrieval in large-scale scenarios.

-----

-----

-----

### Source [61]: https://aclanthology.org/2022.naacl-main.272.pdf

Query: What technical resources explain and benchmark the ColBERT late-interaction architecture, highlighting why multi-vector retrieval outperforms single-vector dense retrievers in large-scale search?

Answer: **ColBERTv2** builds upon the late interaction design of ColBERT, where queries and passages are independently encoded into **multi-vector representations**—specifically, token-level embeddings. The model performs scalable, token-level relevance computations between these sets of vectors. Studies cited in the paper show that this decomposition into token-level interactions improves retrieval effectiveness compared to single-vector methods. However, it also increases storage requirements. ColBERTv2 addresses this by introducing an aggressive residual compression mechanism and a denoised supervision strategy, improving both the quality and scalability of late interaction methods. The paper benchmarks ColBERTv2, demonstrating that its multi-vector late interaction approach consistently outperforms single-vector dense retrievers on key retrieval tasks.

-----

-----

-----

### Source [62]: https://arxiv.org/html/2408.16672v3

Query: What technical resources explain and benchmark the ColBERT late-interaction architecture, highlighting why multi-vector retrieval outperforms single-vector dense retrievers in large-scale search?

Answer: This paper discusses **multi-vector dense retrieval** models like ColBERT, highlighting that ColBERT’s **late interaction scoring** approximates the powerful joint query-document attention seen in cross-encoders while maintaining inference efficiency closer to traditional dense models due to its bi-encoder architecture. The paper describes a new multilingual and long-context ColBERT variant (Jina-ColBERT-v2), which demonstrates strong performance on a variety of retrieval tasks. The results underscore that the multi-vector, late interaction design is crucial for achieving both high accuracy and efficiency in large-scale and multilingual search settings, outperforming single-vector dense retrievers by capturing finer-grained relevance signals.

-----

-----

</details>


## Sources Scraped From Research Results

<details>
<summary>80% Faster Financial Statement Processing for a Fortune 500 Company</summary>

# 80% Faster Financial Statement Processing for a Fortune 500 Company

Applications

Data extraction, process automation, market intelligence, credit ratings, credit memo preparation

###### 80%

acceleration for 10-K & 10-Q document processing

###### 100% accuracy

for table and figure detection

###### Near 95%

table data extraction

## Challenge

## Solution

## Results

Applications

Data extraction, process automation, market intelligence, credit ratings, credit memo preparation

Use Case

Financial statement processing

</details>

<details>
<summary>ColPali: Efficient Document Retrieval with Vision Language Models</summary>

# ColPali: Efficient Document Retrieval with Vision Language Models

## 1 Introduction

https://arxiv.org/html/2407.01449v1/extracted/5696231/images/similarity_maps/similarity_map_energy.png  
Figure 1: For each term in a user query, ColPali identifies the most relevant document image patches (highlighted zones) and computes a query-to-page matching score. We can then swiftly retrieve the most relevant documents from a large pre-indexed corpus.

https://arxiv.org/html/2407.01449v1/extracted/5696231/images/final_architecture.png  
Figure 2: ColPali simplifies document retrieval w.r.t. standard retrieval methods while achieving stronger performances with better latencies. Latencies and results are detailed in section 5 and subsection B.5.

Document Retrieval consists in matching a user query to relevant documents in a given corpus. It is central to many industrial applications, either as a standalone ranking system (search engines) or as part of more complex information extraction or Retrieval Augmented Generation (RAG) pipelines.

Over recent years, pretrained language models have enabled large improvements in text embedding models. In practical industrial settings, however, the main performance bottleneck for efficient document retrieval is not in embedding model performance but in the prior data ingestion pipeline. To index a standard PDF document, many steps are required. First, PDF parsers or Optical Character Recognition (OCR) systems are used to extract words from the pages. Document layout detection models can then be run to segment paragraphs, titles, and other page objects such as tables, figures, and headers. A chunking strategy is then defined to group text passages with some semantical coherence, and modern retrieval setups may even integrate a captioning step to describe visually rich elements in a natural language form, more suitable for embedding models.
In our experiments (Table 2), we typically find that optimizing the ingestion pipeline yields much greater performance on visually rich document retrieval than optimizing the text embedding model.

Contribution 1: ViDoRe.
In this work, we argue that document retrieval systems should not be evaluated solely on the capabilities of text embedding models (Bajaj et al., 2016; Thakur et al., 2021; Muennighoff et al., 2022), but should also consider the context and visual elements of the documents to be retrieved. To this end, we create and openly release ViDoRe, a comprehensive benchmark to evaluate systems on page-level document retrieval with a wide coverage of domains, visual elements, and languages. ViDoRe targets practical document retrieval settings, in which user queries may require both textual and visual understanding to be correctly matched to relevant documents. We highlight the shortcomings of current text-centric systems in these settings. The benchmark leaderboard is hosted publicly at [https://huggingface.co/spaces/vidore/vidore-leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard) to encourage further developments.

Contribution 2: ColPali.
We propose a novel model architecture and training strategy based on Vision Language Models (VLMs) to efficiently index documents purely from their visual features, allowing for subsequent fast query matching with late interaction mechanisms (Khattab and Zaharia, 2020). Our method, ColPali, outperforms all other retrieval systems on ViDoRe while being fast and end-to-end trainable. We release models and code at [https://huggingface.co/vidore](https://huggingface.co/vidore).

## 2 Problem Formulation & Related Work

**Problem Setting**  
In our setting, a retrieval system scores how relevant a document d from corpus 𝒟 is with respect to a query q. Computing the similarity score s(q,d) ∈ ℝ+ for each of the |𝒟| documents in the corpus creates a ranking we can use to extract the most relevant documents. In this work, we focus on page-level retrieval: given a query, is the correct document page retrieved by the system? For coherence with existing literature, we further use the term document to refer to individual pages, i.e. the atomic retrieved elements in our setting. As we focus on practical industrial retrieval applications (RAG, search engines) with potentially large corpora sizes, latency constraints are imposed on scoring systems. Most current retrieval systems can be decomposed into (1) an offline indexation phase in which a document index is built and (2) an online querying phase in which a query is matched to documents from the index and where low latency is vital to the user experience.

Efficient document retrieval systems exhibit joint properties of high retrieval performance (R1), low latency during querying (R2), and high throughput during indexation (R3).

### 2.1 Textual Retrieval Methods

**Document Retrieval in Text Space**  
Statistical methods based on word frequency like TF-IDF (Sparck Jones, 1972) and BM25 (Robertson et al., 1994) are still widely used due to their simplicity and efficiency. More recently, neural embedding models based on fine-tuned large language models display state-of-the-art performance on a variety of text embedding tasks and top the retrieval leaderboards (Muennighoff et al., 2022).

**Neural Retrievers**  
In bi-encoder models (Reimers and Gurevych, 2019; Karpukhin et al., 2020; Wang et al., 2022), documents are independently mapped offline to a dense vector space. Queries are embedded online and matched to documents through a fast cosine distance computation.
A slower, but slightly more performant alternative, cross-encoder systems (Wang et al., 2020; Cohere, 2024) concatenate query and document as a single input sequence and iteratively attribute matching scores to each possible combination. This enables full attention computation between query and document terms but comes at the cost of computational efficiency, as |𝒟| encoding passes must be done online.

**Multi-Vector retrieval via late interaction**  
In the late interaction paradigm (Khattab and Zaharia, 2020), an embedding is pre-computed and indexed per document token. At runtime, similarity can be computed with individual query token embeddings. The idea is to benefit from the rich interaction between individual query and document terms while taking advantage of the offline computation and fast query matching enabled by bi-encoders.

**Retrieval Evaluation**  
Although benchmarks and leaderboards have been developed to evaluate text embedding models (Thakur et al., 2021; Muennighoff et al., 2022), as previously stated, much of the performance improvements in industrial use cases of embedding models stem from the prior data ingestion pipeline. While documents often rely on visual elements to more efficiently convey information to human readers, text-only systems barely tap into these visual cues.

To our knowledge, no benchmark evaluates document retrieval methods by considering both textual and visual document features like a human would.

### 2.2 Integrating Visual features

**Contrastive Vision Language Models**  
Mapping latent representations of textual content to corresponding representations of visual content has been done by aligning disjoint visual and text encoders through contrastive losses (Radford et al., 2021; Zhai et al., 2023). While some OCR capabilities exist in these models, the visual component is often not optimized for text understanding.
The Fine-grained Interactive Language-Image Pre-training (Yao et al., 2021) framework extends the late interaction mechanism to cross-modal vision-language models, relying on max similarity operations between text tokens and image patches.

**Visually Rich Document Understanding**  
To go beyond text, some document-focused models jointly encode text tokens alongside visual or document layout features (Appalaraju et al., 2021; Kim et al., 2021; Huang et al., 2022; Tang et al., 2022).
Large Language transformer Models (LLMs) with strong reasoning capabilities have recently been combined with Vision Transformers (ViTs) (Dosovitskiy et al., 2020) to create VLMs (Alayrac et al., 2022; Liu et al., 2023b; Bai et al., 2023; Laurençon et al., 2024) where image patch vectors from contrastively trained ViT models (Zhai et al., 2023) are fed as input embeddings to the language model and concatenated with the text-token embeddings.

**PaliGemma**  
The PaliGemma-3B model (Lucas Beyer* et al., 2024) extends concepts from Pali3 (Chen et al., 2023), and projects SigLIP-So400m/14 (Alabdulmohsin et al., 2023) patch embeddings into Gemma-2B’s text vector space (Gemma Team et al., 2024). Along with its reasonable size w.r.t. other performant VLMs, an interesting property of PaliGemma’s text model is that it is fine-tuned with full-block attention on the prefix (instruction text and image tokens).

VLMs display enhanced capabilities in Visual Question Answering, captioning, and document understanding (Yue et al., 2023), but are not optimized for retrieval tasks.

## 3 The ViDoRe Benchmark

Existing benchmarks for contrastive vision-language models primarily evaluate retrieval for natural images (Lin et al., 2014; Borchmann et al., 2021; Thapliyal et al., 2022). On the other hand, textual retrieval benchmarks (Muennighoff et al., 2022) are evaluated at the textual passage level and are not tailored for document retrieval tasks. We fill the gap with ViDoRe, a comprehensive benchmark for document retrieval using visual features.

### 3.1 Benchmark Design

ViDoRe is designed to comprehensively evaluate retrieval systems on their capacity to match queries to relevant documents at the page level. This benchmark encompasses multiple orthogonal subtasks, with focuses on various modalities - text, figures, infographics, tables; thematic domains - medical, business, scientific, administrative; or languages - English (eng), French (fra).

| Dataset | \# Queries | Domain |
| --- | --- | --- |
| Academic Tasks |  |  |
| DocVQA (eng) | 500 (500) | Industrial |
| InfoVQA (eng) | 500 (500) | Infographics |
| TAT-DQA (eng) | 1600 (1600) | Varied Modalities |
| arXiVQA (eng) | 500 (500) | Scientific Figures |
| TabFQuAD (fra) | 210 (210) | Tables |
| Practical Tasks |  |  |
| Energy (eng) | 100 (1000) | Scientific |
| Government (eng) | 100 (1000) | Administrative |
| Healthcare (eng) | 100 (1000) | Medical |
| AI (eng) | 100 (1000) | Scientific |
| Shift Project (fra) | 100 (1000) | Environment |

Table 1: ViDoRe comprehensively evaluates multimodal retrieval methods. The size of the document corpus is indicated in parentheses.

**Academic Tasks**  
We repurpose widely used visual question-answering benchmarks for retrieval tasks: for each page-question-answer triplet, we use the question as the query, and the associated page as the gold document (Table 1). These academic datasets either focus on single specific modalities (Mathew et al., 2020, 2021; Li et al., 2024) or target more varied visually rich documents (Zhu et al., 2022). Moreover, we consider TabFQuAD, a human-labeled dataset on tables extracted from French industrial PDF documents released with this work.

**Practical tasks**  
We construct topic-specific retrieval benchmarks spanning multiple domains to go beyond repurposed QA datasets and evaluate retrieval in more realistic industrial situations (e.g. RAG). To achieve this, we collect publicly accessible PDF documents and generate queries pertaining to document pages using Claude-3 Sonnet, a high-quality proprietary vision-language model (Anthropic, 2024). In total, we collect 1,000 document pages per topic, which we associate with 100 queries extensively filtered for quality and relevance by human annotators. The corpus topics are intentionally specific to maximize syntactic proximity between documents, creating challenging retrieval tasks and covering an array of orthogonal domains (Table 1). Query-page pair examples are shown in Appendix E. Answers are generated alongside queries to (1) ground queries and improve their quality and (2) provide resources to foster future work.

**Evaluation Metrics**  
We evaluate performance on our benchmark (Requirement R1) using standard metrics from the retrieval literature (NDCG, Recall@K, MRR). We report NDCG@5 values as the main performance metric in this work and release the complete sets of results along with the models. To validate compliance with practical industrial constraints, we also consider query latencies (R2) and indexing throughputs (R3).

### 3.2 Assessing Current Systems

**Unstructured**
We evaluate retrieval systems representative of those found in standard industrial RAG pipelines. As is common practice, we rely on the Unstructured off-the-shelf tool in the highest resolution settings to construct high-quality text chunks from PDF documents. Unstructured orchestrates the document parsing pipeline, relying on deep learning vision models to detect titles and document layouts (Ge et al., 2021), OCR engines (Smith, 2007) to extract text in non-native PDFs, specialized methods or models to detect and reconstruct tables, and implements a chunking strategy (by-title) that leverages the detected document structure to preserve section boundaries when concatenating texts. As is common practice, in our simplest Unstructured configuration (text-only), only textual elements are kept, and figures, images, and tables are considered noisy information and are filtered out.

**Unstructured + X**
While Unstructured is a strong baseline by itself, we further augment Unstructured’s output by integrating the visual elements. In (+ OCR), tables, charts, and images are run through an OCR engine, processed by Unstructured, and chunked independently. In (+ Captioning), we set up a fully-fledged captioning strategy (Zhao et al., 2023), in which we feed visual elements to a strong proprietary Vision Language Model (Claude-3 Sonnet (Anthropic, 2024)) to obtain highly detailed textual descriptions of the elements.
Both strategies aim to integrate visual elements in the retrieval pipeline but incur significant latency and resource costs (subsection 5.2).

**Embedding Model**
To embed textual chunks, we evaluate Okapi BM25, the de facto standard sparse statistical retrieval method, and the dense encoder of BGE-M3 (Chen et al., 2024), a multilingual neural method with SOTA performance in its size category. Chunks are embedded and scored independently, and page-level scores are obtained by max-pooling over the page’s chunk scores. We empirically validated the max-pooling strategy over sub-page chunks to be more effective than concatenating all page chunks before embedding pagewise.

**Contrastive VLMs**
We also evaluate the strongest available vision-language embedding models: Jina CLIP (Koukounas et al., 2024), Nomic Embed Vision (Nomic, 2024), and SigLIP-So400m/14 (Alabdulmohsin et al., 2023).

**Results**  
From a performance perspective, best results are obtained by combining the Unstructured parser with visual information, either from captioning strategies or by running OCR on the visual elements (Table 2). Little difference is seen between BM25 and BGE-M3 embeddings highlighting the visual information bottleneck. Contrastive VLMs lag behind. Beyond retrieval performance (R1), the indexing latencies (R2) reported in Figure 3 illustrate that PDF parsing pipelines can be very lengthy, especially when incorporating OCR or captioning strategies. Querying latencies at runtime (R3) are very good for all evaluated systems (≤22 ms on NVIDIA L4) due to fast query encoding and cosine similarity matching.

https://arxiv.org/html/2407.01449v1/x1.png  
Figure 3: Offline indexing with ColPali is much simpler and faster compared to standard retrieval methods. Indexing speeds reported are computed on Nvidia L4 GPUs and detailed in subsection B.5.

## 4 Late interaction based Vision Retrieval

### 4.1 Architecture

**Vision-Language Models**  
Encouraged by their strong document understanding capabilities, we propose adapting recent VLMs for retrieval. The key concept is to leverage the alignment between output embeddings of text and image tokens acquired during multi-modal finetuning.
To this extent, we introduce ColPali, a Paligemma-3B extension that is capable of generating ColBERT-style multi-vector representations of text and images (Figure 2).
PaliGemma-3B is a strong candidate due to its small size, the many released checkpoints fine-tuned for different image resolutions and tasks, and the promising performances on various document understanding benchmarks.
We add a projection layer to map the output language modeling embeddings to a vector space of reduced dimension D=128 as used in the ColBERT paper (Khattab and Zaharia, 2020) to keep lightweight bag-of-embedding representations.

**Late Interaction**  
Given query q and document d, we denote as 𝐄𝐪∈ℝNq×D and 𝐄𝐝∈ℝNd×D their respective multi-vector representation in the common embedding space ℝD. The late interaction operator, LI(q,d), is the sum over all query vectors of its maximum dot product ⟨·|·⟩ with each document embedding vector.

|     |     |     |     |
| --- | --- | --- | --- |
|  | LI(q,d)=∑i∈[1,Nq] max j∈[1,Nd]⟨𝐄𝐪(i)|𝐄𝐝(j)⟩ |  | (1) |

**Contrastive Loss**  
The Late Interaction operation is fully differentiable, enabling backpropagation.
Let a batch {qk,dk}k∈[1,b] composed of b query-page pairs, where for all k∈[1,b], the document page dk is the document corresponding to query qk.
Following Khattab and Zaharia (2020), we define our in-batch contrastive loss ℒ as the softmaxed cross-entropy of the positive scores sk+ = LI(dk,qk) w.r.t. the maximal negative scores sk− = max_{l,l≠k} LI(qk,pl).

### 4.2 Model training

**Dataset**  
Our training dataset of 127,460 query-page pairs is comprised of train sets of openly available academic datasets (63%) and a synthetic dataset made up of pages from web-crawled PDF documents and augmented with VLM-generated (Claude-3 Sonnet) pseudo-questions (37%). Our training set is fully English by design, enabling us to study zero-shot generalization to non-English languages. We explicitly verify no multi-page PDF document is used both ViDoRe and in the train set to prevent evaluation contamination. A validation set is created with 2% of the samples to tune hyperparameters.

**Parameters**  
All models are trained for 1 epoch on the train set. Unless specified otherwise, we train models in bfloat16 format, use low-rank adapters (LoRA, Hu et al., 2021) with α=32 and r=32 on the transformer layers from the language model, as well as the final randomly initialized projection layer, and use a paged_adamw_8bit optimizer. We train on an 8 GPU setup with data parallelism, a learning rate of 5e−5 with linear decay with 2.5% warmup steps, and a batch size of 32.

**Query Augmentation**  
As in Khattab and Zaharia (2020), we append 5 <unused0> tokens to the query tokens to serve as a soft, differentiable query expansion or re-weighting mechanism.

## 5 Results

|     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | ArxivQ | DocQ | InfoQ | TabF | TATQ | Shift | AI | Energy | Gov. | Health. | Avg. |
| Unstructured Text only |  |  |  |  |  |  |  |  |  |  |  |
| \- BM25 | - | 34.1 | - | - | 44.0 | 59.6 | 90.4 | 78.3 | 78.8 | 82.6 | - |
| \- BGE-M3 | - | 28.4↓ | - | - | 36.1↓ | 68.5↑ | 88.4↓ | 76.8↓ | 77.7↓ | 84.6↑ | - |
| Unstructured \+ OCR |  |  |  |  |  |  |  |  |  |  |  |
| \- BM25 | 31.6 | 36.8 | 62.9 | 46.5 | 62.7 | 64.3 | 92.8 | 85.9 | 83.9 | 87.2 | 65.5 |
| \- BGE-M3 | 31.4↓ | 25.7↓ | 60.1↓ | 70.8↑ | 50.5↓ | 73.2↑ | 90.2↓ | 83.6↓ | 84.9↑ | 91.1↑ | 66.1↑ |
| Unstructured \+ Captioning |  |  |  |  |  |  |  |  |  |  |  |
| \- BM25 | 40.1 | 38.4 | 70.0 | 35.4 | 61.5 | 60.9 | 88.0 | 84.7 | 82.7 | 89.2 | 65.1 |
| \- BGE-M3 | 35.7↓ | 32.9↓ | 71.9↑ | 69.1↑ | 43.8↓ | 73.1↑ | 88.8↑ | 83.3↓ | 80.4↓ | 91.3↑ | 67.0↑ |
| Contrastive VLMs |  |  |  |  |  |  |  |  |  |  |  |
| Jina-CLIP | 25.4 | 11.9 | 35.5 | 20.2 | 3.3 | 3.8 | 15.2 | 19.7 | 21.4 | 20.8 | 17.7 |
| Nomic-vision | 17.1 | 10.7 | 30.1 | 16.3 | 2.7 | 1.1 | 12.9 | 10.9 | 11.4 | 15.7 | 12.9 |
| SigLIP (Vanilla) | 43.2 | 30.3 | 64.1 | 58.1 | 26.2 | 18.7 | 62.5 | 65.7 | 66.1 | 79.1 | 51.4 |
| Ours |  |  |  |  |  |  |  |  |  |  |  |
| SigLIP (Vanilla) | 43.2 | 30.3 | 64.1 | 58.1 | 26.2 | 18.7 | 62.5 | 65.7 | 66.1 | 79.1 | 51.4 |
| BiSigLIP (+fine-tuning) | 58.5↑ | 32.9↑ | 70.5↑ | 62.7↑ | 30.5↑ | 26.5↑ | 74.3↑ | 73.7↑ | 74.2↑ | 82.3↑ | 58.6↑ |
| BiPali (+LLM) | 56.5↓ | 30.0↓ | 67.4↓ | 76.9↑ | 33.4↑ | 43.7↑ | 71.2↓ | 61.9↓ | 73.8↓ | 73.6↓ | 58.8↑ |
| ColPali (+Late Inter.) | 79.1↑ | 54.4↑ | 81.8↑ | 83.9↑ | 65.8↑ | 73.2↑ | 96.2↑ | 91.0↑ | 92.7↑ | 94.4↑ | 81.3↑ |

Table 2: Comprehensive evaluation of baseline models and our proposed method on ViDoRe. Results are presented using NDCG@5 metrics, and illustrate the impact of different components. Text-only metrics are not computed for benchmarks with only visual elements.

### 5.1 Performance (R1)

We iteratively construct ColPali, starting from an off-the-shelf SigLIP model (Table 2).

**BiSigLIP: Improving a strong model**  
SigLIP is a strong vision-language bi-encoder model, pretrained on the English split of WebLI (Chen et al., 2023), a corpus of billions of image-text pairs. We find that SigLIP largely outperforms both Jina CLIP and Nomic-vision on document retrieval tasks. Further fine-tuning the textual component of this model on our document-oriented dataset (BiSigLIP) yields clear improvements across the board, particularly on figure retrieval (ArxivQA) and table retrieval tasks (TabFQuAD).

**BiPali: Pairing with a language model**  
In the PaliGemma model architecture, SigLIP-generated patch embeddings are fed to a text language model to obtain LLM contextualized output patch embeddings. Note that the SigLIP model used in PaliGemma slightly differs in terms of number patches - 1024 patches for PaliGemma’s vision encoder, and 729 for the standalone SigLIP model. We average pool these representations to obtain a single dense vector, effectively creating a PaliGemma bi-encoder model (BiPali). After fine-tuning on the training dataset, we obtain a model that performs slightly worse in English than the tuned BiSigLIP variant. This can be explained by the fact that contrary to SigLIP, the original PaliGemma is not trained on contrastive matching tasks, but rather on next token prediction. Our contrastive fine-tuning phase on 100K images to transform PaliGemma into a bi-encoder is 5 orders of magnitude smaller than SigLIP’s original contrastive training. However, we see notable improvements in French tasks, indicating that BiPali’s LLM (Gemma 2B) helps multilingual text understanding. This is particularly notable as our training dataset does not contain non-English samples.

**ColPali: Adding Late Interaction**  
One benefit of inputting image patch embeddings through a language model is that they are natively mapped to a latent space similar to textual input (query). This enables leveraging the ColBERT strategy to compute interactions between text tokens and image patches, which enables a step-change improvement in performance compared to BiPali.
Results in Table 2 show that our ColPali model also largely outperforms the strong baselines based on Unstructured and captioning, as well as all evaluated text-image embedding models. The difference is particularly stark on the more visually complex benchmark tasks, such as InfographicVQA, ArxivQA, and TabFQuAD representing respectively infographics, figures, and tables. However, text-centric documents are also better retrieved by the ColPali models across all evaluated domains and languages, making our approach the overall best-performing document-retrieval model.

**Negative Results**  
For extensiveness, we also train ColSigLIP, a late interaction variant of the BiSigLIP model but obtain abysmal performances. We attribute this to the large gaps w.r.t. SigLIP’s pre-training, in which only a pooled latent representation is used in the contrastive loss, which does not optimize the representations of individual patch and token embeddings. Similarly, we train a BiSigLIPPaliGemma variant, in which we retrieve the image representations from the SigLIP model that has been further updated by PaliGemma fine-tuning, and use the text representations from PaliGemma’s text model. After fine-tuning on our dataset, performance is severely inferior to SigLIPVanilla which simply encodes with SigLIP’s original text and vision components. This indicates a logical misalignment between SigLIP embeddings, and Gemma embeddings after PaliGemma training.

### 5.2 Latencies & Memory Footprint

**Online Querying (R2)**  
Logically, querying latencies differ between ColPali and a BGE-M3 embedding model. For BGE, encoding takes about 22 ms for 15 tokens, while encoding a query with ColPali’s language model takes about 30 ms. For smaller corpus sizes, computing the late interaction operation induces marginally small overheads (≈1 ms per 1000 pages in the corpus), and the cosine similarity computation between bi-encoder vectors is even faster. Optimized late interaction engines enable to easily scale corpus sizes to millions of documents with reduced latency degradations.

**Offline Indexing (R3)**
Standard retrieval methods using bi-encoders represent each chunk as a single vector embedding, which is easy to store and fast to compute. However, processing a PDF to get the different chunks is the most time-consuming part (layout detection, OCR, chunking), and using captioning to handle multimodal data will only exacerbate this already lengthy process. On the other hand, ColPali directly encodes pages from their image representation. Although the encoder model is larger than standard retrieval encoders, skipping the preprocessing allows large speedups at indexing (Figure 3).

**Memory Footprint**
Our method requires storing a vector per image patch. We project each PaliGemma vector to a lower dimensional space (D=128) to maximize efficiency, leading to a memory footprint of 256 KB per page. Importantly, the memory footprint of the naive ColBERT indexing strategy can be drastically improved through compression and clustering mechanisms as proposed in the Performance-optimized Late Interaction Driver (Santhanam et al., 2022).

### 5.3 Interpretability

By superimposing the late interaction heatmap on top of the original image, we can visualize the most salient image patches with respect to each term of the query, yielding interpretable insights into model focus zones. As epitomized in Figure 1, we observe ColPali exhibits strong OCR capabilities as both the words "hourly" and "hours" present a high similarity score with the query token <_hour>. We also note particular focus on other non-trivial image features such as the x-axis representing hours being salient. Other visualization examples with similar trends of the model transcending pure OCR are shown in Appendix C.

## 6 Ablation study

https://arxiv.org/html/2407.01449v1/x2.png  
Figure 4: Relative NDCG@5 performance gain w.r.t. the default ColPali (1024 patches). TabFQuAD fine-tuning measures the performance difference on the TabFQuAD task after the introduction of targeted data in the training set. All other results refer to performance deltas averaged on all ViDoRe tasks.

**Should we scale models or patch numbers?**  
We train a variant of PaliGemma with half the number of image patches (512). While there is a clear performance degradation w.r.t. to the 1024-patch ColPali model (Figure 4), memory usage is much lower. While another PaliGemma variant exists with 2048 patches, the different training datamix and the large memory requirements make this model impractical for both training and inference time.
As an alternative to PaliGemma, we train Idefics2-8B (Laurençon et al., 2024), a VLM with a similar architecture and based on a Mistral-7B (Jiang et al., 2023) language backbone and a SigLIP vision encoder paired with a perceiver resampler. The most notable differences with PaliGemma lie in the size of the language model (2B and 7B resp.) and the number of image patches (between 512 and 2048 for PaliGemma, and 60 post-resampling for Idefics2). Our results (Figure 4) suggest language model size has a strong impact on performance, and along with the trained resampler enables more efficient representations for smaller numbers of image embeddings - ColIdefics2 with 60 patches edges out ColPali with 512 patches.
Scaling the number of patches of the smaller ColPali model from 512 to 1024, enables largely surpassing the 60-patch ColIdefics2 while being about twice as fast in terms of training and inference latency.
These results suggest there are tradeoffs between performance (R1), latencies during online querying (R2) and offline indexation phases (R3), and index memory size.

**Should we fine-tune the vision component?**  
We run our contrastive finetuning on a ColPali model in which we also train the vision encoder and the projection layer. Results in Figure 4 show this leads to no significant improvements.

**Do "query augmentation" tokens help?**  
In ColBERT, special tokens are concatenated to the input query to serve as soft query augmentation buffers. Training without these tokens, we observe no significant performance difference (Figure 4) in the English benchmarks. However, performance on the French tasks seems to improve (Table 5).

**Is the Pairwise CE loss best?**  
Training with an in-batch negative contrastive loss, instead of the pairwise CE loss that only considers the hardest negative sample, leads to a slight performance degradation (−2.4%) on the aggregated benchmark.

**Can the model adapt to new tasks?**  
Contrary to more complex multi-step retrieval pipelines, ColPali can be trained end-to-end, directly optimizing the downstream retrieval task which greatly facilitates fine-tuning to boost performance on specialized domains, multilingual retrieval, or specific visual elements the model struggles with. To demonstrate, we add 1552 samples representing French tables and associated queries to the training set. This represents the only French data in the training set, with all other examples being kept unchanged. We see significant NDCG@5 improvements (Figure 4) and even starker Recall@1 gains (+6.63%) on the TabFQuAD benchmark, with no performance degradation on the rest of the benchmark tasks (+0.34%).

## 7 Conclusions

Through the conception of a new benchmark ViDoRe, we established the limits of both modern industrial document retrieval pipelines and off-the-shelf image-text contrastive models for visually rich document retrieval. We introduced ColPali, a novel retrieval model that leverages the latest generative Vision Language models to create highly performing multi-vector embeddings purely from visual document features. ColPali largely outperforms the best existing document retrieval methods while enabling faster corpus indexing time and maintaining low querying latencies, suggesting a very high potential for industrial document retrieval applications. We hope to encourage future work by publicly releasing the ViDoRe benchmark and all models and baselines from our study.

**Future Work.**  
Further performance gains could be obtained by exploring sub-image decomposition (Liu et al., 2023a), optimal image patch resampling strategies (Laurençon et al., 2024), or hard-negative mining.
Subsequently, our vision is to combine visual retrieval and visually grounded query answering to create RAG systems that purely function from visual features.
An interesting line of research could be attempting to generate answers leveraging information stored in the indexed multi-vector patch embeddings.

## Limitations

**Focus.**  
In this work, we evaluate models on document retrieval tasks, covering several modalities (figures, text, tables, infographics). We however primarily focus on PDF-type documents, and evaluating systems on image retrieval with documents stemming from web page screenshots or hand-written documents might be an interesting generalization. We also focus on high-resource languages (English and French) and although we have shown the capacity of the ColPali model to generalize to languages outside of its fine-tuning set, it is unclear how the model would perform on languages that are not as represented in the model’s language backbone. Finally, our setup assumes relevant documents exist, but abstention methods for Information Retrieval systems might be interesting to explore in more practical settings in which confidence estimation might be important (Gisserot-Boukhlef et al., 2024).

**Support.**  
This work relies on multi-vector retrieving derived from the ColBERT late interaction mechanism. Although some vector databases support late interaction engines (Vespa Engine, RAGatouille, QDrant, colbert.ai), many widely used vector retrieval frameworks do not propose native multi-vector support, and some engineering infrastructure efforts may be required to adapt them to work with ColPali (or ColBERT) models.

**Data.**  
In the creation of ViDoRe, we partially rely on synthetic query generation based on a commercial large language model, which may induce some amount of bias in the generated queries. To compensate for this, we have iterated on the prompting strategy and given real query examples to the models to help ground generation in realistic settings. We have further manually verified all synthetic queries through a lengthy process to validate their relevance and their quality. Our benchmark also includes many benchmark tasks with no synthetic data, and result trends observed between all tasks are correlated, further confirming the coherence of our benchmark design.

## Ethical Considerations

**Carbon Footprint.**
Our work fully leverages prior pretrained models and training is not particularly compute-intensive. Furthermore, we rely on low-rank adapters to further reduce the computational resources needed, both during training and for storage. Overall, a training run represents about 40 hours of Mi250x AMD GPUs. Our experiments, in total, represent 1405 Mi250x GPU hours from highly efficient compute clusters running on low-carbon nuclear energy, representing a total of around 15kg CO2 eq.

**Impact.**
We believe our work could have a strong impact on improving industrial document retrieval systems. Our method is efficient, performs well, and the additional support towards visually rich information from documents could go a long way in unlocking knowledge sources previously difficult to index or query.

**Resource Release.**
For transparency, and to foster future work, we release our comprehensive benchmark under open license and host a public leaderboard ([https://huggingface.co/spaces/vidore/vidore-leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard)). Our models are released under the same usage license as the base model (Gemma Research license for ColPali, Apache2.0 for ColIdefics2) and should be used as intended by the VLM license.

## Appendix A Benchmark Datasets

### A.1 Academic Datasets

**DocVQA** (Mathew et al., 2020) includes collected images from the UCSF Industry Documents Library. Questions and answers were manually annotated.

**InfoVQA** (Mathew et al., 2021) includes infographics collected from the Internet using the search query “infographics”. Questions and answers were manually annotated.

**TAT-DQA** (Zhu et al., 2022) is a large-scale Document VQA dataset that was constructed from publicly available real-world financial reports. It focuses on rich tabular and textual content requiring numerical reasoning. Questions and answers were manually annotated by human experts in finance.

**arXivQA** (Li et al., 2024) is a VQA dataset based on figures extracted from arXiv publications. The questions were generated synthetically using GPT-4 Vision.

**TabFQuAD** (Table French Question Answering Dataset) is designed to evaluate TableQA models in realistic industry settings. We create additional queries to augment the existing human-annotated ones using the same method described in subsection A.2.

### A.2 Practical Datasets

**Methodology.**  
Creating a relevant retrieval dataset close to real use cases is a major challenge as the dataset needs to be both sufficiently large for effective fine-tuning and sufficiently diverse to cover a broad range of modalities (full text, tables, charts, …), domains (industry, healthcare, …), and query-document interactions (extractive questions, open-ended questions, …). Our approach to building this dataset involves several steps: (1) we use a web crawler to collect publicly available documents on various themes and sources, (2) we convert these PDFs into a series of images, one per page, and (3) we generate queries related to each image using a VLM.

**Web-Crawler.**  
We implemented a web crawler to efficiently collect large volumes of documents related to a given topic. The crawler is seeded with a user-defined query (e.g. "artificial intelligence") and then uses GPT-3.5 Turbo to brainstorm related topics and subtopics. This query augmentation strategy aims at both broadening and deepening the search. GPT-3.5 Turbo is further used to generate diverse search queries from each subtopic. This query set is then consumed by a pool of parallel workers whose job is to fetch the associated most relevant documents. We use SerpAPI along with a filetype filter (PDF documents only) to programmatically scrape Google Search rankings. Each file is hashed and stored in a Bloom filter (Bloom, 1970) shared among workers to avoid duplicate documents in the final corpus. Unique scraped files are downloaded, and inserted into a SQLite database along with additional metadata.

**Datamix.**  
Using the web crawler, we collected approximately 1,000 documents for each of the following four seeds: "energy", "government reports", "healthcare industry", and "artificial intelligence". These seeds were meticulously hand-picked to align with real-use cases for retrieval models and visually rich pages. We also removed all documents containing any private information. At this stage, we randomly selected 900 files for the training set and 100 files for the test set, ensuring that data leakage into the test set was avoided during subsequent processing steps.

**Query Generation.**  
To increase the efficiency of our query generation scheme and to limit API calls, we generate at most 3 questions per image. From all the documents collected, we randomly sample 10,000 images per theme and call Claude-3 Sonnet with the following prompt:

```
You are an assistant specialized in Multimodal RAG tasks.
The task is the following: given an image from a pdf page, you will have to
generate questions that can be asked by a user to retrieve information from
a large documentary corpus. The question should be relevant to the page, and should not be too specific
or too general. The question should be about the subject of the page, and
the answer needs to be found in the page.
Remember that the question is asked by a user to get some information from a
large documentary corpus that contains multimodal data. Generate a question
that could be asked by a user without knowing the existence and the content
of the corpus.
Generate as well the answer to the question, which should be found in the
page. And the format of the answer should be a list of words answering the
question.
Generate at most THREE pairs of questions and answers per page in a
dictionary with the following format, answer ONLY this dictionary
NOTHING ELSE:
{
    "questions": [
        {
            "question": "XXXXXX",
            "answer": ["YYYYYY"]
        },
        {
            "question": "XXXXXX",
            "answer": ["YYYYYY"]
        },
        {
            "question": "XXXXXX",
            "answer": ["YYYYYY"]
        },
    ]
}
where XXXXXX is the question and [’YYYYYY’] is the corresponding list of answers
that could be as long as needed.
Note: If there are no questions to ask about the page, return an empty list.
Focus on making relevant questions concerning the page.
Here is the page:
```

**Human Validation.**  
We manually validate every single synthetically created query in ViDoRe to ensure quality, query relevance, and consistency with the benchmark objective of evaluating retrieval in practical industrial settings. During this step, we randomly assign document-pair queries to 4 volunteer annotators and instruct them to filter out queries that do not fit the above-listed criteria. We also instruct annotators to flag any documents they deem to contain PII information or content not suited for an academic benchmark. No flag was raised during the entirety of the process, validating our prior PDF collection strategy. 100 queries per topic are collected in this manner. Annotators are colleagues and collaborators of the authors who volunteered to help. Each annotator spent approximately 3 hours filtering the larger query set down to 100 high-quality queries per topic.

## Appendix B Implementation details

### B.1 Codebase

The codebase is written in PyTorch and leverages HuggingFace tooling for model implementations and trainers.

### B.2 Pairwise CE loss

Our in-batch contrastive loss ℒ is defined as the softmaxed cross-entropy of the positive scores sk+ = LI(dk,qk) w.r.t. the maximal negative scores sk− = max_{l,l≠k} LI(qk,pl).

For numerical stability, we reformulate the loss with the softplus function, leading to:

|     |     |     |     |
| --- | --- | --- | --- |
|  | ℒ = (1/b)∑k=1^b softplus(sk− − sk+) |  | (2) |

### B.3 Hyperparameters

Hyperparameters are tuned on a validation split composed of 2% of the training dataset. We find bi-encoder methods to be more sensible to learning rate variations than late interaction-based models and achieve the best performance for all models with a learning rate of 5e−5. We experiment with LoRA rank and α values and do not notice particular improvements past r=α=32. Per-device batch sizes are kept small due to long sequence lengths that complicate scaling past b=4. Simulating larger batch sizes for in-batch negative sampling should enable even better results. We find the best results with global batch size b=32 for 1 epoch on our training set.

### B.4 Embedding size

Minimizing storage footprint can be essential to industrial retrieval systems if databases contain millions of documents. With this criterion in view, we have compared the embedding sizes of the models in our study. As shown in Table 3, ColPali’s embedding size is an order of magnitude larger than BM25 and two orders of magnitude larger than BGE-M3. However, this study is limited to the naive method of storing ColPali’s multi-vector embeddings. In practical scenarios, using cluster centroids can reduce the size of ColPali multi-vector embeddings by up to an order of magnitude (Santhanam et al., 2022) and make it a competitive retrieval system.

| Model                | Embedding size (KB) |
|----------------------|---------------------|
| BGE-M3               | 8.60                |
| BM25 (dense emb.)    | 3.00                |
| BM25 (sparse emb.)   | 1.56 ± 0.51         |
| ColPali (float16)    | 256                 |

Table 3: Comparison of the embedding sizes for the DocVQA test set from ViDoRe w.r.t. different retrieval models. The lower the size the smaller the storage footprint of the model. The mean ± std size is given for the sparse embeddings.

### B.5 Latency computations

All latency computations are done on a NVIDIA L4 GPU. Queries are encoded independently (batch size of 1) to simulate online querying, and pages are encoded with a batch size of 4 for PaliGemma derived models, and 8 for BGE-M3. Reported times include image and text processing time before the model forward pass, as well as query-to-index matching times. We note an interesting feature of ColPali is that all documents have the same sequence length, leading to prior knowledge of runtime and memory consumptions. Query latency experiments are averaged over 1000 queries, and indexing times are measured for a 100 page document. Per page time is obtained by diving total time by 100, corresponding to inverse page throughput.

### B.6 Captioning

Examples of captions generated for visually rich document chunks with Claude-3 Sonnet are shown in Figure 6 and Figure 5. The prompt used for generating the description is the following:

```
You are an assistant specialized in document analysis. Given a table or a figure, you have to provide a detailed summary of the content in maximum 3000 characters. Your summary should be qualitative and not quantitative. Here is the table/figure to analyze: {image}. Answer ONLY with the caption of the table/figure.
```

https://arxiv.org/html/2407.01449v1/x3.png  
Figure 5: Example from the "Energy" test set.

*Caption*: The image depicts the hourly energy generation profile, illustrating the contributions of various energy sources over 24 hours. The data is presented as a stacked bar chart, with the x-axis representing the hours of the day from 1 to 2, and the y-axis showing the average hourly generation in MW. The bars are segmented into different colors, each representing a distinct energy source: nuclear, bio, geothermal, solar, wind, hydro, natural gas, and other imports. The chart provides insights into the temporal variations in energy generation across different sources, highlighting the interplay between baseload and intermittent sources throughout the day.

https://arxiv.org/html/2407.01449v1/x4.png  
Figure 6: Example from the "Government Reports" test set.

*Caption*: The image shows a table titled "System of Record" which outlines the different types of documents or records maintained across various systems or departments within an organization related to project management and construction. The rows list documents like project plans, budgets, schedules, contracts, purchase orders, invoices, change requests, bid submissions, drawings, manuals, meeting minutes, and reports. The columns indicate the system or department responsible for maintaining each record, such as County Servers, Project View, OnBase, CGI Advantage Financial System, and Purchasing Department. The table uses "W" and "T" markers to denote which system or department serves as the primary source (writer) or storage location (trailer) for each type of document.

## Appendix C More similarity maps

In Figure 7, ColPali assigns a high similarity to all patches with the word "Kazakhstan" when given the token <_Kazakhstan>. Moreover, our model seems to exhibit world knowledge capabilities as the patch around the word "Kashagan"—an offshore oil field in Kazakhstan—also shows a high similarity score. On the other hand, in Figure 8, we observe that ColPali is also capable of complex image understanding. Not only are the patches containing the word "formulations" highly similar to the query token _formula, but so is the upper-left molecule structure.

https://arxiv.org/html/2407.01449v1/extracted/5696231/images/similarity_maps/similarity_map_kazakhstan.png  
Figure 7: Similarity of the image patches w.r.t. the underlined token in the user query. This example is from the Shift test set.

https://arxiv.org/html/2407.01449v1/extracted/5696231/images/similarity_maps/similarity_map_ferroelectrics.png  
Figure 8: Similarity of the image patches w.r.t. the underlined token in the user query. This example is from the Healthcare Industry test set.

It is also interesting to highlight that both similarity maps showcase a few white patches with high similarity scores. This behavior might first seem surprising as the white patches should not carry a meaningful signal from the original images. We believe the vectors associated with these patches share a similar role with the ViT registers (Darcet et al., 2023), i.e. these patches were repurposed for internal computations and stored the global information from the whole image.

## Appendix E ViDoRe examples

**Energy**

Query: What types of accounts or products allow investors to defer paying taxes?
https://arxiv.org/html/2407.01449v1/extracted/5696231/images/dataset_samples/energy_1.jpeg

Query: What is the projected peak electricity demand in California for the year 2030?
https://arxiv.org/html/2407.01449v1/extracted/5696231/images/dataset_samples/energy_2.jpeg

Query: What is the estimated total savings for a PV system in Durham under the net metering (flat rate) billing option over the system’s useful life of 25 years?
https://arxiv.org/html/2407.01449v1/extracted/5696231/images/dataset_samples/energy_3.jpeg

---

**Artificial Intelligence**

Query: What are some common outcome areas targeted by TAII for different age groups?
https://arxiv.org/html/2407.01449v1/extracted/5696231/images/dataset_samples/ai_1.jpeg

Query: What did the robot monitor to determine when to activate or deactivate the blower motor and blinker?
https://arxiv.org/html/2407.01449v1/extracted/5696231/images/dataset_samples/ai_2.jpeg

Query: What is the key approach used in the PDP architecture?
https://arxiv.org/html/2407.01449v1/extracted/5696231/images/dataset_samples/ai_3.jpeg

---

**Healthcare Industry**

Query: What is the chemical formula for the ferroelectric material Lead Zirconium Titanate (PZT)?
https://arxiv.org/html/2407.01449v1/extracted/5696231/images/dataset_samples/healthcare_1.jpeg

Query: What government entities are involved in public financing for healthcare in the US?
https://arxiv.org/html/2407.01449v1/extracted/5696231/images/dataset_samples/healthcare_2.jpeg

Query: What does the AVPU scale stand for in assessing the level of consciousness of a seriously ill child?
https://arxiv.org/html/2407.01449v1/extracted/5696231/images/dataset_samples/healthcare_3.jpeg

---

**Government Reports**

Query: What are some mandates for the EPA under the Pollution Prevention Act?
https://arxiv.org/html/2407.01449v1/extracted/5696231/images/dataset_samples/gov_1.jpeg

Query: What is the strategy of KPMG Hazem Hassan?
https://arxiv.org/html/2407.01449v1/extracted/5696231/images/dataset_samples/gov_2.jpeg

Query: What is the trust signal score for the consumer industry best-in-class archetype?
https://arxiv.org/html/2407.01449v1/extracted/5696231/images/dataset_samples/gov_3.jpeg

---

**Shift**

Query: Selon le graphique, quelle est la capacité d’import et la consommation réelle de carburants SAF (biocarburants durables pour l’aviation) prévues en 2050 ?
https://arxiv.org/html/2407.01449v1/extracted/5696231/images/dataset_samples/shift_1.jpeg

Query: Quelle partie de la production pétrolière du Kazakhstan provient de champs en mer ?
https://arxiv.org/html/2407.01449v1/extracted/5696231/images/dataset_samples/shift_2.jpeg

Query: Quels sont les pays ayant la plus grande part des découvertes cumulées de pétrole brut en 2020 (en milliers de barils, hors découvertes cumulées) ?
https://arxiv.org/html/2407.01449v1/extracted/5696231/images/dataset_samples/shift_3.jpeg

</details>

<details>
<summary>Image understanding</summary>

# Image understanding

You can add images to Gemini requests to perform tasks that involve
understanding the contents of the included images. This page shows you how to add
images to your requests to Gemini in Vertex AI by using the
Google Cloud console and the Vertex AI API.

## Supported models

The following table lists the models that support image understanding:

| **Model** | **Media details** | **MIME types** |
| --- | --- | --- |
| [Gemini 2.5 Flash-Lite](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash-lite) | - Maximum images per prompt:<br>   <br>   3,000<br>   <br>   <br>- Maximum image size:<br>   <br>   7 MB<br>   <br>   <br>- Maximum number of output images per prompt:<br>   <br>   10 | - `image/png`<br>- `image/jpeg`<br>- `image/webp` |
| [Gemini 2.0 Flash with image generation](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-0-flash) | - Maximum images per prompt:<br>   <br>   3,000<br>   <br>   <br>- Maximum image size:<br>   <br>   7 MB<br>   <br>   <br>- Maximum number of output images per prompt:<br>   <br>   10<br>   <br>   <br>- Maximum tokens per minute (TPM) per project:<br>   <br>  - High/Medium/Default media resolution:<br>     <br>    - US/Asia: 40 M<br>    - EU: 10 M<br>  - Low media resolution:<br>     <br>    - US/Asia: 10 M<br>    - EU: 3 M | - `image/png`<br>- `image/jpeg`<br>- `image/webp` |
| [Gemini 2.5 Pro](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro) | - Maximum images per prompt:<br>   <br>   3,000<br>   <br>   <br>- Maximum image size:<br>   <br>   7 MB | - `image/png`<br>- `image/jpeg`<br>- `image/webp` |
| [Gemini 2.5 Flash](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash) | - Maximum images per prompt:<br>   <br>   3,000<br>   <br>   <br>- Maximum image size:<br>   <br>   7 MB | - `image/png`<br>- `image/jpeg`<br>- `image/webp` |
| [Gemini 2.0 Flash](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-0-flash) | - Maximum images per prompt:<br>   <br>   3,000<br>   <br>   <br>- Maximum image size:<br>   <br>   7 MB<br>   <br>   <br>- Maximum tokens per minute (TPM) per project:<br>   <br>  - High/Medium/Default media resolution:<br>     <br>    - US/Asia: 40 M<br>    - EU: 10 M<br>  - Low media resolution:<br>     <br>    - US/Asia: 10 M<br>    - EU: 2.6 M | - `image/png`<br>- `image/jpeg`<br>- `image/webp` |
| [Gemini 2.0 Flash-Lite](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-0-flash-lite) | - Maximum images per prompt:<br>   <br>   3,000<br>   <br>   <br>- Maximum image size:<br>   <br>   7 MB<br>   <br>   <br>- Maximum tokens per minute (TPM):<br>   <br>  - High/Medium/Default media resolution:<br>     <br>    - US/Asia: 6.7 M<br>    - EU: 2.6 M<br>  - Low media resolution:<br>     <br>    - US/Asia: 2.6 M<br>    - EU: 2.6 M | - `image/png`<br>- `image/jpeg`<br>- `image/webp` |

The quota metric is
`generate_content_video_input_per_base_model_id_and_resolution`.

For a list of languages supported by Gemini models, see model information
[Google models](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models). To learn
more about how to design multimodal prompts, see
[Design multimodal prompts](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/design-multimodal-prompts).
If you're looking for a way to use Gemini directly from your mobile and
web apps, see the
[Firebase AI Logic client SDKs](https://firebase.google.com/docs/ai-logic) for
Swift, Android, Web, Flutter, and Unity apps.

## Add images to a request

You can add a single image or multiple images in your request to Gemini.

### Single image

The sample code in each of the following tabs shows a different way to identify
what's in an image. This sample works with all Gemini multimodal models.

To send a multimodal prompt by using the Google Cloud console, do the
following:

01. In the Vertex AI section of the Google Cloud console, go to
     the **Vertex AI Studio** page.

02. Click **Open freeform**.

03. Optional: Configure the model and parameters:

    - **Model**: Select a model.
    - **Region**: Select the region that you want to use.
    - **Temperature**: Use the slider or textbox to enter a value for
       temperature.

      The temperature is used for sampling during response generation, which occurs when `topP`
      and `topK` are applied. Temperature controls the degree of randomness in token selection.
      Lower temperatures are good for prompts that require a less open-ended or creative response, while
      higher temperatures can lead to more diverse or creative results. A temperature of `0`
      means that the highest probability tokens are always selected. In this case, responses for a given
      prompt are mostly deterministic, but a small amount of variation is still possible.

      If the model returns a response that's too generic, too short, or the model gives a fallback
      response, try increasing the temperature.

    - **Output token limit**: Use the slider or textbox to enter a value for
       the max output limit.

      Maximum number of tokens that can be generated in the response. A token is
      approximately four characters. 100 tokens correspond to roughly 60-80 words.

      Specify a lower value for shorter responses and a higher value for potentially longer
      responses.

    - **Add stop sequence**: Optional. Enter a stop sequence, which is a
       series of characters that includes spaces. If the model encounters a
       stop sequence, the response generation stops. The stop sequence isn't
       included in the response, and you can add up to five stop sequences.

04. Optional: To configure advanced parameters, click **Advanced** and
     configure as follows:

    - **Top-K**: Use the slider or textbox to enter a value for top-K.
      (not supported for Gemini 1.5).
    - **Top-P**: Use the slider or textbox to enter a value for top-P.
    - **Max responses**: Use the slider or textbox to enter a value for the number of responses to generate.
    - **Streaming responses**: Enable to print responses as they're generated.
    - **Safety filter threshold**: Select the threshold of how likely you are to see responses that could be harmful.
    - **Enable Grounding**: Grounding isn't supported for multimodal prompts.

05. Click **Insert Media**, and select a source for your file.

    - [Upload](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding#upload)
    - [By URL](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding#by-url)
    - [Cloud Storage](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding#cloud-storage)
    - [Google Drive](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding#google-drive)

    Select the file that you want to upload and click **Open**.
    Enter the URL of the file that you want to use and click **Insert**.
    Select the bucket and then the file from the bucket that you want to import and click **Select**.

    1. Choose an account and give consent to Vertex AI Studio to access your account the first time you select this option. You can upload multiple files that have a total size of up to 10 MB. A single file can't exceed 7 MB.
    2. Click the file that you want to add.
    3. Click **Select**.

    The file thumbnail displays in the **Prompt** pane. The total number of tokens also displays. If your prompt data exceeds the [token limit](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-models), the tokens are truncated and aren't included in processing your data.

06. Enter your text prompt in the **Prompt** pane.

07. Optional: To view the **Token ID to text** and **Token IDs**, click the **tokens count** in the **Prompt** pane.

08. Click **Submit**.

09. Optional: To save your prompt to **My prompts**, click save\_alt **Save**.

10. Optional: To get the Python code or a curl command for your prompt, click code **Get code**.


#### Install

```
pip install --upgrade google-genai
```

To learn more, see the
[SDK reference documentation](https://googleapis.github.io/python-genai/).

Set environment variables to use the Gen AI SDK with Vertex AI:

```
# Replace the `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` values
# with appropriate values for your project.
export GOOGLE_CLOUD_PROJECT=GOOGLE_CLOUD_PROJECT
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=True
```

```
from google import genai
from google.genai.types import HttpOptions, Part

client = genai.Client(http_options=HttpOptions(api_version="v1"))
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[\
        "What is shown in this image?",\
        Part.from_uri(\
            file_uri="gs://cloud-samples-data/generative-ai/image/scones.jpg",\
            mime_type="image/jpeg",\
        ),\
    ],
)
print(response.text)
# Example response:
# The image shows a flat lay of blueberry scones arranged on parchment paper. There are ...
```

---

After you set up your environment, you can use REST to test a text prompt. The following sample sends a request to the publisher model endpoint.

You can include images that are stored in Cloud Storage or use
base64-encoded image data.

Before using any of the request data, make the following replacements:

- `PROJECT_ID`: Your [project ID](https://cloud.google.com/resource-manager/docs/creating-managing-projects#identifiers).
- `FILE_URI`:
  - **Cloud Storage bucket URI:** The object must either be publicly readable or reside in the same Google Cloud project that's sending the request.
  - **HTTP URL:** The file URL must be publicly readable. You can specify one video file, one audio file, and up to 10 image files per request.
  - **YouTube video URL:** The YouTube video must be either owned by the account that you used to sign in to the Google Cloud console or is public. Only one YouTube video URL is supported per request.

When specifying a `fileURI`, you must also specify the media type (`mimeType`) of the file.

If you don't have an image file in Cloud Storage, then you can use the following publicly available file:
`gs://cloud-samples-data/generative-ai/image/scones.jpg` with a mime type of
`image/jpeg`. To view this image,
[open the sample image](https://storage.googleapis.com/cloud-samples-data/generative-ai/image/scones.jpg)
file.

- `MIME_TYPE`: The media type of the file specified in the `data` or `fileUri` fields. Acceptable values include:  
    - `application/pdf`
    - `audio/mpeg`
    - `audio/mp3`
    - `audio/wav`
    - `image/png`
    - `image/jpeg`
    - `image/webp`
    - `text/plain`
    - `video/mov`
    - `video/mpeg`
    - `video/mp4`
    - `video/mpg`
    - `video/avi`
    - `video/wmv`
    - `video/mpegps`
    - `video/flv`

- `TEXT`: The text instructions to include in the prompt.
  For example, `What is shown in this image?`

To send your request:

Save the request body in a file named `request.json`:

```
cat > request.json << 'EOF'
{
  "contents": {
    "role": "USER",
    "parts": [\
      {\
        "fileData": {\
          "fileUri": "FILE_URI",\
          "mimeType": "MIME_TYPE"\
        }\
      },\
      {\
        "text": "TEXT"\
      }\
    ]
  }
}
EOF
```

Then execute the following command to send your REST request:

```
curl -X POST \
     -H "Authorization: Bearer $(gcloud auth print-access-token)" \
     -H "Content-Type: application/json; charset=utf-8" \
     -d @request.json \
     "https://aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/global/publishers/google/models/gemini-1.5-flash:generateContent"
```

**Response**

```
{
  "candidates": [\
    {\
      "content": {\
        "role": "model",\
        "parts": [\
          {\
            "text": " The image shows a table with a cup of coffee, a bowl of blueberries, and a plate of scones with blueberries on it. There are also pink flowers on the table."\
          }\
        ]\
      },\
      "finishReason": "STOP",\
      "safetyRatings": [ ... ]
    }\
  ],
  "usageMetadata": {
    "promptTokenCount": 265,
    "candidatesTokenCount": 35,
    "totalTokenCount": 300
  }
}
```

For requests with base64-encoded images, replace:

- `B64_BASE_IMAGE`: The [base64 encoding](https://cloud.google.com/vertex-ai/generative-ai/docs/image/base64-encode) of the image, PDF, or video
  to include inline in the prompt.

Save the request body in a file named `request.json`:

```
cat > request.json << 'EOF'
{
"contents": {
    "role": "USER",
    "parts": [\
      {\
        "inlineData": {\
          "data": "B64_BASE_IMAGE",\
          "mimeType": "MIME_TYPE"\
        }\
      },\
      {\
        "text": "TEXT"\
      }\
    ]
}
}
EOF
```

**Response**

```
{
"candidates": [ ... ],
"usageMetadata": {
    "promptTokenCount": 265,
    "candidatesTokenCount": 35,
    "totalTokenCount": 300
}
}
```

Note:
- Use the [`generateContent`](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.publishers.models/generateContent) method to request the response after it's fully generated.
- The multimodal model ID is located at the end of the URL before the method (for example, `gemini-2.0-flash`). This sample might support other models as well.

---

### Multiple images

Each of the following show a different way to include multiple images in a prompt request. Each sample takes in two sets of the following inputs:

- An image of a popular city landmark
- The media type of the image
- Text indicating the city and landmark in the image

The sample also takes in a third image and media type, but no text. The sample
returns a text response indicating the city and landmark in the third image.

These image samples work with all Gemini multimodal models.

To send a multimodal prompt by using the Google Cloud console (similar to above), upload multiple files or select multiple images.

#### Python SDK example

```
from google import genai
from google.genai.types import HttpOptions, Part

client = genai.Client(http_options=HttpOptions(api_version="v1"))

# Read content from GCS
gcs_file_img_path = "gs://cloud-samples-data/generative-ai/image/scones.jpg"

# Read content from a local file
with open("test_data/latte.jpg", "rb") as f:
    local_file_img_bytes = f.read()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[\
        "Generate a list of all the objects contained in both images.",\
        Part.from_uri(file_uri=gcs_file_img_path, mime_type="image/jpeg"),\
        Part.from_bytes(data=local_file_img_bytes, mime_type="image/jpeg"),\
    ],
)
print(response.text)
# Example response:
# Okay, here's the list of objects present in both images:
# ...
```

#### REST API example

Make the following replacements:

- `PROJECT_ID`: Your [project ID](https://cloud.google.com/resource-manager/docs/creating-managing-projects#identifiers).
- `FILE_URI1`, `FILE_URI2`, `FILE_URI3`: The URI or URL of the files to include in the prompt.
- `MIME_TYPE`: The media type of all the files. For simplicity, this sample uses the same media type for all three images.
- `TEXT1`, `TEXT2`: Text instructions for the images.

Save the request body in a file named `request.json`:

```
cat > request.json << 'EOF'
{
"contents": {
    "role": "USER",
    "parts": [\
      {\
        "fileData": {\
          "fileUri": "FILE_URI1",\
          "mimeType": "MIME_TYPE"\
        }\
      },\
      {\
        "text": "TEXT1"\
      },\
      {\
        "fileData": {\
          "fileUri": "FILE_URI2",\
          "mimeType": "MIME_TYPE"\
        }\
      },\
      {\
        "text": "TEXT2"\
      },\
      {\
        "fileData": {\
          "fileUri": "FILE_URI3",\
          "mimeType": "MIME_TYPE"\
        }\
      }\
    ]
}
}
EOF
```

Send the REST request as above.

**Response**

```
{
"candidates": [\
    {\
      "content": {\
        "role": "model",\
        "parts": [\
          {\
            "text": "city: Rio de Janeiro, Landmark: Christ the Redeemer statue \n"\
          }\
        ]\
      },\
      "finishReason": "STOP",\
      "safetyRatings": [ ... ]
    }\
],
"usageMetadata": {
    "promptTokenCount": 791,
    "candidatesTokenCount": 14,
    "totalTokenCount": 805
}
}
```

---

## Set optional model parameters

Each model has a set of optional parameters that you can set. For more
information, see [Content generation parameters](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/content-generation-parameters).

## Image tokenization

Here's how tokens are calculated for images:

- If both dimensions of an image are less than or equal to 384 pixels, then 258 tokens are used.
- If one dimension of an image is greater than 384 pixels, then the image is cropped into tiles. Each tile size defaults to the smallest dimension (width or height) divided by 1.5. If necessary, each tile is adjusted so that it's not smaller than 256 pixels and not greater than 768 pixels. Each tile is then resized to 768x768 and uses 258 tokens.

## Best practices

When using images, use the following best practices and information for the best results:

- If you want to detect text in an image, use prompts with a single image to produce better results than prompts with multiple images.
- If your prompt contains a single image, place the image before the text prompt in your request.
- If your prompt contains multiple images, and you want to refer to them later in your prompt or have the model refer to them in the model response, it can help to give each image an index before the image. Use `a` `b` `c` or `image 1` `image 2` `image 3` for your index. Example:

  ```
  image 1
  image 2
  image 3

  Write a blogpost about my day using image 1 and image 2. Then, give me ideas for tomorrow based on image 3.
  ```

- Use images with higher resolution; they yield better results.
- Include a few examples in the prompt.
- Rotate images to their proper orientation before adding them to the prompt.
- Avoid blurry images.

## Limitations

While Gemini multimodal models are powerful in many multimodal use cases, it's important to understand the limitations of the models:

- **Content moderation**: The models refuse to provide answers on images that violate safety policies.
- **Spatial reasoning**: The models aren't precise at locating text or objects in images. They might only return the approximated counts of objects.
- **Medical uses**: The models aren't suitable for interpreting medical images (for example, x-rays and CT scans) or providing medical advice.
- **People recognition**: The models aren't meant to be used to identify people who aren't celebrities in images.
- **Accuracy**: The models might hallucinate or make mistakes when interpreting low-quality, rotated, or extremely low-resolution images. The models might also hallucinate when interpreting handwritten text in images documents.

</details>

<details>
<summary>jina-embeddings-v4</summary>

# jina-embeddings-v4

Universal embedding model for multimodal and multilingual retrieval

## Overview

Jina Embeddings V4 is a 3.8 billion parameter multimodal embedding model that provides unified text and image representation capabilities. Built on the Qwen2.5-VL-3B-Instruct backbone, the model features an architecture that supports both single-vector and multi-vector embeddings in the late interaction style, addressing limitations found in traditional CLIP-style dual-encoder models. The model incorporates three specialized task-specific LoRA adapters (60M parameters each) that optimize performance across different retrieval scenarios including asymmetric query-document retrieval, semantic text similarity, and code search without modifying the frozen backbone weights. The model demonstrates strong performance in processing visually rich content such as tables, charts, diagrams, screenshots, and mixed-media formats through a unified processing pathway that reduces the modality gap present in conventional architectures. Supporting multilingual capabilities, the model can handle input texts up to 32,768 tokens with images resized to 20 megapixels, making it suitable for various document retrieval and cross-modal search applications across different languages and domains.

## Methods

Jina Embeddings V4 implements a unified multimodal language model architecture that differs from CLIP-style dual-encoder approaches. The model processes inputs through a shared pathway where images are first converted to token sequences via a vision encoder, then both text and image modalities are processed together by the language model decoder with contextual attention layers. This architecture supports two output modes to accommodate different use cases: single-vector embeddings that produce 2048-dimensional vectors truncatable down to 128 dimensions through Matryoshka Representation Learning, generated via mean pooling for efficient similarity search; and multi-vector embeddings that output 128 dimensions per token via projection layers for late interaction style retrieval. The model includes three task-specific LoRA adapters that provide specialized optimization: the retrieval adapter uses prefix-based asymmetric encoding with hard negatives training for query-document scenarios, the text-matching adapter employs CoSENT loss for semantic similarity tasks, and the code adapter focuses on natural language-to-code retrieval applications. Training occurs in two phases: initial pair training using contrastive InfoNCE loss with both text-text and text-image pairs from over 300 sources, followed by task-specific fine-tuning of the three LoRA adapters using triplet-based methods and specialized loss functions tailored to each domain's requirements.

## Performance

Jina Embeddings V4 achieves competitive performance across multiple benchmark categories. On visual document retrieval, it scores 72.19 average on the JinaVDR benchmark compared to 64.50 for ColPali-v1.2, and 84.11 average on ViDoRe compared to 83.90 for ColPali, with the multi-vector mode reaching 90.17 on ViDoRe. For cross-modal retrieval, the model scores 84.11 on CLIP Benchmark, compared to jina-clip-v2 (81.12) and nllb-clip-large-siglip (83.19). In text retrieval tasks, it achieves 55.97 on MTEB-en and 66.49 on MMTEB, with notable performance in long document processing at 67.11 on LongEmbed compared to 55.66 for its predecessor. The model demonstrates solid semantic text similarity performance with 85.89 on English STS tasks and 72.70 on multilingual STS benchmarks. Code retrieval capabilities reach 71.59 on CoIR benchmark, though specialized models like voyage-code-3 (77.33) achieve higher scores in this domain. The model shows improved cross-modal alignment with a score of 0.71 compared to 0.15 for OpenAI CLIP, addressing the modality gap issue in multimodal models. Multi-vector mode consistently outperforms single-vector mode on visually rich tasks, while single-vector mode provides efficient performance for standard retrieval scenarios.

## Best Practice

To effectively utilize Jina Embeddings V4, select the appropriate LoRA adapter based on your specific application requirements. Use the 'retrieval' adapter for asymmetric query-document retrieval scenarios where queries and documents have different structures, ensuring proper prefixes are applied to distinguish between query and passage content. The 'text-matching' adapter is suitable for semantic similarity tasks and symmetric retrieval where the goal is to find similar content rather than answers to queries, making it appropriate for document clustering, duplicate detection, and content recommendation systems. For programming-related applications, the 'code' adapter is optimized for natural language-to-code retrieval, code-to-code similarity search, and technical question answering scenarios. Choose output modes based on your performance and efficiency requirements: single-vector embeddings offer efficient similarity search and are suitable for storage-constrained environments, with truncatable dimensions allowing reduction from 2048 to 128-512 dimensions with acceptable quality trade-offs, while multi-vector embeddings provide higher precision for complex retrieval tasks, particularly when working with visually rich documents where late interaction scoring captures detailed relationships. The model's unified architecture allows processing of mixed text-image inputs without requiring separate encoders or OCR preprocessing for visual documents. The model's cross-modal alignment capabilities and multilingual support make it suitable for international applications. For production deployments, consider the 60M parameter overhead per LoRA adapter when planning memory requirements, noting that all three adapters can be maintained simultaneously with less than 2% additional memory footprint, enabling flexible task switching during inference.

</details>

<details>
<summary>Lost in OCR Translation? Vision-Based Approaches to Robust Document Retrieval</summary>

# Lost in OCR Translation? Vision-Based Approaches to Robust Document Retrieval

###### Abstract.

Retrieval-Augmented Generation (RAG) has become a popular technique for enhancing the reliability and utility of Large Language Models (LLMs) by grounding responses in external documents. Traditional RAG systems rely on Optical Character Recognition (OCR) to first process scanned documents into text. However, even state-of-the-art OCRs can introduce errors, especially in degraded or complex documents. Recent vision-language approaches, such as ColPali, propose direct visual embedding of documents, eliminating the need for OCR. This study presents a systematic comparison between a vision-based RAG system (ColPali) and more traditional OCR-based pipelines utilizing Llama 3.2 (90B) and Nougat OCR across varying document qualities. Beyond conventional retrieval accuracy metrics, we introduce a semantic answer evaluation benchmark to assess end-to-end question-answering performance. Our findings indicate that while vision-based RAG performs well on documents it has been fine-tuned on, OCR-based RAG is better able to generalize to unseen documents of varying quality.  
We highlight the key trade-offs between computational efficiency and semantic accuracy, offering practical guidance for RAG practitioners in selecting between OCR-dependent and vision-based document retrieval systems in production environments.

## 1. Introduction

Large Language Models (LLMs) have shown improvements across the landscape of natural language processing, enabling promising performance increases across a multitude of tasks such as semantic analysis, question answering, machine translation, and text descriptions. However, despite their strengths, these models suffer from fundamental limitations due to their reliance on static training corpora, often leading to hallucinations or outdated information. Retrieval-Augmented Generation (RAG) addresses this limitation by allowing LLMs to retrieve and cite information from external sources during inference (Lewis et al., [2020](https://arxiv.org/html/2505.05666v1#bib.bib12)). In a typical RAG pipeline, user queries are transformed and stored as vector embeddings. The most relevant documents are then passed into the LLM’s context window along with the original query to guide generation. Leveraging this approach, RAG has been leveraged to boost code-translation quality (Bhattarai et al., [2024](https://arxiv.org/html/2505.05666v1#bib.bib5)) (Bhattarai et al., [2025b](https://arxiv.org/html/2505.05666v1#bib.bib6)) and has likewise been shown to curb hallucinations (Bhattarai et al., [2025a](https://arxiv.org/html/2505.05666v1#bib.bib4)).

In addition to improving response accuracy, the retrieval component of RAG substantially facilitates the application of LLMs to proprietary and evolving document collections. Unlike traditional fine-tuning methods—which are costly, time-consuming, and must be repeated whenever new documents are introduced—RAG enables real-time integration of the latest information through on-demand retrieval. Consequently, RAG has been widely adopted in commercial production settings, powering proprietary systems like ChatGPT, Claude, Perplexity, and Gemini, although detailed technical implementations remain largely unpublished in peer-reviewed venues.

This work specifically addresses the application of RAG to scanned or digitized documents. Traditionally, handling such documents involves first extracting text using OCR algorithms prior to embedding and retrieval. However, OCR preprocessing frequently introduces significant noise, especially in scenarios involving low-quality images, handwritten annotations, or complex layouts. Recent VLMs, such as ColPali (Faysse et al., [2024](https://arxiv.org/html/2505.05666v1#bib.bib7)), offer a promising alternative by directly embedding document images into unified multimodal vector spaces without OCR. These vision-based methods inherently preserve critical spatial relationships, formatting cues, and visual nuances that OCR processes often overlook or misinterpret.

Despite the potential advantages of direct image embedding, critical gaps persist in the current literature. Existing benchmarks for document retrieval systems, such as ViDoRe (Faysse et al., [2024](https://arxiv.org/html/2505.05666v1#bib.bib7)), predominantly evaluate performance using clean, high-quality documents, conditions rarely representative of practical real-world scenarios. Additionally, prior comparative evaluations—including those involving ColPali—have often relied on outdated OCR baselines like Tesseract (Smith, [2007](https://arxiv.org/html/2505.05666v1#bib.bib16)) and have limited their analyses primarily to retrieval accuracy, neglecting comprehensive semantic question-answering capabilities. To date, the field lacks rigorous comparative assessments of OCR-to-RAG versus VLM-to-RAG pipelines across varied document quality conditions, particularly emphasizing downstream semantic accuracy and practical robustness.

Moreover, existing vision-language models suitable for direct embedding (e.g., ColPali, ColQwen, ColSmol) typically employ relatively lightweight architectures with parameter counts ranging between 2–7 billion, whereas current OCR-to-RAG pipelines can leverage significantly larger language models. Specifically, in this study, our OCR-based pipeline integrates Llama 3.2 (90 billion parameters), potentially affording substantial advantages in downstream semantic quality.

While one could theoretically adapt a 90B parameter model for the VLM role in ColPali, implementing this architecture presents significant practical challenges related to performance and memory usage. ColPali relies on generating and storing multiple dense vector embeddings per document (representing image patches), and using a VLM of 90B magnitude would dramatically increase the computational time needed to generate these numerous embeddings. Additionally, the memory and disk space required to store these multi-vector representations would likely become prohibitive for substantial document collections. The subsequent question embedding before the retrieval step would also face substantial slowdowns, potentially rendering the RAG system unusable for interactive applications. Although an operation might tolerate the costs associated with document embedding, the degraded query response times would likely make the system unacceptable to end users.

Our empirical results show that even when using the same query encoder (Qwen2 7B), the VLM-based pipeline exhibits slower query times than the OCR-based system. This slowdown stems from the need to compare the query embedding against many patch-level image embeddings per document during late-interaction retrieval, making latency a practical bottleneck even with mid-sized models.

Given the increase in retrieval speed inherent in scaling the VLM component within this specific dense retrieval architecture, this study focuses on comparing representative, readily deployable model sizes characteristic of each paradigm (lightweight VLM vs. OCR + large LLM). This allows for an evaluation of the trade-offs presented by systems configurations commonly encountered in practice.

This study rigorously compares a VLM-based RAG pipeline (ColPali) against a competitive OCR-based pipeline utilizing Llama 3.2 OCR. We systematically evaluate their performance across multiple document quality levels, explicitly incorporating visual degradation and complexity reflective of real-world conditions. Furthermore, we introduce a comprehensive semantic answer evaluation benchmark that extends beyond traditional retrieval metrics, enabling a deeper understanding of each pipeline’s practical strengths and limitations.

We initially anticipated that the integrated visual-textual embedding of VLM-based approaches would offer superior robustness to visual noise, layout variations, and OCR-induced inaccuracies, potentially resulting in higher reliability for practical applications involving visually imperfect documents. However, our empirical findings indicate that VLM-based approaches struggle to generalize effectively to unseen data that they have not been explicitly fine-tuned on.

### 1.1. Contributions

This study advances the state-of-the-art (SOTA) through three core contributions:

1. We present the first systematic empirical comparison between two leading paradigms for document retrieval augmentation: VLM-based RAG system, specifically ColQwen2 (7 billion parameters), and an advanced OCR-based RAG pipeline leveraging Llama 3.2 (90 billion parameters). To the best of our knowledge, previous evaluations have not explicitly addressed the substantial model-size disparity between lightweight VLM architectures commonly used in visual retrieval (e.g., ColQwen2) and significantly larger VLM for OCR-task.

2. We introduce and execute a rigorous experimental protocol that systematically evaluates both systems across precisely controlled levels of document degradation. By manually categorizing the documents based on their noise levels, we explicitly characterize each system’s robustness and performance under conditions found in real-world deployment scenarios.

3. We propose and validate a novel semantic evaluation benchmark explicitly designed for assessing end-to-end performance in knowledge-intensive document question-answering tasks. Unlike conventional evaluations focused solely on retrieval accuracy, our benchmark incorporates automated semantic metrics including Exact Match, BLEU, ROUGE-1, and ROUGE-L.

## 2. Related Work

This section gives an overview of various document retrieval works related to this research, highlighting the limitations and strengths of differing approaches across both OCR and VLM retrieval methods.

### 2.1. Vision-Language Models for Document Understanding

Early multimodal approaches to document understanding model both text and layout information. LayoutLM introduced unified transformer encoding textual tokens alongside spatial layout positions, greatly improving tasks such as form understanding and receipt parsing (Xu et al., [2020](https://arxiv.org/html/2505.05666v1#bib.bib20)). Further advancements incorporated visual features explicitly, such as DocFormer, which integrates text, layout, and visual embeddings within a transformer architecture (Appalaraju et al., [2021](https://arxiv.org/html/2505.05666v1#bib.bib2)). More recently, LayoutLMv3 extended this methodology by pre-training both text and image modalities with a unified masking strategy, achieving state-of-the-art results across text-heavy and image-intensive document tasks (Huang et al., [2022](https://arxiv.org/html/2505.05666v1#bib.bib9)). Another strong advancement, UDOP (Unifying Document Processing), introduced a generative vision-text-layout transformer model capable of different document AI tasks through a prompting-based framework, achieving state-of-the-art results across multiple document benchmarks (Tang et al., [2023](https://arxiv.org/html/2505.05666v1#bib.bib17)). While powerful, these methods still depend on the extracted textual inputs from OCR, supplemented with visual information and layout context.

### 2.2. OCR Integration with Retrieval-Augmented Generation

RAG systems rely on the quality of input documents for effective retrieval. Liu et al. (2023) showed that increases in OCR noise significantly reduces retrieval accuracy in various tested RAG applications. Zhang et al. (2024) introduced OHRBench, a benchmark for evaluating how various levels of OCR-induced noise affect RAG pipelines (Zhang et al., [2024](https://arxiv.org/html/2505.05666v1#bib.bib23)). In that work, Zhang et al. found that even state-of-the-art OCR systems often fail to construct high-quality knowledge bases for retrieval when the original documents contain what they call semantic and formatting noise.
Piryani et al. (2025) developed MultiOCR-QA, a multilingual dataset designed to test LLM performance on OCR-processed texts, in which the authors found that character-level OCR errors significantly reduce QA accuracy (Piryani et al., [2025](https://arxiv.org/html/2505.05666v1#bib.bib14)).

Transformer-based OCR models like Nougat and Llama 3.2 have improved the OCR bottleneck, but their effectiveness under real-world noise conditions is still limited. Understanding the interaction between OCR fidelity and retrieval outcomes is important for evaluating when traditional OCR pipelines are sufficient compared to newer approaches such as VLM-based retrieval (e.g., ColPali) that claim higher levels of robustness. This work situates itself within this space by directly comparing text embedding based RAG pipelines, driven by OCR, versus vision embedding RAG systems under varying document quality controls.

### 2.3. Vision-Language Models for Document Retrieval

VLMs offer an alternative to OCR methods by embedding entire document images, which includes text, layout, and visual context into a shared representation space. This approach bypasses the traditional OCR step, which could potentially reduce issues arising from text extraction error.
CLIP (Radford et al., [2021](https://arxiv.org/html/2505.05666v1#bib.bib15)) was one of the first large-scale VLMs to align visual and textual embeddings using contrastive learning. SigLIP (Zhai et al., [2023](https://arxiv.org/html/2505.05666v1#bib.bib22)), modified CLIP by introducing a sigmoid-based loss function that allows for more stable training and better scalability across noisy or individual text-image pairs. These models laid the groundwork for more document-specific architectures such as PaliGemma (Beyer et al., [2024](https://arxiv.org/html/2505.05666v1#bib.bib3)), which combines SigLIP’s image encoder with Gemma’s decoder to handle token-level document retrieval tasks.

ColBERT (Contextualized Late Interaction - BERT), expands upon BERT by adding a late interaction mechanism. The main difference from BERT is that this mechanism delays the interaction between query and document until the final scoring stage (Khattab and Zaharia, [2020](https://arxiv.org/html/2505.05666v1#bib.bib10)). Similarly to BERT, ColBERT also stores embeddings at the word-token level for both query and document, instead of a single embedding at the document level, and then afterward scores similarity between each combination of word-document embedding. This in turn allows for a more in-depth similarity calculation between query and document compared to its single-embedding per document counterpart, providing higher retrieval accuracy. ColPali (Faysse et al., [2024](https://arxiv.org/html/2505.05666v1#bib.bib7)) additionally builds on this approach by combining the PaliGemma model with ColBERT-style late interaction, to match visual document embeddings with token-level query embeddings.

### 2.4. ColPali and the ViDoRe Benchmark

ColPali is a RAG design that uses a VLM to bypass the traditional OCR-to-text pipeline by directly embedding image patches across the entire document (Faysse et al., [2024](https://arxiv.org/html/2505.05666v1#bib.bib7)). By utilizing the late interaction mechanism seen in ColBERT, alongside PaliGemma’s ability to generate multi-vector embeddings for document images (patches) that capture both the textual and visual data inside each document, ColPali is able to efficiently match queries with relevant documents. Overall, ColPali aims to increase the accuracy of document retrieval, while also reducing training time by embedding text and visual chunks simultaneously in a single, end-to-end method.  
To evaluate performance, the authors of ColPali proposed the ViDoRe (Visual Document Retrieval) benchmark (Faysse et al., [2024](https://arxiv.org/html/2505.05666v1#bib.bib7)). This dataset includes 127,346 high-quality PDF pages that generally all have well-structured layouts and legible content. Models are scored using metrics like NDCG@5, which tests the ability to retrieve the correct page given a specific query. While ColPali demonstrates strong performance on this benchmark, the controlled nature of the datasets —high resolution, consistent formatting—limits its generalizability. Additionally, in its OCR-to-text RAG baseline, ColPali uses Tesseract, which is not considered state-of-the-art.  
Our work builds upon ViDoRe by testing whether ColPali’s retrieval advantages persist under more realistic and visually degraded conditions while also adding a more competitive OCR method (Llama 3.2 90B). This work also introduces a semantic evaluation step that compares the retrieved content’s answer to a ground truth answer, allowing the ability to move beyond exact page-match metrics and assess downstream utility in RAG pipelines.

### 2.5. Comparative Benchmarks for Vision-Language Models

Recent studies have introduced benchmarks to evaluate the strengths and weaknesses of VLMs across various document understanding tasks. HELM (Holistic Evaluation of Language Models) is a framework developed to provide a standardized, multidimensional evaluation of language models (Liang et al., [2022](https://arxiv.org/html/2505.05666v1#bib.bib13)). VHELM extends this framework to the vision-language setting (Lee et al., [2024](https://arxiv.org/html/2505.05666v1#bib.bib11)). LVLM-eHub provides a large-scale benchmark for evaluating instruction-tuned VLMs on multimodal tasks, to obtain a better understanding of generalization and issues with overfitting (Xu et al., [2023](https://arxiv.org/html/2505.05666v1#bib.bib19)).  
Other evaluations additionally explore architectural comparisons and reliability. For example, Mamba-based models have shown better performance compared to traditional transformer-based VLMs in vision-language tasks (Waleffe et al., [2024](https://arxiv.org/html/2505.05666v1#bib.bib18)). DeCC introduces a task decomposition framework to evaluate answer consistency and the robustness of VLMs (Yang et al., [2024](https://arxiv.org/html/2505.05666v1#bib.bib21)). Overall, these benchmarks highlight both the promise and limitations of VLMs in document-level reasoning, which is one of the primary focuses of this work on image-based retrieval.  
In addition, this work complements these efforts by evaluating VLM-based retrieval approaches, specifically in the context of noisy, real-world documents and directly comparing them against OCR-based RAG pipelines.

https://arxiv.org/html/2505.05666v1/extracted/6423616/advancedai.png  
Figure 1. Overview of the experimental pipeline comparing VLM-based and OCR-based RAG systems.

## 3. Methodology

In this section, we present the detailed methods and formal mathematical descriptions for the RAG pipelines compared in our study. We outline the VLM-based pipeline, the OCR-based pipeline, and the rigorous evaluation framework used to assess both retrieval and semantic accuracy. The overview of the proposed framework is shown in Figure 1.

### 3.1. System Architectures

We rigorously evaluate two distinct RAG system paradigms: a VLM-based approach that directly embeds images, and an OCR-based pipeline that extracts text before embedding.

#### 3.1.1. VLM-Based RAG Pipeline

Our vision-based RAG pipeline follows the ColPali architecture proposed by Faysse et al. (2024), specifically designed to bypass OCR preprocessing. The core architecture utilizes the PaliGemma model, integrating SigLIP-So400m as the image encoder and Gemma-2B as the textual encoder-decoder. Formally, a given document image \(D_i\) is partitioned into \(m\) non-overlapping patches \(\{p_{i,1},p_{i,2},…,p_{i,m}\}\). Each patch \(p_{i,j}\) is encoded by the VLM encoder \(f_{\text{VLM}}\) into embedding vectors:

\[
\mathbf{e}_{i,j}^{(\text{VLM})}=f_{\text{VLM}}(p_{i,j};\theta_{\text{VLM}}),\quad\mathbf{e}_{i,j}^{(\text{VLM})}\in\mathbb{R}^{d},
\]

where \(d\) denotes embedding dimensionality and \(\theta_{\text{VLM}}\) the model parameters. The resulting multi-vector embedding representation is:

\[
E_{i}^{(\text{VLM})}=\{\mathbf{e}_{i,1}^{(\text{VLM})},\dots,\mathbf{e}_{i,m}^{(\text{VLM})}\}.
\]

At retrieval time, queries \(q\) are encoded using a textual embedding model \(g\) parameterized by \(\phi\):

\[
\mathbf{q}=g(q;\phi),\quad\mathbf{q}\in\mathbb{R}^{d}.
\]

Retrieval scoring employs ColBERT-style late-interaction, computing similarity scores between query and document patch embeddings:

\[
s_{i}^{(\text{VLM})}(q,D_{i})=\max_{j\in[1,m]}\frac{\mathbf{q}^{\top}\mathbf{e}_{i,j}^{(\text{VLM})}}
{\|\mathbf{q}\|\|\mathbf{e}_{i,j}^{(\text{VLM})}\|}.
\]

At the time of our experiments, the SOTA model from the ColPali family was ColQwen2 (7B), which we employed for the experiments presented here.

#### 3.1.2. OCR-Based RAG Pipeline

Our OCR-based pipeline leverages advanced OCR extraction combined with semantic-rich text embeddings. Document images \(D_i\) first undergo OCR via an OCR model \(h_{\text{OCR}}\):

\[
T_{i}=h_{\text{OCR}}(D_{i}),
\]

where \(T_i\) is the extracted textual content. This text is subsequently embedded using the Qwen embedding model \(g(\cdot;\phi)\) to produce a single, dense semantic embedding:

\[
\mathbf{e}_{i}^{(\text{OCR})}=g(T_{i};\phi),\quad\mathbf{e}_{i}^{(\text{OCR})}\in\mathbb{R}^{d}.
\]

Documents are stored in a vector database based on these embeddings. Query embedding follows the same procedure, ensuring consistent semantic representation. Retrieval scores for OCR embeddings use cosine similarity:

\[
s_{i}^{(\text{OCR})}(q,D_{i})=\frac{\mathbf{q}^{\top}\mathbf{e}_{i}^{(\text{OCR})}}{\|\mathbf{q}\|\|\mathbf{e}_{i}^{(\text{OCR})}\|}.
\]

This pipeline’s accuracy directly depends on OCR quality, making it potentially sensitive to visual degradation and complex layouts.

### 3.2. DocDeg Dataset

We introduce DocDeg, a curated dataset specifically designed to evaluate Retrieval-Augmented Generation (RAG) systems under realistic visual degradation conditions. DocDeg comprises 4,196 diverse documents obtained from the U.S. Department of Energy’s Office of Scientific and Technical Information (OSTI) and includes a wide variety of document types, such as academic papers, technical reports, memos, presentation slides, and handwritten notes, spanning both scanned and digitally generated quality based on OSTI source ^[https://www.osti.gov/].^ The scanned documents often exhibit real-world noise, artifacts, and quality loss often encountered in practical RAG deployments.

Each document page was manually labeled by two expert annotators into four distinct degradation categories reflecting visual clarity: Level 0 (native digital documents with no degradation), Level 1 (high-quality scans with minimal visual noise or artifacts), Level 2 (imperfect scans exhibiting moderate blur, faint text, or minor artifacts), and Level 3 (severely degraded pages featuring significant distortion, extensive noise, and considerable content loss).

#### 3.2.1. Feature Annotation

To enable a more detailed understanding of retrieval challenges, each document in the DocDeg dataset was manually annotated among a range of 12 structural and content features. Initially, 5,800 document pages were labeled by two annotators. After filtering out approximately 1,600 documents that lacked sufficient textual or visual information (e.g., blank pages, cover sheets, low-content slides), the final DocDeg dataset consists of 4,196 information-rich documents used in our experiments. Our analysis focuses on the degree of visual distortion or degradation. We judged quantified degradation on a scale of 0 to 3, with the following categories:

- **Level 0 (Digital Copy):** Native digital document with no visible degradation.
- **Level 1 (Perfect Scan):** High-quality scanned document with minimal or no noise/artifacts.
- **Level 2 (Imperfect Scan):** Noticeable visual defects such as mild blurring, faded or missing text, or scanning artifacts.
- **Level 3 (Severely Degraded):** Document is almost illegible, with heavy noise, distortion, or severe content loss.

A zoomed in example of each degradation level is provided in Figure 2.

#### 3.2.2. Q&A pair generation

We constructed a rigorous evaluation benchmark by generating ten unique, nuanced question-answer pairs per document using Llama 3.3 (70B). We leveraged Sambanova Systems’ API calls to deploy the LLaMA models for Q&A metadata synthesis and retrieval evaluation. These pairs were carefully designed to capture specific factual details from each document, explicitly avoiding direct verbatim copying from the source text. Leveraging these generated question-answer pairs, we performed two distinct evaluations of RAG system performance: (1) Document Retrieval Assessment, where each generated question was treated as a query to assess the system’s ability to correctly retrieve the corresponding source document, and (2) Semantic Answer Generation Evaluation, where we evaluated whether the RAG-augmented language model could accurately produce the reference answer provided in the generated pairs when conditioned on retrieved document context. This dual-evaluation setup enabled a comprehensive assessment of both retrieval precision and downstream semantic accuracy of the RAG systems under investigation.

### 3.3. Evaluation Framework

To rigorously assess both retrieval effectiveness and downstream semantic accuracy, we employ a dual evaluation framework consisting of standard retrieval metrics and a novel semantic answer evaluation benchmark.

#### 3.3.1. Retrieval Evaluation Metrics

https://arxiv.org/html/2505.05666v1/
https://arxiv.org/html/2505.05666v1/
https://arxiv.org/html/2505.05666v1/
https://arxiv.org/html/2505.05666v1/

Figure 2. Zoomed-in view of documents with different degradation levels

We utilize well-established retrieval metrics to quantitatively measure retrieval performance:

- **Mean Reciprocal Rank (MRR)** evaluates retrieval quality by averaging the reciprocal rank positions of correct documents:

  \[
  \text{MRR} = \frac{1}{\|Q\|} \sum_{q \in Q} \frac{1}{r_q}
  \]

- **Recall@k** measures the proportion of relevant documents retrieved among the top-k results:

  \[
  \text{Recall@k} = \frac{\{ \text{relevant documents in top~} k \}}{\{ \text{relevant documents total} \}}
  \]

- **Normalized Discounted Cumulative Gain (NDCG@k)** quantifies retrieval ranking quality by considering the position and relevance of documents retrieved:

  \[
  \text{NDCG@k} = \frac{ \sum_{i=1}^{k} \frac{2^{rel_i}-1}{\log_2(i+1)}}{ \sum_{i=1}^{k} \frac{2^{rel_i^{ideal}}-1}{\log_2(i+1)} }
  \]

The degradation labels were used to segment the dataset for all retrieval evaluations reported in this paper. Figure 2 provides a zoomed in example for each of the four human evaluated text degradation levels.

#### 3.3.2. Semantic Answer Evaluation

In addition to retrieval metrics, we evaluate semantic accuracy using a robust end-to-end evaluation framework involving automated metrics for semantic comparison. We first create reference answers by generating ten unique question-answer pairs per document using Llama 3.3 (70B) (Grattafiori et al., [2024](https://arxiv.org/html/2505.05666v1#bib.bib8)). These pairs are designed to emphasize nuanced, factual details and avoid verbatim copying from source documents.

Semantic accuracy is evaluated using the following metrics:

- **Exact Match (EM)** measures the strict accuracy by comparing generated answers \(a_i\) to references \(r_i\):

  \[
  \text{EM} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}(a_i=r_i)
  \]

- **BLEU Score** evaluates lexical precision via n-gram overlap:

  \[
  \text{BLEU}(a,r) = BP \cdot \exp\left( \sum_{n=1}^{4} w_{n} \log p_{n}(a, r) \right)
  \]

- **ROUGE Scores (ROUGE-1, ROUGE-L)** measure lexical completeness and structural coherence. ROUGE-1 considers unigram overlap:

  \[
  \text{ROUGE-1} = \frac{ \| a \cap r \| }{ \| r \| }
  \]

  while ROUGE-L uses the longest common subsequence (LCS) overlap:

  \[
  \text{ROUGE-L} = \frac{ (1+\beta^2) \text{LCS}(a, r) }{ \| a \| + \beta^2 \| r \| }, \quad \text{typically } \beta = 1
  \]

### 3.4. Retrieval Performance

RAG practitioners typically deploy systems "out of the box", relying on pre-trained models without additional fine-tuning on their specific document collections due to practical constraints such as cost, time, expertise and data availability. With that in mind, our primary evaluation focuses on how each pipeline performs under realistic, visually degraded conditions using pre-trained models without performing task-specific fine-tuning on the DocDeg dataset for either the VLM-based or the OCR-based retrieval components. This approach allows us to assess the generalizability and robustness of these architectures when faced with unfamiliar document types and degradation patterns, which is crucial for understanding their practical utility.

We report retrieval metrics—Mean Reciprocal Rank (MRR), Recall@5, and Normalized Discounted Cumulative Gain at rank 5 (NDCG@5)—across four levels of increasing document degradation on this dataset.

In addition to DocDeg, we also conducted supplementary experiments on the ViDoRe benchmark (specifically, the DocVQA subset). This evaluation compared three approaches:

- **ColQwen (baseline):** ColPali retrieval with their SOTA model, ColQwen2-v1, fine-tuned on the ViDoRe dataset.
- **OCR+Qwen2 (Nougat):** OCR extraction by rerunning the ViDoRe images through the Nougat OCR model, followed by Qwen2 7B Instruct, again without fine-tuning.
- **OCR+Qwen2 (Llama):** OCR extraction by rerunning the ViDoRe images through the Llama3.2 90B OCR model, followed by Qwen2 7B Instruct, again without fine-tuning.

## 4. Results and Discussion

In this section, the findings from our experiments are presented, an analysis of performance differences is given, and the resulting implications are discussed in the context of document understanding and retrieval systems.

https://arxiv.org/html/2505.05666v1/extracted/6423616/individual_metrics_comparison.png  
Figure 3. Retrieval performance across document quality levels

### 4.1. Retrieval Performance Across Document Quality Levels

To evaluate the robustness and accuracy of each pipeline across realistic conditions, we systematically assessed their retrieval performance over four incremental document degradation levels, ranging from pristine (Level 0) to severely degraded (Level 3). We report Mean Reciprocal Rank (MRR), Recall@5, and Normalized Discounted Cumulative Gain at rank 5 (NDCG@5).

Table 1 compares a vision-language model (VLM)-based pipeline against two OCR-based pipelines, one using nougat OCR and another using Llama 3.2 for OCR.

**Table 1. Retrieval performance metrics across document degradation levels.**

VLM-Based = Vision-Language Model pipeline,  
OCR-Based (Nougat) = OCR pipeline with Nougat OCR,  
OCR-Based (Llama) = OCR pipeline with Llama 3.2 OCR.

| DistortionLevel | Questions | Metric | VLM-Based | OCR-Based(Nougat) | OCR-Based(Llama) |
| --- | --- | --- | --- | --- | --- |
| Level 0 | 11190 | MRR | 0.2971 | 0.2327 | 0.5151 |
|  |  | Recall@5 | 0.4057 | 0.3013 | 0.6415 |
|  |  | NDCG@5 | 0.3242 | 0.2721 | 0.5468 |
| Level 1 | 19154 | MRR | 0.2440 | 0.3218 | 0.4857 |
|  |  | Recall@5 | 0.3573 | 0.4298 | 0.6525 |
|  |  | NDCG@5 | 0.2722 | 0.3627 | 0.5275 |
| Level 2 | 6864 | MRR | 0.2012 | 0.3119 | 0.4579 |
|  |  | Recall@5 | 0.3003 | 0.4093 | 0.6104 |
|  |  | NDCG@5 | 0.2259 | 0.3502 | 0.4961 |
| Level 3 | 4748 | MRR | 0.2098 | 0.2295 | 0.4520 |
|  |  | Recall@5 | 0.3112 | 0.3010 | 0.5925 |
|  |  | NDCG@5 | 0.2350 | 0.2612 | 0.4872 |
| Total | 41956 | MRR | 0.2471 | 0.2858 | 0.4852 |
|  |  | Recall@5 | 0.3554 | 0.3774 | 0.6359 |
|  |  | NDCG@5 | 0.2740 | 0.3212 | 0.5229 |

**Table 2. Retrieval performance metrics across document degradation levels (slides removed).**

| DistortionLevel | Questions | Metric | VLM-Based | OCR-Based(Nougat) | OCR-Based(Llama) |
| --- | --- | --- | --- | --- | --- |
| Level 0 | 3784 | MRR | 0.1905 | 0.2819 | 0.3769 |
|  |  | Recall@5 | 0.2826 | 0.3979 | 0.5322 |
|  |  | NDCG@5 | 0.2134 | 0.3109 | 0.4157 |
| Level 1 | 16974 | MRR | 0.2347 | 0.3416 | 0.4782 |
|  |  | Recall@5 | 0.3476 | 0.4593 | 0.6502 |
|  |  | NDCG@5 | 0.2628 | 0.3711 | 0.5213 |
| Level 2 | 6794 | MRR | 0.2005 | 0.3139 | 0.4557 |
|  |  | Recall@5 | 0.2997 | 0.4122 | 0.6086 |
|  |  | NDCG@5 | 0.2252 | 0.3386 | 0.4941 |
| Level 3 | 4748 | MRR | 0.2098 | 0.2295 | 0.4520 |
|  |  | Recall@5 | 0.3112 | 0.3010 | 0.5925 |
|  |  | NDCG@5 | 0.2997 | 0.2474 | 0.4872 |
| Total | 32300 | MRR | 0.2186 | 0.3123 | 0.4578 |
|  |  | Recall@5 | 0.3244 | 0.4189 | 0.6192 |
|  |  | NDCG@5 | 0.2449 | 0.3390 | 0.4982 |

As illustrated in Table 1 and Figure 3, the Llama OCR-Based pipeline consistently outperformed the VLM-Based pipeline on retrieval accuracy across all degradation levels evaluated on the DocDeg dataset for all metrics. Both systems show declines in accuracy as document quality decreases, though the OCR-Based scores still significantly outperformed the VLM-Based ones even under severe degradation.

Comparing Nougat to Llama further highlights that the specific OCR engine used within the OCR-Based pipeline has a substantial impact on overall retrieval performance. The Llama 3.2 OCR engine consistently achieved higher retrieval metrics than Nougat across all distortion levels. This underscores the critical role that OCR quality plays in downstream retrieval tasks, particularly when deploying OCR-to-RAG systems in realistic, noisy document environments.

Nougat outperformed the VLM-based RAG pipeline on levels one through three and on the weighted average. Interestingly, Nougat performed worse on the highest quality documents. Table 1 shows retrieval results with slideshows removed from the dataset. Both Llama and ColQwen saw a reduction in retrieval accuracy on level zero documents when slides were excluded. Meanwhile, Nougat’s performance on the slideshow-free data improved.

Although ColPali retrieves the top-5 visually relevant documents based on query-image similarity, it does not natively support answer generation. To use ColPali in a full RAG pipeline, an additional step is required: either (1) apply OCR to the retrieved document images and pass the extracted text into a language model, or (2) use a vision-language model capable of question answering over images. Both options introduce additional computational cost, either in the form of OCR latency or VLM inference time. For this study’s semantic answer evaluation, we used the OCR text from the images retrieved by ColPali as context for the LLM, in order to enable a fair comparison across pipelines.

### 4.2. Semantic Answer Accuracy

In addition to retrieval accuracy, we provide a comprehensive evaluation of the pipelines’ end-to-end performance in question-answering scenarios using Exact Match, BLEU, ROUGE-1, and ROUGE-L (Table 3). In addition to automated metrics, a small-scale manual review of approximately 40 generated question-answer pairs was conducted for qualitative verification. While not exhaustive, this review confirmed the quality and relevance of the question-answer pairs to the documents.

**Table 3. Semantic answer quality across document degradation levels.**

VLM-Based = ColQwen2,  
OCR-Based = Llama 3.2 90B for OCR + Qwen2.

| Distortion Level | Questions | Metric | VLM-Based | OCR-Based |
| --- | --- | --- | --- | --- |
| Level 0 | 11190 | Exact Match | 0.0046 | 0.0080 |
|  |  | Average BLEU | 0.0589 | 0.0753 |
|  |  | Average ROUGE-1 | 0.2088 | 0.2547 |
|  |  | Average ROUGE-L | 0.1912 | 0.2370 |
| Level 1 | 19154 | Exact Match | 0.0038 | 0.0066 |
|  |  | Average BLEU | 0.0628 | 0.0804 |
|  |  | Average ROUGE-1 | 0.2167 | 0.2643 |
|  |  | Average ROUGE-L | 0.1960 | 0.2430 |
| Level 2 | 6864 | Exact Match | 0.0033 | 0.0057 |
|  |  | Average BLEU | 0.0587 | 0.0751 |
|  |  | Average ROUGE-1 | 0.1975 | 0.2410 |
|  |  | Average ROUGE-L | 0.1785 | 0.2213 |
| Level 3 | 4748 | Exact Match | 0.0038 | 0.0066 |
|  |  | Average BLEU | 0.0595 | 0.0762 |
|  |  | Average ROUGE-1 | 0.2044 | 0.2493 |
|  |  | Average ROUGE-L | 0.1826 | 0.2263 |
| Total | 41956 | Exact Match | 0.0039 | 0.0068 |
|  |  | Average BLEU | 0.0607 | 0.0777 |
|  |  | Average ROUGE-1 | 0.2100 | 0.2556 |
|  |  | Average ROUGE-L | 0.1903 | 0.2355 |

Results from Table 3 show the semantic answer evaluation. Consistent with the results obtained from retrieval, the OCR-Based pipeline obtained higher scores when compared to the VLM-Based pipeline on every tested metric and degradation level. A notable difference between these results and the retrieval results, is that there does not seem to be an obvious decrease in score as degradation level increases. Overall, the results suggest that the potential benefits of the VLM’s ability to handle visual nuance did not translate to improved semantic performance when compared to the superior results from the OCR-Based pipeline.

### 4.3. Computational Efficiency Analysis

A comparative analysis of indexing time, retrieval latency, and memory consumption for both pipelines is summarized in Table 4. The results reveal that the VLM-based pipeline is more efficient in both embedding generation and retrieval latency compared to the OCR-based pipeline. The VLM-based system achieved lower embedding times (0.4739s per document compared to 0.5503s) and substantially reduced retrieval latency (0.0010s per query vs. 0.0311s per query). Additionally, memory consumption per 1,000 documents was notably lower for the VLM-based pipeline (1.38 GB) relative to the OCR-based pipeline (9.5 GB). This substantial memory efficiency can be primarily attributed to the avoidance of storing extensive OCR-derived textual embeddings.

All experiments not involving Llama-based models were conducted on a node equipped with two NVIDIA A100 GPUs (40GB VRAM each), an AMD EPYC 7502 CPU, and 251 GB of system memory. However, the SentenceTransformer model was executed on a single A100 throughout. The second GPU remained unused. For OCR extraction and question-answer pair generation involving Llama 3.2 and Llama 3.3 models, we utilized a SambaNova SN40L platform with integrated RDU accelerators. Retrieval latency, embedding times, and memory usage metrics reported in Table 4 reflect performance measured on these respective systems. Embedding of text and images was done with a batch size of 1.

**Table 4. Computational Efficiency Comparison**

| Metric | VLM-Based | OCR-Based |
| --- | --- | --- |
| OCR Time (per document) | N/A | 6s |
| Embedding Time (per document) | .4739s | .5503s |
| Retrieval Latency (per query) | 0.04252s | .0311s |
| Memory Usage (per 1k documents) | 1.38 GB | 9.5 GB |

https://arxiv.org/html/2505.05666v1/extracted/6423616/updated_rag_capabilities_radar.png  
Figure 4. Comparative Analysis of RAG System Capabilities

### 4.4. Overall assessment on DocDeg Dataset

We leverage a RADAR plot to showcase overall performance of VLM vs OCR RAG shown in Figure 4. Each axis is a linearly rescaled version of a concrete, published metric—chosen so that a score of 10 corresponds to the best value observed across all pipelines and degradation levels, and a score of 0 would correspond to a value at least two standard deviations worse than the worst observed.

Specifically:

1. **Clean–document retrieval** = \( 10 \times \frac{\text{MRR}_{d=0} - \mu_{\text{MRR}}}{\sigma_{\text{MRR}}} \), clipped to \[0,10\]
2. **Noisy–document retrieval** = \( 10 \times \frac{\text{MRR}_{d=3} - \mu_{\text{MRR}}}{\sigma_{\text{MRR}}} \)
3. **Semantic answer quality** = \( 10 \times \frac{\text{ROUGE--L}_\text{total} - \mu_{R}}{\sigma_{R}} \), where \(\mu_R, \sigma_R\) are the mean and s.d. of ROUGE-L across all runs.
4. **Processing speed** = \( 10 \times \frac{\min(T_\text{ret})}{T_\text{ret}} \), where \(T_\text{ret}\) is per-query latency; lower is better.
5. **Memory efficiency** = \( 10 \times \frac{\min(M)}{M} \), where \(M\) is memory per 1k documents.

All raw values (MRR, ROUGE-L, latency, memory) are reported in Tables 1 and 4. The resulting normalised scores for VLM, OCR+Nougat, and OCR+Llama are plotted without additional smoothing. This transparent mapping ensures that the radar visualisation is a faithful, monotonic transformation of the quantitative results.

The radar plot provides a comprehensive visualization of the relative strengths and weaknesses between VLM-based and OCR-based RAG systems across five critical dimensions. The VLM-based approach demonstrates exceptional performance in processing speed (9.5/10) and memory efficiency (9.0/10), reflecting its significantly faster retrieval latency (0.001s vs 0.031s) and lower memory footprint (1.38GB vs 9.5GB per 1k documents). However, it shows limitations in both clean document retrieval (6.5/10) and noisy document retrieval (6.0/10). In contrast, the OCR-based pipeline with Llama 3.2 exhibits superior performance in clean document retrieval (9.0/10), noisy document retrieval (8.5/10), and semantic answer quality (8.5/10), at the cost of greater computational demands. The Nougat OCR-based system offers a middle ground across most metrics. These findings reveal a clear trade-off: while modern OCR-based pipelines leveraging advanced models like Llama 3.2 provide better retrieval accuracy and semantic quality in realistic document settings, VLM-based approaches offer significant advantages in computational efficiency and resource utilization.

### 4.5. ViDoRe Evaluation Results

To further evaluate retrieval performance under cleaner document conditions, we conducted supplementary experiments on the DocVQA subset of the ViDoRe benchmark. We compared three retrieval pipelines: (1) the ColQwen2 model, representing the ViDoRe authors’ best fine-tuned vision-language system based on ColPali; (2) an OCR-based pipeline using Alibaba-NLP/gte-Qwen2-7B-instruct with Nougat OCR outputs; and (3) the same OCR-based pipeline but preprocessing the images into text with the more advanced Llama 3.2 90B vision model.

**Table 5. ViDoRe DocVQA Retrieval Comparison**

| Metric(Top-5) | ColQwen2 | OCR+Qwen2(Nougat) | OCR+Qwen2(Llama) |
| --- | --- | --- | --- |
| NDCG@5 | 0.6027 | 0.3373 | 0.3373 |
| MAP@5 | 0.5813 | 0.3147 | 0.3147 |
| MRR@5 | 0.5851 | 0.3164 | 0.3164 |
| Recall@5 | 0.6674 | 0.4058 | 0.4058 |

ColQwen2, which was fine-tuned on DocVQA, achieves the strongest retrieval performance, with an NDCG@5 of 0.6027 and a Recall@5 of 0.6674.

Although the pages in ViDoRe are noise free, the benchmark is challenging because its queries are only loosely connected to the page text—typically short key phrases rather than full natural-language questions. Off-the-shelf retrieval models, whether vision-based or OCR-based, therefore struggle not with image quality but with recognising this subtle semantic link between a brief phrase and an entire page.

ColQwen2 succeeds because it has been fine-tuned on ViDoRe’s own query–page pairs, so its embeddings have learned to notice exactly those idiosyncratic links. In contrast, our Nougat- and Llama-OCR pipelines rely on a general-purpose text encoder that has never seen ViDoRe queries; they fall behind simply because their embeddings are semantically mis-aligned, not because their OCR output is faulty. Indeed, Nougat and Llama 3.2 produce almost identical retrieval scores, confirming that on such clean PDFs OCR quality is no longer the bottleneck—semantic alignment between the query and the page embeddings is.

## 5. Conclusion and Future Work

Our study rigorously compared OCR-based and VLM-based RAG pipelines, evaluating their performance and robustness across real degraded document scenarios. Our results show that, without task-specific fine-tuning, OCR-based approaches offer improved retrieval and generation performance in all evaluated settings, with some added cost at indexing time but more efficient performance at query time.

While OCR-based pipelines incur some additional latency during document ingestion, we found that they are actually faster at query time compared to VLM-based pipelines like ColPali. ColPali introduces higher query-time latency, even when using the same query encoder as the OCR pipeline. Furthermore, ColPali does not support end-to-end question answering directly: once documents are retrieved as images, either OCR or a vision-language QA model must be applied to generate a final answer, both of which add additional processing overhead.

Scaling to larger and more powerful vision encoders could potentially improve retrieval performance for VLM-based systems, but this would likely further increase both embedding time and query latency. Future research should compare fine-tuned OCR-based RAG pipelines with fine-tuned VLM-based RAG pipelines with a vision-language QA model to determine whether further performance improvements can be achieved.

## 6. Acknowledgements

This manuscript has been approved for unlimited release and has been assigned LA-UR-25-24280. This research was funded by the LANL ASC grant AI4Coding and the LANL Institutional Computing Program, supported by the U.S. DOE NNSA under Contract No. 89233218CNA000001. We also thank Sambanova Systems for providing access to API calls for LLM inference utilized in this work

## References

- Appalaraju et al. (2021) Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota, Yusheng Xie, and R. Manmatha. 2021.  
  **DocFormer: End-to-End Transformer for Document Understanding.** In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_. 973–983.  
  arXiv:2106.11539 \[cs.CV\]

- Beyer et al. (2024) Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, Thomas Unterthiner, Daniel Keysers, Skanda Koppula, Fangyu Liu, Adam Grycner, Alexey Gritsenko, Neil Houlsby, Manoj Kumar, Keran Rong, Julian Eisenschlos, Rishabh Kabra, Matthias Bauer, Matko Bošnjak, Xi Chen, Matthias Minderer, Paul Voigtlaender, Ioana Bica, Ivana Balazevic, Joan Puigcerver,
  Pinelopi Papalampidi, Olivier Henaff, Xi Xiong, Radu Soricut, Jeremiah Harmsen, and Xiaohua Zhai. 2024.  
  **PaliGemma: A versatile 3B VLM for transfer.**  
  arXiv preprint.  
  arXiv:2407.07726 \[cs.CV\]

- Bhattarai et al. (2025a) Manish Bhattarai, Ryan Barron, Maksim E. Eren, Minh N. Vu, Vesselin Grantcharov, Ismael Ismael, Valentin Stanev, Cynthia Matuszek, Vladimir I Valtchinov, Kim Rasmussen, and Boian S. Alexandrov. 2025a.  
  **HEAL: Hierarchical Embedding Alignment Loss for Improved Retrieval and Representation Learning.** In _Proceedings of the 4th International Workshop on Knowledge-Augmented Methods for Natural Language Processing_. Association for Computational Linguistics, Albuquerque, New Mexic, USA, 205–214.  
  [https://aclanthology.org/2025.knowledgenlp-1.19/](https://aclanthology.org/2025.knowledgenlp-1.19/)

- Bhattarai et al. (2024) Manish Bhattarai, Javier E. Santos, Shawn Jones, Ayan Biswas, Boian Alexandrov, and Daniel O’Malley. 2024.  
  **Enhancing Code Translation in Language Models with Few-Shot Learning via Retrieval-Augmented Generation.** In _2024 IEEE High Performance Extreme Computing Conference (HPEC)_. 1–8.  
  [doi:10.1109/HPEC62836.2024.10938485](https://doi.org/10.1109/HPEC62836.2024.10938485)

- Bhattarai et al. (2025b) Manish Bhattarai, Minh N. Vu, Javier E. Santos, Ismael Ismael, and Daniel O’Malley. 2025b.  
  **Enhancing Cross-Language Code Translation via Task-Specific Embedding Alignment in Retrieval-Augmented Generation.** In _Proceedings of the 4th International Workshop on Knowledge-Augmented Methods for Natural Language Processing_. Association for Computational Linguistics, Albuquerque, New Mexic, USA, 107–117.  
  [https://aclanthology.org/2025.knowledgenlp-1.8/](https://aclanthology.org/2025.knowledgenlp-1.8/)

- Faysse et al. (2024) Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. 2024.  
  **ColPali: Efficient Document Retrieval with Vision Language Models.** In _The Twelfth International Conference on Learning Representations (ICLR)_.  
  arXiv:2402.01410 \[cs.CV\]

- Grattafiori et al. (2024) Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, et al. 2024.  
  **The Llama 3 Herd of Models.**  
  arXiv preprint.  
  arXiv:2407.21783 \[cs.AI\]

- Huang et al. (2022) Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. 2022.  
  **LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking.** In _Proceedings of the 30th ACM International Conference on Multimedia (ACM MM)_. 4233–4242.  
  arXiv:2204.08387 \[cs.CV\]

- Khattab and Zaharia (2020) Omar Khattab and Matei Zaharia. 2020.  
  **ColBERT: Efficient Passage Search via Contextualized Late Interaction over BERT.** In _Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval_. 39–48.  
  arXiv:2004.12832 \[cs.IR\]

- Lee et al. (2024) Tony Lee, Haoqin Tu, Chi Heem Wong, Wenhao Zheng, Yiyang Zhou, Yifan Mai, Josselin Somerville Roberts, Michihiro Yasunaga, Huaxiu Yao, Cihang Xie, and Percy Liang. 2024.  
  **VHELM: A Holistic Evaluation of Vision Language Models.**  
  arXiv preprint.  
  arXiv:2410.07112 \[cs.CV\]

- Lewis et al. (2020) Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020.  
  **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.** In _Advances in Neural Information Processing Systems 33 (NeurIPS 2020)_. 9459–9474.  
  arXiv:2005.11401 \[cs.CL\]

- Liang et al. (2022) Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Karan Santhanam, Jared Quincy Davis, Yura Alexandr Perov, Yacine Jernite, et al. 2022.  
  **Holistic Evaluation of Language Models.** In _Advances in Neural Information Processing Systems 35 (NeurIPS 2022) Datasets and Benchmarks Track_.

- Piryani et al. (2025) Bhawna Piryani, Jamshid Mozafari, Abdelrahman Abdallah, Antoine Doucet, and Adam Jatowt. 2025.  
  **MultiOCR-QA: Dataset for Evaluating Robustness of LLMs in Question Answering on Multilingual OCR Texts.**  
  arXiv preprint.  
  arXiv:2502.16781 \[cs.CL\]

- Radford et al. (2021) Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021.  
  **Learning Transferable Visual Models From Natural Language Supervision.** In _Proceedings of the 38th International Conference on Machine Learning (ICML)_. PMLR, 8748–8763.  
  arXiv:2103.00020 \[cs.CV\]

- Smith (2007) Ray Smith. 2007.  
  **An Overview of the Tesseract OCR Engine.** In _ICDAR ’07: Proceedings of the Ninth International Conference on Document Analysis and Recognition_. IEEE Computer Society, Washington, DC, USA, 629–633.  
  [https://storage.googleapis.com/pub-tools-public-publication-data/pdf/33418.pdf](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/33418.pdf)

- Tang et al. (2023) Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, and Mohit Bansal. 2023.  
  **Unifying Vision, Text, and Layout for Universal Document Processing.** In _CVPR_. 19254–19264.

- Waleffe et al. (2024) Roger Waleffe, Wonmin Byeon, Duncan Riach, Brandon Norick, Vijay Korthikanti, Tri Dao, Albert Gu, Ali Hatamizadeh, Sudhakar Singh, Deepak Narayanan, Garvit Kulshreshtha, Vartika Singh, Jared Casper, Jan Kautz, Mohammad Shoeybi, and Bryan Catanzaro. 2024.  
  **An Empirical Study of Mamba-based Language Models.**  
  arXiv preprint.  
  arXiv:2406.07887 \[cs.CL\]

- Xu et al. (2023) Peng Xu, Wenqi Shao, Kaipeng Zhang, Peng Gao, Shuo Liu, Meng Lei, Fanqing Meng, Siyuan Huang, Yu Qiao, and Ping Luo. 2023.  
  **LVLM-eHub: A Comprehensive Evaluation Benchmark for Large Vision-Language Models.**  
  arXiv preprint.  
  arXiv:2306.09265 \[cs.CV\]

- Xu et al. (2020) Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. 2020.  
  **LayoutLM: Pre-training of Text and Layout for Document Image Understanding.** In _Proceedings of ACM SIGKDD_. 1192–1200.

- Yang et al. (2024) Qian Yang, Weixiang Yan, and Aishwarya Agrawal. 2024.  
  **Decompose and Compare Consistency: Measuring VLMs’ Answer Reliability via Task-Decomposition Consistency Comparison.** In _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)_. Association for Computational Linguistics, 3613–3627.  
  arXiv:2404.16003 \[cs.CV\]

- Zhai et al. (2023) Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. 2023.  
  **Sigmoid Loss for Language Image Pre-Training.** In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_. 11941–11952.  
  [doi:10.1109/ICCV51070.2023.01100](https://doi.org/10.1109/ICCV51070.2023.01100)  
  arXiv:2303.15343 \[cs.CV\]

- Zhang et al. (2024) Junyuan Zhang, Qintong Zhang, Bin Wang, Linke Ouyang, Zichen Wen, Ying Li, Ka-Ho Chow, Conghui He, and Wentao Zhang. 2024.  
  **OCR Hinders RAG: Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation.**  
  arXiv preprint.  
  arXiv:2412.02592 \[cs.CL\]

</details>


## Code Sources

<details>
<summary>Repository analysis for https://github.com/towardsai/course-ai-agents/blob/main/lessons/11_multimodal/notebook.ipynb</summary>

# Repository analysis for https://github.com/towardsai/course-ai-agents/blob/main/lessons/11_multimodal/notebook.ipynb

## Summary
Repository: towardsai/course-ai-agents
File: notebook.ipynb
Lines: 935

Estimated tokens: 47.9k

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

This notebook demonstrates how to build multimodal AI systems that can process and understand both text and visual content using Google's Gemini models. You'll learn to work with images and PDFs in various formats, implement multimodal RAG systems for semantic search, and create AI agents capable of visual reasoning.

We will use the `google-genai` library to interact with Google's Gemini models.

**Learning Objectives:**

1. **Process multimodal content**: Learn to handle images and PDFs in different formats (bytes, base64, URLs) with Gemini models
2. **Implement object detection**: Use multimodal LLMs for visual analysis and structured output generation
3. **Build multimodal RAG systems**: Create embeddings for images and text to enable semantic search across visual content
4. **Develop multimodal AI agents**: Construct ReAct agents that can search through and reason about visual information
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

We will use the `gemini-2.5-flash` model, which is fast, cost-effective, and supports advanced features like tool use.
"""

MODEL_ID = "gemini-2.5-flash"

"""
## 2. Applying multimodal LLMs to images, PDFs, and text

There are three core ways we can process images and PDFs with multimodal models:
1. As raw bytes
2. As base64 encoded strings
3. As URLs

First, let's examine a test image:

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

image_bytes = load_image_as_bytes(image_path=Path("images") / "image_1.jpeg", format="WEBP")

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

#     The image depicts a striking juxtaposition of artificial intelligence and natural life, featuring a large, heavily armored robot with glowing red eyes in what appears to be an industrial or workshop setting. Perched playfully on the robot's left forearm and shoulder is a small, adorable grey tabby kitten, looking curiously towards the robot's head. The robot's metallic body is intricately detailed with circuit-like patterns on its head and glowing red indicators on its chest, showcasing a powerful and advanced design, while the soft, fluffy kitten provides a stark and endearing contrast against the machine's robust frame.

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
Using the same approach, we can easily pass multiple images simultaneously:
"""

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

#     The primary difference between the two images lies in the nature of the interaction between the animals and robots, as well as their respective environments. In the first image, a small, grey kitten appears curious and playful as it stands on the arm of a large, grey, somewhat clunky robot, suggesting a peaceful, almost companionable moment set in a well-lit, industrial-like indoor space. Conversely, the second image depicts a tense and confrontational scene where a large, fluffy white dog is aggressively barking at a sleek, black, humanoid robot, with both subjects poised for a fight in a dimly lit, trash-strewn urban alleyway.

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
### 2.2 As base64 encoded strings

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
pretty_print.wrapped(image_base64, title="Image 1 Base64")
# Output:
#   [93m------------------------------------------ Image 1 Base64 ------------------------------------------[0m

#     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

[... Content truncated due to length ...]

</details>


## YouTube Video Transcripts

<details>
<summary>Although AI research is traditionally split into distinct fields like NLP and computer vision, countless real-world problems require solutions that integrate information across these modalities. In this video, I'll discuss how we can solve such problems using multimodal embeddings, then show how to use them to do things like zero-shot image classification and image search. And if you're new here, welcome. I'm Shaw. I make videos about the things I'm learning about and building in AI.</summary>

Although AI research is traditionally split into distinct fields like NLP and computer vision, countless real-world problems require solutions that integrate information across these modalities. In this video, I'll discuss how we can solve such problems using multimodal embeddings, then show how to use them to do things like zero-shot image classification and image search. And if you're new here, welcome. I'm Shaw. I make videos about the things I'm learning about and building in AI.
[00:30]
And if you enjoyed this content, please consider clicking the subscribe button. That's a great no-cost way you can support me in all the videos that I make. Here we're going to talk about multimodal embeddings. Although the discussion here will focus around CLIP, which works with text and image data, this is a much more general idea that can be extended to many other modalities.
[01:00]
Before talking about multimodal embeddings, it's worth answering the question, what are embeddings? The way I'll define embeddings here are useful numerical representations of data learned through model training. A classic example of this is BERT, which is a popular language model before the era of GPT-3 and all the modern large language models. BERT used to be state-of-the-art and one of the things that it does is masked language modeling.
[01:30]
In other words, you can give it a sequence of text where one of the tokens in that sequence is masked, meaning that it's not visible, and BERT will predict the most likely token that goes in the place of that mask. So if you pass in the sequence, listen to your, the most likely token that goes in this sequence is instincts. So it turns out that through learning how to do this prediction, BERT learns useful representations of text, which can be extended to other NLP tasks.
[02:00]
The basic idea here is that you'll take BERT and you'll drop its head, so the classification head which is doing this masked language modeling, and you'll have this mutilated version of BERT, which instead of doing this token prediction, it'll take an input sequence of text and return a numerical representation of it, where each row in this matrix corresponds to each token in this text sequence.
[02:30]
And then each of these columns corresponds to the embedding dimension, the dimension of this internal representation of text that BERT uses in order to do masked language modeling. And we can go one step further and go from token-level representations to sequence-level representations. So we could do something like take the average across all these tokens in the sequence, and we're left with a one by D matrix, which represents the entire sequence.
[03:00]
And of course, to get these embeddings to be a bit more practical, people will often do additional fine tuning on top of these embeddings, but this is the basic idea of where they are coming from. A key point about embeddings is that these aren't just arbitrary numerical representations. They are typically semantically meaningful such that if we were to look at how text were organized in this embedding space, similar concepts would tend to be located close together while dissimilar concepts will tend to be far apart.
[03:30]
For example, the sequence "a cute puppy" might be relatively close to the sequence "a good boy", while that same sequence "a cute puppy" might be relatively far from a sequence like "funny cat meme". However, this isn't limited to just text. We can generate embeddings for any type of data. Another popular type of data we might work with are images.
[04:00]
So if we had some image embeddings, the space might be structured like this, where we tend to have cats in the top left part, the dogs tend to be in the bottom right part, and then we have a goat further away from these. Although text embeddings and image embeddings are super helpful in that they can be adapted and repurposed to solve other either NLP tasks or computer vision tasks, one major limitation here is that any random text embedding space we might be working with and any random image embedding space we might be working in, don't have any relationship to one another.
[04:30]
There's no way out of the box to directly map this text embedding space to this image embedding space and vice versa, even if they are semantically meaningful in of themselves. And that's something we can plainly see here in that the text and image embedding spaces are not aligned because for this text embedding space, the puppies tend to be in the top right, the cats tend to be at the bottom, while in our image embedding space, the cats tend to be up here and the dogs tend to be down here.
[05:00]
But what if there was a way we could merge these two embedding spaces together? That's exactly the key idea behind multimodal embeddings, which are embeddings which align representations of different data modalities. The basic intuition here is that if we had a multimodal embedding space, we could represent text and images in the same vector space. So now, indeed, text like "a cute puppy" will be close to images of cute puppies.
[05:30]
The text "a cute cat" will be close to images of a cute cat, and the same thing will hold for other concepts. However, this idea is not limited to just images and text. We could just as easily embed audio and images together. Maybe this is a audio file that is a cat meowing. This is a goat making goat noises. We have a puppy with like a cute bark. And then maybe we have like a funny shrieking sound associated with this cat meme here. Another application of this is aligning representations of brain signals with images and text.
[06:00]
What this means is if we were to record someone's brain activity and then represent it in this embedding space, we could, in principle, decode the brain information to generate images and text. So, in essence, reading people's thoughts. And actually, in reference number four, they are aiming to do exactly this with large language models. Intuitively, this idea of multimodal embeddings is pretty simple to understand. We have this embedding space which is agnostic to modality.
[06:30]
So it doesn't matter if it's an image of a cat, a text description of a cat, or the brain signals of someone looking at a picture of a cat, these numerical representations will be relatively similar. But how do we create these aligned numerical representations? In other words, how does this work under the hood? So, the key technique behind creating these types of embeddings is contrastive learning, which is an approach that seeks to represent different views of the same underlying information similarly.
[07:00]
And the way this works is that we'll train a model on two things. One, positive pairs of data, and two, negative pairs. So, in the case of aligning image and text representations, positive pairs might be a picture of a cute cat and a textual caption for this image. And then we might have the text and an image of a cute puppy. And then we might have the same thing for a baby goat.
[07:30]
On the other hand, negative captions might look something like this, where you have the image of a cat, but the caption is "a cute puppy", image of a puppy, and the caption is a goat, and you have a goat and the caption is a cat. The intuition here is that we train a model to maximize the similarity between these positive pairs and minimize the similarity between these negative pairs.
[08:00]
That's the key intuition. In the following slides, we're going to go one level deeper and look at the loss function and the math behind how this is accomplished. If you don't care about the math and how this is working under the hood, feel free to skip ahead to the example code. But if you're interested in the math, we're about to jump right into it. The way we can use contrastive learning to align image and text representations is we can take images, generate image embeddings using an image encoder.
[08:30]
So basically, we take our images and generate a single numerical representation for them. And then we can take all these image embeddings, and we can concatenate them into a matrix that I'll call I sub e. So this will be a N by D matrix, where N is the number of images. So if you have three images here, it'll be 1, 2, 3. And then D, so the number of columns will be the embedding dimension. Then we can do a similar thing for text, so we can get a text encoder from off the shelf, we can generate these text embeddings, and then we can concatenate them into a matrix that I'll call T sub e.
[09:00]
and that will have the same shape. So we'll have n captions, and then they'll have some embedding dimension D. Just to point out here that the representations that we would put into these matrices won't directly come from an image encoder and a text encoder. Instead, these will be multiplied by some learnable weight matrix and then normalized before being organized in this matrix.
[09:30]
So, that weight matrix that we multiply the original embeddings by are the learnable parameters. Once we have these matrices I and T, we can construct this logits matrix. Basically, what that means is we're going to take each image in our image embedding matrix, and then each text sequence in our text embedding matrix, and we're going to compute their similarity.
[10:00]
Typically, this is just a cosine similarity, so you do the dot product between these two matrices. And then you'll divide it by a temperature parameter. That's what this tau parameter is representing. And the reason we call them logits is because at some point, it's going to be the argument in an exponential function. And we'll see that in a little bit here. So taking just those three examples from the previous slide, the similarity between the first image and the first text sequence will be in this one-one position of the matrix.
[10:30]
And then the similarity between this cat image and the sequence "a cute puppy" will be represented by this value here. And then the similarity between this cat image and the text sequence "a cute baby goat" will be represented by this value here, and so on and so forth. Just looking at this, we can see that what we want is to make the logits on the diagonal of this matrix as big as possible. So, in other words, we want to maximize the similarity between the positive pairs.
[11:00]
And then we want to minimize the off diagonal elements, which correspond to negative pairs. One way we can do this is via the contrastive loss. So this might take slightly different forms depending on the context or the paper that you're reading, but here I'm going to follow what was done in developing CLIP, which is reference number three here. And so basically, one way we can achieve this goal of maximizing the similarity of these on diagonal elements and minimizing the similarity between these off diagonal elements is via this equation here.
[11:30]
Which is basically saying for the ith image, so let's say this cat image here, we want the numerator to be as big as possible, so the numerator will be the ii element, so this will be either 1,1 or 2,2 or 3,3. And then we want the denominator to be as small as possible. So if the numerator is big, the denominator is small, that means this fraction becomes big, and then if we take the log of that, we'll still have a big number.
[12:00]
And then we want this number to be as big as possible because the goal of training is to minimize the loss. And then if this number is big and we have a minus sign next to it, then this will be as minimal as possible. That was probably a bit abstract, so let's walk through this step by step. Let's look at just the first image first. With this notation, I call the loss associated with the first image L1. This will consist of taking this one-one logit and then summing over all the logits in this first row.
[12:30]
So we're basically taking this image and comparing it to every single caption. Then we do the same thing for the second image. We have the positive pair similarity here, and then we sum over all the logits in this row. And then we do a similar thing for image number three. So we look at the positive pair similarity, and then we sum over all the logits or similarities in this row. We can do this for every single image in our batch or even in our whole training data set.
[13:00]
And then we can aggregate them to get the final contrastive loss. What that will look like is we'll take the loss according to the first image, the loss according to the second image, and the loss corresponding to the third image. And then we can just take their average, and that'll give us the contrastive loss for the images. But we can do the same exact thing for text. This is how I'm notating contrastive loss for the text.
[13:30]
I've switched the index here from J to I, and then I've changed the summation here to I. I feel this notation might be a bit too subtle, but hopefully explaining it step by step, it makes sense what I mean here. So let's see what this looks like for the first text sequence. We're going to be evaluating a cute cat. So we'll look at logits 1,1 here, and then we'll sum over the logits in this first column.
[14:00]
We'll do the same thing for this second text sequence, a cute puppy, and we'll sum over all the logits in this column. And then finally, we do it for the final text sequence. It's important to note here that generally this logits matrix is asymmetric because the similarity between the text "a cute puppy" and this image of a cat is in general different than the similarity between this image of a puppy and the text sequence "a cute cat".
[14:30]
That's an important thing to note here, and that's the reason why we go through this whole procedure for the images and the text sequences separately. And then we can aggregate the loss over all the text examples, just like we did for the images like this. And then we'll get a total text loss by taking the average of all the examples in our minibatch. We can then combine the image loss and text loss together by taking their average. And then we can write it all out to have this big monstrosity all on one page.
[15:00]
But basically, this first term here corresponds to the image loss. This second term here corresponds to the text loss. And this is how we train the weights which translate the raw image and text encodings into our multimodal embedding space. This will give us a training signal which we can use to update the weights of these projection matrices and just keep doing that until we're satisfied with the loss. So, if that was much more math than you were hoping to get out of this video, I apologize for that.
[15:30]
But let's jump to practical applications of multimodal embeddings. Here, I'm going to use CLIP for two different use cases. This first use case is zero-shot image classification. The meaning of that is we're going to do image classification without explicitly training CLIP to distinguish between the different image classes that we're considering.
[16:00]
The first step is to import transformers. Then I'm going to bring in these two things. And then I'm importing this PIL library, which will allow us to work with images in Python. Next, we'll load in the model and the data processor. The image preprocessing is important because images could be any type of size and shape and all that. The CLIP processor is an abstraction that ensures the data are in a suitable format to be passed through the model. Next, we're going to load in our image, so I'm going to load in this image of a cute cat.
[16:30]
So it's the same one we've seen so far. Then I'm going to define the text classes. So this is a really interesting aspect of using CLIP for zero-shot image classification, because before, if you wanted to do image classification, traditionally that was something that was set at model training. It was implicitly coded into the architecture of the model in that you had this classification head and each value in the output layer corresponded to the probability of class one versus class two versus class three, so on and so forth.
[17:00]
But now, when using CLIP, which is trained via contrastive learning, we actually pass these classes as text inputs. So, with our text and image inputs defined, we can pass these through our processor so it's in the appropriate format for CLIP. And then we can just pass it to the model. Then with this one line of code, we'll generate these outputs. Then we can extract the logits per image. Recall the logits matrix that we saw a few slides ago, where we had an image and we had logit values or similarity values between that image and every single piece of text that we passed into the model.
[17:30]
That's exactly what we're extracting here. We're extracting the similarity score of the input image to both the text inputs. Then we can convert these logits to probabilities via the softmax. And then, this will give us a prediction. What I'm doing here is I'm just doing argmax of the probabilities tensor and using that to pick out the predicted class from this original list that I created. And then I'm just going to print everything like this. So I'll print the predicted class as well as a rounded probability corresponding to that class.
[18:00]
With that, the most probable class is a photo of a cat with a associated probability of 99.79%. So, basically nails the classification of this image. But let's see what happens when we use different text classes. Instead of passing in a photo of a cat and a photo of a dog, which are pretty easy classes to distinguish between, let's try something a bit more nuanced like an ugly cat versus cute cat. And again here, the model basically nails it with a 97% probability of this cute cat class.
[18:30]
And then we can try something even more challenging like trying to distinguish if this is a cat meme or not a cat meme. And it indeed gets that it's not a cat meme, but we can see that the probability dropped significantly. And then as a final test of this model, let's see what happens when we pass in an actual cat meme and give it the class choices of cat meme versus not cat meme. And so here the model again nails it.
[19:00]
It correctly classifies this as a cat meme with a probability of 83%. And so again, what we're doing here using CLIP is we're taking these three entities, we're taking the text sequence of cat meme, the text sequence of not cat meme, and this image of a cat, encoding them in a shared embedding space, and we're evaluating the similarity between this image of a cat and the text sequence "cat meme", and the similarity between this image of a cat and the text sequence "not a cat meme".
[19:30]
And then we can convert that similarity into a probability as well as a class prediction. The key unlock here is that you are not restricted or limited in the different class labels you can use for image classification. You can be as detailed or vague as you like. You can adapt this to endless different use cases, which is pretty amazing. This second example is basically the inverse of zero shot image classification.
[20:00]
There we had an input image and we wanted to match it with one of the input text sequences. Here in example two, we're going to do the exact opposite. So instead of starting with an image, we're going to start with a piece of text, in other words, a search query, and then we're going to match it to a set of images. So essentially, what we're doing is we're doing a search over a set of images.
[20:30]
The way this looks is we'll first load in our images. Here, we have a picture of a cute cat, a picture of a dog, and a picture of a goat. We'll store them in this image list. We're using the PIL library to open the images and just store them in this list. Then we're going to define a query and process the inputs.
[21:00]
Here, our query will be "a cute dog". And then we'll pass this query along with our image list through the processor, so it's in the appropriate format for CLIP. Then we'll run these inputs through our model, get these outputs, extract the logits per text now. Before we did logits per image, now we're doing logits per text. So these are going to be the similarity scores between the input text and all the images that we inputted.
[21:30]
And then we'll convert these logits into probabilities. So with that, we can evaluate the best match. So, I'm doing that again in a similar way. So we have these probabilities, doing argmax, which will give us an integer 0, 1 or 2. We can use that to pick out the best matched image, and then we can take the probability associated with that image, and then we can just print everything like this. So again, the query here was a cute dog, and this is the best matched image with a probability of about 98%.
[22:00]
But again, that was a super easy example. So let's try a trickier query like "something cute but metal." In this case, the model returns the goat, which is indeed cute, but also goats are associated with heavy metal music, and it got a 77% match probability. Reading this, a good boy, the text itself doesn't have anything to do with animals. You know, maybe it's a human boy and he's well behaved.
[22:30]
But a good boy is a colloquialism for dogs that we use often. And the model can pick that up quite easily. So it matches it with a dog with 82% probability. It would be interesting to see if we threw in a picture of a human boy to see how the model would handle that case. This could be something that you do with the example code from the GitHub. And then we can try an extremely controversial query like "the best pet in the world".
[23:00]
For this, the model returns a cat with a 56% match probability. This is likely indicating that on average, people on the internet love cats more than they love dogs. Nevertheless, it's super interesting how we can use this model in order to do search like this. So those were the two examples. Code is on the GitHub, link in the description below.
[23:30]
Let's look ahead to the next video of this series. In the previous video, so part one, we talked about multimodal large language models. So basically, large language models that can process or generate multiple data modalities. In this video, we talked about multimodal embeddings, like those generated by CLIP, which can be used to do things like image search.
[24:00]
So we pass in a query and a set of potential images, and then it'll spit out the best matched image. In the next video of this series, we're going to bring these two ideas together to create a multimodal RAG system. The basic flow will be to take a user query like "What's there to do in Bali?" We'll pass the query into a multimodal retrieval system, which involves using a multimodal embedding model to pick out the documents and images that are most relevant to this query.
[24:30]
We'll take the user query and relevant documents and images to generate a prompt. And then we'll pass that prompt into a multimodal large language model, which can process the user query, relevant text documents, and relevant images to generate a helpful response. And as a final note, if you enjoyed this video and you want to learn more, check out the blog published in Towards Data Science. There I went into some details that I probably missed here. And as always, even though this is going to be a member-only story, you can access it completely for free using the friend link in the description below. And with that, thank you so much for your time and thanks for watching.

</details>


## Additional Sources Scraped

<details>
<summary>arxiv-org</summary>

Documents are visually rich structures that convey information through text, but also figures, page layouts, tables, or even fonts. Since modern retrieval systems mainly rely on the textual information they extract from document pages to index documents -often through lengthy and brittle processes-, they struggle to exploit key visual cues efficiently. This limits their capabilities in many practical document retrieval applications such as Retrieval Augmented Generation (RAG). To benchmark current systems on visually rich document retrieval, we introduce the Visual Document Retrieval Benchmark ViDoRe, composed of various page-level retrieval tasks spanning multiple domains, languages, and practical settings. The inherent complexity and performance shortcomings of modern systems motivate a new concept; doing document retrieval by directly embedding the images of the document pages. We release ColPali, a Vision Language Model trained to produce high-quality multi-vector embeddings from images of document pages. Combined with a late interaction matching mechanism, ColPali largely outperforms modern document retrieval pipelines while being drastically simpler, faster and end-to-end trainable. We release models, data, code and benchmarks under open licenses at [https://hf.co/vidore](https://hf.co/vidore).

---

Document Retrieval consists of matching a user query to relevant documents in a given corpus. It is central to many widespread industrial applications, either as a standalone ranking system (search engines) or as part of more complex information extraction or Retrieval Augmented Generation (RAG) pipelines.

Over recent years, pretrained language models have enabled large improvements in text embedding models. In practical industrial settings, however, the primary performance bottleneck for efficient document retrieval stems not from embedding model performance but from the prior data ingestion pipeline. Indexing a standard PDF document involves several steps. First, PDF parsers or Optical Character Recognition (OCR) systems are used to extract words from the pages. Document layout detection models can then be run to segment paragraphs, titles, and other page objects such as tables, figures, and headers. A chunking strategy is then defined to group text passages with some semantical coherence, and modern retrieval setups may even integrate a captioning step to describe visually rich elements in a natural language form, more suitable for embedding models. In our experiments (Table 2), we typically find that optimizing the ingestion pipeline yields much better performance on visually rich document retrieval than optimizing the text embedding model.

Contribution 1: ViDoRe. In this work, we argue that document retrieval systems should not be evaluated solely on the capabilities of text embedding models (Bajaj et al., 2016; Thakur et al., 2021; Muennighoff et al., 2022), but should also consider the context and visual elements of the documents to be retrieved. To this end, we create and openly release ViDoRe, a comprehensive benchmark to evaluate systems on page-level document retrieval with a wide coverage of domains, visual elements, and languages. ViDoRe addresses practical document retrieval scenarios, where queries often necessitate both textual and visual understanding for accurate document matching. We highlight the shortcomings of current text-centric systems in these settings.

---

Figure 1: ColPali simplifies document retrieval w.r.t. standard retrieval methods while achieving stronger performances with better latencies. Latencies and results are detailed in section 5 and subsection B.4.

Contribution 2: ColPali. We propose a novel concept and model architecture based on Vision Language Models (VLMs) to efficiently index documents purely from their visual features, allowing for subsequent fast query matching with late interaction mechanisms (Khattab & Zaharia, 2020). Our method, ColPali, significantly outperforms all other retrieval systems on $V i D o R e$ while being fast and end-to-end trainable. These results demonstrate the potential and the many benefits of this novel Retrieval in Vision Space concept, which could significantly alter the way document retrieval is approached in the industry moving forward. We release all resources at [https://hf.co/vidore](https://hf.co/vidore).

---

Problem Setting. In our setting, a retrieval system scores how relevant a document $d$ from corpus $\mathcal{D}$ is with respect to a query $q$. Computing the similarity score $s ( q , d ) \in \mathbb{R}$ for each of the $\| \mathcal{D} \|$ documents in the corpus creates a ranking we can use to extract the most relevant documents. In this work, we focus on page-level retrieval: given a query, is the correct document page retrieved by the system? For coherence with existing literature, we further use the term document to refer to individual pages, i.e. the atomic retrieved elements in our setting. As we focus on practical industrial retrieval applications (RAG, search engines) with potentially large corpora sizes, latency constraints are imposed on scoring systems. Most current retrieval systems can be decomposed into (1) an offline indexation phase in which a document index is built and (2) an online querying phase in which a query is matched to documents from the index and where low latency is vital to the user experience.

Under these industrial constraints, we identify three main properties an efficient document retrieval systems should exhibit: (R1) strong retrieval performance, as measured by standard retrieval metrics; (R2) fast online querying, measured through average latencies; (R3) high throughput corpus indexation, i.e. the number of pages that can be embedded in a given timeframe.

## TEXTUAL RETRIEVAL METHODS

**Document Retrieval in Text Space.**

Statistical methods based on word frequency like TF-IDF (Sparck Jones, 1972) and BM25 (Robertson et al., 1994) are still widely used due to their simplicity and efficiency. More recently, neural embedding models based on fine-tuned large language models display state-of-the-art performance on a variety of text embedding tasks and top the retrieval leaderboards (Muennighoff et al., 2022).

**Neural Retrievers.** In bi-encoder models (Reimers & Gurevych, 2019; Karpukhin et al., 2020; Wang et al., 2022), documents are independently mapped offline to a dense vector space. Queries are embedded online and matched to documents through a fast cosine distance computation. A slower, but slightly more performant alternative, cross-encoder systems (Wang et al., 2020; Cohere, 2024) concatenate query and document as a single input sequence and iteratively attribute matching scores to each possible combination. This enables full attention computation between query and document terms but comes at the cost of computational efficiency, as $\| \mathcal D \|$ encoding passes must be done online.

**Multi-Vector retrieval via late interaction.** In the late interaction paradigm introduced by ColBERT (Khattab & Zaharia, 2020), an embedding is pre-computed and indexed per document token. At runtime, similarity can be computed with individual query token embeddings. The idea is to benefit from the rich interaction between individual query and document terms while taking advantage of the offline computation and fast query matching enabled by bi-encoders.

**Retrieval Evaluation.** Although benchmarks and leaderboards have been developed to evaluate text embedding models (Thakur et al., 2021; Muennighoff et al., 2022), much of the performance improvements in industrial use cases of embedding models stem from the prior data ingestion pipeline. While documents often rely on visual elements to more efficiently convey information to human readers, text-only systems barely tap into these visual cues. Other work has also independently studied table or chart retrieval systems through repurposed Question Answering datasets (Zhang et al., 2019; Nowak et al., 2024) but only assessing specialized methods for each task.

To our knowledge, no benchmark evaluates document retrieval systems in practical settings; in an end-to-end manner, across several document types and topics, and by evaluating the use of both textual and visual document features.

---

## INTEGRATING VISUAL FEATURES

**Contrastive Vision Language Models.** Mapping latent representations of textual content to corresponding representations of visual content has been done by aligning disjoint visual and text encoders through contrastive losses (Radford et al., 2021; Zhai et al., 2023). While some OCR capabilities exist in these models, the visual component is often not optimized for text understanding.

The Fine-grained Interactive Language-Image Pre-training (Yao et al., 2021) framework extends the late interaction mechanism to cross-modal Vision Language Models, relying on max similarity operations between text tokens and image patches.

**Visually Rich Document Understanding.** To go beyond text, some document-focused models jointly encode text tokens alongside visual or document layout features (Appalaraju et al., 2021; Kim et al., 2021; Huang et al., 2022; Tang et al., 2022). Large Language transformer Models (LLMs) with strong reasoning capabilities have recently been combined with Vision Transformers (ViTs) (Dosovitskiy et al., 2020) to create VLMs (Alayrac et al., 2022; Liu et al., 2023; Bai et al., 2023; Laurençon et al., 2024b) where image patch vectors from contrastively trained ViT models (Zhai et al., 2023) are fed as input embeddings to the LLM and concatenated with the text-token embeddings.

**PaliGemma.** The PaliGemma-3B model (Beyer et al., 2024) extends concepts from Pali3 (Chen et al., 2023), and projects SigLIP-So400m/14 (Alabdulmohsin et al., 2023) patch embeddings into Gemma-2B’s text vector space (Gemma Team et al., 2024). Along with its reasonable size w.r.t. other performant VLMs, an interesting property of PaliGemma’s text model is that it is fine-tuned with full-block attention on the prefix (instruction text and image tokens).

VLMs display enhanced capabilities in Visual Question Answering, captioning, and document understanding (Yue et al., 2023), but are not optimized for retrieval tasks.

---

Existing benchmarks for contrastive vision-language models primarily evaluate retrieval for natural images (Lin et al., 2014; Borchmann et al., 2021; Thapliyal et al., 2022). On the other hand, textual retrieval benchmarks (Muennighoff et al., 2022) are evaluated at at textual passage level and are not tailored for document retrieval tasks. We fill the gap with ViDoRe, a comprehensive benchmark for document retrieval using visual features.

---

ViDoRe is designed to comprehensively evaluate retrieval systems on their capacity to match queries to relevant documents at the page level. This benchmark encompasses multiple orthogonal subtasks, with focuses on various modalities - text, figures, infographics, tables; thematic domains - medical, business, scientific, administrative; or languages - English, French. Tasks also span varying levels of complexity, in order to capture signals from both weaker and stronger systems. As many systems require large amounts of time to index pages (captioning-based approaches can take dozens of seconds per page for instance), we limit the number of candidate documents for each retrieval task in order to evaluate even complex systems in a reasonable timeframe without sacrificing quality. For trainable retrieval systems, we provide a reference training set that can be used to facilitate comparisons.

Table 1: ViDoRe comprehensively evaluates multimodal retrieval methods.

| Dataset      | Language | # Queries | # Documents | Description                            |
|--------------|----------|-----------|-------------|----------------------------------------|
| Academic Tasks |
| DocVQA      | English  | 500       | 500         | Scanned documents from UCSF Industry   |
| InfoVQA     | English  | 500       | 500         | Infographics scrapped from the web     |
| TAT-DQA     | English  | 1600      | 1600        | High-quality financial reports         |
| arXiVQA     | English  | 500       | 500         | Scientific Figures from arXiv          |
| TabFQuAD    | French   | 210       | 210         | Tables scrapped from the web           |
| Practical Tasks |
| Energy      | English  | 100       | 1000        | Documents about energy                 |
| Government  | English  | 100       | 1000        | Administrative documents               |
| Healthcare  | English  | 100       | 1000        | Medical documents                      |
| AI          | English  | 100       | 1000        | Scientific documents related to AI     |
| Shift Project | French | 100       | 1000        | Environmental reports                  |

---

Academic Tasks. We repurpose widely used visual question-answering benchmarks for retrieval tasks: for each page-question-answer triplet, we use the question as the query, and the associated page as the gold document (Table 1). These academic datasets either focus on single specific modalities or target more varied visually rich documents. Moreover, we consider TabFQuAD, a human-labeled dataset on tables extracted from French industrial PDF documents released with this work.

Practical tasks. We construct topic-specific retrieval benchmarks spanning multiple domains to go beyond repurposed QA datasets and evaluate retrieval in more realistic industrial situations (e.g. RAG). To achieve this, we collect publicly accessible PDF documents and generate queries pertaining to document pages using Claude-3 Sonnet, a high-quality proprietary vision-language model (Anthropic, 2024). In total, we collect 1,000 document pages per topic, which we associate with 100 queries extensively filtered for quality and relevance by human annotators. The corpus topics are intentionally specific to maximize syntactic proximity between documents, creating more challenging retrieval tasks and covering an array of orthogonal domains (Table 1).

Evaluation Metrics. We evaluate performance on our benchmark (Requirement R1) using standard metrics from the retrieval literature (nDCG, Recall@K, MRR). We report $\mathrm{nDCG}@5$ values as the main performance metric in this work and release the complete sets of results along with the models.

To validate compliance with practical industrial requirements, we also consider query latencies (R2) and indexing throughputs (R3).

---

Unstructured. We evaluate retrieval systems representative of those found in standard industrial RAG pipelines. As is common practice, we rely on the Unstructured off-the-shelf tool in the highest resolution settings to construct high-quality text chunks from PDF documents. Unstructured orchestrates the document parsing pipeline, relying on deep learning vision models to detect titles and document layouts, OCR engines to extract text in non-native PDFs, specialized methods or models to detect and reconstruct tables, and implements a chunking strategy that leverages the detected document structure to preserve section boundaries when concatenating texts. In our simplest Unstructured configuration (text-only), only textual elements are kept and figures, images, and tables are considered noisy information and are filtered out.

Unstructured + X. While Unstructured is a strong baseline by itself, we further augment Unstructured’s output by integrating the visual elements. In (+OCR), tables, charts, and images are run through an OCR engine, processed by Unstructured, and chunked independently. In (+Captioning), we set up a fully-fledged captioning strategy, in which we feed visual elements to a strong proprietary Vision Language Model (Claude-3 Sonnet) to obtain highly detailed textual descriptions of the elements. Both strategies aim to integrate visual elements in the retrieval pipeline but incur significant latency and resource costs.

Embedding Model. To embed textual chunks, we evaluate Okapi BM25, the de facto standard sparse statistical retrieval method, and the dense encoder of BGE-M3, a multilingual neural method with SOTA performance in its size category. Chunks are embedded and scored independently, and page-level scores are obtained by max-pooling over the page’s chunk scores.

Contrastive VLMs. We also evaluate the strongest available vision-language embedding models: Jina CLIP, Nomic Embed Vision, and SigLIP-So400m/14.

Results. From a performance perspective, best results are obtained by combining the Unstructured parser with visual information, either from captioning strategies or by running OCR on the visual elements (Table 2). Little difference is seen between BM25 and BGE-M3 embeddings highlighting the visual information bottleneck. Contrastive VLMs lag behind. Beyond retrieval performance (R1), the indexing latencies (R2) illustrate that PDF parsing pipelines can be very lengthy, especially when incorporating OCR or captioning strategies. Querying latencies at runtime (R3) are very good for all evaluated systems (≤22 ms on a NVIDIA L4) due to fast query encoding and cosine similarity matching.

---

Figure 2: Offline document indexing with ColPali is much simpler and faster compared to standard retrieval methods. The PDF Parser results are obtained following the Unstructured settings with BGE-M3. All indexing speeds are averaged per-page latencies.

---

## 4 LATE INTERACTION BASED VISION RETRIEVAL

### 4.1 ARCHITECTURE

**Vision-Language Models.** Encouraged by their strong document understanding capabilities, we propose adapting recent VLMs for retrieval. The key concept is to leverage the alignment between output embeddings of text and image tokens acquired during multi-modal fine-tuning. To this extent, we introduce ColPali, a Paligemma-3B extension that is capable of generating ColBERT-style multivector representations of text and images (Figure 1). PaliGemma-3B is a strong candidate due to its small size, the many released checkpoints fine-tuned for different image resolutions and tasks, and the promising performances on various document understanding benchmarks. We add a projection layer to map each of the language model’s output token embeddings (whether from text or image tokens) to a vector space of reduced dimension $D=128$ as used in the ColBERT paper to keep lightweight bag-of-embedding representations.

**Late Interaction.** Given query $q$ and document $d$, we denote as $\mathbf{E_q} \in \mathbb{R}^{N_q\times D}$ and $\mathbf{E_d} \in \mathbb{R}^{N_d\times D}$ their respective multi-vector representation in the common embedding space $\mathbb{R}^D$, where $N_q$ and $N_d$ are respectively the number of vectors in the query and in the document page embeddings. The late interaction operator, $\operatorname{LI}(q,d)$, is the sum over all query vectors $\mathbf{E_q}^{(j)}$, of its maximum dot product $\langle \cdot \| \cdot \rangle$ with each of the $N_d$ document embedding vectors $\mathbf{E_d(1:N_d)}$.

$$
\mathbf{L}(q, d) = \sum_{i \in [1,N_q]} \operatorname{max}_{j \in [1,N_d]} \langle \mathbf{E_q}^{(i)} \| \mathbf{E_d}^{(j)} \rangle
$$

**Contrastive Loss.** The Late Interaction operation is fully differentiable, enabling backpropagation. Let a batch $\{q_k, d_k\}_{k \in [1, b]}$ composed of $b$ query-page pairs, where for all $k$, the document page $d_k$ is the document corresponding to query $q_k$. Following Khattab & Zaharia (2020), we define our in-batch contrastive loss $\mathcal{L}$ as the softmaxed cross-entropy of the positive scores $s_k^+ = \mathrm{LI}(q_k,d_k)$ w.r.t. to the maximal in-batch negative scores $s_k^- = \operatorname{max}_{l, l \neq k}~\mathrm{LI}(q_k, d_l)$:

$$
\mathcal{L} = - \frac{1}{b} \sum_{k=1}^b \log\left[ \frac{\exp(s_k^+)}{\exp(s_k^+) + \exp(s_k^-)} \right] = \frac{1}{b} \sum_{k=1}^b \log(1 + \exp(s_k^- - s_k^+))
$$

---

**Dataset.** Our training dataset of 118,695 query-page pairs is comprised of train sets of openly available academic datasets (63%) and a synthetic dataset made up of pages from web-crawled PDF documents and augmented with VLM-generated (Claude-3 Sonnet) pseudo-questions (37%). A validation set is created with 2% of the samples to tune hyperparameters.

**Parameters.** All models are trained for 1 epoch on the train set. Unless specified otherwise, we train models in bfloat16 format, use low-rank adapters (LoRA) on the transformer layers from the language model, as well as the final randomly initialized projection layer, and use a paged adamw 8bit optimizer.

**Query Augmentation.** As in Khattab & Zaharia (2020), we append 5 tokens to the query tokens to serve as a soft, differentiable query expansion or re-weighting mechanism.

---

Table 2: Comprehensive evaluation of baseline models and our proposed method on ViDoRe. Results are presented using $\mathrm{nDCG}@5$ metrics, and illustrate the impact of different components. Text-only metrics are not computed for benchmarks with only visual elements.

---

We show performance is achieved iteratively through the combination of three factors:

1. **A carefully crafted task-specific dataset.**
2. **Pairing a pretrained LLM to a vision model to better leverage text semantics from the image.**
3. **Using multi-vector embeddings rather than a single vector representation to better capture the vast amount of visual information present in a document.**

**Fine-tuning a Vision Model on a document retrieval oriented dataset: BiSigLIP.** SigLIP is a strong vision-language bi-encoder producing single vector embeddings, and pretrained on billions of image-text pairs. Further fine-tuning the textual component of this model on our document-oriented dataset (BiSigLIP) yields clear improvements across the board, particularly on figure retrieval and table retrieval tasks.

**Feeding image patches to a LLM: BiPali.** In the PaliGemma model architecture, SigLIP-generated patch embeddings are fed to a text language model and we can obtain LLM contextualized output patch embeddings. This technique aligns the image token representations with the text token embeddings in the LLM’s embeddings space, and augments the vision model embeddings with the language model’s text understanding capabilities. We average pool these representations to obtain a single dense vector, effectively creating a PaliGemma bi-encoder model (BiPali). After fine-tuning on the training dataset, we obtain a model that performs slightly worse in English than the tuned BiSigLIP variant. However, we see notable improvements in French tasks, indicating that BiPali’s LLM (Gemma 2B) helps multilingual text understanding. This is particularly notable as our training dataset does not contain non-English samples.

**Leveraging Multi-Vector Embeddings through Late Interaction: ColPali.** One benefit of inputting image patch embeddings through a language model is that they are natively mapped to a latent space similar to the textual input (query). This enables leveraging the ColBERT strategy to construct one embedding per image patch token, and at inference compute all interactions between text tokens and image patches, resulting in a step-change improvement in performance compared to BiPali.

Results in Table 2 show that our ColPali model also largely outperforms the strong baselines based on Unstructured and captioning, as well as all evaluated text-image embedding models. The difference is particularly stark on the more visually complex benchmark tasks, such as InfographicVQA, ArxivQA, and TabFQuAD, respectively representing infographics, figures, and tables. However, text-centric documents are also better retrieved by the ColPali models across all evaluated domains and languages, making our approach the overall best-performing document-retrieval model.

---

**Online Querying.** Querying latencies differ between ColPali and a BGE-M3 embedding model. For BGE, encoding takes about 22 ms for 15 tokens, while encoding a query with ColPali’s language model takes about 30 ms. For smaller corpus sizes, computing the late interaction operation induces marginally small overheads (approx. 1 ms per 1000 pages in the corpus), and the cosine similarity computation between bi-encoder vectors is even faster. Optimized late interaction engines enable to easily scale corpus sizes to millions of documents with reduced latency degradations.

**Offline Indexing.** Standard retrieval methods using bi-encoders represent each chunk as a single vector embedding, which is easy to store and fast to compute. However, processing a PDF to get the different chunks is the most time-consuming part (layout detection, OCR, chunking), and using captioning to handle multimodal data will only exacerbate this already lengthy process. On the other hand, ColPali directly encodes pages from their image representation. Although the model is larger than standard retrieval encoders, skipping the preprocessing allows large speedups at indexing. As pages are embedded end-to-end in single forward pass, the VRAM usage depends exclusively on the sequence length (number of patches per image) which is fixed as well, enabling efficient batching strategies to fully leverage hardware acceleration.

**Storage Footprint.** Our method requires storing a vector per image patch, along with 6 extra text tokens “Describe the image” concatenated to image patches. We project each PaliGemma vector to a lower dimensional space ($D=128$) to maximize efficiency, leading to a memory footprint of 257.5 KB per page. Memory footprint of the naive ColBERT indexing strategy can be drastically improved through compression and clustering mechanisms with minimal performance hits.

---

**Token pooling.** Token pooling is a CRUDE-compliant method (document addition/deletion-friendly) that aims to reduce the amount of multi-vector embeddings. For ColPali, many image patches share redundant information, e.g. white background patches. By pooling these patches together, we can reduce the amount of embeddings while retaining most information. Retrieval performance with hierarchical mean token pooling on image embeddings shows that with a pool factor of 3, the total number of vectors is reduced by 66.7% while 97.8% of the original performance is maintained. More information dense documents may be prone to worse performance degradation with such pooling techniques.

---

Figure 3: (Left: Token Pooling) Relative performance degradation when reducing the number of stored embeddings per document. (Right: Interpretability) For each term in a user query, ColPali identifies the most relevant document image patches (highlighted zones) and computes a query-topage matching score.

---

**Interpretability:** By superimposing the late interaction heatmap on top of the original image, we can visualize the most salient image patches with respect to each term of the query, yielding interpretable insights into model focus zones.

---

**Tradeoffs between model size and the number of image patches.** We train a variant of PaliGemma with half the number of image patches (512). While we observe a clear performance degradation with respects to to the 1024-patch ColPali model, memory usage is much lower. As an alternative to PaliGemma, we train Idefics2-8B, a VLM with a similar architecture and based on a Mistral-7B language backbone and a SigLIP vision encoder paired with a perceiver resampler. The most notable differences with PaliGemma lie in the size of the language model (2B and 7B resp.) and the number of image patches (between 512 and 2048 for PaliGemma, and 64 post-resampling for Idefics2). Our results suggest better language models enable more efficient representations of image embeddings. However ColIdefics2 (64) remains less accurate than ColPali (1024) while being about twice as slow in terms of training and inference latency. These results suggest there are tradeoffs between performance, latencies during online querying and offline indexation phases, and index memory size.

**Unfreezing the vision component.** We train a ColPali variant by also backpropagating through and updating the vision encoder and the projection layer. This leads to a slight performance degradation.

**Impact of “query augmentation” tokens.** In ColBERT, special tokens are concatenated to the input query to serve as soft query augmentation buffers. Training without these tokens, we observe no significant performance difference in the English benchmarks. However, performance on the French tasks seems to improve.

**Impact of the Pairwise CE loss.** Training with an in-batch negative contrastive loss, instead of the pairwise CE loss that only considers the hardest negative sample, leads to a slight performance degradation on the benchmark.

**Adapting models to new tasks.** ColPali can be trained end-to-end, directly optimizing the downstream retrieval task which greatly facilitates fine-tuning to boost performance on specialized domains, multilingual retrieval, or specific visual elements the model struggles with. To demonstrate, we add samples representing French tables and associated queries to the training set. We see clear $\mathrm{nDCG}@5$ improvements and even starker Recall@$1$ gains on the TabFQuAD benchmark, with no performance degradation on the rest of the benchmark tasks.

**Better VLMs lead to better visual retrievers.** We train the recently released Qwen2-VL 2B, a SOTA 2 billion parameter generative VLM, with the same data and training strategy, obtaining ColQwen2-VL. We observe clear performance improvements over ColPali, showcasing clear performance correlations between generative benchmarks performance and retrieving metrics.

**Out-of-domain generalization.** We train a ColPali variant solely using the recent DocMatix dataset, which we subsample to obtain a comparably-sized train set. Results on ViDoRe show the performance drop is minor, still outperforming the closest baseline method by over 12 points. ColPali generalizes well outside of its training distribution, and our results are not unreasonably boosted with respect to baselines that cannot be fine-tuned on the same data.

---

In this work, we introduced the Visual Document Retrieval Benchmark (ViDoRe), which evaluates document retrieval systems in realistic settings involving visually complex documents. We demonstrated that current retrieval pipelines and contrastive vision-language models struggle to efficiently exploit visual information embedded in documents, leading to suboptimal performance. To address this, we presented ColPali, a novel retrieval method that leverages Vision-Language Models to create high-quality, multi-vector embeddings purely from visual document features. ColPali largely outperforms the best existing document retrieval methods while enabling faster corpus indexing times and maintaining low querying latencies, thus circumventing many pain points of modern document retrieval applications. We hope to drive industrial adoption, and to encourage future work by publicly releasing the ViDoRe benchmark, the data, the codebase, and all models and baselines from our work.

Future Work. Beyond performance improvements that could be obtained through better data, backbone models or training strategies, our vision at term is to combine visual retrieval systems and visually grounded query answering to create end-to-end RAG systems that purely function from image features. This idea is supported by concurrent work showcasing the strong promises of VLMs for visual QA, and may eventually become a new industrial standard for document processing. In this line of work, reliability is key, and confidence estimation techniques for Information Retrieval methods could become central to implement abstention mechanisms, and are particularly interesting given the information rich multi-vector scoring mechanisms of late interaction systems. Expanding benchmarking efforts to cover more languages, modalities, and tasks is also a crucial future research direction.

---

**SIGLIP**

SigLIP (Sigmoid Loss for Language Image Pre-Training) builds upon CLIP (Contrastive LanguageImage Pretraining)—a foundational model that aligns images and text by maximizing the similarity between correct image-text pairs while minimizing it for incorrect ones, leveraging a contrastive loss. Unlike CLIP, which applies the softmax function to the logits, SigLIP uses the sigmoid activation function. This innovation eliminates the need for a global view of all pairwise similarities between images and texts within a batch, enabling more flexible batch size scaling. This approach allows SigLIP to achieve state-of-the-art performance in zero-shot image classification tasks.

---

**PALIGEMMA**

PaliGemma is a 3B-parameter vision-language model. It integrates the SigLIP vision encoder with a Gemma-2B language decoder, connected via a multimodal linear projection layer. The model processes images by segmenting them into a fixed number of Vision Transformer tokens, which are prepended to an optional text prompt.

A distinguishing feature of PaliGemma is its operation as a Prefix-Language Model (Prefix-LM). This design ensures full attention between image tokens and the user-provided input (prefix) while generating outputs auto-regressively (suffix). This architecture allows image tokens to access the task-specific query during processing, facilitating more effective task-dependent reasoning.

PaliGemma was trained in four stages: unimodal pretraining with existing components, extended multimodal pretraining, short high-resolution pretraining, and task-specific fine-tuning.

---

**COLBERT**

ColBERT (Contextualized Late Interaction over BERT) is a retrieval model designed to balance speed and effectiveness in information retrieval tasks. Traditional retrieval models are typically categorized based on their type of interaction: either processing queries and documents independently for efficiency (bi-encoders) or jointly to capture rich contextual relationships (crossencoders). ColBERT combines the advantages of both approaches through a novel late interaction mechanism.

Queries and documents are encoded separately using BERT, enabling offline pre-computation of document representations for scalability. Instead of pooling embeddings into a single vector, ColBERT retains token-level embeddings and employs a MaxSim operator to compute fine-grained similarity scores. For each query token, the model determines the maximum similarity with document tokens, summing these scores to compute relevance.

This architecture preserves the contextual richness of deep language models while significantly improving computational efficiency. By delaying the interaction step, ColBERT supports vector similarity indexing, facilitating end-to-end retrieval from large collections without prohibitive costs. Empirical evaluations on passage search datasets demonstrate that ColBERT achieves competitive effectiveness compared to existing BERT-based models, while executing queries orders of magnitude faster and with drastically reduced computational requirements.

---

## F EXAMPLES FROM THE ViDoRe BENCHMARK

### Energy

Query: What types of accounts or products allow investors to defer paying taxes?

Query: What is the estimated to-

Query: What is the projected

tal savings for a PV system in peak electricity demand in Cali

Durham under the net metering fornia for the year 2030?

(flat rate) billing option over the stem’s useful life of 25 years?

https://arxiv.org/pdf/images/1f9bb209c65efe578ba50868a9c38ce8cb6f246966ae148e4620d11a0e607212.jpg

https://arxiv.org/pdf/images/3960e4108709e4059f26cfe162eb3d936c3067e514cb08ca74f06162a8133d67.jpg

https://arxiv.org/pdf/images/e42b7fab5707392bc051a93970281ac113009ce37c322c63a68c4abebcd55ef7.jpg

---

### Artificial Intelligence

Query: What are some common outcome areas targeted by TAII for different age groups?

Query: What did the robot moni-

Query: What is the key approach

tor to determine when to activate used in the PDP architecture?

or deactivate the blower motor

and blinker?

https://arxiv.org/pdf/images/9016549ff01b9fbab36065594cd1ee58efc6b7d6f0e48d20e3de677225676ae7.jpg

https://arxiv.org/pdf/images/bd0530bcbb330f02782de1f5c1297af8b783f13733b31cc36cd41350f36dfa27.jpg

---

### Healthcare Industry

Query: What is the chemical formula for the ferroelectric material Lead Zirconium Titanate (PZT)?

Query: What government entities are involved in public financing for healthcare in the US?

Query: What does the AVPU scale stand for in assessing the level of consciousness of a seriously ill child?

https://arxiv.org/pdf/images/1444da993a6ee3b1f59c12ed057e76987d6d04ec703b309af1f57ca2a86b3e76.jpg

https://arxiv.org/pdf/images/38372b0dbe77a01b8b69787f56cb4099efbb3dcb110561bbe169fd3229ace1cb.jpg

https://arxiv.org/pdf/images/91878a732ed3319672984664297e6dc86f53c974a9af005b68884f221dcec18c.jpg

---

### Government Reports

Query: What are some mandates for the EPA under the Pollution Prevention Act?

Query: What is the strategy of KPMG Hazem Hassan?

https://arxiv.org/pdf/images/adbe49c71bb05e93ba1043d1c60f64fa6debd89e45f60c536c414803465d5c0f.jpg

Query: What is the trust signal score for the consumer industry best-in-class archetype?

https://arxiv.org/pdf/images/346df2ccef49c90eab7668c90dbca02d5f207f8c91880b4b9767e25c61c384aa.jpg

https://arxiv.org/pdf/images/d05ca9155dc858c869d56a125f9ebfcf2b158cb761d7e7acddc5f4f01f51b550.jpg

https://arxiv.org/pdf/images/73685f10f6769ee7ce32ed6752357e4d12e341abf7bd8c352f5decc74797d729.jpg

---

### Shift

Query: Selon le graphique, quelle est la capacité d’import et la consommation réelle de carburants SAF (biocarburants durables pour l’aviation) prévues en 2050 ?

Query: Quelle partie de la production pétrolière du Kazakhstan provient de champs en mer ?

Query: Quels sont les pays ayant la plus grande part des découvertes cumulées de pétrole brut en 2020 (en milliers de barils, hors découvertes cumulées) ?

https://arxiv.org/pdf/images/58ff928d61e2c6526f0ab107700f7119a4d1e1524fab70beb3e26f175712cb60.jpg

https://arxiv.org/pdf/images/32a1b7a13ef9773e94478799bd2ca046af67fa42f07bd30746fe0e962ef4e9dd.jpg

https://arxiv.org/pdf/images/2289f86f67ae91ed3003029d330ee9e6b6ec529bfa838ea82ced746417f7b1f1.jpg

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

The text can come in a variety of fonts, sizes, and colors, can be rotated or upside down, and contains special symbols. Ready-made OCR software like iText and OCRSpace can detect simple text with high accuracy, but they fail spectacularly when it comes to technical drawings (or any other complex document, for that matter). For example, these tools struggle to detect rotated text.

OCR tools often have trouble detecting rotated text | Image by author

Most OCR tools can be fine-tuned to handle problematic text better. The best approach to recognizing complex text is to use multiple fine-tuned OCR tools along with a balancer that compares the results of each tool and chooses the one that produces the most accurate results.

Another benefit of using fine-tuned OCR software is the increase in recognition speed.

Fine-tuning of OCR software leads to better results | Image by author

By fine-tuning these tools alone, we’ve seen a 200 times decrease in document processing speed. If you add an OCR engine into the equation, like Tesseract, the text recognition quality can be increased up to 99.9%.

## Stage 2: Recognition of special symbols

Each technical drawing includes special symbols of some sort. In the case of floor plan technical drawings, the documents include symbols designating doors, windows, electrical outlets, etc.

These symbols, or labels, look like geometric figures with text inside. They can be difficult to distinguish from their surroundings due to their shape, which blends in perfectly with the rest of the drawing.

In addition, there can be multiple labels representing the same object due to inconsistencies in document design.

Similar looking objects are often detected as the same one | Image by author

Pre-trained computer vision solutions, like OpenCV libraries for symbol detection, work best with photographs of real-life objects. Technical drawings are quite a bit different: they are almost always in black and white and mostly consist of geometric shapes.

We’ve tested multiple OpenCV libraries, each of which resulted in albeit different, yet insufficiently low recognition quality. Unless you develop your own neural network from scratch, any pre-trained computer vision model needs to be built upon to achieve decent recognition quality.

One of the main problems with using [pre-trained CV models](https://hackernoon.com/creating-computer-vision-apps-without-building-media-pipelines?ref=hackernoon.com) is the amount of false positive results they produce. Technical drawings consist of simple geometric shapes, but so do special symbols and labels, which results in CV models detecting random parts of the drawings as labels.

The best way of mitigating this issue is to implement deep learning to detect false positive results and remove them from the final detection results.

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
<summary>image-understanding-gemini-api-google-ai-for-developers</summary>

# Image understanding

Gemini models are built to be multimodal from the ground up, unlocking a wide range of image processing and computer vision tasks including but not limited to image captioning, classification, and visual question answering without having to train specialized ML models.

## Passing images to Gemini

You can provide images as input to Gemini using two methods:

- Passing inline image data: Ideal for smaller files (total request
size less than 20MB, including prompts).
- Uploading images using the File API: Recommended for larger files or for
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

```
import { GoogleGenAI } from "@google/genai";
import * as fs from "node:fs";

const ai = new GoogleGenAI({});
const base64ImageFile = fs.readFileSync("path/to/small-sample.jpg", {
  encoding: "base64",
});

const contents = [\
  {\
    inlineData: {\
      mimeType: "image/jpeg",\
      data: base64ImageFile,\
    },\
  },\
  { text: "Caption this image." },\
];

const response = await ai.models.generateContent({
  model: "gemini-2.5-flash",
  contents: contents,
});
console.log(response.text);

```

```
bytes, _ := os.ReadFile("path/to/small-sample.jpg")

parts := []*genai.Part{
  genai.NewPartFromBytes(bytes, "image/jpeg"),
  genai.NewPartFromText("Caption this image."),
}

contents := []*genai.Content{
  genai.NewContentFromParts(parts, genai.RoleUser),
}

result, _ := client.Models.GenerateContent(
  ctx,
  "gemini-2.5-flash",
  contents,
  nil,
)

fmt.Println(result.Text())

```

```
IMG_PATH="/path/to/your/image1.jpg"

if [[ "$(base64 --version 2>&1)" = *"FreeBSD"* ]]; then
B64FLAGS="--input"
else
B64FLAGS="-w0"
fi

curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent" \
-H "x-goog-api-key: $GEMINI_API_KEY" \
-H 'Content-Type: application/json' \
-X POST \
-d '{
    "contents": [{\
    "parts":[\
        {\
            "inline_data": {\
            "mime_type":"image/jpeg",\
            "data": "'"$(base64 $B64FLAGS $IMG_PATH)"'"\
            }\
        },\
        {"text": "Caption this image."},\
    ]\
    }]
}' 2> /dev/null

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

```
import { GoogleGenAI } from "@google/genai";

async function main() {
  const ai = new GoogleGenAI({});

  const imageUrl = "https://goo.gle/instrument-img";

  const response = await fetch(imageUrl);
  const imageArrayBuffer = await response.arrayBuffer();
  const base64ImageData = Buffer.from(imageArrayBuffer).toString('base64');

  const result = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: [\
    {\
      inlineData: {\
        mimeType: 'image/jpeg',\
        data: base64ImageData,\
      },\
    },\
    { text: "Caption this image." }\
  ],
  });
  console.log(result.text);
}

main();

```

```
package main

import (
  "context"
  "fmt"
  "os"
  "io"
  "net/http"
  "google.golang.org/genai"
)

func main() {
  ctx := context.Background()
  client, err := genai.NewClient(ctx, nil)
  if err != nil {
      log.Fatal(err)
  }

  // Download the image.
  imageResp, _ := http.Get("https://goo.gle/instrument-img")

  imageBytes, _ := io.ReadAll(imageResp.Body)

  parts := []*genai.Part{
    genai.NewPartFromBytes(imageBytes, "image/jpeg"),
    genai.NewPartFromText("Caption this image."),
  }

  contents := []*genai.Content{
    genai.NewContentFromParts(parts, genai.RoleUser),
  }

  result, _ := client.Models.GenerateContent(
    ctx,
    "gemini-2.5-flash",
    contents,
    nil,
  )

  fmt.Println(result.Text())
}

```

```
IMG_URL="https://goo.gle/instrument-img"

MIME_TYPE=$(curl -sIL "$IMG_URL" | grep -i '^content-type:' | awk -F ': ' '{print $2}' | sed 's/\r$//' | head -n 1)
if [[ -z "$MIME_TYPE" || ! "$MIME_TYPE" == image/* ]]; then
  MIME_TYPE="image/jpeg"
fi

# Check for macOS
if [[ "$(uname)" == "Darwin" ]]; then
  IMAGE_B64=$(curl -sL "$IMG_URL" | base64 -b 0)
elif [[ "$(base64 --version 2>&1)" = *"FreeBSD"* ]]; then
  IMAGE_B64=$(curl -sL "$IMG_URL" | base64)
else
  IMAGE_B64=$(curl -sL "$IMG_URL" | base64 -w0)
fi

curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{\
        "parts":[\
            {\
              "inline_data": {\
                "mime_type":"'"$MIME_TYPE"'",\
                "data": "'"$IMAGE_B64"'"\
              }\
            },\
            {"text": "Caption this image."}\
        ]\
      }]
    }' 2> /dev/null

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

```
import {
  GoogleGenAI,
  createUserContent,
  createPartFromUri,
} from "@google/genai";

const ai = new GoogleGenAI({});

async function main() {
  const myfile = await ai.files.upload({
    file: "path/to/sample.jpg",
    config: { mimeType: "image/jpeg" },
  });

  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: createUserContent([\
      createPartFromUri(myfile.uri, myfile.mimeType),\
      "Caption this image.",\
    ]),
  });
  console.log(response.text);
}

await main();

```

```
package main

import (
  "context"
  "fmt"
  "os"
  "google.golang.org/genai"
)

func main() {
  ctx := context.Background()
  client, err := genai.NewClient(ctx, nil)
  if err != nil {
      log.Fatal(err)
  }

  uploadedFile, _ := client.Files.UploadFromPath(ctx, "path/to/sample.jpg", nil)

  parts := []*genai.Part{
      genai.NewPartFromText("Caption this image."),
      genai.NewPartFromURI(uploadedFile.URI, uploadedFile.MIMEType),
  }

  contents := []*genai.Content{
      genai.NewContentFromParts(parts, genai.RoleUser),
  }

  result, _ := client.Models.GenerateContent(
      ctx,
      "gemini-2.5-flash",
      contents,
      nil,
  )

  fmt.Println(result.Text())
}

```

```
IMAGE_PATH="path/to/sample.jpg"
MIME_TYPE=$(file -b --mime-type "${IMAGE_PATH}")
NUM_BYTES=$(wc -c < "${IMAGE_PATH}")
DISPLAY_NAME=IMAGE

tmp_header_file=upload-header.tmp

# Initial resumable request defining metadata.
# The upload url is in the response headers dump them to a file.
curl "https://generativelanguage.googleapis.com/upload/v1beta/files" \
  -H "x-goog-api-key: $GEMINI_API_KEY" \
  -D upload-header.tmp \
  -H "X-Goog-Upload-Protocol: resumable" \
  -H "X-Goog-Upload-Command: start" \
  -H "X-Goog-Upload-Header-Content-Length: ${NUM_BYTES}" \
  -H "X-Goog-Upload-Header-Content-Type: ${MIME_TYPE}" \
  -H "Content-Type: application/json" \
  -d "{'file': {'display_name': '${DISPLAY_NAME}'}}" 2> /dev/null

upload_url=$(grep -i "x-goog-upload-url: " "${tmp_header_file}" | cut -d" " -f2 | tr -d "\r")
rm "${tmp_header_file}"

# Upload the actual bytes.
curl "${upload_url}" \
  -H "x-goog-api-key: $GEMINI_API_KEY" \
  -H "Content-Length: ${NUM_BYTES}" \
  -H "X-Goog-Upload-Offset: 0" \
  -H "X-Goog-Upload-Command: upload, finalize" \
  --data-binary "@${IMAGE_PATH}" 2> /dev/null > file_info.json

file_uri=$(jq -r ".file.uri" file_info.json)
echo file_uri=$file_uri

# Now generate content using that file
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{\
        "parts":[\
          {"file_data":{"mime_type": "'"${MIME_TYPE}"'", "file_uri": "'"${file_uri}"'"}},\
          {"text": "Caption this image."}]\
        }]
      }' 2> /dev/null > response.json

cat response.json
echo

jq ".candidates[].content.parts[].text" response.json

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

```
import {
  GoogleGenAI,
  createUserContent,
  createPartFromUri,
} from "@google/genai";
import * as fs from "node:fs";

const ai = new GoogleGenAI({});

async function main() {
  // Upload the first image
  const image1_path = "path/to/image1.jpg";
  const uploadedFile = await ai.files.upload({
    file: image1_path,
    config: { mimeType: "image/jpeg" },
  });

  // Prepare the second image as inline data
  const image2_path = "path/to/image2.png";
  const base64Image2File = fs.readFileSync(image2_path, {
    encoding: "base64",
  });

  // Create the prompt with text and multiple images

  const response = await ai.models.generateContent({

    model: "gemini-2.5-flash",
    contents: createUserContent([\
      "What is different between these two images?",\
      createPartFromUri(uploadedFile.uri, uploadedFile.mimeType),\
      {\
        inlineData: {\
          mimeType: "image/png",\
          data: base64Image2File,\
        },\
      },\
    ]),
  });
  console.log(response.text);
}

await main();

```

```
// Upload the first image
image1Path := "path/to/image1.jpg"
uploadedFile, _ := client.Files.UploadFromPath(ctx, image1Path, nil)

// Prepare the second image as inline data
image2Path := "path/to/image2.jpeg"
imgBytes, _ := os.ReadFile(image2Path)

parts := []*genai.Part{
  genai.NewPartFromText("What is different between these two images?"),
  genai.NewPartFromBytes(imgBytes, "image/jpeg"),
  genai.NewPartFromURI(uploadedFile.URI, uploadedFile.MIMEType),
}

contents := []*genai.Content{
  genai.NewContentFromParts(parts, genai.RoleUser),
}

result, _ := client.Models.GenerateContent(
  ctx,
  "gemini-2.5-flash",
  contents,
  nil,
)

fmt.Println(result.Text())

```

```
# Upload the first image
IMAGE1_PATH="path/to/image1.jpg"
MIME1_TYPE=$(file -b --mime-type "${IMAGE1_PATH}")
NUM1_BYTES=$(wc -c < "${IMAGE1_PATH}")
DISPLAY_NAME1=IMAGE1

tmp_header_file1=upload-header1.tmp

curl "https://generativelanguage.googleapis.com/upload/v1beta/files" \
  -H "x-goog-api-key: $GEMINI_API_KEY" \
  -D upload-header1.tmp \
  -H "X-Goog-Upload-Protocol: resumable" \
  -H "X-Goog-Upload-Command: start" \
  -H "X-Goog-Upload-Header-Content-Length: ${NUM1_BYTES}" \
  -H "X-Goog-Upload-Header-Content-Type: ${MIME1_TYPE}" \
  -H "Content-Type: application/json" \
  -d "{'file': {'display_name': '${DISPLAY_NAME1}'}}" 2> /dev/null

upload_url1=$(grep -i "x-goog-upload-url: " "${tmp_header_file1}" | cut -d" " -f2 | tr -d "\r")
rm "${tmp_header_file1}"

curl "${upload_url1}" \
  -H "Content-Length: ${NUM1_BYTES}" \
  -H "X-Goog-Upload-Offset: 0" \
  -H "X-Goog-Upload-Command: upload, finalize" \
  --data-binary "@${IMAGE1_PATH}" 2> /dev/null > file_info1.json

file1_uri=$(jq ".file.uri" file_info1.json)
echo file1_uri=$file1_uri

# Prepare the second image (inline)
IMAGE2_PATH="path/to/image2.png"
MIME2_TYPE=$(file -b --mime-type "${IMAGE2_PATH}")

if [[ "$(base64 --version 2>&1)" = *"FreeBSD"* ]]; then
  B64FLAGS="--input"
else
  B64FLAGS="-w0"
fi
IMAGE2_BASE64=$(base64 $B64FLAGS $IMAGE2_PATH)

# Now generate content using both images
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{\
        "parts":[\
          {"text": "What is different between these two images?"},\
          {"file_data":{"mime_type": "'"${MIME1_TYPE}"'", "file_uri": '$file1_uri'}},\
          {\
            "inline_data": {\
              "mime_type":"'"${MIME2_TYPE}"'",\
              "data": "'"$IMAGE2_BASE64"'"\
            }\
          }\
        ]\
      }]
    }' 2> /dev/null > response.json

cat response.json
echo

jq ".candidates[].content.parts[].text" response.json

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

For more examples, check following notebooks in the [Gemini Cookbook](https://github.com/google-gemini/cookbook):

- [2D spatial understanding notebook](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Spatial_understanding.ipynb)
- [Experimental 3D pointing notebook](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/Spatial_understanding_3d.ipynb)

## Segmentation

Starting with Gemini 2.5, models not only detect items but also segment them
and provide their contour masks.

The model predicts a JSON list, where each item represents a segmentation mask.
Each item has a bounding box (" `box_2d`") in the format `[y0, x0, y1, x1]` with
normalized coordinates between 0 and 1000, a label (" `label`") that identifies
the object, and finally the segmentation mask inside the bounding box, as base64
encoded png that is a probability map with values between 0 and 255.
The mask needs to be resized to match the bounding box dimensions, then
binarized at your confidence threshold (127 for the midpoint).

````python
from google import genai
from google.genai import types
from PIL import Image, ImageDraw
import io
import base64
import json
import numpy as np
import os

client = genai.Client()

def parse_json(json_output: str):
  # Parsing out the markdown fencing
  lines = json_output.splitlines()
  for i, line in enumerate(lines):
    if line == "```json":
      json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
      output = json_output.split("```")[0]  # Remove everything after the closing "```"
      break  # Exit the loop once "```json" is found
  return json_output

def extract_segmentation_masks(image_path: str, output_dir: str = "segmentation_outputs"):
  # Load and resize image
  im = Image.open(image_path)
  im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)

  prompt = """
  Give the segmentation masks for the wooden and glass items.
  Output a JSON list of segmentation masks where each entry contains the 2D
  bounding box in the key "box_2d", the segmentation mask in key "mask", and
  the text label in the key "label". Use descriptive labels.
  """

  config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_budget=0) # set thinking_budget to 0 for better results in object detection
  )

  response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[prompt, im], # Pillow images can be directly passed as inputs (which will be converted by the SDK)
    config=config
  )

  # Parse JSON response
  items = json.loads(parse_json(response.text))

  # Create output directory
  os.makedirs(output_dir, exist_ok=True)

  # Process each mask
  for i, item in enumerate(items):
      # Get bounding box coordinates
      box = item["box_2d"]
      y0 = int(box[0] / 1000 * im.size[1])
      x0 = int(box[1] / 1000 * im.size[0])
      y1 = int(box[2] / 1000 * im.size[1])
      x1 = int(box[3] / 1000 * im.size[0])

      # Skip invalid boxes
      if y0 >= y1 or x0 >= x1:
          continue

      # Process mask
      png_str = item["mask"]
      if not png_str.startswith("data:image/png;base64,"):
          continue

      # Remove prefix
      png_str = png_str.removeprefix("data:image/png;base64,")
      mask_data = base64.b64decode(png_str)
      mask = Image.open(io.BytesIO(mask_data))

      # Resize mask to match bounding box
      mask = mask.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)

      # Convert mask to numpy array for processing
      mask_array = np.array(mask)

      # Create overlay for this mask
      overlay = Image.new('RGBA', im.size, (0, 0, 0, 0))
      overlay_draw = ImageDraw.Draw(overlay)

      # Create overlay for the mask
      color = (255, 255, 255, 200)
      for y in range(y0, y1):
          for x in range(x0, x1):
              if mask_array[y - y0, x - x0] > 128:  # Threshold for mask
                  overlay_draw.point((x, y), fill=color)

      # Save individual mask and its overlay
      mask_filename = f"{item['label']}_{i}_mask.png"
      overlay_filename = f"{item['label']}_{i}_overlay.png"

      mask.save(os.path.join(output_dir, mask_filename))

      # Create and save overlay
      composite = Image.alpha_composite(im.convert('RGBA'), overlay)
      composite.save(os.path.join(output_dir, overlay_filename))
      print(f"Saved mask and overlay for {item['label']} to {output_dir}")

# Example usage
if __name__ == "__main__":
  extract_segmentation_masks("path/to/image.png")

````

Check the
[segmentation example](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Spatial_understanding.ipynb#scrollTo=WQJTJ8wdGOKx)
in the cookbook guide for a more detailed example.

https://ai.google.dev/static/gemini-api/docs/images/segmentation.jpgAn example segmentation output with objects and segmentation masks

## Supported image formats

Gemini supports the following image format MIME types:

- PNG - `image/png`
- JPEG - `image/jpeg`
- WEBP - `image/webp`
- HEIC - `image/heic`
- HEIF - `image/heif`

## Capabilities

All Gemini model versions are multimodal and can be utilized in a wide range of
image processing and computer vision tasks including but not limited to image captioning,
visual question and answering, image classification, object detection and segmentation.

Gemini can reduce the need to use specialized ML models depending on your quality and performance requirements.

Some later model versions are specifically trained improve accuracy of specialized tasks in addition to generic capabilities:

- **Gemini 2.0 models** are further trained to support enhanced [object detection](https://ai.google.dev/gemini-api/docs/image-understanding#object-detection).

- **Gemini 2.5 models** are further trained to support enhanced [segmentation](https://ai.google.dev/gemini-api/docs/image-understanding#segmentation) in addition to [object detection](https://ai.google.dev/gemini-api/docs/image-understanding#object-detection).


## Limitations and key technical information

### File limit

Gemini 2.5 Pro/Flash, 2.0 Flash, 1.5 Pro, and 1.5 Flash support a
maximum of 3,600 image files per request.

### Token calculation

- **Gemini 1.5 Flash and Gemini 1.5 Pro**: 258 tokens if both dimensions
<= 384 pixels. Larger images are tiled (min tile 256px, max 768px, resized
to 768x768), with each tile costing 258 tokens.
- **Gemini 2.0 Flash and Gemini 2.5 Flash/Pro**: 258 tokens if both dimensions <= 384 pixels.
Larger images are tiled into 768x768 pixel tiles, each costing 258
tokens.

## Tips and best practices

- Verify that images are correctly rotated.
- Use clear, non-blurry images.
- When using a single image with text, place the text prompt _after_ the image part in the `contents` array.

## What's next

This guide shows you how to upload image files and generate text outputs from image
inputs. To learn more, see the following resources:

- [Files API](https://ai.google.dev/gemini-api/docs/files): Learn more about uploading and managing files for use with Gemini.
- [System instructions](https://ai.google.dev/gemini-api/docs/text-generation#system-instructions):
System instructions let you steer the behavior of the model based on your
specific needs and use cases.
- [File prompting strategies](https://ai.google.dev/gemini-api/docs/files#prompt-guide): The
Gemini API supports prompting with text, image, audio, and video data, also
known as multimodal prompting.
- [Safety guidance](https://ai.google.dev/gemini-api/docs/safety-guidance): Sometimes generative
AI models produce unexpected outputs, such as outputs that are inaccurate,
biased, or offensive. Post-processing and human evaluation are essential to
limit the risk of harm from such outputs.

</details>

<details>
<summary>multi-modal-ml-with-openai-s-clip-pinecone</summary>

# Multi-modal ML with OpenAI's CLIP

Language models (LMs) can not rely on language alone. That is the idea behind the “Experience Grounds Language” paper, that proposes a framework to measure LMs' current and future progress. A key idea is that, beyond a certain threshold LMs need other forms of data, such as visual input \[1\] \[2\].

https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F25e7f2f54b543af8c34c143448a4b0c55f77c6b5-2360x854.png&w=3840&q=75

World Scopes (WS), as datasets become larger in scope and span multiple modalities, the capabilities of models trained with them increase.

The next step beyond well-known language models; BERT, GPT-3, and T5 is _”World Scope 3”_. In World Scope 3, we move from large text-only datasets to large multi-modal datasets. That is, datasets containing information from multiple forms of media, like _both_ images and text.

The world, both digital and real, is multi-modal. We perceive the world as an orchestra of language, imagery, video, smell, touch, and more. This chaotic ensemble produces an inner state, our “model” of the outside world.

AI must move in the same direction. Even specialist models that focus on language or vision must, at some point, have input from the other modalities. How can a model fully understand the concept of the word “person” without _seeing_ a person?

OpenAI **C** ontrastive **L** earning **I** n **P** retraining (CLIP) is a world scope three model. It can comprehend concepts in both text and image and even connect concepts between the two modalities. In this chapter we will learn about multi-modality, how CLIP works, and how to use CLIP for different use cases like encoding, classification, and object detection.

* * *

## Multi-modality

The multi-modal nature of CLIP is powered by two encoder models trained to “speak the same language”. Text inputs are passed to a text encoder, and image inputs to an image encoder \[3\]. These models then create a _vector representation_ of the respective input.

Both models “speak the same language” by encoding similar concepts in text and images into similar vectors. That means that the text “two dogs running across a frosty field” would output a vector similar to an _image_ of two dogs running across a frosty field.

https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fa54a2f1fa0aeac03748c09df0fdfbb42aadc96b7-2430x1278.png&w=3840&q=75

Similar text and images will be encoded into a similar vector space. Dissimilar text and images do not share a similar vector space.

We can think of the language these models speak as the vector space in which they encode vectors. These two models can express nuanced information about text and images through this vector space. However, this “vector language” is far too abstract for us to directly understand.

Rather than directly reading this “language”, we can train other simple neural networks to understand it and make predictions that we can understand. Or we use vector search to identify similar concepts and patterns across text and image domains.

Let’s take a look at an example of CLIP in action.

### Text-to-Image Search

Entering a prompt in the search bar above allows us to search through images based on their _content_ rather than any attached textual metadata. We call this **C** ontent **B** ased **I** mage **R** etrieval (CBIR).

With CBIR, we can search for specific phrases such as “two dogs running across a frosty field”. We can even drop the word “dogs” and replace it with everyday slang for dogs like “good boy” or “mans best friend”, and we return the same images showing dogs running across fields.

CLIP can accurately understand language. It understands that _in the context_ of running across a field, we are likely referring to dogs and do not literally mean good children or someone’s “human” best friend.

Amusingly, the dataset contains no images of the food hot dogs (other than one). So, suppose we search for “hot dogs”. In that case, we first get an image containing a hot dog (and a dog), a dog looking toasty in a warm room, another dog looking warm with wooly clothing, and another dog posing for the camera. All of these portray a hot dog in one sense or another.

* * *

_After being processed by CLIP’s text or image encoder, we are left with vectors. That means we can search across_ **_any_** _modality with_ **_any_** _modality; we can search in either direction. We can also stick to a single modality, like text-to-text or image-to-image._

* * *

Now that we’ve seen what CLIP can do, let’s take a look at _how_ it can do this.

## CLIP

CLIP actually consists of two models trained in parallel. A 12-layer text transformer for building text embeddings and a ResNet or vision transformer (ViT) for building image embeddings \[3\].

https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F539716ea1571e459908c1fdc5a898fea239d8243-2803x1672.png&w=3840&q=75

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

With this, we can apply a loss function that maximizes the similarity between (T1,I1)(T\_1,I\_1)(T1​,I1​) and (T2,I2)(T\_2,I\_2)(T2​,I2​), and minimizes the similarity between (T1,I2)(T\_1,I\_2)(T1​,I2​) and (T2,I1)(T\_2,I\_1)(T2​,I1​). Altogether, this looks like this:

https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fd6868e6dae721512fed8f1287fc9ffe6b6a2cddd-2332x1342.png&w=3840&q=75

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
```

https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fa40f673ed52e07f497c7a39b032c27b33ce9f565-1128x761.png&w=3840&q=75

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

In\[5\]:

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

Out\[5\]:

```
dict_keys(['input_ids', 'attention_mask'])
```

This returns the typical text transformer inputs of `input_ids` and `attention_mask`.

The `input_ids` are token ID values where each token ID is an integer value ID that maps to a specific word or sub-word. For example the phrase _“multi-modality”_ may be split into tokens _\[“multi”, “-”, “modal”, “ity”\]_, which are then mapped to IDs _\[1021, 110, 2427, 425\]_.

A text transformer maps these token IDs to semantic vector embeddings that the model learned during pretraining.

The `attention_mask` is a tensor of 1s and 0s used by the model’s internal mechanisms to “pay attention” to real token IDs and ignore padding tokens.

* * *

_Padding tokens are a special type of token used by text transformers to create input sequences of a fixed length from sentences of varying length. They are appended to the end of shorter sentences, so “hello world” may become “hello world \[PAD\] \[PAD\] \[PAD\]”._

* * *

We then use CLIP to encode all of these text descriptions with `get_text_features` like so:

```python
text_emb = model.get_text_features(
    **tokens
)
```

One important thing to note here is that these embeddings are _not_ normalized. If we plan on using a similarity metric like the dot product, we must normalize the embeddings:

In\[9\]:

```python
print(text_emb.shape)
print(text_emb.min(), text_emb.max())
```

Out\[9\]:

```
torch.Size([21, 512])
tensor(-1.1893, grad_fn=<MinBackward1>) tensor(4.8015, grad_fn=<MaxBackward1>)

```

In\[40\]:

```python
# IF using dot product similarity, must normalize vectors like so...
import numpy as np

# detach text emb from graph, move to CPU, and convert to numpy array
text_emb = text_emb.detach().cpu().numpy()

# calculate value to normalize each vector by
norm_factor = np.linalg.norm(text_emb, axis=1)
norm_factor.shape
```

Out\[40\]:

```
(21,)
```

In\[41\]:

```python
text_emb = text_emb.T / norm_factor
# transpose back to (21, 512)
text_emb = text_emb.T
print(text_emb.shape)
print(text_emb.min(), text_emb.max())
```

Out\[41\]:

```
(21, 512)
-0.1526844 0.53449875

```

Alternatively, we can use cosine similarity as our metric as this only considers angular similarity and not vector magnitude (like dot product). For our examples, we will normalize and use dot product similarity.

We now have our text embeddings; let’s see how to do the same for images.

### Encoding Images

Images will be encoded using the ViT portion of CLIP. Similar to text encoding, we need to preprocess these images using the `preprocessor` like so:

In\[42\]:

```python
data['image'][0].size
```

Out\[42\]:

```
(6000, 3376)
```

In\[43\]:

```python
image_batch = data['image']

images = processor(
    text=None,
    images=image_batch,
    return_tensors='pt'
)['pixel_values'].to(device)
images.shape
```

Out\[43\]:

```
torch.Size([21, 3, 224, 224])
```

Preprocessing images does _not_ produce token IDs like those we saw from preprocessing our text. Instead, preprocessing images consists of resizing the image to a 244x244 array with three color channels (red, green, and blue) and normalizing pixel values into a \[0,1\]\[0,1\] range.

After preprocessing our images, we get the image features with `get_image_features` and normalize them as before:

In\[44\]:

```python
img_emb = model.get_image_features(images)
print(img_emb.shape)
print(img_emb.min(), img_emb.max())
```

Out\[44\]:

```
torch.Size([21, 512])
tensor(-8.6533, grad_fn=<MinBackward1>) tensor(2.6551, grad_fn=<MaxBackward1>)

```

In\[45\]:

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

Out\[45\]:

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

In\[46\]:

```python
from numpy.linalg import norm

cos_sim = np.dot(text_emb, img_emb.T) / (
    norm(text_emb, axis=1) * norm(img_emb, axis=1)
)
cos_sim.shape
```

Out\[46\]:

```
(21, 21)
```

In\[47\]:

```python
import matplotlib.pyplot as plt

plt.imshow(cos_sim)
plt.show()
```

Out\[47\]:

```
<Figure size 432x288 with 1 Axes>
```

And if we perform the same operation for dot product similarity, we should return the same results:

In\[48\]:

```python
dot_sim = np.dot(text_emb, img_emb.T)

plt.imshow(cos_sim)
plt.show()
```

Out\[48\]:

```
<Figure size 432x288 with 1 Axes>
```

Both of these similarity score arrays look the same, and if we check for the difference between the two arrays, we will see that the scores are the same. We see some slight differences due to floating point errors.

In\[51\]:

```python
diff = cos_sim - dot_sim
diff.min(), diff.max()
```

Out\[51\]:

```
(0.0, 2.9802322e-08)
```

Using the embedding functions of CLIP in this way, we can perform a semantic search across the modalities of text and image in any direction. We can search for images with text, text with images, text with text, and images with images.

These use cases are great, but we can make slight modifications to this for many other tasks.

### Classification

One of the most impressive demonstrations of CLIP is its unparalleled zero-shot performance on various tasks. For example, given the `fragment/imagenette` dataset from Hugging Face _Datasets_, we can write a list of brief sentences that align with the ten class labels.

https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Ff841984e7617686f5041ca95797498e2b0b085b5-1348x542.png&w=3840&q=75

We take the original imagenette labels and preappend "a photo of a ..." to each to create a set of CLIP-friendly sentence representations.

From this, we can calculate the cosine similarity between the text embeddings of these ten labels against an image we’d like to classify. The text that returns the highest similarity is our predicted class.

### Object Detection

Another compelling use case of zero-shot CLIP is object detection. We can do this by splitting our images into smaller patches and running each patch through the image encoder of CLIP. We then compare these patch embeddings to a text encoding describing what we are looking for. After calculating the similarity scores for all patches, we can collate them into a map of relevance.

For example, given an image of a butterfly and a cat, we could break it into many small patches. Given the prompt `"a fluffy cat"`, we will return an outline of the cat, whereas the prompt `"a butterfly"` will produce an outline of the butterfly.

https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fbe4800918976efd9d974d9e5453985a5106f2558-2389x1455.png&w=3840&q=75

Zero-shot object detection with CLIP allows us to find specific objects with natural language prompts.

These are only a few of the use cases of CLIP and only scratch the surface of what is possible with this model and others in the scope of multi-modal ML.

* * *

That’s it for this introduction to multi-modal ML with OpenAI’s CLIP. The past years since the CLIP release have seen ever more fascinating applications of the model.

DALL-E 2 is a well-known example of CLIP. The incredible images generated by DALL-E 2 start by embedding the user’s text prompt with CLIP \[4\]. That text embedding is then passed to the diffusion model, which generates some mind-blowing images.

The fields of NLP and CV have mainly progressed independently of each other for the past decade. However, with the introduction of world scope three models, they’re becoming more entwined into a majestic multi-modal field of Machine Learning.

## Resources

\[1\] Y. Bisk et al., [Experience Grounds Language](https://arxiv.org/abs/2004.10151) (2020), EMNLP

\[2\] J. Alammar, [Experience Grounds Language: Improving language models beyond the world of text](https://www.youtube.com/watch?v=WQm7-X4gts4) (2022), YouTube

\[3\] A. Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (2021), arXiv

\[4\] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, M. Chen, [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125) (2022), arXiv

</details>

<details>
<summary>multimodal-rag-with-colpali-milvus-and-vlms</summary>

# Multimodal RAG with Colpali, Milvus and VLMs

In this post, we will see how to doIn this post, we will see how to do multimodal RAG with [colpali](https://arxiv.org/abs/2407.01449), [milvus](https://milvus.io/) and a visual language model (gemini/gpt-4o).

We will build an application to upload a PDF and then do Q&A queries on it. Q&A can be done on both text and visual elements of the PDF. We will not extract text from the PDF; instead, we will treat it as an image and use colpali to get embeddings for the PDF pages. These embeddings will be indexed to Milvus, and then we will use a VLM to do Q&A queries on the PDF pages.

> If you just want to see the code in action, there is a demo at [https://huggingface.co/spaces/saumitras/colpali-milvus](https://huggingface.co/spaces/saumitras/colpali-milvus/). Code for the same is [here](https://github.com/saumitras/colpali-milvus-multimodal-rag/).

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

Full code at [https://github.com/saumitras/colpali-milvus-rag/](https://github.com/saumitras/colpali-milvus-rag/)

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

        qs.extend(list(torch.unbind(embeddings_query.to(model.device))))

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
<summary>scraping-failed-1</summary>

⚠️ Error scraping https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai/: Request Timeout: Failed to scrape URL as the request timed out. Request timed out - No additional error details provided.

</details>

<details>
<summary>scraping-failed</summary>

⚠️ Error scraping https://towardsdatascience.com/multimodal-embeddings-an-introduction-5dc36975966f/: Failed to parse Firecrawl error response as JSON. Status code: 502

</details>

<details>
<summary>start-with-a-prebuilt-agent</summary>

# LangGraph quickstart

This guide shows you how to set up and use LangGraph's **prebuilt**, **reusable** components, which are designed to help you construct agentic systems quickly and reliably.

## Prerequisites

Before you start this tutorial, ensure you have the following:

- An [Anthropic](https://console.anthropic.com/settings/keys) API key

## 1\. Install dependencies

If you haven't already, install LangGraph and LangChain:

```python
pip install -U langgraph "langchain[anthropic]"
```

LangChain is installed so the agent can call the [model](https://python.langchain.com/docs/integrations/chat/).

## 2\. Create an agent

To create an agent, use `create_react_agent`:

```python
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

## 3\. Configure an LLM

To configure an LLM with specific parameters, such as temperature, use `init_chat_model`:

```python
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

model = init_chat_model(
    "anthropic:claude-3-7-sonnet-latest",
    temperature=0
)

agent = create_react_agent(
    model=model,
    tools=[get_weather],
)
```

For more information on how to configure LLMs, see [Models](https://langchain-ai.github.io/langgraph/agents/models/).

## 4\. Add a custom prompt

Prompts instruct the LLM how to behave. Add one of the following types of prompts:

- **Static**: A string is interpreted as a **system message**.
- **Dynamic**: A list of messages generated at **runtime**, based on input or configuration.

Define a fixed prompt string or list of messages:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    # A static prompt that never changes
    prompt="Never answer questions about the weather."
)

agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

Define a function that returns a message list based on the agent's state and configuration:

```python
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.prebuilt import create_react_agent

def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
    user_name = config["configurable"].get("user_name")
    system_msg = f"You are a helpful assistant. Address the user as {user_name}."
    return [{"role": "system", "content": system_msg}] + state["messages"]

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    prompt=prompt
)

agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config={"configurable": {"user_name": "John Smith"}}
)
```

For more information, see [Context](https://langchain-ai.github.io/langgraph/agents/context/).

## 5\. Add memory

To allow multi-turn conversations with an agent, you need to enable [persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) by providing a `checkpointer` when creating an agent. At runtime, you need to provide a config containing `thread_id` — a unique identifier for the conversation (session):

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    checkpointer=checkpointer
)

# Run the agent
config = {"configurable": {"thread_id": "1"}}
sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config
)
ny_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what about new york?"}]},
    config
)
```

When you enable the checkpointer, it stores agent state at every step in the provided checkpointer database (or in memory, if using `InMemorySaver`).

Note that in the above example, when the agent is invoked the second time with the same `thread_id`, the original message history from the first conversation is automatically included, together with the new user input.

For more information, see [Memory](https://langchain-ai.github.io/langgraph/how-tos/memory/add-memory/).

## 6\. Configure structured output

To produce structured responses conforming to a schema, use the `response_format` parameter. The schema can be defined with a `Pydantic` model or `TypedDict`. The result will be accessible via the `structured_response` field.

```python
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent

class WeatherResponse(BaseModel):
    conditions: str

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    response_format=WeatherResponse
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

response["structured_response"]
```

Structured output requires an additional call to the LLM to format the response according to the schema.

</details>

<details>
<summary>the-8-best-ai-image-generators-in-2025-zapier</summary>

# The 8 best AI image generators in 2025

## Get the best AI-generated images using text-to-image AI.

By Harry Guinness · May 23, 2025

https://images.ctfassets.net/lzny33ho1g45/2olcy4TVSWAjqy5dsxLNZd/09b4a18346af97076615d5f1d1407c39/best-ai-image-generator-hero.jpg?fm=jpg&q=31&fit=thumb&w=1520&h=760

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

All these AI image generators take a text prompt and then turn it—as best they can—into a matching image. This opens up some wild possibilities, since your prompt can be anything from "an impressionist oil painting of a Canadian man riding a moose through a forest of maple trees" to "a painting in the style of Vermeer of a large fluffy Irish wolfhound enjoying a pint of beer in a traditional pub" or "a photograph of a donkey on the moon."

https://images.ctfassets.net/lzny33ho1g45/2udOp4paDgOh5HpqG5JRAQ/18abc9476c4705aacf3609edcec4f945/image8.jpeg?

I made this with Midjourney using the prompt "an impressionist oil painting of a Canadian man riding a moose through a forest of maple trees"

Seriously, the only real limits are your imagination, the AI image generator's ability to [comprehend your prompt](https://zapier.com/blog/natural-language-processing/), and any content filters put in place to stop plagiarism, copyright infringement, and bad actors flooding the internet with AI-generated violence or other NSFW content. (That Vermeer prompt used to work reliably, but some more restrictive image generators now block it because it uses a named artist.)

Most AI image generators work in a pretty similar way. [Millions or billions](https://laion.ai/blog/laion-5b/) of image-text pairs are used to train a neural network (basically, a very fancy computer algorithm [modeled loosely on the human brain](https://news.mit.edu/2017/explained-neural-networks-deep-learning-0414)) on _what things are_. By allowing it to process near-countless images, it learns what dogs, the color red, Vermeers, and everything else are. Once this is done, you have an AI that can interpret almost any prompt—though [there is a skill in setting things up](https://zapier.com/blog/ai-art-prompts/) so it can do so accurately.

The next step is to actually render the AI-generated image. The latest generation of AI image generators typically uses a [process called diffusion](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)—though OpenAI's latest foray into image generation uses a slightly different [process called autoregression](https://arxiv.org/abs/2404.02905). In essence, the image generators start with a random field of noise and then edit it in a series of steps to match their interpretation of the prompt. It's kind of like looking up at a cloudy sky, finding a cloud that looks kind of like a dog, and then being able to snap your fingers to keep making it more and more dog-like.

https://images.ctfassets.net/lzny33ho1g45/1LHdvgxMxOKcgWqC2yzoKh/ff7194426828d81a2d8437f4f9c38132/ai-image-generator-dogs.png?

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

**How to use AI image generation at work**

Interested in AI, but not quite sure how you'd use it at work? Here are a few of the ways people are turning to AI image generation in their roles:

- Generating hero images for blog posts

- Creating social media posts

- Generating slide decks and storyboards

- Creating personalized images for customers


Learn more about [how to use AI image generation at work](https://zapier.com/blog/ai-image-examples-for-business/).

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

### [GPT-4o](https://chat.com/)(ChatGPT)

https://images.ctfassets.net/lzny33ho1g45/75DSS8gsgXORvalbs3MCyE/e5c337007c370f28ba0e27584234c762/image13.jpg?

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

### [**Midjourney**](https://www.midjourney.com/explore?tab=top)

https://images.ctfassets.net/lzny33ho1g45/5c2lxK4vhLWzfata4t1eul/5037e39582914b8b1f4be36d945085e3/image12.jpg?

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

### [**Reve**](https://preview.reve.art/)

https://images.ctfassets.net/lzny33ho1g45/1rErUICKuzBtIoT0x1EmHf/4e9492bc64da35bec554e2eb16f4ca02/image7.jpg?

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

### [Ideogram](https://ideogram.ai/)

https://images.ctfassets.net/lzny33ho1g45/7xaiByWYInfO3qQnxkpn9O/05f734289aa1b0f517b3f43eb74f9680/image15-ideogram.jpg?

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

### [Stable Diffusion](https://stability.ai/)

https://images.ctfassets.net/lzny33ho1g45/4Az7EJ5gtpVyQYpyk3J7AX/5077102edb0e3f841c0d3160aeae1bd0/image3.jpeg?

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

### [**FLUX.1**](https://blackforestlabs.ai/)

https://images.ctfassets.net/lzny33ho1g45/5xAzjYy11xVmiruodsWtSo/d7a171fd60b1ffd639829b40a87e8cc7/image5.jpg?

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

### [Adobe Firefly](https://www.adobe.com/products/firefly/features/text-to-image.html)

https://images.ctfassets.net/lzny33ho1g45/4awQwjmU6tXZ9zR8TvLFCm/3835c17c8b44c9a9396d57e77701fdd5/best-ai-image-generator-image2.jpeg?

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

### [**Recraft**](https://www.recraft.ai/)

https://images.ctfassets.net/lzny33ho1g45/3ZU5phnoABT9vgevnLFlcG/6eb9b110464e7ed161fe54816ef0f78c/image2.png?

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

**Related reading:**

- [The best AI productivity tools](https://zapier.com/blog/best-ai-productivity-tools/)

- [The best AI-powered social media management platforms](https://zapier.com/blog/best-ai-social-media-management/)

- [The best AI photo editors](https://zapier.com/blog/best-ai-photo-editor)

- [The best photo editors for iPhone and Android](https://zapier.com/blog/best-photo-editing-apps-iphone-android/)

- [The best free AI tools](https://zapier.com/blog/free-ai-tools/)


_This article was originally published in March 2023. The most recent update was in May 2025._

</details>

<details>
<summary>understanding-multimodal-llms-by-sebastian-raschka-phd</summary>

# Understanding Multimodal LLMs

### An introduction to the main techniques and latest models

It was a wild two months. There have once again been many developments in AI research, with two Nobel Prizes awarded to AI and several interesting research papers published.

Among others, Meta AI released their latest Llama 3.2 models, which include open-weight versions for the 1B and 3B large language models and two multimodal models.

In this article, I aim to explain how multimodal LLMs function. Additionally, I will review and summarize roughly a dozen other recent multimodal papers and models published in recent weeks (including Llama 3.2) to compare their approaches.

[https://substackcdn.com/image/fetch/$s_!Pq2z!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0d76dab1-362f-45b6-9b12-a12ac131edc5_1600x944.png](https://substackcdn.com/image/fetch/$s_!Pq2z!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0d76dab1-362f-45b6-9b12-a12ac131edc5_1600x944.png) _An illustration of a multimodal LLM that can accept different input modalities (audio, text, images, and videos) and returns text as the output modality._

# 1\. Use cases of multimodal LLMs

What are multimodal LLMs? As hinted at in the introduction, multimodal LLMs are large language models capable of processing multiple types of inputs, where each "modality" refers to a specific type of data—such as text (like in traditional LLMs), sound, images, videos, and more. For simplicity, we will primarily focus on the image modality alongside text inputs.

A classic and intuitive application of multimodal LLMs is image captioning: you provide an input image, and the model generates a description of the image, as shown in the figure below.

[https://substackcdn.com/image/fetch/$s_!8kaL!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F93884822-79f1-498d-a33a-8a367ba57134_1500x1222.png](https://substackcdn.com/image/fetch/$s_!8kaL!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F93884822-79f1-498d-a33a-8a367ba57134_1500x1222.png) _Example use of a multimodal LLM explaining [a meme](https://x.com/PainSci/status/1309570607458086914)._

Of course, there are many other use cases. For example, one of my favorites is extracting information from a PDF table and converting it into LaTeX or Markdown.

# 2\. Common approaches to building multimodal LLMs

There are two main approaches to building multimodal LLMs:

- Method A: Unified Embedding Decoder Architecture approach;
- Method B: Cross-modality Attention Architecture approach.

[https://substackcdn.com/image/fetch/$s_!8miE!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F53956ae8-9cd8-474e-8c10-ef6bddb88164_1600x938.png](https://substackcdn.com/image/fetch/$s_!8miE!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F53956ae8-9cd8-474e-8c10-ef6bddb88164_1600x938.png) _The two main approaches to developing multimodal LLM architectures._

As shown in the figure above, the _**Unified Embedding-Decoder Architecture**_ utilizes a single decoder model, much like an unmodified LLM architecture such as GPT-2 or Llama 3.2. In this approach, images are converted into tokens with the same embedding size as the original text tokens, allowing the LLM to process both text and image input tokens together after concatenation.

The _**Cross-Modality Attention Architecture**_ employs a cross-attention mechanism to integrate image and text embeddings directly within the attention layer.

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

We will revisit the discussion on modeling performance and response quality in a later section, where we will discuss NVIDIA's NVLM paper.

This marks the end of what turned out to be a rather extensive introduction to multimodal LLMs. As I write this, I realize that the discussion has become lengthier than initially planned, which probably makes this a good place to conclude the article.

However, to provide a practical perspective, it would be nice to examine a few recent research papers that implement these approaches. So, we will explore these papers in the remaining sections of this article.

# 4\. Recent multimodal models and methods

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

I hope you found reading this article educational and now have a better understanding of how multimodal LLMs work!

</details>

<details>
<summary>what-are-some-real-world-applications-of-multimodal-ai</summary>

# What are some real-world applications of multimodal AI?

Multimodal AI, which processes and combines different data types like text, images, audio, and sensor inputs, has practical applications across industries. By integrating multiple data sources, these systems improve accuracy and functionality in tasks that require contextual understanding. Below are three key areas where multimodal AI is being applied effectively today.

In healthcare, multimodal AI enhances diagnostics and patient care by merging medical imaging, electronic health records (EHRs), and sensor data. For example, a system might analyze a chest X-ray (image), a patient’s symptom descriptions (text), and vital signs from wearables (sensor data) to detect pneumonia. Models like Google’s **Med-PaLM 2** combine vision and language processing to interpret radiology images alongside clinical notes, reducing misdiagnosis risks. Another use case is monitoring postoperative recovery: wearable devices track movement and heart rate, while speech analysis detects pain or fatigue in a patient’s voice, enabling proactive interventions.

Autonomous vehicles rely heavily on multimodal AI to fuse data from cameras, LiDAR, radar, and GPS. A self-driving car processes road signs (visual data), pedestrian movements (video), and proximity sensor readings to navigate safely. Tesla’s Autopilot, for instance, uses neural networks to combine camera feeds with ultrasonic sensors, improving object detection in varied lighting or weather. Similarly, companies like Waymo train models to correlate map data with real-time sensor inputs, ensuring precise localization and path planning. This redundancy across modalities helps address limitations of single-sensor systems, such as camera failures in low light.

Customer service and content moderation also benefit from multimodal approaches. Virtual assistants like Amazon’s Alexa process voice commands while analyzing user history (text) to personalize responses. In moderation, platforms like YouTube use AI to flag harmful content by scanning video frames (images), audio for hate speech, and user comments (text) simultaneously. For example, a post containing violent imagery and threatening text would be detected faster than if each modality were analyzed separately. Tools like **OpenAI’s CLIP** enable cross-modal matching, such as linking inappropriate images to their descriptive captions, improving accuracy in filtering violations. These systems reduce reliance on manual review while scaling to handle large data volumes.

https://milvus.io/images/demos/multimodal-image-search.png

### Multimodal Image Search

Upload images and edit text to enhance intuitive image searches using advanced retrieval technology.

[Try Demo](https://demos.milvus.io/multimodal-image-search/)

</details>

<details>
<summary>what-are-vision-language-models-nvidia-glossary</summary>

# Vision Language Models

Vision language models (VLMs) are multimodal, generative AI models capable of understanding and processing video, image, and text.

## What Are Vision Language Models?

Vision language models are multimodal AI systems built by combining a large language model (LLM) with a vision encoder, giving the LLM the ability to “see.”

With this ability, VLMs can process and provide advanced understanding of video, image, and text inputs supplied in the prompt to generate text responses.

<Base64-Image-Removed>

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

Any off-the-shelf LLM can be used to build a VLM. There are hundreds of VLM variants that combine various LLMs with vision encoders.

<Base64-Image-Removed>

Figure 2: A common three-part architecture for vision language models

## How Are Vision Language Models Trained?

VLMs are trained in several stages that include pretraining, followed by supervised fine-tuning. Optionally, parameter efficient fine-tuning (PEFT) can be applied as a final stage to create a domain-specific VLM on custom data.

The pretraining stage aligns the vision encoder, projector, and LLM to essentially speak the same language when interpreting the text and image input. This is done using large corpora of text and images with image-caption pairs and interleaved image-text data. Once the three components have been aligned through pretraining, the VLM goes through a supervised fine-tuning stage to help it understand how to respond to user prompts.

The data used in this stage are a blend of example prompts with text and/or image input and the expected response of the model. For example, this data could be prompts telling the model to describe the image or to count all the objects in the frame with the expected correct response. After this round of training, the VLM will understand how to best interpret images and respond to user prompts.

<Base64-Image-Removed>

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

Most benchmarks consist of a set of images with several associated questions, often posed as multiple-choice questions. The multiple-choice format is the easiest way to consistently benchmark and compare VLMs. These questions test the VLMs perception, knowledge, and reasoning capabilities. When running these benchmarks, the VLM is provided with the image, question, and several multiple-choice answers it must choose from.

<Base64-Image-Removed>

Figure 4: Example multiple-choice questions for VLMs used in the MMMU benchmark

Source ( [MMMU](https://mmmu-benchmark.github.io/))

The accuracy of the VLM is the number of correct choices over the set of multiple-choice questions. Some benchmarks also include numerical questions where the VLM must perform a specific calculation and be within a certain percentage of the answer to be considered correct. Often these questions and images come from academic sources, such as college-level textbooks.

## How Are Vision Language Models Used?

VLMs are quickly becoming the go-to tool for all types of vision-related tasks due to their flexibility and natural language understanding. VLMs can be easily instructed to perform a wide variety of tasks through natural language:

1. Visual questions-answering
2. Image and video summarization
3. Parsing text and handwritten documents

Previous applications that would have required a large ensemble of specially trained models can now be accomplished with just a single VLM.

VLMs are especially good at summarizing the contents of images and can be prompted to perform specific tasks based on the contents. Take for example, an education use case—a VLM could be given an image of a handwritten math problem, and it could use its optical character recognition and reasoning capabilities to interpret the problem and produce a step-by-step guide on how to solve it. VLMs can not only understand the content of the image but also reason and perform specific tasks.

<Base64-Image-Removed>

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

# What Is Optical Character Recognition (OCR)?

https://blog.roboflow.com/content/images/size/w1200/2024/04/image-730.webp

Have you ever wondered how a computer can understand the words on a photo, just like you do?  That's where Optical Character Recognition, or [OCR](https://roboflow.com/ocr?ref=blog.roboflow.com), steps in. OCR takes the text you see in images – be it from a book, a receipt, or an old letter – and turns it into something your computer can read, edit, and search.

OCR finds widespread applications in tasks such as automated data entry, document digitization, text extraction from images, invoice processing, form recognition, and enhancing accessibility for visually impaired individuals.

Let's explore the fundamentals of OCR, understanding its workings, the challenges it addresses, and why it remains a crucial component of present and future technology.

## What Is Optical Character Recognition?

Optical Character Recognition (OCR) involves converting both handwritten and typed text from various sources, including images, videos, and scanned documents like PDFs, into a digitally editable format.

The output from OCR can be used by a computer to make decisions. Common use cases of OCR include:

Using OCR to read product identifiers on an assembly line. When each identifier is read, a piece of software can update an inventory tracking system to note the package with the given identifier has arrived.

Using OCR for scanned document recognition. This involves scanning printed documents, after which OCR software converts them into searchable and editable text. This method is widely employed to automate the handling of legal documents, extract data from bank statements and invoices, and streamline tasks like invoice processing and financial record-keeping.

Using OCR for “scene text recognition”, wherein an OCR system recognizes text from natural scenes, such as street signs, storefronts, or license plates.

Using OCR for alphanumeric, printed text, such as text that was written on a typewriter, or text that was printed out. But, you can also use OCR on handwriting. This usually involves using a separate system due to the differences in handwriting compared to printed text.

https://blog.roboflow.com/content/images/2024/04/image-733.webp_Application of OCR on the text of a book._ [_Source_](https://www.edenai.co/post/optical-character-recognition-ocr-which-solution-to-choose?ref=blog.roboflow.com).

## How Optical Character Recognition Works

Let's discuss the typical steps modern OCR software uses to read text:

1. **Image pre-processing**: After an image has been collected, the image undergoes pre-processing to enhance image quality, improving recognition. Pre-processing may involve resizing, contrast enhancement, binarization, noise reduction, and other techniques.
2. **Text Detection**: Using a specialized deep-learning model trained on large datasets of images and text, the computer vision model detects regions in the input image that likely contain text. This process is usually a crucial step.
3. **Layout Analysis**: After detecting text regions, the computer vision model conducts layout analysis to determine the structure and order of the text in the image. This step ensures the preservation of context and organizes the output for readability, but is not run by all OCR systems.
4. **Text Recognition**: Detected text regions pass through a deep learning-based text recognition model, utilizing a combination of convolutional neural networks (CNNs) and recurrent neural networks (RNNs). This model recognizes individual characters and words in the input image, converting them into machine-readable text.
5. **Language Model**: The final output undergoes post-processing to remove noise, correct spelling mistakes, and enhance overall accuracy. The predicted sequence of characters may contain errors, especially for long or uncommon words. Language models, acting as word processors, refine the output by predicting the probability of a sequence of words based on the input image. Statistical models and advanced methods, including deep learning, may be employed for this purpose.

https://blog.roboflow.com/content/images/2024/04/image-738.webp_An example OCR system pipeline._

Having acquired an understanding of how OCR operates, let's examine its algorithms and investigate their operational mechanisms, covering the old and the new.

## Traditional Approaches to OCR

The first OCR algorithms rooted in image processing were typically rule-based systems. One well-known OCR that uses this approach is [Tesseract](https://github.com/tesseract-ocr/tesseract?ref=blog.roboflow.com). These systems relied on manually crafted features and heuristic rules to identify characters within images. The approach involved segmenting characters into individual units and applying a set of rules for character classification.

However, the accuracy and performance of these algorithms were often constrained due to the intricate process of developing and fine-tuning the necessary handcrafted features and rules for effective recognition.

### Tesseract

Tesseract, an open-source optical character recognition engine, originated at Hewlett-Packard Laboratories in the 1980s and subsequently became open-source in 2005.

Initially designed to recognize English text exclusively, Tesseract has evolved into a versatile OCR engine. Working from traditional image processing principles, which involves manual logic unlike the deep learning processes in modern systems, Tesseract analyzes images to identify patterns for character recognition.

First, Tesseract preprocesses the image to enhance input quality, a step which encompasses tasks like contrast improvement and noise removal. Following this, Tesseract employs feature extraction techniques, including edge detection and pattern recognition, to identify and recognize characters.

https://blog.roboflow.com/content/images/2024/04/image-741.webp_Tesseract OCR engine pipeline._ [_Source_](https://www.researchgate.net/figure/Tesseract-OCR-engine-architecture_fig4_265087843?ref=blog.roboflow.com).

## Deep Learning Approaches to Optical Character Recognition

With the rise of deep learning, the integration of neural networks into OCR systems has gained substantial popularity. In particular, deep learning methodologies like [Convolutional Neural Networks](https://blog.roboflow.com/what-is-a-convolutional-neural-network/) and Long Short-Term Memory networks are leveraged, for precise text recognition. Neural networks regularly achieve better performance than traditional OCR techniques.

In recent years, there has also been a surge in novel approaches that leverage pre-trained image and text [Transformers](https://blog.roboflow.com/what-is-a-transformer/), a deep learning architecture. Transformers are ushering in a new era of end-to-end optical word recognition.

### PaddleOCR

[Paddle OCR](https://arxiv.org/abs/2009.09941?ref=blog.roboflow.com) is an open-source engine developed by Baidu's PaddlePaddle team. Leveraging deep learning techniques, including CNNs and recurrent neural networks, Paddle OCR excels in accurate text recognition. It comprises two key components: the detector and the extractor. The detector is tasked with pinpointing text within an image or document. It employs various algorithms, such as [EAST (Efficient and Accurate Scene Text)](https://paperswithcode.com/paper/east-an-efficient-and-accurate-scene-text?ref=blog.roboflow.com) or [DB (Differentiable Binarization)](https://arxiv.org/abs/1911.08947?ref=blog.roboflow.com) detectors, to identify text regions.

https://blog.roboflow.com/content/images/2024/04/image-745.webp_DB (Differentiable Binarization) architecture._ [_Source_](https://arxiv.org/pdf/2009.09941.pdf?ref=blog.roboflow.com).

After the detector locates the text, the extractor comes into play, retrieving the text from the image. It employs a blend of Convolutional Neural Networks and Recurrent Neural Networks for precise text recognition. CNNs are utilized to extract features from the text, while RNNs play a crucial role in recognizing the sequence of characters.

https://blog.roboflow.com/content/images/2024/04/image-748.webp_CRNN Extractor architecture._ [_Source_](https://arxiv.org/pdf/1507.05717.pdf?ref=blog.roboflow.com).

Paddle OCR stands out for its remarkable speed, making it among the swiftest OCR engines. Its efficiency is attributed to the utilization of parallel computing and GPU acceleration. This feature renders it particularly suitable for extensive OCR tasks, including document scanning and image recognition. Moreover, its adaptability shines through as it can be tailored and fine-tuned for specific tasks and datasets, enhancing its versatility and robustness in various OCR applications.

### TrOCR

[Transformer-based Optical Character Recognition (TrOCR)](https://arxiv.org/abs/2109.10282?ref=blog.roboflow.com) is one of many transformer-based [OCR models](https://blog.roboflow.com/best-ocr-models-text-recognition/). In contrast to traditional OCR systems, TrOCR adopts a methodology where both input image processing and the generation of corresponding text output occur within a single model.

The encoder segment of TrOCR employs a transformer-based architecture to handle the input image, segmenting it into a grid of patches and extracting visual features from each patch. Simultaneously, the decoder component utilizes a transformer-based model to produce the relevant text output, incorporating the visual features extracted from the image.

https://blog.roboflow.com/content/images/2024/04/image-752.webp_TrOCR Architecture._ [_Source_](https://arxiv.org/pdf/2109.10282.pdf?ref=blog.roboflow.com).

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
