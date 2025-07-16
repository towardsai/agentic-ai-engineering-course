# Research based on provided article guidelines

## Research Results

---

<details>
<summary>What are the fundamental concepts, architectures, and current leading models in multimodal large language models (LLMs) and vision-language models (VLMs)?</summary>

### Source: https://magazine.sebastianraschka.com/p/understanding-multimodal-llms
**Architectures of Multimodal LLMs:**
- **Unified Embedding Decoder Architecture:** This approach uses a **single decoder model**, similar to traditional LLMs (like GPT-2 or Llama 3). Images are converted to tokens matching the size of text tokens, allowing both to be processed together after concatenation.
- **Cross-Modality Attention Architecture:** This method **integrates image and text embeddings directly within the attention layer** using a cross-attention mechanism.

These two approaches represent the main architectural strategies for fusing different data modalities within LLMs.

-----

### Source: https://machinelearning.apple.com/research/grounding-multimodal-large
**Grounding and Action Spaces:**
- **Grounding Multimodal LLMs:** The study explores how to ground multimodal LLMs in various **embodiments and action spaces**, leveraging their multimodal world knowledge.
- For **continuous actions**, a **learned tokenization** of the action space yields strong downstream performance.
- For **discrete actions**, semantically aligning them with the native output token space of the LLM gives optimal results.
- The research generalizes methods through a **unified architecture** and uses **action space adaptors** to connect models to different environments, demonstrating effective strategies across over 114 embodied tasks.

-----

</details>

---

<details>
<summary>What are the principles and real-world use cases of multimodal embedding models (e.g., CLIP), and how are they trained to align multiple data modalities?</summary>

### Source: https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/onnx-pipeline-models-multi-modal-embedding.html
**CLIP multi-modal embedding models** are designed to accept both image and text inputs, generating embeddings for each modality. The core principle is to enable **image-text similarity** computations, allowing users to determine which text best describes a given image by comparing their respective embeddings. The architecture comprises two specialized pipelines: an image embedding pipeline and a text embedding pipeline. Each pipeline performs the necessary pre-processing for its input type and produces vectors (embeddings) that can be directly compared for similarity. For practical usage, two ONNX pipeline models are generated—one for images (suffixed with *_img*) and one for text (suffixed with *_txt*). Both are used together during inference to perform cross-modal similarity tasks. This setup is particularly useful for applications like image search by text query, automatic captioning, and semantic image retrieval.

-----

### Source: https://openai.com/index/clip/
**CLIP** (Contrastive Language–Image Pre-Training) is a neural network that learns to associate images and text by leveraging **natural language supervision**. The model is trained on large-scale collections of image-text pairs sourced from the internet. The training objective is a **contrastive learning task**: given an image, the model predicts which text from a large pool was actually paired with it, and vice versa. This approach compels the model to learn rich visual concepts and their textual descriptions, resulting in embeddings where semantically similar images and texts are close together in the shared space.

CLIP’s principles include:
- **Scalability**: Training on vast internet-scale datasets rather than costly, manually curated labels.
- **Zero-shot generalization**: After training, CLIP can be applied to new image or text classification tasks without additional supervised fine-tuning, simply by comparing similarity between embeddings.
- **Unified representation**: Both images and texts are embedded into a common space, enabling direct cross-modal comparisons.

CLIP enables real-world applications such as image search, content moderation, automatic captioning, and open-domain visual classification—tasks where the alignment of text and image data is essential.

</details>

---

<details>
<summary>What is the ColPali architecture in multimodal retrieval-augmented generation (RAG), and how does it differ from and improve upon standard document retrieval techniques?</summary>

### Source: https://huggingface.co/learn/cookbook/en/multimodal_rag_using_document_retrieval_and_vlms
The ColPali architecture is integrated into a Multimodal Retrieval-Augmented Generation (RAG) system by serving as the **retriever** for document retrieval. In this configuration, ColPali is paired with a Vision Language Model (VLM), specifically Qwen2-VL, to create a system capable of enhancing query responses using both **text-based documents and visual data**. This setup eliminates the need for a complex document processing pipeline that typically relies on Optical Character Recognition (OCR) to extract data. Instead, ColPali uses a Document Retrieval Model to efficiently fetch relevant documents directly based on the user query, streamlining the process and improving efficiency when handling documents that contain both textual and visual information.

-----

### Source: https://arxiv.org/html/2407.01449v2
ColPali is described as a **novel retrieval model** that leverages state-of-the-art generative Vision Language Models (VLMs) to produce **multi-vector embeddings** from visual document features alone. Unlike traditional pipelines, ColPali operates **purely from visual features** without relying on text extraction via OCR. The architecture is **end-to-end trainable** and demonstrates **superior performance** compared to the best existing document retrieval methods, offering both faster corpus indexing and lower query latency. ColPali's ability to embed visual document content directly enables it to outperform both industrial document retrieval pipelines and conventional image-text contrastive models, particularly in retrieving visually rich documents where layout, images, and graphical elements are essential.

-----

### Source: https://blog.vespa.ai/Transforming-the-Future-of-Information-Retrieval-with-ColPali/
ColPali addresses the limitations of standard document retrieval techniques, which typically focus only on text and are "blind" to **visual elements** such as images, tables, and layouts—features often crucial for comprehension in fields like healthcare and finance. Standard pipelines require multiple **preprocessing steps**: text extraction, OCR, and layout analysis. This process can be both time-consuming and error-prone, especially for PDFs or documents where visual features are key.

ColPali, through **Contextualized Late Interaction over PaliGemma**, streamlines retrieval by **embedding the entire rendered document**—including visual elements—into optimized vector representations. This approach eliminates the need for elaborate preprocessing and enables retrieval systems to consider both textual and visual content, thereby **improving relevance, accuracy, and contextual understanding** in information retrieval for complex, visually rich documents.

-----

### Source: https://openreview.net/pdf/ade1cfbcb424ff6033378ecce0808d70afc037dc.pdf
The ColPali architecture utilizes a **specialized text-image retrieval model** that is fine-tuned with a custom dataset to enhance performance for multimodal RAG. The model focuses on generating **high-quality contextual embeddings** from document images. Fine-tuning is performed to optimize the relationships between **text tokens and image patches**, and retrieval precision is further increased by a **late-stage interaction matching mechanism**. This architectural choice enables the system to **bypass OCR and layout analysis**, contributing to faster indexing by processing document images directly. The fine-tuned ColPali model demonstrates strong retrieval performance for **visually rich content** and supports multiple domains and languages, making it a robust solution for advanced multimodal retrieval tasks.

-----

</details>

---

<details>
<summary>How are multimodal AI agents built by integrating LLMs, multimodal embeddings, and retrieval systems, and what are current best practices for practical deployment?</summary>

### Source: https://blog.premai.io/multimodal-llms-architecture-techniques-and-use-cases/
**Multimodal LLMs** are advanced AI systems capable of processing and generating content across multiple data types, or modalities. The main architectural approaches for integrating LLMs with multimodal embeddings are:

- **Unified Embedding Decoder Architecture:** Here, visual inputs are transformed into embedding vectors sharing the same dimensions as text tokens. These embeddings are concatenated and processed by a single decoder model, as seen in Llama 3.2 and GPT-2. This enables seamless handling of different modalities within the same model pipeline.
- **Cross-Modality Attention Architecture:** This approach uses a cross-attention mechanism where visual and textual embeddings interact in the multi-head attention layer, directly connecting image patches to text and enhancing integration between modalities. This is inspired by the original Transformer architecture.
- **Training Strategies:** Multimodal LLMs are typically pretrained by augmenting a language model with visual encoders and aligning them using adapters. Fine-tuning adapts the full model for tasks such as image captioning or visual question answering. Parameter-efficient fine-tuning methods like **LoRA** and **QLoRA** are commonly used to reduce computational overhead, allowing effective adaptation without retraining the entire model.

These strategies collectively enable the flexible and efficient integration of LLMs, multimodal embeddings, and retrieval systems for practical deployment across various application scenarios.

-----

### Source: https://dev.to/mongodb/building-multimodal-ai-applications-with-mongodb-voyage-ai-and-gemini-49g3
**Multimodality** in AI refers to the ability of models to process, understand, and sometimes generate different types of data—such as text, images, audio, and video. Recent advances, including models like Gemini and GPT-4o, can now handle multiple modalities as both input and output. Vendors like Voyage AI and Cohere offer **multimodal embedding models** that map diverse data types into a shared high-dimensional vector space.

For practical deployment:
- **Multimodal embedding models** allow text, images, and tables to be mapped into a common space, enabling retrieval-augmented generation (RAG) applications that use documents with interleaved modalities.
- When building such systems, it is crucial to verify which modalities are supported by the chosen models, as not all support every data type.
- The typical workflow involves storing these multimodal embeddings in a vector store (such as MongoDB), allowing efficient retrieval and downstream processing by LLMs capable of handling the relevant modalities.

Best practices include evaluating embedding model performance for your use case and ensuring seamless integration between the embedding, retrieval, and generative components.

-----

### Source: https://arxiv.org/html/2406.05496v1
**Modularity** is a core principle in building multimodal AI agents. In practice, this involves:

- Constructing architectures from distinct modules, each responsible for a specific function (e.g., vision encoder, language model, adapters).
- Aligning pretrained vision encoders with LLMs using adapter modules. This modular setup enhances flexibility, allowing for efficient adaptation to new tasks and computational efficiency.
- The **projector module** transforms original embeddings (e.g., from images) into the language space, enabling the LLM to process and reason over multimodal inputs.
- Modularity allows for easy integration and replacement of components, supporting practical deployment and adaptability to evolving requirements.

This architecture design has become prominent in state-of-the-art vision-language models, enabling robust performance on multimodal tasks while maintaining efficiency and scalability.

</details>

---

<details>
<summary>Detailed, step-by-step code examples showing Gemini's multimodal capabilities with images and PDFs, covering raw bytes, Base64, and URLs—including any current limitations in API support</summary>

### Source: https://ai.google.dev/gemini-api/docs/image-understanding
The **Gemini API documentation for image understanding** outlines the model’s multimodal image processing features, including **image captioning, classification, and visual question answering**. For advanced tasks (like segmentation in Gemini 2.5), the API returns a **JSON list** for each detected object, including its bounding box and a **Base64-encoded PNG segmentation mask** (probability map). The bounding box uses normalized coordinates, and the segmentation mask is provided as a Base64-encoded PNG that needs to be resized and binarized for further processing. This strongly indicates that the API both outputs and likely supports **Base64-encoded image data**. While segmentation output is described in detail, the documentation here does not explicitly show step-by-step code for inputting images as raw bytes, Base64, or URLs, but the mention of Base64 in outputs suggests at least partial support for this format.

-----

</details>

---

<details>
<summary>Modern benchmark datasets and evaluation metrics for vision-language models and multimodal document processing (especially for documents with complex layouts, diagrams, or tables)</summary>

### Source: https://aclanthology.org/2024.acl-long.420.pdf
**EXAMS-V** is a novel **multimodal, multilingual benchmark** designed to assess vision-language models’ ability to reason over complex documents. Key features include:
- Unified document snapshots that integrate **text, images, tables, graphs, and more**
- **20,932 questions** spanning **11 languages** and **20 academic subjects**
- Tasks requiring reasoning across multiple modalities and structured representations

EXAMS-V is intended to reflect real-world document complexity and diversity, making it particularly relevant for evaluating VLMs on documents with complex layouts, diagrams, or tables. The dataset sets a new standard for challenging, realistic evaluation of vision-language reasoning in academic and professional document contexts.

-----

</details>

---

<details>
<summary>In-depth explanation and visual diagrams illustrating the ColPali architecture: its workflow, innovations like bag-of-embeddings, chunking vs. patching, and reranking functions</summary>

### Source: https://qdrant.tech/blog/qdrant-colpali/
ColPali processes an entire **document as an image** rather than relying on traditional OCR methods. It uses a **Vision Encoder** to convert the image into **multi-vector embeddings** that capture both the text and the visual layout of the document. These embeddings are then passed through a **Large Language Model (LLM)** to create a unified representation retaining textual and visual information.

**Workflow:**
- The input image is split into a **32x32 grid** (1,024 patches).
- Each patch is transformed to capture both local and global context, producing a **128-dimensional vector** for each patch.
- When a user submits a query, ColPali generates **token-level embeddings** for the query.
- The system computes a **similarity matrix** (using the MaxSim function) between the query tokens and document patches. For each query token, the maximum similarity with any patch is selected.
- This **late interaction** technique allows ColPali to effectively match the query to relevant document areas, considering both textual and layout information.

ColPali’s architecture is inspired by **ColBERT** but extends it by combining visual and textual features in a single model. Its late interaction approach enables detailed context capture without the need for multiple passes over the document.

-----

### Source: https://learnopencv.com/multimodal-rag-with-colpali/
ColPali introduces a **bag-of-embeddings** approach, where each document page is treated as an image and encoded into a set of **128-dimensional embeddings** (similar to ColBERT’s method). These embeddings represent both text and visual content and are indexed in vector databases that support multi-vector search.

**Indexing Innovations:**
- ColPali achieves **much faster offline indexing** compared to traditional PDF parsers (0.37s vs. 7.22s per page).

**Online Querying:**
- At inference, the input query is encoded by the language model.
- A **late interaction mechanism**, as in ColBERT, computes the **maximum similarity** between each query token and the pre-indexed document embeddings.
- ColPali then retrieves the top-k most similar document pages as images, which can be passed to a multimodal LLM (like Gemini) for further analysis or response.
- While ColPali’s online encoding is somewhat slower than standard single-vector retrievers for short queries, it enables much richer matching by leveraging multi-vector representations.

-----

### Source: https://milvus.io/docs/use_ColPali_with_milvus.md
ColPali blends **ColBERT’s multi-vector representation** with PaliGemma’s multimodal LLM capabilities, allowing each document (including images, figures, and tables) to be represented as a **unified multi-vector embedding**. This approach is particularly effective for **retrieval-augmented generation (RAG)** in multimodal contexts.

**MaxSim Function:**
- For each word in a query, MaxSim finds the most similar word (or patch) in the document using a similarity metric (cosine similarity or squared L2 distance).
- The sum of these maximum similarities across all query tokens provides the final similarity score for retrieval.

This method enables **fine-grained retrieval** based on both textual and visual cues, significantly enhancing performance over single-vector methods.

-----

### Source: https://huggingface.co/blog/fsommers/document-similarity-colpali
ColPali’s **document embeddings** are generated in an **offline indexing phase**, where each document is encoded as a **bag of embeddings** (one for each token or patch, depending on the model). At retrieval time, a ColBERT-style method is used: the query is represented as a set of embeddings, and similarity is computed via late interaction (MaxSim).

**Bag-of-Embeddings Illustration:**
- Each document and query is represented as a set (bag) of embedding vectors.
- During retrieval, for each embedding in the query, the highest similarity to any embedding in the document is found, and these max similarities are aggregated to score and rank results.

This allows for **precise matching** at the sub-document (token or patch) level, capturing nuanced similarities between query and document content.

-----

### Source: https://arxiv.org/html/2407.01449v2
ColPali’s architecture allows scaling by adjusting the number of **image patches** the model uses. The standard model uses **1,024 patches** (32x32 grid), but variants with fewer (512) or more (2,048) patches have been tested. Fewer patches reduce memory usage but also decrease retrieval accuracy, while more patches improve performance at the cost of much higher memory requirements.

This scalability highlights a **trade-off between efficiency and retrieval quality**, allowing users to tailor the model based on available resources and application needs. The system is also **end-to-end trainable**, further distinguishing it from traditional pipelines that separate layout analysis, OCR, and retrieval.

-----

</details>

---

<details>
<summary>Best practices and recent design patterns for integrating multimodal RAG (using ColPali or similar) and retrieval functions as tools in agent frameworks such as LangGraph, with reference code if available</summary>

### Source: https://learnopencv.com/multimodal-rag-with-colpali/
ColPali's multimodal RAG approach focuses on **efficient retrieval of complex document elements**—including images, tables, and charts—by representing each page as an image. The process involves:

- **Retrieval:** Using `retrieve_top_document()` to select the most relevant image based on query and embeddings.
- **Answer Generation:** Passing the best-matched image and a prompt to a vision-capable LLM (e.g., Gemini Flash) using the `get_answer()` function, which accepts a prompt and image, and generates a response.
- **Prompt Engineering:** Detailed instructions in prompts improve analysis and interpretation, ensuring the LLM provides a clear, context-aware summary.

**Sample code:**
```python
def get_answer(prompt: str, image: Image):
    response = model.generate_content([prompt, image])
    return response.text

def answer_query(query: str, prompt):
    best_image, best_index = retrieve_top_document(
        query=query, document_embeddings=document_embeddings, document_images=images)
    answer = f"Gemini Response\n: {get_answer(prompt, best_image)}"
    return answer, best_image, best_index
```
**Best Practices:**
- **Treat pages as holistic visual units** rather than extracting and mixing modalities separately.
- **Leverage vision-capable LLMs** for direct multimodal input.
- **Design prompts** to clarify expected multimodal reasoning.

This pattern enables robust retrieval and reasoning over visually complex documents, especially when integrating ColPali within agentic frameworks or toolchains.

-----

### Source: https://decodingml.substack.com/p/the-king-of-multi-modal-rag-colpali
A ColPali-based multimodal RAG system comprises five main components:

- **PDF to Image Converter:** Uses pdf2image with high DPI to preserve the original document's visual layout, crucial for retaining context in tables and diagrams.
- **Storage Service:** Manages and serves the converted images for downstream retrieval.
- **ColPali Model:** Generates visual embeddings that capture both spatial and semantic information from the entire document page.
- **Vector Database (e.g., Qdrant):** Stores embeddings, enables fast similarity search, and supports metadata for page-level retrieval.
- **Multimodal LLM (e.g., Claude Sonnet 3.7):** Interprets images in context, generating natural language responses that reference visual elements.

**Best practices:**
- **Maintain the document's visual context** by working with images, not just OCR-extracted text.
- **Store and index images alongside metadata** (e.g., page numbers, sections) in a vector database.
- **Use multimodal LLMs** to interpret and respond to queries, ensuring references to tables, charts, and formatting are retained.

This architecture is well-suited for integration into agent frameworks (such as LangGraph), where each component can be a modular tool—retrieval as one tool, vision LLM as another—coordinated by the agent based on the workflow and user queries.

-----

### Source: https://www.together.ai/blog/multimodal-document-rag-with-llama-3-2-vision-and-colqwen2
ColPali enables **direct indexing and embedding of document pages as images**, eliminating the need for complex, multi-stage extraction pipelines involving OCR and metadata enrichment. This approach supports:

- **Flexible, robust multimodal RAG**: AI systems can reason over images of documents, allowing for richer context and more accurate retrieval and generation, especially for visually structured content (e.g., tables, charts).
- **Integration with vision-enabled LLMs**: Combines ColPali embeddings with models such as Llama 3.2 Vision, allowing the agent to process both the user query and retrieved document images in a single, cohesive step.

This design pattern is particularly effective for agentic frameworks, where tools for retrieval, embedding, and answer generation can be orchestrated dynamically, adapting to the modality and complexity of user queries.

-----

### Source: https://huggingface.co/learn/cookbook/en/multimodal_rag_using_document_retrieval_and_vlms
This HuggingFace notebook demonstrates a complete **multimodal RAG pipeline** using ColPali for retrieval and vision-language models (VLMs) for answer generation.

**Pipeline steps:**
- **Document Preprocessing:** Convert documents (e.g., PDFs) to images, preserving all visual elements.
- **Embedding Generation:** Use ColPali to generate visual embeddings from the images.
- **Indexing:** Store embeddings in a searchable vector database.
- **Querying:** For a user query, retrieve the top-matching image(s) using vector similarity search.
- **Answer Generation:** Pass both the query and the retrieved image(s) to a vision-language model to obtain a context-aware answer.

**Design patterns and best practices:**
- **Holistic retrieval:** Always retrieve at the page/image level rather than extracting and mixing modalities separately.
- **Flexible orchestration:** The modular approach allows easy integration as tools within agent frameworks, where each step (retrieval, embedding, answer generation) can be invoked as a callable tool.
- **Efficient prompt design:** Ensure prompts clearly specify the context and the desired reasoning (e.g., "explain what is shown in this image, referencing any tables or charts").

**Reference code** is provided in the notebook for each step, including sample functions for retrieval and answer generation, which can be adapted into agent tool abstractions.

-----

</details>

---

<details>
<summary>Recent hands-on coding tutorials or official documentation demonstrating Gemini’s multimodal capabilities for images and PDFs, specifically showing raw bytes and Base64 image input (and output), including step-by-step code and known API limitations as of 2024.</summary>

### Source: https://developers.googleblog.com/en/gemini-2-0-level-up-your-apps-with-real-time-multimodal-interactions/
Gemini 2.0 introduces the **Multimodal Live API**, which enables real-time, multimodal interactions using **WebSockets** for low-latency, server-to-server communication. This API supports integration of **visual inputs** alongside text, audio, and video, allowing for responses that combine different modalities without multiple prompts. Developers can utilize this API in Google AI Studio and via the Gemini API to build applications that respond to real-time data.

Key points about multimodal capabilities:
- Allows **real-time streaming** and stateful interactions.
- Supports **function calling, code execution, search grounding, and combining multiple tools** within a single request.
- Designed for efficient, complex AI workflows that require multimodal understanding.

While the blog post highlights the API’s potential for real-time multimodal input and output (including images and video), it does not provide step-by-step code or explicit handling of **raw bytes or Base64 image input/output**. Instead, it refers to demo applications and encourages developers to explore streaming capabilities through these demos.

**API limitations or data encoding specifics** (such as inputting images as raw bytes or Base64) are not detailed in this post; developers are directed to the official documentation and demo applications for code samples and technical specifics.

-----

</details>

---

<details>
<summary>Comprehensive benchmark studies and quantitative comparisons (2023–2025) between traditional OCR engines (e.g., Google Vision, Amazon Textract) and state-of-the-art multimodal LLMs/VLMs (e.g., Gemini, GPT-4o, LLaVA, ColPali) for extraction accuracy, speed, and error rates on complex real-world documents (tables, handwriting, diagrams).</summary>

### Source: https://encord.com/blog/gpt-vision-vs-llava/
**Comparison of GPT-4 Vision and LLaVA on OCR and Document Understanding Tasks**

- **Handwritten Text:** GPT-4 demonstrates superior proficiency in extracting handwritten text, with only two minor errors observed. LLaVA, in contrast, faces significant challenges with handwriting and is less reliable in this area.
- **Rotated and Overlapped Text:** LLaVA struggles with text rotated beyond 90 degrees and fails to read overlapped text effectively. Both models have difficulty with overlapped text, but GPT-4 generally outperforms LLaVA.
- **Mathematical OCR:** GPT-4 accurately interprets mathematical equations, performs calculations, and provides detailed step-by-step reasoning, indicating high OCR and reasoning capability. LLaVA cannot reliably comprehend or process mathematical content in images.
- **Complex Documents:** In tasks involving real-world documents such as invoices, GPT-4 accurately extracts relevant information and answers related questions with precision. LLaVA is prone to providing incorrect or misleading answers in similar scenarios.
- **General VQA (Visual Question Answering):** Both models are competent in understanding and answering questions about general images (e.g., paintings, memes), but LLaVA underperforms when OCR is required for document-based questions.
- **Self-awareness and Recommendations:** LLaVA acknowledges its limitations and may suggest ways to improve performance, but this does not compensate for its lower extraction accuracy compared to GPT-4.

In summary, GPT-4 consistently outperforms LLaVA in extraction accuracy and error rates on complex, real-world documents, especially for handwriting, mathematical content, and structured data, while LLaVA is more limited in these areas.

-----

### Source: https://arxiv.org/html/2410.10594v1
**VisRAG: Vision-based Retrieval-augmented Generation and Benchmarking with State-of-the-Art Multimodal LLMs**

- **Model-based Parsing:** The study benchmarks MiniCPM-V 2.0, a vision-language model, for direct transcription of document images, trained on datasets including ALLaVA and VQA collections with GPT-4V-generated descriptions. This model is designed to evaluate end-to-end document understanding, including complex layouts, charts, and infographics.
- **GPT-4o Capabilities:** GPT-4o is highlighted as OpenAI’s latest multimodal model, capable of processing text, audio, image, and video inputs and generating outputs in various formats. It is used in the VisRAG-Gen pipeline and for synthesizing training data.
- **Benchmark Design:** The benchmarks involve complex document datasets (DocVQA, ChartQA, SlideVQA, InfographicsVQA, TextVQA, ArxivQA), covering tables, diagrams, and diverse layouts, thus enabling quantitative comparisons between state-of-the-art VLMs and OCR-based pipelines.
- **Implementation Details:** The models are fine-tuned on high-capacity hardware (8× NVIDIA A100 80GB GPUs), enabling direct and detailed evaluation of extraction accuracy and speed for large-scale, real-world documents.

While specific extraction accuracy, speed, and error rate numbers are not presented in the excerpt, the benchmarking setup is designed to quantitatively compare traditional OCR pipelines (using extracted text) with modern VLMs/LLMs (such as GPT-4o and MiniCPM-V 2.0) on complex document understanding and extraction tasks.

-----

### Source: https://arxiv.org/html/2504.09249v1
**NoTeS-Bank: Benchmarking Neural Transcription and Search for Handwritten and Complex Documents**

- **Benchmark Focus:** NoTeS-Bank is a large-scale benchmark for evaluating neural transcription (OCR) and search capabilities, with targeted comparisons among models like GPT-4o and other foundation models.
- **Evaluation Protocol:** The protocol aims for rigorous, quantitative analysis, moving beyond qualitative assessments to directly compare extraction accuracy, speed, and error rates on complex visual tasks, including handwritten, tabular, and diagram-rich documents.
- **Findings (partial):** The paper indicates that GPT-4o demonstrates relatively strong performance in multimodal tasks, suggesting its superiority over traditional and earlier models for certain challenging document types.

While the excerpt does not provide detailed numeric results, it affirms that recent benchmarks directly compare state-of-the-art multimodal LLMs/VLMs to traditional OCR systems on real-world complex documents, focusing on extraction accuracy, speed, and error rates.

-----

</details>

---

<details>
<summary>Peer-reviewed papers, blog posts, or official visual diagrams that provide a step-by-step, architectural overview of ColPali’s bag-of-embeddings workflow (chunking vs patching, MaxSim/late interaction, offline indexing/online query flow), including at least one high-resolution or code-generatable diagram for educational re-use.</summary>

### Source: https://qdrant.tech/blog/qdrant-colpali/
ColPali replaces traditional OCR with a **vision encoder** that processes the entire document as an image, creating **multi-vector embeddings** that capture both textual and visual features. The workflow is as follows:

- **Image Preprocessing:** The input image is divided into a **32x32 grid** (1,024 patches).
- **Patch Embedding:** Each patch is transformed into a **128-dimensional vector** capturing both local and global context.
- **Query Processing:** For a text query, ColPali generates **token-level embeddings**, comparing each token with every document patch via a **similarity matrix**.
- **MaxSim Similarity:** This matrix computes the similarity between each query token and each document patch, selecting the maximum similarity for retrieval. This is known as a **late interaction** approach, inspired by ColBERT, allowing ColPali to capture intricate context across both layout and text.

This architecture enables ColPali to analyze both **layout** and **textual content** in a single pass, making retrieval more accurate for structured documents.

> "ColPali’s late interaction strategy is inspired by ColBERT and improves search by analyzing layout and textual content in a single pass."

-----

### Source: https://blog.vespa.ai/retrieval-with-vision-language-models-colpali/
ColPali **embeds PDF page screenshots** (including images, charts, tables) directly into **vector representations** for retrieval.

- **Patch Embedding:** Each grid cell (patch) is projected into a **128-dimensional embedding**; each page produces **1,030 patch embeddings**.
- **Query Embedding:** The textual query is projected into the same 128-dimensional space, one vector per text token.
- **Similarity Scoring:** A **late-interaction (MaxSim)** scoring mechanism computes similarity between query tokens and image patch vectors, mirroring ColBERT’s approach.
- **Efficiency:** With `bfloat16`, a page’s embeddings require 256KB; binarization can reduce this to 8KB.
- **Model Details:** The base model is **PaliGemma 3** (~3B parameters), making it relatively lightweight for a vision-language model.

> "The model uses a late-interaction scoring mechanism (MaxSim) to compute the similarity score between the query token vectors and the page image’s patch vectors."

-----

### Source: https://decodingml.substack.com/p/the-king-of-multi-modal-rag-colpali
A ColPali-based RAG system is composed of five main components:

- **PDF to Image Converter:** Converts PDFs to images, preserving layout and all visual elements.
- **Storage Service:** Stores the converted images for later retrieval.
- **ColPali Model (ColQwen 2.5):** Generates visual embeddings from document images, capturing spatial relationships and formatting context.
- **Vector Database (Qdrant):** Stores and indexes the embeddings for efficient similarity search.
- **Multimodal LLM (Claude Sonnet 3.7):** Interprets retrieved images and generates responses with full awareness of layout and content.

This pipeline ensures **preservation of document structure** and enables precise retrieval and comprehension, especially for tables and figures.

> "ColQwen 2.5 captures spatial relationships, formatting context, and the interplay between text and visual elements."

-----

### Source: https://learnopencv.com/multimodal-rag-with-colpali/
ColPali’s workflow involves two key phases: **offline indexing** and **online querying**.

- **Bag-of-Embeddings:** Each PDF page/image is encoded into multiple **128D patch embeddings** (ColBERT-style bag-of-embeddings).
- **Offline Indexing:** Embeddings are indexed in a vector database such as Vespa or LintDB. ColPali is much faster at this phase (0.37s per page) compared to typical parsers (7.22s per page).
- **Online Querying:**
  - Query is encoded on-the-fly.
  - **Late interaction (MaxSim)** computes maximum similarity between query embeddings and document embeddings.
  - Top-k similar images are returned, which can be sent to a multimodal LLM (e.g., Gemini) for answer generation.

> "A late interaction mechanism similar to the ColBERT ranking model... computes the maximum similarity score between query embeddings and pre-indexed document embeddings."

-----

### Source: https://arxiv.org/html/2407.01449v1
In the **PaliGemma model architecture**, **SigLIP-generated patch embeddings** are processed by a text language model for LLM-contextualized output.

- **Vision-Language Alignment:** ColPali extends PaliGemma-3B to produce **ColBERT-style multi-vector representations** for both text and images.
- **Architecture Diagram:** Figure 2 in the paper visually presents the architecture, showing how patch embeddings from the vision encoder interact with token embeddings from the language model, culminating in a MaxSim-based similarity search.

> "We introduce ColPali, a Paligemma-3B extension that is capable of generating ColBERT-style multi-vector representations of text and images (Figure 2)."

-----

</details>

---

<details>
<summary>Recent (2023–2025) best practice guides and reference implementations for integrating ColPali or similar multimodal RAG retrievers as callable tools in agentic frameworks (e.g., LangGraph, LangChain, ReAct agents), including example code for tool registration and orchestration.</summary>

### Source: https://learnopencv.com/multimodal-rag-with-colpali/
The article introduces **ColPali** as a novel multimodal RAG retriever, designed for efficient extraction of information from documents containing images, tables, charts, and text by treating each page as an image. ColPali leverages Vision Language Models (VLMs) to interpret complex document structures, such as financial reports and legal contracts, where factual accuracy is paramount.

Key points include:
- **ColPali’s architecture** addresses challenges with unstructured data elements, providing higher accuracy in fact and figure extraction compared to traditional retrieval systems.
- The guide presents a **step-by-step tutorial** for building an application that analyzes SEC 10-Q financial reports using ColPali in conjunction with Gemini, a VLM, for both retrieval and generation.
- The workflow involves: 
  1. **Document preprocessing**—each page is converted into an image.
  2. **Embedding and retrieval**—ColPali creates visual embeddings, enabling similarity search based on both visual and textual cues.
  3. **Integration with agentic frameworks**—While the article does not provide direct code snippets for LangChain or LangGraph integration, it outlines the general orchestration: ColPali acts as a callable tool via a unified API, which can be invoked by an agent as part of a pipeline.
- The article references the **ViDoRe benchmark**, highlighting ColPali’s effectiveness in production RAG scenarios.

The guide recommends ColPali for enterprises seeking robust, production-ready multimodal retrieval and includes pointers for further customization, especially for organizations aiming to plug ColPali into agentic frameworks for advanced document analysis.

-----

### Source: https://huggingface.co/learn/cookbook/en/multimodal_rag_using_document_retrieval_and_smol_vlm
This Hugging Face notebook provides a **practical recipe** for building a lightweight **multimodal RAG system** using ColSmolVLM (a variant of ColPali) and SmolVLM, suitable for consumer GPUs and even Google Colab free-tier.

Key implementation steps:
- **Dependency Installation**: 
  ```bash
  pip install -q git+https://github.com/sergiopaniego/byaldi.git@colsmolvlm-support
  ```
- **Tool Registration**: The notebook demonstrates setting up ColSmolVLM as a retriever and SmolVLM as the VLM in a pipeline. The retriever is wrapped as a Python function (tool) that takes a query, retrieves relevant document pages (images), and returns results for downstream processing.
- **Integration with Agentic Frameworks**: While the notebook doesn’t illustrate direct LangChain or LangGraph registration, the modular design (retriever function, VLM function) enables straightforward integration—these functions can be registered as tools in frameworks like LangChain using the `tool` decorator or tool registration APIs.
- **Example Code Snippet**:
  ```python
  from byaldi.colsmolvlm import ColSmolVLM
  retriever = ColSmolVLM(...)
  def retrieve_fn(query):
      return retriever.retrieve(query)
  # Tool registration in agentic frameworks would wrap retrieve_fn
  ```
- **Orchestration**: The pipeline consists of:
  - Query input to the retriever tool
  - Selection of top-k relevant document snippets (images)
  - Passing results to the VLM for final answer generation

This approach serves as a reference implementation for integrating ColPali-like retrievers as callable tools in agentic orchestration pipelines.

-----

### Source: https://arxiv.org/html/2411.04952v1
The M3DocRAG paper formally describes a **multi-modal RAG framework** capable of integrating retrievers like ColPali in both closed-domain and open-domain settings. The system is designed for efficient multi-page and multi-document retrieval and answer generation.

Framework details:
- **Three-stage workflow**:
  1. **Document Embedding**: Convert document pages to RGB images; extract visual embeddings using ColPali.
  2. **Page Retrieval**: Retrieve the top-K most relevant pages for a text query using similarity operators (e.g., MaxSim for ColPali). For scalability, page indices like IVF can be built for rapid search.
  3. **Question Answering**: Use a multi-modal language model (e.g., Qwen2-VL) to generate answers from retrieved pages.
- **Reference Implementation**: The paper offers a detailed illustration (Fig. 3) and clarifies that ColPali can be modularly called as a retriever within agentic pipelines, which then pass outputs to downstream models for orchestration.
- **Flexibility**: The architecture supports integration into agentic frameworks (e.g., ReAct agents) by exposing each stage as a callable function or tool, which can be registered in tool registries or orchestrator modules.

This reference serves as a robust conceptual and practical guide for orchestrating ColPali and similar retrievers as callable tools in agent-based RAG systems.

-----

### Source: https://huggingface.co/learn/cookbook/en/multimodal_rag_using_document_retrieval_and_reranker_and_vlms
This Hugging Face notebook demonstrates a **multimodal RAG system** using **ColQwen2** (another ColPali-like retriever), a reranker model, and a quantized VLM. The workflow is suitable for consumer hardware and provides modular implementation for each stage.

Relevant implementation details:
- **Retriever Setup**: ColQwen2 is wrapped as a retriever function, which takes a query and returns the top relevant document images.
- **Reranker**: A reranker model refines the retrieval outputs for better answer quality.
- **Tool Registration**: Though the notebook focuses on pipeline composition, the retriever, reranker, and VLM modules are all defined as Python functions/classes. These can be registered as tools in agentic frameworks using standard APIs (e.g., LangChain’s `tool` decorator or `Tool` class).
- **Example Code** (simplified):
  ```python
  def retrieve_documents(query):
      # Call ColQwen2 retriever
      return retrieved_pages

  def rerank_documents(retrieved_pages, query):
      # Call reranker model
      return top_ranked_pages

  def generate_answer(pages, query):
      # Call VLM
      return answer

  # These functions can be registered as agent tools
  ```
- **Orchestration**: Agents call these tools in sequence—retriever → reranker → VLM—allowing flexible orchestration in ReAct, LangChain, or LangGraph frameworks.

This notebook provides a clear reference for structuring and registering multimodal RAG retrieval tools in modern agentic pipelines.

-----

</details>

---

<details>
<summary>Authoritative lists and critical, up-to-date descriptions of real-world multimodal embedding models—beyond CLIP—(e.g., Google Imagen, Cohere embeddings, Voyage, Qwen2.5-VL, PaliGemma), including modality support, comparative strengths/limitations, and practical usage scenarios across images, PDFs, video, and audio.</summary>

### Source: https://www.edenai.co/post/best-multimodal-embeddings-apis
**Authoritative List and Capabilities of Multimodal Embedding APIs (2025)**

1. **Amazon Titan Multimodal**
   - **Modality Support:** Text (up to 128 tokens, English) and images.
   - **Strengths:** Generates **1,024-dimensional vectors** for both images and text, facilitating high-accuracy, fast search experiences.
   - **Use Cases:** Image search by text/image, e-commerce, visual semantic search.

2. **Aleph Alpha**
   - **Modality Support:** Text and image (multilingual, multimodal).
   - **Strengths:** Embeddings share a common latent space; advanced visual feature extraction for recognition/classification.
   - **Use Cases:** Multilingual content processing, cross-modal search, content-driven services.

3. **Google’s Multimodal Embeddings API**
   - **Modality Support:** Images and/or text.
   - **Strengths:** Produces **1,408-dimensional vectors**; image and text vectors are interchangeable within the same semantic space.
   - **Use Cases:** Image classification, content moderation, image-to-text and text-to-image retrieval.

4. **Microsoft’s Multimodal Embeddings API**
   - **Modality Support:** Images and text.
   - **Strengths:** Maps both images and text queries to the same multi-dimensional vector space for unified retrieval.
   - **Use Cases:** Unified search, cross-modal information retrieval.

**Practical Limitations:**
- **PDF and video support:** Not explicitly mentioned; typically requires preprocessing (e.g., text/image extraction from PDFs, frame sampling from videos).
- **Audio support:** Not detailed in this source, indicating primary focus on image and text modalities.

**Summary:** Leading APIs from Amazon, Aleph Alpha, Google, and Microsoft are authoritative providers of real-world multimodal embedding services, with broad application across semantic search, content moderation, and cross-modal retrieval. Most focus on text and image, with limited direct support for audio, video, or PDF modalities.

-----

</details>

---

## Sources Scraped From Research Results

---
<details>
<summary>Understanding Multimodal LLMs - by Sebastian Raschka, PhD</summary>

# Understanding Multimodal LLMs

### An introduction to the main techniques and latest models

It was a wild two months. There have once again been many developments in AI research, with two Nobel Prizes awarded to AI and several interesting research papers published.

Among others, Meta AI released their latest Llama 3.2 models, which include open-weight versions for the 1B and 3B large language models and two multimodal models.

In this article, I aim to explain how multimodal LLMs function. Additionally, I will review and summarize roughly a dozen other recent multimodal papers and models published in recent weeks (including Llama 3.2) to compare their approaches.

[![](https://substackcdn.com/image/fetch/$s_!Pq2z!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0d76dab1-362f-45b6-9b12-a12ac131edc5_1600x944.png)](https://substackcdn.com/image/fetch/$s_!Pq2z!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0d76dab1-362f-45b6-9b12-a12ac131edc5_1600x944.png) _An illustration of a multimodal LLM that can accept different input modalities (audio, text, images, and videos) and returns text as the output modality._

# 1\. Use cases of multimodal LLMs

What are multimodal LLMs? As hinted at in the introduction, multimodal LLMs are large language models capable of processing multiple types of inputs, where each "modality" refers to a specific type of data—such as text (like in traditional LLMs), sound, images, videos, and more. For simplicity, we will primarily focus on the image modality alongside text inputs.

A classic and intuitive application of multimodal LLMs is image captioning: you provide an input image, and the model generates a description of the image, as shown in the figure below.

[![](https://substackcdn.com/image/fetch/$s_!8kaL!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F93884822-79f1-498d-a33a-8a367ba57134_1500x1222.png)](https://substackcdn.com/image/fetch/$s_!8kaL!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F93884822-79f1-498d-a33a-8a367ba57134_1500x1222.png) _Example use of a multimodal LLM explaining [a meme](https://x.com/PainSci/status/1309570607458086914)._

Of course, there are many other use cases. For example, one of my favorites is extracting information from a PDF table and converting it into LaTeX or Markdown.

# 2\. Common approaches to building multimodal LLMs

There are two main approaches to building multimodal LLMs:

- Method A: Unified Embedding Decoder Architecture approach;

- Method B: Cross-modality Attention Architecture approach.

[![](https://substackcdn.com/image/fetch/$s_!8miE!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F53956ae8-9cd8-474e-8c10-ef6bddb88164_1600x938.png)](https://substackcdn.com/image/fetch/$s_!8miE!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F53956ae8-9cd8-474e-8c10-ef6bddb88164_1600x938.png) _The two main approaches to developing multimodal LLM architectures._

As shown in the figure above, the _**Unified Embedding-Decoder Architecture**_ utilizes a single decoder model, much like an unmodified LLM architecture such as GPT-2 or Llama 3.2. In this approach, images are converted into tokens with the same embedding size as the original text tokens, allowing the LLM to process both text and image input tokens together after concatenation.

The _**Cross-Modality Attention Architecture**_ employs a cross-attention mechanism to integrate image and text embeddings directly within the attention layer.

In the following sections, we will explore how these methods work on a conceptual level. Then, we will look at recent research papers on multimodal LLMs to see how they are applied in practice.

## **2.1 Method A: Unified Embedding Decoder Architecture**

Let’s begin with the unified embedding decoder architecture, illustrated again in the figure below.

[![](https://substackcdn.com/image/fetch/$s_!Ws6n!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F91955021-7da5-4bc4-840e-87d080152b18_1166x1400.png)](https://substackcdn.com/image/fetch/$s_!Ws6n!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F91955021-7da5-4bc4-840e-87d080152b18_1166x1400.png) _Illustration of the unified embedding decoder architecture, which is an unmodified decoder-style LLM (like GPT-2, Phi-3, Gemma, or Llama 3.2) that receives inputs consisting of image token and text token embeddings._

In the unified embedding-decoder architecture, an image is converted into embedding vectors, similar to how input text is converted into embeddings in a standard text-only LLM.

For a typical text-only LLM that processes text, the text input is usually tokenized (e.g., using Byte-Pair Encoding) and then passed through an embedding layer, as shown in the figure below.

[![](https://substackcdn.com/image/fetch/$s_!dOba!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc97009dd-cee6-455f-87fe-64c33a868e9f_986x858.png)](https://substackcdn.com/image/fetch/$s_!dOba!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc97009dd-cee6-455f-87fe-64c33a868e9f_986x858.png) _Illustration of the standard process for tokenizing text and converting it into token embedding vectors, which are subsequently passed to an LLM during training and inference._

### **2.1.1 Understanding Image encoders**

Analogous to the tokenization and embedding of text, image embeddings are generated using an image encoder module (instead of a tokenizer), as shown in the figure below.

[![](https://substackcdn.com/image/fetch/$s_!PlBh!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F15e9cc2f-95de-4723-9de5-9f2af7573aaa_790x750.png)](https://substackcdn.com/image/fetch/$s_!PlBh!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F15e9cc2f-95de-4723-9de5-9f2af7573aaa_790x750.png) _Illustration of the process for encoding an image into image patch embeddings._

What happens inside the image encoder shown above? To process an image, we first divide it into smaller patches, much like breaking words into subwords during tokenization. These patches are then encoded by a pretrained vision transformer (ViT), as shown in the figure below.

[![](https://substackcdn.com/image/fetch/$s_!_DNf!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffef5f8cb-c76c-4c97-9771-7fdb87d7d8cd_1600x1135.png)](https://substackcdn.com/image/fetch/$s_!_DNf!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffef5f8cb-c76c-4c97-9771-7fdb87d7d8cd_1600x1135.png) _Illustration of a classic vision transformer (ViT) setup, similar to the model proposed in [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (2020)._

Note that ViTs are often used for classification tasks, so I included the classification head in the figure above. However, in this case, we only need the image encoder part.

### **2.1.2 The role of the linear projection module**

The "linear projection" shown in the previous figure consists of a single linear layer (i.e., a fully connected layer). The purpose of this layer is to project the image patches, which are flattened into a vector, into an embedding size compatible with the transformer encoder. This linear projection is illustrated in the figure below. An image patch, flattened into a 256-dimensional vector, is up-projected to a 768-dimensional vector.

[![](https://substackcdn.com/image/fetch/$s_!i9i4!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fee32d720-92d7-48c2-b39d-adf61a870075_1600x681.png)](https://substackcdn.com/image/fetch/$s_!i9i4!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fee32d720-92d7-48c2-b39d-adf61a870075_1600x681.png) _Illustration of a linear projection layer that projects flattened image patches from a 256-dimensional into a 768-dimensional embedding space._

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

[![](https://substackcdn.com/image/fetch/$s_!zjmg!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0d56ea06-d202-4eb7-9e01-9aac492ee309_1522x1206.png)](https://substackcdn.com/image/fetch/$s_!zjmg!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0d56ea06-d202-4eb7-9e01-9aac492ee309_1522x1206.png) _Image tokenization and embedding (left) and text tokenization and embedding (right) side by side._

As you can see in the figure above, I included an additional _**projector**_ module that follows the image encoder. This _projector_ is usually just another _**linear projection**_ layer that is similar to the one explained earlier. The purpose is to project the image encoder outputs into a dimension that matches the dimensions of the embedded text tokens, as illustrated in the figure below. (As we will see later, the projector is sometimes also called adapter, adaptor, or connector.)

[![](https://substackcdn.com/image/fetch/$s_!TaTW!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5d0be64c-da90-4193-86db-804f6a8a0abb_1542x1242.png)](https://substackcdn.com/image/fetch/$s_!TaTW!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5d0be64c-da90-4193-86db-804f6a8a0abb_1542x1242.png) _Another side-by-side comparison between image tokenization and text tokenization, where the role of the projector is to match the text token embedding dimensions._

Now that the image patch embeddings have the same embedding dimension as the text token embeddings, we can simply concatenate them as input to the LLM, as shown in the figure at the beginning of this section. Below is the same figure again for easier reference.

[![](https://substackcdn.com/image/fetch/$s_!FTft!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa219f185-211b-4569-9398-2e080e2c5619_1166x1400.png)](https://substackcdn.com/image/fetch/$s_!FTft!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa219f185-211b-4569-9398-2e080e2c5619_1166x1400.png) _After projecting the image patch tokens into the same dimension as the text token embeddings, we can simply concatenate them as input to a standard LLM._

By the way, the image encoder we discussed in this section is usually a pretrained vision transformer. A popular choice is [CLIP](https://github.com/openai/CLIP) or [OpenCLIP](https://github.com/mlfoundations/open_clip).

However, there are also versions of Method A that operate directly on patches, such as [Fuyu](https://www.adept.ai/blog/fuyu-8b), which is shown in the figure below.

[![](https://substackcdn.com/image/fetch/$s_!LB1L!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F28269d0d-b806-4ae7-bf96-b282affd7e93_1600x645.png)](https://substackcdn.com/image/fetch/$s_!LB1L!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F28269d0d-b806-4ae7-bf96-b282affd7e93_1600x645.png) _Annotated figure of the Fuyu multimodal LLM that operates directly on the image patches without image encoder. (Annotated figure from [https://www.adept.ai/blog/fuyu-8b](https://www.adept.ai/blog/fuyu-8b).)_

As illustrated in the figure above, Fuyu passes the input patches directly into a linear projection (or embedding layer) to learn its own image patch embeddings rather than relying on an additional pretrained image encoder like other models and methods do. This greatly simplifies the architecture and training setup.

## **2.2 Method B: Cross-Modality Attention Architecture**

Now that we have discussed the unified embedding decoder architecture approach to building multimodal LLMs and understand the basic concept behind image encoding, let's talk about an alternative way of implementing multimodal LLMs via cross-attention, as summarized in the figure below.

[![](https://substackcdn.com/image/fetch/$s_!7Xvv!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd9c06055-b959-45d1-87b2-1f4e90ceaf2d_1296x1338.png)](https://substackcdn.com/image/fetch/$s_!7Xvv!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd9c06055-b959-45d1-87b2-1f4e90ceaf2d_1296x1338.png) _An illustration of the Cross-Modality Attention Architecture approach to building multimodal LLMs._

In the Cross-Modality Attention Architecture method depicted in the figure above, we still use the same image encoder setup we discussed previously. However, instead of encoding the patches as input to the LLM, we connect the input patches in the multi-head attention layer via a cross-attention mechanism.

The idea is related and goes back to the original transformer architecture from the 2017 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper, highlighted in the figure below.

[![](https://substackcdn.com/image/fetch/$s_!JYyE!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5d028b95-7965-43e0-b8fc-350609a69377_1370x1582.png)](https://substackcdn.com/image/fetch/$s_!JYyE!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5d028b95-7965-43e0-b8fc-350609a69377_1370x1582.png) _High-level illustration of the cross-attention mechanism used in the original transformer architecture. (Annotated figure from the "Attention Is All You Need" paper: https://arxiv.org/abs/1706.03762.)_

Note that the original "Attention Is All You Need" transformer depicted in the figure above was originally developed for language translation. So, it consists of a text **en** coder (left part of the figure) that takes the sentence to be translated and generates the translation via a text **de** coder (right part of the figure). In the context of multimodal LLM, the encoder is an image encoder instead of a text encoder, but the same idea applies.

How does cross-attention work? Let's have a look at a conceptual drawing of what happens inside the regular self-attention mechanism.

[![](https://substackcdn.com/image/fetch/$s_!HqoQ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff763532b-1eed-4f7d-ae2c-7783d4f4fc46_1440x1194.png)](https://substackcdn.com/image/fetch/$s_!HqoQ!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff763532b-1eed-4f7d-ae2c-7783d4f4fc46_1440x1194.png) _Outline of the regular self-attention mechanism. (This flow depicts one of the heads in a regular multi-head attention module.)_

In the figure above, x is the input, and _Wq_ is a weight matrix used to generate the queries ( _Q_). Similarly, _K_ stands for keys, and _V_ stands for values. A represents the attention scores matrix, and _Z_ are the inputs (x) transformed into the output context vectors. (If this seems confusing, you may find a comprehensive introduction in Chapter 3 of my [Build a Large Language Model from Scratch book](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167/) helpful; alternatively, you may also find my article, [Understanding and Coding Self-Attention, Multi-Head Attention, Cross-Attention, and Causal-Attention in LLMs](https://magazine.sebastianraschka.com/p/understanding-and-coding-self-attention) helpful here.)

In cross-attention, in contrast to self-attention, we have two different input sources, as illustrated in the following figure.

[![](https://substackcdn.com/image/fetch/$s_!3PZD!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffe4cc6f4-ca9a-431b-b572-95a1fda373a7_1508x1120.png)](https://substackcdn.com/image/fetch/$s_!3PZD!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffe4cc6f4-ca9a-431b-b572-95a1fda373a7_1508x1120.png) _Illustration of cross attention, where there can be two different inputs x1 and x2_

As illustrated in the previous two figures, in self-attention, we work with the same input sequence. In cross-attention, we mix or combine two different input sequences.

In the case of the original transformer architecture in the _Attention Is All You Need_ paper, the two inputs _x1_ and _x2_ correspond to the sequence returned by the encoder module on the left ( _x2_) and the input sequence being processed by the decoder part on the right ( _x1_). In the context of a multimodal LLM, _x2_ is the output of an image encoder. (Note that the queries usually come from the decoder, and the keys and values typically come from the encoder.)

Note that in cross-attention, the two input sequences _x1_ and _x2_ can have different numbers of elements. However, their embedding dimensions must match. If we set _x1 = x2_, this is equivalent to self-attention.

# 3\. Unified decoder and cross-attention model training

Now that we have talked a bit about the two major multimodal design choices, let's briefly talk about how we deal with the three major components during model training, which are summarized in the figure below.

[![](https://substackcdn.com/image/fetch/$s_!e2P-!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F24a12032-d32e-41f6-b390-4e321e1ea29f_1600x770.png)](https://substackcdn.com/image/fetch/$s_!e2P-!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F24a12032-d32e-41f6-b390-4e321e1ea29f_1600x770.png) _An overview of the different components in a multimodal LLM. The components numbered 1-3 can be frozen or unfrozen during the multimodal training process._

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

[![](https://substackcdn.com/image/fetch/$s_!fTYU!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7c8578fa-70f2-474f-9e98-87621f2dce96_1600x833.png)](https://substackcdn.com/image/fetch/$s_!fTYU!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7c8578fa-70f2-474f-9e98-87621f2dce96_1600x833.png) _Illustration of the multimodal LLM approach used by Llama 3.2. (Annotated figure from the Llama 3 paper: https://arxiv.org/abs/2407.21783.The video and speech parts are visually occluded to focus the attention on the image part.)_

Note that while the figure also depicts video and speech as possible modalities, the models that were released as of this writing focus only on image and text.

Llama 3.2 uses the cross-attention-based approach. However, it differs a bit from what I wrote about earlier, namely that in multimodal LLM development, we usually freeze the image encoder and only update the LLM parameters during pretraining.

Here, the researchers almost take the opposite approach: they update the image encoder but do not update the language model's parameters. They write that this is intentional and done to preserve the text-only capabilities so that the 11B and 90B multimodal models can be used as drop-in replacements for the Llama 3.1 8B and 70B text-only model on text tasks.

The training itself is done in multiple iterations, starting with the Llama 3.1 text models. After adding the image encoder and projection (here called "adapter") layers, they pretrain the model on image-text data. Then, similar to the Llama 3 model text-only training (I wrote about it in [an earlier article](https://magazine.sebastianraschka.com/i/147749119/llama-overview)), they follow up with instruction and preference finetuning.

Instead of adopting a pretrained model such as CLIP as an image encoder, the researchers used a vision transformer that they pretrained from scratch. Specifically, they adopted the  ViT-H/14 variant (630 million parameters) of the classic vision transformer architecture ( [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)). They then pretrained the ViT on a dataset of 2.5 billion image-text pairs over five epochs; this was done before connecting the image encoder to the LLM. (The image encoder takes 224×224 resolution images and divides them into a 14×14 grid of patches, with each patch sized at 16×16 pixels.)

As the cross-attention layers add a substantial amount of parameters, they are only added in every fourth transformer block. (For the 8B model, this adds 3B parameters, and for the 70B model, this adds 20 billion parameters.)

## **4.2 Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models**

_[The Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models](https://www.arxiv.org/abs/2409.17146)_ paper (September 25, 2024) is notable because it promises to open source not only the model weights but also the dataset and source code similar to the language-only OLMo LLM. (This is great for LLM research as it allows us to take a look at the exact training procedure and code and also lets us run ablation studies and reproduce results on the same dataset.)

If you are wondering why there are two names in the paper title, Molmo refers to the model (Multimodal Open Language Model), and PixMo (Pixels for Molmo) is the dataset.

[![](https://substackcdn.com/image/fetch/$s_!9P0w!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F73337002-8feb-4f1b-a109-1407096e32c5_1104x704.png)](https://substackcdn.com/image/fetch/$s_!9P0w!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F73337002-8feb-4f1b-a109-1407096e32c5_1104x704.png) _Illustration of the Molmo decoder-only approach (Method A). Annotated figure adapted from the Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models paper: https://www.arxiv.org/abs/2409.17146._

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

[![](https://substackcdn.com/image/fetch/$s_!6n6Y!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F45916952-b1ee-4972-a956-e45703e3fe36_1600x927.png)](https://substackcdn.com/image/fetch/$s_!6n6Y!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F45916952-b1ee-4972-a956-e45703e3fe36_1600x927.png) _Overview of the three multimodal approaches. (Annotated figure from the NVLM: Open Frontier-Class Multimodal LLMs paper: https://arxiv.org/abs/2409.11402)_

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

[![](https://substackcdn.com/image/fetch/$s_!Zrt8!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2247e684-253a-462e-afb4-549411d5741a_1490x1068.png)](https://substackcdn.com/image/fetch/$s_!Zrt8!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2247e684-253a-462e-afb4-549411d5741a_1490x1068.png) _An overview of the multimodal Qwen model, which can process input images with various different resolutions natively. (Annotated figure from the Qwen2-VL paper: https://arxiv.org/abs/2409.12191)_

The native resolution input is implemented via a modified ViT by removing the original absolute position embeddings and introducing 2D-RoPE.

They used a classic vision encoder with 675M parameters and LLM backbones of varying sizes, as shown in the table below.

[![](https://substackcdn.com/image/fetch/$s_!NdAJ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2ce9ce4a-d7ec-476d-91cb-29b6f5440b3b_1396x482.png)](https://substackcdn.com/image/fetch/$s_!NdAJ!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2ce9ce4a-d7ec-476d-91cb-29b6f5440b3b_1396x482.png) The components of the different Qwen2-VL models. (Annotated figure from the Qwen2-VL paper: https://arxiv.org/abs/2409.12191)

The training itself consists of 3 stages: (1) pretraining only the image encoder, (2) unfreezing all parameters (including LLM), and (3) freezing the image encoder and instruction-finetuning only the LLM.

## **4.5 Pixtral 12B**

_[Pixtral 12B](https://mistral.ai/news/pixtral-12b/)_ (September 17, 2024), which uses the Method A: Unified Embedding Decoder Architecture approach, is the first multimodal model from Mistral AI. Unfortunately, there is no technical paper or report available, but the Mistral team shared a few interesting tidbits in their [blog post](https://mistral.ai/news/pixtral-12b/).

Interestingly, they chose not to use a pretrained image encoder, instead training one with 400 million parameters from scratch. For the LLM backbone, they used the 12-billion-parameter [Mistral NeMo](https://mistral.ai/news/mistral-nemo/) model.

Similar to Qwen2-VL, Pixtral also supports variable image sizes natively, as illustrated in the figure below.

[![](https://substackcdn.com/image/fetch/$s_!eW3C!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F37bb0f12-4533-4f44-8907-1da868006ff3_1144x726.png)](https://substackcdn.com/image/fetch/$s_!eW3C!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F37bb0f12-4533-4f44-8907-1da868006ff3_1144x726.png) _Illustration of how Pixtral processes images of different sizes. (Annotated figure from the Pixtral blog  post: https://mistral.ai/news/pixtral-12b/)_

## **4.6 MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning**

The _[MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning](https://arxiv.org/abs/2409.20566)_ paper (September 30, 2024) provides practical tips and introduces a mixture-of-experts multimodal model alongside a dense model similar to Molmo. The models span a wide size range, from 1 billion to 30 billion parameters.

The models described in this paper focuse on Method A, a Unified Embedding Transformer Architecture, which structures inputs effectively for multimodal learning.

In addition, the paper has a series of interesting ablation studies looking into data mixtures and the effects of using coordinate tokens.

[![](https://substackcdn.com/image/fetch/$s_!fMsE!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F71b22b97-e901-4c5f-a9c2-67e32c867823_1402x1178.png)](https://substackcdn.com/image/fetch/$s_!fMsE!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F71b22b97-e901-4c5f-a9c2-67e32c867823_1402x1178.png) _Illustration of the MM1.5 approach, which includes additional coordinate tokens to denote bounding boxes. (Annotated figure from the MM1.5 paper: https://arxiv.org/abs/2409.20566.)_

## **4.7 Aria: An Open Multimodal Native Mixture-of-Experts Model**

The _[Aria: An Open Multimodal Native Mixture-of-Experts Model](https://arxiv.org/abs/2410.05993)_ paper (October 8, 2024) introduces another mixture-of-experts model approach, similar to one of the variants in the Molmo and MM1.5 lineups.

The Aria model has 24.9 billion parameters, with 3.5 billion parameters allocated per text token. The image encoder ( [SigLIP](https://arxiv.org/abs/2303.15343)) has 438-million-parameters.

This model is based on a cross-attention approach with the following overall training procedure:

1. Training the LLM backbone entirely from scratch.

2. Pretraining both the LLM backbone and the vision encoder.

## **4.8 Baichuan-Omni**

The _[Baichuan-Omni Technical Report](https://arxiv.org/abs/2410.08565)_ (October 11, 2024) introduces Baichuan-Omni, a 7-billion-parameter multimodal LLM based on Method A: the Unified Embedding Decoder Architecture approach, as shown in the figure below.

[![](https://substackcdn.com/image/fetch/$s_!-IYi!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F142c39bd-2d3f-4813-9363-5ecf616cb784_2102x1326.png)](https://substackcdn.com/image/fetch/$s_!-IYi!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F142c39bd-2d3f-4813-9363-5ecf616cb784_2102x1326.png) _An overview of the Baichuan-Omni model, which can handle various input modalities. (Annotated figure from the Baichuan-Omni paper: https://arxiv.org/abs/2410.08565)_

The training process for Baichuan-Omni involves a three-stage approach:

1. **Projector training**: Initially, only the projector is trained, while both the vision encoder and the language model (LLM) remain frozen.

2. **Vision encoder training**: Next, the vision encoder is unfrozen and trained, with the LLM still frozen.

3. **Full model training**: Finally, the LLM is unfrozen, allowing the entire model to be trained end-to-end.

The model utilizes the SigLIP vision encoder and incorporates the [AnyRes](https://arxiv.org/abs/2204.07156) module to handle high-resolution images through down-sampling techniques.

While the report does not explicitly specify the LLM backbone, it is likely based on the Baichuan 7B LLM, given the model's parameter size and the naming convention.

## **4.9 Emu3: Next-Token Prediction is All You Need**

The _Emu3: Next-Token Prediction is All You Need_ paper (September 27, 2024) presents a compelling alternative to diffusion models for image generation, which is solely based on a transformer-based decoder architecture. Although it's not a multimodal LLM in the classic sense (i.e., models focused on image understanding rather than generation), Emu3 is super interesting as it demonstrates that it's possible to use transformer decoders for image generation, which is a task typically dominated by diffusion methods. (However, note that there have been other similar approaches before, such as [Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation](https://arxiv.org/abs/2406.06525).)

[![](https://substackcdn.com/image/fetch/$s_!IWU7!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F775db9c7-662f-4314-a5c4-c3f5efe0238d_1056x904.png)](https://substackcdn.com/image/fetch/$s_!IWU7!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F775db9c7-662f-4314-a5c4-c3f5efe0238d_1056x904.png) _Emu3 is primarily an LLM for image generation as an alternative to diffusion models. (Annotated figure from the Emu3 paper: https://arxiv.org/abs/2409.18869)_

The researchers trained Emu3 from scratch and then used [Direct Preference Optimization](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb) (DPO) to align the model with human preferences.

The architecture includes a vision tokenizer inspired by [SBER-MoVQGAN](https://arxiv.org/abs/2209.09002). The core LLM architecture is based on Llama 2, yet it is trained entirely from scratch.

## **4.10 Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation**

We previously focused on multimodal LLMs for image understanding and just saw one example for image generation with Emu 3 above. Now, the _[Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2410.13848)_ paper (October 17, 2024) introduces a framework that unifies multimodal understanding and generation tasks within a single LLM backbone.

A key feature of Janus is the decoupling of visual encoding pathways to address the distinct requirements of understanding and generation tasks. The researchers argue that image understanding tasks require high-dimensional semantic representations, while generation tasks require detailed local information and global consistency in images. By separating these pathways, Janus effectively manages these differing needs.

The model employs the SigLIP vision encoder, similar to that used in Baichuan-Omni, for processing visual inputs. For image generation, it utilizes a [Vector Quantized (VQ)](https://arxiv.org/abs/2406.06525) tokenizer to handle the generation process. The base LLM in Janus is the [DeepSeek-LLM](https://arxiv.org/abs/2401.02954) with 1.3 billion parameters.

[![](https://substackcdn.com/image/fetch/$s_!9UFg!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F89d62626-4386-4e73-8992-158550752ce2_1434x692.png)](https://substackcdn.com/image/fetch/$s_!9UFg!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F89d62626-4386-4e73-8992-158550752ce2_1434x692.png) _An overview of the unified decoder-only framework used in Janus. (Annotated figure from the Janus paper: https://arxiv.org/abs/2410.13848.)_

The training process for the model in this image follows three stages, as shown in the figure below.

[![](https://substackcdn.com/image/fetch/$s_!Da5n!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2fb4f079-0771-4d21-8805-fded73134983_1536x648.png)](https://substackcdn.com/image/fetch/$s_!Da5n!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2fb4f079-0771-4d21-8805-fded73134983_1536x648.png) Illustration of the 3-stage training process of the Janus model. (Annotated figure from the Janus paper: https://arxiv.org/abs/2410.13848)

In Stage I, only the projector layers and image output layer are trained while the LLM, understanding, and generation encoders remain frozen. In Stage II, the LLM backbone and text output layer are unfrozen, allowing for unified pretraining across understanding and generation tasks. Finally, in Stage III, the entire model, including the SigLIP image encoder, is unfrozen for supervised fine-tuning, enabling the model to fully integrate and refine its multimodal capabilities.

# Conclusion

As you may have noticed, I almost entirely skipped both the modeling and the computational performance comparisons. First, comparing the performance of LLMs and multimodal LLMs on public benchmarks is challenging due to prevalent data contamination, meaning that the test data may have been included in the training data.

Additionally, the architectural components vary so much that making an apples-to-apples comparison is difficult. So, big kudos to the NVIDIA team for developing NVLM in different flavors, which allowed for a comparison between the decoder-only and cross-attention approaches at least.

In any case, the main takeaway from this article is that multimodal LLMs can be built successfully in many different ways. Below is a figure that summarizes the different components of the models covered in this article.

[![](https://substackcdn.com/image/fetch/$s_!R_9Y!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb043e6d7-78e5-4628-987a-b333d3a58829_2224x1180.png)](https://substackcdn.com/image/fetch/$s_!R_9Y!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb043e6d7-78e5-4628-987a-b333d3a58829_2224x1180.png) An overview of the different models covered in this article along with their subcomponents and training approaches.

I hope you found reading this article educational and now have a better understanding of how multimodal LLMs work!

### Original URL
https://magazine.sebastianraschka.com/p/understanding-multimodal-llms
</details>

---
<details>
<summary>Grounding Multimodal Large Language Models in Actions - Apple Machine Learning Research</summary>

# Grounding Multimodal Large Language Models in Actions

Multimodal Large Language Models (MLLMs) have demonstrated a wide range of capabilities across many domains, including Embodied AI. In this work, we study how to best ground a MLLM into different embodiments and their associated action spaces, with the goal of leveraging the multimodal world knowledge of the MLLM. We first generalize a number of methods through a unified architecture and the lens of action space adaptors. For continuous actions, we show that a learned tokenization allows for sufficient modeling precision, yielding the best performance on downstream tasks. For discrete actions, we demonstrate that semantically aligning these actions with the native output token space of the MLLM leads to the strongest performance. We arrive at these lessons via a thorough study of seven action space adapters on five different environments, encompassing over 114 embodied tasks.

We examine the capability of Multimodal Large Language Models (MLLMs) to tackle diverse domains that extend beyond the traditional language and vision tasks these models are typically trained on. Specifically, our focus lies in areas such as Embodied AI, Games, UI Control, and Planning. To this end, we introduce a process of adapting an MLLM to a Generalist Embodied Agent (GEA). GEA is a single unified model capable of grounding itself across…

### Original URL
https://machinelearning.apple.com/research/grounding-multimodal-large
</details>

---
<details>
<summary>ColPali: Efficient Document Retrieval with Vision Language Models</summary>

# ColPali: Efficient Document Retrieval with Vision Language Models

Manuel Faysse 1,3 Hugues Sibille∗1,4  Tony Wu∗1  Bilel Omrani1

Gautier Viaud1Céline Hudelot3Pierre Colombo2,3

1Illuin Technology  2Equall.ai

3CentraleSupélec, Paris-Saclay  4ETH Zürich

[manuel.faysse@centralesupelec.fr](https://arxiv.org/html/2407.01449v2/manuel.faysse@centralesupelec.fr)

## 1 Introduction

![Refer to caption](https://arxiv.org/html/2407.01449v2/extracted/5705564/images/similarity_maps/similarity_map_energy.png)
Figure 1: For each term in a user query, ColPali identifies the most relevant document image patches (highlighted zones) and computes a query-to-page matching score. We can then swiftly retrieve the most relevant documents from a large pre-indexed corpus.

![Refer to caption](https://arxiv.org/html/2407.01449v2/extracted/5705564/images/final_architecture.png)
Figure 2: ColPali simplifies document retrieval w.r.t. standard retrieval methods while achieving stronger performances with better latencies. Latencies and results are detailed in [section 5](https://arxiv.org/html/2407.01449v2#S5) and [subsection B.5](https://arxiv.org/html/2407.01449v2#A2.SS5).

Document Retrieval consists in matching a user query to relevant documents in a given corpus. It is central to many industrial applications, either as a standalone ranking system (search engines) or as part of more complex information extraction or Retrieval Augmented Generation (RAG) pipelines.

Over recent years, pretrained language models have enabled large improvements in text embedding models. In practical industrial settings, however, the main performance bottleneck for efficient document retrieval is not in embedding model performance but in the prior data ingestion pipeline. To index a standard PDF document, many steps are required. First, PDF parsers or Optical Character Recognition (OCR) systems are used to extract words from the pages. Document layout detection models can then be run to segment paragraphs, titles, and other page objects such as tables, figures, and headers. A chunking strategy is then defined to group text passages with some semantical coherence, and modern retrieval setups may even integrate a captioning step to describe visually rich elements in a natural language form, more suitable for embedding models.
In our experiments ([Table 2](https://arxiv.org/html/2407.01449v2#S5.T2)), we typically find that optimizing the ingestion pipeline yields much greater performance on visually rich document retrieval than optimizing the text embedding model.

Contribution 1: ViDoRe.
In this work, we argue that document retrieval systems should not be evaluated solely on the capabilities of text embedding models, but should also consider the context and visual elements of the documents to be retrieved. To this end, we create and openly release ViDoRe, a comprehensive benchmark to evaluate systems on page-level document retrieval with a wide coverage of domains, visual elements, and languages. ViDoRe targets practical document retrieval settings, in which user queries may require both textual and visual understanding to be correctly matched to relevant documents. We highlight the shortcomings of current text-centric systems in these settings.  
The benchmark leaderboard is hosted publicly at [https://huggingface.co/spaces/vidore/vidore-leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard) to encourage further developments.

Contribution 2: ColPali.
We propose a novel model architecture and training strategy based on Vision Language Models (VLMs) to efficiently index documents purely from their visual features, allowing for subsequent fast query matching with late interaction mechanisms. Our method, ColPali, outperforms all other retrieval systems on ViDoRe while being fast and end-to-end trainable. We release models and code at [https://huggingface.co/vidore](https://huggingface.co/vidore).

## 2 Problem Formulation & Related Work

**Problem Setting.**
In our setting, a retrieval system scores how relevant a document \( d \) from corpus \( \mathcal{D} \) is with respect to a query \( q \). Computing the similarity score \( s(q,d)\in\mathbb{R}_+ \) for each of the \( |\mathcal{D}| \) documents in the corpus creates a ranking we can use to extract the most relevant documents. In this work, we focus on page-level retrieval: given a query, is the correct document page retrieved by the system? For coherence with existing literature, we further use the term document to refer to individual pages, i.e. the atomic retrieved elements in our setting. As we focus on practical industrial retrieval applications (RAG, search engines) with potentially large corpora sizes, latency constraints are imposed on scoring systems. Most current retrieval systems can be decomposed into (1) an offline indexation phase in which a document index is built and (2) an online querying phase in which a query is matched to documents from the index and where low latency is vital to the user experience.

Efficient document retrieval systems exhibit joint properties of high retrieval performance (R1), low latency during querying (R2), and high throughput during indexation (R3).

### 2.1 Textual Retrieval Methods

**Document Retrieval in Text Space.**
Statistical methods based on word frequency like TF-IDF and BM25 are still widely used due to their simplicity and efficiency. More recently, neural embedding models based on fine-tuned large language models display state-of-the-art performance on a variety of text embedding tasks and top the retrieval leaderboards.

**Neural Retrievers.**
In bi-encoder models, documents are independently mapped offline to a dense vector space. Queries are embedded online and matched to documents through a fast cosine distance computation.
A slower, but slightly more performant alternative, cross-encoder systems concatenate query and document as a single input sequence and iteratively attribute matching scores to each possible combination. This enables full attention computation between query and document terms but comes at the cost of computational efficiency, as \( |\mathcal{D}| \) encoding passes must be done online.

**Multi-Vector retrieval via late interaction.**
In the late interaction paradigm, an embedding is pre-computed and indexed per document token. At runtime, similarity can be computed with individual query token embeddings. The idea is to benefit from the rich interaction between individual query and document terms while taking advantage of the offline computation and fast query matching enabled by bi-encoders.

**Retrieval Evaluation.**
Although benchmarks and leaderboards have been developed to evaluate text embedding models, as previously stated, much of the performance improvements in industrial use cases of embedding models stem from the prior data ingestion pipeline. While documents often rely on visual elements to more efficiently convey information to human readers, text-only systems barely tap into these visual cues.

To our knowledge, no benchmark evaluates document retrieval methods by considering both textual and visual document features like a human would.

### 2.2 Integrating Visual Features

**Contrastive Vision Language Models.**
Mapping latent representations of textual content to corresponding representations of visual content has been done by aligning disjoint visual and text encoders through contrastive losses. While some OCR capabilities exist in these models, the visual component is often not optimized for text understanding.
The Fine-grained Interactive Language-Image Pre-training (FILIP) framework extends the late interaction mechanism to cross-modal vision-language models, relying on max similarity operations between text tokens and image patches.

**Visually Rich Document Understanding.**
To go beyond text, some document-focused models jointly encode text tokens alongside visual or document layout features.
Large Language transformer Models (LLMs) with strong reasoning capabilities have recently been combined with Vision Transformers (ViTs) to create VLMs where image patch vectors from contrastively trained ViT models are fed as input embeddings to the language model and concatenated with the text-token embeddings.

**PaliGemma.**
The PaliGemma-3B model extends concepts from Pali3, and projects SigLIP-So400m/14 patch embeddings into Gemma-2B’s text vector space. Along with its reasonable size w.r.t. other performant VLMs, an interesting property of PaliGemma’s text model is that it is fine-tuned with full-block attention on the prefix (instruction text and image tokens).

VLMs display enhanced capabilities in Visual Question Answering, captioning, and document understanding, but are not optimized for retrieval tasks.

## 3 The ViDoRe Benchmark

Existing benchmarks for contrastive vision-language models primarily evaluate retrieval for natural images. On the other hand, textual retrieval benchmarks are evaluated at the textual passage level and are not tailored for document retrieval tasks. We fill the gap with ViDoRe, a comprehensive benchmark for document retrieval using visual features.

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

**Academic Tasks.**
We repurpose widely used visual question-answering benchmarks for retrieval tasks: for each page-question-answer triplet, we use the question as the query, and the associated page as the gold document. These academic datasets either focus on single specific modalities or target more varied visually rich documents. Moreover, we consider TabFQuAD, a human-labeled dataset on tables extracted from French industrial PDF documents released with this work.

**Practical tasks.**
We construct topic-specific retrieval benchmarks spanning multiple domains to go beyond repurposed QA datasets and evaluate retrieval in more realistic industrial situations (e.g. RAG). To achieve this, we collect publicly accessible PDF documents and generate queries pertaining to document pages using Claude-3 Sonnet, a high-quality proprietary vision-language model. Queries are extensively filtered for quality and relevance by human annotators.

**Evaluation Metrics.**
We evaluate performance on our benchmark (Requirement R1) using standard metrics from the retrieval literature (NDCG, Recall@K, MRR). We report NDCG@5 values as the main performance metric in this work and release the complete sets of results along with the models. To validate compliance with practical industrial constraints, we also consider query latencies (R2) and indexing throughputs (R3).

### 3.2 Assessing Current Systems

**Unstructured.**
We evaluate retrieval systems representative of those found in standard industrial RAG pipelines. As is common practice, we rely on the Unstructured off-the-shelf tool in the highest resolution settings to construct high-quality text chunks from PDF documents. Unstructured orchestrates the document parsing pipeline, relying on deep learning vision models to detect titles and document layouts, OCR engines to extract text in non-native PDFs, specialized methods or models to detect and reconstruct tables, and implements a chunking strategy (by-title) that leverages the detected document structure to preserve section boundaries when concatenating texts. In our simplest Unstructured configuration (text-only), only textual elements are kept, and figures, images, and tables are considered noisy information and are filtered out.

**Unstructured + X.**
While Unstructured is a strong baseline by itself, we further augment Unstructured’s output by integrating the visual elements. In (+ OCR), tables, charts, and images are run through an OCR engine, processed by Unstructured, and chunked independently. In (+ Captioning), we set up a fully-fledged captioning strategy, in which we feed visual elements to a strong proprietary Vision Language Model (Claude-3 Sonnet) to obtain highly detailed textual descriptions of the elements.
Both strategies aim to integrate visual elements in the retrieval pipeline but incur significant latency and resource costs.

**Embedding Model.**
To embed textual chunks, we evaluate Okapi BM25, the de facto standard sparse statistical retrieval method, and the dense encoder of BGE-M3, a multilingual neural method with SOTA performance in its size category. Chunks are embedded and scored independently, and page-level scores are obtained by max-pooling over the page’s chunk scores.

**Contrastive VLMs.**
We also evaluate the strongest available vision-language embedding models; Jina CLIP, Nomic Embed Vision, and SigLIP-So400m/14.

**Results.**
Best results are obtained by combining the Unstructured parser with visual information, either from captioning strategies or by running OCR on the visual elements. Little difference is seen between BM25 and BGE-M3 embeddings highlighting the visual information bottleneck. Contrastive VLMs lag behind. Beyond retrieval performance (R1), the indexing latencies (R2) illustrate that PDF parsing pipelines can be very lengthy, especially when incorporating OCR or captioning strategies. Querying latencies at runtime (R3) are very good for all evaluated systems (≤22 ms on NVIDIA L4) due to fast query encoding and cosine similarity matching.

![Refer to caption](https://arxiv.org/html/2407.01449v2/x1.png)
Figure 3: Offline indexing with ColPali is much simpler and faster compared to standard retrieval methods. Indexing speeds reported are computed on Nvidia L4 GPUs.

## 4 Late interaction based Vision Retrieval

### 4.1 Architecture

**Vision-Language Models.**
Encouraged by their strong document understanding capabilities, we propose adapting recent VLMs for retrieval. The key concept is to leverage the alignment between output embeddings of text and image tokens acquired during multi-modal finetuning.
To this extent, we introduce ColPali, a Paligemma-3B extension that is capable of generating ColBERT-style multi-vector representations of text and images.
PaliGemma-3B is a strong candidate due to its small size, the many released checkpoints fine-tuned for different image resolutions and tasks, and the promising performances on various document understanding benchmarks.
We add a projection layer to map the output language modeling embeddings to a vector space of reduced dimension D=128 as used in the ColBERT paper to keep lightweight bag-of-embedding representations.

**Late Interaction.**
Given query \( q \) and document \( d \), we denote as \( \mathbf{E}_{q}\in\mathbb{R}^{N_q \times D} \) and \( \mathbf{E}_{d}\in\mathbb{R}^{N_d \times D} \) their respective multi-vector representation in the common embedding space \( \mathbb{R}^D \). The late interaction operator, \( \text{LI}(q,d) \), is the sum over all query vectors of its maximum dot product with each of the \( N_d \) document embedding vectors.

|     |     |     |     |
| --- | --- | --- | --- |
|  | \( \text{LI}(q,d)=\sum_{i\in [1,N_q]} \max_{j\in [1,N_d]} \langle \mathbf{E}_{q}^{(i)} \| \mathbf{E}_{d}^{(j)} \rangle \) |  | (1) |

**Contrastive Loss.**
The Late Interaction operation is fully differentiable, enabling backpropagation.  
Let a batch \( \{q_k,d_k\}_{k\in [1,b]} \) composed of \( b \) query-page pairs, where for all \( k\in [1,b] \), the document page \( d_k \) is the document corresponding to query \( q_k \).
Following Khattab and Zaharia, we define our in-batch contrastive loss \( \mathcal{L} \) as the softmaxed cross-entropy of the positive scores \( s_k^+ = \text{LI}(d_k, q_k) \) w.r.t. the maximal negative scores \( s_k^- = \max_{l,l\neq k} \text{LI}(q_k, p_l) \).

### 4.2 Model training

**Dataset.**  
Our training dataset of 127,460 query-page pairs is comprised of train sets of openly available academic datasets (63%) and a synthetic dataset made up of pages from web-crawled PDF documents and augmented with VLM-generated (Claude-3 Sonnet) pseudo-questions (37%). Our training set is fully English by design, enabling us to study zero-shot generalization to non-English languages. We explicitly verify no multi-page PDF document is used both in ViDoRe and in the train set to prevent evaluation contamination. A validation set is created with 2% of the samples to tune hyperparameters.

**Parameters.**
All models are trained for 1 epoch on the train set. Unless specified otherwise, we train models in bfloat16 format, use low-rank adapters (LoRA) with \( \alpha=32 \) and \( r=32 \) on the transformer layers from the language model, as well as the final randomly initialized projection layer, and use a paged_adamw_8bit optimizer. We train on an 8 GPU setup with data parallelism, a learning rate of \( 5e-5 \) with linear decay with 2.5% warmup steps, and a batch size of 32.

**Query Augmentation.**
As in Khattab and Zaharia, we append 5 <unused0> tokens to the query tokens to serve as a soft, differentiable query expansion or re-weighting mechanism.

## 5 Results

|     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | ArxivQ | DocQ | InfoQ | TabF | TATQ | Shift | AI | Energy | Gov. | Health. | Avg. |
| Unstructured Text only |  |  |  |  |  |  |  |  |  |  |  |
| - BM25 | - | 34.1 | - | - | 44.0 | 59.6 | 90.4 | 78.3 | 78.8 | 82.6 | - |
| - BGE-M3 | - | 28.4 | - | - | 36.1 | 68.5 | 88.4 | 76.8 | 77.7 | 84.6 | - |
| Unstructured + OCR |  |  |  |  |  |  |  |  |  |  |  |
| - BM25 | 31.6 | 36.8 | 62.9 | 46.5 | 62.7 | 64.3 | 92.8 | 85.9 | 83.9 | 87.2 | 65.5 |
| - BGE-M3 | 31.4 | 25.7 | 60.1 | 70.8 | 50.5 | 73.2 | 90.2 | 83.6 | 84.9 | 91.1 | 66.1 |
| Unstructured + Captioning |  |  |  |  |  |  |  |  |  |  |  |
| - BM25 | 40.1 | 38.4 | 70.0 | 35.4 | 61.5 | 60.9 | 88.0 | 84.7 | 82.7 | 89.2 | 65.1 |
| - BGE-M3 | 35.7 | 32.9 | 71.9 | 69.1 | 43.8 | 73.1 | 88.8 | 83.3 | 80.4 | 91.3 | 67.0 |
| Contrastive VLMs |  |  |  |  |  |  |  |  |  |  |  |
| Jina-CLIP | 25.4 | 11.9 | 35.5 | 20.2 | 3.3 | 3.8 | 15.2 | 19.7 | 21.4 | 20.8 | 17.7 |
| Nomic-vision | 17.1 | 10.7 | 30.1 | 16.3 | 2.7 | 1.1 | 12.9 | 10.9 | 11.4 | 15.7 | 12.9 |
| SigLIP (Vanilla) | 43.2 | 30.3 | 64.1 | 58.1 | 26.2 | 18.7 | 62.5 | 65.7 | 66.1 | 79.1 | 51.4 |
| Ours |  |  |  |  |  |  |  |  |  |  |  |
| SigLIP (Vanilla) | 43.2 | 30.3 | 64.1 | 58.1 | 26.2 | 18.7 | 62.5 | 65.7 | 66.1 | 79.1 | 51.4 |
| BiSigLIP (+fine-tuning) | 58.5 | 32.9 | 70.5 | 62.7 | 30.5 | 26.5 | 74.3 | 73.7 | 74.2 | 82.3 | 58.6 |
| BiPali (+LLM) | 56.5 | 30.0 | 67.4 | 76.9 | 33.4 | 43.7 | 71.2 | 61.9 | 73.8 | 73.6 | 58.8 |
| ColPali (+Late Inter.) | 79.1 | 54.4 | 81.8 | 83.9 | 65.8 | 73.2 | 96.2 | 91.0 | 92.7 | 94.4 | 81.3 |

Table 2: Comprehensive evaluation of baseline models and our proposed method on ViDoRe. Results are presented using NDCG@5 metrics, and illustrate the impact of different components. 

### 5.1 Performance (R1)

We iteratively construct ColPali, starting from an off-the-shelf SigLIP model.

**BiSigLIP: Improving a strong model.**
SigLIP is a strong vision-language bi-encoder model, pretrained on the English split of WebLI, a corpus of billions of image-text pairs. We find that SigLIP largely outperforms both Jina CLIP and Nomic-vision on document retrieval tasks. Further fine-tuning the textual component of this model on our document-oriented dataset (BiSigLIP) yields clear improvements across the board, particularly on figure retrieval (ArxivQA) and table retrieval tasks (TabFQuAD).

**BiPali: Pairing with a language model.**
In the PaliGemma model architecture, SigLIP-generated patch embeddings are fed to a text language model to obtain LLM contextualized output patch embeddings. We average pool these representations to obtain a single dense vector, effectively creating a PaliGemma bi-encoder model (BiPali). After fine-tuning on the training dataset, we obtain a model that performs slightly worse in English than the tuned BiSigLIP variant. However, we see notable improvements in French tasks, indicating that BiPali’s LLM (Gemma 2B) helps multilingual text understanding. This is particularly notable as our training dataset does not contain non-English samples.

**ColPali: Adding Late Interaction.**
One benefit of inputting image patch embeddings through a language model is that they are natively mapped to a latent space similar to textual input (query). This enables leveraging the ColBERT strategy to compute interactions between text tokens and image patches, which enables a step-change improvement in performance compared to BiPali.
Results show that our ColPali model also largely outperforms the strong baselines based on Unstructured and captioning, as well as all evaluated text-image embedding models. The difference is particularly stark on the more visually complex benchmark tasks, such as InfographicVQA, ArxivQA, and TabFQuAD representing respectively infographics, figures, and tables. However, text-centric documents are also better retrieved by the ColPali models across all evaluated domains and languages, making our approach the overall best-performing document-retrieval model.

**Negative Results.**
For extensiveness, we also train ColSigLIP, a late interaction variant of the BiSigLIP model but obtain abysmal performances. We attribute this to the large gaps w.r.t. SigLIP’s pre-training, in which only a pooled latent representation is used in the contrastive loss, which does not optimize the representations of individual patch and token embeddings. Similarly, we train a BiSigLIPPaliGemma variant, in which we retrieve the image representations from the SigLIP model that has been further updated by PaliGemma fine-tuning, and use the text representations from PaliGemma’s text model. After fine-tuning on our dataset, performance is severely inferior to SigLIPVanilla which simply encodes with SigLIP’s original text and vision components. This indicates a logical misalignment between SigLIP embeddings, and Gemma embeddings after PaliGemma training.

### 5.2 Latencies & Memory Footprint

**Online Querying (R2).**
Logically, querying latencies differ between ColPali and a BGE-M3 embedding model. For BGE, encoding takes about 22 ms for 15 tokens, while encoding a query with ColPali’s language model takes about 30 ms. For smaller corpus sizes, computing the late interaction operation induces marginally small overheads (~1 ms per 1000 pages in the corpus), and the cosine similarity computation between bi-encoder vectors is even faster. Optimized late interaction engines enable to easily scale corpus sizes to millions of documents with reduced latency degradations.

**Offline Indexing (R3).**
Standard retrieval methods using bi-encoders represent each chunk as a single vector embedding, which is easy to store and fast to compute. However, processing a PDF to get the different chunks is the most time-consuming part (layout detection, OCR, chunking), and using captioning to handle multimodal data will only exacerbate this already lengthy process. On the other hand, ColPali directly encodes pages from their image representation. Although the encoder model is larger than standard retrieval encoders, skipping the preprocessing allows large speedups at indexing.

**Memory Footprint.**
Our method requires storing a vector per image patch. We project each PaliGemma vector to a lower dimensional space (D=128) to maximize efficiency, leading to a memory footprint of 256 KB per page. Importantly, the memory footprint of the naive ColBERT indexing strategy can be drastically improved through compression and clustering mechanisms as proposed in the Performance-optimized Late Interaction Driver.

### 5.3 Interpretability

By superimposing the late interaction heatmap on top of the original image, we can visualize the most salient image patches with respect to each term of the query, yielding interpretable insights into model focus zones. As epitomized in [Figure 1](https://arxiv.org/html/2407.01449v2#S1.F1), we observe ColPali exhibits strong OCR capabilities as both the words "hourly" and "hours" present a high similarity score with the query token <\_hour>. We also note particular focus on other non-trivial image features such as the x-axis representing hours being salient. Other visualization examples with similar trends of the model transcending pure OCR are shown in [Appendix C](https://arxiv.org/html/2407.01449v2#A3).

## 6 Ablation study

![Refer to caption](https://arxiv.org/html/2407.01449v2/x2.png)
Figure 4: Relative NDCG@5 performance gain w.r.t. the default ColPali (1024 patches). TabFQuAD fine-tuning measures the performance difference on the TabFQuAD task after the introduction of targeted data in the training set. All other results refer to performance deltas averaged on all ViDoRe tasks.

**Should we scale models or patch numbers?**

We train a variant of PaliGemma with half the number of image patches (512). While there is a clear performance degradation w.r.t. to the 1024-patch ColPali model, memory usage is much lower.  
As an alternative to PaliGemma, we train Idefics2-8B, a VLM with a similar architecture and based on a Mistral-7B language backbone and a SigLIP vision encoder paired with a perceiver resampler.  
Our results suggest language model size has a strong impact on performance, and along with the trained resampler enables more efficient representations for smaller numbers of image embeddings - ColIdefics2 with 64 patches edges out ColPali with 512 patches.
Scaling the number of patches of the smaller ColPali model from 512 to 1024, enables largely surpassing the 60-patch ColIdefics2 while being about twice as fast in terms of training and inference latency.  
These results suggest there are tradeoffs between performance (R1), latencies during online querying (R2) and offline indexation phases (R3), and index memory size.

**Should we fine-tune the vision component?**

We run our contrastive finetuning on a ColPali model in which we also train the vision encoder and the projection layer. Results show this leads to no significant improvements.

**Do "query augmentation" tokens help?**

In ColBERT, special tokens are concatenated to the input query to serve as soft query augmentation buffers. Training without these tokens, we observe no significant performance difference in the English benchmarks. However, performance on the French tasks seems to improve.

**Is the Pairwise CE loss best?**

Training with an in-batch negative contrastive loss, instead of the pairwise CE loss that only considers the hardest negative sample, leads to a slight performance degradation (-2.4%) on the aggregated benchmark.

**Can the model adapt to new tasks?**

ColPali can be trained end-to-end, directly optimizing the downstream retrieval task which greatly facilitates fine-tuning to boost performance on specialized domains, multilingual retrieval, or specific visual elements the model struggles with. To demonstrate, we add 1552 samples representing French tables and associated queries to the training set. This represents the only French data in the training set, with all other examples being kept unchanged. We see significant NDCG@5 improvements and even starker Recall@1 gains (+6.63%) on the TabFQuAD benchmark, with no performance degradation on the rest of the benchmark tasks (+0.34%).

## 7 Conclusions

Through the conception of a new benchmark ViDoRe, we established the limits of both modern industrial document retrieval pipelines and off-the-shelf image-text contrastive models for visually rich document retrieval. We introduced ColPali, a novel retrieval model that leverages the latest generative Vision Language models to create highly performing multi-vector embeddings purely from visual document features. ColPali largely outperforms the best existing document retrieval methods while enabling faster corpus indexing time and maintaining low querying latencies, suggesting a very high potential for industrial document retrieval applications. We hope to encourage future work by publicly releasing the ViDoRe benchmark and all models and baselines from our study.

Future Work. Further performance gains could be obtained by exploring sub-image decomposition, optimal image patch resampling strategies, or hard-negative mining.  
Subsequently, our vision is to combine visual retrieval and visually grounded query answering to create RAG systems that purely function from visual features.
An interesting line of research could be attempting to generate answers leveraging information stored in the indexed multi-vector patch embeddings.

## Limitations

**Focus.**  
In this work, we evaluate models on document retrieval tasks, covering several modalities (figures, text, tables, infographics). We however primarily focus on PDF-type documents, and evaluating systems on image retrieval with documents stemming from web page screenshots or hand-written documents might be an interesting generalization. We also focus on high-resource languages (English and French) and although we have shown the capacity of the ColPali model to generalize to languages outside of its fine-tuning set, it is unclear how the model would perform on languages that are not as represented in the model’s language backbone. Finally, our setup assumes relevant documents exist, but abstention methods for Information Retrieval systems might be interesting to explore in more practical settings in which confidence estimation might be important.

**Support.**  
This work relies on multi-vector retrieving derived from the ColBERT late interaction mechanism. Although some vector databases support late interaction engines, many widely used vector retrieval frameworks do not propose native multi-vector support, and some engineering infrastructure efforts may be required to adapt them to work with ColPali (or ColBERT) models.

**Data.**  
In the creation of ViDoRe, we partially rely on synthetic query generation based on a commercial large language model, which may induce some amount of bias in the generated queries. To compensate for this, we have iterated on the prompting strategy and given real query examples to the models to help ground generation in realistic settings. We have further manually verified all synthetic queries through a lengthy process to validate their relevance and their quality. Our benchmark also includes many benchmark tasks with no synthetic data, and result trends observed between all tasks are correlated, further confirming the coherence of our benchmark design.

## Ethical Considerations

**Carbon Footprint.**  
Our work fully leverages prior pretrained models and training is not particularly compute-intensive. Furthermore, we rely on low-rank adapters to further reduce the computational resources needed, both during training and for storage. Overall, a training run represents about 40 hours of Mi250x AMD GPUs. Our experiments, in total, represent 1405 Mi250x GPU hours from highly efficient compute clusters running on low-carbon nuclear energy, representing a total of around 15kg CO2 eq.

**Impact.**  
We believe our work could have a strong impact on improving industrial document retrieval systems. Our method is efficient, performs well, and the additional support towards visually rich information from documents could go a long way in unlocking knowledge sources previously difficult to index or query.

**Resource Release.**  
For transparency, and to foster future work, we release our comprehensive benchmark under open license and host a public leaderboard. Our models are released under the same usage license as the base model (Gemma Research license for ColPali, Apache2.0 for ColIdefics2) and should be used as intended by the VLM license.

---

## Appendix A Benchmark Datasets

### A.1 Academic Datasets

DocVQA includes collected images from the UCSF Industry Documents Library. Questions and answers were manually annotated.

InfoVQA includes infographics collected from the Internet using the search query “infographics”. Questions and answers were manually annotated.

TAT-DQA is a large-scale Document VQA dataset that was constructed from publicly available real-world financial reports. It focuses on rich tabular and textual content requiring numerical reasoning. Questions and answers were manually annotated by human experts in finance.

arXivQA is a VQA dataset based on figures extracted from arXiv publications. The questions were generated synthetically using GPT-4 Vision.

TabFQuAD (Table French Question Answering Dataset) is designed to evaluate TableQA models in realistic industry settings. Additional queries were created to augment the existing human-annotated ones.

### A.2 Practical Datasets

**Methodology.**  
Creating a relevant retrieval dataset close to real use cases is a major challenge as the dataset needs to be both sufficiently large for effective fine-tuning and sufficiently diverse to cover a broad range of modalities (full text, tables, charts, …), domains (industry, healthcare, …), and query-document interactions (extractive questions, open-ended questions, …). Our approach to building this dataset involves several steps: (1) we use a web crawler to collect publicly available documents on various themes and sources, (2) we convert these PDFs into a series of images, one per page, and (3) we generate queries related to each image using a VLM.

**Web-Crawler.**  
We implemented a web crawler to efficiently collect large volumes of documents related to a given topic. The crawler is seeded with a user-defined query (e.g. "artificial intelligence") and then uses GPT-3.5 Turbo to brainstorm related topics and subtopics. This query augmentation strategy aims at both broadening and deepening the search. GPT-3.5 Turbo is further used to generate diverse search queries from each subtopic. This query set is then consumed by a pool of parallel workers whose job is to fetch the associated most relevant documents. We use SerpAPI along with a filetype filter (PDF documents only) to programmatically scrape Google Search rankings. Each file is hashed and stored in a Bloom filter shared among workers to avoid duplicate documents in the final corpus. Unique scraped files are downloaded, and inserted into a SQLite database along with additional metadata.

**Datamix.**  
Using the web crawler, we collected approximately 1,000 documents for each of the following four seeds: "energy", "government reports", "healthcare industry", and "artificial intelligence". These seeds were meticulously hand-picked to align with real-use cases for retrieval models and visually rich pages. We also removed all documents containing any private information. At this stage, we randomly selected 900 files for the training set and 100 files for the test set, ensuring that data leakage into the test set was avoided during subsequent processing steps.

**Query Generation.**  
To increase the efficiency of our query generation scheme and to limit API calls, we generate at most 3 questions per image. From all the documents collected, we randomly sample 10,000 images per theme and call Claude-3 Sonnet with the following prompt:

>You are an assistant specialized in Multimodal RAG tasks.
>The task is the following: given an image from a pdf page, you will have to generate questions that can be asked by a user to retrieve information from a large documentary corpus. The question should be relevant to the page, and should not be too specific or too general. The question should be about the subject of the page, and the answer needs to be found in the page.
>Remember that the question is asked by a user to get some information from a large documentary corpus that contains multimodal data. Generate a question that could be asked by a user without knowing the existence and the content of the corpus.
>Generate as well the answer to the question, which should be found in the page. And the format of the answer should be a list of words answering the question.
>Generate at most THREE pairs of questions and answers per page in a dictionary with the following format, answer ONLY this dictionary NOTHING ELSE:
>{ "questions": [ {"question": "XXXXXX", "answer": ["YYYYYY"] }, ... ] }
>where XXXXXX is the question and ['YYYYYY'] is the corresponding list of answers that could be as long as needed.
>Note: If there are no questions to ask about the page, return an empty list. Focus on making relevant questions concerning the page.
>Here is the page:

**Human Validation.**  
We manually validate every single synthetically created query in ViDoRe to ensure quality, query relevance, and consistency with the benchmark objective of evaluating retrieval in practical industrial settings. During this step, we randomly assign document-pair queries to 4 volunteer annotators and instruct them to filter out queries that do not fit the above-listed criteria. 

## Appendix B Implementation details

### B.1 Codebase

The codebase is written in PyTorch and leverages HuggingFace tooling for model implementations and trainers.

### B.2 Pairwise CE loss

Our in-batch contrastive loss \( \mathcal{L} \) is defined as the softmaxed cross-entropy of the positive scores \( s_k^+ = \text{LI}(d_k, q_k) \) w.r.t. the maximal negative scores \( s_k^- = \max_{l,l\neq k} \text{LI}(q_k, p_l) \).

For numerical stability, we reformulate the loss with the softplus function, leading to:

|     |     |     |     |
| --- | --- | --- | --- |
|  | \( \mathcal{L} = \frac{1}{b} \sum_{k=1}^{b} \texttt{softplus}(s_k^- - s_k^+) \) |  | (2) |

### B.3 Hyperparameters

Hyperparameters are tuned on a validation split composed of 2% of the training dataset. We find bi-encoder methods to be more sensible to learning rate variations than late interaction-based models and achieve the best performance for all models with a learning rate of 5e-5. We experiment with LoRA rank and \( \alpha \) values and do not notice particular improvements past \( r = \alpha = 32 \). Per-device batch sizes are kept small due to long sequence lengths that complicate scaling past \( b=4 \). Simulating larger batch sizes for in-batch negative sampling should enable even better results. We find the best results with global batch size \( b=32 \) for 1 epoch on our training set.

### B.4 Embedding size

Minimizing storage footprint can be essential to industrial retrieval systems if databases contain millions of documents. With this criterion in view, we have compared the embedding sizes of the models in our study.

| Model                   | Embedding size (KB)      |
|-------------------------|-------------------------|
| BGE-M3                  | 8.60                    |
| BM25 (dense emb.)       | 3.00                    |
| BM25 (sparse emb.)      | 1.56 ± 0.51             |
| ColPali (float16)       | 256                     |

Table 3: Comparison of the embedding sizes for the DocVQA test set from ViDoRe w.r.t. different retrieval models. The lower the size the smaller the storage footprint of the model.

### B.5 Latency computations

All latency computations are done on a NVIDIA L4 GPU. Queries are encoded independently (batch size of 1) to simulate online querying, and pages are encoded with a batch size of 4 for PaliGemma derived models, and 8 for BGE-M3. Reported times include image and text processing time before the model forward pass, as well as query-to-index matching times. Query latency experiments are averaged over 1000 queries, and indexing times are measured for a 100 page document.

### B.6 Captioning

Examples of captions generated for visually rich document chunks with Claude-3 Sonnet are shown below. The prompt used for generating the description is the following:

>You are an assistant specialized in document analysis. Given a table or a figure, you have to provide a detailed summary of the content in maximum 3000 characters. Your summary should be qualitative and not quantitative. Here is the table/figure to analyze: {image}. Answer ONLY with the caption of the table/figure.

![Refer to caption](https://arxiv.org/html/2407.01449v2/x3.png)
Figure 5: Example from the "Energy" test set.

*Caption:* The image depicts the hourly energy generation profile, illustrating the contributions of various energy sources over 24 hours. The data is presented as a stacked bar chart, with the x-axis representing the hours of the day from 1 to 2, and the y-axis showing the average hourly generation in MW. The bars are segmented into different colors, each representing a distinct energy source: nuclear, bio, geothermal, solar, wind, hydro, natural gas, and other imports. The chart provides insights into the temporal variations in energy generation across different sources, highlighting the interplay between baseload and intermittent sources throughout the day.

![Refer to caption](https://arxiv.org/html/2407.01449v2/x4.png)
Figure 6: Example from the "Government Reports" test set.

*Caption:* The image shows a table titled "System of Record" which outlines the different types of documents or records maintained across various systems or departments within an organization related to project management and construction. The rows list documents like project plans, budgets, schedules, contracts, purchase orders, invoices, change requests, bid submissions, drawings, manuals, meeting minutes, and reports. The columns indicate the system or department responsible for maintaining each record, such as County Servers, Project View, OnBase, CGI Advantage Financial System, and Purchasing Department. The table uses "W" and "T" markers to denote which system or department serves as the primary source (writer) or storage location (trailer) for each type of document.

## Appendix C More similarity maps

In [Figure 7](https://arxiv.org/html/2407.01449v2#A3.F7), ColPali assigns a high similarity to all patches with the word "Kazakhstan" when given the token <\_Kazakhstan>. Moreover, our model seems to exhibit world knowledge capabilities as the patch around the word "Kashagan" - an offshore oil field in Kazakhstan - also shows a high similarity score. On the other hand, in [Figure 8](https://arxiv.org/html/2407.01449v2#A3.F8), we observe that ColPali is also capable of complex image understanding. Not only are the patches containing the word "formulations" highly similar to the query token _formula, but so is the upper-left molecule structure.

![Refer to caption](https://arxiv.org/html/2407.01449v2/extracted/5705564/images/similarity_maps/similarity_map_kazakhstan.png)
Figure 7: Similarity of the image patches w.r.t. the underlined token in the user query. This example is from the Shift test set.

![Refer to caption](https://arxiv.org/html/2407.01449v2/extracted/5705564/images/similarity_maps/similarity_map_ferroelectrics.png)
Figure 8: Similarity of the image patches w.r.t. the underlined token in the user query. This example is from the Healthcare Industry test set.

It is also interesting to highlight that both similarity maps showcase a few white patches with high similarity scores. This behavior might first seem surprising as the white patches should not carry a meaningful signal from the original images. We believe the vectors associated with these patches share a similar role with the ViT registers, i.e. these patches were repurposed for internal computations and stored the global information from the whole image.

## Appendix E ViDoRe examples

### Energy

Query: What types of accounts or products allow investors to defer paying taxes?  
![Refer to caption](https://arxiv.org/html/2407.01449v2/extracted/5705564/images/dataset_samples/energy_1.jpeg)

Query: What is the projected peak electricity demand in California for the year 2030?  
![Refer to caption](https://arxiv.org/html/2407.01449v2/extracted/5705564/images/dataset_samples/energy_2.jpeg)

Query: What is the estimated total savings for a PV system in Durham under the net metering (flat rate) billing option over the system’s useful life of 25 years?  
![Refer to caption](https://arxiv.org/html/2407.01449v2/extracted/5705564/images/dataset_samples/energy_3.jpeg)

### Artificial Intelligence

Query: What are some common outcome areas targeted by TAII for different age groups?  
![Refer to caption](https://arxiv.org/html/2407.01449v2/extracted/5705564/images/dataset_samples/ai_1.jpeg)

Query: What did the robot monitor to determine when to activate or deactivate the blower motor and blinker?  
![Refer to caption](https://arxiv.org/html/2407.01449v2/extracted/5705564/images/dataset_samples/ai_2.jpeg)

Query: What is the key approach used in the PDP architecture?  
![Refer to caption](https://arxiv.org/html/2407.01449v2/extracted/5705564/images/dataset_samples/ai_3.jpeg)

### Healthcare Industry

Query: What is the chemical formula for the ferroelectric material Lead Zirconium Titanate (PZT)?  
![Refer to caption](https://arxiv.org/html/2407.01449v2/extracted/5705564/images/dataset_samples/healthcare_1.jpeg)

Query: What government entities are involved in public financing for healthcare in the US?  
![Refer to caption](https://arxiv.org/html/2407.01449v2/extracted/5705564/images/dataset_samples/healthcare_2.jpeg)

Query: What does the AVPU scale stand for in assessing the level of consciousness of a seriously ill child?  
![Refer to caption](https://arxiv.org/html/2407.01449v2/extracted/5705564/images/dataset_samples/healthcare_3.jpeg)

### Government Reports

Query: What are some mandates for the EPA under the Pollution Prevention Act?  
![Refer to caption](https://arxiv.org/html/2407.01449v2/extracted/5705564/images/dataset_samples/gov_1.jpeg)

Query: What is the strategy of KPMG Hazem Hassan?  
![Refer to caption](https://arxiv.org/html/2407.01449v2/extracted/5705564/images/dataset_samples/gov_2.jpeg)

Query: What is the trust signal score for the consumer industry best-in-class archetype?  
![Refer to caption](https://arxiv.org/html/2407.01449v2/extracted/5705564/images/dataset_samples/gov_3.jpeg)

### Shift

Query: Selon le graphique, quelle est la capacité d’import et la consommation réelle de carburants SAF (biocarburants durables pour l’aviation) prévues en 2050 ?  
![Refer to caption](https://arxiv.org/html/2407.01449v2/extracted/5705564/images/dataset_samples/shift_1.jpeg)

Query: Quelle partie de la production pétrolière du Kazakhstan provient de champs en mer ?  
![Refer to caption](https://arxiv.org/html/2407.01449v2/extracted/5705564/images/dataset_samples/shift_2.jpeg)

Query: Quels sont les pays ayant la plus grande part des découvertes cumulées de pétrole brut en 2020 (en milliers de barils, hors découvertes cumulées) ?  
![Refer to caption](https://arxiv.org/html/2407.01449v2/extracted/5705564/images/dataset_samples/shift_3.jpeg)

### Original URL
https://arxiv.org/html/2407.01449v2
</details>

---
<details>
<summary>Image understanding  |  Gemini API  |  Google AI for Developers</summary>

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

![A table with cupcakes, with the wooden and glass objects highlighted](https://ai.google.dev/static/gemini-api/docs/images/segmentation.jpg)An example segmentation output with objects and segmentation masks

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

### Original URL
https://ai.google.dev/gemini-api/docs/image-understanding
</details>

---
<details>
<summary>ColPali: Redefining Multimodal RAG with Gemini</summary>

**ColPali** multimodal RAG offers a novel approach for efficient retrieval of elements such as images, tables, charts, and texts by treating each page as an image. This method takes advantage of Vision Language Models (VLM) to understand intricate details in complex documents like financial reports, legal contracts, and technical documents.

[![ColPali: Enhancing Financial Report Analysis with Multimodal RAG and Gemini](https://learnopencv.com/wp-content/uploads/2024/09/Feature-Multimodal-RAG-with-ColPali-Gemini.gif)](https://learnopencv.com/multimodal-rag-with-colpali/)

In documents like these, the accuracy of facts and figures from sources is paramount, as they can directly influence the decisions of investors and stakeholders. Unlike traditional retrieval systems, which may struggle with these elements, ColPali is an excellent contender for a production ready retrieval solution.

We will explore and test this through a much demanding industrial use case by building a **Multimodal RAG application with Colpali and Gemini on finance reports**. Specifically we will examine how to analyze a **10-Q quarterly report**, a critical financial document filed by companies with the U.S. Securities and Exchange Commission (SEC).

The articles primarily discusses:

- Challenges in Processing Unstructured Elements
- Why is ColPali needed?
- Composition of ColPali and how does it work for Multimodal RAG?
- Building a financial report analysis application using ColPali and Gemini
- ViDoRe Benchmark

Individuals and companies looking to enhance their document analysis capabilities with RAG will find this read more useful. As the GenAI space evolves rapidly, for firm’s seeking reliable solutions over mere hype, ColPali is definitely worth exploring.

## **Challenges in Processing Unstructured Elements**

> _**In practical industrial settings, the main performance bottleneck for efficient document retrieval is not in embedding model performance but in the prior data ingestion pipeline.**_
>
> _– **ColPali Paper 2024**_

Consider the case where you need to index a PDF of a financial report containing unstructured elements tables, images, graphs, charts for Multimodal [Retrieval Augmented Generation](https://learnopencv.com/rag-with-llms/). Unlike structured elements modern retrieval systems involve several steps to ensure high quality retrieval. [OCR](https://learnopencv.com/trocr-getting-started-with-transformer-based-ocr/) models extract text elements, Layout detectors like [YOLOX](https://learnopencv.com/yolox-object-detector-paper-explanation-and-custom-training/) detect individual element types into various [document segmentations](https://learnopencv.com/deep-learning-based-document-segmentation-using-semantic-segmentation-deeplabv3-on-custom-dataset/) such as table, charts, figure etc. Elements such as narrative text, table, title, figure caption, image, headers, footers, graph etc. are obtained within a raw list. Tools like Unstructured.io use models such as [tesseract-ocr](https://learnopencv.com/deep-learning-based-text-recognition-ocr-using-tesseract-and-opencv/), **microsoft/table-transformer-structure-recognition (TATR) and YOLOX** internally for table and layout detection.

Using a Multimodal LLM like Gemini, [text summaries](https://learnopencv.com/text-summarization-using-t5/) or figure captions of the images and tables are generated and their embeddings are stored in Vector DB’s such as [Qdrant](https://learnopencv.com/recommendation-system-using-vector-search/), Weaviate, or Pinecone.

During the retrieval stage, dense embedding with fixed output dimension is used to encode user query and results are retrieved based on nearest neighbor distance or cosine similarity. However the quality of the retrieval can be inconsistent and often requires a lot of manual inspection. Curated sets of document chunks with optimal chunk size and chunk overlap length also play a crucial role extracting coherent information about each element.

The shortcomings of current text-centric systems include,

- Complex data extraction and ingestion pipeline with [OCR](https://learnopencv.com/fine-tuning-trocr-training-trocr-to-recognize-curved-text/), chart and layout detection.
- High latency and the need for extensive pre and post processing data curation manually.
- Often poor retrieval results due to loss of context and lack of interaction between elements on a page, as each element type is extracted and indexed separately.

Through our experimentation, we got average to poor results particularly in retrieving table elements, even when searching with exact keywords titles. The quality of retrieval can be improved, particularly in preparing the table and image summaries before ingesting. However this is subjected to further experimentation.

[![ColPali architecture - Standard Retrieval v/s ColPali - Comparison -  How do vision language models (VLM) work for document analysis?](<Base64-Image-Removed>)](https://learnopencv.com/wp-content/uploads/2024/09/ColPal-v-s-Standard-Retrieval-Multimodal-RAG-1.png) **FIG 1**: Standard Retrieval v/s ColPali

## **Why is ColPali needed?**

> _**Instead of trying to transform human documents into something LLMs understand, we should make LLMs understand documents the way humans do**._
>
> _**– Source:**_ [**_Langchain community_**](https://www.reddit.com/r/LangChain/comments/1fc15wg/stop_trying_to_parse_your_documents_and_use/)

As humans, we are naturally visual creatures, often interacting with our environment through visual cues and interpreting information more effectively through images, tables, and diagrams. There’s a saying, “ **A picture is worth a thousand words**” and in the context of Vision Language Models (VLMs), a picture can indeed be worth thousands of words in terms of the data and insights it can convey.

> _**Documents are visually rich structures that convey information through text, as well as tables, figures, page layouts, or fonts.**_
>
>  – _**ColPali Paper 2024**_

In a similar fashion, ColPali enables VLMs like **PaliGemma 3B** to process this rich information not just as text. Rather than breaking down documents into individual isolated components, ColPali enables the model to interpret the entire page as an image, essentially maintaining the underlying context and preserving the layout intact. As a result, ColPali outperforms all SOTA PDF retrieval methods and is end-to-end trainable making it stand out in the RAG or search engines arena.

## **Composition of ColPali**

ColPali developed by Manuel Faysee et al., combines **ColBERT’s late interaction mechanism with PaliGemma**, a Vision LLM to efficiently extract elements from a document by treating each page as an image.

**ColBERT**

But what is [ColBERT](https://arxiv.org/pdf/2004.12832)? ColBERT, is a ranking model based on contextualized late interaction over [BERT](https://learnopencv.com/fine-tuning-bert/). This differs from traditional models wherein a query is compressed into a single vector potentially losing semantically rich information. In ColBERT, every token in the query interacts with all the document embeddings through late interactions preserving granular details of the query.

Mathematically, the LI(q,d) is the late interaction operator, sums over all the query vectors ( `Eqi` ) of maximum dot product with each document embedding vectors `Edj`.

[![ColBERT late interaction  Formula - ColPali architecture](<Base64-Image-Removed>)](https://learnopencv.com/wp-content/uploads/2024/09/Colbert-Embeddings-Late-Interaction-Formula.png)

[![ColBERT late interaction - ColPali architecture- Multimodal RAG with ColPali](<Base64-Image-Removed>)](https://learnopencv.com/wp-content/uploads/2024/09/Colbert-Embeddings-Late-Interaction-Online-Querying-ColPali.png) **FIG 2**: ColBERT Embeddings

ColBERT achieves a high Mean Reciprocal Rank ( `MRR@10`), in retrieval comparable to highly performant [BERT](https://learnopencv.com/bert-bidirectional-encoder-representations-from-transformers/) based models but with reduced computation cost and less latency. ColBERT based embeddings can be more effective is by using a single embedding to retrieve a set of candidate chunks ( **Fast LookUp**) and late interactions ( **Rerank**) effectively reducing the cost.

[![ColBERT Embeddings v/s Dense Embeddings v/s Keyword Embeddings - ColPali architecture](<Base64-Image-Removed>)](https://learnopencv.com/wp-content/uploads/2024/09/Colberrt-Embeddings-Comparison-to-Dense-and-Keyword-Embeddings-ColPali.jpeg) **FIG 3**: ColBERT Embeddings v/s Dense Embeddings

Source: [Leonie Monigatti](https://www.linkedin.com/posts/804250ab_heres-why-colbert-embeddings-are-all-the-activity-7237433301865492483-uay7?utm_source=share&utm_medium=member_desktop)

**PaliGemma**

On the other hand, **PaliGemma** is a VLM composed of **400M SigLIP Vision Encoder** with **Gemma 2B** as language model.

[![PaliGemma - Vision language models (VLM) - ColPali architecture - Multimodal RAG with ColPali](<Base64-Image-Removed>)](https://learnopencv.com/wp-content/uploads/2024/09/SigLIP-PaliGemma-ColPali-Multimodal-RAG-Gemini-Financial-RAG.png) **FIG 4**: PaliGemma Architecture

**Note**: SigLiP is a successor to CLIP utilizing sigmoid loss instead of contrastive loss. To learn more about [Training a CLIP like model from Scratch](https://learnopencv.com/clip-model/) bookmark our article for a later read.

PaliGemma is a lightweight VLM, with robust performance across wide range of tasks like short video caption, visual question answering, text reading, object detection and object segmentation. PaliGemma is a single-turn VLM and is not suited for conversation.

## **How does ColPali Work?**

[![ColPali architecture -Multimodal RAG using ColPali for financial reports](<Base64-Image-Removed>)](https://learnopencv.com/wp-content/uploads/2024/09/ColPali-Internals-Offline-Indexing-Online-Querying-Gemini-1.png) **FIG 5**: ColPali Workflow – Offline Indexing & Online Querying

The ColPali pipeline can be broken into two main phases:

1. **Offline Indexing**

- All the document pages are processed into 1030 patches. These flattened image patches of 128 dimensions each, are fed into the SigLIP vision encoder.
- An intermediate projection layer between SigLIP and Gemma2B projects the image tokens into a shared embedding space.
- These image patches are then passed into Gemma 2B decoder to generate contextualized representation about the image tokens.
- An additional projection layer maps the output of Gemma 2B language model embeddings into a lower dimensional ( `D = 128`) vector space, similar to approach in the ColBERT Paper to create lightweight bag-of-embedding representations.
- These embeddings are then indexed either locally or in Vector DB’s that natively supports ColBERT style embeddings such as [Vespa](https://blog.vespa.ai/retrieval-with-vision-language-models-colpali/), LintDB etc.

[![ColPali Latency Comparison - How is ColPali different from the Unstructured library?](<Base64-Image-Removed>)](https://learnopencv.com/wp-content/uploads/2024/09/Offline-Indexing-ColPali-v-s-Pdf-Parser-Time-Comparison.png) **FIG 6**: ColPali Latency Comparison

The paper reports that a typical PDF Parsers takes about 7.22s during offline indexing while ColPali takes just 0.37s.

2. **Online Querying**

- The query is encoded by the language model on-the-fly.
- A late interaction mechanism similar to the ColBERT ranking model (Khattab and Zaharia, 2020) computes the maximum similarity score between query embeddings and pre- indexed document embeddings.
- Finally, ColPali returns top k similar results(as images), which can be then fed into any Multimodal LLM like Gemini along with user query to get an interpretable response about the table, text or figures.

During the online querying phase, standard retrievers with the BGE embedding model takes about 22ms to encode 15 tokens while ColPali is relatively slow taking around **30 ms/query.**

Until now, we have understood all the nitty gritty details about ColPali’s internal workings. Using this knowledge let’s build an interesting RAG application for 10-Q reports of companies. At a high level our application workflow, looks as follows,

## **Building a Financial Report Analysis App using ColPali and Gemini**

### **Code Walkthrough**

#### Installing Dependencies:

- `colpali-engine and mteb` – For running ColPali engine and benchmarking with MTEB.
- `Transformers` – to load and work with pre-trained transformer models.
- `bitsandbytes` – enables efficient loading of models in 8 bits and 4 bits quantization.
- `einops` – for flexible tensor manipulation, to simplify operations for complex data structures.
- `google-generativeai` – Provides access to Gemini models via API from Google AI Studio.
- `pdf2image` – to convert pdf pages to images list.

We will need to also install `poppler-utils`, an essential package for manipulating PDF files and converting them to other formats.

#### **Import Dependencies**

- AutoProcessor : from the transformers library, is used to automatically load and preprocess data for the model.
- ColPali from paligemma colbert architecture combines PaliGemma VLM with ColBERT’s late interaction mechanism.
- CustomEvaluator from colpali engine helps to evaluate the retrieval performance on accuracy and relevance.
- process_images, process_queries utility functions transform images and queries expected by the ColPali model.

### **Load Model from Hugging Face 🤗**

We will set our HuggingFace Access Token as an environment variable to access the PaliGemma model as it is a gated model. For this we will use `google/paligemma-3b-mix-448` trained on a mixture of downstream tasks like Segmentation, OCR, VQA, Captioning etc.

Let’s load the fine-tuned lora adapter of PaliGemma-3b in `bfloat16` precision occupying around 6 GB vRAM.

Let’s download some quarterly financial pdf files from official websites of U.S Tech firms which we will be using later.

1. **Offline Indexing**

The following function index() is responsible for offline indexing by taking a set of pdf files and converting them to images into a list using the `pdf2image.convert_from_path` package to encode into image embeddings.

In order to efficiently process large datasets (here images), we implement batch processing using a naive PyTorch DataLoader with a batch size of 4. This allows us to handle multiple images in batches, optimizing speed and optimal memory usage.

To ensure the compatible input to the ColPali vision model (SigLIP), we use ColPali AutoProcessor to resize images to `448`, normalize and convert them into tensors. Then ColPali encodes image patches into embeddings, unbind them into individual vectors, and offloads to cpu. The model’s output is processed in inference mode without gradient computation ( `torch.no_grad( )` ). Including model loading, the indexing phase demands 10.5GB vRAM in Colab.

2. **Online Querying**

In the online querying phase, search queries are processed using process_queries() by the language model (Gemma 2B) to convert into embeddings. Similar to the indexing phase the embedding tokens are saved into a list of query tokens qs.

The `process_queries()` takes a paligemma_processor object, query and a dummy image of size 448 with white pixels as a placeholder object. VLMs like PaliGemma are designed and trained to take both query and image simultaneously. Even though we are only interested in passing a text query, a placeholder or blank image is passed along to meet the model’s input structure, ensuring compatibility and preventing errors during inference.

To evaluate the most relevant document page, the query embeddings (qs) and the pre-computed document embeddings (ds) are compared using a CustomEvaluator.

In `CustomEvaluator()` class, `is_multi_vector = True` parameter represents multiple vectors for each query indicating colbert style embeddings and late interaction mechanism.

The top k best page indices of images list, with maximum scores are retrieved by retrieve_top_document().

### Configuring Gemini LLM API

For a better understanding about complex financial tables and image elements like these, we will use one of the most versatile multimodals, **Gemini-Flash** from Google AI Studio. By signing in, developers get **1500 API calls per day** for Gemini-Flash and Gemini 1.0 pro. We also get access to text embedding models which will be part of a typical RAG pipeline.

We will set configs like perplexity, top_k and maximum generation tokens. According to the need and nature of the task, you can adjust these configurations.

Now, everything is set to use Gemini-Flash via API

### Perform Multimodal RAG with ColPali

The `get_answer()` function takes in a prompt and the best image retrieved by ColPali, which is then passed to Gemini Flash to generate a response.

Finally, we define the utility function where we call the retrieval_top_document() to return best_image and its corresponding index in the total images list.

Let’s put it all together. We will provide a prompt to Gemini Flash for proper interpretation about the best retrieved image. For better analysis and interpretation, we will need to set detailed instructions in the prompt.

**Most relevant image index is `7`**

[![ColPali Retrieved Image - Showing the balance sheet of Alphabet Inc -  How do vision language models (VLM) work for document analysis?- Multimodal RAG with ColPali](<Base64-Image-Removed>)](https://learnopencv.com/wp-content/uploads/2024/09/Alphabet-Inc-Balance-Sheet-ColPali-Retrieved-Image-Multimodal-RAG-Gemini-1.png) **FIG 7**: _Search Query_– Alphabet Balance Sheet

**Gemini Response**

|     |
| --- |
| `The image shows the consolidated balance sheets of Alphabet Inc.`<br>`for the periods ending December 31, 2023, and March 31, 2024. `<br>`The balance sheet is divided into three sections: assets, liabilities, `<br>`and stockholders' equity.The assets section includes current assets, `<br>`non-current assets, and total assets. The liabilities section includes `<br>`current liabilities, long-term debt, and total liabilities. The`<br>`stockholders equity section includes preferred stock, common stock,`<br>`accumulated other comprehensive income (loss), retained earnings,`<br>`and total stockholders' equity. The balance sheet shows that `<br>`Alphabet Inc. had total assets of $402,392 million as of December 31,`<br>`2023, and $407,350 million as of March 31, 2024. The company had`<br>`total liabilities of $119,013 million as of December 31, 2023, and `<br>`$114,506 million as of March 31, 2024. The company had total `<br>`stockholders' equity of $283,379 million as of December 31, 2023, `<br>`and $292,844 million as of March 31, 2024.` |

## **Additional Testing**

**Type 1: Chart Interpretation**

**Search Query**: Nvidia Fiscal Shareholder  Returns

**ColPali Retrieved Image**

[![ColPali Retrieved Image - Nvidia Shareholder Returns highlights from Q10 report -  How to analyze financial documents using vision language models](<Base64-Image-Removed>)](https://learnopencv.com/wp-content/uploads/2024/09/Nvidia-fiscal-report-ColPali-Retrieval-Financial-RAG-1.png)

**Gemini Response**

[![Gemini response for bar chart interpretation showing the shareholder returns for Nvidia- Gemini large language model (LLM) - Multimodal retrieval augmented generation tutorial](<Base64-Image-Removed>)](https://learnopencv.com/wp-content/uploads/2024/09/Nvidia-Chart-Interpretation-Gemini-Financial-RAG.png)

**Type 2: Image Interpretation**

**Search Query**: Tesla Optimus

**ColPali Retrieved Image**

[![Tesl Optimus Image from Tesla Q10 2023 reports - How do vision language models (VLM) work for document analysis? - Document retrieval using AI ](<Base64-Image-Removed>)](https://learnopencv.com/wp-content/uploads/2024/09/Tesla-Optimus-ColPali-Retrieval-Financial-RAG-1.png)

**Gemini Response**

[![Gemini response for Image retrieved by ColPali in the document Gemini large language model (LLM) - Can large language models analyze financial data accurately?](<Base64-Image-Removed>)](https://learnopencv.com/wp-content/uploads/2024/09/Tesla-Optimus-Image-Interpretation-Gemini-Financial-RAG.png)

**Type 3: Table Interpretation**

**Search Query**: Describe Microsoft Balance Sheet

**ColPali Retrieved Image**

[![ColPali retrieved Image of Microsoft Balance Sheet - Can large language models analyze financial data accurately? - Multimodal RAG with ColPali](<Base64-Image-Removed>)](https://learnopencv.com/wp-content/uploads/2024/09/Microsoft-balance-sheet-colpali-Retrieval-Financial-RAG-1.png)

**Gemini Response**

[![Gemini Response for ColPali retrieved image from Microsoft Q10 reports document showing their balance sheet - Table Interpretation - Multimodal RAG using ColPali for financial reports](<Base64-Image-Removed>)](https://learnopencv.com/wp-content/uploads/2024/09/Microsoft-Table-Interpretation-Gemini-Financial-RAG.png)

**Type 4: Graph Interpretation**

**Search Query**: Nvidia’s Comparison of 5 Year Cumulative Total Return

**ColPali Retrieved Image**

[![ColPali Retrieval Image  of  Nvidia’s Comparison of 5 Year Cumulative Total Return- How to analyze financial documents using vision language models](<Base64-Image-Removed>)](https://learnopencv.com/wp-content/uploads/2024/09/Nvidia-graph-Interpretation-ColPali-Gem9ni-Multimodal-Financial-RAG.png)

**Gemini Response**

[![Gemini Response of trendline interpretation  Nvidia’s Comparison of 5 Year Cumulative Total Return - Multimodal RAG using ColPali for financial reports - Graph Interpretation](<Base64-Image-Removed>)](https://learnopencv.com/wp-content/uploads/2024/09/Nvidia-Graph-Interpretation-Gemini-Financial-RAG.png)

### **ColPali Training Details From Paper**

The dataset comprises 127,460 query-page pairs along with Claude 3.5 Sonet’s pseudo questions. Training configuration of the model includes a `paged_adamw_8bit optimizer, learning rate of 5e-5` with linear decay with 2.5% warm up steps and a batch size of `32`.Also the LoRA configs of the language model are `α = 32` and `r = 32`. Additionally, Query Augmentation, a technique is used by appending `5 <unused0>` tokens to query tokens serving as placeholders to be learnt during training. This helps the model to focus on most relevant terms by prioritizing which part of the query is important.

ColPali encodes each page as an image directly consuming a memory of about 256KB per page. The paper also presents other models like ColIdefics2 with composed models of Mistral-7B as decoder and SigLIP as vision encoder.

## **Visual Document Retrieval Benchmark –** [**ViDoRe**](https://github.com/illuin-tech/vidore-benchmark)

A benchmark evaluates document retrieval methods by considering both text and visual document features. It is composed of various page-level retrieving tasks across modalities – text, figures, infographics, tables etc. To create this benchmark, the author collected publicly available pdf documents and generated queries using Claude 3.5 Sonet, producing a set of page-question-answer triplets. An important evaluation metric in this benchmark is `NDCG@5` (Normalized Discounted Cumulative Gain at rank 5) to measure the relevance of top-5 retrieved results, with higher score means more relevant retrieval performance.

[![ColPali on ViDoRe Benchmark results - ColPali vs. Unstructured library for document retrieval](<Base64-Image-Removed>)](https://learnopencv.com/wp-content/uploads/2024/09/ViDoRe-Benchmark-ColPali-Performance-SOTA.png) **FIG 8**: ColPali on ViDoRe Benchmark

Apart from these, ColPali performs well on non-english documents as well; which is attributed to multilingual data present in pre pre-training corpus of Gemma 2B.

### **Limitations** of ColPali

- ColPali is mainly evaluated mainly on English and French language documents.
- The paper discusses throughput and latency aspects mainly for PDF-type documents while non-pdf documents are much underexplored.
- As some part of the training dataset query pairs are being synthetically generated from Claude 3.5 Sonet, there might be some bias in the generated queries.
- Limited Vector DB’s natively support ColBERT embeddings which might be an overhead if you’re planning to upgrade or augment ColPali within your existing RAG application. However, community discussion of Vector DB providers like [Weaviate](https://forum.weaviate.io/t/weaviate-colbertv2/2348) suggest that there are workarounds to retrieve ColBERT style embeddings.
- ColPali saves and retrieves the entire page instead of specific chunks. This demands additional tokens and API costs to process the entire page.
- The most relevant information about the search query might be distributed in small chunks across the pages or any random page within the document. Therefore just retrieving the relevant chunks can sometimes be a more optimal solution.

> _**A vision retriever pipeline shouldn’t be dependent on being able to parse the text from the documents as it defeats the purpose. Another way to reduce search space would be to flatten and pad the query vectors (25×128) and perform vector search on doc vectors (1030×128). This method a hack to capture the pages that might contain patches similar to the query. After all, both use the same projection layers to map inputs to 128-dim. In my experiments, this method worked quite well, reducing the search space without having to capture doc texts.**_
>
> _**– Source:**_ [Ayush Chaurasia, LanceDB](https://linkedin.com/posts/ayushchaurasia_late-interaction-efficient-multi-modal-activity-7242162522974273538-xLGp/?utm_source=share&utm_medium=member_desktop)

### **Trivia**

Visualize Heatmaps of our Query from ColPali

**Search Query**: Alphabet Balance Sheet

[![ColPali engine Heatmap Visualization - Document retrieval using AI - ](<Base64-Image-Removed>)](https://learnopencv.com/wp-content/uploads/2024/09/Heatmap-Vis-Alphabet-Inc-Balance-Sheet-ColPali-Retrieved-Image-Multimodal-RAG-Gemini-1.png) **FIG 9**: Heatmap Visualization

_Search Query_ – Alphabet Balance Sheet

## **Key Takeaways**

- Instead of commercial Gemini or Claude, you can also use a host smaller VLM like Phi-Vision or Qwen-2B in colab, to query over a set of private documents or sensitive data.
- Latest improvements and wrappers around ColPali by [Byaldi](https://github.com/AnswerDotAI/byaldi) [RAGatouille](https://github.com/answerdotai/ragatouille) from AnswerDotAI are promising, which brings it closer to production ready servings with an easy to use retrieval pipeline.

## **Conclusion**

The results from ColPali are pretty impressive in handling multimodal data. We got our hands-on testing of this with a most demanding application in GenAI space to chat with documents like finance, law and regulations etc.

ColPali is a big leap in Multimodal Generative AI space, which improves the document retrieval quality by a huge margin. Kudos to the team at Illuin Technology for making this as an open-source project with **MIT license**.

## **References**

- [ColPali: Efficient Document Retrieval with Vision Language Model](https://arxiv.org/abs/2407.01449)
- [ColPali Github](https://github.com/illuin-tech/colpali)
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/pdf/2004.12832)
- [PyVespa](https://pyvespa.readthedocs.io/en/latest/examples/colpali-document-retrieval-vision-language-models.html)
- [Asif Qamar ColPali Session on Support Vectors](https://www.youtube.com/live/A5J9fAqqR_4?feature=shared)
- [Prompt Engineering – Youtube](https://youtu.be/DI9Q60T_054?feature=shared)
- [GoPenAI](https://blog.gopenai.com/colpali-efficient-document-retrieval-with-vision-language-models-cd47e8d83060)

### Original URL
https://learnopencv.com/multimodal-rag-with-colpali/
</details>

---

## Code Sources

---
<details>
<summary>Gitingest: https://github.com/towardsai/course-ai-agents/blob/main/lessons/11_multimodal/notebook.ipynb</summary>

# Repository Analysis

## Summary
Repository: towardsai/course-ai-agents
File: notebook.ipynb
Lines: 932

Estimated tokens: 47.8k

## File Structure
```
Directory structure:
└── notebook.ipynb

```

## Content
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

#   [93m----------------------------------------------------------------------------------------------------[0m


response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        types.Part.from_bytes(data=image_base64, mime_type="image/webp"),
        "Tell me what is in this image in one paragraph.",
    ],
)
response.text

# Output:
#   "This intriguing image depicts a fascinating juxtaposition between a small, fluffy grey tabby kitten and a large, formidable metallic robot. The kitten, with its tail raised and front paw tentatively reaching towards the robot's shoulder, is perched on the robot's left arm, appearing curious and playful. The robot, a heavily armored and detailed machine, features a complex, circuit-board-like pattern on its head and glowing red eyes that give it a powerful and somewhat imposing presence. The background suggests an industrial or workshop environment, with a soft light source entering from a window on the left, highlighting the unusual yet captivating interaction between the delicate creature and the robust machine."

"""
### 2.3 As URLs

At the time of writing this notebook, Gemini works well primarily with GCP Cloud Storage links, which are excellent for production use cases but complicate our simple demonstration.

The code would not change much and would look like this:
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
### 2.4 Object detection with LLMs

As a more exciting example, let's do object detection with multimodal LLMs.
"""

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    ymin: float
    xmin: float
    ymax: float
    xmax: float
    label: str = Field(default="The object found within the bounding box.")


class Detections(BaseModel):
    bounding_boxes: list[BoundingBox]


client = genai.Client()
prompt = """
Detect all of the prominent items in the image. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.
"""

image_bytes, image_size = load_image_as_bytes(
    image_path=Path("images") / "image_1.jpeg", format="WEBP", return_size=True
)

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

width, height = image_size
print("Image size: ", width, height)
detections = cast(Detections, response.parsed)

for bounding_box in detections.bounding_boxes:
    bounding_box.ymin = int(bounding_box.ymin / 1000 * height)
    bounding_box.xmin = int(bounding_box.xmin / 1000 * width)
    bounding_box.ymax = int(bounding_box.ymax / 1000 * height)
    bounding_box.xmax = int(bounding_box.xmax / 1000 * width)
    print(bounding_box)
# Output:
#   Image size:  600 600

#   ymin=163 xmin=306 ymax=598 xmax=600 label='robot'

#   ymin=163 xmin=20 ymax=475 xmax=321 label='kitten'


"""
### 2.5 Working with PDFs

We can treat PDFs similarly to images. Therefore, we can pass PDFs as bytes:

"""

pdf_bytes = (Path("pdfs") / "decoding_ml_article.pdf").read_bytes()

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

#     This document provides a curated list of five (plus a bonus) recommended books for individuals looking to build and ship AI products, particularly those involving Large Language Models (LLMs) and agentic systems, in 2025.

#   

#   The main topics covered by these recommended books include:

#   *   **Designing Machine Learning Systems:** Fundamentals of building production-grade ML systems, MLOps, and infrastructure.

#   *   **Prompt Engineering:** Techniques for effectively engineering prompts for LLMs to ensure flexibility, scalability, and optimal model performance.

#   *   **AI Engineering:** Broader concepts like RAG (Retrieval Augmented Generation), building agentic systems, and LLMOps (observability, user feedback).

#   *   **Building LLMs for Production:** Hands-on implementation of LLM applications, covering core algorithms, RAG techniques (including GraphRAG), agents, fine-tuning, and deployment using frameworks like LangChain and LlamaIndex.

#   *   **LLMs in Production:** Optimization of LLMs, data engineering for LLM apps, serving LLMs at scale, and infrastructure considerations for deployment.

#   *   **LLM Engineer's Handbook (Bonus):** The author's own book, emphasizing end-to-end LLM RAG application development, including architecting, data collection, fine-tuning, evaluation, deployment, and scaling.

#   

#   In essence, the document serves as a guide to essential reading for anyone involved in the practical development and deployment of AI/ML systems, with a strong focus on LLMs.

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


pdf_base64 = load_pdf_as_base64(pdf_path=Path("pdfs") / "decoding_ml_article.pdf")

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

#     This document is an article titled "5 Books to Ship AI Products in 2025" by Paul Iusztin, published on July 03, 2025. It provides a curated list of five (plus a bonus) book recommendations for individuals interested in AI Engineering, with a specific focus on building and deploying Large Language Model (LLM) and agentic systems.

#   

#   **Main Topics:**

#   

#   *   **AI Engineering Foundations:** General principles for designing and building production-grade machine learning and AI systems.

#   *   **Prompt Engineering for LLMs:** Techniques for effectively creating and optimizing prompts for large language models to enhance their performance and scalability.

#   *   **Building Agentic Systems:** Concepts and methodologies for developing AI agents, including aspects like RAG (Retrieval-Augmented Generation), guardrails, caching, and memory management.

#   *   **LLM Productionization:** Practical guidance on implementing, optimizing, and deploying LLMs in real-world production environments, covering topics like data engineering, fine-tuning, scaling, and infrastructure considerations.

#   *   **LLMOps (MLOps for LLMs):** Operational aspects of managing the lifecycle of LLM applications, including monitoring, data versioning, and CI/CD pipelines.

#   *   **End-to-End LLM Project Development:** Comprehensive approaches to architecting and building complete LLM and RAG (Retrieval-Augmented Generation) applications from concept to deployment.

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
## 3. Implementing multimodal RAG for images and text
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

embed_text_with_gemini("This is a test")
# Output:
#   array([-0.02252334, -0.00076438,  0.00240217, ..., -0.00574729,

#          -0.00052345, -0.00213343], shape=(3072,))

from typing import cast


def create_multimodal_embeddings(image_paths: list[Path]) -> list[dict]:
    """
    Create embeddings for both images and text using proper Gemini approach.

    This function processes a list of image paths, generates descriptions for each image
    using Gemini Vision model, and creates embeddings for those descriptions.

    Args:
        image_paths: List of Path objects pointing to image files to process

    Returns:
        list[dict]: List of dictionaries containing image data, descriptions, and embeddings.
                   Each dict contains keys: 'content', 'type', 'filename', 'description', 'embedding'
    """

    docs = []
    for image_path in image_paths:
        image_bytes = cast(bytes, load_image_as_bytes(image_path, format="WEBP", return_size=False))

        image_description = generate_image_description(image_bytes)
        pretty_print.wrapped(f"`{image_description[:500]}...`", title="Generated image description:")

        image_embedding = embed_text_with_gemini(image_description)

        docs.append(
            {
                "content": image_bytes,
                "type": "image",
                "filename": image_path,
                "description": image_description,
                "embedding": image_embedding,
            }
        )

    return docs


image_paths = list(Path("images").glob("*.jpeg"))
all_docs = create_multimodal_embeddings(image_paths)

if len(all_docs) == 0:
    pretty_print.wrapped("No embeddings were created successfully", title="❌")
else:
    pretty_print.wrapped(f"Successfully created {len(all_docs)} embeddings", title="✅")
# Output:
#   [93m----------------------------------- Generated image description: -----------------------------------[0m

#     `This image features a detailed, high-resolution depiction of a futuristic robot and a small kitten, set in what appears to be an industrial or workshop environment.

#   

#   **Objects:**

#   *   **Robot:** A large, imposing, humanoid robot dominates the right side of the frame. Its metallic body is rendered in shades of dark grey, gunmetal, and silver, appearing heavily armored with visible segmented plates, bolts, and mechanical joints. Its head is helmet-like, featuring prominent, intensely glowing red ey...`

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m----------------------------------- Generated image description: -----------------------------------[0m

#     `This image depicts a dramatic and intense confrontation between a large, fluffy white dog and a sleek, dark humanoid robot in a gritty urban alleyway at dusk or night.

#   

#   **Overall Description:** The central focus is the dynamic interaction between the two main subjects, both in aggressive stances, set against a backdrop of a dirty, littered city street. The scene conveys a sense of tension, conflict, and a bizarre, unlikely encounter.

#   

#   **Objects:**

#   

#   *   **White Dog:** A large, Samoyed-like dog wi...`

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m----------------------------------- Generated image description: -----------------------------------[0m

#     `This detailed description is designed for semantic search, covering various visual elements to maximize discoverability.

#   

#   **Image Description for Semantic Search:**

#   

#   A detailed, close-up shot captures a focused African American man diligently working on the internal components of a desktop computer. The scene is illuminated by the computer's cool blue and cyan internal LED lighting, creating a high-tech and concentrated atmosphere.

#   

#   **Subject (Person):**

#   The primary subject is a middle-aged Blac...`

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m----------------------------------- Generated image description: -----------------------------------[0m

#     `This detailed description is designed for semantic search, covering various visual elements to help users find this image through text queries.

#   

#   **Overall Description:**

#   This image depicts a dynamic, high-impact futuristic battle between two distinct humanoid robots or mechs within a dark, high-tech, enclosed arena. The scene captures the climactic moment of a powerful punch, with sparks and debris erupting, conveying intense action and destruction.

#   

#   **Objects:**

#   

#   *   **Left Robot:**

#       *   **A...`

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m------------------------------------------------ ✅ ------------------------------------------------[0m

#     Successfully created 4 embeddings

#   [93m----------------------------------------------------------------------------------------------------[0m


from sklearn.metrics.pairwise import cosine_similarity


def search_multimodal(query_text: str, docs: list[dict], top_k: int = 3) -> list[Any]:
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
    embeddings = [doc["embedding"] for doc in docs]
    similarities = cosine_similarity([query_embedding], embeddings).flatten()

    # Get top results
    top_indices = np.argsort(similarities)[::-1][:top_k]  # type: ignore

    results = []
    for idx in top_indices.tolist():
        results.append({**docs[idx], "similarity": similarities[idx]})

    return results


query = "two robots fighting"
results = search_multimodal(query, all_docs, top_k=1)

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

#   🔍 Embedding query: 'two robots fighting'

#   ✅ Query embedded successfully

#   [93m----------------------------- Results for query = two robots fighting -----------------------------[0m

#     Similarity 0.757

#   [93m----------------------------------------------------------------------------------------------------[0m

#     Filename images/image_4.jpeg

#   [93m----------------------------------------------------------------------------------------------------[0m

#     Description `This detailed description is designed for semantic search, covering various visual elements to help users find this image through text queries.

#   

#   **Overall Description:**

#   This image depicts a dynamic, high-impact futuristic battle between two distinct humanoid robots or mechs within a dark, high-tech, enclosed arena. The scene captures the climactic moment of a powerful punch, with sparks and debris erupting, conveying intense action and destruction.

#   

#   **Objects:**

#   

#   *   **Left Robot:**

#       *   **Appearance:** Sleek, agile, and humanoid in form, with highly polished, reflective metallic silver or chrome armor. It has a streamlined design.

#       *   **Lighting/Glow:** Features prominent electric blue glowing lines and accents across its body, including a bright blue visor or faceplate, chest, and arms, suggesting energy conduits or internal power.

#       *   **Pose:** Frozen in the act of delivering a powerful right-hand punch directly into the chest or shoulder area of the opposing robot. Its ...`

#   [93m----------------------------------------------------------------------------------------------------[0m

#   <IPython.core.display.Image object>

query = "a kitten with a robot"
results = search_multimodal(query, all_docs, top_k=1)

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

#     Description `This image features a detailed, high-resolution depiction of a futuristic robot and a small kitten, set in what appears to be an industrial or workshop environment.

#   

#   **Objects:**

#   *   **Robot:** A large, imposing, humanoid robot dominates the right side of the frame. Its metallic body is rendered in shades of dark grey, gunmetal, and silver, appearing heavily armored with visible segmented plates, bolts, and mechanical joints. Its head is helmet-like, featuring prominent, intensely glowing red eyes that emit a faint red light. The robot's forehead and top of its head are intricately patterned with what resembles glowing circuit board or microchip designs. Its right arm, extended towards the left, is bulky and robust, with a highly detailed hand composed of multiple metallic segments forming the fingers and palm. The robot appears to be looking down or slightly towards the kitten. There are two small, subtle red indicator lights on its upper chest armor.

#   *   **Kitten:** A small, fluffy, ...`

#   [93m----------------------------------------------------------------------------------------------------[0m

#   <IPython.core.display.Image object>

"""
## 4. Building multimodal AI agents
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

    results = search_multimodal(query, all_docs, top_k=1)

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
    5. If you can't find the specific information requested, be honest about limitations
    
    Pay special attention to:
    - Colors and visual characteristics
    - Animal features and breeds
    - Objects and their properties
    - Scene descriptions and context
    
    Always search first using your tools before attempting to answer questions about specific images or visual content.
    """

    agent = create_react_agent(
        model=ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1),
        tools=tools,
        prompt=system_prompt,
    )

    return agent


try:
    react_agent = build_react_agent()

    test_question = "what color is my kitten?"
    pretty_print.wrapped(test_question, title="🧪 Asking question:")

    response = react_agent.invoke(input={"messages": test_question})
    messages = response.get("messages", [])
    if messages:
        final_message = messages[-1].content
    else:
        final_message = "No response from the agent"
    pretty_print.wrapped(final_message, title="🤖 Agent response")
except Exception as e:
    print(f"❌ Error in ReAct agent: {e}")
# Output:
#   [93m---------------------------------------- 🧪 Asking question: ----------------------------------------[0m

#     what color is my kitten?

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m----------------------------------- 🔍 Tool executing search for: -----------------------------------[0m

#     kitten

#   [93m----------------------------------------------------------------------------------------------------[0m

#   

#   🔍 Embedding query: 'kitten'

#   ✅ Query embedded successfully

#   [93m----------------------------------------- 🔍 Found results: -----------------------------------------[0m

#     images/image_1.jpeg

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [93m----------------------------------------- 🤖 Agent response -----------------------------------------[0m

#     Based on the image, your kitten is a light grey tabby with darker grey stripes.

#   [93m----------------------------------------------------------------------------------------------------[0m

### Original URL
https://github.com/towardsai/course-ai-agents/blob/main/lessons/11_multimodal/notebook.ipynb
</details>

---

## Additional Sources Scraped

---
<details>
<summary>Ahead of AI | Sebastian Raschka, PhD | Substack</summary>

![](https://substackcdn.com/image/fetch/$s_!yI7e!,w_600,h_400,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_center/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc534f387-f776-41eb-9c65-f0032b91daee_1988x1430.png)

[Understanding Multimodal LLMs](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms)

[An introduction to the main techniques and latest models](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms)

### Original URL
https://magazine.sebastianraschka.com/
</details>

---
<details>
<summary>What are Vision-Language Models? | NVIDIA Glossary</summary>

# Vision Language Models

Vision language models (VLMs) are multimodal, generative AI models capable of understanding and processing video, image, and text.

## What Are Vision Language Models?

Vision language models are multimodal AI systems built by combining a large language model (LLM) with a vision encoder, giving the LLM the ability to “see.”

With this ability, VLMs can process and provide advanced understanding of video, image, and text inputs supplied in the prompt to generate text responses.

![A diagram showing a vision language model is capable of a variety of use cases: Video Summarization, Image Analysis, Multimodal Chat, Document Parsing (top right to bottom right).](https://www.nvidia.com/content/nvidiaGDC/us/en_US/glossary/vision-language-models/_jcr_content/root/responsivegrid/nv_container_copy/nv_image.coreimg.100.1290.png/1736201901571/metropolis-iva-diagram-vlm-glossary-ces25-3576177-r1--1-.png)

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

![A diagram showing the architecture of a vision language model](https://www.nvidia.com/content/nvidiaGDC/us/en_US/glossary/vision-language-models/_jcr_content/root/responsivegrid/nv_container_copy_co_300503066/nv_image.coreimg.svg/1736168815674/vlm-architecture-diagram.svg)

Figure 2: A common three-part architecture for vision language models

## How Are Vision Language Models Trained?

VLMs are trained in several stages that include pretraining, followed by supervised fine-tuning. Optionally, parameter efficient fine-tuning (PEFT) can be applied as a final stage to create a domain-specific VLM on custom data.

The pretraining stage aligns the vision encoder, projector, and LLM to essentially speak the same language when interpreting the text and image input. This is done using large corpora of text and images with image-caption pairs and interleaved image-text data. Once the three components have been aligned through pretraining, the VLM goes through a supervised fine-tuning stage to help it understand how to respond to user prompts.

The data used in this stage are a blend of example prompts with text and/or image input and the expected response of the model. For example, this data could be prompts telling the model to describe the image or to count all the objects in the frame with the expected correct response. After this round of training, the VLM will understand how to best interpret images and respond to user prompts.

![A diagram showing a three stage training process for VLMs](https://www.nvidia.com/content/nvidiaGDC/us/en_US/glossary/vision-language-models/_jcr_content/root/responsivegrid/nv_container_copy_co_1755415045/nv_image.coreimg.svg/1736168816034/vlm-training-process-diagram.svg)

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

![A graphic showing example multiple choice questions for VLMs across several subjects from the MMMU benchmark.](https://www.nvidia.com/content/nvidiaGDC/us/en_US/glossary/vision-language-models/_jcr_content/root/responsivegrid/nv_container_copy_co_42410027/nv_image.coreimg.100.1290.jpeg/1736168816436/vlm-mmmu-ari.jpeg)

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

![A diagram showing how visual AI Agents can be applied in the real world for warehouse, traffic and sports use cases.](https://www.nvidia.com/content/nvidiaGDC/us/en_US/glossary/vision-language-models/_jcr_content/root/responsivegrid/nv_container_copy_co_531349501/nv_image.coreimg.svg/1736168816834/vlm-real-world-diagram.svg)

Figure 5: video analytics AI agents transform video and image data into real-world insights

With vast amounts of video being produced every day, it's infeasible to review and extract insights from this volume of video that is produced by all industries. VLMs can be integrated into a larger system to build video analytics AI agents capable of detecting specific events when prompted. These systems could be used to detect malfunctioning robots in a warehouse or generate out-of-stock alerts when shelves are empty. Their general understanding goes beyond simple detection and could be used to generate automated reports. For example, an intelligent traffic system could detect, analyze, and produce reports of traffic hazards, such as fallen trees, stalled vehicles, or collisions.

VLMs can be used with technologies like graph databases to understand long videos. This helps them capture the complexity of objects and events in a video. Such systems could be used to summarize operations in a warehouse to find bottlenecks and inefficiencies or produce sports commentary for football, basketball, or soccer games.

## What Are the Challenges of Vision Language Models?

Vision language models are maturing quickly, but they still have some limitations, particularly around spatial understanding and long-context video understanding.

Most VLMs use CLIP-based models as the vision encoder, which are limited to 224x224 or 336x336 image input size. This relatively small input image makes it difficult for small objects and details to be detected. For example, an HD 1080x1920 frame from a video must be downsized or cropped to a much smaller input resolution, making it difficult to retain details for small objects or fine details. To fix this, VLMs are starting to use tiling methods that allow a big image to be broken into smaller pieces and then fed into the model. There's also ongoing research to explore the use of higher-resolution image encoders.

VLMs also have difficulty providing precise locations for objects. The training data for CLIP-based vision encoders consists mostly of short text descriptions of images, like captions. These descriptions don't include detailed, fine-grained object locations, and this limitation impacts CLIP’s spatial understanding. This is inherited by VLMs that use it as a vision encoder. New approaches are exploring the use of ensembling several vision encoders to address these limitations [2408.15998 (arxiv.org)](https://arxiv.org/pdf/2408.15998).

Long video understanding is a challenge due to the need to take into account visual information across potential hours of video to properly analyze or answer questions. Like LLMs, VLMs have limited context length meaning—only a certain number of frames from a video can be included to answer questions. Approaches to increase context length and train VLMs on more video-based data are being researched, such as LongVILA [2408.10188 (arxiv.org)](https://www.arxiv.org/pdf/2408.10188).

VLMs may not have seen enough data for very specific use cases, such as finding manufacturing defects in a specific product line. This limitation can be overcome by fine-tuning the VLM on domain-specific data or using multi-image VLMs with in-context learning to provide examples that can teach the model new information without explicitly training the model. Training the model on domain-specific data with PEFT is another technique that can be used to improve a VLM’s accuracy on custom data.
</assistant

### Original URL
https://www.nvidia.com/en-us/glossary/vision-language-models/
</details>

---
<details>
<summary>Multimodal Embeddings: An Introduction | Towards Data Science</summary>

# Multimodal Embeddings: An Introduction

Mapping text and images into a common space

This is the 2nd article in a [larger series](https://shawhin.medium.com/list/multimodal-ai-fe9521d0e77a) on multimodal AI. In the [previous post](https://towardsdatascience.com/multimodal-models-llms-that-can-see-and-hear-5c6737c981d3), we saw how to augment [large language models (LLMs)](https://shawhin.medium.com/list/large-language-models-llms-8e009ae3054c) to understand new data modalities (e.g., images, audio, video). One such approach relied on encoders that generate vector representations (i.e. embeddings) of non-text data. In this article, I will discuss _multimodal_ embeddings and share what they can do via two practical use cases.

![Image from Canva.](https://towardsdatascience.com/wp-content/uploads/2024/11/1a6BF-kEeo8rd7OW2a3JYGA.png)Image from Canva.

AI research is traditionally split into distinct fields: NLP, computer vision (CV), robotics, human-computer interface (HCI), etc. However, countless practical tasks require the **integration of these different research areas** e.g. autonomous vehicles (CV + robotics), AI agents (NLP + CV + HCI), personalized learning (NLP + HCI), etc.

Although these fields aim to solve different problems and work with different data types, they all share a fundamental process. Namely, **generating useful numerical representations of real-world phenomena**.

Historically, this was done by hand. This means that researchers and practitioners would use their (or other people’s) expertise to explicitly transform data into a more helpful form. Today, however, _these can be derived another way_.

## **Embeddings**

**Embeddings** are **(useful) numerical representations of data learned implicitly through model training**. For example, through learning how to predict text, BERT learned representations of text, which are helpful for many NLP tasks \[1\]. Another example is the Vision Transformer (ViT), trained for image classification on Image Net, which can be repurposed for other applications \[2\].

A key point here is that these learned embedding spaces will have some underlying structure so that **similar concepts are located close together**. As shown in the toy examples below.

![Toy represetation of text and image embeddings, respectively. Image by author.](https://towardsdatascience.com/wp-content/uploads/2024/11/1jpmC6Kx7DxVeikEr15vooA.png)Toy represetation of text and image embeddings, respectively. Image by author.

One **key limitation** of the previously mentioned models is they are restricted to a single data modality, e.g., text or images. Preventing cross-modal applications like image captioning, content moderation, image search, and more. _But what if we could merge these two representations?_

## **Multimodal Embeddings**

Although text and images may look very different to us, in a neural network, these are **represented via the same mathematical object**, i.e., a vector. Therefore, in principle, text, images, or any other data modality can processed by a single model.

This fact underlies **multimodal embeddings**, which **represent multiple data modalities in the same vector space** such that similar concepts are co-located (independent of their original representations).

![Toy representation of multimodal embedding space. Image by author.](https://towardsdatascience.com/wp-content/uploads/2024/11/15d3HBNjNIXLy0oMIvJjxWw.png)Toy representation of multimodal embedding space. Image by author.

For example, CLIP encodes text and images into a shared embedding space \[3\]. A key insight from CLIP is that by aligning text and image representations, the **model is capable of 0-shot image classification on an arbitrary set of target classes** since any input text can be treated as a class label (we will see a concrete example of this later).

However, this idea is not limited to text and images. Virtually any data modalities can be aligned in this way e.g., text-audio, audio-image, text-EEG, image-tabular, and text-video. Unlocking use cases such as video captioning, advanced OCR, audio transcription, video search, and EEG-to-text \[4\].

## **Contrastive Learning**

The standard approach to aligning disparate embedding spaces is **contrastive learning (CL)**. A key intuition of CL is to **represent different views of the same _information_ similarly** \[5\].

This consists of learning representations that **maximize the similarity between positive pairs** and **minimize the similarity of negative pairs**. In the case of an image-text model, a positive pair might be an image with an appropriate caption, while a negative pair would be an image with an irrelevant caption (as shown below).

![Example positive and negative pairs used in contrastive training. Image by author.](https://towardsdatascience.com/wp-content/uploads/2024/11/1AGHBVjzwjXapJSe4aUPrjg.png)Example positive and negative pairs used in contrastive training. Image by author.

**Two key aspects** **of CL** contribute to its effectiveness

1. Since positive and negative pairs can be curated from the data’s inherent structure (e.g., metadata from web images), CL training data **do not require manual labeling**, which unlocks larger-scale training and more powerful representations \[3\].
2. It simultaneously maximizes positive and minimizes negative pair similarity via a special loss function, as demonstrated by CLIP \[3\].

![CLIP's contrastive loss for text-image representation alignment [3]. Image by author.](https://towardsdatascience.com/wp-content/uploads/2024/11/12X1aT8fzFsgbqn23zXmmAA.png)CLIP’s contrastive loss for text-image representation alignment \[3\]. Image by author.

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
```

![Input cat photo. Image from Canva.](https://towardsdatascience.com/wp-content/uploads/2024/11/1Nzo536sqahqm1Q24Ms2vmA.png)Input cat photo. Image from Canva.

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
```

![Best match for query "a cute dog". Image from Canva.](https://towardsdatascience.com/wp-content/uploads/2024/11/14wnqr5p_7N3QD5EkXIQeew.png)Best match for query "a cute dog". Image from Canva.

We see that (again) the model nailed this simple example. But let’s try some trickier examples.

```python
query = "something cute but metal 🤘"
```

``` language-none
>> Match probability:  0.7715
```

![Best match for query "something cute but metal 🤘". Image from Canva.](https://towardsdatascience.com/wp-content/uploads/2024/11/1tIY3_ONQQT_cracAPWm8NQ.png)Best match for query "something cute but metal 🤘". Image from Canva.

```python
query = "a good boy"
```

``` language-none
>> Match probability:  0.8248
```

![Best match for query "a good boy". Image from Canva.](https://towardsdatascience.com/wp-content/uploads/2024/11/14wnqr5p_7N3QD5EkXIQeew.png)Best match for query "a good boy". Image from Canva.

```python
query = "the best pet in the world"
```

``` language-none
>> Match probability:  0.5664
```

![Best match for query "the best pet in the world". Image from Canva.](https://towardsdatascience.com/wp-content/uploads/2024/11/1Nzo536sqahqm1Q24Ms2vmA.png)Best match for query "the best pet in the world". Image from Canva.

Although this last prediction is quite controversial, all the other matches were spot on! This is likely since images like these are ubiquitous on the internet and thus were seen many times in CLIP’s pre-training.

> [**YouTube-Blog/multimodal-ai/2-mm-embeddings at main · ShawhinT/YouTube-Blog**](https://github.com/ShawhinT/YouTube-Blog/tree/main/multimodal-ai/2-mm-embeddings)

## What’s Next?

Multimodal embeddings unlock countless AI use cases that involve multiple data modalities. Here, we saw two such use cases, i.e., 0-shot image classification and image search using CLIP.

Another practical application of models like CLIP is multimodal RAG, which consists of the automated retrieval of multimodal context to an LLM. In the [next article](https://medium.com/towards-data-science/multimodal-rag-process-any-file-type-with-ai-e6921342c903) of this [series](https://shawhin.medium.com/list/multimodal-ai-fe9521d0e77a), we will see how this works under the hood and review a concrete example.

**More on Multimodal models 👇**

> [**Multimodal AI**](https://shawhin.medium.com/list/fe9521d0e77a)

- \[1\] [BERT](https://arxiv.org/abs/1810.04805)
- \[2\] [ViT](https://arxiv.org/abs/2010.11929)
- \[3\] [CLIP](https://arxiv.org/abs/2103.00020)
- \[4\] [Thought2Text: Text Generation from EEG Signal using Large Language Models (LLMs)](https://arxiv.org/abs/2410.07507)
- \[5\] [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)

### Original URL
https://towardsdatascience.com/multimodal-embeddings-an-introduction-5dc36975966f/
</details>

---
<details>
<summary>Error during scraping</summary>

Failed to scrape: None

### Original URL
https://www.youtube.com/watch?v=YOvxh_ma5qE
</details>

---
<details>
<summary>Multi-modal ML with OpenAI's CLIP | Pinecone</summary>

Language models (LMs) can not rely on language alone. That is the idea behind the “Experience Grounds Language” paper, that proposes a framework to measure LMs' current and future progress. A key idea is that, beyond a certain threshold LMs need other forms of data, such as visual input \[1\] \[2\].

![World Scopes (WS), as datasets become larger in scope and span multiple modalities, the capabilities of models trained with them increase.](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F25e7f2f54b543af8c34c143448a4b0c55f77c6b5-2360x854.png&w=3840&q=75)

World Scopes (WS), as datasets become larger in scope and span multiple modalities, the capabilities of models trained with them increase.

The next step beyond well-known language models; BERT, GPT-3, and T5 is _”World Scope 3”_. In World Scope 3, we move from large text-only datasets to large multi-modal datasets. That is, datasets containing information from multiple forms of media, like _both_ images and text.

The world, both digital and real, is multi-modal. We perceive the world as an orchestra of language, imagery, video, smell, touch, and more. This chaotic ensemble produces an inner state, our “model” of the outside world.

AI must move in the same direction. Even specialist models that focus on language or vision must, at some point, have input from the other modalities. How can a model fully understand the concept of the word “person” without _seeing_ a person?

OpenAI **C** ontrastive **L** earning **I** n **P** retraining (CLIP) is a world scope three model. It can comprehend concepts in both text and image and even connect concepts between the two modalities. In this chapter we will learn about multi-modality, how CLIP works, and how to use CLIP for different use cases like encoding, classification, and object detection.

* * *

## Multi-modality

The multi-modal nature of CLIP is powered by two encoder models trained to “speak the same language”. Text inputs are passed to a text encoder, and image inputs to an image encoder \[3\]. These models then create a _vector representation_ of the respective input.

Both models “speak the same language” by encoding similar concepts in text and images into similar vectors. That means that the text “two dogs running across a frosty field” would output a vector similar to an _image_ of two dogs running across a frosty field.

![Similar text and images will be encoded into a similar vector space. Dissimilar text and images do not share a similar vector space.](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fa54a2f1fa0aeac03748c09df0fdfbb42aadc96b7-2430x1278.png&w=3840&q=75)

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

![Architecture diagram of CLIP with the text encoder and ViT or ResNet as the image encoder.](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F539716ea1571e459908c1fdc5a898fea239d8243-2803x1672.png&w=3840&q=75)

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

![Contrastive pretraining with CLIP.](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fd6868e6dae721512fed8f1287fc9ffe6b6a2cddd-2332x1342.png&w=3840&q=75)

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

![Example of text-image pair found in the dataset. Text is stored in the "text" feature and images in the "image" feature.](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fa40f673ed52e07f497c7a39b032c27b33ce9f565-1128x761.png&w=3840&q=75)

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

![We take the original imagenette labels and preappend "a photo of a ..." to each to create a set of CLIP-friendly sentence representations.](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Ff841984e7617686f5041ca95797498e2b0b085b5-1348x542.png&w=3840&q=75)

We take the original imagenette labels and preappend "a photo of a ..." to each to create a set of CLIP-friendly sentence representations.

From this, we can calculate the cosine similarity between the text embeddings of these ten labels against an image we’d like to classify. The text that returns the highest similarity is our predicted class.

### Object Detection

Another compelling use case of zero-shot CLIP is object detection. We can do this by splitting our images into smaller patches and running each patch through the image encoder of CLIP. We then compare these patch embeddings to a text encoding describing what we are looking for. After calculating the similarity scores for all patches, we can collate them into a map of relevance.

For example, given an image of a butterfly and a cat, we could break it into many small patches. Given the prompt `"a fluffy cat"`, we will return an outline of the cat, whereas the prompt `"a butterfly"` will produce an outline of the butterfly.

![Zero-shot object detection with CLIP allows us to find specific objects with natural language prompts.](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fbe4800918976efd9d974d9e5453985a5106f2558-2389x1455.png&w=3840&q=75)

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

### Original URL
https://www.pinecone.io/learn/series/image-search/clip/
</details>

---
<details>
<summary>ColPali: Efficient Document Retrieval with Vision Language Models</summary>

# ColPali: Efficient Document Retrieval with    Vision Language Models

###### Abstract

Documents are visually rich structures that convey information through text, but also figures, page layouts, tables, or even fonts. Since modern retrieval systems mainly rely on the textual information they extract from document pages to index documents -often through lengthy and brittle processes-, they struggle to exploit key visual cues efficiently. This limits their capabilities in many practical document retrieval applications such as Retrieval Augmented Generation (RAG).
To benchmark current systems on visually rich document retrieval, we introduce the Visual Document Retrieval Benchmark ViDoRe, composed of various page-level retrieval tasks spanning multiple domains, languages, and practical settings.
The inherent complexity and performance shortcomings of modern systems motivate a new concept; doing document retrieval by directly embedding the images of the document pages. We release ColPali, a Vision Language Model trained to produce high-quality multi-vector embeddings from images of document pages. Combined with a late interaction matching mechanism, ColPali largely outperforms modern document retrieval pipelines while being drastically simpler, faster and end-to-end trainable.
We release models, data, code and benchmarks under open licenses at [https://hf.co/vidore](https://hf.co/vidore "").

## 1 Introduction

Document Retrieval consists of matching a user query to relevant documents in a given corpus. It is central to many widespread industrial applications, either as a standalone ranking system (search engines) or as part of more complex information extraction or Retrieval Augmented Generation (RAG) pipelines.

Over recent years, pretrained language models have enabled large improvements in text embedding models. In practical industrial settings, however, the primary performance bottleneck for efficient document retrieval stems not from embedding model performance but from the prior data ingestion pipeline. Indexing a standard PDF document involves several steps. First, PDF parsers or Optical Character Recognition (OCR) systems are used to extract words from the pages. Document layout detection models can then be run to segment paragraphs, titles, and other page objects such as tables, figures, and headers. A chunking strategy is then defined to group text passages with some semantical coherence, and modern retrieval setups may even integrate a captioning step to describe visually rich elements in a natural language form, more suitable for embedding models.
In our experiments ( [Table 2](https://arxiv.org/html/2407.01449v6#S5.T2 "Table 2 ‣ 5 Results ‣ ColPali: Efficient Document Retrieval with Vision Language Models")), we typically find that optimizing the ingestion pipeline yields much better performance on visually rich document retrieval than optimizing the text embedding model.

Contribution 1: ViDoRe.
In this work, we argue that document retrieval systems should not be evaluated solely on the capabilities of text embedding models (Bajaj et al., [2016](https://arxiv.org/html/2407.01449v6#bib.bib6 ""); Thakur et al., [2021](https://arxiv.org/html/2407.01449v6#bib.bib50 ""); Muennighoff et al., [2022](https://arxiv.org/html/2407.01449v6#bib.bib40 "")), but should also consider the context and visual elements of the documents to be retrieved. To this end, we create and openly release ViDoRe, a comprehensive benchmark to evaluate systems on page-level document retrieval with a wide coverage of domains, visual elements, and languages. ViDoRe addresses practical document retrieval scenarios, where queries often necessitate both textual and visual understanding for accurate document matching. We highlight the shortcomings of current text-centric systems in these settings.111The ViDoRe benchmark leaderboard is hosted publicly at [https://huggingface.co/spaces/vidore/vidore-leaderboard](https://huggingface.co/spaces/vidore/vidore-leaderboard "") to encourage further developments.

Contribution 2: ColPali.
We propose a novel concept and model architecture based on Vision Language Models (VLMs) to efficiently index documents purely from their visual features, allowing for subsequent fast query matching with late interaction mechanisms (Khattab & Zaharia, [2020](https://arxiv.org/html/2407.01449v6#bib.bib26 "")). Our method, ColPali, significantly outperforms all other retrieval systems on ViDoRe while being fast and end-to-end trainable.
These results demonstrate the potential and the many benefits of this novel Retrieval in Vision Space concept, which could significantly alter the way document retrieval is approached in the industry moving forward.
We release all resources at [https://hf.co/vidore](https://hf.co/vidore "").

![Refer to caption](https://arxiv.org/html/2407.01449v6/extracted/6240861/images/final_architecture.png)

Figure 1: ColPali simplifies document retrieval w.r.t. standard retrieval methods while achieving stronger performances with better latencies. Latencies and results are detailed in [section 5](https://arxiv.org/html/2407.01449v6#S5 "5 Results ‣ ColPali: Efficient Document Retrieval with Vision Language Models") and [subsection B.4](https://arxiv.org/html/2407.01449v6#A2.SS4 "B.4 Latency computations ‣ Appendix B Implementation details ‣ ColPali: Efficient Document Retrieval with Vision Language Models").

## 2 Problem Formulation & Related Work

Problem Setting.
In our setting, a retrieval system scores how relevant a document d from corpus D is with respect to a query q. Computing the similarity score s(q,d)∈ℝ for each of the |D| documents in the corpus creates a ranking we can use to extract the most relevant documents. In this work, we focus on page-level retrieval: given a query, is the correct document page retrieved by the system? For coherence with existing literature, we further use the term document to refer to individual pages, i.e. the atomic retrieved elements in our setting. As we focus on practical industrial retrieval applications (RAG, search engines) with potentially large corpora sizes, latency constraints are imposed on scoring systems. Most current retrieval systems can be decomposed into (1) an offline indexation phase in which a document index is built and (2) an online querying phase in which a query is matched to documents from the index and where low latency is vital to the user experience.

Under these industrial constraints, we identify three main properties an efficient document retrieval systems should exhibit: (R1) strong retrieval performance, as measured by standard retrieval metrics; (R2) fast online querying, measured through average latencies; (R3) high throughput corpus indexation, ie. the number of pages that can be embedded in a given timeframe.

### 2.1 Textual Retrieval Methods

Document Retrieval in Text Space.

Statistical methods based on word frequency like TF-IDF (Sparck Jones, [1972](https://arxiv.org/html/2407.01449v6#bib.bib48 "")) and BM25 (Robertson et al., [1994](https://arxiv.org/html/2407.01449v6#bib.bib45 "")) are still widely used due to their simplicity and efficiency. More recently, neural embedding models based on fine-tuned large language models display state-of-the-art performance on a variety of text embedding tasks and top the retrieval leaderboards (Muennighoff et al., [2022](https://arxiv.org/html/2407.01449v6#bib.bib40 "")).

Neural Retrievers.
In bi-encoder models (Reimers & Gurevych, [2019](https://arxiv.org/html/2407.01449v6#bib.bib44 ""); Karpukhin et al., [2020](https://arxiv.org/html/2407.01449v6#bib.bib25 ""); Wang et al., [2022](https://arxiv.org/html/2407.01449v6#bib.bib52 "")), documents are independently mapped offline to a dense vector space. Queries are embedded online and matched to documents through a fast cosine distance computation.
A slower, but slightly more performant alternative, cross-encoder systems (Wang et al., [2020](https://arxiv.org/html/2407.01449v6#bib.bib55 ""); Cohere, [2024](https://arxiv.org/html/2407.01449v6#bib.bib13 "")) concatenate query and document as a single input sequence and iteratively attribute matching scores to each possible combination. This enables full attention computation between query and document terms but comes at the cost of computational efficiency, as |D| encoding passes must be done online.

Multi-Vector retrieval via late interaction.
In the late interaction paradigm introduced by ColBERT (Khattab & Zaharia, [2020](https://arxiv.org/html/2407.01449v6#bib.bib26 "")), an embedding is pre-computed and indexed per document token. At runtime, similarity can be computed with individual query token embeddings. The idea is to benefit from the rich interaction between individual query and document terms while taking advantage of the offline computation and fast query matching enabled by bi-encoders. See [Appendix E](https://arxiv.org/html/2407.01449v6#A5.SSx3 "ColBERT ‣ Appendix E Model glossary ‣ ColPali: Efficient Document Retrieval with Vision Language Models") for more details.

Retrieval Evaluation.
Although benchmarks and leaderboards have been developed to evaluate text embedding models (Thakur et al., [2021](https://arxiv.org/html/2407.01449v6#bib.bib50 ""); Muennighoff et al., [2022](https://arxiv.org/html/2407.01449v6#bib.bib40 "")), much of the performance improvements in industrial use cases of embedding models stem from the prior data ingestion pipeline. While documents often rely on visual elements to more efficiently convey information to human readers, text-only systems barely tap into these visual cues. Other work has also independently studied table or chart retrieval systems through repurposed Question Answering datasets (Zhang et al., [2019](https://arxiv.org/html/2407.01449v6#bib.bib59 ""); Nowak et al., [2024](https://arxiv.org/html/2407.01449v6#bib.bib42 "")) but only assessing specialized methods for each task.

To our knowledge, no benchmark evaluates document retrieval systems in practical settings; in an end-to-end manner, across several document types and topics, and by evaluating the use of both textual and visual document features.

### 2.2 Integrating Visual features

Contrastive Vision Language Models.
Mapping latent representations of textual content to corresponding representations of visual content has been done by aligning disjoint visual and text encoders through contrastive losses (Radford et al., [2021](https://arxiv.org/html/2407.01449v6#bib.bib43 ""); Zhai et al., [2023](https://arxiv.org/html/2407.01449v6#bib.bib58 "")). While some OCR capabilities exist in these models, the visual component is often not optimized for text understanding.

The Fine-grained Interactive Language-Image Pre-training (Yao et al., [2021](https://arxiv.org/html/2407.01449v6#bib.bib56 "")) framework extends the late interaction mechanism to cross-modal Vision Language Models, relying on max similarity operations between text tokens and image patches.

Visually Rich Document Understanding.
To go beyond text, some document-focused models jointly encode text tokens alongside visual or document layout features (Appalaraju et al., [2021](https://arxiv.org/html/2407.01449v6#bib.bib4 ""); Kim et al., [2021](https://arxiv.org/html/2407.01449v6#bib.bib27 ""); Huang et al., [2022](https://arxiv.org/html/2407.01449v6#bib.bib22 ""); Tang et al., [2022](https://arxiv.org/html/2407.01449v6#bib.bib49 "")).
Large Language transformer Models (LLMs) with strong reasoning capabilities have recently been combined with Vision Transformers (ViTs) (Dosovitskiy et al., [2020](https://arxiv.org/html/2407.01449v6#bib.bib17 "")) to create VLMs (Alayrac et al., [2022](https://arxiv.org/html/2407.01449v6#bib.bib2 ""); Liu et al., [2023](https://arxiv.org/html/2407.01449v6#bib.bib35 ""); Bai et al., [2023](https://arxiv.org/html/2407.01449v6#bib.bib5 ""); Laurençon et al., [2024b](https://arxiv.org/html/2407.01449v6#bib.bib30 "")) where image patch vectors from contrastively trained ViT models (Zhai et al., [2023](https://arxiv.org/html/2407.01449v6#bib.bib58 "")) are fed as input embeddings to the LLM and concatenated with the text-token embeddings.

PaliGemma.
The PaliGemma-3B model (Beyer et al., [2024](https://arxiv.org/html/2407.01449v6#bib.bib7 "")) extends concepts from Pali3 (Chen et al., [2023](https://arxiv.org/html/2407.01449v6#bib.bib11 "")), and projects SigLIP-So400m/14(Alabdulmohsin et al., [2023](https://arxiv.org/html/2407.01449v6#bib.bib1 "")) patch embeddings into Gemma-2B’s text vector space (Gemma Team et al., [2024](https://arxiv.org/html/2407.01449v6#bib.bib19 "")). Along with its reasonable size w.r.t. other performant VLMs, an interesting property of PaliGemma’s text model is that it is fine-tuned with full-block attention on the prefix (instruction text and image tokens). See [Appendix E](https://arxiv.org/html/2407.01449v6#A5 "Appendix E Model glossary ‣ ColPali: Efficient Document Retrieval with Vision Language Models") for more details.

VLMs display enhanced capabilities in Visual Question Answering, captioning, and document understanding (Yue et al., [2023](https://arxiv.org/html/2407.01449v6#bib.bib57 "")), but are not optimized for retrieval tasks.

## 3 The ViDoRe Benchmark

Existing benchmarks for contrastive vision-language models primarily evaluate retrieval for natural images (Lin et al., [2014](https://arxiv.org/html/2407.01449v6#bib.bib34 ""); Borchmann et al., [2021](https://arxiv.org/html/2407.01449v6#bib.bib9 ""); Thapliyal et al., [2022](https://arxiv.org/html/2407.01449v6#bib.bib51 "")). On the other hand, textual retrieval benchmarks (Muennighoff et al., [2022](https://arxiv.org/html/2407.01449v6#bib.bib40 "")) are evaluated at at textual passage level and are not tailored for document retrieval tasks. We fill the gap with ViDoRe, a comprehensive benchmark for document retrieval using visual features.

### 3.1 Benchmark Design

ViDoRe is designed to comprehensively evaluate retrieval systems on their capacity to match queries to relevant documents at the page level. This benchmark encompasses multiple orthogonal subtasks, with focuses on various modalities - text, figures, infographics, tables; thematic domains - medical, business, scientific, administrative; or languages - English, French. Tasks also span varying levels of complexity, in order to capture signals from both weaker and stronger systems.
As many systems require large amounts of time to index pages (captioning-based approaches can take dozens of seconds per page for instance), we limit the number of candidate documents for each retrieval task in order to evaluate even complex systems in a reasonable timeframe without sacrificing quality. For trainable retrieval systems, we provide a reference training set that can be used to facilitate comparisons.

| Dataset | Language | \# Queries | \# Documents | Description |
| --- | --- | --- | --- | --- |
| Academic Tasks |  |  |  |  |
| DocVQA | English | 500 | 500 | Scanned documents from UCSF Industry |
| InfoVQA | English | 500 | 500 | Infographics scrapped from the web |
| TAT-DQA | English | 1600 | 1600 | High-quality financial reports |
| arXiVQA | English | 500 | 500 | Scientific Figures from arXiv |
| TabFQuAD | French | 210 | 210 | Tables scrapped from the web |
| Practical Tasks |  |  |  |  |
| Energy | English | 100 | 1000 | Documents about energy |
| Government | English | 100 | 1000 | Administrative documents |
| Healthcare | English | 100 | 1000 | Medical documents |
| AI | English | 100 | 1000 | Scientific documents related to AI |
| Shift Project | French | 100 | 1000 | Environmental reports |

Table 1: ViDoRe comprehensively evaluates multimodal retrieval methods.

Academic Tasks.
We repurpose widely used visual question-answering benchmarks for retrieval tasks: for each page-question-answer triplet, we use the question as the query, and the associated page as the gold document ( [Table 1](https://arxiv.org/html/2407.01449v6#S3.T1 "Table 1 ‣ 3.1 Benchmark Design ‣ 3 The ViDoRe Benchmark ‣ ColPali: Efficient Document Retrieval with Vision Language Models")). These academic datasets either focus on single specific modalities (Mathew et al., [2020](https://arxiv.org/html/2407.01449v6#bib.bib38 ""); [2021](https://arxiv.org/html/2407.01449v6#bib.bib39 ""); Li et al., [2024](https://arxiv.org/html/2407.01449v6#bib.bib33 "")) or target more varied visually rich documents (Zhu et al., [2022](https://arxiv.org/html/2407.01449v6#bib.bib61 "")). Moreover, we consider TabFQuAD, a human-labeled dataset on tables extracted from French industrial PDF documents released with this work. Details can be found in [subsection A.1](https://arxiv.org/html/2407.01449v6#A1.SS1 "A.1 Academic Datasets ‣ Appendix A Benchmark Datasets ‣ ColPali: Efficient Document Retrieval with Vision Language Models").

Practical tasks.
We construct topic-specific retrieval benchmarks spanning multiple domains to go beyond repurposed QA datasets and evaluate retrieval in more realistic industrial situations (e.g. RAG). To achieve this, we collect publicly accessible PDF documents and generate queries pertaining to document pages using Claude-3 Sonnet, a high-quality proprietary vision-language model (Anthropic, [2024](https://arxiv.org/html/2407.01449v6#bib.bib3 "")). In total, we collect 1,000 document pages per topic, which we associate with 100 queries extensively filtered for quality and relevance by human annotators. The corpus topics are intentionally specific to maximize syntactic proximity between documents, creating more challenging retrieval tasks and covering an array of orthogonal domains ( [Table 1](https://arxiv.org/html/2407.01449v6#S3.T1 "Table 1 ‣ 3.1 Benchmark Design ‣ 3 The ViDoRe Benchmark ‣ ColPali: Efficient Document Retrieval with Vision Language Models")).

Evaluation Metrics.
We evaluate performance on our benchmark (Requirement R1) using standard metrics from the retrieval literature (nDCG, Recall@K, MRR). We report nDCG@5 values as the main performance metric in this work and release the complete sets of results along with the models. To validate compliance with practical industrial requirements ( [section 2](https://arxiv.org/html/2407.01449v6#S2 "2 Problem Formulation & Related Work ‣ ColPali: Efficient Document Retrieval with Vision Language Models")), we also consider query latencies (R2) and indexing throughputs (R3).

### 3.2 Assessing Current Systems

Unstructured. We evaluate retrieval systems representative of those found in standard industrial RAG pipelines. As is common practice, we rely on the Unstructured off-the-shelf tool in the highest resolution settings to construct high-quality text chunks from PDF documents. Unstructured orchestrates the document parsing pipeline, relying on deep learning vision models to detect titles and document layouts (Ge et al., [2021](https://arxiv.org/html/2407.01449v6#bib.bib18 "")), OCR engines (Smith, [2007](https://arxiv.org/html/2407.01449v6#bib.bib47 "")) to extract text in non-native PDFs, specialized methods or models to detect and reconstruct tables, and implements a chunking strategy (by-title) that leverages the detected document structure to preserve section boundaries when concatenating texts. As is common practice, in our simplest Unstructured configuration (text-only), only textual elements are kept and figures, images, and tables are considered noisy information and are filtered out.

Unstructured + X. While Unstructured is a strong baseline by itself, we further augment Unstructured’s output by integrating the visual elements. In (+ OCR), tables, charts, and images are run through an OCR engine, processed by Unstructured, and chunked independently. In (+ Captioning), we set up a fully-fledged captioning strategy (Zhao et al., [2023](https://arxiv.org/html/2407.01449v6#bib.bib60 "")), in which we feed visual elements to a strong proprietary Vision Language Model (Claude-3 Sonnet (Anthropic, [2024](https://arxiv.org/html/2407.01449v6#bib.bib3 ""))) to obtain highly detailed textual descriptions of the elements.
Both strategies aim to integrate visual elements in the retrieval pipeline but incur significant latency and resource costs.

Embedding Model. To embed textual chunks, we evaluate Okapi BM25, the de facto standard sparse statistical retrieval method, and the dense encoder of BGE-M3 (Chen et al., [2024](https://arxiv.org/html/2407.01449v6#bib.bib10 "")), a multilingual neural method with SOTA performance in its size category. Chunks are embedded and scored independently, and page-level scores are obtained by max-pooling over the page’s chunk scores.

Contrastive VLMs. We also evaluate the strongest available vision-language embedding models; Jina CLIP (Koukounas et al., [2024](https://arxiv.org/html/2407.01449v6#bib.bib28 "")), Nomic Embed Vision (Nomic, [2024](https://arxiv.org/html/2407.01449v6#bib.bib41 "")), and SigLIP-So400m/14(Alabdulmohsin et al., [2023](https://arxiv.org/html/2407.01449v6#bib.bib1 "")).

Results. From a performance perspective, best results are obtained by combining the Unstructured parser with visual information, either from captioning strategies or by running OCR on the visual elements ( [Table 2](https://arxiv.org/html/2407.01449v6#S5.T2 "Table 2 ‣ 5 Results ‣ ColPali: Efficient Document Retrieval with Vision Language Models")). Little difference is seen between BM25 and BGE-M3 embeddings highlighting the visual information bottleneck. Contrastive VLMs lag behind. Beyond retrieval performance (R1), the indexing latencies (R2) reported in [Figure 2](https://arxiv.org/html/2407.01449v6#S3.F2 "Figure 2 ‣ 3.2 Assessing Current Systems ‣ 3 The ViDoRe Benchmark ‣ ColPali: Efficient Document Retrieval with Vision Language Models") illustrate that PDF parsing pipelines can be very lengthy, especially when incorporating OCR or captioning strategies. Querying latencies at runtime (R3) are very good for all evaluated systems (≤22 ms on a NVIDIA L4) due to fast query encoding and cosine similarity matching.

![Refer to caption](https://arxiv.org/html/2407.01449v6/x1.png)

Figure 2: Offline document indexing with ColPali is much simpler and faster compared to standard retrieval methods. The PDF Parser results are obtained following the Unstructured settings with BGE-M3 detailed in [subsection 3.2](https://arxiv.org/html/2407.01449v6#S3.SS2 "3.2 Assessing Current Systems ‣ 3 The ViDoRe Benchmark ‣ ColPali: Efficient Document Retrieval with Vision Language Models"). All indexing speeds are averaged per-page latencies. More details in [subsection B.4](https://arxiv.org/html/2407.01449v6#A2.SS4 "B.4 Latency computations ‣ Appendix B Implementation details ‣ ColPali: Efficient Document Retrieval with Vision Language Models")

## 4 Late interaction based Vision Retrieval

### 4.1 Architecture

Vision-Language Models.
Encouraged by their strong document understanding capabilities, we propose adapting recent VLMs for retrieval. The key concept is to leverage the alignment between output embeddings of text and image tokens acquired during multi-modal fine-tuning.
To this extent, we introduce ColPali, a Paligemma-3B extension that is capable of generating ColBERT-style multi-vector representations of text and images ( [Figure 1](https://arxiv.org/html/2407.01449v6#S1.F1 "Figure 1 ‣ 1 Introduction ‣ ColPali: Efficient Document Retrieval with Vision Language Models")).
PaliGemma-3B is a strong candidate due to its small size, the many released checkpoints fine-tuned for different image resolutions and tasks, and the promising performances on various document understanding benchmarks.
We add a projection layer to map each of the language model’s output token embeddings (whether from text or image tokens) to a vector space of reduced dimension D=128 as used in the ColBERT paper (Khattab & Zaharia, [2020](https://arxiv.org/html/2407.01449v6#bib.bib26 "")) to keep lightweight bag-of-embedding representations.

Late Interaction.
Given query q and document d, we denote as E_q∈ℝ^(N_q×D) and E_d∈ℝ^(N_d×D) their respective multi-vector representation in the common embedding space ℝ^D, where N_q and N_d are respectively the number of vectors in the query and in the document page embeddings. The late interaction operator, LI(q,d), is the sum over all query vectors E_q^(j), of its maximum dot product with each of the N_d document embedding vectors E_d^(1:N_d).

|     |     |     |     |
| --- | --- | --- | --- |
|  | LI(q,d)=∑_(i∈[1,N_q]) max_(j∈[1,N_d]) ⟨E_q^(i) | E_d^(j)⟩ |  | (1) |

Contrastive Loss.
The Late Interaction operation is fully differentiable, enabling backpropagation.
Let a batch {q_k,d_k}\_{k∈[1,b]} composed of b query-page pairs, where for all k∈[1,b], the document page d_k is the document corresponding to query q_k.
Following Khattab & Zaharia ( [2020](https://arxiv.org/html/2407.01449v6#bib.bib26 "")), we define our in-batch contrastive loss ℒ as the softmaxed cross-entropy of the positive scores s_k^+ = LI(q_k,d_k) w.r.t. to the maximal in-batch negative scores s_k^- = max_{l, l≠k} LI(q_k, d_l):

|     |     |     |     |
| --- | --- | --- | --- |
|  | ℒ=−(1/b)∑_{k=1}^b log\[exp(s_k^+)/[exp(s_k^+)+exp(s_k^-)]\]= (1/b)∑_{k=1}^b log(1+exp(s_k^−−s_k^+)) |  | (2) |

### 4.2 Model training

Dataset. Our training dataset of 118,695 query-page pairs is comprised of train sets of openly available academic datasets (63%) and a synthetic dataset made up of pages from web-crawled PDF documents and augmented with VLM-generated (Claude-3 Sonnet) pseudo-questions (37%). Dataset split details are given in [subsection A.3](https://arxiv.org/html/2407.01449v6#A1.SS3 "A.3 Training Dataset ‣ Appendix A Benchmark Datasets ‣ ColPali: Efficient Document Retrieval with Vision Language Models").
Our training set is fully English by design, enabling us to study zero-shot generalization to non-English languages. We explicitly verify no multi-page PDF document is used both ViDoRe and in the train set to prevent evaluation contamination. A validation set is created with 2% of the samples to tune hyperparameters. We openly release the training dataset at [https://huggingface.co/datasets/vidore/colpali_train_set](https://huggingface.co/datasets/vidore/colpali_train_set "") for reproducibility and to encourage further research.

Parameters. All models are trained for 1 epoch on the train set. Unless specified otherwise, we train models in bfloat16 format, use low-rank adapters (LoRA, Hu et al. ( [2021](https://arxiv.org/html/2407.01449v6#bib.bib21 ""))) with α=32 and r=32 on the transformer layers from the language model, as well as the final randomly initialized projection layer, and use a paged_adamw_8bit optimizer. We train on an 8 GPU setup with data parallelism, a learning rate of 5e-5 with linear decay with 2.5% warmup steps, and a batch size of 32.

Query Augmentation. As in Khattab & Zaharia ( [2020](https://arxiv.org/html/2407.01449v6#bib.bib26 "")), we append 5 <unused0> tokens to the query tokens to serve as a soft, differentiable query expansion or re-weighting mechanism.

## 5 Results

|     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | ArxivQ | DocQ | InfoQ | TabF | TATQ | Shift | AI | Energy | Gov. | Health. | Avg. |
| Unstructured text-only |  |  |  |  |  |  |  |  |  |  |  |
| - BM25 | - | 34.1 | - | - | 44.0 | 59.6 | 90.4 | 78.3 | 78.8 | 82.6 | - |
| - BGE-M3 | - | 28.4 ↓5.7 | - | - | 36.1 ↓7.9 | 68.5 ↑8.9 | 88.4 ↓2.0 | 76.8 ↓1.5 | 77.7 ↓1.1 | 84.6 ↑2.0 | - |
| Unstructured + OCR |  |  |  |  |  |  |  |  |  |  |  |
| - BM25 | 31.6 | 36.8 | 62.9 | 46.5 | 62.7 | 64.3 | 92.8 | 85.9 | 83.9 | 87.2 | 65.5 |
| - BGE-M3 | 31.4 ↓0.2 | 25.7 ↓11.1 | 60.1 ↓2.8 | 70.8 ↑24.3 | 50.5 ↓12.2 | 73.2 ↑8.9 | 90.2 ↓2.6 | 83.6 ↓2.3 | 84.9 ↑1.0 | 91.1 ↑3.9 | 66.1 ↑0.6 |
| Unstructured + Captioning |  |  |  |  |  |  |  |  |  |  |  |
| - BM25 | 40.1 | 38.4 | 70.0 | 35.4 | 61.5 | 60.9 | 88.0 | 84.7 | 82.7 | 89.2 | 65.1 |
| - BGE-M3 | 35.7 ↓4.4 | 32.9 ↓5.4 | 71.9 ↑1.9 | 69.1 ↑33.7 | 43.8 ↓17.7 | 73.1 ↑12.2 | 88.8 ↑0.8 | 83.3 ↓1.4 | 80.4 ↓2.3 | 91.3 ↑2.1 | 67.0 ↑1.9 |
| Contrastive VLMs |  |  |  |  |  |  |  |  |  |  |  |
| Jina-CLIP | 25.4 | 11.9 | 35.5 | 20.2 | 3.3 | 3.8 | 15.2 | 19.7 | 21.4 | 20.8 | 17.7 |
| Nomic-vision | 17.1 | 10.7 | 30.1 | 16.3 | 2.7 | 1.1 | 12.9 | 10.9 | 11.4 | 15.7 | 12.9 |
| SigLIP (Vanilla) | 43.2 | 30.3 | 64.1 | 58.1 | 26.2 | 18.7 | 62.5 | 65.7 | 66.1 | 79.1 | 51.4 |
| Ours |  |  |  |  |  |  |  |  |  |  |  |
| SigLIP (Vanilla) | 43.2 | 30.3 | 64.1 | 58.1 | 26.2 | 18.7 | 62.5 | 65.7 | 66.1 | 79.1 | 51.4 |
| BiSigLIP (+fine-tuning) | 58.5 ↑15.3 | 32.9 ↑2.6 | 70.5 ↑6.4 | 62.7 ↑4.6 | 30.5 ↑4.3 | 26.5 ↑7.8 | 74.3 ↑11.8 | 73.7 ↑8.0 | 74.2 ↑8.1 | 82.3 ↑3.2 | 58.6 ↑7.2 |
| BiPali (+LLM) | 56.5 ↓-2.0 | 30.0 ↓-2.9 | 67.4 ↓-3.1 | 76.9 ↑14.2 | 33.4 ↑2.9 | 43.7 ↑17.2 | 71.2 ↓-3.1 | 61.9 ↓-11.7 | 73.8 ↓-0.4 | 73.6 ↓-8.8 | 58.8 ↑0.2 |
| ColPali (+Late Inter.) | 79.1 ↑22.6 | 54.4 ↑24.5 | 81.8 ↑14.4 | 83.9 ↑7.0 | 65.8 ↑32.4 | 73.2 ↑29.5 | 96.2 ↑25.0 | 91.0 ↑29.1 | 92.7 ↑18.9 | 94.4 ↑20.8 | 81.3 ↑22.5 |

Table 2: Comprehensive evaluation of baseline models and our proposed method on ViDoRe. Results are presented using nDCG@5 metrics, and illustrate the impact of different components. Text-only metrics are not computed for benchmarks with only visual elements.

### 5.1 Performance (R1)

We show performance is achieved iteratively through the combination of three factors; (1) a carefully crafted task-specific dataset, (2) pairing a pretrained LLM to a vision model to better leverage text semantics from the image, and (3) using multi-vector embeddings rather than a single vector representation to better capture the vast amount of visual information present in a document.

Fine-tuning a Vision Model on a document retrieval oriented dataset: BiSigLIP. SigLIP is a strong vision-language bi-encoder producing single vector embeddings, and pretrained on billions of image-text pairs from the English split of WebLI (Chen et al., [2023](https://arxiv.org/html/2407.01449v6#bib.bib11 "")). Further fine-tuning the textual component of this model on our document-oriented dataset (BiSigLIP) yields clear improvements across the board, particularly on figure retrieval (ArxivQA) and table retrieval tasks (TabFQuAD).

Feeding image patches to a LLM: BiPali. In the PaliGemma model architecture, SigLIP-generated patch embeddings are fed to a text language model and we can obtain LLM contextualized output patch embeddings. This technique aligns the image token representations with the text token embeddings in the LLM’s embeddings space, and augments the vision model embeddings with the language model’s text understanding capabilities. We average pool these representations to obtain a single dense vector, effectively creating a PaliGemma bi-encoder model (BiPali). After fine-tuning on the training dataset, we obtain a model that performs slightly worse in English than the tuned BiSigLIP variant. However, we see notable improvements in French tasks, indicating that BiPali’s LLM (Gemma 2B) helps multilingual text understanding. This is particularly notable as our training dataset does not contain non-English samples.

Leveraging Multi-Vector Embeddings through Late Interaction: ColPali.
One benefit of inputting image patch embeddings through a language model is that they are natively mapped to a latent space similar to the textual input (query). This enables leveraging the ColBERT strategy to construct one embedding per image patch token, and at inference compute all interactions between text tokens and image patches, resulting in a step-change improvement in performance compared to BiPali.
Results in [Table 2](https://arxiv.org/html/2407.01449v6#S5.T2 "Table 2 ‣ 5 Results ‣ ColPali: Efficient Document Retrieval with Vision Language Models") show that our ColPali model also largely outperforms the strong baselines based on Unstructured and captioning, as well as all evaluated text-image embedding models. The difference is particularly stark on the more visually complex benchmark tasks, such as InfographicVQA, ArxivQA, and TabFQuAD, respectively representing infographics, figures, and tables. However, text-centric documents are also better retrieved by the ColPali models across all evaluated domains and languages, making our approach the overall best-performing document-retrieval model.

Negative Results. For extensiveness, we also train ColSigLIP, a late interaction variant of the BiSigLIP model but obtain abysmal performances. We attribute this to the large gaps w.r.t. SigLIP’s pre-training, in which only a pooled latent representation is used in the contrastive loss, which does not optimize the representations of individual patch and token embeddings. Similarly, we train a BiSigLIPPaliGemma variant, in which we retrieve the image representations from the SigLIP model that has been further updated by PaliGemma fine-tuning, and use the text representations from PaliGemma’s text model. After fine-tuning on our dataset, performance is severely inferior to SigLIPVanilla which simply encodes with SigLIP’s original text and vision components. This indicates a logical misalignment between SigLIP embeddings, and Gemma embeddings after PaliGemma training.

### 5.2 Latencies & Memory Footprint

Online Querying. (R2) Logically, querying latencies differ between ColPali and a BGE-M3 embedding model. For BGE, encoding takes about 22 ms for 15 tokens, while encoding a query with ColPali’s language model takes about 30 ms. For smaller corpus sizes, computing the late interaction operation induces marginally small overheads (~1 ms per 1000 pages in the corpus), and the cosine similarity computation between bi-encoder vectors is even faster. Optimized late interaction engines enable to easily scale corpus sizes to millions of documents with reduced latency degradations.

Offline Indexing. (R3)
Standard retrieval methods using bi-encoders represent each chunk as a single vector embedding, which is easy to store and fast to compute. However, processing a PDF to get the different chunks is the most time-consuming part (layout detection, OCR, chunking), and using captioning to handle multimodal data will only exacerbate this already lengthy process. On the other hand, ColPali directly encodes pages from their image representation. Although the model is larger than standard retrieval encoders, skipping the preprocessing allows large speedups at indexing. As pages are embedded end-to-end in single forward pass, the VRAM usage depends exclusively on the sequence length (number of patches per image) which is fixed as well, enabling efficient batching strategies to fully leverage hardware acceleration. ColPali also benefits from most LLM efficiency improvements introduced in the ecosystem such as Flash Attention.

Storage Footprint. Our method requires storing a vector per image patch, along with 6 extra text tokens “Describe the image” concatenated to image patches. We project each PaliGemma vector to a lower dimensional space (D=128) to maximize efficiency, leading to a memory footprint of 257.5 KB per page. Importantly, the memory footprint of the naive ColBERT indexing strategy can be drastically improved through compression and clustering mechanisms.

Token pooling.
Token pooling is a CRUDE-compliant method (document addition/deletion-friendly) that aims to reduce the amount of multi-vector embeddings. For ColPali, many image patches share redundant information, e.g. white background patches. By pooling these patches together, we can reduce the amount of embeddings while retaining most information. Retrieval performance with hierarchical mean token pooling on image embeddings is shown in [Figure 3](https://arxiv.org/html/2407.01449v6#S5.F3 "Figure 3 ‣ 5.3 Interpretability ‣ 5 Results ‣ ColPali: Efficient Document Retrieval with Vision Language Models") (left).
With a pool factor of 3, the total number of vectors is reduced by 66.7% while 97.8% of the original performance is maintained. We note that the Shift dataset—composed of the most text-dense documents—is a clear outlier, showcasing more information dense documents contain less redundant patches and may be prone to worse performance degradation with such pooling techniques.

### 5.3 Interpretability

![Refer to caption](https://arxiv.org/html/2407.01449v6/x2.png)

![Refer to caption](https://arxiv.org/html/2407.01449v6/extracted/6240861/images/similarity_map_energy.png)

Figure 3: (Left: Token Pooling) Relative performance degradation when reducing the number of stored embeddings per document. (Right: Interpretability) For each term in a user query, ColPali identifies the most relevant document image patches (highlighted zones) and computes a query-to-page matching score.

By superimposing the late interaction heatmap on top of the original image, we can visualize the most salient image patches with respect to each term of the query, yielding interpretable insights into model focus zones. As epitomized in [Figure 3](https://arxiv.org/html/2407.01449v6#S5.F3 "Figure 3 ‣ 5.3 Interpretability ‣ 5 Results ‣ ColPali: Efficient Document Retrieval with Vision Language Models") (right), we observe ColPali exhibits strong OCR capabilities as both the words “hourly” and “hours” present a high similarity score with the query token <_hour>. We also note particular focus on other non-trivial image features such as the x-axis representing hours being salient. Other visualization examples are shown in [Appendix D](https://arxiv.org/html/2407.01449v6#A4 "Appendix D More similarity maps ‣ ColPali: Efficient Document Retrieval with Vision Language Models").

## 6 Ablation study

We run various ablations to better understand the mechanisms at play. By default, result deltas reported below refer to nDCG@5 values averaged over all ViDoRe tasks.

Tradeoffs between model size and the number of image patches. We train a variant of PaliGemma with half the number of image patches (512). While we observe a clear performance degradation with respects to to the 1024-patch ColPali model (−24.8 nDCG@5), memory usage is much lower.
As an alternative to PaliGemma, we train Idefics2-8B, a VLM with a similar architecture and based on a Mistral-7B language backbone and a SigLIP vision encoder paired with a perceiver resampler. The most notable differences with PaliGemma lie in the size of the language model (2B and 7B resp.) and the number of image patches (between 512 and 2048 for PaliGemma, and 64 post-resampling for Idefics2). Our results suggest better language models enable more efficient representations of image embeddings - ColIdefics2 with 64 patches largely outperforms ColPali with 512 patches (+20.1 nDCG@5). However ColIdefics2 (64) remains less accurate than ColPali (1024) (−4.7 nDCG@5) while being about twice as slow in terms of training and inference latency.
These results suggest there are tradeoffs between performance (R1), latencies during online querying (R2) and offline indexation phases (R3), and index memory size.

Unfreezing the vision component. We train a ColPali variant by also backpropagating through and updating the vision encoder and the projection layer. This leads to a slight performance degradation (−0.7 nDCG@5). These conclusions may change with larger scales of training data.

Impact of “query augmentation” tokens. In ColBERT, special tokens are concatenated to the input query to serve as soft query augmentation buffers. Training without these tokens, we observe no significant performance difference in the English benchmarks. However, performance on the French tasks seems to improve (+9.8 nDCG@5 on Shift, +6.3 nDCG@5 on TabFQuAD).

Impact of the Pairwise CE loss. Training with an in-batch negative contrastive loss, instead of the pairwise CE loss that only considers the hardest negative sample, leads to a slight performance degradation (−1.6 nDCG@5) on the aggregated benchmark.

Adapting models to new tasks. Contrary to more complex multi-step retrieval pipelines, ColPali can be trained end-to-end, directly optimizing the downstream retrieval task which greatly facilitates fine-tuning to boost performance on specialized domains, multilingual retrieval, or specific visual elements the model struggles with. To demonstrate, we add 1552 samples representing French tables and associated queries to the training set. This represents the only French data in the training set, with all other examples being kept unchanged. We see clear nDCG@5 improvements (+2.6) and even starker Recall@1 gains (+5) on the TabFQuAD benchmark, with no performance degradation on the rest of the benchmark tasks (+0.4 nDCG@5 overall).

Better VLMs lead to better visual retrievers. As improved VLMs are released, it is interesting to observe if improved performances on generative tasks translate once these models are adapted for image retrieval tasks through ColPali training strategies. We train the recently released Qwen2-VL 2B, a SOTA 2 billion parameter generative VLM, with the same data and training strategy, obtaining ColQwen2-VL. To approximately match ColPali’s memory requirements, we limit the number of image patches to 768, slightly less than ColPali’s 1024 patches. We observe clear performance improvements of +5.3 nDCG@5 values over ColPali showcasing clear performance correlations between generative benchmarks performance and retrieving metrics.

Out-of-domain generalization. Some of the datasets in the ViDoRe benchmark have train sets, which we have integrated within the ColPali train set (eg. academic tasks). This is standard in embedding models, and while ColPali also exhibits strong performance on tasks in which this is not the case (French data is never seen by the model during training for instance), it remains interesting to evaluate model performance when training is done on a fully disjoint data distribution. We train a ColPali variant solely using the recent DocMatix dataset, a large scale, synthetically annotated visual document question answering dataset, which we subsample to obtain a comparably-sized train set. Results on ViDoRe show the performance drop is minor (−2.2 nDCG@5), still outperforming the closest baseline method by over 12 points. These results showcase ColPali generalizes well outside of its training distribution, and demonstrate that our results are not unreasonably boosted with respect to baselines (BGE-M3) that cannot be fine-tuned on the same data.

## 7 Conclusions

In this work, we introduced the Visual Document Retrieval Benchmark (ViDoRe), which evaluates document retrieval systems in realistic settings involving visually complex documents. We demonstrated that current retrieval pipelines and contrastive vision-language models struggle to efficiently exploit visual information embedded in documents, leading to suboptimal performance. To address this, we presented ColPali, a novel retrieval method that leverages Vision-Language Models to create high-quality, multi-vector embeddings purely from visual document features. ColPali largely outperforms the best existing document retrieval methods while enabling faster corpus indexing times and maintaining low querying latencies, thus circumventing many pain points of modern document retrieval applications. We hope to drive industrial adoption, and to encourage future work by publicly releasing the ViDoRe benchmark, the data, the codebase, and all models and baselines from our work.

Future Work.
Beyond performance improvements that could be obtained through better data, backbone models or training strategies, our vision at term is to combine visual retrieval systems and visually grounded query answering to create end-to-end RAG systems that purely function from image features. This idea is supported by concurrent work showcasing the strong promises of VLMs for visual QA, and may eventually become a new industrial standard for document processing. In this line of work, reliability is key, and confidence estimation techniques for Information Retrieval methods could become central to implement abstention mechanisms, and are particularly interesting given the information rich multi-vector scoring mechanisms of late interaction systems.
Expanding benchmarking efforts to cover more languages, modalities, and tasks is also a crucial future research direction.

### Original URL
https://arxiv.org/html/2407.01449v6
</details>

---
<details>
<summary>Image understanding  |  Gemini API  |  Google AI for Developers</summary>

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

![A table with cupcakes, with the wooden and glass objects highlighted](https://ai.google.dev/static/gemini-api/docs/images/segmentation.jpg)An example segmentation output with objects and segmentation masks

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

### Original URL
https://ai.google.dev/gemini-api/docs/image-understanding
</details>

---
<details>
<summary>Multimodal RAG with Colpali, Milvus and VLMs</summary>

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

[![Page 4](https://saumitra.me/2024/covid-page-4.png)](https://saumitra.me/2024/covid-page-4.png)

**Page 8: A table showing cases and TPR**

[![Page 8](https://saumitra.me/2024/covid-page-8.png)](https://saumitra.me/2024/covid-page-8.png)

**Page 9: A table showing cases by country**

[![Page 9](https://saumitra.me/2024/covid-page-9.png)](https://saumitra.me/2024/covid-page-9.png)

It would be difficult to extract data from these pages as text in a manner which can be used for querying.
We want to show user the answer and source page(s) from the PDF which contains the answer, like below:

[![rag-demo-screenshot](https://saumitra.me/2024/rag-demo-screenshot.png)](https://saumitra.me/2024/rag-demo-screenshot.png)

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

### Original URL
https://huggingface.co/blog/saumitras/colpali-milvus-multimodal-rag
</details>

---
<details>
<summary>Google Generative AI Embeddings (AI Studio & Gemini API) | 🦜️🔗 LangChain</summary>

Connect to Google's generative AI embeddings service using the `GoogleGenerativeAIEmbeddings` class, found in the [langchain-google-genai](https://pypi.org/project/langchain-google-genai/) package.

This will help you get started with Google's Generative AI embedding models (like Gemini) using LangChain. For detailed documentation on `GoogleGenerativeAIEmbeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/v0.2/api_reference/google_genai/embeddings/langchain_google_genai.embeddings.GoogleGenerativeAIEmbeddings.html).

## Overview

### Integration details

| Provider | Package |
| --- | --- |
| [Google Gemini](https://python.langchain.com/docs/integrations/text_embedding/google-generative-ai) | [langchain-google-genai](https://python.langchain.com/api_reference/google_genai/embeddings/langchain_google_genai.embeddings.GoogleGenerativeAIEmbeddings.html) |

## Setup

To access Google Generative AI embedding models you'll need to create a Google Cloud project, enable the Generative Language API, get an API key, and install the `langchain-google-genai` integration package.

### Credentials

To use Google Generative AI models, you must have an API key. You can create one in Google AI Studio. See the [Google documentation](https://ai.google.dev/gemini-api/docs/api-key) for instructions.

Once you have a key, set it as an environment variable `GOOGLE_API_KEY`:

```python
import getpass
import os

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")
```

To enable automated tracing of your model calls, set your [LangSmith](https://docs.smith.langchain.com/) API key:

```python
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter your LangSmith API key: ")
```

## Installation

```bash
%pip install --upgrade --quiet  langchain-google-genai
```

## Usage

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
vector = embeddings.embed_query("hello, world!")
vector[:5]
```

**API Reference:** [GoogleGenerativeAIEmbeddings](https://python.langchain.com/api_reference/google_genai/embeddings/langchain_google_genai.embeddings.GoogleGenerativeAIEmbeddings.html)

```python
[-0.024917153641581535,\
 0.012005362659692764,\
 -0.003886754624545574,\
 -0.05774897709488869,\
 0.0020742062479257584]
```

## Batch

You can also embed multiple strings at once for a processing speedup:

```python
vectors = embeddings.embed_documents(
    [\
        "Today is Monday",\
        "Today is Tuesday",\
        "Today is April Fools day",\
    ]
)
len(vectors), len(vectors[0])
```

```python
(3, 3072)
```

## Indexing and Retrieval

Embedding models are often used in retrieval-augmented generation (RAG) flows, both as part of indexing data as well as later retrieving it. For more detailed instructions, please see our [RAG tutorials](https://python.langchain.com/docs/tutorials/).

Below, see how to index and retrieve data using the `embeddings` object we initialized above. In this example, we will index and retrieve a sample document in the `InMemoryVectorStore`.

```python
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

```python
'LangChain is the framework for building context-aware reasoning applications'
```

## Task type

`GoogleGenerativeAIEmbeddings` optionally support a `task_type`, which currently must be one of:

- `SEMANTIC_SIMILARITY`: Used to generate embeddings that are optimized to assess text similarity.
- `CLASSIFICATION`: Used to generate embeddings that are optimized to classify texts according to preset labels.
- `CLUSTERING`: Used to generate embeddings that are optimized to cluster texts based on their similarities.
- `RETRIEVAL_DOCUMENT`, `RETRIEVAL_QUERY`, `QUESTION_ANSWERING`, and `FACT_VERIFICATION`: Used to generate embeddings that are optimized for document search or information retrieval.
- `CODE_RETRIEVAL_QUERY`: Used to retrieve a code block based on a natural language query, such as sort an array or reverse a linked list. Embeddings of the code blocks are computed using `RETRIEVAL_DOCUMENT`.

By default, we use `RETRIEVAL_DOCUMENT` in the `embed_documents` method and `RETRIEVAL_QUERY` in the `embed_query` method. If you provide a task type, we will use that for all methods.

```bash
%pip install --upgrade --quiet  matplotlib scikit-learn
```

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

query_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07", task_type="RETRIEVAL_QUERY"
)
doc_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07", task_type="RETRIEVAL_DOCUMENT"
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

**API Reference:** [GoogleGenerativeAIEmbeddings](https://python.langchain.com/api_reference/google_genai/embeddings/langchain_google_genai.embeddings.GoogleGenerativeAIEmbeddings.html)

```python
Document 1
Cosine similarity with query: 0.7892893360164779
---
Document 2
Cosine similarity with query: 0.5438283285204146
---
```

## API Reference

For detailed documentation on `GoogleGenerativeAIEmbeddings` features and configuration options, please refer to the [API reference](https://python.langchain.com/api_reference/google_genai/embeddings/langchain_google_genai.embeddings.GoogleGenerativeAIEmbeddings.html).

## Additional Configuration

You can pass the following parameters to ChatGoogleGenerativeAI in order to customize the SDK's behavior:

- `client_options`: [Client Options](https://googleapis.dev/python/google-api-core/latest/client_options.html#module-google.api_core.client_options) to pass to the Google API Client, such as a custom `client_options["api_endpoint"]`
- `transport`: The transport method to use, such as `rest`, `grpc`, or `grpc_asyncio`.

## Related

- Embedding model [conceptual guide](https://python.langchain.com/docs/concepts/embedding_models/)
- Embedding model [how-to guides](https://python.langchain.com/docs/how_to/#embedding-models)

### Original URL
https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai/
</details>

---
<details>
<summary>Start with a prebuilt agent</summary>

This guide shows you how to set up and use LangGraph's **prebuilt**, **reusable** components, which are designed to help you construct agentic systems quickly and reliably.

If you haven't already, install LangGraph and LangChain:

```md-code__content
pip install -U langgraph "langchain[anthropic]"
```

LangChain is installed so the agent can call the [model](https://python.langchain.com/docs/integrations/chat/).

To create an agent, use `create_react_agent`:

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

To configure an LLM with specific parameters, such as temperature, use `init_chat_model`:

```md-code__content
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

Prompts instruct the LLM how to behave. Add one of the following types of prompts:

- **Static**: A string is interpreted as a **system message**.
- **Dynamic**: A list of messages generated at **runtime**, based on input or configuration.

Define a fixed prompt string or list of messages:

```md-code__content
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

```md-code__content
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

To allow multi-turn conversations with an agent, you need to enable persistence by providing a `checkpointer` when creating an agent. At runtime, you need to provide a config containing `thread_id` — a unique identifier for the conversation (session):

```md-code__content
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

To produce structured responses conforming to a schema, use the `response_format` parameter. The schema can be defined with a `Pydantic` model or `TypedDict`. The result will be accessible via the `structured_response` field.

```md-code__content
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

### Original URL
https://langchain-ai.github.io/langgraph/agents/agents/
</details>

---
<details>
<summary>Complex Document Recognition: OCR Doesn’t Work and Here’s How You Fix It | HackerNoon</summary>

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

By fine-tuning these tools alone, we’ve seen a 200 times decrease in document processing speed.If you add an OCR engine into the equation, like Tesseract, the text recognition quality can be increased up to 99.9%.

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

### Original URL
https://hackernoon.com/complex-document-recognition-ocr-doesnt-work-and-heres-how-you-fix-it
</details>

---
<details>
<summary>What are some real-world applications of multimodal AI?</summary>

# What are some real-world applications of multimodal AI?

Multimodal AI, which processes and combines different data types like text, images, audio, and sensor inputs, has practical applications across industries. By integrating multiple data sources, these systems improve accuracy and functionality in tasks that require contextual understanding. Below are three key areas where multimodal AI is being applied effectively today.

In healthcare, multimodal AI enhances diagnostics and patient care by merging medical imaging, electronic health records (EHRs), and sensor data. For example, a system might analyze a chest X-ray (image), a patient’s symptom descriptions (text), and vital signs from wearables (sensor data) to detect pneumonia. Models like Google’s **Med-PaLM 2** combine vision and language processing to interpret radiology images alongside clinical notes, reducing misdiagnosis risks. Another use case is monitoring postoperative recovery: wearable devices track movement and heart rate, while speech analysis detects pain or fatigue in a patient’s voice, enabling proactive interventions.

Autonomous vehicles rely heavily on multimodal AI to fuse data from cameras, LiDAR, radar, and GPS. A self-driving car processes road signs (visual data), pedestrian movements (video), and proximity sensor readings to navigate safely. Tesla’s Autopilot, for instance, uses neural networks to combine camera feeds with ultrasonic sensors, improving object detection in varied lighting or weather. Similarly, companies like Waymo train models to correlate map data with real-time sensor inputs, ensuring precise localization and path planning. This redundancy across modalities helps address limitations of single-sensor systems, such as camera failures in low light.

Customer service and content moderation also benefit from multimodal approaches. Virtual assistants like Amazon’s Alexa process voice commands while analyzing user history (text) to personalize responses. In moderation, platforms like YouTube use AI to flag harmful content by scanning video frames (images), audio for hate speech, and user comments (text) simultaneously. For example, a post containing violent imagery and threatening text would be detected faster than if each modality were analyzed separately. Tools like **OpenAI’s CLIP** enable cross-modal matching, such as linking inappropriate images to their descriptive captions, improving accuracy in filtering violations. These systems reduce reliance on manual review while scaling to handle large data volumes.

![Multimodal Image Search](https://milvus.io/images/demos/multimodal-image-search.png)

### Multimodal Image Search

Upload images and edit text to enhance intuitive image searches using advanced retrieval technology.

### Original URL
https://milvus.io/ai-quick-reference/what-are-some-realworld-applications-of-multimodal-ai
</details>

---
<details>
<summary>What Is Optical Character Recognition (OCR)? Explained</summary>

# What Is Optical Character Recognition (OCR)?

![optical character recognition](https://blog.roboflow.com/content/images/size/w1200/2024/04/image-730.webp)

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

![](https://blog.roboflow.com/content/images/2024/04/image-733.webp)_Application of OCR on the text of a book._ [_Source_](https://www.edenai.co/post/optical-character-recognition-ocr-which-solution-to-choose?ref=blog.roboflow.com).

## How Optical Character Recognition Works

Let's discuss the typical steps modern OCR software uses to read text:

1. **Image pre-processing**: After an image has been collected, the image undergoes pre-processing to enhance image quality, improving recognition. Pre-processing may involve resizing, contrast enhancement, binarization, noise reduction, and other techniques.
2. **Text Detection**: Using a specialized deep-learning model trained on large datasets of images and text, the computer vision model detects regions in the input image that likely contain text. This process is usually a crucial step.
3. **Layout Analysis**: After detecting text regions, the computer vision model conducts layout analysis to determine the structure and order of the text in the image. This step ensures the preservation of context and organizes the output for readability, but is not run by all OCR systems.
4. **Text Recognition**: Detected text regions pass through a deep learning-based text recognition model, utilizing a combination of convolutional neural networks (CNNs) and recurrent neural networks (RNNs). This model recognizes individual characters and words in the input image, converting them into machine-readable text.
5. **Language Model**: The final output undergoes post-processing to remove noise, correct spelling mistakes, and enhance overall accuracy. The predicted sequence of characters may contain errors, especially for long or uncommon words. Language models, acting as word processors, refine the output by predicting the probability of a sequence of words based on the input image. Statistical models and advanced methods, including deep learning, may be employed for this purpose.


![](https://blog.roboflow.com/content/images/2024/04/image-738.webp)_An example OCR system pipeline._

Having acquired an understanding of how OCR operates, let's examine its algorithms and investigate their operational mechanisms, covering the old and the new.

## Traditional Approaches to OCR

The first OCR algorithms rooted in image processing were typically rule-based systems. One well-known OCR that uses this approach is [Tesseract](https://github.com/tesseract-ocr/tesseract?ref=blog.roboflow.com). These systems relied on manually crafted features and heuristic rules to identify characters within images. The approach involved segmenting characters into individual units and applying a set of rules for character classification.

However, the accuracy and performance of these algorithms were often constrained due to the intricate process of developing and fine-tuning the necessary handcrafted features and rules for effective recognition.

### Tesseract

Tesseract, an open-source optical character recognition engine, originated at Hewlett-Packard Laboratories in the 1980s and subsequently became open-source in 2005.

Initially designed to recognize English text exclusively, Tesseract has evolved into a versatile OCR engine. Working from traditional image processing principles, which involves manual logic unlike the deep learning processes in modern systems, Tesseract analyzes images to identify patterns for character recognition.

First, Tesseract preprocesses the image to enhance input quality, a step which encompasses tasks like contrast improvement and noise removal. Following this, Tesseract employs feature extraction techniques, including edge detection and pattern recognition, to identify and recognize characters.

![](https://blog.roboflow.com/content/images/2024/04/image-741.webp)_Tesseract OCR engine pipeline._ [_Source_](https://www.researchgate.net/figure/Tesseract-OCR-engine-architecture_fig4_265087843?ref=blog.roboflow.com).

## Deep Learning Approaches to Optical Character Recognition

With the rise of deep learning, the integration of neural networks into OCR systems has gained substantial popularity. In particular, deep learning methodologies like [Convolutional Neural Networks](https://blog.roboflow.com/what-is-a-convolutional-neural-network/) and Long Short-Term Memory networks are leveraged, for precise text recognition. Neural networks regularly achieve better performance than traditional OCR techniques.

In recent years, there has also been a surge in novel approaches that leverage pre-trained image and text [Transformers](https://blog.roboflow.com/what-is-a-transformer/), a deep learning architecture. Transformers are ushering in a new era of end-to-end optical word recognition.

### PaddleOCR

[Paddle OCR](https://arxiv.org/abs/2009.09941?ref=blog.roboflow.com) is an open-source engine developed by Baidu's PaddlePaddle team. Leveraging deep learning techniques, including CNNs and recurrent neural networks, Paddle OCR excels in accurate text recognition. It comprises two key components: the detector and the extractor. The detector is tasked with pinpointing text within an image or document. It employs various algorithms, such as [EAST (Efficient and Accurate Scene Text)](https://paperswithcode.com/paper/east-an-efficient-and-accurate-scene-text?ref=blog.roboflow.com) or [DB (Differentiable Binarization)](https://arxiv.org/abs/1911.08947?ref=blog.roboflow.com) detectors, to identify text regions.

![](https://blog.roboflow.com/content/images/2024/04/image-745.webp)_DB (Differentiable Binarization) architecture._ [_Source_](https://arxiv.org/pdf/2009.09941.pdf?ref=blog.roboflow.com).

After the detector locates the text, the extractor comes into play, retrieving the text from the image. It employs a blend of Convolutional Neural Networks and Recurrent Neural Networks for precise text recognition. CNNs are utilized to extract features from the text, while RNNs play a crucial role in recognizing the sequence of characters.

![](https://blog.roboflow.com/content/images/2024/04/image-748.webp)_CRNN Extractor architecture._ [_Source_](https://arxiv.org/pdf/1507.05717.pdf?ref=blog.roboflow.com).

Paddle OCR stands out for its remarkable speed, making it among the swiftest OCR engines. Its efficiency is attributed to the utilization of parallel computing and GPU acceleration. This feature renders it particularly suitable for extensive OCR tasks, including document scanning and image recognition. Moreover, its adaptability shines through as it can be tailored and fine-tuned for specific tasks and datasets, enhancing its versatility and robustness in various OCR applications.

### TrOCR

[Transformer-based Optical Character Recognition (TrOCR)](https://arxiv.org/abs/2109.10282?ref=blog.roboflow.com) is one of many transformer-based [OCR models](https://blog.roboflow.com/best-ocr-models-text-recognition/). In contrast to traditional OCR systems, TrOCR adopts a methodology where both input image processing and the generation of corresponding text output occur within a single model.

The encoder segment of TrOCR employs a transformer-based architecture to handle the input image, segmenting it into a grid of patches and extracting visual features from each patch. Simultaneously, the decoder component utilizes a transformer-based model to produce the relevant text output, incorporating the visual features extracted from the image.

![](https://blog.roboflow.com/content/images/2024/04/image-752.webp)_TrOCR Architecture._ [_Source_](https://arxiv.org/pdf/2109.10282.pdf?ref=blog.roboflow.com).

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

### Original URL
https://blog.roboflow.com/what-is-optical-character-recognition-ocr/
</details>

---
<details>
<summary>The 8 best AI image generators in 2025 | Zapier</summary>

# The 8 best AI image generators in 2025

## Get the best AI-generated images using text-to-image AI.

By Harry Guinness · May 23, 2025

![Hero image with the logos of the best AI image generator tools](https://images.ctfassets.net/lzny33ho1g45/2olcy4TVSWAjqy5dsxLNZd/09b4a18346af97076615d5f1d1407c39/best-ai-image-generator-hero.jpg?fm=jpg&q=31&fit=thumb&w=1520&h=760)

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

![An image made with Midjourney using the prompt "an impressionist oil painting of a Canadian man riding a moose through a forest of maple trees"](https://images.ctfassets.net/lzny33ho1g45/2udOp4paDgOh5HpqG5JRAQ/18abc9476c4705aacf3609edcec4f945/image8.jpeg?)

I made this with Midjourney using the prompt "an impressionist oil painting of a Canadian man riding a moose through a forest of maple trees"

Seriously, the only real limits are your imagination, the AI image generator's ability to [comprehend your prompt](https://zapier.com/blog/natural-language-processing/), and any content filters put in place to stop plagiarism, copyright infringement, and bad actors flooding the internet with AI-generated violence or other NSFW content. (That Vermeer prompt used to work reliably, but some more restrictive image generators now block it because it uses a named artist.)

Most AI image generators work in a pretty similar way. [Millions or billions](https://laion.ai/blog/laion-5b/) of image-text pairs are used to train a neural network (basically, a very fancy computer algorithm [modeled loosely on the human brain](https://news.mit.edu/2017/explained-neural-networks-deep-learning-0414)) on _what things are_. By allowing it to process near-countless images, it learns what dogs, the color red, Vermeers, and everything else are. Once this is done, you have an AI that can interpret almost any prompt—though [there is a skill in setting things up](https://zapier.com/blog/ai-art-prompts/) so it can do so accurately.

The next step is to actually render the AI-generated image. The latest generation of AI image generators typically uses a [process called diffusion](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)—though OpenAI's latest foray into image generation uses a slightly different [process called autoregression](https://arxiv.org/abs/2404.02905). In essence, the image generators start with a random field of noise and then edit it in a series of steps to match their interpretation of the prompt. It's kind of like looking up at a cloudy sky, finding a cloud that looks kind of like a dog, and then being able to snap your fingers to keep making it more and more dog-like.

![A series of images generated from AI: dog-shaped cloud floating in a clear blue sky](https://images.ctfassets.net/lzny33ho1g45/1LHdvgxMxOKcgWqC2yzoKh/ff7194426828d81a2d8437f4f9c38132/ai-image-generator-dogs.png?)

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

![GPT-4o, our pick for the best AI image generator for incorporating AI images into your existing workflows](https://images.ctfassets.net/lzny33ho1g45/75DSS8gsgXORvalbs3MCyE/e5c337007c370f28ba0e27584234c762/image13.jpg?)

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

In addition to GPT-4o image generation through ChatGPT, [OpenAI offers an API](https://zapier.com/blog/openai-api/), which means you can [connect ChatGPT to Zapier](https://zapier.com/apps/chatgpt/integrations) to do things like automatically create images from Google Forms or HubSpot responses—or any other apps you use.

**GPT-4o pricing:** Free users can access it, but if you don't want to run into limits, GPT-4o image generation is included as part of ChatGPT Plus at $20/month.

## The best AI image generator for artistic results

### [**Midjourney**](https://www.midjourney.com/explore?tab=top)

![Midjourney, our pick for the AI image generator with the most artistic results](https://images.ctfassets.net/lzny33ho1g45/5c2lxK4vhLWzfata4t1eul/5037e39582914b8b1f4be36d945085e3/image12.jpg?)

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

## The best AI image generator for adhering to prompts

### [**Reve**](https://preview.reve.art/)

![Reve, our pick for the best AI image generator for adhering to prompts](https://images.ctfassets.net/lzny33ho1g45/1rErUICKuzBtIoT0x1EmHf/4e9492bc64da35bec554e2eb16f4ca02/image7.jpg?)

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

![Ideogram, our pick for the best AI image generator for accurate text](https://images.ctfassets.net/lzny33ho1g45/7xaiByWYInfO3qQnxkpn9O/05f734289aa1b0f517b3f43eb74f9680/image15-ideogram.jpg?)

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

![DreamStudio by Stable Diffusion, our pick for the best AI image generator for customization and control](https://images.ctfassets.net/lzny33ho1g45/4Az7EJ5gtpVyQYpyk3J7AX/5077102edb0e3f841c0d3160aeae1bd0/image3.jpeg?)

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

## Best Stable Diffusion alternative

### [**FLUX.1**](https://blackforestlabs.ai/)

![FLUX.1, our pick for the best Stable Diffusion alternative](https://images.ctfassets.net/lzny33ho1g45/5xAzjYy11xVmiruodsWtSo/d7a171fd60b1ffd639829b40a87e8cc7/image5.jpg?)

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

![Adobe Firefly, our pick for the best AI image generator for integrating AI-generated images into photos](https://images.ctfassets.net/lzny33ho1g45/4awQwjmU6tXZ9zR8TvLFCm/3835c17c8b44c9a9396d57e77701fdd5/best-ai-image-generator-image2.jpeg?)

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

![Recraft, our pick for the best AI image generator for graphic design](https://images.ctfassets.net/lzny33ho1g45/3ZU5phnoABT9vgevnLFlcG/6eb9b110464e7ed161fe54816ef0f78c/image2.png?)

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

### Original URL
https://zapier.com/blog/best-ai-image-generator/
</details>

