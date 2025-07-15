## Global Context of the Lesson

- **What I'm planning to share**: A hands-on lesson presenting the fundamentals of working with multimodal data in the context of LLMs, RAG, and context engineering—all fundamental components required to build industry-level AI agents or LLM workflows. Specifically, we will demonstrate how to work with multimodal LLMs and embedding models used for RAG. We want to blend theory with practicality. Thus, 20% of the article should explain the principles of multimodal LLMs and embedding models, while the rest will focus on practical implementation with use cases such as working with images, audio, and PDFs. As an interesting use case, we will show how to implement a simple ColPali architecture. Also, in the context of processing PDF documents, we want to explain the advantages of using multimodal techniques such as ColPali compared to older OCR-based techniques.
- **Why I think it's valuable**: In the real world, we rarely work only with text data. Often, we have to manipulate multimodal data such as text, images, and PDFs within the same context window or integrate them using various tools that interface with the LLM. The most common tools are retrieval systems that can return different types of data based on specific queries. Thus, knowing how to manipulate multimodal data and integrate it with LLMs, RAG, and agents is a foundational skill in the industry.
- **Who the intended audience is**: Aspiring AI Engineers who are learning for the first time about multimodal LLMs, RAG, and agents.
- **Theory / Practice ratio**: 20% theory / 80% practice
- **Expected length of the article in words** (where 200-250 words ≈ 1 minute of reading time): 3300 words


## Narrative Flow of the Lesson

Follow the next narrative flow when writing the end-to-end lesson:

- What problem are we solving? Why is it essential to solve it?
- Why other solutions do not work and what is wrong with them.
- At a theoretical level, explain our solution or transformation. Highlight:
    - The theoretical foundations.
    - Why is it better than other solutions?
    - What tools or algorithms can we use?
- Provide some hands-on examples.
- Go deeper into the advanced theory.
- Provide a more complex example supporting the advanced theory.
- Connect our solution to the bigger picture and next steps.


## Lesson Outline 

1. The need for multimodal AI
2. Limitations of traditional document processing
3. Foundations of multimodal LLMs
4. Applying multimodal LLMs to images, PDFs, and text
5. Foundations of multimodal embedding models
6. Foundations of multimodal RAG (ColPali architecture)
7. Implementing multimodal RAG for images and text
8. Building multimodal AI agents


## Section 1: The need for multimodal AI
- Why agents need to understand images, documents, and complex layouts
- Industry real-world scenarios and limitations of text-only approaches:
    - Financial reports with embedded charts
    - Medical documents with diagnostics
    - Technical documents with diagrams
    - Analyzing images for customer service
    - Research assistants processing charts and diagrams
    - Object detection and classification
-  **Section length:** 250 words

## Section 2: Limitations of traditional document processing
- Quick overview of traditional document processing workflows done with OCR + Layout detection
- Text extraction challenges with complex layouts
- When traditional methods break down:
    - Advanced OCR engines struggle with handwritten text, poor scans, stylized fonts, or more complex layouts such as nested table extraction or complex multi-column layouts. *[Note: Add some report numbers on this.]*
    - The standard approach of OCR → layout detection → chunking → embedding → search works for simple documents but becomes clunky, brittle, and does not scale well across real-world data.
    - The multi-step nature of traditional document processing creates a cascade effect where errors compound at each stage. 
- Comparison between traditional document processing and interpreting the whole image directly
-  **Section length**: 250 words

## Section 3: Foundations of multimodal LLMs
- Core concepts and architecture:
    - Explain the foundations and how they work using a text-image multimodal LLM as a concrete example.
    - Focus on the core architecture (how the vision encoder, projection layer, and text encoder work) and the training process
    - Focus only on the most successful approaches, which are based on "late interaction" mechanisms.
    - Enumerate popular multimodal LLM architectures and models, such as LLaVA
    - Capabilities and limitations
- Use an image from the research showing the architecture of a text-image multimodal LLM
- Discussion of how it can be expanded to other modalities, such as PDF documents, audio, or video:
    - Popular models
    - Capabilities and limitations
- Include a paragraph on the most popular image generation models, such as Midjourney or Stable Diffusion:
    - Comparison between diffusion image generation models and multimodal LLMs that support generating images (e.g., GPT-4o)
    - Explain that diffusion models are a different family of models than LLMs, which we will not cover in depth in this lesson.
    - In the context of agents, these models can easily be integrated as tools.
-  **Section length**: 500 words

## Section 4: Applying multimodal LLMs to images, PDFs, and text
- Short coding example that supports Section 3, which is purely theoretical. In this section, we show how to use Gemini to work with images and PDFs.
- There are three core ways to process images with LLMs: as raw bytes, Base64, and URLs:
    1. Raw bytes are the easiest way and work well for one-off API calls. However, when storing the image, it can easily get corrupted. (Explain why.)
    2. Base64 is a way to encode raw bytes as strings, allowing images to be stored in your database without getting corrupted. Therefore, we often use this format when storing images directly in a database.
    3. Another popular approach is to use direct URLs, which is useful in two core scenarios: public images from the internet or images stored in a company's data lake, such as AWS S3 or GCP GCS. In most enterprise scenarios, the data will be stored in some sort of bucket. Using this method, the LLM can directly download the media from the bucket instead of passing it around the network, making this the most efficient option since I/O is usually the most common bottleneck of an application.
- Knowing how to work with all three methods is important due to different deployment scenarios. Here are the advantages of each method:
    - URL-based approaches excel in cloud-native environments where content is already distributed and accessible
    - Base64 encoding is easier to implement, as we can store images directly in the database. It is also preferred for on-premises deployments or scenarios requiring strict data locality.
- When working with Gemini with images, show the following scenarios:
    - bytes / one image
    - bytes / two images
    - Base64 / one image
    - URLs / *Note: At the time of writing this lesson, Gemini works well only with GCS images, not public URLs. Thus, for simplicity, we will provide only pseudocode.*
    - bytes / one image / object detection
- When working with Gemini with PDFs, show the following scenarios:
    - bytes / one pdf
    - Base64 / one pdf
- Give step-by-step examples from `Section 2 ("Applying multimodal LLMs to images, PDFs, and text")` of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
-  **Section length**: 500 words (Don't count the code)

## Section 5: Foundations of multimodal embedding models
- Core concepts and architecture:
    - Explain the foundations using a text-image multimodal embedding model, such as CLIP, as a concrete example.
    - Focus on the architecture and training process
    - Enumerate popular multimodal embedding model architectures
    - Capabilities and limitations
- Use an image from the research showing what the architecture of a text-image embedding model looks like
- Explanation of how this relates to RAG:
    - How it can be used to query information: text query → return images/text, or image query → return images/text.
    - *[Generate a mermaid diagram showing how this relates to RAG]*
- Extension of the concept to other modalities, such as video, audio, and PDF documents. 
    - Popular models
    - Capabilities and limitations
    - Example: Extract short video sequences from a video based on a query, leveraging similarities between the video's transcript, audio, and video footage.
-  **Section length**: 400 words

## Section 6: Foundations of multimodal RAG (ColPali architecture)
- Brief introduction to ColPali, explaining that this is the modern architecture for multimodal RAG, which is the most common use of multimodal embeddings when working with AI agents
- ColPali architecture and innovations:
    - The problem it solves
    - The innovation lies in bypassing the entire OCR pipeline that typically involves text extraction, layout detection, chunking, and embedding. Instead, ColPali processes document images directly using vision-language models to understand both textual and visual content simultaneously.
    - Why layout matters for document comprehension. ColPali works great for documents with tables, figures, and other complex visual layouts.
    - Architecture: Used models, offline indexing and online query logic
    - Chunking text vs. Patching images
    - Outputting "bag-of-embeddings"
    - Using ColPali as a reranking system
- Use an image from the research showing the architecture of ColPali
- Paradigm shift comparison:
    - Standard retrieval: OCR, layout detection, chunking, indexing using a text embedding model
    - ColPali: patching and indexing documents as images using a multimodal multi-vector embedding model, where each image outputs a "bag-of-embedding" representation
    - Performance advantages over traditional methods
    - Scalability of ColPali as a reranker
- Real-world example scenarios, mostly related to RAG, where we have to interpret and retrieve complex documents:
    - Financial document analysis with charts, tables, and spatial relationships
    - Technical documentation with diagrams, flowcharts, and complex visuals
    - Research files with images, videos, and diagrams
-  **Section length**: 500 words

## Section 7: Implementing multimodal RAG for images and text
- A more complex coding example where we combine what we have learned into a multimodal RAG exercise.
- Present a simple multimodal RAG example where we populate an in-memory vector database with images and query it with text questions. To keep it simple, we will use only the Gemini client to access an embedding model and keep the vector index as a simple list.
- Since the Gemini Dev API doesn't support image embeddings, we will create a description of each image using Gemini and embed that using the embedding model. Specify that to do it properly, as long as you have a multimodal embedding model, you just have to embed the image instead of text. Everything else remains the same.
- Provide a list of popular multimodal embedding models: Voyage, Cohere, Google Embeddings on Vertex AI (not Gemini Dev), OpenAI CLIP.
- Also highlight that this is not a complete ColPali implementation, as we do not patch the image before embedding or use the ColBERT reranker. Since running `colpali` requires a GPU, we wanted to keep the example lightweight by leveraging Gemini and focusing on how multimodal works in general.
- Specify that the official `colpali` implementation can be found on GitHub at `illuin-tech/colpali`.
- Give step-by-step examples from `Section 3 ("Implementing multimodal RAG for images and text")` of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
-  **Section length**: 400 words (Don't count the code)

## Section 8: Building multimodal AI agents
- Explain how multimodal techniques can be added to AI Agents by:
    - Adding multimodal inputs or outputs to the reasoning LLM behind the agent.
    - Leveraging multimodal retrieval tools, such as in the RAG example, which can be adapted to other modalities.
    - Leveraging other multimodal tools such as deep research or MCP servers that return or act on external resources: company PDF files, screenshots from your computer, audio files from Spotify, or videos from Zoom.
- Create a simple ReAct Engine leveraging LangGraph's `create_react_agent()` and connecting the RAG retrieval function from Section 7 as a tool for the agent, where we ask the agent to find details about an object from our collection of images.
- *[Generate a mermaid diagram of our ReAct agent that we will implement]*
- Give step-by-step examples from `Section 4 ("Building multimodal AI agents")` of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
-  **Section length**: 400 words (Don't count the code)

## Article Code

Links to code that will be used to support the article. Always prioritize this code over any other code found in the sources: 

1. [Notebook 1](https://github.com/towardsai/course-ai-agents/blob/main/lessons/11_multimodal/notebook.ipynb)

## Golden Sources

1. [Understanding Multimodal LLMs](https://magazine.sebastianraschka.com/)
2. [Vision Language Models](https://www.nvidia.com/en-us/glossary/vision-language-models/)
3. [Multimodal Embeddings: An Introduction](https://towardsdatascience.com/multimodal-embeddings-an-introduction-5dc36975966f/)
3. [Multimodal Embeddings: An Introduction](https://www.youtube.com/watch?v=YOvxh_ma5qE)
4. [Multi-modal ML with OpenAI's CLIP](https://www.pinecone.io/learn/series/image-search/clip/)
5. [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/html/2407.01449v6)

## Other Sources

1. [Image understanding with Gemini](https://ai.google.dev/gemini-api/docs/image-understanding)
2. [Multimodal RAG with Colpali, Milvus and VLMs](https://huggingface.co/blog/saumitras/colpali-milvus-multimodal-rag)
3. [Google Generative AI Embeddings (AI Studio & Gemini API)](https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai/)
4. [LangGraph quickstart](https://langchain-ai.github.io/langgraph/agents/agents/)
5. [Complex Document Recognition: OCR Doesn’t Work and Here’s How You Fix It](https://hackernoon.com/complex-document-recognition-ocr-doesnt-work-and-heres-how-you-fix-it)
6. [What are some real-world applications of multimodal AI?](https://milvus.io/ai-quick-reference/what-are-some-realworld-applications-of-multimodal-ai)
7. [What Is Optical Character Recognition (OCR)?](https://blog.roboflow.com/what-is-optical-character-recognition-ocr/)
8. [The 8 best AI image generators in 2025](https://zapier.com/blog/best-ai-image-generator/)