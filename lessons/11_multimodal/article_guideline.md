## Global Context of the Lesson

- **What I’m planning to share**: A hands-on lesson on presenting the fundamentals of working with multimodal data in the context of LLMs, RAG and context engineering, all fundamental components required to build industry-level AI agents or LLM workflows. More concretely we want to show to use work with multimodal LLMs and embeddings models used for RAG. We want to blend theory with practicality. Thus, 20% of the article should explain the principles of multimodal LLMs and embeddings models, while in the rest we plan to show how to implement them with various use cases such as working with images, audio and PDFs. As an interesting use case we will show how to implement a simple ColPali architecture. Also, in the context of processing PDF documents we want to explain the power of using multimodal techniques such as ColPali vs. older OCR-based techniques.
- **Why I think it’s valuable:** In the real-world we rarely work only with text data. Often we have to manipulate multimodal data such as text, images and PDFs within the same context window or with different tools passed to the LLM, where the most common ones are retrieval tools that based on specific queries can return any type of data. Thus, knowing how to manipulate multimodal data and integrate them with LLMs, RAG, and agents it's a foundational skill when working in the industry.
- **Who the intended audience is:** Asipiring AI Engineers who are learning for the first time about multimodal LLMs, RAG and agents.
- **Theory / Practice ratio:** 20% theory / 80% practice
- **Expected length of the article in words** (where 200-250 words ~= 1 minute of reading time): 3300 words


## Narrative Flow of the Lesson

Follow the next narrative flow when writing the end-to-end lesson:

- What problem are we solving? Why is it essential to solve it?
- Why other solutions are not working and what's wrong with them.
- At a theoretical level, explain our solution or transformation. Highlight:
    - The theoretical foundations.
    - Why is it better than other solutions?
    - What tools or algorithms can we use?
- Provide some hands-on examples.
- Go deeper into the advanced theory.
- Provide a more complex example supporting the advanced theory.
- Connect our solution to the bigger picture and next steps.


## Lesson Outline 

1. Why we need multimodal 
2. Understanding the limitations of traditional approaches
3. Explaining the foundations of multimodal LLMs
4. Working with images and PDFs
5. Explaining the foundations of multimodal embedding models
6. Working with multimodal embeddings models
7. Understading ColPali: The king of multimodal RAG
8. Implementing multimodal RAG
9. Building multimodal AI agents


## Section 1: Why we need multimodal 
- Why agents need to understand images, documents, and complex layouts
- Industry real-world scenarios and limitations of text-only approaches:
    - Financial reports with embedded charts
    - Medical documents with diagnostics
    - Technical documents with diagrams
    - Analyzing images for customer service
    - Research assistant processing charts and diagrams
    - Object detection and classification
-  **Section length:** 200 words

## Section 2: Understanding the limitations of traditional approaches
- Quick overview of traditional document processing workflows done with OCR + Layout detection
- Text extraction challenges with complex layouts
- When traditional methods break down:
    - Advanced OCR engines struggle with handwritten text, poor scans, stylized fonts or more complex layouts such as nested table extraction or complex multi-column layout. Also add some report numbers on this.
    - The standard approach of OCR → layout detection → chunking → embedding → search works for simple documents but becomes "clunky, brittle, and doesn't scale well across real-world data"
    - The multi-step nature of traditional document processing creates a cascade effect where errors compound at each stage. 
- Traditional document processing vs. Interpreting the whole image directly
-  **Section length:** 200 words

## Section 3: Explaining the foundations of multimodal LLMs
- Core concepts and architecture:
    - Explain the foundations and how they work using a text-image multimodal LLM as a concrete example.
    - Focus on the core architecture (how the vision encoder, project layer and text encoder work) and training process
    - Focus only on the most successful approaches based on "late interaction" mechanisms.
    - Enumerate popular multimodal LLM architectures and models, such as LLaVa
    - Capabilities and limitations
- Use an image from teh research showing the architecture of a text-image multimodal LLM
- A note on how it can be expended to other modalities, such as PDF documents, audio or video:
    - Popular models
    - Capabilities and limitations
-  **Section length:** 400 words

## Section 4: Working with images and PDFs
- Short coding example that supports `section 3` which is purely theoretical. In this section we show how to use Gemini to work with images and PDFs.
- There are three ways to process images, as URLs, bytes and BASE64. Bytes is not preffered because the data can easily get corrupted, thus we will focus only on BASE64, which is the bytes representation as strings.
    - Explain why bytes can get corrupted and strings not.
- First show to use them as URLs, explaining that we can leverage this method for public files or in enterprise use cases for private files when storing them in company data lakes, such as AWS S3. PROs and CONs.
- Secondly, show how to use them leveraging the BASE64 format, where we load them from disk, encode them in BASE64 and pass them to the LLM. PROs and CONs.
- Knowing how to work with both URLs and BASE64 is important because of different deployment scenarios:
    - URL-based approaches excel in cloud-native environments where content is already distributed and accessible
    - BASE64 encoding is easier to implement as we can store images directly in the database and also preferred for on-premises deployments or scenarios requiring strict data locality
- Give step-by-step examples from section `1. _`, of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
-  **Section length:** 300 words

## Section 5: Explaining the foundations of multimodal embedding models
- Core concepts and architecture:
    - Explain the foundations using a text-image multimodal embedding model, such as CLIP, as an example.
    - Focus on the architecture, training process
    - Enumerate popular multimodal embedding models architectures
    - Capabilities and limitations
- Use an image from teh research showing the architecture of a text-image embedding model looks like
- A quick note on how this relates to RAG:
    - How it can be used to query information: text query -> return images / text or image query -> return images / text
    - Generate a mermaid diagram showing how this related to RAG
- Expand the idea to other modalities such as video, audio and PDF documents. 
    - Popular models
    - Capabilities and limitations
    - Example: Extract short video sequences from a video, based on a query, leveraging similarities between the video's transcript, audio, and video footage.
-  **Section length:** 400 words

## Section 6: Working with multimodal embeddings models
- Short code example that supports `section 5` which is purely theoretical, where we show how to use Gemini Embeddings to work with text and images, embedding them into the same vector space.
- Build a super basic RAG example using LangChain with a InMemoryVectorStore and GoogleGenerativeAIEmbeddings.
- Give step-by-step examples from section `1. _`, of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
-  **Section length:** 400 words

## Section 7: Understading ColPali: The king of multimodal RAG
- ColPali architecture and innovations:
    - The problem it solves
    - The innovation lies in bypassing the entire OCR pipeline that typically involves text extraction, layout detection, chunking, and embedding. Instead, ColPali processes document images directly using vision-language models to understand both textual and visual content simultaneously.
    - Why layout matters for document comprehension. ColPali works great for documents with tables, figures, and other complex visual layouts.
    - Architecture: Used models, offline indexing and online query logic
    - Chunking text vs. Patching images
    - Outputing "bag-of-embeddings"
    - Using ColPali as a rerenker
- Use an image from the research showing the architecture of ColPali
- Paradigm shift: Standard retrieval (OCR, layout detection, chunking, indexing using a text embedding model)  vs. ColPali (patching and indexing documents as images using a multimodal multi-vector embedding model, where for each image it outputs a "bag-of-embedding" representation):
    - Performance Advantages Over Traditional Methods
    - Scalability of ColPali as a reranker
- Real-world examples scenarios, mostly correlated to RAG, where we have to interpret and retrieval complex documents:
    - Financial document analysis with charts, tables, and spatial relationships
    - Technical documentation with diagrams, flowcharts, and complex visuals
    - Research files with images, videos and diagrams
-  **Section length:** 400 words

## Section 8: Implementing multimodal RAG
- A more complex coding example, where we combine what we learnt into a multimodal RAG exercise.
- Use LangChain with a InMemoryVectorStore and GoogleGenerativeAIEmbeddings to embed multiple PDF files as images and then query them using a text-based query simulating the ColPali implementation
- Note on the fact that this is not 100% ColPali, as we don't patch the image before embedding or use the ColBert reranker. As running `colpali` requires a GPU we wanted to keep the example light leveraging Gemini and focus on how multimodal with PDFs works. Specify that the official `colpali` implementation can be found on GitHub at `illuin-tech/colpali`
- Give step-by-step examples from section `1. _`, of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
-  **Section length:** 400 words

## Section 9: Building multimodal AI agents
- Explain how multimodal techniques can be added to AI Agents by:
    - Adding multimodal inputs or outputs to the reasoning LLM behind the agent.
    - Leveraging multimodal retrieval tools, such as in the ColPali example, which can be adapted to other modalities.
    - Levaraging other multimodal tools such as deep research or MCP servers that return or act external resources: company PDF files, screenshots from your computer, audio files from Spotify or videos from Zoom.
- Create a simple ReAct Engine leveraging LangGraph's `create_react_agent()` and hooking the vector store created at `section 8` as a tool to the agent, where we ask the agent to return a PDF report based on a given query
- Generate a mermaid diagram of our ReAct agent that we will implement
- Give step-by-step examples from section `1. _`, of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
-  **Section length:** 400 words

## Article Code

Links to code that will be used to support the article. Always prioritize this code over every other piece of code found in the sources: 

1. [Notebook 1](...)

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
