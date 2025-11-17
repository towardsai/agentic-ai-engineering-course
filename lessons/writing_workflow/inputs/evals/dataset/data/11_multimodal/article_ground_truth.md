# Lesson 11: Multimodal Data

In the previous lessons, we built a solid foundation for creating AI apps. You learned the difference between structured LLM workflows and autonomous AI agents, mastered context engineering, and understood core patterns like ReAct and Retrieval-Augmented Generation (RAG). We have covered the essentials of building systems that can reason and access external knowledge. In this lesson, we will focus on the final piece of the puzzle for the first part of the course: multimodal data. 

Multimodal data is essential because in the real world, enterprise-grade AI applications rarely deal with text alone. They must understand and process images, documents, charts, and tables. Text-only systems are limited to tasks like analyzing complex financial reports, medical documents, and building sketches. To build truly useful AI, you need to interpret this rich, visual information. This lesson covers the ‘how’.

## Limitations of Traditional Document Processing

For years, the standard approach to handling complex documents like invoices, reports, or technical manuals in AI systems was to normalize everything to text. This process typically relies on Optical Character Recognition (OCR) and involves a convoluted, multi-step pipeline. Suppose you need to process a PDF that contains a mix of text, tables, and diagrams.

As illustrated in Figure 1, such a system first starts with preprocessing, such as noise removal. Then, it needs to perform layout detection to identify different regions within the document, such as titles, paragraphs, tables, or diagrams. It then sends the text regions to an OCR model for extraction. For other structures, it needs specialized models: one for tables, another for diagrams, and so on. The final output is a structured format, like JSON, that pieces together the extracted text and metadata.

![*Figure 1: A traditional, multi-step document processing pipeline.*](Lesson%2011%20Multimodal%20Data%20250f9b6f42708088b0e5f3ccb1e43223/image.png)

*Figure 1: A traditional, multi-step document processing pipeline.*

This entire process is fundamentally flawed. It is rigid because it fails when encountering a new data type it was not trained on. Unseen structures (e.g., a new chart type) bypass your detectors and vanish from output. It is fragile, slow, and costly, requiring multiple model calls for a single document and errors compound at each stage. For example, a mistake in layout detection cascades into the OCR step, leading to poorly extracted text that makes downstream RAG systems useless [[1]](https://www.mixedbread.com/blog/the-hidden-ceiling). Traditional OCR systems struggle with handwritten notes, poor-quality scans, and complex layouts like architectural drawings, where performance can drop significantly [[2]](https://arxiv.org/html/2412.02210v2), [[3]](https://www.dataunboxed.io/blog/ocr-vs-vlm-ocr-naive-benchmarking-accuracy-for-scanned-documents). Benchmarks show that even state-of-the-art OCR solutions underperform compared to perfect text extraction, creating a "performance ceiling" that limits retrieval and question-answering accuracy, especially in documents with dense tables, formulas, or embedded diagrams [[1]](https://www.mixedbread.com/blog/the-hidden-ceiling), [[5]](https://www.3rdaiautomation.com/blog/benchmarking-QA-on-complex-indestrial-PDFs).

![*Figure 2: Complex layouts like this floor plan are a significant challenge for traditional OCR systems. (Media from [Hackernoon](https://hackernoon.com/complex-document-recognition-ocr-doesnt-work-and-heres-how-you-fix-it))*](Lesson%2011%20Multimodal%20Data%20250f9b6f42708088b0e5f3ccb1e43223/669eda55-7a5b-4beb-bc0b-b769e5a6d708.png)

*Figure 2: Complex layouts like this floor plan are a significant challenge for traditional OCR systems. (Media from [Hackernoon](https://hackernoon.com/complex-document-recognition-ocr-doesnt-work-and-heres-how-you-fix-it))*

This approach might work for niche, highly specialized tasks, but it does not scale for the flexible and fast AI agents we need today. Modern AI solutions use multimodal LLMs that can directly interpret images and PDFs as native inputs, completely bypassing this brittle OCR workflow. Let us explore how multimodal LLMs work.

## **Foundations of Multimodal LLMs**

Before diving into the code, it's helpful to have a high-level understanding of how multimodal LLMs work. To build AI agents, we don't need to understand every detail, but understanding the basics helps you implement, deploy, and optimize them effectively.

Multimodal LLMs typically use one of two designs: a unified embedding‑decoder that concatenates projected image patch embeddings with text embeddings into an unmodified LLM decoder, or a cross‑modality attention approach where text queries connect to visual features via the cross‑attention module.

![A diagram comparing the Unified Embedding Decoder Architecture and the Cross-modality Attention Architecture for multimodal LLMs.](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F53956ae8-9cd8-474e-8c10-ef6bddb88164_1600x938.png)

*Figure 3: The two primary architectural approaches for building multimodal LLMs - Source* [Understanding Multimodal LLMs](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms) [6]

Let us look closer at the **unified embedding decoder** approach, which is simpler and more common. In this setup, the system passes image information to the LLM as a sequence of input tokens concatenated to the text tokens. The main components are an image encoder, a projector, and the LLM backbone itself [[7]](https://www.nvidia.com/en-us/glossary/multimodal-large-language-models/). In this design, the image is converted into patch embeddings by an image encoder (often ViTs) and then linearly projected to the LLM’s embedding dimension so the image tokens can be concatenated with text tokens and processed by a unmodified decoder [[6]](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms). Many recent systems train this stack in stages—first fitting the projector (with encoders frozen) before unfreezing components for end-to-end fine-tuning for stability [[6]](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms).

![A diagram illustrating the Unified Embedding Decoder Architecture, where image token embeddings and text token embeddings are concatenated and fed into an LLM decoder.](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa219f185-211b-4569-9398-2e080e2c5619_1166x1400.png)

*Figure 4: The Unified Embedding Decoder Architecture concatenates image and text embeddings as input to the LLM - Source* [Understanding Multimodal LLMs](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms) [6]

The **cross-modality attention** architecture takes a different route. Instead of adding image tokens to the input sequence, it injects the visual information directly into the attention mechanism of the LLM during the decoding step. This allows the model to dynamically "look" at the image features while processing the text [[8]](https://blog.premai.io/multimodal-llms-architecture-techniques-and-use-cases/). Concretely, the text stream queries visual features via cross-attention, attending to image keys/values when needed rather than carrying all image tokens through the sequence. This pattern can be more efficient for high-resolution inputs and long contexts while preserving tight text–vision coupling [[6]](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms).

![A diagram showing the Cross-modality Attention Architecture, where an image encoder's output is fed into the cross-attention layers of an LLM.](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd9c06055-b959-45d1-87b2-1f4e90ceaf2d_1296x1338.png)

*Figure 5: The Cross-modality Attention Architecture integrates visual data via attention mechanisms - Source* [Understanding Multimodal LLMs](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms) [6]

Both approaches rely on an **image encoder**. The image encoder takes as input the image and splits it into multiple fixed-size patches. Similar to how we split text into multiple tokens, this process is known as image tokenization.

![A side-by-side comparison of image tokenization (patching)
and text tokenization.](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0d56ea06-d202-4eb7-9e01-9aac492ee309_1522x1206.png)

*Figure 6: Image patching is analogous to text tokenization, converting visual data into a sequence of embeddings - Source* [Understanding Multimodal LLMs](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms) [6]

Then the image encoder (e.g., a ViT) processes these patches with transformer layers to produce semantic image embeddings. Ultimately, a lightweight linear projection maps these vectors to the LLM’s embedding size so they are compatible with the language model’s inputs [[6]](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms). During projection, we ensure that the image and text embeddings are within the same vector space for the LLM to interpret them together.

![A diagram of a Vision Transformer (ViT)
architecture, showing how an image is split into patches and processed to produce embeddings.](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffef5f8cb-c76c-4c97-9771-7fdb87d7d8cd_1600x1135.png)

*Figure 7: A Vision Transformer converts image patches into embeddings - Source* [Understanding Multimodal LLMs](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms) [6]

The real value emerges when the image embeddings and text embeddings align in the same vector space. Models can achieve this alignment through a technique called contrastive learning, which trains them to place semantically similar images and text descriptions close to each other in the embedding space [[10]](https://github.com/openai/CLIP). This shared space enables semantic similarity searches across different data types; you can use a text query to find a relevant image because their vector representations are close to one another [[11]](https://huggingface.co/docs/transformers/v4.22.1/en/model_doc/clip). Contrastive learning achieves this by training the model to maximize the similarity between "positive pairs" (like an image and its correct caption) and minimize the similarity between "negative pairs" (an image and an irrelevant caption) [[12]](https://towardsdatascience.com/multimodal-embeddings-an-introduction-5dc36975966f), [[13]](https://www.pinecone.io/learn/series/image-search/clip/). This ensures that semantically related content, regardless of its original modality, is positioned closely in the shared vector space [[12]](https://towardsdatascience.com/multimodal-embeddings-an-introduction-5dc36975966f). Popular choices for these image encoders include CLIP (Contrastive Language-Image Pre-Training), OpenCLIP, or SigLIP [[10]](https://github.com/openai/CLIP), [[11]](https://huggingface.co/docs/transformers/v4.22.1/en/model_doc/clip).

![Multimodal embeddings align text and image representations in a shared vector space.](https://towardsdatascience.com/wp-content/uploads/2024/11/15d3HBNjNIXLy0oMIvJjxWw.png)

*Figure 8: Multimodal embeddings align text and image representations in a shared vector space - Source* [Understanding Multimodal LLMs](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms) [6]

Each architectural approach has its trade-offs. The unified embedding model is generally simpler to implement and often performs better on OCR-related tasks. The cross-attention model can be computationally efficient, especially with high-resolution images, because it avoids lengthening the input sequence with image tokens [[6]](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms). Hybrid approaches also exist, aiming to combine the best of both worlds. The core takeaway is that multiple architectures can succeed, often with hybrid architectures proving the most successful.

Today, most state-of-the-art models are multimodal, but the level of multimodality can be highly varied. In the open-source community, we have models like Llama 4, Gemma 2, Qwen3, and DeepSeek R1/V3. In the proprietary space, models like OpenAI's GPT-5, Google's Gemini 2.5, and Anthropic's Claude series all have strong multimodal capabilities. This modular encoder-based design also allows us to extend it to other data types like audio or video by simply adding the appropriate encoder [[8]](https://blog.premai.io/multimodal-llms-architecture-techniques-and-use-cases/).

It is essential to distinguish between the multimodal LLMs presented in this section and generative diffusion models like Midjourney or Stable Diffusion used to generate high-quality images or videos. While models like Gemini 2.5 can create pictures, their core architecture differs from diffusion models. Therefore, for building AI agents, we focus only on multimodal LLMs for their reasoning capabilities and ability to understand different types of inputs. On the other hand, as diffusion models are used only to generate multimodal data, they are not the focus of this course. Still, if necessary, they could easily be integrated as tools, extending the agent's generation capabilities [[14]](https://blog.google/technology/ai/google-gemini-ai/).

The field of multimodal AI is evolving rapidly. The goal of this section was not to be exhaustive but to give you a solid intuition for why these models are superior to older OCR-based methods. Now that you understand how LLMs can directly process images or other data modalities, let us see how it works in practice.

## Applying Multimodal LLMs to Images and PDFs

To understand how LLMs work with multimodal data, we will walk you through some hands-on examples using the three primary ways to pass multimodal data, such as images and PDFs, to an LLM API: as raw bytes, as Base64-encoded strings, or as URLs. To support our ideas, we will implement multiple use cases using Gemini, including extracting image captions, comparing multiple images, and, as a more interesting use case, performing 2D object detection on images. Ultimately, to grasp the differences between the two, we will repeat the same process, with PDFs instead of images.

Before digging into the implementation, let’s quickly review some pros and cons of the three core options for passing multimodal data to LLMs.

**Raw bytes** offer the most direct way to handle files and work well for simple, one-off API calls. However, storing raw binary data in many databases can be problematic. Databases often interpret binary input as text, which can lead to data corruption.

**Base64 encoding** solves this storage problem by converting binary data into a string format. This is a common practice for embedding images directly in web pages. For our use case, it ensures that images or documents can be safely stored in a text-friendly database like PostgreSQL or MongoDB. The main downside is that Base64 strings are about 33% larger than the original binary data, which can increase latency and storage costs [[15]](https://www.turing.com/resources/building-high-quality-multimodal-data-pipelines-for-llms).

**URLs** are the most efficient method for enterprise-scale applications. Instead of passing large files back and forth over the network, you can store your data in a private data lake like Amazon S3 or Google Cloud Storage. You then simply pass a secure URL to the LLM. The model's server fetches the data directly, reducing client-side bandwidth and improving performance [[16]](https://arxiv.org/html/2506.06579v1). This method is also useful for public data, as some models can directly access and process content from a public URL.

![*Figure 9: Comparison of data flow for Base64 vs. URL-based multimodal data handling.*](Lesson%2011%20Multimodal%20Data%20250f9b6f42708088b0e5f3ccb1e43223/image%201.png)

*Figure 9: Comparison of data flow for Base64 vs. URL-based multimodal data handling.*

[Diagram](data:image/svg+xml;charset=utf-8,%3Csvg%20aria-roledescription%3D%22flowchart-v2%22%20role%3D%22graphics-document%20document%22%20viewBox%3D%22-32%20-28%20461.390625%20437%22%20style%3D%22max-width%3A%20397.390625px%3B%22%20class%3D%22flowchart%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22461.390625%22%20id%3D%22mermaid-e47045a8-d091-4170-b395-992e34910d18%22%20height%3D%22437%22%3E%3Crect%20x%3D%22-32%22%20y%3D%22-28%22%20width%3D%22461.390625%22%20height%3D%22437%22%20fill%3D%22%23191919%22%2F%3E%3Cstyle%3E%23mermaid-e47045a8-d091-4170-b395-992e34910d18%7Bfont-family%3Aui-sans-serif%2C-apple-system%2CBlinkMacSystemFont%2C%22Segoe%20UI%20Variable%20Display%22%2C%22Segoe%20UI%22%2CHelvetica%2C%22Apple%20Color%20Emoji%22%2CArial%2Csans-serif%2C%22Segoe%20UI%20Emoji%22%2C%22Segoe%20UI%20Symbol%22%3Bfont-size%3A16px%3Bfill%3A%23ccc%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.error-icon%7Bfill%3A%23a44141%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.error-text%7Bfill%3A%23ddd%3Bstroke%3A%23ddd%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.edge-thickness-normal%7Bstroke-width%3A1px%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.edge-thickness-thick%7Bstroke-width%3A3.5px%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.edge-pattern-solid%7Bstroke-dasharray%3A0%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.edge-thickness-invisible%7Bstroke-width%3A0%3Bfill%3Anone%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.edge-pattern-dashed%7Bstroke-dasharray%3A3%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.edge-pattern-dotted%7Bstroke-dasharray%3A2%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.marker%7Bfill%3Alightgrey%3Bstroke%3Alightgrey%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.marker.cross%7Bstroke%3Alightgrey%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20svg%7Bfont-family%3Aui-sans-serif%2C-apple-system%2CBlinkMacSystemFont%2C%22Segoe%20UI%20Variable%20Display%22%2C%22Segoe%20UI%22%2CHelvetica%2C%22Apple%20Color%20Emoji%22%2CArial%2Csans-serif%2C%22Segoe%20UI%20Emoji%22%2C%22Segoe%20UI%20Symbol%22%3Bfont-size%3A16px%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20p%7Bmargin%3A0%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.label%7Bfont-family%3Aui-sans-serif%2C-apple-system%2CBlinkMacSystemFont%2C%22Segoe%20UI%20Variable%20Display%22%2C%22Segoe%20UI%22%2CHelvetica%2C%22Apple%20Color%20Emoji%22%2CArial%2Csans-serif%2C%22Segoe%20UI%20Emoji%22%2C%22Segoe%20UI%20Symbol%22%3Bcolor%3A%23ccc%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.cluster-label%20text%7Bfill%3A%23F9FFFE%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.cluster-label%20span%7Bcolor%3A%23F9FFFE%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.cluster-label%20span%20p%7Bbackground-color%3Atransparent%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.label%20text%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20span%7Bfill%3A%23ccc%3Bcolor%3A%23ccc%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.node%20rect%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.node%20circle%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.node%20ellipse%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.node%20polygon%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.node%20path%7Bfill%3A%231f2020%3Bstroke%3A%23ccc%3Bstroke-width%3A1px%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.rough-node%20.label%20text%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.node%20.label%20text%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.image-shape%20.label%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.icon-shape%20.label%7Btext-anchor%3Amiddle%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.node%20.katex%20path%7Bfill%3A%23000%3Bstroke%3A%23000%3Bstroke-width%3A1px%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.rough-node%20.label%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.node%20.label%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.image-shape%20.label%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.icon-shape%20.label%7Btext-align%3Acenter%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.node.clickable%7Bcursor%3Apointer%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.root%20.anchor%20path%7Bfill%3Alightgrey!important%3Bstroke-width%3A0%3Bstroke%3Alightgrey%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.arrowheadPath%7Bfill%3Alightgrey%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.edgePath%20.path%7Bstroke%3Alightgrey%3Bstroke-width%3A2.0px%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.flowchart-link%7Bstroke%3Alightgrey%3Bfill%3Anone%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.edgeLabel%7Bbackground-color%3Ahsl(0%2C%200%25%2C%2034.4117647059%25)%3Btext-align%3Acenter%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.edgeLabel%20p%7Bbackground-color%3Ahsl(0%2C%200%25%2C%2034.4117647059%25)%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.edgeLabel%20rect%7Bopacity%3A0.5%3Bbackground-color%3Ahsl(0%2C%200%25%2C%2034.4117647059%25)%3Bfill%3Ahsl(0%2C%200%25%2C%2034.4117647059%25)%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.labelBkg%7Bbackground-color%3Argba(87.75%2C%2087.75%2C%2087.75%2C%200.5)%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.cluster%20rect%7Bfill%3Ahsl(180%2C%201.5873015873%25%2C%2028.3529411765%25)%3Bstroke%3Argba(255%2C%20255%2C%20255%2C%200.25)%3Bstroke-width%3A1px%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.cluster%20text%7Bfill%3A%23F9FFFE%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.cluster%20span%7Bcolor%3A%23F9FFFE%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20div.mermaidTooltip%7Bposition%3Aabsolute%3Btext-align%3Acenter%3Bmax-width%3A200px%3Bpadding%3A2px%3Bfont-family%3Aui-sans-serif%2C-apple-system%2CBlinkMacSystemFont%2C%22Segoe%20UI%20Variable%20Display%22%2C%22Segoe%20UI%22%2CHelvetica%2C%22Apple%20Color%20Emoji%22%2CArial%2Csans-serif%2C%22Segoe%20UI%20Emoji%22%2C%22Segoe%20UI%20Symbol%22%3Bfont-size%3A12px%3Bbackground%3Ahsl(20%2C%201.5873015873%25%2C%2012.3529411765%25)%3Bborder%3A1px%20solid%20rgba(255%2C%20255%2C%20255%2C%200.25)%3Bborder-radius%3A2px%3Bpointer-events%3Anone%3Bz-index%3A100%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.flowchartTitleText%7Btext-anchor%3Amiddle%3Bfont-size%3A18px%3Bfill%3A%23ccc%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20rect.text%7Bfill%3Anone%3Bstroke-width%3A0%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.icon-shape%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.image-shape%7Bbackground-color%3Ahsl(0%2C%200%25%2C%2034.4117647059%25)%3Btext-align%3Acenter%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.icon-shape%20p%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.image-shape%20p%7Bbackground-color%3Ahsl(0%2C%200%25%2C%2034.4117647059%25)%3Bpadding%3A2px%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.icon-shape%20rect%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.image-shape%20rect%7Bopacity%3A0.5%3Bbackground-color%3Ahsl(0%2C%200%25%2C%2034.4117647059%25)%3Bfill%3Ahsl(0%2C%200%25%2C%2034.4117647059%25)%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20%3Aroot%7B--mermaid-font-family%3A%22trebuchet%20ms%22%2Cverdana%2Carial%2Csans-serif%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.defaultStyle%26gt%3B*%7Bfill%3A%232d2d2d!important%3Bstroke%3A%23e0e0e0!important%3Bstroke-width%3A1px!important%3Bcolor%3A%23ffffff!important%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.defaultStyle%20span%7Bfill%3A%232d2d2d!important%3Bstroke%3A%23e0e0e0!important%3Bstroke-width%3A1px!important%3Bcolor%3A%23ffffff!important%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.defaultStyle%20tspan%7Bfill%3A%23ffffff!important%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.subgraphStyle%26gt%3B*%7Bfill%3A%232d2d2d!important%3Bstroke%3A%2387ceeb!important%3Bstroke-width%3A1px!important%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.subgraphStyle%20span%7Bfill%3A%232d2d2d!important%3Bstroke%3A%2387ceeb!important%3Bstroke-width%3A1px!important%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.nodeStyle%26gt%3B*%7Bfill%3A%232d2d2d!important%3Bstroke%3A%23e0e0e0!important%3Bstroke-width%3A1px!important%3Bcolor%3A%23ffffff!important%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.nodeStyle%20span%7Bfill%3A%232d2d2d!important%3Bstroke%3A%23e0e0e0!important%3Bstroke-width%3A1px!important%3Bcolor%3A%23ffffff!important%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.nodeStyle%20tspan%7Bfill%3A%23ffffff!important%3B%7D%3C%2Fstyle%3E%3Cg%3E%3Cmarker%20orient%3D%22auto%22%20markerHeight%3D%228%22%20markerWidth%3D%228%22%20markerUnits%3D%22userSpaceOnUse%22%20refY%3D%225%22%20refX%3D%225%22%20viewBox%3D%220%200%2010%2010%22%20class%3D%22marker%20flowchart-v2%22%20id%3D%22mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-pointEnd%22%3E%3Cpath%20style%3D%22stroke-width%3A%201%3B%20stroke-dasharray%3A%201%2C%200%3B%22%20class%3D%22arrowMarkerPath%22%20d%3D%22M%200%200%20L%2010%205%20L%200%2010%20z%22%2F%3E%3C%2Fmarker%3E%3Cmarker%20orient%3D%22auto%22%20markerHeight%3D%228%22%20markerWidth%3D%228%22%20markerUnits%3D%22userSpaceOnUse%22%20refY%3D%225%22%20refX%3D%224.5%22%20viewBox%3D%220%200%2010%2010%22%20class%3D%22marker%20flowchart-v2%22%20id%3D%22mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-pointStart%22%3E%3Cpath%20style%3D%22stroke-width%3A%201%3B%20stroke-dasharray%3A%201%2C%200%3B%22%20class%3D%22arrowMarkerPath%22%20d%3D%22M%200%205%20L%2010%2010%20L%2010%200%20z%22%2F%3E%3C%2Fmarker%3E%3Cmarker%20orient%3D%22auto%22%20markerHeight%3D%2211%22%20markerWidth%3D%2211%22%20markerUnits%3D%22userSpaceOnUse%22%20refY%3D%225%22%20refX%3D%2211%22%20viewBox%3D%220%200%2010%2010%22%20class%3D%22marker%20flowchart-v2%22%20id%3D%22mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-circleEnd%22%3E%3Ccircle%20style%3D%22stroke-width%3A%201%3B%20stroke-dasharray%3A%201%2C%200%3B%22%20class%3D%22arrowMarkerPath%22%20r%3D%225%22%20cy%3D%225%22%20cx%3D%225%22%2F%3E%3C%2Fmarker%3E%3Cmarker%20orient%3D%22auto%22%20markerHeight%3D%2211%22%20markerWidth%3D%2211%22%20markerUnits%3D%22userSpaceOnUse%22%20refY%3D%225%22%20refX%3D%22-1%22%20viewBox%3D%220%200%2010%2010%22%20class%3D%22marker%20flowchart-v2%22%20id%3D%22mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-circleStart%22%3E%3Ccircle%20style%3D%22stroke-width%3A%201%3B%20stroke-dasharray%3A%201%2C%200%3B%22%20class%3D%22arrowMarkerPath%22%20r%3D%225%22%20cy%3D%225%22%20cx%3D%225%22%2F%3E%3C%2Fmarker%3E%3Cmarker%20orient%3D%22auto%22%20markerHeight%3D%2211%22%20markerWidth%3D%2211%22%20markerUnits%3D%22userSpaceOnUse%22%20refY%3D%225.2%22%20refX%3D%2212%22%20viewBox%3D%220%200%2011%2011%22%20class%3D%22marker%20cross%20flowchart-v2%22%20id%3D%22mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-crossEnd%22%3E%3Cpath%20style%3D%22stroke-width%3A%202%3B%20stroke-dasharray%3A%201%2C%200%3B%22%20class%3D%22arrowMarkerPath%22%20d%3D%22M%201%2C1%20l%209%2C9%20M%2010%2C1%20l%20-9%2C9%22%2F%3E%3C%2Fmarker%3E%3Cmarker%20orient%3D%22auto%22%20markerHeight%3D%2211%22%20markerWidth%3D%2211%22%20markerUnits%3D%22userSpaceOnUse%22%20refY%3D%225.2%22%20refX%3D%22-1%22%20viewBox%3D%220%200%2011%2011%22%20class%3D%22marker%20cross%20flowchart-v2%22%20id%3D%22mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-crossStart%22%3E%3Cpath%20style%3D%22stroke-width%3A%202%3B%20stroke-dasharray%3A%201%2C%200%3B%22%20class%3D%22arrowMarkerPath%22%20d%3D%22M%201%2C1%20l%209%2C9%20M%2010%2C1%20l%20-9%2C9%22%2F%3E%3C%2Fmarker%3E%3C%2Fg%3E%3Cg%20class%3D%22subgraphs%22%3E%3Cg%20class%3D%22subgraph%22%3E%3Cg%20data-look%3D%22classic%22%20id%3D%22subGraph1%22%20class%3D%22cluster%22%3E%3Crect%20height%3D%22357%22%20width%3D%22166.34375%22%20y%3D%2212%22%20x%3D%228%22%20style%3D%22%22%2F%3E%3Cg%20transform%3D%22translate(12%2C%2012)%22%20class%3D%22cluster-label%22%3E%3CforeignObject%20height%3D%2224%22%20width%3D%22158.34375%22%3E%3Cdiv%20style%3D%22display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%3E%3Cspan%20class%3D%22nodeLabel%22%3E%3Cp%3EURL%20%2B%20Data%20Lake%20Flow%3C%2Fp%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22subgraph%22%3E%3Cg%20data-look%3D%22classic%22%20id%3D%22subGraph0%22%20class%3D%22cluster%22%3E%3Crect%20height%3D%22288.0862579345703%22%20width%3D%22188.046875%22%20y%3D%2246.456871032714844%22%20x%3D%22201.34375%22%20style%3D%22%22%2F%3E%3Cg%20transform%3D%22translate(205.34375%2C%2046.456871032714844)%22%20class%3D%22cluster-label%22%3E%3CforeignObject%20height%3D%2224%22%20width%3D%22180.046875%22%3E%3Cdiv%20style%3D%22display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%3E%3Cspan%20class%3D%22nodeLabel%22%3E%3Cp%3EBase64%20%2B%20Database%20Flow%3C%2Fp%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22nodes%22%3E%3Cg%20transform%3D%22translate(89.09895833333333%2C%20162)%22%20id%3D%22flowchart-CU-501%22%20class%3D%22node%20default%20nodeStyle%22%3E%3Crect%20height%3D%2254%22%20width%3D%22101.875%22%20y%3D%22-27%22%20x%3D%22-50.9375%22%20style%3D%22fill%3A%232d2d2d%20!important%3Bstroke%3A%23e0e0e0%20!important%3Bstroke-width%3A1px%20!important%22%20class%3D%22basic%20label-container%22%2F%3E%3Cg%20transform%3D%22translate(-20.9375%2C%20-12)%22%20style%3D%22color%3A%23ffffff%20!important%22%20class%3D%22label%22%3E%3Crect%2F%3E%3CforeignObject%20height%3D%2224%22%20width%3D%2241.875%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22color%3A%20rgb(255%2C%20255%2C%20255)%20!important%3B%20display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%20style%3D%22color%3A%23ffffff%20!important%22%3E%3Cp%3EClient%3C%2Fp%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20transform%3D%22translate(91.171875%2C%20330)%22%20id%3D%22flowchart-DL-502%22%20class%3D%22node%20default%20nodeStyle%22%3E%3Crect%20height%3D%2254%22%20width%3D%22132.140625%22%20y%3D%22-27%22%20x%3D%22-66.0703125%22%20style%3D%22fill%3A%232d2d2d%20!important%3Bstroke%3A%23e0e0e0%20!important%3Bstroke-width%3A1px%20!important%22%20class%3D%22basic%20label-container%22%2F%3E%3Cg%20transform%3D%22translate(-36.0703125%2C%20-12)%22%20style%3D%22color%3A%23ffffff%20!important%22%20class%3D%22label%22%3E%3Crect%2F%3E%3CforeignObject%20height%3D%2224%22%20width%3D%2272.140625%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22color%3A%20rgb(255%2C%20255%2C%20255)%20!important%3B%20display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%20style%3D%22color%3A%23ffffff%20!important%22%3E%3Cp%3EData%20Lake%3C%2Fp%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20transform%3D%22translate(89.09895833333333%2C%20246)%22%20id%3D%22flowchart-LLM_API_U-503%22%20class%3D%22node%20default%20nodeStyle%22%3E%3Crect%20height%3D%2254%22%20width%3D%22119.703125%22%20y%3D%22-27%22%20x%3D%22-59.8515625%22%20style%3D%22fill%3A%232d2d2d%20!important%3Bstroke%3A%23e0e0e0%20!important%3Bstroke-width%3A1px%20!important%22%20class%3D%22basic%20label-container%22%2F%3E%3Cg%20transform%3D%22translate(-29.8515625%2C%20-12)%22%20style%3D%22color%3A%23ffffff%20!important%22%20class%3D%22label%22%3E%3Crect%2F%3E%3CforeignObject%20height%3D%2224%22%20width%3D%2259.703125%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22color%3A%20rgb(255%2C%20255%2C%20255)%20!important%3B%20display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%20style%3D%22color%3A%23ffffff%20!important%22%3E%3Cp%3ELLM%20API%3C%2Fp%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20transform%3D%22translate(89.09895833333333%2C%2078)%22%20id%3D%22flowchart-DB_U-504%22%20class%3D%22node%20default%20nodeStyle%22%3E%3Crect%20height%3D%2254%22%20width%3D%22101.328125%22%20y%3D%22-27%22%20x%3D%22-50.6640625%22%20style%3D%22fill%3A%232d2d2d%20!important%3Bstroke%3A%23e0e0e0%20!important%3Bstroke-width%3A1px%20!important%22%20class%3D%22basic%20label-container%22%2F%3E%3Cg%20transform%3D%22translate(-20.6640625%2C%20-12)%22%20style%3D%22color%3A%23ffffff%20!important%22%20class%3D%22label%22%3E%3Crect%2F%3E%3CforeignObject%20height%3D%2224%22%20width%3D%2241.328125%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22color%3A%20rgb(255%2C%20255%2C%20255)%20!important%3B%20display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%20style%3D%22color%3A%23ffffff%20!important%22%3E%3Cp%3EDB_U%3C%2Fp%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20transform%3D%22translate(295.3671875%2C%20211.54312896728516)%22%20id%3D%22flowchart-CB-492%22%20class%3D%22node%20default%20nodeStyle%22%3E%3Crect%20height%3D%2254%22%20width%3D%22101.875%22%20y%3D%22-27%22%20x%3D%22-50.9375%22%20style%3D%22fill%3A%232d2d2d%20!important%3Bstroke%3A%23e0e0e0%20!important%3Bstroke-width%3A1px%20!important%22%20class%3D%22basic%20label-container%22%2F%3E%3Cg%20transform%3D%22translate(-20.9375%2C%20-12)%22%20style%3D%22color%3A%23ffffff%20!important%22%20class%3D%22label%22%3E%3Crect%2F%3E%3CforeignObject%20height%3D%2224%22%20width%3D%2241.875%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22color%3A%20rgb(255%2C%20255%2C%20255)%20!important%3B%20display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%20style%3D%22color%3A%23ffffff%20!important%22%3E%3Cp%3EClient%3C%2Fp%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20transform%3D%22translate(292.3489583333333%2C%20120)%22%20id%3D%22flowchart-DB_B-493%22%20class%3D%22node%20default%20nodeStyle%22%3E%3Cpath%20transform%3D%22translate(-41.8828125%2C%20-34.546590824040116)%22%20style%3D%22fill%3A%232d2d2d%20!important%3Bstroke%3A%23e0e0e0%20!important%3Bstroke-width%3A1px%20!important%22%20class%3D%22basic%20label-container%22%20d%3D%22M0%2C10.031060549360078%20a41.8828125%2C10.031060549360078%200%2C0%2C0%2083.765625%2C0%20a41.8828125%2C10.031060549360078%200%2C0%2C0%20-83.765625%2C0%20l0%2C49.03106054936008%20a41.8828125%2C10.031060549360078%200%2C0%2C0%2083.765625%2C0%20l0%2C-49.03106054936008%22%2F%3E%3Cg%20transform%3D%22translate(-34.3828125%2C%20-2)%22%20style%3D%22color%3A%23ffffff%20!important%22%20class%3D%22label%22%3E%3Crect%2F%3E%3CforeignObject%20height%3D%2224%22%20width%3D%2268.765625%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22color%3A%20rgb(255%2C%20255%2C%20255)%20!important%3B%20display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%20style%3D%22color%3A%23ffffff%20!important%22%3E%3Cp%3EDatabase%3C%2Fp%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20transform%3D%22translate(295.3671875%2C%20295.54312896728516)%22%20id%3D%22flowchart-LLM_API_B-494%22%20class%3D%22node%20default%20nodeStyle%22%3E%3Crect%20height%3D%2254%22%20width%3D%22119.703125%22%20y%3D%22-27%22%20x%3D%22-59.8515625%22%20style%3D%22fill%3A%232d2d2d%20!important%3Bstroke%3A%23e0e0e0%20!important%3Bstroke-width%3A1px%20!important%22%20class%3D%22basic%20label-container%22%2F%3E%3Cg%20transform%3D%22translate(-29.8515625%2C%20-12)%22%20style%3D%22color%3A%23ffffff%20!important%22%20class%3D%22label%22%3E%3Crect%2F%3E%3CforeignObject%20height%3D%2224%22%20width%3D%2259.703125%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22color%3A%20rgb(255%2C%20255%2C%20255)%20!important%3B%20display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%20style%3D%22color%3A%23ffffff%20!important%22%3E%3Cp%3ELLM%20API%3C%2Fp%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edges%20edgePaths%22%3E%3Cpath%20marker-end%3D%22url(%23mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-pointEnd)%22%20style%3D%22%22%20class%3D%22edge-thickness-normal%20edge-pattern-solid%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%22%20id%3D%22L_CB_DB_B_0_0%22%20d%3D%22M312.346%2C184.543L312.346%2C182.876C312.346%2C181.21%2C312.346%2C177.876%2C312.346%2C175.376C312.346%2C172.876%2C312.346%2C171.21%2C311.513%2C170.376C310.68%2C169.543%2C309.013%2C169.543%2C308.84%2C169.543C308.668%2C169.543%2C309.989%2C169.543%2C309.816%2C169.543C309.643%2C169.543%2C307.977%2C169.543%2C307.143%2C168.71C306.31%2C167.876%2C306.31%2C166.21%2C306.31%2C164.543C306.31%2C162.876%2C306.31%2C161.21%2C306.31%2C160.21C306.31%2C159.21%2C306.31%2C158.876%2C306.31%2C158.71L306.31%2C158.543%22%2F%3E%3Cpath%20marker-end%3D%22url(%23mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-pointEnd)%22%20style%3D%22%22%20class%3D%22edge-thickness-normal%20edge-pattern-solid%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%22%20id%3D%22L_DB_B_CB_1_0%22%20d%3D%22M278.388%2C154.543L278.388%2C157.043C278.388%2C159.543%2C278.388%2C164.543%2C278.388%2C168.876C278.388%2C173.21%2C278.388%2C176.876%2C278.388%2C178.71L278.388%2C180.543%22%2F%3E%3Cpath%20marker-end%3D%22url(%23mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-pointEnd)%22%20style%3D%22%22%20class%3D%22edge-thickness-normal%20edge-pattern-solid%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%22%20id%3D%22L_CB_LLM_API_B_2_0%22%20d%3D%22M295.367%2C238.543L295.367%2C241.043C295.367%2C243.543%2C295.367%2C248.543%2C295.367%2C252.876C295.367%2C257.21%2C295.367%2C260.876%2C295.367%2C262.71L295.367%2C264.543%22%2F%3E%3Cpath%20marker-end%3D%22url(%23mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-pointEnd)%22%20style%3D%22%22%20class%3D%22edge-thickness-normal%20edge-pattern-solid%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%22%20id%3D%22L_DB_U_CU_3_0%22%20d%3D%22M89.099%2C105L89.099%2C107.5C89.099%2C110%2C89.099%2C115%2C89.099%2C119.333C89.099%2C123.667%2C89.099%2C127.333%2C89.099%2C129.167L89.099%2C131%22%2F%3E%3Cpath%20marker-end%3D%22url(%23mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-pointEnd)%22%20style%3D%22%22%20class%3D%22edge-thickness-normal%20edge-pattern-solid%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%22%20id%3D%22L_CU_LLM_API_U_4_0%22%20d%3D%22M89.099%2C189L89.099%2C191.5C89.099%2C194%2C89.099%2C199%2C89.099%2C203.333C89.099%2C207.667%2C89.099%2C211.333%2C89.099%2C213.167L89.099%2C215%22%2F%3E%3Cpath%20marker-end%3D%22url(%23mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-pointEnd)%22%20style%3D%22%22%20class%3D%22edge-thickness-normal%20edge-pattern-solid%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%22%20id%3D%22L_LLM_API_U_DL_5_0%22%20d%3D%22M109.049%2C273L109.049%2C275.5C109.049%2C278%2C109.049%2C283%2C109.74%2C285.5C110.431%2C288%2C111.813%2C288%2C112.504%2C289.25C113.195%2C290.5%2C113.195%2C293%2C113.195%2C294.833C113.195%2C296.667%2C113.195%2C297.833%2C113.195%2C298.417L113.195%2C299%22%2F%3E%3Cpath%20marker-end%3D%22url(%23mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-pointEnd)%22%20style%3D%22%22%20class%3D%22edge-thickness-normal%20edge-pattern-solid%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%22%20id%3D%22L_DL_LLM_API_U_6_0%22%20d%3D%22M69.148%2C303L69.148%2C300.5C69.148%2C298%2C69.148%2C293%2C69.148%2C288.667C69.148%2C284.333%2C69.148%2C280.667%2C69.148%2C278.833L69.148%2C277%22%2F%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabels%22%3E%3Cg%20class%3D%22edgeLabel%22%3E%3Cg%20transform%3D%22translate(0%2C%200)%22%20class%3D%22label%22%3E%3CforeignObject%20height%3D%220%22%20width%3D%220%22%3E%3Cdiv%20style%3D%22display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%20class%3D%22labelBkg%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%3E%3Cspan%20class%3D%22edgeLabel%22%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabel%22%3E%3Cg%20transform%3D%22translate(0%2C%200)%22%20class%3D%22label%22%3E%3CforeignObject%20height%3D%220%22%20width%3D%220%22%3E%3Cdiv%20style%3D%22display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%20class%3D%22labelBkg%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%3E%3Cspan%20class%3D%22edgeLabel%22%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabel%22%3E%3Cg%20transform%3D%22translate(0%2C%200)%22%20class%3D%22label%22%3E%3CforeignObject%20height%3D%220%22%20width%3D%220%22%3E%3Cdiv%20style%3D%22display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%20class%3D%22labelBkg%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%3E%3Cspan%20class%3D%22edgeLabel%22%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabel%22%3E%3Cg%20transform%3D%22translate(0%2C%200)%22%20class%3D%22label%22%3E%3CforeignObject%20height%3D%220%22%20width%3D%220%22%3E%3Cdiv%20style%3D%22display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%20class%3D%22labelBkg%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%3E%3Cspan%20class%3D%22edgeLabel%22%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabel%22%3E%3Cg%20transform%3D%22translate(0%2C%200)%22%20class%3D%22label%22%3E%3CforeignObject%20height%3D%220%22%20width%3D%220%22%3E%3Cdiv%20style%3D%22display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%20class%3D%22labelBkg%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%3E%3Cspan%20class%3D%22edgeLabel%22%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabel%22%3E%3Cg%20transform%3D%22translate(0%2C%200)%22%20class%3D%22label%22%3E%3CforeignObject%20height%3D%220%22%20width%3D%220%22%3E%3Cdiv%20style%3D%22display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%20class%3D%22labelBkg%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%3E%3Cspan%20class%3D%22edgeLabel%22%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabel%22%3E%3Cg%20transform%3D%22translate(0%2C%200)%22%20class%3D%22label%22%3E%3CforeignObject%20height%3D%220%22%20width%3D%220%22%3E%3Cdiv%20style%3D%22display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%20class%3D%22labelBkg%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%3E%3Cspan%20class%3D%22edgeLabel%22%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fsvg%3E)

[Diagram](data:image/svg+xml;charset=utf-8,%3Csvg%20aria-roledescription%3D%22flowchart-v2%22%20role%3D%22graphics-document%20document%22%20viewBox%3D%22-32%20-28%20461.390625%20437%22%20style%3D%22max-width%3A%20397.390625px%3B%22%20class%3D%22flowchart%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22461.390625%22%20id%3D%22mermaid-e47045a8-d091-4170-b395-992e34910d18%22%20height%3D%22437%22%3E%3Crect%20x%3D%22-32%22%20y%3D%22-28%22%20width%3D%22461.390625%22%20height%3D%22437%22%20fill%3D%22%23191919%22%2F%3E%3Cstyle%3E%23mermaid-e47045a8-d091-4170-b395-992e34910d18%7Bfont-family%3Aui-sans-serif%2C-apple-system%2CBlinkMacSystemFont%2C%22Segoe%20UI%20Variable%20Display%22%2C%22Segoe%20UI%22%2CHelvetica%2C%22Apple%20Color%20Emoji%22%2CArial%2Csans-serif%2C%22Segoe%20UI%20Emoji%22%2C%22Segoe%20UI%20Symbol%22%3Bfont-size%3A16px%3Bfill%3A%23ccc%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.error-icon%7Bfill%3A%23a44141%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.error-text%7Bfill%3A%23ddd%3Bstroke%3A%23ddd%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.edge-thickness-normal%7Bstroke-width%3A1px%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.edge-thickness-thick%7Bstroke-width%3A3.5px%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.edge-pattern-solid%7Bstroke-dasharray%3A0%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.edge-thickness-invisible%7Bstroke-width%3A0%3Bfill%3Anone%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.edge-pattern-dashed%7Bstroke-dasharray%3A3%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.edge-pattern-dotted%7Bstroke-dasharray%3A2%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.marker%7Bfill%3Alightgrey%3Bstroke%3Alightgrey%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.marker.cross%7Bstroke%3Alightgrey%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20svg%7Bfont-family%3Aui-sans-serif%2C-apple-system%2CBlinkMacSystemFont%2C%22Segoe%20UI%20Variable%20Display%22%2C%22Segoe%20UI%22%2CHelvetica%2C%22Apple%20Color%20Emoji%22%2CArial%2Csans-serif%2C%22Segoe%20UI%20Emoji%22%2C%22Segoe%20UI%20Symbol%22%3Bfont-size%3A16px%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20p%7Bmargin%3A0%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.label%7Bfont-family%3Aui-sans-serif%2C-apple-system%2CBlinkMacSystemFont%2C%22Segoe%20UI%20Variable%20Display%22%2C%22Segoe%20UI%22%2CHelvetica%2C%22Apple%20Color%20Emoji%22%2CArial%2Csans-serif%2C%22Segoe%20UI%20Emoji%22%2C%22Segoe%20UI%20Symbol%22%3Bcolor%3A%23ccc%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.cluster-label%20text%7Bfill%3A%23F9FFFE%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.cluster-label%20span%7Bcolor%3A%23F9FFFE%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.cluster-label%20span%20p%7Bbackground-color%3Atransparent%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.label%20text%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20span%7Bfill%3A%23ccc%3Bcolor%3A%23ccc%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.node%20rect%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.node%20circle%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.node%20ellipse%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.node%20polygon%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.node%20path%7Bfill%3A%231f2020%3Bstroke%3A%23ccc%3Bstroke-width%3A1px%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.rough-node%20.label%20text%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.node%20.label%20text%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.image-shape%20.label%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.icon-shape%20.label%7Btext-anchor%3Amiddle%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.node%20.katex%20path%7Bfill%3A%23000%3Bstroke%3A%23000%3Bstroke-width%3A1px%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.rough-node%20.label%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.node%20.label%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.image-shape%20.label%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.icon-shape%20.label%7Btext-align%3Acenter%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.node.clickable%7Bcursor%3Apointer%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.root%20.anchor%20path%7Bfill%3Alightgrey!important%3Bstroke-width%3A0%3Bstroke%3Alightgrey%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.arrowheadPath%7Bfill%3Alightgrey%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.edgePath%20.path%7Bstroke%3Alightgrey%3Bstroke-width%3A2.0px%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.flowchart-link%7Bstroke%3Alightgrey%3Bfill%3Anone%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.edgeLabel%7Bbackground-color%3Ahsl(0%2C%200%25%2C%2034.4117647059%25)%3Btext-align%3Acenter%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.edgeLabel%20p%7Bbackground-color%3Ahsl(0%2C%200%25%2C%2034.4117647059%25)%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.edgeLabel%20rect%7Bopacity%3A0.5%3Bbackground-color%3Ahsl(0%2C%200%25%2C%2034.4117647059%25)%3Bfill%3Ahsl(0%2C%200%25%2C%2034.4117647059%25)%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.labelBkg%7Bbackground-color%3Argba(87.75%2C%2087.75%2C%2087.75%2C%200.5)%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.cluster%20rect%7Bfill%3Ahsl(180%2C%201.5873015873%25%2C%2028.3529411765%25)%3Bstroke%3Argba(255%2C%20255%2C%20255%2C%200.25)%3Bstroke-width%3A1px%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.cluster%20text%7Bfill%3A%23F9FFFE%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.cluster%20span%7Bcolor%3A%23F9FFFE%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20div.mermaidTooltip%7Bposition%3Aabsolute%3Btext-align%3Acenter%3Bmax-width%3A200px%3Bpadding%3A2px%3Bfont-family%3Aui-sans-serif%2C-apple-system%2CBlinkMacSystemFont%2C%22Segoe%20UI%20Variable%20Display%22%2C%22Segoe%20UI%22%2CHelvetica%2C%22Apple%20Color%20Emoji%22%2CArial%2Csans-serif%2C%22Segoe%20UI%20Emoji%22%2C%22Segoe%20UI%20Symbol%22%3Bfont-size%3A12px%3Bbackground%3Ahsl(20%2C%201.5873015873%25%2C%2012.3529411765%25)%3Bborder%3A1px%20solid%20rgba(255%2C%20255%2C%20255%2C%200.25)%3Bborder-radius%3A2px%3Bpointer-events%3Anone%3Bz-index%3A100%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.flowchartTitleText%7Btext-anchor%3Amiddle%3Bfont-size%3A18px%3Bfill%3A%23ccc%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20rect.text%7Bfill%3Anone%3Bstroke-width%3A0%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.icon-shape%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.image-shape%7Bbackground-color%3Ahsl(0%2C%200%25%2C%2034.4117647059%25)%3Btext-align%3Acenter%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.icon-shape%20p%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.image-shape%20p%7Bbackground-color%3Ahsl(0%2C%200%25%2C%2034.4117647059%25)%3Bpadding%3A2px%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.icon-shape%20rect%2C%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.image-shape%20rect%7Bopacity%3A0.5%3Bbackground-color%3Ahsl(0%2C%200%25%2C%2034.4117647059%25)%3Bfill%3Ahsl(0%2C%200%25%2C%2034.4117647059%25)%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20%3Aroot%7B--mermaid-font-family%3A%22trebuchet%20ms%22%2Cverdana%2Carial%2Csans-serif%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.defaultStyle%26gt%3B*%7Bfill%3A%232d2d2d!important%3Bstroke%3A%23e0e0e0!important%3Bstroke-width%3A1px!important%3Bcolor%3A%23ffffff!important%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.defaultStyle%20span%7Bfill%3A%232d2d2d!important%3Bstroke%3A%23e0e0e0!important%3Bstroke-width%3A1px!important%3Bcolor%3A%23ffffff!important%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.defaultStyle%20tspan%7Bfill%3A%23ffffff!important%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.subgraphStyle%26gt%3B*%7Bfill%3A%232d2d2d!important%3Bstroke%3A%2387ceeb!important%3Bstroke-width%3A1px!important%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.subgraphStyle%20span%7Bfill%3A%232d2d2d!important%3Bstroke%3A%2387ceeb!important%3Bstroke-width%3A1px!important%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.nodeStyle%26gt%3B*%7Bfill%3A%232d2d2d!important%3Bstroke%3A%23e0e0e0!important%3Bstroke-width%3A1px!important%3Bcolor%3A%23ffffff!important%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.nodeStyle%20span%7Bfill%3A%232d2d2d!important%3Bstroke%3A%23e0e0e0!important%3Bstroke-width%3A1px!important%3Bcolor%3A%23ffffff!important%3B%7D%23mermaid-e47045a8-d091-4170-b395-992e34910d18%20.nodeStyle%20tspan%7Bfill%3A%23ffffff!important%3B%7D%3C%2Fstyle%3E%3Cg%3E%3Cmarker%20orient%3D%22auto%22%20markerHeight%3D%228%22%20markerWidth%3D%228%22%20markerUnits%3D%22userSpaceOnUse%22%20refY%3D%225%22%20refX%3D%225%22%20viewBox%3D%220%200%2010%2010%22%20class%3D%22marker%20flowchart-v2%22%20id%3D%22mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-pointEnd%22%3E%3Cpath%20style%3D%22stroke-width%3A%201%3B%20stroke-dasharray%3A%201%2C%200%3B%22%20class%3D%22arrowMarkerPath%22%20d%3D%22M%200%200%20L%2010%205%20L%200%2010%20z%22%2F%3E%3C%2Fmarker%3E%3Cmarker%20orient%3D%22auto%22%20markerHeight%3D%228%22%20markerWidth%3D%228%22%20markerUnits%3D%22userSpaceOnUse%22%20refY%3D%225%22%20refX%3D%224.5%22%20viewBox%3D%220%200%2010%2010%22%20class%3D%22marker%20flowchart-v2%22%20id%3D%22mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-pointStart%22%3E%3Cpath%20style%3D%22stroke-width%3A%201%3B%20stroke-dasharray%3A%201%2C%200%3B%22%20class%3D%22arrowMarkerPath%22%20d%3D%22M%200%205%20L%2010%2010%20L%2010%200%20z%22%2F%3E%3C%2Fmarker%3E%3Cmarker%20orient%3D%22auto%22%20markerHeight%3D%2211%22%20markerWidth%3D%2211%22%20markerUnits%3D%22userSpaceOnUse%22%20refY%3D%225%22%20refX%3D%2211%22%20viewBox%3D%220%200%2010%2010%22%20class%3D%22marker%20flowchart-v2%22%20id%3D%22mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-circleEnd%22%3E%3Ccircle%20style%3D%22stroke-width%3A%201%3B%20stroke-dasharray%3A%201%2C%200%3B%22%20class%3D%22arrowMarkerPath%22%20r%3D%225%22%20cy%3D%225%22%20cx%3D%225%22%2F%3E%3C%2Fmarker%3E%3Cmarker%20orient%3D%22auto%22%20markerHeight%3D%2211%22%20markerWidth%3D%2211%22%20markerUnits%3D%22userSpaceOnUse%22%20refY%3D%225%22%20refX%3D%22-1%22%20viewBox%3D%220%200%2010%2010%22%20class%3D%22marker%20flowchart-v2%22%20id%3D%22mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-circleStart%22%3E%3Ccircle%20style%3D%22stroke-width%3A%201%3B%20stroke-dasharray%3A%201%2C%200%3B%22%20class%3D%22arrowMarkerPath%22%20r%3D%225%22%20cy%3D%225%22%20cx%3D%225%22%2F%3E%3C%2Fmarker%3E%3Cmarker%20orient%3D%22auto%22%20markerHeight%3D%2211%22%20markerWidth%3D%2211%22%20markerUnits%3D%22userSpaceOnUse%22%20refY%3D%225.2%22%20refX%3D%2212%22%20viewBox%3D%220%200%2011%2011%22%20class%3D%22marker%20cross%20flowchart-v2%22%20id%3D%22mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-crossEnd%22%3E%3Cpath%20style%3D%22stroke-width%3A%202%3B%20stroke-dasharray%3A%201%2C%200%3B%22%20class%3D%22arrowMarkerPath%22%20d%3D%22M%201%2C1%20l%209%2C9%20M%2010%2C1%20l%20-9%2C9%22%2F%3E%3C%2Fmarker%3E%3Cmarker%20orient%3D%22auto%22%20markerHeight%3D%2211%22%20markerWidth%3D%2211%22%20markerUnits%3D%22userSpaceOnUse%22%20refY%3D%225.2%22%20refX%3D%22-1%22%20viewBox%3D%220%200%2011%2011%22%20class%3D%22marker%20cross%20flowchart-v2%22%20id%3D%22mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-crossStart%22%3E%3Cpath%20style%3D%22stroke-width%3A%202%3B%20stroke-dasharray%3A%201%2C%200%3B%22%20class%3D%22arrowMarkerPath%22%20d%3D%22M%201%2C1%20l%209%2C9%20M%2010%2C1%20l%20-9%2C9%22%2F%3E%3C%2Fmarker%3E%3C%2Fg%3E%3Cg%20class%3D%22subgraphs%22%3E%3Cg%20class%3D%22subgraph%22%3E%3Cg%20data-look%3D%22classic%22%20id%3D%22subGraph1%22%20class%3D%22cluster%22%3E%3Crect%20height%3D%22357%22%20width%3D%22166.34375%22%20y%3D%2212%22%20x%3D%228%22%20style%3D%22%22%2F%3E%3Cg%20transform%3D%22translate(12%2C%2012)%22%20class%3D%22cluster-label%22%3E%3CforeignObject%20height%3D%2224%22%20width%3D%22158.34375%22%3E%3Cdiv%20style%3D%22display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%3E%3Cspan%20class%3D%22nodeLabel%22%3E%3Cp%3EURL%20%2B%20Data%20Lake%20Flow%3C%2Fp%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22subgraph%22%3E%3Cg%20data-look%3D%22classic%22%20id%3D%22subGraph0%22%20class%3D%22cluster%22%3E%3Crect%20height%3D%22288.0862579345703%22%20width%3D%22188.046875%22%20y%3D%2246.456871032714844%22%20x%3D%22201.34375%22%20style%3D%22%22%2F%3E%3Cg%20transform%3D%22translate(205.34375%2C%2046.456871032714844)%22%20class%3D%22cluster-label%22%3E%3CforeignObject%20height%3D%2224%22%20width%3D%22180.046875%22%3E%3Cdiv%20style%3D%22display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%3E%3Cspan%20class%3D%22nodeLabel%22%3E%3Cp%3EBase64%20%2B%20Database%20Flow%3C%2Fp%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22nodes%22%3E%3Cg%20transform%3D%22translate(89.09895833333333%2C%20162)%22%20id%3D%22flowchart-CU-501%22%20class%3D%22node%20default%20nodeStyle%22%3E%3Crect%20height%3D%2254%22%20width%3D%22101.875%22%20y%3D%22-27%22%20x%3D%22-50.9375%22%20style%3D%22fill%3A%232d2d2d%20!important%3Bstroke%3A%23e0e0e0%20!important%3Bstroke-width%3A1px%20!important%22%20class%3D%22basic%20label-container%22%2F%3E%3Cg%20transform%3D%22translate(-20.9375%2C%20-12)%22%20style%3D%22color%3A%23ffffff%20!important%22%20class%3D%22label%22%3E%3Crect%2F%3E%3CforeignObject%20height%3D%2224%22%20width%3D%2241.875%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22color%3A%20rgb(255%2C%20255%2C%20255)%20!important%3B%20display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%20style%3D%22color%3A%23ffffff%20!important%22%3E%3Cp%3EClient%3C%2Fp%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20transform%3D%22translate(91.171875%2C%20330)%22%20id%3D%22flowchart-DL-502%22%20class%3D%22node%20default%20nodeStyle%22%3E%3Crect%20height%3D%2254%22%20width%3D%22132.140625%22%20y%3D%22-27%22%20x%3D%22-66.0703125%22%20style%3D%22fill%3A%232d2d2d%20!important%3Bstroke%3A%23e0e0e0%20!important%3Bstroke-width%3A1px%20!important%22%20class%3D%22basic%20label-container%22%2F%3E%3Cg%20transform%3D%22translate(-36.0703125%2C%20-12)%22%20style%3D%22color%3A%23ffffff%20!important%22%20class%3D%22label%22%3E%3Crect%2F%3E%3CforeignObject%20height%3D%2224%22%20width%3D%2272.140625%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22color%3A%20rgb(255%2C%20255%2C%20255)%20!important%3B%20display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%20style%3D%22color%3A%23ffffff%20!important%22%3E%3Cp%3EData%20Lake%3C%2Fp%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20transform%3D%22translate(89.09895833333333%2C%20246)%22%20id%3D%22flowchart-LLM_API_U-503%22%20class%3D%22node%20default%20nodeStyle%22%3E%3Crect%20height%3D%2254%22%20width%3D%22119.703125%22%20y%3D%22-27%22%20x%3D%22-59.8515625%22%20style%3D%22fill%3A%232d2d2d%20!important%3Bstroke%3A%23e0e0e0%20!important%3Bstroke-width%3A1px%20!important%22%20class%3D%22basic%20label-container%22%2F%3E%3Cg%20transform%3D%22translate(-29.8515625%2C%20-12)%22%20style%3D%22color%3A%23ffffff%20!important%22%20class%3D%22label%22%3E%3Crect%2F%3E%3CforeignObject%20height%3D%2224%22%20width%3D%2259.703125%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22color%3A%20rgb(255%2C%20255%2C%20255)%20!important%3B%20display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%20style%3D%22color%3A%23ffffff%20!important%22%3E%3Cp%3ELLM%20API%3C%2Fp%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20transform%3D%22translate(89.09895833333333%2C%2078)%22%20id%3D%22flowchart-DB_U-504%22%20class%3D%22node%20default%20nodeStyle%22%3E%3Crect%20height%3D%2254%22%20width%3D%22101.328125%22%20y%3D%22-27%22%20x%3D%22-50.6640625%22%20style%3D%22fill%3A%232d2d2d%20!important%3Bstroke%3A%23e0e0e0%20!important%3Bstroke-width%3A1px%20!important%22%20class%3D%22basic%20label-container%22%2F%3E%3Cg%20transform%3D%22translate(-20.6640625%2C%20-12)%22%20style%3D%22color%3A%23ffffff%20!important%22%20class%3D%22label%22%3E%3Crect%2F%3E%3CforeignObject%20height%3D%2224%22%20width%3D%2241.328125%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22color%3A%20rgb(255%2C%20255%2C%20255)%20!important%3B%20display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%20style%3D%22color%3A%23ffffff%20!important%22%3E%3Cp%3EDB_U%3C%2Fp%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20transform%3D%22translate(295.3671875%2C%20211.54312896728516)%22%20id%3D%22flowchart-CB-492%22%20class%3D%22node%20default%20nodeStyle%22%3E%3Crect%20height%3D%2254%22%20width%3D%22101.875%22%20y%3D%22-27%22%20x%3D%22-50.9375%22%20style%3D%22fill%3A%232d2d2d%20!important%3Bstroke%3A%23e0e0e0%20!important%3Bstroke-width%3A1px%20!important%22%20class%3D%22basic%20label-container%22%2F%3E%3Cg%20transform%3D%22translate(-20.9375%2C%20-12)%22%20style%3D%22color%3A%23ffffff%20!important%22%20class%3D%22label%22%3E%3Crect%2F%3E%3CforeignObject%20height%3D%2224%22%20width%3D%2241.875%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22color%3A%20rgb(255%2C%20255%2C%20255)%20!important%3B%20display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%20style%3D%22color%3A%23ffffff%20!important%22%3E%3Cp%3EClient%3C%2Fp%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20transform%3D%22translate(292.3489583333333%2C%20120)%22%20id%3D%22flowchart-DB_B-493%22%20class%3D%22node%20default%20nodeStyle%22%3E%3Cpath%20transform%3D%22translate(-41.8828125%2C%20-34.546590824040116)%22%20style%3D%22fill%3A%232d2d2d%20!important%3Bstroke%3A%23e0e0e0%20!important%3Bstroke-width%3A1px%20!important%22%20class%3D%22basic%20label-container%22%20d%3D%22M0%2C10.031060549360078%20a41.8828125%2C10.031060549360078%200%2C0%2C0%2083.765625%2C0%20a41.8828125%2C10.031060549360078%200%2C0%2C0%20-83.765625%2C0%20l0%2C49.03106054936008%20a41.8828125%2C10.031060549360078%200%2C0%2C0%2083.765625%2C0%20l0%2C-49.03106054936008%22%2F%3E%3Cg%20transform%3D%22translate(-34.3828125%2C%20-2)%22%20style%3D%22color%3A%23ffffff%20!important%22%20class%3D%22label%22%3E%3Crect%2F%3E%3CforeignObject%20height%3D%2224%22%20width%3D%2268.765625%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22color%3A%20rgb(255%2C%20255%2C%20255)%20!important%3B%20display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%20style%3D%22color%3A%23ffffff%20!important%22%3E%3Cp%3EDatabase%3C%2Fp%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20transform%3D%22translate(295.3671875%2C%20295.54312896728516)%22%20id%3D%22flowchart-LLM_API_B-494%22%20class%3D%22node%20default%20nodeStyle%22%3E%3Crect%20height%3D%2254%22%20width%3D%22119.703125%22%20y%3D%22-27%22%20x%3D%22-59.8515625%22%20style%3D%22fill%3A%232d2d2d%20!important%3Bstroke%3A%23e0e0e0%20!important%3Bstroke-width%3A1px%20!important%22%20class%3D%22basic%20label-container%22%2F%3E%3Cg%20transform%3D%22translate(-29.8515625%2C%20-12)%22%20style%3D%22color%3A%23ffffff%20!important%22%20class%3D%22label%22%3E%3Crect%2F%3E%3CforeignObject%20height%3D%2224%22%20width%3D%2259.703125%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22color%3A%20rgb(255%2C%20255%2C%20255)%20!important%3B%20display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%20style%3D%22color%3A%23ffffff%20!important%22%3E%3Cp%3ELLM%20API%3C%2Fp%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edges%20edgePaths%22%3E%3Cpath%20marker-end%3D%22url(%23mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-pointEnd)%22%20style%3D%22%22%20class%3D%22edge-thickness-normal%20edge-pattern-solid%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%22%20id%3D%22L_CB_DB_B_0_0%22%20d%3D%22M312.346%2C184.543L312.346%2C182.876C312.346%2C181.21%2C312.346%2C177.876%2C312.346%2C175.376C312.346%2C172.876%2C312.346%2C171.21%2C311.513%2C170.376C310.68%2C169.543%2C309.013%2C169.543%2C308.84%2C169.543C308.668%2C169.543%2C309.989%2C169.543%2C309.816%2C169.543C309.643%2C169.543%2C307.977%2C169.543%2C307.143%2C168.71C306.31%2C167.876%2C306.31%2C166.21%2C306.31%2C164.543C306.31%2C162.876%2C306.31%2C161.21%2C306.31%2C160.21C306.31%2C159.21%2C306.31%2C158.876%2C306.31%2C158.71L306.31%2C158.543%22%2F%3E%3Cpath%20marker-end%3D%22url(%23mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-pointEnd)%22%20style%3D%22%22%20class%3D%22edge-thickness-normal%20edge-pattern-solid%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%22%20id%3D%22L_DB_B_CB_1_0%22%20d%3D%22M278.388%2C154.543L278.388%2C157.043C278.388%2C159.543%2C278.388%2C164.543%2C278.388%2C168.876C278.388%2C173.21%2C278.388%2C176.876%2C278.388%2C178.71L278.388%2C180.543%22%2F%3E%3Cpath%20marker-end%3D%22url(%23mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-pointEnd)%22%20style%3D%22%22%20class%3D%22edge-thickness-normal%20edge-pattern-solid%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%22%20id%3D%22L_CB_LLM_API_B_2_0%22%20d%3D%22M295.367%2C238.543L295.367%2C241.043C295.367%2C243.543%2C295.367%2C248.543%2C295.367%2C252.876C295.367%2C257.21%2C295.367%2C260.876%2C295.367%2C262.71L295.367%2C264.543%22%2F%3E%3Cpath%20marker-end%3D%22url(%23mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-pointEnd)%22%20style%3D%22%22%20class%3D%22edge-thickness-normal%20edge-pattern-solid%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%22%20id%3D%22L_DB_U_CU_3_0%22%20d%3D%22M89.099%2C105L89.099%2C107.5C89.099%2C110%2C89.099%2C115%2C89.099%2C119.333C89.099%2C123.667%2C89.099%2C127.333%2C89.099%2C129.167L89.099%2C131%22%2F%3E%3Cpath%20marker-end%3D%22url(%23mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-pointEnd)%22%20style%3D%22%22%20class%3D%22edge-thickness-normal%20edge-pattern-solid%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%22%20id%3D%22L_CU_LLM_API_U_4_0%22%20d%3D%22M89.099%2C189L89.099%2C191.5C89.099%2C194%2C89.099%2C199%2C89.099%2C203.333C89.099%2C207.667%2C89.099%2C211.333%2C89.099%2C213.167L89.099%2C215%22%2F%3E%3Cpath%20marker-end%3D%22url(%23mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-pointEnd)%22%20style%3D%22%22%20class%3D%22edge-thickness-normal%20edge-pattern-solid%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%22%20id%3D%22L_LLM_API_U_DL_5_0%22%20d%3D%22M109.049%2C273L109.049%2C275.5C109.049%2C278%2C109.049%2C283%2C109.74%2C285.5C110.431%2C288%2C111.813%2C288%2C112.504%2C289.25C113.195%2C290.5%2C113.195%2C293%2C113.195%2C294.833C113.195%2C296.667%2C113.195%2C297.833%2C113.195%2C298.417L113.195%2C299%22%2F%3E%3Cpath%20marker-end%3D%22url(%23mermaid-e47045a8-d091-4170-b395-992e34910d18_flowchart-v2-pointEnd)%22%20style%3D%22%22%20class%3D%22edge-thickness-normal%20edge-pattern-solid%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%22%20id%3D%22L_DL_LLM_API_U_6_0%22%20d%3D%22M69.148%2C303L69.148%2C300.5C69.148%2C298%2C69.148%2C293%2C69.148%2C288.667C69.148%2C284.333%2C69.148%2C280.667%2C69.148%2C278.833L69.148%2C277%22%2F%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabels%22%3E%3Cg%20class%3D%22edgeLabel%22%3E%3Cg%20transform%3D%22translate(0%2C%200)%22%20class%3D%22label%22%3E%3CforeignObject%20height%3D%220%22%20width%3D%220%22%3E%3Cdiv%20style%3D%22display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%20class%3D%22labelBkg%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%3E%3Cspan%20class%3D%22edgeLabel%22%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabel%22%3E%3Cg%20transform%3D%22translate(0%2C%200)%22%20class%3D%22label%22%3E%3CforeignObject%20height%3D%220%22%20width%3D%220%22%3E%3Cdiv%20style%3D%22display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%20class%3D%22labelBkg%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%3E%3Cspan%20class%3D%22edgeLabel%22%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabel%22%3E%3Cg%20transform%3D%22translate(0%2C%200)%22%20class%3D%22label%22%3E%3CforeignObject%20height%3D%220%22%20width%3D%220%22%3E%3Cdiv%20style%3D%22display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%20class%3D%22labelBkg%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%3E%3Cspan%20class%3D%22edgeLabel%22%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabel%22%3E%3Cg%20transform%3D%22translate(0%2C%200)%22%20class%3D%22label%22%3E%3CforeignObject%20height%3D%220%22%20width%3D%220%22%3E%3Cdiv%20style%3D%22display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%20class%3D%22labelBkg%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%3E%3Cspan%20class%3D%22edgeLabel%22%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabel%22%3E%3Cg%20transform%3D%22translate(0%2C%200)%22%20class%3D%22label%22%3E%3CforeignObject%20height%3D%220%22%20width%3D%220%22%3E%3Cdiv%20style%3D%22display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%20class%3D%22labelBkg%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%3E%3Cspan%20class%3D%22edgeLabel%22%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabel%22%3E%3Cg%20transform%3D%22translate(0%2C%200)%22%20class%3D%22label%22%3E%3CforeignObject%20height%3D%220%22%20width%3D%220%22%3E%3Cdiv%20style%3D%22display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%20class%3D%22labelBkg%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%3E%3Cspan%20class%3D%22edgeLabel%22%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabel%22%3E%3Cg%20transform%3D%22translate(0%2C%200)%22%20class%3D%22label%22%3E%3CforeignObject%20height%3D%220%22%20width%3D%220%22%3E%3Cdiv%20style%3D%22display%3A%20table-cell%3B%20white-space%3A%20nowrap%3B%20line-height%3A%201.5%3B%20max-width%3A%20200px%3B%20text-align%3A%20center%3B%22%20class%3D%22labelBkg%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%3E%3Cspan%20class%3D%22edgeLabel%22%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fsvg%3E)

<aside>
💡

**Tip:** Use raw bytes for quick, local tests. Use Base64 when you need to store media files directly in a traditional database. Use URLs for scalable, production systems that leverage cloud storage or need to access public web content.

</aside>

Now, let us get into the code. We will start by processing an image as raw bytes to generate a caption:

1. First, let’s display our sample image: a cat interacting with a robot.

```python
from pathlib import Path
from IPython.display import Image as IPythonImage, display

def display_image(image_path: Path) -> None:
    image = IPythonImage(filename=image_path, width=400)
    display(image)

display_image(Path("images") / "image_1.jpeg")
```

![A fluffy grey tabby kitten playfully perched on the arm of a large, dark metallic robot.](Lesson%2011%20Multimodal%20Data%20250f9b6f42708088b0e5f3ccb1e43223/image_1.jpeg)

Figure 10: A fluffy grey tabby kitten playfully perched on the arm of a large, dark metallic robot

1. Next, we define a function to load the image as bytes. We convert it to the `WEBP` format because it is one of the most storage-efficient formats out there:
    
    ```python
    import io
    from typing import Literal
    from PIL import Image as PILImage
    
    def load_image_as_bytes(
        image_path: Path, format: Literal["WEBP", "JPEG", "PNG"] = "WEBP", max_width: int = 600, return_size: bool = False
    ) -> bytes | tuple[bytes, tuple[int, int]]:
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
    ```
    
    The `image_bytes` variable now holds the image data. Its size is 44392 bytes.
    
2. With the image loaded as bytes, we can pass it to the Gemini model along with a text prompt to generate a caption.
    
    ```python
    import google.generativeai as genai
    from google.generativeai import types
    
    # Configure your Gemini client# genai.configure(api_key="YOUR_GOOGLE_API_KEY")
    client = genai.Client()
    MODEL_ID = "gemini-2.5-flash"
    
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
    print(response.text)
    ```
    
    The model returns a detailed description:
    
    ```
    This striking image features a massive, dark metallic robot, its powerful form detailed with intricate circuit patterns on its head and piercing red glowing eyes. Perched playfully on its right arm is a small, fluffy grey tabby kitten, its front paw raised as if exploring or batting at the robot's armored limb, while its gaze is directed slightly off-frame. The robot's large, segmented hand is visible beneath the kitten. The background suggests an industrial or workshop environment, with hints of metal structures and natural light filtering in from an unseen window, creating a dramatic contrast between the soft, vulnerable kitten and the formidable, mechanical sentinel.
    ```
    
3. We can easily adapt this method to pass multiple images simultaneously. For example, we can ask the LLM to compare the previous image with the one below:

![A fluffy white dog in a tense, aggressive stance facing a sleek black robot in a cluttered urban alleyway.](Lesson%2011%20Multimodal%20Data%20250f9b6f42708088b0e5f3ccb1e43223/image_2.jpeg)

Figure 11: A fluffy white dog in a tense, aggressive stance facing a sleek black robot in a cluttered urban alleyway.

1. Now, we ask the model to describe the differences between the two images:
    
    ```python
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
    print(response.text)
    ```
    
    The LLM response highlights the differences:
    
    ```
    The primary difference between the two images lies in the nature of the 
    interaction depicted and their respective settings. In the first image, 
    a small, grey kitten is shown curiously interacting with a large, metallic robot,
    gently perched on its arm within what appears to be a clean, well-lit workshop 
    or industrial space. Conversely, the second image 
    portrays a tense and aggressive confrontation between a 
    fluffy white dog and a sleek black robot, both in combative stances, 
    amidst a cluttered and grimy urban alleyway filled with trash and graffiti.
    ```
    
2. We can follow a similar process for Base64 encoding. We define a helper function to convert the image bytes to a Base64 string.
    
    ```python
    import base64
    from typing import cast
    
    def load_image_as_base64(
        image_path: Path, format: Literal["WEBP", "JPEG", "PNG"] = "WEBP", max_width: int = 600
    ) -> str:
        image_bytes = load_image_as_bytes(image_path=image_path, format=format, max_width=max_width)
        return base64.b64encode(cast(bytes, image_bytes)).decode("utf-8")
    
    image_base64 = load_image_as_base64(image_path=Path("images") / "image_1.jpeg", format="WEBP")
    ```
    
    The `image_base64` variable now holds the Base64 string. The encoded image starts with the following string `UklGRmCtAABXRUJQVlA4IFStAABQ7AKdASpYAlgCPm0ylEekIqInJnQ7gOANiWdtk7FnEo2gDknjPixW9SNSb5P7IbBNhLn87Vtp...` and has a size of 59192 bytes. 
    
3. As expected, the Base64 string is about 33% larger than the raw bytes. Running the code below:
    
    ```python
    print(f"Image as Base64 is {(len(image_base64) - len(image_bytes)) / len(image_bytes) * 100:.2f}% larger than as bytes")
    ```
    
    Outputs:
    
    ```
    Image as Base64 is 33.34% larger than as bytes
    ```
    
4. The API call is nearly identical, simply passing the `image_base64` data instead:
    
    ```python
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Part.from_bytes(data=image_base64, mime_type="image/webp"),
            "Tell me what is in this image in one paragraph.",
        ],
    )
    ```
    
5. For public URLs, Gemini has a built-in `url_context` tool that can automatically parse content from a link. Here, we ask it to summarize the original ReAct paper directly from its arXiv URL:
    
    ```python
    response = client.models.generate_content(
        model=MODEL_ID,
        contents="Based on the provided paper as a PDF, tell me how ReAct works: https://arxiv.org/pdf/2210.03629",
        config=types.GenerateContentConfig(tools=[{"url_context": {}}]),
    )
    print(response.text)
    ```
    
    The LLM response provides a detailed summary of how ReAct works:
    
    ```
    ReAct is a novel paradigm for large language models (LLMs) that combines reasoning (Thought) and acting (Action) in an interleaved manner to solve diverse language and decision-making tasks. This approach allows the model to:
    
    *   **Reason to Act:** Generate verbal reasoning traces to induce, track, and update action plans, and handle exceptions.
    *   **Act to Reason:** Interface with and gather additional information from external sources (like knowledge bases or environments) to incorporate into its reasoning.
    
    **How it works:**
    
    Instead of just generating a direct answer ...
    
    This cycle of Thought, Action, and Observation continues until the task is completed.
    ```
    
6. When working with private data lakes, such as Google Cloud Storage (GCS) or Amazon S3, the process is similar, except you have to use the `from_uri` method:
    
    ```python
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Part.from_uri(uri="gs://gemini-images/image_1.jpeg", mime_type="image/webp"),
            "Tell me what is in this image in one paragraph.",
        ],
    )
    ```
    
7. A more advanced and exciting use case for multimodal LLMs is object detection. We can ask the model to identify prominent items in an image and return their bounding box coordinates. To do that, let’s begin by defining a Pydantic schema to ensure the output is structured correctly:
    
    ```python
    from pydantic import BaseModel, Field
    
    class BoundingBox(BaseModel):
        ymin: float
        xmin: float
        ymax: float
        xmax: float
        label: str
    
    class Detections(BaseModel):
        bounding_boxes: list[BoundingBox]
    
    image_bytes, image_size = load_image_as_bytes(
        image_path=Path("images") / "image_1.jpeg", format="WEBP", return_size=True
    )
    
    ```
    
    1. Next, we define the prompt that guides the LLM to detect 2D bounding boxes normalized to a range of 0-1000 to avoid being sensitive to the image shape:
    
    ```python
    prompt = """
    Detect all of the prominent items in the image.
    The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.
    Also, output the label of the object found within the bounding box.
    """
    ```
    
    1. Ultimately, we load the image and call the LLM:
    
    ```python
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Detections,
    )
    
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/webp"),
            prompt,
        ],
        config=config,
    )
    
    detections = cast(Detections, response.parsed)
    ```
    
    The model correctly identifies the "robot" and "kitten" and provides their coordinates. For our example, the image size is (600, 600) and the detected bounding boxes are:
    
    - ymin=1.0 xmin=450.0 ymax=997.0 xmax=1000.0 label='robot'
    - ymin=269.0 xmin=39.0 ymax=782.0 xmax=530.0 label='kitten'
    
    We can then use a helper function to visualize these boxes on the original image.
    
    ```python
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    
    def visualize_detections(detections: Detections, image_path: Path) -> None:
        plt.clf()
        image = PILImage.open(image_path)
        image_array = np.array(image)
        img_height, img_width = image_array.shape[:2]
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(image_array)
    
        for bbox in detections.bounding_boxes:
            xmin = (bbox.xmin / 1000) * img_width
            ymin = (bbox.ymin / 1000) * img_height
            xmax = (bbox.xmax / 1000) * img_width
            ymax = (bbox.ymax / 1000) * img_height
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=3, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
            ax.text(
                xmin, ymin + 5, bbox.label[:15],
                fontsize=12, color="red", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Object Detection Results: {image_path.name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()
    
    visualize_detections(detections, Path("images") / "image_1.jpeg")
    
    ```
    

![Figure 12: The sample image with red bounding boxes drawn around the robot and the kitten, with their corresponding labels.](Lesson%2011%20Multimodal%20Data%20250f9b6f42708088b0e5f3ccb1e43223/image%202.png)

Figure 12: The sample image with red bounding boxes drawn around the robot and the kitten, with their corresponding labels.

**Working with PDFs** follows the same principles. Since we use the same Gemini model and interface, the process is almost identical to what we did for images. We can load a PDF as bytes or Base64 and ask the model to summarize it. Let's use the legendary *"Attention Is All You Need"* paper as an example. 

1. Here is how the first page of the PDF looks:
    
    ```python
    display_image(Path("images") / "attention_is_all_you_need_0.jpeg")
    
    ```
    

![The first page of the "Attention Is All You Need" paper, featuring the title, authors, and abstract.](Lesson%2011%20Multimodal%20Data%20250f9b6f42708088b0e5f3ccb1e43223/attention_is_all_you_need_0.jpeg)

Figure 13: The first page of the "Attention Is All You Need" paper, featuring the title, authors, and abstract

1. First, let’s process the entire PDF as raw bytes:
    
    ```python
    pdf_bytes = (Path("pdfs") / "attention_is_all_you_need_paper.pdf").read_bytes()
    ```
    
2. Now, we call the LLM to summarize the document:
    
    ```python
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            "What is this document about? Provide a brief summary of the main topics.",
        ],
    )
    print(response.text)
    ```
    
    The LLM response provides a summary of the PDF:
    
    ```
    This document introduces the **Transformer**, a novel neural network architecture designed for **sequence transduction tasks** (like machine translation).
    
    Its main topics include:
    
    1.  **Dispensing with Recurrence and Convolutions**: Unlike previous dominant models (RNNs and CNNs), the Transformer relies *solely* on **attention mechanisms**, eliminating the need for sequential computation.
    2.  **Attention Mechanisms**: It details the **Scaled Dot-Product Attention** and **Multi-Head Attention** as its core building blocks, explaining how they allow the model to weigh different parts of the input sequence.
    3.  **Parallelization and Efficiency**: The paper highlights that the Transformer's architecture allows for significantly more parallelization during training, leading to **faster training times** compared to prior models.
    4.  **Superior Performance**: It demonstrates that the Transformer achieves **state-of-the-art results** on machine translation tasks (English-to-German and English-to-French) and generalizes well to other tasks like English constituency parsing.
    ```
    
3. Alternatively, we can process the PDF as Base64 encoded strings:
    
    ```python
    def load_pdf_as_base64(pdf_path: Path) -> str:
        with open(pdf_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    pdf_base64 = load_pdf_as_base64(pdf_path=Path("pdfs") / "attention_is_all_you_need_paper.pdf")
    
    ```
    
    Now, we call the LLM to summarize the document using the Base64 string:
    
    ```python
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            "What is this document about? Provide a brief summary of the main topics.",
            types.Part.from_bytes(data=pdf_base64, mime_type="application/pdf"),
        ],
    )
    print(response.text)
    ```
    
    The LLM response is similar to the one that inputs raw bytes.
    
4. To further emphasize how you can input PDFs to LLMs as images, especially when they contain complex layouts, let's perform object detection on a page from the *"Attention Is All You Need"* paper. Here is the PDF page we will use as an example for detecting the diagram:
    
    ```python
    display_image(Path("images") / "attention_is_all_you_need_1.jpeg")
    
    ```
    

![A page from the "Attention is All You Need" paper with a red bounding box around the Transformer model architecture diagram.](Lesson%2011%20Multimodal%20Data%20250f9b6f42708088b0e5f3ccb1e43223/attention_is_all_you_need_1.jpeg)

Figure 14: A page from the "Attention is All You Need" paper with a red bounding box around the Transformer model architecture diagram.

1. We define the object detection prompt and load the image as bytes:
    
    ```python
    prompt = """
    Detect all the diagrams from the provided image as 2d bounding boxes.
    The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000.
    Also, output the label of the object found within the bounding box.
    """
    
    image_bytes, image_size = load_image_as_bytes(
        image_path=Path("images") / "attention_is_all_you_need_1.jpeg", format="WEBP", return_size=True
    )
    ```
    
2. Now, we call the LLM to detect the diagram from the PDF page as an image:
    
    ```python
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Detections,
    )
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/webp"),
            prompt,
        ],
        config=config,
    )
    detections = cast(Detections, response.parsed)
    ```
    
    The image size is (600, 776), while the detected bounding box is:
    
    ```
    ymin=88.0 xmin=309.0 ymax=515.0 xmax=681.0 label='diagram'
    ```
    
    1. Finally, we visualize the detections:
    
    ```python
    visualize_detections(detections, Path("images") / "attention_is_all_you_need_1.jpeg")
    ```
    

![Figure 15: The retrieved image, which is a page from the "Attention is All You Need" paper showing the Transformer architecture diagram.](Lesson%2011%20Multimodal%20Data%20250f9b6f42708088b0e5f3ccb1e43223/image%203.png)

Figure 15: The retrieved image, which is a page from the "Attention is All You Need" paper showing the Transformer architecture diagram.

These examples show how easily modern LLMs can ingest and reason over multimodal data, such as images and PDFs, thereby making complex, multi-step OCR pipelines obsolete.

## **Foundations of Multimodal RAG**

One of the most impactful applications of multimodal embeddings is in RAG systems. As we discussed in Lesson 10, feeding large amounts of context into an LLM is inefficient. This problem becomes even more evident with large files like high-resolution images or multi-page PDFs. Trying to fit thousands of document pages into a context window leads to high latency, soaring costs, and degraded performance.

A generic multimodal RAG architecture for text and images involves two main pipelines: ingestion and retrieval. For example, let's assume that we want to store images in our vector database and query them using text inputs. To achieve this, **during ingestion**, we use a multimodal embedding model, such as CLIP, to convert our collection of images into vector embeddings. We then store these embeddings in our vector database of choice for efficient searching. As the image embedding is just a vector similar to a text embedding, you are not limited to any particular vector database.

**During retrieval**, we convert a user's text query into an embedding using the same CLIP embedding model. The system then queries the vector database to find the `top-k` image embeddings most similar to the query text embedding, typically using cosine similarity. Since text and image embeddings exist in a shared vector space, this cross-modal search is working effectively. This is the core technology behind image search engines like Google Images or Apple Photos.

![*Figure 16: A generic multimodal RAG architecture for text-to-image retrieval.*](Lesson%2011%20Multimodal%20Data%20250f9b6f42708088b0e5f3ccb1e43223/image%204.png)

*Figure 16: A generic multimodal RAG architecture for text-to-image retrieval.*

For enterprise use cases focused on document retrieval, as of late 2024, every SOTA approach is based on the ColPali-style architecture, making ColPali a fundamental concept to understand how modern multimodal RAG works [[28]](https://arxiv.org/abs/2407.01449). ColPali standardized the architecture for completely bypassing the traditional OCR pipeline that typically involves text extraction, layout detection, chunking, and embedding in multimodal RAG systems, achieving on-par or better results. Instead of extracting text, it treats each document page as an image and processes it directly with multimodal LLMs to understand both textual and visual content simultaneously. As discussed in previous sections, this preserves all the rich visual context, tables, figures, charts, and layout that is lost during text extraction [[24]](https://dev.to/aws/beyond-text-building-intelligent-document-agents-with-vision-language-models-and-colpali-and-oc). The problem it solves is the loss of visual information when documents are converted to text. This approach works exceptionally well for documents with tables, figures, and other complex visual layouts [[25]](https://learnopencv.com/multimodal-rag-with-colpali/).

The **offline indexing**, or the **ingestion pipeline**, converts PDF documents into high-quality images, similar to what we did in previous sections. These images are then processed by the ColPali model, which generates multi-vector embeddings from the document images. These embeddings capture spatial relationships, formatting context, and the interplay between text and visual elements [[26]](https://decodingml.substack.com/p/the-king-of-multi-modal-rag-colpali). Ultimately, the embeddings are stored in a vector database like Qdrant for efficient similarity search [[26]](https://decodingml.substack.com/p/the-king-of-multi-modal-rag-colpali).

Instead of creating a single embedding for an entire page, ColPali generates a "bag-of-embeddings" or "multi-vector" representation, with one embedding for each image patch. This captures fine-grained details within the document [[25]](https://learnopencv.com/multimodal-rag-with-colpali/).

ColPali commonly uses a PaliGemma‑3B backbone that combines a SigLIP vision encoder with a Gemma‑2B language decoder via a multimodal projection, leveraging the **Unified Embedding Decoder Architecture** presented in the multimodal LLM architecture section. The paper also reports a Qwen‑based variant, called ColQwen2, that swaps the backbone to a Qwen2‑VL family model and achieves the strongest results on the ViDoRe benchmark, which is designed to test how well document retrieval systems can handle documents that contain both text and complex visual elements [[27]](https://blog.vespa.ai/retrieval-with-vision-language-models-colpali/), [[28]](https://arxiv.org/html/2407.01449v6).

For the **online query logic**, ColPali uses a late-interaction mechanism called MaxSim. This algorithm computes similarities by comparing each token from the query embedding against all the patch embeddings of a document image to find the maximum similarity. These maximum similarity values are then summed up into a single aggregate similarity score that is used to rank the documents from the most relevant to least [[25]](https://learnopencv.com/multimodal-rag-with-colpali/). Thus, this query engine can be used as a reranker to reduce the context window by keeping only the most relevant retrieved items.

This approach has proven to be considerably faster and more accurate than OCR-based pipelines, outperforming them on complex document retrieval benchmarks [[2]](https://arxiv.org/html/2412.02210v2). It achieves an 81.3% average nDCG@5 score on the ViDoRe benchmark, showcasing its superior performance. ColPali demonstrates improved latency and fewer failure points compared to traditional OCR pipelines, which require text extraction, layout detection, and chunking [[28]](https://arxiv.org/pdf/2407.01449v6).

As innovations happen quickly in the space, as of 2025, the original ColPali method is no longer the state-of-the-art, as it can be seen in the [ViDoRe Benchmark V2](https://huggingface.co/blog/manu/vidore-v2) [36] (an extension of the original benchmark containing more complex retrieval scenarios). Still, what is important to remember is that ColPali’s architecture inspired all future SOTA methods used for document retrieval, making it a fundamental concept required for multimodal RAG.

![Comparison of a standard retrieval pipeline vs. the ColPali architecture.](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F45e112b5-db9c-420c-9947-05e5593b1621_2044x1236.png)

*Figure 17: Comparison of a standard retrieval pipeline vs. the ColPali architecture. (Media from [ColPali Paper](https://arxiv.org/pdf/2407.01449v6))*

Most modern multimodal RAG architectures use ColPali or a derivative of it to build their multimodal RAG application involving visually complex PDFs or images, such as financial reports with charts and tables or technical manuals with diagrams. The official `colpali` implementation can be found on GitHub at `illuin-tech/colpali`. 

Now that we have covered the theory, let us see how this works in practice and build a simple multimodal RAG system from scratch.

## **Implementing Multimodal RAG for Images, PDFs, and Text**

To connect everything we have learned in this lesson and lesson 10 on RAG, we will build a simple multimodal RAG system. For this mini-project, we will populate an in-memory vector database with a mix of images and PDF pages from the *"Attention Is All You Need"* paper treated as images. This demonstrates how to handle diverse visual content in a single retrieval system.

Our simplified RAG system works by generating a textual description for each image and then embedding this description to create a vector representation. These vectors are stored in our in-memory index. When you provide a text query, we embed it and perform a similarity search against the vectors in our index to find the most relevant images.

![*Figure 18: Architecture of our simplified multimodal RAG example.*](Lesson%2011%20Multimodal%20Data%20250f9b6f42708088b0e5f3ccb1e43223/image%205.png)

*Figure 18: Architecture of our simplified multimodal RAG example.*

<aside>
⚠️

**Why do we generate image descriptions? Isn’t that against what we said in this lesson?** Yes, it is, but it’s a necessary workaround to avoid introducing another API just for this mini-project. 

For more context, the Gemini API via the `google-genai` library does not currently support creating embeddings directly from images. To keep this example self-contained, we decided to generate a text description of each image using Gemini and then embed that description using a text embedding model. As we discussed earlier, translating images to text is generally not recommended in production systems because it can lead to information loss. However, once you have access to a multimodal embedding model, you can directly embed the image bytes while the rest of the RAG system remains conceptually the same, as image and text embeddings reside in the same vector space. Popular multimodal embedding models you could easily integrate include Voyage, Cohere, Google Embeddings on Vertex AI, or open-source CLIP models [[10]](https://github.com/openai/CLIP), [[11]](https://huggingface.co/docs/transformers/v4.22.1/en/model_doc/clip), [[29]](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/onnx-pipeline-models-multi-modal-embedding.html), [[30]](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/generate-multi-modal-embeddings-using-clip.html), [[31]](https://docs.weaviate.io/weaviate/model-providers/transformers/embeddings-multimodal).

</aside>

Now, let us get to the code.

1. First, we display the images that we will embed and load into our mocked vector index. This collection includes standard images mixed with pages from the *"Attention Is All You Need"* paper, which we treat as images to demonstrate handling diverse visual content:

```python
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
```

![Figure 19: A grid of images used within our multimodal RAG system example.](Lesson%2011%20Multimodal%20Data%20250f9b6f42708088b0e5f3ccb1e43223/image%206.png)

Figure 19: A grid of images used within our multimodal RAG system example.

1. We define the `create_vector_index` function, which takes a list of image paths, generates a description for each using Gemini, and then creates a text embedding for that description. For this example, we mock the vector index as a simple Python list because we have only a few images. In a real-world application, you would use a dedicated vector database, such as Qdrant or Milvus, which employs efficient indexing algorithms like HNSW to scale to millions of documents.
    
    Here is our implementation using the description workaround:
    
    ```python
    import numpy as np
    from typing import cast
    
    def create_vector_index(image_paths: list[Path]) -> list[dict]:
        vector_index = []
        for image_path in image_paths:
            image_bytes = cast(bytes, load_image_as_bytes(image_path, format="WEBP"))
            image_description = generate_image_description(image_bytes)
            image_embedding = embed_text_with_gemini(image_description)
    
            if image_embedding is not None:
                vector_index.append({
                    "content": image_bytes,
                    "filename": image_path,
                    "description": image_description,
                    "embedding": image_embedding,
                })
        return vector_index
    
    image_paths = list(Path("images").glob("*.jpeg"))
    vector_index = create_vector_index(image_paths)
    ```
    
    After calling the `create_vector_index` function, we successfully create seven embeddings under the `vector_index` variable. The first element in our `vector_index` has keys for `content`, `filename`, `description`, and `embedding`. The `embedding` is a 3072-dimensional vector, and its description begins with "This image is a page from a technical or scientific document...".
    
    In case you start using a text-image embedding model, you would just have to do the following modifications to the code:
    
    ```python
    image_bytes = # ... load image bytes
    # SKIPPED !# image_description = generate_image_description(image_bytes)
    image_embeddings = embed_with_multimodal_model(image_bytes)
    ```
    
    The `create_vector_index` function depends on two other functions that wrap calls to LLMs. More exactly, it depends on the  `generate_image_description` function used to implement our workaround and create detailed descriptions for each image:
    
    ```python
    def generate_image_description(image_bytes: bytes) -> str:
        try:
            img = PILImage.open(io.BytesIO(image_bytes))
            prompt = "Describe this image in detail for semantic search purposes."
            response = client.models.generate_content(model=MODEL_ID, contents=[prompt, img])
            return response.text.strip() if response and response.text else ""
        except Exception:
            return ""
    ```
    
    And the `embed_text_with_gemini` function used to embed the given description of each image using the `gemini-embedding-001` text embedding model from Gemini:
    
    ```python
    def embed_text_with_gemini(content: str) -> np.ndarray | None:
        try:
            result = client.models.embed_content(model="gemini-embedding-001", contents=[content])
            return np.array(result.embeddings[0].values) if result and result.embeddings else None
        except Exception:
            return None
    ```
    
2. Next, we define our `search_multimodal` function. This function takes a text query, embeds it, and then calculates the cosine similarity against all the embeddings in our `vector_index` to find the top `k` results:
    
    ```python
    from sklearn.metrics.pairwise import cosine_similarity
    
    def search_multimodal(query_text: str, vector_index: list[dict], top_k: int = 3) -> list[dict]:
        query_embedding = embed_text_with_gemini(query_text)
        if query_embedding is None:
            return []
    
        embeddings = [doc["embedding"] for doc in vector_index]
        similarities = cosine_similarity([query_embedding], embeddings).flatten()
    
        top_indices = np.argsort(similarities)[::-1][:top_k]
    
        results = []
        for idx in top_indices.tolist():
            results.append({**vector_index[idx], "similarity": similarities[idx]})
    
        return results
    ```
    
3. Now, let us test our multimodal RAG system. We will search for an image containing the architecture of the Transformer network. The system correctly retrieves the page from the paper containing the model diagram:
    
    ```python
    query = "what is the architecture of the transformer neural network?"
    results = search_multimodal(query, vector_index, top_k=1)
    
    if results:
        display_image(Path(results[0]["filename"]))
    ```
    

![The retrieved image, which is a page from the "Attention is All You Need" paper showing the Transformer architecture diagram.](Lesson%2011%20Multimodal%20Data%20250f9b6f42708088b0e5f3ccb1e43223/attention_is_all_you_need_1%201.jpeg)

Figure 20: The retrieved image for the query about the Transformer architecture.

1. Let us try another query: *"a kitten with a robot."* Again, the system finds the correct image:
    
    ```python
    query = "a kitten with a robot"
    results = search_multimodal(query, vector_index, top_k=1)
    
    if results:
        display_image(Path(results[0]["filename"]))
    ```
    

![The retrieved image of a kitten sitting on a robot's arm.](Lesson%2011%20Multimodal%20Data%20250f9b6f42708088b0e5f3ccb1e43223/image_1%201.jpeg)

Figure 21: The retrieved image for the query "a kitten with a robot.

This simple implementation demonstrates the power of multimodal RAG. Because we normalized both standard images and PDF pages to images, we used the same unified vector index for searching both. You could extend this even further by sampling video footage or translating audio data to spectrograms.

## **Building Multimodal AI Agents**

Now, we will integrate our RAG system into a ReAct agent, bringing together the core skills from Part 1 of this course. We can add multimodal capabilities to AI agents in two primary ways: by enabling the model to handle multimodal inputs and outputs or by using specialized multimodal tools that connect to external systems that process multimodal data. 

Let’s illustrate both methods within a single use case. We will build a ReAct agent that uses our `search_multimodal` function defined in the previous section as a tool. The agent's task is to answer a question that requires finding a specific image and reasoning about its content. This creates a complete multimodal agentic RAG workflow.

![*Figure 22: The workflow of our multimodal ReAct agent.*](Lesson%2011%20Multimodal%20Data%20250f9b6f42708088b0e5f3ccb1e43223/image%207.png)

*Figure 22: The workflow of our multimodal ReAct agent.*

Let us implement this.

1. First, we wrap our `search_multimodal` function into a tool that the agent can call using LangGraph’s `@tool` decorator. The tool's output will include both the image description and the image itself, which the multimodal LLM can process:
    
    ```python
    from langchain_core.tools import tool
    from typing import Any
    
    @tool
    def multimodal_search_tool(query: str) -> dict[str, Any]:
        """
        Search through a collection of images to find relevant content based on a text query.
        """
        results = search_multimodal(query, vector_index, top_k=1)
    
        if not results:
            return {"role": "tool_result", "content": "No relevant content found."}
    
        result = results[0]
        content = [
            {"type": "text", "text": f"Image description: {result['description']}"},
            types.Part.from_bytes(data=result["content"], mime_type="image/jpeg"),
        ]
        return {"role": "tool_result", "content": content}
    
    ```
    
2. Next, we use LangGraph to create the ReAct agent. We will cover LangGraph in detail in Part 2 of the course, but for now, you can think of it as a quick way to define ReAct agents:
    
    ```python
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langgraph.prebuilt import create_react_agent
    
    def build_react_agent() -> Any:
        tools = [multimodal_search_tool]
        system_prompt = "You are a helpful AI assistant that can search through images and text to answer questions."
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", convert_system_message_to_human=True)
        agent_executor = create_react_agent(llm, tools, messages_modifier=system_prompt)
        return agent_executor
    
    react_agent = build_react_agent()
    
    ```
    
    The ReAct agent looks like this:
    

![Figure 23: A high-level view of the LangGraph ReAct agent architecture.](Lesson%2011%20Multimodal%20Data%20250f9b6f42708088b0e5f3ccb1e43223/image%208.png)

Figure 23: A high-level view of the LangGraph ReAct agent architecture.

1. Finally, we ask the agent a question: *"What color is my kitten?"*. The agent correctly reasons that it needs to search for an image of a kitten, calls our tool, receives the image, and then analyzes it to provide the final answer:
    
    ```python
    test_question = "what color is my kitten?"
    
    for chunk in react_agent.stream({"messages": [("user", test_question)]}):
        print(chunk)
        print("---")
    ```
    
    The agent's thought process is transparent. It first receives your query, then decides to call the `multimodal_search_tool` with "kitten" as the query. After the tool executes and retrieves the image and its description, the agent processes this observation. Finally, it formulates the answer based on the visual evidence.
    
    Here is a simplified representation of the agent's output:
    
    ```
    {'messages': [HumanMessage(content='what color is my kitten?')]}
    ---
    {'messages': [AIMessage(content='', tool_calls=[{'name': 'multimodal_search_tool', 'args': {'query': 'kitten'}}])]}
    ---
    {'messages': [ToolMessage(content=[{'type': 'text', 'text': 'Image description: ...'}, {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,...'}}], name='multimodal_search_tool')]}
    ---
    {'messages': [AIMessage(content='The kitten in the image is a fluffy grey tabby.')]}
    ---
    
    ```
    
    The final answer is "The kitten in the image is a fluffy grey tabby."
    

![The retrieved image of a kitten sitting on a robot's arm.](Lesson%2011%20Multimodal%20Data%20250f9b6f42708088b0e5f3ccb1e43223/image_1%202.jpeg)

Figure 24: The image retrieved by the agent to answer the question.

This example combines structured outputs, tools, ReAct, RAG, and multimodal processing to create a functional multimodal agentic RAG proof-of-concept.

## **Conclusion**

This lesson completes our journey through the fundamentals of AI engineering in Part 1. We have seen how to move beyond text-only systems and build powerful multimodal agents that can see and interpret the world more like humans do. By combining concepts like structured outputs, tools, ReAct, and RAG, we constructed a proof-of-concept that can reason about visual data.

These skills will be crucial for the capstone project in Part 2, where we will develop a multi-agent system capable of handling multiple data formats. The research agent will need to process PDFs, videos, and images to extract information or rank relevant resources. Then it will be passed to the writer agent, which will need to process all this information to properly extract key facts from the research when writing.

## **References**

- [1] [The Hidden Ceiling of OCR in RAG](https://www.mixedbread.com/blog/the-hidden-ceiling)
- [2] [CC-OCR: A Comprehensive Benchmark for Chinese Commercial OCR](https://arxiv.org/html/2412.02210v2)
- [3] [OCR vs VLM-OCR: A Naive Benchmarking of Accuracy for Scanned Documents](https://www.dataunboxed.io/blog/ocr-vs-vlm-ocr-naive-benchmarking-accuracy-for-scanned-documents)
- [4] [Complex Document Recognition: OCR Doesn’t Work and Here’s How You Fix It](https://hackernoon.com/complex-document-recognition-ocr-doesnt-work-and-heres-how-you-fix-it)
- [5] [Benchmarking QA on Complex Industrial PDFs](https://www.3rdaiautomation.com/blog/benchmarking-QA-on-complex-indestrial-PDFs)
- [6] [Understanding Multimodal LLMs](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms)
- [7] [What are Vision Language Models (VLMs)?](https://www.nvidia.com/en-us/glossary/vision-language-models/)
- [8] [Multimodal LLMs: Architectures, Techniques, and Use Cases](https://blog.premai.io/multimodal-llms-architecture-techniques-and-use-cases/)
- [9] [A Comprehensive Guide to Multimodal LLMs and How They Work](https://www.ionio.ai/blog/a-comprehensive-guide-to-multimodal-llms-and-how-they-work)
- [10] [OpenAI CLIP GitHub](https://github.com/openai/CLIP)
- [11] [Hugging Face Transformers: CLIP](https://huggingface.co/docs/transformers/v4.22.1/en/model_doc/clip)
- [12] [Multimodal Embeddings: An Introduction](https://towardsdatascience.com/multimodal-embeddings-an-introduction-5dc36975966f)
- [12b] [Multimodal Embeddings: An Introduction - Video](https://www.youtube.com/watch?v=YOvxh_ma5qE)
- [13] [Multi-modal ML with OpenAI's CLIP](https://www.pinecone.io/learn/series/image-search/clip/)
- [14] [Google Gemini: The next generation of AI](https://blog.google/technology/ai/google-gemini-ai/)
- [15] [Building high-quality multimodal data pipelines for LLMs](https://www.turing.com/resources/building-high-quality-multimodal-data-pipelines-for-llms)
- [16] [Efficient Multi-LLM Inference: A Survey](https://arxiv.org/html/2506.06579v1)
- [17] [Towards AI: Multimodal Lesson Image 1](https://raw.githubusercontent.com/towardsai/course-ai-agents/dev/lessons/11_multimodal/images/image_1.jpeg)
- [18] [Towards AI: Multimodal Lesson Image 2](https://raw.githubusercontent.com/towardsai/course-ai-agents/dev/lessons/11_multimodal/images/image_2.jpeg)
- [19] [Towards AI: Multimodal Lesson Object Detection 1](https://raw.githubusercontent.com/towardsai/course-ai-agents/dev/lessons/11_multimodal/images/object_detection_1.png)
- [20] [Towards AI: Multimodal Lesson Attention Paper Page 0](https://raw.githubusercontent.com/towardsai/course-ai-agents/dev/lessons/11_multimodal/images/attention_is_all_you_need_0.jpeg)
- [21] [Towards AI: Multimodal Lesson Attention Paper Page 1](https://raw.githubusercontent.com/towardsai/course-ai-agents/dev/lessons/11_multimodal/images/attention_is_all_you_need_1.jpeg)
- [22] [Towards AI: Multimodal Lesson Object Detection 2](https://raw.githubusercontent.com/towardsai/course-ai-agents/dev/lessons/11_multimodal/images/object_detection_2.png)
- [23] [Use ColPali with Milvus](https://milvus.io/docs/use_ColPali_with_milvus.md)
- [24] [Beyond Text: Building Intelligent Document Agents with Vision Language Models and ColPali](https://dev.to/aws/beyond-text-building-intelligent-document-agents-with-vision-language-models-and-colpali-and-oc)
- [25] [Multimodal RAG with ColPali](https://learnopencv.com/multimodal-rag-with-colpali/)
- [26] [The King of Multi-Modal RAG: ColPali](https://decodingml.substack.com/p/the-king-of-multi-modal-rag-colpali)
- [27] [Retrieval with Vision Language Models: ColPali](https://blog.vespa.ai/retrieval-with-vision-language-models-colpali/)
- [28] [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/pdf/2407.01449v6)
- [29] [ONNX Pipeline Models for Multi-Modal Embedding](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/onnx-pipeline-models-multi-modal-embedding.html)
- [30] [Generate Multi-Modal Embeddings Using CLIP](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/generate-multi-modal-embeddings-using-clip.html)
- [31] [Weaviate: Multimodal Embeddings](https://docs.weaviate.io/weaviate/model-providers/transformers/embeddings-multimodal)
- [32] [LangGraph ReAct Agent Architecture](https://storage.googleapis.com/gweb-cloudblog-publish/images/1_Ai6ddoG.max-1800x1800.png)
- [33] [Build multimodal agents using Gemini, LangChain, and LangGraph](https://cloud.google.com/blog/products/ai-machine-learning/build-multimodal-agents-using-gemini-langchain-and-langgraph)
- [34] [Multimodal RAG with Colpali, Milvus and VLMs](https://huggingface.co/blog/saumitras/colpali-milvus-multimodal-rag)
- [35] [What are some real-world applications of multimodal AI?](https://milvus.io/ai-quick-reference/what-are-some-realworld-applications-of-multimodal-ai)
- [36] [ViDoRe Benchmark V2: Raising the Bar for Visual Retrieval](https://huggingface.co/blog/manu/vidore-v2)