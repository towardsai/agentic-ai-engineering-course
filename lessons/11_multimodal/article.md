# AI Agents: See, Read, Understand
### Unlock AI's full potential beyond text

## Introduction: AI Agents Need to See, Read, and Understand Our World

To build genuinely useful AI systems, they must operate in our world, not just a world of text files. Human information is messy and diverse. It comes as images, audio, videos, and complex documents like financial reports with charts or technical manuals packed with diagrams. Text-only Large Language Models (LLMs) are blind to this reality. They cannot read a graph, interpret a photo, or understand a PDF's layout.

This is a significant roadblock. How can an AI agent assist with financial analysis if it cannot see charts in a 10-Q report? How can it help a doctor if it cannot process a diagnostic scan? It cannot. To build production-grade AI that performs meaningful tasks, we must equip our systems to see, read, and understand multiple data formats.

This article is a hands-on, no-fluff guide to making that happen. We will explore how to use multimodal LLMs, embedding models, and Retrieval-Augmented Generation (RAG) to build advanced AI agents that process and reason about the world as we do. These skills are no longer optional; they are essential for any engineer serious about shipping AI applications that work.

## The Need for Multimodal AI

If we want to build truly effective AI agents, they must evolve beyond text. In almost every industry, visual formats contain critical information that text-only models simply cannot interpret. This is not a niche problem; it is a fundamental limitation preventing AI from tackling a huge range of real-world tasks.

Consider how a financial analyst operates. A company’s quarterly report is more than just text; it is a collection of tables, charts, and figures that tell a complete story. A text-only model might extract words, but it will miss the upward trend in a revenue graph or a critical outlier in a data table.

Similarly, in medicine, a patient’s file often includes X-rays, MRI scans, and handwritten doctor’s notes. An AI assistant reading only typed text misses most of the picture [1](https://milvus.io/ai-quick-reference/what-are-some-realworld-applications-of-multimodal-ai).

This need extends to countless other domains. Imagine a research assistant needing to understand diagrams in scientific papers, or a customer service bot resolving issues faster by seeing a photo of a broken product. Technical agents also interpret schematics and engineering drawings. In all these scenarios, multimodality is not a "nice-to-have" feature. It is a core requirement for building AI agents that perform complex, valuable work in the real world.
![Figure 1: An illustration of a multimodal LLM that can accept different input modalities (audio, text, images, and videos)
 and returns text as the output modality.](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0d76dab1-362f-45b6-9b12-a12ac131edc5_1600x944.png)

## Why Traditional Document Processing Is a Losing Battle

For years, the standard approach to digitizing complex documents has been a clunky, multi-step pipeline relying on Optical Character Recognition (OCR). This typical workflow involves scanning a document, using OCR to extract text, running a layout detection model to identify elements like tables and paragraphs, and then chunking the extracted text for indexing and search [2](https://hackernoon.com/complex-document-recognition-ocr-does-not-work-and-heres-how-you-fix-it), [3](https://blog.roboflow.com/what-is-optical-character-recognition-ocr/). While this was the best we could do for a while, we are now in a losing battle with this approach.

This pipeline is inherently brittle, and you will quickly find its limitations. Even advanced OCR engines struggle with real-world messiness, failing spectacularly with handwritten notes, poor-quality scans, or unconventional fonts. For instance, OCR tools often struggle to distinguish between similar-looking characters like '3' and '8' or 'O' and 'D' [2](https://hackernoon.com/complex-document-recognition-ocr-does-not-work-and-heres-how-you-fix-it). Traditional methods also face significant challenges with complex layouts, such as nested tables or multi-column structures [3](https://blog.roboflow.com/what-is-optical-character-recognition-ocr/). Studies comparing models like GPT-4 Vision against older architectures like LLaVA confirm this: while newer models handle handwriting with minor errors, older systems are often unreliable and struggle with rotated or overlapped text [4](https://encord.com/blog/gpt-vision-vs-llava/).

The bigger problem is that errors compound. A mistake in the OCR stage corrupts the text, and a failure in layout detection loses crucial context, like which text belongs to which table. By the time your data reaches the embedding model, it is often a garbled mess, stripped of its original visual structure. This multi-step process is not only slow and error-prone but also fundamentally flawed because it discards the rich visual information we humans use to understand documents. The modern approach, which we will explore, interprets the document image as a whole, preserving layout and visual cues from the start.

## How Multimodal LLMs Work: A Look Under the Hood

To understand how AI can "see," we need to examine the architecture of multimodal LLMs. These models process information from different modalities, like images and text, and generate a coherent output. While a few architectural patterns exist, the most common and successful approach today uses a "late interaction" mechanism, bringing different data types together at the input stage [5](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms).

Let us use a text-image model as a concrete example. This system has three main components: a vision encoder, a projection layer, and a standard language model.

1.  **Vision Encoder**: The vision encoder breaks down the input image into a grid of smaller patches. A vision model, often a Vision Transformer (ViT), processes these patches. The vision encoder then converts each patch into a numerical representation, or an embedding, that captures its visual features [5](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms).

2.  **Projection Layer**: Image patch embeddings and text token embeddings from the language model do not initially "speak the same language"; they have different dimensions. The projection layer, sometimes called an adapter or connector, is a simple neural network layer. It rescales the image embeddings to match the dimensions of the text embeddings. This alignment is crucial for the next step [5](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms).

3.  **Language Model**: Once image and text embeddings are in the same dimensional space, the LLM processes them together. A standard LLM, like GPT or Llama, takes this combined sequence of text and "image tokens" to understand the context and generate a response. Popular models like LLaVA (Large Language and Vision Assistant), Llama 3.2, and Qwen2-VL build on this principle [5](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms).
![Figure 2: Illustration of the unified embedding decoder architecture, which is an unmodified decoder-style LLM that receives inputs consisting of image token and text token embeddings.](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa219f185-211b-4569-9398-2e080e2c5619_1166x1400.png)


We can extend this architecture to other modalities like PDFs, audio, or video. This involves using specialized encoders for each data type. The core idea remains the same: convert non-text data into embeddings that the LLM can align and process. These models excel at tasks like image captioning, visual question answering, and document understanding [1](https://milvus.io/ai-quick-reference/what-are-some-realworld-applications-of-multimodal-ai). Training typically involves pretraining the vision encoder and then fine-tuning the entire model, or parts of it, on multimodal datasets [5](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms).

It is important to differentiate these models from image *generation* models like Midjourney or Stable Diffusion. Those are typically diffusion models, a different family of AI that excels at creating images from noise based on a text prompt [6](https://zapier.com/blog/best-ai-image-generator/). While some multimodal LLMs, such as GPT-4o, can also generate images, their primary strength in an agentic workflow is understanding and reasoning about multimodal inputs, not just creating them. For an agent, we can simply integrate a diffusion model as another tool it can call when needed.

## Practical Guide: Using Multimodal LLMs for Images and PDFs

Theory is one thing, but practical application is what matters. We will walk through how to use a multimodal LLM in the real world with Google's Gemini Application Programming Interface (API). We will cover the three primary methods for feeding images to an LLM: raw bytes, Base64 encoding, and Uniform Resource Locators (URLs). Understanding these methods is key because the best approach depends entirely on your deployment scenario.

First, let us set up our environment and initialize the Gemini client. We will use the `gemini-2.5-flash` model, which is fast and cost-effective.
```python
#
# This is a cell from a Jupyter Notebook.
# It contains the Python code that is being described in the article.
#
import base64
import io
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types
from IPython.display import Image as IPythonImage
from PIL import Image as PILImage

from lessons.utils import pretty_print
client = genai.Client()
MODEL_ID = "gemini-2.5-flash"
```

### 1. Processing Images as Raw Bytes
Passing raw image bytes is the most direct method and works well for quick, one-off API calls. However, storing raw bytes can be tricky because they can get corrupted if not handled correctly, such as encoding issues when saving to a database.

Here is a helper function to load an image and convert it to bytes.
```python
#
# This is a cell from a Jupyter Notebook.
# It contains the Python code that is being described in the article.
#
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
```
Now, pass an image to Gemini and ask for a description.
```python
#
# This is a cell from a Jupyter Notebook.
# It contains the Python code that is being described in the article.
#
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
```
It outputs:
```text
 [93m----------------------------------------- Image 1 Caption ----------------------------------------- [0m

 The image depicts a striking juxtaposition of artificial intelligence and natural life, featuring a large, 
 heavily armored robot with glowing red eyes in what appears to be an industrial or workshop setting. Perched 
 playfully on the robot's left forearm and shoulder is a small, adorable grey tabby kitten, looking curiously 
 towards the robot's head. The robot's metallic body is intricately detailed with circuit-like patterns on its 
 head and glowing red indicators on its chest, showcasing a powerful and advanced design, while the soft, 
 fluffy kitten provides a stark and endearing contrast against the machine's robust frame.

 [93m---------------------------------------------------------------------------------------------------- [0m
```
Using the same approach, you can easily pass multiple images simultaneously.
```python
#
# This is a cell from a Jupyter Notebook.
# It contains the Python code that is being described in the article.
#
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
```
It outputs:
```text
 [93m------------------------------------ Differences between images ------------------------------------ [0m

 The primary difference between the two images lies in the nature of the interaction between the animals and 
 robots, as well as their respective environments. In the first image, a small, grey kitten appears curious 
 and playful as it stands on the arm of a large, grey, somewhat clunky robot, suggesting a peaceful, almost 
 companionable moment set in a well-lit, industrial-like indoor space. Conversely, the second image depicts 
 a tense and confrontational scene where a large, fluffy white dog is aggressively barking at a sleek, black, 
 humanoid robot, with both subjects poised for a fight in a dimly lit, trash-strewn urban alleyway.

 [93m---------------------------------------------------------------------------------------------------- [0m
```

### 2. Processing Images as Base64 Strings
Base64 is a method for encoding binary data, like images, into a string format. This is extremely useful for storing images in databases or transmitting them in JSON payloads without corruption. It is a preferred method for on-premise deployments or when you need strict data locality.
```python
#
# This is a cell from a Jupyter Notebook.
# It contains the Python code that is being described in the article.
#
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
```
The call to the Gemini API is nearly identical, just with a different `Part` type.
```python
#
# This is a cell from a Jupyter Notebook.
# It contains the Python code that is being described in the article.
#
image_base64 = load_image_as_base64(image_path=Path("images") / "image_1.jpeg", format="WEBP")

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        types.Part.from_bytes(data=image_base64, mime_type="image/webp"),
        "Tell me what is in this image in one paragraph.",
    ],
)
response.text
```

### 3. Processing Images as URLs
Using URLs is the most efficient method for cloud-native applications, especially when your images are stored in a data lake like Amazon Web Services Simple Storage Service (AWS S3) or Google Cloud Storage (GCS). The LLM can download the media directly, which minimizes Input/Output (I/O) bottlenecks in your application. While Gemini currently works best with GCS URLs, the principle applies to any accessible web link. URL-based approaches excel in cloud-native environments where content is already distributed and accessible.

A call using a URL would look like this:
```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        types.Part.from_uri(uri="gs://gemini-images/image_1.jpeg", mime_type="image/webp"),
        "Tell me what is in this image in one paragraph.",
    ],
)
```

### 4. Object Detection with LLMs
As a more exciting example, let us perform object detection with multimodal LLMs. We will define a Pydantic model to structure the output, ensuring the LLM returns bounding box coordinates and labels in a consistent format.
```python
#
# This is a cell from a Jupyter Notebook.
# It contains the Python code that is being described in the article.
#
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
```
It outputs:
```text
Image size:  600 600
ymin=163 xmin=306 ymax=598 xmax=600 label='robot'
ymin=163 xmin=20 ymax=475 xmax=321 label='kitten'
```

### 5. Working with PDFs
You can handle PDFs in much the same way as images, either as raw bytes or Base64 strings. This allows you to pass entire documents to the LLM for summarization or analysis.

Here is how you would pass a PDF as raw bytes:
```python
#
# This is a cell from a Jupyter Notebook.
# It contains the Python code that is being described in the article.
#
pdf_bytes = (Path("pdfs") / "decoding_ml_article.pdf").read_bytes()

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
        "What is this document about? Provide a brief summary of the main topics.",
    ],
)
pretty_print.wrapped(response.text, title="PDF Summary (as bytes)")
```
It outputs:
```text
 [93m-------------------------------------- PDF Summary (as bytes) -------------------------------------- [0m

 This document provides a curated list of five (plus a bonus) recommended books for individuals looking to 
 build and ship AI products, particularly those involving Large Language Models (LLMs) and agentic systems, 
 in 2025.
 
 The main topics covered by these recommended books include:
 *   **Designing Machine Learning Systems:** Fundamentals of building production-grade ML systems, MLOps, and 
 infrastructure.
 *   **Prompt Engineering:** Techniques for effectively engineering prompts for LLMs to ensure flexibility, 
 scalability, and optimal model performance.
 *   **AI Engineering:** Broader concepts like RAG (Retrieval Augmented Generation), building agentic systems, 
 and LLMOps (observability, user feedback).
 *   **Building LLMs for Production:** Hands-on implementation of LLM applications, covering core algorithms, 
 RAG techniques (including GraphRAG), agents, fine-tuning, and deployment using frameworks like LangChain and 
 LlamaIndex.
 *   **LLMs in Production:** Optimization of LLMs, data engineering for LLM apps, serving LLMs at scale, and 
 infrastructure considerations for deployment.
 *   **LLM Engineer's Handbook (Bonus):** The author's own book, emphasizing end-to-end LLM RAG application 
 development, including architecting, data collection, fine-tuning, evaluation, deployment, and scaling.
 
 In essence, a guide to essential reading for anyone involved in the practical development and deployment of 
 AI/ML systems, with a strong focus on LLMs.

 [93m---------------------------------------------------------------------------------------------------- [0m
```
Alternatively, you can pass PDFs as Base64 encoded strings.
```python
#
# This is a cell from a Jupyter Notebook.
# It contains the Python code that is being described in the article.
#
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
```
It outputs:
```text
 [93m------------------------------------- PDF Summary (as base64) ------------------------------------- [0m

 This document is an article titled "5 Books to Ship AI Products in 2025" by Paul Iusztin, published on 
 July 03, 2025. It provides a curated list of five (plus a bonus) book recommendations for individuals 
 interested in AI Engineering, with a specific focus on building and deploying Large Language Model (LLM) 
 and agentic systems.

 **Main Topics:**

 *   **AI Engineering Foundations:** General principles for designing and building production-grade machine 
 learning and AI systems.
 *   **Prompt Engineering for LLMs:** Techniques for effectively creating and optimizing prompts for large 
 language models to enhance their performance and scalability.
 *   **Building Agentic Systems:** Concepts and methodologies for developing AI agents, including aspects 
 like RAG (Retrieval-Augmented Generation), guardrails, caching, and memory management.
 *   **LLM Productionization:** Practical guidance on implementing, optimizing, and deploying LLMs in 
 real-world production environments, covering topics like data engineering, fine-tuning, scaling, and 
 infrastructure considerations.
 *   **LLMOps (MLOps for LLMs):** Operational aspects of managing the lifecycle of LLM applications, 
 including monitoring, data versioning, and CI/CD pipelines.
 *   **End-to-End LLM Project Development:** Comprehensive approaches to architecting and building complete 
 LLM and RAG (Retrieval-Augmented Generation) applications from concept to deployment.

 [93m---------------------------------------------------------------------------------------------------- [0m
```
These examples show just how flexible multimodal LLMs are. By mastering these input methods, you can build robust systems that handle a wide variety of data formats and deployment requirements.

## Foundations of Multimodal Embedding Models

Multimodal LLMs excel at direct reasoning, but their true power in agentic systems often comes from RAG. To build a multimodal RAG system, we first need to understand multimodal embedding models. These models are the engine that powers cross-modal search, allowing us to use a text query to find a relevant image, or vice-versa.

OpenAI's CLIP (Contrastive Language-Image Pre-Training) serves as a foundational example [7](https://www.pinecone.io/learn/series/image-search/clip/). The core idea behind CLIP is to create a shared vector space where both images and text can be represented. In this space, an image of a dog and the text "a photo of a dog" have very similar vector representations, meaning they are located close to each other [8](https://towardsdatascience.com/multimodal-embeddings-an-introduction-5dc36975966f).

We achieve this through a process called **contrastive learning**. The model trains on a massive dataset of image-text pairs from the internet. During training, it learns to maximize the similarity between the embeddings of correct image-text pairs (positive pairs) while minimizing the similarity between incorrect pairs (negative pairs) [7](https://www.pinecone.io/learn/series/image-search/clip/). This process forces the model to learn a rich, shared understanding of concepts across both modalities. The architecture uses two separate encoders—one for images, typically a ViT, and one for text, a Transformer. These encoders project their outputs into this common embedding space.
![Figure 3: Similar text and images will be encoded into a similar vector space. Dissimilar text and images do not share a similar vector space.](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fa54a2f1fa0aeac03748c09df0fdfbb42aadc96b7-2430x1278.png&w=3840&q=75)


This technology forms the backbone of multimodal RAG. You can embed a library of images, and then use a text query to find the most semantically relevant images by comparing the query's text embedding to the images' embeddings. This concept extends beyond images and text to audio and video, enabling powerful search capabilities across all data types. Many providers now offer powerful multimodal embedding models, including Google's Gemini family, Cohere, and Voyage AI [9](https://www.edenai.co/post/best-multimodal-embeddings-apis).

## A Better Approach to RAG: The ColPali Architecture

Standard RAG pipelines for documents often rely on the same brittle OCR-based foundation we discussed earlier. These traditional pipelines extract text, chunk it, embed it, and then attempt retrieval. This approach completely ignores a document's visual layout, which is often critical for understanding tables, charts, and figures. This is where modern architectures like ColPali come in, representing a paradigm shift in how we handle document retrieval.

ColPali, developed by researchers at Illuin Technology, entirely bypasses the fragile OCR pipeline [10](https://arxiv.org/html/2407.01449v6). Instead of extracting text, it treats each document page as an image. This simple yet powerful idea means ColPali preserves all the visual and spatial information that traditional methods discard. It understands that a number in a table cell holds a different meaning than a number in a paragraph because it can literally "see" the table's grid lines [10](https://arxiv.org/html/2407.01449v6), [11](https://blog.vespa.ai/Transforming-the-Future-of-Information-Retrieval-with-ColPali/).

ColPali builds its architecture on a Vision Language Model (VLM), such as PaliGemma, and uses a technique inspired by ColBERT's late-interaction mechanism. Here is how it works:

*   **Offline Indexing**: Instead of chunking text, ColPali divides each document page image into a grid of patches, typically a 32x32 grid resulting in 1,024 patches. Each patch is then encoded into a 128-dimensional vector, creating a "bag-of-embeddings" for the page [12](https://qdrant.tech/blog/qdrant-colpali/), [13](https://learnopencv.com/multimodal-rag-with-colpali/). This means a single page is represented by multiple vectors, each capturing a small region of the document.
    *   💡 This process is much faster than a full OCR pipeline, with reported indexing speeds of 0.37 seconds per page compared to 7.22 seconds for typical PDF parsers [13](https://learnopencv.com/multimodal-rag-with-colpali/).
*   **Online Querying**: When you submit a text query, ColPali also breaks down the query into token-level embeddings. The system then uses a **late-interaction** mechanism, specifically MaxSim, to compute the similarity between each query token and all the document patches [10](https://arxiv.org/html/2407.01449v6), [12](https://qdrant.tech/blog/qdrant-colpali/). MaxSim finds the most similar word or patch in the document for each query word and sums these maximum similarities to provide a final score [14](https://milvus.io/docs/use_ColPali_with_milvus.md). This fine-grained matching allows ColPali to pinpoint the exact regions of a document most relevant to your query, considering both text and layout [10](https://arxiv.org/html/2407.01449v6).
![Figure 4: Offline document indexing with ColPali is much simpler and faster compared to standard retrieval methods.](https://arxiv.org/html/2407.01449v6/x1.png)


This approach is superior for visually rich documents. For financial reports, technical manuals, or scientific papers, where tables and diagrams are essential, ColPali retrieves information with far greater accuracy and context than text-only methods. It has shown superior performance on benchmarks like ViDoRe, which specifically test for visually-rich document retrieval [10](https://arxiv.org/html/2407.01449v6). ColPali also offers lower query latency compared to traditional methods and can scale by adjusting the number of image patches, allowing a trade-off between efficiency and retrieval quality [10](https://arxiv.org/html/2407.01449v6).

## Code-Along: Implementing a Simple Multimodal RAG System

Now, let us combine these concepts into a practical multimodal RAG exercise. We will build a simple system that populates an in-memory vector store with images and then queries it using text. This will serve as a conceptual demonstration of how multimodal search works.

💡 For this example, we will use Gemini to generate a text description for each image and then embed that description. This is a workaround because the publicly available Gemini API does not yet support direct image embedding. In a production system with a true multimodal embedding model, such as those from Voyage, Cohere, or Google's models on Vertex AI, you would embed the image directly. The rest of the RAG pipeline remains the same.

First, we define a function to generate an image description using Gemini's vision capabilities.
```python
#
# This is a cell from a Jupyter Notebook.
# It contains the Python code that is being described in the article.
#
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
```
Next, we define a function to embed text using Gemini's text embedding model.
```python
#
# This is a cell from a Jupyter Notebook.
# It contains the Python code that is being described in the article.
#
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
```
Now, we create a function to process a list of images, generate descriptions, create embeddings, and store everything in a list that will act as our vector database.
```python
#
# This is a cell from a Jupyter Notebook.
# It contains the Python code that is being described in the article.
#
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
```
With our "database" populated, we create a search function. It takes a text query, embeds it, and then uses cosine similarity to find the most relevant images from our collection.
```python
#
# This is a cell from a Jupyter Notebook.
# It contains the Python code that is being described in the article.
#
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
```
Let us test our search function with a query.
```python
#
# This is a cell from a Jupyter Notebook.
# It contains the Python code that is being described in the article.
#
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
```
It outputs:
```text

🔍 Embedding query: 'two robots fighting'
✅ Query embedded successfully
 [93m----------------------------- Results for query = two robots fighting ----------------------------- [0m
 Similarity 0.757
 [93m---------------------------------------------------------------------------------------------------- [0m
 Filename images/image_4.jpeg
 [93m---------------------------------------------------------------------------------------------------- [0m
 Description `This detailed description is designed for semantic search, covering various visual elements to 
 help users find this image through text queries.
 
 **Overall Description:**
 This image depicts a dynamic, high-impact futuristic battle between two distinct humanoid robots or mechs 
 within a dark, high-tech, enclosed arena. The scene captures the climactic moment of a powerful punch, with 
 sparks and debris erupting, conveying intense action and destruction.
 
 **Objects:**
 
 *   **Left Robot:**
     *   **Appearance:** Sleek, agile, and humanoid in form, with highly polished, reflective metallic silver 
 or chrome armor. It has a streamlined design.
     *   **Lighting/Glow:** Features prominent electric blue glowing lines and accents across its body, 
 including a bright blue visor or faceplate, chest, and arms, suggesting energy conduits or internal power.
     *   **Pose:** Frozen in the act of delivering a powerful right-hand punch directly into the chest or 
 shoulder area of the opposing robot. Its ...`
 [93m---------------------------------------------------------------------------------------------------- [0m
```
![Figure 5: Retrieved image for the query 'two robots fighting'](https://github.com/towardsai/course-ai-agents/raw/main/lessons/11_multimodal/images/image_4.jpeg)


This example, while simplified, demonstrates the core logic of a multimodal RAG system. Remember, this is not a full ColPali implementation, as we are not patching the images or using a late-interaction reranker. For those interested in a production-grade solution, the official ColPali implementation can be found on GitHub at `illuin-tech/colpali` [10](https://arxiv.org/html/2407.01449v6).

## Conclusion

We journeyed from the fundamental need for multimodal AI to building a functional agent that can "see" and reason about images. The key takeaway is that the future of AI is not confined to text. Real-world applications demand systems that can understand the rich, varied data formats that we humans use every day.

We have seen that traditional document processing with OCR is a brittle and outdated approach, often destroying the very visual context needed for comprehension. The paradigm shift offered by models like ColPali, which treat documents as images, preserves this vital information and provides a more robust foundation for retrieval.

Through practical, hands-on examples, we covered how to use multimodal LLMs like Gemini and how to build a conceptual multimodal RAG system. Mastering the manipulation of multimodal data is no longer a niche specialty; it is a foundational skill for any engineer looking to build powerful, production-grade AI applications that can truly operate in our world.

## References

- [1] [What are some real-world applications of multimodal AI?](https://milvus.io/ai-quick-reference/what-are-some-realworld-applications-of-multimodal-ai)
- [2] [Complex Document Recognition: OCR Doesn’t Work and Here’s How You Fix It](https://hackernoon.com/complex-document-recognition-ocr-doesnt-work-and-heres-how-you-fix-it)
- [3] [What Is Optical Character Recognition (OCR)?](https://blog.roboflow.com/what-is-optical-character-recognition-ocr/)
- [4] [Comparison of GPT-4 Vision and LLaVA on OCR and Document Understanding Tasks](https://encord.com/blog/gpt-vision-vs-llava/)
- [5] [Understanding Multimodal LLMs](https://magazine.sebastianraschka.com/p/understanding-multimodal-llms)
- [6] [The 8 best AI image generators in 2025](https://zapier.com/blog/best-ai-image-generator/)
- [7] [Multi-modal ML with OpenAI's CLIP](https://www.pinecone.io/learn/series/image-search/clip/)
- [8] [Multimodal Embeddings: An Introduction](https://towardsdatascience.com/multimodal-embeddings-an-introduction-5dc36975966f)
- [9] [Best Multimodal Embeddings APIs (2025)](https://www.edenai.co/post/best-multimodal-embeddings-apis)
- [10] [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/html/2407.01449v6)
- [11] [Transforming the Future of Information Retrieval with ColPali](https://blog.vespa.ai/Transforming-the-Future-of-Information-Retrieval-with-ColPali/)
- [12] [Multimodal Document Retrieval with ColPali and Qdrant](https://qdrant.tech/blog/qdrant-colpali/)
- [13] [Multimodal RAG with ColPali and Gemini](https://learnopencv.com/multimodal-rag-with-colpali/)
- [14] [Use ColPali with Milvus](https://milvus.io/docs/use_ColPali_with_milvus.md)