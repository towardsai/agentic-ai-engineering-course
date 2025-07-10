## Global Context of the Lesson

- **What I’m planning to share**: A hands-on lesson on presenting the fundamentals of working with multimodal data in the context of LLMs, RAG and context engineering, all fundamental components required to build industry-level AI agents or LLM workflows. More concretely we want to show to use work with multimodal LLMs and embeddings models used for RAG. We want to blend theory with practicality. Thus, 20% of the article should explain the principles of multimodal LLMs and embeddings models, while in the rest we plan to show how to implement them with various use cases such as working with images, audio and PDFs. As an interesting use case we will show how to implement a simple ColPali architecture. Also, in the context of processing PDF documents we want to explain the power of using multimodal techniques such as ColPali vs. older OCR-based techniques.
- **Why I think it’s valuable:** In the real-world we rarely work only with text data. Often we have to manipulate multimodal data such as text, images and PDFs within the same context window or with different tools passed to the LLM, where the most common ones are retrieval tools that based on specific queries can return any type of data. Thus, knowing how to manipulate multimodal data and integrate them with LLMs, RAG, and agents it's a foundational skill when working in the industry.
- **Who the intended audience is:** Asipiring AI Engineers who are learning for the first time about multimodal LLMs, RAG and agents.
- **Theory / Practice ratio:** 20% theory / 80% practice
- **Expected length of the article in words** (where 200-250 words ~= 1 minute of reading time): ...


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
8. Implementing ColPali
9. Building multimodal AI agents


## Section 1: Why we need multimodal 
- Why agents need to understand images, documents, and complex layouts
- Real-world scenarios and limitations of text-only approaches

## Section 2: Understanding the limitations of traditional approaches
- OCR-based document processing
- Text extraction challenges with complex layouts
- When traditional methods break down

## Section 3: Explaining the foundations of multimodal LLMs
- Core concepts and architecture. Explain the foundations using a text-image multimodal LLM as a concrete example. 
- Give a note on how it can be expended to other modalities, such as PDFs or audio.
- Capabilities and limitations
- Hands-on: Basic image understanding with multimodal LLMs

## Section 4: Working with images and PDFs
- Short example that supports our theoretical section, where we show how to use Gemini to work with images and PDFs
- First show to use them as URLs, explaining that we can leverage this method for public file or when storing them in company data lakes, such as AWS S3.
- Secondly, show how to use them leveraging the BASE64 format, where we load them from disk, encode them in BASE64 and pass them to the LLM.
- Knowing how to work with both URLs and BASE64 is important because of different deployment scenarios

## Section 5: Explaining the foundations of multimodal embedding models
- Core concepts and architecture. Explain the foundations using a text-image multimodal embedding model, such as CLIP, as an example.
- Give a note on how it can be expended to other modalities, such as PDFs or audio.

## Section 6: Working with multimodal embeddings models
- Short example that supports our theoretical section, where we show how to use Gemini to work with text and images, embedding them into the same vector space.
- Use FAISS to embed multiple images and then query them using a text-based query.
- Emphasize how this relates to RAG pipelines for agents

## Section 7: Understading ColPali: The king of multimodal RAG
- Why layout matters for document comprehension
- ColPali architecture and innovations

## Section 8: Implementing ColPali
- Hands-on: Implementing ColPali step-by-step using Gemini models
- Tables, figures, and complex layouts

## Section 9: Building multimodal AI agents
- Short section, where we theoretically explain how multimodal techniques can be added to AI Agents by adding multimodal inputs to the reasoning LLM and leveraging multimodal retrieval tools, such as in the ColPali example, which can be adapted to other modalities

## Article Code

Links to code that will be used to support the article. Always prioritize this code over every other piece of code found in the sources: 

- [Notebook 1](...)

## Golden Sources

- [Golden Source 1](...)
- [Golden Source 2](...)
- [Golden Source 3](...)
- [Golden Source 4](...)
- [Golden Source 5](...)

## Other Sources

- [Source 1](...)
