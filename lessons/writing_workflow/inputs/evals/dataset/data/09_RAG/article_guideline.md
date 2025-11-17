## Global Context of the Lesson

### What We Are Planning to Share

This lesson provides concise knowledge about the Retrieval Augmented Generation (RAG) method. This method consists of retrieving important/relevant information from external sources based on a user's query and adding it into the context window of the LLM.

RAG is one of the key methods we use for context engineering (taught in Lesson 03 of the course). We will explore how RAG is getting integrated into "agentic" pipelines, transforming agents from relying on static knowledge to reasoning over dynamic, external data sources. A key focus is distinguishing standard RAG from agentic RAG (i.e., a ReAct-style agent equipped with a retrieval tool). The lesson will cover practical architectures, and strategies.

### Why We Think It's Valuable

RAG is one of the core technologies (and part of the context engineering work done by AI Engineers) for building AI agents that are grounded, trustworthy, and knowledgeable. It directly addresses LLM limitations like knowledge cut-offs and hallucinations. For an AI Engineer, mastering RAG is not optional—it's a fundamental skill for creating agents that can leverage proprietary data, access real-time information, and provide accurate, source-backed answers. This lesson provides practical and conceptual knowledge needed to understand RAG-powered systems.

### Expected Length of the Lesson

**2000-2500 words** (without the titles and references), where we assume that 200-250 words ≈ 1 minute of reading time.

### Theory / Practice Ratio

100% theory - 0% real-world examples

## Anchoring the Lesson in the Course

### Details About the Course

This piece is part of a broader course on AI agents and LLM workflows. The course consists of 4 parts, each with multiple lessons.

Thus, it's essential to always anchor this piece into the broader course, understanding where the reader is in its journey. You will be careful to consider the following:
- Use the points of view described below.
- To not reintroduce concepts already taught in the previous lesson.
- To be careful when talking about concepts introduced only in future lessons
- To always reference previous and future lessons when discussing topics outside the piece's scope.

### Lesson Scope

This is Lesson 9 (from part 1) of the course on AI Agents.

### Point of View
The course is created by a team writing for a single reader, also known as the student. Thus, for voice consistency across the course, we will always use 'we,' 'our,' and 'us' to refer to the team who creates the course, and 'you' or 'your' to address the reader. Avoid singular first person and don't use 'we' to refer to the student.

Examples of correct point of view:
- Instead of "Before we can choose between workflows and agents, we need a clear understanding of what they are." word it as "To choose between workflows and agents, you need a clear understanding of what they are."

### Who Is the Intended Audience

Aspiring AI engineers who are learning about RAG for the first time.

### Concepts Introduced in Previous Lessons

In previous lessons of the course, we introduced the following concepts:

Part 1:
- Lesson 1 - AI Engineering & Agent Landscape: Role, stack, and why agents matter now
- Lesson 2 - Workflows vs. Agents: Predefined logic vs. LLM-driven autonomy
- Lesson 3 - Context Engineering: Managing information flow to LLMs
- Lesson 4 - Structured Outputs: Reliable data extraction from LLM responses
- Lesson 5 - Basic Workflow Ingredients: Chaining, parallelization, routing, orchestrator-worker
- Lesson 6 - Agent Tools & Function Calling: Giving your LLM the ability to take action
- Lesson 7 - LLM Planning & Reasoning (ReAct and Plan-and-Executre)
- Lesson 8 - Implementing ReAct: Building a reasoning agent from scratch

As this is only the 9th lesson of the course, we have introduced some of the core concepts. At this point, the reader knows what an LLM is, ideas about LLM workflows, AI agents landscape, structured outputs, tools for agents, and ReAct agents.

### Concepts That Will Be Introduced in Future Lessons

In future lessons of the course, we will introduce the following concepts:

Part 1:
- Lesson 10 (next) - Memory for Agents: Short-term vs. long-term memory (procedural, episodic, semantic)
- Lesson 11 - Multimodal Processing: Documents, images, and complex data

Part 2:

- MCP
- Developing the research agent and the writing agent

Part 3:

- Making the research and writing agents ready for production
- Monitoring
- Evaluations

If you must mention these, keep it high-level and note we will cover them in their respective lessons.

### Anchoring the Reader in the Educational Journey

Within the course we are teaching the reader multiple topics and concepts. Thus, understanding where the reader is in their educational journey is critical for this piece. You have to use only previously introduced concepts, while being reluctant about using concepts that haven't been introduced yet.

When discussing the **concepts introduced in previous lessons** listed in the `Concepts Introduced in Previous Lessons` section, avoid reintroducing them to the reader. Especially don't reintroduce the acronyms. Use them as if the reader already knows what they are. 

Avoid using all the **concepts that haven't been introduced in previous lessons** listed in the `Concepts That Will Be Introduced in Future Lessons` subsection. Whenever another concept requires references to these banned concepts, instead of directly using them, use intuitive analogies or explanations that are more general and easier to understand, as you would explain them to a 7-year-old. For example:
- If the "tools" concept wasn't introduced yet and you have to talk about agents, refer to them as "actions".
- If the "routing" concept wasn't introduced yet and you have to talk about it, refer to it as "guiding the workflow between multiple decisions".
You can use the concepts that haven't been introduced in previous lessons listed in the `Concepts That Will Be Introduced in Future Lessons` subsection only if we explicitly specify them. Still, even in that case, as the reader doesn't know how that concept works, you are only allowed to use the term, while keeping the explanation extremely high-level and intuitive, as if you were explaining it to a 7-year-old.
Whenever you use a concept from the `Concepts That Will Be Introduced in Future Lessons` subsection, explicitly specify in what lesson it will be explained in more detail, leveraging the particulars from the subsection. If not explicitly specified in the subsection, simply state that we will cover it in future lessons without providing a concrete lesson number. 

In all use cases avoid using acronyms that aren't explicitly stated in the guidelines. Rather use other more accessible synonyms or descriptions that are easier to understand by non-experts.

## Narrative Flow of the Lesson

Follow the next narrative flow when writing the end-to-end lesson:

- What problem are we learning to solve? Why is it essential to solve it?
    - Start with a personal story where we encountered the problem
- Why other solutions are not working and what's wrong with them.
- At a theoretical level, explain our solution or transformation. Highlight:
    - The theoretical foundations.
    - Why is it better than other solutions?
    - What tools or algorithms can we use?
- Provide some hands-on examples.
- Go deeper into the advanced theory.
- Provide a more complex example supporting the advanced theory.
- Connect our solution to the bigger field of AI Engineering. Add course next steps.

## Lesson Outline

1.  Introduction: Giving LLMs an Open-Book Exam
2.  The RAG System: Core Components
3.  The RAG Pipeline: Ingestion and Retrieval
4.  Advanced RAG Techniques
5.  Agentic RAG
6.  Conclusion

## Section 1 - Introduction: Giving LLMs an Open-Book Exam

- **Quick reference to what we've learned in previous lessons:** Take the core ideas of what we've learned in previous lessons from the `Concepts Introduced in Previous Lessons` subsection of the `Anchoring the Lesson in the Course` section. Briefly recall Context Engineering
- **Transition to what we'll learn in this lesson:** After presenting what we learned in the past, make a transition to what we will learn in this lesson. Take the core ideas of the lesson from the `What We Are Planning to Share` subsection and highlight the importance and existence of the lesson from the `Why We Think It's Valuable` subsection of the `Global Context of the Lesson` section.
- **Additional guidance:**
    - Start by informing readers of a core problem: LLMs are trained on a fixed dataset, making their knowledge static and prone to hallucination. During training, they are essentially taking a "closed-book exam" on the world's information. We don't have, yet, the techniques to enable the models to learn new information over time (after their initial training, after deployment), to update the model's weights. We can fine-tune them but its not efficient, like learning is for humans. The goal would be to let the models learn from experience, over time. 
    - Introduce RAG as a reliable solution to this problem, we can insert new knowledge using the context window
    - With RAG, we are giving the LLM an "open-book exam" by connecting it to external, real-time knowledge sources. Like humans, we don't need to memorize everything, we can use procedures, manuals, "quick sheets", "cheat sheets", etc. Its similar with LLMs.
    - We’ll contrast retrieval with agent memory in Lesson 10 (next one), where we discuss short- and long-term memory stores that complement RAG.
    - Clarify that RAG is a tool/method, AI Engineers use/implement in the process of "Context Engineering" taught in lesson 03 of the course. In that lesson we covered how it was important to curate the context of LLMs when building applications.
    - Briefly outline the lesson's journey: from the "what" and "how" of basic RAG to the advanced and agentic patterns.

- Transition to Section 2: With the problem and motivation clear, we’ll first decompose RAG into its core components so you can see where each responsibility lives.

-  **Section length:** 400 words

## Section 2 - The RAG System: Core Components

- Understanding these components is the first step in the Context Engineering process (Lesson 03) of designing effective RAG systems
- Break down RAG into three conceptual pillars:
    - **Retrieval:** The engine/system for finding relevant information. Semantic similarity or keyword-based search (BM25) is often used to find the most relevant information. Discuss the central role of vector embeddings and vector databases in searching for semantic similarity. Provide a concise explanation of what vector embeddings, how they are created, and how they are stored in a vector database.
    - **Augmentation:** The process of taking the retrieved information and formatting it into the context of a prompt for the LLM.
    - **Generation:** The final step where the LLM uses the augmented input to generate an answer grounded in the provided data.
- Include a Mermaid diagram that illustrates the flow between the user's query, the Retriever, the Augmentation step, and the Generator.

- Transition to Section 3: Now that you can name each moving part, let’s see how they line up across the two phases of a real system.

-  **Section length:** 400 words

## Section 3 - The RAG Pipeline: Ingestion and Retrieval

- Detail the end-to-end RAG workflow, splitting it into its two distinct phases.
- **Phase 1: Offline Ingestion & Indexing**
    - **Load:** Reading documents from various sources (PDFs, websites, APIs). Example tools: Unstructured, LangChain document loaders, LlamaIndex readers.
    - **Split:** Breaking content into smaller, meaningful pieces with rule-based or semantic chunkers (avoid cutting mid-idea). Example: LangChain `RecursiveCharacterTextSplitter`, LlamaIndex `SemanticSplitter`
    - **Embed:** Using an embedding model to convert each chunk into a vector embedding. Example models: OpenAI text-embedding-3-large/small, google's gemini text-embedding-004, Cohere Embed, Voyage, bge variants via Hugging Face. 
    - **Store:** Loading the embeddings and their corresponding text into a vector database or search index for fast similarity lookups. Example stores: FAISS (local), Milvus, Qdrant, Pinecone, Elasticsearch/OpenSearch (with kNN), Azure AI Search.
- **Phase 2: Online Retrieval & Generation**
    - **Query:** A user asks a question; optionally normalize or expand it. Example: LangChain `Runnable` chain or LlamaIndex `QueryEngine`.
    - **Embed:** Turn the query into a vector with the same embedding model as indexing.
    - **Search:** The query vector is used to find the top-k most similar document chunks in the vector database. Example: vector similarity Elasticsearch/OpenSearch; Pinecone filters + vector similarity; FAISS cosine similarity.
    - **Generate:** Build a prompt that includes the user query, instructions, and retrieved chunks; call the LLM to produce a grounded answer. Example: use structured outputs (remind of Lesson 4 on structured outputs) to format the answer, include citations in the answer.

- Include a more detailed Mermaid diagram showing both offline and online paths, tools (generic, not vendor-specific).

- Transition to Section 4: With the end-to-end path in place, the next question is quality: what are the advanced techniques to make the retrieval more accurate and useful across messy, real-world data?

-  **Section length:** 400 words

## Section 4 - Advanced RAG Techniques

- Dedicate this section to exploring methods that significantly improve retrieval performance.
- **Hybrid Search:** Combining keyword-based search (like BM25) for precision with vector search for capturing semantic meaning.
    - What/Why: Pair BM25 precision on exact terms with vector search to capture paraphrases.
    Example (customer support): A user writes “my bill keeps rolling over.” Keyword search finds “rollover” articles; meaning-based search also surfaces “carryover balance” guides. Together, they cover different wordings of the same issue.
- **Re-ranking:** Using a second "re-ranker" model (e.g., a cross-encoder, Cohere Rerank) to re-order the initial retrieved documents for improved relevance. They take the query and one candidate document together and output a relevance score.
    - What/Why: A second model scores (query, candidate) pairs for relevance, improving ordering.
    Example (product help): For “how to connect my account,” the re-ranker pushes the step-by-step setup guide above a press release and a community thread.
- **Query Transformations:** 
    - Decomposition: Break a complex query into sub-questions, retrieve per sub-question, then merge.
        Question: “What’s our travel policy for conferences in Europe this year?”
        Sub-questions: (1) “Where is the policy?” (2) “What counts as a conference?” (3) “What are the Europe rules?” (4) “What changed this year?” Retrieve for each, then combine.
    - HyDE (Hypothetical Document Expansion): Generate a short, ideal answer draft, embed it, then search.
        Before searching, the system drafts a short, sensible answer like: “Employees attending approved conferences in Europe can book economy flights and up to three hotel nights with daily meal limits.” It then looks for documents that sound like that answer, which helps it land on the actual policy pages.
- **Advanced Chunking Strategies:** Moving beyond fixed-size chunks to methods that preserve more context, such as semantic chunking, layout-aware chunking for complex documents or context enriched chunking (contextual retrieval).
    - Fixed chunks: Splitting a 20-page handbook every 500 words might cut the “Reimbursements” section in half, so you get a paragraph without the cap amounts.
    - Semantic chunks: Splitting by headings keeps the whole “Reimbursements” section together, so the exact numbers and exceptions stay intact and show up in one go. 
    - Layout-aware (tables and forms):
    For a pricing table, keeping each row together (product → price → discount) is better than slicing the page by character count and separating numbers from their labels.
 
- **GraphRAG:** Introducing retrieval from knowledge graphs. **Explain that this technique excels at answering questions about complex relationships and interconnected entities, which are often lost in standard document chunks.** It solves problems where understanding the "how" and "why" between data points is as important as the data itself.
    - Retail multi-hop: “Which shoes get the most size-related returns and were featured in last month’s ads?” The system connects returns → reason: sizing → specific SKUs → marketing calendar, then pulls supporting notes.

    - IT operations multi-hop: “Which incidents were caused by weekend deploys that also touched the login service?” It links change records → deploy time → affected service → incident tickets, then surfaces the write-ups.


- add a mermaid diagram: Show hybrid retrieval flow: BM25 results + vector results → union → re-rank → final context.

- Transition to Section 5: These techniques increase retrieval quality. Next, we’ll see how retrieval becomes one tool that an agent can choose to use as it reasons.

-  **Section length:** 400 words

## Section 5 - Agentic RAG

- Tie directly to Lessons 7–8 (ReAct). Emphasize: Agentic RAG is essentially a ReAct-style agent equipped with a retrieval tool. The agent reasons (Thought), decides an Action (e.g., retrieve), observes results, and iterates.

- Clarify: agents typically use many tools (web search, code execution, databases). Labeling a whole system “agentic RAG” can be too narrow—the retrieval tool is just one of several.

- **First, define the core distinction (the theoretical part):**
    - **Standard RAG:** A linear, pre-determined workflow. It's powerful but rigid. Every query follows the same Path: Retrieve -> Augment -> Generate.
    - **Agentic RAG:** Adaptive and iterative. The agent decides when to retrieve, how to reformulate, which source to search, and whether to chain multiple retrieval and reasoning steps.
- **Explain the capabilities of an agentic approach:**
    - The agent can **iteratively** use the RAG tool, refining its query based on initial results.
        - Example: First pass yields a vague policy. Agent narrows scope (“EU customers, 2024 updates”), retrieves again, and reconciles differences.
    - It can **choose** which part of its knowledge base to search (e.g., `search_emails` vs. `search_tech_docs`).
        - Example: For an outage inquiry, agent selects `search_incident_runbooks` over `search_marketing_pages`.
    - It can **fuse** information from the RAG tool with data from other tools (like a web search) to form a comprehensive answer.
        - Example: Agent retrieves internal policy, then calls web_search to check current regulatory thresholds, then synthesizes.
    - It can even decide to **update** the RAG system's knowledge base with new information it learns (preview Lesson 10). Agent may propose writes to a long-term store (falls under Memory for Agents, covered in Lesson 10).we’ll cover that in the next lesson. Keep the explanation high-level here.
- **Then, show (conceptually) it in action:**
    - Discuss the shift from viewing RAG as an isolated process to a core tool in an agent's toolkit
    - Make a link to the ReAct lesson, where in the "thought" step, the agent *reasons* about when it has a knowledge gap and *decides* to call its RAG tool.
    - Use a conceptual Mermaid diagram to show an agent's main loop, where it can choose between tools like `web_search`, `code_interpreter`, and `internal_knowledge_base` (our RAG tool).
    - Mini “thought process” example (no code):
        Thought: “User asks about ‘2024 EU data retention rules’—our internal policy cites 2023. Likely outdated.”

        Action: retrieve(internal_policy, query=“EU data retention 2024”)

        Observation: Mentions amendments but missing citations.

        Thought: “Need external verification.”

        Action: web_search(“EU data retention 2024 official” )

        Observation: Finds directive update.

        Thought: “Synthesize and cite both; highlight changes from 2023.”

- Frame this as the difference between a simple database lookup and a conversation with a knowledgeable research assistant.

- Transition to Section 6: You now understand both a linear RAG pipeline and how an agent can control retrieval when needed. Let’s wrap up by situating RAG in the wider AI Engineering toolkit and previewing what comes next.

-  **Section length:** 400 words

## Section 6 - Conclusion ...
(Connect our solution to the bigger field of AI Engineering. Add course next steps.)

- Summarize the key takeaways: RAG is the most used solution to the LLM knowledge problem, advanced techniques are important for production-grade quality, and the future of knowledge retrieval is agentic.
- Reiterate the core benefits: reducing hallucinations, enabling customization with proprietary data, and building user trust through verifiable, source-based answers.
- Conclude by positioning RAG not as a niche skill but as a foundational competency for the modern AI Engineer, as a subset of Context Engineering.

- To transition from this lesson to the next, specify what we will learn in future lessons. First mention what we will learn in next lesson, which is Lesson 10. 

    (Lesson 10): Memory for Agents — how short-/long-term memory complements retrieval (high-level pointer).

Next leverage the concepts listed in subsection `Concepts That Will Be Introduced in Future Lessons` to make slight references to other topics we will learn during this course. To stay focused, specify only the ones that are present in this current lesson.

Lightly preview other relevant future topics touched here (e.g., evaluations for retrieval quality and monitoring in production) and note they’ll be covered later in the course.

-  **Section length:** 200 words

## Golden Sources

1. [What Is Retrieval-Augmented Generation, aka RAG?](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/)
2. [A Complete Guide to RAG](https://towardsai.net/p/l/a-complete-guide-to-rag)
3. [Retrieval-Augmented Generation (RAG) Fundamentals First](https://decodingml.substack.com/p/rag-fundamentals-first?utm_source=publication-search)
4. [Your RAG is wrong: Here's how to fix it](https://decodingml.substack.com/p/your-rag-is-wrong-heres-how-to-fix?utm_source=publication-search)
5. [From Local to Global: A GraphRAG Approach to Query-Focused Summarization](https://arxiv.org/html/2404.16130)

## Other Sources

6. [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
7. [What is Agentic RAG](https://weaviate.io/blog/what-is-agentic-rag)
8. [RAG is dead, long live agentic retrieval](https://www.llamaindex.ai/blog/rag-is-dead-long-live-agentic-retrieval)
9. [What is agentic RAG?](https://www.ibm.com/think/topics/agentic-rag)
10. [Build advanced retrieval-augmented generation systems](https://learn.microsoft.com/en-us/azure/developer/ai/advanced-retrieval-augmented-generation)
11. [The Rise of RAG](https://highlearningrate.substack.com/p/the-rise-of-rag)
