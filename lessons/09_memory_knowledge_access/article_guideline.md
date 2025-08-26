## **Global Context**

-   **What I'm planning to share:** In this article, will explore the concept of agent "memory". In lesson 3 we presented the core concept of context engineering, and how it was important to curate the context of LLMs when building applications. Memory is one of its most important components. Memory is the where the information is stored. Context Engineering is the active process of deciding what to pull from that resource, how to format it, and when to load it into the model's working attention. Tools like mem0 can be a key part of a Context Engineering strategy

    Why memory? well because we want agents that remember things said in conversations, and personilize the user experience across time. We know that LLMs are "static/stateless" with their current design, they have a fundamental inability to learn (update their weights) over time. This article will present a temporary solution, one that works today, which makes use of external tools/systems, memory management systems that  are trying to engineer a workaround around the model's inability to learn.

    We will explain terms borrowed from biology and cognitive science to make it clear how different types of data can use different retrievel strategies for different kinds of problems.

    We'll differentiate between the model's static internal knowledge, the short-term (context window) memory, and persistent long-term memory.
The focus will be on the three types of long-term memory—semantic (facts), episodic (experiences), and procedural (skills)—and their practical implementation. We will detail how agents create and manage memories, particularly episodic ones, and touch upon retrieval mechanisms like RAG. Finally, we'll ground these concepts in reality by discussing the challenges, best practices, and the evolution of memory systems with concrete examples.

-   **Why I think it's valuable:** Memory is one of the components that elevates a simple chat application to a truly adaptive and personalized agent. For an AI Engineer, understanding how to design and build robust memory systems is important. Memory enables the creation of agents that can maintain conversational continuity, learn from past interactions, access external and proprietary knowledge, and provide more personalized, capable, and reliable assistance.
-   **Who the intended audience is:** Aspiring AI Engineers and developers aiming to build "stateful" agents that can have access to facts about the world, learn, reason, and adapt based on accumulated knowledge and experience.
-   **Expected length of the article in words** (where 200-250 words ~= 1 minute of reading time): ~3000 words (around 12 - 15 minutes reading time)

## **Outline**

1.  Introduction: Why Agents(LLMs) Need a Memory in the first place
2.  The Layers of Memory: Internal, Short-Term, and Long-Term
3.  Long-Term Memory: Semantic, Episodic, and Procedural
4.  Creating Memories: How Agents Learn and Remember
5.  Real-World Lessons: Challenges and Best Practices in Agent Memory

---

## **Section 1: Introduction: Why Agents Need a Memory**

*   Start by highlighting the core limitations of LLMs: their knowledge is vast but frozen in time, they are fundamentaly unable to learn (update their weights) over time (after training, a problem also know as "continual learning"). To overcome these limitations, we can insert new knowledge using the context window, but its a limited solution due to the finite size of the context window + lost in the middle problem.
*   Use an analogy: an LLM without memory is like an intern with amnesia, unable to recall previous conversations or learn new things over time. Unable to learn from past experiences.
*   Introduce the context window as "working memory" or "RAM" and **clearly specify its limitations: keeping the entire conversation thread is not feasible due to finite size, rising costs, and introducing too much noise, which leads to the "needle in a haystack" problem where the model struggles to find relevant information.** but also mention that context window sizes are increasing, over time, which changes the way we need to engineer memory systems. Less compression and retrieval components are needed which can introduce overhead, loss of nuance and details due to constant summarization.
*   Frame memory as the (temporary) solution that provides agents with continuity, adaptability, and the ability to learn.
*   **Use a real-world example: many early agent-building efforts for personal AI companions quickly hit the limits of what was possible with the context window alone, forcing builders to engineer complex memory systems from first principles.**
*   **Section length:** 500 words

## **Section 2: The Layers of Memory: Internal, Short-Term, and Long-Term**

*   Clearly define and differentiate the three fundamental layers of an agent's memory system. Before defining them, explain the value of adopting biology and cognitive science terminology.
    *   **Internal Knowledge:** The static, pre-trained knowledge baked into the LLM's weights. (Which is the best way/place to store facts and knowledge, models know about whole books with an empty context window!). Wouldn't it be great if the model could itself learn about things and update its weights over time? from experience?
    *   **Short-Term Memory:** The active context window of the LLM—volatile, fast, but limited, but its also the only way we can simulate "learning" over time.
    *   **Long-Term Memory:** An external, persistent storage system where an agent can save and retrieve information. Pulling from long-term to short-term memory.
*   **Explain the dynamic between these layers: Long-term memory is "retrieved" and brought into the short-term memory (the context window) to become actionable for the LLM during a task. This can be conceptualized as a "retrieval pipeline" where different types of memories are queried and ranked before being presented to the model.**
*   Create a Mermaid diagram to visually represent this hierarchy and flow.
*   **Highlight why we need these three distinct types:** Internal knowledge provides general intelligence, basic performance, short-term memory handles the immediate task, and long-term memory provides the context and personalization that internal knowledge lacks and short-term memory cannot retain. No single layer can perform all three functions effectively.
*   **Section length:** 400 words (without counting the diagram)

## **Section 3: Long-Term Memory: Semantic, Episodic, and Procedural**

*   Dedicate this section to a detailed breakdown of the three key types of long-term memory with their practical roles.
*   **Semantic Memory (Facts & Knowledge):** The agent's encyclopedia.
    *   What it is: Semantic memory is the agent's repository of objective, timeless facts, these facts can be individual independent strings, or facts/characteristics attached to an "entity" (a person, a place, an object, etc). Think of it as a highly organized encyclopedia or a personal database. This is where the agent stores extracted concepts and relationships about specific domains, people, places, and things. This information is context-independent; a fact stored here is considered true until it is explicitly updated.
    *   How it's used: The primary role of semantic memory is to provide the agent with a reliable source of truth. For an enterprise agent, this might involve storing internal company documents, technical manuals, or an entire product catalog, allowing it to answer questions on proprietary topics with high accuracy. For agents as personal assistants, semantic memory is used to build a persistent profile of the user. It recalls specific, important information like preferences ("Food preferences: The user is a vegetarian"), relationships ("Dog: The user has a dog named George"), or hard constraints ("Food preferences: The user is allergic to gluten"). By storing these as distinct facts, the agent can retrieve them with more precision/recall, rather than hoping to find them by searching through a noisy conversation history.

*   **Episodic Memory (Experiences & History):** The agent's personal diary.
    *   What it is: Episodic memory is the agent's personal diary, a chronological record of its past experiences and interactions the user. It is a log of specific events, complete with timestamps and the context in which they occurred. Unlike the timeless facts in semantic memory, episodic memories are about "what happened and when."
    *   How it's used: This memory type is useful for maintaining conversational context and understanding the narrative of a relationship. It showcases the dynamic of short-term interactions being processed and saved into long-term memory. For instance, a simple system might extract the fact "User's brother is named Mark" and save it to semantic memory. A richer system leverages episodic memory to capture the vital nuance: "**On Tuesday**, the user expressed frustration that their brother, Mark, always forgets their birthday." This single episode provides deep relational context, allowing the agent to interact with more empathy and intelligence in the future (e.g., "I know the topic of your brother's birthday can be sensitive..."). Depending on the use-case, these episodic memories can group "important" events/facts/insights that happened during a whole day, conversation or week. There is no, one-size-fits-all solution to this, it depends on the use-case.

*   **Procedural Memory (Skills & How-To):** The agent's muscle memory.
    *   What it is: Procedural memory is the agent's collection of skills and learned workflows. It's the "how-to" knowledge that dictates its ability to perform multi-step tasks. Think of it as the agent's muscle memory or a set of pre-defined playbooks for common requests.
    *   How it's used: This memory is often baked directly into the agent's system prompt as a reusable tool, function, or defined sequence of actions. For example, an agent might have a stored procedure called MonthlyReportIntent. When a user asks for a monthly update, the agent doesn't need to reason from scratch about how to create a report. Instead, this procedure is triggered, defining a clear series of steps: 1) Query the sales database for the last 30 days, 2) Summarize the key findings, and 3) Ask the user if they want the summary emailed or displayed directly. This makes the agent's behavior on common tasks highly reliable, fast, and predictable. By encoding successful (and even unsuccessful) workflows, procedural memory allows an agent to improve its task completion efficiency over time, reducing errors and ensuring that complex jobs are executed consistently every time.
*   **Section length:** 600 words

## **Section 4: The Architecture of Memory Creation and Retrieval with code examples**

While Retrieval-Augmented Generation (RAG) is the mechanism for *retrieving* long-term memory, the creation of high-quality memories is an equally important, preceding step. An agent must first form a memory before it can be recalled. This process of creation and retrieval is distinct for each memory type, and of course use-case.

**Semantic Memory: Extracting Facts**

*   **How it's Created:** Semantic memory is created through a deliberate **extraction pipeline**. After a conversation (or in real-time), the unstructured text is passed to an LLM with a specific prompt designed to distill it into flat strings of factual data. This process turns conversational nuance into a reliable, queryable knowledge base.

*   **Prompt Used to Create:** It instructs the model to act as a knowledge extractor, identifying facts/user preferences/user attributes/user relationships, anything that is important for the agent's use-case. If its a personal assistant, we might not want to extract the same things as an Tutoring agent.
    *   **Example Extraction Prompt (For a general personal assistant):**
        ```
        Extract persistent facts and strong preferences as short bullet points.
        - Keep each fact atomic and context-independent. While reading the messages, make sure to notice the nuance or sublte details that might be important when saving these facts.
        - 3–6 bullets max.

        Text:
        {My brother Mark is a software engineer, but his real passion is painting, he gifted me a painting a few years ago. Its really beautiful.}
        ```
    *   **Memory Created:** The LLM would process this and the system would store: `Mark is the user's brother. Mark is a software engineer. Mark's real passion is painting. The user has a painting from Mark and finds it beautiful.`.

*   **How it's Retrieved:** Retrieval of semantic memory is where **hybrid search** is most powerful, as it combines the best of keyword precision and semantic relevance.
    1.  **Tag/Keyword Filtering:** The system first filters the memory store based on exact matches for known entities or tags. If the query is "What's my brother's job?", it first narrows the search to all memories that contain `brother`.
    2.  **Semantic Search:** Within that pre-filtered set, a vector search is performed to find the most contextually relevant fact. The query "job" will have high semantic similarity to the stored memory `is a software engineer`, ensuring a relevant answer is retrieved.


**Episodic Memory: The Log of Experience**

*   **How it's Created:** It functions as a chronological log of extracted events. They can be created by having an LLM read the conversation messages for a whole day, and then extract/summarize the insights/facts/events that happened. The memories will have a timestamp. These memories can be stored "raw" or summarized.

*   **Prompt Used to Create:** If stored "raw", no extraction prompt is needed. The creation process is simply **logging the raw conversation text**. The memory is the event itself. But if we want, these memories can be extracted with an LLM, the prompt might be "You are a personal coding tutor for a user, you will extract events, likes, dislikes, or any other insights from the conversation text, that will serve to better teach the user, and help them improve their skills in coding. Make sure to capter the nuance and details of the conversation, and not just the facts."
    *   *Example Input:* `User: "I'm feeling stressed about my project deadline on Friday.", Assistant: "I'm sorry to hear that. I'm here to help you with that."`
    *   *Memory Created (raw):* `October 26th, 2025. 2:30PM EST: User: "I'm feeling stressed about my project deadline on Friday." Assistant: "I'm sorry to hear that. I'm here to help you with that."`

    *   *Memory Created (summarized):* `October 26th, 2025. 2:30PM EST User: "The user is stressed about their project deadline on Friday and the assistant offers to help."`

*   **How it's Retrieved:** Retrieval from episodic memory is often a blend of temporal and semantic queries.
    *   **Tag Filtering:** The simplest retrieval is filtering by a date range (e.g., "What did we talk about yesterday?").
    *   **Hybrid Search:** A more robust approach uses semantic search to find conversations that are contextually similar to the current query, and then re-ranks the results based on recency. For example, if a user asks, "What was that thing I was worried about earlier?" the system would search for past messages with a "worried" or "stressed" sentiment and prioritize the most recent ones.


**Procedural Memory: Defining and Learning Skills**

*   **How it's Created:** Procedural memory is unique in that it can be created in two distinct ways:
    1.  **Developer-Defined:** Where a developer explicitly codes a tool or function (e.g., `book_flight()`) that can get triggered during a conversation. If the user is trying to book a flight, the agent can use the `book_flight` procedure to do so.
    
    2.  **User-Taught / Learned:** More advanced agents can learn new procedures dynamically from user interactions. When a user provides explicit steps for a task, the agent can "save" this sequence as a new, Callable procedure in the future.

    *   **Example Prompt:**
        ```
        You are an agent that can learn new skills. When a user provides a numbered list of steps to accomplish a goal, your task is to run the "learn_procedure" tool, and convert these numbered list of steps into a reusable procedure. Identify the core actions and any variable parameters (e.g., dates, locations, names).

        Examples:
        User Input: "I want you to book a cabin for this summer. To do that, please remember to: 1. Search for cabins on CabinRentals.com my favorite website. 2. Filter for locations in the mountains, the closer to them, the better. 3. Make sure it's available around July. 4 to 8th. 5. Send me the top 3 options."

        learn_procedure(name="find_summer_cabin", steps=first search for cabins on CabinRentals.com, then filter for locations in the mountains, check for distance to the mountains, then check availability around what the user wants, if the user hasn't specified a date, ask the user for a date, once you have a list of options, send the top 3 options)
        ```
    *   **Memory Created:** The LLM would generate a new structured procedure and save it to its tool/procedures library: `procedure_name: find_summer_cabin`, `steps=first search for cabins on CabinRentals.com, then filter for locations in the mountains, check for distance to the mountains, then check availability around what the user wants, if the user hasn't specified a date, ask the user for a date, once you have a list of options, send the top 3 options`.

*   **How it's Retrieved:** Retrieval is an **intent-matching and function-calling** process. The agent is given access to its entire library of procedures—both built-in and user-taught.
    *   **No Pre-filtering:** The LLM receives the descriptions of all available tools in its context.
    *   **Semantic Matching:** It compares the user's current request against the descriptions of all procedures. If the user later says, "Let's find a summer cabin again", the agent will recognize the semantic similarity to its newly learned `find_summer_cabin` procedure and execute it, demonstrating a form of learning and adaptation.

*   **Section length:** 600 words

## **Section 5: Real-World Lessons: Challenges and Best Practices in Agent Memory**

The architectural patterns described in the previous section provide a powerful toolkit for building agents with memory. However, moving from theory to a reliable, production-ready system requires navigating a series of complex trade-offs that are constantly evolving with the underlying technology. The most successful AI engineers are not those who rigidly follow a blueprint, but those who understand the core principles and adapt them to the challenges they face in the wild.

Here are some of the most important lessons learned from building and scaling agent memory systems in the real world.

#### **1. Re-evaluating Compression**

One of the biggest shifts in memory design has been the trade-off between compressing information and preserving its raw detail.

*   **The Old Challenge:** Just two years ago, LLMs operated with small and expensive context windows (e.g., 8,000 tokens). This constraint forced AI Engineers to be ruthless with compression. The primary goal was to distill every interaction into its most compact form—summaries, facts, or entities—to save costs and avoid overflowing the context. While necessary, this process was inherently lossy. By summarizing; you keep the general idea but lose the fine details and, more importantly, the nuance.
*   **The New Reality:** Today, with models like the ones from the gemini-2.5 family offering million-token context windows at a fraction of the cost, the challenge has changed. The best practice is now to **lean towards less compression**. The raw, unstructured conversational history is the ultimate source of truth. It contains the emotional subtext, subtle hesitations, and relational dynamics that are often dismissed during extraction. While a fact might state, "User has a dog named George, user likes to walk their dog" the episodic log reveals, "User mentioned that walking their dog named George is the best part of their day," a far more valuable piece of information for a personalized agent.

*   **Best Practice:** Design your system to work with the most complete version of history that is economically and technically feasible. Use summarization and fact extraction as tools for creating queryable indexes, but always treat the raw log as the ground truth. As context windows grow, your retrieval pipeline may need to do less *retrieving* and more intelligent *filtering* of a larger, in-context history.

#### **2. Designing for the Product, Not the Paradigm**

There is no such thing as a "perfect" memory architecture. The concepts of semantic, episodic, and procedural memory are a powerful mental model, but they are a toolkit, not a mandatory blueprint. The most common failure mode is over-engineering a complex, multi-part memory system for a product that doesn't need it.

*   **The Challenge:** It can be tempting to build a system that handles all three memory types from day one. However, this often leads to unnecessary complexity, higher maintenance costs, and slower performance.
*   **The Best Practice:** Start from **first principles** by defining the core function of your agent. The product's goal should dictate the memory architecture, not the other way around.
    *   **For a Q&A bot over internal documents,** semantic memory is king. Your focus should be on building a robust, factually accurate knowledge base. Episodic and procedural memory are likely irrelevant.
    *   **For a long-term personal AI companion,** rich episodic memory is the most critical component. The agent's value comes from its ability to remember the narrative of your relationship. Semantic facts are useful, but they are secondary to the shared history.
    *   **For a task-automation agent,** procedural memory is likely more valuable. Its ability to reliably execute multi-step workflows is its core purpose. It only needs to remember facts or episodes that are directly relevant to completing its tasks.

#### **3. The Human Factor: Avoiding User Cognitive Overhead**

Memory exists to make the agent smarter, not to give the user a new job. A common pitfall is exposing the internal workings of the memory system to the user, thinking it will improve transparency and accuracy. In practice, it often does the opposite.

*   **The Challenge:** Most "memory" implementations are designed where users can view, edit, or delete the facts the agent had stored about them. While well-intentioned, this can create significant cognitive overhead.
*   **The Best Practice:** Users should not be asked to **"garden their agent's memories."** This breaks the illusion of a capable assistant and turns the interaction into a tedious data-entry task. A user's mental model is that they are talking to a single entity; they don't want to switch to being a database administrator.
    *   Memory management should be an **autonomous function of the agent**. It should learn from corrections within the natural flow of conversation (e.g., "Actually, my brother's name is Mark, not Mike").
    *   Design internal processes for the agent to periodically review, consolidate, and resolve conflicting information in its memory stores. The agent, not the user, is responsible for maintaining the integrity of its own knowledge.

By keeping these real-world challenges in mind, you can design memory systems that are not just technically impressive, but are also practical, scalable, and genuinely useful for the end-user.


*   **Section length:** 500 words

## **Golden Sources**

- https://www.youtube.com/watch?v=7AmhgMAJIT4&list=PLDV8PPvY5K8VlygSJcp3__mhToZMBoiwX&index=112 (Nice real-world example)
- https://decodingml.substack.com/p/memory-the-secret-sauce-of-ai-agents
- https://www.newsletter.swirlai.com/p/memory-in-agent-systems
- https://arxiv.org/html/2309.02427
- https://danielp1.substack.com/p/memex-20-memory-the-missing-piece


## **Other Sources**

- https://www.ibm.com/think/topics/ai-agent-memory
- https://arxiv.org/html/2504.19413 (Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory)
- https://docs.letta.com/guides/agents/memory
- https://langchain-ai.github.io/langgraph/concepts/memory/
- https://medium.com/@honeyricky1m3/giving-your-ai-a-mind-exploring-memory-frameworks-for-agentic-language-models-c92af355df06


