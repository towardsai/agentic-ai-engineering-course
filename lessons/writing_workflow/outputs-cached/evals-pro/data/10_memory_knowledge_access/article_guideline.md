## Global Context of the Lesson

### What We Are Planning to Share

In this article, will explore the concept of agent "memory". In lesson 3 we presented the core concept of context engineering, and how it was important to curate the context of LLMs when building applications. Memory is one of its most important components. Memory is the where the information is stored. Context Engineering is the active process of deciding what to pull from that resource, how to format it, and when to load it into the model's working attention. Tools like mem0 can be a key part of a Context Engineering strategy

Why memory? well because we want agents that remember things said in conversations, and personilize the user experience across time. We know that LLMs are "static/stateless" with their current design, they have a fundamental inability to learn (update their weights) over time. This article will present a temporary solution, one that works today, which makes use of external tools/systems, memory management systems that  are trying to engineer a workaround around the model's inability to learn.

We will explain terms borrowed from biology and cognitive science to make it clear how different types of data can use different retrievel strategies for different kinds of problems.

We'll differentiate between the model's static internal knowledge, the short-term (context window) memory, and persistent long-term memory.
The focus will be on the three types of long-term memory—semantic (facts), episodic (experiences), and procedural (skills)—and their practical implementation. We will detail how agents create and manage memories, particularly episodic ones, and touch upon retrieval mechanisms like RAG. Finally, we'll ground these concepts in reality by discussing the challenges, best practices, and the evolution of memory systems with concrete examples.

### Why We Think It's Valuable

Memory is one of the components that transforms a simple stateless chat application to a truly adaptive and personalized agent. For an AI Engineer, understanding how to design and build robust memory systems is important. Memory enables the creation of agents that can maintain conversational continuity, learn from past interactions, access external and proprietary knowledge, and provide more personalized, capable, and reliable assistance.

### Expected Length of the Lesson

**2000-2500 words** (without the titles and references), where we assume that 200-250 words ≈ 1 minute of reading time.

### Theory / Practice Ratio

50% theory - 50% real-world examples

## Anchoring the Lesson in the Course

### Details About the Course

This piece is part of a broader course on AI agents and LLM workflows. The course consists of 4 parts, each with multiple lessons. 

Thus, it's essential to always anchor this piece into the broader course, understanding where the reader is in its journey. You will be careful to consider the following:
- The points of view
- To not reintroduce concepts already taught in the previous lesson.
- To be careful when talking about concepts introduced only in future lessons
- To always reference previous and future lessons when discussing topics outside the piece's scope.

### Lesson Scope

This is lesson 9 (from part 1) of the course on AI Agents.

### Point of View
The course is created by a team writing for a single reader, also known as the student. Thus, for voice consistency across the course, we will always use 'we,' 'our,' and 'us' to refer to the team who creates the course, and 'you' or 'your' to address the reader. Avoid singular first person and don't use 'we' to refer to the student.

Examples of correct point of view:
- Instead of "Before we can choose between workflows and agents, we need a clear understanding of what they are." word it as "To choose between workflows and agents, you need a clear understanding of what they are."

### Who Is the Intended Audience

Aspiring AI engineers who are learning about memory for agents for the first time.

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
- Lesson 10 - RAG Deep Dive: Knowledge-augmented retrieval and generation 
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

1. Section 1: Introduction: Why Agents Need a Memory in the first place
2. Section 2: The Layers of Memory: Internal, Short-Term, and Long-Term
3. Section 3: Long-Term Memory: Semantic, Episodic, and Procedural
4. Section 4: Storing Memories: Pros and Cons of Different Approaches
5. Section 5: Memory implementations with code examples
6. Section 6: Real-World Lessons: Challenges and Best Practices
7. Section 7: Conclusion

## Section 1 - Introduction: Why Agents Need a Memory in the first place

- **Quick reference to what we've learned in previous lessons:** Take the core ideas of what we've learned in previous lessons from the `Concepts Introduced in Previous Lessons` subsection of the `Anchoring the Lesson in the Course` section.
- **Transition to what we'll learn in this lesson:** After presenting what we learned in the past, make a transition to what we will learn in this lesson. Take the core ideas of the lesson from the `What We Are Planning to Share` subsection and highlight the importance and existence of the lesson from the `Why We Think It's Valuable` subsection of the `Global Context of the Lesson` section.
- **Additional guidance:**
    - Start by reminding the readers of the core limitations of LLMs of today, which is that their knowledge is vast but frozen in time, they are fundamentaly unable to learn (by updating their weights/parameters) over time (after training/deployment) a problem also know as "continual learning". To overcome these limitations, we can insert new knowledge using the context window, but its a limited solution due to the finite size of the context window + lost in the middle problem.
    - Use an analogy: an LLM without memory is like an intern with amnesia, unable to recall previous conversations or learn new things over time. Basically unable to learn from experience.
    - Explain the context window as "working memory" or "RAM" but clearly remind readers of its limitations: keeping the entire conversation thread (+ additional information) is not realistic due to finite size, rising costs (more tokens leads to higher costs per turn), and introducing too much noise (the model might not need all that information to answer a simple question), there is also the "lost in the middle problem" (the model struggles to correctly use relevant information, when its placed in the middle of the context window).
    - As a counterpoint, also mention that context window sizes are actually increasing over time, which shows how we need to continously adapt and change the way we need to engineer agentic stateful systems. In a future with bigger context windows, or actual learning, less compression or retrieval components will be needed as they introduce overhead and loss of nuance or details.
    - Frame `memory tools` as the temporary solution that provides agents with continuity, adaptability, and the ability to "learn".
    - Use a real-world example: many early agent-building efforts for personal AI companions quickly hit the limits of what was possible with the context window alone, forcing builders to engineer complex memory systems, that had to include a lot of compression and retrieval components. Working with 8k or 16k tokens of context windows posed a different challenge, than now with models that have 1M+ tokens of context windows.
- Transition to Section 2: How can we think of memory in a useful way to build agents, Memory can have different time horizons. We can borrow from biology and cognitive science to understand how memory works in humans.

-  **Section length:** 400 words

## Section 2 - The Layers of Memory: Internal, Short-Term, and Long-Term

- Clearly define and differentiate the three fundamental layers of an agent's memory system. Before defining them, explain the value of adopting biology and cognitive science terminology.
    - **Internal Knowledge:** The static, pre-trained knowledge baked into the LLM's weights. (Which is the best way/place to store facts and knowledge, models know about whole books with an empty context window). Wouldn't it be great if the model could itself learn about things and update its weights over time? from experience?
    - **Short-Term Memory:** The active context window of the LLM—volatile, fast, but limited, but its also the only way we can simulate "learning" over time.
    - **Long-Term Memory:** An external, persistent storage system where an agent can save and retrieve information. Pulling from long-term to short-term memory.
- Explain the dynamic between these layers: Long-term memory is "retrieved" and brought into the short-term memory (the context window) to become actionable/give important information for the LLM during a task. This can be conceptualized as a "retrieval pipeline" where different types of memories are queried in parallel and ranked before being presented to the model.
- Create a Mermaid diagram to visually represent this hierarchy and flow.
- Explain the usefulness of categorizing memory in these three distinct types: Internal knowledge provides general intelligence, basic performance, short-term memory handles the immediate task, and long-term memory provides the context and personalization that internal knowledge lacks and short-term memory cannot retain. No single layer can perform all three functions effectively.
- Transition to Section 3: To better think about long-term memory, we can further borrow from biology and cognitive science to understand how memory works in humans and apply it to agents. 

-  **Section length:** 250-300 words

## Section 3 - Long-Term Memory: Semantic, Episodic, and Procedural

- This section provides a detailed breakdown of the three key types of long-term memory with their practical roles.
    - **Semantic Memory (Facts & Knowledge):** The agent's encyclopedia.
        - What it is: Semantic memory is the agent's repository of individual pieces of knowledge, these "facts" can be individual independent strings "The user is a vegetarian", or they can be attached to an "entity" which can be a person, a place, an object, etc {"food restrictions": "User is a vegetarian"}. The semantic memory is where the agent stores extracted concepts, relationships, etc about specific domains, people, places, and things. What you decide to store and how you decide to structure the memory is highly dependent on the use-case of the Agent. It could even be strcutured as a graph database. We'll come back to the pros and cons of each approach at the end of this section.

        - How it's used: The primary role of semantic memory is to provide the agent with a reliable source of truth. For an enterprise agent, this might involve storing internal company documents, technical manuals, or an entire product catalog, allowing it to answer questions on proprietary topics. For agents as personal assistants, semantic memory can be used to build a persistent profile of each user. It can recall specific, important information like preferences {"music": "User likes rock music"}, relationships {"dog": "User has a dog named George"}, or hard constraints {"food restrictions": "User is allergic to gluten"}. Now, when using the agent, it has the capability of retrieving relevant and potentially important information, rather than depending on a noisy and very long conversation history.

    - **Episodic Memory (Experiences & History):** The agent's personal diary.
        - What it is: Episodic memory is the agent's personal diary, a record of its past interactions with the user. Think of this memory as facts but with a timestamp attached to it, an additional element of time. It is a log of specific events, and the context in which they occurred. Unlike the timeless facts in semantic memory, episodic memories are about "what happened and when."
        - How it's used: This memory type is useful for maintaining conversational context and potentially understand something complex like the dynamics of a relationship. For instance, a simple system of facts might extract "User's brother is named Mark" and "User is frustrated with his brother" and save it to the semantic memory. A system that captures the element of time might save: "On Tuesday, the user expressed frustration that their brother, Mark, always forgets their birthday, I then provided an empathetic response. [created_at=2025-08-25T17:20:04.648191-07:00]"
        This "episode" provides a deeper/nuanced context, allowing the agent to interact with more empathy and intelligence in the future if the topic of his brother comes up again. (e.g., "As you expressed last week, I know the topic of your brother's birthday can be sensitive..."). With the time element, the agent can now also answer question such as "What happened on June 8th?" Depending on the use-case, these episodic memories can group important events/facts/insights that happened during a whole day, a single conversation, over the span of a week, etc. There is no, one-size-fits-all solution to this, depending on the product a different time-scale might be needed.

    - **Procedural Memory (Skills & How-To):** The agent's muscle memory.
        - What it is: Procedural memory is the agent's collection of skills and learned workflows. It's the "how-to" knowledge that dictates its ability to perform multi-step tasks. Think of it as the agent's muscle memory or a set of pre-defined playbooks for common requests.
        - How it's used: This memory is often baked directly into the agent's system prompt as a "reusable tool", function, or defined sequence of actions. For example, an agent might have a stored procedure called MonthlyReportIntent. When a user asks for a monthly update, the agent doesn't need to reason from scratch about how to create a report. Instead, by retrieving from its procedural memory, the procedure can then be used, defining a clear series of steps: 1) Query the sales database for the last 30 days, 2) Summarize the key findings, and 3) Ask the user if they want the summary emailed or displayed directly. This makes the agent's behavior on common tasks highly reliable, fast, and predictable. By encoding successful (and even unsuccessful) workflows, procedural memory allows an agent to improve its task completion efficiency over time, reducing errors and ensuring that complex jobs are executed consistently every time.

- Transition to Section 4: Now that we have an idea of what to save, the benefits of specific types of memories. How should they be saved/stored? What are the three approaches we can experiment with?

-  **Section length:** 400 words

## Section 4 - Storing Memories: Pros and Cons of Different Approaches

This section explores how the way agent's memories are stored is an important architectural decision that directly impacts its performance, complexity, and the ability to scale the product. While the goal is always to provide the right context at the right time, the method of storage involves trade-offs. There is no one-size-fits-all solution; the ideal approach depends entirely on the product's use-case. Let's explore the pros and cons of the three primary methods we are experimenting with as AI Engineers: storing memories as raw strings, as structured entities (like JSON), and within a knowledge graph.

1. Storing memories as raw strings: This is the simplest method, where conversational turns or documents are stored as plain text and typically indexed for vector search.
    - Pros:
        - Simple and fast: This method is the easiest to set up. It involves logging text and creating embeddings, requiring minimal engineering overhead to get started.
        - Preserves nuance: By storing the raw text, the full context, including emotional tone and subtle linguistic cues, is preserved. Nothing is lost in translation to a structured format.
    - Cons:
        - Imprecise retrieval: Relying solely on semantic similarity is not enough most of the time. A query can retrieve text that is semantically related but contextually wrong. For example, asking "What is my brother's job?" might retrieve every past conversation where "brother" and "job" were mentioned, without being able to pinpoint the single correct fact.
        - Difficulty in Updating: Updating the memory is a very important aspect. If a user corrects a piece of information ("My brother is no longer a lawyer, he's a doctor now"), you cannot simply update the old fact. The new information is just another string in a growing log, creating potential contradictions.
        - Lack of Structure: This approach struggles with temporal reasoning and state changes. It cannot easily distinguish between "Barry *was* the CEO" and "Claude *is* the CEO" because the relationship is not explicitly defined, the memory lacks the element of time.

2. Storing Memories as Entities (JSON-like Structures):In this approach, we go from unstructured messy interactions to structured memories. Using an LLM to do so and storing them in a format like JSON.
    - Pros:
        - Structured and precise: Information is organized into key-value pairs (`"user": {"brother": {"job": "Software Engineer"}}`), which allows for precise, field-level filtering. This structure makes it easy to retrieve specific facts without ambiguity. By filtering for "brother", the agent can retrieving everything related to the brother, including his job, his name, his age, etc.
        - Easier to update: If a user's preference changes, only the relevant field in the JSON object needs to be updated, ensuring the memory remains up to date.
        - Ideal for factual data: This method is perfectly suited for semantic memory, where user profiles, preferences, and key relationships are stored as facts/characteristics/preferences.
    - Cons:
        - Increased upfront complexity: This approach requires designing a schema or data model. Deciding what information to extract and how to structure it adds an initial layer of engineering complexity.
        - Potential for schema rigidity: A predefined schema can be inflexible. If the agent encounters information that doesn't fit the existing structure, that data may be lost unless the schema is updated, which can be a complicated process.
        - We can let an LLM dynamically add new entities, new fields, change the structure of the schema, but then updating the memories is more complex, with an increased risk saving duplicated information.
        - Loss of original nuance: The extraction process, by its nature, strips away the rich subtext of the original conversation. The factual memory `"user_likes": ["cats"]` is far less representative than the original message, "Petting my cat is the best part of my day."

3. Storing Memories in a Graph Database: This is the most advanced approach, where memories are stored as a network of nodes (entities) and edges (relationships), forming a knowledge graph.
    - Pros:
        - Represents complex relationships: This is the core strength of a graph. Its good at explicitly defining how different pieces of information are connected. For instance, it can map `(User) -> [HAS_BROTHER] -> (Mark) -> [WORKS_AS] -> (Software Engineer)`. This enables sophisticated queries that trace these connections.
        - Superior contextual and temporal awareness: Knowledge graphs can model context and time as explicit properties of a relationship (e.g., `User -[RECOMMENDED_ON_DATE: "2025-10-25"]-> Restaurant`). This is a more accurate and grounded retrieval than vector search alone.
        - Auditability and explainability: Retrieval is transparent. You can trace the exact path of nodes and edges that led to an answer, making it easier to debug the agent's reasoning and build trust in its outputs.
    - Cons:
        - Highest complexity and cost: This method requires a higher upfront investment in schema design, data modeling, and ongoing maintenance. The process of converting unstructured interactions into structured graph triples is a more complex task than just storing strings.
        - Potential for slower queries: While powerful, graph traversals for complex queries can be slower than a simple vector lookup, which might impact real-time performance if not carefully optimized.
        - Overhead for simple use cases: For many applications, the complexity of implementing and maintaining a graph database is overkill. A simpler entity-based or even string-based approach may be more than sufficient.

- Provide a mermaid diagram to visualize the three approaches.
Tip: the choice of memory storage should be guided by your product's core needs. Start with the simplest architecture that delivers value and evolve it as the demands on your agent grow more complex.

- Transition to Section 5: Now that we know what to save and how to store the memories, let's provide some code examples, using available "memory" tools, like mem0.

-  **Section length:** 400 words

## Section 5 - Memory implementations with code examples
This section goes into more details (with code, using the mem0 library) on how to implement the different types of memories. While Retrieval-Augmented Generation (RAG) is the mechanism for retrieving information, the creation of high-quality memories is an equally important, preceding step. RAG is a method that will be covered in the next lesson. Before the retrieval phase, an agent must first form the memory. This process of creation and retrieval is distinct for each memory type, and of course use-case. To focus on the benefits of each 'category' of memory, and not on the type of storage architecture, we will use the simplistic "storing memories as raw strings" approach. 

- Important: For all the following examples, use the code from the provided Notebook. Do not come up with new code.

0. What is mem0?
    - Provide short description of the tool, what it does, and how it works.
    - Explain that we will use it to implement the different types of memories.

1. Semantic Memory: Extracting Facts:
    - How it's created: Semantic memory is created through a deliberate **extraction pipeline**. After a conversation (or after a single turn), the unstructured text is either saved as a raw string or passed to an LLM with a specific prompt designed to extract flat strings of factual data. This process turns the messy conversation threads into queryable knowledge base.

    - Prompt Used to Create: It instructs the model to act as a knowledge extractor, identifying facts/user preferences/user attributes/user relationships, anything that is important for the agent's use-case. If its a personal assistant, we might not want to extract the same things as an Tutoring agent.
        - Example Extraction Prompt (For a general personal assistant):
        ```
        Extract persistent facts and strong preferences as short bullet points.
        - Keep each fact atomic and context-independent. While reading the messages, make sure to notice the nuance or sublte details that might be important when saving these facts.
        - 3–6 bullets max.

        Text:
        {My brother Mark is a software engineer, but his real passion is painting, he gifted me a painting a few years ago. Its really beautiful.}
        ```
    - Memory Created: The system would store: `Mark is the user's brother. Mark is a software engineer. Mark's real passion is painting. The user has a painting from Mark and finds it beautiful.`.
    - How it's Retrieved: Retrieval of semantic memory is where **hybrid search** is useful (we will come back to retrieval methods in the next lesson), as it combines the best of keyword search and semantic relevance.
        1.  Keyword Filtering: The system first filters the memory store based on exact matches for known entities or tags. If the query is "What's my brother's job?", it first narrows the search to all memories that contain `brother`.
        2.  Semantic Search: Within that pre-filtered set, a vector search is performed to find the most contextually relevant fact. The query "job" will have high semantic similarity to the stored memory `is a software engineer`, ensuring a relevant answer is retrieved.

2. Episodic Memory: The Log of Events

    - How it's Created: It functions as a chronological log of extracted events. They can be created by having an LLM read the conversation messages for a whole day, and then extract/summarize the insights/facts/events that happened. The memories will have a timestamp. These memories can be stored "raw" or summarized.

    - Prompt Used to Create: If stored "raw", no extraction prompt is needed. The creation process is simply logging the raw conversation text. The memory is the event itself. But if we want, these memories can be extracted with an LLM, the prompt might be "You are a personal coding tutor for a user, you will extract events, likes, dislikes, or any other insights from the conversation text, that will serve to better teach the user, and help them improve their skills in coding. Make sure to capter the nuance and details of the conversation, and not just the facts."
        - Example Input: `User: "I'm feeling stressed about my project deadline on Friday.", Assistant: "I'm sorry to hear that. I'm here to help you with that."`
        - Memory Created (raw): `October 26th, 2025. 2:30PM EST: User: "I'm feeling stressed about my project deadline on Friday." Assistant: "I'm sorry to hear that. I'm here to help you with that."`

    - Memory Created (summarized): `October 26th, 2025. 2:30PM EST User: "The user is stressed about their project deadline on Friday and the assistant offers to help."`

    - How it's Retrieved: Retrieval from episodic memory is often a blend of temporal and semantic queries.
        - Keyword Filtering: The simplest retrieval is filtering by a date range (e.g., "What did we talk about yesterday?").
        - Hybrid Search: A more robust approach uses semantic search to find conversations that are contextually similar to the current query, and then re-ranks the results based on recency. For example, if a user asks, "What was that thing I was worried about earlier?" the system would search for past messages with a "worried" or "stressed" sentiment and prioritize the most recent ones.

3. Procedural Memory: Defining and Learning Skills

    - How it's Created: Procedural memory is unique in that it can be created in two distinct ways:
        1.  Developer-Defined: Where a developer explicitly codes a tool or function (e.g., `book_flight()`) that can get triggered during a conversation. If the user is trying to book a flight, the agent can use the `book_flight` procedure to do so.
        2.  User-Taught / Learned: More advanced agents can learn new procedures dynamically from user interactions. When a user provides explicit steps for a task, the agent can "save" this sequence as a new, Callable procedure in the future.

    - Example Prompt:
        ```
        You are an agent that can learn new skills. When a user provides a numbered list of steps to accomplish a goal, your task is to run the "learn_procedure" tool, and convert these numbered list of steps into a reusable procedure. Identify the core actions and any variable parameters (e.g., dates, locations, names).

        Examples:
        User Input: "I want you to book a cabin for this summer. To do that, please remember to: 1. Search for cabins on CabinRentals.com my favorite website. 2. Filter for locations in the mountains, the closer to them, the better. 3. Make sure it's available around July. 4 to 8th. 5. Send me the top 3 options."

        learn_procedure(name="find_summer_cabin", steps=first search for cabins on CabinRentals.com, then filter for locations in the mountains, check for distance to the mountains, then check availability around what the user wants, if the user hasn't specified a date, ask the user for a date, once you have a list of options, send the top 3 options)
        ```
    - Memory Created: The LLM would generate a new structured procedure and save it to its tool/procedures library: `procedure_name: find_summer_cabin`, `steps=first search for cabins on CabinRentals.com, then filter for locations in the mountains, check for distance to the mountains, then check availability around what the user wants, if the user hasn't specified a date, ask the user for a date, once you have a list of options, send the top 3 options`.

    - How it's Retrieved: Retrieval is an **intent-matching and function-calling** process. The agent is given access to its entire library of procedures—both built-in and user-taught.
        - No Pre-filtering: The LLM receives the descriptions of all available tools in its context.
        - Semantic Matching: It compares the user's current request against the descriptions of all procedures. If the user later says, "Let's find a summer cabin again", the agent will recognize the semantic similarity to its newly learned `find_summer_cabin` procedure and execute it, demonstrating a form of learning and adaptation.

- Transition to Section 6: We've seen how to implement the different types of memories with mem0, now let's talk about some additional considerations when building a memory system.

-  **Section length:** 500 words

## Section 6 - Real-World Lessons: Challenges and Best Practices

The architectural patterns described in the previous section provide a useful toolkit for building agents with memory. However, moving from theory to a reliable, production-ready system requires navigating a series of complex trade-offs that are constantly evolving with the underlying technology improving so fast.

Here are some of the most important lessons learned from building and scaling agent memory systems in the real world.

1. Re-evaluating compression: One of the biggest changes while designing memory has been the trade-off between compressing information and preserving its raw detail.

    - The Old Challenge: Just two years ago, LLMs operated with small and expensive context windows (e.g., 8,000 or 16,000 tokens). This constraint forced us, AI Engineers, to be ruthless with compression. The primary goal was to distill every interaction into its most compact form—summaries, facts, or entities—to to be able to fit the relevant information into the context window. While necessary, this process is inherently lossy. By summarizing; you keep the general idea but lose the fine details and nuance. Which might be important for a personalized agent.
    - The New Reality: Today, with models like the ones from the gemini-2.5 family offering million-token context windows at a fraction of the cost, the considerations have changed. The best practice is now to lean towards less compression. The raw, unstructured conversational history is the ultimate source of truth. It contains the emotional subtext, subtle hesitations, and relational dynamics that are often dismissed during extraction. While a fact might state, "User has a dog named George, user likes to walk their dog" the episodic log reveals, "User mentioned that walking their dog named George is the best part of their day," a far more valuable piece of information for a personalized agent.

    - Best Practice: Design your system to work with the most complete version of history that is economically and technically feasible. Use summarization and fact extraction as tools for creating queryable indexes, but always treat the raw log as the ground truth. As LLM get larger context windows, your retrieval pipeline may need to do less *retrieving* and more intelligent *filtering* of a larger, in-context history.

2. Designing for the Product: There is no such thing as a "perfect" memory architecture. The concepts of semantic, episodic, and procedural memory are a powerful mental model, but they are a toolkit, not a mandatory blueprint. The most common failure mode is over-engineering a complex, multi-part memory system for a product that doesn't need it.

    - The Challenge: It can be tempting to build a system that handles these memory types from day one. However, this often leads to unnecessary complexity, higher maintenance costs, and slower performance.
    - The Best Practice: Start from first principles by defining the core function of your agent. The product's goal should dictate the memory architecture, not the other way around.
        - For a Q&A bot over internal documents, a simple RAG pipeline (which is covered in the next lesson) is the best starting point. Your focus should be on building a robust, factually accurate knowledge base.
        - For a long-term personal AI companion, rich memories that include an element of time can be beneficial. The agent's value comes from its ability to remember the narrative of your relationship. Simple semantic facts are useful, but in this use-case, they can't fully represent the dynamic elements of a person's life and thus reduces the ability for the agent to provide good help.
        - For a task-automation agent, something like "procedural memories" is likely useful. The agent can recall and execute multi-step workflows.

3. The Human Factor: The additional user cognitive overhead: Memory exists to make the agent smarter, not to give the user a new job. A common pitfall is exposing the all the internal workings of the memory system to the user, thinking it will improve transparency and accuracy. In practice, it often does the opposite.

    - The Challenge: Most "memory" implementations are designed where users can view, edit, or delete the facts the agent had stored about them. While well-intentioned, this can create significant cognitive overhead.
    - The Best Practice: Users should not be asked to "garden their agent's memories." This breaks the illusion of a capable assistant and turns the interaction into a tedious data-entry task. A user's mental model is that they are talking to a single entity; they don't want to switch to being a database administrator.
        - Memory management should be an autonomous function of the agent. It should learn from corrections within the natural flow of conversation (e.g., "Actually, my brother's name is Mark, not Mike").
        - Design internal processes for the agent to periodically review, consolidate, and resolve conflicting information in its memory stores. The agent, not the user, is responsible for maintaining the integrity of its own knowledge.

-  **Section length:** 500 words

## Section 7 - Conclusion ...
(Connect our solution to the bigger field of AI Engineering. Add course next steps.)
- Conclude the lesson by highlighting that memory sits at the core of AI agents and it's useful in making sure we are building personalized agents, that "learn" over time. Memory tools are temporary solution, for true "continual learning" but at the moment its something that works and we can use.
- To transition from this lesson to the next, specify what we will learn in future lessons. First mention what we will learn in next lesson, which is Lesson 10. Next leverage the concepts listed in subsection `Concepts That Will Be Introduced in Future Lessons` to make slight references to other topics we will learn during this course. To stay focused, specify only the ones that are present in this current lesson.
-  **Section length:** 200 words


## Article Code

Links to code that will be used to support the article. Always prioritize this code over every other piece of code found in the sources: 

1. [Notebook 1](https://github.com/towardsai/agentic-ai-engineering-course/blob/dev/lessons/09_memory_knowledge_access/notebook.ipynb)

## Golden Sources

1. [What is the perfect memory architecture?](https://www.youtube.com/watch?v=7AmhgMAJIT4&list=PLDV8PPvY5K8VlygSJcp3__mhToZMBoiwX&index=112)
2. [Memory: The secret sauce of AI agents](https://decodingml.substack.com/p/memory-the-secret-sauce-of-ai-agents)
3. [Memory in Agent Systems](https://www.newsletter.swirlai.com/p/memory-in-agent-systems)
4. [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/html/2504.19413)
5. [Memex 2.0: Memory The Missing Piece for Real Intelligence](https://danielp1.substack.com/p/memex-20-memory-the-missing-piece)

## Other Sources

1. [What is AI agent memory?](https://www.ibm.com/think/topics/ai-agent-memory)
2. [Cognitive Architectures for Language Agents](https://arxiv.org/html/2309.02427)
3. [Agent Memory](https://docs.letta.com/guides/agents/memory)
4. [Memory](https://langchain-ai.github.io/langgraph/concepts/memory/)
5. [Giving Your AI a Mind: Exploring Memory Frameworks for Agentic Language Models](https://medium.com/@honeyricky1m3/giving-your-ai-a-mind-exploring-memory-frameworks-for-agentic-language-models-c92af355df06)