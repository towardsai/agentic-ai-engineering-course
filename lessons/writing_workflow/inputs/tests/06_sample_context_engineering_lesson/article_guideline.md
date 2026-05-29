## Global Context of the Lesson

### What We Are Planning to Share

We will write a lesson on context engineering, the core foundation of AI engineering and building AI applications. We will start by explaining why prompt engineering is not enough, and then explain why it is no longer sufficient in the world of AI. Next, we will slowly introduce the idea of context engineering, starting with a general overview and how it differs from prompt engineering. Then we will present what makes up the context passed to an LLM, key challenges in production, and tools and solutions for context engineering. Ultimately, we want to connect context engineering to the broader AI engineering field.

### Why We Think It's Valuable

Context engineering is the new fine-tuning. As fine-tuning is required less and less due to the fact that it is expensive, slow, and extremely inflexible in a world where data keeps changing, fine-tuning becomes the last resort when building AI applications. Thus, context engineering becomes a core skill for building successful AI agents or LLM workflows that manage the short-term memory and long-term memory of AI applications to achieve the best performance possible.

### Expected Length of the Lesson
**3000 words** (without the titles and references), where we assume that 200-250 words ≈ 1 minute of reading time.

### Theory / Practice Ratio

70% theory - 30% real-world examples

## Anchoring the Lesson in the Course

### Point of View
The course is created by a team writing for a single reader, also known as the student. Thus, for voice consistency across the course, we will always use 'we,' 'our,' and 'us' to refer to the team who creates the course, and 'you' or 'your' to address the reader. Avoid singular first person and don't use 'we' to refer to the student.

Examples of correct point of view:
- Instead of "Before we can choose between workflows and agents, we need a clear understanding of what they are." word it as "To choose between workflows and agents, you need a clear understanding of what they are."

### Lesson Scope

This is the 3rd lesson from module 1 of a broader course on AI agents and LLM workflows.

### Who Is the Intended Audience

Aspiring AI engineers who are learning about context engineering for the first time.

### Concepts Introduced in Previous Lessons

In previous lessons, we introduced the following concepts:
- What AI agents are at a high-level (NOT how they work) (Lesson 2)
- What LLM workflows are at a high-level (NOT how they work) (Lesson 2)
- How to choose between AI agents and LLM workflows when designing your AI application (Lesson 2)

As this is only the 2nd lesson of the course, we haven't introduced too many concepts. At this point, the reader only knows what an LLM is and a few high-level ideas about the LLM workflows and AI agents landscape.

### Concepts That Will Be Introduced in Future Lessons

In future lessons of the course, we will introduce the following concepts:
- structured outputs (Lesson 4)
- chaining (Lesson 5)
- routing (Lesson 5)
- orchestrator-worker (Lesson 5)
- tools (Lesson 6)
- ReAct agents (Lesson 7 and 8)
- Plan-and-Execute agents (Lesson 7)
- short-term memory (Lesson 9)
- long-term memory (Lesson 9)
    - procedural long-term memory
    - semantic long-term memory
    - episodic long-term memory
- RAG (Lesson 10)
- multimodal LLMs (Lesson 11)
- evaluations
- MCP

As context engineering is the core foundation of AI engineering, we will have to introduce new terms, but we will discuss them in a highly intuitive manner, being careful not to confuse the reader with too many terms that haven't been introduced yet in the course.

### Course Instructions
When discussing the **concepts introduced in previous lessons** listed in the `Concepts Introduced in Previous Lessons` section, avoid reintroducing them to the reader. Especially don't reintroduce the acronyms. Use them as if the reader already knows what they are. 

Avoid using all the **concepts that haven't been introduced in previous lessons** listed in the `Concepts That Will Be Introduced in Future Lessons` subsection. Whenever another concept requires references to these banned concepts, instead of directly using it, use other intuitive and grounded explanations as you would explain them to a 7-year-old. For example:
- If the "tools" concept wasn't introduced yet and you have to talk about agents, refer them to as "actions".
- If the "routing" concept wasn't introduced yet and you have to talk about it, refer it to as "guiding the workflow between multiple decisions".
You can use the concepts that haven't been introduced in previous lessons listed in the `Concepts That Will Be Introduced in Future Lessons` subsection, only if we explicitly specify them. Still, even in that case, as the reader doesn't know how that concept works, you are just allowed to use the term, while keeping the explanation extremely high-level and intuitive, as if you were explaining it to a 7-year-old.
Whenever you use a concept from the `Concepts That Will Be Introduced in Future Lessons` subsection explicitly specify that it will be explained in more detail in future lessons.

In all use cases avoid using acronyms that aren't explicitly stated in the guidelines. Rather use other more accessible synonyms or descriptions that are easier to understand by non-experts.

## Narrative Flow of the Lesson

Follow the next narrative flow when writing the end-to-end lesson:

- What problem are we solving? Why is it essential to solve it?
	- Start with a personal story where we encountered the problem
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

1. When prompt engineering breaks
2. From prompt to context engineering
3. Understanding context engineering
4. What makes up the context
5. Production implementation challenges 
6. Key strategies for context optimization
7. Here is an example
8. Connecting context engineering to AI engineering

## Section 1 - Introduction: When prompt engineering breaks
(The problem.)

- Start the lesson with a short story on the evolution of AI applications:
    - Chatbots (2022): Simple question-and-answer interfaces
    - RAG Systems (2023): Domain-specific knowledge integration
    - Tool-Using Agents (2024): LLMs with function calling capabilities
    - Memory-Enabled Agents (2025 - Now): Stateful, relationship-building systems

- **Quick reference to what we've learned in previous lessons:** Take the core ideas of what we've learned in previous lessons from the `Concepts Introduced in Previous Lessons` subsection of the `Anchoring the Lesson in the Course` section.
- **Transition to what we'll learn in this lesson:** After presenting what we learned in the past, make a transition to what we will learn in this lesson. Take the core ideas of the lesson from the `What We Are Planning to Share` subsection and highlight the importance and existence of the lesson from the `Why We Think It's Valuable` subsection of the `Global Context of the Lesson` section.

- As AI applications grew into complex AI agents and LLM workflows, unlike prompt engineering, which focuses on single LLM calls and individual prompts, context engineering orchestrates the entire end-to-end ecosystem, making sure the LLM gets the right information at the right time.
- Explain that due to the current scale of AI applications, the data we have to manage grew exponentially, which directly reflects in the size of the input passed to the LLMs, which, overly simplified, is also known as the context. 

- **Section length:** 200 words

## Section 2: From prompt to context engineering
(Why it's important to solve it. Current solutions and why they are not ok.)

- Issues with prompt engineering:
    - Single-interaction focus: Optimized for individual interactions rather than sustained, multi-turn conversations. The context is relatively small.
    - Context decay: As context starts to grow exponentially, the LLM becomes more and more confused, not knowing what to focus on and providing hallucinations or misguided answers.
    - The context window challenge: Even if the LLM knows how to pick the right information from the context, the context window (intuitively known as the `RAM` of the LLM) is limited.
    - Costs and latency: Every token makes LLM inference slower and more expensive to run. Thus, the naive idea of throwing everything into the LLM context quickly becomes a bad approach.
- Mention that these concepts will be taught in more detail in the following lessons of the course, such as Lesson 9 on memory and Lesson 10 on RAG.
- Real-world example: In one of our previous projects, we tried to add everything into the context window of the LLM. As it supported windows up to 2 million tokens, we thought, "What could go wrong?" We stuffed everything—our research, intentions, guidelines, examples, reviews, etc. The result? An LLM workflow that takes 30 minutes to run and produces low-quality outputs.
- That's where context engineering kicks in. It addresses these limitations by treating AI applications not as a series of isolated prompts, but as systems that operate through dynamic context gathered from past conversations, databases, tools, and other types of memory. Thus, as AI Engineers, our job is to keep only what's essential in the context when we pass it to the LLM, making it accurate, fast, and cost-effective.
-  **Section length:** 250 words

## Section 3: Understanding context engineering
(At a theoretical level, explain our solution or transformation.)

- Definition: Context engineering is about finding the best way to arrange parts of your memory into the context that's passed to the LLM to squeeze out the best results out of it. It's a solution to an optimization problem in which you have to retrieve the right parts of both your short and long-term memory to solve a specific task without overwhelming the LLM.
- **Example:** When asking a cooking agent about a recipe, instead of passing the whole cookbook to the agent, we retrieve just the information about that recipe, together with personal preferences, such as allergies or taste preferences (e.g., more salty).

- Analogy: `Context as the AI's "RAM"`: "LLMs are like a new kind of operating system where the model is the CPU and its context window is the RAM. Just as an operating system curates what fits into RAM, context engineering manages what information occupies the model's limited context window." - Quote by Andrej Karpathy

- **Prompt engineering vs. context engineering:** Context engineering is not replacing prompt engineering. Instead, prompt engineering is a subset of context engineering. You still work with prompts. Thus, learning how to write them effectively is still a critical skill. But on top of that, it's important to know how to incorporate the right context into the prompt without compromising the LLM's performance.

- Table on `Prompt Engineering` vs. `Context Engineering`. Render it in Markdown.
| Dimension | Prompt Engineering | Context Engineering |
|-----------|-------------------|---------------------|
| Scope | Single interaction optimization | Entire information ecosystem |
| State Management | Stateless function | Stateful due to memory |
| Focus | How to phrase tasks | What information to provide |

- **Context engineering vs. fine-tuning:** Context engineering is the new fine-tuning. In most use cases, you can go far just by leveraging context engineering techniques. As modern LLMs generalize really well, and because fine-tuning is time-consuming and costly, fine-tuning should always be the last resort if nothing else works.
- When starting a new AI project and deciding what key strategy to use to guide the LLM to answer correctly, this is how your decision-making should look, from easy to hard:
    1. Prompt Engineering - Does it solve the problem? Yes. Stop.
    2. Context Engineering: Does it solve the problem? Yes. Stop.
    3. Fine-tuning: Can you make a fine-tuning dataset? Yes. Stop. No. Reframe the problem.
- Mermaid diagram with the workflow from above.
- **Example:** When processing Slack messages from your company, it's sufficient to use a reasoning LLM as the core of the agent and various mechanisms to retrieve specific Slack messages and take actions based on them, such as creating action points or writing emails. Fine-tuning the LLM on writing emails most of the time would be a waste of resources.
- Make a reference to the course explaining that within this course we will show you how to solve most industry use cases using the power of context engineering.

-  **Section length:** 450 words (without the mermaid diagram)

## Section 4: What makes up the context
(Go deeper into the advanced theory.)

- To better understand what contenxt engineering is, let's look at the core elements that built up the context. 
- To anchor the reader into previous techniques such as prompt engineering, better explain how the context is connected to the prompt template and prompt by working them through the high-level workflow:  User Input -> Long-term Memory -> Short-Term Working Memory -> Context -> Prompt Template -> Prompt -> LLM Call -> Answer -> Short-Term Working Memory -> Long-Term Memory -> Repeat
- Along with explaining the steps from above, add a mermaid diagram to support the idea through an illustration
- As these concepts haven't been introduced in the course yet, we will present them at an intuitive 7-year-old level.
- With that in mind, let's present all the core components that can come up when building a single-turn prompt that's passed to the LLM. They can be grouped in the following categories:
    - Short-term working memory, which is often referred to as the state of the agent or workflow, which can contain:
        - user input
        - message history
        - the agent's internal thoughts
        - tool call and outputs
    - Long-term memory, which is usually divided into:
        - procedural long-term memory: What's encoded directly in the code, such as:
            - The system prompt
            - Tools
            - Structured output schemas 
        - episodic long-term memory: Used to remember past user preferences or experiences. Usually persisted in vector or graph databases.
        - semantic long-term memory: Information retrieved from your internal knowledge base (vector, SQL, document or graph databases) or external knowledge base (the open internet through API calls, MCP tools, or web scrapers)
    - Tell the reader that in case they didn't understand everything to "bear with us", as we will learn all these concepts in depth in future lessons, such as structured outputs in Lesson 4, tools in Lesson 6, memory in Lesson 9, RAG in Lesson 10, and working with multimodal data in Lesson 11.
- Add an image from the research illustrating what makes up the context of an AI agent 
- Highlight that even if we talk about what's passed to the LLM in a single turn, most of the elements, such as the conversation history, are preserved across turns within the memory. 
- Connect things together by explaining that all of these are not static components, but are dynamically re-computed on each call or turn. For each conversation turn or new task, the short-term memory grows or the long-term memory can change. A big part of context engineering is knowing how to pick the right components from the memory when building the prompt that's passed to the LLM.
-  **Section length:** 450 words

## Section 5: Production implementation challenges 
(Go deeper into the advanced theory)

- Transition from presenting what context engineering is to what are the core challenges when implementing it in AI agents and LLM workflow solutions.
- All the challenges revolve around a single question: "How can I keep my context as small as possible, while providing enough information to the LLM?"
- Now, we will present four of the most common issues that come up when building AI applications: 
    1. **The context window challenge:** Every AI model has a limited context window, which is the maximum amount of information (tokens) it can process simultaneously. This is similar to your computer's RAM. If you have only 32GB RAM on your machine, that's all you can process at a point in time. 
    2. **Information overload:** Too much context reduces the performance of the LLM by confusing it. This is known as the "lost-in-the-middle" or "needle in the haystack" problem, where LLMs are well-known for mostly remembering what is at the top and bottom of the context window. Processing what's in the middle is always a lottery.
    3. **Context drift:** Conflicting views of truth over time. For example, you can have conflicting statements about the same concept such as "My cat is white" and "My cat is black." This is not quantum physics or the Schrödinger Cat experiment, but it confuses the LLM and prevents it from knowing what to pick.
    4. **Tool confusion:** It can arise in two core scenarios. First, if we add too many tools to an AI agent or workflow (e.g., implementing the orchestrator-worker pattern), it will start confusing the LLM about what is the best tool for the job. Usually this starts with 100+ tools. Secondly, it can appear when the tools' descriptions are poorly written or there are unclear separations between them. If the descriptions are not clearly separated or have overlaps between them, it's a recipe for disaster. In that case, even a human wouldn't know what to pick.

-  **Section length:** 350 words

## Section 6: Key strategies for context optimization
(Go deeper into the advanced theory)

- As stated in the introduction, at the beginning, most AI apps were chatbots over single knowledge bases. But for most AI applications today, this is no longer the case. Modern AI solutions require access to multiple knowledge bases and tools. Context engineering is all about managing this complexity while staying within the desired performance, latency, and cost requirements. 

- Get more hands-on and present four of the most popular context engineering strategies used across the industry: 
    1. **Selecting the right context:** The art of retrieving the right information from the memory as context to solve a given task. A common mistake we see people make when building AI products is providing everything into the context at once. Often, their reasoning is that if they work with models that can handle up to 2 million input tokens, the model can handle all that input. But as mentioned in the previous section, due to the "lost-in-the-middle" problem this often results in poor performance. Also, it translates to increased latency and costs. To solve this:
        1. Use structured outputs to separate different parts of the LLM outputs and pass downstream only what is required (more on this in Lesson 4) 
        2. Use RAG to pass only the factual facts required to answer a given user question (more on this in Lesson 10.)
        3. Reducing the number of available tools to avoid confusing the LLM what actions to take at each step.
        (Create a mermaid diagram that combines techniques 1, 2, 3, 4 and 5 to show how they could work together in a bigger system.)
        4. Temporal Relevance: Ranking time-sensitive data and cutting off irrelevant data points
        5. Repeat core instructions at both the start and the end. Unintuiteveliy, for the most important instructions, it's recommended to repeat yourself across the prompt, even if it translates into more tokens.  
    2. **Context compression:** As the message history grows in the short-term working memory, you have to carefully manage past interactions to keeping your context window in check. The trick is that you cannot just drop past conversation turns, as the LLM still needs to remember what happen. Thus, we need ways to compress key facts from the past, while shrinking the short-term memory. We can do that through:
        1. Creating summaries of past interactions using an LLM
        2. Moving preferences about the user, from the working memory into the long-term memory, which is most often labeled as the episodic memory 
        (create a mermaid diagram to support ideas from point 1 and 2)
        3. Deduplication to avoid repetition
    3. **Isolating Context:**
        - Splitting information across multiple agents or LLM workflows.
        - Create a mermaid diagram showing how we can leverage the orchestrator-worker pattern to do this.
    4. **Format optimization for model clarity:** For example, using XML to clearly structure different parts of the prompt. Also, using YAML instead of JSON when inputting structured data to the LLM, as YAML is more token efficient than JSON.

- **Conclusion:** You always have to understand what is passed to the LLM. Seeing exactly what occupies your context window at every step is key to mastering context engineering. Usually this is done by properly monitoring your traces, tracking what happens at each step, understanding what the inputs and outputs are. As this is a significant step to go from PoC to production, we will have dedicated lessons on this.

-  **Section length:** 600 words

## Section 7: Here is an example
(An example)

- Connect the dots between the theory, challenges, and optimization strategies through some concrete examples.
- Some real-world use cases that often require keeping the context into memory between multiple conversation turn or user sessions:
    - Healthcare: AI systems that have access to patient data, patient history, current symptoms, and medical literature to enable more informed and personalized diagnoses
    - Financial Services: AI systems that have access to enterprise infrastructure such as CRMs, Slack, Zoom, Calendars, and financial data and make financial decisions based on user preferences
    - Project Task Managers: AI systems that have access to enterprise infrastructure such as CRMs, Slack, Zoom, Calendars, and Task managers to automatically understand project requirements, add and update project tasks
    - Content Creator Assistant: AI agent that has access to your research, past content, personality traits, and lesson guidelines to understand what and how to create the given piece of content

- Example Query: `I have a headache. What can I do to stop it? I would prefer not to take any medicine.` - Before the AI sees the customer's question, the system:
    - Retrieves the customer's patient history, life habits, and personal preferences from the episodic memory
    - Reviews modern and tested medical literature aligned with the patient from the semantic memory
    - Extracts key units of information from the memory into context required to answer the question using various tools
    - Formats this information optimally into a prompt and calls the LLM
    - Presents the personalized answer to the user
- Provide a system prompt example in Python, showing how all these elements would look like in the prompt. Use XML to format the context elements. Highlight the order of each element from the prompt. Specify how we used XML to differentiate pieces from the context.

- Here is a potential tech stack that we recommend which could be used to implement all these use cases. Most of these tools will be used across the course:
    - Gemini as multimodal, reasoning, cheap LLM API provider
    - LangGraph as the orchestrator and memory virtual layer
    - PostgreSQL, MongoDB, Redis, Qdrant, Neo4j as databases (Add a note that often it's recommended to keep it simple as you can get super far with only PostgreSQL or MongoDB.)
    - Opik or LangSmith for observability, such as evaluation and monitoring

-  **Section length:** 450 words (without the code example)

## Section 8 - Conclusion - Wrap-up: Connecting context engineering to AI engineering
(Connect our solution to the bigger picture and next steps)

- Context engineering is more of an art than a science. It's about developing the intuition to craft effective prompts, select the right information from memory, and arrange context for optimal results. This discipline helps you determine the minimal yet essential information an LLM needs to perform at its best.
- Context engineering cannot be learned in isolation, as it's a complex field that combines:
    1. **AI Engineering:**  Implement practical solutions such as LLM workflows, RAG, AI Agents, and evaluation pipelines.
    2. **Software Engineering (SWE):** Build your AI product with code that is not just functional, but also scalable and maintainable, and design architectures that can grow with your product's needs.
    3. **Data Engineering:** Design data pipelines that feed curated and validated data into the memory layer.
    4. **Operations (Ops):** Deploy agents on the proper infrastructure to ensure they are reproducible, maintainable, observable, and scalable, including automating processes with CI/CD pipelines.
- Our goal with this course is to teach you how to combine these skills to build production-ready AI products. In the world of AI, we should all think in systems rather than isolated components, having a mindset shift from developers to architects.
- To transition from this lesson to the next, specify what we will learn in future lessons. First mention what we will learn in next lesson, which is Lesson 4. Next leverage the concepts listed in subsection `Concepts That Will Be Introduced in Future Lessons` to make slight references to other topics we will learn during this course. To stay focused, specify only the ones that are present in this current lesson.

-  **Section length:** 250 words

## Golden Sources

1. [Context Engineering](https://blog.langchain.com/context-engineering-for-agents/)
2. [Context Engineering - What it is, and techniques to consider](https://www.llamaindex.ai/blog/context-engineering-what-it-is-and-techniques-to-consider)
3. [The rise of "context engineering"](https://blog.langchain.com/the-rise-of-context-engineering/)
4. [Context Engineering: A Guide With Examples](https://www.datacamp.com/blog/context-engineering)
5. [A Survey of Context Engineering for Large Language Models](https://arxiv.org/pdf/2507.13334)

## Other Sources

1. [+1 for "context engineering" over "prompt engineering".](https://x.com/karpathy/status/1937902205765607626)
2. [Context Engineering 101 cheat sheet](https://x.com/lenadroid/status/1943685060785524824)
3. [Own your context window](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-03-own-your-context-window.md)
4. [Context Engineering Guide](https://nlp.elvissaravia.com/p/context-engineering-guide)
5. [What is Context Engineering?](https://www.pinecone.io/learn/context-engineering/)
