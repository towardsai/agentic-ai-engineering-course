## Global Context of the Lesson

- **Scope:** The 3rd lesson from module 1 of a broader course on AI agents and LLM workflows.
- **What I'm planning to share**: I will write a lesson on context engineering, the core foundation of AI engineering and building AI applications. We will start by explaining why prompt engineering is not enough, and then explain why it is no longer sufficient in the world of AI. Next, I will slowly introduce the idea of context engineering, starting with a general overview and how it differs from prompt engineering. Then we will present what makes up the context passed to an LLM, key challenges in production, and tools and solutions for context engineering. Ultimately, I want to connect context engineering to the broader AI engineering field.
- **Why I think it's valuable:** Context engineering is the new fine-tuning. As fine-tuning is required less and less due to the fact that it is expensive, slow, and extremely inflexible in a world where data keeps changing, fine-tuning becomes the last resort when building AI applications. Thus, context engineering becomes a core skill for building successful AI agents or LLM workflows that manage the short-term memory and long-term memory of AI applications to achieve the best performance possible.
Aspiring AI engineers who are learning about context engineering for the first time. In previous lessons, we introduced the concepts of AI agents and LLM workflows. Only in future lessons will we discuss structured outputs, chaining, routing, the task-executor pattern, tool calling, ReAct agents, memory, RAG, and multimodal models. When discussing concepts introduced in previous lessons, avoid reintroducing them to the reader. Use them as if the reader already knows what they are. When discussing concepts that haven't been introduced in previous lessons, try to avoid using them altogether. If the guideline specifies them, keep them at an extremely high level, as if explaining them to a 7-year-old, while avoiding the use of acronyms that aren't explicitly stated in the guidelines. As context engineering is the core foundation, we will have to introduce new terms, but we should discuss them in a highly intuitive manner. 
- **Theory / Practice ratio:** 90% theory / 10% practice
- **Expected length of the article in words** (where 200-250 words ~= 1 minute of reading time): 2450 words


## Narrative Flow of the Lesson

Follow the next narrative flow when writing the end-to-end lesson:

- What problem are we solving? Why is it essential to solve it?
- Why other solutions are not working and what's wrong with them.
- At a theoretical level, explain our solution or transformation. Highlight:
    - The theoretical foundations.
    - Why is it better than other solutions?
    - What tools or algorithms can we use?
- Go deeper into the advanced theory.
- Provide a more complex example supporting the advanced theory.
- Connect our solution to the bigger picture and next steps.


## Lesson Outline 

1. From prompt to context engineering
2. Understanding context engineering
3. What makes up the context
4. Production implementation challenges 
5. Key strategies for context optimization
6. Here is an example
7. Connecting context engineering to AI engineering


## Section 1 - Introduction: From prompt to context engineering
(The problem. Why it's important to solve it. Current solutions and why they are not ok.)

- Unlike prompt engineering, which focuses on single LLM calls and individual prompts, context engineering orchestrates the entire information ecosystem around AI agents or LLM workflows.
- Explain why prompt engineering is not enough anymore with analogies to the evolution of LLMs and AI applications:
    - Chatbots (2022): Simple question-and-answer interfaces
    - RAG Systems (2023): Domain-specific knowledge integration
    - Tool-Using Agents (2024): LLMs with function calling capabilities
    - Memory-Enabled Agents (2025 - Now): Stateful, relationship-building systems
- As a side note, make sure to mention that these concepts will be taught in more detail in the following lessons.
- Issues with prompt engineering:
    - Single-interaction focus: Optimized for individual interactions rather than sustained, multi-turn conversations. The context is relatively small.
    - Context decay: As context starts to grow exponentially, the LLM becomes more and more confused, not knowing what to focus on (due to the needle in the haystack problem) and providing hallucinations or misguided answers.
    - The context window challenge: Even if the LLM knows how to pick the right information from the context, the context window (intuitively known as the `RAM`/short-term memory/working memory) is limited. We know from practice that 2 million token context inputs to LLMs (that might contain irrelevant information) perform worse than small, focused context inputs.
    - Costs and latency: Every token makes LLM inference slower and more expensive to run. Thus, the naive idea of throwing everything into the LLM context (known as CAG) quickly becomes a bad approach.
- Real-world example: In one of my previous projects, I tried to add everything into my context window: my research, intentions, guidelines, examples, reviews, etc. The result? An LLM workflow that takes 30 minutes to run and produces output that wasn't of high quality.
- Context engineering addresses these limitations by treating AI applications not as a series of isolated prompts, but as systems that operate through dynamic context gathered from past conversations, databases, tools, and other types of memory. Thus, as AI Engineers, our job is to keep only what's essential in the context when we pass it to the LLM, making it accurate, fast, and cost-effective.

-  **Section length:** 400 words

## Section 2: Understanding context engineering
(At a theoretical level, explain our solution or transformation.)

- The complicated answer: "Context engineering is formally defined as the optimization problem of assembling the right information at the right time to get the right answer from an LLM: `Context* = argmax E[Reward(LLM(context), target)]`"
- The easy answer: "Context engineering is about finding the best way to arrange parts of the short and long-term memory into the context for the best results. It's a problem in which you have to retrieve the right parts of both your short and long-term memory to solve a specific task without overwhelming the LLM." More precisely, you have to pick the right units of knowledge from the memory to optimize all the input tokens that are passed to the LLM at every step, such as the instructions and message history of the agent.
- **Example:** When asking a cooking agent about a recipe, instead of passing the whole cookbook to the agent, we retrieve just the information about that recipe, together with personal preferences, such as allergies or taste preferences (e.g., more salty).

- Analogy: `Context as the AI's "RAM"`: "LLMs are like a new kind of operating system where the model is the CPU and its context window is the RAM. Just as an operating system curates what fits into RAM, context engineering manages what information occupies the model's limited context window." - Quote by Andrej Karpathy
- Analogy to memory: The context is a subset of the short-term working memory that is passed to the LLM. It's not the whole short-term memory, as you can keep other aspects in your working memory without using them in the prompt passed to the LLM.

- Prompt engineering vs. context engineering: Context engineering will not replace prompt engineering. Instead, prompt engineering is becoming part of context engineering. You still need to learn how to write effective prompts/instructions while understanding how to incorporate the right context into the prompt without compromising the LLM's performance.
- Table on `Prompt Engineering` vs. `Context Engineering`. Render it in Markdown.
| Dimension | Prompt Engineering | Context Engineering |
|-----------|-------------------|---------------------|
| Scope | Single interaction optimization | Entire information ecosystem |
| Complexity | O(1) context assembly | O(n) multi-component optimization |
| State Management | Stateless function | Stateful with memory |
| Focus | How to phrase tasks | What information to provide |

- Context engineering vs. fine-tuning: Context engineering is the new fine-tuning. In most use cases, you can go far just by leveraging context engineering techniques. As modern LLMs generalize really well, and because fine-tuning is time-consuming and costly, it should always be the last resort if nothing else works.
- When solving a specific problem, this is how your decision-making should look, from easy to hard:
    1. Prompt Engineering: Does it solve my problem? Yes. Stop here.
    2. Context Engineering: Does it solve my problem? Yes. Stop here.
    3. Fine-tuning: Can you make a fine-tuning dataset? Yes. Stop here. No. Reframe the problem.
- **Example:** When processing Slack messages from your company, it's sufficient to use a reasoning LLM as the core of the agent and various mechanisms to retrieve specific Slack messages and take actions based on them, such as creating action points or writing emails.
- Mermaid diagram with the workflow from above.
- Within this course, we will show you how to solve most industry use cases using only context engineering.

-  **Section length:** 500 words (without the mermaid diagram)

## Section 3: What makes up the context
(Go deeper into the advanced theory.)

- Essentially, all the input tokens that are passed to the LLM at once in a single prompt constitute the context, which can be grouped into the following core components that build up the context:
    - The system prompt (procedural long-term memory)
    - Agent's state (short-term working memory): user inputs, message history, ReAct internal components (thought/internal chatter, action/tool call, observation/tool result)
    - User preferences or past experiences (episodic memory): state, vector or graph databases
    - Information retrieved from our internal knowledge base (semantic memory) or external knowledge base (real-time environment factors): vector, SQL, document or graph database OR any other API call or MCP tools
    - Tool schemas (procedural memory)
    - Structured output schemas (procedural memory)
- Add an image from the research illustrating what makes up the context.
- Highlight that all of these are not static prompts, but are dynamically computed on each conversation turn. For each user query, we have to adapt our query, the conversation history grows, and the user preferences from the episodic memory change.
- Explain how things are connected: User query or task -> Long-term memory -> Short-term working memory -> Context -> LLM Call -> Answer -> Short-term working memory -> Long-term memory -> Repeat
- Add a mermaid diagram illustrating the steps from above.

-  **Section length:** 300 words

## Section 4: Production implementation challenges 
(Go deeper into the advanced theory)

- The context window challenge: Every AI model has a limited context window, which is the maximum amount of information (tokens) it can process simultaneously. Even with recent advances reaching millions of tokens, this space fills quickly.
- Information overload or context decay: Too much context reduces the performance of the LLM by confusing it.
- Context drift: Conflicting views of truth over time. For example, you can have conflicting statements about the same concept such as "My cat is white" and "My cat is black." This is not quantum physics or the Schrödinger Cat experiment, but it confuses the LLM and prevents it from knowing what to pick.
- Tool confusion: Adding too many tools with poor descriptions and unclear separations between them confuses the LLM about which one to pick, resulting in failing AI agents.

-  **Section length:** 200 words

## Section 5: Key strategies for context optimization
(Go deeper into the advanced theory)

- In the early days (2022-2023), most AI apps were chatbots and simple RAG systems designed to do question answering over a single knowledge base, often a vector store or other form of database. But for most AI applications today, this is no longer the case. We now see applications that need access to multiple knowledge bases by manipulating multiple tools that can either return more context or perform certain tasks. Context engineering helps us manage this process.

- Selecting the right context: The process of matching information from memory to current needs in the context to solve a given task. A common mistake we see people make when creating agentic AI systems is often providing all the context when it simply isn't required, as it can potentially overcrowd the context limit when it's not necessary. To solve this:
    - Use structured outputs to separate different parts of the LLM outputs and pass downstream only what is required.
    - Avoid CAG and use RAG + ranking to pass only key facts to the LLM.

- Context compression: As the message history grows in the short-term working memory, we have to carefully manage past interactions to avoid overflowing our context window (add funny analogy to memory overflowing from RAM systems), while allowing the agent or LLM workflow to recall past facts. We can do that through:
    - Creating summaries of past interactions using an LLM
    - Moving the working memory into the long-term memory, which is most often the episodic memory (adding key factors back into our knowledge base)
    - Deduplication to avoid repetition

- Context ordering - Note that LLMs usually recall best the first part of the prompt and especially the last part of the prompt, where everything in the middle is lost—it's a no man's land:
    - Temporal Relevance: Prioritizing recent or time-sensitive data and cutting off the rest
    - Reranking techniques adopted from RAG and information retrieval

- Isolating Context: Splitting information across multiple agents or LLM workflows.

- Format optimization for model clarity. For example, using XML and YAML to clearly structure different parts of the prompt.

- Do not let frameworks abstract away the context from you—own the context! Seeing exactly what is passed to the LLM is key to understanding the context engineering process at every step of the LLM app.

- Add an image from the research illustrating some key strategies for context optimization.

-  **Section length:** 450 words

## Section 6: Here is an example
(An example)

- Other real-world use cases:
    - Healthcare: AI systems that have access to patient data, patient history, current symptoms, and medical literature to enable more informed and personalized diagnoses
    - Financial Services: AI systems that have access to enterprise infrastructure such as CRMs, Slack, Zoom, Calendars, and financial data and make financial decisions based on user preferences
    - Project Task Managers: AI systems that have access to enterprise infrastructure such as CRMs, Slack, Zoom, Calendars, and Task managers to automatically understand project requirements, add and update project tasks

- Example Query: `I have a headache. What can I do to stop it? I would prefer not to take any medicine.` - Before the AI sees the customer's question, the system:
    - Retrieves the customer's patient history, life habits, and personal preferences from the episodic memory
    - Reviews modern and tested medical literature aligned with the patient from the semantic memory
    - Extracts key units of information required to answer the question using various tools
    - Formats this information optimally into a prompt and calls the LLM
    - Presents the personalized answer to the user
    - Stops if true, repeats if user asks more questions
- Provide a system prompt example in Python, showing how all these elements would look like in the prompt. Highlight the order of each element from the prompt.

- Key tools:
    - Gemini as multimodal, reasoning, cheap LLM API provider
    - LangChain as the orchestrator
    - PostgreSQL, MongoDB, Redis, Qdrant, Neo4j as databases (P.S. You can get super far with only PostgreSQL or MongoDB)
    - mem0 for memory
    - Opik for observability

-  **Section length:** 350 words (without the code example)

## Section 7 - Conclusion - Wrap-up: Connecting context engineering to AI engineering
(Connect our solution to the bigger picture and next steps)

- Context engineering is more of an art than a science. It's the skill of building intuition on how to write prompts, how to pass the right information into the context, and in the right order. It's about answering questions such as "What are the best input tokens to pass to the LLM to solve a specific task and get the best results?" or "What does the model need to see?"
- We cannot learn context engineering in isolation, as it's a complex but beautiful discipline that combines:
    1. AI Engineering: LLMs, RAG, AI Agents
    2. SWE: Aggregates all the context elements into scalable and maintainable code. Design scalable architectures. Wrap the agents as APIs.
    3. Data Engineering: Building data pipelines for RAG or LLM Workflows.
    4. Ops: Deploy your agents on the right piece of infrastructure to make them reproducible, maintainable, observable, and scalable. Automate it. Write CI/CD pipelines.
    5. AI Research: Fine-tuning
- Next steps: The best way to level up your context engineering skills is to build AI agents or LLM workflows that combine the following principles:
    - RAG (Semantic long-term memory)
    - Tools (Procedural long-term memory)
    - User preferences (Episodic long-term memory)

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
