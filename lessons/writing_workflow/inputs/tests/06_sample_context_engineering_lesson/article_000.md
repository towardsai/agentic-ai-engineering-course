# Lesson 3: Context Engineering
### The Core Skill for Building Production-Ready AI

AI applications have evolved rapidly. In 2022, we had simple chatbots for question-answering. By 2023, Retrieval-Augmented Generation (RAG) systems connected LLMs to domain-specific knowledge. 2024 brought us tool-using agents that could perform actions. Now, we are building memory-enabled agents that remember past interactions and build relationships over time.

In our last lesson, we explored how to choose between AI agents and LLM workflows when designing a system. As these applications grow more complex, prompt engineering, a practice that once served us well, is showing its limits. It optimizes single LLM calls but fails when managing systems with memory, actions, and long interaction histories. The sheer volume of information an agent might need—past conversations, user data, documents, and action descriptions—has grown exponentially. Simply stuffing all this into a prompt is not a viable strategy. This is where context engineering comes in. It is the discipline of orchestrating this entire information ecosystem to ensure the LLM gets exactly what it needs, when it needs it. This skill is becoming a core foundation for AI engineering.

## From Prompt to Context Engineering

Prompt engineering, while effective for simple tasks, is designed for single, stateless interactions. It treats each call to an LLM as a new, isolated event. This approach breaks down in stateful applications where context must be preserved and managed across multiple turns.

As a conversation or task progresses, the context grows. Without a strategy to manage this growth, the LLM’s performance degrades. This is context decay: the model gets confused by the noise of an ever-expanding history. It starts to lose track of the original instructions or key information.

Even with large context windows, a physical limit exists for what you can include. Also, on the operational side, every token adds to the cost and latency of an LLM call. Simply putting everything into the context creates a slow, expensive, and underperforming system. We will explore these concepts in more detail in upcoming lessons, including memory in Lesson 9 and RAG in Lesson 10.

On a recent project, we learned this the hard way. We were working with a model that supported a million-token context window, so we thought, "*What could go wrong*?" We stuffed everything in: our research, guidelines, examples, and user history. The result was an LLM workflow that took 30 minutes to run and produced low-quality outputs.

This is where context engineering becomes essential. It shifts the focus from crafting static prompts to building dynamic systems that manage information flow. As an AI engineer, your job is to select only the most critical pieces of context for each LLM call. This makes your applications accurate, fast, and cost-effective.

## Understanding Context Engineering

Context engineering is about finding the optimal way to arrange information from your application's memory into the context passed to an LLM. It is a solution to an optimization problem where you retrieve the right parts from your short-term and long-term memory to solve a specific task without overwhelming the model. For example, when you ask a cooking agent for a recipe, you do not give it the entire cookbook. Instead, you retrieve the specific recipe, along with personal context like allergies or taste preferences. This precise selection ensures the model receives only the essential information.

Andrej Karpathy explains that LLMs are like a new kind of operating system where the model is the CPU and its context window is the RAM. Just as an operating system manages what fits into your computer’s limited RAM, context engineering manages what information occupies the model’s limited context window [[1]](https://addyo.substack.com/p/context-engineering-bringing-engineering).

Context engineering is not replacing prompt engineering. Instead, prompt engineering is a subset of context engineering. You still work with prompts, so learning how to write them effectively is still a critical skill. But on top of that, it is important to know how to incorporate the right context into the prompt without compromising the LLM's performance.

Table 1: A comparison of prompt engineering and context engineering.

| Dimension | Prompt Engineering | Context Engineering |
|-----------|-------------------|---------------------|
| Scope | Single interaction optimization | Entire information ecosystem |
| State Management | Stateless function | Stateful due to memory |
| Focus | How to phrase tasks | What information to provide |

Context engineering is the new fine-tuning. While fine-tuning has its place, it is expensive, time-consuming, and inflexible. Data changes constantly, making fine-tuning a last resort [[2]](https://www.tabnine.com/blog/your-ai-doesnt-need-more-training-it-needs-context/). For most enterprise use cases, you get better results faster and more cheaply with context engineering [[3]](https://www.tribe.ai/applied-ai/fine-tuning-vs-prompt-engineering). It allows for rapid iteration and adaptation to evolving data without altering the core model, a key advantage in dynamic environments. This approach avoids the computational resources and specialized expertise required for retraining, offering a more agile path to reliable AI applications.

When you start a new AI project, your decision-making process for guiding the LLM should look like the one presented in Image 1.

```mermaid
graph TD
    PE["Prompt Engineering"] --> D1{"Does it solve the problem?"}
    D1 -->|"Yes"| Stop["Stop"]
    D1 -->|"No"| CE["Context Engineering"]
    CE --> D2{"Does it solve the problem?"}
    D2 -->|"Yes"| Stop
    D2 -->|"No"| FT["Fine-tuning"]
    FT --> D3{"Can you make a fine-tuning dataset?"}
    D3 -->|"Yes"| Stop
    D3 -->|"No"| RP["Reframe the problem"]
```
Image 1: A flowchart illustrating the decision-making workflow for choosing between Prompt Engineering, Context Engineering, and Fine-tuning when starting a new AI project.

For instance, if you build an agent to process internal Slack messages, you do not need to fine-tune a model on your company’s communication style. It is more effective to use a powerful reasoning model and engineer the context to retrieve specific messages and enable actions like creating tasks or drafting emails. Throughout this course, we will show you how to solve most industry problems using only context engineering.

## What Makes Up the Context

To master context engineering, you first need to understand what "context" actually is. It is everything the LLM sees in a single turn, dynamically assembled from various memory components before being passed to the model. The high-level workflow, as presented in Image 2, begins when a user input triggers the system to pull relevant information from both long-term and short-term memory. This information is assembled into the final context, inserted into a prompt template, and sent to the LLM. The LLM’s answer then updates the memory, and the cycle repeats.

```mermaid
graph TD
    A["User Input"] --> B["Long-term Memory"]
    B --> C["Short-Term Working Memory"]
    C --> D["Context"]
    D --> E["Prompt Template"]
    E --> F["Prompt"]
    F --> G["LLM Call"]
    G --> H["Answer"]
    H --> C
    H --> B
    C --> D
```
Image 2: A flowchart illustrating the high-level workflow of how context is built and used in an LLM-driven AI system.

These components are grouped into two main categories. We will explain them intuitively for now, as we have future dedicated lessons for all of them.

### Short-Term Working Memory

Short-term working memory is the state of the agent for the current task or conversation. It is volatile and changes with each interaction, helping the agent maintain a coherent dialogue and make immediate decisions. It can include some or all of these components:

*   **User input:** The most recent query or command from the user.
*   **Message history:** The log of the current conversation, allowing the LLM to understand the flow and previous turns.
*   **Agent's internal thoughts:** The reasoning steps the agent takes to decide on its next action.
*   **Action calls and outputs:** The results from any actions the agent has performed, providing information from external systems.

### Long-Term Memory

Long-term memory is more persistent and stores information across sessions, allowing the AI system to remember things beyond a single conversation. We divide it into three types, drawing parallels from human memory [[4]](https://arxiv.org/html/2504.15965v1). An AI system can include some or all of them:

*   **Procedural memory:** This is knowledge encoded directly in the code. It includes the system prompt, which sets the agent's overall behavior. It also includes the definitions of available actions, which tell the agent what it can do, and schemas for structured outputs, which guide the format of its responses. Think of this as the agent's built-in skills.
*   **Episodic memory:** This is memory of specific past experiences, like user preferences or previous interactions. It's used to help the agent personalize its responses based on individual users. We typically store this in vector or graph databases for efficient retrieval.
*   **Semantic memory:** This is the agent’s general knowledge base. It can be internal, like company documents stored in a data lake, or external, accessed via the internet through API calls or web scraping. This memory provides the factual information the agent needs to answer questions.

If this seems like a lot, bear with us. We will cover all these concepts in-depth in future lessons, including structured outputs (Lesson 4), actions (Lesson 6), memory (Lesson 9), and RAG (Lesson 10).

https://github.com/user-attachments/assets/0f1f193f-8e94-4044-a276-576bd7764fd0 
Image 3: Context engineering encompasses a variety of techniques and information sources. (Source [humanlayer/12-factor-agents [5]](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-03-own-your-context-window.md))

The key takeaway is that these components are not static. They are dynamically re-computed for every single interaction. For each conversation turn or new task, the short-term memory grows, or the long-term memory can change. Context engineering involves knowing how to select the right pieces from this vast memory pool to construct the most effective prompt for the task at hand.

## Production Implementation Challenges

Now that we understand what makes up the context, let's look at the core challenges of implementing it in production. These challenges all revolve around a single question: *"How can I keep my context as small as possible while providing enough information to the LLM?"*

Here are four common issues that come up when building AI applications:

1.  **The context window challenge:** Every AI model has a limited context window, the maximum amount of information (tokens) it can process at once. Think of it like your computer's RAM. If your machine has only 32GB of RAM, that is all it can use at one time. While context windows are getting larger, they are not infinite, and treating them as such leads to other problems.
2.  **Information overload:** Just because you can fit a lot of information into the context does not mean you should. Too much context reduces the performance of the LLM by confusing it. This is known as the *"lost-in-the-middle"* or *"needle in the haystack"* problem, where LLMs are known for remembering information best at the beginning and end of the context window. Information in the middle is often overlooked, and performance can drop long before the physical context limit is reached [[6]](https://www.unite.ai/why-large-language-models-forget-the-middle-uncovering-ais-hidden-blind-spot/), [[7]](https://openreview.net/forum?id=5sB6cSblDR).
3.  **Context drift:** This occurs when conflicting views of truth accumulate in the memory over time [[8]](https://viso.ai/deep-learning/concept-drift-vs-data-drift/). For example, the memory might contain two conflicting statements: "*My cat is white*" and "*My cat is black*." This is not Schrodinger's Cat quantum physics experiment; it is a data conflict that confuses the LLM. Without a mechanism to resolve these conflicts, the model's responses become unreliable [[9]](https://erikjlarson.substack.com/p/context-drift-and-the-illusion-of).
4.  **Tool confusion:** The final challenge is tool confusion, which arises in two main ways. First, adding too many actions to an agent can confuse the LLM about the best one for the job, a problem that often appears with over 100 actions [[10]](https://support.talkdesk.com/hc/en-us/articles/39096730105115--Preview-AI-Agent-Platform-Best-Practices). Second, confusion can occur when tool descriptions are poorly written or overlap. If the distinctions between actions are unclear, even a human would struggle to choose the right one [[11]](https://www.forrester.com/blogs/the-state-of-ai-agents-lots-of-potential-and-confusion/).

## Key Strategies for Context Optimization

Initially, most AI applications were chatbots over single knowledge bases. Today, modern AI solutions must manage multiple knowledge bases, tools, and complex conversational histories. Context engineering is about managing this complexity while meeting performance, latency, and cost requirements.

Here are four popular context engineering strategies used across the industry:

### 1. Selecting the Right Context

Retrieving the right information from memory is a critical first step. A common mistake is to provide everything at once, assuming that models with large context windows can handle it. As we have discussed, the "lost-in-the-middle" problem often leads to poor performance, increased latency, and higher costs.

To solve this, consider these approaches:

*   **Use structured outputs:** Define clear schemas for what the LLM should return. This allows you to pass only the necessary, structured information to downstream steps. We will cover this in detail in Lesson 4.
*   **Use RAG:** Instead of providing entire documents, use RAG to fetch only the specific chunks of text needed to answer a user's question. This is a core topic we will explore in Lesson 10.
*   **Reduce the number of available actions:** Rather than giving an agent access to every available action, use various strategies to delegate action subsets to specialized components. For example, a typical pattern is to leverage the orchestrator-worker pattern to delegate subtasks to specialized agents. Studies show that limiting the selection to under 30 tools can triple the agent's selection accuracy [[12]](https://productschool.com/blog/artificial-intelligence/ai-agents-product-managers). Still, the ideal number of tools an agent can use can be highly dependent on what tools you provide, what LLM you use, and how well the actions are defined. That's why evaluating the performance of your AI system on core business metrics is a mandatory step that will help you pick the number. We will learn how to do this in future lessons.
*   **Rank time-sensitive data:** For time-sensitive information, rank it by date and filter out anything no longer relevant.
*   **Repeat core instructions:** For the most important instructions, repeat them at both the start and the end of the prompt. This uses the model's tendency to pay more attention to the context edges, ensuring core instructions are not lost [[13]](https://promptmetheus.com/resources/llm-knowledge-base/lost-in-the-middle-effect).

```mermaid
graph TD
    subgraph "AI System"
        A["Structured Outputs"]
        B["RAG (Retrieval Augmented Generation)"]
        C["Reduced Toolset"]
        D["Temporal Relevance"]
        E["Instruction Repetition"]

        A --> F["Optimized Context"]
        B --> F
        C --> F
        D --> F
        E --> F

        F --> G["Large Language Model (LLM)"]
    end
```
Image 4: An architecture diagram illustrating how various context optimization techniques can be combined within a larger AI system to create an optimized context for an LLM.

### 2. Context Compression

As message history grows in short-term working memory, you must manage past interactions to keep your context window in check. You cannot simply drop past conversation turns, as the LLM still needs to remember what happened. Instead, you need ways to compress key facts from the past.

You can do this through:

1.  **Creating summaries of past interactions:** Use an LLM to replace a long, detailed history with a concise overview.
2.  **Moving user preferences to long-term memory:** Transfer user preferences from working memory to long-term episodic memory. This keeps the working context clean while ensuring preferences are remembered for future sessions.
3.  **Deduplication:** Remove redundant information from the context to avoid repetition.

```mermaid
graph TD
    subgraph "Message History Compression"
        MH["Message History"] --> CC["Context Compression"]
        CC --> LLM["LLM (Summarization)"]
        LLM --> SC["Summarized Context"]
    end

    subgraph "User Preference Management"
        UP["User Preferences"] --> STWM["Short-term Working Memory"]
        STWM --> DEC{"Decision: Persist Preference?"}
        DEC -->|Yes| LTEM["Long-term Episodic Memory"]
        DEC -->|No| STWM
    end
```
Image 5: A flowchart demonstrating context compression strategies, including message history summarization and user preference memory management.

### 3. Isolating Context

Another powerful strategy is to isolate context by splitting information across multiple agents or LLM workflows. This technique is similar to tool isolation (explained in `Selecting the right context`), but it's more general, referring to the whole context. The key idea is that instead of one agent with a massive, cluttered context window, you can have a team of agents, each with a smaller, focused context.

We often implement this using an orchestrator-worker pattern, where a central orchestrator agent breaks down a problem and assigns sub-tasks to specialized worker agents [[14]](https://www.confluent.io/blog/event-driven-multi-agent-systems/). Each worker operates in its own isolated context, improving focus and allowing for parallel processing. We will cover this pattern in more detail in Lesson 5.

```mermaid
graph TD
    Orchestrator["Orchestrator"]

    subgraph "Worker Agent 1"
        Worker1["Worker Agent 1"]
        Context1["Isolated Context Window 1"]
        Worker1 -->|"Uses Context"| Context1
    end

    subgraph "Worker Agent 2"
        Worker2["Worker Agent 2"]
        Context2["Isolated Context Window 2"]
        Worker2 -->|"Uses Context"| Context2
    end

    subgraph "Worker Agent N"
        WorkerN["Worker Agent N"]
        ContextN["Isolated Context Window N"]
        WorkerN -->|"Uses Context"| ContextN
    end

    Orchestrator -->|"Delegates Task"| Worker1
    Orchestrator -->|"Delegates Task"| Worker2
    Orchestrator -->|"Delegates Task"| WorkerN
```
Image 6: Diagram illustrating the orchestrator-worker pattern for isolating context in multi-agent or LLM workflow solutions.

### 4. Format Optimizations

Finally, the way you format the context matters. Models are sensitive to structure, and using clear delimiters can improve performance. Common strategies are to:

*   **Use XML tags:** Wrap different pieces of context in XML-like tags (e.g., `<user_query>`, `<documents>`). This helps the model distinguish between different types of information, while making it easier for the engineer to reference context elements within the system prompt [[15]](https://milvus.io/ai-quick-reference/what-modifications-might-be-needed-to-the-llms-input-formatting-or-architecture-to-best-take-advantage-of-retrieved-documents-for-example-adding-special-tokens-or-segments-to-separate-context).
*   **Prefer YAML over JSON:** When providing structured data as input, YAML is often more token-efficient than JSON, which helps save space in your context window.

To conclude, you always have to understand what is passed to the LLM. Seeing exactly what occupies your context window at every step is key to mastering context engineering. Usually, this is done by properly monitoring your traces, tracking what happens at each step, and understanding what the inputs and outputs are. As this is a significant step to go from PoC to production, we will have dedicated lessons on this.

## Here is an example

Let's connect the theory and strategies discussed earlier with concrete examples. To do that, let's consider several common real-world scenarios:

*   **Healthcare:** An AI assistant accesses a patient's medical history, current symptoms, and the latest medical literature to provide personalized diagnostic support [[16]](https://www.akira.ai/blog/context-engineering).
*   **Financial Services:** AI systems integrate with enterprise tools like Customer Relationship Management (CRM) systems, emails, and calendars, combining real-time market data and client portfolio information to generate tailored financial advice and reports [[17]](https://66degrees.com/building-a-business-case-for-ai-in-financial-services/).
*   **Project Management:** AI systems access enterprise infrastructure like CRMs, Slack, and task managers to automatically understand project requirements, then add and update project tasks.
*   **Content Creator Assistant:** An AI agent uses your research, past content, and personality traits to understand what and how to create a given piece of content.

Let's walk through a specific query to see context engineering in action with the healthcare assistant scenario. A user asks: `I have a headache. What can I do to stop it? I would prefer not to take any medicine.`

Before the AI attempts to answer, a context engineering system performs several steps:

1.  It retrieves the user's patient history, known allergies, and lifestyle habits from episodic memory.
2.  It queries a medical database for non-pharmacological headache remedies from semantic memory.
3.  It uses various tools to assemble the key units of information from both memory types into the final context.
4.  It formats this information into a structured prompt and calls the LLM.
5.  Finally, it presents a personalized, context-aware answer to the user.

Here is a simplified Python example showing how you might structure the context and prompt for the LLM, using XML tags to format the different context elements and YAML to format all input data collections.

1.  First, we define the user's query.
    ```python
    import yaml
    
    user_query = "I have a headache. What can I do to stop it? I would prefer not to take any medicine."
    ```
2.  Next, we define the patient's history, which would typically be retrieved from episodic memory.
    ```python
    patient_history = {
        "patient": {
            "name": "John Doe",
            "age": 45,
            "gender": "M",
            "conditions": ["mild_hypertension"],
            "allergies": [],
            "preferences": {
                "medication_avoidance": True,
                "preferred_treatments": "natural_remedies"
            },
            "habits": {
                "stress_level": "high",
                "work_related": True,
                "caffeine_intake": "3-4_cups_daily"
            }
        }
    }
    ```
3.  Then, we include relevant medical literature, which would be retrieved from semantic memory.
    ```python
    medical_literature = {
        "articles": [
            {
                "id": 1,
                "topic": "dehydration_headaches",
                "finding": "Dehydration is a common cause of tension headaches",
                "treatment": "Rehydration can alleviate symptoms within 30 minutes to three hours"
            },
            {
                "id": 2,
                "topic": "cold_compress",
                "finding": "Applying a cold compress to the forehead and temples can constrict blood vessels",
                "treatment": "Reduces inflammation, helping to relieve migraine pain"
            },
            {
                "id": 3,
                "topic": "caffeine_withdrawal",
                "finding": "Caffeine withdrawal can trigger headaches",
                "treatment": "For regular caffeine consumers, a small amount may alleviate withdrawal headaches"
            },
            {
                "id": 4,
                "topic": "stress_relief",
                "finding": "Stress-relief techniques are effective for tension headaches",
                "treatment": "Deep breathing, meditation, or short walks can help"
            }
        ]
    }
    ```
4.  Finally, we assemble the complete prompt, combining all elements into a structured format. Notice how we format the patient history and medical literature as YAML, instead of passing them directly as plain Python dictionaries, which are equivalent to JSON.
    ```python
    prompt = f"""
    <system_prompt>
    You are a helpful AI medical assistant. Your role is to provide safe, helpful, and personalized health advice based on the provided context. Do not give advice outside of the provided context. Prioritize non-medicinal options as per user preference.
    </system_prompt>
    
    <patient_history>
    {yaml.dump(patient_history)}
    </patient_history>
    
    <medical_literature>
    {yaml.dump(medical_literature)}
    </medical_literature>
    
    <user_query>
    {user_query}
    </user_query>
    
    <instructions>
    Based on all the information above, provide a step-by-step plan for the user to relieve their headache. Structure your response clearly.
    </instructions>
    """
    ```
5.  Ultimately, let's merge everything together and take a look at the final input sent to the LLM.
    ```xml
    <system_prompt>
    You are a helpful AI medical assistant. Your role is to provide safe, helpful, and personalized health advice based on the provided context. Do not give advice outside of the provided context. Prioritize non-medicinal options as per user preference.
    </system_prompt>
    
    <patient_history>
    patient:
      age: 45
      allergies: []
      conditions:
      - mild_hypertension
      gender: M
      habits:
        caffeine_intake: 3-4_cups_daily
        stress_level: high
        work_related: true
      name: John Doe
      preferences:
        medication_avoidance: true
        preferred_treatments: natural_remedies
    
    </patient_history>
    
    <medical_literature>
    articles:
    - finding: Dehydration is a common cause of tension headaches
      id: 1
      topic: dehydration_headaches
      treatment: Rehydration can alleviate symptoms within 30 minutes to three hours
    - finding: Applying a cold compress to the forehead and temples can constrict blood
        vessels
      id: 2
      topic: cold_compress
      treatment: Reduces inflammation, helping to relieve migraine pain
    - finding: Caffeine withdrawal can trigger headaches
      id: 3
      topic: caffeine_withdrawal
      treatment: For regular caffeine consumers, a small amount may alleviate withdrawal
        headaches
    - finding: Stress-relief techniques are effective for tension headaches
      id: 4
      topic: stress_relief
      treatment: Deep breathing, meditation, or short walks can help
    
    </medical_literature>
    
    <user_query>
    I have a headache. What can I do to stop it? I would prefer not to take any medicine.
    </user_query>
    
    <instructions>
    Based on all the information above, provide a step-by-step plan for the user to relieve their headache. Structure your response clearly.
    </instructions>
    ```

To build such a system, you need a robust tech stack. Here is a potential stack we recommend and will use throughout this course:

*   **LLM:** Gemini as a multimodal, reasoning, and cost-effective LLM API provider.
*   **Orchestration:** LangGraph for defining stateful, agentic workflows.
*   **Databases:** PostgreSQL, MongoDB, Redis, Qdrant, and Neo4j. Often, it is effective to keep it simple, as you can achieve much with only PostgreSQL or MongoDB.
*   **Observability:** Opik or LangSmith for evaluation and trace monitoring.

## Connecting Context Engineering to AI Engineering

Context engineering is more of an art than a science. It is about developing the intuition to craft effective prompts, select the right information from memory, and arrange context for optimal results. This discipline helps you determine the minimal yet essential information an LLM needs to perform at its best.

It's important to understand that context engineering cannot be learned in isolation. It is a complex field that combines:

1.  **AI Engineering:** Implement practical solutions such as LLM workflows, RAG, AI Agents, and evaluation pipelines.
2.  **Software Engineering (SWE):** Build your AI product with code that is not just functional, but also scalable and maintainable, and design architectures that can grow with your product's needs.
3.  **Data Engineering:** Design data pipelines that feed curated and validated data into the memory layer.
4.  **Operations (Ops):** Deploy agents on the proper infrastructure to ensure they are reproducible, maintainable, observable, and scalable, including automating processes with CI/CD pipelines.

Our goal with this course is to teach you how to combine these skills to build production-ready AI products. We like to say that in the world of AI, we should all think in systems rather than isolated components, having a mindset shift from developers to architects.

In the next lesson, we will explore structured outputs.

## References

1.  Osmani, A. (2025, July 13). Context Engineering: Bringing engineering discipline to prompts. Elevate. [https://addyo.substack.com/p/context-engineering-bringing-engineering](https://addyo.substack.com/p/context-engineering-bringing-engineering)
2.  Muntean, A. (2025, April 3). Your AI doesn't need more Training—It needs context. Tabnine. [https://www.tabnine.com/blog/your-ai-doesnt-need-more-training-it-needs-context/](https://www.tabnine.com/blog/your-ai-doesnt-need-more-training-it-needs-context/)
3.  (n.d.). Fine-Tuning vs. Prompt Engineering: A Decision Framework for Enterprise AI | Tribe AI. Tribe AI. [https://www.tribe.ai/applied-ai/fine-tuning-vs-prompt-engineering](https://www.tribe.ai/applied-ai/fine-tuning-vs-prompt-engineering)
4.  (n.d.). From Human Memory to AI Memory: A survey on Memory Mechanisms in the Era of LLMS. arXiv. [https://arxiv.org/html/2504.15965v1](https://arxiv.org/html/2504.15965v1)
5.  humanlayer. (n.d.). 12-factor-agents/content/factor-03-own-your-context-window.md at main · humanlayer/12-factor-agents. GitHub. [https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-03-own-your-context-window.md](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-03-own-your-context-window.md)
6.  Zia, T. (2025, July 5). Why large language models forget the middle: Uncovering AI's hidden blind spot. Unite.AI. [https://www.unite.ai/why-large-language-models-forget-the-middle-uncovering-ais-hidden-blind-spot/](https://www.unite.ai/why-large-language-models-forget-the-middle-uncovering-ais-hidden-blind-spot/)
7.  Baker, G. A., Raut, A., Shaier, S., Hunter, L. E., & Von Der Wense, K. (2024, January 1). Lost in the middle, and In-Between: Enhancing language models' ability to reason over long contexts in Multi-Hop QA. OpenReview. [https://openreview.net/forum?id=5sB6cSblDR](https://openreview.net/forum?id=5sB6cSblDR)
8.  (n.d.). Concept Drift vs Data Drift: Why It Matters in AI. viso.ai. [https://viso.ai/deep-learning/concept-drift-vs-data-drift/](https://viso.ai/deep-learning/concept-drift-vs-data-drift/)
9.  Larson, E. J. (2025, July 25). Context, drift, and the illusion of intent. Colligo. [https://erikjlarson.substack.com/p/context-drift-and-the-illusion-of](https://erikjlarson.substack.com/p/context-drift-and-the-illusion-of)
10. (n.d.). Preview AI Agent Platform Best Practices. Talkdesk Support. [https://support.talkdesk.com/hc/en-us/articles/39096730105115--Preview-AI-Agent-Platform-Best-Practices](https://support.talkdesk.com/hc/en-us/articles/39096730105115--Preview-AI-Agent-Platform-Best-Practices)
11. Liu, S. (2025, January 10). The state of AI agents: lots of potential … and confusion. Forrester. [https://www.forrester.com/blogs/the-state-of-ai-agents-lots-of-potential-and-confusion/](https://www.forrester.com/blogs/the-state-of-ai-agents-lots-of-potential-and-confusion/)
12. (n.d.). AI Agents for Product Managers: Tools that work for you. Product School. [https://productschool.com/blog/artificial-intelligence/ai-agents-product-managers](https://productschool.com/blog/artificial-intelligence/ai-agents-product-managers)
13. Promptmetheus. (n.d.). Lost-in-the-Middle effect. [https://promptmetheus.com/resources/llm-knowledge-base/lost-in-the-middle-effect](https://promptmetheus.com/resources/llm-knowledge-base/lost-in-the-middle-effect)
14. Falconer, S. (n.d.). Four design patterns for Event-Driven, Multi-Agent systems. Confluent. [https://www.confluent.io/blog/event-driven-multi-agent-systems/](https://www.confluent.io/blog/event-driven-multi-agent-systems/)
15. (n.d.). What modifications might be needed to the LLM's input formatting or architecture to best take advantage of retrieved documents (for example, adding special tokens or segments to separate context)?. Milvus. [https://milvus.io/ai-quick-reference/what-modifications-might-be-needed-to-the-llms-input-formatting-or-architecture-to-best-take-advantage-of-retrieved-documents-for-example-adding-special-tokens-or-segments-to-separate-context](https://milvus.io/ai-quick-reference/what-modifications-might-be-needed-to-the-llms-input-formatting-or-architecture-to-best-take-advantage-of-retrieved-documents-for-example-adding-special-tokens-or-segments-to-separate-context)
16. (n.d.). Context Engineering: The Complete guide. Akira AI. [https://www.akira.ai/blog/context-engineering](https://www.akira.ai/blog/context-engineering)
17. 66degrees. (2025, April 7). Building a business case for AI in financial Services | 66degrees. 66degrees. [https://66degrees.com/building-a-business-case-for-ai-in-financial-services/](https://66degrees.com/building-a-business-case-for-ai-in-financial-services/)