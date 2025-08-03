# Lesson 3: Context Engineering

AI applications have evolved rapidly. In 2022, we had simple chatbots for question-answering. By 2023, Retrieval-Augmented Generation (RAG) systems connected LLMs to domain-specific knowledge. 2024 brought us tool-using agents that could perform actions. Now, we are building memory-enabled agents that remember past interactions and build relationships over time.

In our last lesson, we explored how to choose between AI agents and LLM workflows when designing a system. As these applications grow more complex, prompt engineering, a practice that once served us well, is showing its limits. It optimizes single LLM calls but fails when managing systems with memory, actions, and long interaction histories. The sheer volume of information an agent might need—past conversations, user data, documents, and action descriptions has grown exponentially. Simply stuffing all this into a prompt is not a viable strategy. This is where context engineering comes in. It is the discipline of orchestrating this entire information ecosystem to ensure the LLM gets exactly what it needs, when it needs it. This skill is becoming a core foundation for AI engineering.

## From prompt to context engineering

Prompt engineering, while effective for simple tasks, is designed for single, stateless interactions. It treats each call to an LLM as a new, isolated event. This approach breaks down in stateful applications where context must be preserved and managed across multiple turns.

As a conversation or task progresses, the context grows. Without a strategy to manage this growth, the LLM’s performance degrades. This is context decay: the model gets confused by the noise of an ever-expanding history. It starts to lose track of original instructions or key information.

Even with large context windows, a physical limit exists for what you can include. Also, on the operational side, every token adds to the cost and latency of an LLM call. Simply putting everything into the context creates a slow, expensive, and underperforming system. We will explore these concepts in more detail in upcoming lessons, including memory in Lesson 9 and RAG in Lesson 10.

On a recent project, we learned this the hard way. We were working with a model that supported a two-million-token context window, so we thought, "What could go wrong?" We stuffed everything in: our research, guidelines, examples, and user history. The result was an LLM workflow that took 30 minutes to run and produced low-quality outputs.

This is where context engineering becomes essential. It shifts the focus from crafting static prompts to building dynamic systems that manage information flow. As an AI engineer, your job is to select only the most critical pieces of context for each LLM call. This makes your applications accurate, fast, and cost-effective.

## Understanding context engineering

Context engineering is about finding the best way to arrange information from your applications memory into the context passed to an LLM. It is a solution to an optimization problem where you retrieve the right parts from your short-term and long-term memory to solve a specific task without overwhelming the model. For example, when you ask a a cooking agent for a recipe, you do not give it the entire cookbook. Instead, you retrieve the specific recipe, along with personal context like allergies or taste preferences. This precise selection ensures the model receives only the essential information.

Andrej Karpathy explains that LLMs are like a new kind of operating system where the model is the CPU and its context window is the RAM [[36]](https://addyo.substack.com/p/context-engineering-bringing-engineering). Just as an operating system manages what fits into your computer’s limited RAM, context engineering manages what information occupies the model’s limited context window.

<div align="center">
<img src="./media/what_makes_up_the_context.png" alt="What makes up the context" width="700"/>
</div>
Figure 1: What makes up the context

How does context engineering relates to prompt engineering? It's simple. Prompt engineering is a subset of context engineering. You still write effective prompts, but you also design a system that feeds the right context into those prompts. This means understanding not just *how* to phrase a task, but *what* information the model needs to perform optimally. This is illustrated in Figure 1, where all the context gathered across multiple conversation turns between the user and the agent is used to create the final prompt, which is passed to the LLM.

| Dimension | Prompt Engineering | Context Engineering |
|-----------|-------------------|---------------------|
| Scope | Single interaction optimization | Entire information ecosystem |
| State Management | Stateless function | Stateful due to memory |
| Focus | How to phrase tasks | What information to provide |

Table 1: A comparison of prompt engineering and context engineering.

Context engineering is the new fine-tuning. While fine-tuning has its place, it is expensive, time-consuming, and inflexible. Data changes constantly, making fine-tuning a last resort [[41]](https://www.tabnine.com/blog/your-ai-doesnt-need-more-training-it-needs-context/). For most enterprise use cases, you get better results faster and more cheaply with context engineering [[43]](https://www.tribe.ai/applied-ai/fine-tuning-vs-prompt-engineering). It allows for rapid iteration and adaptation to evolving data without altering the core model, a key advantage in dynamic environments. This approach avoids the computational resources and specialized expertise required for retraining, offering a more agile path to reliable AI applications.

When you start a new AI project, your decision-making process for guiding the LLM should look like the one presented in Figure 2.

<div align="center">
<img src="./media/prompt_vs_context_engineering_fine_tuning.png" alt="Prompt vs. Context. Fine-tuning" width="700"/>
</div>
Figure 2: A decision-making workflow for choosing between prompt engineering, context engineering, and fine-tuning.

For instance, if you build an agent to process internal Slack messages, you do not need to fine-tune a model on your company’s communication style. It is more effective to use a powerful reasoning model and engineer the context to retrieve specific messages and enable actions like creating tasks or drafting emails. Throughout this course, we will show you how to solve most industry problems using only context engineering.

## What makes up the context

To master context engineering, you first need to understand what "context" actually is. It is everything the LLM sees in a single turn, dynamically assembled from various memory components before being passed to the model.

The high-level workflow, as presented in Figure 3, begins when a user input triggers the system to pull relevant information from both long-term and short-term memory. This information is assembled into the final context, inserted into a prompt template, and sent to the LLM. The LLM’s answer then updates the memory, and the cycle repeats.

```mermaid
graph TD
    A[User Input] --> B{Long-Term Memory};
    B --> C{Short-Term Working Memory};
    C --> D[Context Assembly];
    D --> E[Prompt Template];
    E --> F[Prompt];
    F --> G((LLM Call));
    G --> H[Answer];
    H --> C;
    subgraph Memory Update
        H --> I{Short-Term Memory};
        I --> J{Long-Term Memory};
    end
```
Figure 3: The high-level workflow of how context is assembled and used in an AI system.

These components group into two main categories. We will explain them intuitively for now, as we have future dedicated lessons for all of them:

### Short-term working memory
Short-term working memory is the state of the agent for the current task or conversation. It is volatile and changes with each interaction, helping the agent maintain a coherent dialogue and make immediate decisions.  It can include some or all of these components:
*   **User input:** The most recent query or command from the user.
*   **Message history:** The log of the current conversation, allowing the LLM to understand the flow and previous turns.
*   **Agent's internal thoughts:** The reasoning steps the agent takes to decide on its next action.
*   **Action calls and outputs:** The results from any actions the agent has performed, providing information from external systems.

### Long-term memory
Long-term memory is more persistent and stores information across sessions, allowing the AI system to remember things beyond a single conversation. We divide it into three types, drawing parallels from human memory [[16]](https://arxiv.org/html/2504.15965v1). As before an AI system, can include some or all of them:
*   **Procedural memory:** This is knowledge encoded directly in the code. It includes the system prompt, which sets the agent's overall behavior. Also, it includes the definitions of available actions, which tell the agent what it can do and schemas for structured outputs, which guide the format of its responses. Think of this as the agent's built-in skills.
*   **Episodic memory:** This is memory of specific past experiences, like user preferences or previous interactions. It's used to help the agent personalize its responses based on individual users. We typically store this in vector or graph databases for efficient retrieval.
*   **Semantic memory:** This is the agent’s general knowledge base. It can be internal, like company documents stored in a data lake, or external, accessed via the internet through API calls or web scraping. This memory provides the factual information the agent needs to answer questions.

![A diagram showing that Context Engineering encompasses RAG, Prompt Engineering, State/History, Memory, and Structured Outputs.](https://github.com/user-attachments/assets/0f1f193f-8e94-4044-a276-576bd7764fd0)
Figure 4: Context engineering encompasses a variety of techniques and information sources [[2]](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-03-own-your-context-window.md).

If this seems like a lot, bear with us. We will cover all these concepts in-depth in future lessons, including structured outputs (Lesson 4), actions (Lesson 6), memory (Lesson 9), and RAG (Lesson 10).

<div align="center">
<img src="./media/context_engineering_guide_101.png" alt="Context Engineering Guide 101" width="700"/>
</div>
Figure 5: A detailed illustration of how all the context engineering components work together inside an AI agent


The key takeaway is that these components are not static. They are dynamically re-computed for every single interaction. For each conversation turn or new task, the short-term memory grows or the long-term memory can change. Context engineering involves knowing how to select the right pieces from this vast memory pool to construct the most effective prompt for the task at hand.

## Production implementation challenges

Now that we understand what makes up the context, let's look at the core challenges of implementing it in production. These challenges all revolve around a single question: *"How can I keep my context as small as possible while providing enough information to the LLM?"*

Here are four common issues that come up when building AI applications:

1.  **The context window challenge:** Every AI model has a limited context window, the maximum amount of information (tokens) it can process at once. Think of it like your computer's RAM. If your machine has only 32GB of RAM, that is all it can use at one time. While context windows are getting larger, they are not infinite, and treating them as such leads to other problems.

2.  **Information overload:** Just because you can fit a lot of information into the context does not mean you should. Too much context reduces the performance of the LLM by confusing it. This is known as the *"lost-in-the-middle"* or *"needle in the haystack"* problem, where LLMs are known for remembering information best at the beginning and end of the context window. Information in the middle is often overlooked, and performance can drop long before the physical context limit is reached [[55]](https://www.unite.ai/why-large-language-models-forget-the-middle-uncovering-ais-hidden-blind-spot/), [[8]](https://openreview.net/forum?id=5sB6cSblDR).

3.  **Context drift:** This occurs when conflicting views of truth accumulate in the memory over time [[28]](https://viso.ai/deep-learning/concept-drift-vs-data-drift/). For example, the memory might contain two conflicting statements: "My cat is white" and "My cat is black." This is not the Schrodinger's Cat quantum physics experiment, it is a data conflict that confuses the LLM. Without a mechanism to resolve these conflicts, the model's responses become unreliable [[31]](https://erikjlarson.substack.com/p/context-drift-and-the-illusion-of).

4.  **Tool confusion:** The final challenge is tool confusion, which arises in two main ways. First, adding too many actions to an agent can confuse the LLM about the best one for the job, a problem that often appears with over 100 actions [[38]](https://support.talkdesk.com/hc/en-us/articles/39096730105115--Preview-AI-Agent-Platform-Best-Practices). Second, confusion can occur when tool descriptions are poorly written or overlap. If the distinctions between actions are unclear, even a human would struggle to choose the right one [[49]](https://www.forrester.com/blogs/the-state-of-ai-agents-lots-of-potential-and-confusion/).

## Key strategies for context optimization

Initially, most AI applications were chatbots over single knowledge bases. Today, modern AI solutions must manage multiple knowledge bases, tools, and complex conversational histories. Context engineering is about managing this complexity while meeting performance, latency, and cost requirements.

Here are four popular context engineering strategies used across the industry:

### 1. Selecting the right context
Retrieving the right information from memory is a critical first step. A common mistake is to provide everything at once, assuming that models with large context windows can handle it. As we've discussed, the "lost-in-the-middle" problem means this often leads to poor performance, increased latency, and higher costs.

To solve this, consider these approaches:
*   **Use structured outputs:** Define clear schemas for what the LLM should return. This allows you to pass only the necessary, structured information to downstream steps. We will cover this in detail in Lesson 4.
*   **Use RAG:** Instead of providing entire documents, use RAG to fetch only the specific chunks of text needed to answer a user's question. This is a core topic we will explore in Lesson 10.
*   **Reduce the number of available actions:** Rather than giving an agent access to every available action, use various strategies to delegate action subsets to specialized components. For example, a typical pattern is to leverage the orchestrator-worker pattern to delegate subtasks to specialized agents. Studies show that limiting the selection to under 30 tools can triple the agent's selection accuracy [[45]](https://productschool.com/blog/artificial-intelligence/ai-agents-product-managers). Still the exact number of the perfect number of tools an agent can use can be extremely dependent on what tools you provide, what LLM you use, and how well the actions are defined. That's why evaluating the performance of your AI system on core business metrics it's a mandatory step that will help you pick the number. We will learn how to do this in future lessons.
*   **Rank time-sensitive data:** For time-sensitive information, rank it by date and filter out anything no longer relevant.
*   **Repeat core instructions:** For the most important instructions, repeat them at both the start and the end of the prompt. This uses the model's tendency to pay more attention to the context edges, ensuring core instructions are not lost [[56]](https://promptmetheus.com/resources/llm-knowledge-base/lost-in-the-middle-effect).

```mermaid
graph TD
    subgraph "Context Selection Strategies"
        A[User Query] --> B{Context Selection Engine};
        
        B --> C["RAG: Retrieve Relevant Documents"];
        B --> D["Tool Selection: Filter to <30 Tools"];
        B --> E["Time Ranking: Rank by Date/Relevance"];
        B --> F["Structured Outputs: Define Response Schema"];
        
        C --> G[Selected Context];
        D --> G;
        E --> G;
        F --> G;
    end
    
    subgraph "Context Assembly"
        G --> H[Assemble Context];
        H --> I[Add Core Instructions at Start];
        H --> J[Add Core Instructions at End];
        I --> K[Final Optimized Context];
        J --> K;
    end
    
    K --> L((LLM Call));
    L --> M[Structured Response];
    M --> N[Update Memory];
    N --> A;
```
Figure 6: A workflow showing how the four context selection strategies work together to optimize information retrieval and assembly.

### 2. Context compression
As message history grows in short-term working memory, you must manage past interactions to keep your context window in check. You cannot simply drop past conversation turns, as the LLM still needs to remember what happened. Instead, you need ways to compress key facts from the past.

You can do this through:
*   **Creating summaries of past interactions:** Use an LLM to replace a long, detailed history with a concise overview.
*   **Moving user preferences to long-term memory:** Transfer user preferences from working memory to long-term episodic memory. This keeps the working context clean while ensuring preferences are remembered for future sessions.
*   **Deduplication:** Remove redundant information from the context to avoid repetition.

```mermaid
graph TD
    subgraph "Context Compression"
        A[Long Message History] -- LLM Call --> B(Summarize);
        B --> C[Compressed History];
        A -- LLM Call --> D(Extract Preferences);
        D --> E[Long-Term Episodic Memory];
    end
```
Figure 7: Compressing context by summarizing history and extracting preferences to long-term memory.

### 3. Isolating context
Another powerful strategy is to isolate context by splitting information across multiple agents or LLM workflows. Instead of one agent with a massive, cluttered context window, you can have a team of agents, each with a smaller, focused context.

We often implement this using an orchestrator-worker pattern, where a central orchestrator agent breaks down a problem and assigns sub-tasks to specialized worker agents [[12]](https://www.confluent.io/blog/event-driven-multi-agent-systems/). Each worker operates in its own isolated context, improving focus and allowing for parallel processing. We will cover this pattern in more detail in Lesson 5.

```mermaid
graph TD
    A[User Request] --> B(Orchestrator Agent);
    B --> C{"Worker Agent 1<br>(Context A)"};
    B --> D{"Worker Agent 2<br>(Context B)"};
    B --> E{"Worker Agent 3<br>(Context C)"};
    C --> F[Results];
    D --> F;
    E --> F;
```
Figure 8: The orchestrator-worker pattern isolates context across multiple specialized agents.

### 4. Format optimization
Finally, the way you format the context matters. Models are sensitive to structure, and using clear delimiters can improve performance.
*   **Use XML tags:** Wrap different pieces of context in XML-like tags (e.g., `<user_query>`, `<documents>`). This helps the model distinguish between different types of information [[5]](https://milvus.io/ai-quick-reference/what-modifications-might-be-needed-to-the-llms-input-formatting-or-architecture-to-best-take-advantage-of-retrieved-documents-for-example-adding-special-tokens-or-segments-to-separate-context).
*   **Prefer YAML over JSON:** When providing structured data as input, YAML is often more token-efficient than JSON, which helps save space in your context window.

💡 A final note: Always understand what is passed to the LLM. Seeing exactly what occupies your context window at every step is key to mastering context engineering.

## Here is an example

Let's connect the theories and strategies discussed earlier with concrete examples. Context engineering is fundamental for many advanced AI applications that need to manage memory and maintain state across interactions.

Consider several real-world scenarios where keeping context in memory is essential:
*   **Healthcare:** An AI assistant accesses a patient's medical history (episodic memory), current symptoms (working memory), and the latest medical literature (semantic memory) to provide personalized diagnostic support [[52]](https://www.akira.ai/blog/context-engineering).
*   **Financial Services:** AI systems integrate with enterprise tools like Customer Relationship Management (CRM) systems and calendars, combining real-time market data and client portfolio information to generate tailored financial advice [[50]](https://66degrees.com/building-a-business-case-for-ai-in-financial-services/).
*   **Project Management:** AI systems access enterprise infrastructure like CRMs, Slack, and task managers to automatically understand project requirements, then add and update project tasks.
*   **Content Creator Assistant:** An AI agent uses your research, past content, and personality traits to understand what and how to create a given piece of content.

Let's walk through a specific query to see context engineering in action. A user asks their healthcare assistant: `I have a headache. What can I do to stop it? I would prefer not to take any medicine.`

Before the AI attempts to answer, a context engineering system performs several steps:
1.  It retrieves the user's patient history, known allergies, and lifestyle habits from episodic memory.
2.  It queries a medical database for non-pharmacological headache remedies from semantic memory.
3.  It uses various actions to assemble the key units of information from both memory types into the final context.
4.  It formats this information into a structured prompt and calls the LLM.
5.  Finally, it presents a personalized, context-aware answer to the user.

Here is a simplified Python example showing how you might structure the prompt for the LLM, using XML tags to format the different context elements.

1.  First, we define the user's query:
```python
user_query = "I have a headache. What can I do to stop it? I would prefer not to take any medicine."
```

2.  Next, we define the patient's history, which would typically be retrieved from episodic memory:
```python
patient_history = """
- Patient: John Doe, 45M
- Known Conditions: Mild hypertension
- Allergies: None
- Preferences: Avoids medication unless necessary. Prefers natural remedies.
- Habits: Reports high stress from work, drinks 3-4 cups of coffee daily.
"""
```

3.  Then, we include relevant medical literature, which would be retrieved from semantic memory:
```python
medical_literature = """
- Article 1: Dehydration is a common cause of tension headaches. Rehydration can alleviate symptoms within 30 minutes to three hours.
- Article 2: Applying a cold compress to the forehead and temples can constrict blood vessels and reduce inflammation, helping to relieve migraine pain.
- Article 3: Caffeine withdrawal can trigger headaches. For individuals who consume caffeine regularly, a small amount may alleviate a withdrawal headache.
- Article 4: Stress-relief techniques such as deep breathing, meditation, or a short walk can be effective for tension headaches.
"""
```

4.  Finally, we assemble the complete prompt, combining all elements into a structured format:
```python
prompt = f"""
<system_prompt>
You are a helpful AI medical assistant. Your role is to provide safe, helpful, and personalized health advice based on the provided context. Do not give advice outside of the provided context. Prioritize non-medicinal options as per user preference.
</system_prompt>

<patient_history>
{patient_history}
</patient_history>

<medical_literature>
{medical_literature}
</medical_literature>

<user_query>
{user_query}
</user_query>

<instructions>
Based on all the information above, provide a step-by-step plan for the user to relieve their headache. Structure your response clearly.
</instructions>
"""
```

5.  Printing the assembled prompt shows the final input sent to the LLM:
```python
print(prompt)
```
It outputs:
```xml
<system_prompt>
You are a helpful AI medical assistant. Your role is to provide safe, helpful, and personalized health advice based on the provided context. Do not give advice outside of the provided context. Prioritize non-medicinal options as per user preference.
</system_prompt>

<patient_history>
- Patient: John Doe, 45M
- Known Conditions: Mild hypertension
- Allergies: None
- Preferences: Avoids medication unless necessary. Prefers natural remedies.
- Habits: Reports high stress from work, drinks 3-4 cups of coffee daily.
</patient_history>

<medical_literature>
- Article 1: Dehydration is a common cause of tension headaches. Rehydration can alleviate symptoms within 30 minutes to three hours.
- Article 2: Applying a cold compress to the forehead and temples can constrict blood vessels and reduce inflammation, helping to relieve migraine pain.
- Article 3: Caffeine withdrawal can trigger headaches. For individuals who consume caffeine regularly, a small amount may alleviate a withdrawal headache.
- Article 4: Stress-relief techniques such as deep breathing, meditation, or a short walk can be effective for tension headaches.
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
*   **Observability:** Opik or LangSmith for evaluation and monitoring.

## Connecting context engineering to AI engineering

Context engineering is more of an art than a science. It is about developing the intuition to craft effective prompts, select the right information from memory, and arrange context for optimal results. This discipline helps you determine the minimal yet essential information an LLM needs to perform at its best.

This discipline cannot be learned in isolation. It is a complex field that combines:
1.  **AI Engineering:** LLM workflows, RAG, and AI Agents. This includes implementing evaluation pipelines.
2.  **Software Engineering (SWE):** Aggregating all context elements into scalable and maintainable code, designing scalable architectures, and wrapping agents as APIs.
3.  **Data Engineering:** Building data pipelines to feed information into the long-term memory.
4.  **Operations (Ops):** Deploying your agents on the right infrastructure to ensure they are reproducible, maintainable, observable, and scalable, including automating processes with CI/CD pipelines.

Our goal is to teach you how to master these skills. In the world of AI, we should all think in systems rather than isolated components. We will show you how to build production-ready AI applications.

To transition from this lesson, we will next explore structured outputs in Lesson 4. Later in the course, we will delve into actions (Lesson 6), memory (Lesson 9), and RAG (Lesson 10).

## References

- [2] https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-03-own-your-context-window.md
- [5] https://milvus.io/ai-quick-reference/what-modifications-might-be-needed-to-the-llms-input-formatting-or-architecture-to-best-take-advantage-of-retrieved-documents-for-example-adding-special-tokens-or-segments-to-separate-context
- [8] https://openreview.net/forum?id=5sB6cSblDR
- [12] https://www.confluent.io/blog/event-driven-multi-agent-systems/
- [16] https://arxiv.org/html/2504.15965v1
- [28] https://viso.ai/deep-learning/concept-drift-vs-data-drift/
- [31] https://erikjlarson.substack.com/p/context-drift-and-the-illusion-of
- [36] https://addyo.substack.com/p/context-engineering-bringing-engineering
- [38] https://support.talkdesk.com/hc/en-us/articles/39096730105115--Preview-AI-Agent-Platform-Best-Practices
- [41] https://www.tabnine.com/blog/your-ai-doesnt-need-more-training-it-needs-context/
- [43] https://www.tribe.ai/applied-ai/fine-tuning-vs-prompt-engineering
- [45] https://productschool.com/blog/artificial-intelligence/ai-agents-product-managers
- [49] https://www.forrester.com/blogs/the-state-of-ai-agents-lots-of-potential-and-confusion/
- [50] https://66degrees.com/building-a-business-case-for-ai-in-financial-services/
- [52] https://www.akira.ai/blog/context-engineering
- [55] https://www.unite.ai/why-large-language-models-forget-the-middle-uncovering-ais-hidden-blind-spot/
- [56] https://promptmetheus.com/resources/llm-knowledge-base/lost-in-the-middle-effect