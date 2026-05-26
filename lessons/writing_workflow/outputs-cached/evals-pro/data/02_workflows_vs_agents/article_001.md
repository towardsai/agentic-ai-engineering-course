# Workflows, Agents and The Autonomy Slider
### What only the top 10% AI engineers who ship think about

In the past few months, I have been working on my latest project: **Brown**, my writing assistant. Initially, I was not sure how to properly architect it. I debated where I should use workflows, agents, or hybrids. Thus, I did what I usually do when I am not sure: *"JUST START"*. After weeks of hard work, I managed to get something working. But it was a mess. It was slow. It was complicated. The user experience was terrible. Ultimately, I had to refactor the whole thing.

After, I asked myself: *"What could I have done better to avoid wasting all these weeks?"*

This is a broad question that I will carefully address in this longer series on the AI Agents Foundations. But for now, let's start with the beginning. *When should you design your AI system as an agent or as a workflow?*

As AI Engineers, this is an architectural decision that we face early in our development process. We should always think about what to choose between a predictable, step-by-step workflow where you control every action, or an autonomous agent that can think and decide for itself. We must treat this decision like our lives depend on it. Why? Because this is one of the key decisions that will impact everything from development time and cost to reliability and user experience.

This architectural fork in the road is deceptive. It looks like a simple technical choice. But it is actually a product and business decision. It impacts everything from development costs to user trust.

Choose the wrong approach, and you might end up with an overly rigid system. It breaks when users deviate from expected patterns. It breaks when developers try to add new features. Conversely, you might build an unpredictable agent. It works brilliantly 80% of the time. But it fails beautifully when it matters most. It burns through your API budget in infinite loops. In the worst case, you could waste months of development time rebuilding the entire architecture. This happens because the chosen path could not scale with the complexity of the real world.

In the 2024-2025 landscape, we have seen billion-dollar AI startups succeed or fail based primarily on this architectural decision [[1]](https://techcrunch.com/2025/06/18/here-are-the-24-us-ai-startups-that-have-raised-100m-or-more-in-2025/). The successful companies, teams, and AI engineers know when to use workflows versus agents. More importantly, they know how to combine both approaches effectively.

By the end of this lesson, we will provide you with a framework to make these critical decisions confidently. You will understand the fundamental trade-offs. You will see real-world examples from leading AI companies like Google and Perplexity. You will learn how to design systems that leverage the best of both approaches.

## Understanding the Spectrum: From Workflows to Agents

To choose between workflows and agents, you need a clear understanding of what they are. At this point, we do not focus on the technical specifics of each. We focus on their properties and how they are used.

**LLM workflows** are sequences of tasks involving LLM calls or other operations. These operations include reading and writing data to a database or file system. They are largely predefined and orchestrated by developer-written code. The defining characteristic here is that the steps are defined in advance. This results in deterministic or rule-based paths with predictable execution and explicit control flow [[2]](https://blog.gopenai.com/agentic-workflows-vs-autonomous-ai-agents-do-you-know-the-difference-c21c9bfb20ac), [[3]](https://www.louisbouchard.ai/agents-vs-workflows/).

Think of a workflow as a factory assembly line. The raw material (input) moves through specific stations (LLM calls/functions). It gets processed in a specific order to produce the final product (output). Every step is accounted for. Every outcome is anticipated. If station A fails, we know exactly why. We know exactly where to look.

```mermaid
graph TD
    A["Input"] --> B["LLM (Initial Processing)"]
    B --> C["Retrieval Component"]
    C --> D["LLM (Integrate Retrieval)"]
    D --> E["Tools Component"]
    E --> F["LLM (Integrate Tools)"]
    F --> G["Memory Component"]
    G --> H["LLM (Integrate Memory)"]
    H --> I["Output"]
```
Image 1: A flowchart illustrating a simple LLM workflow with sequential interactions with Retrieval, Tools, and Memory components.

**AI agents**, on the other hand, are systems where an LLM plays a central role in dynamically deciding the sequence of steps. It decides the reasoning and actions to achieve a goal. Here, the steps are not defined in advance. They are dynamically planned based on the task and the current state of the environment [[4]](https://cloud.google.com/discover/what-are-ai-agents).

If workflows are assembly lines, an agent is a skilled human expert tackling an unfamiliar problem. They have a goal. They have access to actions (tools). They adapt their strategy in the moment after each "Eureka" or "Oops" moment. They might try one path, hit a dead end, and backtrack. They might realize they need more information and perform a search. This autonomy is powerful. But it is also dangerous.

```mermaid
graph TD
    A["Human"] --> B["LLM Call"]
    B --> C["Perform Action in Environment"]
    C --> D["Receive Feedback"]
    D -->|"Loop"| B
    D -->|"Stop"| E["Stop"]
```
Image 2: A flowchart illustrating a simple AI agent system.

Both workflows and agents require an **orchestration layer**. But their nature differs. In workflows, the orchestration executes a defined plan. It is the conductor following a score. In agents, the orchestration facilitates the LLM's dynamic planning and execution. It is the improv jazz band leader responding to what just happened [[5]](https://www.anthropic.com/engineering/building-effective-agents).

## Choosing Your Path

Now that we have defined them, let's explore their core differences: Developer-defined logic versus LLM-driven autonomy.

In reality, this is not a binary choice. The decision between workflows and agents exists on a spectrum. We call this the "autonomy slider." At one end, you have fully controlled workflows. At the other, fully autonomous agents.

```mermaid
graph TD
    Main["LLM Workflows vs. AI Agents: A Spectrum"]

    subgraph "LLM Workflows"
        LLM_W_Node["LLM Workflows"]
        LLM_W_Node --> P["Predictability: High"]
        LLM_W_Node --> C["Control: High"]
        LLM_W_Node --> R["Reliability: High (within defined scope)"]
        LLM_W_Node --> ES["Execution: Explicit, Sequential Steps"]
    end

    subgraph "AI Agents"
        AI_A_Node["AI Agents"]
        AI_A_Node --> DD["Decision-Making: Dynamic, Autonomous"]
        AI_A_Node --> LC["Control: Lower (Emergent Behavior)"]
        AI_A_Node --> VR["Reliability: Variable (Context Dependent)"]
        AI_A_Node --> OA["Execution: Goal-Oriented, Adaptive Actions"]
    end

    Main --> LLM_W_Node
    Main --> AI_A_Node
```
Image 3: A diagram illustrating the spectrum between LLM workflows and AI agents, highlighting their core differences in predictability, control, reliability, and execution.

### When to use LLM workflows

Workflows are best for structured, repeatable tasks where the path to the solution is known. Examples include pipelines for data extraction and transformation from sources like the web or Slack. They are great for automated report generation. They work well for document summarization followed by translation. They shine in content repurposing, such as taking an article and transforming it into a LinkedIn post [[6]](https://towardsdatascience.com/a-developer-s-guide-to-building-scalable-ai-workflows-vs-agents/).

The primary strength of workflows is **predictability**. Because the execution path is fixed, costs and latency are consistent. Debugging is straightforward. You can often leverage simpler, smaller models specialized in specific sub-tasks. This lowers operational costs and infrastructure overhead. You don't need a massive reasoning model to just extract a date from an email. A smaller, cheaper model will do just fine in a workflow.

However, they can be rigid. Developing a workflow often requires more time upfront to engineer each step manually. The user experience can feel constrained because the system cannot handle unexpected scenarios outside its code path. If the input format changes slightly, the whole pipeline might break.

Workflows are usually preferred in enterprise or regulated fields like finance or healthcare. When a financial advisor asks for a report, it must contain the correct information every time. In healthcare, AI tools must work with high accuracy as they impact lives. In these high-stakes environments, consistency trumps creativity [[7]](https://www.deloitte.com/us/en/insights/industry/technology/technology-media-and-telecom-predictions/2025/autonomous-generative-ai-agents-still-under-development.html). They are also ideal for MVPs requiring rapid deployment. They are best for scenarios where high throughput (thousands of requests per minute) is more important than complex reasoning.

Let's be real for a second. Most "AI Agents" you see on Twitter demos are actually just well-crafted workflows. And that is fine. In fact, that is usually better. You want your software to work. You don't want it to "think" about whether it wants to work today.

### When to use AI agents

Agents excel at open-ended, dynamic problems. Use them for research synthesis (e.g., "Find everything about the history of transistors"). Use them for dynamic problem-solving like debugging code. Use them for completing tasks in unfamiliar environments, such as booking a flight without a specific API [[4]](https://cloud.google.com/discover/what-are-ai-agents).

Their strength is **adaptability**. They can handle ambiguity and novelty because the steps are dynamically decided. They can recover from errors. They can try alternative paths.

The weakness is that the system becomes prone to errors. As the agent is non-deterministic, performance, latency, and costs can vary with each run. Agents often require larger, more capable LLMs to reason effectively. This makes them more expensive. They also require more LLM calls to understand intent and take actions. This compounds the cost.

If not designed well, there are significant security concerns. An autonomous agent with write permissions could delete data. It could send inappropriate emails. It could hallucinate a command that wipes your database.

There is a running joke in the developer community about the current state of coding agents. "The agent deleted my entire codebase. But that's okay. I wanted to start a new project anyway." This is funny because it is true. We are still in the early days where reliability is the biggest bottleneck.

### The Autonomy Slider and Hybrid Approaches

Most real-world systems are not pure workflows or pure agents. They are hybrid systems that adopt the best of both worlds.

When building an application, you decide where to set the "autonomy slider." A manual process involves a workflow with a human verifying intermediate steps. A more automated one gives the agent greater control with fewer checkpoints. Successful applications like Cursor and Perplexity let the user control this slider. In Cursor, you can tab to complete code (low autonomy). You can use CMD+K to generate a block (medium). Or you can use Composer to refactor a whole feature (high autonomy) [[8]](https://www.youtube.com/watch?v=LCEmiRjPEtQ).

The ultimate goal is to speed up the loop between AI generation and human verification.

```mermaid
graph TD
    A["AI Generates Content"]
    B{"Human Reviews/Verifies"}
    C["Content Accepted"]
    D["Refinement Needed"]

    A --> B
    B -->| "Verified" | C
    B -->| "Needs Refinement" | D
    D --> A
```
Image 4: A flowchart illustrating the AI generation and human verification loop.

## Exploring Common Patterns

To navigate the AI Engineering world, you need to understand the common patterns used to build these systems. We will introduce them here to build your intuition. We will dig into the technical implementation in future lessons.

### LLM Workflow Patterns

**Chaining and Routing:** These are the foundational blocks. Chaining connects multiple LLM calls sequentially. Output A becomes input B. This allows you to break down complex tasks into manageable chunks. Routing acts as a traffic controller. It uses an LLM to classify an input. It directs it to the appropriate downstream path. This ensures that specialized prompts handle specific types of requests.

```mermaid
graph TD
    subgraph "Chaining Workflow"
        C_Input["Input"]
        C_LLM1["LLM Call 1"]
        C_LLM2["LLM Call 2"]
        C_LLM3["LLM Call 3"]
        C_Output["Final Output"]

        C_Input --> C_LLM1
        C_LLM1 --> C_LLM2
        C_LLM2 --> C_LLM3
        C_LLM3 --> C_Output
    end

    subgraph "Routing Workflow"
        R_Input["Input"]
        R_Classifier{"Initial LLM (Classifier)"}
        R_LLMA["Specialized LLM A"]
        R_LLMB["Specialized LLM B"]
        R_TaskC["Specialized Task C"]
        R_Output["Final Output"]

        R_Input --> R_Classifier
        R_Classifier -->|"Route A"| R_LLMA
        R_Classifier -->|"Route B"| R_LLMB
        R_Classifier -->|"Route C"| R_TaskC
        R_LLMA --> R_Output
        R_LLMB --> R_Output
        R_TaskC --> R_Output
    end
```
Image 5: A flowchart illustrating LLM workflow patterns of chaining and routing.

**Orchestrator-Worker:** This pattern bridges the gap between workflows and agents. A central LLM (the orchestrator) understands the user intent. It dynamically plans the necessary sub-tasks. It delegates them to specialized workers. The workers execute the tasks. This often happens in parallel. The orchestrator then synthesizes the results [[5]](https://www.anthropic.com/engineering/building-effective-agents). This is powerful because it allows for dynamic decomposition of problems while keeping the execution of sub-tasks relatively deterministic.

```mermaid
graph TD
    subgraph "Orchestrator LLM"
        A["Receive Task"]
        B["Decompose Task"]
        C["Delegate Subtasks"]
        G["Collect Results"]
        H["Synthesize Final Answer"]
    end

    subgraph "Worker LLMs"
        D["Process Subtask 1"]
        E["Process Subtask 2"]
        F["Process Subtask N"]
    end

    A --> B
    B --> C
    C --> D
    C --> E
    C --> F
    D --> G
    E --> G
    F --> G
    G --> H
```
Image 6: A flowchart illustrating the orchestrator-worker LLM workflow pattern.

**Evaluator-Optimizer Loop:** This pattern is used to auto-correct LLM outputs. An "LLM reviewer" analyzes the initial output. It creates a feedback report (reflection). It passes it back to the generator to improve the answer. It mimics a human writer refining a draft based on an editor's notes. This loop can significantly improve quality without requiring a smarter model. It just requires a model that is good at critiquing.

```mermaid
graph TD
    A["Start"] --> B["Input Prompt"]
    B --> C["Generator LLM"]
    C --> D["Generated Response"]
    D --> E["Evaluator LLM"]
    E --> F["Feedback"]
    F --> G{"Is Response Satisfactory?"}
    G -->|"No"| C
    G -->|"Yes / Limit Met"| H["Final Output"]
```
Image 7: A loop diagram illustrating the evaluator-optimizer LLM workflow pattern.

### Core Components of a ReAct AI Agent

The industry standard for building agents is the **ReAct** (Reason and Act) pattern. At a high level, it loops through a cycle of reasoning about what to do, acting on that decision, and observing the result.

The core components include:
*   **LLM:** The brain that plans actions and interprets outputs.
*   **Actions:** The capability to interact with the external environment (e.g., search the web, run code). We will explore this concept, often called "tools," in Lesson 6.
*   **Short-term Memory:** The working context of the agent. It is similar to RAM in a computer.
*   **Long-term Memory:** Used to access factual data (like company databases) and remember user preferences. We will cover this in Lesson 9.

```mermaid
graph TD
    A["LLM receives Task"] --> B["Reasoning (Plan & Self-reflect)"]
    B --> C{"Decide Action?"}
    C -->|Yes, Action| D["Select & Use Tool"]
    D --> E["Observe Tool Output"]
    E --> B

    subgraph "Memory"
        M1["Short-term Memory"]
        M2["Long-term Memory"]
    end

    B -- "Interacts with" --> M1
    E -- "Updates" --> M1
    M1 -- "Can Update" --> M2
    B -- "Interacts with" --> M2

    C -->|No, Task Complete| F["Task Completed"]
```
Image 8: A flowchart illustrating the high-level dynamics of a ReAct AI agent.

## Zooming In on Our Favorite Examples

To anchor these concepts in reality, let's look at three examples ranging from a simple workflow to a complex hybrid system.

### Document Summarization Workflow

When working in teams, finding the right document is time-consuming. Large documents are hard to scan. Embedded summarization features are essential.

This is a perfect example of a pure workflow. It must be fast. It must be cheap. It must be consistent.

```mermaid
graph TD
    A["Read Document"] --> B["Summarize (LLM Call)"]
    B --> C["Extract Key Points (LLM Call)"]
    C --> D["Save to Database"]
    D --> E["Show Results to User"]
```
Image 9: A flowchart illustrating the document summarization and analysis workflow by Gemini in Google Workspace.

The system reads the document. It chains an LLM call to generate a summary. It chains another call to extract metadata like tags. It saves the results. It displays them. There is no complex decision-making. It is a straight line from input to output [[9]](https://workspaceupdates.googleblog.com/2025/06/summarize-responses-with-gemini-google-forms.html), [[10]](https://workspace.google.com/blog/product-announcements/may-workspace-feature-drop-new-ai-features).

### Coding Agents: Gemini CLI

Writing code requires reading documentation. It requires understanding existing codebases. It requires knowing syntax. A coding assistant helps speed this up. The Gemini CLI tool leverages the ReAct architecture to implement a single-agent system for coding [[11]](https://blog.google/technology/developers/introducing-gemini-cli-open-source-ai-agent/), [[12]](https://cloud.google.com/gemini/docs/codeassist/gemini-cli).

Based on our research from August 2025, here is how it works at a high level. It is implemented in **TypeScript**. It shares similarities with tools like **Cursor**, **Windsurf**, **Claude Code**, and **Warp**.

1.  **Context Gathering:** The system loads the directory structure and conversation history into its working memory.
2.  **LLM Reasoning:** The model analyzes the request against the current context to plan necessary actions.
3.  **Human in the Loop:** Before executing potentially destructive actions, it validates the plan with the user.
4.  **Tool Execution:** It executes actions. These include **grep** to search for specific functions. It includes **listing directory structures** to understand the project layout. It includes **code interpreting** to run snippets. It includes **git** commands to commit changes. It then observes the output.
5.  **Evaluation:** It dynamically evaluates the generated code. It might run a compiler or linter.
6.  **Loop Decision:** The agent decides if the task is done. If not, it repeats the cycle to fix errors.

```mermaid
graph TD
    A["Context Gathering"] --> B["LLM Reasoning"]
    B --> C["Human in the Loop (Validation)"]
    C --> D["Tool Execution"]
    D --> E["Evaluation (Run/Compile Code)"]
    E --> F{"Loop Decision (Task Complete?)"}
    F -->|"No"| B
    F -->|"Yes"| G["End"]
```
Image 10: A flowchart illustrating the operational loop of the Gemini CLI coding assistant.

### Deep Research Hybrid Agents

Researching a new topic is daunting. You don't know where to start. You don't know which sources to trust. Perplexity's Deep Research agent solves this using a hybrid system that combines ReAct reasoning with workflow patterns [[13]](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research).

Unlike the single-agent Gemini CLI, this system likely uses multiple specialized agents orchestrated in parallel. While the code is closed-source, we can infer the architecture:

1.  **Research Planning & Decomposition:** The orchestrator analyzes the query. It breaks it down into targeted sub-questions using the orchestrator-worker pattern.
2.  **Parallel Information Gathering:** Specialized search agents run in parallel for each sub-question. They use web search actions to gather information in isolation. This keeps input tokens manageable.
3.  **Analysis & Synthesis:** Each agent validates and scores its sources based on credibility and relevance. It ranks the top results.
4.  **Iterative Refinement:** The orchestrator reviews the aggregated findings to identify knowledge gaps. If gaps exist, it generates follow-up queries and repeats the cycle.
5.  **Report Generation:** Finally, it compiles everything into a report with inline citations.

```mermaid
graph TD
    A["Research Planning & Decomposition (orchestrator decomposes query)"] --> B["Parallel Information Gathering (multiple specialized search agents use tools)"]
    B --> C["Analysis & Synthesis (agents validate, score, rank, summarize sources)"]
    C --> D{"Iterative Refinement & Gap Analysis (orchestrator identifies gaps, generates follow-up queries)"}
    D -->|"Gaps identified"| A
    D -->|"No gaps / Complete"| E["Report Generation"]
```
Image 11: Flowchart illustrating the iterative multi-step process of Perplexity's Deep Research agent.

This hybrid approach combines the structured planning of workflows with the dynamic adaptation of agents.

## Conclusion: The Challenges of Every AI Engineer

Now that you understand the spectrum from LLM workflows to AI agents, it is important to recognize that every AI Engineer faces these same fundamental challenges. Whether you are at a startup or a Fortune 500 company, you will battle these issues.

You will constantly battle a reliability crisis. Your agent works perfectly in demos. But it becomes unpredictable with real users. You will face context limits where systems lose coherence across long conversations. You will struggle with data integration. You must ensure garbage-in doesn't lead to garbage-out. You will face the cost-performance trap. Sophisticated agents are technically impressive but economically unviable. And you will always have to mitigate security concerns regarding autonomous write permissions.

The good news is that these challenges are solvable. In upcoming lessons, we will cover patterns for building reliable products through specialized evaluation pipelines. We will cover strategies for building hybrid systems. We will cover ways to keep costs under control.

By the end of this course, you will have the knowledge to architect AI systems that are not only powerful but also robust and safe. You will know when to use workflows versus agents. You will know how to build effective hybrid systems that work in the real world.

In the next lesson, **Lesson 3**, we will explore the specific **Workflow Patterns** we introduced today. We will examine how to implement chaining, routing, and the orchestrator-worker pattern in code. This lays the groundwork for the more complex agentic concepts like **tools** and **ReAct** coming later in the course.

## References

1. Butcher, M. (2025, June 18). Here are the 24 U.S. AI startups that have raised $100M or more in 2025. TechCrunch. https://techcrunch.com/2025/06/18/here-are-the-24-us-ai-startups-that-have-raised-100m-or-more-in-2025/
2. (n.d.). Agentic workflows vs autonomous AI agents — Do you know the difference?. GOpenAI. https://blog.gopenai.com/agentic-workflows-vs-autonomous-ai-agents-do-you-know-the-difference-c21c9bfb20ac
3. Bouchard, L. (n.d.). Real agents vs. workflows: The truth behind AI 'agents'. Louis Bouchard. https://www.louisbouchard.ai/agents-vs-workflows/
4. (n.d.). What is an AI agent?. Google Cloud. https://cloud.google.com/discover/what-are-ai-agents
5. Schluntz, E., & Zhang, B. (n.d.). Building effective agents. Anthropic. https://www.anthropic.com/engineering/building-effective-agents
6. (n.d.). A Developer’s Guide to Building Scalable AI: Workflows vs Agents. Towards Data Science. https://towardsdatascience.com/a-developer-s-guide-to-building-scalable-ai-workflows-vs-agents/
7. (n.d.). Autonomous generative AI agents are still under development. Deloitte. https://www.deloitte.com/us/en/insights/industry/technology/technology-media-and-telecom-predictions/2025/autonomous-generative-ai-agents-still-under-development.html
8. Karpathy, A. (n.d.). Software in the era of AI. Y Combinator. https://www.youtube.com/watch?v=LCEmiRjPEtQ
9. (2025, June). Summarize responses with Gemini in Google Forms. Google Workspace Updates. https://workspaceupdates.googleblog.com/2025/06/summarize-responses-with-gemini-google-forms.html
10. (2025, May). New AI features to help you work smarter in Google Workspace. Google Workspace Blog. https://workspace.google.com/blog/product-announcements/may-workspace-feature-drop-new-ai-features
11. (n.d.). Introducing Gemini CLI: your open-source AI agent. Google Blog. https://blog.google/technology/developers/introducing-gemini-cli-open-source-ai-agent/
12. (n.d.). Gemini CLI. Google Cloud. https://cloud.google.com/gemini/docs/codeassist/gemini-cli
13. (n.d.). Introducing Perplexity Deep Research. Perplexity Blog. https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research