## Global Context

- **Scope:** This is the 2nd lesson from module 1 of a broader course on AI agents and LLM workflows.
- **What I'm planning to share:** I want to talk about the two core methodologies of building AI applications, which are LLM Workflows (predefined, orchestrated steps) and Agentic Systems (dynamic, LLM-directed processes). We'll explain each method individually and then compare them by highlighting the pros and cons of each pattern. We'll explore use cases where each approach is most effective, emphasizing the core difference: developer-defined logic versus LLM-driven autonomy. We will highlight throughout the lesson that we will explain each individual concept in more detail in future lessons. We will conclude that most AI applications are a hybrid, a gradient, of both methods. Ultimately, we'll analyze the design and capabilities of prominent, state-of-the-art agent examples (as of 2025, such as deep research agents, coding agents, and task automation agents), deconstructing their operational mechanisms (e.g., planning, tool use, memory, multi-agent architecture) and highlighting common patterns and challenges.
- **Why I think it's valuable:** As an AI Engineer, when building AI applications, you will always have to make decisions between LLM Workflows and AI Agents. Choosing the right architecture is a critical early decision that impacts complexity, flexibility, and suitability for the AI project or product. Understanding where each method shines and how to integrate them effectively is one of the fundamental skills of a successful AI Engineer.
- **Who is the intended audience:** Aspiring AI Engineers learning for the first time about the specifics of LLM workflows, AI agents, and how they are different. The reader has only limited knowledge of LLMs. ONLY in future lessons of the course we will introduce the following concepts: context engineering, structured outputs, chaining, routing, plan and execute, tools, ReAct agents, memory and multimodal LLMs. Thus, use these concepts at a very intuitive level. Any parallels referencing these concepts should be understood by a 7-year-old.
- **Expected length of the article in words (without the titles and references)** (where 200-250 words ≈ 1 minute of reading time): 2650 words


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


## Outline

1. Introduction: The Critical Decision Every AI Engineer Faces
2. Understanding the Spectrum: From Workflows to Agents
3. Choosing your path: PROs and CONs
4. Exploring common patterns
5. Looking at State-of-the-Art (SOTA) examples (2025)
6. Zooming in on our favorite examples
7. The challenges of every AI Engineer

## Section 1 - Introduction: The Critical Decision Every AI Engineer Faces

- Begin with a compelling scenario: An AI Engineer sits down to build their first AI app. They have access to powerful LLMs, but they're faced with a fundamental question that will determine the success or failure of their project: How should they architect their AI system?
- **The Problem:** When building AI applications, engineers face a critical architectural decision early in their development process. Should they create a predictable, step-by-step workflow where they control every action, or should they build an autonomous agent that can think and decide for itself? This decision will impact everything from development time and costs to reliability and user experience.
- **Why This Decision Matters:** Choose the wrong approach and you might end up with:
  - An overly rigid system that breaks when users deviate from expected patterns
  - An unpredictable agent that works brilliantly 80% of the time but fails catastrophically when it matters most
  - Months of development time wasted rebuilding the entire architecture
  - Frustrated users who can't rely on your AI application
- **The Stakes Are High:** In 2024-2025, we've seen billion-dollar AI startups succeed or fail based primarily on this architectural decision. The companies that thrive understand when to use workflows versus agents, and more importantly, how to combine both approaches effectively.
- **What We'll Learn:** This lesson will give you the framework to make this critical decision confidently. You'll understand the fundamental trade-offs, see real-world examples from leading AI companies, and learn how to design systems that leverage the best of both approaches.
- **Section length:** 300 words

## Section 2 - Understanding the Spectrum: From Workflows to Agents

- In this section we want to take a brief look at what LLM Workflows and Agentic Systems are. At this point we don't focus on the technical specifics of each, but rather on their properties and how they are used.
- On **LLM Workflows** we care about:
	- Definition: Systems where a sequence of tasks, potentially involving LLM calls, is largely predefined and orchestrated by developer-written code.
	- Characteristics: The steps are defined in advance, resulting in deterministic or rule-based paths with predictable execution and explicit control flow. 
	- Analogy: A well-defined assembly line.
	- Concepts we will learn in future lessons: chaining, routing, planning and execution
	- Attach an image from the research with a simple LLM Workflow.
- On **Agentic Systems** we care about:
	- Definition: Systems where an LLM (or multiple LLMs) plays a central role in dynamically deciding the sequence of steps, reasoning, and actions to achieve a goal. The steps are not defined in advance, but are dynamically chosen based on the task and current state of the environment.
	- Characteristics: Adaptive, capable of handling novelty, LLM-driven autonomy in decision-making and execution path.
	- Analogy: A skilled human expert tackling an unfamiliar problem.
	- Concepts we will learn in future lessons: tools, planning, and ReAct agents
	- Attach an image from the research of how a simple Agentic System looks.
- **The Role of Orchestration:** Explain that both workflows and agents require an orchestration layer, but their nature differs. In workflows, it executes a defined plan; in agents, it facilitates the LLM's dynamic planning and execution.
- **Section length:** 300 words

## Section 3: Choosing your path: PROs and CONs

- In the previous section we defined each method independently, now we want to explore their core differences: Developer-defined logic (workflows) versus LLM-driven autonomy in reasoning and action selection (agents).
- **When to Use LLM Workflows:**
	- Examples: Structured data extraction and transformation, automated report generation from templates, content summarization followed by translation, form processing and content generation, such as articles or blogs (where the structure is well-defined and requires minimal human feedback).
	- Strengths: Predictability, reliability for well-defined tasks, easier debugging of fixed paths, potentially lower operational costs if simpler models can be used for sub-tasks.
	- Weaknesses: More development time required as each step is manually engineered. Rigidity in handling unexpected scenarios.
	- Usually preferred in enterprises.
- **When to Use Agentic Systems:**
	- Examples: Open-ended research and synthesis, dynamic problem-solving (e.g., debugging code, complex customer support), interactive task completion in unfamiliar environments, and creative content generation requiring iterative refinement (where the structure isn't well defined and needs more human feedback).
	- Strengths: Adaptability to new situations, flexibility to handle ambiguity and complexity as the steps are dynamically decided, and potential for emergent solutions.
	- Weaknesses: The system is more prone to errors. As the agent is non-deterministic, the performance, latency and costs can vary with each call of the agent, making agents often unreliable. 
- **Hybrid Approaches:** Most real-world systems blend elements of both approaches. Thus, in reality, we have a spectrum, a gradient between LLM Workflows and Agentic Systems, where a system adopts what's best from both worlds depending on its use cases.
- Highlight that when building an application you usually have an "autonomy slider" where you decide how much control to give to the LLM versus the user. As you go more manually, you usually use an LLM workflow together with a human that verifies intermediate steps. As you go more automatically, you give more control to the agent with fewer human-in-the-loop steps. Use the Cursor (CMD+K, CMD+L, CMD+I) and Perplexity (search, research, deep research) examples from the Andrej Karpathy "Software Is Changing (Again)" resource.
- The ultimate goal is to speed up the AI generation <-> Human verification loop, which is often achieved through good workflows/agentic architecture and well-designed UI/UX platforms (e.g., Cursor for coding).
- Attach an image from the research showing the gradient between LLM workflows and Agentic systems.
- **Section length:** 400 words

## Section 4: Exploring common patterns

- To introduce the reader to the AI Engineering world, let's introduce the most common patterns used to build AI agents and LLM workflows. Explain them as if this is the first time the reader hears about them.
- LLM Workflows:
	- chaining and routing to glue together multiple LLM calls
	- plan and execute to understand the user intent, call multiple functions, and synthesize them into a final answer (Draw a mermaid diagram with the plan and execute pattern)
	- an evaluator-optimizer loop used to auto-correct the results from an LLM based on a specified score (Draw a mermaid diagram with the evaluator-optimizer loop)
- AI Agents:
	- tool usage to interact and act with the external environment
	- short-term memory: The working memory of the agent (make a comparison to how RAM works for computers)
	- long-term memory: This is used to access factual data about the external world (such as public websites from the internet or private data from a company's databases)
- Add an image from the research showing the core components of an AI agent 
- The goal of this section is not for readers to fully understand how these patterns work, but just to build an intuition about what we will learn during the course. In future lessons, we will dig into all the necessary details. 

- **Section length:** 500 words

## Section 5: Looking at State-of-the-Art (SOTA) examples (2025)

- Introduce concrete LLM workflow examples, such as document summarization and analysis by Gemini in Google Workspace. This streamlines the summarization of emails, meetings, and documents, enhancing communication efficiency. Another significant application is in data-driven insights and decision-making, where companies like Geotab and Kinaxis utilize LLMs to analyze vast datasets, enabling real-time insights for supply chain optimization and fleet management. Additionally, content creation is transforming creative processes, with firms like Adobe and Procter & Gamble leveraging LLMs to generate localized marketing content and photo-realistic images, significantly reducing production time and costs. Lastly, legal and compliance automation is gaining traction, as seen with organizations like Fluna and FreshFields, which employ LLMs to automate legal document analysis and contract drafting, improving accuracy and reducing manual effort. 
- Introduce a selection of prominent, state-of-the-art (SOTA) agent examples, such as deep research agents (from OpenAI or Perplexity), coding assistants (Codex, Claude code, Gemini's CLI, Cursor, or Windsurf), task automation and computer use agents (OpenAI's Operator, Claude computer use) or other relevant 2025 examples.
- For each example:
	- Briefly describe the problem it solves and its functionality. 
	- Highlight what makes it an agent or workflow based on the definitions in Sections 1 and 2.
	- Discuss its potential impact or novelty.
-  **Section length:** 300 words (keep the section short as we will dig into more details in the next section)

## Section 6: Zooming in on our favorite examples

- In Section 5, we briefly described some common examples of agents and workflows. Now, we want to dig deeper into some examples and explain what they do, how they work, and how they combine both agents and workflows into cohesive products. 
- The reader is not yet aware of any complex topics and knows about LLM workflows or AI agents only what was explained previously in this article, so keep this section high-level without any fancy acronyms. Explain everything as if speaking to a 7-year-old.
- **Document summarization and analysis workflow by Gemini in Google Workspace**:
	- Explain how such a workflow would look. Explain that this is a pure and simple workflow as a chain with multiple LLM calls:
		- read document
		- summarize using an LLM call
		- extract key points using another LLM call
		- save results to a database
	- Create a suggestive mermaid diagram highlighting that this is a workflow.
- **Gemini CLI coding assistant:** 
	- To anchor the example for future lessons, specify at a high level that it uses the ReAct pattern to implement the autonomous agentic behavior (agent)
	- Here is a simplified example of how a Gemini CLI for coding with ReAct would look like:
		1. **Planning with ReAct**: Plan what to code
		2. **Reading the Current State of the Project with Tools** Reading files, understanding project structure.
		3. **Code Generation or Modification with Tools** Writing or editing code.
		4. **Execution & Testing with Tools** Running the code, using a linter, running tests.
		5. **Debugging & Iteration with ReAct** Analyzing errors, modifying the plan, and trying again.
	- Functions (tools) used to access the external environment: File system access, code interpreter, web search (for documentation/solutions), version control.
	- Create a mermaid diagram showing how the operational loop works.
- **Perplexity deep research (e.g., for scientific discovery or market research):**
	- To anchor the example for future lessons, specify at a high level that it could be implemented using the plan and execute pattern to gather information from multiple sources (workflow)
	- Here is a simplified example of how a deep research agent with plan and execute would work:
		1. **Query Formulation & Planning with Plan and Execute** Defining research questions as a plan of attack. Each question will be searched on a search engine.
		2. **Search using Tools** Using search engine tool.
		4. **Synthesis & Analysis:** Use an LLM to combine the information from the search tools.
		5. **Citation & Reporting:** Generating a structured output with proper references.
	- Explain how it operates both as an agent and as a workflow, combining structured planning with dynamic adaptation. 
	- Create a mermaid diagram showing how the iterative multi-step process works.
- **ChatGPT Agent:**
	- Explain how the ChatGPT agent works at a higher level than the previous examples.
	- **What it is:** It uses the ReAct pattern to autonomously think, act, and complete complex digital tasks for users by proactively choosing from a set of agentic skills and tools.
	- **Key features:** It uses tools to browse the web, generate downloadable files like PowerPoint presentations and Excel spreadsheets, automate workflows, interact with web page elements, and even control aspects of a computer to fulfill user requests.
	- **Challenges:** Key problems include handling complex multi-step tasks reliably, addressing ethical concerns and data biases, and ensuring user safety and privacy.
-  **Section length:** 500 words (without counting the mermaid diagram code)

## Section 7 - Conclusion: The challenges of every AI Engineer

- **The Reality of AI Engineering:** Now that you understand the spectrum from workflows to agents, it's important to recognize that every AI Engineer—whether working at a startup or a Fortune 500 company faces these same fundamental challenges on a daily basis. These are the real decisions that determine whether your AI application succeeds in production or fails spectacularly.
- **Daily Challenges Every AI Engineer Battles:**
	- **Reliability Crisis:** Your agent works perfectly in demos but becomes unpredictable with real users. LLM reasoning fails, errors compound through multi-step processes, and your "intelligent" system starts making nonsensical decisions.
	- **Context Chaos:** You're constantly battling context limits, trying to maintain coherence across long conversations while your workflow or agent gradually loses track of what it's supposed to be doing.
	- **Data Integration Nightmare:** Every day you're pulling data from Slack, web APIs, Zoom transcripts, SQL databases, and data lakes—but your system struggles to process this multimodal information coherently.
	- **The Cost-Performance Trap:** Your sophisticated agent delivers amazing results but costs $50 per user interaction, while your efficient workflow feels robotic and inflexible.
	- **Security Paranoia:** You're constantly worried about your autonomous agent sending the wrong email, deleting critical files, or exposing sensitive data because it has powerful write permissions.
- **The Good News:** These challenges aren't impossible to overcome. In fact, they're exactly what this course is designed to solve. Throughout our upcoming lessons, we'll tackle each of these problems systematically. You'll learn battle-tested patterns for reliability, proven strategies for context management, practical approaches to multimodal data handling, cost optimization techniques, and security frameworks that let you sleep at night.
- **Your Path Forward:** By the end of this course—especially during our capstone project—you'll have the tools and knowledge to architect AI systems that handle these daily realities gracefully. You'll know when to choose workflows versus agents, how to architect hybrid approaches effectively, and most importantly, how to build AI applications that work reliably in the real world.
- **Section length:** 350 words

## Golden Sources

[Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
[What is an AI agent?](https://cloud.google.com/discover/what-are-ai-agents)
[Real Agents vs. Workflows: The Truth Behind AI 'Agents'](https://www.youtube.com/watch?v=kQxr-uOxw2o&t=1s)
[Exploring the difference between agents and workflows](https://decodingml.substack.com/p/llmops-for-production-agentic-rag)
[A Developer’s Guide to Building Scalable AI: Workflows vs Agents](https://towardsdatascience.com/a-developers-guide-to-building-scalable-ai-workflows-vs-agents/)


## Other Sources

[601 real-world gen AI use cases from the world's leading organizations](https://cloud.google.com/transform/101-real-world-generative-ai-use-cases-from-industry-leaders)
[Stop Building AI Agents: Here’s what you should build instead](https://decodingml.substack.com/p/stop-building-ai-agents)
[Andrej Karpathy: Software Is Changing (Again)](https://www.youtube.com/watch?v=LCEmiRjPEtQ)
[Building Production-Ready RAG Applications: Jerry Liu](https://www.youtube.com/watch?v=TRjq7t2Ms5I)
[Claude Code](https://blog.google/technology/developers/introducing-gemini-cli-open-source-ai-agent/)
[Introducing Perplexity Deep Research](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research)
[Introducing ChatGPT agent: bridging research and action](https://openai.com/index/introducing-chatgpt-agent/)
