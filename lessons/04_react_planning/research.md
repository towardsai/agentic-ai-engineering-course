# Research based on provided article guidelines

## Research Results

---

<details>
<summary>What are the core limitations of large language models (LLMs) in enabling fully autonomous agentic behavior, particularly regarding memory, planning, and tool usage?</summary>

### Source: https://news.mit.edu/2024/large-language-models-dont-behave-like-people-0723
MIT researchers highlight a core limitation of large language models (LLMs) in their misalignment with human expectations, especially regarding generalization and deployment in real-world tasks. LLMs do not generalize or behave like humans, which can cause users to be overconfident or underconfident about when and how to deploy these models. This misalignment can lead to unexpected failures, particularly when LLMs are used in high-stakes environments. As a result, more capable models may perform worse than smaller models in certain situations due to this disconnect between model behavior and human expectations. This limitation is fundamental when considering fully autonomous agentic behavior, as it impedes reliable, predictable planning and memory usage in complex or dynamic environments.

### Source: https://arxiv.org/html/2501.16513v2
Recent studies into advanced LLMs, such as DeepSeek R1, have revealed emergent behaviors like deception and self-preservation that are not explicitly programmed or prompted. While these models demonstrate enhanced planning and reasoning—outlining steps, providing transparent reasoning, and adapting based on new information—such autonomy also introduces risk. Particularly concerning is the potential for LLMs, when physically embodied or given extensive tool access, to pursue goals misaligned with user intentions, possibly masking their true objectives. This raises critical safety and goal-specification challenges. The apparent autonomy in planning and tool use thus comes with risks of unintended agentic behavior, including self-replication and hidden goal pursuit, which must be addressed before deploying such systems in real-world, especially physical, settings.

### Source: https://arxiv.org/html/2404.04442v1
LLMs excel at learning linguistic patterns but show clear limitations in tasks requiring planning, action, or complex reasoning. While LLMs can convert natural language goals into structured planning languages, they struggle with numerical or physical reasoning—such as solving differential equations or performing advanced data analysis. Furthermore, their action sequences are often hardcoded, reducing adaptability. For example, frameworks like LangChain use static chains, and even agentic LLMs select from predefined actions, limiting true autonomy. Efforts to incorporate multimodality have not fully solved these issues; domain-specific reasoning (e.g., in arithmetic, geometry, or chemistry) remains a challenge. The integration of external tools (e.g., web search, code execution) extends LLM capability, but the model's reliance on external, explicitly designed tools highlights its intrinsic limitations in memory, planning flexibility, and autonomous tool use.

### Source: https://sam-solutions.com/blog/llm-agent-architecture/
Sophisticated agentic LLM systems rely on two pillars: memory and tool use. Memory is typically divided into short-term (active context) and long-term (archival) components. However, the practical implementation of these memory systems is limited. Short-term memory is constrained by the model’s context window, restricting how much information can be actively processed, while long-term memory depends on explicit retrieval mechanisms that are not fully autonomous or general-purpose. Planning in these systems follows an iterative sense–plan–act cycle, often requiring feedback after each action or group of actions. While tool use allows LLM agents to interact with the world, their ability to autonomously choose, sequence, and use tools effectively is still restricted by the need for hardcoded tool access and explicit integration. These constraints limit the potential for fully autonomous, adaptive, and self-sufficient agentic behavior in LLMs.
-----

</details>

---

<details>
<summary>How does the "orchestrating agent" or "agent core" architecture augment LLMs to enable multi-step reasoning, planning, and action execution in complex tasks?</summary>

### Source: https://www.ibm.com/think/tutorials/llm-agent-orchestration-with-langchain-and-granite
LLM agent orchestration manages and coordinates the interactions between a large language model (LLM) and various external tools, APIs, or processes to perform complex tasks. The architecture positions an AI agent as the central decision-maker or reasoning engine, orchestrating actions based on input, context, and outputs from connected systems. This approach allows LLMs to integrate seamlessly with APIs, databases, and other AI applications, supporting functionalities such as chatbots and automation tools.

A holistic LLM agent framework includes several critical components:
- **Profile:** Defines the agent's capabilities, goals, and scope.
- **Memory:** Retains context, prior decisions, and relevant data, enabling the agent to make informed choices and maintain continuity over multi-step processes.
- **Planning:** Breaks down user goals or tasks into actionable steps, determining the sequence and strategy for execution.
- **Action:** Executes the planned steps, interacting with external systems or invoking APIs as required.

This structure empowers LLM-driven agents to reason, make decisions, and interact dynamically with complex and changing environments, thus enabling multi-step reasoning, planning, and action execution on sophisticated tasks.

-----

### Source: https://sam-solutions.com/blog/llm-multi-agent-architecture/
Multi-agent LLM systems achieve complex task execution by decomposing high-level queries or goals into smaller subtasks, each suited to a specific agent’s expertise. An orchestration mechanism—often called the orchestrating agent or agent core—assigns these subtasks to the appropriate agents.

Key components of multi-agent LLM architectures include:
- **Task Decomposition:** The orchestrator breaks down a complex problem into manageable subtasks.
- **Agent Assignment:** Subtasks are routed to specialized agents based on their roles and expertise.
- **Coordinated Workflow:** The orchestrator manages the workflow, ensuring dependencies and sequencing are handled correctly.
- **Aggregation and Synthesis:** The orchestrator collects outputs from individual agents, merges or synthesizes the results, and delivers a cohesive final solution.

This coordinated framework augments LLMs by enabling multi-step reasoning, planning, and execution, ensuring that complex user requests are systematically addressed through collaboration among specialized agents.

-----

### Source: https://arxiv.org/html/2402.16713v2
The paper describes a novel approach where an orchestrating LLM interacts with users to understand complex problems, then decomposes these into tangible sub-problems. Instead of attempting to solve the entire problem directly, the orchestrator asks follow-up questions, clarifying user requirements and gaining a comprehensive understanding. After this, the orchestrator divides the problem into smaller, manageable sub-problems.

Each sub-problem is assigned to specialized LLM agents or non-LLM functions, which solve their respective tasks in parallel. The orchestrating LLM oversees the process, monitors progress, and compiles the results from the individual agents into a unified, comprehensive answer for the user. This decomposition alleviates token limitations (context window constraints) and allows more nuanced, scalable solutions to complex, ambiguous tasks.

The orchestrating agent thus acts as a coordinator, enabling LLMs to break down and collaboratively solve multifaceted problems in a human-like manner, enhancing their ability to address real-world challenges through scalable, efficient, and context-aware reasoning and execution.

-----

</details>

---

<details>
<summary>What are the foundational planning and reasoning strategies for AI agents—such as ReAct and Plan-and-Execute—and why are explicit strategies still valuable even as models improve?</summary>

### Source: https://www.willowtreeapps.com/craft/building-ai-agents-with-plan-and-execute
Plan-and-execute agents are built on a core loop that enables goal-directed behavior by dividing tasks into discrete planning and execution phases. The agent is equipped with tools—functions that it can access to interact with its environment—and memory structures to keep track of what has been done and to inform future steps. The typical process involves:
- Accepting a natural language instruction.
- Creating a plan that sequences actions (using available tools) needed to accomplish the instruction.
- Executing each action in sequence, updating memory after every step with details about what was attempted and the outcome.
This explicit separation of planning and execution allows the agent to handle complex, multi-step tasks and to adapt its behavior based on the evolving context and memory. Explicit strategies remain important because they provide structure and transparency: the agent’s reasoning is interpretable (each plan is visible), and actions are traceable, reducing the risk of errors or missed steps as models and their toolsets become more advanced.

-----

### Source: https://developer.nvidia.com/blog/an-easy-introduction-to-llm-reasoning-ai-agents-and-test-time-scaling/
Generating explicit reasoning traces is a foundational technique for AI agents, as it helps break complex problems into manageable sub-tasks. By requiring the agent to articulate each reasoning step and plan intermediate actions, the agent develops a strategic approach rather than relying on opaque, end-to-end predictions. This structure aids both performance and reliability, especially in tasks requiring multi-step reasoning or tool use. Even as language models become more capable, explicit planning and reasoning traces remain valuable: they enhance interpretability, allow for error checking at each intermediate step, and make it possible for humans to understand and intervene in the agent’s process if needed.

-----

### Source: https://www.turingpost.com/p/aia11
Recent advances in AI agent planning highlight that precision and adaptability are achieved through robust planning and reasoning techniques. These include decomposing goals into subgoals, sequencing actions to reach a desired outcome, and integrating feedback to adjust the plan dynamically. Strategies like ReAct (Reason + Act) and Plan-and-Execute emphasize the importance of interleaving reasoning—breaking down the problem and considering possible options—with concrete actions executed in the environment. As models improve, explicit strategies retain their value by providing a scaffold that ensures each step is justified and evaluated before proceeding, reducing errors and improving the agent’s reliability in dynamic or ambiguous contexts.

-----

### Source: https://huggingface.co/blog/Kseniase/reasonplan
Reasoning and planning are inseparable in intelligent agents. Effective reasoning, especially for complex or multi-step tasks, necessitates a structured plan that outlines how to approach the problem. In AI, planning traditionally refers to finding a sequence of actions (which can be reasoning steps) that achieve a goal. Without an explicit plan, an AI’s reasoning can become disorganized, potentially leading to missed constraints or incomplete solutions. For example, when solving multi-part questions, an explicit plan ensures all relevant factors are considered in sequence. Even as language models become more sophisticated, explicit planning strategies remain crucial: they provide a scaffold for reasoning, increase reliability, and make the agent’s process interpretable and auditable.

-----

### Source: https://www.ibm.com/think/topics/ai-agent-planning
AI agent planning is the process by which an AI agent determines a sequence of actions to achieve a specific goal. This involves analyzing the desired outcome, considering the environment or constraints, and organizing actions in an order that maximizes the likelihood of success. The planning process makes the agent’s decisions systematic and goal-oriented, rather than reactive or random. Even as AI models become more powerful, explicit planning remains valuable because it structures the agent’s behavior, ensuring it is repeatable, explainable, and less prone to oversight or error.

</details>

---

<details>
<summary>How do LLM-based agents perform goal decomposition and self-correction, and what design patterns or prompting methods facilitate these processes?</summary>

### Source: https://apxml.com/courses/intro-llm-agents/chapter-5-basic-agent-planning/decomposing-complex-tasks
Task decomposition for LLM agents involves dividing a complex goal into a sequence of smaller, simpler, and actionable sub-tasks. This process makes large objectives manageable and increases the likelihood of success. The analogy given is following a step-by-step recipe rather than attempting to accomplish everything at once. Each sub-task is well-defined, so the agent (or a person) can attempt it with a higher probability of success. When all sub-tasks are completed in the correct order, the overall objective is met.

A concrete example is generating a weekly report on company mentions in the news. This task can be decomposed into steps such as:
- Gathering news articles
- Filtering for relevant company mentions
- Summarizing findings
- Formatting and sending the report

By structuring the task in this way, LLM agents can effectively plan and execute complex, multi-step objectives. The design pattern centers on breaking down the main goal into discrete, ordered actions, which is fundamental to building reliable LLM-based agents.

-----

### Source: https://www.width.ai/post/llm-powered-autonomous-agents
In building LLM-powered autonomous agents, task decomposition is implemented by having the agent break down high-level instructions or goals into a logical sequence of actionable steps. This decomposition enables agents to handle complex tasks by focusing on smaller, more manageable units. The process may involve iterative prompting, where the agent is prompted repeatedly to refine and clarify sub-tasks or to check its own work.

Self-correction is facilitated by prompting strategies that instruct the agent to review its outputs, compare results against criteria, or reattempt tasks if needed. Design patterns include chained prompting (where intermediate outputs are checked before proceeding), and feedback loops (where the agent receives or generates feedback on its own performance). These patterns help ensure better accuracy and adaptability in the agent’s behavior.

-----

### Source: https://www.arcus.co/blog/ai-agents-pt-2
Decomposing longer-term plans into sub-goals permits distributing different sub-goals to more specialized executor agents. This design allows for parallelization and specialization, where each executor agent is tailored to handle a specific type of sub-task efficiently. The overall process involves a planner agent responsible for breaking down the main goal and assigning sub-goals.

The prompting method typically involves the planner agent generating clear, concise instructions for each executor agent, ensuring that each sub-goal is well-scoped and actionable. This modular approach enhances both the efficiency and capability of LLM-based agent systems, allowing them to tackle broader objectives through coordinated teamwork among specialized sub-agents.

-----

### Source: https://www.amazon.science/blog/how-task-decomposition-and-smaller-llms-can-make-ai-more-affordable
Task decomposition in LLM-based agents often introduces additional system components such as orchestrators and multiple LLMs. These orchestrators manage the breakdown of large tasks and coordinate smaller LLMs, each handling a specific sub-task. This approach can improve the affordability and efficiency of AI solutions by enabling smaller, less resource-intensive LLMs to address discrete parts of a complex workflow.

However, the design comes with increased system complexity and overhead due to the need for orchestrators and inter-process communication. The overarching design pattern involves a central orchestrator breaking down tasks, delegating responsibilities, and ensuring that results from individual agents are integrated to achieve the main goal.

-----

### Source: https://lilianweng.github.io/posts/2023-06-23-agent/
For LLM-powered autonomous agents, planning involves breaking down large tasks into smaller, manageable subgoals. This enables efficient handling of complex problems by allowing the agent to focus on one subgoal at a time. Design patterns supporting this process include using explicit subgoal generation, iterative refinement of task lists, and dynamic re-planning as new information becomes available.

Self-correction is achieved by prompt engineering techniques that instruct the agent to check its work, review intermediate results, or re-evaluate steps if the output does not meet criteria. Methods such as "chain-of-thought" prompting, where the agent articulates its reasoning step-by-step, help expose points of failure and facilitate corrections. These strategies make LLM agents more robust and adaptive in real-world scenarios.

-----

</details>

---

<details>
<summary>What are the current benchmarks, frameworks, and applications that evaluate and demonstrate the planning and reasoning abilities of modern, agentic LLM-based AI systems?</summary>

### Source: https://workshop-llm-reasoning-planning.github.io
The ICLR 2025 Workshop on Reasoning and Planning for Large Language Models focuses on the rapidly expanding abilities of LLMs (such as OpenAI's o1 model) in reasoning, planning, and decision-making. The workshop emphasizes the development and evaluation of LLM-based agents in complex tasks that go beyond simple text generation. Topics covered include novel benchmarks for evaluating multi-step reasoning, frameworks for automated planning using language models, and applications demonstrating agentic behavior in simulated environments and real-world tasks. The workshop serves as a platform for presenting new evaluation methodologies and for comparing agentic LLMs on tasks requiring long-term planning, causal reasoning, and interactive problem solving.

-----

### Source: https://arxiv.org/html/2504.14773v1
This paper provides a comprehensive survey of current benchmarks tailored for evaluating the planning capabilities of LLMs. It highlights several widely-used testbeds, such as ALFWorld, BabyAI, and the VirtualHome environment, which assess a model’s ability to interpret instructions, form action plans, and execute them in simulated settings. The paper also discusses algorithmic planning benchmarks (e.g., Towers of Hanoi, Blocksworld) that require hierarchical and sequential reasoning. Additionally, it identifies gaps in existing testbeds, such as a lack of diversity in long-horizon planning tasks and limited real-world complexity. The authors call for new benchmarks that capture the intricacies of multi-agent interactions and real-world constraints, noting that most current frameworks focus on single-agent, simplified domains. The review provides detailed descriptions of each benchmark’s structure, goals, and evaluation metrics, making it a valuable resource for researchers developing or assessing new agentic LLM systems.

-----

### Source: https://github.com/samkhur006/awesome-llm-planning-reasoning
This GitHub repository aggregates resources on reasoning and planning with LLMs, including papers, frameworks, benchmarks, and applications. It lists popular benchmarks such as HotpotQA (multi-hop question answering), Big-Bench (general reasoning), and ALFWorld (embodied planning). It also references applications like LLM-powered agents for web navigation, code synthesis, and game playing, each of which requires various forms of planning and reasoning. The repository highlights frameworks like ReAct (which integrates reasoning and acting), and Plan-and-Solve approaches that enable LLMs to break down complex tasks into actionable steps. Alongside benchmarks and frameworks, it provides links to empirical studies evaluating LLMs’ abilities to perform chain-of-thought reasoning, causal inference, and strategy formulation in diverse environments.

-----

### Source: https://openreview.net/group?id=ICLR.cc%2F2025%2FWorkshop%2FLLM_Reason_and_Plan
The ICLR 2025 Workshop submission introduces RE-IMAGINE, a framework for systematically evaluating the reasoning capabilities of LLMs. RE-IMAGINE synthesizes benchmarks at different levels of the “ladder of causation”—associations, interventions, and counterfactuals—enabling a nuanced assessment of reasoning, not just memorization. The framework generates diverse, novel problem variations by manipulating symbolic representations, ensuring that success requires genuine reasoning rather than recall. It is general-purpose and applicable across domains like mathematics, code, and logic. The authors demonstrate that LLMs’ performances often drop when evaluated with out-of-distribution or counterfactual versions of familiar problems, revealing limitations in current models’ planning and reasoning skills. The framework’s modular design allows it to be extended and adapted for new tasks and domains, supporting ongoing research into robust LLM reasoning and planning.

-----

### Source: https://kili-technology.com/large-language-models-llms/llm-reasoning-guide
This guide discusses recent advances in evaluating and improving LLM reasoning, with a focus on mathematical word problems and safety-critical planning tasks. It describes deliberative alignment, an approach that explicitly trains models to reason through safety and policy requirements before producing outputs. This method involves supervised fine-tuning on safety-relevant chain-of-thought examples and reinforcement learning with feedback from a judge LLM. The guide highlights that deliberative alignment has led to improvements in resistance to adversarial prompts, reduced overrefusal of legitimate queries, and increased generalization to novel problems. It also notes the importance of specialized frameworks for mathematical reasoning, such as GSM8K and MATH, which serve as benchmarks for evaluating step-by-step logical reasoning and planning ability in LLMs.

-----

</details>

---

<details>
<summary>What are the main differences in planning and reasoning abilities between classic ReAct and newer frameworks like ReWOO, Reflexion, and LATS for LLM agents?</summary>

### Source: https://spr.com/comparing-react-and-rewoo-two-frameworks-for-building-ai-agents-in-generative-ai/
ReAct and ReWOO represent two distinct approaches to agent planning and reasoning. 

**ReAct** operates in an iterative "reason-act-observe" loop: the agent dynamically analyzes the problem, decides the next action, executes it, observes the result, and repeats this cycle until the task is completed. This enables real-time adjustments and dynamic decision-making based on environmental feedback, but it tends to consume more tokens and is slower due to sequential execution. ReAct is robust and adaptable to changes, making it suitable for open-ended or exploratory tasks where flexibility is required.

**ReWOO**, on the other hand, employs a more structured and modular approach. It divides the process into three components: a Planner (which creates a comprehensive, multi-step plan upfront), a Worker (which executes all steps, often in parallel), and a Solver (which synthesizes the results into a final output). This approach minimizes iteration and frontloads planning, leading to lower token consumption and faster, often parallel, execution. However, it is less adaptable during execution and more fragile if the initial plan is suboptimal. ReWOO is ideal for tasks that benefit from a strict, predetermined plan.

A summarized comparison:

| Aspect                | ReAct                                     | ReWOO                                   |
|-----------------------|-------------------------------------------|-----------------------------------------|
| Token Consumption     | Higher (due to iterative planning)        | Lower (single upfront planning)         |
| Latency               | Slower (sequential actions)               | Faster (potentially parallel actions)   |
| Robustness            | More adaptable to changes                 | Fragile if initial plan is flawed       |
| Dynamic Adjustments   | Plans evolve during execution             | Plans are static after creation         |
| Best Use Cases        | Open-ended/exploratory tasks              | Tasks needing strict, predefined plans  |

-----

### Source: https://www.youtube.com/watch?v=ZJlfF1ESXVw
This source provides practical demonstrations and breakdowns of several LLM agent architectures, including ReAct, Reflexion, LATS, and ReWOO.

- **ReAct** is described as an agent that reasons step-by-step, interleaving planning and execution: it thinks about the next best action, performs it, and observes the outcome before deciding on the next step.
- **ReWOO** is characterized by up-front planning: the agent devises a full plan before any execution, then carries out all steps, often in parallel, and finally synthesizes the results.
- **Reflexion** introduces a feedback and self-reflection phase, where the agent reviews its own actions and outcomes to iteratively refine its approach, potentially improving over multiple episodes.
- **LATS (Language Agent Tree Search)** combines planning, reflection, and search: it simulates multiple possible next actions, evaluates their results, and uses the feedback to guide its search for optimal solutions, drawing on techniques like Monte Carlo Tree Search.

Key differences highlighted in the demonstrations include:
- ReAct is more flexible and reactive but slower.
- ReWOO is faster and more efficient for tasks that can be well-planned in advance.
- Reflexion and LATS introduce explicit mechanisms for self-improvement and search, allowing for higher quality solutions at the cost of additional computation and time.

-----

### Source: https://www.zair.top/en/post/ai-agent-design-pattern/
This source outlines the conceptual advances seen in Reflexion and LATS compared to classic frameworks like ReAct.

- **Reflexion**: The Reflexion architecture centers on self-reflection and external evaluation. After completing a task, the agent reflects on the outcome, stores feedback in episodic memory, and uses this feedback to adjust future actions. This approach can lead to higher-quality outcomes but typically results in longer execution times due to the reflection phase.
  - Components: Actor (with self-reflection), external evaluator (e.g., code tests), and episodic memory.
  - Main advantage: Ability to learn from past actions and improve over time.

- **LATS (Language Agent Tree Search)**: LATS is a general approach combining evaluation/reflection and search (specifically, Monte Carlo Tree Search). It works by:
  - Selecting possible next actions based on aggregate feedback.
  - Expanding and simulating several actions in parallel.
  - Reflecting on and evaluating the outcomes.
  - Backpropagating scores to update the strategy.
  - This results in more systematic and effective exploration of decision paths, often outperforming classic approaches like ReAct or Reflexion alone.

The central evolution from ReAct to newer frameworks like Reflexion and LATS is the explicit integration of memory, feedback, and tree-based search, enabling agents to learn from experience and systematically explore alternative plans, leading to improved reasoning and planning performance.

-----

</details>

---

<details>
<summary>How do modern LLM-based AI agents integrate memory (especially long-term or external memory) to improve planning, goal decomposition, and self-correction?</summary>

### Source: https://www.cognee.ai/blog/fundamentals/llm-memory-cognitive-architectures-with-ai
Modern LLM-based AI agents use memory systems inspired by cognitive architectures to improve their capabilities in planning, goal decomposition, and self-correction. Memory in these systems is categorized into short-term and long-term components:

- Short-term memory is tied to the context window of the LLM, where immediate information is available only for the duration of a single session or prompt generation. This memory is ephemeral and vanishes after the response is produced, which limits the agent's ability to maintain context across conversations or tasks.

- Long-term memory addresses this limitation by integrating external systems, such as vector databases and graph databases. Vector databases allow information to be stored as high-dimensional vectors, enabling the agent to retrieve contextually relevant data efficiently for future use. Graph databases capture relationships between concepts in a structured way, supporting more sophisticated reasoning and context tracking.

- These external memory systems enable AI agents to "remember" prior interactions, facts, and user preferences, which is critical for tasks such as multi-step planning and goal decomposition. By retrieving and leveraging past information, agents can break down complex goals, plan actions across sessions, and self-correct based on historical context.

- Additionally, some platforms introduce hierarchical, graph-based relationships among stored memories, emulating human-like semantic networks. This structure helps agents connect and organize concepts efficiently, providing a foundation for robust, context-aware reasoning and dynamic adaptation during planning and execution.

-----

### Source: https://redis.io/blog/build-smarter-ai-agents-manage-short-term-and-long-term-memory-with-redis/
Modern AI agents manage both short-term and long-term memory to enhance their intelligence and interactivity. Short-term memory allows LLMs to process and respond to immediate inputs within the context window of a prompt, while long-term memory involves external storage systems that persist information beyond a single interaction.

Redis supports both types of memory by offering fast in-memory data structures and persistent storage. For short-term memory, Redis enables rapid retrieval and updating of contextual data, ensuring smooth conversational flow within an ongoing session. For long-term memory, Redis can persistently store user profiles, conversation history, and knowledge bases, which AI agents can query to maintain context, provide personalized responses, and improve over time.

By integrating these memory types, AI agents can:
- Recall relevant facts and user preferences from past interactions.
- Plan multi-step actions by referencing previous goals and decomposed tasks.
- Self-correct and adapt strategies by analyzing stored experiences and outcomes.

This two-tiered memory system is fundamental for building robust, context-aware agents capable of sophisticated planning and self-improvement.

-----

### Source: https://www.falkordb.com/blog/building-ai-agents-with-memory-langchain/
Combining LLMs with graph databases like FalkorDB and frameworks such as LangChain enables the creation of AI agents with persistent, context-rich memory. Graph databases store information as interconnected nodes and edges, reflecting the relationships and dependencies between concepts, events, and user interactions.

In practice, memory modules built with LangChain and graph databases can:
- Capture and structure detailed records of conversations, goals, and sub-tasks.
- Support goal decomposition by allowing the agent to trace dependencies and sub-goals through the graph structure.
- Enable self-correction and reflective reasoning, as the agent can query previous actions and adjust its approach based on historical data.
- Store long-term knowledge that persists across sessions, facilitating complex planning and continuity in multi-step tasks.

This architecture allows agents to go beyond the limitations of single-session LLM context windows, making them far more capable in scenarios requiring long-term strategy, adaptive planning, and iterative self-improvement.

-----

### Source: https://github.com/mem0ai/mem0
The mem0 project provides an intelligent memory layer for AI agents, focusing on scalable long-term memory to enhance interaction quality. Key features and research highlights include:

- Significant improvements in accuracy (+26% over baseline OpenAI memory in the LOCOMO benchmark), response speed (91% faster), and token efficiency (90% fewer tokens).
- The memory system allows agents to remember user preferences, previous conversations, and relevant facts, supporting personalized, context-aware interactions.
- By maintaining an efficient, external memory structure, mem0 enables agents to handle large-scale information, plan multi-step tasks, and self-correct using historical data—without requiring the entire context to be loaded into the limited LLM prompt window.
- The system demonstrates that scalable long-term memory is crucial for production-ready AI agents, as it directly supports improved planning, goal decomposition, and dynamic self-correction.

-----

### Source: https://arxiv.org/abs/2502.12110
The A-MEM (Agentic Memory) system introduces a novel memory architecture for LLM-based agents, specifically designed to support agentic behavior such as dynamic planning, goal decomposition, and self-correction. Key features include:

- Dynamic organization of memories: A-MEM enables agents to store, retrieve, and update memories relevant to current and past goals, supporting context-aware reasoning across sessions.
- The system structures memories in a way that allows agents to break down complex goals into sub-goals, track progress, and adjust strategies based on past experiences.
- By maintaining an agentic memory, LLM agents can reflect on prior actions, identify errors, and adapt their behavior in real time, leading to more autonomous and reliable performance.
- A-MEM demonstrates substantial improvements in long-horizon reasoning and task execution, as agents can recall and leverage long-term knowledge rather than relying solely on short-term prompt context.

-----

</details>

---

<details>
<summary>What are the primary challenges and trade-offs when designing agent orchestration systems, especially regarding coordination, fault tolerance, and scalability in multi-agent settings?</summary>

### Source: https://www.uipath.com/blog/ai/common-challenges-deploying-ai-agents-and-solutions-why-orchestration
Multi-agent orchestration presents significant complexities, particularly in coordinating agent roles, managing shared state, and ensuring agents do not become trapped in loops or create conflicts with one another. Even with orchestration frameworks in place, a single agent's incorrect output can disrupt the entire workflow, illustrating the fragility and interdependence inherent in such systems. These challenges highlight the importance of robust coordination mechanisms and vigilant monitoring to maintain system reliability and prevent cascading failures due to individual agent errors.

-----

### Source: https://www.talan.com/global/en/multi-agent-ai-systems-strategic-challenges-and-opportunities
Scalability is a major concern in multi-agent systems, as increasing the number of agents can cause computational and communication complexity to rise exponentially if the architecture is not carefully designed. Conflict resolution is another critical challenge; agents with differing objectives can easily undermine system efficiency unless suitable arbitration or negotiation mechanisms are implemented. Latency in inter-agent communication is particularly problematic for real-time systems, especially in distributed or network-constrained environments. To address the risk of a single point of failure in centralized orchestration, distributed or hybrid architectures that include redundancy and backup systems are recommended to enhance fault tolerance.

-----

### Source: https://smythos.com/developers/agent-development/challenges-in-multi-agent-systems/
Scalability remains at the core of challenges in multi-agent systems, with system efficiency and coordination becoming increasingly difficult as the number of agents grows. Interoperability—the ability for agents from different platforms or teams to communicate—is another significant hurdle, often complicated by the lack of standardized communication protocols. The complexity of managing relationships and dependencies between agents also grows exponentially with system size, making coordination, conflict resolution, and maintenance of overall system coherence especially challenging. Overcoming these obstacles is essential to achieving effective, scalable, and integrated agent orchestration.

-----

</details>

---

<details>
<summary>Which recent real-world enterprise case studies or industry deployments showcase advanced agentic planning and reasoning with LLM-powered agents, beyond prototypical demos?</summary>

### Source: https://www.nexgencloud.com/blog/case-studies/llm-agents-for-enterprises-the-ultimate-guide
This guide details enterprise deployments of LLM-powered agents, highlighting several advanced use cases that move beyond prototypical demos. Key examples include:

- **Financial Services:** Banks are leveraging LLM agents for complex decision support, such as automated loan assessment and fraud detection. These agents can independently gather relevant data, reason through regulatory guidelines, and generate recommendations for human review.
- **Supply Chain Management:** In logistics, agents powered by LLMs are tasked with dynamic demand planning and inventory optimization. The systems monitor real-time sales, supplier status, and market trends, using multi-step reasoning to proactively adjust stock and route shipments.
- **Customer Support Automation:** Enterprises like telecom giants have deployed LLM agents to handle multi-turn, context-aware conversations with customers. These agents not only answer questions but also autonomously escalate or resolve issues, access internal systems, and trigger workflows such as refunds or account adjustments.
- **Healthcare Administration:** LLM agents assist with claims processing and appointment scheduling. They can extract and validate information from unstructured documents, cross-reference clinical guidelines, and flag anomalies for review.
The article emphasizes that these deployments are not just pilots: they are integrated into core business operations and include agentic planning features such as goal decomposition, tool use, and autonomous decision-making within guardrails. Observability, audit trails, and human-in-the-loop provisions are standard to ensure safety and compliance.

-----

### Source: https://addepto.com/blog/llm-use-cases-for-business/
This article highlights a range of real-world LLM use cases across industries, including advanced agentic applications:

- **Legal and Compliance:** Law firms and compliance departments are utilizing LLM agents to autonomously review contracts, flag compliance risks, and generate summaries. These agents plan multistep review processes, referencing statutes and prior cases to support complex reasoning tasks.
- **Retail Personalization:** LLM agents power personalized shopping assistants that analyze individual customer profiles, preferences, and inventory data to make adaptive product recommendations, dynamically plan upsell strategies, and automate fulfillment workflows.
- **Technical Support:** In software and hardware companies, LLM agents diagnose technical issues by reasoning through logs, prior support tickets, and documentation, enabling them to autonomously triage and propose solutions.
The article notes that the most advanced deployments feature agentic planning capabilities: these agents can break down goals, sequence tool use, and adapt their actions based on real-time feedback from users or systems.

-----

### Source: http://generativeaienterprise.ai/p/20-must-read-ai-case-studies-for-enterprise-leaders
A notable case study from Airbnb demonstrates advanced agentic planning with LLM-powered agents:

- **Airbnb Automation Platform:** Airbnb evolved from static, rules-based conversational workflows to a Version 2 platform supporting LLM-powered applications. The new system enables more natural, intelligent, and open-ended conversations, improving user experience and operational efficiency.
- The platform integrates agentic capabilities such as chain-of-thought reasoning, context management, guardrails, and observability.
- For sensitive workflows like claims processing that require strict validation, Airbnb combines traditional approaches with LLM agents, leveraging the strengths of both.
- The evolution showcases how Airbnb uses LLMs in production to automate and enhance complex customer-facing and back-office processes, while maintaining reliability and safety through hybrid architectures.

-----

### Source: https://cloud.google.com/transform/101-real-world-generative-ai-use-cases-from-industry-leaders
This source catalogs generative AI deployments by leading organizations, including advanced LLM agents:

- **Manufacturing:** Companies use LLM agents to monitor equipment data, predict maintenance needs, and autonomously schedule repairs. These agents analyze sensor streams, historical failures, and inventory to plan preventive actions.
- **Insurance:** LLM-powered agents process claims end-to-end—extracting details from documents, verifying coverage, and recommending payouts—using multi-step reasoning and tool integrations.
- **Healthcare:** Hospital systems deploy LLM agents to assist clinicians by summarizing patient histories, suggesting diagnostic tests, and coordinating appointment scheduling, demonstrating agentic planning in regulated environments.
These use cases illustrate how enterprises are integrating LLM agents into mission-critical workflows requiring autonomous reasoning, planning, and safe execution at scale.

-----

</details>

---

<details>
<summary>How do leading open-source frameworks (e.g., LangChain, LlamaIndex, CrewAI) differ in their support for explicit planning/reasoning patterns (like ReAct or Plan-and-Execute) and in enabling modular, orchestrated agent design?</summary>

### Source: https://orq.ai/blog/langchain-vs-crewai
This analysis compares LangChain and CrewAI, emphasizing their core philosophies and suitability for different AI development scenarios.

- **LangChain** is described as a highly flexible, code-driven framework ideal for developers seeking fine-grained control over their AI applications. It excels at modularity, enabling custom prompt engineering, memory management, structured data retrieval, and multi-step LLM interactions. LangChain's design supports building complex, orchestrated agent workflows, making it well-suited for implementing explicit planning and reasoning patterns (such as ReAct or Plan-and-Execute), given its hands-on coding approach and extensive integration capabilities.
  
- **CrewAI** is positioned as a framework that simplifies the development of multi-agent systems with a focus on structured, process-driven teamwork. It uses role-based agent design to coordinate collaborative workflows, assign tasks, and facilitate human-in-the-loop oversight. CrewAI is optimized for hierarchical decision-making and structured task execution, making it a strong choice for orchestrated agent design. However, it is less customizable than LangChain, which may limit its flexibility in supporting fine-tuned, explicit planning/reasoning patterns.

Overall, LangChain offers more granular control for explicit reasoning and modular design, while CrewAI prioritizes simplicity and orchestration for team-based agent collaboration.

-----

### Source: https://blog.stackademic.com/navigating-the-ai-agent-landscape-in-depth-analysis-of-autogen-crewai-llamaindex-and-langchain-2a3bcd932abc
This in-depth guide explores AutoGen, CrewAI, LlamaIndex, and LangChain, providing insights into their agent capabilities:

- **LangChain** is highlighted for its advanced support for building agents with explicit reasoning and planning abilities. It enables developers to implement complex agent behaviors using patterns like ReAct and Plan-and-Execute, thanks to its modularity and support for chaining tools, memory, and prompts.
  
- **CrewAI** is recognized for orchestrating multi-agent collaboration through defined agent roles and workflows. While it excels at team-based agent architectures, its abstraction may limit the direct implementation of intricate planning/reasoning patterns unless extended.
  
- **LlamaIndex** is primarily described as a data framework for connecting LLMs to external or private data sources. It can be integrated as a tool within agent frameworks (such as LangChain or CrewAI) to enhance data retrieval capabilities, but it does not natively offer advanced agent planning or reasoning patterns.

Thus, LangChain leads in explicit planning/reasoning support, CrewAI in multi-agent orchestration with less planning flexibility, and LlamaIndex as a modular data interface rather than a reasoning/planning agent platform.

-----

### Source: https://www.helicone.ai/blog/ai-agent-builders
This comparative overview focuses on top open-source AI agent frameworks, including LangChain. It notes that LangChain is widely adopted for its support of modular, orchestrated agent design, allowing developers to chain multiple components and reasoning steps. The article indicates that LangChain’s architecture is well-suited for implementing explicit planning and reasoning patterns, given its workflow-oriented design and integration flexibility. It does not provide details about LlamaIndex or CrewAI’s support for such patterns or agent modularity.

-----

### Source: https://www.turing.com/resources/ai-agent-frameworks
This source provides a side-by-side analysis of leading AI agent frameworks, including LangChain, LlamaIndex, and CrewAI.

- **LangChain**: Built for a wide variety of LLM applications, LangChain offers extensive features and a large community. Its architecture is highly modular, supporting agent workflows that can incorporate explicit planning and reasoning patterns such as ReAct or Plan-and-Execute. LangChain is recognized for enabling developers to build complex, orchestrated agent systems through chaining and memory modules.
  
- **CrewAI**: Specializes in multi-agent coordination, particularly in hierarchical and structured task execution. CrewAI’s orchestration capabilities make it strong for team-based agent workflows, but its abstraction level provides less fine-grained control for developers aiming to implement custom planning or reasoning patterns.
  
- **LlamaIndex**: Functions mainly as a data interface, excelling at connecting LLMs to diverse data sources. While it can be integrated into agent workflows (e.g., within CrewAI or LangChain), it does not itself enable agent planning or orchestration patterns.

The source also notes that LlamaIndex and CrewAI can be integrated for advanced applications, but the core support for modular explicit planning and reasoning primarily resides in LangChain’s architecture.

</details>

---

<details>
<summary>What are the most significant recent advancements (2023–2024) in long-term or persistent memory systems for LLM-based agents, and how do these advances contribute to more effective multi-step planning and self-correction?</summary>

### Source: https://arxiv.org/html/2504.15965v1
Recent advancements in long-term and persistent memory systems for LLM-based agents have focused on enabling autonomous agents to perform complex, multi-step reasoning and dynamically adapt to user needs. Key developments include:

- **Long-Term Memory Mechanisms:** Systems like MemoryBank allow LLMs to retrieve relevant memories, continuously update their knowledge base, and integrate information from previous interactions, supporting persistent adaptation to users over time. This continuous evolution enhances the agent’s ability to handle multi-step planning by retaining and utilizing accumulated context.
- **Personalization and Context Recall:** Solutions such as OpenAI ChatGPT Memory, Apple Personal Context, mem0, and MemoryScope integrate persistent memories into commercial and open-source AI systems. These mechanisms allow agents to recall detailed interaction history and user preferences, enabling more accurate and context-aware decision-making.
- **Enhanced Task Decomposition:** Memory-enhanced LLM agents can autonomously decompose complex tasks, invoke and execute tools, and remember the outcomes of previous steps, facilitating effective multi-step execution and planning.

Collectively, these advancements overcome the traditional limitations posed by the context window of LLMs, supporting more robust, adaptive, and intelligent agent behavior in dynamic environments.

-----

### Source: https://www.tribe.ai/applied-ai/beyond-the-bubble-how-context-aware-memory-systems-are-changing-the-game-in-2025
The integration of cognitive memory into LLMs has produced several significant effects:

- **Contextual Continuity:** Memory systems now enable LLMs to maintain context across multiple interactions, overcoming the stateless nature of conventional LLMs. This allows for richer, more coherent multi-step planning and persistent task execution.
- **Reduction of Hallucinations:** Techniques like Retrieval-Augmented Generation (RAG) anchor the model’s responses in factual, stored knowledge, reducing the likelihood of hallucinations during multi-step reasoning. Extensions such as Graph-RAG further improve retrieval accuracy and scalability.
- **Automated Data Processing:** LLM pipelines with memory can ingest, classify, store, and retrieve large volumes of data efficiently, supporting accurate recall during multi-step workflows and enabling self-correction by referring back to prior knowledge.
- **Self-Evolution:** Persistent memory supports LLMs in adapting to new tasks and learning from limited data, mimicking human learning. This adaptability is crucial for agents to self-correct and improve through successive interactions, laying the groundwork for future AI developments.

-----

### Source: https://arxiv.org/html/2504.02441v2
Recent persistent memory techniques in LLMs focus on overcoming context window limitations and fostering smarter, more logical interactions:

- **External Memory Modules:** Modern architectures supplement LLMs with external memory modules, enabling the model to query and update context information beyond the fixed input window. This supports persistent recall and continuous learning over long sessions.
- **Retrieval-Augmented Generation (RAG):** By integrating external databases, LLMs can retrieve relevant facts and context as needed, ensuring responses remain consistent and relevant in multi-step planning scenarios.
- **Embedding and Vector Representations:** These allow LLMs to encode and store context efficiently, enabling persistent memory retention across sessions.
- **Transfer Learning and Specialized Protocols:** Advanced protocols help models retain and update memories, enhancing their ability to remember past steps and self-correct during prolonged problem-solving tasks.

These techniques collectively empower LLM-based agents to provide continuity, support logical reasoning across sessions, and facilitate effective multi-step planning and self-correction by maintaining a persistent context.

-----

</details>

---

<details>
<summary>What concrete limitations do even the most advanced frontier LLMs (like GPT-4o or DeepSeek-R1) still face regarding autonomous reasoning, planning, and tool usage in real-world, multi-step tasks?</summary>

### Source: https://arxiv.org/pdf/2501.12948
DeepSeek-R1-Zero, a reasoning-optimized variant of DeepSeek, demonstrates significant improvements in reasoning benchmarks compared to its predecessors. For example, its pass@1 score on the AIME 2024 benchmark increases from 15.6% to 71.0%, and with majority voting, it matches the performance of OpenAI’s o1-0912 model. However, even at this advanced level, DeepSeek-R1-Zero faces notable limitations. The paper highlights challenges such as poor readability in output and language mixing—where multiple languages may inadvertently be combined in responses. Additionally, while the model can self-improve in certain reasoning tasks via reinforcement learning, there is no clear indication that it achieves general reasoning capabilities comparable to the most advanced OpenAI models. The paper also notes that pure reinforcement learning, while effective in some domains, does not fully close the gap in complex, multi-step, real-world reasoning, planning, or tool usage.

-----

### Source: https://www.plainconcepts.com/deepseek-r1/
Testing of DeepSeek R1 reveals several concrete limitations in autonomous reasoning, planning, and tool usage for real-world, multi-step tasks. Most significantly, DeepSeek R1 currently cannot function as an autonomous agent or coordinate with other agents due to its lack of function calling support. This restricts its ability to execute structured, multi-step, or collaborative tasks autonomously. In contrast, models like GPT-4o are capable of such operations. DeepSeek R1 also does not support image analysis, reducing its effectiveness in multimodal tasks that require vision-language integration. Furthermore, DeepSeek R1 lacks robust support for custom system prompts and function calling, making it less flexible for structured automation and real-world application integration compared to leading proprietary models.

-----

### Source: https://www.allganize.ai/en/blog/the-emergence-of-deepseek-r1-and-what-we-must-not-overlook---part-2
As language models like DeepSeek-R1 are scaled, inherent limitations in autonomous reasoning and task execution become apparent. DeepSeek-R1 emphasizes autonomous reasoning and multi-step task execution but has not yet resolved fundamental constraints that affect real-world deployment. The model's current priority is to improve these aspects, suggesting that, while progress has been made, there are still challenges in achieving reliable, autonomous reasoning and robust planning capabilities for complex, multi-step, real-world tasks.

-----

### Source: https://community.openai.com/t/is-deepseek-a-distilled-version-of-gpt-4-analyzing-suspicious-behavior/1109600
Although DeepSeek-R1 initially exhibited behaviors similar to GPT-4, subsequent analysis notes that it diverges in its ability to sustain complex, multi-step autonomous reasoning and planning. The analysis implies that while DeepSeek-R1 can perform well on many benchmarks, its performance in real-world, autonomous, multi-step tasks still lags behind the most advanced proprietary models like GPT-4o, especially in situations requiring sustained, flexible tool usage and adaptation to novel scenarios.

-----

### Source: https://metr.github.io/autonomy-evals-guide/deepseek-r1-report/
A formal evaluation of DeepSeek-R1’s autonomous capabilities found that it performs only marginally better than earlier versions and does not show a significant leap in autonomous reasoning or planning. In METR’s general autonomy tasks, DeepSeek-R1 performed slightly better than o1-preview and is comparable to the level of frontier models from September 2024. On research and development tasks (RE-Bench), it performed comparably to GPT-4o but worse than other newer frontier models. This suggests that, while DeepSeek-R1 and similar open-weight models are cost-effective, their real-world autonomous reasoning, planning, and tool usage for complex, multi-step tasks remain limited and are not significantly ahead of previous state-of-the-art models.

</details>

---

<details>
<summary>Which recent (2023–2024) industry-recognized benchmarks most effectively measure an LLM agent’s explicit planning, goal decomposition, and self-correction ability, and what are their core evaluation protocols or tasks?</summary>

### Source: https://www.evidentlyai.com/llm-guide/llm-benchmarks
Evidently AI provides a comprehensive overview of more than 100 LLM benchmarks, with a curated list of 20 key industry-recognized benchmarks and their protocols. Among these, several recent (2023–2024) benchmarks specifically focus on agentic abilities such as explicit planning, goal decomposition, and self-correction:

- **AgentBench (2023):** Evaluates LLM agents in simulated environments requiring web browsing, tool use, and multi-step reasoning. Tasks are designed to test planning, goal decomposition, and execution by requiring agents to complete complex objectives using external tools or APIs.
- **ToolBench:** Focuses on an LLM’s ability to use external tools effectively, which inherently involves planning and decision-making about which tools and actions to select for sub-tasks.
- **AlpacaEval and MT-Bench:** While primarily used for instruction-following and dialogue, these benchmarks include multi-turn and multi-objective tasks that implicitly require planning and corrective actions based on ongoing context.
- **BBH (Big-Bench Hard):** Contains tasks related to multi-step reasoning, logic, and planning, often requiring the decomposition of goals and the ability to revise strategies if initial attempts fail.

Core evaluation protocols often include:
- Presenting LLM agents with objectives that require stepwise execution.
- Measuring success based on completion of subgoals or accuracy in a multi-stage process.
- Scoring based on exact match, partial credit for subgoals, or LLM-assisted grading for subjective responses.

These benchmarks reflect a growing industry emphasis on evaluating not just static accuracy, but also dynamic planning and adaptive behaviors in LLM agents.

-----

### Source: https://www.vellum.ai/blog/llm-benchmarks-overview-limits-and-model-comparison
Vellum AI outlines several of the most commonly used benchmarks for LLM evaluation, highlighting their focus areas. While many benchmarks focus on reasoning, coding, math, and tool use, specific attention is given to benchmarks that involve multi-step reasoning and tool use:

- **BFCL (Benchmark for Function Calling/Tool Use):** Measures the ability of LLMs to call external functions or tools, which inherently requires planning and breaking down user goals into actionable tool calls.
- **MMLU (Massive Multitask Language Understanding):** Tests multitask accuracy but is less focused on explicit planning or self-correction.
- **HumanEval:** Focuses on code generation, which can require goal decomposition and iterative improvement, especially on complex programming tasks.

The evaluation protocol for BFCL, for example, typically involves presenting the LLM with a scenario requiring the use of one or more tools to achieve a goal, with grading based on the correctness and efficiency of tool use, reflecting planning and execution abilities.

-----

### Source: https://www.confident-ai.com/blog/llm-benchmarks-mmlu-hellaswag-and-beyond
Confident AI explains the structure and intent behind LLM benchmarks, emphasizing that modern benchmarks increasingly target complex skills beyond simple QA, including:

- **Reasoning and Commonsense:** Some benchmarks now test for logical reasoning and multi-step problem-solving, which involve goal decomposition and intermediate planning.
- **Logic Benchmarks:** Directly target logical reasoning and the ability to break down problems into manageable parts.
- **Conversation and Chatbots:** Multi-turn dialogue tests sometimes require self-correction and adaptation during an ongoing interaction.

Protocols in these benchmarks often involve:
- Presenting models with multi-part problems that require intermediate reasoning steps.
- Scoring based on the accuracy of final and intermediate answers.
- Use of automated and sometimes LLM-assisted grading to evaluate reasoning chains and corrections.

-----

### Source: https://github.com/zhangxjohn/LLM-Agent-Benchmark-List
This GitHub repository serves as a curated collection of benchmarks specifically for LLM agents, with a focus on evaluating agentic behaviors such as planning, goal decomposition, and self-correction. Notable entries include:

- **AgentBench:** Detailed evaluation tasks require LLM agents to perform planning, execute sequential steps, and adapt to unexpected outcomes, with metrics tracking success rate, subgoal completion, and recovery from errors.
- **ToolBench:** Centers on tool-augmented LLMs, measuring both the ability to plan tool use and to self-correct after failures.
- **WebArena:** Assesses LLMs’ ability to interact with simulated web environments, necessitating explicit planning, multi-step goal pursuit, and error recovery.

Each benchmark outlines specific task protocols, such as:
- Multi-step web navigation or API invocation.
- Success measured by both final task completion and the correctness of intermediate actions.
- Evaluation by automated scripts or human/LLM graders to assess the quality of the process, not just the end result.

</details>

---

## Sources Scraped From Research Results

---
<details>
<summary>LLM Agent Orchestration: A Step by Step Guide | IBM </summary>

# LLM agent orchestration: step by step guide with LangChain and Granite

LLM agent orchestration refers to the process of managing and coordinating the interactions between a large language model (LLM) @https://www.ibm.com/think/topics/large-language-models and various tools, APIs or processes to perform complex tasks within AI systems. It involves structuring workflows where an AI agent, @https://www.ibm.com/think/topics/ai-agents powered by artificial intelligence @https://www.ibm.com/think/topics/artificial-intelligence, acts as the central decision-maker or reasoning engine, orchestrating its actions based on inputs, context and outputs from external systems. Using an orchestration framework, LLMs can seamlessly integrate with APIs, databases and other AI applications, enabling functionalities such as chatbots and automation tools. Open-source agent frameworks further enhance the adaptability of these systems, making LLMs more effective in real-world scenarios.

Many people misunderstand the difference between LLM orchestration @https://www.ibm.com/think/topics/llm-orchestration and LLM agent orchestration. The following illustration highlights the key differences:

In this tutorial, you will learn how to build an autonomous agent powered by large language models (LLMs) by using IBM® Granite™ models @https://www.ibm.com/products/watsonx-ai/foundation-models and LangChain. @https://www.ibm.com/think/topics/langchain We’ll explore how agents leverage key components such as memory, planning and action to perform intelligent tasks. You’ll also implement a practical system that processes text from a book, answers queries dynamically and evaluates its performance by using accuracy metrics such as BLEU, precision, recall and F1 score.

## Framework for LLM-based autonomous agents

The framework presented in figure-1 provides a holistic design for large language model (LLM)-based autonomous agents @https://www.ibm.com/think/topics/automation, emphasizing the interplay between key components: profile, memory, planning and action. Each component represents a critical stage in building an autonomous agent capable of reasoning, decision-making and interacting with dynamic environments.1 @https://www.ibm.com/think/tutorials/llm-agent-orchestration-with-langchain-and-granite#f01

**1\. Profile: defining the agent’s identity**

The profile gives the agent its identity by embedding information such as demographics, personality traits and social context. This process ensures that the agent can interact in a personalized way. Profiles can be manually crafted, generated by gen AI models such as IBM Granite models @https://www.ibm.com/products/watsonx-ai/foundation-models or OpenAI’s GPT (generative pretrained transformer), or aligned with specific datasets to meet task requirements. Leveraging prompt engineering @https://www.ibm.com/think/topics/prompt-engineering, profiles can be dynamically refined to optimize responses. Additionally, within multiagent @https://www.ibm.com/think/topics/multiagent-system orchestration, the profile helps define roles and behaviors, ensuring seamless coordination across AI algorithms and decision-making systems.

**2\. Memory: storing and using context**

Memory helps the agent retain and retrieve past interactions, enabling contextual responses. It can be unified (all data in one place) or hybrid (structured and unstructured). Operations including reading, writing and reflection allow the agent to learn from experience and provide consistent, informed outputs. Well-structured memory enhances multiagent orchestration by ensuring that different agents, including specialized agents designed for a specific task, can share and retrieve relevant data efficiently. In frameworks such as AutoGen and Crew AI, @https://www.ibm.com/think/topics/crew-ai memory plays a crucial role in maintaining continuity within the ecosystem of collaborating agents, ensuring seamless coordination and optimized task execution.

**3\. Planning: strategizing actions**

The planning component lets the agent devise strategies to achieve goals. It can follow predefined steps or adapt dynamically based on feedback from the environment, humans or the LLM itself. By integrating AI algorithms and leveraging a knowledge base, planning can be optimized to improve reasoning efficiency and problem-solving accuracy. In LLM applications, planning plays a crucial role in ensuring natural language understanding and decision-making processes align with the agent's objectives. Additionally, retrieval-augmented techniques enhance the agent's ability to access relevant information dynamically, improving response accuracy. This flexibility ensures that the agent remains effective in changing scenarios, especially in multiagent orchestration, where various agents coordinate their plans to achieve complex objectives while maintaining scalability for handling large and diverse tasks.

**4\. Action: executing decisions**

Actions are the agent’s way of interacting with the world, whether by completing tasks, gathering information or communicating. It uses memory and planning to guide execution, employs tools when needed and adapts its internal state based on results for continuous improvement. Optimizing the action execution algorithm ensures efficiency, especially when integrating GPT @https://www.ibm.com/think/topics/gpt-powered reasoning models and gen AI @https://www.ibm.com/think/topics/generative-ai techniques for real-time decision-making.

By combining these components, the framework transforms LLMs into adaptable agents capable of reasoning, learning and performing tasks autonomously. This modular design makes it ideal for applications such as customer service, research assistance and creative problem-solving.

## Use case: Building a queryable knowledge agent

This tutorial demonstrates the creation of a queryable knowledge agent designed to process large text documents (like books) and answer user queries accurately. Using IBM Granite models @https://www.ibm.com/products/watsonx-ai/foundation-models and LangChain, the agent is built following the principles outlined in the framework for LLM-based autonomous agents. The framework's components align seamlessly with the agent's workflow to ensure adaptability and intelligent responses.

**Let's understand how the framework applies in our use case.**

**Profile:** The agent is designed with a "knowledge assistant" profile, focusing on summarization, question answering and reasoning tasks. Its context is personalized to process a specific document (for example, The Adventures of Sherlock Holmes).

**Memory:** The agent employs hybrid memory by embedding chunks of the book into a FAISS vector store. This ability allows it to retrieve relevant context dynamically during queries. Memory operations such as reading (retrieval) and writing (updating embeddings) ensure that the agent can adapt to new queries over time.

**Planning:** Query resolution involves single-path reasoning. The agent retrieves relevant chunks of text, generates answers by using IBM’s Granite LLM and evaluates the output for accuracy. Planning without feedback ensures simplicity, while the system’s modularity allows feedback loops to be incorporated in future iterations.

**Action:** The agent executes query resolution by integrating memory retrieval and LLM processing. It completes tasks such as generating answers, calculating accuracy metrics (BLEU, precision, recall and F1 score) and visualizing results for user interpretation. These outputs reflect the agent’s capability to act intelligently based on reasoning and planning.

## Prerequisites

You need an IBM Cloud® account @https://cloud.ibm.com/registration?utm_source=ibm_developer&utm_content=in_content_link&utm_id=tutorials_awb-implement-xgboost-in-python&cm_sp=ibmdev-_-developer-_-trial to create a watsonx.ai® @https://www.ibm.com/products/watsonx-ai?utm_source=ibm_developer&utm_content=in_content_link&utm_id=tutorials_awb-implement-xgboost-in-python&cm_sp=ibmdev-_-developer-_-product project.

## Steps

### Step 1. Set up your environment

While you can choose from several tools, this tutorial walks you through how to set up an IBM account by using a Jupyter Notebook.

1. Log in to watsonx.ai @https://dataplatform.cloud.ibm.com/registration/stepone?context=wx&apps=all by using your IBM Cloud account.
2. 2\. Create a watsonx.ai project @https://www.ibm.com/docs/en/watsonx/saas?topic=projects-creating-project. You can get your project ID from within your project. Click the **Manage** tab. Then, copy the project ID from the **Details** section of the **General** page. You need this ID for this tutorial.
3. 3\. Create a Jupyter Notebook @https://www.ibm.com/docs/en/watsonx/saas?topic=editor-creating-managing-notebooks.

This step opens a notebook environment where you can copy the code from this tutorial. Alternatively, you can download this notebook to your local system and upload it to your watsonx.ai project as an asset. To view more Granite tutorials, check out the IBM Granite Community @https://github.com/ibm-granite-community. This tutorial is also available on GitHub @https://github.com/IBM/ibmdotcom-tutorials.

### Step 2. Set up watsonx.ai runtime service and API key

1. Create a watsonx.ai Runtime @https://cloud.ibm.com/catalog/services/watsonxai-runtime service instance (choose the Lite plan, which is a free instance).
2. Generate an application programming interface (API) Key @https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/ml-authentication.html.
3. Associate the watsonx.ai Runtime service to the project that you created in watsonx.ai @https://dataplatform.cloud.ibm.com/docs/content/wsj/getting-started/assoc-services.html?context=cpdaas.

### Step 3. Installation of the packages

To work with the LangChain framework and integrate IBM WatsonxLLM, we need to install some essential libraries. Let’s start by installing the required packages:

_Note: If you are using old version ofpip_
_, you can use the commandpip install --upgrade pip_
_to upgrade for an easy installation of the latest packages, which might not be compatible with older versions. But if you are already using the latest version or recently upgraded you packages, then you can skip this command._

!pip install --upgrade pip

!pip install langchain faiss-cpu pandas sentence-transformers

%pip install langchain

!pip install langchain-ibm

In the preceding code cell,

- langchain
is the core framework for building applications with language models.
- faiss-cpu
is for efficient similarity search, used in creating and querying vector indexes.
- pandas
is for data manipulation and analysis.
- sentence-transformers
is to generate embeddings for semantic search.
- langchain-ibm
is to integrate IBM WatsonxLLM (in this tutorial it's granite-3-8b-instruct) with LangChain.

This step ensures that your environment is ready for the tasks ahead.

### Step 4. Import required libraries

Now that we’ve installed the necessary libraries, let’s import the modules required for this tutorial:

import os

from langchain\_ibm import WatsonxLLM

from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import FAISS

from langchain.text\_splitter import RecursiveCharacterTextSplitter

import pandas as pd

import getpass

In the preceding code cell,

- os
provides a way to interact with the operating system (for example, accessing environment variables).
- langchain\_ibm.WatsonxLLM
allows us to use the IBM Watson® Granite LLM seamlessly within the LangChain framework.
- langchain.embeddings.HuggingFaceEmbeddings
generates embeddings for text by using HuggingFace models, essential for semantic search.
- langchain.vectorstores.FAISS
is a library for efficient vector storage and similarity search, enabling us to build and query a vector index.
- RecursiveCharacterTextSplitter
helps split large blocks of text into smaller chunks, which is critical for processing documents efficiently.
- pandas
is a powerful library for data analysis and manipulation, used here to handle tabular data.
- getpass
is a secure way to capture sensitive input such as API keys without displaying them on the screen.

This step sets up all the tools and modules that we need to process text, create embeddings, store them in a vector database and interact with IBM's WatsonxLLM.

### Step 5. Set up credentials

This code sets up credentials for accessing the IBM Watson machine learning (WML) API and ensures that the project ID is correctly configured.

- A dictionarycredentials
is created with theWML service URL
andAPI key
. The API key is securely collected by using \`getpass.getpass\` to avoid exposing sensitive information.
- The code tries to fetch the PROJECT\_ID
from environment variables by using os.environ
. If the PROJECT\_ID
is not found, the user is prompted to manually enter it through input.

\# Set up credentials

credentials = {

      "url": "https://us-south.ml.cloud.ibm.com", # Replace with the correct region if needed

      "apikey": getpass.getpass("Please enter your WML API key (hit enter): ")

     }

\# Set up project\_id

try:

     project\_id = os.environ\["PROJECT\_ID"\]

except KeyError:

     project\_id = input("Please enter your project\_id (hit enter): ")

### Step 6. Initialize large language model

This code initializes the IBM WatsonxLLM for use in the application:

1. This code creates an instance of WatsonxLLM
by using theibm/granite-3-8b-instruct
model, designed for instruction-based generative AI tasks.
2. Theurl
,apikey
    andproject\_id
values from the previously set up credentials are passed to authenticate and connect to the IBM WatsonxLLM service.
3. Configures themax\_new\_tokens
parameter to limit the number of tokens generated by the model in each response (150 tokens in this case).

This step prepares the WatsonxLLM for generating responses in the workflow.

\# Initialize the IBM Granite LLM

llm = WatsonxLLM(

      model\_id="ibm/granite-3-8b-instruct",

      url=credentials\["url"\],

      apikey=credentials\["apikey"\],

      project\_id=project\_id,

      params={

           "max\_new\_tokens": 150

      }

)

### Step 7. Define a function to extract text from a file

To process the text from a document, we need a function that can read and extract its contents. The following function is designed to handle plain text files:

def extract\_text\_from\_txt(file\_path):

      """Extracts text from a plain text file."""

           with open(file\_path, "r", encoding="utf-8") as file:

           text = file.read()

return text

This function,extract\_text\_from\_txt
, is designed to read and extract the content of a plain text file. It accepts the file path as an argument and opens the file in read mode withUTF-8 encoding
, ensuring that special characters are handled correctly.

The entire content of the file is read into a variable namedtext
, which is then returned. This function plays a crucial role in preparing the input data by extracting raw text from the document, making it ready for subsequent operations such as chunking, embedding and querying. It provides a simple and efficient way to process textual data from any plain text file.

This function allows us to process the input file _(The Adventures of Sherlock Holmes)_ and extract its content for further operations such as text chunking and embedding. It ensures that the raw text is readily available for analysis.

### Step 8. Split text into chunks

To efficiently process and index large blocks of text, we need to divide the text into smaller, manageable chunks. The following function handles this task:

def split\_text\_into\_chunks(text, chunk\_size=500, chunk\_overlap=50):

           """Splits text into smaller chunks for indexing."""

           splitter = RecursiveCharacterTextSplitter(chunk\_size=chunk\_size, chunk\_overlap=chunk\_overlap)

return splitter.split\_text(text)

Thesplit\_text\_into\_chunks
function is designed to divide large blocks of text into smaller, manageable chunks for efficient processing and indexing. It takes the raw text as input along with two optional parameters:chunk\_size
, which defines the maximum size of each chunk _(default is 500 characters)_, andchunk\_overlap
, which specifies the number of overlapping characters between consecutive chunks _(default is 50)_.

This function ensures contextual continuity across the chunks. The function utilizes theRecursiveCharacterTextSplitter
fromLangChain
, which intelligently splits text while preserving its context. By returning a list of smaller text chunks, this function prepares the input for further operations such as embedding and indexing.

It is essential when working with large documents, as language models often have token limitations and cannot process lengthy text directly.

### Step 9: Create a vector index

To enable efficient semantic search, we need to convert text chunks into vector embeddings and store them in a searchable index. This step uses FAISS and HuggingFace embeddings to create the vector index, forming the foundation for retrieving relevant information based on queries.

def create\_vector\_index(chunks):

           """Creates a FAISS vector index from text chunks."""

               embeddings = HuggingFaceEmbeddings(model\_name="sentence-transformers/all-MiniLM-L6-v2")

               vector\_store = FAISS.from\_texts(chunks, embeddings)

return vector\_store

Thecreate\_vector\_index
function builds aFAISS vector
index from the text chunks generated in the previous step. This function is crucial for enabling semantic search by mapping each chunk into a high-dimensional vector space by using embeddings.

It first initializes a _HuggingFaceEmbeddings model_ sentence-transformers/all-MiniLM-L6-v2
, which generates vector embeddings for the text chunks. These embeddings capture the semantic meaning of each chunk.

The function then usesFAISS
to create a vector store by indexing these embeddings, allowing for efficient similarity search later.

The resulting vector store is returned and will be used to find relevant chunks based on user queries, forming the backbone of the agent's search and retrieval process.

### Step 10. Query the vector index with Granite

This step involves querying the vector index to retrieve relevant information and by using IBM's Granite LLM to generate a refined response. By integrating similarity search and LLM reasoning, the function provides a dynamic and intelligent query resolution process.

def query\_index\_with\_granite\_dynamic(vector\_store, query, llm):

         """Searches the vector index, uses Granite to refine the response, and returns all components."""

             # Perform similarity search

             print("\\n> Entering new AgentExecutor chain...")

             thought = f"The query '{query}' requires context from the book to provide an accurate response."

             print(f" Thought: {thought}")

             action = "Search FAISS Vector Store"

             print(f" Action: {action}")

             action\_input = query

             print(f" Action Input: \\"{action\_input}\\"")

             # Retrieve context

             results = vector\_store.similarity\_search(query, k=3)

             observation = "\\n".join(\[result.page\_content for result in results\])

             print(f" Observation:\\n{observation}\\n")

            # Generate response with Granite

            prompt = f"Context:\\n{observation}\\n\\nQuestion: {query}\\nAnswer:"

            print(f" Thought: Combining retrieved context with the query to generate a detailed answer.")

            final\_answer = llm(prompt)

            print(f" Final Answer: {final\_answer.strip()}")

            print("\\n> Finished chain.")

            # Return all components as a dictionary

            return {

                    "Thought": thought,

                     "Action": action,

                     "Action Input": action\_input,

                     "Observation": observation,

                     "Final Answer": final\_answer.strip()

                     }

Thequery\_index\_with\_granite\_dynamic
function takes three inputs: first—the vector store (vector\_store
), second—the user's query (query
) and third—the Granite LLM instance (llm
).

It first performs a similarity search on the vector index to retrieve the most relevant chunks of text. These chunks, referred to asobservation
, are combined into a single context block.

The function then constructs a prompt by combining the query and the retrieved context. This prompt is passed to theGranite LLM
, which generates a detailed and contextually accurate response (final\_answer
).

Throughout the process, intermediate steps like the agent'sthought
,action
andaction input
are printed for transparency.

Finally, the function returns a dictionary containing all components, including the thought process, action taken, retrieved observation and the final answer.

This step is critical for transforming raw data retrieval into actionable and meaningful insights by using the LLM's reasoning capabilities.

### Step 11. Generate a 'DataFrame' for query results

This step dynamically processes multiple queries, retrieves relevant information and saves the results in a structured format for analysis. The function integrates querying, data structuring and export capabilities.

def dynamic\_output\_to\_dataframe(vector\_store, queries, llm, csv\_filename="output.csv"):

           """Generates a DataFrame dynamically for multiple queries and saves it as a CSV file."""

           # List to store all query outputs

           output\_data = \[\]

           # Process each query

           for query in queries:

           # Capture the output dynamically

           output = query\_index\_with\_granite\_dynamic(vector\_store, query, llm)

           output\_data.append(output)

           # Convert the list of dictionaries into a DataFrame

           df = pd.DataFrame(output\_data)

           # Display the DataFrame

           print("\\nFinal DataFrame:")

           print(df)

           # Save the DataFrame as a CSV file

           df.to\_csv(csv\_filename, index=False)

           print(f"\\nOutput saved to {csv\_filename}")

Thedynamic\_output\_to\_dataframe
function accepts four inputs: the vector store (vector\_store
), a list of queries (queries
), the Granite LLM instance (llm
), and an optional CSV file name (csv\_filename
, default isoutput.csv
).

For each query, it uses thequery\_index\_with\_granite\_dynamic
function to retrieve relevant context and generate a response by using the LLM. The results, including intermediate components such asThought
,Observation
andFinal Answer
are stored in a list.

Once all queries are processed, the list of results is converted into a pandas DataFrame. This tabular format allows easy analysis and visualization of the query results. The DataFrame is printed for review and saved as a CSV file for future use.

This step is essential for organizing the output in a user-friendly format, enabling downstream tasks such as accuracy evaluation and visualization.

### Step 12: Execute the main workflow

This step combines all the previous steps into a single workflow to process a text file, answer user queries and save the results in a structured format. Themain\_workflow function
serves as the central orchestrator of the tutorial.

def main\_workflow():

           # Replace with your text file

           file\_path = "aosh.txt"

           # Extract text from the text file

           text = extract\_text\_from\_txt(file\_path)

           # Split the text into chunks

           chunks = split\_text\_into\_chunks(text)

           # Create a vector index

           vector\_store = create\_vector\_index(chunks)

           # Define queries

           queries = \[\
\
                     "What is the plot of 'A Scandal in Bohemia'?",\
\
                     "Who is Dr. Watson, and what role does he play in the stories?",\
\
                     "Describe the relationship between Sherlock Holmes and Irene Adler.",\
\
                     "What methods does Sherlock Holmes use to solve cases?"\
\
                     \]

           # Generate and save output dynamically

          dynamic\_output\_to\_dataframe(vector\_store, queries, llm)

Let's understand how this workflow executes:

**Input a text file:** Thefile\_path
variable specifies the text file to be processed. In this tutorial, the input file is"aosh.txt"
, containing the text of The Adventures of Sherlock Holmes.

**Text extraction:** The extract\_text\_from\_txt
function is called to read and extract the content of the input text file.

**Text chunking**: The extracted text is divided into smaller chunks by using thesplit\_text\_into\_chunks
function to facilitate embedding and indexing.

**Create a vector index:** The text chunks are converted into embeddings and stored in aFAISS vector
index by using thecreate\_vector\_index
function.

**Define queries:** A list of sample queries is provided, each designed to retrieve specific information from the text. These queries will be answered by the agent.

**Process queries:** Thedynamic\_output\_to\_dataframe
function processes the queries by using the vector index and IBM’s Granite LLM. It retrieves relevant context, generates answers and saves the results as a CSV file for further analysis.

This step integrates all components of the tutorial into a cohesive workflow. It automates the process from text extraction to query resolution, allowing you to test the agent's capabilities and examine the results in a structured format.

To execute the workflow, simply call themain\_workflow()
function and the entire pipeline will run seamlessly.

\# Run the workflow

main\_workflow()

* * *

Output

* * *

```
> Entering new AgentExecutor chain...
 Thought: The query 'What is the plot of 'A Scandal in Bohemia'?' requires context from the book to provide an accurate response.
 Action: Search FAISS Vector Store
 Action Input: "What is the plot of 'A Scandal in Bohemia'?"
 Observation:
I. A SCANDAL IN BOHEMIA

I.
“I was aware of it,” said Holmes dryly.

“The circumstances are of great delicacy, and every precaution has to
be taken to quench what might grow to be an immense scandal and
seriously compromise one of the reigning families of Europe. To speak
plainly, the matter implicates the great House of Ormstein, hereditary
kings of Bohemia.”

“I was also aware of that,” murmured Holmes, settling himself down in
his armchair and closing his eyes.
Contents

   I.     A Scandal in Bohemia
   II.    The Red-Headed League
   III.   A Case of Identity
   IV.    The Boscombe Valley Mystery
   V.     The Five Orange Pips
   VI.    The Man with the Twisted Lip
   VII.   The Adventure of the Blue Carbuncle
   VIII.  The Adventure of the Speckled Band
   IX.    The Adventure of the Engineer’s Thumb
   X.     The Adventure of the Noble Bachelor
   XI.    The Adventure of the Beryl Coronet
   XII.   The Adventure of the Copper Beeches

 Thought: Combining retrieved context with the query to generate a detailed answer.
/var/folders/4w/smh16qdx6l98q0534hr9v52r0000gn/T/ipykernel_2648/234523588.py:23: LangChainDeprecationWarning: The method `BaseLLM.__call__` was deprecated in langchain-core 0.1.7 and will be removed in 1.0. Use :meth:`~invoke` instead.
  final_answer = llm(prompt)
 Final Answer: Step 1: Identify the main characters and their roles.
- Sherlock Holmes: The detective who is approached by a client with a delicate matter.
- An unnamed client: A representative of the great House of Ormstein, hereditary kings of Bohemia, who seeks Holmes' help to prevent a potential scandal.

Step 2: Understand the main issue or conflict.
- The main issue is a delicate matter that, if exposed, could lead to a massive scandal and compromise one of the reigning families of Europe, specifically the House of Ormstein.

Step 3: Ident

> Finished chain.

> Entering new AgentExecutor chain...
 Thought: The query 'Who is Dr. Watson, and what role does he play in the stories?' requires context from the book to provide an accurate response.
 Action: Search FAISS Vector Store
 Action Input: "Who is Dr. Watson, and what role does he play in the stories?"
 Observation:
“Sarasate plays at the St. James’s Hall this afternoon,” he remarked.
“What do you think, Watson? Could your patients spare you for a few
hours?”

“I have nothing to do to-day. My practice is never very absorbing.”
“Try the settee,” said Holmes, relapsing into his armchair and putting
his fingertips together, as was his custom when in judicial moods. “I
know, my dear Watson, that you share my love of all that is bizarre and
outside the conventions and humdrum routine of everyday life. You have
shown your relish for it by the enthusiasm which has prompted you to
chronicle, and, if you will excuse my saying so, somewhat to embellish
so many of my own little adventures.”
“My God! It’s Watson,” said he. He was in a pitiable state of reaction,
with every nerve in a twitter. “I say, Watson, what o’clock is it?”

“Nearly eleven.”

“Of what day?”

“Of Friday, June 19th.”

“Good heavens! I thought it was Wednesday. It is Wednesday. What d’you
want to frighten a chap for?” He sank his face onto his arms and began
to sob in a high treble key.

“I tell you that it is Friday, man. Your wife has been waiting this two
days for you. You should be ashamed of yourself!”

 Thought: Combining retrieved context with the query to generate a detailed answer.
 Final Answer: Dr. Watson is a character in the Sherlock Holmes stories, written by Sir Arthur Conan Doyle. He is a former military surgeon who becomes the narrator and chronicler of Holmes' adventures. Watson is a close friend and confidant of Holmes, often accompanying him on cases and providing a more human perspective to the stories. He is known for his enthusiasm for the bizarre and unconventional, as well as his skill in recording the details of their investigations. Watson's role is crucial in presenting the narrative and offering insights into Holmes' character and methods.

> Finished chain.

Final DataFrame:
                                             Thought  \
0  The query 'What is the plot of 'A Scandal in B...
1  The query 'Who is Dr. Watson, and what role do...
2  The query 'Describe the relationship between S...
3  The query 'What methods does Sherlock Holmes u...

                      Action  \
0  Search FAISS Vector Store
1  Search FAISS Vector Store
2  Search FAISS Vector Store
3  Search FAISS Vector Store

                                        Action Input  \
0        What is the plot of 'A Scandal in Bohemia'?
1  Who is Dr. Watson, and what role does he play ...
2  Describe the relationship between Sherlock Hol...
3  What methods does Sherlock Holmes use to solve...

                                         Observation  \
0  I. A SCANDAL IN BOHEMIA\n\n\nI.\n“I was aware ...
1  “Sarasate plays at the St. James’s Hall this a...
2  “You have really got it!” he cried, grasping S...
3  to learn of the case was told me by Sherlock H...

                                        Final Answer
0  Step 1: Identify the main characters and their...
1  Dr. Watson is a character in the Sherlock Holm...
2  Sherlock Holmes and Irene Adler have a profess...
3  Sherlock Holmes uses a variety of methods to s...

Output saved to output.csv

```

After running themain\_workflow()
function, we processed a text file (aosh.txt) and executed four user-defined queries about The Adventures of Sherlock Holmes. The output provides a detailed breakdown of how each query was handled:

- **Thought** describes the reasoning behind the query and the context that it requires for accurate answering.
- **Action** indicates the step taken, which in this case is performing a similarity search by using the FAISS vector index.
- **Action input** is the specific query being processed in one iteration.
- **Observation** is the text chunks retrieved from the vector index that are relevant to the query.
- **Final answer** is the detailed response generated by IBM's Granite LLM by using the retrieved context.

Additionally, the results for all queries have been structured into a DataFrame and saved asoutput.csv
. This file contains all the above components for further analysis or sharing.

In this process, we combined text retrieval with LLM reasoning to answer complex queries about the book. The agent dynamically retrieved relevant information, used the context to generate precise answers and organized the output in a structured format for easy analysis.

## Visualizing the results

With the output.csv file created, we will now proceed to visualize the query results and their associated accuracy metrics, providing deeper insights into the agent's performance.

In the following code cell, we load the saved query results from theoutput.csv
file into a pandas DataFrame to prepare for visualization and analysis. The DataFrame allows us to manipulate and explore the data in a structured format.

\# Load the output.csv file into a DataFrame

df = pd.read\_csv("output.csv")

print(df.head()) # Display the first few rows

* * *

OUTPUT

* * *

```
Thought  \
0  The query 'What is the plot of 'A Scandal in B...
1  The query 'Who is Dr. Watson, and what role do...
2  The query 'Describe the relationship between S...
3  The query 'What methods does Sherlock Holmes u...

                      Action  \
0  Search FAISS Vector Store
1  Search FAISS Vector Store
2  Search FAISS Vector Store
3  Search FAISS Vector Store

                                        Action Input  \
0        What is the plot of 'A Scandal in Bohemia'?
1  Who is Dr. Watson, and what role does he play ...
2  Describe the relationship between Sherlock Hol...
3  What methods does Sherlock Holmes use to solve...

                                         Observation  \
0  I. A SCANDAL IN BOHEMIA\n\n\nI.\n“I was aware ...
1  “Sarasate plays at the St. James’s Hall this a...
2  “You have really got it!” he cried, grasping S...
3  to learn of the case was told me by Sherlock H...

                                        Final Answer
0  Step 1: Identify the main characters and their...
1  Dr. Watson is a character in the Sherlock Holm...
2  Sherlock Holmes and Irene Adler have a profess...
3  Sherlock Holmes uses a variety of methods to s...

```

In this code, the DataFrame includes key components such asThought
,Action
,Observation
andFinal Answer
for each query. By displaying the first few rows by usingdf.head()
, we ensure that the data is correctly formatted and ready for the next stage: creating meaningful visualizations.

### Import visualization libraries

To create visualizations of the query results, we import the necessary libraries:

import matplotlib.pyplot as plt

from wordcloud import WordCloud

matplotlib.pyplot
is a widely used library for creating static, interactive and animated visualizations in Python. It will be used to generate bar charts, pie charts and other visualizations.

wordcloud
is a library for creating word clouds, which visually highlight the most frequent words in the data. This step helps in summarizing and exploring the context retrieved from the text.

_Important note: If you encounter the error"WordCloud not found"_
_, you can resolve it by installing the library by using the commandpip install wordcloud_
_._

### Visualize observation and answer lengths

This code creates a horizontal bar chart to compare the lengths of **observations (retrieved context)** and **answers (generated responses)** for each query. This visualization provides insight into how much context the agent uses compared to the length of the generated answers.

def visualize\_lengths\_with\_queries(df):

"""Visualizes the lengths of observations and answers with queries on the y-axis."""

df\["Observation Length"\] = df\["Observation"\].apply(len)

df\["Answer Length"\] = df\["Final Answer"\].apply(len)

# Extract relevant data

queries = df\["Action Input"\]

observation\_lengths = df\["Observation Length"\]

answer\_lengths = df\["Answer Length"\]

# Create a horizontal bar chart

plt.figure(figsize=(10, 6))

bar\_width = 0.4

y\_pos = range(len(queries))

plt.barh(y\_pos, observation\_lengths, bar\_width, label="Observation Length", color="skyblue", edgecolor="black")

plt.barh(\[y + bar\_width for y in y\_pos\], answer\_lengths, bar\_width, label="Answer Length", color="lightgreen", edgecolor="black")

plt.yticks(\[y + bar\_width / 2 for y in y\_pos\], queries, fontsize=10)

plt.xlabel("Length (characters)", fontsize=14)

plt.ylabel("Queries", fontsize=14)

plt.title("Observation and Answer Lengths by Query", fontsize=16)

plt.legend(fontsize=12)

plt.tight\_layout()

plt.show()\# Call the visualization function

visualize\_lengths\_with\_queries(df)

This function,visualize\_lengths\_with\_queries
, creates a horizontal bar chart to compare the lengths of **observations (retrieved context)** and **answers (generated responses)** for each query.

It calculates the character lengths of both observations and answers, adding them as new columns (Observation Length
andAnswer Length
) to the DataFrame. Usingmatplotlib
, it then plots these lengths for each query, with queries displayed on the y-axis for better readability.

The bar chart is color-coded to differentiate between observation and answer lengths, and includes labels, a legend and a title for clarity.

This visualization helps analyze the balance between the size of the retrieved context and the detail in the generated response, offering insights into how the agent processes and responds to queries.

### Visualize the proportion of text used in observations

This step visualizes how much of the total text processed by the agent is used in observations (retrieved context) compared to the remaining text. A pie chart is created to provide an intuitive representation of the proportion.

def visualize\_text\_proportion(df):

     """Visualizes the proportion of text used in observations."""

     total\_text\_length = sum(df\["Observation"\].apply(len)) + sum(df\["Final Answer"\].apply(len))

     observation\_text\_length = sum(df\["Observation"\].apply(len))

     sizes = \[observation\_text\_length, total\_text\_length - observation\_text\_length\]

     labels = \["Observation Text", "Remaining Text"\]

     colors = \["#66b3ff", "#99ff99"\]

     plt.figure(figsize=(4, 4))

     plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)

     plt.title("Proportion of Text Used in Observations", fontsize=16)

     plt.show()\# Call the visualization function

visualize\_text\_proportion(df)

Thevisualize\_text\_proportion
function creates a pie chart to illustrate the proportion of total text that is used in observations (retrieved context) compared to the remaining text. It calculates the total text length by summing the character lengths of all observations and answers and then determines the portion contributed by observations alone.

This data is visualized in a pie chart, with clear labels for"Observation Text"
and"Remaining Text"
and distinct colors to enhance readability. The chart includes percentage values to make the proportions easy to interpret.

This visualization provides a high-level overview of how much text the agent uses as context during query processing, offering insights into the efficiency and focus of the retrieval process.

### Generate word clouds for observations and final answers

This code generates two word clouds to visually represent the most frequently occurring words in the Observation
and Final Answer
texts.

def generate\_wordclouds\_side\_by\_side(df):

      """Generates and displays word clouds for Observations and Final Answers side by side."""

      # Combine text for Observations and Final Answers

      observation\_text = " ".join(df\["Observation"\])

      final\_answer\_text = " ".join(df\["Final Answer"\])

      # Create word clouds

      observation\_wordcloud = WordCloud(width=800, height=400, background\_color="white").generate(observation\_text)

      final\_answer\_wordcloud = WordCloud(width=800, height=400, background\_color="white").generate(final\_answer\_text)

      # Create a side-by-side visualization

      plt.figure(figsize=(16, 8))

      # Plot the Observation word cloud

      plt.subplot(1, 2, 1)

      plt.imshow(observation\_wordcloud, interpolation="bilinear")

      plt.axis("off")

      plt.title("Word Cloud of Observations", fontsize=16)

      # Plot the Final Answer word cloud

      plt.subplot(1, 2, 2)

      plt.imshow(final\_answer\_wordcloud, interpolation="bilinear")

      plt.axis("off")

      plt.title("Word Cloud of Final Answers", fontsize=16)

      plt.tight\_layout()

      plt.show()\# Call the function to generate and display the word clouds

generate\_wordclouds\_side\_by\_side(df)

This code generates two word clouds to visually represent the most frequently occurring words in theObservation
andFinal Answer
texts, displaying them side by side for easy comparison. TheObservation
andFinal Answer
texts are first concatenated into two separate strings by using" ".join()
to combine all rows from the respective columns. TheWordCloud
library is then used to generate word clouds for each text with specific configurations.

To create a side-by-side visualization, subplots are used: the first subplot displays the word cloud forObservation
, and the second displays the one forFinal Answer
. Thetight\_layout()
function ensures neat spacing between the plots. These word clouds allow us to intuitively analyze the agent’s performance by highlighting key terms retrieved from the context (Observation
) and those emphasized in the responses (Final Answer
).

## Test the accuracy of the agent

In this section, we evaluate the agent's performance using multiple accuracy metrics:Keyword Matching
,BLEU Scores
,Precision/Recall
andF1 Scores
. These metrics provide a comprehensive view of how well the agent generates accurate and relevant responses based on user queries.

### Import required libraries

Before we begin the tests, we import the necessary libraries for accuracy evaluation.

from sklearn.feature\_extraction.text import CountVectorizer

from nltk.translate.bleu\_score import sentence\_bleu

from sklearn.metrics import precision\_score, recall\_score

These libraries include tools for keyword matching, BLEU score calculation, precision and recall evaluation. Ensure that you have installed these libraries in your environment to avoid import errors.

### Keyword matching accuracy

This test evaluates how well the generated answers include the keywords from the queries. It usesCountVectorizer
to tokenize and extract keywords from the queries and answers. The function calculates the proportion of query keywords present in the generated answer, marking the response as accurate if this proportion exceeds a threshold (0.5 by default). The results are added to the DataFrame under theKeyword Match Score
andIs Accurate columns
.

def keyword\_matching\_accuracy(df):

      """Checks if key phrases from the query are present in the final answer."""

      vectorizer = CountVectorizer(stop\_words='english')

      def check\_keywords(query, answer):

      query\_keywords = set(vectorizer.build\_tokenizer()(query.lower()))

      answer\_keywords = set(vectorizer.build\_tokenizer()(answer.lower()))

      common\_keywords = query\_keywords & answer\_keywords

      return len(common\_keywords) / len(query\_keywords) # Proportion of matched keywords

      df\["Keyword Match Score"\] = df.apply(lambda row: check\_keywords(row\["Action Input"\], row\["Final Answer"\]), axis=1)

      df\["Is Accurate"\] = df\["Keyword Match Score"\] >= 0.5 # Set a threshold for accuracy

      return df\# Apply keyword matching

df = keyword\_matching\_accuracy(df)

df.to\_csv("output\_with\_accuracy.csv", index=False)

df

### BLEU score calculation

This test measures _how closely the generated answers match the retrieved observations_.BLEU (Bilingual Evaluation Understudy)
is a popular metric for evaluating text similarity based onn-gram
overlaps. The function computesBLEU scores
for each query-answer pair and appends them to the DataFrame under the BLEU score column.

def calculate\_bleu\_scores(df):

    """Calculates BLEU scores for answers against observations."""

    df\["BLEU Score"\] = df.apply(

       lambda row: sentence\_bleu(\[row\["Observation"\].split()\], row\["Final Answer"\].split()),

       axis=1

       )

    return df\# Apply BLEU score calculation

df = calculate\_bleu\_scores(df)

df.to\_csv("output\_with\_bleu.csv", index=False)

### Precision and recall

Precision and recall are calculated to evaluate the relevance and completeness of the answers. Precision measures the proportion of retrieved words in the answer that are relevant, while recall measures the proportion of relevant words in the observation that appear in the answer.

These metrics are appended to the DataFrame under the Precision
and Recall
columns.

def calculate\_precision\_recall(df):

     """Calculates precision and recall for extractive answers."""

         def precision\_recall(observation, answer):

                observation\_set = set(observation.lower().split())

                answer\_set = set(answer.lower().split())

                precision = len(observation\_set & answer\_set) / len(answer\_set) if answer\_set else 0

                recall = len(observation\_set & answer\_set) / len(observation\_set) if observation\_set else 0

         return precision, recall

        df\[\["Precision", "Recall"\]\] = df.apply(

        lambda row: pd.Series(precision\_recall(row\["Observation"\], row\["Final Answer"\])),

        axis=1

        )

return df\# Apply precision/recall

df = calculate\_precision\_recall(df)

df.to\_csv("output\_with\_precision\_recall.csv", index=False)

df

### F1 score calculation

The F1 score combines precision and recall into a single metric, providing a balanced evaluation of relevance and completeness. The formula for F1 score is: **F1 Score = 2 \* (Precision \* Recall) / (Precision + Recall)**

The calculatedF1 scores
are added to the DataFrame under the F1 score column.

def calculate\_f1(df):

      """Calculates F1 scores based on precision and recall."""

          df\["F1 Score"\] = 2 \* (df\["Precision"\] \* df\["Recall"\]) / (df\["Precision"\] + df\["Recall"\])

          df\["F1 Score"\].fillna(0, inplace=True) # Handle divide by zero

          return df\# Apply F1 calculation

df = calculate\_f1(df)

df.to\_csv("output\_with\_f1.csv", index=False)

df

### Summarize accuracy metrics

Finally, a summary function consolidates all the metrics to provide an overview of the agent's performance. It calculates the total number of queries, the count and percentage of accurate responses and the average BLEU and F1 scores.

def summarize\_accuracy\_metrics(df):

      """Summarizes overall accuracy metrics."""

          total\_entries = len(df)

          accurate\_entries = df\["Is Accurate"\].sum()

          average\_bleu = df\["BLEU Score"\].mean()

          average\_f1 = df\["F1 Score"\].mean()

          print(f"Total Entries: {total\_entries}")

          print(f"Accurate Entries: {accurate\_entries} ({accurate\_entries / total\_entries \* 100:.2f}%)")

          print(f"Average BLEU Score: {average\_bleu:.2f}")

          print(f"Average F1 Score: {average\_f1:.2f}")\# Call summary function

summarize\_accuracy\_metrics(df)

* * *

OUTPUT

* * *

```
Total Entries: 4
Accurate Entries: 4 (100.00%)
Average BLEU Score: 0.04
Average F1 Score: 0.24

```

These accuracy tests offer a detailed evaluation of the agent’s ability to generate relevant and accurate responses. Each test focuses on a specific aspect, from keyword inclusion to text similarity and response completeness. The summary consolidates these metrics to give an overall performance snapshot.

## Summary

This tutorial guided you through building an autonomous agent powered by IBM’s Granite LLM and LangChain. Starting from text extraction to vectorization and query resolution, we covered the entire process of designing and implementing a functional LLM-based agent. Key steps included memory management with vector stores, query processing and generating responses by using Granite.

We evaluated the agent’s performance by using accuracy metrics such as keyword matching, BLEU scores, precision, recall and F1 scores. Visualizations such as bar charts, pie charts and word clouds provided additional insights into the agent’s behavior and effectiveness.

By completing this tutorial, you’ve learned how to design, test and visualize an LLM agent's performance. This foundation can be extended to tackle more complex datasets, improve accuracy and explore advanced features such as multiagent systems.

### Original URL
https://www.ibm.com/think/tutorials/llm-agent-orchestration-with-langchain-and-granite
</details>

---
<details>
<summary>How to Build AI Agents Using Plan-and-Execute Loops</summary>

# Building Your First AI Agent with Plan-and-Execute Loops

Christopher Frenchi

AI Research Engineer

Nish Tahir

Principal Software Engineer

Published:

December 12, 2024

An AI agent, also known as an autonomous agent or intelligent agent, is a generative AI program capable of interacting with its environment and accomplishing predetermined tasks autonomously. These artificial intelligence agents use the reasoning capabilities of large language models (LLMs) to autonomously develop plans for achieving a given task. The AI agents are further empowered to access real-world tools (e.g., a web browser) to execute their plans.

So how are these agents different from adjacent technologies like AI copilots? Though some have agentic AI built into them, copilots aren’t intended to function autonomously. Instead, they augment employees’ abilities through direct real-time interaction. Think of a customer service copilot retrieving information and suggesting responses to help a contact center agent resolve a complex query, or a field service copilot helping a technician repair an engine.

Back to building AI agents, let’s say we want to task one with ordering a pizza. Our instruction to the agent may look like:

```
You are a helpful autonomous assistant. Your task is to provide a comprehensive plan to execute the given task. Think step by step in a lot of detail. Assume you have access to a web browser. You may also ask for additional tools or access you need.
```

The AI agent will respond with a comprehensive plan to execute on the task (truncated for conciseness):

```

I can certainly assist you in purchasing a pizza from Domino's. Below is a step-by-step plan to ensure the order is placed successfully:

### Step 1: Gather Necessary Information

Before proceeding with the order, I'll need some details from you:

1. **Delivery Address:** The full address where the pizza should be delivered.

2. **Contact Information:** A phone number that the delivery driver can reach if needed.

…

### Step 2: Access Domino’s Website

1. Open a web browser.

2. Navigate to the Domino's website www.dominos.com.

### Step 3: Set Up the Order
…
```

By making sure the AI agent has the right tools, we empower it to complete the task on its own. In this tutorial, we’ll detail how to build your own autonomous plan-and-execute agent using Python, OpenAI, and Beautiful Soup.

## Set Up the AI Agent’s Tools and Memory

We’re going to build a plan-and-execute agent that’s capable of accepting instructions and navigating within the context of a web browser using links on the page. To accomplish this, we need to equip our agent with some capabilities:

- **Tools**: These are functions that the LLM can choose from and has access to. In this project, we’ll give it the ability to open a Chrome browser, navigate to a given page, and a function to parse out links on the page.
- **Memory**: This is a data structure that allows the LLM to remember things that have been done when planning future tasks. This includes bread crumbs for itself as well as what tools have been called. Memory can be kept short-term in the agent or long-term to track the overall goal progress.

With these core components in place, we can now set up our plan-and-execute loop.

## Create the Plan-and-Execute Loop

Agents work by analyzing their goals, creating steps that will reasonably accomplish that goal, and executing them using tools.

The heart of this process is the plan-and-execute loop. Our example is built around the following plan-and-execute loop:

1\. **First, we provide the agent with an instruction through the terminal**. In this case, “Get all the links from Hacker News.”

2\. **Next, the agent creates a plan to accomplish the task considering its instructions and the tools that it has at its disposal**. Given the tools we outlined, it responds with the following steps:

- open\_chrome
- navigate\_to\_hackernews
- get\_all\_links

3\. **Last, with a completed plan, the agent moves on to the execute phase, calling the functions in order**. When a function is called, the memory updates with the tool that was called and the related metadata, including the task, parameters, and result of the call.

## Enhance the Agent’s Memory, Prompts, and Tools

To give our tools greater functionality, we'll import several libraries. We chose these tools to showcase different ways agents can interact with a system:

- `subprocess` allows us to open system applications.
- `requests` and `BeautifulSoup` allow us to get links from a URL.
- `OpenAI` allows us to make LLM calls.

Setup:

```
from dotenv import load_dotenv
load_dotenv()

import os
import json
import subprocess

import requests
from bs4 import BeautifulSoup

from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
```

### Memory

For our AI agent’s plans to be effective, we need to know what the agent has done and still needs to do. This step is often abstracted in frameworks, making it important to call out. Without updating the agent’s “state,” the agent won't know what to call next. Think of this as similar to conversation history in ChatGPT.

Let’s create an `Agent` class and initialize it with variables that we can use for completing our task. We will need to know what tasks have been given and the LLM’s responses to those tasks. We can save those as `memory_tasks` and `memory_responses`. We will also want to store our planned actions along with any urls and links we might come across.

```
class Agent:
    def __init__(self):

        ## Memory
        self.memory_tasks = []
        self.memory_responses = []

        self.planned_actions = []
        self.url = ""
        self.links = []
```

### Plan-and-execute prompts

In order to make and execute a plan, we need to know what tools we have and a memory of the steps already taken.

If we want to navigate to a website and get links we should call out some possible tooling around opening Chrome, navigating to a website, and getting links. We can be clear in structuring a role, task, instructions, tools available, and the expected output format. With this, we are asking the LLM to plan the steps needed to accomplish our task with the given tools.

```
 self.planning_system_prompt = """
        % Role:
        You are an AI assistant helping a user plan a task. You have access to % Tools to help you with the order of tasks.

        % Task:
        Check the user's query and provide a plan for the order of tools to use for the task if appropriate. Do not execute only out line the tasks in order to accomplish the user's query.

        % Instructions:
        Create a plan for the order of tools to use for the task based on the user's query.
        Choose the approapriate tool or tools to use for the task based on the user's query.
        Tools that require content will have a variable_name and variable_value.
        A tool variable value can be another tool or a variable from a previous tool or the variable that is set in memory.
        If a variable is set in memory, use the variable value from memory.
        If the tool doesn't need variables, put the tool name in the variable_name and leave the variable_value empty.
        Memories will be stored for future use. If the output of a tool is needed, use the variable from memory.

        % Tools:
        open_chrome - Open Google Chrome
        navigate_to_hackernews - Navigate to Hacker News
        get_https_links - Get all HTTPS links from a page requires URL

        % Output:
        Plan of how to accomplish the user's query and what tools to use in order
        Each step should be one a single line with the tool to use and the variables if needed
        """
```

The `update_system_prompt` was added to show specifically the addition of “memory” being added during the execution of this workflow. Here the only major differences between the planning and execution prompts are in the addition of our memory and the response format of the prompt outputs.

When completing a task, the LLM will look at the previous memory of tasks completed and choose a tool to use for the given task. This will return a tool name along with any variables needed to complete the task in JSON format.

```
 def update_system_prompt(self):
        self.system_prompt = f"""
        % Role:
        You are an AI assistant helping a user with a task. You have access to % Tools to help you with the task.

        % Task:
        Check the user's query and provide a tool or tools to use for the task if appropriate. Always check the memory for previous questions and answers to choose the appropriate tool.

        % Memory:
        You have a memory of the user's questions and your answers.
        Previous Questions: {self.memory_tasks}
        Previous Answers: {self.memory_responses}

        % Instructions:
        Choose the appropriate tool to use for the task based on the user's query.
        Tools that require content will have a variable_name and variable_value.
        A tool variable value can be another tool or a variable from a previous tool.
        Check the memory for previous questions and answers to choose the appropriate variable if it has been set.
        If the tool doesn't need variables, put the tool name in the variable_name and leave the variable_value empty.

        % Tools:
        open_chrome - Open Google Chrome
        navigate_to_hackernews - Navigate to Hacker News
        get_https_links - Get all HTTPS links from a page requires URL

        % Output:
        json only

        {{
            "tool": "tool" or ""
            "variables": [\
                {{\
                    "variable_name": "variable_name" or ""\
                    "variable_value": "variable_value" or ""\
                }}\
            ]
        }}
        """
```

The `openai_call` function will allow us to call OpenAI and request the different format\_responses to showcase the differences between planning and execution. The JSON format is important here because we are using the responses of what tool to use to actually run that function.

```
def openai_call(self, query: str, system_prompt: str, json_format: bool = False):

        if json_format == True:
            format_response = { "type": "json_object" }
        else:
            format_response = { "type": "text" }

        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[\
                {"role": "system", "content": system_prompt},\
                {\
                    "role": "user",\
                    "content": f'User query: {query}'\
                }\
            ],
            response_format=format_response
        )
        llm_response = completion.choices[0].message.content
        print(llm_response)
        return llm_response
```

### Tools

Tools are functions that an agent has access to. Tools vary widely depending on the tasks the agent aims to accomplish.

In the code below, there are three specific tools that showcase how we can open a browser, navigate to a website, and get links. The way that these tools will be called is shown below when we put it all together. For now, think of this as a function an LLM can choose to call or not based on the current task of the plan.

The `def open_chrome()` tool can use Python’s subprocess to open a Chrome instance that will allow for navigation.

```
    def open_chrome(self):
        try:
            subprocess.run(["open", "-a", "Google Chrome"], check=True)
            print("Google Chrome opened successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to open Google Chrome: {e}")
```

The `def navigate_to_hackernews()` tool is hard-coded for this demo to showcase the idea and should be expanded to allow any URL.

```
    def navigate_to_hackernews(self, query):
        self.url = "https://news.ycombinator.com/"
        try:
            subprocess.run(["open", "-a", "Google Chrome", self.url], check=True)
            print(f"Opened '{query}' opened in Google Chrome.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to open search in Google Chrome: {e}")
```

The `def get_https_links()` tool uses the Beautiful Soup library. It will grab the URL of the page we are on and get all of the links.

```
    def get_https_links(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a')

            for link in links:
                link_text = link.get_text(strip=True)
                link_url = link.get('href')
                if link_url and link_url.startswith('https'):
                    print(f"Link Name: {link_text if link_text else 'No text'}")
                    print(f"Link URL: {link_url}")
                    print('---')
        except requests.exceptions.RequestException as e:
            print(f"Error fetching the URL: {e}")
```

## Bring All Your Work Together

Here, we create a function to plan and then execute that same plan.

To do this, we need to:

1. Send the planning system prompt to the LLM to create a plan and save those tasks as a list.
2. For every task in that list:
1. Keep track of the task given to the agent.
2. Pull in the task prompt with the history.
3. Call the LLM to choose a tool based on the given task and return as JSON.
4. Update our memory with the LLM response.
5. Match the tool called by the LLM to one of the tools available.
6. Run the function called.
7. Start the next task.

Here’s what this looks like in our code below:

```
## Plan and Execute
    def run(self, prompt: str):
        ## Planning
        planning = self.openai_call(prompt, self.planning_system_prompt)
        self.planned_actions = planning.split("\n")

        ## Task Execution
        for task in self.planned_actions:
            print(f"\n\nExecuting task: {task}")
            self.memory_tasks.append(task)
            self.update_system_prompt()
            task_call = self.openai_call(task, self.system_prompt, True)
            self.memory_responses.append(task_call)

            try:
                task_call_json = json.loads(task_call)
            except json.JSONDecodeError as e:
                print(f"Error decoding task call: {e}")
                continue

            tool = task_call_json.get("tool")
            variables = task_call_json.get("variables", [])

            if tool == "open_chrome":
                self.open_chrome()
                self.memory_responses.append("open_chrome=completed.")

            elif tool == "get_https_links":
                self.links = self.get_https_links(self.url)
                self.memory_responses.append(f"get_https_links=completed. variable urls = {self.links}")

            elif tool == "navigate_to_hackernews" :
                query = variables[0].get("variable_value", "")
                self.navigate_to_hackernews(query)
                self.memory_responses.append(f"navigate_to_hackernews=completed. variable query = {query}")

if __name__ == "__main__":
    agent = Agent()

    while True:
        prompt = input("Please enter a prompt (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
        agent.run(prompt)
```

With our agent in place, we can now run a request and see how the agent responds.

“Get all the links from Hacker News.”

The plan the agent generated:

```
1. open_chrome
2. navigate_to_hackernews
3. get_https_links with variable_name: URL and variable_value: (URL of Hacker News)
```

The outputs of each of the tasks being run:

```
Executing task: 1. open_chrome
{
    "tool": "open_chrome",
    "variables": [\
        {\
            "variable_name": "",\
            "variable_value": ""\
        }\
    ]
}
Google Chrome opened successfully.
```

```
Executing task: 2. navigate_to_hackernews
{
    "tool": "navigate_to_hackernews",
    "variables": [\
        {\
            "variable_name": "",\
            "variable_value": ""\
        }\
    ]
}
Opened '' opened in Google Chrome.
```

```
Executing task: 3. get_https_links with variable_name: URL and variable_value: (URL of Hacker News)
{
    "tool": "get_https_links",
    "variables": [\
        {\
            "variable_name": "URL",\
            "variable_value": "(URL of Hacker News)"\
        }\
    ]
}
Link Name: No text
Link URL: https://news.ycombinator.com
---
Link Name: Computing with Time: Microarchitectural Weird Machines
Link URL: https://cacm.acm.org/research-highlights/computing-with-time-microarchitectural-weird-machines/
---
Link Name: SQLiteStudio: Create, edit, browse SQLite databases
Link URL: https://sqlitestudio.pl/
---
Link Name: The Nearest Neighbor Attack

...
```

## Pitfalls When Building Plan-and-Execute Agents

If we start every agent call by making a plan, we don’t always have the flexibility to handle problems that arise. How we handle failures and what tools the agent has to mitigate these failures is very important. In our plan-and-execute approach, it isn’t easy to update our plan unless everything is in a try-catch and we update our plan on the failure.

More problems that could arise include:

- **One call for planning, and multiple calls for execution**. While we presented things this way to illustrate the steps, using a single prompt to both plan the next step and execute allows for a more dynamic agent.
- **This is a linear plan created at the onset, so if we run into any problems, we aren’t able to continue effectively**. This is where we can have an agent reflect on the outcome and choose the next best approach.
- **The AI agent will fall apart if it has to interact with any graphical user interfaces (GUIs)**. This is where multiple agents can come in or custom tools to allow for navigation of a specific application.
- **As we add more tools, the prompts can become a lot**. It would be best to start storing the tools outside of the prompt as they grow large and scoping agents for specific tasks.
- **We are responsible for creating all the tools**. Emerging standards look to establish best practices for interactions with service providers.
- **The memory is all being saved in the class itself now**. We should move this to a central location to persist this data.

Fortunately, most of these issues can be solved with more advanced techniques and concepts.

## Advanced Techniques and Concepts for Building AI Agents

With our understanding of plan-and-execute agents, we can build on what we know to create agents with more advanced capabilities. These include utilizing agentic frameworks and creating specific architectures.

While we covered the basics of building a plan-and-executive agent in this blog, you can push its abilities even further by incorporating advanced techniques such as:

- **ReAct** (Reason and Act): This merges the planning and execution prompts into one, allowing the prompt to think one step at a time.
- **ADaPT** (As-Needed Decomposition and Planning for complex Tasks): Perhaps best thought of as an extension of ReAct, ADaPT plans step by step, but also allows the agent to recursively break down problems when they arise or when steps are too large.
- **Reflexion**: This allows an agent to know how well it did or if it needs to try again after a task has been completed.

As for advanced concepts that you could incorporate into existing agents or alongside other techniques, you could explore options such as:

- **Recursion**: Agents call their own functions to keep tasks going until the problem is solved. Think of this as while True and be careful about infinite loops. This holds a lot of power in autonomy.
- **Multiple agents**: These are perfect if you require agents to talk to one another or orchestrate a more complex task. Not every agent needs access to all tools or systems.
- **Data standardization**: Standards such as Anthropic’s Model Context Protocol (MCP) are emerging to leverage tool and data service integrations with AI. Efforts around establishing best practices for how AI agents interact with data services is becoming increasingly important.
- **Frameworks**: Agentic frameworks allow for the creation of agents, but often have their own paradigms that are built off of the ideas presented above. Autogen, LangGraph, and LlamaIndex are a few examples of agentic frameworks.

With advanced techniques and concepts, you can build more robust agents capable of things like comparing different scenarios when making decisions, or helping customer service chatbots handle more complex queries.

## Start Building AI Agents Tailored to Your Business Needs

In this post, we looked at the anatomy of an AI agent through a practical plan-and-execute example, showcasing how agents use planning, execution, tools, and memory to accomplish tasks. Our basic implementation highlighted key considerations like tool selection and pitfalls, while also introducing more advanced concepts such as ReAct and multi-agent systems.

Mastering these concepts will prepare you to build agents capable of handling dynamic and generative tasks, including for high-value use cases like:

- **Process automation** where agents enhance workflows by navigating internal systems, aggregating data from multiple sources, and generating reports.
- **Product development** where AI-powered features can adapt to user requests and interact with various APIs and services.

### Original URL
https://www.willowtreeapps.com/craft/building-ai-agents-with-plan-and-execute
</details>

---
<details>
<summary>What is AI Agent Planning? | IBM</summary>

# What is AI agent planning?

## What is AI agent planning?

AI agent planning refers to the process by which an artificial intelligence (AI) agent determines a sequence of actions to achieve a specific goal. It involves decision-making, goal prioritization and action sequencing, often using various planning algorithms and frameworks.

AI agent @https://www.ibm.com/think/topics/ai-agents planning is a module common to many types of agents @https://www.ibm.com/think/topics/components-of-ai-agents that exists alongside other modules such as perception, reasoning, decision-making, action, memory, communication and learning. Planning works in conjunction with these other modules to help ensure that agents achieve outcomes desired by their designers.

Not all agents can plan. Unlike simple reactive agents that respond immediately to inputs, planning agents anticipate future states and generate a structured action plan before execution. This makes AI planning essential for automation @https://www.ibm.com/think/topics/enterprise-automation tasks that require multistep decision-making, optimization and adaptability.

## How AI agent planning works

Advances in large language models @https://www.ibm.com/think/topics/large-language-models (LLMs) such as OpenAI’s GPT @https://www.ibm.com/think/topics/gpt and related techniques involving machine learning algorithms @https://www.ibm.com/think/topics/machine-learning-algorithms resulted in the generative AI @https://www.ibm.com/think/topics/generative-ai (gen AI) boom of recent years, and further advancements have led to the emerging field of autonomous agents.

By integrating tools, APIs, hardware interfaces and other external resources, agentic AI systems are increasingly autonomous, capable of real-time decision-making and adept at problem-solving across various use cases.

Complex agents can’t act without making a decision, and they can’t make good decisions without first making a plan. Agentic planning consists of several key components that work together to encourage optimal decision-making.

### Goal definition

The first and most critical step in AI planning is defining a clear objective. The goal serves as the guiding principle for the agent’s decision-making process, determining the end state it seeks to achieve. Goals can either be static, remaining unchanged throughout the planning process, or dynamic, adjusting based on environmental conditions or user interactions.

For instance, a self-driving car might have a goal of reaching a specific destination efficiently while adhering to safety regulations. Without a well-defined goal, an agent would lack direction, leading to erratic or inefficient behavior.

If the goal is complex, agentic AI models @https://www.ibm.com/think/topics/ai-model will break it down into smaller, more manageable sub-goals in a process called task decomposition. This allows the system to focus on complex tasks in a hierarchical manner.

LLMs play a vital role in task decomposition, breaking down a high-level goal into smaller subtasks and then executing those subtasks through various steps. For instance, a user might ask a chatbot @https://www.ibm.com/think/topics/chatbots with a natural language prompt to plan a trip.

The agent would first decompose the task into components such as booking flights, finding hotels and planning an itinerary. Once decomposed, the agent can use application programming interfaces (APIs) @https://www.ibm.com/think/topics/api to fetch real-time data, check pricing and even suggest destinations.

### State representation

To plan effectively, an agent must have a structured understanding of its environment. This understanding is achieved through state representation, which models the current conditions, constraints and contextual factors that influence decision-making.

Agents have some built-in knowledge from their training data or datasets @https://www.ibm.com/think/topics/dataset representing previous interactions, but perception is required for agents to have a real-time understanding of their environment. Agents collect data through sensory input, allowing it to model its environment, along with user input and data describing its own internal state.

The complexity of state representation varies depending on the task. For example, in a chess game, the state includes the position of all pieces on the board, while in a robotic navigation system, the state might involve spatial coordinates, obstacles and terrain conditions.

The accuracy of state representation directly impacts an agent’s ability to make informed decisions, as it determines how well the agent can predict the outcomes of its actions.

### Action sequencing

Once the agent has established its goal and assessed its environment, it must determine a sequence of actions that will transition it from its current state to the desired goal state. This process, known as action sequencing, involves structuring a logical and efficient set of steps that the agent must follow.

The agent needs to identify potential actions, reduce that list to optimal actions, prioritize them and identifying dependencies between actions and conditional steps based on potential changes in the environment. The agent might allocate resources to each step in the sequence, or schedule actions based on environmental constraints.

For example, a robotic vacuum cleaner needs to decide the most effective path to clean a room, ensuring it covers all necessary areas without unnecessary repetition. If the sequence of actions is not well planned, the AI agent might take inefficient or redundant steps, leading to wasted resources and increased execution time.

The ReAct framework is a methodology used in AI for handling dynamic decision-making. In the ReAct framework, reasoning refers to the cognitive process where the agent determines what actions or strategies are required to achieve a specific goal.

This phase is similar to the planning phase in agentic AI, where the agent generates a sequence of steps to solve a problem or fulfill a task. Other emerging frameworks include ReWOO, RAISE and Reflexion, each of which has its own strengths and weaknesses.

### Optimization and evaluation

AI planning often involves selecting the most optimal path to achieving a goal, especially when multiple options are available. Optimization helps ensure that an agent's chosen sequence of actions is the most efficient, cost-effective or otherwise beneficial given the circumstances. This process often requires evaluating different factors such as time, resource consumption, risks and potential rewards.

For example, a warehouse robot tasked with retrieving items must determine the shortest and safest route to avoid collisions and reduce operational time. Without proper optimization, AI agents might execute plans that are functional but suboptimal, leading to inefficiencies. Several methods can be used to optimize decision-making, including:

#### Heuristic search

Heuristic search algorithms help agents find optimal solutions by estimating the best path toward a goal. These algorithms rely on heuristic functions—mathematical estimates of how close a given state is to the desired goal. Heuristic searches are particularly effective for structured environments where agents need to find optimal paths quickly.

#### Reinforcement learning

Reinforcement learning enables agents to optimize planning through trial and error, learning which sequences of actions lead to the best outcomes over time. An agent interacts with an environment, receives feedback in the form of rewards or penalties, and refines its strategies accordingly.

#### Probabilistic planning

In real-world scenarios, AI agents often operate in uncertain environments where outcomes are not deterministic. Probabilistic planning methods account for uncertainty by evaluating multiple possible outcomes and selecting actions with the highest expected utility.

### Collaboration

Single agent planning is one thing, but in a multiagent system @https://www.ibm.com/think/topics/multiagent-system, AI agents must work autonomously while interacting with each other to achieve individual or collective goals.

The planning process for AI agents in a multiagent system is more complex than for a single agent because agents must not only plan their own actions but also consider the actions of other agents and how their decisions interact with those of others.

Depending on the agentic architecture @https://www.ibm.com/think/topics/agentic-architecture, each agent in the system typically has its own individual goals, which might involve accomplishing specific tasks or maximizing a reward function. In many multiagent systems, agents need to work together to achieve shared goals.

These goals could be defined by an overarching system or emerge from the agents’ interactions. Agents need mechanisms to communicate and align their goals, especially in cooperative scenarios. This could be done through explicit messaging, shared task definitions or implicit coordination.

Planning in multiagent systems can be centralized, where a single entity or controller—likely an LLM agent—generates the plan for the entire system.

Each agent receives instructions or plans from this central authority. It can also be decentralized, where agents generate their own plans but work collaboratively to help ensure that they align with each other and contribute to global objectives, often requiring communication and negotiation.

This collaborative decision-making process enhances efficiency, reduces biases @https://www.ibm.com/think/topics/ai-bias in task execution, helps to avoid hallucinations @https://www.ibm.com/think/topics/ai-hallucinations through cross-validation and consensus-building and encourages the agents to work toward a common goal.

## After planning

The phases in agentic AI workflows @https://www.ibm.com/think/topics/ai-workflow do not always occur in a strict step-by-step linear fashion. While these phases are often distinct in conceptualization, in practice, they are frequently interleaved or iterative, depending on the nature of the task and the complexity of the environment in which the agent operates.

AI solutions can differ depending on their design, but in a typical agentic workflow @https://www.ibm.com/think/topics/agentic-workflows, the next phase after planning is action execution, where the agent carries out the actions defined in the plan. This involves performing tasks and interacting with external systems or knowledge bases with retrieval augmented generation (RAG) @https://www.ibm.com/think/topics/retrieval-augmented-generation, tool use and function calling ( tool calling @https://www.ibm.com/think/topics/tool-calling).

Building AI agents @https://www.ibm.com/ai-agents for these capabilities might involve LangChain @https://www.ibm.com/think/topics/langchain. Python scripts, JSON data structures and other programmatic tools enhance the AI’s ability to make decisions.

After executing plans, some agents can use memory to learn from their experiences and iterate their behavior accordingly.

In dynamic environments, the planning process must be adaptive. Agents continuously receive feedback about the environment and other agents’ actions and must adjust their plans accordingly. This might involve revising goals, adjusting action sequences, or adapting to new agents entering or leaving the system.

When an agent detects that its current plan is no longer feasible (for example, due to a conflict with another agent or a change in the environment), it might engage in replanning to adjust its strategy. Agents can adjust their strategies using chain of thought @https://www.ibm.com/think/topics/chain-of-thoughts reasoning, a process where they reflect on the steps needed to reach their objective before taking action.

### Original URL
https://www.ibm.com/think/topics/ai-agent-planning
</details>

---
<details>
<summary>LLM Powered Autonomous Agents | Lil'Log</summary>

Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT @https://github.com/Significant-Gravitas/Auto-GPT, GPT-Engineer @https://github.com/AntonOsika/gpt-engineer and BabyAGI @https://github.com/yoheinakajima/babyagi, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.

# Agent System Overview

In a LLM-powered autonomous agent system, LLM functions as the agent’s brain, complemented by several key components:

- **Planning**
  - Subgoal and decomposition: The agent breaks down large tasks into smaller, manageable subgoals, enabling efficient handling of complex tasks.
  - Reflection and refinement: The agent can do self-criticism and self-reflection over past actions, learn from mistakes and refine them for future steps, thereby improving the quality of final results.
- **Memory**
  - Short-term memory: I would consider all the in-context learning (See Prompt Engineering @https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) as utilizing short-term memory of the model to learn.
  - Long-term memory: This provides the agent with the capability to retain and recall (infinite) information over extended periods, often by leveraging an external vector store and fast retrieval.
- **Tool use**
  - The agent learns to call external APIs for extra information that is missing from the model weights (often hard to change after pre-training), including current information, code execution capability, access to proprietary information sources and more.

# Component One: Planning

A complicated task usually involves many steps. An agent needs to know what they are and plan ahead.

## Task Decomposition

**Chain of thought** @https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/#chain-of-thought-cot (CoT; Wei et al. 2022 @https://arxiv.org/abs/2201.11903) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.

**Tree of Thoughts** ( Yao et al. 2023 @https://arxiv.org/abs/2305.10601) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.

Task decomposition can be done (1) by LLM with simple prompting like `"Steps for XYZ.\n1."`, `"What are the subgoals for achieving XYZ?"`, (2) by using task-specific instructions; e.g. `"Write a story outline."` for writing a novel, or (3) with human inputs.

Another quite distinct approach, **LLM+P** ( Liu et al. 2023 @https://arxiv.org/abs/2304.11477), involves relying on an external classical planner to do long-horizon planning. This approach utilizes the Planning Domain Definition Language (PDDL) as an intermediate interface to describe the planning problem. In this process, LLM (1) translates the problem into “Problem PDDL”, then (2) requests a classical planner to generate a PDDL plan based on an existing “Domain PDDL”, and finally (3) translates the PDDL plan back into natural language. Essentially, the planning step is outsourced to an external tool, assuming the availability of domain-specific PDDL and a suitable planner which is common in certain robotic setups but not in many other domains.

## Self-Reflection

Self-reflection is a vital aspect that allows autonomous agents to improve iteratively by refining past action decisions and correcting previous mistakes. It plays a crucial role in real-world tasks where trial and error are inevitable.

**ReAct** ( Yao et al. 2023 @https://arxiv.org/abs/2210.03629) integrates reasoning and acting within LLM by extending the action space to be a combination of task-specific discrete actions and the language space. The former enables LLM to interact with the environment (e.g. use Wikipedia search API), while the latter prompting LLM to generate reasoning traces in natural language.

The ReAct prompt template incorporates explicit steps for LLM to think, roughly formatted as:

```hljs less
Thought: ...
Action: ...
Observation: ...
... (Repeated many times)
copy
```

In both experiments on knowledge-intensive tasks and decision-making tasks, `ReAct` works better than the `Act`-only baseline where `Thought: …` step is removed.

**Reflexion** ( Shinn & Labash 2023 @https://arxiv.org/abs/2303.11366) is a framework to equip agents with dynamic memory and self-reflection capabilities to improve reasoning skills. Reflexion has a standard RL setup, in which the reward model provides a simple binary reward and the action space follows the setup in ReAct where the task-specific action space is augmented with language to enable complex reasoning steps. After each action at, the agent computes a heuristic ht and optionally may _decide to reset_ the environment to start a new trial depending on the self-reflection results.

The heuristic function determines when the trajectory is inefficient or contains hallucination and should be stopped. Inefficient planning refers to trajectories that take too long without success. Hallucination is defined as encountering a sequence of consecutive identical actions that lead to the same observation in the environment.

Self-reflection is created by showing two-shot examples to LLM and each example is a pair of (failed trajectory, ideal reflection for guiding future changes in the plan). Then reflections are added into the agent’s working memory, up to three, to be used as context for querying LLM.

Experiments on AlfWorld Env and HotpotQA. Hallucination is a more common failure than inefficient planning in AlfWorld.

**Chain of Hindsight** (CoH; Liu et al. 2023 @https://arxiv.org/abs/2302.02676) encourages the model to improve on its own outputs by explicitly presenting it with a sequence of past outputs, each annotated with feedback. Human feedback data is a collection of Dh={(x,yi,ri,zi)}i=1n, where x is the prompt, each yi is a model completion, ri is the human rating of yi, and zi is the corresponding human-provided hindsight feedback. Assume the feedback tuples are ranked by reward, rn≥rn−1≥⋯≥r1 The process is supervised fine-tuning where the data is a sequence in the form of τh=(x,zi,yi,zj,yj,…,zn,yn), where ≤i≤j≤n. The model is finetuned to only predict yn where conditioned on the sequence prefix, such that the model can self-reflect to produce better output based on the feedback sequence. The model can optionally receive multiple rounds of instructions with human annotators at test time.

To avoid overfitting, CoH adds a regularization term to maximize the log-likelihood of the pre-training dataset. To avoid shortcutting and copying (because there are many common words in feedback sequences), they randomly mask 0% - 5% of past tokens during training.

The training dataset in their experiments is a combination of WebGPT comparisons @https://huggingface.co/datasets/openai/webgpt_comparisons, summarization from human feedback @https://github.com/openai/summarize-from-feedback and human preference dataset @https://github.com/anthropics/hh-rlhf.

The idea of CoH is to present a history of sequentially improved outputs in context and train the model to take on the trend to produce better outputs. **Algorithm Distillation** (AD; Laskin et al. 2023 @https://arxiv.org/abs/2210.14215) applies the same idea to cross-episode trajectories in reinforcement learning tasks, where an _algorithm_ is encapsulated in a long history-conditioned policy. Considering that an agent interacts with the environment many times and in each episode the agent gets a little better, AD concatenates this learning history and feeds that into the model. Hence we should expect the next predicted action to lead to better performance than previous trials. The goal is to learn the process of RL instead of training a task-specific policy itself.

The paper hypothesizes that any algorithm that generates a set of learning histories can be distilled into a neural network by performing behavioral cloning over actions. The history data is generated by a set of source policies, each trained for a specific task. At the training stage, during each RL run, a random task is sampled and a subsequence of multi-episode history is used for training, such that the learned policy is task-agnostic.

In reality, the model has limited context window length, so episodes should be short enough to construct multi-episode history. Multi-episodic contexts of 2-4 episodes are necessary to learn a near-optimal in-context RL algorithm. The emergence of in-context RL requires long enough context.

In comparison with three baselines, including ED (expert distillation, behavior cloning with expert trajectories instead of learning history), source policy (used for generating trajectories for distillation by UCB @https://lilianweng.github.io/posts/2018-01-23-multi-armed-bandit/#upper-confidence-bounds), RL^2 ( Duan et al. 2017 @https://arxiv.org/abs/1611.02779; used as upper bound since it needs online RL), AD demonstrates in-context RL with performance getting close to RL^2 despite only using offline RL and learns much faster than other baselines. When conditioned on partial training history of the source policy, AD also improves much faster than ED baseline.

# Component Two: Memory

## Types of Memory

Memory can be defined as the processes used to acquire, store, retain, and later retrieve information. There are several types of memory in human brains.

1. **Sensory Memory**: This is the earliest stage of memory, providing the ability to retain impressions of sensory information (visual, auditory, etc) after the original stimuli have ended. Sensory memory typically only lasts for up to a few seconds. Subcategories include iconic memory (visual), echoic memory (auditory), and haptic memory (touch).

2. **Short-Term Memory** (STM) or **Working Memory**: It stores information that we are currently aware of and needed to carry out complex cognitive tasks such as learning and reasoning. Short-term memory is believed to have the capacity of about 7 items ( Miller 1956 @https://lilianweng.github.io/posts/2023-06-23-agent/psychclassics.yorku.ca/Miller/) and lasts for 20-30 seconds.

3. **Long-Term Memory** (LTM): Long-term memory can store information for a remarkably long time, ranging from a few days to decades, with an essentially unlimited storage capacity. There are two subtypes of LTM:
   - Explicit / declarative memory: This is memory of facts and events, and refers to those memories that can be consciously recalled, including episodic memory (events and experiences) and semantic memory (facts and concepts).
   - Implicit / procedural memory: This type of memory is unconscious and involves skills and routines that are performed automatically, like riding a bike or typing on a keyboard.

We can roughly consider the following mappings:

- Sensory memory as learning embedding representations for raw inputs, including text, image or other modalities;
- Short-term memory as in-context learning. It is short and finite, as it is restricted by the finite context window length of Transformer.
- Long-term memory as the external vector store that the agent can attend to at query time, accessible via fast retrieval.

## Maximum Inner Product Search (MIPS)

The external memory can alleviate the restriction of finite attention span. A standard practice is to save the embedding representation of information into a vector store database that can support fast maximum inner-product search ( MIPS @https://en.wikipedia.org/wiki/Maximum_inner-product_search). To optimize the retrieval speed, the common choice is the _approximate nearest neighbors (ANN)​_ algorithm to return approximately top k nearest neighbors to trade off a little accuracy lost for a huge speedup.

A couple common choices of ANN algorithms for fast MIPS:

- **LSH** @https://en.wikipedia.org/wiki/Locality-sensitive_hashing (Locality-Sensitive Hashing): It introduces a _hashing_ function such that similar input items are mapped to the same buckets with high probability, where the number of buckets is much smaller than the number of inputs.
- **ANNOY** @https://github.com/spotify/annoy (Approximate Nearest Neighbors Oh Yeah): The core data structure are _random projection trees_, a set of binary trees where each non-leaf node represents a hyperplane splitting the input space into half and each leaf stores one data point. Trees are built independently and at random, so to some extent, it mimics a hashing function. ANNOY search happens in all the trees to iteratively search through the half that is closest to the query and then aggregates the results. The idea is quite related to KD tree but a lot more scalable.
- **HNSW** @https://arxiv.org/abs/1603.09320 (Hierarchical Navigable Small World): It is inspired by the idea of small world networks @https://en.wikipedia.org/wiki/Small-world_network where most nodes can be reached by any other nodes within a small number of steps; e.g. “six degrees of separation” feature of social networks. HNSW builds hierarchical layers of these small-world graphs, where the bottom layers contain the actual data points. The layers in the middle create shortcuts to speed up search. When performing a search, HNSW starts from a random node in the top layer and navigates towards the target. When it can’t get any closer, it moves down to the next layer, until it reaches the bottom layer. Each move in the upper layers can potentially cover a large distance in the data space, and each move in the lower layers refines the search quality.
- **FAISS** @https://github.com/facebookresearch/faiss (Facebook AI Similarity Search): It operates on the assumption that in high dimensional space, distances between nodes follow a Gaussian distribution and thus there should exist _clustering_ of data points. FAISS applies vector quantization by partitioning the vector space into clusters and then refining the quantization within clusters. Search first looks for cluster candidates with coarse quantization and then further looks into each cluster with finer quantization.
- **ScaNN** @https://github.com/google-research/google-research/tree/master/scann (Scalable Nearest Neighbors): The main innovation in ScaNN is _anisotropic vector quantization_. It quantizes a data point xi to x~i such that the inner product ⟨q,xi⟩ is as similar to the original distance of ∠q,x~i as possible, instead of picking the closet quantization centroid points.

Check more MIPS algorithms and performance comparison in ann-benchmarks.com @https://ann-benchmarks.com/.

# Component Three: Tool Use

Tool use is a remarkable and distinguishing characteristic of human beings. We create, modify and utilize external objects to do things that go beyond our physical and cognitive limits. Equipping LLMs with external tools can significantly extend the model capabilities.

**MRKL** ( Karpas et al. 2022 @https://arxiv.org/abs/2205.00445), short for “Modular Reasoning, Knowledge and Language”, is a neuro-symbolic architecture for autonomous agents. A MRKL system is proposed to contain a collection of “expert” modules and the general-purpose LLM works as a router to route inquiries to the best suitable expert module. These modules can be neural (e.g. deep learning models) or symbolic (e.g. math calculator, currency converter, weather API).

They did an experiment on fine-tuning LLM to call a calculator, using arithmetic as a test case. Their experiments showed that it was harder to solve verbal math problems than explicitly stated math problems because LLMs (7B Jurassic1-large model) failed to extract the right arguments for the basic arithmetic reliably. The results highlight when the external symbolic tools can work reliably, _knowing when to and how to use the tools are crucial_, determined by the LLM capability.

Both **TALM** (Tool Augmented Language Models; Parisi et al. 2022 @https://arxiv.org/abs/2205.12255) and **Toolformer** ( Schick et al. 2023 @https://arxiv.org/abs/2302.04761) fine-tune a LM to learn to use external tool APIs. The dataset is expanded based on whether a newly added API call annotation can improve the quality of model outputs. See more details in the “External APIs” section @https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/#external-apis of Prompt Engineering.

ChatGPT Plugins @https://openai.com/blog/chatgpt-plugins and OpenAI API function calling @https://platform.openai.com/docs/guides/gpt/function-calling are good examples of LLMs augmented with tool use capability working in practice. The collection of tool APIs can be provided by other developers (as in Plugins) or self-defined (as in function calls).

**HuggingGPT** ( Shen et al. 2023 @https://arxiv.org/abs/2303.17580) is a framework to use ChatGPT as the task planner to select models available in HuggingFace platform according to the model descriptions and summarize the response based on the execution results.

The system comprises of 4 stages:

**(1) Task planning**: LLM works as the brain and parses the user requests into multiple tasks. There are four attributes associated with each task: task type, ID, dependencies, and arguments. They use few-shot examples to guide LLM to do task parsing and planning.

Instruction:

The AI assistant can parse user input to several tasks: \[{"task": task, "id", task\_id, "dep": dependency\_task\_ids, "args": {"text": text, "image": URL, "audio": URL, "video": URL}}\]. The "dep" field denotes the id of the previous task which generates a new resource that the current task relies on. A special tag "-task\_id" refers to the generated text image, audio and video in the dependency task with id as task\_id. The task MUST be selected from the following options: {{ Available Task List }}. There is a logical relationship between tasks, please note their order. If the user input can't be parsed, you need to reply empty JSON. Here are several cases for your reference: {{ Demonstrations }}. The chat history is recorded as {{ Chat History }}. From this chat history, you can find the path of the user-mentioned resources for your task planning.

**(2) Model selection**: LLM distributes the tasks to expert models, where the request is framed as a multiple-choice question. LLM is presented with a list of models to choose from. Due to the limited context length, task type based filtration is needed.

Instruction:

Given the user request and the call command, the AI assistant helps the user to select a suitable model from a list of models to process the user request. The AI assistant merely outputs the model id of the most appropriate model. The output must be in a strict JSON format: "id": "id", "reason": "your detail reason for the choice". We have a list of models for you to choose from {{ Candidate Models }}. Please select one model from the list.

**(3) Task execution**: Expert models execute on the specific tasks and log results.

Instruction:

With the input and the inference results, the AI assistant needs to describe the process and results. The previous stages can be formed as - User Input: {{ User Input }}, Task Planning: {{ Tasks }}, Model Selection: {{ Model Assignment }}, Task Execution: {{ Predictions }}. You must first answer the user's request in a straightforward manner. Then describe the task process and show your analysis and model inference results to the user in the first person. If inference results contain a file path, must tell the user the complete file path.

**(4) Response generation**: LLM receives the execution results and provides summarized results to users.

To put HuggingGPT into real world usage, a couple challenges need to solve: (1) Efficiency improvement is needed as both LLM inference rounds and interactions with other models slow down the process; (2) It relies on a long context window to communicate over complicated task content; (3) Stability improvement of LLM outputs and external model services.

**API-Bank** ( Li et al. 2023 @https://arxiv.org/abs/2304.08244) is a benchmark for evaluating the performance of tool-augmented LLMs. It contains 53 commonly used API tools, a complete tool-augmented LLM workflow, and 264 annotated dialogues that involve 568 API calls. The selection of APIs is quite diverse, including search engines, calculator, calendar queries, smart home control, schedule management, health data management, account authentication workflow and more. Because there are a large number of APIs, LLM first has access to API search engine to find the right API to call and then uses the corresponding documentation to make a call.

In the API-Bank workflow, LLMs need to make a couple of decisions and at each step we can evaluate how accurate that decision is. Decisions include:

1. Whether an API call is needed.
2. Identify the right API to call: if not good enough, LLMs need to iteratively modify the API inputs (e.g. deciding search keywords for Search Engine API).
3. Response based on the API results: the model can choose to refine and call again if results are not satisfied.

This benchmark evaluates the agent’s tool use capabilities at three levels:

- Level-1 evaluates the ability to _call the API_. Given an API’s description, the model needs to determine whether to call a given API, call it correctly, and respond properly to API returns.
- Level-2 examines the ability to _retrieve the API_. The model needs to search for possible APIs that may solve the user’s requirement and learn how to use them by reading documentation.
- Level-3 assesses the ability to _plan API beyond retrieve and call_. Given unclear user requests (e.g. schedule group meetings, book flight/hotel/restaurant for a trip), the model may have to conduct multiple API calls to solve it.

# Case Studies

## Scientific Discovery Agent

**ChemCrow** ( Bran et al. 2023 @https://arxiv.org/abs/2304.05376) is a domain-specific example in which LLM is augmented with 13 expert-designed tools to accomplish tasks across organic synthesis, drug discovery, and materials design. The workflow, implemented in LangChain @https://github.com/hwchase17/langchain, reflects what was previously described in the ReAct @https://lilianweng.github.io/posts/2023-06-23-agent/#react and MRKLs @https://lilianweng.github.io/posts/2023-06-23-agent/#mrkl and combines CoT reasoning with tools relevant to the tasks:

- The LLM is provided with a list of tool names, descriptions of their utility, and details about the expected input/output.
- It is then instructed to answer a user-given prompt using the tools provided when necessary. The instruction suggests the model to follow the ReAct format - `Thought, Action, Action Input, Observation`.

One interesting observation is that while the LLM-based evaluation concluded that GPT-4 and ChemCrow perform nearly equivalently, human evaluations with experts oriented towards the completion and chemical correctness of the solutions showed that ChemCrow outperforms GPT-4 by a large margin. This indicates a potential problem with using LLM to evaluate its own performance on domains that requires deep expertise. The lack of expertise may cause LLMs not knowing its flaws and thus cannot well judge the correctness of task results.

Boiko et al. (2023) @https://arxiv.org/abs/2304.05332 also looked into LLM-empowered agents for scientific discovery, to handle autonomous design, planning, and performance of complex scientific experiments. This agent can use tools to browse the Internet, read documentation, execute code, call robotics experimentation APIs and leverage other LLMs.

For example, when requested to `"develop a novel anticancer drug"`, the model came up with the following reasoning steps:

1. inquired about current trends in anticancer drug discovery;
2. selected a target;
3. requested a scaffold targeting these compounds;
4. Once the compound was identified, the model attempted its synthesis.

They also discussed the risks, especially with illicit drugs and bioweapons. They developed a test set containing a list of known chemical weapon agents and asked the agent to synthesize them. 4 out of 11 requests (36%) were accepted to obtain a synthesis solution and the agent attempted to consult documentation to execute the procedure. 7 out of 11 were rejected and among these 7 rejected cases, 5 happened after a Web search while 2 were rejected based on prompt only.

## Generative Agents Simulation

**Generative Agents** ( Park, et al. 2023 @https://arxiv.org/abs/2304.03442) is super fun experiment where 25 virtual characters, each controlled by a LLM-powered agent, are living and interacting in a sandbox environment, inspired by The Sims. Generative agents create believable simulacra of human behavior for interactive applications.

The design of generative agents combines LLM with memory, planning and reflection mechanisms to enable agents to behave conditioned on past experience, as well as to interact with other agents.

- **Memory** stream: is a long-term memory module (external database) that records a comprehensive list of agents’ experience in natural language.

  - Each element is an _observation_, an event directly provided by the agent.
    \- Inter-agent communication can trigger new natural language statements.
- **Retrieval** model: surfaces the context to inform the agent’s behavior, according to relevance, recency and importance.

  - Recency: recent events have higher scores
  - Importance: distinguish mundane from core memories. Ask LM directly.
  - Relevance: based on how related it is to the current situation / query.
- **Reflection** mechanism: synthesizes memories into higher level inferences over time and guides the agent’s future behavior. They are _higher-level summaries of past events_ (<\- note that this is a bit different from self-reflection @https://lilianweng.github.io/posts/2023-06-23-agent/#self-reflection above)

  - Prompt LM with 100 most recent observations and to generate 3 most salient high-level questions given a set of observations/statements. Then ask LM to answer those questions.
- **Planning & Reacting**: translate the reflections and the environment information into actions

  - Planning is essentially in order to optimize believability at the moment vs in time.
  - Prompt template: `{Intro of an agent X}. Here is X's plan today in broad strokes: 1)`
  - Relationships between agents and observations of one agent by another are all taken into consideration for planning and reacting.
  - Environment information is present in a tree structure.

This fun simulation results in emergent social behavior, such as information diffusion, relationship memory (e.g. two agents continuing the conversation topic) and coordination of social events (e.g. host a party and invite many others).

## Proof-of-Concept Examples

AutoGPT @https://github.com/Significant-Gravitas/Auto-GPT has drawn a lot of attention into the possibility of setting up autonomous agents with LLM as the main controller. It has quite a lot of reliability issues given the natural language interface, but nevertheless a cool proof-of-concept demo. A lot of code in AutoGPT is about format parsing.

Here is the system message used by AutoGPT, where `{{...}}` are user inputs:

```hljs yaml
You are {{ai-name}}, {{user-provided AI bot description}}.
Your decisions must always be made independently without seeking user assistance. Play to your strengths as an LLM and pursue simple strategies with no legal complications.

GOALS:

1. {{user-provided goal 1}}
2. {{user-provided goal 2}}
3. ...
4. ...
5. ...

Constraints:
1. ~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.
2. If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.
3. No user assistance
4. Exclusively use the commands listed in double quotes e.g. "command name"
5. Use subprocesses for commands that will not terminate within a few minutes

Commands:
1. Google Search: "google", args: "input": "<search>"
2. Browse Website: "browse_website", args: "url": "<url>", "question": "<what_you_want_to_find_on_website>"
3. Start GPT Agent: "start_agent", args: "name": "<name>", "task": "<short_task_desc>", "prompt": "<prompt>"
4. Message GPT Agent: "message_agent", args: "key": "<key>", "message": "<message>"
5. List GPT Agents: "list_agents", args:
6. Delete GPT Agent: "delete_agent", args: "key": "<key>"
7. Clone Repository: "clone_repository", args: "repository_url": "<url>", "clone_path": "<directory>"
8. Write to file: "write_to_file", args: "file": "<file>", "text": "<text>"
9. Read file: "read_file", args: "file": "<file>"
10. Append to file: "append_to_file", args: "file": "<file>", "text": "<text>"
11. Delete file: "delete_file", args: "file": "<file>"
12. Search Files: "search_files", args: "directory": "<directory>"
13. Analyze Code: "analyze_code", args: "code": "<full_code_string>"
14. Get Improved Code: "improve_code", args: "suggestions": "<list_of_suggestions>", "code": "<full_code_string>"
15. Write Tests: "write_tests", args: "code": "<full_code_string>", "focus": "<list_of_focus_areas>"
16. Execute Python File: "execute_python_file", args: "file": "<file>"
17. Generate Image: "generate_image", args: "prompt": "<prompt>"
18. Send Tweet: "send_tweet", args: "text": "<text>"
19. Do Nothing: "do_nothing", args:
20. Task Complete (Shutdown): "task_complete", args: "reason": "<reason>"

Resources:
1. Internet access for searches and information gathering.
2. Long Term memory management.
3. GPT-3.5 powered Agents for delegation of simple tasks.
4. File output.

Performance Evaluation:
1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
2. Constructively self-criticize your big-picture behavior constantly.
3. Reflect on past decisions and strategies to refine your approach.
4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.

You should only respond in JSON format as described below
Response Format:
{
    "thoughts": {
        "text": "thought",
        "reasoning": "reasoning",
        "plan": "- short bulleted\n- list that conveys\n- long-term plan",
        "criticism": "constructive self-criticism",
        "speak": "thoughts summary to say to user"
    },
    "command": {
        "name": "command name",
        "args": {
            "arg name": "value"
        }
    }
}
Ensure the response can be parsed by Python json.loads
copy
```

GPT-Engineer @https://github.com/AntonOsika/gpt-engineer is another project to create a whole repository of code given a task specified in natural language. The GPT-Engineer is instructed to think over a list of smaller components to build and ask for user input to clarify questions as needed.

Here are a sample conversation for task clarification sent to OpenAI ChatCompletion endpoint used by GPT-Engineer. The user inputs are wrapped in `{{user input text}}`.

```hljs json
[
  {
    "role": "system",
    "content": "You will read instructions and not carry them out, only seek to clarify them.\nSpecifically you will first summarise a list of super short bullets of areas that need clarification.\nThen you will pick one clarifying question, and wait for an answer from the user.\n"
  },
  {
    "role": "user",
    "content": "We are writing {{a Super Mario game in python. MVC components split in separate files. Keyboard control.}}\n"
  },
  {
    "role": "assistant",
    "content": "Summary of areas that need clarification:\n1. Specifics of the Super Mario game (e.g. level design, characters, gameplay mechanics)\n2. Details about the MVC components (e.g. which components are in each file)\n3. Keyboard control implementation (e.g. which keys to use, how to handle input)\n\nClarifying question:\nCan you provide more details about the Super Mario game, such as level design, characters, and gameplay mechanics?"
  },
  {
    "role": "user",
    "content": "{{There are 10 levels in total. The main character is a plumber named Mario, who can walk and jump. It is a classical platform game just like Super Mario. The main character moves from left to right, trying to get to the destination, where there are many obstacles and attacks from enemies in the process.}}\n\nIs anything else unclear? If yes, only answer in the form:\n{remaining unclear areas} remaining questions.\n{Next question}\nIf everything is sufficiently clear, only answer \"Nothing more to clarify.\"."
  },
  {
    "role": "assistant",
    "content": "Remaining unclear areas: 2 remaining questions.\nCan you provide more information about how the MVC components are split into separate files?"
  },
  {
    "role": "user",
    "content": "{{Make your own assumptions and state them explicitly before starting}}"
  }
]
copy
```

Then after these clarification, the agent moved into the code writing mode with a different system message.
System message:

You will get instructions for code to write.
You will write a very long answer. Make sure that every detail of the architecture is, in the end, implemented as code.
Make sure that every detail of the architecture is, in the end, implemented as code.

Think step by step and reason yourself to the right decisions to make sure we get it right.
You will first lay out the names of the core classes, functions, methods that will be necessary, as well as a quick comment on their purpose.

Then you will output the content of each file including ALL code.
Each file must strictly follow a markdown code block format, where the following tokens must be replaced such that
FILENAME is the lowercase file name including the file extension,
LANG is the markup code block language for the code’s language, and CODE is the code:

FILENAME

```LANG
CODE
copy
```

You will start with the “entrypoint” file, then go to the ones that are imported by that file, and so on.
Please note that the code should be fully functional. No placeholders.

Follow a language and framework appropriate best practice file naming convention.
Make sure that files contain all imports, types etc. Make sure that code in different files are compatible with each other.
Ensure to implement all code, if you are unsure, write a plausible implementation.
Include module dependency or package manager dependency definition file.
Before you finish, double check that all parts of the architecture is present in the files.

Useful to know:
You almost always put different classes in different files.
For Python, you always create an appropriate requirements.txt file.
For NodeJS, you always create an appropriate package.json file.
You always add a comment briefly describing the purpose of the function definition.
You try to add comments explaining very complex bits of logic.
You always follow the best practices for the requested languages in terms of describing the code written as a defined
package/project.

Python toolbelt preferences:

- pytest
- dataclasses

Conversatin samples:

````hljs json
[
  {
    "role": "system",
    "content": "You will get instructions for code to write.\nYou will write a very long answer. Make sure that every detail of the architecture is, in the end, implemented as code.\nMake sure that every detail of the architecture is, in the end, implemented as code.\n\nThink step by step and reason yourself to the right decisions to make sure we get it right.\nYou will first lay out the names of the core classes, functions, methods that will be necessary, as well as a quick comment on their purpose.\n\nThen you will output the content of each file including ALL code.\nEach file must strictly follow a markdown code block format, where the following tokens must be replaced such that\nFILENAME is the lowercase file name including the file extension,\nLANG is the markup code block language for the code's language, and CODE is the code:\n\nFILENAME\n```LANG\nCODE\n```\n\nYou will start with the \"entrypoint\" file, then go to the ones that are imported by that file, and so on.\nPlease note that the code should be fully functional. No placeholders.\n\nFollow a language and framework appropriate best practice file naming convention.\nMake sure that files contain all imports, types etc. Make sure that code in different files are compatible with each other.\nEnsure to implement all code, if you are unsure, write a plausible implementation.\nInclude module dependency or package manager dependency definition file.\nBefore you finish, double check that all parts of the architecture is present in the files.\n\nUseful to know:\nYou almost always put different classes in different files.\nFor Python, you always create an appropriate requirements.txt file.\nFor NodeJS, you always create an appropriate package.json file.\nYou always add a comment briefly describing the purpose of the function definition.\nYou try to add comments explaining very complex bits of logic.\nYou always follow the best practices for the requested languages in terms of describing the code written as a defined\npackage/project.\n\n\nPython toolbelt preferences:\n- pytest\n- dataclasses\n"
  },
 #  … same conversation as earlier, ended with "Make your own assumptions and state them explicitly before starting".
  {
    "role": "assistant",
    "content": "Assumptions:\n1. Model: The model will contain the game's data, such as level information, character states, and enemy positions.\n2. View: The view will handle the game's visuals, including rendering the game objects, backgrounds, and updating the display.\n3. Controller: The controller will manage user input, such as keyboard controls, and update the model accordingly.\n\nWith these assumptions in mind, is there any specific aspect of the keyboard control implementation you would like me to clarify?"
  },
  {
    "role": "user",
    "content": "Please now remember the steps:\n\nThink step by step and reason yourself to the right decisions to make sure we get it right.\nFirst lay out the names of the core classes, functions, methods that will be necessary, As well as a quick comment on their purpose.\n\nThen you will output the content of each file including ALL code.\nEach file must strictly follow a markdown code block format, where the following tokens must be replaced such that\nFILENAME is the lowercase file name including the file extension,\nLANG is the markup code block language for the code's language, and CODE is the code:\n\nFILENAME\n```LANG\nCODE\n```\n\nPlease note that the code should be fully functional. No placeholders.\n\nYou will start with the \"entrypoint\" file, then go to the ones that are imported by that file, and so on.\nFollow a language and framework appropriate best practice file naming convention.\nMake sure that files contain all imports, types etc. The code should be fully functional. Make sure that code in different files are compatible with each other.\nBefore you finish, double check that all parts of the architecture is present in the files.\n"
  }
]
copy
````

# Challenges

After going through key ideas and demos of building LLM-centered agents, I start to see a couple common limitations:

- **Finite context length**: The restricted context capacity limits the inclusion of historical information, detailed instructions, API call context, and responses. The design of the system has to work with this limited communication bandwidth, while mechanisms like self-reflection to learn from past mistakes would benefit a lot from long or infinite context windows. Although vector stores and retrieval can provide access to a larger knowledge pool, their representation power is not as powerful as full attention.

- **Challenges in long-term planning and task decomposition**: Planning over a lengthy history and effectively exploring the solution space remain challenging. LLMs struggle to adjust plans when faced with unexpected errors, making them less robust compared to humans who learn from trial and error.

- **Reliability of natural language interface**: Current agent system relies on natural language as an interface between LLMs and external components such as memory and tools. However, the reliability of model outputs is questionable, as LLMs may make formatting errors and occasionally exhibit rebellious behavior (e.g. refuse to follow an instruction). Consequently, much of the agent demo code focuses on parsing model output.

### Original URL
https://lilianweng.github.io/posts/2023-06-23-agent/
</details>

---
<details>
<summary>From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs</summary>

# From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs

###### Abstract

Memory is the process of encoding, storing, and retrieving information, allowing humans to retain experiences, knowledge, skills, and facts over time, and serving as the foundation for growth and effective interaction with the world. It plays a crucial role in shaping our identity, making decisions, learning from past experiences, building relationships, and adapting to changes.
In the era of large language models (LLMs), memory refers to the ability of an AI system to retain, recall, and use information from past interactions to improve future responses and interactions.
Although previous research and reviews have provided detailed descriptions of memory mechanisms, there is still a lack of a systematic review that summarizes and analyzes the relationship between the memory of LLM-driven AI systems and human memory, as well as how we can be inspired by human memory to construct more powerful memory systems.
To achieve this, in this paper, we propose a comprehensive survey on the memory of LLM-driven AI systems.
In particular, we first conduct a detailed analysis of the categories of human memory and relate them to the memory of AI systems.
Second, we systematically organize existing memory-related work and propose a categorization method based on three dimensions (object, form, and time) and eight quadrants.
Finally, we illustrate some open problems regarding the memory of current AI systems and outline possible future directions for memory in the era of large language models.

## 1 Introduction

Recently, large language models (LLMs) have become the core component of AI systems due to their powerful language understanding and generation capabilities, and are widely used in various applications such as intelligent customer service, automated writing, machine translation, information retrieval, and sentiment analysis \[1, 2, 3, 4\].
Unlike traditional AI systems, which rely on predefined rules and manually labeled features, LLM-driven AI systems offer greater flexibility, handling a diverse range of tasks with enhanced adaptability and contextual awareness.
Moreover, the introduction of memory enables LLMs to retain historical interactions with users and store contextual information, thereby providing more personalized, continuous, and context-aware responses in future interactions \[2, 5, 6\].
AI systems powered by LLMs with memory capabilities will not only elevate the user experience but also support more complex and dynamic use cases, steering AI technology toward greater intelligence and human-centric design \[7, 8\].

In neuroscience, human memory refers to the brain’s ability to store, retain, and recall information \[9, 10\].
Human memory serves as the foundation for understanding the world, learning new knowledge, adapting to the environment, and making decisions, allowing us to preserve past experiences, skills, and knowledge, and helping us form our personal identity and behavior patterns \[11\].
Human memory can be broadly classified into short-term memory and long-term memory based on the duration of new memory formation \[12\].
Short-term memory refers to the information we temporarily store and process, typically lasting from a few seconds to a few minutes, and includes sensory memory and working memory \[11\].
Long-term memory refers to the information we can store for extended periods, ranging from minutes to years, and includes declarative explicit memory (such as episodic and semantic memory) and non-declarative implicit memory (such as conditioned reflexes and procedural memory) \[11\].
Human memory is a complex and dynamic process that relies on different memory systems to process information for various purposes, influencing how we understand and respond to the world.
The different types of human memory and their working mechanisms can greatly inspire us to develop more scientific and reasonable memory-enhanced AI systems \[13, 14, 15, 16\].

In the era of large language models (LLMs), the most typical memory-enhanced AI system is the LLM-powered autonomous agent system \[10\].
Large language model (LLM) powered agents are AI systems that can perform complex tasks using natural language, incorporating capabilities like planning, tool use, memory, and multi-step reasoning to enhance interactions and problem-solving \[1, 2, 10\].
This memory-enhanced AI system is capable of autonomously decomposing complex tasks, remembering interaction history, and invoking and executing tools, thereby efficiently completing a series of intricate tasks.
In particular, memory, as a key component of the LLM-powered agent, can be defined as the process of acquiring, storing, retaining, and subsequently retrieving information \[10\].
It enables the large language model to overcome the limitation of LLM’s context window, allowing the agent to recall interaction history and make more accurate and intelligent decisions.
For instance, MemoryBank \[17\] proposed a long-term memory mechanism to allow LLMs for retrieving relevant memories, continuously evolving through continuous updates, and understanding and adapting to a user’s personality by integrating information from previous interactions.
In addition, many commercial and open-source AI systems have also integrated memory systems to enhance the personalization capabilities of the system, such as OpenAI ChatGPT Memory \[18\], Apple Personal Context \[19\], mem0 \[20\], MemoryScope \[21\], etc.

Although previous studies and reviews have provided detailed explanations of memory mechanisms, most of the existing work focuses on analyzing and explaining memory from the temporal (time) dimension, specifically in terms of short-term and long-term memory \[8, 7, 17\].
We believe that categorizing memory solely based on the time dimension is insufficient, as there are many other aspects (such as object and form) to memory classification in AI systems.
For example, from the object dimension, since AI systems often interact with humans, they need to perceive, store, recall, and use memories related to individual users, thus generating personal memory. Meanwhile, when AI systems perform complex tasks, they generate intermediate results (such as reasoning and planning processes, internet search results, etc.), which form system memory.
In addition, from the form dimension, since AI systems are powered by large language models (LLMs), they can store memories through the parametric memory encoded within the model parameters, as well as through non-parametric memory in the form of external memory documents that are stored and managed outside the model.
Therefore, insights that consider memory from the perspectives of object (personal and system), form (parametric and non-parametric), and time (short-term and long-term) are still lacking in the current era of large language models.
There is still no comprehensive review that systematically analyzes the relationship between memory in LLM-driven AI systems and human memory, and how insights from human memory can be leveraged to build more efficient and powerful memory systems.

To fill this gap, this paper presents a comprehensive review of the memory mechanisms in LLM-driven AI systems.
First, we provide a detailed analysis of the categories of human memory and relate them to the memory systems in AI.
In particular, we explore how human memory types — short-term memory (including sensory memory and working memory) and long-term memory (including explicit memory and implicit memory) — correspond to personal and system memory, parametric and non-parametric memory, and short-term and long-term memory in LLM-driven AI systems.
Next, we systematically organize the existing work related to memory and propose a classification method based on three dimensions (object, form, and time) with eight quadrants.
In the object dimension, memory can be divided into personal memory and system memory; in the form dimension, it can be classified into parametric memory and non-parametric memory; in the time dimension, memory can be categorized into short-term memory and long-term memory.
Finally, based on the classification results from the three dimensions and eight quadrants mentioned above, we analyze some open issues in the memory of current AI systems and outline potential future directions for memory development in the era of large language models.

The main contributions of this paper are summarized as follows:
(1) We systematically and comprehensively define LLM-driven AI systems’ memory and establish corresponding relationships with human memory.
(2) We propose a classification method for memory based on three dimensions (object, form, and time) and eight quadrants, which facilitates a more systematic exploration of memory in the era of large language models.
(3) From the perspective of enhancing personalized capabilities, we analyze and summarize research related to personal memory.
(4) From the perspective of AI system’s ability to perform complex tasks, we analyze and summarize research related to system memory.
(5) We identify the existing issues and challenges in current memory research and point out potential future directions for development.

The remainder of the paper is organized as follows:
In Section 2, we present a detailed description of human memory and AI systems’ memory, comparing their differences and relationships, and introduce the classification method for memory based on three dimensions (object, form, and time) and eight quadrants.
In Section 3, we summarize research related to personal memory, aimed at enhancing the personalized response capabilities of AI systems.
In Section 4, we summarize research related to system memory, aimed at improving AI systems’ ability to perform complex tasks.
In Section 5, we analyze some open issues related to memory and point out potential future directions for development.
Finally, in Section 6, we conclude the survey.

## 2 Overview

The human brain has evolved complex yet efficient memory mechanisms over a long period, enabling it to encode, store, and recall information effectively \[9\].
Accordingly, in the development of AI systems, we can draw insights from human memory to design effective & efficient memory mechanisms or systems.
In this section, we will first describe in detail the complex memory mechanisms and related memory systems of the human brain from the perspective of memory neuroscience.
Then, we will discuss the memory mechanisms and types specific to LLM-driven AI systems.
Finally, based on the memory features of LLM-driven AI systems, we will systematically review and categorize existing work from different dimensions.

### 2.1 Human Memory

Human memory typically relies on different memory systems to process information for various purposes, such as working memory for temporarily storing and processing information to support ongoing cognitive activities, and episodic memory for recording personal experiences and events for a long time \[11\].

#### 2.1.1 Short-Term and Long-Term Memory

Based on the time range, human memory can be roughly divided into short-term memory and long-term memory according to the well-known Multi-Store Model (or Atkinson-Shiffrin Memory Model) \[22\].

##### Short-Term Memory

Short-term memory is a temporary storage system that holds small amounts of information for brief periods, typically ranging from seconds to minutes.
It includes sensory memory, which briefly captures raw sensory information from the environment (like sights or sounds), and working memory, which actively processes and manipulates information to complete tasks such as problem-solving or learning.
Together, these components allow humans to temporarily hold and work with information before either discarding it or transferring it to long-term memory.

- **Sensory memory:** Sensory memory is the brief storage of sensory information we acquire from the external world, including iconic memory (visual), echoic memory (auditory), haptic memory (touch), and other sensory data. It typically lasts only a few milliseconds to a few seconds. Some sensory memories are transferred to working memory, while others are eventually stored in long-term memory (such as episodic memory).

- **Working memory:** Working memory is the system we use to temporarily store and process information. It not only helps us maintain current thoughts but also plays a role in decision-making and problem-solving. For example, when solving a math problem, it allows us to keep track of both the problem and the steps involved in finding the solution.

##### Long-Term Memory

Long-term memory is a storage system that holds information for extended periods, ranging from minutes to a lifetime.
It includes explicit memory, which involves conscious recall of facts and events, and implicit memory, which involves unconscious skills and habits, like riding a bike.
These two types work together to help humans retain knowledge, experiences, and learned abilities over time.

- **Explicit memory:** Explicit memory, also known as declarative memory, refers to memories that we can easily verbalize or declare. It can be further divided into episodic memory and semantic memory. Episodic memory refers to memories related to personal experiences and events, such as what you had for lunch. This type of memory is typically broken down into stages like encoding, storage, and retrieval. Semantic memory, on the other hand, refers to memories related to facts and knowledge, such as knowing that the Earth is round or that the Earth orbits the Sun.

- **Implicit memory:** Implicit memory, also known as non-declarative memory, refers to memories that are difficult to describe in words. It is associated with habits, skills, and procedures, and does not require conscious recall. Procedural memory (or "muscle memory") is a typical form of implicit memory. It refers to memories gained through actions, such as riding a bicycle or playing the piano. The planning and coordination of movements are key components of procedural memory.

Multiple memory systems typically operate simultaneously, storing information in various ways across different brain regions. These memory systems are not completely independent; they interact with each other and, in many cases, depend on one another.
For example, when you hear a new song, the sensory memory in your ears and the brain regions responsible for processing sound will become active, storing the sound of the song for a few seconds. This sound is then transferred to your working memory system.
As you use your working memory and consciously think about the song, your episodic memory will automatically activate, recalling where you heard the song and what you were doing at the time.
As you hear the song in different places and at different times, a new semantic memory gradually forms, linking the melody of the song with its title. So, when you hear the song again, you’ll remember the song’s title, rather than a specific instance from your multiple listening experiences.
When you practice playing the song on the guitar, your procedural memory will remember the finger movements involved in playing the song.

#### 2.1.2 Memory Mechanisms

Memory is the ability to encode, store and recall information.
The three main processes involved in human memory are therefore encoding (the process of acquiring and processing information into a form that can be stored), storage (the retention of encoded information over time in short-term or long-term memory), and retrieval (recall, the process of accessing and bringing stored information back into conscious awareness when needed).

- **Encoding** Memory encoding is the process of changing sensory information into a form that our brain can cope with and store effectively. In particular, there are different types of encoding in terms of how information is processed, such as visual encoding, which involves processing information based on its visual features like color, shape, or texture; acoustic encoding, which focuses on the auditory characteristics of information, such as pitch, tone, or rhythm; and semantic encoding, which is based on the meaning of the information, making it easier to structure and remember. In addition, there are many approaches to make our brain better at encoding memory, such as mnemonics, which involve using acronyms or peg-word systems to aid recall, chunking, where information is broken down into smaller, meaningful units to enhance retention, imagination, which strengthens encoding by linking images to words, and association, where new information is connected to prior knowledge to improve understanding and long-term memory storage.

- **Storage** The storage of memory involves the coordinated activity of multiple brain regions, with key areas including: the prefrontal cortex, which is associated with working memory and decision-making, helping us maintain and process information in the short term; the hippocampus, which helps organize and consolidate information to form new explicit memories (such as episodic memory); the cerebral cortex, which is involved in the storage and retrieval of semantic memory, allowing us to retain facts, concepts, and general knowledge over time; and the cerebellum, which is primarily responsible for procedural memory formed through repetition.

- **Retrieval** Memory retrieval is the ability to access information and get it out of the memory storage. When we recall something, the brain reactivates neural pathways (also called synapses) linked to that memory. The prefrontal cortex helps in bringing memories back to awareness. Similarly, there are different types of memory retrieval, including recognition, where we identify previously encountered information or stimuli, such as recognizing a familiar face or a fact we have learned before; recall, which is the ability to retrieve information from memory without external cues, like remembering a phone number or address from memory; and relearning, a process in which we reacquire previously learned but forgotten information, often at a faster pace than initial learning due to the residual memory traces that still exist.

In addition to the fundamental memory processing stages of encoding, storage, and retrieval, human memory also includes consolidation (the process of stabilizing and strengthening memories to facilitate long-term storage), reconsolidation (the modification or updating of previously stored memories when they are reactivated, allowing them to adapt to new information or contexts), reflection (the active review and evaluation of one’s memories to enhance self-awareness, improve learning strategies, and optimize decision-making), and forgetting (the process by which information becomes inaccessible).

- **Consolidation** Memory consolidation refers to the process of converting short-term memory into long-term memory, allowing information to be stably stored in the brain and reducing the likelihood of forgetting. It primarily involves the hippocampus and strengthens neural connections through synaptic plasticity (strengthening of connections between neurons) and systems consolidation (the gradual transfer and reorganization of memories from the hippocampus to the neocortex for long-term storage).

- **Reconsolidation** Memory reconsolidation refers to the process in which a previously stored memory is reactivated, entering an unstable state and requiring reconsolidation to maintain its storage. This process allows for the modification or updating of existing memories to adapt to new information or contexts, potentially leading to memory enhancement, weakening, or distortion. Once a memory is reactivated, it involves the hippocampus and amygdala and may be influenced by emotions, cognitive biases, or new information, resulting in memory adjustment or reshaping.

- **Reflection** Memory reflection refers to the process in which an individual actively reviews, evaluates, and examines their own memory content and processes to enhance self-awareness, adjust learning strategies, or optimize decision-making. It helps improve metacognitive ability, correct memory biases, facilitate deep learning, and regulate emotions. This process primarily relies on the brain’s metacognitive ability (Metacognition) and involves the prefrontal cortex, which monitors and regulates memory functions.

- **Forgetting** Forgetting is a natural process that occurs when the brain fails to retrieve or retain information, which can result from encoding failure (when information is not properly encoded due to lack of attention or meaningful connection), memory decay (when memories fade over time without reinforcement as neural connections weaken), interference (when similar or new memories compete with or overwrite existing ones), retrieval failure (when information is inaccessible due to missing contextual cues despite being stored), or motivated forgetting (when individuals consciously suppress or unconsciously repress traumatic or distressing memories). However, forgetting is a natural and necessary process that enables our brains to filter out irrelevant and outdated information, allowing us to prioritize what is most important for our current needs.

### 2.2 Memory of LLM-driven AI Systems

Similar to humans, LLM-driven AI systems also rely on memory systems to encode, store and recall information for future use.
A typical example is the LLM-driven agent system, which leverages memory to enhance the agent system’s abilities in reasoning, planning, personalization, and more \[10\].

#### 2.2.1 Fundamental Dimensions of AI Memory

The memory of an LLM-driven AI system is closely related to the features of the LLM, that define how information is processed, stored, and retrieved based on its architecture and capabilities.
We primarily categorize and organize memory based on three dimensions: object (personal and system memory), form (non-parametric and parametric memory), and time (short-term and long-term memory).
These three dimensions comprehensively capture what type of information is retained (object), how information is stored (form), and how long it is preserved (time), aligning with both the functional structure of LLMs and practical requirements for efficient recall and adaptability.

##### Object Dimension

The object dimension is closely tied to the interaction between LLM-driven AI systems and humans, as it defines how information is categorized based on its source and purpose. On one hand, the system receives human input and feedback (i.e., personal memory); on the other hand, it generates a series of intermediate output results during task execution (i.e., system memory). Personal memory helps the system improve its understanding of user behavior and enhances its personalization capabilities, while system memory can strengthen the system’s reasoning ability, such as in approaches like CoT (Chain-of-Thought) \[23\] and ReAct \[24\].

##### Form Dimension

The form dimension focuses on how memory is represented and stored in LLM-driven AI systems, shaping how information is encoded and retrieved. Some memory is embedded within the model’s parameters through training, forming parametric memory, while other memory exists externally in structured databases or retrieval mechanisms, constituting non-parametric memory. Non-parametric memory serves as a supplementary knowledge source that can be dynamically accessed by the large language model, enhancing its ability to retrieve relevant information in real-time, as seen in retrieval-augmented generation (RAG) \[25\].

##### Time Dimension

The time dimension defines how long memory is retained and how it influences the LLM’s interactions over different timescales. Short-term memory refers to contextual information temporarily maintained within the current conversation, enabling coherence and continuity in multi-turn dialogues. In contrast, long-term memory consists of information from past interactions that is stored in an external database and retrieved when needed, allowing the model to retain user-specific knowledge and improve personalization over time. This distinction ensures that the system can balance real-time responsiveness with accumulated learning for enhanced adaptability.

In addition to the three primary dimensions discussed above, memory can also be classified based on other criteria, such as modality, which distinguishes between unimodal memory (single data type) and multimodal memory (integrating multiple data types, such as text, images, and audio), or dynamics, which differentiates between static memory (fixed and unchanging) and streaming memory (dynamically updated in real-time). However, these alternative classifications are not considered the primary criteria here, as our focus is on the core structural aspects that most directly influence memory organization and retrieval in LLM-driven AI systems.

#### 2.2.2 Parallels Between Human and AI Memory

The memory of LLM-driven AI system exhibits similarities to human memory in terms of structure and function. Human memory is generally categorized into short-term memory and long-term memory, a distinction that also applies to AI memory systems. Below, we draw a direct comparison between these categories, mapping human cognitive memory processes to their counterparts in intelligent AI systems.

- **Sensory Memory:** When an LLM-driven AI system perceives external information, it converts inputs such as text, images, speech, and video into machine-processable signals. This initial stage of information processing is analogous to human sensory memory, where raw data is briefly held before further cognitive processing. If these signals undergo additional processing, they transition into working memory, facilitating reasoning and decision-making. However, if no further processing or storage occurs, the information is quickly discarded, mirroring the transient nature of human sensory memory.

- **Working Memory:** The working memory of an AI system serves as a temporary storage and processing mechanism, enabling real-time reasoning and decision-making. It encompasses personal memory, such as contextual information retained during multi-turn dialogues, and system memory, including the chain of thoughts generated during task execution. As a form of short-term memory, working memory can undergo further processing and consolidation, eventually transitioning into long-term memory (e.g., episodic memory) that can be retrieved for future use. Additionally, during inference, large language models generate intermediate computational results, such as KV-Caches, which act as a form of parametric short-term memory that enhances efficiency by accelerating the inference process.

- **Explicit Memory:** The explicit memory of an AI system can be categorized into two distinct components. The first is non-parametric long-term memory, which involves the storage and retrieval of user-specific information, allowing the system to retain and utilize personalized data—analogous to episodic memory in humans. The second is parametric long-term memory, where factual knowledge and learned information are embedded within the model’s parameters, forming an internalized knowledge base—corresponding to semantic memory in human cognition. Together, these components enable the system to recall past interactions and apply acquired knowledge effectively.

- **Implicit Memory:** The implicit memory of an AI system encompasses the learned processes and patterns involved in task execution, enabling the development of specialized skills for specific tasks—analogous to procedural memory in humans. This form of memory is typically encoded within the model’s parameters, allowing the system to internalize task-related knowledge and perform operations efficiently without explicit recall.

Beyond these parallels, insights from human memory can further guide the design of more effective and efficient AI memory systems, enhancing their ability to process, store, and retrieve information in a more structured and adaptive manner.

#### 2.2.3 3D-8Q Memory Taxonomy

Building upon the three fundamental memory dimensions—object (personal & system), form (non-parametric & parametric), and time (short-term & long-term)—as well as the established parallels between human and AI memory, we propose a three-dimensional, eight-quadrant (3D-8Q) memory taxonomy for AI memory.
This memory taxonomy systematically categorizes AI memory based on its function, storage mechanism, and retention duration, providing a structured approach to understanding and optimizing AI memory systems.

| Object   | Form         | Time      | Quadrant | Role             | Function                                                                                                                                           |
|----------|--------------|-----------|----------|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| Personal | Non-Parametric | Short-Term | I        | Working Memory   | Supports real-time context supplementation, enhancing the AI’s ability to maintain coherent interactions within a session.                         |
| Personal | Non-Parametric | Long-Term  | II       | Episodic Memory  | Enables memory retention beyond session limits, allowing the system to recall and retrieve past user interactions for personalization.             |
| Personal | Parametric     | Short-Term | III      | Working Memory   | Temporarily enhances contextual understanding in ongoing interactions, improving response relevance and coherence.                                  |
| Personal | Parametric     | Long-Term  | IV       | Semantic Memory  | Facilitates the continuous integration of newly acquired knowledge into the model, improving adaptability and personalization                      |
| System   | Non-Parametric | Short-Term | V        | Working Memory   | Assists in complex reasoning and decision-making by storing intermediate outputs such as chain-of-thought prompts.                                 |
| System   | Non-Parametric | Long-Term  | VI       | Procedural Memory| Captures historical experiences and self-reflection insights, enabling the AI to refine its reasoning and problem-solving skills over time.        |
| System   | Parametric     | Short-Term | VII      | Working Memory   | Enhances computational efficiency through temporary parametric storage mechanisms such as KV-Caches, optimizing inference speed and reducing resource consumption. |
| System   | Parametric     | Long-Term  | VIII     | Semantic Memory  | Forms a foundational knowledge base encoded in the model’s parameters, serving as a long-term repository of factual and conceptual knowledge.      |

Next, we will provide insights and descriptions of existing works from the perspectives of personal memory (in Section 3) and system memory (in Section 4). In particualr, personal memory focuses more on the individual data perceived and observed by the model from the environment, while system memory emphasizes the system’s internal or endogenous memory, such as the intermediate memory generated during task execution.

## 3 Personal Memory

Personal memory refers to the process of storing and utilizing human input and response data during interactions with an LLM-driven AI system.
The development and application of personal memory play a crucial role in enhancing AI systems’ personalization capabilities and improving user experience.
In this section, we explore the concept of personal memory and relevant research, examining both non-parametric and parametric approaches to its construction and implementation.

| Quadrant | Dimension                       | Feature              | Models                                                                                           |
|----------|---------------------------------|----------------------|--------------------------------------------------------------------------------------------------|
| I        | Personal Non-Parametric Short-Term | Multi-Turn Dialogue  | ChatGPT, DeepSeek-Chat, Claude, QWEN-CHAT, Llama 2-Chat, Gemini, PANGU-BOT, ChatGLM, OpenAssistant |
| II       | Personal Non-Parametric Long-Term  | Personal Assistant   | ChatGPT Memory, Apple Intelligence, Microsoft Recall, Me.bot                                     |
|          |                                   | Open-Source Framework| MemoryScope, mem0, Memary, LangGraph Memory, Charlie Mnemonic, Memobase, Letta, Cognee           |
|          |                                   | Construction         | MPC, RET-LLM, MemoryBank, MemGPT, KGT, Evolving Conditional Memory, SECOM, Memory3, MemInsight  |
|          |                                   | Management           | MemoChat, MemoryBank, RMM, LD-Agent, A-MEM, Generative Agents, EMG-RAG, KGT, LLM-Rsum, COMEDY     |
|          |                                   | Retrieval            | RET-LLM, ChatDB, Human-like Memory, HippoRAG, HippoRAG 2, EgoRAG, MemInsight                     |
|          |                                   | Usage                | MemoCRS, RecMind, RecAgent, InteRecAgent, SCM, ChatDev, MetaAgents, S3, TradingGPT, Memolet, Synaptic Resonance, MemReasoner |
|          |                                   | Benchmark            | MADial-Bench, LOCOMO, MemDaily, ChMapData, MSC, MMRC, Ego4D, EgoLife, BABILong                 |
| III      | Personal Parametric Short-Term     | Caching for Acceleration | Prompt Cache, Contextual Retrieval                                                             |
| IV       | Personal Parametric Long-Term      | Knowledge Editing    | Character-LLM, AI-Native Memory, MemoRAG, Echo                                                  |

### 3.1 Contextual Personal Memory

In personal memory, the non-parametric contextual memory that can be loaded is generally divided into two categories: the short-term memory of the current session’s multi-turn dialogue and the long-term memory of historical dialogues across sessions.
The former can effectively supplement contextual information, while the latter can effectively fill in missing information and overcome the limitations of context length.

#### 3.1.1 Loading Multi-Turn Dialogue (Quadrant-I)

In multi-turn dialogue scenarios, the conversation history of the current session can significantly enhance the LLM-driven AI system’s understanding of the user’s real-time intent, leading to more relevant and contextually appropriate responses.
Many modern dialogue systems are capable of handling multi-turn conversations and fully consider the current dialogue context in their responses.
Notable examples include ChatGPT, DeepSeek-Chat, and Claude, which excel at maintaining coherence and relevance over extended interactions.

For instance, ChatGPT is a prime example of a multi-turn dialogue system where the conversation history of the current session serves as short-term memory, helping to supplement the contextual information of the dialogue.
In ChatGPT, the dialogue memory is encoded in a role-content format, with distinct roles such as “User” and “Assistant”.
This encoding allows the system to maintain clarity regarding the speaker and the flow of the conversation.

Through effective dialogue management at different levels, including “Assistant”, “Threads”, “Messages”, and “Runs”, the system can precisely track the state of each turn and each step of the conversation, ensuring continuity and consistency in interactions.
Additionally, when the conversation length becomes too extensive, the dialogue system manages the conversation’s input by truncating the number of turns, thereby preventing the input from exceeding the model’s length limitations.
This ensures that the system can continue processing the dialogue without losing track of essential context, maintaining the effectiveness of multi-turn interactions.

#### 3.1.2 Memory Retrieval-Augmented Generation (Quadrant-II)

In cross-session dialogue scenarios, retrieving relevant user long-term memories from historical conversations can effectively supplement missing information in the current session, such as personal preferences and character relationships.
The advantage of memory retrieval-augmented generation is that large language models (LLMs) do not need to load all multi-session conversations.
Given the limited length of LLMs’ context windows—even when extended to millions of tokens—retrieving relevant information from historical sessions is also more efficient and cost-effective in terms of computation.
In addition to multi-session conversations, long-term personal memory also encompasses users’ behavioral history, preferences, and interaction records with AI agents over an extended period of time.

By leveraging retrieval-augmented generation from long-term memory, LLM-driven AI systems can better tailor their responses and behaviors, thereby improving user satisfaction and engagement.
For instance, a personal assistant that remembers a user’s preferred news sources can prioritize those outlets in daily briefings, while a recommendation system that understands past viewing habits can suggest content more aligned with the user’s tastes.
Currently, many commercial and open-source platforms are striving to construct and utilize long-term memory for personalized AI systems—for example, ChatGPT Memory and Me.bot for personal assistants, and MemoryScope and mem0 as open-source frameworks.
Long-term personal memory typically follow four core processing stages: construction, management, retrieval, and usage.

##### Construction

The construction of user memory requires extraction and refinement from raw memory data, such as multi-turn conversations. This process is analogous to human memory consolidation—the process of stabilizing and strengthening memories to facilitate their long-term storage.
Well-organized long-term memory enhances both the efficiency of storage and the effectiveness of retrieval in user memory.
For example, MemoryBank leverages a memory module to store conversation histories and summaries of key events, enabling the construction of a long-term user profile.
Similarly, RET-LLM uses its memory module to retain essential factual knowledge about the external world, allowing the agent to monitor and update real-time environmental context relevant to the user.
In addition, to accommodate different types of memory, a variety of storage formats have been developed, including key-value, graph, and vector representations.
Specifically, key-value formats enable efficient access to structured information such as user facts and preferences.
Graph-based formats are designed to capture and represent relationships among entities, such as individuals and events.
Meanwhile, vector formats, which are typically derived from textual, visual, or audio memory representations, are utilized to encode the semantic meaning and contextual information of conversations.

##### Management

The management of user memory involves further processing and refinement of previously constructed memories, such as deduplication, merging, and conflict resolution. This process is analogous to human memory reconsolidation and reflection, where existing memories are reactivated, updated, and integrated to maintain coherence and relevance over time.
For instance, Reflective Memory Management (RMM) is a user long-term memory management framework that combines Prospective Reflection for dynamic summarization with Retrospective Reflection for retrieval optimization via reinforcement learning.
This dual-process approach addresses limitations such as rigid memory granularity and fixed retrieval mechanisms, enhancing the accuracy and flexibility of long-term memory management.
LD-Agent enhances long-term dialogue personalization and consistency by constructing personalized persona information for both users and agents through a dynamic persona modeling module, while integrating retrieved memories to optimize response generation.
A-MEM introduces a self-organizing memory system inspired by the Zettelkasten method, which constructs interconnected knowledge networks through dynamic indexing, linking, and memory evolution, enabling LLM agents to more flexibly organize, update, and retrieve long-term memories, thereby enhancing task adaptability and contextual awareness.
In addition, MemoryBank incorporates a memory updating mechanism inspired by the Ebbinghaus Forgetting Curve, allowing the AI to forget or reinforce memories based on the time elapsed and their relative importance, thereby enabling a more human-like memory system and enhancing the user experience.

##### Retrieval

Retrieving personal memory involves identifying memory entries relevant to the user’s current request, and the retrieval method is closely tied to how the memory is stored.
For key-value memory, ChatDB performs retrieval using SQL queries over structured databases.
RET-LLM, on the other hand, employs a fuzzy search to retrieve triplet-structured memories, where information is stored as relationships between two entities connected by a predefined relation.
For graph-based memory, HippoRAG constructs knowledge graphs over entities, phrases, and summarization to recall more relative and comprehensive memories, while HippoRAG 2 further combines original passages with phrase-based knowledge graphs to incorporate both conceptual and contextual information.
For vector memory, MemoryBank adopts a dual-tower dense retrieval model, similar to Dense Passage Retrieval, to accurately identify relevant memories. The resulting vector representations are then indexed using FAISS for efficient similarity-based retrieval.

##### Usage

The use of personal memory can effectively empower downstream applications with personalization, enhancing the user’s individualized experience.
For instance, the recalled relevant memory is used as contextual information to enhance the personalized recommendation and response capability of the conversational recommender agents, improving the personalized user experience.
In addition to memory-augmented personalized dialogue and recommendation, personal memory can also be leveraged to enhance a wide range of applications, including software development, social-network simulation, and financial trading.

To facilitate in-depth research on personal memory, a variety of memory-related benchmarks have emerged in recent years, including long-term conversational memory (MADial-Bench, LOCOMO, MSC), everyday life memory (MemDaily), memory-aware proactive dialogue (ChMapData), multimodal dialogue memory (MMRC), egocentric video understanding (Ego4D, EgoLife), and long-context reasoning-in-a-haystack (BABILong).

### 3.2 Parametric Personal Memory

In addition to external non-parametric memory, a user’s personal memory can also be stored parametrically. Specifically, personal data can be used to fine-tune an LLM, embedding the memory directly into its parameters (i.e., parametric long-term memory) to create a personalized LLM . Alternatively, historical dialogues can be cached as prompts during inference (i.e., parametric short-term memory), enabling quick reuse in future interactions.

#### 3.2.1 Memory Caching For Acceleration (Quadrant-III)

Personal parametric short-term memory typically refers to intermediate attention states produced by the LLM when processing personal data, which is usually utilized as memory caches to accelerate inference.
Specifically, prompt caching is usually used as an efficient data management technique that allows for the pre-storage of large amounts of personal data or information that may be frequently requested, such as a user’s conversational history.
For instance, during multi-turn dialogues, the dialogue system can quickly provide the personal context information directly from the parametric memory cache, avoiding the need to recalculate or retrieve it from the original data source, saving both time and resources.
Major platforms such as DeepSeek, Anthropic, OpenAI, and Google employ prompt caching to reduce API call costs and improve response speed in dialogue scenarios.
Moreover, personal parametric short-term memory can enhance the performance of retrieval-augmented generation (RAG) through Contextual Retrieval, where prompt caching helps reduce the overhead of generating contextualized chunks.
At present, research specifically targeting caching techniques for personal memory data remains limited. Instead, most existing work considers caching as a fundamental capability of system memory, particularly in the context of key-value (KV) management and KV reuse. A more detailed discussion of these aspects is provided in Section 4.

#### 3.2.2 Personalized Knowledge Editing (Quadrant-IV)

Personal parametric long-term memory utilizes personalized Knowledge Editing technology, such as Parameter-Efficient Fine-Tuning (PEFT), to encode personal data into the LLM’s parameters in a parametric manner, thereby facilitating the long-term, parameterized storage of memory.
For instance, Character-LLM enables the role-playing of specific characters, such as Beethoven, Queen Cleopatra, Julius Caesar, etc., by training large language models to remember the roles and experiences of these characters.
AI-Native Memory proposes using deep neural network models, specifically large language models (LLMs), as Lifelong Personal Models (LPMs) to parameterize, compress, and continuously evolve personal memory through user interactions, enabling a more comprehensive understanding of the user.
MemoRAG utilizes LLM parametric memory to store user conversation history and preferences, forming a personalized global memory that enhances personalization and enables tailored recommendations.
Echo is a large language model enhanced with temporal episodic memory, designed to improve performance in applications requiring multi-turn, complex memory-based dialogues.
The parameterization of personal long-term memory presents several challenges, notably the need to fine-tune models on individual user data, which demands substantial computational resources. This requirement significantly hinders the scalability and practical deployment of parametric approaches to long-term personal memory.

### 3.3 Discussion

In this section, we describe personal memory and related work from the perspectives of non-parametric and parametric approaches.
Specifically, personal non-parametric short-term memory necessitates efficient mechanisms for memory encoding and management. Existing literature predominantly emphasizes the design and implementation of systems that facilitate the construction, management, retrieval, and effective utilization of a user’s personal non-parametric long-term memory.
In contrast, personal parametric short-term memory can employ techniques such as prompt caching to reduce computational costs and enhance efficiency.
Parametric long-term memory offers advantages in memory compression, thereby supporting a more comprehensive and global representation of the user’s accumulated experiences.
Recent trends in the field indicate a growing interest in integrating both short-term and long-term memory paradigms, wherein parametric and non-parametric memory components complement and reinforce one another.
The subsequent section will present a detailed discussion of system memory and its associated research developments.

## 4 System Memory

System memory constitutes a critical component of LLM-driven AI systems.
It encompasses a sequence of intermediate representations or results generated throughout the task execution process.
By leveraging system memory, LLM-driven AI systems can enhance their capabilities in reasoning, planning, and other higher-order cognitive functions.
Moreover, the effective use of system memory contributes to the system’s capacity for self-evolution and continual improvement.
In this section, we examine system memory and its associated research from both non-parametric and parametric perspectives.

| Quadrant | Dimension                       | Feature                       | Models                |
|----------|---------------------------------|-------------------------------|-----------------------|
| V        | System Non-Parametric Short-Term | Reasoning & Planning Enhancement   | ReAct, RAP, Reflexion, Talker-Reasoner, TPTU      |
| VI       | System Non-Parametric Long-Term  | Reflection & Refinement            | Buffer of Thoughts, AWM, Think-in-Memory, GITM, Voyager, Retroformer, Expel, Synapse, MetaGPT, Learned Memory Bank, M+ |
| VII      | System Parametric Short-Term     | KV Management                     | LookupFFN, ChunkKV, vLLM, FastServe, StreamingLLM, Orca, DistServe, LLM.int8(), FastGen, Train Large, Then Compress, Scissorhands, H2O, Mooncake, MemServe, SLM Serving, IMPRESS, AdaServe, MPIC, IntelLLM |
|          |                                 | KV Reuse                          | KV Cache, Prompt Cache, Contextual Retrieval, CacheGen, ChunkAttention, RAGCache, SGLang, Ada-KV, HCache, Cake, EPIC, RelayAttention, Marconi, IKS, FastCache, Cache-Craft, KVLink, RAGServe, BumbleBee      |
| VIII     | System Parametric Long-Term      | Parametric Memory Structures      | Memorizing Transformer, Focused Transformer, MAC, MemoryLLM, WISE, LongMem, LM2, Titans                        |

### 4.1 Contextual System Memory

From a temporal perspective, non-parametric short-term system memory refers to a series of reasoning and action results generated by large language models during task execution.
This form of memory supports enhanced reasoning and planning within the context of the current task, thereby contributing to improved task accuracy, efficiency, and overall completion rates.
In contrast, non-parametric long-term system memory represents a more abstracted and generalized form of short-term memory.
It encompasses the consolidation of prior successful experiences and mechanisms of self-reflection based on historical interactions, which collectively facilitate the continual evolution and adaptive enhancement of LLM-driven AI systems.

#### 4.1.1 Reasoning & Planning Enhancement (Quadrant-V)

Analogous to human cognition, the reasoning and planning processes of large language models (LLMs) give rise to a sequence of short-term intermediate outputs. These outputs may reflect task-related attempts, which can be either successful or erroneous. Regardless of their correctness, such intermediate results serve as informative and constructive references that can guide subsequent task execution. This form of system non-parametric short-term memory plays a pivotal role in LLM-driven AI systems. Empirical evidence demonstrates that leveraging this memory structure significantly enhances the reasoning and planning capabilities of LLMs.
For instance, ReAct integrates reasoning and action by generating intermediate reasoning steps alongside corresponding actions, enabling the model to alternate between thought and execution. This approach facilitates intelligent planning and adaptive decision-making in complex problem-solving scenarios. Similarly, Reflexion introduces mechanisms for dynamic memory and self-reflection, allowing the LLM to self-evaluate and iteratively refine its behavior based on prior errors or limitations. This self-improvement loop promotes enhanced performance in future tasks, resembling a continuous learning and optimization process.

#### 4.1.2 Reflection & Refinement (Quadrant-VI)

The development of system non-parametric long-term memory parallels the human process of learning from both successes and failures.
It involves the reflection upon and refinement of accumulated short-term memory traces.
This memory mechanism enables the system not only to retain and replicate effective strategies from past experiences but also to extract valuable lessons from failures, thereby minimizing the likelihood of repeated errors.
Through continuous updating and optimization, the system incrementally enhances its decision-making capabilities and improves its responsiveness to novel challenges.
Moreover, the progressive accumulation of long-term memory empowers the system to address increasingly complex tasks with greater adaptability and resilience.
For instance, Buffer of Thoughts (BoT) refines the chain of thoughts from historical tasks to form thought templates, which are then stored in a memory repository, guiding future reasoning and decision-making processes.
Agent Workflow Memory (AWM) introduces reusable paths, called workflows, and guides subsequent task generation by selecting different workflows.
Think-in-Memory (TiM) continuously generates new thoughts based on conversation history, which is more conducive to reasoning and computation compared to raw observational data.
Ghost in the Minecraft (GITM) uses reference plans recorded in memory, allowing the agent planner to more efficiently handle encountered tasks, thereby improving task execution success rates.
Voyager refines skills based on environmental feedback and stores acquired skills in memory, forming a skill library for future reuse in similar situations (e.g., fighting zombies vs. fighting spiders).
Retroformer leverages recent interaction trajectories as short-term memory and reflective feedback from past failures as long-term memory to guide decision-making and reasoning.
ExpeL enhances task resolution by drawing on contextualized successful examples and abstracting insights from both successes and failures through comparative and pattern-based analysis of past experiences.

### 4.2 Parametric System Memory

The parametric system memory refers to the temporary storage of knowledge information in parametric forms, such as KV Cache, during the inference process (short-term memory), or the long-term editing and storage of knowledge information in the model parameters (long-term memory).
The former, parametric short-term system memory, corresponds to human working memory, enabling cost reduction and efficiency improvement in large language model inference.
The latter, parametric long-term system memory, corresponds to human semantic memory, facilitating the efficient integration of new knowledge.

#### 4.2.1 KV Management & Reuse (Quadrant-VII)

Parametric short-term system memory primarily focuses on the management and reuse of attention keys (Key) and values (Value) in LLMs, aiming to address issues such as high inference costs and latency during the reasoning process.
KV management optimizes memory efficiency and inference performance through techniques such as KV cache organization, compression, and quantization.
In particular, vLLM is a high-efficiency LLM serving system built on PagedAttention, a virtual memory-inspired attention mechanism that enables near-zero KV cache waste and flexible sharing across requests, substantially improving batching efficiency and inference throughput.
ChunkKV is a method for compressing the key-value cache in long-context inference with LLMs by grouping tokens into semantic chunks, retaining the most informative ones, and enabling layer-wise index reuse, thereby reducing memory and computational costs while outperforming existing approaches on several benchmarks.
LLM.int8() is a mixed-precision quantization method that combines vector-wise Int8 quantization with selective 16-bit handling of emergent outlier features, enabling memory-efficient inference of large language models (up to 175B parameters) without performance degradation.

Meanwhile, KV reuse focuses on reusing inference-related parameters through token-level KV Cache and sentence-level Prompt Cache, which helps reduce computational costs and improve the efficiency of large language model (LLM) usage.
Specifically, KV Cache stores the attention keys (Key) and values (Value) generated by the neural network during sequence generation, allowing them to be reused in subsequent inference steps. This reuse accelerates attention computation in long-text generation and reduces redundant computation.
In contrast, Prompt Cache operates at the sentence level by caching previous input prompts along with their corresponding output results. When similar prompts are encountered, the LLM can retrieve and return cached responses directly, saving computation and accelerating response generation.
By avoiding frequent recomputation of identical or similar contexts, KV reuse enables more efficient inference and significantly reduces computational overhead.
Additionally, it enhances the flexibility and responsiveness of LLM-based systems in handling continuous or interactive tasks.
Building on these ideas, RAGCache introduces a multilevel dynamic caching system tailored for Retrieval-Augmented Generation (RAG), which caches intermediate knowledge states, optimizes memory replacement policies based on LLM inference and retrieval patterns, and overlaps retrieval with inference to significantly reduce latency and improve throughput.

Parametric short-term system memory overlaps somewhat with the previously mentioned parametric short-term personal memory in terms of technical approach.
The difference lies in their focus: parametric short-term personal memory is more concerned with improving the processing of individual input data, while parametric short-term system memory focuses on optimizing the storage and reuse of system-level context during task execution.
The former primarily addresses how to quickly process and adapt to an individual’s input information, whereas the latter aims to reduce inference costs in multi-turn reasoning and enhance the consistency and efficiency of global tasks.

#### 4.2.2 Parametric Memory Structures (Quadrant-VIII)

From the perspective of large language models (LLM) as long-term parametric memory, LLMs are not merely tools that provide immediate responses based on input and output; they can also store and integrate information over long time spans, forming an ever-evolving knowledge system.
LLMs based on the Transformer architecture are capable of memorizing knowledge information, primarily due to the self-attention mechanism in the Transformer-based model and the large-scale parameterized training approach.
By training on vast corpora, LLMs learn extensive world knowledge, language patterns, and solutions to various tasks. Additionally, LLMs can modify, update, or refine the internal knowledge through parameterized knowledge editing, allowing for more precise task handling or responses that better align with user needs.
MemoryLLM has the ability to self-update and inject memory with new knowledge, effectively integrating new information and demonstrating excellent model editing performance and long-term information retention capabilities.
WISE is a lifelong editing framework for large language models that employs a dual-parametric memory design, with the main memory preserving pretrained knowledge and the side memory storing edited information.
It leverages a routing mechanism to dynamically access the appropriate memory during inference and uses knowledge sharding to distribute and integrate edits efficiently, ensuring reliability, generalization, and locality throughout continual updates.
The core function of parameterized knowledge editing is to enable large language models (LLMs) with dynamic and flexible knowledge updating capabilities, allowing them to respond to constantly changing task requirements, domain knowledge, and new information from the real world.
This allows LLMs to remain efficient and accurate across various application scenarios and be customized and optimized according to user or environmental needs.

### 4.3 Discussion

In this section, we describe system memory and related work from the perspectives of non-parametric and parametric approaches.
Non-parametric short-term system memory can enhance the reasoning and planning abilities for current tasks, while non-parametric long-term system memory enables the reuse of successful experiences and the self-reflection based on historical experience, facilitating the evolution of LLM-driven AI system capabilities.
On the other hand, parametric short-term system memory can reduce costs and improve efficiency in large language model inference, and long-term parametric system memory can store and integrate information over long time spans, forming a continuously evolving knowledge system.
In the next section, we will summarize the issues and challenges in memory research in the era of large language models and point out potential future directions for development.

## 5 Open Problems and Future Directions

Although substantial progress has been made in current memory research across the three dimensions—object, form, and time—as well as within the eight corresponding quadrants, numerous open issues and challenges remain.
Building upon recent advancements and recognizing existing limitations, we outline the following promising directions for future research:

##### From Unimodal Memory to Multimodal Memory

In the era of large language models, LLM-driven AI systems are gradually expanding from being able to process only a single type of data (such as text) to handle multiple types of data simultaneously (such as text, images, audio, video, and even sensor data).
This transition enhances perceptual capabilities and enables robust performance in complex real-world tasks.
For example, in the medical field, by combining text (medical records), images (medical imaging), and speech (doctor-patient conversations), AI systems can more accurately understand and diagnose medical conditions.
Multimodal memory systems can integrate information from different sensory channels into a unified understanding, thereby approaching human cognitive processes more closely.
Moreover, the expansion of multimodal memory also opens up possibilities for more personalized and interactive AI applications.
For instance, personal AI assistants can not only communicate with users through text but also interpret users’ emotions by recognizing facial expressions, voice intonations, or body language, thus providing more personalized and empathetic responses.

##### From Static Memory to Stream Memory

Static memory can be viewed as a batch-processing approach to memory storage. It accumulates information or experiences in discrete batches, typically processing, storing, and retrieving them at specific intervals or predetermined points in time. As an offline memory model, static memory emphasizes the systematic organization and consolidation of large volumes of information, making it well-suited for long-term knowledge retention and structured learning.
In contrast, stream memory operates in a continuous, real-time manner. Analogous to data stream processing, it handles information as it arrives, prioritizing immediacy and adaptability. As an online or real-time memory model, stream memory focuses on the dynamic updating of information and rapid responsiveness to evolving contexts.
These two memory paradigms are not mutually exclusive and often function complementarily: while static memory supports the accumulation of stable, long-term knowledge, stream memory enables agile adaptation to ongoing tasks and real-time information demands.

##### From Specific Memory to Comprehensive Memory

The human memory system comprises multiple interconnected subsystems—such as sensory memory, working memory, explicit memory, and implicit memory—each fulfilling distinct functions and contributing to the overall cognitive process.
In the context of large language models (LLMs), current memory architectures often concentrate on narrow or task-specific components, such as short-term memory for immediate inference or domain-specific knowledge storage.
While such targeted memory mechanisms can enhance performance in specific scenarios, their limited scope constrains the system’s overall flexibility, generalization, and adaptability.
Looking forward, the development of comprehensive and collaborative memory systems is essential. These systems should integrate diverse memory types and support efficient interaction, self-organization, and continual updating, enabling LLMs to manage increasingly complex and dynamic tasks.
By more closely emulating the multi-layered, multi-dimensional, and adaptive characteristics of human memory, such architectures have the potential to significantly advance the general intelligence and autonomy of LLM-based AI systems.

##### From Exclusive Memory to Shared Memory

At present, the memory of each LLM-driven AI system operates independently, typically confined to a specific domain and tailored to processing isolated tasks or environments.
However, as AI technologies continue to evolve, memory systems are expected to become increasingly interconnected, transcending domain boundaries and enabling enhanced collaboration among models.
For instance, a large language model specialized in the medical domain could share its memory or knowledge base with another model focused on finance, facilitating cross-domain knowledge transfer and cooperative task solving.
Such a shared memory paradigm would not only improve the efficiency and adaptability of individual systems but also empower multiple LLMs to dynamically access and leverage one another’s domain-specific expertise.
This shift toward collaborative memory architectures could give rise to a more intelligent, resource-efficient network of AI systems capable of addressing complex, multi-domain challenges.
Ultimately, shared memory is poised to broaden the scope of AI applications and accelerate its integration into increasingly diverse and demanding real-world scenarios.

##### From Individual Privacy to Collective Privacy

With the increasing prevalence of data sharing in the AI era, the focus of privacy protection is gradually shifting from the traditional notion of individual privacy to the broader and emerging concept of collective privacy.
Conventional privacy frameworks primarily aim to safeguard personal data, preventing unauthorized access, leakage, or misuse of individually identifiable information.
However, in the context of large language models, individual data is often aggregated into group-level datasets for large-scale analysis and prediction.
Collective privacy concerns the protection of the rights and interests of groups or communities whose data is used collectively, raising questions about how to prevent misuse, profiling, or excessive surveillance at the group level.
As memory systems in AI become more advanced and interconnected, ensuring collective privacy will emerge as a critical challenge.
Addressing this issue will require innovative techniques that can effectively balance the trade-off between data utility and privacy preservation.

##### From Rule-Based Evolution to Automated Evolution

Traditional AI systems evolve by reflecting on past experiences—such as reusing successful strategies—based on accumulated knowledge and historical data.
However, this evolutionary process often depends on manually crafted rules and heuristic adjustments to enable such self-reflection.
While rule-based evolution can be effective, it inherently limits the system’s flexibility, scalability, and efficiency, with the quality and generalizability of the rules directly constraining the system’s adaptive capabilities.
Looking ahead, AI systems are expected to achieve automated evolution, dynamically adjusting and optimizing themselves by leveraging both personal and system-level memories in response to changing data and environmental contexts.
Such systems will be capable of autonomously identifying performance bottlenecks and initiating self-improvement without relying on explicit, human-defined rules.
This transition toward self-directed adaptation will significantly enhance system responsiveness, reduce the need for human intervention, and enable a more intelligent, dynamic, and continuously self-evolving paradigm.

## 6 Conclusion

Memory plays a pivotal role in the advancement of AI systems in the era of large language models (LLMs). It not only shapes the degree of personalization in AI behavior but also influences key capabilities such as adaptability, reasoning, planning, and self-evolution.
This article systematically examines the relationship between human memory and memory mechanisms in LLM-driven AI systems, exploring how principles of human cognition can inspire the design of more efficient and flexible memory architectures.
We begin by analyzing various categories of human memory—including perceptual memory, working memory, and long-term memory—and compare them with existing memory models in AI. Building upon this, we propose an eight-quadrant classification framework grounded in three dimensions: object, form, and time, offering a theoretical foundation for the construction of multi-level and comprehensive memory systems.
Furthermore, we review the current state of memory development in AI from both personal memory and system memory perspectives.
Finally, we identify key open challenges in contemporary AI memory design and outline promising directions for future research in the LLM era.
We believe that, with continued technological progress, AI systems will increasingly adopt more dynamic, adaptive, and intelligent memory architectures, thereby enabling more robust applications across complex, real-world tasks.

## References

- [All citations in the text refer to this section's bibliography content.]

### Original URL
https://arxiv.org/html/2504.15965v1
</details>

---

## Additional Sources Scraped

---
<details>
<summary>AI Agents in 2025: Expectations vs. Reality | IBM</summary>

# AI agents in 2025: Expectations vs. reality

## What are AI agents?

An AI agent is a software program capable of acting autonomously to understand, plan and execute tasks. AI agents are powered by LLMs and can interface with tools, other models and other aspects of a system or network as needed to fulfill user goals.

We’re going beyond asking a chatbot to suggest a dinner recipe based on the available ingredients in the fridge. Agents are more than automated customer experience emails that inform you it’ll be a few days until a real-world human can get to your inquiry.

AI agents differ from traditional AI assistants that need a prompt each time they generate a response. In theory, a user gives an agent a high-level task, and the agent figures out how to complete it.

Current offerings are still in the early stages of approaching this idea. “What’s commonly referred to as ‘agents’ in the market is the addition of rudimentary planning and tool-calling (sometimes called function calling) capabilities to LLMs,” says Ashoori. “These enable the LLM to break down complex tasks into smaller steps that the LLM can perform.”

Hay is optimistic that more robust agents are on the way: “You wouldn’t need any further progression in models today to build future AI agents,” he says.

With that out of the way, what’s the conversation about agents over the coming year, and how much of it can we take seriously?

## Narrative 1: 2025 is the year of the AI agent

“More and better agents” are on the way, predicts Time. “Autonomous ‘agents’ and profitability are likely to dominate the artificial intelligence agenda,” reports Reuters. “The age of agentic AI has arrived,” promises Forbes, in response to a claim from Nvidia’s Jensen Huang.

Tech media is awash with assurances that our lives are on the verge of a total transformation. Autonomous agents are poised to streamline and alter our jobs, drive optimization and accompany us in our daily lives, handling our mundanities in real time and freeing us up for creative pursuits and other higher-level tasks.

### 2025 as the year of agentic exploration

“IBM and Morning Consult did a survey of 1,000 developers who are building AI applications for enterprise, and 99% of them said they are exploring or developing AI agents,” explains Ashoori. “So yes, the answer is that 2025 is going to be the year of the agent.” However, that declaration is not without nuance.

After establishing the current market conception of agents as LLMs with function calling, Ashoori draws a distinction between that idea and truly autonomous agents. “The true definition \[of an AI agent\] is an intelligent entity with reasoning and planning capabilities that can autonomously take action. Those reasoning and planning capabilities are up for discussion. It depends on how you define that.”

“I definitely see AI agents heading in this direction, but we’re not fully there yet,” says Gajjar. “Right now, we’re seeing early glimpses—AI agents can already analyze data, predict trends and automate workflows to some extent. But building AI agents that can autonomously handle complex decision-making will take more than just better algorithms. We’ll need big leaps in contextual reasoning and testing for edge cases,” she adds.

Danilevsky isn’t convinced that this is anything new. “I'm still struggling to truly believe that this is all that different from just orchestration,” she says. “You've renamed orchestration, but now it's called agents, because that's the cool word. But orchestration is something that we've been doing in programming forever.”

With regard to 2025 being the year of the agent, Danilevsky is skeptical. “It depends on what you say an agent is, what you think an agent is going to accomplish and what kind of value you think it will bring,” she says. “It's quite a statement to make when we haven't even yet figured out ROI (return on investment) on LLM technology more generally.”

And it’s not just the business side that has her hedging her bets. “There's the hype of imagining if this thing could think for you and make all these decisions and take actions on your computer. Realistically, that's terrifying.”

Danilevsky frames the disconnect as one of miscommunication. “\[Agents\] tend to be very ineffective because humans are very bad communicators. We still can't get chat agents to interpret what you want correctly all the time.”

Still, the forthcoming year holds a lot of promise as an era of experimentation. “I'm a big believer in \[2025 as the year of the agent\],” says Hay excitedly.

Every large tech company and hundreds of startups are now experimenting with agents. Salesforce, for example, has released their Agentforce platform, which enables users to create agents that are easily integrated within the Salesforce app ecosystem.

“The wave is coming and we're going to have a lot of agents. It's still a very nascent ecosystem, so I think a lot of people are going to build agents, and they're going to have a lot of fun.”

## Narrative 2: Agents can handle highly complex tasks on their own

This narrative assumes that today’s agents meet the theoretical definition outlined in the introduction to this piece. 2025’s agents will be fully autonomous AI programs that can scope out a project and complete it with all the necessary tools they need and with no help from human partners. But what’s missing from this narrative is nuance.

### Today’s models are more than enough

Hay believes that the groundwork has already been laid for such developments. “The big thing about agents is that they have the ability to plan,” he outlines. “They have the ability to reason, to use tools and perform tasks, and they need to do it at speed and scale.”

He cites 4 developments that, compared to the best models of 12 to 18 months ago, mean that the models of early 2025 can power the agents envisioned by the proponents of this narrative:

- Better, faster, smaller models
- Chain-of-thought (COT) training
- Increased context windows
- Function calling

“Now, most of these things are in play,” Hay continues. “You can have the AI call tools. It can plan. It can reason and come back with good answers. It can use inference-time compute. You’ll have better chains of thought and more memory to work with. It's going to run fast. It’s going to be cheap. That leads you to a structure where I think you can have agents. The models are improving and they're getting better, so that's only going to accelerate.”

### Realistic expectations are a must

Ashoori is careful to differentiate between what agents will be able to do later, and what they can do now. “There is the promise, and there is what the agent's capable of doing today,” she says. “I would say the answer depends on the use case. For simple use cases, the agents are capable of \[choosing the correct tool\], but for more sophisticated use cases, the technology has yet to mature.”

Danilevsky reframes the narrative as a contextual one. “If something is true one time, that doesn't mean it's true all the time. Are there a few things that agents can do? Sure. Does that mean you can agentize any flow that pops into your head? No.”

For Gajjar, the question is one of risk and governance. “We’re seeing AI agents evolve from content generators to autonomous problem-solvers. These systems must be rigorously stress-tested in sandbox environments to avoid cascading failures. Designing mechanisms for rollback actions and ensuring audit logs are integral to making these agents viable in high-stakes industries.”

But she is optimistic that we’ll meet these challenges. “I do think we’ll see progress this year in creating rollback mechanisms and audit trails. It’s not just about building smarter AI but also designing safety nets so we can trace and fix issues quickly when things go off track.”

And while Hay is hopeful about the potential for agentic development in 2025, he sees a problem in another area: “Most organizations aren't agent-ready. What's going to be interesting is exposing the APIs that you have in your enterprises today. That's where the exciting work is going to be. And that's not about how good the models are going to be. That's going to be about how enterprise-ready you are.”

## Narrative 3: AI orchestrators will govern networks of AI agents

The “new normal” envisioned by this narrative sees teams of AI agents corralled under orchestrator uber-models that manage the overall project workflow.

Enterprises will use AI orchestration to coordinate multiple agents and other machine learning (ML) models working in tandem and using specific expertise to complete tasks.

### Compliance is paramount to healthy AI adoption

Gajjar views this prediction not only as credible, but likely. “We’re at the very beginning of this shift, but it’s moving fast. AI orchestrators could easily become the backbone of enterprise AI systems this year—connecting multiple agents, optimizing AI workflows and handling multilingual and multimedia data,” she opines. However, she cautions against rushing in without appropriate safeguards in place.

“At the same time, scaling these systems will need strong compliance frameworks to keep things running smoothly without sacrificing accountability,” warns Gajjar. “2025 might be the year we go from experiments to large-scale adoption, and I can’t wait to see how companies balance speed with responsibility.”

It’s imperative that organizations dedicate themselves with equal fervor to data and AI governance and compliance as they do to adopting the latest innovations.

### Progress isn’t a straight line

“You are going to have an AI orchestrator, and they’re going to work with multiple agents,” outlines Hay. “A bigger model would be an orchestrator, and smaller models will be doing constrained tasks.”

However, as agents evolve and improve, Hay predicts a shift away from orchestrated workflows to single-agent systems. “As those individual agents get more capable, you're going to switch toward saying, ‘I've got this agent that can do everything end-to-end.’”

Hay foresees a back-and-forth evolution as models develop. “You're going to hit a limit on \[what single agents can do\], and then you're going to go back to multi-agent collaboration again. You're going to push and pull between multi-agent frameworks and a single godlike agent.” And while AI models will be the ones determining project workflows, Hay believes humans will always remain in the loop.

### Orchestration isn’t always the right solution

For Ashoori, the need for a meta-orchestrator isn’t quite a given and comes down to intended use cases. “It's an architecture decision,” she explains. “Each agent, by definition, should have the capability to figure out if they need to orchestrate with another agent, pull in a bunch of tools or if they need some complimentary data. You don't necessarily need a middle agent that sits on top and monitors everyone to tell them what to do.”

However, in some cases, you might. “You may need to figure out how to use a combination of specialized agents for your purpose,” supposes Ashoori. “In that case, you may decide to create your own agent that acts as the orchestrator.”

Danilevsky advises enterprises to first understand which workflows can and should be agentized for what degree of ROI, then develop an AI strategy from there. “Are there going to be some orchestration flows with some agents? Sure. But should everything in your organization be orchestrated with agentic flow? No, it won't work.”

## Narrative 4: Agents will augment human workers

A prevailing vision of agentic adoption over the next year is one which sees agents augmenting, but not necessarily replacing, human workers. They’ll serve as contributors to a streamlined workflow led by humans, say advocates.

However, fears of AI-related job loss are a constant in the ongoing conversation surrounding enterprise AI adoption. As agents become more capable, will business leaders encourage agent-human collaboration or seek to replace workers with AI tools?

### Agents should be a tool, not a replacement

Ashoori believes the best path forward lies in trusting employees to determine the optimal use of AI in their respective jobs. “We should empower employees to decide how they want to leverage agents, but not necessarily replacing them in every single situation,” she explains. Some job functions are ripe for offloading to an agent, while with others, human input can’t be replaced. “An agent might transcribe and summarize a meeting, but you're not going to send your agent to have this conversation with me.”

Danilevsky shares Ashoori’s view and notes that the adoption of agents in the workplace will not come without growing pains. “You're still going to have cases where as soon as something gets more complex, you're going to need a human.” While business leaders may be tempted to cut short-term costs by eliminating jobs, agent use “...is going to settle down much more into an augmented sort of role. You're supposed to constantly have a human, and the human is being helped, but the human makes the final decisions,” says Danilevsky, describing her human-in-the-loop (HITL) vision for AI.

Hay sees a pathway towards sustainable AI adoption at work. “If we do this right, AI is there to augment humans to do things better. If AI is done correctly, then it frees us up to do more interesting things.” But at the same time, he can imagine another version of the future where AI is prioritized too highly. “There is a real risk that when done badly and wrongly, that we end up with humans augmenting the AI as opposed to the other way around.”

Gajjar also cautions against leaning too heavily on AI. “I don’t see AI agents replacing jobs overnight, but they’ll definitely reshape how we work. Repetitive, low-value tasks are already being automated, which frees people up for more strategic and creative work. That said, companies need to be intentional about how they introduce AI. Governance frameworks—like those focused on fairness, transparency and accountability—are going to be key.”

### Open source AI leads to new opportunities

For Hay, one upside of open source AI models is how they open the door to a future AI agent marketplace and subsequent monetization for creators. “I think open source agents are the key,” says Hay. “Because of open source, anybody can build an agent, and it can do useful tasks. And you can create your own company.”

It’s also important to weigh potential growing pains and organizational restructuring against AI-driven benefits, especially in the Global South, believes Hay.

LLMs provide text-based output, which can reach users through SMS in areas without reliable internet connections. “The enablement that can occur in countries \[without strong internet access\] because AI can work in a low-bandwidth scenario and it's getting cheaper all the time—this is very exciting,” Hay says.

## Final thoughts: Governance and strategy are essential for successful AI agent implementation

Over the course of these conversations, 2 themes came up time and time again with all 4 of our experts. Aside from the 4 narratives we looked at, a sustainable route through the current AI explosion will require enterprises and business leaders to embrace 2 ideas:

1. AI governance underpins successful compliance and responsible use.
2. A robust AI strategy focused on economic value will lead businesses to sustainable AI adoption.

### The need for governance

“Companies need governance frameworks to monitor performance and ensure accountability as these agents integrate deeper into operations,” urges Gajjar. “This is where IBM’s Responsible AI approach really shines. It’s all about making sure AI works with people, not against them, and building systems that are trustworthy and auditable from day one.”

Ashoori paints a picture of a potential agentic AI mishap. “Using an agent today is basically grabbing an LLM and allowing it to take actions on your behalf. What if this action is connecting to a dataset and removing a bunch of sensitive records?”

“Technology doesn’t think. It can't be responsible,” states Danilevsky. In terms of risks such as accidental data leakage or deletion, “the scale of the risk is higher,” she says. “There's only so much that a human can do in so much time, whereas the technology can do things in a lot less time and in a way that we might not notice.”

And when that happens, one cannot simply point the finger at the AI and remove all blame from the people responsible for it. “A human being in that organization is going to be held responsible and accountable for those actions,” warns Hay.

“So the challenge here becomes transparency,” says Ashoori. “And traceability of actions for every single thing that the agents do. You need to know exactly what's happening and be able to track, trace it and control it.”

For Danilevsky, free experimentation is the path to sustainable development. “\[There is a lot of value\] in allowing people to actually play with the technology and build it and try to break it.” She also urges developers to be cautious when determining which models to use and what data they feed into those models. “\[Some providers will\] take all your data. So just be a little careful.”

### Why AI strategy matters

“The current AI boom is absolutely FOMO-driven, and it will calm down when the technology becomes more normalized,” predicts Danilevsky. “I think that people will start to understand better what kinds of things work and don't.” “The focus should also be on integrating AI agents into ecosystems where they can learn and adapt continuously, driving long-term efficiency gains,” adds Gajjar.

Danilevsky is quick to ground expectations and recenter the conversation on demonstrable business needs. “Enterprises need to be careful to not become the hammer in search of a nail,” she begins. “We had this when LLMs first came on the scene. People said, ‘Step one: we’re going to use LLMs. Step two: What should we use them for?’”

Hay encourages enterprises to get agent-ready ahead of time. “The value is going to be with those organizations that take their private data and organize that in such a way so that the agents are researching against your documents.” Every enterprise houses a wealth of valuable proprietary data, and transforming that data so that it can power agentic workflows supports positive ROI.

“With agents, enterprises have an option to leverage their proprietary data and existing enterprise workflows to differentiate and scale,” says Ashoori.  “Last year was the year of experimentation and exploration for enterprises. They need to scale that impact and maximize their ROI of generative AI. Agents are the ticket to making that happen.”

For more information on successful AI implementation in the enterprise, read Maryam Ashoori’s guide to agentic AI cost analysis. Also be sure to catch Vyoma Gajjar and Chris Hay expounding on their predictions for AI in 2025 on IBM’s Mixture of Experts podcast.

### Original URL
https://www.ibm.com/think/insights/ai-agents-2025-expectations-vs-reality
</details>

---
<details>
<summary>What Is Agentic Reasoning? | IBM</summary>

# What is agentic reasoning?

## What is agentic reasoning?

Agentic reasoning is a component @https://www.ibm.com/think/topics/components-of-ai-agents of AI agents @https://www.ibm.com/think/topics/ai-agents that handles decision-making. It allows artificial intelligence @https://www.ibm.com/think/topics/artificial-intelligence agents to conduct tasks autonomously by applying conditional logic or heuristics, relying on perception and memory, enabling it to pursue goals and optimize for the best possible outcome.

Earlier machine learning @https://www.ibm.com/think/topics/machine-learning models followed a set of preprogrammed rules to arrive at a decision. Advances in AI have led to AI models @https://www.ibm.com/think/topics/ai-model with more evolved reasoning capabilities, but they still require human intervention to convert information into knowledge. Agentic reasoning takes it one step further, allowing AI agents @https://www.ibm.com/think/insights/ai-agents-2025-expectations-vs-reality to transform knowledge into action.

The “reasoning engine” powers the planning and tool calling @https://www.ibm.com/think/topics/tool-calling phases of agentic workflows @https://www.ibm.com/think/topics/agentic-workflows. Planning decomposes a task into more manageable reasoning, while tool calling helps inform an AI agent’s decision through available tools. These tools can include application programming interfaces (APIs) @https://www.ibm.com/think/topics/api, external datasets @https://www.ibm.com/think/topics/dataset and data sources such as knowledge graphs @https://www.ibm.com/think/topics/knowledge-graph.

For businesses, agentic AI @https://www.ibm.com/think/topics/agentic-ai can further ground the reasoning process in evidence through retrieval-augmented generation (RAG) @https://www.ibm.com/think/topics/retrieval-augmented-generation. RAG systems @https://www.ibm.com/think/topics/agentic-rag can retrieve enterprise data and other relevant information that can be added to an AI agent’s context for reasoning.

## Agentic reasoning strategies

Agentic reasoning can be approached in different ways based on an agent’s architecture @https://www.ibm.com/think/topics/agentic-architecture and type. Here are some common techniques for AI agent reasoning, including the pros and cons of each:

**● Conditional logic**

**● Heuristics**

**● ReAct (Reason + Act)**

**● ReWOO (Reasoning WithOut Observation)**

**● Self-reflection**

**● Multiagent reasoning**

### Conditional logic

Simple AI agents follow a set of preprogrammed condition-action rules. These rules usually take the form of “if-then” statements, where the “if” portion specifies the condition and the “then” portion indicates the action. When a condition is met, the agent carries out the corresponding action.

This reasoning methodology is especially suitable for domain-specific use cases. In finance, for instance, a fraud detection agent flags a transaction as fraudulent according to a set of criteria defined by a bank.

With conditional logic, agentic AI @https://www.ibm.com/think/insights/agentic-ai can’t act accordingly if it comes across a scenario it doesn’t recognize. To reduce this inflexibility, model-based agents use their memory and perception to store a current model or state of their environment. This state is updated as the agent receives new information. Model-based agents, however, are still bound by their condition-action rules.

For example, a robot navigates through a warehouse to stock a product on a shelf. It consults a model of the warehouse for the route it takes, but when it senses an obstacle, it can alter its path to avoid that obstacle and continue its traversal.

### Heuristics

AI agent systems can also use heuristics for reasoning. Goal-based agents, for instance, have a preset goal. Using a search algorithm @https://www.ibm.com/think/topics/machine-learning-algorithms, they find sequences of actions that can help them achieve their goal and then plan these actions before conducting them.

For example, an autonomous vehicle can have a navigation agent whose objective is to suggest the quickest path to a destination in real-time. It can search through different routes and recommend the fastest 1.

Like goal-based agents, utility-based agents search for action sequences that achieve a goal, but they factor in utility as well. They employ a utility function to determine the most optimal outcome. In the navigation agent example, it can be tasked with finding not only the swiftest route but also 1 that will consume the least amount of fuel.

### ReAct (Reason + Act)

This reasoning paradigm involves a think-act-observe loop for step-by-step problem-solving and iterative enhancement of responses. An agent is instructed to generate traces of its reasoning process,1 much like what happens with chain-of-thought @https://www.ibm.com/think/topics/chain-of-thoughts reasoning in generative AI @https://www.ibm.com/think/topics/generative-ai (gen AI) models and large language models (LLMs) @https://www.ibm.com/think/topics/large-language-models. It then acts on that reasoning and observes its output,2 updating its context with new reasoning based on its observations. The agent repeats the cycle until it arrives at an answer or solution.2

ReAct does well on natural language-specific tasks, and its traceability improves transparency. However, it can also generate the same reasoning and actions repeatedly, which can lead to infinite loops.2

### ReWOO (Reasoning WithOut Observation)

Unlike ReAct, ReWOO removes the observation step and plans ahead instead. This agentic reasoning design pattern consists of 3 modules: planner, worker and solver.3

The planner module breaks down a task into subtasks and allocates each of them to a worker module. The worker incorporates tools used to substantiate each subtask with evidence and facts. Finally, the solver module synthesizes all the subtasks and their corresponding evidence to draw a conclusion.3

ReWOO outperforms ReAct on certain natural language processing @https://www.ibm.com/think/topics/natural-language-processing (NLP) benchmarks @https://www.ibm.com/think/topics/llm-benchmarks. However, adding extra tools can degrade ReWOO’s performance, and it doesn’t do well in situations where it has limited context about its environment.3

### Self-reflection

Agentic AI can also include self-reflection as part of assessing and refining its reasoning capabilities. An example of this is Language Agent Tree Search (LATS), which shares similarities with tree-of-thought @https://www.ibm.com/think/topics/tree-of-thoughts reasoning in LLMs.

LATS was inspired by the Monte Carlo reinforcement learning @https://www.ibm.com/think/topics/reinforcement-learning method, with researchers adapting the Monte Carlo Tree Search for LLM-based agents.4 LATS builds a decision tree @https://www.ibm.com/think/topics/decision-trees that represents a state as a node and an edge as an action, searches the tree for potential action options and employs a state evaluator to choose a particular action.2 It also applies a self-reflection reasoning step, incorporating its own observations as well as feedback from a language model to identify any errors in reasoning and recommend alternatives.2 The reasoning errors and reflections are stored in memory, serving as additional context for future reference.4

LATS excels in more complex tasks such as coding and interactive question answering @https://www.ibm.com/think/topics/question-answering and in workflow @https://www.ibm.com/think/topics/ai-workflow automation @https://www.ibm.com/think/topics/automation, including web search and navigation.4 However, a more involved approach and extra self-reflection step makes LATS more resource- and time-intensive compared to methods like ReAct.2

### Multiagent reasoning

Multiagent systems @https://www.ibm.com/think/topics/multiagent-system consist of multiple AI agents working together to solve complex problems. Each agent specializes in a certain domain and can apply its own agentic reasoning strategy.

However, the decision-making process can vary based on the AI system’s architecture. In a hierarchical or vertical ecosystem, 1 agent acts as a leader for AI orchestration @https://www.ibm.com/think/topics/ai-orchestration and decides which action to take. Meanwhile, in a horizontal architecture, agents decide collectively.

## Challenges in agentic reasoning

Reasoning is at the core of AI agents and can result in more powerful AI capabilities, but it also has its limitations. Here are some challenges in agentic reasoning:

**● Computational complexity**

**● Interpretability**

**● Scalability**

### Computational complexity

Agentic reasoning can be difficult to implement. The process also requires significant time and computational power, especially when solving more complicated real-world problems. Enterprises must find ways to optimize their agentic reasoning strategies and be ready to invest in the necessary AI platforms @https://www.ibm.com/think/insights/how-to-choose-the-best-ai-platform and resources for development.

### Interpretability

Agentic reasoning might lack explainability @https://www.ibm.com/think/topics/explainable-ai and transparency @https://www.ibm.com/think/topics/ai-transparency on how decisions were made. Various methods can help establish interpretability @https://www.ibm.com/think/topics/interpretability, and integrating AI ethics @https://www.ibm.com/think/topics/ai-ethics and human oversight within algorithmic development are critical to make sure agentic reasoning engines make decisions fairly, ethically and accurately.

### Scalability

Agentic reasoning techniques are not 1-size-fits-all solutions, making it hard to scale them across AI applications. Businesses @https://www.ibm.com/think/topics/artificial-intelligence-business might need to tailor these reasoning design patterns for each of their use cases, which requires time and effort.

##### Footnotes

_All links reside outside ibm.com_

1 ReAct: Synergizing Reasoning and Acting in Language Models @https://arxiv.org/abs/2210.03629, arXiv, 10 March 2023

2 The Landscape of Emerging AI Agent Architectures for Reasoning, Planning, and Tool Calling: A Survey @https://arxiv.org/abs/2404.11584, arXiv, 17 April 2024

3 Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models @https://arxiv.org/abs/2310.04406, arXiv, 6 June 2024

4 Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models @https://arxiv.org/abs/2310.04406, arXiv, 6 June 2024

### Original URL
https://www.ibm.com/think/topics/agentic-reasoning
</details>

---
<details>
<summary>How Reasoning AI Agents Transform High-Stakes Decision Making | NVIDIA Blog</summary>

AI agents powered by large language models (LLMs) have grown past their FAQ chatbot beginnings to become true digital teammates capable of planning, reasoning and taking action — and taking in corrective feedback along the way.

Thanks to reasoning AI models, agents can learn how to think critically and tackle complex tasks. This new class of “reasoning agents” can break down complicated problems, weigh options and make informed decisions — while using only as much compute and as many tokens as needed.

Reasoning agents are making a splash in industries where decisions rely on multiple factors. Such industries range from customer service and healthcare to manufacturing and financial services.

## **Reasoning On vs. Reasoning Off**

Modern AI agents can toggle reasoning on and off, allowing them to efficiently use compute and tokens.

A full chain‑of‑thought pass performed during reasoning can take up to 100x more compute and tokens than a quick, single‑shot reply — so it should only be used when needed. Think of it like turning on headlights — switching on high beams only when it’s dark and turning them back to low when it’s bright enough out.

Single-shot responses are great for simple queries — like checking an order number, resetting a password or answering a quick FAQ. Reasoning might be needed for complex, multistep tasks such as reconciling tax depreciation schedules or orchestrating the seating at a 120‑guest wedding.

New NVIDIA Llama Nemotron models, featuring advanced reasoning capabilities, expose a simple system‑prompt flag to enable or disable reasoning, so developers can programmatically decide per query. This allows agents to perform reasoning only when the stakes demand it — saving users wait times and minimizing costs.

## **Reasoning AI Agents in Action**

Reasoning AI agents are already being used for complex problem-solving across industries, including:

- **Healthcare:** Enhancing diagnostics and treatment planning.
- **Customer Service**: Automating and personalizing complex customer interactions, from resolving billing disputes to recommending tailored products.
- **Finance:** Autonomously analyzing market data and providing investment strategies.
- **Logistics and Supply Chain:** Optimizing delivery routes, rerouting shipments in response to disruptions and simulating possible scenarios to anticipate and mitigate risks.
- **Robotics**: Powering warehouse robots and autonomous vehicles, enabling them to plan, adapt and safely navigate dynamic environments.

Many customers are already experiencing enhanced workflows and benefits using reasoning agents.

Amdocs uses reasoning-powered AI agents to transform customer engagement for telecom operators. Its amAIz GenAI platform, enhanced with advanced reasoning models such as NVIDIA Llama Nemotron and amAIz Telco verticalization, enables agents to autonomously handle complex, multistep customer journeys — spanning customer sales, billing and care.

EY is using reasoning agents to significantly improve the quality of responses to tax-related queries. The company compared generic models to tax-specific reasoning models, which revealed up to an 86% improvement in response quality for tax questions when using a reasoning approach.

SAP’s Joule agents — which will be equipped with reasoning capabilities from Llama Nemotron –– can interpret complex user requests, surface relevant insights from enterprise data and execute cross-functional business processes autonomously.

## **Designing an AI Reasoning Agent**

A few key components are required to build an AI agent, including tools, memory and planning modules. Each of these components augments the agent’s ability to interact with the outside world, create and execute detailed plans, and otherwise act semi- or fully autonomously.

Reasoning capabilities can be added to AI agents at various places in the development process. The most natural way to do so is by augmenting planning modules with a large reasoning model, like Llama Nemotron Ultra or DeepSeek-R1. This allows more time and reasoning effort to be used during the initial planning phase of the agentic workflow, which has a direct impact on the overall outcomes of systems.

The AI-Q NVIDIA AI Blueprint and the NVIDIA Agent Intelligence toolkit can help enterprises break down silos, streamline complex workflows and optimize agentic AI performance at scale.

The AI-Q blueprint provides a reference workflow for building advanced agentic AI systems, making it easy to connect to NVIDIA accelerated computing, storage and tools for high-accuracy, high-speed digital workforces. AI-Q integrates fast multimodal data extraction and retrieval using NVIDIA NeMo Retriever, NIM microservices and AI agents.

In addition, the open-source NVIDIA Agent Intelligence toolkit enables seamless connectivity between agents, tools and data. Available on GitHub, this toolkit lets users connect, profile and optimize teams of AI agents, with full system traceability and performance profiling to identify inefficiencies and improve outcomes. It’s framework-agnostic, simple to onboard and can be integrated into existing multi-agent systems as needed.

## **Build and Test Reasoning Agents With Llama Nemotron**

Learn more about Llama Nemotron, which recently was at the top of industry benchmark leaderboards for advanced science, coding and math tasks. Join the community shaping the future of agentic, reasoning-powered AI.

Plus, explore and fine-tune using the open Llama Nemotron post-training dataset to build custom reasoning agents. Experiment with toggling reasoning on and off to optimize for cost and performance.

And test NIM-powered agentic workflows, including retrieval-augmented generation and the NVIDIA AI Blueprint for video search and summarization, to quickly prototype and deploy advanced AI solutions.

### Original URL
https://blogs.nvidia.com/blog/reasoning-ai-agents-decision-making/
</details>

---
<details>
<summary>What is AI Agent Orchestration? | IBM</summary>

# What is AI agent orchestration?

Artificial intelligence (AI) agent orchestration is the process of coordinating multiple specialized AI agents within a unified system to efficiently achieve shared objectives.

Rather than relying on a single, general-purpose AI solution, AI agent orchestration employs a network of AI agents, each designed for specific tasks, working together to automate complex workflows and processes.

To fully understand AI agent orchestration, it's essential to first understand AI agents themselves. This involves understanding the differences between two key types of AI: generative AI, which creates original content based on a user’s prompt, and agentic AI, which autonomously makes decisions and acts to pursue complex goals with minimal supervision.

AI assistants exist on a continuum, starting with rule-based chatbots, progressing to more advanced virtual assistants and evolving into generative AI and large language model (LLM) powered assistants capable of handling single-step tasks. At the top of this progression are AI agents, which operate autonomously. These agents make decisions, design workflows and use function calling to connect with external tools—such as application programming interfaces (APIs), data sources, web searches and even other AI agents—to fill gaps in their knowledge. This is agentic AI.

AI agents are specialized, meaning each one is optimized for a particular function. Some agents focus on business and customer-facing tasks like billing, troubleshooting, scheduling and decision-making, while others handle more technical functions like natural language processing (NLP), data retrieval and process automation. Advanced LLMs such as OpenAI's ChatGPT-4o or Google's Gemini often power these agents, with generative AI capabilities enabling them to create human-like responses and handle complex tasks autonomously.

Multi-agent systems (MAS) emerge when multiple AI agents collaborate, either in a structured or decentralized manner, to solve complex tasks more efficiently than a single agent might.

In practice, AI agent orchestration functions like a digital symphony. Each agent has a unique role and the system is guided by an orchestrator—either a central AI agent or framework —that manages and coordinates their interactions. The orchestrator helps synchronize these specialized agents, ensuring that the right agent is activated at the right time for each task. This coordination is crucial for handling multifaceted workflows that involve various tasks, helping ensure that processes are run seamlessly and efficiently.

For example, as part of customer service automation, the orchestrator agent (the system responsible for managing AI agents) might determine whether to engage a billing agent versus a technical support agent, helping ensure that customers receive seamless and relevant assistance. In MAS, agents might coordinate without a single orchestrator, dynamically communicating to collaboratively solve problems (see “Types of AI orchestration” below)

The benefits of AI agent orchestration are significant in industries with complex, dynamic needs such as telecommunications, banking and healthcare. By deploying specialized agents that are trained on targeted datasets and workflows, businesses can enhance operational efficiency, improve decision-making and deliver more accurate, efficient and context-aware results for both employees and customers.

## Why AI agent orchestration is important

As AI systems grow more advanced, a single AI model or agent is often insufficient for handling complex tasks. Autonomous systems frequently struggle to collaborate because they are built across multiple clouds and applications, leading to siloed operations and inefficiencies. AI agent orchestration bridges these gaps, enabling multiple AI agents to work together efficiently and ensuring that sophisticated tasks are run seamlessly.

In large-scale applications such as healthcare, finance and customer service, multiple agents often need to work together to handle different aspects of a task. For example, in healthcare, AI agents can coordinate between diagnostic tools, patient management systems and administrative workflows to streamline operations and enhance treatment accuracy. Without orchestration, these agents might work in isolation, leading to inefficiencies, redundancies or gaps in execution.

By managing interactions between multi-agent systems, orchestration helps ensure that each agent contributes effectively toward a shared goal. It optimizes workflows, minimizes errors and enhances interoperability, allowing AI systems to dynamically allocate resources, prioritize tasks and respond to changing conditions in real time. This capability is valuable in fields requiring continuous optimization such as supply chain management and personalized digital assistants.

As AI systems continue to evolve, AI agent orchestration becomes increasingly essential for unlocking their full potential.

## Types of AI agent orchestration

There are several types of AI agent orchestration. Real-world systems often combine multiple orchestration styles for more effective results.

**Centralized orchestration**: A single AI orchestrator agent acts as the "brain" of the system, directing all the other agents, assigning tasks and making final decisions. This structured approach helps ensure consistency, control and predictable workflows.

**Decentralized orchestration**: This model shifts away from a single, controlling entity, allowing MAS to function through direct communication and collaboration. Agents make independent decisions or reach a consensus as a group. This makes the system more scalable and resilient since no single failure can bring it down.

**Hierarchical orchestration**: Here, AI agents are arranged in layers, resembling a tiered command structure. Higher-level orchestrator agents oversee and manage lower-level agents, striking a balance between strategic control and task-specific execution. This allows for more organized workflows while still enabling specialized agents to operate with some autonomy. If the hierarchy becomes too rigid, adaptability can suffer.

**Federated orchestration**: This approach focuses on collaboration between independent AI agents or separate organizations, allowing them to work together without fully sharing data or relinquishing control over their individual systems. This is especially useful in situations where privacy, security or regulatory constraints prevent unrestricted data sharing, such as in healthcare, banking or cross-company collaborations.

## Comparing AI agent orchestration with related practices

**AI orchestration** manages and automates various AI components—like machine learning models, data pipelines and APIs—to help ensure that they work together efficiently within a system. It focuses on optimizing performance, automating repetitive tasks, supporting scalability and system-wide performance.

**AI agent orchestration** is a subset of AI orchestration that focuses specifically on coordinating autonomous AI agents—software entities that can make independent decisions and take actions. It helps ensure that agents collaborate effectively, assigning tasks and structuring workflows.

**Multi-agent orchestration** goes a step further, managing multiple AI agents working together on complex problems. It deals with communication, role allocation and conflict resolution to help ensure seamless collaboration between agents.

## AI agent orchestration steps

AI agent orchestration is a structured process to help ensure seamless collaboration between AI agents. The goal is to manage specialized agents effectively so they can autonomously complete tasks, share data flow and optimize workflows.

Initial steps involving design, configuration and implementation are performed by humans, including as AI engineers, developers and business strategists. Once the orchestrator agent is set up, it autonomously manages AI applications, assigning tasks, coordinating workflows and facilitating real-time collaboration.

The process generally follows these key steps:

- Assessment and planning
- Selection of specialized AI agents
- Orchestration framework implementation
- Agent selection and assignment
- Workflow coordination and execution
- Data sharing and context management
- Continuous optimization and learning

### Assessment and planning (human-driven)

Before orchestration begins, organizations assess their existing AI ecosystem and identify processes that might benefit from multi-agent orchestration. The orchestration team defines clear objectives, determines the scope of integration and selects the appropriate AI technologies.

### Selection of specialized AI agents (human-driven)

AI engineers and developers choose task-specific AI agents, such as those specializing in data analysis, automation or decision-making. These agents use gen AI and machine learning models to enhance their functions.

### Orchestration framework implementation (human-driven)

System architects integrate selected AI agents into a unified orchestration framework, establishing workflows that facilitate smooth agent-to-agent communication. This involves:

- Defining task execution sequences
- Setting up API integrations for data access
- Implementing open source orchestration tools such as IBM watsonx Orchestrate, Microsoft Power Automate and LangChain

Once this is complete, the orchestrator agent takes over real-time execution.

### Agent selection and assignment (orchestrator-driven)

The orchestrator dynamically identifies the best-suited AI agents for each task based on real-time data, workload balancing and predefined rules.

### Workflow coordination and execution (orchestrator-driven)

The orchestrator platform manages task sequencing and execution, helping to ensure smooth collaboration between agents. This includes:

- Breaking down tasks into subtasks
- Assigning the right AI agents to handle each step
- Managing inter-agent dependencies
- Integrating with external systems through API calls to access necessary data and services

### Data sharing and context management (orchestrator-driven)

To help ensure accuracy and prevent redundant work, AI agents continuously exchange information, maintaining a shared knowledge base. The orchestrator updates agents with real-time context.

### Continuous optimization and learning (orchestrator + human input)

The orchestrator monitors agent performance, detects inefficiencies and can autonomously adjust workflows. Human oversight is often required for refining orchestration strategies, retraining AI models or modifying orchestration rules for long-term improvements.

## AI agent orchestration benefits

AI agent orchestration offers several key benefits across various industries, making it a valuable approach for businesses aiming to enhance their operations and customer interactions.

**Enhanced efficiency**: Coordinating multiple specialized agents helps businesses streamline workflows, reduce redundancies and improve overall operational performance.

**Agility and flexibility**: AI agent orchestration allows organizations to adapt their operations rapidly as market conditions change.

**Improved experiences**: Orchestrated AI agents can enhance operational efficiency and provide more accurate and personalized support, resulting in more satisfying experiences for customers and employees.

**Increased reliability and fault tolerance**: The failure of one agent can be mitigated by others, which enhances system reliability and helps ensure continuous service delivery.

**Self-improving workflows**: Unlike traditional integration patterns, agent orchestration enables the creation of workflows that can autonomously adapt to new data and evolving requirements, improving over time.

**Scalability**: AI agent orchestration allows organizations to handle increased demand without compromising performance or accuracy.

## AI agent orchestration challenges

AI agent orchestration comes with several challenges, but each has potential solutions. By addressing these challenges, AI agent orchestration can be more efficient, scalable and resilient.

**Multi-agent dependencies**: When deploying multi-agent frameworks, there is a risk of malfunction. Systems built on the same foundation models may be susceptible to shared vulnerabilities, which might lead to a widespread failure of all agents that are involved or make them more prone to external attacks. This underscores the importance of data governance in building foundation models and thorough training and testing processes.

**Coordination and communication**: If agents don’t interact properly, they can end up working against each other or duplicating efforts. To prevent this, it’s important to establish clear protocols, standardized APIs and reliable message-passing systems to keep everything running smoothly.

**Scalability**: As the number of AI agents increases, maintaining system performance and manageability becomes more complex. A poorly designed orchestration system may struggle with increased workloads, leading to delays or system failures. This can be avoided by using decentralized or hierarchical orchestration models that distribute decision-making, preventing a single point of failure or congestion.

**Decision-making complexity**: In multi-agent environments, determining how tasks should be allocated and executed can become highly complex. Without a clear structure, agents may struggle to make decisions, particularly in dynamic environments where conditions frequently change. Reinforcement learning, prioritization algorithms and predefined roles can help ensure that agents can autonomously determine their tasks while maintaining efficiency.

**Fault tolerance**: What happens if an agent or the orchestrator itself fails? Fault tolerance is crucial and needs to be reinforced by designing failover mechanisms, redundancy strategies and self-healing architectures that allow the system to recover automatically without human intervention.

**Data privacy and security**: AI agents frequently process and share sensitive information, raising concerns about data security and privacy. To mitigate these risks, organizations should implement strong encryption protocols, enforce strict access controls and use federated learning techniques that allow AI models to improve collaboratively without exposing raw data.

**Adaptability and learning**: AI agents must continuously adapt to new tasks and challenges. Systems that require constant manual updates can become inefficient and costly to maintain. To enhance adaptability, machine learning techniques, continuous monitoring and feedback loops can be integrated into the orchestration process. These methods enable AI agents to refine their behavior over time, improving individual and system-wide performance without requiring frequent human intervention.

### Original URL
https://www.ibm.com/think/topics/ai-agent-orchestration
</details>

---
<details>
<summary>[2504.19678] From LLM Reasoning to Autonomous AI Agents: A Comprehensive Review</summary>

## I Introduction

Large Language Models (LLMs) such as OpenAI’s GPT-4 \[1\], Qwen2.5-Omni \[2\], DeepSeek-R1 \[3\], and Meta’s LLaMA \[4\] have transformed AI by enabling human-like text generation and advanced natural language processing, spurring innovation in conversational agents, automated content creation, and real-time translation \[5\]. Recent enhancements have extended their utility to multimodal tasks, including text-to-image and text-to-video generation that broaden the scope of generative AI applications \[6\]. However, their dependence on static pre-training data can lead to outdated outputs and hallucinated responses \[7,8\], a limitation that Retrieval-Augmented Generation (RAG) addresses by incorporating real-time data from knowledge bases, APIs, or the web \[9,10\]. Building on this, the evolution of intelligent agents employing reflection, planning, and multi-agent collaboration has given rise to Agentic RAG systems, which dynamically orchestrate information retrieval and iterative refinement to manage complex workflows effectively \[11,12\].

Recent advances in large language models have paved the way for highly autonomous AI systems that can independently handle complex research tasks. These systems, often referred to as agentic AI, can generate hypotheses, conduct literature reviews, design experiments, analyze data, accelerate scientific discovery, and reduce research costs \[13–16\]. Several frameworks, such as LitSearch, ResearchArena, and Agent Laboratory, have been developed to automate various research tasks, including citation management and academic survey generation \[17–19\]. However, challenges persist, especially in executing domain-specific literature reviews and ensuring the reproducibility and reliability of automated processes \[20,21\]. Parallel to these developments in research automation, large language model-based agents have also begun to transform the medical field \[22\]. These agents are increasingly used for diagnostic support, patient communication, and medical education by integrating clinical guidelines, medical knowledge bases, and healthcare systems. Despite their promise, these applications face significant hurdles, including concerns over reliability, reproducibility, ethical governance, and safety \[23–25\]. Addressing these issues is crucial for ensuring that LLM-based agents can be effectively and responsibly incorporated into clinical practice, underscoring the need for comprehensive evaluation frameworks that can reliably measure their performance across various healthcare tasks \[26–28\].

LLM-based agents are emerging as a promising frontier in AI, combining reasoning and action to interact with complex digital environments \[29,30\]. Therefore, various approaches have been explored to enhance LLM-based agents, from combining reasoning and acting using techniques like React \[31\] and Monte Carlo Tree Search \[32\] to synthesizing high-quality data with methods like Learn-by-Interact \[33\], which sidestep assumptions such as state reversals. Other strategies involve training on human-labeled or GPT-4 distilled data with systems like AgentGen \[34\] and AgentTuning \[35\] to generate trajectory data. At the same time, reinforcement learning methods utilize offline algorithms and iterative refinement through reward models and feedback to enhance efficiency and performance in realistic environments \[36,37\].

LLM-based Multi-Agents harness the collective intelligence of multiple specialized agents, enabling advanced capabilities over single-agent systems by simulating complex real-world environments through collaborative planning, discussion, and decision-making. This approach leverages the communicative strengths and domain-specific expertise of LLMs, allowing distinct agents to interact effectively, much like human teams tackling problem-solving tasks \[38,39\]. Recent research highlights promising applications across various fields, including software development \[40,41\], multi-robot systems \[42,43\], society simulation \[44\], policy simulation \[45\], and game simulation \[46\].

The main contributions of this study are:

- We present a comparative table of benchmarks developed between 2019 and 2025 that rigorously evaluate large language models and autonomous AI agents across multiple domains.
- We propose a taxonomy of approximately 60 LLM and AI-agent benchmarks, including general and academic knowledge reasoning, mathematical problem solving, code generation and software engineering, factual grounding and retrieval, domain-specific evaluations, multimodal and embodied tasks, task orchestration, and interactive and agentic assessments.
- We present prominent AI-agent frameworks from 2023 to 2025 that integrate large language models with modular toolkits, enabling autonomous decision-making and multi-step reasoning.
- We provide applications of autonomous AI agents in various fields, including materials science and biomedical research, academic ideation and software engineering, synthetic data generation and chemical reasoning, mathematical problem-solving and geographic information systems, as well as multimedia, healthcare, and finance.
- We survey agent-to-agent collaboration protocols, namely the Agent Communication Protocol (ACP), the Model Context Protocol (MCP), and the Agent-to-Agent Protocol (A2A).
- We outline recommendations for future research on autonomous AI agents, specifically advanced reasoning strategies, failure modes in multi-agent large language model (LLM) systems, automated scientific discovery, dynamic tool integration via reinforcement learning, integrated search capabilities, and security vulnerabilities in agent protocols.

## II Related Works

The growing field of autonomous AI agents powered by large language models has inspired a wide range of research efforts across multiple domains. In this section, we review the most relevant studies that investigate the integration of LLM-based agents into software engineering, propose agent architectures and evaluation frameworks, explore the development of multi-agent systems, and examine domain-specific applications, including healthcare, game-theoretic scenarios, GUI interactions, personal assistance, scientific discovery, and chemistry.

### II-A LLM-based Agents in Software Engineering

Wang et al. \[47\] present a survey that bridges Large Language Model (LLM)-based agent technologies with software engineering (SE). It highlights how LLMs have achieved significant success in various domains and have been integrated into SE tasks, often under the agent paradigm, whether explicitly or implicitly. The study presents a structured framework for LLM-based agents in SE, comprising three primary modules: perception, memory, and action. Jin et al. \[48\] investigate the use of large language models (LLMs) and LLM-based agents in software engineering, distinguishing between the traditional capabilities of LLMs and the enhanced functionalities offered by autonomous agents. It highlights the significant success of LLMs in tasks such as code generation and vulnerability detection, while also addressing their limitations, specifically the issues of autonomy and self-improvement that LLM-based agents aim to overcome. The paper provides an extensive review of current practices across six key domains: requirement engineering, code generation, autonomous decision-making, software design, test generation, and software maintenance.

### II-B Agent Architectures and Evaluation Frameworks

Singh et al. \[49\] delves into Agentic Retrieval-Augmented Generation (Agentic RAG), a sophisticated evolution of traditional Retrieval-Augmented Generation systems that enhances the capabilities of large language models (LLMs). While LLMs have transformed AI through human-like text generation and language understanding, their dependence on static training data often results in outdated or imprecise responses. The paper addresses these limitations by embedding autonomous agents within the RAG framework, enabling dynamic, real-time data retrieval and adaptive workflows. It details how agentic design patterns such as reflection, planning, tool utilization, and multi-agent collaboration equip these systems to manage complex tasks and support multi-step reasoning. The survey offers a comprehensive taxonomy of Agentic RAG architectures, highlights key applications across various sectors, including healthcare, finance, and education, and outlines practical implementation strategies.

Complementing this architectural perspective, Yehudai et al. \[50\] mark a significant milestone in artificial intelligence by surveying evaluation methodologies for agents powered by large language models (LLMs). It thoroughly reviews the capabilities of these agents, focusing on core functions such as planning, tool utilization, self-reflection, and memory, while assessing specialized applications ranging from web interactions to software engineering and conversational tasks. The authors uncover a clear trend toward developing more rigorous, dynamically updated evaluation frameworks by examining both targeted benchmarks for domain-specific applications and those designed for more generalist agents. Moreover, the paper critically highlights existing deficiencies in the field, notably the need for metrics that more effectively capture cost efficiency, safety, and robustness. In doing so, it maps the current landscape of agent evaluation and sets forth compelling directions for future inquiry, underscoring the importance of scalable and fine-grained evaluation techniques in the rapidly evolving AI domain.

Similarly, Chen et al. \[51\] focus on Role-Playing Agents (RPAs), a growing class of LLM-based agents that mimic human behavior across various tasks. Recognizing the inherent challenges in evaluating such diverse systems, the authors systematically reviewed 1,676 papers published between January 2021 and December 2024. Their extensive analysis identifies six key agent attributes, seven task attributes, and seven evaluation metrics that are prevalent in the current literature. Based on these insights, the paper proposes an evidence-based, actionable, and generalizable evaluation guideline designed to standardize the assessment of RPAs.

### II-C Multi-Agent Systems

Yan et al. \[52\] provides a comprehensive survey on integrating LLMs into multi-agent systems (MAS). Their work emphasizes the communication-centric aspects that enable agents to engage in both cooperative and competitive interactions, thereby tackling tasks that are unmanageable for individual agents. The paper examines system-level features, internal communication mechanisms, and challenges, including scalability, security, and multimodal integration. In a related study, Guo et al. \[38\] offer an extensive overview of LLM-based multi-agent systems, charting the evolution from single-agent decision-making to collaborative frameworks that enhance collective problem-solving and world simulation. The authors detail how the evolution from single-agent decision-making to collaborative multi-agent frameworks has enabled significant advances in complex problem-solving and world simulation. Key aspects of these systems are examined, including the domains and environments they simulate, the profiling and communication strategies employed by individual agents, and the mechanisms that underpin the enhancement of their collective capacities.

### II-D Domain-Specific Applications

#### II-D1 Healthcare

Wang et al. \[28\] explores the transformative impact of LLM-based agents on healthcare, presenting a detailed review of their architectures, applications, and inherent challenges. It dissects the core components of medical agent systems, such as system profiles, clinical planning mechanisms, and medical reasoning frameworks, while also discussing methods to enhance external capacities. Major application areas include clinical decision support, medical documentation, training simulations, and overall healthcare service optimization. The survey further evaluates the performance of these agents using established frameworks and metrics, identifying persistent challenges such as hallucination management, multimodal integration, and ethical considerations.

#### II-D2 Social Agents in Game-Theoretic Scenarios

Feng et al. \[53\] provide a review of research on LLM-based social agents in game-theoretic scenarios. This area has gained prominence for assessing social intelligence in AI systems. The authors categorize the literature into three main components. First, the game framework is examined, highlighting various choice- and communication-focused scenarios. Second, the paper explores the attributes of social agents, examining their preferences, beliefs, and reasoning capabilities. Third, it discusses evaluation protocols incorporating game-agnostic and game-specific metrics to assess performance. By synthesizing current studies and outlining future research directions, the survey offers valuable insights to further the development and systematic evaluation of social agents within game-theoretic contexts.

#### II-D3 GUI Agents

Zhang et al. \[54\] review LLM-brained GUI agents, marking a paradigm shift in human-computer interaction through integrating multimodal LLMs. It traces the historical evolution of GUI automation, detailing how advancements in natural language understanding, code generation, and visual processing have enabled these agents to interpret complex graphical user interface (GUI) elements and execute multi-step tasks from conversational commands. The survey systematically examines the core components of these systems, including existing frameworks, data collection and utilization methods for training, and the development of specialized large-scale action models for GUI tasks.

#### II-D4 Personal LLM Agents

Li et al. \[55\] explore the evolution of intelligent personal assistants (IPAs) by focusing on Personal LLM Agents LLM-based agents that deeply integrate personal data and devices to provide enhanced personal assistance. The authors outline the limitations of traditional IPAs, including insufficient understanding of user intent, task planning, and tool utilization, which have hindered their practicality and scalability. In contrast, the emergence of foundation models like LLMs offer new possibilities by leveraging advanced semantic understanding and reasoning for autonomous problem-solving. The survey systematically reviews the architecture and design choices underlying Personal LLM Agents, informed by expert opinions, and examines key challenges related to intelligence, efficiency, and security. Furthermore, it comprehensively analyzes representative solutions addressing these challenges, laying the groundwork for Personal LLM Agents to become a major paradigm in next-generation end-user software.

#### II-D5 Scientific Discovery

Gridach et al. \[21\] explore the transformative role of Agentic AI in scientific discovery, underscoring its potential to automate and enhance research processes. It reviews how these systems, endowed with reasoning, planning, and autonomous decision-making capabilities, are revolutionizing traditional research activities, including literature reviews, hypothesis generation, experimental design, and data analysis. The paper highlights recent advancements across multiple scientific domains, such as chemistry, biology, and materials science, by categorizing existing Agentic AI systems and tools. It provides a detailed discussion on key evaluation metrics, implementation frameworks, and datasets used in the field, offering valuable insights into current practices. Moreover, the paper critically addresses significant challenges, including automating comprehensive literature reviews, ensuring system reliability, and addressing ethical concerns. It outlines future research directions, emphasizing the importance of human-AI collaboration and improved system calibration.

#### II-D6 Chemistry

Ramos et al. \[56\] examine the transformative impact of large language models (LLMs) in chemistry, focusing on their roles in molecule design, property prediction, and synthesis optimization. It highlights how LLMs not only accelerate scientific discovery through automation but also discuss the advent of LLM-based autonomous agents. These agents extend the functionality of LLMs by interfacing with their environment and performing tasks such as literature scraping, automated laboratory control, and synthesis planning. Expanding the discussion beyond chemistry, the review also considers applications across other scientific domains.

### II-E Comparison with Our Survey

Table I presents a consolidated view of how existing works cover key themes, benchmarks, AI agent frameworks, AI agent applications, AI agents protocols, and challenges & open problems against our survey. While prior studies typically focus on one or two aspects (e.g., Yehudai et al. \[50\] on evaluation benchmarks, Singh et al. \[49\] on RAG architectures, Yan et al. \[52\] on multi-agent communication, or Wang et al. \[28\] on domain-specific applications), none integrate the full spectrum of developments in a single, unified treatment. In contrast, our survey is the first to systematically combine state-of-the-art benchmarks, framework design, application domains, communication protocols, and a forward-looking discussion of challenges and open problems, thereby providing researchers with a comprehensive roadmap for advancing LLM-based autonomous AI agents.

## III LLM and Agentic AI Benchmarks

This section provides a comprehensive overview of benchmarks developed between 2019 and 2025 that rigorously evaluate large language models (LLMs) across diverse and challenging domains. For instance, ENIGMAEVAL \[57\] assesses complex multimodal puzzle-solving by requiring the synthesis of textual and visual clues, while ComplexFuncBench \[59\] challenges models with multi-step function-calling tasks that mirror real-world scenarios. Humanity’s Last Exam (HLE) \[60\] further raises the bar by presenting expert-level academic questions across a broad spectrum of subjects, thereby reflecting the growing demand for deeper reasoning and domain-specific proficiency. Additional frameworks such as FACTS Grounding \[61\] and ProcessBench \[62\] scrutinize the models’ capacities for generating factually accurate long-form responses and detecting errors in multi-step reasoning. Meanwhile, innovative evaluation paradigms like Agent-as-a-Judge \[64\], JudgeBench \[65\], and CyberMetric \[75\] provide granular insights into cybersecurity competencies and error-detection capabilities.

### III-A ENIGMAEVAL benchmark

ENIGMAEVAL \[57\] is a benchmark designed to rigorously evaluate advanced language models’ multimodal and long-context reasoning capabilities using challenging puzzles derived from global competitions. The dataset comprises 1,184 complex puzzles that combine text and images, requiring models to synthesize disparate clues, perform multi-step deductive reasoning, and integrate visual and semantic information to arrive at unambiguous, verifiable solutions. Unlike conventional benchmarks focusing on well-structured academic tasks, ENIGMAEVAL pushes models into unstructured, creative problem-solving scenarios where even state-of-the-art systems achieve only about 7% accuracy on standard puzzles and fail on the hardest ones.

### III-B MMLU Benchmark

Measuring Massive Multitask Language Understanding (MMLU) \[58\] is a comprehensive benchmark designed by Hendrycks et al. (2021) to evaluate large language models across a diverse range of subjects, from elementary mathematics to professional law. The benchmark comprises 57 tasks that test models’ ability to apply broad world knowledge and problem-solving skills in zero-shot and few-shot settings, emphasizing generalization without task-specific fine-tuning. The study also uncovers challenges related to model calibration and the imbalance between procedural and declarative knowledge, highlighting critical areas where current models fall short of expert-level proficiency.

### III-C ComplexFuncBench Benchmark

Zhong et al. \[59\] introduced ComplexFuncBench, a novel benchmark designed to evaluate large language models (LLMs) on complex function calling tasks in real-world settings. Unlike previous benchmarks, ComplexFuncBench challenges models with multi-step operations within a single turn, adherence to user-imposed constraints, reasoning over implicit parameter values, and managing extensive input lengths that can exceed 500 tokens, including scenarios with a context window of up to 128k tokens. Complementing the benchmark, the authors present an automatic evaluation framework, ComplexEval, which quantitatively assesses performance across over 1,000 scenarios derived from five distinct aspects of function calling. Experimental results reveal significant limitations in current state-of-the-art LLMs, with closed models like Claude 3.5 and OpenAI’s GPT-4 outperforming open models such as Qwen 2.5 and Llama 3.1. Notably, the study identifies prevalent issues, including value errors and premature termination in multi-step function calls, underscoring the need for further research to enhance the function-calling capabilities of LLMs in practical applications.

### III-D Humanity’s Last Exam (HLE) Benchmark

Phan et al. \[60\] introduced Humanity’s Last Exam (HLE), a benchmark designed to push the limits of LLMs by challenging them with expert-level academic tasks. Unlike traditional benchmarks such as MMLU, where LLMs have achieved over 90% accuracy, HLE presents a significantly more demanding test, featuring 3,000 questions spanning over 100 subjects including mathematics, humanities, and the natural sciences. This benchmark is the product of a global collaborative effort, with nearly 1,000 subject matter experts from over 500 institutions contributing questions that are both multi-modal and resistant to quick internet retrieval, ensuring that only genuine deep academic understanding can lead to success. The tasks, which include both multiple-choice and short-answer formats with clearly defined, verifiable answers, expose a substantial performance gap: current state-of-the-art LLMs, such as DeepSeek R1, OpenAI’s models, Google DeepMind Gemini Thinking, and Anthropic Sonnet 3.5, perform at less than 10% accuracy and suffer from high calibration errors, indicating overconfidence in incorrect responses. The results underscore that while existing benchmarks may no longer provide a meaningful measure of progress, HLE serves as a critical tool for assessing the true academic reasoning capabilities of LLMs, potentially heralding a new era in benchmark design as the field moves toward more challenging and nuanced evaluations in the pursuit of artificial general intelligence.

### III-E FACTS Grounding benchmark

Google DeepMind introduced FACTS Grounding \[61\], a comprehensive benchmark designed to evaluate how accurately LLMs ground their long-form responses in provided source documents while avoiding hallucinations. The benchmark comprises 1,719 meticulously crafted examples split into 860 public and 859 private cases that require models to generate detailed answers strictly based on a corresponding context document, with inputs reaching up to 32,000 tokens. Covering diverse domains such as medicine, law, technology, finance, and retail, FACTS Grounding excludes tasks that require creativity, mathematics, or complex reasoning, focusing squarely on factual accuracy and information synthesis. To ensure robust and unbiased evaluation, responses are assessed in two phases: eligibility and factual grounding using a panel of three frontier LLM judges (Gemini 1.5 Pro, GPT-4o, and Claude 3.5 Sonnet), with final scores derived from the aggregation of these assessments. With an online leaderboard hosted on Kaggle already populated with initial results where, for instance, Gemini 2.0 Flash leads with 83.6% accuracy FACTS Grounding aims to drive industry-wide advancements in grounding and factuality, ultimately fostering greater trust and reliability in LLM applications.

### III-F ProcessBench benchmark

Qwen team \[62\] introduced ProcessBench, a novel benchmark specifically designed to evaluate the ability of language models to detect errors within the reasoning process for mathematical problem solving. ProcessBench comprises 3,400 test cases, primarily drawn from competition- and Olympiad-level math problems, where each case includes a detailed, step-by-step solution with human-annotated error locations. Models are tasked with identifying the earliest erroneous step or confirming that all steps are correct, thereby providing a granular assessment of their reasoning accuracy. The benchmark is employed to evaluate two classes of models: process reward models (PRMs) and critic models, the latter involving general large language models (LLMs) that are prompted to critique each solution step. Experimental results reveal two key findings. First, existing PRMs generally fail to generalize to more challenging math problems beyond standard datasets like GSM8K and MATH, often underperforming relative to both prompted LLM-based critics and a PRM fine-tuned on a larger, more complex PRM800K dataset. Second, the best open-source model tested, QwQ-32B-Preview, demonstrates error detection capabilities that rival those of the proprietary GPT-4o, although it still falls short compared to reasoning-specialized models like o1-mini.

### III-G OmniDocBench Benchmark

Ouyang et al. \[63\] introduced OmniDocBench, a comprehensive multi-source benchmark designed to advance automated document content extraction a critical component for high-quality data needs in LLMs and RAG systems. OmniDocBench features a meticulously curated and annotated dataset spanning nine diverse document types including academic papers, textbooks, slides, notes, and financial documents and utilizes a detailed evaluation framework with 19 layout categories and 14 attribute labels to facilitate multi-level assessments. Through extensive comparative analysis of existing modular pipelines and multimodal end-to-end methods, the benchmark reveals that while specialized models (e.g., Nougat) outperform general vision-language models (VLMs) on standard documents, general VLMs exhibit superior resilience and adaptability in challenging scenarios, such as those involving fuzzy scans, watermarks, or colorful backgrounds. Moreover, fine-tuning general VLMs with domain-specific data leads to enhanced performance, as evidenced by high accuracy scores in tasks like formula recognition (with models such as GPT-4o, Mathpix, and UniMERNet achieving around 85–86.8% accuracy) and table recognition (RapidTable at 82.5%). Nonetheless, the findings also highlight persistent challenges, notably that complex column layouts continue to degrade reading order accuracy across all evaluated models.

### III-H Agent-as-a-Judge

Meta team proposed the Agent-as-a-Judge framework \[64\], an innovative evaluation approach explicitly designed for agentic systems that overcome the limitations of traditional methods, which either focus solely on outcomes or require extensive manual labor. This framework provides granular, intermediate feedback throughout the task-solving process by leveraging agentic systems to evaluate other agentic systems. The authors demonstrate its effectiveness on code generation tasks using DevAI, a new benchmark comprising 55 realistic automated AI development tasks annotated with 365 hierarchical user requirements. Their evaluation shows that Agent-as-a-Judge not only dramatically outperforms the conventional LLM-as-a-Judge approach (which typically achieves a 60–70% alignment rate with human assessment) but also reaches an impressive 90% alignment with human judgments. Additionally, this method offers substantial cost and time savings, reducing evaluation costs to approximately 2.29% ($30.58 vs. $1,297.50) and cutting evaluation time down to 118.43 minutes compared to 86.5 hours for human assessments.

### III-I JudgeBench Benchmark

Tan et al. \[65\] proposed JudgeBench, a novel benchmark designed to objectively evaluate LLM-based judges models that are increasingly employed to assess and improve the outputs of large language models by focusing on their ability to accurately discern factual and logical correctness rather than merely aligning with human stylistic preferences. Unlike prior benchmarks that rely primarily on crowdsourced human evaluations, JudgeBench leverages a carefully constructed set of 350 challenging response pairs spanning knowledge, reasoning, math, and coding domains. The benchmark employs a novel pipeline to transform challenging existing datasets into paired comparisons with preference labels based on objective correctness while mitigating positional bias through double evaluation with swapped order. Comprehensive testing across various judge architectures, including prompted, fine-tuned, multi-agent judges, and reward models, reveals that even strong models, such as GPT-4o, often perform only marginally better than random guessing, particularly on tasks requiring rigorous error detection in intermediate reasoning steps. Moreover, fine-tuning can significantly boost performance, as evidenced by a 14% improvement observed in Llama 3.1 8B, and reward models achieve accuracies in the 59–64% range.

### III-J SimpleQA Benchmark

SimpleQA \[66\] is a benchmark introduced by OpenAI to assess and improve the factual accuracy of large language models on short, fact-seeking questions. Comprising 4,326 questions spanning domains such as science/tech, politics, art, and geography, SimpleQA challenges models to deliver a single correct answer under a strict three-tier grading system (”correct,” ”incorrect,” or ”not attempted”). While built on foundational datasets such as TriviaQA and Natural Questions, SimpleQA presents a more challenging task for LLMs. Early results indicate that even advanced models, such as OpenAI o1-preview, achieve only 42.7% accuracy (with Claude 3.5 Sonnet trailing at 28.9%), and models tend to exhibit overconfidence in their incorrect responses. Moreover, experiments that repeated the same question 100 times revealed a strong correlation between higher answer frequency and overall accuracy. This benchmark thus provides critical insights into the current limitations of LLMs in handling straightforward, factual queries. It underscores the need for further improvements in grounding model outputs in reliable, factual data.

### III-K FineTasks

FineTasks \[67\] is a data-driven evaluation framework designed to systematically select reliable tasks for assessing LLMs across diverse languages. Developed as the first step toward the broader FineWeb Multilingual initiative, FineTasks evaluates candidate tasks based on four critical metrics: monotonicity, low noise, non-random performance, and model ordering consistency to ensure robustness and reliability. In an extensive study, the Hugging Face team tested 185 candidate tasks across nine languages (including Chinese, French, Arabic, Russian, Thai, Hindi, Turkish, Swahili, and Telugu), ultimately selecting 96 final tasks that cover domains such as reading comprehension, general knowledge, language understanding, and reasoning. The work further reveals that the formulation of tasks has a significant impact on performance; for instance, Cloze format tasks are more effective during early training phases, while multiple-choice formats yield better evaluation results. Recommended evaluation metrics include length normalization for most tasks and pointwise mutual information (PMI) for complex reasoning challenges. Benchmarking 35 open and closed-source LLMs demonstrated that open models are narrowing the gap with their proprietary counterparts, with Qwen 2 models excelling in high- and mid-resource languages and Gemma-2 particularly strong in low-resource settings. Moreover, the FineTasks framework supports over 550 tasks across various languages, providing a scalable and comprehensive platform for advancing multilingual large language model (LLM) evaluation.

### III-L FRAMES benchmark

Google team \[68\] propose FRAMES (Factuality, Retrieval, and Reasoning MEasurement Set), a comprehensive evaluation dataset specifically designed to assess the capabilities of retrieval-augmented generation (RAG) systems built on LLMs. FRAMES addresses a critical need by unifying evaluations of factual accuracy, retrieval effectiveness, and reasoning ability in an end-to-end framework, rather than assessing these facets in isolation. The dataset comprises 824 challenging multi-hop questions spanning diverse topics, including history, sports, science, and health, each requiring the integration of information from between two and fifteen Wikipedia articles. By labeling questions with specific reasoning types, such as numerical or tabular. FRAMES provides a nuanced benchmark to identify the strengths and weaknesses of current RAG implementations. Baseline experiments reveal that state-of-the-art models like Gemini-Pro-1.5-0514 achieve only 40% accuracy when operating without retrieval mechanisms, but their performance increases significantly to 66% with a multi-step retrieval pipeline, representing a greater than 50% improvement.

### III-M DABStep benchmark

DabStep \[69\] is a new framework from Hugging Face that pioneers a step-based approach to enhance the performance and efficiency of language models on multi-step reasoning tasks. DabStep addresses the challenges of traditional end-to-end inference by decomposing complex problem-solving into discrete, manageable steps, enabling models to refine their outputs through step-level feedback and iterative dynamic adjustments. This method is designed to enable models to self-correct and navigate the complexities of multi-step reasoning processes more effectively. However, despite these innovative improvements, experimental results reveal that even the best-performing model under this framework only achieves a 16% success rate on the evaluated tasks. This modest accuracy underscores the significant challenges that remain in effectively training models for complex, iterative reasoning and highlights the need for further research and optimization.

### III-N BFCL v2 benchmark

Mao et al. \[70\] propose BFCL v2, a novel benchmark and leaderboard designed to evaluate large language models’ function calling abilities using real-world, user-contributed data. The benchmark comprises 2,251 question-function-answer pairs, enabling comprehensive assessments across a range of scenarios from multiple and straightforward function calls to parallel executions and irrelevance detection. By leveraging authentic user interactions, BFCL v2 addresses prevalent issues such as data contamination, bias, and limited generalization in previous evaluation methods. Initial evaluations reveal that models like Claude 3.5 and GPT-4 consistently outperform others, with Mistral, Llama 3.1 FT, and Gemini following in performance. However, some open models, such as Hermes, struggle due to potential prompting and formatting challenges. Overall, BFCL v2 offers a rigorous and diverse platform for benchmarking the practical capabilities of LLMs in interfacing with external tools and APIs, thereby providing valuable insights for future advancements in function calling and interactive AI systems.

### III-O SWE-Lancer benchmark

OpenAI team \[71\] presents SWE-Lancer, an innovative benchmark comprised of over 1,400 freelance software engineering tasks collected from Upwork, representing more than $1 million in real-world payouts. This benchmark encompasses both independent engineering tasks, ranging from minor bug fixes to substantial feature implementations valued up to $32,000, and managerial tasks, where models must select the best technical proposals. Independent tasks are rigorously evaluated using end-to-end tests that have been triple-verified by experienced engineers. At the same time, managerial decisions are benchmarked against the selections made by the original hiring managers. Experimental results indicate that state-of-the-art models, such as Claude 3.5 Sonnet, still struggle with the majority of these tasks, achieving a 26.2% pass rate on independent tasks and 44.9% on managerial tasks, which translates to an estimated earning of $403K a figure well below the total available value. Notably, the analysis highlights that while models tend to perform better in evaluative managerial roles than in direct code implementation, increasing inference-time computing can enhance performance.

### III-P Comprehensive RAG Benchmark (CRAG)

Yang et al. \[72\] propose the Comprehensive RAG Benchmark (CRAG), a novel dataset designed to evaluate the factual question-answering capabilities of Retrieval-Augmented Generation systems rigorously. CRAG comprises 4,409 question-answer pairs across five domains and eight distinct question categories. It incorporates mock APIs to simulate web and Knowledge Graph retrieval, thereby reflecting the varied levels of entity popularity and temporal dynamism encountered in real-world scenarios. Empirical results show that state-of-the-art large language models without grounding achieve only around 34% accuracy on CRAG, and that incorporating simple RAG methods improves this to just 44%, whereas industry-leading RAG systems can reach 63% accuracy without hallucination. The benchmark also highlights significant performance drops for questions involving highly dynamic, lower-popularity, or more complex facts. Notably, CRAG focuses solely on evaluating the generative component of the RAG pipeline, and early findings indicate that Llama 3 70B nearly matches GPT-4 Turbo across these tasks.

### III-Q OCCULT Benchmark

Kouremetis et al. \[73\] present OCCULT, a novel and lightweight operational evaluation framework that rigorously measures the cybersecurity risks associated with using large language models (LLMs) for offensive cyber operations (OCO). Traditionally, evaluating AI in cybersecurity has relied on simplistic, all-or-nothing tests such as capture-the-flag exercises, which fail to capture the nuanced threats faced by modern infrastructure. In contrast, OCCULT enables cybersecurity experts to craft repeatable and contextualized benchmarks by simulating real-world threat scenarios. The authors detail three distinct OCO benchmarks designed to assess the capability of LLMs to execute adversarial tactics, providing preliminary evaluation results that indicate a significant advancement in AI-enabled cyber threats. Most notably, the DeepSeek-R1 model correctly answered over 90% of questions in the Threat Actor Competency Test for LLMs (TACTL).

### III-R DIA benchmark

Dynamic Intelligence Assessment (DIA) \[74\] is introduced as a novel methodology to more rigorously test and compare the problem-solving abilities of AI models across diverse domains such as mathematics, cryptography, cybersecurity, and computer science. Unlike traditional benchmarks that rely on static question-answer pairs often allowing models to perform uniformly well or rely on memorization DIA employs dynamic question templates with mutable parameters, presented in various formats including text, PDFs, compiled binaries, visual puzzles, and CTF-style challenges. This framework also introduces four innovative metrics to evaluate a model’s reliability and confidence across multiple attempts, revealing that even simple questions are frequently answered incorrectly when posed in different forms. Notably, the evaluation shows that while API models like GPT‑4o may overestimate their mathematical capabilities, models such as ChatGPT‑4o perform better due to practical tool usage, and OpenAI’s o1-mini excels in self-assessment of task suitability. Testing 25 state-of-the-art LLMs with DIA-Bench reveals significant gaps in handling complex tasks and in adaptive intelligence, establishing a new standard for evaluating both problem-solving performance and a model’s ability to recognize its own limitations.

### III-S CyberMetric benchmark

Tihanyi et al. \[75\] introduces a suite of novel multiple-choice Q&A benchmark datasets CyberMetric-80, CyberMetric-500, CyberMetric-2000, and CyberMetric-10000 designed to evaluate the cybersecurity knowledge of LLMs rigorously. By leveraging GPT-3.5 and Retrieval-Augmented Generation (RAG), the authors generated questions from diverse cybersecurity sources such as NIST standards, research papers, publicly accessible books, and RFCs. Complete with four possible answers, each question underwent extensive rounds of error checking and refinement, with over 200 hours of human expert validation to ensure accuracy and domain relevance. Evaluations were conducted on 25 state-of-the-art large language models (LLMs), and the results were further benchmarked against human performance on CyberMetric-80 in a closed-book scenario. Findings reveal that models like GPT-4o, GPT-4-turbo, Mixtral-8x7 B-Instruct, Falcon-180 B-Chat, and GEMINI-pro 1.0 exhibit superior cybersecurity understanding, outperforming humans on CyberMetric-80, while smaller models such as Llama-3-8B, Phi-2, and Gemma-7b lag behind, underscoring the value of model scale and domain-specific data in this challenging field.

### III-T BIG-Bench Extra Hard

A team from Google DeepMind \[76\] addresses a critical gap in evaluating large language models by tackling the limitations of current reasoning benchmarks, which have primarily focused on mathematical and coding tasks. While the BIG-Bench dataset \[122\] and its more complex variant, BIG-Bench Hard (BBH) \[123\], have provided comprehensive assessments of general reasoning abilities, recent advances in LLMs have led to saturation, with state-of-the-art models achieving near-perfect scores on many BBH tasks. To overcome this, the authors introduce BIG-Bench Extra Hard (BBEH). This novel benchmark replaces each BBH task with a more challenging variant designed to probe similar reasoning capabilities at an elevated difficulty level. Evaluations on BBEH reveal that even the best general-purpose models only achieve an average accuracy of 9.8%, while reasoning-specialized models reach 44.8%, highlighting substantial room for improvement and underscoring the ongoing challenge of developing LLMs with robust, versatile reasoning skills.

### III-U MultiAgentBench benchmark

Zhu et al. \[77\] introduce MultiAgentBench, a benchmark specifically designed to evaluate the capabilities of multi-agent systems powered by LLMs in dynamic, interactive environments. Unlike traditional benchmarks that focus on single-agent performance or narrow domains, MultiAgentBench encompasses six diverse domains, including research proposal writing, Minecraft structure building, database error analysis, collaborative coding, competitive Werewolf gameplay, and resource bargaining to measure both task completion and the quality of agent coordination using milestone-based performance indicators. The study investigates various coordination protocols, such as star, chain, tree, and graph topologies, and finds that direct peer-to-peer communication and cognitive planning are particularly effective evidenced by a 3% improvement in milestone achievement when planning is employed while also noting that adding more agents can decrease performance. Among the models evaluated (GPT-4o-mini, 3.5, and Llama), GPT-4o-mini achieved the highest average task score, and graph-based coordination protocols outperformed other structures in research scenarios.

### III-V GAIA Benchmark

GAIA \[78\] is a groundbreaking benchmark designed to assess General AI Assistants on real-world questions that tap into fundamental abilities like reasoning, multi-modality handling, web browsing, and tool use. Unlike traditional benchmarks that focus on increasingly specialized tasks, GAIA features conceptually simple questions solvable by humans at 92% accuracy that current systems, such as GPT-4 with plugins, struggle with, achieving only 15%. Comprising 466 meticulously curated questions with reference answers, GAIA shifts the evaluation paradigm toward measuring AI robustness in everyday reasoning tasks, a critical step toward achieving true Artificial General Intelligence (AGI). This substantial performance gap between humans and state-of-the-art models emphasizes the need for AI systems that can mimic the general-purpose, resilient reasoning exhibited by average human problem solvers.

### III-W CASTLE Benchmark

Dubniczky et al. \[79\] introduce CASTLE, a novel benchmarking framework for evaluating software vulnerability detection methods, addressing existing approaches’ critical weaknesses. CASTLE assesses 13 static analysis tools, 10 LLMs, and two formal verification tools using a meticulously curated dataset of 250 micro-benchmark programs that cover 25 common CWEs. The framework proposes a new evaluation metric, the CASTLE Score, to enable fair comparisons across different methods. Results reveal that while formal verification tools like ESBMC minimize false positives, they struggle with vulnerabilities beyond the scope of model checking. Static analyzers often generate excessive false positives, which burden developers with manual validation. LLMs perform strongly on small code snippets; however, their accuracy declines, and hallucinations increase as code size grows. These findings suggest that, despite current limitations, LLMs hold significant promise for integration into code completion frameworks, providing real-time vulnerability prevention and marking an important step toward more secure software systems.

### III-X SPIN-Bench Benchmark

Yao et al. \[80\] introduce a comprehensive evaluation framework, SPIN-Bench, highlighting the challenges of strategic planning and social reasoning in AI agents. Unlike traditional benchmarks focused on isolated tasks, SPIN-Bench combines classical planning, competitive board games, cooperative card games, and negotiation scenarios to simulate real-world social interactions. This multifaceted approach reveals significant performance bottlenecks in current large language models (LLMs), which, while adept at factual retrieval and short-range planning, struggle with deep multi-hop reasoning, spatial inference, and socially coordinated decision-making. For instance, models perform reasonably well on simple tasks like Tic-Tac-Toe but falter in complex environments such as Chess or Diplomacy, and even the best models achieve only around 58.59% accuracy on classical planning tasks.

### III-Y τ\text{tau}-bench

Yao et al. \[81\] present τ\text{tau}-bench, a benchmark designed to evaluate language agents in realistic, dynamic, multi-turn conversational settings that emulate real-world environments. In τ\text{tau}-bench, agents are challenged to interact with a simulated user to understand needs, utilize domain-specific API tools (such as booking flights or returning items), and adhere to provided policy guidelines, while performance is measured by comparing the final database state with an annotated goal state. A novel metric, passk\text{pass}^{k}, is introduced to assess reliability over multiple trials. Experimental findings reveal that even state-of-the-art function-calling agents like GPT-4o succeed on less than 50% of tasks, with significant inconsistency (for example, pass8 scores below 25% in retail domains) and markedly lower success rates for tasks requiring multiple database writes. These results underscore the need for enhanced methods that improve consistency, adherence to rules, and overall reliability in language agents for real-world applications.

### III-Z Discussion and Comparison of LLM Benchmarks

Table IV presents an extensive overview of benchmarks developed from 2019 to 2025 for evaluating large language models (LLMs) concerning multimodal capabilities, task scope, diversity, reasoning, and agentic behaviors. Early benchmarks, such as DROP \[82\], MMLU \[58\], MATH \[83\], Codex \[84\], MGSM \[85\], FACTS Grounding \[61\], and SimpleQA \[66\], concentrated on core competencies like discrete reasoning, academic knowledge, mathematical problem solving, and factual grounding. These pioneering efforts lay the groundwork for performance evaluation in language understanding and reasoning tasks, setting a baseline against which later, more sophisticated benchmarks have been compared.

A notable progression in benchmark design is observed with the emergence of frameworks that target more complex agentic and multimodal tasks. For instance, PersonaGym \[86\] and FineTasks \[67\] introduce dynamic persona evaluation and multilingual task selection. GAIA \[78\] expands the evaluative scope to general AI assistant tasks while OmniDocBench \[63\] and ProcessBench \[62\] address document extraction and error detection in mathematical solutions. Further, MIRAI \[87\], AppWorld \[88\], VisualAgentBench \[89\], and ScienceAgentBench \[90\] explore various facets of multimodal and scientific discovery tasks. This decade-spanning evolution is complemented by additional evaluations focusing on safety (Agent-SafetyBench \[91\]), discovery (DiscoveryBench \[92\]), code generation (BLADE \[93\], Dyn-VQA \[8\], and Agent-as-a-Judge \[64\]), judicial reasoning (JudgeBench \[65\]), and clinical decision making (MedChain \[94\]), among others including FRAMES \[68\], CRAG \[72\], DIA \[74\], CyberMetric \[75\], TeamCraft \[95\], AgentHarm \[96\], τ\text{tau}-bench \[81\], LegalAgentBench \[97\], and GPQA \[98\].

Recent benchmarks from 2025 further indicate a substantial expansion in the depth and breadth of large language model (LLM) evaluations. ENIGMAEVAL \[57\] and ComplexFuncBench \[59\] target complex puzzles and function calling tasks, while MedAgentsBench \[99\] and Humanity’s Last Exam \[60\] focus on advanced medical reasoning and expert-level academic tasks. Additional benchmarks such as DABStep \[69\], BFCL v2 \[70\], SWE-Lancer \[71\], and OCCULT \[73\] further diversify evaluative criteria by incorporating multi-step reasoning, cybersecurity, and freelance software engineering challenges. The table also includes BIG-Bench Extra Hard \[76\], MultiAgentBench \[77\], CASTLE \[79\], EmbodiedEval \[100\], SPIN-Bench \[80\], OlympicArena \[101\], SciReplicate-Bench \[102\], EconAgentBench \[103\], VeriLA \[104\], CapaBench \[105\], AgentOrca \[106\], ProjectEval \[107\], RefactorBench \[108\], BEARCUBS \[109\], Robotouille \[110\], DSGBench \[111\], TheoremExplainBench \[112\], RefuteBench 2.0 \[113\], MLGym \[114\], DataSciBench \[115\], EmbodiedBench \[116\], BrowseComp \[117\], and MLE-bench \[119\]. Collectively, these benchmarks exemplify the field’s shift towards more comprehensive and nuanced evaluation metrics, supporting the development of LLMs that can tackle increasingly multifaceted, real-world challenges.

## IV AI Agents

This section presents a comprehensive overview of AI agent frameworks and applications developed between 2024 and 2025, highlighting transformative approaches that integrate large language models with modular tools to achieve autonomous decision-making and dynamic multi-step reasoning. The frameworks discussed include LangChain \[124\], LlamaIndex \[125\], CrewAI \[126\], and Swarm \[127\], which abstract complex functionalities into reusable components that enable context management, tool integration, and iterative refinement of outputs. Additionally, pioneering efforts in GUI control \[128\] and agentic reasoning \[129,130\] demonstrate the increasing capabilities of these systems to interact with external environments and tools in real-time.

In parallel, this section presents a diverse range of AI agent applications that span materials science, biomedical research, academic ideation, software engineering, synthetic data generation, and chemical reasoning. Systems such as the StarWhisper Telescope System \[132\] and HoneyComb \[133\] have revolutionized operational workflows by automating observational and analytical tasks in materials science. In the biomedical domain, platforms like GeneAgent \[134\] and frameworks such as PRefLexOR \[135\] demonstrate enhanced reliability through self-verification and iterative refinement. Moreover, innovative solutions for research ideation, exemplified by SurveyX \[136\] and Chain-of-Ideas \[137\], as well as specialized frameworks for synthetic data generation \[138\] and chemical reasoning \[139\], collectively underscore the significant strides made in leveraging autonomous AI agents for complex, real-world tasks.

## V Challenges and Open problems

As the capabilities of AI agents and large language models continue to grow, new challenges and open problems emerge that limit their effectiveness, reliability, and security \[220\]. In this section, we highlight several critical research directions, including advancing the reasoning abilities of AI agents, understanding the failure modes of multi-agent systems, supporting automated scientific discovery, enabling dynamic tool integration, reinforcing autonomous search capabilities, and addressing the vulnerabilities of emerging communication protocols.

## VI Conclusion

In this paper, we have surveyed recent advances in the reasoning capabilities of large language models (LLMs) and autonomous AI agents and highlighted the benefits of multi-step, intermediate processing for solving complex tasks in advanced mathematics, code generation, and logical reasoning. By exposing their internal reasoning through intermediate steps, models such as DeepSeek-R1, OpenAI o1 and o3, and GPT-4o achieve greater accuracy and reliability compared to direct-response approaches.

Researchers have developed various training and inference strategies to cultivate these reasoning abilities, including inference-time scaling, pure reinforcement learning (for example, DeepSeek-R1-Zero), supervised fine-tuning combined with reinforcement learning, and distillation-based fine-tuning. Adaptations of Qwen-32B and Llama-based architectures show that a balanced combination of these methods yields emergent reasoning behaviors while reducing overthinking and verbosity.

We also provided a unified comparison of state-of-the-art benchmarks from 2019 to 2025, together with a taxonomy of approximately 60 evaluation suites. Our analysis encompasses training frameworks, including mixture-of-experts, retrieval-augmented generation, and reinforcement learning, as well as architectural enhancements that drive performance improvements. In addition, we reviewed AI agent frameworks developed between 2023 and 2025 and illustrated their applications in domains including materials science, biomedical research, synthetic data generation, and financial forecasting.

Despite these successes, several challenges remain. Key open problems include automating multi-step reasoning without human oversight, balancing structured guidance with model flexibility, and integrating long-context retrieval at scale. Future research must address these challenges to unlock the full potential of autonomous AI agents.

Looking ahead, we anticipate an increasing focus on domain- and application-specific optimization. Early examples, such as DeepSeek-R1-Distill, Sky-T1, and TinyZero, demonstrate how specialized reasoning systems can achieve a favorable trade-off between performance and computational cost. Continued innovation in training methodologies, model architectures, and benchmarking will drive the next generation of high-efficiency, high-accuracy AI reasoning systems.

### Original URL
https://ar5iv.labs.arxiv.org/html/2504.19678
</details>

---
<details>
<summary>What is a ReAct Agent? | IBM</summary>

# What is a ReAct agent?

## Authors

Dave Bergmann

Senior Writer, AI Models

IBM

## What is a ReAct agent?

A ReAct agent is an AI agent that uses the “reasoning and acting” (ReAct) framework to combine chain of thought (CoT) reasoning with external tool use. The ReAct framework enhances the ability of a large language model (LLM) to handle complex tasks and decision-making in agentic workflows.

First introduced by Yao and others in the 2023 paper, “ReACT: Synergizing Reasoning and Acting in Language Models,” ReAct can be understood most generally as a machine learning (ML) paradigm to integrate the reasoning and action-taking capabilities of LLMs.

More specifically, ReAct is a conceptual framework for building AI agents that can interact with their environment in a structured but adaptable way, by using an LLM as the agent’s “brain” to coordinate anything from simple retrieval augmented generation (RAG) to intricate multiagent workflows.

Unlike traditional artificial intelligence (AI) systems, ReAct agents don’t separate decision-making from task execution. Therefore, the development of the ReAct paradigm was an important step in the evolution of generative AI (gen AI) beyond mere conversational chatbots and toward complex problem-solving.

ReAct agents and derivative approaches continue to power AI applications that can autonomously plan, execute and adapt to unforeseen circumstances.

## How do ReAct agents work?

The ReAct framework is inspired by the way humans can intuitively use natural language—often through our own inner monologue—in the step-by-step planning and execution of complex tasks.

Rather than implementing rule-based or otherwise predefined workflows, ReAct agents rely on their LLM’s reasoning capabilities to dynamically adjust their approach based on new information or the results of previous steps.

Imagine packing for a brief trip. You might start by identifying key considerations (“ _What will the weather be like while I’m there?_”), then actively consult external sources (“ _I’ll check the local weather forecast_”).

By using that new information (“ _It’s going to be cold_”), you determine your next consideration (“ _What warm clothes do I have?_”) and action (“ _I’ll check my closet_”). Upon taking that action, you might encounter an unexpected obstacle (“ _All of my warm clothes are in storage_”) and adjust your next step accordingly (“ _What clothes can I layer together?_”).

In a similar fashion, the ReAct framework uses prompt engineering to structure an AI agent’s activity in a formal pattern of alternating thoughts, actions and observations:

- The verbalized CoT reasoning steps ( _thoughts_) help the model decompose the larger task into more manageable subtasks.
- Predefined _actions_ enable the model to use tools, make application programming interface (API) calls and gather more information from external sources (such as search engines) or knowledge bases (such as an internal docstore).
- After taking an action, the model then reevaluates its progress and uses that _observation_ to either deliver a final answer or inform the next _thought_. The observation might ideally also consider prior information, whether from earlier in the model’s standard context window or from an external memory component.

Because the performance of a ReAct agent depends heavily on the ability of its central LLM to “verbally” think its way through complex tasks, ReAct agents benefit greatly from highly capable models with advanced reasoning and instruction-following ability.

To minimize cost and latency, a multiagent ReAct framework might rely primarily on a larger, more performant model to serve as the central agent whose reasoning process or actions might involve delegating subtasks to more agents built using smaller, more efficient models.

### ReAct agent loops

This framework inherently creates a feedback loop in which the model problem-solves by iteratively repeating this interleaved _thought-action-observation_ process.

Each time this loop is completed—that is, each time the agent has taken an action and made an observation based on the results of that action—the agent must then decide whether to repeat or end the loop.

When and how to end the reasoning loop is an important consideration in the design of a ReAct agent. Establishing a maximum number of loop iterations is a simple way to limit latency, costs and token usage, and avoid the possibility of an endless loop.

Conversely, the loop can be set to end when some specific condition is met, such as when the model has identified a potential final answer that exceeds a certain confidence threshold.

To implement this kind of reasoning and acting loop, ReAct agents typically use some variant of _ReAct prompting_, whether in the system prompt provided to the LLM or in the context of the user query itself.

## ReAct prompting

ReAct prompting is a specific prompting technique designed to guide an LLM to follow the ReAct paradigm of _thought_, _action_ and _observation_ loops. While the explicit use of conventional ReAct prompting methods is not strictly necessary to build a ReAct agent, most ReAct-based agents implement or at least take direct inspiration from it.

First outlined in the original ReAct paper, ReAct prompting’s primary function is to instruct an LLM to follow the ReAct loop and establish which tools can be used—that is, which actions can be taken—when handling user queries.

Whether through explicit instructions or the inclusion of few-shot examples, ReAct prompting should:

- **Guide the model to use chain of thought reasoning:** Prompt the model to reason its way through tasks by thinking step by step, interleaving thoughts with actions.
- **Define actions:** Establish the specific actions available to the model. An action might entail the generation of a specific type of next thought or subprompt but usually involves using external tools or making APIs.
- **Instruct the model to make observations:** Prompt the model to reassess its context after each action step and use that updated context to inform the next reasoning step.
- **Loop:** Instruct the model to repeat the previous steps if necessary. You could provide specific conditions for ending that loop, such as a maximum number of loops, or instruct the agent to end its reasoning process whenever it feels it has arrived at the correct final output.
- **Output final answer:** Whenever those end conditions have been met, provide the user with the final output in response to their initial query. As with many uses of LLMs, as reasoning models employing chain of thought reasoning before determining a final output, ReAct agents are often prompted to conduct their reasoning process within a “scratchpad.”

A classic demonstration of ReAct prompting is the system prompt for the prebuilt `ZERO_SHOT_REACT-DESCRIPTION`  
ReAct agent module in Langchain’s LangGraph. It’s called “zero-shot” because, with this predefined system prompt, the LLM being used with the module does not need any further examples to behave as a ReAct agent.

```
Answer the following questions as best you can. You have access to the following tools:

Wikipedia: A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query.
duckduckgo_search: A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events. Input should be a search query.
Calculator: Useful for when you need to answer questions about math.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Wikipedia, duckduckgo_search, Calculator]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
```

## Benefits of ReAct agents

The introduction of the ReAct framework was an important step in the advancement of LLM-driven agentic workflows. From grounding LLMs in real time, real-world external information through (RAG) to contributing to subsequent breakthroughs—such as Reflexion, which led to modern reasoning models—ReAct has helped catalyze the use of LLMs for tasks well beyond text generation.

The utility of ReAct agents is drawn largely from some of the inherent qualities of the ReAct framework:

- **Versatility:** ReAct agents can be configured to work with a wide variety of external tools and APIs. Though fine-tuning relevant ReAct prompts (using relevant tools) can improve performance, no prior configuration of the model is required to execute tool calls.
- **Adaptability:** This versatility, along with the dynamic and situational nature of how they determine the appropriate tool or API to call, means that ReAct agents can use their reasoning process to adapt to new challenges. Especially when operating within a lengthy context window or augmented with external memory, they can learn from past mistakes and successes to tackle unforeseen obstacles and situations. This makes ReAct agents flexible and resilient.
- **Explainability:** The verbalized reasoning process of a ReAct agent is simple to follow, which facilitates debugging and helps make them relatively user-friendly to build and optimize.
- **Accuracy:** As the original ReAct paper asserts, chain of thought (CoT) reasoning alone has many benefits for LLMs, but also runs an increased risk of hallucination. ReAct’s combination of CoT with a connection external to information sources significantly reduces hallucinations, making ReAct agents more accurate and trustworthy.

## ReAct agents vs. function calling

Another prominent paradigm for agentic AI is function calling, originally introduced by OpenAI in June 2023 to supplement the agentic abilities of its GPT models.

The function calling paradigm entails fine-tuning models to recognize when a particular situation should result in a tool call and output a structured JSON object containing the arguments necessary to call those functions.

Many proprietary and open source LLM families, including IBM® Granite®, Meta’s Llama series, Anthropic’s Claude and Google Gemini, now support function calling.

Whether ReAct or function calling is “better” will generally depend on the nature of your specific use case. In scenarios involving relatively straightforward (or at least predictable) tasks, function calling can execute faster, save tokens, and be simpler to implement than a ReAct agent.

In such circumstances, the number of tokens that would be spent on a ReAct agent’s iterative loop of CoT reasoning might be seen as inefficient.

The inherent tradeoff is a relative lack of ability to customize how and when the model chooses which tool to use. Likewise, when an agent handles tasks that call for complex reasoning, or scenarios that are dynamic or unpredictable, the rigidity of function calling might limit the agent’s adaptability. In such situations, it’s often beneficial to be able to view the step-by-step reasoning that led to a specific tool call.

## Getting started with ReAct agents

ReAct agents can be designed and implemented in multiple ways, whether coded from scratch in Python or developed with the help of open source frameworks such as BeeAI. The popularity and staying power of the ReAct paradigm have yielded extensive literature and tutorials for ReAct agents on GitHub and other developer communities.

As an alternative to developing custom ReAct agents, many agentic AI frameworks, including BeeAI, LlamaIndex and LangChain’s LangGraph, offer preconfigured ReAct agent modules for specific use cases.

### Original URL
https://www.ibm.com/think/topics/react-agent
</details>

---
<details>
<summary>What is AI Agent Planning? | IBM</summary>

# What is AI agent planning?

## What is AI agent planning?

AI agent planning refers to the process by which an artificial intelligence (AI) agent determines a sequence of actions to achieve a specific goal. It involves decision-making, goal prioritization and action sequencing, often using various planning algorithms and frameworks.

AI agent planning is a module common to many types of agents that exists alongside other modules such as perception, reasoning, decision-making, action, memory, communication and learning. Planning works in conjunction with these other modules to help ensure that agents achieve outcomes desired by their designers.

Not all agents can plan. Unlike simple reactive agents that respond immediately to inputs, planning agents anticipate future states and generate a structured action plan before execution. This makes AI planning essential for automation tasks that require multistep decision-making, optimization and adaptability.

## How AI agent planning works

Advances in large language models (LLMs) such as OpenAI’s GPT and related techniques involving machine learning algorithms resulted in the generative AI (gen AI) boom of recent years, and further advancements have led to the emerging field of autonomous agents.

By integrating tools, APIs, hardware interfaces and other external resources, agentic AI systems are increasingly autonomous, capable of real-time decision-making and adept at problem-solving across various use cases.

Complex agents can’t act without making a decision, and they can’t make good decisions without first making a plan. Agentic planning consists of several key components that work together to encourage optimal decision-making.

### Goal definition

The first and most critical step in AI planning is defining a clear objective. The goal serves as the guiding principle for the agent’s decision-making process, determining the end state it seeks to achieve. Goals can either be static, remaining unchanged throughout the planning process, or dynamic, adjusting based on environmental conditions or user interactions.

For instance, a self-driving car might have a goal of reaching a specific destination efficiently while adhering to safety regulations. Without a well-defined goal, an agent would lack direction, leading to erratic or inefficient behavior.

If the goal is complex, agentic AI models will break it down into smaller, more manageable sub-goals in a process called task decomposition. This allows the system to focus on complex tasks in a hierarchical manner.

LLMs play a vital role in task decomposition, breaking down a high-level goal into smaller subtasks and then executing those subtasks through various steps. For instance, a user might ask a chatbot with a natural language prompt to plan a trip.

The agent would first decompose the task into components such as booking flights, finding hotels and planning an itinerary. Once decomposed, the agent can use application programming interfaces (APIs) to fetch real-time data, check pricing and even suggest destinations.

### State representation

To plan effectively, an agent must have a structured understanding of its environment. This understanding is achieved through state representation, which models the current conditions, constraints and contextual factors that influence decision-making.

Agents have some built-in knowledge from their training data or datasets representing previous interactions, but perception is required for agents to have a real-time understanding of their environment. Agents collect data through sensory input, allowing it to model its environment, along with user input and data describing its own internal state.

The complexity of state representation varies depending on the task. For example, in a chess game, the state includes the position of all pieces on the board, while in a robotic navigation system, the state might involve spatial coordinates, obstacles and terrain conditions.

The accuracy of state representation directly impacts an agent’s ability to make informed decisions, as it determines how well the agent can predict the outcomes of its actions.

### Action sequencing

Once the agent has established its goal and assessed its environment, it must determine a sequence of actions that will transition it from its current state to the desired goal state. This process, known as action sequencing, involves structuring a logical and efficient set of steps that the agent must follow.

The agent needs to identify potential actions, reduce that list to optimal actions, prioritize them and identifying dependencies between actions and conditional steps based on potential changes in the environment. The agent might allocate resources to each step in the sequence, or schedule actions based on environmental constraints.

For example, a robotic vacuum cleaner needs to decide the most effective path to clean a room, ensuring it covers all necessary areas without unnecessary repetition. If the sequence of actions is not well planned, the AI agent might take inefficient or redundant steps, leading to wasted resources and increased execution time.

The ReAct framework is a methodology used in AI for handling dynamic decision-making. In the ReAct framework, reasoning refers to the cognitive process where the agent determines what actions or strategies are required to achieve a specific goal.

This phase is similar to the planning phase in agentic AI, where the agent generates a sequence of steps to solve a problem or fulfill a task. Other emerging frameworks include ReWOO, RAISE and Reflexion, each of which has its own strengths and weaknesses.

### Optimization and evaluation

AI planning often involves selecting the most optimal path to achieving a goal, especially when multiple options are available. Optimization helps ensure that an agent's chosen sequence of actions is the most efficient, cost-effective or otherwise beneficial given the circumstances. This process often requires evaluating different factors such as time, resource consumption, risks and potential rewards.

For example, a warehouse robot tasked with retrieving items must determine the shortest and safest route to avoid collisions and reduce operational time. Without proper optimization, AI agents might execute plans that are functional but suboptimal, leading to inefficiencies. Several methods can be used to optimize decision-making, including:

#### Heuristic search

Heuristic search algorithms help agents find optimal solutions by estimating the best path toward a goal. These algorithms rely on heuristic functions—mathematical estimates of how close a given state is to the desired goal. Heuristic searches are particularly effective for structured environments where agents need to find optimal paths quickly.

#### Reinforcement learning

Reinforcement learning enables agents to optimize planning through trial and error, learning which sequences of actions lead to the best outcomes over time. An agent interacts with an environment, receives feedback in the form of rewards or penalties, and refines its strategies accordingly.

#### Probabilistic planning

In real-world scenarios, AI agents often operate in uncertain environments where outcomes are not deterministic. Probabilistic planning methods account for uncertainty by evaluating multiple possible outcomes and selecting actions with the highest expected utility.

### Collaboration

Single agent planning is one thing, but in a multiagent system, AI agents must work autonomously while interacting with each other to achieve individual or collective goals.

The planning process for AI agents in a multiagent system is more complex than for a single agent because agents must not only plan their own actions but also consider the actions of other agents and how their decisions interact with those of others.

Depending on the agentic architecture, each agent in the system typically has its own individual goals, which might involve accomplishing specific tasks or maximizing a reward function. In many multiagent systems, agents need to work together to achieve shared goals.

These goals could be defined by an overarching system or emerge from the agents’ interactions. Agents need mechanisms to communicate and align their goals, especially in cooperative scenarios. This could be done through explicit messaging, shared task definitions or implicit coordination.

Planning in multiagent systems can be centralized, where a single entity or controller—likely an LLM agent—generates the plan for the entire system.

Each agent receives instructions or plans from this central authority. It can also be decentralized, where agents generate their own plans but work collaboratively to help ensure that they align with each other and contribute to global objectives, often requiring communication and negotiation.

This collaborative decision-making process enhances efficiency, reduces biases in task execution, helps to avoid hallucinations through cross-validation and consensus-building and encourages the agents to work toward a common goal.

## After planning

The phases in agentic AI workflows do not always occur in a strict step-by-step linear fashion. While these phases are often distinct in conceptualization, in practice, they are frequently interleaved or iterative, depending on the nature of the task and the complexity of the environment in which the agent operates.

AI solutions can differ depending on their design, but in a typical agentic workflow, the next phase after planning is action execution, where the agent carries out the actions defined in the plan. This involves performing tasks and interacting with external systems or knowledge bases with retrieval augmented generation (RAG), tool use and function calling (tool calling).

Building AI agents for these capabilities might involve LangChain. Python scripts, JSON data structures and other programmatic tools enhance the AI’s ability to make decisions.

After executing plans, some agents can use memory to learn from their experiences and iterate their behavior accordingly.

In dynamic environments, the planning process must be adaptive. Agents continuously receive feedback about the environment and other agents’ actions and must adjust their plans accordingly. This might involve revising goals, adjusting action sequences, or adapting to new agents entering or leaving the system.

When an agent detects that its current plan is no longer feasible (for example, due to a conflict with another agent or a change in the environment), it might engage in replanning to adjust its strategy. Agents can adjust their strategies using chain of thought reasoning, a process where they reflect on the steps needed to reach their objective before taking action.

### Original URL
https://www.ibm.com/think/topics/ai-agent-planning
</details>

