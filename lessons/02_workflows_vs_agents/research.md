# Research based on provided article guidelines

## Research Results

---

<details>
<summary>What are the key differences between LLM workflows and agentic systems in recent AI architecture literature and technical documentation?</summary>

### Source: https://www.anthropic.com/research/building-effective-agents
Workflows are described as systems where LLMs and tools are orchestrated through predefined code paths. In these, tasks are decomposed into a sequence of steps—often using prompt chaining—where each LLM call processes the output of the previous step, potentially with programmatic checks to validate intermediate results. This approach is well-suited for tasks that can be broken into fixed subtasks and where predictability and control are desired.

Agentic systems, by contrast, build on augmented LLMs but move beyond fixed workflows. They grant the LLM autonomy to generate its own search queries, select tools, determine what information to retain, and make decisions about how to proceed with tasks. Instead of following a rigid, step-by-step process, agentic systems can adaptively plan and execute actions based on context, user input, and dynamic requirements. This results in higher flexibility and the possibility for emergent behavior, as the agent can optimize its actions in real time rather than follow a static script.

</details>

---

<details>
<summary>How do leading AI companies (Anthropic, OpenAI, Google, etc.) define and implement agentic systems versus LLM workflows in their products and research as of 2025?</summary>

### Source: https://www.fanktank.ch/en/blog/choosing-ai-models-openai-anthropic-google-2025
In 2025, Anthropic, OpenAI, and Google have each advanced their approaches to integrating agentic systems and LLM (large language model) workflows. Anthropic’s release of Claude 4 introduced significant improvements in reasoning and multi-step task execution. Claude 4 is described as more “agentic,” not just generating text but engaging in planning, taking actions, and interacting autonomously with tools and APIs. This agentic capability is distinguished from traditional LLM workflows, where the model primarily responds to prompts in a single, stateless exchange. OpenAI’s latest GPT models have also become more agentic, offering features for multi-step, autonomous task completion and integration with external applications. Google’s Gemini similarly emphasizes agentic features, enabling autonomous workflows and deep integration with business logic, APIs, and productivity tools. In summary, all three companies define agentic systems as AI that can plan, act, and interact autonomously with software and data, while LLM workflows are more focused on generating text or answers in response to direct prompts without persistent context or goal-driven behavior.

-----

### Source: https://www.zdnet.com/article/google-joins-openai-in-adopting-anthropics-protocol-for-connecting-ai-agents-why-it-matters/
In April 2025, Google joined OpenAI in adopting Anthropic’s Model Context Protocol (MCP). MCP is an open-source protocol designed to improve AI agent interoperability and access to data. It enables AI agents—defined here as assistants capable of executing a range of tasks, autonomously or semi-autonomously—to connect to data stores, developer spaces, and business applications. The protocol streamlines integration, allowing agents to access information from sources like Google Drive, GitHub, and Slack without custom connectors for each system. This approach contrasts with classic LLM workflows, which generally require manual data preparation or integration for each use case. By supporting MCP, Google, OpenAI, and Anthropic highlight a shift toward standardized, agentic systems that can autonomously access and utilize external resources, while traditional LLM workflows remain more constrained to single-session, prompt-response interactions.

-----

</details>

---

<details>
<summary>What are the current state-of-the-art AI agents (e.g., Deep Research agents, Devin, Codex) and their common operational mechanisms such as planning, tool use, and memory?</summary>

### Source: https://cognition.ai/blog/introducing-devin
Devin distinguishes itself through a range of autonomous capabilities:
- It can learn to use unfamiliar technologies, adapt to new tools, and build and deploy applications end-to-end.
- Devin autonomously finds and fixes bugs, as demonstrated in benchmarks where it sets up environments, reproduces issues, codes, tests fixes, and compiles reports without external guidance.
- In the SWE-bench benchmark, Devin correctly resolved 13.86% of real-world GitHub issues end-to-end, significantly outperforming prior state-of-the-art models, which managed 1.96% (unassisted) and 4.80% (with file-level hints).
- Devin’s operation is unassisted, handling context gathering and setup from minimal initial input (such as just a GitHub link), demonstrating advanced planning, tool use (environment setup, testing tools), and memory (maintaining context over multi-step tasks).

-----

### Source: https://en.wikipedia.org/wiki/Devin_AI
Devin AI, developed by Cognition Labs, is an autonomous AI assistant specialized in software engineering:
- It performs tasks such as coding, debugging, planning, and problem-solving through advanced machine learning techniques.
- The user initiates tasks via natural language prompts; Devin responds by presenting its plan, executing code, and adjusting its actions based on user feedback or newly identified issues.
- During execution, Devin searches online resources to learn or resolve uncertainties, demonstrating tool use and adaptive learning.
- Devin can create websites and complete complex projects rapidly, such as building websites or compiling computer vision models.
- Benchmarks show Devin fixes 13.86% of real-world issues without human help, outperforming prior models.
- Later versions introduced multi-agent operations, where Devin can delegate tasks to other AI agents, and self-assessment capabilities, prompting clarification from users when uncertain.
- In 2025, Devin introduced features like Devin Wiki (automated documentation) and Devin Search (codebase QA), enhancing its operational memory and knowledge retrieval abilities.

-----

</details>

---

<details>
<summary>What are the major design and orchestration challenges faced when building agentic systems compared to LLM workflows, especially regarding reliability, scalability, and cost?</summary>

### Source: https://sam-solutions.com/blog/llm-agent-architecture/
Agentic LLM systems introduce a range of design and orchestration challenges, particularly in reliability, scalability, and cost, compared to traditional LLM workflows. Unlike a single LLM query, which typically involves a single model call and takes only a few seconds, agentic systems are resource-intensive: an autonomous agent may perform dozens of model calls and tool invocations to accomplish a complex task. This significantly impacts scalability, as the infrastructure must handle many concurrent and often interdependent operations.

The agentic architecture involves not only the LLM backbone but also additional components that enable the system to plan, act, observe, and reflect. The orchestration layer manages the workflow, sequencing actions and handling feedback loops, which adds further complexity. These multi-stage cycles inherently increase the chances of failures, require robust monitoring, and demand careful management to maintain reliability.

Moreover, as agentic systems continuously interact with various tools and data sources, orchestration must account for error handling, state management, and recovery from partial failures, which are less prominent in simple LLM workflows. The increased number of operations and integration points also amplifies operational costs and complicates cost management, since each action—whether a model call or a tool invocation—incurs additional computational expense.

In summary, agentic systems face major challenges in:
- Scaling efficiently due to high resource demands
- Ensuring reliability across multiple interdependent components and steps
- Managing escalating costs from frequent model calls and tool interactions

These challenges are far less pronounced in single-step LLM workflows, which are simpler to orchestrate, scale, and maintain from a cost perspective.

-----

</details>

---

<details>
<summary>What architectural patterns and best practices are recommended for engineers choosing between LLM workflows and more autonomous agentic systems in AI product development?</summary>

### Source: https://guides.library.cmu.edu/LLM_best_practices
This resource from Carnegie Mellon University highlights best practices for integrating AI tools, including large language models (LLMs), into academic and professional workflows. Key recommendations for engineers building LLM workflows or agentic systems include:

- Assessing the specific needs of the workflow to determine whether a simple LLM pipeline or a more autonomous agentic system is appropriate.
- Ensuring modularity in workflow design to facilitate updates, maintenance, and scalability.
- Emphasizing data quality and transparency, as LLMs and agentic systems depend on robust, well-curated datasets for reliable outcomes.
- Incorporating strong monitoring and human-in-the-loop review systems to ensure results remain accurate, especially as models evolve.
- Applying ethical guidelines and bias mitigation strategies throughout the development and deployment process.
- Maintaining clear documentation of AI workflows, including decision points, model selection, and evaluation metrics.

These practices help teams balance efficiency, scalability, and accountability when choosing and architecting between LLM workflows and more autonomous agentic systems.

-----

### Source: https://www.maxiomtech.com/large-language-model-architecture/
This guide outlines technical best practices and architectural patterns for LLM-based systems:

- **Modular Design:** Architect systems with modular components to allow easier scaling, maintenance, and replacement of individual parts.
- **Asynchronous Processing:** Use asynchronous data processing to improve throughput and efficiency, especially for high-volume or latency-sensitive tasks.
- **Data Pipeline Optimization:** Streamline data pipelines to reduce latency and improve data flow, which is essential for real-time or near-real-time LLM applications.
- **Dynamic Resource Allocation:** Employ strategies that allocate computing resources based on demand, helping optimize costs and performance.
- **Continuous Integration/Continuous Deployment (CI/CD):** Integrate CI/CD practices to support regular, reliable updates and rapid iteration.

For LLM workflows, the guide recommends leveraging robust ML frameworks (such as TensorFlow or PyTorch), pre-trained model libraries (like Hugging Face Transformers), and cloud platforms (AWS SageMaker, Google Cloud AI Platform) to ensure flexibility, scalability, and speed from development to production.

-----

### Source: https://www.zenml.io/blog/llm-agents-in-production-architectures-challenges-and-best-practices
This article delves into production-grade architectures for LLM agents, discussing the key differences and best practices for LLM workflows versus agentic systems:

- **LLM Workflows:** Typically involve sequential or modular pipelines where LLMs perform specific, well-defined tasks. Best practices include clear separation of workflow components, robust error handling, and continuous monitoring to ensure reliability and scalability.
- **Agentic Systems:** Require more autonomy, involving multi-step reasoning, state management, and sometimes multi-agent orchestration. Best practices here include:
  - Designing for explicit memory and state tracking to handle complex, multi-turn tasks.
  - Building modular, composable architectures to allow agents to interact flexibly with tools, APIs, and other models.
  - Implementing centralized observability for debugging and monitoring agent behavior.
  - Prioritizing security and access control, since agentic systems may interact autonomously with sensitive systems or data.

The article emphasizes the importance of modularity, observability, robust error handling, and continuous evaluation in both paradigms but notes that agentic systems require additional safeguards and architectural flexibility to manage their higher autonomy.

-----

### Source: https://www.crossml.com/llm-orchestration-in-the-real-world/
This resource addresses best practices for orchestrating LLM workflows and agentic systems in real-world production environments:

- **Modular Pipeline Architecture:** Designing workflows as modular pipelines allows for scalability, easier maintenance, and clearer separation of concerns.
- **Custom Embeddings:** For domain-specific tasks, generating custom embeddings improves the relevance and precision of LLM outputs.
- **Robust Error Handling:** Implement robust error handling to deal with the inherent unpredictability of LLM outputs and external API integrations.
- **Resource Optimization:** Dynamically manage and optimize compute resources to maintain performance and control costs, especially in multi-model or agentic environments.
- **Continuous Monitoring:** Deploy monitoring tools to track model performance, detect errors, and trigger human intervention as needed.
- **Separation of Workflow Components:** Clearly separate different components (e.g., prompt management, response aggregation, tool calls) for transparency and maintainability.
- **Multi-Agent Systems:** For agentic architectures, use multi-agent systems, continuous feedback loops, and context maintenance to ensure agents can collaborate and maintain coherence over complex tasks.

The source highlights that orchestration is crucial for combining multiple AI components, maintaining context, and delivering consistent, reliable results in production, particularly for agentic systems and advanced LLM applications.

-----

</details>

---

<details>
<summary>Which open-source or commercial frameworks (as of 2025) are most used for orchestrating LLM workflows versus agentic systems, and what distinguishing features do they offer for each?</summary>

### Source: https://research.aimultiple.com/llm-orchestration/
AIMultiple provides a comprehensive comparison of the top LLM orchestration frameworks as of 2025, highlighting both open-source and commercial solutions. The most widely used frameworks include:

- **LangChain** (83.8k GitHub stars): Best suited for complex AI workflows, supporting both Python and JavaScript. LangChain is renowned for its modular approach to building LLM-driven applications, allowing for seamless integration, prompt chaining, and workflow management.
- **AutoGen** (38.7k): Focused on multi-agent coordination in Python, making it ideal for agentic systems that require orchestrating multiple autonomous LLM agents to collaborate or compete on tasks.
- **LlamaIndex** (31.2k): Specializes in data integration, supporting both Python and TypeScript. It is particularly effective when connecting LLMs to custom or proprietary data sources, enabling retrieval-augmented generation.
- **crewAI** (25.9k): Designed for role-based agent orchestration in Python, making it a preferred choice for agentic systems where LLM agents assume specialized roles within a workflow.
- **Semantic Kernel by Microsoft** (22.9k): Supports C# and Python, optimized for Azure environments and enterprise-grade orchestration, emphasizing security and scalability.
- **Haystack by Deepset AI** (19k): Focused on building custom NLP pipelines in Python, with a strong emphasis on retrieval, document processing, and integrations for search and question answering.
- **TaskWeaver, Agency Swarm, Microchain, Loft, and IBM watsonx orchestrate**: These frameworks offer specialized features such as agent-based task automation, AI agent networks, lightweight microservices, no-code/low-code automation, and enterprise-level orchestration.

Notably, LangChain and LlamaIndex are most popular for general LLM workflow orchestration, while AutoGen, crewAI, and Agency Swarm are distinguished for agentic, multi-agent systems. Commercial solutions like IBM watsonx orchestrate cater to enterprise use cases, focusing on integration, compliance, and scalability.

-----

### Source: https://labelyourdata.com/articles/llm-orchestration
Label Your Data explains the technical aspects and strategy behind LLM orchestration frameworks. Key features associated with these frameworks include:

- **Workflow Automation**: Orchestration frameworks manage the handoff and sequencing between different LLMs or AI components, automating complex tasks such as data extraction, summarization, and reasoning.
- **Prompt Chaining**: Allows passing context and intermediate outputs between models, which is vital for building multi-step AI workflows.
- **API Integration**: Facilitates connecting LLMs to external data sources, tools, or APIs, ensuring smooth data flow and real-time responsiveness.
- **Monitoring and Adjustment**: Provides dashboards and logs for real-time performance tracking, enabling dynamic prompt tuning and model replacement.
- **Use Case Example**: In a chatbot setting, a smaller LLM might handle routine queries, while a more advanced model tackles complex reasoning tasks. The orchestrator manages which model is used for each query, ensuring efficiency and quality.

These frameworks are essential for streamlining AI development, particularly as applications increasingly require both traditional workflow automation and agentic, multi-agent coordination.

-----

### Source: https://www.upsilonit.com/blog/top-ai-frameworks-and-llm-libraries
UpsilonIT's review of top AI frameworks and LLM libraries in 2025 highlights three major players:

- **LlamaIndex**: Noted for its strength in data integration and retrieval-augmented generation, LlamaIndex is commonly used to connect LLMs with structured or unstructured data sources, making it a go-to framework for applications that require dynamic, data-driven responses.
- **LangChain**: Recognized for its ability to construct complex, modular LLM workflows, LangChain is frequently employed for prompt chaining, workflow automation, and integrating LLMs with external APIs or data stores.
- **Haystack**: Focused on building custom NLP pipelines, Haystack is particularly strong in search, document processing, and question answering, supporting both traditional LLM workflows and more sophisticated retrieval-augmented agentic systems.

These frameworks distinguish themselves by offering robust tools for chaining, orchestration, and extensibility, catering to both conventional LLM workflow orchestration and modern agentic system design.

</details>

---

<details>
<summary>How do state-of-the-art agentic systems (e.g., Deep Research agents, computer control agents like Operator) address the compounding error, consistency, and security challenges unique to multi-step, autonomous operation?</summary>

### Source: https://openai.com/index/introducing-deep-research/
OpenAI’s Deep Research agent is designed to autonomously synthesize large amounts of online information and complete multi-step research tasks. It utilizes a specialized version of the OpenAI o3 model, which is optimized for reasoning and web browsing. The agent’s architecture is tailored to handle complex, multi-stage tasks by continuously gathering and analyzing new information as it proceeds, helping to mitigate compounding errors by verifying and updating its understanding throughout the process. Deep Research also documents its research process, provides explicit citations, and details its thought process, which enhances consistency and transparency in its outputs. The model’s training includes real-world tasks that require not just browsing but also reasoning and data analysis, aiding in robust autonomous operation over extended task sequences.

-----

### Source: https://aisecuritychronicles.org/a-comparison-of-deep-research-ai-agents-52492ee47ca7
Deep Research agents, such as those from OpenAI and Google’s Gemini, perform multi-step reasoning by autonomously formulating search queries, browsing content, analyzing data, and synthesizing findings into structured reports with citations. There are two main architectural approaches:

- Fully Autonomous Agents: These agents operate end-to-end without human intervention, making decisions about research direction and information verification on their own. This autonomy requires robust internal mechanisms for error correction and consistency checking, as there is no external oversight during operation.
- Human-in-the-Loop (HITL) Agents: These agents periodically pause their workflow for human review, particularly after planning or outlining research. This plan review stage acts as a quality control checkpoint, allowing users to catch and correct potential errors early, improving output reliability and consistency.

The choice of approach impacts how agents address compounding errors and consistency challenges. Fully autonomous agents rely on their internal verification and reasoning capabilities, while HITL agents benefit from human oversight to catch and prevent errors during multi-step operations.

-----

</details>

---

<details>
<summary>What are the detailed differences in memory management, context retention, and feedback loops between advanced agentic systems (e.g., Deep Research, Devin) and traditional LLM workflows?</summary>

### Source: https://markovate.com/blog/agentic-ai-architecture/
Agentic AI architecture, as described here, is organized into five interconnected layers that collectively enable advanced memory management, context retention, and feedback loops:

- The **Data Storage & Retrieval Layer** is specifically designed for efficient data management. It employs centralized and distributed repositories, vector stores for rapid retrieval, and knowledge graphs for contextual reasoning. This marks a significant advancement over traditional LLM workflows, which generally rely on limited context windows and ephemeral memory for each prompt.

- In terms of **context retention**, agentic systems utilize knowledge graphs and vector stores to maintain both short- and long-term context, allowing them to reason across sessions and tasks. This enables more sophisticated context awareness compared to traditional LLMs, which often lack persistent memory between interactions.

- **Feedback loops** in agentic systems are realized through the Agent Orchestration Layer, which coordinates multi-agent collaboration and supports self-evaluation, performance monitoring, and continuous improvement. Specialized agents handle planning and execution, while others focus on self-evaluation and incremental learning, creating a dynamic feedback mechanism for ongoing refinement.

- The architecture also integrates safeguards for governance, safety, and iterative validation, ensuring both compliance and continuous improvement, features not typically found in traditional LLM workflows.

Overall, agentic systems are structured for adaptive learning, dynamic decision-making, and continuous feedback, making them more autonomous and context-aware than standard LLM approaches.

-----

### Source: https://hyperdev.substack.com/p/agentic-coding-emerging-tools
Later versions of Devin introduced **multi-agent collaboration**, where Devin can spin up additional, specialized AI agents that work together on complex tasks. This architecture allows for distributed problem-solving and parallelization, which supports advanced forms of memory management and task-specific context retention. Each specialized agent can retain local context relevant to its function, while the orchestrating agent maintains an overview, coordinating state and knowledge across agents.

Traditional LLM workflows, by contrast, are generally stateless between interactions—each prompt is treated in isolation, with limited or no memory of previous exchanges. The multi-agent approach in advanced agentic systems like Devin enables persistent context across tasks, better division of responsibilities, and more robust feedback loops, since agents can critique, monitor, and refine each other’s outputs in a coordinated fashion.

-----

### Source: https://www.youtube.com/watch?v=KfXq9s96tPU
A deep dive into Devin’s architecture reveals several advanced mechanisms for memory management, context retention, and feedback:

- **Memory Management and Context Retention**: Devin employs techniques beyond basic Retrieval Augmented Generation (RAG). For each query, Devin performs preprocessing, advanced filtering, and re-ranking, including multi-hop search. This ensures the system retrieves highly relevant context, which can comprise both source files and documentation (e.g., wiki pages), combining micro and macro context for grounded, accurate outputs.

- **Feedback Loops**: Post-training and reinforcement learning (RL) are leveraged to further optimize Devin for specific domains. The architecture allows for iterative refinement through self-evaluation and RL-based feedback, enabling the system to adapt and improve its performance over time, particularly in narrow domains.

- **Comparison with Traditional LLMs**: Traditional LLM workflows typically use a fixed context window and stateless interactions, with RAG providing only limited contextual augmentation. In contrast, Devin’s architecture includes sophisticated context retrieval and dynamic adaptation, enabling more persistent and relevant memory across interactions.

-----

### Source: https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-5-multi-agent-collaboration/
Multi-agent collaboration is described as a key agentic design pattern. In such systems, multiple specialized agents work together, each with their own state and memory pertaining to their specific tasks. The orchestrating mechanism coordinates these agents, facilitating information sharing and ongoing context retention across the group.

This collaborative setup enables advanced feedback loops, where agents can critique, verify, and build upon each other’s results. In traditional LLM workflows, such persistent, multi-agent memory and internal feedback mechanisms are typically absent—each prompt is processed independently, and context is limited to the prompt’s content and immediate history.

Agentic systems, by contrast, leverage these feedback and memory mechanisms to achieve higher reliability, adaptability, and performance in complex, real-world tasks.

</details>

---

<details>
<summary>What are standard best practices for error mitigation and recovery in multi-step, autonomous agentic systems to prevent compounding errors?</summary>

### Source: https://blog.kodezi.com/effective-error-mitigation-techniques-strategies-and-best-practices/
Best practices for error mitigation in multi-step, autonomous agentic systems emphasize several core strategies:

- **Detailed Documentation:** Comprehensive records of potential attack vectors and corresponding countermeasures are critical. This ensures teams understand and can respond to known and anticipated errors systematically.
- **Real-Time System Monitoring:** Continuous monitoring allows for the rapid detection and correction of faults before they escalate, supporting proactive intervention rather than reactive fixes.
- **Resilience via Microservices:** Building systems as collections of microservices fosters resilience, as individual components can fail or be updated independently, reducing the likelihood of cascading failures.
- **Performance Evaluations:** Regular assessments of system performance help identify weaknesses and inform improvements, thereby reducing the probability of compounding errors.
- **Chaos Engineering:** Structured experiments that introduce failures under controlled conditions help teams understand system behavior under stress, providing insights that inform the design of robust mitigation and recovery strategies.
- **Assurance Cases:** These are structured arguments supported by evidence, demonstrating that the system meets its dependability requirements, which is especially vital for autonomous and AI-driven systems.

Collectively, these practices focus on anticipating, detecting, and correcting errors before they develop into significant system failures, thus preventing error compounding in autonomous agentic workflows.

### Source: https://www.numberanalytics.com/blog/securing-robotics-and-autonomous-systems
Key best practices for error mitigation and recovery in autonomous multi-step systems include:

- **Modular Design:** By structuring systems into independent modules, faults can be isolated, limiting the impact of any single failure and allowing for targeted recovery or replacement of faulty modules.
- **Fault-Tolerant Control Systems:** Implementing control algorithms such as model predictive control and adaptive control enables the system to detect faults and adapt its behavior dynamically, maintaining operational continuity.
- **Sensor and Actuator Redundancy:** Employing duplicate or diverse sensors and actuators ensures that a failure in one does not lead to system-wide breakdowns. This redundancy allows the system to cross-validate data and continue functioning even as individual components fail.

These approaches support robust fault detection, isolation, and recovery, reducing the risk of compounding errors in multi-step autonomous systems.

### Source: https://arxiv.org/pdf/2307.05203
Best practices for error mitigation in complex, multi-step systems, particularly in quantum and agentic workflows, include:

- **Composite Error Mitigation Strategies:** Combining multiple techniques—such as readout mitigation, random gate compilation (“twirling”), and zero-noise extrapolation (dZNE)—addresses different error types and improves robustness.
- **Calibration and Customization:** Regularly calibrating settings for error mitigation (e.g., noise factors, extrapolation functions) ensures strategies remain effective as system and environmental conditions evolve.
- **Targeted Approaches for Specific Error Types:** Augmenting general mitigation strategies with specialized techniques (such as readout error correction for SPAM errors) tailors error handling to the system’s unique vulnerabilities.
- **Continuous Evaluation and Adaptation:** Ongoing studies and adaptations to specific system configurations, noise profiles, and operational contexts are critical for maintaining effective error mitigation and preventing error accumulation across steps.

These practices emphasize a multi-layered and adaptive approach to error mitigation and recovery, crucial for avoiding the propagation of errors through autonomous, multi-step processes.
-----

</details>

---

<details>
<summary>What open-source libraries or frameworks provide state-of-the-art support for orchestrating hybrid LLM workflow/agentic systems and visualizing their execution in 2025?</summary>

### Source: https://www.pracdata.io/p/state-of-workflow-orchestration-ecosystem-2025
This source provides a comprehensive analysis of the open-source workflow orchestration ecosystem in 2025, emphasizing the convergence between traditional workflow engines and LLM-specific orchestration frameworks.

Key highlights include:
- The rise of hybrid workflow/agentic systems, where multiple LLMs, traditional code, and other AI agents interact within complex pipelines.
- Several open-source orchestrators, such as LangChain, AutoGen, and crewAI, are specifically mentioned as leaders in handling LLM-centric agent workflows. These tools feature execution tracing, stepwise debugging, and interactive visualization dashboards to monitor workflow progress and agent decision paths.
- The ecosystem also includes traditional orchestration tools (like Airflow and Prefect) that are increasingly integrating LLM components. However, the LLM-native orchestrators (LangChain, AutoGen, crewAI) offer more advanced visualization and agent coordination capabilities tailored for the hybrid paradigm.
- Visualization is a core differentiator in 2025, with leading frameworks offering graph-based interfaces, agent activity logs, and real-time execution trees to provide transparency and debuggability for complex LLM-powered workflows.

-----

### Source: https://research.aimultiple.com/llm-orchestration/
This source compares the top LLM orchestration frameworks in 2025, focusing on open-source tools with state-of-the-art support for hybrid LLM/agentic workflows and execution visualization.

The leading frameworks highlighted are:

- **LangChain**: With 83.8k GitHub stars, LangChain is praised for its support for complex AI workflows, including hybrid orchestrations, multi-agent coordination, and detailed execution visualization. It supports both Python and JavaScript.
- **AutoGen**: With 38.7k stars, AutoGen is designed for multi-agent coordination, supporting the orchestration of various agentic systems and offering visualization tools for agent interactions and workflow steps.
- **crewAI**: With 25.9k stars, crewAI specializes in role-based agent orchestration and visual tracing of agent roles, task assignments, and execution flows.
- **LlamaIndex**: With 31.2k stars, LlamaIndex focuses on data integration but also includes workflow orchestration and visualization features.
- **Haystack by Deepset AI**: While primarily a custom NLP pipeline framework, Haystack offers orchestration of LLM components and visual pipeline editors for workflow inspection.

The frameworks provide APIs and interfaces for visualizing execution, inspecting agent decisions, and monitoring workflow states, making them state-of-the-art for hybrid LLM workflow and agentic system orchestration in 2025.

-----

### Source: https://airbyte.com/top-etl-tools-for-sources/data-orchestration-tools
This source discusses open-source data orchestration tools relevant for AI and LLM workflow management in 2025. Two notable tools are:

- **MLRun**: A Python-based orchestration tool designed for managing machine learning pipelines and data workflows. It features elastic scaling, real-time data processing, and detailed workflow management. MLRun provides tracking, automation, and deployment support for complex pipelines, including those integrating AI/LLM components. Its workflow visualization tools enable users to monitor data and execution flows across hybrid systems.
- **Metaflow**: Developed at Netflix, Metaflow is an open-source data orchestration framework using a dataflow programming paradigm. Programs are represented as directed graphs (“flows”), allowing for visualization of operations, sequences, branches, and dynamic iterations. Metaflow’s artifact management and visualization features simplify monitoring and debugging of complex workflows, making it suitable for hybrid LLM and agentic system orchestration.

-----

</details>

---

<details>
<summary>What are notable case studies or industry examples (2024–2025) that demonstrate hybrid AI systems blending LLM workflows with agentic autonomy, and what practical lessons have emerged from these deployments?</summary>

### Source: https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage
McKinsey highlights the emergence of agentic AI, which combines large language models (LLMs) with autonomous agent workflows to solve a broad array of business problems. Notable examples from 2024–2025 include deployments in both vertical and horizontal use cases, such as automated legal research, dynamic marketing campaign management, and supply chain optimization. 

Key lessons learned from these deployments include:
- **Balance Between Autonomy and Control:** Successful implementations require careful alignment of agent autonomy with organizational risk tolerance and compliance needs.
- **Iterative Use Case Fine-Tuning:** Organizations often start with pilot projects, iteratively refining the scope and guardrails as agentic systems prove their reliability.
- **Trust and Explainability:** Businesses that prioritize transparency in agent decision-making and maintain clear audit trails see higher adoption and trust among users.
- **Integration with Human Workflows:** Agentic AI is most effective when designed to augment, not replace, human expertise, with seamless hand-offs between agents and people.
- **Scalability Challenges:** Scaling agentic AI across business units presents organizational and technical challenges, including data integration and model retraining needs.

These case studies demonstrate that agentic AI can deliver significant value by automating complex, multistep tasks, but practical rollouts require a nuanced understanding of organizational context and rigorous change management.

-----

### Source: https://edrm.net/2025/06/from-prompters-to-partners-the-rise-of-agentic-ai-in-law-and-professional-practice/
In the legal sector, EDRM reports that law firms in 2024–2025 are leveraging agentic AI systems that blend LLM-driven reasoning with autonomous agents. These systems perform iterative legal research, validate case law, draft and compile arguments, and even recommend litigation strategies.

Case studies show that agentic AI is being used to:
- **Accelerate Document Review:** AI agents autonomously sift through large volumes of discovery documents, flagging relevant content and cross-referencing with case precedents.
- **Enhance Argument Construction:** LLMs generate draft arguments, which agentic systems iteratively refine by consulting legal databases and adapting to new information.
- **Continuous Legal Monitoring:** Agents provide real-time updates on regulatory changes and case law, automatically alerting attorneys to potential impacts on active matters.

Practical lessons include the necessity of robust validation pipelines to ensure accuracy, human oversight for critical decisions, and the importance of transparency for client trust. Early adopters also report efficiency gains but stress the need for ongoing training and adaptation of both the technology and legal professionals.

-----

### Source: https://ctomagazine.com/agentic-ai-in-enterprise/
CTO Magazine examines the cost and ROI trade-offs of deploying agentic AI in enterprise environments. Real-world examples from 2024–2025 showcase agentic systems in sectors including finance, logistics, and HR.

Highlights include:
- **Finance:** Agentic AI automates compliance monitoring, risk analysis, and fraud detection, reducing manual workload and increasing accuracy.
- **Logistics:** Autonomous agents optimize inventory, route planning, and supplier negotiation, reacting in real time to disruptions.
- **HR:** Systems personalize onboarding and training, adapting content to individual employee profiles and learning speeds.

Lessons learned:
- **ROI Realism:** Enterprises find that the initial promise of agentic autonomy must be balanced with the real costs of integration, customization, and governance.
- **Change Management:** Employee acceptance is critical; successful deployments include strong change management and clear communication on how agentic AI augments human roles.
- **Continuous Evaluation:** Ongoing performance monitoring and iterative tuning are necessary to maintain system relevance and effectiveness.

-----

### Source: https://arxiv.org/html/2506.02153v1
This arXiv preprint discusses the transition from large language models to small language models (SLMs) in agentic AI, with several short case studies from 2024–2025. The case studies estimate the extent to which SLMs can replace LLMs in agentic workflows without sacrificing autonomy or task performance.

Findings include:
- **Task-Specific SLMs:** In customer support and basic information retrieval, SLM-based agents achieve comparable results to LLMs at lower cost and latency.
- **Hybrid Architectures:** Workflows with LLM “overseers” delegating subtasks to SLM agents prove effective in settings with resource constraints or privacy requirements.
- **Cost Efficiency:** Organizations report significant reductions in computational costs and environmental impact when shifting suitable tasks to SLMs.

Lessons emphasize the importance of task decomposition, modular architecture, and careful cost–benefit analysis when designing hybrid agentic AI systems.

-----

### Source: https://www.ibm.com/think/insights/agentic-ai
IBM outlines several industry applications of agentic AI that combine LLM-driven intelligence with autonomous agent workflows. Notable 2024–2025 examples include:

- **Marketing:** Agents autonomously manage campaigns, monitor performance, and adjust strategies in real time.
- **Healthcare:** Systems monitor patient data, update treatment recommendations, and provide clinicians with actionable feedback.
- **Cybersecurity:** Agents continuously analyze network activity, detect anomalies, and respond to threats without direct human intervention.
- **Supply Chain:** Agentic AI autonomously places orders and adjusts production schedules based on real-time inventory and demand data.
- **Human Resources:** Personalized onboarding and training paths are generated and dynamically refined for each new hire.

Practical lessons highlight that agentic AI can operate continuously without constant human oversight, maintain long-term goals, and manage multistep tasks. Successful deployments often involve replacing or augmenting traditional SaaS tools, allowing users to interact with complex systems through natural language interfaces, streamlining workflows, and improving overall efficiency.
-----

</details>

---

## Sources Scraped From Research Results

---
<details>
<summary>Building Effective AI Agents \ Anthropic</summary>

[Engineering at Anthropic](https://www.anthropic.com/engineering)

![](https://www-cdn.anthropic.com/images/4zrzovbb/website/039b6648c28eb33070a63a58d49013600b229238-2554x2554.svg)

# Building effective agents

Published Dec 19, 2024

We've worked with dozens of teams building LLM agents across industries. Consistently, the most successful implementations use simple, composable patterns rather than complex frameworks.

Over the past year, we've worked with dozens of teams building large language model (LLM) agents across industries. Consistently, the most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns.

In this post, we share what we’ve learned from working with our customers and building agents ourselves, and give practical advice for developers on building effective agents.

## What are agents?

"Agent" can be defined in several ways. Some customers define agents as fully autonomous systems that operate independently over extended periods, using various tools to accomplish complex tasks. Others use the term to describe more prescriptive implementations that follow predefined workflows. At Anthropic, we categorize all these variations as **agentic systems**, but draw an important architectural distinction between **workflows** and **agents**:

- **Workflows** are systems where LLMs and tools are orchestrated through predefined code paths.
- **Agents**, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks.

Below, we will explore both types of agentic systems in detail. In Appendix 1 (“Agents in Practice”), we describe two domains where customers have found particular value in using these kinds of systems.

## When (and when not) to use agents

When building applications with LLMs, we recommend finding the simplest solution possible, and only increasing complexity when needed. This might mean not building agentic systems at all. Agentic systems often trade latency and cost for better task performance, and you should consider when this tradeoff makes sense.

When more complexity is warranted, workflows offer predictability and consistency for well-defined tasks, whereas agents are the better option when flexibility and model-driven decision-making are needed at scale. For many applications, however, optimizing single LLM calls with retrieval and in-context examples is usually enough.

## When and how to use frameworks

There are many frameworks that make agentic systems easier to implement, including:

- [LangGraph](https://langchain-ai.github.io/langgraph/) from LangChain;
- Amazon Bedrock's [AI Agent framework](https://aws.amazon.com/bedrock/agents/);
- [Rivet](https://rivet.ironcladapp.com/), a drag and drop GUI LLM workflow builder; and
- [Vellum](https://www.vellum.ai/), another GUI tool for building and testing complex workflows.

These frameworks make it easy to get started by simplifying standard low-level tasks like calling LLMs, defining and parsing tools, and chaining calls together. However, they often create extra layers of abstraction that can obscure the underlying prompts ​​and responses, making them harder to debug. They can also make it tempting to add complexity when a simpler setup would suffice.

We suggest that developers start by using LLM APIs directly: many patterns can be implemented in a few lines of code. If you do use a framework, ensure you understand the underlying code. Incorrect assumptions about what's under the hood are a common source of customer error.

See our [cookbook](https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/agents) for some sample implementations.

## Building blocks, workflows, and agents

In this section, we’ll explore the common patterns for agentic systems we’ve seen in production. We'll start with our foundational building block—the augmented LLM—and progressively increase complexity, from simple compositional workflows to autonomous agents.

### Building block: The augmented LLM

The basic building block of agentic systems is an LLM enhanced with augmentations such as retrieval, tools, and memory. Our current models can actively use these capabilities—generating their own search queries, selecting appropriate tools, and determining what information to retain.

![](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2Fd3083d3f40bb2b6f477901cc9a240738d3dd1371-2401x1000.png&w=3840&q=75)The augmented LLM

We recommend focusing on two key aspects of the implementation: tailoring these capabilities to your specific use case and ensuring they provide an easy, well-documented interface for your LLM. While there are many ways to implement these augmentations, one approach is through our recently released [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol), which allows developers to integrate with a growing ecosystem of third-party tools with a simple [client implementation](https://modelcontextprotocol.io/tutorials/building-a-client#building-mcp-clients).

For the remainder of this post, we'll assume each LLM call has access to these augmented capabilities.

### Workflow: Prompt chaining

Prompt chaining decomposes a task into a sequence of steps, where each LLM call processes the output of the previous one. You can add programmatic checks (see "gate” in the diagram below) on any intermediate steps to ensure that the process is still on track.

![](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F7418719e3dab222dccb379b8879e1dc08ad34c78-2401x1000.png&w=3840&q=75)The prompt chaining workflow

**When to use this workflow:** This workflow is ideal for situations where the task can be easily and cleanly decomposed into fixed subtasks. The main goal is to trade off latency for higher accuracy, by making each LLM call an easier task.

**Examples where prompt chaining is useful:**

- Generating Marketing copy, then translating it into a different language.
- Writing an outline of a document, checking that the outline meets certain criteria, then writing the document based on the outline.

### Workflow: Routing

Routing classifies an input and directs it to a specialized followup task. This workflow allows for separation of concerns, and building more specialized prompts. Without this workflow, optimizing for one kind of input can hurt performance on other inputs.

![](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F5c0c0e9fe4def0b584c04d37849941da55e5e71c-2401x1000.png&w=3840&q=75)The routing workflow

**When to use this workflow:** Routing works well for complex tasks where there are distinct categories that are better handled separately, and where classification can be handled accurately, either by an LLM or a more traditional classification model/algorithm.

**Examples where routing is useful:**

- Directing different types of customer service queries (general questions, refund requests, technical support) into different downstream processes, prompts, and tools.
- Routing easy/common questions to smaller models like Claude 3.5 Haiku and hard/unusual questions to more capable models like Claude 3.5 Sonnet to optimize cost and speed.

### Workflow: Parallelization

LLMs can sometimes work simultaneously on a task and have their outputs aggregated programmatically. This workflow, parallelization, manifests in two key variations:

- **Sectioning**: Breaking a task into independent subtasks run in parallel.
- **Voting:** Running the same task multiple times to get diverse outputs.

![](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F406bb032ca007fd1624f261af717d70e6ca86286-2401x1000.png&w=3840&q=75)The parallelization workflow

**When to use this workflow:** Parallelization is effective when the divided subtasks can be parallelized for speed, or when multiple perspectives or attempts are needed for higher confidence results. For complex tasks with multiple considerations, LLMs generally perform better when each consideration is handled by a separate LLM call, allowing focused attention on each specific aspect.

**Examples where parallelization is useful:**

- **Sectioning**:
  - Implementing guardrails where one model instance processes user queries while another screens them for inappropriate content or requests. This tends to perform better than having the same LLM call handle both guardrails and the core response.
  - Automating evals for evaluating LLM performance, where each LLM call evaluates a different aspect of the model’s performance on a given prompt.
- **Voting**:
  - Reviewing a piece of code for vulnerabilities, where several different prompts review and flag the code if they find a problem.
  - Evaluating whether a given piece of content is inappropriate, with multiple prompts evaluating different aspects or requiring different vote thresholds to balance false positives and negatives.

### Workflow: Orchestrator-workers

In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results.

![](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F8985fc683fae4780fb34eab1365ab78c7e51bc8e-2401x1000.png&w=3840&q=75)The orchestrator-workers workflow

**When to use this workflow:** This workflow is well-suited for complex tasks where you can’t predict the subtasks needed (in coding, for example, the number of files that need to be changed and the nature of the change in each file likely depend on the task). Whereas it’s topographically similar, the key difference from parallelization is its flexibility—subtasks aren't pre-defined, but determined by the orchestrator based on the specific input.

**Example where orchestrator-workers is useful:**

- Coding products that make complex changes to multiple files each time.
- Search tasks that involve gathering and analyzing information from multiple sources for possible relevant information.

### Workflow: Evaluator-optimizer

In the evaluator-optimizer workflow, one LLM call generates a response while another provides evaluation and feedback in a loop.

![](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F14f51e6406ccb29e695da48b17017e899a6119c7-2401x1000.png&w=3840&q=75)The evaluator-optimizer workflow

**When to use this workflow:** This workflow is particularly effective when we have clear evaluation criteria, and when iterative refinement provides measurable value. The two signs of good fit are, first, that LLM responses can be demonstrably improved when a human articulates their feedback; and second, that the LLM can provide such feedback. This is analogous to the iterative writing process a human writer might go through when producing a polished document.

**Examples where evaluator-optimizer is useful:**

- Literary translation where there are nuances that the translator LLM might not capture initially, but where an evaluator LLM can provide useful critiques.
- Complex search tasks that require multiple rounds of searching and analysis to gather comprehensive information, where the evaluator decides whether further searches are warranted.

### Agents

Agents are emerging in production as LLMs mature in key capabilities—understanding complex inputs, engaging in reasoning and planning, using tools reliably, and recovering from errors. Agents begin their work with either a command from, or interactive discussion with, the human user. Once the task is clear, agents plan and operate independently, potentially returning to the human for further information or judgement. During execution, it's crucial for the agents to gain “ground truth” from the environment at each step (such as tool call results or code execution) to assess its progress. Agents can then pause for human feedback at checkpoints or when encountering blockers. The task often terminates upon completion, but it’s also common to include stopping conditions (such as a maximum number of iterations) to maintain control.

Agents can handle sophisticated tasks, but their implementation is often straightforward. They are typically just LLMs using tools based on environmental feedback in a loop. It is therefore crucial to design toolsets and their documentation clearly and thoughtfully. We expand on best practices for tool development in Appendix 2 ("Prompt Engineering your Tools").

![](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F58d9f10c985c4eb5d53798dea315f7bb5ab6249e-2401x1000.png&w=3840&q=75)Autonomous agent

**When to use agents:** Agents can be used for open-ended problems where it’s difficult or impossible to predict the required number of steps, and where you can’t hardcode a fixed path. The LLM will potentially operate for many turns, and you must have some level of trust in its decision-making. Agents' autonomy makes them ideal for scaling tasks in trusted environments.

The autonomous nature of agents means higher costs, and the potential for compounding errors. We recommend extensive testing in sandboxed environments, along with the appropriate guardrails.

**Examples where agents are useful:**

The following examples are from our own implementations:

- A coding Agent to resolve [SWE-bench tasks](https://www.anthropic.com/research/swe-bench-sonnet), which involve edits to many files based on a task description;
- Our [“computer use” reference implementation](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo), where Claude uses a computer to accomplish tasks.

![](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F4b9a1f4eb63d5962a6e1746ac26bbc857cf3474f-2400x1666.png&w=3840&q=75)High-level flow of a coding agent

## Combining and customizing these patterns

These building blocks aren't prescriptive. They're common patterns that developers can shape and combine to fit different use cases. The key to success, as with any LLM features, is measuring performance and iterating on implementations. To repeat: you should consider adding complexity _only_ when it demonstrably improves outcomes.

## Summary

Success in the LLM space isn't about building the most sophisticated system. It's about building the _right_ system for your needs. Start with simple prompts, optimize them with comprehensive evaluation, and add multi-step agentic systems only when simpler solutions fall short.

When implementing agents, we try to follow three core principles:

1. Maintain **simplicity** in your agent's design.
2. Prioritize **transparency** by explicitly showing the agent’s planning steps.
3. Carefully craft your agent-computer interface (ACI) through thorough tool **documentation and testing**.

Frameworks can help you get started quickly, but don't hesitate to reduce abstraction layers and build with basic components as you move to production. By following these principles, you can create agents that are not only powerful but also reliable, maintainable, and trusted by their users.

### Acknowledgements

Written by Erik Schluntz and Barry Zhang. This work draws upon our experiences building agents at Anthropic and the valuable insights shared by our customers, for which we're deeply grateful.

## Appendix 1: Agents in practice

Our work with customers has revealed two particularly promising applications for AI agents that demonstrate the practical value of the patterns discussed above. Both applications illustrate how agents add the most value for tasks that require both conversation and action, have clear success criteria, enable feedback loops, and integrate meaningful human oversight.

### A. Customer support

Customer support combines familiar chatbot interfaces with enhanced capabilities through tool integration. This is a natural fit for more open-ended agents because:

- Support interactions naturally follow a conversation flow while requiring access to external information and actions;
- Tools can be integrated to pull customer data, order history, and knowledge base articles;
- Actions such as issuing refunds or updating tickets can be handled programmatically; and
- Success can be clearly measured through user-defined resolutions.

Several companies have demonstrated the viability of this approach through usage-based pricing models that charge only for successful resolutions, showing confidence in their agents' effectiveness.

### B. Coding agents

The software development space has shown remarkable potential for LLM features, with capabilities evolving from code completion to autonomous problem-solving. Agents are particularly effective because:

- Code solutions are verifiable through automated tests;
- Agents can iterate on solutions using test results as feedback;
- The problem space is well-defined and structured; and
- Output quality can be measured objectively.

In our own implementation, agents can now solve real GitHub issues in the [SWE-bench Verified](https://www.anthropic.com/research/swe-bench-sonnet) benchmark based on the pull request description alone. However, whereas automated testing helps verify functionality, human review remains crucial for ensuring solutions align with broader system requirements.

## Appendix 2: Prompt engineering your tools

No matter which agentic system you're building, tools will likely be an important part of your agent. [Tools](https://www.anthropic.com/news/tool-use-ga) enable Claude to interact with external services and APIs by specifying their exact structure and definition in our API. When Claude responds, it will include a [tool use block](https://docs.anthropic.com/en/docs/build-with-claude/tool-use#example-api-response-with-a-tool-use-content-block) in the API response if it plans to invoke a tool. Tool definitions and specifications should be given just as much prompt engineering attention as your overall prompts. In this brief appendix, we describe how to prompt engineer your tools.

There are often several ways to specify the same action. For instance, you can specify a file edit by writing a diff, or by rewriting the entire file. For structured output, you can return code inside markdown or inside JSON. In software engineering, differences like these are cosmetic and can be converted losslessly from one to the other. However, some formats are much more difficult for an LLM to write than others. Writing a diff requires knowing how many lines are changing in the chunk header before the new code is written. Writing code inside JSON (compared to markdown) requires extra escaping of newlines and quotes.

Our suggestions for deciding on tool formats are the following:

- Give the model enough tokens to "think" before it writes itself into a corner.
- Keep the format close to what the model has seen naturally occurring in text on the internet.
- Make sure there's no formatting "overhead" such as having to keep an accurate count of thousands of lines of code, or string-escaping any code it writes.

One rule of thumb is to think about how much effort goes into human-computer interfaces (HCI), and plan to invest just as much effort in creating good _agent_-computer interfaces (ACI). Here are some thoughts on how to do so:

- Put yourself in the model's shoes. Is it obvious how to use this tool, based on the description and parameters, or would you need to think carefully about it? If so, then it’s probably also true for the model. A good tool definition often includes example usage, edge cases, input format requirements, and clear boundaries from other tools.
- How can you change parameter names or descriptions to make things more obvious? Think of this as writing a great docstring for a junior developer on your team. This is especially important when using many similar tools.
- Test how the model uses your tools: Run many example inputs in our [workbench](https://console.anthropic.com/workbench) to see what mistakes the model makes, and iterate.
- [Poka-yoke](https://en.wikipedia.org/wiki/Poka-yoke) your tools. Change the arguments so that it is harder to make mistakes.

While building our agent for [SWE-bench](https://www.anthropic.com/research/swe-bench-sonnet), we actually spent more time optimizing our tools than the overall prompt. For example, we found that the model would make mistakes with tools using relative filepaths after the agent had moved out of the root directory. To fix this, we changed the tool to always require absolute filepaths—and we found that the model used this method flawlessly.

### Original URL
https://www.anthropic.com/research/building-effective-agents
</details>

---
<details>
<summary>Cognition | Introducing Devin, the first AI software engineer</summary>

# Introducing Devin, the first AI software engineer

Setting a new state of the art on the SWE-bench coding benchmark. Meet Devin, the world’s first fully autonomous AI software engineer.

Devin is a tireless, skilled teammate, equally ready to build alongside you or independently complete tasks for you to review.

With Devin, engineers can focus on more interesting problems and engineering teams can strive for more ambitious goals.

## Devin's Capabilities

With our advances in long-term reasoning and planning, Devin can plan and execute complex engineering tasks requiring thousands of decisions. Devin can recall relevant context at every step, learn over time, and fix mistakes.

We've also equipped Devin with common developer tools including the shell, code editor, and browser within a sandboxed compute environment—everything a human would need to do their work.

Finally, we've given Devin the ability to actively collaborate with the user. Devin reports on its progress in real time, accepts feedback, and works together with you through design choices as needed.‍Here's a sample of what Devin can do:

### Devin can learn how to use unfamiliar technologies.

After reading a blog post, Devin runs ControlNet on Modal to produce images with concealed messages for Sara.

### Devin can build and deploy apps end to end

Devin makes an interactive website which simulates the Game of Life! It incrementally adds features requested by the user and then deploys the app to Netlify.

### Devin can autonomously find and fix bugs in codebases

Devin helps Andrew maintain and debug his open source competitive programming book.

### Devin can train and fine tune its own AI models

Devin sets up fine tuning for a large language model given only a link to a research repository on GitHub.

### Devin can address bugs and feature requests in open source repositories

Given just a link to a GitHub issue, Devin does all the setup and context gathering that is needed.

### Devin can contribute to mature production repositories.

This example is part of the SWE-bench benchmark. Devin solves a bug with logarithm calculations in the sympy Python algebra system. Devin sets up the code environment, reproduces the bug, and codes and tests the fix on its own.

### We even tried giving Devin real jobs on Upwork and it could do those too!

Here, Devin writes and debugs code to run a computer vision model. Devin samples the resulting data and compiles a report at the end.

## Devin's Performance

We evaluated Devin on [SWE-bench](https://swebench.com/), a challenging benchmark that asks agents to resolve real-world GitHub issues found in open source projects like Django and scikit-learn.

Devin correctly resolves 13.86%\* of the issues end-to-end, far exceeding the previous state-of-the-art of 1.96%. Even when given the exact files to edit, the best previous models can only resolve 4.80% of issues.

[![](https://cdn.sanity.io/images/2mc9cv2v/production/5dd1cd4fd86149ed2cf4d8ab605f99707040615e-1600x858.png)](https://cdn.sanity.io/images/2mc9cv2v/production/5dd1cd4fd86149ed2cf4d8ab605f99707040615e-1600x858.png)

Devin was evaluated on a random 25% subset of the dataset. Devin was unassisted, whereas all other models were assisted (meaning the model was told exactly which files need to be edited).

We plan to publish a more detailed technical report soon—stay tuned for more details.

### Original URL
https://cognition.ai/blog/introducing-devin
</details>

---
<details>
<summary>Agentic LLM Architecture: How It Works, Types, Key Applications | SaM Solutions</summary>

# Agentic LLM Architecture: A Comprehensive Guide

[Anastasiya Paharelskaya](https://sam-solutions.com/author/a-paharelskaya/)

Updated May 27, 2025

[AI & Machine Learning](https://sam-solutions.com/blog/category/ai-ml/)

Large Language Models (LLMs) have revolutionized AI by enabling natural language interactions and content generation. Now, a new paradigm is emerging: agentic LLM architectures, which transform static models into dynamic AI agents capable of autonomously planning and executing tasks. Instead of simply responding to a single prompt, an _agentic_ LLM can proactively break down complex problems, call on external tools or data sources, and iterate towards a goal with minimal human intervention.

In this comprehensive guide, we will explore what agentic LLM architecture means, how it works under the hood, and what components are needed to build these advanced [AI systems](https://sam-solutions.com/services/ai-software-development/).

**Get AI software built for your business** by SaM Solutions — and start seeing results.

[Explore services](https://sam-solutions.com/services/ai-software-development/?utm_source=blog&utm_medium=post_ID_18087&utm_campaign=cta_post_content_16272)

## **What Is Agentic LLM Architecture**

Agentic LLM architecture is the structured design approach that enables an LLM to function as an autonomous intelligent module rather than a passive model. In a traditional setup, an LLM takes input text and generates an output text – it has no true “agency” beyond that one-shot response. By contrast, an _agentic_ architecture gives the LLM a degree of agency, meaning it can make decisions, take actions, and adjust its behavior to achieve a goal. Practically, this involves surrounding the LLM with additional modules and a workflow that allow it to plan, remember, and interact with its environment.

## **How Agentic LLM Architecture Works**

Agentic LLM systems work by orchestrating a continuous cycle of **planning, action, observation, and reflection** around the LLM.  Agent based LLM architecture works by giving the LLM a structured way to **act autonomously**: it can set goals (intentional planning), carry out tasks step by step (action with self-monitoring), and adapt based on the outcomes (reflection). This stands in contrast to a non-agentic LLM which would have stopped at producing a single answer with no follow-up.

![](https://sam-solutions.com/wp-content/uploads/infographic-1@2x-3.webp)

## **Core Components of Agentic LLM Systems**

There are multiple core components working together, each handling a critical aspect of agency. Below are the key building blocks that typically make up an agentic LLM system:

### LLM backbone

The base large language model that drives understanding and generation (e.g. GPT-4, an open-source LLM, etc.).

### Agent orchestration layer

The control logic that sequences the LLM’s actions and manages the workflow (essentially, the “agent loop” manager).

### Memory modules

Systems for storing and retrieving information so the intelligent module can remember context, including both short-term conversation context and long-term knowledge.

### Planning & reasoning engine

The mechanism that allows the intelligent module to reason through tasks and formulate multi-step solutions (often leveraging the LLM for chain-of-thought planning).

### Tool integration

Interfaces that let the intelligent module use external tools or APIs – extending its capabilities beyond what the LLM knows or can do alone.

### Feedback & learning mechanisms

Components that provide the AI system with feedback on its actions (from the environment or a critic) and enable it to adjust or improve (which can include self-evaluation loops).

### Task decomposition module

A specialized function that breaks complex tasks into smaller sub-tasks, helping the AI system tackle problems step by step.

### Communication protocols

The defined methods by which LLM-based entities communicate, either with other LLM-based entities (in a multi-intelligent module system) or with external systems, ensuring information is exchanged in a structured way.

## **Agentic vs Non-Agentic LLM Architectures**

Agentic vs non-agentic isn’t a binary good/bad choice, but rather a question of the right tool for the task. Many enterprise solutions might start with a non-agentic approach (for simplicity) and only graduate to an agentic approach if needed for greater capability.

## **Types of Agentic LLM Architectures**

Within agentic architectures themselves, there are different structural patterns. The main distinction is between single-agent architectures and multi-agent architectures.

### Single-agent LLM architectures

A single-agent architecture involves just one autonomous intelligent module handling the tasks at hand. This one intelligent module is responsible for all planning, tool use, and interactions with the environment. It makes decisions in a centralized way.

### Multi-agent LLM architectures

Multi-agent architectures involve multiple AI systems working either collaboratively or in a coordinated fashion to achieve a goal. This design can mirror a team of humans: each intelligent module might have a role or specialty, and they communicate with one another as needed.

![](https://sam-solutions.com/wp-content/uploads/infographic-2@2x-3.webp)

## **Key Frameworks for Agentic LLM Architectures**

In the realm of agent design, researchers often speak of different frameworks or paradigms for how LLM-based entities operate. Three broad categories are commonly referenced: Reactive, Deliberative, and Cognitive architectures. While these originated in classical AI agent theory, they map well onto LLM agent designs.

### Reactive LLM agents

A reactive agent is one that maps situations directly to actions, without any deeper reasoning or internal planning. Reactive LLM-based entities operate on a stimulus-response basis: given an input or an environmental trigger, the intelligent module produces an immediate output or action. There is no concept of an explicit memory of past events or an explicit consideration of future consequences in the agent’s decision process.

### Deliberative LLM agents

A deliberative intelligent module goes a step further by incorporating internal modeling, reasoning, and planning. Deliberative LLM-based entities are what we mostly describe when we talk about agentic LLMs: they consider the broader context, they think through the problem (often via chain-of-thought reasoning), and they plan a course of action rather than just reacting.

### Cognitive LLM agents

Cognitive agent architectures are the most advanced category, aiming to emulate human-like cognition, including not just reasoning and planning but also learning, memory consolidation, and adaptation over time. Cognitive LLM AI systems incorporate elements of perception, memory, learning, and meta-reasoning. They are deliberative at their core, but also capable of evolving their behavior and knowledge.

## **Planning in Agentic LLM Systems**

Planning is at the core of agentic behavior, but not all planning approaches are the same. One key distinction is whether the agent’s plan is fixed upfront or whether it evolves with feedback from the environment as the intelligent module operates. We can think of these as two modes: planning without feedback and planning with feedback.

### Planning without feedback

In a “plan then act” approach, an agent formulates a complete or partial plan at the beginning of its task and then executes it step by step without revisiting the plan after every action. Essentially, the agent isn’t checking intermediate results to alter the overall plan unless something goes seriously wrong.

### Planning with feedback

Most sophisticated autonomous LLM systems incorporate feedback into their planning loop. This means after each action (or after a group of actions), the agent looks at what happened and updates its plan or strategy accordingly. It’s an iterative sense–plan–act cycle.

**Leverage AI to transform your business** with custom solutions from SaM Solutions’ expert developers.

[View offer](https://sam-solutions.com/services/ai-software-development/?utm_source=blog&utm_medium=post_ID_18087&utm_campaign=cta_post_content_16263)

## **Memory and Tools in Agentic LLMs**

Memory and tool use are two pillars that significantly enhance an LLM agent’s capabilities. Memory allows the agent to retain and recall information, while tools let it interact with the world and acquire new information or perform actions. We’ll differentiate short-term vs long-term memory and then discuss the essential tools that empower LLM AI systems.

### Short-term vs long-term memory

The difference in practice: short-term memory is everything the agent juggles right now, whereas long-term memory is an archive it can draw upon when needed. Agentic architectures typically use STM to maintain coherence within a single task or conversation, and LTM to carry knowledge across tasks or provide a large knowledge base to draw from.

An illustrative analogy: When writing an article, your short-term memory is the outline and the paragraphs you’re working on – immediately visible and editable. Your long-term memory is the library of information you’ve read about the topic, which you might go back to for reference. Likewise, an agent might keep its current plan and recent results in short-term memory, but if it needs a piece of info it saw earlier (like “What was the user’s request yesterday?” or “Where did I save that intermediate calculation?”), it turns to long-term memory storage to retrieve it.

### Essential tools for LLM agents

Just as a human uses tools (calculator, web browser, software applications) to augment their abilities, LLM-based entities rely on tools to extend beyond what they can do with text alone. Here are some essential categories of tools commonly integrated into LLM AI systems:

- **Search and information retrieval:** Perhaps the most common tool is a web search engine or a document retrieval system. This gives the agent access to fresh information and data outside its training corpus. For example, an agent can use a search tool to fetch the latest news or look up a specific fact (like “current stock price of X” or “the population of Sweden”). In enterprise settings, retrieval might mean querying an internal knowledge base or SharePoint repository.

- **Databases and query engines:** Many tasks require retrieving structured data (sales figures, inventory levels, patient records, etc.). LLM-based entities are equipped with database connectors or query tools (SQL query interface, ElasticSearch queries, etc.) as tools. The agent might translate a natural language question into a database query, execute it, and then work with the result.

- **Calculators and code interpreters:** LLMs are not great at exact math or running complex logical procedures internally (they can approximate small math, but they make mistakes). So, a fundamental tool is a calculator or the ability to execute code. Platforms like OpenAI have introduced “code interpreter” functionalities which essentially let the agent write and run code (Python, for example) to solve problems. This means if the agent needs to do number crunching, parse a CSV file, or any algorithmic task, it can offload that to a sandboxed code execution environment. Even a simple arithmetic tool to multiply or do date calculations is extremely useful to avoid arithmetic errors.

## **Applications of Agentic LLM Architectures**

The concepts of autonomous LLMs come to life in various real-world applications. Here we’ll explore a few key areas where these architectures are being applied or have strong potential:

### Enterprise AI agents

In an enterprise environment, AI LLM-based entities can act as intelligent assistants or autonomous process executors across a range of business functions. The autonomous architecture is particularly valuable here because enterprise tasks often involve multiple steps, complex decision-making, and interaction with various data sources.

### Autonomous research agents

Another exciting application area is using autonomous LLMs for research and analysis tasks that traditionally require significant human effort. Autonomous research LLM-based entities are systems that can investigate a topic, gather information, and produce findings or recommendations with minimal human input.

### Multi-modal agent systems

While much of our discussion has focused on text-based inputs and outputs (since LLMs are fundamentally text-based), the autonomous approach extends to multi-modal LLM-based entities – agents that can process and generate multiple types of data, such as images, audio, video, or even actions in a physical environment (robotics).

![](https://sam-solutions.com/wp-content/uploads/infographic-3@2x-4.webp)

## **Challenges in Agentic LLM Development**

While agentic LLM architectures unlock impressive capabilities, they also introduce a host of challenges. When designing and deploying these systems, enterprise teams must be aware of potential issues and plan how to mitigate them. Here are some of the key challenges:

### Scalability issues

Agentic systems can be resource-intensive. Unlike a single LLM query which might take a few seconds and one model call, an autonomous agent might perform dozens of model calls and tool invocations to accomplish a single complex task. This raises several scalability considerations.

### Hallucination and reliability

Hallucination – the tendency of LLMs to produce incorrect or fabricated information – is a well-known issue. Hallucinations can undermine trust quickly. If a CEO asks the agent for a report and it fabricates a statistic confidently, that’s problematic. Building reliability is thus as important as building capability: sometimes it means dialing back the agent’s “creativity” in favor of correctness, and being transparent about uncertainty (training the agent to say it’s unsure or needs help when appropriate, rather than guessing).

### Security and ethical concerns

When giving an AI agent autonomy, even within a bounded environment, ensuring security and ethical behavior is paramount. An agent connected to tools could, if not properly constrained, attempt actions it shouldn’t. For instance, if integrated with an IT system, you wouldn’t want the agent arbitrarily reading confidential files or executing transactions it wasn’t meant to. Strong permissioning is needed. Each tool should enforce authentication and authorization – essentially, the agent’s identity should be tied to a role that only allows what’s intended. If the agent is supposed to only read data, the [API](https://sam-solutions.com/services/software-engineering/api-development-services/) it uses should not allow deletion or modification unless explicitly allowed.

## **Future of Agentic LLM Architectures**

The field of agentic AI, especially with LLMs at the core, is evolving at breakneck speed. Looking ahead, we can anticipate several trends and advancements that will shape the future of these architectures:

- **More powerful and specialized LLMs:** As base models continue to improve (with GPT-4, GPT-5, and equivalents from other providers or open source), LLM-based entities will inherently become more capable. We’ll see models with even larger context windows and better reasoning abilities, which means LLM-based entities can consider more information at once and make more nuanced decisions. Specialized LLMs fine-tuned for being LLM-based entities (rather than just chat or text completion) might emerge – ones that inherently “know” how to use tools or follow multi-step instructions out of the box.

- **Better integration of symbolic reasoning:** A likely direction is combining the neural capabilities of LLMs with more traditional symbolic AI components. This could mean agents that use logic engines or knowledge graphs alongside LLMs. For example, an agent could consult a knowledge graph of company policies to ensure compliance while using the LLM for language understanding. Such hybrid systems could curb hallucinations and enforce consistency (by using a deterministic reasoning module for parts of the task). Early research is indicating that blending rules-based reasoning with LLM creativity yields more reliable outcomes.


## **Why partner with SaM Solutions?**

Implementing an autonomous LLM architecture from the ground up can be a complex endeavor. It involves cutting-edge AI expertise, [software engineering](https://sam-solutions.com/services/software-engineering/) for integration, and careful consideration of security and ethics. For many enterprises, collaborating with an experienced technology partner is a pragmatic way to accelerate this journey and reduce risk. SaM Solutions is one such company that offers expertise in AI and software development, and partnering with a firm like them can provide several benefits.

Specialized partners stay up-to-date with the latest AI research and best practices. SaM Solutions, for example, would bring a team that understands how to tailor LLM prompts, fine-tune models if necessary, and design robust agent loops. They have likely seen what works and what doesn’t across different scenarios. This expertise can help avoid common pitfalls in agent development (such as misconfigured memory or inefficient planning logic) and implement state-of-the-art techniques that an internal team might not be aware of.

**Ready to implement AI into your digital strategy?** Let SaM Solutions guide your journey.

[Get in touch](https://sam-solutions.com/services/ai-software-development/?utm_source=blog&utm_medium=post_ID_18087&utm_campaign=cta_post_content_16276#feedback)

## Conclusion

In closing, autonomous LLM architecture is a powerful concept that, when implemented thoughtfully, can transform the way businesses leverage AI. It’s an exciting journey of turning what used to be static AI models into lively, task-oriented LLM-based entities that can collaborate with us. With solid strategy, strong technical foundations, and a commitment to responsibility, enterprise leaders can unlock the full potential of this technology in the years ahead.

## FAQ

How does agentic LLM architecture improve over traditional fine-tuning?

Traditional fine-tuning involves training an LLM on a specific task so it can perform that task better. However, a fine-tuned model still generally operates in a single-step, static manner – you give it an input and it gives an output, specialized to that one task. Agentic LLM architecture, on the other hand, doesn’t require training a new model for every task; instead it uses a base LLM and orchestrates it through a series of steps to handle complex tasks dynamically. The improvements are in flexibility and autonomy: an agentic LLM can tackle multi-step problems, use tools to get up-to-date information, and adapt its approach on the fly.

What programming languages are best for developing agentic LLM systems?

Python is currently the most popular language for developing agentic LLM systems. The reason is Python has a rich ecosystem of AI libraries (PyTorch, TensorFlow), natural language processing tools, and frameworks like LangChain or Haystack that support building these kinds of multi-step AI pipelines. Many pre-built integrations for things like vector databases, APIs, and model inference are available as Python packages, which accelerates development. That said, you’re not limited to Python. JavaScript/TypeScript is another language seeing increased usage, especially for LLM-based entities that need to run in web environments or use Node.js-based infrastructure (there are JS libraries for using LLMs and some agent frameworks in development).

How do agentic LLMs handle real-time data processing compared to static models?

Agentic LLMs are much better suited for real-time data or streaming information than static models. A static model (non-agentic) works on a fixed input – for example, it won’t automatically fetch new data once it’s given a prompt. If you want it to process new information, you have to manually include that info in its prompt. An agentic LLM, by design, can incorporate real-time data fetches as part of its operation.

Rate this item:0.501.001.502.002.503.003.504.004.505.00Submit Rating

Rating: **4.8**/5\. From 3 votes.
[Show votes.](https://sam-solutions.com/blog/llm-agent-architecture/#)

Please wait...

- **5** Stars



100.00%



3 votes

- **4** Stars



0.00%



0 votes

- **3** Stars



0.00%



0 votes

- **2** Stars



0.00%



0 votes

- **1** Star



0.00%



0 votes


About the Author

![Anastasiya](https://sam-solutions.com/wp-content/uploads/fly-images/9051/photo_2024-10-22_17-39-16-225x300-1-200x200-c.jpg)

Anastasiya Paharelskaya

Since 2021, Anastasiya has been diving deep into SAP Commerce Cloud, interviewing hundreds of SAP experts. She loves sharing what she learns and making complex topics easy to understand. While SAP Commerce is a big focus, Anastasiya is always excited to explore new tech areas such as AI, AR, and more, bringing fresh, valuable insights to her readers.

[View all posts](https://sam-solutions.com/author/a-paharelskaya/)

Leave a Comment

[Cancel reply](https://sam-solutions.com/blog/llm-agent-architecture/#respond)

Your email address will not be published.Required fields are marked \*

Name \*

Email \*

Comment \*

You may use these HTML tags and attributes: `<a href="" title=""> <abbr title=""> <acronym title=""> <b> <blockquote cite=""> <cite> <code> <del datetime=""> <em> <i> <q cite=""> <s> <strike> <strong> `

## Related Posts

[![Mixture-of-Experts (MoE) LLMs: The Future of Efficient AI Models](https://sam-solutions.com/wp-content/uploads/fly-images/19718/title-4-366x235-c.webp)\\
\\
AI & Machine Learning, Digital Transformation\\
\\
Mixture-of-Experts (MoE) LLMs: The Future of Efficient AI Models](https://sam-solutions.com/blog/moe-llm-architecture/)

[![AI Agents in Retail and Ecommerce: Transforming the Shopping Experience](https://sam-solutions.com/wp-content/uploads/fly-images/19602/AI-agents-in-retail-and-ecommerce-2x-366x235-c.webp)\\
\\
AI & Machine Learning, E-Commerce & CX, Retail & Wholesale\\
\\
AI Agents in Retail and Ecommerce: Transforming the Shopping Experience](https://sam-solutions.com/blog/ai-agents-in-retail/)

[![AI Agents in Healthcare: Revolutionizing Patient Care and Medical Innovation](https://sam-solutions.com/wp-content/uploads/fly-images/19315/AI-agent-development-for-healthcare@2x-366x235-c.webp)\\
\\
AI & Machine Learning, Healthcare\\
\\
AI Agents in Healthcare: Revolutionizing Patient Care and Medical Innovation](https://sam-solutions.com/blog/ai-agents-in-healthcare/)

[![Mamba LLM Architecture: A Breakthrough in Efficient AI Modeling](https://sam-solutions.com/wp-content/uploads/fly-images/19105/Mamba-LLM-architecture@2x-366x235-c.webp)\\
\\
AI & Machine Learning\\
\\
Mamba LLM Architecture: A Breakthrough in Efficient AI Modeling](https://sam-solutions.com/blog/mamba-llm-architecture/)

[![Model Context Protocol (MCP): Unlocking Smarter AI Integration for Your Business](https://sam-solutions.com/wp-content/uploads/fly-images/18742/Model-Context-Protocol-MCP-@2x-366x235-c.webp)\\
\\
AI & Machine Learning\\
\\
Model Context Protocol (MCP): Unlocking Smarter AI Integration for Your Business](https://sam-solutions.com/blog/model-context-protocol/)

[![15 Best AI Tools for Java Developers in 2025 [with Internal Survey Results]](https://sam-solutions.com/wp-content/uploads/fly-images/18712/title@2x-6-366x235-c.webp)\\
\\
AI & Machine Learning, Software Development, Technologies & Tools\\
\\
15 Best AI Tools for Java Developers in 2025 \[with Internal Survey Results\]](https://sam-solutions.com/blog/ai-tools-for-java-developers/)

[AI & Machine Learning, Digital Transformation, Technologies & Tools\\
\\
LLM Multi-Agent Architecture: The Future of AI Collaboration](https://sam-solutions.com/blog/llm-multi-agent-architecture/)

[AI & Machine Learning, Digital Transformation, Technologies & Tools\\
\\
LLM Transformer Architecture: Everything You Need to Know](https://sam-solutions.com/blog/llm-transformer-architecture/)

[AI & Machine Learning, Software Development, Technologies & Tools\\
\\
How to Implement AI in Java: A Step-by-Step Guide](https://sam-solutions.com/blog/how-to-implement-ai-in-java/)

[AI & Machine Learning, Digital Transformation, Technologies & Tools\\
\\
AI and Decision Making: Transforming Choices in the Digital Age](https://sam-solutions.com/blog/ai-decision-making/)

[AI & Machine Learning, Software Development\\
\\
LLM Architecture: A Comprehensive Guide](https://sam-solutions.com/blog/llm-architecture/)

[AI & Machine Learning, Digital Transformation\\
\\
Pattern Recognition in AI: A Comprehensive Guide](https://sam-solutions.com/blog/pattern-recognition-in-ai/)

‹›

### Original URL
https://sam-solutions.com/blog/llm-agent-architecture/
</details>

---
<details>
<summary>Compare Top 11 LLM Orchestration Frameworks in 2025</summary>

Leveraging multiple LLMs concurrently demands significant computational resources, driving up costs and introducing latency challenges. In the evolving landscape of AI, efficient LLM orchestration is essential for optimizing performance while minimizing expenses.

Explore key strategies and tools for managing multiple LLMs effectively. Here is a list of LLM orchestration tools sorted according to the number of GitHub stars:

Updated at 02-06-2025

| Framework | Github Stars | Supported languages | Best for |
| --- | --- | --- | --- |
| LangChain | 83.8k | Python <br> Javascript | Complex AI workflows |
| AutoGen | 38.7k | Python | Multi-agent coordination |
| LlamaIndex | 31.2k | Python <br> Typescript | Data integration |
| crewAI | 25.9k | Python | Role-based agent orchestration |
| Semantic kernel by Microsoft | 22.9k | C#<br> Python | Azure environment |
| Haystack by Deepset AI | 19k | Python | Custom NLP pipelines |
| TaskWeaver | 5.5k | Python | Agent-based task automation |
| Agency Swarm | 3.2k | Python | AI agent networks |
| Microchain | 282 | Python | Lightweight AI microservices |
| Loft | 10 | Python | No-code/low-code AI automatio |
| IBM watsonx orchestrate | Not open-source | Java<br> Javascript | Enterprise use |

## What is orchestration in LLM?

LLM Orchestration involves managing and integrating multiple [Large Language Models (LLMs](https://research.aimultiple.com/large-language-models/)) to perform complex tasks efficiently. It ensures smooth interaction between models, workflows, data sources, and pipelines, optimizing performance as a unified system. Organizations use LLM Orchestration for tasks like natural language generation, machine translation, decision-making, and chatbots.

While LLMs possess strong foundational capabilities, they are limited in real-time learning, retaining context, and solving multistep problems. Also, managing multiple LLMs across various provider APIs adds orchestration complexity.

LLM orchestration frameworks address these challenges by streamlining prompt engineering, API interactions, data retrieval, and state management. These frameworks enable LLMs to collaborate efficiently, enhancing their ability to generate accurate and context-aware outputs.

## What is the best platform for LLM orchestration?

LLM orchestration frameworks are tools designed to manage, coordinate, and optimize the use of Large Language Models (LLMs) in various applications. An LLM orchestration system enables seamless integration with different AI components, facilitate prompt engineering, manage workflows, and enhance performance monitoring.

They are particularly useful for applications involving multi-agent systems, [retrieval-augmented generation (RAG)](https://research.aimultiple.com/retrieval-augmented-generation/), [conversational AI](https://research.aimultiple.com/conversational-ai-platforms/), and autonomous decision-making.

The tools that are explained below are listed based on the alphabetical order:

### Agency Swarm

Agency Swarm is a scalable Multi-Agent System (MAS) framework that provides tools for building distributed AI environments.

**Key features:**

- **Supports large-scale multi-agent coordination** that enables many AI agents to work together efficiently.
- **Includes simulation and visualization tools**  that helps test and monitor agent interactions in a simulated environment.
- **Enables environment-based AI interactions** as AI agents can dynamically respond to changing conditions.

### AutoGen

AutoGen, developed by Microsoft, is an open-source multi-agent orchestration framework that simplifies AI task automation using conversational agents.

![The image shows AutoGen architecture, one of the top LLM orchestration tools.](https://research.aimultiple.com/wp-content/uploads/2025/02/AutoGen-architecture-1224x670.png.webp)Figure 1: AutoGen Architecture[1](https://research.aimultiple.com/llm-orchestration/#easy-footnote-bottom-1-1296361 "AutoGen LinkedIn")

**Key features:**

- **Multi-agent conversation framework** that allows AI agents to communicate and coordinate tasks.
- **Supports various AI models (OpenAI, Azure, custom models)**  that works with different LLM providers.
- **Modular and easy-to-configure system** referring to a customizable setup for various AI applications.

### crewAI

crewAI is an open-source multi-agent framework built on LangChain. It enables role-playing AI agents to collaborate on structured tasks.

**Key features:**

- **Agent-based workflow automation** that assigns AI agents specific roles in task execution.
- **Supports both technical and non-technical users**
- **Enterprise version (crewAI+) available**

### **Haystack**

Haystack is an open-source Python framework that allows for flexible AI pipeline creation using a component-based approach. It supports information retrieval and Q&A applications.

**Key features:**

- **Component-based AI system design**  which is a modular approach for assembling AI functions.
- **Integration with vector databases and LLM providers** enabling to work with various data storage and AI models.
- **Supports semantic search and information extraction**, enabling advanced search and knowledge retrieval.

### **IBM watsonx orchestrate**

IBM watsonx orchestrate is a proprietary AI orchestration framework that leverages natural language processing (NLP) to automate enterprise workflows. It includes prebuilt AI applications and tools designed for HR, procurement, and sales operations.

![The image shows a major enterprise LLM orchestrator that can be deployed on AWS: IBM Watsonx orchestrator](https://images.surferseo.art/3cb93e91-13d4-48fd-8253-4e611a804553.png)Figure 2: IBM watsonx orchestrator [2](https://research.aimultiple.com/llm-orchestration/#easy-footnote-bottom-2-1296362 "IBM Watsonx")

**Key features:**

- **AI-powered workflow automation** that can automate repetitive business processes using AI.
- **Prebuilt applications and skill sets**, providing ready-to-use AI tools for different industries.
- **Enterprise-focused integration**, connecting with existing enterprise software and workflows.

### **LangChain**

LangChain is an open-source Python framework for building LLM applications, focusing on tool augmentation and agent orchestration. It provides interfaces for embedding models, LLMs, and vector stores.

**Key features:**

- **RAG** **support**
- **Integration with multiple LLM components**
- **ReAct framework for** reasoning and action

### **LlamaIndex**

LlamaIndex is an open-source data integration framework designed for building context-augmented LLM applications. It enables easy retrieval of data from multiple sources.

**Key features:**

- **Data connectors for over 160 sources**, allowing AI to access diverse structured and unstructured data.
- **Retrieval-Augmented Generation (RAG) support**
- **Suite of evaluation modules for performance tracking**

### LOFT

LOFT, developed by Master of Code Global, is a Large Language Model-Orchestrator Framework designed to optimize AI-driven customer interactions. Its queue-based architecture ensures high throughput and scalability, making it suitable for large-scale deployments.

![The architecture of LOFT, an LLM orchestration framework among the top LLM orchestration tools](https://research.aimultiple.com/wp-content/uploads/2025/02/Loft-llm-orchestration-architecture-1224x876.png.webp)

**Key features:**

- **Framework agnostic:** Integrates into any backend system without dependencies on HTTP frameworks.
- **Dynamically computed prompts:** Supports custom-generated prompts for personalized user interactions.
- **Event detection & handling:** Advanced capabilities for detecting and managing chat-based events, including handling hallucinations.

### Microchain

Microchain is a lightweight, open-source LLM orchestration framework known for its simplicity but is not actively maintained.

**Key features:**

- **Chain-of-thought reasoning support** that helps AI break down complex problems step by step.
- **Minimalist approach to AI orchestration**

### Semantic Kernel

Semantic Kernel (SK) is an open-source AI orchestration framework by Microsoft. It helps developers integrate large language models (LLMs) like OpenAI’s GPT with traditional programming to create AI-powered applications.

**Key features:**

- **Memory & context handling:** SK allows storage and retrieval of past interactions, helping maintain context over conversations.
- **Embeddings & vector search:** Supports embedding-based searches, making it great for retrieval-augmented generation (RAG) use cases.
- **Multi-modal support:** Works with text, code, images, and more.

### **TaskWeaver**

TaskWeaver is an experimental open-source framework designed for coding-based task execution in AI applications. It prioritizes modular task decomposition.

**Key features**

- **Modular design for decomposing tasks** that breaks down complex processes into manageable AI-driven steps.
- **Declarative task specification**, allowing tasks to be defined in a structured format.
- **Context-aware decision-making**, allowing AI to adapt its actions based on changing inputs.

## How do LLM orchestration tools work?

LLM orchestration frameworks manage the interaction between different components of LLM-driven applications, ensuring structured workflows and efficient execution. The orchestration layer plays a central role in coordinating processes such as prompt management, resource allocation, data preprocessing, and model interactions.

### Orchestration Layer

The orchestration layer acts as the central control system within an LLM-powered application. It manages interactions between various components, including LLMs, prompt templates, vector databases, and AI agents. By overseeing these elements, orchestration ensures cohesive performance across different tasks and environments.

### Key Orchestration Tasks

#### Prompt chain management

- The framework structures and manages LLM inputs (prompts) to optimize output.
- It provides a repository of prompt templates, allowing for dynamic selection based on context and user inputs.
- It sequences prompts logically to maintain structured conversation flows.
- It evaluates responses to refine output quality, detect inconsistencies, and ensure adherence to guidelines.
- Fact-checking mechanisms can be implemented to reduce inaccuracies, with flagged responses directed for human review.

#### LLM resource and performance management

- Orchestration frameworks monitor LLM performance through benchmark tests and real-time dashboards.
- They provide diagnostic tools for root cause analysis (RCA) to facilitate debugging.
- They allocate computational resources efficiently to optimize performance.

#### Data management and preprocessing

- The orchestrator retrieves data from specified sources using connectors or APIs.
- Preprocessing converts raw data into a format compatible with LLMs, ensuring data quality and relevance.
- It refines and structures data to enhance its suitability for processing by different algorithms.

#### LLM integration and interaction

- The orchestrator initiates LLM operations, processes the generated output, and routes it to the appropriate destination.
- It maintains memory stores that enhance contextual understanding by preserving previous interactions.
- Feedback mechanisms assess output quality and refine responses based on historical data.

#### Observability and security measures

- The orchestrator supports monitoring tools to track model behavior and ensure output reliability.
- It implements security frameworks to mitigate risks associated with unverified or inaccurate outputs.

### Additional Enhancements

#### Workflow integration

- Embeds tools, technologies, or processes into existing operational systems to improve efficiency, consistency, and productivity.
- Ensures smooth transitions between different model providers while maintaining prompt and output quality.

#### Changing model providers

- Some frameworks allow switching model providers with minimal changes, reducing operational friction.
- Updating provider imports, adjusting model parameters, and modifying class references facilitate seamless transitions.

#### Prompt management

- Maintains consistency in prompting while helping users iterate and experiment more productively.
- Integrates with CI/CD pipelines to streamline collaboration and automate change tracking.
- Some systems automatically track prompt modifications, helping catch unexpected impacts on prompt quality.

## Why is LLM orchestration important in real-time applications?

LM Orchestration enhances the efficiency, scalability, and reliability of AI-driven language solutions by optimizing resource utilization, automating workflows, and improving system performance. Key benefits include:

- **Better decision-making**: Aggregates insights from multiple LLMs, leading to more informed and strategic decision-making.
- **Cost efficiency**: Optimizes costs by dynamically allocating resources based on workload demand.
- **Enhanced efficiency**: Streamlines LLM interactions and workflows, reducing redundancy, minimizing manual effort, and improving overall operational efficiency.
- **Fault tolerance**: Detects failures and automatically redirects traffic to healthy LLM instances, minimizing downtime and maintaining service availability.
- **Improved accuracy**: Leverages multiple LLMs to enhance language understanding and generation, leading to more precise and context-aware outputs.
- **Load balancing**: Distributes requests across multiple LLM instances to prevent overload, ensuring reliability and improving response times.
- **Lowered technical barriers**: Enables easy implementation without requiring AI expertise, with user-friendly tools like LangFlow simplifying orchestration.
- **Dynamic resource allocation:** Allocates CPU, GPU, memory, and storage efficiently, ensuring optimal model performance and cost-effective operation.
- **Risk mitigation**: Reduces failure risks by ensuring redundancy, allowing multiple LLMs to back up one another.
- **Scalability**: Dynamically manages and integrates LLMs, allowing AI systems to scale up or down based on demand without performance degradation.
- **Seamless integration**: Supports interoperability with external services, including data storage, logging, monitoring, and analytics.
- **Security & compliance**: Centralized control and monitoring ensure adherence to regulatory standards, enhancing sensitive data security and privacy.
- **Version control & updates**: Facilitates seamless model updates and version management without disrupting operations.
- **Workflow automation**: Automates complex processes such as data preprocessing, model training, inference, and postprocessing, reducing developer workload.

Explore [process KPIs](https://research.aimultiple.com/process-kpis/) to understand how to streamline them with LLM orchestration.

### Original URL
https://research.aimultiple.com/llm-orchestration/
</details>

---
<details>
<summary>GenAI paradox: exploring AI use cases | McKinsey</summary>

[iframe](javascript:void(0))GenAI paradox: exploring AI use cases \| McKinsey

[Skip to main content](https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage#skipToMain)

![AskMck-button-logo](<Base64-Image-Removed>)

Seizing the agentic AI advantage

Share

[Print](https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage#/print)

Download

[Save](https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage#/save)

# Seizing the agentic AI advantage

June 13, 2025 \| Report

Share

[Print](https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage#/print)

Download

[Save](https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage#/save)

A CEO playbook to solve the gen AI paradox and unlock scalable impact with AI agents.

### DOWNLOADS

[Full Report (28 pages)](https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage#/download/%2F~%2Fmedia%2Fmckinsey%2Fbusiness%20functions%2Fquantumblack%2Four%20insights%2Fseizing%20the%20agentic%20ai%20advantage%2Fseizing-the-agentic-ai-advantage.pdf%3FshouldIndex%3Dfalse)

### At a glance

- Nearly eight in ten companies report using gen AI—yet just as many report no significant bottom-line impact.1“ [The state of AI: How organizations are rewiring to capture value](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai),” McKinsey, March 12, 2025. Think of it as the “gen AI paradox.”

Byline

## About the authors

This report is a collaborative effort by [Alexander Sukharevsky](https://www.mckinsey.com/our-people/alexander-sukharevsky), [Dave Kerr](https://www.mckinsey.com/our-people/dave-kerr), [Klemens Hjartar](https://www.mckinsey.com/our-people/klemens-hjartar), [Lari Hämäläinen](https://www.mckinsey.com/our-people/lari-hamalainen), [Stéphane Bout](https://www.mckinsey.com/our-people/stephane-bout), and [Vito Di Leo](https://www.mckinsey.com/our-people/vito-di-leo), with Guillaume Dagorret, representing views from QuantumBlack, AI by McKinsey and McKinsey Technology.

- At the heart of this paradox is an imbalance between “horizontal” (enterprise-wide) copilots and chatbots—which have scaled quickly but deliver diffuse, hard-to-measure gains—and more transformative “vertical” (function-specific) use cases—about 90 percent of which remain stuck in pilot mode.
- AI agents offer a way to break out of the gen AI paradox. That’s because agents have the potential to automate complex business processes—combining autonomy, planning, memory, and integration—to shift gen AI from a reactive tool to a proactive, goal-driven virtual collaborator.
- This shift enables far more than efficiency. Agents supercharge operational agility and create new revenue opportunities.
- But unlocking the full potential of agentic AI requires more than plugging agents into existing workflows. It calls for reimagining those workflows from the ground up—with agents at the core.

Share

Sidebar

## Foreword

_by Arthur Mensch, CEO of Mistral AI_

We’re at a moment when gen AI has entered every boardroom, but for many enterprises, it still lingers at the edges of actual impact. Many CEOs have greenlit experiments, spun up copilots, and created promising prototypes, but only a handful have seen the needle move on revenue or impact. This report gets to the heart of that paradox: broad adoption with limited return.

The current diagnosis is this: Today, AI is bolted on. But to deliver real impact, it must be integrated into core processes, becoming a catalyst for business transformation rather than a sidecar tool. Most deployments today use AI in a shallow way—as an assistant that sits alongside existing workflows and processes—rather than as a deeply integrated, engaged, and powerful agent of transformation.

Agentic AI is the catalyst that can make this transition possible, but doing so requires a strategy and a plan to successfully power that transformation. Agents are not simply magical plug-n-play pieces. They must work across systems, reason through ambiguity, and interact with people—not just as tools, but as collaborators. That means CEOs must ask different questions: not “How do we add AI?” but “How do we want decisions to be made, work to flow, and humans to engage in an environment where software can act?”

Redefining how decisions are made, how work is done, and how humans engage with technology requires alignment across goals, tools, and people. That alignment can only happen when openness, transparency, and control are central to your technology and implementation—when builders have an open, extensible, and observable infrastructure and users can easily craft and use agents with the confidence that the work of agents is safe, reliable, and under their control. That alignment creates the trust and effectiveness that is the currency of scalable transformation that delivers results rather than regrets.

The technology to build powerful agents is already here. The opportunity now is to deploy agents in ways that are deeply tied to how value is created and how people work. That requires an architecture that is modular and resilient and, more importantly, an operating model that centers on humans—not just as users but as co-architects of the systems they will be living and working with.

This report lays out the playbook not for tinkering but for reinvention. ROI comes from strong intent: define the outcomes, embed agents deep in core workflows, and redesign operating models around them. Organizations that win will pair a clear strategy with tight feedback loops and disciplined governance, using agents to rethink how decisions are made and how work gets done—and turning novelty into measurable value.

- A new AI architecture paradigm—the agentic AI mesh—is needed to govern the rapidly evolving organizational AI landscape and enable teams to blend custom-built and off-the-shelf agents while managing mounting technical debt and new classes of risk. But the bigger challenge won’t be technical. It will be human: earning trust, driving adoption, and establishing the right governance to manage agent autonomy and prevent uncontrolled sprawl.
- To scale impact in the agentic era, organizations must reset their AI transformation approaches from scattered initiatives to strategic programs; from use cases to business processes; from siloed AI teams to cross-functional transformation squads; and from experimentation to industrialized, scalable delivery.
- Organizations will also need to set up the foundation to effectively operate in the agentic era. They will need to upskill the workforce, adapt the technology infrastructure, accelerate data productization, and deploy agent-specific governance mechanisms. The moment has come to bring the gen AI experimentation chapter to a close—a pivot only the CEO can make.

Chapter 1

## The gen AI paradox: Widespread deployment, minimal impact [The gen AI paradox: Widespread deployment, minimal impact](https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage\#)

## Key Points

Continue to next section

Share

- **_Nearly eight in ten companies have deployed gen AI in some form, but roughly the same percentage report no material impact on earnings._** 1“ [The state of AI: How organizations are rewiring to capture value](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai),” McKinsey, March 12, 2025. **_We call this the “gen AI paradox.”_**
- **_The main issue is an imbalance between “horizontal” and “vertical” use cases. The former, such as employee copilots and chatbots, have been widely deployed but deliver diffuse benefits, while higher-impact vertical, or function-specific, use cases seldom make it out of the pilot phase because of technical, organizational, data, and cultural barriers._**
- **_Unless companies address these barriers, the transformational promise of gen AI will remain largely untapped._**

## Gen AI is everywhere—except in company P&L

Share

Sidebar

## About QuantumBlack, AI by McKinsey

QuantumBlack, McKinsey’s AI arm, has been helping businesses create value from AI since 2009, expanding on McKinsey’s technology work over the past 30 years. QuantumBlack combines an industry-leading tech stack with the strength of McKinsey’s 7,000 technologists, designers, and product managers serving clients in more than 50 countries. With innovations fueled by QuantumBlack Labs—its center for R&D and software development—QuantumBlack delivers the organizational rewiring that businesses need to build, adopt, and scale AI capabilities.

Even before the advent of gen AI, artificial intelligence had already carved out a key place in the enterprise, powering advanced prediction, classification, and optimization capabilities. And the technology’s estimated value potential was already immense— [between $11 trillion and $18 trillion globally](https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/the-economic-potential-of-generative-ai-the-next-productivity-frontier) 2“ [The economic potential of generative AI: The next productivity frontier](https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/the-economic-potential-of-generative-ai-the-next-productivity-frontier),” McKinsey, June 14, 2023.—mainly in the fields of marketing (powering capabilities such as personalized email targeting and customer segmentation), sales (lead scoring), and supply chain (inventory optimization and demand forecasting). Yet AI was largely the domain of experts. As a result, adoption across the rank and file tended to be slow. From 2018 to 2022, for example, AI adoption remained relatively stagnant, with about 50 percent of companies deploying the technology in just one business function, according to McKinsey research (Exhibit 1).

Exhibit 1

![Gen AI has accelerated AI deployment overall.](https://www.mckinsey.com/~/media/mckinsey/business%20functions/quantumblack/our%20insights/seizing%20the%20agentic%20ai%20advantage/svgz-seizing-agentic-ai_ex1-v6.svgz?cq=50&cpy=Center)

We strive to provide individuals with disabilities equal access to our website. If you would like information about this content we will be happy to work with you. Please email us at: [McKinsey\_Website\_Accessibility@mckinsey.com](mailto:McKinsey_Website_Accessibility@mckinsey.com)

Gen AI has extended the reach of traditional AI in three breakthrough areas: information synthesis, content generation, and communication in human language. McKinsey estimates that the technology has the potential to unlock [$2.6 trillion to $4.4 trillion in additional value](https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/the-economic-potential-of-generative-ai-the-next-productivity-frontier) on top of the value potential of traditional analytical AI.3“ [The economic potential of generative AI: The next productivity frontier](https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/the-economic-potential-of-generative-ai-the-next-productivity-frontier),” McKinsey, June 14, 2023.

Two and a half years after the launch of ChatGPT, gen AI has reshaped how enterprises engage with AI. Its potentially transformative power lies not only in the new capabilities gen AI introduces but also in its ability to democratize access to advanced AI technologies across organizations. This democratization has led to widespread growth in awareness of, and experimentation with, AI: According to [McKinsey’s most recent Global Survey on AI](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai),4“ [The state of AI: How organizations are rewiring to capture value](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai),” McKinsey, March 12, 2025. more than 78 percent of companies are now using gen AI in at least one business function (up from 55 percent a year earlier).

However, this enthusiasm has yet to translate into tangible economic results. [More than 80 percent of companies still report no material contribution to earnings](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai) from their gen AI initiatives.5“ [The state of AI: How organizations are rewiring to capture value](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai),” McKinsey, March 12, 2025. What’s more, only [1 percent of enterprises we surveyed view their gen AI strategies as mature](https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/superagency-in-the-workplace-empowering-people-to-unlock-ais-full-potential-at-work).6Hannah Mayer, Lareina Yee, Michael Chui, and Roger Roberts, “ [Superagency in the workplace: Empowering people to unlock AI’s full potential](https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/superagency-in-the-workplace-empowering-people-to-unlock-ais-full-potential-at-work),” McKinsey, January 28, 2025. Call it the “gen AI paradox”: For all the energy, investment, and potential surrounding the technology, at-scale impact has yet to materialize for most organizations.

## At the heart of the gen AI paradox lies an imbalance between horizontal and vertical use cases

Many organizations have deployed horizontal use cases, such as enterprise-wide copilots and chatbots; nearly 70 percent of Fortune 500 companies, for example, use Microsoft 365 Copilot.7Satya Nadella, “Microsoft Fiscal Year 2025 First Quarter Earnings Conference Call,” Microsoft, October 30, 2024. These tools are widely seen as levers to enhance individual productivity by helping employees save time on routine tasks and access and synthesize information more efficiently. But these improvements, while real, tend to be spread thinly across employees. As a result, they are not easily visible in terms of top- or bottom-line results.

By contrast, vertical use cases—those embedded into specific business functions and processes—have seen limited scaling in most companies despite their higher potential for direct economic impact (Exhibit 2). [Fewer than 10 percent of use cases deployed ever make it past the pilot stage](https://www.mckinsey.com/about-us/new-at-mckinsey-blog/mckinsey-alliances-bring-the-power-of-generative-ai-to-clients), according to McKinsey research.8_New at McKinsey Blog_, “ [McKinsey’s ecosystem of strategic alliances brings the power of generative AI to clients](https://www.mckinsey.com/about-us/new-at-mckinsey-blog/mckinsey-alliances-bring-the-power-of-generative-ai-to-clients),” April 2, 2024. Even when they have been fully deployed, these use cases typically have supported only isolated steps of a business process and operated in a reactive mode when prompted by a human, rather than functioning proactively or autonomously. As a result, their impact on business performance also has been limited.

Exhibit 2

![Across business functions, gen AI use cases tend to fall into two categories: horizontal and vertical.](https://www.mckinsey.com/~/media/mckinsey/business%20functions/quantumblack/our%20insights/seizing%20the%20agentic%20ai%20advantage/svgz-seizing-agentic-ai_ex2-v5.svgz?cq=50&cpy=Center)

We strive to provide individuals with disabilities equal access to our website. If you would like information about this content we will be happy to work with you. Please email us at: [McKinsey\_Website\_Accessibility@mckinsey.com](mailto:McKinsey_Website_Accessibility@mckinsey.com)

What accounts for this imbalance? For one thing, horizontally deployed copilots such as Microsoft Copilot or Google AI Workspace are accessible, off-the-shelf solutions that are relatively easy to implement. (In many cases, enabling Microsoft Copilot is as simple as activating an extension to an existing Office 365 contract, requiring no redesign of workflows or major change management efforts.) Rapid deployment of enterprise chatbots also has been driven by risk mitigation concerns. As employees began experimenting with external large language models (LLMs) such as ChatGPT, many organizations implemented internal, secure alternatives to limit data leakage and ensure compliance with corporate security policies.

The limited deployment and narrow scope of vertical use cases can in turn be attributed to six primary factors:

- **Fragmented initiatives.** At many companies, vertical use cases have been identified through a bottom-up, highly granular approach within individual functions. In fact, [fewer than 30 percent of companies report that their CEOs sponsor their AI agenda directly](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai).9“ [The state of AI: How organizations are rewiring to capture value](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai),” McKinsey, March 12, 2025. This has led to a proliferation of disconnected micro-initiatives and a dispersion of AI investments, with limited coordination at the enterprise level.
- **Lack of mature, packaged solutions.** Unlike off-the-shelf horizontal applications, such as copilots, vertical use cases often require custom development. As a result, teams are frequently forced to build from scratch, using emerging, fast-evolving technologies they have limited experience with. While many companies have invested in data scientists to develop AI models, they often lack MLOps engineers, who are critical to industrialize, deploy, and maintain those models in production environments.
- **Technological limitations of LLMs.** Despite their impressive capabilities, the first generation of LLMs faced limitations that significantly constrained their deployment at enterprise scale. First, LLMs can produce inaccurate outputs, which makes them difficult to trust in environments where precision and repeatability are essential. What’s more, despite their power, LLMs are fundamentally passive; they do not act unless prompted and cannot independently drive workflows or make decisions without human initiation. LLMs also have struggled to handle complex workflows involving multiple steps, decision points, or branching logic. Finally, many current LLMs have limited persistent memory, making it difficult to track context over time or operate coherently across extended interactions.
- **Siloed AI teams.** AI centers of excellence have played a crucial role in accelerating awareness and experimentation across many organizations. However, in many cases, these teams have operated in silos—developing AI models independently from core IT, data, or business functions. This autonomy, while useful for rapid prototyping, has often made solutions difficult to scale because of poor integration with enterprise systems, fragmented data pipelines, or a lack of operational alignment.
- **Data accessibility and quality gaps.** These gaps tend to exist for both structured and unstructured data, with unstructured material remaining largely ungoverned in most organizations.
- **Cultural apprehension and organizational inertia.** In many organizations, AI deployments have encountered implicit resistance from business teams and middle management due to fear of disruption, uncertainty around job impact, and lack of familiarity with the technology.

Despite its limited bottom-line impact so far, the first wave of gen AI has been far from wasted. It has enriched employee capabilities, enabled broad experimentation, accelerated AI familiarity across functions, and helped organizations build essential capabilities in prompt engineering, model evaluation, and governance. All of this has laid the groundwork for a more integrated and transformative second phase— [the emerging age of AI agents](https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/why-agents-are-the-next-frontier-of-generative-ai).10Lareina Yee, Michael Chui, Roger Roberts, and Stephen Xu, “ [Why agents are the next frontier of generative AI](https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/why-agents-are-the-next-frontier-of-generative-ai),” _McKinsey Quarterly_, July 24, 2024.

Chapter 2

## From paradox to payoff: How agents can scale AI [From paradox to payoff: How agents can scale AI](https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage\#)

## Key Points

Continue to next section

Share

- **_By automating complex business workflows, agents unlock the full potential of vertical use cases. Forward-looking companies are already harnessing the power of agents to transform core processes._**
- **_To realize the potential of agents, companies must reinvent the way work gets done—changing task flows, redefining human roles, and building agent-centric processes from the ground up._**
- **_Accomplishing this will require a new paradigm for AI architecture—the agentic AI mesh—capable of integrating both custom-built and off-the-shelf agents. But the bigger challenge will not be technical. It will be human: earning trust to drive adoption and establishing the proper governance protocols._**

## The breakthrough: Automating complex business workflows unlocks the full potential of vertical use cases

LLMs have revolutionized how organizations interact with data—enabling information synthesis, content generation, and natural language interaction. But despite their power, LLMs have been fundamentally reactive and isolated from enterprise systems, largely unable to retain memory of past interactions or context across sessions or queries. Their role has been largely limited to enhancing individual productivity through isolated tasks. AI agents mark a major evolution in enterprise AI—extending gen AI from reactive content generation to autonomous, goal-driven execution. Agents can understand goals, break them into subtasks, interact with both humans and systems, execute actions, and adapt in real time—all with minimal human intervention. They do so by combining LLMs with additional technology components providing memory, planning, orchestration, and integration capabilities.

With these new capabilities, AI agents expand the potential of horizontal solutions, upgrading general-purpose copilots from passive tools into proactive teammates that don’t just respond to prompts but also monitor dashboards, trigger workflows, follow up on open actions, and deliver relevant insights in real time. But the real breakthrough comes in the vertical realm, where agentic AI enables the automation of complex business workflows involving multiple steps, actors, and systems—processes that were previously beyond the capabilities of first-generation gen AI tools.

## Agents deliver more than efficiency—they supercharge operational agility and unlock new revenue opportunities

On the operations side, agents take on routine, data-heavy tasks so humans can focus on higher-value work. But they go further, transforming processes in five ways:

- **Agents accelerate execution by eliminating delays between tasks and by enabling parallel processing.** Unlike in traditional workflows that rely on sequential handoffs, agents can coordinate and execute multiple steps simultaneously, reducing cycle time and boosting responsiveness.
- **Agents bring adaptability.** By continuously ingesting data, agents can adjust process flows on the fly, reshuffling task sequences, reassigning priorities, or flagging anomalies before they cascade into failures. This makes workflows not only faster but smarter.
- **Agents enable personalization.** By tailoring interactions and decisions to individual customer profiles or behaviors, agents can adapt the process dynamically to maximize satisfaction and outcomes.
- **Agents bring elasticity to operations.** Because agents are digital, their execution capacity can expand or contract in real time depending on workload, business seasonality, or unexpected surges—something difficult to achieve with fixed human resource models.
- **Agents also make operations more resilient.** By monitoring disruptions, rerouting operations, and escalating only when needed, they keep processes running—whether it’s supply chains navigating port delays or service workflows adapting to system outages.

In a complex supply chain environment, for example, an AI agent could act as an autonomous orchestration layer across sourcing, warehousing, and distribution operations. Connected to internal systems (such as the supply chain planning system or the warehouse management system) and external data sources (such as weather forecasts, supplier feeds, and demand signals), the agent could continuously forecast demand. It could then identify risks, such as delays or disruptions, and dynamically replan transport and inventory flows. Selecting the optimal transport mode based on cost, lead time, and environmental impact, the agent could reallocate stock across warehouses, negotiate directly with external systems, and escalate decisions requiring strategic input. The result: improved service levels, reduced logistics costs, and lower emissions.

Agents can also help spur top-line growth by amplifying existing revenue streams and unlocking entirely new ones:

- **Amplifying existing revenues.** In e-commerce, agents embedded into online stores or apps could proactively analyze user behavior, cart content, and context (for example, seasonality or purchase history) to surface real-time upselling and cross-selling offers. In finance, agents might help customers discover suitable financial products such as loans, insurance plans, or investment portfolios, providing tailored guidance based on financial profiles, life events, and user behavior.
- **Creating new revenue streams.** For industrial companies, agents embedded in connected products or equipment could monitor usage, detect performance thresholds, and autonomously unlock features or trigger maintenance actions—enabling pay-per-use, subscription, or performance-based models of creating revenue. Similarly, service organizations could encapsulate internal expertise—legal reasoning, tax interpretation, and procurement best practices—into AI agents offered as software-as-a-service tools or APIs to clients, partners, or smaller businesses lacking in-house expertise.

In short, agentic AI doesn’t just automate. It redefines how organizations operate, adapt, and create value.

## No longer science fiction: Forward-looking companies are harnessing the power of agents

The following case studies demonstrate how QuantumBlack helps organizations build agent workforces—with outcomes that extend far beyond efficiency gains.

### Case study 1: How a bank used hybrid ‘digital factories’ for legacy app modernization

**The problem:** A large bank needed to modernize its legacy core system, which consisted of 400 pieces of software—a massive undertaking budgeted at more than $600 million. Large teams of coders tackled the project using manual, repetitive tasks, which resulted in difficulty coordinating across silos. They also relied on often slow, error-prone documentation and coding. While first-generation gen AI tools helped accelerate individual tasks, progress remained slow and laborious.

**The agentic approach:** Human workers were elevated to supervisory roles, overseeing squads of AI agents, each contributing to a shared objective in a defined sequence (Exhibit 3). These squads retroactively document the legacy application, write new code, review the code of other agents, and integrate code into features that are later tested by other agents prior to delivery of the end product. Freed from repetitive, manual tasks, human supervisors guide each stage of the process, enhancing the quality of deliverables and reducing the number of sprints required to implement new features.

**Impact:** More than 50 percent reduction in time and effort in the early adopter teams

Exhibit 3

![A large bank upgraded its legacy tech stack with a hybrid AI-human digital factory.](https://www.mckinsey.com/~/media/mckinsey/business%20functions/quantumblack/our%20insights/seizing%20the%20agentic%20ai%20advantage/svgz-seizing-agentic-ai_ex3-v8.svgz?cq=50&cpy=Center)

We strive to provide individuals with disabilities equal access to our website. If you would like information about this content we will be happy to work with you. Please email us at: [McKinsey\_Website\_Accessibility@mckinsey.com](mailto:McKinsey_Website_Accessibility@mckinsey.com)

### Case study 2: How a research firm boosted data quality to derive deeper market insights

**The problem:** A market research and intelligence firm was devoting substantial resources to ensure data quality, relying on a team of more than 500 people whose responsibilities included gathering data, structuring and codifying it, and generating tailored insights for clients. The process, conducted manually, was prone to error, with a staggering 80 percent of mistakes identified by the clients themselves.

**The agentic approach:** A multiagent solution autonomously identifies data anomalies and explains shifts in sales or market share. It analyzes internal signals, such as changes in product taxonomy, and external events identified via web searches, including product recalls or severe weather. The most influential drivers are synthesized, ranked, and prepared for decision-makers. With advanced search and contextual reasoning, the agents often surface insights that would be difficult for human analysts to uncover manually. While not yet in production, the system is fully functional and has demonstrated strong potential to free up analysts for more strategic work.

**Impact:** More than 60 percent potential productivity gain and expected savings of more than $3 million annually.

### Case study 3: How a bank reimagined the way it creates credit-risk memos

**The problem:** Relationship managers (RMs) at a retail bank were spending weeks writing and iterating credit-risk memos to help make credit decisions and fulfill regulatory requirements (Exhibit 4). This process required RMs to manually review and extract information from at least ten different data sources and develop complex nuanced reasoning across interdependent sections—for instance, loan, revenue, and cash joint evolution.

**The agentic approach:** In close collaboration with the bank’s credit-risk experts and RMs, a proof of concept was developed to transform the credit memo workflow using AI agents. The agents assist RMs by extracting data, drafting memo sections, generating confidence scores to prioritize review, and suggesting relevant follow-up questions. In this model, the analyst’s role shifts from manual drafting to strategic oversight and exception handling.

**Impact:** A potential 20 to 60 percent increase in productivity, including a 30 percent improvement in credit turnaround

Exhibit 4

![A retail bank used AI agents to reinvent the process of creating credit-risk memos.](https://www.mckinsey.com/~/media/mckinsey/business%20functions/quantumblack/our%20insights/seizing%20the%20agentic%20ai%20advantage/svgz-seizing-agentic-ai_ex4-v5.svgz?cq=50&cpy=Center)

We strive to provide individuals with disabilities equal access to our website. If you would like information about this content we will be happy to work with you. Please email us at: [McKinsey\_Website\_Accessibility@mckinsey.com](mailto:McKinsey_Website_Accessibility@mckinsey.com)

## Maximizing value from AI agents requires process reinvention

Realizing AI’s full potential in the vertical realm requires more than simply inserting agents into legacy workflows. It instead calls for a shift in design mindset—from automating tasks within an existing process to reinventing the entire process with human and agentic coworkers. That’s because when agents are embedded into a legacy process without redesign, they typically serve as faster assistants—generating content, retrieving data, or executing predefined steps. But the process itself remains sequential, rule bound, and shaped by human constraints.

Reinventing a process around agents means more than layering automation on top of existing workflows—it involves rearchitecting the entire task flow from the ground up. That includes reordering steps, reallocating responsibilities between humans and agents, and designing the process to fully exploit the strengths of agentic AI: parallel execution that collapses cycle time, real-time adaptability that reacts to changing conditions, deep personalization at scale, and elastic capacity that flexes instantly with demand.

Consider a hypothetical customer call center. Before introducing AI agents, the facility was using gen AI tools to assist human support staff by retrieving articles from knowledge bases, summarizing ticket histories, and helping draft responses. While this assistance improved speed and reduced cognitive load, the process itself remained entirely manual and reactive, with human agents still managing every step of diagnosis, coordination, and resolution. The productivity improvement potential was modest, typically boosting resolution time and productivity between 5 and 10 percent.

Now imagine that the call center introduces AI agents but largely preserves the existing workflow—agents are added to assist at specific steps without reconfiguring how work is routed, tracked, or resolved end-to-end. Agents can classify tickets, suggest likely root causes, propose resolution paths, and even autonomously resolve frequent, low-complexity issues (such as password resets). While the impact here can be increased—an estimated 20 to 40 percent savings in time and a 30 to 50 percent reduction in backlog—coordination friction and limited adaptability prevent true breakthrough gains.

But the real shift occurs at the third level, when the call center’s process is reimagined around agent autonomy. In this model, AI agents don’t just respond—they proactively detect common customer issues (such as delayed shipments, failed payments, or service outages) by monitoring patterns across channels, anticipate likely needs, initiate resolution steps automatically (such as issuing refunds, reordering items, or updating account details), and communicate directly with customers via chat or email. Human agents are repositioned as escalation managers and service quality overseers, who are brought in only when agents detect uncertainty or exceptions to typical patterns. Impact at this level is transformative. This could allow a radical improvement of customer service desk productivity. Up to 80 percent of common incidents could be resolved autonomously, with a reduction in time to resolution of 60 to 90 percent (Exhibit 5).

Exhibit 5

![Agents hold the key to breaking through--if processes are reinvented, not just optimized.](https://www.mckinsey.com/~/media/mckinsey/business%20functions/quantumblack/our%20insights/seizing%20the%20agentic%20ai%20advantage/svgz-seizing-agentic-ai_ex5-v6.svgz?cq=50&cpy=Center)

We strive to provide individuals with disabilities equal access to our website. If you would like information about this content we will be happy to work with you. Please email us at: [McKinsey\_Website\_Accessibility@mckinsey.com](mailto:McKinsey_Website_Accessibility@mckinsey.com)

Of course, not every business process requires full reinvention. Simple task automation is sufficient for highly standard, repetitive workflows with limited variability—such as payroll processing, travel expense approvals, or password resets—where gains come primarily from reducing manual effort. In contrast, processes that are complex, cross-functional, prone to exceptions, or tightly linked to business performance often warrant full redesign. Key indicators that call for reinvention include high coordination overhead, rigid sequences that delay responsiveness, frequent human intervention for decisions that could be data driven, and opportunities for dynamic adaptation or personalization. In these cases, redesigning the process around the agent’s ability to orchestrate, adapt, and learn delivers far greater value than simply speeding up existing workflows.

## A new AI architecture paradigm—the agentic AI mesh—is required to orchestrate value in the agentic era

To scale agents, companies will need to overcome a threefold challenge: handling the newfound risks that AI agents bring, blending custom and off-the-shelf agentic systems, and staying agile amid fast-evolving tech (while avoiding lock-ins).

- **Managing a new wave of risks.** Agents introduce a new class of systemic risks that traditional gen AI architectures, designed primarily for isolated LLM-centric use cases, were never built to handle: uncontrolled autonomy, fragmented system access, lack of observability and traceability, expanding surface of attack, and agent sprawl and duplication. What starts as intelligent automation can quickly become operational chaos—unless it is built on a foundation that prioritizes control, scalability, and trust.
- **Blending custom and off-the-shelf agents.** To fully capture the transformative potential of AI agents, organizations must go beyond simply activating agents embedded in software suites. These off-the-shelf agents may streamline routine workflows, but they rarely unlock strategic advantage. Realizing the full potential of agentic AI will require the development of custom-built agents for high-impact processes, such as end-to-end customer resolution, adaptive supply chain orchestration, or complex decision-making. These agents must be deeply aligned with the company’s logic, data flows, and value creation levers—making them difficult to replicate and uniquely powerful.
- **Staying agile amid fast-evolving tech.** Agentic AI is a new technology area, and solutions are evolving very rapidly. Agents will have to support workflows across multiple systems and should not be hardwired within a specific platform. An evolutive and vendor-agnostic architecture is therefore needed.

These challenges cannot be addressed by merely bolting new components, such as memory stores or orchestration engines, on top of existing gen AI stacks. While such capabilities are necessary, they are not sufficient. What’s needed is a fundamental architectural shift: from static, LLM-centric infrastructure to a dynamic, modular, and governed environment built specifically for agent-based intelligence—the agentic AI mesh.

The agentic AI mesh is a composable, distributed, and vendor-agnostic architectural paradigm that enables multiple agents to reason, collaborate, and act autonomously across a wide array of systems, tools, and language models—securely, at scale, and built to evolve with the technology. At the heart of this paradigm are five mutually reinforcing design principles:

- **Composability.** Any agent, tool, or LLM can be plugged into the mesh without system rework.
- **Distributed intelligence.** Tasks can be decomposed and resolved by networks of cooperating agents.
- **Layered decoupling.** Logic, memory, orchestration, and interface functions are decoupled to maximize modularity.
- **Vendor neutrality.** All components can be independently updated or replaced as technology advances, avoiding vendor lock-in and future-proofing the architecture. In particular, open standards such as the Model Context Protocol (MCP) and Agent2Agent (A2A) are preferred to proprietary protocols.
- **Governed autonomy.** Agent behavior is proactively controlled via embedded policies, permissions, and escalation mechanisms that ensure safe, transparent operation.

Share

Sidebar

## Seven interconnected capabilities of the AI agentic mesh

The emerging architecture for agentic AI relies on seven interconnected capabilities:

1. **Agent and workflow discovery** maintains a dynamic catalog of all organizational agents and workflows, enabling reuse across teams and enforcing policies on agent use.
2. **AI asset registry** centralizes governance of system prompts, agent instructions, large-language-model (LLM) configurations, tool definitions, and golden records while creating policies about version control and access.
3. **Observability** provides end-to-end tracing of workflows spanning agentic and procedural systems through standardized metrics, audit logs, and diagnostic capabilities.
4. **Authentication and authorization** enforce fine-grain access controls for communication among agentic systems, procedural systems, and LLMs, enforcing security policies and limiting the “blast radius” of compromised systems or agents.
5. **Evaluations** deliver comprehensive testing of agent pipelines to ensure accuracy and compliance over time.
6. **Feedback management** enables continuous improvement through automated feedback loops that capture performance metrics to evolve agent configurations.
7. **Compliance and risk management** embed policy controls, compliance agents, and ethical guardrails to ensure workflows meet regulatory and institutional standards.

The agentic AI mesh acts as the connective and orchestration layer that enables large-scale, intelligent agent ecosystems to operate safely and efficiently, and continuously evolve. It allows companies to coordinate custom-built and off-the-shelf agents within a unified framework, support multiagent collaboration by allowing agents to share context and delegate tasks, and mitigate key risks such as agent sprawl, autonomy drift, and lack of observability—all while preserving the agility required for a rapid technology evolution (see sidebar “Seven interconnected capabilities of the AI agentic mesh”).

Share

Sidebar

## Foundation models for agents: Five new requirements

For LLMs to function properly in the agentic age, they will need to evolve in a number of critical ways:

1. **Low-latency inference for real-time responsiveness.** Agents embedded in workflows (such as service operations or IT alerts) require subsecond response times with predictable latency, even under compute constraints. Illustrative examples of relevant models include Mistral Small (Mistral AI), Llama 3 8B (Meta), Gemini Nano (Google), and Claude Haiku (Anthropic).
2. **Fine-tuning and controllability for domain-specific agents.** Agents operating in regulated or knowledge-intensive domains (such as finance, legal, and healthcare) need large language models (LLMs) that can be fine-tuned, grounded in enterprise knowledge, and instrumented with external tools (such as RAG and APIs). Illustrative examples of relevant models are Mistral Small and Mistral 8x7B (open weight and fine-tunable, Mistral AI), and Llama 3 8B and 70B (fine-tunable, Meta).
3. **Lightweight deployment for embedded and edge agents.** In cases such as the Internet of Things, field devices, or privacy-sensitive environments, agents must be embedded directly into software or hardware, with minimal compute and memory footprint. Illustrative examples of relevant models include Mistral Small (Mistral AI), Gemini Nano (Google), Llama 3 8B (Meta), and Phi-2 (Microsoft).
4. **Scalable multiagent orchestration across the enterprise.** Enterprises deploying hundreds or thousands of agents require LLMs that can scale efficiently and cost-effectively, ideally using sparse architectures or a mixture of experts. Illustrative examples of relevant models include Mixtral (Mistral AI), Grok-1 (xAI), GPT-3.5 Turbo (OpenAI), and Command R+ (Cohere).
5. **Sovereignty, auditability, and geopolitical resilience for autonomous agents.** Agents embedded in core operations—particularly in public, financial, and critical-infrastructure sectors—must ensure compliance, data sovereignty, traceability, and geopolitical autonomy. This includes avoiding reliance on APIs that are hosted abroad, ensuring data residency, and resisting extraterritorial legal exposure (for example, OpenAI or Anthropic subject to US subpoenas). Illustrative examples of relevant models include Mistral Small/Mixtral (Mistral AI), Falcon 180B (TII UAE), and BloomZ/Bloom (BigScience).

Beyond this architectural evolution, organizations will also have to revisit their LLM strategies. At the core of every custom agent lies a foundation model—the reasoning engine that powers perception, decision-making, and interaction. In the agentic era, the requirements placed on LLMs evolve significantly. Agents are not passive copilots—they are autonomous, persistent, embedded systems. This creates five critical categories of LLM requirements, each aligned with specific deployment contexts, for which different kinds of models will be relevant (see sidebar “Foundational models for agents: Five new requirements”).

Finally, to truly scale agent deployment across the enterprise, the enterprise systems themselves must also evolve.

In the short term, APIs—protocols that allow different software applications to communicate and exchange data—will remain the primary interface for agents to interact with enterprise systems. But in the long term, APIs alone will not suffice. Organizations must begin reimagining their IT architectures around an agent-first model—one in which user interfaces, logic, and data access layers are natively designed for machine interaction rather than human navigation. In such a model, systems are no longer organized around screens and forms but around machine-readable interfaces, autonomous workflows, and agent-led decision flows.

This shift is already underway. Microsoft is embedding agents into the core of Dynamics 365 and Microsoft 365 via Copilot Studio; Salesforce is expanding Agentforce into a multiagent orchestration layer; SAP is rearchitecting its Business Technology Platform (BTP) to support agent integration through Joule. These changes signal a broader transition: The future of enterprise software is not just AI-augmented—it is agent-native.

## The main challenge won’t be technical—it will be human

As agents evolve from passive copilots to proactive actors—and scale across the enterprise—the complexity they introduce will be not only technical but mostly organizational. The real challenge lies in coordination, judgment, and trust. This organizational complexity will play out most visibly across three dimensions: how humans and agents cohabit day-to-day workflows; how organizations establish governance over systems that can act autonomously; and how they prevent unchecked sprawl as agent creation becomes increasingly democratized.

- **Human–agent cohabitation.** Agents won’t just assist humans—they’ll act alongside them. This raises nuanced questions about interaction and coexistence: When should an agent take initiative? When should it defer? How do we maintain human agency and oversight without slowing down the very benefits agents bring? Building clarity around these roles will take time, experimentation, and cultural adjustment. Trust won’t come from technical performance alone—it will hinge on how transparently agents communicate, how predictably they behave, and how intuitively they integrate into daily workflows.
- **Autonomy control.** What makes agents powerful—their ability to act independently—also introduces ambiguity. Unlike traditional tools, agents don’t wait to be instructed. They respond, adapt, and sometimes surprise. Navigating this new reality means confronting edge cases: What if an agent executes too aggressively? Or fails to escalate a subtle issue? The challenge is not to eliminate autonomy but to make it intelligible and aligned with organizational expectations. That alignment won’t be static. It will need to evolve as agents learn, systems shift, and trust deepens. Control mechanisms must also address the risk of hallucinations, or plausible but inaccurate outputs agents may produce.
- **Sprawl containment.** As in the early days of robotic process automation, there’s a real risk of agent sprawl—the uncontrolled proliferation of redundant, fragmented, and ungoverned agents across teams and functions. As low-code and no-code platforms make agent creation accessible to anyone, organizations risk a new kind of shadow IT: agents that multiply across teams, duplicate efforts, or operate without oversight. How do we avoid fragmentation? Who decides what gets built—and what gets retired? Without structured governance, design standards, and life cycle management, agent ecosystems can quickly become fragile, redundant, and unscalable.

Agents unlock the full potential of vertical use cases, offering companies a path to generate value well beyond efficiency gains. But realizing that potential requires a reimagined approach to AI transformation—one tailored to the unique nature of agents and capable of addressing the lingering limitations they alone cannot resolve. This approach is the subject of our next chapter.

Chapter 3

## AI transformation at a tipping point: The CEO mandate in the agentic era [AI transformation at a tipping point: The CEO mandate in the agentic era](https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage\#)

## Key Points

Continue to next section

Share

- **_Generating impact in the agentic era requires organizations to shift from scattered initiatives to strategic programs; from use cases to business processes; from siloed AI teams to cross-functional transformation squads; and from experimentation to industrialized, scalable delivery._**
- **_To scale agents, organizations will also need to set a new foundation by upskilling the workforce, adapting the technology infrastructure, and developing new governance structures for agents._**
- **_The time has come to bring the gen AI experimentation phase to an end—a pivot only the CEO can make._**

## Scaling impact in the agentic era requires a reset of the AI transformation approach

Unlike gen AI tools that could be easily plugged into existing workflows, AI agents demand a more foundational shift, one that requires rethinking business processes and enabling deep integration with enterprise systems. McKinsey has a [proven Rewired playbook for AI-driven transformations](https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/rewired-to-outcompete).11Eric Lamarre, Kate Smaje, and Rodney Zemmel, “ [Rewired to outcompete](https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/rewired-to-outcompete),” _McKinsey Quarterly_, June 20, 2023. To capitalize on the agentic opportunity, organizations must build on that, fundamentally reshaping their AI transformation approach across four dimensions:

## Four critical enablers are required to effectively operate in the agentic era

Redesigning the approach to AI transformation is an important step, but it is not enough. To unlock their full potential at scale, organizations must also activate a robust set of enablers that support the structural, cultural, and technical shifts required to integrate agents into day-to-day operations. These enablers span four dimensions—people, governance, technology architecture, and data—each of which is a foundation for scalable, secure, and high-impact deployment of agents across the enterprise.

- **People: Equip the workforce and introduce new roles.** The workforce must be equipped for new ways of working driven by human–agent collaboration. This involves fostering a “human + agent” mindset through cultural change, targeted training, and supporting early adopters as internal champions. New roles must also be introduced, such as prompt engineers to refine interactions, agent orchestrators to manage agent workflows, and human-in-the-loop designers to handle exceptions and build trust.
- **Governance: Ensure autonomy control and prevent agent sprawl.** With the rise of autonomous agents comes the need for strong governance to avoid risk and uncontrolled sprawl. Enterprises should define governance frameworks that establish agent autonomy levels, decision boundaries, behavior monitoring, and audit mechanisms. Policies for development, deployment, and usage must also be formalized, along with classification systems that group agents by function (such as task automators, domain orchestrators, and virtual collaborators), each with an appropriate oversight model.
- **Technology architecture: Build a foundation for interoperability and scale.** Agents, whether custom-built or off-the-shelf, must operate across a fragmented ecosystem of systems, data, and workflows. In the short term, organizations must evolve their AI architecture from LLM-centric setups to an agentic AI mesh. Beyond this first step, organizations should start preparing for their next-generation architecture, in which all enterprise systems will be reshuffled around agents in terms of user interface, business logic, and day-to-day operations.
- **Data: Accelerate data productization and address quality gaps in unstructured data.** Finally, agents depend on the quality and accessibility of enterprise data. Organizations must transition from use-case-specific data pipelines to reusable data products and extend data governance to unstructured data.
- **Strategy: From scattered tactical initiatives to strategic programs.** With agentic AI set to reshape the foundations of competition, organizations must move beyond bottom-up use case identification and directly align AI initiatives with their most critical strategic priorities. This means not only translating existing goals—such as enhancing operational efficiency, improving customer intimacy, or strengthening compliance—into AI-addressable transformation domains, but also adopting a forward-looking lens. Executives must challenge their organizations to look beyond the boundaries of today’s operating model and explore how AI can be used to reimagine entire segments of the business, create new revenue streams, and build competitive moats that will define leadership in the next decade.
- **Unit of transformation: From use case to business processes.** In the early wave of gen AI adoption, most vertical initiatives focused on plugging a solution into a specific step of an existing process—which tended to deliver narrow gains without changing the overall structure of how work is done. With AI agents, the paradigm shifts entirely. Opportunity now lies not in optimizing isolated tasks but in transforming entire business processes by embedding agents throughout the value chain. As a result, AI initiatives should no longer be scoped around a single use case, but instead around the end-to-end reinvention of a full process or persona journey. In vertical domains, this means moving from the question, “Where can I use AI in this function?” to “What would this function look like if agents ran 60 percent of it?” It involves rethinking workflows, decision logic, human–system interactions, and performance metrics across the board.
- **Delivery model: From siloed AI teams to cross-functional transformation squads.** AI centers of excellence have played a key role in accelerating AI awareness and experimentation across organizations. However, this model reaches its limits in the agentic era—in which agents are deeply embedded into enterprise systems, operate across complex business processes, and rely on high-quality data as their primary fuel. In this context, AI initiatives can no longer be delivered by isolated, specialized AI teams. To succeed at scale, organizations must shift to a cross-functional delivery model, anchored in durable transformation squads composed of business domain experts, process designers, AI and MLOps engineers, IT architects, software engineers, and data engineers.
- **Implementation process: From experimentation to industrialized, scalable delivery.** While the previous phase rightly focused on exploring the potential of gen AI, organizations must now shift to an industrialized delivery model, in which solutions are designed from the outset to scale, both technically and financially. This requires organizations to anticipate the full set of technical prerequisites for enterprise deployment—notably in terms of system integration, day-to-day monitoring, and release management, but also to rigorously estimate future running costs and design a solution to minimize them. Unlike traditional IT systems—for which [annual run costs typically represent 10 to 20 percent of initial build costs](https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/transforming-infrastructure-operations-for-a-hybrid-cloud-world) 12Aykut Atali, Chandra Gnanasambandam, and Bhargs Srivathsan, “ [Transforming infrastructure operations for a hybrid-cloud world](https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/transforming-infrastructure-operations-for-a-hybrid-cloud-world),” McKinsey, October 9, 2019.—gen AI solutions, especially at scale, can incur recurring costs that exceed the initial build investment. Designing for scalability must therefore include not just technical robustness but also economic sustainability, especially for high-volume applications.

## CEOs have a leadership challenge: Bringing the gen AI experimentation phase to a close

The rise of AI agents is more than just a technological shift. Agents represent a strategic inflection point that will redefine how companies operate, compete, and create value. To navigate this transition successfully, organizations must move beyond experimentation and pilot programs and enter a new phase of scaled, enterprise-wide transformation.

This pivot cannot be delegated—it must be initiated and led by the CEO. It will rely on three key actions:

- **Action 1: Conclude the experimentation phase and realign AI priorities.** Conduct a structured review to capture lessons learned, retire unscalable pilots, and formally close the exploratory phase. Refocus efforts on strategic AI programs targeting high-impact domains and processes.
- **Action 2: Redesign the AI governance and operating model.** Set up a strategic AI council involving business leaders, the chief human resources officer, the chief data officer, and the chief information officer. This council should oversee AI direction-setting; coordinate AI, IT, and data investments; and implement rigorous value-tracking mechanisms based on KPIs tied to business outcomes.
- **Action 3: Launch a first lighthouse transformation project and simultaneously initialize the agentic AI tech foundation.** Kick off a select number of high-impact agentic AI–driven workflow transformations in core business areas. In parallel, lay the groundwork for an agentic AI technology foundation by investing in key enablers—technology infrastructure, data quality, governance frameworks, and workforce readiness.

## Conclusion [Conclusion](https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage\#)

Like any truly disruptive technology, AI agents have the power to reshuffle the deck. Done right, they offer laggards a leapfrog opportunity to rewire their competitiveness. Done wrong—or not at all—they risk accelerating the decline of today’s market leaders. This is a moment of strategic divergence.

While the technology will continue to evolve, it is already mature enough to drive real, transformative change across industries. But to realize the full promise of agentic AI, CEOs must rethink their approach to AI transformation—not as a series of scattered pilots but as focused, end-to-end reinvention efforts. That means identifying a few business domains with the highest potential and pulling every lever: from reimagining workflows to redistributing tasks between humans and machines to rewiring the organization based on new operating models.

Some leaders are already moving—not just by deploying fleets of agents but by rewiring their organizations to harness their full disruptive potential. (Moderna, for example, merged its HR and IT leadership13Julien Dupont-Calbo, “L’IA n’est plus un outil, c’est un collègue”: Moderna fusionne sa DRH et sa DSI, \[“AI is no longer a tool, it’s a colleague”: Moderna merges its HR and IT departments\], _Les Echos,_ May 15, 2025.—signaling that AI is not just a technical tool but a workforce-shaping force.) This is a structural move toward a new kind of enterprise. Agentic AI is not an incremental step—it is the foundation of the next-generation operating model. CEOs who act now won’t just gain a performance edge. They will redefine how their organizations think, decide, and execute.

The time for exploration is ending. The time for transformation is now.

##### How relevant and useful is this article for you?

##### About the author(s)

**[Alexander Sukharevsky](https://www.mckinsey.com/our-people/alexander-sukharevsky)** is a senior partner in McKinsey’s London office, where **[Dave Kerr](https://www.mckinsey.com/our-people/dave-kerr)** is a partner; **[Klemens Hjartar](https://www.mckinsey.com/our-people/klemens-hjartar)** is a senior partner in the Copenhagen office; **[Lari Hämäläinen](https://www.mckinsey.com/our-people/lari-hamalainen)** is a senior partner in the Seattle office; **[Stéphane Bout](https://www.mckinsey.com/our-people/stephane-bout)** is a senior partner in the Lyon office; **[Vito Di Leo](https://www.mckinsey.com/our-people/vito-di-leo)** is a partner in the Zurich office; and **Guillaume Dagorret** is a senior fellow with the McKinsey Global Institute and is based in the Paris office.

The authors wish to thank Alena Fedorenko, Annie David, Clarisse Magnin, Lareina Yee, Larry Kanter, Michael Chui, Roger Roberts, Sarah Mulligan, Thomas Vlot, and Timo Mauerhoefer for their contributions to this report.

Talk to us

##### Explore a career with us

[Search openings](https://www.mckinsey.com/careers/search-jobs)

##### Related Articles

[![ Delicate blue and purple tendrils constantly expanding. These tendrils bear a resemblance to neurons, with luminous buds at their tips.](https://www.mckinsey.com/~/media/mckinsey/business%20functions/quantumblack/our%20insights/the%20state%20of%20ai/2025/the%20state%20of%20ai-2155840292-thumb-1536x1536.jpg?cq=50&mw=767&car=16:9&cpy=Center)](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai)

Survey

###### [The state of AI: How organizations are rewiring to capture value](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai)

[![Abstract image of colorful digital lines coming down and spreading forward like a highway.](https://www.mckinsey.com/~/media/mckinsey/business%20functions/quantumblack/our%20insights/superagency%20in%20the%20workplace%20empowering%20people%20to%20unlock%20ais%20full%20potential%20at%20work/superagency-report-1309018746-thumb-1536x1536.jpg?cq=50&mw=767&car=16:9&cpy=Center)](https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/superagency-in-the-workplace-empowering-people-to-unlock-ais-full-potential-at-work)

Report

###### [Superagency in the workplace: Empowering people to unlock AI’s full potential](https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/superagency-in-the-workplace-empowering-people-to-unlock-ais-full-potential-at-work)

[![Image of a hand tracing a light circle in the air](https://www.mckinsey.com/~/media/mckinsey/business%20functions/mckinsey%20digital/our%20insights/why%20agents%20are%20the%20next%20frontier%20of%20generative%20ai/genai-smart-agents-1207090508_1536x1536.png?cq=50&mw=767&car=16:9&cpy=Center)](https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/why-agents-are-the-next-frontier-of-generative-ai)

Article - _McKinsey Quarterly_

###### [Why agents are the next frontier of generative AI](https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/why-agents-are-the-next-frontier-of-generative-ai)

##### Sign up for emails on new Artificial Intelligence articles

Never miss an insight. We'll email you when new articles are published on this topic.

Subscribe

Sign up for emails on new Artificial Intelligence articles

[iframe](https://www.recaptcha.net/recaptcha/api2/anchor?ar=1&k=6LcWCs0UAAAAAEik2NaGkfGH8mGHo1ThxIt-qUoE&co=aHR0cHM6Ly93d3cubWNraW5zZXkuY29tOjQ0Mw..&hl=en&v=GUGrl5YkSwpBsxsF3eY665Ye&size=invisible&cb=phv4dxnqa7bt)

### Original URL
https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage
</details>

---

## Additional Sources Scraped

---
<details>
<summary>Building Effective AI Agents \ Anthropic</summary>

# Building effective agents

Published Dec 19, 2024

We've worked with dozens of teams building LLM agents across industries. Consistently, the most successful implementations use simple, composable patterns rather than complex frameworks.

Over the past year, we've worked with dozens of teams building large language model (LLM) agents across industries. Consistently, the most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns.

In this post, we share what we’ve learned from working with our customers and building agents ourselves, and give practical advice for developers on building effective agents.

## What are agents?

"Agent" can be defined in several ways. Some customers define agents as fully autonomous systems that operate independently over extended periods, using various tools to accomplish complex tasks. Others use the term to describe more prescriptive implementations that follow predefined workflows. At Anthropic, we categorize all these variations as **agentic systems**, but draw an important architectural distinction between **workflows** and **agents**:

- **Workflows** are systems where LLMs and tools are orchestrated through predefined code paths.
- **Agents**, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks.

Below, we will explore both types of agentic systems in detail. In Appendix 1 (“Agents in Practice”), we describe two domains where customers have found particular value in using these kinds of systems.

## When (and when not) to use agents

When building applications with LLMs, we recommend finding the simplest solution possible, and only increasing complexity when needed. This might mean not building agentic systems at all. Agentic systems often trade latency and cost for better task performance, and you should consider when this tradeoff makes sense.

When more complexity is warranted, workflows offer predictability and consistency for well-defined tasks, whereas agents are the better option when flexibility and model-driven decision-making are needed at scale. For many applications, however, optimizing single LLM calls with retrieval and in-context examples is usually enough.

## When and how to use frameworks

There are many frameworks that make agentic systems easier to implement, including:

- [LangGraph](https://langchain-ai.github.io/langgraph/) from LangChain;
- Amazon Bedrock's [AI Agent framework](https://aws.amazon.com/bedrock/agents/);
- [Rivet](https://rivet.ironcladapp.com/), a drag and drop GUI LLM workflow builder; and
- [Vellum](https://www.vellum.ai/), another GUI tool for building and testing complex workflows.

These frameworks make it easy to get started by simplifying standard low-level tasks like calling LLMs, defining and parsing tools, and chaining calls together. However, they often create extra layers of abstraction that can obscure the underlying prompts ​​and responses, making them harder to debug. They can also make it tempting to add complexity when a simpler setup would suffice.

We suggest that developers start by using LLM APIs directly: many patterns can be implemented in a few lines of code. If you do use a framework, ensure you understand the underlying code. Incorrect assumptions about what's under the hood are a common source of customer error.

See our [cookbook](https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/agents) for some sample implementations.

## Building blocks, workflows, and agents

In this section, we’ll explore the common patterns for agentic systems we’ve seen in production. We'll start with our foundational building block—the augmented LLM—and progressively increase complexity, from simple compositional workflows to autonomous agents.

### Building block: The augmented LLM

The basic building block of agentic systems is an LLM enhanced with augmentations such as retrieval, tools, and memory. Our current models can actively use these capabilities—generating their own search queries, selecting appropriate tools, and determining what information to retain.

We recommend focusing on two key aspects of the implementation: tailoring these capabilities to your specific use case and ensuring they provide an easy, well-documented interface for your LLM. While there are many ways to implement these augmentations, one approach is through our recently released [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol), which allows developers to integrate with a growing ecosystem of third-party tools with a simple [client implementation](https://modelcontextprotocol.io/tutorials/building-a-client#building-mcp-clients).

For the remainder of this post, we'll assume each LLM call has access to these augmented capabilities.

### Workflow: Prompt chaining

Prompt chaining decomposes a task into a sequence of steps, where each LLM call processes the output of the previous one. You can add programmatic checks (see "gate” in the diagram below) on any intermediate steps to ensure that the process is still on track.

**When to use this workflow:** This workflow is ideal for situations where the task can be easily and cleanly decomposed into fixed subtasks. The main goal is to trade off latency for higher accuracy, by making each LLM call an easier task.

**Examples where prompt chaining is useful:**

- Generating Marketing copy, then translating it into a different language.
- Writing an outline of a document, checking that the outline meets certain criteria, then writing the document based on the outline.

### Workflow: Routing

Routing classifies an input and directs it to a specialized followup task. This workflow allows for separation of concerns, and building more specialized prompts. Without this workflow, optimizing for one kind of input can hurt performance on other inputs.

**When to use this workflow:** Routing works well for complex tasks where there are distinct categories that are better handled separately, and where classification can be handled accurately, either by an LLM or a more traditional classification model/algorithm.

**Examples where routing is useful:**

- Directing different types of customer service queries (general questions, refund requests, technical support) into different downstream processes, prompts, and tools.
- Routing easy/common questions to smaller models like Claude 3.5 Haiku and hard/unusual questions to more capable models like Claude 3.5 Sonnet to optimize cost and speed.

### Workflow: Parallelization

LLMs can sometimes work simultaneously on a task and have their outputs aggregated programmatically. This workflow, parallelization, manifests in two key variations:

- **Sectioning**: Breaking a task into independent subtasks run in parallel.
- **Voting:** Running the same task multiple times to get diverse outputs.

**When to use this workflow:** Parallelization is effective when the divided subtasks can be parallelized for speed, or when multiple perspectives or attempts are needed for higher confidence results. For complex tasks with multiple considerations, LLMs generally perform better when each consideration is handled by a separate LLM call, allowing focused attention on each specific aspect.

**Examples where parallelization is useful:**

- **Sectioning**:
  - Implementing guardrails where one model instance processes user queries while another screens them for inappropriate content or requests. This tends to perform better than having the same LLM call handle both guardrails and the core response.
  - Automating evals for evaluating LLM performance, where each LLM call evaluates a different aspect of the model’s performance on a given prompt.
- **Voting**:
  - Reviewing a piece of code for vulnerabilities, where several different prompts review and flag the code if they find a problem.
  - Evaluating whether a given piece of content is inappropriate, with multiple prompts evaluating different aspects or requiring different vote thresholds to balance false positives and negatives.

### Workflow: Orchestrator-workers

In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results.

**When to use this workflow:** This workflow is well-suited for complex tasks where you can’t predict the subtasks needed (in coding, for example, the number of files that need to be changed and the nature of the change in each file likely depend on the task). Whereas it’s topographically similar, the key difference from parallelization is its flexibility—subtasks aren't pre-defined, but determined by the orchestrator based on the specific input.

**Example where orchestrator-workers is useful:**

- Coding products that make complex changes to multiple files each time.
- Search tasks that involve gathering and analyzing information from multiple sources for possible relevant information.

### Workflow: Evaluator-optimizer

In the evaluator-optimizer workflow, one LLM call generates a response while another provides evaluation and feedback in a loop.

**When to use this workflow:** This workflow is particularly effective when we have clear evaluation criteria, and when iterative refinement provides measurable value. The two signs of good fit are, first, that LLM responses can be demonstrably improved when a human articulates their feedback; and second, that the LLM can provide such feedback. This is analogous to the iterative writing process a human writer might go through when producing a polished document.

**Examples where evaluator-optimizer is useful:**

- Literary translation where there are nuances that the translator LLM might not capture initially, but where an evaluator LLM can provide useful critiques.
- Complex search tasks that require multiple rounds of searching and analysis to gather comprehensive information, where the evaluator decides whether further searches are warranted.

### Agents

Agents are emerging in production as LLMs mature in key capabilities—understanding complex inputs, engaging in reasoning and planning, using tools reliably, and recovering from errors. Agents begin their work with either a command from, or interactive discussion with, the human user. Once the task is clear, agents plan and operate independently, potentially returning to the human for further information or judgement. During execution, it's crucial for the agents to gain “ground truth” from the environment at each step (such as tool call results or code execution) to assess its progress. Agents can then pause for human feedback at checkpoints or when encountering blockers. The task often terminates upon completion, but it’s also common to include stopping conditions (such as a maximum number of iterations) to maintain control.

Agents can handle sophisticated tasks, but their implementation is often straightforward. They are typically just LLMs using tools based on environmental feedback in a loop. It is therefore crucial to design toolsets and their documentation clearly and thoughtfully. We expand on best practices for tool development in Appendix 2 ("Prompt Engineering your Tools").

**When to use agents:** Agents can be used for open-ended problems where it’s difficult or impossible to predict the required number of steps, and where you can’t hardcode a fixed path. The LLM will potentially operate for many turns, and you must have some level of trust in its decision-making. Agents' autonomy makes them ideal for scaling tasks in trusted environments.

The autonomous nature of agents means higher costs, and the potential for compounding errors. We recommend extensive testing in sandboxed environments, along with the appropriate guardrails.

**Examples where agents are useful:**

The following examples are from our own implementations:

- A coding Agent to resolve [SWE-bench tasks](https://www.anthropic.com/research/swe-bench-sonnet), which involve edits to many files based on a task description;
- Our [“computer use” reference implementation](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo), where Claude uses a computer to accomplish tasks.

## Combining and customizing these patterns

These building blocks aren't prescriptive. They're common patterns that developers can shape and combine to fit different use cases. The key to success, as with any LLM features, is measuring performance and iterating on implementations. To repeat: you should consider adding complexity _only_ when it demonstrably improves outcomes.

## Summary

Success in the LLM space isn't about building the most sophisticated system. It's about building the _right_ system for your needs. Start with simple prompts, optimize them with comprehensive evaluation, and add multi-step agentic systems only when simpler solutions fall short.

When implementing agents, we try to follow three core principles:

1. Maintain **simplicity** in your agent's design.
2. Prioritize **transparency** by explicitly showing the agent’s planning steps.
3. Carefully craft your agent-computer interface (ACI) through thorough tool **documentation and testing**.

Frameworks can help you get started quickly, but don't hesitate to reduce abstraction layers and build with basic components as you move to production. By following these principles, you can create agents that are not only powerful but also reliable, maintainable, and trusted by their users.

## Appendix 1: Agents in practice

Our work with customers has revealed two particularly promising applications for AI agents that demonstrate the practical value of the patterns discussed above. Both applications illustrate how agents add the most value for tasks that require both conversation and action, have clear success criteria, enable feedback loops, and integrate meaningful human oversight.

### A. Customer support

Customer support combines familiar chatbot interfaces with enhanced capabilities through tool integration. This is a natural fit for more open-ended agents because:

- Support interactions naturally follow a conversation flow while requiring access to external information and actions;
- Tools can be integrated to pull customer data, order history, and knowledge base articles;
- Actions such as issuing refunds or updating tickets can be handled programmatically; and
- Success can be clearly measured through user-defined resolutions.

Several companies have demonstrated the viability of this approach through usage-based pricing models that charge only for successful resolutions, showing confidence in their agents' effectiveness.

### B. Coding agents

The software development space has shown remarkable potential for LLM features, with capabilities evolving from code completion to autonomous problem-solving. Agents are particularly effective because:

- Code solutions are verifiable through automated tests;
- Agents can iterate on solutions using test results as feedback;
- The problem space is well-defined and structured; and
- Output quality can be measured objectively.

In our own implementation, agents can now solve real GitHub issues in the [SWE-bench Verified](https://www.anthropic.com/research/swe-bench-sonnet) benchmark based on the pull request description alone. However, whereas automated testing helps verify functionality, human review remains crucial for ensuring solutions align with broader system requirements.

## Appendix 2: Prompt engineering your tools

No matter which agentic system you're building, tools will likely be an important part of your agent. [Tools](https://www.anthropic.com/news/tool-use-ga) enable Claude to interact with external services and APIs by specifying their exact structure and definition in our API. When Claude responds, it will include a [tool use block](https://docs.anthropic.com/en/docs/build-with-claude/tool-use#example-api-response-with-a-tool-use-content-block) in the API response if it plans to invoke a tool. Tool definitions and specifications should be given just as much prompt engineering attention as your overall prompts. In this brief appendix, we describe how to prompt engineer your tools.

There are often several ways to specify the same action. For instance, you can specify a file edit by writing a diff, or by rewriting the entire file. For structured output, you can return code inside markdown or inside JSON. In software engineering, differences like these are cosmetic and can be converted losslessly from one to the other. However, some formats are much more difficult for an LLM to write than others. Writing a diff requires knowing how many lines are changing in the chunk header before the new code is written. Writing code inside JSON (compared to markdown) requires extra escaping of newlines and quotes.

Our suggestions for deciding on tool formats are the following:

- Give the model enough tokens to "think" before it writes itself into a corner.
- Keep the format close to what the model has seen naturally occurring in text on the internet.
- Make sure there's no formatting "overhead" such as having to keep an accurate count of thousands of lines of code, or string-escaping any code it writes.

One rule of thumb is to think about how much effort goes into human-computer interfaces (HCI), and plan to invest just as much effort in creating good _agent_-computer interfaces (ACI). Here are some thoughts on how to do so:

- Put yourself in the model's shoes. Is it obvious how to use this tool, based on the description and parameters, or would you need to think carefully about it? If so, then it’s probably also true for the model. A good tool definition often includes example usage, edge cases, input format requirements, and clear boundaries from other tools.
- How can you change parameter names or descriptions to make things more obvious? Think of this as writing a great docstring for a junior developer on your team. This is especially important when using many similar tools.
- Test how the model uses your tools: Run many example inputs in our [workbench](https://console.anthropic.com/workbench) to see what mistakes the model makes, and iterate.
- [Poka-yoke](https://en.wikipedia.org/wiki/Poka-yoke) your tools. Change the arguments so that it is harder to make mistakes.

While building our agent for [SWE-bench](https://www.anthropic.com/research/swe-bench-sonnet), we actually spent more time optimizing our tools than the overall prompt. For example, we found that the model would make mistakes with tools using relative filepaths after the agent had moved out of the root directory. To fix this, we changed the tool to always require absolute filepaths—and we found that the model used this method flawlessly.

### Original URL
https://www.anthropic.com/engineering/building-effective-agents
</details>

---
<details>
<summary>Real Agents vs. Workflows: The Truth Behind AI 'Agents'</summary>

This video provides a detailed explanation differentiating between what most people currently call "AI agents" and what a "real agent" truly entails, drawing heavily on perspectives from Anthropic and others in the AI community.

The video begins by stating that **what most people refer to as "agents" are often not true agents**. The speaker admits a change in perspective after seeing a recent article by Anthropic that aligns with a more rigorous definition.

Here's a breakdown of the key points:

### What is *Not* a Real Agent (Workflows)

The vast majority of what is labeled an "agent" falls into the category of **"workflows"**. These are described as:

*   **Simple API calls to a language model (LLM)**: This involves just a few lines of code and a prompt, allowing the system to reply to users but not act independently or make decisions.
*   **LLMs with tool access**: This involves giving an LLM access to tools and their documentation, such as the ability to execute SQL queries in a database or access private knowledge. The code for these actions is pre-programmed, with the LLM generating queries that the code then executes, and outputs are sent back to the LLM to formulate an answer.
*   **Hardcoded processes**: Workflows are essentially programs that are made and controlled by developers. They follow specific code lines and integrations, and their behavior is largely predictable, except for the LLM's outputs.
*   **Highly useful and powerful**: Despite not being "agents," workflows are "pretty damn useful" and can be "quite complex and advanced". They are responsible for most advanced AI applications seen today because they are **consistent and predictable**.
*   **Characteristics and Capabilities of Workflows**:
    *   Can implement **intelligent routers** to decide which tool to use and when, or which prompt to employ based on conditions.
    *   Can access various databases and decide which ones to query.
    *   Can execute tasks through action tools via code.
    *   Can work in parallel for efficiency.
    *   Can feature a **main orchestrator model** that selects different "fellow models" for specific tasks and synthesizes results (e.g., deciding if a user query needs a dataset query and then using a "SQL agent" to execute it).
*   **Examples of Workflows**:
    *   **ChatGPT**: Described as a workflow that sometimes uses "canvases" (likely referring to specific tools or modes) and sometimes directly answers questions, even complex ones. It is still all hardcoded.
    *   **Crew AI's "agents"**: These function like predefined workflows assigned to specific tasks.
    *   **Cursor**: Advanced workflows like Cursor are noted for not having as many issues as early agentic systems.
*   **When to use workflows**: If you know exactly what your system needs to do, even if it's advanced, a workflow is the appropriate solution.

### What *Is* a Real Agent

In contrast to workflows, a **real agent** is something much more sophisticated and independent.

*   **Independent Functionality**: A real agent functions independently.
*   **System 2 Thinking**: It is capable of **genuine reasoning, reflection, and recognizing when it lacks knowledge**. This is akin to "system 2 thinking," which is deliberate and reflective, as opposed to "system 1 thinking," which is fast, automatic, and purely based on patterns and learned responses. New models like OpenAI's new 01 and 03 series are beginning to explore system 2-like approaches by having models reason internally before generating responses.
*   **Dynamic Tool Usage and Decision-Making**: A real agent not only knows how to use tools but also **decides when and why to use them** based on deliberate reasoning. It dynamically directs its own processes and tool usage, maintaining control over how it accomplishes tasks.
*   **Planning and Interaction**: Real agents make a plan by exchanging information with the user, understanding their needs, and iterating at a reasoning level to decide on the steps to solve a problem. Ideally, they would even ask for more information or clarification instead of hallucinating.
*   **Replacing a Role/Task**: Agents can be seen as almost **replacing someone or a role in a workflow**. There is no hardcoded path; the agentic system makes its own decisions.
*   **Requirements**: Real agents require a **very powerful LLM**, better than current ones, and an environment in which to evolve, such as a discussion with a user, along with access to tools they can use whenever they see fit.

### Challenges and Current Limitations of Real Agents

Building true agents is significantly more difficult with current models, and the promise often outweighs the present capabilities.

*   **Difficulty in Achievement**: The vision of a flexible, adaptable, independent agent is far harder to achieve with current models.
*   **Reliability Issues**: Current language models are "strangely unreliable" and "get confused," despite showing "dramatically superhuman performance on evals".
*   **Practical Drawbacks**: True agentic systems are:
    *   More susceptible to failures.
    *   More expensive to run and use.
    *   Add latency.
    *   Their results, when they work, are often **completely inconsistent** and "not that exciting now".

### Examples of Agentic Systems and Their Shortcomings

The video provides examples of systems that aim to be true agents but still face significant challenges:

*   **Devin**: Positioned as a **fully autonomous software engineer** with its own computing environment, designed to handle tasks like API integrations and real-time problem-solving independently. While it "offers an intriguing glimpse" into agentic systems, extensive testing showed it **struggles with complex or autonomous tasks**, often providing overcomplicated solutions or pursuing unfeasible paths, despite excelling at simpler, well-defined tasks. Devin aligns with Anthropic's vision but struggles with inconsistency.
*   **Anthropic's Computer Use**: This system aimed to create an autonomous agent with access to a computer, embodying characteristics of a true agent: autonomous decision-making, dynamic tool usage, and environmental interaction. Despite initial hype and promising goals (to replace anyone on a computer), its "decline also serves as a reminder of the challenges in creating practical agentic systems that not only work as intended but do so systematically".

### Conclusion and Recommendations

The overarching message is that **LLMs are simply not ready yet for becoming true agents**.

*   For most problems, the recommendation is to **aim for the simplest possible solution** that can be easily iterated and debugged. This often means using simple LLM calls, perhaps complemented with external knowledge through retrieval systems or light fine-tuning.
*   The effort, time, and money for pursuing true agents should be reserved for **"really complex problems that cannot be solved otherwise"**.

### Sponsor Mention

The video also includes a sponsor, **Originality.ai**, an "awesome tool" designed to detect AI-generated content, check for plagiarism, grammar, readability, and fact-check work, all based on state-of-the-art language models and systems.

### Original URL
https://www.youtube.com/watch?v=kQxr-uOxw2o&t=1s
</details>

---
<details>
<summary>Build Production Agentic RAG With LLMOps at Its Core</summary>

## LLMOps for production agentic RAG

Welcome to Lesson 6 of Decoding ML’s **[Building Your Second Brain AI Assistant Using Agents, LLMs and RAG](https://github.com/decodingml/second-brain-ai-assistant-course/tree/main)** open-source course, where you will learn to architect and build a production-ready Notion-like AI research assistant.

Agents are the latest breakthrough in AI. For the first time in history, we give a machine complete control over its decisions without explicitly telling it. Agents do that through the LLM, the system's brain that interprets the queries and decides what to do next and through the tools that provide access to the external world, such as APIs and databases.

One of the agents' most popular use cases is Agentic RAG, in which agents access a tool that provides them with access to a vector database (or another type of database) to retrieve relevant context dynamically before generating an answer.

Agentic RAG differs from a standard RAG workflow in that the LLM can dynamically choose when it needs context or whether a single query to the database provides enough context.

Agents, relative to workflows, introduce even more randomness into the system. This is a core reason why adding LLMOp best practices such as prompt monitoring and LLM evaluation is a critical step in making your system easy to debug and maintain.

> LLMOps and evaluation are critical in any AI system, but they become even more crucial when working with agents!

In previous lessons of the course, we implemented all the offline pipelines that helped us prepare for advanced RAG, such as populating the MongoDB vector index with the proper data from our Second Brain and fine-tuning a summarization open-source small language model (SLM).

In this lesson, we will take the final step to glue everything together by adding an agentic layer on top of the vector index and an observability module on top of the agent to monitor and evaluate it. These elements will be part of our online inference pipelines, which will turn into the Second Brain AI assistant that the user interacts with, as seen in the demo below:

Thus, in this lesson, we will dive into the fundamentals of **agentic RAG**, exploring how agents powered by LLMs can go beyond traditional retrieval-based workflows to dynamically interact with multiple tools and external systems, such as vector databases.

Next, we will move to our **observability pipeline,** which evaluates the agents using techniques such as LLM-as-judges and heuristics to ensure they work correctly. We will monitor the prompt traces that power the agents to help us debug and understand what happens under the hood.

**While going through this lesson, we will learn the following:**

- Understand what an agent is, how it differs from workflows, and why it’s useful.
- Architect the Agentic RAG module, understanding its components and data flow.
- Build and monitor an agentic LLM application using SmolAgents and Opik.
- Implement prompt monitoring pipelines to track input/output, latency, and metadata.
- Explore RAG evaluation metrics like moderation, hallucination, and answer relevance.
- Create custom evaluation metrics, integrating heuristics and LLM judges.
- Automate observability, ensuring real-time performance tracking.
- Interact with the Second Brain AI assistant via CLI or a beautiful Gradio UI.

---

## 1\. Understanding how LLM-powered agents work

LLM-powered agents combine a **language model, tools, and memory** to process information and take action.

They don’t just generate text—they **reason, retrieve data, and interact with external systems** to complete tasks.

At its core, an agent takes in an input, analyzes what needs to be done, and decides the best way to respond. Instead of working in isolation, it can tap into external tools like APIs, databases, or plugins to enhance its capabilities.

With the reasoning power of LLMs, the agent doesn’t just react—it strategizes. It breaks down the task, plans the necessary steps, and takes action to get the job done efficiently.

The most popular way to design agents is by using the ReAct framework, which models the agent as follows:

- **act:** the LLM calls specific tools
- **observe:** pass the tool output back to the LLM
- **reason:** the LLM reason about the tool output to decide what to do next (e.g., call another tool or respond directly)

Now, let’s understand how agents and RAG fit together.

---

## 2\. Researching Agentic RAG

Unlike a traditional RAG setup's linear, step-by-step nature, Agentic RAG puts an agent at the center of decision-making.

Instead of passively retrieving and generating responses, the agent actively directs the process—deciding what to search for, how to refine queries, and when to use external tools, such as SQL, vector, or graph databases.

For example, instead of querying the vector database just once (what we usually do in a standard RAG workflow), the agent might decide that after its first query, it doesn’t have enough information to provide an answer, making another request to the vector database with a different query.

---

## 3\. Exploring the difference between agents and workflows

Now that we’ve explored LLM-powered agents and Agentic RAGs, let’s take a step back and look at a broader question: “ **How do agents differ from workflows?”** While both help automate tasks, they operate in fundamentally different ways.

A workflow follows a fixed, predefined sequence—every step is planned in advance, making it reliable but rigid (more similar to classic programming).

In contrast, an agent **dynamically decides** what to do next **based on reasoning,** memory, and available tools. Instead of just executing steps, it adapts, learns, and makes decisions on the fly.

Think of a workflow as an assembly line, executing tasks in order, while an agent is like an intelligent assistant, capable of adjusting its approach in real time. This flexibility makes agents powerful for handling unstructured, complex problems that require dynamic decision-making.

Therefore, the trade-off between reliability and adaptability is key—workflows offer stability but are rigid, while agents provide flexibility by making dynamic decisions at the cost of consistency.

Now that we understand the basics of working with agents, let’s dive into the architecture of our Second Brain agent.

---

## 4\. Architecting the Agentic RAG module

When architecting the Agentic RAG module, the goal is to build an intelligent system that efficiently combines retrieval, reasoning, and summarization to generate high-quality responses tailored to user queries.

#### What’s the interface of the pipeline?

The pipeline takes a user query as input (submitted through the Gradio UI).

The output is a refined answer generated by the agent after reasoning, retrieving relevant context from **[MongoDB](https://www.mongodb.com/products/platform/atlas-vector-search?utm_campaign=ai-pilot&utm_medium=creator&utm_term=iusztin&utm_source=blog)** through semantic search, and processing it through the summarization tool.

#### Offline vs. online pipelines

The Agentic RAG module fundamentally differs from the offline ML pipelines we’ve built in previous lessons.

This module is entirely decoupled from the pipelines in Lessons 1-5. It lives in a separate **[second-brain-online](https://github.com/decodingml/second-brain-ai-assistant-course/tree/main/apps/second-brain-online)** folder within our repository as its own standalone Python application.

This separation is intentional—by keeping the offline pipelines (feature and training) fully independent from the online inference system, we ensure a clean architectural divide.

As a quick reminder from Lesson 1, **offline pipelines** are batch pipelines that run on a schedule or trigger. They process input data and store the output artifacts in storage, allowing other pipelines or clients to consume them as needed.

These include the data collection pipeline, ETL pipeline, RAG feature pipeline, dataset generation pipeline, and training pipeline. They operate independently and are decoupled through various storage solutions such as document databases, vector databases, data registries, or model registries.

The Agentic RAG module, on the other hand, belongs to the category of **online pipelines**. It directly interacts with the user and must remain available at all times. The online pipelines available in this project are the agentic inference pipeline, the summarization inference pipeline, and the observability pipeline.

Unlike offline pipelines, these do not require orchestration and function similarly to RESTful APIs, ensuring minimal latency and efficient responses.

#### What does the pipeline’s architecture look like?

The Agentic RAG module operates in real time, instantly responding to user queries without redundant processing.

This module's core is an agent-driven system that reasons independently and dynamically invokes tools to handle user queries. They serve as extensions of the LLM model powering the agent, allowing it to perform tasks it wouldn’t efficiently handle on its own without specialized fine-tuning.

Our agent relies on three main components:

1. **The what can I do tool**, which helps users understand the usages of the system

2. **The retriever tool** that queries MongoDB’s vector index pre-populated during our offline processing

3. **The summarization tool** uses a REST API to call a different model specialized in summarizing web documents.

We specifically picked these ones as they are a perfect use case for showing how to use a tool that runs only with Python, one that calls a database, and one that calls an API (three of the most common scenarios).

The agent layer is powered by the **[SmolAgents](https://github.com/huggingface/smolagents)** framework (by Hugging Face) and orchestrates the reasoning process. A maximum number of steps can be set to ensure the reasoning remains focused and does not take unnecessary iterations to reach a response (avoiding skyrocketing bills).

To provide a seamless user experience, we integrated the agentic inference pipeline with a **[Gradio UI](https://www.gradio.app/)**, making interactions intuitive and accessible. This setup ensures that users can engage with the assistant as naturally as possible, simulating a conversational AI experience.

The interface allows us to track how the agent selects and uses tools during interactions.

For instance, we can see when it calls the **[MongoDB vector search tool](https://www.mongodb.com/products/platform/atlas-vector-search?utm_campaign=ai-pilot&utm_medium=creator&utm_term=iusztin&utm_source=blog)** to retrieve relevant data and how it cycles between retrieving information and reasoning before generating a response.

The agentic inference pipeline is designed to handle user queries in real time, orchestrating a seamless data flow from input to response. To understand how information moves through the system, we break down the interaction between the user, the retrieval process, and the summarization mechanism.

When a user submits a query through the **Gradio UI**, the **Agentic Layer**, an LLM-powered agent, dynamically determines the most suitable tool to process the request.

If additional context is required, the **Retriever Tool** fetches relevant information from the MongoDB vector database, extracting the most relevant chunks. This vector database was previously populated through the RAG feature pipeline in Lesson 5, ensuring the system has preprocessed, structured knowledge readily available for retrieval.

The retrieved data is then refined using the **Summarization Tool**, which enhances clarity before generating the final response. For summarization, we can choose between a custom Summarization Inference Pipeline, which is powered by the Hugging Face model we trained in Lesson 4, or an OpenAI model.

The agent continues reasoning iteratively until it reaches the predefined step limit or it decides it has the final answer, ensuring efficiency while maintaining high response quality.

As a side note, given the simplicity of our use case, the Second Brain AI assistant could have been implemented using a traditional workflow, directly retrieving and responding to queries without an agentic approach.

However, by embracing this modular strategy, we achieve greater scalability and flexibility, allowing the system to integrate new data sources or tools easily in the future.

Now that we understand how the agent works, let’s dig into how we can evaluate it and then into the implementation.

---

## 5\. Understanding how to evaluate an agentic RAG application

When evaluating an Agentic RAG application, it’s important to distinguish between two primary evaluation approaches: **LLM evaluation** and **Application/RAG evaluation**. Each serves a different purpose, and while LLM evaluation assesses the model in isolation, Application/RAG evaluation tests the entire application as a system.

In this case, our primary focus is evaluating the RAG pipeline as a black-box system, assessing how retrieval and reasoning work together to generate the final output.

However, we also provide a brief refresher on key insights from LLM evaluation in Lesson 4 to highlight its role in the broader evaluation process.

#### LLM evaluation

As a brief reminder, LLM evaluation measures response quality without retrieval. In Lesson 4, we tested this by analyzing the model’s ability to generate answers from its internal knowledge.

Popular methods for LLM evaluation include **benchmark-based evaluation** (using standardized datasets), **heuristic evaluation**(ROUGE, BLEU, regex matching, or custom heuristics), semantic-based evaluation (BERT Score), and **LLM-as-a-judge**, where another LLM evaluates the generated outputs.

Each method has strengths and trade-offs. Benchmark-based evaluation provides standardized comparisons but may not fully capture real-world performance, while heuristic methods may offer quick, interpretable insights but often fail to assess deeper contextual understanding. Additionally, LLM-as-a-judge is flexible and scalable, though it introduces potential biases from the evaluating model itself.

#### RAG evaluation

Unlike LLM evaluation, which assesses the model’s ability to generate responses from internal knowledge, RAG evaluation focuses on how well the retrieval and generation processes work together.

Evaluating a RAG application requires analyzing how different components interact. We focus on four key dimensions:

- **User input** – The query submitted by the user.
- **Retrieved context** – The passages or documents fetched from the vector database.
- **Generated output**– The final response produced by the LLM based on retrieved information.
- **Expected output** – The ideal or ground-truth answer, if available, for comparison.

By evaluating these dimensions, we can determine whether the retrieved context is relevant, the response is grounded in the retrieved data, and the system generates complete and accurate answers.

As mentioned, we break the process into two key steps to evaluate a RAG application correctly: retrieval and generation.

Since RAG applications rely on retrieving relevant documents before generating responses, retrieval quality plays a critical role in overall performance. If the retrieval step fails, the LLM will either generate incorrect answers or hallucinate information.

To assess **retrieval step** effectiveness, we can use various ranking-based metrics, including:

- **NDCG (Normalized Discounted Cumulative Gain)** – Measures how well the retrieved documents are ranked, prioritizing the most relevant ones at the top.
- **MRR (Mean Reciprocal Rank)** – Evaluates how early the first relevant document appears in the retrieved results, ensuring high-ranking relevance.

Another option is to visualize the embedding from your vector index (using algorithms such as t-SNE or UMAP) to see if there are any meaningful clusters within your vector space.

On the other hand, during **the generation step**, you can leverage similar strategies we looked at in the LLM evaluation subsection while considering the context dimension.

#### LLM application evaluation

For LLM application evaluation, we take a black-box approach, meaning we assess the entire Agentic RAG module rather than isolating individual components.

We evaluate the entire system by analyzing the input, output, and retrieved context instead of separating retrieval and generation into independent evaluations.

This approach allows us to identify system-wide failures and measure how well the retrieved knowledge contributes to generating accurate and relevant responses.

By evaluating the entire module, we can detect common RAG issues, such as hallucinations caused by missing context or low retrieval recall leading to incomplete answers, ensuring the system performs reliably in real-world scenarios.

#### **How many samples do we need to evaluate our LLM app?**

Naturally, using too few samples for evaluation can lead to misleading conclusions. For example, 5-10 examples are insufficient for capturing meaningful patterns, while 30-50 examples provide a reasonable starting point for evaluation.

Ideally, a dataset of over 400 samples ensures a more comprehensive assessment, helping to uncover biases and edge cases.

#### What else should be monitored along the LLM outputs?

Beyond output quality, **system performance metrics** like latency, throughput, reliability, and costs should be tracked to ensure scalability.

Additionally, **business metrics**—such as conversion rates, user engagement, or behavior influenced by the assistant—help measure the real-world impact of the LLM application.

#### Popular evaluation tools

Several tools specialize in RAG and LLM evaluation, offering similar capabilities for assessing retrieval quality and model performance.

For RAG evaluation, **RAGAS** is widely used to assess retrieval-augmented models, while **ARES** focuses on measuring how well the retrieved context supports the generated response.

**[Opik](https://github.com/comet-ml/opik)** stands out as an open-source solution that provides structured evaluations, benchmarking, and observability for LLM applications, ensuring assessment transparency and consistency.

Other proprietary alternatives include **Langfuse**, **Langsmith**, which is deeply integrated into the LangChain ecosystem for debugging and evaluation, and **Phoenix**.

---

## 6\. Architecting the observability pipeline

​In our observability pipeline, implemented with **[Opik](https://github.com/comet-ml/opik)**, we combine monitoring and evaluation to ensure our application runs smoothly. Monitoring tracks all activities, while evaluation assesses performance and correctness.

#### What’s the interface of the pipeline?

LLMOps observability pipelines consist of two parts: one for monitoring prompts and another for evaluating the RAG module. These pipelines help us track system performance and ensure the application remains reliable.

The **prompt monitoring pipeline** captures entire prompt traces and metadata, such as prompt templates or models used within the chain. It also logs latency and system behavior while providing structured insights through dashboards that help detect and resolve inefficiencies.

The **RAG evaluation pipeline** tests the agentic RAG module using heuristics and LLM judges to assess performance. It receives a set of evaluation prompts and processes them to evaluate accuracy and reasoning quality. The pipeline outputs accuracy assessments, quality scores, and alerts for performance issues, helping maintain system reliability.

We utilize **[Opik](https://github.com/comet-ml/opik)** (by **[Comet ML](https://www.comet.com/site/products/ml-experiment-tracking?utm_source=paul_2nd_brain_course&utm_campaign=opik&utm_medium=lessons)**), an open-source platform, to handle both the monitoring and evaluation of our application. Opik offers comprehensive tracing, automated evaluations, and production-ready dashboards, making it an ideal choice for our needs.

For evaluation, Opik automates performance assessments using both built-in and custom metrics. Users can define a threshold for any metric and configure alerts for immediate intervention if performance falls below the set value.

Now that we have an overview of the interfaces and components let’s dive into more details about each of the 2 pipelines.

#### **The prompt monitoring pipeline**

This component logs and monitors prompt traces. Prompt monitoring is essential to understand how our application interacts with users and identify areas for improvement. By tracking prompts and responses, we can debug issues in LLM reasoning or other issues like latency and costs.

Opik enables us to monitor latency across every phase of the generation process—pre-generation, generation, and post-generation—ensuring our application responds promptly to user inputs. ​

Latency is crucial to the user experience, as it includes multiple factors such as Time to First Token (TTFT), Time Between Tokens (TBT), Tokens Per Second (TPS), and Total Latency. Tracking these metrics helps us optimize response generation and manage hosting costs effectively.

#### **The RAG evaluation pipeline**

As previously mentioned, the RAG evaluation pipeline assesses the performance of our agentic RAG module, which performs application/RAG evaluation.

The pipeline uses built-in heuristics such as Hallucination, Answer Relevance, and Moderation to evaluate response quality. Additionally, we define and integrate a custom metric and LLM judge, which assesses if the LLM's output has appropriate length and density.

This flow can also run as an offline batch pipeline during development to assess performance on test sets. Additionally, it integrates into the CI/CD pipeline to test the RAG application before deployment, ensuring any issues are identified early (similar to integration tests).

Post-deployment, it can run on a schedule to evaluate random samples from production, maintaining consistent application performance. If metrics fall below a certain threshold, we can hook an alarm system that notifies us to address potential issues promptly.

By implementing these components with Opik, we maintain a robust observability pipeline that ensures our application operates efficiently.

A final note is how similar a prompt management tool, such as Opik, is to more standard experiment tracking tools, such as Comet, W&B and MLFlow. But instead of being focused on simple metrics, it’s built around the prompts as their first-class citizen.

---

## 7\. Implementing the agentic RAG module

Now that we’ve understood what it takes to build the agentic RAG and observability pipelines, let’s start implementing them.

The agentic RAG module is implemented using the SmolAgents Hugging Face frame, to build an agent that utilizes three key tools: the MongoDB retriever, the summarizer, and the "What Can I Do" tool.

Since prompt monitoring is closely tied to agent execution, here we will also cover how the system logs input/output data, latency, and other key details for each tool, ensuring full observability with Opik.

#### Building the agent

The core of our agentic RAG module starts with `get_agent()`, a method responsible for initializing the agent:

```
def get_agent(retriever_config_path: Path) -> "AgentWrapper":
    agent = AgentWrapper.build_from_smolagents(
        retriever_config_path=retriever_config_path
    )
    return agent
```

This function builds an `AgentWrapper`, which is a custom class we implemented that extends the agent's functionality by incorporating Opik for tracking all the agent’s interactions.

Building the agent requires a retriever configuration to create the MongoDB retriever tool. As a reminder from Lesson 5, we support multiple retrieval strategies based on retriever type (e.g., parent or contextual), embedding models, and other parameters.

> _**Note**: The retrieval setup is essentially copied from the offline Second Brain app in Lesson 5, ensuring consistency in document search and retrieval methods. This means the retriever is loaded exactly as it was implemented in the previous version, preserving the same retrieval logic and configurations._

#### Wrapping the agent for monitoring

The `AgentWrapper` class extends the base agent to incorporate metadata tracking with Opik. This ensures that every action taken by the agent is logged and traceable:

```
class AgentWrapper:
    def __init__(self, agent: MultiStepAgent) -> None:
        self.__agent = agent

    @property
    def input_messages(self) -> list[dict]:
        return self.__agent.input_messages

    @property
    def agent_name(self) -> str:
        return self.__agent.agent_name

    @property
    def max_steps(self) -> str:
        return self.__agent.max_steps
```

We use composition to wrap the `MultiStepAgent` from SmolAgents and expose its properties. The `MultiStepAgent` enables our agent to execute multi-step reasoning and decision-making processes.

Next, we define a method to build the agent, specifying the retriever configuration and integrating the 3 tools necessary for execution:

```
@classmethod
    def build_from_smolagents(cls, retriever_config_path: Path) -> "AgentWrapper":
        retriever_tool = MongoDBRetrieverTool(config_path=retriever_config_path)
        if settings.USE_HUGGINGFACE_DEDICATED_ENDPOINT:
            logger.warning(
                f"Using Hugging Face dedicated endpoint as the summarizer with URL: {settings.HUGGINGFACE_DEDICATED_ENDPOINT}"
            )
            summarizer_tool = HuggingFaceEndpointSummarizerTool()
        else:
            logger.warning(
                f"Using OpenAI as the summarizer with model: {settings.OPENAI_MODEL_ID}"
            )
            summarizer_tool = OpenAISummarizerTool(stream=False)

        model = LiteLLMModel(
            model_id=settings.OPENAI_MODEL_ID,
            api_base="https://api.openai.com/v1",
            api_key=settings.OPENAI_API_KEY,
        )

        agent = ToolCallingAgent(
            tools=[what_can_i_do, retriever_tool, summarizer_tool],
            model=model,
            max_steps=3,
            verbosity_level=2,
        )

        return cls(agent)
```

This method builds the agent by selecting the retriever configuration, which defines how the MongoDB retriever tool is created and configured.

> **It’s critical** that the retriever config matches the one used during the RAG feature pipeline used to populate the MongoDB vector index.

Next, we build the summarizer tool, which can either be the custom model trained in Lesson 4 and deployed on Hugging Face or an OpenAI model, depending on the settings.

After that, we initialize the LiteLLM model, which powers our AI agent.

Finally, all tools, along with the LLM model, are wrapped inside a `ToolCallingAgent` class with a maximum of three reasoning steps, ensuring structured decision-making and controlled execution flow.

Now that our agent is built, we can define its run function:

```
@opik.track(name="Agent.run")
    def run(self, task: str, **kwargs) -> Any:
        result = self.__agent.run(task, **kwargs)

        model = self.__agent.model
        metadata = {
            "system_prompt": self.__agent.system_prompt,
            "system_prompt_template": self.__agent.system_prompt_template,
            "tool_description_template": self.__agent.tool_description_template,
            "tools": self.__agent.tools,
            "model_id": self.__agent.model.model_id,
            "api_base": self.__agent.model.api_base,
            "input_token_count": model.last_input_token_count,
            "output_token_count": model.last_output_token_count,
        }
        if hasattr(self.__agent, "step_number"):
            metadata["step_number"] = self.__agent.step_number
        opik_context.update_current_trace(
            tags=["agent"],
            metadata=metadata,
        )

        return result
```

The `run` method tracks every execution of the agent using Opik’s `@track()` decorator. It logs key metadata, including the system prompt, tool descriptions, model details, and token counts within the current trace.

Having the skeleton of our agent in place, we can dig into each of the 3 tools that our model calls.

#### Building the MongoDB retriever tool

The first tool integrated is the `MongoDBRetrieverTool`, which allows the agent to find relevant documents using semantic search.

It matches a user query to the most relevant stored documents, helping the agent retrieve context when needed.

To integrate the tool with our agent, we must inherit from the `Tool ` class from SmolAgents. We also have to specify the name, description, inputs, and output type that the LLM uses to infer what the tool does and what its interface is. These are critical elements in integrating your tool with an LLM, as they are the only properties used to integrate the tool with the LLM:

```
class MongoDBRetrieverTool(Tool):
    name = "mongodb_vector_search_retriever"
    description = """Use this tool to search and retrieve relevant documents from a knowledge base using semantic search.
    This tool performs similarity-based search to find the most relevant documents matching the query.
    Best used when you need to:
    - Find specific information from stored documents
    - Get context about a topic
    - Research historical data or documentation
    The tool will return multiple relevant document snippets."""

    inputs = {
        "query": {
            "type": "string",
            "description": """The search query to find relevant documents for using semantic search.
            Should be a clear, specific question or statement about the information you're looking for.""",
        }
    }
    output_type = "string"

    def __init__(self, config_path: Path, **kwargs):
        super().__init__(**kwargs)

        self.config_path = config_path
        self.retriever = self.__load_retriever(config_path)

    def __load_retriever(self, config_path: Path):
        config = yaml.safe_load(config_path.read_text())
        config = config["parameters"]

        return get_retriever(
            embedding_model_id=config["embedding_model_id"],
            embedding_model_type=config["embedding_model_type"],
            retriever_type=config["retriever_type"],
            k=5,
            device=config["device"],
        )

```

The retriever tool is initialized with parameters from one of the retriever config files defined in Lesson 5. The settings include essential parameters such as the embedding model and retrieval type.

Now, we get to the core part of the tool, which is the `forward` method. This method is called when the AI agent uses the tool to search for information.

The `forward` method takes a query from the agent, searches for relevant documents, and returns the results in a format the agent can use.

The method is decorated with `@track`, which means its performance is being monitored with Opik. Before performing the actual search, the method first extracts important search parameters:

```
@track(name="MongoDBRetrieverTool.forward")
    def forward(self, query: str) -> str:
        if hasattr(self.retriever, "search_kwargs"):
            search_kwargs = self.retriever.search_kwargs
        else:
            try:
                search_kwargs = {
                    "fulltext_penalty": self.retriever.fulltext_penalty,
                    "vector_score_penalty": self.retriever.vector_penalty,
                    "top_k": self.retriever.top_k,
                }
            except AttributeError:
                logger.warning("Could not extract search kwargs from retriever.")

                search_kwargs = {}

        opik_context.update_current_trace(
            tags=["agent"],
            metadata={
                "search": search_kwargs,
                "embedding_model_id": self.retriever.vectorstore.embeddings.model,
            },
        )

```

First, we check what type of retriever is used and extract the relevant search parameters. Different retrievers might have different ways of configuring searches, so this code handles various cases.

The key parameters being extracted include:

- `fulltext_penalty`: Adjusts how much weight is given to exact text matches
- `vector_score_penalty`: Influences how semantic similarity affects the ranking
- `top_k`: Determines how many search results to return

These parameters significantly impact the search results. For example, a higher vector score penalty might prioritize results that match the semantic meaning of the query over those with exact keyword matches.

After setting up tracking, the method parses the query, performs the actual search, and formats the results:

```
 try:
            query = self.__parse_query(query)
            relevant_docs = self.retriever.invoke(query)

            formatted_docs = []
            for i, doc in enumerate(relevant_docs, 1):
                formatted_docs.append(
                    f"""
<document id="{i}">
<title>{doc.metadata.get("title")}</title>
<url>{doc.metadata.get("url")}</url>
<content>{doc.page_content.strip()}</content>
</document>
"""
                )

            result = "\n".join(formatted_docs)
            result = f"""
<search_results>
{result}
</search_results>
When using context from any document, also include the document URL as reference, which is found in the <url> tag.
"""
            return result
        except Exception:
            logger.opt(exception=True).debug("Error retrieving documents.")

            return "Error retrieving documents."
```

In this code snippet, we search for documents that match the query and format them in an XML-like structure. Each document includes a title, URL, and content. Additionally, the results are wrapped in tags to make them easy for the AI agent to read.

#### Creating the summarizer tool

In our agentic RAG module, we provide two summarization options: one using Hugging Face’s API and another using OpenAI’s models. Both tools inherit from `Tool` in SmolAgents and are tracked by Opik, ensuring that every summarization step is logged and monitored.

The first option for summarization is the Hugging Face endpoint-based summarizer.

This tool sends the text to an external Hugging Face model that generates a concise summary. The model deployed on Hugging Face is the one we trained in Lesson 4, which was explicitly fine-tuned for document summarization.

```
class HuggingFaceEndpointSummarizerTool(Tool):
    name = "huggingface_summarizer"
    description = """Use this tool to summarize a piece of text. Especially useful when you need to summarize a document."""

    inputs = {
        "text": {
            "type": "string",
            "description": """The text to summarize.""",
        }
    }
    output_type = "string"

    SYSTEM_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a helpful assistant specialized in summarizing documents. Generate a concise TL;DR summary in markdown format having a maximum of 512 characters of the key findings from the provided documents, highlighting the most significant insights

### Input:
{content}

### Response:
"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert settings.HUGGINGFACE_ACCESS_TOKEN is not None, (
            "HUGGINGFACE_ACCESS_TOKEN is required to use the dedicated endpoint. Add it to the .env file."
        )
        assert settings.HUGGINGFACE_DEDICATED_ENDPOINT is not None, (
            "HUGGINGFACE_DEDICATED_ENDPOINT is required to use the dedicated endpoint. Add it to the .env file."
        )

        self.__client = OpenAI(
            base_url=settings.HUGGINGFACE_DEDICATED_ENDPOINT,
            api_key=settings.HUGGINGFACE_ACCESS_TOKEN,
        )
```

The code snippet above initializes the Hugging Face summarizer tool. It verifies that the necessary API credentials are available before setting up the client connection to Hugging Face’s inference endpoint.

To generate a summary, we implement the `forward` method, which is tracked by Opik for monitoring:

```
@track
    def forward(self, text: str) -> str:
        result = self.__client.chat.completions.create(
            model="tgi",
            messages=[\
                {\
                    "role": "user",\
                    "content": self.SYSTEM_PROMPT.format(content=text),\
                },\
            ],
        )

        return result.choices[0].message.content
```

This function sends the input text to the Hugging Face API, applying the predefined system prompt. The generated response is then returned, providing a structured summary.

The second summarization option uses OpenAI’s models to generate summaries. It follows a similar structure to the Hugging Face summarizer but connects to OpenAI’s API instead.

```
class OpenAISummarizerTool(Tool):
    name = "openai_summarizer"
    description = """Use this tool to summarize a piece of text. Especially useful when you need to summarize a document or a list of documents."""

    inputs = {
        "text": {
            "type": "string",
            "description": """The text to summarize.""",
        }
    }
    output_type = "string"

    SYSTEM_PROMPT = """You are a helpful assistant specialized in summarizing documents.
Your task is to create a clear, concise TL;DR summary in plain text.
Things to keep in mind while summarizing:
- titles of sections and sub-sections
- tags such as Generative AI, LLMs, etc.
- entities such as persons, organizations, processes, people, etc.
- the style such as the type, sentiment and writing style of the document
- the main findings and insights while preserving key information and main ideas
- ignore any irrelevant information such as cookie policies, privacy policies, HTTP errors,etc.

Document content:
{content}

Generate a concise summary of the key findings from the provided documents, highlighting the most significant insights and implications.
Return the document in plain text format regardless of the original format.
"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.__client = OpenAI(
            base_url="https://api.openai.com/v1",
            api_key=settings.OPENAI_API_KEY,
        )

```

This summarizer connects to OpenAI’s API and uses a structured prompt to generate high-quality summaries.

Note that because the Hugging Face model was fine-tuned on summarizing documents, it doesn't require careful prompt engineering for the desired results (it has the logic embedded into it), resulting in fewer tokens/requests, which translates to lower costs and better latencies.

#### The "What Can I Do" tool

The third and last integrated tool is the "What Can I Do" tool, which provides a list of available capabilities within the Second Brain assistant and helps users explore relevant topics.

```
@opik.track(name="what_can_i_do")
@tool
def what_can_i_do(question: str) -> str:
    """Returns a comprehensive list of available capabilities and topics in the Second Brain system.

    This tool should be used when:
    - The user explicitly asks what the system can do
    - The user asks about available features or capabilities
    - The user seems unsure about what questions they can ask
    - The user wants to explore the system's knowledge areas

    This tool should NOT be used when:
    - The user asks a specific technical question
    - The user already knows what they want to learn about
    - The question is about a specific topic covered in the knowledge base

    Args:
        question: The user's query about system capabilities. While this parameter is required,
                 the function returns a standard capability list regardless of the specific question.

    Returns:
        str: A formatted string containing categorized lists of example questions and topics
             that users can explore within the Second Brain system.

    Examples:
        >>> what_can_i_do("What can this system do?")
        >>> what_can_i_do("What kind of questions can I ask?")
        >>> what_can_i_do("Help me understand what I can learn here")
    """

    return """
You can ask questions about the content in your Second Brain, such as:

Architecture and Systems:
- What is the feature/training/inference (FTI) architecture?
- How do agentic systems work?
- Detail how does agent memory work in agentic applications?

LLM Technology:
- What are LLMs?
- What is BERT (Bidirectional Encoder Representations from Transformers)?
- Detail how does RLHF (Reinforcement Learning from Human Feedback) work?
- What are the top LLM frameworks for building applications?
- Write me a paragraph on how can I optimize LLMs during inference?

RAG and Document Processing:
- What tools are available for processing PDFs for LLMs and RAG?
- What's the difference between vector databases and vector indices?
- How does document chunk overlap affect RAG performance?
- What is chunk reranking and why is it important?
- What are advanced RAG techniques for optimization?
- How can RAG pipelines be evaluated?

Learning Resources:
- Can you recommend courses on LLMs and RAG?
"""
```

This tool is useful when users are unsure about what they can ask or want to explore different capabilities within the system. Like other tools, it is tracked by Opik for monitoring and observability.

---

## 8\. Building the LLM evaluation pipeline

Now that we have implemented the agentic RAG module, we need a structured way to evaluate its performance. This is where the **LLM evaluation pipeline** comes in, ensuring that our agentic RAG module consistently meets quality and reliability standards.

The evaluation pipeline is built using Opik, which helps us log, analyze, and score the agent’s responses. We will focus strictly on Opik's evaluation logic and how it tracks our agent’s outputs.

Before evaluating our agent, we first need to gather a suitable evaluation dataset. This dataset will help us consistently test performance and track improvements

#### Creating the evaluation dataset

To evaluate the agent properly, we use a dataset of ~30 predefined prompts that cover various scenarios the agent might encounter. This dataset allows us to consistently test our agent’s performance across different iterations, ensuring that changes do not degrade its capabilities.

```
EVALUATION_PROMPTS: List[str] = [\
    """\
Write me a paragraph on the feature/training/inference (FTI) pipelines architecture following the next structure:\
\
- introduction\
- what are its main components\
- why it's powerful\
\
Retrieve the sources when compiling the answer. Also, return the sources you used as context.\
""",\
    "What is the feature/training/inference (FTI) pipelines architecture?",\
    "What is the Tensorflow Recommenders Python package?",\
    """How does RLHF: Reinforcement Learning from Human Feedback work?\
\
Explain to me:\
- what is RLHF\
- how it works\
- why it's important\
- what are the main components\
- what are the main challenges\
""",\
    "List 3 LLM frameworks for building LLM applications and why they are important.",\
    "Explain how does Bidirectional Encoder Representations from Transformers (BERT) work. Focus on what architecture it uses, how it's different from other models and how they are trained.",\
    "List 5 ways or tools to process PDFs for LLMs and RAG",\
    """How can I optimize my LLMs during inference?\
\
Provide a list of top 3 best practices, while providing a short explanation for each, which contains why it's important.\
""",\
    "Explain to me in more detail how does an Agent memory work and why do we need it when building Agentic apps.",\
    "What is the difference between a vector database and a vector index?",\
    "Recommend me a course on LLMs and RAG",\
    "How Document Chunk overlap affects a RAG pipeline and it's performance?",\
    """What is the importance of reranking chunks for RAG?\
Explain to me:\
- what is reranking\
- how it works\
- why it's important\
- what are the main components\
- what are the main trade-offs\
""",\
    "List the most popular advanced RAG techniques to optimize RAG performance and why they are important.",\
    "List what are the main ways of evaluating a RAG pipeline and why they are important.",\
]
```

We could have added more samples, but for the first iteration, having 30 samples is a sweet spot. The core idea is to expand this split with edge case samples you find while developing the application.

We use Opik to store and manage the dataset, as shown in the following code:

```
def get_or_create_dataset(name: str, prompts: list[str]) -> opik.Dataset | None:
    client = opik.Opik()
    try:
        dataset = client.get_dataset(name=name)
    except Exception:
        dataset = None

    if dataset:
        logger.warning(f"Dataset '{name}' already exists. Skipping dataset creation.")

        return dataset

    assert prompts, "Prompts are required to create a dataset."

    dataset_items = []
    for prompt in prompts:
        dataset_items.append(
            {
                "input": prompt,
            }
        )

    dataset = create_dataset(
        name=name,
        description="Dataset for evaluating the agentic app.",
        items=dataset_items,
    )

    return dataset
```

This function ensures the dataset is created if it doesn’t exist, avoiding unnecessary duplication. It logs whether the dataset is new or previously stored and ensures that each prompt is properly formatted before evaluation.

#### Evaluating the agent

The core of the evaluation pipeline is the `evaluate_agent()` function. This function runs the set of predefined prompts through our agent and scores its responses using a combination of built-in and custom metrics.

```
def evaluate_agent(prompts: list[str], retriever_config_path: Path) -> None:
    assert settings.COMET_API_KEY, (
        "COMET_API_KEY is not set. We need it to track the experiment with Opik."
    )

    logger.info("Starting evaluation...")
    logger.info(f"Evaluating agent with {len(prompts)} prompts.")

    def evaluation_task(x: dict) -> dict:
        """Call agentic app logic to evaluate."""
        agent = agents.get_agent(retriever_config_path=retriever_config_path)
        response = agent.run(x["input"])
        context = extract_tool_responses(agent)

        return {
            "input": x["input"],
            "context": context,
            "output": response,
        }
```

In this code section, we first ensure that Opik can log the experiment by asserting that the necessary API keys are set.

Then, we define the `evaluation_task()`, a method that retrieves an instance of our agent, runs an input prompt through it, and captures both the output and retrieval context.

Before running the actual evaluation, we either fetch an existing dataset or create a new one to store our evaluation prompts:

```
# Get or create dataset
    dataset_name = "second_brain_rag_agentic_app_evaluation_dataset"
    dataset = opik_utils.get_or_create_dataset(name=dataset_name, prompts=prompts)
```

Here, `opik_utils.get_or_create_dataset()` is used to manage the datasets dynamically, as detailed earlier in this section.

Once the dataset is set up, we retrieve our agent instance and configure the experiment. The `experiment_config` dictionary defines key parameters for tracking and logging the evaluation:

```
# Evaluate
    agent = agents.get_agent(retriever_config_path=retriever_config_path)
    experiment_config = {
        "model_id": settings.OPENAI_MODEL_ID,
        "retriever_config_path": retriever_config_path,
        "agent_config": {
            "max_steps": agent.max_steps,
            "agent_name": agent.agent_name,
        },
    }
```

Next, we define the scoring metrics used to evaluate the agent's performance. Opik provides built-in evaluation metrics, but we also include custom ones for deeper analysis.

```
scoring_metrics = [\
        Hallucination(),\
        AnswerRelevance(),\
        Moderation(),\
        SummaryDensityHeuristic(),\
        SummaryDensityJudge(),\
    ]
```

The scoring process evaluates the agent’s performance across multiple dimensions:

- **Hallucination**: Measures whether the agent generates false or misleading information.
- **Answer Relevance**: Scores the relevance of the agent's response to the given prompt.
- **Moderation**: Detects potentially inappropriate or unsafe content in responses.

> For **more details on the metrics above** or on how to build custom metrics, check out [Opik’s docs](https://www.comet.com/docs/opik/evaluation/metrics/overview?utm_source=paul_2nd_brain_course&utm_campaign=opik&utm_medium=lessons).

In addition to these built-in Opik metrics, we include two custom components. Both compute the response density (whether the answer is too long or too short) but with different techniques: heuristics or LLM-as-Judges. This is a good example of understanding the difference between the two.

- **SummaryDensityHeuristic**: Evaluates whether a response is too short, too long, or appropriately balanced.
- **SummaryDensityJudge**: Uses an external LLM to judge response density and conciseness.

Finally, we execute the evaluation process using the metrics defined and our evaluation dataset:

```
if dataset:
        evaluate(
            dataset=dataset,
            task=evaluation_task,
            scoring_metrics=scoring_metrics,
            experiment_config=experiment_config,
            task_threads=2,
        )
    else:
        logger.error("Can't run the evaluation as the dataset items are empty.")
```

This code ensures that evaluation runs only when a dataset is available. The `evaluate()` function runs the agent using the `evaluation_task()` method on the evaluation dataset and measures the defined scoring metrics. The [results are then logged in Opik](https://www.comet.com/opik?utm_source=paul_2nd_brain_course&utm_campaign=opik&utm_medium=lessons) for further analysis and comparison.

---

## 9\. Running the code

The best way to set up and run the code is through our **[GitHub repository](https://github.com/decodingml/second-brain-ai-assistant-course/tree/main)**, where we have documented everything you need. We will keep the end-to-end instructions only in our GitHub to avoid having the documentation scattered throughout too many places (which is a pain to maintain and use).

First, you have to ensure that your MongoDB Docker container is running and that your RAG collection is populated.

Next, you can run the agent through the command-line interface (CLI) for a quick test or with a Gradio UI for a more interactive experience.

To quickly test the Agentic RAG inference on a predefined query, you can run the following command from the CLI:

```
make run_agent_query RETRIEVER_CONFIG=configs/compute_rag_vector_index_openai_parent.yaml
```

> _**Note**: The retriever config can be any of the ones defined in Lesson 5, depending on the retrieval strategy you want to use (but they have to match, between the RAG feature pipeline and the inference pipeline)._

For a more interactive experience, you can launch the Gradio UI by executing:

```
make run_agent_app RETRIEVER_CONFIG=configs/compute_rag_vector_index_openai_parent.yaml

```

Additionally, if you want to evaluate the agent’s performance, run the evaluation pipeline using:

```
make evaluate_agent RETRIEVER_CONFIG=configs/compute_rag_vector_index_openai_parent.yaml
```

All the runs, including inference and evaluation, can be tracked directly from the [Opik dashboards](https://www.comet.com/opik?utm_source=paul_2nd_brain_course&utm_campaign=opik&utm_medium=lessons), providing insights into performance and enabling better monitoring of experiments.

For the whole setup and running guide, go to our [GitHub](https://github.com/decodingml/second-brain-ai-assistant-course/tree/main/apps/second-brain-online):

[GO TO GITHUB](https://github.com/decodingml/second-brain-ai-assistant-course/tree/main/apps/second-brain-online)

---

## Conclusion

This was a long lesson—if you're still here, you’ve made it to the end of the **Building Your Second Brain AI Assistant** course. Congrats!

Throughout this lesson, we explored **LLM-powered agents** and how they differ from traditional workflows.

We designed and implemented the Agentic RAG module, integrating **[SmolAgents](https://github.com/huggingface/smolagents)**, **[Gradio](https://www.gradio.app/)**, and **[MongoDB](https://www.mongodb.com/products/platform/atlas-vector-search?utm_campaign=ai-pilot&utm_medium=creator&utm_term=iusztin&utm_source=blog)** to enable dynamic retrieval and reasoning. We then built an observability pipeline using **[Opik](https://github.com/comet-ml/opik)**, ensuring full monitoring and evaluation of our agentic system.

Beyond implementation, we focused on evaluating and improving the agent's performance. We explored prompt monitoring, latency tracking, and response evaluation using built-in and custom metrics, including heuristic-based scoring and LLM-as-a-judge techniques.

With this final lesson, you now have a complete, end-to-end understanding of **architecting, building, and evaluating LLM-powered AI assistants**.

If you haven’t read all the lessons from the Second Brain AI Assistant open-source course, consider starting with **[Lesson 1](https://decodingml.substack.com/p/build-your-second-brain-ai-assistant)** on architecting the end-to-end LLM system.

> 💻 Explore all the lessons and the code in our freely available **[GitHub repository](https://github.com/decodingml/second-brain-ai-assistant-course).**

If you have questions or need clarification, **feel free to ask**. See you in the next session!

---

## References

Decodingml. (n.d.). _GitHub - decodingml/second-brain-ai-assistant-course_. GitHub. [https://github.com/decodingml/second-brain-ai-assistant-course](https://github.com/decodingml/second-brain-ai-assistant-course)

Iusztin P., Labonne M. (2024, October 22). _LLM Engineer’s Handbook \| Data \| Book_. Packt. [https://www.packtpub.com/en-us/product/llm-engineers-handbook-9781836200062](https://www.packtpub.com/en-us/product/llm-engineers-handbook-9781836200062)

Kuligin, L., Zaldívar, J., & Tschochohei, M. (n.d.). _Generative AI on Google Cloud with LangChain: Design scalable generative AI solutions with Python, LangChain, and Vertex AI on Google Cloud._ [https://www.amazon.com/Generative-Google-Cloud-LangChain-generative/dp/B0DKT8DCRT](https://www.amazon.com/Generative-Google-Cloud-LangChain-generative/dp/B0DKT8DCRT)

_Log traces_. (n.d.). [https://www.comet.com/docs/opik/tracing/log\_traces](https://www.comet.com/docs/opik/tracing/log_traces?utm_source=paul_2nd_brain_course&utm_campaign=opik&utm_medium=lessons)

Varshney, T. (2024, June 24). _Introduction to LLM Agents \| NVIDIA Technical Blog_. NVIDIA Technical Blog. [https://developer.nvidia.com/blog/introduction-to-llm-agents/](https://developer.nvidia.com/blog/introduction-to-llm-agents/)

What’s AI by Louis-François Bouchard. (2025, February 2). _Real Agents vs. Workflows: The Truth Behind AI “Agents”_ \[Video\]. YouTube. [https://www.youtube.com/watch?v=kQxr-uOxw2o](https://www.youtube.com/watch?v=kQxr-uOxw2o)

DecodingML. (n.d.). _The Ultimate Prompt Monitoring Pipeline._ Medium. [https://medium.com/decodingml/the-ultimate-prompt-monitoring-pipeline-886cbb75ae25](https://medium.com/decodingml/the-ultimate-prompt-monitoring-pipeline-886cbb75ae25)

DecodingML. (n.d.). _The Engineer’s Framework for LLM RAG Evaluation._ Medium. [https://medium.com/decodingml/the-engineers-framework-for-llm-rag-evaluation-59897381c326](https://medium.com/decodingml/the-engineers-framework-for-llm-rag-evaluation-59897381c326)

Cardenas, E., & Monigatti, L. (n.d.). _What is Agentic RAG?_ Weaviate Blog. [https://weaviate.io/blog/what-is-agentic-rag](https://weaviate.io/blog/what-is-agentic-rag)

---

### Original URL
https://decodingml.substack.com/p/llmops-for-production-agentic-rag
</details>

---
<details>
<summary>Building Production-Ready RAG Applications: Jerry Liu</summary>

This video transcript, presented by Jerry Liu, co-founder and CEO of LlamaIndex, focuses on **building production-ready Retrieval Augmented Generation (RAG) applications**. The presentation highlights the transformative impact of large language models (LLMs) in various use cases like knowledge search, QA, conversational agents, workflow automation, and document processing, leveraging LLMs' reasoning capabilities over diverse data.

### Paradigms for LLMs to Understand Data
The video outlines two primary paradigms for enabling language models to understand data they haven't been trained on:
*   **Retrieval Augmentation (RAG)**: This involves creating a data pipeline to insert context from a data source (e.g., vector database, unstructured text, SQL database) directly into the LLM's input prompt.
*   **Fine-tuning**: This method bakes knowledge directly into the network's weights by updating the model itself, or an adapter on top of it, through a training process over new data.
The presentation primarily focuses on **retrieval augmentation**.

### The Current RAG Stack and Its Challenges
For a typical QA system, the RAG stack consists of two main components:
*   **Data Ingestion**: Loading data from various sources.
*   **Data Querying**: This further breaks down into **retrieval** (fetching relevant information) and **synthesis** (generating a response using the retrieved context).
While LlamaIndex simplifies building this stack (around five lines of code), understanding the underlying components is encouraged for AI engineers.

However, developers deploying these applications in production often encounter significant **challenges with "naive RAG"**. The key issue is **poor response quality**.
Common problems include:
*   **Bad Retrieval Issues**:
    *   **Low Precision**: Not all retrieved chunks are relevant, leading to **hallucination** or "lost in the middle" problems where relevant information is buried in fluff.
    *   **Low Recall**: The system fails to retrieve all necessary information, meaning the "top K" retrieved items aren't sufficient to answer the question.
    *   **Outdated Information**.
*   **LLM-side Issues**: Even beyond retrieval, LLMs can exhibit issues like hallucination, irrelevance, toxicity, and bias.

### Strategies to Improve RAG Performance
To enhance RAG application performance, improvements can be made across the entire pipeline:
*   **Data Optimization**: Storing additional information beyond raw text chunks, optimizing the data pipeline, and experimenting with **chunk sizes**.
*   **Embedding Representation**: Optimizing the embedding model, as pre-trained models might not be optimal for specific data.
*   **Retrieval Algorithm**: Moving beyond simple "top K" similarity searches from a vector database.
*   **Synthesis**: Using LLMs for more sophisticated reasoning, like breaking down complex questions or routing to different data sources, rather than just pure generation.

Before implementing any of these techniques, it's crucial to be **task-specific and establish a way to measure performance through evaluation**.

### Evaluating RAG Systems
Evaluation is essential for defining a benchmark and iterating on system improvements. It involves evaluating both the **end-to-end solution** (input query to output response) and **specific components** like retrieval.

*   **Evaluation on Retrieval**:
    *   **Goal**: Ensure returned content answers the query, is relevant, and avoids fluff.
    *   **Data Set**: Requires an evaluation dataset of input queries and relevant document IDs. This can be human-labeled, derived from user feedback, or synthetically generated.
    *   **Metrics**: Ranking metrics like **success rate, hit rate, MRR (Mean Reciprocal Rank), and NDCG (Normalized Discounted Cumulative Gain)** are used, drawing from decades of Information Retrieval (IR) research.

*   **Evaluation of the Final Response (Synthesis)**:
    *   **Goal**: Evaluate the overall end-to-end quality of the generated response.
    *   **Data Set**: Similar to retrieval, this needs a dataset through human annotations, user feedback, or ground truth reference answers. It can also be synthetically generated using powerful models like GPT-4.
    *   **Method**: Run the full RAG pipeline and use **LLM-based evaluations** (label-free or with labels) to assess predicted outputs.

### Optimizing RAG Systems: Techniques
Once an evaluation benchmark is defined, various optimization techniques can be applied, starting with simpler methods.

**1. Table Stakes RAG Techniques (Basics):**
*   **Better Chunking**: Tuning **chunk size** can significantly impact performance. It's noted that **more retrieved tokens do not always lead to higher performance** due to "lost in the middle" problems, where information in the middle of the LLM's context window can be overlooked. Optimal chunk sizes are data-set dependent.
*   **Metadata Filtering**: This involves adding **structured context** (e.g., page number, document title, summary of adjacent chunks, or even hallucinated questions a chunk answers) to text chunks.
    *   **Benefit**: It allows combining **structured query capabilities** (like a SQL WHERE clause, e.g., filtering by `year = 2021`) with semantic search, significantly **improving the precision of retrieval results** by avoiding irrelevant matches.
*   Hybrid search.

**2. Advanced Retrieval Methods:**
*   Reranking.
*   LlamaIndex offers concepts like recursive retrieval and dealing with embedded tables.
*   **Small to Big Retrieval**:
    *   **Problem**: Embedding large text chunks can be suboptimal for retrieval because "fluff" in the chunk can bias the embedding representation.
    *   **Solution**: **Embed text at a smaller, more granular level** (e.g., sentence level) for retrieval. Then, during synthesis, **expand the context window** to provide the LLM with more information for a proper response.
    *   **Benefits**: Leads to **more precise retrieval** by making smaller chunks more likely to be retrieved. It also helps **avoid "lost in the middle" problems** and can allow for smaller `top K` values without sacrificing relevant context.
    *   A related idea is **embedding a reference to the parent chunk**, a summary, or questions the chunk answers, rather than just the raw text chunk itself, which also improves retrieval performance.

**3. More Expressive and Advanced Concepts (Potentially Harder/Costlier):**
*   **Agents**: These involve using LLMs for **reasoning beyond simple synthesis**, addressing limitations of basic RAG for complex, multi-step questions or comparative analyses.
    *   **Multi-Document Agents**: An architecture where each document is modeled as a set of tools (e.g., a tool to summarize the document, another to perform QA over specific facts).
    *   This paradigm involves retrieving relevant tools from potentially millions of documents and then acting upon them, blending embedding-based retrieval with agent tool use, which is a powerful new capability enabled by LLMs.
*   **Fine-tuning**: This optimizes specific parts of the RAG pipeline for better performance.
    *   **Fine-tuning Embeddings**: Improves the embedding representations for specific data, ensuring the retriever returns more relevant information given a user query. A method involves generating a **synthetic query dataset from raw text chunks using LLMs** to fine-tune the embedding model. Fine-tuning an adapter on top of the model has advantages, such as not requiring base model weights and avoiding re-indexing the entire document corpus if only the query is fine-tuned.
    *   **Fine-tuning LLMs**: Especially useful for weaker LLMs (e.g., GPT-3.5 Turbo, Llama 2 7B) that might struggle with response synthesis, reasoning, or structured outputs compared to larger models. The solution proposed is to **generate a synthetic dataset using a stronger model like GPT-4 and distill that knowledge into the weaker model** to improve its train of thought, response quality, and structured outputs.

All these advanced concepts and production RAG strategies are detailed in the LlamaIndex documentation .

### Original URL
https://www.youtube.com/watch?v=TRjq7t2Ms5I
</details>

