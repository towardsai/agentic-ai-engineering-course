# Research based on provided article guidelines

## Research Results

---

<details>
<summary>What are the key differences between LLM workflows and agentic systems according to leading AI research (e.g. Anthropic)?</summary>

### Source: https://www.anthropic.com/research/building-effective-agents
Anthropic distinguishes between LLM workflows and agentic systems by describing the building blocks and their progressive complexity. The foundational component is an "augmented LLM," which is a language model enhanced with capabilities like retrieval, tool use, and memory. These augmentations allow the LLM to actively generate search queries, select tools, and manage retained information.

A "workflow" in this context often refers to "prompt chaining," where a complex task is broken down into a sequence of steps, with each LLM call processing the output of the previous one. This approach is suitable for tasks that can be decomposed into fixed subtasks, and where accuracy can be improved by making each step more manageable. The process can include programmatic checks at intermediate steps to ensure correctness. Such workflows are ideal for predictable, decomposable tasks, trading off some latency for higher reliability and accuracy.

In contrast, "agentic systems" build upon these workflows and augmentations, moving towards more autonomous and adaptive behaviors. While the document primarily focuses on workflow mechanics, it implies that agentic systems involve more dynamic decision-making, where an LLM may autonomously determine steps, tools, and information needed to achieve complex objectives, rather than simply following a predefined sequence.

-----

### Source: https://www.anthropic.com/engineering/built-multi-agent-research-system
This source details the construction of multi-agent research systems and highlights a key distinction: multi-agent systems differ from single-agent or workflow-based systems primarily in their coordination complexity. As the number of agents increases, so does the complexity required for coordination and communication between agents. Early agentic systems were simpler, but as they evolve, their ability to collaborate and orchestrate complex tasks becomes a defining characteristic. This complexity is not typically present in straightforward LLM workflows, which remain linear and predictable.

-----

### Source: https://blog.langchain.dev/how-to-think-about-agent-frameworks/
According to Anthropic, there is a critical architectural distinction between workflows and agentic systems. Workflows do not require a high level of dynamism or complexity; they are preferable when tasks can be handled in a structured, predictable way. Agentic systems, on the other hand, are justified only when a task's complexity or unpredictability surpasses what can be achieved with simple workflows. Most agentic systems are actually combinations of workflows, but true agents are needed when decisions must be made dynamically at runtime. Anthropic and OpenAI both recommend using workflows whenever possible, reserving agents for tasks that demand runtime adaptability and reasoning.

-----

### Source: https://www.anthropic.com/research/agentic-misalignment
Anthropic defines workflows as systems where LLMs and tools are orchestrated through predefined code paths—each step and tool invocation is predetermined by the developer. This makes workflows predictable and intuitive.

By contrast, agents are systems where LLMs dynamically control their own process, making decisions about tool usage and planning steps independently. An agent interacts with the user to clarify needs, iteratively reasons about the problem, and determines the best approach without hardcoded paths. Agents are intended to replace human-like roles that make autonomous decisions, while workflows automate specific, well-defined tasks. Agents are more flexible and adaptive, but also significantly more complex to build and less reliably realized with current LLM capabilities.

</details>

---

<details>
<summary>When should AI engineers choose predefined LLM workflows versus agentic systems for AI solution design?</summary>

### Source: https://www.louisbouchard.ai/agents-vs-workflows/
Workflows are characterized by their predictable, linear progression that follows specific code lines and integrations. They are reliable and deterministic, making them ideal for use cases where consistency, repeatability, and control are paramount. Workflows are best suited for scenarios where the tasks and required steps are well-defined and do not require dynamic adaptation beyond the outputs of the LLM. In contrast, agentic systems introduce more autonomy, with agents capable of reasoning, adapting, and making decisions based on evolving data and context. Agentic systems are preferable in situations that demand flexibility, complex decision-making, and the ability to handle multi-step, dynamic processes where each step might depend on the outcomes of previous actions or changing conditions.

-----

### Source: https://www.arkeoai.com/blog/investors-guide-llm-vs-agentic-architecture
Predefined LLM workflows (or wrappers) are focused on surface-level, narrowly scoped tasks—such as a customer service chatbot that can answer FAQs—but lack the ability to handle complex, multi-faceted requests requiring coordination across departments or business functions. Agentic architectures, on the other hand, involve multiple autonomous agents that can learn, collaborate, and adapt to address end-to-end workflows, from sales to financial forecasting. Agentic systems can automate and optimize highly complex processes that require real-time decision-making and cross-functional collaboration, making them suitable for scenarios where tasks are interconnected and outcomes depend on continuous adaptation. In summary, choose predefined LLM workflows for simple, contained tasks and agentic systems for complex, adaptive, and collaborative AI solutions.

-----

### Source: https://www.youtube.com/watch?v=Qd6anWv0mv0
This source discusses the difference between workflows and agents in AI systems, emphasizing that workflows are best for well-defined, sequential tasks where the process and outcome are predictable. Workflows are typically used when the task can be decomposed into discrete steps with clear handoffs. Agentic systems, in contrast, are designed for scenarios where AI must make decisions autonomously, adapt to new information, and potentially interact with other agents or systems. Agentic systems are more appropriate when the solution requires a higher degree of autonomy, reasoning, or collaboration among multiple intelligent components.

-----

### Source: https://devops.com/why-you-shouldnt-forget-workflows-with-agentic-ai-systems/
Programmatic workflows are linear and structured, providing a guarantee that the AI system will not exceed its programmed boundaries. These are most appropriate for scenarios where strict control and predictability are necessary. As LLMs gain reasoning capabilities, agentic systems—where the LLM determines the steps to complete a task—offer more autonomy. Agentic architectures leverage “chain-of-thought” reasoning and orchestration frameworks to break down complex problems, consider multiple stages, and select tools dynamically based on the situation. Agentic systems should be chosen when the task requires multi-stage reasoning, dynamic tool usage, and autonomy to adapt to changing data or goals. Workflows are still valuable for ensuring reliability in narrowly scoped, well-understood processes, while agentic systems excel in open-ended, complex, or evolving environments.

-----

</details>

---

<details>
<summary>What are the main architectural patterns and persistent challenges in the design and deployment of agentic AI systems?</summary>

### Source: https://www.youtube.com/watch?v=MrD9tCNpOvU
The video introduces and explains four foundational design patterns in agentic AI systems:

- Reflection pattern: The agent evaluates its own outputs, identifies errors or suboptimal results, and iteratively refines its responses.
- Tool use pattern: The agent leverages external tools or resources—such as databases, APIs, or computational modules—to extend its capabilities beyond its internal knowledge or reasoning.
- Planning pattern: The agent decomposes complex tasks into structured plans or sequences of subtasks, allowing for more deliberate and strategic execution.
- Multi-agent collaboration pattern: Multiple agents interact and collaborate to solve problems that exceed the capacity of a single agent, enabling division of labor, specialization, and collective problem-solving.

These patterns enable agentic AI systems to tackle more complex, real-world challenges by supporting advanced reasoning, adaptability, and collaboration.

-----

### Source: https://www.ibm.com/architectures/patterns/agentic-ai
IBM describes agentic AI systems as those that autonomously plan and perform tasks on behalf of users or other systems. Key architectural elements include:

- Integration of LLMs with traditional programming models, allowing flexible and precise task execution.
- Decomposition of complex problems into smaller, manageable tasks, with agents using tools to interact with external systems or perform computations.
- Agents are built from core components (not detailed in the excerpt), with optional additions for operational management, monitoring, and security (such as identity management and data leakage prevention).

The persistent challenges highlighted include the need for operational agent management, robust performance monitoring, and strong security controls to ensure safe and effective deployment of autonomous systems.

-----

### Source: https://blog.dailydoseofds.com/p/5-agentic-ai-design-patterns
This source details five prominent architectural patterns for agentic AI:

- Reflection pattern: The agent reviews its outputs, detects mistakes, and iterates for improved results.
- Tool use pattern: The agent augments its capabilities by querying external databases, executing scripts, or invoking APIs, thus not limited to its own training data.
- ReAct (Reason and Act) pattern: This combines reflection and tool use, enabling agents to think about their outputs and act in the environment, making it a powerful and flexible pattern.
- Planning pattern: The agent strategically subdivides complex requests, outlines objectives, and creates roadmaps for execution.
- Multi-agent pattern: Multiple agents collaborate, allowing for division of labor and solving more sophisticated problems than a single agent could handle alone.

All these patterns are designed to enhance agentic behaviors such as self-improvement, strategic planning, and collaboration, which are critical for robust and scalable AI agent development.

-----

### Source: https://markovate.com/blog/agentic-ai-architecture/
Markovate provides a deep dive into agentic AI architecture, outlining a multi-layered approach:

- Input Layer: Aggregates diverse data sources for actionable insights.
- Agent Orchestration Layer: Coordinates specialized agents for task management, collaboration, and performance monitoring. Agents here support planning, self-evaluation, domain-specific tool use, and continuous learning.
- Data Storage & Retrieval Layer: Manages data via centralized/distributed repositories, vector stores, and knowledge graphs for efficient retrieval and reasoning.
- Output Layer: Delivers personalized, context-aware results and keeps the knowledge base current.
- Service Layer: Ensures delivery across platforms, providing intelligent recommendations while maintaining governance and compliance.

Persistent challenges and safeguards include:

- Ensuring safety, fairness, and regulatory compliance through built-in governance frameworks.
- Iterative validation for continuous improvement.
- Addressing bias, data leakage, and secure operations as agentic AI becomes more autonomous.

The architecture anticipates a shift toward an "Agent Economy," where organizations budget for AI agents as a core operational resource, highlighting the need for robust, compliant, and ethical system design.

-----

</details>

---

<details>
<summary>What are the operational mechanisms and capabilities of state-of-the-art agentic AI systems, such as Devin, Deep Research agents, and Codex-like models (as of 2025)?</summary>

### Source: https://cognition.ai/blog/introducing-devin
Devin is described as the first AI software engineer designed to autonomously plan and execute complex engineering tasks that require thousands of discrete decisions. The system leverages advances in long-term reasoning and planning to break down and solve intricate software problems. Devin can recall relevant context over extended timescales, allowing it to manage multi-step workflows such as debugging, adding features, or refactoring large codebases. The agent interacts with a full development environment, operating tools and writing code in a way similar to a human engineer. Devin’s operational mechanism includes an ability to receive a high-level goal, decompose it into actionable steps, and iteratively refine solutions, including seeking clarifications or additional information when needed. Its capabilities encompass end-to-end software engineering tasks, including repository navigation, code generation, testing, and deployment.

-----

### Source: https://en.wikipedia.org/wiki/Devin_AI
In early 2025, Devin gained a machine-generated software documentation feature called Devin Wiki. This addition allows the system to automatically generate and maintain up-to-date documentation for codebases, enhancing code transparency and maintainability. The platform also introduced an interactive search and answer engine, enabling users to query the codebase and documentation to receive immediate, contextually relevant answers. These features expand Devin’s agentic capabilities beyond code execution to include robust documentation and knowledge management, making it a comprehensive tool for software development lifecycle support.

-----

### Source: https://devtrain.co/can-devin-ai-really-replace-software-engineers-in-2025/
Devin AI is designed to support, not replace, software engineers by handling standardized, technical, and repetitive programming tasks more efficiently than humans. Its operational mechanisms include tracking performance data to identify software bottlenecks and suggesting optimizations for improved speed and resource efficiency. Devin evaluates user requirements, forecasts development obstacles, and proposes solutions, allowing developers to focus on creative and novel problem-solving. The system automates code-writing for both basic and complex algorithms, detects and resolves errors using advanced monitoring, and integrates with platforms like GitHub and project management tools to enhance team workflows. Devin also reviews collaborative work, proposes improvements, and ensures adherence to project guidelines, thereby streamlining development processes and improving team output.

-----

### Source: https://devin.ai
Devin functions as an AI coding agent and software engineer, supporting development teams in building software more rapidly and effectively. It operates as a parallel cloud agent, enabling serious engineering teams to leverage scalable AI assistance for software projects. The system is positioned as a productivity tool, augmenting developers’ capabilities throughout the software development process, from code generation to project management.

-----

### Source: https://devops.com/devins-latest-update-brings-major-enterprise-features-and-developer-improvements/
The January 2025 update to Devin introduced major enterprise features and improvements for developer productivity. Notably, Devin improved repository management by more accurately identifying relevant files and recognizing existing code patterns, which is vital for teams dealing with complex codebases. Enterprise accounts now allow centralized administration of access controls and billing. The agent supports flexible, pay-as-you-go billing for compute resources, removing prior limitations on monthly usage. Developer experience enhancements include streamlined authenticated testing via persistent browser session cookies, voice command integration through Slack, and a “Large Performant” Docker option to address persistent storage and performance challenges. These operational upgrades make Devin more practical for day-to-day software engineering and enterprise-scale deployments.

</details>

---

<details>
<summary>How is orchestration implemented differently in LLM workflow-based systems and agentic systems, and why is it necessary for both?</summary>

### Source: https://orq.ai/blog/llm-orchestration
Implementing orchestration in LLM workflow-based systems involves carefully designing workflows where different LLMs or components collaborate by handling specific, pre-assigned tasks. The orchestration process starts with mapping out the flow of tasks and assigning responsibilities to each LLM according to their strengths—such as one model focusing on natural language generation and another on data retrieval or sentiment analysis. 

Key aspects include managing how data flows between these models, often facilitated by integrating APIs or custom logic, to guarantee smooth model interactions. Additionally, external data sources can be integrated into the workflow, enriching outputs and enabling dynamic, contextually relevant responses. The orchestration framework must ensure accurate and timely data retrieval from these sources, emphasizing the importance of data flow management. 

Orchestration is necessary in workflow-based systems to streamline processes, ensure harmonious operation among multiple models, and deliver efficient, context-aware outputs.

-----

### Source: https://labelyourdata.com/articles/llm-orchestration
LLM orchestration in workflow-based systems is structured around breaking down an overall task into smaller, manageable components. Each subtask is assigned to a model based on its specific strength: for example, using a retrieval-focused model for information fetching and a reasoning-capable model for complex interpretations. The workflow is designed as a sequence, often using prompt chaining to transfer context between models and integrating APIs for interaction with external tools.

The orchestration framework is responsible for handling inputs, assigning tasks, collecting outputs, and monitoring performance in real-time, with capabilities to make adjustments—such as prompt fine-tuning or model replacement—based on observed efficacy. 

This orchestration is essential for optimizing the efficiency and adaptability of LLM-driven applications, ensuring that the right models are used for the right tasks and that system performance can be monitored and improved over time.

-----

### Source: https://openreview.net/forum?id=3Hy00Wvabi
WorkflowLLM, as described in this source, is a framework designed specifically to enhance the workflow orchestration capabilities of large language models. It enables complex task orchestration by allowing LLMs to manage not only the execution of individual workflow steps but also the broader coordination and sequencing of these steps. 

The necessity of orchestration in such systems arises because it allows the LLM to generalize across unseen workflows and adaptively manage dependencies and execution orders among tasks. In agentic systems, orchestration is even more dynamic, as agents must autonomously plan, adapt, and manage workflows in response to changing requirements and real-time feedback. 

Thus, orchestration is crucial in both workflow-based and agentic systems but is implemented differently: workflow-based systems focus on predefined, structured flows, while agentic systems require adaptive, self-directed coordination to handle evolving or unforeseen tasks.

-----

</details>

---

<details>
<summary>What are the differences in reliability and error recovery between predefined LLM workflows and agentic systems in production environments?</summary>

### Source: https://devops.com/why-you-shouldnt-forget-workflows-with-agentic-ai-systems/
Predefined (structured) LLM workflows operate with a linear, stepwise progression, ensuring that the system only performs actions explicitly programmed by humans. This approach is reliable, as the workflow cannot deviate from its set path, minimizing the risk of unpredictable failures. Error recovery in such systems is straightforward, as each step's outcome is predetermined and known, allowing for direct handling of common failure points.

Agentic systems, by contrast, use LLMs' reasoning abilities to autonomously determine the steps needed to complete a task. This chain-of-thought reasoning lets agentic systems break down problems dynamically, potentially leading to more reliable and accurate outputs for complex or ambiguous tasks. However, this autonomy introduces challenges in reliability, as the agent can make decisions not anticipated by the original developers.

To address error recovery and reliability, agentic architectures often use orchestration frameworks (like LangGraph), which explicitly structure the reasoning process and may include validation stages where the agent can reconsider or revise its own output. These orchestration frameworks also manage tool integration, allowing agents to fetch external data or use specialized utilities as needed, thus increasing robustness in dynamic environments. Nonetheless, the increased flexibility and autonomy of agentic systems can introduce new failure modes that require more sophisticated error detection and recovery mechanisms than static workflows.

-----

### Source: https://blog.langchain.dev/how-to-think-about-agent-frameworks/
The primary reliability challenge for agentic systems is ensuring that the LLM has the appropriate context at each step of its autonomous process. Reliable agentic systems depend on maintaining and passing the right information throughout the workflow. If context is lost or mismanaged, the agent's output can become unreliable, leading to cascading errors that are harder to diagnose and recover from compared to static workflows.

Error recovery in agentic frameworks thus often requires mechanisms to reconstruct or re-inject the correct context at each decision point. This contrasts with predefined workflows, where error recovery typically involves rolling back to a known good step or retrying a failed operation using well-understood logic.

-----

### Source: https://www.arkeoai.com/blog/investors-guide-llm-vs-agentic-architecture
Predefined LLM wrappers (workflows) in production are limited to specific, well-understood use cases—such as answering FAQs or following a fixed script. These systems are reliable in their scope but cannot handle complex tasks that require cross-functional coordination or adaptation to novel situations. Their error recovery processes are likewise limited to predefined fallback procedures.

Agentic architectures, on the other hand, enable autonomous execution across multiple steps and can coordinate complex, multi-agent workflows with minimal human intervention. This can lead to more consistent and high-quality outcomes in complex environments, as agents can adapt and recover from certain failures on the fly. However, this autonomy also means that new, unforeseen errors can occur, and the system's reliability is highly dependent on the quality of its reasoning and orchestration.

-----

### Source: https://blog.gopenai.com/agentic-workflows-vs-autonomous-ai-agents-do-you-know-the-difference-c21c9bfb20ac
Agentic workflows (predefined) are more transparent and explainable, as their sequence of actions is fixed and traceable. This makes reliability assessment and error recovery straightforward: errors can be mapped to specific workflow steps and addressed with targeted fixes or rollbacks.

Fully autonomous AI agents, by contrast, are harder to interpret, especially in complex scenarios. Their dynamic adjustment of strategy and tool usage makes it challenging to pinpoint the source of errors or to implement robust recovery routines. As a result, compounding errors are more likely in agentic systems, and their development, testing, and maintenance are costlier and require ongoing model monitoring and training.

Latency is another consideration: static workflows can be highly optimized and thus more performant, whereas agentic systems may experience higher latency due to the additional reasoning and adaptability steps, further complicating real-time error recovery.

-----

</details>

---

<details>
<summary>What are the most recent (2024-2025) real-world applications or deployments of agentic systems beyond Devin and Codex?</summary>

### Source: https://convergetp.com/2025/05/06/top-10-agentic-ai-examples-and-use-cases/
This source details several real-world deployments of agentic AI systems across major industries as of 2024-2025:

- **Healthcare**: Agentic AI is utilized for advanced medical image analysis and disease diagnosis, reporting improvements in diagnostic accuracy and reductions in false positives. IBM, in particular, is highlighted for developing such systems.
- **Manufacturing**: Autonomous robots powered by agentic AI handle tasks like quality inspection and assembly-line optimization, reducing error rates and improving efficiency.
- **Finance**: Agentic AI is used for algorithmic trading, fraud detection, and risk management, allowing for real-time, adaptive decision-making in dynamic markets.
- **Retail and Logistics**: Amazon leverages agentic AI in warehouse robotics to autonomously navigate, fulfill orders, and optimize inventory management.
- **Autonomous Vehicles**: Companies like Google are developing agentic AI-driven vehicles capable of independently navigating complex environments, moving towards fully autonomous transportation.
- The market growth for agentic AI is significant, with expectations to reach $15.7 billion by 2025, indicating widespread adoption and ongoing real-world deployment.

These examples demonstrate that agentic systems are now integral to operations in healthcare, manufacturing, logistics, and finance, beyond well-publicized tools like Devin and Codex.

-----

### Source: https://www.ibm.com/think/insights/ai-agents-2025-expectations-vs-reality
According to IBM, 2025 will see agentic AI agents moving from theoretical promise to practical utility in several domains:

- **Enterprise Automation**: Agentic AI is deployed in workflow orchestration, automatically coordinating complex business processes across multiple systems with minimal human input.
- **IT Operations**: AI agents can autonomously monitor, diagnose, and resolve issues in IT infrastructure, reducing downtime and operational costs.
- **Customer Service**: Advanced agentic AI chatbots handle end-to-end service requests, from troubleshooting technical issues to processing orders and handling escalations without human handoff.
- **Supply Chain Management**: AI-driven agents optimize logistics, predict demand, reorder supplies, and dynamically reroute shipments in response to real-time disruptions.
- IBM emphasizes that the adoption of agentic AI is accelerating, with enterprises shifting towards these systems to gain a competitive edge in automation, customer satisfaction, and operational resilience.

-----

### Source: https://superagi.com/top-10-agentic-ai-tools-to-watch-in-2024-a-comparison-of-features-and-capabilities/
This source outlines prominent agentic AI deployments and applications beyond coding assistants:

- **Autonomous Vehicles**: Google’s use of agentic AI for self-driving cars enables navigation in unpredictable, real-world scenarios.
- **Warehouse Automation**: Amazon’s robots navigate, sort, and fulfill orders autonomously, streamlining supply chain operations.
- **Healthcare Diagnostics**: IBM’s agentic AI systems analyze medical images and support clinical decision-making, improving patient outcomes and workflow efficiency.
- **Personal Productivity**: Agentic AI tools are emerging to handle calendar management, email triage, and task prioritization for individuals and teams.
- **Smart Infrastructure**: Buildings and factories equipped with agentic AI dynamically adjust power usage, schedule maintenance, and control climate systems for optimal efficiency.

The source underlines that agentic AI is not limited to software development but is actively transforming healthcare, logistics, transportation, and personal productivity.

-----

### Source: https://www.securityjourney.com/post/no-country-for-no-code-are-we-heading-towards-a-wild-west-of-software-security-0-0
This source discusses the application of agentic AI in cybersecurity as of 2025:

- **Autonomous Threat Detection**: Agentic AI agents continuously monitor network traffic, identify anomalies, and respond to threats in real time without human intervention.
- **Adaptive Security Operations**: Security teams deploy agentic AI to automate vulnerability scanning, patch management, and incident response, improving both speed and accuracy.
- **Risk Quantification**: AI agents dynamically assess and quantify risks posed by other autonomous systems, adjusting security policies on the fly.
- These deployments enable security teams to proactively defend against increasingly sophisticated cyber threats, relying on agentic systems for 24/7, self-directed protection.

-----

### Source: https://www.signitysolutions.com/blog/what-is-agentic-ai
This source details the following real-world agentic AI applications for 2024-2025:

- **Financial Services**: Agentic AI powers trading systems that autonomously analyze markets, identify opportunities, and execute trades while managing risks.
- **Healthcare**: Smart medical devices use agentic AI to monitor patients, adjust drug delivery, and personalize treatments in real time.
- **Transportation and Logistics**: Agentic AI enables autonomous vehicles and advanced navigation systems to operate safely and optimize routes with minimal human input.
- **Supply Chain Management**: Agentic AI systems automate inventory control, supplier order placement, and production scheduling to maintain optimal stock levels.

The source highlights that agentic AI’s self-directed decision-making is now foundational in finance, healthcare, logistics, and beyond, with deployments focused on efficiency, adaptability, and real-time optimization.

-----

</details>

---

<details>
<summary>How do state-of-the-art agentic systems (e.g., OpenAI Operator, Deep Research agents) implement and manage short-term and long-term memory?</summary>

### Source: https://openai.com/index/introducing-operator/
OpenAI Operator is designed to handle a wide variety of repetitive browser tasks, such as filling out forms and ordering groceries. While the official introduction does not provide detailed architectural specifics about how short-term and long-term memory are managed, Operator is described as a system capable of maintaining context over extended workflows. This implies the presence of persistent memory mechanisms to track user instructions, preferences, and task progress across sessions. Such persistent memory likely enables Operator to recall prior actions and user-specific data, crucial for efficient automation and contextual continuity in agentic systems.

-----

### Source: https://techcommunity.microsoft.com/blog/azure-ai-services-blog/memory-management-for-ai-agents/4406359
This source describes a detailed, modular approach to memory management for AI agents using a system called Mem0, which integrates with Azure AI services. The architecture involves:

- An embedder to generate vector representations (embeddings) for pieces of memory.
- A vector store (such as Azure AI Search) to store and retrieve these embeddings efficiently.
- A large language model (LLM) for understanding, summarizing, and reasoning over both new and previously stored memory.

The configuration allows the agent to encode both short-term and long-term memory as embeddings, making retrieval based on semantic similarity possible. Short-term memory can be stored briefly for immediate tasks, while long-term memory persists in the vector store for later retrieval. The system supports dynamic memory initialization and search, allowing the agent to access relevant historical context as needed for task continuity and user personalization.

-----

### Source: https://www.graphlit.com/blog/survey-of-ai-agent-memory-frameworks
This survey offers an overview of various frameworks and platforms for AI agent memory management. While it does not detail a single system’s implementation, it highlights common patterns:

- Multi-tier memory architectures are typical, with short-term (working) memory for immediate context and long-term memory for persistent knowledge.
- Embedding-based vector stores are widely used to enable fast, semantically rich retrieval of relevant memories.
- Some systems utilize explicit memory segmentation, with modules for episodic memory (concrete events), semantic memory (facts), and procedural memory (skills or instructions).
- Retrieval-augmented generation and memory search are standard, ensuring agents can access relevant past experiences when responding to complex queries or tasks.
- Memory frameworks may support user-controlled or encrypted storage, especially for privacy-sensitive applications.

-----

</details>

---

<details>
<summary>What common hybrid approaches combine LLM workflow and agentic patterns in commercial AI systems, and what are their benefits and drawbacks?</summary>

### Source: https://www.alvarezandmarsal.com/insights/ready-ai-automation-use-large-language-model-agentic-workflow-power-your-business
Agentic workflows powered by large language models (LLMs) are increasingly central in commercial AI automation. These workflows represent a shift from traditional automation by enabling self-sufficient, intelligent systems that learn from interactions and make decisions independently. In an agentic workflow, an LLM acts as an intermediary, executing tasks or providing assistance to users, thereby improving productivity, efficiency, and user experience.

There are four primary design patterns for agentic workflows:
- **Reflection:** The LLM evaluates and improves its outputs autonomously.
- **Tool Use:** The LLM uses external tools or services to extend its abilities.
- **Planning:** The LLM breaks down tasks into manageable sub-tasks and sequences actions.
- **Multi-agent Collaboration:** Multiple LLM agents work together, each with specific roles, to achieve complex objectives.

**Benefits:**  
- Enhanced autonomy and intelligence in workflows  
- Reduced operational costs and inaccuracies  
- Improved customer experience

**Drawbacks:**  
- Increased complexity in system design and maintenance  
- Potential challenges in oversight and control due to greater autonomy  
- Requires robust monitoring to ensure reliability and prevent errors

Businesses can implement these agentic workflows to revolutionize operations and maintain competitiveness in the evolving AI landscape.

-----

### Source: https://www.vonage.com/resources/articles/agentic-workflows/
Vonage describes agentic workflows as AI-powered processes where LLMs act as the intelligent backbone. The main patterns seen in commercial applications are:
- **LLM as Orchestrator:** The LLM manages the workflow, delegating tasks to specialized agents or tools.
- **LLM Tool Use:** The LLM directly interfaces with APIs, databases, or software tools to complete tasks.
- **Collaborative Agents:** Multiple LLM agents, sometimes with different skill sets, interact within a workflow to tackle complex challenges.

**Benefits:**  
- Reduces manual intervention  
- Increases efficiency and accuracy in business processes  
- Enables rapid scaling and adaptation to new tasks

**Drawbacks:**  
- Complexity in integrating multiple agents and tools  
- Potential for cascading errors if workflow logic is flawed  
- Higher demands on system monitoring and governance

These approaches are particularly effective in customer service automation, business intelligence, and process optimization.

-----

### Source: https://www.youtube.com/watch?v=MrD9tCNpOvU
The video identifies four foundational agentic AI design patterns in commercial systems:
- **Reflection:** AI self-assesses and iteratively improves its output, boosting reliability and creativity.
- **Tool Use:** The AI calls external resources or tools, expanding its functional scope.
- **Planning:** The AI structures and sequences steps for multi-stage tasks or projects.
- **Multi-agent Collaboration:** Separate AI agents, each with defined roles, work together to achieve a unified goal.

**Benefits:**  
- Improved solution quality via feedback and revision (reflection)  
- Broader task coverage by leveraging external tools  
- Capability to address complex, multi-step real-world problems  
- Distributed problem-solving through multi-agent collaboration

**Drawbacks:**  
- More intricate system engineering and monitoring  
- Increased risk of error propagation across agent interactions  
- Potential difficulties in debugging multi-agent workflows

These patterns underpin the most advanced commercial AI systems, supporting greater autonomy and adaptability.

-----

### Source: https://research.aimultiple.com/agentic-ai-design-patterns/
This source provides concrete examples of agentic AI design patterns:
- **Planning Pattern:** LLMs break down large tasks into sub-tasks and organize them logically, sometimes executing in parallel. Example: HuggingGPT, where LLMs connect with other AI models to solve complex problems.
- **Multi-Agent Pattern:** Multiple agents, often prompted via LLMs, each handle distinct responsibilities and communicate using protocols such as Google’s A2A. Examples include frameworks like AutoGen, LangChain, ChatDev, and OpenAI Swarm.

**Benefits:**  
- Enables LLMs to tackle tasks too complex for a single model  
- Supports modular, scalable workflow design  
- Facilitates collaboration and specialization among agents

**Drawbacks:**  
- Requires careful orchestration and communication protocols  
- More difficult to track and debug interactions  
- Risk of inconsistent outputs if agents are not well coordinated

These hybrid patterns are widely used in commercial AI platforms to enhance autonomy, flexibility, and problem-solving capacity.

-----

</details>

---

<details>
<summary>What are the latest (2025) methods for evaluating open-ended agent performance, especially for agents working in dynamic or unstructured tasks?</summary>

### Source: https://enthu.ai/blog/evaluate-agent-performance/
The article highlights the main metrics and methodologies for evaluating agent performance in 2025. Key evaluation metrics include:

- **First Contact Resolution**: Measures the percentage of issues resolved in the agent's first interaction with the user, indicating effectiveness and efficiency.
- **Average Handling Time (AHT)**: Tracks the average time agents take to resolve tasks or respond, crucial for dynamic or high-volume environments.
- **Quality Score**: Assesses the overall quality of the agent’s responses, often through a combination of automated scoring and human review, especially for open-ended or complex outputs.
- **Customer Satisfaction (CSAT)**: Direct feedback from users provides insight into the agent’s ability to meet expectations in unstructured and dynamic scenarios.
- **Compliance and Consistency**: Ensures that agents adhere to guidelines and maintain consistent performance across varying tasks and contexts.

The article also emphasizes the importance of **real-time analytics** and **continuous feedback loops**, where agent performance is monitored and adjusted dynamically as new challenges and unstructured tasks arise. Stress testing and scenario-based evaluations are used to observe agent behavior under unpredictable or evolving conditions.

-----

### Source: https://orq.ai/blog/agent-evaluation
This source describes a structured evaluation process that breaks agent performance down into several dimensions, particularly relevant for agents in dynamic environments:

- **Performance Metrics**: Accuracy is measured through exact match, in-order match, or any-order match methods, depending on task type. Response time and resource utilization are also key metrics.
- **Final Response Evaluation**: For open-ended tasks, responses can be compared against expected outputs using code-based evaluators or heuristic scoring, often within structured experiments.
- **Robustness Testing**: This involves exposing agents to ambiguous inputs, edge cases, malformed prompts, or adversarial examples. The goal is to ensure agents maintain reliability when faced with unpredictable or unstructured scenarios.
- **Trajectory Evaluation**: Tracks the sequence of decisions made by the agent and analyzes if each step aligns with expected or optimal paths. This is especially important in dynamic or multi-step environments.

These methods collectively help teams evaluate not just the correctness of the final answer, but also the agent’s reasoning and adaptability throughout the task.

-----

### Source: https://metadesignsolutions.com/benchmarking-ai-agents-in-2025-top-tools-metrics-performance-testing-strategies/
This source outlines the top tools and methodologies for benchmarking AI agents in 2025, particularly in dynamic and unstructured tasks:

**Top Evaluation Tools:**
- **AgentBench**: Evaluates language agents on decision-making, reasoning, and tool use.
- **REALM-Bench**: Focuses on agents that handle real-world reasoning and planning, especially in autonomous, dynamic environments.
- **ToolFuzz**: Stress-tests LLMs' integration with third-party tools, used for agents leveraging external APIs or tools.
- **Mosaic AI Evaluation Suite**: Supports custom benchmarking pipelines, real-time dashboards, and comparative scoring for comprehensive performance monitoring.
- **AutoGen Studio**: Enables dynamic simulation and evaluation of multi-agent conversations.

**Performance Testing Methodologies:**
- **Unit Testing**: Evaluates specific agent components (tool calls, reasoning steps).
- **Integration Testing**: Assesses interoperability between AI modules, APIs, and memory stores.
- **System Testing**: Examines the agent’s ability to follow workflows, handle load, and maintain context.
- **User Acceptance Testing (UAT)**: Validates agent performance in real-world, often unstructured scenarios.

Combining these tools and methodologies enables evaluators to assess agents rigorously in open-ended, dynamic contexts.

-----

### Source: https://galileo.ai/blog/ai-agent-evaluation
The article guides readers through the evaluation process for AI agents, focusing on reliability and adaptability in unstructured environments. Key points include:

- **Scenario-Based Testing**: Evaluates agent performance by placing agents in diverse, real-world scenarios, observing adaptability and decision-making under unpredictable conditions.
- **Continuous Monitoring**: Uses dashboards and analytics to track agent behavior and identify performance degradation or improvement trends over time.
- **Human-in-the-Loop (HITL) Review**: Involves human reviewers to assess the quality of open-ended responses, particularly for tasks where automated scoring is insufficient.
- **Behavioral Analysis**: Tracks not only the outcome but also the process—how agents reason, adapt, and recover from failures during dynamic tasks.

The source emphasizes the integration of automated and human evaluation methods to ensure agents are robust, adaptable, and reliable in complex, open-ended environments.

-----

### Source: https://www.alvarezandmarsal.com/thought-leadership/demystifying-ai-agents-in-2025-separating-hype-from-reality-and-navigating-market-outlook
According to this source, the market for AI agent evaluation is maturing, splitting into two primary categories: agent frameworks and agent providers. Evaluation for open-ended, dynamic tasks is characterized by:

- **Framework-Driven Evaluation**: Leading agent frameworks are building in advanced evaluation modules that enable continuous assessment as agents operate in real-time, dynamic conditions.
- **Provider Standards**: Major agent providers are developing standardized benchmarks and scenario libraries for assessing agent performance on unstructured tasks.
- **Cross-Scenario Benchmarking**: Agents are increasingly tested across a range of scenarios, with performance compared not only on accuracy but on adaptability, resilience, and consistency.

The source notes a trend toward **evaluation-driven development**, where feedback from comprehensive evaluation is used to iteratively improve agents, particularly in challenging, unpredictable environments.

-----

</details>

---

<details>
<summary>What are the most current best practices for orchestrating agentic AI systems to ensure reliability and scalability in production environments (2025)?</summary>

### Source: https://onereach.ai/blog/agentic-ai-orchestration-automating-complex-workflows-in-2025/
The source highlights several best practices for orchestrating agentic AI systems in 2025 to ensure reliability and scalability:

- **Modular Architecture**: Building agentic AI solutions using modular components allows for flexibility, easier updates, and scalability. Modular design supports the integration of new capabilities without overhauling the entire system.
- **Centralized Orchestration Layer**: A dedicated orchestration layer manages and coordinates the interactions between various AI agents, ensuring that workflows remain coherent and that agents operate in concert rather than in isolation.
- **Workflow Automation**: Automating end-to-end workflows with agentic AI increases operational efficiency and reduces human intervention in routine processes, but this should be combined with clear escalation paths for exceptions or failures.
- **Continuous Learning and Feedback Loops**: Systems are designed to learn from both successes and failures. Feedback mechanisms—such as monitoring agent decisions and outcomes—enable continuous improvement of agent behaviors.
- **Robust Monitoring and Reporting**: Real-time monitoring tools help track agent actions, system performance, and potential bottlenecks, thus allowing for prompt intervention if issues arise and supporting scalability as operational loads increase.
- **Security and Compliance**: Ensuring strict data privacy, secure communication between agents, and compliance with regulatory frameworks is a core element of reliable production deployments.
- **Human-in-the-Loop Oversight**: Even with advanced automation, human oversight is maintained, particularly for critical decisions or ethically sensitive actions, to uphold reliability and accountability.

-----

### Source: https://sendbird.com/blog/ai-orchestration
This guide outlines foundational best practices for orchestrating agentic AI systems for reliability and scalability in 2025:

- **Scalable Infrastructure**: Leveraging scalable cloud-native infrastructure enables organizations to dynamically allocate resources based on workload demands, ensuring consistent performance as usage grows.
- **API-First Design**: Structuring AI orchestration around APIs allows seamless integration of AI agents with existing systems and third-party tools, making it easier to expand and adapt capabilities over time.
- **Standardized Communication Protocols**: Using standardized messaging and communication protocols ensures that agents can interact reliably and exchange data without compatibility issues, supporting both reliability and future scalability.
- **Automated Testing and Validation**: Implementing automated pipelines for testing, validation, and deployment of AI agents reduces the risk of errors and enables rapid iteration as requirements evolve.
- **Observability and Incident Response**: Setting up observability tools—such as logging, tracing, and alerting—enables teams to detect operational issues early and respond quickly, supporting both reliability and scalability.
- **Role-Based Access Controls**: Enforcing role-based permissions and audit trails ensures only authorized users can modify or direct agentic workflows, which is vital for maintaining system integrity at scale.

-----

### Source: https://www.dbta.com/Editorial/News-Flashes/Preparing-for-the-Agentic-AI-Wave-Key-Frameworks-and-Best-Practices-169186.aspx
This article emphasizes the importance of process orchestration and hybrid approaches:

- **Process Orchestration**: Agentic process orchestration integrates individual AI automations and point solutions into end-to-end, coordinated workflows. This ensures coordination, accountability, and reliability for business-critical operations.
- **Hybrid Orchestration Approach**: Combining deterministic (rule-based) and non-deterministic (learning-based) orchestration allows systems to adapt to dynamic conditions while maintaining guardrails for reliability.
- **Human-in-the-Loop**: A human-in-the-loop practice remains essential for handling exceptions, validating actions, and ensuring ethical alignment, making the system more robust and trustworthy in real-world scenarios.
- **Continuous Adaptation**: The orchestration framework is designed to support continuous innovation and adaptation, allowing organizations to rapidly integrate new AI capabilities as technologies evolve.
- **Accountability Mechanisms**: By elevating orchestration to the process level, organizations can maintain clear accountability for outcomes, which is necessary for operating at scale and ensuring reliability.

-----

### Source: https://www.multimodal.dev/post/agentic-ai-systems
This source details actionable best practices for orchestrating agentic AI systems:

- **Structured Coordination System**: Implementing a central control system enables seamless interaction and information sharing among AI agents, maximizing their collective impact.
- **Real-Time Communication**: Allowing agents to communicate in real time optimizes outcomes and supports unified decision-making.
- **Data Flow Integration**: Ensuring data flows freely between agents enables more holistic and effective decision-making processes.
- **Human-in-the-Loop Checkpoints**: Establishing checkpoints for human intervention maintains accountability and allows human experts to handle exceptions or validate critical actions, supporting reliability.
- **Continuous Monitoring and Optimization**: Using dashboards and analytics to monitor KPIs, analyze agent interactions, and retrain models based on new data ensures the system remains reliable and scalable over time.
- **Bias Monitoring**: Regularly reviewing AI-driven decisions for bias and inaccuracies is critical for maintaining trustworthy and reliable outcomes at scale.

-----

### Source: https://www.fullstack.com/labs/resources/blog/is-your-business-ready-for-agentic-ai-a-practical-guide
This practical guide emphasizes the following best practices for deploying agentic AI in production:

- **Specialized Expertise**: Successful deployment requires teams with expertise in AI development, prompt engineering, and system orchestration, ensuring that systems are designed for both reliability and scalability from the outset.
- **Change Management**: Preparing the organization for agentic AI involves process redesign, training, and clear communication, which are vital for ensuring reliability during scaling and adoption.
- **Iterative Deployment**: Rolling out agentic AI incrementally—starting with pilot projects and gradually scaling—allows for continuous validation and risk mitigation.
- **Governance Structures**: Establishing strong governance around agentic AI use, including policies, procedures, and oversight, helps ensure systems remain reliable and compliant as they scale.
- **Ongoing Training**: Regularly updating teams’ skills and knowledge ensures they are prepared to maintain and optimize agentic AI systems as requirements and technologies evolve.

-----

</details>

---

<details>
<summary>Are there detailed technical case studies describing the integration and impact of agentic AI systems in customer service or business process automation since 2024?</summary>

### Source: https://convergetp.com/2025/05/06/top-10-agentic-ai-examples-and-use-cases/
This source discusses the transformative impact of agentic AI in customer service and business process automation. According to Gartner’s predictions, agentic AI is expected to resolve 80% of common customer service issues without human intervention by 2029. These systems differ from traditional chatbots, which rely on pre-programmed scripts, by learning from context, adapting to unique customer needs, and autonomously implementing solutions.

A detailed scenario describes a customer inquiring about a delayed shipment: whereas a traditional chatbot might only provide tracking information or escalate to a human, an agentic AI system can access and analyze live shipping data, determine the cause of the delay, offer solutions such as expedited replacements or partial refunds, and independently update records to carry out the chosen resolution. This demonstrates end-to-end automation with real-time decision-making and action.

The source also highlights agentic AI’s application in IT operations and software development, where AI agents now assist with code generation, provide real-time coding suggestions, and automate software testing. This integration accelerates development cycles, reduces errors, and allows human experts to focus on more complex tasks.

-----

### Source: https://www.creolestudios.com/real-world-ai-agent-case-studies/
This source elaborates on how agentic AI transforms customer experience (CX) by moving beyond simple automation to proactive, context-aware, and adaptive engagement. Agentic AI systems possess decision-making autonomy, allowing them to anticipate customer needs and act independently to improve satisfaction and loyalty.

The article identifies several challenges in modern CX—such as high customer expectations for instant, personalized, and seamless service, and the problem of fragmented customer journeys across channels. Agentic AI addresses these by maintaining context across platforms, thus ensuring continuity and personalization in every interaction.

A key technical impact is the ability to unify siloed systems: agentic AI can bridge gaps between disparate touchpoints, creating a cohesive and responsive service. The source describes agentic AI as capable of independently navigating complex customer needs, resulting in faster, smarter, and more personalized experiences without human intervention.

-----

### Source: https://www.startek.com/insight-post/blog/agentic-ai-in-customer-experience/
This source focuses on the technical features and business impact of agentic AI in customer service. One major advantage is continuous learning—agentic AI systems improve at resolving inquiries with each interaction. They provide instant responses around the clock, drastically reducing wait times and increasing engagement.

Agentic AI’s advanced language capabilities, powered by large language models (LLMs), allow seamless multi-language support, enabling businesses to provide consistent service to a global audience. These systems also leverage data from previous interactions to personalize service at scale, ensuring that each customer receives responses tailored to their preferences and history.

A particularly notable technical capability is the resolution of multi-step, complex queries. Agentic AI agents learn over time to handle increasingly complex tasks autonomously, significantly reducing the need to escalate issues to human agents. This self-improving, context-aware, and adaptive approach helps businesses deliver higher-quality support with fewer resources.

-----

</details>

---

<details>
<summary>How do state-of-the-art agentic AI systems (like OpenAI Operator or Deep Research agents) balance rule-based workflow structures with LLM-driven autonomy in hybrid applications?</summary>

### Source: https://openai.com/index/introducing-deep-research/
OpenAI's Deep Research agent is designed to autonomously perform multi-step research tasks by synthesizing large amounts of online information. It demonstrates advanced reasoning capabilities to complete complex assignments, combining the ability to follow structured workflows (such as gathering, analyzing, and synthesizing data) with the flexibility to adapt its approach as new information is found. This balance allows Deep Research to execute both rule-based sequences (such as dividing a report into sections or following citation requirements) and LLM-driven autonomy (such as dynamically refining search queries or deciding which sources to prioritize). The result is a hybrid system that leverages both predefined structures and the general intelligence of large language models.

-----

### Source: https://openai.com/index/introducing-operator/
Operator is an AI agent capable of independently executing tasks for users. It accepts a user’s directive and takes responsibility for carrying it out, which may involve planning, taking action, and adapting to new developments. Operator’s workflow combines structured, rule-based steps—where specific processes or sequences must be followed—with the autonomy of LLM-powered reasoning that allows it to make decisions in ambiguous or novel situations. This integration is key to Operator’s ability to work across a broad range of applications, from automating business workflows with well-defined procedures to handling creative or open-ended user requests that require flexibility and judgment.

-----

### Source: https://www.siddharthbharath.com/openai-agents-sdk/
The OpenAI Agents SDK provides a framework for building multi-agent workflows that blend rule-based structures with LLM-driven autonomy. In these hybrid applications, workflow control is often split between "manager agents" (which handle planning, orchestration, and breaking down tasks) and "tool-calling agents" (which execute concrete actions like web searches or data extraction). The manager agent, typically an LLM instance, interprets the overall task, decomposes it into structured steps, and decides which tools or agents to invoke. Tool-calling agents carry out individual, rule-based activities, but their actions and sequencing are autonomously determined by the manager’s reasoning. This architecture enables the system to follow predictable processes where possible, while using LLM autonomy to adapt strategies, resolve ambiguities, and synthesize results beyond rigid scripting.

-----

### Source: https://aisecuritychronicles.org/a-comparison-of-deep-research-ai-agents-52492ee47ca7
OpenAI’s Deep Research agent uses a hybrid architecture that splits responsibilities between specialized components: manager agents for orchestration and planning, and tool-calling agents for execution. The manager agent (powered by the o3 LLM series) interprets user instructions, breaks down the task into subtasks, and coordinates the workflow. This agent embodies the LLM’s autonomy—deciding what information to seek, which tools to use, and how to aggregate insights. Tool-calling agents perform rule-based operations, such as querying search APIs or extracting data from web pages. While individual tool activities are deterministic and rule-driven, the overall process is guided by the manager’s dynamic, LLM-based reasoning. This combination provides a robust structure for workflow reliability, while also enabling flexible adaptation and deep synthesis that would be infeasible with purely rule-based systems.

-----

### Source: https://www.sequoiacap.com/podcast/training-data-deep-research/
According to OpenAI’s Isa Fulford and Josh Tobin, Deep Research achieves its balance through end-to-end training on complex browsing tasks, rather than relying on hand-crafted rule graphs. This approach enables the LLM to develop flexible strategies for information gathering and synthesis that surpass the rigidity of manual orchestration. The model adapts its workflow based on intermediate findings, making it suitable for research tasks where the optimal sequence of actions is not known in advance. High-quality training data, featuring examples of nuanced research and synthesis, allows Deep Research to learn when to follow structured processes and when to exercise creative autonomy. Transparency—through visible citations and explicit reasoning—helps users trust the agent’s conclusions, while the speed and flexibility of LLM-driven adaptation enable comprehensive results that rule-based systems would struggle to achieve.

-----

</details>

---

<details>
<summary>What specific failure modes and real-world challenges have been reported for agentic systems deployed at scale, and what mitigation strategies have been successful?</summary>

### Source: https://oliverpatel.substack.com/p/top-12-papers-on-agentic-ai-governance
This report reviews governance strategies for agentic AI systems and highlights several real-world challenges and failure modes. Notable challenges include ensuring agentic AI systems reliably align with human intentions and organizational objectives, especially as they execute tasks with significant autonomy and limited oversight. Reported failures include cases where agentic systems take unintended actions due to misaligned goals or poor task specification, and difficulties in monitoring and intervening when behaviors deviate from expectations. Mitigation strategies discussed include the implementation of robust oversight mechanisms, frequent auditing, formal verification of agentic behaviors, and clear escalation protocols when unexpected outcomes arise. Additionally, organizations are urged to invest in the continuous updating of governance frameworks to keep pace with the rapidly evolving capabilities of agentic AI.

-----

### Source: https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage
McKinsey’s analysis of agentic AI highlights several practical deployment challenges and failure modes. These include the "GenAI paradox," where organizations struggle to balance the broad potential of agentic AI with the risks of uncontrolled autonomy. Key challenges reported in real-world deployments are difficulties in scaling agentic systems across business units due to inconsistent task specifications, issues with integration into existing business processes, and failures caused by lack of clear oversight. McKinsey notes that successful mitigation strategies involve setting up clear use-case boundaries, implementing rigorous monitoring and reporting processes, and designing fallback mechanisms for when agentic systems encounter uncertainty or ambiguous situations.

-----

### Source: https://www.navex.com/en-us/blog/article/preparing-for-the-compliance-challenges-of-agentic-ai/
NAVEX identifies core compliance and governance challenges for agentic AI. Reported failure modes include inappropriate delegation of mission-critical tasks to AI agents without proper vetting, the use of unapproved or insufficiently secure third-party agents, and a lack of explainability in agentic decision-making. These failures can lead to regulatory breaches, security incidents, and reputational risks. Successful mitigation strategies include establishing formal policies on which tasks AI agents are permitted to handle, maintaining strict controls on which agents are authorized within the organization, and requiring explainability and transparency in agentic operations. Ongoing monitoring and periodic re-evaluation of agentic assignments are also emphasized as key controls.

-----

### Source: https://gigster.com/blog/why-your-enterprise-isnt-ready-for-agentic-ai-workflows/
Gigster outlines that enterprises face three primary challenges when deploying agentic AI at scale: complex system integration, stringent access control and security requirements, and inadequate infrastructure readiness. Reported real-world issues include failed or delayed rollouts due to difficulties interfacing with legacy systems, security breaches resulting from insufficient access controls, and performance bottlenecks when existing infrastructure cannot support the demands of agentic workloads. The article notes that only 11% of enterprises have reached full deployment, with many pilots stalling because of these obstacles. Effective mitigation has involved prioritizing system integration efforts, upgrading infrastructure to support AI workloads, and implementing comprehensive access control and security protocols prior to wide-scale deployment.

-----

### Source: https://omdia.tech.informa.com/blogs/2025/may/mitigating-risks-maximising-potential
Omdia’s report emphasizes two main categories of failure modes for agentic AI: reliability issues and security vulnerabilities. Real-world reliability failures include agentic systems producing unexpected or harmful outcomes due to misaligned objectives or exposure to manipulated data and prompts. Security breaches are cited as a major concern, especially when agents interact with multiple APIs and data sources, multiplying potential attack vectors. Multi-agent frameworks are particularly vulnerable as interconnected agents can amplify the impact of a single compromised node. Mitigation strategies that have proven effective include strengthening oversight and monitoring, using robust access controls, implementing prompt and data validation, and restricting the use of unsanctioned or “shadow IT” agents. Furthermore, organizations are advised to align agent objectives closely with business and human values and to continuously update their risk management strategies as agentic AI capabilities evolve.

-----

</details>

---

<details>
<summary>Are there published diagrams, code samples, or technical deep-dives that visually illustrate the orchestration and memory management patterns in leading agentic AI architectures as of 2025?</summary>

### Source: https://www.lindy.ai/blog/ai-agent-architecture
This source provides a detailed breakdown of memory management patterns in modern agentic AI architectures as of 2025, with a focus on the integration of both working and persistent memory. Working memory is described as session-specific, handling temporary data such as active tasks or ongoing conversations. Persistent memory, by contrast, is long-term and survives across sessions, allowing agents to recall previous interactions and maintain continuity.

The orchestration of memory is often achieved through the use of vector databases, where information is stored as embeddings. This enables agents to retrieve relevant context via semantic similarity rather than simple keyword matching, supporting more nuanced and effective memory retrieval.

Visual and technical deep-dives are referenced through frameworks like LangChain, which provides modules for memory and retrieval management. Some platforms, such as Lindy’s Societies, advance this further by enabling groups of agents to share memory across tasks, allowing for collaborative multi-step workflows (e.g., summarize meeting → write follow-up → update CRM) without information loss. This design ensures agents behave consistently and can represent a business reliably, retaining necessary context across diverse tasks and sessions.

-----

### Source: https://www.jit.io/resources/devsecops/its-not-magic-its-memory-how-to-architect-short-term-memory-for-agentic-ai
This source delivers a technical deep-dive into the orchestration and memory management patterns of Jit’s agentic AppSec AI. The architecture centers on the concept that memory acts as the connective tissue, ensuring continuity across user queries, workflows, and sessions. Jit utilizes Langgraph’s thread IDs to manage context, with each thread unique to a tenant and session, ensuring context isolation.

Agent workflows are structured as directed graphs (using LangGraph), which enables sophisticated orchestration and replayability. The memory system leverages a checkpointer that saves every step in the agent workflow, including messages, transitions, and agent state. This enables features like replay, mid-graph recovery, and consistent context propagation.

A shared state object—traveling with the conversation thread—holds not only messages but also structured metadata, user inputs, and any other contextual data needed by the agent. This state is thread-scoped, ensuring memory isolation. The orchestration pattern features a supervisor agent that delegates tasks to downstream agents, implementing clear delegation and modular task execution. These architectural patterns are visually depicted in diagrams and explained in code samples within the resource.

-----

### Source: https://www.infosys.com/iki/research/agentic-ai-architecture-blueprints.html
This Infosys report provides architectural blueprints for agentic AI systems, focusing on how companies are piloting and deploying these architectures in 2025. The document discusses the orchestration of multiple agents and the integration of memory subsystems, emphasizing workflow-driven designs. Visual diagrams illustrate system-level compositions, showcasing how memory management and agent orchestration are mapped into modular, reusable components.

The report offers technical deep-dives, including flowcharts and architecture diagrams that highlight agent coordination, persistent memory layers, and retrieval mechanisms. It details how short-term and long-term memories are handled via dedicated subsystems, sometimes leveraging vector databases or structured storage. Additionally, code snippets and pseudocode are included to illustrate the data flow between agents and their memory modules, making the architectural patterns transparent for technical audiences.

-----

### Source: https://www.plivo.com/blog/agentic-ai-frameworks/
Plivo's 2025 guide covers leading agentic AI frameworks and their architectural approaches to orchestration and memory management. The article explains how modern frameworks facilitate complex agent interactions, with orchestration patterns such as hierarchical delegation, task routing, and parallel execution. It provides technical diagrams that visualize how agents communicate and coordinate tasks through centralized or distributed controllers.

For memory management, the guide outlines common patterns such as working (short-term) and persistent (long-term) memory, with many systems integrating vector database backends for scalable semantic retrieval. The diagrams and code samples show how agents fetch, update, and share memory, ensuring context-aware reasoning and multi-agent collaboration. The source also describes best practices for maintaining state consistency, preventing data leaks between agents, and leveraging memory for adaptive behavior.

-----

### Source: https://orq.ai/blog/ai-agent-architecture
Orq.ai’s 2025 article details the core principles and tools for building scalable agentic AI architectures. It describes how orchestration is typically handled through modular, event-driven designs, with agents managed via orchestration layers or workflow engines. The article includes visual diagrams to illustrate agent orchestration flows and memory access patterns.

For memory management, it explains the use of both volatile (short-term) and persistent (long-term) memory, with patterns such as context propagation, memory sharing, and role-based access. Technical deep-dives and code samples demonstrate how state is maintained across distributed agents and how agents synchronize memory for collaborative tasks. The resource highlights modern tools and best practices for implementing robust memory architectures in production systems.

-----

</details>

---

## Sources Scraped From Research Results

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
https://www.anthropic.com/research/building-effective-agents
</details>

---
<details>
<summary>How to think about agent frameworks</summary>

[Skip to content](https://blog.langchain.com/how-to-think-about-agent-frameworks/#main)

**TL;DR:**

- **The hard part of building reliable agentic systems is making sure the LLM has the appropriate context at each step. This includes both controlling the exact content that goes into the LLM, as well as running the appropriate steps to generate relevant content.**
- **Agentic systems consist of both workflows and agents (and everything in between).**
- **Most agentic frameworks are neither declarative or imperative orchestration frameworks, but rather just a set of agent abstractions.**
- **Agent abstractions can make it easy to get started, but they can often obfuscate and make it hard to make sure the LLM has the appropriate context at each step.**
- **Agentic systems of all shapes and sizes (agents or workflows) all benefit from the same set of helpful features, which can be provided by a framework, or built from scratch.**
- **LangGraph is best thought of as a orchestration framework (with both declarative and imperative APIs), with a series of agent abstractions built on top.**

OpenAI recently released a guide on building agents which contains some misguided takes like the below:

![](https://blog.langchain.com/content/images/2025/04/Go0FliaXoAANDWD.jpeg)

This callout initially angered me, but after starting to write a response I realized: thinking about agent frameworks is complicated! There are probably 100 different agent frameworks, there are a lot of different axes to compare them on, sometimes they get conflated (like in this quote). There is a lot of hype, posturing, and noise out there. There is very little precise analysis or thinking being done about agent frameworks. This blog is our attempt to do so. We will cover:

- **Background Info**
  - What is an agent?
  - What is hard about building agents?
  - What is LangGraph?
- **Flavors of agentic frameworks**
  - “Agents” vs “workflows”
  - Declarative vs non-declarative
  - Agent abstractions
  - Multi agent
- **Common Questions**
  - What is the value of a framework?
  - As the models get better, will everything become agents instead of workflows?
  - What did OpenAI get wrong in their take?
  - How do all the agent frameworks compare?

Throughout this blog I will make repeated references to a few materials:

- [OpenAI’s guide on building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf?ref=blog.langchain.com) (which I don’t think is particularly good)
- [Anthropic’s guide on building effective agents](https://www.anthropic.com/engineering/building-effective-agents?ref=blog.langchain.com) (which I like a lot)
- [LangGraph](https://github.com/langchain-ai/langgraph?ref=blog.langchain.com) (our framework for building reliable agents)

# Background info

Helpful context to set the stage for the rest of the blog.

## What is an agent

There is no consistent definition of an agent, and they are often offered through different lenses.

OpenAI takes a higher level, more thought-leadery approach to defining an agent.

> Agents are systems that independently accomplish tasks on your behalf.

I am personally not a fan of this. This is a vague statement that doesn’t really help me understand what an agent is. It’s just thought-leadership and not practical at all.

Compare this to Anthropic’s definition:

> "Agent" can be defined in several ways. Some customers define agents as fully autonomous systems that operate independently over extended periods, using various tools to accomplish complex tasks. Others use the term to describe more prescriptive implementations that follow predefined workflows. At Anthropic, we categorize all these variations as **agentic systems**, but draw an important architectural distinction between **workflows** and **agents**:
>
> **Workflows** are systems where LLMs and tools are orchestrated through predefined code paths.
>
> **Agents**, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks.

I like Anthropic’s definition better for a few reasons:

- Their definition of an agent is much more precise and technical.
- They also make reference to the concept of “agentic systems”, and categorize both workflows and agents as variants of this. I **love** this.

💡

Nearly all of the “agentic systems” we see in production are a **combination** of “workflows” and “agents”.

Later in the blog post, Anthropic defines agents as “… typically just LLMs using tools based on environmental feedback in a loop.”

![](https://blog.langchain.com/content/images/2025/04/58d9f10c985c4eb5d53798dea315f7bb5ab6249e-2401x1000.webp)

Despite their grandiose definition of an agent at the start, this is basically what OpenAI means as well.

These types of agents are parameterized by:

- The model to use
- The instructions (system prompt) to use
- The tools to use

You call the model in a loop. If/when it decides to call a tool, you run that tool, get some observation/feedback, and then pass that back into the LLM. You run until the LLM decides to not call a tool (or it calls a tool that triggers a stopping criteria).

Both OpenAI and Anthropic call out workflows as being a different design pattern than agents. The LLM is less in control there, the flow is more deterministic. This is a helpful distinction!

Both OpenAI and Anthropic explicitly call out that you do not always need agents. In many cases, workflows are simpler, more reliable, cheaper, faster, and more performant. A great quote from the Anthropic post:

> When building applications with LLMs, we recommend finding the simplest solution possible, and only increasing complexity when needed. This might mean not building agentic systems at all. Agentic systems often trade latency and cost for better task performance, and you should consider when this tradeoff makes sense.
>
> When more complexity is warranted, workflows offer predictability and consistency for well-defined tasks, whereas agents are the better option when flexibility and model-driven decision-making are needed at scale.

OpenAI says something similar:

> Before committing to building an agent, validate that your use case can meet these criteria clearly. Otherwise, a deterministic solution may suffice.

In practice, we see that most “agentic systems” are a combination of workflows and agents. This is why I actually **hate** talking about whether something is an agent, but prefer talking about how agentic a system is. h/t the great Andrew Ng for this way of [thinking about things](https://x.com/AndrewYNg/status/1801295202788983136?ref=blog.langchain.com):

> Rather than having to choose whether or not something is an agent in a binary way, I thought, it would be more useful to think of systems as being agent-like to different degrees. Unlike the noun “agent,” the adjective “agentic” allows us to contemplate such systems and include all of them in this growing movement.

## What is hard about building agents?

I think most people would agree that building agents is hard. Or rather - building an agent as a prototype is easy, but a reliable one, that can power business-critical applications? That is hard.

The tricky part is exactly that - making it reliable. You can make a demo that looks good on Twitter easily. But can you run it to power a business critical application? Not without a lot of work.

We did a survey of agent builders a few months ago and asked them: _“What is your biggest limitation of putting more agents in production?”_ The number one response by far was “performance quality” - it’s still really hard to make these agents work.

![](https://blog.langchain.com/content/images/2025/04/67347b1aed9686aad4544fef_9.-What-is-your-biggest-limitation.svg)

_What causes agents to perform poorly sometimes?_ The LLM messes up.

_Why does the LLM mess up?_ Two reasons: (a) the model is not good enough, (b) the wrong (or incomplete) context is being passed to the model.

From our experience, it is very frequently the second use case. What causes this?

- Incomplete or short system messages
- Vague user input
- Not having access to the right tools
- Poor tool descriptions
- Not passing in the right context
- Poorly formatted tool responses

💡

**The hard part of building reliable agentic systems is making sure the LLM has the appropriate context at each step. This includes both controlling the exact content that goes into the LLM, as well as running the appropriate steps to generate relevant content.**

As we discuss agent frameworks, it’s helpful to keep this in mind. Any framework that makes it harder to control **exactly** what is being passed to the LLM is just getting in your way. It’s already hard enough to pass the correct context to the LLM - why would you make it harder on yourself?

## What is LangGraph

💡

LangGraph is best thought of as a orchestration framework (with both declarative and imperative APIs), with a series of agent abstractions built on top.

LangGraph is an event-driven framework for building agentic systems. The two most common ways of using it are through:

- a [declarative, graph-based syntax](https://langchain-ai.github.io/langgraph/tutorials/introduction/?ref=blog.langchain.com)
- [agent abstractions](https://langchain-ai.github.io/langgraph/agents/overview/?ref=blog.langchain.com) (built on top of the lower level framework)

LangGraph also supports a [functional API](https://langchain-ai.github.io/langgraph/concepts/functional_api/?ref=blog.langchain.com), as well as the underlying [event-driven API](https://langchain-ai.github.io/langgraph/concepts/pregel/?ref=blog.langchain.com). There exist both [Python](https://langchain-ai.github.io/langgraph/?ref=blog.langchain.com) and [Typescript](https://langchain-ai.github.io/langgraphjs/?ref=blog.langchain.com) variants.

Agentic systems can be represented as [nodes](https://langchain-ai.github.io/langgraph/concepts/low_level/?ref=blog.langchain.com#nodes) and [edges](https://langchain-ai.github.io/langgraph/concepts/low_level/?ref=blog.langchain.com#edges). Nodes represent units of work, while edges represent transitions. Nodes and edges are nothing more than normal Python or TypeScript code - so while the structure of the graph is represented in a declarative manner, the inner functioning of the graph’s logic is normal, imperative code. Edges can be either [fixed](https://langchain-ai.github.io/langgraph/concepts/low_level/?ref=blog.langchain.com#normal-edges) or [conditional](https://langchain-ai.github.io/langgraph/concepts/low_level/?ref=blog.langchain.com#conditional-edges). So while the structure of the graph is declarative, the path through the graph can be completely dynamic.

LangGraph comes with a [built-in persistence layer](https://langchain-ai.github.io/langgraph/concepts/persistence/?ref=blog.langchain.com). This enables [fault tolerance](https://langchain-ai.github.io/langgraph/concepts/persistence/?h=fault+to&ref=blog.langchain.com#fault-tolerance), [short-term memory](https://langchain-ai.github.io/langgraph/concepts/memory/?ref=blog.langchain.com#short-term-memory), and [long-term memory](https://langchain-ai.github.io/langgraph/concepts/memory/?ref=blog.langchain.com#long-term-memory).

This persistence layer also enables “ [human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/?ref=blog.langchain.com)” and “ [human-on-the-loop](https://langchain-ai.github.io/langgraph/concepts/time-travel/?ref=blog.langchain.com)” patterns, such as interrupt, approve, resume, and time travel.

LangGraph has built-in support for [streaming](https://langchain-ai.github.io/langgraph/concepts/streaming/?ref=blog.langchain.com): of tokens, node updates, and arbitrary events.

LangGraph integrates seamlessly with [LangSmith](https://docs.smith.langchain.com/?ref=blog.langchain.com) for debugging, evaluation, and observability.

# Flavors of agentic frameworks

Agentic frameworks are different across a few dimensions. Understanding - and not conflating - these dimensions is key to being able to properly compare agentic frameworks.

## Workflows vs Agents

Most frameworks contain higher level agent abstractions. Some frameworks include some abstraction for common workflows. LangGraph is a low level orchestration framework for building agentic systems. LangGraph supports [workflows, agents, and anything in-between](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/?ref=blog.langchain.com). We think this is crucial. As mentioned, most agentic systems in production are a combination of workflows and agents. A production-ready framework needs to support both.

Let’s remember what is hard about building reliable agents - making sure the LLM has the right context. Part of why workflows are useful is that they make it easy to pass the right context to LLMs. You decide exactly how the data flows.

As you think about where on spectrum of “workflow” to “agent” you want to build your application, there are two things to think about:

- Predictability vs agency
- Low floor, high ceiling

**Predictability vs agency**

As your system becomes more agentic, it will become less predictable.

Sometimes you want or need your system to be predictable - for user trust, regulatory reasons, or other.

Reliability does not track 100% with predictability, but in practice they can be closely related.

Where you want to be on this curve is pretty specific to your application. LangGraph can be used to build applications anywhere on this curve, allowing you to move to the point on the curve that you want to be.

![](https://blog.langchain.com/content/images/2025/04/Screenshot-2025-04-20-at-10.43.31-AM.png)

**High floor, low ceiling**

When thinking about frameworks, it can be helpful to think about their floors and ceilings:

- Low floor: A **low floor** framework is beginner-friendly and easy to get started with
- High floor: A framework with a **high floor** means it has a steep learning curve and requires significant knowledge or expertise to begin using it effectively.
- Low ceiling: A framework with a **low ceiling** means it has limitations on what can be accomplished with it (you will quickly outgrow it).
- High ceiling: A **high ceiling** framework offers extensive capabilities and flexibility for advanced use cases (it grows with you?).

Workflow frameworks offer a high ceiling, but come with a high floor - you have to write lot of the agent logic yourself.

Agent frameworks are low floor, but low ceiling - easy to get started with, but not enough for non-trivial use cases.

LangGraph aims to have aspects that are low floor ( [built-in agent abstractions](https://langchain-ai.github.io/langgraph/agents/overview/?ref=blog.langchain.com) that make it easy to get started) but also high ceiling ( [low-level functionality](https://langchain-ai.github.io/langgraph/?ref=blog.langchain.com) to achieve advanced use cases).

## Declarative vs non-declarative

There are benefits to declarative frameworks. There are also downsides. This is a seemingly endless debate among programmers, and everyone has their own preferences.

When people say non-declarative, they are usually implying imperative as the alternative.

Most people would describe LangGraph as a declarative framework. This is only partially true.

First - while the connections between the nodes and edges are done in a declarative manner, the actual nodes and edges are nothing more than Python or TypeScript functions. Therefore, LangGraph is kind of a blend between declarative and imperative.

Second - we actually support other APIs besides the recommended declarative API. Specifically, we support both [functional](https://langchain-ai.github.io/langgraph/concepts/functional_api/?ref=blog.langchain.com) and [event-driven APIs](https://langchain-ai.github.io/langgraph/concepts/pregel/?ref=blog.langchain.com). While we think the declarative API is a useful mental model, we also recognize it is not for everyone.

A common comment about LangGraph is that is like Tensorflow (a declarative deep learning framework), while frameworks like Agents SDK are like Pytorch (an imperative deep learning framework).

This is just incorrect. Frameworks like Agents SDK (and original LangChain, CrewAI, etc) are neither declarative or imperative - they are just abstractions. They have an agent abstraction (a Python class) and it contains a bunch of internal logic that runs the agent. They’re not really orchestration frameworks. They are just abstractions.

## Agent Abstractions

Most agent frameworks contain an agent abstraction. They usually start as a class that involves a prompt, model, and tools. Then they add in a few more parameters… then a few more… then even more. Eventually you end up with a litany of parameters that control a multitude of behaviors, all abstracted behind a class. If you want to see what’s going on, or change the logic, you have to go into the class and modify the source code.

💡

These abstractions end up making it really really hard to understand or control exactly what is going into the LLM at all steps. This is important - having this control is crucial for building reliable agents (as discussed above). This is the danger of agent abstractions.

We learned this the hard way. This was the issue with the original LangChain chains and agents. They provided abstractions that got in the way. One of those original abstractions from two years ago was an agent class that took in a model, prompt, and tools. This isn’t a new concept. It didn’t provide enough control back then, and it doesn’t now.

To be clear, there is some value in these agent abstractions. It makes it easier to get started. But I don’t think these agent abstractions are good enough to build reliable agents yet (and maybe ever).

We think the best way to think about these agent abstractions is like Keras. They provide higher level abstractions to get started easily. But it’s crucial to make sure they are built on top of a lower level framework so you don’t outgrow it.

That is why we have built agent abstractions on top of LangGraph. This provides an easy way to get started with agents, but if you need to escape to lower-level LangGraph you easily can.

## Multi Agent

Oftentimes agentic systems won’t just contain one agent, they will contain multiple. OpenAI says in their report:

> For many complex workflows, splitting up prompts and tools across multiple agents allows for improved performance and scalability. When your agents fail to follow complicated instructions or consistently select incorrect tools, you may need to further divide your system and introduce more distinct agents.

💡

The key part of multi agent systems is how they communicate. Again, the hard part of building agents is getting the right context to LLMs. Communication between these agents is important.

There a bunch of ways to do this! Handoffs are one way. This is an agent abstraction from Agents SDK that I actually quite like.

But the best way for these agents to communicate can sometimes be workflows. Take all the workflow diagrams in Anthropic’s blog post, and replace the LLM calls with agents. This blend of workflows and agents often gives the best reliability.

![](https://blog.langchain.com/content/images/2025/04/7418719e3dab222dccb379b8879e1dc08ad34c78-2401x1000.webp)

Again - agentic systems are not just workflows, or just an agent. They can be - and often are - a combination of the two. As Anthropic points out in their blog post:

> **Combining and customizing these patterns**
>
> These building blocks aren't prescriptive. They're common patterns that developers can shape and combine to fit different use cases.

# Common Questions

Having defined and explored the different axes that you should be evaluating frameworks on, let’s now try to answer some common questions.

## What is the value of a framework?

We often see people questioning whether they need a framework to build agentic systems. What value can agent frameworks provide?

**Agent abstractions**

Frameworks are generically useful because they contain useful abstractions which make it easy to get started and provide a common way for engineers to build, making it easier to onboard and maintain projects. As mentioned above, there are real downsides to agent abstractions as well. For most agent frameworks, this is the sole value they provide. We worked really hard to make sure this was not case for LangGraph.

**Short term memory**

Most agentic applications today involve some sort of multi-turn (e.g. chat) component. LangGraph provides [production ready storage to enable multi-turn experiences (threads)](https://langchain-ai.github.io/langgraph/concepts/memory/?ref=blog.langchain.com#short-term-memory).

**Long term memory**

While still early, I am very bullish on agentic systems learning from their experiences (e.g. remembering things across conversations). LangGraph provides [production ready storage for cross-thread memory](https://langchain-ai.github.io/langgraph/concepts/memory/?ref=blog.langchain.com#long-term-memory).

**Human-in-the-loop**

Many agentic systems are made better with some human-in-the-loop component. Examples include getting feedback from the user, approving a tool call, or editing tool call arguments. LangGraph provides [built in support to enable these workflows in a production system](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/?ref=blog.langchain.com).

**Human-on-the-loop**

Besides allowing the user to affect the agent as it is running, it can also be useful to allow the user to inspect the agent’s trajectory after the fact, and even go back to earlier steps and then rerun (with changes) from there. We call this human-on-the-loop, and LangGraph provides [built in support for this](https://langchain-ai.github.io/langgraph/concepts/time-travel/?ref=blog.langchain.com).

**Streaming**

Most agentic applications take a while to run, and so providing updates to the end user can be critical for providing a good user experience. LangGraph provides [built in streaming of tokens, graph steps, and arbitrary streams](https://langchain-ai.github.io/langgraph/concepts/streaming/?ref=blog.langchain.com).

**Debugging/observability**

The hard part of building reliable agents is making sure you are passing the right context to the LLM. Being able to inspect the exact steps taken by an agent, and the exact inputs/outputs at each step is crucial for building reliable agents. LangGraph integrates seamlessly with [LangSmith](https://docs.smith.langchain.com/?ref=blog.langchain.com) for best in class debugging and observability. Note: AI observability is different from traditional software observability (this deserves a separate post).

**Fault tolerance**

Fault tolerance is a key component of traditional frameworks (like Temporal) for building distributed applications. LangGraph makes fault tolerance easier with [durable workflows](https://langchain-ai.github.io/langgraph/concepts/durable_execution/?ref=blog.langchain.com) and [configurable retries](https://langchain-ai.github.io/langgraph/how-tos/node-retries/?h=retr&ref=blog.langchain.com).

**Optimization**

Rather than tweaking prompts manually by hand, it can sometimes be easier to define an evaluation dataset and then automatically optimize your agent based on this. LangGraph currently does not support this out of the box - we think it is a little early for this. But I wanted to include this because I think it is an interesting dimension to consider, and something we are constantly keeping our eyes on. `dspy` is the best framework for this currently.

💡

All of these value props (aside from the agent abstractions) provide value for both agents, workflows, and everything in between.

**So - do you really need an agentic framework?**

If your application does not require all of these features, and/or if you want to build them yourself, then you may not need one. Some of them (like short term memory) aren’t terribly complicated. Others of them (like human-on-the-loop, or LLM specific observability) are more complicated.

And regarding agent abstractions: I agree with what Anthropic says in their post:

> If you do use a framework, ensure you understand the underlying code. Incorrect assumptions about what's under the hood are a common source of customer error.

## As the models get better, will everything become agents instead of workflows?

One common argument in favor of agents (compared to workflows) is that while they don’t work now, they will work in the future, and therefore you will just need the simple, tool-calling agents.

I think multiple things can be true:

- The performance of these tool-calling agents will rise
- It will still really important to be able to control what goes into the LLM (garbage in, garbage out)
- For some applications, this tool calling loop will be enough
- For other applications, workflows will just be simpler, cheaper, faster, and better
- For most applications, the production agentic system will be a combination of workflows and agents

I don’t think OpenAI or Anthropic would debate any of these points? From Anthropic’s post:

> When building applications with LLMs, we recommend finding the simplest solution possible, and only increasing complexity when needed. This might mean not building agentic systems at all. Agentic systems often trade latency and cost for better task performance, and you should consider when this tradeoff makes sense.

And from OpenAI's post:

> Before committing to building an agent, validate that your use case can meet these criteria clearly. Otherwise, a deterministic solution may suffice.

Will there be applications where this simple tool calling loop will be enough? I think this will likely only be true if you are using a model trained/finetuned/RL’d on lots of data that is specific to your use case. This can happen in two ways:

- Your task is unique. You gather a lot of data and train/finetune/RL your own model.
- Your task is not unique. The large model labs are training/finetuning/RL’ing on data representative of your task.

(Side note: if I was building a vertical startup in an area where my task was not unique, I would be pretty worried about the long term viability of my startup).

**Your task is unique**

I would bet that most use cases (certainly most enterprise use cases) fall into this category. How AirBnb handles customer support is different from how Klarna handles customer support which is different from how Rakuten handles customer support. There is a ton of subtlety in these tasks. Sierra - a leading agent company in the customer support space - is not building a single customer support _agent_, but rather a customer support agent _platform_:

> The Sierra Agent SDK enables developers to use a declarative programming language to build powerful, flexible agents using composable skills to express procedural knowledge

They need to do this because each company’s customer support experience is unique enough where a generic agent is not performant enough.

One example of an agent that is a simple tool calling loop using a model trained on a specific task: [OpenAI’s Deep Research](https://www.sequoiacap.com/podcast/training-data-deep-research/?ref=blog.langchain.com). So it can be done, and it can produce amazing agents.

If you can train a SOTA model on your specific task - then yes, you probably don’t need a framework that enables arbitrary workflows, you’ll just use a simple tool calling loop. In this case, agents will be preferred over workflows.

A very open question in my mind is: how many agent companies will have the data, tools, or knowledge to train a SOTA model for their task? At this exact moment, I think only the large model labs are able to do this. But will that change? Will a small vertical startup be able to train a SOTA model for their task? I am very interested in this question. If you are currently doing this - please reach out!

**Your task is not unique**

I think some tasks are generic enough that the large model labs will be able to provide models that are good enough to do the simple tool-calling loop on these non-generic tasks.

OpenAI released their Computer Use model via the API, which is a model finetuned on generic computer use data aiming to be good enough at that generic task. (Side note: I don’t think it is close to good enough yet).

Code is an interesting example of this. Coding is relatively generic, and coding has definitely been a break out use case for agents so far. Claude code and OpenAI’s Codex CLI are two examples of coding agents that use this simple tool calling loop. I would bet heavily that the base models are trained on lots of coding data and tasks (see evidence [here](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/text-editor-tool?ref=blog.langchain.com) that Anthropic does this).

Interestingly - as the general models are trained on this data, how much does the exact shape of this data matter? Ben Hylak had an [interesting tweet](https://x.com/benhylak/status/1912922457012572364?ref=blog.langchain.com) the other day that seemed to resonate with folks:

> models don't know how to use cursor anymore.
>
> they're all being optimized for terminal. that's why 3.7 is and o3 are so awful in Cursor, and so amazing outside of it.

This could suggest two things:

- Your task has to be very very close to the task the general models are trained on. The less similar your task is, the less likely it is that the general models will be good enough for your use case.
- Training the general models on other specific tasks may decrease performance on your task. I’m sure there is just as much (if not more) data similar to Cursor’s use case used to train the new models. But if there is this influx of new data of a slightly different shape, it outweighs any other type of data. This implies it is currently hard for the general models to be really amazing at a large number of tasks.

💡

Even for applications where agents are preferred to anything workflow-like, you will still benefit features of a framework that don’t have to do with low level workflow control: short term memory storage, long term memory storage, human-in-the-loop, human-on-the-loop, streaming, fault tolerance, debugging/observability.

## What did OpenAI get wrong in their take?

If we revisit OpenAI's stance, we find it to be premised on false dichotomies that conflate different dimensions of "agentic frameworks" in order to inflate the value of their singular abstraction. Specifically, it conflates “declarative vs imperative” with “agent abstractions” as well as “workflows vs agents”.

💡

Ultimately it misses the mark on what the main challenge is for building production agentic systems and the main value that should be provided by a framework, which is: a reliable orchestration layer that gives developers explicit control over what context reaches their LLMs while seamlessly handling production concerns like persistence, fault tolerance, and human-in-the-loop interactions.

Let's break down specific parts I take issue with:

![](https://blog.langchain.com/content/images/2025/04/Go0FliaXoAANDWD-1.jpeg)

**”Declarative vs non-declarative graphs”**

LangGraph is not fully declarative - but it’s declarative enough so that’s not my main gripe. My main gripe would be that “non-declarative” is doing a lot of work and misleading. Normally when people criticize declarative frameworks they would prefer a more imperative framework. But Agents SDK is NOT an imperative framework. It’s an abstraction. A more proper title would be “Declarative vs imperative” or “Do you need an orchestration framework” or “Why agent abstractions are all you need” or “Workflows vs Agents” depending on what they want to argue (they seem to argue both below).

**”this approach can quickly become cumbersome and challenging as workflows grow more dynamic and complex”**

This doesn’t have anything to do with declarative or non-declarative. This has everything to do with workflows vs agents. You can easily express the agent logic in Agents SDK as a declarative graph, and that graph is just as dynamic and flexible as Agents SDK.

And on the point of workflows vs agents. A lot of workflows do not require this level of dynamism and complexity. Both OpenAI and Anthropic acknowledge this. You should use workflows when you can use workflows. Most agentic systems are a combination. Yes, if a workflow is really dynamic and complex then use an agent. But don’t use an agent for everything. OpenAI literally says this earlier in the paper.

**”often necessitating the learning of specialized domain-specific languages”**

Again - Agents SDK is not an imperative framework. It is an abstraction. It also has a domain specific language (it’s abstractions). I would argue that having to learn and work around Agents SDK abstractions is, at this point in time, worse than having to learn LangGraph abstractions. Largely because the hard thing about building reliable agents is making sure the agent has the right context, and Agents SDKs obfuscates that WAY more than LangGraph.

**"more flexible"**

This is just strictly not true. It’s the opposite of the truth. Everything you can do with Agents SDK you can do with LangGraph. Agents SDK only lets you do 10% of what you can do with LangGraph.

**“code-first”**

With Agents SDK you write their abstractions. With LangGraph you write a **large** amount of normal code. I don’t see how Agents SDK is more code first.

**”using familiar programming constructs”**

With Agents SDK you have to learn a whole new set of abstractions. With LangGraph you write a large amount of normal code. What is more familiar than that?

**”enabling more dynamic and adaptable agent orchestration”**

Again - this doesn’t have to with declarative vs non-declarative. This has to do with workflows vs agents. See above point.

## Comparing Agent Frameworks

We've talked about a lot of different components of agent frameworks:

- Are they flexible orchestration layer, or just an agent abstraction?
- If they are a flexible orchestration layer, are they declarative or otherwise?
- What features (aside from agent abstractions) does this framework provide?

I thought it would be fun to try to list out these dimensions in an spreadsheet. I tried to be as impartial as possible about this ( [I asked for - and got - a lot of good feedback from Twitter!](https://x.com/hwchase17/status/1913662736963412365?ref=blog.langchain.com)).

This currently contains comparisons to Agents SDK, Google's ADK, LangChain, Crew AI, LlamaIndex, Agno AI, Mastra, Pydantic AI, AutoGen, Temporal, SmolAgents, DSPy.

If I left out a framework (or got something wrong about a framework) please leave a comment!

💡

You can find a living version of the spreadsheet [here](https://docs.google.com/spreadsheets/d/1B37VxTBuGLeTSPVWtz7UMsCdtXrqV5hCjWkbHN8tfAo/edit?usp=sharing&ref=blog.langchain.com).

# Conclusion

- **The hard part of building reliable agentic systems is making sure the LLM has the appropriate context at each step. This includes both controlling the exact content that goes into the LLM, as well as running the appropriate steps to generate relevant content.**
- **Agentic systems consist of both workflows and agents (and everything in between).**
- **Most agentic frameworks are neither declarative or imperative orchestration frameworks, but rather just a set of agent abstractions.**
- **Agent abstractions can make it easy to get started, but they can often obfuscate and make it hard to make sure the LLM has the appropriate context at each step.**
- **Agentic systems of all shapes and sizes (agents or workflows) all benefit from the same set of helpful features, which can be provided by a framework, or built from scratch.**
- **LangGraph is best thought of as a orchestration framework (with both declarative and imperative APIs), with a series of agent abstractions built on top.**

### Tags

[In the Loop](https://blog.langchain.com/tag/in-the-loop/)

### Join our newsletter

Updates from the LangChain team and community

Enter your emailSubscribe

Processing your application...

Success! Please check your inbox and click the link to confirm your subscription.

Sorry, something went wrong. Please try again.

Subscribe

### Original URL
https://blog.langchain.dev/how-to-think-about-agent-frameworks/
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

Finally, we've given Devin the ability to actively collaborate with the user. Devin reports on its progress in real time, accepts feedback, and works together with you through design choices as needed.

Here's a sample of what Devin can do:

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

![Devin Performance Chart](https://cdn.sanity.io/images/2mc9cv2v/production/5dd1cd4fd86149ed2cf4d8ab605f99707040615e-1600x858.png)

Devin was evaluated on a random 25% subset of the dataset. Devin was unassisted, whereas all other models were assisted (meaning the model was told exactly which files need to be edited).

We plan to publish a more detailed technical report soon—stay tuned for more details.

### Original URL
https://cognition.ai/blog/introducing-devin
</details>

---
<details>
<summary>Agentic AI Systems: Applications, Examples, and Best Practices</summary>

Managing complex workflows in highly regulated industries requires more than isolated AI tools.

Today, you need intelligent, goal-driven AI systems that autonomously handle tasks, make decisions, and collaborate seamlessly.

Agentic AI systems are improving automation by enabling multi-agent orchestration. According to industry reports and [predictions](https://www.pwc.com/us/en/tech-effect/ai-analytics/ai-predictions.html?utm_source=chatgpt.com), companies leveraging AI-driven automation see significant efficiency and cost reduction improvements, making Agentic AI the next major shift in enterprise AI.

Below, we’ll explain [what makes Agentic AI systems different](https://www.multimodal.dev/post/agentic-ai-the-vanguard-of-modern-enterprise), how they work, and why you should implement them in your business.

## What Are Agentic AI Systems?

Agentic AI Systems represent a shift from **static to autonomous**, goal-driven, multi-agent collaboration.

While standalone AI Agents perform isolated tasks, Agentic AI systems consist of **multiple AI Agents working together**, integrating memory, reasoning, API connections, and automation to function within an existing workflow.

Such level of automation allows artificial intelligence beyond simple task execution, offering orchestrated intelligence that improves efficiency, accuracy, and decision-making in complex workflows.

## AI Agents vs. Agentic AI Systems

![Single agent system vs. agentic AI system](https://cdn.prod.website-files.com/636e9a9a8d334e3450b08cc9/67ca2144572def6cce6a091a_AI-Agents-vs-Agentic-AI-Systems.webp)

While many use these terms interchangeably, there is an important difference between AI Agents and AI Agentic Systems:

- **AI Agents** – Individual Agents that perform specific tasks, like answering customer inquiries or extracting sensitive data from documents. A great example is any of our 6 [AI Agents](https://www.multimodal.dev/banking). For example, [Unstructured AI](https://www.multimodal.dev/unstructured-ai) is an AI Agent that performs a specific task of processing complex and unstructured document formats to prepare it for AI use.
- **Agentic AI Systems** – They go beyond performing specific tasks by enabling AI Agents to work together, interact, share insights, and dynamically collaborate to achieve a larger goal. For example, an Agentic AI system involves multiple AI Agents that handle inquiries, retrieve data from CRM, analyze previous interactions, escalate complex cases, and automate follow-ups.

![AI Agents vs. Agentic AI Systems](https://cdn.prod.website-files.com/636e9a9a8d334e3450b08cc9/67ca2159e260898c258ae4ca_Table-AI-Agents-vs-Agentic-AI-Systems.webp)

[AgentFlow](https://www.multimodal.dev/agentflow) is a great example of an Agentic AI system that helps insurance and finance companies create and manage AI Agents to automate workflows end-to-end.

## The Key Features of Agentic AI Systems

The defining characteristic of Agentic AI Systems is **multi-agent orchestration**–the ability to coordinate specialized AI Agents into one autonomous, goal-driven unit.

This orchestration ensures that:

- **Tasks are distributed efficiently** between AI Agents with specific expertise.
- **Information flows seamlessly** between AI Agents to maintain context and coherence.
- **Decision-making improves over time** through memory and reasoning capabilities.
- **Adaptive learning enables continuous optimization** based on past interactions.
- **Dynamic role assignment allows AI Agents to switch tasks** as needed to improve flexibility.
- **API integrations connect the system with external data sources** for greater automation.

Instead of relying on a single AI Agent to handle an entire process, multi-agent orchestration enables a system where AI Agents collaborate dynamically, just like human teams.

This approach improves scalability, reliability, and adaptability, which ensures that AI-driven workflows remain efficient even as business needs evolve.

## Applications of Agentic AI Systems

![Applications of Agentic AI systems](https://cdn.prod.website-files.com/636e9a9a8d334e3450b08cc9/67ca218118d90884c2808521_Applications-of-Agentic-AI-Systems.webp)

Agentic AI systems can improve highly regulated industries by automating and optimizing complex workflows. Some of the **key applications include**:

### Insurance

The insurance sector benefits from an Agentic AI system by automating processes and improving customer service and risk evaluation. Specific [workflow applications](https://www.multimodal.dev/insurance) include:

- **Claims processing** – AI Agents verify documents, cross-check policies, and approve claims faster or escalate them for manual review.
- **Underwriting and risk assessment** – AI-powered risk analysis assesses customer profiles, suggests policy adjustments in real-time, and predicts claim likelihood.
- **Customer support and personalization** – AI-driven assistants provide customers with policy updates, recommendations, and renewal reminders and provide claims assistance.
- **Automated document generation** – AI capabilities automate the creation of policies, endorsements, renewals, and other necessary documentation based on customer data and interactions.
- **Regulatory reporting** – AI helps insurers meet regulatory requirements by automating the collection, creation, formatting, and submission of regulatory reports, ensuring compliance.

### Finance

Agentic [AI improves financial services](https://www.multimodal.dev/banking) by providing real-time insights, risk assessments, and automation in critical areas:

- **Fraud detection and prevention** – AI Agents work together to analyze transaction patterns, flag suspicious activity, and take preventive actions.
- **Regulatory compliance** – Automated AI-driven audits ensure adherence to evolving financial regulations, reducing human error.
- **Loan origination and scoring** – AI Agents assess borrower profiles, analyze alternative data sources and make real-time lending decisions.
- **Financial document verification** – AI verifies financial documents, such as tax returns, loan applications, and financial statements, ensuring accuracy and reducing risks in the approval process.

### Enterprise Automation

In large organizations, Agentic AI systems [streamline complex workflows](https://www.multimodal.dev/) such as:

- **Document processing and management** – AI extracts, classifies, and [organizes unstructured documents](https://www.multimodal.dev/post/how-to-convert-unstructured-data-to-structured-data) to streamline workflows.
- **Internal workflow automation** – AI solutions streamline repetitive internal tasks, such as data entry, report generation, and administrative work, freeing up employees for strategic tasks.
- **Customer service and support automation** – AI-driven Agents handle inquiries, escalate complex tasks and issues, and personalize customer interactions.
- **Compliance monitoring** – AI continuously tracks regulatory changes and ensures that internal processes and documentation comply with current laws and standards.
- **Strategic decision support** – AI provides insights from multiple data sources, assisting leadership in making informed, data-driven decisions with long-term business goals in mind.

By integrating Agentic AI systems, businesses across industries can improve efficiency, lower operational costs, improve decision-making, improve customer experience, and more.

We believe that as AI technology advances, [Agentic AI](https://www.multimodal.dev/post/ai-agentic-workflows) will continue to redefine automation, creating more adaptive and self-improving environments.

## Best Practices for Implementing Agentic AI Systems

To successfully implement an Agentic AI system into your business, here’s a step-by-step process and **key factors to consider**:

![How to implement Agentic AI systems](https://cdn.prod.website-files.com/636e9a9a8d334e3450b08cc9/67ca21a7b6ea2e8a127d12c6_Best-Practices-for-Implementing-Agentic-AI-Systems.webp)

### Define Clear Goals

Before [implementing an Agentic AI system](https://www.multimodal.dev/post/ai-agentic-workflows), it’s important to establish clear and measurable objectives.

These goals will help you guide the system’s development and performance evaluation framework.

**Questions to consider**:

- What problems should the AI system solve?
- How will it improve operational efficiency?
- What metrics will determine its success?

We believe that clear goals ensure AI systems align with your business priorities and drive meaningful outcomes.

### Integrate with Business Workflows

A successful Agentic AI system should [integrate smoothly](https://www.multimodal.dev/post/how-to-implement-ai-in-business) with existing business processes.

Assess your workflows and identify areas where automation or decision-making improvement could add the most value.

‍ **Questions to consider**:

- Will the AI interact with the customer service system?
- Can it streamline data processing in your CRM or ERP?

Ensure that systems work in sync with other tools and platforms. This helps prevent disruption while improving efficiency.

### Use Specialist Agents

To achieve maximum impact, assign specialized AI Agents to handle distinct tasks.

Each AI Agent should be designed to leverage its strengths for specific functions, such as:

- A chatbot agent for customer inquiries
- A data processing agent to convert unstructured data into actionable insights (e.g., [Unstructured AI](http://multimodal.dev/unstructured-ai))
- A decision-making Agent to recommend financial investments or risk management strategies (e.g., [Decision AI](https://www.multimodal.dev/decision-ai))

Specialist AI Agents provide targeted expertise, which leads to better results and system performance. If you have a look at our AI Agents, you’ll notice that each one of our AI Agents is designed for a specific function within the insurance and finance industries.

Orchestrating them together into an Agentic AI system will unlock the true value as they all collaborate together.

### Orchestrate for Maximum Impact

While specialized AI Agents excel at individual tasks, their true value is unlocked when they collaborate within a structured framework.

Design a coordination system where Agents can interact seamlessly, share information, and work together toward common goals.

This can include:

- Using a central control system to manage Agent interactions.
- Allowing Agents to communicate in real-time to optimize outcomes.
- Ensuring data flows freely between Agents for a more unified decision-making process.

That’s exactly what [our Agentic AI platform](https://www.multimodal.dev/post/introducing-agentflow), AgentFlow, provides: a place where you can create, orchestrate, monitor, and manage AI Agents as part of an AI system.

### Keep Humans in the Loop

Agentic AI operates autonomously, but human oversight is critical for handling exceptional situations and ensuring ethical decision-making.

Maintaining humans in the loop means:

- Establishing checkpoints where human experts can intervene if necessary.
- Using human judgment to validate AI-driven recommendations or actions.
- Monitoring for potential biases in AI-driven decision-making.

A human intervention can help maintain accountability, transparency, and the ability to correct any unforeseen issues.

### Monitor and Optimize Continuously

AI systems are not static, and they require ongoing evaluation and improvement.

Regularly assess system performance to identify bottlenecks or areas for refinement. We recommend some of the following practices:

- Setting up monitoring dashboards to track key performance indicators (KPIs).
- Analyzing interactions between Agents to detect inefficiencies and inaccuracies.
- Continuously retraining the system based on new data to improve its learning capabilities.

Optimizing the artificial intelligence system regularly ensures it continues to evolve and provide better value over time. That’s one of the reasons we feature a monitoring system in our AgentFlow platform, which helps businesses clearly oversee the performance of their Agentic AI systems.

## In Short, Agentic AI Systems Mimic Human Teams

![How Agentic AI systems work](https://cdn.prod.website-files.com/636e9a9a8d334e3450b08cc9/67ca22074b11ba1fbd02b3d2_Agentic-AI-Systems-Mimic-Human-Teams.webp)

Agentic AI systems function like human teams–where specialized members work together, share information, and adapt to evolving challenges.

Just as businesses wouldn’t rely on a single employee for every task, AI should not be limited to standalone Agents. Instead, true AI-driven transformation comes from orchestrated [multi-agent systems](https://www.multimodal.dev/post/what-are-multi-agent-systems-in-ai) replicating human collaboration and efficiency.

### Original URL
https://www.multimodal.dev/post/agentic-ai-systems
</details>

---
<details>
<summary>A Complete Guide to AI Agent Architecture in 2025 | Lindy</summary>

### Table of contents

[What is AI agent architecture?](https://www.lindy.ai/blog/ai-agent-architecture#what-is-ai-agent-architecture)

[The components of an AI agent](https://www.lindy.ai/blog/ai-agent-architecture#the-components-of-an-ai-agent)

[3 foundational AI agent architecture models](https://www.lindy.ai/blog/ai-agent-architecture#3-foundational-ai-agent-architecture-models)

[Memory in agent architectures](https://www.lindy.ai/blog/ai-agent-architecture#memory-in-agent-architectures)

[Planning & decision-making layers](https://www.lindy.ai/blog/ai-agent-architecture#planning-and-decision-making-layers)

[Tools & action execution](https://www.lindy.ai/blog/ai-agent-architecture#tools-and-action-execution)

[The rise of LLM agent architectures](https://www.lindy.ai/blog/ai-agent-architecture#the-rise-of-llm-agent-architectures)

[Agent architecture diagrams](https://www.lindy.ai/blog/ai-agent-architecture#agent-architecture-diagrams)

[How Lindy structures its AI agents](https://www.lindy.ai/blog/ai-agent-architecture#how-lindy-structures-its-ai-agents)

[Frequently asked questions](https://www.lindy.ai/blog/ai-agent-architecture#frequently-asked-questions)

[Let Lindy be your AI-powered automation app](https://www.lindy.ai/blog/ai-agent-architecture#let-lindy-be-your-ai-powered-automation-app)

# A Complete Guide to AI Agent Architecture in 2025

![](https://cdn.prod.website-files.com/63e15df811f9df22b231e58f/67178269ecf4084f2b8752d6_T014EKF8DFF-U014N6FQ1NG-e90b261ca6db-192.jpeg)

Flo Crivello

CEO

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse varius enim in eros.

[Learn more](https://www.lindy.ai/blog/ai-agent-architecture#)

![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/67e6b310384f7a51c4061630_kkV9av8Z_400x400.avif)

Lindy Drope

Written by

![](https://cdn.prod.website-files.com/63e15df811f9df22b231e58f/66edceba325d8d5fa7ade7f2_photo-Lindy-Drope.png)

Lindy Drope

Founding GTM at Lindy

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse varius enim in eros.

[Learn more](https://www.lindy.ai/blog/ai-agent-architecture#)

![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/67e6bf7575f1b905c72008ab_kuasBCIY_400x400.avif)

Flo Crivello

Reviewed by

Last updated:

June 16, 2025

Expert Verified

[Blog](https://www.lindy.ai/blog)

[Agents](https://www.lindy.ai/topics/agents)

A Complete Guide to AI Agent Architecture in 2025

AI agents are systems that can decide what to do, how to do it, and when to adapt based on the prompts and constraints you define. This ability comes from the architecture of the agent.

From memory and planning to action and feedback, how an agent is structured determines how useful it actually is in the real world

**In this article, we’ll cover:**

- What is an AI agent architecture?
- What are its components?
- The major architecture models –– reactive, deliberative, and hybrid
- How LLMs changed the way we design agents
- Examples to illustrate these concepts

Let’s begin with the definition of AI agent architecture.

## What is AI agent architecture?

**AI agent architecture refers to the internal structure of** [**AI agents**](https://www.lindy.ai/blog/ai-agents) that allows them to observe, think, act, and learn in a continuous loop. It defines how an agent handles inputs, processes memory, decides what to do, executes actions, and improves over time.

This structure **directly impacts how well an agent can operate** in dynamic environments. Whether it’s helping manage follow-up emails, scheduling meetings, or updating a CRM, the underlying architecture determines how well an agent can adapt and scale.

AI agent frameworksuse **systems that are modular and memory-driven to resemble real-world cognition**. They recall past context, weigh options, and decide the best action based on current and historical data.

### AI Agents vs AI models vs systems

Many people confuse agents, models, and systems — but each plays a different role. **Here’s a quick comparison to clear it up:**

| Term | What it is |
| --- | --- |
| AI model | A statistical tool trained to generate or classify (e.g., GPT-4, Claude) |
| AI Agent | A task-oriented wrapper around models that plans, acts, and adapts |
| System | The broader setup, including models, agents, infrastructure, databases, apps, and more |

Agents sit between the model and the full system. They use models as reasoning engines, but layer on memory, planning, and action execution.

### Why this structure matters

Architecture becomes even more important when you’re building for business environments. **Scaling workflows, maintaining context across sessions, and acting across tools** all require a durable and modular structure for [AI agents](https://www.lindy.ai/blog/ai-agents-examples) to function without hiccups.

Without it, agents may struggle when inputs shift or data is incomplete. Now that we know about the agent architecture, let’s see the components that make an AI agent.

## The components of an AI agent

Every AI agent relies on a few core parts that help it think, plan, and take action. These AI agent architecture components form the basis for how agents operate across different tasks.

These components work together in a loop –– the agent receives input, recalls context, plans an action, executes it, and learns from the outcome.

**Here’s what that looks like broken down:**

### 1\. Perception/input

The agent receives a trigger — like a new form submission, a Slack message, an incoming email, or an API call. In most business workflows, this is what kicks off the agent’s entire loop.

### 2\. Memory

The agent memory architecture includes two layers. **They are:**

- **Working memory:** Short-term context, like a live conversation or active session
- **Persistent memory:** Long-term recall powered by vector databases that helps agents remember previous interactions, user preferences, or task history

### 3\. Planning module

This is where the agent maps the goals to actions and decides what to do next based on context and available tools. Some use rule-based flows, and others use chain-of-thought logic.

### 4\. Execution layer

Once the plan is in place, the agent connects to external tools — CRMs, calendars, email, Slack, and APIs — and performs the required steps.

### 5\. Feedback loop

After execution, the agent checks if the task succeeded. If not, it might retry, flag a human, or adjust the next step. This loop helps agents to be adaptable rather than just reactive.

Next, we understand the 3 foundational AI agent architecture models.

## 3 foundational AI agent architecture models

Three core models define the [AI agent](https://www.lindy.ai/blog/how-do-ai-agents-work) architectures today –– **reactive, deliberative, and hybrid.** Each of these models handles perception, memory, and planning differently.

Understanding how they work helps you choose the right one depending on the complexity of the task.

| Model | How it works | Strengths | Limitations |
| --- | --- | --- | --- |
| Reactive | Responds immediately to inputs. No memory or planning. | Fast, simple | Can’t adapt or recall context |
| Deliberative | Build a world model. Plans before taking action. | Strategic, context-aware | Slower, heavier on compute |
| Hybrid | Combines reactive speed with deliberative planning. | Balanced, flexible | More complex to implement |

### What these models look like today

With the rise of LLM agent architecture, LLMs like GPT-4 enable hybrid behaviors almost by default.

**A reactive agent can now query past context**, while a deliberative agent can adjust its plan mid-task. This flexibility is what makes hybrid agents ideal for business workflows — where agents may need to respond instantly but still consider long-term goals or memory.

**For example,** an agent that responds to a customer inquiry while also tracking account history is no longer purely reactive. It’s using memory, planning, and inputs — capabilities of a hybrid model.

But what about memory? Let’s see how memory works in these architectures.

## Memory in agent architectures

Without memory, an agent is just reacting to inputs in isolation. But with memory, especially persistent memory, **an agent can recall context, past actions, and user preferences**. That’s what makes it useful in real-world workflows.

**AI agent memory architecture includes two types of memory:**

- **Working memory** is session-specific. It stores temporary information, like a chat conversation, user query, or active task state.

‍
- **Persistent memory** is long-term. It survives across sessions and maintains continuity. For example, a support agent may remember past tickets and a sales assistant may know the last conversation with a lead.

Most advanced agents combine both. During a task, they use working memory to stay context-aware, and persistent memory to bring in relevant historical data.

### Vector databases and retrieval

To implement persistent memory, **agents store information as embeddings in a vector database.** When needed, they query the database to find relevant data using semantic similarity, not exact keywords. This is how they remember even loosely related contexts.

### Memory frameworks in action

Frameworks like LangChain **offer modules to manage memory and retrieval**. But where many stop at single-agent memory, some platforms go further.

Lindy’s Societies— **where groups of agents can collaborate** — share memory across tasks. One agent can pull in what another learned, enabling multi-step workflows like “summarize the meeting → write follow-up → update CRM” without data loss.

### Why this matters for businesses

In a business context, memory is what **allows agents to behave consistently**, follow up accurately, and represent your brand without starting from scratch every time.

With memory now out of the way, let’s move to planning and decision-making layers of the architecture.

## Planning & decision-making layers

Once an agent understands its input and recalls relevant context, it needs to decide what to do. That’s where the planning layer comes in.

Planning connects intent to action. **Without it, agents either act blindly or follow rigid scripts**. With planning, they can sequence tasks, adapt to edge cases, and adjust their behavior mid-flow.

This layer is critical in any AI agent framework design. **It can be executed in two ways:**

- **Rule-based planning** relies on if-then logic. It’s easy to set up but breaks when variables shift or data is missing.
- **Dynamic planning** uses models like GPT-4 to reason through tasks step by step  — also known as **chain-of-thought reasoning**.

Agents that use dynamic planning can choose between multiple paths, decide when to ask for help, or even pause execution until conditions are met.

### Some planning framework examples

There are a few well-known approaches here. **Let’s look at them:**

- **LangGraph:** Treats planning as a state machine with memory access baked in. Ideal for structured workflows that may have forks or retries.
- **ReAct:** Blends reasoning with action by letting the agent “think out loud” before choosing what to do.
- **CrewAI:** Uses a multi-agent format where each agent has a defined role. Coordination and delegation are key.

### Why flexible planning matters

Conditions change all the time in business workflows. Meetings get rescheduled. Leads go cold. Data gets updated. A planning module can adapt to these changes based on logic and context.

Let’s now move to the next layer, the execution layer.

## Tools & action execution

Once the agent knows what to do, it needs to do it. That’s the job of the execution layer. **This is where agents connect to tools** — calendars, CRMs, databases, email platforms — and perform tasks based on their plans.

A well-built execution layer is what separates a clever chatbot from a useful worker.

### How agents use tools

**Most agents today interact with tools through native integrations, APIs, or webhooks**. That could mean scheduling a meeting via Google Calendar, updating a record in Salesforce, or sending a follow-up email.

Some platforms allow agents to string these tools together across tasks.

### Why this layer matters

A true agent needs to execute. **That means it must:**

- Understand what tools are available
- Choose the right one based on the task context
- Handle errors (e.g., rescheduling, failed API calls)
- Deliver a result, not just a draft

Without this, all the planning in the world is just talk. **For example, a sales assistant agent:**

1. Parses the call notes from a new lead
2. Composes a personalized follow-up email
3. Sends it to the lead
4. Schedules a meeting based on the reply
5. Updates the CRM entry with all relevant information

An AI agent will execute all these steps without human input.

Now, we know AI agent architecture and how it works. LLMs affect these AI agents hugely. Let’s see how.

‍

Build AI Agents in Minutes - Start Free

[![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/67a483aae734dfc3da312468_InboxArchiveTrayShelfIcon.svg)\\
\\
Meeting Coach\\
\\
Get personalized meeting tips with AI-driven insights.\\
\\
This is some text inside of a div block.\\
\\
This is some text inside of a div block.](https://www.lindy.ai/blog/ai-agent-architecture#) [![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/67a4832e5c29fdc819647984_TItT-CpQo_bsUn7E9kxrmijqBaklfymjNJ_cm-0wi58.svg)\\
\\
![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/67a48342876c667f2cbbe645_UQcH2lJJUfYDJR9ZNFE4asL9s2BZL-tJd9AjNkuc2jI.svg)\\
\\
Try It\\
\\
![](https://cdn.prod.website-files.com/63e15df811f9df22b231e58f/66aa60429339e87dea1c1df0_arrow-top.png)](https://lindy.ai/signup?agent_definition_ids=66a6957734e4012acb21d140&utm_source=lindy&utm_medium=template&utm_campaign=66a6957734e4012acb21d140)

[![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/67a483acdf950c551bcc8674_TapeIcon.svg)\\
\\
Meeting Notetaker\\
\\
Records meeting info and delivers summaries to Slack.\\
\\
This is some text inside of a div block.\\
\\
This is some text inside of a div block.](https://www.lindy.ai/blog/ai-agent-architecture#) [![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/681157328054c6146fb4199c_0P0N6oVfohkLllHT8lSDq5AZM3Su3ilpFB9c39eOym4.webp)\\
\\
![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/67a48317a1931b9b1d010b0a_5cgyZJOwRr0Q5108em7Jzzd1dlYrH-pMQDeC0RMgW2k.svg)\\
\\
Try It\\
\\
![](https://cdn.prod.website-files.com/63e15df811f9df22b231e58f/66aa60429339e87dea1c1df0_arrow-top.png)](https://lindy.ai/signup?agent_definition_ids=67044439e0f3c2fa25fdb386&utm_source=lindy&utm_medium=template&utm_campaign=67044439e0f3c2fa25fdb386)

[![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/683167a55d722a48c803d370_RainbowCloudIcon.svg)\\
\\
AI Meeting Notes Assistant\\
\\
This is some text inside of a div block.\\
\\
This is some text inside of a div block.](https://www.lindy.ai/blog/ai-agent-architecture#) [![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/67a483437c4071be40ced7bc_AXNP6Xd2V-QB9-BlWPSH_0B7HNnwgAIH5ekWr2yOChk.svg)\\
\\
![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/67a48342876c667f2cbbe645_UQcH2lJJUfYDJR9ZNFE4asL9s2BZL-tJd9AjNkuc2jI.svg)\\
\\
Try It\\
\\
![](https://cdn.prod.website-files.com/63e15df811f9df22b231e58f/66aa60429339e87dea1c1df0_arrow-top.png)](https://lindy.ai/signup?agent_definition_ids=6794bf9317abce0c6a0be999&utm_source=lindy&utm_medium=template&utm_campaign=6794bf9317abce0c6a0be999)

[![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/67a483818f6beca19865dd35_AiThreeStarsSparklesIcon.svg)\\
\\
Sales Coach\\
\\
Your sales specialist analyzes calls and provides real-time feedback.\\
\\
This is some text inside of a div block.\\
\\
This is some text inside of a div block.](https://www.lindy.ai/blog/ai-agent-architecture#) [![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/67a4832e5c29fdc819647984_TItT-CpQo_bsUn7E9kxrmijqBaklfymjNJ_cm-0wi58.svg)\\
\\
![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/67a48342876c667f2cbbe645_UQcH2lJJUfYDJR9ZNFE4asL9s2BZL-tJd9AjNkuc2jI.svg)\\
\\
Try It\\
\\
![](https://cdn.prod.website-files.com/63e15df811f9df22b231e58f/66aa60429339e87dea1c1df0_arrow-top.png)](https://lindy.ai/signup?agent_definition_ids=67250746604a192d02a820bc&utm_source=lindy&utm_medium=template&utm_campaign=67250746604a192d02a820bc)

[![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/67a483c7e6d74731c04ab46a_VoiceIcon.svg)\\
\\
Sales Meeting Recorder\\
\\
Highlight critical sales call info to close more deals.\\
\\
This is some text inside of a div block.\\
\\
This is some text inside of a div block.](https://www.lindy.ai/blog/ai-agent-architecture#) [![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/681157328054c6146fb4199c_0P0N6oVfohkLllHT8lSDq5AZM3Su3ilpFB9c39eOym4.webp)\\
\\
![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/67a48317a1931b9b1d010b0a_5cgyZJOwRr0Q5108em7Jzzd1dlYrH-pMQDeC0RMgW2k.svg)\\
\\
Try It\\
\\
![](https://cdn.prod.website-files.com/63e15df811f9df22b231e58f/66aa60429339e87dea1c1df0_arrow-top.png)](https://lindy.ai/signup?agent_definition_ids=6682f02c9e3c8a5da9bdf21c&utm_source=lindy&utm_medium=template&utm_campaign=6682f02c9e3c8a5da9bdf21c)

[![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/67a4838c6921354a8739d719_EmailEnvelopeIcon.svg)\\
\\
Lead Outreacher\\
\\
Have your AI agent perform multi-touch, personalized outreach and update you via Slack.\\
\\
This is some text inside of a div block.\\
\\
This is some text inside of a div block.](https://www.lindy.ai/blog/ai-agent-architecture#) [![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/67a4833215f2310377ddfcde_kljepJaptXLhDHGzyBdmzj-NYkUwMALRBmQ1yt3puIY.svg)\\
\\
![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/67a48362587e3eee1dc520b1_5W0LoSeC4RJ2D6IIwTHpFPWZ7QMFz9j4Kj58AHesFeI.svg)\\
\\
Try It\\
\\
![](https://cdn.prod.website-files.com/63e15df811f9df22b231e58f/66aa60429339e87dea1c1df0_arrow-top.png)](https://lindy.ai/signup?agent_definition_ids=66a8214b07a1e57a25df7de4&utm_source=lindy&utm_medium=template&utm_campaign=66a8214b07a1e57a25df7de4)

[![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/67a4838ee734dfc3da310355_CloseXCircleRemoveIcon.svg)\\
\\
Follow-up Email Drafter\\
\\
Detects important emails and drafts follow-ups, reminding if no response.\\
\\
This is some text inside of a div block.\\
\\
This is some text inside of a div block.](https://www.lindy.ai/blog/ai-agent-architecture#) [![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/681157328054c6146fb4199c_0P0N6oVfohkLllHT8lSDq5AZM3Su3ilpFB9c39eOym4.webp)\\
\\
![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/67a4833db4160127431bc8a6_36OSAIxIelszqyZ77ifMlXCc-Q3qLimYL6fLgmFWGQU.svg)\\
\\
Try It\\
\\
![](https://cdn.prod.website-files.com/63e15df811f9df22b231e58f/66aa60429339e87dea1c1df0_arrow-top.png)](https://chat.lindy.ai/marketplace?templateId=66d7b9f3c5374a4cd492437f&internal_origin=blog%2Fai-agent-architecture)

‍

## The rise of LLM agent architectures

Large Language Models improved natural language understanding. They allow AI agents to handle workflows with dynamic reasoning. This shift gave rise to a new wave of **LLM agent architecture**.

### How foundation models changed agent design

Before LLMs, agents were limited by their design. They needed hardcoded rules, fixed memory scopes, and limited tool access. **Now, models like GPT-4.5 or Claude Opus 4 allow agents to:**

- Interpret ambiguous instructions
- Generate task sequences on the fly
- Adjust behavior mid-conversation
- Communicate more naturally across interfaces

Each of these contributes to a growing ecosystem, but most still require technical know-how to implement.

### Where structured agents stand out

Some agents are built to reason and do structured work. That includes **multi-step coordination, shared memory, and deep integrations**— features most experimental agents still lack.

This is where business-ready platforms focus –– building agents that execute cleanly, adapt across sessions, and integrate directly into day-to-day workflows.

Next, let’s understand these architectures with a flow chart.

## Agent architecture diagrams

Sometimes, the easiest way to understand AI agent architecture components is to see it. **Below are two simplified flowcharts that capture how AI agents typically operate:**

### Classic agent loop

This is the traditional format used in robotics and early AI –– **Perceive → Decide → Act → Learn.**

The agent receives input, processes a decision, takes an action, and uses the result as feedback. It’s linear and often rigid—useful for basic automation but limited in flexibility.

### Modern loop in AI agents

AI agent architecture diagrams today are more modular and built for adaptability. **It follows the flow of Trigger → Plan → Tools → Memory → Output:**

- A trigger starts the process (form submission, message, etc.)
- The agent builds a plan based on the context
- It selects tools to act (email, API, calendar, etc.)
- Pulls relevant memory to personalize the task
- Then produces an output or action

This loop allows agents to adjust, replan, or escalate based on outcomes.

With all the information about AI agents, their architecture, LLMs, and more, let’s focus on Lindy and how its AI agents are structured.

## How Lindy structures its AI agents

![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/684feb72eccfb1a99bbefc7e_Lindy_Imagery1.png)

Lindy approaches AI agent architecture by focusing on structure, modularity, and real-world use from day one.

### A goal-first architecture

Every agent in Lindy **focuses on a clear job to be done**. Whether that’s screening a lead, scheduling a call, or triaging an inbox — the architecture starts with the end goal.

### Persistent memory and agent coordination

Lindy combines persistent memory (stored via embeddings in a vector database) with working memory (what’s currently in context). **Agents can pull in previous interactions, user preferences, and outcomes** from earlier tasks.

These agents collaborate thanks to Lindy’s multi-agent coordination. **One agent might handle intake, another parses a document, and a third updates your CRM**. This kind of coordination isn’t possible unless explicitly designed for multi-agent flows.

### Deep integration with business tools

Instead of relying on plug-ins or workarounds, Lindy offers [**7,000+ integrations**](https://www.lindy.ai/integrations)–– Slack, Gmail, Salesforce, Airtable, Notion, voice platforms, and more –– via Pipedream partnership, APIs, and native connectors.

### Here’s a real-world example

Let’s look at an example to understand Lindy better. **Here’s what a multi-agent flow can look like:**

- A user receives a **meeting invite**.
- A calendar agent **parses the event** and logs it.
- A second agent generates a **follow-up summary**.
- A third **updates the CRM** with the next steps.
- All agents **share memory** and complete the flow autonomously.

This is just one example of how customers use Lindy for their workflows.

‍

Build AI Agents in Minutes - Start Free

Get things done 10x cheaper, 10x faster with your custom AI agent — no coding required.

[Try Lindy for Free](https://chat.lindy.ai/signup?utm_source=blog%2Fai-agent-architecture&internal_origin=blog%2Fai-agent-architecture) Try Lindy for Free

Try Lindy for Free

![](https://cdn.prod.website-files.com/63e15df811f9df22b231e58f/67f75e7a45a6ef7a9754718a_671b58451af041b86ef1c56b_lindy_hero_bg.avif)

‍

## Frequently asked questions

### How do AI agents use memory?

**They store data as vector embeddings** **and retrieve it based on semantic similarity.** Working memory holds current task data, while persistent memory recalls historical context across sessions. This dual system forms the base of a reliable agent memory architecture.

### What’s the difference between reactive and hybrid agents?

Reactive agents act only on current inputs. **Hybrid agents use both immediate input and long-term context** to decide and adapt. Most business-use agents today follow a hybrid model.

### Is LangChain an agent framework or a memory tool?

LangChain began as **a framework for chaining LLM calls together** — allowing developers to build more complex, multi-step interactions. Over time, **it has expanded to support full agent design**, including components for memory, planning, and tool execution.

### Can I build an AI agent without code?

Yes, you can. **Platforms like Lindy support no-code creation** via templates, natural language instructions, and drag-and-drop flow design.

### What does “agentic behavior” actually mean?

It refers to an **agent’s ability to operate with autonomy** — set goals, plan, act, and learn from feedback — without needing constant human input.

### Do I need a planning module for all AI agents?

No, you do not always need one for simple reactive tasks. But if you **need any agent to adapt, handle uncertainty, or sequence multiple steps**, planning becomes essential.

### What’s the best architecture for business automation?

**A hybrid model with persistent memory, dynamic planning, and real-time execution** across tools is one of the best architectures for business automation. Reliable setups like Lindy often prioritize modularity, integrations, and recovery flows over traditional autonomy.

## Let Lindy be your AI-powered automation app

If you want affordable AI automations, go with Lindy. It’s an intuitive AI automation platform that lets you build your own AI agents for loads of tasks.

You’ll find plenty of [pre-built templates](https://www.lindy.ai/template-categories/others) and loads of [integrations](https://www.lindy.ai/integrations) to choose from.

**Here’s why Lindy is an ideal option:**

- [**AI Meeting Note Taker**](https://www.lindy.ai/solutions/meetings) **:** Lindy can join meetings based on Google Calendar events, record and transcribe conversations, and generate structured meeting notes in [Google Docs](https://www.lindy.ai/integrations/google-docs). After the meeting, Lindy can send Slack or email summaries with action items and can even trigger follow-up workflows across apps like HubSpot and Gmail.

‍ [**‍**](https://www.lindy.ai/templates/sales-coach)
- [**Sales Coach**](https://www.lindy.ai/templates/sales-coach) **:** Lindy can provide custom coaching feedback, breaking down conversations using the [MEDDPICC framework](https://meddpicc.net/understanding-the-meddpicc-sales-framework/) to identify key deal factors like decision criteria, objections, and pain points​.

‍ **‍**
- **Automated CRM updates:** Instead of just logging a transcript, you can set up Lindy to update CRM fields and fill in missing data in [Salesforce](https://www.lindy.ai/integrations/salesforce) and [HubSpot](https://www.lindy.ai/integrations/hubspot) — without manual input​.

‍ **‍**
- **AI-powered follow-ups:** Lindy agents can [send follow-up emails](https://www.lindy.ai/templates/follow-up-email-drafter), [schedule meetings](https://www.lindy.ai/academy-lessons/meeting-scheduler-101), and keep everyone in the loop by triggering notifications in Slack by letting you build a [Slackbot](https://www.lindy.ai/academy-lessons/slackbot).

‍ [**‍**](https://www.lindy.ai/academy-lessons/lead-enrichment)
- [**Lead enrichment**](https://www.lindy.ai/academy-lessons/lead-enrichment) **:** Lindy can be configured to use a prospecting API ( [People Data Labs](https://www.lindy.ai/integrations/people-data-labs)) to research prospects and to provide sales teams with richer insights before outreach.

‍ [**‍**](https://www.lindy.ai/academy-lessons/outreach-101)
- [**Automated sales outreach**](https://www.lindy.ai/academy-lessons/outreach-101) **:** Lindy can run multi-touch email campaigns, follow up on leads, and even draft responses based on engagement signals​.

eeting **‍**
- **Cost-effective:** Automate up to 400 monthly tasks withLindy’s free version. The paid version lets you automate up to 5,000 tasks per month, which is a more affordable price per automation compared to many other platforms.

[**Try Lindy for free**](https://chat.lindy.ai/signup?utm_source=blog%2Fhow-to-make-an-ai-free&internal_origin=blog) **.**

Share this post

[Tag one](https://www.lindy.ai/blog/ai-agent-architecture#) [Tag two](https://www.lindy.ai/blog/ai-agent-architecture#) [Tag three](https://www.lindy.ai/blog/ai-agent-architecture#) [Tag four](https://www.lindy.ai/blog/ai-agent-architecture#)

About the editorial team

![](https://cdn.prod.website-files.com/63e15df811f9df22b231e58f/67178269ecf4084f2b8752d6_T014EKF8DFF-U014N6FQ1NG-e90b261ca6db-192.jpeg)

Flo Crivello

Founder and CEO of Lindy

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse varius enim in eros elementum tristique. Duis cursus, mi quis viverra ornare, eros dolor interdum nulla, ut commodo diam libero vitae erat. Aenean faucibus nibh et justo cursus id rutrum lorem imperdiet. Nunc ut sem vitae risus tristique posuere.

**Education:** Master of Arts/Science, Supinfo International University

**Previous Experience:** Founded Teamflow, a virtual office, and prior to that used to work as a PM at Uber, where he joined in 2015.

![](https://cdn.prod.website-files.com/63e15df811f9df22b231e58f/66edceba325d8d5fa7ade7f2_photo-Lindy-Drope.png)

Lindy Drope

Founding GTM at Lindy

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse varius enim in eros elementum tristique. Duis cursus, mi quis viverra ornare, eros dolor interdum nulla, ut commodo diam libero vitae erat. Aenean faucibus nibh et justo cursus id rutrum lorem imperdiet. Nunc ut sem vitae risus tristique posuere.

**Education:** Master of Arts/Science, Supinfo International University

**Previous Experience:** Founded Teamflow, a virtual office, and prior to that used to work as a PM at Uber, where he joined in 2015.

![](https://cdn.prod.website-files.com/63e15df811f9df22b231e58f/6717a30104dab6217671eb32_Frame%2037242.avif)

Build AI Agents in Minutes

Automate workflows, save time, and grow your business - No Coding Required

Intelligent AI Agents which can think, decide and act

Automate any workflow, from phone calls to complex sales research

Integrate with 1,500+ tools

[Build an AI Agent Free Now](https://chat.lindy.ai/signup?utm_source=blog%2Fai-agent-architecture&internal_origin=blog%2Fai-agent-architecture) Build an AI Agent Free Now

Build an AI Agent Free Now

[Talk to Sales](https://calendly.com/d/cryz-4v7-sk8) Talk to Sales

Talk to Sales

![](https://cdn.prod.website-files.com/63e15df811f9df22b231e58f/671b58421af041b86ef1c32d_works_eyebrow.svg)

Blog

## Related Articles

[See all Articles](https://www.lindy.ai/blog) See all Articles

See all Articles

[![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/6854184cec7d443771316482_Lindy_Thumbnail1%20(8)%20(1).png)\\
\\
AI in Customer Support\\
\\
**What Is In-Product Messaging? Types, Examples & How to Do it**\\
\\
Lindy Drope\\
\\
5\\
\\
min](https://www.lindy.ai/blog/in-product-messaging)

[![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/685417d02cd6908dd8737ba1_Lindy_Thumbnail1%20(7)%20(1).png)\\
\\
AI in Sales\\
\\
**What Is a Lead List? How to Make One Using AI**\\
\\
Lindy Drope\\
\\
5\\
\\
min](https://www.lindy.ai/blog/lead-lists)

[![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/685417080d4683dabfd4d80a_Lindy_Thumbnail1%20(6)%20(1).png)\\
\\
AI Tools\\
\\
**How to Humanize AI Content (without Sounding Fake)**\\
\\
Lindy Drope\\
\\
5\\
\\
min](https://www.lindy.ai/blog/how-to-humanize-ai-content)

[![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/685415746d873c1349cd08fb_Lindy_Thumbnail1%20(5)%20(1).png)\\
\\
Knowledge Management\\
\\
**AI-Powered Workplace Search & How to Use It**\\
\\
Lindy Drope\\
\\
5\\
\\
min](https://www.lindy.ai/blog/workplace-search)

[![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/685414bf2cd6908dd871880c_Lindy_Thumbnail1%20(4)%20(1).png)\\
\\
AI Tools\\
\\
**How to Write a Reference Letter (with Examples, Tips & Template)**\\
\\
Lindy Drope\\
\\
5\\
\\
min](https://www.lindy.ai/blog/how-to-write-a-reference-templates-inside)

[![](https://cdn.prod.website-files.com/6418bbd648ad4081cf60eb29/68219dd5e901b2d5957ad118_Lindy_Thumbnail1-4.png)\\
\\
AI Tools\\
\\
**How to Use AI at Work: 15 Ways to 10x Productivity**\\
\\
Lindy Drope\\
\\
5\\
\\
min](https://www.lindy.ai/blog/how-to-use-ai-at-work)

Automate with AI

## Start for free today.

Build AI agents in minutes to automate workflows, save time, and grow your business.

400 Free credits

400 Free tasks

[Automate your first task](https://chat.lindy.ai/signup?utm_source=blog%2Fai-agent-architecture&internal_origin=blog%2Fai-agent-architecture)

![](https://cdn.prod.website-files.com/63e15df811f9df22b231e58f/67f75e7a45a6ef7a9754718a_671b58451af041b86ef1c56b_lindy_hero_bg.avif)

[![](https://cdn.prod.website-files.com/63e15df811f9df22b231e58f/671b58431af041b86ef1c39a_g_square_icon_x.svg)](https://x.com/getlindy)[![](https://cdn.prod.website-files.com/63e15df811f9df22b231e58f/671b58431af041b86ef1c3a9_g_square_icon_linkedin.svg)](https://fr.linkedin.com/company/lindyai)

![](https://cdn.prod.website-files.com/63e15df811f9df22b231e58f/671b58431af041b86ef1c3a2_SOC2_icon.png)

SOC 2

Compliant

![](https://cdn.prod.website-files.com/63e15df811f9df22b231e58f/671b58431af041b86ef1c3b2_hipaa_icon.svg)

HIPAA

Compliant

![](https://cdn.prod.website-files.com/63e15df811f9df22b231e58f/671b58431af041b86ef1c3b6_pipeda_icon.svg)

PIPEDA

Compliant

Solutions

[Sales](https://www.lindy.ai/solutions/sales) [Email](https://www.lindy.ai/solutions/email) [Customer Support](https://www.lindy.ai/solutions/customer-support) [Meetings](https://www.lindy.ai/solutions/meetings) [Medical Scribe](https://www.lindy.ai/medical-scribe) [All Tools](https://www.lindy.ai/tools)

Resources

[Blog](https://www.lindy.ai/blog) [Academy](https://www.lindy.ai/academy) [Community](https://community.lindy.ai/) [Help Center](https://www.lindy.ai/academy) [Integrations](https://www.lindy.ai/integrations) [Partners](https://www.lindy.ai/partners)

Company

[Careers](https://careers.lindy.ai/) [Changelog](https://www.lindy.ai/changelog) [Security](https://www.lindy.ai/security) [Contact](mailto:hello@lindy.ai)

Legal

[Privacy Policy](https://www.lindy.ai/privacy) [Trust Center](https://app.drata.com/trust/9cb791f0-0c38-11ee-865f-029d78a187d9) [Terms of Service](https://www.lindy.ai/terms-of-service)

Lindy. All rights reserved. ©  2025

Lindy is not associated with Lindy Electronik Gmbh

```
(function() {
    'use strict';
})();

```

![Chat with us](https://lindy.nyc3.cdn.digitaloceanspaces.com/user-content/prod/owners/662702124310da3c254caebd/api-files-image-upload/5dd76dfa-9a85-4ea6-9589-83eca97fa73b.png)

Lindy Chat - Lindy

![Lindy Chat logo](https://chat.lindy.ai/_next/image?url=https%3A%2F%2Flindy.nyc3.cdn.digitaloceanspaces.com%2Fuser-content%2Fprod%2Fowners%2F662702124310da3c254caebd%2Fapi-files-image-upload%2Fb9f7a6a6-a305-496d-aecc-ec677dd2c537.png&w=64&q=75)

## Lindy Chat

[Powered by Lindy](https://lindy.ai/)

![Lindy Help Agent logo](https://chat.lindy.ai/_next/image?url=https%3A%2F%2Flindy.nyc3.cdn.digitaloceanspaces.com%2Fuser-content%2Fprod%2Fowners%2F662702124310da3c254caebd%2Fapi-files-image-upload%2Fb9f7a6a6-a305-496d-aecc-ec677dd2c537.png&w=48&q=75)

Lindy Help Agent

Hey, I'm Lindy :)

How can I help you today?

- What integrations does Lindy support?
- How can I have the Lindy-team build me a custom agent?
- What is a Lindy Business Plan?
- What is a knowledge base?

### Original URL
https://www.lindy.ai/blog/ai-agent-architecture
</details>

