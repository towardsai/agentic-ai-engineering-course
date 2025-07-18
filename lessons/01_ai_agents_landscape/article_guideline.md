## Global Context

- **What I'm planning to share**: A comprehensive guide to AI Engineering, focusing on the design and implementation of LLM-powered workflows and autonomous agentic systems. This lesson defines the scope of the course, introduces the modern AI Engineering stack, explains the emergence of the AI Engineer role, and sets the foundation for building sophisticated autonomous systems powered by frameworks like LangGraph and advanced reasoning models. We'll explore why simple LLM API calls are insufficient for complex real-world applications - they lack memory, can't use tools, struggle with multi-step reasoning, and fail to adapt to dynamic environments. Instead, the future lies in building autonomous agents that can maintain context across interactions, utilize external tools, reason through complex problems step-by-step, and continuously learn from their experiences to improve task completion rates.

- **Why I think it's valuable**: For an AI Engineer in 2025, understanding the current landscape, their specific role within it, and the trajectory of AI development towards autonomous agents is critical. While simple LLM API calls might work for basic text generation, real-world applications require systems that can handle complex workflows, maintain state, interact with external services, and adapt to changing requirements. This knowledge helps focus learning efforts on high-impact skills, navigating the rapidly evolving toolchain, and preparing for building the next generation of AI-powered applications that can actually solve complex business problems autonomously. This article lays the foundational understanding for the entire course and gets people excited about the transformative potential of agentic systems they'll be building.

- **Who the intended audience is**: Software engineers, ML engineers, and developers aiming to specialize in AI Engineering. It's ideal for those looking to move beyond basic LLM API calls and build sophisticated, autonomous AI applications capable of complex reasoning, planning, and task execution. Participants should be interested in hands-on development with Python, OpenAI models, Gemini models and frameworks like LangGraph to create integrated, production-considerate agentic systems.

- **Expected length of the article in words**: 2000-2500 words (approximately 8-10 minutes of reading time)

## Outline 

1. Section 1: Navigating the Course: Scope and Essential Skills
2. Section 2: Defining the Modern AI Engineer
3. Section 3: The Shift to Autonomous Agentic Systems
4. Section 4: Capstone Preview: The Research + Writing Agent

## Section 1: Navigating the Course: Scope and Essential Skills

- Clearly define the course's focus: Building integrated agentic systems that can reason, plan, and act. (Those things help in task rate completion success for agentic systems)
- Distinguish essential skills for an AI Engineer working with agents:
    - Proficiency with LLM APIs and core concepts (e.g., LLM context optimization, prompting strategies, retrieval augmented generation).
    - Understanding of agentic architectures (e.g., workflows, ReAct, planning, tool use).
    - Experience with agent frameworks (e.g., LangGraph, OpenAI Agents SDK, LlamaIndex, LangChain).
    - Evaluation techniques for agentic systems.
    - Basic LLMOps principles (monitoring, versioning).
    - **Context engineering**: An emerging discipline that goes beyond prompt engineering to include strategic management of conversation history, memory systems, and information retrieval patterns that enable agents to maintain coherent, goal-oriented behavior across extended interactions (reference LlamaIndex blog on context engineering best practices).
- **Production considerations for agentic systems**: Unlike simple LLM API calls, deploying and maintaining agents introduces unique challenges including observability (tracking multi-step agent reasoning), cost management (agents can trigger multiple expensive model calls), latency optimization (balancing speed vs. reasoning quality), and production monitoring using specialized tools like Opik, LangSmith, or LangFuse for agent performance tracking.
- Contrast with "Fear Of Missing Out" (FOMO) topics, such as training foundational models from scratch, fine-tuning LLMs when task success can be achieved through prompt engineering, and deep theoretical research in specific niche areas (unless directly applicable). Explain that while these are interesting, they are not the core focus for *applying* and *building* with current agent technology.
- Emphasize the value of practical application and system integration.
- **Section length:** 650 words

## Section 2: Defining the Modern AI Engineer

- The "What and Why": A short "Preamble" about *why* the AI Engineer role has come to the forefront over the last ~two years. Now, let's explore the stack they operate within.
- Introduce the AI Engineering stack:
    - **Application Layer:** User interface, business logic, agent orchestration, domain specialization, and includes the **evaluation subsystem** for validation of outputs during feature development and post-production deployment with iterative improvements.
    - **Model Layer:** LLM selection (intelligence, cost, latency, model size, open vs closed models, local vs cloud/third party APIs), prompt engineering, fine-tuning (when necessary)
    - **Infrastructure Layer:** Deployment (of the model and the app), scaling, monitoring, data pipelines for LLMs.
- Clearly Differentiate the AI Engineer Role:
    - From ML Engineers: Focus more on the application side (as opposed to just model development), integration, and system design using pre-trained/foundational models versus training models from scratch. **For ML Engineers transitioning to AI Engineering**: A key consideration is the trade-off between open-source models (greater control, customization potential, cost predictability, but requiring more infrastructure management) versus closed-source models (faster iteration, better performance out-of-the-box, built-in safety features, but higher costs and vendor lock-in concerns).
    - From Full-Stack Developers: Deeper knowledge of LLM capabilities (strengths and limitations), prompt engineering, agentic patterns, experience working with probabilistic systems (as opposed to deterministic systems in standard software applications), application of the "scientific approach" (experiments + iterative improvements) to improving AI apps, and AI-specific infrastructure.
- Explain the AI Engineer's primary responsibility: Building, deploying, ensuring good output quality, and maintaining robust, scalable, and integrated AI systems, particularly agentic ones.
- **Section length:** 550 words

## Section 3: The Shift to Autonomous Agentic Systems

- Explain the evolution from simple LLM interactions (e.g., single-turn Q&A, basic text generation) to complete autonomous systems.
- Define "agentic systems": AI systems that can perceive their environment (to some extent), make decisions, take actions, and learn or adapt.
- Introduce the role of advanced reasoning models (e.g., o3/o4-mini, or other contemporary models with strong reasoning capabilities) as the "brains" of these agents, enabling more complex planning and decision-making.
- Briefly introduce agent development frameworks:
    - LangGraph: Explain its utility for building stateful, multi-actor applications with graphs, suitable for debugging complex agent behaviors.
    - OpenAI Agents SDK (or equivalent if the landscape shifts): Highlight its features for creating assistants that can call tools and maintain context.
- Discuss why this shift is transformative for various industries and applications. And how these 'agents' will continue to improve over time.
- **Section length:** 500 words

## Section 4: Capstone Preview: The Research + Writing Agent

- Introduce the course capstone project: a Research + Writing agent.
- Describe its high-level goal: e.g., given a topic, clear guidelines, and specific criteria, the agent should be able to research the topic using web search tools, synthesize information, and draft a coherent article or report.
- Explain how this project will integrate concepts learned throughout the course: planning, tool use, information synthesis, multi-step reasoning, and potentially self-correction.
- **Technology choice rationale**: Explain why this course primarily uses Gemini models for the capstone project: Gemini offers a generous free tier that makes it accessible for learning and experimentation, while providing best-in-class performance that rivals OpenAI models for reasoning tasks. This combination of accessibility and capability makes it ideal for building sophisticated agentic systems without significant upfront costs.
- Set expectations: This project will demonstrate the power of building an integrated agentic system using the frameworks and models discussed.
- **Section length:** 400 words

## Golden Sources

- [LLM Developers vs Software Developers vs ML Engineers](https://www.louisbouchard.ai/llm-developers/)
- [AI Agents in 2025: Expectations vs Reality - IBM](https://www.ibm.com/think/insights/ai-agents-2025-expectations-vs-reality)
- [What is Agentic Reasoning - IBM](https://www.ibm.com/think/topics/agentic-reasoning)
- [What is LangGraph](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/)
- [OpenAI Agents Python SDK](https://openai.github.io/openai-agents-python/)

## Other Sources

- [A Practical Guide to Building Agents - OpenAI](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)
- [How Reasoning AI Agents Transform High-Stakes Decision Making - NVIDIA](https://blogs.nvidia.com/blog/reasoning-ai-agents-decision-making/)
- [What is LangGraph - IBM](https://www.ibm.com/think/topics/langgraph)
- [Context Engineering - LlamaIndex](https://www.llamaindex.ai/blog/context-engineering-what-it-is-and-techniques-to-consider)
- [Gemini models](https://ai.google.dev/gemini-api/docs/models)