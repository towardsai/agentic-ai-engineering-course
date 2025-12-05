## Outline

1. Introduction: The Critical Decision Every AI Engineer Faces
2. Understanding the Spectrum: From Workflows to Agents
3. Choosing Your Path
4. Conclusion: The Challenges of Every AI Engineer

## Section 1 - Introduction: The Critical Decision Every AI Engineer Faces

- **The Problem:** When building AI applications, engineers face a critical architectural decision early in their development process. Should they create a predictable, step-by-step workflow where they control every action, or should they build an autonomous agent that can think and decide for itself? This is one of the key decisions that will impact everything from the product such as development time and costs to reliability and user experience.
- Quick walkthrough of what we'll learn by the end of this lesson

- **Section length:** 100 words

## Section 2 - Understanding the Spectrum: From Workflows to Agents

- In this section we want to take a brief look at what LLM workflows and AI agents are. At this point we don't focus on the technical specifics of each, but rather on their properties and how they are used.
- On **LLM workflows** we care about:
	- Definition: A sequence of tasks involving LLM calls or other operations such as reading/writing data to a database or file system. It is largely predefined and orchestrated by developer-written code.
	- Characteristics: The steps are defined in advance, resulting in deterministic or rule-based paths with predictable execution and explicit control flow. 
	- Add a Mermaid diagram 
- On **AI agents** we care about:
	- Definition: Systems where an LLM (or multiple LLMs) plays a central role in dynamically deciding (planning) the sequence of steps, reasoning, and actions to achieve a goal. The steps are not defined in advance, but are dynamically planned based on the task and current state of the environment.
	- Characteristics: Adaptive, capable of handling novelty, LLM-driven autonomy in decision-making and execution path.
	- Add a Mermaid diagram
- **Section length:** 200 words (without the mermaid diagram code)

## Section 3: Choosing Your Path

- In the previous section we defined the LLM workflows and AI agents independently, now we want to explore their core differences: Developer-defined logic vs LLM-driven autonomy in reasoning and action selection.
- Attach an image from the research showing the gradient between LLM workflows and AI agents.
- **When to use LLM workflows:**
	- Examples where the structure is well-defined:
		- Pipelines for data extraction and transformation from sources such as the web, messaging tools like Slack, video calls from Zoom, project management tools like Notion, and cloud storage tools like Google Drive
		- Automated report or emails generation from multiple data sources
		- Repetitive daily tasks: Sending emails, posting social media updates, responding to messages
		- Content generation or repurposing, such as transforming articles into social media posts 
	- Strengths: Predictability, reliability for well-defined tasks, easier debugging of fixed paths, potentially lower operational costs as we can leverage simpler and smaller models specialized in given sub-tasks. 
	- Weaknesses: Potentially more development time required as each step is manually engineered. The user experience is rigid as it cannot handle unexpected scenarios. Adding new features can get complex when the application grows, similar to developing standard software tools.
- **When to use AI agents:**
	- Examples: 
		- Open-ended research and synthesis (e.g., researching about WW2)
		- Dynamic problem-solving (e.g., debugging code, complex customer support)
		- Interactive task completion in unfamiliar environments (e.g., booking a flight, where we don't specify the exact sites to use)
	- Strengths: Adaptability to new situations and the flexibility to handle ambiguity and complexity as the steps are dynamically decided.
	- Weaknesses: The system is more prone to errors. As the agent is non-deterministic, the performance, latency and costs can vary with each call of the agent, making agents often unreliable. As agents require LLMs that can generalize better, which are bigger, hence more costly, adopting an agentic solution usually ends up being more costly. AI agents usually require more LLM calls to understand the user intent and take various actions, which can result again in bigger costs per call. If not designed well, there can be huge security concerns, especially on write operations, where it can delete all our data or send inappropriate emails. Ultimately, a huge disadvantage of AI agents is that they are hard to debug and evaluate.
	
- **Hybrid Approaches:** Most real-world systems blend elements of both approaches. Thus, in reality, we have a spectrum, a gradient between LLM workflows and AI agents, where a system adopts what's best from both worlds depending on its use cases.
- Generate a mermaid to illustrate the AI generation and human verification loop

- **Section length:** 200 words

## Section 4 - Conclusion: The Challenges of Every AI Engineer

- **The Reality of AI Engineering:** Now that you understand the spectrum from LLM workflows to AI agents, it's important to recognize that these are one of the core decisions that determine whether your AI application succeeds in production or fails spectacularly.
- To set the scene for future lessons and patterns we will learn, present some daily challenges every AI engineer battles:
	- **Data Integration:** Building pipelines to pull information from Slack, web APIs, SQL databases, and data lakes while ensuring only high-quality data is passed to your AI system (garbage-in, garbage-out principle).
	- **Cost-Performance Trap:** Sophisticated agents deliver impressive results but cost a fortune per user interaction, making them economically unfeasible for many applications.
	- **Security Concerns:** Autonomous agents with powerful write permissions could send wrong emails, delete critical files, or expose sensitive data.

- To transition from this lesson to the next, specify that in the future lesson (lesson 3) we will dig more into the foundations of AI agents and workflows, which is context engineering

- **Section length:** 100 words

## Golden Sources

- [Real Agents vs. Workflows: The Truth Behind AI 'Agents'](https://www.youtube.com/watch?v=kQxr-uOxw2o&t=1s)
- [Exploring the difference between agents and workflows](https://decodingml.substack.com/p/llmops-for-production-agentic-rag)
- [A Developer’s Guide to Building Scalable AI: Workflows vs Agents](https://towardsdatascience.com/a-developers-guide-to-building-scalable-ai-workflows-vs-agents/)
- [Gemini CLI README.md](https://github.com/google-gemini/gemini-cli/blob/main/README.md)
