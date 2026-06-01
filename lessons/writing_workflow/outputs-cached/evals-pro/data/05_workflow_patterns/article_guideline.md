## Global Context of the Lesson

### What We Are Planning to Share

This article explores the fundamental components for building LLM workflows: chaining multiple LLM calls, parallelizing LLM calls, implementing routing with conditional logic, and the orchestrator-worker pattern. We will explain why breaking down complex tasks into chained calls is often more effective than relying on a single, large LLM call. Practical demonstrations will show how to build a sequential workflow for FAQ generation (Generate Questions → Answer Questions → Find Sources) and a routing workflow for customer service intent classification using the Google Gemini library.

### Why We Think It's Valuable

For an AI Engineer, mastering chaining/routing/orchestration is one of the first step towards constructing sophisticated and reliable LLM applications. These techniques provide modularity, improve accuracy, and allow for more controlled and adaptable processing, forming the building blocks for both deterministic workflows and more complex agentic systems.

### Expected Length of the Lesson

**5000 words** (without the titles and references), where we assume that 200-250 words ≈ 1 minute of reading time.

### Theory / Practice Ratio

20% theory - 80% real-world examples

## Achoring the Lesson in the Course

### Details About the Course

This piece is part of a broader course on AI agents and LLM workflows. The course consists of 3 parts, each with multiple lessons. 

Thus, it's essential to always anchor this piece into the broader course, understanding where the reader is in its journey. You will be careful to consider the following:
- The points of view.
- To not reintroduce concepts already thought in the previous lesson.
- To be careful when talking about concepts introduced only in future lessons.
- To always reference previous and future lessons when discussing topics outside the piece's scope.

### Lesson Scope

This is lesson 5 (from part 1) of the course on AI Agents.

### Point of View

The course is created by a team writing for a single reader, also known as the student. Thus, for voice consistency across the course, we will always use 'we,' 'our,' and 'us' to refer to the team who creates the course, and 'you' or 'your' to address the reader. Avoid singular first person and don't use 'we' to refer to the student.

Examples of correct point of view:
- Instead of "Before we can choose between workflows and agents, we need a clear understanding of what they are." word it as "To choose between workflows and agents, you need a clear understanding of what they are."

### Who Is the Intended Audience

Aspiring AI engineers who are learning about AI workflow patterns for the first time.

### Concepts Introduced in Previous Lessons

In previous lessons of the course, we introduced the following concepts:

**Part 1:**

- **Lesson 1 - AI Engineering & Agent Landscape**: Understanding the role, the stack, and why agents matter now
- **Lesson 2 - Workflows vs. Agents**: Grasping the crucial difference between predefined logic and LLM-driven autonomy
- **Lesson 3 - Context Engineering**: The art of managing information flow to LLMs
- **Lesson 4 - Structured Outputs**: Ensuring reliable data extraction from LLM responses

As this is only the 5th lesson of the course, we haven't introduced too many concepts. At this point, the reader only knows what an LLM is and a few high-level ideas about the LLM workflows and AI agents landscape.

### Concepts That Will Be Introduced in Future Lessons

In future lessons of the course, we will introduce the following concepts:

**Part 1:**

- **Lesson 5 - Basic Workflow Ingredients**: Implementing chaining, routing, parallel and the orchestrator-worker patterns
- **Lesson 6 - Agent Tools & Function Calling**: Giving your LLM the ability to take action
- **Lesson 7 - Planning & Reasoning**: Understanding patterns like ReAct (Reason + Act)
- **Lesson 8 - Implementing ReAct**: Building a reasoning agent from scratch
- **Lesson 9 - Agent Memory & Knowledge**: Short-term vs. long-term memory (procedural, episodic, semantic)
- **Lesson 10 - RAG Deep Dive**: Advanced retrieval techniques for knowledge-augmented agents
- **Lesson 11 - Multimodal Processing**: Working with documents, images, and complex data

**Part 2:**

- MCP
- Developing the research agent and the writing agent

**Part 3:**

- Making the research and writing agents ready for production
- Monitoring
- Evaluations

### Anchoring the Reader in the Educational Journey

Within the course we are teaching the reader multiple topics and concepts. Thus, understanding where the reader is in it's educational journey it's critical for this piece. You have to use only previous introduced concepts, while being reluctant about using concepts that haven't been introduced yet.

When discussing the **concepts introduced in previous lessons** listed in the `Concepts Introduced in Previous Lessons` section, avoid reintroducing them to the reader. Especially don't reintroduce the acronyms. Use them as if the reader already knows what they are. 

Avoid using all the **concepts that haven't been introduced in previous lessons** listed in the `Concepts That Will Be Introduced in Future Lessons` subsection. Whenever another concept requires references to these banned concepts, instead of directly using them, use intuitive analogies and or explanations that are more general and easier to understand, as you would explain them to a 7-year-old. For example:
- If the "tools" concept wasn't introduced yet and you have to talk about agents, refer to them as "actions".
- If the "routing" concept wasn't introduced yet and you have to talk about it, refer to it as "guiding the workflow between multiple decisions".
You can use the concepts that haven't been introduced in previous lessons listed in the `Concepts That Will Be Introduced in Future Lessons` subsection, only if we explicitly specify them. Still, even in that case, as the reader doesn't know how that concept works, you are just allowed to use the term, while keeping the explanation extremely high-level and intuitive, as if you were explaining it to a 7-year-old.
Whenever you use a concept from the `Concepts That Will Be Introduced in Future Lessons` subsection, explicitly specify in what lesson it will be explained in more detail, leveraging the particulars from the subsection. If not explicitly specified in the subsection, simply state that we will cover it in future lessons without providing a concrete lesson number. 

In all use cases avoid using acronyms that aren't explicitly stated in the guidelines. Rather use other more accessible synonyms or descriptions that are easier to understand by non-experts.

## Narrative Flow of the Lesson

Follow the next narrative flow when writing the end-to-end lesson:

- What problem are we learning to solve? Why is it essential to solve it?
    - Start with a personal story where we encountered the problem
- Why other solutions are not working and what's wrong with them.
- At a theoretical level, explain our solution or transformation. Highlight:
    - The theoretical foundations.
    - Why is it better than other solutions?
    - What tools or algorithms can we use?
- Provide some hands-on examples.
- Go deeper into the advanced theory.
- Provide a more complex example supporting the advanced theory.
- Connect our solution to the bigger field of AI Engineering. Add course next steps.

## Lesson Outline 

1. The Challenge with Complex Single LLM Calls
2. The Power of Modularity: Why Chain LLM Calls?
3. Building a Sequential Workflow: FAQ Generation Pipeline
4. Optimizing Sequential Workflows With Parallel Processing
5. Introducing Dynamic Behavior: Routing and Conditional Logic
6. Building a Basic Routing Workflow
7. Orchestrator-Worker Pattern: Dynamic Task Decomposition

## Section 1 - The Challenge with Complex Single LLM Calls

- This section is a mix of theory and practice.

- Explain the challenge: Why a single large LLM call for a complex, multi-step task can be problematic:
    - Difficulty in pinpointing errors or specific failures.
    - Lack of modularity; hard to update or improve specific parts.
    - Increased likelihood of "lost in the middle" issues with long contexts.
    - Potentially higher token consumption for prompts trying to do too much.
    - Less reliable outputs in general for complex multi-step tasks.

- Practical example:
    - Start with the setup instructions (importing the libraries and creating the client object, and definin the `MODEL_ID`). Remember that these are the usual setup instructions across the lessons.
    - Show an example of a complex prompt that tries to generate FAQs with questions, answers, and source citations all at once from renewable energy content. Reference the notebook code from the "The Challenge with Complex Single LLM Calls" section. Show:
        - The mock webpage setup (webpage_1, webpage_2, webpage_3 variables). No need to include their whole texts in the lessons.
        - The code in "Example: Complex Single LLM Call", and part of its output.

- While the output might be acceptable, explain that the more the instructions are complex, the more inaccuracies we'd have using a single prompt instead of splitting the problem in multiple sub-tasks (as humans do in life as well). For example, in the generated outputs there are often single sources, but actually sometimes the answers are sourced from multiple sources and there's a miss so.

**Section length:** 600 words

## Section 2 - The Power of Modularity: Why Chain LLM Calls?

- This is a theory-only section.

- Introduce prompt chaining: The concept of connecting multiple LLM calls (or other processing steps) sequentially, where the output of one step becomes the input for the next.
- This is a more manageable solution for complex tasks, where we divide-and-conquer.

- List the benefits of chaining:
    - Improved modularity: Each LLM call focuses on a specific, well-defined sub-task.
    - Enhanced accuracy: Simpler, targeted prompts for each step generally lead to better, more reliable outputs.
    - Easier debugging: Isolate issues to specific links in the chain.
    - Increased flexibility: Individual components can be swapped, updated, or optimized independently.
    - Potential for optimization: Use different models for different steps (e.g., a cheaper/faster model for a simple classification step, a more powerful model for complex generation).

- Discuss the downsides:
    - Some instructions may have sense only "together" and they lose meaning when split into multiple prompts/steps.
    - More costs (as more tokens are used).
    - Higher time to completion, as we have to wait for multiple LLM calls to complete.
    - Some information may be lost after doing multiple steps in a prompt chain (e.g. the first prompt may ask to summarize, while the second prompt may ask to translate, and it may lose some information from the summary while translating).

**Section length:** 400 words

## Section 3 - Building a Sequential Workflow: FAQ Generation Pipeline

- This section is practice-oriented.

- Show how the previous FAQ generation example can be split into a 3-step chain: Generate Questions → Answer Questions → Find Sources. Demonstrate how this sequential approach produces more consistent and traceable results. Use the code from section "Building a Sequential Workflow: FAQ Generation Pipeline" of the notebook, up to the point of running the full workflow. Highlight the total execution time.

- Provide a mermaid diagram illustrating the sequential FAQ generation pipeline, showing the flow from input content through the three stages: Generate Questions → Answer Questions → Find Sources, with arrows indicating data flow between steps.

**Section length:** 800 words

## Section 4 - Optimizing Sequential Workflows With Parallel Processing

- This section is practice-oriented.

- Explain how sequential workflows can be optimized through parallelization. While the sequential workflow works well, we can optimize it by running some steps in parallel, which can significantly reduce the overall processing time. Reference the code from the "Optimizing Sequential Workflows With Parallel Processing" section of the notebook. Highlight the total execution time.

- Compare the running time between sequential and parallel processing approaches. Discuss the trade-offs:
    - Sequential processing: Predictable execution order, easier to debug, higher total processing time.
    - Parallel processing: Significant reduction in processing time, more complex error handling, better resource utilization.

- Important note about rate limits: Mention that parallel processing may hit API rate limits (usually, models with free tiers have limits like 20 calls per minute) and how to handle this in real-world applications.

**Section length:** 600 words

## Section 5 - Introducing Dynamic Behavior: Routing and Conditional Logic

- This section is theory-oriented.

- Explain the need for routing: Not all inputs or intermediate states should be processed the same way. Make one example similar to the tasks example that is in the next sections.
- Discuss how an LLM call itself can be used to make the routing decision (e.g., by classifying input or an intermediate result).
- Explain the concept of "branching" in a workflow and when routing is preferable to trying to optimize a single prompt for multiple types of inputs. It's still a matter of "divide-and-conquer", where we try to keep prompts as specialized as possible, with single responsibility ideally.

**Section length:** 300 words

## Section - Building a Basic Routing Workflow

- This section is practice-oriented.

- Define a clear use case: a preliminary step in a customer service system that classifies the user's query intent and then routes it to a specialized prompt or handler.
- Reference the specific notebook code sections from "Building a Basic Routing Workflow".

- Provide a mermaid diagram illustrating the routing workflow, showing user input → intent classification → conditional branching to different specialized handlers (Technical Support, Billing Inquiry, General Question) → final responses.

**Section length:** 500 words

## Section 7: Orchestrator-Worker Pattern: Dynamic Task Decomposition

- This section is both theory and practice-oriented.

- Define the orchestrator-worker pattern: With orchestrator-worker, an orchestrator breaks down a task and delegates each sub-task to workers, which can run in parallel.
- In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results.
- When to use this workflow: This workflow is well-suited for complex tasks where you can't predict the subtasks needed. The key difference from parallelization is its flexibility—subtasks aren't pre-defined, but determined by the orchestrator based on the specific input.

- As example, include the code from the "Orchestrator-Worker Pattern: Dynamic Task Decomposition" section of the notebook. Show the complete execution flow with the complex customer query example that involves multiple tasks: billing inquiry, product return, and order status update.

- Include a Mermaid diagram showing the flowchart of the orchestrator-worker pattern.

**Section length:** 700 words

## Article code

Links to code that will be used to support the article. Always prioritize this code over every other piece of code found in the sources:

- [Notebook code for the lesson](https://github.com/towardsai/course-ai-agents/blob/dev/lessons/05_workflow_patterns/notebook.ipynb)

## Golden Sources

- [Prompt Chaining Guide](https://www.promptingguide.ai/techniques/prompt_chaining)
- [Building Effective Agents - Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- [Claude 4 Best Practices](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices)
- [LangGraph Workflows](https://langchain-ai.github.io/langgraphjs/tutorials/workflows)
- [Basic Multi-LLM Workflows](https://github.com/hugobowne/building-with-ai/blob/main/notebooks/01-agentic-continuum.ipynb)

## Other Sources

- [Chain Prompts - Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts)