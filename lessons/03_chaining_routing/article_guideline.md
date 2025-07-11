## Global Context

- **What I'm planning to share:** This article explores the fundamental components for building LLM workflows: chaining multiple LLM calls and implementing routing or conditional logic. We will explain why breaking down complex tasks into chained calls is often more effective than relying on a single, large LLM call. Practical demonstrations will show how to build a sequential workflow for FAQ generation (Generate Questions → Answer Questions → Find Sources) and a routing workflow for customer service intent classification using the Google Gemini library.
- **Why I think it's valuable:** For an AI Engineer, mastering chaining and routing is the first step towards constructing sophisticated and reliable LLM applications. These techniques provide modularity, improve accuracy, and allow for more controlled and adaptable processing, forming the building blocks for both deterministic workflows and more complex agentic systems.
- **Who the intended audience is:** AI Engineers and developers looking to build sophisticated LLM applications with chaining and routing capabilities.
- **Expected length of the article in words** (where 200-250 words ~= 1 minute of reading time): 3000-3500 words

## Outline 

1. The Power of Modularity: Why Chain LLM Calls?
2. Building a Sequential Workflow: FAQ Generation Pipeline (with Google Gemini)
3. Introducing Dynamic Behavior: Routing and Conditional Logic
4. Building a Basic Routing Workflow: Classify Intent → Route (with Google Gemini)
5. Orchestrator-Worker

## Section 1: The Power of Modularity: Why Chain LLM Calls?

Explain the challenge: Why a single large LLM call for a complex, multi-step task can be problematic.
- Reduced accuracy and coherence, at least using non-thinking models. Thinking models are better at following complex prompts.
- Difficulty in pinpointing errors or specific failures.
- Lack of modularity; hard to update or improve specific parts.
- Increased likelihood of "lost in the middle" issues with long contexts.
- Potentially higher token consumption for prompts trying to do too much.

Show an example of a complex prompt that tries to generate FAQs with questions, answers, and source citations all at once from renewable energy content. While the output might be acceptable, explain how this approach can be improved and made more reliable through chaining. Show a minimal (but runnable) code example using Gemini-2.5-flash. Here you should explain how to set an environment variable in google Colab (as the students will run the code from Google Colab) and how to set the GOOGLE_API_KEY environment variable (it can be retrieved with `from google.colab import userdata`, and it will be used by the Gemini client), and then proceed by defining the prompt and using it to generate a completion. When talking about the "GOOGLE_API_KEY" api key, you should explain also how to get it. Explain also how to retrieve the API key. You'll find the code of this section below in these guidelines. Comment also the outputs of the code cells if relevant, they are in the code below as well between cells.

Introduce Chaining: The concept of connecting multiple LLM calls (or other processing steps) sequentially, where the output of one step becomes the input for the next.

List the benefits of chaining:
- **Improved Modularity:** Each LLM call focuses on a specific, well-defined sub-task.
- **Enhanced Accuracy:** Simpler, targeted prompts for each step generally lead to better, more reliable outputs.
- **Easier Debugging:** Isolate issues to specific links in the chain.
- **Increased Flexibility:** Individual components can be swapped, updated, or optimized independently.
- **Potential for Optimization:** Use different models for different steps (e.g., a cheaper/faster model for a simple classification step, a more powerful model for complex generation).

## Section 2: Building a Sequential Workflow: FAQ Generation Pipeline (with Google Gemini)

Show how the previous FAQ generation example can be split into a 3-step chain: Generate Questions → Answer Questions → Find Sources. Demonstrate how this sequential approach produces more consistent and traceable results. You'll find the code of this section below in these guidelines. Comment also the outputs of the code cells if relevant, they are in the code below as well between cells.

- Provide a mermaid diagram illustrating the sequential FAQ generation pipeline, showing the flow from input content through the three stages: Generate Questions → Answer Questions → Find Sources, with arrows indicating data flow between steps.

Discuss considerations for each step: e.g., prompt clarity, handling potential API errors.

List the downsides:
- For example, some instructions may have sense only "together" and they lose meaning when split into multiple prompts/steps.
- More costs (as more tokens are used).
- Higher time to completion, as we have to wait for multiple LLM calls to complete.
- Some information may be loss after doing multiple steps in a prompt chain (e.g. the first prompt may ask to summarize, while the second prompt may ask to translate, and it may lose some information from the summary while translating). This can also happen when not using prompt chaining though, but here it is more likely the higher the number of prompts in the chain.

## Section 3: Introducing Dynamic Behavior: Routing and Conditional Logic

Explain the need for routing: Not all inputs or intermediate states should be processed the same way.

Introduce conditional logic (e.g., Python if/elif/else statements) as the mechanism to direct the workflow's path.

Discuss how an LLM call itself can be used to make the routing decision (e.g., by classifying input or an intermediate result).

Explain the concept of "branching" in a workflow.

## Section 4: Building a Basic Routing Workflow: Classify Intent → Route (with Google Gemini)

Define a clear use case: e.g., a preliminary step in a customer service system that classifies the user's query intent and then routes it to a specialized prompt or handler.

You'll find the code of this section below in these guidelines. Comment also the outputs of the code cells if relevant, they are in the code below as well between cells. Step-by-step implementation guide:

**Step 1: Intent Classification Call**
- Design a prompt that instructs the LLM to classify the input text into predefined categories (e.g., "Technical Support," "Billing Inquiry," "General Question").
- Specify how the LLM should output the classification (e.g., a single keyword).
- Make the API call.
- Extract and parse the classification result.
- Include example Python code snippet.

**Step 2: Conditional Logic in Python**
- Use if/elif/else statements based on the classified intent.

**Step 3: Route to Specific Prompts/Handlers**
- Based on the intent, construct and execute a different, specialized LLM call (or trigger a non-LLM action).
- Example:
  - If intent is "Technical Support," use prompt_tech_support.
  - If intent is "Billing Inquiry," use prompt_billing.
- Include example Python code snippets for the conditional routing and subsequent calls.

- Provide a mermaid diagram illustrating the routing workflow, showing user input → intent classification → conditional branching to different specialized handlers (Technical Support, Billing Inquiry, General Question) → final responses.

Discuss challenges: Ensuring robust classification, handling ambiguous or out-of-scope intents, designing effective prompts for each branch.

## Section 5: Orchestrator-Worker

Define the orchestrator worker pattern:
- With orchestrator-worker, an orchestrator breaks down a task and delegates each sub-task to workers.

In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results.

When to use this workflow: This workflow is well-suited for complex tasks where you can't predict the subtasks needed (in coding, for example, the number of files that need to be changed and the nature of the change in each file likely depend on the task). Whereas it's topographically similar, the key difference from parallelization is its flexibility—subtasks aren't pre-defined, but determined by the orchestrator based on the specific input.

Show a code example of this pattern. Make it again a customer care example: the customer asks a query involving multiple questions and multiple actions by the customer care agents, and the LLM will extract them from the question. The orchestrator returns JSON-structured output that can be parsed into a list of tasks. Mention that we'll see later in the course a way to make the LLM output specific data types with a technique named "structured outputs", but for now we use JSON output mode. Each element of the list must have a "query_type" field and other parameters relevant to the specific item. Then, for each element of the list, the code will check the element to see if that element has a particular "query_type", and the code will route that element to the specific Python function to manage it (which must involve using an LLM again). The workers run sequentially in this example, though this could be done concurrently using async/await. Last, when each element of the list has received an answer, there must be a last LLM step where all the answers are collected and a final message/recap is synthesized and sent to the user. You'll find the code of this section below in these guidelines. Comment also the outputs of the code cells if relevant, they are in the code below as well between cells.

Explain the pros and cons of this pattern.

When copying the code from below, try to keep the code cells the same, without splitting their code while explaining them. Remember to comment also the outputs of the cells (it's fine to also comment just a part of the output).

## Article code

Links to code that will be used to support the article. Always prioritize this code over every other piece of code found in the sources: 

- [Notebook code for the lesson](https://github.com/towardsai/course-ai-agents/blob/main/lessons/03_chaining_routing/notebook.ipynb)

## Golden Sources

- [Prompt Chaining Guide](https://www.promptingguide.ai/techniques/prompt_chaining)
- [Building Effective Agents - Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- [Claude 4 Best Practices](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices)
- [Chain Prompts - Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts)
- [LangGraph Workflows](https://langchain-ai.github.io/langgraphjs/tutorials/workflows)

## Other Sources

No additional sources beyond the golden sources.