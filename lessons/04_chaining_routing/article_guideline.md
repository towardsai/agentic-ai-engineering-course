## Global Context

- **What I'm planning to share:** This article explores the fundamental components for building LLM workflows: chaining multiple LLM calls and implementing routing or conditional logic. We will explain why breaking down complex tasks into chained calls is often more effective than relying on a single, large LLM call. Practical demonstrations will show how to build a sequential workflow for FAQ generation (Generate Questions → Answer Questions → Find Sources) and a routing workflow for customer service intent classification using the Google Gemini library.
- **Why I think it's valuable:** For an AI Engineer, mastering chaining and routing is the first step towards constructing sophisticated and reliable LLM applications. These techniques provide modularity, improve accuracy, and allow for more controlled and adaptable processing, forming the building blocks for both deterministic workflows and more complex agentic systems.
- **Who the intended audience is:** AI Engineers and developers looking to build sophisticated LLM applications with chaining and routing capabilities.
- **Expected length of the article in words** (where 200-250 words ~= 1 minute of reading time): 3500-4000 words

## Outline 

1. The Challenge with Complex Single LLM Calls
2. The Power of Modularity: Why Chain LLM Calls?
3. Building a Sequential Workflow: FAQ Generation Pipeline (with Google Gemini)
4. Parallel Processing: Optimizing Sequential Workflows
5. Introducing Dynamic Behavior: Routing and Conditional Logic
6. Building a Basic Routing Workflow: Classify Intent → Route (with Google Gemini)
7. Orchestrator-Worker Pattern: Dynamic Task Decomposition

## Section 1: The Challenge with Complex Single LLM Calls

**Theoretical Foundation:**
Explain the challenge: Why a single large LLM call for a complex, multi-step task can be problematic.
- Difficulty in pinpointing errors or specific failures.
- Lack of modularity; hard to update or improve specific parts.
- Increased likelihood of "lost in the middle" issues with long contexts.
- Potentially higher token consumption for prompts trying to do too much.
- Less reliable outputs when using non-thinking models for complex multi-step tasks.

**Practical Example:**
Explain how to set up the environment in Google Colab, including setting the GOOGLE_API_KEY environment variable using `from google.colab import userdata`, and how to obtain the API key from Google AI Studio.
Show an example of a complex prompt that tries to generate FAQs with questions, answers, and source citations all at once from renewable energy content. Reference the notebook code from the "Setup: Installing Required Dependencies" section through the "Example: Complex Single LLM Call" section, specifically:
- The mock webpage setup (webpage_1, webpage_2, webpage_3 variables)
- The complex prompt that tries to do everything at once (prompt_complex variable)
- The Pydantic classes for structured outputs (FAQ, FAQList classes)
- The single LLM call execution and results

While the output might be acceptable, explain how this approach can be improved through chaining.

**Section length:** 600 words

## Section 2: The Power of Modularity: Why Chain LLM Calls?

Introduce Chaining: The concept of connecting multiple LLM calls (or other processing steps) sequentially, where the output of one step becomes the input for the next.

List the benefits of chaining:
- **Improved Modularity:** Each LLM call focuses on a specific, well-defined sub-task.
- **Enhanced Accuracy:** Simpler, targeted prompts for each step generally lead to better, more reliable outputs.
- **Easier Debugging:** Isolate issues to specific links in the chain.
- **Increased Flexibility:** Individual components can be swapped, updated, or optimized independently.
- **Potential for Optimization:** Use different models for different steps (e.g., a cheaper/faster model for a simple classification step, a more powerful model for complex generation).

Discuss the downsides:
- Some instructions may have sense only "together" and they lose meaning when split into multiple prompts/steps.
- More costs (as more tokens are used).
- Higher time to completion, as we have to wait for multiple LLM calls to complete.
- Some information may be lost after doing multiple steps in a prompt chain (e.g. the first prompt may ask to summarize, while the second prompt may ask to translate, and it may lose some information from the summary while translating).

**Section length:** 500 words

## Section 3: Building a Sequential Workflow: FAQ Generation Pipeline (with Google Gemini)

Show how the previous FAQ generation example can be split into a 3-step chain: Generate Questions → Answer Questions → Find Sources. Demonstrate how this sequential approach produces more consistent and traceable results.

Reference the specific notebook code sections:
- **Question Generation Function:** The `generate_questions()` function, `prompt_generate_questions` variable, and `QuestionList` Pydantic class
- **Answer Generation Function:** The `answer_question()` function and `prompt_answer_question` variable
- **Source Finding Function:** The `find_sources()` function, `prompt_find_sources` variable, and `SourceList` Pydantic class
- **Sequential Workflow Execution:** The `sequential_workflow()` function that combines all three steps

Show the execution results and timing comparison. Discuss considerations for each step: e.g., prompt clarity, handling potential API errors.

- Provide a mermaid diagram illustrating the sequential FAQ generation pipeline, showing the flow from input content through the three stages: Generate Questions → Answer Questions → Find Sources, with arrows indicating data flow between steps.

**Section length:** 800 words

## Section 4: Parallel Processing: Optimizing Sequential Workflows

Explain how sequential workflows can be optimized through parallelization. While the sequential workflow works well, we can optimize it by running some steps in parallel, which can significantly reduce the overall processing time.

Reference the specific notebook code sections:
- **Async Function Implementation:** The `answer_question_async()` and `find_sources_async()` functions
- **Parallel Question Processing:** The `process_question_parallel()` function
- **Parallel Workflow Execution:** The `parallel_workflow()` function using `asyncio.gather()`

Show the timing comparison between sequential and parallel processing approaches. Discuss the trade-offs:
- **Sequential Processing:** Predictable execution order, easier to debug, higher total processing time
- **Parallel Processing:** Significant reduction in processing time, more complex error handling, better resource utilization

Important note about rate limits: Mention that parallel processing may hit API rate limits and how to handle this in real-world applications.

**Section length:** 600 words

## Section 5: Introducing Dynamic Behavior: Routing and Conditional Logic

Explain the need for routing: Not all inputs or intermediate states should be processed the same way.

Introduce conditional logic (e.g., Python if/elif/else statements) as the mechanism to direct the workflow's path.

Discuss how an LLM call itself can be used to make the routing decision (e.g., by classifying input or an intermediate result).

Explain the concept of "branching" in a workflow and when routing is preferable to trying to optimize a single prompt for multiple types of inputs.

**Section length:** 400 words

## Section 6: Building a Basic Routing Workflow: Classify Intent → Route (with Google Gemini)

Define a clear use case: a preliminary step in a customer service system that classifies the user's query intent and then routes it to a specialized prompt or handler.

Reference the specific notebook code sections from "Building a routing workflow" through the specialized handlers:

**Step 1: Intent Classification**
- The `IntentEnum` class defining possible intents
- The `UserIntent` Pydantic class for structured output
- The `classify_intent()` function and `prompt_classification` variable
- Test examples with different query types

**Step 2: Specialized Handler Implementation**
- The specialized prompts: `prompt_technical_support`, `prompt_billing_inquiry`, `prompt_general_question`
- The `handle_query()` function with conditional logic (if/elif/else statements)
- Example executions showing how different intents are routed to different handlers

- Provide a mermaid diagram illustrating the routing workflow, showing user input → intent classification → conditional branching to different specialized handlers (Technical Support, Billing Inquiry, General Question) → final responses.

Discuss challenges: Ensuring robust classification, handling ambiguous or out-of-scope intents, designing effective prompts for each branch.

**Section length:** 800 words

## Section 7: Orchestrator-Worker Pattern: Dynamic Task Decomposition

Define the orchestrator-worker pattern: With orchestrator-worker, an orchestrator breaks down a task and delegates each sub-task to workers.

In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results.

When to use this workflow: This workflow is well-suited for complex tasks where you can't predict the subtasks needed. The key difference from parallelization is its flexibility—subtasks aren't pre-defined, but determined by the orchestrator based on the specific input.

Reference the specific notebook code sections from "Orchestrator-worker pattern" through the complete workflow:

**Orchestrator Implementation:**
- The `QueryTypeEnum`, `Task`, and `TaskList` Pydantic classes
- The `orchestrator()` function and `prompt_orchestrator` variable

**Worker Functions:**
- **Billing Worker:** The `handle_billing_worker()` function and `prompt_billing_worker_extractor`
- **Return Worker:** The `handle_return_worker()` function
- **Status Worker:** The `handle_status_worker()` function

**Response Synthesis:**
- The `synthesizer()` function and `prompt_synthesizer` variable

**Main Pipeline:**
- The `process_user_query()` function that coordinates the entire workflow
- The complex customer query example and its execution

Show the complete execution flow with the complex customer query example that involves multiple tasks: billing inquiry, product return, and order status update.

Explain the pros and cons of this pattern compared to other approaches.

**Section length:** 900 words

## Article code

Links to code that will be used to support the article. Always prioritize this code over every other piece of code found in the sources: 

- [Notebook code for the lesson](https://github.com/towardsai/course-ai-agents/blob/main/lessons/04_chaining_routing/notebook.ipynb)

## Golden Sources

- [Prompt Chaining Guide](https://www.promptingguide.ai/techniques/prompt_chaining)
- [Building Effective Agents - Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- [Claude 4 Best Practices](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices)
- [LangGraph Workflows](https://langchain-ai.github.io/langgraphjs/tutorials/workflows)
- [Basic Multi-LLM Workflows](https://github.com/hugobowne/building-with-ai/blob/main/notebooks/01-agentic-continuum.ipynb)

## Other Sources

- [Chain Prompts - Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts)