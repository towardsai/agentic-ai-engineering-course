## What We Are Planning to Share

In this lesson we are planning to show how to apply the evaluator-optimizer pattern to our Brown writing workflow. We will start by introducing the generic pattern, then apply it to our unrefined writing workflow. Also, we will give an interesting parallel to how the evalautor-optimizer pattern differs from the Reflection pattern (indirectly explaining the reflection pattern as well). Then we will show to implement the evaluator-optimizer pattern from scratch. Ultimately, we will glue everything together within a LangGraph workflow with short-term memory.

## Why We Think It’s Valuable

We think that understanding how to apply the evaluator-optimizer pattern to a custom business use case it's a common and important pattern used to increase the quality of the final output. This pattern it's super powerful in scenarios where latency it's not critical. When there are too many rules to follow, this pattern it enforces the LLM to reason on top of them again and correct it's mistakes automatically creating a self optimization loop that can have huge impact over the final output.

## Who Is the Intended Audience

Aspiring AI engineers who are learning for the first time about AI agents and workflows. People who transition from other fields, such as data engineering, data science or software engineering and want to learn more about building AI agents and workflows.

⚠️ *EVERYTHING WILL BE WRITTEN RELATIVE TO THE LEVEL AND PERSPECTIVE OF THIS INTENDED AUDIENCE.*

## Theory / Practice Ratio

**15% theory – 85% practice**

## Point of View

The course is created by a team writing for a single reader, also known as the student. Thus, for voice consistency across the course, we will always use 'we,' 'our,' and 'us' to refer to the team that creates the course, and 'you' or 'your' to address the reader. Avoid using the singular 'I’ and don't use 'we' to refer to the student.

## Anchoring the Lesson in the Course

### Details About the Course

This piece is part of a broader course on AI agents and LLM workflows. The course consists of 4 parts, each with multiple lessons. 

Thus, it's essential to always anchor this piece in the broader course, understanding where the reader is in their journey. You will be careful to consider the following:

- To not reintroduce concepts already taught in the previous lesson.
- To be careful when talking about concepts introduced only in future lessons
- To always reference previous and future lessons when discussing topics outside the piece's scope.

### Course Syllabus

In previous lessons of the course, we introduced the following concepts:

**Part 1:**

- **L01 - AI Engineering & Agent Landscape**: Understanding the role, the stack, and why agents matter now
- **L02 - Workflows vs. Agents**: Grasping the crucial difference between predefined logic and LLM-driven autonomy
- **L03 - Context Engineering**: The art of managing information flow to LLMs
- **L04 - Structured Outputs**: Ensuring reliable data extraction from LLM responses
- **L05 - Basic Workflow Ingredients**: Implementing chaining, routing, parallel and the orchestrator-worker patterns
- **L06 - Agent Tools & Function Calling**: Giving your LLM the ability to take action
- **L07 - Planning & Reasoning**: Understanding patterns like ReAct (Reason + Act)
- **L08 - Implementing ReAct**: Building a reasoning agent from scratch
- **L09 - Agent Memory & Knowledge**: Short-term vs. long-term memory (procedural, episodic, semantic)
- **L10 - RAG Deep Dive**: Advanced retrieval techniques for knowledge-augmented agents
- **L11 - Multimodal Processing**: Working with documents, images, and complex data

**Part 2A - Overall Capstone projects Presentation:**

- **L12** — Presenting the capstone projects at a high level
- **L13** — AI framework trade‑offs
- **L14** — Capstone projects system design & cost levers.

**Part 2B - Nova Deep Research Agent:**

- **L15** — Presenting the project structure and design of the Nova deep research agent
- **L16** — MCP foundations (server/client; **tools/resources/prompts**; server‑hosted prompt).
- **L17** — Ingestion layer (guideline URL extraction; local/GitHub/YouTube ingest; Firecrawl scraping; file‑first design).
- **L18** — Research loop (generate queries → **Perplexity** with **structured outputs** → optional HITL between steps).
- **L19** — Testing out the Nova deep research agent

**Part 2C - Brown the Writing Workflow:**

- **L20** — Presenting the project structure and design of the Brown writing workflow
- **L21** — Presenting the system design and architecture of Brown the writing workflow
- **L22** — Implementing the foundations of Brown the writing workflow (such as applying the orchestrator-worker pattern to generate media items + context engineering to write high-quality articles that follow a specific pattern) 
- **🚨<<<  **L23** — Applying the evaluator-optimizer pattern to the Brown writing agent to automatically review and edit the article to automatically force adherence to expected requirements >>> CURRENT LESSON - REFERENCE PREVIOUS AND FUTURE LESSONS RELATIVE TO THIS ONE**
- **L24** - Expanding Brown the writing agent with multiple editing workflow that let’s you edit the whole article or just a piece of selected text. Everything is exposed as tools through MCP servers to facilitate human in the loop cycles

**Part 2D:**

- Orchestrating both MCP Servers (Nova + Brown) within a single MCP client, automating the whole logic as a unified workflow
- Demo showing how we used Nova + Brown to research and write professional articles or lessons.

**Part 3:**

With both the capstone projects built, this section focuses on the engineering practices required for production:

- AI Evals
- Prompt and system monitoring
- Tracking costs, latency, performance, and other metrics
- Deployment, Scaling to GCP
- Serve the app over the internet
- MCP Security
- CI/CD
- units tests

**Part 4:**

- In this final part of the course, you will build and submit your own advanced LLM agent, applying what you've learned throughout the previous sections. We provide a complete project template repository, enabling you to either extend our agent pipeline or build your own novel solution. Your project will be reviewed to ensure functionality, relevance, and adherence to course guidelines for the awarding of your course certification.

### CURRENT LESSON:

Now we are writing lesson 23 on using the evaluator-optimizer to implement the article editing and reviewing logic.

## Anchoring the Reader in the Educational Journey

Within the course, we are teaching the reader multiple topics and concepts. Thus, understanding where the reader is in their educational journey is critical for this lesson. You have to use only previously introduced concepts, while being reluctant about using concepts that haven't been introduced yet.

When discussing concepts introduced in previous lessons, avoid reintroducing them to the reader. Especially don't reintroduce the acronyms. Use them as if the reader already knows what they are. 

Avoid using the concepts that will only be introduced in future lessons. Whenever a concept from the current lesson requires references to concepts from future lessons, instead of directly using them, use intuitive analogies or explanations that are more general and easier to understand, as you would explain them to a 7-year-old. For example:

- If the "tools" concept wasn't introduced yet, and you have to talk about tool calling agents, refer to them as "actions".
- If the "routing" concept wasn't introduced yet and you have to talk about it, refer to it as "guiding the workflow between multiple decisions".

The idea is that if you have to use a concept that hasn’t been introduced yet, use real-world analogies that anyone can understand instead of specialized terminology. 

You can use the concepts that haven't been introduced yet only if we explicitly specify them in this guideline. Still, even in that case, as the reader doesn't know how that concept works, you are only allowed to use the term, while keeping the explanation extremely high-level and intuitive, as if you were explaining it to a 7-year-old.

Whenever you use a concept from future lessons, explicitly specify in what lesson it will be explained in more detail. If the lesson number is unclear, specify the part, such as part 3 or 4.

In all use cases, avoid using acronyms that aren't explicitly stated in the guidelines. Rather, use other, more accessible synonyms or descriptions that are easier to understand by non-experts.

## Narrative Flow of the Lesson

- Follow the next narrative flow when writing the end‑to‑end lesson:
- What problem are we learning to solve? Why is it essential to solve it?
- Why other solutions are not working and what's wrong with them.
- At a theoretical level, explain our solution or transformation. Highlight:
    - The theoretical foundations.
    - Why is it better than other solutions?
    - What methods or algorithms can we use?
- Provide simple examples or diagrams where useful.
- Go deeper into the advanced theory.
- Provide a more complex example supporting the advanced theory.
- Connect our solution to the bigger field of AI Engineering. Add course next steps.

---

## Lesson Outline

1. Introduction
2. Explaining the Evaluator-Optimizer Pattern
3. Scoping the New Writing Workflow Architecture
4. The Evaluator-Optimizer vs. Reflection Pattern
5. Modeling our Review Entities
6. Implementing the Article Reviewer (The Evaluator)
7. Hooking the Reviews to the Article Writer (The Optimizer) 
8. Centralizing Our App Configuration
9. Glueing Everything Into Our LangGraph Workflow
10. Adding Short-Term Memory
11. Running the New Writing Workflow
12. Conclusion

## Section  1 - Introduction

- **Quick reference to what we've learned in previous lessons:**  One sentence on what we’ve learnt in previous lessons, with a focus on lessons 20 and 21 as they are part of teaching the Brown capstone project
- **Transition to what we'll learn in this lesson:** After presenting what we learned in the past, make a transition to what we will learn in this lesson. Take the core ideas of the lesson from the `What We Are Planning to Share` section and highlight the importance and existence of the lesson based on the `Why We Think It’s Valuable` section.

- **Transition:** Enumerate what we are going to learn in this lesson with bullet points. Focus on outcomes of this lesson. Don’t blindly list the outline.

- **Section length:** 250 words

## Section 2 - Explaining the Evaluator-Optimizer Pattern

- Start by explaining the generic architecture of the evaluator-optimizer pattern. This is a fundamental AI workflow pattern that mirrors real-world quality assurance processes:
    - **Evaluator**: Analyzes output and identifies issues or areas for improvement
    - **Optimizer**: Takes the feedback and makes targeted improvements
    - The loop continues until a stopping condition is met (e.g., max iterations, quality threshold)

- Create a mermaid diagram showing the generic evaluator-optimizer pattern:
    1. Generate initial output
    2. Evaluate output against requirements
    3. If issues found, optimize/edit the output
    4. Loop back to step 2
    5. Return final output when done

- Here are some real-world examples where this pattern is heavily used:
    1. Code generation and review: Generate code, then review for best practices, security, and style before finalizing
    2. Video script writing: Write scripts, then review for pacing, engagement, and brand alignment
    3. Course material creation: Create lessons, then review for pedagogical effectiveness and alignment with learning objectives
    4. outputing financial reports ensuring that the output adheres to the financial data and overall layout of the report

- Real-world analogy: This approach is extremely similar to how a real-world writing process works:
    1. The writer writes the article (initial draft)
    2. A reviewer provides feedback from outside eyes
    3. The same writer edits the article based on the provided feedback
    4. Repeat steps 2-3 until satisfied

- **Transition:** Now that we understand the generic pattern, let's see how we apply it to our Brown writing workflow.

- **Section length:** 300 words (without the code, URL, or mermaid diagram's code)


## Section 3 - Scoping the New Writing Workflow Architecture

- Recap the 3-step workflow from Lesson 22:
    1. **Load Context into Memory** - Gather guidelines, research, profiles, and examples
    2. **Generate Media Items** - Use the orchestrator-worker pattern to create diagrams
    3. **Write the Article** - Generate the first draft using the ArticleWriter

- Introduce steps 4-5 as the new evaluator-optimizer loop that we are adding in this lesson:
    4. **Review the Article** (Evaluator) - Check the article against all profiles and guidelines
    5. **Edit the Article** (Optimizer) - Fix all identified issues based on the reviews
    - This review-edit pattern continues for a configurable number of iterations, gradually improving the article quality.

- In our case:
    - **Article Reviewer Node** = Evaluator (checks if article follows all the standards)
    - **Article Writer Node** = Optimizer (edits the article based on reviews)

- Include the workflow image from the notebook showing the complete 5-step workflow with the review-edit loop: `https://raw.githubusercontent.com/towardsai/agentic-ai-engineering-course/main/assets/images/l23_writing_workflow.png`

- Explain how this mimics the reviewing-editing process from the real-world of writing professional content.
    1. The writer writes the article (initial draft)
    2. A reviewer provides feedback from outside eyes
    3. The same writer edits the article based on the provided feedback
    4. Repeat steps 2-3 until satisfied

- **To clarify:** Why we don't compute any score into the evaluator optimizer loop as the original pattern states?
    - Because computing a metric on the quality of an article can be highly subjective and noisy. Thus, relying on a noisy threshold to stop the evaluation loop is not something you can easily trust. For example, it could stop faster than you want to or run more iterations than it should. You don't want either scenario. 
    - Still, quantifying the quality of an article can be doable on a limited number of binary metrics, as we will see in the AI Evals lessons (Part 3). But during our reviewing-editing loop we want to ensure it follows ALL the rules from the profiles, not just a subset.
    - We could have done a binary metric on every bullet point from the profiles, but that would have been extremely complicated to follow and debug.
    - Thus, as we want to rely heavily on human in the loop, as we will see in the next lesson (L24), we decided to run a limited number of the evaluation iterations in the first shot, then let the user read the article and decide if it needs more.
    - Doing so, we reduce any potential useless evaluation loops that add latency and cost. Also, we let the user add human feedback on top of the default guidelines to further guide the reviewing process and speed up converging to a high quality output.

- **Transition:** Before diving into the implementation, let's understand how the evaluator-optimizer pattern compares to a similar pattern called Reflection.

- **Section length:** 400 words (without the code, URL, or mermaid diagram's code)


## Section 4 - The Evaluator-Optimizer vs. Reflection Pattern

- Start with a brief explanation of why we are comparing these two patterns: both are used to iteratively improve LLM outputs, but they differ architecturally.

- **Evaluator-Optimizer Pattern:**
    - Uses **separate LLM calls** for evaluation and optimization
    - The evaluator and optimizer are distinct nodes/components in a workflow
    - Clear separation of concerns: one LLM judges, another LLM fixes
    - More transparent and debuggable as you can inspect intermediate states

- **Reflection Pattern:**
    - Uses the **same LLM call** with a reasoning loop
    - The LLM reflects on its own output within a single conversation/call
    - Self-critique: the same model that generated the output also evaluates it
    - More compact but less transparent as reasoning happens internally

- Create two mermaid diagrams side-by-side (or one after another) comparing:
    1. Evaluator-Optimizer: `Generate → Evaluate (LLM 1) → Optimize (LLM 2) → Loop`
    2. Reflection: `Generate → Self-Reflect (same LLM) → Revise → Loop`

- When to use each pattern:
    - **Evaluator-Optimizer**: When you need transparency, debugging capability, or want to use specialized models for different tasks (e.g., cheaper model for evaluation, stronger model for writing)
    - **Reflection**: When latency matters and self-correction is sufficient, or when working within agent loops

- Note: Usually the evaluator-optimizer pattern is used within workflow, while the reflection pattern is purely agentic. Now, as the line beween workflows and agents is blurry, you can apply the evaluator-optimizer pattern to an agentic design, where within your swarm of agents, you have a specialized evaluator and optimizer agents, and the decision to pass information between the two is done by an orchestrator agent, rather than hardcoded within the code as we do in a workflow. Ultimately, the biggest difference between the two methods is that the evaluation (or reflection) is done either by the same entity or by a different specialized one.

- Conclude by noting that for our writing workflow, we chose the evaluator-optimizer pattern because:
    1. We want to inspect reviews separately from the edited article
    2. We can use different model configurations for reviewing vs writing
    3. It's more aligned with how human writers and editors work

- **Transition:** Now that we understand the theoretical foundations, let's start implementing. First, we need to model our review entities.

- **Section length:** 350 words (without the code, URL, or mermaid diagram's code)


## Section 5 - Modeling our Review Entities

- Start with a note that in Lesson 22 we already covered the core entities like `Article`, `ArticleGuideline`, and `ArticleProfiles`. Now we need entities to represent the reviewing logic.

- Explain why we support two types of reviews:
    1. **Whole Article Reviews**: Review the entire article from top to bottom
    2. **Selected Text Reviews**: Review only a specific portion of the article
    - Most of the time, only a section of the article needs editing, not the whole thing. This targeted approach saves time and reduces API costs by only reviewing what matters. We will see that in action in the next lesson on human in the loop.

- Follow the code from the Notebook to explain each entity:
    <define_how_the_code_should_be_explained_step_by_step>
    1. **The `Review` entity**: A single piece of feedback about the article based on the expected writing profiles to ensure the article adheres to them
        - Show the code from the Notebook
        - Explain the key fields: `profile` (which requirement was violated), `location` (where in the article), `comment` (detailed explanation)
        - Show an example `Review` instance
    2. **The `ArticleReviews` entity**: Bundles multiple reviews for the whole article
        - Show the code from the Notebook
        - Explain that it contains the `article` and a `list[Review]`
    3. **The `SelectedText` entity**: Represents a specific portion of the article to review/edit.
        - Show the code from the Notebook
        - Explain the key fields: `article` (full context), `content` (the specific text), `first_line_number`, `last_line_number`
        - Explain that line numbers help locate the selection within the full article
    4. **The `SelectedTextReviews` entity**: Handles reviews for just a portion of the article
        - Show the code from the Notebook
        - Explain that it contains the `article`, `selected_text`, and `list[Review]`
        - Note that this will make much more sense when we apply it in the next human-in-the-loop lesson
    5. Show the entity relationships diagram from the Notebook:
        ```
        Article
          └── ArticleReviews
               └── reviews: list[Review]
        
        Article + SelectedText
          └── SelectedTextReviews  
               ├── selected_text: SelectedText
               └── reviews: list[Review]
        ```
    </define_how_the_code_should_be_explained_step_by_step>

- **Transition:** Now that we have our review entities modeled, let's implement the Article Reviewer node that acts as our evaluator.

- **Section length:** 350 words (without the code, URL, or mermaid diagram's code) 


## Section 6 - Implementing the Article Reviewer (The Evaluator)

- Brief recap (1-2 sentences) that we leverage the same `Node` abstraction from Lesson 22 to implement all our nodes.

- Follow the code from the Notebook to explain the `ArticleReviewer` class:
    <define_how_the_code_should_be_explained_step_by_step>
    1. **The Class and Initialization:**
        - Show the `ArticleReviewer` class `__init__` code
        - Explain key design decisions:
            - `to_review` can be either a full `Article` or `SelectedText` (polymorphic design)
            - Takes all requirements: guideline, profiles
            - No tools needed (empty toolkit) as reviewing is a pure generation task
    2. **Model Extension with Structured Output:**
        - Show the `_extend_model` method code
        - Explain the `ReviewsOutput` intermediate Pydantic model
        - Explain why an intermediate model: keeps LLM output schema simple, while giving us the flexibility to programatically model the final enitty output from the node
    3. **The `ainvoke` Method:**
        - Show the `ainvoke` method code
        - Explain the flow: format system prompt → add selected text instructions if needed → generate reviews → return appropriate type
        - note how we call `to_context()` on all the entities! That's all we have to do!
    4. **The System Prompt (Deep Dive using Prompt Anatomy):**
        - Show the full `system_prompt_template`
        - Apply the Anthropic prompt anatomy template to analyze the system prompt of the article reviewer. Split it in the following groups:
            - **Task context**: "You are Brown, an expert article writer, editor and reviewer..."
            - **Background data**: All the profiles and guidelines passed as context
            - **Detailed task description and rules**: The reviewing rules with priority ranking (special rules > guideline > article profile > other profiles)
            - **Output formatting**: Review structure (profile, location, comment)
            - **Chain of Thought**: Combines the immediate task description and thinking step-by-step phrase into an explicit reasoning steps glueing all the instructions from above to ensure the LLM reasons as expected.
        - What could we further improve it? By adding few-shot examples! We could do that by calling the reviewer on top of an article and then manually checking and improving the results.
    5. **The Selected Text System Prompt:**
        - Show the `selected_text_system_prompt_template`
        - Explain that it extends the main prompt with specialized instructions for selected text
        - Highlight the trick: it adds a new `Chain of Thoughts` section that overrides the original one. 💡 Note: Basically, as the chain of thoughts incorporates the immediate task and sits at the bottom of the prompt, it always dictates to the LLM what to do in the current call, while everything else usually sits as context.
    6. **Example: Reviewing a Whole Article:**
        - Show code to load sample article guideline and profiles
        - Show code to load a sample article to review (the one we generated in the previous lesson)
        - Show code to run the `ArticleReviewer`
        - Show the output: number of reviews generated and sample review details
    7. **Example: Reviewing Selected Text:**
        - Show code to extract a specific section as `SelectedText`
        - Show code to run the `ArticleReviewer` with `SelectedText` (note same class for both inputs)
        - Show the output: reviews for the selected text
    </define_how_the_code_should_be_explained_step_by_step>

- **Transition:** Now that we can generate reviews, let's see how we hook them back to the Article Writer for editing.

- **Section length:** 600 words (without the code, URL, or mermaid diagram's code)

## Section 7 - Hooking the Reviews to the Article Writer (The Optimizer)

- **Design Philosophy**: To keep the "writing" logic contained and avoid duplicated code, the `ArticleWriter` serves dual purposes:
    1. **Writer**: Generates the initial article draft (as we saw in L22)
    2. **Editor**: Edits the article based on reviews (new in this lesson)
    - This mirrors real-world writing processes where the original author both writes and edits their own work based on feedback. It keeps all writing knowledge in one place.

- Follow the code from the Notebook to explain the changes to `ArticleWriter`:
    <define_how_the_code_should_be_explained_step_by_step>
    1. **Changes to `__init__` (reviews parameter):**
        - Show the updated `__init__` code with the new `reviews` parameter
        - Explain that when `reviews=None`, the writer generates a new article from scratch
        - When reviews are provided, it edits the existing article based on the feedback
    2. **Changes to the `ainvoke` Method:**
        - Show the updated `ainvoke` code that handles both writing and editing
        - Explain the context engineering for editing, which is essential our conversation history:
            - **User**: System prompt with all context (guidelines, profiles, etc.)
            - **Assistant**: The previously written article (anchoring the reviews)
            - **User**: The reviews with specific issues to fix
            - **Assistant**: The edited article (generated by LLM)
    3. **The `article_reviews_prompt_template` (Deep Dive using Prompt Anatomy):**
        - Show the full prompt template
        - Analyze using prompt anatomy:
            - Explains the review creation process (for context to understand where the reviews are coming from and how to react to them)
            - Provides clear priority ranking (human feedback > guideline > other profiles)
            - New chain of thought section with editing-specific reasoning steps overriding the chain of thought for writing the article.
    4. **The `selected_text_reviews_prompt_template`:**
        - Show the full prompt template
        - Explain it's similar to article reviews but with special details on manipulating selected text
        - Note that LLMs have zero clue of your business logic, so you must explain all processes clearly
    5. **End-to-End Example: Review and Edit Loop**
        - Use the more comprehensive `inputs/tests/02_sample_medium` directory
        - **Step 1**: Load all necessary context (show code + output)
        - **Step 2**: Generate media items (show code + output)
        - **Step 3**: Write the first draft of the article (show code + output preview)
        - **Step 4**: Review the article (show code + output with sample reviews)
        - **Step 5**: Edit the article based on reviews (show code + output preview)
        - **Step 6**: Compare original vs edited (show comparison stats)
        - **Step 7**: Save the articles (show code + output paths)
    6. Add a note on how to check what actually changed after the review:
        - Go over the reviews and locate changes using the `location` field
        - Use a diff tool to compare files to see how the review was actually applied
        - Mention that after MCP integration, this will be automated
    </define_how_the_code_should_be_explained_step_by_step>

- **Transition:** Before gluing everything into a LangGraph workflow, let's explore our centralized configuration system.

- **Section length:** 600 words (without the code, URL, or mermaid diagram's code) 

## Section 8 - Centralizing Our App Configuration

- Start by explaining why centralized configuration matters. As our system grows more complex, we need:
    - **Single source of truth**: One file that controls everything
    - **Easy experimentation**: Change models, parameters without touching code. Thus, we can easily automate running different experiments, compare them, and pick the best config. More on this in the AI Evals lessons
    - **Environment-specific configs**: Different settings for dev/tests/prod (e.g., for unit tests we can easily mock all the LLM calls)
    - **Version control**: Track configuration changes over time

- Follow the code from the Notebook to explain the `AppConfig` class hierarchy:
    <define_how_the_code_should_be_explained_step_by_step>
    1. **Context Configuration:**
        - Show the `Context` class code
        - Explain it defines all the paths and loaders for different content types (guideline, research, article, profiles, examples)
        - Highlight the `build_article_uri` method for generating iteration filenames
    2. **Tool and Node Configuration:**
        - Show the `ToolConfig` and `NodeConfig` classes
        - Explain that each node can have its own model and configuration
        - Tools (like diagram generators) have their own configs too
    3. **Memory Configuration:**
        - Show the `Memory` class code
        - Explain it controls which checkpointing strategy to use (in_memory, sqlite)
    4. **The Main `AppConfig` Class:**
        - Show the `AppConfig` class code
        - Explain key features:
            - `num_reviews`: How many review-edit iterations to run
            - `nodes`: Configuration for each workflow node
            - `from_yaml()`: Load configuration from YAML file
            - Full Pydantic validation ensures type safety
    5. **Example YAML Configuration:**
        - Show the example `configs/course-gemini-flash.yaml` file content
        - Highlight configuration decisions:
            - **Media generation**: Uses fast Flash model with 0 temperature (deterministic)
            - **Article writing**: Uses Pro model with higher temperature (0.7) for creativity
            - **Reviewing**: Uses Pro model with 0 temperature (strict adherence to rules)
            - **Editing**: Uses Pro model with low temperature (0.1) for focused changes
            - **2 review iterations**: Runs the review-edit loop twice
    6. **Loading and Using the Configuration:**
        - Show code to load configuration from YAML
        - Show the output: configuration summary, node info
    </define_how_the_code_should_be_explained_step_by_step>

- **Transition:** Now that we have our configuration system, let's glue everything together into a LangGraph workflow.

- **Section length:** 450 words (without the code, URL, or mermaid diagram's code)

## Section 9 - Glueing Everything Into Our LangGraph Workflow

- Start with a brief overview: The workflow uses LangGraph's Function API to orchestrate the complete article generation process.
- Quick note on why LangGraph's Function API and not their Graph SDK: We built Brown version 1 with their graph SDK and found that it uselessly overcomplicated the code. For example, writing simple loops or if else clauses took hours instead of minutes. We found that the functional api is a good balance between writing from scratch, while still leveraging the langchain/langgraph rich ecosystem.

- Follow the code from the Notebook to explain the LangGraph workflow:
    <define_how_the_code_should_be_explained_step_by_step>
    1. **The Build Function (`build_generate_article_workflow`):**
        - Show the code
        - Add a callout and reference to docs explaining `@entrypoint` decorator from LangGraph's Function API
        - Explain why this builder pattern: allows injecting infrastructure dependencies (checkpointer) at runtime. To respect our clean architecture patterns 
        - This follows clean architecture: separate infrastructure from business logic
    2. **The Workflow Input (`GenerateArticleInput`):**
        - Show the `TypedDict` code
        - Simple typed input containing just the directory path where all resources are located
    3. **The Main Workflow Function (`_generate_article_workflow`):**
        - Show the main workflow function code
        - Add a callout and reference to docs explaining `get_stream_writer()` for progress reporting
        - Explain the flow:
            - Step 1: Load context (progress 0-2%)
            - Step 2: Generate media items (progress 3-10%)
            - Step 3: Write article (progress 15-20%)
            - Step 4-5: Review-edit loop for `num_reviews` iterations (progress 25-99%)
            - Save iteration files and final article
    4. **Task Functions with Retry Policies:**
        - Show the `@task` decorated functions: `generate_media_items`, `write_article`, `generate_reviews`, `edit_based_on_reviews`
        - Add a callout and reference to docs explaining LangGraph's `@task` decorator and `RetryPolicy`
        - Explain why retry policies matter:
            - Resilience against API failures, rate limits, network issues
            - Automatic recovery without manual intervention
            - Task-level granularity: only retry failed step, not entire workflow
            - Reproduceble debugging. For example, let's say that we want to test the `generate_reviews` step multiple times with the same state. Then due to the checkpointers, we can load existing outputs from the `generate_media_items` and `write_article` steps, and run only that particular step. Provide an example on how to do this.
    </define_how_the_code_should_be_explained_step_by_step>

- Give notes to remind people how our code follows clean code patterns:
    - We started introducing the **app layer** through the LangGraph workflows. Within the LangGraph workflows we glue all our entities and nodes together into a final piece of business logic that is actually usable.
    - Injected new components from the **infrastructure layer** such as the short-term memory (checkpointer). Note how the workflow doesn't care about what short-term memory we use. It cares only that it's a short-term memory component.
    - Note how we inject the infra stuff (memory) only when initializing the workflow (app components)
    - The domain layer (nodes, entities) remains independent of infrastructure choices

- **Transition:** Before running the workflow, let's understand the checkpointing system that provides workflow state persistence.

- **Section length:** 500 words (without the code, URL, or mermaid diagram's code)

## Section 10 - Adding Short-Term Memory

- Checkpointing works hand in hand with the short-term memory concept as it stores everything that happens within a thread. For example, within the checkpointer we can store the messages of a user between sessions. In our use case, we can save all the intermmediate steps of the article.
 
- Still, on top of just plain short-term memory + sotring it, checkpointing matters a lot for workflows, because it provides (basically it works hand in hand with the properties provided by the LangGraph workflow it self):
    1. **Resume from failure**: If a workflow crashes, resume from the last checkpoint (e.g., resume from the last `generate_reviews` step that crashed)
    2. **State inspection**: Examine workflow state at any point
    3. **Debugging**: Step through workflow execution
    4. **Human-in-the-loop**: Pause for human input, then resume

- Follow the code from the Notebook to explain the checkpointer options:
    <define_how_the_code_should_be_explained_step_by_step>
    1. **InMemory Checkpointer:**
        - Show the `build_in_memory_checkpointer` factory code
        - Explain characteristics:
            - **Ephemeral**: State lost when process ends
            - **Development-friendly**: Perfect for testing and iteration
            - **No persistence**: Not suitable for production long-running workflows
        - When to use: Development, testing, short-lived workflows
    2. **SQLite Checkpointer (brief mention):**
        - Show the `build_sqlite_checkpointer` factory code
        - Explain characteristics:
            - **Durable**: State stored on disk between processes
            - **Development-friendly**: Easy to set up as the database is just a file
        - **When to use**: SQLite can get you up to dozens of users before scaling. Thus, it's extremely powerful when first deploying your application, as it avoids setting up a database such as Postgres. It's also super easy to use when testing locally.
    </define_how_the_code_should_be_explained_step_by_step>

- **Transition:** Now let's run the complete workflow with LangGraph integration!

- **Section length:** 300 words (without the code, URL, or mermaid diagram's code) 

## Section 11 - Running the New Writing Workflow

- Add a note reminding that the number of review iterations is controlled by the `num_reviews` attribute from the config (default: `num_reviews=2`).

- Follow the code from the Notebook to run the complete workflow:
    <define_how_the_code_should_be_explained_step_by_step>
    1. **Run the Complete Workflow with LangGraph:**
        - Show code to build workflow with in-memory checkpointer
        - Show code to configure workflow with thread_id. Note: using UUID is an easy win to generate unique thread_ids each run.
        - Show code to run workflow with `astream` and handle events
        - Show the streaming progress events output
            - Loaded Context (Progress: 0-2%)
            - Generated Media (Progress: 3-10%)
            - Wrote Article (Progress: 15-20%)
            - Review-Edit Loop (Progress: 25-99%)
            - Final Save (Progress: 100%)
    2. **Article Evolution Across Iterations:**
        - Show output with file names and sizes:
            - `article_000.md`: Initial draft
            - `article_001.md`: After first review-edit iteration
            - `article_002.md`: After second review-edit iteration (if num_reviews=2).
            - `article.md`: Final refined article. It's the same with `article_002.md`.
        - Show the evolution of each article between different reviewing iterations. Avoid showing the whole article, as they are too big to show. Just pick the first review between each iteration and show how it changed the article.
        - Then, highlight that it's hard to show every changes, but to see everything that happened we recommend:
            - Recommend opening each checkpoint file to see how the article evolved located within the `inputs/tests/02_sample_medium` directory
            - And do a diff between each file to see what changed.
            - IMPORTANT NOTE: This will be a lot easier to do when we integrate the MCP server and trace monitoring in future lessons!
    </define_how_the_code_should_be_explained_step_by_step>

- **Transition:** Now let's wrap up with our conclusion and next steps.

- **Section length:** 400 words (without the code, URL, or mermaid diagram's code)

## Section 12 - Conclusion
(Connect our solution to the bigger field of AI Engineering. Add course next steps.)

- One paragraph conclusion remembering people why this is important: Understanding how to apply the evaluator-optimizer pattern is crucial for building production-quality AI systems that can self-improve and adhere to complex requirements. Get more insights from the `Why We Think It’s Valuable` section at the beginning of the guideline.

- **What we've learnt:**
    1. **The Evaluator-Optimizer Pattern**: How evaluators identify issues against requirements and optimizers fix them through iterative refinement
    2. **Evaluator-Optimizer vs Reflection**: Architectural differences between workflow-based (separate LLM calls) and agent-based (same LLM reasoning loop) approaches
    3. **Review Entities**: How to model reviews for both whole articles and selected text portions
    4. **Article Reviewer Implementation**: Building the evaluator with structured outputs and comprehensive system prompts
    5. **Article Writer as Editor**: How the same writer can generate and edit articles based on reviews
    6. **Centralized Configuration**: Using YAML and Pydantic for type-safe, flexible app configuration
    7. **LangGraph Integration**: Using Function API, tasks with retry policies, and streaming progress
    8. **Checkpointing**: State persistence for workflow resilience and debugging

- **Ideas on how you can further extend this code:**
    1. **Different AI frameworks**: Replace LangGraph with PydanticAI or other frameworks (clean architecture makes swapping easy)
    2. **Improve the reviewer**: Tweak the reviewer prompts or profiles for better results. Get an intuition on how it affects the final prompts. The biggest change you could do here is to add few-shot-examples to the reviewer.
    3. **Modify the configuration**: Change models, temperatures, num_reviews from YAML and observe output changes
    4. **Add a scoring mechanism to the evaluator**: Even so we advided against it, it will be a good exercise to modify the evaluator-optimizer implementation and see how it behaves.

- **Transition to next lessons:**
    - In **Lesson 24 (Human-in-the-Loop)**, we will explore adding two new LangGraph workflows for iteratively editing the whole article or just a selected piece of text to properly add humans in the loop between the generated article and future edit iterations. Finally, we will expose all workflows as MCP tools for seamless integration with tools like Cursor.

- **Section length:** 350 words

## Lesson Code

Links to code that will be used to support the lesson. Always prioritize this code over every other piece of code found in the sources:

1. [Notebook](https://github.com/towardsai/agentic-ai-engineering-course/tree/dev/lessons/23_evaluator_optimizer)


## Golden Sources

1. [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
2. [Prompting 101 | Code w/ Claude](https://www.youtube.com/watch?v=ysPbXH0LpIE)
3. [Stop Building AI Agents. Use These 5 Patterns Instead.](https://www.decodingai.com/p/stop-building-ai-agents-use-these)
4. [Reflection and Working Memory](https://www.newsletter.swirlai.com/p/building-ai-agents-from-scratch-part-8ca)
5. [Reflection Agents](https://blog.langchain.com/reflection-agents/)

## Documentation

1. [Functional API overview](https://docs.langchain.com/oss/python/langgraph/functional-api)
2. [Use the functional API](https://docs.langchain.com/oss/python/langgraph/use-functional-api)
