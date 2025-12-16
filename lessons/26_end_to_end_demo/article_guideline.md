## What We Are Planning to Share

In this lesson we plan to show how to properly add humans in the loop into our Brown writing workflow. We will start by explaining the AI generation / human validation loop theory, then explain how we decouple our workflows using MCP to enable a flexible human in the loop approach. Then we will show how we introduced human feedback support into the Article Reviewer node. Next, we will implement and run two new editing workflows: edit article and edit selected text. Finally, we will expose everything as an MCP server and show how to hook it into MCP clients like Cursor for a coding-like writing experience.

## Why We Think It's Valuable

We think that understanding how to properly add humans in the loop is critical for building practical AI applications. AI systems are imperfect - they hallucinate and make reasoning mistakes. Thus, the human should still do the thinking and planning while delegating the grunt work to the AI. This lesson shows how to create this balance through the AI generation / human validation loop, where the AI generates drafts and the human validates and provides feedback until satisfied with the output.

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
- **L23** — Applying the evaluator-optimizer pattern to the Brown writing agent to automatically review and edit the article to automatically force adherence to expected requirements
- **🚨 *<<< L24 - Expanding Brown the writing agent with multiple editing workflow that let's you edit the whole article or just a piece of selected text. Everything is exposed as tools through MCP servers to facilitate human in the loop cycles >>> CURRENT LESSON - REFERENCE PREVIOUS AND FUTURE LESSONS RELATIVE TO THIS ONE***

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

Now we are writing lesson 24 on adding human-in-the-loop to the Brown writing workflow through editing workflows and MCP server integration.

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
2. Understanding the AI Generation Human Validation Loop
3. Adding Human-In-The-Loop In Our Writing Workflow
4. Introducing Human Feedback Into the Article Reviewer
5. Implementing the Article Editing Workflow
6. Implementing the Selected Text Editing Workflow
7. Serving Brown as an MCP Server
8. Hooking to the MCP Server
9. Running Brown Through the MCP Server (Video)
10. Conclusion

## Section  1 - Introduction

- **Quick reference to what we've learned in previous lessons:**  One sentence on what we’ve learnt in previous lessons, with a focus on lessons 20 and 21 as they are part of teaching the Brown capstone project

- **Transition to what we'll learn in this lesson:** After presenting what we learned in the past, make a transition to what we will learn in this lesson. Take the core ideas of the lesson from the `What We Are Planning to Share` section and highlight the importance and existence of the lesson based on the `Why We Think It's Valuable` section. Emphasize that after generating an article, you'll likely want to refine it further, and that even the best AI-generated content benefits from human review and editing. The perfect balance is to use AI to generate and automate parts of your work, then have you, as the domain expert, review and refine it.

- **Transition:** Enumerate what we are going to learn in this lesson with bullet points. Focus on outcomes of this lesson:
    - Understand the AI generation / human validation loop and why it matters for building practical AI apps
    - Learn how to introduce human feedback into the article reviewer node
    - Implement two new editing workflows: edit article and edit selected text
    - Serve Brown as an MCP server with tools, prompts, and resources
    - Integrate Brown with MCP clients like Cursor for a coding-like writing experience

- **Section length:** 250 words

## Section 2 - Understanding the AI Generation Human Validation Loop

- Quick theoretical section on generally explaining how to design AI apps to properly add humans in the loop.

- **The core problem:** AI generates imperfect results - LLMs can hallucinate, display "jagged intelligence," and make mistakes that no human would. Thus, you need to create a balance between AI output and human input. The user must cooperate with the AI, where most of the thinking is still done by the human, while the grunt work is done by the AI. The role of the human is to plan and give instructions, then delegate the monotonous work to the AI and validate its outputs. Not to delegate thinking and planning to the AI.

- **The AI Generation / Human Validation Loop:** Explain this balance as a loop where:
    1. AI generates a draft
    2. Human validates it and adds feedback to the AI
    3. Repeat this loop until the human decides the output is good enough

- **Speeding up the loop** - Based on Karpathy's insights from the <notes>, explain two strategies:
    1. **Speeding Up Verification**: 
        - Application-specific GUIs help humans audit faster by utilizing visual representation
        - Showing code diffs in red/green reduces cognitive load vs reading raw text
        - Direct actions (accept/reject with simple commands) instead of typing instructions
    2. **Keeping the AI on the Leash**:
        - Avoid overreaction - receiving 10,000 lines of code diff is not useful as humans must still verify
        - Work in small incremental chunks focused on a "single concrete thing"
        - Use concrete prompts to increase probability of successful verification
        - Use auditable intermediate artifacts to constrain the AI

- **Connection to the autonomy slider:** Reference the agents vs workflows lesson (L02). As you give more "autonomy" to the AI app, you will have less human in the loop. The autonomy slider directly impacts how much human input you expect from your users. Karpathy's insight: build products that are "less Iron Man robots and more Iron Man suits" - focus on human augmentation and faster generation/verification loops, rather than fully autonomous agents.

- **Section length:** 400 words

<notes>
<source>[Andrej Karpathy: Software Is Changing (Again)](https://www.youtube.com/watch?v=LCEmiRjPEtQ)</source>
Karpathy frames the relationship between AI generation and human validation as a necessary loop required when "cooperating with AIs" in new LLM applications.
He emphasizes that this loop is essential because LLMs are "fallible systems that aren't yet perfect". They can hallucinate, display "jagged intelligence," and make mistakes that no human would. Consequently, in this cooperative workflow, AIs handle the generation, while humans perform the verification.
The primary goal is to make this generation-verification loop spin as fast as possible so that humans can maximize productivity. Karpathy outlines two major strategies for achieving this:
1. Speeding Up Verification
    To accelerate the human's ability to audit the AI's work, Karpathy stresses the importance of application-specific GUIs (Graphical User Interfaces).
    • Visual Representation: GUIs allow a human to audit the work of fallible systems and go faster because they utilize the human's "computer vision GPU".
    • Reduced Cognitive Load: Reading and interpreting text is effortful, but looking at visual representations is easier and acts as a "highway to your brain".
    • Streamlined Auditing: Instead of receiving output directly as text, a GUI should present changes clearly, such as showing a code diff in red and green.
    • Direct Action: Humans should be able to accept or reject work using simple commands (like Command Y or Command N) rather than having to type out instructions in text.
2. Keeping the AI on the Leash
Karpathy advises against giving the AI excessive autonomy, noting that many people are "way over excited with AI agents". The human remains the bottleneck in the verification process, even if the AI generates a massive output instantly.
    • Avoid Overreaction: The AI can be "way too overreactive". It is not useful, for example, to receive a diff of 10,000 lines of code, as the human must still verify that the code does the correct thing, introduces no bugs, and has no security issues.
    • Incremental Chunks: To manage the risk and maintain speed, Karpathy advocates for working in small incremental chunks and ensuring the work is focused on a "single concrete thing".
    • Concrete Prompts: Spending more time to be more concrete in prompts increases the probability of successful verification. If a prompt is vague, verification might fail, forcing the human to restart the process and "start spinning".
    • Intermediate Artifacts: Another way to keep the AI on the leash is by using auditable intermediate artifacts (like a course syllabus or progression of projects in an education app) to constrain the AI and prevent it from getting "lost in the woods".
This idea of managing autonomy is built into LLM apps via the "autonomy slider," which allows the user to tune the amount of control they give to the tool (ranging from small changes to full repository changes), depending on the complexity of the task. Karpathy ultimately suggests building products that are "less Iron Man robots and more Iron Man suits"—meaning products that focus on human augmentation and faster generation/verification loops, rather than fully autonomous agents. He believes that over the next decade, products will gradually move the autonomy slider from left to right.
</notes>

## Section 3 - Adding Human-In-The-Loop In Our Writing Workflow

- **The problem:** We have the generation step from Lessons 22-23, but how can we properly introduce the human in the loop to adhere to principles discussed in the previous section? We need a way, after the article is generated, to allow the human to review the article, validate it, and provide further instructions on how to improve or change the article.

- **The parallel to coding:** Writing an article or any other piece of content is incredibly similar to writing code. As engineers, this parallel was obvious to us. That's why we introduce human in the loop similar to how AI IDEs such as Cursor or Claude Code do: by directly specifying what you want, pass that feedback to the AI, and then the AI returns with the changes as a diff between the old and new suggested changes.

- **Our solution:** We implemented two new editing workflows that apply the evaluator-optimizer pattern with human feedback:
    1. **Edit Article Workflow**: Reviews and edits the entire article based on human feedback
    2. **Edit Selected Text Workflow**: Reviews and edits only a specific portion of the article
    - These workflows also accept human feedback that is directly plugged into the algorithm along with checking adherence to the profiles. We instructed the evaluator-optimizer loop to always prioritize human feedback over everything else.

- **Decoupling workflows with MCP:** To enable this approach, we decouple the article generation workflow from the editing workflows using MCP servers:
    - The `generate_article` workflow is one independent MCP tool
    - The `edit_article` workflow is another independent MCP tool
    - The `edit_selected_text` workflow is a third independent MCP tool
    - This allows you to generate an article, review it, and then selectively apply additional editing workflows with your human feedback until satisfied

- **Include the workflow diagram** from the notebook showing the three MCP tools and the human feedback loop: `https://raw.githubusercontent.com/iusztinpaul/agentic-ai-engineering-course-data/main/images/l24_writing_workflow.png`

- **Avoiding overreaction:** Editing the whole article can be an overreaction as it can make changes where you don't want any. That's why we implemented the `edit_selected_text` workflow that allows you to edit only a selected piece of text. We apply the same logic as before, but only to the selected portion.

- **Transition:** Now that we understand the overall design, let's see how we introduced human feedback into the article reviewer.

- **Section length:** 400 words (without the code, URL, or mermaid diagram's code)


## Section 4 - Introducing Human Feedback Into the Article Reviewer

- This section explains how we introduced human feedback support into the article reviewer. We'll start by explaining the `HumanFeedback` entity, then show how it's integrated into the `ArticleReviewer` node, and finally demonstrate it with a working example.

- Follow the code from the Notebook to explain the human feedback integration step-by-step:
    <define_how_the_code_should_be_explained_step_by_step>
    1. **The `HumanFeedback` Entity:**
        - Show the `HumanFeedback` Pydantic model code from `brown.entities.reviews`
        - Explain that it's a simple model with a `content` field that implements `ContextMixin`
        - Show how `to_context()` wraps the content in XML tags for the LLM
    2. **Initialization with Human Feedback in ArticleReviewer:**
        - Show the updated `__init__` method of the `ArticleReviewer` class
        - Explain that the `ArticleReviewer` now accepts an optional `human_feedback` parameter
        - This allows the reviewer to work with or without human input
    3. **Human Feedback in the System Prompt:**
        - Show the relevant section of the `system_prompt_template` that handles human feedback
        - Explain the three-part instruction for using human feedback:
            1. Use it to guide the reviewing process and focus on specific rules
            2. Extract one or more action points from the feedback (1 to N depending on ideas present)
            3. Always return at least 1 action point if feedback is provided
        - Show the example review structure with `profile="human_feedback"`
    4. **Injecting Human Feedback into the Prompt:**
        - Show the relevant part of the `ainvoke` method that formats the system prompt
        - Explain how `human_feedback.to_context()` is called if feedback exists, otherwise empty string
    5. **Example: Using ArticleReviewer with Human Feedback:**
        - Import necessary components (show the imports)
        - Load sample inputs:
            - Load article guideline from `article_guideline.md`
            - Load profiles from the profiles directory
            - Load article examples from examples directory
            - Load the sample article to review
        - Show the output: number of characters loaded for each
        - Show a reminder of how the article looks like (first 4000 characters)
        - Create a `HumanFeedback` instance with sample feedback
        - Create the `ArticleReviewer` with human feedback
        - Run the reviewer with `await article_reviewer.ainvoke()`
        - Show the output: number of reviews generated
    6. **Examining the Reviews:**
        - Filter reviews by `profile == "human_feedback"`
        - Show the human feedback reviews with their profile, location, and comment
        - Show all reviews grouped by profile type
        - Highlight how the reviewer generates reviews from multiple sources:
            - Human feedback reviews that directly address specific requests
            - Profile-based reviews (article, structure, mechanics, terminology, tonality) that ensure adherence to style guidelines
    </define_how_the_code_should_be_explained_step_by_step>

- **Transition:** Now that we understand how human feedback integrates with the article reviewer, let's explore the `edit_article` workflow.

- **Section length:** 550 words (without the code, URL, or mermaid diagram's code)


## Section 5 - Implementing the Article Editing Workflow

- The `edit_article` workflow reviews and edits an existing article based on human feedback and the expected requirements. It contains only one loop of the same reviewing-editing logic we already use within the generate article workflow.

- **Context:** The edit article workflow follows the same clean architecture pattern we've used throughout Brown. It leverages the app layer to orchestrate nodes and entities, keeping the code modular and maintainable. Important note on how because we made the entities and nodes 100% orthogonal we can reuse them as they are within the rest of the workflow. We just have to implement the glue code from the app layer that wraps them into a different business use case.

- Follow the code from the Notebook to explain the workflow implementation step-by-step:
    <define_how_the_code_should_be_explained_step_by_step>
    1. **Building the Edit Article Workflow:**
        - Show the `build_edit_article_workflow` function from `brown.workflows.edit_article`
        - Explain it's a factory that creates the workflow with a checkpointer for persistence
        - Explain the `@entrypoint` decorator from LangGraph's Function API
        - Show the `EditArticleInput` TypedDict:
            - `dir_path`: Directory containing the article and supporting files
            - `human_feedback`: Human feedback string to guide the editing
    2. **The Main Workflow Function:**
        - Show the `_edit_article_workflow` function code
        - Explain `get_stream_writer()` for progress reporting
        - Walk through the flow step-by-step:
            1. Load context (progress 0-5%): Use loader builders to load article, guideline, profiles, research, examples
            2. Create human feedback entity from the input string
            3. Generate reviews (progress 20-40%): Run the article reviewer with human feedback
            4. Edit based on reviews (progress 60-80%): Run the article writer with reviews
            5. Return instructions (progress 100%): Return edited article with instructions for MCP client
        - Highlight the return string that instructs the MCP client what to do:
            - Print the edited article
            - Give a summary of changes
            - Always apply changes to source file without waiting for user confirmation
    3. **The `generate_reviews` Task:**
        - Show the `generate_reviews` task function with `@task` decorator and retry policy
        - Explain it builds the model from app config for "review_article" node
        - Creates `ArticleReviewer` with article, guideline, profiles, and human feedback
        - Returns `ArticleReviews`
    4. **The `edit_based_on_reviews` Task:**
        - Show the `edit_based_on_reviews` task function
        - Explain it builds the model from app config for "edit_article" node
        - Creates `ArticleWriter` with all context and the reviews to address
        - Highlight that `ArticleWriter` works in "editing mode" when provided with `reviews` - uses same writer node but reviews guide specific changes
        - Returns the edited `Article`
    5. **Running the Edit Article Workflow:**
        - Import the workflow and checkpointer
        - Build workflow with in-memory checkpointer
        - Configure with thread ID (highlight using UUID for unique IDs)
        - Run with `astream` and handle events
        - Show the streaming progress events output:
            - Loading context, Loaded context
            - Reviewing article, Generated reviews
            - Editing article, Edited article
            - Article editing completed
        - Show the final output with the edited article
    6. **The Power of Human Feedback:**
        - Explain the key advantage: We can use a low number of review loops during initial generation, and further run them dynamically with human in the loop when necessary, with more human guidance
        - This means: Initial generation is faster/cheaper, we don't assume how many iterations needed, user decides, repeat until satisfied
    </define_how_the_code_should_be_explained_step_by_step>

- **Transition:** While the edit article workflow handles entire article edits, you'll often want to refine just a specific section. Let's explore the `edit_selected_text` workflow.

- **Section length:** 600 words (without the code, URL, or mermaid diagram's code)

## Section 6 - Implementing the Selected Text Editing Workflow

- The `edit_selected_text` workflow enables precise, focused edits on selected text portions. The workflow structure is almost identical to `edit_article`, thanks to our clean architecture. The main difference is that it operates on a `SelectedText` entity instead of the full `Article`, reducing overreaction (aka editing the whole article) when we know exactly what we want to edit and where we want to edit.
- This makes everythign fastter and cheaper to run and avoids validation overload, as we have to check the results from Brown only from a small portion of the results.

- Follow the code from the Notebook to explain the workflow implementation step-by-step:
    <define_how_the_code_should_be_explained_step_by_step>
    1. **Building the Edit Selected Text Workflow:**
        - Show the `build_edit_selected_text_workflow` function from `brown.workflows.edit_selected_text`
        - Explain it follows the same pattern as edit article workflow
        - Show the `EditSelectedTextInput` TypedDict:
            - `dir_path`: Directory containing the article and supporting files
            - `human_feedback`: Human feedback to guide the editing
            - `selected_text`: The specific text portion to edit
            - `number_line_before_selected_text`: Starting line number in the article
            - `number_line_after_selected_text`: Ending line number in the article
        - Explain the line numbers help locate the selected text within the larger article context
    2. **The Main Workflow Function:**
        - Show the `_edit_selected_text_workflow` function code
        - Walk through the flow step-by-step:
            1. Load context (progress 0-5%): Load full article and supporting files
            2. Create `SelectedText` entity: Contains selected portion, full article for context, and line numbers
            3. Create human feedback entity
            4. Generate reviews (progress 20-40%): Review the selected text with human feedback
            5. Edit based on reviews (progress 60-80%): Edit the selected text based on reviews
            6. Return instructions (progress 100%): Return edited selected text with instructions for MCP client
    3. **The `generate_reviews` Task for Selected Text:**
        - Show the task function
        - Highlight key differences from article version:
            - Takes `SelectedText` instead of `Article`
            - Returns `SelectedTextReviews` instead of `ArticleReviews`
            - Uses "review_selected_text" node config
        - Explain that `ArticleReviewer` is smart enough to handle both cases - when given `SelectedText`, it focuses reviews on that portion while using full article as context
    4. **The `edit_based_on_reviews` Task for Selected Text:**
        - Show the task function
        - Highlight key differences:
            - Takes `SelectedTextReviews` instead of `ArticleReviews`
            - Returns `SelectedText` instead of `Article`
            - Uses "edit_selected_text" node config
        - Explain that `ArticleWriter` handles both article and selected text editing seamlessly
    5. **Why Edit Selected Text?**
        - Most often we don't want to edit the whole article, but just a small section
        - Benefits: Faster and cheaper edits, more precise changes, iterative refinement of individual sections, better control over editing process
    6. **Running the Edit Selected Text Workflow:**
        - First, explicitly load the selected text from the sample article:
            - Load the article
            - Define `start_line` and `end_line`
            - Extract selected text using line slicing
            - Show the selected text to edit
        - Import and build the workflow with in-memory checkpointer
        - Configure with thread ID
        - Run with `astream` with all required inputs:
            - `dir_path`
            - `human_feedback`
            - `selected_text`
            - `number_line_before_selected_text`
            - `number_line_after_selected_text`
        - Show the streaming progress events
        - Show the final output with the edited selected text
    </define_how_the_code_should_be_explained_step_by_step>

- **Transition:** Now that we have both editing workflows implemented, let's see how to serve them as an MCP server for integration with tools like Cursor.

- **Section length:** 550 words (without the code, URL, or mermaid diagram's code)

## Section 7 - Serving Brown as an MCP Server

- All the MCP code lives in the `brown.mcp` module, keeping the serving layer completely separate from the domain, app, and infrastructure layers. This separation allows us to potentially serve Brown through different interfaces (CLI, FastAPI, etc.) without changing the core logic.
- We won't go again through the concepts behind MCP and FastMCP, as we had a dedicated lesson on it within the Nova research agent capstone project. Here we will jump straight into the implementation and serve Brown as an MCP server with 3 tools, 3 prompts and 2 resources.

- Screesnhot of how Brown looks with Cursor when working and hooked as an MCP server (add placeholder and we will add the image)

- Follow the code from the Notebook to explain the MCP server implementation step-by-step:
    <define_how_the_code_should_be_explained_step_by_step>
    1. **MCP Server and Tools:**
        - Show the MCP server initialization using FastMCP from `brown.mcp.server`
        - Show all three tool implementations:
            - `generate_article`: Takes `dir_path` and `ctx`, generates article from scratch
            - `edit_article`: Takes `article_path`, `human_feedback`, and `ctx`, edits entire article
            - `edit_selected_text`: Takes `article_path`, `human_feedback`, `selected_text`, `first_line_number`, `last_line_number`, and `ctx`, edits selected portion
        - Explain that each tool:
            - Builds the appropriate workflow with an in-memory checkpointer
            - Creates a unique thread ID for each workflow execution
            - Has detailed pydocs and signatures used by MCP client to understand what tool to call and how
            - Streams progress updates to MCP client via the `ctx` (context) parameter
    2. **Progress Reporting with `parse_message`:**
        - Show the `parse_message` helper function
        - Explain it parses workflow streaming messages and reports progress to MCP client
        - Handles both string messages and dictionary progress updates
        - Uses `ctx.info()` for logging and `ctx.report_progress()` for progress reporting
    3. **Running the Edit Selected Text Tool:**
        - Import the MCP server and create an in-memory client
        - Show the client creation: `Client(mcp)`
        - Load the selected text to edit (reminder from previous section)
        - Call the tool with `mcp_client.call_tool("edit_selected_text", {...})`
        - Show the beautiful progress messages while the workflow runs
        - Show the result with the edited selected text
    4. **Important Observation on Applying Changes:**
        - Explain the instructions after the `selected_text` XML block
        - These instructions tell the MCP client (like Cursor) what to do with the edited text
        - In Cursor, it shows the `diff` experience where you manually accept new changes, improving human in the loop experience
    5. **MCP Prompts:**
        - Show all three prompt implementations:
            - `generate_article_prompt`: Takes `dir_path`, triggers article generation
            - `edit_article_prompt`: Takes optional `human_feedback`, triggers article editing
            - `edit_selected_text_prompt`: Takes optional `human_feedback`, triggers selected text editing
        - Explain prompts make it easy for users to trigger tools without reading documentation
        - Show example of getting a prompt: `mcp_client.get_prompt("generate_article_prompt", {...})`
        - Show the prompt message output
        - Highlight that the real beauty is when plugging the server into Cursor - the chatbot picks up instructions from prompts and follows them
    6. **MCP Resources:**
        - Show the resource implementations:
            - `get_app_config`: Returns app configuration as JSON
            - `get_character_profile`: Returns character profile markdown
        - Explain resources provide read-only access to configuration and state
        - Show listing resources: `mcp_client.list_resources()`
        - Show reading a resource: `mcp_client.read_resource("resource://config/app")`
        - Show the app configuration output
        - Show the character profile output
    </define_how_the_code_should_be_explained_step_by_step>

- NOTE: **Clean Separation of MCP Layer as the serving layer:**
        - Explain how MCP layer imports from other layers:
            - Domain layer: Entities like `Article`, `HumanFeedback`, `Review`
            - App layer: Workflows like `build_edit_article_workflow`
            - Infrastructure layer: Builders like `build_loaders`, `build_short_term_memory`
        - The MCP Server implementation is our serving layer, that instantiates the app components (our workflows) and injects the necessary infrastructure components (the checkpoints, file loaders and renderers)
        - Within the serving layer, we initialize all our components, usually the ones from the app and infrastructure layers, and inject the infrastructure instances into the app layer. Like this we keep the two completely decoupled 
        - MCP layer is just a thin serving/interface layer
        - Benefits: Can serve Brown through different methods such as FastAPI or a CLI, MCP code is independent, easy to add new tools/prompts/resources, easier testing

- **Transition:** Now let's see how to connect to the MCP server from MCP clients.

- **Section length:** 650 words (without the code, URL, or mermaid diagram's code)

## Section 8 - Hooking to the MCP Server

- This section covers two ways to connect to Brown's MCP server: using the CLI script and integrating with Cursor.

- **Option 1 - Brown CLI Script:**
    - Brown includes a command-line interface at `lessons/writing_workflow/scripts/brown_mcp_cli.py`
    - Provides three main commands:
        1. `generate-article`: Generate an article from scratch
        2. `edit-article`: Edit an entire article based on human feedback
        3. `edit-selected-text`: Edit a selected section of an article
    - ⚠️ Note: These commands can only be run from the Brown standalone project at `lessons/writing_workflow`
    - Show usage examples:
        ```bash
        # Generate an article
        python scripts/brown_mcp_cli.py generate-article --dir-path /path/to/article

        # Edit an entire article
        python scripts/brown_mcp_cli.py edit-article \
            --dir-path /path/to/article \
            --human-feedback "Make the introduction more engaging"

        # Edit selected text
        python scripts/brown_mcp_cli.py edit-selected-text \
            --dir-path /path/to/article \
            --human-feedback "Make this shorter. Remove all em-dashes." \
            --first-line 10 \
            --last-line 20
        ```
    - The CLI uses the in-memory MCP client to call the MCP server tools

- **Option 2 - Cursor Integration:**
    - Show the Cursor MCP configuration in `.cursor/mcp.json`:
        - Configuration for just Brown
        - Configuration for both Nova and Brown together
    - Explain what the configuration tells Cursor:
        - The name of the MCP server ("brown")
        - How to launch the server using `uv` and Python
        - The working directory
        - Where to load environment variables
    - Instructions to verify: Go to Cursor's settings `Tools & MCP` section and ensure it has discovered `brown` with a green light

- **The Brown + Human-in-the-Loop Writing Experience:**
    - Explain the workflow when using Brown with Cursor:
        1. Generate Initial Article: Use the `generate_article` tool or prompt to create a first draft
        2. Review as Human Expert: Read through the generated article with your domain expertise
        3. Provide Feedback: Select sections that need improvement and provide specific feedback
        4. AI Edits: Use `edit_article` or `edit_selected_text` tools to refine based on your feedback
        5. Iterate: Repeat steps 2-4 until satisfied with the article
    (In the next section and in the demo we will show you a video with how we did this.)
    - This creates a collaborative experience where:
        - AI handles the heavy lifting of writing and editing
        - You guide the direction with your expertise and feedback
        - The workflow is fast and iterative
        - You maintain full control over the final output

- **Note on HTTP transports:** We will show how to use remote HTTP based transports in Part 3 when we cover deployment.

- **CTA:** For more details on running Brown as a standalone project, see the documentation at `lessons/writing_workflow/README.md`

- **Section length:** 400 words (without the code, URL, or mermaid diagram's code)


## Section 9 - Running Brown Through the MCP Server (Video)

- This section provides a video demonstration showing how to use Brown's human-in-the-loop features through the MCP Server in Cursor.

- **Video content overview:**
    - Quick walkthrough of configuring and connecting to the MCP Server
    - Demonstrate the human-in-the-loop workflow:
        1. Generate an initial article using the `generate_article` tool
        2. Review the generated article as a human expert
        3. Provide feedback and use `edit_article` to refine
        4. Select specific sections and use `edit_selected_text` for precise edits
        5. Show the diff experience in Cursor where you accept/reject changes
    - Show how prompts make it easy to trigger workflows without knowing the exact tool parameters
    - Demonstrate the iterative refinement process until satisfied with the output

- **Proof of work:** Highlight that we generated this lesson while doing the course using the exact methods we explained here - a real-world demonstration of the human-in-the-loop writing process.

- **Video link placeholder:** [Link to video demonstration will be provided]

- **Section length:** 150 words


## Section 10 - Conclusion
(Connect our solution to the bigger field of AI Engineering. Add course next steps.)

- One paragraph conclusion: Understanding how to properly add humans in the loop is critical for building practical AI applications. AI systems are imperfect - they hallucinate and make reasoning mistakes. The balance we've built allows AI to handle heavy lifting while keeping humans in control. Leverage insights from the `Why We Think It's Valuable` section.

- **What we've learned:**
    1. **The AI Generation / Human Validation Loop**: How to design AI workflows that balance automation with human expertise, where AI generates and humans validate
    2. **Human Feedback Integration**: How to introduce human feedback into the article reviewer node, creating reviews from both profiles and human input
    3. **Edit Article Workflow**: Implementing a workflow that reviews and edits entire articles based on human feedback and expected requirements
    4. **Edit Selected Text Workflow**: Building focused editing for specific article sections, enabling precise changes without affecting other parts
    5. **MCP Server Integration**: Serving Brown with tools, prompts, and resources while respecting clean architecture principles
    6. **Cursor Integration**: Creating a coding-like writing experience with human-in-the-loop editing

- **Ideas on how you can further extend this code:**
    1. **Hook Brown to Claude Desktop**: Instead of using Cursor, integrate Brown with Claude Desktop for a different AI assistant experience
    2. **Use Resource Templates**: Parameterize the writing profiles and easily add support for all available profiles
    3. **Serve Brown through FastAPI**: Replace the MCP server with a FastAPI REST API for web-based integrations
    4. **Add Guideline Review Tool**: Create a workflow to review and edit your article guidelines themselves

- **Wrapping Up Brown:** With this lesson, we've wrapped up our exploration of the Brown writing workflow covering:
    - **Lesson 22**: Foundation with entities, nodes, orchestrator-worker pattern, and context engineering
    - **Lesson 23**: The evaluator-optimizer pattern with article reviews and iterative refinement
    - **Lesson 24**: Human-in-the-loop with MCP server integration

- **Transition to future lessons:** In Part 2D, we'll explore how to orchestrate Nova and Brown as a multi-agent system within a single MCP client, automating the whole research-to-article workflow. Then in Part 3, we'll cover production engineering practices like AI Evals, monitoring, deployment, and more.

- **Section length:** 350 words

## Lesson Code

Links to code that will be used to support the lesson. Always prioritize this code over every other piece of code found in the sources:

1. [Notebook](https://github.com/towardsai/agentic-ai-engineering-course/blob/dev/lessons/24_human_in_the_loop/notebook.ipynb)


## Golden Sources

1. [Andrej Karpathy: Software Is Changing (Again)](https://www.youtube.com/watch?v=LCEmiRjPEtQ)
2. [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)

## Documentation

1. [The FastMCP Server](https://gofastmcp.com/servers/server)
2. [LangGraph Functional API](https://docs.langchain.com/oss/python/langgraph/functional-api)
3. [Use the functional API](https://docs.langchain.com/oss/python/langgraph/use-functional-api)
