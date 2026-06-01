## Global Context of the Lesson

### What We Are Planning to Share

In this lesson, we will explore agent tool usage. We want to highlight that through tools, we can enable agents to take actions and interact with the external world. By implementing tool calling from scratch, we will understand how the LLM decides which tool to call, how it generates the correct parameters for the function, and how it calls the function. First, we will implement tool definition and calling from scratch, then show how to do it with a popular API such as Gemini. We'll also explore using structured outputs as tools through Pydantic models. Next, we'll examine scenarios requiring multiple tools and introduce tool chaining, before discussing the limitations that lead to more sophisticated patterns like ReAct. Finally, we'll look at the most essential tool categories such as memory access and web search through RAG or code execution. All our ideas will be supported by code.

### Why We Think It's Valuable

For an AI Engineer, tools are what transform an LLM from a simple text generator into an agent that can take actions into the external world. Tools are one of the key components and foundational blocks of an AI agent. Therefore, it's critical to open the black box and understand how an agent works with these tools to be able to build, improve, debug and monitor them.

### Expected Length of the Lesson

**3350 words** (without the titles and references), where we assume that 200-250 words ≈ 1 minute of reading time.

### Theory / Practice Ratio

20% theory - 80% hands-on examples

## Anchoring the Lesson in the Course

### Details About the Course

This piece is part of a broader course on AI agents and LLM workflows. The course consists of 4 parts, each with multiple lessons. 

Thus, it's essential to always anchor this piece into the broader course, understanding where the reader is in their journey. You will be careful to consider the following:
- The points of view
- To not reintroduce concepts already taught in the previous lesson.
- To be careful when talking about concepts introduced only in future lessons
- To always reference previous and future lessons when discussing topics outside the piece's scope.

### Lesson Scope

This is lesson 6 (from part 1) of the course on function calling and tools.

**Lesson 6 - Agent Tools & Function Calling**: Giving your LLM the ability to take action

### Point of View
The course is created by a team writing for a single reader, also known as the student. Thus, for voice consistency across the course, we will always use 'we,' 'our,' and 'us' to refer to the team who creates the course, and 'you' or 'your' to address the reader. Avoid singular first person and don't use 'we' to refer to the student.

Examples of correct point of view:
- Instead of "Before we can choose between workflows and agents, we need a clear understanding of what they are." word it as "To choose between workflows and agents, you need a clear understanding of what they are."

### Who Is the Intended Audience

Aspiring AI engineers who are learning about AI agent function calling and tools for the first time.

### Concepts Introduced in Previous Lessons

In previous lessons of the course, we introduced the following concepts:

**Part 1:**

- **Lesson 1 - AI Engineering & Agent Landscape**: Understanding the role, the stack, and why agents matter now
- **Lesson 2 - Workflows vs. Agents**: Grasping the crucial difference between predefined logic and LLM-driven autonomy
- **Lesson 3 - Context Engineering**: The art of managing information flow to LLMs
- **Lesson 4 - Structured Outputs**: Ensuring reliable data extraction from LLM responses
- **Lesson 5 - Basic Workflow Ingredients**: Implementing chaining, routing, parallel and the orchestrator-worker patterns

As this is only the 6th lesson of the course, we haven't introduced too many concepts. At this point, the reader still doesn't fully understand what an AI agent is, how they plan, reason, and how ReAct works.

### Concepts That Will Be Introduced in Future Lessons

In future lessons of the course, we will introduce the following concepts:

**Part 1:**

- **Lesson 7 - Planning & Reasoning**: Understanding patterns like ReAct (Reason + Act)
- **Lesson 8 - Implementing ReAct**: Building a reasoning agent from scratch
- **Lesson 9 - Agent Memory & Knowledge**: Short-term vs. long-term memory (procedural, episodic, semantic)
- **Lesson 10 - RAG Deep Dive**: Advanced retrieval techniques for knowledge-augmented agents
- **Lesson 11 - Multimodal Processing**: Working with documents, images, and complex data

**Part 2:**

In this section, you'll move from theory to practice by starting your work on the course's central project: an interconnected research and writing agent system. After a deep dive into agentic design patterns and a comparative look at modern frameworks, we'll focus on LangGraph. You will implement the research agent, equipping it with tools for web scraping and analysis. Then, you'll construct the writing workflow to convert research into polished content. Finally, you'll integrate these components, working on the orchestration of a complete, multi-agent pipeline from start to finish.

Other concepts from Part 2:
- MCP

**Part 3:**

With the agent system built, this section focuses on the engineering practices required for production. You will learn to design and implement robust evaluation frameworks to measure and guarantee agent reliability, moving far beyond simple demos. We will cover AI observability, using specialized tools to trace, debug, and understand complex agent behaviors. Finally, you’ll explore optimization techniques for cost and performance and learn the fundamentals of deploying your agent system, ensuring it is scalable and ready for real-world use.

**Part 4:**

In this final part of the course, you will build and submit your own advanced LLM agent, applying what you've learned throughout the previous sections. We provide a complete project template repository, enabling you to either extend our agent pipeline or build your own novel solution. Your project will be reviewed to ensure functionality, relevance, and adherence to course guidelines for the awarding of your course certification.

As AI agent tools are the core foundation of AI engineering, we will have to introduce new terms, but we will discuss them in a highly intuitive manner, being careful not to confuse the reader with too many terms that haven't been introduced yet in the course.

### Anchoring the Reader in the Educational Journey

Within the course we are teaching the reader multiple topics and concepts. Thus, understanding where the reader is in their educational journey is critical for this piece. You have to use only previously introduced concepts, while being reluctant about using concepts that haven't been introduced yet.

When discussing the **concepts introduced in previous lessons** listed in the `Concepts Introduced in Previous Lessons` section, avoid reintroducing them to the reader. Especially don't reintroduce the acronyms. Use them as if the reader already knows what they are. 

Avoid using all the **concepts that haven't been introduced in previous lessons** listed in the `Concepts That Will Be Introduced in Future Lessons` subsection. Whenever another concept requires references to these banned concepts, instead of directly using them, use intuitive analogies or explanations that are more general and easier to understand, as you would explain them to a 7-year-old. For example:
- If the "tools" concept wasn't introduced yet and you have to talk about agents, refer to them as "actions".
- If the "routing" concept wasn't introduced yet and you have to talk about it, refer to it as "guiding the workflow between multiple decisions".
You can use the concepts that haven't been introduced in previous lessons listed in the `Concepts That Will Be Introduced in Future Lessons` subsection only if we explicitly specify them. Still, even in that case, as the reader doesn't know how that concept works, you are just allowed to use the term, while keeping the explanation extremely high-level and intuitive, as if you were explaining it to a 7-year-old.
Whenever you use a concept from the `Concepts That Will Be Introduced in Future Lessons` subsection, explicitly specify in what lesson it will be explained in more detail, leveraging the particulars from the subsection. If not explicitly specified in the subsection, simply state that we will cover it in future lessons without providing a concrete lesson number. 

In all use cases avoid using acronyms that aren't explicitly stated in the guidelines. Rather use other more accessible synonyms or descriptions that are easier to understand by non-experts.

## Narrative Flow of the Lesson

Follow the next narrative flow when writing the end-to-end lesson:

- What problem are we learning to solve? Why is it essential to solve it?
    - Start with a personal story where we encountered the problem
- Why other solutions are not working and what's wrong with them
- At a theoretical level, explain our solution or transformation. Highlight:
    - The theoretical foundations
    - Why is it better than other solutions?
    - What tools or algorithms can we use?
- Provide some hands-on examples
- Go deeper into the advanced theory
- Provide a more complex example supporting the advanced theory
- Connect our solution to the bigger field of AI Engineering. Add course next steps

## Lesson Outline

1. Section 1: Introduction
2. Section 2: Understanding why agents need tools
3. Section 3: Implementing tool calls from scratch
4. Section 4: Implementing a small tool calling framework from scratch
5. Section 5: Implementing production-level tool calls with Gemini
6. Section 6: Using Pydantic models as tools for on-demand structured outputs
7. Section 7: The downsides of running tools in a loop
8. Section 8: Going through popular tools used within the industry
9. Section 9: Conclusion

## Section 1: Introduction
(What problem are we learning to solve? Why is it essential to solve it?)

- **Quick reference to what we've learned in previous lessons:** Take the core ideas of what we've learned in previous lessons from the `Concepts Introduced in Previous Lessons` subsection of the `Anchoring the Lesson in the Course` section
- **Transition to what we'll learn in this lesson:** After presenting what we learned in the past, make a transition to what we will learn in this lesson. Take the core ideas of the lesson from the `What We Are Planning to Share` subsection and highlight the importance and existence of the lesson from the `Why We Think It's Valuable` subsection of the `Global Context of the Lesson` section
- **Section length:** 100 words

## Section 2: Understanding why agents need tools
(Theoretical explanation of the problem we are solving and the solution)

- Before showing how to implement tool calling from scratch and with Gemini, explain in more depth why LLMs need tools in the first place.
- LLMs have one fundamental limitation. They are simple pattern matchers and text generators. They cannot, by themselves, perform actions to interact with the external world on their own. They need help through some additional engineering around them. This is where tools kick in.
- Analogy: The LLM is the brain, while the tools are an LLM's "hands and senses," allowing it to perceive and act in the world beyond its textual interface.
- Thus, tools are the bridge between the LLM's internal reasoning and the external world. With the power of tools, the LLM becomes an AI agent that can interact with the environment and execute specific instructions.
- Use a representative image from the research to explain at a high level how tools work.
- Examples of popular tools that power modern AI agents: 
    - Access real-time information through APIs (e.g., today's weather, latest news).
    - Interact with external databases or other storage solutions (PostgreSQL database, Snowflake data warehouse, S3 data lake)
    - Access agent's long-term memory to remember information beyond their context window
    - Execute code (Python, JavaScript)
    - Perform precise calculations beyond their training data (basic math calculations, sorting, filtering, grouping, etc.)
- **Section length:** 300 words

## Section 3: Implementing tool calls from scratch

- The best way to understand how tools work and how LLMs use them is by implementing them from scratch. That's why the rest of this lesson will focus on showing you how to implement a simple tools framework from scratch and then how to use them with modern LLM APIs such as Gemini
- Before going into the code, quickly list what we will learn in this section, such as how a tool is defined, how their schema looks like, how the LLM discovers what tools are available, how to call them through function schemas, and how we interpret their output.
- Next provide a summary of our end goal, which is to provide the LLM with a list of available tools and let it decide which one to use along with generating the correct arguments necessary to call the tool (which is usually a function). At a high level, the process of calling a tool looks as follows:
    1. **App:** Within the system prompt provide a list of available tools.
    2. **LLM:** Responds with a `function_call` request, specifying the tool and arguments.
    3. **App:** Execute the requested function in your code.
    4. **App:** Send the function's output back to the LLM.
    5. **LLM:** Generate a user-facing response.
- Provide a mermaid diagram illustrating the 5 steps from above, highlighting the request-execute-respond flow of calling a tool. Use our 3 available tools as examples (`search_google_drive`, `send_discord_message` and `summarize_report`)
- Now, let's dig into the code. To make it interesting, we will implement a simple example where we mock searching documents on Google Drive and sending their summaries to Discord.
- Using the code examples from the provided Notebook within the <research> tag, use all the code from the `2. Implementing tool calls from scratch` section to explain how to implement AI tool calls from scratch
- Here is how you should use the code from the `2. Implementing tool calls from scratch` section of the provided Notebook along with other notes:
    1. Define the imports, Gemini `client`, `MODEL_ID`, and `DOCUMENT` (specify that this is used to mock PDF documents) constants.
    2. Define the 3 tools as 3 separate blocks (`search_google_drive`, `send_discord_message` and `summarize_financial_report`). Highlight that to keep the code simple and focus on the tools implementation, all the tools are mocked.
    3. Define the schemas of the 3 tools as 3 separate code blocks (`search_google_drive_schema`, `send_discord_message_schema` and `summarize_financial_report_schema`). The schema is often described in JSON. Specify that this schema is used as input to the LLM to understand what tool to use (based on the description) and how to call it (based on the parameters that contain the variable names, their types, description of each variable and if it's required or optional).
    **Side note:** Explain that this schema is the industry standard when working with modern LLM providers such as OpenAI or Gemini.
    4. Define the `TOOLS` registry along their `TOOLS_BY_NAME` and `TOOLS_SCHEMA` mappings
    5. Show how the `TOOLS_BY_NAME` mapping looks like when it's called.
    6. Show how the first element from the `TOOLS_SCHEMA` look like.
    7. Define and explain the `TOOL_CALLING_SYSTEM_PROMPT`. Highlight the following:
        - The tool usage guidelines
        - The tool call format
        - The response behavior
        - The list of available tools enclosed by XML tags
    8. Theoretical section, where we explain in more detail how tool calling works in practice:
        - Based on the `description` field from the tool schema, the LLM *decides* if a tool call is appropriate to fulfill the user query. That's why writing clear and articulate tool descriptions it's extremely important for building successful AI agents. Also, when passing multiple tools it's critical for the descriptions to distinguish between each other to avoid any future confusion. For example two tools with descriptions `Tool used to search documents` and `Tool used to search files` would foncuse the LLM. You have to be as explicit as possible and define them as `Tool used to search documents on Google Drive` and `Tool used to search files on the disk`.
        - Another disambiguation method when working with AI agents is to be as clear as possible in the system prompts, verbosely stating what you need. For example, instead of saying `search documents`, be explicit and say where to search them from, such as `search documents on Google Drive`. 
        - By defining clear tool descriptions and system prompts, you ensure the AI agent will be able to make the necessary matches and call the right tools. This becomes crucial when scalling up to 50-100 tools / AI Agent. We will dig more into scaling methods in part 2 and 3 of the course.
        - Based on the selected tool, it then *generates* the function name and arguments as structured outputs such as JSON or Pydantic.
        - Add a quick note specifying that the LLM is specially tuned through instruction fine-tuning to interpret tool schema inputs and output tool calls.
    9. Explain how the `TOOL_CALLING_SYSTEM_PROMPT` works by calling the model with the first `USER_PROMPT`
    10. Output the result of the LLM. Highlight how the LLM outputs the function's name and arguments. Everything necessary to call the tool.
    11. Explain how the `TOOL_CALLING_SYSTEM_PROMPT` works with another example by calling the model with the second `USER_PROMPT`
    12. Output the result of the LLM
    13. Now, let's see how to interpret the output from the LLM. Start by extracting the tool call JSON string from the LLM response using the `extract_tool_call` function
    14. Transform the JSON string to a Python dictionary
    15. Now, let's execute the actual function. Start by getting the right tool handler from the `TOOLS_BY_NAME` dictionary
    16. Show that the tool handler is a reference to our `search_google_drive` function
    17. Call the tool
    18. Show the output from the tool
    19. Define the `call_tool` function that aggregates all the steps from above: extract Python dictionary, get tool handler, and call the tools. Using it, we can directly pass the LLM response and get the tool output.
    20. Show example on how to use the `call_tool` function. Highlight that the output is the same from step `18.`
    21. Conclude by showing and explaining that usually we use an LLM to interpret the tool output. To do so, the tool result is sent back to the LLM used to formulate a final response to the user or decide on the next step.
    22. Show the LLM response after it interprets the tool output
- Conclude by saying that this is the basic concept behind tool calling.
- **Section length:** 1000 words (without counting the code or mermaid diagram)

## Section 4: Implementing a tool calling framework from scratch

- Manually defining schemas for every tool we want to use can quickly become cumbersome and hard to scale. That's why all modern AI agents frameworks (e.g., LangGraph) or protocols (e.g., MCP) implement a `@tool` decorator (or something similar) that automatically computes and tracks the schemas of decorated functions.
- Thus, in our writing from scratch exercise, as a natural progression to defining the tool schemas manually, we will implement a small tool framework by building a `@tool` decorator that allows us to automatically compute the schema of each function that we want to use as a tool.
- The end goal is to decorate a function with the `@tool` decorated and based on the function's docstring and signature (input, output parameters and their types) to automatically create a tools registry similar to the `TOOLS` registry from the previous section
- This method also follows good software engineering principles, as we respect the Don't Repeat Yourself (DRY) software principle by having a modular central place to compute the tool schema. In other words, we have a single place that standardizes how we gather tool schemas
- Now, let's dig into the code and rewrite the implementation from the previous section using `@tool` decorators.
- Using the code examples from the provided Notebook within the <research> tag, use all the code from the `3. Implementing a tool calling framework from scratch` section to explain how to implement AI tool calls from scratch
- Here is how you should use the code from the `3. Implementing a tool calling framework from scratch` section of the provided Notebook along with other notes:
    1. Define the `ToolFunction` class. Explain that this class will aggregate the function schema.
    2. Define the `tools` registry and explain that it will be used to aggregate all our decorated tools.
    3. Define the `tool` function, which will be used to decorate future functions using the `@tool` syntax. Explain in more depth how Python decorators work.
    4. Redefine the 3 tools (`search_google_drive_example`, `send_discord_message_example`, `summarize_financial_report_example`) using the `@tool` Python decorator
    5. Show what the `tools` registry contains.
    6. Show the content of the first `search_google_drive_example` tool from the registry, inspecting the `tools` registry and showing it's:
        - name
        - type (highlight that now it's `ToolFunction` instead of a normal Python function)
        - schema (highlight that it's identical with the one manually defined by us)
        - functional handler (highlight that this is how we access the function handler now)
    7. Define the `tools_by_name` and `tools_schema` mappings. 
    8. Output the `tools_schema` content
    9. Call the LLM with the new `tools_schema` variable using the `USER_PROMPT` input
    10. Output the LLM response
    11. Call the tool from the LLM response using the `call_tool` function and output its result
- Voilà! We have our little tool calling framework. This implementation is similar to what LangGraph does under the scenes.
- **Section length:** 450 words (without counting the code or mermaid diagram)


## Section 5: Implementing production-level tool calls with Gemini

- Similar to what we did in Lesson 4 on structured outputs, after writing the tool calling implementation from scratch, we want to show you how you can leverage modern APIs such as Gemini's to make your life easier and your code more robust
- Rather than prompt engineering the LLM on what tools to pick and how to output tool calls in the right format, we will leverage Gemini's `GenerateContentConfig` to define all the available tools.
- Now, let's dig into the code and see how we can leverage Gemini's `GenerateContentConfig` config to use multiple tools.
- Using the code examples from the provided Notebook within the <research> tag, use all the code from the `4. Implementing production-level tool calls with Gemini` section to explain how to implement AI tool calls from scratch
- Here is how you should use the code from the `4. Implementing production-level tool calls with Gemini` section of the provided Notebook along with other notes:
    1. Define and explain the `tools` input and `config` object for Gemini. Highlight that at this point we still use the schemas defined above, but we will soon show how to skip this step.
    2. Call the LLM with the defined `config`. Highlight that we can completely skip defining the huge `TOOL_CALLING_SYSTEM_PROMPT` system prompt that guides the LLM how to use the tools and directly input the `USER_PROMPT`. Emphasize that this is more robust, as we are certain that the LLM is instructed the right way how to use the tools.
    3. Show the LLM response function call
    4. To simplify the implementation even more, and avoid computing the tool schemas manually or through a `@tool` decorator show how Google's genai Python SDK supports taking functions directly as input. Create a new `config` object by passing directly the `search_google_drive` and `send_discord_message` functions. Explain that the SDK automatically creates the schema based on the signature, type hints and pydocs of each function, as we did so far with our from scratch implementations.
    5. Call again the LLM and show the LLM response.
    6. Show how the `function_call` object returned by Gemini looks like
    7. Show how the `function_call.args` returned by Gemini looks like
    8. Show how we access the `tool_handler` based on the function's name from the `TOOLS_BY_NAME` registry (as before)
    9. Call the `tool_handler` manually.
    10. Show the output.
    11. Explain the `call_tool` implementation
    12. Call the `call_tool` function to call the `function_call` from the LLM response in one go
    13. Show the output of the function
    14. Wrap-up with the idea that by using Gemini's native SDK, we managed to reduce the tool implementation from dozens of lines of code to a few.
- Conclude the section by explaining that all popular APIs, along with Google's, such as OpenAI or Anthropic, use the same logic to instruct the LLM how to use different tools, but with minimal interface differences. Thus, what we learnt in this lesson can easily be extrapolated to your API of choice.
- **Section length:** 450 words (without counting the code)


## Section 6: Using Pydantic models as tools for on-demand structured outputs

- To further understand how we can leverage Google's genai Python SDK for function calling, connect this lesson with Lesson 4 on structured outputs and give a concrete, yet extremely popular example on how we can use a Pydantic model as a tool to generate structured outputs dynamically.
- Explain that adding the Pydantic model as a tool is an elegant way to get structured outputs in agentic scenarios, where you want to take multiple intermediate steps using unstructured outputs (which are easily interpreted by an LLM) and dynamically decide when to output the final answer in a structured form as a Pydantic model (which is easily interpreted in Python downstream steps and ensures the final output has the expected schema).
- Create a mermaid diagram with an AI agent that illustrates the concept from Section 6 takes calls multiple tools in a loop, where only the last one is tool call for structured outputs 
- Now, let's dig into the code and see how we can define Pydantic structured outputs as tools.
- Using the code examples from the provided Notebook within the <research> tag, use all the code from the `5. Using Pydantic models as tools for on-demand structured outputs` section to explain how to implement AI tool calls from scratch
- Here is how you should use the code from the `5. Using Pydantic models as tools for on-demand structured outputs` section of the provided Notebook along with other notes:
    1. Define the `DocumentMetadata` Pydantic class
    2. Define the `extraction_tool` object. Explain how we manually defined the function schema leveraging the `DocumentMetadata.model_json_schema()` as parameters similar to how we instructed the LLM to generate structured outputs in Lesson 4.
    3. Define the `config`
    4. Define the `prompt` and call the LLM
    5. Show the function name and arguments output as the response from the LLM
    6. Show we created the `document_metadata` Pydantic object and how it outputs `Validation successful!` if the output args have the right signature
- Conclude by highlighting that this pattern is often used in AI agents that require structured outputs
- **Section length:** 300 words (without counting the code)

## Section 7: The downsides of running tools in a loop

- Until now we focused only on making a single turn, calling a single tool.
- Still, as a natural progression, we want not only to call a single tool, but to build a more sophisticated versions where we allow an LLM to run them in a loop, chaining multiple tools and letting the LLM decide what tool to choose at each step based on the output of previous tools. This is the last piece of the puzzle we need to build a real AI agent! 
- Create a mermaid diagram illustrating the tool calling loop:
    - user prompt
    - tool call
    - tool result
    - tool call
    - tool result
    - ...
- Explain the benefits: flexibility, adaptability, and the ability to handle complex multi-step tasks.
- Now, let's dig into the code and see how we can define Pydantic structured outputs as tools.
- Using the code examples from the provided Notebook within the <research> tag, use all the code from the `6. The downsides of running tools in a loop` section to explain how to implement AI tool calls from scratch
- Here is how you should use the code from the `6. The downsides of running tools in a loop` section of the provided Notebook along with other notes:
    1. Define the new `config` object
    2. Define the user intent through the `USER_PROMPT`
    3. Implement the first LLM call
    4. Show the output of the LLM call
    4. Implement the sequential tool calling loop
    5. Show the outputs of the tool calling loop
- Continue the implementation with a short talk on the limitations of sequential tool loops and why this is not good practice:
    - Doesn't allow the LLM to interpret each tool output before deciding on the next steps.
    - Limited ability to plan (more on this in Lesson 7) ahead or consider multiple approaches as the agent immediately moves to the next function call without pausing to think about what it learned or whether it should change strategy.
    - Can lead to inefficient tool usage or getting stuck in loops.
- **Side Note**: To further optimize tool calling, when tools are independent of each other, we can run them in parallel:
    - Explain scenarios where multiple tools can be called simultaneously in parallel (e.g., fetching financial news and stock prices) as long they are independent from each other 
    - The core benefit is reduced latency
- To conclude, these limitations pushed the industry to develop more sophisticated patterns like **ReAct** (Reasoning and Acting), which we will explore in lessons 7 and 8.
- **Section length:** 400 words (without counting the code)

## Section 8: Popular tools used within the industry

- We want to wrap up the lesson by listing some popular tools that are used across the industry to ground the reader into the real-world and better understand of what's possible to build using tools:
- Here is the list, grouped based on functionality:
    1. **Knowledge & Memory Access:**
        - Tools that query vector databases, document stores, graph databases or other knowledge bases to retrieve relevant context for the LLM. 
        - We can take the idea even further and provide the LLM text-to-SQL tools that can construct SQL/NoSQL queries and interact with classic databases. This pattern is known as text-to-SQL.
        - All these tools are closely related to memory, RAG and agentic RAG, which we will discuss in more detail in Lessons 9 (memory) and 10 (RAG).
    2. **Web Search & Browsing:**
        - Tools that interface with search engine APIs (e.g., Google Search, Bing Search, Brave Search, etc.).
        - Web scraping tools that can fetch and parse content from web pages.
        - These types of tools are usually omnipresent in chatbots and research agents.
    3. **Code Execution:**
        - For example, a Python interpreter tool allows the agent to write and execute Python code in a sandboxed environment. Invaluable for calculations, data manipulation, statistics and data visualizations.
        - Even if Python is the most popular language used for code execution through tools, the pattern is often adapted to other popular languages such as JavaScript.
    4. **Other Popular Tools:**
        - Interacting with external APIs (e.g., calendar, email, project management, calls) omnipresent in enterprise AI apps
        - File system operations (read/write files, list directories) omnipresent in productivity AI apps that have to interact with our OS
- **Section length:** 250 words

## Section 9 - Conclusion: ...
(Connect our solution to the bigger field of AI Engineering. Add course next steps.)

- Conclude the article by highlighting that tool calling sits at the core of AI agents and it's probably the most important skill you need to deeply understand how to build, monitor, and debug AI apps.
- To transition from this lesson to the next, specify what we will learn in future lessons. First mention what we will learn in the next lesson, which is Lesson 7 on the theory behind planning and ReAct agents. Next leverage the concepts listed in subsection `Concepts That Will Be Introduced in Future Lessons` to make slight references to other topics we will learn during this course. To stay focused, specify only the ones that are present in this current lesson.
- **Section length:** 100 words

## Article code

Links to code that will be used to support the article. Always prioritize this code over every other piece of code found in the sources: 

- [Notebook 1](https://github.com/towardsai/course-ai-agents/blob/dev/lessons/06_tools/notebook.ipynb)

## Golden Sources

- [Function calling with the Gemini API](https://ai.google.dev/gemini-api/docs/function-calling)
- [Function calling with OpenAI's API](https://platform.openai.com/docs/guides/function-calling)
- [Tool Calling Agent From Scratch](https://www.youtube.com/watch?v=ApoDzZP8_ck)
- [Efficient Tool Use with Chain-of-Abstraction Reasoning](https://arxiv.org/pdf/2401.17464v3)
- [Building AI Agents from scratch - Part 1: Tool use](https://www.newsletter.swirlai.com/p/building-ai-agents-from-scratch-part)

## Other Sources

- [What is Tool Calling? Connecting LLMs to Your Data](https://www.youtube.com/watch?v=h8gMhXYAv1k)
- [ReAct vs Plan-and-Execute: A Practical Comparison of LLM Agent Patterns](https://dev.to/jamesli/react-vs-plan-and-execute-a-practical-comparison-of-llm-agent-patterns-4gh9)
- [Agentic Design Patterns Part 3, Tool Use](https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-3-tool-use/)
