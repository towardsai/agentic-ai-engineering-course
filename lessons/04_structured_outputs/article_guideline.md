## Global Context of the Lesson

### What We Are Planning to Share

In this article we will talk about structured outputs. We'll introduce **Structured Outputs** as a way to achieve reliable, controlled and formatted data extraction, ensuring type and data quality checks. Common structured output types are JSON, XML, YAML, which can further be translated into Python dictionaries, lists, classes or most commonly Pydantic models. This approach makes manipulating LLM outputs much easier as it serves as the bridge between the LLM (software 3.0) and Python (software 1.0) worlds. First we will implement structured outputs from scratch, then show how to do it directly with a popular API such as Gemini. Ultimately, we will discuss pros and cons between implementing from scratch versus using the LLM API directly.

### Why We Think It's Valuable

For an AI Engineer, it's critical to understand how to implement the bridge between the LLM and Python worlds, which is usually done through structured outputs. Structured outputs are easy to parse, manipulate programmatically, interpret, monitor, and debug. Most importantly, when using Pydantic, they add out-of-the-box data quality checks which are critical in Python and especially in the LLM world.

### Expected Length of the Lesson

**2000 words** (without the titles and references), where we assume that 200-250 words ≈ 1 minute of reading time.

### Theory / Practice Ratio

20% theory - 80% hands-on examples

## Achoring the Lesson in the Course

### Details About the Course

This piece is part of a broader course on AI agents and LLM workflows. The course consists of 3 parts, each with multiple lessons. 

Thus, it's essential to always anchor this piece into the broader course, understanding where the reader is in its journey. You will be careful to consider the following:
- The points of view
- To not reintroduce concepts already thought in the previous lesson.
- To be careful when talking about concepts introduced only in future lessons
- To always reference previous and future lessons when discussing topics outside the piece's scope.

### Lesson Scope

This is lesson 4 (from part 1) of the course on LLM structured ouputs.

### Point of View
The course is created by a team writing for a single reader, also known as the student. Thus, for voice consistency across the course, we will always use 'we,' 'our,' and 'us' to refer to the team who creates the course, and 'you' or 'your' to address the reader. Avoid singular first person and don't use 'we' to refer to the student.

Examples of correct point of view:
- Instead of "Before we can choose between workflows and agents, we need a clear understanding of what they are." word it as "To choose between workflows and agents, you need a clear understanding of what they are."

### Who Is the Intended Audience

Aspiring AI engineers who are learning aboutLLM structured outputs. for the first time.

### Concepts Introduced in Previous Lessons

In previous lessons of the course, we introduced the following concepts:

**Part 1:**

- **Lesson 1 - AI Engineering & Agent Landscape**: Understanding the role, the stack, and why agents matter now
- **Lesson 2 - Workflows vs. Agents**: Grasping the crucial difference between predefined logic and LLM-driven autonomy
- **Lesson 3 - Context Engineering**: The art of managing information flow to LLMs

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

**Part 3:**

- evaluations


### Anchoring the Reader in the Educational Journey

Within the course we are teaching the reader multiple topics and concepts. Thus, understanding where the reader is in it's educational journey it's critical for this piece. You have to use only previous introduced concepts, while being reluctant about suing concepts that haven't been introduced yet.

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

1. Section 1 - Introduction
2. Section 2: Understanding why structured outputs are critical
3. Section 3: Implementing structured outputs from scratch using JSON
4. Section 4: Implementing structured outputs from scratch using Pydantic
5. Section 5: Implementing structured outputs using Gemini and Pydantic
6. Section 6 - Conclusion: Structured Outputs Are Everywhere


## Section 1 - Introduction
(What problem are we learning to solve? Why is it essential to solve it?)

- **Quick reference to what we've learned in previous lessons:** Take the core ideas of what we've learned in previous lessons from the `Concepts Introduced in Previous Lessons` subsection of the `Anchoring the Lesson in the Course` section.
- **Transition to what we'll learn in this lesson:** After presenting what we learned in the past, make a transition to what we will learn in this lesson. Take the core ideas of the lesson from the `What We Are Planning to Share` subsection and highlight the importance and existence of the lesson from the `Why We Think It's Valuable` subsection of the `Global Context of the Lesson` section.
-  **Section length:** 100 words

## Section 2: Understanding why structured outputs are critical

- Before digging into the implementation, we want to highlight at a theoretical level why we need structured outputs when integrating LLM workflows or AI agents in our applications.
- Explore benefits such as:
    1. Easy to parse, manipulate programmatically, interpret, monitor, and debug. Specifically, we can easily work with Python data structures and access attributes of interest.
    2. Most importantly, when using Pydantic, they add out-of-the-box data quality checks which are critical in Python and especially in the LLM world.
    3. Reduces the need for fragile regex or string parsing.
- Conclude the benefits by stating that structured outputs create a clear contract between the LLM (software 3.0) and the rest of the code which requires rigid interfaces (software 1.0).
- Use Cases:
    1. Extracting entities from text (names, dates, locations, tags, keywords). For example, this is a precursor strategy for creating knowledge graphs when implementing GraphRAG.
    2. Formatting LLM output into a predefined data structure for downstream processing to easily manipulate the outputs such as transforming the data and/or filtering out redundant context
    (Generate a mermaid diagram illustrating concept 2)
    3. Data and type validation.
- To transition from theory to practice quickly mention that we will show the reader how to implemented structured outputs in the following ways:
    1. From scratch using JSON
    2. From scratch using Pydantic
    3. Using Gemini SDK and Pydantic
-  **Section length:** 300 words (without counting the mermaid diagram)


## Section 3: Implementing structured outputs from scratch using JSON

- To support our theory section from above and fully understand what happens behind the scenes, we will first implement structured outputs from scratch by prompting the model to output JSON structures.
- Using the code examples from the provided Notebook within the <research>, use all the code from the `2. Implementing structured outputs from scratch using JSON` section to explain how to force the LLM to output data structures in JSON format through prompt engineering and how to parse them to Python dictionaries that can be used within the code.
- Here is how you should use the code from the `2. Implementing structured outputs from scratch using JSON` section of the provided Notebook:
    - Group 1:
        1. Define the Gemini `client` and `MODEL_ID` constant
        1. Show the example `DOCUMENT`
    - Group 2:
        1. Explain the prompt (Quick note on how we wrapped up the JSON exaample and document context within XML tags)
        2. Call the model
        3. Print the output
    - Group 3:
        1. Explain the `extract_json_from_response` Python func
        2. Extract the output from the mode response 
        3. Print the extracted output
-  **Section length:** 300 words (without counting the code)


## Section 4: Implementing structured outputs from scratch using Pydantic

- Now we will show to return structured outputs as Pydantic models instead of raw JSONs or Python dictionaries
- Before going into the code, explain why Pydantic are the go-to method for modeling structured outputs instead of JSON or dictionaries:
    1. 

- Explain that Pydantic objects are the go-to method for modeling structured outputs as they offer field and type checking out-of-the-box. This approach bypasses the ambiguity of Python dictionaries.
- **Data quality example**: If an LLM returns a string instead of an integer or misses a required field, Pydantic raises a ValidationError with a clear explanation of what went wrong and where.
- Explain that Pydantic works hand in hand with Python's standard `typing` library, which is used to define the type from the signature of data structures, functions and classes. Using Pydantic and `typing`, we can enforce the structure (i.e., the data structure's layout) and type for even the most complex data structures.
- Show how we can directly leverage the schema from the Pydantic model to guide the model to output the right data structure. Highlight that similar techniques are used internally by Gemini or OpenAI.
- Other popular options are Python's TypeDicts and DataClass classes. But due to Pydantic out-of-the-box validation mechanisms Pydantic is the most popular and powerful. 
- Explain that Pydantic objects are the de facto method for modeling domain objects in Python, acting as the perfect bridge between the LLM and Python worlds.
- Give step-by-step examples from section `3. Implementing structured outputs from scratch using Pydantic` of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
-  **Section length:** 450 words (without counting the code)

## Section 5: Implementing structured outputs using Gemini and Pydantic

- Explain that so far we focused on implementing structured outputs from scratch, but when working with specific APIs we want to leverage their structured outputs functionality.
- Discuss pros and cons between implementing from scratch versus using the LLM API directly, such as how using the native structured outputs from the LLM API is more accurate and cheaper to use.
- As a more industry-level example, explain how we can directly configure the Gemini API to output Pydantic objects.
- Give step-by-step examples from section `4. Implementing structured outputs using Gemini and Pydantic` of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
-  **Section length:** 300 words (without counting the code)

## Section 6 - Conclusion: Structured Outputs Are Everywhere
(Connect our solution to the bigger field of AI Engineering. Add course next steps.)

- To emphasize the importance of structure outputs emphasize that this pattern is used everywhere, regardless of what you are building (research, writer, coding agents), of what patterns you are using (agents, workflows) or what domain your are working with (finance, medicine, education).
- To transition from this lesson to the next, specify what we will learn in future lessons. First mention what we will learn in next lesson, which is Lesson 5. Next leverage the concepts listed in subsection `Concepts That Will Be Introduced in Future Lessons` to make slight references to other topics we will learn during this course. To stay relevant, specify that we will leverage structured outputs in almost all future lessons.
-  **Section length:** 100 words

## Article code

Links to code that will be used to support the article. Always prioritize this code over every other piece of code found in the sources: 

- [Notebook 1](https://github.com/towardsai/course-ai-agents/blob/main/lessons/04_structured_outputs/notebook.ipynb)

## Golden Sources

- [Gemini API Structured Output](https://ai.google.dev/gemini-api/docs/structured-output)
- [Structured Outputs with Pydantic & OpenAI Function Calling](https://www.youtube.com/watch?v=NGEZsqEUpC0)
- [Structured Outputs with OpenAI](https://platform.openai.com/docs/guides/structured-outputs)
- [Steering Large Language Models with Pydantic](https://pydantic.dev/articles/llm-intro)
- [How to return structured data from a model](https://python.langchain.com/docs/how_to/structured_output/)


## Other Sources

- [YAML vs. JSON: Which Is More Efficient for Language Models?](https://betterprogramming.pub/yaml-vs-json-which-is-more-efficient-for-language-models-5bc11dd0f6df)
