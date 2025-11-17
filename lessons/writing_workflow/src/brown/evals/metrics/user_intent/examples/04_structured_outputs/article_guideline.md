## Global Context of the Lesson

### What We Are Planning to Share

In this lesson, we will talk about structured outputs. We'll introduce **Structured Outputs** as a way to achieve reliable, controlled, and formatted data extraction, ensuring type and data quality checks. Common structured output types are JSON, XML, and YAML, which can further be translated into Python dictionaries, lists, classes, or most commonly Pydantic models. This approach makes manipulating LLM outputs much easier as it serves as the bridge between the LLM (software 3.0) and Python (software 1.0) worlds. First, we will implement structured outputs from scratch, then show how to do it directly with a popular API such as Gemini. Ultimately, we will highlight how structured outputs are used everywhere when building LLM workflows or AI agents.

### Why We Think It's Valuable

For an AI Engineer, it's critical to understand how to implement the bridge between the LLM (software 3.0) and Python (software 1.0) worlds, which is usually done through structured outputs. Structured outputs are easy to parse, manipulate programmatically, interpret, monitor, and debug. Most importantly, when using Pydantic, they add out-of-the-box data structure and quality checks which are critical in Python and especially in the LLM world.

### Expected Length of the Lesson

**1800 words** (without the titles and references), where we assume that 200-250 words ≈ 1 minute of reading time.

### Theory / Practice Ratio

20% theory - 80% hands-on examples

## Achoring the Lesson in the Course

### Details About the Course

This piece is part of a broader course on AI agents and LLM workflows. The course consists of 3 parts, each with multiple lessons. 

Thus, it's essential to write as for a course and always anchor this piece into the broader course, understanding where the reader is in their journey. You will be careful to consider the following:
- The points of view
- To not reintroduce concepts already taught in the previous lesson.
- To be careful when talking about concepts introduced only in future lessons
- To always reference previous and future lessons when discussing topics outside the piece's scope.

### Lesson Scope

This is lesson 4 (from part 1) of the course on LLM structured outputs.

### Point of View
The course is created by a team writing for a single reader, also known as the student. Thus, for voice consistency across the course, we will always use 'we,' 'our,' and 'us' to refer to the team who creates the course, and 'you' or 'your' to address the reader. Avoid singular first person and don't use 'we' to refer to the student.

Examples of correct point of view:
- Instead of "Before we can choose between workflows and agents, we need a clear understanding of what they are." word it as "To choose between workflows and agents, you need a clear understanding of what they are."

### Who Is the Intended Audience

Aspiring AI engineers who are learning about LLM structured outputs for the first time.

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

Within the course, we are teaching the reader multiple topics and concepts. Thus, understanding where the reader is in their educational journey is critical for this piece. You have to use only previously introduced concepts, while being reluctant about using concepts that haven't been introduced yet.

When discussing the **concepts introduced in previous lessons** listed in the `Concepts Introduced in Previous Lessons` section, avoid reintroducing them to the reader. Especially don't reintroduce the acronyms. Use them as if the reader already knows what they are. 

Avoid using all the **concepts that haven't been introduced in previous lessons** listed in the `Concepts That Will Be Introduced in Future Lessons` subsection. Whenever another concept requires references to these banned concepts, instead of directly using them, use intuitive analogies and or explanations that are more general and easier to understand, as you would explain them to a 7-year-old. For example:
- If the "tools" concept wasn't introduced yet and you have to talk about agents, refer to them as "actions".
- If the "routing" concept wasn't introduced yet and you have to talk about it, refer to it as "guiding the workflow between multiple decisions".
You can use the concepts that haven't been introduced in previous lessons listed in the `Concepts That Will Be Introduced in Future Lessons` subsection only if we explicitly specify them. Still, even in that case, as the reader doesn't know how that concept works, you are just allowed to use the term, while keeping the explanation extremely high-level and intuitive, as if you were explaining it to a 7-year-old.
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
- **Section length:** 150 words

## Section 2: Understanding why structured outputs are critical

- Before digging into the implementation, we want to highlight at a theoretical level why we need structured outputs when integrating LLM workflows or AI agents in our applications.
- Explore benefits such as:
    1. Easy to parse, manipulate programmatically, interpret, monitor, and debug. Specifically, we can easily work with Python data structures and access attributes of interest.
    2. Most importantly, when using Pydantic, they add out-of-the-box data quality checks which are critical in Python and especially in the LLM world.
    3. Reduces the need for fragile regex or string parsing.
- Conclude the benefits by stating that structured outputs create a clear contract between the LLM (software 3.0) and the rest of the code which requires rigid interfaces (software 1.0). That's why they became the de facto method for modeling domain objects in Python, acting as the perfect bridge between the two worlds.
- Use Cases:
    1. Extracting entities from text (names, dates, locations, tags, keywords). For example, this is a precursor strategy for creating knowledge graphs when implementing GraphRAG.
    2. Formatting LLM output into a predefined data structure for downstream processing to easily manipulate the outputs such as transforming the data and/or filtering out redundant context
    (Generate a mermaid diagram illustrating concept 2)
    3. Data and type validation.
- To transition from theory to practice, quickly mention that we will show the reader how to implement structured outputs in the following ways:
    1. From scratch using JSON
    2. From scratch using Pydantic
    3. Using Gemini SDK and Pydantic
- **Section length:** 300 words (without counting the mermaid diagram)


## Section 3: Implementing structured outputs from scratch using JSON

- To support our theory section from above and fully understand what happens behind the scenes, we will first implement structured outputs from scratch by prompting the model to output JSON structures.
- Using the code examples from the provided Notebook within the <research> tag, use all the code from the `2. Implementing structured outputs from scratch using JSON` section to explain how to force the LLM to output data structures in JSON format through prompt engineering and how to parse them to Python dictionaries that can be used within the code.
- Here is how you should use the code from the `2. Implementing structured outputs from scratch using JSON` section of the provided Notebook along with other notes:
    1. Define the Gemini `client` and `MODEL_ID` constant
    2.  Show the example `DOCUMENT`
    3. Explain the prompt (Quick note on how we wrapped up the JSON example and document context with XML tags)
    4. Call the model
    5. Print and show the output from the model
    6. Explain the `extract_json_from_response` Python function
    7. Extract the output from the model response 
    8. Print and show the extracted output
- **Section length:** 300 words (without counting the code)


## Section 4: Implementing structured outputs from scratch using Pydantic

- Now we will show how to return structured outputs as Pydantic models instead of raw JSON or Python dictionaries
- Before going into the code, explain why Pydantic is the go-to method for modeling structured outputs instead of JSON or dictionaries. Explain that it offers field and type checking out-of-the-box, bypassing the ambiguity of Python dictionaries. 
- **Data quality example**: If an LLM returns a string instead of an integer or misses a field defined in the Pydantic model, Pydantic raises a `ValidationError` with a clear explanation of what went wrong and where.
- Using the code examples from the provided Notebook within the <research> tag, use all the code from the `3.  Implementing structured outputs from scratch using Pydantic` section to explain how to force the LLM to output data structures as Pydantic models.
- Here is how you should use the code from the `3.  Implementing structured outputs from scratch using Pydantic` section of the provided Notebook along with other notes:
    1. Define the `DocumentMetadata` Pydantic class
    2. A note on how Pydantic works hand in hand with Python's standard `typing` library, which is used to define the type of each attribute from the Pydantic model. Using Pydantic and `typing`, we can enforce the structure (i.e., the data structure's layout) and type out of the LLM output. Still, starting with Python 11, instead of explicitly having to import the types from `typing`, we can use them as is. For example, instead of doing `from typing import List` and define a variable as `tags: List[str]`, we can directly do `tags: list[str]`.
    3. We can even nest Pydantic objects, allowing us to model more complex data structures. Still, we have to be careful not to make them too complicated as it can confuse the LLM, making the process more error-prone.
    4. Provide an example, where we adapt the `DocumentMetadata` with `Summary` and `Tag` Pydantic objects showing how we can nest Pydantic models to create more complex data structures.
    5. Show how we can directly leverage the schema from the Pydantic model to guide the model to output the right data structure. usign the code from the `Injecting Pydantic Schema into the Prompt` subsection.
    6. To better define the terminology, give a note on how the standard term for defining how the output of the LLM looks is called a `schema` or a `contract`. 
    7. Output the schema of the `DocumentMetadata` model
    8. Highlight that similar techniques are used internally by Gemini or OpenAI to enforce the output schema
    9. Explain the new prompt used to extract the document metadata
    10. Call the model and extract the json response
    11. Print and show the output
    12. Map the JSON to the `DocumentMetadata` Pydantic model
    13. Print and show the output
    14. Final note: The core idea is to use Pydantic objects directly in downstream components throughout the code. Not obscure Python dicts where we don't know what's inside, having to pollute the code with if-else statements in case a key is missing or it doesn't has the right type.
 - Conclude the section with some other popular options which are Python's TypedDicts and DataClass classes. Still, these are used just to enforce structure and not the type, as if there is a type mismatch, they will not handle it. That's why due to Pydantic's out-of-the-box validation mechanisms, it became the most popular and powerful way to move data around in LLM workflows and AI agents.

- **Section length:** 600 words (without counting the code)

## Section 5: Implementing structured outputs using Gemini and Pydantic

- So far, we focused on implementing structured outputs from scratch, but when working with specific APIs, such as Gemini or OpenAI, we want to leverage their structured outputs functionality.
- Present some pros on why using the LLM API directly is better than implementing from scratch. Using the native structured outputs from the LLM API is easier, more accurate and cheaper to use.
- Using the code examples from the provided Notebook within the <research> tag, use all the code from the `4. Implementing structured outputs using Gemini and Pydantic` section to explain how to force the LLM to output data structures as Pydantic models using Gemini's SDK.
- Here is how you should use the code from the `4. Implementing structured outputs using Gemini and Pydantic` section of the provided Notebook along with other notes:
    1. Explain the new `GenerateContentConfig` config used to enforce the Pydantic output.
    2. Explain the new prompt. Highlight how small it is now.
    3. Call the model
    4. Show the output. Highlight how by accessing `response.parse`, we have access directly to the response wrapped in our Pydantic model
- **Section length:** 300 words (without counting the code)

## Section 6 - Conclusion: Structured Outputs Are Everywhere
(Connect our solution to the bigger field of AI Engineering. Add course next steps.)

- To emphasize the importance of structured outputs, emphasize that this pattern is used everywhere, regardless of what you are building (research, writer, coding agents), what patterns you are using (agents, workflows), or what domain you are working with (finance, medicine, education).
- To transition from this lesson to the next, specify what we will learn in future lessons. First mention what we will learn in next lesson, which is Lesson 5. Next leverage the concepts listed in subsection `Concepts That Will Be Introduced in Future Lessons` to make slight references to other topics we will learn during this course. To stay relevant, specify that we will leverage structured outputs in almost all future lessons.
- **Section length:** 150 words

## Lesson Code

Links to code that will be used to support the lesson. Always prioritize this code over every other piece of code found in the sources: 

- [Notebook 1](https://github.com/towardsai/course-ai-agents/blob/main/lessons/04_structured_outputs/notebook.ipynb)

## Golden Sources

- [Gemini API Structured Output](https://ai.google.dev/gemini-api/docs/structured-output)
- [Structured Outputs with Pydantic & OpenAI Function Calling](https://www.youtube.com/watch?v=NGEZsqEUpC0)
- [Structured Outputs with OpenAI](https://platform.openai.com/docs/guides/structured-outputs)
- [Steering Large Language Models with Pydantic](https://pydantic.dev/articles/llm-intro)
- [How to return structured data from a model](https://python.langchain.com/docs/how_to/structured_output/)


## Other Sources

- [YAML vs. JSON: Which Is More Efficient for Language Models?](https://betterprogramming.pub/yaml-vs-json-which-is-more-efficient-for-language-models-5bc11dd0f6df)
