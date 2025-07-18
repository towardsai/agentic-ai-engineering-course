## Global Context

- **What I'm planning to share:** In this article we will talk about structured outputs. We'll introduce **Structured Outputs** as a way to achieve reliable, controlled and formatted data extraction, ensuring type and data quality checks. Common structured output types are JSON, XML, YAML, which can further be translated into Python dictionaries, lists, classes or most commonly Pydantic models. This approach makes manipulating LLM outputs much easier as it serves as the bridge between the LLM (software 3.0) and Python (software 1.0) worlds. First we will implement structured outputs from scratch, then show how to do it directly with a popular API such as Gemini. Ultimately, we will discuss pros and cons between implementing from scratch versus using the LLM API directly.
- **Why I think it's valuable:** For an AI Engineer, it's critical to understand how to implement the bridge between the LLM and Python worlds, which is usually done through structured outputs. Structured outputs are easy to parse, manipulate programmatically, interpret, monitor, and debug. Most importantly, when using Pydantic, they add out-of-the-box data quality checks which are critical in Python and especially in the LLM world.
- **Who the intended audience is:** Aspiring AI Engineers learning for the first time about LLM structured outputs.
- **Expected length of the article in words** (where 200-250 words ~= 1 minute of reading time): 2000 words (8-10 minutes reading time)


## Outline

1. Understanding why structured outputs are critical
2. Implementing structured outputs from scratch using JSON
3. Implementing structured outputs from scratch using Pydantic
4. Implementing structured outputs using Gemini and Pydantic


## Section 1: Understanding why structured outputs are critical

- Before digging into the implementation, we want to highlight at a theoretical level why we need structured outputs when integrating LLMs or agents in our applications.
- Explore benefits such as:
    - Easy to parse, manipulate programmatically, interpret, monitor, and debug. Specifically, we can easily work with Python data structures and access attributes of interest.
    - Most importantly, when using Pydantic, they add out-of-the-box data quality checks which are critical in Python and especially in the LLM world.
    - Reduces the need for fragile regex or string parsing.
- Use Cases:
    - Extracting entities from text (names, dates, locations, tags, keywords). For example, this is a precursor for GraphRAG.
    - Formatting LLM output into a predefined data structure for downstream processing.
    - Data and type validation.
- Generate a mermaid diagram to support the ideas from the section.
-  **Section length:** 550 words (without counting the mermaid diagram)


## Section 2: Implementing structured outputs from scratch using JSON

- To support our theory section from above and fully understand what happens behind the scenes, we want to show how to implement structured outputs from scratch by prompting the model to output JSON structures. Explain that we will first show how to implement structured outputs from scratch and then show how to configure it using Gemini.
- Explain how to force the LLM to output data structures in JSON format through prompt engineering and how to parse them to transform them into Python dictionaries that can be used within the code.
- Give a quick note that it is good practice to wrap specific context elements with XML tags for easy parsing.
- Also, specify that instead of JSON, which can be verbose, we can also use YAML or XML to make the generated outputs use fewer tokens.
- Give step-by-step examples from section `2. Implementing structured outputs from scratch using JSON` of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration. 
-  **Section length:** 500 words (without counting the code)


## Section 3: Implementing structured outputs from scratch using Pydantic

- Rewrite the section from above using Pydantic schemas to model the structured outputs.
- Explain that Pydantic objects are the go-to method for modeling structured outputs as they offer field and type checking out-of-the-box. This approach bypasses the ambiguity of Python dictionaries. 
- Explain that Pydantic works hand in hand with Python's standard `typing` library, which is used to define the type from the signature of data structures, functions and classes. Using Pydantic and `typing`, we can enforce the structure (i.e., the data structure's layout) and type for even the most complex data structures.
- Show how we can directly leverage the schema from the Pydantic model to guide the model to output the right data structure. Highlight that similar techniques are used internally by Gemini or OpenAI.
- Other popular options are Python's TypeDicts and DataClass classes. But due to Pydantic out-of-the-box validation mechanisms Pydantic is the most popular and powerful. 
- Explain that Pydantic objects are the de facto method for modeling domain objects in Python, acting as the perfect bridge between the LLM and Python worlds.
- Give step-by-step examples from section `3. Implementing structured outputs from scratch using Pydantic` of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
-  **Section length:** 450 words (without counting the code)


## Section 4: Implementing structured outputs using Gemini and Pydantic

- Explain that so far we focused on implementing structured outputs from scratch, but when working with specific APIs we want to leverage their structured outputs functionality.
- Discuss pros and cons between implementing from scratch versus using the LLM API directly, such as how using the native structured outputs from the LLM API is more accurate and cheaper to use.
- As a more industry-level example, explain how we can directly configure the Gemini API to output Pydantic objects.
- Give step-by-step examples from section `4. Implementing structured outputs using Gemini and Pydantic` of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
-  **Section length:** 300 words (without counting the code)


## Article code

Links to code that will be used to support the article. Always prioritize this code over every other piece of code found in the sources: 

- [Notebook 1](https://github.com/towardsai/course-ai-agents/blob/main/lessons/05_structured_outputs/notebook.ipynb)

## Golden Sources

- [Gemini API Structured Output](https://ai.google.dev/gemini-api/docs/structured-output)
- [Structured Outputs with Pydantic & OpenAI Function Calling](https://www.youtube.com/watch?v=NGEZsqEUpC0)
- [Structured Outputs with OpenAI](https://platform.openai.com/docs/guides/structured-outputs)
- [Steering Large Language Models with Pydantic](https://pydantic.dev/articles/llm-intro)
- [How to return structured data from a model](https://python.langchain.com/docs/how_to/structured_output/)


## Other Sources

- [YAML vs. JSON: Which Is More Efficient for Language Models?](https://betterprogramming.pub/yaml-vs-json-which-is-more-efficient-for-language-models-5bc11dd0f6df)
