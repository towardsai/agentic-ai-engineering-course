## Global Context

- **What I’m planning to share:** In this article we will talk about structure outputs. We'll introduce **Structured Outputs** as a way to achieve reliable data extraction, ensuring type checks and data quality checks. Also, by working with Python dictionaries or Pydantic objects it makes manipulating the LLM outputs a lot easier as it natively acts as the bridge between the LLM (software 3.0) and Python (software 1.0) worlds. First we will implement structured outputs from scratch, then show how to do it with a popular API such as Gemini.
- **Why I think it’s valuable:** For an AI Engineer, it's critical to understand how to implement the bridge between the LLM and Python worlds, which is usually done through structured outputs. These are easily to parse, manipulate programmaticaly, interpret, validate, monitor and debug.
- **Who the intended audience is:** Aspiring AI Engineers learning for the first time about LLM structured outputs.
- **Expected length of the article in words** (where 200-250 words ~= 1 minute of reading time): 1500 words (7-10 minutes reading time)


## Outline

1. Understanding why agents need structured outputs
2. Implementing structured outputs from scratch using JSON
3. Implementing structured outputs from scratch using Pydantic
4. Implementing structured ouputs using Gemini and Pydantic


## Section 1: Understanding why agents need structured outputs

- First we want to highlight at a theoretical level why we need structured ouputs when integrating LLMs in our application.
- Benefits:
    - Greatly improves reliability when needing to extract specific pieces of information or structured data from an LLM's free-text response.
    - Reduces the need for fragile regex or string parsing.
- Use Cases:
    - Extracting entities from text (names, dates, locations).
    - Formatting LLM output into a predefined data structure for downstream processing.
    - Can be used as an alternative to function calling when the goal is just data extraction, not necessarily an action.
- Generate a mermaid diagram to support the idea.
-  **Section length:** 400 words (without counting the mermaid diagram)


## Section 2: Implementing structured outputs from scratch using JSON

- To support our theory section from above and fully understand the behind the scenes, we want to show to implement structured outputs from scratch using JSON schemas into our prompt templates. Explain that we will first show to implement structured outputs from scratch and then show how to configure it using Gemini.
- Explain how to enforce the LLM to output data structures in JSON format through prompt engineering and how to parse them to transform them into Python dicts that can be used within the code
- Give a quick note on the fact that is good practice to wrap specific context elements with XML tags for easy parsing.
- Provide a quick note that instead of JSON, which is extremely verbose, we can also use YAML or XML to make the generated outputs use less tokens
- Give step-by-step examples from section `2. Implementing structured outputs from scratch using JSON` of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
-  **Section length:** 400 words (without counting the code)


## Section 3: Implementing structured outputs from scratch using Pydantic

- Rewrite the section from above using Pydantic schemas to model the structured outputs
- Explain that Pydantic objects are the go-to method to model structured outputs as they offer field and type checking bypassing the ambiguity of Python dictionaries. 
- As a side note, also explain that Pydantic objects are the defacto method to model domain objects in Python acting as the perfect bridge betwee nthe LLM and Python world.
- Give step-by-step examples from section `3. Implementing structured outputs from scratch using Pydantic` of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
-  **Section length:** 300 words (without counting the code)


## Section 4: Implementing structured ouputs using Gemini and Pydantic

- Explain that so far we focused on writing the structured outputs from scratch, but when working with specific APIs we want to leverage their structured outputs functionality as it's usually more accurate and cheaper to use.
- As a more industry-level example, explain how we can directly enforce the Gemini API to output Pydantic objects
- Give step-by-step examples from section `4. Implementing structured outputs using Gemini and Pydantic` of the provided Notebook. Follow the code flow from the Notebook, highlighting each code cell step by step, while utilizing the markdown/text cells for inspiration.
-  **Section length:** 200 words (without counting the code)

## References

