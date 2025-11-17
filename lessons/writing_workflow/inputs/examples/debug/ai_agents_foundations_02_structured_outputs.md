# LLM Structured Outputs: The Only Way to Production AI
### From brute force to elegant, reliable prompts

In a recent project I am working on, our production AI system crashed right before an important demo. Why? Because we were not using structured outputs consistently across our Large Language Model (LLM) workflows. 

Our staging environment had been working fine with simple regex parsing of LLM responses, but when we deployed to production, everything fell apart. Our regex patterns failed to match slightly different response formats, data types were inconsistent, and downstream processes couldn't handle the unpredictable data, causing cascading failures. When demo day arrived, our system was completely unusable.

The problem was clear: we had been relying on fragile string parsing, hoping the LLM would always respond in the exact same format. But in production, especially with AI systems, users will always enter inputs you never expect. Without structured outputs, we had no data validation, no type checking, and no real control over how the output should look. Just like lock files ensure consistent dependencies, structured outputs ensure consistent AI data contracts by defining the expected structure for LLM responses.

In our previous article from the AI Agents Foundations series, we explored the difference between workflows and agents. Now, we will tackle a fundamental challenge: getting reliable information *out* of an LLM.

To understand exactly what happens, we will first write everything from scratch and then move to using popular LLM APIs such as Gemini's GenAI SDK:
1.  From scratch using JSON
2.  From scratch using Pydantic *(We love Pydantic!)*
3.  Using the Gemini SDK and Pydantic

## Understanding why structured outputs are important

Before we start coding, it is important to understand why structured outputs are foundational to building reliable AI applications. When an LLM returns a free-form string, you face the messy task of parsing it. This often involves fragile regular expressions or string-splitting logic that can easily break if the model outputs change slightly [[1]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11751965/), [[2]](https://arxiv.org/html/2506.21585v1). Structured outputs solve this by forcing the model’s response into a predictable format like JSON or Pydantic.

This approach offers several key benefits. First, structured outputs are easy to parse and manipulate programmatically. Instead of wrestling with raw text, you work with clean Python objects, making your code more predictable and easier to debug. Using libraries like Pydantic adds a layer of data and type validation [[3]](https://www.speakeasy.com/blog/pydantic-vs-dataclasses), [[4]](https://codetain.com/blog/validators-approach-in-python-pydantic-vs-dataclasses/). If the LLM returns a string where an integer is expected, your application raises a clear validation error immediately, preventing bad data from propagating.

Furthermore, structured outputs are easier to orchestrate between steps in a workflow. When you know what information you have available, it is much simpler to pass it to the next LLM call or a downstream system like a database or API [[5]](https://www.prompts.ai/en/blog-details/automating-knowledge-graphs-with-llm-outputs), [[6]](https://humanloop.com/blog/structured-outputs). This control also reduces costs. By ensuring the LLM generates only the necessary data without useless artifacts (e.g., "Here is the output you requested..."), you reduce the number of output tokens. 

💡 **Quick Tip:** You can easily compute the costs of running your workflows or agent by plugging in an LLMOps open-source tool such as [Opik](https://www.comet.com/site/?utm_source=newsletter&utm_medium=partner&utm_campaign=paul).

![Structured Outputs](structured_outputs.png)
Image 1: The benefits of structured outputs from LLMs, acting as a bridge between LLM (Software 3.0) and Python (Software 1.0) for downstream processing.

Ultimately, structured outputs create a formal contract between the LLM (Software 3.0) and your application code (Software 1.0). They are the standard method for modeling domain objects in AI engineering, connecting the probabilistic nature of LLMs with deterministic code.

## Implementing Structured Outputs From Scratch Using JSON

To understand how modern LLM APIs such as OpenAI and Gemini work under the hood, we will first implement structured outputs from scratch. 

Our goal is to prompt a model to return a JSON object and then parse it into a Python dictionary. We will use an “LLM-as-judge” evaluation as our example, where we ask an LLM to compare a generated text against a ground-truth document and score it based on predefined criteria. This is a great use case, as it requires extracting specific, structured information from a large context.

1.  First, we define our sample documents for the evaluation. These will serve as the input for our LLM judge:
    ```python
    GENERATED_DOCUMENT = """
    # Q3 2023 Financial Performance Analysis
    
    The Q3 earnings report shows a 20% increase in revenue and a 15% growth in user engagement, 
    beating market expectations. These impressive results reflect our successful product strategy 
    and strong market positioning.
    
    Our core business segments demonstrated remarkable resilience, with digital services leading 
    the growth at 25% year-over-year. The expansion into new markets has proven particularly 
    successful, contributing to 30% of the total revenue increase.
    
    Customer acquisition costs decreased by 10% while retention rates improved to 92%, 
    marking our best performance to date. These metrics, combined with our healthy cash flow 
    position, provide a strong foundation for continued growth into Q4 and beyond.
    """
    
    GROUND_TRUTH_DOCUMENT = """
    # Q3 2023 Financial Performance Analysis
    
    The Q3 earnings report shows a 18% increase in revenue and a 15% growth in user engagement, 
    slightly below market expectations. These results reflect our product strategy adjustments 
    and competitive market positioning challenges.
    
    Our core business segments showed mixed performance, with digital services growing at 
    22% year-over-year. The expansion into new markets has been challenging, contributing 
    to only 15% of the total revenue increase.
    
    Customer acquisition costs increased by 5% while retention rates remained at 88%, 
    indicating areas for improvement. These metrics, combined with our cash flow position, 
    suggest we need strategic adjustments for Q4 growth.
    """
    ```

2.  Next, we craft a prompt that instructs the LLM to evaluate the generated document against the ground truth and format the output as JSON. We provide a clear example of the desired structure and use XML tags like `<document>` to separate inputs from instructions. This is an effective prompt engineering technique for improving clarity and guiding the model's output [[7]](https://aws.amazon.com/blogs/machine-learning/structured-data-response-with-amazon-bedrock-prompt-engineering-and-tool-use/), [[8]](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api). The key is to be explicit about the format, keys, and value types you expect:
    ```python
    prompt = f"""
    You are an expert evaluator. Compare the generated document with the ground truth document and provide a score for each criterion.
    The output must be a single, valid JSON object with the following structure:
    {{
      "scores": [
        {{
          "criterion": "revenue_forecast",
          "score": 0 or 1,
          "reason": "Your reasoning here."
        }},
        {{
          "criterion": "user_growth",
          "score": 0 or 1,
          "reason": "Your reasoning here."
        }},
        {{
          "criterion": "facts",
          "score": 0 or 1,
          "reason": "Your reasoning here."
        }}
      ]
    }}
    
    Here are the documents:

    <generated_document>
    {GENERATED_DOCUMENT}
    </generated_document>

    <ground_truth_document>
    {GROUND_TRUTH_DOCUMENT}
    </ground_truth_document>
    """
    ```

3.  We send the prompt to the model and inspect the raw response. As expected, the model returns a JSON object, wrapped in Markdown *```json* code blocks:
    ```python
    from google import genai

    client = genai.Client()

    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    ```
    It outputs:
    ```json
    ```json
    {
      "scores": [
        {
          "criterion": "revenue_forecast",
          "score": 0,
          "reason": "The generated document claims a 20% revenue increase, while the ground truth states an 18% increase, which is slightly below expectations. The forecast is factually incorrect."
        },
        {
          "criterion": "user_growth",
          "score": 1,
          "reason": "Both documents report a 15% growth in user engagement, so this fact is correctly stated in the generated document."
        },
        {
          "criterion": "facts",
          "score": 0,
          "reason": "The generated document contains several factual inaccuracies regarding revenue, market expansion contribution, customer acquisition costs, and retention rates when compared to the ground truth."
        }
      ]
    }
    ```
    ```

4.  To handle this, we create a helper function to strip the Markdown tags, leaving a clean JSON string that can be safely parsed:
    ```python
    def extract_json_from_response(response: str) -> dict:
        """
        Extracts JSON from a response string that is wrapped in ```json tags.
        """
        response = response.replace("```json", "").replace("```", "")
        return json.loads(response)
    ```

5.  Finally, we parse the string into a Python dictionary, which can now be used in our application.
    ```python
    parsed_response = extract_json_from_response(response.text)
    ```
    It outputs:
    ```json
    {
      "scores": [
        {
          "criterion": "revenue_forecast",
          "score": 0,
          "reason": "The generated document claims a 20% revenue increase, ..."
        },
        {
          "criterion": "user_growth",
          "score": 1,
          "reason": "Both documents report a 15% growth in user engagement, ..."
        },
        {
          "criterion": "facts",
          "score": 0,
          "reason": "The generated document contains several factual ..."
        }
      ]
    }
    ```
This manual method works, but it relies on post-processing and lacks data validation. If the LLM makes a mistake like outputting a string instead of an integer or missing a dictionary key, our application will fail. Next, we will see how Pydantic provides a much more robust solution to this problem.

## Implementing Structured Outputs From Scratch Using Pydantic

Forcing JSON output is an improvement, but it still leaves you with a plain Python dictionary. You cannot be sure what is inside it, whether the keys are correct, or if the values have the right type. This uncertainty can lead to bugs and make your code difficult to maintain. 

Pydantic solves this problem. It is a data validation library that enforces structure and type hints at runtime, ensuring data integrity from the moment it enters your application [[3]](https://www.speakeasy.com/blog/pydantic-vs-dataclasses). It provides a single, clear definition for your data structure and can automatically generate a JSON Schema from your Python class.

**💡 Quick Tip**: I personally love Pydantic. I use it to model any data structure in my Python programs, completely dropping other options such as `@dataclass` or `TypedDict`. 

When an LLM produces output that does not match your Pydantic model, the library raises a `ValidationError` that clearly explains what went wrong. This "fail-fast" behavior is essential for building reliable systems, preventing bad data from moving through your application and causing hard-to-debug errors later. This is a major improvement over simple JSON parsing, as it introduces a validation layer that catches errors early.

1.  We define our desired data structure as a Pydantic class, using standard Python type hints to define the expected type for each field:
    ```python
    from typing import Literal
    from typing_extensions import Annotated
    
    import pydantic
    from pydantic import Ge, Le
    
    class CriterionScore(pydantic.BaseModel):
        """Model holding the score and reason for a specific criterion."""
        criterion: Literal["revenue_forecast", "user_growth", "facts"]
        score: Annotated[int, Ge(0), Le(1)] = pydantic.Field(description="Binary score of the section.")
        reason: str = pydantic.Field(description="The reason for the given score.")
    
    class Scores(pydantic.BaseModel):
        scores: list[CriterionScore]
    ```
    You can also nest Pydantic models to represent more complex, hierarchical data. This allows you to define intricate relationships between different pieces of information. However, it is good practice to keep schemas as simple as possible, as complex nested structures can confuse the LLM and lead to errors.

2.  With our Pydantic model defined, we can automatically generate a JSON Schema from it. A schema is the standard for defining the structure and constraints of your data, acting as a formal contract between your application and the LLM. This contract dictates the expected fields, their types, and any validation rules. Now, instead of providing a fuzzy JSON that explains how our output should look (as we did in the previous section), we provide an explicit schema to the LLM that is compatible with Pydantic. This is similar to the technique used internally by APIs like Gemini and OpenAI to enforce a specific output format [[9]](https://ai.google.dev/gemini-api/docs/structured-output):
    ```python
    schema = Scores.model_json_schema()
    ```
    The generated schema is detailed and includes descriptions from the `Field` definitions to guide the generation process.
    ```json
    {
        "$defs": {
            "CriterionScore": {
                "properties": {
                    "criterion": {
                        "enum": ["revenue_forecast", "user_growth", "facts"],
                        "title": "Criterion",
                        "type": "string"
                    },
                    "score": {
                        "description": "Binary score of the section.",
                        "exclusiveMaximum": 1,
                        "exclusiveMinimum": 0,
                        "title": "Score",
                        "type": "integer"
                    },
                    "reason": {
                        "description": "The reason for the given score.",
                        "title": "Reason",
                        "type": "string"
                    }
                },
                "required": ["criterion", "score", "reason"],
                "title": "CriterionScore",
                "type": "object"
            }
        },
        "properties": {
            "scores": {
                "items": {
                    "$ref": "#/$defs/CriterionScore"
                },
                "title": "Scores",
                "type": "array"
            }
        },
        "required": ["scores"],
        "title": "Scores",
        "type": "object"
    }
    ```

3.  We update our prompt to include this JSON Schema:
    ```python
    prompt = f"""
    Please analyze the following documents and extract evaluation scores.
    The output must be a single, valid JSON object that conforms to the following JSON Schema:
    {json.dumps(schema, indent=2)}
    
    Here are the documents:
    <generated_document>
    {GENERATED_DOCUMENT}
    </generated_document>
    <ground_truth_document>
    {GROUND_TRUTH_DOCUMENT}
    </ground_truth_document>
    """
    ```

4.  We call the model and extract the JSON string as before.
    ```python
    response = client.models.generate_content(model=MODEL_ID, contents=prompt)
    parsed_response = extract_json_from_response(response.text)
    ```
    It outputs:
    ```json
    {
      "scores": [
        {
          "criterion": "revenue_forecast",
          "score": 0,
          "reason": "The generated document overstates revenue growth  ..."
        },
        {
          "criterion": "user_growth",
          "score": 1,
          "reason": "The 15% user engagement growth is correctly reported ...."
        },
        {
          "criterion": "facts",
          "score": 0,
          "reason": "The generated document contains multiple factual ...."
        }
      ]
    }
    ```

5.  But now, the biggest difference, is that we can load the output dictionary into our Pydantic model and validate it:
    ```python
    try:
        scores = Scores.model_validate(parsed_response)
        print("Validation successful!")
    except Exception as e:
        print(f"Validation failed!")
    ```
    It outputs:
    ```
    Validation successful!
    ```
    The `scores` Pydantic object can now be safely used throughout your application. This is the main advantage: you move from unclear dictionaries to clean, predictable Python objects.

While Python’s built-in `dataclasses` or `TypedDict` can define structure, they only provide type hints for static analysis and do not perform runtime validation [[3]](https://www.speakeasy.com/blog/pydantic-vs-dataclasses), [[4]](https://codetain.com/blog/validators-approach-in-python-pydantic-vs-dataclasses/). If the LLM returns a string where an integer is expected, these tools will not catch the error. 

To conclude, Pydantic’s runtime validation, type constraints, and clear schema definitions make it our favorite way for structuring and validating all our domain data structures from our AI apps.

## Implementing Structured Outputs Using Gemini and Pydantic

While Pydantic brings structure and validation, we still had to construct the prompts and handle responses manually. When working with modern APIs such as Gemini and OpenAI, the recommended way to generate structured outputs is by using their native features. This approach is simpler, more accurate, and often more cost-effective than manual prompt engineering, as the vendor will always handle the optimization on top of their models better than your manual prompting [[9]](https://ai.google.dev/gemini-api/docs/structured-output), [[10]](https://www.vellum.ai/blog/when-should-i-use-function-calling-structured-outputs-or-json-mode), [[11]](https://www.googlecloudcommunity.com/gc/AI-ML/Structured-Output-in-vertexAI-BatchPredictionJob/m-p/866640). 

Let’s see how to achieve the same result for our LLM-as-judge example using the Gemini API's native capabilities. The process becomes much simpler.

1.  We define a `GenerateContentConfig` object, instructing the Gemini API to set the `response_mime_type` to `"application/json"` and the `response_schema` to our `Scores` Pydantic model. This configures the model to output JSON that is then automatically converted to the given Pydantic model. This single configuration step replaces the manual schema injection and parsing we did earlier:
    ```python
    from google.genai import types
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Scores
    )
    ```

2.  This configuration makes our prompt significantly shorter and cleaner, eliminating the need to manually inject any type of schema. We simply ask the model to perform the task, as the output format is guided directly by the config:
    ```python
    prompt = f"""
    You are an expert evaluator. Compare the generated document with the ground truth document and provide a score for each criterion.
    
    Here are the documents:
    <generated_document>
    {GENERATED_DOCUMENT}
    </generated_document>
    <ground_truth_document>
    {GROUND_TRUTH_DOCUMENT}
    </ground_truth_document>
    """
    ```

3.  Now, we call the model, passing our simplified prompt and the new configuration object. The API handles the rest, ensuring the output adheres to the schema.
    ```python
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=config
    )
    ```

4.  The Gemini client automatically parses the output for us. By accessing the `response.parsed` attribute, we receive a ready-to-use instance of our `Scores` Pydantic model:
    ```python
    scores = response.parsed
    print(f"Type of the response: `{type(scores)}`")
    ```
    It outputs:
    ```
    Type of the response: `<class '__main__.Scores'>`
    ```
This native approach is robust, efficient, and requires less code. While it is the recommended way for closed-source APIs or AI frameworks, the "from scratch" method remains useful for open-source models that may not have this built-in functionality or when you do not have access to any AI framework.

## The Best Model for Structured Outputs

A final thought on what's the best model for structured outputs: In general, all the latest LLMs support generating JSON, indirectly supporting Pydantic structures. 

However, when building AI systems, there is never the problem of what's the best model, but what's the best model for your given use case. Almost always, you cannot tell which model is better until you actually test them. That's why, when building AI systems, you should ALWAYS adopt a scientific method to find the optimal model (and its configuration):

1.  Configure different parameters (e.g., different models).
2.  Run experiments for each configuration.
3.  Compute business metrics of interest (e.g., using an LLM-as-judge).
4.  Use an LLMOps tool such as [Opik](https://www.comet.com/site/?utm_source=newsletter&utm_medium=partner&utm_campaign=paul) to analyze the results.
5.  Pick the best configuration and iterate if needed.

![Structured Outputs](scientific_method.png)
Image 2: The scientific method for evaluating and optimizing AI systems.

This high-level strategy works for tweaking any model, config or even feature of an AI system.

## Conclusion: Structured Outputs Are Everywhere

The thing is that structured outputs are everywhere! They are a fundamental pattern in AI engineering, connecting the probabilistic nature of LLMs with the deterministic world of software. Whether you are building a simple summarization workflow or a complex research agent, you will use structured outputs to ensure reliability and control.

Remember that this article is part of a longer series of 8 pieces on the AI Agents Foundations that will give you the tools to morph from a Python developer to an AI Engineer.

**Here’s our roadmap:**
1. [Workflows vs. Agents](https://decodingml.substack.com/p/ai-workflows-vs-agents-the-autonomy)
2. **Structured Outputs** _← You just finished this one._
3. Workflow Patterns _← Move to this one (available next Tuesday, 9:00 am CET)_
4. Tools
5. Planning: ReAct & Plan-and-Execute
6. Writing ReAct From Scratch
7. Memory
8. Multimodal Data

See you next week.

[Paul Iusztin](https://www.linkedin.com/in/pauliusztin/)

## References

1. Ntinopoulos, V., Biefer, H. R. C., Tudorache, I., Papadopoulos, N., Odavic, D., Risteski, P., Haeussler, A., & Dzemali, O. (2024). Large language models for data extraction from unstructured and semi-structured electronic health records: a multiple model performance evaluation. *BMJ Health & Care Informatics*, 32(1), e101139. [https://pmc.ncbi.nlm.nih.gov/articles/PMC11751965/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11751965/)
2. (n.d.). Evaluation of LLM-based Strategies for the Extraction of Food Product Information from Online Shops. *arXiv*. [https://arxiv.org/html/2506.21585v1](https://arxiv.org/html/2506.21585v1)
3. Speakeasy Team. (2024, August 29). Type Safety in Python: Pydantic vs. Data Classes vs. Annotations vs. TypedDicts. *Speakeasy*. [https://www.speakeasy.com/blog/pydantic-vs-dataclasses](https://www.speakeasy.com/blog/pydantic-vs-dataclasses)
4. (n.d.). Validators approach in Python - Pydantic vs. Dataclasses. *Codetain*. [https://codetain.com/blog/validators-approach-in-python-pydantic-vs-dataclasses/](https://codetain.com/blog/validators-approach-in-python-pydantic-vs-dataclasses/)
5. (n.d.). Automating Knowledge Graphs with LLM Outputs. *Prompts.ai*. [https://www.prompts.ai/en/blog-details/automating-knowledge-graphs-with-llm-outputs](https://www.prompts.ai/en/blog-details/automating-knowledge-graphs-with-llm-outputs)
6. Kelly, C. (2024, February 13). Structured Outputs: everything you should know. *Humanloop*. [https://humanloop.com/blog/structured-outputs](https://humanloop.com/blog/structured-outputs)
7. (2024, June 26). Structured data response with Amazon Bedrock: Prompt Engineering and Tool Use. *Amazon Web Services*. [https://aws.amazon.com/blogs/machine-learning/structured-data-response-with-amazon-bedrock-prompt-engineering-and-tool-use/](https://aws.amazon.com/blogs/machine-learning/structured-data-response-with-amazon-bedrock-prompt-engineering-and-tool-use/)
8. (n.d.). Best practices for prompt engineering with the OpenAI API. *OpenAI Help Center*. [https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api)
9. (n.d.). Structured output. *Google AI for Developers*. [https://ai.google.dev/gemini-api/docs/structured-output](https://ai.google.dev/gemini-api/docs/structured-output)
10. Sharma, A. (2024, October 10). When should I use function calling, structured outputs or JSON mode? *Vellum AI Blog*. [https://www.vellum.ai/blog/when-should-i-use-function-calling-structured-outputs-or-json-mode](https://www.vellum.ai/blog/when-should-i-use-function-calling-structured-outputs-or-json-mode)
11. (n.d.). Structured Output in vertexAI BatchPredictionJob. *Google Cloud Community*. [https://www.googlecloudcommunity.com/gc/AI-ML/Structured-Output-in-vertexAI-BatchPredictionJob/m-p/866640](https://www.googlecloudcommunity.com/gc/AI-ML/Structured-Output-in-vertexAI-BatchPredictionJob/m-p/866640)

---

## Images

If not otherwise stated, all images are created by the author.