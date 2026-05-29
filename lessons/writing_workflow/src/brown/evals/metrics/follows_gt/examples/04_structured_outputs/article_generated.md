# Lesson 4: Structured Outputs

We will start with a critical component for building production-grade AI systems: structured outputs. This concept is the bridge between the probabilistic, flexible world of Large Language Models (LLMs)—what some call Software 3.0—and the deterministic, rigid world of traditional Python code, or Software 1.0.

LLMs generate text, but our applications need data—objects, lists, and validated fields. Structured outputs provide a reliable contract that forces the model’s free-form text into a predictable format. Mastering this is not just a nice-to-have, it is essential for any AI Engineer who wants to build systems that are reliable, testable, and maintainable. Without it, you are just building fragile applications on a foundation of hope.

## Why Structured Outputs Are Critical

Before we write any code, it is important to understand why forcing an LLM to return structured data is a non-negotiable best practice. At its core, it is about control and reliability. When a model returns a clean JSON object or a validated Pydantic model, the output becomes easy to parse, manipulate, and debug. You can directly access attributes like `response.summary` instead of wrestling with fragile regular expressions to find a summary somewhere in a long string of text [[1]](https://arxiv.org/html/2506.21585v1).

This creates a clear contract between the LLM and your application code. The biggest advantage comes when using libraries like Pydantic, which provide out-of-the-box data and type validation. If the model hallucinates a field or returns a string where you expect an integer, your code will know immediately. This is a level of safety that is crucial in production environments, where unpredictable outputs can cause downstream failures.

Common use cases are everywhere. Entity extraction, a key step in building knowledge graphs for advanced RAG, relies on pulling structured data like names, dates, and locations from unstructured text [[2]](https://www.prompts.ai/en/blog-details/automating-knowledge-graphs-with-llm-outputs). Another is formatting an LLM's output into a predefined structure for downstream systems, like updating a database or displaying information in a user interface. This process ensures that data flows predictably from the language model into the rest of your system.

```mermaid
graph TD
    A[Unstructured Text] --> B{Large Language Model};
    B --> C[Structured Output<br>(JSON / Pydantic)];
    subgraph Python Application (Software 1.0)
        C --> D[Data Transformation];
        C --> E[Database Update];
        C --> F[API Response];
    end
```
Figure 1: A diagram illustrating how structured outputs bridge LLMs with downstream applications.

GraphRAG represents a revolutionary approach to retrieval-augmented generation that leverages knowledge graphs to provide more contextual and accurate responses. Unlike traditional RAG systems that rely on vector similarity search, GraphRAG constructs a comprehensive knowledge graph from the source documents, capturing not just semantic relationships but also explicit entity connections, temporal sequences, and hierarchical structures. This graph-based representation enables more sophisticated reasoning capabilities, allowing the system to traverse relationships between entities, understand causal chains, and maintain context across multiple documents.

Ultimately, structured outputs are the mechanism that translates the intelligence of Software 3.0 into the deterministic logic required by Software 1.0.

## Implementing structured outputs from scratch using JSON

To understand what is happening behind the scenes, we will first implement structured outputs manually. The goal is to guide the LLM to return a JSON object using nothing but clever prompt engineering. This approach gives you a fundamental understanding of how models can be steered, even without dedicated API features.

Our process will involve setting up the client, defining an input document, crafting a detailed prompt, and finally, parsing the model's string response into a Python dictionary.

1.  First, we will configure our Gemini client and define some constants. We will use the `gemini-2.5-flash` model, which is fast and cost-effective.

    ```python
    import json
    
    from google import genai
    from google.genai import types
    from pydantic import BaseModel, Field
    
    from lessons.utils import pretty_print
    
    client = genai.Client()
    MODEL_ID = "gemini-2.5-flash"
    ```

2.  Next, let us define the sample document we want to analyze. It is a simple financial performance summary.

    ```python
    DOCUMENT = """
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
    ```

3.  Now for the most important part: the prompt. We explicitly instruct the model to return a valid JSON object and provide the exact structure we expect. Notice the use of XML tags like `<json>` and `<document>` to clearly separate the instructions from the content. This helps the model understand the different parts of the prompt [[3]](https://aws.amazon.com/blogs/machine-learning/structured-data-response-with-amazon-bedrock-prompt-engineering-and-tool-use/).

    ```python
    prompt = f"""
    Analyze the following document and extract metadata from it. 
    The output must be a single, valid JSON object with the following structure:
    <json>
    {{ 
        "summary": "A concise summary of the article.", 
        "tags": ["list", "of", "relevant", "tags"], 
        "keywords": ["list", "of", "key", "concepts"],
        "quarter": "Q...",
        "growth_rate": "...%",
    }}
    </json>
    
    Here is the document:
    <document>
    {DOCUMENT}
    </document>
    """
    ```

4.  With the prompt ready, we call the model.

    ```python
    response = client.models.generate_content(model=MODEL_ID, contents=prompt)
    ```

    The raw output is a string that includes markdown formatting around the JSON:

    ```
    ```json
    { 
        "summary": "The Q3 2023 financial report highlights a strong performance with a 20% increase in revenue and 15% growth in user engagement, surpassing market expectations. This success is attributed to effective product strategy, strong market positioning, and successful expansion into new markets. The company also improved efficiency, reducing customer acquisition costs by 10% and achieving a 92% customer retention rate.", 
        "tags": [ 
            "Financial Performance", 
            "Q3 2023", 
            "Earnings Report", 
            "Revenue Growth", 
            "User Engagement", 
            "Market Expansion", 
            "Customer Retention", 
            "Business Strategy" 
        ], 
        "keywords": [ 
            "Q3", 
            "Revenue", 
            "Growth", 
            "User engagement", 
            "Digital services", 
            "New markets", 
            "Customer acquisition cost", 
            "Retention rate", 
            "Financial results" 
        ], 
        "quarter": "Q3", 
        "growth_rate": "20%" 
    }
    ```

5.  To make this string useful, we need a helper function to clean it up and parse it into a Python dictionary.

    ```python
    def extract_json_from_response(response: str) -> dict:
        """
        Extracts JSON from a response string that is wrapped in <json> or ```json tags.
        """
    
        response = response.replace("<json>", "").replace("</json>", "")
        response = response.replace("```json", "").replace("```", "")
    
        return json.loads(response)
    
    parsed_response = extract_json_from_response(response.text)
    print(json.dumps(parsed_response, indent=2))
    ```

    The final output is a clean Python dictionary that we can now use in our application:

    ```json
    {
    "summary": "The Q3 2023 financial report highlights a strong performance with a 20% increase in revenue and 15% growth in user engagement, surpassing market expectations. This success is attributed to effective product strategy, strong market positioning, and successful expansion into new markets. The company also improved efficiency, reducing customer acquisition costs by 10% and achieving a 92% customer retention rate.",
    "tags": [
        "Financial Performance",
        "Q3 2023",
        "Earnings Report",
        "Revenue Growth",
        "User Engagement",
        "Market Expansion",
        "Customer Retention",
        "Business Strategy"
    ],
    "keywords": [
        "Q3",
        "Revenue",
        "Growth",
        "User engagement",
        "Digital services",
        "new markets",
        "Customer acquisition cost",
        "Retention rate",
        "Financial results"
    ],
    "quarter": "Q3",
    "growth_rate": "20%"
    }
    ```

This manual method works, but it is brittle. If the model deviates even slightly from the requested format, our `extract_json_from_response` function might fail. This is why for any serious application, we need a more robust solution.

## Implementing structured outputs from scratch using Pydantic

While raw JSON is a step up from unstructured text, it still leaves a lot to be desired. Python dictionaries are flexible, but that flexibility is also a weakness—there are no guarantees about what keys are present or what data types their values hold. This is where Pydantic comes in. It has become the industry standard for data validation in Python, offering robust type and field checking out-of-the-box.

When an LLM's output is parsed into a Pydantic model, it is not just a dictionary; it is an object with a guaranteed structure and validated data types. If the model returns a string for a field defined as an integer, or completely omits a required field, Pydantic will raise a `ValidationError` with a clear message explaining what went wrong [[4]](https://www.speakeasy.com/blog/pydantic-vs-dataclasses). This immediate feedback loop is invaluable for building reliable systems.

1.  Let us start by defining a `RedditThread` class using Pydantic. It mirrors our JSON structure but adds explicit type hints and descriptions for each field. Pydantic works hand-in-hand with Python's standard `typing` library, allowing us to enforce both the structure and the type of the LLM's output. Since Python 3.9, you can use built-in types like `list` directly instead of importing `List` from `typing`:

    ```python
    class RedditThread(BaseModel):
        """A class to hold structured metadata for a Reddit thread."""

        summary: str = Field(description="A concise, 1-2 sentence summary of the thread.")
        tags: list[str] = Field(description="A list of 3-5 high-level tags relevant to the thread.")
        comments: list[str] = Field(description="A list of top-level comment texts extracted from the thread.")
    ```

    You can even nest Pydantic models to represent more complex data structures. For example, we could define a separate `Comment` model and embed it within `RedditThread`. This allows for rich, hierarchical data representation. However, it is a good practice to keep schemas reasonably simple, as overly complex structures can confuse the LLM and increase the likelihood of errors:

    ```python
    class Comment(BaseModel):
        """A single comment in a Reddit thread."""
        text: str = Field(description="The comment text.")
        author: str = Field(description="The username of the comment author.")

    class RedditThreadWithComments(BaseModel):
        """A class to hold structured data for a Reddit thread with rich comments."""
        summary: str = Field(description="A concise, 1-2 sentence summary of the thread.")
        tags: list[str] = Field(description="A list of 3-5 high-level tags relevant to the thread.")
        comments: list[Comment] = Field(description="A list of parsed comments, including author and text.")
    ```

2.  The formal term for defining the expected output of an LLM is a `schema` or `contract`. Pydantic can automatically generate a JSON Schema from our model, which we can then inject into the prompt. This provides the LLM with a much more formal and unambiguous set of instructions compared to our previous example:

    ```python
    schema = RedditThread.model_json_schema()
    print(json.dumps(schema, indent=2))
    ```

    This generates a detailed JSON Schema that precisely defines every field, type, and description. This is similar to the technique that APIs like Gemini and OpenAI use internally to enforce output schemas.

    ```python
    ```json
    {
    "description": "A class to hold structured metadata for a Reddit thread.",
    "properties": {
        "summary": {
        "description": "A concise, 1-2 sentence summary of the thread.",
        "title": "Summary",
        "type": "string"
        },
        "tags": {
        "description": "A list of 3-5 high-level tags relevant to the thread.",
        "items": {
            "type": "string"
        },
        "title": "Tags",
        "type": "array"
        },
        "comments": {
        "description": "A list of top-level comment texts extracted from the thread.",
        "items": {
            "type": "string"
        },
        "title": "Comments",
        "type": "array"
        }
    },
    "required": [
        "summary",
        "tags",
        "comments"
    ],
    "title": "RedditThread",
    "type": "object"
    }
    ```
    ```

3.  Our new prompt now includes this machine-readable schema:

    ```python
    prompt = f"""
    Please analyze the following document and extract metadata from it. 
    The output must be a single, valid JSON object that conforms to the following JSON Schema:
    <json>
    {json.dumps(schema, indent=2)}
    </json>
    
    Here is the document:
    <document>
    {DOCUMENT}
    </document>
    """
    ```

4.  Now we call the model and extract the JSON string, as in the previous example:

    ```python
    response = client.models.generate_content(model=MODEL_ID, contents=prompt)
    parsed_response = extract_json_from_response(response.text)
    
    ```

5. After calling the model and parsing the JSON response as before, we can now validate it against our `RedditThread` model. The result is a fully validated Pydantic object. The core idea is to use these strongly-typed objects throughout your application, not obscure dictionaries where you have to constantly check for keys and types. This eliminates a whole class of potential bugs:

    ```python
    try:
        thread = RedditThread.model_validate(parsed_response)
        print("\nValidation successful!")
        print(thread.model_dump_json(indent=2))
    except Exception as e:
        print(f"\nValidation failed: {e}")
    ```

    It outputs:
    ```json
    Validation successful!
    {
    "summary": "This thread summarizes key takeaways from the latest product launch AMA, including roadmap clarifications, performance improvements, and community feature requests.",
    "tags": [
        "Product",
        "AMA",
        "Roadmap",
        "Performance",
        "Community"
    ],
    "comments": [
        "Thanks for the detailed answers—excited about the Q4 performance work!",
        "Please prioritize better onboarding docs; new users are still confused.",
        "Great to hear native integrations are on the roadmap."
    ]
    }
    ```

While other options like Python's built-in `TypedDict` and `dataclasses` exist, they primarily help with static analysis and do not perform runtime validation [[4]](https://www.speakeasy.com/blog/pydantic-vs-dataclasses). This means that if an LLM returns data with the wrong type, these tools will not catch it during execution. Pydantic, in contrast, actively validates and coerces incoming data to match the expected types, catching invalid or ill-typed data as soon as it is received [[4]](https://www.speakeasy.com/blog/pydantic-vs-dataclasses). 

This out-of-the-box runtime validation is crucial for ensuring the quality and correctness of LLM outputs, making Pydantic the most powerful and popular choice for moving data reliably in LLM workflows and AI agents [[4]](https://www.speakeasy.com/blog/pydantic-vs-dataclasses), [[10]](https://www.youtube.com/watch?v=WRiQD4lmnUk).

## Implementing structured outputs using Gemini and Pydantic

Instead of hoping the model follows instructions in a prompt, the API can be directly commanded to return output that conforms to a specific schema.

This approach is superior for several reasons. The complexity of prompts is reduced, which in turn can lower token costs. More importantly, reliability is improved because the model is constrained at a lower level to generate valid, schema-compliant JSON, reducing the chances of formatting errors or hallucinations [[6]](https://www.vellum.ai/blog/when-should-i-use-function-calling-structured-outputs-or-json-mode). This means less post-processing and fewer retries, saving both time and computational resources.

Let us see how the same result can be achieved using the Gemini SDK's native capabilities.

-  The key is the `GenerateContentConfig` object. It is configured by setting `response_mime_type` to `"application/json"` and passing the `DocumentMetadata` Pydantic class directly to the `response_schema` parameter. The SDK handles converting the Pydantic model into the required JSON Schema and communicating it to the API behind the scenes [[5]](https://ai.google.dev/gemini-api/docs/structured-output).

    ```python
    config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=DocumentMetadata)
    ```

-  With this configuration, the prompt becomes dramatically simpler. Lengthy JSON schema examples or explicit formatting instructions no longer need to be included. The model can simply be asked to perform the task.

    ```python
    prompt = f"""
    Analyze the following document and extract its metadata.
    
    Document:
    --- 
    {DOCUMENT}
    --- 
    """
    ```

This native approach is the recommended way to work with structured data. It is cleaner, less error-prone, and lets you focus on your application's logic instead of fighting with prompt formatting and response parsing.

## Structured Outputs Are Everywhere

Structured outputs are a fundamental pattern in AI engineering that serves as the essential bridge allowing the fluid, text-based world of LLMs to communicate with the rigid, data-driven logic of our applications, appearing universally in nearly every LLM-powered system whether building simple workflows or complex, autonomous agents across any domain from finance to healthcare [7](https://developers.redhat.com/articles/2025/06/03/structured-outputs-vllm-guiding-ai-responses).

Mastering this concept is not just about learning a single technique but about adopting a mindset of reliability and predictability when working with inherently non-deterministic models.

## References

- [1] [LLM-based Extraction of Structured Product Information from E-commerce Web Pages](https://arxiv.org/html/2506.21585v1)
- [2] [Automating Knowledge Graphs with LLM Outputs](https://www.prompts.ai/en/blog-details/automating-knowledge-graphs-with-llm-outputs)
- [3] [Structured data response with Amazon Bedrock prompt engineering and tool use](https://aws.amazon.com/blogs/machine-learning/structured-data-response-with-amazon-bedrock-prompt-engineering-and-tool-use/)
- ...
