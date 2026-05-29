# LLM Structured Outputs: The Definitive Guide
### From brittle strings to validated Pydantic objects

## Introduction

In our previous lessons, we laid the groundwork for AI Engineering. We explored the AI agent landscape, distinguished between rigid workflows and autonomous agents, and covered the essentials of Context Engineering. Now, we've reached a critical junction: how do we make the fluid, often unpredictable outputs of a Large Language Model (LLM) work with the structured, type-safe world of our Python applications? The answer is structured outputs.

I remember a project I worked on early in my career, trying to build a sentiment analysis pipeline for customer reviews. We were getting raw text from the LLM, and my life became a nightmare of writing increasingly complex regular expressions to parse out the sentiment score, the key topics, and the customer's name. Every time the model updated, its phrasing would change slightly, and my regex would break. The system was so brittle it fell over if you looked at it funny. It was a painful lesson, but it taught me the non-negotiable value of forcing models to return data in a predictable format.

This concept is the bridge between what we can call Software 3.0 (the probabilistic, text-in-text-out nature of LLMs) and Software 1.0 (the deterministic, strictly-typed logic of conventional programming). For any AI Engineer, mastering structured outputs isn’t just a nice-to-have; it’s a fundamental skill for building applications that are reliable, predictable, and easy to debug. In fact, many experts believe that solving structured data exchange is the key to unlocking true Artificial General Intelligence (AGI), as it creates the necessary interface for complex agentic systems. Without it, you’re left wrestling with brittle string manipulation and hoping for the best. Let's get it right.

## Understanding why structured outputs are critical

Before we jump into the code, it’s important to understand why forcing an LLM to return structured data is a non-negotiable best practice. When an LLM returns a raw string, you are left to parse it. This often involves fragile methods like regular expressions or string splitting, which can easily break if the model slightly changes its phrasing, adds an extra sentence, or omits a piece of information. This approach is a maintenance nightmare and a recipe for unpredictable bugs in production [[1]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11751965/), [[2]](https://arxiv.org/html/2506.21585v1).

Structured outputs offer several key benefits:
1.  **Out-of-the-box data quality checks:** When using a library like Pydantic, structured outputs enforce a strict contract. Pydantic models validate not just the structure of the data but also the data types (e.g., ensuring a `sentiment_score` is a `float` and not a `string`). If the LLM’s output fails this contract, Pydantic raises a clear validation error [[3]](https://www.speakeasy.com/blog/pydantic-vs-dataclasses).
2.  **Easy programmatic manipulation:** Once parsed into native Python data structures like dictionaries and lists, you can programmatically access the data you need, manipulate it, and pass it to other parts of your application with confidence.
3.  **Improved Token Efficiency:** Using more compact formats like YAML instead of JSON can reduce the token count by up to 48%. Since most API calls are priced per token, this directly lowers costs and can also speed up response times [[20]](https://betterprogramming.pub/yaml-vs-json-which-is-more-efficient-for-language-models-5bc11dd0f6df).
4.  **Reduces fragile parsing:** By providing a consistent format, you eliminate the messy and error-prone task of parsing raw text with fragile regular expressions.

This pattern is essential for many use cases, from extracting entities like names and dates to build knowledge graphs for advanced Retrieval-Augmented Generation (RAG) applications, to formatting LLM outputs for downstream processing in a data pipeline [[4]](https://www.prompts.ai/en/blog-details/automating-knowledge-graphs-with-llm-outputs). Structured outputs ensure that the LLM acts as a reliable component, delivering machine-readable data where free-form text would be problematic.

In the next sections, we'll see how to implement this bridge, starting from scratch with basic JSON and then moving to a more robust Pydantic-based approach before finally using Gemini’s native capabilities.

## Implementing structured outputs from scratch using JSON

To fully appreciate what happens behind the scenes, we’ll first build a structured output system from scratch. Our goal is to prompt the model to return a JSON object and then parse that string output into a usable Python dictionary. This approach relies entirely on prompt engineering, where we explicitly tell the LLM the format we expect.

First, we set up our environment by initializing the Gemini client and defining our constants and the document we want to analyze. This foundational setup ensures we have the necessary tools and data ready for our extraction task.

```python
import json

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Initialize the Gemini Client
client = genai.Client()

# Define Constants
MODEL_ID = "gemini-2.5-flash"

# Example: Extracting Metadata from a Document
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

Next, we craft a detailed prompt. Notice how we use XML tags like `<json>` and `<document>` to clearly separate the instructions, the desired output schema, and the input context. This technique helps the model distinguish between different parts of the prompt, leading to more reliable outputs [[5]](https://aws.amazon.com/blogs/machine-learning/structured-data-response-with-amazon-bedrock-prompt-engineering-and-tool-use/). We explicitly define the JSON structure we expect, including field names and example values, to guide the LLM.

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

response = client.models.generate_content(model=MODEL_ID, contents=prompt)
```

The model’s raw output is a string, often wrapped in markdown code blocks. This is not yet a usable Python object, as it requires an additional parsing step to convert it into a dictionary.

It outputs:
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
```

To make it usable, we need a helper function to clean the string and parse it into a Python dictionary. This function removes any extraneous characters or markdown formatting that the LLM might include, ensuring we get a pure JSON string.

```python
def extract_json_from_response(response: str) -> dict:
    """
    Extracts JSON from a response string that is wrapped in <json> or ```json tags.
    """

    response = response.replace("<json>", "").replace("</json>", "")
    response = response.replace("```json", "").replace("```", "")

    return json.loads(response)

parsed_response = extract_json_from_response(response.text)
print(f"Type of the parsed response: `{type(parsed_response)}`")
print(json.dumps(parsed_response, indent=2))
```

After parsing, we finally have a clean Python dictionary that we can work with in our code. This dictionary now contains the extracted metadata in a structured, accessible format. While this method works, it's brittle. If the model adds extra text or makes a mistake in the JSON syntax, our `extract_json_from_response` function might fail. This is where Pydantic provides a much-needed layer of robustness and reliability.

## Implementing structured outputs from scratch using Pydantic

Using raw JSON or Python dictionaries is a step up from plain text, but it still leaves you vulnerable. Dictionaries in Python are ambiguous; you don't know what keys they contain or what data types their values hold without inspecting them or writing defensive code full of `if-else` statements. This is where Pydantic becomes the go-to tool for modeling structured outputs.

Pydantic provides data validation and type checking out-of-the-box. If an LLM returns a string for a field that should be an integer, or completely omits a required field, Pydantic will raise a `ValidationError` with a clear message explaining what went wrong [[6]](https://codetain.com/blog/validators-approach-in-python-pydantic-vs-dataclasses/). This creates a strong, reliable contract for the data moving between your LLM and your application.

Let's refactor our previous example using a Pydantic model. First, we define a `DocumentMetadata` class that specifies the exact structure and types we expect. Each field is explicitly typed, and we can add descriptions to guide the LLM further.

```python
class DocumentMetadata(BaseModel):
    """A class to hold structured metadata for a document."""

    summary: str = Field(description="A concise, 1-2 sentence summary of the document.")
    tags: list[str] = Field(description="A list of 3-5 high-level tags relevant to the document.")
    keywords: list[str] = Field(description="A list of specific keywords or concepts mentioned.")
    quarter: str = Field(description="The quarter of the financial year described in the document (e.g, Q3 2023).")
    growth_rate: str = Field(description="The growth rate of the company described in the document (e.g, 10%).")
```

💡 **Tip:** Pydantic works hand-in-hand with Python's standard `typing` library. We use type hints like `str` and `list[str]` to define the expected type for each attribute. Starting with Python 10, you can use built-in types like `list` directly, instead of importing `List` from `typing`.

You can even nest Pydantic models to represent more complex, hierarchical data structures. For example, we could create `Summary` and `Tag` objects to make our schema more modular and readable. However, be careful not to over-complicate the schema, as excessively nested structures can confuse the LLM and lead to more errors.

```python
class Tag(BaseModel):
    label: str
    relevance: float = Field(description="A score from 0.0 to 1.0.")

class Summary(BaseModel):
    text: str
    sentiment: str = Field(description="Can be 'positive', 'neutral', or 'negative'.")

class AdvancedDocumentMetadata(BaseModel):
    summary_details: Summary
    extracted_tags: list[Tag]
```

Now, we update our prompt to include this schema. By embedding the JSON schema directly, we provide the LLM with explicit instructions on the expected output format.

```python
prompt = f"""
Please analyze the following document and extract metadata from it. 
The output must be a single, valid JSON object that conforms to the Pydantic model provided.

Here is the document:
<document>
{DOCUMENT}
</document>
"""

response = client.models.generate_content(model=MODEL_ID, contents=prompt)
parsed_response = extract_json_from_response(response.text)
```

Finally, we map the parsed JSON to our `DocumentMetadata` model using `model_validate`. This step attempts to create an instance of our Pydantic class from the dictionary, automatically checking that all required fields are present and that their values match the specified types. If any validation fails, Pydantic will immediately alert us.

```python
try:
    document_metadata = DocumentMetadata.model_validate(parsed_response)
    print("\nValidation successful!")
    print(f"Type of the validated response: `{type(document_metadata)}`")
    print(document_metadata.model_dump_json(indent=2))
except Exception as e:
    print(f"\nValidation failed: {e}")
```

The core idea is to use these validated Pydantic objects throughout your code, not obscure dictionaries. This eliminates defensive programming and makes your application more robust. While Python's built-in `TypedDict` and `dataclasses` can enforce structure, they don't perform runtime type validation [[3]](https://www.speakeasy.com/blog/pydantic-vs-dataclasses). This is why Pydantic has become the standard for moving data around in LLM workflows.

A Note on YAML vs. JSON:

While JSON is the most common format for structured outputs, it's not always the most efficient. YAML (YAML Ain't Markup Language) is often more token-efficient because its syntax relies on indentation, avoiding the curly brackets, quotes, and commas required in JSON. For some use cases, this can result in a token reduction of over 40%, leading to lower costs and faster responses from the LLM. However, JSON's strictness is often an advantage for data validation, so the choice depends on the specific needs of your application.

## Implementing structured outputs using Gemini and Pydantic

So far, we've implemented structured outputs from scratch. While this is a great way to understand the mechanics, most modern LLM APIs, like Google's Gemini, provide native support for this functionality.

Let's see how to achieve the same result using the Gemini SDK's built-in features. The key is the `GenerateContentConfig` object, which allows us to specify the desired output format directly in the API call. We set the `response_mime_type` to `"application/json"` and pass our `DocumentMetadata` Pydantic class directly to the `response_schema` parameter. This tells Gemini exactly what structure and types to expect in its response.

```python
config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=DocumentMetadata)
```

With this configuration, our prompt becomes dramatically simpler. We no longer need to manually inject the JSON schema or provide complex instructions about the output format.

```python
prompt = f"""
Analyze the following document and extract its metadata.

Document:
--- 
{DOCUMENT}
--- 
"""
```

Now, when we call the model, the Gemini API handles the schema enforcement internally.

```python
response = client.models.generate_content(model=MODEL_ID, contents=prompt, config=config)
```

The best part is that the response object now has a `.parsed` attribute, which gives us direct access to the output as a fully validated Pydantic object. There's no need for manual string cleaning or validation steps.

```python
print(f"Type of the response: `{type(response.parsed)}`")
print(response.parsed.model_dump_json(indent=2))
```

This native approach is cleaner, more reliable, and should be your default choice when the underlying API supports it.

## Conclusion: Structured Outputs Are Everywhere

We've covered the why and how of structured outputs, from manual prompt engineering to native API integration with Pydantic. This pattern is not just a niche technique; it's a fundamental building block for almost any application you'll create in AI engineering. Whether you're building a simple workflow to summarize articles, a research agent to analyze scientific papers, or a coding agent to write functions, you will need to get structured data back from the LLM.

This concept is universal. It transcends specific models, domains, and application patterns. As we move forward in this course, you will see structured outputs used again and again. You'll also see them in action when we discuss agent memory in **Lesson 9 - Agent Memory & Knowledge**, where structured data is key to storing and retrieving information effectively. Mastering this skill is a prerequisite for building any advanced AI system that needs to interact reliably with the real world.

## References

- [1] [Systematic evaluation of large language models for data extraction from unstructured and semi-structured electronic health records](https://pmc.ncbi.nlm.nih.gov/articles/PMC11751965/)
- [2] [A Comparative Study of LLM-based Information Extraction: Direct vs. Indirect Approaches](https://arxiv.org/html/2506.21585v1)
- [3] [Pydantic vs. Dataclasses](https://www.speakeasy.com/blog/pydantic-vs-dataclasses)
- [4] [Automating Knowledge Graphs with LLM Outputs](https://www.prompts.ai/en/blog-details/automating-knowledge-graphs-with-llm-outputs)
- [5] [Structured Data Response with Amazon Bedrock: Prompt Engineering and Tool Use](https://aws.amazon.com/blogs/machine-learning/structured-data-response-with-amazon-bedrock-prompt-engineering-and-tool-use/)
- [6] [Validators Approach in Python: Pydantic vs. Dataclasses](https://codetain.com/blog/validators-approach-in-python-pydantic-vs-dataclasses/)
- [7] [Structured output](https://ai.google.dev/gemini-api/docs/structured-output)