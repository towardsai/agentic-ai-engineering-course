# LLM Structured Outputs: The Definitive Guide
### From brittle strings to validated Pydantic objects

## Introduction

In our previous lessons, we laid the groundwork for AI Engineering. We explored the AI agent landscape, distinguished between rigid workflows and autonomous agents, and covered the essentials of Context Engineering. Now, we've reached a critical junction: how do we make the fluid, often unpredictable outputs of a Large Language Model (LLM) work with the structured, type-safe world of our Python applications? The answer is structured outputs.

This concept is the bridge between what we can call Software 3.0 (the probabilistic, text-in-text-out nature of LLMs) and Software 1.0 (the deterministic, strictly-typed logic of conventional programming). For any AI Engineer, mastering structured outputs isn’t just a nice-to-have; it’s a fundamental skill for building applications that are reliable, predictable, and easy to debug. Without it, you’re left wrestling with brittle string manipulation and hoping for the best. Let's get it right.

## Understanding why structured outputs are critical

Before we jump into the code, it’s important to understand why forcing an LLM to return structured data is a non-negotiable best practice. When an LLM returns a raw string, you are left to parse it. This often involves fragile methods like regular expressions or string splitting, which can easily break if the model slightly changes its phrasing, adds an extra sentence, or omits a piece of information. This approach is a maintenance nightmare and a recipe for unpredictable bugs in production [[1]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11751965/), [[2]](https://arxiv.org/html/2506.21585v1).

Structured outputs, like JSON, solve this by providing a consistent format that is easy to parse into native Python data structures like dictionaries and lists. Once parsed, you can programmatically access the data you need, manipulate it, and pass it to other parts of your application with confidence. This makes the entire system easier to interpret, monitor, and debug. This consistency is crucial for integrating LLM outputs into any downstream system, ensuring that your application can reliably consume and process the information.

The real power, however, comes when you pair structured outputs with a validation library like Pydantic. Pydantic models create a strict contract between the LLM and your application code. They enforce not just the structure of the data (the fields that must be present) but also the data types (e.g., ensuring a `sentiment_score` is a `float` and not a `string`). If the LLM’s output fails to meet this contract, Pydantic raises a clear validation error, preventing bad data from corrupting your system [[3]](https://www.speakeasy.com/blog/pydantic-vs-dataclasses). This out-of-the-box data quality check is critical in the LLM world, where outputs can be unpredictable.

This pattern is essential for many use cases, from extracting entities like names and dates to build knowledge graphs for advanced Retrieval-Augmented Generation (RAG) applications, to formatting LLM outputs for downstream processing in a data pipeline [[4]](https://www.prompts.ai/en/blog-details/automating-knowledge-graphs-with-llm-outputs). Structured outputs ensure that the LLM acts as a reliable component, delivering machine-readable data where free-form text would be problematic.

```mermaid
graph TD
    A[Unstructured Text] --> B{LLM};
    B --> C[Structured Output <br> (JSON/Pydantic)];
    C --> D[Downstream System];
    subgraph Downstream System
        D1[Database]
        D2[API Call]
        D3[User Interface]
    end
    C --> D1;
    C --> D2;
    C --> D3;
```
Figure 1: A high-level overview of how structured outputs enable downstream processing.

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

After parsing, we finally have a clean Python dictionary that we can work with in our code. This dictionary now contains the extracted metadata in a structured, accessible format.

It outputs:
```
Type of the parsed response: `<class 'dict'>`
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

While this method works, it's brittle. If the model adds extra text or makes a mistake in the JSON syntax, our `extract_json_from_response` function might fail. This is where Pydantic provides a much-needed layer of robustness and reliability.

## Implementing structured outputs from scratch using Pydantic

Using raw JSON or Python dictionaries is a step up from plain text, but it still leaves you vulnerable. Dictionaries in Python are ambiguous; you don't know what keys they contain or what data types their values hold without inspecting them or writing defensive code full of `if-else` statements. This is where Pydantic becomes the go-to tool for modeling structured outputs.

Pydantic provides data validation and type checking out-of-the-box. If an LLM returns a string for a field that should be an integer, or completely omits a required field, Pydantic will raise a `ValidationError` with a clear message explaining what went wrong [[6]](https://codetain.com/blog/validators-approach-in-python-pydantic-vs-dataclasses/). This creates a strong, reliable contract for the data moving between your LLM and your application. It ensures that the data conforms to your expectations, preventing unexpected errors and making your application more robust.

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

💡 **Tip:** Pydantic works hand-in-hand with Python's standard `typing` library. We use type hints like `str` and `list[str]` to define the expected type for each attribute. Starting with Python 10, you can use built-in types like `list` directly, instead of importing `List` from `typing`. This combination of Pydantic and `typing` allows us to enforce both the structure (i.e., the data structure's layout) and the type of the LLM's output, creating a robust data contract.

You can even nest Pydantic models to represent more complex, hierarchical data structures. For example, we could create `Summary` and `Tag` objects to make our schema more modular and readable. However, be careful not to over-complicate the schema, as excessively nested structures can confuse the LLM and lead to more errors, making the extraction process less reliable. The goal is clarity and precision, not unnecessary complexity.

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

The standard term for defining the expected output format of an LLM is a `schema` or `contract`. With Pydantic, we can automatically generate a JSON Schema from our model and inject it directly into the prompt. This gives the LLM a formal, machine-readable definition of the structure it needs to follow. This is a similar technique to what APIs like Gemini and OpenAI use internally to enforce their output formats, ensuring consistency and reliability [[7]](https://ai.google.dev/gemini-api/docs/structured-output).

```python
schema = DocumentMetadata.model_json_schema()
print(json.dumps(schema, indent=2))
```

It outputs:
```json
{
"description": "A class to hold structured metadata for a document.",
"properties": {
    "summary": {
    "description": "A concise, 1-2 sentence summary of the document.",
    "title": "Summary",
    "type": "string"
    },
    "tags": {
    "description": "A list of 3-5 high-level tags relevant to the document.",
    "items": {
        "type": "string"
    },
    "title": "Tags",
    "type": "array"
    },
    "keywords": {
    "description": "A list of specific keywords or concepts mentioned.",
    "items": {
        "type": "string"
    },
    "title": "Keywords",
    "type": "array"
    },
    "quarter": {
    "description": "The quarter of the financial year described in the document (e.g, Q3 2023).",
    "title": "Quarter",
    "type": "string"
    },
    "growth_rate": {
    "description": "The growth rate of the company described in the document (e.g, 10%).",
    "title": "Growth Rate",
    "type": "string"
    }
},
"required": [
    "summary",
    "tags",
    "keywords",
    "quarter",
    "growth_rate"
],
"title": "DocumentMetadata",
"type": "object"
}
```

Now, we update our prompt to include this schema. By embedding the JSON schema directly, we provide the LLM with explicit instructions on the expected output format, making it much more likely to adhere to our requirements.

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

response = client.models.generate_content(model=MODEL_ID, contents=prompt)
parsed_response = extract_json_from_response(response.text)
```

Finally, we map the parsed JSON to our `DocumentMetadata` model using `model_validate`. This step attempts to create an instance of our Pydantic class from the dictionary, automatically checking that all required fields are present and that their values match the specified types. If any validation fails, Pydantic will immediately alert us, preventing malformed data from propagating through our system.

```python
try:
    document_metadata = DocumentMetadata.model_validate(parsed_response)
    print("\nValidation successful!")
    print(f"Type of the validated response: `{type(document_metadata)}`")
    print(document_metadata.model_dump_json(indent=2))
except Exception as e:
    print(f"\nValidation failed: {e}")
```

It outputs:
```
Validation successful!
Type of the validated response: `<class '__main__.DocumentMetadata'>`
{
"summary": "The Q3 2023 financial report details a 20% increase in revenue and 15% growth in user engagement, surpassing market expectations. This strong performance is attributed to successful product strategy, market expansion, and improved customer acquisition and retention metrics, providing a solid foundation for continued growth.",
"tags": [
    "Financial Performance",
    "Earnings Report",
    "Business Growth",
    "Revenue Growth",
    "Market Expansion"
],
"keywords": [
    "Q3 2023",
    "revenue increase",
    "user engagement",
    "digital services",
    "new markets",
    "customer acquisition costs",
    "retention rates",
    "cash flow",
    "market expectations"
],
"quarter": "Q3 2023",
"growth_rate": "20%"
}
```

The core idea is to use these validated Pydantic objects throughout your code, not obscure dictionaries where you have to guess what's inside. This eliminates defensive programming and makes your application more robust. While Python's built-in `TypedDict` and `dataclasses` can enforce structure, they don't perform runtime type validation [[3]](https://www.speakeasy.com/blog/pydantic-vs-dataclasses). This is why Pydantic has become the standard for moving data around in LLM workflows and AI agents, offering unparalleled reliability and clarity.

## Implementing structured outputs using Gemini and Pydantic

So far, we've implemented structured outputs from scratch. While this is a great way to understand the mechanics, most modern LLM APIs, like Google's Gemini, provide native support for this functionality. Using the native API is almost always better: it's easier to implement, generally more accurate, and can be more cost-effective because the model is optimized for this task, reducing the need for complex prompt engineering and post-processing [[7]](https://ai.google.dev/gemini-api/docs/structured-output).

Let's see how to achieve the same result using the Gemini SDK's built-in features. The key is the `GenerateContentConfig` object, which allows us to specify the desired output format directly in the API call. We set the `response_mime_type` to `"application/json"` and pass our `DocumentMetadata` Pydantic class directly to the `response_schema` parameter. This tells Gemini exactly what structure and types to expect in its response.

```python
config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=DocumentMetadata)
```

With this configuration, our prompt becomes dramatically simpler. We no longer need to manually inject the JSON schema or provide complex instructions about the output format. We can just ask the model to perform the task, relying on the API to enforce the structure.

```python
prompt = f"""
Analyze the following document and extract its metadata.

Document:
--- 
{DOCUMENT}
--- 
"""
```

Now, when we call the model, the Gemini API handles the schema enforcement internally. It constrains the model's generation process to ensure the output strictly adheres to the Pydantic schema we provided.

```python
response = client.models.generate_content(model=MODEL_ID, contents=prompt, config=config)
```

The best part is that the response object now has a `.parsed` attribute, which gives us direct access to the output as a fully validated Pydantic object. There's no need for manual string cleaning or validation steps; the data is ready to be used immediately in our application.

```python
print(f"Type of the response: `{type(response.parsed)}`")
print(response.parsed.model_dump_json(indent=2))
```

It outputs:
```
Type of the response: `<class '__main__.DocumentMetadata'>`
{
"summary": "The Q3 2023 earnings report reveals a strong financial performance with a 20% increase in revenue and 15% growth in user engagement, surpassing market expectations. This success is attributed to effective product strategy, market expansion, reduced customer acquisition costs, and improved retention rates.",
"tags": [
    "Financial Performance",
    "Earnings Report",
    "Business Growth",
    "Market Expansion",
    "Q3 Results"
],
"keywords": [
    "Q3 2023",
    "revenue increase",
    "user engagement",
    "product strategy",
    "market positioning",
    "digital services",
    "new markets",
    "customer acquisition costs",
    "retention rates",
    "cash flow"
],
"quarter": "Q3 2023",
    "growth_rate": "20%"
}
```

This native approach is cleaner, more reliable, and should be your default choice when the underlying API supports it. It demonstrates how foundational models are evolving to better integrate with traditional software engineering practices, making LLMs more predictable and easier to work with in production.

## Conclusion: Structured Outputs Are Everywhere

We've covered the why and how of structured outputs, from manual prompt engineering to native API integration with Pydantic. This pattern is not just a niche technique; it's a fundamental building block for almost any application you'll create in AI engineering. Whether you're building a simple workflow to summarize articles, a research agent to analyze scientific papers, or a coding agent to write functions, you will need to get structured data back from the LLM.

This concept is universal. It transcends specific models, domains, and application patterns. As we move forward in this course, you will see structured outputs used again and again.

In our next lesson, Lesson 5, we will cover the basic ingredients of LLM workflows, where we'll chain multiple LLM calls together. Structured outputs will be the glue that connects these different steps, allowing us to pass validated data from one component to the next. You'll also see them in action when we discuss agent actions (Lesson 6), planning and reasoning (Lesson 7), and advanced RAG techniques (Lesson 10).

## References

- [1] [Systematic evaluation of large language models for data extraction from unstructured and semi-structured electronic health records](https://pmc.ncbi.nlm.nih.gov/articles/PMC11751965/)
- [2] [A Comparative Study of LLM-based Information Extraction: Direct vs. Indirect Approaches](https://arxiv.org/html/2506.21585v1)
- [3] [Pydantic vs. Dataclasses](https://www.speakeasy.com/blog/pydantic-vs-dataclasses)
- [4] [Automating Knowledge Graphs with LLM Outputs](https://www.prompts.ai/en/blog-details/automating-knowledge-graphs-with-llm-outputs)
- [5] [Structured Data Response with Amazon Bedrock: Prompt Engineering and Tool Use](https://aws.amazon.com/blogs/machine-learning/structured-data-response-with-amazon-bedrock-prompt-engineering-and-tool-use/)
- [6] [Validators Approach in Python: Pydantic vs. Dataclasses](https://codetain.com/blog/validators-approach-in-python-pydantic-vs-dataclasses/)
- [7] [Structured output](https://ai.google.dev/gemini-api/docs/structured-output)
