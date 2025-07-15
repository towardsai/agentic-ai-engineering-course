# Steering Large Language Models with Pydantic

In the last year, there's been a big leap in how we use advanced AI programs, especially in how we communicate with them to get specific tasks done. People are not just making chatbots; they're also using these AIs to sort information, improve their apps, and create synthetic data to train smaller task-specific models.

While some have resorted to [threatening human life](https://twitter.com/goodside/status/1657396491676164096?s=20) to generate structured data, we have found that Pydantic is even more effective.

In this post, we will discuss validating structured outputs from language models using Pydantic and OpenAI. We'll show you how to write reliable code. Additionally, we'll introduce a new library called [instructor](https://github.com/jxnl/instructor) that simplifies this process and offers extra features to leverage validation to improve the quality of your outputs.

## Pydantic

Unlike libraries like `dataclasses`, `Pydantic` goes a step further and defines a schema for your dataclass. This schema is used to validate data, but also to generate documentation and even to generate a JSON schema, which is perfect for our use case of generating structured data with language models!

By providing the model with the following prompt, we can generate a JSON schema for a `PythonPackage` dataclass.

```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

class PythonPackage(BaseModel):
    name: str
    author: str

resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[\
        {\
            "role": "user",\
            "content": "Return the `name`, and `author` of pydantic, in a json object."\
        },\
    ]
)

Package.model_validate_json(resp.choices[0].message.content)
```

If everything is fine, we might receive an output similar to `json.loads({"name": "pydantic", "author": "Samuel Colvin"})`. However, if there is an issue, `resp.choices[0].message.content` could include text or code blocks in prose or markdown format that we need to handle appropriately.

**LLM responses with markdown code blocks**

````python
json.loads("""
```json
{
"name": "pydantic",
"author": "Samuel Colvin"
}
```
""")
>>> JSONDecodeError: Expecting value: line 1 column 1 (char 0

````

**LLM responses with prose**

```python
json.loads("""
Ok heres the authors of pydantic: Samuel Colvin, and the name this library

{
  "name": "pydantic",
  "author": "Samuel Colvin"
}
""")
>>> JSONDecodeError: Expecting value: line 1 column 1 (char 0

```

The content may contain valid JSON, but it isn't considered valid JSON without understanding the language model's behavior. However, it could still provide useful information that we need to handle independently. Fortunately, `OpenAI` offers several options to address this situation.

## Calling Tools

While tool-calling was originally designed to make calls to external APIs using JSON schema, its real value lies in allowing us to specify the desired output format. Fortunately, `Pydantic` provides utilities for generating a JSON schema and supports nested structures, which would be difficult to describe in plain text.

In this example, instead of describing the desired output in plain text, we simply provide the JSON schema for the `Packages` class, which includes a list of `Package` objects:

As an exercise, try prompting the model to generate this prompt without using Pydantic!

Now, notice in this example that the prompts we use contain purely the data we want, where the `tools` and `tool_choice` now capture the schemas we want to output. This separation of concerns makes it much easier to organize the 'data' and the 'description' of the data that we want back out.

```python
from typing import List
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

class PythonPackage(BaseModel):
    name: str
    author: str

class Packages(BaseModel):
    packages: List[PythonPackage]

resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[\
        {\
            "role": "user",\
            "content": "Pydantic and FastAPI?",\
        },\
    ],
    tools=[\
        {\
            "type": "function",\
            "function": {\
                "name": "Requirements",\
                "description": "A list of packages and their first authors.",\
                "parameters": Packages.model_json_schema(),\
            },\
        }\
    ],
    tool_choice={
        "type": "function",
        "function": {"name": "Requirements"},
    },
)

Packages.model_validate_json(
    resp.choices[0].message.tool_calls[0].function.arguments
)
```

```json
{
	"packages": [\
		{\
			"name": "pydantic",\
			"author": "Samuel Colvin"\
		},\
		{\
			"name": "fastapi",\
			"author": "Sebastián Ramírez"\
		}\
	]
}
```

## Using `pip install instructor`

The example we provided above is somewhat contrived, but it illustrates how Pydantic can be utilized to generate structured data from language models. Now, let's employ [Instructor](https://jxnl.github.io/instructor/) to streamline this process. Instructor is a compact library that enhances the OpenAI client by offering convenient features. In the upcoming blog post, we will delve into reasking and validation. However, for now, let's explore a practical example.

```python
# pip install instructor
import instructor

client = instructor.patch(OpenAI())

packages = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[\
        {\
            "role": "user",\
            "content": "Pydantic and FastAPI?",\
        },\
    ],
    response_model=Packages,
)

assert isinstance(resp, Packages)
assert isinstance(resp.packages, list)
assert isinstance(resp.packages[0], Package)
```

## Case Study: Search query segmentation

Let's consider a practical example. Imagine we have a search engine capable of comprehending intricate queries. For instance, if we make a request to find "recent advancements in AI", we could provide the following payload:

```json
{
	"rewritten_query": "novel developments advancements ai artificial intelligence machine learning",
	"published_daterange": {
		"start": "2023-09-17",
		"end": "2021-06-17"
	},
	"domains_allow_list": ["arxiv.org"]
}
```

If we peek under the hood, we can see that the query is actually a complex object, with a date range, and a list of domains to search in. We can model this structured output in Pydantic using the instructor library

```python
from typing import List
import datetime
from pydantic import BaseModel

class DateRange(BaseModel):
    start: datetime.date
    end: datetime.date

class SearchQuery(BaseModel):
    rewritten_query: str
    published_daterange: DateRange
    domains_allow_list: List[str]

    async def execute():
        # Return the search results of the rewritten query
        return api.search(json=self.model_dump())
```

This pattern empowers us to restructure the user's query for improved performance, without requiring the user to understand the inner workings of the search backend.

```python
import instructor
from openai import OpenAI

# Enables response_model in the openai client
client = instructor.patch(OpenAI())

def search(query: str) -> SearchQuery:
    return client.chat.completions.create(
        model="gpt-4",
        response_model=SearchQuery,
        messages=[\
            {\
                "role": "system",\
                "content": f"You're a query understanding system for a search engine. Today's date is {datetime.date.today()}"\
            },\
            {\
                "role": "user",\
                "content": query\
            }\
        ],
    )

search("recent advancements in AI")
```

**Example Output**

```json
{
	"rewritten_query": "novel developments advancements ai artificial intelligence machine learning",
	"published_daterange": {
		"start": "2023-12-15",
		"end": "2023-01-01"
	},
	"domains_allow_list": ["arxiv.org"]
}
```

By defining the api payload as a Pydantic model, we can leverage the `response_model` argument to instruct the model to generate the desired output. This is a powerful feature that allows us to generate structured data from any language model!

In our upcoming posts, we will provide more practical examples and explore how we can leverage `Pydantic`'s validation features to ensure that the data we receive is not only valid syntactically but also semantically.