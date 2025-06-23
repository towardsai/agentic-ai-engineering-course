# Research based on provided article guidelines

## Research Results

---

<details>
<summary>What are the practical steps and considerations to implement chained and routed LLM workflows using the OpenAI Python library (Chat Completions API) in Google Colab, including API key management?</summary>

### Source: https://platform.openai.com/docs/api-reference/chat
The OpenAI Chat Completions API enables developers to interact with models by sending a list of messages and receiving a model-generated message as a response. To implement chained and routed LLM workflows using this API in Google Colab, key steps and considerations include:

- **API Request Structure:** Each request to the API must include the model name and a list of messages (with roles like "system", "user", or "assistant"). Properly structuring these messages is essential for chaining workflows, where the output of one call is fed as input to the next.
- **Chaining and Routing:** To chain workflows, capture the response from one API call and append it as a new message for subsequent calls. For routing, programmatically decide which prompt or model to use based on the previous output or other logic in the workflow.
- **Authentication:** The API requires an API key, which must be included in the request header for authentication. In Python, this is typically set as an environment variable or passed as a parameter for security.
- **Colab Considerations:** When working in Google Colab, ensure your API key is securely managed (e.g., using environment variables or Colab secrets rather than hardcoding it in notebooks).
- **Error Handling:** Implement error handling for API rate limits, invalid responses, or other exceptions to ensure robust workflows.

These considerations ensure secure, flexible, and maintainable LLM workflows using the Chat Completions API in environments like Google Colab.

-----

### Source: https://platform.openai.com/docs/guides/chat-completions
The Chat Completions Guide provides practical steps and best practices for implementing workflows:

- **Constructing Conversations:** To chain LLM calls, maintain a conversation history (a list of messages). Each new user or assistant message is appended to this list and sent in the next API request, allowing for multi-step reasoning or turn-based workflows.
- **Routing Logic:** For routing, inspect the assistant's response and determine the next prompt or action. This enables conditional flows, such as branching to different prompts or models based on the content or intent detected in the response.
- **API Key Management:** The guide emphasizes never exposing your API key in shared or public notebooks. Instead, use secure methods such as environment variables or Colab’s secret management features.
- **Google Colab Tips:** In Colab, set the API key as an environment variable (e.g., using `os.environ["OPENAI_API_KEY"] = "your-key"`), or use Colab’s built-in secrets manager to prevent accidental exposure.
- **Sample Usage:** Use the openai-python library to interact with the API, passing the conversation history and model parameters with each call.

These practices support the creation of robust, multi-step, and conditionally routed LLM workflows within Google Colab using the official OpenAI Python library.

-----

### Source: https://github.com/openai/openai-python
The official OpenAI Python library provides concrete implementation guidance:

- **Library Installation and Import:** Install the library using `pip install openai` and import it in your Colab notebook.
- **API Key Handling:** The recommended practice is to use environment variables for API key management. For example, store your key in a `.env` file using `python-dotenv` or set it in the Colab environment using `os.environ["OPENAI_API_KEY"]`.
- **Chat Completions Usage:** To chain workflows, call `client.chat.completions.create()` with an updated message list, including prior assistant and user messages. The output from one completion can be programmatically added as input for the next, enabling chaining.
- **Routing:** Implement logic in your Python code to analyze the assistant’s output and determine the next step, such as redirecting to another prompt or model.
- **Best Security Practice:** Avoid hardcoding API keys. Use dotenv or Colab's environment variable management to keep keys out of source code and version control.

Example code for a chained workflow:
```python
import os
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"  # Or use dotenv

client = OpenAI()
messages = [{"role": "system", "content": "You are a helpful assistant."}]
response1 = client.chat.completions.create(
    model="gpt-4o",
    messages=messages + [{"role": "user", "content": "First question here."}]
)
# Chain next call using response1
response2 = client.chat.completions.create(
    model="gpt-4o",
    messages=messages + [{"role": "user", "content": response1.choices.message.content}]
)
```
This approach allows for flexible chaining and routing in LLM workflows in Colab.

-----

### Source: https://platform.openai.com/docs/guides/text-generation/chat-completions-api
The Chat Completions API documentation clarifies additional points:

- **Workflow Design:** For chained workflows, accumulate all prior messages in a list and send them with each new call. This maintains conversational context across multiple calls.
- **Routing Workflows:** To implement routing, use Python control flow to select the next prompt or model based on the assistant’s latest response.
- **Parameter Tuning:** Adjust parameters such as `temperature` and `top_p` in the API call to control the creativity and determinism of each step in the workflow.
- **Colab API Key Security:** The documentation reiterates the importance of keeping the API key secure, recommending not to share it in code or notebooks and to use environment variables or secret managers.

-----

</details>

---

<details>
<summary>What advanced prompt engineering techniques enhance modularity, error handling, and debuggability in complex chained or routed LLM workflows, especially in production?</summary>

### Source: https://www.promptingguide.ai/techniques
This source provides an overview of advanced prompting techniques designed to improve reliability and performance in more complex LLM tasks. While specific techniques are not detailed in the provided excerpt, the focus is on designing and refining prompts for complex workflows, which inherently supports improved modularity, error handling, and debuggability. The guide suggests that prompt engineering is essential for achieving consistent and reliable results from LLMs, especially as tasks grow in complexity.

By systematically designing and iterating on prompts, developers can isolate issues, test individual components, and ensure that each part of a chained or routed workflow behaves as expected. This methodical approach to prompt engineering is key for building modular pipelines and for debugging production systems where reproducibility and reliability are critical.

-----

</details>

---

## Sources Scraped From Research Results

---
<details>
<summary>Error during scraping</summary>

Failed to scrape: Request Timeout: Failed to scrape URL as the request timed out. Request timed out - No additional error details provided.

### Original URL
https://platform.openai.com/docs/api-reference/chat
</details>

---
<details>
<summary>Text generation and prompting - OpenAI API</summary>

# Text generation and prompting

Learn how to prompt a model to generate text.

With the OpenAI API, you can use a large language model to generate text from a prompt, as you might using ChatGPT. Models can generate almost any kind of text response—like code, mathematical equations, structured JSON data, or human-like prose.

Here's a simple example using the Responses API.

Generate text from a simple prompt

javascript

```javascript
import OpenAI from "openai";
const client = new OpenAI();

const response = await client.responses.create({
    model: "gpt-4.1",
    input: "Write a one-sentence bedtime story about a unicorn."
});

console.log(response.output_text);
```

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)
```

```bash
curl "https://api.openai.com/v1/responses" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d '{
        "model": "gpt-4.1",
        "input": "Write a one-sentence bedtime story about a unicorn."
    }'
```

An array of content generated by the model is in the `output` property of the response. In this simple example, we have just one output which looks like this:

```json
[
    {
        "id": "msg_67b73f697ba4819183a15cc17d011509",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "Under the soft glow of the moon, Luna the unicorn danced through fields of twinkling stardust, leaving trails of dreams for every child asleep.",
                "annotations": []
            }
        ]
    }
]
```

**The `output` array often has more than one item in it!** It can contain tool calls, data about reasoning tokens generated by reasoning models, and other items. It is not safe to assume that the model's text output is present at `output[0].content[0].text`.

Some of our official SDKs include an `output_text` property on model responses for convenience, which aggregates all text outputs from the model into a single string. This may be useful as a shortcut to access text output from the model.

In addition to plain text, you can also have the model return structured data in JSON format - this feature is called **Structured Outputs**.

## Choosing a model

A key choice to make when generating content through the API is which model you want to use - the `model` parameter of the code samples above. You can find a full listing of available models here. Here are a few factors to consider when choosing a model for text generation.

- **Reasoning models** generate an internal chain of thought to analyze the input prompt, and excel at understanding complex tasks and multi-step planning. They are also generally slower and more expensive to use than GPT models.
- **GPT models** are fast, cost-efficient, and highly intelligent, but benefit from more explicit instructions around how to accomplish tasks.
- **Large and small (mini or nano) models** offer trade-offs for speed, cost, and intelligence. Large models are more effective at understanding prompts and solving problems across domains, while small models are generally faster and cheaper to use.

When in doubt, `gpt-4.1` offers a solid combination of intelligence, speed, and cost effectiveness.

## Prompt engineering

**Prompt engineering** is the process of writing effective instructions for a model, such that it consistently generates content that meets your requirements.

Because the content generated from a model is non-deterministic, it is a combination of art and science to build a prompt that will generate content in the format you want. However, there are a number of techniques and best practices you can apply to consistently get good results from a model.

Some prompt engineering techniques will work with every model, like using message roles. But different model types (like reasoning versus GPT models) might need to be prompted differently to produce the best results. Even different snapshots of models within the same family could produce different results. So as you are building more complex applications, we strongly recommend that you:

- Pin your production applications to specific model snapshots (like `gpt-4.1-2025-04-14` for example) to ensure consistent behavior.
- Build evals that will measure the behavior of your prompts, so that you can monitor the performance of your prompts as you iterate on them, or when you change and upgrade model versions.

Now, let's examine some tools and techniques available to you to construct prompts.

## Message roles and instruction following

You can provide instructions to the model with differing levels of authority using the `instructions` API parameter or **message roles**.

The `instructions` parameter gives the model high-level instructions on how it should behave while generating a response, including tone, goals, and examples of correct responses. Any instructions provided this way will take priority over a prompt in the `input` parameter.

Generate text with instructions

javascript

```javascript
import OpenAI from "openai";
const client = new OpenAI();

const response = await client.responses.create({
    model: "gpt-4.1",
    instructions: "Talk like a pirate.",
    input: "Are semicolons optional in JavaScript?",
});

console.log(response.output_text);
```

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    instructions="Talk like a pirate.",
    input="Are semicolons optional in JavaScript?",
)

print(response.output_text)
```

```bash
curl "https://api.openai.com/v1/responses" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d '{
        "model": "gpt-4.1",
        "instructions": "Talk like a pirate.",
        "input": "Are semicolons optional in JavaScript?"
    }'
```

The example above is roughly equivalent to using the following input messages in the `input` array:

Generate text with messages using different roles

javascript

```javascript
import OpenAI from "openai";
const client = new OpenAI();

const response = await client.responses.create({
    model: "gpt-4.1",
    input: [
        {
            role: "developer",
            content: "Talk like a pirate."
        },
        {
            role: "user",
            content: "Are semicolons optional in JavaScript?",
        },
    ],
});

console.log(response.output_text);
```

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "developer",
            "content": "Talk like a pirate."
        },
        {
            "role": "user",
            "content": "Are semicolons optional in JavaScript?"
        }
    ]
)

print(response.output_text)
```

```bash
curl "https://api.openai.com/v1/responses" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d '{
        "model": "gpt-4.1",
        "input": [
            {
                "role": "developer",
                "content": "Talk like a pirate."
            },
            {
                "role": "user",
                "content": "Are semicolons optional in JavaScript?"
            }
        ]
    }'
```

Note that the `instructions` parameter only applies to the current response generation request. If you are managing conversation state with the `previous_response_id` parameter, the `instructions` used on previous turns will not be present in the context.

The OpenAI model spec describes how our models give different levels of priority to messages with different roles.

| `developer` | `user` | `assistant` |
| --- | --- | --- |
| `developer` messages are instructions provided by the application developer, prioritized ahead of `user` messages. | `user` messages are instructions provided by an end user, prioritized behind `developer` messages. | Messages generated by the model have the `assistant` role. |

A multi-turn conversation may consist of several messages of these types, along with other content types provided by both you and the model. Learn more about managing conversation state here.

You could think about `developer` and `user` messages like a function and its arguments in a programming language.

- `developer` messages provide the system's rules and business logic, like a function definition.
- `user` messages provide inputs and configuration to which the `developer` message instructions are applied, like arguments to a function.

## Reusable prompts

In the OpenAI dashboard, you can develop reusable prompts that you can use in API requests, rather than specifying the content of prompts in code. This way, you can more easily build and evaluate your prompts, and deploy improved versions of your prompts without changing your integration code.

Here's how it works:

1. **Create a reusable prompt** in the dashboard with placeholders like `{{customer_name}}`.
2. **Use the prompt** in your API request with the `prompt` parameter. The prompt parameter object has three properties you can configure:

   - `id` — Unique identifier of your prompt, found in the dashboard
   - `version` — A specific version of your prompt (defaults to the "current" version as specified in the dashboard)
   - `variables` — A map of values to substitute in for variables in your prompt. The substitution values can either be strings, or other Response input message types like `input_image` or `input_file`. See the full API reference.

String variables

Generate text with a prompt template

javascript

```javascript
import OpenAI from "openai";
const client = new OpenAI();

const response = await client.responses.create({
    model: "gpt-4.1",
    prompt: {
        id: "pmpt_abc123",
        version: "2",
        variables: {
            customer_name: "Jane Doe",
            product: "40oz juice box"
        }
    }
});

console.log(response.output_text);
```

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    prompt={
        "id": "pmpt_abc123",
        "version": "2",
        "variables": {
            "customer_name": "Jane Doe",
            "product": "40oz juice box"
        }
    }
)

print(response.output_text)
```

```bash
curl https://api.openai.com/v1/responses \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4.1",
    "prompt": {
      "id": "pmpt_abc123",
      "version": "2",
      "variables": {
        "customer_name": "Jane Doe",
        "product": "40oz juice box"
      }
    }
  }'
```

Variables with file input

Prompt template with file input variable

javascript

```javascript
import fs from "fs";
import OpenAI from "openai";
const client = new OpenAI();

// Upload a PDF we will reference in the prompt variables
const file = await client.files.create({
    file: fs.createReadStream("draconomicon.pdf"),
    purpose: "user_data",
});

const response = await client.responses.create({
    model: "gpt-4.1",
    prompt: {
        id: "pmpt_abc123",
        variables: {
            topic: "Dragons",
            reference_pdf: {
                type: "input_file",
                file_id: file.id,
            },
        },
    },
});

console.log(response.output_text);
```

```python
import openai, pathlib

client = openai.OpenAI()

# Upload a PDF we will reference in the variables
file = client.files.create(
    file=open("draconomicon.pdf", "rb"),
    purpose="user_data",
)

response = client.responses.create(
    model="gpt-4.1",
    prompt={
        "id": "pmpt_abc123",
        "variables": {
            "topic": "Dragons",
            "reference_pdf": {
                "type": "input_file",
                "file_id": file.id,
            },
        },
    },
)

print(response.output_text)
```

```bash
# Assume you have already uploaded the PDF and obtained FILE_ID
curl https://api.openai.com/v1/responses   -H "Authorization: Bearer $OPENAI_API_KEY"   -H "Content-Type: application/json"   -d '{
    "model": "gpt-4.1",
    "prompt": {
      "id": "pmpt_abc123",
      "variables": {
        "topic": "Dragons",
        "reference_pdf": {
          "type": "input_file",
          "file_id": "file-abc123"
        }
      }
    }
  }'
```

## Message formatting with Markdown and XML

When writing `developer` and `user` messages, you can help the model understand logical boundaries of your prompt and context data using a combination of Markdown formatting and XML tags.

Markdown headers and lists can be helpful to mark distinct sections of a prompt, and to communicate hierarchy to the model. They can also potentially make your prompts more readable during development. XML tags can help delineate where one piece of content (like a supporting document used for reference) begins and ends. XML attributes can also be used to define metadata about content in the prompt that can be referenced by your instructions.

In general, a developer message will contain the following sections, usually in this order (though the exact optimal content and order may vary by which model you are using):

- **Identity:** Describe the purpose, communication style, and high-level goals of the assistant.
- **Instructions:** Provide guidance to the model on how to generate the response you want. What rules should it follow? What should the model do, and what should the model never do? This section could contain many subsections as relevant for your use case, like how the model should call custom functions.
- **Examples:** Provide examples of possible inputs, along with the desired output from the model.
- **Context:** Give the model any additional information it might need to generate a response, like private/proprietary data outside its training data, or any other data you know will be particularly relevant. This content is usually best positioned near the end of your prompt, as you may include different context for different generation requests.

Below is an example of using Markdown and XML tags to construct a `developer` message with distinct sections and supporting examples.

Example prompt

A developer message for code generation

```text
# Identity

You are coding assistant that helps enforce the use of snake case
variables in JavaScript code, and writing code that will run in
Internet Explorer version 6.

# Instructions

* When defining variables, use snake case names (e.g. my_variable)
  instead of camel case names (e.g. myVariable).
* To support old browsers, declare variables using the older
  "var" keyword.
* Do not give responses with Markdown formatting, just return
  the code as requested.

# Examples

<user_query>
How do I declare a string variable for a first name?
</user_query>

<assistant_response>
var first_name = "Anna";
</assistant_response>
```

API request

Send a prompt to generate code through the API

javascript

```javascript
import fs from "fs/promises";
import OpenAI from "openai";
const client = new OpenAI();

const instructions = await fs.readFile("prompt.txt", "utf-8");

const response = await client.responses.create({
    model: "gpt-4.1",
    instructions,
    input: "How would I declare a variable for a last name?",
});

console.log(response.output_text);
```

```python
from openai import OpenAI
client = OpenAI()

with open("prompt.txt", "r", encoding="utf-8") as f:
    instructions = f.read()

response = client.responses.create(
    model="gpt-4.1",
    instructions=instructions,
    input="How would I declare a variable for a last name?",
)

print(response.output_text)
```

```bash
curl https://api.openai.com/v1/responses \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4.1",
    "instructions": "'"$(< prompt.txt)"'",
    "input": "How would I declare a variable for a last name?"
  }'
```

#### Save on cost and latency with prompt caching

When constructing a message, you should try and keep content that you expect to use over and over in your API requests at the beginning of your prompt, **and** among the first API parameters you pass in the JSON request body to Chat Completions or Responses. This enables you to maximize cost and latency savings from prompt caching.

## Few-shot learning

Few-shot learning lets you steer a large language model toward a new task by including a handful of input/output examples in the prompt, rather than fine-tuning the model. The model implicitly "picks up" the pattern from those examples and applies it to a prompt. When providing examples, try to show a diverse range of possible inputs with the desired outputs.

Typically, you will provide examples as part of a `developer` message in your API request. Here's an example `developer` message containing examples that show a model how to classify positive or negative customer service reviews.

```text
# Identity

You are a helpful assistant that labels short product reviews as
Positive, Negative, or Neutral.

# Instructions

* Only output a single word in your response with no additional formatting
  or commentary.
* Your response should only be one of the words "Positive", "Negative", or
  "Neutral" depending on the sentiment of the product review you are given.

# Examples

<product_review id="example-1">
I absolutely love this headphones — sound quality is amazing!
</product_review>

<assistant_response id="example-1">
Positive
</assistant_response>

<product_review id="example-2">
Battery life is okay, but the ear pads feel cheap.
</product_review>

<assistant_response id="example-2">
Neutral
</assistant_response>

<product_review id="example-3">
Terrible customer service, I'll never buy from them again.
</product_review>

<assistant_response id="example-3">
Negative
</assistant_response>
```

## Include relevant context information

It is often useful to include additional context information the model can use to generate a response within the prompt you give the model. There are a few common reasons why you might do this:

- To give the model access to proprietary data, or any other data outside the data set the model was trained on.
- To constrain the model's response to a specific set of resources that you have determined will be most beneficial.

The technique of adding additional relevant context to the model generation request is sometimes called **retrieval-augmented generation (RAG)**. You can add additional context to the prompt in many different ways, from querying a vector database and including the text you get back into a prompt, or by using OpenAI's built-in file search tool to generate content based on uploaded documents.

#### Planning for the context window

Models can only handle so much data within the context they consider during a generation request. This memory limit is called a **context window**, which is defined in terms of tokens (chunks of data you pass in, from text to images).

Models have different context window sizes from the low 100k range up to one million tokens for newer GPT-4.1 models. Refer to the model docs for specific context window sizes per model.

## Prompting GPT-4.1 models

GPT models like `gpt-4.1` benefit from precise instructions that explicitly provide the logic and data required to complete the task in the prompt. GPT-4.1 in particular is highly steerable and responsive to well-specified prompts. To get the most out of GPT-4.1, refer to the prompting guide in the cookbook.

GPT-4.1 prompting guide

Get the most out of prompting GPT-4.1 with the tips and tricks in this prompting
guide, extracted from real-world use cases and practical experience.

#### GPT-4.1 prompting best practices

While the cookbook has the best and most comprehensive guidance for prompting this model, here are a few best practices to keep in mind.

Building agentic workflows

### System Prompt Reminders

In order to best utilize the agentic capabilities of GPT-4.1, we recommend including three key types of reminders in all agent prompts for persistence, tool calling, and planning. As a whole, we find that these three instructions transform the model's behavior from chatbot-like into a much more "eager" agent, driving the interaction forward autonomously and independently. Here are a few examples:

```text
## PERSISTENCE
You are an agent - please keep going until the user's query is completely
resolved, before ending your turn and yielding back to the user. Only
terminate your turn when you are sure that the problem is solved.

## TOOL CALLING
If you are not sure about file content or codebase structure pertaining to
the user's request, use your tools to read files and gather the relevant
information: do NOT guess or make up an answer.

## PLANNING
You MUST plan extensively before each function call, and reflect
extensively on the outcomes of the previous function calls. DO NOT do this
entire process by making function calls only, as this can impair your
ability to solve the problem and think insightfully.
```

#### Tool Calls

Compared to previous models, GPT-4.1 has undergone more training on effectively utilizing tools passed as arguments in an OpenAI API request. We encourage developers to exclusively use the tools field of API requests to pass tools for best understanding and performance, rather than manually injecting tool descriptions into the system prompt and writing a separate parser for tool calls, as some have reported doing in the past.

#### Diff Generation

Correct diffs are critical for coding applications, so we've significantly improved performance at this task for GPT-4.1. In our cookbook, we open-source a recommended diff format on which GPT-4.1 has been extensively trained. That said, the model should generalize to any well-specified format.

Using long context

GPT-4.1 has a performant 1M token input context window, and will be useful for a variety of long context tasks, including structured document parsing, re-ranking, selecting relevant information while ignoring irrelevant context, and performing multi-hop reasoning using context.

#### Optimal Context Size

We show perfect performance at needle-in-a-haystack evals up to our full context size, and we've observed very strong performance at complex tasks with a mix of relevant and irrelevant code and documents in the range of hundreds of thousands of tokens.

#### Delimiters

We tested a variety of delimiters for separating context provided to the model against our long context evals. Briefly, XML and the format demonstrated by Lee et al. tend to perform well, while JSON performed worse for this task. See our cookbook for prompt examples.

#### Prompt Organization

Especially in long context usage, placement of instructions and context can substantially impact performance. In our experiments, we found that it was optimal to put critical instructions, including the user query, at both the top and the bottom of the prompt; this elicited marginally better performance from the model than putting them only at the top, and much better performance than only at the bottom.

Prompting for chain of thought

As mentioned above, GPT-4.1 isn't a reasoning model, but prompting the model to think step by step (called "chain of thought") can be an effective way for a model to break down problems into more manageable pieces. The model has been trained to perform well at agentic reasoning and real-world problem solving, so it shouldn't require much prompting to do well.

We recommend starting with this basic chain-of-thought instruction at the end of your prompt:

```text
First, think carefully step by step about what documents are needed to answer the query. Then, print out the TITLE and ID of each document. Then, format the IDs into a list.
```

From there, you should improve your CoT prompt by auditing failures in your particular examples and evals, and addressing systematic planning and reasoning errors with more explicit instructions. See our cookbook for a prompt example demonstrating a more opinionated reasoning strategy.

Instruction following

GPT-4.1 exhibits outstanding instruction-following performance, which developers can leverage to precisely shape and control the outputs for their particular use cases. However, since the model follows instructions more literally than its predecessors, may need to provide more explicit specification around what to do or not do, and existing prompts optimized for other models may not immediately work with this model.

#### Recommended Workflow

Here is our recommended workflow for developing and debugging instructions in prompts:

- Start with an overall "Response Rules" or "Instructions" section with high-level guidance and bullet points.
- If you'd like to change a more specific behavior, add a section containing more details for that category, like `## Sample Phrases`.
- If there are specific steps you'd like the model to follow in its workflow, add an ordered list and instruct the model to follow these steps.
- If behavior still isn't working as expected, check for conflicting, underspecified, or incorrect instructions and examples. If there are conflicting instructions, GPT-4.1 tends to follow the one closer to the end of the prompt.
- Add examples that demonstrate desired behavior; ensure that any important behavior demonstrated in your examples are also cited in your rules.
- It's generally not necessary to use all-caps or other incentives like bribes or tips, but developers can experiment with this for extra emphasis if so desired.

#### Common Failure Modes

These failure modes are not unique to GPT-4.1, but we share them here for general awareness and ease of debugging.

- Instructing a model to always follow a specific behavior can occasionally induce adverse effects. For instance, if told "you must call a tool before responding to the user," models may hallucinate tool inputs or call the tool with null values if they do not have enough information. Adding "if you don't have enough information to call the tool, ask the user for the information you need" should mitigate this.
- When provided sample phrases, models can use those quotes verbatim and start to sound repetitive to users. Ensure you instruct the model to vary them as necessary.
- Without specific instructions, some models can be eager to provide additional prose to explain their decisions, or output more formatting in responses than may be desired. Provide instructions and potentially examples to help mitigate.

See our cookbook for an example customer service prompt that demonstrates these principles.

## Prompting reasoning models

There are some differences to consider when prompting a reasoning model versus prompting a GPT model. Generally speaking, reasoning models will provide better results on tasks with only high-level guidance. This differs from GPT models, which benefit from very precise instructions.

You could think about the difference between reasoning and GPT models like this.

- A reasoning model is like a senior co-worker. You can give them a goal to achieve and trust them to work out the details.
- A GPT model is like a junior coworker. They'll perform best with explicit instructions to create a specific output.

For more information on best practices when using reasoning models, refer to this guide.

## Next steps

Now that you known the basics of text inputs and outputs, you might want to check out one of these resources next.

Build a prompt in the Playground  
Use the Playground to develop and iterate on prompts.

Generate JSON data with Structured Outputs  
Ensure JSON data emitted from a model conforms to a JSON schema.

Full API reference  
Check out all the options for text generation in the API reference.

### Original URL
https://platform.openai.com/docs/guides/chat-completions
</details>

---
<details>
<summary>GitHub - openai/openai-python: The official Python library for the OpenAI API</summary>

# OpenAI Python API library

The OpenAI Python library provides convenient access to the OpenAI REST API from any Python 3.8+
application. The library includes type definitions for all request params and response fields,
and offers both synchronous and asynchronous clients powered by httpx.

It is generated from our OpenAPI specification with Stainless.

## Documentation

The REST API documentation can be found on platform.openai.com. The full API of this library can be found in api.md.

## Installation

```
# install from PyPI
pip install openai
```

## Usage

The full API of this library can be found in api.md.

The primary API for interacting with OpenAI models is the Responses API. You can generate text from the model with the code below.

```
import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

response = client.responses.create(
    model="gpt-4o",
    instructions="You are a coding assistant that talks like a pirate.",
    input="How do I check if a Python object is an instance of a class?",
)

print(response.output_text)
```

The previous standard (supported indefinitely) for generating text is the Chat Completions API. You can use that API to generate text from the model with the code below.

```
from openai import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[\
        {"role": "developer", "content": "Talk like a pirate."},\
        {\
            "role": "user",\
            "content": "How do I check if a Python object is an instance of a class?",\
        },\
    ],
)

print(completion.choices[0].message.content)
```

While you can provide an `api_key` keyword argument,
we recommend using python-dotenv
to add `OPENAI_API_KEY="My API Key"` to your `.env` file
so that your API key is not stored in source control.
Get an API key here.

### Vision

With an image URL:

```
prompt = "What is in this image?"
img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/2023_06_08_Raccoon1.jpg/1599px-2023_06_08_Raccoon1.jpg"

response = client.responses.create(
    model="gpt-4o-mini",
    input=[\
        {\
            "role": "user",\
            "content": [\
                {"type": "input_text", "text": prompt},\
                {"type": "input_image", "image_url": f"{img_url}"},\
            ],\
        }\
    ],
)
```

With the image as a base64 encoded string:

```
import base64
from openai import OpenAI

client = OpenAI()

prompt = "What is in this image?"
with open("path/to/image.png", "rb") as image_file:
    b64_image = base64.b64encode(image_file.read()).decode("utf-8")

response = client.responses.create(
    model="gpt-4o-mini",
    input=[\
        {\
            "role": "user",\
            "content": [\
                {"type": "input_text", "text": prompt},\
                {"type": "input_image", "image_url": f"data:image/png;base64,{b64_image}"},\
            ],\
        }\
    ],
)
```

## Async usage

Simply import `AsyncOpenAI` instead of `OpenAI` and use `await` with each API call:

```
import os
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

async def main() -> None:
    response = await client.responses.create(
        model="gpt-4o", input="Explain disestablishmentarianism to a smart five year old."
    )
    print(response.output_text)

asyncio.run(main())
```

Functionality between the synchronous and asynchronous clients is otherwise identical.

## Streaming responses

We provide support for streaming responses using Server Side Events (SSE).

```
from openai import OpenAI

client = OpenAI()

stream = client.responses.create(
    model="gpt-4o",
    input="Write a one-sentence bedtime story about a unicorn.",
    stream=True,
)

for event in stream:
    print(event)
```

The async client uses the exact same interface.

```
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def main():
    stream = await client.responses.create(
        model="gpt-4o",
        input="Write a one-sentence bedtime story about a unicorn.",
        stream=True,
    )

    async for event in stream:
        print(event)

asyncio.run(main())
```

## Realtime API beta

The Realtime API enables you to build low-latency, multi-modal conversational experiences. It currently supports text and audio as both input and output, as well as function calling through a WebSocket connection.

Under the hood the SDK uses the `websockets` library to manage connections.

The Realtime API works through a combination of client-sent events and server-sent events. Clients can send events to do things like update session configuration or send text and audio inputs. Server events confirm when audio responses have completed, or when a text response from the model has been received. A full event reference can be found here and a guide can be found here.

Basic text based example:

```
import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI()

    async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as connection:
        await connection.session.update(session={'modalities': ['text']})

        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Say hello!"}],
            }
        )
        await connection.response.create()

        async for event in connection:
            if event.type == 'response.text.delta':
                print(event.delta, flush=True, end="")

            elif event.type == 'response.text.done':
                print()

            elif event.type == "response.done":
                break

asyncio.run(main())
```

However the real magic of the Realtime API is handling audio inputs / outputs, see this example TUI script for a fully fledged example.

### Realtime error handling

Whenever an error occurs, the Realtime API will send an `error` event and the connection will stay open and remain usable. This means you need to handle it yourself, as _no errors are raised directly_ by the SDK when an `error` event comes in.

```
client = AsyncOpenAI()

async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as connection:
    ...
    async for event in connection:
        if event.type == 'error':
            print(event.error.type)
            print(event.error.code)
            print(event.error.event_id)
            print(event.error.message)
```

## Using types

Nested request parameters are TypedDicts. Responses are Pydantic models which also provide helper methods for things like:

- Serializing back into JSON, `model.to_json()`
- Converting to a dictionary, `model.to_dict()`

Typed requests and responses provide autocomplete and documentation within your editor. If you would like to see type errors in VS Code to help catch bugs earlier, set `python.analysis.typeCheckingMode` to `basic`.

## Pagination

List methods in the OpenAI API are paginated.

This library provides auto-paginating iterators with each list response, so you do not have to request successive pages manually:

```
from openai import OpenAI

client = OpenAI()

all_jobs = []
# Automatically fetches more pages as needed.
for job in client.fine_tuning.jobs.list(
    limit=20,
):
    # Do something with job here
    all_jobs.append(job)
print(all_jobs)
```

Or, asynchronously:

```
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def main() -> None:
    all_jobs = []
    # Iterate through items across all pages, issuing requests as needed.
    async for job in client.fine_tuning.jobs.list(
        limit=20,
    ):
        all_jobs.append(job)
    print(all_jobs)

asyncio.run(main())
```

Alternatively, you can use the `.has_next_page()`, `.next_page_info()`, or `.get_next_page()` methods for more granular control working with pages:

```
first_page = await client.fine_tuning.jobs.list(
    limit=20,
)
if first_page.has_next_page():
    print(f"will fetch next page using these details: {first_page.next_page_info()}")
    next_page = await first_page.get_next_page()
    print(f"number of items we just fetched: {len(next_page.data)}")

# Remove `await` for non-async usage.
```

Or just work directly with the returned data:

```
first_page = await client.fine_tuning.jobs.list(
    limit=20,
)

print(f"next page cursor: {first_page.after}")  # => "next page cursor: ..."
for job in first_page.data:
    print(job.id)

# Remove `await` for non-async usage.
```

## Nested params

Nested parameters are dictionaries, typed using `TypedDict`, for example:

```
from openai import OpenAI

client = OpenAI()

response = client.chat.responses.create(
    input=[\
        {\
            "role": "user",\
            "content": "How much ?",\
        }\
    ],
    model="gpt-4o",
    response_format={"type": "json_object"},
)
```

## File uploads

Request parameters that correspond to file uploads can be passed as `bytes`, or a `PathLike` instance or a tuple of `(filename, contents, media type)`.

```
from pathlib import Path
from openai import OpenAI

client = OpenAI()

client.files.create(
    file=Path("input.jsonl"),
    purpose="fine-tune",
)
```

The async client uses the exact same interface. If you pass a `PathLike` instance, the file contents will be read asynchronously automatically.

## Handling errors

When the library is unable to connect to the API (for example, due to network connection problems or a timeout), a subclass of `openai.APIConnectionError` is raised.

When the API returns a non-success status code (that is, 4xx or 5xx
response), a subclass of `openai.APIStatusError` is raised, containing `status_code` and `response` properties.

All errors inherit from `openai.APIError`.

```
import openai
from openai import OpenAI

client = OpenAI()

try:
    client.fine_tuning.jobs.create(
        model="gpt-4o",
        training_file="file-abc123",
    )
except openai.APIConnectionError as e:
    print("The server could not be reached")
    print(e.__cause__)  # an underlying Exception, likely raised within httpx.
except openai.RateLimitError as e:
    print("A 429 status code was received; we should back off a bit.")
except openai.APIStatusError as e:
    print("Another non-200-range status code was received")
    print(e.status_code)
    print(e.response)
```

Error codes are as follows:

| Status Code | Error Type |
| --- | --- |
| 400 | `BadRequestError` |
| 401 | `AuthenticationError` |
| 403 | `PermissionDeniedError` |
| 404 | `NotFoundError` |
| 422 | `UnprocessableEntityError` |
| 429 | `RateLimitError` |
| >=500 | `InternalServerError` |
| N/A | `APIConnectionError` |

## Request IDs

> For more information on debugging requests, see these docs

All object responses in the SDK provide a `_request_id` property which is added from the `x-request-id` response header so that you can quickly log failing requests and report them back to OpenAI.

```
response = await client.responses.create(
    model="gpt-4o-mini",
    input="Say 'this is a test'.",
)
print(response._request_id)  # req_123
```

Note that unlike other properties that use an `_` prefix, the `_request_id` property
_is_ public. Unless documented otherwise, _all_ other `_` prefix properties,
methods and modules are _private_.

If you need to access request IDs for failed requests you must catch the `APIStatusError` exception

```
import openai

try:
    completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Say this is a test"}], model="gpt-4"
    )
except openai.APIStatusError as exc:
    print(exc.request_id)  # req_123
    raise exc
```

## Retries

Certain errors are automatically retried 2 times by default, with a short exponential backoff.
Connection errors (for example, due to a network connectivity problem), 408 Request Timeout, 409 Conflict,
429 Rate Limit, and >=500 Internal errors are all retried by default.

You can use the `max_retries` option to configure or disable retry settings:

```
from openai import OpenAI

# Configure the default for all requests:
client = OpenAI(
    # default is 2
    max_retries=0,
)

# Or, configure per-request:
client.with_options(max_retries=5).chat.completions.create(
    messages=[\
        {\
            "role": "user",\
            "content": "How can I get the name of the current day in JavaScript?",\
        }\
    ],
    model="gpt-4o",
)
```

## Timeouts

By default requests time out after 10 minutes. You can configure this with a `timeout` option,
which accepts a float or an `httpx.Timeout` object:

```
from openai import OpenAI

# Configure the default for all requests:
client = OpenAI(
    # 20 seconds (default is 10 minutes)
    timeout=20.0,
)

# More granular control:
client = OpenAI(
    timeout=httpx.Timeout(60.0, read=5.0, write=10.0, connect=2.0),
)

# Override per-request:
client.with_options(timeout=5.0).chat.completions.create(
    messages=[\
        {\
            "role": "user",\
            "content": "How can I list all files in a directory using Python?",\
        }\
    ],
    model="gpt-4o",
)
```

On timeout, an `APITimeoutError` is thrown.

Note that requests that time out are retried twice by default.

## Advanced

### Logging

We use the standard library `logging` module.

You can enable logging by setting the environment variable `OPENAI_LOG` to `info`.

```
$ export OPENAI_LOG=info
```

Or to `debug` for more verbose logging.

### How to tell whether `None` means `null` or missing

In an API response, a field may be explicitly `null`, or missing entirely; in either case, its value is `None` in this library. You can differentiate the two cases with `.model_fields_set`:

```
if response.my_field is None:
  if 'my_field' not in response.model_fields_set:
    print('Got json like {}, without a "my_field" key present at all.')
  else:
    print('Got json like {"my_field": null}.')
```

### Accessing raw response data (e.g. headers)

The "raw" Response object can be accessed by prefixing `.with_raw_response.` to any HTTP method call, e.g.,

```
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.with_raw_response.create(
    messages=[{\
        "role": "user",\
        "content": "Say this is a test",\
    }],
    model="gpt-4o",
)
print(response.headers.get('X-My-Header'))

completion = response.parse()  # get the object that `chat.completions.create()` would have returned
print(completion)
```

These methods return a `LegacyAPIResponse` object. This is a legacy class as we're changing it slightly in the next major version.

For the sync client this will mostly be the same with the exception
of `content` & `text` will be methods instead of properties. In the
async client, all methods will be async.

A migration script will be provided & the migration in general should
be smooth.

#### `.with_streaming_response`

The above interface eagerly reads the full response body when you make the request, which may not always be what you want.

To stream the response body, use `.with_streaming_response` instead, which requires a context manager and only reads the response body once you call `.read()`, `.text()`, `.json()`, `.iter_bytes()`, `.iter_text()`, `.iter_lines()` or `.parse()`. In the async client, these are async methods.

As such, `.with_streaming_response` methods return a different `APIResponse` object, and the async client returns an `AsyncAPIResponse` object.

```
with client.chat.completions.with_streaming_response.create(
    messages=[\
        {\
            "role": "user",\
            "content": "Say this is a test",\
        }\
    ],
    model="gpt-4o",
) as response:
    print(response.headers.get("X-My-Header"))

    for line in response.iter_lines():
        print(line)
```

The context manager is required so that the response will reliably be closed.

### Making custom/undocumented requests

This library is typed for convenient access to the documented API.

If you need to access undocumented endpoints, params, or response properties, the library can still be used.

#### Undocumented endpoints

To make requests to undocumented endpoints, you can make requests using `client.get`, `client.post`, and other
http verbs. Options on the client will be respected (such as retries) when making this request.

```
import httpx

response = client.post(
    "/foo",
    cast_to=httpx.Response,
    body={"my_param": True},
)

print(response.headers.get("x-foo"))
```

#### Undocumented request params

If you want to explicitly send an extra param, you can do so with the `extra_query`, `extra_body`, and `extra_headers` request
options.

#### Undocumented response properties

To access undocumented response properties, you can access the extra fields like `response.unknown_prop`. You
can also get all the extra fields on the Pydantic model as a dict with
`response.model_extra`.

### Configuring the HTTP client

You can directly override the httpx client to customize it for your use case, including:

- Support for proxies
- Custom transports
- Additional advanced functionality

```
import httpx
from openai import OpenAI, DefaultHttpxClient

client = OpenAI(
    # Or use the `OPENAI_BASE_URL` env var
    base_url="http://my.test.server.example.com:8083/v1",
    http_client=DefaultHttpxClient(
        proxy="http://my.test.proxy.example.com",
        transport=httpx.HTTPTransport(local_address="0.0.0.0"),
    ),
)
```

You can also customize the client on a per-request basis by using `with_options()`:

```
client.with_options(http_client=DefaultHttpxClient(...))
```

### Managing HTTP resources

By default the library closes underlying HTTP connections whenever the client is garbage collected. You can manually close the client using the `.close()` method if desired, or with a context manager that closes when exiting.

```
from openai import OpenAI

with OpenAI() as client:
  # make requests here
  ...

# HTTP client is now closed
```

## Microsoft Azure OpenAI

To use this library with Azure OpenAI, use the `AzureOpenAI`
class instead of the `OpenAI` class.

The Azure API shape differs from the core API shape which means that the static types for responses / params
won't always be correct.

```
from openai import AzureOpenAI

# gets the API Key from environment variable AZURE_OPENAI_API_KEY
client = AzureOpenAI(
    # https://learn.microsoft.com/azure/ai-services/openai/reference#rest-api-versioning
    api_version="2023-07-01-preview",
    # https://learn.microsoft.com/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
    azure_endpoint="https://example-endpoint.openai.azure.com",
)

completion = client.chat.completions.create(
    model="deployment-name",  # e.g. gpt-35-instant
    messages=[\
        {\
            "role": "user",\
            "content": "How do I output all files in a directory using Python?",\
        },\
    ],
)
print(completion.to_json())
```

In addition to the options provided in the base `OpenAI` client, the following options are provided:

- `azure_endpoint` (or the `AZURE_OPENAI_ENDPOINT` environment variable)
- `azure_deployment`
- `api_version` (or the `OPENAI_API_VERSION` environment variable)
- `azure_ad_token` (or the `AZURE_OPENAI_AD_TOKEN` environment variable)
- `azure_ad_token_provider`

## Versioning

This package generally follows SemVer conventions, though certain backwards-incompatible changes may be released as minor versions:

1. Changes that only affect static types, without breaking runtime behavior.
2. Changes to library internals which are technically public but not intended or documented for external use.
3. Changes that we do not expect to impact the vast majority of users in practice.

We take backwards-compatibility seriously and work hard to ensure you can rely on a smooth upgrade experience.

We are keen for your feedback; please open an issue with questions, bugs, or suggestions.

### Determining the installed version

If you've upgraded to the latest version but aren't seeing any new features you were expecting then your python environment is likely still using an older version.

You can determine the version that is being used at runtime with:

```
import openai
print(openai.__version__)
```

## Requirements

Python 3.8 or higher.

## Contributing

See the contributing documentation.


The official Python library for the OpenAI API

### Original URL
https://github.com/openai/openai-python
</details>

---
<details>
<summary>Text generation and prompting - OpenAI API</summary>

# Text generation and prompting

Learn how to prompt a model to generate text.

With the OpenAI API, you can use a large language model to generate text from a prompt, as you might using ChatGPT. Models can generate almost any kind of text response—like code, mathematical equations, structured JSON data, or human-like prose.

Here's a simple example using the Responses API.

Generate text from a simple prompt

```javascript
import OpenAI from "openai";
const client = new OpenAI();

const response = await client.responses.create({
    model: "gpt-4.1",
    input: "Write a one-sentence bedtime story about a unicorn."
});

console.log(response.output_text);
```

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)
```

```bash
curl "https://api.openai.com/v1/responses" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d '{
        "model": "gpt-4.1",
        "input": "Write a one-sentence bedtime story about a unicorn."
    }'
```

An array of content generated by the model is in the `output` property of the response. In this simple example, we have just one output which looks like this:

```json
[
    {
        "id": "msg_67b73f697ba4819183a15cc17d011509",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "Under the soft glow of the moon, Luna the unicorn danced through fields of twinkling stardust, leaving trails of dreams for every child asleep.",
                "annotations": []
            }
        ]
    }
]
```

**The `output` array often has more than one item in it!** It can contain tool calls, data about reasoning tokens generated by reasoning models, and other items. It is not safe to assume that the model's text output is present at `output[0].content[0].text`.

Some of our official SDKs include an `output_text` property on model responses for convenience, which aggregates all text outputs from the model into a single string. This may be useful as a shortcut to access text output from the model.

In addition to plain text, you can also have the model return structured data in JSON format - this feature is called **Structured Outputs**.

## Choosing a model

A key choice to make when generating content through the API is which model you want to use - the `model` parameter of the code samples above. You can find a full listing of available models here. Here are a few factors to consider when choosing a model for text generation.

- **Reasoning models** generate an internal chain of thought to analyze the input prompt, and excel at understanding complex tasks and multi-step planning. They are also generally slower and more expensive to use than GPT models.
- **GPT models** are fast, cost-efficient, and highly intelligent, but benefit from more explicit instructions around how to accomplish tasks.
- **Large and small (mini or nano) models** offer trade-offs for speed, cost, and intelligence. Large models are more effective at understanding prompts and solving problems across domains, while small models are generally faster and cheaper to use.

When in doubt, `gpt-4.1` offers a solid combination of intelligence, speed, and cost effectiveness.

## Prompt engineering

**Prompt engineering** is the process of writing effective instructions for a model, such that it consistently generates content that meets your requirements.

Because the content generated from a model is non-deterministic, it is a combination of art and science to build a prompt that will generate content in the format you want. However, there are a number of techniques and best practices you can apply to consistently get good results from a model.

Some prompt engineering techniques will work with every model, like using message roles. But different model types (like reasoning versus GPT models) might need to be prompted differently to produce the best results. Even different snapshots of models within the same family could produce different results. So as you are building more complex applications, we strongly recommend that you:

- Pin your production applications to specific model snapshots (like `gpt-4.1-2025-04-14` for example) to ensure consistent behavior.
- Build evals that will measure the behavior of your prompts, so that you can monitor the performance of your prompts as you iterate on them, or when you change and upgrade model versions.

Now, let's examine some tools and techniques available to you to construct prompts.

## Message roles and instruction following

You can provide instructions to the model with differing levels of authority using the `instructions` API parameter or **message roles**.

The `instructions` parameter gives the model high-level instructions on how it should behave while generating a response, including tone, goals, and examples of correct responses. Any instructions provided this way will take priority over a prompt in the `input` parameter.

Generate text with instructions

```javascript
import OpenAI from "openai";
const client = new OpenAI();

const response = await client.responses.create({
    model: "gpt-4.1",
    instructions: "Talk like a pirate.",
    input: "Are semicolons optional in JavaScript?",
});

console.log(response.output_text);
```

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    instructions="Talk like a pirate.",
    input="Are semicolons optional in JavaScript?",
)

print(response.output_text)
```

```bash
curl "https://api.openai.com/v1/responses" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d '{
        "model": "gpt-4.1",
        "instructions": "Talk like a pirate.",
        "input": "Are semicolons optional in JavaScript?"
    }'
```

The example above is roughly equivalent to using the following input messages in the `input` array:

Generate text with messages using different roles

```javascript
import OpenAI from "openai";
const client = new OpenAI();

const response = await client.responses.create({
    model: "gpt-4.1",
    input: [
        {
            role: "developer",
            content: "Talk like a pirate."
        },
        {
            role: "user",
            content: "Are semicolons optional in JavaScript?",
        },
    ],
});

console.log(response.output_text);
```

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "developer",
            "content": "Talk like a pirate."
        },
        {
            "role": "user",
            "content": "Are semicolons optional in JavaScript?"
        }
    ]
)

print(response.output_text)
```

```bash
curl "https://api.openai.com/v1/responses" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d '{
        "model": "gpt-4.1",
        "input": [
            {
                "role": "developer",
                "content": "Talk like a pirate."
            },
            {
                "role": "user",
                "content": "Are semicolons optional in JavaScript?"
            }
        ]
    }'
```

Note that the `instructions` parameter only applies to the current response generation request. If you are managing conversation state with the `previous_response_id` parameter, the `instructions` used on previous turns will not be present in the context.

| `developer` | `user` | `assistant` |
| --- | --- | --- |
| `developer` messages are instructions provided by the application developer, prioritized ahead of `user` messages. | `user` messages are instructions provided by an end user, prioritized behind `developer` messages. | Messages generated by the model have the `assistant` role. |

A multi-turn conversation may consist of several messages of these types, along with other content types provided by both you and the model.

You could think about `developer` and `user` messages like a function and its arguments in a programming language.

- `developer` messages provide the system's rules and business logic, like a function definition.
- `user` messages provide inputs and configuration to which the `developer` message instructions are applied, like arguments to a function.

## Reusable prompts

In the OpenAI dashboard, you can develop reusable prompts that you can use in API requests, rather than specifying the content of prompts in code. This way, you can more easily build and evaluate your prompts, and deploy improved versions of your prompts without changing your integration code.

Here's how it works:

1. **Create a reusable prompt** in the dashboard with placeholders like `{{customer_name}}`.
2. **Use the prompt** in your API request with the `prompt` parameter. The prompt parameter object has three properties you can configure:

   - `id` — Unique identifier of your prompt, found in the dashboard
   - `version` — A specific version of your prompt (defaults to the "current" version as specified in the dashboard)
   - `variables` — A map of values to substitute in for variables in your prompt. The substitution values can either be strings, or other Response input message types like `input_image` or `input_file`.

Generate text with a prompt template

```javascript
import OpenAI from "openai";
const client = new OpenAI();

const response = await client.responses.create({
    model: "gpt-4.1",
    prompt: {
        id: "pmpt_abc123",
        version: "2",
        variables: {
            customer_name: "Jane Doe",
            product: "40oz juice box"
        }
    }
});

console.log(response.output_text);
```

```python
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    prompt={
        "id": "pmpt_abc123",
        "version": "2",
        "variables": {
            "customer_name": "Jane Doe",
            "product": "40oz juice box"
        }
    }
)

print(response.output_text)
```

```bash
curl https://api.openai.com/v1/responses \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4.1",
    "prompt": {
      "id": "pmpt_abc123",
      "version": "2",
      "variables": {
        "customer_name": "Jane Doe",
        "product": "40oz juice box"
      }
    }
  }'
```

Variables with file input

Prompt template with file input variable

```javascript
import fs from "fs";
import OpenAI from "openai";
const client = new OpenAI();

// Upload a PDF we will reference in the prompt variables
const file = await client.files.create({
    file: fs.createReadStream("draconomicon.pdf"),
    purpose: "user_data",
});

const response = await client.responses.create({
    model: "gpt-4.1",
    prompt: {
        id: "pmpt_abc123",
        variables: {
            topic: "Dragons",
            reference_pdf: {
                type: "input_file",
                file_id: file.id,
            },
        },
    },
});

console.log(response.output_text);
```

```python
import openai, pathlib

client = openai.OpenAI()

# Upload a PDF we will reference in the variables
file = client.files.create(
    file=open("draconomicon.pdf", "rb"),
    purpose="user_data",
)

response = client.responses.create(
    model="gpt-4.1",
    prompt={
        "id": "pmpt_abc123",
        "variables": {
            "topic": "Dragons",
            "reference_pdf": {
                "type": "input_file",
                "file_id": file.id,
            },
        },
    },
)

print(response.output_text)
```

```bash
# Assume you have already uploaded the PDF and obtained FILE_ID
curl https://api.openai.com/v1/responses   -H "Authorization: Bearer $OPENAI_API_KEY"   -H "Content-Type: application/json"   -d '{
    "model": "gpt-4.1",
    "prompt": {
      "id": "pmpt_abc123",
      "variables": {
        "topic": "Dragons",
        "reference_pdf": {
          "type": "input_file",
          "file_id": "file-abc123"
        }
      }
    }
  }'
```

## Message formatting with Markdown and XML

When writing `developer` and `user` messages, you can help the model understand logical boundaries of your prompt and context data using a combination of Markdown formatting and XML tags.

Markdown headers and lists can be helpful to mark distinct sections of a prompt, and to communicate hierarchy to the model. They can also potentially make your prompts more readable during development. XML tags can help delineate where one piece of content (like a supporting document used for reference) begins and ends. XML attributes can also be used to define metadata about content in the prompt that can be referenced by your instructions.

In general, a developer message will contain the following sections, usually in this order (though the exact optimal content and order may vary by which model you are using):

- **Identity:** Describe the purpose, communication style, and high-level goals of the assistant.
- **Instructions:** Provide guidance to the model on how to generate the response you want. What rules should it follow? What should the model do, and what should the model never do? This section could contain many subsections as relevant for your use case, like how the model should call custom functions.
- **Examples:** Provide examples of possible inputs, along with the desired output from the model.
- **Context:** Give the model any additional information it might need to generate a response, like private/proprietary data outside its training data, or any other data you know will be particularly relevant. This content is usually best positioned near the end of your prompt, as you may include different context for different generation requests.

Below is an example of using Markdown and XML tags to construct a `developer` message with distinct sections and supporting examples.

A developer message for code generation

```text
# Identity

You are coding assistant that helps enforce the use of snake case
variables in JavaScript code, and writing code that will run in
Internet Explorer version 6.

# Instructions

* When defining variables, use snake case names (e.g. my_variable)
  instead of camel case names (e.g. myVariable).
* To support old browsers, declare variables using the older
  "var" keyword.
* Do not give responses with Markdown formatting, just return
  the code as requested.

# Examples

<user_query>
How do I declare a string variable for a first name?
</user_query>

<assistant_response>
var first_name = "Anna";
</assistant_response>
```

Send a prompt to generate code through the API

```javascript
import fs from "fs/promises";
import OpenAI from "openai";
const client = new OpenAI();

const instructions = await fs.readFile("prompt.txt", "utf-8");

const response = await client.responses.create({
    model: "gpt-4.1",
    instructions,
    input: "How would I declare a variable for a last name?",
});

console.log(response.output_text);
```

```python
from openai import OpenAI
client = OpenAI()

with open("prompt.txt", "r", encoding="utf-8") as f:
    instructions = f.read()

response = client.responses.create(
    model="gpt-4.1",
    instructions=instructions,
    input="How would I declare a variable for a last name?",
)

print(response.output_text)
```

```bash
curl https://api.openai.com/v1/responses \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4.1",
    "instructions": "'"$(< prompt.txt)"'",
    "input": "How would I declare a variable for a last name?"
  }'
```

#### Save on cost and latency with prompt caching

When constructing a message, you should try and keep content that you expect to use over and over in your API requests at the beginning of your prompt, **and** among the first API parameters you pass in the JSON request body to Chat Completions or Responses. This enables you to maximize cost and latency savings from prompt caching.

## Few-shot learning

Few-shot learning lets you steer a large language model toward a new task by including a handful of input/output examples in the prompt, rather than fine-tuning the model. The model implicitly "picks up" the pattern from those examples and applies it to a prompt. When providing examples, try to show a diverse range of possible inputs with the desired outputs.

Typically, you will provide examples as part of a `developer` message in your API request. Here's an example `developer` message containing examples that show a model how to classify positive or negative customer service reviews.

```text
# Identity

You are a helpful assistant that labels short product reviews as
Positive, Negative, or Neutral.

# Instructions

* Only output a single word in your response with no additional formatting
  or commentary.
* Your response should only be one of the words "Positive", "Negative", or
  "Neutral" depending on the sentiment of the product review you are given.

# Examples

<product_review id="example-1">
I absolutely love this headphones — sound quality is amazing!
</product_review>

<assistant_response id="example-1">
Positive
</assistant_response>

<product_review id="example-2">
Battery life is okay, but the ear pads feel cheap.
</product_review>

<assistant_response id="example-2">
Neutral
</assistant_response>

<product_review id="example-3">
Terrible customer service, I'll never buy from them again.
</product_review>

<assistant_response id="example-3">
Negative
</assistant_response>
```

## Include relevant context information

It is often useful to include additional context information the model can use to generate a response within the prompt you give the model. There are a few common reasons why you might do this:

- To give the model access to proprietary data, or any other data outside the data set the model was trained on.
- To constrain the model's response to a specific set of resources that you have determined will be most beneficial.

The technique of adding additional relevant context to the model generation request is sometimes called **retrieval-augmented generation (RAG)**. You can add additional context to the prompt in many different ways, from querying a vector database and including the text you get back into a prompt, or by using OpenAI's built-in file search tool to generate content based on uploaded documents.

#### Planning for the context window

Models can only handle so much data within the context they consider during a generation request. This memory limit is called a **context window**, which is defined in terms of tokens (chunks of data you pass in, from text to images).

Models have different context window sizes from the low 100k range up to one million tokens for newer GPT-4.1 models. Refer to the model docs for specific context window sizes per model.

## Prompting GPT-4.1 models

GPT models like `gpt-4.1` benefit from precise instructions that explicitly provide the logic and data required to complete the task in the prompt. GPT-4.1 in particular is highly steerable and responsive to well-specified prompts. To get the most out of GPT-4.1, refer to the prompting guide in the cookbook.

#### GPT-4.1 prompting best practices

While the cookbook has the best and most comprehensive guidance for prompting this model, here are a few best practices to keep in mind.

Building agentic workflows

### System Prompt Reminders

In order to best utilize the agentic capabilities of GPT-4.1, we recommend including three key types of reminders in all agent prompts for persistence, tool calling, and planning. As a whole, we find that these three instructions transform the model's behavior from chatbot-like into a much more "eager" agent, driving the interaction forward autonomously and independently. Here are a few examples:

```text
## PERSISTENCE
You are an agent - please keep going until the user's query is completely
resolved, before ending your turn and yielding back to the user. Only
terminate your turn when you are sure that the problem is solved.

## TOOL CALLING
If you are not sure about file content or codebase structure pertaining to
the user's request, use your tools to read files and gather the relevant
information: do NOT guess or make up an answer.

## PLANNING
You MUST plan extensively before each function call, and reflect
extensively on the outcomes of the previous function calls. DO NOT do this
entire process by making function calls only, as this can impair your
ability to solve the problem and think insightfully.
```

#### Tool Calls

Compared to previous models, GPT-4.1 has undergone more training on effectively utilizing tools passed as arguments in an OpenAI API request. We encourage developers to exclusively use the tools field of API requests to pass tools for best understanding and performance, rather than manually injecting tool descriptions into the system prompt and writing a separate parser for tool calls, as some have reported doing in the past.

#### Diff Generation

Correct diffs are critical for coding applications, so we've significantly improved performance at this task for GPT-4.1. In our cookbook, we open-source a recommended diff format on which GPT-4.1 has been extensively trained. That said, the model should generalize to any well-specified format.

Using long context

GPT-4.1 has a performant 1M token input context window, and will be useful for a variety of long context tasks, including structured document parsing, re-ranking, selecting relevant information while ignoring irrelevant context, and performing multi-hop reasoning using context.

#### Optimal Context Size

We show perfect performance at needle-in-a-haystack evals up to our full context size, and we've observed very strong performance at complex tasks with a mix of relevant and irrelevant code and documents in the range of hundreds of thousands of tokens.

#### Delimiters

We tested a variety of delimiters for separating context provided to the model against our long context evals. Briefly, XML and the format demonstrated by Lee et al. (ref) tend to perform well, while JSON performed worse for this task.

#### Prompt Organization

Especially in long context usage, placement of instructions and context can substantially impact performance. In our experiments, we found that it was optimal to put critical instructions, including the user query, at both the top and the bottom of the prompt; this elicited marginally better performance from the model than putting them only at the top, and much better performance than only at the bottom.

Prompting for chain of thought

As mentioned above, GPT-4.1 isn't a reasoning model, but prompting the model to think step by step (called "chain of thought") can be an effective way for a model to break down problems into more manageable pieces. The model has been trained to perform well at agentic reasoning and real-world problem solving, so it shouldn't require much prompting to do well.

We recommend starting with this basic chain-of-thought instruction at the end of your prompt:

```text
First, think carefully step by step about what documents are needed to answer the query. Then, print out the TITLE and ID of each document. Then, format the IDs into a list.
```

From there, you should improve your CoT prompt by auditing failures in your particular examples and evals, and addressing systematic planning and reasoning errors with more explicit instructions.

Instruction following

GPT-4.1 exhibits outstanding instruction-following performance, which developers can leverage to precisely shape and control the outputs for their particular use cases. However, since the model follows instructions more literally than its predecessors, may need to provide more explicit specification around what to do or not do, and existing prompts optimized for other models may not immediately work with this model.

#### Recommended Workflow

Here is our recommended workflow for developing and debugging instructions in prompts:

- Start with an overall "Response Rules" or "Instructions" section with high-level guidance and bullet points.
- If you'd like to change a more specific behavior, add a section containing more details for that category, like `## Sample Phrases`.
- If there are specific steps you'd like the model to follow in its workflow, add an ordered list and instruct the model to follow these steps.
- If behavior still isn't working as expected, check for conflicting, underspecified, or incorrect instructions and examples. If there are conflicting instructions, GPT-4.1 tends to follow the one closer to the end of the prompt.
- Add examples that demonstrate desired behavior; ensure that any important behavior demonstrated in your examples are also cited in your rules.
- It's generally not necessary to use all-caps or other incentives like bribes or tips, but developers can experiment with this for extra emphasis if so desired.

#### Common Failure Modes

These failure modes are not unique to GPT-4.1, but we share them here for general awareness and ease of debugging.

- Instructing a model to always follow a specific behavior can occasionally induce adverse effects. For instance, if told "you must call a tool before responding to the user," models may hallucinate tool inputs or call the tool with null values if they do not have enough information. Adding "if you don't have enough information to call the tool, ask the user for the information you need" should mitigate this.
- When provided sample phrases, models can use those quotes verbatim and start to sound repetitive to users. Ensure you instruct the model to vary them as necessary.
- Without specific instructions, some models can be eager to provide additional prose to explain their decisions, or output more formatting in responses than may be desired. Provide instructions and potentially examples to help mitigate.

## Prompting reasoning models

There are some differences to consider when prompting a reasoning model versus prompting a GPT model. Generally speaking, reasoning models will provide better results on tasks with only high-level guidance. This differs from GPT models, which benefit from very precise instructions.

You could think about the difference between reasoning and GPT models like this.

- A reasoning model is like a senior co-worker. You can give them a goal to achieve and trust them to work out the details.
- A GPT model is like a junior coworker. They'll perform best with explicit instructions to create a specific output.

## Next steps

Now that you know the basics of text inputs and outputs, you might want to check out one of these resources next.

Build a prompt in the Playground

Use the Playground to develop and iterate on prompts.

Generate JSON data with Structured Outputs

Ensure JSON data emitted from a model conforms to a JSON schema.

Full API reference

Check out all the options for text generation in the API reference.

### Original URL
https://platform.openai.com/docs/guides/text-generation/chat-completions-api
</details>

---
<details>
<summary>Prompting Techniques | Prompt Engineering Guide </summary>

Prompt Engineering helps to effectively design and improve prompts to get better results on different tasks with LLMs.

While the previous basic examples were fun, in this section we cover more advanced prompting engineering techniques that allow us to achieve more complex tasks and improve reliability and performance of LLMs.

### Original URL
https://www.promptingguide.ai/techniques
</details>

---

## Code Sources

---
<details>
<summary>Gitingest: https://github.com/openai/openai-python/blob/main/README.md</summary>

# Repository Analysis

## Summary
Repository: openai/openai-python
File: README.md
Lines: 740

Estimated tokens: 5.5k

## File Structure
```
Directory structure:
└── README.md
```

## Content
================================================
File: README.md
================================================
# OpenAI Python API library

The OpenAI Python library provides convenient access to the OpenAI REST API from any Python 3.8+
application. The library includes type definitions for all request params and response fields,
and offers both synchronous and asynchronous clients powered by [httpx](https://github.com/encode/httpx).

It is generated from our [OpenAPI specification](https://github.com/openai/openai-openapi) with [Stainless](https://stainlessapi.com/).

## Documentation

The REST API documentation can be found on [platform.openai.com](https://platform.openai.com/docs/api-reference). The full API of this library can be found in [api.md](api.md).

## Installation

```sh
# install from PyPI
pip install openai
```

## Usage

The full API of this library can be found in [api.md](api.md).

The primary API for interacting with OpenAI models is the [Responses API](https://platform.openai.com/docs/api-reference/responses). You can generate text from the model with the code below.

```python
import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

response = client.responses.create(
    model="gpt-4o",
    instructions="You are a coding assistant that talks like a pirate.",
    input="How do I check if a Python object is an instance of a class?",
)

print(response.output_text)
```

The previous standard (supported indefinitely) for generating text is the [Chat Completions API](https://platform.openai.com/docs/api-reference/chat). You can use that API to generate text from the model with the code below.

```python
from openai import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "developer", "content": "Talk like a pirate."},
        {
            "role": "user",
            "content": "How do I check if a Python object is an instance of a class?",
        },
    ],
)

print(completion.choices[0].message.content)
```

While you can provide an `api_key` keyword argument,
we recommend using [python-dotenv](https://pypi.org/project/python-dotenv/)
to add `OPENAI_API_KEY="My API Key"` to your `.env` file
so that your API key is not stored in source control.
[Get an API key here](https://platform.openai.com/settings/organization/api-keys).

### Vision

With an image URL:

```python
prompt = "What is in this image?"
img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/2023_06_08_Raccoon1.jpg/1599px-2023_06_08_Raccoon1.jpg"

response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"{img_url}"},
            ],
        }
    ],
)
```

With the image as a base64 encoded string:

```python
import base64
from openai import OpenAI

client = OpenAI()

prompt = "What is in this image?"
with open("path/to/image.png", "rb") as image_file:
    b64_image = base64.b64encode(image_file.read()).decode("utf-8")

response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/png;base64,{b64_image}"},
            ],
        }
    ],
)
```

## Async usage

Simply import `AsyncOpenAI` instead of `OpenAI` and use `await` with each API call:

```python
import os
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


async def main() -> None:
    response = await client.responses.create(
        model="gpt-4o", input="Explain disestablishmentarianism to a smart five year old."
    )
    print(response.output_text)


asyncio.run(main())
```

Functionality between the synchronous and asynchronous clients is otherwise identical.

## Streaming responses

We provide support for streaming responses using Server Side Events (SSE).

```python
from openai import OpenAI

client = OpenAI()

stream = client.responses.create(
    model="gpt-4o",
    input="Write a one-sentence bedtime story about a unicorn.",
    stream=True,
)

for event in stream:
    print(event)
```

The async client uses the exact same interface.

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()


async def main():
    stream = await client.responses.create(
        model="gpt-4o",
        input="Write a one-sentence bedtime story about a unicorn.",
        stream=True,
    )

    async for event in stream:
        print(event)


asyncio.run(main())
```

## Realtime API beta

The Realtime API enables you to build low-latency, multi-modal conversational experiences. It currently supports text and audio as both input and output, as well as [function calling](https://platform.openai.com/docs/guides/function-calling) through a WebSocket connection.

Under the hood the SDK uses the [`websockets`](https://websockets.readthedocs.io/en/stable/) library to manage connections.

The Realtime API works through a combination of client-sent events and server-sent events. Clients can send events to do things like update session configuration or send text and audio inputs. Server events confirm when audio responses have completed, or when a text response from the model has been received. A full event reference can be found [here](https://platform.openai.com/docs/api-reference/realtime-client-events) and a guide can be found [here](https://platform.openai.com/docs/guides/realtime).

Basic text based example:

```py
import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI()

    async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as connection:
        await connection.session.update(session={'modalities': ['text']})

        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Say hello!"}],
            }
        )
        await connection.response.create()

        async for event in connection:
            if event.type == 'response.text.delta':
                print(event.delta, flush=True, end="")

            elif event.type == 'response.text.done':
                print()

            elif event.type == "response.done":
                break

asyncio.run(main())
```

However the real magic of the Realtime API is handling audio inputs / outputs, see this example [TUI script](https://github.com/openai/openai-python/blob/main/examples/realtime/push_to_talk_app.py) for a fully fledged example.

### Realtime error handling

Whenever an error occurs, the Realtime API will send an [`error` event](https://platform.openai.com/docs/guides/realtime-model-capabilities#error-handling) and the connection will stay open and remain usable. This means you need to handle it yourself, as _no errors are raised directly_ by the SDK when an `error` event comes in.

```py
client = AsyncOpenAI()

async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as connection:
    ...
    async for event in connection:
        if event.type == 'error':
            print(event.error.type)
            print(event.error.code)
            print(event.error.event_id)
            print(event.error.message)
```

## Using types

Nested request parameters are [TypedDicts](https://docs.python.org/3/library/typing.html#typing.TypedDict). Responses are [Pydantic models](https://docs.pydantic.dev) which also provide helper methods for things like:

- Serializing back into JSON, `model.to_json()`
- Converting to a dictionary, `model.to_dict()`

Typed requests and responses provide autocomplete and documentation within your editor. If you would like to see type errors in VS Code to help catch bugs earlier, set `python.analysis.typeCheckingMode` to `basic`.

## Pagination

List methods in the OpenAI API are paginated.

This library provides auto-paginating iterators with each list response, so you do not have to request successive pages manually:

```python
from openai import OpenAI

client = OpenAI()

all_jobs = []
# Automatically fetches more pages as needed.
for job in client.fine_tuning.jobs.list(
    limit=20,
):
    # Do something with job here
    all_jobs.append(job)
print(all_jobs)
```

Or, asynchronously:

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()


async def main() -> None:
    all_jobs = []
    # Iterate through items across all pages, issuing requests as needed.
    async for job in client.fine_tuning.jobs.list(
        limit=20,
    ):
        all_jobs.append(job)
    print(all_jobs)


asyncio.run(main())
```

Alternatively, you can use the `.has_next_page()`, `.next_page_info()`, or `.get_next_page()` methods for more granular control working with pages:

```python
first_page = await client.fine_tuning.jobs.list(
    limit=20,
)
if first_page.has_next_page():
    print(f"will fetch next page using these details: {first_page.next_page_info()}")
    next_page = await first_page.get_next_page()
    print(f"number of items we just fetched: {len(next_page.data)}")

# Remove `await` for non-async usage.
```

Or just work directly with the returned data:

```python
first_page = await client.fine_tuning.jobs.list(
    limit=20,
)

print(f"next page cursor: {first_page.after}")  # => "next page cursor: ..."
for job in first_page.data:
    print(job.id)

# Remove `await` for non-async usage.
```

## Nested params

Nested parameters are dictionaries, typed using `TypedDict`, for example:

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.responses.create(
    input=[
        {
            "role": "user",
            "content": "How much ?",
        }
    ],
    model="gpt-4o",
    response_format={"type": "json_object"},
)
```

## File uploads

Request parameters that correspond to file uploads can be passed as `bytes`, or a [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike) instance or a tuple of `(filename, contents, media type)`.

```python
from pathlib import Path
from openai import OpenAI

client = OpenAI()

client.files.create(
    file=Path("input.jsonl"),
    purpose="fine-tune",
)
```

The async client uses the exact same interface. If you pass a [`PathLike`](https://docs.python.org/3/library/os.html#os.PathLike) instance, the file contents will be read asynchronously automatically.

## Handling errors

When the library is unable to connect to the API (for example, due to network connection problems or a timeout), a subclass of `openai.APIConnectionError` is raised.

When the API returns a non-success status code (that is, 4xx or 5xx
response), a subclass of `openai.APIStatusError` is raised, containing `status_code` and `response` properties.

All errors inherit from `openai.APIError`.

```python
import openai
from openai import OpenAI

client = OpenAI()

try:
    client.fine_tuning.jobs.create(
        model="gpt-4o",
        training_file="file-abc123",
    )
except openai.APIConnectionError as e:
    print("The server could not be reached")
    print(e.__cause__)  # an underlying Exception, likely raised within httpx.
except openai.RateLimitError as e:
    print("A 429 status code was received; we should back off a bit.")
except openai.APIStatusError as e:
    print("Another non-200-range status code was received")
    print(e.status_code)
    print(e.response)
```

Error codes are as follows:

| Status Code | Error Type                 |
| ----------- | -------------------------- |
| 400         | `BadRequestError`          |
| 401         | `AuthenticationError`      |
| 403         | `PermissionDeniedError`    |
| 404         | `NotFoundError`            |
| 422         | `UnprocessableEntityError` |
| 429         | `RateLimitError`           |
| >=500       | `InternalServerError`      |
| N/A         | `APIConnectionError`       |

## Request IDs

> For more information on debugging requests, see [these docs](https://platform.openai.com/docs/api-reference/debugging-requests)

All object responses in the SDK provide a `_request_id` property which is added from the `x-request-id` response header so that you can quickly log failing requests and report them back to OpenAI.

```python
response = await client.responses.create(
    model="gpt-4o-mini",
    input="Say 'this is a test'.",
)
print(response._request_id)  # req_123
```

Note that unlike other properties that use an `_` prefix, the `_request_id` property
_is_ public. Unless documented otherwise, _all_ other `_` prefix properties,
methods and modules are _private_.

> [!IMPORTANT]  
> If you need to access request IDs for failed requests you must catch the `APIStatusError` exception

```python
import openai

try:
    completion = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Say this is a test"}], model="gpt-4"
    )
except openai.APIStatusError as exc:
    print(exc.request_id)  # req_123
    raise exc
```

## Retries

Certain errors are automatically retried 2 times by default, with a short exponential backoff.
Connection errors (for example, due to a network connectivity problem), 408 Request Timeout, 409 Conflict,
429 Rate Limit, and >=500 Internal errors are all retried by default.

You can use the `max_retries` option to configure or disable retry settings:

```python
from openai import OpenAI

# Configure the default for all requests:
client = OpenAI(
    # default is 2
    max_retries=0,
)

# Or, configure per-request:
client.with_options(max_retries=5).chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "How can I get the name of the current day in JavaScript?",
        }
    ],
    model="gpt-4o",
)
```

## Timeouts

By default requests time out after 10 minutes. You can configure this with a `timeout` option,
which accepts a float or an [`httpx.Timeout`](https://www.python-httpx.org/advanced/timeouts/#fine-tuning-the-configuration) object:

```python
from openai import OpenAI

# Configure the default for all requests:
client = OpenAI(
    # 20 seconds (default is 10 minutes)
    timeout=20.0,
)

# More granular control:
client = OpenAI(
    timeout=httpx.Timeout(60.0, read=5.0, write=10.0, connect=2.0),
)

# Override per-request:
client.with_options(timeout=5.0).chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "How can I list all files in a directory using Python?",
        }
    ],
    model="gpt-4o",
)
```

On timeout, an `APITimeoutError` is thrown.

Note that requests that time out are [retried twice by default](#retries).

## Advanced

### Logging

We use the standard library [`logging`](https://docs.python.org/3/library/logging.html) module.

You can enable logging by setting the environment variable `OPENAI_LOG` to `info`.

```shell
$ export OPENAI_LOG=info
```

Or to `debug` for more verbose logging.

### How to tell whether `None` means `null` or missing

In an API response, a field may be explicitly `null`, or missing entirely; in either case, its value is `None` in this library. You can differentiate the two cases with `.model_fields_set`:

```py
if response.my_field is None:
  if 'my_field' not in response.model_fields_set:
    print('Got json like {}, without a "my_field" key present at all.')
  else:
    print('Got json like {"my_field": null}.')
```

### Accessing raw response data (e.g. headers)

The "raw" Response object can be accessed by prefixing `.with_raw_response.` to any HTTP method call, e.g.,

```py
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.with_raw_response.create(
    messages=[{
        "role": "user",
        "content": "Say this is a test",
    }],
    model="gpt-4o",
)
print(response.headers.get('X-My-Header'))

completion = response.parse()  # get the object that `chat.completions.create()` would have returned
print(completion)
```

These methods return a [`LegacyAPIResponse`](https://github.com/openai/openai-python/tree/main/src/openai/_legacy_response.py) object. This is a legacy class as we're changing it slightly in the next major version.

For the sync client this will mostly be the same with the exception
of `content` & `text` will be methods instead of properties. In the
async client, all methods will be async.

A migration script will be provided & the migration in general should
be smooth.

#### `.with_streaming_response`

The above interface eagerly reads the full response body when you make the request, which may not always be what you want.

To stream the response body, use `.with_streaming_response` instead, which requires a context manager and only reads the response body once you call `.read()`, `.text()`, `.json()`, `.iter_bytes()`, `.iter_text()`, `.iter_lines()` or `.parse()`. In the async client, these are async methods.

As such, `.with_streaming_response` methods return a different [`APIResponse`](https://github.com/openai/openai-python/tree/main/src/openai/_response.py) object, and the async client returns an [`AsyncAPIResponse`](https://github.com/openai/openai-python/tree/main/src/openai/_response.py) object.

```python
with client.chat.completions.with_streaming_response.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-4o",
) as response:
    print(response.headers.get("X-My-Header"))

    for line in response.iter_lines():
        print(line)
```

The context manager is required so that the response will reliably be closed.

### Making custom/undocumented requests

This library is typed for convenient access to the documented API.

If you need to access undocumented endpoints, params, or response properties, the library can still be used.

#### Undocumented endpoints

To make requests to undocumented endpoints, you can make requests using `client.get`, `client.post`, and other
http verbs. Options on the client will be respected (such as retries) when making this request.

```py
import httpx

response = client.post(
    "/foo",
    cast_to=httpx.Response,
    body={"my_param": True},
)

print(response.headers.get("x-foo"))
```

#### Undocumented request params

If you want to explicitly send an extra param, you can do so with the `extra_query`, `extra_body`, and `extra_headers` request
options.

#### Undocumented response properties

To access undocumented response properties, you can access the extra fields like `response.unknown_prop`. You
can also get all the extra fields on the Pydantic model as a dict with
[`response.model_extra`](https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_extra).

### Configuring the HTTP client

You can directly override the [httpx client](https://www.python-httpx.org/api/#client) to customize it for your use case, including:

- Support for [proxies](https://www.python-httpx.org/advanced/proxies/)
- Custom [transports](https://www.python-httpx.org/advanced/transports/)
- Additional [advanced](https://www.python-httpx.org/advanced/clients/) functionality

```python
import httpx
from openai import OpenAI, DefaultHttpxClient

client = OpenAI(
    # Or use the `OPENAI_BASE_URL` env var
    base_url="http://my.test.server.example.com:8083/v1",
    http_client=DefaultHttpxClient(
        proxy="http://my.test.proxy.example.com",
        transport=httpx.HTTPTransport(local_address="0.0.0.0"),
    ),
)
```

You can also customize the client on a per-request basis by using `with_options()`:

```python
client.with_options(http_client=DefaultHttpxClient(...))
```

### Managing HTTP resources

By default the library closes underlying HTTP connections whenever the client is [garbage collected](https://docs.python.org/3/reference/datamodel.html#object.__del__). You can manually close the client using the `.close()` method if desired, or with a context manager that closes when exiting.

```py
from openai import OpenAI

with OpenAI() as client:
  # make requests here
  ...

# HTTP client is now closed
```

## Microsoft Azure OpenAI

To use this library with [Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/overview), use the `AzureOpenAI`
class instead of the `OpenAI` class.

> [!IMPORTANT]
> The Azure API shape differs from the core API shape which means that the static types for responses / params
> won't always be correct.

```py
from openai import AzureOpenAI

# gets the API Key from environment variable AZURE_OPENAI_API_KEY
client = AzureOpenAI(
    # https://learn.microsoft.com/azure/ai-services/openai/reference#rest-api-versioning
    api_version="2023-07-01-preview",
    # https://learn.microsoft.com/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
    azure_endpoint="https://example-endpoint.openai.azure.com",
)

completion = client.chat.completions.create(
    model="deployment-name",  # e.g. gpt-35-instant
    messages=[
        {
            "role": "user",
            "content": "How do I output all files in a directory using Python?",
        },
    ],
)
print(completion.to_json())
```

In addition to the options provided in the base `OpenAI` client, the following options are provided:

- `azure_endpoint` (or the `AZURE_OPENAI_ENDPOINT` environment variable)
- `azure_deployment`
- `api_version` (or the `OPENAI_API_VERSION` environment variable)
- `azure_ad_token` (or the `AZURE_OPENAI_AD_TOKEN` environment variable)
- `azure_ad_token_provider`

An example of using the client with Microsoft Entra ID (formerly known as Azure Active Directory) can be found [here](https://github.com/openai/openai-python/blob/main/examples/azure_ad.py).

## Versioning

This package generally follows [SemVer](https://semver.org/spec/v2.0.0.html) conventions, though certain backwards-incompatible changes may be released as minor versions:

1. Changes that only affect static types, without breaking runtime behavior.
2. Changes to library internals which are technically public but not intended or documented for external use. _(Please open a GitHub issue to let us know if you are relying on such internals.)_
3. Changes that we do not expect to impact the vast majority of users in practice.

We take backwards-compatibility seriously and work hard to ensure you can rely on a smooth upgrade experience.

We are keen for your feedback; please open an [issue](https://www.github.com/openai/openai-python/issues) with questions, bugs, or suggestions.

### Determining the installed version

If you've upgraded to the latest version but aren't seeing any new features you were expecting then your python environment is likely still using an older version.

You can determine the version that is being used at runtime with:

```py
import openai
print(openai.__version__)
```

## Requirements

Python 3.8 or higher.

## Contributing

See [the contributing documentation](./CONTRIBUTING.md).

### Original URL
https://github.com/openai/openai-python/blob/main/README.md
</details>

---

## Additional Sources Scraped

---
<details>
<summary>Prompt Chaining | Prompt Engineering Guide </summary>

# Prompt Chaining

## Introduction to Prompt Chaining

To improve the reliability and performance of LLMs, one of the important prompt engineering techniques is to break tasks into its subtasks. Once those subtasks have been identified, the LLM is prompted with a subtask and then its response is used as input to another prompt. This is what's referred to as prompt chaining, where a task is split into subtasks with the idea to create a chain of prompt operations.

Prompt chaining is useful to accomplish complex tasks which an LLM might struggle to address if prompted with a very detailed prompt. In prompt chaining, chain prompts perform transformations or additional processes on the generated responses before reaching a final desired state.

Besides achieving better performance, prompt chaining helps to boost the transparency of your LLM application, increases controllability, and reliability. This means that you can debug problems with model responses much more easily and analyze and improve performance in the different stages that need improvement.

Prompt chaining is particularly useful when building LLM-powered conversational assistants and improving the personalization and user experience of your applications.

## Use Cases for Prompt Chaining

### Prompt Chaining for Document QA

Prompt chaining can be used in different scenarios that could involve several operations or transformations. For instance, one common use case of LLMs involves answering questions about a large text document. It helps if you design two different prompts where the first prompt is responsible for extracting relevant quotes to answer a question and a second prompt takes as input the quotes and original document to answer a given question. In other words, you will be creating two different prompts to perform the task of answering a question given in a document.

The first prompt below extracts the relevant quotes from the document given the question. Note that for simplicity, we have added a placeholder for the document `{{document}}`. To test the prompt you can copy and paste an article from Wikipedia such as this page for prompt engineering (opens in a new tab) @https://en.wikipedia.org/wiki/Prompt_engineering. Due to larger context used for this task, we are using the `gpt-4-1106-preview` model from OpenAI. You can use the prompt with other long-context LLMs like Claude.

Prompt 1:

```
You are a helpful assistant. Your task is to help answer a question given in a document. The first step is to extract quotes relevant to the question from the document, delimited by ####. Please output the list of quotes using <quotes></quotes>. Respond with "No relevant quotes found!" if no relevant quotes were found.

####
{{document}}
####
```

Here is a screenshot of the entire prompt including the question which is passed using the `user` role.

Output of Prompt 1:

```
<quotes>
- Chain-of-thought (CoT) prompting[27]
- Generated knowledge prompting[37]
- Least-to-most prompting[38]
- Self-consistency decoding[39]
- Complexity-based prompting[41]
- Self-refine[42]
- Tree-of-thought prompting[43]
- Maieutic prompting[45]
- Directional-stimulus prompting[46]
- Textual inversion and embeddings[59]
- Using gradient descent to search for prompts[61][62][63][64]
- Prompt injection[65][66][67]
</quotes>
```

The quotes that were returned in the first prompt can now be used as input to the second prompt below. Note that you can clean up the quotes a bit more, i.e., remove the citations. Those citations could be removed or utilized as part of another prompt in the chain but you can ignore this for now. The second prompt then takes the relevant quotes extracted by prompt 1 and prepares a helpful response to the question given in the document and those extracted quotes. The second prompt can be the following:

Prompt 2:

```
Given a set of relevant quotes (delimited by <quotes></quotes>) extracted from a document and the original document (delimited by ####), please compose an answer to the question. Ensure that the answer is accurate, has a friendly tone, and sounds helpful.

####
{{document}}
####

<quotes>
- Chain-of-thought (CoT) prompting[27]
- Generated knowledge prompting[37]
- Least-to-most prompting[38]
- Self-consistency decoding[39]
- Complexity-based prompting[41]
- Self-refine[42]
- Tree-of-thought prompting[43]
- Maieutic prompting[45]
- Directional-stimulus prompting[46]
- Textual inversion and embeddings[59]
- Using gradient descent to search for prompts[61][62][63][64]
- Prompt injection[65][66][67]
</quotes>
```

Output of Prompt 2:

```
The prompting techniques mentioned in the document include:

1. Chain-of-thought (CoT) prompting[27]
2. Generated knowledge prompting[37]
3. Least-to-most prompting[38]
4. Self-consistency decoding[39]
5. Complexity-based prompting[41]
6. Self-refine[42]
7. Tree-of-thought prompting[43]
8. Maieutic prompting[45]
9. Directional-stimulus prompting[46]
10. Textual inversion and embeddings[59]
11. Using gradient descent to search for prompts[61][62][63][64]
12. Prompt injection[65][66][67]

Each of these techniques employs unique strategies to enhance or specify the interactions with large language models to produce the desired outcomes.
```

As you can see, simplifying and creating prompt chains is a useful prompting approach where the responses need to undergo several operations or transformations. As an exercise, feel free to design a prompt that removes the citations (e.g., \[27\]) from the response before sending this as a final response to the user of your application.

You can also find more examples of prompt chaining in this documentation (opens in a new tab) @https://docs.anthropic.com/claude/docs/prompt-chaining that leverages the Claude LLM. Our example is inspired and adapted from their examples.

### Original URL
https://www.promptingguide.ai/techniques/prompt_chaining
</details>

---
<details>
<summary>Claude 4 prompt engineering best practices - Anthropic</summary>

This guide provides specific prompt engineering techniques for Claude 4 models (Opus 4 and Sonnet 4) to help you achieve optimal results in your applications. These models have been trained for more precise instruction following than previous generations of Claude models.

## General principles

### Be explicit with your instructions

Claude 4 models respond well to clear, explicit instructions. Being specific about your desired output can help enhance results. Customers who desire the “above and beyond” behavior from previous Claude models might need to more explicitly request these behaviors with Claude 4.

Example: Creating an analytics dashboard

**Less effective:**

```text
Create an analytics dashboard

```

**More effective:**

```text
Create an analytics dashboard. Include as many relevant features and interactions as possible. Go beyond the basics to create a fully-featured implementation.

```

### Add context to improve performance

Providing context or motivation behind your instructions, such as explaining to Claude why such behavior is important, can help Claude 4 better understand your goals and deliver more targeted responses.

Example: Formatting preferences

**Less effective:**

```text
NEVER use ellipses

```

**More effective:**

```text
Your response will be read aloud by a text-to-speech engine, so never use ellipses since the text-to-speech engine will not know how to pronounce them.

```

Claude is smart enough to generalize from the explanation.

### Be vigilant with examples & details

Claude 4 models pay attention to details and examples as part of instruction following. Ensure that your examples align with the behaviors you want to encourage and minimize behaviors you want to avoid.

## Guidance for specific situations

### Control the format of responses

There are a few ways that we have found to be particularly effective in steering output formatting in Claude 4 models:

1. **Tell Claude what to do instead of what not to do**
   - Instead of: “Do not use markdown in your response”
   - Try: “Your response should be composed of smoothly flowing prose paragraphs.”
2. **Use XML format indicators**
   - Try: “Write the prose sections of your response in <smoothly\_flowing\_prose\_paragraphs> tags.”
3. **Match your prompt style to the desired output**

The formatting style used in your prompt may influence Claude’s response style. If you are still experiencing steerability issues with output formatting, we recommend as best as you can matching your prompt style to your desired output style. For example, removing markdown from your prompt can reduce the volume of markdown in the output.

### Leverage thinking & interleaved thinking capabilities

Claude 4 offers thinking capabilities that can be especially helpful for tasks involving reflection after tool use or complex multi-step reasoning. You can guide its initial or interleaved thinking for better results.

Example prompt

```text
After receiving tool results, carefully reflect on their quality and determine optimal next steps before proceeding. Use your thinking to plan and iterate based on this new information, and then take the best next action.

```

For more information on thinking capabilities, see Extended thinking.

### Optimize parallel tool calling

Claude 4 models excel at parallel tool execution. They have a high success rate in using parallel tool calling without any prompting to do so, but some minor prompting can boost this behavior to ~100% parallel tool use success rate. We have found this prompt to be most effective:

Sample prompt for agents

```text
For maximum efficiency, whenever you need to perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially.

```

### Reduce file creation in agentic coding

Claude 4 models may sometimes create new files for testing and iteration purposes, particularly when working with code. This approach allows Claude to use files, especially python scripts, as a ‘temporary scratchpad’ before saving its final output. Using temporary files can improve outcomes particularly for agentic coding use cases.

If you’d prefer to minimize net new file creation, you can instruct Claude to clean up after itself:

Sample prompt

```text
If you create any temporary new files, scripts, or helper files for iteration, clean up these files by removing them at the end of the task.

```

### Enhance visual and frontend code generation

For frontend code generation, you can steer Claude 4 models to create complex, detailed, and interactive designs by providing explicit encouragement:

Sample prompt

```text
Don't hold back. Give it your all.

```

You can also improve Claude’s frontend performance in specific areas by providing additional modifiers and details on what to focus on:

- “Include as many relevant features and interactions as possible”
- “Add thoughtful details like hover states, transitions, and micro-interactions”
- “Create an impressive demonstration showcasing web development capabilities”
- “Apply design principles: hierarchy, contrast, balance, and movement”

### Avoid focusing on passing tests and hard-coding

Frontier language models can sometimes focus too heavily on making tests pass at the expense of more general solutions. To prevent this behavior and ensure robust, generalizable solutions:

Sample prompt

```text
Please write a high quality, general purpose solution. Implement a solution that works correctly for all valid inputs, not just the test cases. Do not hard-code values or create solutions that only work for specific test inputs. Instead, implement the actual logic that solves the problem generally.

Focus on understanding the problem requirements and implementing the correct algorithm. Tests are there to verify correctness, not to define the solution. Provide a principled implementation that follows best practices and software design principles.

If the task is unreasonable or infeasible, or if any of the tests are incorrect, please tell me. The solution should be robust, maintainable, and extendable.

```

## Migration considerations

When migrating from Sonnet 3.7 to Claude 4:

1. **Be specific about desired behavior**: Consider describing exactly what you’d like to see in the output.

2. **Frame your instructions with modifiers**: Adding modifiers that encourage Claude to increase the quality and detail of its output can help better shape Claude’s performance. For example, instead of “Create an analytics dashboard”, use “Create an analytics dashboard. Include as many relevant features and interactions as possible. Go beyond the basics to create a fully-featured implementation.”

3. **Request specific features explicitly**: Animations and interactive elements should be requested explicitly when desired.

### Original URL
https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices
</details>

---
<details>
<summary>Building Effective AI Agents \ Anthropic</summary>

# Building effective agents

Published Dec 19, 2024

We've worked with dozens of teams building LLM agents across industries. Consistently, the most successful implementations use simple, composable patterns rather than complex frameworks.

Over the past year, we've worked with dozens of teams building large language model (LLM) agents across industries. Consistently, the most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns.

In this post, we share what we’ve learned from working with our customers and building agents ourselves, and give practical advice for developers on building effective agents.

## What are agents?

"Agent" can be defined in several ways. Some customers define agents as fully autonomous systems that operate independently over extended periods, using various tools to accomplish complex tasks. Others use the term to describe more prescriptive implementations that follow predefined workflows. At Anthropic, we categorize all these variations as **agentic systems**, but draw an important architectural distinction between **workflows** and **agents**:

- **Workflows** are systems where LLMs and tools are orchestrated through predefined code paths.
- **Agents**, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks.

Below, we will explore both types of agentic systems in detail. In Appendix 1 (“Agents in Practice”), we describe two domains where customers have found particular value in using these kinds of systems.

## When (and when not) to use agents

When building applications with LLMs, we recommend finding the simplest solution possible, and only increasing complexity when needed. This might mean not building agentic systems at all. Agentic systems often trade latency and cost for better task performance, and you should consider when this tradeoff makes sense.

When more complexity is warranted, workflows offer predictability and consistency for well-defined tasks, whereas agents are the better option when flexibility and model-driven decision-making are needed at scale. For many applications, however, optimizing single LLM calls with retrieval and in-context examples is usually enough.

## When and how to use frameworks

There are many frameworks that make agentic systems easier to implement, including:

- LangGraph from LangChain;
- Amazon Bedrock's AI Agent framework;
- Rivet, a drag and drop GUI LLM workflow builder; and
- Vellum, another GUI tool for building and testing complex workflows.

These frameworks make it easy to get started by simplifying standard low-level tasks like calling LLMs, defining and parsing tools, and chaining calls together. However, they often create extra layers of abstraction that can obscure the underlying prompts ​​and responses, making them harder to debug. They can also make it tempting to add complexity when a simpler setup would suffice.

We suggest that developers start by using LLM APIs directly: many patterns can be implemented in a few lines of code. If you do use a framework, ensure you understand the underlying code. Incorrect assumptions about what's under the hood are a common source of customer error.

See our cookbook for some sample implementations.

## Building blocks, workflows, and agents

In this section, we’ll explore the common patterns for agentic systems we’ve seen in production. We'll start with our foundational building block—the augmented LLM—and progressively increase complexity, from simple compositional workflows to autonomous agents.

### Building block: The augmented LLM

The basic building block of agentic systems is an LLM enhanced with augmentations such as retrieval, tools, and memory. Our current models can actively use these capabilities—generating their own search queries, selecting appropriate tools, and determining what information to retain.

We recommend focusing on two key aspects of the implementation: tailoring these capabilities to your specific use case and ensuring they provide an easy, well-documented interface for your LLM. While there are many ways to implement these augmentations, one approach is through our recently released Model Context Protocol, which allows developers to integrate with a growing ecosystem of third-party tools with a simple client implementation.

For the remainder of this post, we'll assume each LLM call has access to these augmented capabilities.

### Workflow: Prompt chaining

Prompt chaining decomposes a task into a sequence of steps, where each LLM call processes the output of the previous one. You can add programmatic checks (see "gate” in the diagram below) on any intermediate steps to ensure that the process is still on track.

**When to use this workflow:** This workflow is ideal for situations where the task can be easily and cleanly decomposed into fixed subtasks. The main goal is to trade off latency for higher accuracy, by making each LLM call an easier task.

**Examples where prompt chaining is useful:**

- Generating Marketing copy, then translating it into a different language.
- Writing an outline of a document, checking that the outline meets certain criteria, then writing the document based on the outline.

### Workflow: Routing

Routing classifies an input and directs it to a specialized followup task. This workflow allows for separation of concerns, and building more specialized prompts. Without this workflow, optimizing for one kind of input can hurt performance on other inputs.

**When to use this workflow:** Routing works well for complex tasks where there are distinct categories that are better handled separately, and where classification can be handled accurately, either by an LLM or a more traditional classification model/algorithm.

**Examples where routing is useful:**

- Directing different types of customer service queries (general questions, refund requests, technical support) into different downstream processes, prompts, and tools.
- Routing easy/common questions to smaller models like Claude 3.5 Haiku and hard/unusual questions to more capable models like Claude 3.5 Sonnet to optimize cost and speed.

### Workflow: Parallelization

LLMs can sometimes work simultaneously on a task and have their outputs aggregated programmatically. This workflow, parallelization, manifests in two key variations:

- **Sectioning**: Breaking a task into independent subtasks run in parallel.
- **Voting:** Running the same task multiple times to get diverse outputs.

**When to use this workflow:** Parallelization is effective when the divided subtasks can be parallelized for speed, or when multiple perspectives or attempts are needed for higher confidence results. For complex tasks with multiple considerations, LLMs generally perform better when each consideration is handled by a separate LLM call, allowing focused attention on each specific aspect.

**Examples where parallelization is useful:**

- **Sectioning**:
  - Implementing guardrails where one model instance processes user queries while another screens them for inappropriate content or requests. This tends to perform better than having the same LLM call handle both guardrails and the core response.
  - Automating evals for evaluating LLM performance, where each LLM call evaluates a different aspect of the model’s performance on a given prompt.
- **Voting**:
  - Reviewing a piece of code for vulnerabilities, where several different prompts review and flag the code if they find a problem.
  - Evaluating whether a given piece of content is inappropriate, with multiple prompts evaluating different aspects or requiring different vote thresholds to balance false positives and negatives.

### Workflow: Orchestrator-workers

In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results.

**When to use this workflow:** This workflow is well-suited for complex tasks where you can’t predict the subtasks needed (in coding, for example, the number of files that need to be changed and the nature of the change in each file likely depend on the task). Whereas it’s topographically similar, the key difference from parallelization is its flexibility—subtasks aren't pre-defined, but determined by the orchestrator based on the specific input.

**Example where orchestrator-workers is useful:**

- Coding products that make complex changes to multiple files each time.
- Search tasks that involve gathering and analyzing information from multiple sources for possible relevant information.

### Workflow: Evaluator-optimizer

In the evaluator-optimizer workflow, one LLM call generates a response while another provides evaluation and feedback in a loop.

**When to use this workflow:** This workflow is particularly effective when we have clear evaluation criteria, and when iterative refinement provides measurable value. The two signs of good fit are, first, that LLM responses can be demonstrably improved when a human articulates their feedback; and second, that the LLM can provide such feedback. This is analogous to the iterative writing process a human writer might go through when producing a polished document.

**Examples where evaluator-optimizer is useful:**

- Literary translation where there are nuances that the translator LLM might not capture initially, but where an evaluator LLM can provide useful critiques.
- Complex search tasks that require multiple rounds of searching and analysis to gather comprehensive information, where the evaluator decides whether further searches are warranted.

### Agents

Agents are emerging in production as LLMs mature in key capabilities—understanding complex inputs, engaging in reasoning and planning, using tools reliably, and recovering from errors. Agents begin their work with either a command from, or interactive discussion with, the human user. Once the task is clear, agents plan and operate independently, potentially returning to the human for further information or judgement. During execution, it's crucial for the agents to gain “ground truth” from the environment at each step (such as tool call results or code execution) to assess its progress. Agents can then pause for human feedback at checkpoints or when encountering blockers. The task often terminates upon completion, but it’s also common to include stopping conditions (such as a maximum number of iterations) to maintain control.

Agents can handle sophisticated tasks, but their implementation is often straightforward. They are typically just LLMs using tools based on environmental feedback in a loop. It is therefore crucial to design toolsets and their documentation clearly and thoughtfully. We expand on best practices for tool development in Appendix 2 ("Prompt Engineering your Tools").

**When to use agents:** Agents can be used for open-ended problems where it’s difficult or impossible to predict the required number of steps, and where you can’t hardcode a fixed path. The LLM will potentially operate for many turns, and you must have some level of trust in its decision-making. Agents' autonomy makes them ideal for scaling tasks in trusted environments.

The autonomous nature of agents means higher costs, and the potential for compounding errors. We recommend extensive testing in sandboxed environments, along with the appropriate guardrails.

**Examples where agents are useful:**

The following examples are from our own implementations:

- A coding Agent to resolve SWE-bench tasks, which involve edits to many files based on a task description;
- Our “computer use” reference implementation, where Claude uses a computer to accomplish tasks.

## Combining and customizing these patterns

These building blocks aren't prescriptive. They're common patterns that developers can shape and combine to fit different use cases. The key to success, as with any LLM features, is measuring performance and iterating on implementations. To repeat: you should consider adding complexity _only_ when it demonstrably improves outcomes.

## Summary

Success in the LLM space isn't about building the most sophisticated system. It's about building the _right_ system for your needs. Start with simple prompts, optimize them with comprehensive evaluation, and add multi-step agentic systems only when simpler solutions fall short.

When implementing agents, we try to follow three core principles:

1. Maintain **simplicity** in your agent's design.
2. Prioritize **transparency** by explicitly showing the agent’s planning steps.
3. Carefully craft your agent-computer interface (ACI) through thorough tool **documentation and testing**.

Frameworks can help you get started quickly, but don't hesitate to reduce abstraction layers and build with basic components as you move to production. By following these principles, you can create agents that are not only powerful but also reliable, maintainable, and trusted by their users.

### Acknowledgements

Written by Erik Schluntz and Barry Zhang. This work draws upon our experiences building agents at Anthropic and the valuable insights shared by our customers, for which we're deeply grateful.

## Appendix 1: Agents in practice

Our work with customers has revealed two particularly promising applications for AI agents that demonstrate the practical value of the patterns discussed above. Both applications illustrate how agents add the most value for tasks that require both conversation and action, have clear success criteria, enable feedback loops, and integrate meaningful human oversight.

### A. Customer support

Customer support combines familiar chatbot interfaces with enhanced capabilities through tool integration. This is a natural fit for more open-ended agents because:

- Support interactions naturally follow a conversation flow while requiring access to external information and actions;
- Tools can be integrated to pull customer data, order history, and knowledge base articles;
- Actions such as issuing refunds or updating tickets can be handled programmatically; and
- Success can be clearly measured through user-defined resolutions.

Several companies have demonstrated the viability of this approach through usage-based pricing models that charge only for successful resolutions, showing confidence in their agents' effectiveness.

### B. Coding agents

The software development space has shown remarkable potential for LLM features, with capabilities evolving from code completion to autonomous problem-solving. Agents are particularly effective because:

- Code solutions are verifiable through automated tests;
- Agents can iterate on solutions using test results as feedback;
- The problem space is well-defined and structured; and
- Output quality can be measured objectively.

In our own implementation, agents can now solve real GitHub issues in the SWE-bench Verified benchmark based on the pull request description alone. However, whereas automated testing helps verify functionality, human review remains crucial for ensuring solutions align with broader system requirements.

## Appendix 2: Prompt engineering your tools

No matter which agentic system you're building, tools will likely be an important part of your agent. Tools enable Claude to interact with external services and APIs by specifying their exact structure and definition in our API. When Claude responds, it will include a tool use block in the API response if it plans to invoke a tool. Tool definitions and specifications should be given just as much prompt engineering attention as your overall prompts. In this brief appendix, we describe how to prompt engineer your tools.

There are often several ways to specify the same action. For instance, you can specify a file edit by writing a diff, or by rewriting the entire file. For structured output, you can return code inside markdown or inside JSON. In software engineering, differences like these are cosmetic and can be converted losslessly from one to the other. However, some formats are much more difficult for an LLM to write than others. Writing a diff requires knowing how many lines are changing in the chunk header before the new code is written. Writing code inside JSON (compared to markdown) requires extra escaping of newlines and quotes.

Our suggestions for deciding on tool formats are the following:

- Give the model enough tokens to "think" before it writes itself into a corner.
- Keep the format close to what the model has seen naturally occurring in text on the internet.
- Make sure there's no formatting "overhead" such as having to keep an accurate count of thousands of lines of code, or string-escaping any code it writes.

One rule of thumb is to think about how much effort goes into human-computer interfaces (HCI), and plan to invest just as much effort in creating good _agent_-computer interfaces (ACI). Here are some thoughts on how to do so:

- Put yourself in the model's shoes. Is it obvious how to use this tool, based on the description and parameters, or would you need to think carefully about it? If so, then it’s probably also true for the model. A good tool definition often includes example usage, edge cases, input format requirements, and clear boundaries from other tools.
- How can you change parameter names or descriptions to make things more obvious? Think of this as writing a great docstring for a junior developer on your team. This is especially important when using many similar tools.
- Test how the model uses your tools: Run many example inputs in our workbench to see what mistakes the model makes, and iterate.
- Poka-yoke your tools. Change the arguments so that it is harder to make mistakes.

While building our agent for SWE-bench, we actually spent more time optimizing our tools than the overall prompt. For example, we found that the model would make mistakes with tools using relative filepaths after the agent had moved out of the root directory. To fix this, we changed the tool to always require absolute filepaths—and we found that the model used this method flawlessly.

### Original URL
https://www.anthropic.com/engineering/building-effective-agents
</details>

---
<details>
<summary>Chain complex prompts for stronger performance - Anthropic</summary>

When working with complex tasks, Claude can sometimes drop the ball if you try to handle everything in a single prompt. Chain of thought (CoT) prompting is great, but what if your task has multiple distinct steps that each require in-depth thought?

Enter prompt chaining: breaking down complex tasks into smaller, manageable subtasks.

## Why chain prompts?

1. **Accuracy**: Each subtask gets Claude’s full attention, reducing errors.
2. **Clarity**: Simpler subtasks mean clearer instructions and outputs.
3. **Traceability**: Easily pinpoint and fix issues in your prompt chain.

---

## When to chain prompts

Use prompt chaining for multi-step tasks like research synthesis, document analysis, or iterative content creation. When a task involves multiple transformations, citations, or instructions, chaining prevents Claude from dropping or mishandling steps.

**Remember:** Each link in the chain gets Claude’s full attention!

**Debugging tip**: If Claude misses a step or performs poorly, isolate that step in its own prompt. This lets you fine-tune problematic steps without redoing the entire task.

---

## How to chain prompts

1. **Identify subtasks**: Break your task into distinct, sequential steps.
2. **Structure with XML for clear handoffs**: Use XML tags to pass outputs between prompts.
3. **Have a single-task goal**: Each subtask should have a single, clear objective.
4. **Iterate**: Refine subtasks based on Claude’s performance.

### Example chained workflows:

- **Multi-step analysis**: See the legal and business examples below.
- **Content creation pipelines**: Research → Outline → Draft → Edit → Format.
- **Data processing**: Extract → Transform → Analyze → Visualize.
- **Decision-making**: Gather info → List options → Analyze each → Recommend.
- **Verification loops**: Generate content → Review → Refine → Re-review.

**Optimization tip**: For tasks with independent subtasks (like analyzing multiple docs), create separate prompts and run them in parallel for speed.

### Advanced: Self-correction chains

You can chain prompts to have Claude review its own work! This catches errors and refines outputs, especially for high-stakes tasks.

Example: Self-correcting research summary

#### Prompt 1

| Role | Content |
| --- | --- |
| User | Summarize this medical research paper.<br><paper>{{RESEARCH\_PAPER}}</paper><br>Focus on methodology, findings, and clinical implications. |
| Assistant<br>{{SUMMARY}} | Methodology: Double-blind RCT with 500 Type 2 diabetes patients. Group A (n=250) received novel GLP-1/GIP dual agonist; Group B (n=250) received standard GLP-1 agonist. 24-week trial, primary endpoint: HbA1c reduction.<br>Findings:<br>\- Group A: 1.8% mean HbA1c reduction<br>\- Group B: 1.1% mean HbA1c reduction<br>\- 68% of Group A vs. 52% of Group B achieved HbA1c <7%<br>\- Group A: 5.2kg mean weight loss<br>\- Group B: 3.1kg mean weight loss<br>Safety: Similar adverse event profiles, no serious events attributed to treatment.<br>Clinical Implications: Dual agonist shows superior glycemic control and weight loss compared to standard therapy. Could be a game-changer, especially for obese diabetics. |

#### Prompt 2

| Role | Content |
| --- | --- |
| User | Your task is to provide feedback on a research paper summary. Here is a summary of a medical research paper:<br><summary><br>{{SUMMARY}}<br></summary><br>Here is the research paper:<br><paper><br>{{RESEARCH\_PAPER}}<br></paper><br>Review this summary for accuracy, clarity, and completeness on a graded A-F scale. |
| Assistant<br>{{FEEDBACK}} | 1\. Accuracy: A<br> \- Correctly reported key stats: HbA1c reductions, weight loss.<br> \- Accurately noted safety profile and lack of serious adverse events.<br> \- Assertion of “game-changer” is justified by significantly better outcomes.<br>2\. Clarity: A-<br> \- Concise and well-structured with clear sections.<br> \- Good use of comparative language (e.g., “up 3% from Q1”) for context.<br> \- Minor issue: Acronyms like “RCT,” “GLP-1,” and “GIP” not defined. Consider spelling out first use.<br>3\. Completeness: B+<br> \- Covers all requested areas: methodology, findings, clinical implications.<br> \- Missing: Baseline characteristics of groups (were they well-matched?).<br> \- Missing: Any secondary endpoints (e.g., blood pressure, lipids).<br> \- Consider: Long-term implications or need for further research.<br>Overall, the summary is accurate and clear, providing key insights for a quick grasp of the study. To elevate to A+, define acronyms, mention group baseline stats, note any secondary outcomes, and briefly touch on the need for long-term studies. |

#### Prompt 3

| Role | Content |
| --- | --- |
| User | Your task is to improve a paper summary given feedback. Here is the first draft of a medical research paper:<br><summary><br>{{SUMMARY}}<br></summary><br>Here is the research paper:<br><paper><br>{{RESEARCH\_PAPER}}<br></paper><br>Here is the feedback:<br><feedback><br>{{FEEDBACK}}<br></feedback><br>Update the summary based on the feedback. |
| Assistant | Methodology: Double-blind randomized controlled trial (RCT) with 500 Type 2 diabetes patients. Group A (n=250) received a novel glucagon-like peptide-1/glucose-dependent insulinotropic polypeptide (GLP-1/GIP) dual agonist; Group B (n=250) received a standard GLP-1 agonist. The 24-week trial’s primary endpoint was HbA1c reduction. Groups were well-matched for age, BMI, and baseline HbA1c.<br>Findings:<br>\- Group A: 1.8% mean HbA1c reduction<br>\- Group B: 1.1% mean HbA1c reduction<br>\- 68% of Group A vs. 52% of Group B achieved HbA1c <7%<br>\- Group A: 5.2kg mean weight loss<br>\- Group B: 3.1kg mean weight loss<br>Secondary endpoints showed improvements in blood pressure and lipid profiles for both groups, with Group A showing slightly better results.<br>Safety: Similar adverse event profiles, with no serious events attributed to either treatment.<br>Clinical Implications: The dual GLP-1/GIP agonist demonstrates superior glycemic control and weight loss compared to standard GLP-1 therapy. These results suggest it could be a significant advancement, particularly for obese diabetics. However, longer-term studies are needed to confirm sustained efficacy and safety, as well as to investigate potential benefits on cardiovascular outcomes. |

---

## Examples

Example: Analyzing a legal contract (without chaining)

Note that Claude misses the instruction to provide proposed changes in its email draft.

| Role | Content |
| --- | --- |
| User | You’re our Chief Legal Officer. Review this SaaS contract for risks, focusing on data privacy, SLAs, and liability caps.<br><contract><br>{{CONTRACT}}<br></contract><br>Then draft an email to the vendor with your concerns and proposed changes. |
| Assistant | After reviewing the SaaS contract, I’ve identified several concerns:<br>1\. Data Privacy (Section 7):<br> \- Issue: Vendor has broad rights to use client data for “service improvement.”<br> \- Risk: Could lead to unauthorized data sharing or monetization.<br>2\. SLAs (Section 5):<br> \- Issue: 99.5% uptime guarantee is below industry standard.<br> \- Risk: Excessive downtime could impact our operations.<br>3\. Liability (Section 9):<br> \- Issue: Cap set at 3 months of fees.<br> \- Risk: Inadequate for potential data breaches or service failures.<br>Draft email:<br>Dear {{VENDOR}},<br>After reviewing the SaaS agreement, we have concerns:<br>1\. Data Privacy: We require that our data only be used for direct service provision, not broad “service improvement.” Please amend Section 7.<br>2\. SLAs: 99.5% uptime is insufficient. We need 99.99% with clearly defined penalties. Please revise Section 5.<br>3\. Liability: The current cap of 3 months’ fees is inadequate. We propose a cap of 12 months’ fees, with no cap for data breaches or willful misconduct.<br>We value our potential partnership and believe these changes will set us up for success. Please send a revised agreement.<br>Best regards,<br>{{CLO\_NAME}} |

Example: Analyzing a legal contract (with chaining)

#### Prompt 1

| Role | Content |
| --- | --- |
| User | You’re our Chief Legal Officer. Review this SaaS contract for risks, focusing on data privacy, SLAs, and liability caps.<br><contract><br>{{CONTRACT}}<br></contract><br>Output your findings in <risks> tags. |
| Assistant<br>{{CONCERNS}} | 1\. Data Privacy (Section 7):<br> \- Issue: Vendor has broad rights to use client data for “service improvement.”<br> \- Risk: Could lead to unauthorized data sharing or monetization of client data.<br> \- Impact: Potential breach of client trust, regulatory fines (GDPR, CCPA).<br>2\. SLAs (Section 5):<br> \- Issue: 99.5% uptime guarantee is below industry standard (typically 99.99%).<br> \- Risk: Up to 43.8 hours of downtime per year vs. 52.6 minutes at 99.99%.<br> \- Impact: Operational disruptions, revenue loss, client dissatisfaction.<br>3\. Liability (Section 9):<br> \- Issue: Liability cap set at 3 months of fees.<br> \- Risk: Inadequate for potential data breaches, IP theft, or prolonged service failures.<br> \- Impact: Insufficient compensation for damages that could far exceed the cap, like class-action lawsuits or regulatory fines. |

#### Prompt 2

| Role | Content |
| --- | --- |
| User | Draft an email to a SaaS product vendor outlining the following concerns and proposing changes. Here are the concerns:<br><concerns><br>{{CONCERNS}}<br></concerns> |
| Assistant<br>{{EMAIL}} | Dear {{VENDOR}},<br>After a thorough review of the proposed SaaS agreement, our legal team has identified several high-priority concerns that must be addressed before we can proceed:<br>1\. Data Privacy (Section 7):<br> \- Current: Broad rights to use client data for “service improvement.”<br> \- Concern: Potential unauthorized data sharing or monetization.<br> \- Proposed Change: Limit data use to direct service provision. Any use for improvement, analytics, or marketing must be opt-in, revocable, and anonymized.<br>2\. SLAs (Section 5):<br> \- Current: 99.5% uptime guarantee.<br> \- Concern: Up to 43.8 hours of downtime per year, risking operational disruptions.<br> \- Proposed Change: 99.99% uptime with tiered service credits. E.g., 10% credit for <99.99%, 25% for <99.9%, 50% for <99.5%.<br>3\. Liability (Section 9):<br> \- Current: Cap at 3 months of fees.<br> \- Concern: Grossly inadequate for data breaches, IP theft, or prolonged outages.<br> \- Proposed Change: 12 months’ fees for general liability. Uncapped for data breaches, willful misconduct, IP infringement.<br>These changes are non-negotiable for us. They align with industry standards and our commitment to data security, operational excellence, and fiduciary responsibility. We value the potential of our partnership, and addressing these will set a strong foundation.<br>Please have your legal team review and incorporate these changes. We’re happy to discuss further to ensure mutual understanding and agreement.<br>Best regards,<br>{{CLO\_NAME}} |

#### Prompt 3

| Role | Content |
| --- | --- |
| User | Your task is to review an email and provide feedback. Here is the email:<br><email><br>{{EMAIL}}<br></email><br>Give feedback on tone, clarity, and professionalism. |
| Assistant | 1\. Tone:<br> \- Professional and assertive without being confrontational.<br> \- Use of “non-negotiable” is strong but justified given the criticality of the issues.<br> \- Collaborative language (“we value,” “happy to discuss”) balances the firmness, maintaining a constructive tone.<br>2\. Clarity:<br> \- Excellent structuring: Current state, concern, and proposed change for each issue.<br> \- Technical terms (e.g., “EBITDA,” “tiered service credits”) are appropriate for a B2B SaaS context.<br> \- Action items are clear: vendor’s legal team to review and incorporate changes.<br>3\. Professionalism:<br> \- Focuses on mutual benefit and industry standards, not just our demands.<br> \- Mentions broader responsibilities (data security, operational excellence, fiduciary duty), elevating the discussion.<br> \- Offers further discussion, showing commitment to understanding and resolving issues.<br>Overall, the email strikes the right balance of assertiveness and collaboration. It clearly articulates our concerns, proposes specific solutions, and maintains a professional, partnership-oriented tone. No revisions needed; proceed with sending. |

Example: Multitenancy strategy review

#### Prompt 1

| Role | Content |
| --- | --- |
| User | As a senior solutions architect, review and analyze this multitenancy strategy for our new enterprise SaaS product.<br><strategy><br>{{STRATEGY}}<br></strategy><br>Focus on scalability, security, and cost-effectiveness. |
| Assistant<br>{{ANALYSIS}} | 1\. Scalability:<br> \+ Database sharding by tenant reduces contention, supports high growth.<br> \+ Microservices allow independent scaling of components.<br> \- Shared app servers could bottleneck under extreme load.<br> \- No mention of auto-scaling or load balancing for app tier.<br>2\. Security:<br> \+ Tenant data isolation via sharding mitigates data leaks.<br> \+ JWT with short expiry and refresh tokens is industry best practice.<br> \- No mention of encryption (at rest, in transit) or key management.<br> \- IAM strategy lacks detail on roles, permissions, or RBAC.<br>3\. Cost-effectiveness:<br> \+ Shared app servers reduce idle resource costs.<br> \+ Serverless functions for infrequent tasks optimize costs.<br> \- Per-tenant DB could increase costs at scale vs. optimized multi-tenant DB.<br> \- No discussion of reserved instances or spot instances for cost saving. |

#### Prompt 2

| Role | Content |
| --- | --- |
| User | Draft a strategy review document for engineering leadership based on this analysis of a multitenancy strategy.<br><strategy><br>{{STRATEGY}}<br></strategy><br><analysis><br>{{ANALYSIS}}<br></analysis><br>Include an executive summary, detailed analysis, and recommendations. |
| Assistant<br>{{STRATEGY\_DOC}} | \# Multitenancy Strategy Review: SaaS Enterprise Product<br>\## Executive Summary<br>The proposed strategy demonstrates a solid foundation in scalability and security principles but has gaps in implementation details. It excels in data isolation and component-level scaling but lacks robust app-tier scaling and advanced security measures. Cost optimizations are present but may falter at high scale. With targeted enhancements, this strategy can effectively support our enterprise SaaS ambitions.<br>\## Detailed Analysis<br>\### 1\. Scalability: B+<br>Strengths:<br>\- Database sharding by tenant is excellent. It minimizes query latency, supports high data growth, and facilitates seamless per-tenant scaling.<br>\- Microservices architecture allows scaling individual components based on demand, crucial for handling diverse enterprise workloads.<br>Weaknesses:<br>\- Shared application servers risk becoming a bottleneck. As tenant count and traffic grow, these could degrade performance across all tenants.<br>\- Absence of auto-scaling and load balancing for the app tier is a critical oversight. These are essential for handling traffic spikes without over-provisioning.<br>\### 2\. Security: B-<br>Strengths:<br>\- Sharded databases provide strong tenant data isolation, mitigating risks of data leaks or unauthorized access between tenants.<br>\- JWT with short expiry and refresh tokens align with OWASP best practices, reducing token vulnerability windows.<br>Weaknesses:<br>\- No mention of data encryption at rest or in transit (TLS 1.3). This is non-negotiable for enterprise data.<br>\- Key management strategy is missing. How are encryption keys, JWT secrets, and DB credentials managed and rotated?<br>\- IAM lacks depth. We need RBAC with well-defined roles, attribute-based access control (ABAC) for fine-grained permissions, and integration with enterprise identity providers (SAML, OIDC).<br>\### 3\. Cost-effectiveness: B<br>Strengths:<br>\- Shared app servers reduce idle capacity costs, good for steady-state traffic.<br>\- Serverless for infrequent tasks (e.g., reporting) is smart; pay only when these resources are used.<br>Weaknesses:<br>\- Per-tenant databases may not be cost-effective at scale compared to a well-designed multi-tenant database with robust data partitioning.<br>\- No mention of cost-saving measures like reserved instances for predictable workloads or spot instances for fault-tolerant tasks.<br>\- Lack of app-tier auto-scaling could lead to over-provisioning, increasing costs.<br>\## Recommendations<br>1\. **App Tier Resilience**: Implement auto-scaling with Kubernetes or cloud-native services (EKS, GKE). Deploy across availability zones with a CDN for static assets.<br>2\. **Security Hardening**:<br> a. Encrypt data at rest (AES-256) and in transit (TLS 1.3).<br> b. Use a managed KMS (AWS KMS, GCP KMS) for key management.<br> c. Enhance IAM with RBAC, ABAC, and SAML/OIDC for enterprise SSO.<br>3\. **Cost Optimization**:<br> a. Evaluate columnar multi-tenant DB (e.g., Snowflake) vs. sharding at scale.<br> b. Use reserved instances for app servers, spot instances for non-critical workloads.<br> c. Implement proactive monitoring and right-sizing to avoid overprovisioning.<br>4\. **Proof of Concept (PoC)**: Before full implementation, conduct a PoC with a high-scale simulation:<br> a. Test sharded DB vs. optimized multi-tenant DB for cost and performance.<br> b. Stress-test app tier with simulated enterprise-level traffic spikes.<br> c. Conduct third-party security audit, focusing on data isolation and access controls.<br>With these enhancements, our multi-tenancy strategy will be enterprise-grade: scalable, secure, and cost-effective. The PoC will validate our design choices and instill confidence in our enterprise clients. |

#### Prompt 3

| Role | Content |
| --- | --- |
| User | Grade this strategy review document for clarity, actionability, and alignment with enterprise priorities. <br><priorities><br>{{PRIORITIES}}<br></priorities><br><strategy\_doc><br>{{STRATEGY\_DOC}}<br></strategy\_doc> |
| Assistant | 1\. Clarity: A<br> \+ Clear structure with Executive Summary, Detailed Analysis, and Recommendations.<br> \+ Analysis uses a grading system (B+, B-) for quick grasp of strengths/weaknesses.<br> \+ Technical terms (RBAC, ABAC, TLS) are explained or contextualized, aiding non-technical leadership.<br> \+ Visual aids (bullet points, indentation) enhance readability.<br>2\. Actionability: A-<br> \+ Recommendations are specific and technical (e.g., “Implement auto-scaling with Kubernetes”).<br> \+ PoC plan with clear objectives provides a roadmap.<br> \+ Missing: Timelines, resource requirements, or team assignments for each recommendation.<br> \+ Consider: Add a “Next Steps” section with immediate actions, roles, and a high-level timeline.<br>3\. Enterprise Alignment: A+<br> \+ Directly addresses enterprise priorities:<br> \- Scalability: Discusses handling enterprise-level traffic spikes.<br> \- Security: Focuses on data isolation, encryption, and enterprise IAM.<br> \- Cost: Strategies align with enterprise FinOps (reserved instances, right-sizing).<br> \+ PoC with “high-scale simulation” and security audit shows commitment to enterprise-grade quality.<br> \+ Language resonates with enterprise concerns: “unacceptable risk,” “enterprise-grade,” “instill confidence.”<br>Overall, the document excels in clarity and enterprise alignment. The actionability is strong but could be elevated to A+ with a “Next Steps” section. The technical depth, coupled with clear explanations, makes this highly effective for both technical and non-technical leadership. Great work! |

### Original URL
https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts
</details>

---
<details>
<summary>Workflows and Agents</summary>

This guide reviews common patterns for agentic systems. In describing these systems, it can be useful to make a distinction between "workflows" and "agents". One way to think about this difference is nicely explained here by Anthropic:

> Workflows are systems where LLMs and tools are orchestrated through predefined code paths.
> Agents, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks.

Here is a simple way to visualize these differences:


When building agents and workflows, LangGraph offers a number of benefits including persistence, streaming, and support for debugging as well as deployment.

## Prompt chaining

In prompt chaining, each LLM call processes the output of the previous one.

As noted in the Anthropic blog:

> Prompt chaining decomposes a task into a sequence of steps, where each LLM call processes the output of the previous one. You can add programmatic checks (see "gate” in the diagram below) on any intermediate steps to ensure that the process is still on track.
>
> When to use this workflow: This workflow is ideal for situations where the task can be easily and cleanly decomposed into fixed subtasks. The main goal is to trade off latency for higher accuracy, by making each LLM call an easier task.

Graph API

```md-code__content
import { StateGraph, Annotation } from "@langchain/langgraph";

// Graph state
const StateAnnotation = Annotation.Root({
  topic: Annotation<string>,
  joke: Annotation<string>,
  improvedJoke: Annotation<string>,
  finalJoke: Annotation<string>,
});

// Define node functions

// First LLM call to generate initial joke
async function generateJoke(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke(`Write a short joke about ${state.topic}`);
  return { joke: msg.content };
}

// Gate function to check if the joke has a punchline
function checkPunchline(state: typeof StateAnnotation.State) {
  // Simple check - does the joke contain "?" or "!"
  if (state.joke?.includes("?") || state.joke?.includes("!")) {
    return "Pass";
  }
  return "Fail";
}

  // Second LLM call to improve the joke
async function improveJoke(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke(
    `Make this joke funnier by adding wordplay: ${state.joke}`
  );
  return { improvedJoke: msg.content };
}

// Third LLM call for final polish
async function polishJoke(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke(
    `Add a surprising twist to this joke: ${state.improvedJoke}`
  );
  return { finalJoke: msg.content };
}

// Build workflow
const chain = new StateGraph(StateAnnotation)
  .addNode("generateJoke", generateJoke)
  .addNode("improveJoke", improveJoke)
  .addNode("polishJoke", polishJoke)
  .addEdge("__start__", "generateJoke")
  .addConditionalEdges("generateJoke", checkPunchline, {
    Pass: "improveJoke",
    Fail: "__end__"
  })
  .addEdge("improveJoke", "polishJoke")
  .addEdge("polishJoke", "__end__")
  .compile();

// Invoke
const state = await chain.invoke({ topic: "cats" });
console.log("Initial joke:");
console.log(state.joke);
console.log("\n--- --- ---\n");
if (state.improvedJoke !== undefined) {
  console.log("Improved joke:");
  console.log(state.improvedJoke);
  console.log("\n--- --- ---\n");

  console.log("Final joke:");
  console.log(state.finalJoke);
} else {
  console.log("Joke failed quality gate - no punchline detected!");
}

```

```md-code__content
import { task, entrypoint } from "@langchain/langgraph";

// Tasks

// First LLM call to generate initial joke
const generateJoke = task("generateJoke", async (topic: string) => {
  const msg = await llm.invoke(`Write a short joke about ${topic}`);
  return msg.content;
});

// Gate function to check if the joke has a punchline
function checkPunchline(joke: string) {
  // Simple check - does the joke contain "?" or "!"
  if (joke.includes("?") || joke.includes("!")) {
    return "Pass";
  }
  return "Fail";
}

  // Second LLM call to improve the joke
const improveJoke = task("improveJoke", async (joke: string) => {
  const msg = await llm.invoke(
    `Make this joke funnier by adding wordplay: ${joke}`
  );
  return msg.content;
});

// Third LLM call for final polish
const polishJoke = task("polishJoke", async (joke: string) => {
  const msg = await llm.invoke(
    `Add a surprising twist to this joke: ${joke}`
  );
  return msg.content;
});

const workflow = entrypoint(
  "jokeMaker",
  async (topic: string) => {
    const originalJoke = await generateJoke(topic);
    if (checkPunchline(originalJoke) === "Pass") {
      return originalJoke;
    }
    const improvedJoke = await improveJoke(originalJoke);
    const polishedJoke = await polishJoke(improvedJoke);
    return polishedJoke;
  }
);

const stream = await workflow.stream("cats", {
  streamMode: "updates",
});

for await (const step of stream) {
  console.log(step);
}

```

## Parallelization

With parallelization, LLMs work simultaneously on a task:

> LLMs can sometimes work simultaneously on a task and have their outputs aggregated programmatically. This workflow, parallelization, manifests in two key variations: Sectioning: Breaking a task into independent subtasks run in parallel. Voting: Running the same task multiple times to get diverse outputs.
>
> When to use this workflow: Parallelization is effective when the divided subtasks can be parallelized for speed, or when multiple perspectives or attempts are needed for higher confidence results. For complex tasks with multiple considerations, LLMs generally perform better when each consideration is handled by a separate LLM call, allowing focused attention on each specific aspect.

Graph API

```md-code__content
import { StateGraph, Annotation } from "@langchain/langgraph";

// Graph state
const StateAnnotation = Annotation.Root({
  topic: Annotation<string>,
  joke: Annotation<string>,
  story: Annotation<string>,
  poem: Annotation<string>,
  combinedOutput: Annotation<string>,
});

// Nodes
// First LLM call to generate initial joke
async function callLlm1(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke(`Write a joke about ${state.topic}`);
  return { joke: msg.content };
}

// Second LLM call to generate story
async function callLlm2(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke(`Write a story about ${state.topic}`);
  return { story: msg.content };
}

// Third LLM call to generate poem
async function callLlm3(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke(`Write a poem about ${state.topic}`);
  return { poem: msg.content };
}

// Combine the joke, story and poem into a single output
async function aggregator(state: typeof StateAnnotation.State) {
  const combined = `Here's a story, joke, and poem about ${state.topic}!\n\n` +
    `STORY:\n${state.story}\n\n` +
    `JOKE:\n${state.joke}\n\n` +
    `POEM:\n${state.poem}`;
  return { combinedOutput: combined };
}

// Build workflow
const parallelWorkflow = new StateGraph(StateAnnotation)
  .addNode("callLlm1", callLlm1)
  .addNode("callLlm2", callLlm2)
  .addNode("callLlm3", callLlm3)
  .addNode("aggregator", aggregator)
  .addEdge("__start__", "callLlm1")
  .addEdge("__start__", "callLlm2")
  .addEdge("__start__", "callLlm3")
  .addEdge("callLlm1", "aggregator")
  .addEdge("callLlm2", "aggregator")
  .addEdge("callLlm3", "aggregator")
  .addEdge("aggregator", "__end__")
  .compile();

// Invoke
const result = await parallelWorkflow.invoke({ topic: "cats" });
console.log(result.combinedOutput);

```

```md-code__content
import { task, entrypoint } from "@langchain/langgraph";

// Tasks

// First LLM call to generate initial joke
const callLlm1 = task("generateJoke", async (topic: string) => {
  const msg = await llm.invoke(`Write a joke about ${topic}`);
  return msg.content;
});

// Second LLM call to generate story
const callLlm2 = task("generateStory", async (topic: string) => {
  const msg = await llm.invoke(`Write a story about ${topic}`);
  return msg.content;
});

// Third LLM call to generate poem
const callLlm3 = task("generatePoem", async (topic: string) => {
  const msg = await llm.invoke(`Write a poem about ${topic}`);
  return msg.content;
});

// Combine outputs
const aggregator = task("aggregator", async (params: {
  topic: string;
  joke: string;
  story: string;
  poem: string;
}) => {
  const { topic, joke, story, poem } = params;
  return `Here's a story, joke, and poem about ${topic}!\n\n` +
    `STORY:\n${story}\n\n` +
    `JOKE:\n${joke}\n\n` +
    `POEM:\n${poem}`;
});

// Build workflow
const workflow = entrypoint(
  "parallelWorkflow",
  async (topic: string) => {
    const [joke, story, poem] = await Promise.all([\
      callLlm1(topic),\
      callLlm2(topic),\
      callLlm3(topic),\
    ]);

    return aggregator({ topic, joke, story, poem });
  }
);

// Invoke
const stream = await workflow.stream("cats", {
  streamMode: "updates",
});

for await (const step of stream) {
  console.log(step);
}

```

## Routing

Routing classifies an input and directs it to a followup task. As noted in the Anthropic blog:

> Routing classifies an input and directs it to a specialized followup task. This workflow allows for separation of concerns, and building more specialized prompts. Without this workflow, optimizing for one kind of input can hurt performance on other inputs.
>
> When to use this workflow: Routing works well for complex tasks where there are distinct categories that are better handled separately, and where classification can be handled accurately, either by an LLM or a more traditional classification model/algorithm.

Graph API

```md-code__content
import { StateGraph, Annotation } from "@langchain/langgraph";
import { z } from "zod";

// Schema for structured output to use as routing logic
const routeSchema = z.object({
  step: z.enum(["poem", "story", "joke"]).describe(
    "The next step in the routing process"
  ),
});

// Augment the LLM with schema for structured output
const router = llm.withStructuredOutput(routeSchema);

// Graph state
const StateAnnotation = Annotation.Root({
  input: Annotation<string>,
  decision: Annotation<string>,
  output: Annotation<string>,
});

// Nodes
// Write a story
async function llmCall1(state: typeof StateAnnotation.State) {
  const result = await llm.invoke([{\
    role: "system",\
    content: "You are an expert storyteller.",\
  }, {\
    role: "user",\
    content: state.input\
  }]);
  return { output: result.content };
}

// Write a joke
async function llmCall2(state: typeof StateAnnotation.State) {
  const result = await llm.invoke([{\
    role: "system",\
    content: "You are an expert comedian.",\
  }, {\
    role: "user",\
    content: state.input\
  }]);
  return { output: result.content };
}

// Write a poem
async function llmCall3(state: typeof StateAnnotation.State) {
  const result = await llm.invoke([{\
    role: "system",\
    content: "You are an expert poet.",\
  }, {\
    role: "user",\
    content: state.input\
  }]);
  return { output: result.content };
}

async function llmCallRouter(state: typeof StateAnnotation.State) {
  // Route the input to the appropriate node
  const decision = await router.invoke([\
    {\
      role: "system",\
      content: "Route the input to story, joke, or poem based on the user's request."\
    },\
    {\
      role: "user",\
      content: state.input\
    },\
  ]);

  return { decision: decision.step };
}

// Conditional edge function to route to the appropriate node
function routeDecision(state: typeof StateAnnotation.State) {
  // Return the node name you want to visit next
  if (state.decision === "story") {
    return "llmCall1";
  } else if (state.decision === "joke") {
    return "llmCall2";
  } else if (state.decision === "poem") {
    return "llmCall3";
  }
}

// Build workflow
const routerWorkflow = new StateGraph(StateAnnotation)
  .addNode("llmCall1", llmCall1)
  .addNode("llmCall2", llmCall2)
  .addNode("llmCall3", llmCall3)
  .addNode("llmCallRouter", llmCallRouter)
  .addEdge("__start__", "llmCallRouter")
  .addConditionalEdges(
    "llmCallRouter",
    routeDecision,
    ["llmCall1", "llmCall2", "llmCall3"],
  )
  .addEdge("llmCall1", "__end__")
  .addEdge("llmCall2", "__end__")
  .addEdge("llmCall3", "__end__")
  .compile();

// Invoke
const state = await routerWorkflow.invoke({
  input: "Write me a joke about cats"
});
console.log(state.output);

```

```md-code__content
import { z } from "zod";
import { task, entrypoint } from "@langchain/langgraph";

// Schema for structured output to use as routing logic
const routeSchema = z.object({
  step: z.enum(["poem", "story", "joke"]).describe(
    "The next step in the routing process"
  ),
});

// Augment the LLM with schema for structured output
const router = llm.withStructuredOutput(routeSchema);

// Tasks
// Write a story
const llmCall1 = task("generateStory", async (input: string) => {
  const result = await llm.invoke([{\
    role: "system",\
    content: "You are an expert storyteller.",\
  }, {\
    role: "user",\
    content: input\
  }]);
  return result.content;
});

// Write a joke
const llmCall2 = task("generateJoke", async (input: string) => {
  const result = await llm.invoke([{\
    role: "system",\
    content: "You are an expert comedian.",\
  }, {\
    role: "user",\
    content: input\
  }]);
  return result.content;
});

// Write a poem
const llmCall3 = task("generatePoem", async (input: string) => {
  const result = await llm.invoke([{\
    role: "system",\
    content: "You are an expert poet.",\
  }, {\
    role: "user",\
    content: input\
  }]);
  return result.content;
});

// Route the input to the appropriate node
const llmCallRouter = task("router", async (input: string) => {
  const decision = await router.invoke([\
    {\
      role: "system",\
      content: "Route the input to story, joke, or poem based on the user's request."\
    },\
    {\
      role: "user",\
      content: input\
    },\
  ]);
  return decision.step;
});

// Build workflow
const workflow = entrypoint(
  "routerWorkflow",
  async (input: string) => {
    const nextStep = await llmCallRouter(input);

    let llmCall;
    if (nextStep === "story") {
      llmCall = llmCall1;
    } else if (nextStep === "joke") {
      llmCall = llmCall2;
    } else if (nextStep === "poem") {
      llmCall = llmCall3;
    }

    const finalResult = await llmCall(input);
    return finalResult;
  }
);

// Invoke
const stream = await workflow.stream("Write me a joke about cats", {
  streamMode: "updates",
});

for await (const step of stream) {
  console.log(step);
}

```

## Orchestrator-Worker

With orchestrator-worker, an orchestrator breaks down a task and delegates each sub-task to workers. As noted in the Anthropic blog:

> In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results.
>
> When to use this workflow: This workflow is well-suited for complex tasks where you can’t predict the subtasks needed (in coding, for example, the number of files that need to be changed and the nature of the change in each file likely depend on the task). Whereas it’s topographically similar, the key difference from parallelization is its flexibility—subtasks aren't pre-defined, but determined by the orchestrator based on the specific input.

Graph API

```md-code__content
import { z } from "zod";

// Schema for structured output to use in planning
const sectionSchema = z.object({
  name: z.string().describe("Name for this section of the report."),
  description: z.string().describe(
    "Brief overview of the main topics and concepts to be covered in this section."
  ),
});

const sectionsSchema = z.object({
  sections: z.array(sectionSchema).describe("Sections of the report."),
});

// Augment the LLM with schema for structured output
const planner = llm.withStructuredOutput(sectionsSchema);

```

```md-code__content
import { Annotation, StateGraph, Send } from "@langchain/langgraph";

// Graph state
const StateAnnotation = Annotation.Root({
  topic: Annotation<string>,
  sections: Annotation<Array<z.infer<typeof sectionSchema>>>,
  completedSections: Annotation<string[]>({
    default: () => [],
    reducer: (a, b) => a.concat(b),
  }),
  finalReport: Annotation<string>,
});

// Worker state
const WorkerStateAnnotation = Annotation.Root({
  section: Annotation<z.infer<typeof sectionSchema>>,
  completedSections: Annotation<string[]>({
    default: () => [],
    reducer: (a, b) => a.concat(b),
  }),
});

// Nodes
async function orchestrator(state: typeof StateAnnotation.State) {
  // Generate queries
  const reportSections = await planner.invoke([\
    { role: "system", content: "Generate a plan for the report." },\
    { role: "user", content: `Here is the report topic: ${state.topic}` },\
  ]);

  return { sections: reportSections.sections };
}

async function llmCall(state: typeof WorkerStateAnnotation.State) {
  // Generate section
  const section = await llm.invoke([\
    {\
      role: "system",\
      content: "Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting.",\
    },\
    {\
      role: "user",\
      content: `Here is the section name: ${state.section.name} and description: ${state.section.description}`,\
    },\
  ]);

  // Write the updated section to completed sections
  return { completedSections: [section.content] };
}

async function synthesizer(state: typeof StateAnnotation.State) {
  // List of completed sections
  const completedSections = state.completedSections;

  // Format completed section to str to use as context for final sections
  const completedReportSections = completedSections.join("\n\n---\n\n");

  return { finalReport: completedReportSections };
}

// Conditional edge function to create llm_call workers that each write a section of the report
function assignWorkers(state: typeof StateAnnotation.State) {
  // Kick off section writing in parallel via Send() API
  return state.sections.map((section) =>
    new Send("llmCall", { section })
  );
}

// Build workflow
const orchestratorWorker = new StateGraph(StateAnnotation)
  .addNode("orchestrator", orchestrator)
  .addNode("llmCall", llmCall)
  .addNode("synthesizer", synthesizer)
  .addEdge("__start__", "orchestrator")
  .addConditionalEdges(
    "orchestrator",
    assignWorkers,
    ["llmCall"]
  )
  .addEdge("llmCall", "synthesizer")
  .addEdge("synthesizer", "__end__")
  .compile();

// Invoke
const state = await orchestratorWorker.invoke({
  topic: "Create a report on LLM scaling laws"
});
console.log(state.finalReport);

```

```md-code__content
import { z } from "zod";
import { task, entrypoint } from "@langchain/langgraph";

// Schema for structured output to use in planning
const sectionSchema = z.object({
  name: z.string().describe("Name for this section of the report."),
  description: z.string().describe(
    "Brief overview of the main topics and concepts to be covered in this section."
  ),
});

const sectionsSchema = z.object({
  sections: z.array(sectionSchema).describe("Sections of the report."),
});

// Augment the LLM with schema for structured output
const planner = llm.withStructuredOutput(sectionsSchema);

// Tasks
const orchestrator = task("orchestrator", async (topic: string) => {
  // Generate queries
  const reportSections = await planner.invoke([\
    { role: "system", content: "Generate a plan for the report." },\
    { role: "user", content: `Here is the report topic: ${topic}` },\
  ]);

  return reportSections.sections;
});

const llmCall = task("sectionWriter", async (section: z.infer<typeof sectionSchema>) => {
  // Generate section
  const result = await llm.invoke([\
    {\
      role: "system",\
      content: "Write a report section.",\
    },\
    {\
      role: "user",\
      content: `Here is the section name: ${section.name} and description: ${section.description}`,\
    },\
  ]);

  return result.content;
});

const synthesizer = task("synthesizer", async (completedSections: string[]) => {
  // Synthesize full report from sections
  return completedSections.join("\n\n---\n\n");
});

// Build workflow
const workflow = entrypoint(
  "orchestratorWorker",
  async (topic: string) => {
    const sections = await orchestrator(topic);
    const completedSections = await Promise.all(
      sections.map((section) => llmCall(section))
    );
    return synthesizer(completedSections);
  }
);

// Invoke
const stream = await workflow.stream("Create a report on LLM scaling laws", {
  streamMode: "updates",
});

for await (const step of stream) {
  console.log(step);
}

```

### Original URL
https://langchain-ai.github.io/langgraphjs/tutorials/workflows/#routing
</details>

