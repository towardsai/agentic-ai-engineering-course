# LLM Agents: Tools & Structured Outputs
### Empower LLMs to act and extract data reliably

## Beyond Text: Giving LLM Agents the Power to Act

This article explores transforming Large Language Model (LLM) agents from simple text generators into capable systems that interact with the world. We will examine "Tools," the essential bridge enabling agents to perform actions. Our focus will be on the Function Calling mechanism and Gemini's Structured Outputs feature for reliable data extraction. This provides a foundational understanding for building robust AI agents.
![A grid of blue, green, and yellow squares, resembling a pixelated or mosaic pattern. The colors are soft and textured, giving the image an artistic, painted feel.](_1_research.jpeg)

Figure 1: An abstract representation of AI capabilities.

## Beyond Text: Why Agents Need Tools

Let’s be direct: Large Language Models are fundamentally just sophisticated pattern matchers. Developers train them on massive datasets of text, enabling them to generate human-like responses based on learned patterns. But on their own, they remain confined to a digital box, lacking real-world awareness or the ability to act.

This creates several hard limitations. For instance, LLMs have a knowledge cutoff, meaning they are ignorant of any events that occurred after their training data was collected. They cannot access real-time information like today’s weather or the current price of a stock. LLMs are also prone to “hallucination,” where they confidently invent facts or details that are incorrect because their goal is to generate plausible text, not to state verified truths [1](https://dev.to/ahikmah/limitations-of-large-language-models-unpacking-the-challenges-1g16), [2](https://hatchworks.com/blog/gen-ai/large-language-models-guide/).

Furthermore, LLMs cannot interact with external systems. They cannot query your company’s database, send an email, or execute a piece of code. They also struggle with highly structured data formats, such as spreadsheets or databases, and have inherent input and output length constraints, limiting their ability to process very large documents or extended conversations [1](https://dev.to/ahikmah/limitations-of-large-language-models-unpacking-the-challenges-1g16).

These limitations mean LLMs are fundamentally text-in, text-out systems. This is where "Tools" come in. Tools bridge the LLM’s text-based brain to the outside world. They act as the agent's "hands and senses," allowing it to perceive and act beyond its native limitations. A tool is simply a function or an API that the agent’s orchestrator can execute on the LLM’s behalf.

By integrating tools, you transform a passive text generator into a capable system that can perform meaningful tasks. This enables agents to fetch up-to-date information, interact with external services, and execute code, overcoming the inherent limitations of LLMs and expanding their utility in real-world applications [3](https://platform.openai.com/docs/guides/function-calling).

## The Function Calling Mechanism Explained

Function Calling is the mechanism that enables an agent to use tools. Developers describe their custom functions to an LLM, allowing the model to intelligently decide when to "call" them. The term is a bit of a misnomer; the LLM does not execute any code itself. Instead, it generates a structured JSON object that specifies which function to call and what arguments to use. Your application code is responsible for the actual execution [3](https://platform.openai.com/docs/guides/function-calling), [4](https://learn.microsoft.com/en-us/dotnet/ai/conceptual/understanding-openai-functions). This capability is a core feature in models like Gemini, enabling them to interact with the world beyond text generation.

The entire process follows a clear, cyclical flow, as Figure 2 illustrates. An input from a user goes to the LLM, which then decides to use an external tool. The application executes that tool and returns the result to the LLM, which then formulates a final output.

Figure 2: The cyclical flow of tool use in an LLM agent.

Let's break down this flow step-by-step:

1.  **Define the Tool Schema**: First, you must tell the model what tools are available. You do this by providing a schema for each function, typically in a JSON format. This schema includes the function's name, a clear description of what it does, and a definition of its parameters. The description is crucial, as it helps the LLM understand the tool's purpose and decide when to use it. For models like Gemini, you define tools using `functionDeclarations` within a `Tool` object, which mirrors the structure of other popular LLM APIs [3](https://platform.openai.com/docs/guides/function-calling), [5](https://help.openai.com/en/articles/8555517-function-calling-in-the-openai-api), [6](https://leehanchung.github.io/blogs/2024/05/09/tools-for-llms/).

    Here is an example of a function declaration for a `get_weather` tool. Notice how the descriptions for the function and its `location` parameter are clear and explicit.

    ```json
    {
      "name": "get_weather",
      "description": "Get current temperature for a given location.",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "City and country e.g. Bogotá, Colombia"
          }
        },
        "required": ["location"]
      }
    }
    ```

2.  **LLM Decides to Call a Tool**: When a user sends a prompt, you send it to the LLM along with the list of available tool schemas. The model analyzes the user's intent and, based on the tool descriptions, decides whether a tool is needed. If it is, the model generates a JSON object containing the name of the function to call and the arguments it has extracted from the user's prompt. It is important to note that the model might sometimes hallucinate arguments, so your application should include validation [3](https://platform.openai.com/docs/guides/function-calling), [4](https://learn.microsoft.com/en-us/dotnet/ai/conceptual/understanding-openai-functions).

    For a prompt like "What is the weather like in Paris today?", the model would output a function call request like this:

    ```json
    {
      "name": "get_weather",
      "arguments": {
        "location": "Paris, France"
      }
    }
    ```

3.  **Application Executes the Tool**: Your application receives this JSON output. Your code parses this request, identifies the function name (`get_weather`), and executes the corresponding function in your codebase with the provided arguments (`"Paris, France"`). This is where the actual interaction with an external API or database happens [4](https://learn.microsoft.com/en-us/dotnet/ai/conceptual/understanding-openai-functions).

    For example, your Python code might look something like this:

    ```python
    import json
    # Assuming 'types' and 'generativeai' are imported from google.generativeai
    # and 'client' is an initialized generativeai.Client()

    def get_weather(location: str) -> dict:
        """Gets current temperature for a given location."""
        # In a real application, this would call an external weather API.
        # For simplicity, we return a mock response.
        if "paris" in location.lower():
            return {"location": location, "temperature": 14, "unit": "celsius"}
        return {"location": location, "temperature": "N/A", "unit": "N/A"}

    # Simulate the LLM's function call output (what your application receives)
    # This would typically come from response.candidates[0].content.parts[0].function_call
    class MockFunctionCall:
        def __init__(self, name, args):
            self.name = name
            self.args = args

    llm_function_call_output = MockFunctionCall(
        name="get_weather",
        args={"location": "Paris, France"}
    )

    # Your application code to execute the tool
    if llm_function_call_output.name == "get_weather":
        # Extract arguments from the LLM's call
        args_from_llm = llm_function_call_output.args
        
        # Execute the corresponding Python function
        tool_execution_result = get_weather(args_from_llm["location"])
        
        print(f"Executed '{llm_function_call_output.name}' with arguments: {args_from_llm}")
        print(f"Result from tool: {tool_execution_result}")
    ```

4.  **Return the Result to the LLM**: Once your function executes, you take its return value (e.g., the weather data) and send it back to the LLM in a new message. This message tells the model the result of the tool call. The model then uses this new information to generate a final, natural language response for the user, such as "The current temperature in Paris is 14°C." This loop can continue, with the model making multiple tool calls if needed to fulfill a complex request [3](https://platform.openai.com/docs/guides/function-calling).

Function calling gives agents the power to act, but what about ensuring the data they return is reliable? For that, we need a different mechanism.

## Reliable Data Extraction with Structured Outputs

While function calling is about getting an agent to *act*, sometimes you just need it to *structure* information reliably. LLMs naturally produce unstructured text, and trying to parse it with fragile methods like regular expressions is an engineering headache. This is where **Structured Outputs** comes in. It's a critical capability for building robust AI applications that depend on predictable data.

Structured Outputs is a powerful feature, available in models like Gemini, that forces the LLM's response to conform to a specific JSON Schema you provide [7](https://platform.openai.com/docs/guides/structured-outputs). This capability goes far beyond a simple "JSON mode," which only ensures the output is valid JSON but does not guarantee it follows a particular structure or that all required fields are present. With Structured Outputs, you get predictable, machine-readable data every time, eliminating the need for retries or complex parsing logic in your application [8](https://openai.com/index/introducing-structured-outputs-in-the-api/), [9](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs).

The primary benefit of Structured Outputs is its high reliability. In internal evaluations, models using this feature have achieved perfect adherence to complex JSON schemas, a notable improvement compared to previous approaches that scored much lower. This means you can confidently define a Pydantic model in Python or a Zod schema in JavaScript, and the API ensures the LLM's output directly parses into a validated instance of that class. This eliminates the need for manual validation or error correction routines, making your data pipelines far more robust and dependable [7](https://platform.openai.com/docs/guides/structured-outputs), [8](https://openai.com/index/introducing-structured-outputs-in-the-api/).

This feature is invaluable for a wide range of tasks where precise output structure is critical. You can use it for extracting entities from unstructured text, such as names, dates, locations, or organizations. It's also perfect for converting free-form data into a format ready for direct ingestion into a database, or for populating a user interface with model-generated data. Beyond data extraction, Structured Outputs simplifies prompting and allows for explicit refusals, where safety-based model refusals are programmatically detectable, giving you more control over your application's behavior [7](https://platform.openai.com/docs/guides/structured-outputs), [9](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs).

Let's see how to use Structured Outputs to extract metadata from a document. We will start by defining our desired data structure using a Pydantic model. Pydantic is a powerful Python library that leverages type hints to define data structures, automatically generating a JSON Schema from your Python classes. This approach provides a single source of truth for your data structure, ensuring consistency between your code and the LLM's expected output.
```python
from pydantic import BaseModel, Field
from typing import List

class DocumentMetadata(BaseModel):
    """A class to hold structured metadata for a document."""
    summary: str = Field(description="A concise, 1-2 sentence summary of the document.")
    tags: List[str] = Field(description="A list of 3-5 high-level tags relevant to the document.")
    keywords: List[str] = Field(description="A list of specific keywords or concepts mentioned.")
```
When defining your schemas, whether directly in JSON or via Pydantic, adhering to best practices is crucial for optimal performance and reliability. Always use clear and intuitive names for your keys, and provide detailed descriptions for each property. For optional fields, explicitly define them using a union type with `null`, as all fields must be marked as `required` when using strict Structured Outputs. This ensures the model always provides a value, even if it's `null`. Avoid overly deep nesting in your schemas and strive for modular and clear designs with consistent naming conventions to prevent ambiguity [7](https://platform.openai.com/docs/guides/structured-outputs), [10](https://techinfotech.tech.blog/2025/06/09/best-practices-to-build-llm-tools-in-2025/), [11](https://www.beam.cloud/blog/llm-parameters).

Next, we use this Pydantic model to configure the Gemini API call. By setting the `response_schema` parameter, we instruct the model to return a JSON object that strictly matches our `DocumentMetadata` structure. This is a high-level abstraction provided by the Gemini SDK; under the hood, the API uses a `json_schema` response format with `strict: true` to enforce this adherence [8](https://openai.com/index/introducing-structured-outputs-in-the-api/).
```python
import google.generativeai as genai
from google.generativeai import types

# Assume 'client' is an initialized genai.Client()
# Assume 'document' is a string containing the text to analyze

config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=DocumentMetadata
)

prompt = f"""
Please analyze the following document and extract its metadata.

Document:
---
{document}
---
"""

response = client.models.generate_content(
    model="gemini-1.5-flash-latest",
    contents=prompt,
    config=config
)

# The response can be directly parsed into our Pydantic model
metadata = response.parsed
print(metadata.summary)
print(metadata.tags)
```
In this case, the `response.parsed` attribute automatically contains a validated `DocumentMetadata` object. This avoids manual JSON parsing and error handling, ensuring that the data you receive is always in the expected format. It's a robust way to integrate LLM outputs into your application's data flow, significantly reducing the boilerplate code typically needed for data validation.

It’s important to distinguish Structured Outputs from function calling. Function calling is about the LLM *requesting an action* from an external tool, where the output is a call to that tool. Structured Output, on the other hand, is about the LLM *providing data in a specific format* as its final answer. While these two features often work together, the key difference lies in their intent: action versus formatted information. Structured Outputs are ideal when you need the model to return data in a predictable format for further processing, without necessarily triggering external code or services [3](https://platform.openai.com/docs/guides/function-calling), [7](https://platform.openai.com/docs/guides/structured-outputs).

Even with Structured Outputs, it's essential to consider error handling. The model might not always generate a valid response that matches the schema, especially if it refuses a request for safety reasons or if a `max_tokens` limit is reached, leading to an incomplete response. Modern APIs provide mechanisms, such as a `refusal` property in the response, to programmatically detect when the model has refused to fulfill a request, allowing you to handle such edge cases gracefully. Always implement robust error handling for missing or malformed optional and nested parameters to ensure your LLM-powered functions behave predictably [7](https://platform.openai.com/docs/guides/structured-outputs), [8](https://openai.com/index/introducing-structured-outputs-in-the-api/), [11](https://www.beam.cloud/blog/llm-parameters).

Now that we understand how agents can act and structure information, let's explore the types of tools that empower them to perform a wide range of tasks.

## A Toolkit for Agents: Essential Tool Categories

An agent is only as capable as the tools it can access. While you can create any tool you want, most fall into a few essential categories that form the foundation of almost any practical agent. These tools empower agents to perceive and act upon the world beyond their inherent textual limitations.

First, consider tools for **Knowledge and Memory Access**. A common example is a Retrieval-Augmented Generation (RAG) tool. This allows an agent to query a vector database or document store to find relevant information and add it to its context. This capability is crucial for overcoming knowledge cutoffs and grounding responses in factual, up-to-date data. Similarly, database query tools enable agents to interact with structured data in SQL or NoSQL databases, allowing them to answer questions based on real-time business information [6](https://leehanchung.github.io/blogs/2024/05/09/tools-for-llms/), [12](https://sam-solutions.com/blog/llm-agent-architecture/).

Next, we have tools for **Interacting with the Outside World**. A Web Search tool is fundamental here. It allows an agent to use search engine APIs to access up-to-the-minute information from the internet. This is crucial for tasks that require current events, news, or broad general knowledge that might not reside in a private knowledge base. By integrating web search, you give your agent the ability to stay informed and respond to dynamic queries [13](https://www.truefoundry.com/blog/llm-agents).

Finally, there are tools for **Computation**. A **Code Execution** tool, typically a sandboxed Python interpreter, is one of the most powerful additions to an agent's toolkit. This tool allows the agent to perform precise mathematical calculations, manipulate data with libraries like Pandas, or run complex algorithms—tasks that LLMs are notoriously bad at on their own [6](https://leehanchung.github.io/blogs/2024/05/09/tools-for-llms/), [14](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf).

💡 When implementing a code execution tool, security is paramount. You must always run the code in a secure, sandboxed environment, such as a Docker container. This prevents the executed code from accessing or harming your host system, mitigating risks like arbitrary code execution, resource exhaustion, or unauthorized file system access [14](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf), [15](https://dida.do/blog/setting-up-a-secure-python-sandbox-for-llm-agents), [16](https://amirmalik.net/2025/03/07/code-sandboxes-for-llm-ai-agents), [17](https://huggingface.co/docs/smolagents/en/tutorials/secure_code_execution).

These tool categories are powerful, but to truly grasp how they work, it's useful to see how you could build a simple tool-use pattern from scratch. This demystifies the "magic" behind function calling APIs.

## Understanding the Core: The Tool Pattern from Scratch

Modern APIs for function calling are convenient, but they abstract away what is really happening under the hood. To truly understand tool use, building a simple version from scratch is useful. At its core, the mechanism is just a clever combination of prompt engineering and application-side parsing. It is not magic; it is just a well-defined protocol between you and the LLM.

This hands-on approach clarifies the value and convenience of using dedicated and robust function calling APIs. By implementing a basic version ourselves, we gain a deeper appreciation for the complexities these advanced APIs abstract away, highlighting why they are essential for building production-grade AI applications.

Let's demystify this by implementing a basic calculator tool manually. We will follow a pattern similar to ReAct (Reasoning and Acting), where we instruct the model to think step-by-step and declare its intent to use a tool in a specific format [18](https://www.dailydoseofds.com/ai-agents-crash-course-part-10-with-implementation/).

The process involves five conceptual steps:

1.  **Define the Tool in the Prompt**: You start by describing your available tools in the system prompt. This step is pure prompt engineering. You explicitly tell the LLM about the tool's purpose, its required inputs, and the exact format it must use to signal its intent to call the tool.
2.  **LLM "Requests" Tool Use**: When you send a user's query to the LLM, it processes this information. If it determines that a tool is necessary, it generates a specific output string or JSON object. This output is merely the LLM's *declaration* of its intent to use a tool.
3.  **Parse and Dispatch**: Your application code takes over. It continuously monitors the LLM's generated text for the predefined "tool call" format. Once detected, your code parses this structured output, extracting the tool's name and any arguments.
4.  **Execute the Tool**: With the tool name and arguments successfully extracted, your application then executes the corresponding real-world function or API call. This is the point where the agent truly "acts" on the external world.
5.  **Format and Return the Result**: Your application formats the function's return value into a simple string. It then sends this string back to the LLM in the next turn as an "Observation" or context. This action closes the loop, providing the model with the information it requested to formulate a final response.

Let's implement this with a simple Python example. We start by defining a `simple_calculator` function and a `SYSTEM_PROMPT` that tells the LLM about the tool and how to call it.
```python
import json
import re
from google import genai

# Assume 'client' is an initialized genai.Client()

def simple_calculator(expression: str) -> str:
    """A simple calculator that evaluates a mathematical expression."""
    try:
        # A simple, unsafe eval for demonstration. Use a safer alternative in production!
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

SYSTEM_PROMPT = """
You are a helpful assistant. You have access to a simple calculator tool.
To use the calculator, you must output a JSON object in the following format on a single line:
TOOL_CALL: {"tool_name": "calculator", "expression": "2 * 5"}

After you receive the result, you must provide the final answer to the user.
The result will be provided in the format:
OBSERVATION: [tool_output]
"""
```
⚠️ The use of `eval()` is for demonstration purposes only. It is not secure and should not be used in production code with untrusted input. A safer alternative like `asteval` should be used.

The `run_agent_loop` function orchestrates the interaction. It maintains the conversation history and checks the LLM's output for a tool call request. If a request is found, it executes our `simple_calculator` function and sends the result back to the LLM as an observation.
```python
def run_agent_loop(user_prompt: str):
    messages = [
        {"role": "user", "content": SYSTEM_PROMPT},
        {"role": "model", "content": "Understood. I will use the calculator tool when needed."},
        {"role": "user", "content": user_prompt}
    ]

    while True:
        # Generate a response from the LLM
        prompt_text = "\\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        response = client.models.generate_content(
            model="gemini-1.5-flash-latest",
            contents=prompt_text
        )
        llm_output = response.text.strip()
        print(f"LLM Output: {llm_output}")

        messages.append({"role": "model", "content": llm_output})

        # Check if the LLM wants to call a tool
        tool_call_match = re.search(r"TOOL_CALL: (.*)", llm_output)

        if tool_call_match:
            tool_call_json_str = tool_call_match.group(1)
            try:
                tool_call_data = json.loads(tool_call_json_str)
                if tool_call_data.get("tool_name") == "calculator":
                    expression = tool_call_data.get("expression")
                    
                    # Execute the tool
                    result = simple_calculator(expression)
                    print(f"Executing calculator with: {expression} -> Result: {result}")
                    
                    # Send the observation back to the LLM
                    observation = f"OBSERVATION: {result}"
                    messages.append({"role": "user", "content": observation})
                else:
                    print("LLM tried to call an unknown tool.")
                    break
            except json.JSONDecodeError:
                print("LLM output was not valid JSON.")
                break # Exit loop on parsing error
        else:
            # If no tool call is detected, the conversation is over
            print("\\nFinal Answer from Agent:")
            print(llm_output)
            break

# Let's run it with a question
run_agent_loop("What is 45 multiplied by 3.14?")
```
This manual implementation, while illustrative, clearly shows the engineering effort involved. Modern function calling APIs abstract away this complexity, handling the structured formatting, parsing, and iterative looping automatically. This makes building robust agentic applications significantly more convenient and less prone to errors.

## Conclusion

Tools transform LLMs from clever text predictors into capable agents that perform tasks in the real world. They are the essential link allowing a model to access up-to-date information, interact with external systems, and execute code. Understanding these core mechanisms is fundamental for any engineer building production-grade AI applications.

We have seen that Function Calling provides a structured protocol for an agent to request an action, while Structured Outputs ensures the data it provides is reliable and easily parseable. By mastering these concepts, you move beyond simply prompting a model and start engineering robust, practical systems. The future of AI applications is not just about better models; it is about building better systems around them, and tools are the cornerstone of that effort.