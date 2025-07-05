# Research

## Research Results

<details>
<summary>What real-world problems do production teams report when consuming raw, unstructured LLM text, and how do JSON/Pydantic-validated outputs resolve those reliability issues?</summary>

### Source [1]: https://docs.pydantic.dev/latest/concepts/json/

Query: What real-world problems do production teams report when consuming raw, unstructured LLM text, and how do JSON/Pydantic-validated outputs resolve those reliability issues?

Answer: Pydantic's JSON validation features are **particularly beneficial for validating LLM outputs**. The official documentation explains that Pydantic can be used to ensure that data returned from an LLM conforms to a defined schema, providing structure and reliability. For example, using `pydantic_core.from_json` in combination with `BaseModel.model_validate`, developers can parse and validate even partial JSON outputs from LLMs. This is essential because LLMs frequently produce outputs that might be incomplete, malformatted, or inconsistent with the expected schema. By passing LLM-generated JSON through Pydantic validation, production teams can catch errors and enforce data integrity before the data is used downstream. This **mitigates the risks associated with raw, unstructured LLM text**, such as runtime exceptions or logic errors due to unexpected output formats. The documentation notes that all fields should have default values for partial parsing to work reliably, indicating further attention to robustness in production scenarios[1].

-----

-----

-----

### Source [2]: https://arxiv.org/html/2404.15923v1

Query: What real-world problems do production teams report when consuming raw, unstructured LLM text, and how do JSON/Pydantic-validated outputs resolve those reliability issues?

Answer: The study highlights that **constraining LLM outputs to specific schemas is challenging but essential** for production use cases like natural language to SQL conversion. Raw, unstructured LLM output can be unpredictable, leading to integration problems and unreliable downstream processing. To address this, tools such as Guidance, Outlines, JSONFormer, and Guardrails have been developed to enable **constrained decoding**, ensuring that LLMs generate outputs that adhere to predefined structures. Additionally, semantic validation using Pydantic classes ensures both structural and semantic accuracy of outputs. This is crucial for applications like knowledge graph completion, where **precision in data parsing directly impacts the utility and reliability of the system**. By enforcing schema validation, these approaches resolve issues of unpredictability, malformed data, and hallucinations in LLM output, making the results more dependable for production pipelines[2].

-----

-----

-----

### Source [3]: https://blog.kusho.ai/from-chaos-to-order-structured-json-with-pydantic-and-instructor-in-llms/

Query: What real-world problems do production teams report when consuming raw, unstructured LLM text, and how do JSON/Pydantic-validated outputs resolve those reliability issues?

Answer: The blog post describes the **real-world struggle of relying on LLMs to produce perfectly structured JSON**: even with clear prompts, LLM-generated output can be unreliable, leading to parsing failures and integration headaches. In production, this lack of reliability becomes a significant issue, especially when downstream logic expects consistent, well-formed data. The post advocates for using **Pydantic**, a data validation tool, to convert LLM-generated JSON into structured, validated classes. This approach brings order to the inherent chaos of LLM outputs by enforcing type safety, enabling default values, and providing hooks for further checks and transformations. The use of tools like Instructor further improves reliability by automatically retrying when LLM output is malformed. **The main benefit highlighted is that Pydantic validation transforms unpredictable LLM output into application-ready, reliable data**, significantly reducing the risk of errors in production systems[3].

-----

-----

-----

### Source [4]: https://www.tamingllms.com/notebooks/structured_output.html

Query: What real-world problems do production teams report when consuming raw, unstructured LLM text, and how do JSON/Pydantic-validated outputs resolve those reliability issues?

Answer: This resource explains that **structured output modes**—such as JSON mode and Structured Output mode—are becoming increasingly common among LLM providers to address the unreliability of raw text output. Without structured output enforcement, production teams frequently encounter issues including:

- Malformed or inconsistent data structures
- Type mismatches
- Unpredictable formatting that causes downstream failures

Structured Output mode ensures that responses **always adhere to a supplied JSON Schema**, delivering several key benefits:
- **Reliable type-safety**: Eliminates the need for post-hoc validation or retries due to format errors.
- **Explicit refusals**: The model can programmatically indicate when it refuses a request, making error handling more manageable.
- **Simpler prompting**: Developers no longer need complex prompt engineering to elicit consistent output formats.

By leveraging these structured output features, production teams can **avoid the pitfalls of parsing and validating unpredictable raw text**, streamline their integration pipelines, and build more robust, reliable applications[4].
-----

-----

</details>

<details>
<summary>Which prompt-engineering techniques have experts recommended to reliably force models (that lack native function-calling) to emit strict, valid JSON that can be parsed into Python dictionaries?</summary>

### Source [5]: https://build5nines.com/how-to-write-ai-prompts-that-output-valid-json-data/

Query: Which prompt-engineering techniques have experts recommended to reliably force models (that lack native function-calling) to emit strict, valid JSON that can be parsed into Python dictionaries?

Answer: Experts recommend several prompt-engineering techniques for reliably forcing models to emit strict, valid JSON, particularly when the model lacks native function-calling:

- **Be Explicit About JSON Output**: Clearly instruct the model with direct language, such as:  
  ```
  Respond with valid JSON only. Do not include any explanation or extra text.
  ```
  This reduces the chance of conversational or extraneous output and focuses the model on structured data.

- **Tell the LLM What You Want**: Begin the prompt by specifying exactly what you expect as output (e.g., "Generate a list of 3 fictional users"). This clear instruction helps guide the model towards the desired data structure.

- **Keep Prompts Concise and Direct**: Short, unambiguous instructions reduce the risk of formatting drift or natural language contamination in the output.

- **Test and Validate Output**: After generating the response, validate that the output is strictly parseable as JSON. If models tend to insert additional text or commentary, further refine prompts to emphasize "JSON only".

These techniques are particularly useful for applications requiring downstream parsing, such as feeding model output into Python dictionaries or other structured data pipelines[1].

-----

-----

-----

### Source [6]: https://pub.aimind.so/prompts-masterclass-output-formatting-json-5-3a5c177a9095

Query: Which prompt-engineering techniques have experts recommended to reliably force models (that lack native function-calling) to emit strict, valid JSON that can be parsed into Python dictionaries?

Answer: The following prompt-engineering strategies are recommended to reliably obtain strict, valid JSON output from language models:

- **Explicitly Ask for JSON Format**: Directly instruct the model to generate a JSON object, as in:  
  > “Generate a JSON object that contains details about three different fruits, including their names, colors, and tastes.”

- **Provide a JSON Example**: Show the model the structure you expect by including a sample JSON object in your prompt. For example:  
  > “Generate a JSON object similar to {‘name’: ‘Alice’, ‘age’: 30} but for a different person.”

- **Decompose Complex Structures**: For more complex JSON outputs, break the problem into smaller sub-questions across multiple prompts, then recombine the results. For example:  
  1. First prompt: “List the names of three fruits.”
  2. Second prompt: “For each fruit, provide its color and taste.”
  3. Third prompt: “Now create a JSON object that includes the fruits’ names, colors, and tastes.”

These techniques help guide the model towards producing output that is directly parseable as JSON, minimizing the risk of errors in downstream processing[3].

-----

-----

</details>

<details>
<summary>What benchmarks or engineering write-ups compare token usage, latency, and dollar cost between (a) manual JSON prompting and post-parsing, versus (b) native structured-output modes like OpenAI Function Calling or Gemini responseSchema?</summary>

### Source [7]: https://towardsai.net/p/l/openai-json-mode-vs-functions

Query: What benchmarks or engineering write-ups compare token usage, latency, and dollar cost between (a) manual JSON prompting and post-parsing, versus (b) native structured-output modes like OpenAI Function Calling or Gemini responseSchema?

Answer: This source provides a comparative overview of OpenAI's JSON mode and Function Calling, focusing on their intended usage and reliability for structured output:

- **Function Calling** is designed for scenarios where the LLM is expected to call specified functions with arguments, requiring a particular JSON structure. However, if multiple functions are available, the model might still output free-form text instead of making a function call, even if prompted otherwise.
- **JSON Mode** is more flexible, enforcing the model to always output a valid JSON string regardless of the structure, enabling arbitrary JSON formats.
- **Hallucination Risks:** Both methods are susceptible to hallucinations. In function calling, the model might ignore instructions or fabricate argument names/values. In JSON mode, while output is always JSON, the precise format may not always be respected.
- **Token Usage, Latency, and Cost:** The article does not provide quantitative benchmarks comparing token usage, latency, or dollar costs between manual JSON prompting, function calling, or JSON mode. It focuses more on qualitative and reliability considerations.

-----

-----

-----

### Source [9]: https://www.boundaryml.com/blog/schema-aligned-parsing

Query: What benchmarks or engineering write-ups compare token usage, latency, and dollar cost between (a) manual JSON prompting and post-parsing, versus (b) native structured-output modes like OpenAI Function Calling or Gemini responseSchema?

Answer: This blog post analyzes different approaches for extracting structured data from LLMs, including prompting, JSON mode, function calling, and schema-constrained output:

- **Function Calling and JSON Mode:** Both can suffer from schema inaccuracies. For example, a model may output valid JSON that does not match the intended type or schema.
- **Token Usage:** The post notes that relying on a JSON schema can be "incredibly wasteful in the token space," implying that function calling and schema constraints may increase token usage relative to simple prompting.
- **Accuracy:** The author observes that models supporting function calling may have "degraded accuracy with function calling when compared to just prompting based techniques."
- **Model Support:** Not all models support function calling or structured output natively, and those that do may have varying performance.
- The article does not provide explicit quantitative benchmarks regarding latency or cost but highlights qualitative trade-offs in accuracy and token efficiency.

-----

-----

-----

### Source [10]: https://www.vellum.ai/blog/when-should-i-use-function-calling-structured-outputs-or-json-mode

Query: What benchmarks or engineering write-ups compare token usage, latency, and dollar cost between (a) manual JSON prompting and post-parsing, versus (b) native structured-output modes like OpenAI Function Calling or Gemini responseSchema?

Answer: This source provides a guide on when to use function calling, structured outputs, or JSON mode with LLMs:

- **Function Calling** is ideal for applications where you want the model to call external APIs, select among multiple tools, or require ongoing interaction to gather function parameters.
- **Structured Outputs with response_format** are recommended when no further interaction is required, such as the final step in a pipeline.
- **Reliability:** The guidance is to prefer structured outputs over JSON mode for reliability, as JSON mode can be less dependable.
- The article does not present benchmarks or direct comparisons of token usage, latency, or cost. The recommendations are based on functional requirements and output reliability rather than quantitative engineering metrics.

-----

-----

</details>

<details>
<summary>When developers choose YAML or XML instead of JSON for LLM structured output, what advantages (e.g., fewer tokens) and disadvantages (e.g., parser availability, verbosity) are reported?</summary>

### Source [11]: https://betterprogramming.pub/yaml-vs-json-which-is-more-efficient-for-language-models-5bc11dd0f6df

Query: When developers choose YAML or XML instead of JSON for LLM structured output, what advantages (e.g., fewer tokens) and disadvantages (e.g., parser availability, verbosity) are reported?

Answer: This source highlights that **YAML is significantly more cost- and time-efficient than JSON** when used as an output format for language models. YAML allows language models to produce the same content **faster and more cheaply** compared to JSON. The article suggests that it may be more efficient to request output in YAML and then convert the result to JSON in code, rather than requesting JSON directly. However, it also notes that **JSON is generally faster to parse and consume** than YAML. One potential disadvantage of JSON is its strictness with data formats, such as numbers being output as strings, but this can be mitigated by enforcing type conversions on the code side. The source does not discuss XML in detail for language model output, focusing mainly on YAML and JSON.

-----

-----

-----

### Source [12]: https://celerdata.com/glossary/yaml-json-and-xml-a-practical-guide-to-choosing-the-right-format

Query: When developers choose YAML or XML instead of JSON for LLM structured output, what advantages (e.g., fewer tokens) and disadvantages (e.g., parser availability, verbosity) are reported?

Answer: This guide compares **readability, performance, use cases, and community support** for YAML, JSON, and XML:

- **Readability:** YAML is the most human-readable, favoring indentation-based structure and inline comments. JSON is compact and consistent but less user-friendly for manual editing. XML is the most verbose, with explicit tags that can clutter readability, especially in deeply nested documents.
- **Performance:** JSON is the **fastest to parse and generate**, making it suitable for real-time APIs and bandwidth-sensitive applications. YAML is **slower to parse** due to its more complex and flexible syntax, and ambiguities in scalar interpretation can add parser overhead. XML is the most resource-intensive, especially when using namespaces and validation schemas.
- **Parser availability:** JSON has the **widest native support** across programming languages and platforms. YAML’s support is good but not as universal or mature as JSON. XML has extensive legacy support but is considered heavy for new applications.
- **Verbosity:** XML is the **most verbose** of the three. JSON is more compact, while YAML tends to be concise for configuration and similar uses.

-----

-----

-----

### Source [13]: https://wiserli.com/blogs/optimizing-generation-process-with-large-language-models-for-specific-formats/

Query: When developers choose YAML or XML instead of JSON for LLM structured output, what advantages (e.g., fewer tokens) and disadvantages (e.g., parser availability, verbosity) are reported?

Answer: This article finds that **YAML’s human-readable format is often less verbose than JSON**, resulting in more concise prompts and faster processing for language models. YAML’s simplicity can lead to **quicker comprehension by LLMs**, reducing generation time compared to complex JSON structures. In practical terms, using YAML can reduce the number of tokens generated by the model by **approximately 50%**, which improves response time and cost efficiency. The source recommends evaluating the needs of the application to balance readability and computational efficiency, and suggests that strategically incorporating YAML alongside JSON can lead to a more cost-effective and faster response generation process. XML is not discussed in detail for this particular use case.

-----

-----

-----

### Source [14]: https://arxiv.org/html/2407.15734v1

Query: When developers choose YAML or XML instead of JSON for LLM structured output, what advantages (e.g., fewer tokens) and disadvantages (e.g., parser availability, verbosity) are reported?

Answer: This academic paper discusses the **conciseness of JSON** as an output format for LLMs. It reports that, for similar input prompts, **JSON output is much less verbose than free text**. This is attributed to the concise nature of JSON, where field values are closely linked to their keys, resulting in efficient and predictable structures. The paper also introduces "StrictJSON," an even more concise variant, which can represent the same schema in nearly half the tokens required by regular JSON schema definitions. The implication is that **reducing verbosity (and thus token count) improves LLM performance**, as longer context lengths typically degrade task performance. XML is not specifically compared, but by context, it would be more verbose and less efficient for LLM output.

-----

-----

-----

### Source [15]: https://aws.amazon.com/compare/the-difference-between-yaml-and-json/

Query: When developers choose YAML or XML instead of JSON for LLM structured output, what advantages (e.g., fewer tokens) and disadvantages (e.g., parser availability, verbosity) are reported?

Answer: This AWS comparison notes that **YAML is easier to read, comprehend, and write than JSON**, especially for non-developers, due to its indented syntax. JSON, however, is **more widely supported** and is the standard for data objects in most APIs. Regarding data types, JSON supports a fixed set (number, boolean, null, string, array, object), while YAML supports all native data types of dynamic programming languages, making it more flexible for complex data. The source does not provide explicit details about token count or performance in LLM contexts, but it supports the general observation that **YAML is more user-friendly but JSON is more universally supported and easier to parse programmatically**. XML is mentioned as being more verbose and less common in modern APIs.

-----

-----

</details>

<details>
<summary>Are there documented case studies or technical blogs where Pydantic models act as a validation/bridge layer between LLMs and downstream Python code, including how validation errors are handled and retried?</summary>

### Source [16]: https://www.arecadata.com/pydanticai-for-building-agentic-ai-based-llm-applications/

Query: Are there documented case studies or technical blogs where Pydantic models act as a validation/bridge layer between LLMs and downstream Python code, including how validation errors are handled and retried?

Answer: PydanticAI is presented as a framework that leverages **Pydantic models as a type-safe validation and bridge layer between LLMs and downstream Python code**. It highlights real-world applications, such as a bank's customer support agent built on OpenAI's GPT-4o, where Pydantic's dependency injection integrates live customer data with AI-driven logic. The framework emphasizes:

- **Type-Safe Validation:** Ensures LLM outputs are structured and predictable through strict Pydantic integration, preventing malformed data from entering downstream systems.
- **Dependency Injection:** Allows seamless integration with external data and services, further enforcing validation at the integration point.
- **Model-agnostic Design:** Supports multiple LLM providers, making Pydantic the consistent validation and bridging layer regardless of the LLM used.

While the source illustrates PydanticAI’s centrality in validation and integration, it does not detail specific patterns for handling or retrying validation errors, nor does it provide step-by-step error handling workflows.

-----

-----

-----

### Source [17]: https://xebia.com/blog/enforce-and-validate-llm-output-with-pydantic/

Query: Are there documented case studies or technical blogs where Pydantic models act as a validation/bridge layer between LLMs and downstream Python code, including how validation errors are handled and retried?

Answer: This technical blog explicitly demonstrates **using Pydantic models as a validation layer for LLM outputs**. The workflow includes:

- **Validating LLM Output:** After receiving a response from an LLM, `model_validate_json()` is called on the Pydantic model. If the LLM's output doesn't conform to the schema, a `ValidationError` is raised.
- **Error Handling Example:**
  ```python
  from pydantic import ValidationError

  try:
      validated_response = ThoughtAnswerResponse.model_validate_json(raw_response)
  except ValidationError as e:
      print("Unable to validate LLM response.")
      # Add your own error handling here
      raise e
  ```
- **Enforcing Specific Values:** By using `Literal` type hints in Pydantic models, developers can ensure that fields (e.g., a `"difficulty"` field) only accept specific allowable values. If the LLM response uses an invalid value, validation fails.
- **Manual Intervention:** The blog suggests that, upon validation failure, custom error handling or retries (e.g., re-prompting the LLM) can be implemented, but leaves the retry logic to the developer.

This source provides direct examples of Pydantic as a validation bridge, along with a pattern for catching and responding to validation errors, though it does not provide an end-to-end automated retry mechanism.

-----

-----

-----

### Source [18]: https://www.leocon.dev/blog/2024/11/from-chaos-to-control-mastering-llm-outputs-with-langchain-and-pydantic/

Query: Are there documented case studies or technical blogs where Pydantic models act as a validation/bridge layer between LLMs and downstream Python code, including how validation errors are handled and retried?

Answer: This guide shows **how to combine LangChain and Pydantic to enforce type safety and automatic validation on LLM outputs**, transforming unpredictable responses into structured Python objects. The article describes:

- **Extracting and Validating Fields:** Developers define Pydantic models representing the desired schema for LLM outputs. Responses are parsed and validated against these models.
- **Handling Validation Failures:** If the LLM output does not match the expected schema, validation errors are thrown, preventing malformed data from propagating to downstream code.
- **Integration Pattern:** The process involves sending a prompt to the LLM, receiving the raw output, parsing it, and validating it with Pydantic. Only valid, structured data is then passed further into the application.

Although the article emphasizes validation and structured integration, it does not detail retry strategies or elaborate error handling beyond initial validation failure.

-----

-----

-----

### Source [19]: https://pydantic.dev/articles/llm-validation

Query: Are there documented case studies or technical blogs where Pydantic models act as a validation/bridge layer between LLMs and downstream Python code, including how validation errors are handled and retried?

Answer: This official Pydantic article discusses **using Pydantic validators—including LLM-powered custom validators—to enforce rules on LLM outputs**. Key points include:

- **Natural Language Validation Rules:** The `llm_validator` can encode complex or subjective rules (e.g., "don't say objectionable things") and apply them during Pydantic model validation.
- **Validation Error Reporting:** If the LLM output violates the validation rule, the Pydantic model raises a validation error with a detailed message, including a link for additional error context.
- **Structured Validation Workflow:** The example demonstrates how, upon validation failure, errors are surfaced in a structured manner, making it clear to downstream code what failed and why.

While this source shows advanced validation and error reporting, it stops short of providing a full pattern for automated retries or error correction after validation failures.

-----

-----

</details>

<details>
<summary>What published benchmarks or engineering case studies quantify token usage, latency and dollar cost when using native structured-output modes (OpenAI Function Calling, Gemini responseSchema, etc.) versus manual JSON prompting with post-parsing?</summary>

### Source [20]: https://towardsai.net/p/l/openai-json-mode-vs-functions

Query: What published benchmarks or engineering case studies quantify token usage, latency and dollar cost when using native structured-output modes (OpenAI Function Calling, Gemini responseSchema, etc.) versus manual JSON prompting with post-parsing?

Answer: This article compares **OpenAI's JSON mode and Function Calling** for obtaining structured outputs from GPT models. It highlights that **Function Calling** works with specific JSON structures to call specified functions, but the model may not always adhere strictly, sometimes producing free-form text or hallucinating argument names and values. **JSON mode**, in contrast, enforces the model to always output a valid JSON string, though it may not always respect the specified format. 

There is no explicit benchmarking data (latency, token usage, or dollar cost) presented, but the article notes:
- Both approaches can lead to hallucinations or schema inaccuracies.
- JSON mode is more flexible for arbitrary structures.
- Function Calling is limited to the schemas you provide and may not always be strictly followed.

No quantified performance or cost comparisons are given; the discussion is qualitative, focusing on workflow optimization and the behavior differences between the methods.

-----

-----

-----

### Source [21]: https://www.boundaryml.com/blog/schema-aligned-parsing

Query: What published benchmarks or engineering case studies quantify token usage, latency and dollar cost when using native structured-output modes (OpenAI Function Calling, Gemini responseSchema, etc.) versus manual JSON prompting with post-parsing?

Answer: This source discusses **prompting, JSON mode, and Function Calling** for extracting structured data from LLMs. It notes that:
- Most APIs rely on JSON schemas, which can be "incredibly wasteful in the token space."
- **Function Calling** suffers from the same schema inaccuracy as JSON mode. For example, even valid JSON can fail to match complex schema constraints (like expecting an array of integers).
- Many models still do not support these features, and those that do often have "degraded accuracy with function calling when compared to just prompting based techniques."

The article does not provide published benchmarks or dollar cost figures but mentions **token inefficiency** and **accuracy degradation** when using Function Calling compared to manual prompting. The commentary is qualitative and highlights ongoing industry challenges rather than offering measured case studies.

-----

-----

-----

### Source [22]: https://www.vellum.ai/blog/when-should-i-use-function-calling-structured-outputs-or-json-mode

Query: What published benchmarks or engineering case studies quantify token usage, latency and dollar cost when using native structured-output modes (OpenAI Function Calling, Gemini responseSchema, etc.) versus manual JSON prompting with post-parsing?

Answer: This blog post explains when to use **Function Calling, Structured Outputs, or JSON Mode**. It recommends using Structured Outputs for reliable outputs and suggests Function Calling when:
- Making requests to external APIs.
- Letting the model decide between multiple tools.
- Supporting ongoing Assistant-User interactions.

It recommends **response_format** (structured outputs) when no interaction is needed, usually as a final step.

No quantitative benchmarks (token usage, latency, or dollar cost) are provided. The guidance is scenario-based rather than measurement-based.

-----

-----

</details>

<details>
<summary>What prompt-engineering techniques have experts recommended to reliably force models without JSON mode to emit strict, valid JSON—such as wrapping content in XML/JSON tags, suppressing prose, or few-shot schema examples?</summary>

### Source [25]: https://build5nines.com/how-to-write-ai-prompts-that-output-valid-json-data/

Query: What prompt-engineering techniques have experts recommended to reliably force models without JSON mode to emit strict, valid JSON—such as wrapping content in XML/JSON tags, suppressing prose, or few-shot schema examples?

Answer: This article provides several **prompt-engineering strategies** to reliably force AI models to emit valid JSON, especially when JSON mode is not available:

- **Be Explicit About JSON Output**: Clearly instruct the model to "respond with valid JSON only" and to "not include any explanation or extra text." This directness helps minimize the model's tendency to add conversational or explanatory prose, which can corrupt the strict JSON format.
- **Short, Direct Prompts**: Keeping prompts concise reduces the risk of the model drifting into natural language formatting. Overly complex or wordy prompts can cause the model to revert to prose or mix natural language with JSON.
- **Clearly Specify the Output Structure**: Tell the model exactly what you want (e.g., "Generate a list of 3 fictional users"). This sets a clear expectation for the content and structure of the JSON object.
- **Iterate and Test**: The article emphasizes that even with clear prompts, validation and iteration are important because models may still occasionally output invalid JSON. Testing with different prompts and refining the instructions can improve reliability.

Overall, the article highlights the importance of **clarity, brevity, and explicit instructions** to maximize the likelihood of receiving valid, clean JSON output from LLMs that are not in JSON mode[1].

-----

-----

-----

### Source [26]: https://pub.aimind.so/prompts-masterclass-output-formatting-json-5-3a5c177a9095

Query: What prompt-engineering techniques have experts recommended to reliably force models without JSON mode to emit strict, valid JSON—such as wrapping content in XML/JSON tags, suppressing prose, or few-shot schema examples?

Answer: This source outlines several expert-recommended techniques for enforcing strict JSON output:

- **Explicitly Request JSON Format**: Directly ask for JSON output in the prompt, such as: “Generate a JSON object that contains details about three different fruits, including their names, colors, and tastes.”
- **Provide a JSON Example**: Show the model an explicit example of the desired JSON structure (few-shot prompting). For example: “Generate a JSON object similar to {'name': 'Alice', 'age': 30} but for a different person.” This guides the model toward strict adherence to the schema and formatting.
- **Break Down Complex JSON Generation into Steps**: For complicated outputs, use multi-step prompting. First, request a list of items; then, for each, ask for detailed attributes; finally, combine results into a JSON object. This staged approach can yield higher-quality, more reliably valid JSON, especially for complex schemas.

These techniques—**explicit requests, schema examples, and multi-step prompts**—are highlighted as effective for reliably obtaining valid JSON from models that do not support JSON mode[3].

-----

-----

-----

### Source [27]: https://community.openai.com/t/gpt3-5-json-output-format/712044

Query: What prompt-engineering techniques have experts recommended to reliably force models without JSON mode to emit strict, valid JSON—such as wrapping content in XML/JSON tags, suppressing prose, or few-shot schema examples?

Answer: In this official OpenAI developer community discussion, several **practical recommendations** are provided for getting consistent JSON output from models like GPT-3.5, which can be prone to including unwanted prose:

- **Include an Output Example**: Incorporate an explicit example of the desired JSON format directly in your prompt. This visual guidance helps the model understand and replicate the required structure, which often leads to more consistent and valid JSON responses.
- **Use Clear Output Instructions**: The documentation and community posts stress that you must instruct the model to output JSON explicitly, even if a system-level or user-level message is used. Without clear instruction, the model may mix prose with JSON or otherwise not strictly adhere to the format.
- **Validate Model Output**: If formatting problems persist, the community suggests implementing a **response validation layer** in your application, which checks the model output against your schema or requirements before accepting it for downstream use.

Overall, the source reinforces the importance of **explicit prompt instructions, schema examples, and output validation** when working with models that do not natively enforce strict JSON formatting[4].

-----

-----

-----

### Source [28]: https://community.openai.com/t/prompt-integrating-json-or-json-request-after-the-prompt/233567

Query: What prompt-engineering techniques have experts recommended to reliably force models without JSON mode to emit strict, valid JSON—such as wrapping content in XML/JSON tags, suppressing prose, or few-shot schema examples?

Answer: This OpenAI community thread discusses **prompt design approaches** for obtaining reliable JSON output:

- **Direct Integration of JSON Request in Prompt**: Option 2—writing a prompt that directly asks for JSON formatting—is described as the most reliable and recommended method. Testing prompts with the JSON request included from the outset is emphasized, as adding the JSON request later may change the model’s response format.
- **Explicitly Plan Response Format**: If you generate a response first and then request JSON (Option 3), you must still be very precise in specifying the expected format, or the model may produce inconsistent results.
- **Structured Prompts Reduce Errors**: The discussion notes that highly structured, explicit prompts that detail the desired format (including field names, types, and structure) tend to yield more reliable, valid JSON—compared to less constrained or ambiguous prompts.

The consensus from the discussion is that **explicitly requesting JSON and structuring prompts for clarity** is the most effective way to consistently obtain valid JSON from models not in JSON mode[5].

-----

-----

</details>

<details>
<summary>How do developers integrate Pydantic models directly with Gemini’s responseSchema or OpenAI Function Calling, including patterns for validation errors and automatic retries?</summary>

### Source [29]: https://discuss.ai.google.dev/t/response-schema-from-pydantic/50028

Query: How do developers integrate Pydantic models directly with Gemini’s responseSchema or OpenAI Function Calling, including patterns for validation errors and automatic retries?

Answer: A user asks about how to populate Gemini's `response_schema` using a Pydantic model. The user notes that with the OpenAI API, it's possible to pass a Pydantic model directly for structured output. They are looking for an equivalent approach for Gemini, specifically wanting to avoid manually converting the Pydantic model to a schema. The thread does not provide a concrete answer, but it highlights the desire for seamless integration between Pydantic models and Gemini’s `response_schema`, similar to what’s possible with OpenAI’s function calling. The question reflects a common developer need: **automated translation of Pydantic models to response schemas** for LLM API calls.

-----

-----

-----

### Source [30]: https://community.openai.com/t/pydantic-model-responses-api/1147202

Query: How do developers integrate Pydantic models directly with Gemini’s responseSchema or OpenAI Function Calling, including patterns for validation errors and automatic retries?

Answer: A user shares an example of integrating Pydantic models with OpenAI's Responses API. The pattern involves defining a Pydantic model (e.g., `MathReasoning`) and passing it directly to the client’s `responses.parse()` method as the `text` argument. The user also experiments with using `MathReasoning.model_json_schema()`. This demonstrates that **OpenAI's Python SDK can accept Pydantic models directly for structured outputs**, and it will handle the translation to JSON Schema under the hood. However, the user’s post is a request for clarification, not an official answer, and does not cover error handling or retries.

-----

-----

-----

### Source [31]: https://ai.google.dev/gemini-api/docs/structured-output

Query: How do developers integrate Pydantic models directly with Gemini’s responseSchema or OpenAI Function Calling, including patterns for validation errors and automatic retries?

Answer: The Gemini API documentation states: **when using a Pydantic type in Python, the library automatically converts the model to a JSON schema and sends it to the API**. Supported types include primitives, lists, dicts, unions, and user-defined Pydantic models, including nested structures. This means developers do not need to work with schema objects directly; instead, they can simply pass Pydantic models to define the response structure. Additionally, Gemini 2.5 supports JSON Schema directly via the `responseJsonSchema` field, but with some limitations (such as restrictions on recursive references and certain properties). This provides **out-of-the-box integration for structured output and schema translation**, but the documentation does not discuss patterns for validation error handling or automatic retries.

-----

-----

-----

### Source [32]: https://python.useinstructor.com

Query: How do developers integrate Pydantic models directly with Gemini’s responseSchema or OpenAI Function Calling, including patterns for validation errors and automatic retries?

Answer: The Instructor library, built on Pydantic, offers an advanced approach for **type-safe data extraction, validation, and automatic retries** with LLMs, including OpenAI models. Developers can define Pydantic models as usual. When parsing responses, Instructor validates outputs against the model; if validation fails (for example, if the model outputs objectionable content or the wrong type), it can **automatically retry the LLM call**. Example usage shows catching `ValidationError` exceptions to handle output failures. Instructor also allows the use of custom validators (e.g., `llm_validator`) to enforce more complex constraints and integrates seamlessly with OpenAI’s API by patching the client. This pattern enables robust error handling and automated retries, **abstracting the validation/retry logic** from the developer.

-----

-----

-----

### Source [33]: https://ai.pydantic.dev/models/

Query: How do developers integrate Pydantic models directly with Gemini’s responseSchema or OpenAI Function Calling, including patterns for validation errors and automatic retries?

Answer: PydanticAI is a framework that supports multiple AI providers, including OpenAI and Gemini (via compatible APIs). When an agent is instantiated with a provider and model name (for example, `openai:gpt-4o` or `openrouter:google/gemini-2.5-pro-preview`), **PydanticAI automatically selects the correct model class, provider, and profile**, handling the construction of requests and schema transformations for each backend. This means that developers can use the same Pydantic models and have the framework adapt the schema to the requirements of either OpenAI or Gemini. The framework abstracts **schema translation and API compatibility**, but this source does not detail validation error handling or retry patterns.

-----

-----

</details>

<details>
<summary>In production systems, how are structured LLM outputs used for entity extraction to populate knowledge graphs or enable GraphRAG pipelines, and what benefits are reported?</summary>

### Source [34]: https://arxiv.org/html/2402.04437v3

Query: In production systems, how are structured LLM outputs used for entity extraction to populate knowledge graphs or enable GraphRAG pipelines, and what benefits are reported?

Answer: This source introduces a **multi-stage structured entity extraction method using large language models (LLMs)**, specifically through a model called MuSEE. The process is designed to enhance both effectiveness and efficiency for entity extraction tasks, which are critical for populating knowledge graphs.

Key aspects of the approach include:

- **Decomposition into Sub-Tasks:** The extraction task is split into several stages, allowing the model to focus on one sub-task at a time. All predictions in a stage can be processed in parallel, which improves efficiency.
- **Token Reduction via Special Tokens:** The model translates entity types and property keys into unique special tokens (e.g., “ent_type_” and “pk_”), condensing multi-token sequences into single tokens. This reduces the number of output tokens and speeds up generation.
- **Schema Awareness:** All special tokens are derived from a predefined schema available before model training, ensuring that outputs are structured and consistent.
- **Pipeline Benefits:** By producing outputs in a structured format aligned with a schema, the system can directly map extracted entities and their properties to knowledge graph nodes and edges without extensive post-processing.

The reported benefits include **significant improvements in efficiency** (due to reduced token output), **better alignment with downstream applications** like knowledge graph population, and **scalability for production systems** requiring high-throughput, structured data extraction.

-----

-----

-----

### Source [35]: https://ragaboutit.com/building-a-graph-rag-system-with-llm-router-a-comprehensive-coding-walkthrough/

Query: In production systems, how are structured LLM outputs used for entity extraction to populate knowledge graphs or enable GraphRAG pipelines, and what benefits are reported?

Answer: This source describes the **Graph RAG (Retrieval-Augmented Generation with Graphs) system**, which combines natural language processing with knowledge graph technology.

The workflow involves:

- **Data Ingestion and Structuring:** Information from various sources is converted into a structured format suitable for graph construction. This structured representation is essential for subsequent retrieval and generation tasks.
- **Graph-Based Retrieval:** By representing data as a graph, the system enables the discovery of relationships and semantic concepts that traditional retrieval methods may miss.
- **Query Understanding:** Graph RAG systems excel at answering complex queries by leveraging the connections and context provided by the graph structure.

The benefits reported include:

- **Enhanced context-aware generation:** Structured graph representation allows LLMs to deliver more accurate and insightful responses.
- **Sophisticated query handling:** The system can address complex, relationship-rich queries by tapping into the interconnected knowledge base.
- **Integration foundation:** The graph forms the backbone for further applications, such as dynamic knowledge graph augmentation and advanced retrieval pipelines.

-----

-----

-----

### Source [36]: https://learn.microsoft.com/en-us/azure/developer/ai/how-to/extract-entities-using-structured-outputs

Query: In production systems, how are structured LLM outputs used for entity extraction to populate knowledge graphs or enable GraphRAG pipelines, and what benefits are reported?

Answer: This official documentation explains how **structured outputs from Azure OpenAI models** are used for entity extraction and integration into downstream systems like knowledge graphs.

Key points:

- **Predetermined Schema Enforcement:** The model’s output conforms to a predefined JSON schema, ensuring consistency and reducing errors.
- **Seamless Integration:** Structured outputs can be directly mapped to entities and relationships in knowledge graphs, minimizing or eliminating the need for post-processing.
- **Efficiency:** By producing structured data natively, the system optimizes token usage and accelerates data ingestion.
- **Workflow Flexibility:** The structured output mode is suitable for extracting entities from many file types, provided they can be converted to text or images, supporting broad applicability in production environments.
- **Use Cases:** Structured outputs facilitate function calling, structured data extraction, and multi-step workflows, which are essential for complex knowledge graph or GraphRAG pipelines.

Reported benefits include **improved data quality**, **reduced integration effort**, and **higher system reliability**.

-----

-----

-----

### Source [37]: https://arxiv.org/html/2502.10996v2

Query: In production systems, how are structured LLM outputs used for entity extraction to populate knowledge graphs or enable GraphRAG pipelines, and what benefits are reported?

Answer: This research discusses the integration of **structured graphs as context for LLMs** to improve reasoning and factual accuracy in knowledge-intensive tasks.

Highlights include:

- **Explicit Relational Structure:** Graphs provide a clear, relational context for entities, making multi-hop reasoning and entity connections tractable for LLMs.
- **GraphRAG-style Methods:** These methods construct corpus-level graphs and use them to generate summaries for enhanced retrieval, but traditional approaches often rely on static, global graphs with potential issues of redundancy and irrelevance.
- **Dynamic, Query-Specific Graph Construction:** The paper proposes a retrieval-and-structuring framework that builds **query-specific knowledge graphs in real-time** during inference. This approach tailors the graph to the specific needs of each query, improving relevance and reducing computational overhead.
- **Interpretability and Reliability:** Explicitly structured graphs help mitigate reasoning failures common in LLMs by making reasoning chains visible and traceable.

The key benefits reported are **improved interpretability, adaptability to specific queries, and enhanced reasoning capabilities** for downstream applications like GraphRAG pipelines.

-----

-----

</details>

<details>
<summary>What security or reliability analyses demonstrate that schema-validated outputs (e.g., via Pydantic) mitigate risks such as prompt injection, malformed data, or downstream crashes?</summary>

### Source [38]: https://docs.pydantic.dev/1.10/usage/models/

Query: What security or reliability analyses demonstrate that schema-validated outputs (e.g., via Pydantic) mitigate risks such as prompt injection, malformed data, or downstream crashes?

Answer: Pydantic models provide strong data validation by guaranteeing that fields of a model instance conform to their specified types and constraints after parsing and validation. When untrusted or malformed data is passed to a Pydantic model, the library will raise a `ValidationError` if the input does not meet the schema's requirements. This error contains detailed information about all validation failures, including their locations and human-readable messages. By enforcing these constraints at the data ingestion stage, Pydantic helps prevent downstream crashes and application errors caused by malformed or unexpected input. The validation mechanism ensures that only well-formed data is accepted for further processing, which mitigates risks such as application instability and logic errors due to incorrect data types or formats. However, while schema validation addresses malformed data and reliability concerns, the documentation does not specifically discuss prompt injection or other application-layer security threats beyond data conformance[3].

-----

-----

-----

### Source [39]: https://betterstack.com/community/guides/scaling-python/pydantic-explained/

Query: What security or reliability analyses demonstrate that schema-validated outputs (e.g., via Pydantic) mitigate risks such as prompt injection, malformed data, or downstream crashes?

Answer: Pydantic is designed to enforce strict data types, constraints, and serialization rules for Python objects, which enhances both security and reliability. The article provides an example where a password field is validated using a custom validator to ensure it meets complexity requirements (uppercase, lowercase, number). If invalid input is supplied, Pydantic raises a validation error with a descriptive message, preventing the application from accepting weak or malformed data. This mechanism ensures that only properly validated and secure data enters the system, thereby reducing risks such as security vulnerabilities and application errors. In the context of general data validation and reliability, this approach helps protect against malformed data and the associated risks of downstream crashes or logic flaws. However, the article does not specifically address prompt injection or provide formal security analysis related to that threat class[4].

-----

-----

-----

### Source [40]: https://dev.to/devasservice/best-practices-for-using-pydantic-in-python-2021

Query: What security or reliability analyses demonstrate that schema-validated outputs (e.g., via Pydantic) mitigate risks such as prompt injection, malformed data, or downstream crashes?

Answer: The guide outlines best practices for using Pydantic, emphasizing the importance of correct type annotations, providing default values for optional fields, and avoiding overcomplicated or deeply nested models. It explains that incorrect type annotations or missing defaults can lead to validation errors and unexpected behavior, which could potentially manifest as application crashes or logic errors down the line. By following best practices—such as leveraging Pydantic’s configuration options for when validation occurs and documenting complex validation logic—developers can ensure more reliable and maintainable data validation. While the guide focuses on avoiding reliability pitfalls and maximizing maintainability, it does not specifically discuss mitigation of prompt injection or formal security analysis, but stresses that proper schema validation helps prevent malformed data from propagating through the system[1].

-----

-----

</details>

<details>
<summary>What engineering write-ups or case studies provide hard numbers (token count, latency, and dollar cost) comparing manual JSON prompting plus post-parsing with native structured-output modes like OpenAI Function Calling or Gemini responseSchema?</summary>

### Source [41]: https://discuss.ai.google.dev/t/schema-used-in-functioncalling-and-responseschema-diverges/69272

Query: What engineering write-ups or case studies provide hard numbers (token count, latency, and dollar cost) comparing manual JSON prompting plus post-parsing with native structured-output modes like OpenAI Function Calling or Gemini responseSchema?

Answer: This source discusses technical differences between **Google Gemini's Function Calling** and **Structured Output** (via `response_schema`) modes, particularly how their schema handling diverges. It highlights that the `function_declarations` in Function Calling and `response_schema` in Structured Output use different subsets of the OpenAPI 3.0 schema. There is an example of a JSON schema and discussion about serialization and compatibility with OpenAPI specifications. However, this source does **not provide any hard numbers** on token count, latency, or dollar cost when comparing manual JSON prompting and post-parsing with native structured output modes.

-----

-----

-----

### Source [42]: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling

Query: What engineering write-ups or case studies provide hard numbers (token count, latency, and dollar cost) comparing manual JSON prompting plus post-parsing with native structured-output modes like OpenAI Function Calling or Gemini responseSchema?

Answer: This official Google documentation provides a **guide to implementing function calling** with Gemini on Vertex AI. It describes how to submit prompts and function declarations in OpenAPI-compatible schema, and gives code examples for making API calls. The focus is on **how to structure prompts and schema** for the function calling feature, but it does **not include any benchmarking data** such as token count, latency, or dollar cost comparisons between manual JSON prompting and native structured output modes.

-----

-----

-----

### Source [43]: https://addepto.com/blog/google-gemini-api-vs-open-ai-api-main-differences/

Query: What engineering write-ups or case studies provide hard numbers (token count, latency, and dollar cost) comparing manual JSON prompting plus post-parsing with native structured-output modes like OpenAI Function Calling or Gemini responseSchema?

Answer: This article provides an in-depth comparison between the **Google Gemini API** and **OpenAI API**, covering various operational aspects and use cases. It discusses differences in functionality, schema handling, and some qualitative aspects of performance. However, while it offers a comprehensive overview, it does **not present specific engineering benchmarks**—such as **token count, latency, or dollar cost figures**—for manual JSON prompting plus post-parsing versus native structured-output modes like OpenAI Function Calling or Gemini responseSchema.

-----

**No sources provided in the search results contain engineering write-ups or case studies with hard numbers (token count, latency, or dollar cost) directly comparing manual JSON prompting plus post-parsing with native structured-output modes such as OpenAI Function Calling or Gemini responseSchema.**

-----

</details>

<details>
<summary>Which blog posts or papers benchmark JSON, YAML, and XML as LLM output formats, detailing token efficiency, parsing speed, and developer ergonomics?</summary>

### Source [44]: https://celerdata.com/glossary/yaml-json-and-xml-a-practical-guide-to-choosing-the-right-format

Query: Which blog posts or papers benchmark JSON, YAML, and XML as LLM output formats, detailing token efficiency, parsing speed, and developer ergonomics?

Answer: This source provides a **comparative overview** of YAML, JSON, and XML focusing on several key dimensions relevant to LLM output formats:

- **Readability**:
  - **YAML**: Prioritizes human clarity with indentation and inline comments, making it ideal for configuration files edited directly by users.
  - **JSON**: More compact and consistent in syntax; less friendly for manual editing but easy for developers familiar with data objects.
  - **XML**: Most verbose; explicit tags help machine clarity but clutter human readability, especially in deeply nested documents.

- **Performance (Parsing Speed and Token Efficiency)**:
  - **JSON**: Fastest to parse and generate, suitable for real-time APIs and bandwidth-sensitive applications. Efficient streaming and widespread native support.
  - **YAML**: Slower to parse due to complex and flexible syntax, with ambiguities in scalar interpretation adding parser overhead.
  - **XML**: Heaviest and most resource-intensive, especially with namespaces and schema validation.

- **Developer Ergonomics**:
  - **YAML**: Designed for human editing and clarity.
  - **JSON**: Simpler structure, good for automated systems and APIs.
  - **XML**: Better for complex metadata and document validation but harder for day-to-day developer tasks due to verbosity.

The source does not directly benchmark token efficiency in LLM outputs but implies that **JSON is more efficient in parsing speed**, while **YAML is superior for human editing** and **XML** is the least efficient for both parsing and ergonomic use in typical LLM integration scenarios[1].

-----

-----

-----

### Source [45]: https://betterprogramming.pub/yaml-vs-json-which-is-more-efficient-for-language-models-5bc11dd0f6df

Query: Which blog posts or papers benchmark JSON, YAML, and XML as LLM output formats, detailing token efficiency, parsing speed, and developer ergonomics?

Answer: This blog post specifically benchmarks **YAML and JSON as LLM output formats** and addresses **token efficiency, parsing speed, and developer ergonomics**:

- **Parsing Speed**: JSON is generally **faster to parse and consume** than YAML.
- **Token Efficiency**: YAML is “significantly more cost/time-efficient than JSON” when used as an LLM output format, meaning LLMs can produce the same content **faster and cheaper in YAML**. This suggests that YAML, being more concise and flexible, can reduce LLM output token usage.
- **Developer Ergonomics**: YAML’s structure can be more readable and allows for features like inline comments, but JSON’s strictness means easier parsing and type-safety. The article notes that potential issues with JSON’s strictness (such as numbers being output as strings) can be mitigated by schema enforcement or post-parsing conversions.

The post **recommends requesting YAML from LLMs and then converting to JSON in code** for downstream processing, indicating that this approach leverages both YAML’s generation efficiency and JSON’s parsing simplicity. The article also references the use of **YAML comments for chain-of-thought reasoning** in prompts for LLMs, providing additional utility for developer workflows[3].

-----

-----

</details>

<details>
<summary>What documented production case studies show LLM-driven entity extraction feeding knowledge graphs or GraphRAG pipelines, and quantify the improvements in accuracy, scalability, or maintenance effort?</summary>

### Source [47]: https://arxiv.org/pdf/2402.07483

Query: What documented production case studies show LLM-driven entity extraction feeding knowledge graphs or GraphRAG pipelines, and quantify the improvements in accuracy, scalability, or maintenance effort?

Answer: This case study details the use of LLMs for organizational entity extraction to support a question-answering application over governance documents. The team fine-tuned a Llama-2 7B model using QLoRA for efficiency, achieving significant memory savings—reducing trainable parameters by a factor of ~200 compared to full fine-tuning. The extracted entities and their hierarchical relationships were represented as a tree graph, enabling accurate responses to queries about organizational structure. Although this work demonstrates LLM-driven entity extraction feeding a structured graph (and subsequent QA over this graph), the paper emphasizes hardware efficiency (e.g., memory savings, parameter reduction) rather than quantified gains in accuracy, scalability, or maintenance effort in production settings.

-----

-----

-----

### Source [48]: https://arxiv.org/html/2503.07993v1

Query: What documented production case studies show LLM-driven entity extraction feeding knowledge graphs or GraphRAG pipelines, and quantify the improvements in accuracy, scalability, or maintenance effort?

Answer: This paper presents an enterprise framework that leverages LLMs for automated entity extraction, relationship inference, and semantic enrichment to construct unified, activity-centric knowledge graphs from heterogeneous data sources (emails, chats, logs, etc.). The resulting knowledge graph supports advanced analytics, contextual search, personalized recommendations, and expertise discovery. Experimental results are cited as demonstrating successful application in expertise discovery, task management, and data-driven decision-making. While the framework is described as improving the unification and actionable insights from siloed enterprise data, the paper does not provide explicit quantitative metrics on improvements in accuracy, scalability, or maintenance effort compared to traditional methods.

-----

-----

-----

### Source [49]: https://arxiv.org/html/2505.20099v1

Query: What documented production case studies show LLM-driven entity extraction feeding knowledge graphs or GraphRAG pipelines, and quantify the improvements in accuracy, scalability, or maintenance effort?

Answer: This comprehensive survey examines the synthesis of LLMs and knowledge graphs for question answering, focusing on advanced retrieval-augmented generation (RAG) techniques. The paper describes GraphRAG and related methods where LLM-driven entity extraction is used to construct or augment knowledge graphs for retrieval. Approaches like GRAG and LEGO-GraphRAG introduce subgraph retrieval and path refinement, enhancing reasoning and retrieval accuracy. While the survey reviews architectural improvements and theoretical benefits (such as improved reasoning over retrieved knowledge), it does not include detailed production case studies or quantified metrics on accuracy, scalability, or maintenance effort resulting from deploying these pipelines.

-----

-----

-----

### Source [50]: https://neo4j.com/blog/developer/knowledge-graph-extraction-challenges/

Query: What documented production case studies show LLM-driven entity extraction feeding knowledge graphs or GraphRAG pipelines, and quantify the improvements in accuracy, scalability, or maintenance effort?

Answer: This Neo4j blog discusses practical challenges in extracting knowledge graphs, including those built with LLM-driven entity extraction. It highlights technical obstacles such as handling unstructured data, entity disambiguation, and relationship inference. The blog references the Neo4j LLM Knowledge Graph Extraction project, which aims to automate graph population using LLMs. While it outlines real-world integration challenges and qualitative improvements in flexibility and integration, it lacks specific production case studies and does not quantify improvements in accuracy, scalability, or maintenance effort compared to traditional approaches.

-----

-----

</details>

<details>
<summary>How do developers integrate Pydantic models directly with Gemini’s responseSchema (or similar) in code, including patterns for validation errors and automatic retries?</summary>

### Source [52]: https://github.com/langchain-ai/langchain/discussions/21594

Query: How do developers integrate Pydantic models directly with Gemini’s responseSchema (or similar) in code, including patterns for validation errors and automatic retries?

Answer: This LangChain discussion explains that to handle output using Pydantic models in production environments with Gemini Pro, you can use LangChain's `PydanticOutputParser`. This parser allows you to validate and structure LLM outputs according to your Pydantic model schema. When integrating, validation errors such as `ValidationError` or `OutputParserException` may occur if the returned data does not match the schema. These should be handled with comprehensive error handling strategies to ensure robustness. It is suggested that you adapt your retry logic or error handling based on these exceptions to maintain reliable outputs.

Relevant LangChain components for this integration are:
- `libs/langchain/langchain/chains/structured_output/base.py`
- `output_parser_structured.ipynb`
- `libs/core/langchain_core/output_parsers/pydantic.py`

This approach is effective for handling complex or nested data structures, utilizing Pydantic’s validation capabilities to ensure outputs conform to the defined schema. However, the discussion notes this as a starting point and not a complete, production-ready solution.

-----

-----

-----

### Source [53]: https://ai.google.dev/gemini-api/docs/structured-output

Query: How do developers integrate Pydantic models directly with Gemini’s responseSchema (or similar) in code, including patterns for validation errors and automatic retries?

Answer: The official Gemini API documentation states that when using Python, you do **not** need to manually work with Schema objects if you use a Pydantic model. The Python Gemini library will automatically convert the Pydantic model into a JSON schema and submit it to the API. Supported types include `int`, `float`, `bool`, `str`, `list`, and user-defined Pydantic models (including nested structures).

For more advanced schema needs, Gemini supports JSON Schema via the `responseJsonSchema` field (preview feature for Gemini 2.5), which allows you to pass a JSON Schema generated from Pydantic. However, there are some limitations:
- Only works with Gemini 2.5 and above.
- Not all JSON Schema properties are fully supported.
- Recursive references are supported with certain constraints.

A code example is referenced (but not shown in the excerpt) for generating a JSON Schema from a Pydantic model and submitting it with the request. The documentation does not explicitly mention patterns for handling validation errors or automatic retries, but the automatic conversion of Pydantic models to schema is clearly described.

-----

-----

-----

### Source [55]: https://ai.pydantic.dev/models/

Query: How do developers integrate Pydantic models directly with Gemini’s responseSchema (or similar) in code, including patterns for validation errors and automatic retries?

Answer: The PydanticAI documentation describes that it supports Gemini models (including different Gemini API endpoints) and can automatically select the appropriate model class, provider, and profile when using a standard provider:model string (e.g., `openrouter:google/gemini-2.5-pro-preview`). The documentation mentions that different models have different restrictions on JSON schemas, and the same schema transformer logic applies regardless of whether you are using Gemini or OpenAI endpoints.

While this source outlines the ability to integrate Pydantic models with Gemini via automatic schema transformation, it does not detail patterns for handling validation errors or automatic retries in code. The focus is on the provider/model abstraction and compatibility layer, rather than error management or retry strategies.

-----

-----

</details>

<details>
<summary>What security or reliability analyses explain how schema-validated outputs (e.g., via Pydantic) mitigate risks such as prompt injection, malformed data, or downstream crashes?</summary>

### Source [56]: https://docs.pydantic.dev/1.10/usage/models/

Query: What security or reliability analyses explain how schema-validated outputs (e.g., via Pydantic) mitigate risks such as prompt injection, malformed data, or downstream crashes?

Answer: Pydantic provides strong guarantees about **data integrity and schema compliance** by parsing and validating untrusted data before it becomes part of a model instance. After validation, Pydantic ensures that all fields of the resultant model conform to the declared types and constraints, significantly reducing the risk of malformed data entering the application. 

If validation fails, Pydantic raises a `ValidationError` that:
- **Enumerates all validation failures** found in the input, providing detailed error reporting.
- Each error object includes the **location** of the invalid field (`loc`), a **computer-readable type** of the error (`type`), and a **human-readable message** (`msg`).

This mechanism directly mitigates the risks of **downstream crashes** and data-type vulnerabilities, since only validated and correctly shaped data is allowed through. By using schema validation for outputs, applications can avoid unexpected behavior, crashes, or security issues that can be caused by malformed or malicious data (such as prompt injection or type confusion). The comprehensive error reporting also enables developers to implement robust error handling and logging, further improving system reliability and debuggability[2].

-----

-----

-----

### Source [57]: https://betterstack.com/community/guides/scaling-python/pydantic-explained/

Query: What security or reliability analyses explain how schema-validated outputs (e.g., via Pydantic) mitigate risks such as prompt injection, malformed data, or downstream crashes?

Answer: Pydantic enforces **data types, field constraints, and custom validation logic**, ensuring that only data conforming to the specified schema is accepted. For instance, custom field validators (such as password strength checks) can be used to enforce complex business rules or security requirements. When invalid or malicious input is encountered, Pydantic immediately raises a validation error, preventing it from being processed or stored.

By ensuring that only correctly formatted and validated data enters the system, Pydantic reduces the risk of:
- **Security vulnerabilities** (such as prompt injection, where attackers try to manipulate prompts or inject unintended content).
- **Malformed data** causing application errors or undefined behavior.
- **Downstream crashes** resulting from unexpected data types or structures.

This approach provides a clear boundary between untrusted input and trusted, schema-validated data, which is critical for building secure and reliable systems[3].

-----

-----

-----

### Source [58]: https://www.couchbase.com/blog/validate-json-documents-in-python-using-pydantic/

Query: What security or reliability analyses explain how schema-validated outputs (e.g., via Pydantic) mitigate risks such as prompt injection, malformed data, or downstream crashes?

Answer: Using Pydantic for schema validation is especially important when dealing with **unstructured or semi-structured data** (e.g., from NoSQL databases or external services). Without validation, missing or incorrectly typed fields can cause application failures or unpredictable behavior. 

By specifying a Pydantic schema:
- Applications can **detect and reject documents that do not match expected specifications**, preventing them from being processed further.
- Developers can implement **field-specific validation functions** to check content and type, ensuring that even complex requirements are satisfied.
- This process helps **highlight and isolate bad data** before it affects application functionality, mitigating risks from unreliable or malicious sources.

Validating outputs against a schema both ensures compatibility with downstream consumers and protects applications from the operational and security risks of accepting malformed or manipulated data[4].

-----

-----

-----

### Source [59]: https://dev.to/devasservice/best-practices-for-using-pydantic-in-python-2021

Query: What security or reliability analyses explain how schema-validated outputs (e.g., via Pydantic) mitigate risks such as prompt injection, malformed data, or downstream crashes?

Answer: Adhering to best practices with Pydantic, such as using correct type annotations, default values, and efficient parsing methods, further enhances **data validation and reliability**. Proper use of Pydantic configuration options and error handling mechanisms ensures that only well-formed data is accepted and that validation failures are handled gracefully.

Key recommendations include:
- **Documenting models and validation rules** to clarify validation logic.
- **Encapsulating validation logic** correctly to avoid clutter and confusion.
- **Avoiding excessive model complexity** to maintain maintainability and understandability.

By following these practices, developers can maximize the effectiveness of schema validation in mitigating risks from malformed data, reducing the attack surface for prompt injection, and preventing application crashes due to invalid outputs[1].

-----

-----

</details>


## Sources Scraped From Research Results

<details>
<summary>In the world of **Large Language Models (LLMs)**, making them play nice with our applications is key. We want our models to dish out neat JSON for easy integration into our full-stack setups. But relying solely on LLM luck for perfect JSON? Let's be realistic.</summary>

In the world of **Large Language Models (LLMs)**, making them play nice with our applications is key. We want our models to dish out neat JSON for easy integration into our full-stack setups. But relying solely on LLM luck for perfect JSON? Let's be realistic.

Suppose you're incorporating an LLM into your app, striving for precise JSON output. Considering the importance of this data, we might need to save it for the next steps in our logic. You provide clear prompts, cross your fingers, and hope. Yet, hope isn't a strategy, and guarantees are scarce.

Meet [**Pydantic**](https://pypi.org/project/pydantic/?ref=blog.kusho.ai), a handy data validation tool. This tool turns your JSON into a structured class for order in the chaos. Plus, Pydantic brings validations and extra functionality to the table.

We'll also use [**Instructor**](https://pypi.org/project/instructor/?ref=blog.kusho.ai). **Instructor** patches our OpenAI client, empowering it to return our response model (essentially our Pydantic class). Additionally, we can incorporate **Max Retries** to automatically retry when our LLM fails to deliver the desired output

In simple terms, Pydantic takes your JSON, turns it into a class, and lets you add checks and tweaks. The real win? When Pydantic teams up with LLMs, making your applications more reliable and functional.

# Tackling the JSON Conundrum

Reliability in our outputs is really important (else we'll see you in try-catch hell). Consider this scenario: imagine you're crafting a medical application tasked with extracting information from a string. The next logical step is to convert this data into JSON for further analysis. Perhaps, you plan to map this JSON to a class object, storing it either temporarily in memory or persisting it in a database, say using SQLAlchemy.

In this process, the challenge lies in ensuring that the JSON output remains accurate and consistent, ready to be seamlessly integrated into your application's logic. This is where the crux of the problem resides.

Let's take our medical example and flesh it out.

Suppose we want this information :-

```python
medical_info = """Sex: Female, Age: 79
Geographical region: North America
Pathology: Spontaneous pneumothorax
Symptoms:
---------
 - I have chest pain even at rest.
 - I feel pain.
 - The pain is:
     » a knife stroke
 - The pain locations are:
     » upper chest
     » breast(R)
     » breast(L)
 - On a scale of 0-10, the pain intensity is 7
 - On a scale of 0-10, the pain's location precision is 4
 - On a scale of 0-10, the pace at which the pain appear is 9
 - I have symptoms that increase with physical exertion but alleviate with rest.
Antecedents:
-----------
 - I have had a spontaneous pneumothorax.
 - I smoke cigarettes.
 - I have a chronic obstructive pulmonary disease.
 - Some family members have had a pneumothorax.
Differential diagnosis:
----------------------
Unstable angina: 0.262, Stable angina: 0.201, Possible NSTEMI / STEMI: 0.160, GERD: 0.145, Pericarditis: 0.091, Atrial fibrillation: 0.082, Spontaneous pneumothorax: 0.060
"""
```

To be converted into this format:-

```python
json_format = """{
    "patient_info": {
        "sex": "",
        "age": ,
        "geographical_region": "",
    },
    "medical_history": {
        "pathology": "",
        "symptoms": {
            "description": "",
            "pain": {
                "type": "",
                "locations": [],
                "intensity": ,
                "location_precision": ,
                "pace": ,
            },
            "increase_with_exertion": true/false,
            "alleviate_with_rest": true/false,
        },
    },
    "risk_factors": {},
    "differential_diagnosis": [\
    {\
        "disease_name": "",\
        "probability":\
    },\
]
}"""

```

Let's take a sec and think what we are doing. If we assume our llms as a black box:-

```python
def llm(prompt: str, schema: str) -> str:
  pass  # Black Magic, and hope to receive valid json.
```

_**Now let's plead to the AI goddess to convert this into valid JSON.**_

```python
completion = openai_client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{\
        "role": "user",\
        "content": f"Please convert the following information into valid json representing the medical diagnosis ${medical_info}. Please convert the data in the following format and fill in the data ${json_format}"\
    }]
)
# Now let's extract our "valid" json
dump = completion.model_dump()
medical_info  =json.loads(dump["choices"][0]["message"]["content"])
print(json.dumps(medical_info, indent=2)) # A big leap of faith.

```

> In the code, relying solely on json.dumps might lead to errors if the model doesn't provide valid JSON. Adding a try-except block for error handling and incorporating a retry mechanism can be quite cumbersome. Dealing with these uncertainties emphasizes the challenges of ensuring a smooth interaction with the language model output.

By the way output we get looks something like this:-

```json
{
  "patient_info": {
    "sex": "Female",
    "age": 79,
    "geographical_region": "North America"
  },
  "medical_history": {
    "pathology": "Spontaneous pneumothorax",
    "symptoms": {
      "description": "I have chest pain even at rest. I feel pain.",
      "pain": {
        "type": "a knife stroke",
        "locations": [\
          "upper chest",\
          "breast(R)",\
          "breast(L)"\
        ],
        "intensity": 7,
        "location_precision": 4,
        "pace": 9
      },
      "increase_with_exertion": true,
      "alleviate_with_rest": true
    }
  },
//...
      "probability": 0.06
    }
```

The issue of uncertainties becomes more pronounced when dealing with complex data structures or interconnected structures.

We're crossing our fingers, hoping that when we convert our LLM output, a string supposedly in valid JSON format, into our object, everything works smoothly. However, in our current testing example, a couple of issues are still lingering:

**1\. Lack of Type Safety:**

The current approach involves converting a string to a JSON object, and we're essentially relying on the all-powerful AI god to provide us with correct JSON. What if, instead of a birthdate, we need...

**2\. Validation Issues:**

Handling input validation manually is a bit of a headache. To validate, we have to manually check the structure of the JSON, which results in a messy function like this:

```python
def validate_json_structure(json_string):
    try:
        data = json.loads(json_string)

        # Validate patient_info
        patient_info = data.get("patient_info")
        if not patient_info or not isinstance(patient_info, dict):
            return False

        # Validate sex, age, and geographical_region in patient_info
        #... os on with more and more validations within validations
        return True

    except json.JSONDecodeError:
        return False
```

**What a horrible mess**. It's not the most elegant solution. (Psst, we'll soon explore how Pydantic can simplify this mess and add various validations.)

On another note, Pydantic allows us to chain our prompts using inheritance(OOP), as you'll see in an example towards the end of this blog.

## Using Pydantic and Instructor to get a Structured response

We aim for our magical function to receive a schema defined as a Python class or model and return either the same or another class/model. It should look something like this:-

```python
def llm(prompt: str, schema: Model) -> Model:
    pass

```

This is where Pydantic steps in. Let's import the necessary modules and set up our OpenAI client with the help of [Instructor](https://pypi.org/project/instructor/?ref=blog.kusho.ai):

````bash
```bash
$ pip install instructor # To install instructor
```
````

```python
import instructor

instructor_openai_client = instructor.patch(openai.Client(
    api_key=open_ai_key, organization=open_ai_org_key, timeout=20000, max_retries=3
))

```

Overall, Instructor is a user-friendly, transparent, and Pythonic solution for leveraging OpenAI's function calling to extract data. It patches to the OpenAI's library and helps us achieve the `(prompt, model) -> model` structure.

* * *

Next, we define our JSON structure using Pydantic classes. This approach allows us to include additional docstrings for field descriptions and other useful information. All of this aids the language model in generating or extracting information from the context provided by the model.

```shell
$ pip install pydantic # To install pydantic

```

```python
class Symptoms(BaseModel):
    """
        Represents the symptoms of a patient.
    """
    description: str = Field(description="A general scientific and objective description of the symptoms.")
    pain_type: str
    locations: List[str]
    intensity: int
    location_precision: int
    pace: int

class MedicalHistory(BaseModel):
    pathology: str
    symptoms: Symptoms
    increase_with_exertion: bool
    alleviate_with_rest: bool

class RiskFactors(BaseModel):
    spontaneous_history: bool
    smoking_history: bool
    copd_history: bool
    family_history: str

class DifferentialDiagnosis(BaseModel):
    disease_name: str
    probability: float

class PatientInfo(BaseModel):
    sex: Literal['M', 'F']
    age: int
    geographical_region: str

class PatientData(BaseModel):
    patient_info: PatientInfo
    medical_history: MedicalHistory
    risk_factors: RiskFactors
    differential_diagnosis: List[DifferentialDiagnosis]
```

Classes utilising Pydantic to represent our response json

### Now, let's utilize instructor client with desired response\_model

```python
completion = instructor_openai_client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[\
      {\
        "role": "user",\
        "content": f"Please convert the following information into valid JSON representing the medical diagnosis {medical_info}."\
      }\
    ],
    response_model=MedicalInfo # Replace with the appropriate response model
)
print(type(completion))
print(json.dumps(completion.model_dump(), indent=1))

```

**Voila**:-

```shell
<class '__main__.PatientData'> # Notice how the type of data structure we got is a class!!!
{
 "patient_info": {
  "sex": "F",
  "age": 79,
  "geographical_region": "North America"
 },
 "medical_history": {
  "pathology": "Spontaneous pneumothorax",
  "symptoms": {
   "description": "I have chest pain even at rest. I feel pain. The pain is a knife stroke. The pain locations are upper chest, breast(R), breast(L). On a scale of 0-10, the pain intensity is 7. On a scale of 0-10, the pain's location precision is 4. On a scale of 0-10, the pace at which the pain appears is 9. I have symptoms that increase with physical exertion but alleviate with rest.",
   "pain_type": "knife stroke",
   "locations": [\
    "upper chest",\
    "breast(R)",\
    "breast(L)"\
   ],
   "intensity": 7,
   "location_precision": 4,
   "pace": 9
  },
  "increase_with_exertion": true,
  "alleviate_with_rest": true
 },
 "risk_factors": {
...
   "probability": 0.06
  }
 ]
}
```

_Notice how the type of data structure we got is a class!!!_

By setting `response_model` to `MedicalInfo`, we ensure a clear output structure. Pydantic guarantees data adherence, streamlining integration and providing a type hint of `PatientData`.

Pydantic organizes JSON with automatic validation. Deviations trigger validation errors, ensuring data integrity.

Docstrings and field descriptions aid developers and shape the JSON schema for OpenAI. Navigate confidently with structured, validated data, and notice the response type as `PatientData` for seamless integration.

# **Congratulations, It's a class!**

In the next part of this series, we'll talk about LLM validations, seamless retry mechanisms, how you can create complex data structures like directed acyclic graphs (DAGs), and much more using Pydantic. Stay tuned for the next part.

> References:-
>
> 1. This blog post is inspired by an awesome talk by [Jason Liu](https://github.com/jxnl?ref=blog.kusho.ai) watch his [talk](https://www.youtube.com/watch?v=yj-wSRJwrrc&ref=blog.kusho.ai) for better reference.
> 2. [Pydantic](https://pypi.org/project/pydantic/?ref=blog.kusho.ai), [Instructor](https://pypi.org/project/instructor/?ref=blog.kusho.ai)

</details>

<details>
<summary>[https://avatars.githubusercontent.com/u/49721249?s=64&v=4\ MuhammadNizamani](https://github.com/MuhammadNizamani) [May 13, 2024](https://github.com/langchain-ai/langchain/discussions/21594\#discussion-6659856)</summary>

## [https://avatars.githubusercontent.com/u/49721249?s=64&v=4\ MuhammadNizamani](https://github.com/MuhammadNizamani) [May 13, 2024](https://github.com/langchain-ai/langchain/discussions/21594\#discussion-6659856)

|     |
| --- |
| ### Checked other resources<br>- [x]  I added a very descriptive title to this question.<br>- [x]  I searched the LangChain documentation with the integrated search.<br>- [x]  I used the GitHub search to find a similar question and didn't find it.<br>### Commit to Help<br>- [x]  I commit to help with one of those options 👆<br>### Example Code<br>```<br>NA<br>```<br>### Description<br>I am using Gemini pro and I want to handle output using pydentic so I have more hold output.<br>I know so trick around it but those won't work in PROD.<br>I am going to product following code of openai library that uses that concept.<br>````notranslate<br>import pydantic<br>import instructor<br>import openai<br>from enum import Enum<br>from typing import Tuple, ForwardRef, List<br>from pydantic import BaseModel, Field<br># Possible assessment statuses<br>class Status(str, Enum):<br>    correct = "correct"<br>    incorrect = "incorrect"<br># We will use this to model the output of our LLM<br>class QAAssessment(BaseModel):<br>    question: str = Field(..., description="Question")<br>    answer: str = Field(..., description="Answer")<br>    assessment_status: Status = Field(..., description="Assessment status")<br>    assessment_comment: str<br># Let's do a 10 cycle test to assess how stable is the result<br>for i in range(10):<br>    pydtest: QAAssessment = openai.ChatCompletion.create(<br>        model="gpt-3.5-turbo-0613",<br>        temperature=0,<br>        response_model=QAAssessment,<br>        messages=[<br>            {"role": "system", "content": "You are an assisstant that assesses if the Human knows the colors of fruits."},<br>            {"role": "assistant", "content": "What is the color of an apple?"},<br>            {"role": "user", "content": "Blue"},<br>        ]<br>    )<br>    json_response = pydtest.model_dump_json()<br>    json_pretty_print(json_response)<br>    ```<br> <br>### System Info<br>NA<br>```` |

---

### [https://avatars.githubusercontent.com/in/324583?s=64&v=4\ dosubot\[bot\]](https://github.com/apps/dosubot) bot [May 13, 2024](https://github.com/langchain-ai/langchain/discussions/21594\#discussioncomment-9417007)

|     |
| --- |
| To handle output using Pydantic in a production environment with Gemini Pro, you can utilize the LangChain framework's `PydanticOutputParser` class. Here's a streamlined approach to structure the output from the OpenAI library using Pydantic models:<br>1. Define your Pydantic model, for example, `QAAssessment`, to structure the output.<br>2. Use the `PydanticOutputParser` from LangChain to parse the output into your Pydantic model.<br>3. Create a prompt template with LangChain's `PromptTemplate`, incorporating instructions for formatting the output.<br>4. Chain the prompt, model, and parser together to process and structure the output.<br>Here's a concise example:<br>```<br>from langchain_core.output_parsers import PydanticOutputParser<br>from langchain_core.prompts import PromptTemplate<br>from langchain_openai import OpenAI<br>from pydantic import BaseModel, Field<br>from enum import Enum<br>class Status(str, Enum):<br>    correct = "correct"<br>    incorrect = "incorrect"<br>class QAAssessment(BaseModel):<br>    question: str = Field(description="Question")<br>    answer: str = Field(description="Answer")<br>    assessment_status: Status = Field(description="Assessment status")<br>    assessment_comment: str<br>model = OpenAI(model_name="gpt-3.5-turbo-0613", temperature=0.0)<br>parser = PydanticOutputParser(pydantic_object=QAAssessment)<br>prompt = PromptTemplate(<br>    template="Answer the user query.\n{format_instructions}\n{query}\n",<br>    input_variables=["query"],<br>    partial_variables={"format_instructions": parser.get_format_instructions()},<br>)<br>chain = prompt | model | parser<br>parsed_output = chain.invoke({<br>    "query": [<br>        {"role": "system", "content": "You are an assistant that assesses if the Human knows the colors of fruits."},<br>        {"role": "assistant", "content": "What is the color of an apple?"},<br>        {"role": "user", "content": "Blue"},<br>    ]<br>})<br>```<br>This method ensures that the output from language models is structured and validated according to your application's requirements. For more details, refer to the LangChain repository, specifically the structured output base and output parser structured notebook.<br>Remember, when integrating Pydantic models, you might encounter errors such as `ValidationError` or `OutputParserException`. These can be resolved by ensuring the data structure from the OpenAI library matches the Pydantic model's schema and implementing comprehensive error handling.<br>This approach allows for handling complex or nested data structures by leveraging Pydantic's parsing and validation capabilities, ensuring your output is structured as per your defined schema.<br>#### Details<br>**Sources**<br>- [libs/langchain/langchain/chains/structured\_output/base.py](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/structured_output/base.py)<br>- [docs/docs/how\_to/output\_parser\_structured.ipynb](https://github.com/langchain-ai/langchain/blob/master/docs/docs/how_to/output_parser_structured.ipynb)<br>- [libs/core/langchain\_core/output\_parsers/openai\_tools.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/output_parsers/openai_tools.py)<br>- [libs/core/langchain\_core/output\_parsers/openai\_functions.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/output_parsers/openai_functions.py)<br>- [libs/core/langchain\_core/output\_parsers/pydantic.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/output_parsers/pydantic.py)<br>**About Dosu**<br>This response is meant to be useful and save you time. It is **not meant to be a precise solution**, but rather a starting point for your own research.<br>Help me be more useful! Please leave a 👍 if this is helpful and 👎 if it is irrelevant.<br>_To continue the conversation, mention [@dosu](https://github.com/dosu)._ |

</details>

<details>
<summary>How to Write AI Prompts That Output Valid JSON Data</summary>

# How to Write AI Prompts That Output Valid JSON Data

When working with LLMs, an extremely valuable use case is generating structured data as the output response of the AI prompt. Whether you’re building an app, prototyping a data pipeline, or performing data extraction or transformation, receiving structured outputs like JSON saves time and makes downstream processing seamless.

But there’s a challenge: these models are optimized for natural language—not strict formatting. Without careful prompting, you might get outputs that contain JSON or event look like JSON but ultimately fail to parse correctly. To avoid that, you need to guide the model and instruct it to the output you require.

This article walks you through prompt engineering strategies for writing AI prompts that reliably produce valid JSON responses, with examples and sample code.

* * *

## 1\. Be Explicit About JSON Output

The most basic and important instruction you can give is:

```text hljs language-plaintext

Plain textResponse with valid JSON only. Do not include any explanation or extra text.
Copy
```

Generative AI models are trained to respond conversationally by default. This instruction helps shift the tone and format to strictly structured output. Keep the prompt short and direct to minimize risk of formatting drift.

* * *

## 2\. Tell the LLM What You Want

Before the LLM can output JSON data of what you’re looking for, you will need to instruct the AI what you want it to do.

Here’s a simple example of a prompt that tells the LLM what we want:

```text hljs language-plaintext

Plain textGenerate a list of 3 fictional users
Copy
```

## 3\. Include an Example or Schema

You will need to tell the LLM what the output response should look like and how it should be formatted. Specifying JSON is great, but you likely require a very specific schema. You can either explain the schema you require, or give the LLM an example JSON to show it was you need.

### Schema-style

Explaining the schema you require in the JSON output is one method of telling the LLM how to format the data:

```text hljs language-plaintext

Plain textEach item should have: name (string), age (number), signup_date (date)
Copy
```

### Example-style Prompt

A method that will likely increase the accuracy and reliability of the LLM to output the JSON schema you need is to explicitly give it an example of the JSON you want it to output:

```text hljs language-plaintext

Plain textOutput using the following JSON format:

[\
  {\
    "name": "Steve Johnson",\
    "age": 43\
    "signup_date": "2025-01-01"\
  }\
]
Copy
```

Models are very good at copying structure—give it something to copy.

* * *

## 4\. Avoid Overcomplication in Prompts

Maintaining clarity is key and being explicit helps. Avoid vague instructions or extra requirements that could lead to inconsistencies.

Here’s an example of a prompt that might confuse the LLM:

```text hljs language-plaintext

Plain textWrite a list of products in JSON format. Each should have a name, age, and signup_date.
Also make sure the prices are realistic, and don't forget to include at least one out-of-stock item.
Copy
```

Instead, you can see the following prompt is much clearer:

```text hljs language-plaintext

Plain textWrite a list of products, each should have name (string), age (number), signup_date (date)
Copy
```

Simple, direct prompts will generally yield better-structured responses.

* * *

## 5\. Use System Prompt Instructions (If Available)

If you’re using an API like OpenAI’s Chat API or tools like LangChain, be sure to take advantage of the **system prompt** This can be used to instruct the LLM how it should behave and reinforce the expected behavior:

```json hljs

JSON{"role": "system", "content": "You are a JSON generator. Always output valid JSON without explanations."}
Copy
```

This reduces the risk of the model slipping into natural language commentary in the response.

* * *

## 6\. Prepare for Errors

Even well-prompted models sometimes return extra text, incorrect brackets, or malformed syntax. Build safeguards into your workflow:

- **Validate the output** using a parser like `json.loads()` in Python
- **Use temperature=0** for consistent and deterministic formatting
- **Post-process** if necessary to strip markdown artifacts or retry

A clean-up and validation step ensures your pipeline doesn’t break.

* * *

## Full Example: Prompting the LLM and Saving JSON with Python

Here’s a working Python example that:

- Sends a prompt to Azure OpenAI using `langchain-openai`
- Retrieves a response
- Cleans and parses the JSON
- Saves it to a `.json` file

````python hljs

Pythonimport os
import json
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# Set up the Azure OpenAI chat model
chat = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# System prompt to guide the model
system_prompt = """
You are a JSON generator. Your task is to generate valid JSON data based on the provided prompt.
"""

# Your JSON-focused prompt
user_prompt = """
Generate a list of 3 fictional users.

Here's an example of the JSON format you should use:

[\
    {\
        "name": "string",\
        "age": number,\
        "email": "string",\
        "signup_date": "YYYY-MM-DD"\
    }\
]
"""

# Call the chat model directly
response = chat.invoke([\
    SystemMessage(content=system_prompt),\
    HumanMessage(content=user_prompt)\
])

# Invoke the chat model
#response = chat.invoke(prompt)

# Get the response text
response_text = response.content

print("\nRaw Response:\n", response_text)

# Clean up response (remove code block wrappers if present)
response_text = response_text.strip().replace("```json", "").replace("```", "").strip()

print("\n\nCleaned Response JSON:\n", response_text)

# Parse and save JSON
data = json.loads(response_text)

# Save to file
os.makedirs("output", exist_ok=True)
with open("output/users.json", "w") as f:
    json.dump(data, f, indent=4)

print("Saved JSON to output/users.json")
Copy
````

### Explanation

- **System & User Prompts**: The model is guided with both a system-level instruction to behave like a JSON generator, and a user prompt that includes both instructions and an example.
- **Example Format**: Including a sample JSON block in the user prompt helps the model replicate the correct structure.
- **Message Format**: This example uses `SystemMessage` and `HumanMessage` from LangChain to structure the interaction clearly.
- **Raw vs Cleaned Output**: Prints the raw model output before and after removing markdown formatting which is commonly added by the LLM.
- **Validation**: Uses `json.loads()` to ensure the cleaned string is valid JSON.
- **File Output**: Saves the JSON to an `output` directory, which is created if it doesn’t exist.

This approach mirrors how you’d use structured prompts and system guidance in production settings. It’s flexible, clear, and easy to expand for more complex workflows like multi-turn dialogs, pipelines, or evaluation tools.

Here’s an example of the JSON file that is saved at the end of this code:

```json hljs

JSON[\
    {\
        "name": "Alice Evergreen",\
        "age": 28,\
        "email": "alice.evergreen@example.com",\
        "signup_date": "2023-02-15"\
    },\
    {\
        "name": "Michael Stone",\
        "age": 35,\
        "email": "michael.stone@example.com",\
        "signup_date": "2022-11-08"\
    },\
    {\
        "name": "Sofia Bright",\
        "age": 22,\
        "email": "sofia.bright@example.com",\
        "signup_date": "2023-08-21"\
    }\
]
Copy
```

* * *

## Conclusion

Prompt engineering techniques is important when working with LLMs. This is especially true when using the LLM to produce predictable structured data that is valid JSON. This is a powerful technique that unlocks automation, data processing, and tool building automation capabilities using LLMs. With well-structured prompts and a few best practices, you can go from free-form text generation to clean, parsable, and ready-to-use structured data.

Think of JSON prompting as a bridge between natural language creativity and structured logic—master it, and you’ll get the best of both worlds.

</details>

<details>
<summary>Model Providers</summary>

# Model Providers

PydanticAI is model-agnostic and has built-in support for multiple model providers:

- [OpenAI](https://ai.pydantic.dev/models/openai/)
- [Anthropic](https://ai.pydantic.dev/models/anthropic/)
- [Gemini](https://ai.pydantic.dev/models/gemini/) (via two different APIs: Generative Language API and VertexAI API)
- [Groq](https://ai.pydantic.dev/models/groq/)
- [Mistral](https://ai.pydantic.dev/models/mistral/)
- [Cohere](https://ai.pydantic.dev/models/cohere/)
- [Bedrock](https://ai.pydantic.dev/models/bedrock/)

## OpenAI-compatible Providers

In addition, many providers are compatible with the OpenAI API, and can be used with `OpenAIModel` in PydanticAI:

- [DeepSeek](https://ai.pydantic.dev/models/openai/#deepseek)
- [Grok (xAI)](https://ai.pydantic.dev/models/openai/#grok-xai)
- [Ollama](https://ai.pydantic.dev/models/openai/#ollama)
- [OpenRouter](https://ai.pydantic.dev/models/openai/#openrouter)
- [Perplexity](https://ai.pydantic.dev/models/openai/#perplexity)
- [Fireworks AI](https://ai.pydantic.dev/models/openai/#fireworks-ai)
- [Together AI](https://ai.pydantic.dev/models/openai/#together-ai)
- [Azure AI Foundry](https://ai.pydantic.dev/models/openai/#azure-ai-foundry)
- [Heroku](https://ai.pydantic.dev/models/openai/#heroku-ai)

PydanticAI also comes with [`TestModel`](https://ai.pydantic.dev/api/models/test/) and [`FunctionModel`](https://ai.pydantic.dev/api/models/function/)
for testing and development.

To use each model provider, you need to configure your local environment and make sure you have the right
packages installed.

## Models and Providers

PydanticAI uses a few key terms to describe how it interacts with different LLMs:

- **Model**: This refers to the PydanticAI class used to make requests following a specific LLM API
(generally by wrapping a vendor-provided SDK, like the `openai` python SDK). These classes implement a
vendor-SDK-agnostic API, ensuring a single PydanticAI agent is portable to different LLM vendors without
any other code changes just by swapping out the Model it uses. Model classes are named
roughly in the format `<VendorSdk>Model`, for example, we have `OpenAIModel`, `AnthropicModel`, `GeminiModel`,
etc. When using a Model class, you specify the actual LLM model name (e.g., `gpt-4o`,
`claude-3-5-sonnet-latest`, `gemini-1.5-flash`) as a parameter.
- **Provider**: This refers to provider-specific classes which handle the authentication and connections
to an LLM vendor. Passing a non-default _Provider_ as a parameter to a Model is how you can ensure
that your agent will make requests to a specific endpoint, or make use of a specific approach to
authentication (e.g., you can use Vertex-specific auth with the `GeminiModel` by way of the `VertexProvider`).
In particular, this is how you can make use of an AI gateway, or an LLM vendor that offers API compatibility
with the vendor SDK used by an existing Model (such as `OpenAIModel`).
- **Profile**: This refers to a description of how requests to a specific model or family of models need to be
constructed to get the best results, independent of the model and provider classes used.
For example, different models have different restrictions on the JSON schemas that can be used for tools,
and the same schema transformer needs to be used for Gemini models whether you're using `GoogleModel`
with model name `gemini-2.5-pro-preview`, or `OpenAIModel` with `OpenRouterProvider` and model name `google/gemini-2.5-pro-preview`.

When you instantiate an [`Agent`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent) with just a name formatted as `<provider>:<model>`, e.g. `openai:gpt-4o` or `openrouter:google/gemini-2.5-pro-preview`,
PydanticAI will automatically select the appropriate model class, provider, and profile.
If you want to use a different provider or profile, you can instantiate a model class directly and pass in `provider` and/or `profile` arguments.

## Custom Models

To implement support for a model API that's not already supported, you will need to subclass the [`Model`](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.Model) abstract base class.
For streaming, you'll also need to implement the [`StreamedResponse`](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.StreamedResponse) abstract base class.

The best place to start is to review the source code for existing implementations, e.g. [`OpenAIModel`](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/models/openai.py).

For details on when we'll accept contributions adding new models to PydanticAI, see the [contributing guidelines](https://ai.pydantic.dev/contributing/#new-model-rules).

If a model API is compatible with the OpenAI API, you do not need a custom model class and can provide your own [custom provider](https://ai.pydantic.dev/models/openai/#openai-compatible-models) instead.

## Fallback Model

You can use [`FallbackModel`](https://ai.pydantic.dev/api/models/fallback/#pydantic_ai.models.fallback.FallbackModel) to attempt multiple models
in sequence until one successfully returns a result. Under the hood, PydanticAI automatically switches
from one model to the next if the current model returns a 4xx or 5xx status code.

In the following example, the agent first makes a request to the OpenAI model (which fails due to an invalid API key),
and then falls back to the Anthropic model.

fallback\_model.py

```
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIModel

openai_model = OpenAIModel('gpt-4o')
anthropic_model = AnthropicModel('claude-3-5-sonnet-latest')
fallback_model = FallbackModel(openai_model, anthropic_model)

agent = Agent(fallback_model)
response = agent.run_sync('What is the capital of France?')
print(response.data)
#> Paris

print(response.all_messages())
"""
[\
    ModelRequest(\
        parts=[\
            UserPromptPart(\
                content='What is the capital of France?',\
                timestamp=datetime.datetime(...),\
                part_kind='user-prompt',\
            )\
        ],\
        kind='request',\
    ),\
    ModelResponse(\
        parts=[TextPart(content='Paris', part_kind='text')],\
        model_name='claude-3-5-sonnet-latest',\
        timestamp=datetime.datetime(...),\
        kind='response',\
        vendor_id=None,\
    ),\
]
"""

```

The `ModelResponse` message above indicates in the `model_name` field that the output was returned by the Anthropic model, which is the second model specified in the `FallbackModel`.

Each model's options should be configured individually. For example, `base_url`, `api_key`, and custom clients should be set on each model itself, not on the `FallbackModel`.

In this next example, we demonstrate the exception-handling capabilities of `FallbackModel`.
If all models fail, a [`FallbackExceptionGroup`](https://ai.pydantic.dev/api/exceptions/#pydantic_ai.exceptions.FallbackExceptionGroup) is raised, which
contains all the exceptions encountered during the `run` execution.

fallback\_model\_failure.py

```
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIModel

openai_model = OpenAIModel('gpt-4o')
anthropic_model = AnthropicModel('claude-3-5-sonnet-latest')
fallback_model = FallbackModel(openai_model, anthropic_model)

agent = Agent(fallback_model)
try:
    response = agent.run_sync('What is the capital of France?')
except* ModelHTTPError as exc_group:
    for exc in exc_group.exceptions:
        print(exc)

```

Since [`except*`](https://docs.python.org/3/reference/compound_stmts.html#except-star) is only supported
in Python 3.11+, we use the [`exceptiongroup`](https://github.com/agronholm/exceptiongroup) backport
package for earlier Python versions:

fallback\_model\_failure.py

```
from exceptiongroup import catch

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIModel

def model_status_error_handler(exc_group: BaseExceptionGroup) -> None:
    for exc in exc_group.exceptions:
        print(exc)

openai_model = OpenAIModel('gpt-4o')
anthropic_model = AnthropicModel('claude-3-5-sonnet-latest')
fallback_model = FallbackModel(openai_model, anthropic_model)

agent = Agent(fallback_model)
with catch({ModelHTTPError: model_status_error_handler}):
    response = agent.run_sync('What is the capital of France?')

```

By default, the `FallbackModel` only moves on to the next model if the current model raises a
[`ModelHTTPError`](https://ai.pydantic.dev/api/exceptions/#pydantic_ai.exceptions.ModelHTTPError). You can customize this behavior by
passing a custom `fallback_on` argument to the `FallbackModel` constructor.

</details>

<details>
<summary>YAML vs. JSON: Which Is More Efficient for Language Models?</summary>

# YAML vs. JSON: Which Is More Efficient for Language Models?

https://miro.medium.com/v2/resize:fit:700/1*2d8fldvpjiS1paVntoBPYQ.png

Illustration by author. Supercharge your language models: Slash costs by 50% and boost response time 2.5X by switching from JSON to YAML!

In early 2020, I had the unique opportunity to gain access to OpenAI’s GPT-3, a cutting-edge language model that seemed to possess almost magical capabilities. As I delved deeper into the technology, I discovered numerous ways to leverage its power in my personal and professional life, utilizing it as a life hack to expedite tasks and uncover novel concepts.

I quickly realized that working with GPT was not as intuitive as I had initially anticipated. Despite the introduction of ChatGPT, which aimed to bridge the gap and make this groundbreaking technology accessible to a wider audience, users still need a comprehensive understanding of how to maximize the potential of this innovative tool.

Over the past few months, I have conversed with numerous engineers and entrepreneurs who incorporate language models into their services and products. A recurring theme I observed was the attempt to solicit responses from language models in a JSON format. However, I discovered considerable consequences on output quality due to wording, prompt structure, and instructions. These factors can significantly impact a user’s ability to control and fine-tune the output generated by GPT and similar language models.

My intuition from my experiments was that JSON wasn’t an efficient format to ask from a language model for various reasons:

1. Syntax issues: JSON is a sensitive format for quotes, commas, and other reserved symbols, which makes it difficult for language models to follow instructions consistently.
2. Prefix and suffix in the response: Language models tend to wrap the output with unnecessary texts.
3. Excessive costs: JSON format requires opening and closing tags, producing excessive text characters, and increasing the overall tokens and your costs.
4. Excessive execution time: Using language models as part of your application, especially if it’s customer-facing, can be very sensitive to response time. Due to all of the above points, JSON can result in slow and flaky results, which can impact your user experience.

# Empirical Experiments

After sharing my advice about JSON vs YAML a few times, I conducted an empirical study to prove my assumptions.

In order to test how GPT efficiency when it parses text of the same content, I asked GPT to generate a simple list of month names in JSON format and compared it to YAML format and compared using the [Tokenizer tool by OpenAI](https://platform.openai.com/tokenizer) (more about tokens later). This simple example demonstrated about a 50% reduction in costs when using YAML:

https://miro.medium.com/v2/resize:fit:1000/1*Bo5esVY0YsMBQDwURq_YBw.png

The YAML approach here saved 48% in tokens and 25% in characters.

It is clear that YAML is significantly more cost/time-effective than JSON in those cases.

# Deeper Look

Now, let’s look deeper into bigger completion performance time and the penalty for parsing the output as JSON or YAML.

For parsing, I suggest using the [js-yaml](https://www.npmjs.com/package/js-yaml) package for parsing the output into JS objects and [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation) for Python.

I’ve used this prompt to generate a somewhat deterministic test set with a predefined structure and measured results on various completion sizes (x5, x10, and x45, which consumed the whole tokens window):

`Generate basic demographic info about 10 top countries (by population). Should include those fields: country, population, capital, official_language, currency, area_km, gdp_usd, under the root "countries". Output in {{format}} format, reduce other prose.`(format: YAML\|JSON)

Here’s the results I got:

https://miro.medium.com/v2/resize:fit:700/1*_V4DYpfWgStvme6joDbBSg.png

YAML tended to be faster and had a smaller footprint, but the gap degrades when getting closer to max token limit

https://miro.medium.com/v2/resize:fit:700/1*vtMqARGmCh--YIKwI9tvSw.png

Comparing YAML diffs over response length (left) and runtime/tokens (right)

The final [JSON](https://gist.github.com/Livshitz/aa30b7ed96f0310c22f104202c7df776) and [YAML](https://gist.github.com/Livshitz/878f1a596df9eabcd41897cb10eee78a) outputs can be found in the GH gist, accordingly.

If you were using this prompt on the scale of 1 million requests per month using JSON and GPT-4, switching to YAML would result in saving 190 tokens and would save you $11,400 (based on the pricing on this paper’s day) per month with this simple trick.

# Why Does This Happen?

To understand why this happens, we need to understand how language models process text into tokens and tokens back into text.

Language models are machine learning models, and machines don’t really understand “words” as a whole text, so words have to be encoded into a representation that machines can process. Each word could be represented by a unique ID, which is a machine-friendly representation. This is usually referred to as “Index-Based Encoding.” Though it is somewhat inefficient as words with multiple variations like “fun,” “funny,” and “funniest” are semantically close, they will be represented in totally different and distinct IDs.

In 1994, Philip Gage introduced a new data compression technique that replaces common pairs of consecutive bytes with a byte that does not appear in that data. In other words, by splitting words into parts, we could yet represent words by unique token IDs and still store and retrieve them efficiently. This technique is called Byte Pair Encoding (BPE) and is used as subword tokenization. This technique has become the foundation for models such as [BERT](https://github.com/google-research/bert), [GPT](https://openai.com/blog/better-language-models/) models, [RoBERTa](https://arxiv.org/abs/1907.11692), and more.

To properly handle the token “est,” for example, in the cases of “estimate” and “highest” (“est” appears at the beginning or the end but has different meanings), BPE attempts to combine pairs of two bytes or parts of words.

More on how GPT-3 tokens work is described well by Piotr Grudzien [here](https://blog.quickchat.ai/post/tokens-entropy-question/).

Using the [Tokenizer tool by OpenAI](https://platform.openai.com/tokenizer), it can be demonstrated as follows:

https://miro.medium.com/v2/resize:fit:700/1*BytpkdynzqJoZPNY5lq98Q.png

BPE breaking words during subword tokenization

When this concept comes with single characters, such as curly brackets, we see something interesting:

https://miro.medium.com/v2/resize:fit:700/1*-SyvXsNMBxAJHyg_xT5GYw.png

Although we see the same character, BPE decides to categorize them differently

This fundamental behavior alone plays well in how YAML is structured (line breaks and spaces as special characters, without the need to open and close curly brackets, quotes, and commas) compared to JSON, which requires opening and closing tags. Opening and closing tags impact the underlying representation in tokens, eventually causing extra LLM spins and might impact the general ability to follow instructions. So, not only does this save characters, but it also generally helps language models represent words with token IDs that are more common in their BPE vocabulary.

https://miro.medium.com/v2/resize:fit:1000/1*0cYldFGYCDl7mWRUZw2iuw.png

In comparing JSON and YAML, it is evident that the distribution of tokens in JSON is non-consistent, whereas YAML presents a more organized structure. This theoretically enhances the LLM’s capacity to allocate more spins on content rather than focusing on structural aspects, consequently improving the overall output quality.

In conclusion, while JSON is generally faster to parse and consume than YAML, YAML is significantly more cost/time-efficient than JSON and can help language models produce precisely the same content faster and cheaper. Essentially, it is more efficient to request YAML, and convert the result to JSON on the code-side, instead of requesting JSON directly.

It is worth mentioning that the potential compromise might be the strictness of JSON for some formats (numbers could be printed as strings, surrounded with quotes). This can be solved by providing schema or post-parsing the fields into the right data type. Regardless, it could be good practice anyway to enforce data type conversions on code-side.

# **Appendix- Chain-of-Thought using YAML comments:**

In addition to its advantages in speed and cost, YAML offers another significant benefit over JSON — the capacity to include comments.

Take this classic test case from “ [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)” ( [Wei et al. ,2022](https://arxiv.org/abs/2201.11903)):

https://miro.medium.com/v2/resize:fit:700/0*kioxp_e0umir87iU

Imagine you want this output in machine-readable format.

With JSON and no CoT, you’ll get bad results:

https://miro.medium.com/v2/resize:fit:700/1*FvaohbxdpfAFgmDR6rQlQQ.png

No CoT, JSON return, GPT-3.5. Wrong answer, should return 900030

However, by utilizing YAML, you can define a format that accommodates the CoT within comments while presenting the final answer in the assigned key, ultimately producing a parseable output:

https://miro.medium.com/v2/resize:fit:700/1*-PxoVjKFNxO7CCiGe6HwYQ.png

CoT with YAML comments, GPT-3.5, CORRECT answer

</details>


## Code Sources

<details>
<summary>Repository analysis for https://github.com/towardsai/course-ai-agents/blob/main/lessons/05_structured_outputs/notebook.ipynb</summary>

# Repository analysis for https://github.com/towardsai/course-ai-agents/blob/main/lessons/05_structured_outputs/notebook.ipynb

## Summary
Repository: towardsai/course-ai-agents
File: notebook.ipynb
Lines: 601

Estimated tokens: 3.7k

## File tree
```Directory structure:
└── notebook.ipynb

```

## Extracted content
================================================
FILE: lessons/05_structured_outputs/notebook.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Lesson 5: Structured Outputs

This notebook explores **Structured Outputs** - a powerful feature for building capable AI agents with Large Language Models (LLMs). We will use the `google-genai` library to interact with Google's Gemini models and learn different approaches to enforcing structured data formats.

**Learning Objectives:**

1.  **Understand structured outputs** and why they are crucial for reliable data extraction from LLMs.
2.  **Enforce structured data formats (JSON)** from an LLM using prompt engineering techniques.
3.  **Leverage Pydantic models** to define and manage complex data structures for structured outputs, improving code robustness and clarity.
4.  **Use Gemini's native structured output capabilities** for the most reliable and efficient approach.
"""

"""
## 1. Setup

First, let's install the necessary Python libraries.
"""

"""
!pip install -q google-genai pydantic python-dotenv
"""

"""
### Configure Gemini API Key

To use the Gemini API, you need an API key. 

1.  Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  Create a file named `.env` in the root of this project.
3.  Add the following line to the `.env` file, replacing `your_api_key_here` with your actual key:
    ```
    GOOGLE_API_KEY="your_api_key_here"
    ```
The code below will load this key from the `.env` file.
"""

%load_ext autoreload
%autoreload 2
# Output:
#   The autoreload extension is already loaded. To reload it, use:

#     %reload_ext autoreload


from lessons.utils import env

env.load(required_env_vars=["GOOGLE_API_KEY"])
# Output:
#   Trying to load environment variables from `/Users/pauliusztin/Documents/01_projects/TAI/course-ai-agents/.env`

#   Environment variables loaded successfully.


"""
### Import Key Packages
"""

import json

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from lessons.utils import pretty_print

"""
### Initialize the Gemini Client as Our LLM Provider
"""

client = genai.Client()

"""
### Define Constants

We will use the `gemini-2.5-flash` model, which is fast, cost-effective, and supports advanced features like structured outputs.
"""

MODEL_ID = "gemini-2.5-flash"

"""
## 2. Implementing structured outputs from scratch using JSON

Sometimes, you don't need the LLM to take an action, but you need its output in a specific, machine-readable format. Forcing the output to be JSON is a common way to achieve this.

We can instruct the model to do this by **prompting** clearly describing the desired JSON structure in the prompt.
"""

"""
### Example: Extracting Metadata from a Document

Let's imagine we have a markdown document and we want to extract key information like a summary, tags, and keywords into a clean JSON object.
"""

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

prompt = f"""
Analyze the following document and extract metadata from it. 
The output must be a single, valid JSON object with the following structure:
<json>
{{ 
    "summary": "A concise summary of the article.", 
    "tags": ["list", "of", "relevant", "tags"], 
    "keywords": ["list", "of", "key", "concepts"],
    "quarter": "Q3",
    "growth_rate": "15%",
}}
</json>

Here is the document:
<document>
{DOCUMENT}
</document>
"""

response = client.models.generate_content(model=MODEL_ID, contents=prompt)

pretty_print.wrapped_print(text=response.text, title="Raw LLM Output", indent=2)
# Output:
#   [93m------------------------------------------ Raw LLM Output ------------------------------------------[0m

#     ```json

#   {

#       "summary": "The Q3 2023 earnings report showcases strong financial performance, exceeding market expectations with a 20% revenue increase and 15% growth in user engagement. This success is attributed to effective product strategy, strong market positioning, and successful expansion into new markets, particularly in digital services. The company also improved customer acquisition costs by 10% and achieved a 92% retention rate, indicating a solid foundation for future growth.",

#       "tags": [

#           "Financial Performance",

#           "Q3 2023",

#           "Earnings Report",

#           "Business Growth",

#           "Market Expansion",

#           "User Engagement",

#           "Customer Retention"

#       ],

#       "keywords": [

#           "revenue increase",

#           "user engagement growth",

#           "product strategy",

#           "market positioning",

#           "digital services",

#           "new markets",

#           "customer acquisition costs",

#           "retention rates",

#           "cash flow",

#           "Q3 2023"

#       ],

#       "quarter": "Q3",

#       "growth_rate": "15%"

#   }

#   ```

#   [93m----------------------------------------------------------------------------------------------------[0m


def extract_json_from_response(response: str) -> dict:
    """
    Extracts JSON from a response string that is wrapped in <json> or ```json tags.
    """

    response = response.replace("<json>", "").replace("</json>", "")
    response = response.replace("```json", "").replace("```", "")

    return json.loads(response)

"""
You can now reliably parse the JSON string:
"""

parsed_repsonse = extract_json_from_response(response.text)
pretty_print.wrapped_print(
    text=[f"Type of the parsed response: `{type(parsed_repsonse)}`", json.dumps(parsed_repsonse, indent=2)],
    title="Parsed JSON Object",
    indent=2,
)
# Output:
#   [93m---------------------------------------- Parsed JSON Object ----------------------------------------[0m

#     Type of the parsed response: `<class 'dict'>`

#   [93m----------------------------------------------------------------------------------------------------[0m

#     {

#     "summary": "The Q3 2023 earnings report showcases strong financial performance, exceeding market expectations with a 20% revenue increase and 15% growth in user engagement. This success is attributed to effective product strategy, strong market positioning, and successful expansion into new markets, particularly in digital services. The company also improved customer acquisition costs by 10% and achieved a 92% retention rate, indicating a solid foundation for future growth.",

#     "tags": [

#       "Financial Performance",

#       "Q3 2023",

#       "Earnings Report",

#       "Business Growth",

#       "Market Expansion",

#       "User Engagement",

#       "Customer Retention"

#     ],

#     "keywords": [

#       "revenue increase",

#       "user engagement growth",

#       "product strategy",

#       "market positioning",

#       "digital services",

#       "new markets",

#       "customer acquisition costs",

#       "retention rates",

#       "cash flow",

#       "Q3 2023"

#     ],

#     "quarter": "Q3",

#     "growth_rate": "15%"

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
## 3. Implementing structured outputs from scratch using Pydantic

While prompting for JSON is effective, it can be fragile. A more robust and modern approach is to use **Pydantic**. Pydantic allows you to define data structures as Python classes. This gives you:

- **A single source of truth**: The Pydantic model defines the structure.
- **Automatic schema generation**: You can easily generate a JSON Schema from the model.
- **Data validation**: You can validate the LLM's output against the model to ensure it conforms to the expected structure and types.

Let's recreate the previous example using Pydantic.
"""

class DocumentMetadata(BaseModel):
    """A class to hold structured metadata for a document."""

    summary: str = Field(description="A concise, 1-2 sentence summary of the document.")
    tags: list[str] = Field(description="A list of 3-5 high-level tags relevant to the document.")
    keywords: list[str] = Field(description="A list of specific keywords or concepts mentioned.")
    quarter: str = Field(description="The quarter of the financial year described in the document (e.g, Q3 2023).")
    growth_rate: str = Field(description="The growth rate of the company described in the document (e.g, 10%).")

"""
### Injecting Pydantic Schema into the Prompt

We can generate a JSON Schema from our Pydantic model and inject it directly into the prompt. This is a more formal way of telling the LLM what structure to follow.

Note how, along with the field type, we can leverage the Field description automatically to clearly specify to the LLM what each field means.
"""

schema = DocumentMetadata.model_json_schema()
schema
# Output:
#   {'description': 'A class to hold structured metadata for a document.',

#    'properties': {'summary': {'description': 'A concise, 1-2 sentence summary of the document.',

#      'title': 'Summary',

#      'type': 'string'},

#     'tags': {'description': 'A list of 3-5 high-level tags relevant to the document.',

#      'items': {'type': 'string'},

#      'title': 'Tags',

#      'type': 'array'},

#     'keywords': {'description': 'A list of specific keywords or concepts mentioned.',

#      'items': {'type': 'string'},

#      'title': 'Keywords',

#      'type': 'array'},

#     'quarter': {'description': 'The quarter of the financial year described in the document (e.g, Q3 2023).',

#      'title': 'Quarter',

#      'type': 'string'},

#     'growth_rate': {'description': 'The growth rate of the company described in the document (e.g, 10%).',

#      'title': 'Growth Rate',

#      'type': 'string'}},

#    'required': ['summary', 'tags', 'keywords', 'quarter', 'growth_rate'],

#    'title': 'DocumentMetadata',

#    'type': 'object'}

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

parsed_repsonse = extract_json_from_response(response.text)

pretty_print.wrapped_print(
    text=[f"Type of the parsed response: `{type(parsed_repsonse)}`", json.dumps(parsed_repsonse, indent=2)],
    title="Parsed JSON Object",
    indent=2,
)
# Output:
#   [93m---------------------------------------- Parsed JSON Object ----------------------------------------[0m

#     Type of the parsed response: `<class 'dict'>`

#   [93m----------------------------------------------------------------------------------------------------[0m

#     {

#     "summary": "The Q3 2023 earnings report indicates strong financial performance with a 20% revenue increase and 15% growth in user engagement, surpassing market expectations. This success is attributed to robust digital services growth, successful market expansion, reduced customer acquisition costs, and improved retention rates.",

#     "tags": [

#       "Financial Performance",

#       "Earnings Report",

#       "Business Growth",

#       "Revenue Analysis",

#       "Market Expansion"

#     ],

#     "keywords": [

#       "Q3 2023",

#       "revenue increase",

#       "user engagement",

#       "market expectations",

#       "product strategy",

#       "market positioning",

#       "digital services",

#       "new markets",

#       "customer acquisition costs",

#       "retention rates",

#       "cash flow"

#     ],

#     "quarter": "Q3 2023",

#     "growth_rate": "20%"

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
As you can see, conceptually, the results are the same. But now, we can easily validate the output with Pydantic:
"""

try:
    document_metadata = DocumentMetadata.model_validate(parsed_repsonse)
    print("\nValidation successful!")

    pretty_print.wrapped_print(
        ["Type of the validated response: `{type(document_metadata)}`", document_metadata.model_dump_json(indent=2)],
        title="Pydantic Validated Object",
        indent=2,
    )
except Exception as e:
    print(f"\nValidation failed: {e}")
# Output:
#   

#   Validation successful!

#   [93m------------------------------------ Pydantic Validated Object ------------------------------------[0m

#     Type of the validated response: `{type(document_metadata)}`

#   [93m----------------------------------------------------------------------------------------------------[0m

#     {

#     "summary": "The Q3 2023 earnings report indicates strong financial performance with a 20% revenue increase and 15% growth in user engagement, surpassing market expectations. This success is attributed to robust digital services growth, successful market expansion, reduced customer acquisition costs, and improved retention rates.",

#     "tags": [

#       "Financial Performance",

#       "Earnings Report",

#       "Business Growth",

#       "Revenue Analysis",

#       "Market Expansion"

#     ],

#     "keywords": [

#       "Q3 2023",

#       "revenue increase",

#       "user engagement",

#       "market expectations",

#       "product strategy",

#       "market positioning",

#       "digital services",

#       "new markets",

#       "customer acquisition costs",

#       "retention rates",

#       "cash flow"

#     ],

#     "quarter": "Q3 2023",

#     "growth_rate": "20%"

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
## 4. Implementing structured outputs using Gemini and Pydantic

Using Gemini's `GenerateContentConfig` we can enforce the output as a Pydantic object without any special prompt engineering.

We can instruct the model to do this by setting `response_mime_type` to `"application/json"` in the generation configuration, which forces the model's output to be a valid JSON object and the `response_schema` to our Pydantic object.

**Note:** If you use only the `response_mime_type="application/json"` setting you can output raw JSON formats.
"""

config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=DocumentMetadata)

prompt = f"""
Analyze the following document and extract its metadata.

Document:
--- 
{DOCUMENT}
--- 
"""

response = client.models.generate_content(model=MODEL_ID, contents=prompt, config=config)
pretty_print.wrapped_print(
    [f"Type of the response: `{type(response.parsed)}`", response.parsed.model_dump_json(indent=2)],
    title="Pydantic Validated Object",
    indent=2,
)
# Output:
#   [93m------------------------------------ Pydantic Validated Object ------------------------------------[0m

#     Type of the response: `<class '__main__.DocumentMetadata'>`

#   [93m----------------------------------------------------------------------------------------------------[0m

#     {

#     "summary": "The Q3 2023 earnings report reveals a 20% increase in revenue and 15% growth in user engagement, surpassing market expectations due to successful product strategy and market expansion. The company demonstrated strong performance with improved customer acquisition costs and retention rates.",

#     "tags": [

#       "Financial Performance",

#       "Earnings Report",

#       "Business Growth",

#       "Market Expansion",

#       "Customer Metrics"

#     ],

#     "keywords": [

#       "Q3 2023",

#       "revenue increase",

#       "user engagement",

#       "product strategy",

#       "market positioning",

#       "digital services",

#       "new markets",

#       "customer acquisition costs",

#       "retention rates",

#       "cash flow"

#     ],

#     "quarter": "Q3 2023",

#     "growth_rate": "20%"

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
From now on, throughout this course, this native Gemini approach will be how we implement structured outputs to achieve the most reliable and efficient results.
"""

</details>


## YouTube Video Transcripts

<details>
<summary>All right, folks. In this video, what we're going to be doing is we're going to be discussing OpenAI function calling and structured prompting. Now, the idea here is that we want to first discuss what function calling is and how OpenAI implements it. And then discuss structured prompting using the Pydantic library as an example on how that can work in terms of writing code and what does it mean to structure the prompt, et cetera. Right? So,</summary>

All right, folks. In this video, what we're going to be doing is we're going to be discussing OpenAI function calling and structured prompting. Now, the idea here is that we want to first discuss what function calling is and how OpenAI implements it. And then discuss structured prompting using the Pydantic library as an example on how that can work in terms of writing code and what does it mean to structure the prompt, et cetera. Right? So,

[00:30] Um, what we're going to be talking about first is function calling. So, function calling means connecting large language models like ChatGPT to tools that perform actions in the outside world, like checking the weather or searching the web, right? And OpenAI function calling involves four simple steps. We call the model with the query and a set of functions that are defined in the functions parameter. So we explain to the model, these are the tools that are available to you.

[01:00] Then, step number two, the model can choose whether or not to call one or more functions that it has available to it. And the content of that call will be a stringified JSON object that adheres to a custom schema, right? And actually, this has now become a an industry standard. So, we then parse the string into JSON in the code and call the function of the provided arguments if they exist. And then we call the model again by appending the function response as a new message, and let the model summarize the results back to the user.

[01:30] So essentially we say the model, look, these are the tools that you have available. We have a structured way to connect the model to those functions and to send inputs to those functions. Once it has identified through reading, through processing the prompt that a function should be called to solve the problem. And then the model sends the correct arguments to that function, calls the function, the output which is called an observation is integrated into a response that gets summarized to the user.

[02:00] Right? So, we're going to be seeing an example here in Python code on how to do that. I'm going to initialize my OpenAI client. We're going to have, I'm going to have a very simple function that creates a directory in my current folder. And then I'm going to write it as in the JSON schema for the OpenAI function calling API. So, it will have a dictionary with the type, the function, the name of the function, the description, the parameters. Within the parameters, each, the type of the parameter, the object, the properties of that function, of the arguments. So, this is directory, it is a type of string and describes what that uh parameter does. And we set up also a key called required which indicates which arguments are required in that particular function.

[02:46] And then we put this function definition for the OpenAI function calling API inside of a list, which is pretty cool. Now what we have is this little function called run terminal task. We create a variable called messages and inside that variable, we give it a list with the prompts to the model. In this case, we're just saying uh create a folder called Lucas Loves LLMs, which is, you know, why not?

[03:15] And then we set up the tools inside of a list. We call the model. We're calling the GPT 3.5 Turbo 16k. We give the messages parameter that will contain this message with our prompt to the model. We set up the tools and we set up the tool choice to automatic, so that the model can automatically choose to call a function or not. Then we gather the response, and we identify, we we check whether or not tool calls were made in that response, right?

[03:46] And if they were made, what we do is we have a dictionary with the available functions that the model can use. And then we append that response to the messages object, to the messages list. We loop over the tool calls that were made. We gather the name, the function, the arguments of the function, and we call the tool getting the function response, right? We append everything under the messages list, and we call the model with all of that information to integrate and summarize the response, which is then returned to the user like we are doing right here, and we get the output, right?

[04:27] So when I call, we'll get a chat completion object like this. And if we inspect the string that was returned, here we say Lucas Loves LLMs has been created, the folder has been created. And if I check my current folders, we see that the folder was indeed created, which is awesome, right? Now, folks, this is great.

[04:50] Function calling is amazing, right? Function calling introduces this idea of trying to add structure and determinism to the process of interacting with large language models, right? And in the theme of that structured interaction with large language models, a library that has been extremely popular in not only in Python, but now in the large language model universe in terms of frameworks, is a library called Pydantic.

[05:22] Now, what this, uh, this is a data validation library in Python that allows you, that allows us to do some pretty interesting stuff. So essentially, what it allows us to do is, it allows us to give, uh, set up data structures that we can have, and when connected with the OpenAI function calling API, Pydantic allows us to define specifically what is the object we want returned when we prompt the model with something.

[05:57] So you will understand that in a second. So what we're going to be doing is, in addition to OpenAI and Pydantic, we're also going to use the instructor package for this demonstration. And we're going to set up two classes in Pydantic. We're going to set up a class called question that contains three attributes. The question attribute that holds the quiz question text, the options for that question, imagine multiple choice, and the correct answer as an integer that refers to the index of the correct answer in the options list, all right?

[06:40] So, uh, having done that, what we're going to do is, we're going to have a second class called quiz that contains the topic in question for this quiz that we want to create from a web page article or a paper. And a list of questions, which in each of those elements inside of this list will have the an object of the question type.

[07:06] So folks, this is a lot of information, but what we're doing here is, we're setting up data types, right? And we're constructing these customizable, these custom data types with Pydantic. And why this is so cool, because it allows us to prompt the model like ChatGPT, and ask for that model to create something structured out of a prompt that was made in natural language.

[07:32] So I can say, uh, so let's understand that by in practice. So, I'm going to set up my client and now to interact with this, um, and connect OpenAI function calling and the Pydantic API, we're going to be using the instructor package. So, I'm going to set up the client with the instructor.from_openai method, and then I'm going to give the OpenAI client to that method.

[07:56] And then I'm going to define a function called generate a quiz which calls the ChatGPT API with the chat.completions.create method. It sets up the model as GPT-4 Turbo. And then it sets up the messages list, and in that messages list, we feed it a dictionary containing the system message in which we say you're a quiz generation engine specialized in fostering understanding in students given a source of content to be studied.

[08:30] You will be fed content like articles or scientific papers and you will output quizzes aimed at eliciting full understanding of the material, right? Pretty cool system message. And then we're going to give the prompt to the model. And the prompt is going to contain a prompt, right? Just like we've talked about in the initial lesson for this, uh, video course about prompt engineering. But it will also contain the contents of the article or paper.

[08:59] So, uh, in the prompt, we're going to say, I want you to do the following. Identify the main topic of the following article. Just like we've discussed in the beginning of this series, we're breaking the problem down into tasks, right? So identify the main topic of the following article. And then I give all the contents of the following article under the delimiter, uh, quotes, so that we organize what is input text, remember? And what is the instruction?

[09:29] Then, for the second step, I want the model to create a quiz with five questions that revolve around the main topic and can be answered simply by carefully reading and understanding the article, because I want the questions to be grounded on the reference text. Remember our best practices video where we talked about the strategy of grounding questions in, you know, grounding answers in reference text.

[09:56] So that's what we're doing here to create this quiz. One of these questions should check if the student understood the main ideas by testing if the student can transfer its knowledge in a different context. Because the idea with this quiz is to have a quiz that's comprehensive and helps the student learn, um, something new, right? And then we give our little output indicator, which we just say output colon, right? So, when I call this, we can take a look at the output. And what's interesting about this output is, and we'll see it in just a second, is that we will see the structure that we defined with using the Pydantic library.

[10:40] And that's what makes this approach a structured prompting approach, because we're, uh, getting an output that has structured, right? And obviously, we could talk about structured prompting as adding structure in the prompt itself. But when I say structured prompting in this context, I mean,

[11:02] using libraries like the Pydantic OpenAI API to add structure to the output that we get from OpenAI or from ChatGPT. So, as we see here, the output is a quiz object, which is the object that we've defined in the beginning. And it has a topic, it has a list of questions, and each question here is going to be of that question object that we've defined earlier as well.

[11:28] So, what I can do is, I can loop over each question. So I can say for q in quiz_output. dot questions. So for q in and then we can print q.

[11:47] question as well as print q. and as well as loop over the, um, options in that question. So for, So we can say for I, O in enumerate,

[12:08] Q, we can say Q. options. Uh, we can print i,o, we can print the option, and then at the end, we can print the correct answer by saying correct answer, and then here we can say q.correct_answer, which actually is not q.correct answer, it is q.question, q.correct answer, yeah, I think that's correct.

[12:43] Perfect. So now we get the question, we get the options, and we get the correct answer, which, if you ask me, this is a pretty cool application for large language model, as well as for a structured prompting approach that leverages OpenAI function calling, Pydantic, and that's it for this video.

[13:02] And see you in the next video.

</details>


## Additional Sources Scraped

<details>
<summary>a-practical-guide-on-structuring-llm-outputs-with-pydantic</summary>

Integrating large language models (LLMs) into production systems often reveals a fundamental challenge: their outputs are inherently unstructured and unpredictable.

Whether it's missing fields, malformed formats, or incorrect data types, these inconsistencies hinder reliability and scalability.

**The solution?** Leverage [Pydantic](https://docs.pydantic.dev/?ref=developer-service.blog), a Python library that enables runtime data validation using type annotations.

With LLMs like MistralAI (and many others) supporting structured outputs via JSON schemas, combining these tools ensures AI-generated data adheres to strict schemas.

In this guide, we’ll walk through a simple but practical real-world example that:

- Uses the Mistral API to generate structured JSON from a CSV input.
- Validates that output using a `Pydantic` model.
- Implements a retry mechanism for failed validation attempts with an improved prompt.

## Understanding Pydantic

Pydantic is a powerful data validation and parsing library in Python, built around the concept of using standard Python type hints to define data models.

At its core, Pydantic enforces that incoming data matches the specified schema, automatically converting types and raising errors when expectations aren't met.

Originally developed for use with web frameworks like FastAPI, Pydantic has found widespread adoption in domains where data integrity and clarity are critical, including AI and machine learning workflows.

By turning Python classes into data contracts, Pydantic helps eliminate the guesswork often associated with dynamic or external inputs.

### Key Features

- **Runtime Type Checking**: Pydantic enforces type annotations at runtime, ensuring that all incoming data adheres to the expected types. If a mismatch occurs, it raises detailed validation errors that are easy to debug.
- **Automatic Data Parsing and Serialization**: Whether you receive input as strings, dictionaries, or nested structures, Pydantic will automatically parse and coerce data into the appropriate Python objects. It can also serialize models back to JSON or dictionaries for API responses or storage.
- **Integration with Python Type Hints**: Models are defined using familiar Python syntax with type annotations, making it intuitive for developers to describe complex data shapes. This also enables static analysis tools and IDEs to provide better support and autocomplete suggestions.

## The Importance of Structured Outputs in AI

There are three main points when defining the importance of structured outputs for AI responses.

### Consistency

Structured outputs provide a consistent format for AI-generated data, which is essential for seamless downstream processing.

When data adheres to a predefined schema, it becomes straightforward to parse, transform, and integrate into various systems such as databases, APIs, or analytic pipelines.

Consistency eliminates guesswork, reduces the need for custom error-prone parsing logic, and enables automation at scale.

### Reliability

AI models, LLMs, can generate diverse and unpredictable outputs.

This variability can lead to failures if systems expect data in a specific format but receive something unexpected instead.

By enforcing structure through validation, the risk of runtime errors, crashes, or corrupted data is significantly reduced.

Reliable data outputs increase confidence in the AI system’s behavior, making it safer to deploy in production environments.

### Security

Unvalidated or poorly structured inputs and outputs can expose applications to security vulnerabilities such as injection attacks, malformed data exploitation, or denial-of-service scenarios.

Structured data validation acts as a safeguard, ensuring that only well-formed, type-safe data is accepted and processed.

This reduces the attack surface and helps maintain the integrity and confidentiality of AI-driven systems.

## Practical Example: From CSV to Validated JSON

Let's consider the example of processing a CSV file containing data about users, which can have some incomplete data, into structured JSON representing user profiles.

In terms of flow, we will implement this logic:

https://developer-service.blog/content/images/2025/06/image.pngPydantic AI output validation workflow design

### Define Pydantic Model

``` python
import os
import json
from pydantic import BaseModel, ValidationError
from mistralai import Mistral

# -----------------------------
# Pydantic Model for Validation
# -----------------------------
class Person(BaseModel):
    name: str
    age: int
    email: str
```

Here we define a simple Pydantic model to ensure the schema of our intended JSON.

### Function to Call MistralAI API with JSON Mode

``` python
# --------------------------------------
# Function to Call Mistral in JSON Mode
# --------------------------------------
def call_mistral_json_mode(prompt: str, system_message: str = "") -> str:
    """Call the Mistral API with a prompt and optional system message, expecting a JSON object response."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the MISTRAL_API_KEY environment variable.")
    model = "mistral-large-latest"
    client = Mistral(api_key=api_key)
    messages = [\
        {"role": "user", "content": prompt},\
        {"role": "system", "content": system_message},\
    ]
    chat_response = client.chat.complete(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return chat_response.choices[0].message.content
```

Here, we are calling the Mistral API with the model `mistral-large-latest` and enforcing the response to be a `json_object`.

**Note:** Mistral AI provides a `chat.parse` method that receives a Pydantic model directly as the `response_format`. For this example, I kept the logic generic so it can be used with other LLMs.

### Read CSV file

``` python
# -----------------------------
# Read CSV Input from File
# -----------------------------
with open("example_incomplete.csv", "r", encoding="utf-8") as f:
    csv_input = f.read().strip()
```

Here, we simply read the CSV into a variable.

The CSV used in this example is:

``` csv
name,age,email
Alice,30,alice@example.com
Bob,,bob@example.com
Charlie,40,
Diana,25,diana@example.com
```

### Initial Prompt and AI Response

``` python
# -----------------------------
# Initial Prompt Construction
# -----------------------------
model_json_schema = Person.model_json_schema()
prompt = f"""
            Given the following CSV data, return a JSON array of objects with fields: {Person.model_json_schema()}

            CSV:
            {csv_input}

            Example output:
            [\
            {{"name": "Alice", "age": 30, "email": "alice@example.com"}},\
            {{"name": "Bob", "age": 25, "email": "bob@example.com"}}\
            ]
        """

print("\n" + "="*50)
print("Mistral CSV to Structured Example: Attempt 1")
print("="*50 + "\n")
response = call_mistral_json_mode(prompt)
print("Mistral response:\n", response)
```

</details>

<details>
<summary>how-to-return-structured-data-from-a-model-langchain</summary>

Prerequisites

This guide assumes familiarity with the following concepts:

- [Chat models](https://python.langchain.com/docs/concepts/chat_models/)
- [Function/tool calling](https://python.langchain.com/docs/concepts/tool_calling/)

It is often useful to have a model return output that matches a specific [schema](https://python.langchain.com/docs/concepts/structured_outputs/). One common use-case is extracting data from text to insert into a database or use with some other downstream system. This guide covers a few strategies for getting structured outputs from a model.

## The `.with_structured_output()` method

Supported models

You can find a [list of models that support this method here](https://python.langchain.com/docs/integrations/chat/).

This is the easiest and most reliable way to get structured outputs. `with_structured_output()` is implemented for [models that provide native APIs for structuring outputs](https://python.langchain.com/docs/integrations/chat/), like tool/function calling or JSON mode, and makes use of these capabilities under the hood.

This method takes a schema as input which specifies the names, types, and descriptions of the desired output attributes. The method returns a model-like Runnable, except that instead of outputting strings or [messages](https://python.langchain.com/docs/concepts/messages/) it outputs objects corresponding to the given schema. The schema can be specified as a TypedDict class, [JSON Schema](https://json-schema.org/) or a Pydantic class. If TypedDict or JSON Schema are used then a dictionary will be returned by the Runnable, and if a Pydantic class is used then a Pydantic object will be returned.

As an example, let's get a model to generate a joke and separate the setup from the punchline:

```python
pip install -qU "langchain[google-genai]"
```

```python
import getpass
import os

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

from langchain.chat_models import init_chat_model

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
```

### Pydantic class

If we want the model to return a Pydantic object, we just need to pass in the desired Pydantic class. The key advantage of using Pydantic is that the model-generated output will be validated. Pydantic will raise an error if any required fields are missing or if any fields are of the wrong type.

```python
from typing import Optional

from pydantic import BaseModel, Field

# Pydantic
class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )

structured_llm = llm.with_structured_output(Joke)

structured_llm.invoke("Tell me a joke about cats")
```

```python
Joke(setup='Why was the cat sitting on the computer?', punchline='Because it wanted to keep an eye on the mouse!', rating=7)
```

Beyond just the structure of the Pydantic class, the name of the Pydantic class, the docstring, and the names and provided descriptions of parameters are very important. Most of the time `with_structured_output` is using a model's function/tool calling API, and you can effectively think of all of this information as being added to the model prompt.

### TypedDict or JSON Schema

If you don't want to use Pydantic, explicitly don't want validation of the arguments, or want to be able to stream the model outputs, you can define your schema using a TypedDict class. We can optionally use a special `Annotated` syntax supported by LangChain that allows you to specify the default value and description of a field. Note, the default value is _not_ filled in automatically if the model doesn't generate it, it is only used in defining the schema that is passed to the model.

Requirements

- Core: `langchain-core>=0.2.26`
- Typing extensions: It is highly recommended to import `Annotated` and `TypedDict` from `typing_extensions` instead of `typing` to ensure consistent behavior across Python versions.

```python
from typing import Optional

from typing_extensions import Annotated, TypedDict

# TypedDict
class Joke(TypedDict):
    """Joke to tell user."""

    setup: Annotated[str, ..., "The setup of the joke"]

    # Alternatively, we could have specified setup as:

    # setup: str                    # no default, no description
    # setup: Annotated[str, ...]    # no default, no description
    # setup: Annotated[str, "foo"]  # default, no description

    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]

structured_llm = llm.with_structured_output(Joke)

structured_llm.invoke("Tell me a joke about cats")
```

```python
{'setup': 'Why was the cat sitting on the computer?',
 'punchline': 'Because it wanted to keep an eye on the mouse!',
 'rating': 7}
```

Equivalently, we can pass in a [JSON Schema](https://json-schema.org/) dict. This requires no imports or classes and makes it very clear exactly how each parameter is documented, at the cost of being a bit more verbose.

```python
json_schema = {
    "title": "joke",
    "description": "Joke to tell user.",
    "type": "object",
    "properties": {
        "setup": {
            "type": "string",
            "description": "The setup of the joke",
        },
        "punchline": {
            "type": "string",
            "description": "The punchline to the joke",
        },
        "rating": {
            "type": "integer",
            "description": "How funny the joke is, from 1 to 10",
            "default": None,
        },
    },
    "required": ["setup", "punchline"],
}
structured_llm = llm.with_structured_output(json_schema)

structured_llm.invoke("Tell me a joke about cats")
```

```python
{'setup': 'Why was the cat sitting on the computer?',
 'punchline': 'Because it wanted to keep an eye on the mouse!',
 'rating': 7}
```

### Choosing between multiple schemas

The simplest way to let the model choose from multiple schemas is to create a parent schema that has a Union-typed attribute.

#### Using Pydantic

```python
from typing import Union

class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )

class ConversationalResponse(BaseModel):
    """Respond in a conversational manner. Be kind and helpful."""

    response: str = Field(description="A conversational response to the user's query")

class FinalResponse(BaseModel):
    final_output: Union[Joke, ConversationalResponse]

structured_llm = llm.with_structured_output(FinalResponse)

structured_llm.invoke("Tell me a joke about cats")
```

```python
FinalResponse(final_output=Joke(setup='Why was the cat sitting on the computer?', punchline='Because it wanted to keep an eye on the mouse!', rating=7))
```

```python
structured_llm.invoke("How are you today?")
```

```python
FinalResponse(final_output=ConversationalResponse(response="I'm just a computer program, so I don't have feelings, but I'm here and ready to help you with whatever you need!"))
```

#### Using TypedDict

```python
from typing import Optional, Union

from typing_extensions import Annotated, TypedDict

class Joke(TypedDict):
    """Joke to tell user."""

    setup: Annotated[str, ..., "The setup of the joke"]
    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]

class ConversationalResponse(TypedDict):
    """Respond in a conversational manner. Be kind and helpful."""

    response: Annotated[str, ..., "A conversational response to the user's query"]

class FinalResponse(TypedDict):
    final_output: Union[Joke, ConversationalResponse]

structured_llm = llm.with_structured_output(FinalResponse)

structured_llm.invoke("Tell me a joke about cats")
```

```python
{'final_output': {'setup': 'Why was the cat sitting on the computer?',
  'punchline': 'Because it wanted to keep an eye on the mouse!',
  'rating': 7}}
```

```python
structured_llm.invoke("How are you today?")
```

```python
{'final_output': {'response': "I'm just a computer program, so I don't have feelings, but I'm here and ready to help you with whatever you need!"}}
```

Alternatively, you can use tool calling directly to allow the model to choose between options, if your [chosen model supports it](https://python.langchain.com/docs/integrations/chat/). This involves a bit more parsing and setup but in some instances leads to better performance because you don't have to use nested schemas. See [this how-to guide](https://python.langchain.com/docs/how_to/tool_calling/) for more details.

### Streaming

We can stream outputs from our structured model when the output type is a dict (i.e., when the schema is specified as a TypedDict class or JSON Schema dict).

```python
from typing_extensions import Annotated, TypedDict

# TypedDict
class Joke(TypedDict):
    """Joke to tell user."""

    setup: Annotated[str, ..., "The setup of the joke"]
    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]

structured_llm = llm.with_structured_output(Joke)

for chunk in structured_llm.stream("Tell me a joke about cats"):
    print(chunk)
```

```python
{}
{'setup': ''}
{'setup': 'Why'}
{'setup': 'Why was'}
{'setup': 'Why was the'}
{'setup': 'Why was the cat'}
{'setup': 'Why was the cat sitting'}
{'setup': 'Why was the cat sitting on'}
{'setup': 'Why was the cat sitting on the'}
{'setup': 'Why was the cat sitting on the computer'}
{'setup': 'Why was the cat sitting on the computer?'}
{'setup': 'Why was the cat sitting on the computer?', 'punchline': ''}
{'setup': 'Why was the cat sitting on the computer?', 'punchline': 'Because'}
{'setup': 'Why was the cat sitting on the computer?', 'punchline': 'Because it'}
{'setup': 'Why was the cat sitting on the computer?', 'punchline': 'Because it wanted'}
{'setup': 'Why was the cat sitting on the computer?', 'punchline': 'Because it wanted to'}
{'setup': 'Why was the cat sitting on the computer?', 'punchline': 'Because it wanted to keep'}
{'setup': 'Why was the cat sitting on the computer?', 'punchline': 'Because it wanted to keep an'}
{'setup': 'Why was the cat sitting on the computer?', 'punchline': 'Because it wanted to keep an eye'}
{'setup': 'Why was the cat sitting on the computer?', 'punchline': 'Because it wanted to keep an eye on'}
{'setup': 'Why was the cat sitting on the computer?', 'punchline': 'Because it wanted to keep an eye on the'}
{'setup': 'Why was the cat sitting on the computer?', 'punchline': 'Because it wanted to keep an eye on the mouse'}
{'setup': 'Why was the cat sitting on the computer?', 'punchline': 'Because it wanted to keep an eye on the mouse!'}
{'setup': 'Why was the cat sitting on the computer?', 'punchline': 'Because it wanted to keep an eye on the mouse!', 'rating': 7}
```

### Few-shot prompting

For more complex schemas it's very useful to add few-shot examples to the prompt. This can be done in a few ways.

The simplest and most universal way is to add examples to a system message in the prompt:

```python
from langchain_core.prompts import ChatPromptTemplate

system = """You are a hilarious comedian. Your specialty is knock-knock jokes. \
Return a joke which has the setup (the response to "Who's there?") and the final punchline (the response to "<setup> who?").

Here are some examples of jokes:

example_user: Tell me a joke about planes
example_assistant: {{"setup": "Why don't planes ever get tired?", "punchline": "Because they have rest wings!", "rating": 2}}

example_user: Tell me another joke about planes
example_assistant: {{"setup": "Cargo", "punchline": "Cargo 'vroom vroom', but planes go 'zoom zoom'!", "rating": 10}}

example_user: Now about caterpillars
example_assistant: {{"setup": "Caterpillar", "punchline": "Caterpillar really slow, but watch me turn into a butterfly and steal the show!", "rating": 5}}"""

prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{input}")])

few_shot_structured_llm = prompt | structured_llm
few_shot_structured_llm.invoke("what's something funny about woodpeckers")
```

```python
{'setup': 'Woodpecker',
 'punchline': "Woodpecker you a joke, but I'm afraid it might be too 'hole-some'!",
 'rating': 7}
```

When the underlying method for structuring outputs is tool calling, we can pass in our examples as explicit tool calls. You can check if the model you're using makes use of tool calling in its API reference.

```python
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

examples = [\
    HumanMessage("Tell me a joke about planes", name="example_user"),\
    AIMessage(\
        "",\
        name="example_assistant",\
        tool_calls=[\
            {\
                "name": "joke",\
                "args": {\
                    "setup": "Why don't planes ever get tired?",\
                    "punchline": "Because they have rest wings!",\
                    "rating": 2,\
                },\
                "id": "1",\
            }\
        ],\
    ),\
    # Most tool-calling models expect a ToolMessage(s) to follow an AIMessage with tool calls.\
    ToolMessage("", tool_call_id="1"),\
    # Some models also expect an AIMessage to follow any ToolMessages,\
    # so you may need to add an AIMessage here.\
    HumanMessage("Tell me another joke about planes", name="example_user"),\
    AIMessage(\
        "",\
        name="example_assistant",\
        tool_calls=[\
            {\
                "name": "joke",\
                "args": {\
                    "setup": "Cargo",\
                    "punchline": "Cargo 'vroom vroom', but planes go 'zoom zoom'!",\
                    "rating": 10,\
                },\
                "id": "2",\
            }\
        ],\
    ),\
    ToolMessage("", tool_call_id="2"),\
    HumanMessage("Now about caterpillars", name="example_user"),\
    AIMessage(\
        "",\
        tool_calls=[\
            {\
                "name": "joke",\
                "args": {\
                    "setup": "Caterpillar",\
                    "punchline": "Caterpillar really slow, but watch me turn into a butterfly and steal the show!",\
                    "rating": 5,\
                },\
                "id": "3",\
            }\
        ],\
    ),\
    ToolMessage("", tool_call_id="3"),\
]
system = """You are a hilarious comedian. Your specialty is knock-knock jokes. \
Return a joke which has the setup (the response to "Who's there?") \
and the final punchline (the response to "<setup> who?")."""

prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("placeholder", "{examples}"), ("human", "{input}")]
)
few_shot_structured_llm = prompt | structured_llm
few_shot_structured_llm.invoke({"input": "crocodiles", "examples": examples})
```

```python
{'setup': 'Crocodile',
 'punchline': 'Crocodile be seeing you later, alligator!',
 'rating': 6}
```

For more on few shot prompting when using tool calling, see [here](https://python.langchain.com/docs/how_to/tools_few_shot/).

### (Advanced) Specifying the method for structuring outputs

For models that support more than one means of structuring outputs (i.e., they support both tool calling and JSON mode), you can specify which method to use with the `method=` argument.

JSON mode

If using JSON mode you'll have to still specify the desired schema in the model prompt. The schema you pass to `with_structured_output` will only be used for parsing the model outputs, it will not be passed to the model the way it is with tool calling.

To see if the model you're using supports JSON mode, check its entry in the [API reference](https://python.langchain.com/api_reference/langchain/index.html).

```python
structured_llm = llm.with_structured_output(None, method="json_mode")

structured_llm.invoke(
    "Tell me a joke about cats, respond in JSON with `setup` and `punchline` keys"
)
```

```python
{'setup': 'Why was the cat sitting on the computer?',
 'punchline': 'Because it wanted to keep an eye on the mouse!'}
```

### (Advanced) Raw outputs

LLMs aren't perfect at generating structured output, especially as schemas become complex. You can avoid raising exceptions and handle the raw output yourself by passing `include_raw=True`. This changes the output format to contain the raw message output, the `parsed` value (if successful), and any resulting errors:

```python
structured_llm = llm.with_structured_output(Joke, include_raw=True)

structured_llm.invoke("Tell me a joke about cats")
```

```python
{'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_f25ZRmh8u5vHlOWfTUw8sJFZ', 'function': {'arguments': '{"setup":"Why was the cat sitting on the computer?","punchline":"Because it wanted to keep an eye on the mouse!","rating":7}', 'name': 'Joke'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 33, 'prompt_tokens': 93, 'total_tokens': 126}, 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_4e2b2da518', 'finish_reason': 'stop', 'logprobs': None}, id='run-d880d7e2-df08-4e9e-ad92-dfc29f2fd52f-0', tool_calls=[{'name': 'Joke', 'args': {'setup': 'Why was the cat sitting on the computer?', 'punchline': 'Because it wanted to keep an eye on the mouse!', 'rating': 7}, 'id': 'call_f25ZRmh8u5vHlOWfTUw8sJFZ', 'type': 'tool_call'}], usage_metadata={'input_tokens': 93, 'output_tokens': 33, 'total_tokens': 126}),
 'parsed': {'setup': 'Why was the cat sitting on the computer?',
  'punchline': 'Because it wanted to keep an eye on the mouse!',
  'rating': 7},
 'parsing_error': None}
```

## Prompting and parsing model outputs directly

Not all models support `.with_structured_output()`, since not all models have tool calling or JSON mode support. For such models you'll need to directly prompt the model to use a specific format, and use an output parser to extract the structured response from the raw model output.

### Using `PydanticOutputParser`

The following example uses the built-in [`PydanticOutputParser`](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.pydantic.PydanticOutputParser.html) to parse the output of a chat model prompted to match the given Pydantic schema. Note that we are adding `format_instructions` directly to the prompt from a method on the parser:

```python
from typing import List

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class Person(BaseModel):
    """Information about a person."""

    name: str = Field(..., description="The name of the person")
    height_in_meters: float = Field(
        ..., description="The height of the person expressed in meters."
    )

class People(BaseModel):
    """Identifying information about all people in a text."""

    people: List[Person]

# Set up a parser
parser = PydanticOutputParser(pydantic_object=People)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [\
        (\
            "system",\
            "Answer the user query. Wrap the output in `json` tags\n{format_instructions}",\
        ),\
        ("human", "{query}"),\
    ]
).partial(format_instructions=parser.get_format_instructions())
```

Let’s take a look at what information is sent to the model:

```python
query = "Anna is 23 years old and she is 6 feet tall"

print(prompt.invoke({"query": query}).to_string())
```

```python
System: Answer the user query. Wrap the output in `json` tags
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema:
```
{"description": "Identifying information about all people in a text.", "properties": {"people": {"title": "People", "type": "array", "items": {"$ref": "#/definitions/Person"}}}, "required": ["people"], "definitions": {"Person": {"title": "Person", "description": "Information about a person.", "type": "object", "properties": {"name": {"title": "Name", "description": "The name of the person", "type": "string"}, "height_in_meters": {"title": "Height In Meters", "description": "The height of the person expressed in meters.", "type": "number"}}, "required": ["name", "height_in_meters"]}}}
```
Human: Anna is 23 years old and she is 6 feet tall
```

And now let's invoke it:

```python
chain = prompt | llm | parser

chain.invoke({"query": query})
```

```python
People(people=[Person(name='Anna', height_in_meters=1.8288)])
```

For a deeper dive into using output parsers with prompting techniques for structured output, see [this guide](https://python.langchain.com/docs/how_to/output_parser_structured/).

### Custom Parsing

You can also create a custom prompt and parser with [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/concepts/lcel/), using a plain function to parse the output from the model:

```python
import json
import re
from typing import List

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class Person(BaseModel):
    """Information about a person."""

    name: str = Field(..., description="The name of the person")
    height_in_meters: float = Field(
        ..., description="The height of the person expressed in meters."
    )

class People(BaseModel):
    """Identifying information about all people in a text."""

    people: List[Person]

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [\
        (\
            "system",\
            "Answer the user query. Output your answer as JSON that  "\
            "matches the given schema: ```json\n{schema}\n```. "\
            "Make sure to wrap the answer in ```json and ``` tags",\
        ),\
        ("human", "{query}"),\
    ]
).partial(schema=People.schema())

# Custom parser
def extract_json(message: AIMessage) -> List[dict]:
    """Extracts JSON content from a string where JSON is embedded between ```json and ``` tags.

    Parameters:
        text (str): The text containing the JSON content.

    Returns:
        list: A list of extracted JSON strings.
    """
    text = message.content
    # Define the regular expression pattern to match JSON blocks
    pattern = r"```json(.*?)```"

    # Find all non-overlapping matches of the pattern in the string
    matches = re.findall(pattern, text, re.DOTALL)

    # Return the list of matched JSON strings, stripping any leading or trailing whitespace
    try:
        return [json.loads(match.strip()) for match in matches]
    except Exception:
        raise ValueError(f"Failed to parse: {message}")
```

Here is the prompt sent to the model:

```python
query = "Anna is 23 years old and she is 6 feet tall"

print(prompt.format_prompt(query=query).to_string())
```

```python
System: Answer the user query. Output your answer as JSON that  matches the given schema: ```json
{'title': 'People', 'description': 'Identifying information about all people in a text.', 'type': 'object', 'properties': {'people': {'title': 'People', 'type': 'array', 'items': {'$ref': '#/definitions/Person'}}}, 'required': ['people'], 'definitions': {'Person': {'title': 'Person', 'description': 'Information about a person.', 'type': 'object', 'properties': {'name': {'title': 'Name', 'description': 'The name of the person', 'type': 'string'}, 'height_in_meters': {'title': 'Height In Meters', 'description': 'The height of the person expressed in meters.', 'type': 'number'}}, 'required': ['name', 'height_in_meters']}}}
```. Make sure to wrap the answer in ```json and ``` tags
Human: Anna is 23 years old and she is 6 feet tall
```

And here's what it looks like when we invoke it:

```python
chain = prompt | llm | extract_json

chain.invoke({"query": query})
```

```python
[{'people': [{'name': 'Anna', 'height_in_meters': 1.8288}]}]
```

</details>

<details>
<summary>steering-large-language-models-with-pydantic-pydantic</summary>

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

</details>

<details>
<summary>structured-output-gemini-api-google-ai-for-developers</summary>

# Structured output

You can configure Gemini for structured output instead of unstructured text,
allowing precise extraction and standardization of information for further processing.
For example, you can use structured output to extract information from resumes,
standardize them to build a structured database.

Gemini can generate either [JSON](https://ai.google.dev/gemini-api/docs/structured-output#generating-json)
or [enum values](https://ai.google.dev/gemini-api/docs/structured-output#generating-enums) as structured output.

## Generating JSON

There are two ways to generate JSON using the Gemini API:

- Configure a schema on the model
- Provide a schema in a text prompt

Configuring a schema on the model is the **recommended** way to generate JSON,
because it constrains the model to output JSON.

### Configuring a schema (recommended)

To constrain the model to generate JSON, configure a `responseSchema`. The model
will then respond to any prompt with JSON-formatted output.

```
from google import genai
from pydantic import BaseModel

class Recipe(BaseModel):
    recipe_name: str
    ingredients: list[str]

client = genai.Client()
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="List a few popular cookie recipes, and include the amounts of ingredients.",
    config={
        "response_mime_type": "application/json",
        "response_schema": list[Recipe],
    },
)
# Use the response as a JSON string.
print(response.text)

# Use instantiated objects.
my_recipes: list[Recipe] = response.parsed

```

```
import { GoogleGenAI, Type } from "@google/genai";

const ai = new GoogleGenAI({});

async function main() {
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents:
      "List a few popular cookie recipes, and include the amounts of ingredients.",
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            recipeName: {
              type: Type.STRING,
            },
            ingredients: {
              type: Type.ARRAY,
              items: {
                type: Type.STRING,
              },
            },
          },
          propertyOrdering: ["recipeName", "ingredients"],
        },
      },
    },
  });

  console.log(response.text);
}

main();

```

```
package main

import (
    "context"
    "fmt"
    "log"

    "google.golang.org/genai"
)

func main() {
    ctx := context.Background()
    client, err := genai.NewClient(ctx, nil)
    if err != nil {
        log.Fatal(err)
    }

    config := &genai.GenerateContentConfig{
        ResponseMIMEType: "application/json",
        ResponseSchema: &genai.Schema{
            Type: genai.TypeArray,
            Items: &genai.Schema{
                Type: genai.TypeObject,
                Properties: map[string]*genai.Schema{
                    "recipeName": {Type: genai.TypeString},
                    "ingredients": {
                        Type:  genai.TypeArray,
                        Items: &genai.Schema{Type: genai.TypeString},
                    },
                },
                PropertyOrdering: []string{"recipeName", "ingredients"},
            },
        },
    }

    result, err := client.Models.GenerateContent(
        ctx,
        "gemini-2.5-flash",
        genai.Text("List a few popular cookie recipes, and include the amounts of ingredients."),
        config,
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(result.Text())
}

```

```
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent" \
-H "x-goog-api-key: $GEMINI_API_KEY" \
-H 'Content-Type: application/json' \
-d '{
      "contents": [{\
        "parts":[\
          { "text": "List a few popular cookie recipes, and include the amounts of ingredients." }\
        ]\
      }],
      "generationConfig": {
        "responseMimeType": "application/json",
        "responseSchema": {
          "type": "ARRAY",
          "items": {
            "type": "OBJECT",
            "properties": {
              "recipeName": { "type": "STRING" },
              "ingredients": {
                "type": "ARRAY",
                "items": { "type": "STRING" }
              }
            },
            "propertyOrdering": ["recipeName", "ingredients"]
          }
        }
      }
}' 2> /dev/null | head

```

The output might look like this:

```
[\
  {\
    "recipeName": "Chocolate Chip Cookies",\
    "ingredients": [\
      "1 cup (2 sticks) unsalted butter, softened",\
      "3/4 cup granulated sugar",\
      "3/4 cup packed brown sugar",\
      "1 teaspoon vanilla extract",\
      "2 large eggs",\
      "2 1/4 cups all-purpose flour",\
      "1 teaspoon baking soda",\
      "1 teaspoon salt",\
      "2 cups chocolate chips"\
    ]\
  },\
  ...\
]

```

### Providing a schema in a text prompt

Instead of configuring a schema, you can supply a schema as natural language or
pseudo-code in a text prompt. This method is **not recommended**, because it
might produce lower quality output, and because the model is not constrained to
follow the schema.

Here's a generic example of a schema provided in a text prompt:

```
List a few popular cookie recipes, and include the amounts of ingredients.

Produce JSON matching this specification:

Recipe = { "recipeName": string, "ingredients": array<string> }
Return: array<Recipe>

```

Since the model gets the schema from text in the prompt, you might have some
flexibility in how you represent the schema. But when you supply a schema inline
like this, the model is not actually constrained to return JSON. For a more
deterministic, higher quality response, configure a schema on the model, and
don't duplicate the schema in the text prompt.

## Generating enum values

In some cases you might want the model to choose a single option from a list of
options. To implement this behavior, you can pass an _enum_ in your schema. You
can use an enum option anywhere you could use a `string` in the
`responseSchema`, because an enum is an array of strings. Like a JSON schema, an
enum lets you constrain model output to meet the requirements of your
application.

For example, assume that you're developing an application to classify
musical instruments into one of five categories: `"Percussion"`, `"String"`,
`"Woodwind"`, `"Brass"`, or " `"Keyboard"`". You could create an enum to help
with this task.

In the following example, you pass an enum as the
`responseSchema`, constraining the model to choose the most appropriate option.

```
from google import genai
import enum

class Instrument(enum.Enum):
  PERCUSSION = "Percussion"
  STRING = "String"
  WOODWIND = "Woodwind"
  BRASS = "Brass"
  KEYBOARD = "Keyboard"

client = genai.Client()
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='What type of instrument is an oboe?',
    config={
        'response_mime_type': 'text/x.enum',
        'response_schema': Instrument,
    },
)

print(response.text)
# Woodwind

```

```
import { GoogleGenAI, Type } from "@google/genai";

const ai = new GoogleGenAI({});

const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: "What type of instrument is an oboe?",
    config: {
      responseMimeType: "text/x.enum",
      responseSchema: {
        type: Type.STRING,
        enum: ["Percussion", "String", "Woodwind", "Brass", "Keyboard"],
      },
    },
  });

console.log(response.text);

```

```
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent" \
-H "x-goog-api-key: $GEMINI_API_KEY" \
    -H 'Content-Type: application/json' \
    -d '{
          "contents": [{\
            "parts":[\
              { "text": "What type of instrument is an oboe?" }\
            ]\
          }],
          "generationConfig": {
            "responseMimeType": "text/x.enum",
            "responseSchema": {
              "type": "STRING",
              "enum": ["Percussion", "String", "Woodwind", "Brass", "Keyboard"]
            }
          }
    }'

```

The Python library will translate the type declarations for the API. However,
the API accepts a subset of the OpenAPI 3.0 schema
( [Schema](https://ai.google.dev/api/caching#schema)).

There are two other ways to specify an enumeration. You can use a
[`Literal`](https://docs.pydantic.dev/1.10/usage/types/#literal-type):
```
Literal["Percussion", "String", "Woodwind", "Brass", "Keyboard"]
```

And you can also pass the schema as JSON:

```
from google import genai

client = genai.Client()
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='What type of instrument is an oboe?',
    config={
        'response_mime_type': 'text/x.enum',
        'response_schema': {
            "type": "STRING",
            "enum": ["Percussion", "String", "Woodwind", "Brass", "Keyboard"],
        },
    },
)

print(response.text)
# Woodwind

```

Beyond basic multiple choice problems, you can use an enum anywhere in a JSON
schema. For example, you could ask the model for a list of recipe titles and
use a `Grade` enum to give each title a popularity grade:

```
from google import genai

import enum
from pydantic import BaseModel

class Grade(enum.Enum):
    A_PLUS = "a+"
    A = "a"
    B = "b"
    C = "c"
    D = "d"
    F = "f"

class Recipe(BaseModel):
  recipe_name: str
  rating: Grade

client = genai.Client()
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='List 10 home-baked cookie recipes and give them grades based on tastiness.',
    config={
        'response_mime_type': 'application/json',
        'response_schema': list[Recipe],
    },
)

print(response.text)

```

The response might look like this:

```
[\
  {\
    "recipe_name": "Chocolate Chip Cookies",\
    "rating": "a+"\
  },\
  {\
    "recipe_name": "Peanut Butter Cookies",\
    "rating": "a"\
  },\
  {\
    "recipe_name": "Oatmeal Raisin Cookies",\
    "rating": "b"\
  },\
  ...\
]

```

## About JSON schemas

Configuring the model for JSON output using `responseSchema` parameter relies on
`Schema` object to define its structure. This object represents a select
subset of the [OpenAPI 3.0 Schema object](https://spec.openapis.org/oas/v3.0.3#schema-object),
and also adds a `propertyOrdering` field.

Here's a pseudo-JSON representation of all the `Schema` fields:

```
{
  "type": enum (Type),
  "format": string,
  "description": string,
  "nullable": boolean,
  "enum": [\
    string\
  ],
  "maxItems": integer,
  "minItems": integer,
  "properties": {
    string: {
      object (Schema)
    },
    ...
  },
  "required": [\
    string\
  ],
  "propertyOrdering": [\
    string\
  ],
  "items": {
    object (Schema)
  }
}

```

The `Type` of the schema must be one of the OpenAPI
[Data Types](https://spec.openapis.org/oas/v3.0.3#data-types), or a union of
those types (using `anyOf`). Only a subset of fields is valid for each `Type`.
The following list maps each `Type` to a subset of the fields that are valid for
that type:

- `string` -\> `enum`, `format`, `nullable`
- `integer` -\> `format`, `minimum`, `maximum`, `enum`, `nullable`
- `number` -\> `format`, `minimum`, `maximum`, `enum`, `nullable`
- `boolean` -\> `nullable`
- `array` -\> `minItems`, `maxItems`, `items`, `nullable`
- `object` -\> `properties`, `required`, `propertyOrdering`, `nullable`

Here are some example schemas showing valid type-and-field combinations:

```
{ "type": "string", "enum": ["a", "b", "c"] }

{ "type": "string", "format": "date-time" }

{ "type": "integer", "format": "int64" }

{ "type": "number", "format": "double" }

{ "type": "boolean" }

{ "type": "array", "minItems": 3, "maxItems": 3, "items": { "type": ... } }

{ "type": "object",
  "properties": {
    "a": { "type": ... },
    "b": { "type": ... },
    "c": { "type": ... }
  },
  "nullable": true,
  "required": ["c"],
  "propertyOrdering": ["c", "b", "a"]
}

```

For complete documentation of the Schema fields as they're used in the Gemini
API, see the [Schema reference](https://ai.google.dev/api/caching#Schema).

### Property ordering

When you're working with JSON schemas in the Gemini API, the order of properties
is important. By default, the API orders properties alphabetically and does not
preserve the order in which the properties are defined (although the
[Google Gen AI SDKs](https://ai.google.dev/gemini-api/docs/sdks) may preserve this order). If you're
providing examples to the model with a schema configured, and the property
ordering of the examples is not consistent with the property ordering of the
schema, the output could be rambling or unexpected.

To ensure a consistent, predictable ordering of properties, you can use the
optional `propertyOrdering[]` field.

```
"propertyOrdering": ["recipeName", "ingredients"]

```

`propertyOrdering[]` – not a standard field in the OpenAPI specification
– is an array of strings used to determine the order of properties in the
response. By specifying the order of properties and then providing examples with
properties in that same order, you can potentially improve the quality of
results. `propertyOrdering` is only supported when you manually create
`types.Schema`.

### Schemas in Python

When you're using the Python library, the value of `response_schema` must be one
of the following:

- A type, as you would use in a type annotation (see the Python
[`typing` module](https://docs.python.org/3/library/typing.html))
- An instance of
[`genai.types.Schema`](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema)
- The `dict` equivalent of `genai.types.Schema`

The easiest way to define a schema is with a Pydantic type (as shown in the
previous example):

```
config={'response_mime_type': 'application/json',
        'response_schema': list[Recipe]}

```

When you use a Pydantic type, the Python library builds out a JSON schema for
you and sends it to the API. For additional examples, see the
[Python library docs](https://googleapis.github.io/python-genai/index.html#json-response-schema).

The Python library supports schemas defined with the following types (where
`AllowedType` is any allowed type):

- `int`
- `float`
- `bool`
- `str`
- `list[AllowedType]`
- `AllowedType|AllowedType|...`
- For structured types:
  - `dict[str, AllowedType]`. This annotation declares all dict values to
    be the same type, but doesn't specify what keys should be included.
  - User-defined
    [Pydantic models](https://docs.pydantic.dev/latest/concepts/models/). This
    approach lets you specify the key names and define different types for the
    values associated with each of the keys, including nested structures.

### JSON Schema support

[JSON Schema](https://json-schema.org/) is a more recent specification than
OpenAPI 3.0, which the [Schema](https://ai.google.dev/api/caching#Schema) object is based on.
Support for JSON Schema is available as a preview using the field
[`responseJsonSchema`](https://ai.google.dev/api/generate-content#FIELDS.response_json_schema) which
accepts any JSON Schema with the following limitations:

- It only works with Gemini 2.5.
- While all JSON Schema properties can be passed, not all are supported. See
the [documentation](https://ai.google.dev/api/generate-content#FIELDS.response_json_schema) for the
field for more details.
- Recursive references can only be used as the value of a non-required object
property.
- Recursive references are unrolled to a finite degree, based on the size of
the schema.
- Schemas that contain `$ref` cannot contain any properties other than those
starting with a `$`.

Here's an example of generating a JSON Schema with Pydantic and submitting it to
the model:

```
curl "https://generativelanguage.googleapis.com/v1alpha/models/\
gemini-2.5-flash:generateContent" \
    -H "x-goog-api-key: $GEMINI_API_KEY"\
    -H 'Content-Type: application/json' \
    -d @- <<EOF
{
  "contents": [{\
    "parts":[{\
      "text": "Please give a random example following this schema"\
    }]\
  }],
  "generationConfig": {
    "response_mime_type": "application/json",
    "response_json_schema": $(python3 - << PYEOF
from enum import Enum
from typing import List, Optional, Union, Set
from pydantic import BaseModel, Field, ConfigDict
import json

class UserRole(str, Enum):
    ADMIN = "admin"
    VIEWER = "viewer"

class Address(BaseModel):
    street: str
    city: str

class UserProfile(BaseModel):
    username: str = Field(description="User's unique name")
    age: Optional[int] = Field(ge=0, le=120)
    roles: Set[UserRole] = Field(min_items=1)
    contact: Union[Address, str]
    model_config = ConfigDict(title="User Schema")

# Generate and print the JSON Schema
print(json.dumps(UserProfile.model_json_schema(), indent=2))
PYEOF
)
  }
}
EOF

```

Passing JSON Schema directly is not yet supported when using the SDK.

## Best practices

Keep the following considerations and best practices in mind when you're using a
response schema:

- The size of your response schema counts towards the input token limit.
- By default, fields are optional, meaning the model can populate the fields or
skip them. You can set fields as required to force the model to provide a
value. If there's insufficient context in the associated input prompt, the
model generates responses mainly based on the data it was trained on.
- A complex schema can result in an `InvalidArgument: 400` error. Complexity
might come from long property names, long array length limits, enums with
many values, objects with lots of optional properties, or a combination of
these factors.

If you get this error with a valid schema, make one or more of the following
changes to resolve the error:

  - Shorten property names or enum names.
  - Flatten nested arrays.
  - Reduce the number of properties with constraints, such as numbers with
    minimum and maximum limits.
  - Reduce the number of properties with complex constraints, such as
    properties with complex formats like `date-time`.
  - Reduce the number of optional properties.
  - Reduce the number of valid values for enums.
- If you aren't seeing the results you expect, add more context to your input
prompts or revise your response schema. For example, review the model's
response without structured output to see how the model responds. You can then
update your response schema so that it better fits the model's output.


## What's next

Now that you've learned how to generate structured output, you might want to try
using Gemini API tools:

- [Function calling](https://ai.google.dev/gemini-api/docs/function-calling)
- [Code execution](https://ai.google.dev/gemini-api/docs/code-execution)
- [Grounding with Google Search](https://ai.google.dev/gemini-api/docs/grounding)

</details>
