# Crafting JSON Outputs For Cntrolled Text Generation

This post explores four key approaches to generating valid JSON from LLMs—Langchain’s Pydantic parser, Jsonformer, Outlines, and Microsoft’s Guidance. It compares each method’s strengths, limitations, and use cases, offering a practical guide for reliable, structured LLM outputs.

Large Language Models (LLMs) have not only advanced the field of text generation but have also set higher standards for most NLP-related tasks. By being trained on huge volumes of textual data, these models possess the capability to comprehend and produce text that is often indistinguishable from human-written content. With their ability to generate coherent and contextually relevant responses, the first obvious application for these models is chatbots or creative writing assistants.

‍

## Beyond chatbots

Beyond chatbots, their capabilities have also been demonstrated across diverse NLP tasks, including language translation, text summarisation, question answering, and general content generation. For instance, in discriminative tasks such as document classification, sentiment analysis and named entity recognition, LLMs can be used employing [**in-context learning**](https://thegradient.pub/in-context-learning-in-context/) to get few-shot or even zero-shot classification. They are also widely used for data augmentation generating synthetic training data for traditional models \[ [**1**](https://www.semanticscholar.org/paper/Is-a-prompt-and-a-few-samples-all-you-need-Using-in-M%C3%B8ller-Dalsgaard/ca3037fed8ed14dea92985b9f288b05185f867d0)\], \[ [**2**](https://www.semanticscholar.org/paper/Do-Not-Have-Enough-Data-Deep-Learning-to-the-Anaby-Tavor-Carmeli/7eba731a7fd8de712b7b79b5af41a6e2d4dbd191))\], and \[ [**3**](https://www.semanticscholar.org/paper/Generating-Faithful-Synthetic-Data-with-Large-A-in-Veselovsky-Ribeiro/5af9cf0b695faf2eb94d74bf76dab1a311638ca3)\].

From chatbots that can engage in human-like conversations to language translation services that can provide real-time translations, LLMs have transformed the way we interact with software. However, they also present new challenges. Issues such as hallucination, fine-tuning limitations, context size, and memory add new layers of complexity for deterministic systems.

Although large language models excel at producing coherent responses, ensuring their outputs respect a specific format is not guaranteed. Consequently, this can present challenges when utilising the outputs of a language model as input for another system. Here we may want to get more structured information than just text back. To use them as software components we need a reliable interface to be able to connect with external tools. These integrations are possible with friendly serialisation formats like JSON or XML.

The scope of this blog post is to explore recent developments for controlled text generation, more specifically generating valid JSON outputs. Generating structured JSON from language models can be tricky and unreliable sometimes. The generated JSON must be syntactically correct, and it must conform to a schema that specifies the structure of the JSON. This allows developers to more reliably get structured data back from the model.

It is worth mentioning that Open AI has made the task easier by introducing the [**Function calling feature**](https://openai.com/blog/function-calling-and-other-api-updates) in June 2023. However, this feature is only suitable if interoperability with other models or providers is not a requirement. Our focus in this post will be on solutions that can be utilised with a broader range of generative models.

## Approach 1: Prompt engineering and output parsers

Output parsers process the language model response trying to fit it in the desired structure. We can see it as a postprocessing step. The one we will cover here is an [**implementation**](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/output_parsers/pydantic.py) provided by [**Langchain**](https://python.langchain.com/docs) that relies on the well-known [**Pydantic models**](https://docs.pydantic.dev/latest/usage/models/) to define the data structure. For the Langchain library, there are two main components an output parser must implement:

- **Format instructions**: A method that returns a string containing instructions for how the output of a language model should be formatted.
- **Parser**: A method that takes in a string (assumed to be the response from a language model) and parses it into some structure.

#### **Pydantic Output Parser**

This [**output parser**](https://python.langchain.com/docs/modules/model_io/output_parsers/pydantic) allows users to specify an arbitrary JSON schema and query LLMs for JSON outputs that conform to that schema. You’ll have to use an LLM with sufficient capacity to generate well-formed JSON. In the OpenAI family, GPT-3 DaVinci might be enough and GPT3 Curie’s ability already drops off dramatically.

Here is a simple example defining the object _Joke_:

The output format instructions will look like this:

All we have to do is to concatenate the instructions string into the final prompt.

Instantiate the LLM model and query it with the prompt.

The output JSON string will look like this:

Using the `PydanticOutputParser` instance we defined, we convert the JSON string into the `Joke` model.

The output will be an instance of the `Joke`:

#### **Benefits**

1. Automatically generates the output instructions.
2. Parses the generated JSON string into the model.
3. Easy interoperability. It can be used on any capable large language model.

#### **Limitations**

1. Using prompt instructions will not guarantee the desired output.
2. Less capable models might have difficulty following the instructions. You will certainly need a loop that tries again until the model generates a suitable candidate that is accepted by the parser.
3. The more complex the schema, the more chances for the model to make mistakes.
4. Less robust. The model will generate token by token, increasing the chance of making syntax mistakes, for instance failing to generate a valid JSON string.
5. Context size. Complex schemas might require too many tokens in the context, limiting the context size you will have available for the actual prompt and in some cases the output.

## Approach 2: Jsonformer

Approaches based on output parsing are error-prone and less robust. They rely on prompt engineering, fine-tuning, and post-processing, but they still can fail to generate syntactically correct JSON in some cases. Especially smaller and less capable LLMs.

In structured data, many tokens are fixed and predictable. Ideally we wouldn’t need to generate tokens like brackets and field names that we already know. [**_Jsonformer_**](https://github.com/1rgs/jsonformer/) is a wrapper around Hugging Face transformers that fills in these fixed tokens during the generation process and only delegates the generation of content tokens, the actual field values to the language model. This makes it more efficient and robust than prompt engineering and output parsers.

The idea behind this method is to guide the text generation monitoring the logits and tokens sampling.

While sampling the tokens it will check first if the token follows the expected requirements. Example:

- Generating **booleans** : compares logits for **true** and **false**
- Generating **numbers**: squash logits for non-digit tokens before sampling
- Generating **strings**: stops generation on second **”**
- Generating **arrays**: compare logits for “[“, “,”, “]”

Here is an example on how to generate JSON string that follows the specified JSON schema:

#### **Benefits**

1. This method works even with small and less capable models. A 3B parameter model, small for LLMs standards, is already capable of benefiting from this approach.
2. It supports JSON schemas with nested objects.
3. Efficiency. By generating only the content tokens and filling in the fixed tokens, this method is more efficient than generating a full JSON string and parsing it.
4. Flexible and extendable. This library is built on top of the Hugging Face transformers library, making it compatible with any model that supports the Hugging Face interface.

#### **Limitations**

1. The official implementation only offers support for HuggingFace transformers, which is where the **_Jsonformer_** name comes from. In theory, it can be adapted to work with OpenAI API or other libraries as long as they provide the logits. Some contributors have already [**proposed solutions**](https://github.com/1rgs/jsonformer/pull/16) but since OpenAI released Function Calling this is less relevant.
2. It currently only supports a limited subset of JSON Schema types (number, boolean, string, array, object).
3. It looks like the project is not actively maintained at the moment.

## Approach 3: Outlines

‍ [**_Outlines_**](https://github.com/outlines-dev/outlines) is a framework that provides methods for defining top-level restrictions on text completions generated by language models. We can limit the output tokens in terms of type constraints, a predefined list of accepted tokens, and even regular expressions. This [**paper**](https://arxiv.org/abs/2307.09702) details how authors reformulate the problem of text generation in terms of transitions between the states of a finite-state machine.

**Efficient JSON generation**

The implementation provided by **_Outlines_** allows a guide the generation process so the output is “guaranteed” to follow a JSON schema or [**Pydantic model**](https://docs.pydantic.dev/latest/):

The output sequence will look like:

Now let's parse the JSON string:

Parsed object:

#### **Benefits**

1. The method works with union types, optional types, arrays, nested schemas, etc. Some field constraints are [**not supported yet, here**](https://github.com/outlines-dev/outlines/issues/215) you can find the details.
2. The approach is model agnostic as long as you can mask the logits. Open-source LLMs can be used.
3. Integration with HuggingFace transformers models.
4. Project in active development.

#### **Limitations**

1. Currently, it does not support APIs like OpenAI due to API limitation. More details [**here**](https://github.com/outlines-dev/outlines/issues/227).

## Approach 4: Guidance

Microsoft [**_Guidance_**](https://github.com/guidance-ai/guidance/tree/main) is another promising framework that provides a Domain Specific Language (DSL) for prompting. It merges templating and logic control making it possible to have more complex and clever prompts.

With the template language is possible to ensure that generated JSON is always valid. Below we generate a random character profile for a game with perfect syntax every time. A Jupyter Notebook with the complete example is available [**here**](https://github.com/guidance-ai/guidance/blob/main/notebooks/guaranteeing_valid_syntax.ipynb).

In this example, we specify not only the JSON format but the constraints for each attribute using the DSL. Note that ‘ **_age_**’ must respect the regex pattern ‘ **_[0–9]+_**’ and ‘ **_weapon_**’ must accept only values from the ‘ **_valid_weapons_**’ list.

The output as simple Python dictionary:

#### **Benefits**

1. It provides a powerful Domain Specific Language for prompting making it easy to build the template for a JSON response.

#### **Limitations**

1. It doesn’t work well with small LLMs. We need at least Llama-7b to make it work.
2. Since it does not work with smaller models, it is very memory intensive.

## Wrapping up

The conventional prompt engineering method, employing a JSON Schema Parser like Langchain’s Pydantic Output Parser, is a cheap method that can be used with any capable LLM. It won’t work well with small models and is not guarantee that it will work 100% of the times even with GPT-3. If you are planning to use only OpenAI API and don’t expect to move to alternative models later, just use [**Function Calling**](https://openai.com/blog/function-calling-and-other-api-updates).

Jsonformer can be more reliable than plain prompt engineering if you are using [**transformers**](https://huggingface.co/docs/transformers/index). This library can handle JSON schemas with limited complexity even with less capable models. The main benefit is that is does not generate the tokens from the structure, just the attribute values. The downside is that it supports only a subset of JSON Schema types and it is not actively maintained lately.

Guidance is a another promising LLM framework. The main contribution is a DSL for creating complex templates, that we can use to structure valid JSON responses. However, it requires larger models to function effectively and can be very memory intensive. I had trouble to make it work with smaller models. From my experience, you need at least a 7bilion-parameters models.

Outlines is the most complete solution, for instance you can constraint the token generation using regex patterns or predefined vocabulary. It can produce JSON respecting a Pydantic model with no issues, providing a more sophisticated approach to JSON generation than simply plain prompt engineering. In active development this library offers the most advanced solution not only for JSON responses but a variety of other guided text generation use cases. At the moment I write this **_Outlines_** is my favorite choice and Guidance is on my radar for other use cases.

Additionally, projects that are also relevant to mention in the domain are [**LMQL**](https://docs.lmql.ai/en/stable/) (Language Model Query Language) and [**guardrails**](https://github.com/ShreyaR/guardrails). Both with their own merits and limitations are worth checking.