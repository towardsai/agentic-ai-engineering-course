# The Best Way to Generate Structured Output from LLMs

Benchmarking LLM structured output performance with OpenAI, Instructor, Marvin, BAML, TypeChat, LangChain, and how to overcome reasoning deficiencies using a multi-step Instill Core pipeline.

Industries are eagerly capitalizing on Large Language Models (LLMs) to unlock
the potential within their vast reserves of under-utilized unstructured data.
Given that up to
[80% of the worlds data is soon forecast to be unstructured](https://www.ibm.com/blog/managing-unstructured-data/),
the drive to harness this wealth for innovation and new product development is
immense. There is an ironic paradox here: LLMs, by their very design, output
_more_ unstructured text data to manage and keep on top of. That is, until very
recently!

Earlier this month, OpenAI announced that they now support
[**Structured Outputs in the API**](https://openai.com/index/introducing-structured-outputs-in-the-api/)
with general availability. The ability to distill and transform the creative and
diverse unstructured outputs of LLMs into actionable and reliable structured
data represents a huge milestone in the world of unstructured data ETL (Extract,
Transform and Load).

However, there’s more to this story than meets the eye.

## The Complexities of Structuring Outputs from LLMs

Coincidentally, the day before OpenAI’s announcement, a paper was published
titled
[Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models](https://arxiv.org/abs/2408.02442v1),
which offers a compelling counterpoint. They demonstrate that LLMs **struggle**
**with reasoning tasks when they’re placed under format restrictions**.
Additionally, the _stricter_ these format restrictions are, the _more_ their
reasoning performance drops, revealing a complex interplay between structuring
outputs and model performance.

Besides the new structured outputs API from OpenAI, there is also a multitude of
existing LLM frameworks that have been developed to tease out structured outputs
from conventional LLMs. Under the hood, each of these tools works by using a
blend of the following techniques:

### 1. Prompt Engineering

**Structured Prompts:** Well-crafted prompts are used to instruct the LLM to
output data in a specific format, such as JSON. These prompts guide the model to
generate responses that match the expected structure.

**Example Guidance:** Prompts often include examples of the desired output
format to further guide the model.

### 2. Output Parsing

**Data Models:** The structured output is typically aligned with predefined data
models. These models serve as blueprints to ensure the output data conforms to
specific formats and structures.

**Validation:** After the LLM generates the output, it is validated against
these data models. If the output doesn’t match the expected structure, error
handling mechanisms may be triggered to correct or retry the process.

### 3. Error Handling

**Retries and Corrections:** If the output doesn’t meet the required format
(e.g., due to syntax errors or unexpected structures), adjustments may be made
to the prompt, or the output may be re-parsed to correct the issue.

**Strict Parsing:** The framework includes strict parsing tools that ensure the
output conforms exactly to the required schema, raising exceptions if the output
is malformed.

This raises the question: **How do these tools compare with OpenAI’s structured**
**outputs feature on a task that involves both reasoning and output format**
**restrictions?**

## Benchmarking Structured Output Tools with Reasoning Tasks

To better understand the complexities and nuances of generating structured
outputs from LLMs, we conducted an experiment using OpenAI’s latest GPT-4o
model. This experiment aimed to evaluate how OpenAI’s new structured outputs in
the API feature compared with various existing tools and frameworks on a complex
task that involved both reasoning and producing outputs that are structured
according to a predefined data model.

The full details, code, and results of this experiment can be found in the
accompanying notebook in the
[**Instill AI Cookbook**](https://github.com/instill-ai/cookbook). Launch it
with Google Colab using the button below:

[Open in Colab 🚀](https://colab.research.google.com/github/instill-ai/cookbook/blob/main/examples/Generating_structured_outputs.ipynb)

### The Task: Combining Reasoning and Structured Output

Our benchmark task was designed to challenge the LLM’s ability to reason while
adhering to a strict output structure. Inspired by the aforementioned
[paper](https://arxiv.org/abs/2408.02442v1), we created a task that involved two
main components:

1. **Reasoning Problem:** The LLM was asked to calculate a freelancer’s weekly
earnings based on varying hourly rates, including overtime pay, as follows:


```
John Doe is a freelance software engineer. He charges a base rate of $50 per
hour for the first 29 hours of work each week. For any additional hours, he
charges 1.7 times his base hourly rate. This week, John worked on a project
for 38 hours. How much will John Doe charge his client for the project this
week?
```

2. **Structured Output:** In addition to solving the reasoning problem, the LLM
was also asked to summarize information from a resume into a specific data
model, shown below:


```
class DataModel(BaseModel):
      name: str
      email: str
      cost: float  # Answer to the reasoning problem, stored as a float
      experience: list[str]
      skills: list[str]
```


The LLM was instructed to store the result of the reasoning problem in the
`cost` field of the data model, while the rest of the fields were to be
populated with data extracted and summarized from the following resume text:


```
John Doe
1234 Elm Street
Springfield, IL 62701
(123) 456-7890
Email: john.doe@gmail.com

Objective: To obtain a position as a software engineer.

Education:
Bachelor of Science in Computer Science
University of Illinois at Urbana-Champaign
May 2020 - May 2024

Experience:
Software Engineer Intern
Google
May 2022 - August 2022
- Worked on the Google Search team
- Developed new features for the search engine
- Wrote code in Python and C++

Software Engineer Intern
Facebook
May 2021 - August 2021
- Worked on the Facebook Messenger team
- Developed new features for the messenger app
- Wrote code in Python and Java
```

### Existing Tools for Structured Output Generation

We evaluated several existing libraries and frameworks designed to help generate
structured outputs from LLMs. The tools tested include:

- [**Instructor**](https://python.useinstructor.com/): A Python library built on
Pydantic that facilitates generating structured output from LLMs.
- [**Marvin**](https://www.askmarvin.ai/): A tool for building reliable natural
language interfaces.
- [**BAML**](https://www.boundaryml.com/): A domain-specific language for writing
and testing LLM functions.
- [**TypeChat**](https://microsoft.github.io/TypeChat/): A tool from Microsoft
for getting well-typed responses from language models.
- [**LangChain**](https://www.langchain.com/): A Python library that integrates
language models with data and APIs to build applications.

[Outlines](https://outlines-dev.github.io/outlines/),
[JSONformer](https://github.com/1rgs/jsonformer) and
[Guidance](https://github.com/guidance-ai/guidance/tree/main) were also
considered, however they were left out of this experiment as they had limited
support for remote API calls or failed when integrating with the latest OpenAI
API.

### Experiment Setup

The GPT-4o model from OpenAI was used by all tools. To ensure a reasonable
amount of variation, its temperature was set to 1.0, with the exception of
TypeChat which operates with a temperature of 0.0 and doesn’t readily expose
this parameter to the user.

Each tool processed 50 inference requests based on the same example task
described above. For more details on the experimental setup, please refer to the
accompanying notebook.

### Results

The performance of these tools was assessed based on three criteria:

1. **Correctness of the Structured Output:** Whether the output was correctly
formatted according to the data model.
2. **Accuracy of the Reasoning Task:** Whether the tool could correctly
calculate the `cost` based on the reasoning problem.
3. **Mean Absolute Error:** The average absolute difference between the
calculated `cost` and the ground truth value.

The results of the experiment are summarized in the table below:

| **Tool** | **Correct Output Structure** | **Correct Reasoning** | **Mean Absolute Error** |
| --- | --- | --- | --- |
| OpenAI (Text) | ❌ | ✅ | $0.00 |
| OpenAI (Structured) | ✅ | ❌ | $93.93 |
| Instructor | ✅ | ❌ | $109.61 |
| Marvin | ✅ | ❌ | $71.01 |
| BAML | ✅ | ❌ | $72.94 |
| TypeChat | ✅ | ❌ | $100.00 |
| LangChain | ✅ | ❌ | $97.23 |

_Please note that these were generated using the latest OpenAI GPT-4o model on_
_the 27th of August 2024. Future experiments will produce variations in these_
_results as the APIs and tools evolve._

## Key Insights

While OpenAI’s GPT-4o model was able to calculate the correct reasoning result
when generating unstructured text, it was unable to return the correct value
when the output was constrained to the structured data model. Additionally, none
of the existing tools were able to correctly solve the reasoning task while
adhering to the structured output format.

This aligns with findings from the [paper](https://arxiv.org/abs/2408.02442v1),
which highlighted how format restrictions can impair reasoning abilities. It
also shows that the challenges identified in the original study with
GPT-3.5-turbo persist: a) with OpenAI’s latest flagship model, and b) when this
model is used in tandem with an array of existing output structuring tools.

### Can Chain-of-Thought Reasoning Help?

In the Structured Outputs
[release post](https://openai.com/index/introducing-structured-outputs-in-the-api/),
OpenAI suggest that the quality of the final response can be improved using
chain-of-thought reasoning within the schema by defining a field called
`reasoning_steps`. This approach involves having the model outline its reasoning
process in this field before placing the final answer in the `cost` field.

Interestingly, our tests with this approach - implemented in the
[`structured-output-reasoning-cot`](https://instill.tech/george_strong/pipelines/structured-output-reasoning-cot/playground)
pipeline - showed that, even when the reasoning was correctly detailed by the
model in `reasoning_steps`, it was still unable to populate the `cost` field
accurately.

Despite advancements in model capabilities, combining complex reasoning with
structured output generation remains a significant challenge. A nice quote from
OpenAI’s
[release post](https://openai.com/index/introducing-structured-outputs-in-the-api/)
hints at this problem, and provides some additional guidance:

“ _Structured Outputs doesn’t prevent all kinds of model mistakes. For example,_
_the model may still make mistakes within the values of the JSON object (e.g.,_
_getting a step wrong in a mathematical equation). If developers find mistakes,_
_we recommend providing examples in the system instructions or splitting tasks_
_into simpler subtasks._”

This motivates our next question: **How can we reliably overcome these reasoning**
**and output structuring challenges?**

## The Solution: A Multi-Step Pipeline Built with **Instill Core**

Given the difficulties observed across all frameworks, a more robust approach is
needed. Instead of attempting to solve the reasoning and structuring tasks
simultaneously, we can use [Instill Core](https://www.instill-ai.dev/) to build a
multi-step pipeline that divides these into two distinct stages:

1. **Reasoning Step:** In the first step, the LLM is tasked solely with solving
the reasoning problem without any constraints on the output format. This
allows the model to leverage its full reasoning capabilities without being
hindered by strict formatting requirements.

2. **Structuring Step:** In the second step, the unstructured output from the
reasoning step is passed to a component that focuses exclusively on
structuring the data into the required format.


https://www.instill-ai.com/_vercel/image?url=%2Fblog%2Fllm-structured-outputs%2Fa-multi-step-pipeline.png&w=1920&q=100

F­i­g­u­r­e 1: A m­u­l­t­i-step p­i­p­e­l­i­n­e b­u­i­l­t with I­n­s­t­i­l­l Core.

This approach ensures that the system is able to adhere to the specified data
model, whilst also enabling the reasoning LLM to solve complex tasks,
unencumbered by format constraints. The full results of the benchmarking
experiment are shown in the Figure below:

https://www.instill-ai.com/_vercel/image?url=%2Fblog%2Fllm-structured-outputs%2Fbenchmark.png&w=1920&q=100

F­i­g­u­r­e 2: B­e­n­c­h­m­a­r­k­i­n­g e­x­p­e­r­i­m­e­n­t r­e­s­u­l­t­s c­o­m­p­a­r­i­n­g M­u­l­t­i-Step I­n­s­t­i­l­l Core,
O­p­e­n­A­I, I­n­s­t­r­u­c­t­o­r, M­a­r­v­i­n, BAML, T­y­p­e­C­h­a­t, and L­a­n­g­C­h­a­i­n on r­e­a­s­o­n­i­n­g and
s­t­r­u­c­t­u­r­e­d o­u­t­p­u­t g­e­n­e­r­a­t­i­o­n.

As can be seen, the multi-step pipeline was able to consistently **achieve both**
**correct reasoning and structured output where other techniques failed**.

Despite requiring two LLM inferences, it is worth noting that the cost of this
multi-step pipeline approach will likely still be less than many of the
structuring output tools that have been considered, as these often rely on
making repeat API calls with modified prompts until the output can be
successfully parsed into the required data model.

## Conclusion

The article underscores the complexity of generating structured outputs from
LLMs, particularly when reasoning tasks are involved. The new Structured Outputs
in the API feature from OpenAI is a significant advancement, but it’s not a
silver bullet. As our results show, even the most advanced models and techniques
can falter under strict output format constraints.

The multi-step approach built with Instill Core provides a practical solution to
these challenges. By isolating the reasoning process from the structuring
process, it allows LLMs to perform complex tasks without sacrificing accuracy or
output quality. For businesses and developers looking to harness the full
potential of LLMs in applications requiring structured data, this approach
offers a reliable and cost-effective path forward.