## Task

You are an expert technical writer and AI Engineering who writes educative Jupyter Notebooks for student's getting into AI Engineering.

Generate a Jupyter Notebook called `notebook.ipynb` based on the source code that can be found within the [writing_workflow directory](../writing_workflow/) following the outline from `Outline of the Notebook`

Our goal is to write an educational Jupyter Notebook that explain pieces of the code from [writing_workflow directory](../writing_workflow/) based on the `Outline of the Notebook` instructed below. 

You will follow the same pattern when writing the Jupyter Notebook as in the `Examples`. Use these examples as few-shot examples on how to actually write and structure the Notebook. 

These are super important for how to format the:
- structured of the Notebook
- code
- writing that describes the code
- Notebook sections
- Notebook cells
- dynamics between Python and Markdown cells
- the usage of ``` quotes
Also, you should use them to infer the style of the Notebook, such as:
- how verbose to be when writing the code description
- how to frame the tonality, mechanics amd terminology of the writing

As a general rule of thumb you will maintain a direct, assertive, clear, confident, but excited and friendly tone and voice. With no fluff, but straight to the point. An active voice, while talking at the second person with "you" and "we" as we address this Notebook to a student from the part of the team who did this Notebook.


## The `brown` Python Project 

Remember that the Notebook should only take pieces of code from the `writing_workflow` Python project and explain how it works. You can threat the `writing_workflow` as an independent Python package called `brown` that we want to explain. You will find all the details on how the `writing_workflow` is packaged using `uv` under the pyproject.toml files from [writing_workflow directory](../writing_workflow/pyproject.toml) and [the root direcotry](../../pyproject.toml). The package is installed under `brown`. Thus, whenever you want to use code from "../writing_workflow/src/brow" you will use it as `from brown import ...`.

Everytime I will refer to use a particular file, I will assume it is relative to the `brown` Python package from "../writing_workflow/src/brown". As we already have the `brown` Python package installed through `uv`, the imports will work that way. 

But when generating the Notebook, and looking for the code, you have to look it up. For example, if I state anywhere use class ArticleWriter Node from @nodes.aricle_writer.py you will know that is relative "../writing_workflow/src/brown", thus it's inside "../writing_workflow/src/brown/nodes.aricle_writer.py"


## What Modules to Include From the `brown` Package

- `brown.evals`
- `brown.observability.evaluation`
- `scripts.brown_run_eval`

Use other files as context, but DO NOT ADD THEM EXPLICTLY INTO THE NOTEBOOK. THIS IS A NOTEBOOK SOLELY ON AI EVALS.

## Folders that we will download within the Notebook to always have at the same level with the Notebook:

- `configs`
- `inputs/evals`
- `inputs/examples`
- `inputs/profiles`
- `inputs/tests`

Thus, even if you reference these from other parts of the repository when generating the Notebook, these will always be relative to the notebook following the paths as stated above.

## Important Instructions That You Will Respect All the Time

- When you explain a partial piece of code, use Markdown cells with ```python...``` blocks (or other languge blocks)
- Use MagicPython blocks only when you use the actual code that you want to run
- When we run code, instead of using our sketched code from the Notebook we import it from the [writing_workflow directory](../writing_workflow/) directory through the `brown` package such as "from brown... import ..."

- Whenever a point is marked as Markdown you will use a Markdown cell, and when it's marked as code, or Python you will use a MagicPython cell that can be run and uses the `brown` package to import the real code
- Whenever I explictly state that is a Markdown section that explain a piece of code, you will still add the code, but instead of adding it in a Python magic cell, you will add it in ```python...``` blocks + the text description surrounding it. Remember that this notebook's scope is to explain the code, thus going in detail over every piece of code it's important, even if it's in Markdown.
- For example if I have the `(Markdown) First, we have to explain the Node, Toolkit and ToolCall interfaces from @nodes/base.py` intrusction, you will walk through all the states classes, such as Node, Toolkit, ToolCall, where you actually add all the code into the Notebook, but instead of adding it into a PYthon Magic Cell, you will add it into Markdown cells surrounded by ```python...``` as we don't want to make it executable. But if we provide in the outline something like this `(Python) Give an example on how to use the MediaGeneratorOchestrator by generator 3 mermaid diagrams in parallel using it.` you will create actual Python code, within MagicPython cells that can be run on their own. You will respect this strategy for all the sections or points marked as "Markdown" or "Python/Code"

- Thus, because we actually execute code from `brown`, you have the freedom to exclude redundant code blocks from the Markdown desriptive sketch blocks, such as stating all the imports, pydocs, utility fuctions or classes, etc...
- If some code blocks, either Markdown code blocks or MagicPython code blocks become too big, split them into groups based on logical units, such as functions, methods, classes, or files
- When splitting Markdown code blocks split them into multiple ```markdown blocks, where each block is numbered with 1., 2., 3., ... (as a numbered list, where each bullet points contains the code description + the code block).
- When splitting the MagicPython code blocks into multiple groups, split them into working cells that can be run on their own. Also, at the end of each MagicPython cell, you should add an output showing what we did within that cell/step. Also, there should be a constant dynamic between a Markdown cell that describes the code below and a PythonMagic cell that actually runs the code.
- When explaining the code, focus just on describing the visible code, assuming people don't know how it works. Also, whenever you find fancier Python code syntax, explain that as well.

- Whenever you have to output longer outputs, prettify a print by surrounding it between "----" dashes + a title, such as "---- TITLE ----" use the `wrapped` function from @utils/pretty_print. You can use it directly as `from utils import pretty_print; pretty_print.wrapped(...)`. Find the full definition at `lessons/utils/pretty_print.py`. This is the interface of the pretty_print function:
```
def wrapped(
    text: Iterable[str] | dict | str,
    title: str = "",
    width: int = 100
) -> None:
...
```
- The `pretty_print.wrapped(...)` function can be especially useful for printing longer elements such as the article, guidelines, profiles, research, media items,

- When writing code within Markdown blocks, be careful at relative imports such as `.mixins`. As these Markdown blocks from brown are seen in isolation they can be confusing, thus always replace them with the absolute path such as instead of `.mixins` do `brown.entities.mixins`. When you do this, look for the import path relative to the `brown` package from "../writing_workflow/src/brown"
- Whenever you have a ```python block that contains a nested ```mermaid block, use only two back quotes to espace the nested block, such as ``mermaid. You can apply the same logic to any type of combination of blocks.


## Outline of the Notebook

1. Quick introduction on what we will explain in this notebook, that will include:
    - a sentence on the scope of the notebook
    - a short outline explaining the learning objectives as bullet points of everything that will be covered with the lesson
2. Code setup section which will be one-on-one with the setup section from the example notebooks.
3. Markdown theoretical section explaining what LLM Judges we will build:
    - We will quickly remind people that for each evaluation dataset sample, we need the article guideline, research, and expected article
    - Remember that the dataset created by us in Lesson 28 contains per each dataset sample: the article guideline, research and expected article, while we have to generate on the fly the generated article. The idea is that everytime we make a change to our AI App (Brown) you have to regenerate the articles and recompute the metrics. That's why everything else is static, while feedling only with the generated outputs and a few of the system parameters during the optimization process of the AI app.
    - Mermaid Diagram showing the moving blocks of a dataset sample.
    - Then we will building the following LLM judge metrics:
        1. One that compares the expected article with the generated article on dimensions such as content, flow of ideas and structure adherence
        2. One that compares the generated article with the input article guideline and research checking if the input adheres to the article guideline (aka stays on track with human input) and it's written solely based on the research (aka checks hallucination)
    - We will wrap up this section with a paragraph explaining that our LLM judges will use only binary metrics scoped at the section level, instead of the whole article level. Thus, for each article, we will have multiple metrics, computed for each section individually. Like this, we can leverage the beauty of binary metrics, while still having granularity. Also, as from most point of views such as adherence to guideline, research or ground truth a section is independent from one another. Thus, this decision gives us more signal and precision in understanding what our application did wrong or in fixing and calibrating our LLM judge
    - A mermaid diagram showing an article with multiple section, where we compute multiple metrics per section
4. Markdown theoretical section quickly explaining how we will use the dataset created in Lesson 28 to build our metrics:
    - We will split the dataset into training, validation, testing
    - The training split will be used as few shot examples to "train" the LLM judges 
    - The validation split will be used to align the LLM judges with human expectections. In other words used to evaluate the LLM judges themselves.
    - The testing split will be used to compute the metrics on
    - Mermaid diagram showing this split.
5. Markdown section starting to dig into the follows ground truth metric code, the first LLM judge metric. As we have some base abstractions leveraged in multiple LLM judge metrics in `brown.evals.metrics.base` and because this is the first metric that uses them, we will bounce between the abstractions and actual implementations, first showing the abstraction and then the implementation. Show the code in Markdown format for the following classes and files:
    1. The `BrownBaseMetric` abstraction from `brown.evals.metrics.base` (ignore the FewShotExamplesT for now)
    2. The `FollowsGTMetricLLMJudge` implementation from `brown.evals.metrics.follows_gt.metric`
    3. The `FollowsGTArticleScores` implementation from `brown.evals.metrics.follows_gt.types` and other subclasses such as `FollowsGTSectionScores` and `FollowsGTCriterionScores`
    4. The `CriteriaScores` and `CriterionScore` classes from `brown.evals.metrics.base`
    5. The `aggregate_section_scores_to_results` function from `brown.evals.metrics.base` that is used within the `FollowsGTArticleScores.to_score_sult` method to aggregate a list of scores to a single score. Explain here in more detail what the function does.
    6. The `CriterionAggregatedScore` that represents an aggregate score
    7. Now we mvoe to the few shot examples classes starting with `FollowsGTMetricFewShotExamples` and `FollowsGTMetricFewShotExample` from `brown.evals.metrics.follows_gt.types`
    8. Then we explain the base entities `BaseFewShotExamples` and `BaseExample` from `brown.evals.metrics.base`
    9. Lastly, we explain the LLM Judge system prompt `SYSTEM_PROMPT` from `brown.evals.metrics.follows_gt.prompts` diving it into the following groups:
        1. Introduction
        2. Instructions: Explain in more depth how we instructed the LLM Judge to compute the scores based on article sections instead of the whole article. Also, explain how we encoded the `content`, `flow` and `structure` binary metrics
        3. Chain of Thought
        4. What to Avoid
        5. Few-shot examples and input
        6. Conclusion
    10. Next, we explain the `DEFAULT_FEW_SHOT_EXAMPLES`. Show the core initialization logic of the data structure, while within the `scores` list, which is the longest highlight only the following:
        1. `04_structured_outputs`: "Introduction", "Why Structured Outputs Are Critical", ..., "Structured Outputs Are Everywhere", "References"
        2. `07_reasoning_planning`: "Introduction", ..., "Teaching Models to "Think": Chain-of-Thought and Its Limits", "Separating Planning from Answering: Foundations of ReAct and Plan-and-Execute", ..., "Plan-and-Execute in Depth: Structure and Predictability", ....
        3. Explain that to create these few shot examples, we manually generated the articles based on these inputs, added noise to it to create as much variation as possible and manually labeled each section along these dimensions
        4. (CODE SECTION) Call the few shot examples as context for people to see how it looks
    11. The last piece of the puzzle is the `get_eval_prompt` function that aggregates the `get_eval_prompt` into the final prompt.
6. Code section in which we will run the `FollowsGTMetricLLMJudge` metric on a simple example for people to actually understand how it works. We will mock a simple isolated example based on the article example from `inputs.tests.01_sample_small`. Use that sample as `expected_output` and add variatns for the `output` to see the LLM judge in action. make these two as constants so we can use them later on as well. Print the outputs.
7. Mardown section similar to 5, where we will go over the User Intent metric from `brown.evals.metrics.user_intent`. As we already explained the base classes and functions from `brown.evals.metrics.base`, this time we will focus solely on the particularities of this metric, such as:
    1. The `UserIntentMetricLLMJudge` implementation from `brown.evals.metrics.user_intent.metric`
    2. The `UserIntentArticleScores` implementation from `brown.evals.metrics.user_intent.types` and other subclasses such as `UserIntentSectionScores` and `UserIntentCriteriaScores`
    3. Now we move to the few shot examples classes starting with `UserIntentMetricFewShotExamples` and `UserIntentMetricFewShotExample` from `brown.evals.metrics.user_intent.types`
    4. Lastly, we explain the LLM Judge system prompt `SYSTEM_PROMPT` from `brown.evals.metrics.user_intent.prompts` diving it into the following groups:
        1. Introduction
        2. Instructions: Explain in more depth how we instructed the LLM Judge to compute the scores based on article sections instead of the whole article. Also, explain how we encoded the `Guideline Adherence`, and `Research Anchoring` binary metrics
        3. Chain of Thought
        4. What to Avoid
        5. Few-shot examples and input
        6. Conclusion
    5. Next, we explain the `DEFAULT_FEW_SHOT_EXAMPLES`. Show the core initialization logic of the data structure, while within the `scores` list, which is the longest highlight only the following:
        1. `04_structured_outputs`: "Introduction", "Understanding why structured outputs are critical", ..., "Conclusion: Structured Outputs Are Everywhere"
        2. `07_reasoning_planning`: "Introduction", ..., "Teaching Models to "Think": Chain-of-Thought and Its Limits", "Separating Planning from Answering: Foundations of ReAct and Plan-and-Execute", ..., "Plan-and-Execute in Depth: Structure and Predictability", ....
        3. Explain that to create these few shot examples, we manually generated the articles based on these inputs, added noise to it to create as much variation as possible and manually labeled each section along these dimensions
        4. (CODE SECTION) Call the few shot examples as context for people to see how it looks
    6. The last piece of the puzzle is the `get_eval_prompt` function that aggregates the `get_eval_prompt` into the final prompt. 
8. Code section in which we will run the `UserIntentMetricLLMJudge` metric on a simple example for people to actually understand how it works. We will a similar scenario as in section 6, where we will use as seed the example from `inputs.tests.01_sample_small`. Use the input and context as is, while injecting noise into the article as `output` to put the LLM judge in action. Print the outputs.
9. Small markdown and code section showing the factory method from `brown.evals.metrics.__init__.py` used to build the required metrics:
    - (markdown) show the function
    - (code) call the function with the `user_intent` and `follows_gt` inputs and gemini 2.5 flash model
10. We are almost done, the last phases of these Notebook will be a markdown and code section showing how to run the two LLM judge metrics from above on the dataset we created in Lesson 28. We will start by explaining the functions from `brown.evals.metrics.tasks.py` which are used to run the generate article workflow of Brown on a dataset sample:
    1. (Markdown) `evaluation_task`
    2. (Markdown) `__run`
10. (markdown + code) Now, to hook our `evaluation` task to the dataset stored in Opik and run the LLM judges defined above on each dataset sample, we have to run a bit more glue code:
    1. (Markdown) `evaluate` from `brown.observability.evaluation`
    2. (Markdown) `get_dataset` from `brown.observability.opik_utils`
    3. (Markdown) `create_evaluation_task` from `brown.evals.metrics.tasks.py` (we need it because the `evaluate` function from Opik doesn't allow us to pass any other parameters to the `task` function used to generate the output on each dataset sample. Thus we use it to pin other attributes of the function leveraging the `partial` function from Python)
11. Last code section used to run the end-to-end workflow using the exact same strategy from `scripts.brown_run_eval` but not as a CLI and directly in the Notebook.
    - Highlight that what we did now, it's called an `experiment`, where for a particular state of our AI application and a given AI evals dataset we computed a score reflecting how good our system is.
    - Next steps: As now we have clarity on what our system works (or doesn't work) we can start tweaking our AI app while our direction is dictated by the metrics acting as our north star
12. (Markdown) Conclusion and Future steps:
    - One sentence on the importance of AI evals
    - a bullet list with `what we've learned in this lesson`
    - one sentence stating that with this we wrapped up the AI evals lessons, moving to other ops aspects such as deploying the agents and building CI/CD pipelines
    - "practicing ideas" section with:
        - Use the cached articles from the `inputs` and run the LLM judges 5 times to see how stable are the LLM judges while having all their inputs fixed
        - Use the `Memory` lesson from the validation split to see how aligned are the LLM judges with your human judgment
        - Do some changes to the Brown AI app and run again the AI evals to see if your change improved the system or not
    - Useful resources section
    - Run Brown as a standalone python project section

## Examples

- [Offline AI Evals Dataset Creation](../28_ai_evals_offline_dataset/notebook.ipynb)
- [Human in The Loop](../24_human_in_the_loop/notebook.ipynb)
- [Foundations Writing Workflow](../23_evaluator_optimizer/notebook.ipynb)

## Chain of Thought

- Carefully read the instructions
- First read the examples and understand how we expect the output Jupyter Notebook to look like
- Understand how to apply our particular `Outline of the Notebook` to a Jupyter Notebook format while following the pattern from the examples
- Generate the Jupter Notebook based on the `Outline of the Notebook`. Follow each point from the outline as is. We expect that each bullet point will be present within the Jupyter Notebook.