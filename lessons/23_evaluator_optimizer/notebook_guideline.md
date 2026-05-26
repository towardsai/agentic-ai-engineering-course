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

## What Files From the `brown` Package Were Already Explained
- Entities: Article, Guidelines, Media Items, Mixins, Profiles, Research
- Nodes: Article Writer, Media Generator, Tool Nodes + Base abstractions
- Models: Config, Get Model
- Utils (keep only high level: just explain what they do when they are used first time, don't go over the implementation)
- base.py, builders.py, config_app.py, config.py, loaders.py, renderers.py (keep only high level: just explain what they do when they are used first time, don't go over the implementation)
- Inputs: 
    - Examples: Course Lessons
    - Profiles: Explain the role each profile and why we need them
        - Mechaniscs Profile (General to any content type): Extract usage
        - Structure Profile (General to any content type): Extract usage 
        - Terminology Profile (General to any content type): Extract usage
        - Tonality Profile (General to any content type): Extract usage
        - Article Profile (Specific to articles): Used to configure the content type you want to write
        - Character Profiles (Specific to a given voice): Used to inject a particular voice, either yours, such as Paul Iusztin or from another popular character such as Richard Feynmann
    - Potential content types: article, video transcripts, social media post, technical post, marketing article, etc.

## Notebook Guidelines of Previous Lessons:

Use the notebooks of previous lessons to know what code blocks were already explained or not. Super critical to know what to explain again and what to avoid repeating ourselves:

Only define items that weren't defined in previous lessons!

- [Lesson 22](../22_foundations_writing_workflow/notebook_guideline.md)


## What Files to Include From the `brown` Package

- Entities: Review, ArticleReviews, ArticleReviews, SelectedTextReviews, SelectedText from @entities
- Memory: InMemory from @memory
- Nodes: Article Reviewer from @nodes/article_reviewer.py
- Workflows: Generate Article from @workflows.generate_workflow
- config app logic from @config.app

## What to Completely Exclude From The `brown` Package

- Completely ignore as we will touch these only in future parts of the course:
    - evals
    - models/fake_model.py
    - observability


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
 - the reviwing-editing process from writing an article implemented through the evaluator-optimizer pattern
 - including this pattern within the previous article writing workflow
 - the app config layer that allows us to configure everything from the app down to the node level
 - glueing everything together within a LangGraph workflow as a standalone application
2. Code setup section:
    - start with an intro identical with the example Notebook
    - curl the following folders to ensure we have everything we need in the same folder with the notebook: 
        1. configs using this exact command:
        ```bash
            !rm -rf configs
            !curl -L -o configs.zip https://raw.githubusercontent.com/towardsai/agentic-ai-engineering-course/main/data/configs.zip
            !unzip configs.zip
            !rm -rf configs.zip
        ```
        1. inputs using this exact command:
        ```bash
            !rm -rf inputs
            !curl -L -o inputs.zip https://raw.githubusercontent.com/towardsai/agentic-ai-engineering-course/main/data/inputs.zip
            !unzip inputs.zip
            !rm -rf inputs.zip
        ```
        3. Now run a `%ls` command to show the reader what we downloaded
        4. Define constants to the configs and inputs directories as "Path" objects that you will use whenever you have to reference something from these two folders
3. Markdown only section on explaining at a high-level how the writing agent works with the reviewing-editing process added into the loop:
    - As stated into lesson 22, we load the context, generate the media items, and then generate the first draft of the article
    - Then we apply the evaluator-optimzier pattern as follows:
        - We have a Article Reviewer node as the "evaluator" who checks if the article follows all the standards from the profiles. For every rule that is not respected it generates a review
        - Then we leverage the Article Writer as the "optimizer". We pass all these reviews back to the ARticle writer + some extra prompt engineering from what we've seen in lesson 22 to edit the article
        - Then we apply this review + edit pattern in a loop until a given number of iterations. As "writing quality" is highly subjective we decided not to quantize this process with a given score, but just hardcode a given number of iterations. You will see in Lesson 24 on human in the loop why this makes a ton of sense.
    - This strategy is extremely similar to how a real-world process would look like:
        1. The writer writes the article
        2. It asks for feedback from outside eyes
        3. The same writer edits the article based on the provided feedback
    - Generate a mermaid diagram with these 3 steps (+ the auxiliary input, output):
        1. Gather context
        2. Write article
        3. Review article
        4. Edit article
        5. Repeat steps 3 and 4 until a given number of iterations
        6. Refined article
4. Markdown section on explaining all the new entities, found at @entities, required for reviewing the article or only a piece of text from the given article (not a selected text of the article, just the whole article)
    1. Quick explanatation that along reviewing the whole article, we might want to review only a section of the article as most of the time only a section of the article has to be edited, not the whole article
    2. In the previous lesson, lesson 22, we already talked about most of the Pydantic entities such as the `Article`, `ArticleGuideline`, and `ArticleProfiles`, thus we won't go over these.
    3. Present only the ones related to reviweing the whole article: Review, ArticleReviews 
    4. Present the onews related to only reviewing a selected piece of text: SelectedTextReviews, SelectedText
5. Markdown and Python section on explaining the reviewer node, which is equivelant to the evaluator from the evaluator-optimizer pattern:
    1. (Markdown) Note on how we leverage the same `Node` abstraction. 
    2. (Markdown) Explain the ArticleReviewer class from @nodes/article_reviewer.py. Split the logic as follows (show the whole code without the selected text part):
        2.1. The class + the init method
        2.2. The _extend_model method + the ReviewsOutput temporary pydantic object (as the outout from the model differents from the output from the node, we can create this intermediate object)
        2.3. The ainvoke method 
        2.4. Show the system_prompt_template (show the whole system prompt without the human feedback and selected text parts)
        2.5. Show the selected_text_system_prompt_template (only focused on the selected text part of the system prompt)
    3. (Python) Run an example of JUST the reviewer using (assuming the article is already written)
        - article from `inputs/tests/01_sample/article.md`
        - article guideline from `inputs/tests/01_sample/article_guideline.md`
        - article profiles from `inputs/profiles`
        - ignore the human feedback 
        - use the @loaders to load all the context
        - render the edited article using a renderer from @renders under `article_edited.md`
        - run the example with `SupportedModels.GOOGLE_GEMINI_25_FLASH` 
    4. (Python) Repeat the same example, but but the selected text. Use the `Understanding the Spectrum: From Workflows to Agents` section of the article as an example.
6. Markdown section on explaining how to hook the reviews to the article writer node.
    1. (Markdown) Explain that to keep the "writing" logic contained, and avoid duplicated code, the article writer is both the writer and editor. Ultimately, this is similar to real-world scenarios, where the original author both writes the article and also edits it based on reviews.
    2. (Markdown) Explain the new pieces of code related to reviews from the Article Writer Node (ignore the SelectedText ones)
7. Code section on runningn an end-to-end example (without LangGraph)
    - Reload all the necessary context from `inputs/examples`, `inputs/profiles` and `inputs/tests/01_sample`. Use the @loaders to load it. Use the example from @workflows.generate_article.py to see how to do it properly. Be careful to use the right loaders such as `MarkdownArticleLoader` using the right input type
    - Call the orchestartor-worker node to generate the media items (Don't SHOW IT WORKS, we already did it in the previous lesson)
    - Call the Article Writer node to generate the first draft (Don't SHOW IT WORKS, we already did it in the previous lesson)
    - Show the article to the reader
    - Call the Article Reviewer to generate a review of the whole article
    - Show the reviews to the reader
    - Apply the reviews to the article
    - Show the edited article to the reader
    - Use the cdoe from @workflows.generate_article.py as an example
    - run the example with `SupportedModels.GOOGLE_GEMINI_25_FLASH` 
7. Markdown and code section on explaining the app config class. Before glueing all this code into a LangGraph workflow, we want to have a way to configure everything from a single YAML file, thus go over:
    - (markdown) the AppConfig class from @config_app.py with everything around it
    - (markdown) show an example of a config from @configs.course.yaml
    - (markdown) explain that like this we can change everything we want stgarting from the models down to a glanuratity of a node, to the number of iterations, few-shot-examples, etc. just from a single file that repreents the source of truth 
    - (code) load a config from `configs/course.yaml` and show how it looks like
8. Markdown section on gluging everyting together within a LangGraph workflow with inmemory checkpointer. Now we finally want to wrap everything into the app layer, by glueing everythging together into a robust workflow with LangGraph. The whole workflow can be found at @workflows.generate_article.py, while we call it at @mcp.server.py:
    1. (Markdown): Show the build_generate_article_workflow, _generate_article_workflow and GenerateArticleInput components (without the whole implementation of the _generate_article_workflow)
    2. (Markdown): Explain that we could also just do @entrypoint to decorate the _generate_article_workflow, but why we did this will make sense when we run the workflow in the next section. So bear with us
    3. (Markdown): The whole implementation of the _generate_article_workflow function. Highlight how we send the progress bar + call all the steps
    4. (Markdown): generate_media_items step
    5. (Markdown): write_article step
    6. (Markdown): generate_reviews step
    7. (Markdown): edit_based_on_reviews step
    8. (Markdown): Emphasize the retry_policy object which we used to decorate each step to retry in case of failure. That's why having a step with a single piece of logic that makes sense to retry it's extremely important making our workflow more robust to API / Infrastructure failures during inferenece
9. Markdown section as a paranthesis on how to built the inmemory short term memory checkpointer from @memory
10. Code section on runningn an end-to-end example (with LangGraph). Go over the generate_article function from @mcp.server.py showing how to call the generate_article_workflow. At this point completely ignore all the logic related to MCP and focus just on what matters around LangGraph, as follows:
    1. How to built the short term memory
    2. How to initialite the build_generate_article_workflow. That's why we created the build_generate_article_workflow to be able to inject the short_term memory at run time. Thus, we inject the infrastructure only at serving time respecting the clean architecture pattern
    3. How we define the config with the thread_id as UUID4 (ignore the tracer callback)
    4. How we call the langgraph workflow using astream
    5. As the parse_message function is dependent on FastMCP, write something similar with normal prints instead of using ctx.info
11. (Markdown) Conclusion and Future steps
    - What we've learnt:
        - How to implement the evalautor-optimizer pattern from scratch by implementing the reviewing-editing feature for the writing workflow
        - how to create a centralized configuration system
        - how to glue the whole writing workflow with LangGraph and apply retry policies per step
    - Ideas on how you can further extend this code: 
        - Use a different AI Framework than LangGraph Function API to glue the workflow. Some options LangGraph but using the graph API or pydanticAI. are As we applied the clean architecture, we can easily glue our entities and node with whatever AI Framework we want.
        - Play around with the reviewer, get a feeling of how it extract the reviews from the profiles and tweak either the profiles or the reviewer for better results.
    - In lesson 24 we will look on how we can further compose the reweing and editing logic to properly add human in the loop and how to expose everything as an MCP server

## Examples

### Examples Explaining the Research Agent Part 2
This is a completely different project. Ultimately we want to do something similar for the Brown project as well.

- [Source code from where the code was extracted to created the notebooks](../research_agent_part_2/)

Notebooks:
- [FastMCP](../16_fastmcp/notebook.ipynb)
- [Data Ingestion Lesson](../17_data_ingestion/notebook.ipynb)
- [Research Loops](../18_research_loop/notebook.ipynb)

### Examples From Previous Lessons Explaining the Brown Writing Workflow
This are previous Notebooks that we already finished for explaining the [Brown writing workflow](../writing_workflow/)

- [Source code from where the code was extracted to created the notebooks](../writing_workflow/)

Notebooks of previous lessons:
- [Foundations Writing Workflow](../22_foundations_writing_workflow/notebook.ipynb)

## Chain of Thought

- Carefully read the instructions
- First read the examples and understand how we expect the output Jupyter Notebook to look like
- Understand how to apply our particular `Outline of the Notebook` to a Jupyter Notebook format while following the pattern from the examples
- Generate the Jupter Notebook based on the `Outline of the Notebook`. Follow each point from the outline as is. We expect that each bullet point will be present within the Jupyter Notebook.
