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

In lesson 22:
- Entities: Article, Guidelines, Media Items, Mixins, Profiles, Research from @entities
- Nodes: Article Writer, Media Generator, Tool Nodes + Base abstractions from @nodes
- Models: Config, Get Model from @models
- Utils (keep only high level: just explain what they do when they are used first time, don't go over the implementation) from @utils
- @base.py, @builders.py, @config_app.py, @config.py, @loaders.py, @renderers.py (keep only high level: just explain what they do when they are used first time, don't go over the implementation)
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

In lesson 23:
- Entities: Review, ArticleReviews, ArticleReviews, SelectedTextReviews, SelectedText from @entities
- Memory: InMemory from @memory
- Nodes: Article Reviewer from @nodes/article_reviewer.py
- Workflows: Generate Article from @workflows.generate_workflow
- config app logic from @config.app

## Notebook Guidelines of Previous Lessons:

Use the notebooks of previous lessons to know what code blocks were already explained or not. Super critical to know what to explain again and what to avoid repeating ourselves:

Only define items that weren't defined in previous lessons!

- [Lesson 22](../22_foundations_writing_workflow/notebook_guideline.md)
- [Lesson 23](../23_evaluator_optimizer/notebook_guideline.md)


## What Files to Include From the `brown` Package

- Entities: HumanFeedback from @entities
- MCP: tools, prompts, resources from @mcp
- Workflows: Edit Selected Text, Edit Article from @workflows


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
 - the importance of human-feedback and human-in-the-loop
 - MCP implementation for exposing all the workflows as tools, prompts to manage our MCP tools easily when interfacing with the MCP server and resources to easily understand how the MCP server is configured
- to add human in the loop we implemented two new workflows for editing the whole article or just a piece of selected text based on human feedback + the internal enforced rules
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
3. Markdown section on explaining at a high level how we will implement human feedback into the writing workflow and how we will leverage MCP to serve the app and do that:
    - After the article is generated using the writing workflow explain in lessons 22 and 23, as writing is highlighy subjective, there are big changes that you want to further edit the whole article or just pieces of it
    - The process of using AI to generate and automate part of your work and then you as a domain expert to review it is the perfect balance between AI and human
    - Thus, we designed the whole workflow to easily introduce the human in the loop, between generating the first version of the article and then after looking over it to further spin up new reviewing + editing workflow + human feedback to further refine the article
    - Thus, We can use a low number of review loops during article generation, and further run them dynamically with a human in the loop, if it's necessary + more human guidance...
    - To do so, we need to decouple the writing article workflow from future editing workflows. We used MCP Servers to do that, where the generate article workflow and edit article workflow are two separate tools that can be called independently
    - Mermaid diagram with:
        1. Brown as an MCP server
        2. One tool that comes out of brown as the generate article workflow
        3. Another tool that comes out of brown as the edit article workflow
        4. Another tool that comes otu of brown as the edit selected text workflow
        5. A human that looks at the article as an output from point 2, 3 and 4
        6. Point 2, generates an article, then there is a human-feedback loop between the human and points 3 and 4 until the human is satisfied with the results
4. (Markdown + Code) First, let's explain how we introduced the Human Feedback element into our Article Reviewer Node:
    1. (Markdown) First, introduce the HumanFeedback entity from @entities.reviews.py
    2. (Markdown) Then, explain how we introduced and normalized the humand feedback within the ArticleReviewer node from @nodes.article_reviewer.py. To do so, explain only the section related to `Human Feedback` from the `ArticleReviewer`
    3. (Code) Provide a simple example on how the article reviewer node works with human feedback inputs. Use the sample inputs from `inputs/tests/01_sample/article.md`, `inputs/tests/01_sample/article_guideline.md` and profiles from `inputs/profiles`. Load them with the specialized loaders from @loaders. Then use the exact human_feedback as follows: "Make the introduction more engaging, catchy and shorter. Also, expand on the definiton of both workflows and agetns from the first section"
    4. (Code) Show the output reviews emphasis on how now we have the human feedback available along the profiles reviews
    (run the example with `SupportedModels.GOOGLE_GEMINI_25_FLASH`)
5. Markdown section on implementing the workflow that automatically reviews and edits and existing article based on the workflow from @workflows.edit_article.py:
    1. (Markdown) Explain the build_edit_article_workflow, EditArticleInput components
    2. (Markdown) Explain _edit_article_workflow 
    3. (Markdown) Explain generate_reviews
    4. (Markdown) Explain edit_based_on_reviews
    (Highlight that because we used the clean architecture pattern, and we just glue things at the app layer, this is almost identical to the writing workflow explain in lesson 23)
    5. Highlight the importance of <human_feedback>... We can use a low number of review loops during article generation, and further run them dynamically with a human in the loop, if it's necessary + more human guidance...
5. Markdown section on implementing the workflow that automatically reviews and edits a selected piece of text from an article based on the workflow from @workflows.edit_selected_text.py
    1. (Markdown) Explain the build_edit_selected_text_workflow, EditSelectedTextInput components
    2. (Markdown) Explain _edit_selected_text_workflow 
    3. (Markdown) Explain generate_reviews
    4. (Markdown) Explain edit_based_on_reviews
    (Highlight that because we used the clean architecture pattern, and we just glue things at the app layer, this is almost identical to the to the workflow to edit the whole article)
    5. Highlight the importance of <human_feedback>... Most often we don't want to edit the whole article, but just a small section, or apply the human feedback just to a small section...
6. Markdown and code Bring everything together by exposing everything into the MCP server. Now we want to show how we served all the workflows as an MCP server. All the code from the MCP server can be found at @mcp.server.py
    1. (Markdown) Define the MCP server and define the tools.
    2. (Python) Run the tools to edit a piece of text, more exactly the second paragraph from the input article called `Understanding the Spectrum: From Workflows to Agents` + the human_feedback = "expand on the definiton of both workflows and agetns from the first section"
    3. (Markdown) Define the prompts
    4. (Python) Run the same example on editing a piece of but through the prompt
    5. (Markdown) Define the resources
    6. (Python) Call the resources to show we can understand how the MCP server is configured through the resources
    7. Highlight how all the MCP code (the serving/interface layer) is only within the @mcp directory, while we just import other layers such as the infra, app and domain ones. Making the servign layer 100% independent from the others, and allowing us to serve the rest of the codes with other methods easily.
    (We will run all the examples with the MCP client in memory so we can easily import it from @mcp.server and use it inside the Notebook)
7. We wrote a CLI script that let's you easily interface with the Brown writing agent. Available at [../writing_workflow/scripts/brown_mcp_cli.py]. To see how to use it together with Brown as an independent Python project read the docs from [../writing_workflow].
8. The beuty comes when using this through a legit client such as Cursor or Claude:
    1. For example, show the cursor config from "../../.cursor"
    2. Point them to a video where we will show how we use Brown + human in the loop to write professional articles. 
    3. Conclude with the idea that with this design we made writing articles havign a similar experience with coding. Along with your detailed human input you ask AI to generate a first short, then you add your human touch, you ask the AI to do furthe editing, and repeat.
9. (Markdown) Conclusion and Future steps:
    - What we've learnt:
        - How to design human in the loop in our AI apps
        - How to implement the edit article and edit selected text workflows
        - How to serve Brown the writing agent as an MCP server while respecting the clean architecture patterns
        - How to use the Brown writing within MCP Clients such as Cursor
    - Ideas on how you can further extend this code:
        - Hook Brown to Claude instead of Cursor
        - Serve Brown through FastAPI instead of MCPServer
        - Write other workflow to help you out within the writing workflow such as a tool to review + edit your article guideline that servers as input to the article generation workflow
    - With this we wrapped up the Brown writing workflow. In future lessons, we will further explore how we can easily orchestrate the two agents and provide more details on our end-to-end workflow on how to properly use them

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
- [Evaluator Optimizer](../23_evaluator_optimizer/notebook.ipynb)

## Chain of Thought

- Carefully read the instructions
- First read the examples and understand how we expect the output Jupyter Notebook to look like
- Understand how to apply our particular `Outline of the Notebook` to a Jupyter Notebook format while following the pattern from the examples
- Generate the Jupter Notebook based on the `Outline of the Notebook`. Follow each point from the outline as is. We expect that each bullet point will be present within the Jupyter Notebook.