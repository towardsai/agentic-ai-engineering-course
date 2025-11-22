## Task

Generate a Jupyter Notebook called `notebook.ipynb` based on the source code that can be found within the [writing_workflow directory](../writing_workflow/).

Our goal is to write an educational Jupyter Notebook that explain pieces of the code from [writing_workflow directory](../writing_workflow/) based on the `Outline of the Notebook` instructed below. 

You will follow the same pattern when writing the Jupyter Notebook as in the `Examples`. Use these examples as few-shot examples on how to actually write and structure the Notebook. These are super important for:
- how to format the code
- how to format the writing
- how to format the section
- how verbose to be when writing code description

Remember that the Notebook should only take pieces of code from the `writing_workflow` and explain how it works. You can threat the `writing_workflow` as an independent Python package called `brown` that we want to explain. You will find all the details on how the `writing_workflow` is packaged using `uv` under the pyproject.toml files from [writing_workflow directory](../writing_workflow/pyproject.toml) and [the root direcotry](../../pyproject.toml). The package is installed under `brown`. Thus, whenever you want to use code from "../writing_workflow/src/brow" you will use it as `from brown import ...`.

## IMPORTANT INSTRUCTIONS

- When you explain a partial piece of code, use Markdown cells with ```python...``` blocks (or other languge blocks)
- Use MagicPython blocks only when you use the actual code that you want to run
- When we run code, instead of using our sketched code from the Notebook we import it from the [writing_workflow directory](../writing_workflow/) directory through the `brown` package
- Thus, because we actually execute code from `brown`, you have the freedom to sketch some code blocks that have too much redundant code or imports
- Whenever a point is marked as Markdown you will use a Markdown cell, and when it's marked as code, or PYthon you will use a MagicPython cell that can be run and uses the `brown` package to import the real code
- Whenever I explictly state that is a Markdown section that explain a piece of code, you will still add the code, but instead of adding it in a Python magic cell, you will add it in ```python...``` blocks + the text description surrounding it. Remember that this notebook's scope is to explain the code, thus going in detail over every piece of code it's important, even if it's in Markdown.
- For example if I have the `(Markdown) First, we have to explain the Node, Toolkit and ToolCall interfaces from @nodes/base.py` intrusction, you will walk through all the states classes, such as Node, Toolkit, ToolCall, where you actually add all the code into the Notebook, but instead of adding it into a PYthon Magic Cell, you will add it into Markdown cells surrounded by ```python...``` as we don't want to make it executable. You will respect this strategy for all the sections or points marked as "Markdown"
- Whenever you have to output longer outputs, such as the article, guidelines, profiles, research, media items, use the `wrapped` function from @utils/pretty_print. You can use it directly as `from utils import pretty_print; pretty_print.wrapped(...)`. Find the full definition at `lessons/utils/pretty_print.py`. This is the interface of the pretty_print function:
```
def wrapped(
    text: Iterable[str] | dict | str,
    title: str = "",
    width: int = 100
) -> None:
...
```
- When writing code within Markdown blocks, be careful at relative imports such as `.mixins`. As these Markdown blocks from brown are seen in isolation they can be confusing, thus always replace them with the absolute path such as instead of `.mixins` do `brown.entities.mixins`. When you do this, look for the import path relative to the `brown` package from "../writing_workflow/src/brown"
- Whenever you have a ```python block that contains a nested ```mermaid block, use only two back quotes to espace the nested block, such as ``mermaid. You can apply the same logic to any type of combination of blocks.

## What files to include from the `brown` package
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

## What to completely exclude from the `brown` package
- Completely ignore as we will touch these only in future parts of the course:
    - evals
    - models/fake_model.py
    - observability
    - config_app.py (in more depth)


## Folders that we will download within the Notebook to always have at the same level with the Notebook:

- `configs`
- `inputs/evals`
- `inputs/examples`
- `inputs/profiles`
- `inputs/tests`

Thus, even if you reference these from other parts of the repository when generating the Notebook, these will always be relative to the notebook following the paths as stated above.

## Outline of the Notebook

1. Quick introduction on what we will explain in this notebook, that will include:
 - the orchestartor-worker pattern to generate media assets
 - the context engineering behind generating a high quality article
 - the code around modeling the entities using Pydantic and the workflow nodes using custom abstractions
 - the code used to manipulate Markdown files
2. Code setup section:
    - start with an intro identical with the example Notebook
    - curl the following folders to ensure we have everything we need in the same folder with the notebook: 
        1. configs using this exact command:
        ```bash
            !rm -rf configs
            !curl -L -o configs.zip https://raw.githubusercontent.com/iusztinpaul/agentic-ai-engineering-course-data/main/data/configs.zip
            !unzip configs.zip
            !rm -rf configs.zip
        ```
        1. inputs using this exact command:
        ```bash
            !rm -rf inputs
            !curl -L -o inputs.zip https://raw.githubusercontent.com/iusztinpaul/agentic-ai-engineering-course-data/main/data/inputs.zip
            !unzip inputs.zip
            !rm -rf inputs.zip
        ```
        3. Now run a `%ls` command to show the reader what we downloaded
    - (Markdown with Python code) highlight how we load the necessary environment variables within the settings object from @config. Also, how we initialize the settings thorugh the `get_settings()` function with the lru_cache to mimic the singleton pattern
3. Markdown only section on explaining at a high-level how the writing agent works:
    1. First step is to load all the necessary context into memory, such as the article guideline, research, few-shot-examples, and content generation profiles.
    2. Next step is to generate all the required media items using the orchestrator-worker pattern through the MediaGeneratorOrchestrator class + all it's available tools such as the MermaidDiagramGenerator tool
    3. The final step is to pass everything as context into the ArticleWriter which will output the article
    4. Generate a mermaid diagram with the 3 steps
    5. Explain that now we will walk people over these 3 steps
4. Code and Markdown section on explaining all the key components used as context. To do so, we will load each component into memory from `inputs/examples/course_lessons`, `inputs/profiles` and `inputs/tests/01_sample` directories, which we will use an example within this lesson. To do so, we will use the specialized loader classes from @loaders.py. Show to load them using the loaders from loaders.py that can be build using the factory pattern from builders.py. DO NOT SHOW the code from loaders.py and builders.py. Just explain how to use it. You can find examples on how we do it within the @workflows folder. Now go Through the ArticleGuideline, Research, ArticleProfiles, MediaItems, ArticleExamples pydantic models from the @entities folder and follow the next steps:
    1. (Python) set the input directories as constants
    2. ArticleGuideline: 
        - (Markdown) explain the pydantic entity from @entities of the ArticleGuideline and of the ContextMixin, load the examples into memory. Explain how the ContextMixin works and it's importance on standardizing how each entity can be mapped into a context element sorrounded by XML tags when added to a prompt. This is part of context engineering where we can easily swap objects between Python and context.
        - (Python) cut and show the first 1500 characters of the output. 
    3. Research: 
        - (Markdown) explain the pydantic entity from @entities of the Research, load the examples into memory
        - (Python) cut and show the first 1500 characters of the output
    4. Few Shot examples: 
        -  (Markdown) explain the pydantic entity from @entities of the ArticleExamples, load the examples into memory
        - (Python) scut and how the first 1500 characters of the output
    5. Profiles: 
        -  (Markdown) explain the pydantic entity from @entities, load the examples into memory
        - (Python) pretty print the size of each profile in characters
        - (Python) cut and show the first 1500 characters of the output of the ArticleProfile.
5. New Markdown section on the profiles, which are the key to the writing agent. These profiles will be passed to the writing agent. Thus, it's critical to understand how they work. Explain how each works:
    1. Mechaniscs Profile (General to any content type): Extract usage - link to `inputs/profiles/mechanics_profile.md`
    2. Structure Profile (General to any content type): Extract usage - link to `inputs/profiles/structure_profile.md`
    3. Terminology Profile (General to any content type): Extract usage  - link to `inputs/profiles/terminology_profile.md`
    4. Tonality Profile (General to any content type): Extract usage  - link to `inputs/profiles/tonality_profile.md`
    5. Article Profile (Specific to articles): Used to configure the content type you want to write  - link to `inputs/profiles/article_profile.md`
    6. Character Profiles (Specific to a given voice): Used to inject a particular voice, either yours, such as Paul Iusztin or from another popular character such as Richard Feynmann  - link to `inputs/profiles/character_profiles/paul_iusztin.md`
6. Code and Markdown section. Before starting to dig into the nodes and run the writing workflow, Make a paranthesis where we explain the `models` dir and how we actually call the models. Explain the code from @models/config and @models/get_model. 
    1. (Markdown) Start by explaining the `get_model` function
    2. (Markdown) Next move on to the `ModelConfig`, `SupportedModels` and `DEFAULT_MODEL_CONFIGS` structures
    3. (Markdown) Explain that we will use these methods to instantiate and configure models everywhere within the codebase and that the ModelConfig allows us to configure different models at each node from the workflow
    4. (Python) Run an example using `google_genai:gemini-2.5-flash`
7. Code and Markdown section. Finally digging into the nodes. New code section on explaining the MediaGeneratorOrchestrator node from @nodes/media_generator.py by following the next logic:
    1. (Markdown) First, we have to explain the Node, Toolkit and ToolCall interfaces from @nodes/base.py
    2. (Markdown) Explain the MediaGeneratorOrchestrator class from @nodes/media_generator.py that uses the interfaces from above. Split the logic as follows:
        2.1. The class + the init method
        2.2. The _extend_model method
        2.3. The ainvoke method
        2.4. Show the system_prompt_template
    3. (Markdown) Now we actually need to hook some workers as tools, thus explain the ToolNode interface from @nodes/base.py
    4. (Markdown) Explain the MermaidDiagramGenerator class from @nodes/tool_nodes.py that uses the interfaces from above. Split the logic as follows:
        4.1. The class + the init method
        4.2. The _extend_model method
        4.3. The ainvoke method
        4.4. Show the system_prompt_template (from the `Diagram Types & Examples` section provide only the `Process Flow` and `Flowcharts` ones)
    5. (Python) Give an example on how to use the MediaGeneratorOchestrator by generator 3 mermaid diagrams in parallel using it. You can see how it was used in @workflows/generate_article.py. This will be written in MagicPython as we want to run it and show the output.
    6. Run the example using `google_genai:gemini-2.5-flash`
8. Markdown section. Move to explaining the ArticleWriter node from @nodes/article_writer.py by following the next logic:
    1. Explain the ArticleWriter class from @nodes/article_writer.py that uses the interfaces from above. Split the logic as follows:
        1.1. (Markdown) The class + the init method 
        1.2. (Markdown) The ainvoke method (completely ignore the logic related to self.reviews, ArticleReviews, SelectedTextReviews - focus only on the logic for one-shot article generation, without reviews)
        1.3. (Markdown) Show the system_prompt_template (completely ignore the system prompts related to reviews, such as article_reviews_prompt_template and selected_text_reviews_prompt_template)
9. (Python) Final section showing how to run the whole thing, by brining in all the logic from previous sections and using the `inputs/tests/01_sample` directory as an example:
        - the article guideline, research, few shot examples and profiles
        - the media assets from the MediaGeneratorOrchestrator component
        - run the actual ArticleWriter class
        - show the first 200 lines of the output article
        - use the article renderer from @renderers to save the final article.md within the `inputs/tests/01_sample` directory
        - always use the `pretty_print_wrapped` when printing "=" separators, as the function already supports them out of the box
10. (Markdown) Future steps
    - What we've learnt:
        - context engineering: how to properly load all our elements for writing the article and format them using the `ContextMixin` 
        - how to implement from scratch the orchestrator-worker pattern for generating media items
        - how to actually write the article through our ArticleWriter node
    - Ideas on how you can further extend this code:
        - Add image and video generation support within the Orchestrator-Worker layer
        - Extend the writer to other media formats such as social media posts
    - next lessons we will learn how to automate reviewing and editing with the evaluator-optimizer pattern and how to glue everything together into a LangGraph workflow

## Examples
- [FastMCP](../16_fastmcp/notebook.ipynb)
- [Data Ingestion Lesson](../17_data_ingestion/notebook.ipynb)
- [Research Loops](../18_research_loop/notebook.ipynb)

- [Source code from where the code was extracted to created the notebooks](../research_agent_part_2/)


## Chain of Thought

- Carefully read the instructions
- First read the examples and understand how we expect to format the output Jupyter Notebook
- Understand how to structure the output following the pattern from the examples
- Generate the Jupter Notebook based on the `Outline of the Notebook`. Follow each point from the outline as is. We expect that each
bullet point will be present within the Jupyter Notebook