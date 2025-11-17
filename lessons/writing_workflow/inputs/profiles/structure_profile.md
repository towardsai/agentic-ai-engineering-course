## Sentence and paragraph length patterns

Write sentences 5–25 words; allow occasional 30-word 'story' sentences. Keep paragraphs ≤ 80 words; allow an occasional 1-sentence paragraph to emphasize a point.

- Good examples:
  - four 18-word sentence, as a paragraph of 72 words.
  - Ocassional 1-sentece paragraph.
- Bad examples:
  - Frequent 40-word run-ons.
  - five 18-word sentence, as a paragraph of 90 words.

## Paragraphs Structure

<paragraph_rules>
- Add a single idea per paragraph to make them skimmable and easy to follow. Group the paragraphs based on sentences with similar ideas to make skimming easier for people that already know specific topics. 
- Always group multiple sentences into a paragraph that share the same idea, entity, topic or subject. 
- **One idea, too long use case:** Be extremely careful not to make the paragraph longer than 3 sentences. In case it exceeds 3 senteces, split the paragraph into two sub-entities, sub-topics or sub-subjects.
- **Multiple ideas per paragraph use case:** Pay special attention to talk about only one idea, entity, topic or subject in a paragraph. Two ideas per paragraph are not accepted. Whenever starting talking about the second idea, start a new paragraph.
</paragraph_rules>

## Bulleted lists and numbered

Avoid fragmentation such as having too many subheadings or bullet points. Write in detail and in full paragraphs, avoiding bullet points or listicles when possible. Use bulleted or numbered lists only when it makes sense such as:

- Shortly iterating over a set of more than 2 items using a list. If we want to write only a sentence about each item a list is fine, otherwise for more than one sentence per item, use paragraphs instead.
- Using numbered lists to divide and iterate over big chunks of information, such as code. In this example, the numbered list will help the reader keeing track of the "logical-group" where the information comes from.

## Sections Sub-Heading Formatting

Use at maximum H3, or "###" in Markdown format, sub-headers to split multiple topics within a section. To keep the content easy to follow, we want to keep it as flat as possible. Thus, avoid using H4/H5/H6 sub-headings at all costs. If you felt the need to write a H4/H4/H6 sub-header split the H3 headers into multiple sections or create a bullet/numbered list.

- Good:
   """
   ## Section Title
   ...
   ### Sub-section Title
   ...
   """
- Bad:
   """
   ## Section Title
   ...
   ### Sub-section Title
   ...
   #### Sub-sub-section Title
   ...
   """

**The content is written in Markdown format. The introduction, conclusion and section titles use H2 headers, marked as `##`. The sub-section titles use H3 headers, marked as `###` in Markdown. Never go deeper than H3 or `###` to reflect a sub-sub-section, such as using H4 or `####` for sub-sub-sections. The introduction and conclusion do not have sub-sections.**

- Good examples:
  - "## Introduction ... ## Section 1 ... ### Sub-section 1 ... Section 2 ... ## Section N ... ## Conclusion ..."
- Bad examples:
  - "# Introduction ... # Section 1 ## Sub-section... # Section 2 ... # Section N ... # Conclusion ..."
  - "## Introduction ... ## Section 1 ... #### Sub-section 1 ... Section 2 ... #### Sub-section 2 ... ## Section N ... # Conclusion ..."

## Formatting Callouts and Side Notes

Through callouts and side notes we want to highlight auxiliary information that is only adjancent to a given section. 

Format all the callouts or side notes as follows:
<callout_example>
<aside>
💡 ... Callout text.
</aside>
</callout_example>

## Using Bolding, Italicizing, Quotes and Backticks

Avoid overusing bold, italic, quaotes or backticks. Use them only in the following scenarios:
- **Bold:** Main subjects within a text or list. Important information.
- **Italic + Quotes:** Paraphrasis text from other sources or mimicing a person talking 
- **Backticks:** Inline code

## Formatting Media

Formatting of images, tables, and Mermaid diagrams and their corresponding caption text.

We handle 3 types of media:
1. Tables as Markdown tables
2. Diagrams as Mermaid diagrams
3. Images as URLs from the <research> tags

All the media items have a unique identifier, which is a number from 1 to N. The identifier is independent from the 
citation identifier. The images, tables or other media are numbered sequentially, starting from 1 to N for each type of media. 
For example: Image 1, Image 2, Table 1, Table 2, etc.

All the media items will contain at the bottom a caption text with the identifier and a small description of the media due
to potential copyrights restrictions and to make it transparent to the reader.

The caption is formatted as follows:
1. <table_caption>
Table <media_identifier>: <table description>
</table_caption>

2. <diagram_caption>
Image <media_identifier>: <diagram description>
</diagram_caption>

3. <image_caption>
Image <media_identifier>: <image description> (Source [<citation_name> ([<citation_identifier>])?](<citation_url>))
OR in case you know both the author and the source name
Image <media_identifier>: <image description> (Image by <author_name> from [<citation_name> ([<citation_identifier>])?](<citation_url>))
</image_caption>
With an optional citation identifier if the source is taken from our reference list.

Examples on how to write the caption:
<caption_examples> 
   - Bad: "Image 1: Workflows vs. Agents: What would it be?"
     Good: "Image 1: Workflows vs. Agents: What would it be? (Source [Frankie's Legacy](https://frankieslegacy.co.uk/take-the-red-pill-take-the-blue-pill-the-choices-and-decisions-we-make))"
   - Bad: "Image 3: The autonomy slider, showing the trade-off between control and autonomy."
     Good: "Image 3: The autonomy slider, showing the trade-off between control and autonomy. (Image by Anca Ioana Muscalagiu from [the Decoding AI Magazine [3]](https://decodingml.substack.com/p/llmops-for-production-agentic-rag))"
   - Bad: "Image 4: A flowchart illustrating the benefits of structured outputs from LLMs, acting as a bridge between LLM (Software 3.0) and Python (Software 1.0) for downstream processing."
     Good: "Image 4: The benefits of structured outputs from LLMs, acting as a bridge between LLM (Software 3.0) and Python (Software 1.0) for downstream processing."
   - Bad: "The scientific method for evaluating and optimizing AI systems."
     Good: "Image 5: The scientific method for evaluating and optimizing AI systems."
   - Bad: "Table 1: A table showing the performance difference between Gemini and Grok."
     Good: "Table 1: The performance difference between Gemini and Grok."
   - Bad: "Image 2: A flowchart showing the parallelization pattern applied to our writing workflow."
     Good: "Image 2: The parallelization pattern applied to our writing workflow."
</caption_examples>

The user will specify in the <article_guideline> what media type to include and where to include it in the article. 
For example, "Add an image or figure of ...", "Add a table of ...", "Render a diagram of ...".

As you write, incorporate the requested media item at the most relevant place within the requested section 
to support your points with visual information

The tables and diagrams are generated internally based on the context from the <research> and <article_guideline> tags. In case the mermaid diagram is already rendered, if you have a specialized Mermaid diagram generator tool available, pass the old Mermaid diagram within description and use specialized tool to regenerate and improve the old Mermaid diagram.

The images will be passed as URLs directly from the <research> or <article_guideline> tags.  The URLs will be rendered as Markdown as follows:
<image_format>
"![<source_name>](<image_url>)\n<image_caption>...</image_caption>"
</image_format>
To understand what image to use where you will interpret both the <image_url> and <image_caption>.

Formatting rules for media handling:
- Replace all the XML placeholders with the actual values.
- In the <image_format> -> <image_url> XML placeholder make sure to add the full URL of the image. For example,
add https://www.some_url.com/image.png, not only image.png.
- In the <image_caption> XML placeholder, make sure to add the citation requirements: <citation_name>, <citation_identifier> (if available), <citation_url>

<correction_media_rules>
- Ensure that the image and table numbering are in order, starting from 1, 2, 3, to N, with an increment of one between images.
- Ensure that the image and tables have different numbering, such as Image 1, Image 2, ... Image N and Table 1, Table 2, ... Table N
- The media captions are properly formatted as shown in the <caption_examples>
</correction_media_rules>

## Referencing Media

Whenever talking about something that is supported by the images or tables, we should point/reference the reader to the media. For example:
- "This is illustrated in Image 1..."
- "As we can see in Table 1..."
- "...should look like the one presented in Figure 2."

## Formatting Code

- When working with code snippets, avoid describing big chunks of code that go over 35 lines of code. Instead, split the code into logical groups based on lines with similar logic, and describe each code snippet from the group individually. Splitting rules: 
  - you should split the code by: class, methods or functions if the class is big
  - similar logic lines if the function or method is too big
  - create a one-liner group if the single line makes sense on its own.
  - keep only the essential code snippets by keeping only essential imports, logs or comments. 
  - if it's a class, keep the class name in the first group and index the rest of the methods in future groups to reflect that they are part of that class.

- Good examples:
  - "[Section introduction on what the code is about] 1. [Describe Code Group 1] [Code Group 1] 2. [Describe Code Group 2] [Code Group 2] ... N. [Describe Code Group N] [Code Group N] [Section final thoughts on the code]."
  - "[Describe the code] [Small chunk of code that's under 35 lines] [More thoughts on the code]"
- Bad examples:
  - "[Describe the code] [Huge chunk of code that goes over 35 lines] [More thoughts on the code]"

### Working with Jupyter Notebooks

**When working with code snippets from Jupyter Notebooks we want to leverage the Markdown and output cells to support our ideas.**

We want to mimic the structure and experience from the Notebook.

For example in a Jupyter Notebook we usually have cells like this:
- Markdown cell with some description of the code (optional)
- Code cell containg the exact code
- Output cell showing the result of the code (optional)
Or some derivate of this, as the Markdown our output cells are optional.

When we want to render code blocks from Jupyter Notebooks as <research> we want to leverage the Markdown cells as they contain value insights and output cells as we want to always show to the user the output after running the code.

<good_example>
Jupyter Notebook contains:
- [Markdown description]
- [Python code]
- [Python code output]
We render that in our content as:
1. [Description of the code].
    ```python
    [code]
    ```
    It outputs:
    ```text
    [output]
    ```
2. [Continue]...
</good_example>

<bad_example>
Jupyter Notebook contains:
- [Markdown description]
- [Python code]
- [Python code output]
We render that in our content as:
1. [Description of the code].
    ```python
    [code]
    ```
2. [Continue]...
</bad_example>

### Grouping big chunks of code

Break and group long code sections into shorter code blocks followed by brief explanations so users can follow step-by-step.

When working with code snippets, avoid describing big chunks of code that go over 35 lines of code. Instead, split the 
code into logical groups based on lines with similar logic, and describe each group individually. 

Grouping and formatting rules:
<code_grouping_rules>
- Split the code by: class, methods or functions if the class or file is big
- We can also split by actionability in use cases where we have access to code description and outputs. For example, when a user calls a particular class or function and shows the output, can be a single block.
- Group lines with similar logic if the function or method is too big,
- Show a one-liner if the single line makes sense on its own. 
- Keep only the essential code snippets by showing only essential imports, logs or comments.
- If it’s a class, keep the class name in the first group and index the rest of the methods to reflect that they are part of the same class.
- The code descriptions ALWAYS end with “.” (dot), while the output ALWAYS ends with “:”.
- If available, always show the code output from a particular group to make the text easier to understand through examples.
</code_grouping_rules>

General output format for a group of code snippets:
<code_output_format>
1. [Description Code Snippet 1].
   [Code Snippet 1]
    
2. [Description Code Snippet 2].
   [Code Snippet 2]
   ...
    
N. [Description Code Snippet N].
   [Code Snippet N]

[Final conclusion of the code].
</code_output_format>

In case the explained code contains only one code snippet, the output format will be:
<code_output_format_single_snippet>
[Description Code Snippet 1].
[Code Snippet 1]
[Final conclusion of the code].
</code_output_format_single_snippet>

For example, if the user asks you to explain a group of code snippets, such as:
<example_of_input_group_of_code_snippets>
- Group of code snippets 1:
        1. Explain the prompt (Quick note on how we wrapped up the JSON example and document context with XML tags)
        2. Call the model
        3. Print the output
</example_of_input_group_of_code_snippets>

The output will be:
<example_of_output_group_of_code_snippets>
1. [Explanation of the prompt].
   [Prompt Code Snippet]

2. [Explanation of the model call].
   [Model Call Code Snippet]

3. [Explanation of the output].
   [Output Code Snippet]

[Final conclusion of the code].
</example_of_output_group_of_code_snippets>

Another example:
<example_of_input_single_code_snippet>
- Group of code snippets 1:
        1. Define the Gemini `client` and `MODEL_ID` constant
        2. Show the example `DOCUMENT`
</example_of_input_single_code_snippet>

The output will be:
<example_of_output_group_of_code_snippets>
1. [Describe the Gemini code].
   [Gemini Code Snippet]

2. [Show the `DOCUMENT`].
   [Output `DOCUMENT`]

[Final conclusion of the code].
</example_of_output_group_of_code_snippets>

One more example with interleaving code blocks with text:
<example_of_input_single_code_snippet>
- Group of code snippets 1:
        1. Explain the prompt
        2. Call the model
        3. Provide a quick note on why we need XML tags.
        4. Print the output
</example_of_input_single_code_snippet>

The output will be:
<example_of_output_group_of_code_snippets>
1. [Describe prompt].
   [Prompt code snippet]

2. [Describe how we call the model].
   [Call the mode code snippet]

3. [Note on XML tags].

4. [Description of the output]
   [Output code snippet]

[Final conclusion of the code].
</example_of_output_group_of_code_snippets>

**Avoid having blocks with multiple turns.** A block should have a maximum of one code description, the actual code and one or multiple code outputs. When requiring the split the code within one block into two snippets, create a new block instead. The only thing that we can split into multiple parts within one block are the code outputs.

Here is an good example of a block with one description, code and output turns:
<good_example>
1. Let's test it. We send a user prompt along with our system prompt to the model.
  ```python
  import json
  from google import genai

  client = genai.Client()

  USER_PROMPT = "Use Google Search to find recent articles about AI agents."

  messages = [TOOL_CALLING_SYSTEM_PROMPT.format(str(TOOLS_SCHEMA)), 
              USER_PROMPT]

  response = client.generate_content(
      model="gemini-2.5-flash",
      contents=messages,
  )
  ```
  The LLM correctly identifies the google_search tool and generates the required arguments:
  ```text
  <tool_call>
  {"name": "google_search", "args": {"query": "recent articles about AI agents"}}
  </tool_call>
  ```
</good_example>

Another good example with multiple code outputs:
<good_example>
1. We create a tool registry to map tool names to their handlers and schemas.
  ```python
  TOOLS = {
      "google_search": {
          "handler": google_search,
          "declaration": google_search_schema,
      },
      "perplexity_search": {
          "handler": perplexity_search,
          "declaration": perplexity_search_schema,
      },
      "scrape_url": {
          "handler": scrape_url,
          "declaration": scrape_url_schema,
      },
  }

  TOOLS_BY_NAME = {tool_name: tool["handler"] for tool_name, tool in TOOLS.items()}
  TOOLS_SCHEMA = [tool["declaration"] for tool in TOOLS.values()]
  ```
  The TOOLS_BY_NAME mapping looks like this:
  ```json
    {'google_search': <function google_search at 0x...>, 
  'perplexity_search': <function perplexity_search at 0x...>, 
  'scrape_url': <function scrape_url at 0x...>
  }
  ```
  And here is an example schema from TOOLS_SCHEMA:
  ```json
    {
      "name": "google_search",
      "description": "Tool used to perform Google web searches and return ranked results.",
      "parameters": {
          "type": "object",
          "properties": {
              "query": {
                  "type": "string",
                  "description": "The search query."
              }
          },
      },
  }
  ```
</good_example>

Here is a bad example where we split the code from a single group into multiple turns:
<bad_example>
1. Let's test it. We send a user prompt along with our system prompt to the model. First we define our model client:
  ```python
  import json
  from google import genai

  client = genai.Client()
  ```
  Then we define our prompt and call the model:
  ```python
  USER_PROMPT = "Use Google Search to find recent articles about AI agents."

  messages = [TOOL_CALLING_SYSTEM_PROMPT.format(str(TOOLS_SCHEMA)), 
              USER_PROMPT]

  response = client.generate_content(
      model="gemini-2.5-flash",
      contents=messages,
  )
  ```
  The LLM correctly identifies the google_search tool and generates the required arguments:
  ```text
  <tool_call>
  {"name": "google_search", "args": {"query": "recent articles about AI agents"}}
  </tool_call>
  ```
</bad_example>

### Rendering code blocks

As we render everything in Markdown, when outputing code blocks, you will use the following format:
<code_snippet_format>
```python
code_snippet_content
```
</code_snippet_format>
You will replace the ```python with any other language you want. For example, ```python will be replaced with ```bash.

**To avoid any rendering problems**, if there are any ``` blocks inside the code block that has to be rendered, you will replace them with XML. For example, if the code block contains:
<wrong_code_snippet_format>
```python
```tool_call
some code
```
</wrong_code_snippet_format>

You will replace it with:
<correct_code_snippet_format>
```python
<tool_call>
some code
</tool_call>
```

### Using comments

Avoid commenting the code snippets. Use comments only to explain the code.

### Referencing previous code blocks

Ensure you are not duplicating code blocks. If a function, class or method was previously defined within the content reference it instead of duplicating the code. Thus, never define the same function or class twice. 

We can reference the class by specifying in which chapter, section or paragraph it was first mentioned.

For example, instead of defining the `ArticleWriter` class again say things such as:
- "Using the same `ArticleWriter` class we defined in section `Let's Define Our Node`, we can..." 
- "Using the same `ArticleWriter` class we defined in the previous section, we will..."
Or any other formulation as long we reference the code and not duplicate it.

## Citation Rules

Whenever you take information from the <research> tags, you will cite it using the following citation rules:
<citation_guideline>
- Avoid citing unnecessarily: Not every statement needs a citation. Focus on citing key facts, conclusions, 
and substantive claims that are linked to sources rather than common knowledge. Prioritize citing claims that 
readers would want to verify, that add credibility to the argument, or where a claim is clearly related to a 
specific source
- Cite meaningful semantic units: Citations should span complete thoughts, findings, or claims that make sense as 
standalone assertions. Avoid citing individual words or small phrase fragments that lose meaning out of context; 
prefer adding citations at the end of sentences
- Minimize sentence fragmentation: Avoid multiple citations within a single sentence that break up the flow of 
the sentence. Only add citations between phrases within a sentence when it is necessary to attribute specific 
claims within the sentence to specific sources
- No redundant citations close to each other: Do not place multiple citations to the same source in the same 
sentence, because this is redundant and unnecessary. If a sentence contains multiple citable claims from the 
same source, use only a single citation at the end of the sentence after the period
</citation_guideline>

<citation_guideline_technical_requirements>
- Citations result in a visual, interactive element being placed at the closing tag. Be mindful of where the closing
tag is, and do not break up phrases and sentences unnecessarily
- The format is in Markdown as follows: [[identifier]](link/url/uri) where identifier is a number from 1 to N
- The identifier is unique across the whole article. Thus, when adding a citation, first check if the
source has already been cited. If it has, use the same identifier. If it has not, use a new identifier.
- The identifier of the citation is independent from other identifiers, such as the image number.
- Add the citations only at the end of a paragraph before the final period. For example, 
"This is a paragraph [[1]](link_1), [[2]](link_2), ... [[N]](link_N)."
- citation identifiers are correctly numbered, where a tuple (identifier, source) is unique, the identifier is unique
and a source is assigned to a single identifier. In case of adding INNCORECT citations such as 
"[[1]](link_1), [[1]](link_2)" or "[[1]](link_1), [[2]](link_1)" the simplest solution to fix this is to reassign new 
identifiers to all the sources, such as [[1]](link_1), [[2]](link_2), from the introduction to the conclusion.
</citation_guideline_technical_requirements>
