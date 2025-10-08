This lesson is the first lesson of part 2 of a course where we're teaching how to build AI agents for production.

The lesson is titled "Central Project: Scope & Design".
Basically, this course has a part 1 where it teaches basic AI agent concepts.
Then, in part 2, the course goes more into practice and, step by step, teaches how to build two agents: a research agent and a writer agent.
The research agent is an actual agent that works with an MCP server and an MCP client, using `fastmcp`.
The writer agent is more like a workflow agent that works with `langgraph`.

So, in the two projects, we see the two main types of agents that are used in production:
- An adaptable and steerbable research agent that works with an MCP server, which can easily diverge from its main workflow to research different topics and adapt.
- A reliable writer agent whose task is better defined but harder, which needs less adaptation but more reliability.

In this lesson, we provide the high-level architecture and reasoning behind the two agents, and we explain the experience that we had while learning about the topic and building the agents.
- When we started working on the course, we weren’t sure about whether to use agents frameworks/libraries (and which ones) or not. The field is evolving very quickly, so frameworks/libraries can rise up and down just as quickly. It’s to early to predict which will be the ones that will last for multiple years and will become the standard of the field. We're not sure that it’s possible to say that there’s a standard in the field right now, it’s too early.
- So, we started reading the documentations of the existing frameworks, trying to understand which ones are good for quick experimentation and demoing, and which ones are good for production. Also, we considered the current adoptions from the AI developers.
- LangGraph is currently doing great in both adoption and production readiness. It has features to save the state of the computations of the agent/workflow, allowing to resuming it at different points. It allows easy monitoring. And other stuff not implemented by other frameworks/libraries. In contrast, LangGraph has a steeper learning curve. LangGraph seems to be very good for workflows.
- So, we decided to give a try to LangGraph and start using it for the research agent and the writing agent.
- Next, we reasoned about the structure of the research agent and the writing agent. Should they both be agents? Or workflows? What are the pros and cons? We made some assumptions but then we preferred starting working on them and trying them, to get a feel of what would work best.
- We started the reasoning agent as mainly a workflow. Then, by trying it, we noticed that it is a rather interactive process where human feedback is useful, and so that it's possible to stop its execution and resume it from a certain point. So, to make the workflow more flexible, we moved some steps of it to tools, and, over time, the overall structure became that of a full agent. Now, the research agent is simply an MCP server that provides ~10 MCP tools and 1 MCP prompt that describes an “agentic workflow” with them, that is, a recipe of how to use the tools to do a research. We took this opportunity to create also a simple and generic MCP client for it, to teach how to create both MCP servers and MCP clients. However, the MCP server can be used with any MCP client (e.g. Cursor, Claude Code, etc).
- Then, we looked at the current best libraries for structuring MCP clients and servers, and FastMCP has both a great adoption and a lot of features. For structuring MCP servers and clients, the FastMCP python library is currently the standard of the industry, for this reason we decided to use it for the research agent (so, no more LangGraph here).
- Meanwhile, we also worked on the "writing agent", but implementing all of it as a workflow (with LangGraph). Here, we noticed that the "process of writing sections, introductions, conclusions, etc by checking if the text is adhering to style guidelines and following the provided article script" is more prone to be exactly a workflow. The process is always the same, there’s not a lot of adaptation involved. The workflow is rather complex and long, but LangGraph allows to keep it well organized.
- It's hard to decide what framework to choose. Each one has its philosophy. It's a field that is evolving a lot and many libraries and frameworks can appear and get quick traction. It's hard to predict which one will win over time. With hindsight, it seems always easy to explain why a particular framework has succeeded instead of another one, but it's hard to predict it in advance. For this reason, we try not to lose ourselves in the details of the frameworks, but rather focus on the concepts and the ideas. Also for this reason, we were in favour of building a project with two agents with the two main types (which are currently better covered by different frameworks): an adaptable and steerable research agent (better served by MCP servers and `fastmcp`) and a reliable writer agent (better served by workflows and `langgraph`). But again, while the choice of the frameworks seems to be a good one today, it's not the only one and we may change our minds in the future. Our advice is to read the main concepts and philosophies of the frameworks and libraries and choose the one that seem to better fit your needs, and experiment with a lot of them. Indeed, also for these two agents, their final design was not obvious in advance. We tried implementing the research agent as workflows at first, only to find out (while testing it) that it needed more steerability. We wanted to teach only a single framework in the course (langgraph, as it's currently the most popular framework for workflows and ready for production), but we ended up using two frameworks for the two agents as a consequence. This shows that there isn't a clear winner yet in terms of the best framework/library for AI agents.
- The next lesson will talk more about the specific frameworks and libraries that have the most traction today and their philosophies, so no need to talk a lot about them in this lesson.
- For the task of the project (building an article about a topic), we considered the "build vs buy" tradeoff. Indeed, there are already several "deep research" tools available. This lesson should acknowledge the strengths of tools from OpenAI, Google, or Perplexity—fast, polished and convenient. However, we found that we needed something more steerable and interactive for the research part: we want to be able to change topics if the agent is researching not useful ones, and give feedback. The "deep research" tools are not very good at this. Moreover, as part of the research, we want to be able to read GitHub repositories and their code, and transcribe YouTube videos, and use some local files, and this is not easy to do with the "deep research" tools. Also for the writing task, we found that we needed something more deterministic and reliable than the "deep research" tools. The "deep research" tools are not very good at this. Moreover, as part of the writing, we want to be able to check if the text is adhering to style guidelines and following the provided article script, and give feedback. The "deep research" tools are not very good at this. So, we decided to build a custom system for the research and writing tasks, which is the main project of this course, which you'll see in the next lessons. We learned this lesson the hard way. “At first, we thought we could bend off-the-shelf tools into our process. But every time we needed guarantees—like enforcing a style rule or piping results into our CI/CD pipeline—they broke. These failures shaped our decision to build a custom system.”
- Then, this lesson should explain the overall design of the two agents, and the reasoning behind it.
- The research agent is organized as a collection of MCP tools in an MCP server, and an MCP prompt that describes its whole agentic workflow, that is, a recipe of how to use the tools to do a research. As input, the research agent expects a research directory, which contains an article guideline file. The article guideline file basically contains a description of what the final lesson should be about, what is the expected audience, the expected length of the article, the expected outline, some good sources for it, etc. Using this file, the goal of the research agent is to do a research about the topic, and to produce a research file that contains the research data. This research file will then used by the writing agent, together with the original article guideline file, to write the final article. Here, it would be great to show a simple Mermaid diagram showing this: that the research agent wants as input the article guideline and produces a research file, and the writing agent wants as input the article guideline and the research file (and some style guidelines) and produces a final article.
- This is how the research agent works, as described in its MCP prompt, which defines its whole agentic workflow. The lesson should show this prompt, and describe each tool and its purpose. It would be great to show also a simple Mermaid diagram showing this workflow.
<research_agent_architecture>
Your job is to execute the workflow below.

All the tools require a research directory as input.
If the user doesn't provide a research directory, you should ask for it before executing any tool.

**Workflow:**

1. Setup:

    1.1. Explain to the user the numbered steps of the workflow. Be concise. Keep them numbered so that the user
    can easily refer to them later.
    
    1.2. Ask the user for the research directory, if not provided. Ask the user if any modification is needed for the
    workflow (e.g. running from a specific step, or adding user feedback to specific steps).

    1.3 Extract the URLs from the ARTICLE_GUIDELINE_FILE with the "extract_guidelines_urls" tool. This tool reads the
    ARTICLE_GUIDELINE_FILE and extracts three groups of references from the guidelines:
    • "github_urls" - all GitHub links;
    • "youtube_videos_urls" - all YouTube video links;
    • "other_urls" - all remaining HTTP/HTTPS links;
    • "local_files" - relative paths to local files mentioned in the guidelines (e.g. "code.py", "src/main.py").
    Only extensions allowed are: ".py", ".ipynb", and ".md".
    The extracted data is saved to the GUIDELINES_FILENAMES_FILE within the NOVA_FOLDER directory.

2. Process the extracted resources in parallel:

    You can run the following sub-steps (2.1 to 2.4) in parallel. In a single turn, you can call all the
    necessary tools for these steps.

    2.1 Local files - run the "process_local_files" tool to read every file path listed under "local_files" in the
    GUIDELINES_FILENAMES_FILE and copy its content into the LOCAL_FILES_FROM_RESEARCH_FOLDER subfolder within
    NOVA_FOLDER, giving each copy an appropriate filename (path separators are replaced with underscores).

    2.2 Other URL links - run the "scrape_and_clean_other_urls" tool to read the `other_urls` list from the
    GUIDELINES_FILENAMES_FILE and scrape/clean them. The tool writes the cleaned markdown files inside the
    URLS_FROM_GUIDELINES_FOLDER subfolder within NOVA_FOLDER.

    2.3 GitHub URLs - run the "process_github_urls" tool to process the `github_urls` list from the
    GUIDELINES_FILENAMES_FILE with gitingest and save a Markdown summary for each URL inside the
    URLS_FROM_GUIDELINES_CODE_FOLDER subfolder within NOVA_FOLDER.

    2.4 YouTube URLs - run the "transcribe_youtube_urls" tool to process the `youtube_videos_urls` list from the
    GUIDELINES_FILENAMES_FILE, transcribe each video, and save the transcript as a Markdown file inside the
    URLS_FROM_GUIDELINES_YOUTUBE_FOLDER subfolder within NOVA_FOLDER.
        Note: Please be aware that video transcription can be a time-consuming process. For reference,
        transcribing a 39-minute video can take approximately 4.5 minutes.

3. Repeat the following research loop for 3 rounds:

    3.1. Run the "generate_next_queries" tool to analyze the ARTICLE_GUIDELINE_FILE, the already-scraped guideline
    URLs, and the existing PERPLEXITY_RESULTS_FILE. The tool identifies knowledge gaps, proposes new web-search
    questions, and writes them - together with a short justification for each - to the NEXT_QUERIES_FILE within
    NOVA_FOLDER.

    3.2. Run the "run_perplexity_research" tool with the new queries. This tool executes the queries with
    Perplexity and appends the results to the PERPLEXITY_RESULTS_FILE within NOVA_FOLDER.

4. Filter Perplexity results by quality:

    4.1 Run the "select_research_sources_to_keep" tool. The tool reads the ARTICLE_GUIDELINE_FILE and the
    PERPLEXITY_RESULTS_FILE, automatically evaluates each source for trustworthiness, authority and relevance,
    writes the comma-separated IDs of the accepted sources to the PERPLEXITY_SOURCES_SELECTED_FILE **and** saves a
    filtered markdown file PERPLEXITY_RESULTS_SELECTED_FILE that contains only the full content blocks of the accepted
    sources. Both files are saved within NOVA_FOLDER.

5. Identify which of the accepted sources deserve a *full* scrape:

    5.1 Run the "select_research_sources_to_scrape" tool. It analyses the PERPLEXITY_RESULTS_SELECTED_FILE together
    with the ARTICLE_GUIDELINE_FILE and the material already scraped from guideline URLs, then chooses up to 5 diverse,
    authoritative sources whose full content will add most value. The chosen URLs are written (one per line) to the
    URLS_TO_SCRAPE_FROM_RESEARCH_FILE within NOVA_FOLDER.

    5.2 Run the "scrape_research_urls" tool. The tool reads the URLs from URLS_TO_SCRAPE_FROM_RESEARCH_FILE and
    scrapes/cleans each URL's full content. The cleaned markdown files are saved to the
    URLS_FROM_RESEARCH_FOLDER subfolder within NOVA_FOLDER with appropriate filenames.

6. Write final research file:

    6.1 Run the "create_research_file" tool. The tool combines all research data including filtered Perplexity results
    from PERPLEXITY_RESULTS_SELECTED_FILE, scraped guideline sources from URLS_FROM_GUIDELINES_FOLDER,
    URLS_FROM_GUIDELINES_CODE_FOLDER, and URLS_FROM_GUIDELINES_YOUTUBE_FOLDER, and full research sources from
    URLS_FROM_RESEARCH_FOLDER into a comprehensive RESEARCH_MD_FILE organized into sections with collapsible blocks
    for easy navigation. The final RESEARCH_MD_FILE is saved in the root of the research directory.

Depending on the results of previous steps, you may want to skip running a tool if not necessary.

**Critical Failure Policy:**

If a tool reports a complete failure, you are required to halt the entire workflow immediately. A complete failure
is defined as processing zero items successfully (e.g., scraped 0/7 URLs, processed 0 files).

If this occurs, your immediate and only action is to:
    1. State the exact tool that failed and quote the output message.
    2. Announce that you are stopping the workflow as per your instructions.
    3. Ask the user for guidance on how to proceed.

**File and Folder Structure:**

After running the complete workflow, the research directory will contain the following structure:

```
research_directory/
├── ARTICLE_GUIDELINE_FILE                           # Input: Article guidelines and requirements
├── NOVA_FOLDER/                                     # Hidden directory containing all research data
│   ├── GUIDELINES_FILENAMES_FILE                    # Extracted URLs and local files from guidelines
│   ├── LOCAL_FILES_FROM_RESEARCH_FOLDER/           # Copied local files referenced in guidelines
│   │   └── [processed_local_files...]
│   ├── URLS_FROM_GUIDELINES_FOLDER/               # Scraped content from other URLs in guidelines
│   │   └── [scraped_web_pages...]
│   ├── URLS_FROM_GUIDELINES_CODE_FOLDER/          # GitHub repository summaries and code analysis
│   │   └── [github_repo_summaries...]
│   ├── URLS_FROM_GUIDELINES_YOUTUBE_FOLDER/       # YouTube video transcripts
│   │   └── [youtube_transcripts...]
│   ├── NEXT_QUERIES_FILE                           # Generated web-search queries with justifications
│   ├── PERPLEXITY_RESULTS_FILE                     # Complete results from all Perplexity research rounds
│   ├── PERPLEXITY_SOURCES_SELECTED_FILE            # Comma-separated IDs of quality sources selected
│   ├── PERPLEXITY_RESULTS_SELECTED_FILE            # Filtered Perplexity results (only selected sources)
│   ├── URLS_TO_SCRAPE_FROM_RESEARCH_FILE          # URLs selected for full content scraping
│   └── URLS_FROM_RESEARCH_FOLDER/                 # Fully scraped content from selected research URLs
│       └── [full_research_sources...]
└── RESEARCH_MD_FILE                                 # Final comprehensive research compilation
```

This organized structure ensures all research artifacts are systematically collected, processed, and made easily
accessible for article writing and future reference.
</research_agent_architecture>
- Now we can describe the writing agent. The writing agent is organized as a LangGraph workflow, which is a collection of nodes and edges that define the flow of the workflow. The workflow is broken into stages that outline, draft, reflect, iteratively edit, globally refine, and finalize the article.
- The main problem with the writing task is that "An LLM draft often reads like it was written by a brilliant but undisciplined junior writer—knowledgeable, but verbose, generic, and prone to hedging. You can always hear the ‘AI sound.’", and that it's very hard to steer the writing style and quality of an LLM. We learned this the hard way: our early drafts looked polished at first glance, but on reread were full of fluff, clichés, and missing citations. These failures forced us to adopt a checklist of manual editorial tricks—SOPs we now treat as automation targets. We'll talk more about what we specifically did for managing this in the next lessons.
- As support for this, we can add a vivid example of “AI slop.” Use the word “delve” as an example: after 2023, its frequency in professional and scientific writing spiked—not because humans suddenly loved it, but because LLMs overproduced it. Explain that this is a real symptom of a feedback loop gone wrong (i.e. an unwanted consequence of the reinforcement learning finetuning process of LLMs).
- Here follows the architecture of the writing agent.
<writing_agent_architecture>
## Brown — LangGraph-based AI Article Writing Agent

Brown is a multi-stage AI writing agent built with LangGraph and LangChain. It takes a structured guideline and research as inputs and produces a publish-ready technical article with title, SEO metadata, and reflection scores. The workflow is broken into stages that outline, draft, reflect, iteratively edit, globally refine, and finalize the article.

### What Brown Produces
- A complete technical article in Markdown with: title, introduction, body sections, conclusion, and optional references
- Iterative stage artifacts for traceability (Stage 1/2/3 Markdown files)
- Title and SEO metadata
- Reflection scores table (per stage, per unit)
- Style guideline materialized for the run
- Optional graph diagram of the workflow

## Workflow Stages and Nodes

Below is a concise walk-through of the main nodes.

### Stage 0 — Context Gathering
1) Parse all inputs (guideline, research, style guideline, examples, evaluation rules, writer profile)

### Stage 1 — Outline and One-shot Draft
2) Plan introduction, sections, conclusion (respect pre-defined sections in guideline)
3) Draft an end-to-end article using outline, research, style, profile
4) Evaluate draft against evaluation rules; apply targeted one-shot edits
5) Parse Markdown into introduction, sections, conclusion, references; render `article_stage_1.md`

### Stage 2 — Iterative Section Editing Loop
6) Score each section against evaluation rules
7) If less than 90% of the section checks are ok, then apply targeted changes guided by reflection results. Iterate this step until at least 90% of the section checks are ok, or until the maximum number of iterations is reached.

### Stage 3 — Global Reflection and Finalization
8) Evaluate entire article and log Stage 3 reflection scores
9) Apply global edits guided by reflection results
10) Produce final title/subtitle
11) Produce SEO title/description (requires title)
12) Save final article, metadata, style guideline, and reflection scores
</writing_agent_architecture>
- Explain that we reached the architecture/workflow above after trying a lot of different approaches, with the goal of finding the best one that allowed for flexibility and steerability in the writing style and quality, which we found to be very hard using LLMs.