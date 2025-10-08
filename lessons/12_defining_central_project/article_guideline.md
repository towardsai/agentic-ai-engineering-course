## Global Context of the Lesson

### What We Are Planning to Share

This lesson scopes and designs the **central project** that students will implement across the next lessons: a **Research agent and a Writing agent** that together produce a publish‑ready technical article.

We explain **why we split the system into two complementary agent types**, how we arrived there (including false starts), and what each will **input**, **produce**, and **guarantee.** The two agents are:

- Nova, the research agent: An **adaptable, steerable research agent** implemented as an **MCP (Model Context Protocol)** server + client (using **FastMCP**). The research agent can branch, pause, and take human feedback; it ingests web pages, GitHub repos, YouTube transcripts, and local files; it runs iterative research rounds with **Perplexity**; and it enforces a **critical failure policy** plus **quality filters** before producing a consolidated `research.md`.
- Brown, the writer hybrid system (a workflow that has agents as steps): A **reliable writing workflow** implemented as a **LangGraph workflow**. It emphasizes determinism, statefulness, and auditability (outline → draft → reflect → iterate → global refine → finalize), materializes writing profiles, and records reflection scores to reduce the “AI slop” effect (e.g., overuse of words like *delve*).

You will see the **high‑level architecture and reasoning** behind each agent, a **build‑vs‑buy** analysis (why we didn’t rely solely on off‑the‑shelf “deep research” tools), the **inputs/outputs/artifacts** you’ll create, and **where human‑in‑the‑loop** fits. We keep framework internals light here (they come next in Lesson 13); this lesson is about **scope, design decisions, and tradeoffs**.

### Why We Think It's Valuable

Moving from theory to a production‑minded build requires **clear mental models** for when to prefer **steerable agents** versus **reliable workflows**, how to integrate **external tools** without ballooning token costs, and how to design for **pause/resume, monitoring, and auditability**. This lesson aligns expectations on **what we’ll build** and **why**, so later lessons (framework comparison, system design, implementation, evaluation, observability, deployment) plug into a shared blueprint. It also shows how to **future‑proof** decisions in a fast‑moving ecosystem by focusing on **concepts and problem‑fit**, not on any single framework “winner.”

### Expected Length of the Lesson

**2500-3000 words** (without the titles and references), where we assume that 200–250 words ≈ 1 minute of reading time.

### Theory / Practice Ratio

**100% theory.**

---

## Anchoring the Lesson in the Course

### Details About the Course

This piece is part of a broader course on AI agents and LLM workflows. The course consists of **4 parts**, each with multiple lessons.

Thus, it’s essential to always anchor this piece into the broader course, understanding where the reader is in their journey. You will be careful to consider the following:

- The points of view
- To not reintroduce concepts already taught in the previous lessons
- To be careful when talking about concepts introduced only in future lessons
- To always reference previous and future lessons when discussing topics outside the piece’s scope

### Lesson Scope

This is **Lesson 12 of the course,** which is the first lesson of part 2 (out of 4 parts) of the course on **AI Agents & LLM Workflows**. Part 2 of the course will have 22 lessons. This lesson #12, is a **no‑notebook, high‑level, scope‑and‑design** lesson that focuses on defining the central project and the rationale behind splitting it into a research agent (MCP + FastMCP) and a writing agent (LangGraph).

### Point of View

The course is created by a team writing for a single reader, also known as the student. Thus, for voice consistency across the course, we will always use **“we,” “our,” and “us”** to refer to the team who creates the course, and **“you” or “your”** to address the reader. Avoid singular first person and don’t use **“we”** to refer to the student.

**Example of correct point of view:**

- Instead of “Before we can choose between workflows and agents, we need a clear understanding of what they are,” write “To choose between workflows and agents, you need a clear understanding of what they are.”

### Who Is the Intended Audience

Aspiring AI engineers who are designing and building their **first production‑leaning agent system** and are ready to move from Part 1 on theory to a concrete, multi‑lesson, end-to-end project.

### Concepts Introduced in Previous Lessons

In previous lessons of the course, we introduced the following concepts:

**Part 1:**

- **Lesson 1 - AI Engineering & Agent Landscape**: Understanding the role, the stack, and why agents matter now
- **Lesson 2 - Workflows vs. Agents**: Grasping the crucial difference between predefined logic and LLM-driven autonomy
- **Lesson 3 - Context Engineering**: The art of managing information flow to LLMs
- **Lesson 4 - Structured Outputs**: Ensuring reliable data extraction from LLM responses
- **Lesson 5 - Basic Workflow Ingredients**: Implementing chaining, routing, parallel and the orchestrator-worker patterns
- **Lesson 6 - Agent Tools & Function Calling**: Giving your LLM the ability to take action
- **Lesson 7 - Planning & Reasoning**: Understanding patterns like ReAct (Reason + Act)
- **Lesson 8 - Implementing ReAct**: Building a reasoning agent from scratch
- **Lesson 9 - RAG Deep Dive**: Advanced retrieval techniques for knowledge-augmented agents
- **Lesson 10 - Agent Memory & Knowledge**: Short-term vs. long-term memory (procedural, episodic, semantic)
- **Lesson 11 - Multimodal Processing**: Working with documents, images, and complex data

At this point, you know the **foundational patterns** and the **tradeoffs** between workflows and agents; we will **build on** those.

### Concepts That Will Be Introduced in Future Lessons

In future lessons of the course, we will introduce the following concepts:

**Part 2:**

- **Lesson 13 — Frameworks Overview & Comparison** (LangGraph, OpenAI Agents SDK, CrewAI, AutoGen, PydanticAI) and why we emphasize **LangGraph** for workflows and **FastMCP + MCP** when we expect framework churn.
- **Lesson 14 — System‑Design Decision Framework** (models, costs/latency, human‑in‑the‑loop strategy). It’s a more “in-depth” system design overview of this lesson.
- **Lessons 15–18 — Building the Research Agent** with **FastMCP** + **MCP** (server/client, tools, prompt; ingestion; Perplexity research loops; filtering; final `research.md`)
- **Lessons 19–22 — Building the Writing Agent** with **LangGraph** (hybrid systems between workflows and agents, controlling writing profiles via context engineering, reflection/self‑critique, human-in-the-loop editing via MCP)

**Part 3:**

- With the agent system built, this section focuses on the engineering practices required for production. You will learn to design and implement robust evaluation frameworks to measure and guarantee agent reliability, moving far beyond simple demos. We will cover AI observability, using specialized tools to trace, debug, and understand complex agent behaviors. Finally, you’ll explore optimization techniques for cost and performance and learn the fundamentals of deploying your agent system, ensuring it is scalable and ready for real-world use.

**Part 4:**

- In this final part of the course, you will build and submit your own advanced LLM agent, applying what you've learned throughout the previous sections. We provide a complete project template repository, enabling you to either extend our agent pipeline or build your own novel solution. Your project will be reviewed to ensure functionality, relevance, and adherence to course guidelines for the awarding of your course certification.

As lesson on the core foundation of AI engineering, we will have to make references to new terms that haven't been introduced yet. We will discuss them in a highly intuitive manner, being careful not to confuse the reader with too many terms that haven't been introduced yet in the course.

### Anchoring the Reader in the Educational Journey

Within the course, we are teaching the reader multiple topics and concepts. Thus, understanding where the reader is in their educational journey is critical for this piece. You have to use only previously introduced concepts, while being reluctant about using concepts that haven't been introduced yet.

When discussing the **concepts introduced in previous lessons** listed in the `Concepts Introduced in Previous Lessons` section, avoid reintroducing them to the reader. Especially don't reintroduce the acronyms. Use them as if the reader already knows what they are.

Avoid using all the **concepts that haven't been introduced in previous lessons** listed in the `Concepts That Will Be Introduced in Future Lessons` subsection. Whenever another concept requires references to these banned concepts, instead of directly using them, use intuitive analogies or explanations that are more general and easier to understand, as you would explain them to a 7-year-old. 

For example:

- If the "tools" concept wasn't introduced yet and you have to talk about agents, refer to them as "actions".
- If the “LangGraph” framework wasn’t introduced yet, you can refer to it as “Agent Frameworks”
- If the "routing" concept wasn't introduced yet and you have to talk about it, refer to it as "guiding the workflow between multiple decisions". You can use the concepts that haven't been introduced in previous lessons, listed in the `Concepts That Will Be Introduced in Future Lessons` subsection only if we explicitly specify them. Still, even in that case, as the reader doesn't know how that concept works, you are only allowed to use the term, while keeping the explanation extremely high-level and intuitive, as if you were explaining it to a 7-year-old. Whenever you use a concept from the `Concepts That Will Be Introduced in Future Lessons` subsection, explicitly specify in what lesson it will be explained in more detail, leveraging the particulars from the subsection. If not explicitly specified in the subsection, simply state that we will cover it in future lessons without providing a concrete lesson number.

In all use cases avoid using acronyms that aren't explicitly stated in the guidelines. Rather use other more accessible synonyms or descriptions that are easier to understand by non-experts.

---

## Narrative Flow of the Lesson

Follow the next narrative flow when writing the end‑to‑end lesson:

We *build rationale* arc that maps needs → choices:

- Lead with requirements: steerability, *automatic context ingestion*, determinism, checkpoints, and an *app layer* for media.
- For each requirement, state what we *USE* (e.g., robust web→Markdown scrapers, Perplexity) vs. what we *BUILD* (orchestrators, policies, contracts).
- Explain why one framework couldn’t do both jobs well, motivating the split (MCP for interactive research, LangGraph for reliable multi‑stage writing).
- Show the resulting architecture and handoff artifacts (`research.md`, stage files, SEO, scores).
- Close by previewing how future lessons deepen each component.
- Connect our solution to the bigger field of AI Engineering. Add course next steps.

---

## Lesson Outline

1. **Section 1: Introduction**
2. **Section 2: Why Simpler Paths Fall Short — Build vs. Buy & Early Missteps**
3. Section 3: Why LLMs Still Struggle at Writing (and How to Fix It Manually)
4. **Section 4: Two Agents, Two Philosophies (MCP vs. LangGraph)**
5. **Section 5: Agents Design**
6. **Section 6: Conclusion**

---

## Section 1 — Introduction

- **Quick reference to what we've learned in previous lessons:** Take the core ideas of what we've learned in previous lessons from the `Concepts Introduced in Previous Lessons` subsection of the `Anchoring the Lesson in the Course` section.
- **Transition to what we'll learn in this lesson:** After presenting what we learned in the past, make a transition to what we will learn in this lesson. Take the core ideas of the lesson from the `What We Are Planning to Share` subsection and highlight the importance and existence of the lesson from the `Why We Think It's Valuable` subsection of the `Global Context of the Lesson` section.
- **A short “engineering‑scar” story (why this lesson exists):**
    - When we first prototyped this project, we tried to do **research and writing** under one architecture. Tuning for **exploration** made drafting brittle; tuning for **deterministic drafting** killed exploration. After a few long iterations, we split the system: a **steerable research agent** and a **reliable writing workflow**.
- **Section length:** ~400 words

---

## Section 2: The Build vs. Buy Trade-off: Why Not Just Use Off-the-Shelf Tools?

- Having framed the dual nature of our system, we now examine what use case we were building the system for and why off‑the‑shelf options didn’t work for us.
- Anyone setting out to build a custom AI system has to first decide whether they really need to build this or whether off-the-shelf tools are good enough.
- In our case, we wanted to create an agent that can turn ideas into in-depth research and structured technical writing. We wanted a system that was easily tuneable to our preferred writing style, which we could iterate to get better results. We also wanted to make the system open source and adaptable for many other writing use cases after being tuned to follow certain templates.
- “Deep research” tools (which are potential solutions to our task of “taking some article guidelines and turning them into a publishable article”) are **fast, polished, convenient**—but we needed **steerability**: the ability to **change topics mid‑stream**, **approve next search queries**, and weave in **local files**, **GitHub repositories**, and **YouTube transcripts**. We also needed a **deterministic writing process** with **style gates**, **reflection scores**, and **checkpoints**—capabilities we couldn’t reliably get from turnkey tools. We decided to **buy** commodity capabilities (e.g., robust web→markdown scraping, Perplexity’s research API) but **build** the orchestration, policies, and artifacts that guarantee quality.
- Crucially, we’re *not* rebuilding excellent components (scrapers, transcription); we *USE* what’s already great, and we *BUILD* only where flexibility, steerability, or guarantees are missing.
- **Mandatory Artifact (Comparison Table)**:

| Capability | Off-the-Shelf Tool Limitation | Our Custom System’s Advantage |
| --- | --- | --- |
| Custom Structure | We couldn’t programmatically require specific sections or layouts or extra features such as generating Mermaid diagrams. | Enforceable contracts with required sections, diagrams, or tables baked into the workflow. (While still using “commodity capabilities” such as web scrapers. |
| Stylistic Control | There was no option to ban AI slop clichés or enforce our own personal writing profile automatically. | Automated “style judge” nodes enforce tone, banned terms, and revision loops. |
| Planning Heuristics | We needed to prioritize academic sources over blogs, but the tools didn’t let us inject that rule. | Custom heuristics and evaluation criteria steer research toward higher-quality sources. |
| Integration Contract | Piping the output into our CI/CD pipeline was brittle; there was no versioned JSON contract. | Structured JSON schemas govern handoffs, enabling replay, testing, and long-term maintainability. |
| Context Ingestion at Scale | Manual source upload/linking is common (e.g., NotebookLM), which slows iteration and increases human toil. | Automated bulk ingestion of URLs, repos, and files with near‑zero manual steps; the research agent discovers and normalizes context automatically. |
- **Instruction**: After the table, explicitly **explain that each limitation isn’t hypothetical; it actually blocked us**. For instance, not being able to enforce house style meant that we have to waste cycles fixing cliché phrases like “in today’s fast-paced world.” Similarly, manual context ingestion meant repeatedly curating links and uploads; automation here removed a major bottleneck.
- **Section length**: 500-600 words

---

## Section 3: Why LLMs Still Struggle at Writing (and How to Fix It Manually)

More broadly - one of the issues our system is aiming to improve is LLM’s widely known weakness at writing. 

Within ChatGPT writing was an early win for LLMs for tasks such as drafts, emails, rewrites. Yet it’s also where flaws are most visible: generic voice, verbosity, hedging, formulaic openings/closings, overuse of lists where narrative is needed, and that unmistakable “**AI sound.**” Think lines you’ve probably seen a hundred times: “In today’s fast‑paced world…,” “Let’s *delve* into the *realm* of…,” “It’s not just X—it’s Y,” or five bullet points that restate the intro without adding substance.

After the public LLM wave, certain stylistic words (e.g., *delve*, *realm*, *intricate*, *underscore*) exploded across professional writing. In venues like PubMed, analyses show large 2024–2025 spikes in these words consistent with increased LLM‑assisted drafting.

**A plausible causal path:** Preference‑tuning (human feedback) rewards outputs raters *like*. If a significant share of raters favor formal, British‑tinged phrasing, plausibly the case given rater workforces in countries such as Nigeria and Kenya, then models learn to overvalue that register. When millions then copy AI‑authored phrasing, the style feeds back into the public web and scientific corpora, amplifying the loop. The result: **AI slop** language that feels synthetic because it optimizes for average likability, not human voice.

**Why we care:** These tells erode credibility and make writing sound generic. We treat them as *bugs to catch*. Our workflow will: (1) **ban** specific markers (e.g., *delve, leverage, harness, navigate, tapestry, endeavour, groundbreaking, pivotal, it’s not just X it’s Y*), (2) **enforce precision** over puffery, and (3) **prefer narrative paragraphs** unless lists are warranted by content. Beyond style, we also enforce **mechanics and structure:**

- Mechanics: active voice, domain‑appropriate abbreviations, sentence‑length and paragraph‑length targets, consistent tense.
- Structure & formatting: correct code block fencing, image/figure placement, and diagram rendering expectations (e.g., Mermaid).
- Media in/out: use media as *context* during generation (figures, transcripts) and emit media (diagrams) as part of the output.
- App layer: the output isn’t just text; we rely on an *app layer* to render media, package artifacts, and integrate with your tools (site/CMS, repo, CI).

**Takeaway:** Models are pattern machines. If you don’t actively constrain style, you get statistically “safe” language, not your voice.

**Manual practices that work today (and that we’ll encode later):**

- **Don’t outsource your thinking.** Bring the outline, angle, and your expertise first.
- **Front‑load context.** Attach source docs and data; forbid external speculation.
- **Be prescriptive.** Specify audience, tone, length, structure, banned words, allowed jargon, and formatting rules.
- **Combat verbosity.** Enforce *word economy* and “high insight per sentence.”
- **Anti‑slop rules.** Maintain a living **ban list** of overused AI tells (see Section 5).
- **Iterate deliberately.** Ask for *revisions*, not just “edits” (models often summarize instead of improving).
- **Cross‑model critique.** Have one model *critique* another’s draft for unsupported claims and slop.
- **Relentless fact‑checking.** Demand citations, spot‑check key claims, and verify dates.

These tactics lift quality, but they’re repetitive and painful to implement manually. That’s why we’ll automate them.

**Section length:** 500-600 words

## Section 4 — Two Agents, Two Philosophies (MCP vs. LangGraph)

- Even after clearly mapping out the problems you are trying to solve -  you rarely design the perfect system in day one. We initially tried to pick our agent framework to build our entire system but after starting to build we learnt that “one‑framework‑for‑everything” approaches didn’t work for us.
- **Early missteps we learned from:**
    - We initially tried to make the **research agent as LangGraph workflow**. Under testing, research proved highly **interactive**; it needed flexibility, to **pause/resume**, accept **user feedback**, and **run tools in parallel**. Good and complex workflows can give us those features, but we realized an agent, with a good system prompt, could provide the same features and the benefits with no need for “hard-coded flows”.  So, we shifted to an **MCP server+client** with a **single MCP prompt** that teaches the “agentic recipe” for using ~10 tools. The workflow was first defined in code (using the LangGraph framework) and then it was defined in natural language. For the writer, we converged on *MCP + LangGraph*: MCP exposes a clean interface for the UI/app, while LangGraph handles multi‑stage, checkpointed editing.
    - We wanted to teach only **one framework** (LangGraph), but the **problem shapes diverged**. Research favored MCP’s **tool‑exposed server** model and client‑agnostic interoperability (works with Cursor, Claude Code, etc.). Writing favored **LangGraph’s durable state and observability**. **Hence the split is not “versus” but “fit”: MCP for interactive research; MCP + LangGraph for reliable writing.**
- **Takeaway:** When the **process is exploratory**, choose **steerable agents**; when the **process is well‑specified and long‑running**, choose **stateful workflows** with checkpoints and monitoring.
- Given the limitations of a single approach, we reorganized around two **problem‑fit** architectures.
- **Research agent — steerability first:**
    - Very short MCP primer (details in Lesson 13): MCP is a simple, open protocol that lets an agent call tools and access resources via a server it can discover at runtime. It standardizes how the agent sees tools and prompts, so different UIs/hosts can interoperate.
    - MCP server exposes tools, resources, and an MCP prompt that encodes the agentic workflow (a “recipe” for research).
    - MCP client (our orchestrating shell) discovers server capabilities, starts at Step 1, and can pause for human feedback at predefined gates (e.g., after generating next queries).
    - Methods we use: tool parallelism (scrape/ingest in parallel), iterative query→read→select loops (3 rounds by default), source quality filtering, and selective full‑page scraping for the top sources. We push heavy work into tools to keep LLM contexts small and retries cheap.
- **Writing agent — reliability first:**
    - **LangGraph** manages a **multi‑stage state machine**: outline → one‑shot draft → section‑level reflection loops (to ≥90% checks) → global refine → SEO + titles → final artifacts. MCP cleanly connects the UI/app layer to these stages.
    - **Methods we use:** state persistence/checkpointing, explicit **evaluation rules**, and **reflection/self‑critique** to target recurring style defects (e.g., hedging, clichés, “AI slop,” including overuse of *delve*). We **materialize** writing profiles and produce **stage artifacts** for auditability.
- **Why this is better:** This split lets us **specialize**: the research agent maximizes **adaptation and breadth**; the writing agent maximizes **consistency, traceability, and quality**. Both benefit from **human‑in‑the‑loop** at the right points without entangling concerns. And the MCP bridge gives us a clean protocol surface to integrate a UI and app‑layer formatting.
- **Section length:** ~500–600 words

---

## Section 5 — Agents Design

- With the high‑level theory, let’s anchor it in concrete inputs/outputs and a visual flow.
- **Include a system diagram (Mermaid)** showing: `Article Guideline.md` → **Research Agent (MCP)** → `research.md`; then `Article Guideline.md` + `research.md` + Writing Profiles → **Writing Agent** (LangGraph via MCP) → Final Article + SEO + scores → App Layer (renders diagrams/media, packages artifacts).
- **Research agent:**
    - Inputs:
        - File `article_guideline.md` with the article guidelines (topic, audience, expected length, outline, candidate sources, style notes)
    - Outputs:
        - A final **`research.md`** containing curated notes, citations, and collapsible sections.
    - **Agentic recipe via MCP prompt:** The MCP prompt **defines the workflow** and the **contracts** for each step. It also explains how to start, where to pause for human feedback, and when to **halt** on critical failure. Highlight that **any MCP client** can discover this MCP prompt and follow it.
    - **Full workflow (as specified in the prompt):**
        1. **Setup**
            - Explain numbered steps to the user; request the **research directory**; ask whether to run from a specific step or add human‑feedback gates.
            - **Extract guideline references** (URLs grouped as GitHub / YouTube / Other; plus relative local file paths) into a machine‑readable file.
            - Auto‑ingest context: discover and queue dozens of inputs (URLs, repos, files) for ingestion with near‑zero manual effort.
        2. **Parallel processing of extracted resources**
            - **Local files:** copy into an internal folder; normalize notebooks for LLM readability.
            - **Other URLs:** robust scrape + clean to Markdown.
            - **GitHub URLs:** produce repository digests via a code‑aware tool.
            - **YouTube URLs:** transcribe videos to Markdown (note latency scaling with video length).
        3. **3‑round research loop**
            - **Generate next queries** based on current coverage and gaps; write queries + justifications to disk.
            - **Run Perplexity** on those queries; append structured results (including URLs and extracted content) to disk.
        4. **Quality filtering**
            - **Select research sources to keep** based on trustworthiness, authority, and relevance; write IDs and a filtered results file.
        5. **Selective full scrapes**
            - **Choose up to 5 diverse, authoritative URLs** to fully scrape; run full content scrapes and save cleaned Markdown.
        6. **Create the final research file**
            - Consolidate all cleaned materials (guideline URLs, GitHub digests, YouTube transcripts, Perplexity filtered results, and full scrapes) into a single **`research.md`** with collapsible sections for navigation.
        - **Critical Failure Policy:** If any tool reports a **complete failure** (e.g., 0/N URLs succeeded), the agent must **stop immediately**, quote the failure, and **ask the user** how to proceed. This turns “unknown unknowns” into safe, visible interruptions rather than silent data holes.
    - **Include a workflow diagram (Mermaid)** summarizing the research agent: extract guideline references → (parallel) process local files / scrape other URLs / GitHub digests / YouTube transcripts → 3 research rounds (generate queries → run Perplexity) → source quality filtering → select top URLs for **full** scrape → final `research.md`.
- **Writing agent:**
    - Architecture note: Expose the writer through MCP (protocol/UI boundary) and implement multi‑stage logic in LangGraph.
    - Inputs:
        - File `article_guideline.md` with the article guidelines (topic, audience, expected length, outline, candidate sources, style notes)
        - File `research.md` (the output of the research agent) containing curated notes, citations, and collapsible sections.
        - Files with writing profiles.
    - Outputs:
        - A **final article** (Markdown) in the article.md file, with title, intro, body, conclusion (and references when applicable).
        - **SEO metadata**, **reflection score tables**, and **stage artifacts** (e.g., `article.metadata.json` ,`article_stage_1.md`). Include rendered diagrams (e.g., Mermaid) and ensure code blocks/images are formatted for the *app layer* to package/publish.
    - **Include a workflow diagram (Mermaid)** summarizing the writer agent. Here is a concise walk-through of the main nodes.
        - “Stage 0 — Context Gathering”: (1) Parse all inputs (guideline, research, writing profile, examples, evaluation rules, writer profile);
        - “Stage 1 — Outline and One-shot Draft”: (2) Plan introduction, sections, conclusion (respect pre-defined sections in guideline), (3) Draft an end-to-end article using outline, research, style, profile, (4) Evaluate draft against evaluation rules; apply targeted one-shot edits, (5) Parse Markdown into introduction, sections, conclusion, references; render `article_stage_1.md`.
        - “Stage 2 — Iterative Section Editing Loop”: (6) Score each section against evaluation rules, (7) If less than 90% of the section checks are ok, then apply targeted changes guided by reflection results. Iterate this step until at least 90% of the section checks are ok, or until the maximum number of iterations is reached.
        - “Stage 3 — Global Reflection and Finalization”: (8) Evaluate entire article and log Stage 3 reflection scores, (9) Apply global edits guided by reflection results, (10) Produce final title/subtitle, (11) Produce SEO title/description (requires title), (12) Save final article, metadata, writing profile, and reflection scores. Expose final artifacts via MCP to the app layer (for rendering/publish).
- **Provide a mini example of usage of the two agents:**
    - Guideline says: “Explain MCP prompts and why we used them.”
    - Research agent extracts MCP docs link, finds additional authoritative URLs via Perplexity, filters for trustworthiness, fully scrapes 3 chosen pages, and compiles `research.md`. (auto‑ingesting sources with minimal manual setup).
    - Writing agent renders a short intro, sections, and conclusion, passes style checks (ban clichés; avoid *delve*), and emits SEO metadata + reflection scores plus a diagram block, which the app layer renders.
- **Section length:** ~700-800 words

## Section 6 — Conclusion

- **Bigger picture:** We designed for **steerability where exploration and flexibility dominates** (research) and **reliability where quality guarantees matter** (writing). The split clarifies responsibility boundaries, reduces unintended coupling, and makes **evaluation/observability** (Part 3) natural: research and writing produce **inspectable artifacts** that invite metrics, tracing, and audits. Automatic context ingestion and an app layer for media complete the loop from inputs → human‑in‑the‑loop agenting → polished outputs.
- To transition from this lesson to the next, specify what we will learn in future lessons. First mention what we will learn in next lesson, which is Lesson 13. Next leverage the concepts listed in subsection `Concepts That Will Be Introduced in Future Lessons` to make slight references to other topics we will learn during this course. To stay focused, specify only the ones that are present in this current lesson. Make explicit that Lesson 13 deep‑dives MCP vs. LangGraph tradeoffs, and Lessons 15–22 walk through implementing the MCP + FastMCP research agent and the MCP+LangGraph writer.
- **Section length:** ~300 words

---

## Article Code

**No notebook or code** for this lesson.

---

## Golden Sources
- [Additional Notes]("/Users/omar/Documents/ai_repos/course-ai-agents/lessons/12_central_project/l12_notes.md")
- [Introducing Deep Research — OpenAI](https://openai.com/index/introducing-deep-research/)
- [Gemini Deep Research — Overview](https://gemini.google/overview/deep-research/)
- [Introducing Perplexity Deep Research — Perplexity Blog](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research)
- [Why Does ChatGPT “Delve” So Much? — arXiv](https://arxiv.org/abs/2412.11385)
- [Delving into PubMed Records — medRxiv](https://www.medrxiv.org/content/10.1101/2024.05.14.24307373v2.full)

---

## Other Sources

- [AI Chatbots Have Thoroughly Infiltrated Scientific Publishing — Scientific American](https://www.scientificamerican.com/article/chatbots-have-thoroughly-infiltrated-scientific-publishing/)
- [It’s happening: People are starting to talk like ChatGPT — Washington Post](https://www.washingtonpost.com/opinions/2025/08/20/chatgpt-claude-chatbots-language/)

