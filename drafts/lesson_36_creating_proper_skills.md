# Lesson 36: Creating Proper Agent Skills

In the previous lesson, we compared the three channels for giving an agent capabilities: MCP for standardized tool access, skills for packaged know-how, and the CLI for raw execution power. We saw that skills are organized folders of instructions, scripts, and resources that agents discover and load dynamically [3], and we introduced progressive disclosure as the mechanism that makes them cheap to carry around. In this lesson, you will author a proper skill from scratch.

This is a hands-on lesson, but a deliberately lightweight one. You need no API keys, no running servers, and no sample data for the core exercise, just a text editor and the course repository. By the end of this lesson, you will have a real, working skill on your machine: `/nova-research`, an entry point that packages the know-how for driving our research agent, Nova. You will also have a checklist you can apply to any skill you write, including the one you will ship alongside your own MCP server in the Part 4 capstone.

You will learn how to:

- Structure a skill directory and write valid SKILL.md frontmatter
- Design for progressive disclosure so your skill costs almost nothing until it is needed
- Write descriptions that reliably trigger the skill at the right moment
- Choose the right degree of freedom: prose heuristics, templates, or exact scripts
- Build, test, and iterate on the `/nova-research` skill in Claude Code
- Avoid the most common skill-authoring failure modes

## The Anatomy of a Skill

A skill is, at minimum, a directory containing a single file: SKILL.md. That file has two parts: YAML frontmatter (the metadata) and a markdown body (the instructions). Around it, the open Agent Skills standard defines three optional subdirectories: `scripts/` for executable code, `references/` for documentation loaded on demand, and `assets/` for templates, images, and data files [1].

```
my-skill/
├── SKILL.md          # required: frontmatter + instructions
├── scripts/          # optional: executable code
├── references/       # optional: docs loaded on demand
└── assets/           # optional: templates, images, data
```

The frontmatter has two required fields and a handful of optional ones. The constraints are strict, and validators enforce them, so it is worth internalizing the spec exactly [1].

*Table 1: SKILL.md frontmatter fields per the Agent Skills specification*

| Field | Required | Constraints |
| --- | --- | --- |
| `name` | Yes | 1-64 chars; lowercase letters, numbers, hyphens only; no leading, trailing, or consecutive hyphens; must match the parent directory name |
| `description` | Yes | 1-1024 chars; describes both what the skill does and when to use it, with specific keywords |
| `license` | No | License identifier for the skill |
| `compatibility` | No | Max 500 chars; only for specific environment requirements |
| `metadata` | No | Arbitrary string map |
| `allowed-tools` | No | Experimental; space-separated pre-approved tools, e.g. `Bash(git:*) Read` |

For naming, Anthropic's skill-authoring best-practices guide [2] recommends the gerund form (`processing-pdfs`, `analyzing-spreadsheets`); noun phrases like `pdf-processing` are acceptable. Avoid vague names like `helper`, `utils`, or `documents`.

> [!NOTE]
> **Claude Code is more lenient than the spec.** It can infer a missing `name` from the directory, and the slash-command name comes from the directory in any case. This is why you will encounter skills "in the wild" that omit fields the spec marks required. Write to the spec anyway—an explicit, valid `name` plus a strong `description`—so your skill stays portable to other hosts and does not silently break if the folder is renamed [4].

You can check your work mechanically. The reference library from the agentskills project ships a validator:

```bash
skills-ref validate ./my-skill
```

It verifies the frontmatter constraints and directory conventions, catching problems like a `name` that does not match the directory [1]. We recommend running it on every skill before you share it. For worked examples in the same format, Anthropic maintains a public library of real skills at github.com/anthropics/skills, which is a good place to see how experienced authors structure a body and its supporting files [5].

## Designing for Progressive Disclosure

In Lesson 35, we described progressive disclosure as a runtime feature. When you author a skill, it becomes something more: the design principle you write *for*. The standard defines a three-level context budget [1], [3]:

1. **Metadata, always loaded.** The name and description of every installed skill are pre-loaded into the system prompt at startup, at a cost of roughly 100 tokens per skill.
2. **Body, loaded on activation.** When the agent judges the skill relevant, the full SKILL.md body loads. The recommendation is to keep it under 5,000 tokens and under 500 lines.
3. **Resources, loaded as needed.** Linked files in `references/` and elsewhere load only when the agent actually needs them.

Anthropic's engineering team compares this to a well-organized manual that starts with a table of contents, then specific chapters, and finally a detailed appendix, so the agent loads information only as needed, which makes the context a skill can bundle effectively unbounded [3]. The same team's authoring guidance adds the discipline that follows from it: "The context window is a public good." Once loaded, every token of your SKILL.md competes with the conversation history the agent needs to do its job [2].

There is a second, less obvious cost. In Claude Code, once a skill's body loads, it stays in context for the rest of the session, so every line is a recurring token cost [4]. A bloated skill does not just slow down the turn where it activates; it taxes every turn after.

The practical rule: assume the model is already very smart, and only add context it does not have. Challenge each paragraph of your skill with one question: does this justify its token cost? [2]

*Image 1: Diagram of the three progressive disclosure levels — always-loaded metadata (~100 tokens), on-activation SKILL.md body (<5k tokens), and on-demand reference files — shown as widening layers of a funnel.*

## Writing Descriptions That Trigger

The description is the make-or-break field. At runtime, it is the primary activation signal: the model chooses among potentially 100+ installed skills using little more than their names and descriptions (plus, in Claude Code, the optional `when_to_use` field) [2]. A skill with a perfect body and a vague description simply never fires.

Three rules make descriptions work:

**Write in third person, always.** The description is injected into the system prompt, so first person reads incoherently there [2].

**State both what and when.** The spec is explicit: describe what the skill does *and* when to use it, with specific keywords [1]. Compare the spec's own example pair:

- Good: "Extracts text and tables from PDF files, fills PDF forms, and merges multiple PDFs. Use when working with PDF documents or when the user mentions PDFs, forms, or document extraction." [1]
- Poor: "Helps with PDFs." [1]

**Include concrete trigger phrases.** The git-commit example from the best-practices doc shows the pattern: "Generate descriptive commit messages by analyzing git diffs. Use when the user asks for help writing commit messages or reviewing staged changes." [2] For a Nova-flavored research skill, the same pattern looks like: "Runs the full Nova research workflow to produce a research.md for an article. Use when the user wants to research a topic, gather and curate sources, or prepare research for writing." We will refine this into our actual description in the walkthrough below.

> [!NOTE]
> **Claude Code extension: `when_to_use`.** Claude Code follows the open standard but extends it with extra frontmatter fields. One is `when_to_use`, which lets you add trigger context beyond the description. Be aware of the budget: the description plus `when_to_use` are truncated at 1,536 characters in the skill listing, so front-load your most distinctive trigger phrases [4].

## Degrees of Freedom: Prose, Templates, or Scripts

Not every task deserves the same level of prescription. The best-practices guide frames this as *degrees of freedom*: match the specificity of your instructions to the fragility of the task [2].

- **High freedom: prose heuristics.** Use when many valid paths exist, such as code review. Describe principles and let the model choose its route.
- **Medium freedom: templates and pseudocode.** Use when there is a preferred shape with acceptable variation. Provide a template with parameters the model fills in.
- **Low freedom: "run exactly this script."** Use for fragile, sequence-critical operations like a database migration, where deviation causes damage. The agent executes your script and does not improvise.

The analogy in that best-practices guide is useful: high freedom is an open field where any path works; low freedom is a narrow bridge with cliffs on both sides, where you install a guardrail rather than offering advice [2].

*Image 2: A spectrum diagram from high freedom (prose heuristics, open field) through medium (templates, marked trail) to low freedom (exact scripts, narrow bridge with guardrails), with example tasks under each.*

When should you bundle an executable script instead of instructions? Whenever deterministic code beats token generation: "sorting a list via token generation is far more expensive than simply running a sorting algorithm" [3]. Pre-made scripts are more reliable, cheaper, faster, and consistent, and crucially, when executed directly, a script's code need not enter the context; typically only its *output* consumes tokens [2].

Two rules keep bundled scripts honest [2]:

- **Solve, don't punt.** Handle errors inside the script instead of failing back to the model with a stack trace it has to interpret.
- **No voodoo constants.** Justify every configured value; an unexplained `timeout=47` will get "corrected" by a future model or human.

And always make execution intent explicit: "Run analyze_form.py" means execute; "See analyze_form.py for the algorithm" means read. Ambiguity here causes the agent to read a script it should run, paying the token cost you built the script to avoid [2], [3].

## Structuring Skills That Scale

Every skill starts small, and successful ones grow. The rule remains: keep the SKILL.md body under 500 lines. When you approach the limit, split content into reference files instead of letting the body bloat. The same best-practices guide offers three splitting patterns [2]:

1. **High-level guide plus references.** SKILL.md stays a concise guide that links out to FORMS.md, REFERENCE.md, EXAMPLES.md.
2. **Domain-split reference directories.** Separate files per domain (finance.md, sales.md, ...) so only the relevant context loads for a given request.
3. **Conditional details.** Inline the common path; branch to files for rarer cases: "For tracked changes: see REDLINING.md."

Whichever pattern you use, keep file references one level deep from SKILL.md. Deeply nested reference chains cause partial reads: the agent may `head -100` a file and miss the content it needed. For the same reason, give any reference file longer than 100 lines its own table of contents [2].

Three more content rules pay off at scale [2]:

- **One consistent term.** Do not alternate between "field," "box," and "element" for the same thing.
- **No time-sensitive information.** Skills live in repos for months; keep deprecated approaches, if you must document them, in a clearly labeled "old patterns" section.
- **One default, not a menu.** Offer a single recommended path plus an escape hatch, not five options the agent must adjudicate every time.

For workflows, number the steps, and for complex flows, provide a copyable checklist. Where the output format matters, include templates and input/output examples [2].

## Hands-On: Building the /nova-research Skill

Now let's build one for real. Throughout Part 2, you used Nova by starting our custom MCP client and typing `/prompt/full_research_instructions_prompt` to load the server-hosted research workflow. When Nova's server is connected to Claude Code instead, that know-how, which prompt to load, which tools to expect, what to do when it finishes, lives only in your head. We will package it as a project skill so anyone who clones the repo gets it.

> [!NOTE]
> **When is a skill warranted?** The Claude Code docs give a crisp heuristic: create one when you keep pasting the same instructions, checklist, or multi-step procedure into chat, or when a section of CLAUDE.md has grown into a procedure rather than a fact. Unlike CLAUDE.md content, a skill's body loads only when it is used [4].

**Step 1: Create the directory.** Claude Code looks for project skills in `.claude/skills/<name>/SKILL.md`, checked into the repo and shared with your team via git (personal skills live in `~/.claude/skills/` and apply across all your projects) [4]. From the course repository root:

```bash
mkdir -p .claude/skills/nova-research
touch .claude/skills/nova-research/SKILL.md
```

Note the name: `nova-research` is 1-64 characters, lowercase with a single hyphen, and will match the directory, all spec requirements [1].

**Step 2: Write the frontmatter.** The description carries what the skill does, when to use it, and concrete trigger phrases, in third person:

```yaml
---
name: nova-research
description: "Runs the full Nova research workflow to produce a research.md
document for an article. Use when the user wants to research a topic for an
article, gather and curate sources, or prepare research before writing.
Triggers on: 'research this topic', 'run the research workflow', 'prepare
research for this article', 'gather sources for', or any request to turn an
article guideline into a research.md."
---
```

**Step 3: Write the body.** Here is the key design decision. The complete research recipe already exists on Nova's server: it is the workflow text behind the `full_research_instructions_prompt` you loaded in Lessons 16-19, exposed since Lesson 35 as an agent-callable tool, `get_research_instructions`, generated from the same Python function. We will not duplicate it. The skill is the *entry point*; the server-hosted workflow is the *recipe*. The body tells the agent to fetch the workflow and follow it with Nova's tools.

The course repository ships this tool in `mcp_server/src/routers/tools.py`, registered like every other tool (it brings the Part 2 server to 12 tools). If your clone predates it, here is the exact change — one import and one registration. Note the `await`, because the underlying prompt function is async, and the `opik` decorator plus `opik_context` call, which match the observability wiring every Nova tool uses:

```python
from ..prompts.research_instructions_prompt import (
    full_research_instructions_prompt as _get_research_instructions,
)

@mcp.tool()
@opik.track(type="tool")
async def get_research_instructions() -> str:
    """Return the full research workflow instructions."""
    opik_context.update_thread_id()
    return await _get_research_instructions()
```

One caution before we write it: do not tell a skill to "load an MCP prompt." MCP prompts are exposed to *users* as commands — in Claude Code, this one appears as `/mcp__nova__full_research_instructions_prompt` — and whether the model can invoke a prompt programmatically depends on the host application; you cannot assume it can [7]. An agent-callable tool has no such ambiguity, which is exactly why we added one.

One convention matters here: MCP tools referenced from a skill must use fully qualified names in the form `ServerName:tool_name`, or the agent may fail to find the tool [2]. Since our server is registered as `nova`, we write `nova:extract_guidelines_urls`, not just `extract_guidelines_urls`.

Here is the complete final SKILL.md:

```markdown
---
name: nova-research
description: "Runs the full Nova research workflow to produce a research.md
document for an article. Use when the user wants to research a topic for an
article, gather and curate sources, or prepare research before writing.
Triggers on: 'research this topic', 'run the research workflow', 'prepare
research for this article', 'gather sources for', or any request to turn an
article guideline into a research.md."
---

# Nova Research

Run the complete research workflow using the `nova` MCP server.

## Execution

1. If the user did not provide a research directory (the folder containing
   `article_guideline.md`), ask for its absolute path before starting.
2. Call the `nova:get_research_instructions` tool. It returns the complete,
   step-by-step research workflow. Follow it exactly; do not improvise the
   sequence.
3. Use Nova's tools by their fully qualified names, for example:
   - `nova:extract_guidelines_urls` — extract URLs from the article guideline
   - `nova:scrape_research_urls` — scrape the selected research-result sources
   - `nova:run_perplexity_research` — fill gaps with additional queries
   - `nova:create_research_file` — produce the final research.md
4. Pause at every human-in-the-loop decision point defined in the workflow and
   wait for the user's choice before continuing.

## After Completion

Show the user the path to the generated research.md and briefly summarize
which sources were kept and which were discarded.
```

That is the whole skill: about 30 lines, a few hundred tokens, well under every budget.

> [!NOTE]
> **Claude Code extension: passing arguments.** Step 1 of this body asks the user for the research directory, but Claude Code can take it inline instead. Skills accept `$ARGUMENTS` (the whole string after the command), zero-indexed positional arguments (`$0` is the *first*, `$1` the second—shorthand for `$ARGUMENTS[N]`, not the shell's one-based convention), and named arguments, plus an `argument-hint` frontmatter field that feeds slash-menu autocomplete. Add `argument-hint: "[research-dir]"` and reference `$ARGUMENTS` in the body, and `/nova-research ~/articles/context-engineering` passes the path directly, with no prompt round-trip. This is a Claude Code convenience layered on the open format, so keep the "ask if it is missing" fallback for hosts that do not support it [4].

**Why fetch the workflow instead of copying it?** Single source of truth. Nova's research workflow is maintained in one place: a single Python function on the server that backs both the user-facing prompt and the agent-callable tool. If we add a tool or reorder a step, the fetched recipe stays synchronized automatically; the skill itself only needs attention if the server's tool names or contracts change, since the example names in its body would then go stale. Duplicating the full workflow into the skill would create two versions that drift apart silently.

What if you cannot modify the server — say, it belongs to someone else? Then give the skill a recipe it can reach without server changes: inline the workflow in the skill body, or put it in a referenced file next to SKILL.md. The architecture stays the same in every variant: skill = entry point, and the recipe lives wherever the agent can reliably reach it.

> [!NOTE]
> **Coming to the protocol: skills over MCP.** The MCP maintainers have flagged serving skills directly from MCP servers as one of the most exciting extensions in flight, and an active working group is now standardizing it, with a draft Skills Extension — built on the existing Resources primitive — in review [6][8]. The reasoning is exactly the one we just walked through: a server with many tools should ship the domain knowledge for using them, and serving the skill over MCP lets the server author continuously update that know-how without users installing or registering anything. When this lands, the natural home for `/nova-research` is Nova's own server — and the entry-point/recipe architecture you built here transfers unchanged. Today, checking the skill into the repo next to the server is the portable equivalent.

**Step 4: Use it.** There is no registration step. Claude Code watches existing skill directories live: the directory name becomes the slash command `/nova-research`, and the description enters the always-loaded metadata so the agent can also activate it automatically when a request matches [4]. One caveat applies to us: if the top-level `.claude/skills/` directory did not exist when your session started — which is true the first time you add skills to this repository — restart Claude Code once so it begins watching the new directory [4]. Type `/` in Claude Code and you will see it in the menu. Type "I need to prepare research for my next article" and, if the description does its job, the agent activates it on its own.

*Image 3: End-to-end flow of the /nova-research skill — user request, skill match via frontmatter, SKILL.md body loads, the body fetches the workflow via nova:get_research_instructions, the workflow directs calls to nova: tools, and the run ends with research.md on disk.*

Running the workflow end to end against Nova is an optional extension. The connection side is already handled for you: the course repository ships a project `.mcp.json` that defines the `nova` server (Claude Code spawns it over stdio via `uv run`), so opening Claude Code at the repository root and approving the project's MCP servers is enough — verify with `/mcp` that `nova` is connected and its tools are discovered. You will also need `GOOGLE_API_KEY`, `PPLX_API_KEY`, and `FIRECRAWL_API_KEY` configured as in Part 2B; `GITHUB_TOKEN` is optional, for GitHub sources. The core exercise, authoring and triggering the skill, needs none of this.

## Testing and Iterating on Your Skill

A skill has three failure surfaces: it does not trigger when it should, it triggers when it should *not*, or it triggers and then does the wrong thing. Test all three.

**Verify triggering, in both directions.** First the explicit path: type `/` and confirm `/nova-research` appears. Then the implicit path: open a *fresh* session (so nothing in the history hints at the skill) and try natural phrasings: "I need to research context engineering for an article," "gather sources for this guideline." If the skill does not activate, the fix is almost always in the description: add the trigger vocabulary your test phrases actually used. Then test the reverse—phrasings that should *not* fire it, such as "edit this paragraph for tone" or "summarize this file"—and confirm it stays dormant. A skill that activates on the wrong requests is as broken as one that never fires; when it over-triggers, tighten the description rather than broaden it, or—for a side-effectful skill—set `disable-model-invocation: true` so only an explicit `/nova-research` fires it. Make the negative cases genuinely tricky, not obviously unrelated, or they test nothing.

> [!NOTE]
> **When a skill silently won't load.** Malformed frontmatter is the most frustrating trap, because it fails quietly: if the YAML is invalid, Claude Code loads the body with *empty* metadata, so `/nova-research` still works but there is no description to match against—automatic activation never happens, which looks exactly like a weak description. (A `description` that spans several lines inside quotes, as in the examples above, is valid YAML and folds to one string; the real hazard is an auto-formatter reflowing it into an unquoted or otherwise broken form.) When a skill won't auto-activate, rule out loading *before* rewriting the description: run Claude Code with `--debug` to surface the parse error, ask "What skills are available?" to see whether it registered at all, run `skills-ref validate`, confirm the file sits at `.claude/skills/nova-research/SKILL.md` and is not double-nested in a second `nova-research/` folder, and restart the session if you just created the `.claude/skills/` directory [1], [4].

**Practice evaluation-driven development.** The best-practices guide is emphatic: create evaluations *before* writing extensive documentation. The loop is: run the agent on representative tasks without the skill and document the failures; build three scenarios that test those gaps; establish a baseline; write the minimal instructions that pass; iterate. Your evaluations, not your intuitions, are the source of truth for whether the skill helps [2]. This mirrors what we preached about agent evaluation in Part 3: measure before and after, or you are decorating, not engineering.

**Iterate with two instances.** A pattern from the best-practices doc that we find genuinely productive: use "Claude A" as an expert co-author who helps you edit the skill, and "Claude B" as a fresh instance that uses the skill on real tasks. Observe where B stumbles, bring the findings back to A, revise, repeat [2]. The author of a document is the worst judge of its ambiguities; a fresh model instance is a cheap, brutally literal proofreader.

Finally, test with every model you plan to use. Instructions that a stronger model fills in from judgment may need to be spelled out for a smaller one [2].

> [!NOTE]
> **Automate the loop with `skill-creator`.** Once you understand the manual loop, the official `skill-creator` plugin turns it into a repeatable harness: it stores test cases in `evals/evals.json`, runs each in an isolated subagent, grades the assertions, benchmarks with-skill against without-skill pass rates (alongside token and time overhead), and tunes the description by measuring hit rates on should-trigger *and* should-not-trigger prompts. Add it once with `/plugin marketplace add anthropics/claude-plugins-official`, then install `skill-creator`. Do the loop by hand first—the point is to understand what the harness is measuring [9].

## Common Failure Modes and Their Fixes

Here are the anti-patterns we see most often, each with its one-line fix.

- **The vague description.** "Helps with research." The model never picks it among 100+ skills. Fix: what + when + concrete trigger phrases, in third person [1], [2].
- **The monolithic body.** An 800-line SKILL.md taxing every turn after activation. Fix: keep it under 500 lines; split into reference files using one of the three patterns [2].
- **The nested labyrinth.** SKILL.md links to a file that links to a file. The agent partially reads and misses content. Fix: references one level deep; a table of contents in any reference file over 100 lines [2].
- **The dated snapshot.** "As of the current version..." rots in the repo. Fix: no time-sensitive info; quarantine deprecated approaches in an "old patterns" section [2].
- **The options menu.** Five alternative workflows the agent must choose among each run. Fix: one default plus an escape hatch [2].
- **The Windows path.** `scripts\validate.py` breaks on other platforms. Fix: always forward slashes [2].
- **The unqualified tool name.** `extract_guidelines_urls` instead of `nova:extract_guidelines_urls`; the agent may fail to find the tool. Fix: fully qualified `ServerName:tool_name` everywhere [2].
- **The punting script.** A bundled script that dies with a stack trace, dumping the error back on the agent. Fix: solve, don't punt; handle errors inside the script [2].

## Exercise: Package Brown's Writing Workflow as a Skill

Your turn. Brown, our LangGraph writing workflow from Lessons 22-26, exposes three MCP tools — `generate_article`, `edit_article`, and `edit_selected_text` — and using them well requires know-how an agent does not have by default: which tool fits which request, and what arguments each one needs. `generate_article` accepts only a `dir_path` (Brown loads its article, character, structure, mechanics, terminology, and tonality profiles internally), while the two editing tools take an `article_path` plus a `human_feedback` string carrying your instructions — and `edit_selected_text` additionally needs the exact `selected_text` with its `first_line_number` and `last_line_number`. Package that know-how as a second project skill: `brown-writing`.

Apply everything from this lesson:

1. Create `.claude/skills/brown-writing/SKILL.md` with a spec-compliant name matching the directory.
2. Write a third-person description covering what (runs Brown's writing workflow to generate or edit articles) and when (trigger phrases like "generate the article," "edit this article," "rewrite this paragraph with Brown").
3. In the body, teach the routing decision: a new draft → `brown:generate_article` with `dir_path`; a whole-article revision → `brown:edit_article` with `article_path` and `human_feedback`; a specific passage → `brown:edit_selected_text` with `article_path`, the exact `selected_text`, `first_line_number`, `last_line_number`, and `human_feedback`. Have the skill collect any missing arguments from the user before calling.
4. Decide the degree of freedom: the tool routing deserves explicit rules (medium freedom), while the phrasing of `human_feedback` can stay flexible.

Optional extension: add a style dimension. You cannot inject style rules into `generate_article` — its profiles are internal to Brown — but the skill can translate a concise set of style rules into the `human_feedback` it passes to the two editing tools. If the rules grow past a screen or two, split them into a `references/` file, one level deep.

A second extension: think about invocation control. Should the *model* be able to fire this skill on its own, or only the user? Claude Code's `disable-model-invocation: true` restricts triggering to the user, the right choice for side-effectful workflows like article generation, while `user-invocable: false` does the opposite for pure background knowledge [4]. Decide which, if either, fits this skill, and justify it. Then validate from the repository root: `skills-ref validate ./.claude/skills/brown-writing` (the `skills-ref` CLI is not part of the course environment; install it first from the agentskills reference repository at github.com/agentskills/agentskills) [1].

Your acceptance criteria are the pre-share checklist, condensed from the one in the best-practices guide [2]:

- [ ]  Description is specific, third person, with key terms and when-to-use guidance
- [ ]  SKILL.md body is under 500 lines
- [ ]  File references are one level deep
- [ ]  No time-sensitive information; one consistent term throughout
- [ ]  Any bundled scripts handle their own errors
- [ ]  At least three evaluation scenarios, tested with real tasks
- [ ]  Triggers on representative should-fire prompts and stays dormant on tricky should-not-fire ones

## Conclusion

You have now authored a proper agent skill end to end: a spec-compliant directory and frontmatter, a description engineered to trigger, a body designed for progressive disclosure, and an architecture, skill as entry point, server-hosted workflow as recipe, that keeps a single source of truth between your skill and your server. You also have the testing loop (three scenarios, baseline, Claude A/Claude B) and the failure-mode checklist to keep future skills honest.

This matters directly for what comes next. In Part 4, you will build and ship your own MCP server for certification. A bare server is a box of tools; a server that ships *with* a skill packaging its know-how, when to use it, which workflow tool to call, how to sequence the tools, is something a teammate can pick up and use in one command. That is what separates a demo from a product, and it is exactly the pairing we built here: Nova's server plus `/nova-research`. Skills are composable, portable across environments, and part of an open standard [5], so the one you write for your capstone will travel with it. And the protocol itself is moving to meet you: with a Skills-over-MCP extension in active standardization, the server-plus-skill pairing you are practicing now is the shape the ecosystem is converging on [8].

> [!NOTE]
> **Security note:** Skills are instructions and code that your agent will follow and execute. Only install skills from trusted sources, and audit what you install: the bundled scripts, the resource files, and especially any instructions that direct the agent to fetch from external network sources [3]. Treat a third-party skill with the same suspicion you would a third-party dependency, because that is what it is.

## References

1. Agent Skills. (n.d.). Agent Skills Specification. agentskills.io. https://agentskills.io/specification
2. Anthropic. (n.d.). Skill authoring best practices. Claude Docs. https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices
3. Anthropic. (n.d.). Equipping agents for the real world with Agent Skills. Anthropic Engineering. https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
4. Anthropic. (n.d.). Extend Claude with skills. Claude Code Docs. https://code.claude.com/docs/en/skills
5. Anthropic. (2025). Introducing Agent Skills. Claude Blog. https://claude.com/blog/skills
6. Soria Parra, D. (2026, April). The Future of MCP [Conference keynote]. AI Engineer Europe 2026, London. https://www.youtube.com/watch?v=v3Fr2JR47KA
7. Anthropic. (n.d.). Connect Claude Code to tools via MCP. Claude Code Docs. https://code.claude.com/docs/en/mcp
8. Model Context Protocol. (n.d.). Skills Over MCP Working Group charter. https://modelcontextprotocol.io/community/working-groups/skills-over-mcp
9. Anthropic. (2026). Improving skill-creator: Test, measure, and refine Agent Skills. Claude Blog. https://claude.com/blog/improving-skill-creator-test-measure-and-refine-agent-skills