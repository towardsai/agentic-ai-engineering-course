# Lesson 35: MCP vs. Skills vs. CLI: Three Ways to Give Agents Capabilities

In the previous lesson, we set up continuous deployment for our agent systems, closing out the production engineering arc of Part 3. Throughout Parts 2 and 3, we primarily gave our agents capabilities through one channel: the Model Context Protocol (MCP). Nova ships as a FastMCP server. Its Part 2 implementation originally exposed 11 tools, 2 resources, and 1 prompt; with the workflow-retrieval tool introduced in this lesson, it now exposes 12 tools. The deployable Part 3 implementation exposes 10 tools, 4 resources, and 1 prompt. Brown wraps its three LangGraph workflows in three coarse-grained MCP tools. We connected these servers to our own FastMCP client and to MCP-capable applications such as Cursor and Claude Desktop.

MCP is not the only way to give an agent capabilities, and it is not the best choice in every situation. Three recurring capability channels are especially useful: MCP servers, Agent Skills, and command-line interfaces (CLIs). Each solves a different problem, has a different cost profile, and can be combined with the others.

In this lesson, you will learn how to:

- Explain the roles of MCP servers, Agent Skills, and CLIs
- Compare their context cost, discoverability, validation, security, portability, and maintenance burden
- Decide which channel fits a capability instead of defaulting to an MCP tool
- Combine all three channels in one agent system

This prepares you for Lesson 36, where you will author a skill, and for the Part 4 capstone, where you will make these design choices for your own MCP server.

*Image 1: An agent at the center with three capability channels feeding into it: an MCP server (tools, resources, and prompts), a skills folder (`SKILL.md` files), and a terminal (CLI binaries such as `git`, `uv`, and `jq`).*

## From Coding Agents to Knowledge Work: Why Connectivity Is the Next Problem

Before comparing the channels, let's understand why this question matters now.

Coding agents are one of the clearest early examples of useful agents, partly because software work offers unusually strong feedback. Much of the work happens in a controlled environment, and tests, linters, compilers, and human review can verify the result [9]. These conditions reduce ambiguity and make failures easier to detect.

Knowledge-work agents operate in a different environment. A financial analyst, marketer, or operations specialist may need to combine information from several SaaS applications, internal systems, and shared drives. Verification still matters, but connectivity becomes a central engineering problem [9].

No single channel solves every connectivity problem. A useful stack combines skills for domain procedures, CLIs for direct execution in a prepared environment, and MCP for standardized access to tools and context across clients. These channels are not exhaustive—for example, agents may also use browser or computer-use interfaces—but they capture the main design choices in this course.

## The Three Capability Channels

Let's define the three channels precisely, because the ecosystem often blurs them.

An **MCP server** gives an agent access to capabilities through a protocol: tools with JSON Schema input contracts, resources identified by URIs, and reusable prompts. A client discovers them through protocol methods such as `tools/list`, `resources/list`, and `prompts/list`, which SDKs may expose as helpers such as `list_tools()` [7].

A **skill** gives the agent new *know-how*: a folder of instructions, scripts, and resources that it discovers and loads dynamically when a task matches. Skills primarily package procedures and judgment: how to use abilities the agent already has. They can also bundle executable code, which does add capability, but that code still runs through the host's execution tools and permissions [1].

The **CLI** gives the agent access to *existing abilities*: binaries installed in its execution environment and permitted by its host. The agent reaches them through a shell tool, without a custom MCP wrapper.

For any new capability, the first question is not how to write the MCP tool, but which channel the capability belongs in. To answer that, we need a clear picture of each one.

## MCP as a Capability Channel: A Recap

You know MCP well from Lessons 16-19, so we will keep this brief and focus on the properties that matter for comparison.

MCP is no longer tied to one vendor or agent framework. The OpenAI Agents SDK, Google's Agent Development Kit (ADK), and LangChain all provide MCP integrations [11], [12], [13]. This is the payoff we anticipated in Lesson 13: an MCP server can expose one capability surface to many MCP-capable clients. Individual clients may support different subsets of the protocol, so portability still needs to be tested rather than assumed.

MCP defines three server primitives. **Tools** are model-callable operations; they may read, compute, or cause side effects. **Resources** expose contextual data through URI-based read methods. **Prompts** are reusable, server-provided instruction templates that clients typically expose for user selection, such as Nova's `full_research_instructions_prompt` [7]. The familiar POST/GET analogy can be useful, but it is only an analogy: MCP primitives are not HTTP methods.

MCP standardizes four properties that otherwise vary by host or binary:

- **Input contracts.** Every tool declares a JSON Schema for its inputs, which FastMCP generates from Python type hints [8]. FastMCP uses Pydantic's coercive validation by default, so a compatible value such as the string `"10"` can become an integer. Setting `strict_input_validation=True` rejects type mismatches before the tool function runs.
- **Discovery.** After connecting and negotiating capabilities, a client can query the server's advertised tools, resources, and prompts, with names, descriptions, and schemas included. This is how Nova's client prints its startup banner.
- **Transports.** With stdio, the client launches the server as a subprocess and communicates over standard input and output. With Streamable HTTP, the server can run remotely. The capability surface can remain the same even though deployment and transport configuration differ [7].
- **Authorization semantics.** Remote HTTP deployments can use MCP's OAuth-based authorization flow. A server can also enforce application-specific policies, while local stdio servers usually rely on process, environment, and operating-system boundaries [7].

This structure comes with a context cost. Eager MCP clients load every connected tool definition into the model's context upfront, and agents connected to many servers can consume hundreds of thousands of tokens before reading the user's request [6]. This is how clients have traditionally behaved, not an inherent property of the protocol; treat it as a failure mode to account for in clients without progressive discovery. Keep that number in mind; we will come back to it when comparing the channels.

The remedy is **progressive discovery**: defer full tool definitions behind a search or loading mechanism, then load only the tools needed for the task. Claude Code enables MCP tool search by default, and the OpenAI Agents SDK supports deferred hosted MCP tools through `ToolSearchTool` [10], [11]. A custom client can implement a similar pattern by exposing a small search tool rather than injecting the full catalog. Skills apply the same principle to instructions through progressive disclosure.

## Skills: Packaging Know-How for Your Agent

Skills are the new concept in this lesson, so let's cover them in detail.

Agent Skills are organized folders of instructions, scripts, and resources that an agent can discover and load dynamically to perform better at specific tasks [1], [2]. A useful analogy is an onboarding guide for a new hire [1]. A new colleague is already capable; what they lack is your team's procedures, conventions, and hard-won lessons. A skill transfers that know-how to the agent without requiring you to repeat it in every conversation.

Concretely, a skill is a directory containing at minimum a `SKILL.md` file, with optional `scripts/`, `references/`, and `assets/` directories [3]. The core format is an open standard specified at agentskills.io. A standards-compliant skill can be reused by hosts that implement the format, although host-specific frontmatter fields and tool names may still require adaptation. This matters for the same reason MCP mattered in Lesson 13: a shared format reduces coupling to one runtime.

### From a Repeated Prompt to a Reusable Skill

A useful way to understand a skill is as a prompt that has been packaged for reuse. Imagine that every time you start a research task, you paste the same long prompt into the conversation: first inspect the guidelines, then collect sources, then filter them, then write `research.md`. That works, but the procedure lives in your clipboard and depends on you remembering when and how to use it.

A skill improves this workflow in five ways:

- **Reuse:** You write the procedure once and invoke it whenever the same type of task appears. This makes skills a good fit for repetitive work such as reviewing a pull request, preparing a report, or running Nova's research workflow.
- **Discovery and invocation:** The skill's name and description tell the agent when the procedure applies. Depending on the host, you can invoke it directly as a command or let the agent activate it when the request matches.
- **Packaging:** The instructions can travel with scripts, reference documents, examples, and templates instead of being limited to one block of prompt text.
- **Transfer and sharing:** A project skill can be committed to Git, reviewed like code, and shared with everyone who clones the repository. The open format also makes it easier to move the skill to another compatible host, although host-specific tools or frontmatter may need small changes.
- **Editing and versioning:** You can update `SKILL.md` or its supporting files as the procedure improves, inspect the diff, and roll the change out through normal version control. You are maintaining one shared procedure instead of correcting several copied prompts.

The executable part still needs one distinction. A skill's Markdown body is instruction context; loading it does not execute the task deterministically. The user or agent invokes the skill, the model interprets its instructions, and the host calls the required tools. If the skill includes a script, the host may execute that script through its normal permissions. If you need a task to run on a schedule or without a model deciding what to do, use an automation or deterministic workflow rather than treating the skill itself as a scheduler.

Skills can bundle pre-written scripts that the agent executes instead of reimplementing deterministic logic with generated tokens [1]. A useful rule is to keep judgment in the model and move stable, repeatable operations into code.

> [!NOTE]
> **Security note:** A skill may contain instructions and executable code. Install skills only from trusted sources, and audit their dependencies, bundled resources, and any instructions that fetch content from external sources [1]. Treat a third-party skill with the same care as any other code or integration you install.

## Anatomy of a Skill: SKILL.md and Progressive Disclosure

`SKILL.md` has two parts: YAML frontmatter and a Markdown body.

The portable Agent Skills specification requires two frontmatter fields. `name` identifies the skill and must be 1-64 characters, use lowercase letters, numbers, and hyphens without leading, trailing, or consecutive hyphens, and match the parent directory name. `description` must be 1-1024 characters and should explain both what the skill does and when to use it [3]. The body is free-form Markdown containing the procedure, examples, and links to supporting files. Some hosts are more permissive—for example, Claude Code can infer a missing name from the directory—but following the open specification gives you the best portability [5].

The mechanism that makes skills cheap is **progressive disclosure**, and it works in three levels [1], [3]:

1. **At startup**, only the metadata (name + description) of every installed skill is pre-loaded into the system prompt, at roughly 100 tokens per skill.
2. **On activation**, when the agent judges the skill relevant, the full SKILL.md body loads. The spec recommends keeping it under 5,000 tokens (and under 500 lines).
3. **On demand**, linked files (`reference.md`, `forms.md`, and so on) load only if the task actually needs them.

Think of it as a well-organized manual: a table of contents first, then specific chapters, then a detailed appendix, with information loaded only as needed [1]. A skill can therefore include much more supporting material than should be placed in the system prompt, because reference files do not consume context until the agent reads them.

*Image 2: A pyramid showing the three progressive-disclosure levels: skill metadata (about 100 tokens, loaded at startup), the `SKILL.md` body (under 5,000 tokens recommended, loaded on activation), and linked files or scripts (loaded or executed on demand).*

In Claude Code, each skill is user-invocable as a slash command by default: the directory name becomes `/skill-name`. Claude can also activate it automatically when the description matches the request; frontmatter flags can restrict either path, as Lesson 36 explains. No registration is needed, and changes within existing skill directories are detected during the current session. If a skill was already invoked, re-invoke it after editing so the updated content enters the conversation. If the top-level `.claude/skills/` directory did not exist when the session started, Claude Code must be restarted once before it can watch that new directory [5].

The three locations most relevant to this course are personal (`~/.claude/skills/`, available across your projects), project (`.claude/skills/`, checked into the repository), and plugin (`<plugin>/skills/`). Claude Code also supports enterprise-managed skills and nested project skill directories [5].

When should something become a skill? A clean heuristic: create a skill when you keep pasting the same instructions or multi-step procedure into chat, or when a section of your CLAUDE.md has grown into a *procedure* rather than a *fact*. Unlike always-loaded context files, a skill's body loads only when it's used [5].

> [!NOTE]
> **The context window is a public good.** Once Claude Code loads a skill, its rendered instructions remain in the conversation for the rest of the session. Every line therefore competes with conversation history and other useful context. Add only the guidance the agent is unlikely to know without the skill [4], [5].

## A /research Skill for Nova

Let's make this concrete with Nova. In Lesson 16, you ran its workflow from our custom client by typing `/prompt/full_research_instructions_prompt`. When Nova is connected to Claude Code under the server name `nova`, the same server-hosted prompt appears as `/mcp__nova__full_research_instructions_prompt` [10]. In both cases, the user must know which prompt starts the workflow. A skill provides a task-oriented entry point.

There is a design constraint to respect first. MCP prompts are designed as a user-controlled primitive, and Claude Code exposes them as commands. You cannot assume the model can invoke one programmatically from inside a skill because model access depends on the host application [10]. Instead of telling the skill to "load the prompt," we make the workflow agent-callable by adding a small retrieval tool to Nova's server, backed by the same Python function as the prompt:

```python
# mcp_server/src/routers/tools.py: reuses the function behind the MCP prompt
import opik
from fastmcp import FastMCP

from ..prompts.research_instructions_prompt import (
    full_research_instructions_prompt as _get_research_instructions,
)
from ..utils.opik_utils import opik_context


def register_mcp_tools(mcp: FastMCP) -> None:
    @mcp.tool()
    @opik.track(type="tool")
    async def get_research_instructions() -> str:
        """Return the full research workflow instructions."""
        opik_context.update_thread_id()
        return await _get_research_instructions()
```

The `await` is required because the underlying prompt function is asynchronous. The Opik decorator and thread update match the observability pattern used by Nova's other tools. This tool now ships in the course repository, bringing the Part 2 server to 12 tools. The skill therefore has an operation it can reliably call. Here is a minimal `/research` skill you could place at `.claude/skills/research/SKILL.md`. In Lesson 36, you will author the complete `/nova-research` version.

```markdown
---
name: research
description: "Runs deep research with the Nova MCP server. Use when the user asks to research a topic, gather sources, or produce a research document."
---

# Research

1. Call `nova:get_research_instructions` to fetch the research
   workflow, then follow it step by step.
2. Use Nova's tools with fully qualified names
   (e.g. `nova:extract_guidelines_urls`) as the workflow directs.
3. When finished, show the user the path to `research.md` and a
   two-sentence summary of what was found.
```

Note three details. The description is written in the third person and includes natural trigger terms because the skill catalog is the model's primary signal for automatic activation [4]. The tool references use the fully qualified `ServerName:tool_name` form; without the `nova:` prefix, the agent may fail to find the tool [4]. Finally, the body does *not* duplicate the research workflow. It fetches the workflow through an agent-callable tool.

## Skills vs. Prompts vs. Tools

A common question at this point: Nova already has a server-hosted prompt that describes the whole workflow, so why add a skill?

First, separate an ordinary prompt from a skill. An ordinary prompt is instruction text placed directly into one conversation. An MCP prompt stores reusable instruction text on a server so a client can fetch it. A skill goes further: it packages instructions with activation metadata and, optionally, files or scripts that help the agent carry out the procedure. This makes a skill easier to discover, repeat, transfer, edit, and version than a prompt that users copy and paste manually.

That does not mean every good prompt should become a skill. Use an ordinary prompt for a one-off request. Create a skill when the instruction represents a recurring procedure, needs supporting material, or should be shared consistently across users or projects.

In Nova's design, the pieces play these roles:

- **Ordinary prompts** express one-off user intent: "research this topic for me."
- **Tools** provide actions: "search the web," "extract these URLs."
- **MCP prompts** are the recipe: the ordered instructions for how to use the tools.
- **Skills** are the entry point: they define *when* to activate, which server and recipe to use, and what to do before and after.

These are architectural choices, not protocol definitions. An MCP tool can expose a coarse-grained workflow—Brown's three tools do exactly that—a prompt can be a one-liner, and a skill can contain an entire procedure with executable code.

The skill sits between the user and everything else. In our `/research` skill, the body simply says "fetch Nova's workflow and follow it." This keeps a single source of truth: the workflow text lives in one Python function on Nova's server, and that same function now backs both the user-facing prompt and the agent-callable tool. If we improve the workflow, the skill picks up the change automatically; only a change to the server's tool names or contracts would require touching it.

This is also why we did not just point the skill at the prompt. MCP prompts are a *user-facing* primitive: hosts expose them as commands for a human to run (in Claude Code, `/mcp__nova__full_research_instructions_prompt`), and whether the model can invoke them programmatically depends entirely on the host; you should not assume it can [10]. If you cannot, or do not want to, modify the server, the alternative is to inline the workflow in the skill body or in a referenced file. The pattern is unchanged either way: the skill is the entry point, and the recipe lives wherever the agent can reliably reach it. This is why skills are best understood as *complementing* MCP servers, teaching agents the more complex workflows that involve external tools and software, rather than competing with them [1].

## The CLI: The Oldest Capability Channel

The third channel is the oldest one: the command line existed long before agents. Many coding-agent hosts, including Claude Code, provide a shell tool. Subject to the host's permissions and sandbox, the agent can run installed binaries such as `git`, `gh`, `uv`, `ffmpeg`, and `jq`. The Agent Skills specification does not require Bash; script execution depends on the host implementation and its environment [3], [5].

The main appeal of this channel is its low integration cost. An agent does not need an MCP wrapper to use `ffmpeg`; it can invoke the installed binary directly. Installation, authentication, permissions, and environment setup still matter—the missing piece is only the wrapper code. Discovery usually happens through `--help`, man pages, and documentation rather than a shared machine-readable schema. Composition is straightforward: pipes and files let the agent chain commands such as `gh pr list | jq ...`. Our own tooling reflects this: `uv run` starts Nova's server, and Git underpins the deployment workflows from Part 3.

The CLI has a clear boundary condition: CLIs are particularly good when you have a *local* agent, where you can assume a sandbox and a good execution environment [9]. Weaken those assumptions (a remote agent, an environment you do not control, a capability that lives behind someone else's service) and the CLI's advantages shrink. They do not disappear entirely: remote agents can run CLIs in managed sandboxes, and mature CLIs like `gh` and `gcloud` reach network services securely. But you are now assembling discovery, contracts, and auth yourself, tool by tool. This boundary is exactly where MCP becomes valuable again.

The efficiency gains have been quantified. Direct tool calling is token-expensive in eager clients: all tool definitions load upfront, and both intermediate results and definitions flow through context. Letting the agent write code against tool APIs instead of making direct tool calls produced a 98.7% token reduction, from 150,000 tokens down to 2,000, on a Google Drive-to-Salesforce workflow [6]. The broader lesson is that models are adept at writing code, and developers should use that strength to make agent-to-server interactions more efficient [6]. The same logic powers skills' bundled scripts: when executed directly, a script's contents need not enter context; typically only its output does [4].

This pattern is known as **programmatic tool calling**, or "code mode." Instead of orchestrating one model-visible tool call at a time, the model writes a program that composes several operations inside an execution environment [6], [9]. MCP can support this pattern when tools declare output schemas and clients generate code-facing wrappers from their contracts. Output schemas are optional, so clients must still handle tools that return only content blocks.

The trade-off is equally explicit: running agent-generated code requires a secure execution environment with sandboxing, resource limits, and monitoring [6]. A CLI does not provide one shared permission protocol across binaries. Execution guardrails come from the agent host through permission prompts, sandboxing, and allow rules such as `allowed-tools: Bash(git:*)`, with the operating system adding file and process permissions underneath [3], [4]. Authentication is whatever each binary or backing service implements, such as `gh auth`, cloud IAM, or scoped tokens.

MCP does not make a server secure by itself. It standardizes an OAuth-based authorization flow for HTTP deployments and provides schemas and tool annotations that clients and servers can use in their policies. Schemas validate structure; they are not authorization controls. With CLIs, equivalent controls are distributed across the host, operating system, and individual binaries.

## Comparing the Three Channels

*Table 1: Choosing between MCP, skills, and CLIs*

| Question | MCP server | Skill | CLI |
| --- | --- | --- | --- |
| Best fit | A capability needs standardized contracts, remote access, reuse across clients, or complex execution behind a stable service boundary | The agent already has the required abilities but needs reusable procedures and judgment for coordinating them | A mature binary already provides the capability in a controlled environment |
| What it provides | Discoverable tools, resources, and prompts | Instructions with optional scripts, references, templates, and activation metadata | Direct access to installed software |
| How it is discovered | MCP protocol discovery | The skill's name and description | Documentation, `--help`, and model knowledge |
| Context behavior | Depends on the client; tool search can defer definitions until they are needed | Metadata loads first; full instructions load when the skill is activated | Help text, commands, and output enter context when the agent uses them |
| Main trade-off | More integration and operational overhead | The model interprets the instructions, and the required tools must be available in the host | Environment, authentication, and safety controls vary between binaries |

These are starting points, not mutually exclusive categories. A skill can coordinate MCP tools and CLI commands in the same task. The detailed behavior also depends on the client, host permissions, installed dependencies, and deployment environment.

## A Decision Framework: Which Channel When

As in Lesson 13, we reason from decision axes, not brands. When a new capability request lands, walk through three questions, and treat the answers as tendencies, not hard gates.

**Would standardized discovery, schema contracts, multi-client reach, or a stable execution boundary pay for themselves?** Then prefer an MCP server. External APIs and secrets do not automatically require MCP—mature CLIs such as `gh` and `gcloud` handle both—but a server boundary becomes valuable when a capability needs a reusable contract, cross-host access, standardized authorization semantics, resources, or elicitation. This is why Nova is a server: its research tools are used from Claude Code, Cursor, and our own client, and its remote deployment must enforce authentication.

Complexity is another reason to introduce that boundary, but the *kind* of complexity matters. A skill can describe a sophisticated procedure and help the model coordinate many tools. What it does not provide by itself is reliable infrastructure for database access, shared state, secret handling, retries, concurrency, rate limiting, heavy dependencies, or long-running operations. When those concerns belong to one capability, put them behind an MCP server and expose a small, task-shaped tool contract. The model sees a manageable operation while the server owns the implementation complexity.

This does not mean exposing every internal step as an MCP tool. Often the better design is one coarse-grained tool that encapsulates a complete unit of work. Brown follows this pattern: the internal LangGraph workflows are complex, but clients see three workflow-level tools rather than every graph node. A skill can then teach the agent *when* and *how* to use those tools without reproducing their implementation.

**Is it know-how or process, rather than a new ability?** Then it is a skill. If the agent already *could* do the task with existing tools but does it inconsistently, forgets steps, or needs the same long prompt pasted every time, no new tool will fix that. Package the procedure as a skill and let progressive disclosure keep its upfront cost low until it is needed [5].

**Does a mature binary already exist in a controlled environment?** Then start with the CLI. Wrapping `git` in a bespoke MCP tool adds a server and maintenance burden. Build a wrapper only when the shared contract, policy boundary, or remote access justifies it, and use the host's sandbox and permissions to control execution.

One framing ties this together: MCP servers and CLIs provide *abilities*, while skills primarily provide *know-how* for coordinating those abilities. Bundled scripts blur the boundary because a skill can carry executable logic, but that logic still runs through the host's tools and permissions. When deciding between a skill and a tool, ask whether the missing piece is an ability or a procedure.

## Combining the Three Channels

The comparison table is not meant to help you pick a single winner. A single agent setup may use all three channels at once.

A skill can document a CLI: an internal deployment skill might say, "run these `gh` and `git` commands in this order, with these checks," and a supporting host may pre-approve narrowly scoped commands through `allowed-tools` [3], [5]. A skill can also orchestrate MCP tools: our `/research` skill is an entry point over Nova's workflow-retrieval and research tools. Finally, code written or bundled as a script can compose tool APIs while filtering intermediate results before they enter the model's context—the pattern behind Anthropic's 150,000-to-2,000-token example [6].

Map this onto the capstone you will complete in Part 4:

- **Nova stays an MCP server.** Its capabilities need schema-described inputs, work from multiple clients, and can be deployed remotely with authentication. That is the MCP sweet spot.
- **A skill packages the know-how of using Nova.** The `/research` skill makes the workflow one command, shareable via git, and auto-activating on trigger phrases, without duplicating the recipe, which stays in a single function on Nova's server.
- **The CLI underpins everything.** `uv run` launches the server that Claude Code spawns over stdio per `.mcp.json`; git versions the skill alongside the code.

*Image 3: The user invokes `/research`; the skill fetches Nova's workflow through `nova:get_research_instructions`; the agent follows the workflow with Nova's MCP tools; and Nova produces `research.md`.*

This single flow uses all three channels, each doing the job it is best suited for. When you build your own MCP server for certification, make this same three-way allocation deliberately: which parts are abilities, which parts are know-how, and which parts already exist as binaries.

The broader direction is compositional: agents can combine computer use, CLIs, MCP servers, and scripts instead of forcing every capability through one interface [9]. The ecosystem is still developing, so hosts differ in which combinations, permission models, and discovery mechanisms they support.

## Conclusion

The lesson here echoes Lesson 13's closing philosophy: prioritize concepts over brands. "MCP," "skills," and "CLI" are today's names for three durable ideas: discoverable, schema-described contracts for abilities; progressively disclosed, portable know-how; and direct access to the existing software ecosystem. Specific products will change, but an engineer who can assign a capability to the right channel, and combine channels when one is not enough, will adapt as the ecosystem evolves.

Keep the decision framework in mind: standardized contracts and multi-client reach point to MCP; procedure and process point to a skill; an existing mature binary in a controlled environment points to the CLI. The context costs differ too: eager MCP clients load tool definitions up front, progressive-discovery clients defer them, skills load instructions on demand, and CLIs shift more of the safety burden to the host and operating system.

In the next lesson, you will author a proper skill for Nova: writing the frontmatter and body against the agentskills.io spec, applying the authoring best practices (evaluation-driven development, degrees of freedom, the one-level-deep reference rule), and testing activation in Claude Code. Then, in Part 4, you will build and certify your own MCP server, and the skill you author in Lesson 36 will directly upgrade how that capstone is used.

## References

1. Anthropic. (n.d.). Equipping agents for the real world with Agent Skills. Anthropic Engineering. https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
2. Anthropic. (2025). Introducing Agent Skills. Claude Blog. https://claude.com/blog/skills
3. Agent Skills. (n.d.). Agent Skills specification. agentskills.io. https://agentskills.io/specification
4. Anthropic. (n.d.). Skill authoring best practices. Claude Docs. https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices
5. Anthropic. (n.d.). Extend Claude with skills. Claude Code Docs. https://code.claude.com/docs/en/skills
6. Anthropic. (n.d.). Code execution with MCP. Anthropic Engineering. https://www.anthropic.com/engineering/code-execution-with-mcp
7. Model Context Protocol. (n.d.). Model Context Protocol. https://modelcontextprotocol.io/
8. Prefect Technologies, Inc. (n.d.). FastMCP. https://gofastmcp.com/
9. Soria Parra, D. (2026, April). The Future of MCP [Conference keynote]. AI Engineer Europe 2026. https://www.youtube.com/watch?v=v3Fr2JR47KA
10. Anthropic. (n.d.). Connect Claude Code to tools via MCP. Claude Code Docs. https://code.claude.com/docs/en/mcp
11. OpenAI. (n.d.). Model Context Protocol (MCP). OpenAI Agents SDK documentation. https://openai.github.io/openai-agents-python/mcp/
12. Google. (n.d.). Model Context Protocol (MCP). Agent Development Kit documentation. https://google.github.io/adk-docs/mcp/
13. LangChain. (n.d.). Model Context Protocol (MCP). LangChain documentation. https://docs.langchain.com/oss/python/langchain/mcp
