# The What

Repository containing the code and lessons for our Agentic AI Engineering Course.

Everything lives under `lessons/`, which holds two kinds of folders side by side — the shared **code projects** and the numbered **lessons**:

```
lessons/
│
├── 📦 Code projects (the actual codebase)
│   ├── agents_integration/
│   ├── research_agent_part_2/
│   ├── research_agent_part_3/
│   ├── utils/
│   └── writing_workflow/
│
└── 📚 Lessons (prefixed with a number: 01, 02, 03, …)
    ├── 00_templates/
    ├── 01_ai_agents_landscape/
    ├── 02_workflows_vs_agents/
    └── … 
```

The lessons are made out of:
- a Jupyter Notebook that references the code in an educative and illustrative way. Also, it contains in the text cells snippets of code explaining the actual codebase.  
- an article that contains part of the code, but mostly focused to fill in all the theoretical gaps.

# Course Map (lessons ↔ code projects)

The course is split into **Part 1**, **Part 2 (A / B / C)**, and **Part 3**. Each lesson folder is prefixed with its number. Video-only lessons (15, 20, 21) have no folder and are omitted below.

The **central project** built across the course has two halves — a **research agent** and a **writing workflow** — that are later integrated and deployed. Each half maps to specific code projects, built in specific parts. (`utils` is the shared helper package — env loading, pretty-printing, GitHub download — imported by notebooks across **all** parts, never embedded as taught source, so it's not repeated per-part below.)

```
Part 1 — Foundations of Workflows and Agents        → no central project yet (concept lessons only)
│   01_ai_agents_landscape          (L1)
│   02_workflows_vs_agents          (L2)
│   03_context_engineering          (L3)
│   04_structured_outputs           (L4)
│   05_workflow_patterns            (L5)
│   06_tools                        (L6)
│   07_reasoning_planning           (L7)
│   08_react_practice               (L8)
│   09_RAG                          (L9)
│   10_memory_knowledge_access      (L10)
│   11_multimodal                   (L11)
│
Part 2A — Building Agentic Systems; scoping the central project   → code: design/scoping only
│   12_defining_central_project     (L12)
│   13_choosing_our_framework       (L13)   ← writing_workflow (brown) first appears as an illustrative example
│   14_agent_system_design          (L14)
│
Part 2B — The Central Research Agent              → code: research_agent_part_2  (mcp_server + mcp_client)
│   16_fastmcp                      (L16)
│   17_data_ingestion               (L17)
│   18_research_loop                (L18)
│   19_final_outputs                (L19)
│
Part 2C — The Writing Workflow + Integration      → code: writing_workflow (brown) + agents_integration
│   22_foundations_writing_workflow (L22)   ← writing_workflow (brown)
│   23_evaluator_optimizer          (L23)   ← writing_workflow (brown)
│   24_human_in_the_loop            (L24)   ← writing_workflow (brown)
│   25_integrate_agents             (L25)   ← agents_integration (orchestrates research agent + writing workflow)
│   26_end_to_end_demo              (L26)   ← agents_integration
│
Part 3 — Evaluation, Observability, Optimizations, Deployment
    27_observability                (L27)   ← writing_workflow (brown)
    28_ai_evals_offline_dataset     (L28)   ← writing_workflow (brown)
    29_ai_evals_offline_metrics_theory     (L29)   (theory; no central code)
    30_ai_evals_offline_metrics_practice    (L30)   ← writing_workflow (brown)
    31_continuous_integration       (L31)   ← writing_workflow (brown) (CI tests)
    32_authentication_docker        (L32)   ← research_agent_part_3 (deployable research agent)
    33_database_and_files           (L33)   ← research_agent_part_3
    34_continuous_deployment        (L34)   ← research_agent_part_3
│
Extending Agent Capabilities (between Part 3 and the Part 4 capstone; drafts in drafts/, not yet published)
    35_mcp_vs_skills_vs_cli         (L35)   (article only; no notebook — compares MCP, Agent Skills, and CLI as capability channels)
    36_creating_skills              (L36)   ← hands-on skill authoring; adds project skills under .claude/skills/ and a get_research_instructions tool to Nova's server
```
