# The What

Brown the writing workflow: An MCP Server implemented with LangGraph, FastMCP and Gemini that generates articles based on guidelines and research.

## Project Structure

```
writing_workflow/
├── src/brown/                  # Main package (the "brown" writing workflow)
│   ├── base.py                 # Shared base abstractions
│   ├── builders.py             # Object/graph builders
│   ├── config.py               # Workflow configuration
│   ├── config_app.py           # App-level configuration
│   ├── loaders.py              # Input loaders (articles, research, guidelines)
│   ├── renderers.py            # Output renderers
│   ├── entities/               # Domain models (articles, guidelines, research, reviews, …)
│   ├── models/                 # LLM model config & factory (get_model, fake_model)
│   ├── nodes/                  # LangGraph nodes (writer, reviewer, media generator, tools)
│   ├── workflows/              # LangGraph workflows (generate / edit article, edit selection)
│   ├── mcp/                    # FastMCP server exposing the workflows
│   ├── evals/                  # Offline evaluation (dataset, tasks, metrics)
│   │   └── metrics/            #   follows_gt & user_intent metrics (+ few-shot examples)
│   ├── observability/          # Opik tracing, datasets, evaluation hooks
│   ├── memory/                 # Conversation/state memory
│   └── utils/                  # Misc helpers (network, …)
│
├── scripts/                    # CLI entry points (eval dataset, MCP CLI, run eval)
├── configs/                    # Run configs (debug.yaml, course-gemini-flash/pro.yaml)
├── inputs/                     # Test inputs, eval dataset, profiles & examples
│   ├── profiles/               #   article / structure / tonality / terminology profiles
│   ├── tests/                  #   sample lessons (tiny → large) for E2E runs
│   ├── evals/                  #   ground-truth eval dataset
│   └── examples/               #   reference course-lesson outputs
├── outputs-cached/             # Cached eval outputs (flash & pro runs)
├── tests/brown/                # Unit tests mirroring src/brown layout
│   └── fixtures/               # Test fixtures (articles, guidelines, research, configs)
│
├── Makefile                    # Project commands (see "The How" below)
├── pyproject.toml              # Package metadata & dependencies
└── uv.lock                     # Locked dependency versions
```

# The How

Interact with the project through the Makefile commands. If a custom command is not available, run the script directly using uv: `uv run ...`

## Run End-to-End Tests

Whenever testing the code run the `make brown-generate-article` command with the following inputs:
- the test article from `00_sample_tiny`
- the two configs:
  - first with `debug.yaml` to see if the code works well
  - secondly with `course-gemini-flash.yaml` to see if the code works well with the model as well
