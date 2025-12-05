## Brown Writing Agent – Terminal Usage Guide

This folder contains the **Brown writing agent**, a sophisticated AI-powered system for generating and editing high-quality articles based on research data and guidelines.

Brown implements three core workflows:

- **Article Generation**: Create comprehensive articles from scratch using research data and style guidelines
- **Article Editing**: Refine entire articles based on human feedback and requirements
- **Selected Text Editing**: Make targeted improvements to specific article sections

This README shows how to:

- Install the project with `uv`
- Configure API keys
- Run Brown workflows from the terminal
- Use Brown as an **MCP server** in other applications

> [!NOTE]
> This will be run as an independent project from the rest of this repository.

---

## 1. Directory Layout

From the repository root (`course-ai-agents`):

```bash
lessons/writing_workflow/
  ├── configs/              # YAML configuration files
  │   ├── course.yaml       # Production config
  │   └── debug.yaml        # Testing config (fake models)
  ├── inputs/               # Input data and resources
  │   ├── profiles/         # Writing style and character profiles
  │   ├── examples/         # Example articles for context
  │   ├── tests/            # Test input directories with guidelines and research
  │   └── evals/            # Evaluation datasets
  ├── scripts/              # CLI scripts for running workflows
  │   ├── brown_mcp_cli.py  # Main CLI interface for Brown workflows
  │   ├── brown_create_eval_dataset.py
  │   └── brown_run_eval.py
  ├── src/                  # Source code
  │   └── brown/            # Brown agent implementation
  │       ├── workflows/    # Article generation and editing workflows
  │       ├── nodes/        # Workflow nodes (writer, reviewer, editor)
  │       ├── entities/     # Domain models (articles, guidelines, research)
  │       ├── mcp/          # MCP server implementation
  │       ├── models/       # LLM model configuration and wrappers
  │       ├── evals/        # Evaluation framework
  │       └── observability/ # Observability infrastructure (tracing, monitoring, evals)
  └── tests/                # Test suite
```

---

## 2. Prerequisites

- **Python**: The project is pinned to Python `3.12.11` in `pyproject.toml`.  
  If you followed the main course `README.md`, you already have a compatible Python installed.
- **`uv`** package manager: see the root `README.md` for installation instructions.
- **GNU Make**: Used to run all the commands from the project. Get installation instructions using this prompt: `How do I install GNU Make on ${YOUR_OS}`. Change `${YOUR_OS}` with MacOS, Ubuntu, etc.
- All the commands below assume you are using a MacOS / Linux shell with commands based on Bash/zsh. In case you are using Windows, we assume to use a shell compatible with Linux such as WSL.

You do **not** need to manually create or activate virtual environments; `uv` will manage them automatically.

---

## 3. Install Dependencies with `uv`

From the repository root:

```bash
cd lessons/writing_workflow
uv sync
```

This will create a `.venv` inside `writing_workflow` and install all dependencies including LangGraph, LangChain, FastMCP, and LLM providers.

---

## 4. Configure Environment Variables (API Keys)

Brown reads configuration from:

- Real environment variables (`export VAR=...`), **and/or**
- A local `.env` file in the project directory using the `.env.example` as an example.

For a smooth setup, create a `.env` file:

```bash
lessons/writing_workflow/.env
```

You can use `.env.example` as a template: `cp .env.example .env`

### 4.1. Required and optional variables

```bash
GOOGLE_API_KEY=your-google-api-key-here 
OPIK_API_KEY=your-opik-api-key-here
```

**Where to get these keys (all have free tiers):**

- **`GOOGLE_API_KEY` (Gemini)**: Create a key in Google AI Studio (`https://aistudio.google.com/app/apikey`).  
  Google offers a free tier suitable for experimentation and this course.
- **`OPIK_API_KEY`**: Create a free account at `https://www.comet.com/site/products/opik/` and find the API KEY based on this [doc](https://www.comet.com/docs/opik/faq#where-can-i-find-my-opik-api-key-)

You can either:

- Put this content in a `.env` file in `writing_workflow`, or
- Export them once in your shell before running anything:

```bash
export GOOGLE_API_KEY=...
export OPENAI_API_KEY=...
export OPIK_API_KEY=...
```

> **Tip:** If something fails with "missing API key" errors, first verify the variables above are set and visible in the shell where you run `uv`.

---

## 5. Input Directory Structure

Brown expects input directories to contain specific files depending on the workflow:

### For Article Generation:

```bash
your_article_directory/
  ├── article_guideline.md  # Required: Article requirements, outline, style guidelines
  └── research.md           # Required: Research data, sources, and context
```

### For Article Editing:

```bash
your_article_directory/
  ├── article_guideline.md  # Required: Original article requirements
  ├── research.md           # Required: Research data for context
  └── article.md            # Required: Existing article to edit
```

### Example Input Directory

You can find a complete example at:

```bash
lessons/writing_workflow/inputs/tests/00_debug/
  ├── article_guideline.md
  └── research.md
```

This example demonstrates the expected format for guidelines and research data.

---

## 6. Running Commands

Brown provides a convenient `Makefile` with commands for all workflows. All commands should be run from the `writing_workflow` directory.

### 6.1. Generate an Article

To generate a new article from scratch:

```bash
cd lessons/writing_workflow
make brown-generate-article DIR_PATH=inputs/tests/00_debug
```

**Parameters:**
- `DIR_PATH`: Path to the directory containing `article_guideline.md` and `research.md` (relative to `writing_workflow`)

**What happens:**
1. Brown reads the article guidelines and research data
2. Loads character profiles and writing style guidelines from `inputs/profiles/`
3. Generates an initial article draft
4. Runs multiple review and revision iterations (default: 2, configured in `configs/course.yaml`)
5. Saves the final article as `article.md` in the input directory
6. Creates checkpoints for each iteration (e.g., `article_000.md`, `article_001.md`)

### 6.2. Edit an Entire Article

To edit an existing article based on an optional human feedback:

```bash
cd lessons/writing_workflow
make brown-edit-article DIR_PATH=inputs/tests/00_debug HUMAN_FEEDBACK="Make the introduction more engaging and add more technical depth to section 2"
```

**Parameters:**
- `DIR_PATH`: Path to the directory containing `article.md`, `article_guideline.md`, and `research.md`
- `HUMAN_FEEDBACK`: Your feedback or instructions for editing (must be quoted if it contains spaces). Feedback is optional, if not provided the editing will be done solely based on internal evaluation techniques.

**What happens:**
1. Brown reads the existing article, guidelines, and research
2. Analyzes your feedback against the article context
3. Reviews the article and identifies areas for improvement
4. Generates comprehensive edits addressing your feedback
5. Saves the edited article back to `article.md`

### 6.3. Edit a Selected Text Section

To make targeted edits to a specific section of an article:

```bash
cd lessons/writing_workflow
make brown-edit-selected-text DIR_PATH=inputs/tests/00_debug HUMAN_FEEDBACK="Simplify this explanation and add a concrete example" FIRST_LINE=10 LAST_LINE=20
```

**Parameters:**
- `DIR_PATH`: Path to the directory containing `article.md`
- `HUMAN_FEEDBACK`: Your feedback for the selected section
- `FIRST_LINE`: Starting line number of the section to edit (1-indexed)
- `LAST_LINE`: Ending line number of the section to edit (1-indexed)

**What happens:**
1. Brown extracts the specified text section from `article.md`
2. Reviews the section in the context of the full article
3. Applies targeted edits based on your feedback
4. Replaces the original section with the edited version
5. Saves the updated article back to `article.md`

### 6.4. Using Absolute Paths

You can also use absolute paths for `DIR_PATH`:

```bash
make brown-generate-article DIR_PATH=/absolute/path/to/your/article/directory
```

### 6.5. Evaluation Commands

Brown includes an evaluation framework to assess article quality.

**Create an evaluation dataset:**

```bash
cd lessons/writing_workflow
make brown-create-eval-dataset
```

This creates an evaluation dataset from the articles in `inputs/evals/dataset/` with the name `brown-course-lessons`.

**Run evaluations:**

```bash
cd lessons/writing_workflow
make brown-run-eval
```

This runs the evaluation using the `follows_gt` metric (checks if generated articles follow ground truth patterns) and caches results in `outputs/evals/`.

**Parameters:**
- The dataset name is `brown-course-lessons` (created by the previous command)
- Metrics: `follows_gt` (evaluates adherence to ground truth articles)
- Results are cached in `outputs/evals/` for faster re-runs
- Uses 1 worker for sequential processing

---

## 7. Running Full Writing Workflow

This section provides an in-depth look at each workflow and what happens under the hood.

### 7.1. Article Generation Workflow

The article generation workflow is Brown's most comprehensive process, designed to create high-quality articles from research data.

**Input Requirements:**
- `article_guideline.md`: Contains the article outline, target audience, key topics, and style requirements
- `research.md`: Contains all research data, sources, quotes, and context

**Workflow Steps:**

1. **Load Context**
   - Reads article guidelines and research data
   - Loads character profile from `inputs/profiles/character_profiles/paul_iusztin.md`
   - Loads writing style profiles (article, structure, mechanics, terminology, tonality)
   - Loads example articles from `inputs/examples/` for reference

2. **Generate Media Items**
   - Analyzes the article guideline for media requirements (e.g., "Generate mermaid diagram")
   - Uses specialized tools to create diagrams, charts, or other visual elements
   - Embeds media items into the article context

3. **Write Initial Draft**
   - Synthesizes all context (guidelines, research, profiles, examples)
   - Generates a comprehensive first draft following the outline
   - Applies the character's voice and writing style
   - Ensures all key topics from the guideline are covered

4. **Review and Revision Iterations**
   - **Review**: Analyzes the draft against guidelines, research, and style profiles
   - **Identify Issues**: Finds areas for improvement (clarity, depth, structure, style)
   - **Edit**: Generates a revised version addressing the identified issues
   - Repeats for N iterations (default: 2, configurable in `configs/course.yaml`)

5. **Save Final Article**
   - Saves the final article as `article.md`
   - Creates checkpoint files for each iteration (`article_000.md`, `article_001.md`, etc.)
   - Saves metadata including review scores and feedback

**Configuration:**

You can customize the workflow by modifying `configs/course.yaml`:

```yaml
num_reviews: 2  # Number of review iterations

nodes:
  write_article:
    model_id: "google_genai:gemini-2.5-pro"
    model_config:
      temperature: 0.7  # Higher for creative writing
  
  review_article:
    model_id: "google_genai:gemini-2.5-pro"
    model_config:
      temperature: 0.0  # Lower for analytical review
```

**Example Run:**

```bash
cd lessons/writing_workflow
make brown-generate-article DIR_PATH=inputs/tests/02_workflows_vs_agents
```

This will generate an article about "Workflows vs Agents" based on the guidelines and research in that directory.

### 7.2. Article Editing Workflow

The article editing workflow refines an existing article based on human feedback.

**Input Requirements:**
- `article.md`: The existing article to edit
- `article_guideline.md`: Original article requirements
- `research.md`: Research data for context
- Human feedback (provided as a command parameter)

**Workflow Steps:**

1. **Load Context**
   - Reads the existing article, guidelines, and research
   - Loads all style profiles and character profile
   - Parses the human feedback to understand edit requirements

2. **Review Article**
   - Analyzes the article against the human feedback
   - Identifies specific sections that need changes
   - Generates detailed review notes and recommendations

3. **Generate Edits**
   - Creates a comprehensive edited version of the article
   - Addresses all points in the human feedback
   - Maintains consistency with guidelines and research
   - Preserves the article structure and character voice

4. **Return Edited Article**
   - Returns the edited article content
   - The CLI script saves it back to `article.md`

**Example Run:**

```bash
cd lessons/writing_workflow
make brown-edit-article \
  DIR_PATH=inputs/tests/02_workflows_vs_agents \
  HUMAN_FEEDBACK="Add more concrete examples in section 3 and improve the conclusion"
```

### 7.3. Selected Text Editing Workflow

The selected text editing workflow makes targeted improvements to a specific section.

**Input Requirements:**
- `article.md`: The article containing the section to edit
- Line numbers defining the section boundaries
- Human feedback for the specific section

**Workflow Steps:**

1. **Extract Section**
   - Reads the full article for context
   - Extracts the specified text section by line numbers
   - Preserves surrounding context for coherence

2. **Review Section**
   - Analyzes the section in the context of the full article
   - Evaluates against the human feedback
   - Identifies specific improvements needed

3. **Edit Section**
   - Generates an improved version of just that section
   - Ensures consistency with the rest of the article
   - Maintains the same writing style and voice

4. **Apply Changes**
   - Returns the edited section with line number mappings
   - The CLI script replaces the original section in `article.md`

**Example Run:**

First, identify the line numbers of the section you want to edit by viewing `article.md`. Then:

```bash
cd lessons/writing_workflow
make brown-edit-selected-text \
  DIR_PATH=inputs/tests/02_workflows_vs_agents \
  HUMAN_FEEDBACK="Make this explanation clearer and add a code example" \
  FIRST_LINE=45 \
  LAST_LINE=67
```

---

## 8. Troubleshooting & Tips

### Common Issues

- **"Missing API key" or HTTP 401/403 errors**
  - Recheck your `.env` file and/or exported environment variables
  - Ensure `GOOGLE_API_KEY` or `OPENAI_API_KEY` is set correctly
  - Verify the key is valid and has not expired

- **"Config file not found" errors**
  - By default, Brown uses `configs/course.yaml`
  - To use a different config, set the `CONFIG_FILE` environment variable:
    ```bash
    CONFIG_FILE=configs/debug.yaml make brown-generate-article DIR_PATH=inputs/tests/00_debug
    ```

- **"File not found" errors for article_guideline.md or research.md**
  - Verify the `DIR_PATH` is correct and contains the required files
  - Use absolute paths if relative paths are not working
  - Check that file names match exactly (case-sensitive)

- **Articles are too short or missing content**
  - Ensure your `research.md` file contains sufficient context and data
  - Check that `article_guideline.md` has a clear outline and requirements
  - Consider increasing `num_reviews` in the config for more refinement

- **Model rate limits or timeouts**
  - Google's free tier has rate limits; wait a few minutes and retry
  - Consider using a paid API key for higher limits
  - Switch to `debug.yaml` config with fake models for testing workflows without API calls

---

## 9. Additional MCP Server Information

Brown can be used as a standalone **MCP server** that you can integrate with other MCP-compatible clients (Cursor, Claude Desktop, Zed, etc.).

### 9.1. Features

- **Article generation**: Complete workflow from research to final article
- **Article editing**: Full article refinement based on feedback
- **Selected text editing**: Targeted section improvements
- **AI integration**: Built-in support for Gemini and OpenAI models
- **Flexible transport**: Supports both **stdio** and **HTTP** transport protocols

### 9.2. Available Tools

When Brown is running as an MCP server, it exposes the following tools:

- **`generate_article`**: Generate an article from scratch using Brown's article generation workflow
  - Parameters: `dir_path` (path to directory with guidelines and research)
  - Returns: Success confirmation message
  - Side effect: Creates `article.md` in the specified directory

- **`edit_article`**: Edit an entire article based on human feedback
  - Parameters: `article_path` (path to article.md), `human_feedback` (edit instructions)
  - Returns: Edited article content with instructions

- **`edit_selected_text`**: Edit a selected section of an article
  - Parameters: `article_path`, `human_feedback`, `selected_text`, `first_line_number`, `last_line_number`
  - Returns: Edited section with line number mappings

### 9.3. Available Prompts

Brown provides MCP prompts that help trigger workflows:

- **`generate_article_prompt`**: Retrieves a prompt that will trigger the article generation workflow
- **`edit_article_prompt`**: Retrieves a prompt that will trigger the article editing workflow
- **`edit_selected_text_prompt`**: Retrieves a prompt that will trigger the selected text editing workflow

### 9.4. Available Resources

Brown exposes configuration and profile data as MCP resources:

- **`resource://config/app`**: Application configuration (models, file paths, workflow settings)
- **`resource://profiles/character`**: Character profile for consistent writing voice

### 9.5. Using Brown from Other MCP Clients

You can point other MCP-aware tools at Brown. Examples below assume the project lives at:

```bash
/absolute/path/to/course-ai-agents/lessons/writing_workflow
```

#### Cursor & Claude Desktop (stdio)

Add the following configuration to your `.cursor/mcp.json` or `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "brown": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/course-ai-agents/lessons/writing_workflow",
        "run",
        "fastmcp",
        "run",
        "src/brown/mcp/server.py:mcp",
        "--transport",
        "stdio"
      ],
      "env": {
        "GOOGLE_API_KEY": "your-google-api-key-here",
        "OPENAI_API_KEY": "your-openai-api-key-here"
      }
    }
  }
}
```

You can also leverage the `.env` file directly:

```json
{
  "mcpServers": {
    "brown": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/course-ai-agents/lessons/writing_workflow",
        "run",
        "fastmcp",
        "run",
        "src/brown/mcp/server.py:mcp",
        "--transport",
        "stdio"
      ],
      "cwd": "${workspaceFolder}",
      "env": {
        "ENV_FILE_PATH": "${workspaceFolder}/lessons/writing_workflow/.env"
      }
    }
  }
}
```

#### Using Brown in Cursor

Once configured, you can use Brown directly in Cursor's AI chat:

1. Start a new chat in Cursor
2. Enter `/generate_article_prompt` which will take care of calling the generate article tool with the right inputs.

---

## 10. Testing & QA

Brown includes a comprehensive test suite and code quality tools.

### Running Tests

Run the test suite using pytest:

```bash
cd lessons/writing_workflow
make tests
```

This runs all tests in the `tests/` directory using the debug configuration.

### Code Formatting and Linting

Brown uses **ruff** for code formatting and linting.

**Check code formatting:**

```bash
make format-check
```

**Auto-fix formatting issues:**

```bash
make format-fix
```

**Check for linting issues:**

```bash
make lint-check
```

**Auto-fix linting issues:**

```bash
make lint-fix
```

**Run all pre-commit hooks:**

```bash
make pre-commit
```

This runs all quality checks including formatting and linting.

---

## 11. Next Steps

Now that you have Brown set up, you can:

1. **Try the example workflows**: Run article generation on the test inputs in `inputs/tests/`
2. **Create your own articles**: Set up a new directory with your own `article_guideline.md` and `research.md`
3. **Customize the configuration**: Adjust models, review iterations, and other settings in `configs/course.yaml`
4. **Integrate with your IDE**: Set up Brown as an MCP server in Cursor or Claude Desktop
5. **Explore the evaluation framework**: Check out the `evals/` directory for quality assessment tools

For more information about the course and the broader AI agents framework, see the main repository `README.md`.
