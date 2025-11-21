## Nova Research Agent

This folder contains a full research agent implemented as two projects:

- **`mcp_server/`**: Nova Research MCP server exposing research tools (scraping, Perplexity, Firecrawl, etc.)
- **`mcp_client/`**: Interactive MCP client you run in the terminal to control the agent

This README shows how to:

- Install both projects with `uv`
- Configure API keys
- Launch the research agent from the terminal
- Run a **full research workflow**

> [!NOTE]
> This will be run as an independent project from the rest of this repository.

---

## 1. Directory Layout

From the repository root (`course-ai-agents`):

```bash
lessons/research_agent_part_2/
  ├── data/
  │   └── research_function_calling/
  │       └── article_guideline.md      # An example article guideline file to use for the research agent
  ├── mcp_server/      # MCP server (tools, prompts, resources)
  └── mcp_client/      # Interactive terminal client
```

You will install and run `mcp_server` and `mcp_client` as **separate `uv` projects**.

---

## 2. Prerequisites

- **Python**: The projects are pinned to Python `3.12.11` in their `pyproject.toml`.  
  If you followed the main course `README.md`, you already have a compatible Python installed.
- **`uv`** package manager: see the root `README.md` for installation instructions.
- All the commands below assume you are using a MacOS / Linux shell with commands based on Bash/zsh. In case you are using Windows, we assume to use a shell compatible with Linux such as WSL.

You do **not** need to manually create or activate virtual environments; `uv` will manage them per-project.

---

## 3. Install Dependencies with `uv`

You will install dependencies **once per project**.

### 3.1. Install server dependencies (`mcp_server`)

From the repository root:

```bash
cd lessons/research_agent_part_2/mcp_server
uv sync
```

This will create a `.venv` inside `mcp_server` and install all server dependencies.

### 3.2. Install client dependencies (`mcp_client`)

In a separate step:

```bash
cd lessons/research_agent_part_2/mcp_client
uv sync
```

This will create a `.venv` inside `mcp_client` and install all client dependencies.

---

## 4. Configure Environment Variables (API Keys)

Both the **server** and **client** read configuration from:

- Real environment variables (`export VAR=...`), **and/or**
- A local `.env` file in their project directory using the `.env.example` as an example.

For a smooth setup, create **two `.env` files** (you can copy the same content to both):

1. `lessons/research_agent_part_2/mcp_server/.env`
2. `lessons/research_agent_part_2/mcp_client/.env`

You can use `.env.example` as a template: `cp .env.example .env`

### 4.1. Minimum required variables

At minimum, you should set:

```bash
GOOGLE_API_KEY=your-google-api-key-here     # used by Gemini models

# Web research & scraping
PPLX_API_KEY=your-perplexity-api-key-here   # required for Perplexity research
FIRECRAWL_API_KEY=your-firecrawl-api-key-here  # required for web scraping

# Optional: GitHub analysis
GITHUB_TOKEN=your-github-token-here

# Optional: Opik monitoring
OPIK_API_KEY=your-opik-api-key-here
```

**Where to get these keys (all have free tiers):**

- **`GOOGLE_API_KEY` (Gemini)**: Create a key in Google AI Studio (`https://aistudio.google.com/app/apikey`).  
  Google offers a free tier suitable for experimentation and this course.
- **`PPLX_API_KEY` (Perplexity)**: Create a key from the Perplexity settings page (`https://www.perplexity.ai/settings/api`).  
  Perplexity currently offers a paid plan with generous usage; see their docs for any trial limits.
- **`FIRECRAWL_API_KEY`**: Create a key at `https://firecrawl.dev/`.  
  Firecrawl has a free tier that allows you to scrape **around 500 pages**.
- **`GITHUB_TOKEN`**: Create a fine‑grained personal access token from your GitHub settings, with read‑only access to the repositories you want to analyze.
- **`OPIK_API_KEY`**: Create a free account at `https://www.comet.com/site/products/opik/` and find the API KEY based on this [doc](https://www.comet.com/docs/opik/faq#where-can-i-find-my-opik-api-key-)

You can either:

- Put this content in `.env` files in **both** `mcp_server` and `mcp_client`, or
- Export them once in your shell before running anything:

```bash
export GOOGLE_API_KEY=...
export PPLX_API_KEY=...
export FIRECRAWL_API_KEY=...
export OPENAI_API_KEY=...
export GITHUB_TOKEN=...
```

> **Tip:** If something fails with “missing API key” errors, first verify the variables above are set and visible in the shell where you run `uv`.

---

## 5. Running the Research Agent from the Terminal

The **recommended** way to run the research agent locally is:

- Use the **interactive MCP client** in `mcp_client`
- Use the **in‑memory transport** (the MCP server runs in the same process)

You can also run the server as a separate stdio process if you prefer.

### 5.1. Quick start: in‑memory transport (recommended)

From the repository root:

```bash
cd lessons/research_agent_part_2/mcp_client
uv run python -m src.client
```

This will:

- Start the MCP client in your terminal
- Import and run the MCP server from `mcp_server` **in the same process**
- Print available tools, resources, and prompts
- Drop you into an interactive prompt like:

```text
👤 You:
```

You can now type commands and natural language instructions to drive the research workflow.

### 5.2. Alternative: stdio transport (server as separate process)

If you prefer to run the MCP server in its own process:

**Terminal 1 – start the server (stdio):**

```bash
cd lessons/research_agent_part_2/mcp_server
uv run mcp-server --transport stdio
```

**Terminal 2 – start the client (stdio):**

```bash
cd lessons/research_agent_part_2/mcp_client
uv run python -m src.client --transport stdio
```

The client will connect to the server via stdio using the configuration embedded in `src/client.py`.

---

## 6. Running Full Research for `research_function_calling`

Once the client is running (either mode), you can use it to run the **complete research workflow** for the guideline in:

```bash
lessons/research_agent_part_2/data/research_function_calling/article_guideline.md
```

### 6.1. Load the full research workflow prompt

At the `👤 You:` prompt in the MCP client, type:

```text
/prompt/full_research_instructions_prompt
```

This loads the built‑in instructions that describe the full multi‑step workflow (URL extraction, scraping, Perplexity research loop, source selection, final research file creation, etc.).

### 6.2. Tell the agent which research folder to use

After the prompt is loaded, send a natural‑language instruction specifying the **absolute path** to the research directory and that you want the full workflow to run.

To get the absolute path from the terminal:

```bash
cd path/to/course-ai-agents/lessons/research_agent_part_2/data/research_function_calling
pwd
```

Copy the output of `pwd` and use it in your message. For example:

```text
The research folder is /absolute/path/from/pwd. Run the complete workflow from start to finish.
```

The agent will then:

1. Read `article_guideline.md` in that folder
2. Extract and process any guideline URLs (web, GitHub, YouTube)
3. Run multiple rounds of Perplexity‑powered research
4. Filter and select the best sources
5. Scrape selected sources in depth
6. Compile everything into a final `research.md` file

### 6.3. Inspecting the final research file

When the agent reports that the workflow is complete, you should find:

```bash
lessons/research_agent_part_2/data/research_function_calling/research.md
```

Open `research.md` in your editor to review the collected research.  
It is organized into collapsible sections (Perplexity results, scraped guideline sources, additional research sources, etc.) and is designed to be the input to the writing agents used later in the course.

---

## 7. Troubleshooting & Tips

- **Missing tools or prompts shown in the client**
  - Ensure you started the client from `lessons/research_agent_part_2/mcp_client` with `uv run python -m src.client`.
- **“API key missing” or HTTP 401/403 errors**
  - Recheck your `.env` files and/or exported environment variables for `GOOGLE_API_KEY`, `PPLX_API_KEY`, and `FIRECRAWL_API_KEY`.
- **Perplexity or Firecrawl rate limits**
  - If the workflow times out or fails during web research or scraping, try again later or reduce usage while testing.
- **Switching between in‑memory and stdio**
  - If debugging code inside `mcp_server`, in‑memory mode is often simpler.
  - If you want clear separation between client and server processes, use stdio mode as shown above.

With this setup, you can iterate on the research agent implementation and run realistic, end‑to‑end research workflows entirely from your terminal.

---

## 8. Additional MCP Server Information

The `mcp_server` project implements a standalone **MCP server** that you can also use from other MCP‑compatible clients (Cursor, Claude Desktop, Zed, etc.).

### 8.1. Features

- **Research tools**: automated web scraping, GitHub analysis, YouTube transcription, and more.
- **AI integration**: built‑in LLM support for intelligent content processing and research assistance.
- **Content processing**: advanced document processing, URL extraction, and source curation.
- **Flexible transport**: supports both **stdio** and **HTTP** transport protocols.

### 8.2. Using the server from other MCP clients

You can point other MCP‑aware tools at this server. Examples below assume the server lives at:

```bash
/absolute/path/to/course-ai-agents/lessons/research_agent_part_2/mcp_server
```

#### Cursor & Claude Desktop (stdio)

Add the following configuration to your `.cursor/mcp.json` or `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "nova-research": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/course-ai-agents/lessons/research_agent_part_2/mcp_server",
        "run",
        "mcp-server",
        "--transport",
        "stdio"
      ],
      "env": {
        "TOOLS_LOG_LEVEL": "INFO",
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
        "nova": {
            "command": "uv",
            "args": [
                "--directory",
                "/absolute/path/to/course-ai-agents/lessons/research_agent_part_2/mcp_server",
                "run",
                "-m",
                "src.server",
                "--transport",
                "stdio"
            ],
            "cwd": "${workspaceFolder}",
            "env": {
                "ENV_FILE_PATH": "${workspaceFolder}/lessons/research_agent_part_2/mcp_server/.env"
            }
        }
    }
}
```

#### Cursor & Claude Desktop (HTTP)

```json
{
  "mcpServers": {
    "nova-research": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/course-ai-agents/lessons/research_agent_part_2/mcp_server",
        "run",
        "mcp-server",
        "--transport",
        "streamable-http",
        "--port",
        "8000"
      ],
      "env": {
        "TOOLS_LOG_LEVEL": "INFO",
        "GOOGLE_API_KEY": "your-google-api-key-here",
        "OPENAI_API_KEY": "your-openai-api-key-here"
      }
    }
  }
}
```

### 8.3. Available tools and resources

- **Content discovery & processing**
  - `create_research_file_tool`
  - `extract_guidelines_urls_tool`
  - `process_github_urls_tool`
  - `process_local_files_tool`
  - `scrape_research_urls_tool`
  - `scrape_and_clean_other_urls_tool`
  - `transcribe_youtube_videos_tool`
- **AI‑powered analysis**
  - `run_perplexity_research_tool`
  - `generate_next_queries_tool`
- **Content curation**
  - `select_research_sources_to_scrape_tool`
  - `select_research_sources_to_keep_tool`
- **Resources**
  - `get_available_tools_resource`
  - `get_memory_usage_resource`
  - `get_server_config_resource`
  - `get_system_status_resource`
