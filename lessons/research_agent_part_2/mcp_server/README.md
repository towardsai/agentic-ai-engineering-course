# Nova Research MCP Server

This project implements an [MCP server](https://spec.modelcontextprotocol.io/) for research automation with powerful tools for content discovery, web scraping, and document processing.

## Features

- **Research Tools**: Automated web scraping, GitHub analysis, YouTube transcription, and more
- **AI Integration**: Built-in LLM support for intelligent content processing and research assistance
- **Content Processing**: Advanced document processing, URL extraction, and source curation
- **Flexible Transport**: Supports both stdio and HTTP transport protocols
- **Easy Configuration**: Simple setup with environment variables

## Installation

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- API keys for LLM providers (Google Gemini or OpenAI)

### 1. Environment Setup

Create a `.env` file in the project root with your API keys:

```bash
# Choose one of the following LLM providers
GOOGLE_API_KEY=your-google-api-key-here
# OR
OPENAI_API_KEY=your-openai-api-key-here

# Optional: Set logging level
TOOLS_LOG_LEVEL=INFO
```

### 2. Install Dependencies

```bash
cd mcp_server
uv sync
```

## Usage

### Transport Options

The Nova Research MCP Server supports two transport modes:

#### STDIO Transport (Default)
The default transport mode uses standard input/output for communication. This is the standard MCP transport used by most clients.

```bash
# Run with default stdio transport
uv run mcp-server

# Or explicitly specify stdio
uv run mcp-server --transport stdio
```

#### Streamable HTTP Transport
For web-based applications or clients that prefer HTTP communication:

```bash
# Run with Streamable HTTP transport on port 8000 (default)
uv run mcp-server --transport streamable-http

# Run on a custom port
uv run mcp-server --transport streamable-http --port 8080
```

### 3. Adding MCP config to your client

#### For Cursor & Claude Desktop

Add the following to your `.cursor/mcp.json` or `claude_desktop_config.json`:

**Option 1: Using stdio transport (recommended for local development)**
```json
{
  "mcpServers": {
    "nova-research": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/your/mcp_server",
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

**Option 2: Using HTTP transport (for web applications)**
```json
{
  "mcpServers": {
    "nova-research": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/your/mcp_server",
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

#### For Zed

Add the following to your `settings.json`:

```json
{
  "context_servers": {
    "nova-research": {
      "command": {
        "path": "uv",
        "args": [
          "--directory",
          "/path/to/your/mcp_server",
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
      },
      "settings": {}
    }
  }
}
```

#### Using Docker

You can also run the server using Docker:

```bash
# Build the Docker image
docker build -t nova-research-mcp .

# Run with stdio transport
docker run --rm -i \
  -e GOOGLE_API_KEY=your-google-api-key-here \
  -e OPENAI_API_KEY=your-openai-api-key-here \
  -e TOOLS_LOG_LEVEL=INFO \
  nova-research-mcp --transport stdio

# Run with HTTP transport
docker run --rm -i \
  -p 8000:8000 \
  -e GOOGLE_API_KEY=your-google-api-key-here \
  -e OPENAI_API_KEY=your-openai-api-key-here \
  -e TOOLS_LOG_LEVEL=INFO \
  nova-research-mcp --transport streamable-http --port 8000
```

### 4. Testing the Connection

To test that your MCP server is working correctly:

```bash
# Test stdio transport
echo '{"jsonrpc": "2.0", "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}}, "id": 1}' | uv run mcp-server --transport stdio

# Test HTTP transport
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "mcp-session-id: test-session" \
  -d '{"jsonrpc": "2.0", "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}}, "id": 1}'
```

## Available Tools

The Nova Research MCP Server provides the following research tools:

### Content Discovery & Processing
- **create_research_file_tool**: Create research files with proper structure
- **extract_guidelines_urls_tool**: Extract URLs from research guidelines
- **process_github_urls_tool**: Analyze and process GitHub repositories
- **process_local_files_tool**: Process local documents and files
- **scrape_research_urls_tool**: Scrape web content for research
- **scrape_and_clean_other_urls_tool**: Clean and process web content
- **transcribe_youtube_videos_tool**: Transcribe YouTube videos for research

### AI-Powered Analysis
- **run_perplexity_research_tool**: Run AI-powered research queries
- **generate_next_queries_tool**: Generate follow-up research questions

### Content Curation
- **select_research_sources_to_scrape_tool**: Select sources for scraping
- **select_research_sources_to_keep_tool**: Curate final research sources

## Available Resources

- **get_available_tools_resource**: List all available research tools
- **get_memory_usage_resource**: Monitor server memory usage
- **get_server_config_resource**: Get server configuration
- **get_system_status_resource**: Get system status information

## Examples

Here are some example prompts you can use with the MCP server:

1. **Research Planning**
   ```
   Create a research folder structure for "AI in Healthcare" and extract relevant URLs from the guidelines file at /path/to/guidelines.md
   ```

2. **Content Discovery**
   ```
   Analyze the GitHub repository https://github.com/user/repo and generate a summary report
   ```

3. **Web Research**
   ```
   Scrape the following research sources: https://example.com/paper1, https://example.com/paper2 and clean the content
   ```

4. **AI Analysis**
   ```
   Use Perplexity to research the latest developments in quantum computing and generate follow-up questions
   ```

5. **YouTube Research**
   ```
   Transcribe the YouTube video https://youtube.com/watch?v=... for research analysis
   ```

## Development

### Building

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run linter
uv run ruff check .
```

### Project Structure

```
mcp_server/
├── src/
│   ├── config/          # Server configuration
│   ├── models/          # Data models and schemas
│   ├── prompts/         # MCP prompts
│   ├── resources/       # MCP resources
│   ├── routers/         # MCP endpoints
│   ├── tools/           # Research tools
│   ├── utils/           # Utility functions
│   └── server.py        # Main server entry point
├── tests/               # Test files
├── pyproject.toml       # Project configuration
└── README.md           # This file
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `uv run pytest`
6. Submit a pull request

## License

Add your license information here.

## Support

For support and questions:
- Open an issue on GitHub
- Check the documentation
- Review the example configurations in `.mcp.json.sample`
