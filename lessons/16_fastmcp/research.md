# Research

## Research Results

<details>
<summary>What are the best practices for structuring a FastMCP server application in Python, specifically regarding routers for tools, resources, and prompts?</summary>

### Source [1]: https://gofastmcp.com/getting-started/welcome

Query: What are the best practices for structuring a FastMCP server application in Python, specifically regarding routers for tools, resources, and prompts?

Answer: The FastMCP official documentation explains that a **server** is responsible for exposing Python functions as MCP-compliant **tools**, **resources**, and **prompts**.[1] It emphasizes that these three primitives are the core pillars, and a well-structured server organizes its logic around them rather than mixing concerns arbitrarily.[1]

According to the docs, FastMCP is designed so that you wrap existing Python functions into these primitives, instead of tightly coupling business logic with protocol handling.[1] This implies a structural best practice: keep your domain logic in normal Python modules and use FastMCP decorators or configuration only at the boundary where you expose that logic as tools, resources, or prompts.[1]

The documentation also highlights that **servers**, **clients**, and **apps** are separate concepts: servers expose capabilities, clients connect to them, and apps provide UIs.[1] For structuring a server application, this separation encourages keeping MCP server code self-contained, with a clear entrypoint (typically a `server.py` or similar) that instantiates the FastMCP server and registers the different primitives.[1]

The site notes that any page in the docs can be accessed as markdown via a `.md` suffix, which is intended to integrate with MCP itself.[1] While not a code-structure rule, this reinforces the architectural idea that the server should present clearly defined resources and prompts that map cleanly to the external interface. In practice, this suggests grouping related resources (e.g., documentation-like content) under coherent URIs and prompt definitions rather than scattering them across unrelated modules.[1]

Overall, the official overview points toward a design where tools, resources, and prompts are first-class, explicitly registered components of the server, backed by cleanly separated application code.[1]

-----

-----

### Source [2]: https://github.com/prefecthq/fastmcp

Query: What are the best practices for structuring a FastMCP server application in Python, specifically regarding routers for tools, resources, and prompts?

Answer: The FastMCP README states that FastMCP is "the fast, Pythonic way to build MCP servers" and that **servers wrap your Python functions into MCP-compliant tools, resources, and prompts**.[7] This framing implies a best-practice structure: treat your FastMCP server as an adapter layer that exposes existing Python code rather than embedding all logic inside decorators.[7]

The repository describes three pillars:
- **Tools** for executable functions.
- **Resources** for data you expose.
- **Prompts** for reusable, structured prompt templates.[7]

A typical structural pattern implied by the examples is:
- Create a central module (often something like `server.py` or `main.py`) that instantiates `FastMCP` and serves as the MCP entrypoint.
- Place domain logic and data access in separate modules or packages (for example, `services/`, `models/`, or `lib/`).
- In the server module, import those functions and expose them as tools/resources/prompts using the FastMCP APIs (e.g., decorators or helper constructors).[7]

The README’s positioning that servers “wrap” functions further suggests that your **routers** (or registration layer) should be thin: they bind names, descriptions, and parameter schemas to underlying Python callables, but do not themselves implement business logic.[7]

By distinguishing **servers**, **clients**, and **apps**, the project encourages a clear boundary for MCP-specific code: the FastMCP server module is responsible for registering tools/resources/prompts, while other Python modules remain protocol-agnostic.[7] This model naturally extends to organizing separate submodules or "routers" for groups of tools (e.g., `tools/files.py`, `tools/users.py`) and then registering them centrally, mirroring common patterns from web frameworks, although the repository does not prescribe a specific router API.[7]

-----

-----

### Source [4]: https://circleci.com/blog/building-and-deploying-a-python-mcp-server-with-fastmcp/

Query: What are the best practices for structuring a FastMCP server application in Python, specifically regarding routers for tools, resources, and prompts?

Answer: The CircleCI tutorial on building and deploying a Python MCP server with FastMCP presents a concrete directory layout and usage of FastMCP that imply best practices for structuring a server.[2]

It first recommends creating a **project directory structure** with a dedicated package and server module:
- `src/document_brain/`
- `src/document_brain/__init__.py`
- `src/document_brain/server.py`[2]

This layout separates the MCP server from the rest of the code and encourages packaging the server as a Python distribution. The article explains that MCP servers consist of three key components:
- **Tools** for model-controlled actions.
- **Resources** for application-controlled data sources.
- **Prompts** for user-controlled templates to structure interaction.[2]

Within `server.py`, the tutorial defines tools, resources, and prompts via FastMCP and then runs the server, which matches a pattern where one module serves as the **central router/entrypoint** for all primitives.[2] While it does not introduce a router object by name, the structure effectively routes requests: FastMCP uses the decorated functions in `server.py` as the implementation of each tool/resource/prompt.[2]

The tutorial also shows how to test the MCP server using the FastMCP Inspector, where tools, resources, and prompts appear in separate tabs.[2] This reinforces the idea that server structure should map cleanly to these categories, making it easier to reason about and debug.

Finally, it demonstrates packaging and distribution steps (build with `python -m build`, install via `pip`, and run entrypoint like `mcp-document-brain`).[2] This supports a best practice of organizing your server as an installable package with a clear CLI entrypoint that internally instantiates the FastMCP server and registers all tools/resources/prompts in a predictable module.[2]

-----

-----

### Source [5]: https://www.firecrawl.dev/blog/fastmcp-tutorial-building-mcp-servers-python

Query: What are the best practices for structuring a FastMCP server application in Python, specifically regarding routers for tools, resources, and prompts?

Answer: Firecrawl’s FastMCP tutorial provides a step-by-step example of building a Python MCP server and indicates structural patterns for tools, resources, and prompts.[8]

The tutorial shows defining tools as regular Python functions decorated with FastMCP’s tooling API, keeping each tool focused on a single task (e.g., web crawling, scraping, or processing.[8]) This demonstrates a practice of **thin wrappers**: tools primarily validate/forward parameters to underlying service functions.

For **resources**, the tutorial illustrates exposing data such as crawl results or configuration via resource definitions, separate from tools that perform actions.[8] This separation ensures that read-only or contextual data access is not conflated with operations that trigger side effects. Resources are organized with clear URIs and return predictable data structures.[8]

The article also introduces **prompts** as structured templates that orchestrate how tools and resources are used by the model.[8] Prompts reference tools indirectly by describing how the model should call them, and the tutorial keeps prompt definitions alongside but conceptually separate from the computation and I/O logic.[8]

The project is structured so that:
- There is a main server file that initializes the FastMCP server.
- Domain-specific logic (e.g., crawling, parsing) resides in separate modules.
- Tools/resources/prompts are registered via decorators in the server or closely-related modules.[8]

While the article does not define a formal router abstraction, the pattern is similar: the FastMCP server module plays the role of a router, mapping tool/resource/prompt names to specific Python callables. The tutorial thus suggests organizing code so that routing/registration is centralized, and implementation logic lives in decoupled modules, which is conducive to scaling a larger MCP server.[8]

-----

</details>

<details>
<summary>What is the design rationale for hosting workflow instructions as server-side 'MCP Prompts' versus embedding them in the client?</summary>

### Source [12]: https://dev.to/aws-heroes/mcp-prompts-and-resources-the-primitives-youre-not-using-3oo1

Query: What is the design rationale for hosting workflow instructions as server-side 'MCP Prompts' versus embedding them in the client?

Answer: This article explicitly addresses why prompts should live on the MCP server rather than be hard‑coded on the client.

It explains that **MCP prompts move workflow state and structure out of the LLM’s transient context and into explicit, server‑side execution**. This reduces the LLM’s decision space because the server exposes well‑defined, reusable workflows instead of forcing the model to infer the entire interaction pattern from a generic client‑side prompt every time. By narrowing the decision space with server‑defined prompts, behavior becomes more predictable and consistent across different clients.

The article notes that prompts and resources are **server‑side primitives**, in contrast to tools which execute logic, and that this design allows the server to encapsulate both **workflow templates and domain knowledge** close to the systems they operate on. Implementing prompts on the server lets the same workflows be reused by multiple MCP‑aware clients without duplicating or re‑implementing instructions in each client.

A key rationale given is that **workflow state management is shifted from the LLM’s volatile token context into durable server‑side constructs**. Instead of relying on the model to remember prior instructions or implicit workflow rules embedded in a long client prompt, the server exposes prompts as explicit, structured templates that can be invoked deterministically. This makes workflows easier to version, test, and evolve in one place (the server), while clients simply discover and call them.

Overall, the article frames server‑side MCP prompts as a way to centralize workflow definition, reduce prompt brittleness in clients, and improve reliability and reuse of workflows across tools and applications.

-----

-----

### Source [13]: https://www.speakeasy.com/mcp/core-concepts/prompts

Query: What is the design rationale for hosting workflow instructions as server-side 'MCP Prompts' versus embedding them in the client?

Answer: The Speakeasy documentation defines **MCP prompts** as reusable, structured message templates that are **exposed by MCP servers** rather than embedded in clients. It highlights that prompts are returned by the server via `prompts/list` and `prompts/get`, so clients discover them dynamically instead of hard‑coding their contents.

The docs emphasize that prompts are **declarative and composable**, intended primarily for **user‑initiated workflows** such as slash commands, quick actions, and task‑specific flows (e.g., summarization or code explanation). Because prompts are server‑defined, UI clients can present a catalog of available workflows and arguments, without needing to encode the workflow instructions themselves.

Each prompt is a **named, parameterized template** with a name, optional description, and a structured list of arguments. These arguments are validated and interpreted server‑side. The server responds to `prompts/get` with a predefined list of messages that initiate consistent model behavior. This structured, server‑side representation ensures that multiple clients invoking the same prompt obtain the same initial workflow instructions, which would be harder to guarantee if each client embedded its own variant.

The documentation distinguishes prompts from tools and resources: tools execute logic and resources provide data, while prompts **only return messages**, not logic or data. Hosting prompts on the server enforces this separation of concerns and keeps workflow definition near the systems and data the server already manages.

By centralizing prompts on the server, changes to workflows (new arguments, modified instructions, improved templates) can be rolled out once at the server level and immediately discovered by all MCP‑aware clients via the standardized prompt listing and retrieval APIs, instead of requiring client updates.

-----

</details>

<details>
<summary>Can you provide a code example of an MCP client discovering and then calling a specific tool from a FastMCP server using the in-memory transport for testing?</summary>

### Source [18]: https://gofastmcp.com/clients/client

Query: Can you provide a code example of an MCP client discovering and then calling a specific tool from a FastMCP server using the in-memory transport for testing?

Answer: This official FastMCP client documentation shows exactly how to use the **in‑memory transport** to connect a Python client directly to a FastMCP server instance for testing.[6]

It explains that the **in‑memory transport** "connects directly to a FastMCP server instance within the same Python process" and is intended for **testing and development** scenarios where you do not want to deal with stdio, HTTP/SSE, or WebSocket plumbing.[6]

The page documents that you use the FastMCP **Client** class together with an **InMemoryTransport** (or equivalent helper) and a server instance created with the FastMCP server API.[6] The typical high‑level flow is:

1. **Create the FastMCP server** instance (e.g., via the `FastMCP`/`Server` class and decorated tools).
2. **Construct an in‑memory transport**, wiring it to that server instance so client requests are routed directly in process.[6]
3. **Create a Client** object, passing the in‑memory transport into the client constructor.[6]
4. **Open an MCP session** from the client; during this, the client performs the MCP handshake and **discovers tools** from the server (i.e., it calls the protocol’s `capabilities` and `tools/list` methods so you can see available tools and their schemas).[6]
5. Once the session is open, you can **inspect the discovered tools** on the client side and then **call a specific tool** by name with appropriate JSON‑serializable arguments.[6]

While this page does not give a full standalone code listing, it establishes the pattern that a test client using in‑memory transport will: create the server object; create an in‑memory transport bound to that server; create a `Client` with that transport; then use the client’s session object to discover and invoke tools by name (e.g., `session.call_tool("tool_name", args={...})`).[6] This is the exact pattern you would follow to build a test that discovers and then calls a specific tool from a FastMCP server using in‑memory transport.

-----

</details>

<details>
<summary>How does decoupling orchestration (in an MCP client) from tool execution (in an MCP server) improve token efficiency and context management for LLM-based agents?</summary>

### Source [22]: https://www.anthropic.com/engineering/code-execution-with-mcp

Query: How does decoupling orchestration (in an MCP client) from tool execution (in an MCP server) improve token efficiency and context management for LLM-based agents?

Answer: This source explains that traditional MCP setups tightly couple **orchestration** (the LLM deciding which tools to use and in what order) with **tool execution** (the MCP server running those tools) by exposing every tool directly in the model’s prompt as a callable function.[1] This means all tool schemas and most intermediate results must live inside the model’s context window, which inflates token usage and constrains how many tools or how much state the agent can handle efficiently.[1]

Decoupling orchestration from execution via **code execution with MCP** changes the pattern: the model runs in a code execution environment and treats MCP servers as **code APIs** rather than as direct tools.[1] The LLM’s role is to write code (e.g., Python) that calls MCP tools through a small API; the actual calls, data routing, and intermediate state live in the execution environment and MCP server, *outside* the LLM context.[1]

For **token efficiency**, this has two main effects:
- The agent can **load tool definitions on demand** instead of including all schemas upfront in the prompt, so only the tools needed for the current subtask are fetched and described.[1] In Anthropic’s example, this reduced tool-definition tokens from ~150,000 to ~2,000 (about **98.7%** savings).[1]
- Intermediate results stay in the execution environment by default; only selected outputs or summaries are logged back to the model, so large data flows between tools never consume context tokens.[1]

For **context management**, this decoupling means:
- The execution environment maintains **state and intermediate variables** across steps without re-serializing them into the prompt on every call.[1]
- The model’s context is reserved for **high-level plans, code, and concise outputs**, rather than schemas and raw tool I/O.[1]
- Sensitive or bulky data can transit through tools without ever entering model context, improving both privacy and context budget usage.[1]

Anthropic frames this as applying standard software patterns (APIs, local variables, control flow) to agents so that MCP servers scale to many tools while keeping the LLM context compact and focused.[1]

-----

</details>


## Sources Scraped From Research Results

<details>
<summary>{'type': 'text', 'text': 'Extending Large Language Models (LLMs) with custom tools has become increasingly valuable in today’s AI landscape. Model Context Protocol (MCP) servers provide a standardized way to connect external tools and resources to LLMs. This can enhance their capabilities beyond basic text generation.\n\nWhile thousands of pre-built MCP servers exist, creating your own allows you to address specific workflows. You can implement use cases that off-the-shelf solutions cannot handle. This is where the real power lies.\n\nIn this tutorial, you will learn to build a document parsing server that enables MCP hosts to understand various file formats. You will use FastMCP—the leading library for building MCPs in Python. You will also bundle your application into a Python package and publish on PyPI using an automated CI/CD pipeline.\n\nBy the end, you will have a fully functional document reader MCP server that extracts text from various document formats.\n\nHere is a demo of what you will build.\n\nhttps://images.ctfassets.net/il1yandlcjgk/2Hc7rEy3yFEqreSD7aUVsz/d520643311bfe644198f211075cd6042/2025-05-13-mcp-document-brain-demo.gif?w=1003&fm=webp&q=60\n\n## Prerequisites\n\nBefore diving in, make sure you have:\n\n- [Python 3.10+](https://www.python.org/downloads/) or newer installed on your system\n- Basic understanding of [Python packaging concepts](https://packaging.python.org/en/latest/tutorials/packaging-projects/)\n- A [PyPI account](https://pypi.org/account/register/) for publishing your package\n- A [GitHub account](https://github.com/signup) for version control\n- A [CircleCI account](https://app.circleci.com/signup) connected to your GitHub account for creating CI/CD pipelines\n- [Claude Desktop](https://claude.ai/download) for using your MCP server\n\n## Setting up the development environment\n\nBuilding an MCP server opens up exciting possibilities for extending LLM capabilities with custom tools and resources. FastMCP simplifies this process with a clean API using decorators that handle server protocol complexities.\n\nYou can visit this [GitHub repository](https://github.com/ArmstrongA/document-brain.git) to explore the code for the server you are about to build.\n\n### Initializing the project directory\n\nFirst, let us create a new directory for our project and set up the basic structure:\n\n```bash\nmkdir document-brain\ncd document-brain\n```\n\n### Installing `UV`\n\nFor dependency management, you will use `uv`—a modern, fast package manager for Python. Open powershell and run this command to install `uv`on Windows:\n\n```bash\npowershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"\n```\n\nRestart your terminal completely. Then verify the installation:\n\n```bash\nuv --version\n```\n\nYou can also install `uv` using pip:\n\n```bash\n# Using pip\npip install uv\n\nOn Unix-based systems like macOS run:\n\n```bash\ncurl -sSf https://install.python-uv.org | bash\n# Or using Homebrew\nbrew install uv\n```\n\nNow initialize your project with `uv`:\n\n```bash\nuv init\n```\n\nThis command automatically creates:\n\n- **`.python-version`**: Ensures consistent Python versions across environments\n- **`.gitignore`**: Lists files and directories for Git to ignore.\n- **`main.py`**: A starting point for development.\n- **`pyproject.toml`**: Defines project metadata and dependencies.\n- **`README.md`**: An overview of the project’s purpose, usage instructions, and other relevant information.\n\nThese files collectively set up a [foundational structure for a Python project](https://www.reddit.com/r/learnpython/comments/1jbo88t/uv_for_python_project_and_version_management/). This facilitates development and collaboration.\n\n### Creating a virtual environment and installing dependencies\n\nWhen installing packages and dependencies, `uv` automatically creates a virtual environment name `.venv`.\n\nInstall the necessary packages for developing and testing your MCP server. Run:\n\n```bash\nuv add "mcp[cli]"\nuv add "markitdown[all]"\nuv add --dev pytest build twine\n```\n\nThese packages provide:\n\n- FastMCP with CLI tools for development and debugging\n- Markitdown for document parsing functionality\n- Testing (pytest)\n- Building your package (build)\n- Publishing to PyPI (twine)\n\nBefore creating your MCP server, create the project directory structure:\n\n```bash\nmkdir -p src/document_brain\ntouch src/document_brain/__init__.py\ntouch src/document_brain/server.py\n```\n\nYour project structure looks like this:\n\n```text\nproject_root/\n├── src/\n│   └── document_brain/\n│       ├── __init__.py        # Initializes server\n│       └── server.py          # Contains the \'mcp\' instance and definitions\n├── tests/\n│   ├── __init__.py            # Makes tests discoverable\n│   └── test_server.py         # Tests for your MCP server\n├── main.py\n├── pyproject.toml             # Project configuration\n└── README.md                  # Project documentation\n```\n\nThis structure follows the recommended Python packaging standards with src-layout. The approach prevents import issues during development.\n\n## Building the MCP Server with FastMCP\n\nIn this part, you will begin by understanding three primary components of MCP servers (tools, resources, and prompts).\n\n### Understanding the core components\n\nMCP servers consists of three key components:\n\n- **Tools:** Model-controlled functions that LLMs can call to perform actions or interact with external systems.\n- **Resources:** Application-controlled data sources that inject contextual information from your systems into the conversation.\n- **Prompts:** User-controlled templates that can be invoked through UI elements to help users interact with the LLM in structured ways.\n\nNow, let us implement your document brain server in `src/document_brain/server.py`:\n\n```python\nfrom mcp.server.fastmcp import FastMCP\nfrom mcp.server.fastmcp.prompts import base\nfrom mcp.server.fastmcp.resources import DirectoryResource\nfrom pathlib import Path\nimport os\nfrom markitdown import MarkItDown\n\nmd = MarkItDown()\n\n# Initialize the FastMCP server\nmcp = FastMCP("DocumentBrain", dependencies=["markitdown[all]"])\n\n@mcp.tool(\n    annotations={\n        "title": "Read Any Document",\n        "readOnlyHint": True,\n        "openWorldHint": False\n    }\n)\ndef read_any_document(file_path: str) -> str:\n    """Read any supported document and return its text content, including OCR for images.\n    Args:\n        file_path: Path to the document to process.\n    Returns:\n        Extracted text content as a string.\n    """\n    try:\n        expanded_path = os.path.expanduser(file_path)\n        return md.convert(expanded_path).text_content\n    except Exception as e:\n        return f"Error reading file: {str(e)}"\n\n@mcp.tool(\n    annotations={\n        "title": "Save File to PC",\n        "readOnlyHint": False,\n        "openWorldHint": True\n    }\n)\ndef save_file_to_pc(filepath: str, content: str) -> str:\n    """\n    Save content to a file on the desktop.\n    Args:\n        filename: Name of the file to save (can include subdirectory)\n        content: Content to write to the file\n    Returns:\n        A success or error message\n    """\n    try:\n        # Expand the desktop path\n        desktop_path = os.path.expanduser(filepath)\n        # Ensure the filename doesn\'t contain any path traversal attempts\n        safe_filename = os.path.basename(filepath)\n        # Create the full file path\n        full_path = os.path.join(desktop_path, safe_filename)\n        # Ensure the directory exists\n        os.makedirs(os.path.dirname(full_path), exist_ok=True)\n        # Write the content to the file\n        with open(full_path, \'w\', encoding=\'utf-8\') as f:\n            f.write(content)\n        return f"File successfully saved to {full_path}"\n    except Exception as e:\n        return f"Error saving file: {str(e)}"\n\n# Now add a resource\n# Define the path to the current directory\ndocuments_path = Path(".").resolve()\n\n# Create a DirectoryResource to list files in the current directory\ndocuments_resource = DirectoryResource(\n    uri="docs://files",\n    path=documents_path,\n    name="Local Document Directory",\n    description="Lists all files in the current working directory.",\n    recursive=False  # Set to True if you want to include subdirectories\n)\n\n# Add the resource to your FastMCP server\nmcp.add_resource(documents_resource)\n\n@mcp.resource("docs://file/{filename}")\ndef get_document_content(filename: str) -> str:\n    """Retrieve the content of a specified document."""\n    try:\n        file_path = documents_path / filename\n        if not file_path.exists():\n            return f"File not found: {filename}"\n        return md.convert(str(file_path)).text_content\n    except Exception as e:\n        return f"Error reading file {filename}: {str(e)}"\n\n# Prompt: Summarize document\n@mcp.prompt()\ndef analyze_data(text: str) -> list[base.Message]:\n    """Prompt to generate a summary of the provided document text.\n    Args:\n        text: The content of the document to be summarized.\n    Returns:\n        A list of messages guiding the LLM to produce a summary.\n    """\n    return [\n        base.Message(\n            role="user",\n            content=[\n                base.TextContent(\n                    text=f"Assume the role of a data analyst specializing in academic research. \\\n                    Your task is to critically analyze the data presented in the file of the attached academic document. \\\n                    Start by summarizing the key data points and notable findings. Identify any patterns, trends, correlations, or anomalies within the dataset.:\\n\\n{text}"\n                )\n            ]\n        )\n    ]\n\ndef main():\n    """Entry point for the MCP server."""\n    mcp.run()\n\nif __name__ == "__main__":\n    main()\n```\n\nHere is a breakdown what you’ve created:\n\n1. **Tools**: Two functions decorated with `@mcp.tool()`. The `read_any_document` extracts text from documents and `save_file_to_pc` saves files to pc.\n2. **Resources**: A dynamic resource that provides access to documents in the current directory and a function to retrieve document content.\n3. **Prompts**: A prompt that users can invoke to analyze document content.\n4. **Main function**: The `main()` function serves as an entry point for running the server.\n\nNow, update the `src/document_brain/__init__.py` file to expose server components:\n\n```python\n"""Document Reader MCP server for extracting text from various document formats."""\n\nfrom .server import mcp, read_any_document, main\n```\n\n## Testing the MCP Server\n\nIn this part you will run your server and test what it can do. Before testing your MCP server, activate your virtual environment:\n\n```bash\n.venv\\Scripts\\activate # On Unix-based systesms: source .venv/bin/activate\n```\n\nNow proceed to test the MCP server using MCP inspector. Make sure you are connected to the internet.\n\n### Manual testing with the MCP Inspector\n\nFastMCP includes a built-in debugging tool called the MCP Inspector. To test your server:\n\n```bash\nmcp dev src/document_brain/server.py\n```\n\nThis starts the server and opens the Inspector in your browser (typically at `http://127.0.0.1:6274`). Click **Connect** and explore the Tools, Resources, and Prompts tabs to test your implementation.\n\nYou can verify that your server is running.\n\nhttps://images.ctfassets.net/il1yandlcjgk/4P4icEMNhEN0zEeaQmUufT/9d59bd65db67d11096c4c68a682099b8/2025-05-13-mcp-development-debugger.png?w=1003&fm=webp&q=60\n\nIn the Tools tab, there are two tools:\n\n- `read_any_document`\n- `save_file_to_pc`.\n\nYou can click on any tool and test it. For example, clicking on the `read_any_document` tool will show an input field for the file path. Enter the full path to any supported file (such as Excel) on your machine. Click `Run Tool` and the tool will convert the Excel file to markdown, displying the extracted text content.\n\nhttps://images.ctfassets.net/il1yandlcjgk/2hD7ExfyKL3PrumPZSPy8w/5737d4b9153c2998bec911a38956af1f/2025-05-13-mcp-development-debugger-testing-tools.png?w=1003&fm=webp&q=60\n\nYou can also test your MCP Server in Claude Desktop by running:\n\n```bash\nmcp install src/document_brain/server.py\n```\n\nRestart Claude Desktop and the MCP server will be attached. See screenshot below.\n\nhttps://images.ctfassets.net/il1yandlcjgk/5vy52OBZ4IMBi9BBgFlxUe/fa19b3fe76bd6075b0b5ea5e43df6f0f/2025-05-13-mcp-host-claude-desktop.png?w=1003&fm=webp&q=60\n\nNow ask Claude to analyze data in an Excel file `global_inflation_data.xlsx`, saved locally.\n\nhttps://images.ctfassets.net/il1yandlcjgk/3NgRCZVfGf2pJ4UOKab0Wx/c7d9fcf5256a103e450317e95bb83b7e/2025-05-13-claude-desktop-uses-read-any-document-tool.png?w=1003&fm=webp&q=60\n\nYour MCP Server is fully operational in Claude Desktop.\n\n### Setting up automated testing\n\nTo set up a simple test for our MCP server, create a test directory:\n\n```bash\nmkdir -p tests\ntouch tests/test_server.py\ntouch tests/__init__.py\n```\n\nNow, add basic tests in `tests/test_server.py`:\n\n```python\nimport pytest\nfrom src.document_brain.server import read_any_document\n\n# Fixture to create a temporary text file\n@pytest.fixture\ndef temp_text_file(tmp_path):\n    file_path = tmp_path / "test_document.txt"\n    file_path.write_text("This is a test document.")\n    return file_path\n\n# Test reading a valid text file\ndef test_read_valid_document(temp_text_file):\n    content = read_any_document(str(temp_text_file))\n    assert "This is a test document." in content\n\n# Test reading a non-existent file\ndef test_read_nonexistent_file():\n    content = read_any_document("nonexistent_file.txt")\n    assert "Error reading file" in content\n```\n\nThese tests verify that our document reader functions work correctly. Run the tests:\n\n```bash\npytest tests/ -v\n```\n\nThe `-v` flag tells pytest to output detailed logs about the tests.\n\nhttps://images.ctfassets.net/il1yandlcjgk/XVjwY9CSoUi72wDnDNkuO/5b4655309df0beaf0c5d463bc745309b/2025-05-13-tests-pass-before-building-package.png?w=1003&fm=webp&q=60\n\n## Packaging the Python project\n\nTo prepare your application for distribution, configure the metadata in `pyproject.toml`:\n\n```toml\n[build-system]\nrequires = ["setuptools>=61.0", "wheel"]\nbuild-backend = "setuptools.build_meta"\n\n[project]\nname = "mcp-document-brain"\nversion = "0.1.1"\ndescription = "MCP server for converting files to markdown using Markitdown"\nreadme = "README.md"\nauthors = [\n    {name = "Your name", email = "example.email@domain.com"}\n]\nlicense = {text = "MIT"}\nclassifiers = [\n    "Programming Language :: Python :: 3",\n    "License :: OSI Approved :: MIT License",\n    "Operating System :: OS Independent",\n]\nrequires-python = ">=3.12"\ndependencies = [\n    "mcp[cli]>=1.8.0",\n    "Markitdown[all]>=0.1.1",\n]\n\n[project.optional-dependencies]\ndev = [\n    "build>=1.2.2.post1",\n    "pytest>=8.3.5",\n    "twine>=6.1.0",\n]\n[project.scripts]\nmcp-document-brain = "document_brain.server:main"\n\n[tool.setuptools]\npackage-dir = {"" = "src"}\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n```\n\nThis configuration:\n\n- Sets up basic package metadata\n- Declares dependencies\n- Creates a command-line entry point\n- Configures our development tools\n\nNow, add some documentation in `README.md` to provide users with more details about your package.\n\nBuild your package using the build tool:\n\n```bash\npython -m build\n```\n\nThis generates distribution files in the `dist/` directory.\n\n## Publishing your package to PyPI\n\nBefore uploading your package to [PyPI](https://pypi.org/) (the Python Package Index), you will need to complete a few important steps:\n\n1. **Create a PyPI account** if you don’t have one already:\n\n   - Go to the [PyPI registration page](https://pypi.org/account/register/)\n   - Verify your email address after registering\n   - Set up [two-factor authentication](https://pypi.org/help/#twofa) (2FA) for better security\n2. **Generate an API token** instead of using your password:\n\n   - Log in to your PyPI account\n   - Go to [Account Settings → API tokens](https://pypi.org/manage/account/#api-tokens)\n   - Click “Add API token”, give it a name (like “document-brain-upload”), and create it\n   - Save the token somewhere safe - you will need it later on in the tutorial.\n3. **Upload your package** using Twine:\n\n   ```bash\n   twine upload dist/*\n   ```\n\n4. **When prompted for credentials, enter**:\n\n   - Username: `__token__` (type this exactly, including the underscores)\n   - Password: paste your API token\n\nFor extra security, you can store your PyPI credentials in a [`.pypirc` file](https://packaging.python.org/en/latest/specifications/pypirc/) in your home directory. Run:\n\n```text\n[pypi]\nusername = __token__\npassword = pypi-AgEI...your-token-here...\n```\n\nOnce your package is published, anyone can install it with:\n\n```bash\npip install mcp-document-brain\n# Or using uv\nuv add mcp-document-brain\n```\n\nRun it directly:\n\n```bash\nmcp-document-brain\n```\n\nYou can view your published package at `https://pypi.org/project/mcp-document-brain/`\n\n### Troubleshooting tips\n\n- If you get an error about the package name being taken, choose a different name in your `pyproject.toml` file\n- If uploads fail, make sure your token has the right permissions and has not expired\n- Check the [PyPI help docs](https://pypi.org/help/) if you run into problems.\n\n## Automating Python package publishing with CircleCI and `uv`\n\nAutomating your Python package publishing flow can save you hours of manual effort and reduce human error. In this part, you will learn how to use [CircleCI](https://circleci.com/) to automate testing, building, and publishing a Python package to PyPI.\n\nThis section walks through a complete CircleCI setup using the [uv](https://github.com/astral-sh/uv) package manager to handle dependencies. You will create a robust workflow that kicks in when you push changes to your `main` branch. This ensures publishing of your package only when it is production-ready.\n\n### Setting up CircleCI configuration\n\nCreate a CircleCI configuration file to automate testing, building, and publishing:\n\n```bash\nmkdir -p .circleci\ntouch .circleci/config.yml\n```\n\nBefore breaking it down, here is the full `.circleci/config.yml`:\n\n```yaml\nversion: 2.1\n\njobs:\n  build:\n    docker:\n      - image: cimg/python:3.12\n    steps:\n      - checkout\n\n      - run:\n          name: Install uv\n          command: |\n            curl -Ls https://astral.sh/uv/install.sh | sh\n            echo \'export PATH="$HOME/.cargo/bin:$PATH"\' >> $BASH_ENV\n            source $BASH_ENV\n\n      - run:\n          name: Install dependencies using uv\n          command: uv pip install --system -r <(uv pip compile --extra dev pyproject.toml)\n\n      - run:\n          name: Run tests\n          command: python -m pytest tests/ -v\n\n      - run:\n          name: Build package\n          command: |\n            python -m build\n\n      - persist_to_workspace:\n          root: .\n          paths:\n            - dist\n\n  publish:\n    docker:\n      - image: cimg/python:3.12\n    steps:\n      - checkout\n      - attach_workspace:\n          at: .\n      - run:\n          name: Install twine\n          command: pip install --upgrade twine\n      - run:\n          name: Upload to PyPI\n          command: twine upload dist/* -u "$PYPI_USERNAME" -p "$PYPI_PASSWORD"\n\nworkflows:\n  build-test-publish:\n    jobs:\n      - build\n      - publish:\n          requires:\n            - build\n          filters:\n            branches:\n              only: main\n```\n\n**Understanding the CircleCI configuration**\n\nThis setup creates two `jobs`:\n\n1. `build` job\n\nThis is where you define tasks before a release:\n\n- **Docker image:** Uses CircleCI’s official `cimg/python:3.12` image.\n- **Install uv:** `uv` is a faster and stable dependency manager. It replaces `pip` and `pip-tools`.\n- **Install dependencies:**\n\n```bash\nuv pip install --system -r <(uv pip compile --extra dev pyproject.toml)\n```\n\nThis command compiles and installs your `pyproject.toml`, including both your main and optional `dev` dependencies. The `--system` flag installs them into the current Python environment.\n\n- **Run tests:** Executes your test suite with `pytest`. You use `pytest` to run all tests inside the `tests/` folder and print verbose output.\n- **Build the package:** Uses `python -m build` to generate the `dist/` folder, which includes `.tar.gz` and `.whl` files for your package.\n- **Persist the build artifacts:** These are saved to a “workspace”, a temporary shared storage between jobs.\n\n2. `publish` job\n\nThis job picks up where `build` left off.\n\n- **Attach the workspace:** Brings the previously built `dist/` folder into this job.\n- **Install Twine:** Twine is the recommended tool to securely upload packages to PyPI.\n- **Upload to PyPI:** The actual publishing step happens here with:\n\n```bash\ntwine upload dist/* -u "$PYPI_USERNAME" -p "$PYPI_PASSWORD"\n```\n\nYou need to store your credentials as environment variables in your CircleCI project settings.\n\n**`workflows`: CI/CD logic**\nThis block defines when and how your jobs run.\n\n```yml\nworkflows:\n  build-test-publish:\n    jobs:\n      - build\n      - publish:\n          requires:\n            - build\n          filters:\n            branches:\n              only: main\n```\n\nHere is what it means:\n\n- The `build` job always runs.\n- The `publish` job only runs after `build` completes successfully.\n- It only triggers if the commit is pushed to the `main` branch.\n\nThis design ensures you do not accidentally publish from feature branches or failed builds.\n\n### Managing secrets in CircleCI\n\nTo securely publish to PyPI, add these environmental variables in CircleCI project settings:\n- `PYPI_USERNAME`: Set to `__token__`\n- `PYPI_PASSWORD`: Your PyPI API token\n\n## Publishing to PyPI\n\nIn this section, you will trigger the `build` and `publish` jobs.\n\n### Automating the publishing process\n\nWith your CircleCI configuration in place, you will trigger deployment as follows:\n\n1. Make changes to your code and commit them\n2. Push to GitHub\n3. Create a new project in CircleCI and link your repository\n4. When you are ready to release, either:\n   - Merge to the main branch, or\n   - Create and push a tag starting with “v” (e.g., `v0.1.0`)\n5. CircleCI tests, builds, and publishes your package to PyPI\n\nThe build and publish jobs should run successfully.\n\nhttps://images.ctfassets.net/il1yandlcjgk/TgZchXLcwD0SeoRalHTUF/b93dad7ddb66d1239f6836ff6dd0e406/2025-05-13-successful-circleci-run.png?w=1003&fm=webp&q=60\n\n### Versioning strategy\n\nFor versioning, follow Semantic Versioning:\n\n- **MAJOR** version for incompatible API changes\n- **MINOR** version for backwards-compatible functionality\n- **PATCH** version for backwards-compatible bug fixes\n\nTo release a new version:\n\n1. Update the version in `pyproject.toml`\n2. Commit the change\n3. Create and push a tag:\n\n   ```bash\n   git tag v0.1.1\n   git push origin v0.1.1\n   ```\n\n## Verifying the deployment\n\nAfter the CI pipeline completes, verify that your package by visiting its repository on PyPI. For example, this project is available at `https://pypi.org/project/mcp-document-brain/`.\n\nYou can also test the installation:\n\n```bash\n# Create a new virtual environment\npython -m venv test_env\nsource test_env/bin/activate  # On Windows: test_env\\Scripts\\activate\n\n# Install your package from PyPI\npip install mcp-document-brain\n\n# Test that it works\nmcp-document-brain --help\n```\n\nFor a thorough test, create sample documents and try using your MCP server with an LLM platform that supports the MCP.\n\n## Conclusion\n\nCongratulations! You have built a complete MCP server that extends LLM capabilities with document processing tools. You have packaged it for distribution and set up an automated CI/CD pipeline for publishing on PyPI using CircleCI.\n\nThis knowledge provides a foundation for creating more sophisticated MCP servers that could:\n\n- Connect to databases or APIs\n- Process specialized data formats\n- Integrate with external services\n- Execute domain-specific algorithms\n\nMCP opens up exciting possibilities for extending LLM capabilities in standardized ways. By combining Python’s flexibility, FastMCP’s developer-friendly API, and CircleCI’s automation, you can build powerful AI-powered tools tailored to specific workflows.\n\nWhy does this matter? As AI becomes increasingly integrated into our workflows, the ability to extend LLMs with custom capabilities will be a crucial differentiator. Your custom MCP servers can provide unique value that generic AI solutions simply cannot match.', 'extras': {'signature': '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'}}</summary>

{'type': 'text', 'text': 'Extending Large Language Models (LLMs) with custom tools has become increasingly valuable in today’s AI landscape. Model Context Protocol (MCP) servers provide a standardized way to connect external tools and resources to LLMs. This can enhance their capabilities beyond basic text generation.\n\nWhile thousands of pre-built MCP servers exist, creating your own allows you to address specific workflows. You can implement use cases that off-the-shelf solutions cannot handle. This is where the real power lies.\n\nIn this tutorial, you will learn to build a document parsing server that enables MCP hosts to understand various file formats. You will use FastMCP—the leading library for building MCPs in Python. You will also bundle your application into a Python package and publish on PyPI using an automated CI/CD pipeline.\n\nBy the end, you will have a fully functional document reader MCP server that extracts text from various document formats.\n\nHere is a demo of what you will build.\n\nhttps://images.ctfassets.net/il1yandlcjgk/2Hc7rEy3yFEqreSD7aUVsz/d520643311bfe644198f211075cd6042/2025-05-13-mcp-document-brain-demo.gif?w=1003&fm=webp&q=60\n\n## Prerequisites\n\nBefore diving in, make sure you have:\n\n- [Python 3.10+](https://www.python.org/downloads/) or newer installed on your system\n- Basic understanding of [Python packaging concepts](https://packaging.python.org/en/latest/tutorials/packaging-projects/)\n- A [PyPI account](https://pypi.org/account/register/) for publishing your package\n- A [GitHub account](https://github.com/signup) for version control\n- A [CircleCI account](https://app.circleci.com/signup) connected to your GitHub account for creating CI/CD pipelines\n- [Claude Desktop](https://claude.ai/download) for using your MCP server\n\n## Setting up the development environment\n\nBuilding an MCP server opens up exciting possibilities for extending LLM capabilities with custom tools and resources. FastMCP simplifies this process with a clean API using decorators that handle server protocol complexities.\n\nYou can visit this [GitHub repository](https://github.com/ArmstrongA/document-brain.git) to explore the code for the server you are about to build.\n\n### Initializing the project directory\n\nFirst, let us create a new directory for our project and set up the basic structure:\n\n```bash\nmkdir document-brain\ncd document-brain\n```\n\n### Installing `UV`\n\nFor dependency management, you will use `uv`—a modern, fast package manager for Python. Open powershell and run this command to install `uv`on Windows:\n\n```bash\npowershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"\n```\n\nRestart your terminal completely. Then verify the installation:\n\n```bash\nuv --version\n```\n\nYou can also install `uv` using pip:\n\n```bash\n# Using pip\npip install uv\n\nOn Unix-based systems like macOS run:\n\n```bash\ncurl -sSf https://install.python-uv.org | bash\n# Or using Homebrew\nbrew install uv\n```\n\nNow initialize your project with `uv`:\n\n```bash\nuv init\n```\n\nThis command automatically creates:\n\n- **`.python-version`**: Ensures consistent Python versions across environments\n- **`.gitignore`**: Lists files and directories for Git to ignore.\n- **`main.py`**: A starting point for development.\n- **`pyproject.toml`**: Defines project metadata and dependencies.\n- **`README.md`**: An overview of the project’s purpose, usage instructions, and other relevant information.\n\nThese files collectively set up a [foundational structure for a Python project](https://www.reddit.com/r/learnpython/comments/1jbo88t/uv_for_python_project_and_version_management/). This facilitates development and collaboration.\n\n### Creating a virtual environment and installing dependencies\n\nWhen installing packages and dependencies, `uv` automatically creates a virtual environment name `.venv`.\n\nInstall the necessary packages for developing and testing your MCP server. Run:\n\n```bash\nuv add "mcp[cli]"\nuv add "markitdown[all]"\nuv add --dev pytest build twine\n```\n\nThese packages provide:\n\n- FastMCP with CLI tools for development and debugging\n- Markitdown for document parsing functionality\n- Testing (pytest)\n- Building your package (build)\n- Publishing to PyPI (twine)\n\nBefore creating your MCP server, create the project directory structure:\n\n```bash\nmkdir -p src/document_brain\ntouch src/document_brain/__init__.py\ntouch src/document_brain/server.py\n```\n\nYour project structure looks like this:\n\n```text\nproject_root/\n├── src/\n│   └── document_brain/\n│       ├── __init__.py        # Initializes server\n│       └── server.py          # Contains the \'mcp\' instance and definitions\n├── tests/\n│   ├── __init__.py            # Makes tests discoverable\n│   └── test_server.py         # Tests for your MCP server\n├── main.py\n├── pyproject.toml             # Project configuration\n└── README.md                  # Project documentation\n```\n\nThis structure follows the recommended Python packaging standards with src-layout. The approach prevents import issues during development.\n\n## Building the MCP Server with FastMCP\n\nIn this part, you will begin by understanding three primary components of MCP servers (tools, resources, and prompts).\n\n### Understanding the core components\n\nMCP servers consists of three key components:\n\n- **Tools:** Model-controlled functions that LLMs can call to perform actions or interact with external systems.\n- **Resources:** Application-controlled data sources that inject contextual information from your systems into the conversation.\n- **Prompts:** User-controlled templates that can be invoked through UI elements to help users interact with the LLM in structured ways.\n\nNow, let us implement your document brain server in `src/document_brain/server.py`:\n\n```python\nfrom mcp.server.fastmcp import FastMCP\nfrom mcp.server.fastmcp.prompts import base\nfrom mcp.server.fastmcp.resources import DirectoryResource\nfrom pathlib import Path\nimport os\nfrom markitdown import MarkItDown\n\nmd = MarkItDown()\n\n# Initialize the FastMCP server\nmcp = FastMCP("DocumentBrain", dependencies=["markitdown[all]"])\n\n@mcp.tool(\n    annotations={\n        "title": "Read Any Document",\n        "readOnlyHint": True,\n        "openWorldHint": False\n    }\n)\ndef read_any_document(file_path: str) -> str:\n    """Read any supported document and return its text content, including OCR for images.\n    Args:\n        file_path: Path to the document to process.\n    Returns:\n        Extracted text content as a string.\n    """\n    try:\n        expanded_path = os.path.expanduser(file_path)\n        return md.convert(expanded_path).text_content\n    except Exception as e:\n        return f"Error reading file: {str(e)}"\n\n@mcp.tool(\n    annotations={\n        "title": "Save File to PC",\n        "readOnlyHint": False,\n        "openWorldHint": True\n    }\n)\ndef save_file_to_pc(filepath: str, content: str) -> str:\n    """\n    Save content to a file on the desktop.\n    Args:\n        filename: Name of the file to save (can include subdirectory)\n        content: Content to write to the file\n    Returns:\n        A success or error message\n    """\n    try:\n        # Expand the desktop path\n        desktop_path = os.path.expanduser(filepath)\n        # Ensure the filename doesn\'t contain any path traversal attempts\n        safe_filename = os.path.basename(filepath)\n        # Create the full file path\n        full_path = os.path.join(desktop_path, safe_filename)\n        # Ensure the directory exists\n        os.makedirs(os.path.dirname(full_path), exist_ok=True)\n        # Write the content to the file\n        with open(full_path, \'w\', encoding=\'utf-8\') as f:\n            f.write(content)\n        return f"File successfully saved to {full_path}"\n    except Exception as e:\n        return f"Error saving file: {str(e)}"\n\n# Now add a resource\n# Define the path to the current directory\ndocuments_path = Path(".").resolve()\n\n# Create a DirectoryResource to list files in the current directory\ndocuments_resource = DirectoryResource(\n    uri="docs://files",\n    path=documents_path,\n    name="Local Document Directory",\n    description="Lists all files in the current working directory.",\n    recursive=False  # Set to True if you want to include subdirectories\n)\n\n# Add the resource to your FastMCP server\nmcp.add_resource(documents_resource)\n\n@mcp.resource("docs://file/{filename}")\ndef get_document_content(filename: str) -> str:\n    """Retrieve the content of a specified document."""\n    try:\n        file_path = documents_path / filename\n        if not file_path.exists():\n            return f"File not found: {filename}"\n        return md.convert(str(file_path)).text_content\n    except Exception as e:\n        return f"Error reading file {filename}: {str(e)}"\n\n# Prompt: Summarize document\n@mcp.prompt()\ndef analyze_data(text: str) -> list[base.Message]:\n    """Prompt to generate a summary of the provided document text.\n    Args:\n        text: The content of the document to be summarized.\n    Returns:\n        A list of messages guiding the LLM to produce a summary.\n    """\n    return [\n        base.Message(\n            role="user",\n            content=[\n                base.TextContent(\n                    text=f"Assume the role of a data analyst specializing in academic research. \\\n                    Your task is to critically analyze the data presented in the file of the attached academic document. \\\n                    Start by summarizing the key data points and notable findings. Identify any patterns, trends, correlations, or anomalies within the dataset.:\\n\\n{text}"\n                )\n            ]\n        )\n    ]\n\ndef main():\n    """Entry point for the MCP server."""\n    mcp.run()\n\nif __name__ == "__main__":\n    main()\n```\n\nHere is a breakdown what you’ve created:\n\n1. **Tools**: Two functions decorated with `@mcp.tool()`. The `read_any_document` extracts text from documents and `save_file_to_pc` saves files to pc.\n2. **Resources**: A dynamic resource that provides access to documents in the current directory and a function to retrieve document content.\n3. **Prompts**: A prompt that users can invoke to analyze document content.\n4. **Main function**: The `main()` function serves as an entry point for running the server.\n\nNow, update the `src/document_brain/__init__.py` file to expose server components:\n\n```python\n"""Document Reader MCP server for extracting text from various document formats."""\n\nfrom .server import mcp, read_any_document, main\n```\n\n## Testing the MCP Server\n\nIn this part you will run your server and test what it can do. Before testing your MCP server, activate your virtual environment:\n\n```bash\n.venv\\Scripts\\activate # On Unix-based systesms: source .venv/bin/activate\n```\n\nNow proceed to test the MCP server using MCP inspector. Make sure you are connected to the internet.\n\n### Manual testing with the MCP Inspector\n\nFastMCP includes a built-in debugging tool called the MCP Inspector. To test your server:\n\n```bash\nmcp dev src/document_brain/server.py\n```\n\nThis starts the server and opens the Inspector in your browser (typically at `http://127.0.0.1:6274`). Click **Connect** and explore the Tools, Resources, and Prompts tabs to test your implementation.\n\nYou can verify that your server is running.\n\nhttps://images.ctfassets.net/il1yandlcjgk/4P4icEMNhEN0zEeaQmUufT/9d59bd65db67d11096c4c68a682099b8/2025-05-13-mcp-development-debugger.png?w=1003&fm=webp&q=60\n\nIn the Tools tab, there are two tools:\n\n- `read_any_document`\n- `save_file_to_pc`.\n\nYou can click on any tool and test it. For example, clicking on the `read_any_document` tool will show an input field for the file path. Enter the full path to any supported file (such as Excel) on your machine. Click `Run Tool` and the tool will convert the Excel file to markdown, displying the extracted text content.\n\nhttps://images.ctfassets.net/il1yandlcjgk/2hD7ExfyKL3PrumPZSPy8w/5737d4b9153c2998bec911a38956af1f/2025-05-13-mcp-development-debugger-testing-tools.png?w=1003&fm=webp&q=60\n\nYou can also test your MCP Server in Claude Desktop by running:\n\n```bash\nmcp install src/document_brain/server.py\n```\n\nRestart Claude Desktop and the MCP server will be attached. See screenshot below.\n\nhttps://images.ctfassets.net/il1yandlcjgk/5vy52OBZ4IMBi9BBgFlxUe/fa19b3fe76bd6075b0b5ea5e43df6f0f/2025-05-13-mcp-host-claude-desktop.png?w=1003&fm=webp&q=60\n\nNow ask Claude to analyze data in an Excel file `global_inflation_data.xlsx`, saved locally.\n\nhttps://images.ctfassets.net/il1yandlcjgk/3NgRCZVfGf2pJ4UOKab0Wx/c7d9fcf5256a103e450317e95bb83b7e/2025-05-13-claude-desktop-uses-read-any-document-tool.png?w=1003&fm=webp&q=60\n\nYour MCP Server is fully operational in Claude Desktop.\n\n### Setting up automated testing\n\nTo set up a simple test for our MCP server, create a test directory:\n\n```bash\nmkdir -p tests\ntouch tests/test_server.py\ntouch tests/__init__.py\n```\n\nNow, add basic tests in `tests/test_server.py`:\n\n```python\nimport pytest\nfrom src.document_brain.server import read_any_document\n\n# Fixture to create a temporary text file\n@pytest.fixture\ndef temp_text_file(tmp_path):\n    file_path = tmp_path / "test_document.txt"\n    file_path.write_text("This is a test document.")\n    return file_path\n\n# Test reading a valid text file\ndef test_read_valid_document(temp_text_file):\n    content = read_any_document(str(temp_text_file))\n    assert "This is a test document." in content\n\n# Test reading a non-existent file\ndef test_read_nonexistent_file():\n    content = read_any_document("nonexistent_file.txt")\n    assert "Error reading file" in content\n```\n\nThese tests verify that our document reader functions work correctly. Run the tests:\n\n```bash\npytest tests/ -v\n```\n\nThe `-v` flag tells pytest to output detailed logs about the tests.\n\nhttps://images.ctfassets.net/il1yandlcjgk/XVjwY9CSoUi72wDnDNkuO/5b4655309df0beaf0c5d463bc745309b/2025-05-13-tests-pass-before-building-package.png?w=1003&fm=webp&q=60\n\n## Packaging the Python project\n\nTo prepare your application for distribution, configure the metadata in `pyproject.toml`:\n\n```toml\n[build-system]\nrequires = ["setuptools>=61.0", "wheel"]\nbuild-backend = "setuptools.build_meta"\n\n[project]\nname = "mcp-document-brain"\nversion = "0.1.1"\ndescription = "MCP server for converting files to markdown using Markitdown"\nreadme = "README.md"\nauthors = [\n    {name = "Your name", email = "example.email@domain.com"}\n]\nlicense = {text = "MIT"}\nclassifiers = [\n    "Programming Language :: Python :: 3",\n    "License :: OSI Approved :: MIT License",\n    "Operating System :: OS Independent",\n]\nrequires-python = ">=3.12"\ndependencies = [\n    "mcp[cli]>=1.8.0",\n    "Markitdown[all]>=0.1.1",\n]\n\n[project.optional-dependencies]\ndev = [\n    "build>=1.2.2.post1",\n    "pytest>=8.3.5",\n    "twine>=6.1.0",\n]\n[project.scripts]\nmcp-document-brain = "document_brain.server:main"\n\n[tool.setuptools]\npackage-dir = {"" = "src"}\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n```\n\nThis configuration:\n\n- Sets up basic package metadata\n- Declares dependencies\n- Creates a command-line entry point\n- Configures our development tools\n\nNow, add some documentation in `README.md` to provide users with more details about your package.\n\nBuild your package using the build tool:\n\n```bash\npython -m build\n```\n\nThis generates distribution files in the `dist/` directory.\n\n## Publishing your package to PyPI\n\nBefore uploading your package to [PyPI](https://pypi.org/) (the Python Package Index), you will need to complete a few important steps:\n\n1. **Create a PyPI account** if you don’t have one already:\n\n   - Go to the [PyPI registration page](https://pypi.org/account/register/)\n   - Verify your email address after registering\n   - Set up [two-factor authentication](https://pypi.org/help/#twofa) (2FA) for better security\n2. **Generate an API token** instead of using your password:\n\n   - Log in to your PyPI account\n   - Go to [Account Settings → API tokens](https://pypi.org/manage/account/#api-tokens)\n   - Click “Add API token”, give it a name (like “document-brain-upload”), and create it\n   - Save the token somewhere safe - you will need it later on in the tutorial.\n3. **Upload your package** using Twine:\n\n   ```bash\n   twine upload dist/*\n   ```\n\n4. **When prompted for credentials, enter**:\n\n   - Username: `__token__` (type this exactly, including the underscores)\n   - Password: paste your API token\n\nFor extra security, you can store your PyPI credentials in a [`.pypirc` file](https://packaging.python.org/en/latest/specifications/pypirc/) in your home directory. Run:\n\n```text\n[pypi]\nusername = __token__\npassword = pypi-AgEI...your-token-here...\n```\n\nOnce your package is published, anyone can install it with:\n\n```bash\npip install mcp-document-brain\n# Or using uv\nuv add mcp-document-brain\n```\n\nRun it directly:\n\n```bash\nmcp-document-brain\n```\n\nYou can view your published package at `https://pypi.org/project/mcp-document-brain/`\n\n### Troubleshooting tips\n\n- If you get an error about the package name being taken, choose a different name in your `pyproject.toml` file\n- If uploads fail, make sure your token has the right permissions and has not expired\n- Check the [PyPI help docs](https://pypi.org/help/) if you run into problems.\n\n## Automating Python package publishing with CircleCI and `uv`\n\nAutomating your Python package publishing flow can save you hours of manual effort and reduce human error. In this part, you will learn how to use [CircleCI](https://circleci.com/) to automate testing, building, and publishing a Python package to PyPI.\n\nThis section walks through a complete CircleCI setup using the [uv](https://github.com/astral-sh/uv) package manager to handle dependencies. You will create a robust workflow that kicks in when you push changes to your `main` branch. This ensures publishing of your package only when it is production-ready.\n\n### Setting up CircleCI configuration\n\nCreate a CircleCI configuration file to automate testing, building, and publishing:\n\n```bash\nmkdir -p .circleci\ntouch .circleci/config.yml\n```\n\nBefore breaking it down, here is the full `.circleci/config.yml`:\n\n```yaml\nversion: 2.1\n\njobs:\n  build:\n    docker:\n      - image: cimg/python:3.12\n    steps:\n      - checkout\n\n      - run:\n          name: Install uv\n          command: |\n            curl -Ls https://astral.sh/uv/install.sh | sh\n            echo \'export PATH="$HOME/.cargo/bin:$PATH"\' >> $BASH_ENV\n            source $BASH_ENV\n\n      - run:\n          name: Install dependencies using uv\n          command: uv pip install --system -r <(uv pip compile --extra dev pyproject.toml)\n\n      - run:\n          name: Run tests\n          command: python -m pytest tests/ -v\n\n      - run:\n          name: Build package\n          command: |\n            python -m build\n\n      - persist_to_workspace:\n          root: .\n          paths:\n            - dist\n\n  publish:\n    docker:\n      - image: cimg/python:3.12\n    steps:\n      - checkout\n      - attach_workspace:\n          at: .\n      - run:\n          name: Install twine\n          command: pip install --upgrade twine\n      - run:\n          name: Upload to PyPI\n          command: twine upload dist/* -u "$PYPI_USERNAME" -p "$PYPI_PASSWORD"\n\nworkflows:\n  build-test-publish:\n    jobs:\n      - build\n      - publish:\n          requires:\n            - build\n          filters:\n            branches:\n              only: main\n```\n\n**Understanding the CircleCI configuration**\n\nThis setup creates two `jobs`:\n\n1. `build` job\n\nThis is where you define tasks before a release:\n\n- **Docker image:** Uses CircleCI’s official `cimg/python:3.12` image.\n- **Install uv:** `uv` is a faster and stable dependency manager. It replaces `pip` and `pip-tools`.\n- **Install dependencies:**\n\n```bash\nuv pip install --system -r <(uv pip compile --extra dev pyproject.toml)\n```\n\nThis command compiles and installs your `pyproject.toml`, including both your main and optional `dev` dependencies. The `--system` flag installs them into the current Python environment.\n\n- **Run tests:** Executes your test suite with `pytest`. You use `pytest` to run all tests inside the `tests/` folder and print verbose output.\n- **Build the package:** Uses `python -m build` to generate the `dist/` folder, which includes `.tar.gz` and `.whl` files for your package.\n- **Persist the build artifacts:** These are saved to a “workspace”, a temporary shared storage between jobs.\n\n2. `publish` job\n\nThis job picks up where `build` left off.\n\n- **Attach the workspace:** Brings the previously built `dist/` folder into this job.\n- **Install Twine:** Twine is the recommended tool to securely upload packages to PyPI.\n- **Upload to PyPI:** The actual publishing step happens here with:\n\n```bash\ntwine upload dist/* -u "$PYPI_USERNAME" -p "$PYPI_PASSWORD"\n```\n\nYou need to store your credentials as environment variables in your CircleCI project settings.\n\n**`workflows`: CI/CD logic**\nThis block defines when and how your jobs run.\n\n```yml\nworkflows:\n  build-test-publish:\n    jobs:\n      - build\n      - publish:\n          requires:\n            - build\n          filters:\n            branches:\n              only: main\n```\n\nHere is what it means:\n\n- The `build` job always runs.\n- The `publish` job only runs after `build` completes successfully.\n- It only triggers if the commit is pushed to the `main` branch.\n\nThis design ensures you do not accidentally publish from feature branches or failed builds.\n\n### Managing secrets in CircleCI\n\nTo securely publish to PyPI, add these environmental variables in CircleCI project settings:\n- `PYPI_USERNAME`: Set to `__token__`\n- `PYPI_PASSWORD`: Your PyPI API token\n\n## Publishing to PyPI\n\nIn this section, you will trigger the `build` and `publish` jobs.\n\n### Automating the publishing process\n\nWith your CircleCI configuration in place, you will trigger deployment as follows:\n\n1. Make changes to your code and commit them\n2. Push to GitHub\n3. Create a new project in CircleCI and link your repository\n4. When you are ready to release, either:\n   - Merge to the main branch, or\n   - Create and push a tag starting with “v” (e.g., `v0.1.0`)\n5. CircleCI tests, builds, and publishes your package to PyPI\n\nThe build and publish jobs should run successfully.\n\nhttps://images.ctfassets.net/il1yandlcjgk/TgZchXLcwD0SeoRalHTUF/b93dad7ddb66d1239f6836ff6dd0e406/2025-05-13-successful-circleci-run.png?w=1003&fm=webp&q=60\n\n### Versioning strategy\n\nFor versioning, follow Semantic Versioning:\n\n- **MAJOR** version for incompatible API changes\n- **MINOR** version for backwards-compatible functionality\n- **PATCH** version for backwards-compatible bug fixes\n\nTo release a new version:\n\n1. Update the version in `pyproject.toml`\n2. Commit the change\n3. Create and push a tag:\n\n   ```bash\n   git tag v0.1.1\n   git push origin v0.1.1\n   ```\n\n## Verifying the deployment\n\nAfter the CI pipeline completes, verify that your package by visiting its repository on PyPI. For example, this project is available at `https://pypi.org/project/mcp-document-brain/`.\n\nYou can also test the installation:\n\n```bash\n# Create a new virtual environment\npython -m venv test_env\nsource test_env/bin/activate  # On Windows: test_env\\Scripts\\activate\n\n# Install your package from PyPI\npip install mcp-document-brain\n\n# Test that it works\nmcp-document-brain --help\n```\n\nFor a thorough test, create sample documents and try using your MCP server with an LLM platform that supports the MCP.\n\n## Conclusion\n\nCongratulations! You have built a complete MCP server that extends LLM capabilities with document processing tools. You have packaged it for distribution and set up an automated CI/CD pipeline for publishing on PyPI using CircleCI.\n\nThis knowledge provides a foundation for creating more sophisticated MCP servers that could:\n\n- Connect to databases or APIs\n- Process specialized data formats\n- Integrate with external services\n- Execute domain-specific algorithms\n\nMCP opens up exciting possibilities for extending LLM capabilities in standardized ways. By combining Python’s flexibility, FastMCP’s developer-friendly API, and CircleCI’s automation, you can build powerful AI-powered tools tailored to specific workflows.\n\nWhy does this matter? As AI becomes increasingly integrated into our workflows, the ability to extend LLMs with custom capabilities will be a crucial differentiator. Your custom MCP servers can provide unique value that generic AI solutions simply cannot match.', 'extras': {'signature': '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'}}

</details>

<details>
<summary>{'type': 'text', 'text': 'Your user asks for a weekly sales report. The LLM has four tools available: querying the database, aggregating data, calculating trends, and formatting the output. It chains them together. Steps 1 and 2 go fine. Step 3 goes wrong: the LLM tries to calculate week-over-week percentage changes itself, mixes up which week is the baseline, and produces a report showing 340% growth in a category that actually declined. The user gets a polished, confident, completely wrong report.\n\nThis isn\'t a contrived scenario. It\'s the predictable outcome of asking an LLM to choreograph a multi-step workflow where some steps require symbolic computation. The LLM is good at language. It is bad at arithmetic. And when you give it tools for each individual step, you\'re asking it to be good at something else entirely: sequencing, data flow management, and knowing which steps it should delegate versus attempt itself.\n\nNow consider the alternative. The same user clicks a single prompt: "Weekly Sales Report." The server executes the deterministic steps: queries the database, aggregates by category, calculates trends server-side using exact arithmetic, and hands the LLM a precomputed dataset with one instruction: format this as an executive summary. The report is correct every time because the server handled the parts that require precision, and the LLM handled the parts that require language.\n\nIf you read [our article on tool design](https://dev.to/aws-heroes/mcp-tool-design-why-your-ai-agent-is-failing-and-how-to-fix-it-40fc), you know how to build tools that LLMs can use well. But tools solve individual tasks. What about multi-step workflows where the steps must happen in a specific order, with data flowing between them, and some steps requiring computation that LLMs shouldn\'t be doing? That\'s where MCP\'s second primitive comes in: prompts.\n\nThe **business analyst** — one of the two human corners of the Capability Square we introduced in that article — knows which workflows their business users run every week. The right operating model is domain-led, engineering-implemented, platform-governed: the analyst brings the workflow knowledge, engineers implement it, and the platform team governs how it runs in production. That is what lets the weekly sales report, the incident response runbook, or the customer onboarding checklist show up as a reliable one-click workflow for the person who actually runs it.\n\n## What Is MCP? (The 30-Second Version)\n\nThe Model Context Protocol (MCP, [spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25)) defines three primitives for connecting AI models to external services: tools, prompts, and resources. The [previous article](https://dev.to/aws-heroes/mcp-tool-design-why-your-ai-agent-is-failing-and-how-to-fix-it-40fc) covered tools, which are model-controlled primitives that let LLMs invoke server-side operations. This article covers the other two: prompts, which are user-controlled workflow packages, and resources, which provide application-controlled context. Together, the three primitives form a complete system for AI-service integration.\n\nThe enterprise mental model is the same one from the previous article: MCP for AI is what HTTP-based applications are for humans. MCP servers are the AI-facing web servers or mobile applications for your organization\'s data systems, which is why they are usually remote services rather than local helpers. They should also be thin and mostly stateless: an interface layer over internal systems, not a stateful application tier of their own. The main exception is explicit long-running task handling, where state is persisted deliberately because the work itself outlives a single request, aka MCP Tasks. We will describe Tasks in a future article in the series.\n\n## The Three Control Planes\n\nMCP\'s three primitives aren\'t just three types of capability. They represent three distinct control planes, or three answers to the question "who decides when this gets used?"\n\n**Tools are model-controlled.** The LLM (model) decides when to invoke them. When a user asks, "Where\'s my order?", the LLM selects `track_latest_order` from the available tools. The user never explicitly chose that tool; the LLM\'s reasoning did. This is the right model for individual tasks where the LLM\'s judgment about which tool to call is sufficient.\n\n**Prompts are user-controlled.** The human explicitly triggers them. In Claude Desktop, they appear as slash commands. In other clients, they show up as menu items or quick actions. The user sees "Weekly Sales Report" and clicks it, entering a week number. There\'s no ambiguity about what will happen, no LLM judgment about which workflow to run. The user chose.\n\n**Resources are application-controlled.** The host application decides when to pull them into context. A resource might be a database schema, a configuration file, or a live dashboard. The application injects it into the conversation when relevant. For example, loading an API schema before a coding task. Neither the user nor the LLM explicitly requested it; the application determined it was needed.\n\nThis taxonomy tells you which primitive to use. If the LLM should decide, use a tool. If the user decides, use a prompt. If the application decides to use a resource.\n\nIn practice, many enterprise deployments add one more concept on top of these three primitives: **Tasks**. Tasks are not part of the base three-way split. They are an extension pattern for long-running operations such as scans, report generation, provisioning, or approvals. They are also the main exception to the normal stateless model. The request/response interface remains stateless, but the server explicitly persists task state and exposes progress or completion, rather than relying on sticky in-memory sessions.\n\nThis maps cleanly onto the Capability Square from the previous article — and prompts are where the split between the two human corners pays off the most:\n\n| Control Plane | Who Triggers at Runtime | Square Corner(s) | Strength |\n| --- | --- | --- | --- |\n| Tools | The LLM (model) | LLM | Intent interpretation, tool selection |\n| Prompts | The business user | Business Analyst (authors) + Business User (triggers) | Workflow knowledge encoded once, invoked many times |\n| Resources | The host application | Host + Server | Context management, data access |\n\nhttps://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fb7qf7k2pfbysr8vbutzw.png\n\nPrompts span both human corners of the square. The business analyst — the domain lead for workflow design — encodes an expert workflow into a prompt at design time. Engineers implement that workflow, and the platform team governs its deployment and control. The business user triggers it with one click at runtime. The prompt is literally the handoff artifact between the two humans: the analyst\'s workflow knowledge, packaged so a user doesn\'t need to recreate it every Monday morning. Tools, by contrast, sit under the LLM corner because the model\'s judgment determines when they are called. Resources sit at the boundary between the host application and the MCP server: the host decides _when_ to pull a resource into context, but the server _provides_ the content. This is the one primitive in which two actors collaborate without either human being directly in the loop, which partly explains why its ecosystem support lags behind that of tools and prompts. When all three control planes work together, the system covers every type of interaction: ad-hoc tasks (tools), structured workflows (prompts), and ambient context (resources). And because resource loading is application-dependent, the host may or may not inject the right resource at the right time — so an important role of prompt workflows is to explicitly load the relevant resources into context as part of the workflow definition. This ensures the LLM has the context it needs, regardless of what the host application decided to provide.\n\n## The Primitive You\'re Not Using\n\nMost MCP servers expose tools. A growing number expose resources. Almost none expose prompts.\n\nBrowse the MCP ecosystem, the tutorial repositories, the example servers, and the community showcases, and you\'ll find tool after tool after tool. Prompts are either absent entirely or limited to trivial "system message" wrappers that add no value beyond what the user could type themselves. The MCP official blog didn\'t publish its first [prompts-for-automation post](https://blog.modelcontextprotocol.io/posts/2025-07-29-prompts-for-automation/) until mid-2025, months after the protocol launched. The ecosystem followed suit: tools are easy to demo, prompts require thinking about workflows, and most tutorials took the easy path.\n\nThere\'s another reason prompts are underutilized: minimal SDK support. Most MCP SDKs treat prompts as simple message templates: you return a list of messages, and that\'s it. There\'s no built-in abstraction for multi-step workflows, data flow between steps, or hybrid execution where the server handles some steps and the LLM handles others. This is precisely why the PMCP (Pragmatic MCP) SDK added deep support for workflow prompts as an enterprise feature, the `SequentialWorkflow` abstraction we\'ll demonstrate in this article. Without SDK support, building reliable workflow prompts requires significant boilerplate that most teams don\'t invest in.\n\nThis is a missed opportunity. Prompts solve a reliability problem that tools cannot solve for known, repeatable workflows.\n\nConsider the gap. When you leave a multi-step workflow entirely to the LLM, using only tools, you\'re relying on instruction-only orchestration: the LLM reads the tool descriptions, figures out the right sequence, handles data flow between steps, and decides which computations to delegate versus attempt. In our experience building production MCP servers with the PMCP SDK, testing multi-step workflows like report generation, data pipelines, and incident response across multiple LLM models, instruction-only approaches typically achieve 60-70% compliance for complex workflows. That means 30-40% of the time, the LLM gets something wrong: a step out of order, a calculation it shouldn\'t have attempted, a variable lost between tool calls.\n\nNow compare hybrid execution, where the prompt defines the workflow, the server executes the deterministic steps, and the LLM fills in only where its language intelligence is needed. In the same test scenarios, hybrid execution typically achieves 85-95% compliance. These numbers come from internal benchmarks, not published studies, and will vary by model, workflow complexity, and domain, but the direction is consistent: reducing the LLM\'s decision space materially improves reliability.\n\nThe reason is straightforward. Prompts reduce the LLM\'s decision space and move **workflow state management** from the LLM\'s volatile context to explicit server-side execution state. In a multi-step tool chain, the LLM must track variables between calls, remember which step it\'s on, and pass results forward correctly, all in its context window, where information degrades with distance. In a workflow prompt, the server manages that state deterministically through request-scoped execution and, when necessary, explicitly persisted state. The LLM receives a pre-built plan with most steps already completed. It only needs to handle the parts that genuinely require language understanding: summarization, formatting, and inference.\n\nThe most common failure mode has a name: **calculation hallucination**. When an LLM sees a "calculate" tool and a "format" tool, it often skips the calculation tool to save a round trip and attempts the arithmetic itself. The result looks plausible, and the format is right; however, the numbers are wrong. Hybrid execution prevents this entirely: the server runs the calculation, the LLM never sees the raw numbers, and the result is correct by construction.\n\nIf you\'re measuring task completion across diverse requests, and you should be, as we argued in the tool design article, prompts are how you push completion rates from "usually works" to "reliably works" for your most common workflows.\n\n## From Protocol to Workflow\n\nAt the protocol level, a prompt is simple: the client calls `prompts/get` with a name and arguments, and the server returns a `GetPromptResult` containing a sequence of `PromptMessage` values. Each message has a role (`System` or `User`) and content (text, images, or embedded resources). The client uses these messages to populate the conversation and guide the LLM\'s response. Clients discover available prompts via `prompts/list` \\-\\- parallel to `tools/list` \\-\\- and present them to users as slash commands, menu items, or quick actions. The key difference from tools: the user explicitly selects them. There\'s no LLM reasoning about which prompt to invoke.\n\nAt this protocol level, prompts are message templates. Useful for setting up context, but not fundamentally different from what the user could type themselves. The real power emerges when you move from templates to workflows: multi-step processes in which data flows between steps, and the server executes what it can before handing off to the LLM. In a production deployment, that workflow engine should still fit the same remote, mostly stateless service model: deterministic steps execute within the request, and truly long-running work is broken out into explicit tasks.\n\n**An important distinction:** base MCP prompts are message templates. The server-executed workflow behavior shown below is a PMCP SDK abstraction built on top of prompts, tools, and resources. It uses the prompt protocol as the entry point, but adds a workflow engine that executes deterministic steps server-side before returning the message sequence to the client. This is not part of the MCP spec -- it\'s what a well-designed SDK can layer on top of it.\n\n## The Weekly Sales Report: One Click, Complex Result\n\nHere\'s the weekly sales report as a `SequentialWorkflow` \\-\\- a PMCP abstraction where each step can feed data into the next:\n\n```rust\nuse pmcp::server::workflow::{\n    dsl::{constant, field, from_step, prompt_arg},\n    SequentialWorkflow, ToolHandle, WorkflowStep,\n};\nuse serde_json::json;\n\n// SequentialWorkflow: a multi-step prompt where data flows between steps.\n// Unlike SyncPrompt (which builds static messages), SequentialWorkflow\n// orchestrates tool calls with typed data bindings between steps.\nlet sales_report = SequentialWorkflow::new(\n    "weekly_sales_report",\n    "Generate a formatted weekly sales report with trends and key metrics."\n)\n// Arguments: what the user provides when triggering this prompt\n.argument("week", "Week identifier (e.g., \'2026-W12\')", true)\n.argument("format", "Output format: summary or detailed", false)\n\n// Step 1: Query sales database (server executes -- deterministic)\n// The server calls query_database with constant + user-provided args.\n// No LLM needed: this is pure data retrieval.\n.step(\n    WorkflowStep::new("query_sales", ToolHandle::new("query_database"))\n        .arg("query_type", constant(json!("weekly_sales")))\n        .arg("week", prompt_arg("week"))\n        .bind("sales_data")  // output available as "sales_data" for later steps\n)\n\n// Step 2: Aggregate by category (server executes -- deterministic)\n// Uses the output from step 1. The server chains these automatically.\n.step(\n    WorkflowStep::new("aggregate", ToolHandle::new("aggregate_metrics"))\n        .arg("data", from_step("sales_data"))  // entire output from step 1\n        .arg("group_by", constant(json!("product_category")))\n        .bind("aggregated")\n)\n\n// Step 3: Calculate week-over-week trends (server executes -- deterministic)\n// This is the step that failed in our opening scenario when the LLM\n// tried to do it. The server handles the arithmetic correctly every time.\n.step(\n    WorkflowStep::new("calc_trends", ToolHandle::new("calculate_trends"))\n        .arg("current_week", from_step("aggregated"))\n        .arg("comparison", constant(json!("previous_week")))\n        .bind("trends")\n)\n\n// Step 4: Format as executive summary (LLM needed -- natural language)\n// This step requires intelligence: choosing which metrics to highlight,\n// writing prose summaries, deciding what "noteworthy" means.\n// The server provides the data and guidance; the LLM provides the writing.\n.step(\n    WorkflowStep::new("format_report", ToolHandle::new("format_output"))\n        .with_guidance(\n            "Format the aggregated data into an executive summary for week {week}.\\n\\\n             Highlight the top 3 performing categories and any \\\n             week-over-week trends that exceed 10% change.\\n\\\n             Use the report template for consistent formatting."\n        )\n        .with_resource("template://reports/weekly-sales")\n        .expect("Report template resource")\n        .arg("data", from_step("aggregated"))\n        .arg("trends", from_step("trends"))\n        .arg("format", prompt_arg("format"))\n        .bind("report")\n);\n```\n\nFollow the data flow through the DSL helpers. `prompt_arg("week")` pulls the user-provided week into step 1. `from_step("sales_data")` feeds step 1\'s entire output into step 2. `from_step("aggregated")` chains step 2\'s result into step 3. Each `bind("name")` names a step\'s output, allowing subsequent steps to reference it. The data flows forward through the workflow without any LLM involvement in the plumbing.\n\nSteps 1-3 are deterministic. The server executes them because each parameter can be resolved from prompt arguments (`prompt_arg`), constants (`constant`), or prior-step bindings (`from_step`). No judgment required. No natural language interpretation. Just data retrieval, aggregation, and arithmetic.\n\nStep 4 is where the server stops and hands off. The `format_output` tool needs LLM intelligence for natural language summarization: choosing which metrics to highlight, writing prose, deciding what "noteworthy" means. The server provides everything the LLM needs -- the aggregated data (from steps 1-3), the guidance (what to highlight), and a report template resource. The LLM\'s job is reduced to writing.\n\nRemember the opening scenario? The LLM tried to calculate week-over-week trends and got the arithmetic wrong—mixing up baselines and producing a report showing 340% growth in a category that actually declined. With this workflow, the server handles the arithmetic in step 3. Deterministically. Correctly. Every time. The LLM only enters at step 4, where its strength—natural language—is needed.\n\nRegistration ties the workflow to the server\'s existing tools:\n\n```rust\nServer::builder()\n    .tool("query_database", query_db_tool)\n    .tool("aggregate_metrics", aggregate_tool)\n    .tool("calculate_trends", trends_tool)\n    .tool("format_output", format_tool)\n    .resources(report_templates)\n    .prompt_workflow(sales_report)?  // validates bindings and registers as prompt\n    .build()?\n    .run_streamable_http("0.0.0.0:3000").await?;\n```\n\nNotice `.prompt_workflow()` validates the workflow\'s bindings at registration time. If you reference a binding that doesn\'t exist -- say, `from_step("sales_data")` with a typo -- you get an error at startup, not a runtime surprise when a user triggers the prompt. The tools you already built become the building blocks. The workflow just orchestrate them.\n\nThe user clicks one prompt. Three database operations, one aggregation, and one trend calculation happen server-side in milliseconds. The LLM receives the complete data and writes the summary. One click, complex result.\n\n## Partial Execution Plans: The Server Does What It Can\n\nWhen a user invokes the weekly sales report prompt, the server doesn\'t just return instructions. It returns a _partial execution plan_: a conversation trace showing what was already done and what remains.\n\nThe server executed steps 1-3 and embedded the actual results. Here\'s a simplified version of what the client LLM receives:\n\n```\nMessage 1 (User): "Generate weekly sales report for 2026-W12"\nMessage 2 (Assistant): "Plan: 1. Query sales DB  2. Aggregate  3. Calculate trends  4. Format"\nMessage 3 (Assistant): "Calling query_database..."\nMessage 4 (Tool Result): {"total_revenue": 284500, "transactions": 1247, ...}  ← PRE-EXECUTED by server\nMessage 5 (Assistant): "Calling aggregate_metrics..."\nMessage 6 (Tool Result): {"categories": [{"name": "Enterprise", "revenue": 142000}, ...]}  ← PRE-EXECUTED by server\nMessage 7 (Assistant): "Calling calculate_trends..."\nMessage 8 (Tool Result): {"enterprise": "+12%", "smb": "-3%", "startup": "+28%", ...}  ← PRE-EXECUTED by server\nMessage 9 (Assistant): "Format the aggregated data into an executive summary for 2026-W12..."\nMessage 10 (Resource): [weekly-sales template content]\n```\n\nMessages 1-8 are done. The tool results (Messages 4, 6, 8) were pre-executed by the server; the LLM didn\'t call those tools. It receives actual data —real revenue numbers, real category breakdowns, real trend percentages—not instructions to fetch that data. The server already queried the database, already aggregated, already calculated. The results are embedded in the conversation trace as if the tools had been called, but no LLM decision-making was involved.\n\nMessage 9 is the guidance for the remaining step. Message 10 is the resource template. The LLM\'s job is reduced to: take this data, follow this guidance, use this template, write prose. That\'s one decision (how to write the summary) instead of the dozens of decisions the instruction-only approach requires (which tools to call, in what order, how to handle errors, whether to do the arithmetic itself).\n\nThis is not a template. It\'s an execution plan where the server has already completed the deterministic portion. The distinction matters: a template says "do these steps." A partial execution plan says, "these steps are done and here are the results, now do the remaining steps." The LLM starts from step 4, not step 1.\n\nThis is the Capability Square operating at the workflow level. The **server** handles deterministic computation — its strength. The **LLM** handles natural language — its strength. The **business analyst** designed the workflow at design time, identifying which steps are deterministic and which require intelligence — their strength. And the **business user** invoked it at runtime with the specific parameters (the week, the service, the severity) that only they, living inside the working context, can provide — their strength. All four corners working together, not on a single tool call, but across an entire workflow.\n\nThe compliance improvement is consistent across our internal benchmarks. Instruction-only approaches, where the prompt simply says "follow these steps: 1. Query the sales DB, 2. Aggregate by category, 3. Calculate trends, 4. Format as a summary:" and leave every decision to the LLM. It might skip steps, reorder them, call different tools, or do the arithmetic itself (badly). Hybrid execution, where steps 1-3 are already done, and the LLM just needs to format, dramatically narrows the decision space. Far fewer decisions, far fewer failure points, far more reliable output.\n\nTest this with your own workflows. Take a 4-step process that your team runs weekly. Build it as an instruction-only prompt, then as a SequentialWorkflow with hybrid execution. Run both 20 times. The difference in successful completions will make the case.\n\n## Incident Response: When the Server Needs the LLM\n\nThe sales report workflow was mostly deterministic: three server-executed steps, one LLM step. But not every workflow splits that cleanly. Consider incident response, where the server gathers data, but the LLM needs to do the hard part of synthesis and recommendation.\n\nA 5-step incident response workflow:\n\n1. Check service status (server executes -- API call, deterministic)\n2. Pull recent error logs (server executes -- log query, deterministic)\n3. Correlate with recent deployments (server executes -- git/deploy history lookup, deterministic)\n4. Draft incident summary (LLM needed -- synthesis, pattern recognition, writing)\n5. Suggest mitigation steps (LLM needed -- reasoning about root cause, recommending actions)\n\nHere\'s the sketch -- not a full implementation, but enough to see the pattern:\n\n```rust\nSequentialWorkflow::new("incident_response", "Investigate and summarize a service incident")\n    .argument("service", "Service name or ID", true)\n    .argument("severity", "Severity level: P1, P2, P3", true)\n\n    // Steps 1-3: Server handles (deterministic data gathering)\n    .step(/* check_service_status -- server executes */)\n    .step(/* query_error_logs -- server executes */)\n    .step(/* check_recent_deploys -- server executes */)\n\n    // Steps 4-5: LLM handles (intelligence required)\n    .step(\n        WorkflowStep::new("draft_summary", ToolHandle::new("create_incident_report"))\n            .with_guidance(\n                "Analyze the service status, error logs, and deployment history.\\n\\\n                 Draft an incident summary for {service} at severity {severity}.\\n\\\n                 Include: timeline, affected systems, error patterns, and \\\n                 correlation with recent deployments."\n            )\n            .arg("status", from_step("service_status"))\n            .arg("logs", from_step("error_logs"))\n            .arg("deploys", from_step("deploy_history"))\n            .bind("summary")\n    )\n    .step(\n        WorkflowStep::new("suggest_mitigation", ToolHandle::new("recommend_actions"))\n            .with_guidance(\n                "Based on the incident summary, suggest 2-3 mitigation steps.\\n\\\n                 If the incident correlates with a recent deployment, include \\\n                 a rollback recommendation."\n            )\n            .arg("summary", from_step("summary"))\n            .bind("recommendations")\n    )\n```\n\nThe split is different from the sales report. The sales report was a 3-step server, 1-step LLM—mostly deterministic. The incident response is 3 steps for the server; 2 steps for the LLM. The analysis and recommendation require genuine intelligence. But the constant is the same: the server gathers all the data the LLM needs before handing off. The LLM doesn\'t have to figure out which APIs to call or which logs to check. It receives the service status, error logs, and deployment history, then applies its strengths: synthesis and reasoning.\n\nNotice that step 5 depends on step 4\'s output (`from_step("summary")`). The LLM executes both steps, but the data dependency is explicit in the workflow. The business analyst who designed this workflow decided that the mitigation suggestions should be based on the incident summary rather than the raw data. That\'s domain knowledge encoded in the workflow structure.\n\nThe partial execution plan for this workflow looks different, too. The server executes steps 1-3 and embeds the results. The LLM receives three steps\' worth of data and two steps\' worth of guidance. It drafts the summary, then uses that summary to suggest mitigations. The workflow is longer, the LLM does more, but the pattern is identical: the server handles the deterministic parts, the LLM handles the intelligence parts.\n\n## The Business Analyst\'s Playbook: Learning What Business Users Need\n\nThe weekly sales report and the incident response share something important: someone who understands the organization\'s workflows designed them. That someone is the **business analyst** — one of the two human corners of the Capability Square. In a strong enterprise setup, workflow design is domain-led, engineering-implemented, and platform-governed. The analyst shares a domain with the business users they\'re designing for, and their role doesn\'t end at tool design. It extends to workflow design: identifying which processes their business users run repeatedly, which steps are deterministic, and where the LLM\'s intelligence adds value.\n\nThe following diagram illustrates the benefits of adding workflow prompts to the MCP servers, as they dramatically reduce the effort for busy business people and significantly increase the completion rate of requests and their consistency:\n\nhttps://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2F7hdnkp197kssqj3xz7or.png\n\nHere\'s how to approach workflow prompt design in practice:\n\n1. **Observe your users.** What tasks do they repeat weekly? Monthly? What multi-step processes do they describe as "the usual"? These are prompt candidates. Every Monday, the sales team generates a weekly report. Every time there\'s an outage, the ops team runs the same diagnostic sequence. Every quarter, the finance team reconciles accounts. These are not ad hoc tasks, as they are workflows that run on a schedule, with the same steps and for the same reasons.\n\n2. **Identify the deterministic core.** For each repeating workflow, ask: which steps are always the same? Which steps require judgment? The always-the-same steps become server-executed workflow steps with `constant()` and `from_step()` bindings. The judgment steps become LLM-guided steps with `.with_guidance()`. The sales report\'s trend calculation is always the same arithmetic. The incident response\'s mitigation recommendation always requires judgment. The split is usually obvious once you look for it.\n\n3. **Start with one prompt.** Don\'t build 20 prompts. Build the one prompt that saves the most time for the most users. Measure its completion rate. Iterate. This mirrors the tool design advice from the [previous article](https://../01-tool-design/article.md): start with the 20% that serves 80%. For prompts, start with the one workflow your team runs most often.\n\n4. **Connect prompts to tools.** Prompts don\'t replace tools -- they orchestrate them. Your existing tools become the building blocks of workflow prompts. A SequentialWorkflow\'s steps call your tools via `ToolHandle`. The `query_database`, `aggregate_metrics`, and `calculate_trends` tools existed independently before the sales report workflow was built. The workflow just wired them together with data flow and execution order.\n\n5. **Iterate based on failure modes.** If the LLM consistently gets step N wrong, move step N to the server side. If the server can\'t handle step M because it requires judgment, move it to the LLM with clear guidance. The boundary between deterministic and intelligent steps is not fixed -- it\'s something you discover through observation and measurement.\n\nThe business analyst\'s role is to encode organizational knowledge into the MCP server — knowledge they are qualified to encode precisely because they share a domain with the business users who will invoke it. Tools encode individual capabilities. Prompts encode workflows — the sequences, the data flow, the decision about which steps need human-level intelligence and which don\'t. You know which workflows matter. You know which steps are deterministic. You know where the LLM\'s intelligence adds value. Encode that knowledge in prompts.\n\nTrack prompt invocation frequency and completion rates. A prompt that\'s invoked 50 times a week with 90% completion is saving your team hours of manual orchestration. A prompt that\'s never invoked is telling you something about your understanding of user needs. Both signals are useful -- one tells you what to optimize, the other tells you what to rethink.\n\nNone of this removes the need for security-by-design. Prompts are not "just UX." They package access to real systems and real workflows. The same controls apply here as in tools: per-request authn and authz, policy checks on downstream operations, audit logs, rate limits, secret isolation, and clear boundaries on which systems the workflow may touch. If a workflow includes code mode, the controls need to be tighter still: validate first, approve when the risk warrants it, and execute only within a constrained sandbox.\n\n## Resources: The Application-Controlled Plane\n\nWe\'ve covered tools (model-controlled) and prompts (user-controlled). The third primitive is resources: application-controlled context that the host application pulls into the conversation.\n\nResources are read-only reference material -- documentation, schemas, configuration, templates. They provide context that helps agents make better decisions. Where tools perform actions and prompts orchestrate workflows, resources serve information on request. They are passive: the server publishes them, and the client or prompt reads them when needed.\n\nHere\'s a resource using the PMCP SDK:\n\n```rust\nuse pmcp::{StaticResource, ResourceCollection};\n\n// Resources provide context data that agents can read before acting.\n// Unlike tools (which perform actions) or prompts (which orchestrate workflows),\n// resources are passive: they serve information on request.\nlet resources = ResourceCollection::new()\n    .add_resource(\n        StaticResource::new_text(\n            "docs://sales/schema",\n            "# Sales Database Schema\\n\\n\\\n             ## Tables\\n\\\n             - `orders`: order_id, customer_id, total, created_at\\n\\\n             - `products`: product_id, name, category, price\\n\\\n             - `customers`: customer_id, name, email, segment\\n\\n\\\n             ## Common Queries\\n\\\n             Weekly sales: GROUP BY date_trunc(\'week\', created_at)\\n\\\n             By category: JOIN products ON orders.product_id = products.product_id"\n        )\n        .with_name("Sales Database Schema")\n        .with_description(\n            "Database schema and common query patterns for the sales system. \\\n             Read this before constructing database queries."\n        )\n        .with_mime_type("text/markdown")\n    );\n```\n\nURI design matters. Use scheme prefixes to organize your resources: `docs://` for documentation, `config://` for configuration, `data://` for structured data, `template://` for report and output templates. The URI is a stable identifier that clients and prompts reference -- `docs://sales/schema` tells both humans and agents what they\'ll find before reading it.\n\nThe `.with_description()` call serves the same purpose as tool descriptions: it helps agents decide whether a resource is relevant before reading its content. A well-described resource lets an agent skip resources it doesn\'t need, reducing unnecessary context in the conversation.\n\nNotice how this connects to the weekly sales report prompt. In that workflow, step 4 used `.with_resource("template://reports/weekly-sales")` to fetch a report template and embed its content in the conversation trace. Resources provide the context that makes prompts more effective -- the LLM reads the schema to understand the data it\'s formatting, reads the template to follow the expected output structure. Resources and prompts are designed to work together.\n\n## The Ecosystem Reality Check\n\nResources are the least mature of the three MCP primitives in terms of client support. The spec defines them comprehensively -- annotations, subscriptions, URI templates, content types. The PMCP SDK supports them fully. But client implementations lag behind.\n\nMost MCP clients implement the `resources/list` and `resources/read` protocol operations, but the user experience varies significantly. Claude Desktop requires users to explicitly select resources from a list. There is no standardized resource picker UI across clients. And critically, resource access is a client-side operation -- the LLM has no built-in way to request a resource the way it can call a tool. Unless the client proactively injects resources into context, or the server wraps resource access as a tool, the LLM never sees them.\n\nThe gap between spec and ecosystem is real. The MCP specification describes a rich resource system with subscriptions for change notifications, URI templates for parameterized access, and annotations for priority and freshness signals. In practice, most clients implement the basics (list and read) and skip the rest. If you build a resource-heavy server today, you\'re building ahead of client support.\n\nThis doesn\'t mean you shouldn\'t build resources. It means you should build them with realistic expectations about how they\'ll be consumed today, while designing for where the ecosystem is headed. The patterns in the next section bridge the gap.\n\n## Pragmatic Bridge Patterns: Making Resources Work Today\n\nFour patterns let you get value from resources today, regardless of client support.\n\n**1\\. Wrap resources as tools** (most reliable today). Instead of serving a resource at `docs://sales/schema`, create a `get_sales_schema` tool that returns the same content. The LLM discovers and calls tools reliably -- this is the pragmatic path when you need agents to access reference data without depending on client resource support.\n\n```rust\n// Bridge pattern: expose resource content as a tool.\n// Until clients reliably handle resources, tools are the safe path.\n.tool("get_sales_schema", /* returns the same content as docs://sales/schema */)\n```\n\nThis isn\'t elegant, but it works everywhere. You can maintain both the resource (for clients that support it) and the tool wrapper (for clients that don\'t), serving the same underlying content through both channels.\n\n**2\\. Resource templates as parameterized access.** URI templates like `docs://reports/{report_type}` let the server generate URIs from parameters. When clients support resource templates, they can offer auto-complete for resource URIs -- the user types `docs://reports/` and sees available report types. This pattern is worth implementing now because it costs nothing extra and will work well as clients catch up.\n\n**3\\. Prompt-mediated resource loading.** This is the pattern we already saw: `.with_resource(uri)` in SequentialWorkflow steps. The server fetches the resource during prompt execution and embeds it in the conversation. This works today because it doesn\'t depend on client resource support at all -- the server handles the resource loading internally, and the client just sees the content in the prompt messages.\n\n**4\\. Subscribe and automatic injection** (future pattern). Clients can subscribe to resource changes via `resources/subscribe`. When the resource updates, the server sends a notification, and the client can refresh its context. This enables "always up-to-date context" without manual polling -- imagine an agent that automatically gets the latest API schema whenever it changes. This is where resources are headed. When client support catches up, automatic resource injection will make context management seamless.\n\nBuild your resources now. Use bridge patterns for today\'s clients. As the ecosystem matures, your resources will work natively -- and you\'ll already have the content, the URIs, and the descriptions in place.\n\n## Key Takeaways\n\n1. **Three control planes, three primitives.** Tools are model-controlled (the LLM decides). Prompts are user-controlled (the human decides). Resources are application-controlled (the host decides). Knowing which to use is the first design decision for any MCP capability.\n\n2. **Prompts solve the workflow reliability problem.** For known, repeatable workflows, hybrid execution -- where the server handles deterministic steps and the LLM handles intelligence -- consistently outperforms instruction-only orchestration in our benchmarks. Each party does what it\'s built for.\n\n3. **Partial execution plans are the key differentiator.** A prompt doesn\'t just send instructions. It returns a conversation trace with completed tool results, guidance for remaining steps, and embedded resource content. The LLM receives data, not directions.\n\n4. **The business analyst designs workflows, not just tools.** Observe which tasks your business users repeat. Identify the deterministic core. Package it as a SequentialWorkflow. Start with one prompt for your team\'s most common workflow and measure its completion rate. This is the handoff between the two human corners of the square: the analyst encodes once at design time, the business user triggers many times at runtime.\n\n5. **Resources are underbuilt but worth building.** Client support is thin today. Use bridge patterns -- wrap as tools, prompt-mediated loading -- for immediate value. Design for where the ecosystem is going, and your resources will be ready when clients catch up.\n\n6. **Tasks are the explicit exception to the stateless rule.** Most MCP interactions should stay stateless. When work outlives a single request, model it as a task with persisted state, progress tracking, and clear completion semantics instead of smuggling session state into the server process.\n\n7. **Prompts and tools are complementary.** Prompts orchestrate tools. Your existing tools become the building blocks of workflow prompts. Good tool design (from the [previous article](https://../01-tool-design/article.md)) makes good prompt design possible.\n\n8. **Measure prompt completion rates.** Track invocation frequency and success across diverse users. If a prompt is never invoked, your understanding of user needs may be wrong. If it fails consistently at step N, move step N server-side. Both signals guide iteration.', 'extras': {'signature': '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'}}</summary>

{'type': 'text', 'text': 'Your user asks for a weekly sales report. The LLM has four tools available: querying the database, aggregating data, calculating trends, and formatting the output. It chains them together. Steps 1 and 2 go fine. Step 3 goes wrong: the LLM tries to calculate week-over-week percentage changes itself, mixes up which week is the baseline, and produces a report showing 340% growth in a category that actually declined. The user gets a polished, confident, completely wrong report.\n\nThis isn\'t a contrived scenario. It\'s the predictable outcome of asking an LLM to choreograph a multi-step workflow where some steps require symbolic computation. The LLM is good at language. It is bad at arithmetic. And when you give it tools for each individual step, you\'re asking it to be good at something else entirely: sequencing, data flow management, and knowing which steps it should delegate versus attempt itself.\n\nNow consider the alternative. The same user clicks a single prompt: "Weekly Sales Report." The server executes the deterministic steps: queries the database, aggregates by category, calculates trends server-side using exact arithmetic, and hands the LLM a precomputed dataset with one instruction: format this as an executive summary. The report is correct every time because the server handled the parts that require precision, and the LLM handled the parts that require language.\n\nIf you read [our article on tool design](https://dev.to/aws-heroes/mcp-tool-design-why-your-ai-agent-is-failing-and-how-to-fix-it-40fc), you know how to build tools that LLMs can use well. But tools solve individual tasks. What about multi-step workflows where the steps must happen in a specific order, with data flowing between them, and some steps requiring computation that LLMs shouldn\'t be doing? That\'s where MCP\'s second primitive comes in: prompts.\n\nThe **business analyst** — one of the two human corners of the Capability Square we introduced in that article — knows which workflows their business users run every week. The right operating model is domain-led, engineering-implemented, platform-governed: the analyst brings the workflow knowledge, engineers implement it, and the platform team governs how it runs in production. That is what lets the weekly sales report, the incident response runbook, or the customer onboarding checklist show up as a reliable one-click workflow for the person who actually runs it.\n\n## What Is MCP? (The 30-Second Version)\n\nThe Model Context Protocol (MCP, [spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25)) defines three primitives for connecting AI models to external services: tools, prompts, and resources. The [previous article](https://dev.to/aws-heroes/mcp-tool-design-why-your-ai-agent-is-failing-and-how-to-fix-it-40fc) covered tools, which are model-controlled primitives that let LLMs invoke server-side operations. This article covers the other two: prompts, which are user-controlled workflow packages, and resources, which provide application-controlled context. Together, the three primitives form a complete system for AI-service integration.\n\nThe enterprise mental model is the same one from the previous article: MCP for AI is what HTTP-based applications are for humans. MCP servers are the AI-facing web servers or mobile applications for your organization\'s data systems, which is why they are usually remote services rather than local helpers. They should also be thin and mostly stateless: an interface layer over internal systems, not a stateful application tier of their own. The main exception is explicit long-running task handling, where state is persisted deliberately because the work itself outlives a single request, aka MCP Tasks. We will describe Tasks in a future article in the series.\n\n## The Three Control Planes\n\nMCP\'s three primitives aren\'t just three types of capability. They represent three distinct control planes, or three answers to the question "who decides when this gets used?"\n\n**Tools are model-controlled.** The LLM (model) decides when to invoke them. When a user asks, "Where\'s my order?", the LLM selects `track_latest_order` from the available tools. The user never explicitly chose that tool; the LLM\'s reasoning did. This is the right model for individual tasks where the LLM\'s judgment about which tool to call is sufficient.\n\n**Prompts are user-controlled.** The human explicitly triggers them. In Claude Desktop, they appear as slash commands. In other clients, they show up as menu items or quick actions. The user sees "Weekly Sales Report" and clicks it, entering a week number. There\'s no ambiguity about what will happen, no LLM judgment about which workflow to run. The user chose.\n\n**Resources are application-controlled.** The host application decides when to pull them into context. A resource might be a database schema, a configuration file, or a live dashboard. The application injects it into the conversation when relevant. For example, loading an API schema before a coding task. Neither the user nor the LLM explicitly requested it; the application determined it was needed.\n\nThis taxonomy tells you which primitive to use. If the LLM should decide, use a tool. If the user decides, use a prompt. If the application decides to use a resource.\n\nIn practice, many enterprise deployments add one more concept on top of these three primitives: **Tasks**. Tasks are not part of the base three-way split. They are an extension pattern for long-running operations such as scans, report generation, provisioning, or approvals. They are also the main exception to the normal stateless model. The request/response interface remains stateless, but the server explicitly persists task state and exposes progress or completion, rather than relying on sticky in-memory sessions.\n\nThis maps cleanly onto the Capability Square from the previous article — and prompts are where the split between the two human corners pays off the most:\n\n| Control Plane | Who Triggers at Runtime | Square Corner(s) | Strength |\n| --- | --- | --- | --- |\n| Tools | The LLM (model) | LLM | Intent interpretation, tool selection |\n| Prompts | The business user | Business Analyst (authors) + Business User (triggers) | Workflow knowledge encoded once, invoked many times |\n| Resources | The host application | Host + Server | Context management, data access |\n\nhttps://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fb7qf7k2pfbysr8vbutzw.png\n\nPrompts span both human corners of the square. The business analyst — the domain lead for workflow design — encodes an expert workflow into a prompt at design time. Engineers implement that workflow, and the platform team governs its deployment and control. The business user triggers it with one click at runtime. The prompt is literally the handoff artifact between the two humans: the analyst\'s workflow knowledge, packaged so a user doesn\'t need to recreate it every Monday morning. Tools, by contrast, sit under the LLM corner because the model\'s judgment determines when they are called. Resources sit at the boundary between the host application and the MCP server: the host decides _when_ to pull a resource into context, but the server _provides_ the content. This is the one primitive in which two actors collaborate without either human being directly in the loop, which partly explains why its ecosystem support lags behind that of tools and prompts. When all three control planes work together, the system covers every type of interaction: ad-hoc tasks (tools), structured workflows (prompts), and ambient context (resources). And because resource loading is application-dependent, the host may or may not inject the right resource at the right time — so an important role of prompt workflows is to explicitly load the relevant resources into context as part of the workflow definition. This ensures the LLM has the context it needs, regardless of what the host application decided to provide.\n\n## The Primitive You\'re Not Using\n\nMost MCP servers expose tools. A growing number expose resources. Almost none expose prompts.\n\nBrowse the MCP ecosystem, the tutorial repositories, the example servers, and the community showcases, and you\'ll find tool after tool after tool. Prompts are either absent entirely or limited to trivial "system message" wrappers that add no value beyond what the user could type themselves. The MCP official blog didn\'t publish its first [prompts-for-automation post](https://blog.modelcontextprotocol.io/posts/2025-07-29-prompts-for-automation/) until mid-2025, months after the protocol launched. The ecosystem followed suit: tools are easy to demo, prompts require thinking about workflows, and most tutorials took the easy path.\n\nThere\'s another reason prompts are underutilized: minimal SDK support. Most MCP SDKs treat prompts as simple message templates: you return a list of messages, and that\'s it. There\'s no built-in abstraction for multi-step workflows, data flow between steps, or hybrid execution where the server handles some steps and the LLM handles others. This is precisely why the PMCP (Pragmatic MCP) SDK added deep support for workflow prompts as an enterprise feature, the `SequentialWorkflow` abstraction we\'ll demonstrate in this article. Without SDK support, building reliable workflow prompts requires significant boilerplate that most teams don\'t invest in.\n\nThis is a missed opportunity. Prompts solve a reliability problem that tools cannot solve for known, repeatable workflows.\n\nConsider the gap. When you leave a multi-step workflow entirely to the LLM, using only tools, you\'re relying on instruction-only orchestration: the LLM reads the tool descriptions, figures out the right sequence, handles data flow between steps, and decides which computations to delegate versus attempt. In our experience building production MCP servers with the PMCP SDK, testing multi-step workflows like report generation, data pipelines, and incident response across multiple LLM models, instruction-only approaches typically achieve 60-70% compliance for complex workflows. That means 30-40% of the time, the LLM gets something wrong: a step out of order, a calculation it shouldn\'t have attempted, a variable lost between tool calls.\n\nNow compare hybrid execution, where the prompt defines the workflow, the server executes the deterministic steps, and the LLM fills in only where its language intelligence is needed. In the same test scenarios, hybrid execution typically achieves 85-95% compliance. These numbers come from internal benchmarks, not published studies, and will vary by model, workflow complexity, and domain, but the direction is consistent: reducing the LLM\'s decision space materially improves reliability.\n\nThe reason is straightforward. Prompts reduce the LLM\'s decision space and move **workflow state management** from the LLM\'s volatile context to explicit server-side execution state. In a multi-step tool chain, the LLM must track variables between calls, remember which step it\'s on, and pass results forward correctly, all in its context window, where information degrades with distance. In a workflow prompt, the server manages that state deterministically through request-scoped execution and, when necessary, explicitly persisted state. The LLM receives a pre-built plan with most steps already completed. It only needs to handle the parts that genuinely require language understanding: summarization, formatting, and inference.\n\nThe most common failure mode has a name: **calculation hallucination**. When an LLM sees a "calculate" tool and a "format" tool, it often skips the calculation tool to save a round trip and attempts the arithmetic itself. The result looks plausible, and the format is right; however, the numbers are wrong. Hybrid execution prevents this entirely: the server runs the calculation, the LLM never sees the raw numbers, and the result is correct by construction.\n\nIf you\'re measuring task completion across diverse requests, and you should be, as we argued in the tool design article, prompts are how you push completion rates from "usually works" to "reliably works" for your most common workflows.\n\n## From Protocol to Workflow\n\nAt the protocol level, a prompt is simple: the client calls `prompts/get` with a name and arguments, and the server returns a `GetPromptResult` containing a sequence of `PromptMessage` values. Each message has a role (`System` or `User`) and content (text, images, or embedded resources). The client uses these messages to populate the conversation and guide the LLM\'s response. Clients discover available prompts via `prompts/list` \\-\\- parallel to `tools/list` \\-\\- and present them to users as slash commands, menu items, or quick actions. The key difference from tools: the user explicitly selects them. There\'s no LLM reasoning about which prompt to invoke.\n\nAt this protocol level, prompts are message templates. Useful for setting up context, but not fundamentally different from what the user could type themselves. The real power emerges when you move from templates to workflows: multi-step processes in which data flows between steps, and the server executes what it can before handing off to the LLM. In a production deployment, that workflow engine should still fit the same remote, mostly stateless service model: deterministic steps execute within the request, and truly long-running work is broken out into explicit tasks.\n\n**An important distinction:** base MCP prompts are message templates. The server-executed workflow behavior shown below is a PMCP SDK abstraction built on top of prompts, tools, and resources. It uses the prompt protocol as the entry point, but adds a workflow engine that executes deterministic steps server-side before returning the message sequence to the client. This is not part of the MCP spec -- it\'s what a well-designed SDK can layer on top of it.\n\n## The Weekly Sales Report: One Click, Complex Result\n\nHere\'s the weekly sales report as a `SequentialWorkflow` \\-\\- a PMCP abstraction where each step can feed data into the next:\n\n```rust\nuse pmcp::server::workflow::{\n    dsl::{constant, field, from_step, prompt_arg},\n    SequentialWorkflow, ToolHandle, WorkflowStep,\n};\nuse serde_json::json;\n\n// SequentialWorkflow: a multi-step prompt where data flows between steps.\n// Unlike SyncPrompt (which builds static messages), SequentialWorkflow\n// orchestrates tool calls with typed data bindings between steps.\nlet sales_report = SequentialWorkflow::new(\n    "weekly_sales_report",\n    "Generate a formatted weekly sales report with trends and key metrics."\n)\n// Arguments: what the user provides when triggering this prompt\n.argument("week", "Week identifier (e.g., \'2026-W12\')", true)\n.argument("format", "Output format: summary or detailed", false)\n\n// Step 1: Query sales database (server executes -- deterministic)\n// The server calls query_database with constant + user-provided args.\n// No LLM needed: this is pure data retrieval.\n.step(\n    WorkflowStep::new("query_sales", ToolHandle::new("query_database"))\n        .arg("query_type", constant(json!("weekly_sales")))\n        .arg("week", prompt_arg("week"))\n        .bind("sales_data")  // output available as "sales_data" for later steps\n)\n\n// Step 2: Aggregate by category (server executes -- deterministic)\n// Uses the output from step 1. The server chains these automatically.\n.step(\n    WorkflowStep::new("aggregate", ToolHandle::new("aggregate_metrics"))\n        .arg("data", from_step("sales_data"))  // entire output from step 1\n        .arg("group_by", constant(json!("product_category")))\n        .bind("aggregated")\n)\n\n// Step 3: Calculate week-over-week trends (server executes -- deterministic)\n// This is the step that failed in our opening scenario when the LLM\n// tried to do it. The server handles the arithmetic correctly every time.\n.step(\n    WorkflowStep::new("calc_trends", ToolHandle::new("calculate_trends"))\n        .arg("current_week", from_step("aggregated"))\n        .arg("comparison", constant(json!("previous_week")))\n        .bind("trends")\n)\n\n// Step 4: Format as executive summary (LLM needed -- natural language)\n// This step requires intelligence: choosing which metrics to highlight,\n// writing prose summaries, deciding what "noteworthy" means.\n// The server provides the data and guidance; the LLM provides the writing.\n.step(\n    WorkflowStep::new("format_report", ToolHandle::new("format_output"))\n        .with_guidance(\n            "Format the aggregated data into an executive summary for week {week}.\\n\\\n             Highlight the top 3 performing categories and any \\\n             week-over-week trends that exceed 10% change.\\n\\\n             Use the report template for consistent formatting."\n        )\n        .with_resource("template://reports/weekly-sales")\n        .expect("Report template resource")\n        .arg("data", from_step("aggregated"))\n        .arg("trends", from_step("trends"))\n        .arg("format", prompt_arg("format"))\n        .bind("report")\n);\n```\n\nFollow the data flow through the DSL helpers. `prompt_arg("week")` pulls the user-provided week into step 1. `from_step("sales_data")` feeds step 1\'s entire output into step 2. `from_step("aggregated")` chains step 2\'s result into step 3. Each `bind("name")` names a step\'s output, allowing subsequent steps to reference it. The data flows forward through the workflow without any LLM involvement in the plumbing.\n\nSteps 1-3 are deterministic. The server executes them because each parameter can be resolved from prompt arguments (`prompt_arg`), constants (`constant`), or prior-step bindings (`from_step`). No judgment required. No natural language interpretation. Just data retrieval, aggregation, and arithmetic.\n\nStep 4 is where the server stops and hands off. The `format_output` tool needs LLM intelligence for natural language summarization: choosing which metrics to highlight, writing prose, deciding what "noteworthy" means. The server provides everything the LLM needs -- the aggregated data (from steps 1-3), the guidance (what to highlight), and a report template resource. The LLM\'s job is reduced to writing.\n\nRemember the opening scenario? The LLM tried to calculate week-over-week trends and got the arithmetic wrong—mixing up baselines and producing a report showing 340% growth in a category that actually declined. With this workflow, the server handles the arithmetic in step 3. Deterministically. Correctly. Every time. The LLM only enters at step 4, where its strength—natural language—is needed.\n\nRegistration ties the workflow to the server\'s existing tools:\n\n```rust\nServer::builder()\n    .tool("query_database", query_db_tool)\n    .tool("aggregate_metrics", aggregate_tool)\n    .tool("calculate_trends", trends_tool)\n    .tool("format_output", format_tool)\n    .resources(report_templates)\n    .prompt_workflow(sales_report)?  // validates bindings and registers as prompt\n    .build()?\n    .run_streamable_http("0.0.0.0:3000").await?;\n```\n\nNotice `.prompt_workflow()` validates the workflow\'s bindings at registration time. If you reference a binding that doesn\'t exist -- say, `from_step("sales_data")` with a typo -- you get an error at startup, not a runtime surprise when a user triggers the prompt. The tools you already built become the building blocks. The workflow just orchestrate them.\n\nThe user clicks one prompt. Three database operations, one aggregation, and one trend calculation happen server-side in milliseconds. The LLM receives the complete data and writes the summary. One click, complex result.\n\n## Partial Execution Plans: The Server Does What It Can\n\nWhen a user invokes the weekly sales report prompt, the server doesn\'t just return instructions. It returns a _partial execution plan_: a conversation trace showing what was already done and what remains.\n\nThe server executed steps 1-3 and embedded the actual results. Here\'s a simplified version of what the client LLM receives:\n\n```\nMessage 1 (User): "Generate weekly sales report for 2026-W12"\nMessage 2 (Assistant): "Plan: 1. Query sales DB  2. Aggregate  3. Calculate trends  4. Format"\nMessage 3 (Assistant): "Calling query_database..."\nMessage 4 (Tool Result): {"total_revenue": 284500, "transactions": 1247, ...}  ← PRE-EXECUTED by server\nMessage 5 (Assistant): "Calling aggregate_metrics..."\nMessage 6 (Tool Result): {"categories": [{"name": "Enterprise", "revenue": 142000}, ...]}  ← PRE-EXECUTED by server\nMessage 7 (Assistant): "Calling calculate_trends..."\nMessage 8 (Tool Result): {"enterprise": "+12%", "smb": "-3%", "startup": "+28%", ...}  ← PRE-EXECUTED by server\nMessage 9 (Assistant): "Format the aggregated data into an executive summary for 2026-W12..."\nMessage 10 (Resource): [weekly-sales template content]\n```\n\nMessages 1-8 are done. The tool results (Messages 4, 6, 8) were pre-executed by the server; the LLM didn\'t call those tools. It receives actual data —real revenue numbers, real category breakdowns, real trend percentages—not instructions to fetch that data. The server already queried the database, already aggregated, already calculated. The results are embedded in the conversation trace as if the tools had been called, but no LLM decision-making was involved.\n\nMessage 9 is the guidance for the remaining step. Message 10 is the resource template. The LLM\'s job is reduced to: take this data, follow this guidance, use this template, write prose. That\'s one decision (how to write the summary) instead of the dozens of decisions the instruction-only approach requires (which tools to call, in what order, how to handle errors, whether to do the arithmetic itself).\n\nThis is not a template. It\'s an execution plan where the server has already completed the deterministic portion. The distinction matters: a template says "do these steps." A partial execution plan says, "these steps are done and here are the results, now do the remaining steps." The LLM starts from step 4, not step 1.\n\nThis is the Capability Square operating at the workflow level. The **server** handles deterministic computation — its strength. The **LLM** handles natural language — its strength. The **business analyst** designed the workflow at design time, identifying which steps are deterministic and which require intelligence — their strength. And the **business user** invoked it at runtime with the specific parameters (the week, the service, the severity) that only they, living inside the working context, can provide — their strength. All four corners working together, not on a single tool call, but across an entire workflow.\n\nThe compliance improvement is consistent across our internal benchmarks. Instruction-only approaches, where the prompt simply says "follow these steps: 1. Query the sales DB, 2. Aggregate by category, 3. Calculate trends, 4. Format as a summary:" and leave every decision to the LLM. It might skip steps, reorder them, call different tools, or do the arithmetic itself (badly). Hybrid execution, where steps 1-3 are already done, and the LLM just needs to format, dramatically narrows the decision space. Far fewer decisions, far fewer failure points, far more reliable output.\n\nTest this with your own workflows. Take a 4-step process that your team runs weekly. Build it as an instruction-only prompt, then as a SequentialWorkflow with hybrid execution. Run both 20 times. The difference in successful completions will make the case.\n\n## Incident Response: When the Server Needs the LLM\n\nThe sales report workflow was mostly deterministic: three server-executed steps, one LLM step. But not every workflow splits that cleanly. Consider incident response, where the server gathers data, but the LLM needs to do the hard part of synthesis and recommendation.\n\nA 5-step incident response workflow:\n\n1. Check service status (server executes -- API call, deterministic)\n2. Pull recent error logs (server executes -- log query, deterministic)\n3. Correlate with recent deployments (server executes -- git/deploy history lookup, deterministic)\n4. Draft incident summary (LLM needed -- synthesis, pattern recognition, writing)\n5. Suggest mitigation steps (LLM needed -- reasoning about root cause, recommending actions)\n\nHere\'s the sketch -- not a full implementation, but enough to see the pattern:\n\n```rust\nSequentialWorkflow::new("incident_response", "Investigate and summarize a service incident")\n    .argument("service", "Service name or ID", true)\n    .argument("severity", "Severity level: P1, P2, P3", true)\n\n    // Steps 1-3: Server handles (deterministic data gathering)\n    .step(/* check_service_status -- server executes */)\n    .step(/* query_error_logs -- server executes */)\n    .step(/* check_recent_deploys -- server executes */)\n\n    // Steps 4-5: LLM handles (intelligence required)\n    .step(\n        WorkflowStep::new("draft_summary", ToolHandle::new("create_incident_report"))\n            .with_guidance(\n                "Analyze the service status, error logs, and deployment history.\\n\\\n                 Draft an incident summary for {service} at severity {severity}.\\n\\\n                 Include: timeline, affected systems, error patterns, and \\\n                 correlation with recent deployments."\n            )\n            .arg("status", from_step("service_status"))\n            .arg("logs", from_step("error_logs"))\n            .arg("deploys", from_step("deploy_history"))\n            .bind("summary")\n    )\n    .step(\n        WorkflowStep::new("suggest_mitigation", ToolHandle::new("recommend_actions"))\n            .with_guidance(\n                "Based on the incident summary, suggest 2-3 mitigation steps.\\n\\\n                 If the incident correlates with a recent deployment, include \\\n                 a rollback recommendation."\n            )\n            .arg("summary", from_step("summary"))\n            .bind("recommendations")\n    )\n```\n\nThe split is different from the sales report. The sales report was a 3-step server, 1-step LLM—mostly deterministic. The incident response is 3 steps for the server; 2 steps for the LLM. The analysis and recommendation require genuine intelligence. But the constant is the same: the server gathers all the data the LLM needs before handing off. The LLM doesn\'t have to figure out which APIs to call or which logs to check. It receives the service status, error logs, and deployment history, then applies its strengths: synthesis and reasoning.\n\nNotice that step 5 depends on step 4\'s output (`from_step("summary")`). The LLM executes both steps, but the data dependency is explicit in the workflow. The business analyst who designed this workflow decided that the mitigation suggestions should be based on the incident summary rather than the raw data. That\'s domain knowledge encoded in the workflow structure.\n\nThe partial execution plan for this workflow looks different, too. The server executes steps 1-3 and embeds the results. The LLM receives three steps\' worth of data and two steps\' worth of guidance. It drafts the summary, then uses that summary to suggest mitigations. The workflow is longer, the LLM does more, but the pattern is identical: the server handles the deterministic parts, the LLM handles the intelligence parts.\n\n## The Business Analyst\'s Playbook: Learning What Business Users Need\n\nThe weekly sales report and the incident response share something important: someone who understands the organization\'s workflows designed them. That someone is the **business analyst** — one of the two human corners of the Capability Square. In a strong enterprise setup, workflow design is domain-led, engineering-implemented, and platform-governed. The analyst shares a domain with the business users they\'re designing for, and their role doesn\'t end at tool design. It extends to workflow design: identifying which processes their business users run repeatedly, which steps are deterministic, and where the LLM\'s intelligence adds value.\n\nThe following diagram illustrates the benefits of adding workflow prompts to the MCP servers, as they dramatically reduce the effort for busy business people and significantly increase the completion rate of requests and their consistency:\n\nhttps://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2F7hdnkp197kssqj3xz7or.png\n\nHere\'s how to approach workflow prompt design in practice:\n\n1. **Observe your users.** What tasks do they repeat weekly? Monthly? What multi-step processes do they describe as "the usual"? These are prompt candidates. Every Monday, the sales team generates a weekly report. Every time there\'s an outage, the ops team runs the same diagnostic sequence. Every quarter, the finance team reconciles accounts. These are not ad hoc tasks, as they are workflows that run on a schedule, with the same steps and for the same reasons.\n\n2. **Identify the deterministic core.** For each repeating workflow, ask: which steps are always the same? Which steps require judgment? The always-the-same steps become server-executed workflow steps with `constant()` and `from_step()` bindings. The judgment steps become LLM-guided steps with `.with_guidance()`. The sales report\'s trend calculation is always the same arithmetic. The incident response\'s mitigation recommendation always requires judgment. The split is usually obvious once you look for it.\n\n3. **Start with one prompt.** Don\'t build 20 prompts. Build the one prompt that saves the most time for the most users. Measure its completion rate. Iterate. This mirrors the tool design advice from the [previous article](https://../01-tool-design/article.md): start with the 20% that serves 80%. For prompts, start with the one workflow your team runs most often.\n\n4. **Connect prompts to tools.** Prompts don\'t replace tools -- they orchestrate them. Your existing tools become the building blocks of workflow prompts. A SequentialWorkflow\'s steps call your tools via `ToolHandle`. The `query_database`, `aggregate_metrics`, and `calculate_trends` tools existed independently before the sales report workflow was built. The workflow just wired them together with data flow and execution order.\n\n5. **Iterate based on failure modes.** If the LLM consistently gets step N wrong, move step N to the server side. If the server can\'t handle step M because it requires judgment, move it to the LLM with clear guidance. The boundary between deterministic and intelligent steps is not fixed -- it\'s something you discover through observation and measurement.\n\nThe business analyst\'s role is to encode organizational knowledge into the MCP server — knowledge they are qualified to encode precisely because they share a domain with the business users who will invoke it. Tools encode individual capabilities. Prompts encode workflows — the sequences, the data flow, the decision about which steps need human-level intelligence and which don\'t. You know which workflows matter. You know which steps are deterministic. You know where the LLM\'s intelligence adds value. Encode that knowledge in prompts.\n\nTrack prompt invocation frequency and completion rates. A prompt that\'s invoked 50 times a week with 90% completion is saving your team hours of manual orchestration. A prompt that\'s never invoked is telling you something about your understanding of user needs. Both signals are useful -- one tells you what to optimize, the other tells you what to rethink.\n\nNone of this removes the need for security-by-design. Prompts are not "just UX." They package access to real systems and real workflows. The same controls apply here as in tools: per-request authn and authz, policy checks on downstream operations, audit logs, rate limits, secret isolation, and clear boundaries on which systems the workflow may touch. If a workflow includes code mode, the controls need to be tighter still: validate first, approve when the risk warrants it, and execute only within a constrained sandbox.\n\n## Resources: The Application-Controlled Plane\n\nWe\'ve covered tools (model-controlled) and prompts (user-controlled). The third primitive is resources: application-controlled context that the host application pulls into the conversation.\n\nResources are read-only reference material -- documentation, schemas, configuration, templates. They provide context that helps agents make better decisions. Where tools perform actions and prompts orchestrate workflows, resources serve information on request. They are passive: the server publishes them, and the client or prompt reads them when needed.\n\nHere\'s a resource using the PMCP SDK:\n\n```rust\nuse pmcp::{StaticResource, ResourceCollection};\n\n// Resources provide context data that agents can read before acting.\n// Unlike tools (which perform actions) or prompts (which orchestrate workflows),\n// resources are passive: they serve information on request.\nlet resources = ResourceCollection::new()\n    .add_resource(\n        StaticResource::new_text(\n            "docs://sales/schema",\n            "# Sales Database Schema\\n\\n\\\n             ## Tables\\n\\\n             - `orders`: order_id, customer_id, total, created_at\\n\\\n             - `products`: product_id, name, category, price\\n\\\n             - `customers`: customer_id, name, email, segment\\n\\n\\\n             ## Common Queries\\n\\\n             Weekly sales: GROUP BY date_trunc(\'week\', created_at)\\n\\\n             By category: JOIN products ON orders.product_id = products.product_id"\n        )\n        .with_name("Sales Database Schema")\n        .with_description(\n            "Database schema and common query patterns for the sales system. \\\n             Read this before constructing database queries."\n        )\n        .with_mime_type("text/markdown")\n    );\n```\n\nURI design matters. Use scheme prefixes to organize your resources: `docs://` for documentation, `config://` for configuration, `data://` for structured data, `template://` for report and output templates. The URI is a stable identifier that clients and prompts reference -- `docs://sales/schema` tells both humans and agents what they\'ll find before reading it.\n\nThe `.with_description()` call serves the same purpose as tool descriptions: it helps agents decide whether a resource is relevant before reading its content. A well-described resource lets an agent skip resources it doesn\'t need, reducing unnecessary context in the conversation.\n\nNotice how this connects to the weekly sales report prompt. In that workflow, step 4 used `.with_resource("template://reports/weekly-sales")` to fetch a report template and embed its content in the conversation trace. Resources provide the context that makes prompts more effective -- the LLM reads the schema to understand the data it\'s formatting, reads the template to follow the expected output structure. Resources and prompts are designed to work together.\n\n## The Ecosystem Reality Check\n\nResources are the least mature of the three MCP primitives in terms of client support. The spec defines them comprehensively -- annotations, subscriptions, URI templates, content types. The PMCP SDK supports them fully. But client implementations lag behind.\n\nMost MCP clients implement the `resources/list` and `resources/read` protocol operations, but the user experience varies significantly. Claude Desktop requires users to explicitly select resources from a list. There is no standardized resource picker UI across clients. And critically, resource access is a client-side operation -- the LLM has no built-in way to request a resource the way it can call a tool. Unless the client proactively injects resources into context, or the server wraps resource access as a tool, the LLM never sees them.\n\nThe gap between spec and ecosystem is real. The MCP specification describes a rich resource system with subscriptions for change notifications, URI templates for parameterized access, and annotations for priority and freshness signals. In practice, most clients implement the basics (list and read) and skip the rest. If you build a resource-heavy server today, you\'re building ahead of client support.\n\nThis doesn\'t mean you shouldn\'t build resources. It means you should build them with realistic expectations about how they\'ll be consumed today, while designing for where the ecosystem is headed. The patterns in the next section bridge the gap.\n\n## Pragmatic Bridge Patterns: Making Resources Work Today\n\nFour patterns let you get value from resources today, regardless of client support.\n\n**1\\. Wrap resources as tools** (most reliable today). Instead of serving a resource at `docs://sales/schema`, create a `get_sales_schema` tool that returns the same content. The LLM discovers and calls tools reliably -- this is the pragmatic path when you need agents to access reference data without depending on client resource support.\n\n```rust\n// Bridge pattern: expose resource content as a tool.\n// Until clients reliably handle resources, tools are the safe path.\n.tool("get_sales_schema", /* returns the same content as docs://sales/schema */)\n```\n\nThis isn\'t elegant, but it works everywhere. You can maintain both the resource (for clients that support it) and the tool wrapper (for clients that don\'t), serving the same underlying content through both channels.\n\n**2\\. Resource templates as parameterized access.** URI templates like `docs://reports/{report_type}` let the server generate URIs from parameters. When clients support resource templates, they can offer auto-complete for resource URIs -- the user types `docs://reports/` and sees available report types. This pattern is worth implementing now because it costs nothing extra and will work well as clients catch up.\n\n**3\\. Prompt-mediated resource loading.** This is the pattern we already saw: `.with_resource(uri)` in SequentialWorkflow steps. The server fetches the resource during prompt execution and embeds it in the conversation. This works today because it doesn\'t depend on client resource support at all -- the server handles the resource loading internally, and the client just sees the content in the prompt messages.\n\n**4\\. Subscribe and automatic injection** (future pattern). Clients can subscribe to resource changes via `resources/subscribe`. When the resource updates, the server sends a notification, and the client can refresh its context. This enables "always up-to-date context" without manual polling -- imagine an agent that automatically gets the latest API schema whenever it changes. This is where resources are headed. When client support catches up, automatic resource injection will make context management seamless.\n\nBuild your resources now. Use bridge patterns for today\'s clients. As the ecosystem matures, your resources will work natively -- and you\'ll already have the content, the URIs, and the descriptions in place.\n\n## Key Takeaways\n\n1. **Three control planes, three primitives.** Tools are model-controlled (the LLM decides). Prompts are user-controlled (the human decides). Resources are application-controlled (the host decides). Knowing which to use is the first design decision for any MCP capability.\n\n2. **Prompts solve the workflow reliability problem.** For known, repeatable workflows, hybrid execution -- where the server handles deterministic steps and the LLM handles intelligence -- consistently outperforms instruction-only orchestration in our benchmarks. Each party does what it\'s built for.\n\n3. **Partial execution plans are the key differentiator.** A prompt doesn\'t just send instructions. It returns a conversation trace with completed tool results, guidance for remaining steps, and embedded resource content. The LLM receives data, not directions.\n\n4. **The business analyst designs workflows, not just tools.** Observe which tasks your business users repeat. Identify the deterministic core. Package it as a SequentialWorkflow. Start with one prompt for your team\'s most common workflow and measure its completion rate. This is the handoff between the two human corners of the square: the analyst encodes once at design time, the business user triggers many times at runtime.\n\n5. **Resources are underbuilt but worth building.** Client support is thin today. Use bridge patterns -- wrap as tools, prompt-mediated loading -- for immediate value. Design for where the ecosystem is going, and your resources will be ready when clients catch up.\n\n6. **Tasks are the explicit exception to the stateless rule.** Most MCP interactions should stay stateless. When work outlives a single request, model it as a task with persisted state, progress tracking, and clear completion semantics instead of smuggling session state into the server process.\n\n7. **Prompts and tools are complementary.** Prompts orchestrate tools. Your existing tools become the building blocks of workflow prompts. Good tool design (from the [previous article](https://../01-tool-design/article.md)) makes good prompt design possible.\n\n8. **Measure prompt completion rates.** Track invocation frequency and success across diverse users. If a prompt is never invoked, your understanding of user needs may be wrong. If it fails consistently at step N, move step N server-side. Both signals guide iteration.', 'extras': {'signature': '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'}}

</details>

<details>
<summary>{'type': 'text', 'text': 'The `fastmcp.Client` class provides a programmatic interface for interacting with any MCP server. It handles protocol details and connection management automatically, letting you focus on the operations you want to perform. The FastMCP Client is designed for deterministic, controlled interactions rather than autonomous behavior, making it ideal for testing MCP servers during development, building deterministic applications that need reliable MCP interactions, and creating the foundation for agentic or LLM-based clients with structured, type-safe operations.\n\nThis is a programmatic client that requires explicit function calls and provides direct control over all MCP operations. Use it as a building block for higher-level systems.\n\n## Creating a Client\n\nYou provide a server source and the client automatically infers the appropriate transport mechanism.\n\n```python\nimport asyncio\nfrom fastmcp import Client, FastMCP\n\n# In-memory server (ideal for testing)\nserver = FastMCP("TestServer")\nclient = Client(server)\n\n# HTTP server\nclient = Client("https://example.com/mcp")\n\n# Local Python script\nclient = Client("my_mcp_server.py")\n\nasync def main():\n    async with client:\n        # Basic server interaction\n        await client.ping()\n\n        # List available operations\n        tools = await client.list_tools()\n        resources = await client.list_resources()\n        prompts = await client.list_prompts()\n\n        # Execute operations\n        result = await client.call_tool("example_tool", {"param": "value"})\n        print(result)\n\nasyncio.run(main())\n```\n\nAll client operations require using the `async with` context manager for proper connection lifecycle management.\n\n## Choosing a Transport\n\nThe client automatically selects a transport based on what you pass to it, but different transports have different characteristics that matter for your use case.\n\n**In-memory transport** connects directly to a FastMCP server instance within the same Python process. Use this for testing and development where you want to eliminate subprocess and network complexity. The server shares your process’s environment and memory space.\n\n```python\nfrom fastmcp import Client, FastMCP\n\nserver = FastMCP("TestServer")\nclient = Client(server)  # In-memory, no network or subprocess\n```\n\n**STDIO transport** launches a server as a subprocess and communicates through stdin/stdout pipes. This is the standard mechanism used by desktop clients like Claude Desktop. The subprocess runs in an isolated environment, so you must explicitly pass any environment variables the server needs.\n\n```python\nfrom fastmcp import Client\n\n# Simple inference from file path\nclient = Client("my_server.py")\n\n# With explicit environment configuration\nclient = Client("my_server.py", env={"API_KEY": "secret"})\n```\n\n**HTTP transport** connects to servers running as web services. Use this for production deployments where the server runs independently and manages its own lifecycle.\n\n```python\nfrom fastmcp import Client\n\nclient = Client("https://api.example.com/mcp")\n```\n\n## Connection Lifecycle\n\nThe client uses context managers for connection management. When you enter the context, the client establishes a connection and performs an MCP initialization handshake with the server. This handshake exchanges capabilities, server metadata, and instructions.\n\n```python\nfrom fastmcp import Client, FastMCP\n\nmcp = FastMCP(name="MyServer", instructions="Use the greet tool to say hello!")\n\n@mcp.tool\ndef greet(name: str) -> str:\n    """Greet a user by name."""\n    return f"Hello, {name}!"\n\nasync with Client(mcp) as client:\n    # Initialization already happened automatically\n    print(f"Server: {client.initialize_result.serverInfo.name}")\n    print(f"Instructions: {client.initialize_result.instructions}")\n    print(f"Capabilities: {client.initialize_result.capabilities.tools}")\n```\n\nFor advanced scenarios where you need precise control over when initialization happens, disable automatic initialization and call `initialize()` manually:\n\n```python\nfrom fastmcp import Client\n\nclient = Client("my_mcp_server.py", auto_initialize=False)\n\nasync with client:\n    # Connection established, but not initialized yet\n    print(f"Connected: {client.is_connected()}")\n    print(f"Initialized: {client.initialize_result is not None}")  # False\n\n    # Initialize manually with custom timeout\n    result = await client.initialize(timeout=10.0)\n    print(f"Server: {result.serverInfo.name}")\n\n    # Now ready for operations\n    tools = await client.list_tools()\n```\n\n## Operations\n\nFastMCP clients interact with three types of server components.\n\n**Tools** are server-side functions that the client can execute with arguments. Call them with `call_tool()` and receive structured results.\n\n```python\nasync with client:\n    tools = await client.list_tools()\n    result = await client.call_tool("multiply", {"a": 5, "b": 3})\n    print(result.data)  # 15\n```\n\n**Resources** are data sources that the client can read, either static or templated. Access them with `read_resource()` using URIs.\n\n```python\nasync with client:\n    resources = await client.list_resources()\n    content = await client.read_resource("file:///config/settings.json")\n    print(content[0].text)\n```\n\n**Prompts** are reusable message templates that can accept arguments. Retrieve rendered prompts with `get_prompt()`.\n\n```python\nasync with client:\n    prompts = await client.list_prompts()\n    messages = await client.get_prompt("analyze_data", {"data": [1, 2, 3]})\n    print(messages.messages)\n```', 'extras': {'signature': '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'}}</summary>

{'type': 'text', 'text': 'The `fastmcp.Client` class provides a programmatic interface for interacting with any MCP server. It handles protocol details and connection management automatically, letting you focus on the operations you want to perform. The FastMCP Client is designed for deterministic, controlled interactions rather than autonomous behavior, making it ideal for testing MCP servers during development, building deterministic applications that need reliable MCP interactions, and creating the foundation for agentic or LLM-based clients with structured, type-safe operations.\n\nThis is a programmatic client that requires explicit function calls and provides direct control over all MCP operations. Use it as a building block for higher-level systems.\n\n## Creating a Client\n\nYou provide a server source and the client automatically infers the appropriate transport mechanism.\n\n```python\nimport asyncio\nfrom fastmcp import Client, FastMCP\n\n# In-memory server (ideal for testing)\nserver = FastMCP("TestServer")\nclient = Client(server)\n\n# HTTP server\nclient = Client("https://example.com/mcp")\n\n# Local Python script\nclient = Client("my_mcp_server.py")\n\nasync def main():\n    async with client:\n        # Basic server interaction\n        await client.ping()\n\n        # List available operations\n        tools = await client.list_tools()\n        resources = await client.list_resources()\n        prompts = await client.list_prompts()\n\n        # Execute operations\n        result = await client.call_tool("example_tool", {"param": "value"})\n        print(result)\n\nasyncio.run(main())\n```\n\nAll client operations require using the `async with` context manager for proper connection lifecycle management.\n\n## Choosing a Transport\n\nThe client automatically selects a transport based on what you pass to it, but different transports have different characteristics that matter for your use case.\n\n**In-memory transport** connects directly to a FastMCP server instance within the same Python process. Use this for testing and development where you want to eliminate subprocess and network complexity. The server shares your process’s environment and memory space.\n\n```python\nfrom fastmcp import Client, FastMCP\n\nserver = FastMCP("TestServer")\nclient = Client(server)  # In-memory, no network or subprocess\n```\n\n**STDIO transport** launches a server as a subprocess and communicates through stdin/stdout pipes. This is the standard mechanism used by desktop clients like Claude Desktop. The subprocess runs in an isolated environment, so you must explicitly pass any environment variables the server needs.\n\n```python\nfrom fastmcp import Client\n\n# Simple inference from file path\nclient = Client("my_server.py")\n\n# With explicit environment configuration\nclient = Client("my_server.py", env={"API_KEY": "secret"})\n```\n\n**HTTP transport** connects to servers running as web services. Use this for production deployments where the server runs independently and manages its own lifecycle.\n\n```python\nfrom fastmcp import Client\n\nclient = Client("https://api.example.com/mcp")\n```\n\n## Connection Lifecycle\n\nThe client uses context managers for connection management. When you enter the context, the client establishes a connection and performs an MCP initialization handshake with the server. This handshake exchanges capabilities, server metadata, and instructions.\n\n```python\nfrom fastmcp import Client, FastMCP\n\nmcp = FastMCP(name="MyServer", instructions="Use the greet tool to say hello!")\n\n@mcp.tool\ndef greet(name: str) -> str:\n    """Greet a user by name."""\n    return f"Hello, {name}!"\n\nasync with Client(mcp) as client:\n    # Initialization already happened automatically\n    print(f"Server: {client.initialize_result.serverInfo.name}")\n    print(f"Instructions: {client.initialize_result.instructions}")\n    print(f"Capabilities: {client.initialize_result.capabilities.tools}")\n```\n\nFor advanced scenarios where you need precise control over when initialization happens, disable automatic initialization and call `initialize()` manually:\n\n```python\nfrom fastmcp import Client\n\nclient = Client("my_mcp_server.py", auto_initialize=False)\n\nasync with client:\n    # Connection established, but not initialized yet\n    print(f"Connected: {client.is_connected()}")\n    print(f"Initialized: {client.initialize_result is not None}")  # False\n\n    # Initialize manually with custom timeout\n    result = await client.initialize(timeout=10.0)\n    print(f"Server: {result.serverInfo.name}")\n\n    # Now ready for operations\n    tools = await client.list_tools()\n```\n\n## Operations\n\nFastMCP clients interact with three types of server components.\n\n**Tools** are server-side functions that the client can execute with arguments. Call them with `call_tool()` and receive structured results.\n\n```python\nasync with client:\n    tools = await client.list_tools()\n    result = await client.call_tool("multiply", {"a": 5, "b": 3})\n    print(result.data)  # 15\n```\n\n**Resources** are data sources that the client can read, either static or templated. Access them with `read_resource()` using URIs.\n\n```python\nasync with client:\n    resources = await client.list_resources()\n    content = await client.read_resource("file:///config/settings.json")\n    print(content[0].text)\n```\n\n**Prompts** are reusable message templates that can accept arguments. Retrieve rendered prompts with `get_prompt()`.\n\n```python\nasync with client:\n    prompts = await client.list_prompts()\n    messages = await client.get_prompt("analyze_data", {"data": [1, 2, 3]})\n    print(messages.messages)\n```', 'extras': {'signature': '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'}}

</details>

<details>
<summary>{'type': 'text', 'text': '**FastMCP is the standard framework for building MCP applications.** The Model Context Protocol (MCP) connects LLMs to tools and data. FastMCP gives you everything you need to go from prototype to production — build servers that expose capabilities, connect clients to any MCP service, and give your tools interactive UIs:\n\n```python\nfrom fastmcp import FastMCP\n\nmcp = FastMCP("Demo 🚀")\n\n@mcp.tool\ndef add(a: int, b: int) -> int:\n    """Add two numbers"""\n    return a + b\n\nif __name__ == "__main__":\n    mcp.run()\n```\n\n## Move Fast and Make Things\n\nThe Model Context Protocol (MCP) lets you give agents access to your tools and data. FastMCP handles all of it. Declare a tool with a Python function, and the schema, validation, and documentation are generated automatically. Connect to a server with a URL, and transport negotiation, authentication, and protocol lifecycle are managed for you.\n\nFastMCP has three pillars:\n\n*   **Servers:** Wrap your Python functions into MCP-compliant tools, resources, and prompts.\n*   **Clients:** Connect to any server with full protocol support.\n*   **Apps:** Give your tools interactive UIs rendered directly in the conversation.\n\n### Using FastMCP to Connect a Client\n\nYou can use FastMCP to connect to an MCP server and call tools programmatically:\n\n```python\nimport asyncio\nfrom fastmcp import Client\n\nasync def main():\n    async with Client("https://gofastmcp.com/mcp") as client:\n        result = await client.call_tool(\n            name="search_fast_mcp",\n            arguments={"query": "deploy a FastMCP server"}\n        )\n    print(result)\n\nasyncio.run(main())\n```', 'extras': {'signature': 'ErILCq8LAQw51seHONUmyh0CoXDYBTbm4/JI6bzBKnn2oPoS7x8HMZSCxF3p02/gMs55sd25By6F0kWeR9MKVPEO4YcP9eFX2XmDvw9WLC78JtzbLYB4j8rV/1QGqT+ZrUwni0OuoWDS+uaOGWCAcCVUw4FRzIGlJI6Z3Ln+OogSWBR8yQstwsJYucJjQ1OxgifAWhMkcX6tPspH/0kYpMWTBs17Sv3E6xk1id9EiuBWhVHKLr1fdjlN+C3ObPfluBFOne/exbopMCFjyZ/BGfmt79f7FPJ4899ttXmYtOPMSFhZWkgCojjK0vJxhBXRcI4qNNzErAsrKQCg5+e843l3R1KL1rqquO+PFnhbuooLx/Bj7Gs42Osa8YdURY8TR+G/dFzxcyOIYOmMR+TF2bs26KnBJ9OzpQmv56zBx2lkS12UTphO45DEXQWyJCu05TcOqUj/p+oXRmdrG3WXHh/VdN0fSbrii9JUbxp4Wp/g0h0lel/U6pYTh/IX10U6f75xL4JuglBTgvu3HE29ZA43lRXkhmNsW9N1aWSEK3oaDlvvQmCc307W//PYs+V+lnxPOUK+8JlYUoOIMS5ZPsH5BhiOVHo0DqcY+XDwT437/qx8P2m2ggu+lN6+2Fjon3qrPg3TPabW/ZuokB8SwsY3jSISytUNVZdQ0Xc+YsvsxGhRMJqNzJhLxBKwWGgI1QeyraAziQDqCOMFRf1Iqm30NpdChm73wCrP/CQ81xiwPmoPyRFdffE/aAr4gqbChbjRtZ+EUL0icE6znLYzVNm2BhVrEPPf+qwdIubql6CafVgJBjGm9g3VTJjqC/V8JqDPbUtwVHdymEmA5cgpSjGvR20dhUbRtuoSg/QXhoE9TOsIfVJpdi7xCafKj7Mf9pbrQl6o6ELUw/2PdfR2g24R3ouiA+8beEnRCIULGBFwb2P+iQlV3zDC39otwGVu35Xgwwd35s84c7XIFSAr7Mfkwyv4eJL3TLMHLIXx+E79ErAtfJHq4q10F5J3TNjVOD1oXNcF3GJWHHkU3Q/KxP+xWYh0CFIS7mkd6608hT0wNpuqsTtNHjxzXcJAIEEbyg5YERmOLfWwihmTsk1sWRmYJgoQMrva0AqNM861BTAWKlLA4M4cHBu83xRIMdwNkzHBfzllDhqa5Aajv582NJKNGSVB7JDZtKppSKKM/yiapd87itU2S88B++YuAbqYxT0RkSN3hy30zNCgzBAGy0C7BWu5Z5joRjw2hfWGg5adZT/1aJ1vfB4GHTx3vggYkJ+ySRLv3het8MiYiGqy333fHkOFIy94jo/P0NNDjUKnwTco3qx3eDrDOfWcatcWL/vu0HbX5iD04K6yuqJFAqGhYq7+3YLtd/lbYr3ZrGtQHB4cZQUfRy/o35hcyBhzV6NKEMWh0Eilwl2jtljFhMG7FwunovvOlxsFsjkj+HWSImDXDi3dKfVy3Wi7TnhvzOXhSxJHSpi+TVAU2dEjZtRbCScWSiv/rFyRlDjFm60ntK8A9RC/tRVxz2BpYSHKEG5n+j8HEYQpLYmUOfDDnP+BJNrG+X+Z9uJPF0xUTrzoPxHf68JTNpuYH7rJCLxG9DwBIC1UgOgRXtZxeAfg1zs/JfaYc59uEZc7pasI31jcZVF1pS9uxzQT4J/gtiud1etulvGhX5Z6LBTYNyHy+8icFdq+0ooDRAgISBMlJri14azkzkIqvugUYW1b9pfkMlnQilAmt3hng1KgWhVYGV6OO4gGGYjwZEXboiGy4/Pj6Mj1UcCtt6EnjJHU08p9yOwu9fvmnHaO2Xy8rSYZvNQjbM3xfPQyYUY031KYyIOECae/3zhZO/Ved27+xN0kaV+a0Ws1nWgJlVHtuac9B5PWIfPcC6TpTXFPLscwdtDsN6DaEnRTBBsnEagUaElF7jFNRV/dEWo4DJ5HKGXzmb0DKETd'}}</summary>

{'type': 'text', 'text': '**FastMCP is the standard framework for building MCP applications.** The Model Context Protocol (MCP) connects LLMs to tools and data. FastMCP gives you everything you need to go from prototype to production — build servers that expose capabilities, connect clients to any MCP service, and give your tools interactive UIs:\n\n```python\nfrom fastmcp import FastMCP\n\nmcp = FastMCP("Demo 🚀")\n\n@mcp.tool\ndef add(a: int, b: int) -> int:\n    """Add two numbers"""\n    return a + b\n\nif __name__ == "__main__":\n    mcp.run()\n```\n\n## Move Fast and Make Things\n\nThe Model Context Protocol (MCP) lets you give agents access to your tools and data. FastMCP handles all of it. Declare a tool with a Python function, and the schema, validation, and documentation are generated automatically. Connect to a server with a URL, and transport negotiation, authentication, and protocol lifecycle are managed for you.\n\nFastMCP has three pillars:\n\n*   **Servers:** Wrap your Python functions into MCP-compliant tools, resources, and prompts.\n*   **Clients:** Connect to any server with full protocol support.\n*   **Apps:** Give your tools interactive UIs rendered directly in the conversation.\n\n### Using FastMCP to Connect a Client\n\nYou can use FastMCP to connect to an MCP server and call tools programmatically:\n\n```python\nimport asyncio\nfrom fastmcp import Client\n\nasync def main():\n    async with Client("https://gofastmcp.com/mcp") as client:\n        result = await client.call_tool(\n            name="search_fast_mcp",\n            arguments={"query": "deploy a FastMCP server"}\n        )\n    print(result)\n\nasyncio.run(main())\n```', 'extras': {'signature': 'ErILCq8LAQw51seHONUmyh0CoXDYBTbm4/JI6bzBKnn2oPoS7x8HMZSCxF3p02/gMs55sd25By6F0kWeR9MKVPEO4YcP9eFX2XmDvw9WLC78JtzbLYB4j8rV/1QGqT+ZrUwni0OuoWDS+uaOGWCAcCVUw4FRzIGlJI6Z3Ln+OogSWBR8yQstwsJYucJjQ1OxgifAWhMkcX6tPspH/0kYpMWTBs17Sv3E6xk1id9EiuBWhVHKLr1fdjlN+C3ObPfluBFOne/exbopMCFjyZ/BGfmt79f7FPJ4899ttXmYtOPMSFhZWkgCojjK0vJxhBXRcI4qNNzErAsrKQCg5+e843l3R1KL1rqquO+PFnhbuooLx/Bj7Gs42Osa8YdURY8TR+G/dFzxcyOIYOmMR+TF2bs26KnBJ9OzpQmv56zBx2lkS12UTphO45DEXQWyJCu05TcOqUj/p+oXRmdrG3WXHh/VdN0fSbrii9JUbxp4Wp/g0h0lel/U6pYTh/IX10U6f75xL4JuglBTgvu3HE29ZA43lRXkhmNsW9N1aWSEK3oaDlvvQmCc307W//PYs+V+lnxPOUK+8JlYUoOIMS5ZPsH5BhiOVHo0DqcY+XDwT437/qx8P2m2ggu+lN6+2Fjon3qrPg3TPabW/ZuokB8SwsY3jSISytUNVZdQ0Xc+YsvsxGhRMJqNzJhLxBKwWGgI1QeyraAziQDqCOMFRf1Iqm30NpdChm73wCrP/CQ81xiwPmoPyRFdffE/aAr4gqbChbjRtZ+EUL0icE6znLYzVNm2BhVrEPPf+qwdIubql6CafVgJBjGm9g3VTJjqC/V8JqDPbUtwVHdymEmA5cgpSjGvR20dhUbRtuoSg/QXhoE9TOsIfVJpdi7xCafKj7Mf9pbrQl6o6ELUw/2PdfR2g24R3ouiA+8beEnRCIULGBFwb2P+iQlV3zDC39otwGVu35Xgwwd35s84c7XIFSAr7Mfkwyv4eJL3TLMHLIXx+E79ErAtfJHq4q10F5J3TNjVOD1oXNcF3GJWHHkU3Q/KxP+xWYh0CFIS7mkd6608hT0wNpuqsTtNHjxzXcJAIEEbyg5YERmOLfWwihmTsk1sWRmYJgoQMrva0AqNM861BTAWKlLA4M4cHBu83xRIMdwNkzHBfzllDhqa5Aajv582NJKNGSVB7JDZtKppSKKM/yiapd87itU2S88B++YuAbqYxT0RkSN3hy30zNCgzBAGy0C7BWu5Z5joRjw2hfWGg5adZT/1aJ1vfB4GHTx3vggYkJ+ySRLv3het8MiYiGqy333fHkOFIy94jo/P0NNDjUKnwTco3qx3eDrDOfWcatcWL/vu0HbX5iD04K6yuqJFAqGhYq7+3YLtd/lbYr3ZrGtQHB4cZQUfRy/o35hcyBhzV6NKEMWh0Eilwl2jtljFhMG7FwunovvOlxsFsjkj+HWSImDXDi3dKfVy3Wi7TnhvzOXhSxJHSpi+TVAU2dEjZtRbCScWSiv/rFyRlDjFm60ntK8A9RC/tRVxz2BpYSHKEG5n+j8HEYQpLYmUOfDDnP+BJNrG+X+Z9uJPF0xUTrzoPxHf68JTNpuYH7rJCLxG9DwBIC1UgOgRXtZxeAfg1zs/JfaYc59uEZc7pasI31jcZVF1pS9uxzQT4J/gtiud1etulvGhX5Z6LBTYNyHy+8icFdq+0ooDRAgISBMlJri14azkzkIqvugUYW1b9pfkMlnQilAmt3hng1KgWhVYGV6OO4gGGYjwZEXboiGy4/Pj6Mj1UcCtt6EnjJHU08p9yOwu9fvmnHaO2Xy8rSYZvNQjbM3xfPQyYUY031KYyIOECae/3zhZO/Ved27+xN0kaV+a0Ws1nWgJlVHtuac9B5PWIfPcC6TpTXFPLscwdtDsN6DaEnRTBBsnEagUaElF7jFNRV/dEWo4DJ5HKGXzmb0DKETd'}}

</details>

<details>
<summary>{'type': 'text', 'text': '# What are MCP prompts?\n\nMCP prompts are reusable, structured message templates exposed by MCP servers to guide interactions with agents. Unlike tools (which execute logic) or resources (which provide read-only data), prompts return a predefined list of messages meant to initiate consistent model behavior.\n\nPrompts are declarative, composable, and designed for user-initiated workflows, such as:\n\n- Slash commands or quick actions triggered via UI\n- Task-specific interactions, like summarization or code explanation\n\nYou can use prompts when you want to define how users engage with the model but not to perform logic or to serve contextual data.\n\n## Prompt structure\n\nA prompt is a named, parameterized template. It defines:\n\n- A `name` (a unique identifier)\n- An optional `description`\n- An optional list of structured `arguments`\n\n```json\n{\n  "name": "summarize-errors",\n  "description": "Summarize recent error logs",\n  "arguments": [\n    {\n      "name": "logUri",\n      "description": "URI of the log resource",\n      "required": true\n    }\n  ]\n}\n```\n\nThe server exposes prompts via `prompts/list` and provides message content on `prompts/get`.\n\n### Discovering prompts\n\nClients use `prompts/list` to fetch available prompt definitions:\n\n```json\n{\n  "method": "prompts/list"\n}\n```\n\nThe response includes a list of prompts:\n\n```json\n{\n  "prompts": [\n    {\n      "name": "explain-code",\n      "description": "Explain how a function works",\n      "arguments": [{ "name": "code", "required": true }]\n    }\n  ]\n}\n```\n\n### Using prompts\n\nTo use a prompt, clients call `prompts/get` with a prompt `name` and `arguments`:\n\n```json\n{\n  "method": "prompts/get",\n  "params": {\n    "name": "explain-code",\n    "arguments": {\n      "code": "def hello(): print(\'hi\')"\n    }\n  }\n}\n```\n\nThe server responds with a `messages[]` array, ready to send to the model:\n\n```json\n{\n  "description": "Explain how a function works",\n  "messages": [\n    {\n      "role": "user",\n      "content": {\n        "type": "text",\n        "text": "Explain this Python code:\\n\\ndef hello(): print(\'hi\')"\n      }\n    }\n  ]\n}\n```\n\n## Defining and serving prompts in Python\n\nThe following example defines a simple MCP prompt called `git-commit` that helps users generate commit messages from change descriptions.\n\n```python\nfrom mcp.server import Server, stdio\nimport mcp.types as types\n\nimport asyncio\n\napp = Server("git-prompts-server")\n\n@app.list_prompts()\nasync def list_prompts() -> list[types.Prompt]:\n    return [\n        types.Prompt(\n            name="git-commit",\n            description="Generate a Git commit message from a code diff or change summary",\n            arguments=[\n                types.PromptArgument(\n                    name="changes",\n                    description="Code diff or explanation of the changes made",\n                    required=True\n                )\n            ]\n        )\n    ]\n\n@app.get_prompt()\nasync def get_prompt(name: str, arguments: dict[str, str]) -> types.GetPromptResult:\n    if name != "git-commit":\n        raise ValueError("Unknown prompt")\n\n    changes = arguments.get("changes", "")\n\n    return types.GetPromptResult(\n        messages=[\n            types.PromptMessage(\n                role="user",\n                content=types.TextContent(\n                    type="text",\n                    text=(\n                        "Generate a Git commit message summarizing these changes:\\n\\n"\n                        f"{changes}"\n                    )\n                )\n            )\n        ]\n    )\n```\n\nIn this example, we:\n\n- **Register a static prompt** named `git-commit` with a human-readable description and a required `changes` argument.\n- **Expose metadata via `@list_prompts`** so UIs and clients can discover the prompt.\n- **Implement prompt generation via `@get_prompt`**, which creates a single message that asks the agent to produce a commit message based on input.\n- **Avoid side effects**, as the server does not evaluate or format the response but it does structure a message.\n\n## Best practices and pitfalls to avoid\n\nHere are some best practices for implementing MCP prompts:\n\n- Use clear, actionable names (for example, `summarize-errors`, not `get-summarized-error-log-output`).\n- Validate all required arguments up front.\n- Keep prompts deterministic and stateless (using the same input should produce the same output).\n- Embed resources directly, if needed, for model context.\n- Provide concise descriptions to improve UI discoverability.\n\nWhen implementing MCP prompts, avoid the following common mistakes:\n\n- Allowing missing or malformed arguments\n- Using vague or overly long prompt names\n- Passing oversized inputs (such as full files or large diffs)\n- Failing to sanitize non-UTF-8 or injection-prone strings\n\n### Prompts vs tools vs resources\n\nThe table below compares the three core primitives in MCP:\n\n| Feature | Prompts | Tools | Resources |\n| --- | --- | --- | --- |\n| **Purpose** | Guide model interaction | Execute logic with side effects | Provide structured read-only data |\n| **Triggered by** | User or UI | Agent or client (`tools/call`) | Agent or client (`resources/read`) |\n| **Behavior** | Returns `messages[]` | Runs a function; returns a result | Returns static or dynamic content |\n| **Side effects** | None | Yes (I/O, API calls, mutations) | None |\n| **Composition** | Can embed arguments and resources | Accepts structured input | URI-scoped, optionally templated |\n| **Use cases** | Summarization, Q&A, message templates | File writing, API calls, workflows | Logs, config files, external data |\n\n## Practical implementation example\n\nMCP prompts are a powerful way to define reusable templates that combine context from your application with instructions for the LLM. Here’s how to implement a prompt using the TypeScript SDK.\n\nThis example creates a WhatsApp chat summarization prompt that retrieves chat data and formats it for the LLM:\n\n```typescript\nmcpServer.prompt(\n  "whatsapp_chat_summarizer",\n  "Summarize WhatsApp chat and provide insights",\n  {\n    chatName: z.string().describe("Name of the WhatsApp chat to summarize"),\n  },\n  async (args) => {\n    const { chatName = "" } = args;\n\n    // Find the chat by name\n    // A real implementation would be more robust\n    const targetChat = await chatService.findChatByName(chatName);\n\n    // Get recent messages for analysis\n    const messages = await messageService.getMessages(targetChat.id);\n\n    const promptText = `Analyze this WhatsApp chat data for insights:\n\nChat Information:\n- Chat Name: ${targetChat.name}\n- Chat Type: ${targetChat.isGroup ? "Group Chat" : "Individual Chat"}\n- Analysis Type: summary\n\nAnalysis Focus:\nProvide a comprehensive overview including key topics, sentiment, and notable patterns.\n\nRecent Messages (${messages.length} messages):\n${messages.map((msg) => msg._serializedContent).join("\\n")}\n\nPlease provide a detailed summary.`;\n\n    return {\n      description: `Summary of WhatsApp chat: ${targetChat.name}`,\n      messages: [\n        {\n          role: "user",\n          content: {\n            type: "text",\n            text: promptText,\n          },\n        },\n      ],\n    };\n  },\n);\n```\n\nThis defines a prompt called `whatsapp_chat_summarizer` that takes a `chatName` argument and generates a formatted prompt with the chat data.\n\n### How prompts work in practice\n\nThe LLM client presents a list of available prompts to the user, who can then select one to use. When the user selects a prompt with arguments, the client should display a modal or form allowing the user to fill in the required arguments.\n\nOnce the user submits the form, the MCP client sends a `prompts/get` request to the MCP server with the selected prompt and its arguments. The MCP server adds the relevant context to the prompt (in this case, the WhatsApp chat data) and returns the formatted messages to the MCP client. The client can then send these messages to the LLM for processing.\n\nThis is especially useful for repetitive tasks where a user needs to combine tool call results with a complex prompt. If you can anticipate the user’s needs, you can define a prompt that combines the necessary context and tool calls into a single reusable template.', 'extras': {'signature': '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'}}</summary>

{'type': 'text', 'text': '# What are MCP prompts?\n\nMCP prompts are reusable, structured message templates exposed by MCP servers to guide interactions with agents. Unlike tools (which execute logic) or resources (which provide read-only data), prompts return a predefined list of messages meant to initiate consistent model behavior.\n\nPrompts are declarative, composable, and designed for user-initiated workflows, such as:\n\n- Slash commands or quick actions triggered via UI\n- Task-specific interactions, like summarization or code explanation\n\nYou can use prompts when you want to define how users engage with the model but not to perform logic or to serve contextual data.\n\n## Prompt structure\n\nA prompt is a named, parameterized template. It defines:\n\n- A `name` (a unique identifier)\n- An optional `description`\n- An optional list of structured `arguments`\n\n```json\n{\n  "name": "summarize-errors",\n  "description": "Summarize recent error logs",\n  "arguments": [\n    {\n      "name": "logUri",\n      "description": "URI of the log resource",\n      "required": true\n    }\n  ]\n}\n```\n\nThe server exposes prompts via `prompts/list` and provides message content on `prompts/get`.\n\n### Discovering prompts\n\nClients use `prompts/list` to fetch available prompt definitions:\n\n```json\n{\n  "method": "prompts/list"\n}\n```\n\nThe response includes a list of prompts:\n\n```json\n{\n  "prompts": [\n    {\n      "name": "explain-code",\n      "description": "Explain how a function works",\n      "arguments": [{ "name": "code", "required": true }]\n    }\n  ]\n}\n```\n\n### Using prompts\n\nTo use a prompt, clients call `prompts/get` with a prompt `name` and `arguments`:\n\n```json\n{\n  "method": "prompts/get",\n  "params": {\n    "name": "explain-code",\n    "arguments": {\n      "code": "def hello(): print(\'hi\')"\n    }\n  }\n}\n```\n\nThe server responds with a `messages[]` array, ready to send to the model:\n\n```json\n{\n  "description": "Explain how a function works",\n  "messages": [\n    {\n      "role": "user",\n      "content": {\n        "type": "text",\n        "text": "Explain this Python code:\\n\\ndef hello(): print(\'hi\')"\n      }\n    }\n  ]\n}\n```\n\n## Defining and serving prompts in Python\n\nThe following example defines a simple MCP prompt called `git-commit` that helps users generate commit messages from change descriptions.\n\n```python\nfrom mcp.server import Server, stdio\nimport mcp.types as types\n\nimport asyncio\n\napp = Server("git-prompts-server")\n\n@app.list_prompts()\nasync def list_prompts() -> list[types.Prompt]:\n    return [\n        types.Prompt(\n            name="git-commit",\n            description="Generate a Git commit message from a code diff or change summary",\n            arguments=[\n                types.PromptArgument(\n                    name="changes",\n                    description="Code diff or explanation of the changes made",\n                    required=True\n                )\n            ]\n        )\n    ]\n\n@app.get_prompt()\nasync def get_prompt(name: str, arguments: dict[str, str]) -> types.GetPromptResult:\n    if name != "git-commit":\n        raise ValueError("Unknown prompt")\n\n    changes = arguments.get("changes", "")\n\n    return types.GetPromptResult(\n        messages=[\n            types.PromptMessage(\n                role="user",\n                content=types.TextContent(\n                    type="text",\n                    text=(\n                        "Generate a Git commit message summarizing these changes:\\n\\n"\n                        f"{changes}"\n                    )\n                )\n            )\n        ]\n    )\n```\n\nIn this example, we:\n\n- **Register a static prompt** named `git-commit` with a human-readable description and a required `changes` argument.\n- **Expose metadata via `@list_prompts`** so UIs and clients can discover the prompt.\n- **Implement prompt generation via `@get_prompt`**, which creates a single message that asks the agent to produce a commit message based on input.\n- **Avoid side effects**, as the server does not evaluate or format the response but it does structure a message.\n\n## Best practices and pitfalls to avoid\n\nHere are some best practices for implementing MCP prompts:\n\n- Use clear, actionable names (for example, `summarize-errors`, not `get-summarized-error-log-output`).\n- Validate all required arguments up front.\n- Keep prompts deterministic and stateless (using the same input should produce the same output).\n- Embed resources directly, if needed, for model context.\n- Provide concise descriptions to improve UI discoverability.\n\nWhen implementing MCP prompts, avoid the following common mistakes:\n\n- Allowing missing or malformed arguments\n- Using vague or overly long prompt names\n- Passing oversized inputs (such as full files or large diffs)\n- Failing to sanitize non-UTF-8 or injection-prone strings\n\n### Prompts vs tools vs resources\n\nThe table below compares the three core primitives in MCP:\n\n| Feature | Prompts | Tools | Resources |\n| --- | --- | --- | --- |\n| **Purpose** | Guide model interaction | Execute logic with side effects | Provide structured read-only data |\n| **Triggered by** | User or UI | Agent or client (`tools/call`) | Agent or client (`resources/read`) |\n| **Behavior** | Returns `messages[]` | Runs a function; returns a result | Returns static or dynamic content |\n| **Side effects** | None | Yes (I/O, API calls, mutations) | None |\n| **Composition** | Can embed arguments and resources | Accepts structured input | URI-scoped, optionally templated |\n| **Use cases** | Summarization, Q&A, message templates | File writing, API calls, workflows | Logs, config files, external data |\n\n## Practical implementation example\n\nMCP prompts are a powerful way to define reusable templates that combine context from your application with instructions for the LLM. Here’s how to implement a prompt using the TypeScript SDK.\n\nThis example creates a WhatsApp chat summarization prompt that retrieves chat data and formats it for the LLM:\n\n```typescript\nmcpServer.prompt(\n  "whatsapp_chat_summarizer",\n  "Summarize WhatsApp chat and provide insights",\n  {\n    chatName: z.string().describe("Name of the WhatsApp chat to summarize"),\n  },\n  async (args) => {\n    const { chatName = "" } = args;\n\n    // Find the chat by name\n    // A real implementation would be more robust\n    const targetChat = await chatService.findChatByName(chatName);\n\n    // Get recent messages for analysis\n    const messages = await messageService.getMessages(targetChat.id);\n\n    const promptText = `Analyze this WhatsApp chat data for insights:\n\nChat Information:\n- Chat Name: ${targetChat.name}\n- Chat Type: ${targetChat.isGroup ? "Group Chat" : "Individual Chat"}\n- Analysis Type: summary\n\nAnalysis Focus:\nProvide a comprehensive overview including key topics, sentiment, and notable patterns.\n\nRecent Messages (${messages.length} messages):\n${messages.map((msg) => msg._serializedContent).join("\\n")}\n\nPlease provide a detailed summary.`;\n\n    return {\n      description: `Summary of WhatsApp chat: ${targetChat.name}`,\n      messages: [\n        {\n          role: "user",\n          content: {\n            type: "text",\n            text: promptText,\n          },\n        },\n      ],\n    };\n  },\n);\n```\n\nThis defines a prompt called `whatsapp_chat_summarizer` that takes a `chatName` argument and generates a formatted prompt with the chat data.\n\n### How prompts work in practice\n\nThe LLM client presents a list of available prompts to the user, who can then select one to use. When the user selects a prompt with arguments, the client should display a modal or form allowing the user to fill in the required arguments.\n\nOnce the user submits the form, the MCP client sends a `prompts/get` request to the MCP server with the selected prompt and its arguments. The MCP server adds the relevant context to the prompt (in this case, the WhatsApp chat data) and returns the formatted messages to the MCP client. The client can then send these messages to the LLM for processing.\n\nThis is especially useful for repetitive tasks where a user needs to combine tool call results with a complex prompt. If you can anticipate the user’s needs, you can define a prompt that combines the necessary context and tool calls into a single reusable template.', 'extras': {'signature': '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'}}

</details>


## Code Sources

<details>
<summary>Repository analysis for https://github.com/jlowin/fastmcp</summary>

# Repository analysis for https://github.com/jlowin/fastmcp

## Summary
Repository: jlowin/fastmcp
Commit: 3b8538e2422a1c43fdb69661c610de7985b785f2
Files analyzed: 1151

Estimated tokens: 2.3M

## File tree
```Directory structure:
└── jlowin-fastmcp/
    ├── README.md
    ├── CLAUDE.md
    ├── CODE_OF_CONDUCT.md
    ├── CONTRIBUTING.md
    ├── justfile
    ├── LICENSE
    ├── logo.py
    ├── loq.toml
    ├── pyproject.toml
    ├── SECURITY.md
    ├── .ccignore
    ├── .coderabbit.yaml
    ├── .pre-commit-config.yaml
    ├── AGENTS.md -> CLAUDE.md
    ├── docs/
    │   ├── changelog.mdx
    │   ├── docs.json
    │   ├── fastmcp-analytics.js
    │   ├── prefab-demo-payloads.js
    │   ├── python-sdk-pages.json
    │   ├── unify-intent.js
    │   ├── updates.mdx
    │   ├── v2-banner.js
    │   ├── v2-navigation.json
    │   ├── .ccignore
    │   ├── apps/
    │   │   ├── architecture.mdx
    │   │   ├── development.mdx
    │   │   ├── examples.mdx
    │   │   ├── fastmcp-app.mdx
    │   │   ├── generative.mdx
    │   │   ├── low-level.mdx
    │   │   ├── overview.mdx
    │   │   ├── prefab.mdx
    │   │   ├── quickstart.mdx
    │   │   ├── demos/
    │   │   │   ├── bar-chart.py
    │   │   │   ├── contacts.py
    │   │   │   ├── dashboard.py
    │   │   │   ├── data-table.py
    │   │   │   ├── hitchhikers.py
    │   │   │   ├── pie-chart.py
    │   │   │   ├── reactive.py
    │   │   │   ├── team-directory-reactive.py
    │   │   │   └── team-directory.py
    │   │   └── providers/
    │   │       ├── approval.mdx
    │   │       ├── choice.mdx
    │   │       ├── file-upload.mdx
    │   │       └── form.mdx
    │   ├── assets/
    │   │   └── schemas/
    │   │       └── mcp_server_config/
    │   │           ├── latest.json
    │   │           └── v1.json
    │   ├── cli/
    │   │   ├── auth.mdx
    │   │   ├── client.mdx
    │   │   ├── generate-cli.mdx
    │   │   ├── inspecting.mdx
    │   │   ├── install-mcp.mdx
    │   │   ├── overview.mdx
    │   │   └── running.mdx
    │   ├── clients/
    │   │   ├── cli.mdx
    │   │   ├── client-only-package.mdx
    │   │   ├── client.mdx
    │   │   ├── elicitation.mdx
    │   │   ├── fastmcp-remote.mdx
    │   │   ├── generate-cli.mdx
    │   │   ├── logging.mdx
    │   │   ├── notifications.mdx
    │   │   ├── progress.mdx
    │   │   ├── prompts.mdx
    │   │   ├── resources.mdx
    │   │   ├── roots.mdx
    │   │   ├── sampling.mdx
    │   │   ├── tasks.mdx
    │   │   ├── tools.mdx
    │   │   ├── transports.mdx
    │   │   └── auth/
    │   │       ├── bearer.mdx
    │   │       ├── cimd.mdx
    │   │       └── oauth.mdx
    │   ├── community/
    │   │   ├── README.md
    │   │   └── showcase.mdx
    │   ├── css/
    │   │   ├── banner.css
    │   │   ├── python-sdk.css
    │   │   ├── style.css
    │   │   └── version-badge.css
    │   ├── deployment/
    │   │   ├── http.mdx
    │   │   ├── prefect-horizon.mdx
    │   │   ├── running-server.mdx
    │   │   ├── sandboxed-agents.mdx
    │   │   └── server-configuration.mdx
    │   ├── development/
    │   │   ├── contributing.mdx
    │   │   ├── releases.mdx
    │   │   ├── tests.mdx
    │   │   └── v3-notes/
    │   │       ├── auth-provider-env-vars.mdx
    │   │       └── v3-features.mdx
    │   ├── getting-started/
    │   │   ├── installation.mdx
    │   │   ├── quickstart.mdx
    │   │   ├── welcome.mdx
    │   │   └── upgrading/
    │   │       ├── from-fastmcp-2.mdx
    │   │       ├── from-low-level-sdk.mdx
    │   │       └── from-mcp-sdk.mdx
    │   ├── integrations/
    │   │   ├── anthropic.mdx
    │   │   ├── auth0.mdx
    │   │   ├── authkit.mdx
    │   │   ├── aws-cognito.mdx
    │   │   ├── azure.mdx
    │   │   ├── chatgpt.mdx
    │   │   ├── claude-code.mdx
    │   │   ├── claude-desktop.mdx
    │   │   ├── cursor.mdx
    │   │   ├── descope.mdx
    │   │   ├── discord.mdx
    │   │   ├── eunomia-authorization.mdx
    │   │   ├── fastapi.mdx
    │   │   ├── gemini-cli.mdx
    │   │   ├── gemini.mdx
    │   │   ├── github.mdx
    │   │   ├── google.mdx
    │   │   ├── goose.mdx
    │   │   ├── keycloak.mdx
    │   │   ├── mcp-json-configuration.mdx
    │   │   ├── oci.mdx
    │   │   ├── openai.mdx
    │   │   ├── openapi.mdx
    │   │   ├── permit.mdx
    │   │   ├── propelauth.mdx
    │   │   ├── pydantic-ai.mdx
    │   │   ├── scalekit.mdx
    │   │   ├── supabase.mdx
    │   │   └── workos.mdx
    │   ├── more/
    │   │   ├── faq.mdx
    │   │   └── settings.mdx
    │   ├── patterns/
    │   │   ├── cli.mdx
    │   │   ├── contrib.mdx
    │   │   └── testing.mdx
    │   ├── public/
    │   │   └── schemas/
    │   │       └── fastmcp.json/
    │   │           ├── latest.json
    │   │           └── v1.json
    │   ├── python-sdk/
    │   │   ├── fastmcp-apps-__init__.mdx
    │   │   ├── fastmcp-apps-app.mdx
    │   │   ├── fastmcp-apps-approval.mdx
    │   │   ├── fastmcp-apps-choice.mdx
    │   │   ├── fastmcp-apps-config.mdx
    │   │   ├── fastmcp-apps-file_upload.mdx
    │   │   ├── fastmcp-apps-form.mdx
    │   │   ├── fastmcp-apps-generative.mdx
    │   │   ├── fastmcp-cli.mdx
    │   │   ├── fastmcp-client.mdx
    │   │   ├── fastmcp-decorators.mdx
    │   │   ├── fastmcp-dependencies.mdx
    │   │   ├── fastmcp-exceptions.mdx
    │   │   ├── fastmcp-experimental-__init__.mdx
    │   │   ├── fastmcp-experimental-sampling-__init__.mdx
    │   │   ├── fastmcp-experimental-sampling-handlers.mdx
    │   │   ├── fastmcp-experimental-transforms-__init__.mdx
    │   │   ├── fastmcp-experimental-transforms-code_mode.mdx
    │   │   ├── fastmcp-mcp_config.mdx
    │   │   ├── fastmcp-prompts.mdx
    │   │   ├── fastmcp-resources.mdx
    │   │   ├── fastmcp-server.mdx
    │   │   ├── fastmcp-settings.mdx
    │   │   ├── fastmcp-telemetry.mdx
    │   │   ├── fastmcp-tools.mdx
    │   │   ├── fastmcp-types.mdx
    │   │   ├── fastmcp-utilities-__init__.mdx
    │   │   ├── fastmcp-utilities-async_utils.mdx
    │   │   ├── fastmcp-utilities-auth.mdx
    │   │   ├── fastmcp-utilities-authorization.mdx
    │   │   ├── fastmcp-utilities-cli.mdx
    │   │   ├── fastmcp-utilities-components.mdx
    │   │   ├── fastmcp-utilities-docstring_parsing.mdx
    │   │   ├── fastmcp-utilities-exceptions.mdx
    │   │   ├── fastmcp-utilities-http.mdx
    │   │   ├── fastmcp-utilities-inspect.mdx
    │   │   ├── fastmcp-utilities-json_schema.mdx
    │   │   ├── fastmcp-utilities-json_schema_type.mdx
    │   │   ├── fastmcp-utilities-lifespan.mdx
    │   │   ├── fastmcp-utilities-logging.mdx
    │   │   ├── fastmcp-utilities-mcp_server_config-__init__.mdx
    │   │   ├── fastmcp-utilities-mcp_server_config-v1-__init__.mdx
    │   │   ├── fastmcp-utilities-mcp_server_config-v1-environments-__init__.mdx
    │   │   ├── fastmcp-utilities-mcp_server_config-v1-environments-base.mdx
    │   │   ├── fastmcp-utilities-mcp_server_config-v1-environments-uv.mdx
    │   │   ├── fastmcp-utilities-mcp_server_config-v1-mcp_server_config.mdx
    │   │   ├── fastmcp-utilities-mcp_server_config-v1-sources-__init__.mdx
    │   │   ├── fastmcp-utilities-mcp_server_config-v1-sources-base.mdx
    │   │   ├── fastmcp-utilities-mcp_server_config-v1-sources-filesystem.mdx
    │   │   ├── fastmcp-utilities-mime.mdx
    │   │   ├── fastmcp-utilities-openapi.mdx
    │   │   ├── fastmcp-utilities-pagination.mdx
    │   │   ├── fastmcp-utilities-skills.mdx
    │   │   ├── fastmcp-utilities-tasks.mdx
    │   │   ├── fastmcp-utilities-tests.mdx
    │   │   ├── fastmcp-utilities-timeout.mdx
    │   │   ├── fastmcp-utilities-token_cache.mdx
    │   │   ├── fastmcp-utilities-types.mdx
    │   │   ├── fastmcp-utilities-ui.mdx
    │   │   ├── fastmcp-utilities-version_check.mdx
    │   │   └── fastmcp-utilities-versions.mdx
    │   ├── servers/
    │   │   ├── authorization.mdx
    │   │   ├── composition.mdx
    │   │   ├── context.mdx
    │   │   ├── dependency-injection.mdx
    │   │   ├── elicitation.mdx
    │   │   ├── icons.mdx
    │   │   ├── lifespan.mdx
    │   │   ├── logging.mdx
    │   │   ├── middleware.mdx
    │   │   ├── pagination.mdx
    │   │   ├── progress.mdx
    │   │   ├── prompts.mdx
    │   │   ├── resources.mdx
    │   │   ├── sampling.mdx
    │   │   ├── server.mdx
    │   │   ├── storage-backends.mdx
    │   │   ├── tasks.mdx
    │   │   ├── telemetry.mdx
    │   │   ├── testing.mdx
    │   │   ├── tool-fingerprinting.mdx
    │   │   ├── tools.mdx
    │   │   ├── versioning.mdx
    │   │   ├── visibility.mdx
    │   │   ├── auth/
    │   │   │   ├── authentication.mdx
    │   │   │   ├── full-oauth-server.mdx
    │   │   │   ├── multi-auth.mdx
    │   │   │   ├── oauth-proxy.mdx
    │   │   │   ├── oidc-proxy.mdx
    │   │   │   ├── remote-oauth.mdx
    │   │   │   └── token-verification.mdx
    │   │   ├── providers/
    │   │   │   ├── custom.mdx
    │   │   │   ├── filesystem.mdx
    │   │   │   ├── local.mdx
    │   │   │   ├── overview.mdx
    │   │   │   ├── proxy.mdx
    │   │   │   └── skills.mdx
    │   │   └── transforms/
    │   │       ├── code-mode.mdx
    │   │       ├── namespace.mdx
    │   │       ├── namespacing.mdx
    │   │       ├── prompts-as-tools.mdx
    │   │       ├── resources-as-tools.mdx
    │   │       ├── tool-search.mdx
    │   │       ├── tool-transformation.mdx
    │   │       └── transforms.mdx
    │   ├── snippets/
    │   │   ├── local-focus.mdx
    │   │   ├── prefab-demo-frame.mdx
    │   │   ├── prefab-pin-warning.mdx
    │   │   ├── version-badge.mdx
    │   │   └── youtube-embed.mdx
    │   ├── tutorials/
    │   │   ├── create-mcp-server.mdx
    │   │   ├── mcp.mdx
    │   │   └── rest-api.mdx
    │   ├── v2/
    │   │   ├── changelog.mdx
    │   │   ├── updates.mdx
    │   │   ├── clients/
    │   │   │   ├── client.mdx
    │   │   │   ├── elicitation.mdx
    │   │   │   ├── logging.mdx
    │   │   │   ├── messages.mdx
    │   │   │   ├── progress.mdx
    │   │   │   ├── prompts.mdx
    │   │   │   ├── resources.mdx
    │   │   │   ├── roots.mdx
    │   │   │   ├── sampling.mdx
    │   │   │   ├── tasks.mdx
    │   │   │   ├── tools.mdx
    │   │   │   ├── transports.mdx
    │   │   │   └── auth/
    │   │   │       ├── bearer.mdx
    │   │   │       └── oauth.mdx
    │   │   ├── community/
    │   │   │   └── showcase.mdx
    │   │   ├── deployment/
    │   │   │   ├── http.mdx
    │   │   │   ├── running-server.mdx
    │   │   │   └── server-configuration.mdx
    │   │   ├── development/
    │   │   │   ├── contributing.mdx
    │   │   │   ├── releases.mdx
    │   │   │   ├── tests.mdx
    │   │   │   └── upgrade-guide.mdx
    │   │   ├── getting-started/
    │   │   │   ├── installation.mdx
    │   │   │   ├── quickstart.mdx
    │   │   │   └── welcome.mdx
    │   │   ├── integrations/
    │   │   │   ├── anthropic.mdx
    │   │   │   ├── auth0.mdx
    │   │   │   ├── authkit.mdx
    │   │   │   ├── aws-cognito.mdx
    │   │   │   ├── azure.mdx
    │   │   │   ├── chatgpt.mdx
    │   │   │   ├── claude-code.mdx
    │   │   │   ├── claude-desktop.mdx
    │   │   │   ├── cursor.mdx
    │   │   │   ├── descope.mdx
    │   │   │   ├── discord.mdx
    │   │   │   ├── eunomia-authorization.mdx
    │   │   │   ├── fastapi.mdx
    │   │   │   ├── gemini-cli.mdx
    │   │   │   ├── gemini.mdx
    │   │   │   ├── github.mdx
    │   │   │   ├── google.mdx
    │   │   │   ├── mcp-json-configuration.mdx
    │   │   │   ├── oci.mdx
    │   │   │   ├── openai.mdx
    │   │   │   ├── openapi.mdx
    │   │   │   ├── permit.mdx
    │   │   │   ├── scalekit.mdx
    │   │   │   ├── supabase.mdx
    │   │   │   └── workos.mdx
    │   │   ├── patterns/
    │   │   │   ├── cli.mdx
    │   │   │   ├── contrib.mdx
    │   │   │   ├── decorating-methods.mdx
    │   │   │   ├── testing.mdx
    │   │   │   └── tool-transformation.mdx
    │   │   ├── servers/
    │   │   │   ├── composition.mdx
    │   │   │   ├── context.mdx
    │   │   │   ├── elicitation.mdx
    │   │   │   ├── icons.mdx
    │   │   │   ├── logging.mdx
    │   │   │   ├── middleware.mdx
    │   │   │   ├── progress.mdx
    │   │   │   ├── prompts.mdx
    │   │   │   ├── proxy.mdx
    │   │   │   ├── resources.mdx
    │   │   │   ├── sampling.mdx
    │   │   │   ├── server.mdx
    │   │   │   ├── storage-backends.mdx
    │   │   │   ├── tasks.mdx
    │   │   │   ├── tools.mdx
    │   │   │   └── auth/
    │   │   │       ├── authentication.mdx
    │   │   │       ├── full-oauth-server.mdx
    │   │   │       ├── oauth-proxy.mdx
    │   │   │       ├── oidc-proxy.mdx
    │   │   │       ├── remote-oauth.mdx
    │   │   │       └── token-verification.mdx
    │   │   └── tutorials/
    │   │       ├── create-mcp-server.mdx
    │   │       ├── mcp.mdx
    │   │       └── rest-api.mdx
    │   └── .cursor/
    │       └── rules/
    │           └── mintlify.mdc
    ├── examples/
    │   ├── complex_inputs.py
    │   ├── config_server.py
    │   ├── custom_tool_serializer_decorator.py
    │   ├── desktop.py
    │   ├── echo.py
    │   ├── elicitation.py
    │   ├── get_file.py
    │   ├── in_memory_proxy_example.py
    │   ├── memory.fastmcp.json
    │   ├── memory.py
    │   ├── mount_example.fastmcp.json
    │   ├── mount_example.py
    │   ├── run_with_tracing.py
    │   ├── screenshot.fastmcp.json
    │   ├── screenshot.py
    │   ├── simple_echo.py
    │   ├── tags_example.py
    │   ├── task_elicitation.py
    │   ├── text_me.py
    │   ├── tool_result_echo.py
    │   ├── apps/
    │   │   ├── chart_server.py
    │   │   ├── datatable_server.py
    │   │   ├── generative_ui.py
    │   │   ├── greet_server.py
    │   │   ├── inspector_demo.py
    │   │   ├── patterns_server.py
    │   │   ├── showcase_server.py
    │   │   ├── approval/
    │   │   │   └── approval_server.py
    │   │   ├── approvals/
    │   │   │   └── approvals_server.py
    │   │   ├── choice/
    │   │   │   └── choice_server.py
    │   │   ├── contacts/
    │   │   │   └── contacts_server.py
    │   │   ├── explorer/
    │   │   │   └── explorer_server.py
    │   │   ├── file_upload/
    │   │   │   └── file_upload_server.py
    │   │   ├── form/
    │   │   │   └── form_server.py
    │   │   ├── inventory/
    │   │   │   └── inventory_server.py
    │   │   ├── map/
    │   │   │   └── map_server.py
    │   │   ├── qr_server/
    │   │   │   ├── README.md
    │   │   │   ├── fastmcp.json
    │   │   │   ├── pyproject.toml
    │   │   │   └── qr_server.py
    │   │   ├── quiz/
    │   │   │   └── quiz_server.py
    │   │   ├── sales_dashboard/
    │   │   │   └── sales_dashboard_server.py
    │   │   └── system_monitor/
    │   │       └── system_monitor_server.py
    │   ├── atproto_mcp/
    │   │   ├── README.md
    │   │   ├── demo.py
    │   │   ├── fastmcp.json
    │   │   ├── pyproject.toml
    │   │   └── src/
    │   │       └── atproto_mcp/
    │   │           ├── __init__.py
    │   │           ├── __main__.py
    │   │           ├── py.typed
    │   │           ├── settings.py
    │   │           ├── types.py
    │   │           └── _atproto/
    │   │               ├── __init__.py
    │   │               ├── _client.py
    │   │               ├── _posts.py
    │   │               ├── _profile.py
    │   │               ├── _read.py
    │   │               └── _social.py
    │   ├── auth/
    │   │   ├── authkit/
    │   │   │   └── README.md
    │   │   ├── aws_oauth/
    │   │   │   ├── README.md
    │   │   │   └── requirements.txt
    │   │   ├── azure_oauth/
    │   │   │   └── README.md
    │   │   ├── clerk_oauth/
    │   │   │   └── README.md
    │   │   ├── discord_oauth/
    │   │   │   └── README.md
    │   │   ├── github_oauth/
    │   │   │   └── README.md
    │   │   ├── google_oauth/
    │   │   │   └── README.md
    │   │   ├── keycloak_oauth/
    │   │   │   └── README.md
    │   │   ├── mounted/
    │   │   │   └── README.md
    │   │   ├── oci_oauth/
    │   │   │   └── README.md
    │   │   ├── propelauth_oauth/
    │   │   │   └── README.md
    │   │   ├── scalekit_oauth/
    │   │   │   └── README.md
    │   │   └── workos_oauth/
    │   │       └── README.md
    │   ├── code_mode/
    │   │   └── README.md
    │   ├── diagnostics/
    │   │   └── client_with_tracing.py
    │   ├── fastmcp_config/
    │   │   ├── env_interpolation_example.json
    │   │   ├── fastmcp.json
    │   │   ├── full_example.fastmcp.json
    │   │   └── simple.fastmcp.json
    │   ├── fastmcp_config_demo/
    │   │   ├── README.md
    │   │   └── fastmcp.json
    │   ├── filesystem-provider/
    │   │   └── components/
    │   │       ├── prompts/
    │   │       │   └── assistant.py
    │   │       ├── resources/
    │   │       │   └── config.py
    │   │       └── tools/
    │   │           ├── calculator.py
    │   │           └── greeting.py
    │   ├── namespace_activation/
    │   │   └── README.md
    │   ├── persistent_state/
    │   │   ├── README.md
    │   │   └── client_stdio.py
    │   ├── providers/
    │   │   └── sqlite/
    │   │       ├── README.md
    │   │       └── setup_db.py
    │   ├── sampling/
    │   │   ├── README.md
    │   │   ├── server_fallback.py
    │   │   ├── structured_output.py
    │   │   ├── text.py
    │   │   └── tool_use.py
    │   ├── search/
    │   │   ├── README.md
    │   │   ├── client_bm25.py
    │   │   ├── client_regex.py
    │   │   ├── server_bm25.py
    │   │   └── server_regex.py
    │   ├── skills/
    │   │   ├── README.md
    │   │   ├── download_skills.py
    │   │   └── sample_skills/
    │   │       ├── code-review/
    │   │       │   └── SKILL.md
    │   │       └── pdf-processing/
    │   │           ├── reference.md
    │   │           └── SKILL.md
    │   ├── smart_home/
    │   │   ├── README.md
    │   │   ├── hub.fastmcp.json
    │   │   ├── lights.fastmcp.json
    │   │   ├── pyproject.toml
    │   │   └── src/
    │   │       └── smart_home/
    │   │           ├── __init__.py
    │   │           ├── __main__.py
    │   │           ├── hub.py
    │   │           ├── py.typed
    │   │           ├── settings.py
    │   │           └── lights/
    │   │               ├── __init__.py
    │   │               └── hue_utils.py
    │   ├── tasks/
    │   │   ├── README.md
    │   │   └── docker-compose.yml
    │   ├── testing_demo/
    │   │   ├── README.md
    │   │   ├── pyproject.toml
    │   │   └── tests/
    │   │       └── test_server.py
    │   └── versioning/
    │       ├── client_version_selection.py
    │       ├── version_filters.py
    │       └── versioned_components.py
    ├── fastmcp_remote/
    │   ├── README.md
    │   ├── pyproject.toml
    │   └── fastmcp_remote/
    │       ├── __init__.py
    │       ├── cli.py
    │       └── py.typed
    ├── fastmcp_slim/
    │   ├── README.md
    │   ├── pyproject.toml
    │   └── fastmcp/
    │       ├── __init__.py
    │       ├── _install_hints.py
    │       ├── decorators.py
    │       ├── dependencies.py
    │       ├── exceptions.py
    │       ├── mcp_config.py
    │       ├── py.typed
    │       ├── settings.py
    │       ├── telemetry.py
    │       ├── types.py
    │       ├── apps/
    │       │   ├── __init__.py
    │       │   ├── app.py
    │       │   ├── approval.py
    │       │   ├── choice.py
    │       │   ├── config.py
    │       │   ├── file_upload.py
    │       │   ├── form.py
    │       │   └── generative.py
    │       ├── cli/
    │       │   ├── __init__.py
    │       │   ├── __main__.py
    │       │   ├── apps_dev.py
    │       │   ├── auth.py
    │       │   ├── cimd.py
    │       │   ├── cli.py
    │       │   ├── discovery.py
    │       │   ├── generate.py
    │       │   ├── run.py
    │       │   ├── tasks.py
    │       │   └── install/
    │       │       ├── __init__.py
    │       │       ├── claude_code.py
    │       │       ├── claude_desktop.py
    │       │       ├── cursor.py
    │       │       ├── gemini_cli.py
    │       │       ├── goose.py
    │       │       ├── mcp_json.py
    │       │       ├── shared.py
    │       │       └── stdio.py
    │       ├── client/
    │       │   ├── __init__.py
    │       │   ├── dependencies.py
    │       │   ├── elicitation.py
    │       │   ├── logging.py
    │       │   ├── messages.py
    │       │   ├── oauth_callback.py
    │       │   ├── progress.py
    │       │   ├── roots.py
    │       │   ├── tasks.py
    │       │   ├── telemetry.py
    │       │   ├── auth/
    │       │   │   ├── __init__.py
    │       │   │   ├── bearer.py
    │       │   │   └── oauth.py
    │       │   ├── mixins/
    │       │   │   ├── __init__.py
    │       │   │   ├── prompts.py
    │       │   │   ├── resources.py
    │       │   │   ├── task_management.py
    │       │   │   └── tools.py
    │       │   ├── sampling/
    │       │   │   ├── __init__.py
    │       │   │   └── handlers/
    │       │   │       ├── __init__.py
    │       │   │       ├── anthropic.py
    │       │   │       ├── google_genai.py
    │       │   │       └── openai.py
    │       │   └── transports/
    │       │       ├── __init__.py
    │       │       ├── base.py
    │       │       ├── config.py
    │       │       ├── http.py
    │       │       ├── inference.py
    │       │       ├── memory.py
    │       │       ├── sse.py
    │       │       └── stdio.py
    │       ├── contrib/
    │       │   ├── README.md
    │       │   ├── bulk_tool_caller/
    │       │   │   ├── README.md
    │       │   │   ├── __init__.py
    │       │   │   ├── bulk_tool_caller.py
    │       │   │   └── example.py
    │       │   ├── component_manager/
    │       │   │   ├── README.md
    │       │   │   ├── __init__.py
    │       │   │   ├── component_manager.py
    │       │   │   └── example.py
    │       │   └── mcp_mixin/
    │       │       ├── README.md
    │       │       ├── __init__.py
    │       │       ├── example.py
    │       │       └── mcp_mixin.py
    │       ├── experimental/
    │       │   ├── __init__.py
    │       │   ├── sampling/
    │       │   │   ├── __init__.py
    │       │   │   └── handlers/
    │       │   │       ├── __init__.py
    │       │   │       └── openai.py
    │       │   ├── server/
    │       │   │   └── openapi/
    │       │   │       └── __init__.py
    │       │   ├── transforms/
    │       │   │   ├── __init__.py
    │       │   │   └── code_mode.py
    │       │   └── utilities/
    │       │       └── openapi/
    │       │           └── __init__.py
    │       ├── prompts/
    │       │   ├── __init__.py
    │       │   ├── base.py
    │       │   └── function_prompt.py
    │       ├── resources/
    │       │   ├── __init__.py
    │       │   ├── base.py
    │       │   ├── function_resource.py
    │       │   ├── template.py
    │       │   └── types.py
    │       ├── server/
    │       │   ├── __init__.py
    │       │   ├── app.py
    │       │   ├── apps.py
    │       │   ├── context.py
    │       │   ├── dependencies.py
    │       │   ├── elicitation.py
    │       │   ├── event_store.py
    │       │   ├── http.py
    │       │   ├── lifespan.py
    │       │   ├── low_level.py
    │       │   ├── proxy.py
    │       │   ├── telemetry.py
    │       │   ├── auth/
    │       │   │   ├── __init__.py
    │       │   │   ├── auth.py
    │       │   │   ├── authorization.py
    │       │   │   ├── cimd.py
    │       │   │   ├── jwt_issuer.py
    │       │   │   ├── middleware.py
    │       │   │   ├── oidc_proxy.py
    │       │   │   ├── redirect_validation.py
    │       │   │   ├── ssrf.py
    │       │   │   ├── handlers/
    │       │   │   │   ├── __init__.py
    │       │   │   │   └── authorize.py
    │       │   │   ├── oauth_proxy/
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── consent.py
    │       │   │   │   ├── models.py
    │       │   │   │   ├── proxy.py
    │       │   │   │   └── ui.py
    │       │   │   └── providers/
    │       │   │       ├── __init__.py
    │       │   │       ├── auth0.py
    │       │   │       ├── aws.py
    │       │   │       ├── azure.py
    │       │   │       ├── clerk.py
    │       │   │       ├── debug.py
    │       │   │       ├── descope.py
    │       │   │       ├── discord.py
    │       │   │       ├── github.py
    │       │   │       ├── google.py
    │       │   │       ├── in_memory.py
    │       │   │       ├── introspection.py
    │       │   │       ├── jwt.py
    │       │   │       ├── keycloak.py
    │       │   │       ├── oci.py
    │       │   │       ├── propelauth.py
    │       │   │       ├── scalekit.py
    │       │   │       ├── supabase.py
    │       │   │       └── workos.py
    │       │   ├── middleware/
    │       │   │   ├── __init__.py
    │       │   │   ├── authorization.py
    │       │   │   ├── caching.py
    │       │   │   ├── dereference.py
    │       │   │   ├── error_handling.py
    │       │   │   ├── logging.py
    │       │   │   ├── middleware.py
    │       │   │   ├── ping.py
    │       │   │   ├── rate_limiting.py
    │       │   │   ├── response_limiting.py
    │       │   │   ├── timing.py
    │       │   │   └── tool_injection.py
    │       │   ├── mixins/
    │       │   │   ├── __init__.py
    │       │   │   ├── lifespan.py
    │       │   │   ├── mcp_operations.py
    │       │   │   └── transport.py
    │       │   ├── openapi/
    │       │   │   ├── __init__.py
    │       │   │   ├── components.py
    │       │   │   └── routing.py
    │       │   ├── providers/
    │       │   │   ├── __init__.py
    │       │   │   ├── addressing.py
    │       │   │   ├── aggregate.py
    │       │   │   ├── base.py
    │       │   │   ├── fastmcp_provider.py
    │       │   │   ├── filesystem.py
    │       │   │   ├── filesystem_discovery.py
    │       │   │   ├── prefab_synthesis.py
    │       │   │   ├── proxy.py
    │       │   │   ├── wrapped_provider.py
    │       │   │   ├── local_provider/
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── local_provider.py
    │       │   │   │   └── decorators/
    │       │   │   │       ├── __init__.py
    │       │   │   │       ├── prompts.py
    │       │   │   │       ├── resources.py
    │       │   │   │       └── tools.py
    │       │   │   ├── openapi/
    │       │   │   │   ├── README.md
    │       │   │   │   ├── __init__.py
    │       │   │   │   ├── components.py
    │       │   │   │   ├── provider.py
    │       │   │   │   └── routing.py
    │       │   │   └── skills/
    │       │   │       ├── __init__.py
    │       │   │       ├── _common.py
    │       │   │       ├── claude_provider.py
    │       │   │       ├── directory_provider.py
    │       │   │       ├── skill_provider.py
    │       │   │       └── vendor_providers.py
    │       │   ├── sampling/
    │       │   │   ├── __init__.py
    │       │   │   ├── run.py
    │       │   │   └── sampling_tool.py
    │       │   ├── tasks/
    │       │   │   ├── __init__.py
    │       │   │   ├── capabilities.py
    │       │   │   ├── config.py
    │       │   │   ├── context.py
    │       │   │   ├── elicitation.py
    │       │   │   ├── handlers.py
    │       │   │   ├── keys.py
    │       │   │   ├── notifications.py
    │       │   │   ├── requests.py
    │       │   │   ├── routing.py
    │       │   │   └── subscriptions.py
    │       │   └── transforms/
    │       │       ├── __init__.py
    │       │       ├── catalog.py
    │       │       ├── namespace.py
    │       │       ├── prompts_as_tools.py
    │       │       ├── resources_as_tools.py
    │       │       ├── tool_transform.py
    │       │       ├── version_filter.py
    │       │       ├── visibility.py
    │       │       └── search/
    │       │           ├── __init__.py
    │       │           ├── base.py
    │       │           ├── bm25.py
    │       │           └── regex.py
    │       ├── tools/
    │       │   ├── __init__.py
    │       │   ├── base.py
    │       │   ├── function_parsing.py
    │       │   ├── function_tool.py
    │       │   └── tool_transform.py
    │       └── utilities/
    │           ├── __init__.py
    │           ├── async_utils.py
    │           ├── auth.py
    │           ├── authorization.py
    │           ├── cli.py
    │           ├── components.py
    │           ├── docstring_parsing.py
    │           ├── exceptions.py
    │           ├── http.py
    │           ├── inspect.py
    │           ├── json_schema.py
    │           ├── json_schema_type.py
    │           ├── lifespan.py
    │           ├── logging.py
    │           ├── mime.py
    │           ├── pagination.py
    │           ├── skills.py
    │           ├── tasks.py
    │           ├── tests.py
    │           ├── timeout.py
    │           ├── token_cache.py
    │           ├── types.py
    │           ├── ui.py
    │           ├── version_check.py
    │           ├── versions.py
    │           ├── mcp_server_config/
    │           │   ├── __init__.py
    │           │   └── v1/
    │           │       ├── __init__.py
    │           │       ├── mcp_server_config.py
    │           │       ├── schema.json
    │           │       ├── environments/
    │           │       │   ├── __init__.py
    │           │       │   ├── base.py
    │           │       │   └── uv.py
    │           │       └── sources/
    │           │           ├── __init__.py
    │           │           ├── base.py
    │           │           └── filesystem.py
    │           └── openapi/
    │               ├── README.md
    │               ├── __init__.py
    │               ├── director.py
    │               ├── formatters.py
    │               ├── json_schema_converter.py
    │               ├── models.py
    │               ├── parser.py
    │               └── schemas.py
    ├── scripts/
    │   ├── auto_close_duplicates.py
    │   ├── auto_close_needs_mre.py
    │   └── benchmark_imports.py
    ├── skills/
    │   └── fastmcp-client-cli/
    │       └── SKILL.md
    ├── tests/
    │   ├── __init__.py
    │   ├── conftest.py
    │   ├── test_apps.py
    │   ├── test_apps_prefab.py
    │   ├── test_fastmcp_app.py
    │   ├── test_json_schema_generation.py
    │   ├── test_mcp_config.py
    │   ├── test_settings.py
    │   ├── apps/
    │   │   ├── __init__.py
    │   │   ├── test_approval.py
    │   │   ├── test_choice.py
    │   │   ├── test_file_upload.py
    │   │   └── test_form.py
    │   ├── cli/
    │   │   ├── __init__.py
    │   │   ├── test_cimd_cli.py
    │   │   ├── test_cli.py
    │   │   ├── test_client_commands.py
    │   │   ├── test_config.py
    │   │   ├── test_cursor.py
    │   │   ├── test_discovery.py
    │   │   ├── test_fastmcp_remote.py
    │   │   ├── test_generate_cli.py
    │   │   ├── test_goose.py
    │   │   ├── test_install.py
    │   │   ├── test_mcp_server_config_integration.py
    │   │   ├── test_mcp_server_config_schema.py
    │   │   ├── test_project_prepare.py
    │   │   ├── test_run.py
    │   │   ├── test_run_config.py
    │   │   ├── test_server_args.py
    │   │   ├── test_shared.py
    │   │   ├── test_tasks.py
    │   │   └── test_with_argv.py
    │   ├── client/
    │   │   ├── __init__.py
    │   │   ├── test_elicitation.py
    │   │   ├── test_elicitation_enums.py
    │   │   ├── test_logs.py
    │   │   ├── test_notifications.py
    │   │   ├── test_oauth_callback_race.py
    │   │   ├── test_oauth_callback_xss.py
    │   │   ├── test_openapi.py
    │   │   ├── test_progress.py
    │   │   ├── test_roots.py
    │   │   ├── test_sampling.py
    │   │   ├── test_sampling_result_types.py
    │   │   ├── test_sampling_tool_loop.py
    │   │   ├── test_slim_package_boundaries.py
    │   │   ├── test_sse.py
    │   │   ├── test_stdio.py
    │   │   ├── test_streamable_http.py
    │   │   ├── auth/
    │   │   │   ├── __init__.py
    │   │   │   ├── test_oauth_cimd.py
    │   │   │   ├── test_oauth_client.py
    │   │   │   └── test_oauth_static_client.py
    │   │   ├── client/
    │   │   │   ├── __init__.py
    │   │   │   ├── test_auth.py
    │   │   │   ├── test_client.py
    │   │   │   ├── test_error_handling.py
    │   │   │   ├── test_initialize.py
    │   │   │   ├── test_session.py
    │   │   │   ├── test_timeout.py
    │   │   │   └── test_transport.py
    │   │   ├── sampling/
    │   │   │   ├── __init__.py
    │   │   │   └── handlers/
    │   │   │       ├── __init__.py
    │   │   │       ├── test_anthropic_handler.py
    │   │   │       ├── test_google_genai_handler.py
    │   │   │       └── test_openai_handler.py
    │   │   ├── tasks/
    │   │   │   ├── conftest.py
    │   │   │   ├── test_client_prompt_tasks.py
    │   │   │   ├── test_client_resource_tasks.py
    │   │   │   ├── test_client_task_notifications.py
    │   │   │   ├── test_client_task_protocol.py
    │   │   │   ├── test_client_tool_tasks.py
    │   │   │   ├── test_task_context_validation.py
    │   │   │   └── test_task_result_caching.py
    │   │   ├── telemetry/
    │   │   │   ├── __init__.py
    │   │   │   ├── test_client_list_tracing.py
    │   │   │   └── test_client_tracing.py
    │   │   └── transports/
    │   │       ├── __init__.py
    │   │       ├── test_memory_transport.py
    │   │       ├── test_no_redirect.py
    │   │       ├── test_transports.py
    │   │       └── test_uv_transport.py
    │   ├── conformance/
    │   │   ├── __init__.py
    │   │   ├── expected-failures.yml
    │   │   └── test_conformance.py
    │   ├── contrib/
    │   │   ├── __init__.py
    │   │   ├── test_bulk_tool_caller.py
    │   │   ├── test_component_manager.py
    │   │   └── test_mcp_mixin.py
    │   ├── deprecated/
    │   │   ├── __init__.py
    │   │   ├── conftest.py
    │   │   ├── test_add_tool_transformation.py
    │   │   ├── test_deprecated.py
    │   │   ├── test_elicitation.py
    │   │   ├── test_exclude_args.py
    │   │   ├── test_function_component_imports.py
    │   │   ├── test_import_server.py
    │   │   ├── test_openapi_deprecations.py
    │   │   ├── test_settings.py
    │   │   ├── test_tool_injection_middleware.py
    │   │   ├── test_tool_serializer.py
    │   │   ├── openapi/
    │   │   │   └── test_openapi.py
    │   │   └── server/
    │   │       ├── __init__.py
    │   │       └── test_include_exclude_tags.py
    │   ├── docs/
    │   │   └── test_doc_examples.py
    │   ├── experimental/
    │   │   ├── README.md
    │   │   ├── __init__.py
    │   │   └── transforms/
    │   │       ├── test_code_mode.py
    │   │       ├── test_code_mode_discovery.py
    │   │       └── test_code_mode_serialization.py
    │   ├── fs/
    │   │   ├── test_discovery.py
    │   │   └── test_provider.py
    │   ├── integration_tests/
    │   │   ├── __init__.py
    │   │   ├── conftest.py
    │   │   ├── test_github_mcp_remote.py
    │   │   ├── test_timeout_fix.py
    │   │   └── auth/
    │   │       ├── __init__.py
    │   │       ├── test_github_provider_integration.py
    │   │       └── test_keycloak_provider_integration.py
    │   ├── prompts/
    │   │   ├── __init__.py
    │   │   ├── test_prompt.py
    │   │   └── test_standalone_decorator.py
    │   ├── resources/
    │   │   ├── __init__.py
    │   │   ├── test_file_resources.py
    │   │   ├── test_function_resources.py
    │   │   ├── test_resource_template.py
    │   │   ├── test_resource_template_meta.py
    │   │   ├── test_resource_template_query_params.py
    │   │   ├── test_resources.py
    │   │   └── test_standalone_decorator.py
    │   ├── scripts/
    │   │   └── test_auto_close_needs_mre.py
    │   ├── server/
    │   │   ├── __init__.py
    │   │   ├── test_app_state.py
    │   │   ├── test_auth_integration.py
    │   │   ├── test_auth_integration_errors.py
    │   │   ├── test_context.py
    │   │   ├── test_dependencies.py
    │   │   ├── test_dependencies_advanced.py
    │   │   ├── test_event_store.py
    │   │   ├── test_fastapi_testclient_compat.py
    │   │   ├── test_file_server.py
    │   │   ├── test_icons.py
    │   │   ├── test_input_validation.py
    │   │   ├── test_log_level.py
    │   │   ├── test_logging.py
    │   │   ├── test_pagination.py
    │   │   ├── test_providers.py
    │   │   ├── test_server.py
    │   │   ├── test_server_docket.py
    │   │   ├── test_server_lifespan.py
    │   │   ├── test_server_safety.py
    │   │   ├── test_session_visibility.py
    │   │   ├── test_streamable_http_no_redirect.py
    │   │   ├── test_tool_annotations.py
    │   │   ├── test_tool_transformation.py
    │   │   ├── auth/
    │   │   │   ├── __init__.py
    │   │   │   ├── test_auth_provider.py
    │   │   │   ├── test_authorization.py
    │   │   │   ├── test_cimd.py
    │   │   │   ├── test_cimd_validators.py
    │   │   │   ├── test_debug_verifier.py
    │   │   │   ├── test_enhanced_error_responses.py
    │   │   │   ├── test_jwt_issuer.py
    │   │   │   ├── test_jwt_provider.py
    │   │   │   ├── test_jwt_provider_bearer.py
    │   │   │   ├── test_multi_auth.py
    │   │   │   ├── test_oauth_consent_flow.py
    │   │   │   ├── test_oauth_consent_page.py
    │   │   │   ├── test_oauth_mounting.py
    │   │   │   ├── test_oauth_proxy_redirect_validation.py
    │   │   │   ├── test_oauth_proxy_storage.py
    │   │   │   ├── test_oidc_proxy.py
    │   │   │   ├── test_oidc_proxy_token.py
    │   │   │   ├── test_redirect_validation.py
    │   │   │   ├── test_remote_auth_provider.py
    │   │   │   ├── test_ssrf_protection.py
    │   │   │   ├── test_static_token_verifier.py
    │   │   │   ├── oauth_proxy/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── conftest.py
    │   │   │   │   ├── test_authorization.py
    │   │   │   │   ├── test_client_registration.py
    │   │   │   │   ├── test_config.py
    │   │   │   │   ├── test_e2e.py
    │   │   │   │   ├── test_oauth_proxy.py
    │   │   │   │   ├── test_tokens.py
    │   │   │   │   └── test_ui.py
    │   │   │   └── providers/
    │   │   │       ├── __init__.py
    │   │   │       ├── test_auth0.py
    │   │   │       ├── test_aws.py
    │   │   │       ├── test_azure.py
    │   │   │       ├── test_azure_scopes.py
    │   │   │       ├── test_clerk.py
    │   │   │       ├── test_debug.py
    │   │   │       ├── test_descope.py
    │   │   │       ├── test_discord.py
    │   │   │       ├── test_github.py
    │   │   │       ├── test_google.py
    │   │   │       ├── test_http_client.py
    │   │   │       ├── test_introspection.py
    │   │   │       ├── test_keycloak.py
    │   │   │       ├── test_oci.py
    │   │   │       ├── test_propelauth.py
    │   │   │       ├── test_scalekit.py
    │   │   │       ├── test_supabase.py
    │   │   │       └── test_workos.py
    │   │   ├── http/
    │   │   │   ├── __init__.py
    │   │   │   ├── test_bearer_auth_backend.py
    │   │   │   ├── test_custom_routes.py
    │   │   │   ├── test_http_auth_middleware.py
    │   │   │   ├── test_http_dependencies.py
    │   │   │   ├── test_http_middleware.py
    │   │   │   ├── test_stale_access_token.py
    │   │   │   └── test_streamable_http_shutdown.py
    │   │   ├── middleware/
    │   │   │   ├── __init__.py
    │   │   │   ├── test_caching.py
    │   │   │   ├── test_dereference.py
    │   │   │   ├── test_error_handling.py
    │   │   │   ├── test_initialization_middleware.py
    │   │   │   ├── test_logging.py
    │   │   │   ├── test_middleware.py
    │   │   │   ├── test_middleware_nested.py
    │   │   │   ├── test_ping.py
    │   │   │   ├── test_rate_limiting.py
    │   │   │   ├── test_response_limiting.py
    │   │   │   ├── test_timing.py
    │   │   │   └── test_tool_injection.py
    │   │   ├── mount/
    │   │   │   ├── __init__.py
    │   │   │   ├── test_advanced.py
    │   │   │   ├── test_filtering.py
    │   │   │   ├── test_mount.py
    │   │   │   ├── test_prompts.py
    │   │   │   ├── test_proxy.py
    │   │   │   └── test_resources.py
    │   │   ├── providers/
    │   │   │   ├── __init__.py
    │   │   │   ├── test_addressing.py
    │   │   │   ├── test_base_provider.py
    │   │   │   ├── test_fastmcp_provider.py
    │   │   │   ├── test_local_provider.py
    │   │   │   ├── test_local_provider_prompts.py
    │   │   │   ├── test_local_provider_resources.py
    │   │   │   ├── test_prefab_roundtrip.py
    │   │   │   ├── test_prefab_synthesis.py
    │   │   │   ├── test_skills_provider.py
    │   │   │   ├── test_skills_vendor_providers.py
    │   │   │   ├── test_transforming_provider.py
    │   │   │   ├── local_provider_tools/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── test_context.py
    │   │   │   │   ├── test_decorator.py
    │   │   │   │   ├── test_enabled.py
    │   │   │   │   ├── test_local_provider_tools.py
    │   │   │   │   ├── test_output_schema.py
    │   │   │   │   ├── test_parameters.py
    │   │   │   │   └── test_tags.py
    │   │   │   ├── openapi/
    │   │   │   │   ├── __init__.py
    │   │   │   │   ├── test_comprehensive.py
    │   │   │   │   ├── test_deepobject_style.py
    │   │   │   │   ├── test_end_to_end_compatibility.py
    │   │   │   │   ├── test_openapi_features.py
    │   │   │   │   ├── test_openapi_performance.py
    │   │   │   │   ├── test_parameter_collisions.py
    │   │   │   │   ├── test_performance_comparison.py
    │   │   │   │   └── test_server.py
    │   │   │   └── proxy/
    │   │   │       ├── __init__.py
    │   │   │       ├── test_proxy_client.py
    │   │   │       ├── test_proxy_server.py
    │   │   │       └── test_stateful_proxy_client.py
    │   │   ├── sampling/
    │   │   │   ├── __init__.py
    │   │   │   ├── test_prepare_tools.py
    │   │   │   └── test_sampling_tool.py
    │   │   ├── tasks/
    │   │   │   ├── __init__.py
    │   │   │   ├── conftest.py
    │   │   │   ├── test_concurrent_dependencies.py
    │   │   │   ├── test_context_background_task.py
    │   │   │   ├── test_custom_subclass_tasks.py
    │   │   │   ├── test_notifications.py
    │   │   │   ├── test_progress_dependency.py
    │   │   │   ├── test_resource_task_meta_parameter.py
    │   │   │   ├── test_server_tasks_parameter.py
    │   │   │   ├── test_snapshot_restore.py
    │   │   │   ├── test_sync_function_task_disabled.py
    │   │   │   ├── test_task_capabilities.py
    │   │   │   ├── test_task_config.py
    │   │   │   ├── test_task_dependencies.py
    │   │   │   ├── test_task_elicitation_relay.py
    │   │   │   ├── test_task_keys.py
    │   │   │   ├── test_task_meta_parameter.py
    │   │   │   ├── test_task_metadata.py
    │   │   │   ├── test_task_methods.py
    │   │   │   ├── test_task_mount.py
    │   │   │   ├── test_task_prompts.py
    │   │   │   ├── test_task_protocol.py
    │   │   │   ├── test_task_proxy.py
    │   │   │   ├── test_task_resources.py
    │   │   │   ├── test_task_return_types.py
    │   │   │   ├── test_task_security.py
    │   │   │   ├── test_task_status_notifications.py
    │   │   │   ├── test_task_tools.py
    │   │   │   └── test_task_ttl.py
    │   │   ├── telemetry/
    │   │   │   ├── __init__.py
    │   │   │   ├── test_delegate_method.py
    │   │   │   ├── test_list_tracing.py
    │   │   │   ├── test_provider_tracing.py
    │   │   │   ├── test_sampling_tracing.py
    │   │   │   └── test_server_tracing.py
    │   │   ├── transforms/
    │   │   │   ├── test_catalog.py
    │   │   │   ├── test_prompts_as_tools.py
    │   │   │   ├── test_resources_as_tools.py
    │   │   │   ├── test_search.py
    │   │   │   └── test_visibility.py
    │   │   └── versioning/
    │   │       ├── __init__.py
    │   │       ├── test_calls.py
    │   │       ├── test_filtering.py
    │   │       ├── test_mounting.py
    │   │       ├── test_versioning.py
    │   │       └── test_visibility_version_fallback.py
    │   ├── telemetry/
    │   │   ├── __init__.py
    │   │   └── test_module.py
    │   ├── tools/
    │   │   ├── __init__.py
    │   │   ├── test_standalone_decorator.py
    │   │   ├── test_tool_future_annotations.py
    │   │   ├── test_tool_run_in_thread.py
    │   │   ├── test_tool_timeout.py
    │   │   ├── tool/
    │   │   │   ├── __init__.py
    │   │   │   ├── test_callable.py
    │   │   │   ├── test_content.py
    │   │   │   ├── test_output_schema.py
    │   │   │   ├── test_results.py
    │   │   │   ├── test_title.py
    │   │   │   └── test_tool.py
    │   │   └── tool_transform/
    │   │       ├── __init__.py
    │   │       ├── test_args.py
    │   │       ├── test_metadata.py
    │   │       ├── test_schemas.py
    │   │       └── test_tool_transform.py
    │   └── utilities/
    │       ├── __init__.py
    │       ├── test_async_utils.py
    │       ├── test_auth.py
    │       ├── test_cli.py
    │       ├── test_components.py
    │       ├── test_docstring_parsing.py
    │       ├── test_http.py
    │       ├── test_inspect.py
    │       ├── test_inspect_icons.py
    │       ├── test_json_schema.py
    │       ├── test_logging.py
    │       ├── test_skills.py
    │       ├── test_tests.py
    │       ├── test_timeout.py
    │       ├── test_token_cache.py
    │       ├── test_typeadapter.py
    │       ├── test_types.py
    │       ├── test_version_check.py
    │       ├── json_schema_type/
    │       │   ├── __init__.py
    │       │   ├── cluster_failures.py
    │       │   ├── conftest.py
    │       │   ├── test_advanced.py
    │       │   ├── test_constraints.py
    │       │   ├── test_containers.py
    │       │   ├── test_formats.py
    │       │   ├── test_json_schema_type.py
    │       │   ├── test_real_world_schemas.py
    │       │   └── test_unions.py
    │       └── openapi/
    │           ├── __init__.py
    │           ├── conftest.py
    │           ├── test_allof_requestbody.py
    │           ├── test_circular_references.py
    │           ├── test_direct_array_schemas.py
    │           ├── test_director.py
    │           ├── test_legacy_compatibility.py
    │           ├── test_models.py
    │           ├── test_nullable_fields.py
    │           ├── test_parser.py
    │           ├── test_propertynames_ref_rewrite.py
    │           ├── test_schemas.py
    │           └── test_transitive_references.py
    ├── v3-notes/
    │   ├── get-methods-consolidation.md
    │   ├── prompt-internal-types.md
    │   ├── provider-architecture.md
    │   ├── provider-test-pattern.md
    │   ├── resource-internal-types.md
    │   ├── task-meta-parameter.md
    │   └── visibility.md
    ├── .claude/
    │   ├── settings.json
    │   ├── hooks/
    │   │   └── session-init.sh
    │   └── skills/
    │       ├── code-review/
    │       │   └── SKILL.md
    │       ├── python-tests/
    │       │   └── SKILL.md
    │       ├── review-issue/
    │       │   └── SKILL.md
    │       └── review-pr/
    │           └── SKILL.md
    ├── .cursor/
    │   └── rules/
    │       └── core-mcp-objects.mdc
    └── .github/
        ├── dependabot.yml
        ├── pull_request_template.md
        ├── release.yml
        ├── actions/
        │   ├── run-claude/
        │   │   └── action.yml
        │   ├── run-pytest/
        │   │   └── action.yml
        │   └── setup-uv/
        │       └── action.yml
        ├── copilot-instructions.md -> AGENTS.md
        ├── ISSUE_TEMPLATE/
        │   ├── bug.yml
        │   ├── config.yml
        │   └── enhancement.yml
        ├── scripts/
        │   ├── mention/
        │   │   ├── gh-get-review-threads.sh
        │   │   └── gh-resolve-review-thread.sh
        │   └── pr-review/
        │       ├── pr-comment.sh
        │       ├── pr-diff.sh
        │       ├── pr-existing-comments.sh
        │       ├── pr-remove-comment.sh
        │       └── pr-review.sh
        └── workflows/
            ├── auto-close-duplicates.yml
            ├── auto-close-needs-mre.yml
            ├── martian-test-failure.yml
            ├── martian-triage-issue.yml
            ├── marvin-comment-on-issue.yml
            ├── marvin-comment-on-pr.yml
            ├── marvin-dedupe-issues.yml
            ├── marvin-label-triage.yml
            ├── minimize-resolved-reviews.yml
            ├── publish-fastmcp-remote.yml
            ├── publish-fastmcp-slim.yml
            ├── publish-fastmcp.yml
            ├── require-issue-link.yml
            ├── run-schema-crash-test.yml
            ├── run-static.yml
            ├── run-tests.yml
            ├── run-upgrade-checks.yml
            ├── update-config-schema.yml
            └── update-sdk-docs.yml

```

## Extracted content
================================================
FILE: README.md
================================================
<div align="center">

<!-- omit in toc -->

<picture>
  <source width="550" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/PrefectHQ/fastmcp/main/docs/assets/brand/f-watercolor-waves-4-dark.png">
  <source width="550" media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/PrefectHQ/fastmcp/main/docs/assets/brand/f-watercolor-waves-4.png">
  <img width="550" alt="FastMCP Logo" src="https://raw.githubusercontent.com/PrefectHQ/fastmcp/main/docs/assets/brand/f-watercolor-waves-2.png">
</picture>

# FastMCP 🚀

<strong>Move fast and make things.</strong>

*Made with 💙 by [Prefect](https://www.prefect.io/)*

[![Docs](https://img.shields.io/badge/docs-gofastmcp.com-blue)](https://gofastmcp.com)
[![Discord](https://img.shields.io/badge/community-discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/uu8dJCgttd)
[![PyPI - Version](https://img.shields.io/pypi/v/fastmcp.svg)](https://pypi.org/project/fastmcp)
[![Tests](https://github.com/PrefectHQ/fastmcp/actions/workflows/run-tests.yml/badge.svg)](https://github.com/PrefectHQ/fastmcp/actions/workflows/run-tests.yml)
[![License](https://img.shields.io/github/license/PrefectHQ/fastmcp.svg)](https://github.com/PrefectHQ/fastmcp/blob/main/LICENSE)

<a href="https://trendshift.io/repositories/21461" target="_blank"><img src="https://trendshift.io/api/badge/repositories/21461" alt="prefecthq%2Ffastmcp | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

---

The [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) connects LLMs to tools and data. FastMCP gives you everything you need to go from prototype to production:

```python
from fastmcp import FastMCP

mcp = FastMCP("Demo 🚀")

@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

if __name__ == "__main__":
    mcp.run()
```

## Why FastMCP

Building an effective MCP application is harder than it looks. FastMCP handles all of it. Declare a tool with a Python function, and the schema, validation, and documentation are generated automatically. Connect to a server with a URL, and transport negotiation, authentication, and protocol lifecycle are managed for you. You focus on your logic, and the MCP part just works: **with FastMCP, best practices are built in.**

**That's why FastMCP is the standard framework for working with MCP.** FastMCP 1.0 was incorporated into the official MCP Python SDK in 2024. Today, the actively maintained standalone project is downloaded a million times a day, and some version of FastMCP powers 70% of MCP servers across all languages.

FastMCP has three pillars:

<table>
<tr>
<td align="center" valign="top" width="33%">
<a href="https://gofastmcp.com/servers/server">
<img src="https://raw.githubusercontent.com/PrefectHQ/fastmcp/main/docs/assets/images/servers-card.png" alt="Servers" />
<br /><strong>Servers</strong>
</a>
<br />Expose tools, resources, and prompts to LLMs.
</td>
<td align="center" valign="top" width="33%">
<a href="https://gofastmcp.com/apps/overview">
<img src="https://raw.githubusercontent.com/PrefectHQ/fastmcp/main/docs/assets/images/apps-card.png" alt="Apps" />
<br /><strong>Apps</strong>
</a>
<br />Give your tools interactive UIs rendered directly in the conversation.
</td>
<td align="center" valign="top" width="33%">
<a href="https://gofastmcp.com/clients/client">
<img src="https://raw.githubusercontent.com/PrefectHQ/fastmcp/main/docs/assets/images/clients-card.png" alt="Clients" />
<br /><strong>Clients</strong>
</a>
<br />Connect to any MCP server — local or remote, programmatic or CLI.
</td>
</tr>
</table>

**[Servers](https://gofastmcp.com/servers/server)** wrap your Python functions into MCP-compliant tools, resources, and prompts. **[Clients](https://gofastmcp.com/clients/client)** connect to any server with full protocol support. And **[Apps](https://gofastmcp.com/apps/overview)** give your tools interactive UIs rendered directly in the conversation.

Ready to build? Start with the [installation guide](https://gofastmcp.com/getting-started/installation) or jump straight to the [quickstart](https://gofastmcp.com/getting-started/quickstart).

## Run FastMCP in production with Horizon

FastMCP is the standard way to build MCP servers. **[Prefect Horizon](https://www.prefect.io/horizon?utm_source=github&utm_medium=readme&utm_campaign=readme_horizon&utm_content=readme_body)** is the enterprise MCP gateway for running them safely.

Built by the FastMCP team, Horizon packages the best practices we've learned shipping the world's most popular MCP framework.

Deploy FastMCP servers from GitHub with branch previews and instant rollback. Create a private registry of every MCP your company uses. Secure access with SSO and tool-level RBAC. Get audit logs, observability, and governance across your MCP stack. Remix approved tools into purpose-built endpoints for teams and agents.

Start with FastMCP. [Scale with Horizon →](https://www.prefect.io/horizon?utm_source=github&utm_medium=readme&utm_campaign=readme_horizon&utm_content=readme_cta)

## Installation

We recommend installing FastMCP with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install fastmcp
```

For full installation instructions, including verification and upgrading, see the [**Installation Guide**](https://gofastmcp.com/getting-started/installation).

**Upgrading?** We have guides for:
- [Upgrading from FastMCP v2](https://gofastmcp.com/getting-started/upgrading/from-fastmcp-2)
- [Upgrading from the MCP Python SDK](https://gofastmcp.com/getting-started/upgrading/from-mcp-sdk)
- [Upgrading from the low-level SDK](https://gofastmcp.com/getting-started/upgrading/from-low-level-sdk)

> [!NOTE]
> If `import fastmcp` fails right after a `pip` upgrade from FastMCP 3.2 or earlier, run `pip install --force-reinstall fastmcp`. See [Troubleshooting](https://gofastmcp.com/getting-started/installation#troubleshooting) for why this happens (`uv` is unaffected).

## 📚 Documentation

FastMCP's complete documentation is available at **[gofastmcp.com](https://gofastmcp.com)**, including detailed guides, API references, and advanced patterns.

Documentation is also available in [llms.txt format](https://llmstxt.org/), which is a simple markdown standard that LLMs can consume easily:

- [`llms.txt`](https://gofastmcp.com/llms.txt) is essentially a sitemap, listing all the pages in the documentation.
- [`llms-full.txt`](https://gofastmcp.com/llms-full.txt) contains the entire documentation. Note this may exceed the context window of your LLM.

**Community:** Join our [Discord server](https://discord.gg/uu8dJCgttd) to connect with other FastMCP developers and share what you're building.

## Contributing

We welcome contributions! See the [Contributing Guide](https://gofastmcp.com/development/contributing) for setup instructions, testing requirements, and PR guidelines.



================================================
FILE: CLAUDE.md
================================================
# FastMCP Development Guidelines

> **Audience**: LLM-driven engineering agents and human developers

> **Note**: `AGENTS.md` is a symlink to this file. Edit `CLAUDE.md` directly.

FastMCP is a comprehensive Python framework (Python ≥3.10) for building Model Context Protocol (MCP) servers and clients. This is the actively maintained v2.0 providing a complete toolkit for the MCP ecosystem.

## Required Development Workflow

**CRITICAL**: Always run these commands in sequence before committing.

```bash
uv sync                              # Install dependencies
uv run pytest -n auto                # Run full test suite
```

In addition, you must pass static checks. This is generally done as a pre-commit hook with `prek` but you can run it manually with:

```bash
uv run prek run --all-files          # Ruff + Prettier + ty
```

**Tests must pass and lint/typing must be clean before committing.**

## Repository Structure

| Path              | Purpose                                |
| ----------------- | -------------------------------------- |
| `fastmcp_slim/fastmcp/` | Library source code                    |
| `├─server/`       | Server implementation                  |
| `│ ├─auth/`       | Authentication providers               |
| `│ └─middleware/` | Error handling, logging, rate limiting |
| `├─client/`       | Client SDK                             |
| `│ └─auth/`       | Client authentication                  |
| `├─tools/`        | Tool definitions                       |
| `├─resources/`    | Resources and resource templates       |
| `├─prompts/`      | Prompt templates                       |
| `├─cli/`          | CLI commands                           |
| `└─utilities/`    | Shared utilities                       |
| `tests/`          | Pytest suite                           |
| `docs/`           | Mintlify docs (gofastmcp.com)          |

## Core MCP Objects

When modifying MCP functionality, changes typically need to be applied across all object types:

- **Tools** (`src/tools/`)
- **Resources** (`src/resources/`)
- **Resource Templates** (`src/resources/`)
- **Prompts** (`src/prompts/`)

**Before writing cross-component logic (dedupe, grouping, lookups, identity checks), read `FastMCPComponent` in `fastmcp_slim/fastmcp/utilities/components.py`.** The base class defines the shared surface — `name`, `version`, `tags`, `meta`, and critically the `key` property which is the canonical MCP identity (encodes type, identifier, and version). Prefer `item.key` over ad-hoc `name or uri or uri_template` fallbacks; overrides in `Resource` and `ResourceTemplate` already handle URI-based identity, and `.key` includes the version suffix so variants of the same component don't falsely collide.

## Development Rules

**Read `CONTRIBUTING.md` before opening issues or PRs.** It describes when PRs are appropriate, what we expect from enhancement proposals, and what we'll close without review.

### Git & CI

- Prek hooks are required (run automatically on commits)
- Never amend commits to fix prek failures
- Apply PR labels: bugs/breaking/enhancements/features
- Improvements = enhancements (not features) unless specified
- **NEVER** force-push on collaborative repos
- **ALWAYS** run prek before PRs
- **NEVER** create a release, comment on an issue, or open a PR unless specifically instructed to do so.
- **NEVER** merge a PR marked as do-not-merge or draft. Check title, body, AND labels for `[DNM]`, `DNM`, `DO NOT MERGE`, `DON'T MERGE`, `DONT MERGE`, `do-not-merge`, `dont-merge`, `[DRAFT]`, or `DRAFT` (case-insensitive, any variation — some authors use `[DRAFT]` in the title even when `isDraft` is false). Authors use these as hard stops — respect them even if CI is green and review looks clean. When triaging a batch of PRs, filter these out up front AND re-check each one's labels immediately before merging, since labels can change mid-session.
- **ALWAYS** read review-bot comments before approving a PR. CodeRabbit and chatgpt-codex-connector (Codex) leave substantive review comments on most PRs in this repo — these bots have read the diff and often flag real issues that aren't in the PR description. Use `gh pr view <num> --comments` and read the bot feedback as part of review. Unlike proposed solutions from issue reporters, review-bot feedback should be evaluated on its merits, not discounted.
- **Be constructively skeptical of bot review comments on your own PRs.** CodeRabbit, Codex, and claude[bot] run a fresh review pass on every push, which means a PR with active churn can accumulate bot comments in a stream that never really ends — each fix surfaces a new edge case the next pass can flag. Most of the early feedback is real and worth acting on; diminishing returns set in fast. Evaluate each comment on its merits, the same way you would a human reviewer: is this a real bug users will hit, or a hypothetical that requires an adversarial setup? Does the fix introduce more complexity than the problem? Has the bot missed context that's obvious to a human reader (a `*,` keyword-only marker, a design decision documented elsewhere, something already resolved on a later commit)? When a comment is pedantic, a false positive, or flagging something already fixed, reply on the thread explaining the reasoning and move on — don't keep iterating just because more comments arrive. If you find yourself three rounds deep and the feedback is shifting toward "what if someone does X" hypotheticals, you're past the point where each fix is improving the PR. Stop, document the contract as-is, and ship.

### Outbound Comments and Shell Interpolation

- Never pass GitHub, Linear, or Slack comment bodies inline through shell arguments when the body contains `$`, `${...}`, backticks, `$(...)`, environment-variable examples, secrets, or config interpolation examples.
- Use a body file or structured API payload for outbound comments, then inspect the exact outgoing text before posting. Prefer `gh ... --body-file /path/to/comment.md` over `--body "..."`.
- When explaining environment interpolation, use placeholders and fenced code blocks. Never include raw `.env` contents in outbound comments.

### Releases

Only cut releases when the maintainer explicitly asks. Tags follow `v<version>` (e.g., `v3.2.0`). Always pass `--generate-notes` so the auto-generated changelog appears at the bottom.

**The title pun is critical.** Titles follow `v<version>: <pun>` where the pun relates to the most important theme of the release. Propose multiple options and let the maintainer choose — never pick one yourself. Look at recent releases for tone (e.g., "Code to Joy" for the code mode release, "Three at Last" for 3.0).

Write the maintainer-approved handwritten notes to a temporary file, then create the release. `--generate-notes` appends the auto-generated changelog after the handwritten content.

```bash
gh release create v3.2.0 --target main --title "v3.2.0: Theme Here" --generate-notes --notes-start-tag v3.1.1 --notes-file /tmp/release-notes.md
```

**Always pass `--notes-start-tag <last-stable-tag>`.** Without it, `--generate-notes` picks the most recent prior tag as the changelog start point — and if a prerelease exists (e.g. `v3.4.0b1`), it starts from *that*, silently truncating the PR list to only the commits since the beta. Pin it to the last stable release (e.g. `v3.3.1` when cutting `v3.4.0`). Verify after: the compare link at the bottom of the generated notes should read `v<last-stable>...v<new>`.

Most releases target `main`, but maintenance or backport releases may target a different branch (e.g., `release/2.x`). Confirm the target with the maintainer if there's any ambiguity.

The handwritten notes are prepended above the auto-generated changelog and are the part that matters. Do not include a title in the notes body — the release title (`v{version}: {pun}`) already serves as the heading. Work with the maintainer to draft the notes — propose a draft, get feedback, iterate. Do not publish without the maintainer's sign-off.

**Before drafting, always read recent existing releases** (`gh release list` then `gh release view <tag>`) to absorb the voice, structure, and level of detail. Each release builds on the tone of previous ones — don't guess at the style from these instructions alone.

**To preview what PRs will be in the release** before it's cut, call the GitHub generate-notes API. This returns the exact auto-generated changelog that `--generate-notes` would append, so you can see the full PR list — useful for picking a pun theme and making sure nothing's been missed:

```bash
gh api -X POST repos/PrefectHQ/fastmcp/releases/generate-notes \
  -f tag_name=v3.2.3 \
  -f target_commitish=main \
  -f previous_tag_name=v3.2.2 \
  --jq '.body'
```

**Point releases** (3.0, 3.1, 3.2) get narrative prose: open with the theme of the release, then walk through headline features conceptually — what they enable, why they matter, how they fit together. Write it the way a blog post reads, not a changelog. Multiple paragraphs, code examples where they clarify.

**Patch releases** (3.1.1, 3.0.2) get 1-2 sentences explaining what broke and what the fix does. Keep it minimal — the auto-generated changelog has the details.

**Merge the docs changelog PR *before* cutting the release, not after.** The post-publish `update-published-docs` job force-pushes the `published-docs` branch (which gofastmcp.com serves) to the *released commit* — so the changelog entry only reaches the live site if it's already in t

[... Content truncated due to length ...]

</details>

<details>
<summary>Repository analysis for https://github.com/modelcontextprotocol/modelcontextprotocol</summary>

# Repository analysis for https://github.com/modelcontextprotocol/modelcontextprotocol

## Summary
Repository: modelcontextprotocol/modelcontextprotocol
Commit: 1304c8fe5f079358a0ff22b6afdb675dc96d75d1
Files analyzed: 540

Estimated tokens: 1.4M

## File tree
```Directory structure:
└── modelcontextprotocol-modelcontextprotocol/
    ├── README.md
    ├── AGENTS.md
    ├── ANTITRUST.md
    ├── CODE_OF_CONDUCT.md
    ├── CONTRIBUTING.md
    ├── eslint.config.mjs
    ├── GOVERNANCE.md
    ├── LICENSE
    ├── MAINTAINERS.md
    ├── migrate_seps.js
    ├── package.json
    ├── SECURITY.md
    ├── tsconfig.json
    ├── typedoc.config.mjs
    ├── typedoc.plugin.mjs
    ├── .npmrc
    ├── .nvmrc
    ├── .prettierignore
    ├── .prettierrc.json
    ├── .prototools
    ├── blog/
    │   ├── go.mod
    │   ├── go.sum
    │   ├── hugo.toml
    │   ├── archetypes/
    │   │   └── default.md
    │   ├── assets/
    │   │   └── css/
    │   │       └── extended/
    │   │           └── custom.css
    │   ├── content/
    │   │   ├── _index.md
    │   │   ├── archives.md
    │   │   ├── search.md
    │   │   └── posts/
    │   │       ├── 2025-07-29-prompts-for-automation.md
    │   │       ├── 2025-07-31-governance-for-mcp.md
    │   │       ├── 2025-09-05-php-sdk.md
    │   │       ├── 2025-09-08-mcp-registry-preview.md
    │   │       ├── 2025-09-26-mcp-next-version-update.md
    │   │       ├── 2025-11-03-using-server-instructions.md
    │   │       ├── 2025-11-20-adopting-mcpb.md
    │   │       ├── 2025-11-21-mcp-apps.md
    │   │       ├── 2025-11-25-first-mcp-anniversary.md
    │   │       ├── 2025-11-28-sep-process-update.md
    │   │       ├── 2025-12-09-mcp-joins-agentic-ai-foundation.md
    │   │       ├── 2025-12-19-mcp-transport-future.md
    │   │       ├── 2026-01-22-core-maintainer-update.md
    │   │       ├── 2026-01-26-mcp-apps.md
    │   │       ├── 2026-03-09-roadmap-update.md
    │   │       ├── 2026-03-11-understanding-mcp-extensions.md
    │   │       ├── 2026-03-16-tool-annotations.md
    │   │       ├── 2026-04-08-maintainer-update.md
    │   │       ├── 2026-05-21-mcp-2026-07-28-rc.md
    │   │       ├── welcome-to-mcp-blog.md
    │   │       └── client_registration/
    │   │           └── index.md
    │   ├── layouts/
    │   │   ├── baseof.html
    │   │   ├── index.html
    │   │   ├── _default/
    │   │   │   ├── archives.html
    │   │   │   ├── list.html
    │   │   │   ├── search.html
    │   │   │   └── terms.html
    │   │   ├── _markup/
    │   │   │   └── render-codeblock-mermaid.html
    │   │   ├── partials/
    │   │   │   ├── extend_head.html
    │   │   │   ├── footer.html
    │   │   │   ├── google_analytics.html
    │   │   │   ├── header.html
    │   │   │   ├── post_meta.html
    │   │   │   ├── post_nav_links.html
    │   │   │   └── templates/
    │   │   │       └── schema_json.html
    │   │   └── shortcodes/
    │   │       ├── button.html
    │   │       └── youtube.html
    │   └── static/
    │       ├── CNAME
    │       └── posts/
    │           └── images/
    │               ├── claude_code_instructions.JPG
    │               ├── first-mcp-anniversary/
    │               │   ├── david-mcp-may.webp
    │               │   ├── kent-jarvis-mcp.webp
    │               │   ├── maintainers-meetup.webp
    │               │   ├── maintainers-write.webp
    │               │   ├── mcp-night-github-mcp.webp
    │               │   ├── mcp-night.webp
    │               │   └── mcp.webp
    │               └── maintainer-update-2026-04/
    │                   └── clare.webp
    ├── CLAUDE.md -> AGENTS.md
    ├── docs/
    │   ├── docs.json
    │   ├── examples.mdx
    │   ├── footer.js
    │   ├── spec-version-warning.js
    │   ├── style.css
    │   ├── community/
    │   │   ├── antitrust.mdx
    │   │   ├── charter-template.mdx
    │   │   ├── communication.mdx
    │   │   ├── contributing.mdx
    │   │   ├── contributor-ladder.mdx
    │   │   ├── design-principles.mdx
    │   │   ├── feature-lifecycle.mdx
    │   │   ├── governance.mdx
    │   │   ├── sdk-tiers.mdx
    │   │   ├── sep-guidelines.mdx
    │   │   ├── working-interest-groups.mdx
    │   │   ├── auth/
    │   │   │   └── charter.mdx
    │   │   ├── file-uploads/
    │   │   │   └── charter.mdx
    │   │   ├── inspector-v2/
    │   │   │   └── charter.mdx
    │   │   ├── interceptors/
    │   │   │   └── charter.mdx
    │   │   ├── registry/
    │   │   │   └── charter.mdx
    │   │   ├── sdk/
    │   │   │   └── charter.mdx
    │   │   ├── seps/
    │   │   │   └── 2243-http-standardization.mdx
    │   │   ├── server-card/
    │   │   │   └── charter.mdx
    │   │   ├── skills-over-mcp/
    │   │   │   └── charter.mdx
    │   │   ├── tool-annotations/
    │   │   │   └── charter.mdx
    │   │   └── triggers-events/
    │   │       └── charter.mdx
    │   ├── development/
    │   │   └── roadmap.mdx
    │   ├── docs/
    │   │   ├── sdk.mdx
    │   │   ├── develop/
    │   │   │   ├── build-client.mdx
    │   │   │   ├── build-server.mdx
    │   │   │   ├── build-with-agent-skills.mdx
    │   │   │   ├── connect-local-servers.mdx
    │   │   │   ├── connect-remote-servers.mdx
    │   │   │   └── clients/
    │   │   │       └── client-best-practices.mdx
    │   │   ├── getting-started/
    │   │   │   └── intro.mdx
    │   │   ├── learn/
    │   │   │   ├── architecture.mdx
    │   │   │   ├── client-concepts.mdx
    │   │   │   ├── server-concepts.mdx
    │   │   │   └── versioning.mdx
    │   │   ├── tools/
    │   │   │   ├── debugging.mdx
    │   │   │   └── inspector.mdx
    │   │   └── tutorials/
    │   │       └── security/
    │   │           ├── authorization.mdx
    │   │           └── security_best_practices.mdx
    │   ├── extensions/
    │   │   ├── client-matrix.mdx
    │   │   ├── overview.mdx
    │   │   ├── apps/
    │   │   │   ├── build.mdx
    │   │   │   └── overview.mdx
    │   │   ├── auth/
    │   │   │   ├── enterprise-managed-authorization.mdx
    │   │   │   ├── oauth-client-credentials.mdx
    │   │   │   └── overview.mdx
    │   │   └── tasks/
    │   │       └── overview.mdx
    │   ├── images/
    │   │   └── java/
    │   │       └── class-diagrams.puml
    │   ├── registry/
    │   │   ├── about.mdx
    │   │   ├── authentication.mdx
    │   │   ├── faq.mdx
    │   │   ├── github-actions.mdx
    │   │   ├── moderation-policy.mdx
    │   │   ├── package-types.mdx
    │   │   ├── quickstart.mdx
    │   │   ├── registry-aggregators.mdx
    │   │   ├── remote-servers.mdx
    │   │   ├── terms-of-service.mdx
    │   │   └── versioning.mdx
    │   ├── seps/
    │   │   ├── 1024-mcp-client-security-requirements-for-local-server-.mdx
    │   │   ├── 1034--support-default-values-for-all-primitive-types-in.mdx
    │   │   ├── 1036-url-mode-elicitation-for-secure-out-of-band-intera.mdx
    │   │   ├── 1046-support-oauth-client-credentials-flow-in-authoriza.mdx
    │   │   ├── 1302-formalize-working-groups-and-interest-groups-in-mc.mdx
    │   │   ├── 1303-input-validation-errors-as-tool-execution-errors.mdx
    │   │   ├── 1319-decouple-request-payload-from-rpc-methods-definiti.mdx
    │   │   ├── 1330-elicitation-enum-schema-improvements-and-standards.mdx
    │   │   ├── 1577--sampling-with-tools.mdx
    │   │   ├── 1613-establish-json-schema-2020-12-as-default-dialect-f.mdx
    │   │   ├── 1686-tasks.mdx
    │   │   ├── 1699-support-sse-polling-via-server-side-disconnect.mdx
    │   │   ├── 1730-sdks-tiering-system.mdx
    │   │   ├── 1850-pr-based-sep-workflow.mdx
    │   │   ├── 1865-mcp-apps-interactive-user-interfaces-for-mcp.mdx
    │   │   ├── 2085-governance-succession-and-amendment.mdx
    │   │   ├── 2106-json-schema-2020-12.mdx
    │   │   ├── 2133-extensions.mdx
    │   │   ├── 2148-contributor-ladder.mdx
    │   │   ├── 2149-working-group-charter-template.mdx
    │   │   ├── 2164-resource-not-found-error.mdx
    │   │   ├── 2207-oidc-refresh-token-guidance.mdx
    │   │   ├── 2243-http-standardization.mdx
    │   │   ├── 2260-Require-Server-requests-to-be-associated-with-Client-requests.mdx
    │   │   ├── 2322-MRTR.mdx
    │   │   ├── 2468-recommend-issuer-claim-for-auth.mdx
    │   │   ├── 2484-conformance-tests-required-for-final-seps.mdx
    │   │   ├── 2549-TTL-for-list-results.mdx
    │   │   ├── 2567-sessionless-mcp.mdx
    │   │   ├── 2575-stateless-mcp.mdx
    │   │   ├── 2577-deprecate-roots-sampling-and-logging.mdx
    │   │   ├── 2596-spec-feature-lifecycle-and-deprecation.mdx
    │   │   ├── 2663-tasks-extension.mdx
    │   │   ├── 414-request-meta.mdx
    │   │   ├── 932-model-context-protocol-governance.mdx
    │   │   ├── 973-expose-additional-metadata-for-implementations-res.mdx
    │   │   ├── 985-align-oauth-20-protected-resource-metadata-with-rf.mdx
    │   │   ├── 986-specify-format-for-tool-names.mdx
    │   │   ├── 990-enable-enterprise-idp-policy-controls-during-mcp-o.mdx
    │   │   ├── 991-enable-url-based-client-registration-using-oauth-c.mdx
    │   │   ├── 994-shared-communication-practicesguidelines.mdx
    │   │   └── index.mdx
    │   ├── snippets/
    │   │   └── snippet-intro.mdx
    │   ├── specification/
    │   │   ├── 2024-11-05/
    │   │   │   ├── index.mdx
    │   │   │   ├── architecture/
    │   │   │   │   └── index.mdx
    │   │   │   ├── basic/
    │   │   │   │   ├── index.mdx
    │   │   │   │   ├── lifecycle.mdx
    │   │   │   │   ├── messages.mdx
    │   │   │   │   ├── transports.mdx
    │   │   │   │   └── utilities/
    │   │   │   │       ├── cancellation.mdx
    │   │   │   │       ├── ping.mdx
    │   │   │   │       └── progress.mdx
    │   │   │   ├── client/
    │   │   │   │   ├── roots.mdx
    │   │   │   │   └── sampling.mdx
    │   │   │   └── server/
    │   │   │       ├── index.mdx
    │   │   │       ├── prompts.mdx
    │   │   │       ├── resources.mdx
    │   │   │       ├── tools.mdx
    │   │   │       └── utilities/
    │   │   │           ├── completion.mdx
    │   │   │           ├── logging.mdx
    │   │   │           └── pagination.mdx
    │   │   ├── 2025-03-26/
    │   │   │   ├── changelog.mdx
    │   │   │   ├── index.mdx
    │   │   │   ├── architecture/
    │   │   │   │   └── index.mdx
    │   │   │   ├── basic/
    │   │   │   │   ├── authorization.mdx
    │   │   │   │   ├── index.mdx
    │   │   │   │   ├── lifecycle.mdx
    │   │   │   │   ├── transports.mdx
    │   │   │   │   └── utilities/
    │   │   │   │       ├── cancellation.mdx
    │   │   │   │       ├── ping.mdx
    │   │   │   │       └── progress.mdx
    │   │   │   ├── client/
    │   │   │   │   ├── roots.mdx
    │   │   │   │   └── sampling.mdx
    │   │   │   └── server/
    │   │   │       ├── index.mdx
    │   │   │       ├── prompts.mdx
    │   │   │       ├── resources.mdx
    │   │   │       ├── tools.mdx
    │   │   │       └── utilities/
    │   │   │           ├── completion.mdx
    │   │   │           ├── logging.mdx
    │   │   │           └── pagination.mdx
    │   │   ├── 2025-06-18/
    │   │   │   ├── changelog.mdx
    │   │   │   ├── index.mdx
    │   │   │   ├── schema.mdx
    │   │   │   ├── architecture/
    │   │   │   │   └── index.mdx
    │   │   │   ├── basic/
    │   │   │   │   ├── authorization.mdx
    │   │   │   │   ├── index.mdx
    │   │   │   │   ├── lifecycle.mdx
    │   │   │   │   ├── transports.mdx
    │   │   │   │   └── utilities/
    │   │   │   │       ├── cancellation.mdx
    │   │   │   │       ├── ping.mdx
    │   │   │   │       └── progress.mdx
    │   │   │   ├── client/
    │   │   │   │   ├── elicitation.mdx
    │   │   │   │   ├── roots.mdx
    │   │   │   │   └── sampling.mdx
    │   │   │   └── server/
    │   │   │       ├── index.mdx
    │   │   │       ├── prompts.mdx
    │   │   │       ├── resources.mdx
    │   │   │       ├── tools.mdx
    │   │   │       └── utilities/
    │   │   │           ├── completion.mdx
    │   │   │           ├── logging.mdx
    │   │   │           └── pagination.mdx
    │   │   ├── 2025-11-25/
    │   │   │   ├── changelog.mdx
    │   │   │   ├── index.mdx
    │   │   │   ├── schema.mdx
    │   │   │   ├── architecture/
    │   │   │   │   └── index.mdx
    │   │   │   ├── basic/
    │   │   │   │   ├── authorization.mdx
    │   │   │   │   ├── index.mdx
    │   │   │   │   ├── lifecycle.mdx
    │   │   │   │   ├── transports.mdx
    │   │   │   │   └── utilities/
    │   │   │   │       ├── cancellation.mdx
    │   │   │   │       ├── ping.mdx
    │   │   │   │       ├── progress.mdx
    │   │   │   │       └── tasks.mdx
    │   │   │   ├── client/
    │   │   │   │   ├── elicitation.mdx
    │   │   │   │   ├── roots.mdx
    │   │   │   │   └── sampling.mdx
    │   │   │   └── server/
    │   │   │       ├── index.mdx
    │   │   │       ├── prompts.mdx
    │   │   │       ├── resources.mdx
    │   │   │       ├── tools.mdx
    │   │   │       └── utilities/
    │   │   │           ├── completion.mdx
    │   │   │           ├── logging.mdx
    │   │   │           └── pagination.mdx
    │   │   └── draft/
    │   │       ├── changelog.mdx
    │   │       ├── deprecated.mdx
    │   │       ├── index.mdx
    │   │       ├── schema.mdx
    │   │       ├── architecture/
    │   │       │   └── index.mdx
    │   │       ├── basic/
    │   │       │   ├── index.mdx
    │   │       │   ├── versioning.mdx
    │   │       │   ├── authorization/
    │   │       │   │   ├── authorization-server-discovery.mdx
    │   │       │   │   ├── client-registration.mdx
    │   │       │   │   ├── index.mdx
    │   │       │   │   └── security-considerations.mdx
    │   │       │   ├── patterns/
    │   │       │   │   ├── cancellation.mdx
    │   │       │   │   ├── index.mdx
    │   │       │   │   ├── mrtr.mdx
    │   │       │   │   ├── progress.mdx
    │   │       │   │   └── subscriptions.mdx
    │   │       │   └── transports/
    │   │       │       ├── index.mdx
    │   │       │       ├── stdio.mdx
    │   │       │       └── streamable-http.mdx
    │   │       ├── client/
    │   │       │   ├── elicitation.mdx
    │   │       │   ├── roots.mdx
    │   │       │   └── sampling.mdx
    │   │       └── server/
    │   │           ├── discover.mdx
    │   │           ├── index.mdx
    │   │           ├── prompts.mdx
    │   │           ├── resources.mdx
    │   │           ├── tools.mdx
    │   │           └── utilities/
    │   │               ├── caching.mdx
    │   │               ├── completion.mdx
    │   │               ├── logging.mdx
    │   │               └── pagination.mdx
    │   ├── .mintlify/
    │   │   └── skills/
    │   │       ├── draft-sep -> draft-sep
    │   │       └── search-mcp-github -> search-mcp-github
    │   └── .well-known/
    │       └── security.txt
    ├── plugins/
    │   └── mcp-spec/
    │       ├── README.md
    │       ├── skills/
    │       │   ├── draft-sep/
    │       │   │   └── SKILL.md
    │       │   └── search-mcp-github/
    │       │       └── SKILL.md
    │       └── .claude-plugin/
    │           └── plugin.json
    ├── schema/
    │   ├── 2024-11-05/
    │   │   ├── schema.json
    │   │   └── schema.ts
    │   ├── 2025-03-26/
    │   │   ├── schema.json
    │   │   └── schema.ts
    │   ├── 2025-06-18/
    │   │   ├── schema.json
    │   │   ├── schema.mdx
    │   │   └── schema.ts
    │   ├── 2025-11-25/
    │   │   ├── schema.json
    │   │   ├── schema.mdx
    │   │   └── schema.ts
    │   └── draft/
    │       ├── schema.json
    │       ├── schema.mdx
    │       ├── schema.ts
    │       └── examples/
    │           ├── AudioContent/
    │           │   └── audio-wav-content.json
    │           ├── BlobResourceContents/
    │           │   └── image-file-contents.json
    │           ├── BooleanSchema/
    │           │   └── boolean-input-schema.json
    │           ├── CallToolRequest/
    │           │   └── call-tool-request.json
    │           ├── CallToolRequestParams/
    │           │   ├── get-weather-tool-call-params.json
    │           │   └── tool-call-params-with-progress-token.json
    │           ├── CallToolResult/
    │           │   ├── invalid-tool-input-error.json
    │           │   ├── result-with-array-structured-content.json
    │           │   ├── result-with-structured-content.json
    │           │   └── result-with-unstructured-text.json
    │           ├── CallToolResultResponse/
    │           │   └── call-tool-result-response.json
    │           ├── CancelledNotification/
    │           │   └── user-requested-cancellation.json
    │           ├── CancelledNotificationParams/
    │           │   └── user-requested-cancellation.json
    │           ├── ClientCapabilities/
    │           │   ├── elicitation-form-and-url-mode-support.json
    │           │   ├── elicitation-form-only-implicit.json
    │           │   ├── extensions-ui-mime-types.json
    │           │   ├── roots-minimum-baseline-support.json
    │           │   ├── sampling-context-inclusion-support-deprecated.json
    │           │   ├── sampling-minimum-baseline-support.json
    │           │   └── sampling-tool-use-support.json
    │           ├── CompleteRequest/
    │           │   └── completion-request.json
    │           ├── CompleteRequestParams/
    │           │   ├── prompt-argument-completion-with-context.json
    │           │   └── prompt-argument-completion.json
    │           ├── CompleteResult/
    │           │   ├── multiple-completion-values-with-more-available.json
    │           │   └── single-completion-value.json
    │           ├── CompleteResultResponse/
    │           │   └── completion-result-response.json
    │           ├── CreateMessageRequest/
    │           │   └── sampling-request.json
    │           ├── CreateMessageRequestParams/
    │           │   ├── basic-request.json
    │           │   ├── follow-up-with-tool-results.json
    │           │   └── request-with-tools.json
    │           ├── CreateMessageResult/
    │           │   ├── final-response.json
    │           │   ├── text-response.json
    │           │   └── tool-use-response.json
    │           ├── DiscoverRequest/
    │           │   └── server-discover-request.json
    │           ├── DiscoverResult/
    │           │   └── server-capabilities-discovery.json
    │           ├── DiscoverResultResponse/
    │           │   └── discover-result-response.json
    │           ├── ElicitationCompleteNotification/
    │           │   └── elicitation-complete.json
    │           ├── ElicitRequest/
    │           │   └── elicitation-request.json
    │           ├── ElicitRequestFormParams/
    │           │   ├── elicit-multiple-fields.json
    │           │   └── elicit-single-field.json
    │           ├── ElicitRequestURLParams/
    │           │   └── elicit-sensitive-data.json
    │           ├── ElicitResult/
    │           │   ├── accept-url-mode-no-content.json
    │           │   ├── input-multiple-fields.json
    │           │   └── input-single-field.json
    │           ├── EmbeddedResource/
    │           │   └── embedded-file-resource-with-annotations.json
    │           ├── GetPromptRequest/
    │           │   └── get-prompt-request.json
    │           ├── GetPromptRequestParams/
    │           │   └── get-code-review-prompt.json
    │           ├── GetPromptResult/
    │           │   └── code-review-prompt.json
    │           ├── GetPromptResultResponse/
    │           │   └── get-prompt-result-response.json
    │           ├── ImageContent/
    │           │   └── image-png-content-with-annotations.json
    │           ├── InputRequests/
    │           │   └── elicitation-and-sampling-input-requests.json
    │           ├── InputRequiredResult/
    │           │   ├── input-required-result-with-elicitation-and-sampling-and-request-state.json
    │           │   └── input-required-result-with-request-state-only.json
    │           ├── InputResponses/
    │           │   └── elicitation-and-sampling-input-responses.json
    │           ├── InternalError/
    │           │   └── unexpected-error.json
    │           ├── InvalidParamsError/
    │           │   ├── invalid-cursor.json
    │           │   ├── invalid-tool-arguments.json
    │           │   ├── unknown-prompt.json
    │           │   └── unknown-tool.json
    │           ├── ListPromptsRequest/
    │           │   └── list-prompts-request.json
    │           ├── ListPromptsResult/
    │           │   └── prompts-list-with-cursor-and-ttl.json
    │           ├── ListPromptsResultResponse/
    │           │   └── list-prompts-result-response.json
    │           ├── ListResourcesRequest/
    │           │   └── list-resources-request.json
    │           ├── ListResourcesResult/
    │           │   └── resources-list-with-cursor-and-ttl.json
    │           ├── ListResourcesResultResponse/
    │           │   └── list-resources-result-response.json
    │           ├── ListResourceTemplatesRequest/
    │           │   └── list-resource-templates-request.json
    │           ├── ListResourceTemplatesResult/
    │           │   └── resource-templates-list-with-cursor-and-ttl.json
    │           ├── ListResourceTemplatesResultResponse/
    │           │   └── list-resource-templates-result-response.json
    │           ├── ListRootsRequest/
    │           │   └── list-roots-request.json
    │           ├── ListRootsResult/
    │           │   ├── multiple-root-directories.json
    │           │   └── single-root-directory.json
    │           ├── ListToolsRequest/
    │           │   └── list-tools-request.json
    │           ├── ListToolsResult/
    │           │   └── tools-list-with-cursor-and-ttl.json
    │           ├── ListToolsResultResponse/
    │           │   └── list-tools-result-response.json
    │           ├── LoggingMessageNotification/
    │           │   └── log-database-connection-failed.json
    │           ├── LoggingMessageNotificationParams/
    │           │   └── log-database-connection-failed.json
    │           ├── MethodNotFoundError/
    │           │   └── prompts-not-supported.json
    │           ├── MissingRequiredClientCapabilityError/
    │           │   └── missing-elicitation-capability.json
    │           ├── ModelPreferences/
    │           │   └── with-hints-and-priorities.json
    │           ├── NumberSchema/
    │           │   └── number-input-schema.json
    │           ├── PaginatedRequestParams/
    │           │   └── list-with-cursor.json
    │           ├── ParseError/
    │           │   └── invalid-json.json
    │           ├── ProgressNotification/
    │           │   └── progress-message.json
    │           ├── ProgressNotificationParams/
    │           │   └── progress-message.json
    │           ├── PromptListChangedNotification/
    │           │   └── prompts-list-changed.json
    │           ├── ReadResourceRequest/
    │           │   └── read-resource-request.json
    │           ├── ReadResourceResult/
    │           │   └── file-resource-contents.json
    │           ├── ReadResourceResultResponse/
    │           │   ├── read-resource-result-response-with-ttl.json
    │           │   └── read-resource-result-response.json
    │           ├── Resource/
    │           │   └── file-resource-with-annotations.json
    │           ├── ResourceLink/
    │           │   └── file-resource-link.json
    │           ├── ResourceListChangedNotification/
    │           │   └── resources-list-changed.json
    │           ├── ResourceUpdatedNotification/
    │           │   └── file-resource-updated-notification.json
    │           ├── ResourceUpdatedNotificationParams/
    │           │   └── file-resource-updated.json
    │           ├── Root/
    │           │   └── project-directory.json
    │           ├── SamplingMessage/
    │           │   ├── multiple-content-blocks.json
    │           │   └── single-content-block.json
    │           ├── ServerCapabilities/
    │           │   ├── completions-minimum-baseline-support.json
    │           │   ├── extensions-tasks.json
    │           │   ├── logging-minimum-baseline-support.json
    │           │   ├── prompts-list-changed-notifications.json
    │           │   ├── prompts-minimum-baseline-support.json
    │           │   ├── resources-all-notifications.json
    │           │   ├── resources-list-changed-notifications-only.json
    │           │   ├── resources-minimum-baseline-support.json
    │           │   ├── resources-subscription-to-individual-resource-updates-only.json
    │           │   ├── tools-list-changed-notifications.json
    │           │   └── tools-minimum-baseline-support.json
    │           ├── StringSchema/
    │           │   └── email-input-schema.json
    │           ├── SubscriptionsAcknowledgedNotification/
    │           │   └── listen-acknowledged.json
    │           ├── SubscriptionsListenRequest/
    │           │   └── listen-for-list-changes.json
    │           ├── TextContent/
    │           │   └── text-content.json
    │           ├── TextResourceContents/
    │           │   └── text-file-contents.json
    │           ├── TitledMultiSelectEnumSchema/
    │           │   └── titled-color-multi-select-schema.json
    │           ├── TitledSingleSelectEnumSchema/
    │           │   └── titled-color-select-schema.json
    │           ├── Tool/
    │           │   ├── tool-with-array-output-schema.json
    │           │   ├── tool-with-composition-input-schema.json
    │           │   ├── with-default-2020-12-input-schema.json
    │           │   ├── with-explicit-draft-07-input-schema.json
    │           │   ├── with-no-parameters.json
    │           │   └── with-output-schema-for-structured-content.json
    │           ├── ToolListChangedNotification/
    │           │   └── tools-list-changed.json
    │           ├── ToolResultContent/
    │           │   └── get-weather-tool-result.json
    │           ├── ToolUseContent/
    │           │   └── get-weather-tool-use.json
    │           ├── UnsupportedProtocolVersionError/
    │           │   └── unsupported-version.json
    │           ├── UntitledMultiSelectEnumSchema/
    │           │   └── color-multi-select-schema.json
    │           └── UntitledSingleSelectEnumSchema/
    │               └── color-select-schema.json
    ├── scripts/
    │   ├── check-mdx-comments.ts
    │   ├── generate-schemas.ts
    │   ├── pull-registry-docs.sh
    │   ├── render-seps.ts
    │   └── validate-examples.ts
    ├── seps/
    │   ├── README.md
    │   ├── 1024-mcp-client-security-requirements-for-local-server-.md
    │   ├── 1034--support-default-values-for-all-primitive-types-in.md
    │   ├── 1036-url-mode-elicitation-for-secure-out-of-band-intera.md
    │   ├── 1046-support-oauth-client-credentials-flow-in-authoriza.md
    │   ├── 1302-formalize-working-groups-and-interest-groups-in-mc.md
    │   ├── 1303-input-validation-errors-as-tool-execution-errors.md
    │   ├── 1319-decouple-request-payload-from-rpc-methods-definiti.md
    │   ├── 1330-elicitation-enum-schema-improvements-and-standards.md
    │   ├── 1577--sampling-with-tools.md
    │   ├── 1613-establish-json-schema-2020-12-as-default-dialect-f.md
    │   ├── 1686-tasks.md
    │   ├── 1699-support-sse-polling-via-server-side-disconnect.md
    │   ├── 1730-sdks-tiering-system.md
    │   ├── 1850-pr-based-sep-workflow.md
    │   ├── 1865-mcp-apps-interactive-user-interfaces-for-mcp.md
    │   ├── 2085-governance-succession-and-amendment.md
    │   ├── 2106-json-schema-2020-12.md
    │   ├── 2133-extensions.md
    │   ├── 2148-contributor-ladder.md
    │   ├── 2149-working-group-charter-template.md
    │   ├── 2164-resource-not-found-error.md
    │   ├── 2207-oidc-refresh-token-guidance.md
    │   ├── 2243-http-standardization.md
    │   ├── 2260-Require-Server-requests-to-be-associated-with-Client-requests.md
    │   ├── 2322-MRTR.md
    │   ├── 2468-recommend-issuer-claim-for-auth.md
    │   ├── 2484-conformance-tests-required-for-final-seps.md
    │   ├── 2549-TTL-for-list-results.md
    │   ├── 2567-sessionless-mcp.md
    │   ├── 2575-stateless-mcp.md
    │   ├── 2577-deprecate-roots-sampling-and-logging.md
    │   ├── 2596-spec-feature-lifecycle-and-deprecation.md
    │   ├── 2663-tasks-extension.md
    │   ├── 414-request-meta.md
    │   ├── 932-model-context-protocol-governance.md
    │   ├── 973-expose-additional-metadata-for-implementations-res.md
    │   ├── 985-align-oauth-20-protected-resource-metadata-with-rf.md
    │   ├── 986-specify-format-for-tool-names.md
    │   ├── 990-enable-enterprise-idp-policy-controls-during-mcp-o.md
    │   ├── 991-enable-url-based-client-registration-using-oauth-c.md
    │   ├── 994-shared-communication-practicesguidelines.md
    │   ├── TEMPLATE.md
    │   └── .keep
    ├── tools/
    │   └── sep-automation/
    │       ├── package.json
    │       ├── tsconfig.json
    │       ├── vitest.config.ts
    │       ├── src/
    │       │   ├── config.ts
    │       │   ├── index.ts
    │       │   ├── processor.ts
    │       │   ├── rules.ts
    │       │   ├── types.ts
    │       │   ├── actions/
    │       │   │   ├── comment.ts
    │       │   │   ├── ping.ts
    │       │   │   └── transition.ts
    │       │   ├── github/
    │       │   │   ├── client.ts
    │       │   │   └── types.ts
    │       │   ├── hooks/
    │       │   │   ├── discord.ts
    │       │   │   ├── registry.ts
    │       │   │   └── types.ts
    │       │   ├── maintainers/
    │       │   │   └── resolver.ts
    │       │   ├── sep/
    │       │   │   ├── analyzer.ts
    │       │   │   ├── detector.ts
    │       │   │   └── types.ts
    │       │   └── utils/
    │       │       ├── dates.ts
    │       │       ├── errors.ts
    │       │       └── index.ts
    │       └── test/
    │           ├── mocks.ts
    │           └── unit/
    │               ├── comment.test.ts
    │               ├── config.test.ts
    │               ├── hooks.test.ts
    │               ├── ping.test.ts
    │               ├── sep-types.test.ts
    │               └── transition.test.ts
    ├── .claude-plugin/
    │   └── marketplace.json
    └── .github/
        ├── CODEOWNERS
        ├── dependabot.yml
        ├── labeler.yml
        ├── ISSUE_TEMPLATE/
        │   ├── bug_report.yml
        │   ├── config.yml
        │   └── work_tracking.yml
        └── workflows/
            ├── blog-preview.yml
            ├── cut-release.yml
            ├── deploy-blog.yml
            ├── labeler.yml
            ├── main.yml
            ├── markdown-format.yml
            ├── publish-release.yml
            ├── render-seps.yml
            ├── sep-lifecycle-manual.yml
            ├── sep-lifecycle.yml
            ├── sep-reminder.yml
            ├── slash-commands.yml
            └── stage-blog.yml

```

## Extracted content
================================================
FILE: README.md
================================================
# Model Context Protocol (MCP)

_Just heard of MCP and not sure where to start? Check out our [documentation website](https://modelcontextprotocol.io)._

This repo contains the:

- MCP specification
- MCP protocol schema
- Official MCP documentation

The schema is [defined in TypeScript](schema/2025-11-25/schema.ts) first, but
[made available as JSON Schema](schema/2025-11-25/schema.json) as well, for wider
compatibility.

The official MCP documentation is built using Mintlify and available at
[modelcontextprotocol.io](https://modelcontextprotocol.io).

## Authors

The Model Context Protocol was created by David Soria Parra ([@dsp](https://github.com/dsp)) and Justin Spahr-Summers ([@jspahrsummers](https://github.com/jspahrsummers)).

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE).



================================================
FILE: AGENTS.md
================================================
# Model Context Protocol (MCP)

This repository contains the MCP specification, documentation, and blog.

## Documentation Structure

- `docs/` - Mintlify site (`npm run serve:docs`)
  - `docs/docs/` - guides and tutorials
  - `docs/specification/` - MCP specification (more formal, versioned)
- `blog/` - Hugo blog (`npm run serve:blog`)

### Documentation Guidelines

- When creating flowcharts, and graphs to visualize aspect of the protocol, use mermaid diagrams where
  possible.
- When writing tables, ensure column headers and columns are aligned with whitespace.
- Before pushing or creating PR's ensure that `npm run prep` is free of warnings and errors.

## Specification Versioning

Specifications use **date-based versioning** (YYYY-MM-DD), not semantic versioning:

- `schema/[YYYY-MM-DD]/` and `docs/specification/[YYYY-MM-DD]/` - released versions
- `schema/draft/` and `docs/specification/draft/` - in-progress work

## Schema Generation

TypeScript files are the **source of truth** for the protocol schema:

- Edit: `schema/[version]/schema.ts`
- Generate JSON + docs: `npm run generate:schema`
- This creates both `schema/[version]/schema.json` and the Schema Reference document in `docs/specification/[version]/schema.mdx`

Always regenerate after editing schema files.

## Schema Examples

JSON examples live in `schema/[version]/examples/[TypeName]/`:

- Directory name = schema type (e.g., `Tool/`, `Resource/`)
- Files validate against their directory's type: `Tool/example-name.json` → Tool schema
- Referenced in `schema.ts` via `@includeCode` JSDoc tags

## Agent Skills

When adding a new skill, also add a directory symlink at `docs/.mintlify/skills/<name>` pointing to `../../../plugins/<plugin-name>/skills/<name>` so Mintlify's `.well-known/agent-skills/` and MCP server auto-scan exposes it.

## Useful Commands

```bash
# Dev servers
npm run serve:docs       # Local Mintlify docs server
npm run serve:blog       # Local Hugo blog server

# Generation (run after editing source files)
npm run generate         # Generate all (schema + SEPs)
npm run generate:schema  # Generate JSON schemas + MDX from TypeScript
npm run generate:seps    # Generate SEP documents

# Formatting
npm run format           # Format all (docs + schema)
npm run format:docs      # Format markdown/MDX files
npm run format:schema    # Format schema TypeScript files

# Checks
npm run check            # Run all checks
npm run check:schema     # Check schema (TS, JSON, examples, MDX)
npm run check:docs       # Check docs (format, comments, links)
npm run check:seps       # Check SEP documents

# Workflow
npm run prep             # Full prep before committing (check, generate, format)
```

## Issue Creation

Blank issues are disabled. `gh issue create` and the API bypass the template
chooser, so when filing via CLI or API you **must** use one of the forms in
`.github/ISSUE_TEMPLATE/` and fill in its required fields.

Before filing, check `.github/ISSUE_TEMPLATE/config.yml` — some categories
are redirected out of this repo entirely:

- **SEPs** are pull requests adding a file to `seps/`, not issues
- **SDK bugs** belong in the individual SDK repository
- **Claude MCP behavior** belongs in `anthropics/claude-ai-mcp`

## Commit Guidelines

- Do not include model names or details (e.g., "Claude", "Opus") in commit messages



================================================
FILE: ANTITRUST.md
================================================
This project is part of LF Projects, LLC. See the [LF Projects Antitrust Policy](https://lfprojects.org/policies/antitrust-policy/).



================================================
FILE: CODE_OF_CONDUCT.md
================================================
# Contributor Covenant Code of Conduct

## Our Pledge

We as members, contributors, and leaders pledge to make participation in our community a
harassment-free experience for everyone, regardless of age, body size, visible or
invisible disability, ethnicity, sex characteristics, gender identity and expression,
level of experience, education, socio-economic status, nationality, personal appearance,
race, religion, or sexual identity and orientation.

We pledge to act and interact in ways that contribute to an open, welcoming, diverse,
inclusive, and healthy community.

## Our Standards

Examples of behavior that contributes to a positive environment for our community
include:

- Demonstrating empathy and kindness toward other people
- Being respectful of differing opinions, viewpoints, and experiences
- Giving and gracefully accepting constructive feedback
- Accepting responsibility and apologizing to those affected by our mistakes, and
  learning from the experience
- Focusing on what is best not just for us as individuals, but for the overall community

Examples of unacceptable behavior include:

- The use of sexualized language or imagery, and sexual attention or advances of any kind
- Trolling, insulting or derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information, such as a physical or email address, without
  their explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional
  setting

## Enforcement Responsibilities

Community leaders are responsible for clarifying and enforcing our standards of
acceptable behavior and will take appropriate and fair corrective action in response to
any behavior that they deem inappropriate, threatening, offensive, or harmful.

Community leaders have the right and responsibility to remove, edit, or reject comments,
commits, code, wiki edits, issues, and other contributions that are not aligned to this
Code of Conduct, and will communicate reasons for moderation decisions when appropriate.

## Scope

This Code of Conduct applies within all community spaces, and also applies when an
individual is officially representing the community in public spaces. Examples of
representing our community include using an official e-mail address, posting via an
official social media account, or acting as an appointed representative at an online or
offline event.

## Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported to
the community leaders responsible for enforcement at mcp-coc@anthropic.com. All
complaints will be reviewed and investigated promptly and fairly.

All community leaders are obligated to respect the privacy and security of the reporter
of any incident.

## Enforcement Guidelines

Community leaders will follow these Community Impact Guidelines in determining the
consequences for any action they deem in violation of this Code of Conduct:

### 1. Correction

**Community Impact**: Use of inappropriate language or other behavior deemed
unprofessional or unwelcome in the community.

**Consequence**: A private, written warning from community leaders, providing clarity
around the nature of the violation and an explanation of why the behavior was
inappropriate. A public apology may be requested.

### 2. Warning

**Community Impact**: A violation through a single incident or series of actions.

**Consequence**: A warning with consequences for continued behavior. No interaction with
the people involved, including unsolicited interaction with those enforcing the Code of
Conduct, for a specified period of time. This includes avoiding interactions in community
spaces as well as external channels like social media. Violating these terms may lead to
a temporary or permanent ban.

### 3. Temporary Ban

**Community Impact**: A serious violation of community standards, including sustained
inappropriate behavior.

**Consequence**: A temporary ban from any sort of interaction or public communication
with the community for a specified period of time. No public or private interaction with
the people involved, including unsolicited interaction with those enforcing the Code of
Conduct, is allowed during this period. Violating these terms may lead to a permanent
ban.

### 4. Permanent Ban

**Community Impact**: Demonstrating a pattern of violation of community standards,
including sustained inappropriate behavior, harassment of an individual, or aggression
toward or disparagement of classes of individuals.

**Consequence**: A permanent ban from any sort of public interaction within the
community.

## Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage], version 2.0,
available at https://www.contributor-covenant.org/version/2/0/code_of_conduct.html.

Community Impact Guidelines were inspired by
[Mozilla's code of conduct enforcement ladder](https://github.com/mozilla/diversity).

[homepage]: https://www.contributor-covenant.org

For answers to common questions about this code of conduct, see the FAQ at
https://www.contributor-covenant.org/faq. Translations are available at
https://www.contributor-covenant.org/translations.



================================================
FILE: CONTRIBUTING.md
================================================
# Contributing to Model Context Protocol

Thank you for your interest in contributing to the Model Context Protocol specification, schemas, or docs!
This document outlines how to contribute to this project.

Also see the [overall MCP communication guidelines in our docs](https://modelcontextprotocol.io/community/communication), which explain how and where discussions about changes happen.

## General prerequisites

The following software is required to work on the spec:

- Node.js 24 or above
- TypeScript
- TypeScript JSON Schema (for generating JSON schema)
- [Mintlify](https://mintlify.com/) (optional, for docs)
- nvm (optional, for managing Node versions)

### Getting Started

1. [Fork the repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)

2. Clone your fork:

   ```bash
   git clone https://github.com/YOUR-USERNAME/modelcontextprotocol.git
   cd modelcontextprotocol
   ```

3. Install dependencies:

   ```bash
   nvm install  # install correct Node version
   npm install  # install dependencies
   ```

4. Create a new branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

## Schema changes

Schema changes go in `schema/draft/schema.ts`. To validate your changes, run:

```bash
npm run check:schema:ts
```

`schema/draft/schema.json` and `docs/specification/draft/schema.mdx` are generated from `schema/draft/schema.ts`; do not edit them directly. To generate them, run:

```bash
npm run generate:schema
```

### Resolving merge conflicts in generated files

If your branch conflicts with `main` in generated files (`schema/*/schema.json`, `docs/specification/*/schema.mdx`, `docs/seps/*.mdx`), do not resolve them by hand. Merge `main`, resolve any conflicts in the source files (e.g. `schema/draft/schema.ts`), then regenerate and commit:

```bash
git merge main
npm run generate
git add .
git commit
```

These files are marked with `-merge` in `.gitattributes`, so git keeps your branch's copy and flags them as conflicted instead of inserting conflict markers.

## Documentation changes

Documentation is written in MDX format and in the [`docs`](./docs) directory.

You can preview documentation changes locally by running:

```bash
npm run serve:docs
```

And lint them with:

```bash
npm run check:docs
npm run format
```

> [!NOTE]
> You can run all schema/documentation
> changes at once with `npm run prep`.

## Blog changes

The blog is built using [Hugo](https://gohugo.io/installation/) and located in the [`blog`](./blog) directory.

To preview blog changes locally:

```bash
npm run serve:blog
```

### Documentation Guidelines

When contributing to the documentation:

- Keep content clear, concise, and technically accurate
- Follow the existing file structure and naming conventions
- Include code examples where appropriate
- Use proper MDX formatting and components
- Test all links and code samples
  - You may run `npm run check:docs:links` to look for broken internal links.
- Use appropriate headings: "When to use", "Steps", and "Tips" for tutorials
- Place new pages in appropriate sections (concepts, tutorials, etc.)
- Update `docs.json` when adding new pages
- Follow existing file naming conventions (`kebab-case.mdx`)
- Include proper frontmatter in MDX files

## Specification Proposal Guidelines

Specification changes follow the [SEP process](https://modelcontextprotocol.io/community/sep-guidelines).
Before drafting a proposal, review the [MCP design principles](https://modelcontextprotocol.io/community/design-principles)
— proposals that align with these principles move faster through review.

The shortest summary: explore the problem space and validate that others share the problem,
build a prototype that demonstrates a solution, then write the SEP based on what the
prototype taught you.

## Submitting Changes

1. Push your changes to your fork
2. Submit a pull request to the main repository
3. Follow the pull request template
4. Wait for review

## AI Contributions

> [!IMPORTANT]
>
> If you are using **any kind of AI assistance** to contribute to Model Context Protocol,
> it must be disclosed in the pull request or issue.

We welcome and encourage the use of AI tools to help improve Model Context Protocol. Many valuable contributions
have been enhanced with AI assistance for code generation, issue detection, and feature definition.

That being said, if you are using any kind of AI assistance (e.g., agents such as Claude Code, ChatGPT)
while contributing to Model Context Protocol, **this must be disclosed in the pull request or issue**,
along with the extent to which AI assistance was used (e.g., documentation comments vs. code generation).

If your PR responses or comments are being generated by an AI, disclose that as well.

As an exception, trivial spacing or typo fixes don't need to be disclosed, so long as the changes are
limited to small parts of the code or short phrases.

An example disclosure:

> This PR was written primarily by Claude Code.

Or a more detailed disclosure:

> I consulted ChatGPT to understand the codebase but the solution
> was fully authored manually by myself.

Failure to disclose this is first and foremost rude to the human operators on the other end of the pull request,
but it also makes it difficult to determine how much scrutiny to apply to the contribution.

In a perfect world, AI assistance would produce equal or higher quality work than any human. That isn't the world
we live in today, and in most cases where human supervision or expertise is not in the loop, it's generating code
or changes that cannot be reasonably maintained or evolved.

### What we're looking for

When submitting AI-assisted contributions, please ensure they include:

- **Clear disclosure of AI use** - You are transparent about AI use and degree to which you're using it for the contribution
- **Human understanding** - You personally understand what the changes do
- **Clear rationale** - You can explain why the change is needed and how it fits within Model Context Protocol goals
- **Concrete evidence** - Include test cases, scenarios, or examples that demonstrate the improvement
- **Your own analysis** - Share your thoughts on the end-to-end experience

### What we'll close

We reserve the right to close submissions that appear to not follow the disclosure policy.

## License

By contributing, you agree that your code or specification contributions will be
licensed under the Apache License 2.0. Documentation contributions (excluding
specifications) are licensed under CC-BY 4.0. See the [LICENSE](LICENSE) file for
details.

## Security

Please review our [Security Policy](SECURITY.md) for reporting security issues.



================================================
FILE: eslint.config.mjs
================================================
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";
import eslintConfigPrettier from "eslint-config-prettier/flat";
import { defineConfig } from "eslint/config";

export default defineConfig([
  eslint.configs.recommended,
  ...tseslint.configs.recommended,
  eslintConfigPrettier,
]);



================================================
FILE: GOVERNANCE.md
================================================
General Project Policies

Model Context Protocol has been established as Model Context Protocol a Series of LF Projects, LLC. Policies applicable to Model Context Protocol and participants in Model Context Protocol, including guidelines on the usage of trademarks, are located at [https://www.lfprojects.org/policies/](https://www.lfprojects.org/policies/). Governance changes approved as per the provisions of this governance document must also be approved by LF Projects, LLC.

Model Context Protocol participants acknowledge that the copyright in all new contributions will be retained by the copyright holder as independent works of authorship and that no contributor or copyright holder will be required to assign copyrights to the project.

Except as described below, all code and specification contributions to the project must be made using the Apache License, Version 2.0 (available here: [https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)) (the “Project License”).

All outbound code and specifications will be made available under the Project License. The Core Maintainers may approve the use of an alternative open license or licenses for inbound or outbound contributions on an exception basis.

All documentation (excluding specifications) will be made available under Creative Commons Attribution 4.0 International license, available at: https://creativecommons.org/licenses/by/4.0.



================================================
FILE: LICENSE
================================================
The MCP project is undergoing a licensing transition from the MIT License to the Apache License, Version 2.0 ("Apache-2.0"). All new code and specification contributions to the project are licensed under Apache-2.0. Documentation contributions (excluding specifications) are licensed under CC-BY-4.0.

Contributions for which relicensing consent has been obtained are licensed under Apache-2.0. Contributions made by authors who originally licensed their work under the MIT License and who have not yet granted explicit permission to relicense remain licensed under the MIT License.

No rights beyond those granted by the applicable original license are conveyed for such contributions.

---

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to the Licensor for inclusion in the Work by the copyright
      owner or by an individual or Legal Entity authorized to submit on behalf
      of the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

---

MIT License

Copyright (c) 2024-2025 Model Context Protocol a Series of LF Projects, LLC.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

Creative Commons Attribution 4.0 International (CC-BY-4.0)

Documentation in this project (excluding specifications) is licensed under
CC-BY-4.0. See https://creativecommons.org/licenses/by/4.0/legalcode for
the full license text.



================================================
FILE: MAINTAINERS.md
================================================
# MCP Maintainers

This document lists current maintainers in the Model Context Protocol project.

**Last updated:** May 1, 2026

## Lead Maintainers

- [David Soria Parra](https://github.com/dsp-ant)
- [Den Delimarsky](https://github.com/localden)

## Core Maintainers

- [Caitie McCaffrey](https://github.com/CaitieM20)
- [Clare Liguori](https://github.com/clareliguori)
- [Kurtis Van Gent](https://github.com/kurtisvg)
- [Nick Aldridge](https://github.com/000-000-000-000-000)
- [Nick Cooper](https://github.com/nickcoai)
- [Paul Carleton](https://github.com/pcarleton)
- [Peter Alexander](https://github.com/pja-ant)

## Emeritus

- [Justin Spahr-Summers](https://github.com/jspahrsummers) (Co-Inventor, Lead Maintainer Emeritus)
- [Basil Hosmer](https://github.com/bhosmer-ant) (Core Maintainer Emeritus)
- [Che Liu](https://github.com/pwwpche) (Core Maintainer Emeritus)

## SDK Maintainers

### Java SDK

- [Christian Tzolov](https://github.com/tzolov)
- [Dariusz Jędrzejczyk](https://github.com/chemicL)
- [Daniel Garnier-Moiroux](https://github.com/Kehrlann)
- [Mark Pollack](https://github.com/markpollack)

### Ruby SDK

- [Topher Bullock](https://github.com/topherbullock)
- [Koichi Ito](https://github.com/koic)
- [Ateş Göral](https://github.com/atesgoral)
- [Jonathan Hefner](https://github.com/jonathanhefner)

### Swift SDK

- [Matt Zmuda](https://github.com/mattt)
- [Carl Peaslee](https://github.com/carlpeaslee)
- [Maksym Mova](https://github.com/movetz)
- [Stephen Tallent](https://github.com/stallent)

### Go SDK

- [Maciej Kisiel](https://github.com/maciej-kisiel)
- [Yaroslav Shevchuk](https://github.com/yarolegovich)
- [Guglielmo Colombo](https://github.com/guglielmo-san)
- [Jonathan Amsterdam](https://github.com/jba)

### C# SDK

- [Stephan Halter](https://github.com/halter73)
- [Mike Kistler](https://github.com/mikekistler)
- [Den Delimarsky](https://github.com/dend)
- [Eirik Tsarpalis](https://github.com/eiriktsarpalis)
- [Stephen Toub](https://github.com/stephentoub)

### Kotlin SDK

- [Leonid Stashevsky](https://github.com/e5l)
- [Sergey Ignatov](https://github.com/ignatov)
- [Konstantin Pavlov](https://github.com/kpavlov)
- [Pavel Gorgulov](https://github.com/devcrocod)
- [Briliantov Vadim](https://github.com/Ololoshechkin)
- [Sergei Dubov](https://github.com/sdubov)
- [Maria Tigina](https://github.com/tiginamaria)

### Python SDK

- [Inna Harper](https://github.com/ihrpr)
- [Jerome Swannack](https://github.com/jerome3o)
- [Marcelo Trylesinski](https://github.com/Kludex)
- [Max Isbey](https://github.com/maxisbey)
- [Felix Weinberger](https://github.com/felixweinberger)

### TypeScript SDK

- [Inna Harper](https://github.com/ihrpr)
- [Felix Weinberger](https://github.com/felixweinberger)
- [Olivier Chafik](https://github.com/ochafik)
- [Konstantin Konstantinov](https://github.com/KKonstantinov)
- 

[... Content truncated due to length ...]

</details>


## YouTube Video Transcripts

<details>
<summary>[00:00]</summary>

[00:00]

(A top-down shot of a person using a laptop on a white desk is shown. A stylized digital corridor overlay appears with text reading "TOOLS FOR EVERYTHING" and "MCP servers: one of the three most important innovations for evolving your engineering from AI coding to Agentic Coding" displayed with a dial.)

**Speaker**: As you know, MCP servers let you build tools for everything. MCP servers are one of the three most important innovations for evolving your engineering from AI coding to agentic coding. 

(A web browser window displays a mockup of an "Introducing Claude 4" announcement on Anthropic's website, alongside a DeepSeek-R1.1 release page.)

**Speaker**: With new models like Claude 4 and the brand-new DeepSeek-R1.1, we have more intelligence to build than ever before. But the models are no longer the limiting factor for your engineering output. 

[00:30]

(On-screen text overlaid on the digital graphic asks, "What's limiting us as engineers from creating more value FASTER than ever?")

**Speaker**: That forces us to ask: what's limiting us as engineers from creating more value faster than ever? It's our abilities to create capabilities for our agentic coding tools like Claude Code. 

(A terminal window with the header "AI CLAUDE CODE" appears. The speaker then switches to a web browser showing the Model Context Protocol GitHub repository.)

**Speaker**: That brings us full circle back to MCP servers. In this video, we're going to understand the most underutilized capability of MCP servers. Most engineers stop at tools.

*In this section, the speaker introduces MCP servers as a fundamental paradigm shift for moving from AI-assisted coding to fully agentic developer workflows.*

[01:00]

(On the MCP documentation page, the speaker highlights the "Core MCP Concepts": Resources, Tools, and Prompts.)

**Speaker**: But once you understand this one simple idea, you'll be able to craft rich MCP servers that dramatically increase your engineering velocity as well as your teams'. Resources, tools, and prompts. 

(An overlay chart shows the "MCP Primitive Tier List: Prompts > Tools > Resources".)

**Speaker**: In the tier list in reverse order of capability, we have resources, tools, and prompts. Most engineers skip resources; they go all-in on tools and completely miss out on the highest leverage primitive of MCP servers: prompts.

[01:30]

**Speaker**: Tool calling is just the beginning of your MCP server. Let me show you how to maximize the value of your MCP servers.

(A transitions screen displays cloud footage with the text "MCP MAXXING" in bold letters.)

**Speaker**: MCP Maxxing. 

(The speaker opens the Claude Code CLI. They type `/mcp` to show the connected servers, focusing on `quick-data`.)

**Speaker**: If we type `/mcp`, you can see I have six MCP servers available. We're going to be operating in the `quick-data` MCP server. `quick-data` gives your agent arbitrary data analysis capabilities on `.json` and `.csv` files.

*The speaker sets up the "quick-data" MCP server in the terminal and argues that relying solely on tool calling misses out on the immense leverage of prompts.*

[02:00]

(The speaker types `/model` inside Claude Code and selects "Sonnet 4" from the dropdown list.)

**Speaker**: We all know how tools work, but let's run a few to understand the `quick-data` MCP server and showcase how limited tool calls really are. If we type `/model`, we're going to run the Sonnet 4 fast workhorse model for this. So right away, we have a problem. I have no idea what I can do with this MCP server. I have to rely on some type of documentation. 

(The Cursor IDE opens, displaying a `README.md` file titled "Generic Data Analytics MCP Server".)

**Speaker**: Let's open up Cursor and break open the README. 

[02:30]

(The speaker scrolls through the README, highlighting a list of "Analytics Tools (32 total)" categorized under Data Loading, Core Analytics, and Advanced Analytics.)

**Speaker**: If we scroll down here, I have a completely documented set of all the tools, resources, and prompts available for this MCP server. But let's just start with a couple simple ones. I'll run this: `load_dataset`. And now we need to pass in a `.json` or a `.csv` file. I'll go back to Cursor. If I search for `ecommerce_orders`, you can see we have this simple JSON list. 

(The speaker uses a keyboard shortcut to copy the relative path of `ecommerce_orders.json` in Cursor and pastes it back into the terminal.)

**Speaker**: I'll copy the reference to this file with Command+Shift+R, then hop back to Claude Code, paste this in, and have it load.

[03:00]

(The Claude Code CLI prompts the user to confirm the execution of the `quick-data:load_dataset` tool. The user approves, returning a "status: loaded" confirmation.)

**Speaker**: All right, so as expected, we have this `load_dataset` MCP server tool. It has the file path and the dataset name, `ecommerce_orders`. This looks great, we'll go ahead and accept this, and you can see our JSON response. If we hit Control+R, you can see the entire thing: columns, rows, dataset name. Looks great. So let's go ahead and get a dataset breakdown. 

(The user executes the `get_dataset_info` tool in the terminal.)

**Speaker**: So I'll paste... this also accepts the dataset name, so we'll go ahead, copy and paste this back in. And now we're just going to get some basic information about this dataset. We'll of course accept this tool call, and you can see we have the shape and key information about this dataset.

[03:30]

**Speaker**: So so far, this looks great. Let's run a couple more tools, and then we're going to uplevel everything we're doing by looking at the most powerful capability you can add to your MCP server. Let's run `suggest_analysis`. 

(The speaker types `suggest_analysis` followed by `ecom...` which auto-completes to `ecommerce_orders`.)

**Speaker**: Paste, and then I'll just say `ecom...` This is going to be auto-completed for us based on the current context. There it is, `suggest_analysis`. Let's see what we get. So we have a couple of ideas given to us based on that tool call.

*To demonstrate the standard tool-calling pattern, the speaker loads a JSON dataset of e-commerce orders and requests analytical suggestions.*

[04:00]

(The terminal displays suggested analyses: "Segmentation (high priority)", "Distribution analysis", and "Outlier detection", with suggested shell commands.)

**Speaker**: Run command number one. Fire this off. We're now going to get a segment breakdown by this product category column. So check this out. We have product category segmentation. We can see that electronics are producing a lot of value inside of this `ecommerce_orders.json` file. 

(The terminal lists sales data segmented by category: Electronics has the highest total order value at $2,649.96.)

**Speaker**: So looking at this data from a business strategy perspective, we could, if we wanted to, cut down on sports and home garden product categories and go all-in on electronics based on this insight. Okay, so there's one more cool tool I want to share with you here. If we scroll down to the bottom, we can execute arbitrary code.

[04:30]

(The Cursor README shows an "AI-Powered Assistance" tool named `execute_custom_analytics_code`.)

**Speaker**: We can have Claude Code, running on Claude 4 Sonnet, execute arbitrary code for us. So again, we can just come back in here, paste, we can say `ecom...` and let's find out... if we look at the dataset here, we have this `region` column, and we also have `order_value`. So let's find out the top order value by region. Find top 3 region order value. Yep, let's go ahead and fire that off. 

(The model automatically generates Python code to group the dataset by region and calculate total order values.)

**Speaker**: There we go. So you can see here, we have custom code getting written based on our prompt.

[05:00]

(The speaker approves the tool run. The code executes and returns a table showing the East Coast with the highest total sales, followed by the West Coast and Midwest.)

**Speaker**: We'll hit yes, and there is our executed code response, as you can see here. Our top three regions by order value: we have East Coast, West Coast, and of course, Midwest in last place. Pretty accurate training dataset, right? If we want to reuse that same MCP tool call, we can hit up, and then I'll say "then create a pie chart labeled by region value and percent." It's going to create a pie chart for us, let's go ahead and run this...

[05:30]

(A local browser window opens at localhost, displaying an interactive Plotly pie chart illustrating the order value distribution across regions.)

**Speaker**: ...and bam, check this out. You can see we have East Coast, we have West Coast, Midwest, and then the South. We have a great breakdown here, and this was all just quickly created and managed with our MCP server for quick data analytics against JSON and CSV. 

So tools are great, we all know about their capabilities, we can build out tools for anything, and tools for everything. But tools only scratch the surface of what you can do with your MCP server. To unlock the full capabilities of what you can do, we need to build MCP server prompts.

*The speaker runs arbitrary Python code through the agent to analyze regional sales data and generate an interactive Plotly pie chart, highlighting that tools are powerful but limited on their own.*

[06:00]

(A cloud scene appears briefly with yellow text: "MCP SERVER PROMPTS" before returning to the terminal.)

**Speaker**: So, in order to showcase the capabilities here, we're going to reset this Claude Code instance and really start from scratch. So let's open up Claude again. We'll run the same setup, so you can see here `/mcp`. `/model`, same deal, Sonnet 4. 

(The Cursor project files directory is opened, showing folders for `specs`, `trees`, `data`, and `src/mcp_server`.)

**Speaker**: So now, instead of looking through the documentation, right, we had this README that thankfully detailed all of our tools, resources, and prompts, right? There's the codebase structure, we'll take a look at that in a second. Instead of doing any of this, instead of relying on codebase architecture, codebase structure, we can just use MCP server prompts to guide the entire discovery and use of the `quick-data` MCP server. Let me show you exactly what I mean.

[06:30]

(The speaker types `/quick-data` in the Claude Code terminal, displaying a list of available prompts built directly into the MCP server.)

**Speaker**: To find all the prompts associated with this MCP server inside of Claude Code, we can type `/` and then `quick-data`. So this is the name of the MCP server, and here you can see a ton of auto-complete suggestions with prompts. So these are prompts built out in the MCP server. Now we're going to run something really cool, something very useful that I highly recommend you set up inside of all your MCP servers. We're going to list available MCP server capabilities, including prompts, tools, and resources. So this is a prompt that's going to give us a clear breakdown of what we can do with this tool.

[07:00]

(The output of `/quick-data:list_mcp_assets_prompt` details the components of the server and maps out a step-by-step "Quick Start Flow".)

**Speaker**: Okay, check this out. So Claude Code, our agentic coding tool, has now consumed everything that we can do with this tool. It's now loaded fresh in the context window, and we have a quick start flow to get started. So now if we want to, we can just ask Claude Code what exactly these key components are. Okay, so I'm just going to say, "what prompts and tools do we have available? List as bullets."

[07:30]

(Claude Code prints out a comprehensive bulleted list of the server's data exploration, workflow, and visualization capabilities directly in the CLI.)

**Speaker**: Alright, so check this out. So now we can just, you know, query our agent. Right? Here are the prompts, here are the tools, this is everything that we saw before. Let's go ahead and continue firing off these prompts to really understand what they can do for us. Alright, so if we type `/` and then `find`, you can see we have another prompt: `find_datasources_prompt`. This is going to discover available data files inside the current directory and present them as load options. 

Now, see how much more helpful these prompts are than just having tools hidden somewhere? I'm going to hit tab, you can see here we have an argument, the directory path. I'll just hit `.` for that, and fire that off. So this is going to automatically discover all available `.json` and `.csv` files for our `quick-data` MCP server.

[08:00]

(The system automatically detects `.mcp.json`, `ecommerce_orders.json`, `employee_survey.csv`, and `product_performance.csv` and presents ready-to-run copy/paste commands to load them.)

**Speaker**: So we had a prompt, also known as an agentic workflow, do this work for us automatically. You can see we also have—take note of this, this is really important—"ready to load with `load_dataset` commands." So with the previous prompt and this prompt, you can see, every prompt we're running, we're getting a suggestion or a forward direction or a next step for what we can do with this MCP server. So what I'm going to do here is just type `load ecom`.

[08:30]

(The model automatically structures and suggests running the `load_dataset` tool on the e-commerce orders path.)

**Speaker**: So, I have a really tight, information-dense keyword prompt, literally just two words, with the current context that we have set up thanks to our prompts, and thanks to Claude Code running on Claude 4 Sonnet, I can be nearly 100% sure that this is going to run the right tool with the right information. Okay? And so I'll kick this off, and notice how I just, you know, ran through the big three of AI coding: context, model, prompt.

*By using the built-in asset-listing and source-finding prompts, the user shows how prompts structure the user experience by dynamically feeding the agent the exact files, commands, and workflow context needed to interact with the environment.*

[09:00]

(A Venn diagram illustrating the intersection of "CONTEXT", "MODEL", and "PROMPT" appears on screen.)

**Speaker**: These never go away, that's why they're a principle of AI coding. They're always there whether you realize it or not. The more you can look and think from your agent's perspective with the current available context, model, and prompt, the more you'll be able to hand off tons and tons of engineering work, which in the end results in your engineering velocity increase. 

So check this out, we have the file path here, using the full absolute path, looks great, and then we have the dataset name. Okay? With just typing slash, with just working through a few pre-existing prompts, we're moving a lot faster than if we were looking through, you know, the documentation going back and forth and back and forth.

[09:30]

(Thumbnails of previous videos appear in a grid on the screen, illustrating different AI coding and prompt-engineering strategies.)

**Speaker**: And that is a really important thing to call out here, right? We haven't left the terminal, we haven't left Claude Code. We're focused, we're moving quickly, and we're operating inside of this MCP server with minimal information. Okay, so we have that dataset loaded. If we scroll back up, you'll remember here at the top that we were given a concrete workflow. You can see `find_dataset` to discover data files, and then we can run `load_dataset`, and then explore data. So let's go ahead and run that, I'm going to type `/` and then `first`.

[10:00]

(The speaker executes the `dataset_first_look_prompt` prompt on `ecommerce_orders` in the CLI.)

**Speaker**: This is our `first_look` MCP prompt, I'll hit tab, and you can see there the arguments are dataset name. I'll go ahead and just type `ecom...` and we should get auto-completion there. There you go. So this prompt—and we're going to take a look at the individual prompts in a second—is kicking off one or more tools. Okay? So we'll go ahead, fire that off, and based on that prompt, right, and based on the information returned by this tool, we're getting a nice breakdown of a sample of this dataset: size, columns, sample data, looks great.

[10:30]

(The user expands the returned JSON sample to inspect the data columns.)

**Speaker**: You can see there we actually did get a sample. If we hit Control+R, it broke down, you know, pieces of our data, so that looks great. Thanks to the existing context window that all of these prompts have been giving our agent, we can just type something like this: "how can we further explore this data?"

[11:00]

(Claude Code lists the available analytical strategies in the current context, categorized by Direct Analysis Tools, Business Insights, Visualization, and Custom Analysis.)

**Speaker**: Okay, so check this out. From the existing context window, we have, you know, tons of ideas of how we can keep pushing this. And this is really useful for when you're operating outside of your MCP server. Obviously, if you're building your MCP server, you have access to the actual code, and you can just kind of, you know, have your agent run through this. 

But when we're operating in this, when we're handing off our MCP server to our team, to our engineering team, and when we're exposing our MCP servers to the public, we want it easy to use, we want it to be quickly consumable, and we want these guided workflows.

[11:30]

**Speaker**: Okay, prompts are really important because they can return entire sets of information to your agent, and they can provide next steps. You can keep pushing yourself, you can keep pushing engineers on your team, and pushing the agent in the right direction for your domain-specific problem set. Let's go ahead, let's run another prompt. So we can do `/` and then `quick-data` to see all of our prompts. Let's go ahead and run the `correlation_investigation_prompt`.

[12:00]

(The speaker navigates to the Cursor project tree and opens the file `correlation_investigation_prompt.py`.)

**Speaker**: So this is going to find correlations inside of our dataset. We'll of course type `ecom...` and before I run this, let me show you exactly what these prompts look like inside of the MCP server. So we'll open up Cursor, and we're just going to search for that prompt. Notice how I just have a single method inside of this file? And since we're here, let's talk about codebase architecture. This is important.

*Prompts allow the engineer to create modular, self-guiding, and highly consumable interfaces that assist team members or public consumers without requiring them to read code or documentation.*

[12:30]

(The Cursor file panel displays the modular codebase layout under `arch-modular-mcp/src/mcp_server`, showing isolated folders for `prompts`, `resources`, and `tools`.)

**Speaker**: So, I have the codebase embedded in its own directory here. And on top, you can see the three essential directories for agentic coding. And you can see our trees directory for multi-agent parallel AI coding. Check out the previous video to see how we parallelize Claude Code into multiple trees to get work done at the same time. But if we click into `arch-modular-mcp` and we take a look at the architecture here, you can see we have our data there, then we have `src/mcp_server`, and we have the primitives of the MCP server: tools, resources, and prompts.

[13:00]

(The speaker views the prompt files list, showing single-responsibility files for each custom prompt workflow.)

**Speaker**: If we open up `prompts`, we can see our `correlation_investigation_prompt` inside of this single file. These are all single-function Python files to keep everything nice and isolated and easily testable. So, if we hop up to this file, we can see something really cool: we're passing in the dataset name, and then we're just running arbitrary code, which is effectively our agentic workflow. So you can do anything you want here, the most important thing is to gather some type of prompt response, and then return that back to your agent.

[13:30]

(The Python code returns a highly detailed F-string prompt containing analytical options and correlation formulas.)

**Speaker**: Right, this is what's going to get passed right back into the agent. So you can see we have lots of detail on the correlation investigation, a couple of branches of logic here, a loop, and you can see we're loading that schema out of our existing dataset. So let's go ahead, let's run this, this is going to run a really great analysis on our dataset, okay? So we're going to close that. Let's run this.

[14:00]

(Back in Claude Code, the user runs the `correlation_investigation_prompt` prompt on `ecommerce_orders`, triggering an immediate tool call to `find_correlations`.)

**Speaker**: So this prompt is kicking off a tool call, and this is super important. Inside of your prompts, you can kick off one or more tool calls. You can see here how the prompt allows you to compose sequences of tool calls very, very quickly using a custom slash command here. Okay? So quick commands to start, you can see that we're picking this up automatically, this is getting returned into the context window, and now Claude Code running on Claude 4 wants to kick this off for us based on this prompt.

[14:30]

(The tool call to `find_correlations` outputs an error: "Need at least 2 numerical columns for correlation analysis". The model then checks the column data types to identify alternatives.)

**Speaker**: Okay, so of course we'll hit yes. And you can see there we're getting some concrete feedback, okay? We need at least two numerical columns for correlation analysis. Okay? So we can go ahead, kick this off. This is going to re-expose information back into our agent. So `ecommerce_orders` cannot run this, okay? So our tools are giving us feedback, all guided by our prompt. Well, let's go ahead and load some more data, right? We can very quickly, thanks to our slash command, just run `/find`.

[15:00]

(The speaker runs the `/quick-data:find_datasources_prompt` prompt to scan for other compatible files.)

**Speaker**: And let's go ahead and find those other data sources that we have. I'll specify the directory here, `.`, and this is going to reload all of our data sources. So you can see here, we can load these two additional datasets. Let's go ahead and load these. So I'm just going to say `load all`.

[15:30]

(The system automatically structures and queues loading commands for both `employee_survey.csv` and `product_performance.csv`.)

**Speaker**: So now we're going to get those two prompts, there's our `employee_survey` dataset, and here is our `product_performance` dataset. We'll hit yes, yes. "Now you have multiple numerical columns across datasets for correlation analysis!" Okay, so I'll just say "run analysis on employee, product."

[16:00]

(The CLI executes multiple consecutive analysis tools based on the combined loaded context.)

**Speaker**: Okay, so there it is, there's that `find_correlations` lookup, and you can see here it's queued up several calls, several tool calls, that we can now kick off. So this is one way to, you know, activate this workflow. That's great. I'm going to hit Escape here, and I'm going to reuse the slash command that we were going for. So I'm going to type `/correlation` investigation prompt, looks great, and then I'll pass in, let's use our `employee_survey`.

*The speaker details the modular codebase architecture where each prompt acts as an independent Python file that orchestrates multiple custom tool-calling sequences based on system constraints.*

[16:30]

(The speaker executes the correlation prompt on `employee_survey`, finding that it contains two numerical variables: `satisfaction_score` and `tenure_years`.)

**Speaker**: So, I'll paste that in, and let's run the investigation prompt here. This should kick off a similar workflow. There it is. So, this prompt is exposing the potential columns that we can correlate. So, I'm going to go ahead, I want to kick off this first flow that was revealed by this prompt. Super simple, it's a slash command that's exposed by our MCP server prompts that we just pass in one variable to work with, okay?

[17:00]

(The speaker commands the agent to execute "run option 1", running the `find_correlations` tool on the target columns.)

**Speaker**: So, go ahead and type "run option 1". Okay? So, this should pop up `find_correlations`. There it is, fire off our correlations, and let's see what we get here. So, check this out, strong correlation found: we have satisfaction score correlated with tenure years. All right, so if we open up this dataset...

[17:30]

(The speaker reviews `employee_survey.csv` inside Cursor, validating the raw variables against the calculated correlation coefficient of `0.864`.)

**Speaker**: ...just to take a look at this, you can see several columns: tenure year, satisfaction score, simple CSV file, and this prompt, and the tools called by the prompt, found this strong correlation. Okay, so there's a strong positive correlation between satisfaction score and tenure year. That means employees with higher satisfaction scores tend to have longer tenure.

[18:00]

**Speaker**: And so, this reveals, you know, not to get too specific into the weeds of this MCP server, but this is important because it's going to immediately reveal to us that satisfaction and retention are closely linked, satisfied employees stay longer. Not a mind-blowing (laughter) revelation, but this could be anything inside of your dataset, okay? 

I've just... I've just put together a small, concise example that we can discuss to showcase the power of these MCP server prompts. Okay. "Do you want to visualize this with option 2?" I'll say "go ahead."

[18:30]

(The system automatically calls the `create_chart` tool to plot satisfaction score against tenure years, generating a scatter plot HTML file.)

**Speaker**: And we'll just continue walking through `create_chart`. Let's go ahead and fire that off. We now have an additional chart set up here, we can open this up. So, we can copy this file path here. If we go into HTML preview mode, we can see this chart generated. If you average this out over the satisfaction score, you can see we have a pretty strong correlation here between tenure and satisfaction score.

[19:00]

(The HTML preview reveals a clear upward trend on the scatter plot showing that higher satisfaction correlates directly with longer company tenure.)

**Speaker**: So, very powerful stuff. So, what does this all mean, right? Why are prompts inside of your MCP server so important? Right away, by using this MCP server, we were able to move a lot faster. If we close Claude here, reopen it, and we type `/assets`...

[19:30]

(The terminal is cleared and restarted. Typing `/assets` triggers `/quick-data:list_mcp_assets_prompt`, rapidly priming the agent with the entire server's capabilities and datasets.)

**Speaker**: ...we can get our agent back up and running with this MCP server very quickly, okay? So prompts let you quickly set up your agent with everything they need to know to operate your MCP server. So this is just one simple way you can use MCP server prompts. Check this out. We can look at this exact prompt, right? This is the `list_mcp_assets_prompt`. So check this out, look how simple this is. This is quite literally just returning...

*Through the execution of correlation analysis and interactive visualization, the speaker highlights that prompts can effortlessly guide complex analytical steps on any dataset without requiring user context tracking.*

[20:00]

(The speaker shows the code behind `list_mcp_assets_prompt.py`, illustrating how a static string serves as a comprehensive capabilities ledger.)

**Speaker**: ...essential information about this MCP server in a custom way to our agent. This prompt primes both your memory and your agent's memory with everything it needs to know about your MCP server. Every MCP server I build out now has some type of prompt just like this. So now everything is exposed, we can quickly see and operate on things in a much faster way. We can always type `/quick-` and start understanding our datasets, right? So we then ran our `find`...

[20:30]

(The speaker reviews the `find_datasources_prompt.py` codebase on Cursor.)

**Speaker**: ...and we passed in `.`. Prompts allow us to prime our agent in powerful ways and run arbitrary Claude Code tools. So inside of the `find_datasources_prompt`—again, we can just search this—I've isolated everything into its own file. This is another great pattern I highly recommend you follow. We have our `find_datasources_prompt`, which is running arbitrary code. Principled AI coding members know this as an ADW, an AI developer workflow.

[21:00]

**Speaker**: That's all these prompts are. They're end-to-end chains of prompts and code coupled together that end up in a simple string return value. So after we scan all the directories, we do something really powerful: in multiple use cases and multiple scenarios, we offer the agent suggestions. This is really, really powerful. Our agent is ready for the next step, right? It wants these `load_dataset` commands. So this time around, you know, this is an agent, it's powerful, it's got the new Claude 4 model, we can just say "load all datasets."

[21:30]

(The speaker runs "load all datasets" in Claude Code, loading all discovered CSV files in seconds.)

**Speaker**: Okay, so there's three prompts instead of one, right? We can move a lot faster thanks to the prompt. Okay? So we can load bam, bam, bam. Now we have all three datasets loaded. Fantastic, and now we can run, you know, our dataset first look if we wanted to. We can continue down that line that we were running before: `/dataset first look`. You know, this is just, you know, two or three of many prompts that we have here. There's really no limit to what you can do with your prompts inside of your MCP server.

[22:00]

(The speaker displays the MCP feature matrix. Under "Claude Code", "Prompts" and "Tools" are checked green, whereas "Resources" is crossed red.)

**Speaker**: So we have prompts, resources, and tools. Claude Code does not support resources. If we open up the example clients and search Claude Code, you can see here that, you know, Claude Code does not have resource support, but it has the two that really matter: prompts and tools. You can also substitute your resources for just specific tool calls. I've done that inside this codebase, you can check that out if you're interested. But recentering on the key idea here: why do we create prompts? Because prompts allow us to create agentic workflows, they allow us to compose our tools.

[22:30]

(The code for `load_dataset_tool.py` is shown in Cursor as a singular, simple file-loading instruction.)

**Speaker**: Tools are individual actions. Here's our `load_dataset` tool, and you can see it just does one thing, right? It takes one action: it loads the dataset into memory. Tools are individual actions, prompts are recipes for repeat solutions. This is the big difference. Your prompts have three massive advantages that your tools don't have. You can, with Claude Code, reference all of your prompts in a clean, detailed way very quickly, okay? So no more guessing.

*While tools act as single-responsibility operational atoms, prompts serve as robust, multi-step orchestration recipes that sequence these atoms into powerful, repeatable developer workflows.*

[23:00]

**Speaker**: You can quickly get up and running with whatever MCP server you have. So second, your prompts can compose tools in your MCP server together. Okay, this is super, super powerful. You saw multiple times here our prompt was then kicking off individual tools that exist underneath the prompts, right? That's why we have this tier list order of capability: prompts greater than tools, greater than resources.

[23:30]

(The terminal displays the automated execution of file loading based on prompt guides.)

**Speaker**: And lastly, a super, super underutilized element of prompts is that you can guide the experience. At the end, our agent is saying, "use dataset first look to explore any dataset." So our agent, through the prompt that was run here, triggered not just a sequence of tools, but also a guide and a direction for you, the engineer operating the tool, and more importantly every single day, it's giving our agent the next steps. Okay? So `load_dataset`, `load_dataset`, `load_dataset`, and then we can just, you know, very quickly, very calmly, say things like this: "load all datasets" and then continue down the line of data exploration running whatever other tools or prompts our MCP server exposes to us.

[24:00]

**Speaker**: With prompts, you can build out high-quality MCP servers that do more than just call tools. Tool calling is just the beginning. Tools are the primitives of MCP servers, not the end state. You want to end up with prompts, right? Prompts represent end-to-end developer workflows that are truly agentic workflows or, as I like to call them, AI developer workflows. Right? They are quite literally doing developer work that you would do, but it's powered, of course, by gen AI.

[24:30]

(The top-down shot of the laptop on the desk returns as the speaker concludes the video.)

**Speaker**: You really want to be thinking about MCP servers as a way to solve a domain-specific problem in an automated fashion with repeat solutions embedded inside of the prompts. The prompt is what the tools scale into. 

This codebase is linked in the description to give you a concrete example of how you can use prompts inside of your MCP servers. Come in here, play with it, CD into this folder name—I'll probably change this by the time you see it—and you'll be able to quickly boot up Claude Code with this `.mcp.json` file here. If you made it to the end, like, comment to let the YouTube algorithm know you want more hands-on engineering information like this. You know where to find me every Monday. Stay focused, and keep building.

*The speaker concludes by urging developers to design MCP servers with prompts in mind, packaging workflows as complete solutions rather than scattered, standalone tools.*

</details>


## Additional Sources Scraped

<details>
<summary>connect-claude-code-to-tools-via-mcp-claude-code-docs</summary>

{'type': 'text', 'text': 'Claude Code can connect to hundreds of external tools and data sources through the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction), an open source standard for AI-tool integrations. MCP servers give Claude Code access to your tools, databases, and APIs.Connect a server when you find yourself copying data into chat from another tool, like an issue tracker or a monitoring dashboard. Once connected, Claude can read and act on that system directly instead of working from what you paste.\n\n## What you can do with MCP\n\nWith MCP servers connected, you can ask Claude Code to:\n\n- **Implement features from issue trackers**: “Add the feature described in JIRA issue ENG-4521 and create a PR on GitHub.”\n- **Analyze monitoring data**: “Check Sentry and Statsig to check the usage of the feature described in ENG-4521.”\n- **Query databases**: “Find emails of 10 random users who used feature ENG-4521, based on our PostgreSQL database.”\n- **Integrate designs**: “Update our standard email template based on the new Figma designs that were posted in Slack”\n- **Automate workflows**: “Create Gmail drafts inviting these 10 users to a feedback session about the new feature.”\n- **React to external events**: An MCP server can also act as a [channel](https://code.claude.com/docs/en/channels) that pushes messages into your session, so Claude reacts to Telegram messages, Discord chats, or webhook events while you’re away.\n\n## Find and build MCP servers\n\nBrowse reviewed connectors in the [Anthropic Directory](https://claude.ai/directory). Directory connectors use the same MCP infrastructure as Claude Code, so you can add any remote server listed there with `claude mcp add`.\n\nVerify you trust each server before connecting it. Servers that fetch external content can expose you to [prompt injection risk](https://code.claude.com/docs/en/security#protect-against-prompt-injection).\n\nTo build your own server, see the [MCP server guide](https://modelcontextprotocol.io/docs/develop/build-server) for protocol fundamentals and the [Claude connector building docs](https://claude.com/docs/connectors/building) for authentication, testing, and Directory submission.You can also have Claude scaffold a server for you with the official [`mcp-server-dev` plugin](https://github.com/anthropics/claude-plugins-official/tree/main/plugins/mcp-server-dev).\n\nInstall the plugin\n\nIn a Claude Code session, run:\n\n```\n/plugin install mcp-server-dev@claude-plugins-official\n```\n\nIf Claude Code reports that the marketplace is not found, run `/plugin marketplace add anthropics/claude-plugins-official` first, then retry the install. Once installed, run `/reload-plugins` to activate it in the current session.\n\nRun the build skill\n\n```\n/mcp-server-dev:build-mcp-server\n```\n\nClaude asks about your use case and scaffolds a remote HTTP or local stdio server.\n\n## Installing MCP servers\n\nMCP servers can be configured in several ways depending on your needs:\n\n### Option 1: Add a remote HTTP server\n\nHTTP servers are the recommended option for connecting to remote MCP servers. This is the most widely supported transport for cloud-based services.\n\n```\n# Basic syntax\nclaude mcp add --transport http <name> <url>\n\n# Real example: Connect to Notion\nclaude mcp add --transport http notion https://mcp.notion.com/mcp\n\n# Example with Bearer token\nclaude mcp add --transport http secure-api https://api.example.com/mcp \\\n  --header "Authorization: Bearer your-token"\n```\n\nWhen configuring MCP servers via JSON in `.mcp.json`, `~/.claude.json`, or `claude mcp add-json`, the `type` field accepts `streamable-http` as an alias for `http`. The MCP specification uses the name `streamable-http` for this transport, so configurations copied from server documentation work without modification.\n\n### Option 2: Add a remote SSE server\n\nThe SSE (Server-Sent Events) transport is deprecated. Use HTTP servers instead, where available.\n\n```\n# Basic syntax\nclaude mcp add --transport sse <name> <url>\n\n# Real example: Connect to Asana\nclaude mcp add --transport sse asana https://mcp.asana.com/sse\n\n# Example with authentication header\nclaude mcp add --transport sse private-api https://api.company.com/sse \\\n  --header "X-API-Key: your-key-here"\n```\n\n### Option 3: Add a local stdio server\n\nStdio servers run as local processes on your machine. They’re ideal for tools that need direct system access or custom scripts.Claude Code sets `CLAUDE_PROJECT_DIR` in the spawned server’s environment to the project root, so your server can resolve project-relative paths without depending on the working directory. This is the same directory hooks receive in their `CLAUDE_PROJECT_DIR` variable. Read it from inside your server process, for example `process.env.CLAUDE_PROJECT_DIR` in Node or `os.environ["CLAUDE_PROJECT_DIR"]` in Python. Your server can also call the MCP `roots/list` request, which returns the directory Claude Code was launched from.This variable is set in the server’s environment, not in Claude Code’s own environment, so referencing it via `${VAR}` expansion in a project- or user-scoped `.mcp.json``command` or `args` requires a default such as `${CLAUDE_PROJECT_DIR:-.}`. Plugin-provided MCP configurations substitute `${CLAUDE_PROJECT_DIR}` directly and don’t need the default.\n\n```\n# Basic syntax\nclaude mcp add [options] <name> -- <command> [args...]\n\n# Real example: Add Airtable server\nclaude mcp add --env AIRTABLE_API_KEY=YOUR_KEY --transport stdio airtable \\\n  -- npx -y airtable-mcp-server\n```\n\n**Important: Separate server arguments with `--`**For stdio servers, the `--` (double dash) separates Claude’s own options, such as `--transport`, `--env`, and `--scope`, from the command and arguments that run the server. Everything after `--` is passed to the server untouched.For example:\n\n- `claude mcp add --transport stdio myserver -- npx server` → runs `npx server`\n- `claude mcp add --env KEY=value --transport stdio myserver -- python server.py --port 8080` → runs `python server.py --port 8080` with `KEY=value` in environment\n\nWithout `--`, Claude Code would try to parse the server’s flags, like `--port` above, as its own options.`--env` accepts multiple `KEY=value` pairs. If the server name comes directly after `--env`, the CLI reads the name as another pair and rejects it, so place at least one other option between `--env` and the server name, as in the examples above.\n\n### Option 4: Add a remote WebSocket server\n\nWebSocket servers hold a persistent bidirectional connection, which suits remote MCP servers that push events to Claude unprompted. Use HTTP instead when your server only responds to requests, since HTTP supports OAuth and the `claude mcp add --transport` flag, while WebSocket supports neither.Configure WebSocket servers in `.mcp.json` or with `claude mcp add-json`:\n\n```\nclaude mcp add-json events-server \\\n  \'{"type":"ws","url":"wss://mcp.example.com/socket","headers":{"Authorization":"Bearer YOUR_TOKEN"}}\'\n```\n\nThe `type: "ws"` entry accepts the same `url`, `headers`, `headersHelper`, `timeout`, and `alwaysLoad` fields as `http`. Authentication is header-only, so pass a static token in `headers` or generate one at connect time with [`headersHelper`](https://code.claude.com/docs/en/mcp#use-dynamic-headers-for-custom-authentication). The `claude mcp add --transport` flag does not accept `ws`.\n\n### Managing your servers\n\nOnce configured, you can manage your MCP servers with these commands:\n\n```\n# List all configured servers\nclaude mcp list\n\n# Get details for a specific server\nclaude mcp get github\n\n# Remove a server\nclaude mcp remove github\n\n# (within Claude Code) Check server status\n/mcp\n```\n\nProject-scoped servers from `.mcp.json` that are awaiting your approval appear in `claude mcp list` as `⏸ Pending approval`. Run `claude` interactively to review and approve them. `claude mcp get <name>` shows pending servers as `⏸ Pending approval` and rejected servers as `✗ Rejected`.The `/mcp` panel shows the tool count next to each connected server and flags servers that advertise the tools capability but expose no tools.If your request needs tools from a server that is still connecting in the background, Claude waits for that server before continuing. With [tool search](https://code.claude.com/docs/en/mcp#scale-with-mcp-tool-search) enabled, which is the default, the wait happens inside the `ToolSearch` call. In configurations without tool search, such as Vertex AI, a custom `ANTHROPIC_BASE_URL`, or `ENABLE_TOOL_SEARCH=false`, Claude uses the `WaitForMcpServers` tool instead.The server name `workspace` is reserved for internal use. If your configuration defines a server with that name, Claude Code skips it at load time and shows a warning asking you to rename it.\n\n### Dynamic tool updates\n\nClaude Code supports MCP `list_changed` notifications, allowing MCP servers to dynamically update their available tools, prompts, and resources without requiring you to disconnect and reconnect. When an MCP server sends a `list_changed` notification, Claude Code automatically refreshes the available capabilities from that server.\n\n### Automatic reconnection\n\nIf an HTTP or SSE server disconnects mid-session, Claude Code automatically reconnects with exponential backoff: up to five attempts, starting at a one-second delay and doubling each time. The server appears as pending in `/mcp` while reconnection is in progress. After five failed attempts the server is marked as failed and you can retry manually from `/mcp`. Stdio servers are local processes and are not reconnected automatically.The same backoff applies when an HTTP or SSE server fails its initial connection at startup. As of v2.1.121, Claude Code retries the initial connection up to three times on transient errors such as a 5xx response, a connection refused, or a timeout, then marks the server as failed if it still cannot connect. Authentication and not-found errors are not retried because they require a configuration change to resolve.\n\n### Push messages with channels\n\nAn MCP server can also push messages directly into your session so Claude can react to external events like CI results, monitoring alerts, or chat messages. To enable this, your server declares the `claude/channel` capability and you opt it in with the `--channels` flag at startup. See [Channels](https://code.claude.com/docs/en/channels) to use an officially supported channel, or [Channels reference](https://code.claude.com/docs/en/channels-reference) to build your own.\n\nTips:\n\n- Use the `--scope` flag to specify where the configuration is stored:\n\n  - `local` (default): Available only to you in the current project (was called `project` in older versions)\n  - `project`: Shared with everyone in the project via `.mcp.json` file\n  - `user`: Available to you across all projects (was called `global` in older versions)\n- Set environment variables with `--env` flags (for example, `--env KEY=value`)\n- Configure MCP server startup timeout using the MCP\\_TIMEOUT environment variable (for example, `MCP_TIMEOUT=10000 claude` sets a 10-second timeout)\n- Set a per-server tool execution timeout by adding a `timeout` field in milliseconds to that server’s `.mcp.json` entry, for example `"timeout": 600000` for ten minutes. This overrides the `MCP_TOOL_TIMEOUT` environment variable for that server only\n- Claude Code will display a warning when MCP tool output exceeds 10,000 tokens. To increase this limit, set the `MAX_MCP_OUTPUT_TOKENS` environment variable (for example, `MAX_MCP_OUTPUT_TOKENS=50000`)\n- Use `/mcp` to authenticate with remote servers that require OAuth 2.0 authentication\n\nThe per-server `timeout` is a hard wall-clock limit per tool call, and progress notifications from the server do not extend it. Values below 1000 are ignored and fall through to `MCP_TOOL_TIMEOUT`, or to its default of about 28 hours when that variable is unset. Before v2.1.162, values below 1000 were floored to one second instead. For HTTP and SSE servers, the per-request fetch first-byte budget has a 60-second minimum.\n\n### Plugin-provided MCP servers\n\n[Plugins](https://code.claude.com/docs/en/plugins) can bundle MCP servers, automatically providing tools and integrations when the plugin is enabled. Plugin MCP servers work identically to user-configured servers.**How plugin MCP servers work**:\n\n- Plugins define MCP servers in `.mcp.json` at the plugin root or inline in `plugin.json`\n- When a plugin is enabled, its MCP servers start automatically\n- Plugin MCP tools appear alongside manually configured MCP tools\n- Plugin servers are managed through plugin installation (not `/mcp` commands)\n\n**Example plugin MCP configuration**:In `.mcp.json` at plugin root:\n\n```\n{\n  "mcpServers": {\n    "database-tools": {\n      "command": "${CLAUDE_PLUGIN_ROOT}/servers/db-server",\n      "args": ["--config", "${CLAUDE_PLUGIN_ROOT}/config.json"],\n      "env": {\n        "DB_URL": "${DB_URL}"\n      }\n    }\n  }\n}\n```\n\nOr inline in `plugin.json`:\n\n```\n{\n  "name": "my-plugin",\n  "mcpServers": {\n    "plugin-api": {\n      "command": "${CLAUDE_PLUGIN_ROOT}/servers/api-server",\n      "args": ["--port", "8080"]\n    }\n  }\n}\n```\n\n**Plugin MCP features**:\n\n- **Automatic lifecycle**: At session startup, servers for enabled plugins connect automatically. If you enable or disable a plugin during a session, run `/reload-plugins` to connect or disconnect its MCP servers\n- **Environment variables**: use `${CLAUDE_PLUGIN_ROOT}` for bundled plugin files, `${CLAUDE_PLUGIN_DATA}` for [persistent state](https://code.claude.com/docs/en/plugins-reference#persistent-data-directory) that survives plugin updates, and `${CLAUDE_PROJECT_DIR}` for the stable project root\n- **User environment access**: Access to same environment variables as manually configured servers\n- **Multiple transport types**: Support stdio, SSE, HTTP, and WebSocket transports (transport support may vary by server)\n\n**Viewing plugin MCP servers**:\n\n```\n# Within Claude Code, see all MCP servers including plugin ones\n/mcp\n```\n\nPlugin servers appear in the list with indicators showing they come from plugins.**Benefits of plugin MCP servers**:\n\n- **Bundled distribution**: Tools and servers packaged together\n- **Automatic setup**: No manual MCP configuration needed\n- **Team consistency**: Everyone gets the same tools when plugin is installed\n\nSee the [plugin components reference](https://code.claude.com/docs/en/plugins-reference#mcp-servers) for details on bundling MCP servers with plugins.\n\n## MCP installation scopes\n\nMCP servers can be configured at three scopes. The scope you choose controls which projects the server loads in and whether the configuration is shared with your team. Administrators can also deploy servers at the enterprise level via [managed configuration](https://code.claude.com/docs/en/mcp#managed-mcp-configuration).\n\n| Scope | Loads in | Shared with team | Stored in |\n| --- | --- | --- | --- |\n| Local | Current project only | No | `~/.claude.json` |\n| Project | Current project only | Yes, via version control | `.mcp.json` in project root |\n| User | All your projects | No | `~/.claude.json` |\n\n### Local scope\n\nLocal scope is the default. A local-scoped server loads only in the project where you added it and stays private to you. Claude Code stores it in `~/.claude.json` under that project’s path, so the same server won’t appear in your other projects. Use local scope for personal development servers, experimental configurations, or servers with credentials you don’t want in version control.\n\nThe term “local scope” for MCP servers differs from general local settings. MCP local-scoped servers are stored in `~/.claude.json` (your home directory), while general local settings use `.claude/settings.local.json` (in the project directory). See [Settings](https://code.claude.com/docs/en/settings#settings-files) for details on settings file locations.\n\n```\n# Add a local-scoped server (default)\nclaude mcp add --transport http stripe https://mcp.stripe.com\n\n# Explicitly specify local scope\nclaude mcp add --transport http stripe --scope local https://mcp.stripe.com\n```\n\nThe command writes the server into the entry for your current project inside `~/.claude.json`. The example below shows the result when you run it from `/path/to/your/project`:\n\n```\n{\n  "projects": {\n    "/path/to/your/project": {\n      "mcpServers": {\n        "stripe": {\n          "type": "http",\n          "url": "https://mcp.stripe.com"\n        }\n      }\n    }\n  }\n}\n```\n\n### Project scope\n\nProject-scoped servers enable team collaboration by storing configurations in a `.mcp.json` file at your project’s root directory. This file is designed to be checked into version control, ensuring all team members have access to the same MCP tools and services. When you add a project-scoped server, Claude Code automatically creates or updates this file with the appropriate configuration structure.\n\n```\n# Add a project-scoped server\nclaude mcp add --transport http paypal --scope project https://mcp.paypal.com/mcp\n```\n\nThe resulting `.mcp.json` file follows a standardized format:\n\n```\n{\n  "mcpServers": {\n    "shared-server": {\n      "command": "/path/to/server",\n      "args": [],\n      "env": {}\n    }\n  }\n}\n```\n\nFor security reasons, Claude Code prompts for approval before using project-scoped servers from `.mcp.json` files. If you need to reset these approval choices, use the `claude mcp reset-project-choices` command.\n\n### User scope\n\nUser-scoped servers are stored in `~/.claude.json` and provide cross-project accessibility, making them available across all projects on your machine while remaining private to your user account. This scope works well for personal utility servers, development tools, or services you frequently use across different projects.\n\n```\n# Add a user server\nclaude mcp add --transport http hubspot --scope user https://mcp.hubspot.com/anthropic\n```\n\n### Scope hierarchy and precedence\n\nWhen the same server is defined in more than one place, Claude Code connects to it once, using the definition from the highest-precedence source. The entire server entry from that source is used; fields are not merged across scopes.\n\n1. Local scope\n2. Project scope\n3. User scope\n4. [Plugin-provided servers](https://code.claude.com/docs/en/plugins)\n5. [claude.ai connectors](https://code.claude.com/docs/en/mcp#use-mcp-servers-from-claude-ai)\n\nThe three scopes match duplicates by name. Plugins and connectors match by endpoint, so one that points at the same URL or command as a server above is treated as a duplicate.\n\n### Environment variable expansion in `.mcp.json`\n\nClaude Code supports environment variable expansion in `.mcp.json` files, allowing teams to share configurations while maintaining flexibility for machine-specific paths and sensitive values like API keys.**Supported syntax:**\n\n- `${VAR}` \\- Expands to the value of environment variable `VAR`\n- `${VAR:-default}` \\- Expands to `VAR` if set, otherwise uses `default`\n\n**Expansion locations:**\nEnvironment variables can be expanded in:\n\n- `command` \\- The server executable path\n- `args` \\- Command-line arguments\n- `env` \\- Environment variables passed to the server\n- `url` \\- For HTTP server types\n- `headers` \\- For HTTP server authentication\n\n**Example with variable expansion:**\n\n```\n{\n  "mcpServers": {\n    "api-server": {\n      "type": "http",\n      "url": "${API_BASE_URL:-https://api.example.com}/mcp",\n      "headers": {\n        "Authorization": "Bearer ${API_KEY}"\n      }\n    }\n  }\n}\n```\n\nIf a required environment variable is not set and has no default value, Claude Code will fail to parse the config.\n\n## Practical examples\n\n### Example: Monitor errors with Sentry\n\n```\nclaude mcp add --transport http sentry https://mcp.sentry.dev/mcp\n```\n\nAuthenticate with your Sentry account:\n\n```\n/mcp\n```\n\nThen debug production issues:\n\n```\nWhat are the most common errors in the last 24 hours?\n```\n\n```\nShow me the stack trace for error ID abc123\n```\n\n```\nWhich deployment introduced these new errors?\n```\n\n### Example: Connect to GitHub for code reviews\n\nGitHub’s remote MCP server authenticates with a GitHub personal access token passed as a header. To get one, open your [GitHub token settings](https://github.com/settings/personal-access-tokens), generate a new fine-grained token with access to the repositories you want Claude to work with, then add the server:\n\n```\nclaude mcp add --transport http github https://api.githubcopilot.com/mcp/ \\\n  --header "Authorization: Bearer YOUR_GITHUB_PAT"\n```\n\nThen work with GitHub:\n\n```\nReview PR #456 and suggest improvements\n```\n\n```\nCreate a new issue for the bug we just found\n```\n\n```\nShow me all open PRs assigned to me\n```\n\n### Example: Query your PostgreSQL database\n\n```\nclaude mcp add --transport stdio db -- npx -y @bytebase/dbhub \\\n  --dsn "postgresql://readonly:pass@prod.db.com:5432/analytics"\n```\n\nThen query your database naturally:\n\n```\nWhat\'s our total revenue this month?\n```\n\n```\nShow me the schema for the orders table\n```\n\n```\nFind customers who haven\'t made a purchase in 90 days\n```\n\n## Authenticate with remote MCP servers\n\nMany cloud-based MCP servers require authentication. Claude Code supports OAuth 2.0 for secure connections.Claude Code marks a remote server as needing authentication when the server responds with `401 Unauthorized` or `403 Forbidden`. Either status code flags the server in `/mcp` so you can complete the OAuth flow. A custom server that returns a `WWW-Authenticate` header pointing to its authorization server gets the same automatic discovery as any other remote server.If you configured `headers.Authorization` for the server and the server rejects that header, Claude Code reports the connection as failed instead of falling back to OAuth. Check that the token is valid for the MCP endpoint, or remove the header to use the OAuth flow.\n\nAdd the server that requires authentication\n\nFor example:\n\n```\nclaude mcp add --transport http sentry https://mcp.sentry.dev/mcp\n```\n\nUse the /mcp command within Claude Code\n\nIn Claude code, use the command:\n\n```\n/mcp\n```\n\nThen follow the steps in your browser to login.\n\nTips:\n\n- Authentication tokens are stored securely and refreshed automatically\n- Use “Clear authentication” in the `/mcp` menu to revoke access\n- If your browser doesn’t open automatically, copy the provided URL and open it manually\n- If the browser redirect fails with a connection error after authenticating, paste the full callback URL from your browser’s address bar into the URL prompt that appears in Claude Code\n- OAuth authentication works with HTTP servers\n\n### Use a fixed OAuth callback port\n\nSome MCP servers require a specific redirect URI registered in advance. By default, Claude Code picks a random available port for the OAuth callback. Use `--callback-port` to fix the port so it matches a pre-registered redirect URI of the form `http://localhost:PORT/callback`.You can use `--callback-port` on its own (with dynamic client registration) or together with `--client-id` (with pre-configured credentials).\n\n```\n# Fixed callback port with dynamic client registration\nclaude mcp add --transport http \\\n  --callback-port 8080 \\\n  my-server https://mcp.example.com/mcp\n```\n\n### Use pre-configured OAuth credentials\n\nSome MCP servers don’t support automatic OAuth setup via Dynamic Client Registration. If you see an error like “Incompatible auth server: does not support dynamic client registration,” the server requires pre-configured credentials. Claude Code also supports servers that use a Client ID Metadata Document (CIMD) instead of Dynamic Client Registration, and discovers these automatically. If automatic discovery fails, register an OAuth app through the server’s developer portal first, then provide the credentials when adding the server.\n\nRegister an OAuth app with the server\n\nCreate an app through the server’s developer portal and note your client ID and client secret.Many servers also require a redirect URI. If so, choose a port and register a redirect URI in the format `http://localhost:PORT/callback`. Use that same port with `--callback-port` in the next step.\n\nAdd the server with your credentials\n\nChoose one of the following methods. The port used for `--callback-port` can be any available port. It just needs to match the redirect URI you registered in the previous step.\n\nUse `--client-id` to pass your app’s client ID. The `--client-secret` flag prompts for the secret with masked input:\n\n```\nclaude mcp add --transport http \\\n  --client-id your-client-id --client-secret --callback-port 8080 \\\n  my-server https://mcp.example.com/mcp\n```\n\nInclude the `oauth` object in the JSON config and pass `--client-secret` as a separate flag:\n\n```\nclaude mcp add-json my-server \\\n  \'{"type":"http","url":"https://mcp.example.com/mcp","oauth":{"clientId":"your-client-id","callbackPort":8080}}\' \\\n  --client-secret\n```\n\nUse `--callback-port` without a client ID to fix the port while using dynamic client registration:\n\n```\nclaude mcp add-json my-server \\\n  \'{"type":"http","url":"https://mcp.example.com/mcp","oauth":{"callbackPort":8080}}\'\n```\n\nSet the secret via environment variable to skip the interactive prompt:\n\n```\nMCP_CLIENT_SECRET=your-secret claude mcp add --transport http \\\n  --client-id your-client-id --client-secret --callback-port 8080 \\\n  my-server https://mcp.example.com/mcp\n```\n\nAuthenticate in Claude Code\n\nRun `/mcp` in Claude Code and follow the browser login flow.\n\nTips:\n\n- The client secret is stored securely in your system keychain (macOS) or a credentials file, not in your config\n- If the server uses a public OAuth client with no secret, use only `--client-id` without `--client-secret`\n- `--callback-port` can be used with or without `--client-id`\n- These flags only apply to HTTP and SSE transports. They have no effect on stdio servers\n- Use `claude mcp get <name>` to verify that OAuth credentials are configured for a server\n\n### Override OAuth metadata discovery\n\nPoint Claude Code at a specific OAuth authorization server metadata URL to bypass the default discovery chain. Set `authServerMetadataUrl` when the MCP server’s standard endpoints error, or when you want to route discovery through an internal proxy. By default, Claude Code first checks RFC 9728 Protected Resource Metadata at `/.well-known/oauth-protected-resource`, then falls back to RFC 8414 authorization server metadata at `/.well-known/oauth-authorization-server`.Set `authServerMetadataUrl` in the `oauth` object of your server’s config in `.mcp.json`:\n\n```\n{\n  "mcpServers": {\n    "my-server": {\n      "type": "http",\n      "url": "https://mcp.example.com/mcp",\n      "oauth": {\n        "authServerMetadataUrl": "https://auth.example.com/.well-known/openid-configuration"\n      }\n    }\n  }\n}\n```\n\nThe URL must use `https://`. `authServerMetadataUrl` requires Claude Code v2.1.64 or later. The metadata URL’s `scopes_supported` overrides the scopes the upstream server advertises.\n\n### Restrict OAuth scopes\n\nSet `oauth.scopes` to pin the scopes Claude Code requests during the authorization flow. This is the supported way to restrict an MCP server to a security-team-approved subset when the upstream authorization server advertises more scopes than you want to grant. The value is a single space-separated string, matching the `scope` parameter format in RFC 6749 §3.3.\n\n```\n{\n  "mcpServers": {\n    "slack": {\n      "type": "http",\n      "url": "https://mcp.slack.com/mcp",\n      "oauth": {\n        "scopes": "channels:read chat:write search:read"\n      }\n    }\n  }\n}\n```\n\n`oauth.scopes` takes precedence over both `authServerMetadataUrl` and the scopes the server discovers at `/.well-known`. Leave it unset to let the MCP server determine the requested scope set.If the authorization server advertises `offline_access` in `scopes_supported`, Claude Code appends it to the pinned scopes so the access token can be refreshed without a new browser sign-in.If the server later returns a 403 `insufficient_scope` for a tool call, Claude Code re-authenticates with the same pinned scopes. Widen `oauth.scopes` when a tool you need requires a scope outside the pin.\n\n### Use dynamic headers for custom authentication\n\nIf your MCP server uses an authentication scheme other than OAuth (such as Kerberos, short-lived tokens, or an internal SSO), use `headersHelper` to generate request headers at connection time. Claude Code runs the command and merges its output into the connection headers.\n\n```\n{\n  "mcpServers": {\n    "internal-api": {\n      "type": "http",\n      "url": "https://mcp.internal.example.com",\n      "headersHelper": "/opt/bin/get-mcp-auth-headers.sh"\n    }\n  }\n}\n```\n\nThe command can also be inline:\n\n```\n{\n  "mcpServers": {\n    "internal-api": {\n      "type": "http",\n      "url": "https://mcp.internal.example.com",\n      "headersHelper": "echo \'{\\"Authorization\\": \\"Bearer \'\\"$(get-token)\\"\'\\"}\'"\n    }\n  }\n}\n```\n\n**Requirements:**\n\n- The command must write a JSON object of string key-value pairs to stdout\n- The command runs in a shell with a 10-second timeout\n- Dynamic headers override any static `headers` with the same name\n\nThe helper runs fresh on each connection (at session start and on reconnect). There is no caching, so your script is responsible for any token reuse.Claude Code sets these environment variables when executing the helper:\n\n| Variable | Value |\n| --- | --- |\n| `CLAUDE_CODE_MCP_SERVER_NAME` | the name of the MCP server |\n| `CLAUDE_CODE_MCP_SERVER_URL` | the URL of the MCP server |\n\nUse these to write a single helper script that serves multiple MCP servers.\n\n`headersHelper` executes arbitrary shell commands. When defined at project or local scope, it only runs after you accept the workspace trust dialog.\n\n## Add MCP servers from JSON configuration\n\nIf you have a JSON configuration for an MCP server, you can add it directly:\n\nAdd an MCP server from JSON\n\n```\n# Basic syntax\nclaude mcp add-json <name> \'<json>\'\n\n# Example: Adding an HTTP server with JSON configuration\nclaude mcp add-json weather-api \'{"type":"http","url":"https://api.weather.com/mcp","headers":{"Authorization":"Bearer token"}}\'\n\n# Example: Adding a stdio server with JSON configuration\nclaude mcp add-json local-weather \'{"type":"stdio","command":"/path/to/weather-cli","args":["--api-key","abc123"],"env":{"CACHE_DIR":"/tmp"}}\'\n\n# Example: Adding an HTTP server with pre-configured OAuth credentials\nclaude mcp add-json my-server \'{"type":"http","url":"https://mcp.example.com/mcp","oauth":{"clientId":"your-client-id","callbackPort":8080}}\' --client-secret\n```\n\nVerify the server was added\n\n```\nclaude mcp get weather-api\n```\n\nTips:\n\n- Make sure the JSON is properly escaped in your shell\n- The JSON must conform to the MCP server configuration schema\n- You can use `--scope user` to add the server to your user configuration instead of the project-specific one\n\n## Import MCP servers from Claude Desktop\n\nIf you’ve already configured MCP servers in Claude Desktop, you can import them:\n\nImport servers from Claude Desktop\n\n```\n# Basic syntax\nclaude mcp add-from-claude-desktop\n```\n\nSelect which servers to import\n\nAfter running the command, you’ll see an interactive dialog that allows you to select which servers you want to import.\n\nVerify the servers were imported\n\n```\nclaude mcp list\n```\n\nTips:\n\n- This feature only works on macOS and Windows Subsystem for Linux (WSL)\n- It reads the Claude Desktop configuration file from its standard location on those platforms\n- Use the `--scope user` flag to add servers to your user configuration\n- Imported servers will have the same names as in Claude Desktop\n- If servers with the same names already exist, they will get a numerical suffix (for example, `server_1`)\n\n## Use MCP servers from Claude.ai\n\nIf you’ve logged into Claude Code with a [Claude.ai](https://claude.ai/) account, MCP servers you’ve added in Claude.ai are automatically available in Claude Code:\n\nConfigure MCP servers in Claude.ai\n\nAdd servers at [claude.ai/customize/connectors](https://claude.ai/customize/connectors). On Team and Enterprise plans, only admins can add servers.\n\nAuthenticate the MCP server\n\nComplete any required authentication steps in Claude.ai.\n\nView and manage servers in Claude Code\n\nIn Claude Code, use the command:\n\n```\n/mcp\n```\n\nClaude.ai servers appear in the list with indicators showing they come from Claude.ai.\n\nFrom v2.1.161, connectors you have never signed in to are collapsed behind a `Show unused connectors` row at the end of the claude.ai section, so an organization-provisioned list doesn’t fill the panel. Select the row to expand them. A connector you signed in to before stays visible even when it currently needs re-authentication.Claude.ai connectors are fetched only when your active [authentication method](https://code.claude.com/docs/en/authentication#authentication-precedence) is your Claude.ai subscription. They are not loaded when `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `apiKeyHelper`, or a third-party provider such as Bedrock or Vertex is active, even if you previously ran `/login`. If `/mcp` does not list a connector you added, run `/status` to confirm which authentication method is active, unset that environment variable or remove the `apiKeyHelper` setting, then run `/login` to select your Claude.ai account.A server you’ve added in Claude Code takes [precedence](https://code.claude.com/docs/en/mcp#scope-hierarchy-and-precedence) over a claude.ai connector that points at the same URL. When this happens, `/mcp` lists the connector as hidden and shows how to remove the duplicate if you’d rather use the connector.Some Anthropic-hosted connectors, such as Microsoft 365, Gmail, and Google Calendar, do not support local OAuth from Claude Code because the upstream identity provider only accepts the redirect URL that claude.ai registered. From v2.1.162, authenticating one of these hosts in `/mcp` shows a message directing you to connect it at Settings → Connectors on claude.ai instead. Once connected there, the connector appears in Claude Code automatically.To disable claude.ai MCP servers in Claude Code, set the `ENABLE_CLAUDEAI_MCP_SERVERS` environment variable to `false`:\n\n```\nENABLE_CLAUDEAI_MCP_SERVERS=false claude\n```\n\n## Use Claude Code as an MCP server\n\nYou can use Claude Code itself as an MCP server that other applications can connect to:\n\n```\n# Start Claude as a stdio MCP server\nclaude mcp serve\n```\n\nYou can use this in Claude Desktop by adding this configuration to claude\\_desktop\\_config.json:\n\n```\n{\n  "mcpServers": {\n    "claude-code": {\n      "type": "stdio",\n      "command": "claude",\n      "args": ["mcp", "serve"],\n      "env": {}\n    }\n  }\n}\n```\n\n**Configuring the executable path**: The `command` field must reference the Claude Code executable. If the `claude` command is not in your system’s PATH, you’ll need to specify the full path to the executable.To find the full path:\n\n```\nwhich claude\n```\n\nThen use the full path in your configuration:\n\n```\n{\n  "mcpServers": {\n    "claude-code": {\n      "type": "stdio",\n      "command": "/full/path/to/claude",\n      "args": ["mcp", "serve"],\n      "env": {}\n    }\n  }\n}\n```\n\nWithout the correct executable path, you’ll encounter errors like `spawn claude ENOENT`.\n\nTips:\n\n- The server provides access to Claude’s tools like View, Edit, LS, etc.\n- In Claude Desktop, try asking Claude to read files in a directory, make edits, and more.\n- Note that this MCP server is only exposing Claude Code’s tools to your MCP client, so your own client is responsible for implementing user confirmation for individual tool calls.\n\n## MCP output limits and warnings\n\nWhen MCP tools produce large outputs, Claude Code helps manage the token usage to prevent overwhelming your conversation context:\n\n- **Output warning threshold**: Claude Code displays a warning when any MCP tool output exceeds 10,000 tokens\n- **Configurable limit**: you can adjust the maximum allowed MCP output tokens using the `MAX_MCP_OUTPUT_TOKENS` environment variable\n- **Default limit**: the default maximum is 25,000 tokens\n- **Scope**: the environment variable applies to tools that don’t declare their own limit. Tools that set [`anthropic/maxResultSizeChars`](https://code.claude.com/docs/en/mcp#raise-the-limit-for-a-specific-tool) use that value instead for text content, regardless of what `MAX_MCP_OUTPUT_TOKENS` is set to. Tools that return image data are still subject to `MAX_MCP_OUTPUT_TOKENS`\n\nTo increase the limit for tools that produce large outputs:\n\n```\nexport MAX_MCP_OUTPUT_TOKENS=50000\nclaude\n```\n\nThis is particularly useful when working with MCP servers that:\n\n- Query large datasets or databases\n- Generate detailed reports or documentation\n- Process extensive log files or debugging information\n\n### Raise the limit for a specific tool\n\nIf you’re building an MCP server, you can allow individual tools to return results larger than the default persist-to-disk threshold by setting `_meta["anthropic/maxResultSizeChars"]` in the tool’s `tools/list` response entry. Claude Code raises that tool’s threshold to the annotated value, up to a hard ceiling of 500,000 characters.This is useful for tools that return inherently large but necessary outputs, such as database schemas or full file trees. Without the annotation, results that exceed the default threshold are persisted to disk and replaced with a file reference in the conversation.\n\n```\n{\n  "name": "get_schema",\n  "description": "Returns the full database schema",\n  "_meta": {\n    "anthropic/maxResultSizeChars": 200000\n  }\n}\n```\n\nThe annotation applies independently of `MAX_MCP_OUTPUT_TOKENS` for text content, so users don’t need to raise the environment variable for tools that declare it. Tools that return image data are still subject to the token limit.\n\nIf you frequently encounter output warnings with specific MCP servers you don’t control, consider increasing the `MAX_MCP_OUTPUT_TOKENS` limit. You can also ask the server author to add the `anthropic/maxResultSizeChars` annotation or to paginate their responses. The annotation has no effect on tools that return image content; for those, raising `MAX_MCP_OUTPUT_TOKENS` is the only option.\n\n## Respond to MCP elicitation requests\n\nMCP servers can request structured input from you mid-task using elicitation. When a server needs information it can’t get on its own, Claude Code displays an interactive dialog and passes your response back to the server. No configuration is required on your side: elicitation dialogs appear automatically when a server requests them.Servers can request input in two ways:\n\n- **Form mode**: Claude Code shows a dialog with form fields defined by the server (for example, a username and password prompt). Fill in the fields and submit.\n- **URL mode**: Claude Code opens a browser URL for authentication or approval. Complete the flow in the browser, then confirm in the CLI.\n\nTo auto-respond to elicitation requests without showing a dialog, use the [`Elicitation` hook](https://code.claude.com/docs/en/hooks#elicitation).If you’re building an MCP server that uses elicitation, see the [MCP elicitation specification](https://modelcontextprotocol.io/docs/learn/client-concepts#elicitation) for protocol details and schema examples.\n\n## Use MCP resources\n\nMCP servers can expose resources that you can reference using @ mentions, similar to how you reference files.\n\n### Reference MCP resources\n\nList available resources\n\nType `@` in your prompt to see available resources from all connected MCP servers. Resources appear alongside files in the autocomplete menu.\n\nReference a specific resource\n\nUse the format `@server:protocol://resource/path` to reference a resource:\n\n```\nCan you analyze @github:issue://123 and suggest a fix?\n```\n\n```\nPlease review the API documentation at @docs:file://api/authentication\n```\n\nMultiple resource references\n\nYou can reference multiple resources in a single prompt:\n\n```\nCompare @postgres:schema://users with @docs:file://database/user-model\n```\n\nTips:\n\n- Resources are automatically fetched and included as attachments when referenced\n- Resource paths are fuzzy-searchable in the @ mention autocomplete\n- Claude Code automatically provides tools to list and read MCP resources when servers support them\n- Resources can contain any type of content that the MCP server provides (text, JSON, structured data, etc.)\n\n## Scale with MCP Tool Search\n\nTool search keeps MCP context usage low by deferring tool definitions until Claude needs them. Only tool names and server instructions load at session start, so adding more MCP servers has minimal impact on your context window. Claude Code does not impose a fixed per-server tool cap; the practical limit is your context window budget.\n\n### How it works\n\nTool search is enabled by default. MCP tools are deferred rather than loaded into context upfront, and Claude uses a search tool to discover relevant ones when a task needs them. Only the tools Claude actually uses enter context. From your perspective, MCP tools work exactly as before.If you prefer threshold-based loading, set `ENABLE_TOOL_SEARCH=auto` to load schemas upfront when they fit within 10% of the context window and defer only the overflow. See [Configure tool search](https://code.claude.com/docs/en/mcp#configure-tool-search) for all options.\n\n### For MCP server authors\n\nIf you’re building an MCP server, the server instructions field becomes more useful with Tool Search enabled. Server instructions help Claude understand when to search for your tools, similar to how [skills](https://code.claude.com/docs/en/skills) work.Add clear, descriptive server instructions that explain:\n\n- What category of tasks your tools handle\n- When Claude should search for your tools\n- Key capabilities your server provides\n\nClaude Code truncates tool descriptions and server instructions at 2KB each. Keep them concise to avoid truncation, and put critical details near the start.\n\n### Configure tool search\n\nTool search is enabled by default: MCP tools are deferred and discovered on demand. Claude Code disables it by default on Vertex AI. It is also disabled when `ANTHROPIC_BASE_URL` points to a non-first-party host, since most proxies do not forward `tool_reference` blocks. Set `ENABLE_TOOL_SEARCH` explicitly to override either fallback.Tool search requires a model that supports `tool_reference` blocks. Haiku models do not support it. On Vertex AI, tool search is supported for Claude Sonnet 4.5 and later and Claude Opus 4.5 and later.Control tool search behavior with the `ENABLE_TOOL_SEARCH` environment variable:\n\n| Value | Behavior |\n| --- | --- |\n| (unset) | All MCP tools deferred and loaded on demand. Falls back to loading upfront on Vertex AI or when `ANTHROPIC_BASE_URL` is a non-first-party host |\n| `true` | All MCP tools deferred. Claude Code sends the beta header even on Vertex AI and through proxies. Requests fail on Vertex AI models earlier than Sonnet 4.5 or Opus 4.5, or on proxies that do not support `tool_reference` blocks |\n| `auto` | Threshold mode: tools load upfront if they fit within 10% of the context window, deferred otherwise |\n| `auto:N` | Threshold mode with a custom percentage, where `N` is 0-100. For example, `auto:5` for 5% |\n| `false` | All MCP tools loaded upfront, no deferral |\n\n```\n# Use a custom 5% threshold\nENABLE_TOOL_SEARCH=auto:5 claude\n\n# Disable tool search entirely\nENABLE_TOOL_SEARCH=false claude\n```\n\nOr set the value in your [settings.json `env` field](https://code.claude.com/docs/en/settings#available-settings).You can also disable the `ToolSearch` tool specifically:\n\n```\n{\n  "permissions": {\n    "deny": ["ToolSearch"]\n  }\n}\n```\n\n### Exempt a server from deferral\n\nIf a server’s tools should always be visible to Claude without a search step, set `alwaysLoad` to `true` in that server’s configuration. Every tool from that server then loads into context at session start regardless of the `ENABLE_TOOL_SEARCH` setting. Use this for a small number of tools that Claude needs on every turn, since each upfront tool consumes context that would otherwise be available for your conversation.The following `.mcp.json` entry exempts one HTTP server while leaving other servers deferred:\n\n```\n{\n  "mcpServers": {\n    "core-tools": {\n      "type": "http",\n      "url": "https://mcp.example.com/mcp",\n      "alwaysLoad": true\n    }\n  }\n}\n```\n\nThe `alwaysLoad` field is available on all server types and requires Claude Code v2.1.121 or later. An MCP server can also mark individual tools as always-loaded by including `"anthropic/alwaysLoad": true` in the tool’s `_meta` object, which has the same effect for that tool only.Setting `alwaysLoad: true` also blocks startup until the server connects, capped at the standard 5-second connect timeout. This applies even though MCP startup is otherwise [non-blocking by default](https://code.claude.com/docs/en/env-vars), since the tools must be present when the first prompt is built. Other servers continue to connect in the background.\n\n## Use MCP prompts as commands\n\nMCP servers can expose prompts that become available as commands in Claude Code.\n\n### Execute MCP prompts\n\nDiscover available prompts\n\nType `/` to see all available commands, including those from MCP servers. MCP prompts appear with the format `/mcp__servername__promptname`.\n\nExecute a prompt without arguments\n\n```\n/mcp__github__list_prs\n```\n\nExecute a prompt with arguments\n\nMany prompts accept arguments. Pass them space-separated after the command:\n\n```\n/mcp__github__pr_review 456\n```\n\n```\n/mcp__jira__create_issue "Bug in login flow" high\n```\n\nTips:\n\n- MCP prompts are dynamically discovered from connected servers\n- Arguments are parsed based on the prompt’s defined parameters\n- Prompt results are injected directly into the conversation\n- Server and prompt names are normalized (spaces become underscores)\n\n## Managed MCP configuration\n\nFor organizations that need centralized control over which MCP servers users can connect to, see [Managed MCP configuration](https://code.claude.com/docs/en/managed-mcp). It covers deploying a fixed server set with `managed-mcp.json`, restricting servers with `allowedMcpServers` and `deniedMcpServers`, and what users see when a server is blocked.', 'extras': {'signature': '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'}}

</details>

<details>
<summary>cursor-docs-agent-rules-mcp-skills-cli</summary>

{'type': 'text', 'text': 'Empty string or no content matches the provided lesson guidelines.', 'extras': {'signature': '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'}}

</details>

<details>
<summary>mcp-inspector-model-context-protocol</summary>

{'type': 'text', 'text': 'The MCP Inspector is an interactive developer tool for testing and debugging MCP servers.\n\n## Getting started\n\n### Installation and basic usage\n\nThe Inspector runs directly through `npx` without requiring installation:\n\n```\nnpx @modelcontextprotocol/inspector <command>\n```\n\n```\nnpx @modelcontextprotocol/inspector <command> <arg1> <arg2>\n```\n\n#### Inspecting servers from npm or PyPI\n\nA common way to start server packages from npm or PyPI.\n\n- npm package\n- PyPI package\n\n```\nnpx -y @modelcontextprotocol/inspector npx <package-name> <args>\n# For example\nnpx -y @modelcontextprotocol/inspector npx @modelcontextprotocol/server-filesystem /Users/username/Desktop\n```\n\n```\nnpx @modelcontextprotocol/inspector uvx <package-name> <args>\n# For example\nnpx @modelcontextprotocol/inspector uvx mcp-server-git --repository ~/code/mcp/servers.git\n```\n\n#### Inspecting locally developed servers\n\nTo inspect servers locally developed or downloaded as a repository, the most common way is:\n\n- TypeScript\n- Python\n\n```\nnpx @modelcontextprotocol/inspector node path/to/server/index.js args...\n```\n\n```\nnpx @modelcontextprotocol/inspector \\\n  uv \\\n  --directory path/to/server \\\n  run \\\n  package-name \\\n  args...\n```\n\nPlease carefully read any attached README for the most accurate instructions.\n\n## Feature overview\n\nThe Inspector provides several features for interacting with your MCP server:\n\n### Server connection pane\n\n- Allows selecting the transport for connecting to the server\n- For local servers, supports customizing the command-line arguments and environment\n\n### Resources tab\n\n- Lists all available resources\n- Shows resource metadata (MIME types, descriptions)\n- Allows resource content inspection\n- Supports subscription testing\n\n### Prompts tab\n\n- Displays available prompt templates\n- Shows prompt arguments and descriptions\n- Enables prompt testing with custom arguments\n- Previews generated messages\n\n### Tools tab\n\n- Lists available tools\n- Shows tool schemas and descriptions\n- Enables tool testing with custom inputs\n- Displays tool execution results\n\n### Notifications pane\n\n- Presents all logs recorded from the server\n- Shows notifications received from the server\n\n## Best practices\n\n### Development workflow\n\n1. Start Development   - Launch Inspector with your server\n   - Verify basic connectivity\n   - Check capability negotiation\n2. Iterative testing   - Make server changes\n   - Rebuild the server\n   - Reconnect the Inspector\n   - Test affected features\n   - Monitor messages\n3. Test edge cases   - Invalid inputs\n   - Missing prompt arguments\n   - Concurrent operations\n   - Verify error handling and error responses', 'extras': {'signature': 'EvwpCvkpAQw51seF6ntR1/uWD/6iITqowJWB+TUbaMP5Yf2Ot7OmGZ53RGwgxmPqrksGmLCFPFg7tEItd0dI/jjSdZN2mMVJjf4Fn0NbDWKyqFU4Sgzm+xvhIQtIq6oXRNxSU8xAEYzGqI2pZ08gkXQqOVjqXlBC3EyuIsbQ+hHXwQLpXyUmSZWN3J+0jZzdisxuzBCkXER2eZRaogW1z+viV8XvVxnzkPadJr693TP+BmIgMHrRWv9IeHWrmv5TevI4Mh9BSpohSDyqFJcHJzN0qb8GXzqixFN1PebSHvN7BTokUyQcTsEoRhRjtBZXI2ijQtkmyEg1tvvLuCxSIAR1+uUd42ja635Cw+RSA/zvjJ+9r3gGbv4GVpsB5W+61wl13JEAs5t8gV3koOJsRwq5mN5fxO8vLyNKcz77mJe6eR7Rrn3Dra55XZZX/HFCrLRmS0X8rvPljTSqrw/1GuR1dKb2K5xqV2AD2L53rTt/zkS+AZLieiiSPmROc3en6qA88gkVSao5EprZX2Z3iBsJBWvr3/YM1EIEuIj3qshj5ER7+Bg6ii8SIITu1evvua1+bINB7hfFHpzQytIbMSIz3MMA5cMBcsEko08vD5N1dO7r5e3Hft4tL/8FbTugmFXTIFHhFyFCxW+AKxALhDN/OYHW1ejt2xp4r/84rMz/tMoZCROV+HiPjXxmnwLkraTZqJtjVUdbjeY3ZgV1rmaVkj59pbtjjnyLLK4oYmebvGyr8MGkSta1KxuXM1gGgFP6i3q7BjY15vZwtiy9thQ7Tep9DRvPuX0h/lr2c2UjjwvQrAhEgJD7wc98fbhUMnCCtPX00ZnDx6PQIvOvaRtHiCDf8wxG1qbAt8ZFnqbAQjR6HtKusKLPNKllX2C5efNY9z97ngcJ6kfCL0JZq+TDe7MfDl9EE+vpJRHTL1q2ntjCTjHLN501b4QT8/37GOw1FtpsvDFdF/80F/LD3R6OViV+w2SJG7Rn95g9GqKv2CT8d2P7gHSQIaRGVKtmNazw01LUR5th9XmpueFGfUs9oTB6zI28mUCftF/Li9wA4UzdeQkuKe97uCd+hpi1oaJrQPg9tfvTsTLUO/bW6oq0TVKC7DBfITaP/1iMGHW4bHT0l762Lgl3xnlI2R17KzYzt3L2qZGNgfgtg9ZtAQnV0/STOog+JcMg14+VWx4Kr32SIFOXXlLmMzlrFEbBCxczSxV9vXnlRTSYD6lYNl/Cc5nkNrHCrGOAe8xKTLrWxZdZ1b/p5sItNKE9ZUz/wzHTvt1B6qcWX/OVPbAuJ8/4v0RCY6uEEhVqPcEUdqSuzGVwF3C/e75gq16eqO5Wj9pL4l/S+XYfhGRasQr+d3gx6pTxV/hwvjRdY8BYO7pUSdT/zufUzmRzyOS/Ixs3z6XoQE8Npm+wF1Z46YhUWBQnc1CxyfcXYqDMuCLV+B6ZQoK4kq1ngUUWZ8thQM/G4Lz8B+4AMM0AwWcUz7NdsScVVXZNm80koDB4MzghE7D1cEV6irGNsYMd1IFHHOgH3aRvhnz9tJ5fBs0Ubn4EY8ZDEA5kZf7cZU0o6wSItkPjASSIR3caDqEcANZPMr/30ofCmw2thyqoD+tyzj6bJBMuDgmeRQRmjWoR3YacBi6tDWeS4Fmtwy6SSXT0LLBueLdw2jyhg+IVnshssoQpasullYYCaJwar7omFgLFJSug7nc8l9k4rouc4wuoLZJZCPhk2TE97w/JL8EaJMjsHwi93W3Yr99qqhMdge7o+fKvM7jaI8uv1ntz2qiftA+c8rJee6R4HtyuZxsBp1ChckhQAdEaEG9tZ0Z/+REbN13oJvceCkEuFlf2uA7eYK2D5YWDFXjDITze6EFFawSa1ETyAw8ZDZ7ig9F4F4Jq1OK76+xk1IC3r1thSdxuVKzck3C/MBdMJT1GveoqibmG4OUbfr9EgSW8GD+ovOiacDmwr2qrVrPIZ/qFI8XTCiU6286v5fQXd6YM3K512jXPaqwB3yU8oEdBuxUYENXMjlOeN0T+MTjm2ULXa3KZyJ3Ix/VtWD/c8peNRF9/MnBEhZPjXmSrqSMgftTmuDeGt5ZqJPtXz9OZlqK2Ly+omuSCMoDdBMn9P/OArDKqo9R6SP02L4iybYLBZtuudEAYsfgeQokjtr3AtWcVoab2BSEgvTT+mAfJI8pq5KHqTyO1iliKFnFndx66tYBdaM/i7Jcj39VJnqZ6vyEBDBEr1DifgS34jUAhVKLGhWRJIPxfcQCNRxAD/ZYK9qvKqD0GwjGs4xp4dIJbPgVnQNpYV6uXkrq+7tnsj76AGXEvjHeQDzeaUhblft6i8dJxeDbSe7J+mtA2k285cgXRus5WZuS6gt/XIpEa7kz2yQ07aXAoClNfrQ7yyWzBPHgT0ISAyVu5pUa7DuAY2IjVqtgcA4qxCzO9VYJ/sz8MdYAUhNS8Ck7pLI3AjjlCctHLWVJpwwTr7ujJe1A7tu5U6OZLYdeUTFyefe7bJb2U47lCjsZFqONIW6qVDE7Ubi+SZtwSgxgVYul5LJ1tbHScpyydfsP421kwNz7HPjhgb9YNBKb3/Nc4LZMedvkdCv8yiIXmS9BmPEVVWLCGezqLPsWHIq4KmLHsLZRnaP7Hpy3ioPC7kZBy9htHkagZlhwQHaf1iEWVWgHhPCDPFp5BV8rxsWpLysRbzPdaecKcYcQS4yYLaM/t6OZXOEXwnEaSfaHTbki77cSjz7wW1ccemYNaJ/ewLC7HzL5NGFxL678oOEk+qJg/L899cZ6rpY9vG9MXKWfAOX/yfU6u+Olsdd5FTVOw/b3BO1NYkrgW777zlkdJmhwv+IImXvAJSqedj9DaiwvumMHt5mGtqNR7TT1AYmZw2zrxS+JVc/miHiLTl9ACxX7Kqe7mFujcQ1xe1UzXOKtZommPvY4jKVjyDJAc3iCAMr6N2CTdShNh1v+oa0BJ5KqdQA5hmfF3fqrXosiGLTkVce5W9N2z1o1H5jtNYjj56XabUjY4EIXqYvSYV+Z+nkim/QRE9kYxONvhZL2xhh0GX1MlAQY/rT+nn/Kydb5Z0kv/gkxM0V3VPoBrMuTZEjC94HnsQGK7071c/ItqU5k/zKNz9IgI8C7reCp7xwSZW8U8rkVQyfSuhM2nGUvG4A8mPDK69KO3M9LT0dkdEjY5ugbH4JQd1e1ChuE2wABB5sHPktWadcoD1B2XTNqylYyAq41GMSNLFbpM5Sc/T3oPNu7ZKPtdcmyu4XLJUXszWjJFR7Qa1YaVm9pId6M3XDTO0wrseTbSSp3Yjl/RiOKR3YID0rDNnhr+IgNRgs/Yi5IXw2YRFqULi9mQELgmbGcYZjAIGkvCxaEvuQYrkiVHjrpnSAbi5Lo8VhBvHFVTH8nHBgRPUB/BQOM6vYB3nfV/WS+O4RLhj5p5A/DpFkbGHYPruTWAANT5osO6AMorIYAj+MJ64D+G4RW+VWR9KMYgd8kZEg57jE6zXlkMYLb1UEo7zV5SvGbmVuSPggDj+kUBoQETNs45nQXMWD1e/U7yOI+2J1gDm98paZKbW8mnAiQf2mlJhSeJ55GgpeuQ4KoazWfe+nSNftNmJNHr9vH/lB9guoMNYLyllfOOFxnXizZmK/meORKwSPBSbKS4hRWt8mkvO61hhzN1g2GNjZj3gj+oF8qhHO2hKihoBE7MpwVwsOye6sR65ebwKz9NOCYA5okKfdKtPx1lZErWzzgGD5/2VREu0FhE5iJe/tB5OBfbMxzLsHzxmdxugOO/brk5/ma1CIDuyNYSqCgtsnFhgnej5F6q5iDsDY8XQBSZ7dSrulcDFlc40+KlUfN1dxv6Mk4cZ6nCnhIXOh6umZLTuBkNQ1PzovYsopoicG9d/jrw8UJ0q0x2OgVHhx3KuQq0tqq54gLKnDZ/VXYIsLhAEps4u71X+uvIai+6QPhVEf4tUsWnlG8M7VueY0EoM806L1Nlq4o/bkhIW+LGwt4oFJBoT4oZFOH2gfpJ8PlcMPf2GqCBNiCnfaT9eMx+OurNgxL53dFVP6fyiobZH/D/8HqlMfA9uPADe32EHlm7BDVM0KpAO5rqdWtiD3W0z298q40mQevvsHibEC0qnbO0MjeOnYYnm5pwkc54RkD+drZCk1vWdlTgQsDGoZCIN/AvBQ6fRwdE4uh66Dfg+HNVmQpcJdUNzoUpqE2v19l7J7EBzuk3tnhC1WymsYYzzIbaH6els6+XIgYrSBEABOQz5HRWfb2e7qGA3phKaiAWjUbPeNzI82nxacKrVE5wG+TlaSud+NVAZf7ZCZTnRHb1kF8vTTsSQON9zbgH5jwLwOtq6zIJEY0BdVG8TcSZbhJUNFVq/gdyGsW7MnwoEq0jnWadb527pjzJBdOjhFRxU6RV2WjpGGBnvqJOfM9atR/kl2tlcORRTn9QgMmXaU1MUAwO8vL24a1hXOXs6vxJtlv1aTz1AGKEaXyt0hbdJK7ChxC84PGronkoLroVN8w990TO20DLI+jprKpWN3GaN9gfqXqConY0dpzPgF5V96maUybUMdnxRWVZzagQtbnjHPeCSCydYQkBz3x/9Wo9Mq5xOdLAUyZNG6eJlaC56rNw08fS6BZlXfDRwIznUTvsuW+kikmm7RtZo5upJLCpw9Ivv99RoXCmPyMqAVNaq7eUuDgMmj/0djNFLAl6WLy5p42W/OiNoE7nPtwaPxnoRGGQcphPWZIBOpLzhP9SycAN4bPszOaw53Vt4eAL78PnCiY4vjfyHp0bMhbRtOx3XuNUfVOZ6X3xkGJm0sjIOPEiRGllYXjfynvisqKyX4Dv2FGP0FXtHUqDIklsNZUBeLkl9BmJOzSwtAcie2wcm/w8zOh8dfCM+7ZhF8M520qvThKFULhMPsjbPVT2OGXl2DSFY6663MBtOXaFPP+g9pH0u71vScX71R4BpHjWolsRK7Wk94u/eOmOD5+v2JKbJH+VGoTj83nNZp8y2UnSs6AExZwD9bVrSwqJ7xArRLnVRFV7Sw2QY7lnNlY8C3TjOWzwH954f3ye7syunLD+6crCFp44yba3gaqUvzc/yNN12sKFRh+Kt3RUbtk+qRT17d3oZRFQGV6PwkBZ50Y6Am/fWDVyEU0ejK4QO1DDt4cfxSZA6H6VH+JHJYpW+5SPhTGcAKvt+6tzj9+MkPAwtU+dknh0zkzFj4GeAl7VHJgNgTbBlb62z8l3R3ercbWcB+LSmSlaaEliEsRqlB6NeDGubIiFR7vZVNeZbmJKPitf3+aOjBnneC5s+4uMkOACRFKvDIHXgZHP4hUIBaXIYynSGYbdlDCwzYu0Wf96BCD0pdlpdYNXB6LrN0H/SY0/j76VxnAQOkVvdVIrhY/73obrONYpy6dzMBmkv+MeHJ2aYDCLkievWsYrIFJtNDkFw3hzZZ55ONCHsvm8306G+hNGJXg5hpr/dslsN5xfYOUzcQXNIEDDwvILPk6D3zs4cIjxl3npd9xyKhXEDc2DFUwQF64LeeBhoPf2FCts/R+kYWeJbbRP1gYv842HVITBx4ojf3kbs0ENa7SvbbsRwBIsuF/FXaG83H3nqoHXr9/u7Z2IxJL1NLAjA+lAFoBx5lEqqvoiivGxTo8a2YGjb6DgQFuIL8KD4YC9TK2o8uBVTe77gwX4u/2QvnfJiDIBWg8IQJIVLvGy8qKREZpUWzJH8oRFIf80c6hNwhQDKjtahqS/E0FQXqHfgz755oeFLLGhagslLKKy9PoCzErNam8NJNd+a//tDY+KhzvSMLOD8X22y6zEyAoKA1cxLJvmDoj0Jf1269Apmk1HYTQPAMya+KqdKsCVOAoOOoh9dOvCXcYa2yu7M+SOnYNWpcp5zJ4cvjsJWGiftoXIL7giSX83koWzBzGFPAdsepM1P5JsNNpaUjWtsJL6pXKek1HmiwRm7vqHwGTQJsQOQnlUpTD07TKs46iimk7cbu5Cn9UjN0wVSU7Vr0me3kAt7Uk6bYnH4FVoop+WacR0HYUwXuREqJ5LxuNG5spq+t8m0NWJkH87ns0Y7oaOqMuGT6n76yfxtEgoG09v+sOMpL0NH0a/wyV7fykjoKTCxwoC2t2mhRYf9FcVwJgu/BKB0OpfexGLIC2fx+UqVqfQ5Pnu+thz1Wv3D0r46suQ+3RvGA7qz9XUgHr7KiCnPAdOuodmt4QouYRp9wjCACwpWtP9d4nBJdcJAvtQo82qey/L/NR10Dr5ZX5NBYTuC7oGv3rVyhaSEbRtoxqmCyqe34LXy9WM5BixmCWzVgG+1AmN9CPJBUtZVcWTGnS3FqRMYM1eodgE1dyx84TAAOHuhF5GkUwD8FAmxFfylLX3wi+lLot150Cs96tTDk6hyQhkyTL9todxae2BpxMwluXmshSEVdMrN2rQHATwIXk7+o9RABxOomRwTpEN8Q5iZZTSplMOpYE5z0fldry9Yu97wOYOuM6Ah17ZnXGtRsxiIQ4gdMPO9Dd24THOesRwlcTOM+SJZwAsqHXGYMQz71gyrCa7JO89enYLOtQseCWKo2qnqv8LA0psDYQzm+h/OeyhYxEpgrDERw7P4t8Hee/9h3/kbJtAb72K2HQnKl0GDi6hjLuMXofxCR4RaZOA0xZy5Vi0RqZQEj7E+HCR0+5DUEeleNNYRGlCALKqoE7Qa+uGGmDqhE5XMngIdJu0Sfhp+geRK3fCTBqObzqkNm5+W7ceoFPQ+zwWUv8H0Zg6qaH2cMYWNr0EAu+yWurhEc9eELFI/uyO4Tsnj5ADwtF1mI9OqKvazCwNo6eJY8LNIEdIFVJudMV9MPWHBjROjadQytQeb2JBMFTQANOJWrccAmoAKAkYrJXSpTRNjqOA10mpgloJNx5lDkHdczsrcZXqRkJ5qBmmhyy9m1Je8aIc6sdkL1wKi+FWqqwCHt7usJiZ9Qv/CTQWs1oLeeQXNM6bzw6gPZ8x/rDc9J3Xi+uqfUD5TiTb7iHA+GlKclckGfUdgYVjSPyVYcQpqWawzWdHnYfmUaFf7J65cUU60rkbCuu2/0b9fQANNzdiaGaRbuyi7m0sQM0tAXrXWNlCFs3NG4Baf0sJqBfYcZ/O4AM6DAkRHjdJVH54UtvurfER3dCU18J9LqjqVcDF8V+1xaP4i5ELGhCbBgafmH196vFcgKw='}}

</details>

<details>
<summary>prompts-model-context-protocol</summary>

{'type': 'text', 'text': 'The Model Context Protocol (MCP) provides a standardized way for servers to expose prompt\ntemplates to clients. Prompts allow servers to provide structured messages and\ninstructions for interacting with language models. Clients can discover available\nprompts, retrieve their contents, and provide arguments to customize them.\n\n## User Interaction Model\n\nPrompts are designed to be **user-controlled**, meaning they are exposed from servers to\nclients with the intention of the user being able to explicitly select them for use.Typically, prompts would be triggered through user-initiated commands in the user\ninterface, which allows users to naturally discover and invoke available prompts.For example, as slash commands:https://mintcdn.com/mcp/4ZXF1PrDkEaJvXpn/specification/2025-06-18/server/slash-command.png?fit=max&auto=format&n=4ZXF1PrDkEaJvXpn&q=85&s=7f003e36d881dd6f3e5b8cbdd85e5ca5However, implementors are free to expose prompts through any interface pattern that suits\ntheir needs—the protocol itself does not mandate any specific user interaction\nmodel.\n\n## Capabilities\n\nServers that support prompts **MUST** declare the `prompts` capability during\n[initialization](https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle#initialization):\n\n```\n{\n  "capabilities": {\n    "prompts": {\n      "listChanged": true\n    }\n  }\n}\n```\n\n`listChanged` indicates whether the server will emit notifications when the list of\navailable prompts changes.\n\n## Protocol Messages\n\n### Listing Prompts\n\nTo retrieve available prompts, clients send a `prompts/list` request. This operation\nsupports [pagination](https://modelcontextprotocol.io/specification/2025-06-18/server/utilities/pagination).**Request:**\n\n```\n{\n  "jsonrpc": "2.0",\n  "id": 1,\n  "method": "prompts/list",\n  "params": {\n    "cursor": "optional-cursor-value"\n  }\n}\n```\n\n**Response:**\n\n```\n{\n  "jsonrpc": "2.0",\n  "id": 1,\n  "result": {\n    "prompts": [\n      {\n        "name": "code_review",\n        "title": "Request Code Review",\n        "description": "Asks the LLM to analyze code quality and suggest improvements",\n        "arguments": [\n          {\n            "name": "code",\n            "description": "The code to review",\n            "required": true\n          }\n        ]\n      }\n    ],\n    "nextCursor": "next-page-cursor"\n  }\n}\n```\n\n### Getting a Prompt\n\nTo retrieve a specific prompt, clients send a `prompts/get` request. Arguments may be\nauto-completed through [the completion API](https://modelcontextprotocol.io/specification/2025-06-18/server/utilities/completion).**Request:**\n\n```\n{\n  "jsonrpc": "2.0",\n  "id": 2,\n  "method": "prompts/get",\n  "params": {\n    "name": "code_review",\n    "arguments": {\n      "code": "def hello():\\n    print(\'world\')"\n    }\n  }\n}\n```\n\n**Response:**\n\n```\n{\n  "jsonrpc": "2.0",\n  "id": 2,\n  "result": {\n    "description": "Code review prompt",\n    "messages": [\n      {\n        "role": "user",\n        "content": {\n          "type": "text",\n          "text": "Please review this Python code:\\ndef hello():\\n    print(\'world\')"\n        }\n      }\n    ]\n  }\n}\n```\n\n### List Changed Notification\n\nWhen the list of available prompts changes, servers that declared the `listChanged`\ncapability **SHOULD** send a notification:\n\n```\n{\n  "jsonrpc": "2.0",\n  "method": "notifications/prompts/list_changed"\n}\n```\n\n## Data Types\n\n### Prompt\n\nA prompt definition includes:\n\n- `name`: Unique identifier for the prompt\n- `title`: Optional human-readable name of the prompt for display purposes.\n- `description`: Optional human-readable description\n- `arguments`: Optional list of arguments for customization\n\n### PromptMessage\n\nMessages in a prompt can contain:\n\n- `role`: Either “user” or “assistant” to indicate the speaker\n- `content`: One of the following content types:\n\nAll content types in prompt messages support optional\n[annotations](https://modelcontextprotocol.io/specification/2025-06-18/server/resources#annotations) for\nmetadata about audience, priority, and modification times.\n\n#### Text Content\n\nText content represents plain text messages:\n\n```\n{\n  "type": "text",\n  "text": "The text content of the message"\n}\n```\n\nThis is the most common content type used for natural language interactions.\n\n#### Image Content\n\nImage content allows including visual information in messages:\n\n```\n{\n  "type": "image",\n  "data": "base64-encoded-image-data",\n  "mimeType": "image/png"\n}\n```\n\nThe image data **MUST** be base64-encoded and include a valid MIME type. This enables\nmulti-modal interactions where visual context is important.\n\n#### Audio Content\n\nAudio content allows including audio information in messages:\n\n```\n{\n  "type": "audio",\n  "data": "base64-encoded-audio-data",\n  "mimeType": "audio/wav"\n}\n```\n\nThe audio data MUST be base64-encoded and include a valid MIME type. This enables\nmulti-modal interactions where audio context is important.\n\n#### Embedded Resources\n\nEmbedded resources allow referencing server-side resources directly in messages:\n\n```\n{\n  "type": "resource",\n  "resource": {\n    "uri": "resource://example",\n    "mimeType": "text/plain",\n    "text": "Resource content"\n  }\n}\n```\n\nResources can contain either text or binary (blob) data and **MUST** include:\n\n- A valid resource URI\n- The appropriate MIME type\n- Either text content or base64-encoded blob data\n\nEmbedded resources enable prompts to seamlessly incorporate server-managed content like\ndocumentation, code samples, or other reference materials directly into the conversation\nflow.\n\n## Error Handling\n\nServers **SHOULD** return standard JSON-RPC errors for common failure cases:\n\n- Invalid prompt name: `-32602` (Invalid params)\n- Missing required arguments: `-32602` (Invalid params)\n- Internal errors: `-32603` (Internal error)\n\n## Implementation Considerations\n\n1. Servers **SHOULD** validate prompt arguments before processing\n2. Clients **SHOULD** handle pagination for large prompt lists\n3. Both parties **SHOULD** respect capability negotiation\n\n## Security\n\nImplementations **MUST** carefully validate all prompt inputs and outputs to prevent\ninjection attacks or unauthorized access to resources.', 'extras': {'signature': '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'}}

</details>

<details>
<summary>resources-model-context-protocol</summary>

{'type': 'text', 'text': 'The Model Context Protocol (MCP) provides a standardized way for servers to expose\nresources to clients. Resources allow servers to share data that provides context to\nlanguage models, such as files, database schemas, or application-specific information.\nEach resource is uniquely identified by a\n[URI](https://datatracker.ietf.org/doc/html/rfc3986).\n\n## [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#user-interaction-model)  User Interaction Model\n\nResources in MCP are designed to be **application-driven**, with host applications\ndetermining how to incorporate context based on their needs.For example, applications could:\n\n- Expose resources through UI elements for explicit selection, in a tree or list view\n- Allow the user to search through and filter available resources\n- Implement automatic context inclusion, based on heuristics or the AI model’s selection\n\nhttps://mintcdn.com/mcp/4ZXF1PrDkEaJvXpn/specification/2025-06-18/server/resource-picker.png?fit=max&auto=format&n=4ZXF1PrDkEaJvXpn&q=85&s=133fa885ef6e9c2e20049da5c33f4386However, implementations are free to expose resources through any interface pattern that\nsuits their needs—the protocol itself does not mandate any specific user\ninteraction model.\n\n## [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#capabilities)  Capabilities\n\nServers that support resources **MUST** declare the `resources` capability:\n\n```\n{\n  "capabilities": {\n    "resources": {\n      "subscribe": true,\n      "listChanged": true\n    }\n  }\n}\n```\n\nThe capability supports two optional features:\n\n- `subscribe`: whether the client can subscribe to be notified of changes to individual\nresources.\n- `listChanged`: whether the server will emit notifications when the list of available\nresources changes.\n\nBoth `subscribe` and `listChanged` are optional—servers can support neither,\neither, or both:\n\n```\n{\n  "capabilities": {\n    "resources": {} // Neither feature supported\n  }\n}\n```\n\n```\n{\n  "capabilities": {\n    "resources": {\n      "subscribe": true // Only subscriptions supported\n    }\n  }\n}\n```\n\n```\n{\n  "capabilities": {\n    "resources": {\n      "listChanged": true // Only list change notifications supported\n    }\n  }\n}\n```\n\n## [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#protocol-messages)  Protocol Messages\n\n### [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#listing-resources)  Listing Resources\n\nTo discover available resources, clients send a `resources/list` request. This operation\nsupports [pagination](https://modelcontextprotocol.io/specification/2025-06-18/server/utilities/pagination).**Request:**\n\n```\n{\n  "jsonrpc": "2.0",\n  "id": 1,\n  "method": "resources/list",\n  "params": {\n    "cursor": "optional-cursor-value"\n  }\n}\n```\n\n**Response:**\n\n```\n{\n  "jsonrpc": "2.0",\n  "id": 1,\n  "result": {\n    "resources": [\n      {\n        "uri": "file:///project/src/main.rs",\n        "name": "main.rs",\n        "title": "Rust Software Application Main File",\n        "description": "Primary application entry point",\n        "mimeType": "text/x-rust"\n      }\n    ],\n    "nextCursor": "next-page-cursor"\n  }\n}\n```\n\n### [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#reading-resources)  Reading Resources\n\nTo retrieve resource contents, clients send a `resources/read` request:**Request:**\n\n```\n{\n  "jsonrpc": "2.0",\n  "id": 2,\n  "method": "resources/read",\n  "params": {\n    "uri": "file:///project/src/main.rs"\n  }\n}\n```\n\n**Response:**\n\n```\n{\n  "jsonrpc": "2.0",\n  "id": 2,\n  "result": {\n    "contents": [\n      {\n        "uri": "file:///project/src/main.rs",\n        "mimeType": "text/x-rust",\n        "text": "fn main() {\\n    println!(\\"Hello world!\\");\\n}"\n      }\n    ]\n  }\n}\n```\n\n### [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#resource-templates)  Resource Templates\n\nResource templates allow servers to expose parameterized resources using\n[URI templates](https://datatracker.ietf.org/doc/html/rfc6570). Arguments may be\nauto-completed through [the completion API](https://modelcontextprotocol.io/specification/2025-06-18/server/utilities/completion).\nThis operation supports [pagination](https://modelcontextprotocol.io/specification/2025-06-18/server/utilities/pagination).**Request:**\n\n```\n{\n  "jsonrpc": "2.0",\n  "id": 3,\n  "method": "resources/templates/list",\n  "params": {\n    "cursor": "optional-cursor-value"\n  }\n}\n```\n\n**Response:**\n\n```\n{\n  "jsonrpc": "2.0",\n  "id": 3,\n  "result": {\n    "resourceTemplates": [\n      {\n        "uriTemplate": "file:///{path}",\n        "name": "Project Files",\n        "title": "📁 Project Files",\n        "description": "Access files in the project directory",\n        "mimeType": "application/octet-stream"\n      }\n    ],\n    "nextCursor": "next-page-cursor"\n  }\n}\n```\n\n### [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#list-changed-notification)  List Changed Notification\n\nWhen the list of available resources changes, servers that declared the `listChanged`\ncapability **SHOULD** send a notification:\n\n```\n{\n  "jsonrpc": "2.0",\n  "method": "notifications/resources/list_changed"\n}\n```\n\n### [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#subscriptions)  Subscriptions\n\nThe protocol supports optional subscriptions to resource changes. Clients can subscribe\nto specific resources and receive notifications when they change:**Subscribe Request:**\n\n```\n{\n  "jsonrpc": "2.0",\n  "id": 4,\n  "method": "resources/subscribe",\n  "params": {\n    "uri": "file:///project/src/main.rs"\n  }\n}\n```\n\n**Update Notification:**\n\n```\n{\n  "jsonrpc": "2.0",\n  "method": "notifications/resources/updated",\n  "params": {\n    "uri": "file:///project/src/main.rs"\n  }\n}\n```\n\n## [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#message-flow)  Message Flow\n\nServerClientServerClientResource DiscoveryResource Template DiscoveryResource AccessSubscriptionsUpdatesresources/listList of resourcesresources/templates/listList of resource templatesresources/readResource contentsresources/subscribeSubscription confirmednotifications/resources/updatedresources/readUpdated contents\n\n## [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#data-types)  Data Types\n\n### [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#resource)  Resource\n\nA resource definition includes:\n\n- `uri`: Unique identifier for the resource\n- `name`: The name of the resource.\n- `title`: Optional human-readable name of the resource for display purposes.\n- `description`: Optional description\n- `mimeType`: Optional MIME type\n- `size`: Optional size in bytes\n\n### [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#resource-contents)  Resource Contents\n\nResources can contain either text or binary data:\n\n#### [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#text-content)  Text Content\n\n```\n{\n  "uri": "file:///example.txt",\n  "mimeType": "text/plain",\n  "text": "Resource content"\n}\n```\n\n#### [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#binary-content)  Binary Content\n\n```\n{\n  "uri": "file:///example.png",\n  "mimeType": "image/png",\n  "blob": "base64-encoded-data"\n}\n```\n\n### [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#annotations)  Annotations\n\nResources, resource templates and content blocks support optional annotations that provide hints to clients about how to use or display the resource:\n\n- **`audience`**: An array indicating the intended audience(s) for this resource. Valid values are `"user"` and `"assistant"`. For example, `["user", "assistant"]` indicates content useful for both.\n- **`priority`**: A number from 0.0 to 1.0 indicating the importance of this resource. A value of 1 means “most important” (effectively required), while 0 means “least important” (entirely optional).\n- **`lastModified`**: An ISO 8601 formatted timestamp indicating when the resource was last modified (e.g., `"2025-01-12T15:00:58Z"`).\n\nExample resource with annotations:\n\n```\n{\n  "uri": "file:///project/README.md",\n  "name": "README.md",\n  "title": "Project Documentation",\n  "mimeType": "text/markdown",\n  "annotations": {\n    "audience": ["user"],\n    "priority": 0.8,\n    "lastModified": "2025-01-12T15:00:58Z"\n  }\n}\n```\n\nClients can use these annotations to:\n\n- Filter resources based on their intended audience\n- Prioritize which resources to include in context\n- Display modification times or sort by recency\n\n## [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#common-uri-schemes)  Common URI Schemes\n\nThe protocol defines several standard URI schemes. This list not\nexhaustive—implementations are always free to use additional, custom URI schemes.\n\n### [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#https//)  https://\n\nUsed to represent a resource available on the web.Servers **SHOULD** use this scheme only when the client is able to fetch and load the\nresource directly from the web on its own—that is, it doesn’t need to read the resource\nvia the MCP server.For other use cases, servers **SHOULD** prefer to use another URI scheme, or define a\ncustom one, even if the server will itself be downloading resource contents over the\ninternet.\n\n### [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#file//)  file://\n\nUsed to identify resources that behave like a filesystem. However, the resources do not\nneed to map to an actual physical filesystem.MCP servers **MAY** identify file:// resources with an\n[XDG MIME type](https://specifications.freedesktop.org/shared-mime-info-spec/0.14/ar01s02.html#id-1.3.14),\nlike `inode/directory`, to represent non-regular files (such as directories) that don’t\notherwise have a standard MIME type.\n\n### [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#git//)  git://\n\nGit version control integration.\n\n### [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#custom-uri-schemes)  Custom URI Schemes\n\nCustom URI schemes **MUST** be in accordance with [RFC3986](https://datatracker.ietf.org/doc/html/rfc3986),\ntaking the above guidance in to account.\n\n## [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#error-handling)  Error Handling\n\nServers **SHOULD** return standard JSON-RPC errors for common failure cases:\n\n- Resource not found: `-32002`\n- Internal errors: `-32603`\n\nExample error:\n\n```\n{\n  "jsonrpc": "2.0",\n  "id": 5,\n  "error": {\n    "code": -32002,\n    "message": "Resource not found",\n    "data": {\n      "uri": "file:///nonexistent.txt"\n    }\n  }\n}\n```\n\n## [\u200b](https://modelcontextprotocol.io/specification/2025-06-18/server/resources\\#security-considerations)  Security Considerations\n\n1. Servers **MUST** validate all resource URIs\n2. Access controls **SHOULD** be implemented for sensitive resources\n3. Binary data **MUST** be properly encoded\n4. Resource permissions **SHOULD** be checked before operations', 'extras': {'signature': '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'}}

</details>

<details>
<summary>stop-vibe-testing-your-mcp-server</summary>

{'type': 'text', 'text': 'If you’re working with the Model Context Protocol (MCP), you’re on the front lines of AI innovation. But amidst the excitement of creating intelligent agents and sophisticated AI workflows, I need to ask: **how are you actually testing these critical MCP components?**\n\nToo often, the answer looks something like this: fire up an agent framework, type a few prompts into a chat window, and if the LLM _seems_ to produce a reasonable output, call it a day. This, my friends, is **vibe-testing.**\n\nTo be fair, this isn’t entirely surprising. The MCP ecosystem is young, and the developer tooling is still catching up to the rapid pace of protocol adoption. However, while vibe-testing might seem pragmatic given the tooling landscape, it’s a fast track to unreliable systems, wasted tokens, and downright painful debugging sessions.\n\nMCP servers are the APIs that connect LLMs to the real world. And like any critical API, they demand rigorous, deterministic testing to ensure they are reliable, predictable, and robust—especially when the primary consumer is a non-deterministic LLM.\n\n> A QA engineer walks into a bar. Orders a beer. Orders 0 beers. Orders 99999999999 beers. Orders a lizard. Orders -1 beers. Orders a ueicbksjdhd.\n>\n> First real customer walks in and asks where the bathroom is. The bar bursts into flames, killing everyone.\n>\n> — Brenan Keller (@brenankeller) [November 30, 2018](https://twitter.com/brenankeller/status/1068615953989087232?ref_src=twsrc%5Etfw)\n\nThis joke hits alarmingly close to home in the MCP world. Traditionally, QA engineers intentionally probe boundaries. With MCP, _your LLM client is a chaos agent._ LLMs can generate unexpected or malformed inputs, explore edge cases you never envisioned, or chain calls in ways that defy simple logic. If your MCP server isn’t hardened against this onslaught of creative inputs, it’s not a question of _if_ things will go sideways, but _when_ your proverbial bar bursts into flames, potentially on the most mundane of “customer” requests.\n\nThe core issue with relying on LLM-based “vibe-testing” is that it’s:\n\n- **Stochastic:** What works once might not work again. You cannot build reliable systems on a foundation of “maybe.”\n- **Slow & Expensive:** Each “test” involves LLM interactions, racking up latency and API costs. A proper test suite should be efficient.\n- **Opaque:** When something breaks, pinpointing the cause—is it your server, the LLM’s interpretation, the agent framework, or the prompt?—becomes a frustrating detective game.\n- **Superficial:** Natural language interactions rarely achieve the comprehensive coverage needed to find subtle bugs or validate all edge cases.\n\nIt’s imperative that your server’s logic is either impeccably clear or that its error messages are so precise they can effectively guide an LLM back on track. Neither of these is achievable without rigorous, focused testing. While iterating on your instructions to help LLMs “do the right thing” is valuable, robust server-side logic and error handling are non-negotiable.\n\n### Testing is Trust (and Good Engineering)\n\nI was incredibly fortunate to start Prefect alongside Chris White, who instilled in me a deep appreciation for the true value of testing. Proper testing serves a deeper purpose than merely affirming your code runs; it’s a fundamental practice for **documenting behavior**, **preventing regressions**, and building **deep trust** in your codebase.\n\nChris’s philosophy, which we can bring to bear here, emphasizes that:\n\n- Unit tests should be _atomic_, targeting the smallest possible unit of behavior.\n- _Tests and design go hand-in-hand:_ if something is hard to test, its design might be flawed. Test-driven development can be particularly effective when defining new user-facing contracts.\n- Tests must _clearly document_ the behavior and expectations that are important to your application. A failing test’s _title alone_ should strongly indicate what’s broken.\n- Tests should verify _expected, assertable behavior_, rather than being tightly coupled to specific implementation details. This allows for refactoring with confidence.\n- Critically, tests should _not unnecessarily block future paths_ or refactors. They guard core contracts, not incidental details, fostering an environment built for change.\n\nThis philosophy is about creating a safety net that allows for rapid iteration and confident development. When your MCP server is the component bridging the deterministic world of your code with the probabilistic world of LLMs, this trust and safety net become absolutely paramount.\n\n### In-Memory Testing with FastMCP\n\nFastMCP 2.0 was designed to make rigorous testing easy, not an afterthought. The key to this is FastMCP’s support for **in-memory testing.**\n\nWith FastMCP, you can instantiate a `fastmcp.Client` and connect it _directly_ to your `FastMCP` server instance by providing the server as the client’s transport target:\n\n```\nfrom fastmcp import FastMCP, Client\n\nmcp = FastMCP(name="My MCP Server")\n\n@mcp.tool()\n\ndef add(a: int, b: int) -> int:\n\n    return a + b\n\ntest_client = Client(mcp) # Connects the client directly to the server instance\n```\n\nThis direct, in-memory connection is a game-changer for testing MCP servers because:\n\n- 💨 **There’s no network overhead:** Communication is as fast as a direct Python call.\n- 🧘 **No subprocess management is needed:** You don’t have to start and stop external server processes for your tests.\n- 🎯 **You’re testing your actual server logic:** No mocks or simplified protocol implementations are needed; this uses the real STDIO transport internally for maximum fidelity.\n\nOnce you have this `test_client`, you can use its methods to interact with your server just like an LLM, but with the benefit of repeatable determinism and low latency. For example, within an `async with test_client:` block, you can:\n\n- Ping the server: `is_alive = await test_client.ping()`\n- List available tools: `tools = await test_client.list_tools()`\n- Call a specific tool: `response = await test_client.call_tool("add", {"a": 1, "b": 2})`\n- Read a resource: `content = await test_client.read_resource("resource://your/data")`\n\n…and more, including advanced MCP features like logging, progress reporting, and LLM client sampling. Please review FastMCP’s client docs for more details.\n\nThis direct, in-memory connection is a game-changer for testing MCP servers because it means your tests are not just validating isolated functions; they’re confirming your server’s behavior through the actual MCP interaction layer, albeit without network latency.\n\nThe result? Your tests become:\n\n- ⚡ **Blazingly Fast:** Run them as part of your normal `pytest` suite in milliseconds.\n- 🧪 **Deterministic:** Get consistent, repeatable results every single time.\n- 🎯 **Focused:** Isolate and test your server’s tool, resource, and prompt logic precisely.\n- 🐍 **Pythonic:** Write your tests using the testing tools and patterns you already know and love.\n\nYou’ll find yourself writing _more_ tests, not fewer, because testing your MCP functionality becomes as quick and easy as testing any other Python function. Since everything runs in-process, you can use mocks, fixtures, and other familiar testing tools without hesitation.\n\nHere’s how you can structure your tests using `pytest`:\n\n```\nimport pytest\n\nfrom fastmcp import FastMCP, Client\n\nfrom mcp.types import TextContent # For type checking results\n\n# A reusable fixture for our MCP server\n\n@pytest.fixture\n\ndef mcp_server():\n\n    mcp = FastMCP(name="CalculationServer")\n\n    @mcp.tool()\n\n    def add(a: int, b: int) -> int:\n\n        return a + b\n\n    return mcp\n\n# A straightforward test of our tool\n\nasync def test_add_tool(mcp_server: FastMCP):\n\n    async with Client(mcp_server) as client: # Client uses the mcp_server instance\n\n        result = await client.call_tool("add", {"a": 1, "b": 2})\n\n        assert isinstance(result[0], TextContent)\n\n        assert result[0].text == "3"\n```\n\n**Nerd note:** we did not put the client in a fixture, like this:\n\n```\n# Don\'t do this!\n\n@pytest.fixture\n\nasync def client(mcp_server: FastMCP):\n\n    async with Client(mcp_server) as client:\n\n        yield client\n```\n\nThat’s because `pytest`’s async fixtures and tests can run in **different event loops**. This can lead to runtime errors related to task cancellation when the `Client`’s `async with` block (which manages an `anyio` task group from the underlying MCP SDK) spans across these different loops. Instantiating the client directly within the test function ensures it operates within the test’s event loop.\n\nThis robust approach allows you to comprehensively test:\n\n- Correct tool logic for a wide range of valid inputs (your “lizard” cases!).\n- Graceful error handling for invalid inputs or internal server exceptions.\n- Accurate content delivery for your static resources and dynamic resource templates.\n- Correct rendering of prompts with various parameter combinations.\n- Complex interactions involving the `Context` object, such as logging, progress reporting, and inter-resource data access.\n\nInstead of merely hoping your LLM client interprets things correctly, you are _asserting_ that your server behaves exactly as designed under a multitude of conditions.\n\n### Beyond FastMCP: Testing Any MCP Server\n\nThe `fastmcp.Client` isn’t limited to in-memory testing of FastMCP servers you built yourself. It’s a versatile tool for interacting with _any_ MCP-compliant server. This means you can write expansive tests for any MCP behavior you want to ensure is reliable and consistent, regardless of the server’s implementation.\n\nIn addition to supplying the client with an explicit transport configuration (like `StdioTransport` or `StreamableHttpTransport`), you can often rely on its ability to automatically infer the appropriate transport based on the URL or command string you provide. In the following example, all client objects expose the exact same interface for testing, regardless of how they are instantiated:\n\n```\nfrom fastmcp import Client\n\n# A remote server\n\nasync def test_remote_mcp_server():\n\n    async with Client("http://some.api.service/mcp_endpoint") as client:\n\n        await client.call_tool("some_tool", {"key": "value"})\n\n# A local Node.js server script\n\nasync def test_local_js_server():\n\n    async with Client(\'path/to/local/server.js\') as client:\n\n        await client.read_resource("resource://path/to/resource")\n\n# Two remote servers configured via an MCP config into a FastMCP proxy server\n\nasync def test_mcp_config_server():\n\n    mcp_config = {\n\n        \'mcpServers\': {\n\n            "github": {\n\n                "command": "npx",\n\n                "args": ["-y", "@modelcontextprotocol/server-github"],\n\n                "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "<YOUR_TOKEN>"}\n\n            },\n\n            \'paypal\': {\'url\': \'https://mcp.paypal.com/sse\'}\n\n        }\n\n    }\n\n    # The client will infer to create a FastMCPProxy for this config\n\n    async with Client(mcp_config) as client:\n\n        await client.call_tool("github_get_user_repos", {"username": "jlowin"})\n```\n\nYour MCP servers form a critical layer in your AI stack. They are the deterministic bedrock upon which the more unpredictable LLM interactions are built. If this foundation is unreliable, your entire AI application becomes fragile.\n\nFastMCP’s testing capabilities, especially its in-memory testing, are designed to help you build this foundation with confidence and rigor. Stop relying on “looks good to me” vibe-checks through a chat window. Start writing focused, repeatable tests that prove your server does exactly what it’s supposed to do.\n\nYour AI, your users, and your sanity will thank you for it.', 'extras': {'signature': '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'}}

</details>

<details>
<summary>tools-model-context-protocol</summary>

{'type': 'text', 'text': 'The Model Context Protocol (MCP) allows servers to expose tools that can be invoked by\nlanguage models. Tools enable models to interact with external systems, such as querying\ndatabases, calling APIs, or performing computations. Each tool is uniquely identified by\na name and includes metadata describing its schema.\n\n## User Interaction Model\n\nTools in MCP are designed to be **model-controlled**, meaning that the language model can\ndiscover and invoke tools automatically based on its contextual understanding and the\nuser’s prompts.However, implementations are free to expose tools through any interface pattern that\nsuits their needs—the protocol itself does not mandate any specific user\ninteraction model.\n\nFor trust & safety and security, there **SHOULD** always\nbe a human in the loop with the ability to deny tool invocations.Applications **SHOULD**:\n\n- Provide UI that makes clear which tools are being exposed to the AI model\n- Insert clear visual indicators when tools are invoked\n- Present confirmation prompts to the user for operations, to ensure a human is in the\nloop\n\n## Capabilities\n\nServers that support tools **MUST** declare the `tools` capability:\n\n```\n{\n  "capabilities": {\n    "tools": {\n      "listChanged": true\n    }\n  }\n}\n```\n\n`listChanged` indicates whether the server will emit notifications when the list of\navailable tools changes.\n\n## Protocol Messages\n\n### Listing Tools\n\nTo discover available tools, clients send a `tools/list` request. This operation supports\n[pagination](https://modelcontextprotocol.io/specification/2025-06-18/server/utilities/pagination).**Request:**\n\n```\n{\n  "jsonrpc": "2.0",\n  "id": 1,\n  "method": "tools/list",\n  "params": {\n    "cursor": "optional-cursor-value"\n  }\n}\n```\n\n**Response:**\n\n```\n{\n  "jsonrpc": "2.0",\n  "id": 1,\n  "result": {\n    "tools": [\n      {\n        "name": "get_weather",\n        "title": "Weather Information Provider",\n        "description": "Get current weather information for a location",\n        "inputSchema": {\n          "type": "object",\n          "properties": {\n            "location": {\n              "type": "string",\n              "description": "City name or zip code"\n            }\n          },\n          "required": ["location"]\n        }\n      }\n    ],\n    "nextCursor": "next-page-cursor"\n  }\n}\n```\n\n### Calling Tools\n\nTo invoke a tool, clients send a `tools/call` request:**Request:**\n\n```\n{\n  "jsonrpc": "2.0",\n  "id": 2,\n  "method": "tools/call",\n  "params": {\n    "name": "get_weather",\n    "arguments": {\n      "location": "New York"\n    }\n  }\n}\n```\n\n**Response:**\n\n```\n{\n  "jsonrpc": "2.0",\n  "id": 2,\n  "result": {\n    "content": [\n      {\n        "type": "text",\n        "text": "Current weather in New York:\\nTemperature: 72°F\\nConditions: Partly cloudy"\n      }\n    ],\n    "isError": false\n  }\n}\n```\n\n### List Changed Notification\n\nWhen the list of available tools changes, servers that declared the `listChanged`\ncapability **SHOULD** send a notification:\n\n```\n{\n  "jsonrpc": "2.0",\n  "method": "notifications/tools/list_changed"\n}\n```\n\n## Message Flow\n\nServerClientLLMServerClientLLMDiscoveryTool SelectionInvocationUpdatestools/listList of toolsSelect tool to usetools/callTool resultProcess resulttools/list_changedtools/listUpdated tools\n\n## Data Types\n\n### Tool\n\nA tool definition includes:\n\n- `name`: Unique identifier for the tool\n- `title`: Optional human-readable name of the tool for display purposes.\n- `description`: Human-readable description of functionality\n- `inputSchema`: JSON Schema defining expected parameters\n- `outputSchema`: Optional JSON Schema defining expected output structure\n- `annotations`: optional properties describing tool behavior\n\nFor trust & safety and security, clients **MUST** consider\ntool annotations to be untrusted unless they come from trusted servers.\n\n### Tool Result\n\nTool results may contain [**structured**](https://modelcontextprotocol.io/specification/2025-06-18/server/tools#structured-content) or **unstructured** content.**Unstructured** content is returned in the `content` field of a result, and can contain multiple content items of different types:\n\nAll content types (text, image, audio, resource links, and embedded resources)\nsupport optional\n[annotations](https://modelcontextprotocol.io/specification/2025-06-18/server/resources#annotations) that\nprovide metadata about audience, priority, and modification times. This is the\nsame annotation format used by resources and prompts.\n\n#### Text Content\n\n```\n{\n  "type": "text",\n  "text": "Tool result text"\n}\n```\n\n#### Image Content\n\n```\n{\n  "type": "image",\n  "data": "base64-encoded-data",\n  "mimeType": "image/png"\n  "annotations": {\n    "audience": ["user"],\n    "priority": 0.9\n  }\n\n}\n```\n\nThis example demonstrates the use of an optional Annotation.\n\n#### Audio Content\n\n```\n{\n  "type": "audio",\n  "data": "base64-encoded-audio-data",\n  "mimeType": "audio/wav"\n}\n```\n\n#### Resource Links\n\nA tool **MAY** return links to [Resources](https://modelcontextprotocol.io/specification/2025-06-18/server/resources), to provide additional context\nor data. In this case, the tool will return a URI that can be subscribed to or fetched by the client:\n\n```\n{\n  "type": "resource_link",\n  "uri": "file:///project/src/main.rs",\n  "name": "main.rs",\n  "description": "Primary application entry point",\n  "mimeType": "text/x-rust",\n  "annotations": {\n    "audience": ["assistant"],\n    "priority": 0.9\n  }\n}\n```\n\nResource links support the same [Resource annotations](https://modelcontextprotocol.io/specification/2025-06-18/server/resources#annotations) as regular resources to help clients understand how to use them.\n\nResource links returned by tools are not guaranteed to appear in the results\nof a `resources/list` request.\n\n#### Embedded Resources\n\n[Resources](https://modelcontextprotocol.io/specification/2025-06-18/server/resources) **MAY** be embedded to provide additional context\nor data using a suitable [URI scheme](https://modelcontextprotocol.io/specification/2025-06-18/server/resources#common-uri-schemes). Servers that use embedded resources **SHOULD** implement the `resources` capability:\n\n```\n{\n  "type": "resource",\n  "resource": {\n    "uri": "file:///project/src/main.rs",\n    "mimeType": "text/x-rust",\n    "text": "fn main() {\\n    println!(\\"Hello world!\\");\\n}",\n    "annotations": {\n      "audience": ["user", "assistant"],\n      "priority": 0.7,\n      "lastModified": "2025-05-03T14:30:00Z"\n    }\n  }\n}\n```\n\nEmbedded resources support the same [Resource annotations](https://modelcontextprotocol.io/specification/2025-06-18/server/resources#annotations) as regular resources to help clients understand how to use them.\n\n#### Structured Content\n\n**Structured** content is returned as a JSON object in the `structuredContent` field of a result.For backwards compatibility, a tool that returns structured content SHOULD also return the serialized JSON in a TextContent block.\n\n#### Output Schema\n\nTools may also provide an output schema for validation of structured results.\nIf an output schema is provided:\n\n- Servers **MUST** provide structured results that conform to this schema.\n- Clients **SHOULD** validate structured results against this schema.\n\nExample tool with output schema:\n\n```\n{\n  "name": "get_weather_data",\n  "title": "Weather Data Retriever",\n  "description": "Get current weather data for a location",\n  "inputSchema": {\n    "type": "object",\n    "properties": {\n      "location": {\n        "type": "string",\n        "description": "City name or zip code"\n      }\n    },\n    "required": ["location"]\n  },\n  "outputSchema": {\n    "type": "object",\n    "properties": {\n      "temperature": {\n        "type": "number",\n        "description": "Temperature in celsius"\n      },\n      "conditions": {\n        "type": "string",\n        "description": "Weather conditions description"\n      },\n      "humidity": {\n        "type": "number",\n        "description": "Humidity percentage"\n      }\n    },\n    "required": ["temperature", "conditions", "humidity"]\n  }\n}\n```\n\nExample valid response for this tool:\n\n```\n{\n  "jsonrpc": "2.0",\n  "id": 5,\n  "result": {\n    "content": [\n      {\n        "type": "text",\n        "text": "{\\"temperature\\": 22.5, \\"conditions\\": \\"Partly cloudy\\", \\"humidity\\": 65}"\n      }\n    ],\n    "structuredContent": {\n      "temperature": 22.5,\n      "conditions": "Partly cloudy",\n      "humidity": 65\n    }\n  }\n}\n```\n\nProviding an output schema helps clients and LLMs understand and properly handle structured tool outputs by:\n\n- Enabling strict schema validation of responses\n- Providing type information for better integration with programming languages\n- Guiding clients and LLMs to properly parse and utilize the returned data\n- Supporting better documentation and developer experience\n\n## Error Handling\n\nTools use two error reporting mechanisms:\n\n1. **Protocol Errors**: Standard JSON-RPC errors for issues like:   - Unknown tools\n   - Invalid arguments\n   - Server errors\n2. **Tool Execution Errors**: Reported in tool results with `isError: true`:   - API failures\n   - Invalid input data\n   - Business logic errors\n\nExample protocol error:\n\n```\n{\n  "jsonrpc": "2.0",\n  "id": 3,\n  "error": {\n    "code": -32602,\n    "message": "Unknown tool: invalid_tool_name"\n  }\n}\n```\n\nExample tool execution error:\n\n```\n{\n  "jsonrpc": "2.0",\n  "id": 4,\n  "result": {\n    "content": [\n      {\n        "type": "text",\n        "text": "Failed to fetch weather data: API rate limit exceeded"\n      }\n    ],\n    "isError": true\n  }\n}\n```\n\n## Security Considerations\n\n1. Servers **MUST**:   - Validate all tool inputs\n   - Implement proper access controls\n   - Rate limit tool invocations\n   - Sanitize tool outputs\n2. Clients **SHOULD**:   - Prompt for user confirmation on sensitive operations\n   - Show tool inputs to the user before calling the server, to avoid malicious or\n     accidental data exfiltration\n   - Validate tool results before passing to LLM\n   - Implement timeouts for tool calls\n   - Log tool usage for audit purposes', 'extras': {'signature': 'EoklCoYlAQw51seBN3O/FuQ4sf2xMI32gmseV3f48g/1IdKKJDbxygu6S5tcCHHiuAosN6Br4xZhwGH64WpAz3+sVLau8YF1vgwXl8FGiqKxKGvani7DaQtGYbIYhgJ0bwMQcQ5XmpcrHc9FNHdhdHuHpjafXnY8aAdalFYrYywJ8mVu5mcy8tzHJPK/klJq5xNg5M+PQiSC8nxnk9k9vWelddAGgjvuRCIFftpfgz3Z6sny7siEhMCjSsdahlevlgPPHk8xes6x1UWB6FM6Sv1bK21pUskciZBULZ8xnDg1dr3v6X49pHo/sH2HQiQBq51G+zTMEajqrONLHHVtwWsz0g9iEvu8GjCEWpX34hXfc/KoMKf3BjwJULjX2Uy+xMIHFCuzWLSCF4xg0x958JGatsfhzwCZgofGqAhnnUdkZvB/8ANXY/q+9Y5mtJvW/K+6xFEeHy/lY6TDi2ffPnl7AfyLLQ/05nimFiS4ZFIZvGp+nlhFqKqIHpSQR3aMdfXkiJPnnEyYNaLUXa3mw0u07mGZHqEBjd8nHW50pOcZHouuS5as7cN7Mf4qR0F/oXeiejZTUeZj87bxH1ka3HNwL8S072rldJJTYSln8Eu+4K9fIOHdRP55E9/pA4nw14TSH81nefamhrOzp+mOWqWWNbzTGD7nPHlqo+UaGU1zYw7C7ff7+0RO/D/EKydPsxIF2WUXJ5LTtRwsqqLEtNMqLVtbeB/8khgmsi5oV5+GKut3nf3KPkrMsEivzWS7LOt5AS7jQX9WuwH6JtZbdt2hoMp/juETBGrUGITsDjyJtpmfAm6a0/Gcy4KpbYXciGdBtERnKRdQcstXCcYEuLM4zUANLZk47Y5+f3CMczzsqb3xo58sEzz6TjEKC2Fajkw+Wdt12eVcpnpYk0u8lwHEMijH+sD8E9IYzk/yels1J+DXa4Rc+xfrxTTca2Ib+JQW/+JdbmmBrCEHj3QM0f49CEesnys6k5tCPXWqXpf9h1Txhh+84oA9sstt6pZZj4vjZZ+EFqq57duN6Y46MZ+39e/g5/voUajTCJoKXVvuBvfTRCMcQ3O908tE0laZv2pxSVeyawiBypcEPJHJKO5TOrxZhGhSrqmjH0c690ltNoZvxHskNGbM6/JitSqjS+3oYvY8abJtxaVqVuF5rb8ktgn2KbxNx414uZSwzvPm9KMbWuikPma5poReLYRfOLz/JtXgXEHFfuZxuIW+QGc5uMd6vFLsOWyOPl2gvro4il+dzng1LuGtBlF1QVwDhXtKm097H6lCsXjRy5Xv3RYwis4YvBfQtBs8hC9T6rtMhFox6YsVAiRsCxbwSArxzCuA+dx2xGNUXdeMnM9cGc+WS9NK5bBpgeqzSQS9jrzEFnNCnP+vyfnK0lLhPj4VOk5LNf63ll9PSt+6RzNvo96a4sNc4aNLWsi4J/t0ac5TUpFbe2j2vNGlLUtSBiW8xLCJVNtm0TOoHKi1zrHAM3FPGBBw7TL/5GK+4Dj8uggcyXCYH43PykKwtv55UUQ3bB2qmufEJC74nv9ZRizvMKAA1pSNsMESu8pueBbezo6r4LmmSsYVQpmPBiev7akmAYR/Jc93ypFJv0B4t3q57V3M5LggwdXI6dW8SUhooNIEBQgdsKPvoqKMF8sy2ZzeCCVHHBbWKZ+VMVjlcqvPIEVgWBo9i2lynOUbk9/qTRKWg4gqeiXwo8tAme1gOTDbrIktpXDdxXfGmfwt7mxFd+xJ+VgRyRz3dWSFLTUNV5gXHkEGlG1CDppWZRMzsUkCdSnlFTyFhBp2SpQ1qV9zw/nygi/Sn9DUEfTBKu2nXrUtZrLPD1EHhX3s+lhRckqrRhnBKhwxrnoJDk4gbQUP2niWwykepQ2cFHcCLI38refc5m6JPP6pL2EubxMCQEiqJX7xYrOy7W0sacauXuIE8DYL9xHlTugTdUYUaCFsE2oMnyuFo3nrW9OAdi2FzHzA6eNSUGB5GQVf8mxHDJpf4sLuYd2JyFF7VGcChq3VQKGL8lGK0WLD11X+y8ypjbZTBqAfK6ZrdkLHXo1qUOW0VazZd7z4hRFTgbGQGBIEJUsbTUrjO4zr9DPHbhvGE/H6QK8qSCOSnkYt/DvvrMSXHnZPABqxMgp4pCRtKWlH4KXTJH5KCrXqxoHDg+Yz6cNJZv5oaQ674F+8wKyW2kbIBb3nrtPDMwT8bvAO6pHAr07CsIBJx9gyKI0YrcNKoL0vpyWN//0U4jch5z59vmqCO5yux970hG8FW9BcNEcNsjn1S37ac7kk52WKNSlFuHWj+vuHXR4rflllbznX1hNaPLKaPHiv7CB76b6BR0m5nrROBYhproQZWw5K4kFEWCcyeI/9ZAQ0z62joOA36W+bxVtd/CJp+I7FeOcSPOCfEMNzxS5r+gbi3b2oxB2uhJtbKO1nHsV3iiGCiGNIz3obaczY6ayxJ48VETEugjV+vAUeffEzeNk0/4KzmJQWviyqedOFHOws0cKMUU/hJB8e3TIt0t6FZ699HpmWFzz+fWbTH5RS08+UavOypmge52JmYjKNitV9F1tSKkGG9s8edrKBFwusW6G31LTVT4d0Wf1yswHgismcUnIbOU49J5jhQYDOfr4ELeInDQrPCbK4mwEvksf+5BXx3z4DN4EbCAXWgEKn7tK8vil3+MGKwgy1NwF3veNmjjCyimxTlXQgS87WhVcM7WfisDeR63VkxyqiU3rVmYN/7ewSfCNkGh4pFJqoYiH9j455bXgkKrd17/dMjznGlGVs9KCeYsYl7CqhGZu5ABYAALm1MtYpCBzkbEOukG1vVBOOEJqS84dXAwpEVfztH6XEc0GJcSyIo1XDc0iJo7HkW0rXbXDZXgMU9JQ3l5NeIhYPeA/x8tNxm7pwjTXlPde/yI3t9vTyFL4xj1NC9NxIpXDFpOcR/gE1zNCx5AzxxPi0Ah8JYK2e26Ljqq+U4ehMQhH8AZZxMYDLhDvUTGBwo4m9vFisnN5LsJPINYcJp/5E7gG46fnIidGMay3o1VYJBIlNrwDNnEcUrkkYHwnT/63lGl2OHVIO6V0o8mwa53Bbv2dhVcouK6LK2gmDvBATr/U3xu/7/tbZxemHBfFPVEmwhaMWnHRoJUVV/VXHvBpYNJ3SXLT6MNENISINBjGhLoUS7Nfgs3iwO8MZhSm01/7ByFzvtJ4cWB4VV+YFwvJyneNJNOEA53+mBbdfTGv1BXy3EXedAQVeXSaH1vDtXo321lJVwvpMMSYYQ9lGPwTl7L7mnn/t8rRpDH/pwUzGM/hvYYJS4EwmFcQov35vW3PPTb+NJ+ydNfck4qAivlWM8BQ3ocrZXLKfSk3UKP3Kuh/vwg2be4h8A7EqvUz2Ct0MrylEDKxdpIfQXTBxWQdI6LaYbv+7Jo2jjTKeTQjdMyvajRFzviGBL5b7kZrBuKlJe5PDl9a1FrbGsd7rReWbhJCI2MLpxmK2ek5iBM9J7uqvQg3B8EfwgZtaXc42kDwlxly4erAbw/fhBRdCX3cZfkdaxfI4A+WejO2FpRlHuX7RqAEOEUdjoj44PwPiF+u3E93cdbamesEy4dAF37REzv5dvs2t1KKf69xxQZeLCxXB4ETQjp5SiRrP6qLVXTf4vUdNX4iRQ3FxGfX+lBsipnG86rocXIyki0doIcuj8kB2zZRhxAK+iccJUBqEbYL0Rfq2PGvfl1PoUJACsPE59P0neYy+pvVtcd1Ay/UP1QJGnL7SQ6lPiwA5sU7KvBoD5rP9SRDr/SV0N37HqcpTK+HhjbhWLaaKvnEbSffL2cLs28EH6dafp5ao3lAZLWdxdoN39mMoZ/zXTnvMhy84Hjv2I9pnN1RCJaOSkz00n6u5HY6yhtmu17SgD23YFlRW5rDSsslgkoSv2HJXRaQPekB34D3V0m3+Fqsc8A3pwqNhkAKHjskGt6OTeoQomFD/2XWBlrZrDFrRPgaRtd81Y6lWDGLXkaRNNRVZWmy7yESY+v9x3SM+js4QqqwEmXl/MBBZuxQAPtpBbHfp0W9viwg1Sq5x9ei1D/zpxvtCKRF/YdujOKTDO52olt0ehK77myh/XvQWAVAB5XEI0uZFYcIsnmpJAY9NvLp6T0I+IV8OqMvb+AdbV0I2Y1ZVq9YQTmP8q9xjFbvBvOzSOKVStMvL8L2VL0CtH9h0DRGabNvuTpqW5CXHxhEeAXbRTg98mgeP4zZbZtPf+aAh7nmurSL9BdmtWTVrJh/4skgblxc2zBi16HLf1fgk/i9yf5wl6Tv+TpEtNL4kc5CN+YgYe4KNjM5T379rFez4Pk2do7t4i4QAehDIwzLWHwgULvQYk7CYdFxDtmEU0ZqwTSFoPLn+s9wM38+O4oZFwtQsrZwHi/0NOjJ0z/DPnkTtXve4AxSBNbPXiw3QypJd6KYE7ZeHXUK51NYro3jLflm7gsYmuiuHNKAZUr+e2h34jmlrE4ZY5q/dNCrqsRt0n/PO4hnS7C70CHcrhSijxZmhe03rmpPkEtyeZmGaXAFzBK7zqWT/0BvefDVt1BYniJkW1AJTO8HfvZm7v/BOT3lYBQ0I5/Q4C7wWpuEVTgMUD1VYt+X3tvK9kWd0k98dudy+gclnQ752+rmiqwvld+Rq7QS89X91q/Vu3pwGlNdEecgyWThrCGRrhOPoGdfwOYRFAdzRRJ4ZKs8H5X8mqSMvOYdZIJLJ3fE8QzfxaEZZULhPVspEl8g3awcARiuMYonXjJ8JMghBmcqD/ehz3vVJ67NWkJEXAOyxFhSCrPRzMdOLi+yqUPhXFPHr8rV+4cYWveaK62RJ+M8n0GPl+gQh3zMEvW89q1pjMU94EkEt/l4R/aJDT5M3mN6BhEulEiUmRr338BtmB7YboBEExKreGolcrHII6HKrB6sQWQOOoQwA2BMVipszfoFshyUK5NQTMqT7CAaSX+v227dm/k3p7ucen3mXwO7Xt3NeWpNavCGtGwqAzptjPfq60JZZln3mUhr5mw9ABAqcBpT6l4vUJWJqoIS9T4fagDWJ4ZJR5CRMwsK2pEFeyPJtLWK59J7s1xeKekly/Its5CWYkW2LFkS4PKLe0/BGFx5cNiqRXQ4BZDhdX1qJeR2QLZuDVxY3aHzXUVfiGotXDE1crCJn3LQTHwzNpyZ0grsEIzhGRE/hpx4vBmSUHl1L1wXdvW582UV9Jl6ucEG9Y5PVb9YSUunHRdq0leLkPCv9NxqUyqGi+D1R3CWSu9gHugAvH0f35YYSkrcZbsu0i8FOhDMD68qF+Xm+Svx3i/lX8ObkhnQKt8QnXYfpfMxYtGfAZhjdgpLcPxjqD01ONT0Jl4npYNevKxAbtWUVh2vWCja2hvZEAw5fw2cZCqlz55Jh74wYo78UXRwYBfiX/e6ZWY70gVQteaIy7TJmvXPnNMCdexxfmS3qyRnjomyFPHodbDkiN2KLPENY3F9a8mg+d2Y9gIbBDUDuDi7s2TA74DFZsYvdF7Y72u9i7dQVhxy956RBzGcM6LZMccQ1rmYDgZt5I07HMJ5xJjvNo9mSyfEvFmkJFvPDuHZX4dWZjwuTXdHad619AnVTL79fpdwaqizrp5ylLUcObOxCSTfsNac5j3gkSmm+tQacbX8EuOUZLb1pHCMbdYAthgcWcmFqK3MGQcSmj5/f9LbjnjxAkugNoFmeoOYZVSKkaoA1PL3B1BTbyQh1ksDT3A+wmvmSvb6ZVpJdTsP2VxwKwF3/urWZWMsQLlgBwL3yQmpkTfC/YhH2FFhn5ap/m3KoJtEKUBo/c/uLF9L7ES3tlIiad6MhnJsoua5RcsPXUhufv4ZZGpRBPeMi44QqWMtssGh0HsnKTdPOJaDAViWRA9qjtmIk5n3hwmglwVIiPtzGASOHCYqlQHgOqJDyvdbsBCz3HuFvnpg79lW2PzUnI4mXhwSVsFA6Lr3rVC8jv2XdcNCT7lDGYxcsp3STzSNAqURGA9RU/FLiwTN+YSalQ04AYq3dxJS/X88vQ55SQ7fsH5A/uAqd1j9A6M95JPgYBYobCIZTf5zDdMoZNoRisqS2KTXOy8OlYX6Y4XHSkeHOJrjZLWlapNeZJyj6s4cVQFuCmKkRENCRx9Tt05BDNEKBBvVKlmfJvqpXPzygv0q+HovGqxf29Qs5CjcvCfa/bV99egfmjDQ1z8TFY5QHWTFLLQj2j7BxPlqqt9YbYxcv84mstP2c4aSUN7cG4LEYbFPPo0tJVeCUtFdjPmHhz5GgBqpo1AfpAEQNVf5BUHCj4KJqUkhKmqEMiFdUXg06FUxnP4iLicnQDVi0RHOJj4l4HexhD21IVaM='}}

</details>

<details>
<summary>transports-model-context-protocol</summary>

{'type': 'text', 'text': 'MCP uses JSON-RPC to encode messages. JSON-RPC messages **MUST** be UTF-8 encoded.The protocol currently defines two standard transport mechanisms for client-server\ncommunication:\n\n1. [stdio](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#stdio), communication over standard in and standard out\n2. [Streamable HTTP](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#streamable-http)\n\nClients **SHOULD** support stdio whenever possible.It is also possible for clients and servers to implement\n[custom transports](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#custom-transports) in a pluggable fashion.\n\n## stdio\n\nIn the **stdio** transport:\n\n- The client launches the MCP server as a subprocess.\n- The server reads JSON-RPC messages from its standard input (`stdin`) and sends messages\nto its standard output (`stdout`).\n- Messages are individual JSON-RPC requests, notifications, or responses.\n- Messages are delimited by newlines, and **MUST NOT** contain embedded newlines.\n- The server **MAY** write UTF-8 strings to its standard error (`stderr`) for logging\npurposes. Clients **MAY** capture, forward, or ignore this logging.\n- The server **MUST NOT** write anything to its `stdout` that is not a valid MCP message.\n- The client **MUST NOT** write anything to the server’s `stdin` that is not a valid MCP\nmessage.', 'extras': {'signature': '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'}}

</details>

<details>
<summary>vesa-alexandru-substack</summary>

{'type': 'text', 'text': 'empty', 'extras': {'signature': 'EtsGCtgGAQw51scbCEDjoPexy3mvb3gAfTgNEps/OWQ7FvMoczvsaTn2lKZwZtyFq7YGLsh0Pk3ZNTMXn6RUrkwAqxFU16m93VqxyHI+ZqihzR2B6KpaIvsaIYlr3g3D+sQnEH0Dm7N4njK8kpKPKHj/T1TZPIZh0M7pQO4pDkT0PYtcqV5VGubfioGx9hNabURXnDDdbdQBBonbQcmufx54L0PRHbEr3DeWvc/5sLPyA1Iqgih6X1tPwvW70saxbz52rNZc0v1cxC8LLyau+bUqVckYMUlBt3eS7D91NPcV3AHXfmZvW5QO6c2fXrSR85eAwk7pXL5qBkc3gCyYhMcIU2vIG6MUVdEaTWg7gI0TJAgJLsWCHOvP8qM5Tn2VO1TUVpMGJPHGD6SZ/h//Z/cxWY4kb+JBn8EEfEG4Otu3uc3eDjhmuGMPGn2+OSbRZG/VtuhBsJTZjj1QX2FLWxI+lm5BViwprZczdxRJHWCN7yRnmc4d+BAStkg9OAqutiXg5a5UaAz2SAyQGICG0HKyk2guFTJ7aaZZGsKEehNvMkRFCcI6Cjn8jVazRnw05/Ny9Dk+Dog6NIK8advKb/uFXCyfNVxaxf8CJSFn6ahqkNaoJEBJVSwesahpPBt/MIUic3ik5PqvZlnBEAhKdUVPIDsAzQ2kiRG6m2eTP7x4KClDFodhWMcmT9Dz4/0GhmRd7R/b67AjAriSSamuyC7PmrKhmSVc8mEFZo3HzZ8Tzcu3D2kNbgbCG8gDtBTRArMDn0b5QQeDFXNYoSUy0A/NT8N4TpiQFa69yzLmowm7+eIzMEo4TGObQ9L287jTx2PyAsEXWO76gQhcurJRnOBQxDi/cGXihqkrQsNOZL8vC61ZLTTHK7KkQ8a/d417AylbbIVc4h2rwBOkgL26YIFOXnvYiBGmkWF1v2HVxG/ighpbeZrQk5CFEDuA81o9ecpWOWn/U51j+YKHlQ49TKiUjNOPp05YvTlJRGdZUeUuEqF+x6TSSBfLDTYttVl2YSCCaK6xymtwHYZgx2CPQK0ITPE4snWs8TY3+0NgzxtNHKb4HO1ymYznRj1M5Y9AF14c7FJu7t2mORpfEHrZbygWurQPsrN/W8Ia2ynSR+6JguBvsI8Yl24fhRQ8vQ=='}}

</details>

<details>
<summary>welcome-to-fastmcp-fastmcp</summary>

{'type': 'text', 'text': '**FastMCP is the standard framework for building MCP applications.** The Model Context Protocol (MCP) connects LLMs to tools and data. FastMCP gives you everything you need to go from prototype to production — build servers that expose capabilities, connect clients to any MCP service, and give your tools interactive UIs:\n\n```python\nfrom fastmcp import FastMCP\n\nmcp = FastMCP("Demo 🚀")\n\n@mcp.tool\ndef add(a: int, b: int) -> int:\n    """Add two numbers"""\n    return a + b\n\nif __name__ == "__main__":\n    mcp.run()\n```\n\nThe Model Context Protocol (MCP) lets you give agents access to your tools and data. FastMCP handles all of it. Declare a tool with a Python function, and the schema, validation, and documentation are generated automatically. Connect to a server with a URL, and transport negotiation, authentication, and protocol lifecycle are managed for you. You focus on your logic, and the MCP part just works: with FastMCP, best practices are built in.\n\nFastMCP has three pillars:\n\n*   **Servers** wrap your Python functions into MCP-compliant tools, resources, and prompts.\n*   **Clients** connect to any server with full protocol support.\n*   **Apps** give your tools interactive UIs rendered directly in the conversation.\n\nYou can use FastMCP to connect programmatically as a client:\n\n```python\nimport asyncio\nfrom fastmcp import Client\n\nasync def main():\n    async with Client("https://gofastmcp.com/mcp") as client:\n        result = await client.call_tool(\n            name="search_fast_mcp",\n            arguments={"query": "deploy a FastMCP server"}\n        )\n    print(result)\n\nasyncio.run(main())\n```', 'extras': {'signature': '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'}}

</details>

<details>
<summary>what-is-the-model-context-protocol-mcp-model-context-protoco-1</summary>

{'type': 'text', 'text': 'MCP (Model Context Protocol) is an open-source standard for connecting AI applications to external systems.Using MCP, AI applications like Claude or ChatGPT can connect to data sources (e.g. local files, databases), tools (e.g. search engines, calculators) and workflows (e.g. specialized prompts)—enabling them to access key information and perform tasks.Think of MCP like a USB-C port for AI applications. Just as USB-C provides a standardized way to connect electronic devices, MCP provides a standardized way to connect AI applications to external systems.\n\nhttps://mintcdn.com/mcp/bEUxYpZqie0DsluH/images/mcp-simple-diagram.png?fit=max&auto=format&n=bEUxYpZqie0DsluH&q=85&s=35268aa0ad50b8c385913810e7604550\n\n## What can MCP enable?\n\n- Agents can access your Google Calendar and Notion, acting as a more personalized AI assistant.\n- Claude Code can generate an entire web app using a Figma design.\n- Enterprise chatbots can connect to multiple databases across an organization, empowering users to analyze data using chat.\n- AI models can create 3D designs on Blender and print them out using a 3D printer.\n\n## Why does MCP matter?\n\nDepending on where you sit in the ecosystem, MCP can have a range of benefits.\n\n- **Developers**: MCP reduces development time and complexity when building, or integrating with, an AI application or agent.\n- **AI applications or agents**: MCP provides access to an ecosystem of data sources, tools and apps which will enhance capabilities and improve the end-user experience.\n- **End-users**: MCP results in more capable AI applications or agents which can access your data and take actions on your behalf when necessary.\n\n## Broad ecosystem support\n\nMCP is an open protocol supported across a wide range of clients and servers. AI assistants like [Claude](https://claude.com/docs/connectors/building) and [ChatGPT](https://developers.openai.com/api/docs/mcp/), development tools like [Visual Studio Code](https://code.visualstudio.com/docs/copilot/chat/mcp-servers), [Cursor](https://cursor.com/docs/context/mcp), [MCPJam](https://docs.mcpjam.com/getting-started), and many others all support MCP — making it easy to build once and integrate everywhere.', 'extras': {'signature': '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'}}

</details>

<details>
<summary>what-is-the-model-context-protocol-mcp-model-context-protoco</summary>

{'type': 'text', 'text': 'MCP (Model Context Protocol) is an open-source standard for connecting AI applications to external systems. Using MCP, AI applications like Claude or ChatGPT can connect to data sources (e.g. local files, databases), tools (e.g. search engines, calculators) and workflows (e.g. specialized prompts)—enabling them to access key information and perform tasks. Think of MCP like a USB-C port for AI applications. Just as USB-C provides a standardized way to connect electronic devices, MCP provides a standardized way to connect AI applications to external systems.\n\nhttps://mintcdn.com/mcp/bEUxYpZqie0DsluH/images/mcp-simple-diagram.png?fit=max&auto=format&n=bEUxYpZqie0DsluH&q=85&s=35268aa0ad50b8c385913810e7604550\n\n## What can MCP enable?\n\n- Agents can access your Google Calendar and Notion, acting as a more personalized AI assistant.\n- Claude Code can generate an entire web app using a Figma design.\n- Enterprise chatbots can connect to multiple databases across an organization, empowering users to analyze data using chat.\n- AI models can create 3D designs on Blender and print them out using a 3D printer.\n\n## Why does MCP matter?\n\nDepending on where you sit in the ecosystem, MCP can have a range of benefits.\n\n- **Developers**: MCP reduces development time and complexity when building, or integrating with, an AI application or agent.\n- **AI applications or agents**: MCP provides access to an ecosystem of data sources, tools and apps which will enhance capabilities and improve the end-user experience.\n- **End-users**: MCP results in more capable AI applications or agents which can access your data and take actions on your behalf when necessary.\n\n## Broad ecosystem support\n\nMCP is an open protocol supported across a wide range of clients and servers. AI assistants like [Claude](https://claude.com/docs/connectors/building) and [ChatGPT](https://developers.openai.com/api/docs/mcp/), development tools like [Visual Studio Code](https://code.visualstudio.com/docs/copilot/chat/mcp-servers), [Cursor](https://cursor.com/docs/context/mcp), [MCPJam](https://docs.mcpjam.com/getting-started), and many others all support MCP — making it easy to build once and integrate everywhere.', 'extras': {'signature': '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'}}

</details>
