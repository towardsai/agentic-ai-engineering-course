# Agentic AI Engineering Course

Python modules for the Agentic AI Engineering course by Towards AI and Decoding AI. [Find more about the course](https://academy.towardsai.net/courses/agent-engineering).

## Installation

```bash
pip install agentic-ai-engineering-course
```

## Usage

### Environment Utilities

```python
from utils import load

# Load environment variables from .env file
load()

# Load with custom path and required variables
load(dotenv_path="path/to/.env", required_env_vars=["API_KEY", "SECRET"])
```

### Pretty Print Utilities

```python
from utils import wrapped, function_call, Color

# Pretty print text with custom formatting
wrapped("Hello World", title="My Message", header_color=Color.BLUE)

# Pretty print function calls
function_call(
    function_call=my_function_call,
    title="Function Execution",
    header_color=Color.GREEN
)
```

## Development

This package is part of the Agentic AI Engineering course materials. For the full course experience, visit the main repository.

## Authors

- Paul Iusztin (p.b.iusztin@gmail.com)
- Fabio Chiusano (chiusanofabio94@gmail.com)
- Omar Solano (omarsolano27@gmail.com)

## License

Apache License 2.0 - see LICENSE file for details.