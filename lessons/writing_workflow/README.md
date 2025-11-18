# Brown

## Running Brown 











## Running the Research Agent (Nova)

Setup:
1. In `src/nova/mcp_client/`, create an `.env` file (using `.env.example` as template) with the API keys.
2. In `src/nova/mcp_server/`, create an `.env` file (using `.env.example` as template) with the API keys.

Then, to run the research agent Nova, use a command like the following:

`make run-research-agent RESEARCH_FOLDER=$(pwd)/inputs/articles/1_ai_engineering_and_agent_landscape`

The final output is the `research.md` file in the `RESEARCH_FOLDER` folder.

You can update the agent instructions by editing the file `src/nova/mcp_client/src/instructions.md`. For example, you can tell the agent to ask for human feedback after each research round.

## GitHub Token Setup (Classic)

To enable the research agent to access your private GitHub repositories, you need to create a 'classic' personal access token.

1.  **Navigate to Token Creation:**
    - Go to [https://github.com/settings/personal-access-tokens](https://github.com/settings/personal-access-tokens).
    - Click on the "Generate new token" dropdown, and select **"Generate new token (classic)"**.

2.  **Configure the Classic Token:**
    - **Note:** Give your token a descriptive name (e.g., `course-agents-writing-research-agent`).
    - **Expiration:** Select an appropriate expiration date for your token.
    - **Select scopes:** To allow the agent to access your private repositories, select the following scope(s):
        - check the box next to `repo` (Full control of private repositories).

3.  **Generate and copy the token:**
    - Click "Generate token" at the bottom of the page and copy the generated token.
    - You will need to add this to the `GITHUB_TOKEN` environment variable in the `.env` file in the `src/nova/mcp_server/` directory.

## Running the Writer Agent (Brown)

... to be updated ...
