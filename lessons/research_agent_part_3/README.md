# Nova MCP Server - Quick Start

## Running with Docker (Recommended)

### 1. Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys:
# - GOOGLE_API_KEY (required)
# - PPLX_API_KEY (required)  
# - FIRECRAWL_API_KEY (required)

# Install dependencies (creates uv.lock)
uv sync
```

### 2. Start Everything

```bash
# Start PostgreSQL + MCP Server
docker compose up -d

# Watch logs
docker compose logs -f
```

The server will:
- Start PostgreSQL on port 5432
- Run database migrations automatically
- Start MCP server on port 8000

### 3. Stop Services

```bash
docker compose down
```

---

## Running Locally (Without Docker)

### 1. Start Database Only

```bash
docker compose up -d postgres
```

### 2. Setup Environment

```bash
# Copy and edit .env
cp .env.example .env

# Or export variables directly
export DATABASE_URL="postgresql+asyncpg://nova:nova_dev_password@localhost:5432/nova_research"
export GOOGLE_API_KEY="your-key"
export PPLX_API_KEY="your-key"
export FIRECRAWL_API_KEY="your-key"
```

### 3. Run Migrations

```bash
uv run alembic upgrade head
```

### 4. Start Server

```bash
uv run python -m src.server --transport streamable-http --port 8000
```

---

## Useful Commands

```bash
# View logs
docker compose logs -f mcp-server

# Check status
docker compose ps

# Access database
docker compose exec postgres psql -U nova -d nova_research

# Run migrations manually
docker compose exec mcp-server alembic upgrade head

# Reset everything (WARNING: deletes data)
docker compose down -v
docker compose up -d
```