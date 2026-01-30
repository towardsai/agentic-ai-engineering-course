# Nova MCP Server - Docker Setup Guide

This guide explains how to run the Nova MCP Server with PostgreSQL using Docker Compose.

## Architecture

The Docker setup includes:
- **PostgreSQL 16** - Database for storing research data
- **Nova MCP Server** - The research agent server with async SQLAlchemy

## Prerequisites

- Docker & Docker Compose installed
- API keys (see Configuration section below)

## Quick Start

### 1. Navigate to the mcp_server directory

```bash
cd /path/to/nova/mcp_server
```

### 2. Create `.env` file

Copy the example file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your required API keys:
- `GOOGLE_API_KEY` - Required for Gemini models
- `PPLX_API_KEY` - Required for Perplexity research
- `FIRECRAWL_API_KEY` - Required for web scraping

Optional keys:
- `GITHUB_TOKEN` - For GitHub repository analysis
- `OPENAI_API_KEY` - For GPT models
- `OPIK_API_KEY` - For monitoring

### 3. Install dependencies (first time only)

This generates the `uv.lock` file needed for Docker build:

```bash
uv sync
```

### 4. Start the services

```bash
docker compose up -d
```

This will:
1. Start PostgreSQL database
2. Build the MCP server container
3. Run database migrations automatically
4. Start the MCP server on port 8000

### 5. Verify services are running

```bash
# Check container status
docker compose ps

# View logs
docker compose logs -f

# Check just the server logs
docker compose logs -f mcp-server

# Check just the database logs
docker compose logs -f postgres
```

## Docker Compose Commands

### Start services
```bash
docker compose up -d              # Start in detached mode
docker compose up                 # Start with logs visible
```

### Stop services
```bash
docker compose stop               # Stop containers (keep data)
docker compose down               # Stop and remove containers (keep data)
docker compose down -v            # Stop, remove containers AND delete database data
```

### Rebuild after code changes
```bash
docker compose up -d --build      # Rebuild and restart
```

### View logs
```bash
docker compose logs -f            # All services
docker compose logs -f mcp-server # Just the server
docker compose logs -f postgres   # Just the database
```

## Database Management

### Access PostgreSQL directly

```bash
docker compose exec postgres psql -U nova -d nova_research
```

Common PostgreSQL commands:
```sql
-- List all tables
\dt

-- Describe articles table
\d+ articles

-- Query articles
SELECT id, user_id, status, created_at FROM articles;

-- Exit
\q
```

### Run migrations manually

Migrations run automatically on server startup, but you can also run them manually:

```bash
# Check current migration version
docker compose exec mcp-server alembic current

# Show migration history
docker compose exec mcp-server alembic history

# Apply all pending migrations
docker compose exec mcp-server alembic upgrade head

# Rollback one migration
docker compose exec mcp-server alembic downgrade -1

# Generate a new migration (after model changes)
docker compose exec mcp-server alembic revision --autogenerate -m "description"
```

### Reset database (WARNING: deletes all data)

```bash
docker compose down -v
docker compose up -d
```

## Development Workflow

### Option 1: Run everything in Docker

Best for production-like testing:

```bash
# Start services
docker compose up -d

# View logs as you work
docker compose logs -f mcp-server

# Restart after code changes (with rebuild)
docker compose up -d --build
```

### Option 2: Database in Docker, Server locally

Best for active development with hot reload:

```bash
# Start only PostgreSQL
docker compose up -d postgres

# Set environment variables
export DATABASE_URL="postgresql+asyncpg://nova:nova_dev_password@localhost:5432/nova_research"
export GOOGLE_API_KEY="your-key"
export PPLX_API_KEY="your-key"
export FIRECRAWL_API_KEY="your-key"

# Run migrations
uv run alembic upgrade head

# Start server locally
uv run python -m src.server --transport streamable-http --port 8000
```

## Configuration

### Environment Variables

All configuration is done via environment variables in `.env`:

| Variable | Required | Description |
|----------|----------|-------------|
| `POSTGRES_USER` | No (default: nova) | PostgreSQL username |
| `POSTGRES_PASSWORD` | No (default: nova_dev_password) | PostgreSQL password |
| `POSTGRES_DB` | No (default: nova_research) | Database name |
| `DATABASE_URL` | No (auto-generated) | Full database connection URL |
| `GOOGLE_API_KEY` | Yes | Google Gemini API key |
| `PPLX_API_KEY` | Yes | Perplexity API key |
| `FIRECRAWL_API_KEY` | Yes | Firecrawl API key |
| `GITHUB_TOKEN` | No | GitHub personal access token |
| `OPENAI_API_KEY` | No | OpenAI API key |
| `OPIK_API_KEY` | No | Opik monitoring key |
| `LOG_LEVEL` | No (default: 20/INFO) | Logging level |
| `SERVER_PORT` | No (default: 8000) | Server port |

### Ports

Default ports (can be changed in `.env`):
- PostgreSQL: `5432`
- MCP Server: `8000`

## Database Schema

The initial migration creates the `articles` table:

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key (auto-generated) |
| `user_id` | TEXT | User ID from Descope authentication |
| `guideline_text` | TEXT | Full article guideline content |
| `status` | ENUM | Workflow status |
| `extracted_urls` | JSON | Extracted URLs from guidelines |
| `perplexity_results` | TEXT | Raw Perplexity research results |
| `perplexity_results_selected` | TEXT | Filtered Perplexity results |
| `perplexity_sources_selected` | TEXT | Comma-separated selected source IDs |
| `urls_to_scrape_from_research` | TEXT | URLs selected for full scraping |
| `created_at` | TIMESTAMP | Creation timestamp |
| `updated_at` | TIMESTAMP | Last update timestamp |

## Troubleshooting

### Container won't start

Check logs:
```bash
docker compose logs mcp-server
```

Common issues:
1. Missing `.env` file - Copy `.env.example` to `.env`
2. Missing `uv.lock` - Run `uv sync` first
3. Port conflict - Change `SERVER_PORT` in `.env`

### Database connection errors

```bash
# Check if PostgreSQL is healthy
docker compose ps

# View PostgreSQL logs
docker compose logs postgres

# Test connection manually
docker compose exec postgres psql -U nova -d nova_research -c "SELECT 1;"
```

### Migration errors

```bash
# Check current migration state
docker compose exec mcp-server alembic current

# Try to apply migrations manually
docker compose exec mcp-server alembic upgrade head

# If migrations are corrupted, reset database (WARNING: deletes data)
docker compose down -v
docker compose up -d
```

## Cloud SQL Migration (Future)

When deploying to GCP Cloud SQL:

1. Create a Cloud SQL PostgreSQL instance
2. Update `DATABASE_URL` in production environment:
   ```
   DATABASE_URL=postgresql+asyncpg://user:pass@/nova_research?host=/cloudsql/PROJECT:REGION:INSTANCE
   ```
3. Run migrations:
   ```bash
   alembic upgrade head
   ```

No code changes needed - the same SQLAlchemy models and migrations work with both local PostgreSQL and Cloud SQL.

## Data Persistence

Database data is stored in a Docker volume named `postgres_data`. This persists across container restarts.

To backup your data:
```bash
docker compose exec postgres pg_dump -U nova nova_research > backup.sql
```

To restore from backup:
```bash
cat backup.sql | docker compose exec -T postgres psql -U nova nova_research
```

## Health Checks

The setup includes health checks:
- PostgreSQL: Checks `pg_isready` every 5 seconds
- MCP Server: Checks HTTP endpoint at `/health` every 30 seconds

## Security Notes

For production deployment:
1. Change default PostgreSQL credentials
2. Use secrets management for API keys
3. Enable SSL/TLS for database connections
4. Run containers as non-root user (already configured)
5. Use environment-specific `.env` files
6. Never commit `.env` files to version control

