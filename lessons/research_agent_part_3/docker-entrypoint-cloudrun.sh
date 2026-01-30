#!/bin/bash
# Cloud Run entrypoint script for Nova MCP Server
# This script handles database migrations and starts the server with Cloud Run settings

set -e

echo "🔄 Running database migrations..."
if alembic upgrade head 2>&1; then
    echo "✅ Database migrations complete"
else
    echo "❌ Migration failed"
    echo "⚠️  Server will continue but database may be in inconsistent state"
fi

# Cloud Run sets the PORT environment variable
# Default to 8080 which is Cloud Run's default
export SERVER_PORT="${PORT:-8080}"
# Cloud Run requires listening on 0.0.0.0
export SERVER_HOST="0.0.0.0"

echo "🚀 Starting Nova MCP Server on ${SERVER_HOST}:${SERVER_PORT}..."

# Start the server with HTTP transport
exec python -m src.server --transport http

