#!/bin/bash
set -e

echo "🔄 Running database migrations..."
if alembic upgrade head 2>&1; then
    echo "✅ Database migrations complete"
else
    echo "❌ Migration failed"
    echo "⚠️  Server will continue but database may be in inconsistent state"
fi

echo "🚀 Starting Nova MCP Server..."
exec python -m src.server --transport http

