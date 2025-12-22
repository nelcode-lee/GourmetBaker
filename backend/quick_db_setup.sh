#!/bin/bash
# Quick database setup helper script

echo "=========================================="
echo "Database Setup Helper"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Creating from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file"
fi

echo ""
echo "Choose your database option:"
echo "1. Neon PostgreSQL (Recommended - Free, easy setup)"
echo "2. Local PostgreSQL"
echo "3. Docker PostgreSQL"
echo "4. I already have a DATABASE_URL"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "📝 Neon Setup Instructions:"
        echo "1. Go to https://neon.tech and sign up"
        echo "2. Create a new project"
        echo "3. Enable 'vector' extension in Extensions tab"
        echo "4. Copy your connection string from Connection Details"
        echo ""
        read -p "Paste your Neon DATABASE_URL here: " db_url
        if [ ! -z "$db_url" ]; then
            # Update .env file
            if grep -q "^DATABASE_URL=" .env; then
                sed -i '' "s|^DATABASE_URL=.*|DATABASE_URL=$db_url|" .env
            else
                echo "DATABASE_URL=$db_url" >> .env
            fi
            echo "✅ DATABASE_URL updated in .env"
        fi
        ;;
    2)
        echo ""
        read -p "Enter local PostgreSQL connection string (e.g., postgresql://user:pass@localhost:5432/dbname): " db_url
        if [ ! -z "$db_url" ]; then
            if grep -q "^DATABASE_URL=" .env; then
                sed -i '' "s|^DATABASE_URL=.*|DATABASE_URL=$db_url|" .env
            else
                echo "DATABASE_URL=$db_url" >> .env
            fi
            echo "✅ DATABASE_URL updated in .env"
        fi
        ;;
    3)
        echo ""
        echo "Starting Docker PostgreSQL with pgvector..."
        docker-compose -f docker-compose.db.yml up -d 2>/dev/null || {
            echo "Creating docker-compose.db.yml..."
            cat > docker-compose.db.yml << 'EOF'
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_USER: raguser
      POSTGRES_PASSWORD: ragpass
      POSTGRES_DB: rag_documents
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
EOF
            docker-compose -f docker-compose.db.yml up -d
            echo "✅ Docker PostgreSQL started"
            echo "DATABASE_URL=postgresql://raguser:ragpass@localhost:5432/rag_documents" >> .env
            echo "✅ DATABASE_URL added to .env"
        }
        ;;
    4)
        echo ""
        read -p "Enter your DATABASE_URL: " db_url
        if [ ! -z "$db_url" ]; then
            if grep -q "^DATABASE_URL=" .env; then
                sed -i '' "s|^DATABASE_URL=.*|DATABASE_URL=$db_url|" .env
            else
                echo "DATABASE_URL=$db_url" >> .env
            fi
            echo "✅ DATABASE_URL updated in .env"
        fi
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Testing database connection..."
echo "=========================================="
echo ""

# Test connection
python3 setup_database.py

