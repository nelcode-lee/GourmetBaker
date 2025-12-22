#!/bin/bash
# Helper script to update DATABASE_URL in .env file

echo "=========================================="
echo "Update DATABASE_URL in .env"
echo "=========================================="
echo ""
echo "Your Neon connection string should look like:"
echo "postgresql://username:password@ep-xxxxx.region.aws.neon.tech/dbname?sslmode=require"
echo ""
read -p "Paste your Neon DATABASE_URL: " db_url

if [ -z "$db_url" ]; then
    echo "❌ No connection string provided"
    exit 1
fi

# Update .env file
cd "$(dirname "$0")"
if grep -q "^DATABASE_URL=" .env; then
    # Use sed with backup (macOS compatible)
    sed -i '' "s|^DATABASE_URL=.*|DATABASE_URL=$db_url|" .env
    echo "✅ DATABASE_URL updated in .env file"
else
    echo "DATABASE_URL=$db_url" >> .env
    echo "✅ DATABASE_URL added to .env file"
fi

echo ""
echo "Testing connection..."
source venv/bin/activate 2>/dev/null || echo "Note: Activate venv manually if needed"
python3 setup_database.py

