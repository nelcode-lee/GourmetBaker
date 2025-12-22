#!/bin/bash
# Helper script to update DATABASE_URL after password reset

echo "=========================================="
echo "Neon Password Reset Helper"
echo "=========================================="
echo ""
echo "Steps to reset password in Neon:"
echo "1. Go to https://console.neon.tech"
echo "2. Select your project"
echo "3. Go to Settings → Database (or look for 'Reset Password')"
echo "4. Click 'Reset Password' or 'Change Password'"
echo "5. Copy the NEW connection string from Connection Details"
echo ""
echo "The new connection string should look like:"
echo "postgresql://user:NEW_PASSWORD@ep-xxxxx.region.aws.neon.tech/dbname?sslmode=require"
echo ""
read -p "Paste your NEW Neon connection string here: " new_db_url

if [ -z "$new_db_url" ]; then
    echo "❌ No connection string provided"
    exit 1
fi

# Remove any quotes or spaces
new_db_url=$(echo "$new_db_url" | sed "s/^['\"]//; s/['\"]$//; s/^[[:space:]]*//; s/[[:space:]]*$//")

cd "$(dirname "$0")"

# Update .env file
if grep -q "^DATABASE_URL=" .env; then
    # Use sed with backup (macOS compatible)
    sed -i '' "s|^DATABASE_URL=.*|DATABASE_URL=$new_db_url|" .env
    echo "✅ DATABASE_URL updated in .env file"
else
    echo "DATABASE_URL=$new_db_url" >> .env
    echo "✅ DATABASE_URL added to .env file"
fi

echo ""
echo "Testing connection..."
source venv/bin/activate 2>/dev/null || echo "Note: Activate venv manually if needed"
python3 check_db_connection.py

