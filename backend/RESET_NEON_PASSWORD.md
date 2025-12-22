# Reset Neon Database Password

## Option 1: Get Fresh Connection String (Easiest)

1. Go to **https://console.neon.tech**
2. Select your project
3. Go to **Settings** → **Connection Details** (or just **Connection Details** in sidebar)
4. You'll see connection strings - copy the one that says **"Connection string"** or **"URI"**
5. It should look like:
   ```
   postgresql://neondb_owner:NEW_PASSWORD@ep-xxxxx.region.aws.neon.tech/neondb?sslmode=require
   ```
6. Update your `.env` file with this new connection string

## Option 2: Reset Password in Neon

1. Go to **Neon Dashboard** → Your Project
2. Go to **Settings** → **Database** (or look for password/reset options)
3. Click **"Reset Password"** or **"Change Password"**
4. Copy the new password immediately
5. Update your connection string in `.env`

## Option 3: Use Different Connection Method

Neon sometimes provides multiple connection strings:
- **Pooled connection** (recommended for serverless)
- **Direct connection**
- **Transaction mode**

Try the **pooled connection** string if available.

## Update .env File

After getting the new connection string:

```bash
cd backend
# Edit .env file
nano .env
# Find DATABASE_URL= line
# Replace with: DATABASE_URL=postgresql://user:NEW_PASSWORD@host/dbname?sslmode=require
# Save: Ctrl+X, Y, Enter
```

Or use this command (replace NEW_CONNECTION_STRING):
```bash
cd backend
sed -i '' 's|^DATABASE_URL=.*|DATABASE_URL=NEW_CONNECTION_STRING|' .env
```

## Test Connection

```bash
cd backend
source venv/bin/activate
python check_db_connection.py
```

## Common Issues

**"password authentication failed"**
- Password in .env doesn't match Neon
- Connection string has wrong format
- Using old/expired connection string

**"connection refused"**
- Wrong host/endpoint
- Network/firewall issue
- Neon project paused (free tier)

**"database does not exist"**
- Wrong database name in connection string
- Database was deleted

