# Database Setup Guide

## Option 1: Neon PostgreSQL (Recommended - Easiest)

Neon is a serverless PostgreSQL with built-in pgvector support.

### Steps:

1. **Create a Neon account**
   - Go to [neon.tech](https://neon.tech)
   - Sign up (free tier available)

2. **Create a new project**
   - Click "New Project"
   - Choose a name and region
   - Select PostgreSQL version (15+ recommended)

3. **Enable pgvector extension**
   - In your Neon dashboard, go to "Extensions"
   - Enable the "vector" extension
   - Or run: `CREATE EXTENSION vector;` in the SQL editor

4. **Get connection string**
   - In Neon dashboard, go to "Connection Details"
   - Copy the connection string (looks like: `postgresql://user:password@host/dbname`)

5. **Add to .env file**
   ```bash
   cd backend
   # Edit .env file and add:
   DATABASE_URL=postgresql://user:password@host/dbname
   ```

6. **Test connection**
   ```bash
   python setup_database.py
   ```

## Option 2: Local PostgreSQL with pgvector

### Steps:

1. **Install PostgreSQL**
   ```bash
   # macOS
   brew install postgresql@15
   brew services start postgresql@15
   
   # Linux (Ubuntu/Debian)
   sudo apt-get install postgresql-15 postgresql-contrib
   sudo systemctl start postgresql
   ```

2. **Install pgvector extension**
   ```bash
   # macOS
   brew install pgvector
   
   # Linux
   git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
   cd pgvector
   make
   sudo make install
   ```

3. **Create database**
   ```bash
   createdb rag_documents
   psql rag_documents -c "CREATE EXTENSION vector;"
   ```

4. **Add to .env file**
   ```bash
   DATABASE_URL=postgresql://localhost:5432/rag_documents
   # Or with username:
   DATABASE_URL=postgresql://your_username@localhost:5432/rag_documents
   ```

5. **Test connection**
   ```bash
   python setup_database.py
   ```

## Option 3: Docker PostgreSQL with pgvector

### Steps:

1. **Create docker-compose.db.yml**
   ```yaml
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
   ```

2. **Start database**
   ```bash
   docker-compose -f docker-compose.db.yml up -d
   ```

3. **Add to .env file**
   ```bash
   DATABASE_URL=postgresql://raguser:ragpass@localhost:5432/rag_documents
   ```

4. **Test connection**
   ```bash
   python setup_database.py
   ```

## Testing the Connection

After setting up your DATABASE_URL in `.env`, run:

```bash
cd backend
python setup_database.py
```

This will:
- ✅ Test the database connection
- ✅ Check for pgvector extension
- ✅ Create all required tables
- ✅ Show you what was created

## Troubleshooting

### Connection refused
- Check if PostgreSQL is running
- Verify the host and port in DATABASE_URL
- Check firewall settings

### Authentication failed
- Verify username and password
- Check if user has proper permissions
- For Neon: Make sure you're using the correct connection string

### pgvector extension not found
- For Neon: Enable it in the dashboard
- For local: Install pgvector extension
- For Docker: Use the pgvector/pgvector image

### Module 'asyncpg' not found
```bash
pip install asyncpg
```

## Next Steps

Once the database is set up:
1. Start the FastAPI server: `uvicorn app.main:app --reload`
2. Start Celery worker: `celery -A app.worker.tasks celery_app worker --loglevel=info`
3. Test document upload in the frontend

