"""
Database setup script - Tests connection and initializes schema
"""
import asyncio
import sys
from app.core.config import get_settings
from app.core.db.database import init_db, get_db_pool, close_db_pool

async def test_connection():
    """Test database connection"""
    settings = get_settings()
    print(f"Testing connection to database...")
    print(f"Database URL: {settings.DATABASE_URL[:50]}...")  # Show first 50 chars
    
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            # Test basic query
            result = await conn.fetchval("SELECT version();")
            print(f"✅ Connection successful!")
            print(f"PostgreSQL version: {result[:50]}")
            
            # Check if pgvector extension exists
            has_vector = await conn.fetchval("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                );
            """)
            
            if has_vector:
                print("✅ pgvector extension is installed")
            else:
                print("⚠️  pgvector extension not found - attempting to create it...")
                try:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    print("✅ pgvector extension created successfully!")
                except Exception as e:
                    print(f"⚠️  Could not create extension automatically: {str(e)}")
                    print("   Please enable it manually in Neon SQL Editor:")
                    print("   CREATE EXTENSION IF NOT EXISTS vector;")
            
            return True
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False

async def setup_database():
    """Initialize database schema"""
    print("\n" + "="*60)
    print("Initializing database schema...")
    print("="*60)
    
    try:
        await init_db()
        print("✅ Database schema initialized successfully!")
        
        # Verify tables were created
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            
            print(f"\n📊 Created tables ({len(tables)}):")
            for table in tables:
                print(f"   - {table['table_name']}")
        
        return True
    except Exception as e:
        print(f"❌ Schema initialization failed: {str(e)}")
        return False

async def main():
    """Main setup function"""
    print("="*60)
    print("Cranswick Technical Standards Agent - Database Setup")
    print("="*60)
    print()
    
    # Test connection
    if not await test_connection():
        print("\n❌ Cannot proceed without database connection.")
        print("\nPlease check:")
        print("1. DATABASE_URL is set correctly in .env file")
        print("2. Database server is running and accessible")
        print("3. Credentials are correct")
        sys.exit(1)
    
    # Initialize schema
    if not await setup_database():
        print("\n❌ Database setup incomplete.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ Database setup complete!")
    print("="*60)
    print("\nYou can now start the FastAPI server:")
    print("  uvicorn app.main:app --reload")
    
    # Clean up
    await close_db_pool()

if __name__ == "__main__":
    asyncio.run(main())

