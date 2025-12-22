#!/usr/bin/env python3
"""
Check database connection and diagnose issues
"""
import asyncio
import asyncpg
from urllib.parse import quote_plus, unquote
from app.core.config import get_settings

async def test_connection():
    settings = get_settings()
    db_url = settings.DATABASE_URL
    
    print("="*60)
    print("Database Connection Diagnostic")
    print("="*60)
    print(f"\nConnection string (first 50 chars): {db_url[:50]}...")
    
    # Parse connection string
    if db_url.startswith('postgresql://'):
        parts = db_url.replace('postgresql://', '').split('@')
        if len(parts) == 2:
            auth = parts[0]
            if ':' in auth:
                user, password = auth.split(':', 1)
                print(f"\nUsername: {user}")
                print(f"Password length: {len(password)}")
                print(f"Password (first 10 chars): {password[:10]}...")
    
    print("\nTesting connection...")
    try:
        conn = await asyncpg.connect(db_url)
        version = await conn.fetchval('SELECT version()')
        print("✅ Connection successful!")
        print(f"PostgreSQL: {version[:60]}...")
        
        # Check pgvector
        has_vector = await conn.fetchval("""
            SELECT EXISTS(
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            );
        """)
        if has_vector:
            print("✅ pgvector extension is installed")
        else:
            print("⚠️  pgvector extension not found")
        
        await conn.close()
        return True
    except asyncpg.InvalidPasswordError as e:
        print(f"\n❌ Password authentication failed")
        print("\nPossible causes:")
        print("1. Password in .env doesn't match Neon database")
        print("2. Password was reset in Neon but .env wasn't updated")
        print("3. Using wrong connection string (different user/database)")
        print("\nSolution:")
        print("1. Go to Neon dashboard → Connection Details")
        print("2. Copy the ENTIRE connection string")
        print("3. Update DATABASE_URL in .env file")
        return False
    except Exception as e:
        print(f"\n❌ Connection error: {type(e).__name__}")
        print(f"Error: {str(e)[:200]}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_connection())
    if not result:
        print("\n" + "="*60)
        print("Get your connection string from:")
        print("https://console.neon.tech → Your Project → Connection Details")
        print("="*60)

