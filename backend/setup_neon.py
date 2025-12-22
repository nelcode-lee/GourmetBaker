#!/usr/bin/env python3
"""
Interactive script to set up Neon database connection
"""
import os
import sys

def update_env_file(db_url):
    """Update DATABASE_URL in .env file"""
    env_file = '.env'
    
    if not os.path.exists(env_file):
        print(f"❌ {env_file} file not found!")
        return False
    
    # Read current .env
    with open(env_file, 'r') as f:
        lines = f.readlines()
    
    # Update or add DATABASE_URL
    updated = False
    new_lines = []
    for line in lines:
        if line.strip().startswith('DATABASE_URL='):
            new_lines.append(f'DATABASE_URL={db_url}\n')
            updated = True
        else:
            new_lines.append(line)
    
    if not updated:
        # Add at the end
        new_lines.append(f'\nDATABASE_URL={db_url}\n')
    
    # Write back
    with open(env_file, 'w') as f:
        f.writelines(new_lines)
    
    return True

def main():
    print("="*60)
    print("Neon Database Setup")
    print("="*60)
    print()
    print("To get your connection string:")
    print("1. Go to https://neon.tech and sign up/login")
    print("2. Create a new project")
    print("3. Enable 'vector' extension in Extensions tab")
    print("4. Go to Connection Details and copy the connection string")
    print()
    print("The connection string looks like:")
    print("postgresql://user:pass@ep-xxx.region.aws.neon.tech/dbname?sslmode=require")
    print()
    
    db_url = input("Paste your Neon DATABASE_URL here: ").strip()
    
    if not db_url:
        print("❌ No connection string provided. Exiting.")
        sys.exit(1)
    
    if not db_url.startswith('postgresql://'):
        print("⚠️  Warning: Connection string should start with 'postgresql://'")
        confirm = input("Continue anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            sys.exit(1)
    
    if update_env_file(db_url):
        print("✅ DATABASE_URL updated in .env file!")
        print()
        print("Next step: Test the connection")
        print("Run: python3 setup_database.py")
    else:
        print("❌ Failed to update .env file")
        sys.exit(1)

if __name__ == "__main__":
    main()

