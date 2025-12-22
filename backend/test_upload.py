#!/usr/bin/env python3
"""
Test script to diagnose document upload issues
"""
import requests
import sys
from pathlib import Path

def test_upload():
    """Test document upload endpoint"""
    url = "http://localhost:8000/api/documents/upload"
    
    # Create a test text file
    test_file = Path("/tmp/test_document.txt")
    test_file.write_text("This is a test document for upload testing.")
    
    print(f"📤 Testing upload to: {url}")
    print(f"📄 Test file: {test_file}")
    print(f"📏 File size: {test_file.stat().st_size} bytes")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'files': ('test_document.txt', f, 'text/plain')}
            response = requests.post(url, files=files)
        
        print(f"\n📊 Response Status: {response.status_code}")
        print(f"📋 Response Headers: {dict(response.headers)}")
        print(f"📝 Response Body: {response.json()}")
        
        if response.status_code == 200:
            print("\n✅ Upload successful!")
            return True
        else:
            print(f"\n❌ Upload failed with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to server. Is it running?")
        print("   Start with: cd backend && source venv/bin/activate && uvicorn app.main:app --reload")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_upload()
    sys.exit(0 if success else 1)

