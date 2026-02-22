import sys
import time
import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000/api/v1"

def test_health():
    print("\nTesting Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        print(f"✅ Health OK: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Health Failed: {e}")
        return False

def test_upload():
    print("\nTesting Upload Endpoint...")
    file_path = Path("data/documents/raw/sample_company_overview.md")
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
        
    try:
        with open(file_path, "rb") as f:
            files = {"file": ("sample_company_overview.md", f, "text/markdown")}
            response = requests.post(f"{BASE_URL}/documents/upload", files=files)
        
        response.raise_for_status()
        print(f"✅ Upload OK: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Upload Failed: {e}")
        return False

def test_list_docs():
    print("\nTesting List Documents...")
    try:
        response = requests.get(f"{BASE_URL}/documents")
        response.raise_for_status()
        docs = response.json()
        print(f"✅ List Docs OK: Found {len(docs)} documents")
        return True
    except Exception as e:
        print(f"❌ List Docs Failed: {e}")
        return False

def test_chat():
    print("\nTesting Chat Endpoint...")
    try:
        payload = {"query": "Who is the CEO of MHK Tech?"}
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Chat OK: {data['response'][:100]}...")
        if data.get("sources"):
             print(f"   Sources found: {len(data['sources'])}")
        return True
    except Exception as e:
        print(f"❌ Chat Failed: {e}")
        return False

if __name__ == "__main__":
    print("Waiting for server to be ready...")
    # Retries for health check
    for i in range(5):
        if test_health():
            break
        time.sleep(2)
    else:
        print("Server not ready, aborting tests.")
        sys.exit(1)
        
    success = True
    success &= test_upload()
    success &= test_list_docs()
    success &= test_chat()
    
    if success:
        print("\n✅ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)
