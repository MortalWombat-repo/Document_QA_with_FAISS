import requests
from pathlib import Path

url = "http://localhost:8001/upload"

# Option 1: Upload via HTTP Form File
file_path = Path("/workspaces/69204832/Document_QA_with_FAISS-main/test_folder/contract.pdf")

if not file_path.exists():
    raise FileNotFoundError(f"File not found: {file_path}")

with open(file_path, "rb") as f:
    files = {"files": f}
    data = {
        "user_id": "123",
        "session": "false"
    }

    response = requests.post(url, data=data, files=files)

print("Option 1 (Upload via form):")
print("Status Code:", response.status_code)
print("Response:", response.json())
