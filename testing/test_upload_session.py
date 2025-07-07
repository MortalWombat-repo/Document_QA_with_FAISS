import requests
from pathlib import Path

url = "http://localhost:8001/upload"
file_path = Path("/home/newuser/Downloads/contract_template.pdf")

if not file_path.exists():
    print(f"Error: File {file_path} does not exist on the host.")
    exit(1)

data = {
    "user_id": "123",
    "session": "true"
}

with open(file_path, "rb") as f:
    files = [("files", (file_path.name, f, "application/pdf"))]
    response = requests.post(url, data=data, files=files)

print("Status Code:", response.status_code)
print("Response:", response.json())
