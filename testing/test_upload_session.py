import requests
from pathlib import Path

url = "http://localhost:8001/upload"
<<<<<<< HEAD
file_path = Path("/home/newuser/Downloads/contract_template.pdf")
=======
file_path = Path("/workspaces/69204832/Document_QA_with_FAISS-main/test_folder/contract_sample.pdf")
>>>>>>> f32d650 (Proper project root: move contents to repo root)

if not file_path.exists():
    print(f"Error: File {file_path} does not exist on the host.")
    exit(1)

data = {
<<<<<<< HEAD
    "user_id": "123",
=======
>>>>>>> f32d650 (Proper project root: move contents to repo root)
    "session": "true"
}

with open(file_path, "rb") as f:
    files = [("files", (file_path.name, f, "application/pdf"))]
    response = requests.post(url, data=data, files=files)

print("Status Code:", response.status_code)
<<<<<<< HEAD
print("Response:", response.json())
=======
print("Response:", response.json())
>>>>>>> f32d650 (Proper project root: move contents to repo root)
