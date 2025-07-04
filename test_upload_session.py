import requests
from pathlib import Path

url = "http://localhost:8000/upload"

# Option 2: Upload using manual path
manual_path = str(file_path.resolve())

data = {
    "user_id": "123",
    "session": "true",
    "manual_paths": manual_path
}

response_manual = requests.post(url, data=data)

print("\nOption 2 (Manual path):")
print("Status Code:", response_manual.status_code)
print("Response:", response_manual.json())