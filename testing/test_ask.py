import requests

url = "http://localhost:8002/ask"
data = {
<<<<<<< HEAD
    "user_query": "ANTITRUST LAWS",
    "user_language": "en",
    "top_k": 1
}

response = requests.post(url, json=data)
print(response.json())
=======
    "user_query": "The invoices must contain which information?",
    "user_language": "en",
    "top_k": 1,
    "user_id": "123",
    "session_id": "manual"

}

response = requests.post(url, json=data)
print(response.json())
>>>>>>> f32d650 (Proper project root: move contents to repo root)
