import requests

url = "http://localhost:8002/ask"
data = {
    "user_query": "The invoices must contain which information?",
    "user_language": "en",
    "top_k": 1,
    "user_id": "123",
    "session_id": "manual"
}

response = requests.post(url, json=data)
print(response.json())
