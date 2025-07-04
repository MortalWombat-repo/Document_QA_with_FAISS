import requests

url = "http://localhost:8000/ask"
data = {
    "user_query": "What information should invoices contain?",
    "user_language": "en",
    "top_k": 1
}

response = requests.post(url, json=data)
print(response.json())