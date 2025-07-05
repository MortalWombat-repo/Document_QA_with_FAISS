import requests

url = "http://localhost:8002/ask"
data = {
    "user_query": "Koje informacije trebaju sadržavati fakture?",
    "user_language": "hr",
    "top_k": 1
}

response = requests.post(url, json=data)
print(response.json())