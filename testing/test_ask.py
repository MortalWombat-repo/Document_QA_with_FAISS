import requests

url = "http://localhost:8002/ask"
data = {
    "user_query": "ANTITRUST LAWS",
    "user_language": "en",
    "top_k": 1
}

response = requests.post(url, json=data)
print(response.json())