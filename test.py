import requests

url = "http://127.0.0.1:5000/api/orders"

order = {
    "side": "buy",
    "price": 100,
    "quantity": 5
}

response = requests.post(url, json=order)

print("Status Code:", response.status_code)
print("Response:", response.text)