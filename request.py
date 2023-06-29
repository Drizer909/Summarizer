import requests

url  = 'http://localhost:5000/api/summarize'
text = "I love eating them and they are good for watching TV and looking at movies! It is not too sweet. I like to transfer them to a zip lock baggie so they stay fresh so I can take my time eating them. " 

r = requests.post( url, json = { 'text' : text})

print(r.json())