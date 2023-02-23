import requests
import json

# param = {'Action':'RUN'}
res = requests.post('http://127.0.0.1:5000/check')
print(res.text)