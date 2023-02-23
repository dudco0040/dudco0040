import requests
import json

param = {'Action':'RUN'}
res = requests.post('http://127.0.0.1:5000/SmartSensorAI', data=json.dumps(param))
print(res.text)  # 함수에서 return 해준 값을 프린트