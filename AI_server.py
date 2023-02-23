import json
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'SmartSensor AI!'

@app.route('/check',methods=['POST'])  # Health Check
def check():
    dict = {"state":"normal"}
    return jsonify(dict)

@app.route('/SmartSensorAI/',methods=['POST'])  # RUN
def get_data():
    action = json.loads(request.get_data()) # 파이썬 데이터 형식으로 변환해서 가져오기 {Action:RUN}
    print(action)

    if action['Action'] == "RUN":
        # Information Format
        dict = [{'alert': 'T01',
                'room': 'R01',
                'error': 'E01',
                'state': 'S01',
                'etc': 'etc'},
                {'alert': 'T01',
                 'room': 'R01',
                 'error': 'E01',
                 'state': 'S01',
                 'etc': 'etc'},
                {'alert': 'T01',
                 'room': 'R01',
                 'error': 'E01',
                 'state': 'S01',
                 'etc': 'etc'}
        ]
        return jsonify(dict)
    else:
        return "None"
# key의 value가 RUN인 경우, Information Format을 반환

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

