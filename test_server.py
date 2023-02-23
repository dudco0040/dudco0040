import argparse
import pandas as pd
from hoonpyutils import UtilsCommon as utils
from tqdm import tqdm
import json
from flask import Flask, request, jsonify



def run_flask_restfull_api(args):
    app = Flask(__name__)

    #df = args
    ini = utils.get_ini_parameters(args.ini_file)
    args.threshold_temp01 = int(ini['R01']['r01_threshold_temp'])
    args.threshold_humi01 = int(ini['R01']['r01_threshold_Humi'])

    print(args.threshold_temp01)
    print(args.threshold_humi01)
    df = args.data
    print(df)
    print("ok")

    @app.route('/')
    def hello_world():
        return 'SmartSensor AI!'

    @app.route('/check', methods=['POST'])  # Health Check
    def check():
        dict = {"state": "normal"}
        return jsonify(dict)

    @app.route('/SmartSensorAI', methods=['POST'])
    def sensor():
        action = json.loads(request.get_data())  # 파이썬 데이터 형식으로 변환해서 가져오기 {Action:RUN}
        print(action)

        #return action #
        #아래부터 고치면 되는데, flask에서 다른 변수 불러오는게 안된다.면? 어떤 식으로 사용할 수 있는지..
        #그리고 불러온 값으로 threshold 값도 사용하는 코드로 바꿔야함


        if action['Action'] == "RUN":
            print("----------------")
            print(df)
            #dict = "good~"
            #return dict
            #
            dict_list = []
            r = df['M_ID'][0]  # room에 따라 파일이 따로 저장되므로 M_ID의 값은 모두 같은 값.
            room = "R0{}".format(r)

            print('room',room)
            for temp in df['Temp']:  # 임의로 컬럼명은 temp(test_df)로 지정 -> 컬럼명도 추후에 ini 파일로 설정해서 사용해도 좋을 듯 .
                print(temp)
                if temp >= 30:
                    dict = {'room': room,
                            'error': 'E09',  # 온도 센서
                            'state': 'S01',  # 높다
                            'etc': 'etc'}
                    print(dict)
                    dict_list.append(dict)

            for humi in df['Huni']:  # csv 파일 만들때 오타
                if humi >= 31:
                    dict = {'room': room,
                            'error': 'E10',  # 습도 센서
                            'state': 'S01',  # 높다
                            'etc': 'etc'}
                    dict_list.append(dict)
            print(dict_list)  #여기까지 다 실행됐는데 return만 안돼

            return jsonify(dict_list)
        else:
            return "None"


    app.run(host="0.0.0.0", port=5000)




def load_data(args):
    PATH = args.path
    args.data = pd.read_csv(PATH+"R01_test_df.csv")   # 임의 데이터로 실행(수정)
    #print(args.data.head())
    return args


    # def sensor(df, args):
    #     # room에 대한 정보는 파일명에 기재되어 있음 -> 아마두~
    #     # 파일명으로 match해서 room의 위치를 파악하는 것은 해보는 걸로 -> 파일이 1일씩 저장되는지 또한..
    #     # 아래 참고
    #     # for f in file_list:
    #     #     if fnmatch.fnmatch(f, "*_sBD1.csv"):
    #     #         normal_tag1.append(f)
    #     #
    #     #     elif fnmatch.fnmatch(f, "*_sBD2.csv"):
    #     #         normal_tag2.append(f)
    #     #
    #     #     elif fnmatch.fnmatch(f, "*_sBD1_abnormal.csv"):
    #     #         abnormal_tag1.append(f)
    #     #     elif fnmatch.fnmatch(f, "*_sBD2_abnormal.csv"):
    #     #         abnormal_tag2.append(f)
    #     print(df)
    #     print(args.threshold_temp01)
    #     # run을 요청하면, DB에 접속해서 데이터를 가져온다. -> .csv 파일로 저장된 걸 다운로드?(1일치?) -> 그러면 tag data와 차이가 뭐지 ..
    #     # -> DB에는 실시간으로 값이 저장되는 것? -> 그걸 가져와서 쓰면 우리도 실시간으로 받아오는 것처럼 쓸 수 있는 것인가?
    #     # 요청 기준 시점으로 값을 쌓아서 알려주면 ? 그때 테이블을 가져와서 사용 할수도 있고, 아니면 내가 DB를 조회하면서 적절한 수치가 쌓이면 사용 -> 어찌 됐건 테이블 형태로 받아온다.
    #
    #     dict_list = []
    #     room = df['M_ID'][0]  # room에 따라 파일이 따로 저장되므로 M_ID의 값은 모두 같은 값.
    #     print()
    #
    #     for temp in df['Temp']:  # 임의로 컬럼명은 temp(test_df)로 지정 -> 컬럼명도 추후에 ini 파일로 설정해서 사용해도 좋을 듯 .
    #         if temp >= 30:
    #             dict = {'room': room,
    #                     'error': 'E09',  # 온도 센서
    #                     'state': 'S01',  # 높다
    #                     'etc': 'etc'}
    #             dict_list.append(dict)
    #
    #     for humi in df['Huni']:  # csv 파일 만들때 오타
    #         if humi >= 31:
    #             dict = {'room': room,
    #                     'error': 'E10',  # 습도 센서
    #                     'state': 'S01',  # 높다
    #                     'etc': 'etc'}
    #             dict_list.append(dict)
    #     print(dict_list)
    #     # return dict_list#


def main(args):
    ini = utils.remove_comments_in_ini(utils.get_ini_parameters(args.ini_file))

    args.threshold_temp01 = int(ini['R01']['r01_threshold_temp'])
    args.threshold_humi01 = int(ini['R01']['r01_threshold_Humi'])

    print(args.threshold_temp01)
    print(args.threshold_humi01)

    #return args
    #sensor(load_data(),args)
    print("1")
    load_data(args)
    run_flask_restfull_api(args)


# input parameter
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ini_file", default="schema.ini")
    parser.add_argument("--path", default="./")
    args = parser.parse_args()

    return args



if __name__ == '__main__':
    main(parse_arguments())



#1차 모델에 대해서도 생각해보기