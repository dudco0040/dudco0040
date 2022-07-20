# LMDB 데이터셋 생성 코드 

""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """
''' # 수정된 CRNN version 링크 - 좀더 이해하기 좋은듯 (전체적인 내용은 동일한 것으로 보임)


python create_lmdb_dataset.py --inputPath data/ --gtFile data/gt.txt --outputPath result/ 그중，
inputPath와 gtFile의 경로를 실제 그림 경로로 결합합니다
label의 공간을 \t$sed-s's//\t/g' label.txt > label_t.txt로 바꿉니다
'''

import fire
import os
import lmdb   #lmdb 패키지 사용
import cv2

import numpy as np


def checkImageIsValid(imageBin):    #이미지 검증 
    if imageBin is None:  # 데이터 값이 존재하지 않는 경우(값의 부재) FALSE
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)   #np.frombuffer: 바이너리 파일 읽어오기
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)   #cv2.imdecode: 이미지 불러오기
    imgH, imgW = img.shape[0], img.shape[1]   #image shape 이미지 사이즈 결정
    if imgH * imgW == 0:  #이미지 사이즈가 0인 경우 FALSE
        return False
    return True


#cache : 캐시는 컴퓨터 과학에서 데이터나 값을 미리 복사해 놓는 임시 장소를 가리킨다.
#write cache 일괄적으로 DB에 업데이트 되는 DB쓰기 로드를 처리?
#DB관련 내용 

#함수 정의
def writeCache(env, cache):    
    with env.begin(write=True) as txn:  # lmdb 데이터베이스 내용 조회
        for k, v in cache.items():  #캐시에 저장된 key,value값 순서대로 가져와서 
            txn.put(k, v)  #lmdb 데이터 추가 (put)

            
#데이터 셋 생성하기 
def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation. #학습과 테스트를 위한 LMDB 데이터 셋을 생성
    ARGS:
        inputPath  : input folder path where starts imagePath   # 입력데이터가 있는 경로
        outputPath : LMDB output path   #LMDB로 변환되어 출력될 경로 
        gtFile     : list of image path and label   #이미지 경로 및 라벨 목록 (.txt)
        checkValid : if true, check the validity of every image   #TRUE: 모든 이미지의 유효성을 확인한다. 
    """
    os.makedirs(outputPath, exist_ok=True)   # make dir 디렉터리 생성 - LMDB 출력 경로 생성, 
    #exist_ok : 해당 디렉터리가 존재할 경우는 에러 발생없이 그냥 넘어가고, 없을 경우 디렉터리를 생성 해줌
    
    #***
    env = lmdb.open(outputPath, map_size=1099511627776)   #map_size로 설정한 크기만큼 data.mdb파일이 생성됨,너무 큰 경우 안될 수 도 있음 (참고 링크: https://beok.tistory.com/99)
    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as data:  #이미지 경로 및 라벨 목록 (.txt파일) 불러오기
        datalist = data.readlines()   #한 줄씩 읽어오기

    nSamples = len(datalist)    #데이터 개수(샘플 개수)
    for i in range(nSamples):  
        imagePath, label = datalist[i].strip('\n').split('\t')  #tab기분으로 문자열 분리 -> 이미지 경로, label 값으로 분리 
        imagePath = os.path.join(inputPath, imagePath)  #input 경로/이미지 경로(label)

        # # 영문, 숫자 데이터만 사용 
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        if not os.path.exists(imagePath):  #이미지 경로가 존재하지 않으면, 문자 출력 후 이어서 계속 
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:  # rb : 바이크 형식으로 읽기 (경로 시 자주 사용?)
            imageBin = f.read()  
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:   
                print('error occured', i)    #오류 발생시
                with open(outputPath + '/error_image_log.txt', 'a') as log:  #에러 발생 기록을 저장 (i번째 datalist에서 에러 발생)
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt  #str_encode() 문자열의 부호화된 내부 코드 값(인코딩)을 제공: 문자열을 변경(내부 코드 값으로 변경)  #count로 나눴을때 나머지값을 key에 저장 -- ?  count 
        labelKey = 'label-%09d'.encode() % cnt

        #캐시에 저장
        cache[imageKey] = imageBin  #이미지 
        cache[labelKey] = label.encode()   #라벨명 인코딩(문자 인식)

        if cnt % 1000 == 0:  #10000번에 한번씩? 캐시 작성
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))  #written된 cnt/nSamples 출력
        cnt += 1  #1씩 cnt 증감
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()  #샘플의 번호 저장
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)  #n번 샘플의 데이터셋 생성 완료 출력


if __name__ == '__main__':
    fire.Fire(createDataset)   # 파라미터 값 :inputPath, gtFile, outputPath 
    
   
    #실행 : python create_lmdb_dataset.py --inputPath data/ --gtFile data/gt.txt --outputPath result/