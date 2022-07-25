import os
import sys
import re
import six
import math
import lmdb
import torch

from natsort import natsorted
import itertools
from PIL import Image
from copy import deepcopy
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms


#배치의 데이터 비율 균형 설정을 위한 class
class Batch_Balanced_Dataset(object):

    def __init__(self, opt):  #class 내부에서 사용하면 객체를 생성할 때 처리한 내용을 작성 가능
        """
        Modulate the data ratio in the batch. 
       # batch 의 데이터 비율 변경
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
       # 예를 들어 select_data가 "MJ-ST"이고 batch_ratio가 "0.5-0.5"일 때
       # 배치의 50%는 MJ로 채워지고 나머지 50%는 ST로 채워집니다.

        opt.batch_ratio: 하나의 batch에 포함된 서로 다른 데이터 집합의 비율
        opt.total_data_usage_ratio: 각 데이터 및 이 데이터셋의 사용량은 1(100%)입니다.
        """
        self.opt = opt  # 클래스 변수 생성 
        log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')  #파일을 추가 모드('a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')   #log_dataset.txt 파일에 추가해줌 
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n') 
        assert len(opt.select_data) == len(opt.batch_ratio)   #assert: 원하는 조건의 변수 값을 보증받을 때까지 assert로 테스트(길이가 같을 때)

        # 각 datalogger에 collate 함수를 적용하여 batch 전체를 직접 출력합니다
        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):  #select data,배치 안의 데이터 비율 하나씩 불어오기(batch ratio)
            #opt.batch_ratio: 하나의 batch에 포함된 서로 다른 데이터 집합의 비율
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)   #배치 사이즈에 비율을 곱해서 더 큰 값을 사용 
            print(dashed_line)   #-------------
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d]) #hierarchical_dataset: 계층 군집 분석
            total_number_dataset = len(_dataset) # 현재 데이터 세트에 포함된 이미지 수
            log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            총 데이터 수는 opt.total_data_usage_ratio로 수정할 수 있습니다.
            ex) opt.total_data_data_details = 1은 100% 사용량을 나타내고 0.2는 20% 사용량을 나타냅니다.
            본 문서의 4.2 섹션을 참조하십시오.
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio)) #사용된 비율
            if opt.fix_dataset_num != -1: number_dataset = opt.fix_dataset_num
            dataset_split = [number_dataset, total_number_dataset - number_dataset] # List[int] e.g. [50, 50]
            indices = range(total_number_dataset)
            
            # accumulate함수： _accumulate([1,2,3,4,5]) --> 1 3 6 10 15  (누적 결과 반환)

            #Subset은 indices에 따라 하나의 데이터 세트의 부분 집합을 취하는 것이고, 
            #indice는 opt.total_data_usage_ratio에 따라 값을 취하는 것
            
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(  #파이토치로 데이터 불러오기
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate, pin_memory=False, drop_last=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    #batch 가져오기
    def get_batch(self, meta_target_index=-1, no_pseudo=False): # meta_target_index가 지정되면 meta_target_index 데이터 집합은 무시됩니다.
        #빈 리스트 생성
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):  #enumerate: 반복문 사용시 몇 번째(i) 반복인지 알 수 있음
            if i == meta_target_index: continue
            # 더미 태그 데이터 세트를 샘플링하지 않고, 현재 더미 태그 데이터 세트를 포함하도록 요청하면 건너뜁니다.
            if i == len(self.dataloader_iter_list) - 1 and no_pseudo and self.has_pseudo_label_dataset(): continue 
            try:
                image, text = data_loader_iter.next()  #데이터 넘어가면서 
                balanced_batch_images.append(image)  #빈리스트에 이미지 추가
                balanced_batch_texts += text         #빈 리스트에 텍스트 추가
                
            #데이터셋 이미지 수가 부족하면 반복기를 다시 구축하여 훈련합니다.
            except StopIteration:  
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            #값 오류 넘어가기
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)  # torch.cat: 한 줄로 텐서 쌓기(dim=0)

        return balanced_batch_images, balanced_batch_texts
    
    def get_meta_test_batch(self, meta_target_index=-1): # meta_target_index가 지정되면 meta_target_index 데이터 집합은 무시됩니다
        
        # target domain 정하기
        if meta_target_index == self.opt.source_num:   #source_num = 4
            assert len(self.data_loader_list) == self.opt.source_num + 1, 'There is no target dataset'
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):  #data_loader_iter: 5개 도메인, 
            if i == meta_target_index:  #지정한 target domain과 같을 시
                try:
                    image, text = data_loader_iter.next()
                    balanced_batch_images.append(image)
                    balanced_batch_texts += text
                except StopIteration: # 데이터셋 이미지 수가 부족하면 반복기를 다시 구축하여 훈련합니다
                    self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                    image, text = self.dataloader_iter_list[i].next()
                    balanced_batch_images.append(image)
                    balanced_batch_texts += text
                except ValueError:
                    pass
        # print(balanced_batch_images[0].shape)
        balanced_batch_images = torch.cat(balanced_batch_images, 0)  # torch.cat: 한 줄로 텐서 쌓기(dim=0)

        return balanced_batch_images, balanced_batch_texts
    
    #target domain dataset 더하기
    def add_target_domain_dataset(self, dataset, opt):
        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD) #***  imgH=32, imgW=100, keep_ratio_with_pad=False
        avg_batch_size = opt.batch_size // opt.source_num   #평균 배치 사이즈 96/4 = 24
        batch_size = len(dataset) if len(dataset) <= avg_batch_size else avg_batch_size  #평균 배치 사이즈와 비교(평균보다 크지 않도록)
        self_training_loader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size,
                        shuffle=True,  # 'True' to check training progress with validation function. 
                                          # TRUE 검증 기능으로 교육 진행 상황을 확인 
                        num_workers=int(opt.workers), pin_memory=False, collate_fn=_AlignCollate, drop_last=True)
        if self.has_pseudo_label_dataset():
            self.data_loader_list[opt.source_num] = self_training_loader
            self.dataloader_iter_list[opt.source_num] = (iter(self_training_loader))
        else:
            self.data_loader_list.append(self_training_loader)
            self.dataloader_iter_list.append(iter(self_training_loader))
    
    #  데이터 셋에 pseudo 라벨 추가(위의 방법이랑 유사)
    def add_pseudo_label_dataset(self, dataset, opt):   
        avg_batch_size = opt.batch_size // opt.source_num  #평균 배치 사이즈 96/4 = 24
        batch_size = len(dataset) if len(dataset) <= avg_batch_size else avg_batch_size
        self_training_loader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size,
                        shuffle=True,  # validation function를 통해 교육 진행 상황을 점검하는 '참'.
                        num_workers=int(opt.workers), pin_memory=False, collate_fn=self_training_collate)
        if self.has_pseudo_label_dataset():
            self.data_loader_list[opt.source_num] = self_training_loader
            self.dataloader_iter_list[opt.source_num] = (iter(self_training_loader))
        else:
            self.data_loader_list.append(self_training_loader)
            self.dataloader_iter_list.append(iter(self_training_loader))
       
    # opt.source_num '+ 1' 부분 제외하고 위와 동일한 코드  - opt.source_num 가 뭔지 찾아보기
    def add_residual_pseudo_label_dataset(self, dataset, opt):
        avg_batch_size = opt.batch_size // opt.source_num
        batch_size = len(dataset) if len(dataset) <= avg_batch_size else avg_batch_size
        self_training_loader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size,
                        shuffle=True,  # validation function를 통해 교육 진행 상황을 점검하는 '참'.
                        num_workers=int(opt.workers), pin_memory=False, collate_fn=self_training_collate)
        if self.has_residual_pseudo_label_dataset():
            self.data_loader_list[opt.source_num + 1] = self_training_loader
            self.dataloader_iter_list[opt.source_num + 1] = (iter(self_training_loader))
        else:
            self.data_loader_list.append(self_training_loader)
            self.dataloader_iter_list.append(iter(self_training_loader))

    def has_pseudo_label_dataset(self):
        return True if len(self.data_loader_list) > self.opt.source_num else False

    def has_residual_pseudo_label_dataset(self):
        return True if len(self.data_loader_list) > self.opt.source_num + 1 else False
#--------------------------------------------------------------------------------------------------------------------Class Batch_Balanced_Dataset

#batch sampling
class Batch_Balanced_Sampler(object):
    def __init__(self, dataset_len, batch_size):
        dataset_len.insert(0,0)
        self.dataset_len = dataset_len
        self.start_index = list(itertools.accumulate(self.dataset_len))[:-1]
        self.batch_size = batch_size  # 각 하위 데이터 집합의 batchsize
        self.counter = 0

    def __len__(self):
        return self.dataset_len

    def __iter__(self):
        data_index = []
        while True:
            for i in range(len(self.start_index)):
                data_index.extend([self.start_index[i] + (self.counter * self.batch_size + j) % self.dataset_len[i + 1] for j in range(self.batch_size)])
            yield data_index
            data_index = []
            self.counter += 1
        
#--------------------------------------------------------------------------------------------------------------------Class Batch_Balanced_Sampler
#위에랑 같은 Class..?+ a (hierarchical_dataset,,LmdbDataset,RawDataset,ResizeNormalize,self_training, tersor2im, ...)

class Batch_Balanced_Dataset0(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.

        opt.batch_ratio: 하나의 batch에 포함된 서로 다른 데이터 집합의 비율
        opt.total_data_usage_ratio: 예각 데이터 및 이 데이터셋의 사용량은 1(100%)입니다.
        """
        self.opt = opt
        log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        # 각 dataloader에 collate 함수를 적용하여 batch 전체를 직접 출력합니다.
        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        self.batch_size_list = []
        Total_batch_size = 0

        self.dataset_list = []
        self.dataset_len_list = []

        self.pseudo_dataloader = None
        self.pseudo_batch_size = -1

        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset) # 현재 데이터 세트에 포함된 이미지 수
            
            
            log.write(_dataset_log)

            """
            총 데이터 수는 opt.total_data_usage_ratio로 수정할 수 있습니다.
            ex) opt.total_data_data_details = 1은 100% 사용량을 나타내고 0.2는 20% 사용량을 나타냅니다.
            본 문서의 4.2 섹션을 참조하십시오.
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio)) # 사용된 비율
            if opt.fix_dataset_num != -1: number_dataset = opt.fix_dataset_num
            dataset_split = [number_dataset, total_number_dataset - number_dataset] # List[int] e.g. [50, 50]
            indices = range(total_number_dataset)
            
            # accumulate函数： _accumulate([1,2,3,4,5]) --> 1 3 6 10 15  (누적)
            # Subset은 indices에 따라 하나의 데이터 세트의 부분 집합을 취하는 것이고, indice는 opt.total_data_usage_ratio에 따라 값을 취하는 것이다.
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            self.batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            self.dataset_list.append(_dataset)
            self.dataset_len_list.append(number_dataset)



        concatenated_dataset = ConcatDataset(self.dataset_list)
        assert len(concatenated_dataset) == sum(self.dataset_len_list)

        batch_sampler = Batch_Balanced_Sampler(self.dataset_len_list, _batch_size)
        self.data_loader = iter(torch.utils.data.DataLoader(
            concatenated_dataset,
            batch_sampler=batch_sampler,
            num_workers=int(opt.workers),
            collate_fn=_AlignCollate, pin_memory=False))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(self.batch_size_list)
        self.batch_size_list = list(map(int, self.batch_size_list))
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self, meta_target_index=-1, no_pseudo=False): # 如果指定了meta_target_index，则忽略第meta_target_index个数据集
        
        imgs, texts = next(self.data_loader)
        # 如果未指定或指定为伪标签数据集，则直接返回所有
        if meta_target_index == -1 or meta_target_index >= len(self.batch_size_list): return imgs, texts
        start_index_list = list(itertools.accumulate(self.batch_size_list))
        start_index_list.insert(0, 0)

        ret_imgs, ret_texts = [], []
        for i in range(len(self.batch_size_list)): 
            if i == meta_target_index: continue
            ret_imgs.extend(imgs[start_index_list[i] : start_index_list[i] + self.batch_size_list[i]])
            ret_texts.extend(texts[start_index_list[i] : start_index_list[i] + self.batch_size_list[i]])
        ret_imgs = torch.stack(ret_imgs, 0)
        
        # assert self.has_pseudo_label_dataset() == True, 'Pseudo label dataset can\'t be empty'
        if self.has_pseudo_label_dataset():
            try:
                psuedo_imgs, pseudo_texts = next(self.pseudo_dataloader_iter)
            except StopIteration:
                self.pseudo_dataloader_iter = iter(self.pseudo_dataloader)
                psuedo_imgs, pseudo_texts = next(self.pseudo_dataloader_iter)
            ret_imgs = torch.cat([ret_imgs, psuedo_imgs], 0)
            ret_texts += pseudo_texts

        return ret_imgs, ret_texts

    def get_meta_test_batch(self, meta_target_index=-1): # 如果指定了meta_target_index，则忽略第meta_target_index个数据集
        
        assert meta_target_index != -1, 'Meta target index should be specified'
        if meta_target_index >= len(self.batch_size_list) and self.has_pseudo_label_dataset(): 
            try:
                img, text = next(self.pseudo_dataloader_iter)
            except StopIteration:
                self.pseudo_dataloader_iter = iter(self.pseudo_dataloader)
                img, text = next(self.pseudo_dataloader_iter)

            return img, text
        
        imgs, texts = next(self.data_loader)
        start_index_list = list(itertools.accumulate(self.batch_size_list))
        start_index_list.insert(0, 0)
        ret_img, ret_text = None, None
        for i in range(len(self.batch_size_list)): 
            if i == meta_target_index:
                ret_img = imgs[start_index_list[i]:start_index_list[i] + self.batch_size_list[i]]
                ret_text = texts[start_index_list[i]:start_index_list[i] + self.batch_size_list[i]]

        return ret_img, ret_text

    def add_pseudo_label_dataset(self, dataset, opt):
        avg_batch_size = opt.batch_size // opt.source_num
        batch_size = len(dataset) if len(dataset) <= avg_batch_size else avg_batch_size
        self.pseudo_batch_size = batch_size
        self.pseudo_dataloader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size,
                        shuffle=True,  # 'True' to check training progress with validation function.
                        num_workers=int(opt.workers), pin_memory=False, collate_fn=self_training_collate)
        self.pseudo_dataloader_iter = iter(self.pseudo_dataloader)


    def has_pseudo_label_dataset(self):
        return True if self.pseudo_dataloader else False


#계층 군집 분석
def hierarchical_dataset(root, opt, select_data='/', pseudo=False):  
    """ select_data='/' contains all sub-directory of root directory   #select_data=filename'은(는) 루트 디렉토리의 모든 하위 디렉토리를 포함합니다.
            """
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root+'/', followlinks=True):
        print(dirpath, dirnames, filenames)
        if not dirnames:   # dirnames가 비어 있을 때, 즉 현재 dirpath 아래에 lmdb 파일만 포함되어 있을 때 동작합니다.
            select_flag = False
            for selected_d in select_data: # select_data는 문자열, e.g. 'MJ', 'ST'
                
                # dirpath에 select_data가 포함되어 있으면 현재 디렉터리가 대상 디렉터리임을 나타냅니다. select_flag는 True를 설정합니다.
                if selected_d in dirpath: 
                    select_flag = True
                    break

            if select_flag:
                dataset = LmdbDataset(dirpath, opt, pseudo=pseudo)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    # 모든 데이터 세트를 결합합니다. 예를 들어, dataset_list에는 MJ_train, MJ_valid, MJ_test가 포함되어 있습니다.
    concatenated_dataset = ConcatDataset(dataset_list) 

    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):

    def __init__(self, root, opt, pseudo=False):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)  #lmdb 열기
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)  #프로그램을 정상 종료

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            if self.opt.data_filtering_off:
                # 필터링 없이 빠른 검사 또는 벤치마크 평가 가능
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                # use --data_filtering_off and only evaluate on alphabets and digits.
                # 특수 문자 레이블을 가진 IC15-2077 & CUTE 데이터 세트를 평가하려면
                # --data_data_digits_off를 사용하고 알파벳과 숫자에 대해서만 평가합니다.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    if self.opt.pseudo_dataset_num != -1 and pseudo and index > self.opt.pseudo_dataset_num:
                        break
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')
                    # print(label)

                    if len(label) > self.opt.batch_max_length:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue

                    # 기본적으로 opt.character에 없는 문자가 포함된 이미지는 필터링됩니다.
                    # 이 필터링 대신 utils.py의 'opt.character'에 [UNK] 토큰을 추가할 수 있습니다.
                    out_of_char = f'[^{self.opt.character}]'
                    # if re.search(out_of_char, label.lower()): # opt.char에 소문자가 포함되어 있지 않기 때문에 모든 번호판은 필터링됩니다.
                    if re.search(out_of_char, label):
                        continue

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

            # if not self.opt.sensitive:
            #     label = label.lower()

            # 영숫자로만 교육하고 평가합니다.(또는 train.py 에서 미리 정의된 문자 집합)
            out_of_char = f'[^{self.opt.character}]'
            label = re.sub(out_of_char, '', label)

        return (img, label)


class RawDataset(Dataset):    

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):  #root 경로 파일 탐색(dirpath, dirnames, filenames 가져오기)-lmdb?
            for name in filenames:
                _, ext = os.path.splitext(name) #파일이름에서 확장자만 ext에 저장
                ext = ext.lower()  #소문자화
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':   #파일의 확장자(ext)가 이미지 확장자인 경우 
                    self.image_path_list.append(os.path.join(dirpath, name))   # 경로와 파일 이름을 조인해서 image_path_list에 추가

        self.image_path_list = natsorted(self.image_path_list)  #정렬
        self.nSamples = len(self.image_path_list)  # 이미지 파일(sample) 갯수 

    def __len__(self):  #샘플의 길이 
        return self.nSamples

    def __getitem__(self, index):  #image_path_list[index]로 불러오기 값 가져오기

        try:
            #image open
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # color image 인 경우
            else:
                img = Image.open(self.image_path_list[index]).convert('L')  # 아닌 경우

        except IOError:  #손상된 이미지 처리 
            print(f'Corrupted image for {index}')   #손상된 이미지의 index 번호를 출력  
            # 손상된 이미지에 대해 더미 이미지 및 더미 레이블을 만든다.
            # image.new 이미지 크기 재설정 (https://www.geeksforgeeks.org/python-pil-image-new-method/)
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH)) 
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])


#크기 조정 정규화
class ResizeNormalize(object):    

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()  #tensor로 변환

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

# padding 
class NormalizePAD(object):  

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()  # image shape
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)  #0으로 채움
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object): 

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1  #color image 이면 채널 3 -> 3차원
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)  

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels

def self_training_collate(batch):
    imgs, labels = [], []
    for img, label in batch:
        imgs.append(img)
        labels.append(label)
    
    return torch.stack(imgs), labels  # 새로운 차원으로 주어진 텐서들을 붙임(image) 3차원

# self training을 위한 데이터셋
class SelfTrainingDataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels
    
    def __getitem__(self, index):
        return self.imgs[index], self.labels[index]

    def __len__(self):
        assert len(self.imgs) == len(self.labels)
        return len(self.imgs)



def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0   #최종 이미지 shape?
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
#--------------------------------------------------------------------------------------------------------------------Class
