import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CTCLabelConverter(object):
    """ 텍스트 레이블과 텍스트 인덱스 간에 변환 """

    def __init__(self, character):
        # character (str): 사용 가능한 문자 집합.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # 참고: 0은(는) CTCloss에 필요한 'CTCblank' 토큰용으로 예약되었습니다.
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # CTCloss에 대한 더미 '[CTCblank]' 토큰(인덱스 0)

    def encode(self, text, batch_max_length=25):
        """텍스트 레이블을 텍스트 인덱스로 변환합니다.
        input:
            text: 각 이미지의 텍스트 레이블입니다. [sig_size]
            batch_max_length: 배치에 포함된 텍스트 레이블의 최대 길이. 기본적으로 25

        output:
            text: CTLoss에 대한 텍스트 인덱스. [batch_size, batch_max_length] 
            length: 각 텍스트의 길이입니다. [sig_size]
        """
        length = [len(s) for s in text]

        # 패딩에 사용되는 지수(=0)는 CTC 손실 계산에 영향을 미치지 않습니다. 모든 label 동일 길이, 여유자리는 0으로 채움
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ 텍스트 인덱스를 텍스트 레이블로 변환합니다. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # 반복된 문자와 공백을 제거합니다.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts


class CTCLabelConverterForBaiduWarpctc(object):
    """ baidu warpctc에 대한 text-label과 text-index 사이에서 변환 """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """텍스트 레이블을 텍스트 색인으로 변환합니다.
        input:
            text: 각 이미지의 텍스트 레이블입니다. [sig_size]
        output:
            text: CTCLoss에 대한 연결된 텍스트 인덱스.
                    [sum(text_index_0)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: 각 텍스트의 길이입니다. [sig_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):   # 반복된 문자와 공백을 제거합니다.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): 사용 가능한 문자 집합.
        # [GO] 주의 디코더의 시작 토큰입니다. [s] 토큰을 사용할 수 있습니다.        
        # [GO]는 시작 토큰, 대응 인덱스 0, [s]는 종료 토큰, 대응 인덱스 1
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ 텍스트 레이블을 텍스트 색인으로 변환합니다.
        input:
            text: text labels of each image. [batch_size] 각 요소는 문자열입니다.
            batch_max_length: 배치에 포함된 텍스트 레이블의 최대 길이. 기본적으로 25

        output:
            text : 주의 디코더 입력.  [batch_size x (max_length+2)] [GO] 토큰의 경우 +1이고 [s] 토큰의 경우 +1입니다.
                text[:, 0]은 [GO] 토큰이고 텍스트는 [s] 토큰 뒤에 [GO] 토큰으로 채워집니다.
            length : [s] 토큰도 카운트하는 주의 디코더의 출력 길이입니다. [3, 7, …] [sig_size]
            attention의 경우 출력 라벨은 "03534534610000..."입니다. 여기서 0은 GO, 1은 S, S는 모두 GO로 채워집니다. 길이batch_max_length + 2로
        """
        length = [len(s) + 1 for s in text]  # 문장 끝에 [s]에 대해 +1입니다. [GO]不需要计入长度
        # batch_max_length = max(length) # 다중 모드 설정에는 허용되지 않습니다.
        batch_max_length += 1
        # 첫 번째 단계에서 [GO]에 대해 +1을 추가합니다. batch_text는 [s] 토큰 뒤에 [GO] 토큰으로 패딩됩니다. [s]之后的用[GO]也就是0补齐
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0) #길이가 batch_max_length + 2인 것과 같습니다
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]   
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device)) #length 벡터의 길이는 batch_max_length + 1입니다. 즉, [GO]는 길이를 포함하지 않습니다.

    def decode(self, text_index, length): # length는 attention에서 사용할 수 없고 CTCloss에서 사용할 수 있습니다. 포맷을 통일합니다.
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """토치의 평균을 계산합니다.손실 평균에 사용되는 텐서."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel() # 텐서 속 원소의 수를 되돌리면 loss는 1이 된다.
        v = v.data.sum() # Tensor의 모든 원소의 합을 되돌리는 것은 loss에 있어서 loss 자체의 값이다.
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
