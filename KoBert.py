import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

#KoBERT 토크나이저 및 모델 불러오기

tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=2)  # 이 예제에서는 이진 분류를 수행하도록 설정

#간단한 예제 데이터 생성

data = {'text': ['너무 좋아요!', '별로에요.', '훌륭한 영화입니다.', '실망스러웠어요.'],
        'label': [1, 0, 1, 0]}
df = pd.DataFrame(data)