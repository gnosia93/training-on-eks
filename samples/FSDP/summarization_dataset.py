import pandas as pd
from datasets import Dataset, DatasetDict
from datasets import load_dataset
import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import numpy as np
import torch
#from torch.utils.data import Dataset, DataLoader

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

class wikihow(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, print_text=False):
        # self.dataset =  load_dataset('wikihow', 'all', data_dir='data/wikihow/', split=type_path)
        
        # 1. 절대 경로 설정
        current_path = os.getcwd()
        # data/wikihow/ 폴더의 절대 경로 생성
        data_path = current_path + "/data/wikihow/"
    
        # 1. Pandas를 사용하여 CSV 로드 (파싱 에러 자동 처리)
        # lineterminator 설정을 통해 문장 내 줄바꿈 이슈를 방지합니다.
        df_train = pd.read_csv(data_path + 'wikihowAll.csv', on_bad_lines='skip', engine='python')
        df_val = pd.read_csv(data_path + 'wikihowSep.csv', on_bad_lines='skip', engine='python')
        
        # 2. Pandas DataFrame을 Hugging Face Dataset 객체로 변환
        full_dataset = DatasetDict({
            'train': Dataset.from_pandas(df_train),
            'validation': Dataset.from_pandas(df_val)
        }) 
        self.dataset = full_dataset[type_path] 

         # --- 이 지점에 방어 코드 추가 ---
        actual_size = len(self.dataset)
        num_samples = min(num_samples, actual_size)
        
        if num_samples:
            self.dataset = self.dataset.select(list(range(0, num_samples)))
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text

    def __len__(self):
        return self.dataset.shape[0]

    def clean_text(self, text):
        # 만약 text가 리스트로 들어온다면 첫 번째 요소를 추출 (중요)
        if isinstance(text, list):
            text = text[0] if len(text) > 0 else ""
    
        # 이제 문자열이 확실하므로 replace 사용 가능
        text = text.replace('Example of text:', '')
        text = text.replace('Example of Summary:', '')
        text = text.replace('\n','')
        text = text.replace('``', '')
        text = text.replace('"', '')

        return text


    def convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)

        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch['text']))
#         input_ = self.clean_text(example_batch['text']) + " </s>"
#         target_ = self.clean_text(example_batch['headline']) + " </s>"

        input_ = self.clean_text(example_batch['text'])
        target_ = self.clean_text(example_batch['headline'])

        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length,
                                                     padding='max_length', truncation=True, return_tensors="pt")

        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length,
                                                     padding='max_length', truncation=True, return_tensors="pt")


        return source, targets

    def __getitem__(self, index):
        # 하드 코딩 ... [rank0]: IndexError: index 150 is out of bounds for dimension 0 with size 150 ....
        # 실제 데이터셋 크기에 맞춰 인덱스를 순환시킴 (에러 방지 + 데이터 활용)
        actual_size = len(self.dataset)

        # 1. index가 리스트인 경우 (DataLoader가 배치를 한꺼번에 요청할 때)
        if isinstance(index, list):
            safe_index = [i % actual_size for i in index]
        # 2. index가 단일 숫자인 경우
        else:
            safe_index = index % actual_size
            
        source, targets = self.convert_to_features(self.dataset[safe_index])

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

def get_dataset(tokenizer, type_path, num_samples, args):
      return wikihow(tokenizer=tokenizer, type_path=type_path, num_samples=num_samples,  input_length=max_input_length,
                        output_length=max_output_length)
