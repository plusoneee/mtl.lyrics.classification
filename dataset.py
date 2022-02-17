import numpy as np
import pandas as pd
from transformers import BertTokenizer
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader, Dataset

class DatasetUtils:

    @staticmethod
    def get_loader(dataset, batch_size=32):
        sample_weights = [0] * len(dataset)

        class_weights = dataset.get_class_weights()

        for idx, data in enumerate(dataset):
            label = data['targets'].int()
            class_weight = class_weights[label]
            sample_weights[idx] = class_weight

        sampler = WeightedRandomSampler(
            sample_weights, 
            num_samples=len(sample_weights), 
            replacement=True
        )
        loader = DataLoader(dataset, batch_size, sampler=sampler)
        return loader

    @staticmethod
    def get_tokenizer(name='bert-base-uncased'):
        tokenizer = BertTokenizer.from_pretrained(name)
        return tokenizer

class MoodyLyrics(Dataset):
    def __init__(self, filepath, tokenizer, max_len=128, mode='train'):
        assert mode in ['train', 'val']

        self.df = pd.read_csv(filepath)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode

        # label to index
        self.label_map = {
            'happy': 0,
            'angry': 1,
            'sad': 2,
            'relaxed': 3
        }

        self.len = len(self.df)
        self.train_len = int(self.len * 0.8)
        
        if self.mode == 'train':
            self.df = self.df[: self.train_len]
            print('Train size:', len(self.df))
        else: 
            self.df = self.df[self.train_len:]
            print('Validation size:', len(self.df))
            print(self.df)

    def label_value_counts(self):
        return self.df.mood.value_counts()

    def get_class_weights(self):
        label_vc = self.label_value_counts()
        # happy, angry, sad, relaxed 4 classes
        _class_weights = [0] * len(label_vc) 

        for labelname, _count in label_vc.items():
            classindex = self.label_map[labelname]
            _class_weights[classindex] = 100/_count
        return _class_weights

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        lyric = self.df.lyric.replace(r'(<.*\/>)', '').iloc[idx]
        
        _label_str = self.df.mood.iloc[idx]
        label = self.label_map[_label_str]

        inputs = self.tokenizer.encode_plus(
            text=lyric,
            text_pair=None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(label, dtype=torch.float)
        }

if __name__ == '__main__':
    
    tokenizer = DatasetUtils.get_tokenizer()
    dataset = MoodyLyrics(
        filepath='MoodylyricDataset/SegmentMoodyLyrics4Q.csv',
        tokenizer=tokenizer,
        mode='train'
    )
    train_loader = DatasetUtils.get_loader(
        dataset,
        32
    )


    '''
    happy, angry, sad, relaxed = 0, 0, 0, 0
    for data in train_loader:
        labels = data['targets'].int()
        happy += torch.sum(labels==0)
        angry += torch.sum(labels==1)
        sad += torch.sum(labels==2)
        relaxed += torch.sum(labels==3)
    print(happy, angry, sad, relaxed)
    '''