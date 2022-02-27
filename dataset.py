import pandas as pd
import torch
from transformers import BertTokenizer, XLNetTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler

class DatasetUtils:
    @staticmethod
    def get_loader(dataset, batch_size=32, train=True):
        
        if train:
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
        else:
            loader = DataLoader(dataset, batch_size)
        return loader

    @staticmethod
    def get_tokenizer(name='xlnet'):
        assert name.lower() in ['bert', 'xlnet']
        print(f"===== Create {name.upper()} Tokenizer =====")

        if name == 'bert':
            tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased'
            )
        elif name == 'xlnet':
            tokenizer = XLNetTokenizer.from_pretrained(
                'xlnet-base-cased',
                do_lower_case = True,
                remove_space= True
                )

        return tokenizer

class MoodyLyrics(Dataset):
    def __init__(self, root_dir, tokenizer, max_len=128, mode='train'):
        
        assert mode in ['train', 'val']
        if mode == 'train':
            self.df = pd.read_csv(f'{root_dir}/train.csv')
            print('* Train size:', len(self.df))
        else: 
            self.df = pd.read_csv(f'{root_dir}/test.csv')
            print('* Validation size:', len(self.df))

        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # label to index
        self.label_map = {
            'happy': 0,
            'angry': 1,
            'sad': 2,
            'relaxed': 3
        }

    def label_value_counts(self):
        return self.df.mood.value_counts()

    def get_class_weights(self):
        label_vc = self.label_value_counts()
        
        # happy, angry, sad, relaxed 4 classes
        _class_weights = [0] * len(label_vc) 
        for labelname, _count in label_vc.items():
            classindex = self.label_map[labelname]
            _class_weights[classindex] = 10000/_count
        return _class_weights
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        lyric = self.df.lyric.iloc[idx]
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
            'targets': torch.tensor(label, dtype=torch.long)
        }

if __name__ == '__main__':

    # BERT example
    tokenizer = DatasetUtils.get_tokenizer('bert')
    dataset = MoodyLyrics(
        root_dir='MoodylyricDataset',
        tokenizer=tokenizer,
        mode='train',
        max_len=256
    )
    train_loader = DatasetUtils.get_loader(dataset, 1)

    # XLNET example
    tokenizer = DatasetUtils.get_tokenizer('xlnet')
    dataset = MoodyLyrics(
        root_dir='MoodylyricDataset',
        tokenizer=tokenizer,
        mode='train',
        max_len=1024
    )
    train_loader = DatasetUtils.get_loader(dataset, 1, train=True)
