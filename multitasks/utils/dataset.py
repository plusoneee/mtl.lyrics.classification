import pandas as pd
import torch
from transformers import BertTokenizer, XLNetTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler

class DatasetUtils:
    @staticmethod
    def get_loader(dataset, batch_size=32, train=True, multitask=False):
        
        if not train:
            return  DataLoader(dataset, batch_size)
        
        sample_weights = [0] * len(dataset)
        class_weights = dataset.get_class_weights()
        for idx, data in enumerate(dataset):

            if multitask:
                # targets: (emotion, valence, arousal)
                label = data['targets'][0].int()
            else:
                # targets: emotion
                label = data['targets'].int()
            class_weight = class_weights[label]
            sample_weights[idx] = class_weight

        sampler = WeightedRandomSampler(
            sample_weights, 
            num_samples=len(sample_weights), 
            replacement=True
        )
        return DataLoader(dataset, batch_size, sampler=sampler)

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
    def __init__(self, root_dir, tokenizer, max_len=128, mode='train', multitask=False):
        
        assert mode in ['train', 'val']
        if mode == 'train':
            self.df = pd.read_csv(f'{root_dir}/train.csv')
            print('* Train size:', len(self.df))
        else: 
            self.df = pd.read_csv(f'{root_dir}/test.csv')
            print('* Validation size:', len(self.df))

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.multitask = multitask # label, valence, arousal

        # labelname to index
        self.label_map = {
            'happy': 0,
            'angry': 1,
            'sad': 2,
            'relaxed': 3
        }

        # labelname to valence value
        self.label_map_valence = {
            'happy': 1,
            'angry': 0,
            'sad': 0,
            'relaxed': 1
        }

        # labelname to arouosal value
        self.label_map_arousal = {
            'happy': 1,
            'angry': 1,
            'sad': 0,
            'relaxed': 0,
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

        if self.multitask:
            emotion = self.label_map[_label_str]
            valence = self.label_map_valence[_label_str]
            arousal = self.label_map_arousal[_label_str]
            targets = (emotion, valence, arousal)
        else:
            targets = self.label_map[_label_str]
        
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
            'targets': torch.tensor(targets, dtype=torch.long)
        }


def run_dataset_example(model_name='xlent', multitask=False):
    tokenizer = DatasetUtils.get_tokenizer(model_name)
    dataset = MoodyLyrics(
        root_dir='Dataset',
        tokenizer=tokenizer,
        mode='train',
        max_len=256,
        multitask=multitask
    )
    return dataset

if __name__ == '__main__':

    # targets for multi-task 
    dataset = run_dataset_example('xlnet', multitask=False)
    print(dataset[10]['targets'])
    
    # targets for single-task 
    dataset = run_dataset_example('xlnet', multitask=True)
    print(dataset[10]['targets'])