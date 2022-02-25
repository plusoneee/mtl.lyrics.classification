
import torch
import torch.nn as nn
from transformers import BertModel
from transformers import  XLNetModel

class LyricsEmotionXLNet(nn.Module):
    def __init__(self, dropout_p=0.1, out_dim=4):
        super(LyricsEmotionXLNet, self).__init__()

        self.xlnet = XLNetModel.from_pretrained(
            'xlnet-base-cased'
        )
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, ids, mask, token_type_ids):
        
        last_hidden_state = self.xlnet(
            input_ids=ids,
            attention_mask=mask,\
            token_type_ids=token_type_ids
        )
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        outputs = self.classifier(mean_last_hidden_state)
        return outputs

    def pool_hidden_state(self, last_hidden_state):
        """
        Pool the output vectors into a single mean vector 
        Shape(batch, 1024, 768) -> Shape(batch, 768)
        """
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state

class LyricsEmotionBERT(nn.Module):
    def __init__(self, dropout_p=0.1, out_dim=4):
        super(LyricsEmotionBERT, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, ids, mask, token_type_ids):
        _, output= self.bert(
            ids, 
            attention_mask = mask, 
            token_type_ids = token_type_ids
            )
        output = self.dropout(output)
        output = self.classifier(output)
        return output

if __name__ == '__main__':
    model = LyricsEmotionXLNet()
    print(model)