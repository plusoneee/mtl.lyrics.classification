import torch
import torch.nn as nn
from transformers import  XLNetModel

class MultiTaskLyricsEmotionXLNet(nn.Module):
    def __init__(self, dropout_p=0.1, out_dim=4):
        super(MultiTaskLyricsEmotionXLNet, self).__init__()

        self.xlnet = XLNetModel.from_pretrained(
            'xlnet-base-cased'
        )
        self.shared_middle_layer = nn.Sequential(
            nn.Linear(768, 32),
            nn.Dropout(dropout_p),
        )
        self.emotion_classifier = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(32, out_dim),
        )
        self.valence_classifier =  nn.Sequential(
            nn.Linear(32, 1)
        )
        self.arousal_classifier =  nn.Sequential(
            nn.Linear(32, 1)
        )

    def forward(self, ids, mask, token_type_ids):
        last_hidden_state = self.xlnet(
            input_ids=ids,
            attention_mask=mask,\
            token_type_ids=token_type_ids
        )
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        outputs = self.shared_middle_layer(mean_last_hidden_state)
        
        valence_out = self.valence_classifier(outputs)
        arousal_out = self.arousal_classifier(outputs)
        emotion_out = self.emotion_classifier(outputs)
        return (emotion_out, valence_out, arousal_out)

    def pool_hidden_state(self, last_hidden_state):
        """
        Pool the output vectors into a single mean vector 
        Shape(batch, 1024, 768) -> Shape(batch, 768)
        """
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state

if __name__ == '__main__':
    model = MultiTaskLyricsEmotionXLNet()