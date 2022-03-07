import torch
import torch.nn as nn

class MultiTaskLossWrapper(nn.Module):
    def __init__(self):
        super(MultiTaskLossWrapper, self).__init__()
        self.emotion_task_loss = nn.CrossEntropyLoss()
        self.av_tasks_loss = nn.BCEWithLogitsLoss()
        self.weights = [0.8, 0.1, 0.1]

    def forward(self, outputs, targets):
        emotion_out = outputs[0]
        valence_out = outputs[1]
        arousal_out = outputs[2]
        emotion_target = targets[:, 0].to(torch.int64)
        valence_target = targets[:, 1].unsqueeze(1).float()
        arousal_target = targets[:, 2].unsqueeze(1).float()
        
        e_loss = self.emotion_task_loss(emotion_out, emotion_target)
        v_loss = self.av_tasks_loss(valence_out, valence_target)
        a_loss = self.av_tasks_loss(arousal_out, arousal_target)
        
        weighted_loss = e_loss * self.weights[0] + v_loss * self.weights[1] + a_loss * self.weights[2]
        return weighted_loss