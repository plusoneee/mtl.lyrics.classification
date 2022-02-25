import os
import torch
from model import LyricsEmotionBERT, LyricsEmotionXLNet
from torch.utils.tensorboard import SummaryWriter
import datetime

class LyricsEmotionTrainer:

    def __init__(self, epcoh_num, learning_rate=1e-6, checkpoint_root='./checkpoints'):
        torch.cuda.empty_cache()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = LyricsEmotionXLNet()
        self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
        self.model.to(self.device)
        self.lossfunc = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        self.epoch_number = epcoh_num
        runing_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.checkpoint_root = checkpoint_root + os.sep + runing_datetime
        os.mkdir(self.checkpoint_root)
        log_dir = "logs/" + runing_datetime
        self.writer = SummaryWriter(log_dir)
        self.best_epoch = 0
        self.train_accuracies = []
        self.valid_accuracies = []
 
    def test_accuracy(self, loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for idx, data in enumerate(loader):
                ids, mask, token_type_ids, targets = self._parse_loaded(data)
                targets = targets.to('cpu')
                outputs = self.model(ids, mask, token_type_ids).to('cpu')
                pred = torch.argmax(outputs.data, 1) # row
                correct += (pred == targets).sum()
                total += targets.size(0)
            
        acc = 100.0 * float(correct/total)
        return acc

    def train_one_step(self, train_loader):
        ep_loss = 0
        self.model.train()
        for idx, data in enumerate(train_loader):
            self.optimizer.zero_grad()

            ids, mask, token_type_ids, targets = self._parse_loaded(data)
            outputs = self.model(ids, mask, token_type_ids)
            loss = self.lossfunc(outputs, targets)
            loss.backward()
            self.optimizer.step()
            ep_loss += loss.item()

        return ep_loss/idx

    def validate_one_step(self, valid_loader):
        ep_loss = 0
        self.model.eval()
        with torch.no_grad():
            for idx, data in enumerate(valid_loader):
                ids, mask, token_type_ids, targets = self._parse_loaded(data)
                targets = targets.to('cpu')
                outputs = self.model(ids, mask, token_type_ids).to('cpu')
                loss = self.lossfunc(outputs, targets)
                ep_loss += loss.item()

        return ep_loss/idx


    def training(self, train_loader, valid_loader, patience=5):
        triger_times = 0
        min_valid_loss = None

        for epoch in range(self.epoch_number):
            train_loss = self.train_one_step(train_loader)
            train_acc = self.test_accuracy(train_loader)

            valid_loss = self.validate_one_step(valid_loader)
            valid_acc = self.test_accuracy(valid_loader)

            print('Epoch: {} \n\t - Avgerage Training Loss: {:.6f}\n\t - Average Validation Loss: {:.6f}'.format(
                epoch + 1, 
                train_loss,
                valid_loss
                ))

            print('\t - Training Accuracy: {:.6f}\n\t - Validation Accuracy: {:.6f}'.format(
                train_acc,
                valid_acc
                ))

            self.writer.add_scalar('loss/training', train_loss, epoch+1)
            self.writer.add_scalar('loss/validation', valid_loss, epoch+1)
            self.writer.add_scalar('accuracy/training', train_acc, epoch+1)
            self.writer.add_scalar('accuracy/validation', valid_acc, epoch+1)

            self.train_accuracies.append(train_acc)
            self.valid_accuracies.append(valid_acc)

            if min_valid_loss is None:
                min_valid_loss = train_loss

            elif valid_loss > min_valid_loss:
                triger_times += 1

                if triger_times >= patience:
                    print(f' - Early Stoping at {epoch+1} epoch')

                    return epoch+1

            elif valid_loss < min_valid_loss and valid_acc > 50.0:
                triger_times = 0
                # create checkpoint variable and add important data
                checkpoint = {
                    'epoch': epoch + 1,
                    'valid_loss_min': valid_loss,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }

                checkpoint_path = self.checkpoint_root + os.sep + f'epoch-{epoch+1}-loss-{valid_loss:.3f}-ac-{valid_acc}.pt'
                print('** Validation loss decreased ({:.6f} --> {:.6f}). Saving model ..'.format(min_valid_loss, valid_loss))
                # save checkpoint as best model
                self.save_checkpoint(checkpoint, checkpoint_path)
                min_valid_loss = valid_loss
                self.best_epoch = epoch+1

            print('\t * TrigerTimes/Patience: {}/{}'.format(triger_times, patience))
        return epoch


    def load_checkpoint(self, checkpoint_path):
        """
        checkpoint_path: path to save checkpoint
        """
        # load check point
        checkpoint = torch.load(checkpoint_path)
        
        # initialize state_dict  & optimizer from checkpoint
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # initialize valid_loss_min from checkpoint to valid_loss_min
        valid_loss_min = checkpoint['valid_loss_min']
        
        return checkpoint['epoch'], valid_loss_min.item()

    def save_checkpoint(self, state, checkpoint_path):
        torch.save(state, checkpoint_path)
        
    def _parse_loaded(self, data):
        ids = data['ids'].to(self.device)
        mask = data['mask'].to(self.device)
        token_type_ids = data['token_type_ids'].to(self.device)
        targets = data['targets'].to(self.device)
        return ids, mask, token_type_ids, targets