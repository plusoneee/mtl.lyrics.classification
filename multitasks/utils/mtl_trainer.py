import os
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.mtl_loss import MultiTaskLossWrapper
import datetime

class MultiTaskTrainer():
    def __init__(self, 
                model, 
                epcoh_num, 
                learning_rate=1e-6, 
                checkpoint_root='checkpoints',
                log_path = 'logs'
                ):
        torch.cuda.empty_cache()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
        self.model.to(self.device)
        
        self.epoch_number = epcoh_num
        self.learning_rate = learning_rate

        self.lossfunc = self.lossfunction
        self.optimizer = self.adamw_optimizer
        
        self.runing_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        if not os.path.isdir(checkpoint_root):
            print('* Root Dir not exist ... Create it.')
            os.mkdir(checkpoint_root)

        self.checkpoint_root = checkpoint_root + os.sep + self.runing_datetime
        os.mkdir(self.checkpoint_root)

        self.log_path = log_path
        self.writer = self.tfboard_writer
        
        self.best_epoch = 0
        self.valid_accuracies = []

    @property
    def tfboard_writer(self):
        log_dir = self.log_path + os.sep + self.runing_datetime
        print(f"* Tensorboard Log @ {log_dir}")
        return SummaryWriter(log_dir)
        
    @property
    def lossfunction(self):
        lossfunc = MultiTaskLossWrapper()
        return lossfunc

    @property
    def adamw_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
        ]
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

    def calculat_output_correctly(self, outputs, targets):
        pred = torch.argmax(outputs.data, 1) # row
        return (pred == targets).sum()

    def validate_one_step(self, loader):
        self.model.eval()
        correct, total, loss = 0, 0, 0
        with torch.no_grad():
            for idx, data in enumerate(loader):
                ids, mask, token_type_ids, targets = self._parse_loaded(data)
                targets = targets
                outputs = self.model(ids, mask, token_type_ids)
                correct += self.calculat_output_correctly(outputs[0].to('cpu'), targets[:, 0].to('cpu'))
                total += targets.size(0)
                loss += self.lossfunc(outputs, targets).item()
            acc = 100.0 * float(correct/total)
        
        return acc, loss/idx

    def train_one_step(self, train_loader):
        ep_loss, correct, total = 0, 0, 0
        self.model.train()
        for idx, data in enumerate(train_loader):
            self.optimizer.zero_grad()
            ids, mask, token_type_ids, targets = self._parse_loaded(data)
            outputs = self.model(ids, mask, token_type_ids)

            # emotion task acc
            correct += self.calculat_output_correctly(outputs[0], targets[:, 0])          
            total += targets.size(0)
  
            # multi-tasks loss
            loss = self.lossfunc(outputs, targets)
            loss.backward()
            self.optimizer.step()
            ep_loss += loss.item()

        acc = 100.0 * float(correct/total)
        return acc, ep_loss/idx
    
    def training(self, train_loader, valid_loader, patience=5, acc_baseline=80.0):
        triger_times = 0
        min_valid_loss = None

        for epoch in range(self.epoch_number):
            train_acc, train_loss = self.train_one_step(train_loader)
            valid_acc, valid_loss = self.validate_one_step(valid_loader)
            
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

            self.valid_accuracies.append(valid_acc)

            if min_valid_loss is None:
                min_valid_loss = train_loss

            elif valid_loss > min_valid_loss:
                triger_times += 1

                if triger_times >= patience:
                    print(f' - Early Stoping at {epoch+1} epoch')

                    return epoch+1

            elif valid_loss < min_valid_loss and valid_acc >= acc_baseline:
                triger_times = 0
                # create checkpoint variable and add important data
                checkpoint_path = self.checkpoint_root + os.sep + f'epoch-{epoch+1}-loss-{valid_loss:.3f}-ac-{valid_acc}.pt'
                print('** Validation loss decreased ({:.6f} --> {:.6f}). Saving model ..'.format(min_valid_loss, valid_loss))
                # save checkpoint as best model
                self.save_checkpoint(checkpoint_path)
                min_valid_loss = valid_loss
                self.best_epoch = epoch+1

            print('\t * TrigerTimes/Patience: {}/{}'.format(triger_times, patience))
        return epoch

    def load_checkpoint(self, checkpoint_path):
        # load check point
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint)

    def save_checkpoint(self, checkpoint_path):
        #torch.save(self.model.state_dict(), checkpoint_path)
        torch.save(self.model.module.state_dict(), checkpoint_path)
    
    def _parse_loaded(self, data):
        ids = data['ids'].to(self.device)
        mask = data['mask'].to(self.device)
        token_type_ids = data['token_type_ids'].to(self.device)
        targets = data['targets'].to(self.device)
        return ids, mask, token_type_ids, targets

    def add_hparams(self, hparms:dict, metric:dict):
        self.writer.add_hparams(hparms, metric)