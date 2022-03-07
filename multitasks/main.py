from utils.dataset import MoodyLyrics, DatasetUtils
from utils.mtl_model import MultiTaskLyricsEmotionXLNet
from utils.mtl_trainer import MultiTaskTrainer
from itertools import product
import config as cnf

if __name__ == '__main__':

    tokenizer = DatasetUtils.get_tokenizer(cnf.MODEL_NAME)

    # parameters define
    parameters = dict(
        lr = cnf.LEARING_RATES,
        batch_size = cnf.BATCH_SIZE
    )

    params = [v for v in parameters.values()]
    for lr, batch_size in product(*params):
        # only support xlnet for mtl now.
        assert cnf.MODEL_NAME == 'xlnet' 
        model = MultiTaskLyricsEmotionXLNet()
        
        print(f'- Learning rate :{lr}, batch size: {batch_size}')
        
        # Datasets & DataLoader:
        train_dataset = MoodyLyrics(
            root_dir=cnf.DATA_PATH,
            tokenizer=tokenizer,
            mode='train',
            max_len=cnf.MAX_LENGTH,
            multitask=cnf.MULTI_TASKS
        )
        
        val_dataset = MoodyLyrics(
            root_dir=cnf.DATA_PATH,
            tokenizer=tokenizer,
            mode='val',
            max_len=cnf.MAX_LENGTH,
            multitask=cnf.MULTI_TASKS
        )
        train_loader = DatasetUtils.get_loader(train_dataset, batch_size, train=True, multitask=cnf.MULTI_TASKS)
        val_loader = DatasetUtils.get_loader(val_dataset, batch_size, train=False, multitask=cnf.MULTI_TASKS)
        
        # trainer
        trainer = MultiTaskTrainer(
            model,
            epcoh_num=cnf.EPOCHS_NUM,
            learning_rate=lr,
            checkpoint_root=cnf.CHECKPOINT_ROOT,
            log_path=cnf.LOG_PATH
        )

        stop_epoch = trainer.training(train_loader, val_loader, patience=cnf.EARLY_STOPING_PATIENCE)
        hparms = {   
            'stop_epoch': stop_epoch,
            'lr':lr,
            'train_batch_size':batch_size,
            'best_epoch': trainer.best_epoch
            }
        
        metric = {
            'max_val_acc': max(trainer.valid_accuracies),
            'mean_val_acc': sum(trainer.valid_accuracies)/len(trainer.valid_accuracies)
            }
        
        trainer.add_hparams(hparms, metric)