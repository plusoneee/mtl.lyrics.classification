
from trainer import LyricsEmotionTrainer
from dataset import MoodyLyrics, DatasetUtils
from itertools import product

if __name__ == '__main__':

    tokenizer = DatasetUtils.get_tokenizer('xlnet')
    
    # parameters define
    parameters = dict(
        lr = [1e-6, 5e-7, 3e-7, 1e-7],
        batch_size = [4]
    )

    params = [v for v in parameters.values()]

    for lr, batch_size in product(*params):
        print(f'- Learning rate :{lr}, batch size: {batch_size}')
        train_dataset = MoodyLyrics(
            root_dir='Dataset',
            tokenizer=tokenizer,
            mode='train',
            max_len=1024
        )
        val_dataset = MoodyLyrics(
            root_dir='Dataset',
            tokenizer=tokenizer,
            mode='val',
            max_len=1024
        )

        train_loader = DatasetUtils.get_loader(train_dataset, batch_size)
        val_loader = DatasetUtils.get_loader(val_dataset, batch_size)
        trainer = LyricsEmotionTrainer(
            epcoh_num=100,
            learning_rate=lr
        )

        stop_epoch = trainer.training(train_loader, val_loader, patience=5)
        trainer.writer.add_hparams(
        {   
            'stop_epoch': stop_epoch,
            'lr':lr,
            'train_batch_size':batch_size,
            'test_batch_size':batch_size,
            'best_epoch': trainer.best_epoch
         },{
            'avg val acc': sum(trainer.valid_accuracies)/len(trainer.valid_accuracies)
        })