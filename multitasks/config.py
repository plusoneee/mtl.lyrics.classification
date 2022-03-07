# training settings
MODEL_NAME = 'xlnet'
EPOCHS_NUM = 200
BATCH_SIZE = [8]
LEARING_RATES = [1e-5, 5e-6, 1e-6, 5e-7]
EARLY_STOPING_PATIENCE = 5
MULTI_TASKS = True
CHECKPOINT_ROOT = 'checkpoints'
LOG_PATH = 'logs'

# dataset settings
DATA_PATH = 'data'
MAX_LENGTH = 800
