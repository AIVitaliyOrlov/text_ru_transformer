import os

from easydict import EasyDict

from configs.data_config import data_cfg
from configs.model_config import model_cfg
from utils.enums import InferenceType

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

experiment_cfg = EasyDict()
experiment_cfg.seed = 0
experiment_cfg.num_epochs = 300

# Train parameters
experiment_cfg.train = EasyDict()
experiment_cfg.train.batch_size = 32
experiment_cfg.train.learning_rate = 1e-1
experiment_cfg.train.weight_decay = 1e-2
experiment_cfg.train.warmup_steps = 1000
experiment_cfg.train.label_smoothing = 0
experiment_cfg.train.optimizer = 'Adam'  # from (Adam, AdamW)
experiment_cfg.train.optimizer_params = {
    'Adam': {'betas': (0.9, 0.999), 'eps': 1e-8}, 'AdamW': {'betas': (0.9, 0.98), 'eps': 1e-9}
}
experiment_cfg.train.continue_train = False
experiment_cfg.train.checkpoint_from_epoch = None
experiment_cfg.train.log_frequency = 100
experiment_cfg.train.log_window = 50
experiment_cfg.train.validation_frequency = 50000
experiment_cfg.train.validation_batch_size = 32
experiment_cfg.train.inference_frequency = 2

# Overfit parameters
experiment_cfg.overfit = EasyDict()
experiment_cfg.overfit.num_iterations = 10000



experiment_cfg.mlflow = EasyDict()
experiment_cfg.mlflow.tracking_uri = 'http://127.0.0.1:8080'
experiment_cfg.mlflow.env_path = os.path.join(ROOT_DIR, '.env')
experiment_cfg.mlflow.project = 'transform_1'
experiment_cfg.mlflow.experiment_name = 'Test_transformer_'
experiment_cfg.mlflow.dataset_version = os.path.join(ROOT_DIR, 'data/raw_data_train_ru.txt')
experiment_cfg.mlflow.dataset_preprocessing = os.path.join(ROOT_DIR, 'data/tokenized_data_train_ru.pickle')
experiment_cfg.mlflow.run_id = None
experiment_cfg.mlflow.dependencies_path = os.path.join(ROOT_DIR, 'requirements.txt')

# Checkpoints parameters
experiment_cfg.checkpoints_dir = os.path.join(ROOT_DIR, 'experiments', experiment_cfg.mlflow.experiment_name,
                                              'checkpoints')
experiment_cfg.checkpoint_save_frequency = 1000
experiment_cfg.checkpoint_name = 'checkpoint_%s'
experiment_cfg.best_checkpoint_name = 'best_checkpoint'

experiment_cfg.model = model_cfg
experiment_cfg.data = data_cfg.data_set
