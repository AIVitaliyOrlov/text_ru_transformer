import sys
import unittest

import pandas as pd
from torch.utils.data import DataLoader

from configs.data_config import data_cfg
from configs.experiment_config import experiment_cfg
from data_set.text_dataset import TextDataset
from executors.trainer import Trainer
from executors.cub_trainer import Trainer as CubTrainer
from utils.data_utils import collate_function
from utils.enums import SetType
from utils.interpretation import TransformerInterpretation


def test():
    # Load all tests from the test_attention module
    # Comment out or remove tests for the stack that you don't implement from TestAttention class
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='tests', pattern='test_encoder.py')

    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


def train():
    trainer = Trainer(experiment_cfg)
    trainer.fit()
   # trainer.batch_overfit()
#    dataset = TextDataset(data_cfg.data_set, SetType.train)
    #print(dataset.__len__())


def predict():
    trainer = Trainer(experiment_cfg, init_logger=False)
    dataset = getattr(sys.modules[__name__], experiment_cfg.data.name)

    # Get data to make predictions on
    test_dataset = dataset(experiment_cfg.data, SetType.test)
    test_dataloader = DataLoader(
        test_dataset, experiment_cfg.train.validation_batch_size, collate_fn=collate_function, shuffle=False
    )

    # Get predictions
    model_path = experiment_cfg.best_checkpoint_name
    predictions, sample_pair_ids = trainer.predict(model_path, test_dataloader, experiment_cfg.inference)

    # Save results to submission file
    test_results_df = pd.DataFrame({'ID': sample_pair_ids, 'prediction': predictions})
    test_results_df['prediction'] = test_results_df['prediction'].replace('', ' ')
    test_results_df.to_csv('test_predictions.csv', index=False)


def interpret():
    data_cfg = experiment_cfg.data
    trainer = Trainer(experiment_cfg, False)
    trainer.load(experiment_cfg.best_checkpoint_name)

    source_lang_preprocessor = trainer.train_dataset.preprocessors[data_cfg.source_lang]
    target_lang_preprocessor = trainer.train_dataset.preprocessors[data_cfg.target_lang]

    interpreter = TransformerInterpretation(trainer, source_lang_preprocessor, target_lang_preprocessor)
    text = "Tom got a letter from Mary today."
    interpreter.visualize_attention_rollout(text, is_decoder=True)


def train_vit():
    trainer = CubTrainer(experiment_cfg)
    # trainer.batch_overfit()
    trainer.fit()


def predict_vit():
    trainer = CubTrainer(experiment_cfg, init_logger=False)
    dataset = getattr(sys.modules[__name__], experiment_cfg.data.name)

    # Get data to make predictions on
    test_dataset = dataset(experiment_cfg.data, SetType.test, transforms=experiment_cfg.data.eval_transforms)
    test_dataloader = DataLoader(test_dataset, experiment_cfg.train.batch_size, shuffle=False)

    # Get predictions
    model_path = experiment_cfg.best_checkpoint_name
    predictions, image_paths = trainer.predict(model_path, test_dataloader)

    # Save results to submission file
    test_results_df = pd.DataFrame({'ID': image_paths, 'prediction': predictions})
    test_results_df.to_csv('test_predictions.csv', index=False)


if __name__ == '__main__':
    train()
    # predict()
    # interpret()
    # train_vit()
