import os
import sys

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from models.vision_transformer import VisionTransformer
from utils.common_functions import set_seed
from utils.enums import SetType
from utils.logger import NeptuneLogger
from utils.metrics import accuracy_score, balanced_accuracy_score


class Trainer:
    """A class for training Vision Transformer on CUB dataset."""

    def __init__(self, config, init_logger=True):
        self.config = config
        set_seed(self.config.seed)

        self._prepare_data()
        self._prepare_model()

        self.global_step = 0

        self._init_logger(init_logger)

    def _init_logger(self, init_logger):
        if init_logger:
            self.logger = NeptuneLogger(self.config.neptune)
            if not self.config.train.continue_train:
                self.logger.log_hyperparameters(self.config)

    def _prepare_data(self):
        """Preparing training and validation data."""
        data_cfg = self.config.data
        dataset = getattr(sys.modules[__name__], data_cfg.name)
        batch_size = self.config.train.batch_size
        train_transforms = data_cfg.train_transforms
        validation_transforms = data_cfg.eval_transforms

        self.train_dataset = dataset(data_cfg, SetType.train, transforms=train_transforms)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size, drop_last=True, shuffle=True)

        self.eval_train_dataset = dataset(data_cfg, SetType.train, transforms=validation_transforms)
        self.eval_train_dataloader = DataLoader(self.eval_train_dataset, batch_size, shuffle=False)

        self.validation_dataset = dataset(data_cfg, SetType.validation, transforms=validation_transforms)
        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=batch_size, shuffle=False)

    def _prepare_model(self):
        """Preparing model, optimizer and loss function."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_class = getattr(sys.modules[__name__], self.config.model.name)
        model_kwargs = {'input_channels': self.config.data.channels_num, 'classes_num': self.config.data.num_classes}
        self.model = model_class(self.config.model, **model_kwargs).to(self.device)

        self.optimizer = getattr(optim, self.config.train.optimizer)(
            self.model.parameters(), lr=self.config.train.learning_rate,
            **self.config.train.optimizer_params[self.config.train.optimizer]
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.train.label_smoothing)

        # TODO: Implement scheduler initialization: initialize warmup_scheduler
        #           (for the first self.config.train.warmup_steps steps) and cosine_scheduler (for all the rest steps)
        self.warmup_steps = self.config.train.warmup_steps
        self.warmup_scheduler = ...
        self.cosine_scheduler = ...
        raise NotImplementedError

    def save(self, filepath: str):
        """Saves trained model."""
        os.makedirs(self.config.checkpoints_dir, exist_ok=True)
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'global_step': self.global_step,
                'scheduler_state_dict': {
                    'warmup_scheduler': self.warmup_scheduler.state_dict(),
                    'cosine_scheduler': self.cosine_scheduler
                }
            },
            os.path.join(self.config.checkpoints_dir, filepath)
        )

    def load(self, filepath: str):
        """Loads trained model."""
        checkpoint = torch.load(os.path.join(self.config.checkpoints_dir, filepath), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.warmup_scheduler.load_state_dict(checkpoint['scheduler_state_dict']['warmup_scheduler'])
        self.cosine_scheduler.load_state_dict(checkpoint['scheduler_state_dict']['cosine_scheduler'])

    def update_best_params(self, valid_metric, best_metric):
        """Update best parameters: saves model if metrics exceeds the best values achieved."""
        if best_metric < valid_metric:
            self.save(self.config.best_checkpoint_name)
            best_metric = valid_metric
        return best_metric

    def make_step(self, batch: dict, update_model=False):
        """This method performs one step, including forward pass, calculation of the target function, backward
        pass and updating the model weights (if update_model is True).

        Args:
            batch: batch data
            update_model (bool): if True it is necessary to perform a backward pass and update the model weights

        Returns:
            loss: loss function value
            output: model output (batch_size x num_classes)
        """
        # TODO: To implement training epoch pipeline, do the following:
        #       1. Get needed information from the batch (image, target)
        #       2. Move data to self.device
        #       3. Get model prediction
        #       5. Compute loss
        #       6. Update weights using standard pipeline
        #       7. (Optional) Log learning rate (from self.optimizer.param_groups[0]['lr'])
        #       8. Return loss value and outputs
        raise NotImplementedError

    def train_epoch(self, epoch: int):
        self.model.train()
        # TODO: Implement epoch training pipeline: for each batch in self.train_dataloader:
        #       1. Make training step
        #       2. Get predictions with argmax and compute metric
        #       3. Log training data
        #       4. Increment self.global_step (it is used to control scheduler steps)
        raise NotImplementedError

    def fit(self):
        """The main model training loop."""
        start_epoch, best_metric = 0, 0

        if self.config.train.continue_train:
            epoch = self.config.train.checkpoint_from_epoch
            self.load(self.config.checkpoint_name % epoch)
            start_epoch = epoch + 1

        for epoch in range(start_epoch, self.config.num_epochs):
            self.train_epoch(epoch)

            self.evaluate(epoch, self.eval_train_dataloader, SetType.train)
            valid_metric = self.evaluate(epoch, self.validation_dataloader, SetType.validation)

            if epoch % self.config.checkpoint_save_frequency == 0:
                self.save(self.config.checkpoint_name % epoch)

            best_metric = self.update_best_params(valid_metric, best_metric)

    @torch.no_grad()
    def evaluate(self, epoch: int, dataloader: DataLoader, set_type: SetType):
        """Evaluation.

        The method is used to make the model performance evaluation on training/validation/test data.

        Args:
            epoch: current epoch
            dataloader: dataloader to make evaluation on
            set_type: set type
        """
        # TODO: Implement a standard evaluation method for the given dataloader:
        #       1. For each batch in the given dataloader:
        #           a) make step with update_model=False parameter
        #           b) add loss, outputs and targets to the corresponding lists
        #       2. Calculate total loss as the average of all losses
        #       3. Get all predictions with argmax
        #       4. Compute metric
        #       5. Log evaluation data
        #       6. Return metric result
        self.model.eval()
        total_loss, all_outputs, all_targets = [], [], []
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, model_path, dataloader):
        """Gets model predictions for a given dataloader."""
        self.load(model_path)
        self.model.eval()
        all_predictions, all_image_paths = [], []

        for batch in dataloader:
            all_image_paths.extend(batch['path'])

            output = self.model(batch['image'].to(self.device))
            all_predictions.append(output.argmax(-1))

        all_predictions = torch.cat(all_predictions)

        return all_predictions.tolist(), all_image_paths

    def batch_overfit(self):
        """One batch overfitting.

        This feature can be useful for debugging and evaluating your model's ability to learn and update its weights.
        """
        self.model.train()
        batch = next(iter(self.train_dataloader))

        for i in range(self.config.overfit.num_iterations):
            loss_value, output = self.make_step(batch, update_model=True)
            balanced_accuracy = balanced_accuracy_score(batch['target'].numpy(), output.argmax(-1))
            print(i, loss_value, balanced_accuracy, accuracy_score(batch['target'].numpy(), output.argmax(-1)))
