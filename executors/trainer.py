import os
import random
import sys
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_set.text_dataset import TextDataset
from models.transformer import TextDecoderTransformer

from executors.sampler import RandomSortingSampler, BatchRandomSortingSampler
from models.transformer import Transformer
from utils.common_functions import set_seed
from utils.data_utils import get_sequence_mask, collate_function
from utils.enums import SetType, InferenceType
from utils.logger import NeptuneLogger, MLFlowLogger
from utils.training_utils import custom_lr_schedule
from torch.nn.functional import softmax


class Trainer:
    """A class for model training."""

    def __init__(self, config, init_logger=True):
        self.config = config
        set_seed(self.config.seed)

        self._prepare_data()
        self._prepare_model()

        self._init_logger(init_logger)

    def _init_logger(self, init_logger):
        if init_logger:
            self.logger = MLFlowLogger(self.config.mlflow)  # NeptuneLogger(self.config.neptune)
            if not self.config.train.continue_train:
                self.logger.log_hyperparameters(self.config)

    def _prepare_data(self):
        """Preparing training and validation data."""
        data_cfg = self.config.data
        dataset = getattr(sys.modules[__name__], data_cfg.name)
        batch_size = self.config.train.batch_size

        self.train_dataset = dataset(data_cfg, SetType.train)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=BatchRandomSortingSampler(self.train_dataset, batch_size=batch_size, shuffle=False),
            collate_fn=collate_function
        )

        self.validation_dataset = dataset(data_cfg, SetType.validation)
        self.validation_dataloader = DataLoader(
            self.validation_dataset,
            batch_sampler=BatchRandomSortingSampler(
                self.validation_dataset, batch_size=self.config.train.validation_batch_size, shuffle=False),
            collate_fn=collate_function
        )

    def _prepare_model(self):
        """Preparing model, optimizer and loss function."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_class = getattr(sys.modules[__name__], self.config.model.name)
        vocabulary_size = self.train_dataset.get_vocabulary_size()
        model_kwargs = {
            'vocabulary_size': vocabulary_size,
        }
        self.model = model_class(self.config, **model_kwargs).to(self.device)

        self.optimizer = getattr(optim, self.config.train.optimizer)(
            self.model.parameters(), lr=self.config.train.learning_rate,
            **self.config.train.optimizer_params[self.config.train.optimizer]
        )
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.config.data.special_tokens.index('[PAD]') - 1,
            label_smoothing=self.config.train.label_smoothing
        )

        lr_lambda = lambda step: custom_lr_schedule(
            cur_step=step, emb_size=self.config.model.d_model, warmup_steps=self.config.train.warmup_steps
        )
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        #self.metric = evaluate.load("bleu")

    def save(self, filepath: str):
        """Saves trained model."""
        os.makedirs(self.config.checkpoints_dir, exist_ok=True)
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            },
            os.path.join(self.config.checkpoints_dir, filepath)
        )

    def load(self, filepath: str):
        """Loads trained model."""
        checkpoint = torch.load(os.path.join(self.config.checkpoints_dir, filepath), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

#    def update_best_params(self, valid_metric, best_metric):
#        """Update best parameters: saves model if metrics exceeds the best values achieved."""
#        if best_metric < valid_metric:
#            self.save(self.config.best_checkpoint_name)
#            best_metric = valid_metric
#        return best_metric

    def make_step(self, batch: dict, step:int, update_model=False):
        """This method performs one step, including forward pass, calculation of the target function, backward
        pass and updating the model weights (if update_model is True).

        Args:
            batch: batch data
            update_model (bool): if True it is necessary to perform a backward pass and update the model weights

        Returns:
            loss: loss function value
            output: model output (batch_size x num_classes)
        """
        decoder_inputs, decoder_outputs, decoder_mask = batch
        decoder_inputs = decoder_inputs.to(self.device)

        decoder_outputs = decoder_outputs.to(self.device)
        decoder_mask = decoder_mask.to(self.device)
        outputs = self.model(decoder_inputs, decoder_mask)
        loss = self.criterion(outputs.reshape(-1, outputs.shape[-1]), decoder_outputs.reshape(-1) -1)

        if update_model:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.logger.save_metrics(SetType.train.name, 'learning_rate', self.optimizer.param_groups[0]['lr'], step)

        return loss.item(), outputs.detach().cpu().numpy(), decoder_outputs.detach().cpu().numpy()

    def evaluate_train(self, losses: list[float], predictions: list[list[int]], decoder_outputs: list[list[int]]):
        """Evaluates model performance on a part of training data (taken by sliding window).

        Args:
            losses: list of batch losses
            predictions: list of predictions (each predicted sequence is a list of token ids)
            decoder_outputs: list of targets (each target is a list of token ids)
            encoder_inputs: list of source sentences (each source sentence is a list of token ids)
            target_lang_preprocessor: preprocessor to decode target
            source_lang_processor: preprocessor to decode source

        Returns:
            Mean loss, BLEU score and text with translations to log
        """
        losses = losses[-self.config.train.log_window:]
        predictions = predictions[-self.config.train.log_window:]
        decoder_outputs = decoder_outputs[-self.config.train.log_window:]

        train_predictions_decoded = self.train_dataset.tokenizer.decode(predictions, batch=True)
        train_targets_decoded = self.train_dataset.tokenizer.decode(decoder_outputs, batch=True)

        random_sample_num = random.randint(0, len(predictions) - 1)
        output_to_show = f'Source:     {train_targets_decoded[random_sample_num]}\n' \
                         f'Prediction: {train_predictions_decoded[random_sample_num]}\n'

        return np.mean(losses), output_to_show

    def train_epoch(self, epoch: int, best_metric: float):
        """Train the model on training data for one epoch.

        The method goes through all train_dataloader batches and calls the self.make_step() method at each step. With
            specified frequency it evaluates model on the part of training data and on the validation data.
        """
        self.model.train()
        steps_done = epoch * len(self.train_dataloader)
        train_losses, train_predictions, train_decoder_outputs = [], [], []



        pad_idx = self.config.data.special_tokens.index("[PAD]")
        total_steps = self.train_dataloader.batch_sampler.__len__()
        with tqdm(total=total_steps) as pbar:
            for step, batch in enumerate(self.train_dataloader):
                pbar.update()
                loss, output, decoder_outputs = self.make_step(batch, update_model=True, step=step)
                train_losses.append(loss)
                prediction_with_pad = output.argmax(axis=-1) + 1
                train_predictions.extend(
                    [prediction_with_pad[i][decoder_outputs[i] != pad_idx].tolist() for i in range(len(decoder_outputs))]
                )

                train_decoder_outputs.extend(decoder_outputs.tolist())

                # Evaluate performance on the validation data
            #    if step % self.config.train.validation_frequency == 0:
             #       valid_loss = self.evaluate(self.validation_dataloader)
             #       self.logger.save_metrics(SetType.validation.name, 'loss', valid_loss, step=steps_done + step)


                # Evaluate performance on the part of training data
                if step % self.config.train.log_frequency == 0 and step != 0:
                    train_loss, output_to_show = self.evaluate_train(
                        train_losses, train_predictions, train_decoder_outputs
                    )

                    self.logger.save_metrics(SetType.train.name, 'loss', train_loss, step=steps_done + step)
                    #self.logger.save_metrics(SetType.train.name, 'bleu', train_metric, step=steps_done + step)
                    print(f"Epoch: {epoch} step: {step}")
                    print(output_to_show)
                    train_losses, train_predictions, train_decoder_outputs = [], [], []

                if step % self.config.checkpoint_save_frequency == 0:
                    self.save(self.config.checkpoint_name % (steps_done + step))



        return best_metric

    def fit(self):
        """The main model training loop."""
        start_epoch, best_metric = 0, 0

        if self.config.train.continue_train:
            step = self.config.train.checkpoint_from_epoch
            self.load(self.config.checkpoint_name % step)
            start_epoch = step // len(self.train_dataloader) + 1

        for epoch in range(start_epoch, self.config.num_epochs):
            best_metric = self.train_epoch(epoch, best_metric)
            if epoch % self.config.train.inference_frequency == 0:
                total_loss = self.evaluate(self.validation_dataloader, inference=True)
                #_, train_eval_metric = self.evaluate(self.train_eval_dataloader, inference=True)
                #best_metric = self.update_best_params(valid_metric, best_metric)

                step = max(0, epoch * len(self.train_dataloader) - 1)
                #self.logger.save_metrics(SetType.validation.name + '_eval', 'bleu_inference', valid_metric, step=step)
                #self.logger.save_metrics(SetType.train.name + '_eval', 'bleu_inference', train_eval_metric, step=step)

        #_, valid_metric = self.evaluate(self.validation_dataloader, inference=True)
        #_, train_eval_metric = self.evaluate(self.train_eval_dataloader, inference=True)
        #self.update_best_params(valid_metric, best_metric)

        last_step = self.config.num_epochs * len(self.train_dataloader) - 1
        #self.logger.save_metrics(SetType.validation.name + '_eval', 'bleu_inference', valid_metric, step=last_step)
        #self.logger.save_metrics(SetType.train.name + '_eval', 'bleu_inference', train_eval_metric, step=last_step)
        self.save(self.config.checkpoint_name % last_step)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, inference: bool = False):
        """Evaluation.

        The method is used to make the model performance evaluation on training/validation/test data.

        Args:
            dataloader: dataloader to make evaluation on
            inference: a boolean indicating whether to get predictions through inference
        """
        self.model.eval()

        pad_idx = self.config.data.special_tokens.index("[PAD]")
        #total_loss, all_predictions, all_decoder_outputs = [], [], []
        total_loss = []
        for batch in dataloader:
            loss, output, decoder_outputs = self.make_step(batch, update_model=False, step=0)

            total_loss.append(loss)
            """
            
            
            if inference:
                raise NotImplementedError
                #all_predictions.extend(self.inference(encoder_inputs, self.config.inference))
            else:
                prediction_with_pad = output.argmax(axis=-1) + 1
                all_predictions.extend(
                    [prediction_with_pad[i][decoder_outputs[i] != pad_idx].tolist() for i in
                     range(len(decoder_outputs))]
                )
            all_decoder_outputs.extend(decoder_outputs.tolist())
"""
        total_loss = np.mean(total_loss)
       # all_targets_decoded = self.train_dataset.decode(all_decoder_outputs, batch=True)
       # all_predictions_decoded = self.train_dataset.decode(all_predictions, batch=True)


        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        self.model.train()
        return total_loss

    def inference_step(self, encoded_input: torch.Tensor, decoded_sequence: torch.Tensor, source_mask: torch.Tensor):
        """Gets model decoder output given encoder output and sequence made by decoder at current step.

        Args:
            encoded_input: source sequences passed through the model encoder (batch size, source sequence length, d_model)
            decoded_sequence: sequences with all tokens generated up to the current inference step (batch size, generated sequence length)
            source_mask: a sequence mask with ones at positions that should be masked out (for encoder outputs)

        Returns:
            Model output generated wrt the already generated sequence (decoded_sequence)
        """
        target_mask = get_sequence_mask(decoded_sequence, mask_future_positions=True, device=self.device)

        with torch.no_grad():
            decoder_inputs_embeddings = self.model.positional_encoding(self.model.embeddings_decoder(decoded_sequence))
            decoder_output, decoder_self_attention_weights, decoder_encoder_attention_weights = self.model.decoder(
                decoder_inputs_embeddings, encoded_input, source_mask, target_mask)
            output = self.model.output(decoder_output)

        return output, decoder_self_attention_weights, decoder_encoder_attention_weights

    @torch.no_grad()
    def inference(self, sequence: torch.Tensor, inference_config, return_attention=False):
        raise NotImplementedError
        """Makes inference with auto-regressive decoding for the given sequence."""
        self.model.eval()
        batch_size = sequence.size(0)
        sos_token_id = self.config.data.special_tokens.index("[SOS]")
        eos_token_id = self.config.data.special_tokens.index("[EOS]")
        inference_step = 0
        decoded_sequence = torch.ones((batch_size, 1), dtype=torch.int32, device=self.device) * sos_token_id
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        input_mask = get_sequence_mask(sequence, device=self.device)
        encoder_inputs_embeddings = self.model.positional_encoding(self.model.embeddings_encoder(sequence))
        encoded_input, encoder_attention_weights = self.model.encoder(encoder_inputs_embeddings, input_mask)

        while not finished_sequences.all() and inference_step < inference_config.stop_predict:
            output, decoder_self_attention_weights, decoder_encoder_attention_weights = self.inference_step(
                encoded_input, decoded_sequence, input_mask)
            if inference_config.type == InferenceType.greedy.value:
                current_token = torch.argmax(output, dim=-1)[:, inference_step].view(-1, 1) + 1
            elif inference_config.type == InferenceType.temperature.value:
                output = output / (inference_config.temperature_value + inference_config.eps)
                probabilities = softmax(output, dim=-1)
                current_token = probabilities[:, inference_step, :].multinomial(num_samples=1) + 1
            else:
                raise Exception('Unknown inference type!')

            decoded_sequence = torch.hstack([decoded_sequence, current_token])
            finished_sequences |= current_token.squeeze() == eos_token_id
            inference_step += 1

        eos_subsequence_mask = torch.cummax(decoded_sequence == eos_token_id, dim=1).values
        decoded_sequence = decoded_sequence.masked_fill(eos_subsequence_mask, eos_token_id)
        if return_attention:
            return decoded_sequence.cpu().tolist(), encoder_attention_weights, decoder_self_attention_weights, decoder_encoder_attention_weights
        else:
            return decoded_sequence.cpu().tolist()

    @torch.no_grad()
    def predict(self, model_path: str, dataloader: DataLoader, inference_config):
        raise NotImplementedError
        """Gets model predictions for a given dataloader."""
        self.load(model_path)
        self.model.eval()

        target_lang_preprocessor = self.train_dataset.preprocessors[self.config.data.target_lang]
        all_predictions, all_sample_ids = [], []

        for sample in dataloader:
            sample_id, encoder_inputs, _, _, _, _ = sample
            encoder_inputs = encoder_inputs.to(self.device)
            prediction = self.inference(encoder_inputs, inference_config)

            all_predictions.extend(target_lang_preprocessor.decode(prediction, batch=True))
            all_sample_ids.extend(sample_id.view(-1).cpu().tolist())

        return all_predictions, all_sample_ids

    def batch_overfit(self):
        """One batch overfitting.

        This feature can be useful for debugging and evaluating your model's ability to learn and update its weights.
        """
        self.model.train()
        data_set = self.train_dataset
        pad_idx = self.config.data.special_tokens.index("[PAD]")
        batch = next(iter(self.train_dataloader))

        for step in tqdm(range(self.config.overfit.num_iterations)):
            torch.cuda.empty_cache()
            loss, output, decoder_outputs = self.make_step(batch, update_model=True, step=step)

            if step % 10 == 0:
                prediction_with_pad = output.argmax(axis=-1) +1
               # predictions = [
               #     prediction_with_pad[i][decoder_outputs[i] != pad_idx].tolist() for i in range(len(decoder_outputs))
               # ]

                random_sample_num = random.randint(0, len(batch) - 1)
                print(f'Step: {step} loss :{loss}')

                targets_decoded = self.train_dataset.tokenizer.decode(decoder_outputs[random_sample_num])
                predictions_decoded = self.train_dataset.tokenizer.decode(prediction_with_pad[random_sample_num])

                output_to_show = f'Target:     {targets_decoded}\n' \
                                 f'Prediction: {predictions_decoded}\n'
                print(output_to_show)
