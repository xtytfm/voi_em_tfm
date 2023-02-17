import argparse
import logging
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from models.models import ModelCnnLstm
from tools.process_data import split_data, load_audio_duration_fixed, add_adwgn, get_mel_from_data, standard_scaler

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("Train-model")

EMOTIONS_DICT = {0: 'happy', 1: 'angry', 2: 'other', 3: 'sadness'}
# EMOTIONS = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 0: 'surprise'}


DEFAULT_CONFIG = {
    'paths': {
        "manifest_path": "/home/xabierdetoro/xabi/tfm/gits/scripts/master_radvness.xlsx",
        # "manifest_path": "/home/xabierdetoro/xabi/tfm/gits/data/master_join.xlsx",
        "model_name": "ser_model_ravdness",
        "model_dir": "/home/xabierdetoro/xabi/tfm/gits/output/",

    },
    'input': {"sample_rate": 16000,
              "duration": 3

              },
    'train': {"aug_data": True,
              "max_epochs": 1500,
              "batch_size": 16,
              "patience": 15,
              "learning_rate": 0.001,

              },
    "num_classes": 4}


class Config:
    def __init__(self, config_path):
        if os.path.isfile(config_path):
            with open(config_path, "r") as yaml_file:
                self.config = yaml.load(yaml_file, Loader=yaml.FullLoader)
        else:
            raise FileNotFoundError(f"El fichero {config_path} no existe.")


class EarlyStopper:
    """
    Para un entrenamiento si el conjunto de validación no se mejora en X (patience) épocas.
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def prepare_data(X, Y, sample_rate, augmentation=False, dataset_type="train"):
    logger.info(f"Generating mels_spectrogram for {dataset_type}")
    if augmentation:
        logger.info(f"Augmentation data")
        X, Y = add_adwgn(X, Y)
        logger.info(f"Shape of {dataset_type} after augmentation: {X.shape}")

    X = get_mel_from_data(X=X, sample_rate=sample_rate)

    logger.info(f"Standard scaler")
    X = standard_scaler(X)
    logger.info(f"Shape before train: {X.shape}")
    logger.info(f"End preparadate {dataset_type} for train")
    return X, Y


def loss_fnc(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions, target=targets)


def make_train_step(model, loss_fnc, optimizer):
    def train_step(X, Y):
        # set model to train mode
        model.train()
        # forward pass
        output_logits, output_softmax, attention_weights_norm = model(X)
        predictions = torch.argmax(output_softmax, dim=1)
        accuracy = torch.sum(Y == predictions) / float(len(Y))
        # compute loss
        loss = loss_fnc(output_logits, Y)
        # compute gradients
        loss.backward()
        # update parameters and zero gradients
        optimizer.step()
        optimizer.zero_grad()
        return loss.item(), accuracy * 100

    return train_step


def make_validate_fnc(model, loss_fnc):
    def validate(X, Y):
        with torch.no_grad():
            model.eval()
            output_logits, output_softmax, attention_weights_norm = model(X)
            predictions = torch.argmax(output_softmax, dim=1)
            accuracy = torch.sum(Y == predictions) / float(len(Y))
            loss = loss_fnc(output_logits, Y)
        return loss.item(), accuracy * 100, predictions

    return validate


def main_train(config_dict: Dict):
    test = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config_dict['paths'].get("do_split"):
        df = pd.read_excel(config_dict['paths']['manifest_path'])
        train, val, test = split_data(df, train_split=0.7, seed=1234)
        train.to_csv("train.csv", index=False)
        val.to_csv("val.csv", index=False)
        test.to_csv("test.csv", index=False)
    else:
        train = pd.read_csv(config_dict['paths']['train_path'])
        val = pd.read_csv(config_dict['paths']['val_path'])

    OUTPUT_MODEL = os.path.join(config_dict['paths']['model_dir'], config_dict['paths']['model_name'] +
                                "_best_result.pt")

    logger.info(f"Train shape: {train.shape}")
    logger.info(f"Validation shape: {val.shape}")
    if test is not None:
        logger.info(f"Test shape: {test.shape}")

    logger.info(f"Start process reading data")
    train_array = load_audio_duration_fixed(train, duration=config_dict['input']['duration'],
                                            sample_rate=config_dict['input']['sample_rate'])

    X_train, Y_train = prepare_data(X=train_array, Y=train['label'].values,
                                    sample_rate=config_dict['input']['sample_rate'],
                                    augmentation=config_dict['train']['aug_data'])

    val_array = load_audio_duration_fixed(val, duration=config_dict['input']['duration'],
                                          sample_rate=config_dict['input']['sample_rate'])
    X_val, Y_val = prepare_data(X=val_array, Y=val['label'].values, sample_rate=config_dict['input']['sample_rate'],
                                augmentation=config_dict['train']['aug_data'],
                                dataset_type='validation')

    DATASET_SIZE = X_train.shape[0]
    logger.info(f'device is {device}')

    model = ModelCnnLstm(num_classes=config_dict['num_classes']).to(device)

    OPTIMIZER = torch.optim.SGD(model.parameters(), lr=config_dict['train']['learning_rate'],
                                weight_decay=1e-3, momentum=0.8)

    early_stopper = EarlyStopper(patience=config_dict['train']['patience'], min_delta=0)

    train_step = make_train_step(model, loss_fnc, optimizer=OPTIMIZER)
    validate = make_validate_fnc(model, loss_fnc)
    best_accuracy = 0
    warm_up_epochs = 10
    losses = []
    val_losses = []

    # Training loop
    batch_size = config_dict['train']['batch_size']
    for epoch in range(config_dict['train']['max_epochs']):
        # shuffle data
        ind = np.random.permutation(DATASET_SIZE)
        X_train = X_train[ind, :, :, :]
        Y_train = Y_train[ind]
        epoch_acc = 0
        epoch_loss = 0
        iters = int(DATASET_SIZE / batch_size)
        for i in range(iters):
            batch_start = i * batch_size
            batch_end = min(batch_start + batch_size, DATASET_SIZE)
            actual_batch_size = batch_end - batch_start
            X = X_train[batch_start:batch_end, :, :, :]
            Y = Y_train[batch_start:batch_end]
            X_tensor = torch.tensor(X, device=device).float()
            Y_tensor = torch.tensor(Y, dtype=torch.long, device=device)
            loss, acc = train_step(X_tensor, Y_tensor)
            epoch_acc += acc * actual_batch_size / DATASET_SIZE
            epoch_loss += loss * actual_batch_size / DATASET_SIZE
            # logger.info(f"Epoch {epoch}: iteration {i}/{iters}")
        X_val_tensor = torch.tensor(X_val, device=device).float()
        Y_val_tensor = torch.tensor(Y_val, dtype=torch.long, device=device)
        val_loss, val_acc, _ = validate(X_val_tensor, Y_val_tensor)
        losses.append(epoch_loss)
        val_losses.append(val_loss)
        logger.info(f"Epoch {epoch} --> loss:{epoch_loss:.4f}, acc:{epoch_acc:.2f}%, val_loss:{val_loss:.4f},"
                    f" val_acc:{val_acc:.2f}%")

        if epoch > warm_up_epochs:
            if best_accuracy < val_acc:
                logger.info(f'Old accuracy {best_accuracy}, improve to {val_acc}')
                best_accuracy = val_acc
                logger.info(f"Saving model best val accuracy in {OUTPUT_MODEL}")
                torch.save(model.state_dict(), OUTPUT_MODEL)

        if early_stopper.early_stop(validation_loss=val_loss):
            logger.info(f"Early stopping in epoch {epoch}, because validation loss did not improve in "
                        f"{config_dict['train']['patience']}.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenamiento modelo SER')
    parser.add_argument('-c', '--config', type=str, help='Path del fichero de configuracion', required=False)
    args = parser.parse_args()
    if args.config is None:
        config = DEFAULT_CONFIG
        logger.info("Usando párametros por defecto, ya que no se ha pasado ningún fichero de configuración")
    else:
        logger.info(f"Leyendo el fichero de configuracion {args.config}")
        cfg = Config(args.config)
        config = cfg.config
    main_train(config)
