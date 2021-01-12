import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data as data
from catalyst.dl import CheckpointCallback, SupervisedRunner
from fastprogress import progress_bar

# local imports
import src.models as local_models
from src.callbacks import F1Callback, LWLRAPCallback, mAPCallback
from src.criterian import PANNsLoss
from src.dataloader import PANNsDataset
from src.helper_train import clear_memory, get_model
from src.metrics import calculate_overall_lwlrap_sklearn


# Arguments:
class Config:
    root = "/Users/bayartsogtyadamsuren" + \
        "/Projects/kaggle/rfcx-2020/data/sample_data"
    train_csv_path = os.path.join(root, "train_tp.csv")
    test_csv_path = os.path.join(root, "test.csv")
    train_period = 10
    test_period = 30
    num_epochs = 1
    batch_size = 2
    transforms = []
    use_fold = 0
    num_classes = 24
    model_name = "PANNsDense121Att"

    model_config = {
        "sample_rate": 32000,
        "window_size": 1024,
        "hop_size": 320,
        "mel_bins": 64,
        "fmin": 50,
        "fmax": 14000,
        "classes_num": 527,
    }


def main():

    config = Config()
    print(config)

    raise Exception("stop")

    # Data Load
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", device)

    train = pd.read_csv(config.train_csv_path)
    test = pd.read_csv(config.test_csv_path)
    train_fold = train.query("kfold != @use_fold")
    valid_fold = train.query("kfold == @use_fold")

    # loaders
    loaders = {
        "train": data.DataLoader(PANNsDataset(train_fold,
                                              period=config.train_period,
                                              transforms=config.transforms,
                                              data_path=os.path.join(
                                                  config.root, "train"),
                                              is_train=True),
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 num_workers=2,
                                 pin_memory=True,
                                 drop_last=True),
        "valid": data.DataLoader(PANNsDataset(valid_fold,
                                              period=config.train_period,
                                              transforms=None,
                                              data_path=os.path.join(
                                                  config.root, "train"),
                                              is_train=True),
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=2,
                                 pin_memory=True,
                                 drop_last=False)
    }

    test_loader = data.DataLoader(PANNsDataset(test,
                                               period=config.test_period,
                                               transforms=None,
                                               data_path=os.path.join(
                                                   config.root, "test"),
                                               is_train=False),
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=2,
                                  pin_memory=True,
                                  drop_last=False),

    model = getattr(local_models, config.model_name)(**config.model_config)
    model.to(device)

    print("Training...")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Loss
    criterion = PANNsLoss().to(device)

    # callbacks
    callbacks = [
        F1Callback(prefix="f1"),
        mAPCallback(prefix="mAP"),
        LWLRAPCallback(prefix="LWLRAP"),
        CheckpointCallback(save_n_best=0)
    ]

    # Train
    warnings.simplefilter("ignore")

    runner = SupervisedRunner(
        device=device,
        input_key="waveform",
        input_target_key="targets")

    runner.train(
        model=model,
        criterion=criterion,
        loaders=loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config.num_epochs,
        verbose=True,
        logdir=f"fold{config.use_fold}",
        callbacks=callbacks,
        main_metric="epoch_LWLRAP",
        minimize_metric=False)

    oof = validate_fold(config, loaders["valid"])
    pred = predict(config, test_loader)

    return oof, pred


def validate_fold(config, loader):
    weights_path = f"fold{config.use_fold}/checkpoints/best.pth"
    model = get_model(config.model_config, weights_path)

    # Validation
    y_true = []
    y_pred = []
    y_recording_ids = []

    for batch in progress_bar(loader):
        y_true.append(batch['targets'].numpy())
        y_recording_ids.extend(batch['recording_id'].numpy().tolist())

        with torch.no_grad():
            prediction = model(batch['waveform'].cuda())
            framewise_outputs =\
                prediction["framewise_output"].detach().cpu().numpy()
            y_pred.append(framewise_outputs)

    y_true = np.concatenate(y_true, 0)
    y_pred = np.concatenate(y_pred, 0)

    print("y_true.shape, y_pred.shape:", y_true.shape, y_pred.shape)
    print("SCORE:", calculate_overall_lwlrap_sklearn(y_true,
                                                     y_pred.max(axis=1)))

    clear_memory(model)

    colnames = ["recording_id"] + [f"s{i}" for i in range(config.num_classes)]
    df_pred = pd.DataFrame(y_pred.max(axis=1), columns=colnames[1:])
    df_pred["recording_id"] = y_recording_ids
    df_pred = df_pred[colnames]

    df_pred.to_csv("oof_{config.use_fold}.csv", index=False)

    return df_pred


def predict(config, loader):
    weights_path = f"fold{config.use_fold}/checkpoints/best.pth"
    model = get_model(config.model_config, weights_path)

    # Validation
    y_pred = []
    y_recording_ids = []

    for batch in progress_bar(loader):
        y_recording_ids.extend(batch['recording_id'].numpy().tolist())

        with torch.no_grad():
            prediction = model(batch['waveform'].cuda())
            framewise_outputs = prediction["framewise_output"].detach(
            ).cpu().numpy()
            y_pred.append(framewise_outputs)

    y_pred = np.concatenate(y_pred, 0)

    clear_memory(model)

    colnames = ["recording_id"] + [f"s{i}" for i in range(config.num_classes)]
    df_pred = pd.DataFrame(y_pred.max(axis=1), columns=colnames[1:])
    df_pred["recording_id"] = y_recording_ids
    df_pred = df_pred[colnames]
    df_pred.to_csv("prediction_{config.use_fold}.csv", index=False)

    return df_pred