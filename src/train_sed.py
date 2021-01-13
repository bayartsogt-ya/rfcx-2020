import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data as data
from catalyst.dl import CheckpointCallback, SupervisedRunner
from tqdm import tqdm

# local imports
import src.models as local_models
from src.callbacks import F1Callback, LWLRAPCallback, mAPCallback
from src.criterian import PANNsLoss, FocalLoss, ImprovedPANNsLoss, ImprovedFocalLoss
from src.dataloader import PANNsDataset
from src.helper_train import clear_memory, get_model
from src.metrics import calculate_overall_lwlrap_sklearn


def train_valid_test(config):

    # Data Load
    print('Setup complete. Using torch %s %s' % (torch.__version__,
                                                 torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", device)

    train = pd.read_csv(config.train_csv_path)
    test = pd.read_csv(config.test_csv_path)

    use_fold = config.use_fold
    print("CURRENT FOLD:", use_fold)

    train_fold = train.query("kfold != @use_fold").reset_index(drop=True)
    valid_fold = train.query("kfold == @use_fold").reset_index(drop=True)

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
                                  drop_last=False)

    model = getattr(local_models, config.model_name)(**config.model_config)
    model.to(device)

    print("Training...")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(
    #     0.9, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=True)

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=config.eta_min)

    # Loss
    # criterion = PANNsLoss().to(device)
    criterion = ImprovedPANNsLoss().to(device)

    # callbacks
    callbacks = [
        F1Callback(prefix="f1"),
        mAPCallback(prefix="mAP"),
        LWLRAPCallback(prefix="LWLRAP"),
        CheckpointCallback(save_n_best=1)
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
        logdir=os.path.join(config.output_dir, f"fold{config.use_fold}"),
        callbacks=callbacks,
        main_metric="epoch_LWLRAP",
        minimize_metric=False)

    oof = validate_fold(config, loaders["valid"], device)
    pred = predict(config, test_loader, device)

    return oof, pred


def validate_fold(config, loader, device) -> pd.DataFrame:
    weights_path = os.path.join(
        config.output_dir, f"fold{config.use_fold}/checkpoints/best.pth")
    model = get_model(config.model_name, config.model_config,
                      weights_path, device)

    # Validation
    y_true = []
    y_pred = []
    y_recording_ids = []

    for batch in tqdm(loader):
        y_true.append(batch['targets'].numpy())
        y_recording_ids.extend(batch['recording_id'])

        with torch.no_grad():
            prediction = model(batch['waveform'].to(device))
            framewise_outputs =\
                prediction["framewise_output"].detach().cpu().numpy()
            y_pred.append(framewise_outputs)

    y_true = np.concatenate(y_true, 0)
    y_pred = np.concatenate(y_pred, 0)

    print("OOF SCORE:", calculate_overall_lwlrap_sklearn(y_true,
                                                         y_pred.max(axis=1)))

    clear_memory(model)

    colnames = ["recording_id"] + [f"s{i}" for i in range(config.num_classes)]
    df_pred = pd.DataFrame(y_pred.max(axis=1), columns=colnames[1:])
    df_pred["recording_id"] = y_recording_ids

    df_pred = df_pred.groupby("recording_id")[colnames].mean().reset_index()

    df_pred.to_csv(os.path.join(config.output_dir,
                                f"oof_{config.use_fold}.csv"), index=False)

    return df_pred


def predict(config, loader, device) -> pd.DataFrame:
    weights_path = os.path.join(
        config.output_dir, f"fold{config.use_fold}/checkpoints/best.pth")
    model = get_model(config.model_name, config.model_config,
                      weights_path, device)

    # Validation
    y_pred = []
    y_recording_ids = []

    for batch in tqdm(loader):
        y_recording_ids.extend(batch['recording_id'])

        with torch.no_grad():
            prediction = model(batch['waveform'].to(device))
            framewise_outputs =\
                prediction["framewise_output"].detach().cpu().numpy()
            y_pred.append(framewise_outputs)

    y_pred = np.concatenate(y_pred, 0)

    clear_memory(model)

    colnames = ["recording_id"] + [f"s{i}" for i in range(config.num_classes)]
    df_pred = pd.DataFrame(y_pred.max(axis=1), columns=colnames[1:])
    df_pred["recording_id"] = y_recording_ids

    # group by recording_id and take a mean
    df_pred = df_pred.groupby("recording_id")[colnames].mean().reset_index()

    df_pred.to_csv(os.path.join(config.output_dir,
                                f"prediction_{config.use_fold}.csv"), index=False)

    return df_pred
