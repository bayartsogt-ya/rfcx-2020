import os
import math
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset


class PANNsDataset(Dataset):
    def __init__(self,
                 df,
                 period=10,
                 transforms=None,
                 data_path="../input/rfcx-species-audio-detection/train",
                 is_train=True):
        self.period = period
        self.transforms = transforms
        self.data_path = data_path
        self.is_train = is_train

        # if it is test:
        if not is_train:
            df = self.create_test_df(df)

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx, :]

        # read and crop
        y, sr = self.create_audio_for_period(
            row.recording_id, row.t_min, row.t_max)

        if self.transforms:
            y = self.transforms(samples=y, sample_rate=sr)

        # onehot labeling
        label = np.zeros(24, dtype='f')

        if self.is_train:
            label[row.species_id] = 1

        return {
            "waveform": y,
            "targets": torch.tensor(label, dtype=torch.float),
            "recording_id": row.recording_id,
        }

    def create_audio_for_period(self, recording_id, t_min, t_max):
        y, sr = sf.read(os.path.join(self.data_path, recording_id + ".flac"))

        len_y = len(y)
        effective_length = sr * self.period
        tmin, tmax = round(sr * t_min), round(sr * t_max)

        if len_y < effective_length:
            start = np.random.randint(effective_length - len_y)
            new_y = np.zeros(effective_length, dtype=y.dtype)
            new_y[start:start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            center = round((tmin + tmax) / 2)
            big = center - effective_length
            if big < 0:
                big = 0
            start = np.random.randint(big, center)
            y = y[start:start + effective_length]
            if len(y) < effective_length:
                new_y = np.zeros(effective_length, dtype=y.dtype)
                start1 = np.random.randint(effective_length - len(y))
                new_y[start1:start1 + len(y)] = y
                y = new_y.astype(np.float32)
            else:
                y = y.astype(np.float32)
        else:
            y = y.astype(np.float32)
            start = 0

        return y, sr

    def create_test_df(self, df):
        recording_ids = []
        t_mins = []
        t_maxs = []

        for _, row in df.iterrows():
            n_of_samples = math.ceil(row["length"] / self.period)

            for i in range(n_of_samples):
                recording_ids.append(row.recording_id)
                t_mins.append(i * self.period)
                t_maxs.append((i + 1) * self.period)

        df = pd.DataFrame({
            "recording_id": recording_ids,
            "t_min": t_mins,
            "t_max": t_maxs,
        })

        return df
