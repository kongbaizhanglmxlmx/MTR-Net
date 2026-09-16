import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


def normalize_data(data, scaler=None):
    data = np.asarray(data, dtype=np.float32)
    if np.any(np.isnan(data)):
        data = np.nan_to_num(data)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    print("Data normalized")

    return data, scaler


def get_data_dim(dataset):
    if dataset == "OpenMRG":
        return 2

    raise ValueError(f"Unknown dataset: {dataset}")

def get_target_dims(dataset):
    if dataset == "OpenMRG":
        return None

    raise ValueError(f"Unknown dataset: {dataset}")
def get_data(
    dataset,
    max_train_size=None,
    max_test_size=None,
    normalize=False,
    train_start=0,
    test_start=0
):
    if dataset != "OpenMRG":
        raise ValueError(f"Unknown dataset: {dataset}")

    prefix = os.path.join(
        "datasets",
        "data",
        "processed"
    )

    train_end = (
        None
        if max_train_size is None
        else train_start + max_train_size
    )

    test_end = (
        None
        if max_test_size is None
        else test_start + max_test_size
    )

    x_dim = get_data_dim(dataset)

    with open(
        os.path.join(prefix, "OpenMRG_train.pkl"),
        "rb"
    ) as f:
        train_data = pickle.load(f).reshape(
            (-1, x_dim)
        )[train_start:train_end]

    with open(
        os.path.join(prefix, "OpenMRG_train_label.pkl"),
        "rb"
    ) as f:
        train_label = pickle.load(f).reshape(
            -1
        )[train_start:train_end]

    with open(
        os.path.join(prefix, "OpenMRG_test.pkl"),
        "rb"
    ) as f:
        test_data = pickle.load(f).reshape(
            (-1, x_dim)
        )[test_start:test_end]

    with open(
        os.path.join(prefix, "OpenMRG_test_label.pkl"),
        "rb"
    ) as f:
        test_label = pickle.load(f).reshape(
            -1
        )[test_start:test_end]

    if normalize:
        train_data, scaler = normalize_data(
            train_data
        )
        test_data, _ = normalize_data(
            test_data,
            scaler=scaler
        )

    print("Dataset:", dataset)
    print("Train:", train_data.shape)
    print("Train label:", train_label.shape)
    print("Test:", test_data.shape)
    print("Test label:", test_label.shape)

    return (
        train_data,
        train_label
    ), (
        test_data,
        test_label
    )


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        data,
        window,
        target_dim=None,
        horizon=1,
        labels=None,
        label_mode="last",
    ):
        self.data = data
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon
        self.labels = labels
        self.label_mode = label_mode

        if self.label_mode != "last":
            raise ValueError('Only label_mode="last" is supported.')

        if self.labels is not None and len(self.labels) != len(self.data):
            raise ValueError("labels 长度必须与 data 长度一致")

    def __getitem__(self, index):
        x = self.data[index:index + self.window]

        y_forecast = self.data[
            index + self.window:
            index + self.window + self.horizon
        ]

        if self.labels is None:
            y_cls = 0.0
        else:
            y_cls = float(
                self.labels[index + self.window - 1]
            )

        return x, y_forecast, y_cls

    def __len__(self):
        return len(self.data) - self.window - self.horizon + 1
def create_data_loaders(train_dataset, batch_size, val_split=0.1, shuffle=True, test_dataset=None):
    train_loader, val_loader, test_loader = None, None, None
    if val_split == 0.0:
        print(f"train_size: {len(train_dataset)}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

        print(f"train_size: {len(train_indices)}")
        print(f"validation_size: {len(val_indices)}")

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def plot_losses(losses, save_path="", plot=True):
    """
    :param losses: dict with losses
    :param save_path: path where plots get saved
    """

    plt.plot(losses["train_forecast"], label="Forecast loss")
    plt.plot(losses["train_recon"], label="Recon loss")
    plt.plot(losses["train_total"], label="Total loss")
    plt.title("Training losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/train_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()

    plt.plot(losses["val_forecast"], label="Forecast loss")
    plt.plot(losses["val_recon"], label="Recon loss")
    plt.plot(losses["val_total"], label="Total loss")
    plt.title("Validation losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/validation_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()


def load(model, PATH, device="cpu"):
    """
    Loads the model's parameters from the path mentioned
    :param PATH: Should contain pickle file
    """
    model.load_state_dict(torch.load(PATH, map_location=device))


def get_series_color(y):
    if np.average(y) >= 0.95:
        return "black"
    elif np.average(y) == 0.0:
        return "black"
    else:
        return "black"


def get_y_height(y):
    if np.average(y) >= 0.95:
        return 1.5
    elif np.average(y) == 0.0:
        return 0.1
    else:
        return max(y) + 0.1


