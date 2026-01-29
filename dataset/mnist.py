# dataset/mnist.py
import os
import urllib.request
import gzip
import pickle
import numpy as np
import struct

# MNIST 下载地址（按顺序尝试）
url_bases = [
    "https://yann.lecun.com/exdb/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
key_file = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_label": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_label": "t10k-labels-idx1-ubyte.gz",
}

# 当前文件所在目录：.../dataset
dataset_dir = os.path.dirname(os.path.abspath(__file__))

# 原始 gzip 下载到这里
save_dir = os.path.join(dataset_dir, "mnist")
os.makedirs(save_dir, exist_ok=True)

# 解析后的 pickle 缓存
save_file = os.path.join(save_dir, "mnist.pkl")


def _download(file_name: str) -> None:
    file_path = os.path.join(save_dir, file_name)
    if os.path.exists(file_path):
        return

    print(f"Downloading {file_name} ...")
    last_err = None
    for base in url_bases:
        try:
            urllib.request.urlretrieve(base + file_name, file_path)
            print("Done")
            return
        except Exception as e:
            last_err = e
    raise last_err


def _download_mnist() -> None:
    for v in key_file.values():
        _download(v)


def _load_label(file_name: str) -> np.ndarray:
    file_path = os.path.join(save_dir, file_name)
    with gzip.open(file_path, "rb") as f:
        # IDX label: magic(4), num(4), labels...
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in {file_name}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    if labels.shape[0] != num:
        raise ValueError("Label count mismatch")
    return labels


def _load_img(file_name: str) -> np.ndarray:
    file_path = os.path.join(save_dir, file_name)
    with gzip.open(file_path, "rb") as f:
        # IDX image: magic(4), num(4), rows(4), cols(4), pixels...
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in {file_name}")
        img = np.frombuffer(f.read(), dtype=np.uint8)

    img = img.reshape(num, rows, cols)  # (N, 28, 28)
    return img


def _convert_numpy() -> dict:
    dataset = {}
    dataset["train_img"] = _load_img(key_file["train_img"])
    dataset["train_label"] = _load_label(key_file["train_label"])
    dataset["test_img"] = _load_img(key_file["test_img"])
    dataset["test_label"] = _load_label(key_file["test_label"])
    return dataset


def _init_mnist() -> None:
    _download_mnist()
    dataset = _convert_numpy()
    print(f"Creating pickle file: {save_file}")
    with open(save_file, "wb") as f:
        pickle.dump(dataset, f, -1)
    print("Done")


def _change_one_hot_label(X: np.ndarray) -> np.ndarray:
    # X: (N,) labels in [0..9]
    T = np.zeros((X.shape[0], 10), dtype=np.int32)
    T[np.arange(X.shape[0]), X] = 1
    return T


def load_mnist(normalize: bool = True,
              flatten: bool = True,
              one_hot_label: bool = False):
    """
    Returns:
      (x_train, t_train), (x_test, t_test)

    normalize:
      True  -> x in [0.0, 1.0] float32
      False -> x in [0, 255]  uint8

    flatten:
      True  -> x shape (N, 784)
      False -> x shape (N, 1, 28, 28)

    one_hot_label:
      True  -> t shape (N, 10)
      False -> t shape (N,)
    """
    if not os.path.exists(save_file):
        _init_mnist()

    with open(save_file, "rb") as f:
        dataset = pickle.load(f)

    x_train, t_train = dataset["train_img"], dataset["train_label"]
    x_test, t_test = dataset["test_img"], dataset["test_label"]

    # 归一化
    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
    else:
        # 保持 uint8 更符合 “0~255”
        x_train = x_train.astype(np.uint8)
        x_test = x_test.astype(np.uint8)

    # 标签 one-hot
    if one_hot_label:
        t_train = _change_one_hot_label(t_train)
        t_test = _change_one_hot_label(t_test)

    # flatten / reshape
    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)  # (N, 784)
        x_test = x_test.reshape(x_test.shape[0], -1)
    else:
        # (N, 1, 28, 28)
        x_train = x_train.reshape(-1, 1, 28, 28)
        x_test = x_test.reshape(-1, 1, 28, 28)

    return (x_train, t_train), (x_test, t_test)
