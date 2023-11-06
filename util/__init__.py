# 获取mnist数据集
import pathlib
import sys
from typing import Iterable

import numpy as np

SN3_PASCALVINCENT_TYPEMAP = {
    8: np.uint8,
    9: np.int8,
    11: np.int16,
    12: np.int32,
    13: np.float32,
    14: np.float64,
}

DATA_NAME = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
    "test_images": "t10k-images-idx3-ubyte",
    "test_labels": "t10k-labels-idx1-ubyte"
}


class MNIST:
    def __init__(self, root="data", transform=None):
        self.__train_images = None
        self.__train_labels = None
        self.__test_images = None
        self.__test_labels = None
        self.__transform = transform
        self.__path = pathlib.Path(root, "MNIST/raw")

    @property
    def train_images(self):
        if self.__train_images is None:
            data = self.load_data(DATA_NAME["train_images"])
            self.__train_images = self.transform_data(self.__transform, data)
        return self.__train_images

    @property
    def train_labels(self):
        if self.__train_labels is None:
            self.__train_labels = self.load_data(DATA_NAME["train_labels"])
        return self.__train_labels

    @property
    def test_images(self):
        if self.__test_images is None:
            data = self.load_data(DATA_NAME["test_images"])
            self.__test_images = self.transform_data(self.__transform, data)
        return self.__test_images

    @property
    def test_labels(self):
        if self.__test_labels is None:
            self.__test_labels = self.load_data(DATA_NAME["test_labels"])
        return self.__test_labels

    def load_data(self, name):
        assert pathlib.Path(self.__path, name).exists(), f"{pathlib.Path(self.__path, name)} not exists"
        with open(pathlib.Path(self.__path, name), "rb") as f:
            data = f.read()
            magic = self.get_int(data[0:4])
            nd = magic % 256  # nd表示维度
            ty = magic // 256  # ty表示数据类型
            assert 1 <= nd <= 3
            assert 8 <= ty <= 14
            np_type = SN3_PASCALVINCENT_TYPEMAP[ty]
            s = [self.get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]  # s表示shape

            num_bytes_per_value = np.iinfo(np_type).bits // 8
            # MNIST数据集是大端序，如果是小端序则需要反转
            needs_byte_reversal = sys.byteorder == "little" and num_bytes_per_value > 1
            parsed = np.frombuffer(bytearray(data), dtype=np_type, offset=(4 * (nd + 1)))
            if needs_byte_reversal:
                parsed = np.flip(parsed, axis=0)

            assert parsed.shape[0] == np.prod(s)
            parsed = parsed.reshape(*s)
        return parsed

    @staticmethod
    def get_int(b_data: bytes):
        return int(b_data.hex(), 16)

    @staticmethod
    def transform_data(transform, data):
        if not transform:
            return data
        # 可迭代类型
        elif isinstance(transform, Iterable):
            for trans in transform:
                data = trans(data)
            return data
        else:
            return transform(data)
