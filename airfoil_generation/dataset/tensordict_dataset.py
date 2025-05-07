from typing import Union
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, LazyMemmapStorage


class TensorDictDataset(torch.utils.data.Dataset):
    """
    Overview:
        Dataset for storing and loading data in a tensordict format.
    Interface:
        ``__init__``, ``__getitem__``, ``__len__``.
    """

    def __init__(
        self,
        keys: list,
        max_size: int = 100000,
    ):
        """
        Overview:
            Initialization method of class
        Arguments:
            env_id (:obj:`str`): The environment id
        """

        super().__init__()
        self.keys = keys
        self.storage = LazyMemmapStorage(max_size=max_size)

    def __getitem__(self, index):
        """
        Overview:
            Get data by index
        Arguments:
            index (:obj:`int`): Index of data
        Returns:
            data (:obj:`dict`): Data dict

        """

        data = self.storage.get(index=index)
        return data

    def __len__(self):
        return self.storage._len

    def append(self, data: Union[dict, TensorDict], batch_size: int = None):
        """
        Overview:
            Append data to the dataset
        Arguments:
            data (:obj:`dict`): Data dict
        """

        for key in self.keys:
            assert key in data, f"key {key} not in data"

        if isinstance(data, dict):
            assert batch_size is not None, "batch_size is None, please set it first"

            if isinstance(batch_size, torch.Size):
                torch_batch_size = batch_size
            elif isinstance(batch_size, int):
                torch_batch_size = torch.Size([batch_size])
            elif isinstance(batch_size, list):
                torch_batch_size = torch.Size(batch_size)
            elif isinstance(batch_size, tuple):
                torch_batch_size = torch.Size(batch_size)
            else:
                raise ValueError(f"batch_size type {type(batch_size)} not supported")

            def to_torch_tensor(data, batch_size):
                if isinstance(data, np.ndarray):
                    return torch.from_numpy(data).float()
                elif isinstance(data, torch.Tensor):
                    return data
                elif isinstance(data, bool):
                    return torch.tensor(data)
                elif isinstance(data, int):
                    return torch.tensor(data, dtype=torch.int64)
                elif isinstance(data, float):
                    return torch.tensor(data, dtype=torch.float32)
                elif isinstance(data, list):
                    return torch.tensor(data, dtype=torch.float32)
                elif isinstance(data, tuple):
                    return torch.tensor(data, dtype=torch.float32)
                elif isinstance(data, dict):
                    return TensorDict(
                        {
                            key: to_torch_tensor(data[key], batch_size=batch_size)
                            for key in data.keys()
                        },
                        batch_size=batch_size,
                    )
                elif isinstance(data, TensorDict):
                    assert data.batch_size == torch.Size(
                        batch_size
                    ), f"data batch size {data.batch_size} not equal to {batch_size}"
                    return data
                else:
                    raise ValueError(f"data type {type(data)} not supported")

            data = TensorDict(
                {
                    key: to_torch_tensor(data[key], batch_size=torch_batch_size)
                    for key in self.keys
                },
                batch_size=torch_batch_size,
            )
        else:
            torch_batch_size = data.batch_size
            if batch_size is not None:
                assert (
                    data.batch_size == torch_batch_size
                ), f"data batch size {data.batch_size} not equal to {torch_batch_size}"
            batch_size = data.batch_size[0]

        self.storage.set(
            range(self.__len__(), self.__len__() + batch_size),
            data,
        )

    def save(self, path: Union[str, Path]):
        """
        Overview:
            Save the dataset to a file
        Arguments:
            path (:obj:`Union[str, Path]`): Path to save the dataset to
        """
        if isinstance(path, str):
            path = Path(path)
        self.storage.save(path=path)

    def load(self, path: Union[str, Path]):
        """
        Overview:
            Load the dataset from a file
        Arguments:
            path (:obj:`Union[str, Path]`): Path to load the dataset from
        """
        if isinstance(path, str):
            path = Path(path)
        self.storage.load(path=path)
