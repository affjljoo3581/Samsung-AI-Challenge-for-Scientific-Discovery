import json
import zlib
from base64 import b64decode
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import Dataset, Sampler

from encoding import MolecularEncoder


class MolecularParser:
    """A parser class for compressed molecular data.

    The standard MoT dataset contains gzip-compressed molecular json data.
    `TabularDataset` will extract the input data from the tabular dataset and then
    requires to parse the data into the input sequences to the parser. This class helps
    to decompress, parse, and even augmentations. You can simply pass this instance to
    `TabularDataset` when creating.

    Args:
        encoder: A molecular encoder. This encoder will create the input sequences from
            the molecular structure data.
        bond_drop_prob: The probability of dropping molecular bonds. Default is `0.1`.
    """

    def __init__(self, encoder: MolecularEncoder, bond_drop_prob: float = 0.1):
        self.encoder = encoder
        self.bond_drop_prob = bond_drop_prob

    def __call__(self, molecular: str) -> Dict[str, List]:
        molecular = zlib.decompress(b64decode(molecular)).decode()
        molecular = json.loads(molecular)

        if np.random.rand() < self.bond_drop_prob:
            # We will drop the molecular bonds with probability of 15%. That is, the
            # expectation of the number of dropped molecular bonds is 85% of the
            # original one. Note that you can only control the molecular selecting
            # probability, not the individual bond dropping probability.
            molecular["bonds"] = [
                bond for bond in molecular["bonds"] if np.random.rand() > 0.15
            ]

        return self.encoder.encode(molecular)


class TabularDataset(Dataset):
    """A dataset class for large-scale tabular file.

    This class is designed to handle large-scale tabular dataset. Unlike large datasets
    are usually used through `IterableDataset`, this class supports random access by
    using pre-calculated seeking points. The seeking points will be computed when this
    class is initialized, but you can set your own (and customized) seeking points.

    Args:
        filename: The name of dataset file.
        input_parser: A parser for input text.
        input_column: The name of input column.
        label_columns: A collection of names of label columns.
        labels_mean_std: The statistics of the labels. If given, the labels will be
            normalized by their means and stds. Default is `None`.
        seeking_points: A list of line-start positions. This class will seek to the
            certain file position and read a line. If `None`, it will be manually
            generated. Default is `None`.
    """

    def __init__(
        self,
        filename: str,
        input_parser: Callable[[str], Dict],
        input_column: str,
        label_columns: Iterable[str],
        labels_mean_std: Optional[Dict[str, Tuple[int, int]]] = None,
        seeking_points: Optional[Iterable[int]] = None,
    ):
        self.filename = filename
        self.input_column = input_column
        self.input_parser = input_parser
        self.label_columns = label_columns
        self.labels_mean_std = labels_mean_std or {}

        self.dataset_fp = None
        self.dataset_columns = None

        if seeking_points is None:
            # Since this class does not load the entire file data to memory, a random
            # example access for the file is basically impossible. To support fetching
            # random examples, this class requires `seeking_points` which contains the
            # seeking position of examples in the file. If it is `None`, this class
            # creates `seeking_points` manually.
            with open(filename, "r") as fp:
                seeking_points = [len(fp.readline())]
                for line in fp:
                    seeking_points.append(seeking_points[-1] + len(line))
            seeking_points.pop(-1)
        self.seeking_points = seeking_points

    def __len__(self) -> int:
        return len(self.seeking_points)

    def __getitem__(self, index: int) -> Dict[str, List[Union[int, float]]]:
        if self.dataset_fp is None:
            # If you use multi-processing on `DataLoader`, this dataset will be first
            # initialized on main process, and then copied to the other processes.
            # Therefore a file pointer which is created at main process does not work on
            # the other processes. So we use lazy-initialization to resolve this
            # problem.
            self.dataset_fp = open(self.filename, "r")
            self.dataset_columns = self.dataset_fp.readline().strip().split(",")
        self.dataset_fp.seek(self.seeking_points[index])

        example = self.dataset_fp.readline()
        example = dict(zip(self.dataset_columns, example.strip().split(",")))

        labels = []
        for name in self.label_columns:
            mean, std = 0, 1
            if name in self.labels_mean_std:
                mean, std = self.labels_mean_std[name]
            label = (float(example[name]) - mean) / std
            labels.append(min(max(label, -5), 5))

        return {**self.input_parser(example[self.input_column]), "labels": labels}


class BucketBatchSampler(Sampler):
    """A batch sampler for sequence bucketing.

    This class creates buckets according to the length of examples. It first sorts the
    lengths and creates index map. Then it groups them into buckets and shuffle
    randomly. This makes each batch has examples of which lengths are almost same. It
    leads the decrement of unnecessary and wasted paddings, hence, you can reduce the
    padded sequence lengths and entire computational costs.

    Args:
        sequence_lengths: A list of sequence lengths.
        batch_size: The number of examples in each batch.
    """

    def __init__(self, sequence_lengths: Iterable[int], batch_size: int):
        indices = np.argsort(sequence_lengths)
        indices = indices[: len(indices) // batch_size * batch_size]
        self.buckets = indices.reshape(-1, batch_size)

    def __len__(self) -> int:
        return self.buckets.shape[0]

    def __iter__(self) -> Iterator[Iterable[int]]:
        for index in np.random.permutation(self.buckets.shape[0]):
            yield self.buckets[index]
