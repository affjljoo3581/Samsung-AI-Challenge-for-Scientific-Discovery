import argparse
import json
import os
import pickle
import zlib
from base64 import b64decode
from typing import Iterable, Optional, Tuple

import numpy as np
import tqdm


def create_file_descriptions(
    filename: str, input_column: str, random_seed: Optional[int] = None
) -> Tuple[Iterable[int], Iterable[int]]:
    """Create seeking points and sequence lengths of dataset.

    Args:
        filename: The name of dataset file.
        input_column: The name of input column.
        random_seed: A random seed of index shuffling. Note that this function will
            shuffle the collected seeking points and text lengths. Default is `None`.

    Returns:
        The collected seeking points and sequence lengths of the dataset.
    """
    seeking_points, sequence_lengths = [], []

    # Get total number of lines in the file to show the progress bar for calculating and
    # creating file index.
    total_lines = 0
    with open(filename, "r") as fp:
        for _ in fp:
            total_lines += 1

    with open(filename, "r") as fp:
        columns = fp.readline().strip().split(",")
        input_column_idx = columns.index(input_column)

        last_seeking_point = fp.tell()
        for line in tqdm.tqdm(fp, total=total_lines - 1):
            structure = line.strip().split(",")[input_column_idx]
            structure = json.loads(zlib.decompress(b64decode(structure)).decode())
            sequence_lengths.append(len(structure["atoms"]))

            seeking_points.append(last_seeking_point)
            last_seeking_point += len(line)

    if random_seed is not None:
        np.random.seed(random_seed)
    random_indices = np.random.permutation(len(seeking_points))

    seeking_points = np.array(seeking_points)
    sequence_lengths = np.array(sequence_lengths)
    return seeking_points[random_indices], sequence_lengths[random_indices]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--input_column", default="structure")
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    seeking_points, sequence_lengths = create_file_descriptions(
        args.input_file, args.input_column, args.random_seed
    )
    seeking_points = seeking_points.tolist()
    sequence_lengths = sequence_lengths.tolist()

    with open(os.path.splitext(args.input_file)[0] + ".index", "wb") as fp:
        pickle.dump((seeking_points, sequence_lengths), fp)
