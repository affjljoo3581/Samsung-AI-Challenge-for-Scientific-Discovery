import argparse

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("submissions", nargs="+")
    parser.add_argument("output")
    args = parser.parse_args()

    ensembled = [pd.read_csv(name, index_col="uid") for name in args.submissions]
    (sum(ensembled) / len(ensembled)).to_csv(args.output)
