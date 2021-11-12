import argparse
import json
import lzma
import multiprocessing as mp
import zlib
from base64 import b64encode
from typing import Dict, List

import pandas as pd
import requests
import tqdm

PUBCHEMQC_BASE_URL = "http://pubchemqc.riken.jp/"
PUBCHEMQC_EXCITED_POSTFIX = ".td-b3lyp_6-31g+(d).log.xz"

PUBCHEMQC_COLUMNS = [
    "cid",
    "s1_energy",
    "s2_energy",
    "s3_energy",
    "s4_energy",
    "s5_energy",
    "s6_energy",
    "s7_energy",
    "s8_energy",
    "s9_energy",
    "s10_energy",
    "structure",
]


def parse_gamess_log(data: str) -> List[float]:
    excitation_energies = []
    for line in data[data.rindex("SINGLET EXCITATIONS") :].splitlines():
        line = line.strip()
        if line.startswith("STATE") and line.endswith("EV"):
            excitation_energies.append(float(line.split()[5]))
    return excitation_energies


def parse_mol_structure(data: str) -> Dict:
    data = data.splitlines()[3:]
    num_atoms, num_bonds = int(data[0][:3]), int(data[0][3:6])

    atoms = []
    for line in data[1 : 1 + num_atoms]:
        x, y, z = float(line[:10]), float(line[10:20]), float(line[20:30])
        x, y, z = round(x, 2), round(y, 2), round(z, 2)
        charge = [0, 3, 2, 1, "^", -1, -2, -3][int(line[36:39])]
        atoms.append([x, y, z, line[31:34].strip(), charge])

    bonds = []
    for line in data[1 + num_atoms : 1 + num_atoms + num_bonds]:
        bonds.append([int(line[:3]) - 1, int(line[3:6]) - 1, int(line[6:9])])

    for line in data[1 + num_atoms + num_bonds :]:
        if not line.startswith("M  CHG") and not line.startswith("M  RAD"):
            continue
        for i in range(int(line[6:9])):
            idx = int(line[10 + 8 * i : 13 + 8 * i]) - 1
            value = int(line[14 + 8 * i : 17 + 8 * i])

            atoms[idx][4] = (
                [":", "^", "^^"][value - 1] if line.startswith("M  RAD") else value
            )

    return {"atoms": atoms, "bonds": bonds}


def collect_urls_from_group(group: str):
    with requests.get(PUBCHEMQC_BASE_URL + group + ".html") as resp:
        return [
            line[line.index('href="') + 6 : line.index(PUBCHEMQC_EXCITED_POSTFIX)]
            for line in resp.text.splitlines()
            if PUBCHEMQC_EXCITED_POSTFIX in line
        ]


def download_energies_and_structure(compound: str) -> Dict:
    try:
        logfile_path = PUBCHEMQC_BASE_URL + compound + PUBCHEMQC_EXCITED_POSTFIX
        molfile_path = PUBCHEMQC_BASE_URL + compound + ".mol"

        with requests.get(logfile_path, stream=True) as resp:
            with lzma.LZMAFile(resp.raw, "r") as fp:
                energies = parse_gamess_log(fp.read().decode())
                energies = [round(e, 2) for e in energies]

        with requests.get(molfile_path) as resp:
            structure = parse_mol_structure(resp.text)

        # Compress the structure JSON string to reduce the entire file size.
        structure = json.dumps(structure)
        structure = zlib.compress(structure.encode())
        structure = b64encode(structure).decode()

        row = [int(compound.split("/")[-1])] + energies + [structure]
        return dict(zip(PUBCHEMQC_COLUMNS, row))
    except Exception:
        return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", type=int, default=mp.cpu_count())
    parser.add_argument("--subset", default=None)
    args = parser.parse_args()

    # Collect compound groups from the main page.
    with requests.get(PUBCHEMQC_BASE_URL) as resp:
        data = resp.text
        groups = [
            line[line.index(">") + 1 : line.rindex("</a>")]
            for line in data[data.index("<pre>") : data.index("</pre>")].splitlines()
            if "<a href=" in line
        ]

    # Split the compound groups into subset.
    if args.subset is not None:
        idx, total = map(int, args.subset.split("/"))
        length = len(groups) // total

        groups = groups[(idx - 1) * length :]
        if idx < total:
            groups = groups[:length]

    # Collect all available (and excited) compound URLs by using multiprocessing.
    with mp.Pool(args.cores) as pool:
        iterator = pool.imap_unordered(collect_urls_from_group, groups)

        compound_urls = []
        for urls in tqdm.tqdm(iterator, total=len(groups)):
            compound_urls.extend(urls)

    # Download excitation energies and molecular structure by using multiprocessing.
    with mp.Pool(args.cores) as pool:
        iterator = pool.imap_unordered(download_energies_and_structure, compound_urls)
        compound_dataset = list(tqdm.tqdm(iterator, total=len(compound_urls)))

    dataset = pd.DataFrame(compound_dataset).dropna()
    dataset.to_csv("pubchemqc-excitations-3m.csv", index=False)
