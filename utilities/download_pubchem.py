import argparse
import ftplib
import json
import multiprocessing as mp
import os
import zlib
from base64 import b64encode
from typing import Dict, List, Optional, Set

import numpy as np
import tqdm

PUBCHEM_FTP_HOST = "ftp.ncbi.nlm.nih.gov"
COMPOUND_3D_BASE = "/pubchem/Compound_3D/01_conf_per_cmpd/SDF/"

PUBCHEM_COLUMNS = [
    "cid",
    "conformer_rmsd",
    "mmff94_energy",
    "shape_selfoverlap",
    "feature_selfoverlap",
    "structure",
]
PUBCHEM_ATTRIBUTES = [
    "PUBCHEM_COMPOUND_CID",
    "PUBCHEM_CONFORMER_RMSD",
    "PUBCHEM_MMFF94_ENERGY",
    "PUBCHEM_SHAPE_SELFOVERLAP",
    "PUBCHEM_FEATURE_SELFOVERLAP",
]


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


def create_example_from_sdf(data: str, cids: Set) -> Optional[str]:
    structure, attributes = data.split("M  END")
    attributes = {
        a[: a.index(">")]: a[a.index(">") + 1 :].strip()
        for a in attributes.split("> <")[1:]
    }

    if (
        "PUBCHEM_COMPOUND_CID" not in attributes
        or int(attributes["PUBCHEM_COMPOUND_CID"]) in cids
    ):
        return None
    cids.add(int(attributes["PUBCHEM_COMPOUND_CID"]))

    if any(attr not in attributes for attr in PUBCHEM_ATTRIBUTES):
        return None
    items = [attributes[attr] for attr in PUBCHEM_ATTRIBUTES]

    structure = parse_mol_structure(structure.lstrip())
    structure = zlib.compress(json.dumps(structure).encode())
    structure = b64encode(structure).decode()
    return ",".join(items + [structure])


def download_and_save_sdf(remote_file: str, output_file: str):
    buffers, cids = "", set()
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)

    with open(output_file, "w") as fp:
        # This function is a callback for collecting chunked buffers, decompress the
        # buffers, split them into multiple SDF data and write the parsed examples.
        def retrieve_callback(data: bytes):
            nonlocal buffers
            buffers += decoder.decompress(data).decode()

            for sdf_data in buffers.split("$$$$")[:-1]:
                example = create_example_from_sdf(sdf_data, cids)
                if example is not None:
                    fp.write(example + "\n")

            buffers = buffers.split("$$$$")[-1]

        with ftplib.FTP(PUBCHEM_FTP_HOST, user="anonymous") as ftp:
            ftp.retrbinary(f"RETR {remote_file}", retrieve_callback)


def download_process(
    process_idx: int,
    remote_files: List[str],
    output_dir: str,
    max_trials: int = 10,
):
    if process_idx == 0:
        remote_files = tqdm.tqdm(remote_files)

    for remote_file in remote_files:
        name = "compound_3d_" + os.path.basename(remote_file).replace(".sdf.gz", "")
        output_file = os.path.join(output_dir, name + ".csv")

        for i in range(max_trials):
            try:
                download_and_save_sdf(remote_file, output_file)
                break
            except Exception:
                print(f"[*] {name}; attempt {i} is failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="pubchem_compounds")
    parser.add_argument("--num_cores", type=int, default=mp.cpu_count())
    parser.add_argument("--max_trials", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with ftplib.FTP(PUBCHEM_FTP_HOST, user="anonymous") as ftp:
        remote_files = [f for f in ftp.nlst(COMPOUND_3D_BASE) if f.endswith(".sdf.gz")]

    processes = []
    for i, remote_files in enumerate(np.array_split(remote_files, args.num_cores)):
        process = mp.Process(
            target=download_process,
            args=(i, remote_files.tolist(), args.output_dir, args.max_trials),
            daemon=True,
        )
        process.start()
        processes.append(process)

    for p in processes:
        p.join()
