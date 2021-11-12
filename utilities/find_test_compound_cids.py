import argparse
import json
from typing import List

import pandas as pd
import requests
import tqdm
from rdkit import Chem

PUBCHEM_ROOT = "https://pubchem.ncbi.nlm.nih.gov"
CIDS_BY_SMILES_QUERY = "/rest/pug/compound/smiles/cids/JSON"
CIDS_BY_FORMULA_QUERY = "/rest/pug/compound/formula/{}/JSON"
GET_FROM_LISTKEY_QUERY = "/rest/pug/compound/listkey/{}/cids/JSON"


def get_cids_by_smiles(smiles: str) -> List[int]:
    query_url = PUBCHEM_ROOT + CIDS_BY_SMILES_QUERY
    with requests.post(query_url, data={"smiles": smiles}) as resp:
        result = resp.json()
        if "IdentifierList" in result and result["IdentifierList"]["CID"][0] != 0:
            return result["IdentifierList"]["CID"]
    return []


def get_cids_by_formula(formula: str) -> List[int]:
    with requests.get(PUBCHEM_ROOT + CIDS_BY_FORMULA_QUERY.format(formula)) as resp:
        listkey = resp.json()["Waiting"]["ListKey"]
    query_url = PUBCHEM_ROOT + GET_FROM_LISTKEY_QUERY.format(listkey)

    for i in range(100):
        with requests.get(query_url) as resp:
            result = resp.json()
            if "Waiting" in result:
                if i == 99:
                    print("[*] Failed to get CIDs from listkey. Error occured.")
                continue
            if "IdentifierList" in result and result["IdentifierList"]["CID"][0] != 0:
                return result["IdentifierList"]["CID"]
    return []


def main(args: argparse.Namespace):
    test_df = pd.read_csv(args.test_file)
    test_smiles = test_df.SMILES

    test_cids = []
    for smiles in tqdm.tqdm(test_smiles):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), kekuleSmiles=True)
        cids = get_cids_by_smiles(smiles)

        if not cids:
            # If corresponding SMILES is not found, then just exclude all compounds
            # which have same formula.
            formula = Chem.MolToInchi(Chem.MolFromSmiles(smiles)).split("/")[1]
            cids = get_cids_by_formula(formula)
            print(
                f"[*] SMILES [{smiles}] not found."
                f" Instead, find {len(cids)} compounds by [{formula}]."
            )

        test_cids += cids

    with open("test_compound_cids.json", "w") as fp:
        json.dump(list(set(test_cids)), fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_file")
    args = parser.parse_args()

    main(args)
