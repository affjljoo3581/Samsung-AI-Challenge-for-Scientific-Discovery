from typing import Dict, List, Optional, Union

import numpy as np
import torch

MOLECULAR_ATOMS = (
    "H,He,Li,Be,B,C,N,O,F,Ne,Na,Mg,Al,Si,P,S,Cl,Ar,K,Ca,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,"
    "Ga,Ge,As,Se,Br,Kr,Rb,Sr,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Sb,Te,I,Xe,Cs,Ba,La,Ce,"
    "Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi,Po,At,"
    "Rn,Fr,Ra,Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr,Rf,Db,Sg,Bh,Hs,Mt,Ds,Rg,Cn,"
    "Nh,Fl,Mc,Lv,Ts,Og"
).split(",")

MOLECULAR_CHARGES = list(range(-15, 16)) + [":", "^", "^^"]
MOLECULAR_BOND_TYPES = [1, 2, 3, 4, 5, 6, 7, 8]


class MolecularEncoder:
    """Molecular structure encoder class.

    This class is a kind of tokenizers for MoT model. Every transformer models have
    their own subword tokenizers (and it even creates attention masks), and of course
    MoT needs its own input encoder. While 3D-molecular structure data is not as simple
    as sentences which are used in common transformer model, we create new input encoder
    which creates input encodings from the 3D-molecular structure data. Using this, you
    can simply encode the structure data and pass to the MoT model.

    Args:
        cls_token: The name of classification token. Default is `[CLS]`.
        pad_token: The name of padding token. Default is `[PAD]`.
    """

    # This field is a part of MoT configurations. If you are using MoT model with this
    # encoder class, then you can simply define the number of embeddings and attention
    # types using this field. The vocabularies are predefined, so you do not need to
    # handle the vocabulary sizes.
    mot_config = dict(
        num_embeddings=[len(MOLECULAR_ATOMS) + 2, len(MOLECULAR_CHARGES) + 2],
        num_attention_types=len(MOLECULAR_BOND_TYPES) + 2,
    )

    def __init__(
        self,
        cls_token: str = "[CLS]",
        pad_token: str = "[PAD]",
    ):
        self.vocab1 = [pad_token, cls_token] + MOLECULAR_ATOMS
        self.vocab2 = [pad_token, cls_token] + MOLECULAR_CHARGES
        self.vocab3 = [pad_token, cls_token] + MOLECULAR_BOND_TYPES
        self.cls_token = cls_token
        self.pad_token = pad_token

    def collect_input_sequences(self, molecular: Dict[str, List]) -> Dict[str, List]:
        """Collect input sequences from the molecular structure data.

        Args:
            molecular: The molecular data which contains 3D atoms and their bonding
                informations.

        Returns:
            A dictionary which contains the input tokens and 3d positions of the atoms.
        """
        input_ids = [
            [self.vocab1.index(self.cls_token)],
            [self.vocab2.index(self.cls_token)],
        ]
        position_ids = [[0.0, 0.0, 0.0]]
        attention_mask = [1] * (len(molecular["atoms"]) + 1)

        for atom in molecular["atoms"]:
            input_ids[0].append(self.vocab1.index(atom[3]))
            input_ids[1].append(self.vocab2.index(atom[4]))
            position_ids.append(atom[:3])

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def create_attention_type_ids(self, molecular: Dict) -> np.ndarray:
        """Create an attention types from the molecular structure data.

        MoT supports attention types which are applied to the attention scores
        relatively. Using this, you can give attention weights (bond types) directly to
        the self-attention module. This method creates the attention type array by using
        the bond informations in the molecular structure.

        Args:
            molecular: The molecular data which contains 3D atoms and their bonding
                informations.

        Returns:
            The attention type array from the bond informations.
        """
        max_seq_len = len(molecular["atoms"]) + 1

        attention_type_ids = np.empty((max_seq_len, max_seq_len), dtype=np.int64)
        attention_type_ids.fill(self.vocab3.index(self.pad_token))

        attention_type_ids[0, :] = self.vocab3.index(self.cls_token)
        attention_type_ids[:, 0] = self.vocab3.index(self.cls_token)

        for first, second, bond_type in molecular["bonds"]:
            attention_type_ids[first + 1, second + 1] = self.vocab3.index(bond_type)
            attention_type_ids[second + 1, first + 1] = self.vocab3.index(bond_type)

        return attention_type_ids

    def encode(self, molecular: Dict[str, List]) -> Dict[str, Union[List, np.ndarray]]:
        """Encode the molecular structure data to the model inputs.

        Args:
            molecular: The molecular data which contains 3D atoms and their bonding
                informations.

        Returns:
            An encoded output which contains input ids, 3d positions, position mask, and
            attention types.
        """
        return {
            **self.collect_input_sequences(molecular),
            "attention_type_ids": self.create_attention_type_ids(molecular),
        }

    def collate(
        self,
        encodings: List[Dict[str, Union[List, np.ndarray]]],
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> Dict[str, Union[List, np.ndarray, torch.Tensor]]:
        """Collate the encodings of which lengths are different to each other.

        The lengths of encoded molecular structure data are not exactly same. To group
        the sequences into the batch requires equal lengths. To resolve the problem,
        this class supports sequence and attention mask paddings. Using this, you can
        pad the encodings to desired lengths or match to the longest sequences. In
        addition, this method automatically converts the sequences to torch tensors.

        Args:
            encodings: The batch of encodings.
            max_length: The desired maximum length of sequences. Default is `None`.
            pad_to_multiple_of: To match the sequence length to be multiple of certain
                factor. Default is `None`.

        Returns:
            The collated batched encodings which contain converted tensors.
        """
        longest_length = max(len(enc["input_ids"][0]) for enc in encodings)
        max_length = min(max_length or longest_length, longest_length)

        if pad_to_multiple_of is not None:
            max_length = max_length + pad_to_multiple_of - 1
            max_length = max_length // pad_to_multiple_of * pad_to_multiple_of

        padding_id_1 = self.vocab1.index(self.pad_token)
        padding_id_2 = self.vocab2.index(self.pad_token)

        for enc in encodings:
            num_paddings = max_length - len(enc["input_ids"][0])

            if num_paddings >= 0:
                enc["input_ids"][0] += [padding_id_1] * num_paddings
                enc["input_ids"][1] += [padding_id_2] * num_paddings
                enc["position_ids"] += [[0.0, 0.0, 0.0]] * num_paddings
                enc["attention_mask"] += [0] * num_paddings
                enc["attention_type_ids"] = np.pad(
                    enc["attention_type_ids"],
                    pad_width=((0, num_paddings), (0, num_paddings)),
                    constant_values=self.vocab3.index(self.pad_token),
                )
            else:
                # If the encoded sequences are longer than the maximum length, then
                # truncate the sequences and attention mask.
                enc["input_ids"][0] = enc["input_ids"][0][:max_length]
                enc["input_ids"][1] = enc["input_ids"][1][:max_length]
                enc["position_ids"] = enc["position_ids"][:max_length]
                enc["attention_mask"] = enc["attention_mask"][:max_length]
                enc["attention_type_ids"] = enc["attention_type_ids"][
                    :max_length, :max_length
                ]

        # Collect all sequences into their batch and convert them to torch tensor. After
        # that, you can use the sequences to the model because all inputs are converted
        # to the tensors. Since we use two `input_ids` and handle them on the list, they
        # will be converted individually.
        encodings = {k: [enc[k] for enc in encodings] for k in encodings[0]}
        encodings["input_ids"] = [
            torch.tensor([x[0] for x in encodings["input_ids"]]),
            torch.tensor([x[1] for x in encodings["input_ids"]]),
        ]
        encodings["position_ids"] = torch.tensor(encodings["position_ids"])
        encodings["attention_mask"] = torch.tensor(encodings["attention_mask"])
        encodings["attention_type_ids"] = torch.tensor(encodings["attention_type_ids"])

        if "labels" in encodings:
            encodings["labels"] = torch.tensor(encodings["labels"])

        return encodings
