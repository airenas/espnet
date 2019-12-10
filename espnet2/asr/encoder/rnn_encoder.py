import logging
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.rnn.encoders import RNN
from espnet.nets.pytorch_backend.rnn.encoders import RNNP
from espnet.nets.pytorch_backend.rnn.encoders import VGG2L
from espnet2.asr.encoder.abs_encoder import AbsEncoder


class Encoder(AbsEncoder):
    def __init__(
        self,
        idim: int,
        etype: str = "blstmp",
        elayers: int = 4,
        eunits: int = 320,
        eprojs: int = 320,
        dropout: float = 0.0,
        subsample: Optional[Sequence[int]] = (2, 2, 1, 1),
        in_channel: int = 1,
    ):
        assert check_argument_types()
        super().__init__()
        self.eprojs = eprojs
        self.etype = etype

        typ = etype.lstrip("vgg").rstrip("p")
        if typ not in ["lstm", "gru", "blstm", "bgru"]:
            raise ValueError(
                "Error: need to specify an appropriate encoder architecture"
            )

        if subsample is None:
            subsample = np.ones(elayers + 1, dtype=np.int)
        else:
            subsample = subsample[:elayers]
            # Append 1 at the beginning because the second or later is used
            subsample = np.pad(
                np.array(subsample, dtype=np.int),
                [1, elayers - len(subsample)],
                mode="constant",
                constant_values=1,
            )

        if etype.startswith("vgg"):
            if etype[-1] == "p":
                self.enc = torch.nn.ModuleList(
                    [
                        VGG2L(in_channel),
                        RNNP(
                            get_vgg2l_odim(idim, in_channel=in_channel),
                            elayers,
                            eunits,
                            eprojs,
                            subsample,
                            dropout,
                            typ=typ,
                        ),
                    ]
                )
                logging.info("Use CNN-VGG + " + typ.upper() + "P for encoder")
            else:
                self.enc = torch.nn.ModuleList(
                    [
                        VGG2L(in_channel),
                        RNN(
                            get_vgg2l_odim(idim, in_channel=in_channel),
                            elayers,
                            eunits,
                            eprojs,
                            dropout,
                            typ=typ,
                        ),
                    ]
                )
                logging.info("Use CNN-VGG + " + typ.upper() + " for encoder")
        else:
            if etype[-1] == "p":
                self.enc = torch.nn.ModuleList(
                    [
                        RNNP(
                            idim,
                            elayers,
                            eunits,
                            eprojs,
                            subsample,
                            dropout,
                            typ=typ,
                        )
                    ]
                )
                logging.info(
                    typ.upper() + " with every-layer projection for encoder"
                )
            else:
                self.enc = torch.nn.ModuleList(
                    [RNN(idim, elayers, eunits, eprojs, dropout, typ=typ)]
                )
                logging.info(typ.upper() + " without projection for encoder")

    def out_dim(self) -> int:
        return self.eprojs

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prev_states is None:
            prev_states = [None] * len(self.enc)
        assert len(prev_states) == len(self.enc)

        current_states = []
        for module, prev_state in zip(self.enc, prev_states):
            xs_pad, ilens, states = module(
                xs_pad, ilens, prev_state=prev_state
            )
            current_states.append(states)

        if self.etype.endswith("p"):
            xs_pad.masked_fill_(make_pad_mask(ilens, xs_pad, 1), 0.0)
        else:
            xs_pad = xs_pad.masked_fill(make_pad_mask(ilens, xs_pad, 1), 0.0)
        return xs_pad, ilens, current_states