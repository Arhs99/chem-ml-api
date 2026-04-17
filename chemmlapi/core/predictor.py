# chem-ml-api - FastAPI inference service for chemprop ADMET models
# Copyright (C) 2026  Kostas Papadopoulos <kostasp97@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader

from chemmlapi.core.loader import LoadedAssay


def _forward_all(assay: LoadedAssay, smiles: list[str]) -> np.ndarray:
    dps = [MoleculeDatapoint.from_smi(s) for s in smiles]
    dataset = MoleculeDataset(dps)

    stack: list[np.ndarray] = []
    for model in assay.models:
        loader = build_dataloader(
            dataset,
            batch_size=assay.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loader:
                y = model(batch.bmg, batch.V_d, batch.X_d)
                chunks.append(y.detach().cpu().numpy())
        stack.append(np.concatenate(chunks, axis=0))
    return np.stack(stack, axis=0)


def predict(assay: LoadedAssay, smiles: list[str]) -> pd.DataFrame:
    if not smiles:
        cols = {"smiles": [], "prediction": []}
        return pd.DataFrame(cols)

    stacked = _forward_all(assay, smiles)

    if assay.is_mve:
        means_per_model = stacked[..., 0, 0]
        vars_per_model = stacked[..., 0, 1]
        if len(assay.models) == 1:
            mean = means_per_model[0]
            std = np.sqrt(vars_per_model[0])
        else:
            mean = means_per_model.mean(axis=0)
            total_var = vars_per_model.mean(axis=0) + means_per_model.var(axis=0)
            std = np.sqrt(total_var)
    else:
        pred_per_model = stacked[..., 0]
        if len(assay.models) == 1:
            mean = pred_per_model[0]
            std = None
        else:
            mean = pred_per_model.mean(axis=0)
            std = pred_per_model.std(axis=0, ddof=0)

    out: dict[str, object] = {
        "smiles": list(smiles),
        "prediction": np.asarray(mean, dtype=np.float64),
    }
    if std is not None:
        out["std"] = np.asarray(std, dtype=np.float64)
    return pd.DataFrame(out)
