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

import math
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from chemmlapi.core.loader import LoadedAssay, load_registry
from chemmlapi.core.predictor import predict

_WORKER_REGISTRY: dict[str, LoadedAssay] = {}


def _worker_init(config_path: str) -> None:
    import torch

    torch.set_num_threads(1)
    global _WORKER_REGISTRY
    _WORKER_REGISTRY = load_registry(config_path)


def _worker_predict(assay_name: str, smiles_chunk: list[str]) -> pd.DataFrame:
    return predict(_WORKER_REGISTRY[assay_name], smiles_chunk)


def _prime(_i: int) -> int:
    return 0


def chunk_predict(
    pool: ProcessPoolExecutor,
    assay: str,
    smiles: list[str],
    n_processes: int,
) -> pd.DataFrame:
    if not smiles:
        return pd.DataFrame({"smiles": [], "prediction": []})
    n = min(n_processes, max(1, len(smiles)))
    chunk = math.ceil(len(smiles) / n)
    chunks = [smiles[i : i + chunk] for i in range(0, len(smiles), chunk)]
    dfs = list(pool.map(_worker_predict, [assay] * len(chunks), chunks))
    return pd.concat(dfs, ignore_index=True)
