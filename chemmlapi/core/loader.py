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

from dataclasses import dataclass
from pathlib import Path

from chemprop.models import MPNN
from chemprop.nn.predictors import MveFFN

from chemmlapi.configs.registry import load_config, resolve_model_dir


@dataclass
class LoadedAssay:
    name: str
    models: list[MPNN]
    is_mve: bool
    batch_size: int


def _find_checkpoints(model_dir: Path, ensemble_glob: str) -> list[Path]:
    subdirs = sorted(p for p in model_dir.glob(ensemble_glob) if p.is_dir())
    candidates: list[Path] = []
    for d in subdirs:
        best = d / "best.pt"
        if best.is_file():
            candidates.append(best)
            continue
        ckpt_dir = d / "checkpoints"
        if ckpt_dir.is_dir():
            ckpts = sorted(ckpt_dir.glob("*.ckpt"))
            if ckpts:
                candidates.append(ckpts[0])

    if candidates:
        return candidates

    root_best = model_dir / "best.pt"
    if root_best.is_file():
        return [root_best]
    root_ckpt_dir = model_dir / "checkpoints"
    if root_ckpt_dir.is_dir():
        root_ckpts = sorted(root_ckpt_dir.glob("*.ckpt"))
        if root_ckpts:
            return [root_ckpts[0]]

    raise FileNotFoundError(
        f"No chemprop checkpoints (best.pt or checkpoints/*.ckpt) found under "
        f"{model_dir} with ensemble_glob={ensemble_glob!r}"
    )


def _load_single(path: Path) -> MPNN:
    if path.suffix == ".pt":
        model = MPNN.load_from_file(str(path), map_location="cpu")
    else:
        model = MPNN.load_from_checkpoint(str(path), map_location="cpu")
    return model.eval()


def load_registry(config_path: str | Path) -> dict[str, LoadedAssay]:
    config = load_config(config_path)
    registry: dict[str, LoadedAssay] = {}
    for entry in config.assays:
        model_dir = resolve_model_dir(config_path, entry)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"model_dir does not exist: {model_dir}")
        paths = _find_checkpoints(model_dir, entry.ensemble_glob)
        models = [_load_single(p) for p in paths]
        registry[entry.name] = LoadedAssay(
            name=entry.name,
            models=models,
            is_mve=isinstance(models[0].predictor, MveFFN),
            batch_size=entry.batch_size,
        )
    return registry
