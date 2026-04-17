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

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class RegistryEntry(BaseModel):
    name: str = Field(min_length=1)
    model_dir: str = Field(min_length=1)
    ensemble_glob: str = Field(default="model_*", min_length=1)
    batch_size: int = Field(default=64, gt=0)


class RegistryConfig(BaseModel):
    assays: list[RegistryEntry] = Field(min_length=1)

    @field_validator("assays")
    @classmethod
    def _unique_names(cls, v: list[RegistryEntry]) -> list[RegistryEntry]:
        names = [e.name for e in v]
        if len(names) != len(set(names)):
            raise ValueError("assay names must be unique")
        return v


def load_config(config_path: str | Path) -> RegistryConfig:
    with open(config_path) as f:
        data = json.load(f)
    return RegistryConfig.model_validate(data)


def resolve_model_dir(config_path: str | Path, entry: RegistryEntry) -> Path:
    base = Path(config_path).resolve().parent
    return (base / entry.model_dir).resolve()
