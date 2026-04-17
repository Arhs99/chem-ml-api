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

import pytest
from pydantic import ValidationError

from chemmlapi.configs.registry import load_config, resolve_model_dir


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data))
    return path


def test_minimal_valid_config(tmp_path: Path):
    cfg = _write(tmp_path / "config.json", {
        "assays": [{"name": "logD", "model_dir": "./models/logd"}]
    })
    config = load_config(cfg)
    entry = config.assays[0]
    assert entry.name == "logD"
    assert entry.model_dir == "./models/logd"
    assert entry.inverse_transform == "none"
    assert entry.ensemble_glob == "model_*"
    assert entry.batch_size == 64


def test_full_valid_config(tmp_path: Path):
    cfg = _write(tmp_path / "config.json", {
        "assays": [{
            "name": "logD",
            "model_dir": "./models/logd",
            "inverse_transform": "log10",
            "ensemble_glob": "replicate_*",
            "batch_size": 128,
        }]
    })
    entry = load_config(cfg).assays[0]
    assert entry.inverse_transform == "log10"
    assert entry.ensemble_glob == "replicate_*"
    assert entry.batch_size == 128


def test_rejects_unknown_inverse_transform(tmp_path: Path):
    cfg = _write(tmp_path / "config.json", {
        "assays": [{"name": "x", "model_dir": "./m", "inverse_transform": "sigmoid"}]
    })
    with pytest.raises(ValidationError):
        load_config(cfg)


def test_rejects_missing_model_dir(tmp_path: Path):
    cfg = _write(tmp_path / "config.json", {"assays": [{"name": "x"}]})
    with pytest.raises(ValidationError):
        load_config(cfg)


def test_rejects_empty_assays(tmp_path: Path):
    cfg = _write(tmp_path / "config.json", {"assays": []})
    with pytest.raises(ValidationError):
        load_config(cfg)


def test_rejects_duplicate_assay_names(tmp_path: Path):
    cfg = _write(tmp_path / "config.json", {
        "assays": [
            {"name": "logD", "model_dir": "./a"},
            {"name": "logD", "model_dir": "./b"},
        ]
    })
    with pytest.raises(ValidationError):
        load_config(cfg)


def test_rejects_zero_batch_size(tmp_path: Path):
    cfg = _write(tmp_path / "config.json", {
        "assays": [{"name": "x", "model_dir": "./m", "batch_size": 0}]
    })
    with pytest.raises(ValidationError):
        load_config(cfg)


def test_resolve_model_dir_is_config_relative(tmp_path: Path):
    cfg = _write(tmp_path / "config.json", {
        "assays": [{"name": "x", "model_dir": "./sub/model"}]
    })
    entry = load_config(cfg).assays[0]
    assert resolve_model_dir(cfg, entry) == (tmp_path / "sub" / "model").resolve()


def test_resolve_model_dir_supports_parent_traversal(tmp_path: Path):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    cfg = _write(cfg_dir / "config.json", {
        "assays": [{"name": "x", "model_dir": "../models/m"}]
    })
    entry = load_config(cfg).assays[0]
    assert resolve_model_dir(cfg, entry) == (tmp_path / "models" / "m").resolve()
