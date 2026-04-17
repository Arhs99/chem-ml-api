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
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from chemmlapi.configs.registry import load_config, resolve_model_dir


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data))
    return path


class RegistryConfigTests(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)

    def test_minimal_valid_config(self):
        cfg = _write(self.tmp_path / "config.json", {
            "assays": [{"name": "logD", "model_dir": "./models/logd"}]
        })
        entry = load_config(cfg).assays[0]
        self.assertEqual(entry.name, "logD")
        self.assertEqual(entry.model_dir, "./models/logd")
        self.assertEqual(entry.inverse_transform, "none")
        self.assertEqual(entry.ensemble_glob, "model_*")
        self.assertEqual(entry.batch_size, 64)

    def test_full_valid_config(self):
        cfg = _write(self.tmp_path / "config.json", {
            "assays": [{
                "name": "logD",
                "model_dir": "./models/logd",
                "inverse_transform": "log10",
                "ensemble_glob": "replicate_*",
                "batch_size": 128,
            }]
        })
        entry = load_config(cfg).assays[0]
        self.assertEqual(entry.inverse_transform, "log10")
        self.assertEqual(entry.ensemble_glob, "replicate_*")
        self.assertEqual(entry.batch_size, 128)

    def test_rejects_unknown_inverse_transform(self):
        cfg = _write(self.tmp_path / "config.json", {
            "assays": [{"name": "x", "model_dir": "./m", "inverse_transform": "sigmoid"}]
        })
        with self.assertRaises(ValidationError):
            load_config(cfg)

    def test_rejects_missing_model_dir(self):
        cfg = _write(self.tmp_path / "config.json", {"assays": [{"name": "x"}]})
        with self.assertRaises(ValidationError):
            load_config(cfg)

    def test_rejects_empty_assays(self):
        cfg = _write(self.tmp_path / "config.json", {"assays": []})
        with self.assertRaises(ValidationError):
            load_config(cfg)

    def test_rejects_duplicate_assay_names(self):
        cfg = _write(self.tmp_path / "config.json", {
            "assays": [
                {"name": "logD", "model_dir": "./a"},
                {"name": "logD", "model_dir": "./b"},
            ]
        })
        with self.assertRaises(ValidationError):
            load_config(cfg)

    def test_rejects_zero_batch_size(self):
        cfg = _write(self.tmp_path / "config.json", {
            "assays": [{"name": "x", "model_dir": "./m", "batch_size": 0}]
        })
        with self.assertRaises(ValidationError):
            load_config(cfg)

    def test_resolve_model_dir_is_config_relative(self):
        cfg = _write(self.tmp_path / "config.json", {
            "assays": [{"name": "x", "model_dir": "./sub/model"}]
        })
        entry = load_config(cfg).assays[0]
        self.assertEqual(
            resolve_model_dir(cfg, entry),
            (self.tmp_path / "sub" / "model").resolve(),
        )

    def test_resolve_model_dir_supports_parent_traversal(self):
        cfg_dir = self.tmp_path / "configs"
        cfg_dir.mkdir()
        cfg = _write(cfg_dir / "config.json", {
            "assays": [{"name": "x", "model_dir": "../models/m"}]
        })
        entry = load_config(cfg).assays[0]
        self.assertEqual(
            resolve_model_dir(cfg, entry),
            (self.tmp_path / "models" / "m").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
