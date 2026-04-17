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

from chemmlapi.core.loader import load_registry

from tests._markers import LOGD_CHECKPOINT_DIR, REQUIRES_LOGD


class LoaderErrorTests(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)

    def test_missing_model_dir_raises(self):
        cfg = self.tmp_path / "config.json"
        cfg.write_text(json.dumps({
            "assays": [{"name": "x", "model_dir": "./nowhere"}]
        }))
        with self.assertRaises(FileNotFoundError):
            load_registry(cfg)

    def test_empty_model_dir_raises(self):
        empty = self.tmp_path / "models" / "empty"
        empty.mkdir(parents=True)
        cfg = self.tmp_path / "config.json"
        cfg.write_text(json.dumps({
            "assays": [{"name": "x", "model_dir": "./models/empty"}]
        }))
        with self.assertRaises(FileNotFoundError):
            load_registry(cfg)


@REQUIRES_LOGD
class LogDLoaderIntegrationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cfg = Path(cls._tmp.name) / "config.json"
        cfg.write_text(json.dumps({
            "assays": [{
                "name": "logD",
                "model_dir": str(LOGD_CHECKPOINT_DIR),
            }]
        }))
        cls.registry = load_registry(cfg)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_logd_assay_is_registered(self):
        self.assertIn("logD", self.registry)

    def test_at_least_one_model_loaded(self):
        self.assertGreaterEqual(len(self.registry["logD"].models), 1)

    def test_is_mve_is_bool(self):
        self.assertIsInstance(self.registry["logD"].is_mve, bool)


if __name__ == "__main__":
    unittest.main()
