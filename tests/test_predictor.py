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

import numpy as np
import pandas as pd

from chemmlapi.core.loader import load_registry
from chemmlapi.core.predictor import predict

from tests._markers import LOGD_CHECKPOINT_DIR, REQUIRES_LOGD


CURATED_SMILES = [
    "CCO",
    "CCN",
    "c1ccccc1",
    "CC(=O)O",
    "CN1CCCC1c1cccnc1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Cn1cnc2n(C)c(=O)n(C)c(=O)c12",
    "O=C(O)c1ccccc1O",
    "OC(=O)Cc1ccccc1",
    "CCc1ccccc1",
]


@REQUIRES_LOGD
class LogDPredictorIntegrationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cfg = Path(cls._tmp.name) / "config.json"
        cfg.write_text(json.dumps({
            "assays": [{
                "name": "logD",
                "model_dir": str(LOGD_CHECKPOINT_DIR),
                "inverse_transform": "none",
            }]
        }))
        cls.registry = load_registry(cfg)
        cls.df = predict(cls.registry["logD"], CURATED_SMILES)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_one_row_per_input(self):
        self.assertEqual(len(self.df), len(CURATED_SMILES))

    def test_smiles_column_preserves_order(self):
        self.assertEqual(list(self.df["smiles"]), list(CURATED_SMILES))

    def test_prediction_is_finite_float(self):
        preds = np.asarray(self.df["prediction"], dtype=float)
        self.assertTrue(np.isfinite(preds).all())

    def test_reference_round_trip(self):
        ref_path = Path(__file__).resolve().parent / "fixtures" / "reference_predictions.csv"
        if not ref_path.is_file():
            self.skipTest(f"no local reference csv at {ref_path}")
        ref = pd.read_csv(ref_path)
        self.assertEqual(list(self.df["smiles"]), list(ref["smiles"]))
        np.testing.assert_allclose(
            np.asarray(self.df["prediction"], dtype=float),
            np.asarray(ref["prediction"], dtype=float),
            atol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
