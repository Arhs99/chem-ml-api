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
import os
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from tests._markers import LOGD_CHECKPOINT_DIR, REQUIRES_LOGD


class PredictRequestSchemaTests(unittest.TestCase):

    def _import(self):
        from chemmlapi.interfaces.api import PredictRequest
        return PredictRequest

    def test_rejects_empty_smiles_list(self):
        PredictRequest = self._import()
        with self.assertRaises(ValidationError):
            PredictRequest(assay="logD", smiles=[])

    def test_rejects_too_many_smiles(self):
        PredictRequest = self._import()
        with self.assertRaises(ValidationError):
            PredictRequest(assay="logD", smiles=["CCO"] * 10_001)

    def test_rejects_missing_assay(self):
        PredictRequest = self._import()
        with self.assertRaises(ValidationError):
            PredictRequest(smiles=["CCO"])

    def test_rejects_blank_assay(self):
        PredictRequest = self._import()
        with self.assertRaises(ValidationError):
            PredictRequest(assay="", smiles=["CCO"])

    def test_valid_request_defaults(self):
        PredictRequest = self._import()
        req = PredictRequest(assay="logD", smiles=["CCO"])
        self.assertEqual(req.assay, "logD")
        self.assertEqual(req.smiles, ["CCO"])
        self.assertFalse(req.include_std)


_API_KEY = "test-key-7e4fa3"


@REQUIRES_LOGD
class ApiIntegrationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        cfg_path = Path(cls._tmp.name) / "config.json"
        cfg_path.write_text(json.dumps({
            "assays": [{
                "name": "logD",
                "model_dir": str(LOGD_CHECKPOINT_DIR),
            }]
        }))

        cls._prev_env = {
            k: os.environ.get(k)
            for k in ("CHEMML_CONFIG", "CHEMML_PROCESSES", "CHEMML_API_KEY")
        }
        os.environ["CHEMML_CONFIG"] = str(cfg_path)
        os.environ["CHEMML_PROCESSES"] = "2"
        os.environ["CHEMML_API_KEY"] = _API_KEY
        cls.addClassCleanup(cls._restore_env)

        from fastapi.testclient import TestClient

        from chemmlapi.interfaces.api import app

        client_cm = TestClient(app)
        cls.client = client_cm.__enter__()
        cls.addClassCleanup(client_cm.__exit__, None, None, None)

    @classmethod
    def _restore_env(cls):
        for k, v in cls._prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_health_does_not_require_api_key(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_assays_rejects_missing_key(self):
        r = self.client.get("/assays")
        self.assertEqual(r.status_code, 401)

    def test_assays_with_key_lists_logD(self):
        r = self.client.get("/assays", headers={"X-API-Key": _API_KEY})
        self.assertEqual(r.status_code, 200)
        self.assertIn("logD", r.json()["assays"])

    def test_predict_wrong_key_rejected(self):
        r = self.client.post(
            "/predict",
            headers={"X-API-Key": "wrong"},
            json={"assay": "logD", "smiles": ["CCO"]},
        )
        self.assertEqual(r.status_code, 401)

    def test_predict_returns_one_result_per_valid_smiles(self):
        r = self.client.post(
            "/predict",
            headers={"X-API-Key": _API_KEY},
            json={"assay": "logD", "smiles": ["CCO", "c1ccccc1"]},
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(len(body["results"]), 2)
        for row in body["results"]:
            self.assertIsInstance(row["prediction"], float)
            self.assertIsNone(row["std"])
        meta = body["metadata"]
        self.assertEqual(meta["molecules_processed"], 2)
        self.assertEqual(meta["molecules_invalid"], 0)
        self.assertEqual(meta["processes"], 2)

    def test_predict_mixed_valid_and_invalid(self):
        r = self.client.post(
            "/predict",
            headers={"X-API-Key": _API_KEY},
            json={"assay": "logD", "smiles": ["CCO", "not-a-smiles-string"]},
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(len(body["results"]), 1)
        self.assertEqual(body["metadata"]["molecules_invalid"], 1)

    def test_predict_all_invalid_returns_422(self):
        r = self.client.post(
            "/predict",
            headers={"X-API-Key": _API_KEY},
            json={"assay": "logD", "smiles": ["not-a-smiles-string"]},
        )
        self.assertEqual(r.status_code, 422)

    def test_predict_unknown_assay_returns_404(self):
        r = self.client.post(
            "/predict",
            headers={"X-API-Key": _API_KEY},
            json={"assay": "nope", "smiles": ["CCO"]},
        )
        self.assertEqual(r.status_code, 404)

    def test_predict_include_std_is_null_for_single_non_mve(self):
        r = self.client.post(
            "/predict",
            headers={"X-API-Key": _API_KEY},
            json={"assay": "logD", "smiles": ["CCO"], "include_std": True},
        )
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.json()["results"][0]["std"])


if __name__ == "__main__":
    unittest.main()
