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

import hmac
import os
import secrets
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from rdkit import Chem

from chemmlapi.configs.registry import load_config
from chemmlapi.interfaces.parallel import _prime, _worker_init, chunk_predict

_pool: Optional[ProcessPoolExecutor] = None
_assays: list[str] = []
_n_processes: int = 1


class PredictRequest(BaseModel):
    assay: str = Field(..., min_length=1)
    smiles: list[str] = Field(..., min_length=1, max_length=10000)
    include_std: bool = False


class MoleculePrediction(BaseModel):
    smiles: str
    prediction: float
    std: Optional[float] = None


class PredictResponse(BaseModel):
    results: list[MoleculePrediction]
    metadata: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool, _assays, _n_processes
    config_path = str(Path(os.environ.get("CHEMML_CONFIG", "config.json")).resolve())
    _n_processes = int(os.environ.get("CHEMML_PROCESSES", 1))
    _assays = [e.name for e in load_config(config_path).assays]
    _pool = ProcessPoolExecutor(
        max_workers=_n_processes,
        initializer=_worker_init,
        initargs=(config_path,),
    )
    list(_pool.map(_prime, range(_n_processes)))
    try:
        yield
    finally:
        pool, _pool = _pool, None
        pool.shutdown(wait=True, cancel_futures=True)


app = FastAPI(
    title="chem-ml-api",
    description="FastAPI inference service for chemprop ADMET models",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def check_api_key(request: Request, call_next):
    api_key = os.environ.get("CHEMML_API_KEY")
    if api_key and request.url.path not in ("/health", "/docs", "/openapi.json"):
        provided = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(provided, api_key):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )
    return await call_next(request)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/assays")
def assays():
    return {"assays": list(_assays)}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    if _pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    if req.assay not in _assays:
        raise HTTPException(status_code=404, detail=f"Unknown assay: {req.assay}")

    valid: list[str] = []
    invalid: list[str] = []
    for smi in req.smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid.append(smi)
        else:
            valid.append(Chem.MolToSmiles(mol))

    if not valid:
        raise HTTPException(
            status_code=422,
            detail=f"No valid SMILES provided. Invalid: {invalid}",
        )

    t0 = time.monotonic()
    df = chunk_predict(_pool, req.assay, valid, _n_processes)
    elapsed = time.monotonic() - t0

    has_std = "std" in df.columns
    results = [
        MoleculePrediction(
            smiles=str(row["smiles"]),
            prediction=float(row["prediction"]),
            std=(float(row["std"]) if has_std and req.include_std else None),
        )
        for _, row in df.iterrows()
    ]

    return PredictResponse(
        results=results,
        metadata={
            "elapsed_seconds": round(elapsed, 3),
            "molecules_processed": len(valid),
            "molecules_invalid": len(invalid),
            "processes": _n_processes,
        },
    )


def run():
    import uvicorn

    host = os.environ.get("CHEMML_HOST", "0.0.0.0")
    port = int(os.environ.get("CHEMML_PORT", "8000"))
    uvicorn.run(
        "chemmlapi.interfaces.api:app",
        host=host,
        port=port,
        workers=1,
    )


def generate_key():
    print(secrets.token_urlsafe(32))


if __name__ == "__main__":
    run()
