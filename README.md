# chem-ml-api

FastAPI inference service that exposes chemprop 2.2.3 ADMET models over HTTP.
CPU-only, multi-assay via a JSON registry, persistent `ProcessPoolExecutor` for per-request SMILES sharding.

## Install

```bash
conda env create -f environment.yml
conda activate chem-ml-api
pip install -e .
```

## Configure

```bash
cp config.example.json config.json
```

Each entry in `config.json` describes an assay backed by a chemprop checkpoint:

```json
{
  "assays": [
    {
      "name": "logD",
      "model_dir": "./models/logd",
      "ensemble_glob": "model_*",
      "batch_size": 64
    }
  ]
}
```

- `name` — any string; used in `POST /predict` as `{"assay": <name>}`.
- `model_dir` — directory containing the chemprop checkpoint. Resolved **relative to `config.json`'s directory**. Two layouts are supported:
  - a single `best.pt` at the root of `model_dir`;
  - an ensemble where `<ensemble_glob>/best.pt` matches multiple subdirectories (e.g. `model_0/best.pt`, `model_1/best.pt`, …).
  - If no `best.pt` is found, `checkpoints/*.ckpt` is tried as a fallback.
- `ensemble_glob` — defaults to `"model_*"`.
- `batch_size` — chemprop DataLoader batch size per worker.

## Run

```bash
export CHEMML_CONFIG=$PWD/config.json
export CHEMML_API_KEY=$(chemmlapi-genkey)   # optional; omit to disable auth entirely
chemmlapi                                    # uvicorn on 0.0.0.0:8000 (single worker)
```

## Environment variables

| Variable            | Default           | Purpose                                                                                          |
|---------------------|-------------------|--------------------------------------------------------------------------------------------------|
| `CHEMML_HOST`       | `0.0.0.0`         | uvicorn bind address                                                                             |
| `CHEMML_PORT`       | `8000`            | uvicorn port                                                                                     |
| `CHEMML_CONFIG`     | `config.json`     | Path to the assay registry JSON                                                                  |
| `CHEMML_PROCESSES`  | `1`               | Size of the in-process `ProcessPoolExecutor`. Independent from uvicorn workers (hardcoded to 1). |
| `CHEMML_API_KEY`    | *(unset)*         | When set, requests must carry `X-API-Key`. Unset disables auth.                                  |

## HTTP API

### `GET /health`

Always public (no `X-API-Key` required).

```
$ curl -s http://localhost:8000/health
{"status":"ok"}
```

### `GET /assays`

Lists registered assay names.

```
$ curl -s -H "X-API-Key: $CHEMML_API_KEY" http://localhost:8000/assays
{"assays":["logD"]}
```

### `POST /predict`

Request body:

```json
{
  "assay": "logD",
  "smiles": ["CCO", "c1ccccc1"],
  "include_std": false
}
```

- `smiles` — 1 to 10000 strings. Each is canonicalized through RDKit; invalid strings are counted and excluded (422 if all are invalid).
- `include_std` — when `true`, populates the `std` field per result. Returns `null` on a single non-MVE regressor (no native uncertainty).

Response body:

```json
{
  "results": [
    {"smiles": "CCO",      "prediction": -1.140, "std": null},
    {"smiles": "c1ccccc1", "prediction":  3.553, "std": null}
  ],
  "metadata": {
    "elapsed_seconds": 0.021,
    "molecules_processed": 2,
    "molecules_invalid": 0,
    "processes": 2
  }
}
```

## Uncertainty semantics

- **Single non-MVE model** → `std = null`. No native uncertainty.
- **Ensemble of non-MVE models** → `std` is the sample std across ensemble member predictions.
- **Single MVE model** → `std = sqrt(variance)` from the MVE head.
- **Ensemble of MVE models** → total variance via the law of total variance: `mean(σ²) + var(μ)`.

Predictions are returned as raw model outputs — no post-hoc inverse transform is applied. If the model was trained on a transformed target (e.g. `log10(y)`), the consumer is responsible for inverting that on their side.

## Testing

```bash
python -m unittest discover -s tests
```

Unit tests always run. Integration tests (via `tests/_markers.py`) skip automatically unless a chemprop checkpoint is present at `tests/fixtures/logd/best.pt` or `tests/fixtures/logd/model_*/best.pt`.

## License

GPL-3.0-or-later. See `LICENSE`.
