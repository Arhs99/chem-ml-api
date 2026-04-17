# chem-ml-api

FastAPI inference service that exposes chemprop 2.2.3 ADMET models over HTTP.
CPU-only, multi-assay via a JSON registry, persistent `ProcessPoolExecutor` for per-request SMILES sharding.

> Early development — full docs arrive with the first release.

## Install

```bash
conda env create -f environment.yml
conda activate chem-ml-api
pip install -e .
```

## Configure

```bash
cp config.example.json config.json
# edit config.json so each assay entry points at a chemprop model directory
```

## Run

```bash
export CHEMML_CONFIG=$PWD/config.json
export CHEMML_API_KEY=$(chemmlapi-genkey)
chemmlapi
```

## Environment variables

| Variable            | Default            | Purpose                                                                                          |
|---------------------|--------------------|--------------------------------------------------------------------------------------------------|
| `CHEMML_HOST`       | `0.0.0.0`          | uvicorn bind address                                                                             |
| `CHEMML_PORT`       | `8000`             | uvicorn port                                                                                     |
| `CHEMML_CONFIG`     | `config.json`      | Path to the assay registry JSON                                                                  |
| `CHEMML_PROCESSES`  | `os.cpu_count()`   | Size of the in-process `ProcessPoolExecutor`. Independent from uvicorn workers (hardcoded to 1). |
| `CHEMML_API_KEY`    | *(unset)*          | When set, requests must carry `X-API-Key`. Unset disables auth.                                  |

## License

GPL-3.0-or-later. See `LICENSE`.
