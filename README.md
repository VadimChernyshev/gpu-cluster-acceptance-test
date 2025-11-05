# DistilBERT SST-2 Multi-Node Training Test

This repository packages a fixed-configuration DistilBERT fine-tuning job on the GLUE SST-2 sentiment task. The workload is intentionally opinionated: it always trains for one epoch with predefined hyperparameters so platform engineers can validate cluster health quickly without juggling command-line flags.

## Repository Layout

- `Dockerfile` – container definition based on `nvcr.io/nvidia/pytorch:24.07-py3`
- `train_ddp.py` – distributed training script (runs as-is, no CLI parameters)
- `requirements.txt` – additional Python dependencies (Transformers + Datasets)
- `ci/build_and_test.yaml` – reference pipeline for non-GitHub CI systems
- `.github/workflows/build.yml` – GitHub Actions pipeline for build/test/push

## Prerequisites

- Docker with NVIDIA Container Toolkit on GPU hosts
- Internet access the first time you download DistilBERT weights and GLUE SST-2 splits
- Optional: Python ≥3.10 if executing the script directly outside Docker

## Quick Start

### Build the container

```bash
docker build -t distilbert-sst2-test .
```

### Run on a GPU node

```bash
docker run --rm --gpus all distilbert-sst2-test
```

### Multi-node launch

Use `torchrun` (or `python -m torch.distributed.run`). Example on two nodes with four GPUs each:

```bash
torchrun \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT:-29500} \
  train_ddp.py
```

When running inside the container, override the entrypoint:

```bash
docker run --rm --gpus all \
  --entrypoint torchrun \
  distilbert-sst2-test \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT:-29500} \
  /workspace/train_ddp.py
```

### Smoke tests / short runs

For quick validation (e.g., CI), import the module and tweak the top-level constants before calling `main()`:

```bash
python - <<'PY'
import train_ddp
from datasets import load_dataset as hf_load_dataset

def sliced_dataset(*args, **kwargs):
    if kwargs.get("split") == "train":
        kwargs["split"] = "train[:2%]"
    return hf_load_dataset(*args, **kwargs)

train_ddp.load_dataset = sliced_dataset
train_ddp.BATCH_SIZE = 8
train_ddp.NUM_WORKERS = 0
train_ddp.main()
PY
```

Adjust the split fraction as needed for larger or smaller smoke tests.

## Continuous Integration

Both CI definitions follow the same flow:

1. Build the container image.
2. Run a CPU-mode smoke test that monkey-patches the dataset loader to use a small subset.
3. On pushes to `main`, push the resulting image to `ghcr.io/<owner>/<repo>`.

Ensure the GitHub workflow token has `packages: write` scope so the push to GHCR succeeds.

## Local Development (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121  # or the CPU wheel
pip install -r requirements.txt
python train_ddp.py
```

Extend `train_ddp.py` or monkey-patch the constants (as in the smoke test snippet) if you need different hyperparameters, evaluation loops, or custom metrics for your validation workflows.
