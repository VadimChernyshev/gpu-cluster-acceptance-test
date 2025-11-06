# DistilBERT SST-2 Multi-Node Training Test

This repository packages a fixed-configuration DistilBERT fine-tuning job on the GLUE SST-2 sentiment task. The workload is intentionally opinionated: it always trains for one epoch with predefined hyperparameters so platform engineers can validate cluster health quickly without juggling command-line flags.

## Repository Layout

- `Dockerfile` – container definition based on `nvcr.io/nvidia/pytorch:24.07-py3`
- `train_ddp.py` – distributed training script (runs as-is, no CLI parameters)
- `test_smoke.py` – quick functional smoke test used by CI and local checks
- `requirements.txt` – additional Python dependencies (Transformers + Datasets)
- `.github/workflows/build_and_test.yml` – GitHub Actions workflow that builds the image and runs the smoke test

## Prerequisites

- Docker with NVIDIA Container Toolkit on GPU hosts
- Internet access the first time you download DistilBERT weights and GLUE SST-2 splits
- Optional: Python ≥3.10 if executing the scripts directly outside Docker

## Quick Start

### Build the container

```bash
docker build -t gpu-cluster-training-test .
```

### Run on a GPU node

```bash
docker run --rm --gpus all gpu-cluster-training-test
```

### Submit on SLURM

Example `sbatch` script for a four-node run, one GPU per node:

```bash
#!/bin/bash
#SBATCH --job-name=distilbert-acceptance
#SBATCH -N 4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=O-%x_%j.log
#SBATCH --error=E-%x_%j.log

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=12345
IMAGE="ghcr.io/vadimchernyshev/distilbert-sst2-test:latest"

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "WORLD_SIZE=$SLURM_NTASKS"

echo "[INFO] Pulling image on all nodes..."
srun -N $SLURM_NNODES docker pull $IMAGE

srun docker run --rm --gpus all \
  -e MASTER_ADDR=$MASTER_ADDR \
  -e MASTER_PORT=$MASTER_PORT \
  -e WORLD_SIZE=$SLURM_NTASKS \
  -e RANK=$SLURM_PROCID \
  -e LOCAL_RANK=$SLURM_LOCALID \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HF_HOME=/cache/hf \
  -e HF_DATASETS_CACHE=/cache/hf/datasets \
  -e TRANSFORMERS_CACHE=/cache/hf/transformers \
  -v /mnt/fast-cache:/cache \
  $IMAGE
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
  gpu-cluster-training-test \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT:-29500} \
  /workspace/train_ddp.py
```

### Smoke tests / short runs

For quick validation (e.g., CI or pre-flight checks) execute the bundled script:

```bash
python test_smoke.py
```

Or through the container:

```bash
docker run --rm \
  --entrypoint python \
  gpu-cluster-training-test \
  test_smoke.py
```

The script downloads the DistilBERT model and tokenizer, performs a forward pass, executes a tiny backward pass, and exits successfully if everything works.

## Continuous Integration

The GitHub Actions workflow at `.github/workflows/build_and_test.yml`:

1. Builds the container image.
2. Runs the CPU-mode smoke test inside the container (`test_smoke.py`).


Extend `train_ddp.py` or patch the module constants if you need different hyperparameters, evaluation loops, or custom metrics for your validation workflows.
