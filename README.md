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

## Quick Start

### Build the container

```bash
docker build -t gpu-cluster-training-test .
```

### Run on a GPU node

```bash
docker run --rm --gpus all gpu-cluster-training-test
```

### Run on a single node with multiple GPUs

Use `torchrun` to spawn one process per GPU. Inside the container:

```bash
docker run --rm --gpus all \
  --entrypoint torchrun \
  gpu-cluster-training-test \
  --nproc_per_node=4 \
  /workspace/train_ddp.py
```

Adjust `--nproc_per_node` to match how many GPUs you want to use on the node.

### Submit on SLURM

Create two helper scripts for launching on a SLURM-managed cluster. The first submits the job, pulls the container on all nodes, and fans out the per-task script; the second runs inside each task, wires up the environment variables, and starts the container.

`sbatch.sh` (4 nodes, 1 GPU per node):

```bash
#!/bin/bash
#SBATCH --job-name=distilbert-acceptance
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH -D .
#SBATCH --output=logs/O-%x_%j.log
#SBATCH --error=logs/E-%x_%j.log

set -euo pipefail

IMAGE="ghcr.io/vadimchernyshev/distilbert-sst2-test:latest"

echo "[INFO] Pulling image on all nodes..."
srun -N "$SLURM_NNODES" docker pull "$IMAGE"

echo "[INFO] Launching containers..."
srun --nodes="$SLURM_NNODES" bash ./srun.sh
```

`srun.sh`:

```bash
#!/bin/bash
set -euo pipefail

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=12347
IMAGE="ghcr.io/vadimchernyshev/distilbert-sst2-test:latest"

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "WORLD_SIZE=$SLURM_NTASKS"
echo "Starting on node: $(hostname) | rank=$SLURM_PROCID local_rank=$SLURM_LOCALID world_size=$SLURM_NTASKS"

docker run --rm \
  --network=host \
  -e MASTER_ADDR="$MASTER_ADDR" \
  -e MASTER_PORT="$MASTER_PORT" \
  -e WORLD_SIZE="$SLURM_NTASKS" \
  -e RANK="$SLURM_PROCID" \
  -e LOCAL_RANK="$SLURM_LOCALID" \
  -e SLURM_PROCID="$SLURM_PROCID" \
  -e SLURM_LOCALID="$SLURM_LOCALID" \
  -e SLURM_NTASKS="$SLURM_NTASKS" \
  "$IMAGE" 2>&1 | tee "logs/training.log"
```

Before submitting, make the scripts executable and prepare the log directory:

```bash
mkdir -p logs
chmod +x sbatch.sh srun.sh
```

Submit with:

```bash
sbatch sbatch.sh
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

Extend `train_ddp.py` or patch the module constants if you need different hyperparameters, evaluation loops, or custom metrics for your validation workflows.
