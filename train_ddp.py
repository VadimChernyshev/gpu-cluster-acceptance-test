"""Minimal multi-node DDP fine-tuning of DistilBERT on GLUE SST-2."""

import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

# Hardcoded config for acceptance test
EPOCHS = 1
BATCH_SIZE = 32
MAX_LENGTH = 128
MODEL_NAME = "distilbert-base-uncased"
LEARNING_RATE = 5e-5
SEED = 17
LOG_INTERVAL = 25


def init_dist():
    cuda = torch.cuda.is_available()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        backend = "nccl" if cuda else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if cuda:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda", 0) if cuda else torch.device("cpu")

    return rank, world_size, local_rank, device, distributed


def build_dataloader(rank, world_size, device, distributed):
    raw_train = load_dataset("glue", "sst2", split="train")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    ds = raw_train.map(
        tokenize,
        batched=True,
        remove_columns=[c for c in raw_train.column_names if c != "label"],
        desc="Tokenizing train",
    )
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    if distributed:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=not distributed,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    return loader


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True

    rank, world_size, local_rank, device, distributed = init_dist()

    mode = "DDP" if distributed else "single-process"
    device_kind = "GPU" if device.type == "cuda" else "CPU"
    print(f"TRAINING MODE={mode.upper()} WORLD_SIZE={world_size} DEVICE={device} MODEL={MODEL_NAME}")
    print(f"RUNNING ON {device_kind}!")
    print(f"rank={rank}")

    train_loader = build_dataloader(rank, world_size, device, distributed)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    start = time.time()
    for epoch in range(EPOCHS):
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        total_loss = torch.zeros(1, device=device)
        total_examples = torch.zeros(1, device=device)

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = batch["labels"]

            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            total_loss += loss.detach() * bs
            total_examples += bs

            if rank == 0 and ((step + 1) % LOG_INTERVAL == 0 or step == 0):
                avg_loss = (total_loss / total_examples).item()
                print(f"epoch={epoch} step={step + 1} avg_loss={avg_loss:.4f}")

        if distributed:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_examples, op=dist.ReduceOp.SUM)

        if rank == 0 and total_examples.item() > 0:
            epoch_loss = (total_loss / total_examples).item()
            print(f"epoch={epoch} train_loss={epoch_loss:.4f}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print(f"Finished in {time.time() - start:.2f}s.")


if __name__ == "__main__":
    main()
