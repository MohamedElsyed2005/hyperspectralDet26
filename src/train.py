"""Train the 16-band hyperspectral detector and track mAP@[0.5:0.95]."""

import argparse
import random
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from . import config
from .dataset import HyperspectralDetDataset, collate_fn
from .model import build_model
from .utils.metrics import evaluate_coco_map


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


from torch.cuda.amp import GradScaler, autocast

def train_one_epoch(model, optimizer, loader, device, epoch, scaler, accum_steps=4):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    
    for i, (images, targets) in enumerate(pbar):
        # Skip batch if any target has 0 boxes
        if any(t["boxes"].numel() == 0 for t in targets):
            continue

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast():
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps
        pbar.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

    return running_loss / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = HyperspectralDetDataset()
    print(f"Total training samples found: {len(full_dataset)}")
    
    n_val = max(1, int(len(full_dataset) * config.VAL_SPLIT))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(config.SEED)
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn
    )

    model = build_model().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed weights from {args.resume}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA
    )

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_map = 0.0
    scaler = GradScaler()

    for epoch in range(1, args.epochs + 1):
        # inside the epoch loop:
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler, accum_steps=4)
        scheduler.step()

        # Evaluate real competition mAP
        map_50_95, map_50 = evaluate_coco_map(model, val_loader, device)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Val mAP@[0.5:0.95] = {map_50_95:.4f} | Val mAP@0.5 = {map_50:.4f}")

        torch.save(model.state_dict(), config.CHECKPOINT_DIR / "last.pt")
        if map_50_95 > best_map:
            best_map = map_50_95
            torch.save(model.state_dict(), config.CHECKPOINT_DIR / "best.pt")
            print(f"--> Saved new BEST checkpoint (mAP = {best_map:.4f})")


if __name__ == "__main__":
    main()